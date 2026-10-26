"""Deterministically adapt the verified collector; never imports a model or runs inference."""
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1] / 'phen_intermittent_full_observations_20261026_v1' / 'collector_v5.py'
BASE_SHA = 'a65e6dd949c551921163a40d38a47684108147973b4fac245e0abe1ba90ca60d'
original = BASE.read_text()
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
text = original


def replace(old, new):
    global text
    assert text.count(old) == 1, (old[:100], text.count(old))
    text = text.replace(old, new)


def block(start, end, new):
    global text
    left, right = text.index(start), text.index(end)
    text = text[:left] + new.rstrip() + '\n\n\n' + text[right:]


replace('Single-forward, native PHEN Train/Validation observation collector.',
        'Frozen PHEN head-input extraction for six missing Train/Validation cohorts.')
replace("SCHEMA = 'phen_full_native_observations_v1'", "SCHEMA = 'phen_time_head_hidden_extraction_v1'")
block("SCALARS = (", "EXPECTED =", """REFERENCE_SCALARS = ('pred_quantity', 'time_nll')
REFERENCE_FIELDS = (*IDENTITY, 'history_length', 'target_gap', 'pred_quantity', 'time_nll', 'time_location_float64', 'time_sigma_float64')
TOLERANCES = {'pred_quantity': {'atol': 1e-5, 'rtol': 1e-5}, 'time_nll': {'atol': 1e-4, 'rtol': 1e-4}}
BASE_COLLECTOR_SHA256 = 'a65e6dd949c551921163a40d38a47684108147973b4fac245e0abe1ba90ca60d'""")
block('def verify_numeric_gate_revision(', 'def state_sha(', '''def qualification_binding(spec, job):
    fields = ('schema', 'runner_sha256', 'science_spec', 'execution_device', 'expected_device_name',
              'expected_device_uuid', 'batch_size', 'chunk_rows', 'num_threads', 'population_counts',
              'input_order', 'comparison_tolerances', 'reference_samples', 'split_manifest_record', 'working_dir')
    return json_sha({'job': job, 'configuration': {key: spec.get(key) for key in fields}})


def verify_qualification(spec, job):
    if spec['mode'] == 'qualification':
        return None
    gate = spec['qualification_gate'][job['id']]
    done = json.loads(verify_ref(gate['job_complete']).read_text())
    exit_record = json.loads(verify_ref(gate['actual_exit']).read_text())
    require(exit_record['actual_child_exit'] == 0 and exit_record['job'] == job['id'], 'Smoke actual exit failed')
    require(exit_record['collector_sha256'] == spec['runner_sha256'], 'Smoke used another extractor')
    require(done['status'] == 'qualification_complete'
            and done['qualification_binding'] == qualification_binding(spec, job), 'Smoke scientific/execution binding differs')
    require(done['input_and_model_immutability_verified'] and done['context_state_unchanged'], 'Smoke model changed')
    require(set(done['splits']) == set(job['splits']), 'Smoke cohort coverage differs')
    for split in job['splits']:
        require(done['splits'][split]['rows'] == done['splits'][split]['native_reference_rows_checked'] == 2048,
                'Smoke must cover exactly 2048 parity-checked events per cohort')
    return gate
''')
block('def read_reference(', 'def fixed_order(', '''def read_reference(ref):
    """Read SHA-bound compact saved outputs; every index is its population target index."""
    import numpy as np
    path = verify_ref(ref)
    with np.load(path, allow_pickle=False) as stored:
        require(set(stored.files) == set(REFERENCE_FIELDS), 'Reference fields differ')
        arrays = {key: stored[key].copy() for key in REFERENCE_FIELDS}
    n = len(arrays['target_index'])
    require(all(len(a) == n for a in arrays.values()), 'Reference lengths differ')
    require(np.array_equal(arrays['target_index'], np.arange(n)), 'Reference target indices not exhaustive/sorted')
    for key, array in arrays.items():
        if key != 'entity_id': require(np.isfinite(array).all(), 'Nonfinite saved reference')
    return arrays


def reference_rows(arrays, indices):
    return [{**{key: arrays[key][int(index)].item() for key in REFERENCE_FIELDS}, 'batch_index': i // 128}
            for i, index in enumerate(indices)]
''')
block('class NativeObserver(', 'def export_batch(', '''class NativeObserver(AbstractContextManager):
    """Observe only the native head input and time outputs; original returns stay intact."""
    def __init__(self, model):
        self.model, self.values, self.handles = model, {}, []
        self.calls = {'encode': 0, 'time': 0, 'quantity': 0, 'head': 0}

    def __enter__(self):
        import torch
        model = self.model
        for name in ('encode_task_states', '_lognormal_time_terms', 'quantity_outputs'):
            require(name not in model.__dict__, 'Existing model override: ' + name)
        original_encode, original_time, original_quantity = model.encode_task_states, model._lognormal_time_terms, model.quantity_outputs

        def encode(dts, quantities, mask, *, memory_write_mask=None):
            self.calls['encode'] += 1
            lengths = mask.sum(1).long()
            require(bool((lengths >= 2).all()) and mask.shape[1] == 256, 'Invalid native input shape')
            expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
            require(torch.equal(mask, expected), 'Native valid positions must be right-padded prefixes')
            ids, target = torch.arange(len(lengths), device=mask.device), lengths - 1
            require(bool((quantities[ids, target] == 0).all()), 'Withheld target quantity leaked')
            write = mask.clone(); write[ids, target] = False
            require(memory_write_mask is not None and torch.equal(memory_write_mask, write), 'Target write mask changed')
            self.values['target_gap'] = dts[ids, target].detach().clone()
            self.values['history_length'] = lengths - 1
            self.values['valid_mask_sha256'] = hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()
            self.values['observed_mask_sha256'] = hashlib.sha256(memory_write_mask.detach().cpu().numpy().tobytes()).hexdigest()
            return original_encode(dts, quantities, mask, memory_write_mask=memory_write_mask)

        def time_terms(hidden, target):
            self.calls['time'] += 1
            result = original_time(hidden, target)
            mu, sigma, _ = result
            require(mu.dtype == sigma.dtype == torch.float64, 'Native time-head arithmetic changed')
            require(torch.equal(target, self.values['target_gap']), 'Time target differs')
            self.values['time_hidden'] = hidden.detach().clone()
            self.values['time_location_float64'] = mu.detach().clone()
            self.values['time_sigma_float64'] = sigma.detach().clone()
            return result

        def quantity(hidden, target):
            self.calls['quantity'] += 1
            result = original_quantity(hidden, target)
            self.values['quantity_hidden'] = hidden.detach().clone()
            return result

        def head(module, args, result):
            self.calls['head'] += 1
            self.values['head_hidden'] = args[0].detach().clone()

        model.encode_task_states, model._lognormal_time_terms, model.quantity_outputs = encode, time_terms, quantity
        self.handles = [model.quantity_head.register_forward_hook(head)]
        return self

    def __exit__(self, *_):
        for handle in self.handles: handle.remove()
        for name in ('encode_task_states', '_lognormal_time_terms', 'quantity_outputs'): delattr(self.model, name)

    def validate(self, outputs):
        import torch
        value = self.values
        require(all(n == 1 for n in self.calls.values()), 'Expected one native call per batch')
        require(torch.equal(value['head_hidden'], value['quantity_hidden']) and torch.equal(value['head_hidden'], value['time_hidden']), 'Heads used different inputs')
        require(value['head_hidden'].dtype == torch.float32 and value['head_hidden'].ndim == 2 and value['head_hidden'].shape[1] == 64, 'Expected float32 head input width64')
        require(torch.equal(value['history_length'], outputs['history_length']), 'History mask count mismatch')
        require(bool((value['target_gap'] == value['target_gap'].round()).all()) and bool((value['target_gap'] >= 1).all()), 'Invalid recorded target gap')
        for key, tensor in value.items():
            if isinstance(tensor, torch.Tensor): require(bool(torch.isfinite(tensor).all()), 'Nonfinite observed ' + key)
        return value
''')
block('def export_batch(', 'class ChunkWriter:', '''def export_batch(outputs, observed, identities):
    import numpy as np
    tensors = {'true_quantity': outputs['true_qty'], 'pred_quantity': outputs['pred_qty'],
               'time_nll': outputs['time_loss'], 'history_length': outputs['history_length']}
    for key in ('target_gap', 'time_location_float64', 'time_sigma_float64'):
        tensors[key] = observed[key]
    arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    require(all(len(a) == len(identities) and np.isfinite(a).all() for a in arrays.values()), 'Invalid compact native output')
    require(np.array_equal(arrays['true_quantity'], np.asarray([r['raw_quantity'] for r in identities])), 'Target quantity differs')
    for key in IDENTITY: arrays[key] = np.asarray([row[key] for row in identities])
    return arrays, observed['head_hidden'].detach().cpu().numpy()


def compare_reference(arrays, hidden, references, tolerances, output, label):
    """Exact identities; only unchanged quantity/NLL gates; mu/sigma diagnostics only."""
    import numpy as np
    require(tolerances == TOLERANCES, 'Parity tolerances changed')
    require(len(references) == len(arrays['target_index']) == len(hidden), 'Parity rows differ')
    maximum, failures = {}, []
    for i, ref in enumerate(references):
        for key in (*IDENTITY, 'history_length', 'target_gap'):
            require(arrays[key][i] == ref[key], 'Reference identity mismatch: ' + key)
        for key in (*REFERENCE_SCALARS, 'time_location_float64', 'time_sigma_float64'):
            actual, expected = float(arrays[key][i]), float(ref[key])
            require(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite parity value')
            error = abs(actual - expected)
            maximum[key] = max(maximum.get(key, 0.), error)
            if key in REFERENCE_SCALARS:
                allowed = tolerances[key]['atol'] + tolerances[key]['rtol'] * abs(expected)
                if error > allowed:
                    failures.append({'target_index': int(ref['target_index']), 'field': key, 'actual': actual,
                                     'expected': expected, 'absolute_error': error, 'allowed_error': allowed})
    if failures:
        write_json(output / ('PARITY_FAILURE_' + label + '.json'), {'status': 'failed_preserved',
                   'tolerances': tolerances, 'max_absolute_errors': maximum, 'failures': failures})
        raise ValueError('Native reference parity failed: ' + label)
    return maximum
''')
replace("        columns = {key: value for key, value in arrays.items() if key not in ('prototype_indices', 'topk_similarity')}\n        for i in range(4):\n            columns[f'prototype_index_{i}'] = arrays['prototype_indices'][:, i].astype(np.int16)\n            columns[f'prototype_cosine_{i}'] = arrays['topk_similarity'][:, i].astype(np.float32)",
        '        columns = dict(arrays)')
replace("('log_qty_mse_sum', arrays['log_qty_mse']), ('time_nll_sum', arrays['time_nll'])", "('time_nll_sum', arrays['time_nll']),")
replace("    runner_sha = sha(__file__)\n    source =", "    runner_sha = sha(__file__)\n    qualification_gate = verify_qualification(spec, job)\n    verify_ref(spec['science_spec'])\n    source =")
replace("              'numeric_gate_revision': verify_numeric_gate_revision(spec),\n              'log_mse_gate_revision': verify_log_mse_gate_revision(spec),", "              'qualification_gate': qualification_gate, 'qualification_binding': qualification_binding(spec, job),\n              'science_spec': spec['science_spec'], 'base_collector_sha256': BASE_COLLECTOR_SHA256,")
replace("        ref = read_reference(job['reference_results'][split])\n        sample = json.loads(verify_ref(spec['reference_samples'][split]).read_text())", "        all_ref = read_reference(job['reference_results'][split])\n        sample = json.loads(verify_ref(spec['reference_samples'][split]).read_text())\n        ref = reference_rows(all_ref, [r['target_index'] for r in sample['targets']])")
replace("        writer = ChunkWriter(output, split, spec['chunk_rows'], save_hidden=(kind == 'best' and split == 'train'))", "        writer = ChunkWriter(output, split, spec['chunk_rows'], save_hidden=True)")
replace("        require(len(truth) == spec['population_counts'][split] == POPULATIONS[split], 'Population count mismatch')", "        require(len(truth) == spec['population_counts'][split] == POPULATIONS[split], 'Population count mismatch')\n        require(len(all_ref['target_index']) == len(truth), 'Saved reference population count differs')")
replace("'log_qty_mse_sum': 0., ", '')
replace("                if count < 2048:\n                    require(count + size <= 2048 and all(r['batch_index'] == bi for r in ref[count:count + size]), 'Reference batch structure changed')\n                    parity = compare_reference(arrays, hidden, ref[count:count + size],\n                                               spec['comparison_tolerances'], output, f'{split}_{bi:05d}')\n                    for key, value in parity.items(): maximum[key] = max(maximum.get(key, 0.), value)\n                    parity_count += size", "                if count < 2048:\n                    require(count + size <= 2048 and all(r['batch_index'] == bi for r in ref[count:count + size]), 'Reference batch structure changed')\n                parity = compare_reference(arrays, hidden, reference_rows(all_ref, arrays['target_index']),\n                                           spec['comparison_tolerances'], output, f'{split}_{bi:05d}')\n                for key, value in parity.items(): maximum[key] = max(maximum.get(key, 0.), value)\n                parity_count += size")
replace("parity_count == min(2048, len(indices))", "parity_count == len(indices)")
replace("                   'log_qty_mse': accumulator['log_qty_mse_sum'] / n, 'time_nll':", "                   'time_nll':")
replace("    verify_numeric_gate_revision(spec)\n    verify_log_mse_gate_revision(spec)\n    result.update", "    verify_ref(spec['science_spec'])\n    if qualification_gate:\n        verify_ref(qualification_gate['job_complete']); verify_ref(qualification_gate['actual_exit'])\n    result.update")
block('def validate_spec(', 'def main(', '''def validate_spec(spec):
    require(spec['schema'] == SCHEMA and spec['runner_sha256'] == sha(__file__), 'Extractor/spec binding differs')
    require(spec['runner_filename'] == Path(__file__).name == 'collector_hidden.py', 'Unexpected extractor name')
    require(Path(spec['working_dir']).is_absolute() and isinstance(spec['split_manifest_record'], dict), 'Isolated input layout required')
    require(spec['mode'] in ('qualification', 'full'), 'Unadmitted scope')
    require(spec['batch_size'] == 128 and spec['chunk_rows'] == 4096 and spec['qualification_batches'] == 16, 'Batch or 2048-event smoke setting changed')
    require(spec['population_counts'] == POPULATIONS, 'Population counts changed')
    require(spec['comparison_tolerances'] == TOLERANCES, 'Quantity/timeNLL tolerances changed')
    require(spec['input_order'] == 'reference_2048_then_remaining_target_index', 'Input order changed')
    require(spec['execution_device'] in ('cpu', 'cuda:0') and 1 <= spec['num_threads'] <= 8, 'Unadmitted device/threads')
    require(1 <= spec['child_timeout_seconds'] <= 14400, 'Missing bounded child timeout')
    verify_ref(spec['science_spec'])
    if spec['execution_device'].startswith('cuda'):
        require(bool(spec['expected_device_name']), 'GPU identity missing')
        canonical_gpu_uuid(spec['expected_device_uuid'])
    require(bool(spec['jobs']) and len({j['id'] for j in spec['jobs']}) == len(spec['jobs']), 'Duplicate/empty jobs')
    for job in spec['jobs']:
        require(job['dataset_id'] == 'intermittent_frozen_5000' and job['arm'] == 'full' and EXPECTED.get((job['seed'], job['checkpoint_kind'])) == job['expected_epoch'], 'Unadmitted checkpoint')
        require(Path(job['id']).name == job['id'] and job['id'] not in ('.', '..'), 'Unsafe job ID')
        require(job['splits'] == (['validation'] if job['checkpoint_kind'] == 'best' else ['train', 'validation']), 'Only six missing hidden cohorts admitted')
        if spec['mode'] == 'full': require(job['id'] in spec.get('qualification_gate', {}), 'Full requires bound successful 2048-event smoke')
''')
replace("            child = subprocess.run(command, env=environment, stdout=out, stderr=err, check=False)", "            child = subprocess.Popen(command, env=environment, stdout=out, stderr=err)\n            timed_out = False\n            try:\n                child.wait(timeout=spec['child_timeout_seconds'])\n            except subprocess.TimeoutExpired:\n                child.kill(); child.wait(); timed_out = True")
replace("'seconds': time.monotonic() - started, 'spec_sha256': args.spec_sha,", "'seconds': time.monotonic() - started, 'timed_out': timed_out, 'spec_sha256': args.spec_sha,")
replace("    spec = json.loads(spec_path.read_text()); validate_spec(spec)", "    spec = json.loads(spec_path.read_text()); validate_spec(spec)\n    require(output.is_relative_to(Path(spec['allowed_output_root']).resolve()), 'Output escaped the new attempt root')")
out = ROOT / 'collector_hidden.py'
if '--verify-existing' in sys.argv:
    assert out.read_text() == text, 'Generated adapter differs from checked file'
    print(json.dumps({'status': 'PASS', 'adapter_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
    raise SystemExit(0)
with out.open('x') as f:
    f.write(text)
diff = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile=str(BASE), tofile=str(out)))
(ROOT / 'collector_hidden.diff').write_text(diff)
(ROOT / 'ADAPTER_PROVENANCE.json').write_text(json.dumps({'base_path': str(BASE), 'base_sha256': BASE_SHA,
    'adapter_path': str(out), 'adapter_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'scope': 'copy frozen collector guards; narrow observer/output and bind full saved scalar parity; never modify prior files'}, indent=2)+'\n')
print(out)
