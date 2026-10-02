"""Release test only from complete, current, identity-bound validation evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from evaluate import SCHEMA, TRUTH_FIELDS, json_bytes, row_digests

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REQUIRED_CHECKS = {
    'target_quantity_perturbation_preserves_prediction',
    'target_quantity_perturbation_preserves_time_loss',
    'target_gap_perturbation_preserves_quantity_prediction',
    'single_vs_batch_prediction_matches', 'single_vs_batch_time_loss_matches',
    'parameters_unchanged', 'finite_outputs',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_receipt(receipt, key, phase, expected_row, contract, dataset_spec, evaluator_sha):
    """Pure binding check; a receipt cannot qualify a different source or scope."""
    require(receipt.get('status') == 'complete', 'Incomplete receipt')
    require(tuple(receipt.get(k) for k in ('dataset', 'model', 'seed')) == key,
            'Receipt condition mismatch')
    require(receipt.get('split') == 'validation', 'Only validation can qualify release')
    for field in ('registry_sha256', 'dataset_manifest_sha256'):
        require(receipt.get(field) == contract[field], f'Stale receipt {field}')
    require(receipt.get('contract_sha256') == contract['_sha256'], 'Stale receipt contract')
    require(receipt.get('runner_sha256') == evaluator_sha, 'Stale receipt evaluator')
    require(receipt.get('data_file_sha256') == contract['dataset_sha256'][key[0]],
            'Receipt dataset identity mismatch')
    for field in ('checkpoint_file_sha256', 'state_tensor_sha256', 'selected_epoch', 'evaluator_source_bundle'):
        require(receipt.get(field) == expected_row[field], f'Receipt checkpoint binding mismatch: {field}')
    count = dataset_spec['populations']['validation']['target_count']
    require(receipt.get('expected_target_count') == count, 'Receipt population mismatch')
    if 'validation_count' in expected_row:
        require(count == expected_row['validation_count'], 'Registry validation count mismatch')
    full = phase == 'validation_full'
    batches = None if full else (1 if phase == 'qualification_cpu' else 4)
    expected_rows = count if full else min(count, 128 * batches)
    require(receipt.get('prediction_rows') == expected_rows, 'Receipt prediction row count mismatch')
    require(receipt.get('max_batches') == batches, 'Receipt batch qualification scope mismatch')
    require(receipt.get('batch_size') == 128, 'Receipt batch size mismatch')
    require(receipt.get('full_population') is full, 'Receipt full population scope mismatch')
    require(receipt.get('evaluation_scope') == ('full_population' if full else 'validation_qualification'),
            'Receipt evaluation scope mismatch')
    require(receipt.get('heldout_read') is False and receipt.get('retrained') is False,
            'Qualification must neither read test nor train')
    checks = receipt.get('qualification', {})
    require(set(checks) == REQUIRED_CHECKS and all(v is True for v in checks.values()),
            'Missing or failed causal/state qualification checks')
    runtime = receipt.get('runtime', {})
    require(runtime.get('device') == ('cpu' if phase == 'qualification_cpu' else 'cuda'),
            'Receipt execution device mismatch')
    for field, limit in (('cpu_peak_rss_bytes', 'rss_limit_gib'),
                         ('cuda_peak_reserved_bytes', 'gpu_memory_limit_gib')):
        value = runtime.get(field)
        require(isinstance(value, (float, int)) and 0 <= value <= contract['resources'][limit] * 1024**3,
                f'Receipt exceeds resource bound: {field}')


def conditions(phase, expected_keys, contract, seal_sha, evidence):
    base = HERE / 'runs' / phase / 'attempt1'
    manifest_path, terminal_path = base / 'run_manifest.json', base / 'terminal_manifest.json'
    manifest, terminal = read(manifest_path), read(terminal_path)
    require(terminal.get('status') == 'complete', f'Incomplete phase: {phase}')
    require(terminal.get('run_manifest_sha256') == sha(manifest_path), 'Terminal manifest binding mismatch')
    require(manifest.get('phase') == phase and terminal.get('phase') == phase, 'Phase mismatch')
    require(manifest.get('code_seal_sha256') == seal_sha, 'Stale phase code seal')
    require(manifest.get('contract_sha256') == contract['_sha256'], 'Stale phase contract')
    for field in ('registry_sha256', 'dataset_manifest_sha256'):
        require(manifest.get(field) == contract[field], f'Stale phase {field}')
    specs = manifest.get('conditions', [])
    result = {}
    for item in specs:
        key = tuple(item[k] for k in ('dataset', 'model', 'seed'))
        require(key not in result, 'Duplicate qualification condition')
        path = (base / item['output_dir']).resolve()
        require(path.is_relative_to(base.resolve()), 'Condition output escapes phase directory')
        result[key] = path
    require(set(result) == expected_keys, f'Wrong condition set for {phase}')
    require(terminal.get('total') == len(expected_keys) and not terminal.get('failures')
            and not terminal.get('unstarted'), 'Incomplete terminal condition inventory')
    expected_ids = {f'{d}__{m}__seed{s}' for d, m, s in expected_keys}
    require(set(terminal.get('completed', [])) == expected_ids
            and len(terminal['completed']) == len(expected_ids), 'Terminal completed conditions mismatch')
    for path in (manifest_path, terminal_path):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    return result


def rows(path, receipt, evidence):
    frames, count = [], 0
    targets, truth = hashlib.sha256(), hashlib.sha256()
    require(receipt.get('schema') == SCHEMA, 'Receipt schema mismatch')
    require(receipt.get('truth_digest_fields') == list(TRUTH_FIELDS), 'Receipt truth digest contract mismatch')
    for part in receipt['parts']:
        p = (path / part['path']).resolve()
        require(p.is_relative_to(path), 'Prediction part escapes condition directory')
        require(sha(p) == part['sha256'], 'Prediction part SHA mismatch')
        frame = pl.read_parquet(p)
        require({k: str(v) for k, v in frame.schema.items()} == SCHEMA, 'Prediction schema mismatch')
        require(frame.height == part['rows'] and frame.height > 0, 'Prediction part row count mismatch')
        require(frame['target_index'].to_list() == list(range(count, count + frame.height)),
                'Noncontiguous prediction target indices')
        require(part['target_index_start'] == count and part['target_index_end'] == count + frame.height - 1,
                'Part target index contract mismatch')
        identities, true_digest = row_digests(frame.iter_rows(named=True))
        require(identities == part['target_identity_sha256'] and true_digest == part['truth_sha256'],
                'Part identity/truth digest mismatch')
        for row in frame.iter_rows(named=True):
            targets.update((row['target_id'] + '\n').encode('utf-8'))
            truth.update(json_bytes([row[k] for k in TRUTH_FIELDS]) + b'\n')
        frames.append(frame)
        count += frame.height
        evidence[str(p.relative_to(ROOT))] = part['sha256']
    require(count == receipt['prediction_rows'], 'Incomplete prediction rows')
    require(targets.hexdigest() == receipt['target_identity_sha256'] and truth.hexdigest() == receipt['truth_sha256'],
            'Overall identity/truth digest mismatch')
    return pl.concat(frames)


def main():
    seal_path, contract_path = HERE / 'code_seal.json', HERE / 'execution_contract.json'
    seal, contract = read(seal_path), read(contract_path)
    seal_sha, contract['_sha256'] = sha(seal_path), sha(contract_path)
    required = ['evaluate.py', 'dispatch.py', 'qualify.py', 'pipeline.py', 'execution_contract.json',
                'dataset_manifest.json', 'validation_references.json', 'local_test_receipt.json']
    for name in required:
        require(str((HERE / name).relative_to(ROOT)) in seal['files'], f'Missing sealed release file: {name}')
    for relative, expected in seal['files'].items():
        require(sha(ROOT / relative) == expected, f'Sealed file changed: {relative}')
    require(read(HERE / 'local_test_receipt.json').get('status') == 'passed', 'Local unit tests did not pass')
    require(sha(ROOT / contract['registry_path']) == contract['registry_sha256'], 'Registry changed')
    require(sha(HERE / 'dataset_manifest.json') == contract['dataset_manifest_sha256'], 'Dataset manifest changed')
    registry = read(ROOT / contract['registry_path'])
    registry_rows = {(r['dataset'], r['model'], r['seed']): r for r in registry['rows']}
    require(len(registry_rows) == len(registry['rows']) == 108, 'Unexpected registry conditions')
    all_keys = {(d, m, s) for d in contract['approval']['datasets']
                for m in contract['approval']['models'] for s in contract['approval']['seeds']}
    require(set(registry_rows) == all_keys, 'Registry differs from approved condition grid')
    sample_keys = {key for key in registry_rows if key[2] == 42}
    full_keys = {key for key in sample_keys if key[1] == 'titantpp_history_mlp'}
    require(len(sample_keys) == 36 and len(full_keys) == 4, 'Wrong qualification grid')
    datasets = {d['dataset']: d for d in read(HERE / 'dataset_manifest.json')['datasets']}
    evidence = {str((HERE / name).relative_to(ROOT)): sha(HERE / name) for name in required}
    evidence[contract['registry_path']] = contract['registry_sha256']
    cpu = conditions('qualification_cpu', sample_keys, contract, seal_sha, evidence)
    cuda = conditions('qualification_cuda', sample_keys, contract, seal_sha, evidence)
    full = conditions('validation_full', full_keys, contract, seal_sha, evidence)
    evaluator_sha = sha(HERE / 'evaluate.py')
    checks = []
    for key, path in cuda.items():
        g, c = read(path / 'receipt.json'), read(cpu[key] / 'receipt.json')
        for receipt, phase, directory in ((g, 'qualification_cuda', path), (c, 'qualification_cpu', cpu[key])):
            check_receipt(receipt, key, phase, registry_rows[key], contract, datasets[key[0]], evaluator_sha)
            evidence[str((directory / 'receipt.json').relative_to(ROOT))] = sha(directory / 'receipt.json')
        gf, cf = rows(path, g, evidence).head(c['prediction_rows']), rows(cpu[key], c, evidence)
        require(gf['target_id'].to_list() == cf['target_id'].to_list(), 'CPU/CUDA targets differ')
        errors = {}
        for col in TRUTH_FIELDS:
            require(gf[col].to_list() == cf[col].to_list(), f'CPU/CUDA truth/history differ: {key}, {col}')
        for col in ('predicted_raw_quantity', 'time_nll'):
            left, right = gf[col].to_numpy(), cf[col].to_numpy()
            require(np.isfinite(left).all() and np.isfinite(right).all(), 'Nonfinite CPU/CUDA outputs')
            np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-4, err_msg=str((key, col)))
            errors[col] = float(np.max(np.abs(left - right)))
        checks.append({'condition': list(key), 'cpu_cuda_max_absolute_differences': errors,
                       'all_causal_and_state_checks_passed': True,
                       'cuda_peak_reserved_bytes': g['runtime']['cuda_peak_reserved_bytes'],
                       'cpu_peak_rss_bytes': g['runtime']['cpu_peak_rss_bytes'],
                       'sample_inference_targets_per_second': g['inference_targets_per_second']})
    refs = read(HERE / 'validation_references.json')
    replays = []
    for key, path in full.items():
        r = read(path / 'receipt.json')
        check_receipt(r, key, 'validation_full', registry_rows[key], contract, datasets[key[0]], evaluator_sha)
        evidence[str((path / 'receipt.json').relative_to(ROOT))] = sha(path / 'receipt.json')
        f, expected = rows(path, r, evidence), refs['references'][key[0]]
        require(r['prediction_rows'] == expected['count'], 'Frozen reference population differs')
        error = f['predicted_raw_quantity'].to_numpy() - f['raw_quantity'].to_numpy()
        metrics = {'qty_mae': float(np.abs(error).mean()), 'qty_rmse': float(np.sqrt(np.square(error).mean())),
                   'time_nll': float(f['time_nll'].mean())}
        require(all(np.isfinite(v) for v in metrics.values()), 'Nonfinite full validation metrics')
        for name, value in metrics.items():
            np.testing.assert_allclose(value, expected[name], rtol=refs['relative_tolerance'],
                                       atol=refs['absolute_tolerance'], err_msg=str((key, name)))
        replays.append({'dataset': key[0], 'count': len(f), 'metrics': metrics, 'frozen_selected_replay_matched': True,
                        'elapsed_seconds': r['elapsed_seconds'], 'inference_targets_per_second': r['inference_targets_per_second']})
    result = {'status': 'passed', 'scope': 'validation_only_before_legacy_test_release',
              'code_seal_sha256': seal_sha, 'contract_sha256': contract['_sha256'],
              'evidence_hashes': evidence, 'checks': checks, 'full_validation_replays': replays,
              'cpu_cuda_tolerance': {'rtol': 1e-4, 'atol': 1e-4},
              'unit_tests_required': 'local_test_receipt.json verified in code seal',
              'full_population_loader_enumerated': True, 'output_streaming_exercised': True,
              'test_inference_before_this_gate': False}
    # An exclusive gate preserves prior evidence and cannot overwrite another attempt.
    with (HERE / 'qualification_gate.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': 'passed', 'conditions': len(checks), 'full_validation_replays': len(replays)}))


if __name__ == '__main__':
    main()
