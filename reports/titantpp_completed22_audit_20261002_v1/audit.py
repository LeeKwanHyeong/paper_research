"""Audit collected, authorized validation artifacts on CPU; never run a forward pass.

Imports use a hash-verified copy of each campaign's frozen source. Original
artifacts and the current working scientific code are never modified.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

from collect import ROOT, REPORT, DEST, canonical, sha

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/titantpp_cpu_audit_mpl')
sys.dont_write_bytecode = True
read = lambda p: json.loads(Path(p).read_text())
METRICS = ('qty_mae', 'qty_rmse', 'time_nll')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def finite(v):
    if isinstance(v, torch.Tensor):
        return bool(torch.isfinite(v).all())
    if isinstance(v, dict):
        return all(finite(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return all(finite(x) for x in v)
    return not isinstance(v, float) or math.isfinite(v)


def check_rng(cp):
    rng = cp['rng_state']
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator(device='cpu').set_state(rng['torch'])
    torch.Generator(device='cpu').set_state(cp['train_loader_generator_state'])
    require(bool(rng['cuda']) and all(x.dtype == torch.uint8 and x.numel() > 0
                                    for x in rng['cuda']), 'Missing native CUDA RNG bytes')


def check_optimizer(cp, model, data, steps):
    raw = cp['optimizer_state_dict']
    require(finite(raw), 'Nonfinite optimizer state')
    require({int(v['step'].item()) for v in raw['state'].values()} == {steps},
            'Optimizer counters disagree with full-epoch exposure')
    opt = training.build_optimizer(model, lr=data['optimizer']['lr'],
                                  time_head_lr_multiplier=data['model']['time_head_lr_multiplier'])
    require(training.optimizer_group_contract(opt) == cp['optimizer_group_contract'],
            'Fresh CPU optimizer parameter groups differ')
    opt.load_state_dict(raw)
    require(training.optimizer_group_contract(opt) == cp['optimizer_group_contract'],
            'Restored CPU optimizer parameter groups differ')
    names = {id(v): k for k, v in model.named_parameters()}
    missing = []
    for group in opt.param_groups:
        require(group['weight_decay'] == data['optimizer']['weight_decay']
                and tuple(group['betas']) == (0.9, 0.999) and group['eps'] == 1e-8,
                'AdamW hyperparameters differ')
        require(not group['amsgrad'] and not group['maximize'], 'Unexpected AdamW variant')
        for param in group['params']:
            state = opt.state.get(param)
            if state is None:
                missing.append(names[id(param)])
                continue
            for name in ('exp_avg', 'exp_avg_sq'):
                require(state[name].shape == param.shape and finite(state[name]),
                        'Restored optimizer tensor shape/finiteness mismatch')
    require(len(opt.state) == len(raw['state']), 'Optimizer state count changed on load')
    expected_missing = []
    require(missing == expected_missing, 'Unexpected absent optimizer state: ' + repr(missing))
    return {'restored_entries': len(raw['state']), 'structurally_unused_parameters_without_state': missing}


def check_state(cp, model, key='model_state_dict', hash_key='model_state_sha256'):
    state = cp[key]
    require(finite(state), 'Nonfinite checkpoint tensor')
    digest = canonical_state_dict_sha256(state)
    require(digest == cp[hash_key], 'Embedded model tensor SHA mismatch')
    model.load_state_dict(state, strict=True)
    require(canonical_state_dict_sha256(model.state_dict()) == digest,
            'Strict-loaded CPU model tensor SHA changed')
    return digest


def audit_job(kind, original, contract, qualification, job):
    d = original / 'run' / job['id']
    run = d / 'runs' / job['arm'] / variant_for(job['arm']) / f"seed_{job['seed']}"
    c, q = contract, qualification
    data = next(x for x in c['datasets'] if x['dataset_id'] == job['dataset'])
    h = read(run / 'history.json')['history']
    exp = read(run / 'exposure.json')
    timing = read(run / 'epoch_timing.json')
    summary = read(run / 'summary.json')
    replay = read(run / 'endpoint_replays.json')
    n = len(h)
    require(finite(h), 'Nonfinite history')
    best = min(h, key=lambda x: x['val_qty_rmse'])
    manifest = read(d / 'terminal_manifest.json')
    require(manifest['scientific_success'] is True and manifest['job'] == job
            and manifest['contract_sha256'] == canonical(c), 'Terminal manifest identity')
    for rel, digest in manifest['files'].items():
        require(sha(d / rel) == digest, f'Terminal file SHA: {rel}')
    collection = read(original / 'collection_manifest.json')
    status = collection['host_status_at_collection']
    completions = collection.get('terminal_completion_index', status.get('completed', status.get('outcomes', {})))
    require(completions[job['id']]['terminal_manifest_sha256'] == sha(d / 'terminal_manifest.json'),
            'Supervisor terminal SHA differs')

    initial = q['initialization'][job['dataset']][str(job['seed'])][job['arm']]
    require(read(d / 'initialization.json') == q['initialization'][job['dataset']][str(job['seed'])],
            'Native per-job initialization differs from qualification')
    scientific_contract_sha = canonical(c)
    receipt = read(d / 'input_receipt.json')
    require(receipt == q['inputs'][job['dataset']]['receipt'], 'Native input receipt changed after qualification')
    identity = data['inherited_data_identity']
    require(receipt['held_out_materialized'] is False and receipt['populations'] == identity['populations'],
            'Population identity/held-out scope')
    require(receipt['input_identity'] == {'data_sha256': identity['data']['sha256'],
            'split_manifest_sha256': identity['split_manifest']['sha256']}, 'Data/split SHA')
    require(receipt['all_train_rows']['quantity_sha256'] == data['statistics']['all_train_quantity_sha256'],
            'Train-only quantity statistics identity')
    for key in ('train_log_mean', 'train_log_std', 'raw_scale'):
        require(receipt[key] == data['statistics'][key], 'Frozen train statistics changed')

    last = torch.load(run / 'last_epoch_state.pt', map_location='cpu', weights_only=False)
    selected = torch.load(run / 'best_val_qty_rmse_model.pt', map_location='cpu', weights_only=False)
    model, _ = build_model(data, job['arm'])
    require(all(p.device.type == 'cpu' for p in model.parameters()), 'Non-CPU model')
    require(sum(p.numel() for p in model.parameters()) == summary['parameter_count'], 'Parameter count')
    state_hashes = {}
    for role, cp, epoch in [('last', last, n), ('selected', selected, best['epoch'])]:
        validate_checkpoint_route(cp, job['arm'])
        require(cp.get('epoch', cp.get('best_epoch')) == epoch and cp['seed'] == job['seed'], 'Checkpoint epoch/seed')
        require(cp['initial_state_sha256'] == initial, 'Native initial hash chain')
        require(cp['evaluation_scope'] == 'validation_only' and cp['held_out_test_evaluated'] is False,
                'Checkpoint evaluation scope')
        require(cp['variant'] == variant_for(job['arm']) and cp['backbone'] == job['arm'], 'Checkpoint model route')
        require(cp['interface_meta']['execution_contract_sha256'] == scientific_contract_sha,
                'Scientific checkpoint contract SHA')
        require(cp['interface_meta']['source_files_sha256'] == c['source']['files_sha256'], 'Checkpoint source closure')
        require(cp['source_revision'] == c['source']['base_git_revision'], 'Source revision')
        require(cp['checkpoint_monitor'] == c['training']['monitor']
                and cp['checkpoint_monitor_history_key'] == 'val_qty_rmse', 'RMSE selection contract')
        state_hashes[role] = check_state(cp, model)
        for key in ('resume_identity', 'interface_meta', 'source_revision', 'source_revision_history',
                    'optimizer_group_contract', 'checkpoint_monitor', 'checkpoint_monitor_history_key',
                    'checkpoint_selection', 'encoder_config', 'selection_formula'):
            require(cp[key] == summary[key], 'Checkpoint/summary identity: ' + key)
    state_hashes['embedded_best'] = check_state(last, model, 'best_state_dict', 'best_state_sha256')
    require(state_hashes['selected'] == state_hashes['embedded_best'] == summary['checkpoint_state_sha256'],
            'Selected checkpoint differs from embedded best')
    require(last['history'] == h and last['best_epoch'] == best['epoch']
            and last['best_selection_value'] == best['val_qty_rmse'], 'Saved history/selection mismatch')
    require(summary['initial_state_sha256'] == initial, 'Summary initial hash')
    require(summary['evaluation_scope'] == 'validation_only', 'Summary evaluation scope')
    for key in METRICS:
        require(close(summary['best_val_' + key], best['val_' + key]), 'Summary selected metric: ' + key)
    args = last['resume_identity']['arguments']
    expected_args = {'batch_size': 128, 'epochs': 300, 'min_epochs': 40, 'early_stopping_patience': 40,
                     'lr': data['optimizer']['lr'], 'grad_clip': data['optimizer']['grad_clip'],
                     'max_seq_len': data['loader']['max_seq_len'], 'max_train_batches': None,
                     'max_val_batches': None, 'max_series': None, 'lambda_log_qty': 1.0, 'lambda_tail': 0.0,
                     'data_sha256': identity['data']['sha256'], 'split_manifest_sha256': identity['split_manifest']['sha256']}
    require(all(args[k] == v for k, v in expected_args.items()), 'Actual training arguments differ from frozen contract')

    populations = identity['populations']
    train_count, val_count = (populations[k]['target_count'] for k in ('train', 'validation'))
    batches = math.ceil(train_count / 128)
    require(data['steps_per_epoch'] == batches, 'Full train steps per epoch')
    arm_audit = audit_arm(h, summary, exp, data, c['training'])
    require(all(x['train_all_finite'] and x['train_batch_count'] == batches
                and x['train_event_count'] == train_count for x in h), 'Full train exposure history')
    require(len(exp['train']) == n and len(exp['validation']) == n + 1, 'Epoch and final-validation exposure counts')
    require(all(x['count'] == val_count and x['batches'] == math.ceil(val_count / 128)
                for x in exp['validation']), 'Full validation exposure')
    require(len({x['batch_order_sha256'] for x in exp['validation']}) == 1, 'Validation order changed')
    references = []
    refs = c.get('reuse', [])
    if job['host'] == 'a100':
        refs = [{'host': 'a100', 'dataset': ds, 'seed': 42, 'arm': 'titantpp_history_mlp', 'file_sha256': {'exposure.json': val['exposure_sha256']}} for ds, val in c['baseline_replays'].items()]
    for ref in refs:
        if ref['host'] == job['host'] and ref['dataset'] == job['dataset'] and ref['seed'] == job['seed']:
            name = f"reference_exposure/{ref['dataset']}__{ref['seed']}__{ref['arm']}.json"
            require(sha(original / name) == ref['file_sha256']['exposure.json'], 'Reference exposure SHA')
            references.append((name, read(original / name)))
    require(len(references) == 1, 'Expected one existing History MLP exposure reference')
    overlaps = []
    for name, baseline in references:
        overlap = min(n, len(baseline['train']))
        require(overlap > 0 and exp['train'][:overlap] == baseline['train'][:overlap], 'Common train batch prefix: ' + name)
        require(exp['validation'][0] == baseline['validation'][0], 'Common validation population/order: ' + name)
        overlaps.append({'path': name, 'sha256': sha(original / name), 'epochs': overlap})
    optimizer_states = check_optimizer(last, model, data, n * batches)
    check_rng(last)
    require([x['epoch'] for x in timing['epochs']] == list(range(1, n + 1)), 'Continuous epoch timing')
    require(all(x['elapsed_seconds'] > 0 and math.isfinite(x['elapsed_seconds'])
                and x['finished_at_unix'] >= x['started_at_unix'] for x in timing['epochs']), 'Invalid timing')
    claim = read(original / 'claims' / (job['id'] + '.json'))
    require(claim['job'] == job and claim['contract_sha256'] == canonical(c), 'Claim identity')
    require(timing['epochs'][-1]['finished_at_unix'] <= claim['deadline_unix']
            <= read(original / 'start_permit.json')['deadline_unix'], 'Recorded condition deadline exceeded/extended')
    epoch_receipt = read(run / 'server_checkpoint_receipt.json')
    require(epoch_receipt['epoch'] == n and epoch_receipt['mac_ack_required'] is False, 'Last epoch receipt')
    for name, digest in epoch_receipt['files'].items():
        if name == 'exposure.json':
            epoch_exp = {**exp, 'validation': exp['validation'][:-1]}
            encoded = (json.dumps(epoch_exp, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
            require(hashlib.sha256(encoded).hexdigest() == digest, 'Pre-final-validation exposure SHA')
        else:
            require(sha(run / name) == digest, 'Epoch receipt SHA: ' + name)

    require(replay['status'] == 'complete' and replay['job'] == job and replay['evaluation_scope'] == 'validation_only',
            'Replay completion/scope')
    require(replay['best_epoch'] == best['epoch'] and replay['completed_epochs'] == n
            and replay['global_steps'] == n * batches, 'Replay history/step identity')
    for role, epoch in [('selected', best['epoch']), ('last', n)]:
        endpoint = replay[role]
        require(endpoint['evaluation_scope'] == 'validation_only' and endpoint['held_out_test_evaluated'] is False,
                'Endpoint scope')
        audit_replay_accounting(endpoint)
        require(endpoint['state_sha256'] == state_hashes[role] and endpoint['count'] == val_count,
                'Endpoint tensor/population identity')
        for key in METRICS:
            require(close(endpoint[key], h[epoch - 1]['val_' + key]), 'Replay/history metric: ' + role + '/' + key)
        for part in ('quantity', 'history', 'additional_history'):
            require(endpoint[part + '_boundaries'] == data[{'quantity': 'quantity_boundaries_all_train_rows',
                    'history': 'history_boundaries', 'additional_history': 'additional_history_boundaries'}[part]],
                    'Replay frozen boundaries')
            cells = endpoint[part + '_cells']
            require(close(sum(x['time_nll'] * x['count'] for x in cells if x['count']),
                          endpoint['time_nll'] * val_count), 'Partition time NLL accounting')
            if role == 'selected' and part in ('quantity', 'history'):
                # summary.history_rows denotes history-length strata, not epochs.
                if part == 'quantity':
                    by_label = {x['label']: x for x in summary['quantity_rows']}
                    require(set(by_label) == {f'frozen_quantity_bin_{i}' for i in range(5)}, 'Summary quantity bin identity')
                    strata = [by_label[f'frozen_quantity_bin_{i}'] for i in range(5)]
                else:
                    strata = sorted(summary['history_rows'], key=lambda x: x['stratum_order'])
                # Legacy summary omits empty history bins; endpoint records
                # explicitly preserve them as zero count and unavailable metrics.
                summary_cells = cells if part == 'quantity' else [x for x in cells if x['count']]
                require(len(strata) == len(summary_cells), 'Summary nonempty strata count')
                for a, b in zip(strata, summary_cells):
                    if part == 'history':
                        require(a['stratum_order'] == b['bin'], 'Summary history bin identity')
                    require(a['count'] == b['count'], 'Summary stratum population')
                    if a['count']:
                        require(all(close(a[k], b[k]) for k in METRICS), 'Summary stratum metrics')

    return {'job': job, 'status': 'passed', 'completed_epochs': n, 'selected_epoch': best['epoch'],
            'optimizer_steps': n * batches, 'optimizer_cpu_audit': optimizer_states,
            'checkpoint_files': {name: sha(run / name) for name in ('best_val_qty_rmse_model.pt', 'last_epoch_state.pt')},
            'model_tensor_sha256': state_hashes, 'native_initial_state_sha256': initial,
            'native_initialization_evidence_reused': True, 'strict_cpu_loading': True,
            'rng_and_shuffle_cpu_restore': True, 'cuda_rng_bytes_verified_not_executed': True,
            'full_train_targets_per_epoch': train_count, 'validation_targets': val_count,
            'baseline_batch_prefixes': overlaps, 'frozen_exposure_audit': arm_audit,
            'first_strict_rmse_and_first_stop_verified': True, 'epoch_exposure_prefix_sha_verified': True,
            'replay_records_audited_without_rerun': 2, 'replay_partition_accounting_verified': True,
            'selected_validation': {k: replay['selected'][k] for k in METRICS},
            'last_validation': {k: replay['last'][k] for k in METRICS},
            'fit_elapsed_seconds': summary['elapsed_seconds'],
            'fit_elapsed_scope': 'train_one',
            'epoch_timing_seconds_sum': sum(x['elapsed_seconds'] for x in timing['epochs']),
            'parameter_count': summary['parameter_count'], 'peak_allocated_bytes': summary['cuda_peak_memory_allocated_bytes'],
            'peak_reserved_bytes': summary['cuda_peak_memory_reserved_bytes'],
            'verified_terminal_files': len(manifest['files'])}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('host', choices=('5080', '5090', 'pro4500', 'a100')); args = ap.parse_args()
    host = args.host; target = DEST / host; original = target / 'original'
    require(not (target / 'terminal_audit.json').exists(), 'Preserve completed audit; no overwrite')
    scope = read(REPORT / 'scope.json'); entry = scope['hosts'][host]; pinned = entry['jobs']
    BUNDLE = Path(entry['bundle']); EXPECTED = entry['contract_sha256']; CLOSURE = entry['source_closure_sha256']
    require(sha(entry['snapshot']) == entry['snapshot_sha256'], 'Pinned observation SHA')
    collection = read(original / 'collection_manifest.json'); retrieval = read(target / 'retrieval_receipt.json')
    require(sha(target / 'original.tar') == retrieval['archive_sha256'], 'Archive SHA')
    require(retrieval['scope_sha256'] == sha(REPORT / 'scope.json'), 'Pinned scope SHA')
    require(collection['pinned_jobs'] == pinned and collection['running_jobs_excluded'], 'Collection subset identity')
    for rel, record in collection['files'].items():
        require(sha(original / rel) == record['sha256'] and (original / rel).stat().st_size == record['bytes'],
                'Collection file SHA/size: ' + rel)
    c = read(original / 'execution_contract.json')
    require(canonical(c) == EXPECTED == canonical(read(BUNDLE / 'execution_contract.json')), 'Original contract SHA')
    require(canonical(c['source']['files']) == c['source']['files_sha256'] == CLOSURE, 'Source closure SHA')
    require(len(c['source']['files']) == (110 if host == 'a100' else 109), 'Scientific source count')
    require(c['evaluation_scope'] == 'validation_only' and c['held_out_test_evaluated'] is False, 'Contract evaluation scope')
    for rel, digest in c['source']['files'].items():
        require(sha(original / 'source' / rel) == digest, 'Frozen source SHA: ' + rel)
    approval = read(original / 'approval.json'); permit = read(original / 'start_permit.json')
    require(approval['approved'] and approval['contract_sha256'] == EXPECTED
            and permit['contract_sha256'] == EXPECTED and permit['approval_sha256'] == canonical(approval), 'Approval chain')
    q = read(original / 'qualification/receipt.json')
    require(q['status'] == 'passed' and q['contract_sha256'] == EXPECTED
            and q['source_files_sha256'] == CLOSURE, 'Native qualification chain')
    training_permit = read(original / 'training_permit.json')
    require(training_permit['contract_sha256'] == EXPECTED and
            training_permit['start_permit_sha256'] == canonical(permit), 'Training permit chain')
    require(q['correctness']['status'] == 'passed', 'Native correctness qualification')
    if host == 'a100':
        require(training_permit['qualification_sha256'] == sha(original/'qualification/receipt.json'), 'A100 native permit')
        require(q['selected_concurrency'] == 3 and q['held_out_test_evaluated'] is False, 'A100 qualified concurrency/scope')
        require(set(q['parallel_measurements']) == {'1','2','3','6'}, 'A100 concurrency probes')
        for value in q['parallel_measurements'].values():
            require(value['measured_steps'] == 180 and value['synthetic_optimizer_updates_including_warmup'] == 198, 'A100 qualification work accounting')
    else:
        require(training_permit['qualifications'][host] == q, 'Native training permit')
        require(q['synthetic_optimizer_updates'] == 75 and q['correctness_optimizer_updates'] == 15
                and q['cost_probe_optimizer_updates'] == 60 and q['owned_stop_test'] == 'passed', 'Native correctness qualification')
    if host == 'pro4500':
        reservation = read(original/'source_reservation.json')
        require(canonical(reservation) == training_permit['source_reservation_sha256'], 'PRO4500 source reservation')
        parent = read(ROOT/'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/execution_contract.json')
        require(canonical(parent) == c['parent_contract_sha256'], 'Parent contract SHA')
        changed = [k for k,v in c['source']['files'].items() if parent['source']['files'].get(k) != v]
        require(changed == ['paper/scripts/run_pakdd_extension.py'], 'Transfer scientific source differences')
    work = REPORT / 'audit_workspace' / host
    for rel, digest in c['source']['files'].items():
        p = work / rel; p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): shutil.copyfile(original / 'source' / rel, p)
        require(sha(p) == digest, 'CPU import source SHA: ' + rel)
    (work / 'sample_data').mkdir(exist_ok=True)
    sys.path.insert(0, str(work))
    global np, torch, training, build_model, variant_for, audit_arm, audit_replay_accounting, validate_checkpoint_route, canonical_state_dict_sha256
    import numpy as np
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.run_local_detail_benchmark import audit_arm, audit_replay_accounting
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    if host == 'a100':
        from paper.scripts.run_titantpp_mlp_candidates import build_model, variant_for
    else:
        from paper.scripts.run_pakdd_extension import build_model, variant_for
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    rows = []
    for item in pinned:
        job = item['job']; require(job in c['jobs'] and job['host'] == host, 'Pinned job contract identity')
        require(sha(original / 'run' / job['id'] / 'terminal_manifest.json') == item['terminal_manifest_sha256'],
                'Pinned observation manifest SHA')
        row = audit_job('extension', original, c, q, job); rows.append(row)
        print(json.dumps({'job': job['id'], 'status': 'passed', 'epochs': row['completed_epochs'],
                          'selected': row['selected_epoch'], 'optimizer_steps': row['optimizer_steps']}), flush=True)
    require(len(rows) == len(pinned), 'Incomplete authorized audit scope')
    for rel, record in collection['files'].items():
        require(sha(original / rel) == record['sha256'], 'Original changed during audit: ' + rel)
    result = {'status': 'passed', 'host': host, 'audited_unix': time.time(),
              'scope_sha256': sha(REPORT / 'scope.json'),
              'collected_unix': collection['collected_unix'], 'verified_retrieved_files': len(collection['files']),
              'verified_source_files': len(c['source']['files']), 'verified_terminal_checkpoints': len(rows) * 2,
              'verified_replay_roles': len(rows) * 2, 'conditions': rows,
              'contract_sha256': EXPECTED, 'source_closure_sha256': CLOSURE,
              'archive_sha256': retrieval['archive_sha256'], 'audit_script_sha256': sha(__file__),
              'native_qualification_sha256': sha(original / 'qualification/receipt.json'),
              'native_runtime': q['runtime'],
              'local_cpu_audit_runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                                          'numpy': np.__version__, 'platform': platform.platform()},
              'evaluation_scope': 'validation_only', 'new_training_updates': 0, 'new_model_forward_calls': 0,
              'new_replay_calls': 0, 'gpu_calls': 0, 'originals_unchanged': True,
              'limitations': ['CPU strict loading and saved-record accounting; no new prediction/replay.',
                             'Native CUDA initialization/correctness is reused from SHA-verified qualification evidence; '
                             'the local CPU runtime is not the native training runtime.',
                             'CUDA RNG bytes are preserved, not executed on CPU.',
                             'Original data files were not reread; native input SHA/population receipts and exposure prefixes are audited.',
                             '22 newly audited conditions at pinned 21:57 KST cutoff; running and later completions excluded. Prior 13 audit reused separately.']}
    (target / 'terminal_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'result': str(target / 'terminal_audit.json'), 'status': 'passed', 'conditions': len(rows)}), flush=True)
if __name__ == '__main__': main()
