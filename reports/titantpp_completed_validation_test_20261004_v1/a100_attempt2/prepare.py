"""Bind twelve recovered A100 selected checkpoints; never contact a server."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1'
CAMPAIGNS = {
    'mlp_candidates': ('titantpp_mlp_candidates_a100_20261002_v1_retry1',
                       '4b1ba84f73119fce0ef57ba24954af5b3739762e3c68732a57e3801a1fad2019',
                       '11ad8517823e9a72c5c0aa50e507e0489f95039c0fd0d91775137e502a1c6034'),
    'routing_placement': ('titantpp_routing_placement_a100_20261003_v1',
                          '1db5a7e0cf97b8a667439b86cca8455a6e98158775c46ef1d19efdd26a5cbea0',
                          '2b8156fab28cf44e9c2fd4cd1f2888110f89f98874af5820a60037d44cc51870'),
}
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def write(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def select_first_minimum(history):
    require(bool(history), 'Empty history')
    require([r['epoch'] for r in history] == list(range(1, len(history) + 1)),
            'History must preserve all chronological epochs, including resumed originals')
    finite = [r for r in history if math.isfinite(r['val_qty_rmse'])]
    require(bool(finite), 'No finite validation metric')
    return min(finite, key=lambda row: row['val_qty_rmse'])


def inventory():
    rows, bindings, bundles, deployment = [], [], {}, {}
    for name, (directory, contract_sha, closure) in CAMPAIGNS.items():
        bundle = ROOT / 'search_artifacts' / directory
        original = bundle / 'retrieved/original'
        contract_path = bundle / 'execution_contract.json'
        contract = read(contract_path)
        require(canonical(contract) == contract_sha, 'Scientific contract canonical SHA changed')
        require(contract['source']['files_sha256'] == closure, 'Unexpected source closure')
        # Training's closure format is SHA256 of compact, sorted JSON source map.
        require(canonical(contract['source']['files']) == closure, 'Source closure does not bind its files')
        source = bundle / 'source'
        for path, digest in contract['source']['files'].items():
            require(sha(source / path) == digest, 'Frozen source bytes changed: ' + path)
            deployment[relative(source / path)] = digest
        bundles[name] = {'source_root': relative(source), 'source_files': contract['source']['files'],
                         'source_closure_sha256': closure, 'datasets': contract['datasets']}
        snapshot_path = bundle / 'retrieved/terminal_snapshot.json'
        snapshot = read(snapshot_path)
        require(read(bundle / 'retrieved/retrieval_receipt.json')['status'] == 'sha_verified',
                'Original retrieval is not SHA verified')
        manifests = sorted(original.glob('**/terminal_manifest.json'))
        require(len(manifests) == 6, 'Each campaign must contain exactly six terminal fits')
        for path in manifests:
            terminal = read(path)
            job = terminal['job']
            require(terminal['status'] == 'complete' and terminal['scientific_success'] is True,
                    'Condition is not scientifically complete')
            require(terminal['contract_sha256'] == contract_sha and job in contract['jobs'],
                    'Terminal job/contract changed')
            require(job['seed'] == 42 and job['dataset'] in DATASETS, 'Unexpected condition')
            resume = name == 'mlp_candidates' and job['dataset'] == 'intermittent_frozen_5000'
            require(('resume/run' in str(path.relative_to(original))) == resume,
                    'Intermittent must bind resumed final evidence, not the original two-epoch run')
            observed = snapshot['resume_runs'] if resume else snapshot['runs']
            if isinstance(observed, dict):
                observed = list(observed.values())
            matches = [r for r in observed if r.get('job') == job]
            require(len(matches) == 1 and matches[0]['terminal_manifest'] == terminal,
                    'Terminal differs from preserved terminal observation')
            folder = path.parent
            selected_name = next(f for f in terminal['files'] if f.endswith('/best_val_qty_rmse_model.pt'))
            checkpoint = folder / selected_name
            history_path = checkpoint.with_name('history.json')
            replay_path = checkpoint.with_name('endpoint_replays.json')
            bound = {}
            for f in (checkpoint, history_path, replay_path, path):
                digest = sha(f)
                if f != path:
                    require(terminal['files'][str(f.relative_to(folder))] == digest,
                            'Terminal-bound original bytes changed: ' + str(f))
                bound[relative(f)] = digest
                deployment[relative(f)] = digest
            replay = read(replay_path)
            require(replay['status'] == 'complete' and replay['job'] == job and
                    replay['evaluation_scope'] == 'validation_only' and
                    replay['held_out_test_evaluated'] is False, 'Invalid endpoint replay')
            require('last' in replay and 'selected' in replay, 'Both endpoint replays required')
            history = read(history_path)['history']
            selected = select_first_minimum(history)
            require(selected['epoch'] == replay['best_epoch'], 'Selection changed')
            for metric in ('qty_mae', 'qty_rmse', 'time_nll'):
                require(math.isclose(selected['val_' + metric], replay['selected'][metric],
                                     rel_tol=1e-10, abs_tol=1e-8), 'Selected history/replay differs')
            spec = next(d for d in contract['datasets'] if d['dataset_id'] == job['dataset'])
            require(replay['selected']['count'] == spec['inherited_data_identity']['populations']['validation']['target_count'],
                    'Validation population changed')
            row = {'dataset': job['dataset'], 'model': job['arm'], 'seed': 42,
                   'endpoint': 'selected', 'evaluator_source_bundle': name,
                   'checkpoint_path': relative(checkpoint), 'checkpoint_file_sha256': sha(checkpoint),
                   'state_tensor_sha256': replay['selected']['state_sha256'],
                   'selected_epoch': replay['best_epoch'], 'validation_count': replay['selected']['count']}
            rows.append(row)
            bindings.append({**row, 'job_id': job['id'], 'resumed_same_fit': resume,
                'scientific_success': True, 'terminal_manifest_path': relative(path),
                'terminal_manifest_sha256': sha(path), 'original_files': bound,
                'snapshot_path': relative(snapshot_path), 'snapshot_sha256': sha(snapshot_path),
                'training_contract_path': relative(contract_path), 'training_contract_canonical_sha256': contract_sha,
                'history_path': relative(history_path), 'history_sha256': sha(history_path),
                'first_validation_minimum_confirmed': True,
                'validation_reference': {k: replay['selected'][k] for k in ('count', 'qty_mae', 'qty_rmse', 'time_nll')},
                'validation_tail_reference': replay['selected']['tail'],
                'tail_threshold': spec['quantity_boundaries_all_train_rows'][3],
                'selected_endpoint_replay_path': relative(replay_path),
                'selected_endpoint_replay_sha256': sha(replay_path),
                'last_binary_cpu_audit': 'not performed by this evaluation preparation'})
    order = lambda r: (DATASETS.index(r['dataset']), r['model'])
    rows.sort(key=order)
    bindings.sort(key=order)
    require(len(rows) == len({(r['dataset'], r['model'], r['seed']) for r in rows}) == 12,
            'Incomplete or duplicate A100 conditions')
    return rows, bindings, bundles, deployment


def main():
    rows, bindings, bundles, deployment = inventory()
    scope_path = HERE.parent / 'scope_and_approval.json'
    scope = read(scope_path)
    require(scope['approval']['test_inference_authorized'] is True and
            scope['approval']['retraining_authorized'] is False, 'Parent scope approval missing')
    for name in ('evaluate.py', 'dataset_manifest.json'):
        require(not (HERE / name).exists(), 'Preparation is immutable; refusing overwrite')
        (HERE / name).write_bytes((LEGACY / name).read_bytes())
    write('evaluation_registry.json', {'rows': rows, 'bundles': bundles})
    write('selected_binding.json', {'selection_unchanged': True, 'rows': bindings,
          'not_scheduled': [{'dataset': 'insta_market_basket', 'model': m, 'seed': 42,
                             'status': 'not_scheduled'} for m in sorted({r['model'] for r in rows})]})
    baseline = []
    legacy_registry = read(ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json')
    for dataset in DATASETS:
        folder = LEGACY / 'runs/test/attempt1' / f'{dataset}__titantpp_history_mlp__seed42'
        receipt_path = folder / 'receipt.json'
        receipt = read(receipt_path)
        require(receipt['status'] == 'complete' and receipt['full_population'] and
                receipt['dataset'] == dataset and receipt['model'] == 'titantpp_history_mlp' and
                receipt['seed'] == 42 and receipt['split'] == 'test' and not receipt['retrained'],
                'Baseline seed42 Test receipt invalid')
        selected_row = next(r for r in legacy_registry['rows'] if
            (r['dataset'], r['model'], r['seed']) == (dataset, 'titantpp_history_mlp', 42))
        require(all(receipt[k] == selected_row[k] for k in
                    ('checkpoint_file_sha256', 'state_tensor_sha256', 'selected_epoch')),
                'Baseline receipt no longer matches its selected checkpoint registry')
        require(receipt['runner_sha256'] == sha(LEGACY / 'evaluate.py'), 'Baseline evaluator changed')
        files = {relative(receipt_path): sha(receipt_path)}
        for part in receipt['parts']:
            p = folder / part['path']
            require(sha(p) == part['sha256'], 'Legacy prediction part changed')
            files[relative(p)] = part['sha256']
        baseline.append({'dataset': dataset, 'model': receipt['model'], 'seed': 42,
            'folder': relative(folder), 'receipt_sha256': sha(receipt_path), 'files': files,
            'identity': {k: receipt[k] for k in ('data_file_sha256', 'expected_target_count',
                'prediction_rows', 'target_identity_sha256', 'truth_sha256', 'loader')}})
        deployment.update(files)
    write('comparison_binding.json', {'rows': baseline, 'baseline_predictions_reused': True})
    resources = dict(scope['resources'])
    resources['root'] += '/a100_workspace'
    manifest = read(HERE / 'dataset_manifest.json')
    contract = {'schema_version': 1, 'route': scope['route'], 'not_independent_untouched_data': True,
        'approval': {**scope['approval'], 'datasets': list(DATASETS),
                     'models': sorted({r['model'] for r in rows}), 'seeds': [42]},
        'scope_and_approval_sha256': sha(scope_path),
        'registry_sha256': sha(HERE / 'evaluation_registry.json'),
        'selected_binding_sha256': sha(HERE / 'selected_binding.json'),
        'comparison_binding_sha256': sha(HERE / 'comparison_binding.json'),
        'dataset_manifest_sha256': sha(HERE / 'dataset_manifest.json'),
        'dataset_sha256': {r['dataset']: r['sha256'] for r in manifest['datasets']},
        'resources': resources, 'source_training_gpu': 'A100 SXM',
        'evaluation_gpu': 'RTX 5080; cross-device replay qualification, not native A100 runtime reproduction',
        'validation_gate': {'rtol': 1e-5, 'atol': 1e-5, 'all_12_full_validation_pass_before_any_test': True,
                            'metrics': ['qty_mae', 'qty_rmse', 'time_nll']},
        'tail_rule': 'raw_quantity > frozen all-train quantity_boundaries[3]',
        'prediction_protocol': scope['prediction_protocol'],
        'evaluator': {'path': relative(HERE / 'evaluate.py'), 'sha256': sha(HERE / 'evaluate.py'),
                      'parent_path': relative(LEGACY / 'evaluate.py'), 'byte_identical': True},
        'code_sha256': {n: sha(HERE / n) for n in ('prepare.py', 'run_all.py', 'verify.py')},
        'interpretation': 'Exploratory seed42-only existing-split reevaluation; no 3seed SD or efficiency claim.'}
    write('execution_contract.json', contract)
    deployment.update({relative(HERE / n): sha(HERE / n) for n in
        ('prepare.py', 'evaluate.py', 'run_all.py', 'verify.py', 'evaluation_registry.json', 'selected_binding.json',
         'comparison_binding.json', 'dataset_manifest.json', 'execution_contract.json')})
    deployment[relative(scope_path)] = sha(scope_path)
    write('deployment_manifest.json', {'remote_root': resources['root'], 'files': deployment,
        'data_files_reuse_existing_5080_bytes': {r['path']: r['sha256'] for r in manifest['datasets']},
        'read_only_symlinks_allowed': True, 'remote_execution_started': False})
    write('preparation_receipt.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'conditions': len(rows), 'source_bundles': len(bundles), 'inference_splits_required': 24,
        'registry_sha256': sha(HERE / 'evaluation_registry.json'),
        'contract_sha256': sha(HERE / 'execution_contract.json'), 'selected_binary_tensor_preflight': 'pending',
        'remote_execution_started': False, 'instacart': 'not_scheduled'})
    print(json.dumps({'conditions': len(rows), 'source_bundles': len(bundles), 'remote_execution_started': False}))


if __name__ == '__main__':
    main()
