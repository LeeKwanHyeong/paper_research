"""Prepare a local immutable aggregate Test batch; no SSH or scientific inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINING = ROOT / 'search_artifacts/titantpp_history_width8_12_dual_20261004_v1'
REFERENCE = ROOT / 'reports/titantpp_completed_validation_test_20261004_v1/width16'
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
MODELS = ('titantpp_history_mlp_width8', 'titantpp_history_mlp_width12')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    return f"{row['dataset']}__{row['model']}__seed{row['seed']}"


def scientific_job(job):
    return job['dataset'], job['arm'], job['seed']


def dataset_science(datasets):
    """Drop only operational input paths; preserve bytes, split, population, and model."""
    values = deepcopy(datasets)
    for spec in values:
        for key in ('inherited_data_identity', 'parent_data_identity'):
            for identity in ('data', 'split_manifest'):
                value = spec.get(key, {}).get(identity, {})
                if 'path' in value:
                    value['path'] = '<operational-path>'
    return values


def first_minimum(history):
    rows = history['history']
    require(bool(rows) and [row['epoch'] for row in rows] == list(range(1, len(rows) + 1)), 'History epochs are not contiguous from one')
    require(all(math.isfinite(row['val_qty_rmse']) and row.get('train_all_finite') is True
                and all(not isinstance(value, float) or math.isfinite(value) for value in row.values())
                for row in rows), 'History is not all finite')
    return min(rows, key=lambda row: row['val_qty_rmse'])


def prepare(args):
    os.environ['MPLCONFIGDIR'] = str(HERE / 'cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(HERE / 'cache')
    sys.path.insert(0, str(ROOT))
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    parent_path = TRAINING / 'execution_contract.json'
    parent = read(parent_path)
    contracts = {sha_json(read(path)): read(path) for path in [parent_path, *args.training_contract]}
    require(sha_json(parent) == read(TRAINING / 'current.json')['contract_sha256'], 'Parent canonical contract identity differs')
    authorization_path = ROOT / 'reports/titantpp_capacity_test_handoff_20261004_v1/approval.json'
    authorization = read(authorization_path)
    require(authorization['approved'] is True and authorization['parent_contract_sha256'] == sha_json(parent), 'Human authorization/parent contract differs')
    for value in contracts.values():
        require(value['source']['files'] == parent['source']['files'], 'Training source closure differs from parent')
        require(value['source']['files_sha256'] == parent['source']['files_sha256'], 'Training closure digest differs')
        require(dataset_science(value['datasets']) == dataset_science(parent['datasets']) and value['training'] == parent['training']
                and value['quantity_variants'] == parent['quantity_variants'], 'Derived training scientific contract changed')
    source = TRAINING / 'frozen_source'
    for relative, digest in parent['source']['files'].items():
        require(sha(source / relative) == digest, f'Frozen source changed: {relative}')
    approved = {scientific_job(job): job for job in parent['jobs']}
    expected = {key for key in approved if key[0] in args.datasets}
    if args.job_ids:
        require(set(args.job_ids) <= {job['id'] for job in parent['jobs']}, 'Unknown requested job ID')
        expected = {key for key in expected if approved[key]['id'] in args.job_ids}
    require(bool(expected), 'Empty batch')
    require(not any(key[0] == 'intermittent_frozen_5000' for key in expected)
            or len([key for key in expected if key[0] == 'intermittent_frozen_5000']) == 6,
            'Intermittent Test requires all six training conditions to be complete')
    rows, bindings, discovered = [], [], {}
    roots = args.original_root or [TRAINING / 'retrieved/5080/run', TRAINING / 'retrieved/5090/run']
    for original_root in roots:
        for terminal_path in Path(original_root).rglob('terminal_manifest.json'):
            terminal = read(terminal_path)
            job = terminal['job']
            key = scientific_job(job)
            if key not in expected:
                continue
            require(key not in discovered, 'Duplicate original condition; pass only the canonical original root')
            require(job['id'] == approved[key]['id'], 'Job scientific ID changed')
            require(terminal['status'] == 'complete' and terminal['scientific_success'] is True, 'Training is not complete')
            require(terminal['contract_sha256'] in contracts, 'Unbound original/derived training contract')
            folder = terminal_path.parent
            names = ('best_val_qty_rmse_model.pt', 'history.json', 'endpoint_replays.json', 'initialization.json', 'input_receipt.json', 'status.json')
            selected_files = {}
            for name in names:
                matching = [relative for relative in terminal['files'] if Path(relative).name == name]
                require(len(matching) == 1, f'Missing or ambiguous original file: {name}')
                relative = matching[0]
                path = folder / relative
                require(path.resolve().is_relative_to(folder.resolve()), 'Terminal path escapes original folder')
                require(sha(path) == terminal['files'][relative], f'Original binary/file SHA mismatch: {path}')
                selected_files[name] = path
            endpoint = read(selected_files['endpoint_replays.json'])
            require(endpoint['status'] == 'complete' and endpoint['evaluation_scope'] == 'validation_only'
                    and endpoint['held_out_test_evaluated'] is False, 'Original endpoint completion/scope differs')
            for point in ('selected', 'last'):
                require(endpoint[point]['evaluation_scope'] == 'validation_only' and endpoint[point]['held_out_test_evaluated'] is False,
                        'Original selected/last endpoint scope differs')
            selected = endpoint['selected']
            history = read(selected_files['history.json'])
            best = first_minimum(history)
            require(endpoint['completed_epochs'] == len(history['history']), 'Completed epoch count differs from history')
            for point in ('selected', 'last'):
                require(endpoint[point]['count'] == selected['count']
                        and endpoint[point]['quantity_boundaries'] == selected['quantity_boundaries']
                        and all(math.isfinite(endpoint[point][key]) for key in ('qty_rmse', 'qty_mae', 'time_nll')),
                        'Selected/last endpoint population or finiteness differs')
            checkpoint = selected_files['best_val_qty_rmse_model.pt']
            payload = torch_load_checkpoint(checkpoint, map_location='cpu')
            state = canonical_state_dict_sha256(payload['model_state_dict'])
            require(payload['best_epoch'] == endpoint['best_epoch'] == best['epoch'], 'First Validation minimum epoch differs')
            require(math.isclose(best['val_qty_rmse'], selected['qty_rmse'], rel_tol=1e-12, abs_tol=1e-12), 'History/endpoint RMSE differs')
            require(state == selected['state_sha256'] == payload['model_state_sha256'], 'Selected tensor SHA differs')
            require((payload['backbone'], payload['seed'], payload['variant']) == (job['arm'], job['seed'], 'count_only_log_regression'), 'Checkpoint condition differs')
            require(payload['checkpoint_monitor'] == 'validation_raw_quantity_rmse' and payload['checkpoint_monitor_history_key'] == 'val_qty_rmse' and payload['checkpoint_selection'] == 'best_validation_raw_quantity_rmse', 'Frozen selector differs')
            require(payload['evaluation_scope'] == 'validation_only' and payload['held_out_test_evaluated'] is False, 'Original checkpoint scope differs')
            require(payload['source_revision'] == parent['source']['git_revision'] and payload['source_revision_history'] == [payload['source_revision']], 'Checkpoint source revision differs')
            require(read(selected_files['initialization.json'])[job['arm']] == payload['initial_state_sha256'], 'Initialization digest differs')
            row = {'dataset': job['dataset'], 'model': job['arm'], 'seed': job['seed'], 'endpoint': 'selected',
                   'evaluator_source_bundle': 'width8_12', 'checkpoint_path': str(checkpoint.relative_to(ROOT)),
                   'checkpoint_file_sha256': sha(checkpoint), 'state_tensor_sha256': state,
                   'selected_epoch': best['epoch'], 'validation_count': selected['count'],
                   'initial_state_sha256': payload['initial_state_sha256'], 'source_revision': payload['source_revision']}
            rows.append(row)
            bindings.append({'condition': identity(row), 'job': job, 'training_contract_sha256': terminal['contract_sha256'],
                             'terminal_manifest_path': str(terminal_path.relative_to(ROOT)), 'terminal_manifest_sha256': sha(terminal_path),
                             'original_files': {str(path.relative_to(ROOT)): sha(path) for path in selected_files.values()},
                             'selected_epoch': best['epoch'], 'first_full_validation_minimum_verified': True,
                             'validation_reference': selected, 'operational_host_does_not_change_scientific_identity': True})
            discovered[key] = folder
    require(set(discovered) == expected, f'Missing completed original conditions: {sorted(expected - set(discovered))}')
    rows.sort(key=lambda row: (DATASETS.index(row['dataset']), row['seed'], MODELS.index(row['model'])))
    selected_datasets = {row['dataset'] for row in rows}
    registry = {'rows': rows, 'bundles': {'width8_12': {'source_root': str(source.relative_to(ROOT)),
                'source_files': parent['source']['files'], 'source_closure_sha256': parent['source']['files_sha256'],
                'datasets': [spec for spec in parent['datasets'] if spec['dataset_id'] in selected_datasets]}}}
    dataset_manifest = read(REFERENCE / 'dataset_manifest.json')
    dataset_manifest['datasets'] = [d for d in dataset_manifest['datasets'] if d['dataset'] in selected_datasets]
    if args.dataset_root:
        for data in dataset_manifest['datasets']:
            data['path'] = str(Path(args.dataset_root) / data['path'])
    reference_contract = read(REFERENCE / 'execution_contract.json')
    output = Path(args.output).resolve()
    require(output.is_relative_to(HERE), 'Batch outputs must remain in this new report directory')
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    write('evaluation_registry.json', registry)
    write('dataset_manifest.json', dataset_manifest)
    write('selected_binding.json', {'selection_unchanged': True, 'rows': bindings})
    contract = {'schema_version': 1, 'approval': {'test_inference_authorized': True, 'retraining_authorized': False,
                'allowed_splits': ['validation', 'test'], 'datasets': list(DATASETS), 'models': list(MODELS), 'seeds': [42, 52, 62],
                'user_instruction': authorization['user_instruction'], 'source_approval_path': str(authorization_path.relative_to(ROOT)),
                'source_approval_sha256': sha(authorization_path)},
                'registry_sha256': sha(output / 'evaluation_registry.json'), 'dataset_manifest_sha256': sha(output / 'dataset_manifest.json'),
                'dataset_sha256': {d['dataset']: d['sha256'] for d in dataset_manifest['datasets']},
                'evaluator_sha256': sha(HERE / 'evaluate.py'), 'selected_binding_sha256': sha(output / 'selected_binding.json'),
                'population_references': {d: reference_contract['population_references'][d] for d in selected_datasets},
                'validation_references': {b['condition']: b['validation_reference'] for b in bindings},
                'validation_gate': {'conditions': len(rows), 'all_before_test': True, 'rel_tol': 1e-5, 'abs_tol': 1e-5},
                'resources': {**reference_contract['resources'], 'root': args.remote_root, 'condition_timeout_seconds': 900,
                              'campaign_timeout_seconds': 7200, 'output_limit_gib': 1, 'concurrent_workers': 1},
                'environment': reference_contract['environment'], 'raw_predictions_written': False,
                'time_metric': 'recorded_positive_integer_time_nll', 'tail_definition': 'raw_quantity > final fixed TRAIN quantity boundary',
                'selector': 'earliest minimum full Validation raw quantity RMSE; no reselection',
                'derivation': {'prior_evaluator_path': str((REFERENCE / 'evaluate.py').relative_to(ROOT)),
                               'prior_evaluator_sha256': sha(REFERENCE / 'evaluate.py'),
                               'changes': ['streaming aggregates replace PredictionWriter', 'checkpoint selection/source guards']},
                'deferred_scope': {'dataset': 'intermittent_frozen_5000', 'conditions': 6, 'only_after_all_six_training_complete': True}}
    write('execution_contract.json', contract)
    files = {str((HERE / name).relative_to(ROOT)): sha(HERE / name) for name in ('prepare.py', 'evaluate.py', 'pipeline.py')}
    write('code_seal.json', {'files': files})
    write('preparation_receipt.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'conditions': len(rows),
        'full_population_splits_required': len(rows) * 2, 'new_training': False, 'remote_execution_started': False,
        'registry_sha256': sha(output / 'evaluation_registry.json'), 'contract_sha256': sha(output / 'execution_contract.json'),
        'original_binary_sha_verified': True, 'first_full_validation_minimum_verified': True})
    print(json.dumps({'output': str(output), 'conditions': len(rows), 'new_training': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=['yellow_trip_hourly', 'raf_spare_parts'])
    parser.add_argument('--original-root', action='append', type=Path)
    parser.add_argument('--training-contract', action='append', type=Path, default=[])
    parser.add_argument('--job-ids', nargs='+')
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--dataset-root')
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args())
