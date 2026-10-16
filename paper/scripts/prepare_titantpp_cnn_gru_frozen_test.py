"""Seal the explicitly approved nine-condition Test batch locally; no inference."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
RESUME = ROOT / 'search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1'
OUT = ROOT / 'search_artifacts/titantpp_cnn_gru_test_20261005_v1'
ORIGINAL = RESUME / 'recovery/ssh_info_attempt2/original'
REFERENCE = ROOT / 'reports/titantpp_completed_validation_test_20261004_v1/width16'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not (OUT / 'execution_contract.json').exists(), 'Batch is already sealed'
    approval = read(OUT / 'approval.json')
    assert approval['approved'] is True and approval['conditions'] == 9 and not approval['checkpoint_reselection']
    parent = read(RESUME / 'parent_execution_contract.json')
    assert canonical(parent) == approval['parent_contract_sha256']
    assert len(parent['source']['files']) == 117 and canonical(parent['source']['files']) == approval['source_closure_sha256']
    accepted = []
    for source_name, name in [('parent_execution_contract.json', 'parent_execution_contract.json'),
                              ('execution_contract.json', 'resume_execution_contract.json'),
                              ('execution_contract_5090.json', 'resume_execution_contract_5090.json')]:
        shutil.copyfile(RESUME / source_name, OUT / name)
        accepted.append({'path': name, 'canonical_sha256': canonical(read(OUT / name))})
    for relative, digest in parent['source']['files'].items():
        source, target = ORIGINAL / 'source' / relative, OUT / 'source' / relative
        assert sha(source) == digest, relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    marker = OUT / 'source/sample_data/.keep'
    marker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ORIGINAL / 'source/sample_data/.keep', marker)
    rows, refs, bindings = [], {}, []
    for job in parent['jobs']:
        folder = ORIGINAL / 'run' / job['id']
        for host in ('5080', '5090'):
            recovered = RESUME / 'retrieved' / host / 'run' / job['id']
            if recovered.exists():
                folder = recovered
        terminal_path = folder / 'terminal_manifest.json'
        terminal = read(terminal_path)
        assert terminal['scientific_success'] and terminal['status'] == 'complete' and terminal['job']['id'] == job['id']
        assert terminal['contract_sha256'] in {r['canonical_sha256'] for r in accepted}
        target_folder = OUT / 'original/run' / job['id']
        target_folder.mkdir(parents=True)
        selected_files = {}
        for relative, digest in terminal['files'].items():
            path = (folder / relative).resolve()
            assert path.is_relative_to(folder.resolve()) and sha(path) == digest
            target = target_folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            selected_files[Path(relative).name] = target
        shutil.copyfile(terminal_path, target_folder / 'terminal_manifest.json')
        endpoint = read(selected_files['endpoint_replays.json'])
        history = read(selected_files['history.json'])['history']
        first = min(history, key=lambda r: (r['val_qty_rmse'], r['epoch']))
        selected = endpoint['selected']
        assert endpoint['status'] == 'complete' and not endpoint['held_out_test_evaluated']
        assert first['epoch'] == endpoint['best_epoch'] and endpoint['completed_epochs'] == len(history)
        binding = {'terminal_manifest_path': str((target_folder / 'terminal_manifest.json').relative_to(OUT)),
                   'terminal_manifest_sha256': sha(terminal_path), 'training_contract_sha256': terminal['contract_sha256'],
                   'history_path': str(selected_files['history.json'].relative_to(OUT)), 'history_sha256': sha(selected_files['history.json']),
                   'endpoint_path': str(selected_files['endpoint_replays.json'].relative_to(OUT)), 'endpoint_sha256': sha(selected_files['endpoint_replays.json'])}
        cp = selected_files['best_val_qty_rmse_model.pt']
        row = {'dataset': job['dataset'], 'model': job['arm'], 'seed': 42, 'job': terminal['job'], 'endpoint': 'selected',
               'evaluator_source_bundle': 'cnn_gru', 'checkpoint_path': str(cp.relative_to(OUT)), 'checkpoint_file_sha256': sha(cp),
               'state_tensor_sha256': selected['state_sha256'], 'selected_epoch': first['epoch'], 'selected_metric_value': first['val_qty_rmse'],
               'initial_state_sha256': endpoint['initial_state_sha256'], 'source_revision': parent['source']['base_git_revision'],
               'validation_count': selected['count'], 'original_binding': binding}
        rows.append(row)
        refs[f"{row['dataset']}__{row['model']}__seed42"] = selected
        bindings.append({'job': terminal['job'], 'original_location': str(folder.relative_to(ROOT)), **binding,
                         'checkpoint_sha256': sha(cp), 'selection_unchanged': True, 'first_minimum_epoch': first['epoch'],
                         'completed_epochs': endpoint['completed_epochs'], 'parameter_count': endpoint['parameter_count']})
    assert len(rows) == 9 and len({r['job']['id'] for r in rows}) == 9
    registry = {'rows': rows, 'bundles': {'cnn_gru': {'source_root': 'source', 'source_files': parent['source']['files'],
        'source_closure_sha256': parent['source']['files_sha256'], 'datasets': parent['datasets']}}}
    manifest = read(REFERENCE / 'dataset_manifest.json')
    manifest['datasets'] = [d for d in manifest['datasets'] if d['dataset'] in approval['datasets']]
    for data in manifest['datasets']:
        for identity, field in [(data, 'path'), (data['split_manifest'], 'path')]:
            source = ROOT / identity[field]
            assert sha(source) == identity['sha256'], source
            target = OUT / 'data' / data['dataset'] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            identity[field] = str(target.relative_to(OUT))
    write('evaluation_registry.json', registry)
    write('dataset_manifest.json', manifest)
    write('selected_binding.json', {'selection_unchanged': True, 'rows': bindings})
    (OUT / 'operation').mkdir()
    prior = ROOT / 'reports/titantpp_width8_12_test_20261004_v1/evaluate.py'
    shutil.copyfile(prior, OUT / 'operation/prior_evaluate.py')
    adapter = ROOT / 'paper/scripts/evaluate_titantpp_cnn_gru_frozen_test.py'
    shutil.copyfile(adapter, OUT / 'operation/evaluate_titantpp_cnn_gru_frozen_test.py')
    contract = read('/private/tmp/titantpp_cnn_gru_frozen_test_contract_example.json')
    contract = {k:v for k,v in contract.items() if not k.startswith('example_')}
    contract['approval']['source_approval_sha256'] = sha(OUT / 'approval.json')
    contract.update(registry_sha256=sha(OUT / 'evaluation_registry.json'), dataset_manifest_sha256=sha(OUT / 'dataset_manifest.json'),
        dataset_sha256={d['dataset']: d['sha256'] for d in manifest['datasets']}, operation_adapter_sha256=sha(adapter),
        accepted_training_contracts=accepted, validation_references=refs,
        population_references={d['dataset']:read(REFERENCE / 'execution_contract.json')['population_references'][d['dataset']] for d in manifest['datasets']})
    assert contract['prior_evaluator']['sha256'] == sha(prior)
    contract['runtime_expected'].update(torch='2.11.0+cu130', polars='1.39.3', numpy='2.4.4', cuda='13.0')
    contract['environment'] = read(REFERENCE / 'execution_contract.json')['environment']
    contract['environment'].update(POLARS_MAX_THREADS='4', SOURCE_REVISION=parent['source']['base_git_revision'])
    write('execution_contract.json', contract)
    files = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    write('deployment_manifest.json', {'files': files, 'bytes':sum((OUT/p).stat().st_size for p in files),
        'remote_root':contract['resources']['root'], 'new_training':False, 'new_rental':False, 'created_unix':time.time()})
    print(json.dumps({'conditions':len(rows),'file_count':len(files),'bytes':sum((OUT/p).stat().st_size for p in files), 'test_started':False}))


if __name__ == '__main__':
    main()
