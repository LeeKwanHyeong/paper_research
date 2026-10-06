"""Seal nine completed quantity-selected CNN+GRU54 checkpoints; no inference."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'search_artifacts/titantpp_cnn_gru54_test_3seed_20261006_v1'
ORIGINAL = ROOT / 'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1'
PRIOR = ROOT / 'search_artifacts/titantpp_cnn_gru_test_20261005_v1'
COMPARISON = ROOT / 'reports/titantpp_cnn_gru54_final_validation_comparison_20261006_v1'
TRAIN_SHA = '845a891db284f2cf88aa2ab17bee47716ea8dc11e6268646a71c772073ab7130'
SOURCE_SHA = '4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def copy_verified(source, target, digest=None):
    if digest is not None:
        assert sha(source) == digest, source
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    shutil.copyfile(source, target)
    assert sha(target) == sha(source)


def main():
    assert not (BUNDLE / 'execution_contract.json').exists(), 'Sealed attempt preserved; no retry'
    assert (BUNDLE / 'operation/evaluate_titantpp_cnn_gru_frozen_test.py').is_file(), 'Reviewed evaluator must exist before sealing'
    assert (COMPARISON / 'comparison_receipt.json').is_file(), 'Final Validation comparison must be recorded before sealing'
    training = read(ORIGINAL / 'execution_contract.json')
    done = read(ORIGINAL / 'analysis/completion_receipt.json')
    assert canonical(training) == TRAIN_SHA == done['contract_sha256']
    assert done['status'] == 'nine_conditions_Validation_originals_complete'
    assert done['canonical_conditions'] == 9 and done['new_fits'] == 6 and done['reused'] == 3
    for relative, digest in done['outputs'].items():
        assert sha(ORIGINAL / 'analysis' / relative) == digest
    BUNDLE.mkdir(parents=True, exist_ok=True)
    approval = {'approved': True, 'test_inference_authorized': True,
        'user_request': '오케이 그러면 다음 작업 진행하자',
        'preceding_authorization': '진행하고, Test까지 전부 뽑아서 정렬하고 어떤게 가장 좋은지 비교해보자.',
        'scope': 'Original CNN+GRU54 nine quantity-selected checkpoints, frozen Validation then previously exposed Test3seed',
        'conditions': 9, 'seeds': [42, 52, 62], 'models': ['titantpp_cnn_gru54'],
        'new_training': False, 'checkpoint_reselection': False, 'head_refits_included': False,
        'raw_predictions_written': False, 'new_rental': False, 'shared_runtime_changed': False,
        'automatic_retry': False, 'created_unix': time.time()}
    write(BUNDLE / 'approval.json', approval)
    old_contract = read(PRIOR / 'execution_contract.json')
    accepted = []
    for entry in old_contract['accepted_training_contracts']:
        copy_verified(PRIOR / entry['path'], BUNDLE / entry['path'])
        assert canonical(read(BUNDLE / entry['path'])) == entry['canonical_sha256']
        accepted.append(entry)
    copy_verified(ORIGINAL / 'execution_contract.json', BUNDLE / 'original_3seed_training_contract.json')
    accepted.append({'path': 'original_3seed_training_contract.json', 'canonical_sha256': TRAIN_SHA})
    old_registry = read(PRIOR / 'evaluation_registry.json')
    old_bundle = old_registry['bundles']['cnn_gru']
    bundles = {'cnn_gru_117': {**old_bundle, 'source_root': 'source117'},
        'cnn_gru_123': {'source_root': 'source123', 'source_files': training['source']['files'],
            'source_closure_sha256': SOURCE_SHA, 'datasets': training['datasets']}}
    for key, bundle in bundles.items():
        origin = PRIOR / 'source' if key == 'cnn_gru_117' else ORIGINAL / 'source'
        assert canonical(bundle['source_files']) == bundle['source_closure_sha256']
        for relative, digest in bundle['source_files'].items():
            copy_verified(origin / relative, BUNDLE / bundle['source_root'] / relative, digest)
        marker = BUNDLE / bundle['source_root'] / 'sample_data/.keep'
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
    rows, references, bindings = [], {}, []
    origins = [(r['job'], PRIOR / 'original/run' / r['job']['id'], 'cnn_gru_117', r['source_revision'])
        for r in old_registry['rows'] if r['model'] == 'titantpp_cnn_gru54']
    origins += [(j, ORIGINAL / 'retrieved' / j['host'] / 'original/run' / j['id'],
        'cnn_gru_123', training['source']['base_git_revision']) for j in training['jobs']]
    for job, folder, source_key, revision in origins:
        terminal = read(folder / 'terminal_manifest.json')
        assert terminal['status'] == 'complete' and terminal['scientific_success'] and terminal['job'] == job
        assert terminal['contract_sha256'] in {a['canonical_sha256'] for a in accepted}
        target = BUNDLE / 'original/run' / job['id']
        selected_files = {}
        for relative, digest in terminal['files'].items():
            original = (folder / relative).resolve()
            assert original.is_relative_to(folder.resolve())
            copy_verified(original, target / relative, digest)
            selected_files[original.name] = target / relative
        copy_verified(folder / 'terminal_manifest.json', target / 'terminal_manifest.json')
        ep = read(selected_files['endpoint_replays.json'])
        history = read(selected_files['history.json'])['history']
        assert [r['epoch'] for r in history] == list(range(1, len(history) + 1))
        best = min(history, key=lambda r: r['val_qty_rmse'])
        assert ep['best_epoch'] == best['epoch'] and ep['completed_epochs'] == len(history)
        assert ep['evaluation_scope'] == 'validation_only' and ep['held_out_test_evaluated'] is False
        cp = selected_files['best_val_qty_rmse_model.pt']
        binding = {'terminal_manifest_path': str((target / 'terminal_manifest.json').relative_to(BUNDLE)),
            'terminal_manifest_sha256': sha(target / 'terminal_manifest.json'),
            'training_contract_sha256': terminal['contract_sha256'],
            'history_path': str(selected_files['history.json'].relative_to(BUNDLE)),
            'history_sha256': sha(selected_files['history.json']),
            'endpoint_path': str(selected_files['endpoint_replays.json'].relative_to(BUNDLE)),
            'endpoint_sha256': sha(selected_files['endpoint_replays.json'])}
        row = {'dataset': job['dataset'], 'model': job['arm'], 'seed': job['seed'], 'job': job,
            'endpoint': 'selected', 'evaluator_source_bundle': source_key,
            'checkpoint_path': str(cp.relative_to(BUNDLE)), 'checkpoint_file_sha256': sha(cp),
            'state_tensor_sha256': ep['selected']['state_sha256'], 'selected_epoch': best['epoch'],
            'selected_metric_value': best['val_qty_rmse'], 'initial_state_sha256': ep['initial_state_sha256'],
            'source_revision': revision, 'validation_count': ep['selected']['count'], 'original_binding': binding}
        rows.append(row)
        references[f"{row['dataset']}__{row['model']}__seed{row['seed']}"] = ep['selected']
        bindings.append({'job_id': job['id'], 'source_original': str(folder.relative_to(ROOT)),
            'checkpoint_sha256': sha(cp), 'selected_epoch': best['epoch'], 'selection_changed': False})
    assert len(rows) == len({r['job']['id'] for r in rows}) == 9
    rows.sort(key=lambda r: (r['dataset'], r['seed']))
    write(BUNDLE / 'evaluation_registry.json', {'rows': rows, 'bundles': bundles})
    manifest = copy.deepcopy(read(PRIOR / 'dataset_manifest.json'))
    for data in manifest['datasets']:
        for identity, field in ((data, 'path'), (data['split_manifest'], 'path')):
            source = PRIOR / identity[field]
            target = BUNDLE / 'data' / data['dataset'] / source.name
            copy_verified(source, target, identity['sha256'])
            identity[field] = str(target.relative_to(BUNDLE))
    write(BUNDLE / 'dataset_manifest.json', manifest)
    write(BUNDLE / 'selected_binding.json', {'rows': bindings, 'selection_unchanged': True})
    # Preserve prior exposure metadata without reading any Test performance.
    write(BUNDLE / 'evaluation_lineage.json', {'Test_previously_accessed': True,
        'independent_untouched_Test': False, 'seed42_checkpoint_Test_previously_evaluated': True,
        'seed52_62_new_checkpoint_Test_inference': True, 'fresh_replay_splits': ['validation', 'test'],
        'prior_registry_sha256': sha(PRIOR / 'evaluation_registry.json'),
        'original_reuse_registry_sha256': sha(ORIGINAL / 'reuse_seed42_registry.json'),
        'head_refits_Test_evaluated': False})
    comparison_receipt = read(COMPARISON / 'comparison_receipt.json')
    assert comparison_receipt['status'] == 'complete' and comparison_receipt['candidate_conditions'] == 9
    assert comparison_receipt['frozen_contract_sha256'] == TRAIN_SHA
    assert all(comparison_receipt[k] is False for k in ('held_out_performance_read', 'mixed_performance_file_read', 'raw_prediction_read'))
    for relative, digest in comparison_receipt['outputs'].items():
        assert sha(COMPARISON / relative) == digest, relative
    gate = {'passed': True, 'evaluation_scope': 'validation_only',
        'held_out_test_evaluated': False, 'conditions': 9,
        'comparison_receipt_path': 'final_validation_comparison_receipt.json',
        'comparison_receipt_sha256': sha(COMPARISON / 'comparison_receipt.json'),
        'scientific_adoption_interpretation': 'Integrity/replay permission does not imply scientific superiority; frozen strong-dominance rule remains separate.',
        'rows': [{k: r[k] for k in ('dataset', 'model', 'seed', 'checkpoint_file_sha256',
            'state_tensor_sha256', 'selected_epoch')} for r in rows]}
    copy_verified(COMPARISON / 'comparison_receipt.json', BUNDLE / 'final_validation_comparison_receipt.json')
    write(BUNDLE / 'comparison_gate.json', gate)
    operation = BUNDLE / 'operation'
    operation.mkdir(exist_ok=True)
    (operation / 'eval_supervisor.py').write_text('''"""Own one bounded frozen evaluation and preserve its real exit code."""
from pathlib import Path
import hashlib,json,subprocess,time
root=Path(__file__).resolve().parents[1]
c=json.loads((root/'execution_contract.json').read_text())
command=['timeout','--signal=TERM','--kill-after=15s',str(c['resources']['campaign_timeout_seconds']),c['resources']['python'],str(root/'operation/evaluate_titantpp_cnn_gru_frozen_test.py'),'--campaign',str(root),'--mode','pipeline','--device','cuda']
started=time.time()
with (root/'supervisor.claim').open('x') as f:json.dump({'started_unix':started,'command':command,'automatic_retry':False},f)
process=subprocess.Popen(command)
code=process.wait()
with (root/'supervisor_process_exit.json').open('x') as f:json.dump({'returncode':code,'command':command,'started_unix':started,'completed_unix':time.time(),'contract_sha256':hashlib.sha256((root/'execution_contract.json').read_bytes()).hexdigest(),'automatic_retry':False},f,indent=2)
raise SystemExit(code)
''')
    copy_verified(PRIOR / 'operation/prior_evaluate.py', operation / 'prior_evaluate.py', old_contract['evaluator_sha256'])
    deploy = (ROOT / 'paper/scripts/deploy_titantpp_cnn_gru_frozen_test.py').read_text()
    deploy = deploy.replace("titantpp_cnn_gru_test_20261005_v1", "titantpp_cnn_gru54_test_3seed_20261006_v1")
    deploy = deploy.replace("shell='exec '+shlex.join(command)", "shell='exec '+shlex.join([c['resources']['python'],str(root/'operation/eval_supervisor.py')])")
    (operation / 'deploy_batch.py').write_text(deploy)
    retrieve = (ROOT / 'paper/scripts/retrieve_titantpp_cnn_gru_test_artifacts.py').read_text()
    retrieve = retrieve.replace("'campaign.lock','worker.lock'", "'campaign.lock','worker.lock','supervisor.claim','supervisor_process_exit.json'")
    (operation / 'retrieve_batch.py').write_text(retrieve)
    c = copy.deepcopy(old_contract)
    c.update(campaign_kind='titantpp_cnn_gru54_frozen_test_3seed_v1',
        source_closure_sha256=SOURCE_SHA, original_3seed_training_contract_sha256=TRAIN_SHA,
        registry_sha256=sha(BUNDLE / 'evaluation_registry.json'),
        dataset_manifest_sha256=sha(BUNDLE / 'dataset_manifest.json'),
        operation_adapter_sha256=sha(operation / 'evaluate_titantpp_cnn_gru_frozen_test.py'),
        accepted_training_contracts=accepted, validation_references=references,
        evaluation_lineage_sha256=sha(BUNDLE / 'evaluation_lineage.json'),
        test_exposure_lineage=read(BUNDLE / 'evaluation_lineage.json'),
        final_validation_comparison_gate={'path': 'comparison_gate.json', 'sha256': sha(BUNDLE / 'comparison_gate.json')},
        original_training_completion_sha256=sha(ORIGINAL / 'analysis/completion_receipt.json'),
        evaluation_interpretation='Previously exposed frozen Test3seed; heterogeneous reused seed42; no independent untouched evaluation')
    c['approval'].update(models=['titantpp_cnn_gru54'], seeds=[42, 52, 62], source_approval_sha256=sha(BUNDLE / 'approval.json'))
    c['resources']['root'] = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru54_test_3seed_20261006_v1_5080'
    write(BUNDLE / 'execution_contract.json', c)
    write(BUNDLE / 'current.json', {'status': 'sealed_not_launched', 'contract': 'execution_contract.json',
        'contract_sha256': sha(BUNDLE / 'execution_contract.json'), 'canonical_sha256': canonical(c),
        'checkpoint_conditions': 9, 'new_training': False, 'host': '5080', 'remote_root': c['resources']['root']})
    files = {str(p.relative_to(BUNDLE)): sha(p) for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(BUNDLE / 'deployment_manifest.json', {'files': files, 'bytes': sum((BUNDLE/p).stat().st_size for p in files),
        'remote_root': c['resources']['root'], 'new_training': False, 'new_rental': False, 'created_unix': time.time()})
    print(json.dumps({'status': 'sealed', 'conditions': len(rows), 'file_count': len(files),
        'bytes': sum((BUNDLE/p).stat().st_size for p in files), 'test_started': False}))


if __name__ == '__main__':
    main()
