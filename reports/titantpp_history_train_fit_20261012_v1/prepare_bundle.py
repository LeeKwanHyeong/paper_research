"""Prepare only the approved frozen history27 Train fit-state evaluation."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import time

PROJECT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
PARENT = PROJECT / 'search_artifacts/titantpp_history_diagnostics_20261012_v2'
BUNDLE = PROJECT / 'search_artifacts/titantpp_history_train_fit_20261012_v1'
REPORT = PROJECT / 'reports/titantpp_history_train_fit_20261012_v1'
PARENT_CANONICAL = '70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d'
PARENT_RAW = '08ad2039571fed876166f28d23c633e7dcc8cb7e8a8246650f52e928c2fd735a'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    assert sha(PARENT / 'contract.json') == PARENT_RAW
    parent = json.loads((PARENT / 'contract.json').read_bytes())
    assert parent['contract_sha256'] == PARENT_CANONICAL
    assert canonical({k:v for k,v in parent.items() if k != 'contract_sha256'}) == PARENT_CANONICAL
    assert sha(PARENT / 'analysis/completion_receipt.json') == 'f689243b21dcd556af81b33117f994191bc6b2d0a139634153f187172560f5ea'
    assert sha(PARENT / 'originals/manifest.json') == 'e2dd3217e2657d9fbabc9589c1d53bd9dad0707f2eeb8d7af1763a0248e65588'
    verified = json.loads((PARENT / 'originals/LOCAL_VERIFICATION.json').read_bytes())
    assert verified['status'] == 'verified'
    assert not BUNDLE.exists(), 'Existing fit-state bundle: no automatic overwrite/retry'
    c = copy.deepcopy(parent)
    c.pop('contract_sha256')
    c.pop('repair', None)
    c.update(schema='titantpp_history_train_fit_20261012_v1', created_unix=time.time(),
        evaluation_scope='selected_checkpoint_Train_fit_state_plus_reused_Validation_Test',
        parent_contract_sha256=PARENT_CANONICAL, parent_contract_raw_sha256=PARENT_RAW,
        instrumentation_revision='Train-only adapter; prior diagnostic source preserved')
    c['authorization'] = {
        'human_request': '이거 먼저 진행하자',
        'approved_task': '동일 선택 checkpoint의 Train/Validation/Test 적합 상태 확인 및 수량/시간 원인별 기존 증적 분석·최소 후속 계약',
        'train_inference_authorized': True, 'existing_Test_access_authorized': True,
        'training': False, 'validation_test_inference': False,
        'checkpoint_or_hyperparameter_reselection': False, 'new_dataset': False,
        'new_retrieval_training': False, 'route_bypass_interventions': False,
        'shared_Runtime_or_auth_changes': False, 'manuscript_edit': False,
    }
    c['execution'].update(
        remote_root='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_train_fit_20261012_v1_5080',
        gate='native_synthetic3_qualification_before_Train27; existing_Validation_Test_reused_only')
    c['diagnostics']['expected_split_evaluations'] = 27
    c['diagnostics']['split_scope'] = ['train']
    c['train_fit'] = {
        'expected_train_populations': {ds: spec['definition']['inherited_data_identity']['populations']['train']
            for ds, spec in c['datasets'].items()},
        'evaluation_splits': ['train'], 'new_fits': 0, 'validation_test_inference': 0,
        'model_mode': 'eval', 'gradients': False, 'shuffle': False, 'drop_last': False,
        'first_event_not_a_target': True, 'reference_train_metrics_available': False,
        'train_loss_logs_are_selected_checkpoint_raw_metrics': False,
    }
    references = {}
    for job in c['jobs']:
        references[job['id']] = {}
        for split, name in [('validation', 'validation_replay.json'), ('test', 'test_metrics.json')]:
            rel = f'results/{job["id"]}/{name}'
            value = json.loads((PARENT / rel).read_bytes())
            assert value['status'] == 'complete' and value['job_id'] == job['id']
            assert value['checkpoint_sha256'] == job['checkpoint']['sha256']
            assert value['state_sha256'] == value['state_after_sha256'] == job['checkpoint_state_sha256']
            telemetry_rel = value['telemetry_path']
            assert sha(PARENT / telemetry_rel) == value['telemetry_sha256']
            references[job['id']][split] = {'relative_path': rel, 'sha256': sha(PARENT / rel),
                'telemetry_path': telemetry_rel, 'telemetry_sha256': value['telemetry_sha256']}
    c['reused_diagnostics'] = {
        'local_bundle_name': PARENT.name, 'contract_sha256': PARENT_CANONICAL,
        'contract_raw_sha256': PARENT_RAW,
        'manifest_sha256': sha(PARENT / 'originals/manifest.json'),
        'archive_sha256': '4efb0453b5590eac18e5bcbfa29062970c9c5c291f8672cc14ad32ddce24c3f7',
        'analysis_completion_sha256': sha(PARENT / 'analysis/completion_receipt.json'),
        'reference_receipts': references, 'new_validation_evaluations': 0, 'new_test_evaluations': 0,
    }
    c['contract_sha256'] = canonical(c)
    inputs = json.loads((PARENT / 'input_integrity.json').read_bytes())['files']
    copied = {}
    for rel, expected in inputs.items():
        if rel == 'contract.json':
            continue
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        src, dest = PARENT / rel, BUNDLE / rel
        assert src.is_file() and not src.is_symlink() and sha(src) == expected, rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        assert sha(dest) == expected
        copied[rel] = expected
    write(BUNDLE / 'contract.json', c)
    copied['contract.json'] = sha(BUNDLE / 'contract.json')
    write(BUNDLE / 'input_integrity.json', {'schema': 'frozen_train_fit_inputs_v1',
        'contract_sha256': c['contract_sha256'], 'files': copied})
    write(REPORT / 'APPROVAL.json', {'human_request': '이거 먼저 진행하자', 'authorized': True,
        'authorized_scope': c['authorization'], 'contract_sha256': c['contract_sha256'],
        'observed_unix': time.time(), 'new_fits': 0, 'new_Train_evaluations': 27,
        'reused_Validation_evaluations': 27, 'reused_Test_evaluations': 27})
    print(json.dumps({'status': 'prepared', 'contract_sha256': c['contract_sha256'],
        'input_files': len(copied), 'train_populations': c['train_fit']['expected_train_populations']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
