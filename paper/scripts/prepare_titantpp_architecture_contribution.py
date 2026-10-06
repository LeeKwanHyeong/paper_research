"""Freeze the explicitly approved 54-fit campaign; local files only, no Test."""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
NAME = 'titantpp_architecture_contribution_dual_20261006_v1'
BUNDLE = PROJECT / 'search_artifacts' / NAME
DESIGN = PROJECT / 'search_artifacts/titantpp_architecture_contribution_contract_20261006_v1'
PARENT = PROJECT / 'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1'
DESIGN_SHA = 'e94093fd3b94bae1ee241dfc33d02c80176fd75d2f3bc999f104bc5d41202f9d'
ARMS = ['P0_H0', 'P1_H0', 'P0_H1', 'P1_H1', 'PF_H1', 'P1_HC']
ADDITIONS = ['models/TPPs/CountAwareTitanArchitectureContribution.py',
             'paper/scripts/run_titantpp_architecture_contribution.py',
             'paper/scripts/qualify_titantpp_architecture_contribution.py',
             'paper/tests/test_titantpp_architecture_contribution_model.py']
USER_INSTRUCTION = '1. 여섯 구조 구현·동작 검증 2. 5080·5090 환경과 실행 예산 검증 3. 세 데이터 × 3seed × 6구조의 54조건 학습. 진행하자. 승인한다'


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(attempt=1):
    assert attempt in (1, 2), 'Unknown explicitly prepared execution attempt'
    assert not (BUNDLE / 'execution_contract.json').exists(), 'Already frozen; no replacement/retry'
    design_validator = load_module(DESIGN / 'control/validate_contract.py', 'frozen_design_validation')
    design_validator.validate_contract(PROJECT, DESIGN)
    design, registry = read(DESIGN / 'design_contract.json'), read(DESIGN / 'source_registry.json')
    assert canonical(design) == DESIGN_SHA
    parent = read(PARENT / 'execution_contract.json')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    # The new implementation must exist in the committed revision before any sync.
    for name in ADDITIONS:
        committed = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=PROJECT)
        assert hashlib.sha256(committed).hexdigest() == sha(PROJECT / name), 'Uncommitted source: ' + name
    source = BUNDLE / 'source'
    source.mkdir(exist_ok=False)
    files = dict(registry['files'])
    original_root = PROJECT / registry['source_root']
    for name, digest in files.items():
        assert sha(original_root / name) == digest, 'Frozen parent changed ' + name
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / name, target)
    for name in ADDITIONS:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
        files[name] = sha(target)
    hosts = deepcopy(parent['hosts'])
    for host, spec in hosts.items():
        root = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + NAME + '_' + host
        if attempt == 2:
            root += '_attempt2'
        spec.update(root=root, source_root=root + '/source', operation_root=root + '/operation',
                    output_dir=root + '/run', tmux=NAME + '_' + host + ('_attempt2' if attempt == 2 else ''))
        spec['environment'].update(SOURCE_REVISION=revision, PYTHONPATH=root + '/source',
            MPLCONFIGDIR=root + '/cache/matplotlib', XDG_CACHE_HOME=root + '/cache')
    datasets = deepcopy(design['datasets'])
    input_files = {}
    for data in datasets:
        dataset = data['dataset_id']
        old_data = next(d for d in parent['datasets'] if d['dataset_id'] == dataset)
        host = design['execution_plan']['proposed_hosts'][dataset]
        for kind in ('data', 'split_manifest'):
            record = old_data['inherited_data_identity'][kind]
            relative = record['path']
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            assert sha(PARENT / 'source' / relative) == record['sha256']
            assert record['sha256'] == data['inherited_data_identity'][kind]['sha256']
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT / 'source' / relative, target)
            input_files[relative] = record['sha256']
            data['inherited_data_identity'][kind]['path'] = hosts[host]['source_root'] + '/' + relative
    # The frozen project resolver recognizes this directory; no dataset is read
    # through it. A file preserves the empty directory in the files-only package.
    sentinel = source / 'sample_data/.frozen_root_sentinel'
    sentinel.parent.mkdir(parents=True, exist_ok=False)
    sentinel.write_bytes(b'')
    input_files['sample_data/.frozen_root_sentinel'] = sha(sentinel)
    bootstrap_code = "from pathlib import Path; from paper.scripts import run_titantpp_architecture_contribution as e; e.install_hooks(); assert e.ROOT == Path.cwd(); print('isolated frozen bootstrap passed')"
    subprocess.run([sys.executable, '-c', bootstrap_code], cwd=source,
        env={**os.environ, 'PYTHONPATH': str(source), 'PYTHONDONTWRITEBYTECODE': '1'}, check=True)
    operations = {str(p.relative_to(BUNDLE)): sha(p) for p in sorted((BUNDLE / 'operation').glob('*.py'))}
    assert {'operation/campaign.py', 'operation/monitor.py', 'operation/qualify.py'} <= set(operations)
    jobs = [{'id': f"{block['dataset']}__{block['seed']}__{arm}", 'host': host,
             'dataset': block['dataset'], 'seed': block['seed'], 'arm': arm}
            for host in hosts for block in design['execution_plan']['paired_blocks']
            if block['proposed_host'] == host for arm in block['ordered_arms']]
    c = {'schema': NAME, 'execution_attempt': attempt, 'design_contract': design, 'design_sha256': DESIGN_SHA,
         'architecture': deepcopy(design['architecture']),
         'objective': design['objective'], 'arms': ARMS, 'seeds': [42, 52, 62],
         'hosts': hosts, 'datasets': datasets, 'dataset_sha256': {d['dataset_id']: canonical(d) for d in datasets},
         'jobs': jobs, 'canonical_conditions': 54, 'new_fits': 54, 'already_terminal_reused': 0,
         'source': {'base_git_revision': revision,
             'parent_base_git_revision': parent['source']['base_git_revision'],
             'git_revision': revision, 'files': files, 'files_sha256': canonical(files),
             'parent_source_closure': registry['closure_sha256'],
             'parent_science117_closure': registry['parent_science117_closure'],
             'provenance': 'Exact frozen123 dependencies plus committed additive implementation; no shared Factory edit'},
         'operation': {'files': operations, 'files_sha256': canonical(operations)},
         'input_files': input_files,
         'training': {'batch_size': 128, 'maximum_epochs': 300, 'minimum_epochs': 40,
             'patience': 40, 'monitor': 'validation_raw_quantity_rmse',
             'tie': 'strict_earliest_finite_minimum', 'warm_start': False},
         'limits': {'total_wall_seconds': 168*3600, 'per_condition_seconds': 36*3600,
             'workers_per_host': 1, 'automatic_retry': False, 'qualification_seconds': 5400,
             'peak_device_fraction_max': .8, 'cpu_rss_max_bytes': 16*1024**3,
             'owned_root_max_bytes': 16*1024**3, 'minimum_free_disk_bytes': 5*1024**3},
         'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
         'authorization': {'current_user_instruction': USER_INSTRUCTION,
             'implementation': True, 'native_qualification': True, 'fresh_54_training': True,
             'new_Test_access': False, 'shared_Runtime_changes': False},
         'operational_path_amendment': 'Only inherited_data_identity data/split_manifest paths remapped to SHA-pinned exclusive source/inputs/data; all scientific dataset fields unchanged',
         'runtime_layout': {'project_root_sentinels': ['models', 'utils', 'sample_data'],
             'sample_data_sentinel_is_empty_not_a_dataset': True},
         'manual_preparation_repair': None if attempt == 1 else {
             'previous_attempt': 'revisions/attempt1', 'reason': 'Missing empty sample_data project-root sentinel before qualification/fit',
             'previous_real_data_optimizer_updates': 0, 'previous_qualification_claim_created': False,
             'claims_and_remote_roots_preserved': True, 'automatic_retry': False},
         'diagnostic': deepcopy(parent.get('diagnostic', {})),
         'reporting': {'mandatory_time_harm': True, 'same_selected_epoch_all_metrics': True,
             'full_Validation_and_fixed_Train_diagnostics_separate': True,
             'independent_evaluation': False, 'historical_Test_exposure': True},
         'deferred_datasets': ['instacart'],
         'policy': {'automatic_retry': False, 'single_worker_per_host': True,
             'originals_claims_unfavorable_results_preserved': True}}
    operation = load_module(BUNDLE / 'operation/campaign.py', 'new_operation_validation')
    operation.validate(c, verify_files=False)
    digest = canonical(c)
    now = time.time()
    approval = {'schema': 'titantpp_architecture_contribution_execution_approval_v1',
        'approved': True, 'contract_sha256': digest, 'hosts': ['5080', '5090'],
        'user_instruction': USER_INSTRUCTION, 'objective_reply': '수량 우위가 주목표, 시간 손해는 필수 보고',
        'recorded_unix': now, 'local_implementation_native_qualification_and_fresh_54_fit_training': True,
        'Test_access': False, 'shared_Runtime_or_auth_change': False,
        'previous_design_approval_preserved': str(DESIGN.relative_to(PROJECT) / 'approval.json')}
    lease_started = now if attempt == 1 else read(BUNDLE / 'revisions/attempt1/start_permit.json')['started_at_unix']
    start = {'schema': 'titantpp_architecture_contribution_start_v1', 'contract_sha256': digest,
        'approval_sha256': canonical(approval), 'started_at_unix': lease_started,
        'deadline_unix': lease_started + 168*3600, 'automatic_retry': False, 'workers_per_host': 1}
    for name, value in [('execution_contract.json', c), ('approval.json', approval), ('start_permit.json', start)]:
        write(BUNDLE / name, value)
    write(BUNDLE / 'current.json', {'schema': NAME, 'contract_path': 'execution_contract.json',
        'canonical_sha256': digest, 'source_closure_sha256': c['source']['files_sha256'],
        'operation_closure_sha256': c['operation']['files_sha256'],
        'hosts': {host: spec['root'] for host, spec in hosts.items()},
        'status': 'frozen_awaiting_native_qualification', 'training_started': False,
        'actual_deadline_KST': dt.datetime.fromtimestamp(start['deadline_unix'], dt.timezone(dt.timedelta(hours=9))).isoformat()})
    write(BUNDLE / 'source_registry.json', c['source'])
    write(BUNDLE / 'operations_manifest.json', c['operation'])
    for host in hosts:
        staging = BUNDLE / 'deployment' / ('package_' + host)
        staging.mkdir(parents=True, exist_ok=False)
        package_files = {}
        for name in ['execution_contract.json', 'approval.json', 'start_permit.json']:
            shutil.copyfile(BUNDLE / name, staging / name)
        for prefix in ('source', 'operation'):
            names = files if prefix == 'source' else operations
            for name in names:
                relative = str(Path(prefix) / name) if prefix == 'source' else name
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(BUNDLE / relative, target)
        for name in input_files:
            target = staging / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        package_files = {str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        write(staging / 'package_manifest.json', {'contract_sha256': digest, 'host': host, 'files': package_files})
        archive = BUNDLE / 'deployment' / ('package_' + host + '.tar.gz')
        with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for p in sorted(staging.rglob('*')):
                if p.is_file():
                    tar.add(p, arcname=str(p.relative_to(staging)), recursive=False)
        write(BUNDLE / 'deployment' / ('package_receipt_' + host + '.json'),
              {'host': host, 'archive_sha256': sha(archive), 'file_count': len(package_files),
               'contract_sha256': digest, 'no_Test_metrics_or_predictions_packaged': True})
    print(json.dumps({'status': 'frozen', 'contract_sha256': digest, 'source_file_count': len(files),
        'source_closure': c['source']['files_sha256'], 'operation_closure': c['operation']['files_sha256'],
        'new_fits': len(jobs), 'deadline_unix': start['deadline_unix']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    prepare(args.attempt)
