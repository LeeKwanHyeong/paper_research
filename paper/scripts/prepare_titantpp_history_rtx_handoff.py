"""Prepare the approved twelve fresh RTX fits from sealed history129 source.

Stdlib only. Preparation never calls SSH, a GPU API, or a trainer. The prepared
contract still needs an A100 queue release and native qualification/admission.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = 'titantpp_contribution_history_rtx_handoff_20261010_v1'
TARGET = PROJECT / 'search_artifacts' / SCHEMA
PARENT = PROJECT / 'search_artifacts/titantpp_contribution_history_recovery_20261010_v1'
RTX = PROJECT / 'search_artifacts/titantpp_contribution_history_20261008_v1'
PARALLEL = PROJECT / 'search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1'
PARENT_SHA = 'da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899'
RTX_SHA = '7594a484d29dbe5efb633d2e78d08734d5c24661761639874ed64e77b2c11f92'
PARALLEL_SHA = '4fcf1923c2bfbec8ad7b4bf7af506ee4f9b254dd8695e6b77035dd913b5bc12a'
SOURCE_SHA = 'c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a'
SCIENCE_REVISION = '57da14a2a7a5687e1bdef7d3164f3e8c0051da59'
ASSIGNMENTS = {'5080': [['raf_spare_parts', 62]],
               '5090': [['intermittent_frozen_5000', s] for s in (42, 52, 62)]}
SCIENCE_FIELDS = ('stage', 'arms', 'seeds', 'runner_modules', 'datasets',
    'dataset_sha256', 'source', 'input_files', 'training', 'architecture',
    'design_sha256', 'diagnostic', 'measurement_arms', 'cost_limits',
    'evaluation_scope', 'held_out_test_evaluated', 'claim_contract_v2_rawSHA', 'policy')
USER_INSTRUCTION = ('이력 보정27 중 아직 시작하지 않은12개를5080/5090에 나누어 실행. '
    '5080 RAF seed62 세 대조군,5090 Intermittent seed42/52/62 세 대조군씩. '
    '진행 중 A100 RAF seed52 세 조건은 유지하며 중복 실행 없이 인계한다. '
    'ㅇㅇ 그렇게 진행하고 해당 실험들이 전부 종료된 이후 논문 초안 작업 기준으로 '
    '다음 작업 순서도 재정렬해두자.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe relative path')
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Path escape')
    return path


def write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode() if isinstance(content, str) else content
    if path.exists():
        require(path.read_bytes() == payload, 'Preserved existing file differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def write_json(path, value):
    write_new(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def function(text, name):
    node = next(n for n in ast.parse(text).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return ast.get_source_segment(text, node)


def replace_node(text, name, replacement):
    node = next(n for n in ast.parse(text).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])


def scientific_fingerprint(contract):
    return canonical({key: contract[key] for key in SCIENCE_FIELDS})


def expected_jobs(arms):
    return [{'id': f'{d}__{seed}__{arm}', 'host': host, 'dataset': d,
        'seed': seed, 'arm': arm}
        for host, groups in ASSIGNMENTS.items() for d, seed in groups for arm in arms]


def verify_frozen(bundle, expected):
    contract = read(bundle / 'execution_contract.json')
    require(canonical(contract) == expected, 'Frozen contract changed: ' + str(bundle))
    require(contract['source']['files_sha256'] == SOURCE_SHA
        and canonical(contract['source']['files']) == SOURCE_SHA, 'Scientific source closure drift')
    require(contract['source']['git_revision'] == SCIENCE_REVISION, 'Frozen science revision drift')
    require(len(contract['source']['files']) == 129 and len(contract['input_files']) == 7,
        'Frozen source/input file count changed')
    for name, digest in {**contract['source']['files'], **contract['input_files']}.items():
        require(sha(safe(bundle / 'source', name)) == digest, 'Frozen source/input drift: ' + name)
    require(canonical(contract['operation']['files']) == contract['operation']['files_sha256'],
        'Frozen operation closure drift')
    for name, digest in contract['operation']['files'].items():
        require(sha(safe(bundle, name)) == digest, 'Frozen operation drift: ' + name)
    return contract


VALIDATE = '''
def validate(c,*,source_root=None,operation_root=None,verify_files=True):
    require(c['stage']=='history' and c['schema']==HANDOFF_SCHEMA,'Foreign RTX handoff')
    require(sha_json({k:c[k] for k in SCIENCE_FIELDS})==SCIENCE_FINGERPRINT,'Scientific identity drift')
    require(c['assignments']==HANDOFF_ASSIGNMENTS and set(c['hosts'])=={'5080','5090'},'RTX host scope drift')
    require(c['jobs']==expected_jobs(c) and len(c['jobs'])==12 and len({j['id'] for j in c['jobs']})==12,'Twelve fresh identities drift')
    require(c['parallelism']=={'maximum_workers':3,'grouping':'same_dataset_seed_three_arms','per_process_cuda_fraction':.30,'aggregate_gpu_fraction':.8},'Parallel operational scope drift')
    require(c['limits']['total_wall_seconds']==168*3600 and c['limits']['per_condition_seconds']==36*3600
        and c['limits']['workers_per_host']==3 and not c['limits']['automatic_retry'],'Budget drift')
    require(c['source']['files_sha256']==SOURCE_CLOSURE and sha_json(c['source']['files'])==SOURCE_CLOSURE,'Source closure drift')
    require(sha_json(c['operation']['files'])==c['operation']['files_sha256'],'Operation closure drift')
    require(c['handoff']['parent_contract_sha256']==A100_PARENT_SHA and c['handoff']['fresh_fits']==12
        and c['handoff']['original_execution_ids_preserved'] is True and c['handoff']['automatic_retry'] is False,'Handoff scope drift')
    for host,spec in c['hosts'].items():
        require(spec==PINNED_HOSTS[host],'Pinned RTX runtime/root/GPU drift')
    if verify_files:
        sr=Path(source_root or ROOT);op=Path(operation_root or Path(__file__).resolve().parent)
        require(os.environ['SOURCE_REVISION']==c['source']['git_revision'],'Source revision environment drift')
        for p,v in {**c['source']['files'],**c['input_files']}.items():
            require(sha_file(safe_relative(sr,p))==v,'Source/input drift '+p)
        for p,v in c['operation']['files'].items():
            require(sha_file(safe_relative(op.parent,p))==v,'Operation drift '+p)
    return c
'''

RELEASE = '''
def verify_ownership_release(c,root):
    release=read(root/'ownership_release.json')
    require(release['contract_sha256']==sha_json(c) and release['parent_contract_sha256']==A100_PARENT_SHA,
        'Ownership release contract binding drift')
    require(release['execution_ids']==[j['id'] for j in c['jobs']]
        and release['A100_queue_exclusion_verified'] is True and release['unstarted_verified'] is True
        and release['released_to_rtx'] is True,'A100 twelve-job release not verified')
    proof=root/'handoff/ownership_release_source.json'
    require(sha_file(proof)==release['source_receipt_raw_sha256'],'Queue release raw proof drift')
    return release
'''

BATCH = '''
def batch_groups(c,host):
    jobs=[j for j in c['jobs'] if j['host']==host]
    require(len(jobs)==(3 if host=='5080' else 9),'RTX handoff count drift')
    batches=[jobs[i:i+3] for i in range(0,len(jobs),3)]
    require(all(len(b)==3 and len({(j['dataset'],j['seed']) for j in b})==1
        and [j['arm'] for j in b]==c['arms'] for b in batches),'Parallel group scope drift')
    return batches
'''

GPU_CHECK = '''
def check_active_gpu(c,host,root,active):
    import rtx_gpu_operations
    return rtx_gpu_operations.check_active_gpu(sys.modules[__name__],c,host,root,active)
'''


def build_runtime(original, parallel, science, hosts):
    """Retain frozen RTX science; transplant only verified parallel operations."""
    text = replace_node(original, 'validate', VALIDATE)
    constants = ('\nHANDOFF_SCHEMA = ' + repr(SCHEMA)
        + '\nHANDOFF_ASSIGNMENTS = ' + repr(ASSIGNMENTS)
        + '\nSCIENCE_FIELDS = ' + repr(SCIENCE_FIELDS)
        + '\nSCIENCE_FINGERPRINT = ' + repr(scientific_fingerprint(science))
        + '\nSOURCE_CLOSURE = ' + repr(SOURCE_SHA)
        + '\nA100_PARENT_SHA = ' + repr(PARENT_SHA)
        + '\nPINNED_HOSTS = ' + repr(hosts) + '\n')
    text = text.replace('ROOT = None', constants + '\nROOT = None')
    for name in ('PulseBudget', 'dispatch', 'verify_training_permit'):
        text = replace_node(text, name, function(parallel, name))
    # These are operational process/lease wrappers only. The scientific fit
    # from the first data admission through selected/last replay stays exact.
    text = replace_node(text, 'run_fit', function(parallel, 'run_fit'))
    auth = function(text, 'authorization')
    auth = auth.replace('    validate(c)', '    validate(c)\n    verify_ownership_release(c,root)', 1)
    text = replace_node(text, 'authorization', auth)
    additions = [function(parallel, name) for name in (
        'job_lease_path', 'await_job_lease', 'owned_root_bytes', 'verify_timeout_parent',
        'verify_gpu_membership', 'write_parallel_pulse', 'validate_terminal', 'audit_completed_batch')]
    additions += [RELEASE, BATCH, GPU_CHECK]
    text = text.replace('\ndef run_fit(c,host,root,job,deadline):',
        '\n' + '\n\n'.join(additions) + '\ndef run_fit(c,host,root,job,deadline):', 1)
    tree = ast.parse(text)
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    require(len(names) == len(set(names)), 'Duplicate operation symbol')
    allowed = {'validate', 'authorization', 'PulseBudget', 'dispatch', 'verify_training_permit', 'run_fit'}
    for node in ast.parse(original).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in allowed:
            require(function(original, node.name) == function(text, node.name),
                'Scientific/qualification function changed: ' + node.name)
    marker = "    budget();data=next(d for d in c['datasets'] if d['dataset_id']==job['dataset'])"
    require(function(original, 'run_fit').split(marker, 1)[1]
        == function(text, 'run_fit').split(marker, 1)[1], 'Scientific fit body changed')
    return text


def make_contract(parent, hosts, operation_files):
    contract = deepcopy(parent)
    contract.update(schema=SCHEMA, assignments=deepcopy(ASSIGNMENTS), hosts=deepcopy(hosts),
        jobs=expected_jobs(parent['arms']), prior_stage=None)
    contract['limits']['workers_per_host'] = 3
    contract['parallelism'] = {'maximum_workers': 3, 'grouping': 'same_dataset_seed_three_arms',
        'per_process_cuda_fraction': .30, 'aggregate_gpu_fraction': .8}
    contract['operation'] = {'files': operation_files, 'files_sha256': canonical(operation_files)}
    contract['handoff'] = {'parent_contract_sha256': PARENT_SHA,
        'fresh_fits': 12, 'original_execution_ids_preserved': True,
        'A100_release_required_before_any_GPU_work': True, 'automatic_retry': False,
        'same_dataset_seed_three_arms_same_physical_GPU': True,
        'cross_seed_GPU_Runtime_difference_must_be_reported': True}
    require(scientific_fingerprint(contract) == scientific_fingerprint(parent), 'Scientific contract changed')
    require(len(contract['jobs']) == 12, 'Handoff count drift')
    return contract


def host_specs(parent):
    hosts = deepcopy(parent['hosts'])
    for host, spec in hosts.items():
        root = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + SCHEMA + '_' + host
        spec.update(root=root, source_root=root + '/source', operation_root=root + '/operation',
            output_dir=root + '/run', tmux=SCHEMA + '_' + host,
            assigned_datasets=sorted({d for d, seed in ASSIGNMENTS[host]}), old_owner_roots=[])
        spec['environment'].update(PYTHONPATH=root + '/source', MPLCONFIGDIR=root + '/cache/matplotlib',
            XDG_CACHE_HOME=root + '/cache')
    return hosts


def validate_local_contract(runtime, contract):
    namespace = {'__name__': 'rtx_handoff_local_validation', '__file__': '/operation/campaign.py'}
    exec(compile(runtime, '/operation/campaign.py', 'exec'), namespace)
    namespace['validate'](contract, verify_files=False)
    for host in ASSIGNMENTS:
        namespace['batch_groups'](contract, host)
    return namespace


def package(target, files):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for name in sorted(files):
            path = safe(target, name)
            require(sha(path) == files[name], 'Package input drift: ' + name)
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mode = 0o644
            with path.open('rb') as stream:
                archive.addfile(info, stream)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode='wb', mtime=0) as stream:
        stream.write(payload.getvalue())
    write_new(target / 'deployment/package.tar.gz', compressed.getvalue())
    manifest = {'files': files, 'archive_raw_sha256': sha(target / 'deployment/package.tar.gz'),
        'file_count': len(files), 'network_calls': 0}
    write_json(target / 'deployment/package_manifest.json', manifest)
    return manifest


def prepare(args):
    active = verify_frozen(PARENT, PARENT_SHA)
    parent = verify_frozen(RTX, RTX_SHA)
    parallel_contract = verify_frozen(PARALLEL, PARALLEL_SHA)
    require(scientific_fingerprint(active) == scientific_fingerprint(parent), 'A100/RTX scientific identity drift')
    ids = {j['id'] for j in active['jobs']}
    require(all(j['id'] in ids for j in expected_jobs(parent['arms'])), 'Foreign original execution identity')
    hosts = host_specs(parent)
    original = (RTX / 'operation/campaign.py').read_text()
    parallel = (PARALLEL / 'operation/campaign.py').read_text()
    runtime = build_runtime(original, parallel, parent, hosts)
    robust_name = 'operation/recovery_operations.py'
    require(robust_name in active['operation']['files'], 'Verified GPU transition helper absent')
    robust = (PARENT / robust_name).read_bytes()
    operation_files = {'operation/campaign.py': hashlib.sha256(runtime.encode()).hexdigest(),
        'operation/rtx_gpu_operations.py': hashlib.sha256(robust).hexdigest()}
    contract = make_contract(parent, hosts, operation_files)
    validate_local_contract(runtime, contract)
    target = Path(args.output).resolve()
    for name, digest in {**active['source']['files'], **active['input_files']}.items():
        origin = safe(PARENT / 'source', name)
        destination = safe(target / 'source', name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(sha(destination) == digest, 'Existing copied source/input differs: ' + name)
        else:
            shutil.copyfile(origin, destination)
        require(sha(destination) == digest, 'Copied source/input drift: ' + name)
    write_new(target / 'operation/campaign.py', runtime)
    write_new(target / 'operation/rtx_gpu_operations.py', robust)
    write_json(target / 'execution_contract.json', contract)
    digest = canonical(contract)
    approval = {'approved': True, 'contract_sha256': digest, 'hosts': list(hosts), 'new_fits': 12,
        'user_instruction': USER_INSTRUCTION, 'Test_access': False,
        'native_qualification_and_concurrency_required': True, 'A100_queue_release_required': True}
    write_json(target / 'approval.json', approval)
    current = {'schema': SCHEMA, 'canonical_sha256': digest, 'contract_sha256': digest,
        'hosts': {h: s['root'] for h, s in hosts.items()}, 'source_closure_sha256': SOURCE_SHA,
        'operation_closure_sha256': contract['operation']['files_sha256'],
        'status': 'prepared_not_authorized_to_train', 'new_fits': 12,
        'native_qualification': 'required', 'concurrency_admission': 'required',
        'A100_queue_release': 'required', 'network_calls': 0, 'GPU_calls': 0}
    write_json(target / 'current.json', current)
    seal = {'contract_sha256': digest, 'source_files': active['source']['files'],
        'input_files': active['input_files'], 'operation_files': operation_files,
        'project_files': {'paper/scripts/prepare_titantpp_history_rtx_handoff.py': sha(Path(__file__))},
        'A100_parent_contract_sha256': PARENT_SHA, 'RTX_parent_contract_sha256': RTX_SHA,
        'parallel_parent_contract_sha256': PARALLEL_SHA}
    write_json(target / 'control/integrity_receipt.json', seal)
    files = {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*'))
        if p.is_file() and str(p.relative_to(target)).split('/')[0] in ('source', 'operation')}
    files.update({name: sha(target / name) for name in ('execution_contract.json', 'approval.json')})
    manifest = package(target, files)
    return {'status': current['status'], 'bundle': str(target), 'canonical_sha256': digest,
        'operation_closure_sha256': contract['operation']['files_sha256'], 'source_closure_sha256': SOURCE_SHA,
        'control_integrity_raw_sha256': sha(target / 'control/integrity_receipt.json'),
        'archive_raw_sha256': manifest['archive_raw_sha256'], 'file_count': manifest['file_count'],
        'jobs_per_host': {h: sum(j['host'] == h for j in contract['jobs']) for h in hosts},
        'network_calls': 0, 'GPU_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', default=str(TARGET))
    args = parser.parse_args()
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(PROJECT / 'paper/tests'),
            pattern='test_titantpp_history_rtx_bundle.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    print(json.dumps(prepare(args), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
