"""Explicit four-job queue exclusion; no work occurs at import.

The old dispatcher continues seeds42/52. Its exclusive first seed62 claim is
reserved, never deleted, so it cannot create any seed62 worker. This is not a
claim that the old dispatcher or GPU has been released or has exited.
"""
from __future__ import annotations
import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types

PROJECT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
SCHEMA = 'titantpp_components_seed62_handoff_20261013_v1'
PARENT_SCHEMA = 'titantpp_cnn_gru_components_20261012_v5'
PARENT_SHA = 'e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074'
SOURCE_SHA = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
OP_SHA = '12c836dc5fe1d9146b83a352ba77abc297f4d69c6f68d83ad00935a2bc442903'
LOCAL40_SHA = '16e317c3c16dddede95c6aa6174026dca0ab772879bf438c3d1e6b752eff7074'
ROOT_PARENT = Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts')
PARENT_ROOT = ROOT_PARENT / (PARENT_SCHEMA + '_5090')
DESTINATION_ROOT = ROOT_PARENT / (SCHEMA + '_5080')
ARMS = ('titantpp_cnn_gru54_correction_disabled', 'titantpp_cnn_gru54_cnn_only',
        'titantpp_cnn_gru54_gru_only', 'titantpp_cnn_gru54')
DATASET = 'intermittent_frozen_5000'
STARTED = 1791794258.15949
DEADLINE = 1792399058.15949
SCIENCE_FIELDS = ('source', 'input_files', 'datasets', 'dataset_sha256', 'training',
                  'diagnostic', 'architecture', 'analysis', 'parallelism', 'arms',
                  'seeds', 'policy', 'launch_policy', 'design_sha256', 'runner_modules',
                  'evaluation_scope', 'held_out_test_evaluated', 'limits')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 64 * 1024**2, 'Oversized metadata')
    return json.loads(path.read_bytes())


def safe(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and bool(p.parts), 'Unsafe relative path')
    result = Path(root) / p
    require(not result.is_symlink() and result.resolve().is_relative_to(Path(root).resolve()), 'Unsafe owned path')
    return result


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def transferred_jobs(host='5090'):
    return [{'id': f'{DATASET}__62__{arm}', 'host': host,
             'dataset': DATASET, 'seed': 62, 'arm': arm} for arm in ARMS]


def descriptor(parent, helper_sha):
    return {'parent_contract_sha256': PARENT_SHA, 'parent_source_sha256': SOURCE_SHA,
        'parent_operation_sha256': OP_SHA, 'parent_root': str(PARENT_ROOT),
        'parent_gpu_uuid': parent['hosts']['5090']['gpu_uuid'],
        'parent_python': parent['hosts']['5090']['python'],
        'parent_campaign_sha256': parent['operation']['files']['operation/campaign.py'],
        'administrative_helper_sha256': helper_sha,
        'parent_jobs': transferred_jobs(), 'destination_jobs': transferred_jobs('5080'),
        'first_claim_path': str(PARENT_ROOT / 'claims' / (transferred_jobs()[0]['id'] + '.json')),
        'kind': 'partial_seed62_queue_exclusion_not_full_owner_release',
        'queue_exclusion_required': True, 'parent_full_owner_release_required': False,
        'parent42_and52_natural_progress_preserved': True,
        'original_host_lease': {'started_at_unix': STARTED, 'deadline_unix': DEADLINE}}


def check_parent_contract(parent):
    require(canonical(parent) == PARENT_SHA, 'Frozen parent contract mismatch')
    require(parent['source']['files_sha256'] == SOURCE_SHA and
            parent['operation']['files_sha256'] == OP_SHA, 'Frozen source/operation mismatch')
    require([j for j in parent['jobs'] if j['host'] == '5090' and j['seed'] == 62] == transferred_jobs(),
            'Exact parent four-job mapping changed')


def check_parent_seals(root, parent):
    require(Path(root) == PARENT_ROOT and not Path(root).is_symlink(), 'Foreign parent root')
    check_parent_contract(parent)
    require(read(root / 'execution_contract.json') == parent, 'Remote parent contract changed')
    for name, digest in {**parent['source']['files'], **parent['input_files']}.items():
        require(sha(safe(root / 'source', name)) == digest, 'Frozen parent source/input changed: ' + name)
    for name, digest in parent['operation']['files'].items():
        require(sha(safe(root, name)) == digest, 'Frozen parent operation changed: ' + name)
    start = read(root / 'start_permit.json'); approval = read(root / 'approval.json')
    qualification = read(root / 'qualification/receipt.json'); permit = read(root / 'training_permit.json')
    require(approval['approved'] is True and approval['contract_sha256'] == PARENT_SHA
            and approval['hosts'] == ['5090'] and start['contract_sha256'] == PARENT_SHA
            and start['approval_sha256'] == canonical(approval), 'Parent approval/start binding changed')
    require(qualification['status'] == 'passed' and qualification['contract_sha256'] == PARENT_SHA
            and qualification['host'] == '5090' and qualification['source_files_sha256'] == SOURCE_SHA
            and qualification['runtime']['gpu']['uuid'] == parent['hosts']['5090']['gpu_uuid']
            and all(qualification['runtime'][k] == v for k, v in parent['hosts']['5090']['runtime_expected'].items()),
            'Qualified parent Runtime changed')
    require(permit['contract_sha256'] == PARENT_SHA and permit['start_permit_sha256'] == canonical(start)
            and permit['qualification_sha256'] == sha(root / 'qualification/receipt.json')
            and permit['qualification'] == qualification, 'Parent training permit changed')
    for name, digest in qualification['qualification_files'].items():
        require(sha(safe(root, name)) == digest, 'Parent native qualification evidence changed')
    dispatch = next(n for n in ast.parse((root / 'operation/campaign.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'dispatch')
    text = ast.unparse(dispatch)
    require(text.index("root / 'claims'") < text.index('subprocess.Popen('), 'Claim must precede worker creation')
    require('exclusive=True' in text and '.unlink(' not in text and 'os.remove(' not in text,
            'Immutable exclusive claim protocol changed')


def never_started(root, jobs, barrier=None):
    checked = []
    for job in jobs:
        name = job['id']
        paths = ['run/' + name, 'logs/' + name + '.log', 'leases/' + name + '.json',
                 'worker_exits/' + name + '.json', 'worker_identities/' + name + '.json',
                 'progress/' + name + '.json']
        claim = 'claims/' + name + '.json'
        if Path(root) / claim != barrier:
            paths.append(claim)
        for relative in paths:
            path = safe(root, relative)
            require(not path.exists() and not path.is_symlink(), 'Transfer job already attempted: ' + relative)
            checked.append(relative)
    return checked


def process(pid):
    p = Path('/proc') / str(pid)
    before = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    argv = (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    after = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    require(before[19] == after[19] and after[0] not in ('Z', 'X'), 'Process identity changed')
    return {'pid': pid, 'ppid': int(after[1]), 'pgrp': int(after[2]), 'start_ticks': after[19], 'argv': argv}


def inventory(root, parent):
    spec = parent['hosts']['5090']
    state = read(root / 'status.json')
    ids = {state['supervisor_pid'], *(v['worker_group_pid'] for v in state.get('active_jobs', {}).values())}
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, text=True, capture_output=True, timeout=20).stdout
    pids = []
    for row in output.splitlines():
        parts = [s.strip() for s in row.split(',')]
        require(len(parts) == 2, 'Unresolved GPU inventory')
        if parts[0] == spec['gpu_uuid']:
            pids.append(int(parts[1]))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
        check=True, text=True, capture_output=True, timeout=20).stdout.splitlines()
    require(spec['gpu_uuid'] in [v.strip() for v in gpu], 'Fixed parent GPU absent')
    ids.update(pids)
    return {'observed_unix': time.time(), 'host': '5090', 'root': str(root),
        'contract_sha256': PARENT_SHA, 'gpu_uuid': spec['gpu_uuid'], 'gpu_pids': pids,
        'status': state, 'processes': {str(pid): process(pid) for pid in ids},
        'leases': {name: read(root / 'leases' / (name + '.json')) for name in state.get('active_jobs', {})}}


def verify_running_parent(observed, parent, now):
    spec = parent['hosts']['5090']
    require(observed['host'] == '5090' and observed['root'] == str(PARENT_ROOT)
            and observed['contract_sha256'] == PARENT_SHA and observed['gpu_uuid'] == spec['gpu_uuid'],
            'Parent observation identity mismatch')
    require(-10 <= now - observed['observed_unix'] < 90, 'Stale observation')
    state = observed['status']; active = state.get('active_jobs', {})
    require(state.get('status') == 'running' and active and -10 <= now - state['updated_unix'] < 90
            and observed['observed_unix'] < state['deadline_unix'], 'Fresh live parent dispatcher required')
    allowed = {j['id']: j for j in parent['jobs'] if j['host'] == '5090' and j['seed'] in (42, 52)}
    require(set(active) <= set(allowed) and len(active) <= 3, 'Seed62 already active or foreign active scope')
    require(0 < len(observed['gpu_pids']) <= 3, 'Actual current GPU worker witness required')
    require(len({v['job']['seed'] for v in active.values()}) == 1, 'Different seed workers mixed')
    expected = [spec['python'], spec['operation_root'] + '/campaign.py', '--contract',
                spec['root'] + '/execution_contract.json', '--host', '5090', '--mode', 'dispatch']
    processes = observed['processes']; owner = processes[str(state['supervisor_pid'])]
    require(owner['argv'] == state['supervisor_argv'] == expected, 'Exact live dispatcher PID/argv mismatch')
    for name, worker in active.items():
        proc = processes[str(worker['worker_group_pid'])]; lease = observed['leases'][name]
        require(worker['job'] == allowed[name] and proc['argv'] == worker['worker_guard_argv']
                and proc['ppid'] == state['supervisor_pid'] and proc['pgrp'] == worker['worker_group_pid'],
                'Active timeout/job identity mismatch')
        require(lease['job'] == name and lease['contract_sha256'] == PARENT_SHA
                and lease['worker_group_pid'] == worker['worker_group_pid']
                and lease['supervisor_pid'] == state['supervisor_pid']
                and observed['observed_unix'] < lease['expires_unix'] <= worker['condition_deadline_unix'],
                'Active lease identity mismatch')
    for pid in observed['gpu_pids']:
        proc = processes[str(pid)]
        owners = [v for v in active.values() if proc['ppid'] == v['worker_group_pid']]
        require(len(owners) == 1 and proc['pgrp'] == owners[0]['worker_group_pid']
                and proc['argv'] == owners[0]['worker_guard_argv'][4:], 'Foreign parent GPU process')


def atomic_exclusive(path, payload):
    raw = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.seed62-exclusion-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        os.unlink(temporary)
    return raw


def reserve_boundary(root, destination, approval, *, probe=inventory, verifier=check_parent_seals, now=time.time):
    require(Path(root) == PARENT_ROOT, 'Foreign reservation root')
    parent = read(root / 'execution_contract.json')
    check_parent_contract(parent); validate_contract(destination, parent)
    require(isinstance(approval, str) and approval.strip(), 'Explicit queue migration approval missing')
    folder = safe(root, 'handoff_seed62'); folder.mkdir(exist_ok=True)
    write_exclusive(folder / 'intent.json', {'destination_contract_sha256': canonical(destination),
        'approval': approval, 'started_unix': now(), 'automatic_retry': False})
    try:
        verifier(root, parent)
        for name in ('failure.json', 'phases/train_failure.json', 'phases/train_complete.json'):
            require(not safe(root, name).exists(), 'Parent already terminal or failed')
        observation = probe(root, parent); verify_running_parent(observation, parent, now())
        jobs = transferred_jobs(); before = never_started(root, jobs)
        barrier = root / 'claims' / (jobs[0]['id'] + '.json')
        payload = {'schema': SCHEMA + '_reservation', 'kind': 'administrative_queue_exclusion_not_scientific_claim',
            'parent_contract_sha256': PARENT_SHA, 'parent_source_sha256': SOURCE_SHA,
            'parent_operation_sha256': OP_SHA, 'parent_root': str(root),
            'destination_contract_sha256': canonical(destination), 'destination_root': str(DESTINATION_ROOT),
            'parent_jobs': jobs, 'destination_jobs': transferred_jobs('5080'), 'first_claim_path': str(barrier),
            'barrier_job': jobs[0],
            'user_approval': approval, 'created_unix': now(), 'scientific_worker_created': False,
            'process_signals_sent': 0, 'parent_full_owner_released': False}
        raw = atomic_exclusive(barrier, payload)
        after = never_started(root, jobs, barrier)
        require(barrier.read_bytes() == raw, 'Barrier changed after publication')
        proof = {'schema': SCHEMA + '_queue_exclusion_proof', 'queue_exclusion_verified': True,
            'parent_full_owner_released': False, 'parent_exit_inferred': False,
            'parent_contract_sha256': PARENT_SHA, 'destination_contract_sha256': canonical(destination),
            'parent_source_sha256': SOURCE_SHA, 'parent_operation_sha256': OP_SHA,
            'administrative_helper_sha256': destination['seed62_handoff']['administrative_helper_sha256'],
            'barrier_path': str(barrier), 'barrier_raw_sha256': hashlib.sha256(raw).hexdigest(),
            'transfer_jobs': jobs, 'destination_jobs': transferred_jobs('5080'),
            'barrier_base64': base64.b64encode(raw).decode(), 'reservation': payload,
            'before_never_started': before, 'after_never_started': after,
            'parent_observation': observation, 'completed_unix': now(), 'automatic_retry': False,
            'claim_before_Popen_sealed': True, 'parent42_and52_natural_progress_preserved': True,
            'expected_parent_future_stop': 'FileExistsError before seed62 NONE Popen; preserve native exit1/failure'}
        validate_queue_proof(proof, destination, parent)
        write_exclusive(folder / 'queue_exclusion_proof.json', proof)
        return proof
    except BaseException as error:
        write_exclusive(folder / 'failure.json', {'error': type(error).__name__, 'message': str(error),
            'destination_contract_sha256': canonical(destination), 'automatic_retry': False,
            'barrier_preserved_if_created': True, 'completed_unix': now()})
        raise


def validate_contract(c, parent):
    check_parent_contract(parent)
    require(c['schema'] == SCHEMA and c['stage'] == 'components', 'Foreign handoff schema')
    administrative = {'schema', 'assignments', 'jobs', 'hosts', 'operation', 'operational_replacement'}
    require(set(c) == set(parent) | {'seed62_handoff'}, 'Unexpected contract fields')
    for field in set(parent) - administrative:
        require(c[field] == parent[field], 'Unapproved original contract field changed: ' + field)
    for field in SCIENCE_FIELDS:
        require(c[field] == parent[field], 'Scientific field changed: ' + field)
    require(c['assignments'] == {'5080': [[DATASET, 62]], '5090': []}
            and c['jobs'] == transferred_jobs('5080'), 'Only same four-arm seed62 mapping is permitted')
    handoff = c['seed62_handoff']
    require(handoff == descriptor(parent, handoff['administrative_helper_sha256']), 'Queue exclusion descriptor changed')
    require(len(handoff['administrative_helper_sha256']) == 64
            and all(ch in '0123456789abcdef' for ch in handoff['administrative_helper_sha256']), 'Helper fingerprint missing')
    require(c['operational_replacement'] == {'parent_contract_sha256': PARENT_SHA,
        'host_leases': {'5080': {'started_at_unix': STARTED, 'deadline_unix': DEADLINE}},
        'parent_originals_preserved': True, 'parent42_and52_natural_progress_preserved': True}, 'Original host lease or preservation changed')
    require(set(c['hosts']) == {'5080', '5090'}, 'Pinned hosts changed')
    for host, spec in c['hosts'].items():
        expected = deepcopy(parent['hosts'][host]); root = str(ROOT_PARENT / (SCHEMA + '_' + host))
        expected.update(root=root, source_root=root + '/source', operation_root=root + '/operation', output_dir=root + '/run', tmux=SCHEMA + '_' + host)
        expected['environment'].update(PYTHONPATH=root + '/source', MPLCONFIGDIR=root + '/cache/matplotlib', XDG_CACHE_HOME=root + '/cache')
        if host == '5080': expected['assigned_datasets'] = [DATASET]
        if host == '5090': expected['assigned_datasets'] = []
        require(spec == expected, 'Destination GPU/Runtime/root/environment changed')
    ops = c['operation']['files']
    require(set(ops) == set(parent['operation']['files']) and canonical(ops) == c['operation']['files_sha256'], 'Operation closure changed')
    for name in ('operation/frozen_runtime.py', 'operation/qualification.py', 'operation/pipeline.py'):
        require(ops[name] == parent['operation']['files'][name], 'Frozen trainer/qualifier/pipeline changed')


def validate_queue_proof(proof, c, parent):
    validate_contract(c, parent)
    meta = c['seed62_handoff']
    require(proof['schema'] == SCHEMA + '_queue_exclusion_proof' and proof['queue_exclusion_verified'] is True
            and proof['parent_full_owner_released'] is False and proof['parent_exit_inferred'] is False
            and proof['automatic_retry'] is False and proof['claim_before_Popen_sealed'] is True
            and proof['parent42_and52_natural_progress_preserved'] is True, 'Not a partial queue exclusion proof')
    require(proof['parent_contract_sha256'] == PARENT_SHA and proof['destination_contract_sha256'] == canonical(c)
            and proof['parent_source_sha256'] == SOURCE_SHA and proof['parent_operation_sha256'] == OP_SHA
            and proof['administrative_helper_sha256'] == meta['administrative_helper_sha256'], 'Proof source/contract drift')
    raw = base64.b64decode(proof['barrier_base64'], validate=True)
    require(hashlib.sha256(raw).hexdigest() == proof['barrier_raw_sha256'] and json.loads(raw) == proof['reservation'], 'Barrier raw binding changed')
    p = proof['reservation']
    require(p['schema'] == SCHEMA + '_reservation' and p['kind'] == 'administrative_queue_exclusion_not_scientific_claim'
            and p['parent_contract_sha256'] == PARENT_SHA and p['parent_source_sha256'] == SOURCE_SHA
            and p['parent_operation_sha256'] == OP_SHA and p['parent_root'] == str(PARENT_ROOT)
            and p['destination_root'] == str(DESTINATION_ROOT) and p['destination_contract_sha256'] == canonical(c)
            and p['parent_jobs'] == transferred_jobs() and p['destination_jobs'] == transferred_jobs('5080')
            and p['barrier_job'] == transferred_jobs()[0]
            and p['first_claim_path'] == proof['barrier_path'] == meta['first_claim_path']
            and p['scientific_worker_created'] is False and p['process_signals_sent'] == 0
            and p['parent_full_owner_released'] is False and isinstance(p['user_approval'], str) and p['user_approval'].strip(), 'Reservation scope or owner semantics changed')
    expected = []
    for j in transferred_jobs():
        name = j['id']; expected += ['run/' + name, 'logs/' + name + '.log', 'leases/' + name + '.json', 'worker_exits/' + name + '.json', 'worker_identities/' + name + '.json', 'progress/' + name + '.json', 'claims/' + name + '.json']
    require(proof['before_never_started'] == expected
            and proof['after_never_started'] == [v for v in expected if str(PARENT_ROOT / v) != meta['first_claim_path']], 'Unstarted evidence coverage changed')
    require(proof['transfer_jobs'] == transferred_jobs() and proof['destination_jobs'] == transferred_jobs('5080'), 'Proof transfer/destination mappings changed')
    verify_running_parent(proof['parent_observation'], parent, p['created_unix'])
    require(proof['parent_observation']['observed_unix'] <= p['created_unix'] <= proof['completed_unix'], 'Reservation time order changed')
    return proof


def remote_reservation_source(c, approval):
    """Return one-shot source for root's explicitly authorized remote transport."""
    text = (PROJECT / 'paper/scripts/titantpp_components_seed62_handoff.py').read_text()
    require(hashlib.sha256(text.encode()).hexdigest() == c['seed62_handoff']['administrative_helper_sha256'], 'Administrative helper changed; SSH0')
    standalone = text.replace('from __future__ import annotations\n', '', 1)
    return "__name__='seed62_admin_remote'\n" + standalone + '\n' + \
        'print(json.dumps(reserve_boundary(PARENT_ROOT, json.loads(' + repr(json.dumps(c)) + '), ' + repr(approval) + '), ensure_ascii=False, allow_nan=False))\n'


def check_bundle(bundle, *, require_queue_exclusion=True):
    from paper.scripts import launch_titantpp_cnn_gru_components_v5 as parent_launcher
    bundle = Path(bundle).resolve(); c = read(bundle / 'execution_contract.json')
    parent_bundle = PROJECT / 'search_artifacts' / PARENT_SCHEMA
    require(sha(parent_bundle / 'local_tools_integrity.json') == LOCAL40_SHA, 'Original local40 raw seal changed; SSH0')
    original, parent_gate = parent_launcher.check_bundle(parent_bundle)
    require(parent_gate['contract_sha256'] == PARENT_SHA, 'Original frozen parent gate changed; SSH0')
    parent = read(bundle / 'ownership/parent_execution_contract.json'); validate_contract(c, parent)
    require(parent == original, 'Preserved parent contract changed')
    from paper.scripts.prepare_titantpp_components_seed62_handoff import adapter_source
    expected_adapter = adapter_source((parent_bundle / 'operation/campaign.py').read_text())
    require(c['operation']['files']['operation/campaign.py'] == hashlib.sha256(expected_adapter.encode()).hexdigest(), 'Adapter exceeds allowed validate/authorization scope')
    require(read(bundle / 'current.json') == {'schema': SCHEMA, 'contract_sha256': canonical(c), 'bundle': str(bundle), 'status': 'prepared_seed62_queue_exclusion'}, 'Current pointer changed')
    for name, digest in {**c['source']['files'], **c['input_files']}.items():
        require(sha(safe(bundle / 'source', name)) == digest, 'Source/input changed')
    for name, digest in c['operation']['files'].items(): require(sha(safe(bundle, name)) == digest, 'Sealed operation changed')
    seal = read(bundle / 'local_tools_integrity.json')
    require(seal['files_sha256'] == canonical(seal['files']), 'Local tools closure changed')
    for name, digest in seal['files'].items(): require(sha(safe(PROJECT, name)) == digest, 'Local tool changed')
    pending = read(bundle / 'preparation_pending.json')
    require(pending['contract_sha256'] == canonical(c) and pending['contract_raw_sha256'] == sha(bundle / 'execution_contract.json')
            and pending['local_tools_integrity_raw_sha256'] == sha(bundle / 'local_tools_integrity.json')
            and pending['parent_contract_raw_sha256'] == sha(bundle / 'ownership/parent_execution_contract.json'), 'Preparation pin changed')
    hold = read(bundle / 'launch_hold.json')
    require(hold['hold'] is True and hold['contract_sha256'] == canonical(c), 'Immutable original hold changed')
    require(c['seed62_handoff']['administrative_helper_sha256'] == sha(PROJECT / 'paper/scripts/titantpp_components_seed62_handoff.py'), 'Administrative helper pin changed')
    if require_queue_exclusion:
        proof = read(bundle / 'ownership/queue_exclusion_source.json'); validate_queue_proof(proof, c, parent)
        receipt = read(bundle / 'preparation_receipt.json')
        require(receipt['contract_sha256'] == canonical(c) and receipt['queue_exclusion_source_raw_sha256'] == sha(bundle / 'ownership/queue_exclusion_source.json')
                and receipt['payload_sha256'] == sha(bundle / 'deployment.tar.gz')
                and receipt['contract_raw_sha256'] == sha(bundle / 'execution_contract.json'), 'Final preparation receipt changed')
        parent_launcher.inspect_archive(bundle / 'deployment.tar.gz', expected_members(c, receipt))
    return c, {'contract_sha256': canonical(c), 'queue_exclusion_required': require_queue_exclusion, 'remote_calls': 0}


def expected_members(c, receipt):
    members = {'execution_contract.json': receipt['contract_raw_sha256'], **c['operation']['files']}
    members.update({'source/' + n: digest for n, digest in {**c['source']['files'], **c['input_files']}.items()})
    members.update({'ownership/parent_execution_contract.json': receipt['parent_contract_raw_sha256'],
                    'ownership/queue_exclusion_source.json': receipt['queue_exclusion_source_raw_sha256']})
    return members


def launcher_for(bundle):
    """Isolated launcher namespace: the original module and REMOTE_SCRIPT stay untouched."""
    from paper.scripts import launch_titantpp_cnn_gru_components_v5 as original
    source = Path(original.__file__).read_text(); module = types.ModuleType('seed62_isolated_launcher')
    module.__file__ = original.__file__; exec(compile(source, original.__file__, 'exec'), module.__dict__)
    module.SCHEMA = SCHEMA
    module.validate_contract = lambda c: validate_contract(c, read(Path(bundle) / 'ownership/parent_execution_contract.json'))
    module.check_bundle = lambda path: check_bundle(path)
    module.expected_members = lambda c, raw: expected_members(c, read(Path(bundle) / 'preparation_receipt.json'))
    original_auth = module.authorization
    def authorization(path, c, host):
        require(host == '5080', 'Seed62 new execution only permits5080')
        approval = original_auth(path, c, host)
        require(approval['queue_exclusion_source_raw_sha256'] == sha(Path(path) / 'ownership/queue_exclusion_source.json'), 'Approval queue exclusion raw pin changed')
        return approval
    module.authorization = authorization
    require(module.REMOTE_SCRIPT == original.REMOTE_SCRIPT, 'Original transport changed')
    return module


def run_mode(bundle, mode, host='5080', *, remote=None):
    require(host == '5080', 'Old5090 dispatcher must not be relaunched')
    return launcher_for(bundle).run_mode(bundle, mode, host, remote=remote)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--mode', choices=('check', 'deploy', 'qualify', 'train', 'status'), required=True)
    args = p.parse_args(); print(json.dumps(run_mode(args.bundle, args.mode), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
