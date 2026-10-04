"""Execution-only seed62 handoff; the 114 scientific files remain unchanged.

No import-time training or network access. This standalone stdlib entrypoint
imports the sealed parent engine only for qualify/fit/dispatch. Source reserve
uses O_EXCL claims and never signals, stops or rewrites the running supervisor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

NAME = 'titantpp_history_capacity_handoff_20261004_v1'
PARENT_SHA = '8434041a6a301995a8d22baa77d5ca0da4eb2cd942d091a8aa8cef379406c48d'
SOURCE_SHA = '41a4a4ff79e5e17d907b9ee8966723004516ee2e42608d31c7f1a871f0acd3e1'
DATASET = 'intermittent_frozen_5000'
ARMS = ('titantpp_history_mlp_width8', 'titantpp_history_mlp_width12')
DESTINATION = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + NAME + '_5080'
ENGINE = 'paper/scripts/run_titantpp_history_capacity_campaign.py'
RESERVATION_DIR = 'handoff_seed62'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    else:
        temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temporary, path)


def target_jobs(parent):
    jobs = [j for j in parent['jobs'] if j['host'] == '5090' and j['dataset'] == DATASET and j['seed'] == 62]
    require([j['arm'] for j in jobs] == list(ARMS), 'Expected exactly the seed62 width8/12 pair')
    return jobs


def derive_contract(parent):
    """Allow only explicit execution ownership and SHA-preserving path changes."""
    require(canonical(parent) == PARENT_SHA, 'Parent contract changed')
    selected = {j['id'] for j in target_jobs(parent)}
    result = deepcopy(parent)
    spec = result['hosts']['5080']
    spec.update(root=DESTINATION, source_root=DESTINATION + '/source', output_dir=DESTINATION + '/run',
                tmux=NAME + '_5080', assigned_datasets=[DATASET])
    for job in result['jobs']:
        if job['id'] in selected:
            job['host'] = '5080'
    reference = DESTINATION + '/references/width4/' + DATASET + '/62'
    for row in result['reuse']:
        if row['host'] == '5090' and row['dataset'] == DATASET and row['seed'] == 62:
            row.update(host='5080', remote_run=reference)
    result['baseline_replays'][DATASET]['62']['checkpoint'] = reference + '/best_val_qty_rmse_model.pt'
    data = next(d for d in result['datasets'] if d['dataset_id'] == DATASET)
    for kind in ('data', 'split_manifest'):
        identity = data['inherited_data_identity'][kind]
        identity['path'] = DESTINATION + '/data/' + Path(identity['path']).name
    result['dataset_sha256'][DATASET] = canonical(data)
    return result


def contract_diff(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after):
        return [change for key in before for change in contract_diff(before[key], after[key], prefix + '/' + key)]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [change for i, (a, b) in enumerate(zip(before, after)) for change in contract_diff(a, b, prefix + '/' + str(i))]
    return [] if before == after else [{'path': prefix, 'before': before, 'after': after}]


def validate_contract(parent, operational):
    require(canonical(parent) == PARENT_SHA, 'Parent contract changed')
    require(canonical(parent['source']['files']) == parent['source']['files_sha256'] == SOURCE_SHA
            and len(parent['source']['files']) == 114, 'Scientific source closure changed')
    require(operational == derive_contract(parent), 'Non-whitelisted operational/scientific change')
    return operational


def verify_source(parent, source):
    source = Path(source).resolve()
    for name, digest in parent['source']['files'].items():
        relative = Path(name)
        path = source / relative
        require(not relative.is_absolute() and '..' not in relative.parts and not path.is_symlink()
                and path.resolve().is_relative_to(source), 'Unsafe source path')
        require(sha(path) == digest, 'Scientific source changed: ' + name)


def verify_payload(bundle, parent):
    bundle = Path(bundle).resolve()
    data = next(d for d in parent['datasets'] if d['dataset_id'] == DATASET)
    paths = [(bundle / 'data' / Path(data['inherited_data_identity'][kind]['path']).name,
              data['inherited_data_identity'][kind]['sha256']) for kind in ('data', 'split_manifest')]
    ref = next(row for row in parent['reuse'] if row['dataset'] == DATASET and row['seed'] == 62)
    paths.extend((bundle / 'references/width4' / DATASET / '62' / name, digest) for name, digest in ref['file_sha256'].items())
    for path, digest in paths:
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(bundle), 'Unsafe/missing transferred input')
        require(sha(path) == digest, 'Transferred input/reference SHA changed: ' + str(path))


def load(bundle, verify_bytes=True):
    bundle = Path(bundle)
    parent, c, meta = (read(bundle / n) for n in ('parent_execution_contract.json', 'execution_contract.json', 'handoff_contract.json'))
    validate_contract(parent, c)
    approval, start = read(bundle / 'approval.json'), read(bundle / 'start_permit.json')
    parent_start, parent_approval = read(bundle / 'parent_start_permit.json'), read(bundle / 'parent_approval.json')
    pointer = read(bundle / 'parent_current.json')
    require(parent_approval['approved'] is True and parent_approval['contract_sha256'] == PARENT_SHA, 'Parent authority changed')
    require(parent_start['contract_sha256'] == pointer['contract_sha256'] == PARENT_SHA
            and parent_start['approval_sha256'] == canonical(parent_approval), 'Parent start/current changed')
    require(pointer['source_closure_sha256'] == SOURCE_SHA
            and pointer['hosts'] == {h: s['root'] for h, s in parent['hosts'].items()}, 'Parent ownership changed')
    require(approval['approved'] is True and approval['user_instruction'] and approval['hosts'] == ['5080', '5090']
            and approval['contract_sha256'] == canonical(c) and approval['parent_contract_sha256'] == PARENT_SHA
            and approval['transfer_job_ids'] == [j['id'] for j in target_jobs(parent)], 'Transfer authority changed')
    expected_start = {**parent_start, 'contract_sha256': canonical(c), 'approval_sha256': canonical(approval)}
    require(start == expected_start and start['deadline_unix'] - start['started_at_unix'] == 604800, 'Inherited deadline changed')
    require(meta == {'schema': NAME, 'parent_contract_sha256': PARENT_SHA, 'contract_sha256': canonical(c),
                     'source_closure_sha256': SOURCE_SHA, 'job_ids': approval['transfer_job_ids'],
                     'execution_host': '5080', 'origin_host': '5090', 'root': DESTINATION,
                     'adapter_sha256': sha(Path(__file__)), 'contract_diff': contract_diff(parent, c),
                     'parent_start_permit_sha256': canonical(parent_start), 'approval_sha256': canonical(approval),
                     'start_permit_sha256': canonical(start), 'selection_and_loss_changed': False,
                     'cross_gpu_efficiency_claim': False}, 'Handoff/adapter seal changed')
    if verify_bytes:
        verify_source(parent, bundle / 'source')
    return parent, c, meta


def live_deadline(bundle):
    start = read(Path(bundle) / 'start_permit.json')
    require(start['started_at_unix'] <= time.time() < start['deadline_unix'], 'Inherited campaign deadline expired')
    return start['deadline_unix']


def verify_native(bundle, parent, c):
    q = read(Path(bundle) / 'qualification/receipt.json')
    spec = c['hosts']['5080']
    require(q['status'] == 'passed' and q['host'] == '5080' and q['contract_sha256'] == canonical(c)
            and q['source_files_sha256'] == SOURCE_SHA and q['held_out_test_evaluated'] is False, 'Own native qualification missing')
    require(q['runtime']['gpu']['uuid'] == spec['gpu_uuid'] and all(q['runtime'].get(k) == v for k, v in spec['runtime_expected'].items()), 'Destination runtime changed')
    require(set(q['inputs']) == set(q['initialization']) == set(q['baselines']) == set(q['measurements']) == {DATASET}, 'Qualification dataset coverage changed')
    require(set(q['initialization'][DATASET]) == set(q['baselines'][DATASET]) == {'62'}
            and set(q['measurements'][DATASET]) == set(ARMS), 'Qualification seed/arm coverage changed')
    admitted = q['inputs'][DATASET]['receipt']
    data = next(d for d in parent['datasets'] if d['dataset_id'] == DATASET)
    require(admitted['held_out_materialized'] is False
            and admitted['populations'] == data['inherited_data_identity']['populations'], 'Native admitted population changed')
    initial = q['initialization'][DATASET]['62']
    baseline = q['baselines'][DATASET]['62']
    original = parent['baseline_replays'][DATASET]['62']
    require(set(initial) == {*ARMS, 'titantpp_history_mlp'} and baseline['status'] == 'passed'
            and baseline['checkpoint_sha256'] == original['checkpoint_sha256']
            and baseline['initial_state_sha256'] == initial['titantpp_history_mlp'] == original['initial_state_sha256'], 'Historical initialization/baseline seal changed')
    require(sha(Path(bundle) / 'qualification' / (DATASET + '__62_baseline_diagnostic.json')) == baseline['diagnostic_sha256'], 'Native baseline diagnostic changed')
    return q


def claim_payload(parent, c, job, qualification):
    return {'job': job, 'contract_sha256': PARENT_SHA, 'administrative_transfer': True,
            'destination': DESTINATION, 'destination_contract_sha256': canonical(c),
            'destination_qualification_sha256': canonical(qualification), 'user_approved': True,
            'no_source_fit_permitted': True}


def validate_reservation(parent, c, q, receipt):
    jobs = target_jobs(parent)
    require(receipt['status'] == 'reserved_no_source_fit' and receipt['parent_contract_sha256'] == PARENT_SHA
            and receipt['destination_contract_sha256'] == canonical(c) and receipt['destination'] == DESTINATION
            and receipt['qualification_sha256'] == canonical(q), 'Foreign/partial reservation')
    expected = [{'job_id': j['id'], 'claim_sha256': item['claim_sha256'], 'claim': claim_payload(parent, c, j, q)}
                for j, item in zip(jobs, receipt['reserved'])]
    require(len(receipt['reserved']) == 2 and receipt['reserved'] == expected, 'Reservation pair/claim changed')
    for item in receipt['reserved']:
        require(hashlib.sha256((json.dumps(item['claim'], indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()).hexdigest() == item['claim_sha256'], 'Reservation raw SHA changed')
    return receipt


def reserve(bundle, source_root):
    """One explicitly invoked reservation; no SSH, signal or supervisor edits."""
    parent, c, _ = load(bundle, verify_bytes=False)
    source = Path(source_root)
    require(str(source) == parent['hosts']['5090']['root'], 'Wrong source owner')
    require(read(source / 'execution_contract.json') == parent and read(source / 'start_permit.json') == read(Path(bundle) / 'parent_start_permit.json'), 'Live source contract/start changed')
    verify_source(parent, source / 'source')
    live_deadline(bundle)
    q = verify_native(bundle, parent, c)
    jobs = target_jobs(parent)
    path = source / RESERVATION_DIR / 'reservation.json'
    if path.exists():
        receipt = validate_reservation(parent, c, q, read(path))
        for item in receipt['reserved']:
            require(sha(source / 'claims' / (item['job_id'] + '.json')) == item['claim_sha256']
                    and not (source / 'run' / item['job_id']).exists(), 'Reserved source condition changed')
        return receipt
    state = read(source / 'status.json')
    require(state['status'] == 'running' and state['active_job']['id'] not in {j['id'] for j in jobs}, 'Source is not running a non-transferred fit')
    for job in jobs:
        require(not (source / 'run' / job['id']).exists(), 'Source fit already started')
        require(not (source / 'claims' / (job['id'] + '.json')).exists(), 'Source claim exists; explicit partial-state reconciliation required')
    receipt = {'status': 'reserving', 'parent_contract_sha256': PARENT_SHA, 'destination_contract_sha256': canonical(c),
               'destination': DESTINATION, 'qualification_sha256': canonical(q), 'source_active_job': state['active_job'],
               'source_supervisor_pid': state['supervisor_pid'], 'reserved': [], 'reserved_unix': time.time()}
    for job in jobs:
        claim = claim_payload(parent, c, job, q)
        claim_path = source / 'claims' / (job['id'] + '.json')
        write(claim_path, claim, exclusive=True)  # O_EXCL is the no-duplicate linearization point.
        receipt['reserved'].append({'job_id': job['id'], 'claim_sha256': sha(claim_path), 'claim': claim})
    for job in jobs:
        require(not (source / 'run' / job['id']).exists(), 'Source won fit race; destination must not launch')
    receipt['status'] = 'reserved_no_source_fit'
    validate_reservation(parent, c, q, receipt)
    write(path, receipt, exclusive=True)
    return receipt


def issue_permit(bundle):
    parent, c, _ = load(bundle)
    live_deadline(bundle)
    q = verify_native(bundle, parent, c)
    receipt = validate_reservation(parent, c, q, read(Path(bundle) / 'source_reservation.json'))
    permit = {'schema': 'titantpp_history_capacity_training_permit_v1', 'host': '5080',
              'contract_sha256': canonical(c), 'start_permit_sha256': canonical(read(Path(bundle) / 'start_permit.json')),
              'qualifications': {'5080': q}, 'parent_contract_sha256': PARENT_SHA,
              'source_reservation_sha256': canonical(receipt), 'transfer_job_ids': [j['id'] for j in target_jobs(parent)]}
    path = Path(bundle) / 'training_permit.json'
    if path.exists():
        require(read(path) == permit, 'Previously issued permit changed')
    else:
        write(path, permit, exclusive=True)
    return permit


def verify_permit(bundle, parent, c):
    q = verify_native(bundle, parent, c)
    receipt = validate_reservation(parent, c, q, read(Path(bundle) / 'source_reservation.json'))
    permit = read(Path(bundle) / 'training_permit.json')
    require(permit['parent_contract_sha256'] == PARENT_SHA and permit['source_reservation_sha256'] == canonical(receipt)
            and permit['transfer_job_ids'] == [j['id'] for j in target_jobs(parent)], 'Source reservation not bound to permit')
    return permit


@contextmanager
def original_scope(engine, assignments, seeds):
    current = engine.ASSIGNMENTS, engine.SEEDS
    engine.ASSIGNMENTS, engine.SEEDS = assignments, seeds
    try:
        yield
    finally:
        engine.ASSIGNMENTS, engine.SEEDS = current


def load_engine(bundle, parent, c, mode, seed=42):
    source = Path(c['hosts']['5080']['source_root']).resolve()
    require(Path(bundle).resolve() == Path(DESTINATION), 'Wrong destination root')
    require(Path.cwd().resolve() == source and Path(sys.executable).resolve() == Path(c['hosts']['5080']['python']).resolve(), 'Wrong native checkout/interpreter')
    expected = dict(c['hosts']['5080']['environment'])
    expected['PYTHONHASHSEED'] = str(seed)
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'Pinned process environment changed')
    verify_source(parent, source)
    verify_payload(bundle, parent)
    require(not any(n == 'paper' or n.startswith(('paper.', 'models.', 'simple_lab_test.', 'data_loader.')) for n in sys.modules), 'Scientific modules were imported before the frozen source was selected')
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_sealed_capacity_parent', source / ENGINE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return bind_engine(engine, parent, c, mode, seed)


def bind_engine(engine, parent, c, mode, seed=42):
    """Replace only operational validation and qualification coverage, never fit."""
    original_validate = engine.validate
    assignments, seeds = deepcopy(engine.ASSIGNMENTS), engine.SEEDS

    def validate(operational):
        validate_contract(parent, operational)
        with original_scope(engine, assignments, seeds):
            original_validate(parent)
        return operational

    engine.validate = validate
    engine.common.ENVIRONMENT = {**engine.common.ENVIRONMENT, 'PYTHONHASHSEED': str(seed)}
    if mode == 'qualify':
        engine.ASSIGNMENTS = {'5080': [DATASET], '5090': []}
        engine.SEEDS = (62,)
    validate(c)
    return engine


def admit_terminal(bundle, c, job):
    folder = Path(bundle) / 'run' / job['id']
    manifest = read(folder / 'terminal_manifest.json')
    require(manifest['status'] == 'complete' and manifest['scientific_success'] is True
            and manifest['job'] == job and manifest['contract_sha256'] == canonical(c), 'Foreign or unsuccessful terminal')
    for name, digest in manifest['files'].items():
        relative = Path(name)
        path = folder / relative
        require(not relative.is_absolute() and '..' not in relative.parts
                and not any(token in part.lower() for part in relative.parts for token in ('test', 'heldout', 'held_out'))
                and not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()), 'Unsafe terminal file')
        require(sha(path) == digest, 'Terminal original SHA changed')
    return {'terminal_manifest_sha256': sha(folder / 'terminal_manifest.json')}


def dispatch(bundle, parent, c, engine):
    root = Path(bundle)
    campaign_deadline = live_deadline(root)
    verify_permit(root, parent, c)
    engine.verify_training_permit(c, root)
    require(not engine.common.gpu_pids(c['hosts']['5080']), 'Destination GPU occupied')
    require(shutil.which('timeout') is not None, 'Independent native timeout unavailable')
    jobs = [j for j in c['jobs'] if j['id'] in {j['id'] for j in target_jobs(parent)}]
    results, child, job = {}, None, None
    with (root / 'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / 'supervisor.claim', {'pid': os.getpid(), 'contract_sha256': canonical(c), 'started_unix': time.time()}, exclusive=True)
        try:
            for job in jobs:
                engine.validate(c); live_deadline(root); verify_permit(root, parent, c)
                require(not engine.common.gpu_pids(c['hosts']['5080']), 'Other destination GPU owner detected')
                require(not (root / 'run' / job['id']).exists(), 'Run exists; no implicit retry/resume')
                deadline = min(campaign_deadline, time.time() + 36 * 3600)
                write(root / 'claims' / (job['id'] + '.json'), {'job': job, 'contract_sha256': canonical(c), 'deadline_unix': deadline}, exclusive=True)

                def pulse():
                    now = time.time(); require(now < deadline, 'Absolute deadline exhausted')
                    write(root / 'server_lease.json', {'job': job['id'], 'contract_sha256': canonical(c), 'issued_unix': now,
                                                     'expires_unix': min(now + 90, deadline), 'supervisor_pid': os.getpid()})
                    write(root / 'status.json', {'status': 'running', 'active_job': job, 'supervisor_pid': os.getpid(),
                                               'worker_group_pid': None if child is None else child.pid, 'completed': results,
                                               'updated_unix': now, 'deadline_unix': campaign_deadline,
                                               'condition_deadline_unix': deadline, 'mac_required': False,
                                               'parent_contract_sha256': PARENT_SHA, 'execution_host': '5080'})

                pulse()
                log = root / 'logs' / (job['id'] + '.log'); log.parent.mkdir(exist_ok=True)
                command = ['timeout', '--signal=TERM', '--kill-after=15s', str(max(1, int(deadline - time.time()))),
                           c['hosts']['5080']['python'], str(Path(__file__).resolve()), '--bundle', str(root),
                           '--mode', 'fit', '--job', job['id'], '--deadline', str(deadline)]
                with log.open('xb') as stream:
                    child = subprocess.Popen(command, cwd=c['hosts']['5080']['source_root'],
                                             env={**os.environ, 'PYTHONHASHSEED': str(job['seed'])}, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    while child.poll() is None:
                        pulse(); time.sleep(5)
                    require(child.returncode == 0, 'Worker exited; no automatic retry: ' + str(child.returncode))
                results[job['id']] = admit_terminal(root, c, job)
                child = None
            write(root / 'status.json', {'status': 'complete', 'completed': results, 'completed_conditions': len(results),
                                       'updated_unix': time.time(), 'mac_required': False, 'parent_contract_sha256': PARENT_SHA})
        except BaseException as exc:
            if child is not None:
                engine.stop_owned(child)
            failure = {'status': 'failed', 'type': type(exc).__name__, 'message': str(exc), 'active_job': job,
                       'completed': results, 'updated_unix': time.time(), 'automatic_retry': False,
                       'traceback': traceback.format_exc()}
            write(root / 'failure.json', failure, exclusive=True); write(root / 'status.json', failure)
            raise


def observe(bundle):
    """Pure local collector for one SSH invocation; named validation JSON only."""
    parent, c, meta = load(bundle)
    root = Path(bundle).resolve()
    require(root == Path(DESTINATION), 'Wrong observation root')
    result = {'host': '5080', 'root': str(root), 'observed_unix': time.time(), 'contract_sha256': canonical(c),
              'source_closure_sha256': SOURCE_SHA, 'files': {}, 'file_sha256': {}, 'runs': [],
              'handoff_contract': meta, 'original_binary_sha_checked': False, 'original_cpu_replay_executed': False}

    def record(path):
        if not path.exists():
            return None
        require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Unsafe observation path')
        raw = path.read_bytes(); require(len(raw) < 20 * 1024 * 1024, 'Oversized observation JSON')
        result['file_sha256'][str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for name in ('status.json', 'failure.json', 'progress.json', 'server_lease.json', 'supervisor.claim', 'source_reservation.json'):
        result['files'][name] = record(root / name)
    for job in c['jobs']:
        if job['id'] not in meta['job_ids']:
            continue
        folder = root / 'run' / job['id']
        leaf = folder / 'runs' / job['arm'] / c['quantity_variants'][job['arm']] / ('seed_' + str(job['seed']))
        row = {'job': job, 'status': record(folder / 'status.json'), 'manifest': record(folder / 'terminal_manifest.json')}
        for key, name in (('history', 'history.json'), ('timing', 'epoch_timing.json'), ('checkpoint', 'server_checkpoint_receipt.json'),
                          ('endpoints', 'endpoint_replays.json'), ('diagnostic', 'selected_train_validation_diagnostic.json')):
            row[key] = record(leaf / name)
        row['binary_sizes'] = {name: (leaf / name).stat().st_size if (leaf / name).is_file() and not (leaf / name).is_symlink() else 0
                               for name in ('last_epoch_state.pt', 'best_val_qty_rmse_model.pt')}
        result['runs'].append(row)
    for key, command in (('gpu', ['nvidia-smi', '--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader']),
                         ('compute', ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader'])):
        outcome = subprocess.run(command, capture_output=True, text=True, timeout=15)
        result[key] = {'returncode': outcome.returncode, 'stdout': outcome.stdout}
    outcome = subprocess.run(['ps', '-eo', 'pid,ppid,pgid,args'], capture_output=True, text=True, timeout=15)
    result['ps_returncode'] = outcome.returncode
    result['processes'] = [line for line in outcome.stdout.splitlines() if str(root) in line and 'python3 -' not in line]
    return result


def observe_source(bundle, source_root):
    """Supplement an original5090 snapshot inside the same SSH invocation."""
    parent, c, meta = load(bundle, verify_bytes=False)
    root = Path(source_root)
    require(str(root) == parent['hosts']['5090']['root'] and read(root / 'execution_contract.json') == parent, 'Wrong source snapshot ownership')
    verify_source(parent, root / 'source')
    result = {'parent_contract_sha256': PARENT_SHA, 'destination_contract_sha256': canonical(c),
              'observed_unix': time.time(), 'files': {}, 'file_sha256': {}, 'reservation_verified': False,
              'intentional_queue_stop_verified': False, 'source_effective_complete': False}
    def record(name):
        path = root / name
        if not path.exists():
            return None
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Unsafe source evidence')
        raw = path.read_bytes(); require(len(raw) < 20 * 1024 * 1024, 'Oversized source evidence')
        result['file_sha256'][name] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw); result['files'][name] = value
        return value
    receipt = record(RESERVATION_DIR + '/reservation.json')
    claims = {j['id']: record('claims/' + j['id'] + '.json') for j in target_jobs(parent)}
    if receipt is None:
        return result
    q = verify_native(bundle, parent, c)
    validate_reservation(parent, c, q, receipt)
    for item in receipt['reserved']:
        require(claims[item['job_id']] == item['claim']
                and result['file_sha256']['claims/' + item['job_id'] + '.json'] == item['claim_sha256']
                and not (root / 'run' / item['job_id']).exists(), 'Source reservation changed/fit started')
    result['reservation_verified'] = True
    result['transferred_job_ids'] = meta['job_ids']
    failure = record('failure.json')
    first = target_jobs(parent)[0]
    if failure is None or failure.get('type') != 'FileExistsError' or failure.get('active_job') != first:
        return result
    require(str(root / 'claims' / (first['id'] + '.json')) in failure.get('message', ''), 'Source failure is not the reserved claim')
    expected = [j for j in parent['jobs'] if j['host'] == '5090' and j['seed'] in (42, 52)]
    require(set(failure['completed']) == {j['id'] for j in expected}, 'Source remaining completion unconfirmed')
    for job in expected:
        name = 'run/' + job['id'] + '/terminal_manifest.json'
        manifest = record(name)
        require(manifest and manifest['status'] == 'complete' and manifest['scientific_success'] is True
                and manifest['job'] == job and manifest['contract_sha256'] == PARENT_SHA
                and result['file_sha256'][name] == failure['completed'][job['id']]['terminal_manifest_sha256'], 'Source completed manifest not admitted')
    result['intentional_queue_stop_verified'] = True
    result['source_effective_complete'] = True
    result['original_binary_sha_checked'] = False
    result['original_cpu_replay_executed'] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--mode', choices=('qualify', 'reserve', 'permit', 'dispatch', 'fit', 'observe', 'observe-source'), required=True)
    parser.add_argument('--source-root'); parser.add_argument('--job'); parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.mode == 'reserve':
        require(args.source_root is not None, 'Source root required')
        output = reserve(args.bundle, args.source_root)
    elif args.mode == 'observe':
        output = observe(args.bundle)
    elif args.mode == 'observe-source':
        require(args.source_root is not None, 'Source root required')
        output = observe_source(args.bundle, args.source_root)
    elif args.mode == 'permit':
        output = issue_permit(args.bundle)
    else:
        parent, c, _ = load(args.bundle)
        live_deadline(args.bundle)
        job = next((j for j in c['jobs'] if j['id'] == args.job), None)
        if args.mode == 'fit':
            require(job is not None and job['id'] in {j['id'] for j in target_jobs(parent)} and job['host'] == '5080', 'Only the approved transferred pair may fit')
            require(args.deadline is not None and time.time() < args.deadline <= min(live_deadline(args.bundle), time.time() + 36 * 3600), 'Fit deadline enlarged/expired')
            claim = read(Path(args.bundle) / 'claims' / (job['id'] + '.json'))
            require(claim == {'job': job, 'contract_sha256': canonical(c), 'deadline_unix': args.deadline}, 'Fit is not supervisor-claimed')
            verify_permit(args.bundle, parent, c)
        engine = load_engine(args.bundle, parent, c, args.mode, job['seed'] if args.mode == 'fit' else 42)
        if args.mode == 'qualify':
            engine.qualify(c, '5080', Path(args.bundle))
        elif args.mode == 'dispatch':
            dispatch(args.bundle, parent, c, engine)
        else:
            engine.run_fit(c, '5080', Path(args.bundle), job, args.deadline)
        return
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
