from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
import observe_titantpp_capacity_handoff_hourly as observer


def contracts():
    parent = json.loads((observer.B / 'parent_execution_contract.json').read_text())
    return parent, observer.handoff.derive_contract(parent)


def reserved_snapshot():
    parent, c = contracts()
    gpu = parent['hosts']['5090']['gpu_uuid']
    raw = {'host': '5090', 'root': parent['hosts']['5090']['root'], 'observed_unix': 1791110400,
           'contract_sha256': observer.handoff.PARENT_SHA, 'source_closure_sha256': observer.handoff.SOURCE_SHA,
           'files': {'status.json': {'status': 'running', 'completed': {}}}, 'file_sha256': {},
           'runs': [{'job': j, 'status': None, 'manifest': None, 'history': None, 'timing': None,
                     'checkpoint': None, 'endpoints': None, 'diagnostic': None, 'binary_sizes': {}}
                    for j in parent['jobs'] if j['host'] == '5090'],
           'processes': [], 'ps_returncode': 0, 'gpu': {'returncode': 0, 'stdout': gpu + ', RTX 5090'},
           'compute': {'returncode': 0, 'stdout': ''},
           'transferred_run_folder_exists': {j['id']: False for j in observer.handoff.target_jobs(parent)}}
    q = {'native_qualification': 'fixture'}
    items = []
    for job in observer.handoff.target_jobs(parent):
        claim = observer.handoff.claim_payload(parent, c, job, q)
        digest = observer.hashlib.sha256((json.dumps(claim, indent=2, ensure_ascii=False) + '\n').encode()).hexdigest()
        name = 'claims/' + job['id'] + '.json'
        raw['files'][name] = deepcopy(claim)
        raw['file_sha256'][name] = digest
        items.append({'job_id': job['id'], 'claim_sha256': digest, 'claim': claim})
    raw['files']['handoff_seed62/reservation.json'] = {
        'status': 'reserved_no_source_fit', 'parent_contract_sha256': observer.handoff.PARENT_SHA,
        'destination_contract_sha256': observer.handoff.canonical(c), 'destination': observer.handoff.DESTINATION,
        'qualification_sha256': observer.handoff.canonical(q), 'reserved': items}
    return raw, parent, c, q


def test_transferred_conditions_are_excluded_from_source_queue():
    raw, parent, c, q = reserved_snapshot()
    result = observer.analyse_source(raw, parent, c, q)
    assert len(result['rows']) == 4
    assert result['transferred_count'] == 2
    assert all(row['job']['seed'] != 62 for row in result['rows'])
    assert result['status_counts']['pending'] == 4


def test_live_claim_tampering_blocks_transfer_count():
    raw, parent, c, q = reserved_snapshot()
    raw['files']['claims/' + observer.handoff.target_jobs(parent)[0]['id'] + '.json']['no_source_fit_permitted'] = False
    with pytest.raises(ValueError, match='Live source claim'):
        observer.analyse_source(raw, parent, c, q)


def test_source_started_fit_blocks_duplicate_owner():
    raw, parent, c, q = reserved_snapshot()
    next(r for r in raw['runs'] if r['job']['seed'] == 62)['status'] = {'status': 'training'}
    with pytest.raises(ValueError, match='Transferred source fit'):
        observer.analyse_source(raw, parent, c, q)


def test_sidecar_worker_requires_exact_root_and_gpu():
    parent, c = contracts()
    spec = c['hosts']['5080']; job = observer.handoff.target_jobs(parent)[0]
    command = f"{spec['python']} {spec['root']}/control/capacity_handoff_runtime.py --bundle {spec['root']} --mode fit --job {job['id']}"
    raw = {'processes': ['103 102 102 ' + command], 'ps_returncode': 0,
           'gpu': {'returncode': 0, 'stdout': spec['gpu_uuid'] + ', NVIDIA RTX 5080, 99, 1000, 16000'},
           'compute': {'returncode': 0, 'stdout': '103, ' + spec['gpu_uuid'] + ', python, 1000'}}
    rows, fits, supervisors, pids, available = observer.target_process_evidence(raw, c, '5080')
    assert available and pids == {103} and fits[0]['job_id'] == job['id']
    raw['processes'][0] = raw['processes'][0].replace('--bundle ' + spec['root'], '--bundle /foreign')
    assert observer.target_process_evidence(raw, c, '5080')[1] == []
    raw['compute']['stdout'] = '103, GPU-foreign, python, 1000'
    assert observer.target_process_evidence(raw, c, '5080')[3] == set()


def test_unrelated_failure_is_not_expected_handoff_boundary(monkeypatch):
    raw, parent, c, q = reserved_snapshot()
    raw['files']['failure.json'] = {'status': 'failed', 'traceback': 'RuntimeError: unrelated'}
    monkeypatch.setattr(observer.old, 'analyse', lambda *a: {
        'status_counts': {'completed': 4}, 'GPU_UUID_verified': True,
        'actual_owned_worker_pids': [], 'supervisor_pids': [], 'server_terminal_verified': False, 'rows': []})
    assert observer.analyse_source(raw, parent, c, q)['approved_handoff_boundary_verified'] is False
    first = observer.handoff.target_jobs(parent)[0]
    raw['files']['failure.json'].update(type='FileExistsError', active_job=first,
        message='File exists: ' + parent['hosts']['5090']['root'] + '/claims/' + first['id'] + '.json')
    assert observer.analyse_source(raw, parent, c, q)['approved_handoff_boundary_verified'] is True


@pytest.mark.parametrize(('returncode', 'stderr', 'category', 'summary'), [
    (255, 'ssh: connect to host private-host port 22: Connection timed out',
     'ssh_connection_failed', 'Connection timed out'),
    (255, 'ssh: connect to host 100.118.62.81 port 22: Operation timed out',
     'ssh_connection_failed', 'Connection timed out'),
    (1, 'Traceback (most recent call last):\nValueError: Out of range float values are not JSON compliant: NaN',
     'remote_observer_failed', 'JSON non-finite value'),
    (255, 'Traceback (most recent call last):\nRuntimeError: hidden detail',
     'remote_observer_failed', 'RuntimeError'),
    (255, 'unrecognized private error', 'ssh_or_remote_command_failed', 'No allowlisted'),
])
def test_failed_snapshot_retains_safe_diagnostics(returncode, stderr, category, summary):
    parent, c = contracts()
    calls = []
    secret = 'DO_NOT_PERSIST_CREDENTIAL'

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], returncode, secret,
            stderr + '\nAuthorization: Bearer ' + secret + '\npassword=' + secret + '\n' + 'x' * 10000)

    with pytest.raises(observer.ObservationCommandError) as error:
        observer.snapshot_once('5080', parent, c, runner)
    evidence = error.value.evidence
    assert len(calls) == 1
    assert evidence['returncode'] == returncode
    assert evidence['category'] == category
    assert summary in evidence['stderr_summary']
    assert len(evidence['stderr_summary']) <= 512
    assert secret not in json.dumps(evidence) + str(error.value)
    assert 'private-host' not in json.dumps(evidence)
    assert '100.118.62.81' not in json.dumps(evidence)


def test_snapshot_timeout_does_not_expose_command_or_partial_output():
    parent, c = contracts()
    secret = 'TIMEOUT_SECRET'
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(['ssh', secret], 90, output=secret.encode(),
                                        stderr=('Permission denied: ' + secret).encode())

    with pytest.raises(observer.ObservationCommandError) as error:
        observer.snapshot_once('5080', parent, c, runner)
    assert len(calls) == 1
    assert error.value.evidence['category'] == 'observation_timeout'
    assert error.value.evidence['returncode'] is None
    assert error.value.evidence['timeout_seconds'] == 90
    assert secret not in json.dumps(error.value.evidence) + str(error.value)


def test_run_once_persists_failure_evidence_without_training_failure(tmp_path, monkeypatch, capsys):
    parent, c = contracts()
    monkeypatch.setattr(observer, 'B', tmp_path)
    monkeypatch.setattr(observer.old, 'load_contract', lambda *args: parent)
    monkeypatch.setattr(observer.handoff, 'load', lambda *args: (parent, c, {}))
    monkeypatch.setattr(observer.handoff, 'verify_native', lambda *args: {})
    monkeypatch.setattr(observer.handoff, 'validate_reservation', lambda *args: None)
    (tmp_path / 'source_reservation.json').write_text('{}')
    calls = []

    def runner(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0], 255, 'STDOUT_SECRET',
                                           'Permission denied (publickey). token=STDERR_SECRET')

    assert observer.run_once('5080', runner) == 1
    assert len(calls) == 1
    receipt = json.loads((tmp_path / 'hourly_monitor/latest_5080.json').read_text())
    analysis = json.loads(Path(receipt['analysis']).read_text())
    evidence = json.loads(Path(receipt['observation_error']).read_text())
    assert receipt['observation_error_evidence'] == analysis['observation_error_evidence'] == evidence
    assert receipt['SSH_invocations'] == 1
    assert receipt['observed_unix'] is None and receipt['snapshot'] is None
    assert receipt['status_counts']['failed'] == 0 and receipt['status_counts']['unknown'] == 2
    assert receipt['status'] == 'observation_failed_not_training_failure'
    assert receipt['server_terminal_verified'] is False
    persisted = ''.join(p.read_text() for p in tmp_path.rglob('*') if p.is_file()) + capsys.readouterr().out
    assert 'STDOUT_SECRET' not in persisted and 'STDERR_SECRET' not in persisted
    assert 'Permission denied' in Path(receipt['report']).read_text()


@pytest.mark.parametrize('host', ['5080', '5090'])
def test_successful_snapshot_is_unchanged_and_uses_one_ssh(host):
    parent, c = contracts()
    snapshot = {'host': host, 'observed_unix': 1791110400}
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, json.dumps(snapshot), '')

    assert observer.snapshot_once(host, parent, c, runner) == snapshot
    assert len(calls) == 1


def test_terminal_cache_preserves_original_time_and_validates_sha(tmp_path):
    snapshot = tmp_path / 'host_snapshot.json'
    raw = {'observed_unix': 1791110400}
    snapshot.write_text(json.dumps(raw))
    receipt = tmp_path / 'latest_5080.json'
    receipt.write_text(json.dumps({'server_terminal_verified': True, 'snapshot': str(snapshot),
        'snapshot_sha256': observer.hashlib.sha256(snapshot.read_bytes()).hexdigest()}))

    def analyser(value):
        return dict(value, server_terminal_verified=True)

    recovered, analysis = observer.cached_terminal(receipt, analyser)
    assert recovered == raw and analysis['observed_unix'] == raw['observed_unix']
    assert analysis['reused_terminal_evidence'] is True
    snapshot.write_text(json.dumps({'observed_unix': 1791110401}))
    with pytest.raises(ValueError, match='Terminal cache SHA changed'):
        observer.cached_terminal(receipt, analyser)
