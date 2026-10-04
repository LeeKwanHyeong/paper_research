from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper/scripts'))
import observe_titantpp_capacity_handoff_hourly as observer


def contracts():
    parent = json.loads((observer.PARENT / 'execution_contract.json').read_text())
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
           'compute': {'returncode': 0, 'stdout': ''}}
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
        'actual_owned_worker_pids': [], 'supervisor_pids': [], 'server_terminal_verified': False})
    assert observer.analyse_source(raw, parent, c, q)['approved_handoff_boundary_verified'] is False
    raw['files']['failure.json']['traceback'] = 'FileExistsError: ' + observer.handoff.target_jobs(parent)[0]['id']
    assert observer.analyse_source(raw, parent, c, q)['approved_handoff_boundary_verified'] is True
