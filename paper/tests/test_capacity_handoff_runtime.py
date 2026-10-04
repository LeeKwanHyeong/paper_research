"""Offline contract/race/deadline tests; no GPU, SSH or ignored artifacts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/capacity_handoff_runtime.py'
spec = importlib.util.spec_from_file_location('capacity_handoff_under_test', SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    dest, source = tmp_path / 'destination', tmp_path / 'source5090'
    monkeypatch.setattr(r, 'DESTINATION', str(dest))
    files = {'scientific/' + str(i) + '.py': __import__('hashlib').sha256(str(i).encode()).hexdigest() for i in range(114)}
    monkeypatch.setattr(r, 'SOURCE_SHA', r.canonical(files))
    hosts = {h: {'root': str(tmp_path / ('old' + h)), 'source_root': str(tmp_path / ('old' + h) / 'source'),
                 'output_dir': str(tmp_path / ('old' + h) / 'run'), 'tmux': 'original' + h,
                 'assigned_datasets': [r.DATASET] if h == '5090' else ['yellow_trip_hourly', 'raf_spare_parts'],
                 'environment': {'PYTHONHASHSEED': '42'}, 'runtime_expected': {'torch': '2.11.0+cu130'},
                 'python': '/pinned/python', 'gpu_uuid': 'GPU-' + h} for h in ('5080', '5090')}
    hosts['5090']['root'] = str(source)
    jobs = [{'id': f'{d}__{seed}__{arm}', 'dataset': d, 'seed': seed, 'arm': arm, 'host': host}
            for host, datasets in [('5080', ['yellow_trip_hourly', 'raf_spare_parts']), ('5090', [r.DATASET])]
            for d in datasets for seed in (42, 52, 62) for arm in r.ARMS]
    data = {'dataset_id': r.DATASET, 'model': {'lambda_log_qty': 1.0},
            'inherited_data_identity': {'data': {'path': '/original/data.parquet', 'sha256': 'data'},
                                       'split_manifest': {'path': '/original/split.json', 'sha256': 'split'},
                                       'populations': {'train': {'target_count': 10}, 'validation': {'target_count': 7}}}}
    parent = {'hosts': hosts, 'jobs': jobs, 'datasets': [data], 'dataset_sha256': {r.DATASET: r.canonical(data)},
              'reuse': [{'host': '5090', 'dataset': r.DATASET, 'seed': 62, 'remote_run': '/original/ref',
                         'arm': 'titantpp_history_mlp', 'file_sha256': {'exposure.json': 'exposure'}}],
              'baseline_replays': {r.DATASET: {'62': {'checkpoint': '/original/checkpoint.pt', 'checkpoint_sha256': 'checkpoint',
                                                   'initial_state_sha256': 'historic', 'metrics': {'qty_rmse': 1.0}}}},
              'source': {'files': files, 'files_sha256': r.SOURCE_SHA},
              'limits': {'total_wall_seconds': 604800, 'per_condition_seconds': 129600, 'workers_per_host': 1, 'automatic_retry': False},
              'training': {'batch_size': 128, 'maximum_epochs': 300, 'minimum_epochs': 40, 'patience': 40},
              'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
              'quantity_variants': dict.fromkeys(r.ARMS, 'count_only_log_regression')}
    monkeypatch.setattr(r, 'PARENT_SHA', r.canonical(parent))
    c = r.derive_contract(parent)
    parent_approval = {'approved': True, 'contract_sha256': r.PARENT_SHA}
    parent_start = {'contract_sha256': r.PARENT_SHA, 'approval_sha256': r.canonical(parent_approval),
                    'started_at_unix': time.time() - 10, 'deadline_unix': time.time() - 10}
    parent_start['deadline_unix'] = parent_start['started_at_unix'] + 604800
    pointer = {'contract_sha256': r.PARENT_SHA, 'source_closure_sha256': r.SOURCE_SHA,
               'hosts': {h: s['root'] for h, s in parent['hosts'].items()}}
    approval = {'approved': True, 'user_instruction': 'Explicitly migrate this pair', 'hosts': ['5080', '5090'],
                'contract_sha256': r.canonical(c), 'parent_contract_sha256': r.PARENT_SHA,
                'transfer_job_ids': [j['id'] for j in r.target_jobs(parent)]}
    start = {**parent_start, 'contract_sha256': r.canonical(c), 'approval_sha256': r.canonical(approval)}
    meta = {'schema': r.NAME, 'parent_contract_sha256': r.PARENT_SHA, 'contract_sha256': r.canonical(c),
            'source_closure_sha256': r.SOURCE_SHA, 'job_ids': approval['transfer_job_ids'], 'execution_host': '5080',
            'origin_host': '5090', 'root': r.DESTINATION, 'adapter_sha256': r.sha(SCRIPT),
            'contract_diff': r.contract_diff(parent, c), 'parent_start_permit_sha256': r.canonical(parent_start),
            'approval_sha256': r.canonical(approval), 'start_permit_sha256': r.canonical(start),
            'selection_and_loss_changed': False, 'cross_gpu_efficiency_claim': False}
    for name, value in [('parent_execution_contract.json', parent), ('execution_contract.json', c),
                        ('parent_approval.json', parent_approval), ('parent_start_permit.json', parent_start),
                        ('parent_current.json', pointer), ('approval.json', approval), ('start_permit.json', start),
                        ('handoff_contract.json', meta)]:
        r.write(dest / name, value)
    for directory in (dest / 'source', source / 'source'):
        for i, name in enumerate(files):
            path = directory / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(i))
    r.write(source / 'execution_contract.json', parent)
    r.write(source / 'start_permit.json', parent_start)
    r.write(source / 'status.json', {'status': 'running', 'active_job': next(j for j in jobs if j['host'] == '5090' and j['seed'] == 42), 'supervisor_pid': 999})
    diagnostic = dest / 'qualification' / (r.DATASET + '__62_baseline_diagnostic.json')
    r.write(diagnostic, {'validation_only': True})
    q = {'status': 'passed', 'host': '5080', 'contract_sha256': r.canonical(c), 'source_files_sha256': r.SOURCE_SHA,
         'held_out_test_evaluated': False, 'runtime': {'gpu': {'uuid': 'GPU-5080'}, 'torch': '2.11.0+cu130'},
         'inputs': {r.DATASET: {'receipt': {'held_out_materialized': False, 'populations': data['inherited_data_identity']['populations']}}},
         'initialization': {r.DATASET: {'62': {**dict.fromkeys(r.ARMS, 'candidate'), 'titantpp_history_mlp': 'historic'}}},
         'baselines': {r.DATASET: {'62': {'status': 'passed', 'checkpoint_sha256': 'checkpoint', 'initial_state_sha256': 'historic', 'diagnostic_sha256': r.sha(diagnostic)}}},
         'measurements': {r.DATASET: dict.fromkeys(r.ARMS, {})}}
    r.write(dest / 'qualification/receipt.json', q)
    return dest, source, parent, c, q


def reserve_copy(prepared):
    dest, source, parent, c, q = prepared
    receipt = r.reserve(dest, source)
    r.write(dest / 'source_reservation.json', receipt)
    return receipt


def test_exact_derivation_preserves_science_and_parent(prepared):
    dest, source, parent, c, q = prepared
    original = deepcopy(parent)
    r.validate_contract(parent, c); r.load(dest)
    assert parent == original
    assert c['source'] == parent['source'] and c['training'] == parent['training'] and c['limits'] == parent['limits']
    assert len([x for x in r.contract_diff(parent, c) if x['path'].startswith('/jobs/')]) == 2
    assert c['datasets'][0]['model'] == parent['datasets'][0]['model']


@pytest.mark.parametrize('kind', ['loss', 'batch', 'deadline', 'runtime', 'seed', 'source'])
def test_non_operational_changes_rejected(prepared, kind):
    _, _, parent, c, _ = prepared
    bad = deepcopy(c)
    if kind == 'loss': bad['datasets'][0]['model']['lambda_log_qty'] = 2
    if kind == 'batch': bad['training']['batch_size'] = 64
    if kind == 'deadline': bad['limits']['total_wall_seconds'] += 1
    if kind == 'runtime': bad['hosts']['5080']['runtime_expected']['torch'] = 'other'
    if kind == 'seed': bad['jobs'][-1]['seed'] = 52
    if kind == 'source': bad['source']['files']['scientific/0.py'] = 'changed'
    with pytest.raises(RuntimeError, match='Non-whitelisted'): r.validate_contract(parent, bad)


def test_source_bytes_and_adapter_seal_are_mandatory(prepared):
    dest, _, _, _, _ = prepared
    (dest / 'source/scientific/0.py').write_text('changed')
    with pytest.raises(RuntimeError, match='Scientific source changed'): r.load(dest)
    meta = r.read(dest / 'handoff_contract.json'); meta['adapter_sha256'] = 'changed'; r.write(dest / 'handoff_contract.json', meta)
    with pytest.raises(RuntimeError, match='adapter seal'): r.load(dest, verify_bytes=False)


def test_qualification_filter_restores_full_parent_validation(prepared):
    _, _, parent, c, _ = prepared
    called = []
    assignments = {'5080': ['yellow_trip_hourly', 'raf_spare_parts'], '5090': [r.DATASET]}
    engine = SimpleNamespace(ASSIGNMENTS=deepcopy(assignments), SEEDS=(42, 52, 62), common=SimpleNamespace(ENVIRONMENT={}))
    original_fit = lambda *args: None
    engine.run_fit = original_fit
    def original_validate(value):
        assert value == parent and engine.ASSIGNMENTS == assignments and engine.SEEDS == (42, 52, 62)
        called.append(value)
    engine.validate = original_validate
    r.bind_engine(engine, parent, c, 'qualify')
    assert engine.ASSIGNMENTS == {'5080': [r.DATASET], '5090': []} and engine.SEEDS == (62,)
    engine.validate(c)
    assert len(called) == 2 and engine.SEEDS == (62,) and engine.run_fit is original_fit


def test_reservation_idempotent_without_touching_running_fit(prepared):
    dest, source, parent, c, q = prepared
    state = (source / 'status.json').read_bytes()
    lease = source / 'server_lease.json'; lease.write_text('existing running lease')
    first = r.reserve(dest, source); second = r.reserve(dest, source)
    assert first == second and len(first['reserved']) == 2
    assert (source / 'status.json').read_bytes() == state and lease.read_text() == 'existing running lease'
    assert all(r.read(source / 'claims' / (j['id'] + '.json'))['no_source_fit_permitted'] for j in r.target_jobs(parent))
    assert not (dest / 'training_permit.json').exists()


def test_source_wins_claim_race_no_destination_permit(prepared, monkeypatch):
    dest, source, parent, c, q = prepared
    real_write = r.write
    first = source / 'claims' / (r.target_jobs(parent)[0]['id'] + '.json')
    def raced(path, value, exclusive=False):
        if Path(path) == first:
            real_write(first, {'source_owned': True}, exclusive=True)
        return real_write(path, value, exclusive)
    monkeypatch.setattr(r, 'write', raced)
    with pytest.raises(FileExistsError): r.reserve(dest, source)
    assert not (source / r.RESERVATION_DIR / 'reservation.json').exists()
    with pytest.raises(FileNotFoundError): r.issue_permit(dest)


def test_partial_reservation_kept_no_automatic_rollback_or_retry(prepared, monkeypatch):
    dest, source, parent, _, _ = prepared
    real_write = r.write
    second = source / 'claims' / (r.target_jobs(parent)[1]['id'] + '.json')
    def interrupted(path, value, exclusive=False):
        if Path(path) == second: raise OSError('Interrupted after first reservation')
        return real_write(path, value, exclusive)
    monkeypatch.setattr(r, 'write', interrupted)
    with pytest.raises(OSError): r.reserve(dest, source)
    assert (source / 'claims' / (r.target_jobs(parent)[0]['id'] + '.json')).exists()
    assert not (source / r.RESERVATION_DIR / 'reservation.json').exists()
    with pytest.raises(RuntimeError, match='partial-state reconciliation'): r.reserve(dest, source)
    assert not (dest / 'training_permit.json').exists()


def test_started_source_condition_and_foreign_gpu_rejected(prepared):
    dest, source, parent, c, q = prepared
    (source / 'run' / r.target_jobs(parent)[0]['id']).mkdir(parents=True)
    with pytest.raises(RuntimeError, match='already started'): r.reserve(dest, source)
    q['runtime']['gpu']['uuid'] = 'GPU-5090'; r.write(dest / 'qualification/receipt.json', q)
    with pytest.raises(RuntimeError, match='runtime changed'): r.verify_native(dest, parent, c)


def test_permit_binds_both_raw_claims_and_own_native_receipt(prepared):
    dest, _, parent, c, q = prepared
    receipt = reserve_copy(prepared)
    permit = r.issue_permit(dest)
    assert permit == r.issue_permit(dest) and permit['qualifications'] == {'5080': q}
    receipt['reserved'][1]['claim_sha256'] = 'wrong'; r.write(dest / 'source_reservation.json', receipt)
    with pytest.raises(RuntimeError, match='raw SHA'): r.verify_permit(dest, parent, c)


def test_inherited_absolute_deadline_cannot_reset(prepared):
    dest, _, _, _, _ = prepared
    start = r.read(dest / 'start_permit.json'); start['deadline_unix'] += 1; r.write(dest / 'start_permit.json', start)
    with pytest.raises(RuntimeError, match='Inherited deadline changed'): r.load(dest)


def test_terminal_raw_sha_and_safe_scope(prepared):
    dest, _, _, c, _ = prepared
    job = c['jobs'][-1]; folder = dest / 'run' / job['id']; folder.mkdir(parents=True)
    (folder / 'history.json').write_text('{}')
    manifest = {'status': 'complete', 'scientific_success': True, 'job': job, 'contract_sha256': r.canonical(c),
                'files': {'history.json': r.sha(folder / 'history.json')}}
    r.write(folder / 'terminal_manifest.json', manifest)
    assert r.admit_terminal(dest, c, job)['terminal_manifest_sha256'] == r.sha(folder / 'terminal_manifest.json')
    (folder / 'history.json').write_text('changed')
    with pytest.raises(RuntimeError, match='SHA changed'): r.admit_terminal(dest, c, job)
    manifest['files'] = {'heldout/result.json': 'wrong'}; r.write(folder / 'terminal_manifest.json', manifest)
    with pytest.raises(RuntimeError, match='Unsafe terminal'): r.admit_terminal(dest, c, job)


def test_dispatch_runs_only_pair_and_uses_native_bounds(prepared, monkeypatch):
    dest, _, parent, c, _ = prepared
    reserve_copy(prepared); r.issue_permit(dest)
    commands = []
    class Child:
        pid = 100
        returncode = 0
        def poll(self): return 0
    def launch(command, **kwargs):
        commands.append((command, kwargs)); return Child()
    monkeypatch.setattr(r.subprocess, 'Popen', launch)
    monkeypatch.setattr(r.shutil, 'which', lambda name: '/usr/bin/timeout')
    monkeypatch.setattr(r, 'admit_terminal', lambda root, contract, job: {'terminal_manifest_sha256': job['id']})
    engine = SimpleNamespace(validate=lambda x: r.validate_contract(parent, x),
                             verify_training_permit=lambda *args: True, common=SimpleNamespace(gpu_pids=lambda spec: set()))
    r.dispatch(dest, parent, c, engine)
    assert [command[command.index('--job') + 1] for command, _ in commands] == [j['id'] for j in r.target_jobs(parent)]
    assert all(command[:3] == ['timeout', '--signal=TERM', '--kill-after=15s'] and 0 < int(command[3]) <= 129600 for command, _ in commands)
    assert all(kw['env']['PYTHONHASHSEED'] == '62' and kw['start_new_session'] for _, kw in commands)
    assert r.read(dest / 'status.json')['completed_conditions'] == 2


def test_observe_after_deadline_is_read_only_and_reports_two_jobs(prepared, monkeypatch):
    dest, _, parent, c, _ = prepared
    monkeypatch.setattr(r.time, 'time', lambda: r.read(dest / 'start_permit.json')['deadline_unix'] + 1)
    calls = []
    def runner(command, **kwargs):
        calls.append(command); return SimpleNamespace(returncode=0, stdout='')
    monkeypatch.setattr(r.subprocess, 'run', runner)
    before = sorted(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file())
    result = r.observe(dest)
    assert len(result['runs']) == 2 and result['contract_sha256'] == r.canonical(c)
    assert result['original_binary_sha_checked'] is False and result['original_cpu_replay_executed'] is False
    assert len(calls) == 3 and sorted(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()) == before


def test_source_snapshot_missing_or_changed_reservation(prepared):
    dest, source, _, _, _ = prepared
    assert r.observe_source(dest, source)['reservation_verified'] is False
    reserve_copy(prepared)
    assert r.observe_source(dest, source)['reservation_verified'] is True
    claim = next((source / 'claims').glob('*.json')); claim.write_text('{}')
    with pytest.raises(RuntimeError, match='reservation changed'): r.observe_source(dest, source)


def test_source_intentional_stop_requires_all_four_admitted_originals(prepared):
    dest, source, parent, _, _ = prepared
    reserve_copy(prepared)
    expected = [j for j in parent['jobs'] if j['host'] == '5090' and j['seed'] in (42, 52)]
    completed = {}
    for job in expected:
        manifest = source / 'run' / job['id'] / 'terminal_manifest.json'
        r.write(manifest, {'status': 'complete', 'scientific_success': True, 'job': job, 'contract_sha256': r.PARENT_SHA})
        completed[job['id']] = {'terminal_manifest_sha256': r.sha(manifest)}
    first = r.target_jobs(parent)[0]
    failure = {'type': 'FileExistsError', 'active_job': first, 'message': str(source / 'claims' / (first['id'] + '.json')), 'completed': completed}
    r.write(source / 'failure.json', failure)
    result = r.observe_source(dest, source)
    assert result['intentional_queue_stop_verified'] and result['source_effective_complete']
    completed[expected[0]['id']]['terminal_manifest_sha256'] = 'wrong'; r.write(source / 'failure.json', failure)
    with pytest.raises(RuntimeError, match='not admitted'): r.observe_source(dest, source)


def test_source_real_failure_is_not_administrative_completion(prepared):
    dest, source, parent, _, _ = prepared
    reserve_copy(prepared)
    r.write(source / 'failure.json', {'type': 'RuntimeError', 'active_job': r.target_jobs(parent)[0], 'message': 'fit failed'})
    result = r.observe_source(dest, source)
    assert result['reservation_verified'] and not result['intentional_queue_stop_verified'] and not result['source_effective_complete']
