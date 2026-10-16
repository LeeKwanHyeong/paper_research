"""Synthetic seed and shared-queue contracts; no GPU/model/data execution."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from paper.scripts import titantpp_cross_fusion_seed_extension_runtime as r


@pytest.fixture
def contract(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.py').write_text('unchanged\n')
    hosts = {host: {'root': str(tmp_path / host), 'parent_root': str(tmp_path / ('parent' + host)),
        'source_root': str(source), 'python': '/python', 'gpu_uuid': 'GPU-' + host,
        'runtime_expected': {'python': '3.12.13'}, 'environment': {}} for host in ('5080', '5090')}
    assignment = {'taxi': ['5080', '5080'], 'raf': ['5090', '5090'], 'inter': ['5080', '5090']}
    jobs = [{'id': f'{d}__{arm}__seed{seed}', 'dataset': d, 'arm': arm, 'host': assignment[d][i], 'seed': seed}
        for d in assignment for i, arm in enumerate(('pointwise', 'cross')) for seed in (52, 62)]
    inventory = {'model.py': r.sha_file(source / 'model.py')}
    return {'schema': r.SCHEMA, 'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
        'jobs': jobs, 'datasets': [{'dataset_id': d, 'quantity_boundaries_all_train_rows': []} for d in assignment],
        'hosts': hosts, 'architecture': {'arms': ['pointwise', 'cross']},
        'training': {'maximum_epochs': 300, 'minimum_epochs': 40, 'patience': 40, 'batch_size': 128,
                     'monitor': 'validation_raw_quantity_rmse'},
        'limits': {'fit_wall_seconds': 129600, 'queue_wall_seconds': 259200},
        'source': {'files': inventory, 'files_sha256': r.sha_json(inventory)}, 'input_files': inventory}


def bind_parent(c, host):
    parent = deepcopy(c)
    parent['schema'] = r.parent_runtime.SCHEMA
    parent['limits'] = {'fit_wall_seconds': 129600}
    parent['jobs'] = [{**j, 'seed': 42, 'id': j['id'].replace('seed52', 'seed42')}
                      for j in parent['jobs'] if j['seed'] == 52]
    for h in parent['hosts']:
        parent['hosts'][h]['root'] = parent['hosts'][h]['parent_root']
    root = Path(c['hosts'][host]['parent_root'])
    r.write(root / 'execution_contract.json', parent)
    q = {'status': 'passed', 'host': host, 'contract_sha256': r.sha_json(parent),
         'parallel_memory_admission': {'passed': True, 'maximum_workers': 3},
         'runtime': {'python': '3.12.13', 'gpu': {'uuid': 'GPU-' + host}}}
    r.write(root / 'qualification.json', q)
    c['parent'] = {'contract_sha256': r.sha_file(root / 'execution_contract.json'),
        'canonical_sha256': r.sha_json(parent), 'source_files_sha256': parent['source']['files_sha256'],
        'qualification_sha256': {host: r.sha_file(root / 'qualification.json')}}
    return parent


def test_matrix(contract):
    for host in ('5080', '5090'):
        assert len(r.validate_contract(contract, host)) == 6
    for mutation in ('seed', 'id', 'host'):
        broken = deepcopy(contract)
        broken['jobs'][0][mutation] = {'seed': 42, 'id': 'oops', 'host': '5090'}[mutation]
        with pytest.raises(ValueError):
            r.validate_contract(broken, '5080')


@pytest.mark.parametrize('occupied,new,gpu,slots', [
    (3, [], [], 0), (1, [22], [11, 22], 1), (0, [22, 23], [], 1), (0, [], [], 3)])
def test_shared_slots_includes_launching(occupied, new, gpu, slots):
    state = {'occupied_job_ids': list(range(occupied)), 'live_pids': [11] if occupied else []}
    assert r.available_slots(state, new, gpu) == slots


def test_unknown_gpu_and_overcapacity():
    with pytest.raises(ValueError, match='Unidentified'):
        r.available_slots({'occupied_job_ids': [], 'live_pids': []}, [], [99])
    with pytest.raises(ValueError, match='concurrency'):
        r.available_slots({'occupied_job_ids': [1, 2, 3], 'live_pids': []}, [22], [])


def test_parent_binding_and_changed_assignment(contract):
    bind_parent(contract, '5080')
    r.verify_parent_binding(contract, '5080')
    for kind in ('assignment', 'source', 'gpu', 'input', 'qualification'):
        c = deepcopy(contract)
        if kind == 'assignment':
            c['jobs'][0]['host'], c['jobs'][4]['host'] = c['jobs'][4]['host'], c['jobs'][0]['host']
        elif kind == 'source':
            c['source']['files']['model.py'] = 'bad'
        elif kind == 'gpu':
            c['hosts']['5080']['gpu_uuid'] = 'bad'
        elif kind == 'input':
            c['input_files']['other'] = 'bad'
        else:
            c['parent']['qualification_sha256']['5080'] = 'bad'
        with pytest.raises(ValueError):
            r.verify_parent_binding(c, '5080')


def test_parent_race_reserves_slots_and_failure_blocks(contract):
    p = bind_parent(contract, '5080')
    root = Path(contract['hosts']['5080']['parent_root'])
    state = r.parent_occupancy(p, '5080')
    assert len(state['occupied_job_ids']) == 3 and state['live_pids'] == []
    job = next(j for j in p['jobs'] if j['host'] == '5080')
    r.write(root / 'run' / job['id'] / 'worker_exit.json', {'job_id': job['id'], 'actual_os_returncode': 0, 'timeout': False})
    assert len(r.parent_occupancy(p, '5080')['occupied_job_ids']) == 3
    r.write(root / 'run' / job['id'] / 'fit_receipt.json', {'status': 'complete', 'job': job, 'contract_sha256': r.sha_json(p)})
    assert len(r.parent_occupancy(p, '5080')['occupied_job_ids']) == 2
    r.write(root / 'status.json', {'actual_worker_exits': {'bad': {'actual_os_returncode': 1}}})
    with pytest.raises(ValueError, match='failed'):
        r.parent_occupancy(p, '5080')


def test_identity_rejects_pid_reuse(contract):
    job, target = contract['jobs'][0], contract['hosts']['5080']
    worker = {'pid': 5, 'pgid': 5, 'start_ticks': 100, 'job': job}
    identity = {'pid': 5, 'pgid': 5, 'start_ticks': 100, 'cwd': target['source_root'],
        'argv': ['/python', '-m', r.MODULE, '--contract', target['root'] + '/execution_contract.json',
                 '--host', '5080', '--mode', 'fit', '--job-id', job['id']]}
    r.check_worker(identity, worker, job, target, r.MODULE)
    identity['start_ticks'] = 101
    with pytest.raises(ValueError, match='reused'):
        r.check_worker(identity, worker, job, target, r.MODULE)


@pytest.mark.parametrize('seed', [52, 62])
def test_fit_passes_actual_seed(contract, monkeypatch, seed):
    job = next(j for j in contract['jobs'] if j['seed'] == seed and j['host'] == '5080')
    target = contract['hosts']['5080']
    monkeypatch.chdir(target['source_root'])
    monkeypatch.setenv('PYTHONHASHSEED', str(seed))
    r.write(Path(target['root']) / 'start.json', {'host': '5080', 'contract_sha256': r.sha_json(contract)})
    monkeypatch.setattr(r, 'proc_identity', lambda pid: {'start_ticks': 100})
    monkeypatch.setattr(r, 'runtime_check', lambda t: {'checked': True})
    runner = ModuleType('paper.scripts.run_titantpp_cross_fusion')
    runner.ARMS, runner.VARIANT = ['pointwise', 'cross'], 'same'
    runner.install_hooks = lambda: None
    runner.training_args = lambda *a: SimpleNamespace(epochs=300, min_epochs=40, early_stopping_patience=40,
        batch_size=128, checkpoint_monitor='validation_raw_quantity_rmse', max_train_batches=None,
        max_val_batches=None, force_rerun=False)
    runner.time_interface = lambda *a, **k: {}
    base = ModuleType('paper.scripts.run_multilag_detail_execution')
    base.prepare_admitted_data = lambda *a, **k: (None, {'same': True})
    training = ModuleType('paper.scripts.count_aware_tpp_backbone.training')
    captured = []
    def train(**kwargs):
        captured.append((kwargs['seed'], kwargs['args'].seed))
        output = Path(target['root']) / 'run' / job['id']
        for name in ('best.pt', 'last_epoch_state.pt'):
            (output / name).write_bytes(b'synthetic')
        r.write(output / 'history.json', {'history': [{'epoch': 1}]})
        r.write(output / 'summary.json', {})
        return ({'status': 'success', 'seed': seed, 'held_out_test_evaluated': False,
            'checkpoint_path': str(output / 'best.pt'), 'best_epoch': 1,
            'best_val_qty_rmse': 1, 'best_val_qty_mae': 1}, None, None)
    training.train_one = train
    training.earliest_strict_minimum = lambda h, metric_key: h[0]
    package = ModuleType('paper.scripts.count_aware_tpp_backbone')
    package.training = training
    for module in (runner, base, training, package):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    r.fit(contract, '5080', job['id'])
    assert captured == [(seed, seed)]


def test_reuse_blocked_before_parent_access(contract, monkeypatch):
    target = contract['hosts']['5080']
    monkeypatch.chdir(target['source_root'])
    r.write(Path(target['root']) / 'start.json', {})
    with pytest.raises(ValueError, match='Existing attempt'):
        r.admit(contract, '5080')


def test_proc_exit_between_reads_returns_none(tmp_path):
    proc = tmp_path / '123'
    proc.mkdir()
    fields = ['S', '1', '123'] + ['0'] * 17
    fields[19] = '100'
    (proc / 'stat').write_text('123 (worker) ' + ' '.join(fields))
    # cmdline disappeared after stat: a normal exit, not an identity failure.
    assert r.proc_identity(123, tmp_path) is None


def test_failed_new_child_prevents_remaining_launches(contract, monkeypatch):
    target = contract['hosts']['5080']
    root = Path(target['root'])
    monkeypatch.chdir(target['source_root'])
    r.write(root / 'qualification.json', {'status': 'passed', 'contract_sha256': r.sha_json(contract)})
    monkeypatch.setattr(r, 'verify_files', lambda *a: None)
    monkeypatch.setattr(r, 'verify_parent_binding', lambda *a: ({}, {}))
    monkeypatch.setattr(r, 'parent_occupancy', lambda *a: {'occupied_job_ids': [], 'live_pids': []})
    monkeypatch.setattr(r, 'gpu_snapshot', lambda *a: {'compute_pids': []})
    monkeypatch.setattr(r.time, 'sleep', lambda *a: None)
    launched = []
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(launched)
            self.command = command
            self.exited = False
            launched.append(self)
        def poll(self):
            self.exited = True
            return 1 if self.pid == 100 else 0
    monkeypatch.setattr(r.subprocess, 'Popen', Child)
    def identity(pid):
        child = next(x for x in launched if x.pid == pid)
        if child.exited:
            return None
        return {'pid': pid, 'pgid': pid, 'start_ticks': pid, 'argv': child.command, 'cwd': target['source_root']}
    monkeypatch.setattr(r, 'proc_identity', identity)
    assert r.dispatch(contract, '5080', root / 'execution_contract.json') == 1
    assert len(launched) == 3  # Remaining three never start after first failure.
    receipt = r.read(root / 'final_receipt.json')
    assert receipt['status'] == 'failed' and len(receipt['pending_job_ids']) == 3
    assert len(receipt['actual_worker_exits']) == 3
    assert receipt['owned_worker_pids_absent'] is True
    assert (root / 'dispatch_failure.json').exists()
