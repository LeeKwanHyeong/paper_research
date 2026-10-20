"""CPU/stdlib checks of the bounded train-only profiling supervisor."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from paper.scripts import titantpp_shared_efficiency_benchmark as bench


@pytest.fixture
def contract(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'ROOT', tmp_path)
    monkeypatch.setattr(bench, 'SOURCE', tmp_path / 'source')
    (tmp_path / 'source').mkdir()
    model = dict(hidden_dim=64, lambda_log_qty=1., lambda_tail=0.,
                 time_head_lr_multiplier=1., quantity_variant='count_only_log_regression',
                 time_head_mode='heteroscedastic_lognormal_duration')
    datasets = [dict(dataset_id=name, loader=dict(max_seq_len=length, batch_size=128),
                     model=dict(model), optimizer=dict(lr=.001, grad_clip=1.,
                                                     weight_decay=.01, name='AdamW'))
                for name, length in bench.DATASETS.items()]
    return dict(models=[bench.MLP, bench.MAC], datasets=datasets,
                host=dict(root=str(tmp_path), source_root=str(tmp_path / 'source'),
                          python=sys.executable, gpu_uuid=bench.GPU_UUID,
                          runtime_expected=dict(python='3.12.13', torch='2.11.0+cu130', cudnn=92000)),
                environment={}, source_files={}, operational_files={},
                scope=dict(train_only=True, accuracy_evaluation=False, checkpoint_writes=False,
                           full_epochs=False, full_fits=False, held_out=False, validation=False),
                protocol=dict(seed=42, batch_size=128, repetitions=3, modes=['train','eval'],
                              warmup_batches_per_worker=5, measured_batches_per_worker=32,
                              workers=36, parallel_workers=1, automatic_retry=False))


def test_exact_36_paired_jobs_and_counterbalanced_order(contract):
    jobs = bench.jobs(contract)
    assert len(jobs) == len(set(jobs)) == 36
    assert bench.TARGETS == 4736
    assert sum(37 for _, _, mode, _ in jobs if mode == 'train') == 666
    for dataset in bench.DATASETS:
        for mode in ('train', 'eval'):
            for rep in (1, 2, 3):
                arms = [arm for ds, arm, m, r in jobs if (ds,m,r)==(dataset,mode,rep)]
                assert arms == ([bench.MAC,bench.MLP] if rep == 2 else [bench.MLP,bench.MAC])


@pytest.mark.parametrize('mutation', [
    lambda c: c['models'].__setitem__(0, 'titantpp_history_mlp'),
    lambda c: c['host'].__setitem__('gpu_uuid', '5080'),
    lambda c: c['host']['runtime_expected'].__setitem__('cudnn', 91900),
    lambda c: c['host']['runtime_expected'].__setitem__('python', '3.12.12'),
    lambda c: c['datasets'][0]['loader'].__setitem__('max_seq_len', 256),
    lambda c: c['datasets'][0]['model'].__setitem__('lambda_log_qty', 0.),
    lambda c: c['scope'].__setitem__('validation', True),
    lambda c: c['scope'].__setitem__('checkpoint_writes', True),
    lambda c: c['protocol'].__setitem__('parallel_workers', 2),
    lambda c: c['protocol'].__setitem__('workers', 24),
    lambda c: c['protocol'].__setitem__('warmup_batches_per_worker', 4),
    lambda c: c['protocol'].__setitem__('automatic_retry', True),
    lambda c: c.__setitem__('max_seconds', 1801),
    lambda c: c.__setitem__('max_seconds', float('nan')),
    lambda c: c.__setitem__('child_timeout_seconds', 1801),
])
def test_unapproved_contract_rejected_before_ml_import(contract, mutation):
    mutation(contract)
    with pytest.raises(RuntimeError):
        bench.validate_contract(contract)


def test_optional_budget_is_bounded(contract):
    assert bench.validate_contract(contract) == (1800., 1800.)
    contract['max_seconds'] = 1200
    contract['child_timeout_seconds'] = 300
    assert bench.validate_contract(contract) == (1200., 300.)


def test_approval_and_both_source_hashes_verified(contract, tmp_path):
    source = tmp_path / 'source' / 'model.py'
    source.write_text('frozen')
    operational = tmp_path / 'benchmark.py'
    operational.write_text('operation')
    contract['source_files'] = {'model.py': bench.sha(source)}
    contract['source_closure_sha256'] = bench.canonical(contract['source_files'])
    contract['operational_files'] = {'benchmark.py': bench.sha(operational)}
    bench.write(tmp_path / 'contract.json', contract)
    bench.write(tmp_path / 'approval.json', {'contract_sha256': bench.canonical(contract), 'approved': True})
    assert bench.verify() == contract
    source.write_text('tampered')
    with pytest.raises(RuntimeError, match='Frozen source mismatch'):
        bench.verify()
    source.write_text('frozen')
    operational.write_text('tampered')
    with pytest.raises(RuntimeError, match='Operational source mismatch'):
        bench.verify()
    bench.write(tmp_path / 'approval.json', {'contract_sha256': 'foreign', 'approved': True})
    with pytest.raises(RuntimeError, match='Approval identity'):
        bench.verify()


def test_train_predicate_precedes_only_materialization():
    events = []
    class Lazy:
        def filter(self, predicate):
            assert predicate == ('eq', 'train')
            events.append('filter_train'); return self
        def collect(self):
            assert events == ['scan', 'filter_train']
            events.append('collect'); return self
        def sort(self, columns):
            assert columns == ['oper_part_no','seq']
            events.append('sort'); return self
    class Column:
        def __eq__(self, other): return ('eq', other)
    def scan(path):
        assert path == 'original.parquet'
        events.append('scan'); return Lazy()
    def col(name):
        assert name == 'chronological_split'; return Column()
    bench.train_only_frame(types.SimpleNamespace(scan_parquet=scan,col=col), 'original.parquet')
    assert events == ['scan','filter_train','collect','sort']


@pytest.mark.parametrize('marker', ['claim.exclusive','claim.json','status.json','results/old.json','batches/old.pt','qualification/old.json','child_exits/old.json'])
def test_existing_root_claim_or_outputs_cannot_retry(contract, tmp_path, marker):
    target = tmp_path / marker
    target.parent.mkdir(exist_ok=True)
    target.write_text('{}')
    with pytest.raises(RuntimeError, match='Existing'):
        bench.reject_existing_execution()


def test_owner_contract_parent_and_deadline_all_required(contract, tmp_path, monkeypatch):
    claim = dict(pid=os.getppid(), deadline_unix=bench.time.time()+30,
                 contract_sha256=bench.canonical(contract))
    bench.write(tmp_path / 'claim.json', claim)
    assert bench.assert_owner(contract) == claim
    monkeypatch.setattr(bench.os, 'getppid', lambda: -1)
    with pytest.raises(RuntimeError, match='Wrong supervisor'):
        bench.assert_owner(contract)


def test_real_nonzero_child_exit_is_retained_without_retry(contract, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'verify', lambda: contract)
    monkeypatch.setattr(bench, 'gpu_snapshot', lambda: dict(gpu=bench.GPU_UUID,compute='',unix=0))
    original_popen = subprocess.Popen
    launches = []
    def failing_child(command, **kwargs):
        launches.append(command)
        return original_popen([sys.executable, '-c', "print('synthetic retained stdout',flush=True); raise RuntimeError('synthetic child failure')"], **kwargs)
    monkeypatch.setattr(bench.subprocess, 'Popen', failing_child)
    assert bench.controller() == 1
    assert len(launches) == 1
    manifest = bench.read(tmp_path / 'terminal_manifest.json')
    assert manifest['status'] == 'failed_no_automatic_retry'
    assert manifest['controller_os_exit_observed'] is False
    assert manifest['automatic_retry'] is False
    [child] = manifest['actual_child_exits']
    assert child['returncode'] == 1
    assert child['actual_child_exit_observed'] is True
    assert child['pid_absent_after_wait'] is True
    assert child['timed_out'] is False
    assert 'synthetic retained stdout' in (tmp_path / child['log']).read_text()
    assert 'Traceback' in (tmp_path / child['log']).read_text()
    assert (tmp_path / 'error.json').is_file()
    with pytest.raises(RuntimeError, match='Existing'):
        bench.controller()
    assert len(launches) == 1


def test_safe_timeout_terminates_only_spawned_child(contract, tmp_path, monkeypatch):
    contract['child_timeout_seconds'] = .1
    monkeypatch.setattr(bench, 'verify', lambda: contract)
    monkeypatch.setattr(bench, 'gpu_snapshot', lambda: dict(gpu=bench.GPU_UUID,compute='',unix=0))
    original_popen = subprocess.Popen
    def slow_child(command, **kwargs):
        return original_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
    monkeypatch.setattr(bench.subprocess, 'Popen', slow_child)
    assert bench.controller() == 1
    [child] = bench.read(tmp_path / 'terminal_manifest.json')['actual_child_exits']
    assert child['timed_out'] is True
    assert child['actual_child_exit_observed'] is True
    assert child['pid_absent_after_wait'] is True
    assert child['returncode'] != 0


def test_complete_mocked_serial_schedule_requires_all41_exits(contract, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'verify', lambda: contract)
    monkeypatch.setattr(bench, 'gpu_snapshot', lambda: dict(gpu=bench.GPU_UUID,compute='',unix=0))
    monkeypatch.setattr(bench, 'process_absent', lambda pid: True)
    (tmp_path / 'supervise.py').write_text('frozen supervisor')
    contract['operational_files']['supervise.py'] = bench.sha(tmp_path / 'supervise.py')
    launched = []
    class FinishedChild:
        returncode = 0
        def __init__(self, command, **kwargs):
            self.pid = 100000 + len(launched)
            assert kwargs['start_new_session'] is True
            assert kwargs['cwd'] == tmp_path / 'source'
            launched.append(command)
            if command[2] == 'worker':
                args = dict(zip(command[3::2], command[4::2]))
                label = f"{args['--dataset']}__{args['--arm']}__{args['--mode']}__r{args['--rep']}"
                bench.write(tmp_path / 'results' / f'{label}.json',
                            dict(status='success',contract_sha256=bench.canonical(contract)))
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
    monkeypatch.setattr(bench.subprocess, 'Popen', FinishedChild)
    assert bench.controller() == 0
    assert len(launched) == 41
    assert [c[2] for c in launched[:5]] == ['qualify']*4 + ['prepare']
    status = bench.read(tmp_path / 'status.json')
    assert status['status'] == 'complete'
    assert status['child_exit_count'] == 41
    assert status['worker_exit_count'] == len(status['completed']) == 36
    assert status['gpu_final_empty'] is True
    manifest = bench.read(tmp_path / 'terminal_manifest.json')
    assert len(manifest['actual_child_exits']) == 41
    assert manifest['files']['supervise.py'] == contract['operational_files']['supervise.py']
    for relative, checksum in manifest['files'].items():
        assert bench.sha(tmp_path / relative) == checksum


def test_preparation_cannot_read_real_data_without_qualification(contract, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'verify', lambda: contract)
    bench.write(tmp_path / 'claim.json', dict(pid=os.getppid(),
                deadline_unix=bench.time.time()+30,contract_sha256=bench.canonical(contract)))
    with pytest.raises(FileNotFoundError, match='qualification'):
        bench.prepare()


def test_exact_frozen_builders_receive_original_configuration(contract, monkeypatch):
    calls = []
    raw = types.ModuleType('paper.scripts.run_titantpp_shared_history_raw9')
    def shared_builder(data, arm):
        calls.append((data,arm))
        return 'shared-model', {'explicit_raw9_tail_and_metadata_defaults': True}
    raw.build_model = shared_builder
    mac = types.ModuleType('paper.scripts.titans_mac_observed_time_runtime')
    mac_calls = []
    def mac_builder(data):
        mac_calls.append(data)
        return 'frozen-mac-model', {}
    mac.build_model = mac_builder
    monkeypatch.setitem(sys.modules, raw.__name__, raw)
    monkeypatch.setitem(sys.modules, mac.__name__, mac)
    data = copy.deepcopy(contract['datasets'][0])
    data['statistics'] = dict(train_log_mean=1.2,train_log_std=.8)
    data['model'].update(time_scale=3.,time_sigma_floor=.001,
                         time_initial_location=.2,time_initial_scale=.8,
                         time_observation_contract=dict(mode='positive_integer_round_clamp_v1',unit='month',top_code=None))
    assert bench.model_for(data,bench.MLP) == 'shared-model'
    assert len(calls) == 1
    assert calls[0][0] is data and calls[0][1] == bench.MLP
    assert bench.model_for(data,bench.MAC) == 'frozen-mac-model'
    assert mac_calls == [data] and mac_calls[0] is data
    with pytest.raises(RuntimeError, match='Unknown arm'):
        bench.model_for(data,'titantpp_history_mlp')
