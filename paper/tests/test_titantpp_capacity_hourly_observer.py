"""Synthetic read-only observer checks; no SSH, GPU, checkpoints or held-out data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import Mock
import pytest

SPEC = importlib.util.spec_from_file_location('capacity_hourly', Path(__file__).resolve().parents[1] / 'scripts/observe_titantpp_history_capacity_hourly.py')
o = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(o)


def sha(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def endpoint(rmse, mae, nll, path):
    def cell(count):
        return {'count': count, 'qty_sse': rmse ** 2 * count, 'qty_rmse': rmse, 'qty_mae': mae, 'time_nll': nll}
    quantity = [dict(cell(1), bin=i) for i in range(5)]
    history = [dict(cell(n), bin=i) for i, n in enumerate((1, 2, 2))]
    return dict(cell(5), evaluation_scope='validation_only', held_out_test_evaluated=False,
        time_metric='recorded_positive_integer_time_nll', quantity_boundaries=[1, 2, 3, 4],
        state_sha256='a' * 64, checkpoint_path=path, quantity_cells=quantity,
        history_cells=history, tail=cell(1), body=cell(3))


@pytest.fixture
def evidence(monkeypatch):
    jobs = [{'id': f'dataset{i}__42__width8', 'host': h, 'dataset': f'dataset{i}', 'arm': 'width8', 'seed': 42}
            for h, count in [('5080', 12), ('5090', 6)] for i in range(count)]
    # Distinct host roots are part of the frozen identity even when fixture IDs overlap.
    c = {'jobs': jobs, 'hosts': {h: {'root': '/remote/' + h, 'source_root': '/remote/' + h + '/source',
          'python': '/python/' + h, 'gpu_uuid': 'GPU-' + h} for h in ('5080', '5090')},
         'quantity_variants': {'width8': 'count_only_log_regression'},
         'training': {'minimum_epochs': 40, 'maximum_epochs': 300, 'patience': 40},
         'datasets': [{'dataset_id': f'dataset{i}', 'quantity_boundaries_all_train_rows': [1, 2, 3, 4],
           'inherited_data_identity': {'populations': {'validation': {'target_count': 5, 'identity': 'original'}}}}
           for i in range(12)]}
    monkeypatch.setattr(o, 'CONTRACT', o.canonical(c)); monkeypatch.setattr(o, 'SOURCE', 's' * 64)
    host = '5080'
    s = {'host': host, 'root': c['hosts'][host]['root'], 'observed_unix': time.time(),
         'contract_sha256': o.CONTRACT, 'source_closure_sha256': o.SOURCE,
         'files': {'status.json': {'status': 'complete', 'completed': {}}}, 'file_sha256': {}, 'runs': [],
         'processes': [], 'ps_returncode': 0, 'gpu': {'returncode': 0, 'stdout': 'GPU-5080, name, 0, 0, 100'},
         'compute': {'returncode': 0, 'stdout': ''}}
    for job in [j for j in jobs if j['host'] == host]:
        folder = 'run/' + job['id']; prefix = 'runs/width8/count_only_log_regression/seed_42/'
        history = {'history': [{'epoch': i, 'val_qty_rmse': 2 if i == 1 else 3,
                    'val_qty_mae': 1 if i == 1 else 1.1, 'val_time_nll': .5 if i == 1 else .6} for i in range(1, 41)]}
        timing = {'epochs': [{'epoch': i, 'elapsed_seconds': 10} for i in range(1, 41)]}
        e = {'status': 'complete', 'job': job, 'best_epoch': 1, 'completed_epochs': 40,
             'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
             'selected': endpoint(2, 1, .5, s['root'] + '/' + folder + '/' + prefix + 'best_val_qty_rmse_model.pt'),
             'last': endpoint(3, 1.1, .6, s['root'] + '/' + folder + '/' + prefix + 'last_epoch_state.pt')}
        diagnostic = {'dataset': job['dataset'], 'seed': 42, 'model': 'width8', 'epoch': 1,
              'provenance': {'state_sha256': 'a' * 64, 'held_out_test_evaluated': False,
                 'selection_unchanged': True, 'new_training': False},
              'validation': {'split': 'validation', 'full_population': True, 'population': {'target_count': 5, 'identity': 'original'},
                  'qty_rmse': 2, 'qty_mae': 1, 'time_nll': .5}}
        cp = {'epoch': 40, 'contract_sha256': o.CONTRACT, 'files': {'history.json': sha(history), 'epoch_timing.json': sha(timing)}}
        records = {'history.json': history, 'epoch_timing.json': timing, 'server_checkpoint_receipt.json': cp,
                   'endpoint_replays.json': e, 'selected_train_validation_diagnostic.json': diagnostic}
        manifest = {'status': 'complete', 'scientific_success': True, 'job': job, 'contract_sha256': o.CONTRACT,
             'files': {prefix + n: sha(v) for n, v in records.items()}}
        manifest['files'].update({prefix + n: 'b' * 64 for n in ('last_epoch_state.pt', 'best_val_qty_rmse_model.pt')})
        s['file_sha256'].update({folder + '/' + prefix + n: sha(v) for n, v in records.items()})
        s['file_sha256'][folder + '/terminal_manifest.json'] = sha(manifest)
        s['files']['status.json']['completed'][job['id']] = {'terminal_manifest_sha256': sha(manifest)}
        s['runs'].append({'job': job, 'manifest': manifest, 'status': {'status': 'complete', 'job': job},
            'history': history, 'timing': timing, 'checkpoint': cp, 'endpoints': e, 'diagnostic': diagnostic,
            'binary_sizes': {'last_epoch_state.pt': 100, 'best_val_qty_rmse_model.pt': 100}})
    return c, s


def test_all_twelve_terminal_records_verified_without_binary_hash(evidence):
    c, s = evidence; a = o.analyse(s, c, '5080')
    assert a['status_counts']['completed'] == 12 and a['server_terminal_verified'] is True
    assert a['binary_checkpoint_hashed'] is False


@pytest.mark.parametrize('bad', ['manifest_sha', 'small_sha', 'replay_metric', 'replay_scope', 'population'])
def test_declared_completion_with_bad_evidence_is_unknown(evidence, bad):
    c, s = evidence; raw = s['runs'][0]; job = raw['job']
    if bad == 'manifest_sha': s['file_sha256']['run/' + job['id'] + '/terminal_manifest.json'] = 'wrong'
    elif bad == 'small_sha': raw['manifest']['files'][next(iter(raw['manifest']['files']))] = 'wrong'
    elif bad == 'replay_metric': raw['endpoints']['selected']['qty_mae'] = 99
    elif bad == 'replay_scope': raw['endpoints']['last']['held_out_test_evaluated'] = True
    else: raw['diagnostic']['validation']['population']['target_count'] = 99
    a = o.analyse(s, c, '5080')
    assert a['rows'][0]['state'] == 'unknown' and a['status_counts']['completed'] == 11
    assert not a['server_terminal_verified'] and a['rows'][0]['issue']


def training_snapshot(evidence):
    c, s = evidence; raw = s['runs'][0]; raw['manifest'] = None
    job = raw['job']; status = s['files']['status.json']; status['status'] = 'running'
    del status['completed'][job['id']]
    status.update(active_job=job, supervisor_pid=3, worker_group_pid=7)
    raw['status'] = {'status': 'training', 'job': job}
    base = '/python/5080 paper/scripts/run_titantpp_history_capacity_campaign.py --contract /remote/5080/execution_contract.json --host 5080 --mode '
    s['processes'] = ['3 1 3 ' + base + 'dispatch', '7 3 7 timeout ' + base + 'fit --job ' + job['id'],
                      '15 7 7 ' + base + 'fit --job ' + job['id']]
    s['compute']['stdout'] = '15, GPU-5080, python, 100'
    return c, s


def test_training_uses_process_ancestry_and_actual_GPU(evidence):
    c, s = training_snapshot(evidence); a = o.analyse(s, c, '5080')
    assert a['rows'][0]['state'] == 'training' and a['actual_owned_gpu_pids'] == [15]
    assert a['declared_workers_match_processes']
    assert a['rows'][0]['eta']['remaining_seconds'] == 10


def test_saved_training_without_owned_process_is_unknown(evidence):
    c, s = training_snapshot(evidence); s['processes'] = []; s['compute']['stdout'] = ''
    a = o.analyse(s, c, '5080')
    assert a['rows'][0]['state'] == 'unknown' and a['rows'][0]['eta'] is None


def test_GPU_UUID_mismatch_does_not_confirm_training(evidence):
    c, s = training_snapshot(evidence); s['gpu']['stdout'] = 'GPU-foreign, name, 0, 0, 100'
    assert o.analyse(s, c, '5080')['rows'][0]['state'] == 'unknown'


def test_ssh_failure_records_this_failure_without_reusing_past(tmp_path, monkeypatch, evidence):
    c, _ = evidence; monkeypatch.setattr(o, 'load_contract', lambda bundle: c)
    run = Mock(return_value=subprocess.CompletedProcess([], 255, '', 'private provider details'))
    assert o.run_once('5080', tmp_path, run) == 1 and run.call_count == 1
    latest = o.read(tmp_path / 'hourly_monitor/latest_5080.json')
    assert latest['observed_unix'] is None and latest['SSH_invocations'] == 1
    assert not latest['fresh_observation'] and not latest['reused_terminal_evidence']
    assert 'private provider details' not in Path(latest['analysis']).read_text()


def test_terminal_reuse_preserves_actual_time_without_SSH(tmp_path, monkeypatch, evidence):
    c, s = evidence; monkeypatch.setattr(o, 'load_contract', lambda bundle: c)
    first = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps(s), ''))
    assert o.run_once('5080', tmp_path, first) == 0 and first.call_count == 1
    later = Mock(side_effect=AssertionError('No second SSH for verified terminal host'))
    assert o.run_once('5080', tmp_path, later) == 0; later.assert_not_called()
    latest = o.read(tmp_path / 'hourly_monitor/latest_5080.json')
    assert latest['observed_unix'] == s['observed_unix'] and latest['reused_terminal_evidence']
    assert not latest['fresh_observation']


def test_changed_contract_blocks_before_ssh(tmp_path):
    o.write(tmp_path / 'execution_contract.json', {'changed': True})
    o.write(tmp_path / 'current.json', {'bundle': str(tmp_path), 'contract_sha256': 'wrong'})
    run = Mock(side_effect=AssertionError('No SSH with foreign contract'))
    assert o.run_once('5080', tmp_path, run) == 1; run.assert_not_called()


def test_started_fit_failure_is_counted_once_not_for_every_condition(evidence):
    c, s = training_snapshot(evidence)
    s['files']['status.json']['status'] = 'failed'
    s['files']['failure.json'] = {'status': 'failed', 'active_job': s['runs'][0]['job']}
    s['processes'] = []; s['compute']['stdout'] = ''
    a = o.analyse(s, c, '5080')
    assert a['rows'][0]['state'] == 'failed' and a['status_counts']['failed'] == 1
    assert not a['server_terminal_verified']


def test_prestart_host_failure_is_not_a_failed_fit(evidence):
    c, s = training_snapshot(evidence); raw = s['runs'][0]
    raw['history'] = None; raw['status'] = None; raw['checkpoint'] = None; raw['timing'] = None
    s['files']['status.json']['status'] = 'failed'
    s['files']['failure.json'] = {'status': 'failed', 'active_job': raw['job']}
    s['processes'] = []; s['compute']['stdout'] = ''
    a = o.analyse(s, c, '5080')
    assert a['rows'][0]['state'] == 'unknown' and a['status_counts']['failed'] == 0


def test_heldout_manifest_metadata_is_rejected_before_proof(evidence):
    c, s = evidence; s['runs'][0]['manifest']['files']['Test/secret.json'] = 'a' * 64
    a = o.analyse(s, c, '5080')
    assert a['rows'][0]['state'] == 'unknown' and 'held-out' in a['rows'][0]['issue']
