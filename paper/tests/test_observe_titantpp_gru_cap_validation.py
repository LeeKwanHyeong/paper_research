"""Synthetic observation evidence only: no remote calls, models or data loaded."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from paper.scripts import observe_titantpp_gru_cap_validation as monitor
from paper.scripts import launch_titantpp_cnn_gru_components_v5 as launcher


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def put(snapshot, name, value):
    content = raw(value)
    snapshot['files'][name] = {'base64': base64.b64encode(content).decode(),
        'raw_sha256': hashlib.sha256(content).hexdigest()}
    return hashlib.sha256(content).hexdigest()


def get(snapshot, name):
    return json.loads(base64.b64decode(snapshot['files'][name]['base64']))


def contract():
    jobs = []
    for dataset in ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000'):
        for seed in (42, 52, 62):
            host = '5090' if dataset == 'intermittent_frozen_5000' and seed != 62 else '5080'
            jobs.append({'id': f'{dataset}__{seed}__{monitor.ARM}', 'dataset': dataset,
                'seed': seed, 'arm': monitor.ARM, 'host': host})
    return {'jobs': jobs, 'hosts': {host: {'root': '/synthetic/cap_' + host,
        'operation_root': '/synthetic/cap_' + host + '/operation', 'python': '/synthetic/python',
        'gpu_uuid': 'GPU-' + host} for host in monitor.HOSTS},
        'training': {'minimum_epochs': 40, 'maximum_epochs': 300},
        'limits': {'per_condition_seconds': 129600, 'total_wall_seconds': 604800},
        'source': {'files_sha256': 'f' * 64}, 'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False}


def terminal_fixture(c, host):
    spec, digest = c['hosts'][host], monitor.canonical(c)
    now = time.time()
    dispatcher = [spec['python'], spec['operation_root'] + '/campaign.py', '--contract',
        spec['root'] + '/execution_contract.json', '--host', host, '--mode', 'dispatch']
    snapshot = {'host': host, 'root': spec['root'], 'contract_sha256': digest,
        'source_files_sha256': c['source']['files_sha256'], 'gpu_uuid': spec['gpu_uuid'],
        'gpu_inventory_uuids': [spec['gpu_uuid']], 'gpu_pids': [], 'processes': {},
        'files': {}, 'terminal_hashes': {}, 'checkpoints': {}, 'observed_unix': now}
    sources = {100: set(), 101: set()}
    for name in ('supervisor.claim', 'phases/train_launch.json', 'phases/train_exit.json'):
        put(snapshot, name, {'phase': 'train', 'contract_sha256': digest, 'pid': 101, 'ppid': 100,
            'argv': dispatcher, 'os_exit_code': 0})
        sources[100].add(name)
        sources[101].add(name)
    put(snapshot, 'phases/train_complete.json', {'phase': 'train', 'host': host, 'contract_sha256': digest,
        'subprocess_os_exit_code': 0, 'gpu_compute_pids': [], 'completed_unix': now - 5})
    completed = {}
    for index, job in enumerate(monitor.registry(c)[host]):
        jid = job['id']
        relative = f"runs/{job['arm']}/count_only_log_regression/seed_{job['seed']}"
        prefix = 'run/' + jid + '/'
        history = {'history': [{'epoch': epoch, 'val_qty_rmse': 1. if epoch <= 2 else 2.}
            for epoch in range(1, 41)]}
        timing = {'scope': 'train_iteration_through_validation_and_checkpoint_save',
            'epochs': [{'epoch': epoch, 'elapsed_seconds': float(epoch)} for epoch in range(1, 41)]}
        manifest = {name: 'a' * 64 for name in ('initialization.json', 'input_receipt.json', 'two_epoch_cost_receipt.json')}
        manifest['status.json'] = put(snapshot, prefix + 'status.json', {'status': 'complete', 'job': job})
        for name in ('best_val_qty_rmse_model.pt', 'last_epoch_state.pt', 'exposure.json', 'endpoint_replays.json',
                     'summary.json', 'server_checkpoint_receipt.json', 'train.log'):
            manifest[relative + '/' + name] = 'a' * 64
        manifest[relative + '/history.json'] = put(snapshot, prefix + relative + '/history.json', history)
        manifest[relative + '/epoch_timing.json'] = put(snapshot, prefix + relative + '/epoch_timing.json', timing)
        terminal_sha = put(snapshot, prefix + 'terminal_manifest.json', {'status': 'complete',
            'scientific_success': True, 'job': job, 'contract_sha256': digest, 'files': manifest,
            'held_out_test_evaluated': False, 'evaluation_scope': 'validation_only', 'completed_unix': now - 20})
        deadline = now + 10000
        guard, worker = 200 + index, 300 + index
        command = ['timeout', '--signal=TERM', '--kill-after=15s', '10000', spec['python'],
            spec['operation_root'] + '/campaign.py', '--contract', spec['root'] + '/execution_contract.json',
            '--host', host, '--mode', 'fit', '--job', jid, '--deadline', str(deadline)]
        worker_name = 'worker_exits/' + jid + '.json'
        worker_sha = put(snapshot, worker_name, {'actual_child_exit_observed': True, 'returncode': 0,
            'job': job, 'contract_sha256': digest, 'supervisor_pid': 101, 'worker_group_pid': guard,
            'pid': worker, 'ppid': guard, 'argv': command, 'observed_unix': now - 10})
        sources[guard] = {worker_name}
        sources[worker] = {worker_name}
        sources[101].add(worker_name)
        put(snapshot, 'claims/' + jid + '.json', {'job': job, 'contract_sha256': digest, 'deadline_unix': deadline})
        put(snapshot, 'leases/' + jid + '.json', {'job': jid, 'contract_sha256': digest, 'worker_group_pid': guard,
            'supervisor_pid': 101, 'issued_unix': now - 40, 'expires_unix': now + 20})
        snapshot['terminal_hashes'][jid] = {'all_files_verified': True, 'files': {name: True for name in manifest}}
        completed[jid] = {'worker_exit_sha256': worker_sha, 'terminal_manifest_sha256': terminal_sha}
    put(snapshot, 'status.json', {'status': 'complete', 'completed': completed, 'completed_conditions': len(completed),
        'held_out_test_evaluated': False})
    snapshot['process_observations'] = {str(pid): {'pid': pid, 'state': 'absent', 'absence_observed': True,
        'evidence_sources': sorted(names)} for pid, names in sources.items()}
    snapshot['observed_owner_pids'] = sorted(sources)
    return snapshot


class CapObservationTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        self.s = terminal_fixture(self.c, '5080')

    def verify(self, snapshot=None):
        return monitor.verify_snapshot(snapshot or self.s, self.c, '5080', launcher)

    def test_exact_nine_terminal_requires_seven_and_two_workers(self):
        reports = {host: monitor.verify_snapshot(terminal_fixture(self.c, host), self.c, host, launcher)
            for host in monitor.HOSTS}
        counts = monitor.unique_counts(self.c, reports)
        self.assertEqual([reports[h]['completed'] for h in monitor.HOSTS], [7, 2])
        self.assertEqual(counts['completed'], 9)
        self.assertTrue(counts['all_training_terminal_verified'])
        self.assertEqual(reports['5080']['jobs'][0]['best_epoch'], 1)
        self.assertFalse(reports['5080']['pipeline_os_exit_directly_observed'])

    def test_registry_rejects_duplicate_and_wrong_host_before_remote(self):
        c = deepcopy(self.c)
        c['jobs'][0] = deepcopy(c['jobs'][1])
        with self.assertRaises(ValueError):
            monitor.registry(c)
        c = deepcopy(self.c)
        c['jobs'][-1]['host'] = '5090'
        with self.assertRaises(ValueError):
            monitor.registry(c)

    def test_wrapper_absence_is_required_even_with_empty_gpu(self):
        self.s['process_observations'].pop('100')
        with self.assertRaisesRegex(ValueError, 'explicitly absent'):
            self.verify()

    def test_actual_scientific_worker_absence_is_required(self):
        self.s['process_observations'].pop('300')
        with self.assertRaisesRegex(ValueError, 'not absent'):
            self.verify()

    def test_racy_or_reused_pid_never_becomes_complete(self):
        for state in ('live', 'query_error', 'reused_pid'):
            snapshot = deepcopy(self.s)
            snapshot['process_observations']['200']['state'] = state
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.verify(snapshot)

    def test_absence_boolean_and_receipt_source_must_be_present(self):
        self.s['process_observations']['300']['absence_observed'] = False
        with self.assertRaises(ValueError):
            self.verify()
        self.s['process_observations']['300']['absence_observed'] = True
        self.s['process_observations']['300']['evidence_sources'] = []
        with self.assertRaisesRegex(ValueError, 'probe source'):
            self.verify()

    def test_binary_witness_failure_blocks_terminal(self):
        witness = next(iter(self.s['terminal_hashes'].values()))
        witness['files'][next(iter(witness['files']))] = False
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.verify()

    def test_worker_actual_exit_nonzero_is_not_completed(self):
        name = 'worker_exits/' + self.c['jobs'][0]['id'] + '.json'
        item = get(self.s, name)
        item['returncode'] = 1
        put(self.s, name, item)
        with self.assertRaisesRegex(ValueError, 'actual OS exit'):
            self.verify()

    def test_history_cannot_bypass_terminal_manifest(self):
        name = next(n for n in self.s['files'] if n.endswith('/history.json'))
        item = get(self.s, name)
        item['history'][0]['val_qty_rmse'] = 0.
        put(self.s, name, item)
        with self.assertRaisesRegex(ValueError, 'captured file SHA'):
            self.verify()

    def test_fixed_gpu_and_owner_probe_coverage_are_required(self):
        self.s['gpu_inventory_uuids'] = ['GPU-other']
        with self.assertRaisesRegex(ValueError, 'GPU inventory'):
            self.verify()
        self.s['gpu_inventory_uuids'] = ['GPU-5080']
        self.s['observed_owner_pids'].remove(300)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.verify()

    def test_cache_retains_original_time_and_rechecks_raw_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory).resolve()
            output = bundle / 'reports'
            path = bundle / 'control/status/5080/snapshot.json'
            monitor.write(path, self.s)
            report = self.verify()
            bridge = {'synthetic_bridge': True}
            monitor.save_cache(bundle, output, self.c, '5080', path, self.s, report, bridge)
            with patch.object(monitor, 'qualification_bridge', return_value=bridge):
                loaded = monitor.load_cache(bundle, output, self.c, '5080', launcher)
                self.assertEqual(loaded[1]['observed_unix'], self.s['observed_unix'])
                with patch.object(monitor.transport, 'read_status_once', side_effect=AssertionError('remote forbidden')):
                    observed = monitor.observe_host(bundle, output, self.c, '5080', launcher, loaded)
                self.assertEqual(observed['remote_calls'], 0)
                self.assertEqual(observed['observed_unix'], self.s['observed_unix'])
                path.write_bytes(path.read_bytes() + b' ')
                with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                    monitor.load_cache(bundle, output, self.c, '5080', launcher)

    def test_cache_file_and_sidecar_are_both_required(self):
        with tempfile.TemporaryDirectory() as directory:
            target = monitor.cache_path(directory, '5080')
            monitor.write(target, {})
            with self.assertRaisesRegex(ValueError, 'Incomplete terminal cache'):
                monitor.load_cache(directory, directory, self.c, '5080', launcher)

    def test_unlaunched_host_has_zero_remote_and_no_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            def forbidden(*args):
                raise AssertionError('remote forbidden')
            report = monitor.observe_host(directory, directory, self.c, '5090', launcher, acquire=forbidden)
            self.assertEqual(report['remote_calls'], 0)
            self.assertEqual([r['status'] for r in report['jobs']], ['waiting', 'waiting'])
            self.assertTrue(all(r['eta']['status'] == 'unconfirmed' for r in report['jobs']))

    def test_transport_error_preserves_uncertainty_and_does_not_claim_learning_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor.write(Path(directory) / 'control/intents/train_5090.json', {})
            def failed(*args):
                raise OSError('synthetic connection failure')
            report = monitor.observe_host(directory, directory, self.c, '5090', launcher, acquire=failed)
            self.assertEqual(report['remote_calls'], 1)
            self.assertFalse(report['training_failure_inferred'])
            self.assertEqual(report['completed'], None)
            self.assertTrue(report['user_action_required'])

    def test_eta_uses_only_own_last_ten_positive_complete_timings(self):
        report = self.verify()
        row = report['jobs'][0]
        row['status'] = 'running'
        report['status'] = 'training_running'
        report['live_process_probes_verified'] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.json'
            monitor.write(path, self.s)
            result = monitor.enrich(report, self.s, self.c, '5080', path, 1, 'synthetic', launcher)
        eta = result['jobs'][0]['eta']
        self.assertEqual(eta['recent_epoch_count'], 10)
        self.assertEqual(eta['median_epoch_seconds'], 35.5)
        self.assertEqual(eta['best_epoch'], 1)
        self.assertEqual(eta['conditional_early_stop_seconds'], 35.5)
        self.assertEqual(eta['conditional_maximum_seconds'], 260 * 35.5)

    def test_unconfirmed_live_identity_cannot_inflate_running_count(self):
        reports = {host: monitor.verify_snapshot(terminal_fixture(self.c, host), self.c, host, launcher)
            for host in monitor.HOSTS}
        reports['5090']['status'] = 'training_identity_unconfirmed'
        reports['5090']['jobs'][0]['status'] = 'running'
        counts = monitor.unique_counts(self.c, reports)
        self.assertEqual(counts['running'], 0)
        self.assertEqual(counts['unconfirmed'], 1)
        self.assertFalse(counts['all_training_terminal_verified'])

    def test_external_monitor_pin_requires_execution_integrity_document(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(monitor, 'PROJECT', Path(directory)):
            project = Path(directory)
            bundle = project / 'search_artifacts/cap/execution'
            output = project / 'reports/cap'
            names = ['paper/scripts/observe_titantpp_gru_cap_validation.py',
                'paper/scripts/titantpp_components_status_transport_v2.py',
                'paper/scripts/launch_titantpp_gru_cap_validation.py',
                'paper/scripts/launch_titantpp_cnn_gru_components_v5.py',
                'paper/scripts/titantpp_components_terminal_verifier.py',
                'paper/scripts/titantpp_component_hourly_eta.py',
                'search_artifacts/cap/execution/execution_integrity.json']
            for name in names:
                monitor.write(project / name, {'synthetic': name})
            files = {name: monitor.sha(project / name) for name in names}
            seal = {'contract_sha256': monitor.canonical(self.c), 'files': files,
                'files_sha256': monitor.canonical(files)}
            path = output / 'monitor_integrity.json'
            monitor.write(path, seal)
            digest = monitor.sha(path)
            monitor.verify_monitor(output, digest, self.c, bundle)
            (bundle / 'execution_integrity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                monitor.verify_monitor(output, digest, self.c, bundle)
            files.pop(names[-1])
            seal['files_sha256'] = monitor.canonical(files)
            path.write_bytes(raw(seal))
            with self.assertRaisesRegex(ValueError, 'seal incomplete'):
                monitor.verify_monitor(output, monitor.sha(path), self.c, bundle)

    def test_live_gpu_worker_requires_stable_process_identity(self):
        snapshot = {'gpu_inventory_uuids': ['GPU-5080'], 'gpu_pids': [300], 'processes': {},
            'process_observations': {}}
        files = {'status.json': raw({'supervisor_pid': 101, 'active_jobs': {'fit': {'worker_group_pid': 200}}}),
            'phases/train_launch.json': raw({'ppid': 100})}
        for pid, ppid in ((100, 1), (101, 100), (200, 101), (300, 200)):
            process = {'pid': pid, 'ppid': ppid, 'pgrp': 200 if pid in (200, 300) else 100,
                'argv': ['synthetic'], 'start_ticks': pid, 'identity_stable_during_query': True,
                'observation_state': 'live'}
            snapshot['processes'][str(pid)] = process
            snapshot['process_observations'][str(pid)] = {'state': 'live', 'actual': deepcopy(process)}
        report = {'status': 'training_running'}
        self.assertTrue(monitor.live_probes(snapshot, self.c, '5080', files, report)['live_process_probes_verified'])
        snapshot['process_observations']['300']['actual']['start_ticks'] = 999
        with self.assertRaisesRegex(ValueError, 'racy or unresolved'):
            monitor.live_probes(snapshot, self.c, '5080', files, report)


if __name__ == '__main__':
    unittest.main()
