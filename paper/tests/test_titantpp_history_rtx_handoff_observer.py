import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from paper.scripts import observe_titantpp_history_rtx_handoff as obs

class ObserverRegression(unittest.TestCase):
    def setUp(self):
        self.c = obs.read(obs.BUNDLE / 'execution_contract.json')
        self.host = '5080'
        self.jobs = [j for j in self.c['jobs'] if j['host'] == self.host]
        self.snapshot = {'contract_sha256': obs.CONTRACT, 'root': self.c['hosts'][self.host]['root'], 'rawSHA_verified': True, 'host': self.host, 'observed_unix': 1000., 'gpu_processes': '', 'processes': [], 'jobs': {j['id']: {'job': j, 'status': None, 'claim': None} for j in self.jobs}, 'status.json': None, 'supervisor_exit.json': None}
    def complete_snapshot(self):
        for job in self.jobs:
            self.snapshot['jobs'][job['id']].update(terminal={'job': job, 'status': 'complete', 'scientific_success': True, 'held_out_test_evaluated': False, 'contract_sha256': obs.CONTRACT}, endpoint={'completed_epochs': 49, 'best_epoch': 9}, worker_exit={'job': job, 'actual_child_exit_observed': True, 'returncode': 0, 'contract_sha256': obs.CONTRACT}, terminal_manifest_sha256='a' * 64)
        self.snapshot['status.json'] = {'status': 'complete', 'completed_conditions': 3}
        self.snapshot['supervisor_exit.json'] = {'actual_child_exit_observed': True, 'returncode': 0, 'contract_sha256': obs.CONTRACT, 'phase': 'dispatch'}
        self.snapshot['pipeline_complete.json'] = {'status': 'complete', 'completed_conditions': 3, 'completed_unix': 900., 'contract_sha256': obs.CONTRACT, 'held_out_test_evaluated': False, 'supervisor_exit_sha256': 'b' * 64, 'actual_exit_and_idle_observation': {'owned_processes': [], 'gpu_pids': []}}
        self.snapshot['pipeline_heartbeat.json'] = {'phase': 'complete', 'contract_sha256': obs.CONTRACT, 'updated_unix': 901.}
        self.snapshot['raw_files_sha256'] = {'supervisor_exit.json': 'b' * 64, 'pipeline_complete.json': 'c' * 64, 'pipeline_heartbeat.json': 'd' * 64}
        return self.snapshot
    def active(self):
        j = self.jobs[0]
        root = self.c['hosts'][self.host]['root']
        python = self.c['hosts'][self.host]['python']
        sup = [python, root + '/operation/campaign.py', '--contract', root + '/execution_contract.json', '--host', self.host, '--mode', 'dispatch']
        fit = [python, root + '/operation/campaign.py', '--contract', root + '/execution_contract.json', '--host', self.host, '--mode', 'fit', '--job', j['id'], '--deadline', '1500.0']
        guard = ['timeout', '--signal=TERM', '--kill-after=15s', '600', *fit]
        self.snapshot['status.json'] = {'status': 'running', 'supervisor_pid': 10, 'supervisor_argv': sup, 'active_jobs': {j['id']: {'worker_group_pid': 11, 'worker_guard_argv': guard, 'condition_deadline_unix': 1500.}}}
        self.snapshot['processes'] = [{'pid': 10, 'ppid': 9, 'argv': sup}, {'pid': 11, 'ppid': 10, 'argv': guard}, {'pid': 12, 'ppid': 11, 'argv': fit}]
        self.snapshot['gpu_processes'] = self.c['hosts'][self.host]['gpu_uuid'] + ',12'
        self.snapshot['jobs'][j['id']].update(claim={'job': j, 'contract_sha256': obs.CONTRACT, 'deadline_unix': 1500.}, lease={'job': j['id'], 'contract_sha256': obs.CONTRACT, 'supervisor_pid': 10, 'worker_group_pid': 11, 'issued_unix': 999., 'expires_unix': 1080.})
        return j
    def test_qualification_does_not_count_as_training(self):
        self.snapshot['pipeline_heartbeat.json'] = {'phase': 'qualification'}
        result = obs.classify(self.c, self.host, self.snapshot)
        self.assertEqual(result['counts']['waiting'], 3)
        self.assertEqual(result['counts']['running'], 0)
    def test_exact_owned_first_save_and_foreign_gpu(self):
        self.active()
        result = obs.classify(self.c, self.host, self.snapshot)
        self.assertEqual(result['counts']['running'], 1)
        self.assertTrue(result['rows'][0]['first_checkpoint_not_saved'])
        self.snapshot['gpu_processes'] += '\n' + self.c['hosts'][self.host]['gpu_uuid'] + ',999'
        self.assertEqual(obs.classify(self.c, self.host, self.snapshot)['counts']['unconfirmed'], 1)
    def test_wrong_ppid_or_argv_or_lease_is_unconfirmed_not_failed(self):
        self.active()
        for mutate in [lambda s: s['processes'][2].update(ppid=999), lambda s: s['processes'][2]['argv'].append('--wrong'), lambda s: s['jobs'][self.jobs[0]['id']]['lease'].update(expires_unix=900)]:
            snapshot = copy.deepcopy(self.snapshot)
            mutate(snapshot)
            counts = obs.classify(self.c, self.host, snapshot)['counts']
            self.assertEqual(counts['unconfirmed'], 1)
            self.assertEqual(counts['failed'], 0)
    def test_checkpoint_selector_first_tie_and_own_eta(self):
        j = self.active()
        self.snapshot['jobs'][j['id']].update(checkpoint_receipt={'epoch': 3}, checkpoint_stable=True, history={'history': [{'val_qty_rmse': 5}, {'val_qty_rmse': 4}, {'val_qty_rmse': 4}]}, timing={'epochs': [{'epoch': 1, 'elapsed_seconds': 10}, {'epoch': 2, 'elapsed_seconds': 20}, {'epoch': 3, 'elapsed_seconds': 30}]})
        row = obs.classify(self.c, self.host, self.snapshot)['rows'][0]
        self.assertEqual((row['stored_epoch'], row['best_epoch']), (3, 2))
        self.assertEqual(row['ETA']['median_epoch_seconds'], 20)
        self.assertEqual(row['ETA']['unchanged_best_seconds'], 39 * 20)
    def test_eta_only_recent_completed_finite_rows(self):
        rows = [{'epoch': i, 'elapsed_seconds': i} for i in range(1, 21)] + [{'epoch': 21, 'elapsed_seconds': 99999}, {'epoch': 22, 'elapsed_seconds': float('nan')}]
        eta = obs.own_eta({'epochs': rows}, 20, 10)
        self.assertEqual(eta['median_epoch_seconds'], 15.5)
        self.assertEqual(eta['sample_count'], 10)
        self.assertIn('unconfirmed', obs.own_eta({}, 0, 0)['status'])
    def test_unique27_rejects_double_counting_transferred_ids(self):
        parent = obs.read(obs.A100 / 'execution_contract.json')['jobs'][:15]
        a100 = [{'job': j, 'state': 'complete'} for j in parent]
        rtx = [{'job': j, 'state': 'waiting'} for j in self.c['jobs']]
        self.assertEqual(obs.unique_counts(a100, rtx)['complete'], 15)
        rtx[-1]['job'] = parent[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            obs.unique_counts(a100, rtx)
    def test_no_launch_means_zero_remote_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            fetch = Mock(side_effect=AssertionError('must never call SSH'))
            result = obs.observe_host('5080', bundle=Path(tmp), fetch=fetch, verifier=lambda _: self.c)
            self.assertEqual(result['remote_calls'], 0)
            fetch.assert_not_called()
    def test_terminal_cache_preserves_actual_time_and_zero_remote(self):
        self.complete_snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            path = bundle / 'hourly_monitor/snapshot.json'
            obs.write(path, self.snapshot)
            out = {**obs.classify(self.c, self.host, self.snapshot), 'verified_terminal': True, 'host': '5080', 'root': self.c['hosts']['5080']['root'], 'contract_sha256': obs.CONTRACT, 'observed_kst': 'original timestamp', 'snapshot_path': str(path), 'snapshot_raw_sha256': obs.sha(path)}
            cache = bundle / 'hourly_monitor/terminal_cache_5080.json'
            obs.write(cache, out)
            obs.write(bundle / 'hourly_monitor/terminal_pointer_5080.json', {'sha256': obs.sha(cache)})
            fetch = Mock(side_effect=AssertionError('must never call SSH'))
            result = obs.observe_host('5080', bundle=bundle, fetch=fetch, verifier=lambda _: self.c)
            self.assertEqual(result['observed_kst'], 'original timestamp')
            self.assertEqual(result['remote_calls'], 0)
            fetch.assert_not_called()
    def test_query_failure_attempts_once_and_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            obs.write(bundle / 'deployment/launch_receipt_5080.json', {'contract_sha256': obs.CONTRACT, 'host': '5080', 'root': self.c['hosts']['5080']['root'], 'gpu_uuid': self.c['hosts']['5080']['gpu_uuid'], 'actual_launch': True})
            obs.write(bundle / 'control/hosts/5080/integrity_receipt.json', {'contract_sha256': obs.CONTRACT, 'files': {}})
            fetch = Mock(side_effect=RuntimeError('SSH unavailable'))
            result = obs.observe_host('5080', bundle=bundle, fetch=fetch, verifier=lambda _: self.c)
            fetch.assert_called_once()
            self.assertEqual(result['counts']['failed'], 0)
            self.assertEqual(result['counts']['unconfirmed'], 3)
            self.assertTrue(Path(result['error_evidence_path']).exists())
    def test_successful_endpoints_need_all_actual_terminal_evidence(self):
        snapshot = self.complete_snapshot()
        self.assertTrue(obs.classify(self.c, self.host, snapshot)['verified_terminal'])
        for mutate in [lambda s: s.pop('pipeline_complete.json'), lambda s: s.pop('raw_files_sha256'), lambda s: s['supervisor_exit.json'].update(returncode=1), lambda s: s['pipeline_heartbeat.json'].update(phase='dispatch'), lambda s: s['pipeline_heartbeat.json'].update(updated_unix=899.), lambda s: s['jobs'][self.jobs[0]['id']].pop('worker_exit'), lambda s: s['pipeline_complete.json'].update(supervisor_exit_sha256='e' * 64)]:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(snapshot)
                mutate(changed)
                self.assertFalse(obs.classify(self.c, self.host, changed)['verified_terminal'])
    def test_insufficient_cache_refreshes_once_preserving_old_evidence(self):
        fresh = copy.deepcopy(self.complete_snapshot())
        self.snapshot.pop('pipeline_complete.json')
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            snapshot_path = bundle / 'hourly_monitor/snapshot_old.json'
            obs.write(snapshot_path, self.snapshot)
            cache = bundle / 'hourly_monitor/terminal_cache_5080.json'
            old = {**obs.classify(self.c, self.host, self.snapshot), 'verified_terminal': True, 'snapshot_path': str(snapshot_path), 'snapshot_raw_sha256': obs.sha(snapshot_path)}
            obs.write(cache, old)
            original_bytes = cache.read_bytes()
            obs.write(bundle / 'hourly_monitor/terminal_pointer_5080.json', {'sha256': obs.sha(cache)})
            obs.write(bundle / 'deployment/launch_receipt_5080.json', {'contract_sha256': obs.CONTRACT, 'host': '5080', 'root': self.c['hosts']['5080']['root'], 'gpu_uuid': self.c['hosts']['5080']['gpu_uuid'], 'actual_launch': True})
            obs.write(bundle / 'control/hosts/5080/integrity_receipt.json', {'contract_sha256': obs.CONTRACT, 'files': {}})
            fetch = Mock(return_value=fresh)
            result = obs.observe_host('5080', bundle=bundle, fetch=fetch, verifier=lambda _: self.c)
            fetch.assert_called_once()
            self.assertTrue(result['verified_terminal'])
            preserved = list(cache.parent.glob('terminal_cache_5080_superseded_*.json'))
            self.assertEqual(len(preserved), 1)
            self.assertEqual(preserved[0].read_bytes(), original_bytes)
            obs.observe_host('5080', bundle=bundle, fetch=fetch, verifier=lambda _: self.c)
            self.assertEqual(fetch.call_count, 1)
    def test_verified_parent_endpoint_replaces_stale_checkpoint_epoch(self):
        jobs = obs.read(obs.A100 / 'execution_contract.json')['jobs'][:15]
        cached = [{'job': j, 'state': 'complete' if i < 14 else 'training', 'stored_epoch': 40, 'best_epoch': 2} for i, j in enumerate(jobs)]
        records = [{'id': j['id'], 'last_epoch': 61 + i, 'best_epoch': 21 + i, 'terminal_manifest_sha256': 'a' * 64, 'endpoint_raw_sha256': 'b' * 64, 'provenance': {'archive_sha256': 'c' * 64}} for i, j in enumerate(jobs)]
        rows = obs.parent_epoch_rows(cached, records, {j['id'] for j in jobs})
        self.assertEqual((rows[-1]['stored_epoch'], rows[-1]['best_epoch']), (75, 35))
        self.assertEqual(rows[-1]['state'], 'complete')
        self.assertEqual(rows[-1]['original_provenance']['archive_sha256'], 'c' * 64)

if __name__ == '__main__':
    unittest.main()
