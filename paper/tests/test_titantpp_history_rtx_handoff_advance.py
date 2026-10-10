import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from paper.scripts import advance_titantpp_history_rtx_handoff as advance
from paper.scripts import observe_titantpp_history_rtx_handoff as observer

class AdvanceRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bundle = Path(self.tmp.name)
        self.c = observer.read(observer.BUNDLE / 'execution_contract.json')
        spec = self.c['hosts']['5080']
        self.health = {'host': '5080', 'root': spec['root'], 'contract_sha256': observer.CONTRACT, 'observed_unix': time.time(), 'reboot_required': False, 'gpu': {'rc': 0, 'stdout': spec['gpu_uuid'] + ', NVIDIA GeForce RTX5080'}, 'gpu_pids': {'rc': 0, 'stdout': ''}, 'runtime': {'rc': 0, 'stdout': json.dumps({**spec['runtime_expected'], 'cuda_available': True, 'device_count': 1})}, 'python_exists': True, 'tmux_exists': True, 'maintenance_query_rc': 0, 'maintenance_processes': [], 'package_lock_query_rc': 0, 'package_locks_held': []}
        self.deploy = Mock(return_value={'status': 'deployed'})
        self.launch = Mock(return_value={'status': 'launched'})
    def call(self, health=None):
        query = Mock(return_value=health or self.health)
        result = advance.activate_5080(self.c, bundle=self.bundle, health_query=query, deploy=self.deploy, launch=self.launch)
        return result, query
    def test_reboot_and_NVML_mismatch_are_wait_not_repair(self):
        for change, expected in [({'reboot_required': True}, 'external_wait_reboot_required'), ({'gpu': {'rc': 18, 'stdout': 'NVML mismatch'}}, 'external_wait_NVML_or_driver_unhealthy')]:
            result, query = self.call({**self.health, **change})
            self.assertEqual(result['status'], expected)
            query.assert_called_once()
        self.deploy.assert_not_called()
        self.launch.assert_not_called()
    def test_busy_GPU_never_deploys(self):
        result, _ = self.call({**self.health, 'gpu_pids': {'rc': 0, 'stdout': 'GPU,123'}})
        self.assertEqual(result['status'], 'external_wait_GPU_busy')
        self.deploy.assert_not_called()
    def test_wrong_runtime_does_not_reinstall(self):
        result, _ = self.call({**self.health, 'runtime': {'rc': 0, 'stdout': '{}'}})
        self.assertEqual(result['status'], 'external_wait_existing_runtime_mismatch')
        self.deploy.assert_not_called()
    def test_prior_intent_has_no_query_retry_or_launch(self):
        observer.write(self.bundle / 'deployment/deploy_intent_5080.json', {'preserved': True})
        result, query = self.call()
        self.assertIn('preserved_incomplete_attempt', result['status'])
        query.assert_not_called()
        self.deploy.assert_not_called()
        self.launch.assert_not_called()
    def test_already_launched_does_not_repeat_activation(self):
        observer.write(self.bundle / 'deployment/launch_receipt_5080.json', {'actual_launch': True})
        result, query = self.call()
        self.assertEqual(result['status'], 'already_launched')
        query.assert_not_called()
        self.deploy.assert_not_called()
    def test_healthy_executes_each_step_exactly_once(self):
        result, query = self.call()
        self.assertEqual(result['status'], 'native_pipeline_launched_not_fit_confirmation')
        query.assert_called_once()
        self.deploy.assert_called_once_with('5080')
        self.launch.assert_called_once_with('5080')
    def test_deploy_failure_blocks_launch_and_future_attempt(self):
        self.deploy.side_effect = RuntimeError('preserved partial deployment')
        result, _ = self.call()
        self.assertIn('preserved_incomplete_attempt', result['status'])
        self.launch.assert_not_called()
        again, query = self.call()
        query.assert_not_called()
        self.assertEqual(self.deploy.call_count, 1)
        self.assertIn('preserved_incomplete_attempt', again['status'])
    def test_old_health_bytes_and_observation_time_preserved(self):
        path = self.bundle / 'deployment/health_5080.json'
        path.parent.mkdir()
        raw = b'{"host":"5080", "observed_unix": 123.0, "historical":true}\n'
        path.write_bytes(raw)
        self.call()
        histories = list(path.parent.glob('health_5080_19700101*.json'))
        self.assertEqual(len(histories), 1)
        self.assertEqual(histories[0].read_bytes(), raw)
        self.assertEqual(observer.read(path)['observed_unix'], self.health['observed_unix'])
    def test_missing_monitor_seal_blocks_all_activation_and_observation(self):
        activation = Mock(side_effect=AssertionError('no activation'))
        observed = Mock(side_effect=AssertionError('no SSH'))
        finalize = Mock(side_effect=AssertionError('no finalization'))
        with self.assertRaisesRegex(ValueError, 'Monitor seal missing'):
            advance.advance(bundle=self.bundle, verifier=lambda _: self.c, activate=activation, observe=observed, finalize=finalize)
        activation.assert_not_called()
        observed.assert_not_called()
        finalize.assert_not_called()
    def test_exact_shutdown_waiter_is_benign_but_active_upgrade_or_locks_block(self):
        waiter = '1514 1 unattended-upgr /usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal'
        evidence = advance.maintenance_evidence(waiter, '', {'/var/lib/dpkg/lock': [8, 1, 42]})
        self.assertEqual(len(evidence['benign_shutdown_waiters']), 1)
        self.assertEqual(advance.health_decision(self.c, '5080', {**self.health, **evidence}), 'healthy_idle')
        for process, locks in [(waiter + ' --extra', ''), ('20 1 apt apt upgrade', ''), ('21 1 dpkg dpkg --configure -a', ''), ('22 1 unattended-upgr /usr/bin/python3 /usr/bin/unattended-upgrade', ''), (waiter, '1: POSIX ADVISORY WRITE 333 08:01:42 0 EOF')]:
            with self.subTest(process=process, locks=locks):
                evidence = advance.maintenance_evidence(process, locks, {'/var/lib/dpkg/lock': [8, 1, 42]})
                self.assertEqual(advance.health_decision(self.c, '5080', {**self.health, **evidence}), 'external_wait_OS_maintenance')
        self.assertEqual(advance.health_decision(self.c, '5080', {**self.health, 'package_lock_query_rc': 1}), 'external_wait_OS_maintenance')

    def test_remote_health_source_is_valid_without_exec(self):
        compile(advance.health_code(self.c, '5080'), 'health_read_only', 'exec')
    def test_observe_only_skips_activation_and_calls_each_observer_finalizer_once(self):
        parent = observer.read(observer.A100 / 'execution_contract.json')['jobs'][:15]
        a100_rows = [{'job': job, 'state': 'complete'} for job in parent]
        activation = Mock(side_effect=AssertionError('activation forbidden'))
        observed = Mock(side_effect=lambda host, **kw: {'rows': [{'job': j, 'state': 'waiting'} for j in self.c['jobs'] if j['host'] == host]})
        finalized = Mock(return_value={'status': 'pending', 'remote_calls': 0})
        observer.write(self.bundle / 'control/monitor_integrity.json', {'contract_sha256': observer.CONTRACT, 'files': {}, 'project_files': {}})
        result = advance.advance(observe_only=True, bundle=self.bundle, verifier=lambda _: self.c, instacart_check=lambda: {'completed': 9}, a100_check=lambda: (a100_rows, {'remote_calls': 0}), activate=activation, observe=observed, finalize=finalized)
        activation.assert_not_called()
        self.assertEqual(observed.call_count, 2)
        self.assertEqual([call.args[0] for call in observed.call_args_list], ['5080', '5090'])
        finalized.assert_called_once()
        self.assertEqual(result['counts']['complete'], 15)
        self.assertEqual(result['counts']['waiting'], 12)
        self.assertEqual(result['A100_SSH_calls'], 0)

if __name__ == '__main__':
    unittest.main()
