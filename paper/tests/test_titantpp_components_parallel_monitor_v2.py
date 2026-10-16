"""Synthetic owner-probe regressions; no remote calls or scientific evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import titantpp_components_parallel_monitor_v2 as monitor
from paper.tests import test_titantpp_components_parallel_monitor as legacy_tests


class OwnerProbeTests(unittest.TestCase):
    def setUp(self):
        self.legacy = legacy_tests.ParallelMonitorTests()
        self.legacy.setUp()
        self.c = self.legacy.c
        self.snapshot, self.records, self.proof = self.legacy.fixture()
        self.snapshot['gpu_inventory_uuids'] = [self.c['hosts']['5090']['gpu_uuid']]
        sources = {101: set(), 100: set()}
        for name in ('supervisor.claim', 'phases/train_launch.json', 'phases/train_exit.json'):
            sources[101].add(name)
            sources[100].add(name)
        for index, job in enumerate(self.legacy.retained):
            name = 'worker_exits/' + job['id'] + '.json'
            pid = 201 + index
            self.records[name] = {'worker_group_pid': pid, 'supervisor_pid': 101,
                                  'contract_sha256': monitor.original.canonical(self.c), 'job': job}
            sources[pid] = {name}
            sources[101].add(name)
        self.snapshot['process_observations'] = {
            str(pid): {'pid': pid, 'state': 'absent', 'evidence_sources': sorted(names)}
            for pid, names in sources.items()}
        self.snapshot['observed_owner_pids'] = sorted(sources)
        self.legacy.encode(self.snapshot, self.records)

    def verify(self):
        with patch('paper.scripts.titantpp_components_terminal_verifier.verify_completed_job',
                   side_effect=lambda s, c, h, j: {'job': j, 'status': 'complete'}):
            return monitor.planned_parent_terminal(self.snapshot, self.c, self.proof)

    def test_exact_admin_boundary_requires_all_ten_explicit_absent_probes(self):
        result = self.verify()
        self.assertEqual(result['owner_process_probe_count'], 10)
        self.assertTrue(result['owner_process_absence_verified'])
        self.assertEqual(result['administrative_dispatcher_actual_exit'], 1)
        self.assertFalse(result['parent_original12_complete'])
        self.assertFalse(result['pipeline_wrapper_OS_exit_directly_observed'])

    def test_legacy_empty_processes_does_not_prove_absence(self):
        self.snapshot.pop('process_observations')
        with self.assertRaisesRegex(ValueError, 'probes missing'):
            self.verify()

    def test_live_reused_query_error_or_racy_owner_never_becomes_terminal(self):
        for state in ('live', 'reused_pid', 'query_error', 'disappeared_during_observation'):
            with self.subTest(state=state):
                self.snapshot['process_observations']['101']['state'] = state
                with self.assertRaisesRegex(ValueError, 'explicitly absent'):
                    self.verify()

    def test_pipeline_wrapper_is_an_independent_required_probe(self):
        self.snapshot['process_observations'].pop('100')
        with self.assertRaisesRegex(ValueError, 'explicitly absent: 100'):
            self.verify()

    def test_any_exited_worker_missing_from_probes_blocks_cache(self):
        self.snapshot['observed_owner_pids'].remove(208)
        with self.assertRaisesRegex(ValueError, 'coverage incomplete'):
            self.verify()

    def test_probe_source_is_bound_to_original_claim_and_exit_files(self):
        self.snapshot['process_observations']['101']['evidence_sources'].remove('supervisor.claim')
        with self.assertRaisesRegex(ValueError, 'explicitly absent'):
            self.verify()

    def test_wrong_fixed_gpu_inventory_is_rejected_even_when_compute_list_empty(self):
        self.snapshot['gpu_inventory_uuids'] = ['GPU-foreign']
        with self.assertRaisesRegex(ValueError, 'Fixed GPU'):
            self.verify()

    def test_malformed_owner_pid_is_rejected(self):
        self.records['supervisor.claim']['pid'] = True
        self.records['phases/train_launch.json']['pid'] = True
        self.records['phases/train_exit.json']['pid'] = True
        self.legacy.encode(self.snapshot, self.records)
        with self.assertRaises(ValueError):
            self.verify()

    def test_worker_returncode_or_binary_manifest_remains_a_hard_gate(self):
        with patch('paper.scripts.titantpp_components_terminal_verifier.verify_completed_job',
                   side_effect=ValueError('worker exit or binary mismatch')):
            with self.assertRaisesRegex(ValueError, 'worker exit or binary mismatch'):
                monitor.planned_parent_terminal(self.snapshot, self.c, self.proof)

    def test_all36_flag_needs_both_new_owner_absence_facts(self):
        groups = self.legacy.groups()
        counts = monitor.unique36_counts(self.c, self.legacy.destination, groups)
        self.assertFalse(counts['all_training_terminal_verified'])
        groups['parent_5090']['owner_process_absence_verified'] = True
        self.assertFalse(monitor.unique36_counts(self.c, self.legacy.destination, groups)['all_training_terminal_verified'])
        groups['handoff_5080']['owner_process_absence_verified'] = True
        self.assertTrue(monitor.unique36_counts(self.c, self.legacy.destination, groups)['all_training_terminal_verified'])

    def test_error_report_retains_transport_evidence_without_training_failure_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            path.write_text(json.dumps({'transport_os_exit_code': 1, 'stderr': {'raw_sha256': 'abc'}}))
            error = RuntimeError('status transport exited1')
            error.evidence_receipt_path = path
            result = monitor._error_report('5090', 8, None, None, 1, error)
        self.assertEqual(result['transport_evidence_path'], str(path))
        self.assertEqual(len(result['transport_evidence_raw_sha256']), 64)
        self.assertFalse(result['training_failure_inferred'])
        self.assertFalse(result['identity_verified'])
        self.assertEqual(result['jobs'], [])

    def test_acquire_uses_status_only_transport_once(self):
        expected = (Path('/synthetic/snapshot.json'), {'synthetic': True})
        with patch.object(monitor.status_transport, 'read_status_once', return_value=expected) as call:
            self.assertEqual(monitor._acquire('/synthetic/bundle', self.c, '5090', monitor.original), expected)
        call.assert_called_once_with('/synthetic/bundle', monitor.original, self.c, '5090',
                                     monitor.PROJECT_OUTPUT / 'diagnostics')

    def live_fixture(self):
        self.records['status.json'].update(status='running', supervisor_pid=101,
            active_jobs={'synthetic-fit': {'worker_group_pid': 201}})
        self.legacy.encode(self.snapshot, self.records)
        self.snapshot['gpu_pids'] = [301]
        self.snapshot['processes'] = {}
        for pid in (100, 101, 201, 301):
            actual = {'pid': pid, 'ppid': 100, 'pgrp': pid, 'argv': ['synthetic-only'],
                      'start_ticks': 123, 'identity_stable_during_query': True}
            self.snapshot['processes'][str(pid)] = {**actual, 'observation_state': 'live'}
            self.snapshot['process_observations'][str(pid)] = {'pid': pid, 'state': 'live', 'actual': actual}
        return {'status': 'training_running', 'identity_verified': True}

    def test_live_identity_requires_stable_probes_as_well_as_legacy_argv_match(self):
        report = self.live_fixture()
        result = monitor.verify_live_probes(self.snapshot, self.c, '5090', report)
        self.assertTrue(result['live_process_probes_verified'])
        self.snapshot['process_observations']['201']['state'] = 'reused_pid'
        self.snapshot['process_observations']['201']['actual']['identity_stable_during_query'] = False
        # Original argv/PPID/pgrp remain identical; new start-tick race gate must reject.
        with self.assertRaisesRegex(ValueError, 'reused, racy or unresolved'):
            monitor.verify_live_probes(self.snapshot, self.c, '5090', report)

    def test_live_gpu_probe_error_never_produces_verified_eta(self):
        report = self.live_fixture()
        self.snapshot['process_observations']['301']['state'] = 'query_error'
        with self.assertRaisesRegex(ValueError, 'reused, racy or unresolved'):
            monitor.verify_live_probes(self.snapshot, self.c, '5090', report)


if __name__ == '__main__':
    unittest.main()
