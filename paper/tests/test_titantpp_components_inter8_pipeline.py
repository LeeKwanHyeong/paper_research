"""Synthetic supervisor control-flow tests; every native call is mocked."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper.scripts import run_titantpp_components_inter8_test_pipeline as pipeline


class Inter8PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c = {'contract_sha256': 'c' * 64, 'cohort': 'components_inter42_52_source131',
                  'jobs': [{'id': 'synthetic-' + str(i)} for i in range(8)],
                  'runtimes': {'5090': {'python': '/synthetic/python', 'gpu_uuid': 'GPU-synthetic'}},
                  'evaluator_integrity': {'files': {'pipeline.py': self.sha(pipeline.__file__)}}}
        (self.root / 'permits').mkdir()
        for split in ('validation', 'test'):
            pipeline.write(self.root / 'permits' / (split + '.json'), {})

        def require(value, message):
            if not value:
                raise ValueError(message)

        self.evaluator = SimpleNamespace(verify_contract=lambda root: self.c,
            verify_partial_metadata=lambda root, c: {'authorization_root': str(self.root)},
            sha=self.sha, read=lambda p: json.loads(Path(p).read_text()), require=require,
            verify_permit=lambda *args: {'deadline_unix': time.time() + 60},
            verify_validation_receipt=lambda r, c, j: r,
            verify_global_validation_gate=lambda root, c: json.loads((root / 'validation_gate.json').read_text()))

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def seed_validation(self, fail_index=None):
        for i, job in enumerate(self.c['jobs']):
            folder = self.root / 'results' / job['id']
            pipeline.write(folder / 'validation_replay.json', {'status': 'complete'})
            pipeline.write(folder / 'validation_worker_exit.json', {'job_id': job['id'],
                'contract_sha256': self.c['contract_sha256'], 'actual_child_exit_observed': True,
                'exit_code': 1 if i == fail_index else 0, 'completed_unix': time.time()})

    def test_any_validation_nonzero_exit_prevents_gate_creation(self):
        self.seed_validation(fail_index=5)
        with self.assertRaises(ValueError):
            pipeline.make_gate(self.root, self.c, self.evaluator)
        self.assertFalse((self.root / 'validation_gate.json').exists())

    def test_all_eight_actual_exits_are_bound_to_subset_gate(self):
        self.seed_validation()
        gate = pipeline.make_gate(self.root, self.c, self.evaluator)
        self.assertEqual(gate['job_count'], 8)
        self.assertEqual(len(gate['validation_receipts']), 8)
        self.assertEqual(len(gate['validation_worker_exits']), 8)
        self.assertFalse(gate['original36_global_gate_modified'])

    def test_existing_pipeline_claim_prevents_launch(self):
        pipeline.write(self.root / 'pipeline.claim', {'preserve': True})
        with self.assertRaises(FileExistsError):
            pipeline.run(self.root, self.evaluator, popen=lambda *a, **k: self.fail('Unexpected launch'))
        self.assertEqual(json.loads((self.root / 'pipeline.claim').read_text()), {'preserve': True})

    def test_worker_failure_records_actual_exit_and_test_never_starts(self):
        calls = []
        def child(argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(pid=1234, wait=lambda: 1)
        def gpu(argv, **kwargs):
            output = 'GPU-synthetic\n' if '--query-gpu=uuid' in argv else ''
            return subprocess.CompletedProcess(argv, 0, output, '')
        with patch.object(pipeline.subprocess, 'run', side_effect=gpu):
            with self.assertRaises(ValueError):
                pipeline.run(self.root, self.evaluator, popen=child)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index('--split') + 1], 'validation')
        self.assertFalse((self.root / 'validation_gate.json').exists())
        exit_receipt = json.loads((self.root / 'results/synthetic-0/validation_worker_exit.json').read_text())
        self.assertEqual(exit_receipt['exit_code'], 1)
        self.assertTrue(exit_receipt['actual_child_exit_observed'])
        self.assertFalse(json.loads((self.root / 'pipeline_failure.json').read_text())['automatic_retry'])

    def gpu_response(self, argv, **kwargs):
        output = 'GPU-synthetic\n' if '--query-gpu=uuid' in argv else ''
        return subprocess.CompletedProcess(argv, 0, output, '')

    def cpu_child(self, argv, **kwargs):
        # This fabricated sleeper executes no evaluator, data or model code.
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        self.addCleanup(lambda: child.poll() is not None or (child.kill(), child.wait()))
        self.children.append(child)
        return child

    def cancellation_record(self):
        paths = list((self.root / 'results/synthetic-0').glob('validation_worker_cancelled_exit_*.json'))
        self.assertEqual(len(paths), 1)
        value = json.loads(paths[0].read_text())
        self.assertTrue(value['actual_child_exit_observed'])
        self.assertLess(value['exit_code'], 0)
        self.assertFalse(value['automatic_retry'])
        self.assertFalse(value['normal_exit_receipt_present'])
        return value

    def test_launch_receipt_write_failure_stops_only_owned_cpu_child(self):
        self.children = []
        original_write = pipeline.write
        previous_handler = signal.getsignal(signal.SIGTERM)
        def fail_launch(path, value):
            if Path(path).parent.name == 'worker_launches':
                raise OSError('fabricated launch receipt write failure')
            return original_write(path, value)
        with patch.object(pipeline.subprocess, 'run', side_effect=self.gpu_response), \
                patch.object(pipeline, 'write', side_effect=fail_launch):
            with self.assertRaisesRegex(OSError, 'fabricated launch receipt'):
                pipeline.run(self.root, self.evaluator, popen=self.cpu_child)
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].poll())
        value = self.cancellation_record()
        self.assertEqual(value['pid'], self.children[0].pid)
        self.assertEqual(value['cleanup_actions'], ['TERM_own_child_group'])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
        self.assertFalse((self.root / 'validation_gate.json').exists())

    def test_parent_sigterm_stops_owned_cpu_child_and_restores_handler(self):
        self.children = []
        original_write = pipeline.write
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminate_after_launch(path, value):
            result = original_write(path, value)
            if Path(path).parent.name == 'worker_launches':
                os.kill(os.getpid(), signal.SIGTERM)
            return result
        with patch.object(pipeline.subprocess, 'run', side_effect=self.gpu_response), \
                patch.object(pipeline, 'write', side_effect=terminate_after_launch):
            with self.assertRaises(pipeline.PipelineTermination):
                pipeline.run(self.root, self.evaluator, popen=self.cpu_child)
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].poll())
        value = self.cancellation_record()
        self.assertEqual(value['cancellation_reason'], 'PipelineTermination')
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
        failure = json.loads((self.root / 'pipeline_failure.json').read_text())
        self.assertTrue(failure['cancelled_active_child']['actual_child_exit_observed'])
        self.assertFalse((self.root / 'validation_gate.json').exists())

    def test_keyboard_interrupt_cancels_active_child_with_actual_exit(self):
        child = SimpleNamespace(pid=2345)
        child.poll = lambda: None
        def wait(timeout=None):
            if timeout is None: raise KeyboardInterrupt()
            return -signal.SIGTERM
        child.wait = wait
        with patch.object(pipeline.subprocess, 'run', side_effect=self.gpu_response), \
                patch.object(pipeline.os, 'getpgid', return_value=2345), \
                patch.object(pipeline.os, 'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):
                pipeline.run(self.root, self.evaluator, popen=lambda *a, **k: child)
        kill.assert_called_once_with(2345, signal.SIGTERM)
        self.assertEqual(self.cancellation_record()['cancellation_reason'], 'KeyboardInterrupt')

    def test_bounded_term_wait_escalates_only_verified_child_group(self):
        from unittest.mock import Mock
        child = SimpleNamespace(pid=2345, poll=Mock(return_value=None),
            wait=Mock(side_effect=[subprocess.TimeoutExpired('synthetic', 10), -signal.SIGKILL]))
        with patch.object(pipeline.os, 'getpgid', return_value=2345), patch.object(pipeline.os, 'killpg') as kill:
            result = pipeline.stop_owned_child(child)
        self.assertTrue(result['actual_child_exit_observed'])
        self.assertEqual(result['exit_code'], -signal.SIGKILL)
        self.assertEqual(kill.call_args_list, [unittest.mock.call(2345, signal.SIGTERM),
                                             unittest.mock.call(2345, signal.SIGKILL)])
        self.assertEqual([c.kwargs['timeout'] for c in child.wait.call_args_list], [10, 10])

    def test_foreign_process_group_is_never_signalled(self):
        child = SimpleNamespace(pid=2345, poll=lambda: None)
        with patch.object(pipeline.os, 'getpgid', return_value=9999), patch.object(pipeline.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'ownership changed'):
                pipeline.stop_owned_child(child)
        kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
