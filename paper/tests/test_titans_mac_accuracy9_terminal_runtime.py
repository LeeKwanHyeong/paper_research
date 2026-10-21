"""CPU-only synthetic finish contract/evidence/queue regression tests."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import titans_mac_accuracy9_terminal_runtime as m


class TerminalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.up = self.base / 'v5', self.base / 'v4'
        self.root.mkdir(); self.up.mkdir()
        self.keys = ['reference_' + str(i) for i in range(6)] + [
            'intermittent__seed42', 'intermittent__seed52', 'intermittent__seed62']
        jobs = [{'id': key, 'seed': 42} for key in self.keys]
        self.c = dict(root=str(self.up), source_root=str(self.up / 'source'),
                      python='/python', boot_id='boot', gpu_uuid='gpu',
                      limits={'queue_wall_seconds': 36000, 'evaluation_wall_seconds': 100},
                      absolute_host_deadline_unix=4000000000,
                      reference_jobs=self.keys[:6], training_jobs=self.keys[6:],
                      validation_jobs=self.keys, test_jobs=self.keys, jobs=jobs, environment={})
        self.old = {**self.c, 'root': str(self.base / 'v3')}
        self.epoch = m.time.time() - 20
        self.start = datetime.fromtimestamp(self.epoch, timezone.utc).isoformat()
        self.middle = datetime.fromtimestamp(self.epoch + 1, timezone.utc).isoformat()
        self.end = datetime.fromtimestamp(self.epoch + 2, timezone.utc).isoformat()
        self.f = dict(schema=m.SCHEMA, root=str(self.root), upstream_root=str(self.up),
                      upstream_contract_path=str(self.up / 'recovery_contract.json'),
                      upstream_contract_sha256=m.sha_json(self.c),
                      queue_started_at=self.start,
                      **{k: self.c[k] for k in ('python', 'boot_id', 'gpu_uuid', 'limits',
                                               'absolute_host_deadline_unix')})
        self.put(self.up / 'recovery_contract.json', self.c)
        for name in m.OPERATION_NAMES:
            p = self.root / name; p.parent.mkdir(exist_ok=True); p.write_text(name)
        self.f['operation_files'] = {name: m.sha_file(self.root / name) for name in m.OPERATION_NAMES}
        self.put(self.root / 'AUTHORIZATION.json', {'instruction': 'finish approved'})
        self.f['authorization'] = dict(instruction='finish approved',
            path=str(self.root / 'AUTHORIZATION.json'), sha256=m.sha_file(self.root / 'AUTHORIZATION.json'))
        self.sha_patch = patch.object(m, 'UPSTREAM_SHA', m.sha_json(self.c))
        self.sha_patch.start(); self.addCleanup(self.sha_patch.stop)
        self.r = SimpleNamespace(SCHEMA='recovery', verify_originals=Mock(return_value={'file_count': 1141}),
            worker_command=lambda c, path, job, stage: [c['python'], str(path), job['id'], stage],
            verify_receipt=lambda c, job, stage: self.bind(job['id'], stage),
            gpu_state=Mock(return_value={'compute_pids': []}), boot_id=lambda: 'boot',
            proc_identity=Mock(return_value=None))
        self.bridge = SimpleNamespace(r=self.r, MODULE='old_bridge',
            verify_files=Mock(return_value=(self.old, {})), terminal_ready=Mock(return_value=True),
            historical_identity_absent=Mock(return_value=True), initial_identity=Mock())
        self.worker = SimpleNamespace(effective_scientific=Mock(return_value={'jobs': jobs}),
            science=SimpleNamespace(verify_train_receipt=Mock(), verify_validation_gate=Mock()))
        self.failure = dict(stage='validation', at=self.middle,
            traceback='in _identity\n' + m.RACE_MESSAGE + '\n',
            children_left_running={self.keys[6]: 106, self.keys[8]: 108})
        self.stages = {}
        for stage, keys in (('reference', self.keys[:6]), ('train', self.keys[6:]), ('validation', self.keys)):
            result = dict(stage=stage, status='failed' if stage == 'validation' else 'complete',
                          exits={}, receipts={}, pending_job_ids=[],
                          failure={k: v for k, v in self.failure.items() if k != 'children_left_running'}
                                  if stage == 'validation' else None)
            for key in keys:
                folder = self.up / 'run' / key
                contract = self.c if stage == 'validation' else self.old
                cmd = self.r.worker_command(contract, Path(contract['root']) / 'recovery_contract.json',
                                            {'id': key}, stage)
                ex = self.exit_record(self.up / 'logs' / (key + '.' + stage), cmd,
                                      pid=100 + self.keys.index(key), key=key, stage=stage)
                self.put(folder / (stage + '_exit.json'), ex)
                self.put(folder / (stage + '_launch.json'), {k: ex[k] for k in ('pid', 'start_ticks', 'command')})
                self.put(folder / (stage + '_recovery_receipt.json'), {'status': 'complete'})
                self.put(folder / 'train_receipt.json', {'selected_epoch': 1})
                result['exits'][key], result['receipts'][key] = ex, self.bind(key, stage)
            self.stages[stage] = result
        cmd = ['/python', '-u', '-m', 'old_bridge', '--contract', self.f['upstream_contract_path'],
               '--contract-sha', m.sha_json(self.c), '--mode', 'dispatch']
        ex = self.exit_record(self.up / 'dispatcher', cmd, pid=3, code=1)
        ex['cleanup_error'] = None
        self.put(self.up / 'dispatcher_exit.json', ex)
        self.put(self.up / 'dispatcher_launch.json', dict(command=cmd, contract_sha256=m.sha_json(self.c),
            identity={'pid': 3, 'start_ticks': 500}))
        self.save_upstream()
        self.f['upstream_contract_raw_sha256'] = m.sha_file(self.up / 'recovery_contract.json')
        self.pin_evidence()
        self.put(self.up / 'source' / 'frozen.py', {'source': 1})

    def pin_evidence(self):
        self.f['upstream_evidence_files'] = {name: m.sha_file(self.up / name) for name in
            ('dispatcher_launch.json', 'dispatcher_exit.json', 'dispatch_failure.json',
             'validation_stage_receipt.json', 'final_receipt.json')}

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def bind(self, key, stage):
        name = 'run/' + key + '/' + stage + '_recovery_receipt.json'
        return {'path': name, 'sha256': m.sha_file(self.up / name)}

    def exit_record(self, prefix, command, *, pid=10, key=None, stage=None, code=0):
        ex = dict(pid=pid, start_ticks=500, command=command, actual_os_returncode=code,
                  timeout=False, boot_id='boot', started_at=self.start, observed_exit_at=self.end)
        if key is not None:
            ex.update(job_id=key, stage=stage)
        for stream in ('stdout', 'stderr'):
            p = Path(str(prefix) + '.' + stream + '.log'); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'closed')
            ex[stream + '_sha256'], ex[stream + '_bytes'] = m.sha_file(p), p.stat().st_size
        return ex

    def save_upstream(self):
        self.put(self.up / 'dispatch_failure.json', self.failure)
        self.put(self.up / 'validation_stage_receipt.json', self.stages['validation'])
        self.put(self.up / 'final_receipt.json', dict(schema='recovery', status='failed',
            contract_sha256=m.sha_json(self.c), stages=self.stages,
            candidate_test_count=0, held_out_test_evaluated=False))
        if 'upstream_evidence_files' in self.f:
            self.pin_evidence()

    def preflight(self):
        return m.preflight(self.f, self.c, self.bridge, self.worker)

    def test_admits_exact_completed_workers_without_rewriting_failed_receipts(self):
        before = m.snapshot(self.up)
        admission = self.preflight()
        self.assertEqual(admission['preexisting_v4_files'], before)
        self.assertEqual(admission['validated_stages']['validation']['status'], 'complete')
        self.assertEqual(m.read(self.up / 'validation_stage_receipt.json')['status'], 'failed')
        self.assertEqual(m.snapshot(self.up), before)
        self.assertEqual(self.preflight(), admission)
        self.bridge.verify_files.assert_called()
        self.bridge.terminal_ready.assert_called()
        self.r.verify_originals.assert_called()
        self.assertEqual(self.worker.science.verify_train_receipt.call_count, 18)

    def test_existing_test_or_gate_prevents_dispatch(self):
        for name in ('validation_gate.json', 'run/' + self.keys[0] + '/test.intent.json'):
            p = self.up / name; self.put(p, {})
            with self.assertRaisesRegex(ValueError, 'Existing gate or Test output'):
                self.preflight()
            p.unlink()

    def test_source_change_after_admission_blocks(self):
        self.preflight()
        self.put(self.up / 'source' / 'frozen.py', {'source': 2})
        with self.assertRaisesRegex(ValueError, 'Admission changed'):
            self.preflight()

    def test_nonzero_validation_exit_blocks(self):
        key = self.keys[0]
        self.stages['validation']['exits'][key]['actual_os_returncode'] = 1
        self.put(self.up / 'run' / key / 'validation_exit.json', self.stages['validation']['exits'][key])
        self.save_upstream()
        with self.assertRaisesRegex(ValueError, 'Actual owned OS exit'):
            self.preflight()

    def test_validation_receipt_missing_blocks(self):
        self.stages['validation']['receipts'].pop(self.keys[0]); self.save_upstream()
        with self.assertRaisesRegex(ValueError, 'Incomplete historical stage'):
            self.preflight()

    def test_nonterminal_v3_blocks(self):
        self.bridge.terminal_ready.return_value = False
        with self.assertRaisesRegex(ValueError, 'not terminal'):
            self.preflight()

    def test_unknown_failure_is_rejected(self):
        self.failure['traceback'] = 'different failure'; self.save_upstream()
        with self.assertRaisesRegex(ValueError, 'Unapproved upstream failure'):
            self.preflight()

    def test_closed_worker_log_sha_is_required(self):
        (self.up / 'logs' / (self.keys[0] + '.validation.stdout.log')).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Closed worker log changed'):
            self.preflight()

    def test_live_upstream_owner_blocks(self):
        self.bridge.historical_identity_absent.return_value = False
        with self.assertRaisesRegex(ValueError, 'still alive'):
            self.preflight()

    def test_queue_deadline_is_inherited_not_reset(self):
        deadline = m.queue_deadline(self.f)
        self.assertAlmostEqual(deadline, self.epoch + 36000, places=5)
        with patch.object(m.time, 'time', return_value=deadline + 1):
            with self.assertRaisesRegex(ValueError, 'Inherited queue deadline expired'):
                self.preflight()

    def test_existing_attempt_blocks_retry(self):
        self.put(self.root / 'dispatch_attempt.json', {})
        with self.assertRaisesRegex(ValueError, 'Existing terminal attempt'):
            self.preflight()

    def test_only_exact_test_additions_are_allowed(self):
        baseline = m.snapshot(self.up)
        self.put(self.up / 'validation_gate.json', {})
        self.assertEqual(set(m.verify_preserved(self.f, self.c, baseline)), {'validation_gate.json'})
        with self.assertRaisesRegex(ValueError, 'Incomplete fixed Test'):
            m.verify_preserved(self.f, self.c, baseline, completed=True)
        self.put(self.up / 'status_added.json', {})
        with self.assertRaisesRegex(ValueError, 'Unapproved addition'):
            m.verify_preserved(self.f, self.c, baseline)

    def test_preexisting_status_cannot_be_overwritten(self):
        self.put(self.up / 'status.json', {'original': True})
        baseline = m.snapshot(self.up)
        self.put(self.up / 'status.json', {'original': False})
        with self.assertRaisesRegex(ValueError, 'Preexisting v4 file changed'):
            m.verify_preserved(self.f, self.c, baseline)

    def test_complete_addition_matrix_has_73_files(self):
        baseline = m.snapshot(self.up)
        names = m.allowed_additions(self.c)
        self.assertEqual(len(names), 73)
        for name in names:
            self.put(self.up / name, {})
        self.assertEqual(len(m.verify_preserved(self.f, self.c, baseline, completed=True)), 73)

    def test_operation_source_change_blocks(self):
        (self.root / 'source/titans_mac_accuracy9_terminal_identity.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Operation file changed'):
            self.preflight()

    def test_pinned_failure_raw_bytes_cannot_change(self):
        p = self.up / 'dispatch_failure.json'
        p.write_text(p.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Pinned upstream evidence changed'):
            self.preflight()

    def test_contract_raw_bytes_cannot_change(self):
        p = self.up / 'recovery_contract.json'
        p.write_text(p.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'contract bytes changed'):
            self.preflight()

    def test_cleanup_rejects_training_lease_without_signalling(self):
        self.put(self.root / 'leases' / (self.keys[0] + '.json'), dict(job_id=self.keys[0], stage='train'))
        with patch.object(m.os, 'killpg') as kill:
            with self.assertRaisesRegex(ValueError, 'not owned by this Test attempt'):
                m.cleanup_test_leases(self.f, self.c, self.bridge)
        kill.assert_not_called()

    def test_cache_paths_are_separate_but_scientific_environment_is_retained(self):
        c = {**self.c, 'environment': {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONPATH': self.c['source_root']}}
        result = m.process_environment(self.f, c, 52)
        self.assertEqual(result['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.assertEqual(result['PYTHONPATH'], self.c['source_root'])
        self.assertEqual(result['PYTHONHASHSEED'], '52')
        for k in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR'):
            self.assertTrue(Path(result[k]).is_relative_to(self.root))

    def test_test_queue_launches_only_test_with_original_commands(self):
        class Child:
            returncode = 0
            def __init__(self, pid): self.pid = pid
            def poll(self): return 0
        launched = []
        def popen(command, **kwargs):
            launched.append((command, kwargs))
            return Child(1000 + len(launched))
        self.r._exit_record = lambda c, child, prefix, cmd, **kw: dict(pid=child.pid, start_ticks=500,
            command=cmd, actual_os_returncode=child.returncode, timeout=kw['timeout'])
        self.r.verify_receipt = Mock(side_effect=lambda c, job, stage: {'job': job['id'], 'stage': stage})
        helper = SimpleNamespace(make_terminal_identity=Mock(return_value=lambda *a: {'start_ticks': 500}))
        before = m.snapshot(self.up)
        with patch.object(m.subprocess, 'Popen', side_effect=popen), patch.object(m.time, 'sleep'):
            result = m.test_queue(self.f, self.c, self.bridge, helper)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['exits']), 9)
        self.assertEqual(len(launched), 9)
        self.assertTrue(all(cmd[-1] == 'test' for cmd, _ in launched))
        self.assertTrue(all(cmd[1] == self.f['upstream_contract_path'] for cmd, _ in launched))
        self.assertFalse((self.up / 'status.json').exists())
        self.assertTrue((self.root / 'status.json').exists())
        m.verify_preserved(self.f, self.c, before)

    def test_nonzero_test_exit_prevents_more_launches(self):
        class Child:
            pid, returncode = 1000, 1
            def poll(self): return 1
        self.r._exit_record = lambda c, child, prefix, cmd, **kw: dict(pid=child.pid, start_ticks=500,
            command=cmd, actual_os_returncode=1, timeout=False)
        helper = SimpleNamespace(make_terminal_identity=lambda **kw: lambda *args: {'start_ticks': 500})
        self.r.verify_receipt = Mock()
        with patch.object(m.subprocess, 'Popen', return_value=Child()) as launch, patch.object(m.time, 'sleep'):
            result = m.test_queue(self.f, self.c, self.bridge, helper)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(launch.call_count, 3)
        self.assertEqual(len(result['pending_job_ids']), 6)
        self.r.verify_receipt.assert_not_called()

    def test_failed_gate_never_launches_test(self):
        self.r.validation_gate = Mock(side_effect=ValueError('gate refused'))
        with patch.object(m.Path, 'cwd', return_value=Path(self.c['source_root'])), \
             patch.object(m, 'test_queue') as queue:
            with self.assertRaisesRegex(ValueError, 'gate refused'):
                m.dispatch(self.f, self.c, self.bridge, self.worker, Mock())
        queue.assert_not_called()
        self.assertFalse((self.root / 'final_receipt.json').exists())
        self.assertEqual(m.read(self.up / 'final_receipt.json')['status'], 'failed')

    def test_final_receipt_is_separate_and_binds_27_successful_stages(self):
        before = m.snapshot(self.up)
        self.r.validation_gate = lambda c, stages: self.put(self.up / 'validation_gate.json', {'status': 'passed'})
        def queue(*args):
            self.worker.science.verify_validation_gate.assert_called_once()
            for name in m.allowed_additions(self.c) - {'validation_gate.json'}:
                self.put(self.up / name, {})
            result = dict(stage='test', status='complete', failure=None, pending_job_ids=[],
                exits={k: dict(actual_os_returncode=0, timeout=False) for k in self.keys},
                receipts={k: {'status': 'complete'} for k in self.keys})
            self.put(self.root / 'test_stage_receipt.json', result)
            return result
        with patch.object(m.Path, 'cwd', return_value=Path(self.c['source_root'])), \
             patch.object(m, 'test_queue', side_effect=queue):
            self.assertEqual(m.dispatch(self.f, self.c, self.bridge, self.worker, Mock()), 0)
        final = m.read(self.root / 'final_receipt.json')
        self.assertEqual(final['actual_scientific_worker_exit0_count'], 27)
        self.assertEqual(final['candidate_test_count'], 9)
        self.assertEqual(final['new_training_worker_count'], 0)
        self.assertEqual(final['new_validation_worker_count'], 0)
        self.assertEqual(len(final['added_v4_files']), 73)
        self.assertEqual(final['admission']['sha256'], m.sha_file(self.root / 'admission.json'))
        self.assertEqual(final['artifact_manifest']['sha256'], m.sha_file(self.root / 'artifact_manifest.json'))
        self.assertEqual(m.read(self.up / 'final_receipt.json')['status'], 'failed')
        m.verify_preserved(self.f, self.c, before, completed=True)

    def test_unknown_gpu_workload_prevents_test_launch(self):
        self.r.gpu_state.side_effect = ValueError('Unknown GPU workload')
        helper = SimpleNamespace(make_terminal_identity=lambda **kw: Mock())
        with patch.object(m.subprocess, 'Popen') as launch:
            result = m.test_queue(self.f, self.c, self.bridge, helper)
        self.assertEqual(result['status'], 'failed')
        launch.assert_not_called()
        self.assertEqual(len(result['pending_job_ids']), 9)


if __name__ == '__main__':
    unittest.main()
