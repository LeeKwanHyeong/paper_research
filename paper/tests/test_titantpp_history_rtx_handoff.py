import ast
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from paper.scripts import titantpp_history_rtx_handoff as h

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / 'search_artifacts/titantpp_contribution_history_recovery_20261010_v1'

class HandoffRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'claims').mkdir()
        self.contract = json.loads((BUNDLE / 'execution_contract.json').read_text())
        self.current, self.waiting = h.groups(self.contract)
        self.probe = lambda root: {'observed_unix': time.time(), 'gpu': h.GPU, 'gpu_processes': 'owned', 'processes': [{'pid': 5, 'ppid': 4, 'argv': ' '.join([h.PYTHON, str(root / 'operation/campaign.py'), '--contract', str(root / 'execution_contract.json'), '--host', 'a100', '--mode', 'dispatch'])}]}
        self.write('status.json', {'status': 'running', 'updated_unix': time.time(), 'active_jobs': {j['id']: {} for j in self.current}, 'supervisor_pid': 5})
    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
    def reserve(self):
        return h.reserve_boundary(self.root, 'user approved RTX handoff', probe=self.probe, verifier=lambda root: self.contract)
    def test_reservation_is_not_release_or_scientific_claim(self):
        out = self.reserve()
        self.assertEqual(out['status'], 'reserved_only_not_released')
        self.assertFalse(out['payload']['scientific_worker_created'])
        self.assertEqual(len(list((self.root / 'claims').iterdir())), 1)
        self.assertEqual(h.sha(Path(out['barrier_path'])), out['barrier_raw_sha256'])
    def test_existing_worker_claim_preserved(self):
        p = self.root / 'claims' / (self.waiting[0]['id'] + '.json')
        p.write_text('actual worker claim')
        with self.assertRaisesRegex(ValueError, 'already attempted'):
            self.reserve()
        self.assertEqual(p.read_text(), 'actual worker claim')
    def test_atomic_race_does_not_overwrite_dispatcher_claim(self):
        p = self.root / 'claims/race.json'
        p.write_text('dispatcher won')
        with self.assertRaises(FileExistsError):
            h.atomic_exclusive(p, {'reservation': True})
        self.assertEqual(p.read_text(), 'dispatcher won')
        self.assertEqual(list((self.root / 'claims').iterdir()), [p])
    def test_transfer_run_or_lease_blocks_reservation(self):
        for directory in ('run', 'leases', 'worker_exits', 'progress', 'logs'):
            with self.subTest(directory=directory):
                suffix = '' if directory == 'run' else '.log' if directory == 'logs' else '.json'
                path = self.root / directory / (self.waiting[-1]['id'] + suffix)
                path.parent.mkdir(exist_ok=True)
                path.touch()
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    self.reserve()
                path.unlink()
    def test_changed_active_batch_fails_closed(self):
        status = h.read(self.root / 'status.json')
        status['active_jobs'] = {self.waiting[0]['id']: {}}
        self.write('status.json', status)
        with self.assertRaisesRegex(ValueError, 'batch changed'):
            self.reserve()
    def test_actual_sealed_dispatch_finishes_current_then_blocks_next(self):
        out = self.reserve()
        tree = ast.parse((BUNDLE / 'operation/campaign.py').read_text())
        dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dispatch')
        # The exact frozen boundary statement, rather than a reimplementation.
        claim_write = next(n for n in ast.walk(dispatch) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'write_json' and "'claims'" in ast.unparse(n))
        events = ['current_workers_terminal', 'current_batch_exposure_audit']
        def write_json(path, payload, exclusive=False):
            with path.open('xb' if exclusive else 'wb') as stream:
                stream.write(json.dumps(payload).encode())
        ns = {'common': SimpleNamespace(write_json=write_json), 'root': self.root, 'job': self.waiting[0], 'c': self.contract, 'deadline': 999, 'sha_json': h.canonical}
        with self.assertRaises(FileExistsError):
            exec(compile(ast.Module(body=[claim_write], type_ignores=[]), 'sealed_dispatch_boundary', 'exec'), ns)
            events.append('next_worker_started')
        self.assertNotIn('next_worker_started', events)
        self.assertEqual(h.sha(Path(out['barrier_path'])), out['barrier_raw_sha256'])
    def test_release_rejects_changed_barrier_or_live_owner(self):
        out = self.reserve()
        with self.assertRaisesRegex(ValueError, 'Reservation changed'):
            h.confirm_release(self.root, 'wrong', probe=self.probe, verifier=lambda root: self.contract)
        with self.assertRaisesRegex(ValueError, 'absence required'):
            h.confirm_release(self.root, out['barrier_raw_sha256'], probe=self.probe, verifier=lambda root: self.contract)
    def prepare_terminal(self):
        out = self.reserve()
        self.write('supervisor_exit.json', {'actual_child_exit_observed': True, 'returncode': 1, 'contract_sha256': h.CONTRACT, 'supervisor_exception': None})
        self.write('failure.json', {'type': 'FileExistsError', 'message': '[Errno 17] File exists: ' + out['barrier_path'], 'active_jobs': [], 'owned_children_stopped': [], 'active_batch': self.waiting[:3]})
        self.write('pipeline_failure.json', {'phase': 'dispatch', 'contract_sha256': h.CONTRACT})
        for job in self.contract['jobs'][:15]:
            reused = job['id'] in self.contract['recovery']['reused_jobs']
            prefix = 'recovery_parent/run' if reused else 'run'
            self.write(prefix + '/' + job['id'] + '/terminal_manifest.json', {'job': job, 'contract_sha256': self.contract['recovery']['parent_contract_sha256'] if reused else h.CONTRACT, 'scientific_success': True, 'status': 'complete', 'held_out_test_evaluated': False, 'files': {}})
        return out
    def confirm(self, out):
        return h.confirm_release(self.root, out['barrier_raw_sha256'], probe=lambda root: {'processes': [], 'gpu_processes': '', 'observed_unix': time.time()}, verifier=lambda root: self.contract)
    def test_release_requires_all15_terminals_and_preserves_exit1(self):
        out = self.prepare_terminal()
        released = self.confirm(out)
        self.assertEqual(released['completed_count'], 15)
        self.assertEqual(released['operational_exit_preserved'], 1)
        self.assertFalse(released['parent_scientific_completion'])
        path = self.root / 'run' / self.current[-1]['id'] / 'terminal_manifest.json'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.confirm(out)
    def test_unexpected_failure_or_stopped_worker_never_releases(self):
        out = self.prepare_terminal()
        failure = h.read(self.root / 'failure.json')
        failure['owned_children_stopped'] = [{'worker_group_pid': 123}]
        self.write('failure.json', failure)
        with self.assertRaisesRegex(ValueError, 'Unexpected failure'):
            self.confirm(out)

if __name__ == '__main__':
    unittest.main()
