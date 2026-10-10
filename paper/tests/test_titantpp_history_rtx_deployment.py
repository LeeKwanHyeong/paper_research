import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import deploy_titantpp_history_rtx_handoff as deploy


class OwnershipReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temp.name)
        self.contract = deploy.read(deploy.B / 'execution_contract.json')
        self.parent = deploy.read(deploy.PARENT / 'execution_contract.json')
        self.reservation = deploy.read(deploy.RESERVATION)
        (self.bundle / 'execution_contract.json').write_text(json.dumps(self.contract))
        self.proof = {'status': 'released_after_planned_operational_handoff',
            'completed_count': 15, 'scientific_failures': 0, 'operational_exit_preserved': 1,
            'transfer_jobs': self.parent['jobs'][15:],
            'barrier_raw_sha256': self.reservation['barrier_raw_sha256'],
            'terminal_manifest_sha256': {j['id']: 'a'*64 for j in self.parent['jobs'][:15]},
            'supervisor_exit_raw_sha256': 'b'*64, 'failure_raw_sha256': 'c'*64,
            'pipeline_failure_raw_sha256': 'd'*64,
            'actual_idle_observation': {'gpu': deploy.GPU + ', NVIDIA A100', 'processes': [], 'gpu_processes': ''}}
        self.path = self.bundle / 'proof.json'

    def tearDown(self): self.temp.cleanup()

    def bind(self):
        self.path.write_text(json.dumps(self.proof))
        with patch.object(deploy, 'B', self.bundle):
            return deploy.bind_release(self.path)

    def test_exact_twelve_released_and_raw_bytes_bound(self):
        result = self.bind()
        self.assertEqual(result['execution_ids'], [j['id'] for j in self.contract['jobs']])
        self.assertEqual(result['source_receipt_raw_sha256'], deploy.sha(self.path))
        self.assertEqual(self.path.read_bytes(), (self.bundle / 'handoff/ownership_release_source.json').read_bytes())

    def test_live_owner_blocks_training_bridge(self):
        self.proof['actual_idle_observation']['processes'] = [{'pid': 100}]
        with self.assertRaisesRegex(ValueError, 'owner remains'): self.bind()
        self.assertFalse((self.bundle / 'ownership_release.json').exists())

    def test_only_reservation_is_not_release(self):
        self.proof['status'] = 'reserved_only_not_released'
        with self.assertRaisesRegex(ValueError, 'release incomplete'): self.bind()

    def test_changed_job_or_partial_completion_rejected(self):
        self.proof['transfer_jobs'] = self.proof['transfer_jobs'][:-1]
        with self.assertRaisesRegex(ValueError, 'job drift'): self.bind()
        self.proof['transfer_jobs'] = self.parent['jobs'][15:]
        self.proof['completed_count'] = 14
        with self.assertRaisesRegex(ValueError, 'release incomplete'): self.bind()

    def test_prior_release_bytes_not_overwritten(self):
        self.bind()
        original = (self.bundle / 'ownership_release.json').read_bytes()
        with self.assertRaises(FileExistsError): self.bind()
        self.assertEqual(original, (self.bundle / 'ownership_release.json').read_bytes())

    def test_foreign_gpu_and_missing_sha_proof_rejected(self):
        self.proof['actual_idle_observation']['gpu'] = 'GPU-FOREIGN, NVIDIA A100'
        with self.assertRaisesRegex(ValueError, 'Foreign A100 GPU'): self.bind()
        self.proof['actual_idle_observation']['gpu'] = deploy.GPU + ', NVIDIA A100'
        self.proof['failure_raw_sha256'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'SHA binding missing'): self.bind()

    def test_different_barrier_and_missing_terminal_rejected(self):
        self.proof['barrier_raw_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'raw binding drift'): self.bind()
        self.proof['barrier_raw_sha256'] = self.reservation['barrier_raw_sha256']
        self.proof['terminal_manifest_sha256'].pop(next(iter(self.proof['terminal_manifest_sha256'])))
        with self.assertRaisesRegex(ValueError, 'fifteen terminal'): self.bind()


if __name__ == '__main__': unittest.main()
