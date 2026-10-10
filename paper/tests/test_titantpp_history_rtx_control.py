"""Offline fail-closed pipeline tests for host-specific RTX ownership."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('rtx_control_prepare',
    PROJECT / 'paper/scripts/prepare_titantpp_history_rtx_control.py')
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


def load_pipeline(host):
    originals = g.read_original_controls()
    text = g.transform_pipeline(originals['pipeline.py'], host)
    namespace = {'__name__': 'pipeline_unit_test', '__file__': '/control/pipeline.py'}
    exec(compile(text, '/control/pipeline.py', 'exec'), namespace)
    return namespace, originals


class RTXControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline, cls.originals = load_pipeline('5090')
        cls.real_contract = g.read(g.BUNDLE / 'execution_contract.json')

    def test_host_binding_and_provider_deadline_removed(self):
        for host in ('5080', '5090'):
            namespace, _ = load_pipeline(host)
            self.assertEqual(namespace['HOST'], host)
            text = g.transform_pipeline(self.originals['pipeline.py'], host)
            self.assertNotIn("c['budget']", text)
            self.assertIn("hard=start['deadline_unix']", text)
            self.assertIn("set(c['hosts'])=={'5080','5090'}", text)
            admission = g.transform_admission(self.originals['concurrency_admission.py'], host)
            self.assertNotIn("c['budget']", admission)
            self.assertIn("read(root / 'start_permit.json')['deadline_unix']", admission)

    def fixture(self, folder, host='5090'):
        pipeline, _ = load_pipeline(host)
        root = Path(folder)
        c = self.real_contract
        start = {'deadline_unix': 9999999999.0}
        qualification = {'status': 'passed', 'host': host, 'contract_sha256': g.canonical(c),
            'runtime': {**c['hosts'][host]['runtime_expected'],
                'gpu': {'uuid': c['hosts'][host]['gpu_uuid']}}}
        idle = {'gpu_pids': [], 'owned_processes': []}
        op = SimpleNamespace(validate=lambda *a, **kw: True,
            authorization=lambda *a, **kw: start,
            verify_training_permit=lambda *a, **kw: True)
        return pipeline, root, c, start, qualification, idle, op

    def phase_runner(self, root, c, start, qualification, *, failed=None, completed=None):
        pipeline = self.pipeline
        def run(root, c, start, phase):
            exit_receipt = {'phase': phase, 'contract_sha256': g.canonical(c),
                'actual_child_exit_observed': True, 'returncode': 1 if phase == failed else 0,
                'supervisor_exception': None}
            if phase == 'qualification':
                pipeline['write'](root / 'qualification/receipt.json', qualification, exclusive=True)
            elif phase == 'concurrency':
                receipt = {'status': 'passed', 'contract_sha256': g.canonical(c),
                    'arms': c['arms'], 'workers': 3, 'shape': [128, 256],
                    'simultaneous_ready_verified': True, 'capacity_admission_only': True,
                    'scientific_optimizer_updates': 0,
                    'actual_child_exits': {a: {'actual_exit_observed': True, 'returncode': 0} for a in c['arms']},
                    'files': {}}
                pipeline['write'](root / 'concurrency_admission/receipt.json', receipt, exclusive=True)
            elif phase == 'dispatch':
                count = completed if completed is not None else sum(j['host'] == qualification['host'] for j in c['jobs'])
                pipeline['write'](root / 'status.json', {'status': 'complete', 'completed_conditions': count})
                pipeline['write'](root / 'supervisor_exit.json', exit_receipt, exclusive=True)
            return exit_receipt
        return run

    def test_5090_completion_uses_nine_jobs_not_twelve(self):
        with tempfile.TemporaryDirectory() as folder:
            p, root, c, start, q, idle, op = self.fixture(folder)
            result = p['execute'](root, c, start, op,
                phase_runner=self.phase_runner(root, c, start, q), probe=lambda *a: idle)
            self.assertEqual(result['completed_conditions'], 9)
            self.assertTrue((root / 'training_permit.json').exists())

    def test_5080_completion_uses_three_jobs_not_twelve(self):
        with tempfile.TemporaryDirectory() as folder:
            p, root, c, start, q, idle, op = self.fixture(folder, '5080')
            result = p['execute'](root, c, start, op,
                phase_runner=self.phase_runner(root, c, start, q), probe=lambda *a: idle)
            self.assertEqual(result['completed_conditions'], 3)

    def test_foreign_all_host_completion_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p, root, c, start, q, idle, op = self.fixture(folder)
            with self.assertRaises(ValueError):
                p['execute'](root, c, start, op,
                    phase_runner=self.phase_runner(root, c, start, q, completed=12), probe=lambda *a: idle)
            self.assertFalse((root / 'pipeline_complete.json').exists())
            self.assertTrue((root / 'pipeline_failure.json').exists())

    def test_native_qualification_failure_never_dispatches(self):
        with tempfile.TemporaryDirectory() as folder:
            p, root, c, start, q, idle, op = self.fixture(folder)
            with self.assertRaises(ValueError):
                p['execute'](root, c, start, op,
                    phase_runner=self.phase_runner(root, c, start, q, failed='qualification'),
                    probe=lambda *a: idle)
            self.assertFalse((root / 'training_permit.json').exists())
            self.assertFalse((root / 'concurrency_admission').exists())

    def test_admission_nonzero_does_not_authorize_training(self):
        with tempfile.TemporaryDirectory() as folder:
            p, root, c, start, q, idle, op = self.fixture(folder)
            with self.assertRaises(ValueError):
                p['execute'](root, c, start, op,
                    phase_runner=self.phase_runner(root, c, start, q, failed='concurrency'),
                    probe=lambda *a: idle)
            self.assertFalse((root / 'training_permit.json').exists())
            self.assertFalse((root / 'supervisor_exit.json').exists())

    def test_owner_or_gpu_present_blocks_qualification(self):
        for field in ('gpu_pids', 'owned_processes'):
            with tempfile.TemporaryDirectory() as folder:
                p, root, c, start, q, idle, op = self.fixture(folder)
                observed = {**idle, field: [1234]}
                with self.assertRaises(ValueError):
                    p['execute'](root, c, start, op,
                        phase_runner=self.phase_runner(root, c, start, q),
                        probe=lambda *a: observed)
                self.assertFalse((root / 'qualification/receipt.json').exists())

    def test_foreign_qualification_gpu_runtime_or_host_rejected(self):
        for field in ('gpu', 'runtime', 'host'):
            with tempfile.TemporaryDirectory() as folder:
                p, root, c, start, q, idle, op = self.fixture(folder)
                if field == 'gpu':
                    q['runtime']['gpu']['uuid'] = c['hosts']['5080']['gpu_uuid']
                elif field == 'runtime':
                    q['runtime']['cudnn'] = 91900
                else:
                    q['host'] = '5080'
                exit_receipt = {'phase': 'qualification', 'contract_sha256': g.canonical(c),
                    'actual_child_exit_observed': True, 'returncode': 0, 'supervisor_exception': None}
                with self.assertRaises(ValueError):
                    p['make_training_permit'](c, start, q, 'raw', exit_receipt, idle)

    def test_failed_native_exit_or_unobserved_exit_rejected(self):
        for changes in ({'returncode': 1}, {'actual_child_exit_observed': False},
                        {'supervisor_exception': 'timeout'}):
            with tempfile.TemporaryDirectory() as folder:
                p, root, c, start, q, idle, op = self.fixture(folder)
                exit_receipt = {'phase': 'qualification', 'contract_sha256': g.canonical(c),
                    'actual_child_exit_observed': True, 'returncode': 0, 'supervisor_exception': None, **changes}
                with self.assertRaises(ValueError):
                    p['make_training_permit'](c, start, q, 'raw', exit_receipt, idle)

    def reseal_fixture(self, folder):
        root = Path(folder)
        (root / 'control').mkdir()
        (root / 'deployment').mkdir()
        (root / 'execution_contract.json').write_text(json.dumps(self.real_contract))
        package = root / 'deployment/package_5090.tar.gz'
        package.write_bytes(b'unchanged unit test archive')
        for name in self.real_contract['operation']['files']:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(g.BUNDLE / name, destination)
        previous = {'contract_sha256': g.CONTRACT_SHA,
            'files': {'deployment/package_5090.tar.gz': g.sha(package)},
            'project_files': {'paper/scripts/deploy_titantpp_history_rtx_handoff.py': '0' * 64}}
        path = root / 'control/local_tools_integrity.json'
        path.write_bytes(g.encoded(previous))
        return root, path.read_bytes(), package

    def test_prelaunch_reseal_preserves_prior_raw_and_remote_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root, previous, package = self.reseal_fixture(folder)
            package_sha = g.sha(package)
            result = g.reseal_prelaunch(SimpleNamespace(bundle=str(root)))
            preserved = root / result['preserved_previous_seal']
            self.assertEqual(preserved.read_bytes(), previous)
            self.assertEqual(g.sha(package), package_sha)
            self.assertEqual(g.sha(root / 'control/local_tools_integrity.json'), result['integrity_raw_sha256'])
            self.assertIn('paper/tests/test_titantpp_history_rtx_deployment.py', result['project_files'])

    def test_deployment_attempt_permit_or_run_blocks_resealing(self):
        for name in ('deployment/deploy_intent_5090.json', 'deployment/launch_receipt_5090.json',
                     'start_permit_5090.json', 'pipeline.claim', 'run'):
            with tempfile.TemporaryDirectory() as folder:
                root, previous, package = self.reseal_fixture(folder)
                path = root / name
                if name == 'run':
                    path.mkdir()
                else:
                    path.write_text('{}')
                with self.assertRaises(ValueError):
                    g.reseal_prelaunch(SimpleNamespace(bundle=str(root)))
                self.assertEqual((root / 'control/local_tools_integrity.json').read_bytes(), previous)

    def test_changed_remote_package_cannot_be_resealed(self):
        with tempfile.TemporaryDirectory() as folder:
            root, previous, package = self.reseal_fixture(folder)
            package.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'control/package drift'):
                g.reseal_prelaunch(SimpleNamespace(bundle=str(root)))
            self.assertEqual((root / 'control/local_tools_integrity.json').read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
