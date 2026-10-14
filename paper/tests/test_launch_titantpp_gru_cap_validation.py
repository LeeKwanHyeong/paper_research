"""Operation boundaries: synthetic metadata only, no native execution."""
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper.scripts import launch_titantpp_gru_cap_validation as launch


class OperationalBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = launch.read(launch.PREPARED / "execution_contract.json")
        cls.parent = launch.read(launch.PARENT / "execution_contract.json")

    def overlay(self):
        c = deepcopy(self.prepared)
        c.update(science_preparation={"path": "preparation/execution_contract.json",
            "raw_sha256": launch.PREPARED_RAW, "canonical_sha256": launch.PREPARED_CANONICAL},
            execution_authorization={"approved": True, "user_instruction": launch.USER_APPROVAL,
                "fresh_fit_count": 9, "Test": False, "automatic_retry": False,
                "preserve_original_CNN_FULL": True},
            runner_modules={launch.CAPPED: "paper.scripts.run_titantpp_gru_residual_cap"},
            evaluation_scope="validation_only", parallelism={"mode": "sequential_per_host", "workers_per_host": 1},
            operation={"files": {}, "files_sha256": launch.canonical({})})
        c["hosts"] = {}
        for host, planned in self.prepared["hosts"].items():
            old = self.parent["hosts"][host]; root = planned["planned_root"]
            spec = deepcopy(old)
            spec.update(root=root, source_root=root + "/source", operation_root=root + "/operation",
                output_dir=root + "/run", tmux=launch.SCHEMA + "_" + host)
            spec["environment"] = {k: v.replace(old["root"], root) for k, v in old["environment"].items()}
            c["hosts"][host] = spec
        return c

    def validate(self, c):
        return launch.validate_contract(c, self.prepared, self.parent)

    def test_fixed_contract_accepts_scoped_authorization(self):
        self.validate(self.overlay())

    def test_science_and_budget_changes_blocked(self):
        for path, value in [(('training', 'maximum_epochs'), 301), (('limits', 'per_condition_seconds'), 129601),
                            (('candidate', 'rho'), .5), (('objective_weights', 'unexpected'), 1)]:
            with self.subTest(path=path):
                c = self.overlay(); c[path[0]][path[1]] = value
                with self.assertRaises(ValueError): self.validate(c)

    def test_ownership_model_test_and_approval_changes_blocked(self):
        for mutate in [lambda c: c['jobs'][0].update(host='5090'),
                       lambda c: c['jobs'][0].update(arm='titantpp_cnn_gru54'),
                       lambda c: c.update(evaluation_scope='test'),
                       lambda c: c['execution_authorization'].update(approved=False),
                       lambda c: c['execution_authorization'].update(automatic_retry=True)]:
            c = self.overlay(); mutate(c)
            with self.assertRaises(ValueError): self.validate(c)

    def test_runtime_root_and_environment_changes_blocked(self):
        for mutate in [lambda s: s.update(gpu_uuid='GPU-foreign'), lambda s: s.update(root='/tmp/foreign'),
                       lambda s: s['runtime_expected'].update(cudnn=1),
                       lambda s: s['environment'].update(PYTHONHASHSEED='52')]:
            c = self.overlay(); mutate(c['hosts']['5080'])
            with self.assertRaises(ValueError): self.validate(c)

    def test_history_preparation_authorization_remains_unchanged(self):
        c = self.overlay(); self.validate(c)
        self.assertFalse(c['authorization']['execution_authorized'])
        self.assertEqual(c['source'], self.prepared['source'])

    def test_foreign_bundle_rejected_before_remote(self):
        with self.assertRaises(ValueError): launch.check_bundle('/tmp/other-campaign')

    def test_no_prepare_overwrite(self):
        with patch.object(Path, 'exists', return_value=True), patch.object(launch, 'frozen_preparation') as read:
            with self.assertRaises(ValueError): launch.prepare_execution()
            read.assert_not_called()

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['../foreign', '/absolute']:
                with self.assertRaises(ValueError): launch.safe(folder, name)

    def qualified(self, host='5080'):
        c = self.overlay(); q = {'inputs': {}, 'initialization': {}, 'synthetic': {},
            'scientific_optimizer_updates': 0, 'held_out_test_evaluated': False,
            'real_Validation_predictions': False}
        for job in (j for j in c['jobs'] if j['host'] == host):
            dataset = job['dataset']; data = next(d for d in c['datasets'] if d['dataset_id'] == dataset)
            receipt = {'held_out_materialized': False, 'populations': data['inherited_data_identity']['populations']}
            q['inputs'][dataset] = {'input_receipt': receipt, 'input_receipt_sha256': launch.canonical(receipt),
                'interface_sha256': {launch.CAPPED: 'a' * 64}}
            q['initialization'][dataset] = {str(s): {launch.CAPPED: 'b' * 64} for s in c['seeds']}
            q['synthetic'][dataset + '__' + str(job['seed'])] = {'status': 'passed', 'device': 'cuda:0',
                'seed': job['seed'], 'shape': [128, data['loader']['max_seq_len']],
                'synthetic_optimizer_updates': 4, 'scientific_optimizer_updates': 0,
                'held_out_test_evaluated': False, 'real_data_loaded': False,
                'initial_FULL_candidate_tensors_outputs': 'exact',
                'registered_parameters': job['expected_registered_parameters'],
                'active_parameters': job['expected_active_parameters'],
                'initial_state_sha256': {launch.CAPPED: 'b' * 64},
                'cap_bound_zero_unsaturated_saturated': 'passed', 'all_gradients_finite': True,
                'correction_parameter_names_with_gradient': ['fabricated_parameter'],
                'withheld_target': 'invariant', 'future_prefix': 'invariant', 'peak_allocated_bytes': 100}
        return c, q

    def test_full_qualification_required_for_each_host(self):
        for host in ('5080', '5090'):
            c, q = self.qualified(host); launch.verify_qualification_coverage(q, c, host)
            q['synthetic'].pop(next(iter(q['synthetic'])))
            with self.assertRaises(ValueError): launch.verify_qualification_coverage(q, c, host)

    def test_fabricated_success_cannot_skip_native_checks(self):
        for key, value in [('device', 'cpu'), ('shape', [1, 6]), ('synthetic_optimizer_updates', 0),
                           ('registered_parameters', 1), ('withheld_target', 'unknown'),
                           ('all_gradients_finite', False), ('real_data_loaded', True)]:
            c, q = self.qualified(); next(iter(q['synthetic'].values()))[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.verify_qualification_coverage(q, c, '5080')

    def test_qualification_target_and_tensor_identity_required(self):
        c, q = self.qualified()
        item = next(iter(q['inputs'].values())); item['input_receipt_sha256'] = 'c' * 64
        with self.assertRaises(ValueError): launch.verify_qualification_coverage(q, c, '5080')
        c, q = self.qualified()
        next(iter(q['synthetic'].values()))['initial_state_sha256'][launch.CAPPED] = 'c' * 64
        with self.assertRaises(ValueError): launch.verify_qualification_coverage(q, c, '5080')


if __name__ == '__main__':
    unittest.main()
