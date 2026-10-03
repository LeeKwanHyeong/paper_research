"""Reject contract changes that would invalidate or expand a diagnostic."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('cell_contract_validator', ROOT / 'paper/scripts/validate_intermittent_cell_contract.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class CellContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((ROOT / 'paper/contracts/intermittent_mixed_cell_diagnosis_v1.json').read_text())

    def reject(self, mutation, *, verify_local_evidence=False):
        altered = copy.deepcopy(self.original)
        mutation(altered)
        with self.assertRaises((ValueError, KeyError)):
            validator.validate_contract(altered, verify_local_evidence=verify_local_evidence)

    def test_verified_contract_and_local_source_evidence(self):
        result = validator.validate_contract(self.original, verify_local_evidence=True)
        self.assertTrue(result['valid_measurement_contract'])
        self.assertFalse(result['gpu_execution_ready'])

    def test_cannot_self_authorize_gpu(self):
        self.reject(lambda c: c['authorization'].update(gpu_execution_authorized=True))

    def test_cannot_reuse_old_training_approval(self):
        self.reject(lambda c: c['authorization'].update(inherited_training_approval_applies=True))

    def test_held_out_target_rejected(self):
        self.reject(lambda c: c['validation'].update(split='test'))

    def test_state_missing_or_duplicated(self):
        self.reject(lambda c: c['states'].pop())
        self.reject(lambda c: c['states'].__setitem__(1, copy.deepcopy(c['states'][0])))

    def test_other_checkpoint_or_time_selector_rejected(self):
        self.reject(lambda c: c['states'][0]['checkpoint'].update(path='/tmp/other.pt'))
        self.reject(lambda c: c['states'][0].update(selector='legacy_time_loss'))

    def test_different_epochs_not_called_common_step(self):
        self.reject(lambda c: c['states'][0].update(epoch=120, global_step=369240))

    def test_boundary_and_equality_changes_rejected(self):
        self.reject(lambda c: c['cells'].update(quantity_boundaries=[2, 31, 47, 187]))
        self.reject(lambda c: c['cells'].update(boundary_equality='upper_bin'))

    def test_sample_budget_and_identity_changes_rejected(self):
        self.reject(lambda c: c['train_probe'].update(selected_sample_count=16384))
        self.reject(lambda c: c['train_probe'].update(expected_selected_indices_sha256='0' * 64))

    def test_subset_forward_or_cell_mean_rejected(self):
        self.reject(lambda c: c['train_probe'].update(forward_graphs_per_batch=15))
        self.reject(lambda c: c['train_probe'].update(cell_normalization='mean(loss_vector[cell_mask])'))

    def test_coefficient_refit_and_clipping_change_rejected(self):
        self.reject(lambda c: c['train_probe'].update(coefficient_refit_allowed=True))
        self.reject(lambda c: c['clipping'].update(max_norm=5))
        self.reject(lambda c: c['arms']['mixed_original']['condition']['mixed_objective'].update(alpha=0))

    def test_budget_extension_or_retry_rejected(self):
        self.reject(lambda c: c['limits'].update(max_wall_seconds=14400))
        self.reject(lambda c: c['limits'].update(automatic_retry_or_resume=True))

    def test_source_drift_rejected(self):
        self.reject(lambda c: c['source']['files'].update({'models/TPPs/CountAwareTPP.py': '0' * 64}))

    def test_parent_model_change_rejected_even_with_evidence(self):
        self.reject(lambda c: c['dataset']['model'].update(hidden_dim=128), verify_local_evidence=True)

    def test_coordinated_runtime_change_rejected_against_original(self):
        def change(c):
            c['server']['runtime_expected']['numpy'] = '0.0.0'
            c['training_identity']['runtime']['numpy'] = '0.0.0'
        self.reject(change, verify_local_evidence=True)

    def test_cell_clipping_semantics_change_rejected(self):
        self.reject(lambda c: c['clipping'].update(application='clip each cell independently'), verify_local_evidence=True)

    def test_replay_tolerance_relaxation_rejected(self):
        self.reject(lambda c: c['validation_gates'].update(replay_absolute_tolerance=1000.0), verify_local_evidence=True)

    def test_other_frozen_definition_change_rejected(self):
        self.reject(lambda c: c['train_probe'].update(zero_norm_policy='zero means perfect alignment'), verify_local_evidence=True)


if __name__ == '__main__':
    unittest.main()
