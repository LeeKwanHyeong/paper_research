"""CPU adapter contracts on fabricated inputs; no study data or remote access."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.TPPs.CountAwareTitanCNNGRUResidualCap import CAPPED, CNN, FULL
from paper.scripts import run_titantpp_gru_residual_cap as runner


class GRUResidualCapAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.install_hooks()

    def contract(self):
        return {"training": deepcopy(runner.TRAINING), "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False, "test_evaluation": False,
            "source": {"base_git_revision": "fabricated_revision"}}

    def test_initial_states_and_all_active_elements_update_three_seeds(self):
        for seed in (42, 52, 62):
            receipt = runner.synthetic_check(seed=seed)
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["active_correction_gradient_parameter_count"],
                {CNN: 576, FULL: 25308, CAPPED: 25308})
            self.assertFalse(receipt["real_data_loaded"])
            self.assertFalse(receipt["real_training_started"])

    def test_baseline_routes_metadata_and_initial_states_unchanged(self):
        data = runner.fabricated_data()
        for arm in (CNN, FULL):
            torch.manual_seed(52)
            model, meta = runner.build_model(data, arm)
            torch.manual_seed(52)
            original, original_meta = runner.original.build_model(data, arm)
            self.assertIs(type(model), type(original))
            self.assertEqual(meta, original_meta)
            self.assertEqual(runner.metadata(arm=arm), runner.original.metadata(arm=arm))
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_raf_length_preserves_parent_capacity(self):
        data = runner.fabricated_data()
        data["loader"]["max_seq_len"] = 84
        counts = {}
        for arm in (CNN, FULL, CAPPED):
            model, _ = runner.build_model(data, arm)
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(counts, {CNN: 79427, FULL: 104159, CAPPED: 104159})

    def test_training_arguments_lock_loss_head_loader_budget_and_readonly_references(self):
        data = runner.fabricated_data()
        contract = self.contract()
        fake = SimpleNamespace(force_rerun=False, checkpoint_monitor="validation_raw_quantity_rmse")
        with patch.object(runner.base, "training_args", return_value=deepcopy(fake)):
            actual = runner.training_args(contract, data, Path("fabricated_new_cap_fit"))
            self.assertEqual((actual.epochs, actual.min_epochs, actual.early_stopping_patience), (300, 40, 40))
            self.assertEqual(actual.execution_role, "fresh_gru_residual_cap_validation")
            for arm in (CNN, FULL):
                with self.assertRaises(ValueError):
                    runner.training_args(contract, data, Path("fabricated_new_cap_fit"), arm)
            for key, value in (("evaluation_scope", "test"), ("test_evaluation", True), ("held_out_test_evaluated", True)):
                bad = {**contract, key: value}
                with self.assertRaises(ValueError):
                    runner.training_args(bad, data, Path("fabricated_new_cap_fit"))
            for key, value in (("maximum_epochs", 301), ("minimum_epochs", 10), ("patience", 20),
                    ("monitor", "validation_time_nll"), ("warm_start", True), ("tie", "last")):
                bad = deepcopy(contract)
                bad["training"][key] = value
                with self.assertRaises(ValueError):
                    runner.training_args(bad, data, Path("fabricated_new_cap_fit"))
            for section, key, value in (("model", "lambda_log_qty", .5), ("model", "lambda_tail", 1.),
                    ("model", "time_head_lr_multiplier", 2.), ("model", "time_head_mode", "exponential"),
                    ("optimizer", "lr", .01), ("optimizer", "weight_decay", 0.),
                    ("optimizer", "grad_clip", 2.), ("loader", "batch_size", 64),
                    ("loader", "drop_last", True), ("loader", "num_workers", 1)):
                bad = deepcopy(data)
                bad[section][key] = value
                with self.assertRaises(ValueError):
                    runner.training_args(contract, bad, Path("fabricated_new_cap_fit"))
            with patch.object(Path, "glob", return_value=iter([Path("old.pt")])):
                with self.assertRaises(ValueError):
                    runner.training_args(contract, data, Path("fabricated_new_cap_fit"))

    def test_real_argument_builder_on_fabricated_metadata_without_data_access(self):
        data = runner.fabricated_data()
        with patch.object(runner.base, "prepare_admitted_data", side_effect=AssertionError("data access")):
            actual = runner.training_args(self.contract(), data, Path("fabricated_new_cap_fit"))
        self.assertEqual(actual.model_role, runner.role_for_arm(CAPPED))
        self.assertEqual((actual.lr, actual.weight_decay, actual.grad_clip), (.001, .01, 1.))
        self.assertEqual((actual.lambda_log_qty, actual.lambda_tail, actual.time_head_lr_multiplier), (1., 0., 1.))
        self.assertEqual(actual.time_head_mode, "heteroscedastic_lognormal_duration")
        self.assertEqual((actual.epochs, actual.min_epochs, actual.early_stopping_patience), (300, 40, 40))
        self.assertFalse(actual.force_rerun)

    def test_launch_rejects_wrong_role_or_multiple_arms(self):
        from paper.scripts.count_aware_tpp_backbone import observed_time
        for args, backbones in ((SimpleNamespace(model_role=runner.role_for_arm(FULL)), (CAPPED,)),
                (SimpleNamespace(model_role=runner.role_for_arm(CAPPED)), (FULL,)),
                (SimpleNamespace(model_role=runner.role_for_arm(CAPPED)), (CAPPED, FULL))):
            with self.assertRaises(ValueError):
                observed_time.validate_launch(args, backbones, (runner.VARIANT,))

    def test_native_training_entry_and_cli_training_are_blocked(self):
        with patch.object(runner.base, "prepare_admitted_data", side_effect=AssertionError("data access")):
            with self.assertRaises(PermissionError):
                runner.run_fit(self.contract(), "5080", Path("fabricated"), {}, 0)
        result = subprocess.run([sys.executable, "-m", "paper.scripts.run_titantpp_gru_residual_cap", "--train"],
            cwd=runner.ROOT, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--synthetic-check", result.stderr)

    def test_replay_uses_validation_population_and_cap_checkpoint(self):
        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
        data = runner.fabricated_data()
        data["loader"].update(batch_size=2, lookback_weeks=168)
        data.update(quantity_boundaries_all_train_rows=[1, 2, 2, 3], history_boundaries=[1, 3],
            inherited_data_identity={"populations": {"validation": {"target_count": 2}}})
        model, meta = runner.build_model(data, CAPPED)
        interface = {"time_head": model.time_head_contract()}
        state = model.state_dict()
        payload = {"backbone": CAPPED, "encoder_config": meta, "model_state_dict": state,
            "model_state_sha256": canonical_state_dict_sha256(state),
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False, "variant": runner.VARIANT,
            "interface_meta": interface, "resume_identity": {"backbone": CAPPED, "interface_meta": interface,
                "checkpoint_monitor": "validation_raw_quantity_rmse",
                "arguments": {"model_role": runner.role_for_arm(CAPPED)}}}
        dt = torch.tensor([[0., 1., 3., 5.], [0., 4., 2., 6.]])
        qty = torch.tensor([[0., 2., 4., 8.], [1., 0., 5., 2.]])
        mask = torch.ones_like(dt, dtype=torch.bool)
        loader = [(None, dt, mask, None, qty)]
        with patch("paper.scripts.run_taxi_quantity_interface_ablation.make_loader", return_value=loader) as make:
            receipt = runner._replay_validation(payload, Path("fabricated_not_opened.pt"), data, None,
                lambda: None, device="cpu")
        self.assertEqual(make.call_args.kwargs["target_split"], "validation")
        self.assertEqual(receipt["count"], 2)


if __name__ == "__main__":
    unittest.main()
