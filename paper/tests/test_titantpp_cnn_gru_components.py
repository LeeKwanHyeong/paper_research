"""Fabricated CPU contracts for the CNN×GRU factorial; no research data access."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.TPPs.CountAwareTitanCNNGRU import CountAwareTitanCNNGRU, KERNEL_KEYS, observed_context, causal_depthwise_residual
from models.TPPs.CountAwareTitanCNNHistoryControls import CountAwareTitanCNNHistoryControl
from models.TPPs.CountAwareTitanCNNGRUComponents import (
    FULL, NONE, CNN, GRU, ARMS, ALL_ARMS, ACTIVE_PARAMETERS, ACTIVE_CORRECTION_PARAMETERS,
    CountAwareTitanCNNGRUComponents, correction_parameters, validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_gru_components as runner
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


class CNNGRUComponentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.install_hooks()
        cls.data = runner.fabricated_data()

    def model(self, arm, seed=42):
        torch.manual_seed(seed)
        model, meta = runner.build_model(self.data, arm)
        return model.eval(), meta

    def inputs(self):
        dt = torch.tensor([[0., 1., 3., 5., 2., 7.], [0., 4., 2., 6., 1., 8.]])
        qty = torch.tensor([[0., 2., 4., 8., 3., 1.], [1., 0., 5., 2., 7., 4.]])
        return dt, qty, torch.ones_like(dt, dtype=torch.bool)

    def activate(self, model):
        generator = torch.Generator().manual_seed(77)
        with torch.no_grad():
            for parameter in correction_parameters(model).values():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .05)

    def payload(self, arm):
        model, meta = self.model(arm)
        interface = {"time_head": model.time_head_contract()}
        return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False, "variant": runner.VARIANT,
            "interface_meta": interface, "resume_identity": {"backbone": arm, "interface_meta": interface,
                "checkpoint_monitor": "validation_raw_quantity_rmse",
                "arguments": {"model_role": runner.role_for_arm(arm)}}}

    def test_full_and_none_exact_original_routes_metadata_forward_gradient_update(self):
        dt, qty, mask = self.inputs()
        for arm, expected_type in ((FULL, CountAwareTitanCNNGRU), (NONE, CountAwareTitanCNNHistoryControl)):
            actual, meta = self.model(arm)
            torch.manual_seed(42)
            expected, original_meta = runner.original.build_model(self.data, arm)
            self.assertIs(type(actual), expected_type)
            self.assertEqual(meta, original_meta)
            for mode in (False, True):
                actual.train(mode)
                expected.train(mode)
                torch.manual_seed(91)
                a = target_outputs(expected, dt, mask, qty, lambda_log_qty=1.)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(91)
                b = target_outputs(actual, dt, mask, qty, lambda_log_qty=1.)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), (arm, key))
            a["joint_loss"].mean().backward()
            b["joint_loss"].mean().backward()
            expected_params = dict(expected.named_parameters())
            for name, parameter in actual.named_parameters():
                reference = expected_params[name]
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.equal(parameter.grad, reference.grad), name)
                else:
                    self.assertIsNone(parameter.grad, name)
            for model in (actual, expected):
                torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01).step()
            for name, value in expected.state_dict().items():
                self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)

    def test_all_common_initial_tensors_and_rng_exact_all_three_seeds(self):
        for seed in (42, 52, 62):
            digest = runner.initial_states(self.data, seed)
            self.assertEqual(set(digest), set(ALL_ARMS))
            self.assertEqual(len(set(digest.values())), 4)

    def test_registered_active_and_production_optimizer_budgets(self):
        from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
        for arm in ALL_ARMS:
            model, meta = self.model(arm)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 115167)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), ACTIVE_PARAMETERS[arm])
            self.assertEqual(sum(p.numel() for p in correction_parameters(model).values() if p.requires_grad),
                ACTIVE_CORRECTION_PARAMETERS[arm])
            optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
            self.assertEqual(optimizer.defaults["weight_decay"], .01)
            optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertEqual(optimized, {id(p) for p in model.parameters() if p.requires_grad})
            if arm in ARMS:
                self.assertEqual(meta["registered_parameter_count"], 115167)
                self.assertEqual(meta["active_trainable_parameter_count"], ACTIVE_PARAMETERS[arm])
                self.assertIn("no capacity matching", meta["active_budget_limitations"])

    def test_cnn_only_true_bypass_of_gru_and_no_mlp(self):
        model, _ = self.model(CNN)
        self.activate(model)
        dt, qty, mask = self.inputs()
        with patch.object(model.multilag_detail, "forward", side_effect=AssertionError("Inactive GRU called")), \
                patch("models.TPPs.CountAwareTitanCNNGRU.causal_depthwise_residual", wraps=causal_depthwise_residual) as cnn:
            result = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
        self.assertEqual(cnn.call_count, 3)
        result["joint_loss"].mean().backward()
        self.assertTrue(all(p.grad is None for name, p in correction_parameters(model).items()
            if name.startswith("multilag_detail.")))
        self.assertTrue(all(model.state_dict()[name].shape == (3, 64) for name in KERNEL_KEYS))
        self.assertFalse(any("input_projections" in name for name in model.state_dict()))

    def test_gru_only_true_bypass_of_cnn_and_actual_recurrence(self):
        model, _ = self.model(GRU)
        self.activate(model)
        dt, qty, mask = self.inputs()
        with patch("models.TPPs.CountAwareTitanCNNGRU.causal_depthwise_residual", side_effect=AssertionError("Inactive CNN called")), \
                patch.object(model.multilag_detail, "forward", wraps=model.multilag_detail.forward) as gru:
            result = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
        self.assertEqual(gru.call_count, 1)
        result["joint_loss"].mean().backward()
        params = dict(model.named_parameters())
        self.assertTrue(all(params[name].grad is None for name in KERNEL_KEYS))
        hidden = torch.randn(2, 6, 64, generator=torch.Generator().manual_seed(9))
        changed = hidden.clone()
        changed[:, :-1] += 5.
        self.assertFalse(torch.equal(model.multilag_detail(hidden, mask)[:, -1],
            model.multilag_detail(changed, mask)[:, -1]))

    def test_inactive_weights_do_not_affect_outputs(self):
        dt, qty, mask = self.inputs()
        for arm in (NONE, CNN, GRU):
            model, _ = self.model(arm)
            self.activate(model)
            with torch.no_grad():
                expected = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
                for parameter in correction_parameters(model).values():
                    if not parameter.requires_grad:
                        parameter.fill_(10000.)
                actual = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), (arm, key))

    def test_synthetic_every_active_element_gradient_and_inactive_unchanged_each_seed(self):
        for seed in (42, 52, 62):
            result = runner.synthetic_check(seed=seed, include_causality=False)
            self.assertEqual(result["active_correction_gradient_parameter_count"], ACTIVE_CORRECTION_PARAMETERS)
            self.assertEqual(result["synthetic_optimizer_updates"], 16)
            self.assertEqual(result["seed"], seed)
            self.assertFalse(result["real_data_loaded"])
            self.assertFalse(result["held_out_test_evaluated"])

    def test_causal_withheld_prefix_padding_and_reset_explicit_tolerance(self):
        result = runner.check_synthetic_causality()
        self.assertEqual(result["future_prefix"], "invariant")
        self.assertEqual(result["withheld_valid"], "reset")
        self.assertEqual(result["different_shape_comparison_tolerance"], {"rtol": 1e-5, "atol": 1e-6})
        self.assertTrue(all(value < 1e-6 for value in result["gru_padding_max_absolute_residual"].values()))
        self.assertTrue(all(value < 1e-6 for value in result["gru_reset_max_absolute_residual"].values()))

    def test_cnn_withheld_reset_context_no_prior_segment_leak(self):
        model, _ = self.model(CNN)
        self.activate(model)
        hidden = torch.randn(1, 7, 64, generator=torch.Generator().manual_seed(19))
        valid = torch.ones(1, 7, dtype=torch.bool)
        write = valid.clone()
        write[:, 2] = False
        hidden[:, 2] = float("nan")
        _, indices, available = observed_context(valid, memory_write_mask=write)
        kernel = model.encoder.layers[0].attn.causal_q_kernel
        reset = causal_depthwise_residual(hidden, kernel, indices, available)
        _, indices, available = observed_context(valid[:, 3:])
        fresh = causal_depthwise_residual(hidden[:, 3:], kernel, indices, available)
        self.assertTrue(torch.equal(reset[:, 3:], fresh))
        self.assertTrue(bool((reset[:, 2] == 0).all()))

    def test_checkpoint_roundtrip_all_routes_and_component_relabelling_rejected(self):
        from models.TPPs import CountAwareFactory as factory
        for arm in ALL_ARMS:
            payload = self.payload(arm)
            factory.validate_checkpoint_route(payload, arm)
            restored, _ = self.model(arm, seed=99)
            restored.load_state_dict(payload["model_state_dict"], strict=True)
            for other in ALL_ARMS:
                if other == arm:
                    continue
                with self.assertRaises((ValueError, RuntimeError)):
                    factory.validate_checkpoint_route(payload, other)
                wrong, _ = self.model(other)
                with self.assertRaises((ValueError, RuntimeError)):
                    wrong.load_state_dict(payload["model_state_dict"], strict=False)

    def test_checkpoint_split_head_selector_role_budget_tamper_rejected(self):
        for arm in ARMS:
            for key, value in (("evaluation_scope", "test"), ("held_out_test_evaluated", True), ("variant", "other")):
                bad = self.payload(arm)
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_checkpoint(bad, arm)
            for key, value in (("checkpoint_monitor", "test_raw_quantity_rmse"), ("backbone", FULL)):
                bad = self.payload(arm)
                bad["resume_identity"][key] = value
                with self.assertRaises(ValueError):
                    validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            bad["encoder_config"]["active_correction_parameter_count"] += 1
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            bad["model_state_dict"]["cnn_gru_component_identity"] = bad["model_state_dict"]["cnn_gru_component_identity"].float()
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            bad["model_state_dict"]["multilag_detail.gru.weight_hh_l0"] = torch.zeros(1, 54)
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)

    def test_state_load_incomplete_nonfinite_rejected_before_mutation(self):
        for arm in ARMS:
            model, _ = self.model(arm)
            expected = deepcopy(model.state_dict())
            bad = deepcopy(expected)
            bad.pop("quantity_head.weight")
            with self.assertRaises(RuntimeError):
                model.load_state_dict(bad, strict=False)
            bad = deepcopy(expected)
            bad["quantity_head.weight"][0, 0] = float("nan")
            with self.assertRaises(RuntimeError):
                model.load_state_dict(bad)
            for name, value in expected.items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_training_args_reject_scope_budget_and_existing_checkpoint(self):
        training = {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "monitor": "validation_raw_quantity_rmse", "tie": "strict_earliest_finite_minimum", "warm_start": False}
        c = {"training": training}
        fake = SimpleNamespace(force_rerun=False, checkpoint_monitor="validation_raw_quantity_rmse")
        with patch.object(runner.original, "training_args", return_value=deepcopy(fake)), \
                patch.object(runner.base, "training_args", return_value=deepcopy(fake)):
            for arm in ALL_ARMS:
                args = runner.training_args(c, self.data, Path("fabricated_no_checkpoint"), arm)
                self.assertEqual((args.epochs, args.min_epochs, args.early_stopping_patience), (300, 40, 40))
            for key, value in (("evaluation_scope", "test"), ("test_evaluation", True), ("held_out_test_evaluated", True)):
                with self.assertRaises(ValueError):
                    runner.training_args({**c, key: value}, self.data, Path("fabricated_no_checkpoint"), CNN)
            for key, value in (("monitor", "test_raw_quantity_rmse"), ("warm_start", True), ("maximum_epochs", 301)):
                with self.assertRaises(ValueError):
                    runner.training_args({"training": {**training, key: value}}, self.data, Path("fabricated_no_checkpoint"), CNN)
            with patch.object(Path, "glob", return_value=iter([Path("existing.pt")])):
                with self.assertRaises(ValueError):
                    runner.training_args(c, self.data, Path("fabricated_no_checkpoint"), CNN)

    def test_replay_both_component_routes_only_validation_full_population(self):
        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
        dt, qty, mask = self.inputs()
        data = deepcopy(self.data)
        data["loader"].update(batch_size=2, lookback_weeks=168)
        data.update(quantity_boundaries_all_train_rows=[1, 2, 2, 3], history_boundaries=[1, 3],
            inherited_data_identity={"populations": {"validation": {"target_count": 2}}})
        for arm in ARMS:
            payload = self.payload(arm)
            payload["model_state_sha256"] = canonical_state_dict_sha256(payload["model_state_dict"])
            loader = [(None, dt, mask, None, qty)]
            with patch("paper.scripts.run_taxi_quantity_interface_ablation.make_loader", return_value=loader) as make:
                result = runner._replay_validation(payload, Path("fabricated_not_opened.pt"), data, None,
                    lambda: None, device="cpu")
            self.assertEqual(make.call_args.kwargs["target_split"], "validation")
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["evaluation_scope"], "validation_only")
            self.assertFalse(result["held_out_test_evaluated"])


if __name__ == "__main__":
    unittest.main()
