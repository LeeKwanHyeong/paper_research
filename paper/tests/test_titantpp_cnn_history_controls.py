"""Fabricated CPU Q1 contracts; no experimental data or checkpoints loaded."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from models.TPPs.CountAwareTitanCNNGRU import (
    CountAwareTitanCNNGRU, causal_depthwise_residual, observed_context,
)
from models.TPPs.CountAwareTitanCNNHistoryControls import (
    ARMS, BASELINE, CURRENT_ONLY, DISABLED, CORRECTION_PARAMETERS,
    CurrentStateGRUCorrection, correction_parameters, current_pointwise_residual,
    metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_history_controls as runner
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


class CNNHistoryControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = runner.fabricated_data()
        runner.install_hooks()

    def model(self, arm=CURRENT_ONLY, seed=42):
        torch.manual_seed(seed)
        model, meta = runner.build_model(self.data, arm)
        return model.eval(), meta

    def inputs(self):
        dt = torch.tensor([[0., 1., 3., 5., 2., 7.], [0., 4., 2., 6., 1., 8.]])
        qty = torch.tensor([[0., 2., 4., 8., 3., 1.], [1., 0., 5., 2., 7., 4.]])
        return dt, qty, torch.ones_like(dt, dtype=torch.bool)

    def activate(self, model):
        # Exercise learned paths, which are zero-initialized in every real arm.
        generator = torch.Generator().manual_seed(77)
        with torch.no_grad():
            for parameter in correction_parameters(model).values():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .05)

    def payload(self, arm):
        model, meta = self.model(arm)
        interface = {"time_head": model.time_head_contract()}
        return {"backbone": arm, "encoder_config": meta,
            "model_state_dict": model.state_dict(), "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False, "variant": runner.VARIANT,
            "interface_meta": interface,
            "resume_identity": {"backbone": arm, "interface_meta": interface,
                "checkpoint_monitor": "validation_raw_quantity_rmse",
                "arguments": {"model_role": role_for_arm(arm)}}}

    def test_full_route_exact_original_model_outputs_rng_and_updates(self):
        full, _ = self.model(BASELINE)
        torch.manual_seed(42)
        original, _ = runner.cnn.build_model(self.data, BASELINE)
        self.assertIs(type(full), CountAwareTitanCNNGRU)
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, full.state_dict()[name]), name)
        dt, qty, mask = self.inputs()
        for train in (False, True):
            full.train(train)
            original.train(train)
            torch.manual_seed(91)
            expected = target_outputs(original, dt, mask, qty, lambda_log_qty=1.)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(91)
            actual = target_outputs(full, dt, mask, qty, lambda_log_qty=1.)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)
        expected["joint_loss"].mean().backward()
        actual["joint_loss"].mean().backward()
        for (_, a), (_, b) in zip(original.named_parameters(), full.named_parameters()):
            self.assertIsNotNone(a.grad)
            self.assertTrue(torch.equal(a.grad, b.grad))
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.001) for model in (original, full)]
        for optimizer in optimizers:
            optimizer.step()
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, full.state_dict()[name]), name)

    def test_common_initial_parameters_and_construction_rng_preserved(self):
        for seed in (42, 52, 62):
            digests = runner.initial_states(self.data, seed)
            self.assertEqual(set(digests), set(runner.ALL_ARMS))
            self.assertEqual(len(set(digests.values())), 3)

    def test_registered_and_active_budgets_and_semantic_limits(self):
        for arm in runner.ALL_ARMS:
            with self.subTest(arm=arm):
                model, meta = self.model(arm)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 115167)
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                    89859 if arm == DISABLED else 115167)
                parameters = correction_parameters(model)
                self.assertEqual(sum(p.numel() for p in parameters.values()), CORRECTION_PARAMETERS)
                if arm in ARMS:
                    self.assertEqual(meta["active_correction_parameter_count"],
                        0 if arm == DISABLED else CORRECTION_PARAMETERS)
                    self.assertTrue(meta["current_encoder_contains_observed_prefix"])
                    self.assertFalse(meta["explicit_previous_correction_state"])
                    if arm == CURRENT_ONLY:
                        self.assertIn("depth", meta["active_budget_limitations"])
        model, _ = self.model()
        self.assertIsInstance(model.multilag_detail, CurrentStateGRUCorrection)

    def test_zero_initial_control_paths_preserve_full_initial_outputs_and_dropout_rng(self):
        dt, qty, mask = self.inputs()
        reference = None
        reference_rng = None
        for arm in runner.ALL_ARMS:
            model, _ = self.model(arm)
            model.train()
            torch.manual_seed(314)
            outputs = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            rng = torch.get_rng_state().clone()
            if reference is None:
                reference, reference_rng = outputs, rng
            for key in reference:
                self.assertTrue(torch.equal(reference[key], outputs[key]), (arm, key))
            self.assertTrue(torch.equal(reference_rng, rng))

    def test_all_active_correction_elements_update_with_finite_gradients(self):
        result = runner.synthetic_check()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["active_correction_gradient_parameter_count"], {
            BASELINE: 25308, DISABLED: 0, CURRENT_ONLY: 25308})
        self.assertEqual(result["synthetic_optimizer_updates"], 12)
        self.assertFalse(result["real_data_loaded"])
        self.assertFalse(result["held_out_test_evaluated"])
        self.assertEqual(result["causality"]["current_only_explicit_prior_hidden_input"], "absent")

    def test_current_gru_invariant_to_prior_hidden_values_given_same_current_hidden(self):
        current, _ = self.model()
        full, _ = self.model(BASELINE)
        self.activate(current)
        self.activate(full)
        generator = torch.Generator().manual_seed(2)
        hidden = torch.randn(2, 6, 64, generator=generator)
        altered = hidden.clone()
        altered[:, :-1] = torch.randn(2, 5, 64, generator=generator) * 5.
        mask = torch.ones(2, 6, dtype=torch.bool)
        expected = current.multilag_detail(hidden, mask)
        actual = current.multilag_detail(altered, mask)
        self.assertTrue(torch.equal(expected[:, -1], actual[:, -1]))
        self.assertFalse(torch.equal(full.multilag_detail(hidden, mask)[:, -1],
            full.multilag_detail(altered, mask)[:, -1]))
        # Count gating is intentionally retained: first observed row has zero
        # correction even though the hidden value itself can contain history.
        self.assertTrue(bool((expected[:, 0] == 0).all()))

    def test_current_cnn_invariant_to_prior_qkv_values_and_all_bases_distinct(self):
        generator = torch.Generator().manual_seed(3)
        hidden = torch.randn(2, 6, 64, generator=generator)
        changed = hidden.clone()
        changed[:, :-1] += 7.
        kernel = torch.randn(3, 64, generator=generator)
        mask = torch.ones(2, 6, dtype=torch.bool)
        _, sources, available = observed_context(mask)
        current = current_pointwise_residual(hidden, kernel, available)
        actual = current_pointwise_residual(changed, kernel, available)
        self.assertTrue(torch.equal(current[:, -1], actual[:, -1]))
        self.assertFalse(torch.equal(causal_depthwise_residual(hidden, kernel, sources, available)[:, -1],
            causal_depthwise_residual(changed, kernel, sources, available)[:, -1]))
        self.assertFalse(torch.equal(hidden, torch.tanh(hidden)))
        self.assertFalse(torch.equal(torch.tanh(hidden), torch.nn.functional.gelu(hidden)))

    def test_disabled_bypasses_modules_and_stays_invariant_to_correction_weights(self):
        model, _ = self.model(DISABLED)
        dt, qty, mask = self.inputs()
        expected = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
        self.activate(model)
        with patch.object(model.multilag_detail, "forward", side_effect=AssertionError("Disabled GRU called")):
            actual = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        actual["joint_loss"].mean().backward()
        self.assertTrue(all(p.grad is None for p in correction_parameters(model).values()))

    def test_current_gru_padding_skip_withheld_reset_and_nonfinite_safety(self):
        model, _ = self.model()
        self.activate(model)
        hidden = torch.randn(2, 7, 64, generator=torch.Generator().manual_seed(4))
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[:, 2] = False
        hidden[:, 2] = float("nan")
        padded = model.multilag_detail(hidden, mask)
        keep = [0, 1, 3, 4, 5, 6]
        compact = model.multilag_detail(hidden[:, keep], mask[:, keep])
        self.assertTrue(torch.equal(padded[:, keep], compact))
        self.assertTrue(bool((padded[:, 2] == 0).all()))
        valid = torch.ones_like(mask)
        write = valid.clone()
        write[:, 2] = False
        reset = model.multilag_detail(hidden, valid, memory_write_mask=write)
        self.assertTrue(bool((reset[:, 2:4] == 0).all()))
        self.assertTrue(bool(torch.isfinite(reset).all()))
        with self.assertRaises(ValueError):
            model.multilag_detail(hidden, valid)
        empty = model.multilag_detail(hidden[:, :0], valid[:, :0])
        self.assertEqual(tuple(empty.shape), (2, 0, 64))

    def test_full_model_future_causality_and_withheld_observation_invariance(self):
        dt, qty, mask = self.inputs()
        for arm in ARMS:
            model, _ = self.model(arm)
            self.activate(model)
            with torch.no_grad():
                expected = model.encode_task_states(dt, qty, mask)[0]
                future_dt, future_qty = dt.clone(), qty.clone()
                future_dt[:, 4:], future_qty[:, 4:] = 900., 9999.
                actual = model.encode_task_states(future_dt, future_qty, mask)[0]
                self.assertTrue(torch.equal(expected[:, :4], actual[:, :4]), arm)
                write = mask.clone()
                write[:, -1] = False
                expected = model.encode_task_states(dt, qty, mask, memory_write_mask=write)[0]
                future_dt, future_qty = dt.clone(), qty.clone()
                future_dt[:, -1], future_qty[:, -1] = float("nan"), float("inf")
                actual = model.encode_task_states(future_dt, future_qty, mask, memory_write_mask=write)[0]
                self.assertTrue(torch.equal(expected, actual), arm)
                self.assertTrue(bool((actual[:, -1] == 0).all()))

    def test_current_hidden_still_contains_encoder_prefix_information(self):
        model, _ = self.model()
        dt, qty, mask = self.inputs()
        with torch.no_grad():
            expected = model._encode_base(dt, qty, mask)
            changed_qty = qty.clone()
            changed_qty[:, :-1] += 5.
            actual = model._encode_base(dt, changed_qty, mask)
        self.assertFalse(torch.equal(expected[:, -1], actual[:, -1]))

    def test_checkpoint_roundtrip_and_cross_arm_relabelling_rejected(self):
        for arm in ARMS:
            payload = self.payload(arm)
            self.assertTrue(validate_checkpoint(payload, arm))
            from models.TPPs import CountAwareFactory as factory
            factory.validate_checkpoint_route(payload, arm)
            same, _ = self.model(arm, seed=99)
            same.load_state_dict(payload["model_state_dict"], strict=True)
            for other in (BASELINE, *ARMS, "titantpp_current_only_param_matched"):
                if other == arm:
                    continue
                with self.assertRaises(ValueError):
                    validate_checkpoint(payload, other)
            wrong, _ = self.model(next(other for other in ARMS if other != arm))
            with self.assertRaises(ValueError):
                wrong.load_state_dict(payload["model_state_dict"], strict=False)

    def test_checkpoint_split_head_selector_semantics_and_shape_tamper_rejected(self):
        for arm in ARMS:
            for field, value in (("evaluation_scope", "test"), ("held_out_test_evaluated", True),
                    ("variant", "other")):
                bad = self.payload(arm)
                bad[field] = value
                with self.assertRaises(ValueError):
                    validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            bad["encoder_config"]["active_correction_parameter_count"] += 1
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            bad["resume_identity"]["checkpoint_monitor"] = "test_raw_quantity_rmse"
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            key = "multilag_detail.gru.weight_hh_l0"
            bad["model_state_dict"][key] = bad["model_state_dict"][key][:1]
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            bad["model_state_dict"]["cnn_gru_identity"] = torch.zeros(32, dtype=torch.uint8)
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)
            bad = self.payload(arm)
            bad["model_state_dict"]["cnn_history_control_identity"] = bad["model_state_dict"]["cnn_history_control_identity"].float()
            with self.assertRaises(ValueError):
                validate_checkpoint(bad, arm)

    def test_state_load_rejects_incomplete_and_nonfinite_before_mutation(self):
        model, _ = self.model()
        original = deepcopy(model.state_dict())
        incomplete = deepcopy(original)
        incomplete.pop("time_head.norm.weight") if "time_head.norm.weight" in incomplete else incomplete.pop(next(
            key for key in incomplete if not key.endswith("identity")))
        with self.assertRaises(RuntimeError):
            model.load_state_dict(incomplete, strict=False)
        corrupted = deepcopy(original)
        key = next(name for name, value in corrupted.items() if value.dtype.is_floating_point)
        corrupted[key].view(-1)[0] = float("nan")
        with self.assertRaises(RuntimeError):
            model.load_state_dict(corrupted)
        for name, value in original.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_mask_shape_and_nonfinite_observed_inputs_rejected(self):
        model, _ = self.model()
        dt, qty, mask = self.inputs()
        with self.assertRaises(ValueError):
            model._encode_base(dt, qty, mask.float())
        with self.assertRaises(ValueError):
            model._encode_base(dt, qty[:, :-1], mask)
        with self.assertRaises(ValueError):
            model._encode_base(dt, qty, mask, memory_write_mask=mask[:, :-1])
        dt[0, 1] = float("nan")
        with self.assertRaises(ValueError):
            model._encode_base(dt, qty, mask)
        with self.assertRaises(ValueError):
            metadata(32, arm=CURRENT_ONLY)

    def test_replay_adapter_uses_control_route_and_full_validation_only(self):
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
            with patch("paper.scripts.run_taxi_quantity_interface_ablation.make_loader", return_value=loader) as make_loader:
                result = runner._replay_validation(payload, Path("fabricated_not_opened.pt"), data, None,
                    lambda: None, device="cpu")
            self.assertEqual(make_loader.call_args.kwargs["target_split"], "validation")
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["tail"]["count"], 1)
            self.assertEqual(result["evaluation_scope"], "validation_only")
            self.assertFalse(result["held_out_test_evaluated"])
            for key in ("qty_rmse", "qty_mae", "time_nll"):
                self.assertTrue(torch.isfinite(torch.tensor(result[key])))
                self.assertTrue(torch.isfinite(torch.tensor(result["tail"][key])))


if __name__ == "__main__":
    unittest.main()
