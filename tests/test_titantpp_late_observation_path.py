"""Fabricated-only tests of matched late paths; no data, GPU or remote work."""
from copy import deepcopy
import io
import unittest

import torch
from torch.nn import functional as F

from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth
from models.TPPs.CountAwareTitanLateObservationPath import (
    ARMS, MODE_BY_ARM, CountAwareTitanLateObservationPath, LatePath,
    identity, metadata,
)


KWARGS = {
    "quantity_variant": "count_only_log_regression",
    "time_head_mode": "heteroscedastic_lognormal_duration",
    "train_log_std": .8, "time_scale": 1.,
    "time_initial_location": .2, "time_initial_scale": .8,
    "time_observation_contract": {
        "mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None,
    },
}


def make_model(mode="observation", seed=42):
    torch.manual_seed(seed)
    return CountAwareTitanLateObservationPath(
        64, 1.2, 64, late_path_mode=mode, **deepcopy(KWARGS)).eval()


def make_parent(seed=42):
    torch.manual_seed(seed)
    return CountAwareTitanHistoryWidth(
        64, 1.2, 64, history_mlp_width=16, **deepcopy(KWARGS)).eval()


def inputs():
    return (torch.tensor([[1., 2., 5., 3., 4., 6.], [2., 1., 3., 8., 2., 4.]]),
            torch.tensor([[2., 4., 3., 8., 5., 7.], [1., 6., 2., 5., 4., 3.]]),
            torch.ones(2, 6, dtype=torch.bool))


def activate_path(model):
    """Deterministic nonzero branch without consuming the global generator."""
    with torch.no_grad():
        for index, parameter in enumerate(model.late_path.parameters()):
            value = torch.arange(parameter.numel(), dtype=parameter.dtype)
            parameter.copy_(((value % 17 - 8) / (73. + index)).reshape_as(parameter))


class LateObservationPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_two_arm_contract_and_parameter_counts(self):
        self.assertEqual(len(ARMS), 2)
        self.assertEqual(set(MODE_BY_ARM.values()), {"observation", "hidden"})
        totals = []
        for arm, mode in MODE_BY_ARM.items():
            with self.subTest(mode=mode):
                model = make_model(mode)
                meta = metadata(mode=mode)
                self.assertEqual(meta["candidate_name"], arm)
                self.assertEqual(meta["history_mlp_width"], 16)
                self.assertEqual(meta["late_path_rank"], 8)
                self.assertEqual(meta["late_path_parameter_count"], 1024)
                self.assertEqual(sum(p.numel() for p in model.late_path.parameters()), 1024)
                self.assertIsNone(model.late_path.input_projection.bias)
                self.assertIsNone(model.late_path.output_projection.bias)
                self.assertEqual(tuple(model.late_path.input_projection.weight.shape), (8, 64))
                self.assertEqual(tuple(model.late_path.output_projection.weight.shape), (64, 8))
                self.assertTrue(torch.equal(model.late_path_identity, identity(mode=mode)))
                self.assertEqual(meta["late_path_insertion"], "after_encoder2_before_existing_static_hard_lmm")
                totals.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(totals[0], totals[1])
        self.assertEqual(totals[0] - sum(p.numel() for p in make_parent().parameters()), 1024)

    def test_every_common_tensor_and_global_rng_match_parent_for_all_seeds(self):
        for seed in (42, 52, 62):
            parent = make_parent(seed)
            parent_rng = torch.get_rng_state().clone()
            for mode in MODE_BY_ARM.values():
                with self.subTest(seed=seed, mode=mode):
                    model = make_model(mode, seed)
                    self.assertTrue(torch.equal(parent_rng, torch.get_rng_state()))
                    common = parent.state_dict()
                    candidate = model.state_dict()
                    self.assertEqual(set(candidate) - set(common), {
                        "late_path_identity", "late_path.input_projection.weight",
                        "late_path.output_projection.weight"})
                    for name, value in common.items():
                        self.assertTrue(torch.equal(value, candidate[name]), name)
                    self.assertEqual(model.time_head_contract(), parent.time_head_contract())

    def test_zero_output_is_bitwise_parent_for_encoder_and_both_heads(self):
        dt, qty, valid = inputs()
        with torch.no_grad():
            for seed in (42, 52, 62):
                parent = make_parent(seed)
                expected = parent.encode_task_states(dt, qty, valid)[0]
                expected_quantity = parent.predict_quantity(expected)
                expected_time = parent.log_observation_dt(expected.flatten(0, 1), torch.full_like(dt, 2.).flatten())
                for mode in MODE_BY_ARM.values():
                    with self.subTest(seed=seed, mode=mode):
                        model = make_model(mode, seed)
                        self.assertEqual(torch.count_nonzero(model.late_path.output_projection.weight).item(), 0)
                        actual = model.encode_task_states(dt, qty, valid)[0]
                        self.assertTrue(torch.equal(actual, expected))
                        for a, b in zip(model.predict_quantity(actual), expected_quantity):
                            self.assertTrue(torch.equal(a, b))
                        self.assertTrue(torch.equal(model.log_observation_dt(actual.flatten(0, 1), torch.full_like(dt, 2.).flatten()), expected_time))

    def test_nonzero_formula_is_after_encoder2_and_before_static_memory(self):
        dt, qty, valid = inputs()
        valid[0, 1] = False
        observed = valid.clone()
        observed[1, 3] = False
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate_path(model)
                common_hidden = CountAwareTitanHistoryWidth._encode_base(
                    model, dt, qty, valid, memory_write_mask=observed)
                safe_dt = torch.where(observed, dt, 0.)
                safe_qty = torch.where(observed, qty, 0.)
                e0 = model.encoder.input_proj(model.continuous_features(safe_dt, safe_qty, observed))
                source = e0 if mode == "observation" else common_hidden
                safe_source = torch.where(observed.unsqueeze(-1), source, 0.)
                correction = F.linear(F.gelu(F.linear(safe_source,
                    model.late_path.input_projection.weight), approximate="none"),
                    model.late_path.output_projection.weight)
                correction = torch.where(observed.unsqueeze(-1), correction, 0.)
                expected = common_hidden + correction
                seen_source, seen_lmm = [], []
                source_hook = model.late_path.register_forward_pre_hook(
                    lambda module, args: seen_source.append(args[0].detach().clone()))
                memory_hook = model.lmm.register_forward_pre_hook(
                    lambda module, args: seen_lmm.append(args[0].detach().clone()))
                try:
                    encoded = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)[0]
                finally:
                    source_hook.remove()
                    memory_hook.remove()
                self.assertTrue(torch.equal(seen_source[0], source))
                self.assertTrue(torch.equal(seen_lmm[0], expected))
                expected_encoded = torch.where(observed.unsqueeze(-1), model.lmm(expected), 0.)
                self.assertTrue(torch.equal(encoded, expected_encoded))

    def test_observation_source_excludes_position_and_context_hidden_does_not(self):
        dt = torch.full((1, 5), 2.)
        qty = torch.full((1, 5), 4.)
        valid = torch.ones_like(dt, dtype=torch.bool)
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                sources = []
                hook = model.late_path.register_forward_pre_hook(
                    lambda module, args: sources.append(args[0].clone()))
                try:
                    model._encode_base(dt, qty, valid)
                    changed_dt, changed_qty = dt.clone(), qty.clone()
                    changed_dt[:, :3] = torch.tensor([[11., 9., 7.]])
                    changed_qty[:, :3] = torch.tensor([[40., 1., 16.]])
                    model._encode_base(changed_dt, changed_qty, valid)
                finally:
                    hook.remove()
                if mode == "observation":
                    self.assertTrue(torch.equal(sources[0][:, 0], sources[0][:, -1]))
                    self.assertTrue(torch.equal(sources[0][:, -1], sources[1][:, -1]))
                else:
                    self.assertFalse(torch.allclose(sources[0][:, 0], sources[0][:, -1]))
                    self.assertFalse(torch.allclose(sources[0][:, -1], sources[1][:, -1]))

    def test_nonzero_path_prefix_is_causal_and_unchanged_by_appended_future(self):
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate_path(model)
                original = model.encode_task_states(dt, qty, valid)[0]
                changed_dt, changed_qty = dt.clone(), qty.clone()
                changed_dt[:, 4:] += 100.
                changed_qty[:, 4:] += 1000.
                altered = model.encode_task_states(changed_dt, changed_qty, valid)[0]
                torch.testing.assert_close(original[:, :4], altered[:, :4], rtol=1e-6, atol=1e-6)
                prefix = model.encode_task_states(dt[:, :4], qty[:, :4], valid[:, :4])[0]
                torch.testing.assert_close(original[:, :4], prefix, rtol=1e-6, atol=1e-6)

    def test_padded_and_withheld_nan_inf_values_are_ignored_and_outputs_zero(self):
        dt, qty, valid = inputs()
        valid[0, 1] = False
        observed = valid.clone()
        observed[1, 3] = False
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate_path(model)
                expected = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)[0]
                changed_dt, changed_qty = dt.clone(), qty.clone()
                changed_dt[~observed] = float("nan")
                changed_qty[~observed] = float("inf")
                actual = model.encode_task_states(changed_dt, changed_qty, valid, memory_write_mask=observed)[0]
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue((actual[~observed] == 0).all())

    def test_observed_nonfinite_or_negative_values_are_rejected(self):
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            model = make_model(mode)
            for column in (0, 1):
                for bad in (float("nan"), float("inf"), -1.):
                    with self.subTest(mode=mode, column=column, bad=bad):
                        values = [dt.clone(), qty.clone()]
                        values[column][0, 2] = bad
                        with self.assertRaises(ValueError):
                            model.encode_task_states(*values, valid)

    def test_all_absent_inputs_remain_finite_zero_after_static_memory(self):
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate_path(model)
                missing = torch.full((2, 4), float("nan"))
                output = model.encode_task_states(missing, missing, torch.zeros(2, 4, dtype=torch.bool))[0]
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_eval_preserves_every_parameter_buffer_context_and_inputs(self):
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate_path(model)
                state = deepcopy(model.state_dict())
                before_inputs = (dt.clone(), qty.clone(), valid.clone())
                contexts = [layer.attn._ctx_mem for layer in model.encoder.layers]
                first = model.encode_task_states(dt, qty, valid)[0]
                second = model.encode_task_states(dt, qty, valid)[0]
                self.assertTrue(torch.equal(first, second))
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                for old, layer in zip(contexts, model.encoder.layers):
                    self.assertIs(layer.attn._ctx_mem, old)
                for old, current in zip(before_inputs, (dt, qty, valid)):
                    self.assertTrue(torch.equal(old, current))

    def test_quantity_gradient_reaches_both_new_projections_after_first_update(self):
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode):
                model = make_model(mode)
                # The inherited quantity head starts with exactly zero weights;
                # give this synthetic gradient fixture a learned-like readout.
                with torch.no_grad():
                    model.quantity_head.weight.copy_(torch.linspace(-.04, .04, 64).reshape(1, 64))
                common = {n: p.detach().clone() for n, p in model.named_parameters()
                          if not n.startswith("late_path.")}
                optimizer = torch.optim.SGD(model.late_path.parameters(), lr=.03)
                for step in range(2):
                    model.zero_grad(set_to_none=True)
                    encoded = model.encode_task_states(dt, qty, valid)[0]
                    prediction = model.predict_quantity(encoded)[1]
                    (prediction - qty).square().mean().backward()
                    output_grad = model.late_path.output_projection.weight.grad
                    self.assertTrue(torch.isfinite(output_grad).all())
                    self.assertGreater(output_grad.abs().sum().item(), 0.)
                    input_grad = model.late_path.input_projection.weight.grad
                    self.assertTrue(torch.isfinite(input_grad).all())
                    if step == 0:
                        self.assertEqual(input_grad.abs().sum().item(), 0.)
                    else:
                        self.assertGreater(input_grad.abs().sum().item(), 0.)
                    optimizer.step()
                for name, value in common.items():
                    self.assertTrue(torch.equal(value, dict(model.named_parameters())[name]), name)

    def test_state_roundtrip_preserves_nonzero_predictions_for_both_arms(self):
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate_path(model)
                state = deepcopy(model.state_dict())
                buffer = io.BytesIO()
                torch.save(state, buffer)
                buffer.seek(0)
                loaded = torch.load(buffer, map_location="cpu", weights_only=True)
                restored = make_model(mode, seed=62)
                restored.load_state_dict(loaded)
                expected = model.encode_task_states(dt, qty, valid)[0]
                actual = restored.encode_task_states(dt, qty, valid)[0]
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(model.predict_quantity(expected)[1], restored.predict_quantity(actual)[1]))
                self.assertTrue(torch.equal(model.log_observation_dt(expected.flatten(0, 1), dt.flatten()),
                                            restored.log_observation_dt(actual.flatten(0, 1), dt.flatten())))

    def test_wrong_arm_missing_identity_and_incomplete_or_malformed_state_rejected(self):
        for mode in MODE_BY_ARM.values():
            model = make_model(mode)
            original = deepcopy(model.state_dict())
            wrong = "hidden" if mode == "observation" else "observation"
            with self.subTest(mode=mode, case="wrong_arm"), self.assertRaises(ValueError):
                make_model(wrong).load_state_dict(original, strict=False)
            mutations = {
                "missing_identity": lambda s: s.pop("late_path_identity"),
                "identity_dtype": lambda s: s.update(late_path_identity=s["late_path_identity"].long()),
                "identity_shape": lambda s: s.update(late_path_identity=s["late_path_identity"][:-1]),
                "missing_new_parameter": lambda s: s.pop("late_path.input_projection.weight"),
                "missing_common_parameter": lambda s: s.pop("encoder.input_proj.weight"),
                "new_parameter_shape": lambda s: s.update({"late_path.output_projection.weight": torch.zeros(64, 7)}),
                "common_parameter_shape": lambda s: s.update({"encoder.input_proj.weight": torch.zeros(63, 2)}),
                "extra_parameter": lambda s: s.update(unexpected_tensor=torch.zeros(1)),
            }
            for name, mutate in mutations.items():
                with self.subTest(mode=mode, case=name):
                    state = deepcopy(original)
                    mutate(state)
                    before = deepcopy(model.state_dict())
                    with self.assertRaises((ValueError, RuntimeError)):
                        model.load_state_dict(state, strict=False)
                    for key, value in before.items():
                        self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

    def test_constructor_width_mode_and_branch_shape_mask_contracts(self):
        for hidden in (32, 65, True, 64.):
            with self.subTest(hidden=hidden), self.assertRaises(ValueError):
                CountAwareTitanLateObservationPath(hidden, 1.2, 64, **deepcopy(KWARGS))
        for width in (4, 8, 12, True, 16.):
            with self.subTest(width=width), self.assertRaises(ValueError):
                CountAwareTitanLateObservationPath(64, 1.2, 64, history_mlp_width=width, **deepcopy(KWARGS))
        for mode in ("other", "", None, 0):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                CountAwareTitanLateObservationPath(64, 1.2, 64, late_path_mode=mode, **deepcopy(KWARGS))
        branch = LatePath()
        for shape in ((2, 4), (2, 4, 63), (2, 4, 64, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                branch(torch.zeros(shape), torch.ones(2, 4, dtype=torch.bool))
        for invalid in (torch.ones(2, 4), torch.ones(2, 3, dtype=torch.bool)):
            with self.subTest(mask=tuple(invalid.shape)), self.assertRaises(ValueError):
                branch(torch.zeros(2, 4, 64), invalid)

    def test_target_output_does_not_supply_future_quantity_or_gap_to_new_path(self):
        from paper.scripts.count_aware_tpp_backbone.core import target_outputs
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate_path(model)
                original = target_outputs(model, dt, valid, qty, lambda_log_qty=1.)
                changed_dt, changed_qty = dt.clone(), qty.clone()
                changed_dt[:, -1] += 20.
                changed_qty[:, -1] += 100.
                altered = target_outputs(model, changed_dt, valid, changed_qty, lambda_log_qty=1.)
                self.assertTrue(torch.equal(original["pred_qty"], altered["pred_qty"]))
                self.assertFalse(torch.equal(original["true_qty"], altered["true_qty"]))
                self.assertFalse(torch.equal(original["time_loss"], altered["time_loss"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
