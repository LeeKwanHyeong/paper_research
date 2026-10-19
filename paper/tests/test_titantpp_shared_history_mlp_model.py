"""Local fabricated-input tests; no dataset, remote server or training job."""
from copy import deepcopy
import io
import unittest

import torch
from torch.nn import functional as F

from models.TPPs.CountAwareTitanCoreAblation import CountAwareTitanCoreAblation
from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth
from models.TPPs.CountAwareTitanSharedHistoryMLP import (
    ARMS, MODE_BY_ARM, CORRECTION_PARAMETER_COUNT,
    CountAwareTitanSharedHistoryMLP, SharedHistoryCorrection,
    identity, metadata, role_for_arm, validate_checkpoint,
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


def make_model(mode="shared", seed=42):
    torch.manual_seed(seed)
    return CountAwareTitanSharedHistoryMLP(
        64, 1.2, 180, history_sharing_mode=mode, **deepcopy(KWARGS)).eval()


def make_parent(seed=42):
    torch.manual_seed(seed)
    return CountAwareTitanHistoryWidth(
        64, 1.2, 180, history_mlp_width=16, **deepcopy(KWARGS)).eval()


def inputs(length=12):
    index = torch.arange(2 * length, dtype=torch.float32).reshape(2, length)
    return index.remainder(7) + 1, index.remainder(11) + 2, torch.ones(2, length, dtype=torch.bool)


def activate(correction):
    with torch.no_grad():
        for branch, projection in enumerate(correction.output_projections):
            value = torch.arange(projection.weight.numel(), dtype=projection.weight.dtype)
            projection.weight.copy_(((value.remainder(19) - 9) / (97 + branch)).reshape_as(projection.weight))


def literal_loop(correction, hidden, valid, write=None):
    """Independent event loop: padding skips, withheld resets, eight thresholds."""
    result = torch.zeros_like(hidden)
    for batch in range(hidden.shape[0]):
        segment = []
        for event in range(hidden.shape[1]):
            if not bool(valid[batch, event]):
                continue
            if write is not None and not bool(write[batch, event]):
                segment = []
                continue
            for branch, lag in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
                if len(segment) < lag:
                    continue
                pair = torch.cat((hidden[batch, event], hidden[batch, segment[-1]]))
                projection = (correction.input_projection if correction.mode == "shared"
                              else correction.input_projections[branch])
                result[batch, event] += correction.output_projections[branch](
                    F.gelu(projection(pair), approximate="none")) / 8
            segment.append(event)
    return result


def payload(model):
    mode = model.history_sharing_mode
    arm = next(arm for arm, selected_mode in MODE_BY_ARM.items() if selected_mode == mode)
    meta = metadata(mode=mode)
    meta["time_head"] = model.time_head_contract()
    interface = {"time_head": deepcopy(meta["time_head"])}
    return {
        "backbone": arm, "encoder_config": meta,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "variant": "count_only_log_regression", "interface_meta": interface,
        "resume_identity": {
            "backbone": arm, "interface_meta": deepcopy(interface),
            "checkpoint_monitor": "validation_raw_quantity_rmse",
            "arguments": {"model_role": role_for_arm(arm)},
        },
        "model_state_dict": deepcopy(model.state_dict()),
    }


class SharedHistoryModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_two_arms_metadata_and_equal_active_parameter_counts(self):
        self.assertEqual(ARMS, ("titantpp_history_mlp_shared15_v1", "titantpp_history_mlp_prefix_capacity_v1"))
        totals = []
        for arm, mode in MODE_BY_ARM.items():
            with self.subTest(mode=mode):
                model = make_model(mode)
                meta = metadata(mode=mode)
                correction = model.multilag_detail
                self.assertEqual(meta["candidate_name"], arm)
                self.assertEqual(meta["model_role"], role_for_arm(arm))
                self.assertEqual(meta["history_relation_parameter_count"], 9600)
                self.assertEqual(sum(p.numel() for p in correction.parameters()), CORRECTION_PARAMETER_COUNT)
                self.assertTrue(all(p.requires_grad for p in correction.parameters()))
                self.assertEqual(len(list(correction.parameters())), 9 if mode == "shared" else 16)
                self.assertTrue(all(not name.endswith("bias") for name, _ in correction.named_parameters()))
                self.assertNotIn("history_width_identity", model.state_dict())
                self.assertFalse(hasattr(model, "late_path"))
                self.assertEqual(meta["multilag_detail_availability_lags"], [1, 2, 4, 8, 16, 32, 64, 128])
                self.assertEqual(meta["multilag_detail_lags"], [1] * 8)
                totals.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(totals[0], totals[1])
        self.assertEqual(sum(p.numel() for p in make_parent().multilag_detail.parameters()), 24576)
        self.assertEqual(sum(p.numel() for p in make_parent().parameters()) - totals[0], 24576 - 9600)

    def test_shared_input_is_one_parameter_used_by_all_eight_branches(self):
        correction = SharedHistoryCorrection()
        self.assertIsInstance(correction.input_projection.weight, torch.nn.Parameter)
        self.assertFalse(hasattr(correction, "input_projections"))
        self.assertEqual(tuple(correction.input_projection.weight.shape), (15, 128))
        self.assertEqual(sum(name == "input_projection.weight" for name, _ in correction.named_parameters()), 1)

    def test_active_parameter_count_matches_at_every_observed_prefix_including_raf(self):
        observed_lengths = (1, 2, 3, 5, 9, 17, 33, 65, 129)
        expected_counts = (0, 2880, 3840, 4800, 5760, 6720, 7680, 8640, 9600)
        for prefix, (length, expected) in enumerate(zip(observed_lengths, expected_counts)):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(91 + prefix)
                hidden, target = torch.randn(2, length, 64), torch.randn(2, length, 64)
            valid = torch.ones(2, length, dtype=torch.bool)
            for mode in MODE_BY_ARM.values():
                with self.subTest(active_branches=prefix, mode=mode):
                    correction = SharedHistoryCorrection(mode=mode)
                    optimizer = torch.optim.SGD(correction.parameters(), lr=.2)
                    F.mse_loss(correction(hidden, valid), target).backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    F.mse_loss(correction(hidden, valid), target).backward()
                    active = sum(parameter.numel() for parameter in correction.parameters()
                                 if parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0)
                    self.assertEqual(active, expected)
                    self.assertEqual(metadata(mode=mode)["active_prefix_parameter_counts"][prefix], expected)
        # RAF's maximum sequence length84 admits the first seven thresholds.
        self.assertEqual(expected_counts[7], 8640)

    def test_first_eligible_branch_is_exact_same_function_under_copied_tensors(self):
        shared = SharedHistoryCorrection(mode="shared")
        independent = SharedHistoryCorrection(mode="independent")
        self.assertEqual(tuple(independent.input_projections[0].weight.shape), (15, 128))
        self.assertEqual([projection.weight.shape[0] for projection in independent.input_projections], [15] + [5] * 7)
        with torch.no_grad():
            independent.input_projections[0].weight.copy_(shared.input_projection.weight)
            activate(shared)
            independent.output_projections[0].weight.copy_(shared.output_projections[0].weight)
            hidden = torch.arange(2 * 64, dtype=torch.float32).reshape(1, 2, 64) / 71
            valid = torch.ones(1, 2, dtype=torch.bool)
            torch.testing.assert_close(shared(hidden, valid), independent(hidden, valid), rtol=1e-6, atol=1e-6)

    def test_common_tensors_global_rng_and_head_contract_are_bitwise_parent(self):
        for seed in (42, 52, 62):
            parent = make_parent(seed)
            expected_rng = torch.get_rng_state().clone()
            reference = {name: value for name, value in parent.state_dict().items()
                         if not name.startswith("multilag_detail.") and name != "history_width_identity"}
            for mode in MODE_BY_ARM.values():
                with self.subTest(seed=seed, mode=mode):
                    model = make_model(mode, seed)
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                    actual = {name: value for name, value in model.state_dict().items()
                              if not name.startswith("multilag_detail.") and name != "shared_history_identity"}
                    self.assertEqual(set(reference), set(actual))
                    for name, value in reference.items():
                        self.assertTrue(torch.equal(value, actual[name]), name)
                    self.assertEqual(model.time_head_contract(), parent.time_head_contract())

    def test_zero_initial_correction_matches_parent_encoder_and_heads_exactly(self):
        dt, qty, valid = inputs()
        with torch.no_grad():
            parent = make_parent()
            hidden = parent.encode_task_states(dt, qty, valid)[0]
            for mode in MODE_BY_ARM.values():
                with self.subTest(mode=mode):
                    model = make_model(mode)
                    self.assertTrue(all(torch.count_nonzero(p.weight) == 0
                                        for p in model.multilag_detail.output_projections))
                    actual = model.encode_task_states(dt, qty, valid)[0]
                    self.assertTrue(torch.equal(hidden, actual))
                    for value, expected in zip(model.predict_quantity(actual), parent.predict_quantity(hidden)):
                        self.assertTrue(torch.equal(value, expected))
                    self.assertTrue(torch.equal(model.log_observation_dt(actual.flatten(0, 1), dt.flatten()),
                                                parent.log_observation_dt(hidden.flatten(0, 1), dt.flatten())))

    def test_vectorized_formula_matches_explicit_event_loop_including_all_thresholds(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            hidden = torch.randn(2, 142, 64, dtype=torch.float64)
        valid = torch.ones(2, 142, dtype=torch.bool)
        valid[0, [0, 6, 9, 15, 21]] = False
        write = valid.clone()
        write[1, 7] = False
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                correction = SharedHistoryCorrection(mode=mode).double()
                activate(correction)
                expected = literal_loop(correction, hidden, valid, write)
                actual = correction(hidden, valid, memory_write_mask=write)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_every_output_threshold_activates_at_exact_observed_count(self):
        hidden = torch.ones(1, 130, 64)
        valid = torch.ones(1, 130, dtype=torch.bool)
        for mode in MODE_BY_ARM.values():
            for branch, lag in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
                with self.subTest(mode=mode, branch=branch), torch.no_grad():
                    correction = SharedHistoryCorrection(mode=mode)
                    projections = ([correction.input_projection] if mode == "shared"
                                   else correction.input_projections)
                    for projection in projections:
                        projection.weight.fill_(.01)
                    correction.output_projections[branch].weight.fill_(.02)
                    result = correction(hidden, valid)
                    self.assertEqual(torch.count_nonzero(result[:, :lag]).item(), 0)
                    self.assertTrue((result[:, lag] > 0).all())

    def test_padding_skips_and_withheld_resets_without_reading_absent_values(self):
        hidden = torch.arange(12 * 64, dtype=torch.float32).reshape(1, 12, 64) / 101
        valid = torch.ones(1, 12, dtype=torch.bool)
        valid[0, [1, 6]] = False
        write = valid.clone()
        write[0, 7] = False
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                correction = SharedHistoryCorrection(mode=mode)
                activate(correction)
                clean = correction(hidden, valid, memory_write_mask=write)
                altered = hidden.clone()
                altered[~write] = float("nan")
                result = correction(altered, valid, memory_write_mask=write)
                self.assertTrue(torch.equal(clean, result))
                self.assertEqual(torch.count_nonzero(result[0, [0, 7, 8]]).item(), 0)
                changed_past = hidden.clone()
                changed_past[:, :7] += 100
                torch.testing.assert_close(correction(changed_past, valid, memory_write_mask=write)[:, 8:], result[:, 8:])
                torch.testing.assert_close(result, literal_loop(correction, hidden, valid, write), rtol=2e-6, atol=2e-6)

    def test_correction_and_full_encoder_are_causal_with_nonzero_outputs(self):
        dt, qty, valid = inputs(14)
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate(model.multilag_detail)
                original = model.encode_task_states(dt, qty, valid)[0]
                changed_dt, changed_qty = dt.clone(), qty.clone()
                changed_dt[:, 8:] += 500
                changed_qty[:, 8:] += 1000
                changed = model.encode_task_states(changed_dt, changed_qty, valid)[0]
                short = model.encode_task_states(dt[:, :8], qty[:, :8], valid[:, :8])[0]
                torch.testing.assert_close(original[:, :8], changed[:, :8], rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(original[:, :8], short, rtol=1e-6, atol=1e-6)

    def test_insertion_is_between_existing_blocks_with_static_memory_unchanged(self):
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate(model.multilag_detail)
                features = model.continuous_features(dt, qty, valid)
                encoded = model.encoder.input_proj(features)
                if model.encoder.use_pos_emb:
                    encoded = encoded + model.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
                first = model.encoder.layers[0](encoded, mask=valid)
                second = model.encoder.layers[1](first + model.multilag_detail(first, valid), mask=valid)
                expected = model.lmm(second)
                actual = model.encode_task_states(dt, qty, valid)[0]
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(len(model.encoder.layers), 2)

    def test_all_correction_parameters_get_gradients_after_first_output_update(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73)
            hidden = torch.randn(2, 137, 64)
            target = torch.randn_like(hidden)
        valid = torch.ones(2, 137, dtype=torch.bool)
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode):
                correction = SharedHistoryCorrection(mode=mode)
                optimizer = torch.optim.SGD(correction.parameters(), lr=.2)
                F.mse_loss(correction(hidden, valid), target).backward()
                for name, parameter in correction.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    if name.startswith("input"):
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
                    else:
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                F.mse_loss(correction(hidden, valid), target).backward()
                for name, parameter in correction.named_parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_shared_input_gradient_accumulates_from_all_active_branches(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            hidden = torch.randn(1, 131, 64)
        valid = torch.ones(1, 131, dtype=torch.bool)
        correction = SharedHistoryCorrection()
        activate(correction)
        # Linear scalar objective makes contributions additive across branches.
        correction(hidden, valid).sum().backward()
        jointly = correction.input_projection.weight.grad.clone()
        separate = torch.zeros_like(jointly)
        original = deepcopy(correction.state_dict())
        for branch in range(8):
            correction.load_state_dict(original)
            with torch.no_grad():
                for index, projection in enumerate(correction.output_projections):
                    if index != branch:
                        projection.weight.zero_()
            correction.zero_grad(set_to_none=True)
            correction(hidden, valid).sum().backward()
            contribution = correction.input_projection.weight.grad
            self.assertGreater(contribution.abs().sum().item(), 0)
            separate += contribution
        torch.testing.assert_close(jointly, separate, rtol=2e-6, atol=2e-6)

    def test_batch_independence_and_repeated_eval_preserve_state_and_context(self):
        dt, qty, valid = inputs()
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode), torch.no_grad():
                model = make_model(mode)
                activate(model.multilag_detail)
                before = deepcopy(model.state_dict())
                contexts = [layer.attn._ctx_mem for layer in model.encoder.layers]
                before_inputs = tuple(value.clone() for value in (dt, qty, valid))
                batched = model.encode_task_states(dt, qty, valid)[0]
                singles = torch.cat([model.encode_task_states(dt[i:i+1], qty[i:i+1], valid[i:i+1])[0]
                                     for i in range(2)])
                repeated = model.encode_task_states(dt, qty, valid)[0]
                torch.testing.assert_close(batched, singles, rtol=1e-6, atol=1e-6)
                self.assertTrue(torch.equal(batched, repeated))
                for name, value in before.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                for before_context, layer in zip(contexts, model.encoder.layers):
                    self.assertIs(before_context, layer.attn._ctx_mem)
                for old, current in zip(before_inputs, (dt, qty, valid)):
                    self.assertTrue(torch.equal(old, current))

    def test_observed_bad_values_rejected_absent_values_and_empty_shapes_safe(self):
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode):
                correction = SharedHistoryCorrection(mode=mode)
                activate(correction)
                valid = torch.ones(1, 3, dtype=torch.bool)
                for bad in (float("nan"), float("inf"), -float("inf")):
                    hidden = torch.ones(1, 3, 64)
                    hidden[0, 0, 0] = bad
                    with self.assertRaises(ValueError):
                        correction(hidden, valid)
                self.assertEqual(correction(torch.empty(2, 0, 64), torch.empty(2, 0, dtype=torch.bool)).shape, (2, 0, 64))
                output = correction(torch.full((2, 3, 64), float("nan")), torch.zeros(2, 3, dtype=torch.bool))
                self.assertEqual(torch.count_nonzero(output).item(), 0)
                for mask in (valid.float(), torch.ones(1, 2, dtype=torch.bool)):
                    with self.assertRaises(ValueError):
                        correction(torch.ones(1, 3, 64), mask)
                    with self.assertRaises(ValueError):
                        correction(torch.ones(1, 3, 64), valid, memory_write_mask=mask)
                for hidden in (torch.ones(1, 3), torch.ones(1, 3, 63), torch.ones(1, 3, 64, dtype=torch.long)):
                    with self.assertRaises(ValueError):
                        correction(hidden, valid)

    def test_invalid_dimensions_modes_and_common_science_changes_rejected(self):
        for dimension in (32, True, 64.):
            with self.assertRaises(ValueError):
                SharedHistoryCorrection(dimension)
        for mode in ("width16", None, 5):
            with self.assertRaises(ValueError):
                SharedHistoryCorrection(mode=mode)
        for key, value in (("quantity_variant", "lognormal_quantity"), ("memory_mode", "none"),
                           ("lambda_tail", .1), ("time_head_mode", "legacy_clamped")):
            options = deepcopy(KWARGS)
            options[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CountAwareTitanSharedHistoryMLP(64, 1.2, 180, **options)

    def test_checkpoint_roundtrip_same_arm_and_reject_cross_arm_before_mutation(self):
        for mode in MODE_BY_ARM.values():
            with self.subTest(mode=mode):
                model = make_model(mode)
                activate(model.multilag_detail)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                state = torch.load(buffer, weights_only=True)
                target = make_model(mode, 52)
                target.load_state_dict(state)
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, target.state_dict()[name]), name)
                other = make_model("independent" if mode == "shared" else "shared")
                before = deepcopy(other.state_dict())
                with self.assertRaises(ValueError):
                    other.load_state_dict(state, strict=False)
                for name, value in before.items():
                    self.assertTrue(torch.equal(value, other.state_dict()[name]), name)
                with self.assertRaises(ValueError):
                    make_parent().load_state_dict(state, strict=False)

    def test_malformed_state_rejected_atomically_even_strict_false(self):
        for mode in MODE_BY_ARM.values():
            model = make_model(mode)
            state = deepcopy(model.state_dict())
            correction_key = next(key for key in state if key.startswith("multilag_detail."))
            cases = []
            altered = deepcopy(state); altered.pop("shared_history_identity"); cases.append(altered)
            altered = deepcopy(state); altered["shared_history_identity"] = state["shared_history_identity"].float(); cases.append(altered)
            altered = deepcopy(state); altered["core_ablation_identity"][0] ^= 1; cases.append(altered)
            altered = deepcopy(state); altered[correction_key] = torch.zeros(1); cases.append(altered)
            altered = deepcopy(state); altered[correction_key].flatten()[0] = float("nan"); cases.append(altered)
            altered = deepcopy(state); altered["dummy.weight"] = torch.zeros(1); cases.append(altered)
            for index, altered in enumerate(cases):
                with self.subTest(mode=mode, case=index):
                    with self.assertRaises((ValueError, RuntimeError)):
                        model.load_state_dict(altered, strict=False)
                    for name, value in state.items():
                        self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_payload_binds_metadata_role_scope_head_selector_and_projection_shapes(self):
        for arm, mode in MODE_BY_ARM.items():
            reference = payload(make_model(mode))
            self.assertTrue(validate_checkpoint(reference, arm))
            cases = []
            for key, value in (("backbone", "titantpp_history_mlp_width16"), ("held_out_test_evaluated", True),
                               ("evaluation_scope", "test"), ("variant", "raw_mse")):
                altered = deepcopy(reference); altered[key] = value; cases.append(altered)
            altered = deepcopy(reference); altered["encoder_config"]["history_relation_sharing"] = "other"; cases.append(altered)
            altered = deepcopy(reference); altered["resume_identity"]["arguments"]["model_role"] = "other"; cases.append(altered)
            altered = deepcopy(reference); altered["resume_identity"]["checkpoint_monitor"] = "mae"; cases.append(altered)
            altered = deepcopy(reference); altered["encoder_config"]["time_head"]["time_scale"] = float("nan"); cases.append(altered)
            altered = deepcopy(reference); altered["model_state_dict"]["shared_history_identity"][0] ^= 1; cases.append(altered)
            key = next(key for key in reference["model_state_dict"] if key.startswith("multilag_detail."))
            altered = deepcopy(reference); altered["model_state_dict"][key] = torch.zeros(1); cases.append(altered)
            altered = deepcopy(reference); altered["model_state_dict"][key].flatten()[0] = float("inf"); cases.append(altered)
            altered = deepcopy(reference); altered["model_state_dict"]["history_width_identity"] = torch.zeros(32, dtype=torch.uint8); cases.append(altered)
            altered = deepcopy(reference); altered["model_state_dict"]["multilag_detail.dummy"] = torch.zeros(1); cases.append(altered)
            for index, altered in enumerate(cases):
                with self.subTest(mode=mode, case=index), self.assertRaises(ValueError):
                    validate_checkpoint(altered, arm)
            with self.assertRaises(ValueError):
                validate_checkpoint(reference, "titantpp_history_mlp_width16")
            with self.assertRaises(ValueError):
                validate_checkpoint(reference, next(name for name in ARMS if name != arm))

    def test_summary_requires_bound_digest_positive_epoch_and_identifies_renamed_arm(self):
        reference = payload(make_model())
        reference.pop("model_state_dict")
        reference.update(checkpoint_state_sha256="a" * 64, best_epoch=3)
        self.assertTrue(validate_checkpoint(reference, ARMS[0]))
        for digest, epoch in (("x" * 64, 3), ("a" * 63, 3), ("a" * 64, True), ("a" * 64, 0)):
            altered = deepcopy(reference)
            altered.update(checkpoint_state_sha256=digest, best_epoch=epoch)
            with self.subTest(digest=digest, epoch=epoch), self.assertRaises(ValueError):
                validate_checkpoint(altered, ARMS[0])
        renamed = payload(make_model())
        renamed["backbone"] = "unrelated"
        renamed["encoder_config"] = {}
        with self.assertRaises(ValueError):
            validate_checkpoint(renamed, "unrelated")
        self.assertFalse(validate_checkpoint({"backbone": "unrelated"}, "unrelated"))


if __name__ == "__main__":
    unittest.main()
