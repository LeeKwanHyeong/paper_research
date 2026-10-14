"""Synthetic CPU checks for a single frozen-rho GRU output cap; no data access."""
from copy import deepcopy
import io
import unittest

import torch

from models.TPPs.CountAwareTitanCNNGRU import CountAwareTitanCNNGRU
from models.TPPs.CountAwareTitanCNNGRUComponents import CountAwareTitanCNNGRUComponents
from models.TPPs.CountAwareTitanCNNGRUResidualCap import (
    FULL, CNN, CAPPED, RHO, IDENTITY_BUFFER, CountAwareTitanCNNGRUResidualCap,
    cap_gru_residual, correction_parameters, metadata, identity, role_for_arm, validate_checkpoint,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


class GRUResidualCapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, kind="capped", seed=42, max_len=256):
        torch.manual_seed(seed)
        options = dict(hidden_dim=64, train_log_mean=1.2, train_log_std=.8, max_seq_len=max_len,
            quantity_variant="count_only_log_regression", time_head_mode="heteroscedastic_lognormal_duration",
            time_scale=1., time_initial_location=.2, time_initial_scale=.8,
            time_observation_contract={"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None})
        if kind == "capped":
            return CountAwareTitanCNNGRUResidualCap(**options)
        if kind == "cnn":
            return CountAwareTitanCNNGRUComponents(**options, component_arm=CNN)
        return CountAwareTitanCNNGRU(**options, candidate_arm=FULL)

    def inputs(self):
        gaps = torch.tensor([[0., 1., 3., 2., 5., 7.], [0., 0., 2., 4., 1., 8.]])
        q = torch.tensor([[0., 2., 4., 8., 3., 11.], [0., 0., 5., 2., 7., 13.]])
        valid = torch.tensor([[False, True, True, True, True, True], [False, False, True, True, True, True]])
        return gaps, q, valid

    def activate(self, model):
        generator = torch.Generator().manual_seed(77)
        with torch.no_grad():
            for parameter in correction_parameters(model).values():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .08)

    def payload(self, model=None):
        model = self.model() if model is None else model
        head = model.time_head_contract()
        interface = {"time_head": head}
        return dict(backbone=CAPPED, encoder_config={**metadata(), "time_head": head},
            model_state_dict=model.state_dict(), evaluation_scope="validation_only", held_out_test_evaluated=False,
            variant="count_only_log_regression", interface_meta=interface,
            resume_identity={"backbone": CAPPED, "interface_meta": interface,
                "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": role_for_arm(CAPPED)}})

    def test_saturated_direction_and_strict_norm_bound_for_all_output_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            generator = torch.Generator().manual_seed(987)
            base = torch.randn(24, 12, 64, dtype=torch.float64, generator=generator).to(dtype)
            raw = (torch.randn(24, 12, 64, dtype=torch.float64, generator=generator) * 40.).to(dtype)
            limited = cap_gru_residual(base, raw)
            radius = RHO * torch.linalg.vector_norm(base.double(), dim=-1)
            self.assertTrue(bool((torch.linalg.vector_norm(limited.double(), dim=-1) <= radius).all()), dtype)
            self.assertEqual(limited.dtype, dtype)
            cosine = torch.nn.functional.cosine_similarity(raw.double(), limited.double(), dim=-1)
            self.assertTrue(torch.allclose(cosine, torch.ones_like(cosine), atol=2e-5, rtol=2e-5), dtype)

    def test_unsaturated_tiny_values_and_zero_residual_preserve_identity_and_gradients(self):
        for dtype, small in ((torch.float32, 1e-30), (torch.float64, 1e-140)):
            base = torch.full((2, 3, 64), small, dtype=dtype, requires_grad=True)
            raw = torch.full_like(base, small * .01, requires_grad=True)
            limited = cap_gru_residual(base, raw)
            self.assertTrue(torch.equal(limited, raw))
            limited.sum().backward()
            self.assertTrue(torch.equal(raw.grad, torch.ones_like(raw)))
            self.assertIsNone(base.grad)
        base = torch.ones(1, 3, 64, requires_grad=True)
        raw = torch.zeros_like(base, requires_grad=True)
        cap_gru_residual(base, raw).sum().backward()
        self.assertTrue(torch.equal(raw.grad, torch.ones_like(raw)))
        self.assertIsNone(base.grad)

    def test_zero_reference_and_saturated_gradient_are_finite_and_radius_is_detached(self):
        for magnitude in (0., 100.):
            base = torch.zeros(2, 3, 64, requires_grad=True)
            raw = torch.full_like(base, magnitude, requires_grad=True)
            limited = cap_gru_residual(base, raw)
            self.assertTrue(torch.equal(limited, torch.zeros_like(limited)))
            limited.sum().backward()
            self.assertTrue(bool(torch.isfinite(raw.grad).all()))
            self.assertTrue(torch.equal(raw.grad, torch.zeros_like(raw)))
            self.assertIsNone(base.grad)
        base = torch.randn(2, 3, 64, requires_grad=True)
        raw = (torch.randn(2, 3, 64) * 20.).requires_grad_(True)
        (cap_gru_residual(base, raw) * torch.arange(64.)).sum().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertGreater(float(raw.grad.abs().sum()), 0.)
        self.assertIsNone(base.grad)

    def test_dtype_rounding_subnormal_huge_and_repeated_projection_respect_bound(self):
        for dtype in (torch.float32, torch.float64):
            smallest = torch.nextafter(torch.tensor(0., dtype=dtype), torch.tensor(1., dtype=dtype)).item()
            base = torch.zeros(1, 3, 64, dtype=dtype)
            raw = torch.zeros_like(base)
            base[0, 0, 0], raw[0, 0, 0] = smallest * 7, smallest * 23
            base[0, 1, :], raw[0, 1, :] = 1e-20, 1e20
            base[0, 2, :], raw[0, 2, :] = 1e20, 1e30
            limited = cap_gru_residual(base, raw)
            radius = RHO * torch.linalg.vector_norm(base.double(), dim=-1)
            self.assertTrue(bool(torch.isfinite(limited).all()))
            self.assertTrue(bool((torch.linalg.vector_norm(limited.double(), dim=-1) <= radius).all()))
            self.assertTrue(torch.equal(cap_gru_residual(base, limited), limited))

    def test_exact_parent_initial_tensors_rng_counts_and_zero_output_forward(self):
        gaps, q, valid = self.inputs()
        for seed in (42, 52, 62):
            parent = self.model("full", seed)
            expected_rng = torch.get_rng_state().clone()
            candidate = self.model("capped", seed)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertEqual(set(candidate.state_dict()) - set(parent.state_dict()), {IDENTITY_BUFFER})
            for name, value in parent.state_dict().items():
                self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
            self.assertEqual(list(dict(candidate.named_parameters())), list(dict(parent.named_parameters())))
            self.assertEqual(sum(p.numel() for p in candidate.parameters()), 115167)
            self.assertEqual(sum(p.numel() for p in candidate.parameters() if p.requires_grad), 115167)
            for training in (False, True):
                parent.train(training)
                candidate.train(training)
                torch.manual_seed(99)
                a = target_outputs(parent, gaps, valid, q, lambda_log_qty=1.)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(99)
                b = target_outputs(candidate, gaps, valid, q, lambda_log_qty=1.)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key in a:
                    self.assertTrue(torch.equal(a[key], b[key]), (seed, training, key))
        # RAF's fixed shorter positional table has the same count as its FULL.
        a, b = self.model("full", max_len=84), self.model("capped", max_len=84)
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))

    def test_zero_initialized_output_projection_learns_then_recurrent_gradients_flow(self):
        gaps, q, valid = self.inputs()
        model = self.model().eval()
        params = dict(model.named_parameters())
        before = {name: value.detach().clone() for name, value in correction_parameters(model).items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = target_outputs(model, gaps, valid, q, lambda_log_qty=1.)["joint_loss"].mean()
            loss.backward()
            output = params["multilag_detail.output_projection.weight"].grad
            self.assertIsNotNone(output)
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertGreater(float(output.abs().sum()), 0.)
            if step >= 1:
                for name in ("multilag_detail.gru.weight_ih_l0", "multilag_detail.gru.weight_hh_l0", "multilag_detail.input_projection.weight"):
                    self.assertIsNotNone(params[name].grad)
                    self.assertTrue(bool(torch.isfinite(params[name].grad).all()))
                    self.assertGreater(float(params[name].grad.abs().sum()), 0., name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        for name in ("multilag_detail.output_projection.weight", "multilag_detail.gru.weight_hh_l0"):
            self.assertFalse(torch.equal(before[name], params[name].detach()))

    def test_padding_withheld_reset_and_future_prefix_invariance(self):
        model = self.model().eval()
        self.activate(model)
        correction = model.multilag_detail
        hidden = torch.randn(2, 7, 64, generator=torch.Generator().manual_seed(19))
        valid = torch.ones(2, 7, dtype=torch.bool)
        write = valid.clone()
        write[:, 2] = False
        hidden[:, 2] = float("nan")
        reset = correction(hidden, valid, memory_write_mask=write)
        fresh = correction(hidden[:, 3:], valid[:, 3:])
        self.assertTrue(torch.allclose(reset[:, 3:], fresh, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.equal(reset[:, 2], torch.zeros_like(reset[:, 2])))
        compact = hidden[:, 3:]
        padded = torch.cat((torch.full((2, 3, 64), float("nan")), compact), 1)
        padded_mask = torch.cat((torch.zeros(2, 3, dtype=torch.bool), valid[:, 3:]), 1)
        result = correction(padded, padded_mask)
        self.assertTrue(torch.allclose(result[:, 3:], fresh, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.equal(result[:, :3], torch.zeros_like(result[:, :3])))
        changed = compact.clone()
        changed[:, 2:] += 50.
        self.assertTrue(torch.equal(correction(changed, valid[:, 3:])[:, :2], fresh[:, :2]))

    def test_target_quantity_gap_block_and_single_batch_match_after_activation(self):
        model = self.model().eval().requires_grad_(False)
        self.activate(model)
        gaps, q, valid = self.inputs()
        with torch.inference_mode():
            base = target_outputs(model, gaps, valid, q, lambda_log_qty=1.)
            changed_q = q.clone()
            changed_q[:, -1] += 12345.
            cq = target_outputs(model, gaps, valid, changed_q, lambda_log_qty=1.)
            changed_gaps = gaps.clone()
            changed_gaps[:, -1] += 123.
            ct = target_outputs(model, changed_gaps, valid, q, lambda_log_qty=1.)
            single = target_outputs(model, gaps[:1], valid[:1], q[:1], lambda_log_qty=1.)
        self.assertTrue(torch.equal(base["pred_qty"], cq["pred_qty"]))
        self.assertTrue(torch.equal(base["time_loss"], cq["time_loss"]))
        self.assertTrue(torch.equal(base["pred_qty"], ct["pred_qty"]))
        for key in ("pred_qty", "time_loss"):
            self.assertTrue(torch.allclose(base[key][:1], single[key], rtol=1e-5, atol=1e-5), key)

    def test_strict_state_serialization_and_checkpoint_relabelling_or_rho_tamper(self):
        model = self.model()
        payload = self.payload(model)
        self.assertTrue(validate_checkpoint(payload, CAPPED))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.model(seed=99)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        self.assertTrue(all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items()))
        for other in (FULL, CNN, "foreign"):
            with self.assertRaises(ValueError):
                validate_checkpoint(payload, other)
        for original in (self.model("full"), self.model("cnn")):
            self.assertFalse(validate_checkpoint({"model_state_dict": original.state_dict()}, FULL))
            with self.assertRaises((ValueError, RuntimeError)):
                restored.load_state_dict(original.state_dict(), strict=False)
        for key, value in (("residual_cap_rho", .5), ("residual_cap_radius", "attached"), ("residual_cap_parameter_count", 1)):
            changed = deepcopy(payload)
            changed["encoder_config"][key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint(changed, CAPPED)
        changed = deepcopy(payload)
        changed["model_state_dict"][IDENTITY_BUFFER][0] ^= 1
        with self.assertRaises(ValueError):
            validate_checkpoint(changed, CAPPED)
        with self.assertRaises(ValueError):
            restored.load_state_dict(changed["model_state_dict"], strict=False)

    def test_invalid_rho_shape_nonfinite_and_missing_state_fail_before_mutation(self):
        with self.assertRaises(ValueError):
            metadata(rho=.5)
        with self.assertRaises(ValueError):
            identity(rho=.5)
        with self.assertRaises(ValueError):
            cap_gru_residual(torch.zeros(1, 2, 63), torch.ones(1, 2, 63))
        with self.assertRaises(ValueError):
            cap_gru_residual(torch.zeros(1, 2, 64), torch.full((1, 2, 64), float("nan")))
        model = self.model()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        incomplete = {name: value for name, value in before.items() if name != "quantity_head.weight"}
        with self.assertRaises((ValueError, RuntimeError)):
            model.load_state_dict(incomplete, strict=False)
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()))


if __name__ == "__main__":
    unittest.main()
