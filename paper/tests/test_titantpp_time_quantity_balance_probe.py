"""Fabricated-only checks for the no-update Train-gradient diagnostic."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts import probe_titantpp_time_quantity_balance as probe


class ToyModel(nn.Module):
    def __init__(self, zero_quantity=False):
        super().__init__()
        self.shared = nn.Parameter(torch.tensor([.4, .6], dtype=torch.float64))
        self.time_head = nn.Parameter(torch.tensor([.3, .2], dtype=torch.float64))
        self.quantity_head = nn.Parameter(torch.tensor([0., 0.] if zero_quantity else [.5, .7], dtype=torch.float64))
        self.quantity_bias = nn.Parameter(torch.tensor(1., dtype=torch.float64))
        self.inactive = nn.Parameter(torch.tensor(7., dtype=torch.float64))
        self.dropout = nn.Dropout(.2)
        self.register_buffer("identity", torch.tensor([1, 2], dtype=torch.uint8))
        self._ctx_mem = None
        self.calls = 0


def toy_forward(model, dts, mask, quantities, *, lambda_log_qty):
    model.calls += 1
    hidden = model.dropout(dts[:, :2]) * model.shared
    t = (hidden * model.time_head).sum(1)
    p = (hidden * model.quantity_head).sum(1) + model.quantity_bias
    truth = quantities[:, -1]
    log_mse = (torch.log1p(p) - torch.log1p(truth)).square()
    return {"time_loss": (t - 2.).square(), "quantity_train_loss": log_mse,
        "log_qty_loss": log_mse, "pred_qty": p, "true_qty": truth}


def inputs():
    dt = torch.tensor([[1., 2., 3.], [2., 1., 1.]], dtype=torch.float64)
    q = torch.tensor([[1., 1., 2.], [2., 2., 3.]], dtype=torch.float64)
    return dt, torch.ones_like(dt, dtype=torch.bool), q


class GradientArithmeticTests(unittest.TestCase):
    def test_exact_group_norm_cosine_joint_and_lambda(self):
        names = ["shared", "time", "quantity", "unused"]
        parameters = [torch.zeros(2), torch.zeros(1), torch.zeros(1), torch.zeros(3)]
        t = [torch.tensor([3., 4.]), torch.tensor([12.]), None, None]
        q = [torch.tensor([0., -5.]), None, torch.tensor([2.]), None]
        result = probe.compare_gradients(names, parameters, t, q, 100., 2.)
        shared = result["groups"]["shared_graph"]
        self.assertEqual(shared["parameter_names"], ["shared"])
        self.assertEqual(shared["time_gradient_norm"], 5.)
        self.assertEqual(shared["quantity_gradient_norm"], 5.)
        self.assertAlmostEqual(shared["gradient_cosine"], -.8)
        self.assertEqual(shared["lambda_quantity_for_norm_equality"], 1.)
        self.assertAlmostEqual(result["global"]["lambda1_joint_gradient_norm"], math.sqrt(158))
        self.assertTrue(result["global"]["lambda1_joint_norm_exceeds_clip1"])
        self.assertFalse(result["global"]["actual_clipping_applied"])
        self.assertEqual(result["loss_scalar"]["time_to_quantity_scalar_ratio"], 50.)
        self.assertEqual(result["groups"]["inactive_none"]["parameter_names"], ["unused"])

    def test_zero_gradient_is_graph_present_but_not_transmitted(self):
        r = probe.compare_gradients(["shared"], [torch.zeros(2)], [torch.tensor([1., 2.])],
            [torch.zeros(2)], 1., 0.)
        self.assertEqual(r["groups"]["shared_graph"]["parameter_tensor_count"], 1)
        self.assertEqual(r["groups"]["shared_nonzero_both"]["parameter_tensor_count"], 0)
        self.assertEqual(r["groups"]["shared_time_nonzero_quantity_zero"]["parameter_names"], ["shared"])
        self.assertIsNone(r["groups"]["shared_graph"]["gradient_cosine"])
        self.assertIsNone(r["groups"]["shared_graph"]["lambda_quantity_for_norm_equality"])
        self.assertIsNone(r["loss_scalar"]["time_to_quantity_scalar_ratio"])
        json.dumps(r, allow_nan=False)

    def test_opposing_gradients_cancel_without_clipping(self):
        r = probe.compare_gradients(["p"], [torch.zeros(2)], [torch.tensor([3., 4.])],
            [torch.tensor([-3., -4.])], 1., 1.)
        self.assertEqual(r["global"]["lambda1_joint_gradient_norm"], 0.)
        self.assertEqual(r["global"]["gradient_cosine"], -1.)
        self.assertFalse(r["global"]["lambda1_joint_norm_exceeds_clip1"])

    def test_nonfinite_gradient_rejected(self):
        with self.assertRaises(ValueError):
            probe.compare_gradients(["x"], [torch.zeros(1)], [torch.tensor([float("inf")])], [None], 1., 1.)

    def test_shape_and_inventory_mismatches_rejected(self):
        for t, q in (([], [None]), ([torch.ones(2)], [None])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                probe.compare_gradients(["x"], [torch.zeros(1)], t, q, 1., 1.)

    def test_cap_normalized_losses_match_hand_calculation(self):
        truth = torch.tensor([2., 4.], requires_grad=False)
        pred = torch.tensor([4., 3.], requires_grad=True)
        log = (torch.log1p(pred) - torch.log1p(truth)).square()
        value = {"time_loss": pred.square(), "quantity_train_loss": log,
            "log_qty_loss": log, "pred_qty": pred, "true_qty": truth}
        losses = probe.loss_components(value, 10.)
        self.assertAlmostEqual(float(losses["cap_normalized_raw_mse"].detach()), .025, places=7)
        self.assertAlmostEqual(float(losses["cap_normalized_raw_mae"].detach()), .15, places=7)
        g = torch.autograd.grad(losses["cap_normalized_raw_mse"], pred, retain_graph=True)[0]
        torch.testing.assert_close(g, torch.tensor([.02, -.01]))
        g = torch.autograd.grad(losses["cap_normalized_raw_mae"], pred)[0]
        torch.testing.assert_close(g, torch.tensor([.05, -.05]))

    def test_mae_zero_subgradient_and_invalid_losses(self):
        p = torch.tensor([2.], requires_grad=True)
        z = (p-p.detach()).square()
        value = {"time_loss": p.square(), "quantity_train_loss": z,
            "log_qty_loss": z, "pred_qty": p, "true_qty": p.detach()}
        self.assertEqual(float(torch.autograd.grad(probe.loss_components(value, 10.)["cap_normalized_raw_mae"], p)[0]), 0.)
        for cap in (0., -1., float("nan")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                probe.loss_components(value, cap)
        changed = dict(value, quantity_train_loss=z+1)
        with self.assertRaises(ValueError):
            probe.loss_components(changed, 10.)


class SameForwardInvarianceTests(unittest.TestCase):
    def test_one_forward_four_grad_calls_and_no_grad_state_mutation(self):
        model = ToyModel().eval()
        model.shared.grad = torch.ones_like(model.shared)
        original_grad = model.shared.grad.clone()
        original = probe.state_sha(model)
        rng_before = probe.rng_sha(probe.capture_rng())
        with probe.preserve_rng():
            with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
                result = probe.analyze_model_batch(model, *inputs(), cap=10., forward=toy_forward)
                self.assertEqual(grad.call_count, 4)
        self.assertEqual(model.calls, 1)
        self.assertFalse(model.training)
        self.assertEqual(probe.state_sha(model), original)
        self.assertTrue(torch.equal(model.shared.grad, original_grad))
        self.assertEqual(rng_before, probe.rng_sha(probe.capture_rng()))
        self.assertTrue(result["rng_unchanged_by_gradient_computation"])
        self.assertTrue(result["dropout_rng_consumed_by_forward"])
        self.assertEqual(set(result["comparisons"]), set(probe.QUANTITY_LOSSES))
        for comparison in result["comparisons"].values():
            self.assertEqual(comparison["groups"]["shared_graph"]["parameter_names"], ["shared"])
            self.assertIn("inactive", comparison["groups"]["inactive_none"]["parameter_names"])
            self.assertIn("quantity_head", comparison["groups"]["quantity_only_graph"]["parameter_names"])

    def test_frozen_parameters_temporarily_enabled_then_restored(self):
        model = ToyModel().requires_grad_(False)
        with probe.preserve_rng():
            result = probe.analyze_model_batch(model, *inputs(), cap=10., forward=toy_forward)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertGreater(result["comparisons"]["existing_log_mse"]["groups"]["shared_graph"]["quantity_gradient_norm"], 0.)

    def test_fresh_zero_quantity_head_is_not_falsely_called_active_shared(self):
        model = ToyModel(zero_quantity=True)
        with probe.preserve_rng():
            result = probe.analyze_model_batch(model, *inputs(), cap=10., forward=toy_forward)
        for r in result["comparisons"].values():
            self.assertEqual(r["groups"]["shared_graph"]["quantity_gradient_norm"], 0.)
            self.assertIsNone(r["groups"]["shared_graph"]["lambda_quantity_for_norm_equality"])
            self.assertEqual(r["groups"]["shared_nonzero_both"]["parameter_tensor_count"], 0)
            self.assertGreater(r["groups"]["quantity_only_graph"]["quantity_gradient_norm"], 0.)

    def test_episodic_context_mutation_rejected_and_restored(self):
        model = ToyModel()
        model._ctx_mem = torch.ones(2)
        def mutated(*args, **kwargs):
            result = toy_forward(*args, **kwargs)
            args[0]._ctx_mem = torch.zeros(2)
            return result
        with self.assertRaisesRegex(ValueError, "episodic context"), probe.preserve_rng():
            probe.analyze_model_batch(model, *inputs(), cap=10., forward=mutated)
        self.assertTrue(torch.equal(model._ctx_mem, torch.ones(2)))

    def test_parameter_or_buffer_mutation_rejected(self):
        model = ToyModel()
        def mutated(*args, **kwargs):
            result = toy_forward(*args, **kwargs)
            args[0].identity.add_(1)
            return result
        with self.assertRaisesRegex(ValueError, "tensor values"), probe.preserve_rng():
            probe.analyze_model_batch(model, *inputs(), cap=10., forward=mutated)

    def test_rng_restoration_even_on_exception_and_no_cpu_cuda_initialization(self):
        before = probe.rng_sha(probe.capture_rng())
        with self.assertRaises(RuntimeError), probe.preserve_rng():
            torch.rand(5)
            raise RuntimeError("fabricated failure")
        self.assertEqual(before, probe.rng_sha(probe.capture_rng()))
        with patch.object(torch.cuda, "is_initialized", return_value=False), patch.object(torch.cuda, "get_rng_state_all") as cuda:
            self.assertIsNone(probe.capture_rng()["cuda"])
            cuda.assert_not_called()

    def test_real_fresh_mlp16_fabricated_train_batch(self):
        from paper.scripts import run_titantpp_mlp_cap36 as cap36
        with probe.preserve_rng():
            torch.manual_seed(42)
            torch.set_num_threads(4)
            data = cap36.fabricated_data()
            data["loader"]["max_seq_len"] = 8
            model, _ = cap36.build_model(data, probe.ARM)
            original = probe.state_sha(model)
            dts = torch.tensor([[1., 2., 1., 3.], [1., 1., 2., 1.]])
            q = torch.tensor([[1., 2., 3., 4.], [2., 3., 1., 2.]])
            r = probe.analyze_model_batch(model, dts, torch.ones_like(dts, dtype=torch.bool), q, cap=10.)
            self.assertTrue(bool((model.quantity_head.weight == 0).all()))
            self.assertEqual(probe.state_sha(model), original)
            for comparison in r["comparisons"].values():
                self.assertEqual(comparison["groups"]["shared_graph"]["quantity_gradient_norm"], 0.)
                self.assertIsNone(comparison["groups"]["shared_graph"]["gradient_cosine"])
                self.assertGreater(comparison["groups"]["quantity_only_graph"]["quantity_gradient_norm"], 0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            json.dumps(r, allow_nan=False)


class SamplingAndBindingTests(unittest.TestCase):
    def test_recording_sampler_preserves_exact_seeded_loader_batches(self):
        dataset = TensorDataset(torch.arange(43))
        original = DataLoader(dataset, batch_size=7, shuffle=True, generator=torch.Generator().manual_seed(52))
        copied = DataLoader(dataset, batch_size=7, shuffle=True, generator=torch.Generator().manual_seed(52))
        wrapped, sampler = probe.recording_loader(copied)
        expected = [batch[0].tolist() for batch in original]
        actual = [batch[0].tolist() for batch in wrapped]
        self.assertEqual(actual, expected)
        self.assertEqual(sampler.batches, expected)

    def test_cli_bounded_and_selected_arguments(self):
        args = probe.parse_args(["--dataset", "raf_spare_parts", "--seed", "42", "--output", "/tmp/probe.json"])
        self.assertEqual(args.num_batches, 3)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.initialization, "fresh")
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            probe.parse_args(["--dataset", "raf_spare_parts", "--seed", "32", "--output", "/tmp/probe.json"])
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            probe.parse_args(["--dataset", "raf_spare_parts", "--seed", "42", "--num-batches", "4", "--output", "/tmp/probe.json"])

    def test_selected_checkpoint_bound_load_and_reject_wrong_identity(self):
        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
        model = ToyModel()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "selected.pt"
            state = copy.deepcopy(model.state_dict())
            digest = canonical_state_dict_sha256(state)
            cp = {"backbone": probe.ARM, "seed": 42, "variant": "count_only_log_regression",
                "checkpoint_monitor": "validation_raw_quantity_rmse", "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False, "best_epoch": 3, "model_state_dict": state,
                "model_state_sha256": digest, "resume_identity": {"arguments": {"dataset_contract": "raf_spare_parts"}}}
            torch.save(cp, path)
            bound = dict(expected_sha=probe.sha_file(path), expected_state_sha=digest,
                dataset="raf_spare_parts", seed=42, expected_epoch=3)
            with patch("models.TPPs.CountAwareFactory.validate_checkpoint_route"):
                receipt = probe.load_selected(model, path, **bound)
                self.assertEqual(receipt["state_sha256"], digest)
                for key, value in (("expected_sha", "0"*64), ("expected_state_sha", "0"*64),
                    ("dataset", "yellow_trip_hourly"), ("seed", 52), ("expected_epoch", 4)):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        probe.load_selected(model, path, **{**bound, key: value})
            self.assertEqual(probe.sha_file(path), bound["expected_sha"])

    def test_batch_identity_rejects_non_train_and_wrong_truth(self):
        class Dataset:
            index = [(0, 0), (0, 1)]
            parts = ["A"]
            seq_lists = [[1, 2, 3]]
            val_lists = [[1., 2., 3.]]
            split_lists = [["train", "train", "train"]]
        d = Dataset()
        dt, mask, q = inputs()
        batch = (None, dt, mask, torch.tensor([0, 0]), q)
        identity = probe.batch_identity(d, [0, 1], batch)
        self.assertEqual([r["target_seq"] for r in identity["targets"]], [2, 3])
        q[1, -1] = 4.
        with self.assertRaisesRegex(ValueError, "truth"):
            probe.batch_identity(d, [0, 1], batch)
        q[1, -1] = 3.
        d.split_lists = [["train", "train", "not_train"]]
        with self.assertRaisesRegex(ValueError, "Non-Train"):
            probe.batch_identity(d, [0, 1], batch)


if __name__ == "__main__":
    unittest.main()
