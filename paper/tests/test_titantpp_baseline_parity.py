"""Small CPU parity and failure-detection checks; no research data is loaded."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import random
import unittest

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_titantpp_baseline_parity.py"
SPEC = importlib.util.spec_from_file_location("baseline_parity_under_test", SCRIPT)
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)


class TinyModel(torch.nn.Module):
    def __init__(self, *, mean=False, extra=False, changed=False, rng_drift=False):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.dropout = torch.nn.Dropout(.1)
        if mean:
            self.register_buffer(parity.IDENTITY_BUFFER, torch.tensor([1], dtype=torch.int64))
        if extra:
            self.register_buffer("unexpected", torch.tensor([1.]))
        if changed:
            with torch.no_grad():
                self.linear.weight[0, 0] += .125
        if rng_drift:
            random.random()


def tiny_outputs(model, dts, mask, quantities, *, lambda_log_qty):
    hidden = torch.stack((dts.mean(1), quantities[:, :-1].mean(1), mask.float().mean(1)), 1)
    prediction = model.linear(model.dropout(hidden)).squeeze(-1)
    time = prediction.square() + 1.
    quantity = (prediction - quantities[:, -1]).square()
    zero = prediction * 0.
    return {"joint_loss": time + lambda_log_qty * quantity, "time_loss": time,
        "quantity_train_loss": quantity, "log_qty_loss": quantity,
        "quantity_distribution_nll": zero, "quantity_location_huber": zero,
        "quantity_scale": torch.ones_like(prediction), "tail_aux_loss": zero,
        "tail_indicator": zero, "true_qty": quantities[:, -1], "pred_qty": prediction,
        "history_length": mask.sum(1) - 1}


def optimizer(model, *, lr, time_head_lr_multiplier):
    return torch.optim.AdamW(model.parameters(), lr=lr)


class BaselineParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return {"seed": 42, "expected_trainable_parameters": 4,
            "data_spec": {"model": {"lambda_log_qty": 1.}},
            "optimizer": {"lr": .001, "time_head_lr_multiplier": 1., "grad_clip": 1.,
                "weight_decay": .01, "eps": 1e-8, "betas": [.9, .999]},
            "synthetic": {"updates_per_length": 1}, "real_train": {"max_updates": 3}}

    def pair(self, **bad):
        return (lambda: (TinyModel(), {}), lambda: (TinyModel(mean=True, **bad), {}))

    def run_case(self, builders=None, build_optimizer=optimizer):
        batch = parity.synthetic_batch(2, 12, 77, "cpu")
        receipt = {"cases": []}
        parity.run_pair_case(self.pair() if builders is None else builders,
            (batch, {key: value.clone() for key, value in batch.items()}), tiny_outputs,
            build_optimizer, self.config(), "cpu", "small_synthetic", receipt)
        return receipt

    def test_natural_construction_dropout_gradients_and_update_pass(self):
        receipt = self.run_case()
        case = receipt["cases"][0]
        self.assertEqual(case["status"], "passed")
        self.assertEqual(len(case["updates"]), 1)
        self.assertTrue(all(row["passed"] for row in case["comparisons"]))
        self.assertTrue(all(row["passed"] for row in case["updates"][0]["comparisons"]))
        self.assertGreater(case["updates"][0]["update"][0]["changed_parameter_count"], 0)

    def test_changed_parameter_rejected_before_forward(self):
        with self.assertRaisesRegex(parity.ParityError, "initial_named_parameters"):
            self.run_case(self.pair(changed=True))

    def test_unexpected_buffer_rejected(self):
        with self.assertRaisesRegex(parity.ParityError, "initial_named_buffers"):
            self.run_case(self.pair(extra=True))

    def test_constructor_rng_drift_rejected(self):
        with self.assertRaisesRegex(parity.ParityError, "constructor_rng"):
            self.run_case(self.pair(rng_drift=True))

    def test_no_optimizer_update_rejected(self):
        class NoUpdate(torch.optim.AdamW):
            def step(self, closure=None):
                return None
        def no_update(model, *, lr, time_head_lr_multiplier):
            return NoUpdate(model.parameters(), lr=lr)
        with self.assertRaisesRegex(parity.ParityError, "changed no parameter"):
            self.run_case(build_optimizer=no_update)

    def test_signed_zero_is_not_bitwise_equal(self):
        comparison = parity.compare_tensors({"x": torch.tensor([0.])}, {"x": torch.tensor([-0.])})
        self.assertFalse(comparison["passed"])
        self.assertTrue(comparison["tensors"]["x"]["numeric_equal"])
        self.assertEqual(comparison["max_abs"], 0.)
        self.assertNotEqual(comparison["left_digest"], comparison["right_digest"])

    def test_shape_dtype_keys_and_finite_guards(self):
        left = {"x": torch.ones(2)}
        for right in ({"x": torch.ones(1)}, {"x": torch.ones(2, dtype=torch.float64)},
            {"y": torch.ones(2)}, {"x": torch.ones(2), "extra": torch.zeros(1)}):
            with self.subTest(right=list(right)):
                self.assertFalse(parity.compare_tensors(left, right)["passed"])
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(parity.ParityError, "Nonfinite"):
                parity.compare_tensors(left, {"x": torch.tensor([value, 1.])})

    def test_none_gradient_asymmetry_rejected(self):
        self.assertFalse(parity.compare_tensors({"p": None}, {"p": torch.zeros(1)})["passed"])

    def test_optimizer_scalar_and_tensor_drift_rejected(self):
        base = {"state": {0: {"step": torch.tensor(1.), "exp_avg": torch.tensor([.25])}},
            "param_groups": [{"params": [0], "lr": .001}]}
        changed = deepcopy(base)
        changed["param_groups"][0]["lr"] = .002
        self.assertFalse(parity.compare_tree(base, changed)["passed"])
        changed = deepcopy(base)
        changed["state"][0]["exp_avg"][0] += .01
        self.assertFalse(parity.compare_tree(base, changed)["passed"])

    def test_rng_restore_and_each_rng_family_drift(self):
        parity.seed_all(42, "cpu")
        before = parity.capture_rng("cpu")
        for consume in (random.random, np.random.random, lambda: torch.rand(1)):
            parity.restore_rng(before)
            consume()
            self.assertFalse(parity.compare_rng(before, parity.capture_rng("cpu"))["passed"])
        parity.restore_rng(before)
        self.assertTrue(parity.compare_rng(before, parity.capture_rng("cpu"))["passed"])

    def test_update_step_counter_guard(self):
        with self.assertRaisesRegex(parity.ParityError, "counter mismatch"):
            parity.assert_updated({"p": torch.zeros(1)}, {"p": torch.ones(1)},
                {"state": {0: {"step": torch.tensor(0.)}}}, step=1)

    def test_input_byte_drift_rejected(self):
        left = parity.synthetic_batch(2, 12, 1, "cpu")
        right = {key: value.clone() for key, value in left.items()}
        right["quantities"][0, -1] += 1.
        self.assertFalse(parity.compare_tensors(left, right)["passed"])


if __name__ == "__main__":
    unittest.main()
