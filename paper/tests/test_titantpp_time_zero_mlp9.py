"""Quantity-only control: fabricated-only gradient/update and lifecycle checks."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from paper.scripts import run_titantpp_time_zero_mlp9 as worker


class QuantityOnlyObjectiveTests(unittest.TestCase):
    def test_joint_is_quantity_object_and_time_gradient_is_none(self):
        time = torch.nn.Parameter(torch.tensor([.2, .4]))
        qty = torch.nn.Parameter(torch.tensor([.3, .6]))
        initial_time = time.detach().clone()
        def original(*args, **kwargs):
            t, q = time.square(), qty.square()
            return {"joint_loss": t+q, "time_loss": t, "quantity_train_loss": q,
                "log_qty_loss": q, "unchanged": "preserved"}
        value = worker.make_weighted_target_outputs(original, 0.)(None, lambda_log_qty=1.)
        self.assertIs(value["joint_loss"], value["quantity_train_loss"])
        self.assertEqual(value["unchanged"], "preserved")
        gt, gq = torch.autograd.grad(value["joint_loss"].mean(), (time, qty), allow_unused=True)
        self.assertIsNone(gt)
        torch.testing.assert_close(gq, qty.detach())
        optimizer = torch.optim.AdamW([time, qty], lr=.01, weight_decay=.01)
        value = worker.make_weighted_target_outputs(original, 0.)(None, lambda_log_qty=1.)
        value["joint_loss"].mean().backward(); optimizer.step()
        self.assertTrue(torch.equal(initial_time, time))
        self.assertNotIn(time, optimizer.state)
        self.assertIsNone(time.grad)

    def test_nonfinite_unused_time_is_preserved_not_sanitized(self):
        q = torch.tensor([2.], requires_grad=True)
        bad_time = torch.tensor([float("nan")])
        original = {"joint_loss": bad_time+q, "time_loss": bad_time,
            "quantity_train_loss": q, "log_qty_loss": q}
        result = worker.make_weighted_target_outputs(lambda *a, **k: original, 0.)(None, lambda_log_qty=1.)
        self.assertIs(result["time_loss"], bad_time)
        self.assertTrue(bool(torch.isnan(result["time_loss"]).all()))
        self.assertIs(result["joint_loss"], q)
        self.assertNotIn("weighted_time_loss", result)

    def test_nonzero_or_foreign_quantity_objective_rejected(self):
        for alpha in (.1, 1., 10., float("nan"), True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                worker.weight_identity(alpha)
        value = {"joint_loss": torch.ones(1), "time_loss": torch.ones(1),
            "quantity_train_loss": torch.ones(1)*2, "log_qty_loss": torch.ones(1)}
        with self.assertRaises(ValueError):
            worker.make_weighted_target_outputs(lambda *a, **k: value, 0.)(None, lambda_log_qty=1.)

    def test_real_mlp16_two_step_quantity_oracle_exact_and_time_head_fixed(self):
        result = worker.quantity_only_synthetic("cpu")
        self.assertTrue(result["quantity_only_gradient_and_adamw_oracle_exact"])
        self.assertTrue(result["shared_nonzero_gradient_observed"])
        self.assertTrue(result["quantity_head_updated"])
        proof = result["time_head_no_training_verification"]
        worker.verify_time_head_proof(proof)
        self.assertEqual(proof["optimizer_steps_checked"], 2)
        self.assertEqual(proof["initial_time_head_sha256"], proof["final_time_head_sha256"])
        self.assertTrue(proof["optimizer_state_absent"])

    def test_original_fresh_initial_tensors_rng_and_metadata_are_identical(self):
        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
        torch.manual_seed(52)
        model_a, meta_a = worker.sealed_weight.build_model(worker.fabricated_data(), worker.ARMS[0])
        rng_a = torch.get_rng_state().clone()
        with worker.runtime_hooks():
            torch.manual_seed(52)
            model_b, meta_b = worker.build_model(worker.fabricated_data(), worker.ARMS[0])
            rng_b = torch.get_rng_state().clone()
        self.assertEqual(canonical_state_dict_sha256(model_a.state_dict()), canonical_state_dict_sha256(model_b.state_dict()))
        self.assertEqual(meta_a, meta_b)
        self.assertTrue(torch.equal(rng_a, rng_b))


class QuantityOnlyLifecycleTests(unittest.TestCase):
    def test_private_adapter_does_not_change_original_worker_globals(self):
        from paper.scripts.count_aware_tpp_backbone import core, training
        worker.inherited.install_hooks()
        before = (core.target_outputs, training.target_outputs, training.build_model, training.build_optimizer,
            worker.sealed_weight.SCHEMA, worker.sealed_weight.TIME_ALPHAS, worker.sealed_weight.make_weighted_target_outputs)
        old_sha = worker.sha_file(worker._SEALED_PATH)
        with self.assertRaises(RuntimeError):
            with worker.runtime_hooks():
                with self.assertRaises(ValueError):
                    with worker.runtime_hooks():
                        pass
                raise RuntimeError("fabricated failure")
        self.assertEqual(before, (core.target_outputs, training.target_outputs, training.build_model, training.build_optimizer,
            worker.sealed_weight.SCHEMA, worker.sealed_weight.TIME_ALPHAS, worker.sealed_weight.make_weighted_target_outputs))
        self.assertEqual(old_sha, worker.sha_file(worker._SEALED_PATH))
        self.assertIsNone(worker._TRACKER)

    def test_alpha_and_quantity_only_metadata_tamper_rejected(self):
        identity = worker.weight_identity(0.)
        interface = {"time_weight_identity": identity}
        payload = {"time_alpha": 0., "quantity_only": True, "time_head_trained": False,
            "time_weight_identity": identity, "interface_meta": interface,
            "resume_identity": {"interface_meta": copy.deepcopy(interface),
                "time_weight_identity": identity, "arguments": {"time_alpha": 0.}}}
        worker.validate_weight_binding(payload, 0.)
        for key, value in (("time_alpha", .1), ("quantity_only", False), ("time_head_trained", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_weight_binding({**payload, key: value}, 0.)

    def test_actual_proof_rejects_missing_or_changed_head_state(self):
        result = worker.quantity_only_synthetic("cpu")
        proof = result["time_head_no_training_verification"]
        for key, value in (("optimizer_steps_checked", 0), ("optimizer_state_absent", False),
            ("all_time_head_gradients_none", False), ("final_time_head_sha256", "0"*64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.verify_time_head_proof({**proof,key:value})
        with self.assertRaises(ValueError):
            worker.verify_time_head_proof(None)
        earlier = {**proof, "optimizer_steps_checked": 1}
        worker.verify_checkpoint_fit_proof(earlier, proof)
        with self.assertRaises(ValueError):
            worker.verify_checkpoint_fit_proof(proof, earlier)
        with self.assertRaises(ValueError):
            worker.verify_checkpoint_fit_proof({**earlier, "time_head_initialization": "foreign policy"}, proof)

    def test_synthetic_admission_and_validation_only_scope(self):
        result = worker.synthetic_check(device="cpu")
        self.assertEqual(result["schema"], worker.SCHEMA)
        self.assertEqual(result["time_alphas"], [0.])
        self.assertEqual(set(result["cuda_memory_peaks"]), set(worker.ARMS))
        self.assertTrue(result["quantity_only"])
        self.assertFalse(result["time_head_trained"])
        self.assertTrue(result["time_nll_is_diagnostic"])
        self.assertFalse(result["held_out_test_evaluated"])
        with self.assertRaises(ValueError):
            worker.run_stage({}, "5090", "unused", "test")

    def test_full_fabricated_training_checkpoint_validation_and_actual_head_proof(self):
        # Reuse the frozen lifecycle fixture without changing its file. It
        # already exercises strict state routing, first-minimum checkpoint,
        # prediction replay, no-repeat training, and wrong-alpha rejection.
        from paper.tests import test_titantpp_time_weight_mlp18 as fixture
        from paper.scripts.count_aware_tpp_backbone import training
        captured = []
        original_run = worker.run_stage
        original_evaluate = training.evaluate
        evaluation_calls, first_actual_rmse = 0, None
        def force_earlier_selection(*args, **kwargs):
            # A selector-only fabricated fixture: force the second epoch's
            # monitor above epoch1. The final selected-state replay still uses
            # the original real numerical evaluation, with no override.
            nonlocal evaluation_calls, first_actual_rmse
            value = original_evaluate(*args, **kwargs)
            evaluation_calls += 1
            if evaluation_calls == 1:
                first_actual_rmse = value["qty_rmse"]
            elif evaluation_calls == 2:
                value = {**value, "qty_rmse": first_actual_rmse + 1.}
            return value
        def run(*args, **kwargs):
            value = original_run(*args, **kwargs)
            captured.append(copy.deepcopy(value))
            return value
        with patch.object(fixture, "worker", worker), patch.object(worker, "run_stage", side_effect=run), \
            patch.object(training, "evaluate", side_effect=force_earlier_selection):
            fixture.BindingAndIsolationTests.test_fabricated_full_train_selected_checkpoint_validation_only(self)
        train = next(r for r in captured if r["stage"] == "train")
        validation = next(r for r in captured if r["stage"] == "validation")
        # The controlled monitor deliberately selects the first of two epochs.
        self.assertEqual(train["selected_epoch"], 1)
        self.assertEqual(train["summary"]["completed_epochs"], 2)
        self.assertLess(train["selected_epoch"], train["summary"]["completed_epochs"])
        for receipt in (train, validation):
            self.assertTrue(receipt["quantity_only"])
            self.assertFalse(receipt["time_head_trained"])
            self.assertTrue(receipt["time_nll_is_diagnostic"])
            self.assertEqual(receipt["schema"], worker.SCHEMA)
            self.assertEqual(receipt["time_alpha"], 0.)
            proof = worker.verify_time_head_proof(receipt["time_head_no_training_verification"])
            self.assertEqual(proof["optimizer_steps_checked"], 2)
            self.assertFalse(proof["step_count_is_selection_epoch"])
        self.assertEqual(train["time_head_no_training_verification"], validation["time_head_no_training_verification"])
        self.assertEqual(train["time_head_no_training_verification"], train["summary"]["time_head_no_training_verification"])


if __name__ == "__main__":
    unittest.main()
