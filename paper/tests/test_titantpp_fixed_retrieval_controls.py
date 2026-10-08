"""Fabricated CPU contracts; no research dataset or checkpoint is accessed."""
from __future__ import annotations

from copy import deepcopy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from models.TPPs import CountAwareTitanContextualRetrieval as old
from models.TPPs.CountAwareTitanFixedRetrievalControls import (
    ARMS, GENERIC512, EXTRA_PARAMETERS, CountAwareTitanFixedRetrievalControls,
    fixed_concatenation_scores, metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_fixed_retrieval_controls as runner
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


class FixedRetrievalControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = runner.fabricated_data()
        runner.install_hooks()

    def model(self, arm=GENERIC512, seed=42):
        torch.manual_seed(seed)
        model, meta = runner.build_model(self.data, arm)
        return model.eval(), meta

    def inputs(self):
        dt = torch.tensor([[0., 1., 3., 5., 2., 7.], [0., 4., 2., 6., 1., 8.]])
        quantity = torch.tensor([[0., 2., 4., 8., 3., 1.], [1., 0., 5., 2., 7., 4.]])
        return dt, quantity, torch.ones_like(dt, dtype=torch.bool)

    def payload(self, arm=GENERIC512):
        model, meta = self.model(arm)
        interface = {"time_head": model.time_head_contract(),
            "train_target_mean": self.data["statistics"]["train_log_mean"],
            "train_target_std": self.data["statistics"]["train_log_std"]}
        return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "variant": runner.VARIANT, "interface_meta": interface,
            "resume_identity": {"backbone": arm, "interface_meta": interface,
                "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": role_for_arm(arm)}}}

    def test_exact_parameter_budget_no_temperature_no_extra_output_projection(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model, meta = self.model(arm)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 115167 + EXTRA_PARAMETERS[arm])
                score = {name: p for name, p in model.named_parameters() if name.startswith("lmm.") and name != "lmm.mem"}
                self.assertEqual(sum(p.numel() for p in score.values()), EXTRA_PARAMETERS[arm])
                self.assertEqual(tuple(model.lmm.mem.shape), (1, 64, 64))
                self.assertEqual(model.lmm.topk, 4)
                self.assertEqual(meta["retrieval_fixed_temperature"], 1.)
                self.assertFalse(meta["retrieval_gain_learning"])
                self.assertFalse(meta["retrieval_added_output_projection"])
                if arm == GENERIC512:
                    self.assertEqual(set(score), {"lmm.query_weight"})
                    self.assertEqual(tuple(score["lmm.query_weight"].shape), (8, 64))

    def test_frozen_three_arms_metadata_state_forward_and_rng_exact(self):
        dt, quantity, mask = self.inputs()
        for arm in ARMS[:3]:
            torch.manual_seed(42)
            frozen, frozen_meta = runner.contextual.build_model(self.data, arm)
            frozen_rng = torch.get_rng_state().clone()
            actual, actual_meta = self.model(arm)
            self.assertTrue(torch.equal(frozen_rng, torch.get_rng_state()))
            self.assertEqual(frozen_meta, actual_meta)
            self.assertEqual(set(frozen.state_dict()), set(actual.state_dict()))
            for name, value in frozen.state_dict().items():
                self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)
            for training in (False, True):
                frozen.train(training)
                actual.train(training)
                torch.manual_seed(91)
                expected = target_outputs(frozen, dt, mask, quantity, lambda_log_qty=1.)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(91)
                output = target_outputs(actual, dt, mask, quantity, lambda_log_qty=1.)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(value, output[key]) for key, value in expected.items()))

    def test_all_arms_common_initial_state_rng_and_zero_weighted_output(self):
        states = runner.initial_states(self.data, 42)
        self.assertEqual(set(states), set(ARMS))
        dt, quantity, mask = self.inputs()
        reference, rng = None, None
        for arm in ARMS[1:]:
            model, _ = self.model(arm)
            model.train()
            torch.manual_seed(13)
            result = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
            if reference is None:
                reference, rng = result, torch.get_rng_state().clone()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(value, result[key]) for key, value in reference.items()))

    def test_new_generic_equals_old_generic_when_old_temperature_gate_zero(self):
        torch.manual_seed(42)
        frozen, _ = runner.contextual.build_model(self.data, old.ARMS[4])
        actual, _ = self.model()
        frozen.eval()
        weights = torch.arange(512.).reshape(8, 64) / 5120.
        with torch.no_grad():
            frozen.lmm.query_weight.copy_(weights)
            actual.lmm.query_weight.copy_(weights)
        dt, quantity, mask = self.inputs()
        expected = target_outputs(frozen, dt, mask, quantity, lambda_log_qty=1.)
        result = target_outputs(actual, dt, mask, quantity, lambda_log_qty=1.)
        self.assertTrue(all(torch.equal(value, result[key]) for key, value in expected.items()))

    def test_generic_formula_train_normalization_scale_and_temperature(self):
        model, meta = self.model()
        weights = torch.linspace(-.3, .2, 512).reshape(8, 64)
        with torch.no_grad():
            model.lmm.query_weight.copy_(weights)
        h = torch.arange(1., 65.).reshape(1, 1, 64) / 64.
        features = torch.tensor([[[.2, -.3, .4, .5, .1, .3, .2, -.2]]])
        _, trace = model.lmm.retrieve_context(h, features)
        query = h + .5 * (features @ weights)
        scores = F.normalize(query, dim=-1) @ F.normalize(model.lmm.mem, dim=-1).transpose(-2, -1)
        expected, chosen = torch.topk(scores, 4, dim=-1)
        self.assertTrue(torch.equal(trace["prototype_indices"], chosen))
        self.assertTrue(torch.equal(trace["topk_similarity"], expected))
        self.assertTrue(torch.equal(trace["weights"], expected.softmax(-1)))
        self.assertTrue(torch.equal(trace["temperature"], torch.ones(1, 1)))
        self.assertTrue(bool((scores.abs() <= 1.000001).all()))
        fixed, fixed_meta = self.model(ARMS[2])
        self.assertEqual(meta["retrieval_train_statistics"], fixed_meta["retrieval_train_statistics"])
        dt, quantity, mask = self.inputs()
        self.assertTrue(torch.equal(model.observed_features(dt, quantity, mask), fixed.observed_features(dt, quantity, mask)))

    def test_fixed_concat_equivalence_finite_nonzero_and_zero_domain_explicit(self):
        torch.manual_seed(7)
        h = torch.randn(2, 5, 64, dtype=torch.float64)
        phi = torch.randn(2, 5, 8, dtype=torch.float64)
        wt, wq = torch.randn(4, 64, dtype=torch.float64), torch.randn(4, 64, dtype=torch.float64)
        memory = torch.randn(2, 64, 64, dtype=torch.float64)
        audit = fixed_concatenation_scores(h, phi, wt, wq, memory)
        self.assertTrue(bool(audit["concat_cosine_equivalent_domain"].all()))
        self.assertTrue(torch.allclose(audit["fixed_scores"], audit["concat_cosine_scores"], atol=1e-14, rtol=1e-14))
        self.assertTrue(torch.allclose(audit["fixed_scores"], audit["concat_dot_half_scores"], atol=1e-14, rtol=1e-14))
        zero = fixed_concatenation_scores(torch.zeros(1, 1, 64, dtype=torch.float64),
            torch.ones(1, 1, 8, dtype=torch.float64), torch.zeros(4, 64, dtype=torch.float64),
            torch.ones(4, 64, dtype=torch.float64), memory[:1])
        self.assertFalse(bool(zero["concat_cosine_equivalent_domain"].any()))
        self.assertTrue(torch.allclose(zero["fixed_scores"], zero["concat_dot_half_scores"], atol=1e-14, rtol=1e-14))
        self.assertFalse(torch.allclose(zero["fixed_scores"], zero["concat_cosine_scores"], atol=1e-14, rtol=1e-14))

    def test_same_subset_aggregation_diagnostic_has_no_extra_fit_or_reselection(self):
        model, _ = self.model(ARMS[2])
        dt, quantity, mask = self.inputs()
        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            diag = runner.retrieval_diagnostics(model, dt, quantity, mask)
            trace = model.retrieval_trace(dt, quantity, mask)
        self.assertTrue(torch.equal(diag["prototype_indices"], trace["prototype_indices"]))
        self.assertTrue(torch.equal(diag["topk_similarity"], trace["topk_similarity"]))
        expanded = model.lmm.mem.expand(dt.size(0), -1, -1).unsqueeze(1).expand(-1, dt.size(1), -1, -1)
        selected = expanded.gather(2, diag["prototype_indices"].unsqueeze(-1).expand(-1, -1, -1, 64))
        self.assertTrue(torch.equal(diag["mean_on_same_subset"], selected.mean(2)))
        self.assertTrue(torch.equal(diag["softmax_on_same_subset"], (selected * trace["weights"].unsqueeze(-1)).sum(2)))
        self.assertTrue(diag["diagnostic_only_no_checkpoint_reselection"])
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in state_before.items()))
        self.assertEqual(diag["identifiability"]["scope"], "split_vs_joint_query_normalization_inductive_bias_not_semantic_novelty_proof")

    def test_every_fixed_and_generic512_added_element_is_active_with_finite_updates(self):
        result = runner.synthetic_check(arms=ARMS[2:])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["active_score_gradient_parameter_count"], {arm: 512 for arm in ARMS[2:]})
        self.assertFalse(result["real_data_loaded"])
        self.assertFalse(result["held_out_test_evaluated"])

    def test_nondegenerate_new_query_is_causal_future_withheld_padding_and_reset(self):
        model, _ = self.model()
        with torch.no_grad():
            model.lmm.query_weight.copy_(torch.linspace(-.2, .3, 512).reshape(8, 64))
        dt, quantity, mask = self.inputs()
        write = mask.clone()
        write[:, -1] = False
        with torch.no_grad():
            reference = model.encode_task_states(dt, quantity, mask, memory_write_mask=write)[0]
            future_dt, future_qty = dt.clone(), quantity.clone()
            future_dt[:, 4], future_qty[:, 4] = 1000., 99999.
            future = model.encode_task_states(future_dt, future_qty, mask, memory_write_mask=write)[0]
            self.assertTrue(torch.equal(reference[:, :4], future[:, :4]))
            future_dt[:, -1], future_qty[:, -1] = float("nan"), float("inf")
            self.assertTrue(torch.equal(future, model.encode_task_states(future_dt, future_qty, mask, memory_write_mask=write)[0]))
            padded = mask.clone()
            padded[:, 2] = False
            keep = [0, 1, 3, 4, 5]
            self.assertTrue(torch.equal(model.observed_features(dt, quantity, padded, memory_write_mask=write)[:, keep],
                model.observed_features(dt[:, keep], quantity[:, keep], mask[:, keep], memory_write_mask=write[:, keep])))
            reset = write.clone()
            reset[:, 2] = False
            features = model.observed_features(dt, quantity, mask, memory_write_mask=reset)
            self.assertTrue(bool((features[:, 2] == 0).all()))
            self.assertTrue(bool((features[:, 3, [2, 3, 6, 7]] == 0).all()))

    def test_checkpoint_relabel_shape_finiteness_normalizer_and_temperature_rejected(self):
        payload = self.payload()
        self.assertTrue(validate_checkpoint(payload, GENERIC512))
        from models.TPPs.CountAwareFactory import validate_checkpoint_route
        validate_checkpoint_route(payload, GENERIC512)
        for wrong in (*ARMS[:3], old.ARMS[4], runner.BASELINE):
            with self.assertRaises(ValueError):
                validate_checkpoint(payload, wrong)
        changes = []
        changed = deepcopy(payload)
        changed["encoder_config"]["retrieval_fixed_temperature"] = 2.
        changes.append(changed)
        changed = deepcopy(payload)
        changed["encoder_config"]["retrieval_train_statistics"]["quantity_log_mean"] += 1.
        changes.append(changed)
        changed = deepcopy(payload)
        changed["model_state_dict"]["lmm.context_gate_weight"] = torch.zeros(8)
        changes.append(changed)
        changed = deepcopy(payload)
        changed["model_state_dict"]["lmm.query_weight"] = torch.zeros(7, 64)
        changes.append(changed)
        changed = deepcopy(payload)
        changed["model_state_dict"]["lmm.query_weight"][0, 0] = float("nan")
        changes.append(changed)
        changed = deepcopy(payload)
        changed["evaluation_scope"] = "test"
        changes.append(changed)
        for changed in changes:
            with self.assertRaises(ValueError):
                validate_checkpoint(changed, GENERIC512)
        model, _ = self.model()
        model.load_state_dict(payload["model_state_dict"])
        changed = deepcopy(payload["model_state_dict"])
        changed["contextual_retrieval_identity"][0] ^= 1
        with self.assertRaises(ValueError):
            model.load_state_dict(changed)

    def test_existing_frozen_checkpoint_stays_compatible(self):
        for arm in ARMS[:3]:
            payload = self.payload(arm)
            self.assertTrue(validate_checkpoint(payload, arm))
            self.assertTrue(old.validate_checkpoint(payload, arm))

    def test_training_args_and_interface_preserve_budget_and_explain_normalization(self):
        contract = {"training": {"max_epochs": 300}}
        with patch.object(runner.base, "training_args", return_value=SimpleNamespace()) as stub:
            args = runner.training_args(contract, self.data, "fabricated", GENERIC512)
        self.assertEqual(stub.call_args.args[0]["model_role"], role_for_arm(GENERIC512))
        self.assertEqual((args.epochs, args.min_epochs, args.early_stopping_patience), (300, 40, 40))
        with patch.object(runner.base, "time_interface", return_value={}):
            interface = runner.time_interface(self.data, None, contract, arm=GENERIC512)
        design = interface["backbone_design"]
        self.assertEqual(design["identifiability"]["common_temperature"], 1.)
        self.assertEqual(design["model_metadata"]["retrieval_extra_parameter_count"], 512)
        self.assertEqual(design["identifiability"]["generic512_query_normalization"], "one_unit_joint_query")

    def test_streaming_replay_returns_same_epoch_six_metrics_with_fabricated_loader(self):
        from pathlib import Path
        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
        payload = self.payload()
        payload["model_state_sha256"] = canonical_state_dict_sha256(payload["model_state_dict"])
        data = deepcopy(self.data)
        dt, quantity, mask = self.inputs()
        model, _ = self.model()
        with torch.no_grad():
            expected = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
        target_count = expected["true_qty"].numel()
        data.update(quantity_boundaries_all_train_rows=[1., 2., 3., 3.], history_boundaries=[2., 4.],
            inherited_data_identity={"populations": {"validation": {"target_count": target_count}}})
        data["loader"].update(batch_size=128, lookback_weeks=168)
        budget_calls = []
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fabricated_selected.pt"
            torch.save(payload, checkpoint)
            with patch("paper.scripts.run_taxi_quantity_interface_ablation.make_loader", return_value=[(None, dt, mask, None, quantity)]) as loader:
                result = runner.replay_checkpoint(checkpoint, data, None, lambda: budget_calls.append(True), device="cpu")
        self.assertEqual(loader.call_args.kwargs["target_split"], "validation")
        self.assertEqual(result["count"], target_count)
        self.assertEqual(result["evaluation_scope"], "validation_only")
        self.assertFalse(result["held_out_test_evaluated"])
        self.assertGreaterEqual(len(budget_calls), 2)
        for scope in (result, result["tail"]):
            self.assertGreater(scope["count"], 0)
            for metric in ("qty_rmse", "qty_mae", "time_nll"):
                self.assertTrue(torch.isfinite(torch.tensor(scope[metric])))
        self.assertEqual(result["tail"], {key: value for key, value in result["quantity_cells"][-1].items() if key != "bin"})

    def test_runner_rejects_wrong_role_and_alternative_arms(self):
        from paper.scripts.count_aware_tpp_backbone import observed_time, constants
        args = SimpleNamespace(model_role=role_for_arm(GENERIC512))
        with self.assertRaises(ValueError):
            observed_time.validate_launch(args, (ARMS[2],), (runner.VARIANT,))
        with self.assertRaises(ValueError):
            constants.validate_model_role_contract(model_role=role_for_arm(GENERIC512), backbones=(ARMS[2],))
        with self.assertRaises(ValueError):
            runner.initial_states(self.data, 42, arms=(old.ARMS[4],))
        with self.assertRaises(ValueError):
            runner.initial_states(self.data, 42, arms=(GENERIC512, GENERIC512))


if __name__ == "__main__":
    unittest.main()
