"""Fabricated CPU contracts for contextual search; no dataset/checkpoint reads."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

import torch

from models.TPPs.CountAwareTitanContextualRetrieval import (
    ARMS, BASELINE, EXTRA_PARAMETERS, CountAwareTitanContextualRetrieval,
    metadata, role_for_arm, validate_checkpoint,
)
from models.TPPs.CountAwareTitanCNNGRU import CountAwareTitanCNNGRU
from paper.scripts import run_titantpp_contextual_retrieval as runner
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


class ContextualRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = runner.fabricated_data()
        runner.install_hooks()

    def model(self, arm=ARMS[3], seed=42, data=None):
        torch.manual_seed(seed)
        model, meta = runner.build_model(self.data if data is None else data, arm)
        return model.eval(), meta

    def inputs(self):
        dt = torch.tensor([[0., 1., 3., 5., 2., 7.], [0., 4., 2., 6., 1., 8.]])
        qty = torch.tensor([[0., 2., 4., 8., 3., 1.], [1., 0., 5., 2., 7., 4.]])
        mask = torch.ones_like(dt, dtype=torch.bool)
        return dt, qty, mask

    def payload(self, arm):
        model, meta = self.model(arm)
        interface = {"time_head": model.time_head_contract(),
            "train_target_mean": self.data["statistics"]["train_log_mean"],
            "train_target_std": self.data["statistics"]["train_log_std"]}
        return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "variant": runner.VARIANT, "interface_meta": interface,
            "resume_identity": {"backbone": arm, "interface_meta": interface,
                "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": role_for_arm(arm)}}}

    def test_bank_budget_and_no_gain(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model, meta = self.model(arm)
                self.assertEqual(tuple(model.lmm.mem.shape), (1, 64, 64))
                self.assertEqual(model.lmm.topk, 4)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 115167 + EXTRA_PARAMETERS[arm])
                self.assertEqual(sum(p.numel() for name, p in model.named_parameters()
                    if name.startswith("lmm.") and name != "lmm.mem"), EXTRA_PARAMETERS[arm])
                self.assertFalse(meta["retrieval_gain_learning"])
                self.assertFalse(meta["retrieval_added_output_projection"])
                self.assertEqual(model.candidate_arm, BASELINE)

    def test_mean_exact_parent_state_and_full_forward(self):
        model, _ = self.model(ARMS[0])
        torch.manual_seed(42)
        parent, _ = runner.cnn.build_model(self.data, BASELINE)
        parent.eval()
        self.assertIsInstance(parent, CountAwareTitanCNNGRU)
        self.assertEqual(set(model.state_dict()) - set(parent.state_dict()), {"contextual_retrieval_identity"})
        for name, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        dt, qty, mask = self.inputs()
        for train in (False, True):
            model.train(train)
            parent.train(train)
            torch.manual_seed(91)
            expected = target_outputs(parent, dt, mask, qty, lambda_log_qty=1.)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(91)
            actual = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_all_common_tensors_postconstruction_and_dropout_rng(self):
        digest = runner.initial_states(self.data, 42)
        self.assertEqual(set(digest), set(ARMS))
        self.assertEqual(len(set(digest.values())), 5)
        dt, qty, mask = self.inputs()
        reference_rng, reference_output = None, None
        for arm in ARMS:
            model, _ = self.model(arm)
            model.train()
            torch.manual_seed(314)
            output = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            final_rng = torch.get_rng_state().clone()
            if reference_rng is None:
                reference_rng = final_rng
            self.assertTrue(torch.equal(reference_rng, final_rng))
            if arm != ARMS[0]:
                if reference_output is None:
                    reference_output = output
                for key in output:
                    self.assertTrue(torch.equal(reference_output[key], output[key]), key)

    def test_point_normalization_then_summary_and_missing_gap(self):
        data = deepcopy(self.data)
        data["model"].update(time_scale=2., time_initial_location=1., time_initial_scale=2.)
        data["statistics"].update(train_log_mean=1., train_log_std=2.)
        model, _ = self.model(data=data)
        dt = torch.tensor([[0., 2. * math.exp(1.), 2. * math.exp(3.)]])
        qty = torch.tensor([[0., math.expm1(1.), math.expm1(3.)]])
        mask = torch.ones_like(dt, dtype=torch.bool)
        features = model.observed_features(dt, qty, mask)
        self.assertTrue(torch.equal(features[0, 0, :4], torch.zeros(4)))
        self.assertTrue(torch.allclose(features[0, 0, 4:], torch.tensor([-.5, -.5, 0., 0.])))
        self.assertTrue(torch.allclose(features[0, 1, :4], torch.zeros(4), atol=1e-7))
        self.assertTrue(torch.allclose(features[0, 2, :4], torch.tensor([1., .5, .5, 1.]), atol=1e-6))
        self.assertTrue(torch.allclose(features[0, 2, 4:], torch.tensor([1., 1./6.,
            math.sqrt(((-.5 - 1./6.) ** 2 + (0. - 1./6.) ** 2 + (1. - 1./6.) ** 2) / 3.), 1.]), atol=1e-6))

    def test_window_32_skips_padding_and_excludes_older_observation(self):
        model, _ = self.model()
        values = torch.arange(1., 41.).unsqueeze(0)
        mask = torch.ones_like(values, dtype=torch.bool)
        features = model.observed_features(values, values, mask)
        dt2, qty2 = values.clone(), values.clone()
        dt2[:, :8], qty2[:, :8] = 10000., 99999.
        changed = model.observed_features(dt2, qty2, mask)
        self.assertTrue(torch.equal(features[:, -1], changed[:, -1]))
        dt2[:, 8] = 10000.
        self.assertFalse(torch.equal(features[:, -1], model.observed_features(dt2, qty2, mask)[:, -1]))

    def test_fabricated_causality_mask_reset_contract(self):
        result = runner.check_synthetic_causality()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["real_data_loaded"])

    def test_observed_nonfinite_rejected_and_unobserved_nonfinite_ignored(self):
        model, _ = self.model()
        dt, qty, mask = self.inputs()
        dt[0, 1] = float("nan")
        with self.assertRaises(ValueError):
            model.observed_features(dt, qty, mask)
        write = mask.clone()
        write[0, 1] = False
        self.assertTrue(bool(torch.isfinite(model.observed_features(dt, qty, mask, memory_write_mask=write)).all()))
        with self.assertRaises(ValueError):
            model.observed_features(torch.ones(2, 3), torch.ones(2, 2), torch.ones(2, 3, dtype=torch.bool))

    def test_zero_statistic_constant_and_unavailable_summaries(self):
        data = deepcopy(self.data)
        model, _ = self.model(data=data)
        # The parent constructor requires positive quantity Train std. This
        # isolated fallback exercise does not relax that frozen constructor.
        model.retrieval_statistics = {**model.retrieval_statistics, "quantity_log_std": 0.}
        dt, qty, mask = self.inputs()
        features = model.observed_features(dt, qty, mask)
        self.assertTrue(bool((features[..., 4:] == 0).all()))
        constant = model.observed_features(torch.ones(2, 4), torch.ones(2, 4), mask[:, :4])
        self.assertTrue(bool((constant[..., [2, 3, 6, 7]] == 0).all()))
        empty = model.observed_features(torch.empty(2, 0), torch.empty(2, 0), torch.empty(2, 0, dtype=torch.bool))
        self.assertEqual(tuple(empty.shape), (2, 0, 8))

    def test_native_topk_ties_uniform_weights_equal_mean(self):
        residuals, indices = [], []
        encoded, features = torch.zeros(2, 3, 64), torch.zeros(2, 3, 8)
        for arm in ARMS:
            model, _ = self.model(arm)
            if arm == ARMS[0]:
                residual, trace = model.lmm.retrieve(encoded)
            else:
                residual, trace = model.lmm.retrieve_context(encoded, features)
                self.assertTrue(torch.equal(trace["weights"], torch.full((2, 3, 4), .25)))
            residuals.append(residual)
            indices.append(trace["prototype_indices"])
        for residual, chosen in zip(residuals, indices):
            self.assertTrue(torch.equal(residuals[0], residual))
            self.assertTrue(torch.equal(indices[0], chosen))

    def test_score_formulas_bounded_gate_and_temperature(self):
        h = torch.arange(1., 65.).reshape(1, 1, 64) / 64.
        phi = torch.tensor([[[.2, -.3, .4, .5, .1, .3, .2, -.2]]])
        for arm in ARMS[2:]:
            model, _ = self.model(arm)
            with torch.no_grad():
                for name, parameter in model.lmm.named_parameters():
                    if name != "mem":
                        parameter.fill_(.1)
            residual, trace = model.lmm.retrieve_context(h, phi)
            keys = torch.nn.functional.normalize(model.lmm.mem, dim=-1).transpose(-2, -1)
            if arm == ARMS[4]:
                q = h + .5 * (phi @ model.lmm.query_weight)
                scores = torch.nn.functional.normalize(q, dim=-1) @ keys
                tau = torch.exp(math.log(2.) * torch.tanh(phi @ model.lmm.context_gate_weight + model.lmm.context_gate_bias))
            else:
                tq = h + phi[..., :4] @ model.lmm.time_query_weight
                qq = h + phi[..., 4:] @ model.lmm.quantity_query_weight
                alpha = (torch.sigmoid(phi @ model.lmm.context_gate_weight + model.lmm.context_gate_bias)
                    if arm == ARMS[3] else torch.full((1, 1), .5))
                scores = alpha[..., None] * (torch.nn.functional.normalize(tq, dim=-1) @ keys) + (
                    1. - alpha[..., None]) * (torch.nn.functional.normalize(qq, dim=-1) @ keys)
                tau = torch.ones(1, 1)
            expected_scores, expected_indices = torch.topk(scores, 4, dim=-1)
            self.assertTrue(torch.equal(trace["prototype_indices"], expected_indices))
            self.assertTrue(torch.equal(trace["topk_similarity"], expected_scores))
            self.assertTrue(torch.equal(trace["weights"], (expected_scores / tau[..., None]).softmax(-1)))
            self.assertTrue(bool(torch.isfinite(residual).all()))
            if arm in ARMS[3:]:
                for bias in (-100., 100.):
                    with torch.no_grad():
                        model.lmm.context_gate_bias.fill_(bias)
                    _, bound = model.lmm.retrieve_context(h, phi)
                    self.assertTrue(bool(((bound["temperature"] >= .5) & (bound["temperature"] <= 2.)).all()))
                    self.assertTrue(bool(((bound["alpha"] >= 0) & (bound["alpha"] <= 1)).all()))

    def test_every_added_parameter_active_and_gate_learns_fabricated_only(self):
        result = runner.synthetic_check()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["active_score_gradient_parameter_count"], EXTRA_PARAMETERS)
        self.assertEqual(result["parameter_count"][ARMS[3]], result["parameter_count"][ARMS[4]])
        self.assertEqual(len(result["gate_learning"]), 2)
        self.assertFalse(result["real_data_loaded"])
        self.assertFalse(result["held_out_test_evaluated"])

    def test_checkpoint_and_model_state_reject_arm_statistic_and_shape_tamper(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                payload = self.payload(arm)
                self.assertTrue(validate_checkpoint(payload, arm))
                for expected in (BASELINE, "titantpp", next(other for other in ARMS if other != arm)):
                    with self.assertRaises(ValueError):
                        validate_checkpoint(payload, expected)
                for key, value in (("evaluation_scope", "test"), ("held_out_test_evaluated", True),
                        ("variant", "wrong_objective")):
                    changed = {**payload, key: value}
                    with self.assertRaises(ValueError):
                        validate_checkpoint(changed, arm)
                changed = deepcopy(payload)
                changed["encoder_config"]["retrieval_train_statistics"]["quantity_log_mean"] += 1.
                with self.assertRaises(ValueError):
                    validate_checkpoint(changed, arm)
                changed = deepcopy(payload)
                changed["model_state_dict"]["lmm.mem"] = torch.zeros(1, 65, 64)
                with self.assertRaises(ValueError):
                    validate_checkpoint(changed, arm)
                model, _ = self.model(arm)
                with self.assertRaises((ValueError, RuntimeError)):
                    model.load_state_dict(changed["model_state_dict"])
                changed = deepcopy(payload["model_state_dict"])
                changed["contextual_retrieval_identity"][0] ^= 1
                with self.assertRaises(ValueError):
                    model.load_state_dict(changed)
                model.load_state_dict(payload["model_state_dict"])

    def test_process_local_factory_training_alias_and_explicit_roles(self):
        from models.TPPs import CountAwareFactory as factory
        from paper.scripts.count_aware_tpp_backbone import training, constants
        self.assertIs(factory.build_count_aware_model, training.build_model)
        self.assertIs(factory.validate_checkpoint_route, training.validate_checkpoint_route)
        self.assertTrue(set(ARMS).issubset(constants.SUPPORTED_BACKBONES))
        self.assertTrue(set(role_for_arm(arm) for arm in ARMS).issubset(training.OBSERVED_TIME_ROLES))
        for arm in ARMS:
            self.assertIsInstance(self.model(arm)[0], CountAwareTitanContextualRetrieval)
        with self.assertRaises(ValueError):
            metadata(arm="titantpp_unknown")
        with self.assertRaises(ValueError):
            metadata(32, arm=ARMS[0])
        with self.assertRaises(ValueError):
            runner.initial_states(self.data, 42, arms=(ARMS[0], ARMS[0]))


if __name__ == "__main__":
    unittest.main()
