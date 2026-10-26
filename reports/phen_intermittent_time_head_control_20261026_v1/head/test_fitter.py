"""Synthetic tests only; no real observation/checkpoint is loaded."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import fitter as f


def state():
    return {'v_t.weight': torch.zeros(1, 64), 'b_t': torch.tensor([0.]),
            'time_scale_weight.weight': torch.zeros(1, 64), 'w_raw': torch.tensor([-.2])}


def data():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(80, 64)).astype(np.float32)
    y = np.full(80, 5.)
    fit, dev = np.arange(80) < 64, np.arange(80) >= 64
    return x, y, fit, dev


class MathTests(unittest.TestCase):
    def test_native_head_formula_and_parameter_scope(self):
        x = torch.tensor([[.1] * 64, [-.2] * 64], dtype=torch.float64)
        s = state()
        s['v_t.weight'].fill_(.01)
        s['time_scale_weight.weight'].fill_(.02)
        for arm, count in [('original', 0), ('widthonly', 65), ('joint', 130)]:
            head = f.TimeHead(s, arm)
            mu, sigma = head(x)
            torch.testing.assert_close(mu, x @ s['v_t.weight'].double().T[:, 0] + s['b_t'].double())
            expected_sigma = .001 + torch.nn.functional.softplus(x @ s['time_scale_weight.weight'].double().T[:, 0] + s['w_raw'].double())
            torch.testing.assert_close(sigma, expected_sigma)
            self.assertEqual(sum(p.numel() for p in head.parameters()), 130)
            self.assertEqual(sum(p.numel() for p in head.parameters() if p.requires_grad), count)

    def test_recorded_probability_gradcheck_and_first_bin(self):
        target = torch.tensor([1., 2., 9.], dtype=torch.float64)
        mu = torch.tensor([-.2, .3, 1.2], dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor([.6, .5, .3], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda m, s: f.recorded_nll(target, m, s), (mu, sigma)))
        expected = -torch.special.log_ndtr((torch.log(torch.tensor(.5, dtype=torch.float64)) - mu[0]) / sigma[0])
        torch.testing.assert_close(f.recorded_nll(target, mu, sigma)[0], expected)

    def test_extreme_tail_not_clipped_and_gradient_finite(self):
        mu = torch.tensor([-10.], dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor([.01], dtype=torch.float64, requires_grad=True)
        loss = f.recorded_nll(torch.tensor([2.]), mu, sigma)
        self.assertGreater(float(loss.detach()), 1e5)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(mu.grad).all())
        self.assertTrue(torch.isfinite(sigma.grad).all())

    def test_invalid_target_and_parameter_shape_rejected(self):
        with self.assertRaises(ValueError):
            f.recorded_nll(torch.tensor([1.2]), torch.tensor([0.]), torch.tensor([1.]))
        bad = state()
        bad['b_t'] = torch.zeros(2)
        with self.assertRaises(ValueError):
            f.TimeHead(bad, 'joint')

    def test_frozen_probability_source_hash_precedes_import(self):
        f.probability_helper.cache_clear()
        try:
            with patch.object(f, 'file_sha', return_value='not_the_frozen_sha'):
                with self.assertRaisesRegex(ValueError, 'source hash differs'):
                    f.probability_helper()
        finally:
            f.probability_helper.cache_clear()


class SplitTests(unittest.TestCase):
    def test_chronological_per_entity_split_does_not_follow_input_order(self):
        ids = np.array(['a'] * 5 + ['b'] * 10)
        seq = np.r_[np.arange(5), np.arange(10)]
        index = np.arange(15)
        perm = np.random.default_rng(3).permutation(15)
        fit, dev, counts = f.chronological_masks(ids[perm], seq[perm], index[perm])
        self.assertEqual((fit.sum(), dev.sum()), (12, 3))
        for entity in ('a', 'b'):
            choose = ids[perm] == entity
            self.assertLess(seq[perm][fit & choose].max(), seq[perm][dev & choose].min())
        self.assertTrue((counts.fit_last_seq < counts.dev_first_seq).all())

    def test_short_entity_and_ambiguous_chronology_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fewer than two'):
            f.chronological_masks(['a'], [0], [0])
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            f.chronological_masks(['a', 'a'], [1, 1], [0, 1])

    def test_mask_overlap_and_empty_dev_rejected(self):
        x, y, fit, dev = data()
        with self.assertRaises(ValueError):
            f.fit_arm(x, y, np.ones(80, dtype=bool), dev, state(), 'original')
        with self.assertRaises(ValueError):
            f.fit_arm(x, y, np.ones(80, dtype=bool), np.zeros(80, dtype=bool), state(), 'original')


class FitTests(unittest.TestCase):
    def test_original_epoch_zero_no_parameter_or_input_mutation(self):
        x, y, fit, dev = data()
        s, x_before = state(), x.copy()
        before = f.state_sha(s)
        result = f.fit_arm(x, y, fit, dev, s, 'original')
        self.assertEqual(result['receipt']['selected_epoch'], 0)
        self.assertEqual(result['receipt']['epochs_completed'], 0)
        self.assertEqual(result['receipt']['trainable_parameters'], 0)
        self.assertEqual(f.state_sha(s), before)
        np.testing.assert_array_equal(x, x_before)

    def test_width_only_never_changes_location_and_updates_width(self):
        result = f.fit_arm(*data(), state(), 'widthonly', f.FitConfig(maximum_epochs=3, batch_size=16))
        initial = f.validate_head_state(state())
        for key in ('v_t.weight', 'b_t'):
            self.assertTrue(torch.equal(result['last_state'][key], initial[key]))
        self.assertFalse(torch.equal(result['last_state']['w_raw'], initial['w_raw']))
        self.assertLessEqual(result['receipt']['selected_dev_nll'], result['curves'][0]['dev_nll'])

    def test_joint_deterministic_and_no_hidden_gradient(self):
        cfg = f.FitConfig(maximum_epochs=3, batch_size=16)
        first = f.fit_arm(*data(), state(), 'joint', cfg)
        second = f.fit_arm(*data(), state(), 'joint', cfg)
        self.assertEqual(first['receipt']['selected_head_sha256'], second['receipt']['selected_head_sha256'])
        self.assertEqual(first['curves'], second['curves'])
        self.assertFalse(torch.equal(first['last_state']['b_t'], f.validate_head_state(state())['b_t']))
        self.assertFalse(first['receipt']['validation_used_for_selection'])

    def test_epoch_zero_wins_exact_ties_and_patience_stops(self):
        result = f.fit_arm(*data(), state(), 'joint', f.FitConfig(learning_rate=0., maximum_epochs=20, patience=2))
        self.assertEqual(result['receipt']['selected_epoch'], 0)
        self.assertEqual(result['receipt']['epochs_completed'], 2)
        self.assertEqual(result['receipt']['stop_reason'], 'early_stopping')

    def test_min_delta_only_controls_patience_not_absolute_selection(self):
        # evaluate calls: epoch0 dev, epoch0 fit, then each epoch fit and dev.
        values = [1., 2., 2., 1. - 1e-10, 2., 1. - 2e-10]
        with patch.object(f, 'evaluate', side_effect=values):
            result = f.fit_arm(*data(), state(), 'joint',
                               f.FitConfig(learning_rate=0., maximum_epochs=10, patience=2, minimum_improvement=1e-8))
        self.assertEqual(result['receipt']['selected_epoch'], 2)
        self.assertEqual(result['receipt']['epochs_completed'], 2)
        self.assertFalse(result['curves'][-1]['significant_improvement'])
        self.assertTrue(result['curves'][-1]['selected'])

    def test_artifacts_are_separate_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            original_path = Path(directory) / 'source.pt'
            torch.save({'model_state_dict': state(), 'configuration': {'a': 1}}, original_path)
            before = f.file_sha(original_path)
            s = f.extract_head_state(original_path)
            dest = Path(directory) / 'fit'
            result = f.fit_arm(*data(), s, 'joint', f.FitConfig(maximum_epochs=1), output_dir=dest)
            self.assertEqual(before, f.file_sha(original_path))
            self.assertTrue((dest / 'initial_head.pt').is_file())
            self.assertTrue((dest / 'selected_head.pt').is_file())
            self.assertTrue((dest / 'last_head.pt').is_file())
            loaded = torch.load(dest / 'selected_head.pt', weights_only=True)
            self.assertEqual(f.state_sha(loaded), result['receipt']['selected_head_sha256'])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                f.fit_arm(*data(), s, 'joint', output_dir=dest)

    def test_failure_preserves_attempt_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / 'failed'
            x, y, fit, dev = data()
            y[0] = .5
            with self.assertRaises(ValueError):
                f.fit_arm(x, y, fit, dev, state(), 'joint', output_dir=dest)
            self.assertTrue((dest / 'FAILURE.json').is_file())

    def test_prediction_batching_and_event_weighted_evaluation(self):
        x, y, _, _ = data()
        single = f.predict(state(), x, y, batch_size=1)
        batches = f.predict(state(), x, y, batch_size=17)
        for key in single:
            np.testing.assert_allclose(single[key], batches[key], atol=1e-14)
        xx, yy = f.prepare_data(x, y)
        got = f.evaluate(f.TimeHead(state(), 'original'), xx, yy, batch_size=17)
        self.assertAlmostEqual(got, single['time_nll'].mean(), places=14)


if __name__ == '__main__':
    unittest.main()
