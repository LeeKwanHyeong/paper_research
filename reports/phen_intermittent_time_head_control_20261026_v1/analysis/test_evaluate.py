"""Synthetic evaluator tests; selected Validation predictions are never loaded."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import evaluate_selected as e


def synthetic_metadata():
    n = 8
    return pd.DataFrame({'target_index': np.arange(n), 'entity_id': ['a', 'b'] * 4,
        'target_seq': np.arange(n), 'raw_quantity': [187, 188, 5, 8, 9, 10, 11, 12],
        'target_gap': [1, 2, 3, 4, 1, 2, 3, 4], 'history_length': [1, 63, 64, 127, 128, 192, 193, 250],
        'condition_id': np.arange(n), 'two_route_feasibility': [True] * 6 + [False] * 2,
        'head_role': ['fit'] * 4 + ['dev'] * 4})


def selection_fixture(arm='joint'):
    spec = {'fit_config': {'minimum_improvement': 1e-8}, 'arms': {arm: {'trainable_parameters': 130}},
            'split': {'fit_rows': 10, 'dev_rows': 4}}
    curves = [{'epoch': 0, 'dev_nll': 1.}, {'epoch': 1, 'dev_nll': 1. - 1e-10},
              {'epoch': 2, 'dev_nll': 1. - 1e-10}]
    receipt = {'selected_epoch': 1, 'selected_dev_nll': 1. - 1e-10, 'config': spec['fit_config'],
        'epochs_completed': 2, 'trainable_parameters': 130, 'validation_used_for_selection': False,
        'input_hidden_unchanged': True, 'input_checkpoint_unchanged': True, 'fit_events': 10, 'dev_events': 4}
    return curves, receipt, spec


class SelectionTests(unittest.TestCase):
    def test_absolute_minimum_and_earliest_tie(self):
        curves, receipt, spec = selection_fixture()
        self.assertEqual(e.selection_check(curves, receipt, 'joint', spec), 1)
        for epoch in [0, 2]:
            bad = {**receipt, 'selected_epoch': epoch}
            with self.assertRaisesRegex(ValueError, 'absolute internal-dev minimum'):
                e.selection_check(curves, bad, 'joint', spec)

    def test_validation_selection_and_missing_epoch_zero_rejected(self):
        curves, receipt, spec = selection_fixture()
        with self.assertRaisesRegex(ValueError, 'Validation selected'):
            e.selection_check(curves, {**receipt, 'validation_used_for_selection': True}, 'joint', spec)
        with self.assertRaisesRegex(ValueError, 'Epoch zero'):
            e.selection_check(curves[1:], receipt, 'joint', spec)

    def test_all_fit_completion_precedes_metadata_and_validation_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'SCIENCE_SPEC.json').write_text(json.dumps({'bindings': {}}))
            spec_sha = e.sha(root / 'SCIENCE_SPEC.json')
            gate = root / 'GO.json'
            gate.write_text(json.dumps({'approved': True, 'science_spec_sha256': spec_sha,
                                       'analysis_code_sha256': e.sha(e.__file__)}))
            with patch.object(e, 'ROOT', root), patch.object(e, 'SPEC_SHA', spec_sha), \
                    patch.object(e, 'module', return_value=object()), \
                    patch.object(e, 'completed_fits', side_effect=ValueError('Not all fits complete')), \
                    patch.object(e, 'load_metadata') as metadata:
                with self.assertRaisesRegex(ValueError, 'Not all fits complete'):
                    e.run(root / 'output', gate)
                metadata.assert_not_called()
                self.assertTrue((root / 'output/FAILURE.json').is_file())

    def test_binding_detects_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'input'
            p.write_bytes(b'original')
            bound = e.record(p)
            self.assertEqual(e.verify(bound), p.resolve())
            p.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                e.verify(bound)


class GroupTests(unittest.TestCase):
    def test_fixed_groups_history_boundaries_and_head_roles(self):
        meta = synthetic_metadata()
        groups = {(scope, group): mask for scope, group, mask in e.group_masks(meta, 'train')}
        self.assertEqual(len(groups), 24)
        expected = {'history_1_63': [0, 1], 'history_64_127': [2, 3],
                    'history_128_192': [4, 5], 'history_193_plus': [6, 7],
                    'matched_532_history_128_192': [4, 5], 'quantity_gt_187_posthoc': [1]}
        for label, indices in expected.items():
            np.testing.assert_array_equal(np.flatnonzero(groups[('whole', label)]), indices)
        self.assertEqual(groups[('head_fit', 'history_128_192')].sum(), 0)
        self.assertEqual(groups[('head_dev', 'all')].sum(), 4)
        self.assertEqual(len(list(e.group_masks(meta, 'validation'))), 8)

    def test_original_baseline_zero_and_frozen_quantity_metrics(self):
        helper = e.module(e.TIME_HELPER, e.TIME_HELPER_SHA, 'synthetic_time_diagnostics')
        meta = synthetic_metadata()
        mu, sigma = np.zeros(8), np.full(8, .5)
        diagnostics = helper.time_diagnostics(meta.target_gap, mu, sigma)
        base = diagnostics['recalculated_time_nll']
        quantity = meta.raw_quantity.to_numpy() + np.arange(8)
        original = e.event_frame(meta, {'mu': mu, 'sigma': sigma, 'time_nll': base}, diagnostics,
                                 quantity, base, base, 'original')
        joint = e.event_frame(meta, {'mu': mu, 'sigma': sigma, 'time_nll': base + .3}, diagnostics,
                              quantity, base, base, 'joint')
        np.testing.assert_array_equal(original.nll_change_vs_original, np.zeros(8))
        np.testing.assert_allclose(joint.nll_change_vs_original, .3)
        a, b = e.summarize(original), e.summarize(joint)
        self.assertEqual(a['quantity_mae'], b['quantity_mae'])
        self.assertEqual(a['quantity_rmse'], b['quantity_rmse'])
        self.assertEqual(a['quantity_mae'], 3.5)
        self.assertEqual(e.summarize(original.iloc[:0])['status'], 'empty')

    def test_paired_direct_contrast_uses_same_entity_draws(self):
        helper = e.module(e.TIME_HELPER, e.TIME_HELPER_SHA, 'synthetic_ci_helper')
        entities = np.repeat(np.arange(25), 2)
        width = entities.astype(float) / 20
        joint = width + 2
        values = np.column_stack([np.zeros(50), width, joint, joint - width])
        result = helper.cluster_matrix_ci(values, entities, seed_offset=0)
        self.assertEqual(result['executed_replicates'], 1000)
        self.assertEqual(result['estimate'][0], 0)
        self.assertEqual(result['lower'][0], 0)
        self.assertEqual(result['upper'][0], 0)
        self.assertAlmostEqual(result['estimate'][3], 2)
        self.assertAlmostEqual(result['lower'][3], 2)
        self.assertAlmostEqual(result['upper'][3], 2)
        self.assertAlmostEqual(result['lower'][2] - result['lower'][1], 2)


class MathTests(unittest.TestCase):
    def test_direct_head_math_matches_fitter_in_identical_batches(self):
        fit = e.module(e.ROOT / 'head/fitter.py', e.FITTER_SHA, 'synthetic_selected_fitter')
        rng = np.random.default_rng(3)
        x = rng.normal(size=(17, 64)).astype(np.float32)
        state = {name: torch.from_numpy(rng.normal(size=shape)).double() for name, shape in fit.HEAD_SHAPES.items()}
        mu, sigma = e.direct_math(state, x, batch_size=5)
        result = fit.predict(state, x, np.full(17, 3.), batch_size=5)
        np.testing.assert_array_equal(mu, result['mu'])
        np.testing.assert_array_equal(sigma, result['sigma'])

    def test_declared_parity_limits_and_nonfinite_fail(self):
        tolerance = {'atol': 1e-5, 'rtol': 1e-5}
        self.assertEqual(e.parity([1. + 1.9e-5], [1.], tolerance)['violations'], 0)
        for values in ([1. + 2.1e-5], [float('nan')]):
            with self.assertRaises(ValueError):
                e.parity(values, [1.], tolerance)


if __name__ == '__main__':
    unittest.main()
