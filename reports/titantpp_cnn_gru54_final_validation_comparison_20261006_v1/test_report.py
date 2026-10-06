"""Evidence boundary and frozen-selector regressions; local Validation JSON only."""
import copy
import json
from pathlib import Path
import unittest

import build_report as report


class FrozenValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.bindings, cls.contract, _, _ = report.collect()
        cls.summary = report.summarize(cls.rows)
        cls.seed_pairs, cls.pairs = report.paired(cls.rows)
        cls.example = next(b for b in cls.bindings if b['model'] == report.S2P2 and b['dataset'] == 'yellow_trip_hourly' and b['seed'] == 52)
        cls.endpoint = report.read(report.ROOT / cls.example['endpoint']['path'])
        cls.history = report.read(report.ROOT / cls.example['history']['path'])
        cls.input = report.read(report.ROOT / cls.example['input_receipt']['path'])
        cls.spec = next(s for s in cls.contract['datasets'] if s['dataset_id'] == 'yellow_trip_hourly')

    def test_mixed_performance_and_heldout_paths_are_denied_before_read(self):
        for path in ('reports/titantpp_three_dataset_final_comparison_20261005_v1/metrics_per_seed.csv', 'search_artifacts/fake/runs/test/receipt.json', 'search_artifacts/fake/test__cnn/endpoint_replays.json', 'search_artifacts/fake/leaderboard/runs.csv'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                report.approved_path(report.ROOT / path)

    def test_changed_population_and_unbound_packaging_are_rejected(self):
        changed = copy.deepcopy(self.input)
        changed['populations']['validation']['target_quantity_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'validation population differs'):
            report.verify_population(changed, self.spec)
        changed = copy.deepcopy(self.input)
        changed['input_identity']['data_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'package equivalence'):
            report.verify_population(changed, self.spec)

    def test_time_nll_from_another_epoch_is_rejected(self):
        changed = copy.deepcopy(self.endpoint)
        changed['selected']['time_nll'] += 0.1
        with self.assertRaisesRegex(ValueError, 'same-epoch metric differs: time_nll'):
            report.verify_endpoint(changed, self.history, self.example, 'yellow_trip_hourly')

    def test_wrong_checkpoint_state_and_epoch_are_rejected(self):
        changed = copy.deepcopy(self.example)
        changed['state_tensor_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Selected state binding differs'):
            report.verify_endpoint(self.endpoint, self.history, changed, 'yellow_trip_hourly')
        changed = copy.deepcopy(self.example)
        changed['selected_epoch'] += 1
        with self.assertRaisesRegex(ValueError, 'Selected epoch binding differs'):
            report.verify_endpoint(self.endpoint, self.history, changed, 'yellow_trip_hourly')

    def test_partial_seed_or_duplicate_condition_cannot_create_summary(self):
        with self.assertRaisesRegex(ValueError, 'coverage incomplete'):
            report.check_coverage(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate metric row'):
            report.check_coverage(self.rows + [self.rows[0]])

    def test_final_original_cnn_values_and_sample_sd_match_frozen_final9(self):
        expected = {'yellow_trip_hourly': (79.06278740446082, 2.6667544641315613), 'intermittent_frozen_5000': (1.7296402532053372, 0.04773942415300527), 'raf_spare_parts': (34.184122896363355, 0.23335918147452908)}
        for dataset, (mean, sd) in expected.items():
            row = next(r for r in self.summary if (r['dataset'], r['model'], r['view']) == (dataset, report.CNN, 'overall'))
            self.assertEqual(row['qty_rmse_mean'], mean)
            self.assertEqual(row['qty_rmse_sample_sd'], sd)
        self.assertEqual(len(self.bindings), 123)
        self.assertEqual(len(self.rows), 246)
        self.assertEqual(len(self.seed_pairs), 684)
        self.assertEqual(len(self.pairs), 228)

    def test_scientific_tradeoff_is_preserved_and_near_zero_ratio_suppressed(self):
        gates = report.scientific_gate(self.pairs)
        self.assertTrue(all(g['status'] == 'FAIL_TRADEOFF' for g in gates))
        taxi_s2p2 = next(p for p in self.pairs if (p['dataset'], p['reference'], p['view'], p['metric']) == ('yellow_trip_hourly', report.S2P2, 'overall', 'time_nll'))
        self.assertGreater(taxi_s2p2['delta_mean'], 0)
        self.assertEqual(taxi_s2p2['worsened_seeds'], 3)
        for p in self.seed_pairs:
            if abs(p['reference_value']) < 1e-6:
                self.assertIsNone(p['relative_change_percent'])

    def test_gate_checkpoint_sha_and_all_output_source_sha_are_bound(self):
        gate = json.loads((report.HERE / 'strict_validation_gate.json').read_text())
        self.assertTrue(gate['passed'])
        self.assertEqual(gate['conditions'], 9)
        self.assertEqual(gate['evaluation_scope'], 'validation_only')
        self.assertFalse(gate['held_out_test_evaluated'])
        self.assertEqual(len(gate['rows']), 9)
        self.assertEqual({r['seed'] for r in gate['rows']}, {42, 52, 62})
        for row in gate['rows']:
            for key in ('checkpoint_file_sha256', 'state_tensor_sha256', 'source_closure_sha256'):
                self.assertEqual(len(row[key]), 64)
        receipt = json.loads((report.HERE / 'comparison_receipt.json').read_text())
        self.assertFalse(receipt['held_out_test_evaluated'])
        for path, expected in receipt['source_files_sha256'].items():
            self.assertEqual(report.sha(report.ROOT / path), expected, path)
        for name, expected in receipt['outputs'].items():
            self.assertEqual(report.sha(report.HERE / name), expected, name)

    def test_alternate_packaged_time_equivalence_is_runtime_bound(self):
        alternate = [b for b in self.bindings if not b['input_file_SHA_exactly_canonical']]
        self.assertEqual({(b['dataset'], b['model'], b['seed']) for b in alternate}, {('intermittent_frozen_5000', 'titantpp_all_available_history_mlp', 62), ('intermittent_frozen_5000', 'titantpp_current_only_param_matched', 62)})
        for binding in alternate:
            evidence = binding['time_input_equivalence']
            self.assertEqual(evidence['status'], 'passed')
            self.assertEqual(evidence['expected_validation_target_dt_sha256'], '5ac9547d2fcbcb8f9e5394a88fc379d1c895683db8ea96b7ed6009d374eddea7')
            self.assertEqual(evidence['observation_likelihood']['unit'], 'week')
            self.assertEqual(len(evidence['frozen_source_checks']), 2)


if __name__ == '__main__':
    unittest.main()
