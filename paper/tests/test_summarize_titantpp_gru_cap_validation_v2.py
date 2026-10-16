"""Regression tests for immutable native nested Validation receipts."""
import copy
import unittest
from unittest.mock import patch

from paper.scripts import summarize_titantpp_gru_cap_validation_v2 as summary
from paper.tests import test_summarize_titantpp_gru_cap_validation as fixtures


def nested(flat):
    def population(prefix, count):
        return dict(count=count, represented_n=count, qty_sse=0.0,
                    qty_absolute_error_sum=0.0, qty_signed_error_sum=0.0,
                    time_nll_sum=0.0, qty_bias=0.0,
                    qty_rmse=flat[prefix + '_qty_rmse'],
                    qty_mae=flat[prefix + '_qty_mae'], time_nll=flat[prefix + '_time_nll'])
    return dict(**population('whole', 10), split='validation',
                quantity_boundaries=[], quantity_cells=[], tail=population('tail', 2),
                tail_definition='train_fixed', bias_definition='prediction_minus_truth')


class NestedSummaryTests(fixtures.CapValidationSummaryTests):
    def setUp(self):
        self.mapping_patch = patch.object(fixtures, 'summary', summary)
        self.mapping_patch.start()
        self.addCleanup(self.mapping_patch.stop)
        super().setUp()

    def make_baseline(self, dataset, seed, arm):
        row = super().make_baseline(dataset, seed, arm)
        path = summary.Path(row['validation_receipt_path'])
        receipt = summary.read(path)
        receipt['metrics'] = nested(receipt['metrics'])
        receipt['schema'] = 'titantpp_components_taxi_raf24_frozen_evaluation_v1'
        receipt['original_population_identity'] = row['validation_population']
        receipt['target_identity_sha256'] = 'different-evaluation-row-encoding'
        receipt['truth_sha256'] = 'different-evaluation-truth-encoding'
        row['validation_receipt_sha256'] = fixtures.dump(path, receipt)
        return row

    def test_wrong_original_population_rejected_even_when_evaluation_digest_matches(self):
        row = self.baselines[0]
        path = summary.Path(row['validation_receipt_path'])
        receipt = summary.read(path)
        receipt['original_population_identity']['target_identity_sha256'] = 'wrong'
        receipt['target_identity_sha256'] = row['validation_population']['target_identity_sha256']
        row['validation_receipt_sha256'] = fixtures.dump(path, receipt)
        self.contract['baseline_references_sha256'] = summary.canonical(self.baselines)
        with self.assertRaisesRegex(ValueError, 'Frozen baseline population differs'):
            summary.verify_baselines(self.contract)

    def test_v1_regression_and_v2_exact_mapping(self):
        from paper.scripts import summarize_titantpp_gru_cap_validation as original
        with self.assertRaisesRegex(ValueError, 'fixed six metrics'):
            original.summarize(self.contract_path, self.retrieval, self.index_path)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = summary.summarize(self.contract_path, self.retrieval, self.index_path)
        self.assertEqual(result['candidate_fit_count'], 9)
        for row in self.baselines:
            receipt = summary.read(summary.Path(row['validation_receipt_path']))
            self.assertEqual(summary.baseline_metrics(receipt['metrics'], 10), row['metrics'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})


class NativeMappingTests(unittest.TestCase):
    def setUp(self):
        self.flat = {name: float(i + 1) for i, name in enumerate(summary.METRICS)}
        self.native = nested(self.flat)

    def test_exact_flat_and_nested_values_agree(self):
        self.assertEqual(summary.baseline_metrics(self.flat, 10), self.flat)
        self.assertEqual(summary.baseline_metrics(self.native, 10), self.flat)

    def test_missing_tail_nonfinite_test_split_mixed_keys_and_count_rejected(self):
        cases = []
        value = copy.deepcopy(self.native); value.pop('tail'); cases.append(value)
        value = copy.deepcopy(self.native); value['tail']['qty_mae'] = float('nan'); cases.append(value)
        value = copy.deepcopy(self.native); value['split'] = 'test'; cases.append(value)
        value = copy.deepcopy(self.native); value['whole_qty_rmse'] = 99.; cases.append(value)
        value = copy.deepcopy(self.native); value['count'] = 9; cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                summary.baseline_metrics(value, 10)


if __name__ == '__main__':
    unittest.main()
