"""Synthetic guards for misleading partial/duplicate/population comparisons."""
import copy
import unittest
import build_report as report


def valid_rows():
    rows = []
    for dataset in report.DATASETS:
        for model in report.ORDER:
            if dataset == 'raf_spare_parts' and model == 'titantpp':
                continue
            for seed in [None] if model in report.SIMPLE else sorted(report.SEEDS):
                for split in ('validation', 'test'):
                    for view in ('overall', 'tail'):
                        rows.append({'dataset': dataset, 'model': model, 'seed': seed, 'split': split, 'view': view, 'selected_epoch': None if seed is None else 7, 'count': 10 if view == 'tail' else 100, 'threshold': report.THRESHOLDS[dataset] if view == 'tail' else None, 'qty_rmse': 4 + (seed or 0)/100, 'qty_mae': 2, 'time_nll': None if seed is None else 1})
    return rows


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.rows = valid_rows()

    def test_complete_has_123_conditions_and_516_rows(self):
        report.check_coverage(self.rows)
        self.assertEqual(len(self.rows), 516)
        self.assertEqual(len({(r['dataset'], r['model'], r['seed']) for r in self.rows if r['seed'] is not None}), 123)

    def test_partial_seed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            report.check_coverage(self.rows[:-1])

    def test_duplicate_row_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report.check_coverage(self.rows + [self.rows[0]])

    def test_mixed_population_is_rejected(self):
        self.rows[0]['count'] = 99
        with self.assertRaisesRegex(ValueError, 'Population counts'):
            report.check_coverage(self.rows)

    def test_tail_threshold_change_is_rejected(self):
        self.rows[1]['threshold'] += 1
        with self.assertRaisesRegex(ValueError, 'TRAIN threshold'):
            report.check_coverage(self.rows)

    def test_cross_split_epoch_change_is_rejected(self):
        self.rows[2]['selected_epoch'] = 8
        with self.assertRaisesRegex(ValueError, 'Selected epoch changed'):
            report.check_coverage(self.rows)

    def test_ddof1_and_paired_delta_are_not_model_sd_subtraction(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['model'] == report.WIDTHS[1]:
                row['qty_rmse'] += {42: 1, 52: -1, 62: 2}[row['seed']]
        summaries, raw = report.paired(rows)
        pair = next(r for r in summaries if r['model'] == report.WIDTHS[1] and r['metric'] == 'qty_rmse')
        self.assertAlmostEqual(pair['delta_mean'], 2/3)
        self.assertAlmostEqual(pair['delta_sample_sd'], (7/3)**0.5)
        self.assertEqual(pair['improved_seeds'], 1)
        self.assertEqual(pair['worst_delta_seed'], 62)
        self.assertEqual(len(raw), 1368)

    def test_small_time_nll_is_not_rounded_to_zero(self):
        self.assertEqual(report.fmt(1.3e-9, 'time_nll'), '1.300e-09')


if __name__ == '__main__':
    unittest.main()
