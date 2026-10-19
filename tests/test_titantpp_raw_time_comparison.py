"""Independent arithmetic and paired-input comparison boundaries."""
from copy import deepcopy
import math
import statistics
import unittest

from paper.scripts import compare_titantpp_raw_time_mlp27 as comparison


def records():
    rows = []
    for dataset in comparison.DATASETS:
        for index, seed in enumerate(comparison.SEEDS, 1):
            host = "5080" if dataset == comparison.DATASETS[0] or (
                dataset == comparison.DATASETS[2] and seed == 42) else "5090"
            for loss in comparison.LOSSES:
                for transform in comparison.TRANSFORMS:
                    value = index * (1. if transform == "log1p" else .5)
                    row = {"count": 100, "qty_rmse": value, "qty_mae": value / 2., "time_nll": float(index)}
                    rows.append({"dataset": dataset, "host": host, "seed": seed, "loss_mode": loss,
                        "time_input": transform, "time_alpha": 0., "time_head_trained": False,
                        "target_identity": {"dataset": dataset, "truth_hash": "same"},
                        "metrics": {"overall": deepcopy(row), "cap_hit": {**row, "count": 10},
                            "tail": {"count": 0, "qty_rmse": None, "qty_mae": None, "time_nll": None}}})
    return rows


class ComparisonTests(unittest.TestCase):
    def test_exact_arithmetic_and_sample_sd(self):
        result = comparison.aggregate(records())
        self.assertEqual((len(result["summary"]), len(result["seed_differences"]), len(result["decisions"])),
                         (162, 243, 9))
        row = next(r for r in result["summary"] if r["dataset"] == comparison.DATASETS[0]
                   and r["loss_mode"] == "raw_mae" and r["time_input"] == "raw"
                   and r["cohort"] == "overall" and r["metric"] == "qty_rmse")
        self.assertEqual((row["mean"], row["sample_sd_ddof1"]), (1., .5))
        delta = next(r for r in result["seed_differences"] if r["dataset"] == comparison.DATASETS[0]
                     and r["loss_mode"] == "raw_mae" and r["seed"] == 52
                     and r["cohort"] == "overall" and r["metric"] == "qty_rmse")
        self.assertEqual((delta["absolute_difference"], delta["relative_difference_percent"], delta["worsened"]),
                         (-1., -50., False))
        self.assertTrue(result["common_improvement_across_datasets_and_losses"])
        self.assertTrue(all(r["mean"] is None for r in result["summary"] if r["cohort"] == "tail"))

    def test_wrong_matrix_and_duplicate_are_rejected(self):
        for mutated in (records()[:-1], records() + records()[:1]):
            with self.assertRaisesRegex(ValueError, "exact stored"):
                comparison.aggregate(mutated)
        mutated = records()
        mutated[-1] = deepcopy(mutated[-2])
        with self.assertRaises(ValueError):
            comparison.aggregate(mutated)

    def test_pair_boundaries(self):
        for field, value in (("host", "5090"), ("target_identity", {}), ("time_alpha", .1),
                             ("time_head_trained", True)):
            with self.subTest(field=field):
                mutated = records()
                mutated[1][field] = value
                with self.assertRaises(ValueError):
                    comparison.aggregate(mutated)

    def test_bad_metrics_counts_and_fake_empty(self):
        for cohort, metric, value in (("overall", "qty_rmse", math.nan),
                ("overall", "qty_mae", -1.), ("overall", "time_nll", True),
                ("tail", "qty_mae", 0.), ("overall", "count", 99)):
            with self.subTest(cohort=cohort, metric=metric):
                mutated = records()
                mutated[1]["metrics"][cohort][metric] = value
                with self.assertRaises(ValueError):
                    comparison.aggregate(mutated)

    def test_seed_exception_and_different_loss_winner_do_not_disappear(self):
        mutated = records()
        raw = next(r for r in mutated if r["dataset"] == comparison.DATASETS[0] and r["seed"] == 42
                   and r["loss_mode"] == "log_mse" and r["time_input"] == "raw")
        raw["metrics"]["overall"]["qty_mae"] = 1.
        result = comparison.aggregate(mutated)
        decision = next(r for r in result["decisions"]
                        if r["dataset"] == comparison.DATASETS[0] and r["loss_mode"] == "log_mse")
        self.assertTrue(decision["both_quantity_means_improved"])
        self.assertFalse(decision["all_three_seeds_improved_both"])
        self.assertTrue(any(r["worsened"] for r in result["seed_differences"]))

    def test_zero_reference_relative_delta_is_na(self):
        mutated = records()
        mutated[0]["metrics"]["overall"]["qty_mae"] = 0.
        result = comparison.aggregate(mutated)
        row = next(r for r in result["seed_differences"] if r["dataset"] == comparison.DATASETS[0]
                   and r["loss_mode"] == "log_mse" and r["seed"] == 42
                   and r["cohort"] == "overall" and r["metric"] == "qty_mae")
        self.assertIsNone(row["relative_difference_percent"])
        self.assertTrue(row["worsened"])


if __name__ == "__main__":
    unittest.main()
