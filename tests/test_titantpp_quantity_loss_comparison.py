"""Arithmetic and comparison boundaries independent of native training."""
from copy import deepcopy
import math
import unittest

from paper.scripts.compare_titantpp_quantity_loss_mlp18 import aggregate, DATASETS, SEEDS, LOSSES


def fabricated_records():
    records = []
    for dataset in DATASETS:
        for i, seed in enumerate(SEEDS, 1):
            host = "5080" if dataset == DATASETS[0] or (dataset == DATASETS[2] and seed == 42) else "5090"
            for loss, multiplier in zip(LOSSES, (1., .5, .8)):
                quantity = i * multiplier
                cohort = {"count": 100, "qty_rmse": quantity, "qty_mae": quantity / 2., "time_nll": float(i)}
                records.append({"dataset": dataset, "seed": seed, "host": host, "loss_mode": loss,
                    "time_alpha": 0., "time_head_trained": False, "target_identity": {"dataset": dataset, "hash": "same"},
                    "metrics": {"overall": cohort, "cap_hit": {**cohort, "count": 10},
                        "tail": {"count": 0, "qty_rmse": None, "qty_mae": None, "time_nll": None}}})
    return records


class ComparisonTests(unittest.TestCase):
    def test_arithmetic_matches_closed_form_and_empty_population(self):
        result = aggregate(fabricated_records())
        self.assertEqual((len(result["summary"]), len(result["seed_differences"]), len(result["decisions"])), (81, 243, 6))
        row = next(x for x in result["summary"] if x["dataset"] == DATASETS[0]
                   and x["cohort"] == "overall" and x["loss_mode"] == "raw_mse" and x["metric"] == "qty_rmse")
        self.assertEqual((row["mean"], row["sample_sd_ddof1"]), (1., .5))
        delta = next(x for x in result["seed_differences"] if x["dataset"] == DATASETS[0]
                     and x["cohort"] == "overall" and x["seed"] == 52 and x["metric"] == "qty_rmse"
                     and x["left_loss_mode"] == "raw_mse" and x["right_loss_mode"] == "log_mse")
        self.assertEqual((delta["absolute_difference"], delta["relative_difference_percent"], delta["worsened"]), (-1., -50., False))
        self.assertTrue(all(x["mean"] is None and x["sample_sd_ddof1"] is None
                            for x in result["summary"] if x["cohort"] == "tail"))
        self.assertTrue(all(x["quantity_both_means_improved_vs_log"] for x in result["decisions"]))

    def test_duplicate_missing_condition_is_rejected(self):
        records = fabricated_records()
        records[-1] = deepcopy(records[-2])
        with self.assertRaisesRegex(ValueError, "exact18"):
            aggregate(records)

    def test_different_target_host_scope_or_head_policy_is_rejected(self):
        for field, value in (("target_identity", {}), ("host", "other"), ("time_alpha", 1.), ("time_head_trained", True)):
            with self.subTest(field=field):
                records = fabricated_records()
                records[1][field] = value
                with self.assertRaises(ValueError):
                    aggregate(records)

    def test_different_population_is_rejected(self):
        records = fabricated_records()
        records[1]["metrics"]["overall"]["count"] = 99
        with self.assertRaisesRegex(ValueError, "counts"):
            aggregate(records)

    def test_nonfinite_negative_or_fake_empty_metrics_are_rejected(self):
        for cohort, metric, value in (("overall", "qty_rmse", math.nan), ("overall", "qty_mae", -1.),
                                      ("tail", "qty_rmse", 0.), ("overall", "time_nll", True)):
            with self.subTest(cohort=cohort, metric=metric, value=value):
                records = fabricated_records()
                records[1]["metrics"][cohort][metric] = value
                with self.assertRaises(ValueError):
                    aggregate(records)

    def test_zero_reference_relative_difference_is_na_and_worsening_preserved(self):
        records = fabricated_records()
        records[0]["metrics"]["overall"]["qty_mae"] = 0.
        result = aggregate(records)
        row = next(x for x in result["seed_differences"] if x["dataset"] == DATASETS[0]
                   and x["cohort"] == "overall" and x["seed"] == 42 and x["metric"] == "qty_mae"
                   and x["left_loss_mode"] == "raw_mse" and x["right_loss_mode"] == "log_mse")
        self.assertIsNone(row["relative_difference_percent"])
        self.assertTrue(row["worsened"])


if __name__ == "__main__":
    unittest.main()
