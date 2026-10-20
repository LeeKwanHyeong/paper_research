import copy
import math
import unittest

from paper.scripts import compare_titantpp_shared_raw_raf3_test as c


def records():
    rows = []
    for model in (c.SHARED, c.MLP16, *c.EXTERNAL):
        for n, seed in enumerate(c.SEEDS):
            values = {"count": 5226, "qty_rmse": float(n + 1), "qty_mae": float(n + 1), "time_nll": -1.0}
            rows.append({"model": model, "seed": seed, "dataset": "raf_spare_parts", "split": "test",
                "data_variant": "original_uncapped", "selected_epoch": 9,
                "target_identity": {"target_count": 5226, "target_identity_sha256": "a" * 64,
                                    "target_quantity_sha256": "b" * 64},
                "metrics": {"overall": values, "tail": {**values, "count": 39}}})
    return rows


class ComparisonTests(unittest.TestCase):
    def test_sample_sd_and_no_input_mutation(self):
        rows = records(); before = copy.deepcopy(rows)
        result = c.aggregate(rows)
        self.assertEqual(result["summary"][c.SHARED]["overall"]["qty_rmse"],
                         {"mean": 2.0, "sample_sd_ddof1": 1.0})
        self.assertEqual(rows, before)
        self.assertFalse(result["external_both_quantity_means_improved"])

    def test_deterioration_is_retained(self):
        rows = records(); rows[0]["metrics"]["overall"]["qty_mae"] += 3
        out = c.aggregate(rows)
        delta = next(x for x in out["seed_differences"] if x["reference_model"] == c.MLP16
                     and x["seed"] == 42 and x["group"] == "overall" and x["metric"] == "qty_mae")
        self.assertEqual(delta["absolute_difference"], 3)
        self.assertEqual(delta["relative_difference_percent"], 300)
        self.assertTrue(delta["worsened"])

    def test_missing_seed_rejected(self):
        with self.assertRaises(ValueError): c.aggregate(records()[:-1])

    def test_duplicate_rejected(self):
        rows = records()
        with self.assertRaises(ValueError): c.aggregate(rows + [rows[0]])

    def test_population_mismatch_rejected(self):
        rows = records(); rows[-1]["target_identity"]["target_quantity_sha256"] = "c" * 64
        with self.assertRaises(ValueError): c.aggregate(rows)

    def test_nan_rejected(self):
        rows = records(); rows[0]["metrics"]["overall"]["qty_rmse"] = math.nan
        with self.assertRaises(ValueError): c.aggregate(rows)

    def test_validation_or_cap_rejected(self):
        for field, value in [("split", "validation"), ("data_variant", "cap")]:
            rows = records(); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): c.aggregate(rows)

    def test_zero_reference_relative_difference_is_na(self):
        rows = records()
        for row in rows: row["metrics"]["overall"]["qty_mae"] = 0.0
        out = c.aggregate(rows)
        self.assertTrue(all(x["relative_difference_percent"] is None for x in out["seed_differences"]
                            if x["group"] == "overall" and x["metric"] == "qty_mae"))

    def test_worker_baseline_key_preserved_and_required(self):
        row = records()[0]
        baseline = {**copy.deepcopy(row["metrics"]), "time_metric": "N/A_no_time_model"}
        for group in c.GROUPS: baseline[group]["time_nll"] = None
        receipt = {"metrics": row["metrics"], "simple_quantity_baselines":
                   {"last": baseline, "history_mean": copy.deepcopy(baseline)}}
        self.assertEqual(c.simple_quantity_baselines(receipt), receipt["simple_quantity_baselines"])
        with self.assertRaises(ValueError): c.simple_quantity_baselines({"metrics": row["metrics"]})


if __name__ == "__main__":
    unittest.main()
