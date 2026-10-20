import copy
import math
import unittest

from paper.scripts import compare_titantpp_shared_inter3_interim_test as c
from paper.scripts import prepare_titantpp_shared_inter3_interim_references as refs


def records(split="test"):
    count = c.COUNTS[split]
    rows = []
    for model in (c.SHARED, c.MLP16, *c.EXTERNAL):
        for n, seed in enumerate(c.SEEDS):
            values = {"count": count, "qty_rmse": float(n + 1), "qty_mae": float(n + 1), "time_nll": -1.0}
            rows.append({"model": model, "seed": seed, "dataset": c.DATASET, "split": split,
                "data_variant": "original_uncapped", "selected_epoch": 5,
                "target_identity": {"target_count": count, "target_identity_sha256": "a" * 64,
                                    "target_quantity_sha256": "b" * 64},
                "metrics": {"overall": values, "tail": {**values, "count": 1141}}})
    return rows


class InterimComparisonTests(unittest.TestCase):
    def test_both_splits_sample_sd_and_no_mutation(self):
        for split in c.COUNTS:
            with self.subTest(split=split):
                rows = records(split)
                before = copy.deepcopy(rows)
                result = c.aggregate(rows, split)
                self.assertEqual(result["summary"][c.SHARED]["overall"]["qty_rmse"],
                                 {"mean": 2.0, "sample_sd_ddof1": 1.0})
                self.assertEqual(result["records"], before)
                self.assertEqual(rows, before)
                self.assertEqual(result["split"], split)
                self.assertTrue(result["interim_checkpoint_snapshot"])
                self.assertFalse(result["training_campaign_complete"])
                self.assertFalse(result["checkpoint_reselected"])
                self.assertEqual(result["held_out_test_evaluated"], split == "test")

    def test_adverse_all_six_metric_differences_retained(self):
        rows = records()
        for group in c.GROUPS:
            for metric in c.METRICS:
                rows[0]["metrics"][group][metric] += 3.0
        result = c.aggregate(rows)
        for group in c.GROUPS:
            for metric in c.METRICS:
                difference = next(r for r in result["seed_differences"] if r["reference_model"] == c.MLP16
                                  and r["seed"] == 42 and r["group"] == group and r["metric"] == metric)
                self.assertEqual(difference["absolute_difference"], 3)
                self.assertEqual(difference["relative_difference_percent"], 300)
                self.assertTrue(difference["worsened"])
        self.assertEqual(len(result["seed_differences"]), 7 * 3 * 6)

    def test_external_best_is_primary_and_mlp16_excluded(self):
        rows = records()
        for row in rows:
            if row["model"] == c.MLP16:
                row["metrics"]["overall"]["qty_rmse"] = 0
                row["metrics"]["overall"]["qty_mae"] = 0
            elif row["model"] == "thp":
                row["metrics"]["overall"]["qty_rmse"] = 0.5
        result = c.aggregate(rows)
        self.assertEqual(result["external_metric_bests"]["qty_rmse"]["external_model"], "thp")
        self.assertEqual(result["comparison_secondary"], c.MLP16)
        self.assertFalse(result["external_metric_bests"]["qty_rmse"]["shared_strictly_better"])

    def test_missing_duplicate_and_foreign_endpoints_rejected(self):
        for mutate in (lambda r: r[:-1], lambda r: r[:-1] + [r[0]],
                       lambda r: [{**r[0], "split": "validation"}, *r[1:]],
                       lambda r: [{**r[0], "data_variant": "cap"}, *r[1:]],
                       lambda r: [{**r[0], "dataset": "raf_spare_parts"}, *r[1:]]):
            with self.assertRaises(ValueError):
                c.aggregate(mutate(records()))

    def test_population_quantity_identity_and_nonfinite_rejected(self):
        rows = records()
        rows[-1]["target_identity"]["target_quantity_sha256"] = "c" * 64
        with self.assertRaises(ValueError): c.aggregate(rows)
        rows = records(); rows[0]["metrics"]["overall"]["time_nll"] = math.nan
        with self.assertRaises(ValueError): c.aggregate(rows)

    def test_frozen_selected_epoch_and_split_count_enforced(self):
        with self.assertRaises(ValueError):
            c.aggregate(records(), selected_epochs={42: 5, 52: 5, 62: 11})
        rows = records("validation"); rows[0]["metrics"]["overall"]["count"] = c.COUNTS["test"]
        with self.assertRaises(ValueError): c.aggregate(rows, "validation")

    def test_zero_relative_reference_and_time_na_baselines(self):
        rows = records()
        for row in rows:
            row["metrics"]["overall"]["qty_mae"] = 0.0
        result = c.aggregate(rows)
        self.assertTrue(all(r["relative_difference_percent"] is None for r in result["seed_differences"]
                            if r["group"] == "overall" and r["metric"] == "qty_mae"))
        row = rows[0]
        baseline = {**copy.deepcopy(row["metrics"]), "time_metric": "N/A_no_time_model"}
        for group in c.GROUPS: baseline[group]["time_nll"] = None
        receipt = {"metrics": row["metrics"], "simple_quantity_baselines":
                   {"last": baseline, "history_mean": copy.deepcopy(baseline)}}
        self.assertEqual(c.simple_quantity_baselines(receipt), receipt["simple_quantity_baselines"])
        receipt["simple_quantity_baselines"]["last"]["overall"]["time_nll"] = 0.0
        with self.assertRaises(ValueError): c.simple_quantity_baselines(receipt)

    def test_reference_selection_first_finite_strict_minimum(self):
        history = [{"epoch": 1, "val_qty_rmse": math.nan}, {"epoch": 2, "val_qty_rmse": 3.0},
                   {"epoch": 3, "val_qty_rmse": 2.0}, {"epoch": 4, "val_qty_rmse": 2.0}]
        self.assertEqual(refs.selected_history(history)["epoch"], 3)
        with self.assertRaises(ValueError): refs.selected_history(history[::-1])
        with self.assertRaises(ValueError): refs.selected_history([history[0]])

    def test_actual_child_exit_required(self):
        from pathlib import Path
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "exit.json"
            for item in ({"actual_os_returncode": 0, "timeout": False},
                         {"actual_os_returncode": 1, "timeout": False, "actual_child_exit_observed": True}):
                file.write_text(json.dumps(item))
                with self.assertRaises(ValueError): c.verify_actual_exit(file)
            file.write_text(json.dumps({"actual_os_returncode": 0, "timeout": False, "actual_child_exit_observed": True}))
            self.assertEqual(c.verify_actual_exit(file)["actual_os_returncode"], 0)


if __name__ == "__main__":
    unittest.main()
