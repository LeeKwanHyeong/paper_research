from copy import deepcopy
import unittest

from paper.scripts import compare_titantpp_shared_raf_taxi6_test as c


def records():
    rows = []
    for dataset, count in c.DATASETS.items():
        identity = {"target_count": count, "target_identity_sha256": dataset,
                    "target_quantity_sha256": "truth-" + dataset}
        for index, model in enumerate((c.SHARED, c.MLP16, *c.EXTERNAL)):
            for seed in c.SEEDS:
                rows.append({"dataset": dataset, "model": model, "seed": seed,
                    "split": "test", "data_variant": "original_uncapped",
                    "selected_epoch": c.EPOCHS[dataset][seed] if model == c.SHARED else 2,
                    "target_identity": deepcopy(identity), "metrics": {
                        group: {"count": count if group == "overall" else 7,
                            "qty_rmse": 10.0 + index + (seed - 42) / 10,
                            "qty_mae": 5.0 + index + (seed - 42) / 10,
                            "time_nll": 0.3 + index} for group in c.GROUPS}})
    return rows


class CompareTests(unittest.TestCase):
    def test_complete_two_dataset_statistics_and_all_signs(self):
        result = c.aggregate(records())
        self.assertFalse(result["cross_dataset_raw_error_pooled"])
        for dataset in c.DATASETS:
            x = result["datasets"][dataset]
            self.assertEqual(x["summary"][c.SHARED]["overall"]["qty_rmse"],
                {"mean": 11.0, "sample_sd_ddof1": 1.0})
            self.assertEqual(len(x["seed_differences"]), 7 * 3 * 6)
            self.assertTrue(all(r["absolute_difference"] < 0 for r in x["seed_differences"]))

    def test_worsening_is_preserved_and_metric_bests_can_differ(self):
        rows = records()
        for row in rows:
            if row["model"] == "rmtpp":
                row["metrics"]["overall"]["qty_mae"] = 0.5
        result = c.aggregate(rows)
        for x in result["datasets"].values():
            self.assertFalse(x["external_both_quantity_means_improved"])
            bad = [r for r in x["seed_differences"] if r["reference_model"] == "rmtpp"
                and r["group"] == "overall" and r["metric"] == "qty_mae"]
            self.assertTrue(all(r["worsened"] for r in bad))

    def test_missing_seed_rejected(self):
        with self.assertRaises(ValueError): c.aggregate(records()[:-1])

    def test_duplicate_endpoint_rejected(self):
        rows = records(); rows[-1] = deepcopy(rows[-2])
        with self.assertRaises(ValueError): c.aggregate(rows)

    def test_test_chosen_candidate_epoch_rejected(self):
        rows = records(); rows[0]["selected_epoch"] += 1
        with self.assertRaises(ValueError): c.aggregate(rows)

    def test_target_mismatch_rejected(self):
        rows = records(); rows[-1]["target_identity"]["target_quantity_sha256"] = "foreign"
        with self.assertRaises(ValueError): c.aggregate(rows)

    def test_cap_or_other_dataset_rejected(self):
        for field, value in (("data_variant", "cap"), ("dataset", "intermittent_frozen_5000")):
            with self.subTest(field=field):
                rows = records(); rows[0][field] = value
                with self.assertRaises(ValueError): c.aggregate(rows)

    def test_cross_dataset_support_not_mixed(self):
        rows = records(); rows[-1]["metrics"]["tail"]["count"] = 8
        with self.assertRaises(ValueError): c.aggregate(rows)

    def test_report_has_both_datasets_and_runtime_limitation(self):
        text = c.markdown(c.aggregate(records()))
        self.assertIn("원본 RAF", text); self.assertIn("원본 Taxi", text)
        self.assertIn("q>3449", text); self.assertIn("q>200", text)
        self.assertIn("학습5090/cuDNN92000", text)


if __name__ == "__main__": unittest.main()
