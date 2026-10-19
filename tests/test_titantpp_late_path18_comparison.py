"""Synthetic exact-matrix, arithmetic and local-only reporting contracts."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import paper.scripts as scripts_package
from paper.scripts import select_compare_titantpp_late_path18 as comparison


def _metrics(value, seed_index):
    return {"overall": {"count": 100, "qty_rmse": value, "qty_mae": value / 2., "time_nll": seed_index - 2.},
        "cap_hit": {"count": 10, "qty_rmse": value * 2., "qty_mae": value, "time_nll": seed_index / 3.},
        "tail": {"count": 0, "qty_rmse": None, "qty_mae": None, "time_nll": None}}


def selection_records():
    rows = []
    for dataset, scale in zip(comparison.DATASETS, (100., 10., 1.)):
        for index, seed in enumerate(comparison.SEEDS, 1):
            for transform in comparison.TRANSFORMS:
                for loss in comparison.LOSSES:
                    factor = .5 if (transform, loss) == ("raw", "raw_mae") else 1.
                    rows.append({"dataset": dataset, "host": comparison._expected_host(dataset, seed),
                        "seed": seed, "time_input": transform, "loss_mode": loss, "time_alpha": 0.,
                        "time_head_trained": False, "selected_epoch": index + 1,
                        "target_identity": {"dataset": dataset, "target_hash": str(seed)},
                        "scope": "validation_only", "held_out_test_evaluated": False,
                        "metrics": _metrics(scale * index * factor, index)})
    return rows


def arm_records():
    rows = []
    for dataset in comparison.DATASETS:
        for index, seed in enumerate(comparison.SEEDS, 1):
            for model, factor in ((comparison.BASELINE, 1.), (comparison.OBSERVATION, .5), (comparison.HIDDEN, .75)):
                rows.append({"dataset": dataset, "host": comparison._expected_host(dataset, seed), "seed": seed,
                    "model": model, "time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.,
                    "time_head_trained": False, "selected_epoch": index + 1,
                    "target_identity": {"dataset": dataset, "target_hash": str(seed)},
                    "metrics": _metrics(index * factor, index)})
    return rows


def with_metric_metadata(rows):
    for row in rows:
        row["metrics"].update(comparison.METRIC_METADATA, cap_value=comparison.CAP_VALUES[row["dataset"]])
    return rows


class SelectionTests(unittest.TestCase):
    def test_actual_receipt_metric_metadata_is_validated_and_preserved(self):
        rows = with_metric_metadata(selection_records())
        result = comparison.select_common_condition(rows)
        preserved = [r for score in result["condition_scores"] for ds in score["datasets"] for r in ds["seed_records"]]
        self.assertEqual(len(preserved), 54)
        for row in preserved:
            self.assertEqual(row["metrics"]["cap_value"], comparison.CAP_VALUES[row["dataset"]])
            for key, value in comparison.METRIC_METADATA.items():
                self.assertEqual(row["metrics"][key], value)
        for key, value in (("cap_value", 999), ("cap_value", True), ("cap_value", math.nan),
                           ("cap_hit_rule", "q >= cap_value"), ("tail_rule", "q >= cap_value"),
                           ("time_metric", "physical_time_rmse")):
            mutated = deepcopy(rows)
            mutated[0]["metrics"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    comparison.select_common_condition(mutated)
        mutated = deepcopy(rows)
        del mutated[0]["metrics"]["cap_hit_rule"]
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            comparison.select_common_condition(mutated)

    def test_all_six_scores_and_seed_values_are_preserved(self):
        rows = selection_records()
        before = deepcopy(rows)
        result = comparison.select_common_condition(rows)
        self.assertEqual(result["selected_condition"], {"time_input": "raw", "loss_mode": "raw_mae", "time_alpha": 0.})
        self.assertEqual(result["selected_scores"], {"macro_rmse_ratio": .5, "macro_mae_ratio": .5})
        self.assertEqual((len(result["condition_scores"]), result["input_endpoint_count"]), (6, 54))
        self.assertEqual(sum(len(ds["seed_records"]) for c in result["condition_scores"] for ds in c["datasets"]), 54)
        self.assertEqual(rows, before)

    def test_macro_weighting_does_not_pool_dataset_raw_errors(self):
        rows = selection_records()
        for row in rows:
            value = row["metrics"]["overall"]["qty_rmse"]
            if (row["time_input"], row["loss_mode"]) == ("raw", "raw_mae"):
                # Undo default .5 first. This setting gains 50% on the largest
                # raw-scale dataset but loses 50% on each remaining dataset.
                ratio = .5 if row["dataset"] == comparison.DATASETS[0] else 1.5
                row["metrics"]["overall"]["qty_rmse"] = value * 2. * ratio
        result = comparison.select_common_condition(rows)
        self.assertEqual(result["selected_condition"], comparison.REFERENCE_CONDITION)
        losing = next(r for r in result["condition_scores"] if r["condition"]["time_input"] == "raw"
                      and r["condition"]["loss_mode"] == "raw_mae")
        self.assertAlmostEqual(losing["macro_rmse_ratio"], 7. / 6.)

    def test_mae_is_only_secondary_after_exact_rmse_tie(self):
        rows = selection_records()
        for row in rows:
            if row["time_input"] == "raw" and row["loss_mode"] == "raw_mae":
                row["metrics"]["overall"]["qty_rmse"] *= 2.
        selected = comparison.select_common_condition(rows)["selected_condition"]
        self.assertEqual((selected["time_input"], selected["loss_mode"]), ("raw", "raw_mae"))

    def test_exact_tie_order_is_deterministic_and_not_input_order(self):
        rows = selection_records()
        for row in rows:
            if row["time_input"] == "raw" and row["loss_mode"] == "raw_mae":
                for metric in ("qty_rmse", "qty_mae"):
                    row["metrics"]["overall"][metric] *= 2.
        random.Random(7).shuffle(rows)
        self.assertEqual(comparison.select_common_condition(rows)["selected_condition"], comparison.REFERENCE_CONDITION)
        for row in rows:
            if row["time_input"] == "raw" and row["loss_mode"] == "raw_mse":
                row["metrics"]["overall"]["qty_rmse"] *= 1. - 1e-12
        self.assertEqual(comparison.select_common_condition(rows)["selected_condition"]["loss_mode"], "raw_mse")

    def test_missing_duplicate_or_wrong_matrix_rejected(self):
        for rows in (selection_records()[:-1], selection_records() + selection_records()[:1]):
            with self.assertRaisesRegex(ValueError, "54-endpoint"):
                comparison.select_common_condition(rows)
        rows = selection_records()
        rows[0]["model"] = comparison.OBSERVATION
        with self.assertRaisesRegex(ValueError, "stored MLP16"):
            comparison.select_common_condition(rows)

    def test_scope_count_identity_host_and_selection_boundaries(self):
        for key, value in (("scope", "test"), ("split", "test"), ("held_out_test_evaluated", True),
                           ("evaluation_scope", "mixed"), ("time_alpha", True), ("time_head_trained", True),
                           ("selected_epoch", 0), ("selected_epoch", True), ("target_identity", {}), ("host", "5090")):
            with self.subTest(key=key, value=value):
                rows = selection_records()
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    comparison.select_common_condition(rows)
        rows = selection_records()
        rows[1]["target_identity"] = {"other": "target"}
        with self.assertRaisesRegex(ValueError, "target/native"):
            comparison.select_common_condition(rows)
        rows = selection_records()
        rows[1]["metrics"]["overall"]["count"] = 101
        with self.assertRaisesRegex(ValueError, "cohort counts"):
            comparison.select_common_condition(rows)

    def test_bad_metrics_and_empty_cohort_cannot_be_silently_dropped(self):
        for cohort, metric, value in (("overall", "qty_rmse", math.inf), ("overall", "qty_mae", -1.),
                ("overall", "time_nll", math.nan), ("overall", "count", True), ("tail", "qty_mae", 0.)):
            with self.subTest(cohort=cohort, metric=metric):
                rows = selection_records()
                rows[0]["metrics"][cohort][metric] = value
                with self.assertRaises(ValueError):
                    comparison.select_common_condition(rows)
        rows = selection_records()
        for row in rows:
            if row["time_input"] == "log1p" and row["loss_mode"] == "log_mse":
                row["metrics"]["overall"]["qty_mae"] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            comparison.select_common_condition(rows)


class AggregateTests(unittest.TestCase):
    def test_actual_receipt_metric_metadata_is_preserved_in_endpoint_records(self):
        rows = with_metric_metadata(arm_records())
        result = comparison.aggregate(rows, comparison.REFERENCE_CONDITION)
        self.assertEqual(result["records"], rows)
        self.assertEqual(len(result["records"]), 27)
        self.assertTrue(all(row["metrics"]["time_metric"] == "recorded_positive_integer_time_nll"
                            for row in result["records"]))
        rows[1]["metrics"].pop("cap_value")
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            comparison.aggregate(rows, comparison.REFERENCE_CONDITION)

    def test_mean_sample_sd_all_contrasts_and_empty_tail(self):
        rows = arm_records()
        before = deepcopy(rows)
        result = comparison.aggregate(rows, comparison.REFERENCE_CONDITION)
        self.assertEqual((len(result["summary"]), len(result["seed_differences"]), len(result["decisions"])), (81, 243, 9))
        summary = next(r for r in result["summary"] if r["model"] == comparison.OBSERVATION
                       and r["cohort"] == "overall" and r["metric"] == "qty_rmse")
        self.assertEqual((summary["mean"], summary["sample_sd_ddof1"]), (1., .5))
        difference = next(r for r in result["seed_differences"] if r["candidate_model"] == comparison.OBSERVATION
                          and r["reference_model"] == comparison.BASELINE and r["seed"] == 52
                          and r["cohort"] == "overall" and r["metric"] == "qty_rmse")
        self.assertEqual((difference["absolute_difference"], difference["relative_difference_percent"], difference["worsened"]),
                         (-1., -50., False))
        self.assertTrue(result["observation_improves_both_vs_baseline_and_hidden_all_datasets"])
        self.assertTrue(all(r["mean"] is None and r["sample_sd_ddof1"] is None
                            for r in result["summary"] if r["cohort"] == "tail"))
        self.assertEqual(rows, before)

    def test_all_seed_exceptions_survive_favorable_averages(self):
        rows = arm_records()
        row = next(r for r in rows if r["dataset"] == comparison.DATASETS[0] and r["seed"] == 42
                   and r["model"] == comparison.OBSERVATION)
        row["metrics"]["overall"]["qty_mae"] = .6
        result = comparison.aggregate(rows, comparison.REFERENCE_CONDITION)
        decision = next(r for r in result["decisions"] if r["dataset"] == comparison.DATASETS[0]
                        and r["candidate_model"] == comparison.OBSERVATION and r["reference_model"] == comparison.BASELINE)
        self.assertTrue(decision["both_quantity_means_improved"])
        self.assertFalse(decision["all_three_seeds_improved_both"])
        self.assertTrue(any(r["worsened"] for r in result["seed_differences"]))

    def test_zero_reference_relative_delta_na_and_time_nll_diagnostic(self):
        rows = arm_records()
        rows[0]["metrics"]["overall"]["qty_mae"] = 0.
        result = comparison.aggregate(rows, comparison.REFERENCE_CONDITION)
        row = next(r for r in result["seed_differences"] if r["seed"] == 42
                   and r["candidate_model"] == comparison.OBSERVATION and r["reference_model"] == comparison.BASELINE
                   and r["cohort"] == "overall" and r["metric"] == "qty_mae")
        self.assertIsNone(row["relative_difference_percent"])
        self.assertTrue(row["worsened"])
        self.assertTrue(all(row["time_metric_diagnostic_only"] for row in result["summary"] if row["metric"] == "time_nll"))

    def test_missing_duplicate_mixed_condition_and_identity_rejected(self):
        for rows in (arm_records()[:-1], arm_records() + arm_records()[:1]):
            with self.assertRaisesRegex(ValueError, "27-endpoint"):
                comparison.aggregate(rows, comparison.REFERENCE_CONDITION)
        for key, value in (("time_input", "raw"), ("loss_mode", "raw_mse"), ("host", "5090"),
                           ("time_head_trained", True), ("target_identity", {"other": 1})):
            rows = with_metric_metadata(arm_records())
            rows[1][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    comparison.aggregate(rows, comparison.REFERENCE_CONDITION)

    def test_arm_alias_and_selected_condition_wrapper(self):
        rows = arm_records()
        for row in rows:
            row["arm"] = row.pop("model")
        result = comparison.aggregate(rows, {"selected_condition": comparison.REFERENCE_CONDITION})
        self.assertEqual(result["common_condition"], comparison.REFERENCE_CONDITION)
        rows[0]["model"] = comparison.OBSERVATION
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            comparison.aggregate(rows, comparison.REFERENCE_CONDITION)

    def test_randomized_summary_and_differences_match_independent_arithmetic(self):
        rng = random.Random(19)
        rows = arm_records()
        for row in rows:
            for cohort in ("overall", "cap_hit"):
                for metric in comparison.METRICS:
                    row["metrics"][cohort][metric] = rng.random() * 30.
        result = comparison.aggregate(rows, comparison.REFERENCE_CONDITION)
        for summary in result["summary"]:
            if summary["cohort"] == "tail":
                continue
            values = [r["metrics"][summary["cohort"]][summary["metric"]] for r in rows
                      if r["dataset"] == summary["dataset"] and r["model"] == summary["model"]]
            self.assertEqual(summary["mean"], statistics.mean(values))
            self.assertEqual(summary["sample_sd_ddof1"], statistics.stdev(values))
        for delta in result["seed_differences"]:
            if delta["cohort"] == "tail":
                continue
            x, y = (next(r for r in rows if r["dataset"] == delta["dataset"] and r["seed"] == delta["seed"]
                         and r["model"] == delta[key])["metrics"][delta["cohort"]][delta["metric"]]
                    for key in ("candidate_model", "reference_model"))
            self.assertEqual(delta["absolute_difference"], x - y)


class MainTests(unittest.TestCase):
    def test_local_verified_archive_reporting_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = ModuleType("paper.scripts.manage_titantpp_late_path18")
            module.BUNDLE, module.REPORT = root / "bundle", root / "reports"
            module.sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            module.read = lambda path: json.loads(Path(path).read_text())
            def write(path, payload):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as stream:
                    json.dump(payload, stream, allow_nan=False, sort_keys=True)
            module.write = write
            module.runtime = SimpleNamespace(local_path=lambda base, name: Path(base) / name)
            rows = with_metric_metadata(arm_records())
            jobs, refs, terminal_by_host = [], [], {}
            for index, row in enumerate(rows):
                if row["model"] == comparison.BASELINE:
                    path = root / "references" / f"{index}.json"
                    write(path, row)
                    refs.append({**row, "path": str(path), "sha256": module.sha(path)})
                    continue
                job = {**row, "arm": row["model"], "id": f"job{index}"}
                jobs.append(job)
            contract = {"evaluation_scope": "validation_only", "held_out_test_authorized": False,
                "common_condition": deepcopy(comparison.REFERENCE_CONDITION), "jobs": jobs,
                "baseline_validation_references": refs}
            module.verify_local = lambda: contract
            module.baseline_validation_references = lambda: refs
            for host in ("5080", "5090"):
                retrieval = module.BUNDLE / "retrieval" / host
                originals = retrieval / "originals"
                inventory = {}
                local_jobs = [job for job in jobs if job["host"] == host]
                for job in local_jobs:
                    receipt = {**job, "status": "complete", "validation_replay_passed": True,
                        "held_out_test_evaluated": False, "time_input_transform": "log1p_nonnegative_float32",
                        "quantity_input_transform": "log1p_nonnegative_float32"}
                    path = originals / "run" / job["id"] / "validation_receipt.json"
                    write(path, receipt)
                    inventory[str(path.relative_to(originals))] = module.sha(path)
                write(originals / "dispatcher_exit.json", {"actual_os_returncode": 0})
                inventory["dispatcher_exit.json"] = module.sha(originals / "dispatcher_exit.json")
                terminal_by_host[host] = {"stages": {stage: {"exits": {
                    job["id"]: {"actual_os_returncode": 0, "timeout": False} for job in local_jobs}}
                    for stage in ("train", "validation")}}
                write(retrieval / "manifest.json", {"files": inventory})
                (retrieval / "original.tar.gz").write_bytes(b"synthetic archive; collector is mocked")
                write(retrieval / "receipt.json", {"status": "complete", "file_count": len(inventory),
                    "archive_sha256": module.sha(retrieval / "original.tar.gz"),
                    "manifest_sha256": module.sha(retrieval / "manifest.json")})
            module.verify_collected = lambda contract, host, originals: terminal_by_host[host]
            with patch.dict(sys.modules, {module.__name__: module}), patch.object(
                    scripts_package, "manage_titantpp_late_path18", module, create=True), patch(
                    "paper.scripts.compare_titantpp_time_zero_mlp9.verify_zero_proof", return_value={"synthetic": True}):
                result = comparison.main()
                receipt_path = module.BUNDLE / "retrieval/final_receipt.json"
                receipt = module.read(receipt_path)
                self.assertEqual((receipt["new_fit_count"], receipt["baseline_reference_count"],
                    receipt["comparison_model_count"], receipt["actual_scientific_child_exit0_count"],
                    receipt["actual_dispatcher_exit0_count"]), (18, 9, 27, 36, 2))
                self.assertFalse(receipt["held_out_test_evaluated"])
                self.assertEqual(len(receipt["report_files"]), 4)
                self.assertEqual(len(result["records"]), 27)
                self.assertTrue(all(row["metrics"]["cap_hit_rule"] == "q == cap_value"
                                    and row["metrics"]["tail_rule"] == "q > cap_value"
                                    for row in result["records"]))
                original_receipt = receipt_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "no overwrite"):
                    comparison.main()
                self.assertEqual(receipt_path.read_bytes(), original_receipt)


if __name__ == "__main__":
    unittest.main()
