"""Synthetic three-loss common-condition gate and separate v2 local reporting."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import paper.scripts as scripts_package
from paper.scripts import select_compare_titantpp_late_path18_v2 as comparison


def metrics(value, index, dataset):
    return {"overall": {"count": 100, "qty_rmse": value, "qty_mae": value / 2., "time_nll": index - 2.},
        "cap_hit": {"count": 10, "qty_rmse": value * 2., "qty_mae": value, "time_nll": index / 3.},
        "tail": {"count": 0, "qty_rmse": None, "qty_mae": None, "time_nll": None},
        **comparison.METRIC_METADATA, "cap_value": comparison.CAP_VALUES[dataset]}


def selection_records():
    rows = []
    for dataset, scale in zip(comparison.DATASETS, (100., 10., 1.)):
        for index, seed in enumerate(comparison.SEEDS, 1):
            for loss in comparison.LOSSES:
                rows.append({"dataset": dataset, "host": comparison._impl._expected_host(dataset, seed),
                    "seed": seed, "time_input": "log1p", "loss_mode": loss, "time_alpha": 0.,
                    "time_head_trained": False, "selected_epoch": index + 1,
                    "target_identity": {"dataset": dataset, "target_hash": str(seed)},
                    "scope": "validation_only", "held_out_test_evaluated": False,
                    "time_input_transform": "log1p_nonnegative_float32",
                    "quantity_input_transform": "log1p_nonnegative_float32",
                    "metrics": metrics(scale * index * (.5 if loss == "raw_mae" else 1.), index, dataset)})
    return rows


def arm_records():
    rows = []
    for row in selection_records():
        if row["loss_mode"] != "log_mse":
            continue
        for model, factor in ((comparison.BASELINE, 1.), (comparison.OBSERVATION, .5), (comparison.HIDDEN, .75)):
            copy = deepcopy(row)
            copy["model"] = model
            for cohort in ("overall", "cap_hit"):
                for metric in comparison.METRICS[:2]:
                    copy["metrics"][cohort][metric] *= factor
            rows.append(copy)
    return rows


class SelectionTests(unittest.TestCase):
    def test_exact27_and_all_three_scores_preserve_all_metrics_and_metadata(self):
        rows = selection_records()
        before = deepcopy(rows)
        result = comparison.select_common_condition(rows)
        self.assertEqual(result["selected_condition"], {"time_input": "log1p", "loss_mode": "raw_mae", "time_alpha": 0.})
        self.assertEqual((result["input_endpoint_count"], result["condition_count"], len(result["condition_scores"])), (27, 3, 3))
        self.assertEqual(result["selected_scores"], {"macro_rmse_ratio": .5, "macro_mae_ratio": .5})
        self.assertEqual(result["records"], before)
        self.assertEqual(rows, before)
        self.assertEqual(result["raw_time_endpoints_used"], 0)
        self.assertEqual(sum(len(ds["seed_records"]) for score in result["condition_scores"] for ds in score["datasets"]), 27)
        self.assertTrue(all(row["metrics"]["time_metric"] == "recorded_positive_integer_time_nll"
                            for row in result["records"]))

    def test_dataset_ratios_not_pooled_raw_errors(self):
        rows = selection_records()
        for row in rows:
            if row["loss_mode"] == "raw_mae":
                ratio = .5 if row["dataset"] == comparison.DATASETS[0] else 1.5
                row["metrics"]["overall"]["qty_rmse"] *= ratio * 2.
        result = comparison.select_common_condition(rows)
        self.assertEqual(result["selected_condition"], comparison.REFERENCE_CONDITION)
        self.assertAlmostEqual(result["condition_scores"][2]["macro_rmse_ratio"], 7. / 6.)

    def test_mae_secondary_then_fixed_loss_order_without_tolerance(self):
        rows = selection_records()
        for row in rows:
            if row["loss_mode"] == "raw_mae":
                row["metrics"]["overall"]["qty_rmse"] *= 2.
        self.assertEqual(comparison.select_common_condition(rows)["selected_condition"]["loss_mode"], "raw_mae")
        for row in rows:
            if row["loss_mode"] == "raw_mae":
                row["metrics"]["overall"]["qty_mae"] *= 2.
        random.Random(2).shuffle(rows)
        self.assertEqual(comparison.select_common_condition(rows)["selected_condition"]["loss_mode"], "log_mse")
        for row in rows:
            if row["loss_mode"] == "raw_mse":
                row["metrics"]["overall"]["qty_rmse"] *= 1. - 1e-12
        self.assertEqual(comparison.select_common_condition(rows)["selected_condition"]["loss_mode"], "raw_mse")

    def test_missing_duplicate54_and_raw_time_are_rejected(self):
        for rows in (selection_records()[:-1], selection_records() + selection_records()[:1], selection_records() * 2):
            with self.assertRaisesRegex(ValueError, "log-time27"):
                comparison.select_common_condition(rows)
        rows = selection_records()
        rows[0]["time_input"] = "raw"
        with self.assertRaises(ValueError):
            comparison.select_common_condition(rows)

    def test_scope_targets_host_and_input_transforms_are_fixed(self):
        for key, value in (("scope", "test"), ("held_out_test_evaluated", True), ("host", "5090"),
                ("time_alpha", 1.), ("time_head_trained", True), ("selected_epoch", 0),
                ("quantity_input_transform", "raw_nonnegative_float32"),
                ("time_input_transform", "raw_nonnegative_float32"), ("model", comparison.OBSERVATION)):
            rows = selection_records()
            rows[0][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    comparison.select_common_condition(rows)
        rows = selection_records()
        rows[1]["target_identity"] = {"different": True}
        with self.assertRaisesRegex(ValueError, "target/native"):
            comparison.select_common_condition(rows)

    def test_metric_finiteness_cap_definition_and_zero_reference(self):
        for key, value in (("cap_value", 999.), ("cap_hit_rule", "q >= cap_value"),
                           ("time_metric", "duration_mae")):
            rows = selection_records()
            rows[0]["metrics"][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    comparison.select_common_condition(rows)
        for value in (math.nan, math.inf, True, -1.):
            rows = selection_records()
            rows[0]["metrics"]["overall"]["qty_rmse"] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    comparison.select_common_condition(rows)
        rows = selection_records()
        for row in rows:
            if row["loss_mode"] == "log_mse":
                row["metrics"]["overall"]["qty_mae"] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            comparison.select_common_condition(rows)


class AggregationTests(unittest.TestCase):
    def test_same_three_arm_aggregation_preserves_metadata_and_seed_exceptions(self):
        rows = arm_records()
        rows[1]["metrics"]["overall"]["qty_mae"] = rows[0]["metrics"]["overall"]["qty_mae"] + .1
        result = comparison.aggregate(rows, comparison.REFERENCE_CONDITION)
        self.assertEqual((len(result["summary"]), len(result["seed_differences"]), len(result["decisions"])), (81, 243, 9))
        self.assertEqual(result["records"], rows)
        self.assertTrue(any(row["worsened"] for row in result["seed_differences"]))
        self.assertTrue(all(row["mean"] is None for row in result["summary"] if row["cohort"] == "tail"))
        self.assertTrue(all(row["time_metric_diagnostic_only"] for row in result["summary"] if row["metric"] == "time_nll"))

    def test_both_aggregate_condition_and_endpoint_must_keep_log_inputs(self):
        rows = arm_records()
        with self.assertRaises(ValueError):
            comparison.aggregate(rows, {**comparison.REFERENCE_CONDITION, "time_input": "raw"})
        rows[1]["quantity_input_transform"] = "raw_nonnegative_float32"
        with self.assertRaisesRegex(ValueError, "remain log1p"):
            comparison.aggregate(rows, comparison.REFERENCE_CONDITION)


class MainTests(unittest.TestCase):
    def test_main_uses_new_manager_only_and_preserves_three_score_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = ModuleType("paper.scripts.manage_titantpp_late_path18_v2")
            module.BUNDLE, module.REPORT = root / "bundle_v2", root / "reports_v2"
            module.sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            module.read = lambda path: json.loads(Path(path).read_text())
            def write(path, payload):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as stream:
                    json.dump(payload, stream, allow_nan=False, sort_keys=True)
            module.write = write
            module.runtime = SimpleNamespace(local_path=lambda base, name: Path(base) / name)
            jobs, refs, terminal_by_host = [], [], {}
            rows = arm_records()
            for index, row in enumerate(rows):
                if row["model"] == comparison.BASELINE:
                    path = root / "references" / f"{index}.json"
                    write(path, row)
                    refs.append({**row, "path": str(path), "sha256": module.sha(path)})
                else:
                    jobs.append({**row, "arm": row["model"], "id": f"job{index}"})
            selector_inputs = selection_records()
            for row in selector_inputs:
                if row["loss_mode"] == "raw_mae":
                    for metric in comparison.METRICS[:2]:
                        row["metrics"]["overall"][metric] *= 2.
            selection = comparison.select_common_condition(selector_inputs)
            self.assertEqual(selection["selected_condition"], comparison.REFERENCE_CONDITION)
            contract = {"evaluation_scope": "validation_only", "held_out_test_authorized": False,
                "common_condition": deepcopy(comparison.REFERENCE_CONDITION), "common_condition_selection": selection,
                "jobs": jobs, "baseline_validation_references": refs}
            module.verify_local = lambda: contract
            module.baseline_validation_references = lambda: refs
            for host in ("5080", "5090"):
                retrieval = module.BUNDLE / "retrieval" / host
                originals = retrieval / "originals"
                inventory = {}
                local_jobs = [job for job in jobs if job["host"] == host]
                for job in local_jobs:
                    receipt = {**job, "status": "complete", "validation_replay_passed": True, "held_out_test_evaluated": False}
                    path = originals / "run" / job["id"] / "validation_receipt.json"
                    write(path, receipt)
                    inventory[str(path.relative_to(originals))] = module.sha(path)
                write(originals / "dispatcher_exit.json", {"actual_os_returncode": 0})
                inventory["dispatcher_exit.json"] = module.sha(originals / "dispatcher_exit.json")
                terminal_by_host[host] = {"stages": {stage: {"exits": {
                    job["id"]: {"actual_os_returncode": 0, "timeout": False} for job in local_jobs}}
                    for stage in ("train", "validation")}}
                write(retrieval / "manifest.json", {"files": inventory})
                (retrieval / "original.tar.gz").write_bytes(b"synthetic; collector mocked")
                write(retrieval / "receipt.json", {"status": "complete", "file_count": len(inventory),
                    "archive_sha256": module.sha(retrieval / "original.tar.gz"),
                    "manifest_sha256": module.sha(retrieval / "manifest.json")})
            module.verify_collected = lambda contract, host, originals: terminal_by_host[host]
            forbidden_old = SimpleNamespace(verify_local=lambda: self.fail("Old manager must not be used"))
            with patch.dict(sys.modules, {module.__name__: module}), patch.object(
                    scripts_package, "manage_titantpp_late_path18_v2", module, create=True), patch.object(
                    scripts_package, "manage_titantpp_late_path18", forbidden_old, create=True), patch(
                    "paper.scripts.compare_titantpp_time_zero_mlp9.verify_zero_proof", return_value={"synthetic": True}):
                result = comparison.main()
                final = module.read(module.BUNDLE / "retrieval/final_receipt.json")
                self.assertEqual((final["new_fit_count"], final["baseline_reference_count"],
                    final["comparison_model_count"], final["actual_scientific_child_exit0_count"],
                    final["actual_dispatcher_exit0_count"]), (18, 9, 27, 36, 2))
                self.assertFalse(final["held_out_test_evaluated"])
                self.assertEqual(len(result["common_condition_selection"]["condition_scores"]), 3)
                self.assertEqual(len(result["records"]), 27)
                self.assertTrue(all(row["metrics"]["cap_hit_rule"] == "q == cap_value" for row in result["records"]))
                self.assertEqual(len(final["report_files"]), 4)
                with self.assertRaisesRegex(ValueError, "no overwrite"):
                    comparison.main()


if __name__ == "__main__":
    unittest.main()
