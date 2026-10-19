"""Independent numerical and contract checks; fabricated endpoints, no inference."""
from copy import deepcopy
import math
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper.scripts import compare_titantpp_shared_history_cap_execution as comparison


def records():
    rows = []
    for dataset in comparison.DATASETS:
        for seed, offset in zip(comparison.SEEDS, (0.0, 2.0, 4.0)):
            for model, delta in zip(comparison.MODELS, (0.0, -1.0, 0.5)):
                value = 10.0 + offset + delta
                rows.append({"dataset": dataset, "seed": seed, "model": model,
                    "host": "5090" if model == comparison.BASELINE else "5080",
                    **comparison.CONDITION, "scope": "validation_only", "held_out_test_evaluated": False,
                    "time_head_trained": False, "selected_epoch": seed // 10,
                    "target_identity": {"target_count": 10, "target_identity_sha256": "1" * 64,
                                        "target_quantity_sha256": "2" * 64},
                    "metrics": {**comparison.DEFINITIONS, "cap_value": comparison.CAPS[dataset],
                        "overall": {"count": 10, "qty_rmse": value, "qty_mae": value / 2, "time_nll": value / 4},
                        "cap_hit": {"count": 3, "qty_rmse": value + 1, "qty_mae": value / 2 + 1, "time_nll": value / 4 + 1},
                        "tail": {"count": 0, "qty_rmse": None, "qty_mae": None, "time_nll": None}}})
    return rows


class ComparisonTests(unittest.TestCase):
    def test_hand_calculated_mean_sample_sd_and_all_counts(self):
        result = comparison.aggregate(records())
        row = next(r for r in result["summary"] if r["model"] == comparison.SHARED
                   and r["cohort"] == "overall" and r["metric"] == "qty_rmse")
        self.assertEqual((row["mean"], row["sample_sd_ddof1"]), (11.0, 2.0))
        self.assertEqual(len(result["summary"]), 81)
        self.assertEqual(len(result["seed_differences"]), 243)
        self.assertEqual(len(result["decisions"]), 9)
        self.assertTrue(result["shared_improves_both_vs_baseline_and_control_all_datasets"])

    def test_differences_relative_and_recorded_cross_host(self):
        result = comparison.aggregate(records())
        difference = result["seed_differences"][0]
        self.assertEqual(difference["absolute_difference"], -1.0)
        self.assertEqual(difference["relative_difference_percent"], -10.0)
        self.assertEqual((difference["candidate_host"], difference["reference_host"]), ("5080", "5090"))
        self.assertFalse(difference["worsened"])
        control = next(r for r in result["seed_differences"] if r["candidate_model"] == comparison.CONTROL
                       and r["cohort"] == "overall")
        self.assertTrue(control["worsened"])

    def test_empty_tail_all_na(self):
        result = comparison.aggregate(records())
        tails = [r for r in result["summary"] if r["cohort"] == "tail"]
        self.assertTrue(all(r["mean"] is None and r["sample_sd_ddof1"] is None for r in tails))
        self.assertTrue(all(r["absolute_difference"] is None and r["worsened"] is None
                            for r in result["seed_differences"] if r["cohort"] == "tail"))

    def test_zero_reference_denominator_is_na_and_worsening_retained(self):
        data = records()
        data[0]["metrics"]["overall"]["qty_rmse"] = 0.0
        result = comparison.aggregate(data)
        delta = result["seed_differences"][0]
        self.assertEqual(delta["absolute_difference"], 9.0)
        self.assertIsNone(delta["relative_difference_percent"])
        self.assertTrue(delta["worsened"])
        self.assertFalse(result["decisions"][0]["all_three_seeds_improved_both"])

    def test_negative_time_nll_relative_uses_absolute_denominator(self):
        data = records()
        data[0]["metrics"]["overall"]["time_nll"] = -2.0
        data[1]["metrics"]["overall"]["time_nll"] = -1.0
        result = comparison.aggregate(data)
        delta = next(r for r in result["seed_differences"] if r["candidate_model"] == comparison.SHARED
                     and r["cohort"] == "overall" and r["metric"] == "time_nll")
        self.assertEqual(delta["relative_difference_percent"], 50.0)
        self.assertTrue(delta["time_metric_diagnostic_only"])

    def test_matrix_missing_duplicate_unknown_arm(self):
        for mutate in (lambda x: x.pop(), lambda x: x.append(deepcopy(x[0])),
                       lambda x: x[0].update(model="unknown"), lambda x: x[0].update(seed=43)):
            with self.subTest(mutation=mutate):
                data = records()
                mutate(data)
                with self.assertRaises(ValueError):
                    comparison.aggregate(data)

    def test_common_condition_and_scope_rejected(self):
        for field, value in (("time_input", "raw"), ("loss_mode", "raw_mse"), ("time_alpha", 1.0),
                             ("time_alpha", False), ("time_head_trained", True), ("scope", "test"),
                             ("held_out_test_evaluated", True), ("selected_epoch", 0), ("selected_epoch", True)):
            with self.subTest(field=field):
                data = records()
                data[0][field] = value
                with self.assertRaises(ValueError):
                    comparison.aggregate(data)

    def test_new_candidate_host_must_be_5080_baseline_host_preserved(self):
        data = records()
        data[1]["host"] = "5090"
        with self.assertRaisesRegex(ValueError, "5080"):
            comparison.aggregate(data)
        result = comparison.aggregate(records())
        self.assertTrue(all(r["host"] == "5090" for r in result["records"] if r["model"] == comparison.BASELINE))

    def test_target_mismatch_seed_identity_and_count_rejected(self):
        changes = (lambda r: r[1]["target_identity"].update(target_identity_sha256="3" * 64),
                   lambda r: [x["target_identity"].update(target_identity_sha256="4" * 64) for x in r[3:6]],
                   lambda r: r[1]["metrics"]["overall"].update(count=11),
                   lambda r: r[1]["target_identity"].update(target_count=True))
        for change in changes:
            with self.subTest(change=change):
                data = records()
                change(data)
                with self.assertRaises(ValueError):
                    comparison.aggregate(data)

    def test_metrics_nonfinite_negative_missing_and_false_count_rejected(self):
        for value in (math.inf, -math.inf, math.nan, -1.0, True, None):
            with self.subTest(value=value):
                data = records()
                data[1]["metrics"]["overall"]["qty_rmse"] = value
                with self.assertRaises(ValueError):
                    comparison.aggregate(data)
        for change in (lambda r: r["metrics"]["tail"].pop("qty_rmse"),
                       lambda r: r["metrics"]["cap_hit"].update(count=True),
                       lambda r: r["metrics"]["tail"].update(qty_mae=0.0)):
            data = records()
            change(data[1])
            with self.assertRaises(ValueError):
                comparison.aggregate(data)

    def test_cohort_definitions_and_population_rejected(self):
        for change in (lambda r: r["metrics"].update(cap_value=66.0),
                       lambda r: r["metrics"].update(cap_hit_rule="q >= cap_value"),
                       lambda r: r["metrics"]["cap_hit"].update(count=11),
                       lambda r: r["metrics"]["tail"].update(count=1, qty_rmse=0., qty_mae=0., time_nll=0.)):
            data = records()
            change(data[1])
            with self.assertRaises(ValueError):
                comparison.aggregate(data)

    def test_seed_exception_is_preserved_even_if_mean_improves(self):
        data = records()
        data[1]["metrics"]["overall"].update(qty_rmse=10.2, qty_mae=5.1)
        result = comparison.aggregate(data)
        row = result["decisions"][0]
        self.assertTrue(row["both_quantity_means_improved"])
        self.assertFalse(row["all_three_seeds_improved_both"])

    def test_pure_function_does_not_mutate_records_and_no_raw_pooling(self):
        data = records()
        before = deepcopy(data)
        result = comparison.aggregate(data)
        self.assertEqual(data, before)
        self.assertFalse(result["raw_errors_pooled"])
        self.assertFalse(result["held_out_test_evaluated"])
        self.assertEqual({r["dataset"] for r in result["summary"]}, set(comparison.DATASETS))

    def test_markdown_all_seed_differences_six_metrics_and_limitations(self):
        text = comparison.markdown(comparison.aggregate(records()))
        self.assertIn("상한 도달 TimeNLL", text)
        self.assertIn("표본SD(ddof1)", text)
        self.assertIn("5080/5090", text)
        self.assertIn("cuDNN91900/92000", text)
        self.assertIn("q>cap", text)
        self.assertIn("Test는 평가하지 않았다", text)
        self.assertEqual(sum(1 for line in text.splitlines() if " − " in line), 27)

    def test_completed_local_cli_binds_all_outputs_and_preserves_baseline_host(self):
        """The verifier is mocked; this checks comparator receipt wiring only."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, report = root / "bundle", root / "report"
            report.mkdir()
            original = bundle / "retrieval/5080/originals"
            def write(path, value):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as stream:
                    json.dump(value, stream, allow_nan=False)
            def sha(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            def read(path):
                return json.loads(Path(path).read_text())
            proof = {"initial_time_head_sha256": "a" * 64, "final_time_head_sha256": "a" * 64,
                "optimizer_steps_checked": 1, "time_head_parameter_names": ["v_t.weight"],
                "all_time_head_gradients_none": True, "optimizer_state_absent": True,
                "time_head_parameters_unchanged": True}
            jobs, references, inventory = [], [], {}
            for row in records():
                identifier = f"{row['dataset']}__{row['model']}__{row['seed']}"
                endpoint = {**row, "arm": row["model"], "status": "complete", "validation_replay_passed": True,
                            "quantity_only": True, "time_head_no_training_verification": proof,
                            "checkpoint_sha256": "b" * 64}
                if row["model"] == comparison.BASELINE:
                    reference = {**row, "arm": comparison.BASELINE}
                    for field, hash_field, value in (("selected_checkpoint", "checkpoint_sha256", {}),
                            ("manifest_path", "manifest_sha256", {}),
                            ("completion_receipt_path", "completion_receipt_sha256", {"status": "complete"})):
                        path = root / "baseline" / identifier / f"{field}.json"
                        write(path, value)
                        reference.update({field: str(path), hash_field: sha(path)})
                    endpoint["checkpoint_sha256"] = reference["checkpoint_sha256"]
                    path = root / "baseline" / identifier / "validation.json"
                    write(path, endpoint)
                    reference.update(path=str(path), sha256=sha(path))
                    references.append(reference)
                else:
                    jobs.append({"id": identifier, "arm": row["model"], "host": "5080",
                                 "dataset": row["dataset"], "seed": row["seed"], **comparison.CONDITION})
                    path = original / "run" / identifier / "validation_receipt.json"
                    write(path, endpoint)
                    inventory[str(path.relative_to(original))] = sha(path)
            path = original / "dispatcher_exit.json"
            write(path, {"actual_os_returncode": 0})
            inventory[str(path.relative_to(original))] = sha(path)
            retrieval = bundle / "retrieval/5080"
            (retrieval / "original.tar.gz").write_bytes(b"fabricated archive; verifier mocked")
            write(retrieval / "manifest.json", {"files": inventory})
            write(retrieval / "receipt.json", {"status": "complete", "archive_sha256": sha(retrieval / "original.tar.gz"),
                "manifest_sha256": sha(retrieval / "manifest.json"), "file_count": len(inventory)})
            contract = {"common_condition": comparison.CONDITION, "evaluation_scope": "validation_only",
                "held_out_test_authorized": False, "jobs": jobs, "baseline_validation_references": references}
            write(bundle / "execution_contract.json", contract)
            exits = {j["id"]: {"actual_os_returncode": 0, "timeout": False} for j in jobs}
            manager = SimpleNamespace(BUNDLE=bundle, REPORT=report, read=read, sha=sha, write=write,
                verify_local=lambda: contract, verify_collected=lambda *args: {"stages": {
                    stage: {"exits": exits} for stage in ("train", "validation")}},
                runtime=SimpleNamespace(local_path=lambda base, name: base / name))
            import paper.scripts
            with patch.dict(sys.modules, {"paper.scripts.manage_titantpp_shared_history_cap_execution": manager}), \
                 patch.object(paper.scripts, "manage_titantpp_shared_history_cap_execution", manager, create=True):
                result = comparison.main([])
                self.assertEqual(result["new_fit_count"], 18)
                self.assertEqual(result["baseline_reference_count"], 9)
                final = read(bundle / "retrieval/final_receipt.json")
                self.assertEqual(final["actual_scientific_worker_exit0_count"], 36)
                for path, digest in final["report_files"].items():
                    self.assertEqual(sha(path), digest)
                with self.assertRaisesRegex(ValueError, "no overwrite"):
                    comparison.main([])


if __name__ == "__main__":
    unittest.main()
