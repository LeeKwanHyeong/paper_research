"""실제 학습·원격·데이터 행 접근 없이 준비 계약의 변조 차단을 검사한다."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import prepare_titantpp_gru_residual_cap as prepare


def fixture():
    parent = {
        "source": {"files": {"frozen_science.py": "a" * 64}, "git_revision": "old_revision", "base_git_revision": "old_base"},
        "training": {"batch_size": 128, "minimum_epochs": 40, "maximum_epochs": 300,
                     "patience": 40, "monitor": "validation_raw_quantity_rmse",
                     "tie": "strict_earliest_finite_minimum", "warm_start": False},
        "limits": {"per_condition_seconds": 129600, "total_wall_seconds": 604800,
                   "workers_per_host": 3, "automatic_retry": False},
        "input_files": {"inputs/train_validation.parquet": "b" * 64},
        "datasets": [], "hosts": {},
    }
    for dataset in prepare.DATASETS:
        parent["datasets"].append({"dataset_id": dataset,
            "loader": {"max_seq_len": 84 if dataset == "raf_spare_parts" else 256,
                       "batch_size": 128, "drop_last": False, "num_workers": 0},
            "model": {"hidden_dim": 64, "lambda_log_qty": 1., "lambda_tail": 0.,
                      "time_head_lr_multiplier": 1., "time_head_mode": "heteroscedastic_lognormal_duration"},
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.},
            "statistics": {"train_log_mean": .5, "train_log_std": .2}})
    for host in ("5080", "5090"):
        parent["hosts"][host] = {"python": f"/native/{host}/python3.12", "gpu_uuid": f"GPU-{host}",
            "runtime_expected": {"cudnn": 91900 if host == "5080" else 92000}}
    refs = []
    for dataset in prepare.DATASETS:
        count = 104159 if dataset == "raf_spare_parts" else 115167
        for seed in prepare.SEEDS:
            for arm in (prepare.CNN, prepare.FULL):
                refs.append({"id": f"{dataset}__{seed}__{arm}", "dataset": dataset, "seed": seed,
                    "arm": arm, "source_closure": prepare.SOURCE131,
                    "training_host": prepare.planned_host(dataset, seed), "retraining": False,
                    "replacement_of_original_FULL": False, "selected_epoch": 13,
                    "checkpoint_sha256": "c" * 64, "history_sha256": "d" * 64,
                    "metrics": {key: 1. for key in prepare.ANALYSIS["metrics"]},
                    "physical_parameter_counts": {"actual_registered_parameters": count,
                        "actual_active_parameters_before_evaluation_freeze": count if arm == prepare.FULL else count - 24732}})
    refs.sort(key=lambda item: item["id"])
    extension = {path: "e" * 64 for path in prepare.EXTENSION_PATHS}
    contract = prepare.make_contract(parent, refs, extension,
        prepare.PROJECT / "search_artifacts/titantpp_gru_residual_cap_validation_synthetic")
    return parent, refs, contract


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.refs, self.contract = fixture()

    def validate(self, contract):
        return prepare.validate_contract(contract, self.parent, expected_references=self.refs)

    def test_nine_fresh_fits_eighteen_immutable_references_no_execution(self):
        self.validate(self.contract)
        self.assertEqual((len(self.contract["jobs"]), len(self.contract["baseline_references"])), (9, 18))
        self.assertFalse(self.contract["authorization"]["execution_authorized"])
        self.assertEqual(self.contract["authorization"]["training_permits"], [])
        self.assertEqual(self.contract["started_fit_count"], 0)
        self.assertFalse(self.contract["test_evaluation"])

    def test_dataset_parameter_counts_and_host_remain_distinct(self):
        self.validate(self.contract)
        for job in self.contract["jobs"]:
            self.assertEqual(job["expected_registered_parameters"],
                             104159 if job["dataset"] == "raf_spare_parts" else 115167)
            self.assertEqual(job["expected_registered_parameters"], job["expected_active_parameters"])
            self.assertEqual(job["host"], prepare.planned_host(job["dataset"], job["seed"]))

    def test_candidate_metadata_is_the_model_definition_and_identity(self):
        value = self.contract["candidate"]
        self.assertEqual(value["metadata"], prepare.model_metadata(64, arm=prepare.CANDIDATE, rho=.25))
        self.assertEqual(value["identity_buffer_hex"],
                         hashlib.sha256(json.dumps(value["metadata"], sort_keys=True).encode()).hexdigest())
        changed = deepcopy(self.contract)
        changed["candidate"]["metadata"]["residual_cap_rounding"] = "no_guard"
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            self.validate(changed)

    def test_cap_rho_norm_reference_or_added_parameter_cannot_change(self):
        for key, value in (("rho", .5), ("norm_dtype", "float32"), ("reference", "attached_base"),
                           ("epsilon_floor", True), ("new_trainable_parameters", 1), ("sweep", True)):
            with self.subTest(key=key):
                changed = deepcopy(self.contract)
                changed["candidate"][key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(changed)

    def test_budget_loss_head_loader_or_train_statistics_cannot_change(self):
        for area, key, value in (("training", "maximum_epochs", 301), ("training", "patience", 20),
                                  ("limits", "per_condition_seconds", 172800),
                                  ("objective_weights", "log_quantity", 2.)):
            with self.subTest(area=area, key=key):
                changed = deepcopy(self.contract)
                changed[area][key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(changed)
        for area, key, value in (("model", "time_head_mode", "new_head"), ("loader", "max_seq_len", 999),
                                  ("optimizer", "lr", .01), ("statistics", "train_log_mean", 999.)):
            with self.subTest(area=area, key=key):
                changed = deepcopy(self.contract)
                changed["datasets"][0][area][key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(changed)

    def test_test_scope_seed_expansion_or_duplicate_job_cannot_pass(self):
        changes = [("test_evaluation", True), ("evaluation_scope", "test"), ("seeds", [42, 52, 72]),
                   ("held_out_test_evaluated", True), ("started_fit_count", 1)]
        for key, value in changes:
            with self.subTest(key=key):
                changed = deepcopy(self.contract)
                changed[key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(changed)
        changed = deepcopy(self.contract)
        changed["jobs"][-1] = deepcopy(changed["jobs"][0])
        with self.assertRaises(RuntimeError):
            self.validate(changed)

    def test_approval_runtime_or_host_fabrication_cannot_pass(self):
        for key, value in (("execution_authorized", True), ("training_permits", ["fake"]),
                           ("new_5090_training_authorized", True), ("remote_calls", 1)):
            with self.subTest(key=key):
                changed = deepcopy(self.contract)
                changed["authorization"][key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(changed)
        for key, value in (("live_runtime_queried", True), ("training_authorized", True),
                           ("gpu_uuid", "GPU-other"), ("python", "/other/python")):
            with self.subTest(key=key):
                changed = deepcopy(self.contract)
                changed["hosts"]["5080"][key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(changed)
        changed = deepcopy(self.contract)
        changed["jobs"][0]["host"] = "5090"
        with self.assertRaises(RuntimeError):
            self.validate(changed)

    def test_parent_source_and_extension_list_are_frozen(self):
        for action in ("parent", "closure", "extra", "bad_sha"):
            with self.subTest(action=action):
                changed = deepcopy(self.contract)
                source = changed["source"]
                if action == "parent":
                    source["parent_files"]["frozen_science.py"] = "f" * 64
                elif action == "closure":
                    source["files_sha256"] = "f" * 64
                elif action == "extra":
                    source["extension_files"]["another.py"] = "f" * 64
                else:
                    path = prepare.EXTENSION_PATHS[0]
                    source["extension_files"][path] = "bad"
                    source["files"][path] = "bad"
                    source["files_sha256"] = prepare.canonical(source["files"])
                with self.assertRaises(RuntimeError):
                    self.validate(changed)

    def test_reference_replacement_cannot_be_hidden_by_rehashing(self):
        changed = deepcopy(self.contract)
        changed["baseline_references"][0]["checkpoint_sha256"] = "f" * 64
        changed["baseline_references_sha256"] = prepare.canonical(changed["baseline_references"])
        with self.assertRaisesRegex(RuntimeError, "원래 기준선"):
            self.validate(changed)
        changed = deepcopy(self.contract)
        changed["baseline_references"][0]["metrics"]["whole_qty_rmse"] = 0.
        changed["baseline_references_sha256"] = prepare.canonical(changed["baseline_references"])
        with self.assertRaisesRegex(RuntimeError, "원래 기준선"):
            self.validate(changed)

    def test_selection_uses_first_strict_finite_minimum_and_rejects_bad_history(self):
        history = {"history": [{"epoch": 1, "val_qty_rmse": float("nan")},
            {"epoch": 2, "val_qty_rmse": 2.}, {"epoch": 3, "val_qty_rmse": 1.},
            {"epoch": 4, "val_qty_rmse": 1.}]}
        self.assertEqual(prepare.earliest_selection(history), (3, 1.))
        for rows in ([], [{"epoch": 1, "val_qty_rmse": float("inf")}],
                     [{"epoch": 2, "val_qty_rmse": 1.}, {"epoch": 1, "val_qty_rmse": 0.}],
                     [{"epoch": 1, "val_qty_rmse": 1.}, {"epoch": 1, "val_qty_rmse": 0.}]):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                prepare.earliest_selection({"history": rows})

    def test_output_is_exclusive_and_confined(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "search_artifacts").mkdir()
            target = root / "search_artifacts/titantpp_gru_residual_cap_validation_synthetic"
            with patch.object(prepare, "PROJECT", root):
                self.assertEqual(prepare.validate_output_path(target), target.resolve())
                target.mkdir()
                with self.assertRaisesRegex(RuntimeError, "덮어쓰"):
                    prepare.validate_output_path(target)
                with self.assertRaises(RuntimeError):
                    prepare.validate_output_path(root / "outside/titantpp_gru_residual_cap_validation_new")
                link = root / "search_artifacts/titantpp_gru_residual_cap_validation_link"
                link.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, "symbolic"):
                    prepare.validate_output_path(link)
                with self.assertRaises(RuntimeError):
                    prepare.confined(root, "../outside.py")

    def test_sha_mismatch_or_changed_reference_leaves_no_bundle(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "search_artifacts").mkdir()
            target = root / "search_artifacts/titantpp_gru_residual_cap_validation_synthetic"
            with patch.object(prepare, "PROJECT", root), patch.object(prepare, "read_parent",
                    side_effect=RuntimeError("원본 SHA 불일치")):
                with self.assertRaisesRegex(RuntimeError, "SHA"):
                    prepare.prepare(target)
                self.assertFalse(target.exists())
            original = root / "metadata.json"
            original.write_text("original")
            evidence = prepare.Evidence()
            evidence.verify(original, prepare.sha(original))
            original.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "변경"):
                evidence.recheck()


if __name__ == "__main__":
    unittest.main()
