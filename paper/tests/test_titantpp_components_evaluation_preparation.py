"""Operational artifact binding tests using only temporary synthetic files.

These tests neither load a model nor establish readiness of actual recovered
originals. Synthetic lineage pins replace constants inside this process only.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper.scripts import prepare_titantpp_components_frozen_evaluation as prepare


class BindingFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.sources = {}
        for index in range(131):
            name = f"module_{index:03d}.py"
            self.sources[name] = self.write_bytes("source/" + name, f"synthetic source {index}\n".encode())
        self.source_closure = prepare.canonical(self.sources)
        self.definitions = {}
        self.design = {"schema": "titantpp_components_final_Test_design_v1",
                       "fresh_Test_condition_count": 36, "datasets": {}}
        self.bindings = {"schema": "titantpp_components_checkpoint_bindings_v1",
                         "sources": {"components": {"closure": self.source_closure,
                             "revision": prepare.REVISION, "root": "source", "files": self.sources}},
                         "datasets": {}, "jobs": []}
        for dataset in prepare.DATASETS:
            lookback, max_seq_len, time_scale, time_unit = {
                "yellow_trip_hourly": (168, 256, 1.0, "hour"),
                "intermittent_frozen_5000": (520, 256, 3.0, "week"),
                "raf_spare_parts": (84, 84, 6.0, "month")}[dataset]
            time_contract = {"mode": "positive_integer_round_clamp_v1", "unit": time_unit, "top_code": None}
            definition = {"dataset_id": dataset,
                          "quantity_boundaries_all_train_rows": [0, 5, prepare.TAILS[dataset]],
                          "inherited_data_identity": {"populations": {
                              "validation": {"target_count": 5}}},
                          "model": {"time_observation_contract": time_contract,
                                    "time_scale": time_scale, "architecture": "synthetic_frozen",
                                    "hidden_dim": 64, "quantity_variant": "count_only_log_regression",
                                    "time_head_mode": "heteroscedastic_lognormal_duration",
                                    "train_mean": 0.5, "train_std": 0.25},
                          "loader": {"batch_size": 128, "max_seq_len": max_seq_len,
                                     "lookback_weeks": lookback, "num_workers": 0,
                                     "drop_last": False, "shuffle": False}}
            self.definitions[dataset] = definition
            data = f"data/{dataset}.bin"
            train = f"data/{dataset}_train.bin"
            split = f"data/{dataset}_split.json"
            data_sha = self.write_bytes(data, b"synthetic opaque data; no real observations")
            train_sha = self.write_bytes(train, b"synthetic training data" if dataset == "raf_spare_parts"
                                         else b"synthetic opaque data; no real observations")
            split_sha = self.write_json(split, {"synthetic": True, "chronological": True})
            definition["inherited_data_identity"].update(
                data={"sha256": train_sha}, split_manifest={"sha256": split_sha})
            if dataset == "raf_spare_parts":
                definition["parent_data_identity"] = {"data": {"sha256": data_sha}}
            population = {"count": 5, "target_identity_sha256": "a" * 64,
                          "truth_sha256": "b" * 64, "time_sha256": "c" * 64}
            self.design["datasets"][dataset] = {
                "full_data_byte_sha256": data_sha, "expected_Test_target_count": 5,
                "split_manifest_sha256": split_sha,
                "time_observation_contract": time_contract, "time_scale": time_scale,
                "max_seq_len_including_target": max_seq_len, "lookback_seq_units": lookback}
            self.bindings["datasets"][dataset] = {
                "data_path": data, "data_sha256": data_sha,
                "training_data_path": train, "training_data_sha256": train_sha,
                "split_manifest_path": split, "split_manifest_sha256": split_sha,
                "tail_threshold": prepare.TAILS[dataset], "expected_test_count": 5,
                "definition": copy.deepcopy(definition),
                "validation_population": dict(population), "test_population": dict(population)}
        training = {"datasets": list(self.definitions.values()), "synthetic_contract": True}
        self.parent = prepare.canonical(training)
        self.bindings["training_contract_path"] = "evidence/training_contract.json"
        self.bindings["training_contract_sha256"] = self.write_json(
            self.bindings["training_contract_path"], training)
        monitor = {"all_training_terminal_verified": True,
                   "totals": {"total": 36, "completed": 36, "running": 0, "waiting": 0,
                              "unconfirmed": 0, "all_training_terminal_verified": True},
                   "parent_contract_sha256": self.parent,
                   "destination_contract_sha256": prepare.HANDOFF}
        self.bindings["training_monitor_path"] = "evidence/monitor.json"
        self.bindings["training_monitor_sha256"] = self.write_json("evidence/monitor.json", monitor)
        for dataset in prepare.DATASETS:
            for seed in prepare.SEEDS:
                for arm in prepare.ARMS:
                    self.bindings["jobs"].append(self.job(dataset, seed, arm))
        evaluator_path = "operational/evaluator.py"
        evaluator_files = {evaluator_path: self.write_bytes(evaluator_path, b"# synthetic adapter\n")}
        self.bindings["evaluator_integrity"] = {
            "files": evaluator_files, "closure": prepare.canonical(evaluator_files)}
        self.runtimes = {host: {"host": host, "gpu_uuid": prepare.GPU_UUIDS[host],
                          "cudnn": 91900 if host == "5080" else 92000,
                          "python": "/synthetic/bin/python", "python_version": "3.12.13",
                          "torch_num_threads": 1, "interop_threads": 1, "torch": "2.6.0",
                          "numpy": "2.0.1", "polars": "1.20.0",
                          "cuda": "12.8", "dtype": "float32",
                          "deterministic_algorithms": True, "TF32": False}
                         for host in ("5080", "5090")}

    def write_bytes(self, name, payload):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return prepare.sha(path)

    def write_json(self, name, value):
        return self.write_bytes(name, json.dumps(value, sort_keys=True, allow_nan=False).encode())

    def history(self, name, values=None):
        values = values or [9.0, 5.0, 5.0] + [6.0] * 37
        content = "epoch,val_qty_rmse\n" + "".join(
            f"{epoch},{value}\n" for epoch, value in enumerate(values, start=1))
        return self.write_bytes(name, content.encode())

    def job(self, dataset, seed, arm):
        identity = f"{dataset}__{seed}__{arm}"
        host = prepare.host_for(dataset, seed)
        origin = prepare.HANDOFF if dataset == "intermittent_frozen_5000" and seed == 62 else self.parent
        prefix = "originals/" + identity
        checkpoint = prefix + "/best_val_qty_rmse_model.pt"
        last = prefix + "/last_epoch_state.pt"
        checkpoint_sha = self.write_bytes(checkpoint, (identity + " selected").encode())
        last_sha = self.write_bytes(last, (identity + " last").encode())
        metadata_path = prefix + "/checkpoint_metadata.json"
        metadata = {"backbone": arm, "seed": seed, "best_epoch": 2,
                    "model_state_sha256": "d" * 64, "initial_state_sha256": "e" * 64,
                    "source_revision": prepare.REVISION,
                    "source_revision_history": [prepare.REVISION],
                    "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
                    "checkpoint_monitor_history_key": "val_qty_rmse",
                    "checkpoint_monitor": "validation_raw_quantity_rmse",
                    "checkpoint_selection": "best_validation_raw_quantity_rmse",
                    "selected_metric_value": 5.0}
        metadata_sha = self.write_json(metadata_path, metadata)
        history_path = prefix + "/history.csv"
        history_sha = self.history(history_path)
        terminal_path = prefix + "/terminal.json"
        terminal_job = {"id": identity, "host": host, "dataset": dataset, "seed": seed, "arm": arm}
        endpoint_path = prefix + "/validation_endpoints.json"
        reference = {"split": "validation", "evaluation_scope": "validation_only",
                     "held_out_test_evaluated": False, "count": 5,
                     "qty_rmse": 5.0, "qty_mae": 2.0, "time_nll": 1.0,
                     "tail": {"count": 2, "qty_rmse": 8.0, "qty_mae": 3.0, "time_nll": 1.5}}
        endpoint = {"status": "complete", "job": terminal_job, "best_epoch": 2,
                    "completed_epochs": 40, "initial_state_sha256": "e" * 64,
                    "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
                    "selected": {**copy.deepcopy(reference), "state_sha256": "d" * 64,
                                 "checkpoint_path": "/synthetic/remote/best_val_qty_rmse_model.pt"}}
        endpoint_sha = self.write_json(endpoint_path, endpoint)
        terminal = {"status": "complete", "scientific_success": True,
                    "held_out_test_evaluated": False, "contract_sha256": origin,
                    "job": terminal_job,
                    "files": {"remote/best_val_qty_rmse_model.pt": checkpoint_sha,
                              "remote/last_epoch_state.pt": last_sha,
                              "remote/history.csv": history_sha,
                              "remote/validation_endpoints.json": endpoint_sha}}
        terminal_sha = self.write_json(terminal_path, terminal)
        exit_path = prefix + "/worker_exit.json"
        exited = {"actual_child_exit_observed": True, "returncode": 0,
                  "contract_sha256": origin, "job": terminal_job}
        exit_sha = self.write_json(exit_path, exited)
        return {"cohort": prepare.COHORT, "study": "components", "id": identity,
                "dataset": dataset, "seed": seed, "arm": arm,
                "training_host": host, "evaluation_host": host,
                "frozen_source_closure": self.source_closure,
                "source_revision_history": [prepare.REVISION],
                "checkpoint_state_sha256": "d" * 64, "initial_state_sha256": "e" * 64,
                "checkpoint": {"relative_path": checkpoint, "sha256": checkpoint_sha},
                "checkpoint_metadata_path": metadata_path, "checkpoint_metadata_sha256": metadata_sha,
                "history_path": history_path, "history_sha256": history_sha,
                "history_original_relative_path": "remote/history.csv",
                "validation_endpoint_path": endpoint_path, "validation_endpoint_sha256": endpoint_sha,
                "validation_endpoint_original_relative_path": "remote/validation_endpoints.json",
                "selected_epoch": 2, "selected_metric_value": 5.0,
                "terminal_manifest_path": terminal_path, "terminal_manifest_sha256": terminal_sha,
                "checkpoint_original_relative_path": "remote/best_val_qty_rmse_model.pt",
                "last_checkpoint": {"relative_path": last, "sha256": last_sha},
                "last_original_relative_path": "remote/last_epoch_state.pt",
                "worker_exit_path": exit_path, "worker_exit_sha256": exit_sha,
                "validation_reference": reference}

    def refresh_json(self, owner, path_key, sha_key, mutate):
        name = owner[path_key]
        value = prepare.read(self.root / name)
        mutate(value)
        owner[sha_key] = self.write_json(name, value)

    def pins(self):
        return patch.multiple(prepare, SOURCE=self.source_closure, PARENT=self.parent,
                              TEST_COUNTS={dataset: 5 for dataset in prepare.DATASETS})

    def reseal_history(self, job, values):
        job["history_sha256"] = self.history(job["history_path"], values)
        self.refresh_json(job, "terminal_manifest_path", "terminal_manifest_sha256",
                          lambda value: value["files"].update({
                              job["history_original_relative_path"]: job["history_sha256"]}))

    def reseal_endpoint(self, job, mutate):
        self.refresh_json(job, "validation_endpoint_path", "validation_endpoint_sha256", mutate)
        self.refresh_json(job, "terminal_manifest_path", "terminal_manifest_sha256",
                          lambda value: value["files"].update({
                              job["validation_endpoint_original_relative_path"]: job["validation_endpoint_sha256"]}))

    def build(self):
        with self.pins():
            return prepare.build_contract(self.root, self.design, self.bindings, self.runtimes)


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="components_binding_synthetic_", dir="/private/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = BindingFixture(self.temporary.name)

    @property
    def job(self):
        return self.fixture.bindings["jobs"][0]

    def reject(self, expected):
        with self.assertRaisesRegex(ValueError, expected):
            self.fixture.build()

    def test_36_synthetic_originals_bind_without_granting_execution(self):
        contract = self.fixture.build()
        self.assertEqual(len(contract["jobs"]), 36)
        self.assertEqual({j["id"] for j in contract["jobs"]}, prepare.expected_ids())
        self.assertEqual(sum(j["evaluation_host"] == "5080" for j in contract["jobs"]), 28)
        self.assertEqual(sum(j["evaluation_host"] == "5090" for j in contract["jobs"]), 8)
        self.assertEqual(contract["authorization"], {"permit_required": True, "Test_authorized_by_binding": False})
        self.assertFalse(contract["new_training"])
        self.assertFalse(contract["checkpoint_reselection"])
        self.assertFalse(contract["test_is_independent"])
        payload = {k: v for k, v in contract.items() if k != "contract_sha256"}
        self.assertEqual(prepare.canonical(payload), contract["contract_sha256"])

    def test_prepared_synthetic_contract_passes_real_evaluator_static_guard(self):
        from paper.scripts import evaluate_titantpp_components_frozen as evaluator

        # Only the operational adapter is copied, never scientific source/model/data.
        evaluator_path = "operational/evaluator.py"
        evaluator_files = {evaluator_path: self.fixture.write_bytes(
            evaluator_path, Path(evaluator.__file__).read_bytes())}
        self.fixture.bindings["evaluator_integrity"] = {
            "files": evaluator_files, "closure": prepare.canonical(evaluator_files)}
        contract = self.fixture.build()
        self.fixture.write_json("contract.json", contract)
        with patch.multiple(evaluator, SOURCE_CLOSURE=self.fixture.source_closure,
                            TRAINING_CONTRACT=self.fixture.parent):
            bound = evaluator.verify_contract(self.fixture.root, require_bound=True)
        self.assertEqual(bound["contract_sha256"], contract["contract_sha256"])
        self.assertEqual(len(bound["jobs"]), 36)
        self.assertFalse(bound["authorization"]["Test_authorized_by_binding"])

    def test_missing_and_duplicate_conditions_rejected(self):
        original = self.fixture.bindings["jobs"]
        self.fixture.bindings["jobs"] = original[:-1]
        self.reject("exactly36")
        self.fixture.bindings["jobs"] = original[:-1] + [copy.deepcopy(original[0])]
        self.reject("exactly36")

    def test_training_must_be_terminal_before_binding(self):
        self.fixture.refresh_json(self.fixture.bindings, "training_monitor_path", "training_monitor_sha256",
                                  lambda value: value.update(all_training_terminal_verified=False))
        self.reject("terminal proof is incomplete")

    def test_actual_scientific_exit0_required(self):
        self.fixture.refresh_json(self.job, "worker_exit_path", "worker_exit_sha256",
                                  lambda value: value.update(returncode=1))
        self.reject("Actual scientific worker exit0")

    def test_observed_exit_required_even_when_zero(self):
        self.fixture.refresh_json(self.job, "worker_exit_path", "worker_exit_sha256",
                                  lambda value: value.update(actual_child_exit_observed=False))
        self.reject("Actual scientific worker exit0")

    def test_foreign_cohort_rejected(self):
        self.job["cohort"] = "history27_source129"
        self.reject("Foreign cohort")

    def test_source131_lineage_required(self):
        self.job["frozen_source_closure"] = "f" * 64
        self.reject("Scientific lineage changed")

    def test_source_file_byte_corruption_rejected(self):
        (self.fixture.root / "source/module_007.py").write_bytes(b"modified science")
        self.reject("Artifact SHA mismatch")

    def test_frozen_training_contract_required(self):
        self.fixture.refresh_json(self.fixture.bindings, "training_contract_path", "training_contract_sha256",
                                  lambda value: value.update(synthetic_contract=False))
        self.reject("Original scientific training contract changed")

    def test_selected_byte_corruption_rejected(self):
        (self.fixture.root / self.job["checkpoint"]["relative_path"]).write_bytes(b"other checkpoint")
        self.reject("Artifact SHA mismatch")

    def test_last_cannot_substitute_for_selected(self):
        self.job["checkpoint"] = copy.deepcopy(self.job["last_checkpoint"])
        self.job["checkpoint_original_relative_path"] = self.job["last_original_relative_path"]
        self.reject("Last checkpoint cannot replace selected")

    def test_tensor_metadata_mismatch_rejected(self):
        self.fixture.refresh_json(self.job, "checkpoint_metadata_path", "checkpoint_metadata_sha256",
                                  lambda value: value.update(model_state_sha256="f" * 64))
        self.reject("Selected metadata mismatch: model_state_sha256")

    def test_tied_minimum_uses_first_epoch(self):
        contract = self.fixture.build()
        self.assertEqual(contract["jobs"][0]["selected_epoch"], 2)
        self.job["selected_epoch"] = 3
        self.fixture.refresh_json(self.job, "checkpoint_metadata_path", "checkpoint_metadata_sha256",
                                  lambda value: value.update(best_epoch=3))
        self.reject("original first finite minimum")

    def test_middle_nonfinite_history_keeps_original_finite_selector(self):
        values = [9.0, 5.0, float("nan"), float("inf")] + [6.0] * 36
        self.fixture.reseal_history(self.job, values)
        self.assertEqual(self.fixture.build()["jobs"][0]["selected_epoch"], 2)

    def test_all_nonfinite_history_rejected(self):
        self.job["history_sha256"] = self.fixture.history(self.job["history_path"], [float("nan")] * 40)
        self.reject("No completed finite Validation history")

    def test_incomplete_and_out_of_order_history_rejected(self):
        self.job["history_sha256"] = self.fixture.history(self.job["history_path"], [5.0] * 39)
        self.reject("No completed finite Validation history")
        self.job["history_sha256"] = self.fixture.write_bytes(
            self.job["history_path"], b"epoch,val_qty_rmse\n1,9\n3,5\n")
        self.reject("Incomplete or unordered history")

    def test_test_endpoint_cannot_replace_validation_reference(self):
        self.job["validation_reference"]["split"] = "test"
        self.reject("Foreign Validation endpoint")

    def test_six_metrics_must_exist_and_be_finite(self):
        self.job["validation_reference"]["tail"]["time_nll"] = float("nan")
        self.reject("Validation six metrics unavailable")

    def test_native_host_cannot_be_reassigned(self):
        self.job["evaluation_host"] = "5090"
        self.reject("Native host reassignment")

    def test_runtime_gpu_and_cudnn_required(self):
        self.fixture.runtimes["5080"]["gpu_uuid"] = prepare.GPU_UUIDS["5090"]
        self.reject("Native Runtime mismatch")
        self.fixture.runtimes["5080"]["gpu_uuid"] = prepare.GPU_UUIDS["5080"]
        self.fixture.runtimes["5090"]["cudnn"] = 91900
        self.reject("Native Runtime mismatch")

    def test_two_runtimes_required(self):
        self.fixture.runtimes.pop("5090")
        self.reject("Both native runtimes")

    def test_precision_changes_rejected(self):
        self.fixture.runtimes["5080"]["TF32"] = True
        self.reject("Native evaluation precision")

    def test_frozen_train_statistics_cannot_be_reestimated(self):
        self.fixture.bindings["datasets"][prepare.DATASETS[0]]["definition"]["model"]["train_mean"] = 1.0
        self.reject("Train statistics, architecture or preprocessing changed")

    def test_train_tail_threshold_not_reestimated(self):
        self.fixture.bindings["datasets"][prepare.DATASETS[0]]["tail_threshold"] += 1
        self.reject("Data identity changed")

    def test_test_population_count_cannot_change(self):
        self.fixture.bindings["datasets"][prepare.DATASETS[0]]["test_population"]["count"] += 1
        self.reject("Test count changed")

    def test_validation_population_must_match_training_contract(self):
        self.fixture.bindings["datasets"][prepare.DATASETS[0]]["validation_population"]["count"] += 1
        self.reject("Original Validation target count changed")

    def test_validation_endpoint_population_must_match_bound_dataset(self):
        self.job["validation_reference"]["count"] += 1
        self.fixture.reseal_endpoint(self.job, lambda value: value["selected"].update(count=6))
        self.reject("Cross-arm Validation population mismatch")

    def test_boolean_selected_epoch_rejected(self):
        self.job["selected_epoch"] = True
        self.reject("Missing finite selected epoch/value")

    def test_boolean_selected_metric_rejected(self):
        self.job["selected_metric_value"] = True
        self.reject("Missing finite selected epoch/value")

    def test_history_self_sha_cannot_replace_original_terminal_seal(self):
        self.job["history_sha256"] = self.fixture.history(
            self.job["history_path"], [10.0, 5.0, 5.0] + [7.0] * 37)
        self.reject("Selection history not bound by original terminal")

    def test_original_json_history_supported(self):
        self.job["history_path"] = str(Path(self.job["history_path"]).with_suffix(".json"))
        self.job["history_original_relative_path"] = "remote/history.json"
        rows = [{"epoch": index, "val_qty_rmse": value} for index, value in enumerate(
            [9.0, 5.0, 5.0] + [6.0] * 37, start=1)]
        self.job["history_sha256"] = self.fixture.write_json(self.job["history_path"], {"history": rows})
        self.fixture.refresh_json(self.job, "terminal_manifest_path", "terminal_manifest_sha256",
                                  lambda value: value["files"].update({
                                      "remote/history.json": self.job["history_sha256"]}))
        self.assertEqual(self.fixture.build()["jobs"][0]["selected_epoch"], 2)

    def test_endpoint_self_sha_cannot_replace_original_terminal_seal(self):
        self.fixture.refresh_json(self.job, "validation_endpoint_path", "validation_endpoint_sha256",
                                  lambda value: value["selected"]["tail"].update(qty_rmse=123456.0))
        self.reject("Validation endpoint not bound by original terminal")

    def test_mutated_reference_cannot_replace_original_selected_metrics(self):
        self.job["validation_reference"]["tail"]["qty_rmse"] = 123456.0
        self.reject("Validation reference differs from original selected endpoint")

    def test_original_endpoint_reinference_roundoff_tolerance(self):
        self.job["validation_reference"]["qty_rmse"] = 5.0 + 5e-9
        self.fixture.reseal_endpoint(self.job, lambda value: value["selected"].update(qty_rmse=5.0 + 5e-9))
        self.assertEqual(self.fixture.build()["jobs"][0]["selected_epoch"], 2)
        self.assertEqual(self.job["selected_metric_value"], 5.0)
        self.assertEqual(prepare.read(self.fixture.root / self.job["checkpoint_metadata_path"])
                         ["selected_metric_value"], 5.0)

    def test_endpoint_reinference_difference_outside_tolerance_rejected(self):
        self.job["validation_reference"]["qty_rmse"] = 5.0 + 3e-8
        self.fixture.reseal_endpoint(self.job, lambda value: value["selected"].update(qty_rmse=5.0 + 3e-8))
        self.reject("Foreign Validation endpoint")

    def test_endpoint_job_epoch_and_tensor_identity_required(self):
        original = prepare.read(self.fixture.root / self.job["validation_endpoint_path"])
        for mutate, error in (
                (lambda value: value["job"].update(seed=52), "identity/epoch differs"),
                (lambda value: value.update(best_epoch=3), "identity/epoch differs"),
                (lambda value: value.update(completed_epochs=39), "identity/epoch differs"),
                (lambda value: value["selected"].update(state_sha256="f" * 64), "tensor/split differs")):
            with self.subTest(error=error):
                self.fixture.reseal_endpoint(self.job, lambda value: (value.clear(), value.update(copy.deepcopy(original))))
                self.fixture.reseal_endpoint(self.job, mutate)
                self.reject(error)

    def test_changing_full_data_and_design_together_does_not_change_training_lineage(self):
        dataset = "raf_spare_parts"
        spec = self.fixture.bindings["datasets"][dataset]
        changed = self.fixture.write_bytes(spec["data_path"], b"replacement synthetic full data")
        spec["data_sha256"] = changed
        self.fixture.design["datasets"][dataset]["full_data_byte_sha256"] = changed
        self.reject("Dataset bytes/count differ from frozen training lineage")

    def test_changed_training_data_sha_rejected_by_lineage(self):
        spec = self.fixture.bindings["datasets"]["raf_spare_parts"]
        spec["training_data_sha256"] = self.fixture.write_bytes(
            spec["training_data_path"], b"replacement synthetic training data")
        self.reject("Dataset bytes/count differ from frozen training lineage")

    def test_changed_split_and_design_sha_rejected_by_lineage(self):
        dataset = prepare.DATASETS[0]
        spec = self.fixture.bindings["datasets"][dataset]
        changed = self.fixture.write_json(spec["split_manifest_path"], {"synthetic": True, "other_split": True})
        spec["split_manifest_sha256"] = changed
        self.fixture.design["datasets"][dataset]["split_manifest_sha256"] = changed
        self.reject("Dataset bytes/count differ from frozen training lineage")

    def test_test_count_and_design_cannot_override_training_population(self):
        dataset = prepare.DATASETS[0]
        spec = self.fixture.bindings["datasets"][dataset]
        spec["expected_test_count"] = 6
        spec["test_population"]["count"] = 6
        self.fixture.design["datasets"][dataset]["expected_Test_target_count"] = 6
        self.reject("Dataset bytes/count differ from frozen training lineage")

    def test_native_runtime_missing_or_invalid_fingerprint_rejected(self):
        original = copy.deepcopy(self.fixture.runtimes["5080"])
        for key in ("python", "python_version", "torch_num_threads", "interop_threads"):
            with self.subTest(missing=key):
                self.fixture.runtimes["5080"] = copy.deepcopy(original)
                del self.fixture.runtimes["5080"][key]
                self.reject("Incomplete native Runtime fingerprint")
        for key, invalid in (("python", "relative-python"), ("python_version", ""),
                             ("python_version", "synthetic-version"),
                             ("torch_num_threads", True), ("interop_threads", 2), ("torch", "")):
            with self.subTest(invalid=key):
                self.fixture.runtimes["5080"] = copy.deepcopy(original)
                self.fixture.runtimes["5080"][key] = invalid
                self.reject("Invalid native Runtime fingerprint")

    def test_absolute_and_escape_paths_rejected(self):
        for name, error in (("/private/tmp/synthetic.pt", "must be relative"),
                            ("../synthetic.pt", "escapes immutable root")):
            with self.subTest(name=name):
                self.job["checkpoint"]["relative_path"] = name
                self.reject(error)

    def test_symlink_cannot_escape_immutable_root(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_external_", dir="/private/tmp") as external:
            target = Path(external) / "synthetic.pt"
            target.write_bytes(b"synthetic external checkpoint")
            (self.fixture.root / "escape.pt").symlink_to(target)
            self.job["checkpoint"]["relative_path"] = "escape.pt"
            self.job["checkpoint"]["sha256"] = prepare.sha(target)
            self.reject("escapes immutable root")

    def test_evaluator_byte_tampering_rejected(self):
        (self.fixture.root / "operational/evaluator.py").write_bytes(b"modified adapter")
        self.reject("Artifact SHA mismatch")

    def test_cli_only_creates_immutable_contract_not_permit(self):
        design = self.fixture.root / "design.json"
        bindings = self.fixture.root / "bindings.json"
        runtimes = self.fixture.root / "runtimes.json"
        for path, value in ((design, self.fixture.design), (bindings, self.fixture.bindings),
                            (runtimes, self.fixture.runtimes)):
            self.fixture.write_json(path.name, value)
        before = {p.relative_to(self.fixture.root) for p in self.fixture.root.rglob("*")}
        argv = ["--root", str(self.fixture.root), "--design", str(design),
                "--bindings", str(bindings), "--runtimes", str(runtimes)]
        stdout = io.StringIO()
        with self.fixture.pins(), contextlib.redirect_stdout(stdout):
            prepare.main(argv)
        after = {p.relative_to(self.fixture.root) for p in self.fixture.root.rglob("*")}
        self.assertEqual(after - before, {Path("contract.json")})
        result = json.loads(stdout.getvalue())
        self.assertFalse(result["Test_permit_created"])
        self.assertEqual(result["job_count"], 36)
        with self.fixture.pins(), self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
            prepare.main(argv)


if __name__ == "__main__":
    unittest.main()
