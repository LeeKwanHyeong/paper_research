"""Offline contract, identity, aggregate, gate, and one-shot failure tests."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATH = ROOT / "paper/scripts/evaluate_titantpp_retrieval_test.py"
if not RUNTIME_PATH.exists():
    RUNTIME_PATH = Path(__file__).with_name("evaluate_titantpp_retrieval_test.py")
spec = importlib.util.spec_from_file_location("retrieval_test", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def metrics(count=8268, tail=79):
    aggregate = runtime.QuantityMetrics()
    aggregate.add([0.] * (count - tail) + [3450.] * tail,
                  [1.] * (count - tail) + [3452.] * tail, [2.] * count)
    return aggregate.finish()


def contract():
    source = {f"closure/file_{index}.py": hashlib.sha256(str(index).encode()).hexdigest() for index in range(126)}
    data_spec = {"dataset_id": "yellow_trip_hourly", "data_sha256": runtime.DATA_SHA,
        "quantity_boundaries_all_train_rows": list(runtime.BOUNDS),
        "loader": {"batch_size": 128, "lookback_weeks": 168, "max_seq_len": 256, "num_workers": 0, "drop_last": False},
        "model": {"lambda_log_qty": 1., "quantity_variant": "count_only_log_regression",
                  "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None}}}
    reference = metrics()
    rows = []
    for host in runtime.HOSTS:
        for arm in runtime.ARMS:
            identity = host + "__yellow_trip_hourly__42__" + arm
            rows.append({"id": identity, "host": host, "arm": arm, "seed": 42, "dataset": "yellow_trip_hourly",
                         "selected_epoch": 120, "selected_metric_value": reference["qty_rmse"], "source_revision": runtime.SOURCE_REVISION,
                         "checkpoint_path": "checkpoints/" + identity + ".pt", "checkpoint_sha256": "1" * 64,
                         "state_sha256": "2" * 64, "validation_reference": deepcopy(reference)})
    return {"schema": runtime.SCHEMA, "approval": {"test_inference_authorized": True, "user_request": "Test까지 뽑아봐",
        "new_training": False, "checkpoint_reselection": False, "raw_predictions_written": False, "automatic_retry": False},
        "source_root": "source", "source_files": source, "source_closure_sha256": runtime.canonical(source),
        "source_revision": runtime.SOURCE_REVISION, "data_spec": data_spec,
        "data": {"path": "data/yellow_trip_hourly_with_split.parquet", "sha256": runtime.DATA_SHA,
                 "split_manifest_path": "data/yellow_trip_hourly_split_manifest.json", "split_manifest_sha256": runtime.SPLIT_SHA},
        "hosts": {host: {"runtime_expected": runtime.RUNTIMES[host], "gpu_uuid": runtime.GPU_UUIDS[host]} for host in runtime.HOSTS},
        "resources": deepcopy(runtime.RESOURCES), "rows": rows,
        "population_references": {split: {"target_count": count, "tail_count": tail, "target_identity_sha256": "3" * 64,
            "target_quantity_sha256": "4" * 64, "target_dt_sha256": "5" * 64}
            for split, count, tail in (("validation", 8268, 79), ("test", 8327, 83))},
        "evaluator_sha256": runtime.sha_file(RUNTIME_PATH),
        "test_exposure_lineage": {"dataset_Test_previously_accessed": True,
                                 "these10_checkpoint_Test_previously_evaluated": False, "independent_untouched_Test": False}}


def receipt(c, host, split, contract_sha):
    pop = c["population_references"][split]
    aggregate = metrics(pop["target_count"], pop["tail_count"])
    rows = []
    for original in runtime.host_rows(c, host):
        row = {key: original[key] for key in ("id", "host", "arm", "seed", "dataset", "selected_epoch", "checkpoint_sha256", "state_sha256")}
        row.update(state_sha256_before=row["state_sha256"], state_sha256_after=row["state_sha256"],
                   checkpoint_file_sha256_after=row["checkpoint_sha256"], population=pop, metrics=deepcopy(aggregate),
                   validation_comparison=runtime.validation_comparison(aggregate, original["validation_reference"]) if split == "validation" else None)
        rows.append(row)
    return {"schema": runtime.SCHEMA, "status": "complete", "host": host, "split": split, "contract_sha256": contract_sha,
        "source_closure_sha256": runtime.SOURCE_SHA, "evaluator_sha256": c["evaluator_sha256"], "rows": rows,
        "runtime": {**runtime.RUNTIMES[host], "threads": 4, "gpu": {"uuid": runtime.GPU_UUIDS[host]}},
        "new_training": False, "checkpoint_reselection": False, "raw_predictions_written": False, "automatic_retry": False}


class ImportAndPathTests(unittest.TestCase):
    def test_import_is_stdlib_only_and_does_not_execute(self):
        program = """import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy', 'polars', 'paper', 'models', 'data_loader', 'simple_lab_test'):
        raise AssertionError('Unexpected heavy/scientific import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_path(sys.argv[1], run_name='offline_import_check')
"""
        completed = subprocess.run([sys.executable, "-c", program, str(RUNTIME_PATH)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_confines_absolute_parent_and_symlink_paths(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            for path in ("../escape", "/etc/passwd"):
                with self.assertRaises(ValueError):
                    runtime.confined(root, path)
            (root / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                runtime.confined(root, "link/result.json")

    def test_scientific_imports_pin_utils_and_root_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            file = source / "sealed.py"
            file.write_text("# sealed scientific module\n")
            c = {"source_files": {"sealed.py": runtime.sha_file(file)}}
            for name in ("utils", "utils.helper", "models", "simple_lab_test", "data_loader"):
                with patch.dict(runtime.sys.modules, {name: SimpleNamespace(__file__=str(file))}, clear=True):
                    runtime.scientific_imports(c, root)
                with patch.dict(runtime.sys.modules, {name: SimpleNamespace(__file__=str(root / "foreign.py"))}, clear=True):
                    with self.assertRaisesRegex(ValueError, "Foreign scientific module"):
                        runtime.scientific_imports(c, root)
            file.write_text("# changed bytes\n")
            with patch.dict(runtime.sys.modules, {"utils.helper": SimpleNamespace(__file__=str(file))}, clear=True):
                with self.assertRaisesRegex(ValueError, "Unsealed scientific module"):
                    runtime.scientific_imports(c, root)

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "malformed.json"
            for text in ('{"scope":1,"scope":2}', '{"metric":NaN}', '{"metric":Infinity}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    runtime.read(path)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        self.source = patch.object(runtime, "SOURCE_SHA", self.c["source_closure_sha256"])
        self.data = patch.object(runtime, "DATA_SPEC_SHA", runtime.canonical(self.c["data_spec"]))
        self.source.start()
        self.data.start()
        self.addCleanup(self.source.stop)
        self.addCleanup(self.data.stop)

    def test_exact_scope_and_five_arms_on_both_hosts(self):
        runtime.validate_contract(self.c)
        self.assertEqual([r["arm"] for r in runtime.host_rows(self.c, "5080")], list(runtime.ARMS))
        for key in ("new_training", "checkpoint_reselection", "raw_predictions_written", "automatic_retry"):
            changed = deepcopy(self.c)
            changed["approval"][key] = True
            with self.assertRaisesRegex(ValueError, "scope"):
                runtime.validate_contract(changed)

    def test_duplicate_condition_or_changed_selection_rejected(self):
        mutations = [lambda c: c["rows"].__setitem__(9, c["rows"][0]),
                     lambda c: c["rows"][0].__setitem__("selected_epoch", 0),
                     lambda c: c["rows"][0].__setitem__("seed", 52),
                     lambda c: c["rows"][0].__setitem__("source_revision", "foreign"),
                     lambda c: c["rows"][0].__setitem__("selected_metric_value", 100.),
                     lambda c: c["rows"][0].__setitem__("checkpoint_path", "../original.pt")]
        for mutate in mutations:
            changed = deepcopy(self.c)
            mutate(changed)
            with self.assertRaises(ValueError):
                runtime.validate_contract(changed)

    def test_changed_source_model_runtime_or_budget_rejected(self):
        mutations = [lambda c: c["source_files"].__setitem__("closure/file_0.py", "f" * 64),
                     lambda c: c["data_spec"]["model"].__setitem__("lambda_log_qty", .5),
                     lambda c: c["hosts"]["5080"]["runtime_expected"].__setitem__("cudnn", 92000),
                     lambda c: c["resources"].__setitem__("host_split_timeout_seconds", 901),
                     lambda c: c["population_references"]["test"].__setitem__("target_count", 8268)]
        for mutate in mutations:
            changed = deepcopy(self.c)
            mutate(changed)
            with self.assertRaises(ValueError):
                runtime.validate_contract(changed)

    def test_selected_metadata_and_state_digest(self):
        row = self.c["rows"][0]
        payload = {"backbone": row["arm"], "seed": 42, "best_epoch": 120, "evaluation_scope": "validation_only",
                   "source_revision": runtime.SOURCE_REVISION, "source_revision_history": [runtime.SOURCE_REVISION],
                   "selected_metric_value": row["selected_metric_value"],
                   "held_out_test_evaluated": False, "variant": "count_only_log_regression",
                   "checkpoint_monitor": "validation_raw_quantity_rmse", "checkpoint_monitor_history_key": "val_qty_rmse",
                   "checkpoint_selection": "best_validation_raw_quantity_rmse", "model_state_sha256": row["state_sha256"]}
        runtime.validate_checkpoint(payload, row, row["state_sha256"])
        for key, value in (("seed", 52), ("best_epoch", 119), ("epoch", 119), ("backbone", runtime.ARMS[1]),
                           ("source_revision", "foreign"), ("source_revision_history", [runtime.SOURCE_REVISION, "foreign"]),
                           ("selected_metric_value", row["selected_metric_value"] + 1e-12),
                           ("checkpoint_selection", "best_validation_joint_objective"), ("model_state_sha256", "a" * 64)):
            changed = {**payload, key: value}
            with self.assertRaises(ValueError):
                runtime.validate_checkpoint(changed, row, row["state_sha256"])


class AggregateTests(unittest.TestCase):
    def test_boundary_equality_lower_bin_and_tail_strict(self):
        q = [0., 7., 8., 686., 687., 1562., 1563., 3449., 3450.]
        p = [quantity + index for index, quantity in enumerate(q)]
        temporal = [float(index) / 2 for index in range(9)]
        aggregate = runtime.QuantityMetrics()
        aggregate.add(q[:4], p[:4], temporal[:4])
        aggregate.add(q[4:], p[4:], temporal[4:])
        result = aggregate.finish()
        self.assertEqual([c["count"] for c in result["quantity_cells"]], [2, 2, 2, 2, 1])
        self.assertEqual(result["tail"]["count"], 1)
        self.assertEqual(result["qty_sse"], sum(index ** 2 for index in range(9)))
        self.assertEqual(result["qty_absolute_error_sum"], sum(range(9)))
        self.assertEqual(result["qty_signed_error_sum"], sum(range(9)))
        self.assertEqual(result["time_nll_sum"], sum(temporal))
        runtime.validate_metrics(result, {"target_count": 9, "tail_count": 1})

    def test_signed_bias_and_empty_cells(self):
        aggregate = runtime.QuantityMetrics()
        aggregate.add([3., 3450.], [1., 3452.], [-1., 3.])
        result = aggregate.finish()
        self.assertEqual(result["qty_bias"], 0.)
        self.assertEqual(result["qty_mae"], 2.)
        self.assertEqual(result["qty_rmse"], 2.)
        self.assertEqual(result["time_nll"], 1.)
        self.assertIsNone(result["quantity_cells"][1]["qty_mae"])
        runtime.validate_metrics(result, {"target_count": 2, "tail_count": 1})

    def test_invalid_or_mismatched_inputs_rejected(self):
        for q, p, t in (([math.nan], [1.], [1.]), ([1.], [math.inf], [1.]), ([1.], [1.], [math.nan]),
                        ([-1.], [1.], [1.]), ([1.], [-1.], [1.]), ([1., 2.], [1.], [1.])):
            with self.assertRaises(ValueError):
                runtime.QuantityMetrics().add(q, p, t)

    def test_receipt_cannot_hide_count_or_sufficient_statistic_change(self):
        aggregate = metrics(10, 1)
        for key, value in (("count", 9), ("qty_sse", 99.), ("qty_mae", math.nan)):
            changed = deepcopy(aggregate)
            changed[key] = value
            with self.assertRaises(ValueError):
                runtime.validate_metrics(changed, {"target_count": 10, "tail_count": 1})

    def test_absolute_validation_tolerance_not_relative(self):
        actual = metrics()
        reference = deepcopy(actual)
        reference["qty_mae"] += 5e-6
        self.assertTrue(runtime.validation_comparison(actual, reference)["passed"])
        reference["tail"]["time_nll"] += 2e-5
        with self.assertRaisesRegex(ValueError, "tolerance"):
            runtime.validation_comparison(actual, reference)


class PopulationTests(unittest.TestCase):
    def population(self):
        dataset = SimpleNamespace(parts=["A", "B"], index=[(0, 1), (1, 0)],
            seq_lists=[[1, 2, 3], [1, 2]], val_lists=[[0., 1., 3500.], [0., 2.]],
            dt_lists=[[1., 1., 2.], [1., 3.]], split_lists=[["train", "validation", "test"], ["validation", "test"]])
        h = hashlib.sha256(b"hard_lmm_target_identity_v1\0test\0")
        h.update(b'["A","B"]')
        for values in ((0, 1), (2, 1), (3, 2)):
            h.update(struct.pack("<qq", *values))
        expected = {"target_count": 2, "tail_count": 1, "target_identity_sha256": h.hexdigest(),
            "target_quantity_sha256": hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + struct.pack("<dd", 3500., 2.)).hexdigest(),
            "target_dt_sha256": hashlib.sha256(b"frozen_lognormal_target_dt_v1\0" + struct.pack("<dd", 2., 3.)).hexdigest(),
            "split": "test", "entity_count": 2, "body_count": 1, "tail_min_exclusive": 3449, "body_max_inclusive": 1562,
            "target_dt_native_float32_raw_sha256": hashlib.sha256(struct.pack("<ff", 2., 3.)).hexdigest()}
        return dataset, expected

    def test_original_identity_durations_and_metadata(self):
        dataset, expected = self.population()
        quantities, actual = runtime.population_identity(dataset, "test", expected)
        self.assertEqual(quantities, [3500., 2.])
        self.assertEqual(actual, expected)

    def test_changed_order_future_prefix_or_target_split_rejected(self):
        for mutate in (lambda d: d.index.reverse(), lambda d: d.seq_lists[0].__setitem__(1, 3),
                       lambda d: d.split_lists[0].__setitem__(2, "validation"), lambda d: d.dt_lists[0].__setitem__(2, 4.)):
            dataset, expected = self.population()
            mutate(dataset)
            with self.assertRaises(ValueError):
                runtime.population_identity(dataset, "test", expected)

    def test_validation_prefix_refuses_test_context(self):
        dataset, expected = self.population()
        dataset.split_lists[0] = ["train", "test", "validation"]
        dataset.split_lists[1][-1] = "validation"
        with self.assertRaisesRegex(ValueError, "prefix"):
            runtime.population_identity(dataset, "validation", expected)


class GateAndClaimTests(unittest.TestCase):
    setUp = ContractTests.setUp

    def gate(self, root, contract_sha="a" * 64):
        rows = []
        for host in runtime.HOSTS:
            path = "results/validation_" + host + ".json"
            runtime.write_exclusive(root / path, receipt(self.c, host, "validation", contract_sha))
            rows.append({"host": host, "path": path, "sha256": runtime.sha_file(root / path)})
        runtime.write_exclusive(root / "validation_gate.json", {"passed": True, "conditions": 10,
            "contract_sha256": contract_sha, "receipts": rows})

    def test_global_gate_requires_both_complete_validation_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.gate(root)
            self.assertEqual(runtime.verify_validation_gate(self.c, root, "a" * 64), runtime.sha_file(root / "validation_gate.json"))
            result_path = root / "results/validation_5090.json"
            result = runtime.read(result_path)
            result["rows"][0]["state_sha256_after"] = "a" * 64
            result_path.write_text(json.dumps(result))
            gate = runtime.read(root / "validation_gate.json")
            gate["receipts"][1]["sha256"] = runtime.sha_file(result_path)
            (root / "validation_gate.json").write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, "state"):
                runtime.verify_validation_gate(self.c, root, "a" * 64)

    def test_foreign_contract_partial_gate_and_raw_sha_drift_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.gate(root)
            with self.assertRaisesRegex(ValueError, "gate"):
                runtime.verify_validation_gate(self.c, root, "b" * 64)
            path = root / "results/validation_5080.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "raw receipt SHA"):
                runtime.verify_validation_gate(self.c, root, "a" * 64)
            gate = runtime.read(root / "validation_gate.json")
            gate["receipts"] = gate["receipts"][:1]
            (root / "validation_gate.json").write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, "both"):
                runtime.verify_validation_gate(self.c, root, "a" * 64)

    def test_failed_one_shot_is_retained_and_never_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "execution_contract.json"
            runtime.write_exclusive(path, self.c)
            calls = []
            def fail(*args):
                calls.append(args)
                raise RuntimeError("synthetic worker failure")
            with patch.object(runtime, "verify_files"):
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    runtime.pipeline(path, "5080", "validation", runner=fail)
                with self.assertRaisesRegex(ValueError, "retry prohibited"):
                    runtime.pipeline(path, "5080", "validation", runner=fail)
            self.assertEqual(len(calls), 1)
            self.assertEqual(runtime.read(root / "results/validation_5080.json")["status"], "failed")
            self.assertTrue((root / "claims/validation_5080.json").exists())

    def test_test_never_launches_without_global_validation_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "execution_contract.json"
            runtime.write_exclusive(path, self.c)
            with patch.object(runtime, "verify_files"), patch.object(runtime, "run_worker") as worker:
                with self.assertRaises(FileNotFoundError):
                    runtime.pipeline(path, "5080", "test", runner=worker)
                worker.assert_not_called()
            self.assertEqual(runtime.read(root / "results/test_5080.json")["status"], "failed")

    def test_second_claim_and_foreign_worker_launch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "execution_contract.json"
            runtime.write_exclusive(path, self.c)
            runtime.claim(root, self.c, path, "5080", "validation", "token", 100.)
            with self.assertRaises(FileExistsError):
                runtime.claim(root, self.c, path, "5080", "validation", "token", 100.)
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "one-shot"):
                    runtime.worker(path, "5080", "validation")


class ResourceAccountingTests(unittest.TestCase):
    def test_owned_process_group_rss_and_gpu_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for pid, group, rss in ((41, 41, 5), (42, 41, 7), (99, 99, 100)):
                process = root / str(pid)
                process.mkdir()
                (process / "stat").write_text(f"{pid} (worker with spaces) S 1 {group} 0 0")
                (process / "status").write_text(f"Name: worker\nVmRSS: {rss} kB\n")
            pids, rss = runtime.process_group(41, proc_root=root)
            self.assertEqual(pids, {41, 42})
            self.assertEqual(rss, 12 * 1024)
        completed = SimpleNamespace(stdout="41, 12\n42, 8\n99, 999\n")
        with patch.object(runtime.subprocess, "run", return_value=completed):
            self.assertEqual(runtime.gpu_process_bytes({41, 42}), 20 * 1024 ** 2)
        with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(stdout="41, N/A\n")):
            with self.assertRaisesRegex(ValueError, "monitoring unavailable"):
                runtime.gpu_process_bytes({41})


if __name__ == "__main__":
    unittest.main()
