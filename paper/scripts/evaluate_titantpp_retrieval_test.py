"""One-shot aggregate inference for the ten frozen retrieval checkpoints.

Only stdlib is imported until the contract's scientific source, data, and selected
checkpoint bytes have been verified. Validation replay must pass for all ten
conditions before held-out inference is admitted. No training or prediction export.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import struct
import subprocess
import sys
import time
import traceback

SCHEMA = "titantpp_retrieval_frozen_test_20261008_v1"
SOURCE_SHA = "918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a"
SOURCE_REVISION = "4274043717cf24640d9a18fb35231a1f7ba50c8e"
DATA_SPEC_SHA = "ab3b436e802e5c4d4667a4bff8e7d2cf56b0db77e8b3514da3936a0d28773639"
DATA_SHA = "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46"
SPLIT_SHA = "4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85"
ARMS = tuple("titantpp_retrieval_" + name for name in ("mean", "weighted", "fixed", "adaptive", "generic"))
HOSTS = ("5080", "5090")
BOUNDS = (7., 686., 1562., 3449.)
METRICS = ("qty_rmse", "qty_mae", "time_nll")
RESOURCES = {"threads": 4, "host_split_timeout_seconds": 900,
             "rss_limit_bytes": 16 * 1024 ** 3, "gpu_memory_limit_bytes": 8 * 1024 ** 3}
RUNTIMES = {host: {"cuda": "13.0", "cudnn": 91900 if host == "5080" else 92000,
                  "numpy": "2.4.4", "polars": "1.39.3", "torch": "2.11.0+cu130"} for host in HOSTS}
GPU_UUIDS = {"5080": "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6",
             "5090": "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def confined(root, name):
    name = Path(name)
    require(not name.is_absolute() and ".." not in name.parts and bool(name.parts), "Unsafe contract path")
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Contract path escapes owned root")
    return path


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def host_rows(c, host):
    require(host in HOSTS, "Foreign host")
    return sorted((row for row in c["rows"] if row["host"] == host), key=lambda row: ARMS.index(row["arm"]))


def validate_contract(c):
    """Validate frozen selection and execution scope without importing ML libraries."""
    require(c.get("schema") == SCHEMA, "Foreign Test contract")
    approval = c["approval"]
    require(approval.get("test_inference_authorized") is True and approval.get("user_request") == "Test까지 뽑아봐"
            and all(approval.get(key) is False for key in ("new_training", "checkpoint_reselection",
                    "raw_predictions_written", "automatic_retry")), "Inference authorization/scope changed")
    require(c.get("source_root") == "source" and c.get("source_revision") == SOURCE_REVISION
            and len(c["source_files"]) == 126 and canonical(c["source_files"]) == SOURCE_SHA
            == c.get("source_closure_sha256"), "Frozen source closure changed")
    for path, value in c["source_files"].items():
        confined(Path("/tmp"), path)
        require(digest(value), "Invalid source digest")
    data = c["data"]
    require(data.get("path") == "data/yellow_trip_hourly_with_split.parquet" and data.get("sha256") == DATA_SHA
            and data.get("split_manifest_path") == "data/yellow_trip_hourly_split_manifest.json"
            and data.get("split_manifest_sha256") == SPLIT_SHA, "Frozen data bytes changed")
    spec = c["data_spec"]
    require(canonical(spec) == DATA_SPEC_SHA and spec.get("dataset_id") == "yellow_trip_hourly"
            and spec.get("data_sha256") == DATA_SHA
            and tuple(spec["quantity_boundaries_all_train_rows"]) == BOUNDS, "Frozen dataset or quantity strata changed")
    require(all(spec["loader"].get(key) == value for key, value in
                (("batch_size", 128), ("lookback_weeks", 168), ("max_seq_len", 256), ("num_workers", 0), ("drop_last", False)))
            and spec["model"].get("lambda_log_qty") == 1.
            and spec["model"].get("quantity_variant") == "count_only_log_regression"
            and spec["model"].get("time_observation_contract") ==
                {"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None}, "Frozen model/loader changed")
    require(c.get("resources") == RESOURCES and set(c["hosts"]) == set(HOSTS), "Frozen runtime resources/hosts changed")
    for host in HOSTS:
        require(c["hosts"][host].get("runtime_expected") == RUNTIMES[host]
                and c["hosts"][host].get("gpu_uuid") == GPU_UUIDS[host], "Qualified host runtime changed")
    rows = c["rows"]
    expected = {(host, arm) for host in HOSTS for arm in ARMS}
    require(len(rows) == 10 and {(row["host"], row["arm"]) for row in rows} == expected
            and len({row["id"] for row in rows}) == 10, "Exactly ten selected conditions required")
    for row in rows:
        require(row["id"] == row["host"] + "__yellow_trip_hourly__42__" + row["arm"]
                and row.get("seed") == 42 and row.get("dataset") == "yellow_trip_hourly"
                and row.get("source_revision") == SOURCE_REVISION
                and type(row["selected_epoch"]) is int and 1 <= row["selected_epoch"] <= 300,
                "Frozen selected checkpoint identity changed")
        require(row.get("checkpoint_path") == "checkpoints/" + row["id"] + ".pt"
                and digest(row.get("checkpoint_sha256")) and digest(row.get("state_sha256")), "Frozen selected bytes/state binding missing")
        reference = row["validation_reference"]
        require(reference.get("count", reference.get("target_count")) == 8268 and reference["tail"].get("count") == 79
                and tuple(reference["quantity_boundaries"]) == BOUNDS, "Validation reference population changed")
        require(all(type(cell.get(key)) in (int, float) and math.isfinite(cell[key])
                    for cell in (reference, reference["tail"]) for key in METRICS), "Nonfinite validation reference")
        require(type(row.get("selected_metric_value")) in (int, float)
                and math.isfinite(row["selected_metric_value"])
                and math.isclose(row["selected_metric_value"], reference["qty_rmse"], rel_tol=0., abs_tol=1e-5),
                "Selected validation RMSE binding changed")
    require(set(c["population_references"]) == {"validation", "test"}, "Missing frozen split population")
    for split, expected_count in (("validation", 8268), ("test", 8327)):
        population = c["population_references"][split]
        require(population.get("target_count") == expected_count
                and type(population.get("tail_count")) is int and 0 < population["tail_count"] <= expected_count
                and all(digest(population.get(key)) for key in
                    ("target_identity_sha256", "target_quantity_sha256", "target_dt_sha256")), "Frozen population metadata missing")
    require(c["population_references"]["validation"]["tail_count"] == 79, "Frozen Validation tail changed")
    require(digest(c.get("evaluator_sha256")), "Missing evaluator SHA binding")
    lineage = c["test_exposure_lineage"]
    require(lineage.get("dataset_Test_previously_accessed") is True
            and lineage.get("these10_checkpoint_Test_previously_evaluated") is False
            and lineage.get("independent_untouched_Test") is False, "Test exposure lineage changed")
    return c


def verify_files(c, root):
    for name, value in c["source_files"].items():
        require(sha_file(confined(root / "source", name)) == value, "Frozen source SHA mismatch: " + name)
    for path, value in ((c["data"]["path"], DATA_SHA), (c["data"]["split_manifest_path"], SPLIT_SHA)):
        require(sha_file(confined(root, path)) == value, "Frozen data SHA mismatch: " + path)
    for row in c["rows"]:
        require(sha_file(confined(root, row["checkpoint_path"])) == row["checkpoint_sha256"], "Checkpoint file SHA mismatch: " + row["id"])
    require(sha_file(Path(__file__)) == c["evaluator_sha256"], "Evaluator file SHA mismatch")


def validate_metrics(metrics, population):
    require(metrics.get("count") == population["target_count"] and metrics["tail"].get("count") == population["tail_count"]
            and tuple(metrics["quantity_boundaries"]) == BOUNDS, "Evaluated split population changed")
    cells = metrics["quantity_cells"]
    require(len(cells) == 5 and [cell["bin"] for cell in cells] == list(range(5))
            and sum(cell["count"] for cell in cells) == metrics["count"]
            and metrics["tail"] == {key: value for key, value in cells[-1].items() if key != "bin"}, "Strata do not reconstruct full/tail population")
    for cell in (metrics, *cells, metrics["tail"]):
        require(type(cell["count"]) is int and cell["count"] >= 0, "Invalid sufficient-statistic count")
        for key in ("qty_sse", "qty_absolute_error_sum", "qty_signed_error_sum", "time_nll_sum"):
            require(type(cell.get(key)) in (int, float) and math.isfinite(cell[key]), "Nonfinite sufficient statistics")
        require(cell["qty_sse"] >= 0 and cell["qty_absolute_error_sum"] >= 0, "Negative error sufficient statistics")
        if cell["count"]:
            require(all(type(cell.get(key)) in (int, float) and math.isfinite(cell[key]) for key in (*METRICS, "qty_bias")), "Nonfinite evaluated metrics")
            for key, expected in (("qty_rmse", math.sqrt(cell["qty_sse"] / cell["count"])),
                                  ("qty_mae", cell["qty_absolute_error_sum"] / cell["count"]),
                                  ("qty_bias", cell["qty_signed_error_sum"] / cell["count"]),
                                  ("time_nll", cell["time_nll_sum"] / cell["count"])):
                require(math.isclose(cell[key], expected, rel_tol=1e-12, abs_tol=1e-10), "Metric differs from sufficient statistics")
        else:
            require(all(cell.get(key) is None for key in (*METRICS, "qty_bias")), "Empty stratum has a score")
    for key in ("qty_sse", "qty_absolute_error_sum", "qty_signed_error_sum", "time_nll_sum"):
        require(math.isclose(metrics[key], math.fsum(cell[key] for cell in cells), rel_tol=1e-10, abs_tol=1e-8), "Strata do not reconstruct full sufficient statistics")
    if "quantity_cell_counts" in population:
        require([cell["count"] for cell in cells] == population["quantity_cell_counts"], "Frozen quantity-cell counts changed")


def validation_comparison(metrics, reference):
    deviations = {scope + "_" + key: abs(cell[key] - original[key])
        for scope, cell, original in (("full", metrics, reference), ("tail", metrics["tail"], reference["tail"]))
        for key in METRICS}
    require(all(math.isfinite(value) and value <= 1e-5 for value in deviations.values()), "Validation replay exceeds absolute tolerance 1e-5")
    return {"passed": True, "absolute_tolerance": 1e-5, "absolute_deviations": deviations}


def verify_result(result, c, contract_sha, host, split):
    require(result.get("schema") == SCHEMA and result.get("status") == "complete" and result.get("split") == split
            and result.get("host") == host and result.get("contract_sha256") == contract_sha
            and result.get("source_closure_sha256") == SOURCE_SHA
            and result.get("evaluator_sha256") == c["evaluator_sha256"]
            and all(result.get(key) is False for key in ("new_training", "checkpoint_reselection", "raw_predictions_written", "automatic_retry")), "Invalid evaluation receipt")
    runtime = result["runtime"]
    require(all(runtime.get(key) == value for key, value in RUNTIMES[host].items())
            and runtime.get("gpu", {}).get("uuid") == GPU_UUIDS[host] and runtime.get("threads") == 4, "Receipt CUDA runtime changed")
    expected = host_rows(c, host)
    rows = result["rows"]
    require(len(rows) == 5 and [row["id"] for row in rows] == [row["id"] for row in expected], "Evaluation receipt condition order/membership changed")
    population = c["population_references"][split]
    for row, original in zip(rows, expected):
        require(all(row.get(key) == original[key] for key in
                    ("id", "host", "arm", "seed", "dataset", "selected_epoch", "checkpoint_sha256", "state_sha256"))
                and row.get("state_sha256_before") == row.get("state_sha256_after") == original["state_sha256"]
                and row.get("checkpoint_file_sha256_after") == original["checkpoint_sha256"], "Receipt selection/state/bytes changed")
        require(row.get("population") == population, "Receipt target identity changed")
        validate_metrics(row["metrics"], population)
        if split == "validation":
            comparison = validation_comparison(row["metrics"], original["validation_reference"])
            require(row.get("validation_comparison") == comparison, "Validation comparison receipt changed")
    return result


def verify_validation_gate(c, root, contract_sha):
    gate_path = root / "validation_gate.json"
    gate = read(gate_path)
    require(gate.get("passed") is True and gate.get("conditions") == 10
            and gate.get("contract_sha256") == contract_sha, "Global ten-condition Validation gate missing")
    receipts = gate["receipts"]
    require(len(receipts) == 2 and {row.get("host") for row in receipts} == set(HOSTS), "Global gate requires both host receipts")
    for row in receipts:
        require(row.get("path") == "results/validation_" + row["host"] + ".json" and digest(row.get("sha256")), "Validation gate receipt path changed")
        path = confined(root, row["path"])
        require(sha_file(path) == row["sha256"], "Validation gate raw receipt SHA mismatch")
        verify_result(read(path), c, contract_sha, row["host"], "validation")
    return sha_file(gate_path)


class QuantityMetrics:
    """Streaming Python doubles (IEEE float64), with no retained predictions."""

    def __init__(self, bounds=BOUNDS):
        require(tuple(bounds) == BOUNDS, "Frozen quantity boundaries changed")
        self.bounds = tuple(bounds)
        self.sums = [[0, 0., 0., 0., 0.] for _ in range(5)]

    def add(self, quantities, predictions, temporal):
        q, p, t = (list(values) for values in (quantities, predictions, temporal))
        require(len(q) == len(p) == len(t), "Metric shapes differ")
        additions = [[] for _ in range(5)]
        for quantity, prediction, loss in zip(q, p, t):
            quantity, prediction, loss = float(quantity), float(prediction), float(loss)
            require(all(math.isfinite(value) for value in (quantity, prediction, loss))
                    and quantity >= 0 and prediction >= 0, "Invalid metric inputs")
            error = prediction - quantity
            additions[bisect.bisect_left(self.bounds, quantity)].append((abs(error), error ** 2, error, loss))
        for row, values in zip(self.sums, additions):
            row[0] += len(values)
            for column in range(4):
                row[column + 1] = math.fsum((row[column + 1], math.fsum(value[column] for value in values)))
        require(all(math.isfinite(value) for row in self.sums for value in row), "Metric aggregation overflow")

    @staticmethod
    def _finish(values):
        count, absolute, squared, signed, temporal = values
        return {"count": count, "qty_sse": squared, "qty_absolute_error_sum": absolute,
                "qty_signed_error_sum": signed, "time_nll_sum": temporal,
                "qty_rmse": math.sqrt(squared / count) if count else None,
                "qty_mae": absolute / count if count else None, "qty_bias": signed / count if count else None,
                "time_nll": temporal / count if count else None}

    def finish(self):
        overall = [sum(row[0] for row in self.sums)] + [math.fsum(row[column] for row in self.sums) for column in range(1, 5)]
        return {**self._finish(overall), "quantity_boundaries": list(self.bounds),
                "quantity_cells": [{"bin": index, **self._finish(row)} for index, row in enumerate(self.sums)],
                "tail": self._finish(self.sums[-1]), "tail_definition": "raw_quantity > 3449",
                "bias_definition": "prediction minus target; target-defined strata are descriptive"}


def population_identity(dataset, split, expected):
    """Use the original HistoryMLP identity and frozen duration conventions."""
    require(split in ("validation", "test"), "Foreign evaluation split")
    positions = [(int(part), int(end) + 1) for part, end in dataset.index]
    part = [item[0] for item in positions]
    position = [item[1] for item in positions]
    seq = [int(dataset.seq_lists[p][t]) for p, t in positions]
    quantities = [float(dataset.val_lists[p][t]) for p, t in positions]
    durations = [float(dataset.dt_lists[p][t]) for p, t in positions]
    for p, t in positions:
        require(dataset.split_lists[p][t] == split, "Target split differs from admitted split")
        require(all(int(dataset.seq_lists[p][i]) < int(dataset.seq_lists[p][t]) for i in range(t)), "Observed prefix contains a future/duplicate sequence")
        require(all(value in ({"train", "validation"} if split == "validation" else {"train", "validation", "test"})
                    for value in dataset.split_lists[p][:t]), "Observed prefix exposes a disallowed split")
    require(all(math.isfinite(q) and q >= 0 for q in quantities)
            and all(math.isfinite(dt) and dt >= 1 for dt in durations), "Invalid frozen targets")
    identity = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
    identity.update(json.dumps([str(part) for part in dataset.parts], ensure_ascii=False, separators=(",", ":")).encode())
    for values in (part, position, seq):
        identity.update(struct.pack("<" + "q" * len(values), *values))
    actual = {"target_count": len(quantities), "target_identity_sha256": identity.hexdigest(),
              "target_quantity_sha256": hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + struct.pack("<" + "d" * len(quantities), *quantities)).hexdigest(),
              "target_dt_sha256": hashlib.sha256(b"frozen_lognormal_target_dt_v1\0" + struct.pack("<" + "d" * len(durations), *durations)).hexdigest(),
              "tail_count": sum(q > 3449 for q in quantities)}
    extras = {"split": split, "entity_count": len(dataset.parts), "body_count": sum(q <= 1562 for q in quantities),
              "tail_min_exclusive": 3449, "body_max_inclusive": 1562,
              "target_dt_native_float32_raw_sha256": hashlib.sha256(struct.pack("<" + "f" * len(durations), *durations)).hexdigest()}
    actual.update({key: value for key, value in extras.items() if key in expected})
    if "quantity_cell_counts" in expected:
        actual["quantity_cell_counts"] = [sum(bisect.bisect_left(BOUNDS, q) == index for q in quantities) for index in range(5)]
    require(actual == expected, "Eligible target population identity changed")
    return quantities, actual


def validate_checkpoint(payload, row, state_digest):
    """Pure metadata checks; route validation and strict loading happen separately."""
    require(payload.get("backbone") == row["arm"] and payload.get("seed") == 42
            and payload.get("best_epoch") == row["selected_epoch"]
            and payload.get("epoch", row["selected_epoch"]) == row["selected_epoch"], "Selected checkpoint arm/seed/epoch changed")
    require(payload.get("source_revision") == row["source_revision"] == SOURCE_REVISION
            and payload.get("source_revision_history") == [SOURCE_REVISION], "Selected checkpoint source revision/lineage changed")
    require(type(payload.get("selected_metric_value")) in (int, float)
            and math.isfinite(payload["selected_metric_value"])
            and payload["selected_metric_value"] == row["selected_metric_value"], "Selected checkpoint metric value changed")
    require(payload.get("evaluation_scope") == "validation_only" and payload.get("held_out_test_evaluated") is False
            and payload.get("variant") == "count_only_log_regression"
            and payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
            and payload.get("checkpoint_monitor_history_key") == "val_qty_rmse"
            and payload.get("checkpoint_selection") == "best_validation_raw_quantity_rmse",
            "Frozen checkpoint scope/selector changed")
    require(payload.get("model_state_sha256") == state_digest == row["state_sha256"], "Selected checkpoint state digest changed")


def frozen_import(name, source):
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve().is_relative_to(source), "Foreign scientific import: " + name)
    return module


def bootstrap(c, root, host):
    verify_files(c, root)
    source = (root / "source").resolve()
    require(Path.cwd().resolve() == source, "Evaluation cwd must be the frozen source")
    require(all(os.environ.get(key) == value for key, value in
                (("PYTHONHASHSEED", "42"), ("OMP_NUM_THREADS", "4"), ("MKL_NUM_THREADS", "4"),
                 ("OPENBLAS_NUM_THREADS", "4"), ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"), ("NVIDIA_TF32_OVERRIDE", "0"))), "Pinned launch environment changed")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    runtime = frozen_import("paper.scripts.quantity_comparison_runtime", source)
    runtime.configure_runtime("cuda:0", 4)
    identity = runtime.runtime_identity("cuda:0")
    require(all(identity.get(key) == value for key, value in RUNTIMES[host].items())
            and identity["gpu"]["uuid"] == GPU_UUIDS[host] and identity["threads"] == 4,
            "Actual pinned CUDA runtime changed")
    engine = frozen_import("paper.scripts.run_titantpp_contextual_retrieval", source)
    engine.install_hooks()
    core = frozen_import("paper.scripts.count_aware_tpp_backbone.core", source)
    loader_module = frozen_import("paper.scripts.run_taxi_quantity_interface_ablation", source)
    runner = frozen_import("simple_lab_test.search.common.runner", source)
    factory = frozen_import("models.TPPs.CountAwareFactory", source)
    return runtime, identity, engine, core, loader_module, runner, factory


def scientific_imports(c, root):
    source = (root / "source").resolve()
    for name, module in tuple(sys.modules.items()):
        if name in ("paper", "models", "simple_lab_test", "data_loader", "utils") or name.startswith(
                ("paper.", "models.", "simple_lab_test.", "data_loader.", "utils.")):
            path = getattr(module, "__file__", None)
            if path:
                path = Path(path).resolve()
                require(path.is_relative_to(source), "Foreign scientific module loaded: " + name)
                relative = path.relative_to(source).as_posix()
                require(relative in c["source_files"] and sha_file(path) == c["source_files"][relative], "Unsealed scientific module loaded: " + name)


def evaluate_rows(c, root, host, split, contract_sha, deadline):
    runtime, identity, engine, core, loader_module, runner, factory = bootstrap(c, root, host)
    import polars as pl
    import torch
    def budget():
        require(time.monotonic() < deadline, "Host/split evaluation deadline exceeded")
        require(max(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)) <= RESOURCES["gpu_memory_limit_bytes"], "CUDA allocation ceiling exceeded")
    allowed = ["train", "validation"] if split == "validation" else ["train", "validation", "test"]
    frame = pl.scan_parquet(confined(root, c["data"]["path"])).filter(pl.col("chronological_split").is_in(allowed)).collect().sort(["oper_part_no", "seq"])
    require(set(frame["chronological_split"].to_list()) <= set(allowed), "Frame contains a disallowed split")
    frame = core.prepare_count_frame(frame)
    loader = loader_module.make_loader(frame, target_split=split, batch_size=128, lookback_weeks=168,
                                      max_seq_len=256, shuffle=False, generator=None)
    target_quantity, population = population_identity(loader.dataset, split, c["population_references"][split])
    completed = []
    torch.cuda.reset_peak_memory_stats(0)
    for row in host_rows(c, host):
        budget()
        path = confined(root, row["checkpoint_path"])
        require(sha_file(path) == row["checkpoint_sha256"], "Checkpoint changed before inference")
        payload = runner.torch_load_checkpoint(path, map_location="cpu")
        factory.validate_checkpoint_route(payload, row["arm"])
        state = runner.canonical_state_dict_sha256(payload["model_state_dict"])
        validate_checkpoint(payload, row, state)
        model, _ = engine.build_model(c["data_spec"], row["arm"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model = model.to("cuda:0").eval().requires_grad_(False)
        before = runner.canonical_state_dict_sha256(model.state_dict())
        require(before == state and not model.training and all(not p.requires_grad for p in model.parameters()), "Evaluation state/gradient policy changed")
        metrics, position = QuantityMetrics(), 0
        with torch.no_grad():
            for _, dt, mask, _, quantity in loader:
                budget()
                outputs = core.target_outputs(model, dt.to("cuda:0"), mask.to("cuda:0"), quantity.to("cuda:0"), lambda_log_qty=1.)
                q, p, temporal = (outputs[key].detach().cpu().double().tolist() for key in ("true_qty", "pred_qty", "time_loss"))
                require(q == target_quantity[position:position + len(q)], "Inference target order changed")
                metrics.add(q, p, temporal)
                position += len(q)
                budget()
        require(position == population["target_count"], "Evaluation skipped/duplicated targets")
        after = runner.canonical_state_dict_sha256(model.state_dict())
        require(before == after == state, "Evaluation mutated model parameters or buffers")
        require(sha_file(path) == row["checkpoint_sha256"], "Evaluation mutated checkpoint file")
        aggregate = metrics.finish()
        validate_metrics(aggregate, population)
        comparison = validation_comparison(aggregate, row["validation_reference"]) if split == "validation" else None
        completed.append({key: row[key] for key in ("id", "host", "arm", "seed", "dataset", "selected_epoch", "checkpoint_sha256", "state_sha256")}
                         | {"state_sha256_before": before, "state_sha256_after": after,
                            "checkpoint_file_sha256_after": sha_file(path), "population": population,
                            "metrics": aggregate, "validation_comparison": comparison})
        del outputs, payload, model
        torch.cuda.empty_cache()
    budget()
    scientific_imports(c, root)
    verify_files(c, root)
    return {"schema": SCHEMA, "status": "complete", "host": host, "split": split, "contract_sha256": contract_sha,
            "source_closure_sha256": SOURCE_SHA, "evaluator_sha256": c["evaluator_sha256"], "runtime": identity,
            "rows": completed, "new_training": False, "checkpoint_reselection": False, "raw_predictions_written": False,
            "automatic_retry": False, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "Test_exposure_lineage": c["test_exposure_lineage"],
            "completed_unix": time.time()}


def process_group(pgid, *, proc_root=Path("/proc")):
    require(proc_root.is_dir(), "Native /proc resource monitoring is required")
    pids, rss = set(), 0
    for folder in proc_root.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                pids.add(int(folder.name))
                lines = (folder / "status").read_text().splitlines()
                rss += sum(int(line.split()[1]) * 1024 for line in lines if line.startswith("VmRSS:"))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return pids, rss


def gpu_process_bytes(pids):
    completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True, timeout=10)
    memory = 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
            require(fields[1].isdigit(), "GPU process memory monitoring unavailable")
            memory += int(fields[1]) * 1024 ** 2
    return memory


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def run_worker(c, root, contract_path, host, split, token, deadline):
    log = root / "logs" / (split + "_" + host + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--contract", str(contract_path), "--host", host,
               "--split", split, "--worker"]
    process, max_rss, max_gpu = None, 0, 0
    environment = dict(os.environ)
    environment["TITANTPP_RETRIEVAL_TEST_WORKER_TOKEN"] = token
    try:
        with log.open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       cwd=root / "source", env=environment)
            while process.poll() is None:
                require(time.monotonic() < deadline, "Host/split 900-second deadline exceeded")
                pids, rss = process_group(process.pid)
                max_rss = max(max_rss, rss)
                require(max_rss <= RESOURCES["rss_limit_bytes"], "Owned process-group RSS ceiling exceeded")
                max_gpu = max(max_gpu, gpu_process_bytes(pids))
                require(max_gpu <= RESOURCES["gpu_memory_limit_bytes"], "Owned process-group GPU ceiling exceeded")
                time.sleep(.25)
            require(process.returncode == 0, "Frozen inference worker failed; retained log " + str(log))
        result = read(root / "receipts" / (split + "_" + host + ".json"))
        result.update(peak_observed_process_group_rss_bytes=max_rss, peak_observed_process_group_gpu_bytes=max_gpu)
        return result
    finally:
        stop_owned(process)


def claim(root, c, contract_path, host, split, token, deadline):
    path = root / "claims" / (split + "_" + host + ".json")
    require(not (root / "results" / (split + "_" + host + ".json")).exists(), "Existing split result; retry prohibited")
    write_exclusive(path, {"schema": SCHEMA, "host": host, "split": split,
        "contract_sha256": sha_file(contract_path), "parent_pid": os.getpid(), "started_unix": time.time(),
        "deadline_monotonic": deadline, "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "automatic_retry": False})
    return path


def pipeline(contract_path, host, split, *, runner=run_worker):
    contract_path = Path(contract_path).resolve()
    root = contract_path.parent
    c = validate_contract(read(contract_path))
    require(host in HOSTS and split in ("validation", "test"), "Foreign host/split")
    verify_files(c, root)
    contract_sha = sha_file(contract_path)
    token, deadline = secrets.token_hex(32), time.monotonic() + RESOURCES["host_split_timeout_seconds"]
    claim(root, c, contract_path, host, split, token, deadline)
    result_path = root / "results" / (split + "_" + host + ".json")
    try:
        gate_sha = verify_validation_gate(c, root, contract_sha) if split == "test" else None
        result = runner(c, root, contract_path, host, split, token, deadline)
        verify_result(result, c, contract_sha, host, split)
        require(sha_file(contract_path) == contract_sha, "Contract bytes changed during evaluation")
        verify_files(c, root)
        result["validation_gate_sha256"] = gate_sha
        result["completed_unix"] = time.time()
        write_exclusive(result_path, result)
        return result
    except BaseException as error:
        write_exclusive(result_path, {"schema": SCHEMA, "status": "failed", "host": host, "split": split,
            "contract_sha256": contract_sha, "exception": type(error).__name__, "message": str(error),
            "automatic_retry": False, "raw_predictions_written": False, "new_training": False,
            "checkpoint_reselection": False, "failed_unix": time.time()})
        raise


def worker(contract_path, host, split):
    contract_path = Path(contract_path).resolve()
    root, contract_sha = contract_path.parent, sha_file(contract_path)
    c = validate_contract(read(contract_path))
    launch = read(root / "claims" / (split + "_" + host + ".json"))
    token = os.environ.get("TITANTPP_RETRIEVAL_TEST_WORKER_TOKEN", "")
    require(launch.get("host") == host and launch.get("split") == split and launch.get("contract_sha256") == contract_sha
            and launch.get("parent_pid") == os.getppid() and token
            and launch.get("token_sha256") == hashlib.sha256(token.encode()).hexdigest(), "Owned one-shot worker launch missing")
    if split == "test":
        verify_validation_gate(c, root, contract_sha)
    result = evaluate_rows(c, root, host, split, contract_sha, launch["deadline_monotonic"])
    write_exclusive(root / "receipts" / (split + "_" + host + ".json"), result)


def interrupted(signum, frame):
    raise InterruptedError("Evaluation interrupted by signal " + str(signum))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.worker:
            worker(args.contract, args.host, args.split)
        else:
            result = pipeline(args.contract, args.host, args.split)
            print(json.dumps({"status": result["status"], "host": args.host, "split": args.split, "conditions": len(result["rows"])}))
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
