#!/usr/bin/env python3
"""Hash-bound diagnostic inference of already selected models, one split/process.

This operational adapter never trains, refits, reselects, edits old artifacts or
connects to a server. Scientific model/loader functions are imported only from
the explicitly pinned frozen source after approval and byte checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SCHEMA = "titantpp_frozen_row_diagnostics_v1"
SPLITS = ("train", "validation", "test")
PREFIXES = {"paper", "models", "data_loader", "simple_lab_test", "utils"}
ARMS = {"titantpp_cnn_gru54", "titantpp_cnn_gru54_cnn_only"}
EXTERNALS = {"rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head"}
TRUTH_FIELDS = ("target_id", "entity_id", "seq", "split", "recorded_date", "raw_quantity",
                "recorded_gap", "history_length", "last_quantity", "mean_quantity")
METRICS = ("qty_rmse", "qty_mae", "time_nll")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def canonical(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            result.update(block)
    return result.hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    require(not Path(relative).is_absolute(), "Artifact path must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Artifact path escapes its pinned root")
    return path


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_reference(root, reference):
    path = confined(root, reference["path"])
    require(sha(path) == reference["sha256"], "Pinned reference bytes changed")
    return read(path)


def verify_contract(root, expected_sha, job_id, split, *, now=None):
    path = Path(root) / "contract.json"
    require(sha(path) == expected_sha, "Diagnostic contract byte SHA mismatch")
    c = read(path)
    require(c.get("schema") == SCHEMA and c.get("new_training") is False
            and c.get("frozen_selection") is True, "Wrong diagnostic scope")
    require(c.get("allowed_splits") == list(SPLITS) and split in SPLITS,
            "Split is outside the explicit diagnostic authorization")
    require(c.get("evaluator_sha256") == sha(__file__), "Diagnostic evaluator bytes differ")
    approval = verify_reference(root, c["approval"])
    require(approval.get("authorized") is True and approval.get("new_training") is False
            and approval.get("frozen_selection") is True
            and approval.get("allowed_splits") == list(SPLITS), "Diagnostic approval is missing or changed")
    clock = time.time() if now is None else now
    require(type(c.get("deadline_unix")) in (float, int) and clock < c["deadline_unix"],
            "Diagnostic deadline expired; automatic retry is forbidden")
    jobs = c.get("jobs", [])
    require(len(jobs) == len({j.get("id") for j in jobs}), "Duplicate diagnostic job identity")
    matching = [j for j in jobs if j.get("id") == job_id]
    require(len(matching) == 1, "Unknown diagnostic job")
    job = matching[0]
    require(split in job.get("allowed_splits", []), "This model split is not approved for new inference")
    require(job.get("family") in ("component", "external"), "Unsupported scientific family")
    require(job.get("seed") in (42, 52, 62), "Unapproved seed")
    allowed = ARMS if job["family"] == "component" else EXTERNALS
    require(job.get("arm") in allowed, "Unapproved model route")
    require(job["id"] == f"{job['dataset']}__{job['seed']}__{job['arm']}", "Unsafe job identity")
    require(job["dataset"] in ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000"),
            "Unapproved dataset")
    require(set(job["expected_populations"]) == set(SPLITS), "All three target populations must be bound")
    require(set(job["reference_receipts"]) == {"validation", "test"}, "Existing evaluation reference missing")
    return c, job


def file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Operational adapter cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_job(job):
    """Normalize pinned component parents or pinned external registry entries."""
    parent = Path(job["parent_root"]).resolve()
    if job["family"] == "component":
        require(sha(parent / "contract.json") == job["parent_contract_raw_sha256"], "Parent contract changed")
        evaluator = confined(parent, job["parent_evaluator_relative_path"])
        require(sha(evaluator) == job["parent_evaluator_sha256"], "Parent operational evaluator changed")
        helper = file_module(evaluator, "_pinned_component_evaluator")
        c = helper.verify_contract(parent)
        rows = [r for r in c["jobs"] if r["id"] == job["parent_job_id"]]
        require(len(rows) == 1, "Parent condition is absent or duplicate")
        original = rows[0]
        require(all(original[k] == job[k] for k in ("dataset", "seed", "arm")), "Parent model identity differs")
        source = c["sources"]["components"]
        source_root = helper.verify_source(parent, source)
        definition = c["datasets"][job["dataset"]]["definition"]
        data = c["datasets"][job["dataset"]]
        checkpoint = original["checkpoint"]
        return dict(parent=parent, helper=helper, original=original, source_root=source_root,
                    source_files=source["files"], source_closure=source["closure"], source=source,
                    definition=definition, data=dict(path=data["data_path"], sha256=data["data_sha256"]),
                    checkpoint=dict(path=checkpoint["relative_path"], sha256=checkpoint["sha256"]),
                    state_sha=original["checkpoint_state_sha256"], runtime=c["runtimes"][original["evaluation_host"]],
                    route=helper.ROUTE, selected_epoch=original["selected_epoch"])
    registry = verify_reference(parent, job["legacy_registry"])
    rows = [r for r in registry["rows"] if (r["dataset"], r["model"], r["seed"])
            == (job["dataset"], job["arm"], job["seed"])]
    require(len(rows) == 1 and rows[0] == job["legacy_registry_row"], "Pinned external registry row changed")
    original = rows[0]
    bundle = registry["bundles"][original["evaluator_source_bundle"]]
    require(bundle == job["source_bundle"], "Pinned external source bundle changed")
    definitions = [d for d in bundle["datasets"] if d["dataset_id"] == job["dataset"]]
    require(len(definitions) == 1 and definitions[0] == job["definition"], "Frozen external data/model definition changed")
    require(original["endpoint"] == "selected" and original["selection"] == "strict_first_min_validation_raw_quantity_rmse",
            "External checkpoint is not the frozen Validation selection")
    source_root = confined(parent, bundle["source_root"])
    return dict(parent=parent, helper=None, original=original, source_root=source_root,
                source_files=bundle["source_files"], source_closure=bundle["source_closure_sha256"], source=bundle,
                definition=job["definition"], data=job["data"],
                checkpoint=dict(path=original["checkpoint_path"], sha256=original["checkpoint_file_sha256"]),
                state_sha=original["state_tensor_sha256"], runtime=job["runtime"], route=None,
                selected_epoch=original["selected_epoch"])


def verify_science(spec, *, imports=False):
    root, files = spec["source_root"], spec["source_files"]
    require(files, "Empty scientific source manifest")
    for relative, expected in files.items():
        require(sha(confined(root, relative)) == expected, "Frozen scientific bytes differ: " + relative)
    imported = {}
    if imports:
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] not in PREFIXES or not getattr(module, "__file__", None):
                continue
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(root), "Current/nonfrozen scientific import: " + name)
            relative = path.relative_to(root).as_posix()
            require(relative in files and sha(path) == files[relative], "Unsealed scientific import: " + name)
            imported[name] = relative
    return imported


def isolate(source):
    require(not any(name.split(".")[0] in PREFIXES for name in sys.modules),
            "A fresh standalone process is required")
    sys.dont_write_bytecode = True
    sys.path[:] = [str(source)] + [p for p in sys.path if p and not (Path(p).resolve() / "paper").is_dir()]


def check_runtime(actual, expected):
    """Native versions, device identity and numerical policy are all explicit."""
    require(Path(sys.executable).resolve() == Path(expected["python"]).resolve(), "Wrong native Python executable")
    for key, expected_key in (("python", "python_version"), ("torch", "torch"), ("cuda", "cuda"),
                              ("cudnn", "cudnn"), ("numpy", "numpy"), ("polars", "polars"),
                              ("threads", "torch_num_threads"), ("interop_threads", "interop_threads")):
        require(actual.get(key) == expected[expected_key], "Native Runtime changed: " + key)
    require(actual.get("gpu", {}).get("uuid") == expected["gpu_uuid"], "Wrong physical GPU")
    require(actual.get("default_dtype") == "torch.float32" and actual.get("deterministic_algorithms") is True
            and actual.get("cuda_matmul_tf32") is False and actual.get("cudnn_tf32") is False,
            "Native numerical policy differs")
    for key, value in expected.get("native", {}).items():
        require(actual.get(key) == value, "Detailed native fingerprint differs: " + key)


def qualify(torch, model, target_outputs, dts, mask, quantities, outputs, top_code):
    altered = quantities.clone()
    altered[:, -1] += 12345.
    quantity_probe = target_outputs(model, dts, mask, altered, lambda_log_qty=1.)
    altered_dt = dts.clone()
    altered_dt[:, -1] = (torch.where(dts[:, -1] > 1, torch.ones_like(dts[:, -1]),
                                    torch.full_like(dts[:, -1], 2.)) if top_code else dts[:, -1] + 123.)
    gap_probe = target_outputs(model, altered_dt, mask, quantities, lambda_log_qty=1.)
    single = target_outputs(model, dts[:1], mask[:1], quantities[:1], lambda_log_qty=1.)
    checks = {
        "target_quantity_perturbation_preserves_prediction": torch.allclose(outputs["pred_qty"], quantity_probe["pred_qty"], rtol=1e-6, atol=1e-6),
        "target_quantity_perturbation_preserves_time_loss": torch.allclose(outputs["time_loss"], quantity_probe["time_loss"], rtol=1e-6, atol=1e-6),
        "target_gap_perturbation_preserves_quantity_prediction": torch.allclose(outputs["pred_qty"], gap_probe["pred_qty"], rtol=1e-6, atol=1e-6),
        "single_vs_batch_prediction_matches": torch.allclose(outputs["pred_qty"][:1], single["pred_qty"], rtol=1e-5, atol=1e-5),
        "single_vs_batch_time_loss_matches": torch.allclose(outputs["time_loss"][:1], single["time_loss"], rtol=1e-5, atol=1e-5),
    }
    require(all(checks.values()), "Target causality or batch consistency failed: " + str(checks))
    return checks


def compare_metrics(actual, reference):
    for group in (None, "tail"):
        left, right = (actual, reference) if group is None else (actual[group], reference[group])
        require(left["count"] == right["count"], "Native replay target count differs")
        for key in METRICS:
            if left["count"] == 0:
                require(left[key] is None and right[key] is None, "Empty metric is not N/A")
            else:
                require(math.isfinite(left[key]) and math.isfinite(right[key])
                        and math.isclose(left[key], right[key], rel_tol=1e-5, abs_tol=1e-5),
                        "Native selected-model replay metric differs: " + str(group) + "/" + key)
    if "quantity_cells" in reference:
        require(actual["quantity_boundaries"] == reference["quantity_boundaries"],
                "Native replay quantity boundaries differ")
        left_cells, right_cells = actual["quantity_cells"], reference["quantity_cells"]
        require(len(left_cells) == len(right_cells) == 5, "Native replay must preserve all five quantity bins")
        require([cell.get("bin") for cell in left_cells] == [cell.get("bin") for cell in right_cells] == list(range(5)),
                "Native replay quantity bin identity/order differs")
        for left, right in zip(left_cells, right_cells):
            require(type(right.get("count")) is int and left["count"] == right["count"] >= 0,
                    "Native replay quantity bin target count differs")
            # Older external endpoints retain MAE rather than an explicit AE sum.
            # Recover that sum from its recorded mean and exact event count.
            expected_sse = right["qty_sse"]
            expected_absolute = right.get("qty_absolute_error_sum")
            if expected_absolute is None:
                expected_absolute = right["qty_mae"] * right["count"] if right["count"] else 0.
            for key, expected in (("qty_sse", expected_sse), ("qty_absolute_error_sum", expected_absolute)):
                require(type(expected) in (float, int) and math.isfinite(expected) and expected >= 0
                        and math.isclose(left[key], expected, rel_tol=1e-5, abs_tol=1e-5),
                        "Native replay quantity bin sufficient sum differs: " + str(right["bin"]) + "/" + key)


def verify_population(actual, expected):
    require(actual["count"] == expected.get("count", expected.get("target_count")), "Bound population count differs")
    for key in ("target_identity_sha256", "truth_sha256", "time_sha256"):
        if key in expected:
            require(actual[key] == expected[key], "Bound raw target population differs: " + key)


class RawMetrics:
    """Operational sufficient sums; identical units/boundaries to prior receipts."""
    def __init__(self, bounds, np):
        self.np = np
        self.bounds = np.asarray(bounds, dtype=np.float64)
        require(self.bounds.shape == (4,) and np.isfinite(self.bounds).all()
                and np.all(np.diff(self.bounds) >= 0), "Invalid frozen quantity bins")
        self.sums = np.zeros((6, 5), dtype=np.float64)

    def add(self, q, prediction, time_nll):
        np = self.np
        q, p, t = (np.asarray(v, dtype=np.float64) for v in (q, prediction, time_nll))
        require(q.ndim == 1 and q.shape == p.shape == t.shape
                and all(np.isfinite(v).all() for v in (q, p, t))
                and (q >= 0).all() and (p >= 0).all(), "Invalid metric inputs")
        groups, error = np.searchsorted(self.bounds, q, side="left"), p - q
        for index in range(6):
            selected = np.ones(len(q), dtype=bool) if index == 0 else groups == index - 1
            e = error[selected]
            self.sums[index] += [selected.sum(), abs(e).sum(), (e ** 2).sum(), e.sum(), t[selected].sum()]
        require(np.isfinite(self.sums).all(), "Raw error sum overflow")

    @staticmethod
    def cell(sums):
        n, absolute, squared, signed, temporal = sums
        return dict(count=int(n), represented_n=float(n), qty_sse=float(squared),
                    qty_absolute_error_sum=float(absolute), qty_signed_error_sum=float(signed), time_nll_sum=float(temporal),
                    qty_rmse=math.sqrt(squared / n) if n else None, qty_mae=float(absolute / n) if n else None,
                    qty_bias=float(signed / n) if n else None, time_nll=float(temporal / n) if n else None)

    def finish(self):
        require(self.np.allclose(self.sums[1:].sum(axis=0), self.sums[0], rtol=1e-10, atol=1e-8),
                "Quantity bins do not reconstruct total errors")
        return {**self.cell(self.sums[0]), "quantity_boundaries": self.bounds.tolist(),
                "quantity_cells": [{"bin": index, **self.cell(row)} for index, row in enumerate(self.sums[1:])],
                "tail": self.cell(self.sums[-1]), "tail_definition": "raw_quantity > quantity_boundaries[-1]",
                "bias_definition": "prediction minus target; target-defined strata are descriptive"}


class PredictionWriter:
    def __init__(self, root, pl, *, part_rows=8192):
        self.root, self.pl, self.part_rows = Path(root), pl, int(part_rows)
        require(self.part_rows > 0, "Prediction chunk size must be positive")
        self.rows, self.parts, self.count = [], [], 0
        self.root.mkdir(parents=True, exist_ok=False)

    def add(self, rows):
        self.rows.extend(rows)
        while len(self.rows) >= self.part_rows:
            self.flush(self.part_rows)

    def flush(self, count=None):
        count = len(self.rows) if count is None else count
        if not count:
            return
        rows, self.rows = self.rows[:count], self.rows[count:]
        path = self.root / f"part-{len(self.parts):05d}.parquet"
        with path.open("xb") as stream:
            self.pl.DataFrame(rows).write_parquet(stream, compression="zstd")
            stream.flush()
            os.fsync(stream.fileno())
        self.parts.append({"path": path.name, "sha256": sha(path), "rows": len(rows), "bytes": path.stat().st_size})
        self.count += len(rows)

    def finish(self):
        self.flush()
        require(sum(p["rows"] for p in self.parts) == self.count, "Prediction manifest count differs")
        for part in self.parts:
            require(sha(self.root / part["path"]) == part["sha256"], "Prediction bytes changed")
        return {"parts": self.parts, "prediction_rows": self.count, "compression": "zstd"}


def verify_reused_validation(root, job, bridge, entry, receipt, actual_exit, contract_sha):
    """An explicit byte-bound bridge can reuse a completed component Validation.

    The gate belongs to this new contract; only the scientific receipt and its
    actual worker exit retain the previous contract. No old artifact is edited.
    """
    require(job["family"] == "component", "Only an explicitly completed component Validation may be reused")
    require(isinstance(bridge, dict) and set(bridge) == {"receipt_sha256", "worker_exit_sha256", "contract_sha256"},
            "Reused Validation bridge must pin exactly receipt/exit/old contract")
    require(bridge["contract_sha256"] != contract_sha
            and receipt.get("contract_raw_sha256") == actual_exit.get("contract_sha256") == bridge["contract_sha256"],
            "Reused Validation old contract differs")
    require(all(entry.get(key) == bridge[key] for key in ("receipt_sha256", "worker_exit_sha256")),
            "Gate references differ from explicit reused Validation byte pins")
    require(actual_exit.get("split") == "validation", "Reused exit belongs to another split")
    spec = resolve_job(job)
    require(all(receipt.get(key) == job[key] for key in ("dataset", "arm", "seed", "family")),
            "Reused Validation scientific job identity differs")
    for key, expected in (("checkpoint_file_sha256", spec["checkpoint"]["sha256"]),
                          ("state_sha256_before", spec["state_sha"]), ("state_sha256_after", spec["state_sha"]),
                          ("selected_epoch", spec["selected_epoch"]), ("source_closure", spec["source_closure"]),
                          ("data_sha256", spec["data"]["sha256"])):
        require(receipt.get(key) == expected, "Reused Validation science binding differs: " + key)
    require(receipt.get("new_training") is False and receipt.get("checkpoint_reselection") is False
            and receipt.get("full_population") is True and receipt.get("raw_predictions_written") is True,
            "Reused Validation is not a complete frozen diagnostic")
    qualification = receipt.get("qualification", {})
    keys = {"target_quantity_perturbation_preserves_prediction", "target_quantity_perturbation_preserves_time_loss",
            "target_gap_perturbation_preserves_quantity_prediction", "single_vs_batch_prediction_matches",
            "single_vs_batch_time_loss_matches", "parameters_unchanged"}
    require(all(qualification.get(key) is True for key in keys), "Reused Validation causality/batch/state qualification is incomplete")
    verify_population(receipt["population"], job["expected_populations"]["validation"])
    original = job["expected_populations"]["validation"].get("original_population_identity")
    if original:
        require(receipt.get("original_population_identity") == original, "Reused original loader population differs")
    reference = verify_reference(spec["parent"], job["reference_receipts"]["validation"])
    compare_metrics(receipt["metrics"], reference.get("metrics", reference))
    require(receipt.get("runtime") == receipt.get("runtime_after"), "Reused native Runtime changed during inference")
    check_runtime(receipt["runtime"], spec["runtime"])
    base = "results/" + job["id"] + "/validation/"
    manifest = verify_reference(root, dict(path=base + "predictions_manifest.json", sha256=receipt["predictions_manifest_sha256"]))
    require(manifest.get("prediction_rows") == receipt.get("prediction_rows") == receipt["population"]["count"]
            == sum(part["rows"] for part in manifest["parts"]), "Reused Validation prediction manifest count differs")
    require(len(manifest["parts"]) == len({part["path"] for part in manifest["parts"]}), "Reused prediction manifest duplicates a part")
    for part in manifest["parts"]:
        path = confined(root, base + "predictions/" + part["path"])
        require(path.stat().st_size == part["bytes"] and sha(path) == part["sha256"], "Reused Validation prediction bytes differ")


def verify_validation_gate(root, job, contract_sha):
    gate = read(Path(root) / "validation_gate.json")
    require(gate.get("status") == "complete" and gate.get("contract_sha256") == contract_sha
            and gate.get("actual_worker_exit0") is True,
            "New diagnostic Validation gate is missing or foreign")
    entry = gate.get("jobs", {}).get(job["id"])
    require(isinstance(entry, dict), "New diagnostic Validation gate omits this model")
    base = "results/" + job["id"] + "/"
    receipt = verify_reference(root, dict(path=base + "validation/receipt.json", sha256=entry["receipt_sha256"]))
    actual_exit = verify_reference(root, dict(path=base + "validation_worker_exit.json", sha256=entry["worker_exit_sha256"]))
    require(receipt.get("status") == "complete" and receipt.get("split") == "validation"
            and receipt.get("job_id") == job["id"]
            and receipt.get("reference_metrics_match") == "PASS", "Diagnostic Validation did not pass")
    require(actual_exit.get("exit_code") == 0 and actual_exit.get("actual_child_exit_observed") is True
            and actual_exit.get("job_id") == job["id"], "Diagnostic Validation actual worker exit0 missing")
    require(sha(Path(root) / "contract.json") == contract_sha, "Current gate contract byte SHA differs")
    c = read(Path(root) / "contract.json")
    reuse = c.get("reused_validation", {})
    require(isinstance(reuse, dict) and set(reuse) <= {j["id"] for j in c["jobs"]}, "Reused Validation bridge has a foreign job ID")
    if job["id"] in reuse:
        verify_reused_validation(root, job, reuse[job["id"]], entry, receipt, actual_exit, contract_sha)
    else:
        require(receipt.get("contract_raw_sha256") == contract_sha, "Diagnostic Validation contract differs; no reuse bridge")
    return sha(Path(root) / "validation_gate.json")


def worker(root, job_id, split, contract_sha):
    root = Path(root).resolve()
    c, job = verify_contract(root, contract_sha, job_id, split)
    gate_sha = verify_validation_gate(root, job, contract_sha) if split == "test" else None
    output = root / "results" / job_id / split
    output.mkdir(parents=True, exist_ok=False)  # A claim/failure prohibits repeat.
    started, started_unix, phase = time.monotonic(), time.time(), "integrity"
    write_exclusive(output / "claim.json", {"job_id": job_id, "split": split, "contract_raw_sha256": contract_sha,
                                           "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv})
    def budget():
        require(time.time() < c["deadline_unix"], "Diagnostic deadline expired; automatic retry forbidden")
    try:
        spec = resolve_job(job)
        verify_science(spec)
        parent, definition = spec["parent"], spec["definition"]
        checkpoint, data_path = confined(parent, spec["checkpoint"]["path"]), confined(parent, spec["data"]["path"])
        require(sha(checkpoint) == spec["checkpoint"]["sha256"], "Selected checkpoint bytes changed")
        require(sha(data_path) == spec["data"]["sha256"], "Frozen input bytes changed")
        references = {key: verify_reference(parent, value) for key, value in job["reference_receipts"].items()}
        budget()
        isolate(spec["source_root"])
        os.environ["MPLCONFIGDIR"] = str(output / "runtime_cache" / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(output / "runtime_cache")
        import numpy as np
        import polars as pl
        import torch
        numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
        numerical.configure_runtime("cuda:0", spec["runtime"]["torch_num_threads"])
        runtime = numerical.runtime_identity("cuda:0")
        check_runtime(runtime, spec["runtime"])
        if spec["route"]:
            importlib.import_module(spec["route"]).install_hooks()
        factory = importlib.import_module("models.TPPs.CountAwareFactory")
        core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
        loader_module = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
        runner = importlib.import_module("simple_lab_test.search.common.runner")
        verify_science(spec, imports=True)
        payload = runner.torch_load_checkpoint(checkpoint, map_location="cpu")
        require(payload.get("best_epoch") == spec["selected_epoch"] and payload.get("backbone") == job["arm"]
                and payload.get("seed") == job["seed"], "Selected checkpoint arm/seed/epoch differs")
        require(payload.get("variant") == "count_only_log_regression" and payload.get("held_out_test_evaluated") is False
                and payload.get("evaluation_scope") == "validation_only"
                and payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
                and payload.get("checkpoint_selection") == "best_validation_raw_quantity_rmse",
                "Original checkpoint objective/selection differs")
        if spec["helper"]:
            metadata = verify_reference(parent, {"path": spec["original"]["checkpoint_metadata_path"],
                                                 "sha256": spec["original"]["checkpoint_metadata_sha256"]})
            spec["helper"].validate_payload(payload, spec["original"], spec["source"], definition, metadata)
            history = spec["helper"].read_selection_history(confined(parent, spec["original"]["selection_history_path"]))
            require(spec["helper"].strict_selected_epoch(history) == (spec["selected_epoch"], spec["original"]["selected_metric_value"]),
                    "Original strict-first Validation selection differs")
        factory.validate_checkpoint_route(payload, job["arm"])
        require(runner.canonical_state_dict_sha256(payload["model_state_dict"]) == spec["state_sha"], "Selected tensor SHA differs")
        phase = "population"
        allowed = list(SPLITS[:SPLITS.index(split) + 1])
        frame = pl.scan_parquet(data_path).filter(pl.col("chronological_split").is_in(allowed)).collect().sort(["oper_part_no", "seq"])
        require(frame.height > 0 and set(frame["chronological_split"].unique()) <= set(allowed), "Disallowed split materialized")
        require(not frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().any()).item(), "Duplicate target key")
        rank = frame["chronological_split"].replace_strict({"train": 0, "validation": 1, "test": 2})
        checked = frame.with_columns(rank.alias("_rank")).with_columns(pl.col("_rank").cum_max().over("oper_part_no").alias("_max"))
        require(checked.filter(pl.col("_rank") < pl.col("_max")).height == 0, "Per-series chronological target order differs")
        train_q = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
        stats = definition["statistics"]
        for key, value in (("train_log_mean", float(np.log1p(train_q).mean())), ("train_log_std", float(np.log1p(train_q).std()))):
            require(math.isclose(value, stats[key], rel_tol=0, abs_tol=1e-12), "Frozen Train-only initialization differs: " + key)
        if stats.get("all_train_quantity_sha256"):
            digest = hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0" + train_q.astype("<f8").tobytes()).hexdigest()
            require(digest == stats["all_train_quantity_sha256"], "Frozen all-Train quantity identity differs")
        frame = core.prepare_count_frame(frame)
        loader = loader_module.make_loader(frame, target_split=split, batch_size=128, shuffle=False, generator=None,
                                          lookback_weeks=definition["loader"]["lookback_weeks"], max_seq_len=definition["loader"]["max_seq_len"])
        dataset = loader.dataset
        ix = np.asarray(dataset.index, dtype=np.int64).reshape(-1, 2)
        require(len(ix) == job["expected_populations"][split].get("count", job["expected_populations"][split].get("target_count")) > 0,
                "Eligible next-event target count differs")
        index_digest = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
        index_digest.update(json.dumps([str(p) for p in dataset.parts], ensure_ascii=False, separators=(",", ":")).encode())
        part, position = ix[:, 0], ix[:, 1] + 1
        seq = np.fromiter((dataset.seq_lists[p][t] for p, t in zip(part, position)), dtype=np.int64)
        target_q = np.fromiter((dataset.val_lists[p][t] for p, t in zip(part, position)), dtype=np.float64)
        for array in (part, position, seq):
            index_digest.update(array.astype("<i8").tobytes())
        inherited_identity = {"target_count": len(ix), "target_identity_sha256": index_digest.hexdigest(),
                              "target_quantity_sha256": hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + target_q.astype("<f8").tobytes()).hexdigest()}
        expected_population = job["expected_populations"][split]
        inherited_reference = (expected_population.get("original_population_identity")
                               or (expected_population if "target_quantity_sha256" in expected_population else None)
                               or definition.get("inherited_data_identity", {}).get("populations", {}).get(split))
        if inherited_reference:
            require(inherited_identity == inherited_reference, "Original next-event population differs")
        metadata = frame.select(["oper_part_no", "seq", "chronological_split", "demand_dt"])
        offsets, total = [], 0
        for values in dataset.seq_lists:
            offsets.append(total)
            total += len(values)
        require(total == frame.height, "Metadata-to-loader mapping differs")
        del frame, checked
        config = {key: value for key, value in definition["model"].items()
                  if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
        model, _ = factory.build_count_aware_model(job["arm"], **config, train_log_mean=stats["train_log_mean"],
                                                  train_log_std=stats["train_log_std"], max_seq_len=definition["loader"]["max_seq_len"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        active = sum(p.numel() for p in model.parameters() if p.requires_grad)
        registered = sum(p.numel() for p in model.parameters())
        model.to("cuda:0").eval().requires_grad_(False)
        before = runner.canonical_state_dict_sha256(model.state_dict())
        require(before == spec["state_sha"] and not model.training and not any(p.requires_grad for p in model.parameters()), "Evaluation model state differs")
        del payload
        metrics = RawMetrics(definition["quantity_boundaries_all_train_rows"], np)
        writer = PredictionWriter(output / "predictions", pl)
        target_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
        time_digest = hashlib.sha256(b"titantpp_components_recorded_target_time_v1\0" + split.encode() + b"\0")
        ordered_metadata_digest = hashlib.sha256()
        processed, log_sum, inference_seconds, qualification = 0, 0., 0., {}
        phase = "inference"
        with torch.inference_mode():
            for batch_index, (_, dts, mask, part_indices, quantities) in enumerate(loader):
                budget()
                dts, mask, quantities = (v.to("cuda:0") for v in (dts, mask, quantities))
                torch.cuda.synchronize()
                batch_started = time.monotonic()
                value = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
                torch.cuda.synchronize()
                inference_seconds += time.monotonic() - batch_started
                if batch_index == 0:
                    qualification = qualify(torch, model, core.target_outputs, dts, mask, quantities, value,
                                            definition["model"]["time_observation_contract"]["top_code"])
                for key in ("true_qty", "pred_qty", "time_loss", "log_qty_loss"):
                    require(bool(torch.isfinite(value[key]).all()), "Nonfinite diagnostic output: " + key)
                history_mask = mask.cpu().clone()
                history_mask[:, -1] = False
                history_length = history_mask.sum(dim=1)
                quantity_cpu = quantities.cpu().double()
                last = quantity_cpu[:, -2].numpy()
                mean = ((quantity_cpu * history_mask).sum(dim=1) / history_length).numpy()
                q, p, temporal, log_error = (value[key].cpu().numpy().astype(np.float64) for key in ("true_qty", "pred_qty", "time_loss", "log_qty_loss"))
                require(np.array_equal(q, target_q[processed:processed + len(q)]), "Target quantity order differs")
                metrics.add(q, p, temporal)
                log_sum += float(log_error.sum())
                gaps, lengths = dts[:, -1].cpu().tolist(), value["history_length"].cpu().tolist()
                time_digest.update(np.asarray(gaps, dtype="<f8").tobytes())
                rows = []
                for offset in range(len(q)):
                    index = processed + offset
                    pi, end = dataset.index[index]
                    require(int(part_indices[offset]) == pi and int(history_length[offset]) == lengths[offset] >= 1,
                            "History or target order differs")
                    raw = metadata.row(offsets[pi] + end + 1, named=True)
                    entity, sequence = str(dataset.parts[pi]), int(dataset.seq_lists[pi][end + 1])
                    require(str(raw["oper_part_no"]) == entity and int(raw["seq"]) == sequence and raw["chronological_split"] == split,
                            "Target metadata differs")
                    date = str(raw["demand_dt"])
                    ordered_metadata_digest.update((json.dumps([entity, sequence, date], separators=(",", ":")) + "\n").encode())
                    target_id = hashlib.sha256(json_bytes([job["dataset"], spec["data"]["sha256"], entity, sequence, split, date])).hexdigest()
                    truth = dict(target_id=target_id, entity_id=entity, seq=sequence, split=split, recorded_date=date,
                                 raw_quantity=float(q[offset]), recorded_gap=float(gaps[offset]), history_length=int(lengths[offset]),
                                 last_quantity=float(last[offset]), mean_quantity=float(mean[offset]))
                    target_digest.update((target_id + "\n").encode())
                    truth_digest.update(json_bytes([truth[key] for key in TRUTH_FIELDS]) + b"\n")
                    start = end + 1 - int(lengths[offset])
                    start_date = str(metadata.row(offsets[pi] + start, named=True)["demand_dt"])
                    last_date = str(metadata.row(offsets[pi] + end, named=True)["demand_dt"])
                    history_values = quantity_cpu[offset][history_mask[offset]].numpy()
                    error = float(p[offset] - q[offset])
                    rows.append({**truth, "target_index": index, "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
                                 "predicted_raw_quantity": float(p[offset]), "raw_error": error, "absolute_error": abs(error),
                                 "squared_error": error ** 2, "time_nll": float(temporal[offset]), "log_qty_mse": float(log_error[offset]),
                                 "quantity_bin": int(np.searchsorted(definition["quantity_boundaries_all_train_rows"], q[offset], side="left")),
                                 "history_first_date": start_date, "history_last_date": last_date,
                                 "history_quantity_std": float(history_values.std()), "history_quantity_max": float(history_values.max()),
                                 "selected_epoch": spec["selected_epoch"], "checkpoint_state_sha256": before})
                writer.add(rows)
                processed += len(q)
        aggregate = {"split": split, **metrics.finish(), "fixed_model_log_qty_mse": log_sum / processed}
        population = dict(count=processed, target_identity_sha256=target_digest.hexdigest(), truth_sha256=truth_digest.hexdigest(),
                          time_sha256=time_digest.hexdigest())
        if "target_quantity_sha256" in expected_population:
            require(inherited_identity == expected_population, "Bound Train target identity differs")
        else:
            verify_population(population, expected_population)
        if expected_population.get("ordered_entity_seq_recorded_date_sha256"):
            require(ordered_metadata_digest.hexdigest() == expected_population["ordered_entity_seq_recorded_date_sha256"],
                    "Ordered entity/seq/date identity differs")
        if split != "train":
            reference = references[split]
            if job["family"] == "external" and split == "validation":
                reference = reference["selected"]
            compare_metrics(aggregate, reference.get("metrics", reference))
        manifest = writer.finish()
        require(manifest["prediction_rows"] == processed, "Raw row count differs")
        phase = "final_integrity"
        after = runner.canonical_state_dict_sha256(model.state_dict())
        require(after == before and sha(checkpoint) == spec["checkpoint"]["sha256"]
                and sha(data_path) == spec["data"]["sha256"], "Inference mutated checkpoint/model/input")
        imported = verify_science(spec, imports=True)
        verify_contract(root, contract_sha, job_id, split)
        require(resolve_job(job)["state_sha"] == spec["state_sha"], "Scientific parent changed")
        if split == "test":
            require(verify_validation_gate(root, job, contract_sha) == gate_sha, "Diagnostic Validation gate changed")
        runtime_after = numerical.runtime_identity("cuda:0")
        require(runtime_after == runtime, "Runtime changed during diagnostic inference")
        check_runtime(runtime_after, spec["runtime"])
        budget()
        write_exclusive(output / "predictions_manifest.json", manifest)
        receipt = {"schema": SCHEMA, "status": "complete", "job_id": job_id, "family": job["family"],
                   **{key: job[key] for key in ("dataset", "arm", "seed")}, "split": split,
                   "contract_raw_sha256": contract_sha, "checkpoint_file_sha256": spec["checkpoint"]["sha256"],
                   "state_sha256_before": before, "state_sha256_after": after, "selected_epoch": spec["selected_epoch"],
                   "source_closure": spec["source_closure"], "data_sha256": spec["data"]["sha256"],
                   "metrics": aggregate, "population": population, "original_population_identity": inherited_identity,
                   "truth_digest_fields": TRUTH_FIELDS, "raw_predictions_written": True, "full_population": True,
                   "prediction_rows": processed, "predictions_manifest_sha256": sha(output / "predictions_manifest.json"),
                   "reference_metrics_match": "PASS" if split != "train" else "not_available_new_diagnostic",
                   "validation_gate_sha256": gate_sha,
                   "reused_validation": c.get("reused_validation", {}).get(job_id) if split == "test" else None,
                   "qualification": {**qualification, "parameters_unchanged": True}, "runtime": runtime, "runtime_after": runtime_after,
                   "registered_parameters": registered, "active_parameters_before_freeze": active, "new_training": False,
                   "checkpoint_reselection": False, "automatic_retry": False, "test_independent": False,
                   "imported_frozen_modules": imported, "evaluator_sha256": sha(__file__), "pid": os.getpid(), "ppid": os.getppid(),
                   "argv": sys.argv, "started_unix": started_unix, "completed_unix": time.time(),
                   "elapsed_seconds": time.monotonic() - started, "inference_seconds": inference_seconds,
                   "train_population_definition": "complete eligible next-event targets; the first row of each entity has no history and is excluded"}
        write_exclusive(output / "receipt.json", receipt)
        print(json.dumps({key: receipt[key] for key in ("status", "job_id", "split", "prediction_rows", "elapsed_seconds")}), flush=True)
        return receipt
    except BaseException as error:
        write_exclusive(output / "failure.json", dict(status="failed", phase=phase, job_id=job_id, split=split,
            contract_raw_sha256=contract_sha, error=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            automatic_retry=False, elapsed_seconds=time.monotonic() - started))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--contract-sha", required=True)
    args = parser.parse_args(argv)
    worker(args.root, args.job, args.split, args.contract_sha)


if __name__ == "__main__":
    main()
