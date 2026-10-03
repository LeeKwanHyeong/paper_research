#!/usr/bin/env python3
"""Bounded validation inference for immutable quantity-comparison checkpoints.

No optimizer, checkpoint publication, resume, training, or held-out evaluation.
The implementation lives outside the frozen training-source directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = "quantity_checkpoint_diagnosis_v1"
INSTACART_SCHEMA = "quantity_checkpoint_diagnosis_v2"
CASES = ("B_log_original", "raw_original", "log_softplus", "raw_softplus")
SCOPES = ("primary", "last120")
PACKAGES = {"paper", "models", "simple_lab_test", "data_loader", "utils"}
METRICS = ("raw_quantity_rmse", "quantity_mae", "quantity_train_loss", "legacy_time_loss", "log_quantity_mse")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(c):
    require(c["schema"] in {SCHEMA, INSTACART_SCHEMA} and c["status"] == "approved_existing_checkpoint_inference", "Unapproved diagnosis contract")
    require(c["optimizer_updates"] == 0 and c["held_out_evaluated"] is False and c["seed"] == 42, "Inference-only scope changed")
    require(c["device"] == "cuda:0" and c["threads"] == 4, "Diagnosis runtime scope changed")
    require(c["max_concurrent_gpu_jobs"] == 1, "Only one diagnosis GPU job is permitted")
    for key, maximum in (("max_wall_seconds", 7200), ("max_output_bytes", 64 * 1024**2), ("max_checkpoint_bytes", 64 * 1024**2)):
        require(type(c[key]) is int and 0 < c[key] <= maximum, f"Invalid bounded limit: {key}")
    require(c["runtime_expected"]["device"] == c["device"], "Inference device identity mismatch")
    require(c["replay"] == {"absolute_tolerance": 1e-5, "relative_tolerance": 1e-5,
                            "paired_delta_absolute_tolerance": 0.001, "paired_delta_relative_tolerance": 0.01}, "Replay policy changed")
    require(sha_json(c["source"]["files"]) == c["source"]["sha256"], "Source manifest digest mismatch")
    expected_datasets = {"insta_market_basket"} if c["schema"] == INSTACART_SCHEMA else {"intermittent_frozen_5000", "yellow_trip_hourly"}
    require({d["dataset_id"] for d in c["datasets"]} == expected_datasets
            and len(c["datasets"]) == len(expected_datasets), "Dataset scope differs from diagnosis schema")
    for d in c["datasets"]:
        ids = set()
        data = d["data"]
        require(d["dataset_id"] == data["dataset_id"], "Dataset identity mismatch")
        require(len(d["states"]) == 8 and {(s["scope"], s["case"]) for s in d["states"]}
                == {(scope, case) for scope in SCOPES for case in CASES}, "Exactly eight fixed states per dataset are required")
        require(len(data["quantity_boundaries_all_train_rows"]) == 4, "Four frozen quantity boundaries required")
        for values in (data["quantity_boundaries_all_train_rows"], data["history_boundaries"]):
            require(values and all(math.isfinite(v) for v in values) and all(a < b for a, b in zip(values, values[1:])), "Invalid frozen bins")
        base = d["states"][0]["arm_contract"]
        for s in d["states"]:
            require(s["id"] not in ids and Path(s["id"]).name == s["id"], "Duplicate/unsafe state ID")
            ids.add(s["id"])
            arm = s["arm_contract"]
            require(sha_json(arm) == s["arm_contract_sha256"], "Arm contract digest mismatch")
            require(arm["identity"]["source"]["files"] == c["source"]["files"]
                    and arm["identity"]["data"] == data["inherited_data_identity"], "Arm source/data binding differs")
            require(arm["identity"]["dataset_id"] == d["dataset_id"] and arm["condition"] == {
                "case": s["case"], "mu": data["statistics"]["train_log_mean"], "raw_scale": data["statistics"]["raw_scale"]}, "Arm condition/data mismatch")
            require(arm["epochs"] == 120 and arm["seed"] == c["seed"], "Training budget/seed differs")
            if c["schema"] == INSTACART_SCHEMA:
                require(arm["validation_loader"]["target_splits"] == ["validation"], "Only validation loader targets are admitted")
            for key in ("initial_state_sha256", "model", "validation_loader", "expected_validation_targets", "train_loader", "identity"):
                require(arm[key] == base[key], f"Within-dataset arm identity differs: {key}")
            require(s["selector"] == ("last" if s["scope"] == "last120" else "raw_quantity_rmse"), "Selector scope mismatch")
            require(s["scope"] != "last120" or s["epoch"] == 120, "Last scope must be epoch 120")
            require(1 <= s["epoch"] <= 120 and s["expected"]["epoch"] == s["epoch"], "Expected epoch mismatch")
            require(s["global_step"] == s["expected"]["global_step"] == s["epoch"] * arm["train_loader"]["batches"], "Checkpoint step budget mismatch")
            count = data["inherited_data_identity"]["populations"]["validation"]["target_count"]
            require(s["expected"]["validation_count"] == arm["expected_validation_targets"] == count, "Validation count mismatch")
            require(all(math.isfinite(s["expected"][k]) for k in METRICS), "Nonfinite replay reference")


def verify_checkpoint_payload(payload, state, hash_state):
    require(payload.get("schema_version") == "quantity_comparison_engine_v1", "Checkpoint schema mismatch")
    require(payload.get("contract_sha256") == state["arm_contract_sha256"], "Checkpoint contract mismatch")
    if state["selector"] == "last":
        require(payload.get("contract") == state["arm_contract"] and payload.get("epoch") == state["epoch"], "Last checkpoint scope mismatch")
        require(payload.get("history", [])[-1] == state["expected"], "Last checkpoint history reference differs")
        recorded = payload.get("model_state_sha256")
    else:
        require(payload.get("condition") == state["arm_contract"]["condition"], "Selected condition mismatch")
        require(payload.get("selector") == state["selector"] and payload.get("applicable") is True
                and payload.get("best_epoch") == state["epoch"]
                and payload.get("best_value") == state["expected"]["raw_quantity_rmse"], "Selected checkpoint scope mismatch")
        recorded = payload.get("state_sha256")
    require(payload.get("global_step") == state["global_step"], "Checkpoint global step mismatch")
    require(hash_state(payload["model_state_dict"]) == recorded == state["state_sha256"], "Checkpoint tensor-state mismatch")
    return payload["model_state_dict"]


def load_checkpoint(path, *, last):
    import numpy as np
    import torch
    # Best states require no pickle extensions. Last states contain only these
    # NumPy RNG constructors in addition to the normal weights-only safe set.
    safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype("uint32")), np._core.numeric._frombuffer] if last else []
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


def collect_rows(model, loader, objective, engine, statistics, case, device, limit):
    import numpy as np
    import torch
    require(not model.training, "Evaluation mode is required")
    columns = {}
    with torch.no_grad():
        for batch in loader:
            limit()
            args = engine._batch_tensors(batch, device)
            # The trained case link is functional; model.predict_quantity would
            # silently select B's original link for softplus-link checkpoints.
            outputs = objective.joint_causal_batch_objective(model, *args, statistics=statistics, case=objective.QuantityCase(case))
            for key in ("true_qty", "pred_qty", "quantity_train_loss", "log_qty_loss", "time_loss", "history_length"):
                value = outputs[key]
                require(value.ndim == 1 and value.numel() == len(args[0]) and bool(torch.isfinite(value).all()), f"Invalid inference output: {key}")
                columns.setdefault(key, []).append(value.detach().cpu().numpy())
    require(columns, "No validation rows evaluated")
    rows = {key: np.concatenate(parts) for key, parts in columns.items()}
    ds = loader.dataset
    rows["series_ids"] = np.asarray([str(ds.parts[p]) for p, _ in ds.index])
    rows["target_position"] = np.asarray([i + 1 for _, i in ds.index], dtype=np.int64)
    rows["target_seq"] = np.asarray([int(ds.seq_lists[p][i + 1]) for p, i in ds.index], dtype=np.int64)
    truth = np.asarray([ds.val_lists[p][i + 1] for p, i in ds.index], dtype=np.float32)
    require(len(truth) == len(ds) and np.array_equal(truth, rows["true_qty"]), "Validation target order/count differs")
    require(all(p.grad is None for p in model.parameters()), "Inference created gradients")
    return rows


def metrics(rows):
    import numpy as np
    error = rows["pred_qty"].astype(np.float64) - rows["true_qty"].astype(np.float64)
    return {"raw_quantity_rmse": float(np.sqrt(np.mean(error**2))), "quantity_mae": float(np.mean(abs(error))),
            "quantity_train_loss": float(np.mean(rows["quantity_train_loss"], dtype=np.float64)),
            "legacy_time_loss": float(np.mean(rows["time_loss"], dtype=np.float64)),
            "log_quantity_mse": float(np.mean(rows["log_qty_loss"], dtype=np.float64))}


def verify_replay(actual, expected, policy):
    checks = {key: {"actual": actual[key], "expected": expected[key],
                   "passed": math.isclose(actual[key], expected[key], abs_tol=policy["absolute_tolerance"], rel_tol=policy["relative_tolerance"])} for key in METRICS}
    require(all(item["passed"] for item in checks.values()), f"Checkpoint metric replay failed: {checks}")
    return {"passed": True, "checks": checks}


def verify_pair_replay(actual_b, actual_c, expected_b, expected_c, policy):
    checks = {}
    for key in ("raw_quantity_rmse", "quantity_mae"):
        before, after = expected_c[key] - expected_b[key], actual_c[key] - actual_b[key]
        tolerance = min(policy["paired_delta_absolute_tolerance"], policy["paired_delta_relative_tolerance"] * abs(before))
        same_sign = (before > 0) == (after > 0) and (before < 0) == (after < 0)
        passed = same_sign and abs(after - before) <= tolerance
        checks[key] = {"expected_delta": before, "replay_delta": after, "maximum_distortion": tolerance, "passed": passed}
    require(all(item["passed"] for item in checks.values()), f"Paired effect replay failed: {checks}")
    return {"passed": True, "checks": checks}


def paired_diagnosis(base, candidate, quantity_boundaries, history_boundaries):
    """Additive raw-SSE accounting; each partition uses the same paired rows."""
    import numpy as np
    for key in ("true_qty", "history_length", "series_ids", "target_position", "target_seq"):
        require(np.array_equal(base[key], candidate[key]), f"Paired target identity mismatch: {key}")
    truth = base["true_qty"].astype(np.float64)
    n = len(truth)
    require(n > 0, "Empty diagnostic population")
    errors = [r["pred_qty"].astype(np.float64) - truth for r in (base, candidate)]
    require(all(np.isfinite(error).all() for error in errors), "Nonfinite paired errors")
    delta = errors[1]**2 - errors[0]**2

    def cell(mask):
        count = int(mask.sum())
        result = {"n": count, "fraction": count / n, "series_count": int(np.unique(base["series_ids"][mask]).size)}
        for label, error in zip(("B", "candidate"), errors):
            e = error[mask]
            result[label] = {"rmse": float(np.sqrt(np.mean(e**2))) if count else None,
                             "mae": float(np.mean(abs(e))) if count else None,
                             "bias": float(np.mean(e)) if count else None,
                             "under_proportion": float(np.mean(e < 0)) if count else None,
                             "over_proportion": float(np.mean(e > 0)) if count else None,
                             "exact_tie_proportion": float(np.mean(e == 0)) if count else None,
                             "raw_sse": float(np.sum(e**2)),
                             "raw_mse_contribution": float(np.sum(e**2)) / n}
        result.update(delta_raw_sse=float(delta[mask].sum()), delta_raw_mse_contribution=float(delta[mask].sum()) / n,
                      delta_absolute_error_contribution=float((abs(errors[1][mask]) - abs(errors[0][mask])).sum()) / n)
        return result

    quantity = np.searchsorted(quantity_boundaries, truth, side="left")
    history = np.searchsorted(history_boundaries, base["history_length"], side="left")
    overall = cell(np.ones(n, dtype=bool))
    groups = {"quantity_cells": [cell(quantity == q) | {"bin_index": q} for q in range(len(quantity_boundaries) + 1)],
              "history_cells": [cell(history == h) | {"bin_index": h} for h in range(len(history_boundaries) + 1)],
              "cross_cells": [cell((quantity == q) & (history == h)) | {"quantity_bin_index": q, "history_bin_index": h}
                              for q in range(len(quantity_boundaries) + 1) for h in range(len(history_boundaries) + 1)]}
    audit = {}
    for name, cells in groups.items():
        checks = {"count": sum(c["n"] for c in cells) == n}
        for key in ("delta_raw_sse", "delta_raw_mse_contribution", "delta_absolute_error_contribution"):
            checks[key] = math.isclose(sum(c[key] for c in cells), overall[key], rel_tol=1e-10, abs_tol=1e-8)
        for label in ("B", "candidate"):
            checks[label + "_raw_sse"] = math.isclose(sum(c[label]["raw_sse"] for c in cells), overall[label]["raw_sse"], rel_tol=1e-10, abs_tol=1e-8)
        require(all(checks.values()), f"Partition SSE reconstruction failed: {name}")
        audit[name] = checks
    ids, inverse = np.unique(base["series_ids"], return_inverse=True)
    series_delta = np.bincount(inverse, weights=delta)
    positive = np.maximum(series_delta, 0)
    negative = np.maximum(-series_delta, 0)
    order = np.argsort(-positive, kind="stable")
    def share(fraction):
        count = max(1, math.ceil(len(ids) * fraction))
        return {"series_count": count, "positive_delta_sse_fraction": float(positive[order[:count]].sum() / positive.sum()) if positive.sum() else None}
    concentration = {"definition": "positive part of each series net candidate-minus-B raw SSE",
                     "positive_delta_sse": float(positive.sum()), "negative_delta_sse_magnitude": float(negative.sum()),
                     "top_1_percent": share(.01), "top_5_percent": share(.05),
                     "largest_positive_series": [{"series_id": str(ids[i]), "delta_raw_sse": float(series_delta[i])} for i in order[:10] if positive[i] > 0]}
    return {"n": n, "binning": {"quantity_boundaries": quantity_boundaries, "history_boundaries": history_boundaries,
                                   "boundary_equality": "lower_bin", "scope": "frozen_all_train_rows"},
            "overall": overall, **groups,
            "body": cell(truth <= quantity_boundaries[2]), "tail": cell(truth > quantity_boundaries[3]),
            "body_tail_definition": "body y<=train p95; tail y>train p99; intermediate p95<p<=p99 is excluded from both",
            "series_concentration": concentration, "additivity_audit": {"passed": True, "partitions": audit}}


@contextmanager
def wall_budget(seconds):
    def expired(signum, frame):
        raise TimeoutError("Diagnosis wall-time limit reached; no retry")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def write_json(path, value, maximum):
    path = Path(path)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    used = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    require(used + len(encoded) <= maximum, "Diagnosis output byte budget exceeded")
    with path.open("xb") as handle:
        handle.write(encoded)


def execute(c):
    validate_contract(c)
    require(Path(sys.executable).resolve() == Path(c["python_executable"]).resolve(), "Interpreter differs from inference contract")
    source, output = Path(c["source"]["root"]).resolve(), Path(c["output_dir"]).resolve()
    require(not output.exists() and not output.is_relative_to(source), "Refusing existing output or source write")
    for name, digest in c["source"]["files"].items():
        path = source / name
        require(path.resolve().is_relative_to(source) and sha_file(path) == digest, f"Frozen source changed: {name}")
    require(not any(name.split(".")[0] in PACKAGES for name in sys.modules), "Diagnosis requires a clean process")
    require(all(os.environ.get(k) == v for k, v in c["runtime_expected"]["environment"].items()), "Numerical environment differs")
    # An idle-device check immediately precedes CUDA initialization. An advisory
    # lock excludes another instance of this evaluator on the same output parent.
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = (output.parent / "quantity_checkpoint_diagnosis_gpu0.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    require(not busy, "GPU has an existing compute process; no automatic retry")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from paper.scripts import quantity_comparison_data as data_helper
    from paper.scripts import quantity_objective_comparison as objective
    from paper.scripts import quantity_comparison_runtime as runtime
    from paper.scripts import run_quantity_comparison as frozen
    from paper.scripts import run_time_quantity_diagnostic as loader_builder
    from paper.scripts import time_quantity_diagnostic as engine
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    runtime.configure_runtime(c["device"], c["threads"])
    observed_runtime = runtime.runtime_identity(c["device"])
    require(observed_runtime == c["runtime_expected"], "Observed inference runtime differs")
    loader_builder.validate_import_origins()
    output.mkdir()
    write_json(output / "contract.json", c, c["max_output_bytes"])
    write_json(output / "started.json", {"pid": os.getpid(), "contract_sha256": sha_json(c), "runtime": observed_runtime}, c["max_output_bytes"])
    started = time.monotonic()
    def limit():
        require(time.monotonic() - started < c["max_wall_seconds"], "Diagnosis wall-time budget exceeded")
    results = []
    try:
        with wall_budget(c["max_wall_seconds"]):
            for d in c["datasets"]:
                limit()
                data = d["data"]
                frame, metadata = data_helper.prepare_quantity_comparison_data(data)
                metadata = frozen.bind_frozen_statistics(data, metadata)
                require(metadata["held_out_materialized"] is False, "Held-out data materialized")
                statistics = objective.QuantityStatistics(mu=data["statistics"]["train_log_mean"], raw_scale=data["statistics"]["raw_scale"])
                states = {(s["scope"], s["case"]): s for s in d["states"]}
                for scope in SCOPES:
                    baseline_rows = baseline_metrics = baseline_state = None
                    for case in CASES:
                        state = states[scope, case]
                        limit()
                        path = Path(state["path"])
                        require(0 < path.stat().st_size <= c["max_checkpoint_bytes"], "Checkpoint byte limit exceeded")
                        require(sha_file(path) == state["checkpoint_sha256"], "Checkpoint file hash differs")
                        payload = load_checkpoint(path, last=state["selector"] == "last")
                        weights = verify_checkpoint_payload(payload, state, canonical_state_dict_sha256)
                        model, train, validation, _ = loader_builder.build_arm_inputs({**data, "seed": c["seed"]}, frame, metadata)
                        arm = state["arm_contract"]
                        require(canonical_state_dict_sha256(model.state_dict()) == arm["initial_state_sha256"], "Recreated initial model differs")
                        require(engine._model_spec(model) == arm["model"], "Model settings/source mismatch")
                        require(engine._loader_spec(validation, training=False) == arm["validation_loader"], "Validation loader identity mismatch")
                        del train
                        model.load_state_dict(weights, strict=True)
                        model.to(c["device"]).eval()
                        require(canonical_state_dict_sha256(model.state_dict()) == state["state_sha256"], "Loaded weights differ")
                        rows = collect_rows(model, validation, objective, engine, statistics, case, c["device"], limit)
                        actual = metrics(rows)
                        replay = verify_replay(actual, state["expected"], c["replay"])
                        require(canonical_state_dict_sha256(model.state_dict()) == state["state_sha256"], "Inference mutated model state")
                        require(sha_file(path) == state["checkpoint_sha256"], "Checkpoint changed during inference")
                        report = {"dataset_id": d["dataset_id"], "state_id": state["id"], "case": case, "scope": scope,
                                  "selector": state["selector"], "epoch": state["epoch"], "global_step": state["global_step"],
                                  "checkpoint_sha256": state["checkpoint_sha256"], "state_sha256": state["state_sha256"],
                                  "arm_contract_sha256": state["arm_contract_sha256"], "n": len(rows["true_qty"]),
                                  "metrics": actual, "replay": replay, "evaluation_scope": "validation_only",
                                  "optimizer_updates": 0, "unchanged_model_state": True}
                        if case == CASES[0]:
                            baseline_rows, baseline_metrics, baseline_state = rows, actual, state
                        else:
                            report["paired_replay"] = verify_pair_replay(baseline_metrics, actual, baseline_state["expected"], state["expected"], c["replay"])
                            report["versus_B"] = paired_diagnosis(baseline_rows, rows, data["quantity_boundaries_all_train_rows"], data["history_boundaries"])
                        write_json(output / (d["dataset_id"] + "__" + state["id"] + ".json"), report, c["max_output_bytes"])
                        results.append({k: report[k] for k in ("dataset_id", "state_id", "case", "scope", "epoch", "global_step", "n", "metrics")})
                        print(json.dumps({"completed_state": state["id"], "metrics": actual, "elapsed_seconds": time.monotonic() - started}), flush=True)
                        del model, validation, payload, weights, rows
                        gc.collect()
                    del baseline_rows
                del frame, metadata
            loader_builder.validate_import_origins()
            for name, digest in c["source"]["files"].items():
                require(sha_file(source / name) == digest, "Frozen source changed during diagnosis")
        runtime_note = ("Intermittent inference migrated to 5080; aggregate replay is tolerance-based, not a bitwise-equivalence claim"
                        if c["schema"] == SCHEMA else
                        "Inference runtime is pinned separately from training runtime; aggregate replay is tolerance-based, not a bitwise-equivalence claim")
        final = {"schema": c["schema"], "status": "complete", "contract_sha256": sha_json(c), "states_completed": len(results),
                 "evaluation_scope": "validation_only", "held_out_evaluated": False, "optimizer_updates": 0,
                 "inference_runtime": observed_runtime, "replay_tolerance": c["replay"], "elapsed_seconds": time.monotonic() - started,
                 "results": results, "interpretation_limits": ["single-seed validation", "primary selectors can be different epochs", "time score is legacy clamped loss", runtime_note]}
        write_json(output / "summary.json", final, c["max_output_bytes"])
        return final
    except Exception as error:
        failure = {"schema": c["schema"], "status": "failed", "contract_sha256": sha_json(c), "states_completed": len(results),
                   "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.monotonic() - started,
                   "automatic_retry": False, "optimizer_updates": 0, "held_out_evaluated": False}
        write_json(output / "failure.json", failure, c["max_output_bytes"])
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    validate_contract(contract)
    if args.execute:
        execute(contract)
    else:
        print(json.dumps({"valid": True, "contract_sha256": sha_json(contract), "checkpoint_loaded": False}))


if __name__ == "__main__":
    main()
