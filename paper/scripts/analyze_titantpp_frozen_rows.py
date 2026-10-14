"""Analyze immutable diagnostic predictions; never import or run a scientific model.

All per-example joins are exact on the ten frozen truth fields.  Grouping and
decompositions describe observed errors and do not identify a causal mechanism.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import polars as pl

TRUTH_FIELDS = ("target_id", "entity_id", "seq", "split", "recorded_date",
                "raw_quantity", "recorded_gap", "history_length",
                "last_quantity", "mean_quantity")
SPLITS = ("train", "validation", "test")
CNN = "titantpp_cnn_gru54_cnn_only"
FULL = "titantpp_cnn_gru54"
EXTERNALS = ("rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head")
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
DISPLAY = {"yellow_trip_hourly": "Taxi", "raf_spare_parts": "RAF",
           "intermittent_frozen_5000": "Intermittent", CNN: "CNN", FULL: "CNN+GRU",
           "rmtpp": "RMTPP", "thp": "THP", "nhp": "NHP", "sahp": "SAHP",
           "s2p2_matched_head": "S2P2", "attnhp_matched_head": "AttNHP"}
METRICS = ("qty_rmse", "qty_mae", "qty_bias", "qty_mse", "fixed_model_log_qty_mse")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def confined(root, path):
    result = (Path(root) / path).resolve()
    require(result.is_relative_to(Path(root).resolve()), "Artifact path escapes its declared root")
    return result


def near(a, b, *, rel=1e-9, absolute=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=absolute)


def safe_share(part, whole):
    return None if abs(float(whole)) <= 1e-10 else float(part) / float(whole)


def summarize(values):
    values = [float(v) for v in values if v is not None]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
            "ddof": 1}


class Evidence:
    def __init__(self):
        self.files = {}

    def verify(self, path, expected):
        path = Path(path).resolve()
        require(isinstance(expected, str) and len(expected) == 64, "Missing byte SHA")
        if str(path) in self.files:
            require(self.files[str(path)] == expected, "Conflicting artifact byte SHA")
            return path
        actual = sha(path)
        require(actual == expected, "Artifact byte SHA differs: " + str(path))
        self.files[str(path)] = actual
        return path

    def read(self, path, expected):
        return json.loads(self.verify(path, expected).read_text())


def fixed_definition(job, evidence):
    if job["family"] == "external":
        return job["definition"]
    root = Path(job["local_parent_root"])
    contract = evidence.read(root / "contract.json", job["parent_contract_raw_sha256"])
    return contract["datasets"][job["dataset"]]["definition"]


def expected_selection(job, evidence):
    if job["family"] == "external":
        row = job["legacy_registry_row"]
        evidence.verify(confined(job["local_parent_root"], row["checkpoint_path"]),
                        row["checkpoint_file_sha256"])
        return {"epoch": row["selected_epoch"], "state": row["state_tensor_sha256"],
                "file": row["checkpoint_file_sha256"], "source": row["source_closure_sha256"]}
    root = Path(job["local_parent_root"])
    contract = evidence.read(root / "contract.json", job["parent_contract_raw_sha256"])
    matching = [row for row in contract["jobs"] if row["id"] == job["parent_job_id"]]
    require(len(matching) == 1, "Scientific parent omits the frozen selected model")
    row = matching[0]
    evidence.verify(confined(root, row["checkpoint"]["relative_path"]), row["checkpoint"]["sha256"])
    return {"epoch": row["selected_epoch"], "state": row["checkpoint_state_sha256"],
            "file": row["checkpoint"]["sha256"], "source": row["frozen_source_closure"]}


def expected_data_sha(job, evidence):
    if job["family"] == "external":
        return job["data"]["sha256"]
    root = Path(job["local_parent_root"])
    parent = evidence.read(root / "contract.json", job["parent_contract_raw_sha256"])
    return parent["datasets"][job["dataset"]]["data_sha256"]


def receipt_contract_bridge(job, entry, current_sha, current_contract):
    """Only an explicitly SHA-bound component Validation cache may bridge attempts."""
    mapping = current_contract.get("reused_validation", {})
    bridge = mapping.get(job["id"])
    if entry["split"] != "validation" or bridge is None:
        return current_sha, False
    require(job["family"] == "component" and job["host"] == "5080",
            "Validation cache bridge is outside the authorized component/native host")
    require(entry["receipt"]["sha256"] == bridge["receipt_sha256"]
            and entry["worker_exit"]["sha256"] == bridge["worker_exit_sha256"],
            "Validation cache is not the exact current-contract approved receipt/worker exit")
    old_sha = bridge["contract_sha256"]
    require(isinstance(old_sha, str) and len(old_sha) == 64 and old_sha != current_sha,
            "Validation cache bridge does not name a distinct preserved old contract")
    return old_sha, True


def verify_native_runtime_receipt(actual, expected):
    for key, expected_key in (("python", "python_version"), ("torch", "torch"), ("cuda", "cuda"),
                              ("cudnn", "cudnn"), ("numpy", "numpy"), ("polars", "polars"),
                              ("threads", "torch_num_threads"), ("interop_threads", "interop_threads")):
        require(actual.get(key) == expected[expected_key], "Native runtime binding differs: " + key)
    require(actual.get("gpu", {}).get("uuid") == expected["gpu_uuid"], "Native GPU UUID differs")
    if "resolved_python_executable" in actual:
        require(actual["resolved_python_executable"] == expected.get("resolved_python_executable", expected["python"]),
                "Directly recorded native resolved Python identity differs")
    require(actual.get("default_dtype") == "torch.float32" and actual.get("deterministic_algorithms") is True
            and actual.get("cuda_matmul_tf32") is False and actual.get("cudnn_tf32") is False,
            "Native numerical policy differs")


def verify_executable_proof(expected, worker_exit, evaluator_source):
    """Attest the sealed live resolution guard; do not invent an unstored path."""
    argv = worker_exit.get("argv", [])
    require(isinstance(argv, list) and len(argv) >= 2 and argv[0] == expected["python"],
            "Actual worker Python argv differs from the fixed native contract")
    tree = ast.parse(evaluator_source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require("check_runtime" in functions and "worker" in functions,
            "Sealed evaluator omits runtime resolution checks")
    comparison = ast.parse("Path(sys.executable).resolve() == Path(expected['python']).resolve()", mode="eval").body
    guards = [node for node in ast.walk(functions["check_runtime"])
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "require" and node.args
              and ast.dump(node.args[0], include_attributes=False) == ast.dump(comparison, include_attributes=False)]
    require(guards, "Sealed evaluator omits the live executable resolution guard")
    checked = {node.args[0].id for node in ast.walk(functions["worker"])
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "check_runtime" and node.args and isinstance(node.args[0], ast.Name)}
    require({"runtime", "runtime_after"} <= checked,
            "Sealed evaluator does not check native runtime before and after inference")
    return {"kind": "sealed_evaluator_live_executable_resolution_guard",
            "worker_argv_python": argv[0], "native_contract_python": expected["python"],
            "expected_resolved_path_in_contract": expected.get("resolved_python_executable"),
            "direct_resolved_path_stored": False,
            "before_and_after_guard_in_successful_worker": True}


def evaluator_sha_for_receipt(recorded_sha, current_sha, *, authorized_cache):
    require(isinstance(recorded_sha, str) and len(recorded_sha) == 64
            and all(c in "0123456789abcdef" for c in recorded_sha), "Invalid evaluator source SHA")
    if not authorized_cache:
        require(recorded_sha == current_sha, "Fresh worker used a different diagnostic evaluator")
    # Cache authorization was already established by the exact receipt byte SHA,
    # which binds this recorded historical evaluator SHA along with all fields.
    return recorded_sha


def canonical_frame(frame, *, split, bounds):
    """Normalize types and recompute raw errors, without replacing stored truth."""
    required = set(TRUTH_FIELDS) | {"predicted_raw_quantity"}
    require(required <= set(frame.columns), "Missing raw diagnostic fields")
    require(frame.height > 0 and frame["target_id"].n_unique() == frame.height,
            "Duplicate or empty raw target identity")
    require(not frame.select(pl.any_horizontal([pl.col(c).is_null() for c in required]).any()).item(),
            "Null raw prediction or truth field")
    require(frame["split"].unique().to_list() == [split], "Wrong row split")
    numeric = ("raw_quantity", "recorded_gap", "last_quantity", "mean_quantity",
               "predicted_raw_quantity")
    frame = frame.with_columns(
        *[pl.col(c).cast(pl.Float64) for c in numeric],
        pl.col("seq").cast(pl.Int64), pl.col("history_length").cast(pl.Int64),
        *[pl.col(c).cast(pl.String) for c in ("target_id", "entity_id", "split", "recorded_date")],
    )
    require(frame.select(pl.all_horizontal([pl.col(c).is_finite() for c in numeric]).all()).item(),
            "Nonfinite raw predictions or truth")
    require(frame.select(((pl.col("history_length") >= 1) & (pl.col("raw_quantity") >= 0)).all()).item(),
            "Invalid observed history or raw quantity")
    error = frame["predicted_raw_quantity"].to_numpy() - frame["raw_quantity"].to_numpy()
    for name, computed in (("raw_error", error), ("squared_error", error ** 2),
                           ("absolute_error", np.abs(error))):
        if name in frame.columns:
            require(np.allclose(frame[name].to_numpy(), computed, rtol=1e-10, atol=1e-10),
                    "Saved raw error differs from prediction/truth: " + name)
    bin_index = np.searchsorted(np.asarray(bounds), frame["raw_quantity"].to_numpy(), side="left")
    if "quantity_bin" in frame.columns:
        require(np.array_equal(frame["quantity_bin"].to_numpy(), bin_index), "Frozen quantity bin differs")
    frame = frame.with_columns(pl.Series("raw_error", error), pl.Series("squared_error", error ** 2),
                               pl.Series("absolute_error", np.abs(error)),
                               pl.Series("quantity_bin", bin_index))
    if "log_qty_mse" in frame.columns:
        require(frame.select((pl.col("log_qty_mse").is_finite() & (pl.col("log_qty_mse") >= 0)).all()).item(),
                "Invalid saved fixed-model log loss")
    return frame


def align_pair(left, right):
    """Exact identity join; a sorted order alone is never treated as identity."""
    require(left.height == right.height and left.height > 0, "Paired target counts differ")
    for frame in (left, right):
        require(frame["target_id"].n_unique() == frame.height, "Duplicate paired target ID")
    suffixes = {c: c + "_right" for c in right.columns if c != "target_id"}
    joined = left.join(right.rename(suffixes), on="target_id", how="inner", validate="1:1")
    require(joined.height == left.height, "Paired targets do not coincide")
    for column in TRUTH_FIELDS[1:]:
        require(joined.select((pl.col(column) == pl.col(column + "_right")).all()).item(),
                "Paired frozen truth differs: " + column)
    for column in ("history_first_date", "history_last_date", "history_quantity_std", "history_quantity_max"):
        if column in left.columns and column in right.columns:
            require(joined.select((pl.col(column) == pl.col(column + "_right")).all()).item(),
                    "Paired observed-history context differs: " + column)
    return joined.with_columns(
        (pl.col("squared_error") - pl.col("squared_error_right")).alias("delta_sq"),
        (pl.col("absolute_error") - pl.col("absolute_error_right")).alias("delta_abs"),
        (pl.col("raw_error") - pl.col("raw_error_right")).alias("delta_error"),
    )


def model_metrics(frame):
    n = frame.height
    sse, absolute, signed = (float(frame[c].sum()) for c in
                             ("squared_error", "absolute_error", "raw_error"))
    result = {"count": n, "qty_sse": sse, "qty_abs_error_sum": absolute,
              "qty_signed_error_sum": signed, "qty_mse": sse / n,
              "qty_rmse": math.sqrt(sse / n), "qty_mae": absolute / n,
              "qty_bias": signed / n,
              "fixed_model_log_qty_mse": float(frame["log_qty_mse"].mean()) if "log_qty_mse" in frame.columns else None,
              "mean_truth": float(frame["raw_quantity"].mean()),
              "mean_truth_squared": float((frame["raw_quantity"] ** 2).mean()),
              "entity_count": frame["entity_id"].n_unique(),
              "recorded_date_min": frame["recorded_date"].min(),
              "recorded_date_max": frame["recorded_date"].max()}
    result["error_sse_over_truth_sumsq"] = safe_share(sse, result["mean_truth_squared"] * n)
    return result


def paired_group(frame, key, *, overall_count=None):
    """Every disjoint partition must conserve target count and raw error sums."""
    overall_count = frame.height if overall_count is None else overall_count
    keys = [key] if isinstance(key, str) else list(key)
    grouped = frame.group_by(keys).agg(
        pl.len().alias("count"),
        pl.col("squared_error").sum().alias("left_sse"),
        pl.col("squared_error_right").sum().alias("right_sse"),
        pl.col("absolute_error").sum().alias("left_abs_sum"),
        pl.col("absolute_error_right").sum().alias("right_abs_sum"),
        pl.col("raw_error").sum().alias("left_signed_sum"),
        pl.col("raw_error_right").sum().alias("right_signed_sum"),
        pl.col("delta_sq").sum().alias("delta_sse"),
        pl.col("delta_abs").sum().alias("delta_abs_sum"),
        pl.col("raw_quantity").sum().alias("truth_sum"),
        pl.col("last_quantity").sum().alias("last_quantity_sum"),
        pl.col("mean_quantity").sum().alias("mean_quantity_sum"),
    ).sort(keys)
    require(int(grouped["count"].sum()) == frame.height, "Group count conservation failed")
    for original, aggregate in (("squared_error", "left_sse"),
                                ("squared_error_right", "right_sse"),
                                ("absolute_error", "left_abs_sum"),
                                ("absolute_error_right", "right_abs_sum"),
                                ("raw_error", "left_signed_sum"),
                                ("raw_error_right", "right_signed_sum"),
                                ("delta_sq", "delta_sse"), ("delta_abs", "delta_abs_sum")):
        require(near(grouped[aggregate].sum(), frame[original].sum(), absolute=1e-6),
                "Grouped error conservation failed: " + aggregate)
    return grouped.with_columns(
        (pl.col("count") / overall_count).alias("population_share"),
        (pl.col("left_sse") / pl.col("count")).alias("left_mse"),
        (pl.col("right_sse") / pl.col("count")).alias("right_mse"),
        (pl.col("delta_sse") / pl.col("count")).alias("conditional_mse_gap"),
        (pl.col("delta_sse") / overall_count).alias("population_mse_gap_contribution"),
        (pl.col("delta_abs_sum") / overall_count).alias("population_mae_gap_contribution"),
        (pl.col("left_abs_sum") / pl.col("count")).alias("left_mae"),
        (pl.col("right_abs_sum") / pl.col("count")).alias("right_mae"),
        (pl.col("left_signed_sum") / pl.col("count")).alias("left_bias"),
        (pl.col("right_signed_sum") / pl.col("count")).alias("right_bias"),
        (pl.col("truth_sum") / pl.col("count")).alias("mean_truth"),
        (pl.col("last_quantity_sum") / pl.col("count")).alias("mean_last_quantity"),
        (pl.col("mean_quantity_sum") / pl.col("count")).alias("mean_history_quantity"),
    )


def add_groups(frame):
    history = (pl.when(pl.col("history_length") <= 64).then(pl.lit("1..64"))
               .when(pl.col("history_length") <= 128).then(pl.lit("65..128"))
               .otherwise(pl.lit("129+")))
    def direction(column):
        return (pl.when(pl.col("raw_quantity") < pl.col(column)).then(pl.lit("target_below_history"))
                .when(pl.col("raw_quantity") == pl.col(column)).then(pl.lit("target_equal_history"))
                .otherwise(pl.lit("target_above_history")))
    return frame.with_columns(history.alias("history_bin"),
                              pl.col("recorded_date").str.slice(0, 6).alias("calendar_month"),
                              direction("last_quantity").alias("last_quantity_direction"),
                              direction("mean_quantity").alias("mean_quantity_direction"))


def entity_summary(groups):
    gap = groups["conditional_mse_gap"].to_numpy()
    return {"entity_count": groups.height,
            "entities_left_mse_lower": int(np.sum(gap < 0)),
            "entities_right_mse_lower": int(np.sum(gap > 0)),
            "entities_tied": int(np.sum(gap == 0)),
            "equal_entity_mean_mse_gap": float(np.mean(gap)),
            "equal_entity_median_mse_gap": float(np.median(gap)),
            "descriptive_largest_positive_contributions": groups.sort("delta_sse", descending=True).head(10).to_dicts(),
            "descriptive_largest_negative_contributions": groups.sort("delta_sse").head(10).to_dicts(),
            "posthoc_ranking_warning": "Top entities are descriptive attribution, not a preselected subgroup or generalization claim."}


def analyze_pair(left, right):
    paired = add_groups(align_pair(left, right))
    grouped = {name: paired_group(paired, column).to_dicts() for name, column in (
        ("quantity", "quantity_bin"), ("history", "history_bin"),
        ("recorded_gap_exact", "recorded_gap"), ("calendar_month", "calendar_month"),
        ("last_quantity_direction", "last_quantity_direction"),
        ("mean_quantity_direction", "mean_quantity_direction"),
        ("quantity_by_history", ["quantity_bin", "history_bin"]),
        ("quantity_by_last_direction", ["quantity_bin", "last_quantity_direction"]),
        ("quantity_by_mean_direction", ["quantity_bin", "mean_quantity_direction"]),
        ("quantity_by_calendar_month", ["quantity_bin", "calendar_month"]),
        ("quantity_by_recorded_gap", ["quantity_bin", "recorded_gap"]),
    )}
    entities = paired_group(paired, "entity_id")
    lm, rm = model_metrics(left), model_metrics(right)
    result = {"count": paired.height, "identity_join": "exact ten-field frozen truth PASS",
              "mse_gap": lm["qty_mse"] - rm["qty_mse"],
              "rmse_gap": lm["qty_rmse"] - rm["qty_rmse"],
              "mae_gap": lm["qty_mae"] - rm["qty_mae"],
              "bias_gap": lm["qty_bias"] - rm["qty_bias"],
              "log_mse_gap": (lm["fixed_model_log_qty_mse"] - rm["fixed_model_log_qty_mse"]
                              if all(m["fixed_model_log_qty_mse"] is not None for m in (lm, rm)) else None),
              "groups": grouped, "entities": entity_summary(entities)}
    return result, entities


def shared_entity_decomposition(validation, test):
    """Exact membership/weights/conditional-gap decomposition, not causal."""
    v = {row["entity_id"]: row for row in validation.to_dicts()}
    t = {row["entity_id"]: row for row in test.to_dicts()}
    common = sorted(set(v) & set(t))
    require(common, "No common entity; shared-entity decomposition is undefined")
    n_v, n_t = sum(r["count"] for r in v.values()), sum(r["count"] for r in t.values())
    nv = sum(v[e]["count"] for e in common)
    nt = sum(t[e]["count"] for e in common)
    global_v = sum(r["delta_sse"] for r in v.values()) / n_v
    global_t = sum(r["delta_sse"] for r in t.values()) / n_t
    shared_v = sum(v[e]["delta_sse"] for e in common) / nv
    shared_t = sum(t[e]["delta_sse"] for e in common) / nt
    mix = within = 0.
    change_rows = []
    for entity in common:
        pv, pt = v[entity]["count"] / nv, t[entity]["count"] / nt
        dv, dt = v[entity]["conditional_mse_gap"], t[entity]["conditional_mse_gap"]
        m = (pt - pv) * (dt + dv) / 2
        w = (pt + pv) / 2 * (dt - dv)
        mix += m
        within += w
        change_rows.append({"entity_id": entity, "validation_count": v[entity]["count"],
                            "test_count": t[entity]["count"], "validation_mse_gap": dv,
                            "test_mse_gap": dt, "weight_mix_contribution": m,
                            "within_entity_contribution": w})
    require(near(mix + within, shared_t - shared_v, absolute=1e-6), "Shared-entity conservation failed")
    test_membership, validation_membership = global_t - shared_t, shared_v - global_v
    require(near(test_membership + mix + within + validation_membership,
                 global_t - global_v, absolute=1e-6), "Global entity decomposition failed")
    return {"validation_entity_count": len(v), "test_entity_count": len(t),
            "shared_entity_count": len(common), "validation_only_entities": len(set(v) - set(t)),
            "test_only_entities": len(set(t) - set(v)),
            "validation_shared_rows": nv, "test_shared_rows": nt,
            "validation_shared_fraction": nv / n_v, "test_shared_fraction": nt / n_t,
            "validation_global_mse_gap": global_v, "test_global_mse_gap": global_t,
            "validation_shared_mse_gap": shared_v, "test_shared_mse_gap": shared_t,
            "global_mse_gap_change": global_t - global_v,
            "test_membership_term": test_membership,
            "validation_membership_term": validation_membership,
            "shared_entity_weight_mix_term": mix, "shared_entity_within_term": within,
            "shared_entity_mean_gap_change": statistics.mean(t[e]["conditional_mse_gap"] - v[e]["conditional_mse_gap"] for e in common),
            "shared_entity_mean_gap_validation": statistics.mean(v[e]["conditional_mse_gap"] for e in common),
            "shared_entity_mean_gap_test": statistics.mean(t[e]["conditional_mse_gap"] for e in common),
            "descriptive_largest_within_entity_worsening": sorted(change_rows, key=lambda r: r["within_entity_contribution"], reverse=True)[:10],
            "decomposition_interpretation": "Exact descriptive terms. Within-entity terms include different targets, histories, quantity severity and time periods; they do not identify a causal effect."}


def truth_digests(frame):
    target, truth = hashlib.sha256(), hashlib.sha256()
    for row in frame.select(TRUTH_FIELDS).iter_rows():
        target.update((row[0] + "\n").encode())
        truth.update(json_bytes(list(row)) + b"\n")
    return target.hexdigest(), truth.hexdigest()


def verify_metric_receipt(metrics, actual):
    for name in ("qty_rmse", "qty_mae", "qty_sse"):
        if name in metrics:
            require(near(actual[name], metrics[name], rel=1e-8, absolute=1e-8),
                    "Stored aggregate differs from actual rows: " + name)
    require(metrics["count"] == actual["count"], "Aggregate count differs")
    if "fixed_model_log_qty_mse" in metrics:
        require(near(metrics["fixed_model_log_qty_mse"], actual["fixed_model_log_qty_mse"], rel=1e-8, absolute=1e-8),
                "Fixed-model log aggregate differs")


def verify_exit_payload(value, expected_contract, *, job_id=None, split=None, pipeline=False):
    require(value.get("exit_code") == 0 and value.get("actual_child_exit_observed") is True,
            "Actual diagnostic OS exit0 is missing")
    require(value.get("contract_sha256") == expected_contract,
            "Actual exit is not bound to the current native contract")
    if job_id is not None:
        require(value.get("job_id") == job_id and value.get("split") == split,
                "Worker exit identity differs")
    if pipeline:
        require(value.get("contract_unchanged") is True and value.get("supervisor_exit_code") == 0,
                "Pipeline supervisor did not verify unchanged contract and its own exit0")


def verify_pipeline_hosts(refs):
    require(len(refs) == 2 and {ref["host"] for ref in refs} == {"5080", "5090"},
            "Pipeline exits must cover both distinct native hosts")


def load_new_rows(bundle, entry, job, bounds, evidence, contracts, contract_payloads):
    split = entry["split"]
    receipt_ref, exit_ref = entry["receipt"], entry["worker_exit"]
    receipt_path = confined(bundle, receipt_ref["path"])
    receipt = evidence.read(receipt_path, receipt_ref["sha256"])
    worker_exit = evidence.read(confined(bundle, exit_ref["path"]), exit_ref["sha256"])
    expected_contract, cached = receipt_contract_bridge(job, entry, contracts[job["host"]],
                                                         contract_payloads[job["host"]])
    verify_exit_payload(worker_exit, expected_contract, job_id=job["id"], split=split)
    require(receipt.get("status") == "complete" and receipt.get("job_id") == job["id"]
            and receipt.get("split") == split and receipt.get("full_population") is True
            and receipt.get("raw_predictions_written") is True, "Incomplete raw predictions")
    require(receipt.get("contract_raw_sha256") == expected_contract, "Worker contract bytes differ")
    require(receipt.get("state_sha256_before") == receipt.get("state_sha256_after")
            and receipt.get("qualification", {}).get("parameters_unchanged") is True,
            "Fixed-model parameters changed or verification is missing")
    require(all(receipt.get(key) is False for key in ("new_training", "checkpoint_reselection", "automatic_retry")),
            "Diagnostic scope includes forbidden training/reselection/retry")
    require(receipt.get("runtime") == receipt.get("runtime_after"), "Runtime changed during inference")
    verify_native_runtime_receipt(receipt["runtime"], job["runtime"])
    evaluator_sha = evaluator_sha_for_receipt(receipt.get("evaluator_sha256"),
                                              contract_payloads[job["host"]]["evaluator_sha256"],
                                              authorized_cache=cached)
    evaluator_path = confined(bundle, "control/evaluator_sources/" + evaluator_sha + ".py")
    evidence.verify(evaluator_path, evaluator_sha)
    executable_proof = verify_executable_proof(job["runtime"], worker_exit, evaluator_path.read_text())
    executable_proof["evaluator_source_sha256"] = evaluator_sha
    executable_proof["historical_source_allowed_by_exact_validation_cache_SHA"] = cached
    executable_proof["direct_resolved_path_stored"] = "resolved_python_executable" in receipt["runtime"]
    require(receipt["data_sha256"] == expected_data_sha(job, evidence), "Frozen input data SHA differs")
    selection = expected_selection(job, evidence)
    require(receipt.get("selected_epoch") == selection["epoch"]
            and receipt.get("state_sha256_before") == selection["state"]
            and receipt.get("checkpoint_file_sha256") == selection["file"]
            and receipt.get("source_closure") == selection["source"],
            "Diagnostic receipt does not use the original frozen selected checkpoint/source")
    require(all(receipt["qualification"].values()), "Synthetic qualification did not pass")
    if split != "train":
        require(receipt.get("reference_metrics_match") == "PASS", "Native Validation/Test replay did not pass")
    manifest_path = receipt_path.parent / "predictions_manifest.json"
    manifest = evidence.read(manifest_path, receipt["predictions_manifest_sha256"])
    frames = []
    for part in manifest["parts"]:
        path = confined(receipt_path.parent / "predictions", part["path"])
        evidence.verify(path, part["sha256"])
        require(path.stat().st_size == part["bytes"], "Prediction part size differs")
        frame = pl.read_parquet(path)
        require(frame.height == part["rows"], "Prediction part count differs")
        frames.append(frame)
    frame = canonical_frame(pl.concat(frames), split=split, bounds=bounds)
    require(frame.height == receipt["prediction_rows"] == manifest["prediction_rows"] == receipt["population"]["count"],
            "Raw population count differs")
    expected_population = job["expected_populations"][split]
    require(frame.height == expected_population.get("count", expected_population.get("target_count")),
            "Raw target count differs from original frozen population")
    original_population = expected_population.get("original_population_identity")
    if original_population is None and "target_quantity_sha256" in expected_population:
        original_population = expected_population
    if original_population is not None:
        require(receipt["original_population_identity"] == original_population,
                "Original next-event target/quantity identity differs")
    for key, expected in (("dataset", job["dataset"]), ("arm", job["arm"]), ("seed", job["seed"]),
                          ("selected_epoch", receipt["selected_epoch"]),
                          ("checkpoint_state_sha256", receipt["state_sha256_before"])):
        require(frame[key].unique().to_list() == [expected], "Raw row model/selection identity differs: " + key)
    td, vd = truth_digests(frame)
    require(td == receipt["population"]["target_identity_sha256"] and vd == receipt["population"]["truth_sha256"],
            "Raw common-truth digest differs")
    actual = model_metrics(frame)
    verify_metric_receipt(receipt["metrics"], actual)
    return frame, {"kind": "new_fixed_checkpoint_diagnostic", "receipt_path": str(receipt_path),
                   "execution_mode": "SHA_authorized_validation_reuse" if cached else "fresh_current_native_contract",
                   "receipt_contract_sha256": expected_contract,
                   "native_executable_evidence": executable_proof,
                   "receipt_sha256": receipt_ref["sha256"], "worker_exit_sha256": exit_ref["sha256"],
                   "selected_epoch": receipt["selected_epoch"], "state_tensor_sha256": receipt["state_sha256_before"],
                   "checkpoint_file_sha256": receipt["checkpoint_file_sha256"], "source_closure": receipt["source_closure"],
                   "runtime": receipt["runtime"], "common_truth_sha256": vd,
                   "metrics": actual}


def load_legacy_test(bundle, reuse, job, bounds, evidence):
    root = Path(__file__).resolve().parents[2]
    receipt_path = confined(root, reuse["receipt"])
    receipt = evidence.read(receipt_path, reuse["receipt_sha256"])
    require(receipt.get("status") == "complete" and receipt.get("full_population") is True
            and receipt.get("split") == "test", "Legacy Test receipt is incomplete")
    registry = job["legacy_registry_row"]
    require(receipt["selected_epoch"] == registry["selected_epoch"]
            and receipt["checkpoint_file_sha256"] == registry["checkpoint_file_sha256"]
            and receipt["state_tensor_sha256"] == registry["state_tensor_sha256"],
            "Legacy Test checkpoint differs from frozen selection")
    frames = []
    for part in reuse["prediction_parts"]:
        path = confined(root, part["path"])
        evidence.verify(path, part["sha256"])
        frame = pl.read_parquet(path)
        require(frame.height == part["count"], "Reused prediction count differs")
        frames.append(frame)
    frame = canonical_frame(pl.concat(frames), split="test", bounds=bounds)
    require(frame.height == receipt["prediction_rows"], "Reused whole Test count differs")
    td, vd = truth_digests(frame)
    require(td == receipt["target_identity_sha256"], "Reused Test target digest differs")
    return frame, {"kind": "existing_Test_raw_predictions_reused_no_inference", "receipt_path": str(receipt_path),
                   "receipt_sha256": reuse["receipt_sha256"], "selected_epoch": receipt["selected_epoch"],
                   "state_tensor_sha256": receipt["state_tensor_sha256"],
                   "checkpoint_file_sha256": receipt["checkpoint_file_sha256"],
                   "source_closure": receipt["source_closure_sha256"], "runtime": receipt["runtime"],
                   "common_truth_sha256": vd, "metrics": model_metrics(frame)}


def group_summary(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key)), **{field: summarize([row[field] for row in values]) for field in fields}}
            for key, values in sorted(groups.items())]


def fallacy_scan():
    return [
        {"name": "Simpson's paradox", "status": "CHECKED", "action": "Micro, equal-entity and exact shared-entity decomposition are separately reported."},
        {"name": "Ecological fallacy", "status": "CHECKED", "action": "Per-example paired errors do not identify personal/entity causal effects."},
        {"name": "Non-collapsibility", "status": "NOT_APPLICABLE", "action": "No odds ratios or nonlinear regression coefficients are interpreted."},
        {"name": "Confounding", "status": "CAUTION", "action": "Component active capacity differs; external campaign/source/runtime differs. Histories and regimes are descriptive controls, not causal adjustment."},
        {"name": "Base rate neglect", "status": "CHECKED", "action": "Every group reports n and population contribution; conditional error is not substituted for the global contribution."},
        {"name": "Regression to the mean", "status": "CAUTION", "action": "Worst-entity ranks are posthoc descriptions with no improvement or independent replication claim."},
        {"name": "Survivorship bias", "status": "CHECKED", "action": "Only complete, approved 72 fixed fits with exact populations enter; all 216 model/split rows are required."},
        {"name": "Look-elsewhere effect", "status": "CAUTION", "action": "All seven pairs and declared partitions are retained; no formal significance claim or favorable slice-only conclusion."},
        {"name": "Garden of forking paths", "status": "CAUTION", "action": "Exploratory diagnostics after Test exposure; no new thresholds, checkpoint selection, fitting or exclusions."},
        {"name": "Correlation versus causation", "status": "CAUTION", "action": "Arithmetic attribution and observed associations do not prove loss/GRU/distribution causal mechanisms."},
        {"name": "Reverse causality", "status": "CHECKED", "action": "History fields precede targets; target-relative strata are outcome-conditioned diagnostics, not predictors available before the target."},
    ]


def run(bundle, execution_receipt, execution_sha):
    bundle = Path(bundle).resolve()
    evidence = Evidence()
    execution = evidence.read(execution_receipt, execution_sha)
    require(execution.get("status") == "complete", "Diagnostic execution has not completed")
    bindings_ref = execution["bindings"]
    bindings = evidence.read(confined(bundle, bindings_ref["path"]), bindings_ref["sha256"])
    contracts, contract_payloads = {}, {}
    require(len(execution["contracts"]) == 2, "Exactly two current native contracts are required")
    for ref in execution["contracts"]:
        contract = evidence.read(confined(bundle, ref["path"]), ref["sha256"])
        require(contract.get("new_training") is False and contract.get("frozen_selection") is True,
                "Wrong native diagnostic contract")
        require(contract.get("host") == ref["host"], "Native contract host differs")
        contracts[ref["host"]] = ref["sha256"]
        contract_payloads[ref["host"]] = contract
    require(set(contracts) == {"5080", "5090"}, "Two native host contracts are required")
    verify_pipeline_hosts(execution["pipeline_exits"])
    for ref in execution["pipeline_exits"]:
        exit_value = evidence.read(confined(bundle, ref["path"]), ref["sha256"])
        verify_exit_payload(exit_value, contracts[ref["host"]], pipeline=True)
    jobs = {j["id"]: j for j in bindings["jobs"]}
    require(len(jobs) == len(bindings["jobs"]) == 72, "Expected exactly 72 unique fixed fits")
    entries = {(e["job_id"], e["split"]): e for e in execution["jobs"]}
    expected = {(j["id"], split) for j in jobs.values() for split in j["allowed_splits"]}
    require(len(entries) == len(execution["jobs"]) == 162 and set(entries) == expected,
            "Expected complete unique 162 new diagnostic model/split jobs")
    reused = {r["id"]: r for r in bindings["external_Test_reuse"]}
    require(len(reused) == 54, "Expected 54 immutable external Test reuses")
    definitions = {}
    for job in jobs.values():
        definition = fixed_definition(job, evidence)
        bounds = definition["quantity_boundaries_all_train_rows"]
        require(len(bounds) == 4 and bounds == sorted(bounds), "Invalid frozen five quantity bins")
        require(definition["history_boundaries"] == [64, 128], "History cutoffs differ from frozen contract")
        if job["dataset"] in definitions:
            require(definitions[job["dataset"]]["quantity_boundaries_all_train_rows"] == bounds,
                    "Model families do not share fixed quantity bins")
        else:
            definitions[job["dataset"]] = definition
    model_rows, pair_rows, entity_reversals = [], [], []
    for dataset in DATASETS:
        bounds = definitions[dataset]["quantity_boundaries_all_train_rows"]
        for seed in (42, 52, 62):
            cnn_job = jobs[f"{dataset}__{seed}__{CNN}"]
            cnn_frames = {}
            for split in SPLITS:
                frame, info = load_new_rows(bundle, entries[(cnn_job["id"], split)], cnn_job, bounds, evidence, contracts, contract_payloads)
                cnn_frames[split] = frame
                model_rows.append({"job_id": cnn_job["id"], "dataset": dataset, "arm": CNN, "seed": seed,
                                   "split": split, **info})
            for other in (FULL,) + EXTERNALS:
                job = jobs[f"{dataset}__{seed}__{other}"]
                entity_groups = {}
                for split in SPLITS:
                    if other in EXTERNALS and split == "test":
                        frame, info = load_legacy_test(bundle, reused[job["id"]], job, bounds, evidence)
                    else:
                        frame, info = load_new_rows(bundle, entries[(job["id"], split)], job, bounds, evidence, contracts, contract_payloads)
                    model_rows.append({"job_id": job["id"], "dataset": dataset, "arm": other, "seed": seed,
                                       "split": split, **info})
                    left, right = (frame, cnn_frames[split]) if other == FULL else (cnn_frames[split], frame)
                    pair, grouped = analyze_pair(left, right)
                    entity_groups[split] = grouped
                    pair_rows.append({"comparison_family": "component" if other == FULL else "external",
                                      "dataset": dataset, "seed": seed, "split": split,
                                      "left_arm": FULL if other == FULL else CNN,
                                      "right_arm": CNN if other == FULL else other, **pair})
                entity_reversals.append({"dataset": dataset, "seed": seed,
                                         "left_arm": FULL if other == FULL else CNN,
                                         "right_arm": CNN if other == FULL else other,
                                         **shared_entity_decomposition(entity_groups["validation"], entity_groups["test"])})
            print(json.dumps({"dataset": dataset, "seed": seed, "paired_splits_analyzed": 21}), flush=True)
    require(len(model_rows) == 216 and len(pair_rows) == 189 and len(entity_reversals) == 63,
            "Complete analysis population count differs")
    metric_flat = [{**{k: r[k] for k in ("dataset", "arm", "seed", "split")}, **r["metrics"]} for r in model_rows]
    model_summary = group_summary(metric_flat, ("dataset", "arm", "split"), METRICS)
    pair_summary = group_summary(pair_rows, ("dataset", "left_arm", "right_arm", "split"),
                                 ("mse_gap", "rmse_gap", "mae_gap", "bias_gap", "log_mse_gap"))
    composition_summary = group_summary(entity_reversals, ("dataset", "left_arm", "right_arm"),
                                        ("global_mse_gap_change", "test_membership_term", "validation_membership_term",
                                         "shared_entity_weight_mix_term", "shared_entity_within_term",
                                         "validation_shared_mse_gap", "test_shared_mse_gap"))
    by_model = defaultdict(dict)
    for row in metric_flat:
        by_model[(row["dataset"], row["arm"], row["seed"])][row["split"]] = row
    generalization = []
    for (dataset, arm, seed), rows in sorted(by_model.items()):
        require(set(rows) == set(SPLITS), "A fixed fit is missing one split")
        generalization.append({"dataset": dataset, "arm": arm, "seed": seed,
                               "train_raw_rmse": rows["train"]["qty_rmse"],
                               "validation_raw_rmse": rows["validation"]["qty_rmse"],
                               "test_raw_rmse": rows["test"]["qty_rmse"],
                               "validation_minus_train_mse": rows["validation"]["qty_mse"] - rows["train"]["qty_mse"],
                               "test_minus_train_mse": rows["test"]["qty_mse"] - rows["train"]["qty_mse"],
                               "log_mse": {s: rows[s]["fixed_model_log_qty_mse"] for s in SPLITS},
                               "error_sse_over_truth_sumsq": {s: rows[s]["error_sse_over_truth_sumsq"] for s in SPLITS}})
    for path, expected in evidence.files.items():
        require(sha(path) == expected, "Source bytes changed during analysis: " + path)
    cache_count = sum(row.get("execution_mode") == "SHA_authorized_validation_reuse" for row in model_rows)
    declared_caches = sum(len(c.get("reused_validation", {})) for c in contract_payloads.values())
    require(cache_count == declared_caches, "Approved Validation caches were omitted or double counted")
    return {"schema": "titantpp_frozen_row_analysis_v1", "status": "complete",
            "verification_status": "ANALYZED", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "execution_receipt_path": str(Path(execution_receipt).resolve()),
            "execution_receipt_sha256": execution_sha, "bindings_sha256": bindings_ref["sha256"],
            "analysis_source_sha256": sha(__file__), "new_training": False, "checkpoint_reselection": False,
            "analysis_model_inference": False, "analysis_remote_calls": 0,
            "all_source_bytes_rechecked_after_analysis": True,
            "unique_native_split_conditions": 162, "fresh_current_native_split_conditions": 162 - cache_count,
            "SHA_authorized_validation_reuses": cache_count, "existing_external_test_reuses": 54,
            "model_split_count": len(model_rows), "paired_split_count": len(pair_rows),
            "group_contract": {"quantity_boundaries": {d: definitions[d]["quantity_boundaries_all_train_rows"] for d in DATASETS},
                               "history_boundaries": [64, 128], "recorded_gap": "each exact observed positive-integer value; no learned cutoffs",
                               "calendar": "recorded YYYYMM", "entity": "exact existing entity_id",
                               "regimes": "raw target below/equal/above last quantity and observed-history mean; outcome-conditioned exploratory diagnosis"},
            "model_splits": model_rows, "model_summaries": model_summary,
            "pair_splits": pair_rows, "pair_summaries": pair_summary,
            "shared_entity_reversal_splits": entity_reversals,
            "shared_entity_reversal_summaries": composition_summary,
            "fixed_fit_train_validation_test": generalization,
            "fallacy_scan": fallacy_scan(), "fallacy_coverage": "11/11",
            "limitations": [
                "Fixed-model Train errors are in-sample diagnostic errors, not an independent generalization estimate.",
                "No new checkpoint selection, thresholds, model fitting, correction or Test-driven architecture choice is performed.",
                "Quantity/target-to-history strata use the observed target and cannot be inference-time model-selection rules.",
                "Raw Train/Validation/Test populations differ in quantity severity, date and eligible target composition; raw gaps alone do not prove overfitting.",
                "A lower fixed-model log loss and higher raw RMSE reveals an objective/metric tradeoff but does not prove that changing loss would repair it.",
                "CNN+GRU and CNN have different active parameter counts; this experiment cannot separate GRU design from model capacity or optimization effects.",
                "External comparisons share frozen targets and quantity heads but differ in historical source/campaign/runtime; this is not a fresh same-runtime architecture experiment.",
                "Intermittent seeds42/52 use native5090/cuDNN92000 and seed62 native5080/cuDNN91900; paired component arms match within seed but cross-seed runtimes differ.",
                "Taxi and Intermittent chronological splits are entity-relative and have overlapping global calendar ranges, so common-calendar future deployment is not established.",
                "This Test has prior access history; exploratory diagnoses and top-entity ranks are not independent confirmatory evidence.",
                "Arithmetic membership/weight/within-entity attribution includes different targets and histories inside each entity and is not a causal decomposition.",
                "Existing external Test rows do not contain original log-space model losses; those fields remain N/A rather than being replaced by a loss derived from raw predictions.",
            ], "verified_source_files": evidence.files}


def markdown(out):
    lines = ["# 고정 checkpoint의 행별 오류 진단", "", "## Material Passport", "",
             "- Workflow: academic-research-suite / experiment-agent / validate",
             "- Verification Status: ANALYZED",
             "- 대상: 선택 모델72개, Train·Validation·Test216개 모델/분할 결과",
             f"- 고유 진단 조건162개(현재 계약의 실제 추론{out['fresh_current_native_split_conditions']}개, SHA 승인된 Validation 재사용{out['SHA_authorized_validation_reuses']}개)와 기존 외부 Test54개를 읽었습니다. 분석 코드 자체의 학습·추론·원격 호출은0입니다.",
             "- 기준: 최초 strict finite Validation raw 수량 RMSE 최소 epoch를 그대로 유지했습니다.",
             "- 통계: 3seed 평균과 표본SD, 행별 정확한 쌍 비교. 통계적 유의성·원인 확정은 주장하지 않습니다.", "",
             "## 고정 모델의 Train·Validation·Test raw 오류", "",
             "Train은 각 개체의 첫 행을 제외한 모든 예측 가능 next-event 대상입니다. 학습 중 온라인 log 손실과 고정 모델의 Train raw 오류를 혼용하지 않습니다.", "",
             "| 데이터셋 | 모델 | Train RMSE | Validation RMSE | Test RMSE | Train MAE | Validation MAE | Test MAE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    lookup = {(r["dataset"], r["arm"], r["split"]): r for r in out["model_summaries"]}
    for dataset in DATASETS:
        for arm in (FULL, CNN) + EXTERNALS:
            cells = []
            for metric in ("qty_rmse", "qty_mae"):
                for split in SPLITS:
                    stat = lookup[(dataset, arm, split)][metric]
                    cells.append(f"{stat['mean']:.5f}±{stat['sample_sd']:.5f}")
            lines.append("| " + " | ".join([DISPLAY[dataset], DISPLAY[arm], *cells]) + " |")
    lines += ["", "## 비교를 분리한 순위 변화", "",
              "차이는 왼쪽 모델−오른쪽 모델입니다. 양수면 오른쪽 모델의 오류가 작습니다. MSE 차이는 정확히 합산할 수 있고 RMSE 차이는 합산하지 않습니다.", "",
              "| 데이터셋 | 비교 | Train RMSE 차이 | Validation RMSE 차이 | Test RMSE 차이 | Train log MSE 차이 | Validation log MSE 차이 |", "|---|---|---:|---:|---:|---:|---:|"]
    pairs = {(r["dataset"], r["left_arm"], r["right_arm"], r["split"]): r for r in out["pair_summaries"]}
    for dataset in DATASETS:
        for left, right in ((FULL, CNN),) + tuple((CNN, a) for a in EXTERNALS):
            rmse = [pairs[(dataset, left, right, s)]["rmse_gap"]["mean"] for s in SPLITS]
            logs = [pairs[(dataset, left, right, s)]["log_mse_gap"]["mean"] for s in SPLITS[:2]]
            lines.append("| " + " | ".join([DISPLAY[dataset], f"{DISPLAY[left]}−{DISPLAY[right]}",
                                                *(f"{v:+.5f}" for v in rmse),
                                                *("N/A" if v is None else f"{v:+.6f}" for v in logs)]) + " |")
    lines += ["", "## 같은 개체의 오류 변화와 개체 구성 변화", "",
              "Validation→Test MSE 차이 변화를 네 항으로 합산했습니다. 한쪽에만 있는 개체의 구성, 공통 개체의 대상 수 가중치, 공통 개체 내부 오류 변화가 모두 포함됩니다. 이 수학적 분해는 인과효과를 식별하지 않습니다.", "",
              "| 데이터셋 | 비교 | 전체 MSE 격차 변화 | Test 개체구성 항 | Validation 개체구성 항 | 공통 개체 가중치 항 | 공통 개체 내 오류 항 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in out["shared_entity_reversal_summaries"]:
        values = [row[k]["mean"] for k in ("global_mse_gap_change", "test_membership_term", "validation_membership_term", "shared_entity_weight_mix_term", "shared_entity_within_term")]
        lines.append("| " + " | ".join([DISPLAY[row["dataset"]], f"{DISPLAY[row['left_arm']]}−{DISPLAY[row['right_arm']]}", *(f"{v:+.6f}" for v in values)]) + " |")
    lines += ["", "## 구간별 행 진단", "",
              "ROW_ANALYSIS.json의 pair_splits에는 각 seed·분할의 고정 수량5구간, history64/128, 실제 recorded gap, 월, target 대 last/mean 수량 방향, 수량과 history/방향의 교차 구간이 있습니다. 모든 구간은 대상 수·SSE·절대오차·signed bias를 보존합니다.",
              "개체별 집계는 모든 개체로 계산했습니다. 보고서의 상위10개 목록은 사후 오류 기여 설명용이며 새 하위집단 결론이나 선택 규칙으로 사용하지 않습니다.", "",
              "## 해석의 한계", ""]
    lines.extend("- " + item for item in out["limitations"])
    lines += ["", "## 통계적 오류 점검", "", "11/11 항목을 검토했습니다. 표본은 seed3개이고 개별 사건은 독립 실험 반복으로 취급하지 않습니다.", ""]
    lines.extend(f"- {item['name']}: {item['status']} — {item['action']}" for item in out["fallacy_scan"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--execution-receipt", required=True, type=Path)
    parser.add_argument("--execution-receipt-sha", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    out = run(args.bundle, args.execution_receipt, args.execution_receipt_sha)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("ROW_ANALYSIS.json", json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          ("ROW_ANALYSIS.md", markdown(out))):
        path = args.output_directory / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
        print(json.dumps({"path": str(path.resolve()), "sha256": sha(path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
