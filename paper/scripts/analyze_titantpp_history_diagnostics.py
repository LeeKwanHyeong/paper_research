#!/usr/bin/env python3
"""Stdlib-only analysis of the authorized, verified history27 diagnostics.

This reader cannot import Torch, load tensors, evaluate a model, contact a
server, select an epoch, or fit a model. It summarizes immutable sufficient
statistics only after all 54 native evaluations and their actual exits pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

KIND = "titantpp_history_diagnostics_20261012_v1"
SCHEMA = "titantpp_history_diagnostic_analysis_v1"
PARENT = "f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c"
SOURCE = "c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a"
CONTRACT = "38fb8820071fdb9b13f1576528d03da952a5a81a7e51b1a9db5d0cd97553ace4"
FULL = "titantpp_cnn_gru54"
ARMS = (FULL, FULL + "_correction_disabled", FULL + "_current_only")
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
SEEDS = (42, 52, 62)
METRICS = ("qty_rmse", "qty_mae", "time_nll")
SUM_KEYS = ("squared_error", "absolute_error", "signed_error", "time_nll")
NATIVE_SUMS = {"squared_error": "qty_sse", "absolute_error": "qty_absolute_error_sum",
               "signed_error": "qty_signed_error_sum", "time_nll": "time_nll_sum"}
PARTITIONS = {"all", "series", "quantity_bin", "history_bin", "last_input_gap_bin",
    "recorded_target_gap_bin", "left_truncated", "series_validation_membership",
    "recorded_target_calendar", "last_observed_input_calendar",
    "quantity_by_history_by_truncation", "quantity_by_series_validation_membership",
    "history_vs_train_max", "quantity_by_history_vs_train_max"}
FEATURES = ("duration_location", "duration_scale", "quantity_head_logit", "shared_head_hidden_norm",
            "cnn_q_residual_base_ratio", "cnn_k_residual_base_ratio", "cnn_v_residual_base_ratio",
            "gru_correction_base_ratio", "duration_standardized_log_residual")


class AnalysisError(ValueError):
    """Missing or inconsistent binding, population, exit, or statistic."""


def require(condition, message):
    if not condition:
        raise AnalysisError(message)


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite number: " + label)
    return float(value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), "Absolute or invalid artifact path")
    target = (Path(root) / relative).resolve()
    require(target.is_relative_to(Path(root).resolve()), "Artifact path escapes root: " + relative)
    return target


def read_bound(root, relative, files, expected=None):
    raw = safe(root, relative).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, "Artifact bytes differ: " + relative)
    if relative in files:
        require(files[relative] == digest, "Artifact changed during analysis: " + relative)
    files[relative] = digest
    return json.loads(raw)


def close(a, b, label, tolerance=None):
    tolerance = tolerance or {"absolute": 1e-7, "relative": 1e-9}
    require(math.isclose(finite(a, label), finite(b, label), rel_tol=tolerance["relative"],
                        abs_tol=tolerance["absolute"]), "Statistic differs: " + label)


def positive_count(value, label):
    require(type(value) is int and value > 0, "Invalid target count: " + label)
    return value


def check_exit(value, label, job_id=None, contract=None):
    require(type(value.get("exit_code")) is int and value["exit_code"] == 0
            and value.get("actual_child_exit_observed") is True, "Actual exit0 missing: " + label)
    if job_id is not None:
        require(value.get("job_id") == job_id and value.get("contract_sha256") == contract,
                "Worker identity differs: " + label)
        require(value.get("timed_out") is False and value.get("supervisor_aborted") is False,
                "Worker timed out or was aborted: " + label)


def require_contract_pin(c, expected_contract):
    require(isinstance(expected_contract, str) and len(expected_contract) == 64
            and all(ch in "0123456789abcdef" for ch in expected_contract),
            "An explicit exact lowercase SHA256 contract pin is required")
    require(c.get("contract_sha256") == expected_contract
            and canonical({k: v for k, v in c.items() if k != "contract_sha256"}) == expected_contract,
            "Pinned diagnostic contract canonical SHA differs")


def validate_contract(c, expected_contract=CONTRACT):
    require(c.get("schema") == KIND and c.get("parent_contract_sha256") == PARENT,
            "Foreign diagnostic contract")
    require_contract_pin(c, expected_contract)
    require(c.get("expected_seeds") == list(SEEDS) and set(c["sources"]) == {"history"}
            and c["sources"]["history"]["closure"] == SOURCE, "Frozen science closure differs")
    expected = {(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}
    require(len(c["jobs"]) == 27 and len({j["id"] for j in c["jobs"]}) == 27
            and {(j["dataset"], j["arm"], j["seed"]) for j in c["jobs"]} == expected
            and set(c["datasets"]) == set(DATASETS), "Selected history27 matrix differs")
    for job in c["jobs"]:
        require(job["study"] == "history" and job["frozen_source_closure"] == SOURCE
                and type(job["selected_epoch"]) is int and job["selected_epoch"] > 0,
                "Selected scientific identity differs")
    require(c["diagnostics"]["expected_split_evaluations"] == 54
            and c["diagnostics"]["aggregate_only"] is True
            and c["diagnostics"]["parameter_updates"] == 0
            and c["diagnostics"]["encoder_path_interventions"] == 0
            and c["diagnostics"]["stored_Test_is_independent"] is False,
            "Authorized observational scope differs")
    if "repair" in c:
        require(c["repair"].get("automatic_retry_or_resume_prior_run") is False
                and c["repair"].get("prior_failure_and_claims_preserved") is True
                and c["repair"].get("scientific_source_input_checkpoint_changed") is False
                and c["repair"].get("scope_still_selected27_Validation27_then_Test27") is True,
                "Separate sealed repair provenance differs")


def verify_originals(root, c, files):
    verification = read_bound(root, "originals/LOCAL_VERIFICATION.json", files)
    require(verification.get("status") == "verified"
            and verification.get("contract_sha256") == c["contract_sha256"]
            and verification.get("all_archive_members_and_local_files_sha_verified") is True
            and Path(verification["verified_root"]).resolve() == safe(root, "originals/verified"),
            "Verified immutable local originals are required")
    receipt = read_bound(root, "originals/receipt.json", files)
    require(receipt.get("status") == "verified_terminal_archived"
            and receipt.get("contract_sha256") == c["contract_sha256"]
            and receipt.get("actual_pipeline_exit0") is True
            and receipt.get("actual_worker_exit0_count") == 54
            and receipt.get("all27_validation_gate_pass") is True
            and receipt.get("native_qualification_pass") is True
            and receipt.get("own_processes_absent") is True and receipt.get("own_gpu_pids_absent") is True,
            "Terminal retrieval proof is incomplete")
    manifest = read_bound(root, "originals/manifest.json", files, verification["manifest_sha256"])
    archive = safe(root, "originals/originals.tar.gz")
    files["originals/originals.tar.gz"] = sha(archive)
    require(files["originals/originals.tar.gz"] == verification["archive_sha256"] == receipt["archive_sha256"]
            and files["originals/manifest.json"] == receipt["manifest_sha256"], "Archive/manifest bytes differ")
    require(manifest.get("contract_sha256") == c["contract_sha256"]
            and len(manifest["files"]) == verification["file_count"] == receipt["file_count"],
            "Original manifest identity/count differs")
    for relative, digest in manifest["files"].items():
        require(sha(safe(root, relative)) == digest, "Materialized original differs: " + relative)
        files[relative] = digest
        archived_relative = "originals/verified/" + relative
        require(sha(safe(root, archived_relative)) == digest, "Immutable extracted original differs: " + relative)
        files[archived_relative] = digest
    return verification


def verify_seals(root, c, files):
    seal = read_bound(root, "operation_integrity.json", files)
    inputs = read_bound(root, "input_integrity.json", files)
    require(seal["contract_sha256"] == inputs["contract_sha256"] == c["contract_sha256"],
            "Operation/input contract binding differs")
    for relative, digest in {**inputs["files"], **seal["files"]}.items():
        require(relative not in files or files[relative] == digest, "Seal bindings conflict: " + relative)
        require(sha(safe(root, relative)) == digest, "Sealed source/input/operation changed: " + relative)
        files[relative] = digest
    return seal


def verify_pipeline(root, c, files):
    complete = read_bound(root, "control/pipeline_complete.json", files)
    latest = read_bound(root, "control/pipeline_latest.json", files)
    actual = read_bound(root, "control/pipeline_exit.json", files)
    check_exit(actual, "pipeline OS exit")
    require(complete.get("status") == "complete" and complete.get("contract_sha256") == c["contract_sha256"]
            and complete.get("validation_count") == complete.get("test_count") == 27
            and complete.get("actual_all_worker_exit0") is True,
            "54-split pipeline completion is missing")
    require(latest.get("status") == latest.get("phase") == "complete"
            and latest.get("active") == [] and latest.get("contract_sha256") == c["contract_sha256"]
            and set(latest["validation_completed"]) == set(latest["test_completed"]) == {j["id"] for j in c["jobs"]},
            "Last pipeline phase is incomplete")
    gate = read_bound(root, "validation_gate.json", files, complete["validation_gate_sha256"])
    require(gate.get("schema") == "all27_native_diagnostic_validation_gate_v1"
            and gate.get("status") == "PASS" and gate.get("job_count") == 27
            and gate.get("contract_sha256") == c["contract_sha256"]
            and set(gate["validation_receipts"]) == set(gate["validation_worker_exits"]) == {j["id"] for j in c["jobs"]},
            "All27 native Validation gate is incomplete")
    qualification = read_bound(root, "control/native_qualification.json", files)
    require(qualification.get("status") == "PASS" and qualification.get("synthetic_only") is True
            and qualification.get("contract_sha256") == c["contract_sha256"]
            and len(qualification["routes"]) == 3, "Three-route native qualification is missing")
    for route, digest in qualification["routes"].items():
        r = read_bound(root, f"control/native_probes/{route}/receipt.json", files, digest)
        x = read_bound(root, f"control/native_probes/{route}/worker_exit.json", files)
        check_exit(x, "native probe " + route)
        require(r.get("status") == "PASS" and r.get("arm") in ARMS
                and r.get("contract_sha256") == c["contract_sha256"]
                and r.get("source_closure") == SOURCE and r.get("checkpoint_reads") == 0
                and r.get("real_data_loaded") is False and r.get("optimizer_updates") == 0
                and r.get("native_pinned_contract_checked") is True
                and r.get("hooked_and_unhooked_all_original_outputs_bitwise_equal") is True
                and r.get("telemetry_finite") is True and all(v is True for v in r["qualification"].values()),
                "Native hook noninterference proof differs: " + route)
    return gate


def metrics_of_cell(cell):
    n = positive_count(cell.get("count"), "stratum")
    sums = cell["sums"]
    sse, sae, signed, temporal = (finite(sums[k], k) for k in SUM_KEYS)
    require(sse >= 0 and sae >= 0, "Negative error mass")
    bias = signed / n
    variance = sse / n - bias * bias
    require(variance >= -1e-8 * max(1., sse / n), "Impossible signed-error second moment")
    return {"count": n, "qty_sse": sse, "qty_absolute_error_sum": sae,
            "qty_signed_error_sum": signed, "time_nll_sum": temporal,
            "qty_mse": sse / n, "qty_rmse": math.sqrt(sse / n), "qty_mae": sae / n,
            "qty_bias": bias, "error_population_variance": max(0., variance), "time_nll": temporal / n}


def summarize_features(cell):
    n = positive_count(cell["count"], "feature stratum")
    result = {}
    for name, a in cell["features"].items():
        require(a["count"] == n, "Feature observation count differs")
        mean = finite(a["sum"], name) / n
        variance = finite(a["sumsq"], name) / n - mean * mean
        require(variance >= -1e-8 * max(1., a["sumsq"] / n), "Impossible feature second moment")
        signed_mean = cell["sums"]["signed_error"] / n
        temporal_mean = cell["sums"]["time_nll"] / n
        signed_cov = a["sum_product_signed_error"] / n - mean * signed_mean
        temporal_cov = a["sum_product_time_nll"] / n - mean * temporal_mean
        close(mean, a["mean"], "feature mean " + name)
        close(max(0., variance), a["population_variance"], "feature variance " + name)
        close(signed_cov, a["covariance_with_signed_error"], "feature signed-error covariance " + name)
        close(temporal_cov, a["covariance_with_time_nll"], "feature TimeNLL covariance " + name)
        v = {"mean": mean, "population_sd": math.sqrt(max(0., variance)),
             "minimum": finite(a["minimum"], name), "maximum": finite(a["maximum"], name),
             "covariance_with_signed_error": signed_cov, "covariance_with_time_nll": temporal_cov}
        if name in FEATURES:
            result[name] = v
    return result


def check_telemetry(telemetry, native, tolerance):
    require(telemetry.get("schema") == "aggregate_association_telemetry_v1"
            and telemetry.get("association_is_causal_effect") is False
            and telemetry.get("raw_prediction_rows_written") is False
            and telemetry.get("target_quantity_and_recorded_target_calendar_and_gap_bins_are_posthoc") is True,
            "Telemetry interpretation boundary differs")
    n = positive_count(telemetry["count"], "telemetry")
    require(n == native["count"] and telemetry["quantity_boundaries"] == native["quantity_boundaries"],
            "Telemetry target count or bins differ")
    groups = telemetry["strata"]
    require(PARTITIONS | {"tail_by_series"} <= set(groups), "Missing required diagnostic strata")
    require(len(groups["all"]) == 1 and groups["all"][0]["key"] == "all", "Invalid all-target stratum")
    all_cell = groups["all"][0]
    all_metrics = metrics_of_cell(all_cell)
    for key in METRICS + ("qty_bias",):
        close(all_metrics[key], native[key], "telemetry all " + key, tolerance)
    for dimension, cells in groups.items():
        require(len({x["key"] for x in cells}) == len(cells), "Duplicate diagnostic stratum key")
        expected = native["tail"] if dimension == "tail_by_series" else native
        require(sum(positive_count(x["count"], dimension) for x in cells) == expected["count"],
                "Stratum partition target count differs: " + dimension)
        for key, native_key in NATIVE_SUMS.items():
            close(math.fsum(finite(x["sums"][key], dimension) for x in cells), expected[native_key],
                  "Partition sufficient statistic " + dimension + "/" + key, tolerance)
        for cell in cells:
            m = metrics_of_cell(cell)
            for key in METRICS + ("qty_bias",):
                close(m[key], cell[key], "Stored cell metric " + key, tolerance)
            summarize_features(cell)
    by_quantity = {x["key"]: x for x in groups["quantity_bin"]}
    for cell in native["quantity_cells"]:
        if cell["count"] == 0:
            require(str(cell["bin"]) not in by_quantity, "Zero-bin target count differs")
            continue
        observed = by_quantity[str(cell["bin"])]
        require(observed["count"] == cell["count"], "Quantity bin target count differs")
        for key in METRICS + ("qty_bias",):
            close(metrics_of_cell(observed)[key], cell[key], "Quantity bin replay " + key, tolerance)
    return groups


def compare_metrics(a, b, tolerance, label):
    require(a.get("split") == b.get("split") and a["count"] == b["count"]
            and a["tail"]["count"] == b["tail"]["count"], "Native reference population differs: " + label)
    for pa, pb in ((a, b), (a["tail"], b["tail"])):
        for key in METRICS:
            close(pa[key], pb[key], "Native replay " + label + "/" + key, tolerance)
        # Original selected-Validation summaries predate signed/absolute sums.
        # Compare available original moments; native references bind all moments.
        for key in tuple(NATIVE_SUMS.values()) + ("qty_bias",):
            if key in pb:
                close(pa[key], pb[key], "Native replay " + label + "/" + key, tolerance)


def collect(root, expected_contract=CONTRACT):
    root = Path(root).resolve()
    files = {}
    c = read_bound(root, "contract.json", files)
    validate_contract(c, expected_contract)
    require(not (root / "control/pipeline_failure.json").exists(),
            "This root has a preserved failed pipeline; it cannot become a completed diagnostic analysis")
    for job in c["jobs"]:
        for name in ("validation_replay_failure.json", "test_metrics_failure.json"):
            require(not safe(root, f"results/{job['id']}/{name}").exists(),
                    "This root has a preserved failed worker; no automatic retry or completion reinterpretation")
    verification = verify_originals(root, c, files)
    seal = verify_seals(root, c, files)
    gate = verify_pipeline(root, c, files)
    tolerance = c["diagnostics"]["same_native_tolerance"]
    gate_created = finite(gate["created_unix"], "Validation gate creation")
    rows, telemetry_by_job = [], {}
    split_identities, split_strata = {}, {}
    for job in sorted(c["jobs"], key=lambda j: (j["dataset"], j["seed"], j["arm"])):
        row = {k: job[k] for k in ("id", "dataset", "arm", "seed", "selected_epoch")}
        row.update(checkpoint_sha256=job["checkpoint"]["sha256"], state_sha256=job["checkpoint_state_sha256"],
                   source_closure=job["frozen_source_closure"], splits={})
        telemetry_by_job[job["id"]] = {}
        for split, name, exit_name in (("validation", "validation_replay.json", "validation_worker_exit.json"),
                                      ("test", "test_metrics.json", "worker_exit.json")):
            relative = f"results/{job['id']}/{name}"
            r = read_bound(root, relative, files,
                gate["validation_receipts"][job["id"]] if split == "validation" else None)
            xrelative = f"results/{job['id']}/{exit_name}"
            x = read_bound(root, xrelative, files,
                gate["validation_worker_exits"][job["id"]] if split == "validation" else None)
            check_exit(x, job["id"] + "/" + split, job["id"], c["contract_sha256"])
            require(r.get("status") == "complete" and r.get("schema") == KIND
                    and r.get("contract_sha256") == c["contract_sha256"] and r.get("split") == split,
                    "Native receipt identity differs")
            for key in ("id", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256"):
                require(r.get(key) == job[key], "Native selected identity differs: " + key)
            require(r.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
                    and r.get("state_sha256") == r.get("state_after_sha256") == job["checkpoint_state_sha256"]
                    and r.get("source_closure") == r.get("frozen_source_closure") == SOURCE
                    and r.get("native_reference_match") == r.get("diagnostics_match") == "PASS"
                    and r.get("aggregate_own_six_metrics_match") == "PASS"
                    and r.get("new_fits") == r.get("interventions") == r.get("optimizer_updates") == 0
                    and r.get("gradients_enabled") is False and r.get("no_forward_replacement") is True
                    and r.get("retrained") is False and r.get("reselected") is False
                    and r.get("raw_predictions_written") is False and r.get("test_independent") is False,
                    "Observational integrity or selected-state binding differs")
            require(r["evaluator_sha256"] == seal["files"]["control/evaluate.py"]
                    and r["telemetry_module_sha256"] == seal["files"]["control/titantpp_history_diagnostic_telemetry.py"]
                    and r["operation_integrity_sha256"] == files["operation_integrity.json"],
                    "Native operation file binding differs")
            require(r["runtime"]["gpu"]["uuid"] == c["runtime"]["gpu_uuid"], "Native physical GPU differs")
            for runtime_key in ("torch", "cuda", "cudnn", "numpy", "polars"):
                require(r["runtime"][runtime_key] == c["runtime"][runtime_key], "Native Runtime differs: " + runtime_key)
            expected_argv = [c["runtime"]["python"], c["execution"]["remote_root"] + "/control/evaluate.py",
                "--root", c["execution"]["remote_root"], "--job-id", job["id"], "--split", split, "--device", "cuda"]
            require(r["pid"] == x["pid"] and r["ppid"] == x["ppid"]
                    and x["argv"] == expected_argv and r["argv"] == expected_argv[1:]
                    and finite(r["completed_unix"], "receipt end") <= finite(x["completed_unix"], "actual child end"),
                    "Native process/actual-exit bridge differs")
            if split == "validation":
                require(r.get("validation_match") == "PASS" and r["validation_replay_check"]["passed"] is True
                        and r["completed_unix"] <= x["completed_unix"] <= gate_created,
                        "Validation completed after its global gate")
                compare_metrics(r["metrics"], job["validation_reference"], tolerance, job["id"])
            else:
                require(r["validation_gate_sha256"] == files["validation_gate.json"]
                        and finite(r["test_started_unix"], "Test start") >= gate_created,
                        "Test started before or outside all27 gate")
            ref = c["diagnostics"]["reference_receipts"][job["id"]][split]
            reference = read_bound(root, ref["relative_path"], files, ref["sha256"])
            require(reference.get("status") == "complete" and reference.get("contract_sha256") == PARENT
                    and reference.get("id") == job["id"] and reference.get("split") == split
                    and reference.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
                    and reference.get("state_sha256") == job["checkpoint_state_sha256"]
                    and r["native_reference_sha256"] == ref["sha256"], "Original native reference binding differs")
            compare_metrics(r["metrics"], reference["metrics"], tolerance, job["id"] + "/" + split)
            for key in ("target_identity_sha256", "truth_sha256", "original_population_identity"):
                require(r[key] == reference[key], "Exact native target/truth-history binding differs")
            identity = (r["target_identity_sha256"], r["truth_sha256"])
            splitkey = (job["dataset"], split)
            require(split_identities.setdefault(splitkey, identity) == identity, "Arms/seeds do not share exact targets")
            telemetry = read_bound(root, r["telemetry_path"], files, r["telemetry_sha256"])
            require(telemetry.get("job_id") == job["id"] and telemetry.get("split") == split
                    and telemetry.get("contract_sha256") == c["contract_sha256"], "Telemetry identity differs")
            dspec = c["datasets"][job["dataset"]]["diagnostics"]
            for key in ("input_history_boundaries", "input_last_gap_boundaries", "target_gap_boundaries", "train_history_max"):
                require(telemetry.get(key) == dspec[key], "Frozen Train-derived telemetry boundary differs: " + key)
            groups = check_telemetry(telemetry, r["metrics"], tolerance)
            pattern = {dimension: {cell["key"]: cell["count"] for cell in cells} for dimension, cells in groups.items()}
            require(split_strata.setdefault(splitkey, pattern) == pattern, "Arms/seeds have unmatched stratum targets/counts")
            row["splits"][split] = {"metrics": r["metrics"], "native_reference_match": "PASS",
                "target_identity_sha256": identity[0], "truth_history_sha256": identity[1],
                "telemetry_path": r["telemetry_path"], "telemetry_sha256": r["telemetry_sha256"],
                "receipt_sha256": files[relative], "actual_worker_exit_sha256": files[xrelative],
                "feature_associations": {dimension: [{"key": cell["key"], "count": cell["count"],
                    "observed_features": summarize_features(cell)} for cell in cells]
                    for dimension, cells in groups.items() if dimension != "series"}}
            telemetry_by_job[job["id"]][split] = telemetry
        rows.append(row)
    return c, rows, telemetry_by_job, files, verification


def contrast_cells(full_cells, control_cells, whole_count):
    """Exact SSE/SAE/NLL accounting; RMSE differences never add across cells."""
    full = {x["key"]: x for x in full_cells}
    control = {x["key"]: x for x in control_cells}
    require(set(full) == set(control), "Matched strata have different keys")
    nwhole = positive_count(whole_count, "contrast denominator")
    cells = []
    for key in sorted(full):
        f, c = full[key], control[key]
        require(f["count"] == c["count"], "Matched stratum counts differ: " + key)
        fm, cm = metrics_of_cell(f), metrics_of_cell(c)
        dsse = fm["qty_sse"] - cm["qty_sse"]
        dsae = fm["qty_absolute_error_sum"] - cm["qty_absolute_error_sum"]
        dnll = fm["time_nll_sum"] - cm["time_nll_sum"]
        cells.append({"key": key, "count": f["count"], "whole_count": nwhole,
            "target_share": f["count"] / nwhole, "full": fm, "control": cm,
            "difference": {"qty_rmse": fm["qty_rmse"] - cm["qty_rmse"],
                "qty_mae": fm["qty_mae"] - cm["qty_mae"], "time_nll": fm["time_nll"] - cm["time_nll"]},
            "sse_difference": dsse, "absolute_error_sum_difference": dsae, "time_nll_sum_difference": dnll,
            "whole_mse_contribution": dsse / nwhole, "whole_mae_contribution": dsae / nwhole,
            "whole_time_nll_contribution": dnll / nwhole,
            "bias_squared_mse_contribution": (fm["qty_bias"] ** 2 - cm["qty_bias"] ** 2) * f["count"] / nwhole,
            "centered_error_variance_mse_contribution":
                (fm["error_population_variance"] - cm["error_population_variance"]) * f["count"] / nwhole})
    positive = math.fsum(max(0., x["whole_mse_contribution"]) for x in cells)
    negative = -math.fsum(min(0., x["whole_mse_contribution"]) for x in cells)
    net = math.fsum(x["whole_mse_contribution"] for x in cells)
    for cell in cells:
        cell["share_of_positive_gross_mse"] = max(0., cell["whole_mse_contribution"]) / positive if positive else None
        close(cell["bias_squared_mse_contribution"] + cell["centered_error_variance_mse_contribution"],
              cell["whole_mse_contribution"], "Bias/variance error accounting")
    return {"cells": cells, "covered_count": sum(x["count"] for x in cells),
            "whole_count": nwhole, "positive_gross_whole_mse": positive,
            "negative_gross_whole_mse_magnitude": negative, "net_whole_mse_contribution": net,
            "denominator": "all targets in this split; each dimension is a separate partition",
            "rmse_is_additive": False, "association_is_causal": False}


def build_contrasts(rows, telemetry):
    lookup = {(r["dataset"], r["seed"], r["arm"]): r for r in rows}
    result = []
    for ds in DATASETS:
        for seed in SEEDS:
            full = lookup[(ds, seed, FULL)]
            for control_arm in ARMS[1:]:
                control = lookup[(ds, seed, control_arm)]
                value = {"dataset": ds, "seed": seed, "full_id": full["id"], "control_id": control["id"],
                         "control_arm": control_arm, "splits": {}}
                for split in ("validation", "test"):
                    f, c = (telemetry[r["id"]][split] for r in (full, control))
                    dimensions = {name: contrast_cells(cells, c["strata"][name], f["count"])
                                  for name, cells in f["strata"].items()}
                    fm, cm = (r["splits"][split]["metrics"] for r in (full, control))
                    for name, dimension in dimensions.items():
                        expected = (fm["tail"]["qty_sse"] - cm["tail"]["qty_sse"]) / fm["count"] if name == "tail_by_series" else (fm["qty_sse"] - cm["qty_sse"]) / fm["count"]
                        close(dimension["net_whole_mse_contribution"], expected, "Partition contrast " + name,
                              {"absolute": 1e-5, "relative": 1e-8})
                    value["splits"][split] = {"overall": {k: fm[k] - cm[k] for k in METRICS},
                        "tail": {k: fm["tail"][k] - cm["tail"][k] for k in METRICS},
                        "count": fm["count"], "tail_count": fm["tail"]["count"],
                        "dimensions": dimensions}
                result.append(value)
    return result


def quantity_bin_range(boundaries, key):
    """Describe the actual frozen bin; equality belongs to its lower bin."""
    require(isinstance(key, str) and key.isdigit(), "Invalid quantity-bin key")
    index = int(key)
    require(0 <= index <= len(boundaries), "Quantity-bin key outside frozen boundaries")
    if index == 0:
        return f"q ≤ {boundaries[0]:g}"
    if index == len(boundaries):
        return f"q > {boundaries[-1]:g}"
    return f"{boundaries[index - 1]:g} < q ≤ {boundaries[index]:g}"


def focus_quantity_bin(dataset, boundaries):
    """Select the stated physical range rather than assuming a numeric bin id."""
    if dataset == DATASETS[1]:
        matches = [i for i in range(1, len(boundaries))
                   if boundaries[i - 1] == 2 and boundaries[i] == 31]
        require(len(matches) == 1, "Intermittent focus requires frozen interval (2,31]")
        return str(matches[0])
    return str(len(boundaries))


def report_text(c, rows, contrasts):
    lines = ["# 저장 모델의 관측 진단 — 완료", "",
        "선택 checkpoint27개를 변경 없이 Validation·Test에서 각각 관측했습니다. 27개 Validation gate, 54개 실제 worker exit0, pipeline 실제 OS exit0, 원본 회수 archive/manifest SHA와 모든 원본 파일, 대상·truth/history digest 및 원래 여섯 지표의 재현을 검증한 뒤 분석했습니다.", "",
        "새 학습·새 epoch 선택·경로 제거 개입은 0입니다. 기존 Test는 이전에 노출된 자료이며 이번 진단도 독립 일반화 평가가 아닙니다.", "",
        "## 전체·큰 수량에서의 대조", "",
        "차이는 full − 대조입니다. 양수이면 손해입니다. 큰 수량은 원래 Train에서 고정한 상한 초과 구간입니다.", "",
        "| 자료 | seed | 대조 | split | 전체 ΔRMSE | 전체 ΔMAE | 전체 ΔTimeNLL | 큰 수량 ΔRMSE | 큰 수량 ΔMAE | 큰 수량 ΔTimeNLL |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    if "repair" in c:
        lines[6:6] = [
            "이 보고서는 별도 봉인한 복구 번들의 실제 완료 결과입니다. 이전 v1은 운영 metadata 오류로 Validation 완료0·Test 완료0·pipeline exit1이었고 원래 실패·claim을 보존했습니다. 과학 source·입력·checkpoint를 변경하거나 실패 결과를 수치 실패로 해석하지 않았습니다.", ""]
    short = {DATASETS[0]: "Taxi", DATASETS[1]: "Intermittent", DATASETS[2]: "RAF"}
    armshort = {ARMS[1]: "보정 제거", ARMS[2]: "current-only"}
    bounds = {ds: c["datasets"][ds]["definition"]["quantity_boundaries_all_train_rows"] for ds in DATASETS}
    focus = {ds: focus_quantity_bin(ds, bounds[ds]) for ds in DATASETS}
    for x in contrasts:
        for split, part in x["splits"].items():
            values = [part[p][k] for p in ("overall", "tail") for k in METRICS]
            lines.append(f"| {short[x['dataset']]} | {x['seed']} | {armshort[x['control_arm']]} | {split} | "
                         + " | ".join(f"{v:.6g}" for v in values) + " |")
    lines += ["", "## 손해·이득의 위치를 모두 포함한 분해", "",
        "각 차원은 같은 대상을 나눈 별도 분해입니다. 차원 간 합을 더하면 중복 집계입니다. ΔSSE/N은 전체 ΔMSE에 합산되며, RMSE 차이는 합산되지 않습니다. 조건별 signed bias는 예측−참값이고, centered error variance는 조건 내 오차분산입니다. 모델의 불확실성·seed 분산·calibration을 뜻하지 않습니다.", ""]
    for ds, seed, dimension in ((DATASETS[0], 52, "tail_by_series"), (DATASETS[1], 42, "quantity_by_history_by_truncation"),
                                (DATASETS[2], 42, "quantity_by_series_validation_membership"),
                                (DATASETS[2], 52, "quantity_by_series_validation_membership"),
                                (DATASETS[2], 62, "quantity_by_series_validation_membership")):
        for x in contrasts:
            if x["dataset"] != ds or x["seed"] != seed:
                continue
            d = x["splits"]["test"]["dimensions"][dimension]
            selected = sorted(d["cells"], key=lambda cell: cell["whole_mse_contribution"], reverse=True)
            if ds == DATASETS[1]:
                selected = [cell for cell in selected if cell["key"].startswith(focus[ds] + "/")]
            elif ds == DATASETS[2]:
                selected = [cell for cell in selected if cell["key"].startswith(focus[ds] + "/")]
            focused_range = quantity_bin_range(bounds[ds], focus[ds])
            lines += [f"### {short[ds]} seed{seed}, {armshort[x['control_arm']]} — Test", "",
                f"분해 차원 `{dimension}`; 선택 수량 구간 **{focused_range}** (quantity_bin={focus[ds]}); 전체 N={d['whole_count']}. 양의 손해 합 {d['positive_gross_whole_mse']:.6g}, 음의 이득 크기 {d['negative_gross_whole_mse_magnitude']:.6g}, 순 ΔMSE 기여 {d['net_whole_mse_contribution']:.6g}. 아래는 지정 구간의 기여이며 상세 JSON에는 모든 구간과 이득도 포함합니다.", "",
                "| 구간 key | N | ΔSSE/N전체 | full bias | 대조 bias | bias² 기여 | centered 오차분산 기여 | ΔTimeNLL 기여 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for cell in selected[:12]:
                lines.append(f"| {cell['key']} | {cell['count']} | {cell['whole_mse_contribution']:.6g} | {cell['full']['qty_bias']:.6g} | {cell['control']['qty_bias']:.6g} | {cell['bias_squared_mse_contribution']:.6g} | {cell['centered_error_variance_mse_contribution']:.6g} | {cell['whole_time_nll_contribution']:.6g} |")
            lines.append("")
    raf_tail_support = []
    for x in contrasts:
        if x["dataset"] == DATASETS[2]:
            cells = x["splits"]["test"]["dimensions"]["quantity_by_history_vs_train_max"]["cells"]
            within = sum(cell["count"] for cell in cells if cell["key"] == focus[DATASETS[2]] + "/within_train_max")
            above = sum(cell["count"] for cell in cells if cell["key"] == focus[DATASETS[2]] + "/above_train_max")
            require(within + above == x["splits"]["test"]["tail_count"], "RAF tail/history support count differs")
            raf_tail_support.append((within, above))
    require(len(raf_tail_support) == 6 and len(set(raf_tail_support)) == 1,
            "RAF matched tail/history support differs across seeds/controls")
    within, above = raf_tail_support[0]
    support_fact = f"RAF의 큰 수량 {within + above}건 중 {within}건은 Train 최대 이력 이하, {above}건은 초과였습니다."
    if above == 0:
        support_fact += " 큰 수량 손해는 이력14 초과 사례에서 발생하지 않았으므로, 15~19개 이력의 지원 범위 외삽만으로 이 손해를 설명할 수 없습니다. 이전 원인 후보를 이 관측에 따라 좁힙니다."
    lines += ["### RAF: Train 최대 이력과 큰 수량 손해의 실제 위치 — Test", "",
        "원래 Train의 최대 입력 이력14를 기준으로 나눈 추가 분해입니다. " + support_fact
        + " 이 구조에는 lag16에서 새로 켜지는 분기가 없으며, 이력 길이와 손해의 연관성을 원인으로 확정하지 않습니다.", "",
        "| seed | 대조 | 수량bin/이력 지원 | N | ΔSSE/N전체 | ΔTimeNLL 기여 |",
        "|---:|---|---|---:|---:|---:|"]
    for x in contrasts:
        if x["dataset"] != DATASETS[2]:
            continue
        d = x["splits"]["test"]["dimensions"]["quantity_by_history_vs_train_max"]
        for cell in d["cells"]:
            if cell["key"].endswith("above_train_max") or cell["key"].startswith(focus[DATASETS[2]] + "/"):
                lines.append(f"| {x['seed']} | {armshort[x['control_arm']]} | {cell['key']} | {cell['count']} | {cell['whole_mse_contribution']:.6g} | {cell['whole_time_nll_contribution']:.6g} |")
    lines += ["", "### Intermittent seed42: 기록된 목표 gap별 TimeNLL 손해·이득 — Test", "",
        "다음 gap은 모델에 공급되지 않은 사후 설명 변수입니다. TimeNLL 합 차이를 전체 N으로 나눠 합산합니다. μ·σ와 standardized log residual은 연속 lognormal 파라미터의 설명용 관측치이며 양의 정수 관측 확률질량 likelihood를 대신하지 않습니다.", "",
        "| 대조 | 목표gap bin | N | ΔTimeNLL 합 | 전체 TimeNLL 기여 | full μ 평균 | 대조 μ 평균 | full σ 평균 | 대조 σ 평균 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    rowlookup = {r["id"]: r for r in rows}
    for x in contrasts:
        if x["dataset"] != DATASETS[1] or x["seed"] != 42:
            continue
        dimension = "recorded_target_gap_bin"
        fassoc, cassoc = ({cell["key"]: cell["observed_features"] for cell in
            rowlookup[job_id]["splits"]["test"]["feature_associations"][dimension]}
            for job_id in (x["full_id"], x["control_id"]))
        for cell in sorted(x["splits"]["test"]["dimensions"][dimension]["cells"],
                           key=lambda cell: cell["whole_time_nll_contribution"], reverse=True):
            f, cc = fassoc[cell["key"]], cassoc[cell["key"]]
            lines.append(f"| {armshort[x['control_arm']]} | {cell['key']} | {cell['count']} | {cell['time_nll_sum_difference']:.6g} | {cell['whole_time_nll_contribution']:.6g} | {f['duration_location']['mean']:.6g} | {cc['duration_location']['mean']:.6g} | {f['duration_scale']['mean']:.6g} | {cc['duration_scale']['mean']:.6g} |")
    lines += ["## 실제 관측된 출력·이력의 연관성", "",
        "duration location·scale는 기록된 정수 시간 likelihood의 실제 hidden 입력에서 읽은 파라미터입니다. scale은 수량 예측의 불확실성이 아닙니다. CNN/GRU residual norm과 base 대비 비율은 부호 없는 벡터 크기이며 수량을 증가·감소시킨 효과가 아닙니다. covariance는 실제 joint moment로 계산한 관측 연관성이고, 경로의 인과 효과나 구성요소의 필요성 증명으로 해석할 수 없습니다.", "",
        "| 자료 | seed | arm | Test 구간 | N | μ 평균 | σ 평균 | CNN Q 잔차/base 평균 | GRU 보정/base 평균 | GRU 비율−signed error covariance |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        for cell in r["splits"]["test"]["feature_associations"]["quantity_bin"]:
            if cell["key"] != focus[r["dataset"]]:
                continue
            f = cell["observed_features"]
            norms = ([f['cnn_q_residual_base_ratio']['mean'], f['gru_correction_base_ratio']['mean'],
                      f['gru_correction_base_ratio']['covariance_with_signed_error']] if r["arm"] == FULL else [None] * 3)
            focused_range = quantity_bin_range(bounds[r["dataset"]], cell["key"])
            lines.append(f"| {short[r['dataset']]} | {r['seed']} | {armshort.get(r['arm'], 'full')} | {focused_range} (quantity_bin={cell['key']}) | {cell['count']} | {f['duration_location']['mean']:.6g} | {f['duration_scale']['mean']:.6g} | "
                         + " | ".join(f"{v:.6g}" if v is not None else "해당 경로 계측 대상 아님" for v in norms) + " |")
    lines += ["", "## 확인된 사실·원인 후보·미확인", "",
        "- **확인된 사실:** 위 분해는 동일 대조·동일 seed·동일 split·동일 대상에서 발생한 손해 위치와 이득의 상쇄를 보존합니다. 모든 좋은 seed와 조건도 JSON에 포함합니다. 시계열·이력·truncation·Validation series 포함 여부·입력 직전 gap·기록된 다음 gap·입력/목표 달력별로 개별 합과 전체 분모를 기록했습니다.",
        "- **원인 후보:** 특정 입력 이력과 보정 출력 크기·시간 head의 관측 연관성은 추가 기전 검토의 후보입니다. 같은 대상에서도 checkpoint/seed별 출력은 달라집니다. 관측 연관성만으로 CNN·GRU의 어느 경로가 손해를 만들었는지 확정하지 않습니다.",
        "- **미확인:** independently trained CNN-only/GRU-only ablation, 새 자료의 일반화, 설계 참신성·통계 유의성·투고 채택 가능성은 이번 계측으로 확인되지 않았습니다. 추가 학습·경로 개입·새 자료 평가는 별도 승인 범위입니다.", "",
        "실제 입력 이력은 Train에서 고정한 경계로 나눴습니다. 목표 수량·목표 시간 gap·목표 달력 구간은 사후 오류 설명용이며 미래 값을 이용한 운영 gating으로 사용할 수 없습니다. Intermittent의 기록된 달력은 합성 자료의 기록이며 새 미래의 관측 자료로 해석하지 않습니다.", "",
        "희소한 큰 수량83/39건과 같은 구간의 event는 series 안에서 연관될 수 있습니다. 표의 기여와 covariance는 해당 고정 대상의 기술통계이며 독립 표본의 인과 추정이나 유의성 검정 결과가 아닙니다.", "",
        "## 원본 추적", "",
        f"- 계약 canonical SHA `{c['contract_sha256']}`; 부모 Test 계약 `{PARENT}`.",
        f"- 과학 source129 closure `{SOURCE}`. 선택 checkpoint와 실제 native receipt의 SHA는 `diagnostic_conditions.json`의 각 조건 및 source_files에 기록했습니다.",
        "- 지표와 분해의 원본은 `results/<job_id>/{validation_replay,test_metrics,validation_telemetry,test_telemetry}.json`입니다. 기존 원본은 덮어쓰지 않았습니다.", ""]
    return "\n".join(lines)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def analyze(root, verify_only=False, expected_contract=CONTRACT):
    root = Path(root).resolve()
    c, rows, telemetry, files, verification = collect(root, expected_contract)
    contrasts = build_contrasts(rows, telemetry)
    if verify_only:
        return {"status": "verified", "jobs": 27, "split_evaluations": 54, "contrasts": 18,
                "contract_sha256": c["contract_sha256"], "new_fits": 0}
    analysis = root / "analysis"
    completion_path = analysis / "completion_receipt.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_bytes())
        require(completion.get("status") == "complete" and completion.get("contract_sha256") == c["contract_sha256"]
                and completion.get("source_files") == files
                and completion.get("approved_contract_sha256") == expected_contract
                and completion.get("analysis_script_sha256") == sha(__file__), "Existing analysis source bindings differ")
        for relative, expected in completion["analysis_files"].items():
            require(sha(safe(root, relative)) == expected, "Existing exclusive analysis output differs")
        return {"status": "verified_existing_complete", "jobs": 27, "split_evaluations": 54}
    require(not analysis.exists() or analysis.is_dir() and not any(analysis.iterdir()),
            "Partial analysis is preserved; no automatic overwrite/retry")
    require(not (root / "analysis_claim.json").exists(), "Existing analysis claim is preserved; no automatic retry")
    for relative, expected in files.items():
        require(sha(safe(root, relative)) == expected, "Input changed before exclusive analysis: " + relative)
    payload = {"schema": SCHEMA, "status": "complete", "created_unix": time.time(),
        "contract_sha256": c["contract_sha256"], "parent_contract_sha256": PARENT,
        "approved_contract_sha256": expected_contract, "repair_provenance": c.get("repair"),
        "existing_Test_is_independent": False, "new_fits": 0, "new_inference_by_analysis": 0,
        "remote_calls_by_analysis": 0, "path_interventions": 0, "reselection": False,
        "difference_direction": "full minus control; positive quantity/time metric difference is harm",
        "variance_definition": "conditional within-stratum signed prediction-error population variance; not model uncertainty or seed variance",
        "causal_effect_established": False, "rmse_difference_is_additive": False,
        "target_defined_strata_are_posthoc": True, "rows": rows, "contrasts": contrasts,
        "source_files": files, "local_originals_verification": verification,
        "analysis_script_sha256": sha(__file__)}
    # Format and reject nonfinite serialization before creating any output claim.
    json.dumps(payload, allow_nan=False)
    report = report_text(c, rows, contrasts)
    write_new(root / "analysis_claim.json", {"contract_sha256": c["contract_sha256"],
        "approved_contract_sha256": expected_contract, "created_unix": time.time(),
        "analysis_script_sha256": sha(__file__), "purpose": "exclusive verified aggregate diagnostic analysis"})
    analysis.mkdir(exist_ok=True)
    write_new(analysis / "diagnostic_conditions.json", payload)
    with (analysis / "DIAGNOSTIC_REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    outputs = {"analysis/diagnostic_conditions.json": sha(analysis / "diagnostic_conditions.json"),
               "analysis/DIAGNOSTIC_REPORT.md": sha(analysis / "DIAGNOSTIC_REPORT.md"),
               "analysis_claim.json": sha(root / "analysis_claim.json")}
    write_new(completion_path, {"schema": SCHEMA, "status": "complete", "completed_unix": time.time(),
        "contract_sha256": c["contract_sha256"], "jobs": 27, "split_evaluations": 54,
        "approved_contract_sha256": expected_contract, "repair_provenance": c.get("repair"),
        "actual_native_worker_exit0_count": 54, "all27_validation_gate_verified": True,
        "actual_pipeline_os_exit0_verified": True, "all_local_originals_verified": True,
        "all_native_reference_metrics_target_truth_history_match": True,
        "new_fits": 0, "new_inference_by_analysis": 0, "existing_Test_is_independent": False,
        "source_files": files, "analysis_files": outputs, "analysis_script_sha256": sha(__file__)})
    return {"status": "complete", "jobs": 27, "split_evaluations": 54, "contrasts": 18,
            "contract_sha256": c["contract_sha256"], "analysis_files": outputs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True,
                        help="Exact approved canonical contract SHA; never infer a pin from the inspected file")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.root, args.verify_only, args.contract_sha256), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
