#!/usr/bin/env python3
"""Stdlib-only selected-checkpoint Train/Validation/Test fit accounting.

Only immutable aggregates are read. This module cannot import a model, load a
tensor checkpoint, run inference, contact a server, fit a model, or reselect an
epoch. It fails closed before writing if any native completion binding differs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

CONTRACT = "76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf"
OPERATION = "157791af8ce197cfc858659e195eda314b2d1de7ca139971d627e51d6b91819b"
POPULATION = "e2bd494fc4519904fae8fce0049ede45bb4a842771a63be4e31ff7886fb07508"
POPULATION_PATH = Path(__file__).resolve().parents[2] / "reports/titantpp_test_reversal_diagnosis_20261012_v1/population_diagnosis.json"
PARENT = "70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d"
SOURCE = "c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a"
KIND = "titantpp_history_train_fit_20261012_v1"
SCHEMA = "titantpp_history_selected_train_fit_analysis_v1"
SPLITS = ("train", "validation", "test")
FULL = "titantpp_cnn_gru54"
ARMS = (FULL, FULL + "_correction_disabled", FULL + "_current_only")
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
SEEDS = (42, 52, 62)
METRICS = ("qty_rmse", "qty_mae", "time_nll", "qty_bias")


def helper():
    path = Path(__file__).resolve().with_name("analyze_titantpp_history_diagnostics.py")
    spec = importlib.util.spec_from_file_location("_verified_history_diagnostic_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = helper()
require, finite, close, sha, canonical, safe, read_bound = (
    D.require, D.finite, D.close, D.sha, D.canonical, D.safe, D.read_bound)


def validate_contract(c, expected_contract):
    D.require_contract_pin(c, expected_contract)
    require(c.get("schema") == KIND and c.get("parent_contract_sha256") == PARENT,
            "Foreign Train diagnostic contract")
    require(c.get("expected_seeds") == list(SEEDS) and set(c["sources"]) == {"history"}
            and c["sources"]["history"]["closure"] == SOURCE, "Scientific matrix/source changed")
    matrix = {(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}
    require(len(c["jobs"]) == 27 and len({j["id"] for j in c["jobs"]}) == 27
            and {(j["dataset"], j["arm"], j["seed"]) for j in c["jobs"]} == matrix,
            "Selected checkpoint27 matrix differs")
    fit = c["train_fit"]
    require(fit["evaluation_splits"] == ["train"] and fit["new_fits"] == 0
            and fit["validation_test_inference"] == 0 and fit["model_mode"] == "eval"
            and fit["gradients"] is False and fit["drop_last"] is False
            and fit["shuffle"] is False and fit["reference_train_metrics_available"] is False,
            "Train-only diagnostic scope changed")
    require(c["diagnostics"]["expected_split_evaluations"] == 27
            and c["diagnostics"]["parameter_updates"] == c["diagnostics"]["encoder_path_interventions"] == 0,
            "Additional scientific execution is outside this contract")
    for ds in DATASETS:
        require(fit["expected_train_populations"][ds]
                == c["datasets"][ds]["definition"]["inherited_data_identity"]["populations"]["train"],
                "Original eligible Train population binding differs")
    reuse = c["reused_diagnostics"]
    require(reuse["contract_sha256"] == PARENT
            and reuse["new_validation_evaluations"] == reuse["new_test_evaluations"] == 0,
            "Existing Validation/Test reuse scope changed")


def verify_originals(root, c, files):
    verification = read_bound(root, "originals/LOCAL_VERIFICATION.json", files)
    receipt = read_bound(root, "originals/receipt.json", files)
    require(verification.get("status") == "verified"
            and verification.get("contract_sha256") == c["contract_sha256"]
            and verification.get("all_archive_members_and_local_files_sha_verified") is True
            and Path(verification["verified_root"]).resolve() == safe(root, "originals/verified"),
            "Immutable Train originals are not locally verified")
    require(receipt.get("status") == "verified_terminal_archived"
            and receipt.get("contract_sha256") == c["contract_sha256"]
            and receipt.get("actual_pipeline_exit0") is True
            and receipt.get("actual_worker_exit0_count") == 27
            and receipt.get("native_qualification_pass") is True
            and receipt.get("all27_train_population_pass") is True
            and receipt.get("reused_validation_count") == receipt.get("reused_test_count") == 27
            and receipt.get("new_fits") == receipt.get("validation_test_inference") == 0
            and receipt.get("operation_integrity_sha256") == OPERATION
            and receipt.get("own_processes_absent") is True
            and receipt.get("own_gpu_pids_absent") is True,
            "Train terminal retrieval proof is incomplete")
    manifest = read_bound(root, "originals/manifest.json", files, verification["manifest_sha256"])
    files["originals/originals.tar.gz"] = sha(safe(root, "originals/originals.tar.gz"))
    require(files["originals/originals.tar.gz"] == receipt["archive_sha256"] == verification["archive_sha256"]
            and files["originals/manifest.json"] == receipt["manifest_sha256"], "Train archive binding differs")
    require(manifest["contract_sha256"] == c["contract_sha256"]
            and len(manifest["files"]) == verification["file_count"] == receipt["file_count"],
            "Train original manifest count/identity differs")
    for relative, digest in manifest["files"].items():
        for prefix in ("", "originals/verified/"):
            target = prefix + relative
            require(sha(safe(root, target)) == digest, "Train immutable original differs: " + target)
            files[target] = digest
    return verification


def verify_terminal(root, c, files):
    require(not safe(root, "control/pipeline_failure.json").exists(), "Preserved Train pipeline failure")
    complete = read_bound(root, "control/pipeline_complete.json", files)
    latest = read_bound(root, "control/pipeline_latest.json", files)
    actual = read_bound(root, "control/pipeline_exit.json", files)
    D.check_exit(actual, "Train pipeline OS exit")
    ids = {j["id"] for j in c["jobs"]}
    require(complete.get("status") == "complete" and complete.get("contract_sha256") == c["contract_sha256"]
            and complete.get("train_count") == 27
            and complete.get("reused_validation_count") == complete.get("reused_test_count") == 27
            and complete.get("actual_all_worker_exit0") is True and complete.get("new_fits") == 0,
            "Train27 completion/reused54 bridge is incomplete")
    require(set(complete["train_receipts"]) == set(complete["train_worker_exits"]) == ids,
            "Train pipeline completion omits native receipts or actual exits")
    require(latest.get("status") == latest.get("phase") == "complete" and latest.get("active") == []
            and latest.get("contract_sha256") == c["contract_sha256"]
            and set(latest.get("train_completed", [])) == ids, "Last Train pipeline phase is incomplete")
    qualification = read_bound(root, "control/native_qualification.json", files)
    expected_routes = {"history__" + arm for arm in ARMS}
    require(qualification.get("status") == "PASS" and qualification.get("synthetic_only") is True
            and qualification.get("contract_sha256") == c["contract_sha256"]
            and set(qualification["routes"]) == expected_routes, "Native three-route Train qualification is incomplete")
    qualification_end = finite(qualification["completed_unix"], "Native Train qualification end")
    for route, digest in qualification["routes"].items():
        arm = route.removeprefix("history__")
        p = read_bound(root, f"control/native_probes/{route}/receipt.json", files, digest)
        x = read_bound(root, f"control/native_probes/{route}/worker_exit.json", files)
        D.check_exit(x, "Train synthetic probe")
        require(p.get("status") == "PASS" and p.get("arm") == arm and p.get("source_closure") == SOURCE
                and p.get("contract_sha256") == c["contract_sha256"]
                and p.get("checkpoint_reads") == p.get("optimizer_updates") == 0
                and p.get("real_data_loaded") is False and p.get("native_pinned_contract_checked") is True
                and p.get("hooked_and_unhooked_all_original_outputs_bitwise_equal") is True
                and p.get("telemetry_finite") is True and all(v is True for v in p["qualification"].values())
                and finite(x["completed_unix"], "Train probe actual exit") <= qualification_end,
                "Native Train hook noninterference proof differs")
    return qualification_end, complete


def raw_quantity_baselines(native, baselines, tolerance):
    require(set(baselines) == {"last", "mean"}, "Simple quantity baseline pair missing")
    for name, b in baselines.items():
        require(b["count"] == native["count"] and b["tail"]["count"] == native["tail"]["count"]
                and b["quantity_boundaries"] == native["quantity_boundaries"], "Baseline population differs")
        require(all("time_nll" not in p and "time_nll_sum" not in p
                    for p in (b, b["tail"], *b["quantity_cells"])), "Quantity baseline has fabricated time score")
        for p in (b, b["tail"], *b["quantity_cells"]):
            n = p["count"]
            if not n:
                continue
            for field, value in (("qty_rmse", math.sqrt(p["qty_sse"] / n)),
                                 ("qty_mae", p["qty_absolute_error_sum"] / n),
                                 ("qty_bias", p["qty_signed_error_sum"] / n)):
                close(p[field], value, "Quantity baseline moment " + name + "/" + field, tolerance)


def selected_identity_equal(job, row):
    for key in ("id", "dataset", "arm", "seed", "selected_epoch"):
        require(job[key] == row[key], "Train/reused selected identity differs: " + key)
    require(job["checkpoint"]["sha256"] == row["checkpoint_sha256"]
            and job["checkpoint_state_sha256"] == row["state_sha256"]
            and job["frozen_source_closure"] == row["source_closure"] == SOURCE,
            "Train/reused checkpoint tensors or frozen source differ")


def population_description(telemetry, metrics):
    n = D.positive_count(metrics["count"], "split")
    all_cell = telemetry["strata"]["all"][0]
    tail_cells = [x for x in telemetry["strata"]["quantity_bin"]
                  if x["key"] == str(len(telemetry["quantity_boundaries"]))]
    require(len(tail_cells) == 1 and tail_cells[0]["count"] == metrics["tail"]["count"],
            "Large-quantity target population differs")
    tail = tail_cells[0]
    tail_mse = metrics["tail"]["qty_sse"] / tail["count"]
    tail_bias = metrics["tail"]["qty_signed_error_sum"] / tail["count"]
    centered = tail_mse - tail_bias * tail_bias
    require(centered >= -1e-8 * max(1., tail_mse), "Impossible tail error second moment")
    return {"count": n, "tail_count": tail["count"], "tail_share": tail["count"] / n,
        "true_quantity_mean": all_cell["sums"]["true_quantity"] / n,
        "predicted_quantity_mean": all_cell["sums"]["predicted_quantity"] / n,
        "tail_true_quantity_mean": tail["sums"]["true_quantity"] / tail["count"],
        "tail_predicted_quantity_mean": tail["sums"]["predicted_quantity"] / tail["count"],
        "tail_mse_whole_denominator_contribution": metrics["tail"]["qty_sse"] / n,
        "tail_share_of_whole_sse": metrics["tail"]["qty_sse"] / metrics["qty_sse"] if metrics["qty_sse"] else None,
        "tail_bias_squared_share_of_tail_mse": tail_bias * tail_bias / tail_mse if tail_mse else None,
        "tail_centered_error_population_variance": max(0., centered),
        "target_defined_strata_are_posthoc": True, "population_is_cross_split_identical": False}


def read_population_diagnosis(c):
    require(sha(POPULATION_PATH) == POPULATION, "Previously fixed descriptive population bytes differ")
    p = json.loads(POPULATION_PATH.read_bytes())
    require(p.get("status") == "PASS_DESCRIPTIVE_POPULATION_IDENTITIES"
            and p.get("source_closure") == SOURCE
            and p.get("contract_sha256") == D.PARENT
            and p.get("model_inference_calls") == p.get("training_calls") == p.get("checkpoint_loads") == p.get("remote_calls") == 0,
            "Existing population source/scope differs")
    require(set(p["datasets"]) == set(DATASETS), "Existing descriptive population datasets differ")
    for ds in DATASETS:
        require(c["datasets"][ds]["diagnostics"]["population_diagnosis_raw_sha256"] == POPULATION
                and p["datasets"][ds]["data"]["sha256"] == c["datasets"][ds]["data_sha256"],
                "Fixed descriptive population/data binding differs")
    return p


def add_population_distribution(description, native_split, prior_split, dataset, split):
    """Attach already-pinned target distribution moments, never infer them from errors."""
    require(prior_split["population_identity"]["target_count"] == description["count"]
            and prior_split["native_target_identity_sha256"] == native_split["target_identity_sha256"]
            and prior_split["native_truth_history_sha256"] == native_split["truth_history_sha256"],
            "Existing descriptive/native target-history identity differs: " + dataset + "/" + split)
    q = prior_split["quantity"]
    tail = prior_split["quantity_bin_target_statistics"][-1]["true_quantity"]
    require(q["count"] == description["count"] and tail["count"] == description["tail_count"],
            "Descriptive target distribution counts differ")
    close(q["mean"], description["true_quantity_mean"], "Descriptive whole target mean")
    close(tail["mean"], description["tail_true_quantity_mean"], "Descriptive tail target mean")
    description.update(true_quantity_population_sd=finite(q["population_sd"], "Target population SD"),
        true_quantity_population_variance=q["population_sd"] ** 2,
        true_quantity_minimum=q["minimum"], true_quantity_maximum=q["maximum"],
        tail_true_quantity_population_sd=finite(tail["population_sd"], "Tail target population SD"),
        tail_true_quantity_population_variance=tail["population_sd"] ** 2,
        tail_true_quantity_minimum=tail["minimum"], tail_true_quantity_maximum=tail["maximum"],
        distribution_source_sha256=POPULATION)


def collect(root, diagnostic_root, expected_contract=CONTRACT):
    root, diagnostic_root = Path(root).resolve(), Path(diagnostic_root).resolve()
    files = {}
    c = read_bound(root, "contract.json", files)
    validate_contract(c, expected_contract)
    population = read_population_diagnosis(c)
    require(not safe(root, "analysis_failure.json").exists(), "Prior analysis failure must remain preserved")
    verification = verify_originals(root, c, files)
    seal = D.verify_seals(root, c, files)
    require(files["operation_integrity.json"] == OPERATION, "Approved Train operation raw SHA differs")
    qualification_end, complete = verify_terminal(root, c, files)
    reused = c["reused_diagnostics"]
    require(diagnostic_root.name == reused["local_bundle_name"], "Reused diagnostic root identity differs")
    for relative, expected in (("contract.json", reused["contract_raw_sha256"]),
            ("originals/manifest.json", reused["manifest_sha256"]),
            ("originals/originals.tar.gz", reused["archive_sha256"]),
            ("analysis/completion_receipt.json", reused["analysis_completion_sha256"])):
        require(sha(safe(diagnostic_root, relative)) == expected, "Reused immutable diagnostic pin differs")
    pc, prior_rows, telemetry, prior_files, prior_verification = D.collect(diagnostic_root, expected_contract=PARENT)
    require(pc["contract_sha256"] == PARENT and prior_verification["status"] == "verified", "Reused diagnostic completion differs")
    previous_completion = json.loads(safe(diagnostic_root, "analysis/completion_receipt.json").read_bytes())
    require(previous_completion.get("status") == "complete"
            and previous_completion.get("contract_sha256") == PARENT
            and previous_completion.get("actual_native_worker_exit0_count") == 54
            and previous_completion.get("actual_pipeline_os_exit0_verified") is True
            and previous_completion.get("all_local_originals_verified") is True
            and previous_completion.get("source_files") == prior_files
            and previous_completion.get("analysis_script_sha256") == sha(D.__file__),
            "Reused diagnostic analysis/native evidence bridge differs")
    for relative, digest in previous_completion["analysis_files"].items():
        require(sha(safe(diagnostic_root, relative)) == digest, "Reused diagnostic analytical output differs")
    prior = {r["id"]: r for r in prior_rows}
    tolerance = c["diagnostics"]["same_native_tolerance"]
    rows, split_identities, train_patterns, baselines_by_split = [], {}, {}, {}
    for job in sorted(c["jobs"], key=lambda j: (j["dataset"], j["seed"], j["arm"])):
        selected_identity_equal(job, prior[job["id"]])
        row = prior[job["id"]]
        for split in ("validation", "test"):
            ref = reused["reference_receipts"][job["id"]][split]
            for path_key, sha_key in (("relative_path", "sha256"), ("telemetry_path", "telemetry_sha256")):
                require(sha(safe(diagnostic_root, ref[path_key])) == ref[sha_key], "Reused native split bytes differ")
            r = read_bound(diagnostic_root, ref["relative_path"], prior_files, ref["sha256"])
            row["splits"][split]["simple_quantity_baselines"] = r["simple_quantity_baselines"]
        relative = f"results/{job['id']}/train_metrics.json"
        require(not safe(root, f"results/{job['id']}/train_metrics_failure.json").exists(), "Preserved Train worker failure")
        r = read_bound(root, relative, files, complete["train_receipts"][job["id"]])
        exit_path = f"results/{job['id']}/train_worker_exit.json"
        x = read_bound(root, exit_path, files, complete["train_worker_exits"][job["id"]])
        D.check_exit(x, job["id"] + "/train", job["id"], c["contract_sha256"])
        require(r.get("status") == "complete" and r.get("schema") == KIND
                and r.get("contract_sha256") == c["contract_sha256"] and r.get("split") == "train",
                "Native Train receipt differs")
        for key in ("id", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256"):
            require(r.get(key) == job[key], "Native Train selected identity differs: " + key)
        require(r.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
                and r.get("state_sha256") == r.get("state_after_sha256") == job["checkpoint_state_sha256"]
                and r.get("source_closure") == r.get("frozen_source_closure") == SOURCE
                and r.get("train_population_match") == r.get("diagnostics_match") == r.get("aggregate_own_six_metrics_match") == "PASS"
                and r.get("new_fits") == r.get("interventions") == r.get("optimizer_updates") == r.get("validation_test_inference") == 0
                and r.get("gradients_enabled") is False and r.get("inference_mode_enabled") is True
                and r.get("no_forward_replacement") is True and r.get("retrained") is False
                and r.get("reselected") is False and r.get("raw_predictions_written") is False
                and r.get("prior_train_metric_reproduction_claimed") is False
                and r.get("train_state_eval_not_training_loss") is True
                and r.get("held_out_test_evaluated") is False, "Train observation/state invariants differ")
        expected = c["train_fit"]["expected_train_populations"][job["dataset"]]
        require(r["original_population_identity"] == expected
                and r["metrics"]["count"] == r["prediction_rows"] == r["expected_target_count"] == expected["target_count"],
                "Exact eligible Train target population differs")
        require(r["data_sha256"] == c["datasets"][job["dataset"]]["training_data_sha256"]
                and all(v is True for v in r["qualification"].values())
                and r["shuffle"] is False and r["drop_last"] is False
                and r["batch_size"] == 128 and r["num_workers"] == 0,
                "Train data/qualification/loader invariants differ")
        require(r["evaluator_sha256"] == seal["files"]["control/evaluate.py"]
                and r["telemetry_module_sha256"] == seal["files"]["control/titantpp_history_diagnostic_telemetry.py"]
                and r["operation_integrity_sha256"] == files["operation_integrity.json"], "Train operation bytes differ")
        argv = [c["runtime"]["python"], c["execution"]["remote_root"] + "/control/evaluate.py",
            "--root", c["execution"]["remote_root"], "--job-id", job["id"], "--split", "train", "--device", "cuda"]
        require(r["pid"] == x["pid"] and r["ppid"] == x["ppid"]
                and x["argv"] == argv and r["argv"] == argv[1:]
                and qualification_end <= finite(r["started_unix"], "Train start")
                <= finite(r["completed_unix"], "Train receipt end") <= finite(x["completed_unix"], "Train actual exit"),
                "Train qualification/native process/actual-exit bridge differs")
        require(r["runtime"]["gpu"]["uuid"] == c["runtime"]["gpu_uuid"], "Train physical GPU differs")
        for key in ("torch", "cuda", "cudnn", "numpy", "polars"):
            require(r["runtime"][key] == c["runtime"][key], "Train Runtime differs: " + key)
        identity = (r["target_identity_sha256"], r["truth_sha256"])
        require(split_identities.setdefault(job["dataset"], identity) == identity, "Train arms/seeds have different truth/history")
        t = read_bound(root, r["telemetry_path"], files, r["telemetry_sha256"])
        require(t["job_id"] == job["id"] and t["split"] == "train" and t["contract_sha256"] == c["contract_sha256"],
                "Train telemetry identity differs")
        dspec = c["datasets"][job["dataset"]]["diagnostics"]
        for key in ("input_history_boundaries", "input_last_gap_boundaries", "target_gap_boundaries", "train_history_max"):
            require(t[key] == dspec[key], "Train-derived diagnostic boundary differs")
        groups = D.check_telemetry(t, r["metrics"], tolerance)
        pattern = {name: {v["key"]: v["count"] for v in cells} for name, cells in groups.items()}
        require(train_patterns.setdefault(job["dataset"], pattern) == pattern, "Train strata targets/counts differ between arms/seeds")
        telemetry[job["id"]]["train"] = t
        row["splits"]["train"] = {"metrics": r["metrics"], "simple_quantity_baselines": r["simple_quantity_baselines"],
            "target_identity_sha256": identity[0], "truth_history_sha256": identity[1],
            "telemetry_path": r["telemetry_path"], "telemetry_sha256": r["telemetry_sha256"],
            "receipt_sha256": files[relative], "actual_worker_exit_sha256": files[exit_path],
            "prior_train_metric_reproduction_claimed": False,
            "feature_associations": {name: [{"key": cell["key"], "count": cell["count"],
                "observed_features": D.summarize_features(cell)} for cell in cells]
                for name, cells in groups.items() if name in ("all", "quantity_bin", "recorded_target_gap_bin")}}
        for split in SPLITS:
            value = row["splits"][split]
            raw_quantity_baselines(value["metrics"], value["simple_quantity_baselines"], tolerance)
            key = (job["dataset"], split)
            require(baselines_by_split.setdefault(key, value["simple_quantity_baselines"]) == value["simple_quantity_baselines"],
                    "Input-only quantity baselines differ across model/seed")
            value["population_description"] = population_description(telemetry[job["id"]][split], value["metrics"])
            add_population_distribution(value["population_description"], value,
                population["datasets"][job["dataset"]]["splits"][split], job["dataset"], split)
        rows.append(row)
    return c, rows, telemetry, files, prior_files, verification


def split_gap(train_metrics, other_metrics):
    """Descriptive cross-population difference, never a pure-overfit verdict."""
    return {"overall": {k: other_metrics[k] - train_metrics[k] for k in METRICS},
        "tail": {k: other_metrics["tail"][k] - train_metrics["tail"][k] for k in METRICS},
        "different_target_populations": True, "pure_overfitting_established": False,
        "new_independent_generalization_established": False}


def build_accounting(rows, telemetry):
    lookup = {(r["dataset"], r["seed"], r["arm"]): r for r in rows}
    contrasts = []
    for ds in DATASETS:
        for seed in SEEDS:
            full = lookup[(ds, seed, FULL)]
            for arm in ARMS[1:]:
                control = lookup[(ds, seed, arm)]
                value = {"dataset": ds, "seed": seed, "full_id": full["id"], "control_id": control["id"],
                         "control_arm": arm, "splits": {}}
                for split in SPLITS:
                    fm, cm = full["splits"][split]["metrics"], control["splits"][split]["metrics"]
                    require(fm["count"] == cm["count"] and fm["tail"]["count"] == cm["tail"]["count"],
                            "Unpaired within-split contrast")
                    f, c = telemetry[full["id"]][split], telemetry[control["id"]][split]
                    dimensions = {name: D.contrast_cells(f["strata"][name], c["strata"][name], fm["count"])
                        for name in ("quantity_bin", "recorded_target_gap_bin", "tail_by_series")}
                    for name, accounting in dimensions.items():
                        scope = "tail" if name == "tail_by_series" else None
                        fp, cp = (fm[scope], cm[scope]) if scope else (fm, cm)
                        close(accounting["net_whole_mse_contribution"], (fp["qty_sse"] - cp["qty_sse"]) / fm["count"],
                            "Paired additive MSE accounting", {"absolute": 1e-5, "relative": 1e-8})
                    value["splits"][split] = {"overall": {k: fm[k] - cm[k] for k in METRICS},
                        "tail": {k: fm["tail"][k] - cm["tail"][k] for k in METRICS}, "dimensions": dimensions}
                contrasts.append(value)
    gaps = [{"id": row["id"], "dataset": row["dataset"], "arm": row["arm"], "seed": row["seed"],
             "splits": {split: split_gap(row["splits"]["train"]["metrics"], row["splits"][split]["metrics"])
                        for split in ("validation", "test")}} for row in rows]
    means = []
    for ds in DATASETS:
        for arm in ARMS:
            group = [lookup[(ds, s, arm)] for s in SEEDS]
            value = {"dataset": ds, "arm": arm, "seeds": list(SEEDS), "sd_definition": "sample SD over three seeds; not a confidence interval", "splits": {}}
            for split in SPLITS:
                value["splits"][split] = {}
                for scope in ("overall", "tail"):
                    field = []
                    for row in group:
                        m = row["splits"][split]["metrics"]
                        field.append(m if scope == "overall" else m["tail"])
                    require(len({m["count"] for m in field}) == 1, "Seed mean populations differ")
                    value["splits"][split][scope] = {"count_per_seed": field[0]["count"],
                        **{k: {"mean": statistics.mean(m[k] for m in field),
                               "sample_sd": statistics.stdev(m[k] for m in field)} for k in METRICS}}
            means.append(value)
    return contrasts, gaps, means


def report_text(c, rows, contrasts, gaps, means):
    short = {"yellow_trip_hourly": "Taxi", "intermittent_frozen_5000": "Intermittent", "raf_spare_parts": "RAF"}
    armshort = {FULL: "full", ARMS[1]: "보정 제거", ARMS[2]: "current-only"}
    lines = ["# 같은 선택 모델의 Train·Validation·Test 적합 상태", "",
        "**완료:** 선택 checkpoint27개의 새 Train27 actual worker exit0·pipeline OS exit0·native qualification과 원본 SHA를 검증했다. Validation27·Test27은 완료된 v2 원본을 재사용했다. 새 학습·재선택·경로 개입·새 Validation/Test 추론은0이다.", "",
        "Train은 선택 checkpoint의 eval/no_grad 성능이며 학습 당시 mini-batch loss나 마지막 epoch loss가 아니다. 각 split의 대상·수량 분포가 다르므로 아래 격차를 순수한 과적합 또는 새로운 독립 일반화 증거로 판정하지 않는다. 평균 오차(bias)는 예측−실제다.", "",
        "## split의 모집단을 확인한다 — 완료", "",
        "| dataset | split | 전체 N | 큰 수량 N / 비중 | 전체 실제 수량 평균±SD | 큰 수량 실제 수량 평균±SD |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for ds in DATASETS:
        row = next(x for x in rows if x["dataset"] == ds and x["arm"] == FULL and x["seed"] == 42)
        for split in SPLITS:
            p = row["splits"][split]["population_description"]
            lines.append(f"| {short[ds]} | {split} | {p['count']} | {p['tail_count']} / {p['tail_share']:.3%} | {p['true_quantity_mean']:.6g}±{p['true_quantity_population_sd']:.6g} | {p['tail_true_quantity_mean']:.6g}±{p['tail_true_quantity_population_sd']:.6g} |")
    lines.extend(["", "큰 수량은 기존 Train 경계 Taxi>3449·Intermittent>187·RAF>200을 모든 split에 그대로 적용했다. 대상 수와 비중을 함께 보며, RMSE를 합산하지 않고 SSE/전체 N으로 분해한다.", "",
                  "## 좋은 seed와 모든 대조의 여섯 지표를 유지한다 — 완료"])
    for ds in DATASETS:
        lines.extend(["", f"### {short[ds]}", "", "| seed | arm | split | 전체 RMSE / MAE / bias | 큰 수량 RMSE / MAE / bias | 전체 / 큰 수량 TimeNLL |",
            "| ---: | --- | --- | ---: | ---: | ---: |"])
        for row in rows:
            if row["dataset"] != ds:
                continue
            for split in SPLITS:
                m = row["splits"][split]["metrics"]
                t = m["tail"]
                lines.append(f"| {row['seed']} | {armshort[row['arm']]} | {split} | {m['qty_rmse']:.6g} / {m['qty_mae']:.6g} / {m['qty_bias']:+.6g} | {t['qty_rmse']:.6g} / {t['qty_mae']:.6g} / {t['qty_bias']:+.6g} | {m['time_nll']:.6g} / {t['time_nll']:.6g} |")
    lines.extend(["", "## 세 seed의 평균과 표본 SD를 확인한다 — 완료", "",
        "| dataset | arm | split | 전체 RMSE 평균±SD | 큰 수량 RMSE 평균±SD | 전체 TimeNLL 평균±SD |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for value in means:
        for split in SPLITS:
            o, t = value["splits"][split]["overall"], value["splits"][split]["tail"]
            fmt = lambda x: f"{x['mean']:.6g}±{x['sample_sd']:.6g}"
            lines.append(f"| {short[value['dataset']]} | {armshort[value['arm']]} | {split} | {fmt(o['qty_rmse'])} | {fmt(t['qty_rmse'])} | {fmt(o['time_nll'])} |")
    lines.extend(["", "JSON에는 여섯 지표와 bias의 평균/표본 SD를 모두 보존했다. SD는 독립 표본 표준오차나 유의성·신뢰구간이 아니다.", "",
        "## 단순 수량 기준선을 보존한다 — 완료", "", "| dataset | split | 기준선 | 전체 RMSE / MAE / bias | 큰 수량 RMSE / MAE / bias |",
        "| --- | --- | --- | ---: | ---: |"])
    for ds in DATASETS:
        row = next(x for x in rows if x["dataset"] == ds and x["arm"] == FULL and x["seed"] == 42)
        for split in SPLITS:
            for name, m in row["splits"][split]["simple_quantity_baselines"].items():
                t = m["tail"]
                lines.append(f"| {short[ds]} | {split} | {'직전 수량' if name == 'last' else '과거 평균'} | {m['qty_rmse']:.6g} / {m['qty_mae']:.6g} / {m['qty_bias']:+.6g} | {t['qty_rmse']:.6g} / {t['qty_mae']:.6g} / {t['qty_bias']:+.6g} |")
    lines.extend(["", "수량 전용 기준선에 시간 점수0을 붙이지 않았다.", "", "## 수량 head와 시간 head 진단을 구분한다 — 완료", "",
        "| dataset | seed | arm | split | 평균 σ | 평균 standardized log residual / SD | 평균 수량 head logit |", "| --- | ---: | --- | --- | ---: | ---: | ---: |"])
    for row in rows:
        for split in SPLITS:
            f = row["splits"][split]["feature_associations"]["all"][0]["observed_features"]
            z = f["duration_standardized_log_residual"]
            lines.append(f"| {short[row['dataset']]} | {row['seed']} | {armshort[row['arm']]} | {split} | {f['duration_scale']['mean']:.6g} | {z['mean']:.6g} / {z['population_sd']:.6g} | {f['quantity_head_logit']['mean']:.6g} |")
    lines.extend(["", "σ는 log-duration scale이며 수량 uncertainty나 raw 시간 SD가 아니다. standardized residual은 정수 관측의 잠재 연속 잔차가 아니므로 정규분포 coverage로 해석하지 않는다. 실제 시간 점수는 원래 정수 확률질량이다. 평균 σ/logit에서 개별 NLL·log 수량 오차를 재구성하지 않는다.", "",
        "## 해석과 후속 실행의 경계를 유지한다 — 완료", "",
        "- Train tail의 높은 오차는 현 checkpoint가 학습 대상의 큰 수량 point를 정확히 재현하지 못한다는 관측이다. 목표 수량으로 조건을 건 평균 오차에는 평균회귀·확률성이 포함될 수 있어 곧바로 용량 부족·최적화 실패 또는 underfitting으로 판정하지 않는다. split 변화는 입력/대상 분포와 시간 변화를 함께 포함한다.",
        "- 수량 log1p MSE와 raw RMSE selector의 상충은 코드/수학 후보다. 현재 계측에는 원별 log1p 오차합이 없어 효과를 확정하지 않는다. 수량 과소예측과 Intermittent 시간 scale/location 취약성을 별도 가설로 둔다.",
        "- paired full−두 대조의18개 비교에 Train·Validation·Test 전체와 큰 수량·시간 손해를 모두 기록했다. 각 수량 bin/시간 gap/series의 SSE·SAE·NLL 기여는 JSON에 보존했고 RMSE 차이를 더하지 않았다.",
        "- 필요한 추가 aggregate 추론·한 가지 head 개입·학습 계약은 별도로 승인해야 한다. 이번 분석은 원고 수정·임의 σ floor·tail weight/LR 탐색·검색·새 학습을 실행하지 않았다.", "",
        "## 증적을 연결한다 — 완료", "",
        f"- Train canonical contract `{c['contract_sha256']}`; reused diagnostic contract `{PARENT}`; frozen source closure `{SOURCE}`.",
        "- `train_fit_conditions.json`의 각81개 split entry는 선택 epoch·checkpoint/state·target/truth-history·native receipt/actual exit/telemetry SHA에 연결된다. source_files와 reused_source_files는 두 root를 구분한다.",
        f"- 대상 수량의 SD·범위는 기존 model-inference0 모집단 문서 `{POPULATION_PATH}` rawSHA `{POPULATION}`에서 가져와 이번 native target/truth-history·수량 평균·개수와 정확히 맞췄다. 오류만으로 목표 수량 분산을 역산하지 않았다.",
        "- 본 로컬 reader는 stdlib 집계만 수행한다. model import·checkpoint tensor load·inference·원격 호출·학습·재선택은0이다.", ""])
    return "\n".join(lines)


def analyze(root, diagnostic_root, expected_contract=CONTRACT, verify_only=False):
    root = Path(root).resolve()
    c, rows, telemetry, files, prior_files, verification = collect(root, diagnostic_root, expected_contract)
    contrasts, gaps, means = build_accounting(rows, telemetry)
    if verify_only:
        return {"status": "verified", "jobs": 27, "new_train_evaluations": 27, "reused_split_evaluations": 54,
            "joined_split_entries": 81, "contrasts": len(contrasts), "contract_sha256": c["contract_sha256"]}
    analysis = root / "analysis"
    completion_path = analysis / "completion_receipt.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_bytes())
        require(completion["status"] == "complete" and completion["contract_sha256"] == expected_contract
                and completion["analysis_script_sha256"] == sha(__file__)
                and completion["source_files"] == files and completion["reused_source_files"] == prior_files
                and completion["existing_population_diagnosis"] == {"path": str(POPULATION_PATH), "sha256": POPULATION},
                "Existing fit analysis source bindings differ")
        for relative, digest in completion["analysis_files"].items():
            require(sha(safe(root, relative)) == digest, "Existing fit analysis bytes differ")
        return {"status": "verified_existing_complete", "jobs": 27, "joined_split_entries": 81}
    require(not (root / "analysis_claim.json").exists(), "Preserved exclusive analysis claim; automatic retry is forbidden")
    payload = {"schema": SCHEMA, "status": "complete", "created_unix": time.time(),
        "contract_sha256": c["contract_sha256"], "reused_contract_sha256": PARENT,
        "new_train_evaluations": 27, "reused_validation_evaluations": 27, "reused_test_evaluations": 27,
        "new_validation_test_inference": 0, "new_fits": 0, "new_inference_by_analysis": 0,
        "remote_calls_by_analysis": 0, "path_interventions": 0, "reselection": 0,
        "existing_Test_is_independent": False, "pure_overfitting_established": False,
        "causal_effect_established": False, "log_quantity_error_reconstructed": False,
        "continuous_time_density_used": False, "rows": rows, "contrasts": contrasts,
        "split_gaps": gaps, "seed_means_sample_sd": means, "source_files": files,
        "reused_source_files": prior_files, "local_originals_verification": verification,
        "existing_population_diagnosis": {"path": str(POPULATION_PATH), "sha256": POPULATION},
        "analysis_script_sha256": sha(__file__), "diagnostic_analysis_helper_sha256": sha(D.__file__)}
    json.dumps(payload, allow_nan=False)
    report = report_text(c, rows, contrasts, gaps, means)
    D.write_new(root / "analysis_claim.json", {"contract_sha256": expected_contract, "created_unix": time.time(),
        "analysis_script_sha256": sha(__file__), "purpose": "exclusive verified selected-checkpoint split-fit analysis"})
    analysis.mkdir(exist_ok=True)
    D.write_new(analysis / "train_fit_conditions.json", payload)
    with (analysis / "TRAIN_FIT_REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(report)
    outputs = {p: sha(root / p) for p in ("analysis/train_fit_conditions.json", "analysis/TRAIN_FIT_REPORT.md", "analysis_claim.json")}
    D.write_new(completion_path, {"schema": SCHEMA, "status": "complete", "completed_unix": time.time(),
        "contract_sha256": expected_contract, "actual_native_train_worker_exit0_count": 27,
        "actual_pipeline_os_exit0_verified": True, "reused_validation_count": 27, "reused_test_count": 27,
        "joined_split_entries": 81, "all_local_originals_verified": True, "same_selected_checkpoint27_verified": True,
        "prior_train_metrics_reproduction_claimed": False, "new_fits": 0, "new_inference_by_analysis": 0,
        "source_files": files, "reused_source_files": prior_files, "analysis_files": outputs,
        "existing_population_diagnosis": {"path": str(POPULATION_PATH), "sha256": POPULATION},
        "analysis_script_sha256": sha(__file__), "diagnostic_analysis_helper_sha256": sha(D.__file__)})
    return {"status": "complete", "jobs": 27, "new_train_evaluations": 27, "reused_split_evaluations": 54,
        "joined_split_entries": 81, "contrasts": len(contrasts), "analysis_files": outputs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--diagnostic-root", type=Path, required=True)
    p.add_argument("--expected-contract-sha256", required=True)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(analyze(args.root, args.diagnostic_root, args.expected_contract_sha256, args.verify_only), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
