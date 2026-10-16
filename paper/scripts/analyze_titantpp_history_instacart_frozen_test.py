#!/usr/bin/env python3
"""Verify and summarize the authorized frozen history27 + Instacart9 Test comparison.

This stdlib-only reader never trains, chooses an epoch, or opens another campaign.
Validation reference/replay and Test metrics remain separately labelled throughout.
An existing completion is reusable only while all bound input/output bytes agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


SCHEMA = "titantpp_history_instacart_test_analysis_v1"
SCOPE = "test_final_comparison"
SEEDS = (42, 52, 62)
FULL = "titantpp_cnn_gru54"
ARMS = {
    "history": (FULL, "titantpp_cnn_gru54_correction_disabled", "titantpp_cnn_gru54_current_only"),
    "instacart": (FULL, "titantpp_history_mlp_width8", "titantpp_history_mlp_width12"),
}
METRICS = ("qty_rmse", "qty_mae", "time_nll")
METRIC_KEYS = tuple(f"{population}_{metric}" for population in ("overall", "tail") for metric in METRICS)


class AnalysisError(ValueError):
    """A required frozen binding, exit, or population does not agree."""


def require(condition, message):
    if not condition:
        raise AnalysisError(message)


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_bound(path):
    """Parse exactly the bytes whose SHA is recorded, avoiding a split read/hash."""
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    require(target.is_relative_to(root), f"Path leaves the bundle: {relative}")
    return target


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def digest_value(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"Invalid SHA256 for {name}")
    return value


def contract_digest(contract):
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def flatten(metrics, split, label):
    require(isinstance(metrics, dict) and metrics.get("split") == split,
            f"Wrong or missing split for {label}: expected {split}")
    tail = metrics.get("tail")
    require(isinstance(tail, dict), f"Missing tail population: {label}")
    counts = [metrics.get("count"), tail.get("count")]
    require(all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in counts)
            and counts[1] <= counts[0], f"Invalid population counts: {label}")
    values = {}
    for population, source in (("overall", metrics), ("tail", tail)):
        if "split" in source:
            require(source["split"] == split, f"Mixed split in {label}/{population}")
        for metric in METRICS:
            value = source.get(metric)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                    f"Missing/nonfinite metric: {label}/{population}/{metric}")
            require(metric == "time_nll" or value >= 0, f"Negative quantity metric: {label}/{population}/{metric}")
            values[f"{population}_{metric}"] = float(value)
    return values


def validate_contract(contract):
    digest_value(contract.get("contract_sha256"), "contract")
    require(contract["contract_sha256"] == contract_digest(contract), "Contract canonical SHA mismatch")
    require(contract.get("expected_seeds") == list(SEEDS), "Frozen seed set differs")
    require(contract.get("evaluation_scope") == SCOPE, "Wrong contract evaluation scope")
    require(contract.get("metrics_schema") == "quantity_time_tail_v1", "Wrong frozen metrics schema")
    tolerance = contract.get("validation_replay_tolerance")
    require(isinstance(tolerance, dict), "Missing frozen Validation replay tolerance")
    for key in ("absolute", "relative"):
        value = tolerance.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0, f"Invalid replay tolerance: {key}")
    jobs = contract.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 36, "Exactly 36 frozen jobs are required")
    identities, ids = set(), set()
    for job in jobs:
        job_id = job.get("id")
        require(isinstance(job_id, str) and job_id and all(c.isalnum() or c in "_-" for c in job_id),
                "Unsafe or empty job id")
        study, arm, dataset, seed = (job.get(k) for k in ("study", "arm", "dataset", "seed"))
        require(study in ARMS and arm in ARMS[study] and isinstance(dataset, str) and dataset,
                f"Unknown study/dataset/arm: {job_id}")
        require(seed in SEEDS and not isinstance(seed, bool), f"Unexpected seed: {job_id}")
        epoch = job.get("selected_epoch")
        require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch > 0,
                f"Invalid selected epoch: {job_id}")
        digest_value(job.get("checkpoint", {}).get("sha256"), f"checkpoint/{job_id}")
        digest_value(job.get("frozen_source_closure"), f"frozen source/{job_id}")
        flatten(job.get("validation_reference"), "validation", f"contract/{job_id}")
        identity = (study, dataset, arm, seed)
        require(job_id not in ids and identity not in identities, "Duplicate scientific identity or job id")
        ids.add(job_id)
        identities.add(identity)
    for study, count in (("history", 3), ("instacart", 1)):
        datasets = {job["dataset"] for job in jobs if job["study"] == study}
        require(len(datasets) == count, f"Wrong {study} dataset count")
        expected = {(study, dataset, arm, seed) for dataset in datasets for arm in ARMS[study] for seed in SEEDS}
        require({identity for identity in identities if identity[0] == study} == expected,
                f"Incomplete {study} arm/seed matrix")
    datasets = contract.get("datasets")
    require(isinstance(datasets, dict) and set(datasets) == {job["dataset"] for job in jobs},
            "Frozen dataset registry differs")
    for dataset, spec in datasets.items():
        digest_value(spec.get("data_sha256"), f"Test raw data/{dataset}")
        digest_value(spec.get("training_data_sha256"), f"Validation raw data/{dataset}")
        count = spec.get("expected_test_count")
        require(type(count) is int and count > 0, f"Missing expected Test count: {dataset}")
    return jobs


def verify_replay(reference, replay, contract, label):
    tolerance = contract["validation_replay_tolerance"]
    for metric in METRIC_KEYS:
        require(math.isclose(reference[metric], replay[metric], rel_tol=tolerance["relative"],
                             abs_tol=tolerance["absolute"]), f"Validation replay tolerance exceeded: {label}/{metric}")


def finite_timestamp(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and value >= 0, f"Invalid observed timestamp: {label}")
    return float(value)


def verify_exit(worker_exit, job, contract, label):
    require(type(worker_exit.get("exit_code")) is int and worker_exit["exit_code"] == 0
            and worker_exit.get("job_id") == job["id"]
            and worker_exit.get("contract_sha256") == contract["contract_sha256"],
            f"Actual successful worker exit is missing: {label}")
    if "actual_child_exit_observed" in worker_exit:
        require(worker_exit["actual_child_exit_observed"] is True, f"Worker exit not observed: {label}")


def verify_native_bindings(native, job, contract, split):
    require(native.get("split") == split, f"Wrong native output split: {job['id']}")
    for key, expected in (("contract_sha256", contract["contract_sha256"]), ("job_id", job["id"]),
                          ("selected_epoch", job["selected_epoch"]),
                          ("checkpoint_sha256", job["checkpoint"]["sha256"]),
                          ("frozen_source_closure", job["frozen_source_closure"])):
        require(native.get(key) == expected and type(native.get(key)) is type(expected),
                f"Native frozen {key} differs: {job['id']}/{split}")
    for job_key, native_key in (("checkpoint_state_sha256", "state_sha256"),
                                ("initial_state_sha256", "initial_state_sha256")):
        if job_key in job:
            require(native.get(native_key) == job[job_key], f"Native tensor/initial state differs: {job['id']}/{split}")


def validation_gate(bundle, contract, jobs, files):
    gate, gate_sha = read_bound(bundle / "validation_gate.json")
    files["validation_gate.json"] = gate_sha
    expected_ids = {job["id"] for job in jobs}
    require(gate.get("status") == "PASS" and gate.get("contract_sha256") == contract["contract_sha256"]
            and gate.get("job_count") == 36 and set(gate.get("validation_receipts", {})) == expected_ids,
            "Global Validation36 gate is incomplete or unbound")
    created = finite_timestamp(gate.get("created_unix"), "Validation gate creation")
    native_rows = {}
    for job in jobs:
        name = f"results/{job['id']}/validation_replay.json"
        exit_name = f"results/{job['id']}/validation_worker_exit.json"
        native, native_sha = read_bound(safe(bundle, name))
        worker_exit, exit_sha = read_bound(safe(bundle, exit_name))
        files.update({name: native_sha, exit_name: exit_sha})
        require(gate["validation_receipts"][job["id"]] == native_sha,
                f"Global gate Validation receipt raw SHA differs: {job['id']}")
        verify_native_bindings(native, job, contract, "validation")
        verify_exit(worker_exit, job, contract, f"{job['id']}/Validation")
        for key in ("target_identity_sha256", "truth_sha256"):
            digest_value(native.get(key), f"Validation native {key}/{job['id']}")
        require(native.get("data_identity", {}).get("raw_sha256")
                == contract["datasets"][job["dataset"]]["training_data_sha256"],
                f"Native Validation raw data SHA differs: {job['id']}")
        require(native.get("validation_replay_check", {}).get("passed") is True,
                f"Native Validation replay check did not pass: {job['id']}")
        reference = flatten(job["validation_reference"], "validation", job["id"])
        replay = flatten(native.get("metrics"), "validation", f"native/{job['id']}")
        verify_replay(reference, replay, contract, f"native/{job['id']}")
        require(native["metrics"]["count"] == job["validation_reference"]["count"]
                and native["metrics"]["tail"]["count"] == job["validation_reference"]["tail"]["count"],
                f"Native Validation counts changed: {job['id']}")
        require(finite_timestamp(native.get("completed_unix"), f"{job['id']}/Validation completion") <= created,
                f"Validation completed after global gate: {job['id']}")
        if "completed_unix" in worker_exit:
            require(finite_timestamp(worker_exit["completed_unix"], f"{job['id']}/Validation exit") <= created,
                    f"Validation worker exit occurred after global gate: {job['id']}")
        native_rows[job["id"]] = native
    return gate, gate_sha, native_rows


def baseline_metrics(value, split, label):
    """Quantity-only simple baselines have no fabricated TimeNLL."""
    require(isinstance(value, dict) and set(value) == {"last", "mean"},
            f"Both last and mean simple baselines are required: {label}")
    def no_timing_score(node):
        if isinstance(node, dict):
            require("time_nll" not in node and "time_nll_sum" not in node,
                    f"Quantity baseline must not invent TimeNLL: {label}")
            for child in node.values():
                no_timing_score(child)
        elif isinstance(node, list):
            for child in node:
                no_timing_score(child)
    no_timing_score(value)
    for name, metrics in value.items():
        require(name in ("last", "mean"), f"Unknown simple baseline: {label}/{name}")
        require(metrics.get("split") == split, f"Wrong baseline split: {label}/{name}")
        for population in (metrics, metrics.get("tail", {})):
            require(isinstance(population.get("count"), int) and not isinstance(population.get("count"), bool)
                    and population["count"] > 0, f"Invalid baseline count: {label}/{name}")
            for metric in ("qty_rmse", "qty_mae"):
                require(isinstance(population.get(metric), (int, float)) and not isinstance(population[metric], bool)
                        and math.isfinite(population[metric]) and population[metric] >= 0,
                        f"Invalid baseline metric: {label}/{name}/{metric}")
    return value


def collect(bundle):
    bundle = Path(bundle).resolve()
    contract_path = bundle / "contract.json"
    contract, contract_raw_sha = read_bound(contract_path)
    jobs = validate_contract(contract)
    files = {"contract.json": contract_raw_sha}
    gate, gate_sha, validation_rows = validation_gate(bundle, contract, jobs, files)
    records = []
    for job in jobs:
        name = f"results/{job['id']}/receipt.json"
        exit_name = f"results/{job['id']}/worker_exit.json"
        receipt, receipt_sha = read_bound(safe(bundle, name))
        worker_exit, worker_exit_sha = read_bound(safe(bundle, exit_name))
        files.update({name: receipt_sha, exit_name: worker_exit_sha})
        for key in ("job_id", "dataset", "arm", "seed", "selected_epoch"):
            expected = job["id"] if key == "job_id" else job[key]
            require(receipt.get(key) == expected and type(receipt.get(key)) is type(expected),
                    f"Frozen {key} differs: {job['id']}")
        for key, expected in (("contract_sha256", contract["contract_sha256"]),
                              ("checkpoint_sha256", job["checkpoint"]["sha256"]),
                              ("frozen_source_closure", job["frozen_source_closure"])):
            require(receipt.get(key) == expected, f"Frozen {key} differs: {job['id']}")
        verify_exit(worker_exit, job, contract, job["id"])
        require(receipt.get("retrained") is False and receipt.get("reselected") is False,
                f"Training/reselection was not ruled out: {job['id']}")
        require(receipt.get("test_is_independent") is False, "Previously exposed Test must not become independent evidence")
        if "evaluation_scope" in receipt:
            require(receipt["evaluation_scope"] == SCOPE, f"Wrong receipt scope: {job['id']}")
        check = receipt.get("validation_replay_check", {})
        require(check.get("passed") is True, f"Validation replay did not pass: {job['id']}")
        ref = flatten(receipt.get("validation_reference"), "validation", f"reference/{job['id']}")
        require(ref == flatten(job["validation_reference"], "validation", f"contract/{job['id']}"),
                f"Validation reference metrics changed: {job['id']}")
        replay = flatten(receipt.get("validation_replay"), "validation", f"replay/{job['id']}")
        verify_replay(ref, replay, contract, job["id"])
        test = flatten(receipt.get("test"), "test", f"test/{job['id']}")
        require(receipt["validation_replay"] == validation_rows[job["id"]]["metrics"],
                f"Combined Validation replay differs from gated native output: {job['id']}")
        require(receipt.get("validation_gate_sha256") == gate_sha,
                f"Test receipt global Validation gate SHA differs: {job['id']}")
        require(finite_timestamp(receipt.get("test_started_unix"), f"{job['id']}/Test start") >= gate["created_unix"],
                f"Test started before all Validation36 passed: {job['id']}")
        test_name = f"results/{job['id']}/test_metrics.json"
        native_test, native_test_sha = read_bound(safe(bundle, test_name))
        files[test_name] = native_test_sha
        verify_native_bindings(native_test, job, contract, "test")
        for key in ("target_identity_sha256", "truth_sha256"):
            digest_value(native_test.get(key), f"Test native {key}/{job['id']}")
        spec = contract["datasets"][job["dataset"]]
        require(native_test.get("data_identity", {}).get("raw_sha256") == spec["data_sha256"],
                f"Native Test raw data SHA differs: {job['id']}")
        flatten(native_test.get("metrics"), "test", f"native/{job['id']}/test")
        require(native_test["metrics"] == receipt["test"], f"Combined Test differs from native output: {job['id']}")
        started = finite_timestamp(native_test.get("test_started_unix"), f"{job['id']}/native Test start")
        require(started == receipt["test_started_unix"] and started >= gate["created_unix"]
                and finite_timestamp(native_test.get("completed_unix"), f"{job['id']}/Test completion") >= started,
                f"Test/global Validation gate temporal order differs: {job['id']}")
        for population in ("overall", "tail"):
            get_count = lambda obj: obj["count"] if population == "overall" else obj["tail"]["count"]
            require(get_count(receipt["validation_reference"]) == get_count(job["validation_reference"])
                    == get_count(receipt["validation_replay"]), f"Validation target counts changed: {job['id']}")
        data = receipt.get("data_identity")
        require(isinstance(data, dict), f"Missing data identity: {job['id']}")
        for key in ("target_identity_sha256", "validation_target_identity_sha256", "raw_sha256",
                    "test_truth_sha256", "validation_truth_sha256"):
            digest_value(data.get(key), f"data/{key}/{job['id']}")
        require(data["raw_sha256"] == spec["data_sha256"] and receipt["test"]["count"] == spec["expected_test_count"],
                f"Frozen Test raw data SHA/count differs: {job['id']}")
        require(data["target_identity_sha256"] == native_test["target_identity_sha256"]
                and data["test_truth_sha256"] == native_test["truth_sha256"]
                and data["validation_target_identity_sha256"] == validation_rows[job["id"]]["target_identity_sha256"]
                and data["validation_truth_sha256"] == validation_rows[job["id"]]["truth_sha256"],
                f"Combined/native target or truth identity differs: {job['id']}")
        for key in ("target_identity_sha256", "validation_target_identity_sha256"):
            if key in job.get("data_identity", {}):
                require(data.get(key) == job["data_identity"][key], f"Frozen data identity differs: {job['id']}/{key}")
        baselines = receipt.get("simple_quantity_baselines")
        baseline_metrics(baselines, "test", job["id"])
        require(baselines == native_test.get("simple_quantity_baselines"),
                f"Combined/native simple baseline values differ: {job['id']}")
        for baseline in baselines.values():
            require(baseline["count"] == receipt["test"]["count"]
                    and baseline["tail"]["count"] == receipt["test"]["tail"]["count"],
                    f"Simple baseline target counts differ: {job['id']}")
        records.append({
            "id": job["id"], "study": job["study"], "dataset": job["dataset"], "arm": job["arm"],
            "seed": job["seed"], "selected_epoch": job["selected_epoch"],
            "checkpoint_sha256": job["checkpoint"]["sha256"], "frozen_source_closure": job["frozen_source_closure"],
            "validation_reference": receipt["validation_reference"], "validation_replay": receipt["validation_replay"],
            "validation_replay_check": check, "test": receipt["test"], "data_identity": data,
            "runtime": receipt.get("runtime", {}), "simple_quantity_baselines": baselines,
            "original_provenance": job.get("original_provenance", {}), "training_host": job.get("training_host"),
            "source_revision": job.get("source_revision"), "source_revision_history": job.get("source_revision_history"),
            "initial_state_sha256": job.get("initial_state_sha256"),
            "receipt_sha256": files[name], "worker_exit_sha256": files[exit_name],
            "validation_gate_sha256": gate_sha, "test_started_unix": started,
            "retrained": False, "reselected": False, "test_is_independent": False,
        })
    for study, dataset in sorted({(r["study"], r["dataset"]) for r in records}):
        group = [r for r in records if (r["study"], r["dataset"]) == (study, dataset)]
        for split in ("validation_reference", "validation_replay", "test"):
            require(len({(r[split]["count"], r[split]["tail"]["count"]) for r in group}) == 1,
                    f"Unmatched target counts: {study}/{dataset}/{split}")
        for key in ("target_identity_sha256", "validation_target_identity_sha256", "raw_sha256",
                    "test_truth_sha256", "validation_truth_sha256"):
            present = [r["data_identity"].get(key) for r in group]
            require(all(value is not None for value in present) and len(set(present)) == 1,
                    f"Unmatched population/raw identity: {study}/{dataset}/{key}")
        baseline_rows = [r["simple_quantity_baselines"] for r in group]
        require(all(row is not None for row in baseline_rows) and all(row == baseline_rows[0] for row in baseline_rows),
                f"Unmatched simple baseline population/statistics: {study}/{dataset}")
    return contract, records, files


def summarize(records):
    cells = {(r["study"], r["dataset"], r["arm"], r["seed"]): r for r in records}
    groups, pairs, directions = [], [], []
    for study, dataset in sorted({(r["study"], r["dataset"]) for r in records}):
        for arm in ARMS[study]:
            rows = [cells[(study, dataset, arm, seed)] for seed in SEEDS]
            splits = {}
            for split in ("validation_reference", "validation_replay", "test"):
                flat = [flatten(row[split], "test" if split == "test" else "validation", row["id"]) for row in rows]
                splits[split] = {key: {"mean": statistics.mean(row[key] for row in flat),
                                      "sample_sd": statistics.stdev(row[key] for row in flat),
                                      "seeds": list(SEEDS), "n": 3} for key in METRIC_KEYS}
            groups.append({"study": study, "dataset": dataset, "arm": arm, **splits})
        for control in ARMS[study][1:]:
            metric_counts = {key: {"improved": 0, "equal": 0, "worsened": 0,
                                  "validation_to_test_sign_flips": 0} for key in METRIC_KEYS}
            for seed in SEEDS:
                full, other = (cells[(study, dataset, arm, seed)] for arm in (FULL, control))
                changes = {}
                for split in ("validation_reference", "test"):
                    a = flatten(full[split], "test" if split == "test" else "validation", full["id"])
                    b = flatten(other[split], "test" if split == "test" else "validation", other["id"])
                    changes[split] = {key: a[key] - b[key] for key in METRIC_KEYS}
                for key in METRIC_KEYS:
                    delta = changes["test"][key]
                    metric_counts[key]["improved" if delta < 0 else "worsened" if delta > 0 else "equal"] += 1
                    if changes["validation_reference"][key] * delta < 0:
                        metric_counts[key]["validation_to_test_sign_flips"] += 1
                pairs.append({"study": study, "dataset": dataset, "seed": seed, "full_arm": FULL,
                              "control_arm": control, "delta_definition": "full_minus_control",
                              "validation_reference_delta": changes["validation_reference"],
                              "test_delta": changes["test"],
                              "test_overall_RMSE_strict_improvement": changes["test"]["overall_qty_rmse"] < 0,
                              "test_is_independent": False})
            directions.append({"study": study, "dataset": dataset, "full_arm": FULL,
                               "control_arm": control, "metrics": metric_counts,
                               "all_three_seeds_Test_RMSE_strict_improvement":
                               metric_counts["overall_qty_rmse"]["improved"] == 3})
    return {"seed_summary": groups, "paired_comparisons": pairs, "direction_counts": directions,
            "test_is_independent": False, "statistical_significance_established": False,
            "Test_selection_or_tuning_performed": False,
            "interpretation": "Descriptive frozen Test comparison; not a new independent confirmation or selection gate."}


def report_text(records, summary):
    lines = ["# 이력 보정27·Instacart9 동결 checkpoint Test 최종 비교", "",
             "Validation에서 선택한 checkpoint와 epoch를 고정한 36조건의 기존 Test 평가다. "
             "재학습·Test 재선택·Test 조정은 수행하지 않았으며, 과거 Test 접근 이력 때문에 새 독립 확인으로 해석하지 않는다.", "",
             "## 조건별 Test 결과", "", "|연구|데이터|방식|seed|선택 epoch|전체 RMSE|전체 MAE|전체 TimeNLL|큰 수량 RMSE|큰 수량 MAE|큰 수량 TimeNLL|",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(records, key=lambda r: (r["study"], r["dataset"], r["arm"], r["seed"])):
        flat = flatten(row["test"], "test", row["id"])
        lines.append("|" + "|".join([row["study"], row["dataset"], row["arm"], str(row["seed"]),
                                     str(row["selected_epoch"]), *(f"{flat[key]:.9g}" for key in METRIC_KEYS)]) + "|")
    lines += ["", "## 3seed 평균 ± 표본 표준편차", "",
              "|평가 split|연구|데이터|방식|전체 RMSE|전체 MAE|전체 TimeNLL|큰 수량 RMSE|큰 수량 MAE|큰 수량 TimeNLL|",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("validation_reference", "validation_replay", "test"):
        for group in summary["seed_summary"]:
            lines.append("|" + "|".join([split, group["study"], group["dataset"], group["arm"],
                *(f"{group[split][key]['mean']:.9g} ± {group[split][key]['sample_sd']:.9g}" for key in METRIC_KEYS)]) + "|")
    lines += ["", "## 같은 seed의 full − 대조: 모든 유리·불리 조건", "",
              "양수는 full의 오류 손해이며, 0은 동률이다. Validation 개발 판정과 Test 기술적 비교를 구분한다.", "",
              "|연구|데이터|seed|대조|split|전체 RMSE Δ|전체 MAE Δ|전체 TimeNLL Δ|큰 수량 RMSE Δ|큰 수량 MAE Δ|큰 수량 TimeNLL Δ|",
              "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for pair in summary["paired_comparisons"]:
        for split in ("validation_reference_delta", "test_delta"):
            lines.append("|" + "|".join([pair["study"], pair["dataset"], str(pair["seed"]), pair["control_arm"],
                        split, *(f"{pair[split][key]:+.9g}" for key in METRIC_KEYS)]) + "|")
    lines += ["", "## Test 방향과 Validation→Test 부호 반전", "",
              "|연구|데이터|대조|지표|개선|동률|손해|부호 반전|",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for direction in summary["direction_counts"]:
        for key in METRIC_KEYS:
            count = direction["metrics"][key]
            lines.append("|" + "|".join([direction["study"], direction["dataset"], direction["control_arm"], key,
                *(str(count[name]) for name in ("improved", "equal", "worsened", "validation_to_test_sign_flips"))]) + "|")
    baselines = {}
    for row in records:
        if row["simple_quantity_baselines"] is not None:
            baselines[(row["study"], row["dataset"])] = row["simple_quantity_baselines"]
    if baselines:
        lines += ["", "## 같은 Test 대상의 단순 수량 기준선", "",
                  "수량 기준선에는 TimeNLL을 정의하지 않는다.", "",
                  "|연구|데이터|기준선|전체 RMSE|전체 MAE|큰 수량 RMSE|큰 수량 MAE|",
                  "|---|---|---|---:|---:|---:|---:|"]
        for (study, dataset), values in sorted(baselines.items()):
            for name, metrics in values.items():
                lines.append("|" + "|".join([study, dataset, name, *(f"{value:.9g}" for value in (
                    metrics["qty_rmse"], metrics["qty_mae"], metrics["tail"]["qty_rmse"], metrics["tail"]["qty_mae"]))]) + "|")
    lines += ["", "## 검증 및 해석 범위", "",
              "- 계약의 36개 고유 조건, 4dataset 각9개, seed42/52/62를 중복 없이 검증했다.",
              "- 선택 epoch·checkpoint SHA·동결 source closure·계약과 실제 worker exit0를 대조했다.",
              "- 전체36개 Validation replay와 실제 종료가 모두 통과한 전역 gate의 SHA 및 생성 시각을 검증했다. 모든 Test 추론은 그 이후 시작했다.",
              "- 각 조건의 Validation 원참조·허용오차 replay와 native Test 원본을 대조했다. Test와 Validation 수치는 별도 필드에 보존했다.",
              "- Test/Validation raw SHA와 target/truth digest를 native 원본·계약에 대조하고, 데이터별 모든 arm·seed의 동일 평가 대상을 확인했다.",
              "- Test에서 epoch·구조·설정을 고르지 않았으며, 손해와 부호 반전도 제외하지 않았다.",
              "- 기존 Test 접근 이력을 유지하므로 미노출 독립 평가·통계 유의성·모든 자료의 일반화 증명이 아니다.",
              "- 물리 GPU·Runtime 차이는 조건별 runtime과 원래 학습 출처에 남는다. 추론·동시 실행 시간을 구조 효율로 해석하지 않는다.", ""]
    return "\n".join(lines)


def verify_completed(bundle):
    receipt = read(bundle / "analysis/completion_receipt.json")
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "complete"
            and receipt.get("distinct_conditions") == 36 and receipt.get("test_is_independent") is False,
            "Invalid completed analysis receipt")
    for relative, digest in {**receipt["source_files"], **receipt["analysis_files"]}.items():
        require(sha(safe(bundle, relative)) == digest, f"Completed evidence bytes changed: {relative}")
    return receipt


def analyze(bundle):
    bundle = Path(bundle).resolve()
    completion = bundle / "analysis/completion_receipt.json"
    if completion.exists():
        return verify_completed(bundle)
    analysis = bundle / "analysis"
    require(analysis.resolve().is_relative_to(bundle), "Analysis output leaves the bundle")
    require(not analysis.exists() or analysis.is_dir() and not any(analysis.iterdir()),
            "Incomplete analysis directory is preserved; no automatic overwrite/retry")
    claim = bundle / "analysis_claim.json"
    require(not claim.exists(), "Existing analysis claim is preserved; no automatic retry")
    contract, records, sources = collect(bundle)
    summary = summarize(records)
    payload = {"schema": SCHEMA, "contract_sha256": contract["contract_sha256"], "evaluation_scope": SCOPE,
               "conditions": records, **summary}
    for relative, digest in sources.items():
        require(sha(safe(bundle, relative)) == digest, f"Input changed during analysis: {relative}")
    write_new(claim, {"schema": SCHEMA, "contract_sha256": contract["contract_sha256"],
                      "created_unix": time.time(), "purpose": "exclusive immutable final Test analysis"})
    if analysis.exists():
        require(analysis.is_dir() and not any(analysis.iterdir()), "Analysis directory changed after exclusive claim")
    else:
        analysis.mkdir()
    write_new(analysis / "test_conditions.json", payload)
    report = analysis / "TEST_FINAL_COMPARISON.md"
    with report.open("x", encoding="utf-8") as stream:
        stream.write(report_text(records, summary))
    files = {f"analysis/{name}": sha(analysis / name) for name in ("test_conditions.json", "TEST_FINAL_COMPARISON.md")}
    files["analysis_claim.json"] = sha(claim)
    receipt = {"schema": SCHEMA, "status": "complete", "completed_unix": time.time(),
               "contract_sha256": contract["contract_sha256"], "distinct_conditions": 36,
               "study_counts": {"history": 27, "instacart": 9}, "evaluation_scope": SCOPE,
               "Validation_reference_and_replay_preserved": True, "per_condition_Validation_replay_passed": True,
               "global_Validation36_gate_PASS_before_any_Test_verified": True,
               "actual_worker_exit0_verified": True, "retrained": False, "reselected": False,
               "test_is_independent": False, "statistical_significance_established": False,
               "source_files": sources, "analysis_files": files}
    write_new(completion, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.bundle_root)
    print(json.dumps({key: result[key] for key in ("status", "distinct_conditions", "study_counts", "test_is_independent")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
