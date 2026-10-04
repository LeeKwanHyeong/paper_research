#!/usr/bin/env python3
"""Link locally retrieved capacity fits to frozen, same-seed Validation results.

This aggregate-only reader has no network, training or prediction loader. A fit
is admitted only after its terminal manifest, original file hashes, strict first
minimum and both Validation endpoint replays have been checked. Checkpoint byte
hash verification is distinct from a subsequent CPU inference audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_history_width8_12_dual_20261004_v1"
BASELINE = ROOT / "reports/titantpp_width_time_adoption_20261004_v1"
OUTPUT = ROOT / "reports/titantpp_history_capacity_comparison_20261004_v1"
METRICS = ("qty_rmse", "qty_mae", "tail_qty_rmse", "body_qty_rmse", "time_nll")
SEEDS = (42, 52, 62)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def first_minimum(history):
    require(history and [r["epoch"] for r in history] == list(range(1, len(history) + 1)),
            "History epochs are missing or duplicated")
    values = [r["val_qty_rmse"] for r in history]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values),
            "Nonfinite Validation selection history")
    return 1 + min(range(len(values)), key=values.__getitem__)


def audit_endpoint(value, data):
    require(value.get("evaluation_scope") == "validation_only"
            and value.get("held_out_test_evaluated") is False, "Endpoint is outside Validation scope")
    require(value.get("time_metric") == "recorded_positive_integer_time_nll", "Time metric changed")
    require(value["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
            "Validation population count changed")
    require(value["quantity_boundaries"] == data["quantity_boundaries_all_train_rows"],
            "Quantity strata changed")

    def cell(v):
        require(type(v.get("count")) is int and v["count"] >= 0, "Invalid cell count")
        require(isinstance(v.get("qty_sse"), (int, float)) and math.isfinite(v["qty_sse"])
                and v["qty_sse"] >= 0, "Invalid cell SSE")
        if not v["count"]:
            require(v["qty_sse"] == 0 and all(v.get(k) is None for k in ("qty_rmse", "qty_mae", "time_nll")),
                    "Empty cell metrics must be unavailable")
            return
        require(all(isinstance(v.get(k), (int, float)) and math.isfinite(v[k])
                    for k in ("qty_rmse", "qty_mae", "time_nll")), "Nonfinite endpoint metric")
        require(v["qty_rmse"] >= 0 and v["qty_mae"] >= 0
                and close(v["qty_rmse"] ** 2 * v["count"], v["qty_sse"]), "Endpoint RMSE/SSE mismatch")

    cell(value)
    cell(value["tail"])
    cell(value["body"])
    for kind, n in (("quantity", 5), ("history", 3)):
        cells = value[kind + "_cells"]
        require(len(cells) == n and [v["bin"] for v in cells] == list(range(n)), "Invalid disjoint bins")
        for v in cells:
            cell(v)
        require(sum(v["count"] for v in cells) == value["count"], "Disjoint counts differ")
        require(close(sum(v["qty_sse"] for v in cells), value["qty_sse"]), "Disjoint SSE differs")
        for k in ("qty_mae", "time_nll"):
            require(close(sum(v[k] * v["count"] for v in cells if v["count"]), value[k] * value["count"]),
                    "Disjoint " + k + " differs")
    bins = value["quantity_cells"]
    require(value["tail"] == {k: v for k, v in bins[4].items() if k != "bin"}, "Tail is not original upper bin")
    require(value["body"]["count"] == sum(v["count"] for v in bins[:3])
            and close(value["body"]["qty_sse"], sum(v["qty_sse"] for v in bins[:3])), "Body accounting changed")


def metric_row(dataset, seed, width, epoch, endpoint, *, host, source):
    return {"dataset": dataset, "seed": seed, "width": width, "split": "validation",
            "selected_epoch": epoch, "count": endpoint["count"], "qty_rmse": endpoint["qty_rmse"],
            "qty_mae": endpoint["qty_mae"], "tail_qty_rmse": endpoint["tail"]["qty_rmse"],
            "body_qty_rmse": endpoint["body"]["qty_rmse"], "time_nll": endpoint["time_nll"],
            "tail_count": endpoint["tail"]["count"], "host": host, "source": source}


def baseline_rows(contract, baseline_dir=BASELINE):
    """Revalidate the recorded endpoint bindings; never open mixed result CSVs."""
    bindings = read(baseline_dir / "validation_source_bindings.json")
    require(bindings["scope"] == "validation_only" and bindings["heldout_read"] is False, "Foreign baseline scope")
    require(read(baseline_dir / "verification.json")["status"] == "passed", "Baseline verification missing")
    with (baseline_dir / "full_validation_metrics.csv").open(newline="") as stream:
        frozen = list(csv.DictReader(stream))
    require(len(frozen) == 18 and all(r["split"] == "validation" for r in frozen), "Baseline CSV scope changed")
    table = {(r["dataset"], int(r["seed"]), int(r["width"])): r for r in frozen}
    require(len(table) == 18, "Duplicate baseline condition")
    data_by_id = {d["dataset_id"]: d for d in contract["datasets"]}
    rows, sources = [], []
    for binding in bindings["rows"]:
        key = (binding["dataset"], binding["seed"], binding["width"])
        require(key in table and key[1] in SEEDS and key[2] in (4, 16), "Foreign baseline condition")
        ep, hp = (ROOT / binding[k] for k in ("endpoint_path", "history_path"))
        require(sha_file(ep) == binding["endpoint_sha256"] and sha_file(hp) == binding["history_sha256"],
                "Baseline source file SHA changed")
        endpoints, history = read(ep), read(hp)["history"]
        selected = endpoints["selected"]
        require(endpoints["best_epoch"] == first_minimum(history), "Baseline selected epoch changed")
        require(selected["state_sha256"] == binding["state_tensor_sha256"], "Baseline state binding differs")
        audit_endpoint(selected, data_by_id[key[0]])
        row = metric_row(*key, endpoints["best_epoch"], selected, host="5080", source=str(ep.relative_to(ROOT)))
        expected = table[key]
        require(int(expected["selected_epoch"]) == row["selected_epoch"] and int(expected["count"]) == row["count"],
                "Baseline CSV selection/population differs")
        require(all(close(float(expected[k]), row[k]) for k in METRICS if k != "body_qty_rmse"),
                "Baseline CSV/endpoint metrics differ")
        rows.append(row)
        sources.append({**binding, "endpoint_and_history_rechecked": True,
                        "checkpoint_bytes_inherited_prior_verification": True})
    require(len(rows) == len({(r["dataset"], r["seed"], r["width"]) for r in rows}) == 18,
            "Baseline source bindings are incomplete or duplicated")
    return rows, sources


def verify_manifest(folder, job, contract_sha):
    manifest = read(folder / "terminal_manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("scientific_success") is True,
            "Terminal scientific success missing")
    require(manifest.get("contract_sha256") == contract_sha and manifest.get("job") == job,
            "Foreign terminal contract/job")
    require(isinstance(manifest.get("files"), dict) and manifest["files"], "Terminal file map missing")
    for rel, digest in manifest["files"].items():
        name = Path(rel)
        require(not name.is_absolute() and ".." not in name.parts and name.parts,
                "Unsafe terminal file path")
        require(not any("test" in part.lower() or "heldout" in part.lower() or "held_out" in part.lower()
                        for part in name.parts), "Held-out file forbidden in capacity fit")
        p = folder / name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(folder.resolve()),
                "Terminal file missing or outside retrieved fit: " + rel)
        require(sha_file(p) == digest, "Terminal original file SHA mismatch: " + rel)
    return manifest


def verified_fit(folder, job, contract):
    """Return admitted aggregates and source proof; do not execute inference."""
    digest = sha_json(contract)
    manifest = verify_manifest(folder, job, digest)
    run = folder / "runs" / job["arm"] / "count_only_log_regression" / ("seed_" + str(job["seed"]))
    needed = ("endpoint_replays.json", "history.json", "selected_train_validation_diagnostic.json",
              "width4_candidate_comparison.json", "best_val_qty_rmse_model.pt", "last_epoch_state.pt")
    for name in needed:
        require(str((run / name).relative_to(folder)) in manifest["files"], "Required proof not sealed: " + name)
    endpoints, history, diagnostic, comparison = (read(run / name) for name in needed[:4])
    history = history["history"]
    require(endpoints.get("status") == "complete" and endpoints.get("job") == job
            and endpoints.get("evaluation_scope") == "validation_only"
            and endpoints.get("held_out_test_evaluated") is False, "Endpoint completion/scope differs")
    require(40 <= len(history) <= 300 and endpoints["completed_epochs"] == len(history), "Epoch budget differs")
    epoch = first_minimum(history)
    require(endpoints["best_epoch"] == epoch, "Selected checkpoint is not strict first minimum")
    data = next(d for d in contract["datasets"] if d["dataset_id"] == job["dataset"])
    for label, index in (("selected", epoch - 1), ("last", len(history) - 1)):
        endpoint = endpoints[label]
        audit_endpoint(endpoint, data)
        require(all(close(endpoint[k], history[index]["val_" + k]) for k in ("qty_rmse", "qty_mae", "time_nll")),
                label + " replay differs from same-epoch history")
        filename = "best_val_qty_rmse_model.pt" if label == "selected" else "last_epoch_state.pt"
        require(endpoint["checkpoint_path"] == contract["hosts"][job["host"]]["root"] + "/run/" + job["id"]
                + "/runs/" + job["arm"] + "/count_only_log_regression/seed_" + str(job["seed"]) + "/" + filename,
                "Replay checkpoint path differs from owned fit")
    require(diagnostic["dataset"] == job["dataset"] and diagnostic["seed"] == job["seed"]
            and diagnostic["model"] == job["arm"] and diagnostic["epoch"] == epoch,
            "Diagnostic selection/identity differs")
    provenance = diagnostic["provenance"]
    require(provenance["held_out_test_evaluated"] is False and provenance["selection_unchanged"] is True
            and provenance["new_training"] is False and provenance["state_sha256"] == endpoints["selected"]["state_sha256"],
            "Diagnostic checkpoint/scope differs")
    validation = diagnostic["validation"]
    require(validation["split"] == "validation" and validation["full_population"] is True
            and validation["population"] == data["inherited_data_identity"]["populations"]["validation"],
            "Diagnostic Validation population differs")
    require(all(close(validation[k], endpoints["selected"][k]) for k in ("qty_rmse", "qty_mae", "time_nll"))
            and close(validation["tail"]["qty_rmse"], endpoints["selected"]["tail"]["qty_rmse"]),
            "Diagnostic/endpoint metrics differ")
    train = diagnostic["train_sample"]
    require(train is not None and train["split"] == "train" and train["held_out_test_evaluated"] is False
            and train["new_training"] is False and train["state_sha256"] == provenance["state_sha256"],
            "TRAIN diagnostic missing/foreign")
    require(train["population"] == data["inherited_data_identity"]["populations"]["train"], "TRAIN population differs")
    sample = train["sampling"]
    require(all(sample[k] == v for k, v in contract["diagnostic"]["train_sampling"].items()), "TRAIN sample contract differs")
    require(sample["sample_n"] == train["count"] and close(sample["represented_n"], train["represented_n"])
            and close(sample["represented_n"], train["population"]["target_count"]), "TRAIN N/n accounting differs")
    require(comparison["population_and_sample_identity_verified"] is True and comparison["selection_unchanged"] is True
            and comparison["dataset"] == job["dataset"] and comparison["seed"] == job["seed"],
            "Paired TRAIN sample identity proof missing")
    require(comparison["candidate"]["model"] == job["arm"] and comparison["candidate"]["epoch"] == epoch
            and comparison["candidate"]["provenance"]["state_sha256"] == provenance["state_sha256"],
            "Paired candidate checkpoint differs")
    width = int(job["arm"].rsplit("width", 1)[1])
    row = metric_row(job["dataset"], job["seed"], width, epoch, endpoints["selected"],
                     host=job["host"], source=str(run / "endpoint_replays.json"))
    row.update(train_sample_qty_rmse=train["qty_rmse"], train_sample_qty_mae=train["qty_mae"],
               train_sample_tail_qty_rmse=train["tail"]["qty_rmse"], train_sample_time_nll=train["time_nll"],
               train_sample_n=train["count"], train_represented_n=train["represented_n"],
               train_sample_delta_vs_width4=json.dumps(comparison["candidate_minus_baseline"]["train_sample"], sort_keys=True))
    proof = {"job": job, "folder": str(folder), "terminal_manifest_sha256": sha_file(folder / "terminal_manifest.json"),
             "contract_sha256": digest, "source_git_revision": contract["source"]["git_revision"],
             "source_closure_sha256": contract["source"]["files_sha256"], "original_files_sha256": manifest["files"],
             "selected_state_sha256": endpoints["selected"]["state_sha256"],
             "last_state_sha256": endpoints["last"]["state_sha256"], "selected_first_minimum_verified": True,
             "selected_and_last_validation_replays_verified": True, "checkpoint_binary_sha_verified": True,
             "cpu_inference_audit_performed_here": False, "completed_unix": manifest["completed_unix"],
             "train_sampling": sample, "train_comparator_width16_available": False,
             "train_comparator_note": "Width4 matched sample is sealed by the fit. Width16 time-diagnostic sampling differs; no TRAIN comparison is inferred."}
    return row, proof


def latest_observations(bundle):
    by_host = {}
    for path in sorted((bundle / "observations").glob("*/*.json")):
        host = path.stem
        if host not in ("5080", "5090"):
            continue
        observation = read(path)
        if host not in by_host or observation["observed_unix"] > by_host[host][1]["observed_unix"]:
            by_host[host] = (path, observation)
    return by_host


def status_from_observation(job, observations):
    if job["host"] not in observations:
        return "not_observed", None
    _, observation = observations[job["host"]]
    files = observation["files"]
    fit = files.get("run/" + job["id"] + "/status.json", {})
    host = files.get("status.json", {})
    if fit.get("status") == "complete" or job["id"] in host.get("completed", {}):
        return "completed_not_retrieved", observation["observed_unix"]
    if files.get("failure.json") or host.get("status") in ("failed", "failure"):
        return "host_failure_status_unconfirmed", observation["observed_unix"]
    if fit.get("status") == "training" or host.get("active_job", {}).get("id") == job["id"]:
        processes = {}
        for line in observation.get("processes", []):
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2 and fields[0].isdigit():
                processes[int(fields[0])] = shlex.split(fields[1])
        gpu_pids = {int(line.split(",", 1)[0].strip()) for line in observation.get("gpu", "").splitlines()
                    if line.split(",", 1)[0].strip().isdigit()}
        def argument(command, name, value):
            return name in command and command.index(name) + 1 < len(command) and command[command.index(name) + 1] == value
        worker_pids = {pid for pid, command in processes.items()
                       if any(p.endswith("run_titantpp_history_capacity_campaign.py") for p in command)
                       and argument(command, "--mode", "fit") and argument(command, "--job", job["id"])}
        progress = files.get("progress.json", {})
        if progress.get("job") == job["id"] and "pid" in progress:
            worker_pids &= {progress["pid"]}
        supervisor = processes.get(host.get("supervisor_pid"), [])
        live = bool(worker_pids & gpu_pids) and argument(supervisor, "--mode", "dispatch")
        return ("training_confirmed_at_observation" if live else "training_status_unconfirmed"), observation["observed_unix"]
    return "waiting_at_last_observation", observation["observed_unix"]


def group_decision(rows, dataset, candidate_width, baseline_width):
    candidate = {r["seed"]: r for r in rows if r["dataset"] == dataset and r["width"] == candidate_width}
    baseline = {r["seed"]: r for r in rows if r["dataset"] == dataset and r["width"] == baseline_width}
    base = {"dataset": dataset, "candidate_width": candidate_width, "baseline_width": baseline_width,
            "verified_seeds": sorted(candidate), "required_seeds": list(SEEDS)}
    if set(candidate) != set(SEEDS) or set(baseline) != set(SEEDS):
        return {**base, "category": "invalid_or_incomplete", "three_seed_conclusion": False}
    require(all(all(r.get(k) is not None and math.isfinite(r[k]) for k in METRICS)
                for r in [*candidate.values(), *baseline.values()]), "Missing/nonfinite group metric")
    statistics_by_metric = {}
    for metric in METRICS:
        a, b = [baseline[s][metric] for s in SEEDS], [candidate[s][metric] for s in SEEDS]
        delta = [bb - aa for aa, bb in zip(a, b)]
        statistics_by_metric[metric] = {"baseline_mean": statistics.mean(a), "baseline_sample_sd": statistics.stdev(a),
            "candidate_mean": statistics.mean(b), "candidate_sample_sd": statistics.stdev(b),
            "mean_delta": statistics.mean(delta), "paired_seed_deltas": dict(zip(SEEDS, delta)),
            "improved_seed_count": sum(d < 0 for d in delta), "worst_paired_seed_delta": max(delta)}
    primary = statistics_by_metric["qty_rmse"]
    screen = primary["mean_delta"] < 0 and primary["improved_seed_count"] >= 2
    joint = screen and all(statistics_by_metric[k]["mean_delta"] <= 0
                           for k in ("qty_mae", "tail_qty_rmse", "time_nll"))
    tail_gain = statistics_by_metric["tail_qty_rmse"]["mean_delta"] < 0
    secondary_loss = any(statistics_by_metric[k]["mean_delta"] > 0
                         for k in ("qty_mae", "body_qty_rmse", "time_nll"))
    category = "joint_improvement" if joint else "tradeoff" if (screen or tail_gain) and secondary_loss else "no_stable_capacity_gain"
    return {**base, "category": category, "three_seed_conclusion": True, "metrics": statistics_by_metric,
            "majority_primary_screen_passed": screen, "statistical_significance_established": False}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate(bundle, output):
    contract = read(bundle / "execution_contract.json")
    pointer = read(bundle / "current.json")
    require(pointer["contract_sha256"] == sha_json(contract), "Current capacity contract changed")
    require(contract["evaluation_scope"] == "validation_only" and contract["held_out_test_evaluated"] is False,
            "Campaign is not Validation-only")
    require(len(contract["jobs"]) == 18 and len({j["id"] for j in contract["jobs"]}) == 18, "Campaign scope changed")
    require(contract["source"]["files_sha256"] == sha_json(contract["source"]["files"]), "Source closure changed")
    for rel, digest in contract["source"]["files"].items():
        require(sha_file(bundle / "frozen_source" / rel) == digest, "Frozen capacity source changed: " + rel)
    criteria = read(BASELINE / "adoption_criteria.json")
    gate = read(bundle / "launch_gate.json")
    require(sha_file(BASELINE / "adoption_criteria.json") == gate["criteria_file_sha256"], "Pre-launch adoption criteria changed")
    require(criteria["user_fixed_followup_baseline_width"] == 16 and criteria["seeds"] == list(SEEDS), "Adoption baseline/seeds changed")
    rows, sources = baseline_rows(contract)
    observations = latest_observations(bundle)
    conditions, proofs, failures = [], [], []
    for job in contract["jobs"]:
        folder = bundle / "retrieved" / job["host"] / "run" / job["id"]
        state, observed = status_from_observation(job, observations)
        condition = {**job, "status": state, "observed_unix": observed, "retrieved_folder": str(folder)}
        if (folder / "terminal_manifest.json").exists():
            try:
                row, proof = verified_fit(folder, job, contract)
                rows.append(row)
                proofs.append(proof)
                condition.update(status="complete_verified", selected_epoch=row["selected_epoch"])
            except (ValueError, KeyError, OSError, TypeError) as error:
                condition.update(status="retrieved_unverified", verification_error=str(error))
                failures.append({"job": job, "verification_error": str(error)})
        conditions.append(condition)
    baseline_table = {(r["dataset"], r["seed"], r["width"]): r for r in rows if r["width"] in (4, 16)}
    deltas = []
    for row in rows:
        if row["width"] not in (8, 12):
            continue
        for bw in (4, 16):
            base = baseline_table[row["dataset"], row["seed"], bw]
            deltas.append({"dataset": row["dataset"], "seed": row["seed"], "candidate_width": row["width"],
                "baseline_width": bw, "candidate_selected_epoch": row["selected_epoch"],
                "baseline_selected_epoch": base["selected_epoch"], "split": "validation",
                **{k + "_candidate_minus_baseline": row[k] - base[k] for k in METRICS}})
    groups = [group_decision(rows, d, w, bw) for d in criteria["datasets"] for w in (8, 12) for bw in (4, 16)]
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=True)
    dest = output / "snapshots" / stamp
    dest.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "titantpp_history_capacity_comparison_v1", "generated_utc": now.isoformat(),
               "evaluation_scope": "validation_only", "heldout_read": False, "new_training": False,
               "user_baseline_width": 16, "deferred_dataset": "insta_market_basket",
               "contract_sha256": sha_json(contract), "source_closure_sha256": contract["source"]["files_sha256"],
               "status_counts": dict(Counter(x["status"] for x in conditions)), "total_conditions": 18,
               "groups": groups, "baseline_full_validation_csv_sha256": sha_file(BASELINE / "full_validation_metrics.csv"),
               "adoption_criteria_sha256": sha_file(BASELINE / "adoption_criteria.json"),
               "observations": {h: {"path": str(p), "sha256": sha_file(p), "actual_observed_unix": o["observed_unix"]}
                                for h, (p, o) in observations.items()},
               "limitations": ["Terminal file byte SHA and recorded GPU Validation replays checked; no new CPU inference audit.",
                   "TRAIN is an N_bin/n_bin weighted fixed sample, not a full-population evaluation.",
                   "Width16 TRAIN time-diagnostic sample differs; only Validation is paired to width16.",
                   "Body is the original first three quantity bins (q <= boundary[2]); tail is q > boundary[3].",
                   "Intermittent width8/12 GPU is5090 while reused width4/16 GPU is5080; no efficiency claim."]}
    write_csv(dest / "full_validation_metrics.csv", rows)
    write_csv(dest / "paired_validation_deltas.csv", deltas or [{"dataset": "", "seed": "", "candidate_width": "", "baseline_width": ""}])
    write_csv(dest / "condition_status.csv", conditions)
    write_json(dest / "source_bindings.json", {"baselines": sources, "candidate_fits": proofs, "verification_failures": failures})
    write_json(dest / "summary.json", summary)
    kst = now.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    lines = ["# 폭8·12 조건별 원본 검증과 Validation 비교", "", f"생성 시각: {kst}. 실제 서버 관측 시각은 각 snapshot의 observation 기록을 따릅니다.", "",
             f"신규 18조건 중 원본 검증 완료 **{len(proofs)}/18**입니다. 폭16은 사용자가 고정한 기준선, 폭4는 기존 대조군입니다. Instacart는 후속 실험입니다.", "",
             "기존 폭4·16의 Validation 원본 연결 18개는 endpoint/history SHA와 지표·선택 epoch를 재확인했습니다. 신규 폭8·12는 회수 완료된 원본에만 terminal success·전체 파일 SHA·최초 최소 RMSE 선택 epoch·selected/last replay 검증을 적용합니다. checkpoint binary 회수 SHA 확인과 CPU 재추론 감사는 구분합니다.", "",
             "| 데이터 | seed | 폭 | 상태 | 선택 epoch |", "|---|---:|---:|---|---:|"]
    for c in conditions:
        lines.append(f"| {c['dataset']} | {c['seed']} | {c['arm'].rsplit('width',1)[1]} | {c['status']} | {c.get('selected_epoch','—')} |")
    if proofs:
        lines += ["", "| 데이터 | seed | 폭 | 전체 RMSE | MAE | 큰 수량 RMSE | Time NLL |", "|---|---:|---:|---:|---:|---:|---:|"]
        keys = {(r["dataset"], r["seed"]) for r in rows if r["width"] in (8, 12)}
        for r in sorted((r for r in rows if (r["dataset"], r["seed"]) in keys), key=lambda r:(r["dataset"],r["seed"],r["width"])):
            lines.append(f"| {r['dataset']} | {r['seed']} | {r['width']} | {r['qty_rmse']:.6f} | {r['qty_mae']:.6f} | {r['tail_qty_rmse']:.6f} | {r['time_nll']:.6f} |")
    lines += ["", "전체 Validation은 같은 모집단의 원래 선택 checkpoint에서 읽었습니다. 큰 수량은 원래 TRAIN 경계 초과(택시3449, Intermittent187, RAF200)입니다. 같은 seed 비교가 paired_validation_deltas.csv에 보존됩니다.", "",
              "Train은 고정 수량 구간별 표본과 N/n 가중치의 진단입니다. 폭4와 표본 동일성은 fit 자체의 기록으로 확인합니다. 폭16 시간 진단의 다른 표본과는 Train 결과를 섞지 않습니다.", "",
              "3seed 모두 검증되기 전에는 평균·표본 표준편차나 폭 채택 결론을 만들지 않습니다. 완료 그룹에는 사전에 고정한 전체 RMSE·seed 다수 개선 기준과 MAE·큰 수량·시간 손해를 함께 적용합니다.", "",
              "재생성 명령(로컬 회수 원본만 사용, 네트워크·학습·Test 열람 없음):", "",
              "```sh", "/usr/local/bin/python3 paper/scripts/compare_titantpp_history_capacity.py", "```", "",
              f"증적: [이번 snapshot](snapshots/{stamp}/summary.json), [Validation 지표](snapshots/{stamp}/full_validation_metrics.csv), [원본 연결](snapshots/{stamp}/source_bindings.json), [18조건 목록](snapshots/{stamp}/condition_status.csv).", "",
              "**현재 결과 연결 — 진행 중**", "- 회수·검증된 조건만 비교에 포함하고, 나머지 학습·미회수 상태를 남깁니다.", "",
              "**3seed 용량 효과 판단 — 다음 작업**", "- 각 데이터에서 폭8·12의 세 seed 원본 검증 후 폭4·16과 평균·표본 표준편차 및 불리한 seed를 비교합니다.", ""]
    report = "\n".join(lines)
    (dest / "README.md").write_text(report)
    (output / "README.md").write_text(report)
    write_json(output / "latest.json", {"snapshot": str(dest.relative_to(output)), "summary_sha256": sha_file(dest / "summary.json"),
                                         "generated_utc": now.isoformat()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "search_artifacts" / NAME)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = generate(args.bundle.resolve(), args.output.resolve())
    print(json.dumps({"total": summary["total_conditions"], "status_counts": summary["status_counts"],
                      "report": str(args.output.resolve() / "README.md")}))


if __name__ == "__main__":
    main()
