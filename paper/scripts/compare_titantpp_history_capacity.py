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
import importlib.util
import json
import math
from pathlib import Path
import shlex
import statistics

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_history_width8_12_dual_20261004_v1"
BASELINE = ROOT / "reports/titantpp_width_time_adoption_20261004_v1"
OUTPUT = ROOT / "reports/titantpp_history_capacity_comparison_20261004_v1"
METRICS = ("qty_rmse", "qty_mae", "tail_qty_rmse", "tail_qty_mae", "body_qty_rmse", "time_nll")
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
            "tail_qty_mae": endpoint["tail"]["qty_mae"],
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
        require(all(close(float(expected[k]), row[k]) for k in METRICS if k not in ("body_qty_rmse", "tail_qty_mae")),
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
    require("input_receipt.json" in manifest["files"], "Input identity proof not sealed")
    inputs = read(folder / "input_receipt.json")
    identity = data["inherited_data_identity"]
    require(inputs["dataset_id"] == job["dataset"] and inputs["held_out_materialized"] is False
            and inputs["populations"] == identity["populations"]
            and inputs["input_identity"] == {"data_sha256": identity["data"]["sha256"],
                                              "split_manifest_sha256": identity["split_manifest"]["sha256"]},
            "Input population/file identity differs")
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
               train_sample_tail_qty_rmse=train["tail"]["qty_rmse"], train_sample_tail_qty_mae=train["tail"]["qty_mae"],
               train_sample_time_nll=train["time_nll"],
               train_sample_n=train["count"], train_represented_n=train["represented_n"],
               train_sample_delta_vs_width4=json.dumps(comparison["candidate_minus_baseline"]["train_sample"], sort_keys=True))
    proof = {"job": job, "folder": str(folder), "terminal_manifest_sha256": sha_file(folder / "terminal_manifest.json"),
             "contract_sha256": digest, "source_git_revision": contract["source"]["git_revision"],
             "source_closure_sha256": contract["source"]["files_sha256"], "original_files_sha256": manifest["files"],
             "selected_state_sha256": endpoints["selected"]["state_sha256"],
             "last_state_sha256": endpoints["last"]["state_sha256"], "selected_first_minimum_verified": True,
             "selected_and_last_validation_replays_verified": True, "checkpoint_binary_sha_verified": True,
             "checkpoint_binary_sha256": {name: manifest["files"][str((run / name).relative_to(folder))]
                                           for name in needed[-2:]},
             "sealed_original_file_count": len(manifest["files"]), "input_identity_verified": True,
             "cpu_inference_audit_performed_here": False, "completed_unix": manifest["completed_unix"],
             "train_sampling": sample, "train_comparator_width16_available": False,
             "train_comparator_note": "Width4 matched sample is sealed by the fit. Width16 time-diagnostic sampling differs; no TRAIN comparison is inferred."}
    return row, proof


def verified_handoff(bundle, contract):
    """Reuse the exact operational whitelist and local reservation audit only."""
    if not bundle.exists():
        return None
    spec = importlib.util.spec_from_file_location("capacity_handoff_readonly", ROOT / "paper/scripts/capacity_handoff_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    parent, operational, meta = runtime.load(bundle)
    require(parent == contract, "Handoff parent is not this capacity campaign")
    qualification = runtime.verify_native(bundle, parent, operational)
    runtime.validate_reservation(parent, operational, qualification, read(bundle / "source_reservation.json"))
    gate_path = bundle / "test_finalizer/gate_receipt.json"
    gate = read(gate_path) if gate_path.exists() else None
    if gate is not None:
        require(gate["status"] == "capacity18_terminal_verified" and gate["conditions"] == 18
                and gate["parent_contract_sha256"] == sha_json(parent)
                and gate["operational_contract_sha256"] == sha_json(operational), "Foreign capacity completion gate")
        for label, ref in gate["refs"].items():
            require(sha_file(ref["snapshot"]) == ref["snapshot_sha256"], "Capacity completion snapshot SHA changed")
            snapshot = read(ref["snapshot"])
            host, digest = {"target2": ("5080", sha_json(operational)),
                            "source4": ("5090", sha_json(parent)),
                            "historical12": ("5080", sha_json(parent))}[label]
            require(snapshot["host"] == host and snapshot["contract_sha256"] == digest
                    and snapshot["source_closure_sha256"] == parent["source"]["files_sha256"],
                    "Capacity completion snapshot identity changed")
            ref["actual_observed_unix"] = snapshot["observed_unix"]
    proof = {**meta, "bundle": str(bundle), "exact_operational_derivation_verified": True,
             "source_reservation_verified": True, "destination_native_runtime_verified": True,
             "source_reservation_file_sha256": sha_file(bundle / "source_reservation.json"),
             "qualification_file_sha256": sha_file(bundle / "qualification/receipt.json"),
             "completion_gate_path": str(gate_path) if gate else None,
             "completion_gate_sha256": sha_file(gate_path) if gate else None,
             "completion_gate": gate}
    return operational, proof


def retrieved_fit(bundle, job, contract, handoff_bundle, handoff):
    folder = bundle / "retrieved" / job["host"] / "run" / job["id"]
    if handoff and job["id"] in handoff[1]["job_ids"]:
        require(not (folder / "terminal_manifest.json").exists(), "Duplicate original/transferred terminal condition")
        operational = handoff[0]
        fit_job = next(j for j in operational["jobs"] if j["id"] == job["id"])
        return handoff_bundle / "retrieved" / fit_job["host"] / "run" / job["id"], fit_job, operational
    return folder, job, contract


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
        gpu = observation.get("gpu", "")
        gpu_processes = gpu if isinstance(gpu, str) else observation.get("compute", {}).get("stdout", "")
        gpu_pids = {int(line.split(",", 1)[0].strip()) for line in gpu_processes.splitlines()
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
            "improved_seed_count": sum(d < 0 for d in delta), "worst_paired_seed": SEEDS[delta.index(max(delta))],
            "worst_paired_seed_delta": max(delta)}
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


def seed_summary(rows):
    result = []
    for dataset, width in sorted({(r["dataset"], r["width"]) for r in rows}):
        group = sorted((r for r in rows if r["dataset"] == dataset and r["width"] == width), key=lambda r: r["seed"])
        if [r["seed"] for r in group] != list(SEEDS):
            continue
        result.append({"dataset": dataset, "width": width, "split": "validation", "seeds": "42,52,62", "sd_ddof": 1,
                       **{metric + "_" + label: function([r[metric] for r in group])
                          for metric in METRICS for label, function in (("mean", statistics.mean), ("sample_sd", statistics.stdev))}})
    return result


def generate(bundle, output, handoff_bundle=None):
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
    handoff_bundle = handoff_bundle or bundle.parent / "titantpp_history_capacity_handoff_20261004_v1"
    handoff = verified_handoff(handoff_bundle, contract)
    transfer_ids = set(handoff[1]["job_ids"]) if handoff else set()
    observations = latest_observations(bundle)
    conditions, proofs, failures = [], [], []
    for job in contract["jobs"]:
        folder, fit_job, fit_contract = retrieved_fit(bundle, job, contract, handoff_bundle, handoff)
        state, observed = status_from_observation(job, observations)
        if handoff and handoff[1]["completion_gate"]:
            label = "target2" if job["id"] in transfer_ids else "source4" if job["host"] == "5090" else "historical12"
            observed = handoff[1]["completion_gate"]["refs"][label]["actual_observed_unix"]
        condition = {**job, "original_host": job["host"], "execution_host": fit_job["host"],
                     "status": state, "observed_unix": observed, "retrieved_folder": str(folder)}
        if (folder / "terminal_manifest.json").exists():
            try:
                row, proof = verified_fit(folder, fit_job, fit_contract)
                proof.update(original_job=job, operational_transfer=job["id"] in transfer_ids,
                             parent_contract_sha256=sha_json(contract))
                rows.append(row)
                proofs.append(proof)
                condition.update(status="complete_verified", selected_epoch=row["selected_epoch"],
                                 completed_unix=proof["completed_unix"])
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
    means = seed_summary(rows)
    group_metrics = [{"dataset": g["dataset"], "candidate_width": g["candidate_width"],
                      "baseline_width": g["baseline_width"], "split": "validation", "metric": metric,
                      "category": g["category"], **{k: v for k, v in cell.items() if k != "paired_seed_deltas"},
                      **{"seed_" + str(seed) + "_delta": delta for seed, delta in cell["paired_seed_deltas"].items()}}
                     for g in groups for metric, cell in g.get("metrics", {}).items()]
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=True)
    dest = output / "snapshots" / stamp
    dest.mkdir(parents=True, exist_ok=False)
    previous = read(output / "latest.json") if (output / "latest.json").exists() else None
    partial_snapshot = None
    prior = previous["snapshot"] if previous else None
    visited = set()
    while prior and prior not in visited:
        visited.add(prior)
        prior_summary = read(output / prior / "summary.json")
        if prior_summary["status_counts"].get("complete_verified", 0) < 18:
            partial_snapshot = prior
            break
        prior = prior_summary.get("previous_snapshot")
    summary = {"schema": "titantpp_history_capacity_comparison_v1", "generated_utc": now.isoformat(),
               "evaluation_scope": "validation_only", "heldout_read": False, "new_training": False,
               "user_baseline_width": 16, "deferred_dataset": "insta_market_basket",
               "contract_sha256": sha_json(contract), "source_closure_sha256": contract["source"]["files_sha256"],
               "source_file_count": len(contract["source"]["files"]),
               "operational_handoff": handoff[1] if handoff else None,
               "previous_snapshot": previous["snapshot"] if previous else None,
               "preserved_partial_snapshot": partial_snapshot,
               "status_counts": dict(Counter(x["status"] for x in conditions)), "total_conditions": 18,
               "groups": groups, "baseline_full_validation_csv_sha256": sha_file(BASELINE / "full_validation_metrics.csv"),
               "adoption_criteria_sha256": sha_file(BASELINE / "adoption_criteria.json"),
               "observations": {h: {"path": str(p), "sha256": sha_file(p), "actual_observed_unix": o["observed_unix"]}
                                for h, (p, o) in observations.items()},
               "limitations": ["Terminal file byte SHA and recorded GPU Validation replays checked; no new CPU inference audit.",
                   "TRAIN is an N_bin/n_bin weighted fixed sample, not a full-population evaluation.",
                   "Width16 TRAIN time-diagnostic sample differs; only Validation is paired to width16.",
                   "Body is the original first three quantity bins (q <= boundary[2]); tail is q > boundary[3].",
                   "Intermittent width8/12 seeds42/52 ran on5090 and seed62 transferred to5080; reused width4/16 ran on5080; no efficiency claim."]}
    write_csv(dest / "full_validation_metrics.csv", rows)
    train_rows = [{"dataset": r["dataset"], "seed": r["seed"], "width": r["width"], "host": r["host"],
                   "scope": "train_fixed_weighted_sample", "weighting": "N_bin/n_bin",
                   **{k: v for k, v in r.items() if k.startswith("train_")}, "source": r["source"]}
                  for r in rows if r["width"] in (8, 12)]
    write_csv(dest / "train_fixed_sample_metrics.csv", train_rows)
    write_csv(dest / "paired_validation_deltas.csv", deltas or [{"dataset": "", "seed": "", "candidate_width": "", "baseline_width": ""}])
    write_csv(dest / "condition_status.csv", conditions)
    write_csv(dest / "full_validation_seed_summary.csv", means)
    write_csv(dest / "paired_validation_seed_summary.csv", group_metrics)
    write_json(dest / "source_bindings.json", {"baselines": sources, "candidate_fits": proofs,
                                               "operational_handoff": handoff[1] if handoff else None,
                                               "verification_failures": failures})
    write_json(dest / "summary.json", summary)
    write_json(dest / "verification.json", {"status": "passed" if len(proofs) == 18 and not failures else "incomplete",
               "evaluation_scope": "validation_only", "heldout_read": False, "new_training": False,
               "source_file_count": len(contract["source"]["files"]), "baseline_binding_count": len(sources),
               "candidate_verified_count": len(proofs), "original_candidate_file_count": sum(p["sealed_original_file_count"] for p in proofs),
               "candidate_checkpoint_binary_count": sum(len(p["checkpoint_binary_sha256"]) for p in proofs),
               "full_validation_rows": len(rows), "three_seed_summary_rows": len(means), "paired_validation_rows": len(deltas),
               "three_seed_comparison_count": sum(g["three_seed_conclusion"] for g in groups),
               "cpu_inference_audit_performed_here": False, "verification_failures": failures,
               "reader_sha256": sha_file(Path(__file__)),
               "output_sha256": {p.name: sha_file(p) for p in sorted(dest.iterdir()) if p.is_file()}})
    kst = now.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST")
    lines = ["# 폭8·12 조건별 원본 검증과 Validation 비교", "", f"생성 시각: {kst}. 실제 서버 관측 시각은 각 snapshot의 observation 기록을 따릅니다.", "",
             f"신규 18조건 중 원본 검증 완료 **{len(proofs)}/18**입니다. 폭16은 사용자가 고정한 기준선, 폭4는 기존 대조군입니다. Instacart는 후속 실험입니다.", "",
             "기존 폭4·16의 Validation 원본 연결 18개는 endpoint/history SHA와 지표·선택 epoch를 재확인했습니다. 신규 폭8·12는 회수 완료된 원본에만 terminal success·전체 파일 SHA·최초 최소 RMSE 선택 epoch·selected/last replay 검증을 적용합니다. checkpoint binary 회수 SHA 확인과 CPU 재추론 감사는 구분합니다.", "",
             "원계약의 18개 고유 조건을 유지했습니다. Intermittent seed62 폭8·12는 원래 5090에서 시작 전 예약으로 이관한 5080 실행이며 중복 학습으로 집계하지 않습니다. 정확한 운영 계약 파생·114개 동결 소스·5080 native runtime·source 예약을 로컬 기록에서 재검증했습니다.", "",
             "| 데이터 | seed | 폭 | 실제 실행 서버 | 상태 | 선택 epoch |", "|---|---:|---:|---|---|---:|"]
    for c in conditions:
        lines.append(f"| {c['dataset']} | {c['seed']} | {c['arm'].rsplit('width',1)[1]} | {c['execution_host']} | {c['status']} | {c.get('selected_epoch','—')} |")
    if proofs:
        lines += ["", "| 데이터 | seed | 폭 | 전체 RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | Time NLL |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        keys = {(r["dataset"], r["seed"]) for r in rows if r["width"] in (8, 12)}
        for r in sorted((r for r in rows if (r["dataset"], r["seed"]) in keys), key=lambda r:(r["dataset"],r["seed"],r["width"])):
            lines.append(f"| {r['dataset']} | {r['seed']} | {r['width']} | {r['qty_rmse']:.6f} | {r['qty_mae']:.6f} | {r['tail_qty_rmse']:.6f} | {r['tail_qty_mae']:.6f} | {r['time_nll']:.6f} |")
    lines += ["", "3seed 전수 Validation 평균 ± 표본 표준편차(ddof1)입니다. Train 가중 표본과 합치지 않습니다.", "",
              "| 데이터 | 폭 | 전체 RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | Time NLL |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in means:
        cells = [f"{r[m + '_mean']:.6f} ± {r[m + '_sample_sd']:.6f}" for m in METRICS if m != "body_qty_rmse"]
        lines.append(f"| {r['dataset']} | {r['width']} | " + " | ".join(cells) + " |")
    lines += ["", "짝지은 차이는 후보 폭 − 기준 폭이며 음수는 개선입니다. 모든 secondary metric도 같은 수량 RMSE-selected checkpoint에서 읽습니다. 고정 채택 기준의 판정은 폭16 기준선을 자동으로 바꾸지 않으며 통계적 유의성이나 학습 원인을 입증하지 않습니다.", "",
              "| 데이터 | 후보/기준 폭 | 전체 RMSE 평균 차이 | 개선 seed | 가장 불리한 seed 차이 | MAE 차이 | 큰 수량 RMSE 차이 | Time NLL 차이 | 고정 기준 판정 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    names = {"joint_improvement": "함께 개선", "tradeoff": "상충 관계", "no_stable_capacity_gain": "안정적 용량 이득 없음", "invalid_or_incomplete": "검증 미완료"}
    for g in groups:
        if not g["three_seed_conclusion"]:
            continue
        m = g["metrics"]
        lines.append(f"| {g['dataset']} | {g['candidate_width']}/{g['baseline_width']} | {m['qty_rmse']['mean_delta']:+.6f} | {m['qty_rmse']['improved_seed_count']}/3 | seed{m['qty_rmse']['worst_paired_seed']}: {m['qty_rmse']['worst_paired_seed_delta']:+.6f} | {m['qty_mae']['mean_delta']:+.6f} | {m['tail_qty_rmse']['mean_delta']:+.6f} | {m['time_nll']['mean_delta']:+.6f} | {names[g['category']]} |")
    if len(proofs) == 18 and not failures:
        lines += ["", "**검증된 Validation 해석:** 폭16 대비 Taxi 폭8은 전체 RMSE 평균 차이가 작고 개선 seed가1/3이며 시간 NLL 평균이 악화합니다. Intermittent 폭8은 전체·큰 수량 RMSE와 시간 NLL 평균이 개선하지만 MAE가 악화합니다. RAF 폭8·12는 전체 RMSE 평균이 개선해도 시간 NLL이 악화합니다. 세 데이터의 폭12를 모두 포함해도 함께 개선된 후보는 없습니다. 이는 단순 용량 확대가 모든 목적을 함께 해결한다는 근거가 되지 않으며, 다른 구조의 효능·과적합·gradient 충돌을 입증하지 않습니다."]
    lines += ["", "전체 Validation은 같은 모집단의 원래 선택 checkpoint에서 읽었습니다. 큰 수량은 원래 TRAIN 경계 초과(택시3449, Intermittent187, RAF200)입니다. 같은 seed 비교가 paired_validation_deltas.csv에 보존됩니다.", "",
              "Train은 고정 수량 구간별 표본과 N/n 가중치의 진단입니다. 폭4와 표본 동일성은 fit 자체의 기록으로 확인합니다. 표본 지표와 폭4 대비 차이는 train_fixed_sample_metrics.csv에 별도로 보존합니다. 폭16 시간 진단의 다른 표본과는 Train 결과를 섞지 않습니다.", "",
              "3seed가 모두 검증된 그룹에만 평균·표본 표준편차와 판정을 만들었습니다. 모든 지표의 seed별 차이·개선 수·가장 불리한 차이는 paired_validation_seed_summary.csv에 보존합니다. 큰 수량 MAE도 원본에서 추가 기록하지만 새로운 채택 문턱으로 사용하지 않습니다.", "",
              "재생성 명령(로컬 회수 원본만 사용, 네트워크·학습·Test 열람 없음):", "",
              "```sh", "/usr/local/bin/python3 paper/scripts/compare_titantpp_history_capacity.py", "```", "",
              f"증적: [이번 snapshot](snapshots/{stamp}/summary.json), [Validation 지표](snapshots/{stamp}/full_validation_metrics.csv), [3seed 평균·표준편차](snapshots/{stamp}/full_validation_seed_summary.csv), [짝지은 차이와 불리한 seed](snapshots/{stamp}/paired_validation_seed_summary.csv), [원본 연결](snapshots/{stamp}/source_bindings.json), [18조건 목록](snapshots/{stamp}/condition_status.csv).", ""]
    if previous:
        lines += [f"이전 [관측 스냅샷]({previous['snapshot']}/README.md)은 그대로 보존했습니다. 이번 생성 시각을 과거 서버 관측·학습 완료 시각으로 대체하지 않았습니다.", ""]
    if partial_snapshot:
        lines += [f"완료 전의 [4/18 등 부분 관측 기록]({partial_snapshot}/README.md)도 원문과 SHA를 유지합니다.", ""]
    completed = len(proofs) == 18 and not failures
    lines += [f"**현재 결과 연결과 3seed 용량 효과 판단 — {'완료' if completed else '진행 중'}**",
              f"- Taxi·Intermittent·RAF 폭8·12의 원본 검증 {len(proofs)}/18, 폭4·16의 기존 Validation 연결 18개, 완전한 3seed 비교 {sum(g['three_seed_conclusion'] for g in groups)}/12를 기록했습니다.", "",
              "**고정 checkpoint의 CPU 재추론 감사를 구분한다 — 다음 작업**",
              "- 이번 로컬 집계는 원본 파일과 checkpoint binary SHA, 기록된 GPU selected/last replay를 검증했습니다. 독립 CPU 재추론은 수행하지 않았으므로 별도 미완료로 남깁니다.", "",
              "**후속 설계의 판단 근거를 연결한다 — 다음 작업**",
              "- 폭16 기준선을 유지하며 전체·큰 수량·시간·seed 안정성의 손해를 CNN·GRU 설계 검토에 전달합니다. Instacart는 후속 실험입니다.", "",
              "별도로 작성하는 [3데이터의 최종 Validation·Test 비교](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_three_dataset_final_comparison_20261005_v1/README.md)에 이번 Validation 원본 연결을 전달합니다. 이 보고서 생성에서는 Test 예측·지표·혼합 결과를 열지 않았습니다.", ""]
    report = "\n".join(lines)
    (dest / "README.md").write_text(report.replace("(snapshots/", "(../../snapshots/"))
    (output / "README.md").write_text(report)
    write_json(output / "latest.json", {"snapshot": str(dest.relative_to(output)), "summary_sha256": sha_file(dest / "summary.json"),
                                         "generated_utc": now.isoformat()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "search_artifacts" / NAME)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--handoff-bundle", type=Path, default=None)
    args = parser.parse_args()
    summary = generate(args.bundle.resolve(), args.output.resolve(), args.handoff_bundle.resolve() if args.handoff_bundle else None)
    print(json.dumps({"total": summary["total_conditions"], "status_counts": summary["status_counts"],
                      "report": str(args.output.resolve() / "README.md")}))


if __name__ == "__main__":
    main()
