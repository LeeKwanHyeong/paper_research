"""Pure evidence checks for the original36 IDs after the approved seed62 handoff.

An administrative dispatcher exit1 remains exit1. It is never rewritten as a
successful original12 campaign or as a scientific worker failure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

from paper.scripts import launch_titantpp_cnn_gru_components_v5 as original
from paper.scripts import titantpp_components_parallel_monitor as previous_monitor
from paper.scripts import titantpp_components_status_transport_v2 as status_transport

PROJECT_OUTPUT = original.PROJECT / "reports/titantpp_components_seed62_handoff_20261013_v1/hourly_monitor_v2"
LEGACY_OUTPUT = PROJECT_OUTPUT.parent / "hourly_monitor"
LEGACY_SEAL_SHA = "97bf43492d069a9c41e157d1b69b1cfbb0b8182f1c49af9cd5ac073f4b2ebc09"

ADMIN_TERMINAL = "planned_queue_exclusion_terminal_verified"
TERMINAL = "training_complete_remote_verified"
NONE = "titantpp_cnn_gru54_correction_disabled"


def verify_owner_absence(snapshot, c, host, jobs):
    """Require explicit /proc probes for dispatcher, wrapper and exited workers."""
    files = original.decode_snapshot(snapshot, c, host)
    expected = {}

    def add(pid, source):
        original.require(type(pid) is int and pid > 0, "Malformed owner PID")
        expected.setdefault(str(pid), set()).add(source)

    for name in ("supervisor.claim", "phases/train_launch.json", "phases/train_exit.json"):
        value = json.loads(files[name])
        original.require(value["contract_sha256"] == original.canonical(c), "Owner probe contract mismatch")
        add(value["pid"], name)
        add(value["ppid"], name)
    for job in jobs:
        name = "worker_exits/" + job["id"] + ".json"
        value = json.loads(files[name])
        original.require(value["job"] == job and value["contract_sha256"] == original.canonical(c),
                         "Worker probe identity mismatch")
        add(value["worker_group_pid"], name)
        add(value["supervisor_pid"], name)
    original.require(snapshot.get("gpu_inventory_uuids", []).count(c["hosts"][host]["gpu_uuid"]) == 1,
                     "Fixed GPU inventory identity unresolved")
    observations = snapshot.get("process_observations", {})
    original.require(isinstance(observations, dict) and observations, "Explicit owner process probes missing")
    inspected = snapshot.get("observed_owner_pids", [])
    original.require(isinstance(inspected, list) and all(type(pid) is int and pid > 0 for pid in inspected)
                     and {int(pid) for pid in expected} <= set(inspected), "Owner PID probe coverage incomplete")
    original.require(snapshot["gpu_pids"] == [] and snapshot["processes"] == {},
                     "Owner or GPU process remains observed")
    for pid, sources in expected.items():
        observation = observations.get(pid, {})
        original.require(observation.get("pid") == int(pid) and type(observation.get("pid")) is int
                         and observation.get("state") == "absent"
                         and sources <= set(observation.get("evidence_sources", [])),
                         "Owner PID not explicitly absent: " + pid)
    return {"owner_process_absence_verified": True, "owner_process_probe_count": len(expected),
            "owner_process_probe_pids": sorted(int(pid) for pid in expected),
            "owner_process_observations_sha256": original.canonical(observations)}


def planned_parent_terminal(snapshot, c, queue_proof):
    report = previous_monitor.planned_parent_terminal(snapshot, c, queue_proof)
    retained = [j for j in c["jobs"] if j["host"] == "5090" and j["seed"] in (42, 52)]
    return {**report, **verify_owner_absence(snapshot, c, "5090", retained)}


def verify_live_probes(snapshot, c, host, report):
    """Do not promote a reused or racy PID to verified live training."""
    if report["status"] != "training_running":
        return report
    files = original.decode_snapshot(snapshot, c, host)
    state = json.loads(files["status.json"])
    active = state.get("active_jobs", {})
    original.require(isinstance(active, dict), "Live active-job shape unresolved")
    launch = json.loads(files["phases/train_launch.json"])
    pids = {state["supervisor_pid"], launch["ppid"], *snapshot["gpu_pids"],
            *(item["worker_group_pid"] for item in active.values())}
    original.require(all(type(pid) is int and pid > 0 for pid in pids), "Malformed live owner PID")
    original.require(snapshot.get("gpu_inventory_uuids", []).count(c["hosts"][host]["gpu_uuid"]) == 1,
                     "Live fixed GPU inventory identity unresolved")
    for pid in pids:
        observation = snapshot.get("process_observations", {}).get(str(pid), {})
        process = snapshot.get("processes", {}).get(str(pid), {})
        actual = observation.get("actual", {})
        original.require(observation.get("state") == "live"
                         and actual.get("identity_stable_during_query") is True
                         and process.get("observation_state") == "live"
                         and process.get("identity_stable_during_query") is True
                         and all(actual.get(key) == process.get(key) for key in
                                 ("pid", "ppid", "pgrp", "argv", "start_ticks")),
                         "Live owner PID query is reused, racy or unresolved: " + str(pid))
    return {**report, "live_process_probes_verified": True}


def verify_destination_terminal(snapshot, c, report):
    if report["status"] != TERMINAL:
        return verify_live_probes(snapshot, c, "5080", report)
    return {**report, **verify_owner_absence(snapshot, c, "5080", c["jobs"])}


def unique36_counts(c, destination, groups):
    """Count each original fit ID once across three distinct execution roots."""
    original.require(len(c["jobs"]) == len({j["id"] for j in c["jobs"]}) == 36,
                     "Original36 ID registry changed")
    moved = [j for j in c["jobs"] if j["host"] == "5090" and j["seed"] == 62]
    expected_destination = [{**j, "host": "5080"} for j in moved]
    original.require(destination["jobs"] == expected_destination,
                     "Destination must contain only the same four seed62 IDs")
    registry = {
        "original_5080": [j for j in c["jobs"] if j["host"] == "5080"],
        "parent_5090": [j for j in c["jobs"] if j["host"] == "5090" and j["seed"] != 62],
        "handoff_5080": expected_destination,
    }
    original.require(set(groups) == set(registry), "Execution root scope changed")
    seen = set()
    counts = {"total": 36, "completed": 0, "running": 0, "waiting": 0, "unconfirmed": 0}
    for name, expected in registry.items():
        report = groups[name]
        rows = report.get("jobs", [])
        if not rows and report.get("user_action_required") is True:
            counts["unconfirmed"] += len(expected)
            continue
        original.require({r["job"]["id"] for r in rows} == {j["id"] for j in expected}
                         and len(rows) == len(expected), "Root job registry is missing or duplicated")
        lookup = {j["id"]: j for j in expected}
        for row in rows:
            jid = row["job"]["id"]
            original.require(jid not in seen and row["job"] == lookup[jid],
                             "Duplicate fit ID or wrong physical owner")
            seen.add(jid)
            key = {"complete": "completed", "running": "running",
                   "running_before_first_checkpoint": "running", "waiting": "waiting"}.get(row["status"])
            counts[key or "unconfirmed"] += 1
    original.require(sum(counts[k] for k in ("completed", "running", "waiting", "unconfirmed")) == 36,
                     "Unique36 count does not close")
    counts["all_training_terminal_verified"] = (counts["completed"] == 36
        and groups["original_5080"]["status"] == TERMINAL
        and groups["parent_5090"]["status"] == ADMIN_TERMINAL
        and groups["handoff_5080"]["status"] == TERMINAL
        and groups["parent_5090"].get("owner_process_absence_verified") is True
        and groups["handoff_5080"].get("owner_process_absence_verified") is True)
    return counts


def _enrich(snapshot, report, c, host, path, calls, origin):
    from paper.scripts.titantpp_component_hourly_eta import estimate_fit_eta
    from paper.scripts.observe_titantpp_components_hourly import kst
    files = original.decode_snapshot(snapshot, c, host)
    rows = []
    for item in report["jobs"]:
        row = dict(item)
        job = row["job"]
        eta = {"status": "unconfirmed", "reason": "Not yet a verified completed epoch of this fit"}
        if row["status"] == "complete":
            eta = {"status": "complete", "conditional_early_stop_seconds": 0, "conditional_maximum_seconds": 0}
        elif (report["status"] == "training_running" and report.get("identity_verified") is True
              and row["status"] == "running" and row.get("saved_epoch")):
            folder = "run/" + job["id"] + "/runs/" + job["arm"] + "/count_only_log_regression/seed_" + str(job["seed"])
            eta = estimate_fit_eta(json.loads(files[folder + "/history.json"]),
                                   json.loads(files[folder + "/epoch_timing.json"]), row["saved_epoch"])
            if eta["status"] == "estimated":
                original.require(eta["best_epoch"] == row["best_epoch"], "ETA/history selection mismatch")
                claim = json.loads(files["claims/" + job["id"] + ".json"])
                remaining = claim["deadline_unix"] - snapshot["observed_unix"]
                eta.update(remaining_condition_seconds=remaining,
                           maximum_scenario_exceeds_deadline=eta["conditional_maximum_seconds"] > remaining,
                           conditional_early_stop_KST=kst(snapshot["observed_unix"] + eta["conditional_early_stop_seconds"]),
                           conditional_maximum_KST=kst(snapshot["observed_unix"] + eta["conditional_maximum_seconds"]))
        if report.get("identity_verified") is not True and row["status"] in ("running", "running_before_first_checkpoint"):
            row["status"] = "unconfirmed"
        row["eta"] = eta
        rows.append(row)
    return {**report, "transport_evidence_path": snapshot.get("transport_evidence_path"),
            "transport_evidence_raw_sha256": snapshot.get("transport_evidence_sha256"),
            "jobs": rows, "snapshot_path": str(path), "snapshot_raw_sha256": original.sha(path),
            "observed_KST": kst(snapshot["observed_unix"]), "observed_unix": snapshot["observed_unix"],
            "remote_calls": calls, "observation_source": origin, "whole_queue_eta": "unconfirmed",
            "local_binary_original_retrieval": "not_performed",
            "user_action_required": report.get("user_action_required", report["status"] not in (TERMINAL, ADMIN_TERMINAL, "training_running"))}


def _acquire(bundle, c, host, launcher):
    return status_transport.read_status_once(bundle, launcher, c, host, PROJECT_OUTPUT / "diagnostics")


def _error_report(host, total, path, snapshot, calls, error):
    from paper.scripts.observe_titantpp_components_hourly import kst
    result = {"host": host, "status": "query_or_evidence_unconfirmed", "jobs": [], "total": total,
              "completed": None, "remote_calls": calls, "identity_verified": False,
              "user_action_required": True, "training_failure_inferred": False,
              "reason": type(error).__name__ + ": " + str(error), "whole_queue_eta": "unconfirmed"}
    if getattr(error, "evidence_receipt_path", None):
        receipt = Path(error.evidence_receipt_path)
        result.update(transport_evidence_path=str(receipt), transport_evidence_raw_sha256=original.sha(receipt))
    if path is not None and Path(path).exists():
        result.update(snapshot_path=str(path), snapshot_raw_sha256=original.sha(path))
    if snapshot is not None and type(snapshot.get("observed_unix")) in (float, int):
        result.update(observed_unix=snapshot["observed_unix"], observed_KST=kst(snapshot["observed_unix"]))
    return result


def observe_groups(parent_bundle, destination_bundle, output):
    from paper.scripts import titantpp_components_terminal_verifier as verifier
    from paper.scripts import titantpp_components_seed62_handoff as handoff
    parent, _ = original.check_bundle(parent_bundle)
    target, _ = handoff.check_bundle(destination_bundle)
    proof_path = Path(destination_bundle) / "ownership/queue_exclusion_source.json"
    proof = original.read(proof_path)
    # Pin mismatches in existing caches fail before any live-host status call.
    target_cache = Path(output) / "terminal_cache/5080.json"
    if target_cache.exists():
        cached_target = verifier.verified_terminal(destination_bundle, output, target, "5080")
        original.require(cached_target is not None, "Destination terminal cache unresolved; SSH0")
        verify_destination_terminal(cached_target[1], target, cached_target[2])
    parent_cache = Path(output) / "terminal_cache/parent_5090.json"
    if parent_cache.exists():
        item = original.read(parent_cache)
        cached_path = Path(item["snapshot_path"])
        original.require(cached_path.is_relative_to(Path(parent_bundle) / "control/status/5090")
                         and original.sha(cached_path) == item["snapshot_raw_sha256"]
                         and original.sha(proof_path) == item["queue_proof_raw_sha256"],
                         "Administrative terminal cache pin changed; SSH0")
        cached_snapshot = original.read(cached_path)
        original.require(cached_snapshot["observed_unix"] == item["original_observed_unix"],
                         "Administrative terminal cache time changed; SSH0")
        original.require(verifier.qualification_bridge(parent_bundle, parent, "5090", cached_snapshot)
                         == item["qualification_bridge"], "Administrative terminal cache permit bridge changed; SSH0")
        planned_parent_terminal(cached_snapshot, parent, proof)
    original_output = original.PROJECT / "reports/titantpp_cnn_gru_components_20261012_v5/hourly_monitor_v2"
    cached = verifier.verified_terminal(parent_bundle, original_output, parent, "5080")
    original.require(cached is not None, "The original5080 root must have its verified terminal cache; SSH0")
    path, snapshot, report = cached
    groups = {"original_5080": _enrich(snapshot, report, parent, "5080", path, 0,
                                       "verified_terminal_cache_original_observation")}

    path = snapshot = None
    calls = 0
    try:
        cache = Path(output) / "terminal_cache/parent_5090.json"
        if cache.exists():
            item = original.read(cache)
            path = Path(item["snapshot_path"])
            original.require(path.is_relative_to(Path(parent_bundle) / "control/status/5090")
                             and original.sha(path) == item["snapshot_raw_sha256"]
                             and original.sha(proof_path) == item["queue_proof_raw_sha256"],
                             "Administrative parent cache source SHA changed; SSH0")
            snapshot = original.read(path)
            original.require(snapshot["observed_unix"] == item["original_observed_unix"],
                             "Administrative parent cache original time changed; SSH0")
            calls, origin = 0, "verified_administrative_terminal_cache_original_observation"
        else:
            calls, origin = 1, "single_read_only_remote_status_call"
            path, snapshot = _acquire(parent_bundle, parent, "5090", original)
        bridge = verifier.qualification_bridge(parent_bundle, parent, "5090", snapshot)
        files = original.decode_snapshot(snapshot, parent, "5090")
        state = json.loads(files["status.json"])
        if state["status"] == "failed":
            report = planned_parent_terminal(snapshot, parent, proof)
            if not cache.exists():
                original.write_exclusive(cache, {"snapshot_path": str(path), "snapshot_raw_sha256": original.sha(path),
                    "original_observed_unix": snapshot["observed_unix"],
                    "queue_proof_raw_sha256": original.sha(proof_path), "qualification_bridge": bridge,
                    "scope": "planned pre-seed62 administrative exit1; original12 is not complete"})
            else:
                original.require(bridge == item["qualification_bridge"], "Administrative parent cache permit bridge changed")
        else:
            report = verify_live_probes(snapshot, parent, "5090", verifier.training_status(snapshot, parent, "5090"))
            moved = {j["id"] for j in proof["transfer_jobs"]}
            original.require(not moved & set(state.get("active_jobs", {})), "Original parent spawned a transferred fit")
            report["jobs"] = [j for j in report["jobs"] if j["job"]["id"] not in moved]
            report["total"] = 8
            report["completed"] = sum(j["status"] == "complete" for j in report["jobs"])
            report["transferred_jobs"] = proof["transfer_jobs"]
        groups["parent_5090"] = _enrich(snapshot, {**report, "qualification_bridge": bridge}, parent, "5090", path, calls, origin)
    except Exception as error:
        groups["parent_5090"] = _error_report("5090", 8, path, snapshot, calls, error)

    path = snapshot = None
    calls = 0
    try:
        namespace = handoff.launcher_for(destination_bundle)
        namespace.authorization(destination_bundle, target, "5080")
        if not (Path(destination_bundle) / "control/intents/train_5080.json").exists():
            groups["handoff_5080"] = {"status": "waiting", "host": "5080", "root": target["hosts"]["5080"]["root"],
                "total": 4, "completed": 0, "remote_calls": 0, "user_action_required": False,
                "jobs": [{"job": j, "status": "waiting", "saved_epoch": None, "best_epoch": None,
                          "eta": {"status": "unconfirmed"}} for j in target["jobs"]]}
        else:
            cached = verifier.verified_terminal(destination_bundle, output, target, "5080")
            if cached:
                path, snapshot, report = cached
                report = verify_destination_terminal(snapshot, target, report)
                calls, origin = 0, "verified_terminal_cache_original_observation"
            else:
                calls, origin = 1, "single_read_only_remote_status_call"
                path, snapshot = _acquire(destination_bundle, target, "5080", namespace)
                bridge = verifier.qualification_bridge(destination_bundle, target, "5080", snapshot)
                report = {**verify_destination_terminal(snapshot, target, verifier.training_status(snapshot, target, "5080")), "qualification_bridge": bridge}
                calls, origin = 1, "single_read_only_remote_status_call"
                if report["status"] == TERMINAL:
                    verifier.seed_terminal_cache(destination_bundle, output, "5080", path, checker=handoff.check_bundle)
            groups["handoff_5080"] = _enrich(snapshot, report, target, "5080", path, calls, origin)
    except Exception as error:
        groups["handoff_5080"] = _error_report("5080", 4, path, snapshot, calls, error)
    return parent, target, groups


def verify_monitor(output, digest, parent, target):
    previous_monitor.verify_monitor(LEGACY_OUTPUT, LEGACY_SEAL_SHA, parent, target)
    original.require(original.sha(output / "monitor_integrity.json") == digest, "Pinned parallel monitor seal changed; SSH0")
    seal = original.read(output / "monitor_integrity.json")
    original.require(seal["parent_contract_sha256"] == original.canonical(parent)
                     and seal["destination_contract_sha256"] == original.canonical(target)
                     and seal["files_sha256"] == original.canonical(seal["files"]), "Parallel monitor contract/closure mismatch; SSH0")
    for name, expected in seal["files"].items():
        original.require(original.sha(original.safe(original.PROJECT, name)) == expected, "Parallel monitor code changed; SSH0")
    original.require(original.sha(output / "approval.json") == seal["approval_raw_sha256"], "Parallel monitor approval changed; SSH0")
    approval = original.read(output / "approval.json")
    original.require(approval.get("maintenance_approved") is True
                     and approval.get("owner_absence_probes_required") is True
                     and bool(approval.get("maintenance_user_instruction", "").strip())
                     and approval.get("legacy_monitor_raw_sha256") == LEGACY_SEAL_SHA,
                     "Observer repair approval scope mismatch; SSH0")
    original.require(approval["approved"] is True and approval["period_seconds"] == 3600
                     and approval["remote_read_only"] is True and approval["held_out_test_evaluated"] is False
                     and approval["parent_contract_sha256"] == original.canonical(parent)
                     and approval["destination_contract_sha256"] == original.canonical(target), "Parallel monitor approval scope mismatch; SSH0")


def main():
    import argparse
    import fcntl
    from paper.scripts import titantpp_components_seed62_handoff as handoff
    from paper.scripts.observe_titantpp_components_hourly import kst, duration
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=original.PROJECT / "search_artifacts" / handoff.PARENT_SCHEMA)
    p.add_argument("--handoff-bundle", type=Path, default=original.PROJECT / "search_artifacts" / handoff.SCHEMA)
    p.add_argument("--output", type=Path, default=PROJECT_OUTPUT)
    p.add_argument("--monitor-integrity-sha", required=True)
    args = p.parse_args()
    output = args.output.resolve()
    original.require(output == PROJECT_OUTPUT, "Unexpected parallel monitor output; SSH0")
    parent, _ = original.check_bundle(args.bundle)
    target, _ = handoff.check_bundle(args.handoff_bundle)
    verify_monitor(output, args.monitor_integrity_sha, parent, target)
    with (output / "observer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = uuid.uuid4().hex
        original.write_exclusive(output / "rounds" / (token + "_intent.json"),
            {"started_unix": time.time(), "maximum_status_calls_per_physical_host": 1,
             "original5080_cache_status_calls": 0, "remote_mutations": 0, "transport": "status_only_v2"})
        parent, target, groups = observe_groups(args.bundle, args.handoff_bundle, output)
        counts = unique36_counts(parent, target, groups)
        result = {"schema": "titantpp_components_seed62_parallel_hourly_v2", "parent_contract_sha256": original.canonical(parent),
                  "destination_contract_sha256": original.canonical(target), "report_generated_KST": kst(time.time()),
                  "groups": groups, "totals": counts, "all_training_terminal_verified": counts["all_training_terminal_verified"],
                  "monitor_integrity_raw_sha256": args.monitor_integrity_sha, "remote_mutations": 0,
                  "held_out_test_evaluated": False, "originals_retrieval": "not_performed",
                  "whole_queue_eta": "unconfirmed", "pipeline_wrapper_OS_exit_directly_observed": False}
        path = output / "rounds" / (token + ".json")
        original.write_exclusive(path, result)
        lines = ["# CNN·GRU36 병렬 관측 v2", "", f"보고 생성: {result['report_generated_KST']}. 완료{counts['completed']}·진행{counts['running']}·대기{counts['waiting']}·미확정{counts['unconfirmed']}.", "",
                 "| root·GPU | 실제 원관측 | dataset·seed·arm | 상태 | 저장/best | best 미갱신·속도 유지 | 300epoch·속도 유지 |",
                 "|---|---|---|---|---|---|---|"]
        for name, group in groups.items():
            if not group["jobs"]:
                lines.append(f"| {name}·{group['host']} | {group.get('observed_KST', '조회 미확정')} | 해당 {group['total']}개 ID | {group['status']} | 미확정/미확정 | — | — |")
            for row in group["jobs"]:
                job, eta = row["job"], row.get("eta", {})
                early = duration(eta["conditional_early_stop_seconds"]) if eta.get("status") == "estimated" else "—"
                maximum = duration(eta["conditional_maximum_seconds"]) if eta.get("status") == "estimated" else "—"
                if eta.get("maximum_scenario_exceeds_deadline"): maximum += " †현재 조건기한 초과"
                saved = str(row.get("saved_epoch")) if row.get("saved_epoch") is not None else "저장 전"
                best = str(row.get("best_epoch")) if row.get("best_epoch") is not None else "미확정"
                lines.append(f"| {name}·{job['host']} | {group.get('observed_KST', '미시작')} | {job['dataset']}·{job['seed']}·{job['arm']} | {row['status']} | {saved}/{best} | {early} | {maximum} |")
        lines += ["", "원본5080 종료 캐시는 해당 root의 과거 종료 증적이다. 새5080 handoff root와 별도 관측한다. 원5090 계획된 행정 exit1·failure·claim은 보존하고, 실제 과학 worker exit0와 구분한다.",
                  "ETA는 각 fit 자체 최근 최대10 finite 양수 epoch 중앙값이며 best·속도 변화에 따라 달라진다. 전체큐 종료시각·로컬 원본 회수·최종Validation 집계는 미확정이다. Test·새학습·자동 retry/resume·원본회수는 수행하지 않았다."]
        md = path.with_suffix(".md")
        with md.open("x") as f: f.write("\n".join(lines) + "\n")
        pointer = {"round_path": str(path), "round_raw_sha256": original.sha(path), "markdown_path": str(md), "markdown_raw_sha256": original.sha(md)}
        temporary = output / (token + ".latest.tmp")
        original.write_exclusive(temporary, pointer)
        temporary.replace(output / "latest.json")
        print(json.dumps({**pointer, "totals": counts, "groups": {n: {k: v for k, v in g.items() if k not in ("jobs", "qualification_bridge", "transferred_jobs", "gpu_pids")} for n, g in groups.items()}}, ensure_ascii=False))


if __name__ == "__main__": main()
