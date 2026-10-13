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

ADMIN_TERMINAL = "planned_queue_exclusion_terminal_verified"
TERMINAL = "training_complete_remote_verified"
NONE = "titantpp_cnn_gru54_correction_disabled"


def planned_parent_terminal(snapshot, c, queue_proof):
    """Validate an actual stopped parent with8 scientific successes and exit1."""
    from paper.scripts.titantpp_components_terminal_verifier import verify_completed_job

    host = "5090"
    files = original.decode_snapshot(snapshot, c, host)

    def record(name):
        original.require(name in files, "Missing planned boundary evidence: " + name)
        return json.loads(files[name])

    digest = original.canonical(c)
    transferred = [j for j in c["jobs"] if j["host"] == host and j["seed"] == 62]
    retained = [j for j in c["jobs"] if j["host"] == host and j["seed"] in (42, 52)]
    original.require(len(transferred) == 4 and len(retained) == 8,
                     "Unexpected original parent partition")
    original.require(queue_proof["parent_contract_sha256"] == digest
                     and queue_proof["transfer_jobs"] == transferred,
                     "Queue exclusion original job identities changed")
    barrier_job = next(j for j in transferred if j["arm"] == NONE)
    relative = "claims/" + barrier_job["id"] + ".json"
    original.require(queue_proof["barrier_path"] == str(Path(c["hosts"][host]["root"]) / relative)
                     and relative in files
                     and hashlib.sha256(files[relative]).hexdigest() == queue_proof["barrier_raw_sha256"],
                     "Planned exclusive barrier changed")
    state, failure = record("status.json"), record("failure.json")
    original.require(state == failure and state["status"] == "failed"
                     and state["type"] == "FileExistsError"
                     and queue_proof["barrier_path"] in state["message"]
                     and state["active_batch"] == transferred
                     and state["active_jobs"] == [] and state["owned_children_stopped"] == []
                     and state["automatic_retry"] is False,
                     "Failure does not prove the planned pre-spawn queue boundary")
    original.require(set(state["completed"]) == {j["id"] for j in retained},
                     "Expected8 retained fits are not all terminal")
    original.require(snapshot["gpu_pids"] == [] and snapshot["processes"] == {},
                     "Parent owner or GPU processes remain")
    launch, exited, claim = [record(n) for n in
                            ("phases/train_launch.json", "phases/train_exit.json", "supervisor.claim")]
    expected = [c["hosts"][host]["python"], c["hosts"][host]["root"] + "/operation/campaign.py",
                "--contract", c["hosts"][host]["root"] + "/execution_contract.json",
                "--host", host, "--mode", "dispatch"]
    original.require(exited["phase"] == "train" and exited["os_exit_code"] == 1
                     and exited["contract_sha256"] == digest
                     and launch["pid"] == exited["pid"] == claim["pid"]
                     and launch["ppid"] == exited["ppid"] == claim["ppid"]
                     and launch["argv"] == exited["argv"] == claim["argv"] == expected
                     and claim["contract_sha256"] == digest,
                     "Actual administrative dispatcher exit or identity mismatch")
    phase_failure = record("phases/train_failure.json")
    original.require(phase_failure["contract_sha256"] == digest
                     and phase_failure["phase"] == "train"
                     and phase_failure["child_pid"] == exited["pid"]
                     and phase_failure["retry"] is False
                     and "phases/train_complete.json" not in files,
                     "Original failed phase was modified or completion was fabricated")
    for job in transferred:
        prefix = "run/" + job["id"] + "/"
        original.require(not any(name.startswith(prefix) for name in files)
                         and "worker_exits/" + job["id"] + ".json" not in files
                         and "leases/" + job["id"] + ".json" not in files,
                         "A transferred original fit has execution evidence")
        if job != barrier_job:
            original.require("claims/" + job["id"] + ".json" not in files,
                             "Unexpected transferred original claim")
    rows = [verify_completed_job(snapshot, c, host, job) for job in retained]
    return {"status": ADMIN_TERMINAL, "host": host, "root": snapshot["root"],
            "contract_sha256": digest, "observed_unix": snapshot["observed_unix"],
            "completed": 8, "total": 8, "jobs": rows, "transferred_jobs": transferred,
            "identity_verified": True, "user_action_required": False,
            "administrative_dispatcher_actual_exit": 1,
            "original_dispatcher_exit_unix": exited["completed_unix"],
            "parent_original12_complete": False, "scientific_failure_inferred": False,
            "pipeline_wrapper_OS_exit_directly_observed": False,
            "original_failure_and_claim_preserved": True}


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
        and groups["handoff_5080"]["status"] == TERMINAL)
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
    return {**report, "jobs": rows, "snapshot_path": str(path), "snapshot_raw_sha256": original.sha(path),
            "observed_KST": kst(snapshot["observed_unix"]), "observed_unix": snapshot["observed_unix"],
            "remote_calls": calls, "observation_source": origin, "whole_queue_eta": "unconfirmed",
            "local_binary_original_retrieval": "not_performed",
            "user_action_required": report.get("user_action_required", report["status"] not in (TERMINAL, ADMIN_TERMINAL, "training_running"))}


def _acquire(bundle, c, host, launcher):
    approval = launcher.authorization(bundle, c, host)
    token = uuid.uuid4().hex
    snapshot = launcher.one_call(bundle, launcher.SSHRemote(), host, "status",
                                {"contract": c, "contract_sha256": original.canonical(c),
                                 "host": host, "approval": approval}, label="status_" + token)
    path = Path(bundle) / "control/status" / host / (token + ".json")
    original.write_exclusive(path, snapshot)
    return path, snapshot


def _error_report(host, total, path, snapshot, calls, error):
    from paper.scripts.observe_titantpp_components_hourly import kst
    result = {"host": host, "status": "query_or_evidence_unconfirmed", "jobs": [], "total": total,
              "completed": None, "remote_calls": calls, "identity_verified": False,
              "user_action_required": True, "training_failure_inferred": False,
              "reason": type(error).__name__ + ": " + str(error), "whole_queue_eta": "unconfirmed"}
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
        verifier.verified_terminal(destination_bundle, output, target, "5080")
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
            report = verifier.training_status(snapshot, parent, "5090")
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
                calls, origin = 0, "verified_terminal_cache_original_observation"
            else:
                calls, origin = 1, "single_read_only_remote_status_call"
                path, snapshot = _acquire(destination_bundle, target, "5080", namespace)
                bridge = verifier.qualification_bridge(destination_bundle, target, "5080", snapshot)
                report = {**verifier.training_status(snapshot, target, "5080"), "qualification_bridge": bridge}
                calls, origin = 1, "single_read_only_remote_status_call"
                if report["status"] == TERMINAL:
                    verifier.seed_terminal_cache(destination_bundle, output, "5080", path, checker=handoff.check_bundle)
            groups["handoff_5080"] = _enrich(snapshot, report, target, "5080", path, calls, origin)
    except Exception as error:
        groups["handoff_5080"] = _error_report("5080", 4, path, snapshot, calls, error)
    return parent, target, groups


def verify_monitor(output, digest, parent, target):
    original.require(original.sha(output / "monitor_integrity.json") == digest, "Pinned parallel monitor seal changed; SSH0")
    seal = original.read(output / "monitor_integrity.json")
    original.require(seal["parent_contract_sha256"] == original.canonical(parent)
                     and seal["destination_contract_sha256"] == original.canonical(target)
                     and seal["files_sha256"] == original.canonical(seal["files"]), "Parallel monitor contract/closure mismatch; SSH0")
    for name, expected in seal["files"].items():
        original.require(original.sha(original.safe(original.PROJECT, name)) == expected, "Parallel monitor code changed; SSH0")
    original.require(original.sha(output / "approval.json") == seal["approval_raw_sha256"], "Parallel monitor approval changed; SSH0")
    approval = original.read(output / "approval.json")
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
    p.add_argument("--output", type=Path, default=original.PROJECT / "reports" / handoff.SCHEMA / "hourly_monitor")
    p.add_argument("--monitor-integrity-sha", required=True)
    args = p.parse_args()
    output = args.output.resolve()
    original.require(output == original.PROJECT / "reports" / handoff.SCHEMA / "hourly_monitor", "Unexpected parallel monitor output; SSH0")
    parent, _ = original.check_bundle(args.bundle)
    target, _ = handoff.check_bundle(args.handoff_bundle)
    verify_monitor(output, args.monitor_integrity_sha, parent, target)
    with (output / "observer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = uuid.uuid4().hex
        original.write_exclusive(output / "rounds" / (token + "_intent.json"),
            {"started_unix": time.time(), "maximum_status_calls_per_physical_host": 1,
             "original5080_cache_status_calls": 0, "remote_mutations": 0})
        parent, target, groups = observe_groups(args.bundle, args.handoff_bundle, output)
        counts = unique36_counts(parent, target, groups)
        result = {"schema": "titantpp_components_seed62_parallel_hourly_v1", "parent_contract_sha256": original.canonical(parent),
                  "destination_contract_sha256": original.canonical(target), "report_generated_KST": kst(time.time()),
                  "groups": groups, "totals": counts, "all_training_terminal_verified": counts["all_training_terminal_verified"],
                  "monitor_integrity_raw_sha256": args.monitor_integrity_sha, "remote_mutations": 0,
                  "held_out_test_evaluated": False, "originals_retrieval": "not_performed",
                  "whole_queue_eta": "unconfirmed", "pipeline_wrapper_OS_exit_directly_observed": False}
        path = output / "rounds" / (token + ".json")
        original.write_exclusive(path, result)
        lines = ["# CNN·GRU36 병렬 관측", "", f"보고 생성: {result['report_generated_KST']}. 완료{counts['completed']}·진행{counts['running']}·대기{counts['waiting']}·미확정{counts['unconfirmed']}.", "",
                 "| root·GPU | 실제 원관측 | dataset·seed·arm | 상태 | 저장/best | best 미갱신·속도 유지 | 300epoch·속도 유지 |",
                 "|---|---|---|---|---|---|---|"]
        for name, group in groups.items():
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
