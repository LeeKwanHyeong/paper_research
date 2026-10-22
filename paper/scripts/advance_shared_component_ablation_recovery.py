"""Separately authorized recovery of the preserved continuation transport failure.

The frozen controller remains the only implementation of remote checks, collection
and fixed Test/cost launch.  This module owns separate local state and uses the
original advisory lock; it never starts or resumes a scientific fit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

from paper.scripts import advance_shared_component_ablation as upstream
from paper.scripts import manage_shared_component_ablation as manager

ROOT, BUNDLE = manager.ROOT, manager.BUNDLE
REPORT = manager.REPORT / "recovery_v1"
CONTROLLER = "paper/scripts/advance_shared_component_ablation_recovery.py"
TEST = "paper/tests/test_advance_shared_component_ablation_recovery.py"
UPSTREAM_CONTROLLER_SHA256 = "24cde3076a194c53864642349ce3aee66b591c4fd66b25518510be7717cbdb79"
HISTORICAL_ERROR = "RuntimeError: Transport failed; original bytes preserved; no retry"
HOSTS, PHASES = upstream.HOSTS, upstream.PHASES
read, sha, write, canonical, require = manager.read, manager.sha, manager.write, manager.canonical, manager.require


def now():
    return datetime.now(timezone.utc).isoformat()


def regular_files(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Evidence directory missing or symbolic")
    paths = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "Symbolic evidence path")
    return {str(path) for path in paths if path.is_file()}


def pinned(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve())
        and sha(path) == expected, "Bound recovery file changed: " + str(path))


def verify_preservation(binding):
    path = REPORT / "PRESERVATION.json"
    pinned(path, binding.get("preservation_sha256"))
    files = read(path).get("files")
    require(isinstance(files, dict) and bool(files), "Preservation inventory missing or empty")
    for name, item in files.items():
        target = Path(name)
        require(target.is_absolute() and isinstance(item, dict) and type(item.get("bytes")) is int
            and item["bytes"] >= 0, "Invalid preservation inventory entry")
        pinned(target, item.get("sha256"))
        require(target.stat().st_size == item["bytes"], "Preserved evidence size changed: " + name)
    old_files = regular_files(upstream.REPORT)
    new_launches = {str(upstream.REPORT / ("TEST_LAUNCHED_" + host + ".json")) for host in HOSTS}
    require(old_files - new_launches <= set(files), "Unbound original continuation evidence")
    old_collection = regular_files(BUNDLE / "retrieval/fit_validation/5090")
    require(old_collection <= set(files), "Unbound original 5090 collection evidence")
    return files


def verified_collection(c, host, phase):
    """The original archive validator plus a checked verification marker."""
    folder = BUNDLE / "retrieval" / phase / host
    if folder.exists() or folder.is_symlink():
        regular_files(folder)
    root = upstream.collected(c, host, phase)
    if root is None:
        return None
    marker, receipt = read(folder / "VERIFIED.json"), read(folder / "collection_receipt.json")
    require(marker.get("status") == "verified" and marker.get("host") == host
        and marker.get("phase") == phase and marker.get("contract_sha256") == canonical(c)
        and marker.get("manifest_sha256") == receipt.get("manifest_sha256")
        and marker.get("archive_sha256") == receipt.get("archive_sha256"), "Collection verification marker differs")
    return root


def all_collections(c):
    return {(host, phase): verified_collection(c, host, phase) for phase in PHASES for host in HOSTS}


def verify_historical_failure(binding):
    failure_path = upstream.REPORT / "FAILURE.json"
    pinned(failure_path, binding.get("historical_failure_sha256"))
    failure = read(failure_path)
    require(failure.get("status") == "failed" and failure.get("error") == HISTORICAL_ERROR
        and failure.get("automatic_retry") is False
        and failure.get("training_failure_inferred_from_transport") is False,
        "Only the explicitly approved historical transport failure may be recovered")
    folder = Path(failure["round_path"])
    require(folder.parent == upstream.REPORT / "rounds" and not folder.is_symlink(), "Historical round escaped original root")
    check = read(folder / "5080/check.json")
    require(check.get("actual_os_returncode") == 255 and check.get("timeout") is False
        and check.get("transport_error") is None and check.get("timeout_seconds") == upstream.CHECK_SECONDS
        and check.get("command", [None])[0] == "ssh" and "5080" in check["command"][:-1],
        "Historical failure is not the approved SSH255 observation")
    for stream in ("stdout", "stderr"):
        target = folder / "5080" / ("check." + stream)
        require(sha(target) == check.get(stream + "_sha256")
            and target.stat().st_size == check.get(stream + "_bytes"), "Historical transport bytes differ")
    latest = read(upstream.REPORT / "LATEST.json")
    require(latest.get("round_path") == str(folder / "ROUND.json") and latest.get("status") == "failed"
        and sha(folder / "ROUND.json") == latest.get("round_sha256"), "Historical failure pointer differs")
    return failure


def verify(binding_sha, authorization_sha):
    binding_path, auth_path = REPORT / "CONTROL_BINDINGS.json", REPORT / "AUTHORIZATION.json"
    pinned(binding_path, binding_sha); binding = read(binding_path)
    require(binding.get("campaign") == manager.NAME and binding.get("report_root") == str(REPORT)
        and binding.get("bundle_root") == str(BUNDLE), "Recovery path identity differs")
    pinned(auth_path, authorization_sha)
    require(binding.get("authorization_sha256") == authorization_sha, "Recovery authorization SHA differs")
    auth = read(auth_path)
    flags = ("recovery_followup_only", "initial_fit_adoption_only", "short_terminal_checks", "collect_once",
        "global_validation_then_test_cost", "local_comparison", "preserve_existing", "mac_deferred",
        "no_scientific_retry", "allow_new_test_launch_receipts_in_original_continuation", "no_new_training", "no_mac_execution")
    require(auth.get("status") == "approved" and auth.get("campaign") == manager.NAME
        and all(auth.get(key) is True for key in flags), "Recovery scope is not approved")
    instruction = auth.get("user_instruction")
    require(isinstance(instruction, str) and bool(instruction.strip())
        and hashlib.sha256(instruction.encode("utf-8")).hexdigest() == binding.get("user_instruction_sha256"),
        "Exact recovery instruction is not bound")
    require(auth.get("maximum_check_seconds") == upstream.CHECK_SECONDS
        and auth.get("minimum_check_interval_seconds") == upstream.INTERVAL_SECONDS
        and auth.get("transfer_timeout_seconds") == upstream.TRANSFER_SECONDS
        and type(auth.get("maximum_recovery_rounds")) is int and 0 < auth["maximum_recovery_rounds"] <= 96,
        "Recovery bounded execution policy differs")
    for key in ("upstream_binding_sha256", "upstream_authorization_sha256", "historical_failure_sha256"):
        require(binding.get(key) == auth.get(key) and bool(binding.get(key)), "Recovery approval binding differs: " + key)
    old_binding, c, initial = upstream.verify(binding["upstream_binding_sha256"], binding["upstream_authorization_sha256"])
    require(old_binding["files"].get(upstream.CONTROLLER) == UPSTREAM_CONTROLLER_SHA256,
        "Original approved controller revision differs")
    deadline = auth.get("absolute_recovery_deadline_unix")
    require(type(deadline) in (int, float) and math.isfinite(deadline)
        and deadline <= min(c["hosts"][host]["absolute_host_deadline_unix"] for host in HOSTS),
        "Recovery deadline exceeds the original host authorization")
    files = binding.get("files")
    require(isinstance(files, dict) and {CONTROLLER, TEST} <= set(files), "Recovery source and test bindings missing")
    for name, digest in files.items():
        pinned(Path(name) if Path(name).is_absolute() else ROOT / name, digest)
    qa_path = REPORT / "INDEPENDENT_QA.json"; pinned(qa_path, binding.get("recovery_qa_sha256"))
    qa = read(qa_path)
    require(qa.get("status") == "PASS" and qa.get("independent_reviewer") is True
        and isinstance(qa.get("reviewer"), str) and bool(qa["reviewer"].strip())
        and all(qa.get("files_reviewed_sha256", {}).get(name) == files[name] for name in (CONTROLLER, TEST)),
        "Recovery requires its own source-bound independent review")
    verify_preservation(binding); verify_historical_failure(binding)
    require(verified_collection(c, "5090", "fit_validation") is not None,
        "Previously verified 5090 fit collection is required before recovery")
    return binding, c, initial


def latest_round():
    path = REPORT / "LATEST.json"
    if not path.exists():
        return None, None
    latest = read(path); target = Path(latest.get("round_path", ""))
    require(target.name == "ROUND.json" and target.parent.parent == REPORT / "rounds"
        and not target.is_symlink() and sha(target) == latest.get("round_sha256"), "Recovery latest round changed")
    result, intent = read(target), read(target.parent / "INTENT.json")
    require(result.get("status") == latest.get("status")
        and latest.get("check_started_unix") == intent.get("check_started_unix"), "Recovery latest identity differs")
    return latest, result


def transport_verified(folder, stem):
    item = read(folder / (stem + ".json"))
    require(item.get("actual_os_returncode") == 0 and item.get("timeout") is False
        and item.get("transport_error") is None, "Existing action transport failed; no automatic retry")
    for stream in ("stdout", "stderr"):
        require(sha(folder / (stem + "." + stream)) == item.get(stream + "_sha256"), "Existing action transport bytes changed")
    return upstream.json_response((folder / (stem + ".stdout")).read_bytes())


def validate_rounds_and_launches(binding, c, collections, *, active_folder=None):
    historical = Path(read(upstream.REPORT / "FAILURE.json")["round_path"])
    for base in (upstream.REPORT, REPORT):
        for intent in (base / "rounds").glob("*/INTENT.json"):
            if base == REPORT and intent.parent == active_folder:
                continue
            require((intent.parent / "ROUND.json").is_file(), "Interrupted prior round preserved; manual reconciliation required")
            result = read(intent.parent / "ROUND.json")
            if base == upstream.REPORT:
                require(result.get("status") != "failed" or intent.parent == historical,
                    "Another original failed round is not authorized for recovery")
            else:
                require(read(intent).get("binding_sha256") == sha(REPORT / "CONTROL_BINDINGS.json"),
                    "Prior recovery round binding differs")
                require(result.get("status") in ("waiting", "complete", "failed"), "Unknown recovery round status")
    latest, _ = latest_round()
    rounds = [path for path in (REPORT / "rounds").glob("*/INTENT.json") if path.parent != active_folder]
    if rounds:
        newest = max(rounds, key=lambda p: read(p)["check_started_unix"])
        require(latest is not None and latest["round_path"] == str(newest.parent / "ROUND.json"),
            "Interrupted recovery publication preserved; no retry")
    for path in (upstream.REPORT / "rounds").glob("*/TEST_LAUNCH_ATTEMPT_*.json"):
        require(False, "Original Test launch attempt is outside this recovery")
    launches = {host: read(upstream.REPORT / ("TEST_LAUNCHED_" + host + ".json")) for host in HOSTS
        if (upstream.REPORT / ("TEST_LAUNCHED_" + host + ".json")).exists()}
    attempts = {host: list((REPORT / "rounds").glob("*/TEST_LAUNCH_ATTEMPT_" + host + ".json")) for host in HOSTS}
    require(len(launches) in (0, len(HOSTS)), "Partial Test launch preserved; no automatic completion or retry")
    for host in HOSTS:
        require(len(attempts[host]) == (1 if host in launches else 0), "Existing Test launch attempt is unresolved or repeated")
        if host not in launches:
            require(collections[(host, "test_cost")] is None, "Test collection without admitted launch")
            continue
        launch, folder = launches[host], attempts[host][0].parent
        require((folder / "INTENT.json").is_file() and (folder / "ROUND.json").is_file(),
            "Orphan Test launch attempt preserved; no automatic retry")
        intent, transition = read(folder / "INTENT.json"), read(folder / "ROUND.json")
        require(intent.get("binding_sha256") == sha(REPORT / "CONTROL_BINDINGS.json")
            and intent.get("contract_sha256") == canonical(c)
            and transition.get("status") == "waiting" and transition.get("phase") == "test_cost"
            and transition.get("transition") == "global_Validation36_passed_Test_cost_launched"
            and transition.get("hosts", {}).get(host) == launch,
            "Test launch attempt has no bound, completed recovery transition")
        require(launch == transport_verified(folder, "test_launch_" + host), "Test launch receipt differs from actual transport")
        require(launch.get("status") == "launched" and launch.get("phase") == "test_cost"
            and launch.get("host") == host and launch.get("contract_sha256") == canonical(c)
            and upstream.valid_identity(launch.get("identity")) and upstream.valid_identity(launch.get("supervisor_identity"))
            and launch["identity"]["cwd"] == c["hosts"][host]["source_root"]
            and launch["supervisor_identity"]["cwd"] == c["hosts"][host]["source_root"], "Admitted Test launch identity differs")
        attempt = read(attempts[host][0])
        gate_sha = sha(BUNDLE / "global_validation_v1/global_validation_gate.json")
        require(attempt.get("host") == host and attempt.get("contract_sha256") == canonical(c)
            and attempt.get("global_gate_sha256") == gate_sha == launch.get("global_validation_gate_sha256"),
            "Test launch global gate binding differs")
    if launches:
        require(len({attempts[host][0].parent for host in HOSTS}) == 1,
            "Test launches belong to different recovery transitions")
    if (BUNDLE / "global_validation_v1").exists():
        manager.verify_global_gate(c, BUNDLE / "global_validation_v1")
        require(bool(launches), "Existing gate without completed launch transition; preserve")
    if (BUNDLE / "comparison_v1").exists():
        require((REPORT / "COMPLETE.json").is_file(), "Existing or partial comparison has no recovery completion receipt")
    return launches


def check_budget(c):
    auth = read(REPORT / "AUTHORIZATION.json")
    require(time.time() < auth["absolute_recovery_deadline_unix"]
        and all(time.time() < c["hosts"][host]["absolute_host_deadline_unix"] for host in HOSTS),
        "Approved recovery/host deadline expired; no further action")
    require(len(list((REPORT / "rounds").glob("*/INTENT.json"))) <= auth["maximum_recovery_rounds"],
        "Approved recovery round limit exceeded")


def boundary(binding_sha, authorization_sha):
    binding, c, _ = verify(binding_sha, authorization_sha)
    check_budget(c)
    all_collections(c)
    return binding, c


def comparison_outputs_exist():
    output = BUNDLE / "comparison_v1"
    require(all((output / name).is_file() and not (output / name).is_symlink()
        and (output / name).stat().st_size > 0 for name in ("REPORT.json", "REPORT.md", "TABLE.csv")),
        "Comparison outputs missing or empty")


def collection_marker(c, binding_sha, *, create=False):
    path = REPORT / "COLLECTION_COMPLETE.json"
    expected = {"status": "collection_complete", "binding_sha256": binding_sha,
        "contract_sha256": canonical(c), "collections": {phase: {host: {
            "receipt_sha256": sha(BUNDLE / "retrieval" / phase / host / "collection_receipt.json"),
            "verified_sha256": sha(BUNDLE / "retrieval" / phase / host / "VERIFIED.json")}
            for host in HOSTS} for phase in PHASES}}
    if create and not path.exists():
        write(path, {**expected, "at": now()})
    require(path.is_file() and not path.is_symlink(), "Collection completion marker missing or symbolic")
    actual = read(path)
    require(all(actual.get(key) == value for key, value in expected.items()), "Collection completion marker changed")
    return sha(path)


def verify_completion(binding, c, binding_sha, folder=None):
    complete = read(REPORT / "COMPLETE.json")
    comparison_outputs_exist()
    require(complete.get("status") == "complete" and complete.get("binding_sha256") == binding_sha
        and complete.get("contract_sha256") == canonical(c)
        and bool(complete.get("files")) and complete["files"] == upstream.completion_inventory(),
        "Cached recovery completion evidence changed or is incomplete")
    collections = all_collections(c)
    require(all(collections.values()), "All four verified collections are required")
    require(complete.get("collection_complete_sha256") == collection_marker(c, binding_sha),
        "Cached collection completion marker changed")
    validate_rounds_and_launches(binding, c, collections, active_folder=folder)
    _, latest = latest_round()
    require(latest is not None and latest.get("complete_sha256") == sha(REPORT / "COMPLETE.json"),
        "Cached completion receipt changed")
    manager.verify_global_gate(c, BUNDLE / "global_validation_v1")
    if folder is not None:
        require(upstream.compare(read(upstream.REPORT / "CONTROL_BINDINGS.json"), c, folder, cached=True)["status"] == "complete",
            "Cached comparison did not verify")
    else:
        from paper.scripts import compare_shared_component_ablation as comparator
        output = BUNDLE / "comparison_v1"; result = comparator.aggregate(c, BUNDLE / "retrieval")
        require(read(output / "REPORT.json") == result
            and (output / "REPORT.md").read_text(encoding="utf-8") == comparator.render_markdown(result)
            and (output / "TABLE.csv").read_text(encoding="utf-8") == comparator.render_csv(result),
            "Cached comparison differs from pure metric recomputation")
    return {"status": "complete", "cache_verified": True, "remote_calls": 0, "output": str(BUNDLE / "comparison_v1")}


def cached_failure():
    path = REPORT / "FAILURE.json"
    if not path.exists():
        return None
    failure = read(path); seal = read(REPORT / "FAILURE_SEAL.json")
    require(seal.get("failure_sha256") == sha(path) and failure.get("status") == "failed"
        and failure.get("automatic_retry") is False and failure.get("training_failure_inferred_from_transport") is False,
        "Cached recovery failure changed")
    return {**failure, "cached_failure": True, "remote_calls": 0}


def publish_round(folder, value):
    value = {**value, "round_path": str(folder), "observed_at": now()}
    if (REPORT / "COMPLETE.json").exists():
        value["complete_sha256"] = sha(REPORT / "COMPLETE.json")
    write(folder / "ROUND.json", value)
    pointer = REPORT / (".LATEST." + uuid.uuid4().hex + ".json")
    write(pointer, {"round_path": str(folder / "ROUND.json"), "round_sha256": sha(folder / "ROUND.json"),
        "status": value["status"], "check_started_unix": read(folder / "INTENT.json")["check_started_unix"]})
    os.replace(pointer, REPORT / "LATEST.json")
    return value


def preserve_failure(error, folder=None, *, context=None):
    value = {"status": "failed", "error": type(error).__name__ + ": " + str(error), "at": now(),
        "round_path": str(folder) if folder else None, "automatic_retry": False, "further_remote_calls": 0,
        "training_failure_inferred_from_transport": False, **(context or {})}
    write(REPORT / "FAILURE.json", value)
    write(REPORT / "FAILURE_SEAL.json", {"failure_sha256": sha(REPORT / "FAILURE.json")})
    return value


def check_only(binding_sha, authorization_sha):
    binding, c, _ = verify(binding_sha, authorization_sha)
    with upstream.advance_lock() as acquired:
        if not acquired:
            return {"status": "busy", "remote_calls": 0}
        failure = cached_failure()
        if failure:
            return failure
        collections = all_collections(c)
        validate_rounds_and_launches(binding, c, collections)
        if (REPORT / "COMPLETE.json").exists():
            return verify_completion(binding, c, binding_sha)
        check_budget(c)
        require(len(list((REPORT / "rounds").glob("*/INTENT.json"))) < read(REPORT / "AUTHORIZATION.json")["maximum_recovery_rounds"],
            "Approved recovery round limit reached")
        return {"status": "ready_to_advance", "remote_calls": 0,
            "collected_fit_hosts": [host for host in HOSTS if collections[(host, "fit_validation")]],
            "collected_test_cost_hosts": [host for host in HOSTS if collections[(host, "test_cost")]]}


def advance(binding_sha, authorization_sha):
    binding, c, initial = verify(binding_sha, authorization_sha)
    with upstream.advance_lock() as acquired:
        if not acquired:
            return {"status": "busy", "remote_calls": 0}
        failure = cached_failure()
        if failure:
            return failure
        folder = None
        context = {"failure_phase": "local_admission", "collection_complete": False}
        try:
            collections = all_collections(c)
            launches = validate_rounds_and_launches(binding, c, collections)
            latest, _ = latest_round()
            complete = (REPORT / "COMPLETE.json").exists()
            if not complete:
                check_budget(c)
                auth = read(REPORT / "AUTHORIZATION.json")
                require(len(list((REPORT / "rounds").glob("*/INTENT.json"))) < auth["maximum_recovery_rounds"],
                    "Approved recovery round limit reached")
                if latest and time.time() - latest["check_started_unix"] < upstream.INTERVAL_SECONDS:
                    return {"status": "waiting_next_check", "remote_calls": 0,
                        "next_check_unix": latest["check_started_unix"] + upstream.INTERVAL_SECONDS}
            folder = REPORT / "rounds" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + uuid.uuid4().hex[:12])
            write(folder / "INTENT.json", {"check_started_unix": time.time(), "binding_sha256": binding_sha,
                "contract_sha256": canonical(c), "at": now()})
            if complete:
                # Cache validation is read-only locally apart from its own audit round,
                # and remains available after the remote execution deadline.
                context = {"failure_phase": "completion_cache", "collection_complete": True}
                result = verify_completion(binding, c, binding_sha, folder)
                verify(binding_sha, authorization_sha)
                return publish_round(folder, result)
            statuses = {}
            for host in HOSTS:
                launch = launches.get(host, initial["hosts"][host]); phase = launch["phase"]
                if collections[(host, phase)] is not None:
                    continue
                context = {"failure_phase": "terminal_check", "failure_host": host, "native_phase": phase,
                    "collection_complete": False}
                boundary(binding_sha, authorization_sha)
                value = upstream.short_check(c, host, phase, launch, folder / host)
                boundary(binding_sha, authorization_sha)
                statuses[host] = value
                if value["status"] == "terminal":
                    context["failure_phase"] = "collection"
                    boundary(binding_sha, authorization_sha)
                    upstream.collect(c, host, phase)
                    boundary(binding_sha, authorization_sha)
                    collections[(host, phase)] = verified_collection(c, host, phase)
            if all(collections[(host, "fit_validation")] for host in HOSTS) and not launches:
                context = {"failure_phase": "global_validation_gate", "collection_complete": False}
                boundary(binding_sha, authorization_sha)
                gate = upstream.build_global_gate(c, {host: collections[(host, "fit_validation")] for host in HOSTS},
                    BUNDLE / "global_validation_v1")
                boundary(binding_sha, authorization_sha)
                for host in HOSTS:
                    context = {"failure_phase": "test_launch", "failure_host": host, "collection_complete": False}
                    boundary(binding_sha, authorization_sha)
                    launches[host] = upstream.launch_test(c, host, gate, folder)
                    boundary(binding_sha, authorization_sha)
                return publish_round(folder, {"status": "waiting", "phase": "test_cost",
                    "transition": "global_Validation36_passed_Test_cost_launched", "hosts": launches})
            if all(collections[(host, "test_cost")] for host in HOSTS):
                context = {"failure_phase": "local_comparison", "collection_complete": True}
                boundary(binding_sha, authorization_sha)
                marker_sha = collection_marker(c, binding_sha, create=True)
                old_binding = read(upstream.REPORT / "CONTROL_BINDINGS.json")
                require(upstream.comparator_qualified(old_binding), "Independent comparator review missing")
                result = upstream.compare(old_binding, c, folder)
                boundary(binding_sha, authorization_sha)
                require(result["status"] == "complete", "Comparison did not complete")
                comparison_outputs_exist()
                inventory = upstream.completion_inventory()
                require(bool(inventory), "Empty completion evidence inventory")
                write(REPORT / "COMPLETE.json", {**result, "binding_sha256": binding_sha,
                    "contract_sha256": canonical(c), "collection_complete_sha256": marker_sha,
                    "files": inventory, "at": now()})
                verify(binding_sha, authorization_sha)
                return publish_round(folder, result)
            verify(binding_sha, authorization_sha)
            return publish_round(folder, {"status": "waiting", "hosts": statuses,
                "collected_fit_hosts": [host for host in HOSTS if collections[(host, "fit_validation")]],
                "collected_test_cost_hosts": [host for host in HOSTS if collections[(host, "test_cost")]]})
        except BaseException as error:
            try:
                verify(binding_sha, authorization_sha)
            except BaseException as changed:
                context["post_failure_integrity_error"] = type(changed).__name__ + ": " + str(changed)
            value = preserve_failure(error, folder, context=context)
            return publish_round(folder, value) if folder is not None else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check-only", "advance"), default="check-only")
    parser.add_argument("--binding-sha", required=True); parser.add_argument("--authorization-sha", required=True)
    args = parser.parse_args(argv)
    result = (advance if args.mode == "advance" else check_only)(args.binding_sha, args.authorization_sha)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
