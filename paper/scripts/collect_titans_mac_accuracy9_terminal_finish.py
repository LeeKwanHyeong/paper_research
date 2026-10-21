"""Test-only MAC terminal recovery: bounded check, one immutable retrieval and local comparison.

The failed v4 dispatcher and all original scientific bytes stay preserved. This
collector admits the separately sealed v5 controller without rewriting v4's
failure into completion. It never starts, retries, resumes or signals GPU work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import traceback
import uuid

from paper.scripts import collect_titans_mac_accuracy9_recovery as legacy
from paper.scripts import compare_titans_mac_accuracy9 as comparison

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/titans_mac_accuracy9_recovery_20261021_v1/execution_v5"
BUNDLE = ROOT / "search_artifacts/titans_mac_accuracy9_recovery_20261021_v5"
UPSTREAM_BUNDLE = ROOT / "search_artifacts/titans_mac_accuracy9_recovery_20261021_v4"
FINISH_SCHEMA = "titans_mac_accuracy9_terminal_finish_v1"
SCHEMA = "titans_mac_accuracy9_terminal_collection_v1"
COMPARISON_SCHEMA = "titans_mac_terminal_shared_three_seed_comparison_v1"
UPSTREAM_SHA = "191d2534245ab6c02efff817fe9e21bb43e6589415b1839e22f1a7dec40797c2"
CHECK_SECONDS, TRANSFER_SECONDS, MINIMUM_CHECK_INTERVAL_SECONDS = 45, 900, 900
IDENTITY_KEYS = legacy.IDENTITY_KEYS
OPERATION_NAMES = {"source/titans_mac_accuracy9_terminal_runtime.py", "source/titans_mac_accuracy9_terminal_identity.py"}
TEST_NAMES = {"test_claim.json", "test_launch.json", "test_lease.json", "test_exit.json", "test.intent.json",
              "test_receipt.json", "test_recovery_receipt.json", "test_predictions.jsonl.gz"}
LIFECYCLE_FILES = {"artifact_manifest.json", "final_receipt.json", "dispatcher.stdout.log",
                   "dispatcher.stderr.log", "dispatcher_exit.json", "supervisor.stdout.log", "supervisor.stderr.log"}
require, read, sha, canonical, write = legacy.require, legacy.read, legacy.sha, legacy.canonical, legacy.write
local, mapped, inventory, relative = legacy.local, legacy.mapped, legacy.inventory, legacy.relative
_same_metrics, _proof = legacy._same_metrics, legacy._proof
verify_continuation_import = legacy.verify_continuation_import


def report_dir():
    return REPORT / "collection_v1"


def retrieval_dir():
    return BUNDLE / "collection_v1/retrieval_v1"


def verify_finish_scope(f, c, finish):
    require(f.get("schema") == FINISH_SCHEMA and f.get("upstream_contract_sha256") == canonical(c) == UPSTREAM_SHA
            and f.get("upstream_root") == c["root"] and f.get("upstream_contract_path") == c["root"] + "/recovery_contract.json"
            and f["root"] != c["root"] and f["boot_id"] == c["boot_id"] and f["gpu_uuid"] == c["gpu_uuid"]
            and f["python"] == c["python"] and f["limits"] == c["limits"]
            and f["absolute_host_deadline_unix"] == c["absolute_host_deadline_unix"],
            "Finish changed the frozen scientific scope or resource budget")
    require(canonical(read(local(finish, "finish_contract.json"))) == canonical(f), "Archived finish contract changed")
    root = finish.parent / "recovery"
    require(sha(local(root, "recovery_contract.json")) == f["upstream_contract_raw_sha256"],
            "Pinned upstream contract bytes changed")
    pins = f["upstream_evidence_files"]
    require(isinstance(pins, dict) and {"final_receipt.json", "dispatch_failure.json", "dispatcher_exit.json",
            "dispatcher_launch.json", "validation_stage_receipt.json"} <= set(pins), "Empty raw upstream evidence bindings")
    inventory(root, pins)
    operations = f["operation_files"]
    require(isinstance(operations, dict) and set(operations) == OPERATION_NAMES, "Wrong terminal operation closure")
    for name, digest in operations.items():
        require(relative(name).parts[0] == "source" and len(relative(name).parts) == 2 and sha(local(finish, name)) == digest, "Terminal operation bytes changed")

    authorization = f["authorization"]
    auth_path = mapped(finish, f["root"], authorization["path"])
    require(authorization.get("instruction") and sha(auth_path) == authorization["sha256"], "Unbound terminal authorization")


def verify_admission(f, c, root, finish, final):
    binding = final["admission"]
    require(binding["path"] == "admission.json" and sha(local(finish, binding["path"])) == binding["sha256"],
            "Admission binding changed")
    value = read(local(finish, binding["path"]))
    require(value.get("schema") == FINISH_SCHEMA and value.get("status") == "passed"
            and value.get("contract_sha256") == canonical(f) and value.get("upstream_contract_sha256") == canonical(c),
            "Missing terminal admission")
    before = value["preexisting_v4_files"]
    require(isinstance(before, dict) and bool(before) and canonical(before) == value["preexisting_v4_files_sha256"]
            and {"final_receipt.json", "dispatcher_exit.json", "dispatch_failure.json", "validation_stage_receipt.json"} <= set(before),
            "Empty or incomplete v4 preservation snapshot")
    inventory(root, before)
    additions = {"validation_gate.json"} | {"run/" + key + "/" + name for key in c["test_jobs"] for name in TEST_NAMES}
    require(not set(before).intersection(additions) and set(final["added_v4_files"]) == additions,
            "Existing or unapproved gate/Test artifact; fixed Test is not retryable")
    old_final = read(local(root, "final_receipt.json"))
    old_stage = read(local(root, "validation_stage_receipt.json"))
    failure = read(local(root, "dispatch_failure.json"))
    require(old_final.get("status") == "failed" and old_final.get("candidate_test_count") == 0
            and old_final.get("held_out_test_evaluated") is False
            and set(old_final["stages"]) == {"reference", "train", "validation"}
            and old_final["stages"]["validation"] == old_stage and old_stage.get("status") == "failed" and old_stage.get("pending_job_ids") == []
            and set(old_stage.get("exits", {})) == set(old_stage.get("receipts", {})) == set(c["validation_jobs"])
            and failure.get("stage") == "validation" and failure.get("traceback", "").rstrip().endswith(
                "ValueError: Child identity disappeared without observable OS exit")
            and "in initial_identity" in failure["traceback"] and "in _identity" in failure["traceback"]
            and old_stage.get("failure", {}).get("traceback") == failure["traceback"],
            "Only the fully successful replay with the exact historical terminal poll race may finish")
    for name in ("reference", "train"):
        require(old_final["stages"][name]["status"] == "complete"
                and value["validated_stages"][name] == final["stages"][name] == old_final["stages"][name],
                "Already completed fit stage changed")
    admitted = {**old_stage, "status": "complete", "failure": None, "imported_without_execution": True,
                "control_failure_preserved": {"path": c["root"] + "/validation_stage_receipt.json",
                    "sha256": sha(local(root, "validation_stage_receipt.json"))}}
    require(value["validated_stages"]["validation"] == final["stages"]["validation"] == admitted,
            "Admission view changed scientific replay exits or receipts")
    for key, filename in (("validation_failure_preserved", "dispatch_failure.json"),
                          ("upstream_dispatcher_exit", "dispatcher_exit.json")):
        item = value[key]
        require(item["path"] == c["root"] + "/" + filename and item["sha256"] == sha(local(root, filename)), "Preserved failure binding changed")
    old_dispatcher = read(local(root, "dispatcher_exit.json"))
    require(f["queue_started_at"] == old_dispatcher["started_at"], "Original queue budget was reset")
    children = failure.get("children_left_running", {})
    expected_children = {key for key in c["training_jobs"] if key.endswith(("__seed42", "__seed62"))}
    require(set(children) == expected_children and len(children) == 2, "Historical failed child set changed")
    for key, pid in children.items():
        ex = old_stage["exits"][key]
        require(ex["pid"] == pid and datetime.fromisoformat(ex["observed_exit_at"]) >= datetime.fromisoformat(failure["at"]),
                "Historical active worker did not subsequently drain")
    test = final["test_stage_receipt"]
    require(test["path"] == "test_stage_receipt.json" and sha(local(finish, test["path"])) == test["sha256"]
            and read(local(finish, test["path"])) == final["stages"]["test"], "Final Test stage binding changed")
    return value


def verify_terminal_evidence(f, c, evidence):
    """Pure local SHA/selector/exit verification; never import models or read GPU state."""
    evidence = Path(evidence)
    root, original_root = evidence / "recovery", evidence / "original"
    finish = evidence / "finish"
    verify_finish_scope(f, c, finish)
    require(canonical(read(local(root, "recovery_contract.json"))) == canonical(c), "Archived recovery contract changed")
    original_path = mapped(original_root, c["immutable_originals"]["root"], c["scientific_contract"]["path"])
    require(sha(original_path) == c["scientific_contract"]["raw_sha256"], "Original scientific contract bytes changed")
    original = read(original_path)
    require(canonical(original) == c["scientific_contract"]["canonical_sha256"]
            and c["source"]["files"] == original["source"]["files"]
            and canonical(c["source"]["files"]) == c["source"]["files_sha256"]
            and len(c["source"]["files"]) == 989, "Frozen source989 contract changed")
    immutable = c["immutable_originals"]["files"]
    require(len(immutable) == 1141 and canonical(immutable) == c["immutable_originals"]["files_sha256"],
            "Frozen original1141 inventory changed")
    inventory(original_root, immutable)
    bridge = "continuation_from" in c
    source_files = {**c["source"]["files"], **c["input_files"], **c["operation_files"], **c.get("bridge_files", {})}
    inventory(root / "source", source_files)
    for name in c.get("required_source_directories", []):
        require(local(root / "source", name).is_dir(), "Required source directory missing")
    inventory(original_root / "source", {**c["source"]["files"], **c["input_files"]})
    require(canonical(c["operation_files"]) == c["operation_files_sha256"], "Operation closure changed")
    effective = deepcopy(original)
    effective["hosts"]["5090"]["root"] = c["root"]
    effective_sha = canonical(effective)
    jobs = c["jobs"]
    ids = {j["id"] for j in jobs}
    require(len(jobs) == len(ids) == 9 and ids == {j["id"] for j in original["jobs"]}, "Original nine-job matrix changed")
    final = read(local(finish, "final_receipt.json"))
    require(final.get("schema") == FINISH_SCHEMA and final.get("status") == "complete"
            and final.get("contract_sha256") == canonical(f)
            and final.get("upstream_contract_sha256") == canonical(c)
            and final.get("candidate_test_count") == 9
            and final.get("actual_scientific_worker_exit0_count") == 27
            and final.get("new_training_worker_count") == final.get("new_validation_worker_count") == 0
            and final.get("owned_worker_pids_absent") is True
            and final.get("dispatcher_own_exit_directly_observed") is False,
            "Incomplete terminal Test-only finish")
    preserved = final["immutable_originals_verification"]
    require(preserved.get("file_count") == 1141 and preserved.get("files_sha256") == canonical(immutable)
            and preserved.get("original_files_unchanged") is True, "Native original preservation proof changed")
    bound = final["artifact_manifest"]
    manifest_path = local(finish, bound["path"])
    require(sha(manifest_path) == bound["sha256"], "Native terminal manifest changed")
    terminal_manifest = read(manifest_path)
    files = terminal_manifest["files"]
    require(files and len(files) == bound["file_count"], "Empty or truncated native terminal manifest")
    inventory(finish, files)
    previous = verify_continuation_import(c, root)
    admission = verify_admission(f, c, root, finish, final)
    preexisting, added = admission["preexisting_v4_files"], final["added_v4_files"]
    require(final.get("preexisting_v4_files_unchanged") is True
            and not set(preexisting).intersection(added), "Preexisting and newly added v4 artifacts overlap")
    inventory(root, {**preexisting, **added})
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == set(preexisting) | set(added), "Unbound or missing upstream artifact")
    excluded = set(terminal_manifest["excluded_lifecycle_files"])
    require(excluded == LIFECYCLE_FILES, "Unapproved manifest lifecycle exclusions")
    actual_finish = {str(p.relative_to(finish)) for p in finish.rglob("*") if p.is_file() and p.relative_to(finish).parts[0] != "cache"}
    require(actual_finish <= set(files) | excluded and set(files) <= actual_finish and {"finish_contract.json", "admission.json", "start.json",
            "dispatcher_launch.json", "test_stage_receipt.json"} <= set(files), "Unbound or missing terminal artifact")
    require({str(p.relative_to(original_root)) for p in original_root.rglob("*") if p.is_file()} == set(immutable),
            "Original archive inventory is incomplete or has extra files")

    def direct_exit(value, prefix, launch, *, stage=None, job=None, command_contract=None, stream_root=None):
        require(type(value.get("actual_os_returncode")) is int and value["actual_os_returncode"] == 0
                and value.get("timeout") is False and type(value.get("pid")) is int and value["pid"] > 0
                and type(value.get("start_ticks")) is int and value["start_ticks"] > 0
                and value.get("observed_exit_at") and value.get("started_at"), "Missing actual successful child exit")
        require(all(value.get(k) == launch.get(k) for k in ("pid", "start_ticks", "command", "started_at", "boot_id"))
                and value.get("boot_id") == c["boot_id"], "Child exit/launch identity changed")
        if stage is not None:
            require(value.get("stage") == stage and value.get("job_id") == job["id"], "Child exit job changed")
            context = c if command_contract is None else command_contract
            from paper.scripts import titans_mac_accuracy9_recovery_runtime as previous_runtime
            builder = previous_runtime.worker_command
            require(value["command"] == builder(context, Path(context["root"]) / "recovery_contract.json", job, stage),
                    "Scientific child command changed")
        for stream in ("stdout", "stderr"):
            path = local(root if stream_root is None else stream_root, prefix + "." + stream + ".log")
            require(type(value.get(stream + "_bytes")) is int and value[stream + "_bytes"] == path.stat().st_size
                    and value[stream + "_sha256"] == sha(path), "Closed child log bytes/SHA changed")
        return value["pid"]

    launch, start = read(local(finish, "dispatcher_launch.json")), read(local(finish, "start.json"))
    identity = launch["identity"]
    expected_command = [f["python"], "-u", f["root"] + "/source/titans_mac_accuracy9_terminal_runtime.py",
                        "--contract", f["root"] + "/finish_contract.json", "--contract-sha", canonical(f), "--mode", "dispatch"]
    require(launch["command"] == expected_command and identity["argv"] == expected_command
            and identity["cwd"] == c["source_root"] and identity["pgid"] == identity["pid"], "Terminal dispatcher command changed")
    require(all(start["identity"].get(k) == identity.get(k) for k in IDENTITY_KEYS)
            and start["identity"]["pid"] == identity["pid"] and identity["argv"] == launch["command"]
            and launch["contract_sha256"] == start["contract_sha256"] == canonical(f), "Dispatcher identity changed")
    dispatcher_launch = {**identity, "command": launch["command"], "started_at": launch["at"], "boot_id": c["boot_id"]}
    dispatcher_exit = read(local(finish, "dispatcher_exit.json"))
    require(dispatcher_exit.get("cleanup_error") is None, "Dispatcher cleanup failed")
    owned = {direct_exit(dispatcher_exit, "dispatcher", dispatcher_launch, stream_root=finish)}
    inherited_owned = set()
    inherited_identities = []
    require(set(final["stages"]) == set(("reference", "train", "validation", "test")), "Missing native stage")
    train_receipts = {}
    gate = read(local(root, "validation_gate.json"))
    require(gate.get("status") == "passed" and gate.get("host") == "5090"
            and gate.get("contract_sha256") == effective_sha and set(gate.get("jobs", {})) == ids,
            "All-nine native Validation gate changed")
    for stage, keys in (("reference", c["reference_jobs"]), ("train", c["training_jobs"]),
                        ("validation", c["validation_jobs"]), ("test", c["test_jobs"])):
        terminal = final["stages"][stage]
        require(terminal.get("status") == "complete" and set(terminal["exits"]) == set(terminal["receipts"]) == set(keys),
                "Incomplete native stage matrix: " + stage)
        for job in [j for j in jobs if j["id"] in keys]:
            folder = "run/" + job["id"] + "/"
            exited = read(local(root, folder + stage + "_exit.json"))
            require(exited == terminal["exits"][job["id"]], "Terminal exit differs from its actual record")
            imported_stage = bridge and stage in ("reference", "train")
            owner = direct_exit(exited, "logs/" + job["id"] + "." + stage,
                read(local(root, folder + stage + "_launch.json")), stage=stage, job=job,
                command_contract=previous if imported_stage else None, stream_root=finish if stage == "test" else root)
            (inherited_owned if imported_stage else owned).add(owner)
            if imported_stage:
                inherited_identities.append({"pid": exited["pid"], "start_ticks": exited["start_ticks"],
                    "argv": exited["command"], "cwd": previous["source_root"], "pgid": exited["pid"]})
            binding = terminal["receipts"][job["id"]]
            path = local(root, binding["path"])
            require(str(path.relative_to(root)) == folder + stage + "_recovery_receipt.json"
                    and sha(path) == binding["sha256"], "Stage receipt binding changed")
            value = read(path)
            require(value.get("schema") == legacy.r.STAGE_SCHEMA and value.get("status") == "complete"
                    and value.get("stage") == stage and value.get("job_id") == job["id"]
                    and value.get("contract_sha256") == canonical(c)
                    and value.get("scientific_contract_sha256") == canonical(original)
                    and value.get("scientific_contract_raw_sha256") == c["scientific_contract"]["raw_sha256"]
                    and value.get("effective_scientific_contract_sha256") == effective_sha,
                    "Recovery receipt identity changed")
            require(all(value.get(k) == job[k] for k in ("host", "dataset", "arm", "seed", "time_alpha", "loss_mode", "time_input")),
                    "Recovery scientific condition changed")
            require(all(relative(name).is_relative_to(Path("run") / job["id"]) for name in value["files"]),
                    "Recovery inventory escaped assigned job")
            inventory(root, value["files"])
            selected = mapped(root, c["root"], value["selected_checkpoint"])
            require(selected.is_relative_to(root / "run" / job["id"])
                    and sha(selected) == value["checkpoint_sha256"]
                    and value["files"].get(str(selected.relative_to(root))) == value["checkpoint_sha256"]
                    and isinstance(value.get("state_sha256"), str) and len(value["state_sha256"]) == 64,
                    "Selected binary/tensor binding changed")
            _proof(value["time_head_training_verification"])
            train_path = local(root, folder + "train_receipt.json")
            train = read(train_path)
            require(train.get("contract_sha256") == effective_sha and train.get("stage") == "train"
                    and train.get("source_files_sha256") == c["source"]["files_sha256"]
                    and train.get("source_revision") == original["source"]["base_git_revision"]
                    and all(train.get(k) == job[k] for k in ("host", "dataset", "arm", "seed", "time_alpha", "loss_mode", "time_input"))
                    and all(train.get(k) == value.get(k) for k in
                        ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256", "time_head_training_verification")),
                    "Scientific selected endpoint changed")
            inventory(root, train["files"])
            summary = train["summary"]
            require(summary["best_epoch"] == value["selected_epoch"]
                    and summary["checkpoint_state_sha256"] == value["state_sha256"]
                    and summary["time_head_training_verification"] == value["time_head_training_verification"],
                    "Selected tensor/summary differs")
            history = read(selected.parent / "history.json")["history"]
            require(history and [row["epoch"] for row in history] == list(range(1, len(history) + 1)), "Incomplete selection history")
            finite = [row for row in history if type(row.get("val_qty_rmse")) in (int, float) and math.isfinite(row["val_qty_rmse"])]
            require(finite and min(finite, key=lambda row: row["val_qty_rmse"])["epoch"] == value["selected_epoch"],
                    "Selected epoch is not the first strict finite Validation minimum")
            train_receipts[job["id"]] = train
            if stage == "reference":
                origin_receipt_path = mapped(original_root, c["immutable_originals"]["root"], job["origin_train_receipt"]["path"])
                origin_exit_path = mapped(original_root, c["immutable_originals"]["root"], job["origin_train_exit"]["path"])
                require(sha(origin_receipt_path) == job["origin_train_receipt"]["sha256"]
                        and sha(origin_exit_path) == job["origin_train_exit"]["sha256"], "Historical fit binding changed")
                origin, old_exit = read(origin_receipt_path), read(origin_exit_path)
                old_launch = read(local(original_root, folder + "train_launch.json"))
                require(type(old_exit.get("actual_os_returncode")) is int and old_exit["actual_os_returncode"] == 0
                        and old_exit.get("timeout") is False and old_exit.get("job_id") == job["id"]
                        and old_exit.get("stage") == "train" and old_exit.get("observed_exit_at")
                        and all(old_exit.get(k) == old_launch.get(k) for k in ("pid", "start_ticks", "started_at"))
                        and type(old_exit.get("pid")) is int and old_exit["pid"] > 0
                        and type(old_exit.get("start_ticks")) is int and old_exit["start_ticks"] > 0
                        and old_exit == read(local(root, folder + "train_exit.json")), "Historical fit actual exit0 changed")
                inventory(original_root, origin["files"])
                require(value["lineage"]["reference_retrained"] is False
                        and all(origin.get(k) == train.get(k) for k in ("selected_epoch", "checkpoint_sha256", "state_sha256", "time_head_training_verification")),
                        "Completed historical fit was replaced")
            if stage == "train":
                lineage = value["lineage"]
                copy = lineage["resume_checkpoint"]
                require(lineage["resumed_from_epoch"] == job["resume_epoch"] and lineage["next_epoch"] == job["resume_epoch"] + 1
                        and copy["original_path"] == job["origin_resume_checkpoint"]["path"]
                        and copy["original_sha256"] == copy["copy_sha256"] == job["origin_resume_checkpoint"]["sha256"]
                        and sha(mapped(root, c["root"], copy["copy_path"])) == copy["copy_sha256"]
                        and lineage["resume_validation"]["validated"] is True,
                        "Interrupted checkpoint continuity changed")
            if stage in ("validation", "test"):
                data = next(d for d in original["datasets"] if d["dataset_id"] == job["dataset"])
                identity = (data["inherited_data_identity"]["populations"]["validation"] if stage == "validation"
                            else data["full_data_identity"]["populations"]["test"])
                require(value["parameters_unchanged"] is True and value["context_unchanged"] is True
                        and value["native_runtime_checked"] is True and value["held_out_test_evaluated"] is (stage == "test")
                        and value["target_identity"] == identity
                        and value["metrics"]["overall"]["count"] == identity["target_count"],
                        "Native replay proof incomplete")
            if stage == "validation":
                require(value["validation_replay_passed"] is True and value["individual_predictions_match"] is True
                        and value["target_identity"] == train["target_identity"], "Validation endpoint changed")
                _same_metrics(value["metrics"], train["validation_reference"]["metrics"])
                from paper.scripts.run_titantpp_mlp_cap36 import compare_prediction_files
                compared = compare_prediction_files(mapped(root, c["root"], value["prediction_path"]),
                    mapped(root, c["root"], train["validation_reference"]["prediction_path"]))
                require(compared == value["target_identity"]["target_count"], "Incomplete Validation individual predictions")
                for name in ("train", "validation"):
                    require(sha(local(root, folder + name + "_exit.json")) == gate["jobs"][job["id"]][name + "_exit_sha256"],
                            "Validation gate actual exit changed")
                require(sha(local(root, folder + "validation_receipt.json")) == gate["jobs"][job["id"]]["validation_receipt_sha256"],
                        "Validation gate replay receipt changed")
    # The failed v4/v3 dispatchers remain historical exit1; neither is turned into an exit0.
    for historical in (root, root / "continuation_v3"):
        launch = read(local(historical, "dispatcher_launch.json"))
        exited = read(local(historical, "dispatcher_exit.json"))
        identity = launch["identity"]
        require(type(exited.get("actual_os_returncode")) is int and exited["actual_os_returncode"] == 1
                and exited.get("timeout") is False and exited.get("cleanup_error") is None
                and exited.get("pid") == identity["pid"] and exited.get("start_ticks") == identity["start_ticks"]
                and exited.get("command") == identity["argv"] == launch["command"],
                "Historical failed dispatcher identity changed")
        for stream in ("stdout", "stderr"):
            log = local(historical, "dispatcher." + stream + ".log")
            require(exited[stream + "_sha256"] == sha(log) and exited[stream + "_bytes"] == log.stat().st_size,
                    "Historical dispatcher closed logs changed")
        inherited_identities.append(identity)
    observation = read(local(evidence, "collection_observation.json"))
    require(observation.get("schema") == SCHEMA and observation.get("contract_sha256") == canonical(f)
            and observation.get("upstream_contract_sha256") == canonical(c)
            and observation.get("boot_id") == c["boot_id"] and observation.get("owned_pids_absent") is True
            and set(observation.get("owned_pids", [])) == owned
            and len(owned) == 19 and len(inherited_owned) == 9
            and observation.get("gpu", {}).get("uuid") == c["gpu_uuid"]
            and observation.get("gpu", {}).get("compute_pids") == []
            and observation.get("original_files_unchanged") is True
            and observation.get("supervisor_own_os_exit_directly_observed") is False
            and observation.get("inherited_process_identities_absent") is True
            and {canonical(i) for i in observation.get("inherited_process_identities", [])}
                == {canonical(i) for i in inherited_identities}, "Missing native collection ownership observation")
    return {"status": "verified", "scientific_fit_count": 9, "historical_fit_exit0_count": 6,
            "resumed_fit_exit0_count": 3, "validation_exit0_count": 9, "test_exit0_count": 9,
            "actual_scientific_stage_exit0_count": 27, "new_training_worker_count": 0,
            "new_validation_worker_count": 0, "current_test_worker_exit0_count": 9,
            "reference_copy_child_exit0_count": 6, "dispatcher_exit0_count": 1,
            "historical_failed_dispatcher_exit1_count": 2, "source_file_count": 989,
            "immutable_original_file_count": 1141, "preexisting_v4_file_count": len(preexisting),
            "preexisting_v4_files_sha256": canonical(preexisting),
            "final_receipt_sha256": sha(local(finish, "final_receipt.json")),
            "artifact_manifest_sha256": sha(manifest_path), "supervisor_own_os_exit_directly_observed": False}


def _manager():
    return importlib.import_module("paper.scripts.manage_titans_mac_accuracy9_terminal_finish")


def verify_local(expected):
    f, report, bundle = _manager().verify_local(expected)
    require(Path(report) == REPORT and Path(bundle) == BUNDLE and canonical(f) == expected,
            "Terminal controller paths or contract changed")
    c = read(UPSTREAM_BUNDLE / "recovery_contract.json")
    require(canonical(c) == UPSTREAM_SHA == f["upstream_contract_sha256"], "Pinned upstream contract changed")
    return f, c


def terminal_program(f, c):
    """One short read only; never poll a scientific worker or launch a finisher."""
    return f'''import hashlib,json,pathlib,sys,time
sys.dont_write_bytecode=True
sys.path.insert(0,{c['source_root']!r})
from paper.scripts import titans_mac_accuracy9_recovery_runtime as r
p=pathlib.Path({f['root']!r});f=r.read(p/'finish_contract.json')
r.require(r.sha_json(f)=={canonical(f)!r} and r.boot_id()==f['boot_id'],'Finish contract/boot changed')
for n,h in f['operation_files'].items():r.require(r.sha_file(p/n)==h,'Finish operation changed')
l=r.read(p/'dispatcher_launch.json');s=r.read(p/'start.json');i=l['identity']
r.require(i and l['contract_sha256']==s['contract_sha256']==r.sha_json(f) and i['argv']==l['command'] and all(s['identity'][k]==i[k] for k in {IDENTITY_KEYS!r}),'Dispatcher launch binding changed')
v={{'schema':{SCHEMA!r},'observed_at':r.now(),'contract_sha256':r.sha_json(f),'upstream_contract_sha256':{canonical(c)!r},'boot_id':r.boot_id(),'dispatcher_identity':i,'read_only_check':True}}
if (p/'dispatcher_exit.json').exists():
 raw=(p/'dispatcher_exit.json').read_bytes()
 try:x=json.loads(raw)
 except Exception:
  sys.stderr.buffer.write(b'PRESERVED_INCOMPLETE_NATIVE_RECORD_BEGIN\\n'+raw+b'\\nPRESERVED_INCOMPLETE_NATIVE_RECORD_END\\n');raise
 r.require(x['pid']==i['pid'] and x['start_ticks']==i['start_ticks'] and x['command']==i['argv'],'Terminal identity changed')
 v.update(status='terminal' if type(x.get('actual_os_returncode')) is int and x['actual_os_returncode']==0 and x.get('timeout') is False and x.get('cleanup_error') is None else 'terminal_failure',dispatcher_exit=x)
else:
 live=r.proc_identity(i['pid'])
 r.require(live is not None and all(live[k]==i[k] for k in {IDENTITY_KEYS!r}),'Dispatcher disappeared or identity changed')
 r.require(time.time()<min(f['absolute_host_deadline_unix'],__import__('datetime').datetime.fromisoformat(f['queue_started_at']).timestamp()+f['limits']['queue_wall_seconds']),'Original deadline expired')
 v.update(status='waiting')
print(json.dumps(v,allow_nan=False),flush=True)
'''


def collection_program(f, c):
    """Stream one archive after terminal exit0; leave native evidence unchanged."""
    return f'''import io,json,pathlib,sys,tarfile
sys.dont_write_bytecode=True
sys.path.insert(0,{c['source_root']!r})
from paper.scripts import titans_mac_accuracy9_recovery_runtime as r
p=pathlib.Path({f['root']!r});u=pathlib.Path({c['root']!r});f=r.read(p/'finish_contract.json');c=r.read(u/'recovery_contract.json')
r.require(r.sha_json(f)=={canonical(f)!r} and r.sha_json(c)=={canonical(c)!r} and r.boot_id()==f['boot_id']==c['boot_id'],'Finish scientific contract/boot changed')
for n,h in f['operation_files'].items():r.require(r.sha_file(p/n)==h,'Finish operation changed')
for n,h in {{**c['source']['files'],**c['input_files'],**c['operation_files'],**c.get('bridge_files',{{}})}}.items():r.require(r.sha_file(u/'source'/n)==h,'Frozen source/input changed')
a=r.read(p/'admission.json');v=r.read(p/'final_receipt.json');x=r.read(p/'dispatcher_exit.json');l=r.read(p/'dispatcher_launch.json');s=r.read(p/'start.json');i=l['identity']
r.require(v['schema']=={FINISH_SCHEMA!r} and v['status']=='complete' and v['contract_sha256']==r.sha_json(f) and v['upstream_contract_sha256']==r.sha_json(c) and v['candidate_test_count']==9 and v['actual_scientific_worker_exit0_count']==27 and v['new_training_worker_count']==v['new_validation_worker_count']==0,'Native finish incomplete')
r.require(type(x.get('actual_os_returncode')) is int and x['actual_os_returncode']==0 and x.get('timeout') is False and x.get('cleanup_error') is None and x['pid']==i['pid'] and x['start_ticks']==i['start_ticks'] and x['command']==i['argv']==l['command'],'Dispatcher actual exit0 missing')
r.require(l['contract_sha256']==s['contract_sha256']==r.sha_json(f) and all(s['identity'][k]==i[k] for k in {IDENTITY_KEYS!r}),'Dispatcher start changed')
before=a['preexisting_v4_files'];added=v['added_v4_files'];r.require(before and not set(before).intersection(added) and r.sha_json(before)==a['preexisting_v4_files_sha256'],'Preservation snapshot changed')
for n,h in {{**before,**added}}.items():r.require(r.sha_file(u/n)==h,'Upstream preservation changed')
r.require({{str(q.relative_to(u)) for q in u.rglob('*') if q.is_file()}}==set(before)|set(added),'Unbound upstream artifact')
b=v['artifact_manifest'];r.require(r.sha_file(p/b['path'])==b['sha256'],'Finish manifest changed');m=r.read(p/b['path'])['files'];r.require(m and len(m)==b['file_count'],'Empty manifest')
for n,h in m.items():r.require(r.sha_file(p/n)==h,'Finish artifact changed')
pids=[x['pid']];inherited=[]
for stage,key in (('reference','reference_jobs'),('train','training_jobs'),('validation','validation_jobs'),('test','test_jobs')):
 stage_value=v['stages'][stage];r.require(stage_value['status']=='complete' and set(stage_value['exits'])==set(stage_value['receipts'])==set(c[key]),'Incomplete stage')
 for job in [j for j in c['jobs'] if j['id'] in c[key]]:
  ex=stage_value['exits'][job['id']];r.require(type(ex.get('actual_os_returncode')) is int and ex['actual_os_returncode']==0 and ex.get('timeout') is False,'Scientific child did not exit0')
  r.require(ex==r.read(u/'run'/job['id']/(stage+'_exit.json')),'Scientific exit changed')
  bind=stage_value['receipts'][job['id']];r.require(r.sha_file(u/bind['path'])==bind['sha256'],'Scientific receipt changed')
  if stage in ('reference','train'):inherited.append({{'pid':ex['pid'],'start_ticks':ex['start_ticks'],'argv':ex['command'],'cwd':c['continuation_from']['root']+'/source','pgid':ex['pid']}})
  else:pids.append(ex['pid'])
for oldroot in (u,u/'continuation_v3'):
 identity=r.read(oldroot/'dispatcher_launch.json')['identity'];inherited.append(identity)
for identity in inherited:
 live=r.proc_identity(identity['pid']);r.require(live is None or live['start_ticks']!=identity['start_ticks'],'Historical identity remains')
gpu=r.gpu_state(c);r.require(all(r.proc_identity(pid) is None for pid in pids) and gpu['compute_pids']==[],'Owner/GPU remains')
old=pathlib.Path(c['immutable_originals']['root']);r.require(len(c['source']['files'])==989 and len(c['immutable_originals']['files'])==1141,'Frozen inventories changed')
for n,h in c['immutable_originals']['files'].items():r.require(r.sha_file(old/n)==h,'Immutable original changed')
observation={{'schema':{SCHEMA!r},'observed_at':r.now(),'contract_sha256':r.sha_json(f),'upstream_contract_sha256':r.sha_json(c),'boot_id':r.boot_id(),'owned_pids':sorted(set(pids)),'owned_pids_absent':True,'gpu':gpu,'original_files_unchanged':True,'supervisor_own_os_exit_directly_observed':False,'inherited_process_identities':inherited,'inherited_process_identities_absent':True}}
blob=(json.dumps(observation,sort_keys=True,allow_nan=False)+'\\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for namespace,native in (('recovery',u),('finish',p)):
  for q in sorted(native.rglob('*')):
   n=q.relative_to(native)
   if namespace=='finish' and n.parts[0]=='cache':continue
   r.require(not q.is_symlink(),'Archive symlink')
   if q.is_file():t.add(q,arcname=namespace+'/'+str(n),recursive=False)
   elif q.is_dir():
    info=tarfile.TarInfo(namespace+'/'+str(n));info.type=tarfile.DIRTYPE;info.mode=0o755;t.addfile(info)
 for n,h in sorted(c['immutable_originals']['files'].items()):
  q=old/n;r.require(r.sha_file(q)==h,'Original changed while archiving');t.add(q,arcname='original/'+n,recursive=False)
 info=tarfile.TarInfo('collection_observation.json');info.size=len(blob);t.addfile(info,io.BytesIO(blob))
'''


def transport(f, c, program, folder, *, label, timeout, output=None):
    folder.mkdir(parents=True, exist_ok=False)
    out = Path(output) if output is not None else folder / "stdout"
    err = folder / "stderr"
    native = "cd " + shlex.quote(c["source_root"]) + " && " + shlex.join(
        ["env", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + c["source_root"], f["python"], "-u", "-c", program])
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10",
               "-o", "ServerAliveCountMax=2", "5090", native]
    began, child, timed_out, error = time.monotonic(), None, False, None
    with out.open("xb") as stdout, err.open("xb") as stderr:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, cwd=ROOT)
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                child.kill()  # Only the local SSH transport; never a scientific process.
                child.wait()
        except OSError as exc:
            error = str(exc)
    record = {"schema": SCHEMA, "label": label, "at": legacy.r.now(), "timeout_seconds": timeout,
        "actual_transport_exit": None if child is None else child.returncode, "transport_timeout": timed_out,
        "transport_error": error, "elapsed_seconds": time.monotonic() - began, "command": command,
        "stdout_path": str(out), "stdout_bytes": out.stat().st_size, "stdout_sha256": sha(out),
        "stderr_path": str(err), "stderr_bytes": err.stat().st_size, "stderr_sha256": sha(err),
        "native_scientific_process_signaled": False, "automatic_retry": False}
    write(folder / "TRANSPORT.json", record, exclusive=True)
    require(record["actual_transport_exit"] == 0 and not timed_out and error is None,
            "SSH transport failed; raw bytes and actual exit preserved at " + str(folder))
    return record


def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = [m.name for m in members]
        require(members and len(names) == len(set(names)), "Empty archive or duplicate member")
        for member in members:
            require((member.isfile() or (member.isdir() and member.size == 0))
                    and not member.issparse() and member.size >= 0, "Archive contains a nonregular member")
            name = relative(member.name)
            require(name.parts[0] in ("recovery", "finish", "original") or str(name) == "collection_observation.json",
                    "Unexpected archive namespace")
            local(destination, member.name)
        stream.extractall(destination, members=members, filter="data")


@contextmanager
def advance_lock():
    with (report_dir() / "advance.lock").open("a+") as stream:
        acquired = False
        try:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_cache(f, c):
    marker = read(report_dir() / "COLLECTION_COMPLETE.json")
    root = retrieval_dir()
    require(marker["contract_sha256"] == canonical(f) and marker["upstream_contract_sha256"] == canonical(c)
            and sha(root / "archive.tar.gz") == marker["archive_sha256"]
            and sha(root / "manifest.json") == marker["manifest_sha256"]
            and sha(root / "receipt.json") == marker["receipt_sha256"], "Completed retrieval cache changed")
    receipt = read(root / "receipt.json")
    require(marker == {**receipt, "receipt_sha256": sha(root / "receipt.json"), "remote_calls": 2},
            "Completion marker differs from the actual receipt")
    transfer_path = Path(marker["collection_transport_path"])
    require(transfer_path.resolve().is_relative_to(report_dir().resolve())
            and sha(transfer_path) == marker["collection_transport_sha256"], "Collection transport binding changed")
    transfer = read(transfer_path)
    require(type(transfer.get("actual_transport_exit")) is int and transfer["actual_transport_exit"] == 0
            and transfer.get("transport_timeout") is False and transfer.get("transport_error") is None
            and Path(transfer["stdout_path"]) == root / "archive.tar.gz"
            and transfer["stdout_sha256"] == marker["archive_sha256"]
            and transfer["stdout_bytes"] == (root / "archive.tar.gz").stat().st_size,
            "Actual successful collection transport evidence changed")
    for stream in ("stdout", "stderr"):
        path = Path(transfer[stream + "_path"])
        require(path.resolve().is_relative_to(REPORT.resolve()) or path.resolve().is_relative_to(BUNDLE.resolve()),
                "Transport stream escaped the collection tree")
        require(sha(path) == transfer[stream + "_sha256"] and path.stat().st_size == transfer[stream + "_bytes"],
                "Collection stream raw bytes changed")
    manifest = read(root / "manifest.json")["files"]
    inventory(root / "evidence", manifest)
    actual = {str(p.relative_to(root / "evidence")) for p in (root / "evidence").rglob("*") if p.is_file()}
    require(actual == set(manifest), "Cached collection has missing or unbound files")
    verified = verify_terminal_evidence(f, c, root / "evidence")
    require(all(marker.get(k) == v for k, v in verified.items() if k != "status"), "Cached scientific verification changed")
    return {**marker, "cache_verified": True, "remote_calls": 0}


def advance(expected, *, check_local=False):
    f, c = verify_local(expected)
    folder = report_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "COLLECTION_COMPLETE.json").exists():
        return verify_cache(f, c)
    if check_local:
        return {"status": "not_collected", "remote_calls": 0, "contract_sha256": expected}
    with advance_lock() as acquired:
        if not acquired:
            return {"status": "busy", "remote_calls": 0}
        if (folder / "FAILURE.json").exists() or (folder / "COLLECTION_ATTEMPT.json").exists():
            return {"status": "blocked_preserved_failure_or_attempt", "remote_calls": 0, "automatic_retry": False}
        latest = read(folder / "LATEST.json") if (folder / "LATEST.json").exists() else None
        if latest and time.time() - latest["checked_unix"] < MINIMUM_CHECK_INTERVAL_SECONDS:
            return {"status": "waiting_next_check", "remote_calls": 0, "contract_sha256": expected}
        require(read(REPORT / "LAUNCHED.json")["contract_sha256"] == expected, "Approved terminal launch missing")
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:12]
        round_dir = folder / "rounds" / stamp
        round_dir.mkdir(parents=True, exist_ok=False)
        try:
            check = transport(f, c, terminal_program(f, c), round_dir / "check", label="read_only_terminal", timeout=CHECK_SECONDS)
            terminal = json.loads(Path(check["stdout_path"]).read_text())
            require(terminal.get("schema") == SCHEMA and terminal.get("contract_sha256") == expected
                    and terminal.get("upstream_contract_sha256") == canonical(c)
                    and terminal.get("read_only_check") is True and terminal.get("boot_id") == c["boot_id"]
                    and terminal.get("observed_at"), "Terminal response changed")
            require(terminal.get("status") in ("waiting", "terminal", "terminal_failure"), "Unknown terminal state")
            result = {"schema": SCHEMA, "contract_sha256": expected, "checked_unix": time.time(),
                "status": terminal["status"], "observation": terminal, "remote_calls": 1,
                "transport_path": str(round_dir / "check/TRANSPORT.json"),
                "transport_sha256": sha(round_dir / "check/TRANSPORT.json")}
            write(round_dir / "ROUND.json", result, exclusive=True)
            write(folder / "LATEST.json", result)
            require(terminal["status"] != "terminal_failure", "Native terminal dispatcher exited unsuccessfully")
            if terminal["status"] == "waiting":
                return result
            write(folder / "COLLECTION_ATTEMPT.json", {"at": legacy.r.now(), "contract_sha256": expected,
                "round": str(round_dir), "automatic_retry": False}, exclusive=True)
            destination = retrieval_dir()
            destination.mkdir(parents=True, exist_ok=False)
            transfer = transport(f, c, collection_program(f, c), round_dir / "collection", label="first_collection",
                                 timeout=TRANSFER_SECONDS, output=destination / "archive.tar.gz")
            safe_extract(destination / "archive.tar.gz", destination / "evidence")
            verified = verify_terminal_evidence(f, c, destination / "evidence")
            files = {str(p.relative_to(destination / "evidence")): sha(p)
                     for p in sorted((destination / "evidence").rglob("*")) if p.is_file()}
            write(destination / "manifest.json", {"schema": SCHEMA, "files": files}, exclusive=True)
            receipt = {"schema": SCHEMA, **verified, "status": "collection_complete", "contract_sha256": expected,
                "upstream_contract_sha256": canonical(c), "at": legacy.r.now(),
                "archive_sha256": sha(destination / "archive.tar.gz"),
                "manifest_sha256": sha(destination / "manifest.json"), "file_count": len(files),
                "collection_transport_path": str(round_dir / "collection/TRANSPORT.json"),
                "collection_transport_sha256": sha(round_dir / "collection/TRANSPORT.json"),
                "actual_collection_transport_exit": transfer["actual_transport_exit"], "automatic_retry": False}
            write(destination / "receipt.json", receipt, exclusive=True)
            complete = {**receipt, "receipt_sha256": sha(destination / "receipt.json"), "remote_calls": 2}
            write(folder / "COLLECTION_COMPLETE.json", complete, exclusive=True)
            return complete
        except Exception:
            write(folder / "FAILURE.json", {"schema": SCHEMA, "status": "blocked", "at": legacy.r.now(),
                "contract_sha256": expected, "round": str(round_dir), "traceback": traceback.format_exc(),
                "automatic_retry": False, "scientific_failure_inferred_from_transport": False}, exclusive=True)
            raise


def comparison_records(evidence, c):
    root = Path(evidence) / "recovery"
    return [{"dataset": job["dataset"], "split": split, "seed": job["seed"], "model": job["arm"],
        **{key: value[key] for key in ("selected_epoch", "selected_checkpoint", "checkpoint_sha256",
                                     "state_sha256", "target_identity", "metrics")}}
        for job in c["jobs"] for split in ("validation", "test")
        for value in [read(local(root, "run/" + job["id"] + "/" + split + "_receipt.json"))]]


def comparison_result(f, c, collection, *, created_at):
    evidence = retrieval_dir() / "evidence"
    origin = mapped(evidence / "original", c["immutable_originals"]["root"], c["scientific_contract"]["path"])
    require(sha(origin) == c["scientific_contract"]["raw_sha256"], "Original scientific contract bytes changed")
    original = read(origin)
    require(canonical(original) == c["scientific_contract"]["canonical_sha256"], "Original scientific contract changed")
    table, reproduction, shared_proof = comparison.verify_shared_evidence(
        comparison.PAPER, expected_manifest_sha=comparison.PAPER_MANIFEST_SHA)
    comparison.verify_common_conditions(original, reproduction)
    populations = {(d, s): comparison.expected_population(reproduction, d, s)
                   for d in comparison.DATASETS for s in comparison.SPLITS}
    source_files = {str(path): sha(path) for path in (Path(__file__), Path(comparison.__file__),
        Path(_manager().__file__), BUNDLE / "finish_contract.json", UPSTREAM_BUNDLE / "recovery_contract.json",
        report_dir() / "COLLECTION_COMPLETE.json", retrieval_dir() / "receipt.json", retrieval_dir() / "manifest.json")}
    return {"schema": COMPARISON_SCHEMA, "status": "complete", "created_at": created_at,
        "data_variant": "original_uncapped", "models": [comparison.SHARED, comparison.MAC],
        "seeds": list(comparison.SEEDS), "remote_calls": 0, "new_training": 0, "new_inference": 0,
        "held_out_test_evaluated": True,
        "runtime_context": comparison.evaluation_runtime_context(original, reproduction),
        "source_proof": {"finish_contract_sha256": canonical(f), "upstream_contract_sha256": canonical(c),
            "original_scientific_contract_raw_sha256": sha(origin),
            "original_scientific_contract_sha256": canonical(original), "mac_archive_sha256": collection["archive_sha256"],
            "mac_manifest_sha256": collection["manifest_sha256"], "mac_receipt_sha256": collection["receipt_sha256"],
            "collection_verified_locally": True, "files_sha256": source_files, "shared": shared_proof},
        **comparison.compare_records(table["records"], comparison_records(evidence, c), populations)}


JSON_NAME, MARKDOWN_NAME = "MAC_SHARED_THREE_SEED.json", "MAC_SHARED_THREE_SEED.md"
OUTPUT_NAMES = {"ATTEMPT.json", JSON_NAME, MARKDOWN_NAME}


def verify_comparison(destination, f, c, collection):
    require(destination.is_dir() and not destination.is_symlink()
            and {p.name for p in destination.iterdir()} == OUTPUT_NAMES | {"MANIFEST.json", "COMPLETE.json"},
            "Partial or unbound comparison output; preserve existing attempt")
    manifest = read(destination / "MANIFEST.json")
    require(manifest.get("schema") == COMPARISON_SCHEMA and manifest.get("status") == "complete"
            and set(manifest.get("files", {})) == OUTPUT_NAMES, "Empty or incomplete comparison inventory")
    inventory(destination, manifest["files"])
    result = read(destination / JSON_NAME)
    stamp = result["created_at"]
    require(datetime.fromisoformat(stamp).tzinfo is not None, "Comparison time has no timezone")
    require(read(destination / "ATTEMPT.json") == {"schema": COMPARISON_SCHEMA, "at": stamp,
        "contract_sha256": canonical(f), "automatic_retry": False}, "Comparison attempt changed")
    expected = comparison_result(f, c, collection, created_at=stamp)
    require(result == expected and (destination / MARKDOWN_NAME).read_text() == comparison.render_markdown(expected),
            "Pure comparison recomputation differs from cached outputs")
    require(read(destination / "COMPLETE.json") == {"schema": COMPARISON_SCHEMA, "status": "comparison_complete",
        "contract_sha256": canonical(f), "manifest_sha256": sha(destination / "MANIFEST.json"),
        "file_count": len(OUTPUT_NAMES), "source_proof_sha256": canonical(expected["source_proof"])},
        "Comparison completion binding changed")
    return {"status": "comparison_complete_verified", "report": str(destination), "cache_verified": True,
            "seed_metric_comparison_count": 108, "remote_calls": 0, "outputs_created": False}


def compare(expected, *, check_only=False):
    f, c = verify_local(expected)
    collection = advance(expected, check_local=True)
    destination = REPORT / "comparison_v1"
    if collection.get("status") != "collection_complete":
        require(not destination.exists(), "Partial comparison exists without a verified collection")
        require(check_only, "Terminal MAC collection is not complete")
        return {"status": "waiting_for_verified_MAC_collection", "remote_calls": 0, "outputs_created": False}
    require(collection.get("cache_verified") is True and collection.get("remote_calls") == 0,
            "Comparison requires a freshly verified local collection cache")
    if destination.exists():
        return verify_comparison(destination, f, c, collection)
    stamp = datetime.now(timezone.utc).isoformat()
    result = comparison_result(f, c, collection, created_at=stamp)
    if check_only:
        return {"status": "comparison_ready_outputs_not_written", "remote_calls": 0,
                "outputs_created": False, "seed_metric_comparison_count": 108}
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / "ATTEMPT.json", {"schema": COMPARISON_SCHEMA, "at": stamp,
        "contract_sha256": canonical(f), "automatic_retry": False}, exclusive=True)
    write(destination / JSON_NAME, result, exclusive=True)
    with (destination / MARKDOWN_NAME).open("x", encoding="utf-8") as stream:
        stream.write(comparison.render_markdown(result))
    write(destination / "MANIFEST.json", {"schema": COMPARISON_SCHEMA, "status": "complete",
        "files": {name: sha(destination / name) for name in sorted(OUTPUT_NAMES)}}, exclusive=True)
    write(destination / "COMPLETE.json", {"schema": COMPARISON_SCHEMA, "status": "comparison_complete",
        "contract_sha256": canonical(f), "manifest_sha256": sha(destination / "MANIFEST.json"),
        "file_count": len(OUTPUT_NAMES), "source_proof_sha256": canonical(result["source_proof"])}, exclusive=True)
    return {**verify_comparison(destination, f, c, collection), "outputs_created": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--mode", choices=("advance", "check-local", "compare"), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    require(not args.check_only or args.mode == "compare", "--check-only applies only to local comparison")
    value = (compare(args.contract_sha, check_only=args.check_only) if args.mode == "compare"
             else advance(args.contract_sha, check_local=args.mode == "check-local"))
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
