"""Short MAC recovery completion checks and one immutable, verified retrieval.

This controller never launches, resumes, evaluates, signals or retries native
scientific work.  Its only remote mutation is the one authorized archive stream;
the archive and its observation are created in memory, leaving remote files alone.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gzip
import importlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time
import traceback
import uuid

from paper.scripts import manage_titans_mac_accuracy9_recovery as manager

r = manager.r
require, read, sha, canonical, write = manager.require, manager.read, manager.sha, manager.canonical, manager.write
SCHEMA = "titans_mac_accuracy9_recovery_collection_v1"
CHECK_SECONDS, TRANSFER_SECONDS, MINIMUM_CHECK_INTERVAL_SECONDS = 45, 900, 900
IDENTITY_KEYS = ("pid", "start_ticks", "argv", "cwd", "pgid")


@contextmanager
def controller_context(completion_bridge=False):
    """Use the separately sealed bridge controller without changing v3 defaults."""
    global manager, r
    previous = manager, r
    try:
        if completion_bridge:
            manager = importlib.import_module("paper.scripts.manage_titans_mac_accuracy9_completion_bridge")
            r = importlib.import_module("paper.scripts.titans_mac_accuracy9_completion_bridge")
        yield
    finally:
        manager, r = previous


def report_dir():
    return manager.REPORT / "collection_v1"


def retrieval_dir():
    return manager.BUNDLE / "collection_v1" / "retrieval_v1"


def relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name and str(path) == name and not path.is_absolute()
            and ".." not in path.parts and "\\" not in name and path.parts != (".",), "Unsafe archive path")
    return Path(name)


def local(root, name):
    root = Path(root).resolve()
    path = root / relative(name)
    require(not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
            and path.resolve().is_relative_to(root), "Archive symlink or path escape")
    return path


def mapped(root, remote_root, name):
    path, remote = PurePosixPath(name), PurePosixPath(remote_root)
    require(path.is_absolute() and path.is_relative_to(remote) and ".." not in path.parts,
            "Absolute artifact escaped its original root")
    return local(root, str(path.relative_to(remote)))


def inventory(root, files):
    require(isinstance(files, dict) and bool(files), "Empty artifact inventory")
    for name, digest in files.items():
        require(isinstance(digest, str) and len(digest) == 64 and sha(local(root, name)) == digest,
                "Artifact bytes/SHA changed: " + name)


def _same_metrics(a, b):
    require(type(a) is type(b), "Metric structure changed")
    if isinstance(a, dict):
        require(a.keys() == b.keys(), "Metric keys changed")
        for key in a:
            _same_metrics(a[key], b[key])
    elif isinstance(a, list):
        require(len(a) == len(b), "Metric length changed")
        for x, y in zip(a, b):
            _same_metrics(x, y)
    elif type(a) in (int, float):
        require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5),
                "Validation metric changed")
    else:
        require(a == b, "Metric metadata changed")


def _proof(value):
    require(value.get("schema") == "time_head_joint_training_verification_v1"
            and value.get("time_alpha") == 1.0
            and type(value.get("optimizer_steps_checked")) is int and value["optimizer_steps_checked"] > 0
            and type(value.get("joint_loss_observations")) is int and value["joint_loss_observations"] > 0,
            "Missing cumulative joint time-head proof")
    for name in ("initial_time_head_sha256", "final_time_head_sha256", "checkpoint_time_head_sha256"):
        digest = value.get(name)
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Invalid joint tensor SHA")
    require(value["initial_time_head_sha256"] != value["checkpoint_time_head_sha256"]
            and value["initial_time_head_sha256"] != value["final_time_head_sha256"]
            and all(value.get(k) is True for k in ("finite_time_head_gradient_observed",
                "nonzero_time_head_gradient_observed", "time_head_parameters_changed", "optimizer_state_present",
                "joint_loss_includes_time", "all_trainable_parameters_in_optimizer")), "Incomplete joint proof")


def _contains_identity(value, identity):
    if isinstance(value, dict):
        if all(value.get(key) == identity.get(key) for key in IDENTITY_KEYS):
            return True
        return any(_contains_identity(item, identity) for item in value.values())
    return isinstance(value, list) and any(_contains_identity(item, identity) for item in value)


def verify_continuation_import(c, root):
    """Bind actual v3 fits to relocated v4 bytes without inventing a v4 fit."""
    continuation = c["continuation_from"]
    raw_root = root / "continuation_v3"
    imported = read(local(root, "import_receipt.json"))
    require(imported.get("schema") == "titans_mac_accuracy9_completion_import_v1", "Missing completion import provenance")
    inventory(raw_root, imported["origin_files"])
    require(set(imported["origin_files"]) == {str(p.relative_to(raw_root)) for p in raw_root.rglob("*") if p.is_file()},
            "Unbound or missing preserved v3 files")
    binding = imported["original_contract"]
    require(binding["path"] == "continuation_v3/recovery_contract.json", "Wrong continuation contract path")
    raw_contract = local(root, binding["path"])
    previous = read(raw_contract)
    require(sha(raw_contract) == binding["sha256"] == continuation["contract_raw_sha256"]
            and canonical(previous) == binding["canonical_sha256"] == continuation["contract_sha256"]
            and previous["root"] == continuation["root"], "Preserved v3 contract identity changed")
    require(previous["scientific_contract"] == c["scientific_contract"]
            and previous["source"] == c["source"] and previous["operation_files"] == c["operation_files"]
            and previous["input_files"] == c["input_files"] and previous["boot_id"] == c["boot_id"],
            "Completion bridge changed the original scientific closure")
    prior_final = read(local(raw_root, "final_receipt.json"))
    prior_exit = read(local(raw_root, "dispatcher_exit.json"))
    failure = read(local(raw_root, "dispatch_failure.json"))
    require(prior_final.get("status") == "failed" and prior_final.get("contract_sha256") == canonical(previous)
            and set(prior_final.get("stages", {})) == {"reference", "train"}
            and prior_final.get("held_out_test_evaluated") is False
            and type(prior_exit.get("actual_os_returncode")) is int and prior_exit["actual_os_returncode"] == 1
            and prior_exit.get("timeout") is False and prior_exit.get("cleanup_error") is None
            and failure.get("stage") == "train" and not failure.get("job_id")
            and failure.get("traceback", "").rstrip().endswith("ValueError: Owned child identity changed")
            and "in _identity" in failure.get("traceback", ""),
            "Only the preserved initial identity-race control failure may be bridged")
    require(set(imported["jobs"]) == {j["id"] for j in c["jobs"]}, "Not all nine fits were imported")
    from paper.scripts import titans_mac_accuracy9_recovery_runtime as previous_runtime
    for job in c["jobs"]:
        item = imported["jobs"][job["id"]]
        stage = "reference" if job["recovery_mode"] == "reference" else "train"
        require(item["stage"] == stage, "Imported fit stage changed")
        for key in ("scientific_receipt", "recovery_receipt", "exit", "launch"):
            entry = item[key]
            if key == "launch" and entry.get("original_path") is None:
                require(entry.get("original_sha256") is None
                        and entry.get("identity_supplemented") is True
                        and entry.get("reconstructed_from_initial_live_observation") is True
                        and stage == "train" and job["dataset"] == "intermittent_frozen_5000" and job["seed"] == 62,
                        "Only the observed missing seed62 launch may be reconstructed")
                absent = "run/" + job["id"] + "/train_launch.json"
                require(absent not in imported["origin_files"] and not local(raw_root, absent).exists(),
                        "A preserved actual launch cannot be replaced")
                raw_exit_entry = item["exit"]
                raw_exit_path = (mapped(raw_root, previous["root"], raw_exit_entry["original_path"])
                    if PurePosixPath(raw_exit_entry["original_path"]).is_absolute() else
                    local(root, raw_exit_entry["original_path"]) if relative(raw_exit_entry["original_path"]).parts[0] == "continuation_v3"
                    else local(raw_root, raw_exit_entry["original_path"]))
                raw_exit = read(raw_exit_path)
                proof = item["start_ticks_proof"]
                observation = (mapped(root, c["root"], proof["observation_path"])
                    if PurePosixPath(proof["observation_path"]).is_absolute() else local(root, proof["observation_path"]))
                identity = proof["identity"]
                require(sha(observation) == proof["observation_sha256"]
                        == continuation["initial_live_observation"]["sha256"]
                        and _contains_identity(read(observation), identity)
                        and identity["pid"] == raw_exit["pid"]
                        and identity["argv"] == raw_exit["command"]
                        and identity["cwd"] == previous["source_root"] and identity["pgid"] == identity["pid"],
                        "Reconstructed launch has no pinned initial live identity")
                relocated = (mapped(root, c["root"], entry["relocated_path"])
                    if PurePosixPath(entry["relocated_path"]).is_absolute() else local(root, entry["relocated_path"]))
                expected = {key: raw_exit[key] for key in ("job_id", "stage", "pid", "command", "started_at", "boot_id")}
                expected["start_ticks"] = identity["start_ticks"]
                require(sha(relocated) == entry["relocated_sha256"] and read(relocated) == expected,
                        "Reconstructed launch changed actual exit or initial identity")
                continue
            raw = (mapped(raw_root, previous["root"], entry["original_path"])
                   if PurePosixPath(entry["original_path"]).is_absolute() else
                   local(root, entry["original_path"]) if relative(entry["original_path"]).parts[0] == "continuation_v3"
                   else local(raw_root, entry["original_path"]))
            relocated = (mapped(root, c["root"], entry["relocated_path"])
                         if PurePosixPath(entry["relocated_path"]).is_absolute() else local(root, entry["relocated_path"]))
            require(sha(raw) == entry["original_sha256"] and sha(relocated) == entry["relocated_sha256"],
                    "Imported raw/relocated artifact SHA changed")
            if key in ("exit", "launch"):
                original_record, record = read(raw), read(relocated)
                require(original_record.get("job_id") == record.get("job_id") == job["id"]
                        and original_record.get("stage") == record.get("stage") == stage, "Imported process job changed")
                if entry.get("identity_supplemented"):
                    require(job["dataset"] == "intermittent_frozen_5000" and job["seed"] == 62
                            and stage == "train" and original_record.get("start_ticks") is None,
                            "Only the observed seed62 identity may be supplemented")
                    proof = item["start_ticks_proof"]
                    observation = (mapped(root, c["root"], proof["observation_path"])
                        if PurePosixPath(proof["observation_path"]).is_absolute() else local(root, proof["observation_path"]))
                    require(sha(observation) == proof["observation_sha256"]
                            == continuation["initial_live_observation"]["sha256"], "Initial live observation SHA changed")
                    identity = proof["identity"]
                    require(type(identity.get("start_ticks")) is int and identity["start_ticks"] > 0
                            and identity["pid"] == original_record["pid"]
                            and identity["argv"] == previous_runtime.worker_command(previous,
                                Path(previous["root"]) / "recovery_contract.json", job, stage)
                            and identity["cwd"] == previous["source_root"] and identity["pgid"] == identity["pid"]
                            and _contains_identity(read(observation), identity), "Exact initial live process identity missing")
                    require(record == {**original_record, "start_ticks": identity["start_ticks"]},
                            "Identity supplement changed another original field")
                else:
                    require(record == original_record, "Unsupplemented exit/launch changed")
                if key == "exit":
                    require(type(original_record.get("actual_os_returncode")) is int
                            and original_record["actual_os_returncode"] == 0
                            and original_record.get("timeout") is False, "Imported fit has no actual v3 exit0")
        inventory(root, item["files"])
        for name, digest in item["files"].items():
            require(imported["origin_files"].get(name) == digest, "Imported artifact has no identical v3 original")
    return previous


def verify_terminal_evidence(c, evidence):
    """Pure local SHA/selector/exit verification; never import models or read GPU state."""
    evidence = Path(evidence)
    root, original_root = evidence / "recovery", evidence / "original"
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
    final = read(local(root, "final_receipt.json"))
    require(final.get("schema") == r.SCHEMA and final.get("status") == "complete"
            and final.get("contract_sha256") == canonical(c)
            and final.get("resumed_fit_count") == 3 and final.get("preserved_completed_fit_count") == 6
            and final.get("validation_replay_count") == final.get("candidate_test_count") == 9
            and final.get("held_out_test_evaluated") is True and final.get("owned_worker_pids_absent") is True,
            "Incomplete native MAC recovery")
    preserved = final["immutable_originals_verification"]
    require(preserved.get("file_count") == 1141 and preserved.get("files_sha256") == canonical(immutable)
            and preserved.get("original_files_unchanged") is True, "Native original preservation proof changed")
    bound = final["artifact_manifest"]
    manifest_path = local(root, bound["path"])
    require(sha(manifest_path) == bound["sha256"], "Native terminal manifest changed")
    terminal_manifest = read(manifest_path)
    files = terminal_manifest["files"]
    require(files and len(files) == bound["file_count"], "Empty or truncated native terminal manifest")
    inventory(root, files)
    previous = verify_continuation_import(c, root) if bridge else None
    if bridge:
        require(final.get("current_new_training_count") == 0 and final.get("imported_completed_fit_count") == 9
                and final.get("imported_resumed_fit_count") == 3
                and final.get("imported_reference_copy_exit0_count") == 6, "Bridge fit provenance counts changed")
    excluded = set(terminal_manifest["excluded_lifecycle_files"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    expected = set(files) | {"source/" + name for name in source_files} | excluded
    require(actual == expected.intersection(actual) and set(files) <= actual
            and {"recovery_contract.json", "start.json", "dispatcher_launch.json", "validation_gate.json"} <= set(files),
            "Terminal manifest omitted an unexcluded artifact")
    require({str(p.relative_to(original_root)) for p in original_root.rglob("*") if p.is_file()} == set(immutable),
            "Original archive inventory is incomplete or has extra files")

    def direct_exit(value, prefix, launch, *, stage=None, job=None, command_contract=None):
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
            builder = previous_runtime.worker_command if command_contract is not None else r.worker_command
            require(value["command"] == builder(context, Path(context["root"]) / "recovery_contract.json", job, stage),
                    "Scientific child command changed")
        for stream in ("stdout", "stderr"):
            path = local(root, prefix + "." + stream + ".log")
            require(type(value.get(stream + "_bytes")) is int and value[stream + "_bytes"] == path.stat().st_size
                    and value[stream + "_sha256"] == sha(path), "Closed child log bytes/SHA changed")
        return value["pid"]

    launch, start = read(local(root, "dispatcher_launch.json")), read(local(root, "start.json"))
    identity = launch["identity"]
    require(all(start["identity"].get(k) == identity.get(k) for k in IDENTITY_KEYS)
            and start["pid"] == identity["pid"] and identity["argv"] == launch["command"]
            and launch["contract_sha256"] == start["contract_sha256"] == canonical(c), "Dispatcher identity changed")
    dispatcher_launch = {**identity, "command": launch["command"], "started_at": launch["at"], "boot_id": c["boot_id"]}
    dispatcher_exit = read(local(root, "dispatcher_exit.json"))
    require(dispatcher_exit.get("cleanup_error") is None, "Dispatcher cleanup failed")
    owned = {direct_exit(dispatcher_exit, "dispatcher", dispatcher_launch)}
    inherited_owned = set()
    inherited_identities = []
    require(set(final["stages"]) == set(r.STAGES), "Missing native stage")
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
                command_contract=previous if imported_stage else None)
            (inherited_owned if imported_stage else owned).add(owner)
            if imported_stage:
                inherited_identities.append({"pid": exited["pid"], "start_ticks": exited["start_ticks"],
                    "argv": exited["command"], "cwd": previous["source_root"], "pgid": exited["pid"]})
            binding = terminal["receipts"][job["id"]]
            path = local(root, binding["path"])
            require(str(path.relative_to(root)) == folder + stage + "_recovery_receipt.json"
                    and sha(path) == binding["sha256"], "Stage receipt binding changed")
            value = read(path)
            require(value.get("schema") == r.STAGE_SCHEMA and value.get("status") == "complete"
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
    observation = read(local(evidence, "collection_observation.json"))
    require(observation.get("schema") == SCHEMA and observation.get("contract_sha256") == canonical(c)
            and observation.get("boot_id") == c["boot_id"] and observation.get("owned_pids_absent") is True
            and set(observation.get("owned_pids", [])) == owned
            and observation.get("gpu", {}).get("compute_pids") == []
            and observation.get("original_files_unchanged") is True
            and observation.get("supervisor_own_os_exit_directly_observed") is False,
            "Missing native collection ownership observation")
    if bridge:
        inherited = observation.get("inherited_process_identities", [])
        require(len(owned) == 19 and len(inherited_owned) == 9
                and observation.get("inherited_process_identities_absent") is True
                and {i["pid"] for i in inherited} == inherited_owned
                and {canonical(i) for i in inherited} == {canonical(i) for i in inherited_identities},
                "Imported fit ownership observation missing")
    else:
        require(len(owned) == 28, "Expected 27 current child PIDs and one dispatcher PID")
    return {"status": "verified", "scientific_fit_count": 9, "historical_fit_exit0_count": 6,
            "resumed_fit_exit0_count": 3, "validation_exit0_count": 9, "test_exit0_count": 9,
            "actual_scientific_stage_exit0_count": 27, "current_scientific_child_exit0_count": 18 if bridge else 21,
            "current_new_training_count": 0 if bridge else 3, "imported_completed_fit_count": 9 if bridge else 0,
            "reference_copy_child_exit0_count": 6, "dispatcher_exit0_count": 1,
            "source_file_count": 989, "immutable_original_file_count": 1141,
            "final_receipt_sha256": sha(local(root, "final_receipt.json")),
            "artifact_manifest_sha256": sha(manifest_path), "supervisor_own_os_exit_directly_observed": False}


def terminal_program(c):
    return f'''import hashlib,json,pathlib,sys,time
import importlib
r=importlib.import_module({r.__name__!r})
from paper.scripts.titans_mac_accuracy9_recovery_runtime import boot_id as native_boot_id
p=pathlib.Path({c['root']!r});c=r.read(p/'recovery_contract.json')
r.require(r.sha_json(c)=={canonical(c)!r} and native_boot_id()==c['boot_id'],'Recovery contract/boot changed')
reads=[]
def complete_record(path):
 began=time.monotonic();deadline=began+1.;raw=b'';error=None
 recent=0.<=time.time()-path.stat().st_mtime<=2.
 for attempt in range(51):
  raw=path.read_bytes()
  try:
   value=json.loads(raw)
   r.require(isinstance(value,dict),'Lifecycle record must be an object')
   reads.append({{'path':str(path),'incomplete_read_count':attempt,'elapsed_seconds':time.monotonic()-began,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'bounded_complete_record_read':True}})
   return value
  except (json.JSONDecodeError,UnicodeDecodeError) as exc:error=str(exc)
  if not recent or time.monotonic()>=deadline or attempt==50:break
  time.sleep(.02)
 report={{'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'error':error,'elapsed_seconds':time.monotonic()-began,'bounded_read_exhausted':True}}
 sys.stderr.buffer.write((json.dumps(report,sort_keys=True)+'\\n').encode()+b'PRESERVED_INCOMPLETE_NATIVE_RECORD_BEGIN\\n'+raw+b'\\nPRESERVED_INCOMPLETE_NATIVE_RECORD_END\\n');sys.stderr.buffer.flush()
 raise ValueError('Lifecycle record stayed incomplete within the bounded one-second read')
l=complete_record(p/'dispatcher_launch.json');s=complete_record(p/'start.json');i=l['identity']
r.require(l['contract_sha256']==s['contract_sha256']==r.sha_json(c) and s['pid']==i['pid'] and i['argv']==l['command'],'Launch binding changed')
r.require(all(s['identity'][k]==i[k] for k in {IDENTITY_KEYS!r}),'Dispatcher identity changed')
v={{'schema':{SCHEMA!r},'observed_at':r.now(),'contract_sha256':r.sha_json(c),'boot_id':native_boot_id(),'dispatcher_identity':i,'read_only_check':True,'lifecycle_record_reads':reads}}
if (p/'dispatcher_exit.json').exists():
 x=complete_record(p/'dispatcher_exit.json')
 r.require(x['pid']==i['pid'] and x['start_ticks']==i['start_ticks'] and x['command']==i['argv'],'Terminal identity changed')
 v.update(status='terminal' if type(x.get('actual_os_returncode')) is int and x['actual_os_returncode']==0 and x.get('timeout') is False and x.get('cleanup_error') is None else 'terminal_failure',dispatcher_exit=x)
else:
 live=r.proc_identity(i['pid'])
 r.require(live is not None and all(live[k]==i[k] for k in {IDENTITY_KEYS!r}),'Dispatcher disappeared or identity changed')
 r.require(time.time()<c['absolute_host_deadline_unix'],'Original deadline expired')
 v.update(status='waiting')
print(json.dumps(v,allow_nan=False),flush=True)
'''


def collection_program(c):
    return f'''import io,json,pathlib,sys,tarfile
import importlib
r=importlib.import_module({r.__name__!r})
from paper.scripts.titans_mac_accuracy9_recovery_runtime import boot_id as native_boot_id
p=pathlib.Path({c['root']!r});c=r.read(p/'recovery_contract.json')
r.require(r.sha_json(c)=={canonical(c)!r},'Recovery contract changed');r.verify_files(c)
old=pathlib.Path(c['immutable_originals']['root']);f=r.read(p/'final_receipt.json');x=r.read(p/'dispatcher_exit.json')
l=r.read(p/'dispatcher_launch.json');s=r.read(p/'start.json');i=l['identity']
r.require(f['status']=='complete' and f['contract_sha256']==r.sha_json(c) and f['resumed_fit_count']==3 and f['preserved_completed_fit_count']==6 and f['validation_replay_count']==f['candidate_test_count']==9,'Native stages incomplete')
r.require(type(x.get('actual_os_returncode')) is int and x['actual_os_returncode']==0 and x.get('timeout') is False and x.get('cleanup_error') is None,'Dispatcher failed')
r.require(x['pid']==s['pid']==i['pid'] and x['start_ticks']==i['start_ticks'] and x['command']==l['command']==i['argv'],'Dispatcher identity changed')
r.require(all(s['identity'][k]==i[k] for k in {IDENTITY_KEYS!r}),'Start identity changed')
pids=[x['pid']];inherited=[]
for stage,key in (('reference','reference_jobs'),('train','training_jobs'),('validation','validation_jobs'),('test','test_jobs')):
 v=f['stages'][stage];r.require(v['status']=='complete' and set(v['exits'])==set(v['receipts'])==set(c[key]),'Incomplete stage')
 for j in [j for j in c['jobs'] if j['id'] in c[key]]:
  e=v['exits'][j['id']];r.require(type(e.get('actual_os_returncode')) is int and e['actual_os_returncode']==0 and e.get('timeout') is False,'Child failed')
  r.require(r.read(p/'run'/j['id']/(stage+'_exit.json'))==e,'Actual child exit changed')
  r.require(r.verify_receipt(c,j,stage)==v['receipts'][j['id']],'Stage receipt changed')
  if 'continuation_from' in c and stage in ('reference','train'):
   inherited.append({{'pid':e['pid'],'start_ticks':e['start_ticks'],'argv':e['command'],'cwd':c['continuation_from']['root']+'/source','pgid':e['pid']}})
  else:pids.append(e['pid'])
  if stage=='reference':
   e=r.read(pathlib.Path(j['origin_train_exit']['path']));r.require(type(e.get('actual_os_returncode')) is int and e['actual_os_returncode']==0 and e.get('timeout') is False,'Historical fit failed')
b=f['artifact_manifest'];r.require(r.sha_file(p/b['path'])==b['sha256'],'Native manifest changed');m=r.read(p/b['path'])['files']
r.require(m and len(m)==b['file_count'],'Empty manifest')
for n,h in m.items():r.require(r.sha_file(r.local_path(p,n))==h,'Native manifest artifact changed')
gpu=r.gpu_state(c);r.require(all(r.proc_identity(pid) is None for pid in pids) and gpu['compute_pids']==[],'Owner/GPU remains')
for identity in inherited:
 live=r.proc_identity(identity['pid']);r.require(live is None or live['start_ticks']!=identity['start_ticks'],'Inherited process identity remains')
r.require(len(c['source']['files'])==989 and len(c['immutable_originals']['files'])==1141,'Frozen inventories changed')
observation={{'schema':{SCHEMA!r},'observed_at':r.now(),'contract_sha256':r.sha_json(c),'boot_id':native_boot_id(),'owned_pids':sorted(set(pids)),'owned_pids_absent':True,'gpu':gpu,'original_files_unchanged':True,'supervisor_own_os_exit_directly_observed':False,'inherited_process_identities':inherited,'inherited_process_identities_absent':True}}
blob=(json.dumps(observation,sort_keys=True,allow_nan=False)+'\\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n in c.get('required_source_directories',[]):
  q=r.local_path(p/'source',n);r.require(q.is_dir() and not q.is_symlink(),'Required source directory missing')
  info=tarfile.TarInfo('recovery/source/'+n);info.type=tarfile.DIRTYPE;info.mode=0o755;t.addfile(info)
 for q in sorted(p.rglob('*')):
  n=q.relative_to(p)
  if n.parts[0]=='cache':continue
  r.require(not q.is_symlink(),'Unexpected recovery symlink')
  if q.is_file():t.add(q,arcname='recovery/'+str(n),recursive=False)
 for n,h in sorted(c['immutable_originals']['files'].items()):
  q=r.local_path(old,n);r.require(r.sha_file(q)==h,'Original changed while archiving');t.add(q,arcname='original/'+n,recursive=False)
 info=tarfile.TarInfo('collection_observation.json');info.size=len(blob);t.addfile(info,io.BytesIO(blob))
'''


def _command(c, program):
    native = "cd " + shlex.quote(c["source_root"]) + " && " + shlex.join(
        ["env", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + c["source_root"], c["python"], "-u", "-c", program])
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=2", "5090", native]


def transport(c, program, folder, *, label, timeout, output=None):
    folder.mkdir(parents=True, exist_ok=False)
    out = Path(output) if output is not None else folder / "stdout"
    err = folder / "stderr"
    began, child, timed_out, error = time.monotonic(), None, False, None
    command = _command(c, program)
    with out.open("xb") as stdout, err.open("xb") as stderr:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, cwd=manager.ROOT)
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                child.kill()  # Only this local SSH transport, never a native scientific process.
                child.wait()
        except OSError as exc:
            error = str(exc)
    record = {"schema": SCHEMA, "label": label, "at": r.now(), "timeout_seconds": timeout,
              "actual_transport_exit": None if child is None else child.returncode, "transport_timeout": timed_out,
              "transport_error": error, "elapsed_seconds": time.monotonic() - began, "command": command,
              "stdout_path": str(out), "stdout_bytes": out.stat().st_size, "stdout_sha256": sha(out),
              "stderr_path": str(err), "stderr_bytes": err.stat().st_size, "stderr_sha256": sha(err),
              "native_scientific_process_signaled": False, "automatic_retry": False}
    write(folder / "TRANSPORT.json", record, exclusive=True)
    require(record["actual_transport_exit"] == 0 and not timed_out and error is None,
            "SSH transport failed; actual exit and raw bytes preserved at " + str(folder))
    return record


def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = [m.name for m in members]
        require(members and len(names) == len(set(names)), "Empty archive or duplicate member")
        for member in members:
            require((member.isfile() or (member.isdir() and member.size == 0))
                    and not member.issparse() and member.size >= 0, "Archive contains nonregular member")
            name = relative(member.name)
            require(name.parts[0] in ("recovery", "original") or str(name) == "collection_observation.json",
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


def verify_cache(c):
    marker = read(report_dir() / "COLLECTION_COMPLETE.json")
    root = retrieval_dir()
    require(marker["contract_sha256"] == canonical(c)
            and sha(root / "archive.tar.gz") == marker["archive_sha256"]
            and sha(root / "manifest.json") == marker["manifest_sha256"]
            and sha(root / "receipt.json") == marker["receipt_sha256"], "Completed retrieval cache changed")
    require(sha(marker["collection_transport_path"]) == marker["collection_transport_sha256"], "Collection transport record changed")
    transfer = read(marker["collection_transport_path"])
    require(type(transfer.get("actual_transport_exit")) is int and transfer["actual_transport_exit"] == 0
            and transfer.get("transport_timeout") is False and transfer.get("transport_error") is None
            and transfer["stdout_sha256"] == marker["archive_sha256"]
            and sha(transfer["stderr_path"]) == transfer["stderr_sha256"], "Actual collection transport proof changed")
    manifest = read(root / "manifest.json")["files"]
    inventory(root / "evidence", manifest)
    actual = {str(p.relative_to(root / "evidence")) for p in (root / "evidence").rglob("*") if p.is_file()}
    require(actual == set(manifest), "Cached retrieval has missing or unbound artifacts")
    verify_terminal_evidence(c, root / "evidence")
    return {**marker, "cache_verified": True, "remote_calls": 0}


def advance(expected, *, check_local=False):
    c = manager.verify_local(expected)
    folder = report_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "COLLECTION_COMPLETE.json").exists():
        return verify_cache(c)
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
        require(read(manager.REPORT / "LAUNCHED.json")["contract_sha256"] == expected, "Approved recovery launch missing")
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:12]
        round_dir = folder / "rounds" / stamp
        round_dir.mkdir(parents=True, exist_ok=False)
        try:
            check = transport(c, terminal_program(c), round_dir / "check", label="read_only_terminal", timeout=CHECK_SECONDS)
            raw = Path(check["stdout_path"]).read_text()
            terminal = json.loads(raw)
            require(terminal.get("schema") == SCHEMA and terminal.get("contract_sha256") == expected
                    and terminal.get("read_only_check") is True and terminal.get("boot_id") == c["boot_id"]
                    and terminal.get("observed_at"), "Terminal check response changed")
            require(terminal.get("status") in ("waiting", "terminal", "terminal_failure"), "Unknown terminal state")
            result = {"schema": SCHEMA, "contract_sha256": expected, "checked_unix": time.time(),
                      "status": terminal["status"], "observation": terminal, "remote_calls": 1,
                      "transport_path": str(round_dir / "check" / "TRANSPORT.json"),
                      "transport_sha256": sha(round_dir / "check" / "TRANSPORT.json")}
            write(round_dir / "ROUND.json", result, exclusive=True)
            write(folder / "LATEST.json", result)
            require(terminal["status"] != "terminal_failure", "Native dispatcher exited unsuccessfully; not a retrieval failure")
            if terminal["status"] == "waiting":
                return result
            write(folder / "COLLECTION_ATTEMPT.json", {"at": r.now(), "contract_sha256": expected,
                "round": str(round_dir), "automatic_retry": False}, exclusive=True)
            destination = retrieval_dir()
            destination.mkdir(parents=True, exist_ok=False)
            transfer = transport(c, collection_program(c), round_dir / "collection", label="first_collection",
                                 timeout=TRANSFER_SECONDS, output=destination / "archive.tar.gz")
            safe_extract(destination / "archive.tar.gz", destination / "evidence")
            verified = verify_terminal_evidence(c, destination / "evidence")
            files = {str(p.relative_to(destination / "evidence")): sha(p)
                     for p in sorted((destination / "evidence").rglob("*")) if p.is_file()}
            write(destination / "manifest.json", {"schema": SCHEMA, "files": files}, exclusive=True)
            receipt = {"schema": SCHEMA, **verified, "status": "collection_complete", "contract_sha256": expected,
                "at": r.now(), "archive_sha256": sha(destination / "archive.tar.gz"),
                "manifest_sha256": sha(destination / "manifest.json"), "file_count": len(files),
                "collection_transport_path": str(round_dir / "collection" / "TRANSPORT.json"),
                "collection_transport_sha256": sha(round_dir / "collection" / "TRANSPORT.json"),
                "actual_collection_transport_exit": transfer["actual_transport_exit"], "automatic_retry": False}
            write(destination / "receipt.json", receipt, exclusive=True)
            complete = {**receipt, "receipt_sha256": sha(destination / "receipt.json"), "remote_calls": 2}
            write(folder / "COLLECTION_COMPLETE.json", complete, exclusive=True)
            return complete
        except Exception:
            failure = {"schema": SCHEMA, "status": "blocked", "at": r.now(), "contract_sha256": expected,
                       "round": str(round_dir), "traceback": traceback.format_exc(), "automatic_retry": False,
                       "scientific_failure_inferred_from_transport": False}
            write(folder / "FAILURE.json", failure, exclusive=True)
            raise


def comparison_records(evidence, c):
    """Return the 18 immutable MAC endpoints for the existing pure comparator."""
    verify_terminal_evidence(c, evidence)
    root = Path(evidence) / "recovery"
    return [{"dataset": job["dataset"], "split": split, "seed": job["seed"], "model": job["arm"],
             **{key: value[key] for key in ("selected_epoch", "selected_checkpoint", "checkpoint_sha256", "state_sha256", "target_identity", "metrics")}}
            for job in c["jobs"] for split in ("validation", "test")
            for value in [read(local(root, "run/" + job["id"] + "/" + split + "_receipt.json"))]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--mode", choices=("advance", "check-local"), required=True)
    parser.add_argument("--completion-bridge", action="store_true")
    args = parser.parse_args(argv)
    with controller_context(args.completion_bridge):
        result = advance(args.contract_sha, check_local=args.mode == "check-local")
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
