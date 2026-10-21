"""Finish MAC9 Validation/Test after the preserved v3 training queue drains.

This controller never launches a training/reference worker or changes the v3
attempt.  Its only exceptional admission is the pinned Popen/exec identity race,
and successful training still requires actual parent-observed exit zero.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from paper.scripts import titans_mac_accuracy9_recovery_runtime as r

MODULE = "paper.scripts.titans_mac_accuracy9_completion_bridge"
SCHEMA, STAGE_SCHEMA, STAGES = r.SCHEMA, r.STAGE_SCHEMA, r.STAGES
IMPORT_SCHEMA = "titans_mac_accuracy9_completion_import_v1"
BRIDGE_FILE = "paper/scripts/titans_mac_accuracy9_completion_bridge.py"
IDENTITY_KEYS = ("pid", "start_ticks", "pgid", "argv", "cwd")
require, read, write = r.require, r.read, r.write
sha_file, sha_json, local_path = r.sha_file, r.sha_json, r.local_path
worker_command, verify_receipt = r.worker_command, r.verify_receipt
proc_identity, gpu_state, now = r.proc_identity, r.gpu_state, r.now


def _parent_pid(pid):
    # comm may contain spaces and closing parentheses. PPid is the second field
    # after the final ')' in /proc/PID/stat.
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])


def _birth_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def initial_identity(process, command, source, start_ticks=None):
    """Allow only our own newly forked child a bounded exec handoff interval."""
    if start_ticks is not None:
        return r.bounded._identity(process, command, source, start_ticks)
    deadline = time.monotonic() + 2.0
    first_ticks = None
    while True:
        live = r.proc_identity(process.pid)
        if live is None:
            if process.poll() is not None:
                return None
            # /proc/PID/cmdline may be transiently empty during exec. It is not
            # an OS exit: retain only this direct child's stable stat identity.
            require(_parent_pid(process.pid) == os.getpid(), "Initial exec identity is not a live direct child")
            ticks = _birth_ticks(process.pid)
            if first_ticks is None:
                first_ticks = ticks
            require(ticks == first_ticks, "Initial exec child PID was reused")
            require(time.monotonic() < deadline, "Initial child exec identity handoff timed out")
            time.sleep(.01)
            continue
        require(process.poll() is None and _parent_pid(process.pid) == os.getpid(),
                "Initial exec identity is not a live direct child")
        if first_ticks is None:
            first_ticks = live["start_ticks"]
        require(live["pid"] == process.pid and live["start_ticks"] == first_ticks,
                "Initial exec child PID was reused")
        if (live["pgid"] == process.pid and live["argv"] == command
                and live["cwd"] == str(source)):
            return live
        require(time.monotonic() < deadline, "Initial child exec identity handoff timed out")
        time.sleep(.01)


def validate_continuation(c):
    r.validate_contract(c)
    binding = c["continuation_from"]
    old_root, root = Path(binding["root"]), Path(c["root"])
    require(old_root.is_absolute() and root != old_root
            and not root.is_relative_to(old_root) and not old_root.is_relative_to(root),
            "Continuation requires a separate sibling root")
    old_path = old_root / "recovery_contract.json"
    require(sha_file(old_path) == binding["contract_raw_sha256"], "v3 contract bytes changed")
    original = read(old_path)
    require(sha_json(original) == binding["contract_sha256"], "v3 contract identity changed")
    r.validate_contract(original)
    # Relocation and the separately sealed bridge are the sole admitted changes.
    expected = deepcopy(original)
    expected.update(root=c["root"], source_root=c["source_root"], environment=c["environment"])
    for key in ("continuation_from", "bridge_files", "bridge_files_sha256",
                "continuation_only", "fresh_training_authorized", "tmux_session",
                "operation_git_revision", "created_at"):
        if key not in c:
            continue
        expected[key] = c[key]
    require(c == expected, "Continuation changed a scientific/operational contract")
    require(c.get("continuation_only") is True and c.get("fresh_training_authorized") is False,
            "Bridge must forbid all new training")
    require(set(c["bridge_files"]) == {BRIDGE_FILE}
            and sha_json(c["bridge_files"]) == c["bridge_files_sha256"], "Unsealed bridge closure")
    for name, digest in c["bridge_files"].items():
        require(sha_file(local_path(Path(c["source_root"]), name)) == digest, "Bridge source changed")
    proof = binding["initial_live_observation"]
    proof_path = Path(proof["path"])
    require(proof_path.is_relative_to(root) and ".." not in proof_path.parts
            and sha_file(proof_path) == proof["sha256"], "Initial live observation bytes changed")
    observed = read(proof_path)
    require(observed["boot_id"] == c["boot_id"] and len(observed["jobs"]) == 3,
            "Initial live observation boot/matrix changed")
    require(observed["status"]["status"] == "draining_after_failure"
            and observed["status"]["stage"] == "train"
            and observed["status"]["pending_job_ids"] == []
            and observed["status"]["contract_sha256"] == sha_json(original),
            "Initial observation does not prove the exact draining queue")
    _race_failure(observed["status"]["failure"])
    identities = {}
    for row in observed["jobs"]:
        key, identity = row["job_id"], row["actual_identity"]
        job = next(j for j in c["jobs"] if j["id"] == key)
        expected_command = r.worker_command(original, old_path, job, "train")
        require(key in c["training_jobs"] and key not in identities
                and row["exact_command_cwd_pgid_match"] is True
                and identity["argv"] == expected_command and identity["cwd"] == original["source_root"]
                and identity["pid"] == identity["pgid"] and type(identity["start_ticks"]) is int
                and identity["start_ticks"] > 0, "Unproven original training child identity")
        identities[key] = identity
    require(set(identities) == set(c["training_jobs"]), "Initial training identity matrix changed")
    return original, identities


def verify_files(c):
    original, identities = validate_continuation(c)
    r.verify_files(c)
    r.verify_files(original)
    return original, identities


def _race_failure(failure):
    require(failure.get("stage") == "train" and not failure.get("job_id")
            and failure.get("traceback", "").rstrip().endswith("ValueError: Owned child identity changed")
            and "in _identity" in failure["traceback"], "v3 failure is outside the admitted initial exec race")


def _success_exit(exited, *, stage, key):
    require(type(exited.get("actual_os_returncode")) is int and exited["actual_os_returncode"] == 0
            and exited.get("timeout") is False and exited.get("stage") == stage
            and exited.get("job_id") == key and type(exited.get("pid")) is int
            and exited["pid"] > 0 and exited.get("started_at") and exited.get("observed_exit_at"),
            "Original child has no actual successful OS exit")


def _ticks_proven(c, old, key, exited, identities):
    identity = identities[key]
    job = next(j for j in old["jobs"] if j["id"] == key)
    require(exited["pid"] == identity["pid"] and exited["boot_id"] == c["boot_id"]
            and exited["command"] == r.worker_command(old, Path(old["root"]) / "recovery_contract.json", job, "train"),
            "Actual training exit is not the independently observed child")
    ticks = exited.get("start_ticks")
    require(ticks is None or ticks == identity["start_ticks"], "Actual training exit PID ticks differ")
    return identity["start_ticks"]


def historical_identity_absent(exited, ticks=None):
    """PID reuse is another process, never historical worker ownership."""
    ticks = exited.get("start_ticks") if ticks is None else ticks
    require(type(ticks) is int and ticks > 0, "Historical exit has no proven process birth")
    live = r.proc_identity(exited["pid"])
    return live is None or live["start_ticks"] != ticks


def _read_ready_json(path, c):
    """An exclusive writer exposes its empty pathname before its first write.

    Only incomplete JSON gets a short local grace. A malformed, settled file is
    retained byte-for-byte and blocks; it is never interpreted as a fit failure.
    """
    began = time.monotonic()
    while True:
        raw = path.read_bytes()
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if time.monotonic() - began < .2:
                time.sleep(.01)
                continue
            if time.time() - path.stat().st_mtime < 2.0:
                return None
            destination = Path(c["root"]) / "upstream_invalid_json.raw"
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(raw)
                write(Path(c["root"]) / "upstream_invalid_json.json", {"original_path": str(path),
                    "observed_at": r.now(), "preserved_raw_sha256": sha_file(destination),
                    "original_workers_signalled": False}, exclusive=True)
            raise ValueError("Settled upstream JSON is invalid; raw bytes preserved")


def terminal_ready(c, old, identities):
    """No SSH and no process signaling; false means the exact v3 queue is draining."""
    root = Path(old["root"])
    require(r.boot_id() == c["boot_id"], "Boot changed while waiting for original children")
    require(time.time() < c["absolute_host_deadline_unix"], "Original absolute host deadline expired")
    failure_path = root / "dispatch_failure.json"
    require(failure_path.exists(), "Expected preserved v3 identity-race failure is missing")
    failure = _read_ready_json(failure_path, c)
    if failure is None:
        return False
    _race_failure(failure)
    proof = c["continuation_from"]["initial_live_observation"]
    require(sha_file(proof["path"]) == proof["sha256"]
            and {k: v for k, v in failure.items() if k != "children_left_running"}
                == read(proof["path"])["status"]["failure"]
            and failure.get("children_left_running") == {k: v["pid"] for k, v in identities.items()},
            "Original failure differs from the pinned initial exec race")
    for key, identity in identities.items():
        exit_path = root / "run" / key / "train_exit.json"
        if exit_path.exists():
            exited = _read_ready_json(exit_path, c)
            if exited is None:
                return False
            try:
                _success_exit(exited, stage="train", key=key)
                _ticks_proven(c, old, key, exited, identities)
            except Exception:
                destination = Path(c["root"]) / "upstream_failure_observation.json"
                if not destination.exists():
                    write(destination, {"observed_at": r.now(), "original_path": str(exit_path),
                        "original_sha256": sha_file(exit_path), "actual_parent_exit_record": exited,
                        "original_workers_signalled": False}, exclusive=True)
                raise
        live = r.proc_identity(identity["pid"])
        if live is not None:
            matches = all(live[k] == identity[k] for k in IDENTITY_KEYS)
            require(matches or (exit_path.exists() and live["start_ticks"] != identity["start_ticks"]),
                    "Previously observed training PID changed without an actual exit")
    final_path, dispatcher_path = root / "final_receipt.json", root / "dispatcher_exit.json"
    if not (final_path.exists() and dispatcher_path.exists()):
        return False
    final, dispatcher = _read_ready_json(final_path, c), _read_ready_json(dispatcher_path, c)
    if final is None or dispatcher is None:
        return False
    require(final["schema"] == r.SCHEMA and final["status"] == "failed"
            and final["contract_sha256"] == sha_json(old) and set(final["stages"]) == {"reference", "train"}
            and final.get("held_out_test_evaluated") is False, "Unexpected original terminal state")
    require(type(dispatcher.get("actual_os_returncode")) is int and dispatcher["actual_os_returncode"] == 1
            and dispatcher.get("timeout") is False and dispatcher.get("cleanup_error") is None,
            "Original dispatcher did not exit solely after the drained identity race")
    launch = read(root / "dispatcher_launch.json")
    require(launch["contract_sha256"] == sha_json(old) and launch["identity"]
            and dispatcher["pid"] == launch["identity"]["pid"]
            and dispatcher["start_ticks"] == launch["identity"]["start_ticks"]
            and dispatcher["command"] == launch["command"] and dispatcher["boot_id"] == c["boot_id"],
            "Original dispatcher exit identity changed")
    for stream in ("stdout", "stderr"):
        path = root / ("dispatcher." + stream + ".log")
        require(dispatcher[stream + "_sha256"] == sha_file(path)
                and dispatcher[stream + "_bytes"] == path.stat().st_size, "Original dispatcher log changed")
    require(historical_identity_absent(dispatcher), "Original dispatcher remains alive")
    for stage, keys in (("reference", old["reference_jobs"]), ("train", old["training_jobs"])):
        result = final["stages"][stage]
        require(result["status"] == ("complete" if stage == "reference" else "failed")
                and set(result["exits"]) == set(result["receipts"]) == set(keys)
                and result.get("pending_job_ids") == [], "Original stage did not drain all successful children")
        if stage == "train":
            _race_failure(result["failure"])
        for key in keys:
            job = next(j for j in old["jobs"] if j["id"] == key)
            folder = root / "run" / key
            exited = read(folder / (stage + "_exit.json"))
            require(exited == result["exits"][key], "Original exit differs from its terminal receipt")
            _success_exit(exited, stage=stage, key=key)
            if stage == "train":
                ticks = _ticks_proven(c, old, key, exited, identities)
            else:
                ticks = exited.get("start_ticks")
            require(historical_identity_absent(exited, ticks), "Successful original worker remains alive")
            require(r.verify_receipt(old, job, stage) == result["receipts"][key], "Original result binding changed")
    r.gpu_state(c)
    return True


def _copy(source, destination, digest=None):
    require(not source.is_symlink() and source.is_file(), "Only regular original files may be copied")
    expected = sha_file(source) if digest is None else digest
    require(sha_file(source) == expected, "Original file changed before copy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    require(sha_file(source) == sha_file(destination) == expected, "Original file changed while copying")
    return expected


def _relocate(value, old_root, new_root):
    """Relocate only absolute paths below the old attempt, preserving other data."""
    if isinstance(value, dict):
        return {k: _relocate(v, old_root, new_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_relocate(v, old_root, new_root) for v in value]
    if isinstance(value, str) and value.startswith(str(old_root) + "/"):
        return str(new_root / Path(value).relative_to(old_root))
    return value


def import_completed(c, old, identities):
    """Preserve raw v3 evidence, then bind copies of the same nine selected fits."""
    from paper.scripts import run_titans_mac_accuracy9_recovery as worker
    science = worker.science
    root, old_root = Path(c["root"]), Path(old["root"])
    require(terminal_ready(c, old, identities), "Original queue is still draining")
    require(not (root / "run").exists() and not (root / "continuation_v3").exists(), "Existing import attempt; no retry")
    origin_files = {}
    for path in sorted(old_root.rglob("*")):
        relative = path.relative_to(old_root)
        if relative.parts[0] in ("source", "cache") or not path.is_file():
            continue
        origin_files[str(relative)] = _copy(path, root / "continuation_v3" / relative)
    copied_original = read(root / "continuation_v3/recovery_contract.json")
    require(copied_original == old, "Copied original contract differs")
    original_science = worker.effective_scientific(old)
    relocated_science = worker.effective_scientific(c)
    stages = {s: {"stage": s, "status": "complete", "exits": {}, "receipts": {},
                  "failure": None, "pending_job_ids": [], "executed_in_root": str(old_root),
                  "imported_without_execution": True} for s in ("reference", "train")}
    imported = {"schema": IMPORT_SCHEMA, "status": "complete", "created_at": r.now(),
                "original_contract": {"path": "continuation_v3/recovery_contract.json",
                    "sha256": sha_file(root / "continuation_v3/recovery_contract.json"),
                    "canonical_sha256": sha_json(old)}, "origin_files": origin_files,
                "new_training_worker_count": 0, "jobs": {}}
    for job in c["jobs"]:
        key = job["id"]; stage = "train" if key in c["training_jobs"] else "reference"
        folder, old_folder = root / "run" / key, old_root / "run" / key
        old_train = read(old_folder / "train_receipt.json")
        reference_job = next(j for j in original_science["jobs"] if j["id"] == key)
        science.verify_train_receipt(original_science, "5090", reference_job, old_root, old_train)
        row = {"stage": stage, "files": {}}
        # Operational claims/leases always stay raw in continuation_v3. Reusing
        # them as new execution leases would falsely imply new training ownership.
        for path in sorted(old_folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(old_root)
            if path.parent == old_folder:
                replaced = {"train_receipt.json", stage + "_recovery_receipt.json",
                            stage + "_exit.json", stage + "_launch.json"}
                if stage == "reference":
                    replaced.add("train_exit.json")
                if path.name in replaced or path.name.endswith(("_lease.json", "_claim.json")):
                    continue
            row["files"][str(relative)] = _copy(path, root / relative, origin_files[str(relative)])
        old_wrapper = read(old_folder / (stage + "_recovery_receipt.json"))
        relocated = {**old_train, **science.base_receipt(relocated_science, "5090", reference_job, "train"),
                     "selected_checkpoint": _relocate(old_train["selected_checkpoint"], old_root, root),
                     "summary": {**old_train["summary"], "checkpoint_path": _relocate(old_train["summary"]["checkpoint_path"], old_root, root)},
                     "validation_reference": {**old_train["validation_reference"], "prediction_path": _relocate(old_train["validation_reference"]["prediction_path"], old_root, root)}}
        science.verify_train_receipt(relocated_science, "5090", reference_job, root, relocated)
        write(folder / "train_receipt.json", relocated, exclusive=True)
        for kind, name in (("scientific_receipt", "train_receipt.json"),):
            row[kind] = {"original_path": "continuation_v3/run/" + key + "/" + name,
                         "original_sha256": origin_files["run/" + key + "/" + name],
                         "relocated_path": "run/" + key + "/" + name,
                         "relocated_sha256": sha_file(folder / name)}
        # Preserve the original historical Train exit as a byte-identical copy
        # for the six reference jobs. The reference worker's own exit is separate.
        if stage == "reference":
            _copy(old_folder / "train_exit.json", folder / "train_exit.json", origin_files["run/" + key + "/train_exit.json"])
        exited = read(old_folder / (stage + "_exit.json"))
        launch_path = old_folder / (stage + "_launch.json")
        supplemented = stage == "train" and exited.get("start_ticks") is None
        if stage == "train":
            ticks = _ticks_proven(c, old, key, exited, identities)
            exited = {**exited, "start_ticks": ticks}
        if launch_path.exists():
            launch = read(launch_path)
        else:
            require(supplemented, "Original successful launch is missing")
            launch = {k: exited[k] for k in ("job_id", "stage", "pid", "start_ticks", "command", "started_at", "boot_id")}
        if supplemented:
            launch = {**launch, "start_ticks": exited["start_ticks"]}
            proof = c["continuation_from"]["initial_live_observation"]
            row["start_ticks_proof"] = {"observation_path": str(Path(proof["path"]).relative_to(root)),
                "observation_sha256": proof["sha256"], "identity": identities[key]}
        for kind, value in (("exit", exited), ("launch", launch)):
            name = stage + "_" + kind + ".json"
            write(folder / name, value, exclusive=True)
            row[kind] = {"original_path": "continuation_v3/run/" + key + "/" + name,
                         "original_sha256": origin_files.get("run/" + key + "/" + name),
                         "relocated_path": "run/" + key + "/" + name,
                         "relocated_sha256": sha_file(folder / name), "identity_supplemented": supplemented}
            if kind == "launch" and not launch_path.exists():
                row[kind].update(original_path=None, reconstructed_from_initial_live_observation=True)
        for stream in ("stdout", "stderr"):
            name = "logs/" + key + "." + stage + "." + stream + ".log"
            row["files"][name] = _copy(old_root / name, root / name, origin_files[name])
            require(exited[stream + "_sha256"] == sha_file(root / name)
                    and exited[stream + "_bytes"] == (root / name).stat().st_size, "Closed original worker log changed")
        wrapper = {**relocated, "schema": r.STAGE_SCHEMA, "stage": stage,
            "contract_sha256": sha_json(c), "scientific_contract_sha256": old_wrapper["scientific_contract_sha256"],
            "scientific_contract_raw_sha256": old_wrapper["scientific_contract_raw_sha256"],
            "effective_scientific_contract_sha256": sha_json(relocated_science),
            "recovery_mode": job["recovery_mode"], "lineage": _relocate(old_wrapper["lineage"], old_root, root),
            "original_files_modified": False, "scientific_worker_source_unchanged": True,
            "metrics": relocated["validation_reference"]["metrics"]}
        name = stage + "_recovery_receipt.json"
        wrapper["files"] = science.file_inventory(root, [p for p in folder.rglob("*") if p.is_file() and p.name != name])
        write(folder / name, wrapper, exclusive=True)
        row["recovery_receipt"] = {"original_path": "continuation_v3/run/" + key + "/" + name,
            "original_sha256": origin_files["run/" + key + "/" + name],
            "relocated_path": "run/" + key + "/" + name, "relocated_sha256": sha_file(folder / name)}
        stages[stage]["exits"][key] = exited
        stages[stage]["receipts"][key] = r.verify_receipt(c, job, stage)
        imported["jobs"][key] = row
    write(root / "import_receipt.json", imported, exclusive=True)
    for name, digest in origin_files.items():
        require(sha_file(local_path(old_root, name)) == digest, "v3 original changed during import")
    return stages


def preflight(c):
    old, identities = verify_files(c)
    allowed = []
    for identity in identities.values():
        live = r.proc_identity(identity["pid"])
        if live is not None:
            require(all(live[k] == identity[k] for k in IDENTITY_KEYS), "Existing training identity changed")
            allowed.append(identity["pid"])
    return {"status": "passed", "new_training_worker_count": 0, "gpu": r.gpu_state(c, allowed),
            "original_root": old["root"], "live_original_training_pids": allowed}


def dispatch(c, path):
    old, identities = verify_files(c)
    root, source = Path(c["root"]), Path(c["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Wrong bridge cwd")
    require(not any((root / n).exists() for n in ("dispatch_attempt.json", "run", "final_receipt.json", "dispatch_failure.json")),
            "Existing bridge attempt; no retry")
    write(root / "dispatch_attempt.json", {"at": r.now(), "contract_sha256": sha_json(c), "new_training_worker_count": 0}, exclusive=True)
    write(root / "start.json", {"at": r.now(), "pid": os.getpid(), "identity": r.proc_identity(os.getpid()),
        "boot_id": r.boot_id(), "contract_sha256": sha_json(c)}, exclusive=True)
    began = time.monotonic()
    while not terminal_ready(c, old, identities):
        require(time.monotonic() - began < c["limits"]["queue_wall_seconds"], "Inherited queue deadline expired")
        write(root / "status.json", {"status": "waiting_for_preserved_training", "observed_at": r.now(),
            "original_root": old["root"], "new_training_worker_count": 0, "contract_sha256": sha_json(c)})
        time.sleep(min(30.0, max(.001, c["absolute_host_deadline_unix"] - time.time())))
    verify_files(c)
    stages = import_completed(c, old, identities)
    (root / "logs").mkdir(exist_ok=True)
    # Only this new process uses the safer Popen handoff; preserved source bytes
    # and the old dispatcher process are never modified.
    previous_identity = r._identity
    r._identity = initial_identity
    try:
        for stage in ("validation", "test"):
            verify_files(c)
            stages[stage] = r.run_stage(c, path, stage, began)
            if stages[stage]["status"] != "complete":
                break
            if stage == "validation":
                r.validation_gate(c, stages)
    finally:
        r._identity = previous_identity
    gpu = r.gpu_state(c)
    absent = all(historical_identity_absent(exited) for stage in stages.values() for exited in stage["exits"].values())
    verify_files(c)
    complete = set(stages) == set(r.STAGES) and all(v["status"] == "complete" for v in stages.values()) and absent
    final = {"schema": r.SCHEMA, "status": "complete" if complete else "failed", "completed_at": r.now(),
        "contract_sha256": sha_json(c), "stages": stages, "resumed_fit_count": 3,
        "preserved_completed_fit_count": 6, "new_training_worker_count": 0,
        "current_new_training_count": 0, "imported_completed_fit_count": 9,
        "imported_resumed_fit_count": 3, "imported_reference_copy_exit0_count": 6,
        "validation_replay_count": len(stages.get("validation", {}).get("receipts", {})),
        "candidate_test_count": len(stages.get("test", {}).get("receipts", {})),
        "held_out_test_evaluated": complete, "gpu_after": gpu, "owned_worker_pids_absent": absent,
        "immutable_originals_verification": r.verify_originals(c), "dispatcher_own_exit_directly_observed": False,
        "continuation_from": c["continuation_from"],
        "import_receipt": {"path": "import_receipt.json", "sha256": sha_file(root / "import_receipt.json")}}
    final["artifact_manifest"] = r.artifact_manifest(root)
    write(root / "final_receipt.json", final, exclusive=True)
    return 0 if complete else 1


def supervise(c, path):
    preflight(c)
    root, source = Path(c["root"]), Path(c["source_root"])
    write(root / "launch_attempt.json", {"at": r.now(), "contract_sha256": sha_json(c)}, exclusive=True)
    command = [c["python"], "-u", "-m", MODULE, "--contract", str(path), "--contract-sha", sha_json(c), "--mode", "dispatch"]
    prefix, started = root / "dispatcher", r.now()
    with prefix.with_name("dispatcher.stdout.log").open("xb") as out, prefix.with_name("dispatcher.stderr.log").open("xb") as err:
        child = subprocess.Popen(command, cwd=source, env={**os.environ, **c["environment"], "PYTHONDONTWRITEBYTECODE": "1"},
                                 stdout=out, stderr=err, start_new_session=True)
    identity = initial_identity(child, command, source)
    ticks = identity["start_ticks"] if identity else None
    write(root / "dispatcher_launch.json", {"at": started, "identity": identity, "command": command,
        "contract_sha256": sha_json(c)}, exclusive=True)
    timed_out, cleanup_error, cleanup_attempted = False, None, False
    def cleanup_once():
        nonlocal cleanup_error, cleanup_attempted
        cleanup_attempted = True
        try:
            r.cleanup_leased_workers(c, path)
        except Exception:
            cleanup_error = traceback.format_exc()
    try:
        child.wait(timeout=max(.001, min(c["limits"]["queue_wall_seconds"], c["absolute_host_deadline_unix"] - time.time())))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # v4 has evaluation leases only. Imported training leases are never
            # installed under run/, so cleanup cannot touch the v3 fits.
            cleanup_once()
            initial_identity(child, command, source, ticks)
            r.terminate_owned_child(child)
    if child.returncode != 0 or timed_out:
        if not cleanup_attempted:
            cleanup_once()
    record = r._exit_record(c, child, prefix, command, timeout=timed_out, started_at=started, start_ticks=ticks)
    record["cleanup_error"] = cleanup_error
    write(root / "dispatcher_exit.json", record, exclusive=True)
    return 0 if child.returncode == 0 and not timed_out and cleanup_error is None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--mode", choices=("preflight", "supervise", "dispatch"), required=True)
    args = parser.parse_args(argv)
    contract = read(args.contract)
    require(sha_json(contract) == args.contract_sha, "Pinned bridge contract changed")
    if args.mode == "preflight":
        import json
        print(json.dumps(preflight(contract), allow_nan=False)); return 0
    return supervise(contract, args.contract) if args.mode == "supervise" else dispatch(contract, args.contract)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
