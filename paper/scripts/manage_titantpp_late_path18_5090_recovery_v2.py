"""Finite, isolated recovery of the four 5090 fits interrupted by reboot.

The original scientific contract and every original file remain immutable.
Only this new operation contract and its new root can acquire new artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import time
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_mlp16_late_path18_5090_recovery_20261019_v2"
FAILED_NAME = "titantpp_mlp16_late_path18_5090_recovery_20261019_v1"
FAILURE_DIAGNOSIS_SHA256 = "08565e0fce203533fd311090f00b0a7e45bb005dc9ef541d9193638836bdcb26"
REPORT = ROOT / "reports" / NAME
BUNDLE = ROOT / "search_artifacts" / NAME
ORIGIN = ROOT / "search_artifacts" / FAILED_NAME / "origin_archive"
OLD_NAME = "titantpp_mlp16_late_path18_20261019_v2"
OPERATIONS = (
    "paper/scripts/manage_titantpp_late_path18_5090_recovery_v2.py",
    "paper/scripts/titantpp_late_path18_5090_recovery_runtime.py",
    "paper/scripts/run_titantpp_late_path18_recovery.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_origin():
    manifest = read(ORIGIN / "manifest.json")
    require(manifest["files"] and sha(ORIGIN / "originals.tar.gz") == manifest["archive_sha256"], "Original archive changed")
    for name, digest in manifest["files"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "Invalid original path")
        require(sha(ORIGIN / "originals" / path) == digest, "Original copy changed: " + name)
    science = read(ORIGIN / "originals/execution_contract.json")
    require(canonical(manifest["files"]) == manifest["files_sha256"], "Original inventory changed")
    require(sha(ORIGIN / "originals/execution_contract.json") == manifest["source_contract_raw_sha256"], "Original contract changed")
    return manifest, science


def failed_predecessor_evidence():
    """Bind the preserved import failure; it was not a scientific training run."""
    diagnosis_path = ROOT / "reports" / FAILED_NAME / "qualification_failure_diagnosis.json"
    require(sha(diagnosis_path) == FAILURE_DIAGNOSIS_SHA256, "Preserved qualification diagnosis changed")
    diagnosis = read(diagnosis_path)
    require(diagnosis.get("read_only") is True and diagnosis.get("training_run_exists") is False
            and diagnosis.get("gpu_processes") == {"exit": 0, "stdout": "", "stderr": ""}, "Qualification failure scope changed")
    files = diagnosis["qualification_files"]
    for name, value in files.items():
        raw = value["text"].encode()
        require(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"],
                "Preserved failure raw text changed: " + name)
    direct_exit = json.loads(files["qualification_exit.json"]["text"])
    require(direct_exit.get("actual_os_returncode") == 1 and direct_exit.get("timeout") is False
            and "Could not locate the paper_research project root" in files["qualification_titantpp_mlp16_late_observation_v1_0.stderr.log"]["text"],
            "Unexpected prior native qualification failure")
    bundle = ROOT / "search_artifacts" / FAILED_NAME
    prior = read(bundle / "recovery_contract.json")
    return {"campaign": FAILED_NAME, "root": prior["root"],
        "contract_raw_sha256": sha(bundle / "recovery_contract.json"), "contract_canonical_sha256": canonical(prior),
        "diagnosis_path": str(diagnosis_path), "diagnosis_sha256": FAILURE_DIAGNOSIS_SHA256,
        "preserved_failure_files": {name: value["sha256"] for name, value in files.items()},
        "qualification_actual_os_returncode": 1, "real_data_training_started": False,
        "confirmed_cause": "Deployment omitted source/sample_data required by the frozen project-root resolver",
        "repair": "Include only the empty source/sample_data directory in a new isolated deployment",
        "scientific_source_changed": False, "original_attempt_retried": False}


def package_deployment(contract, contract_path):
    """Export unchanged science and inputs, plus its required empty root marker."""
    require(canonical(read(contract_path)) == canonical(contract), "Packaged recovery contract differs")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        marker = tarfile.TarInfo("source/sample_data")
        marker.type, marker.mode, marker.mtime = tarfile.DIRTYPE, 0o755, 0
        archive.addfile(marker)
        for name, digest in {**contract["source"]["files"], **contract["operation_files"]}.items():
            path = ROOT / name if name in contract["operation_files"] else ORIGIN / "originals/source" / name
            require(sha(path) == digest, "Packaged source bytes changed: " + name)
            archive.add(path, arcname="source/" + name, recursive=False)
        for name, digest in contract["input_files"].items():
            path = ORIGIN / "originals/source" / name
            require(sha(path) == digest, "Recovery input bytes changed: " + name)
            archive.add(path, arcname="source/" + name, recursive=False)
        archive.add(contract_path, arcname="recovery_contract.json", recursive=False)
    return buffer.getvalue()


def installation_code(root, deployment_sha256):
    """The same bounded, explicit-directory extractor is tested and sent remotely."""
    return "\n".join((
        "import io,tarfile,sys,json,hashlib", "from pathlib import Path", "p=Path(" + repr(str(root)) + ")",
        "assert not p.exists(),'Existing recovery root'", "b=sys.stdin.buffer.read()",
        "assert hashlib.sha256(b).hexdigest()==" + repr(deployment_sha256),
        "t=tarfile.open(fileobj=io.BytesIO(b),mode='r:gz');members=t.getmembers()",
        "assert len({m.name for m in members})==len(members),'Duplicate deployment member'",
        "assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (m.isfile() or (m.isdir() and m.name.rstrip('/')=='source/sample_data')) for m in members),'Unsafe deployment member'",
        "assert any(m.isdir() and m.name.rstrip('/')=='source/sample_data' for m in members),'Missing project-root marker'",
        "p.mkdir();t.extractall(p,filter='data')", "print(json.dumps({'status':'installed','root':str(p)}))"))


def prepare():
    from paper.scripts import manage_titantpp_late_path18_v2 as parent
    original = parent.verify_local(archive=True)
    require(parent.source_snapshot()["files_sha256"] == original["source"]["files_sha256"], "Original scientific sources changed")
    manifest, science = verify_origin()
    require(science == original, "Archived and prepared scientific contracts differ")
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *OPERATIONS], cwd=ROOT, text=True).splitlines())
    require(tracked == set(OPERATIONS), "Commit recovery operation files before preparing")
    require(not subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *OPERATIONS], cwd=ROOT, text=True).strip(), "Uncommitted recovery source changes")
    old = science["hosts"]["5090"]
    new_root = str(Path(old["root"]).parent / (NAME + "_5090"))
    jobs = []
    for job in science["jobs"]:
        if job["host"] != "5090":
            continue
        item = dict(job)
        prefix = "run/" + job["id"] + "/"
        if job["dataset"] == "raf_spare_parts":
            receipt_name = prefix + "train_receipt.json"
            receipt = read(ORIGIN / "originals" / receipt_name)
            checkpoint = receipt["selected_checkpoint"]
            relative = str(Path(checkpoint).relative_to(old["root"]))
            require(manifest["files"][relative] == receipt["checkpoint_sha256"], "RAF selected checkpoint changed")
            item.update(recovery_mode="reference", origin_train_receipt={"path": old["root"] + "/" + receipt_name, "sha256": manifest["files"][receipt_name]},
                        origin_selected_checkpoint={"path": checkpoint, "sha256": receipt["checkpoint_sha256"]})
        elif job["id"] in manifest["checkpoint_metadata"]:
            metadata = manifest["checkpoint_metadata"][job["id"]]
            require(metadata["scalars"]["checkpoint_type"] == "epoch_resume" and metadata["scalars"]["checkpoint_schema_version"] == 2, "Incomplete epoch resume state")
            launch = read(ORIGIN / "originals" / (prefix + "train_launch.json"))
            item.update(recovery_mode="resume", origin_resume_checkpoint={"path": metadata["path"], "sha256": metadata["sha256"]},
                        original_started_at=launch["started_at"], resume_epoch=metadata["scalars"]["epoch"])
        else:
            require(not any(name.startswith(prefix) for name in manifest["files"]), "Fresh recovery job has existing artifacts")
            item.update(recovery_mode="fresh")
        jobs.append(item)
    require(len(jobs) == 10 and sum(j["recovery_mode"] == "reference" for j in jobs) == 6
            and sum(j["recovery_mode"] == "resume" for j in jobs) == 3
            and sum(j["recovery_mode"] == "fresh" for j in jobs) == 1, "Recovery scope changed")
    environment = {**old["environment"], "PYTHONPATH": new_root + "/source", "MPLCONFIGDIR": new_root + "/cache/matplotlib", "XDG_CACHE_HOME": new_root + "/cache"}
    operation_files = {name: sha(ROOT / name) for name in OPERATIONS}
    contract = {
        "schema": "titantpp_late_path18_5090_recovery_v1", "campaign": NAME, "created_at": now(), "host": "5090",
        "root": new_root, "source_root": new_root + "/source", "python": old["python"], "gpu_uuid": old["gpu_uuid"],
        "environment": environment, "runtime_expected": old["runtime_expected"], "boot_id": "ce4f7a36-ed56-4abe-8c14-1c4b6aeb48d6",
        "absolute_host_deadline_unix": old["absolute_host_deadline_unix"], "limits": science["limits"], "max_workers": 3,
        "scientific_contract": {"path": old["root"] + "/execution_contract.json", "raw_sha256": manifest["source_contract_raw_sha256"], "canonical_sha256": canonical(science)},
        "source": {"files": science["source"]["files"], "files_sha256": science["source"]["files_sha256"], "source_revision": environment["SOURCE_REVISION"]},
        "input_files": science["input_files"],
        "operation_files": operation_files, "operation_files_sha256": canonical(operation_files),
        "operation_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "immutable_originals": {"root": old["root"], "files": manifest["files"], "files_sha256": canonical(manifest["files"])},
        "jobs": jobs, "training_jobs": [j["id"] for j in jobs if j["recovery_mode"] != "reference"], "validation_jobs": [j["id"] for j in jobs],
        "common_condition": science["common_condition"], "evaluation_scope": "validation_only", "held_out_test_authorized": False,
        "held_out_test_evaluated": False,
        "authorization": "2026-10-19 user explicitly approved root-cause diagnosis and recovery of remaining Intermittent4 on5090, followed by Validation replay and original retrieval; old originals/results/checkpoints remain immutable",
        "root_cause": {"confirmed": "5090 reboot at2026-10-19 17:24:09 KST", "reboot_trigger": "not established; privileged system journal unavailable", "original_unknown_exits_preserved": True},
        "origin_archive_binding": {"manifest_sha256": sha(ORIGIN / "manifest.json"), "archive_sha256": manifest["archive_sha256"]},
        "failed_predecessor": failed_predecessor_evidence(),
        "deployment_layout": {"required_empty_directories": ["source/sample_data"], "scientific_source_changed": False},
    }
    from paper.scripts import titantpp_late_path18_5090_recovery_runtime as runtime
    runtime.validate_contract(contract, science=science)
    write(BUNDLE / "recovery_contract.json", contract)
    with (BUNDLE / "deployment.tar.gz").open("xb") as handle:
        handle.write(package_deployment(contract, BUNDLE / "recovery_contract.json"))
    ready = {"status": "prepared", "created_at": now(), "contract_raw_sha256": sha(BUNDLE / "recovery_contract.json"),
             "contract_canonical_sha256": canonical(contract), "deployment_sha256": sha(BUNDLE / "deployment.tar.gz"),
             "training_jobs": 4, "resumed_fits": 3, "fresh_fits": 1, "historical_raf_retrained": False, "validation_replays": 10,
             "held_out_test_evaluated": False, "operation_files": operation_files}
    write(REPORT / "PREPARED.json", ready)
    return ready


def verify_local():
    prepared = read(REPORT / "PREPARED.json")
    require(sha(BUNDLE / "recovery_contract.json") == prepared["contract_raw_sha256"] and sha(BUNDLE / "deployment.tar.gz") == prepared["deployment_sha256"], "Recovery preparation changed")
    contract = read(BUNDLE / "recovery_contract.json")
    require(canonical(contract) == prepared["contract_canonical_sha256"], "Recovery canonical contract changed")
    require(contract.get("failed_predecessor") == failed_predecessor_evidence()
            and contract.get("deployment_layout") == {"required_empty_directories": ["source/sample_data"], "scientific_source_changed": False},
            "Preserved failure or isolated directory repair binding changed")
    for name, digest in contract["operation_files"].items():
        require(sha(ROOT / name) == digest, "Recovery operation source changed: " + name)
    _, science = verify_origin()
    from paper.scripts import titantpp_late_path18_5090_recovery_runtime as runtime
    runtime.validate_contract(contract, science=science)
    return contract


def ssh(command, *, stdin=None, timeout=180):
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6", "5090", command], input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return SimpleNamespace(returncode=None, stdout=error.stdout or b"", stderr=error.stderr or b"", timed_out=True)


def transport(phase, result):
    folder = BUNDLE / "transport" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + phase)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "stdout").write_bytes(result.stdout)
    (folder / "stderr").write_bytes(result.stderr)
    write(folder / "receipt.json", {"observed_at": now(), "actual_transport_exit": result.returncode,
        "transport_timeout": getattr(result, "timed_out", False),
        "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr), "stdout_sha256": sha(folder / "stdout"), "stderr_sha256": sha(folder / "stderr")})
    require(result.returncode == 0, "Recovery " + phase + " failed; original transport preserved at " + str(folder))
    return result.stdout


def invocation(contract, mode):
    environment = " ".join(shlex.quote(key + "=" + value) for key, value in contract["environment"].items())
    return "cd " + shlex.quote(contract["source_root"]) + " && env " + environment + " " + shlex.quote(contract["python"]) + " -u -m paper.scripts.titantpp_late_path18_5090_recovery_runtime --contract " + shlex.quote(contract["root"] + "/recovery_contract.json") + " --mode " + mode


def deploy():
    contract = verify_local()
    write(REPORT / "deployment.intent.json", {"at": now(), "root": contract["root"], "automatic_retry": False})
    code = installation_code(contract["root"], sha(BUNDLE / "deployment.tar.gz"))
    remote = "env PYTHONDONTWRITEBYTECODE=1 " + shlex.quote(contract["python"]) + " -c " + shlex.quote(code)
    installed = json.loads(transport("install", ssh(remote, stdin=(BUNDLE / "deployment.tar.gz").read_bytes())))
    tmux = "/opt/miniconda3/envs/ai_env/bin/tmux"
    session = "late_path18_recovery_5090_20261019_v2"
    launch = invocation(contract, "launch_dispatch") + " > " + shlex.quote(contract["root"] + "/launcher.stdout.log") + " 2> " + shlex.quote(contract["root"] + "/launcher.stderr.log")
    transport("launch", ssh(shlex.quote(tmux) + " new-session -d -s " + shlex.quote(session) + " " + shlex.quote(launch)))
    receipt = {"status": "isolated_pipeline_dispatched", "at": now(), "installed": installed, "tmux": session, "contract_raw_sha256": sha(BUNDLE / "recovery_contract.json"), "native_qualification_and_training_not_yet_asserted": True}
    write(REPORT / "LAUNCHED.json", receipt)
    return receipt


def status():
    contract = verify_local()
    value = json.loads(transport("status", ssh(invocation(contract, "status"))))
    path = REPORT / "observations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    write(path, value)
    return {"path": str(path), "sha256": sha(path), "snapshot": value}


def verify_terminal_evidence(contract, execution_root=None, origin_root=None, *, check_live=False):
    """Rehash terminal evidence, optionally remapping an immutable local archive.

    Archived absolute paths keep their original meaning.  Only their filesystem
    lookup is remapped; no receipt, checkpoint or original contract is rewritten.
    Process absence is a fresh assertion only when ``check_live`` is requested.
    """
    import math
    from paper.scripts import titantpp_late_path18_5090_recovery_runtime as runtime
    from paper.scripts.titantpp_quantity_loss_mlp18_runtime import verify_time_head_proof

    remote = Path(contract["root"])
    original_remote = Path(contract["immutable_originals"]["root"])
    root = Path(execution_root) if execution_root is not None else remote
    originals = Path(origin_root) if origin_root is not None else original_remote
    required_files, owned = set(), set()

    def relative(name):
        p = Path(name)
        require(not p.is_absolute() and p.parts and ".." not in p.parts and str(p) != ".", "Unsafe terminal artifact path")
        return p

    def local(name, *, original=False):
        base = originals if original else root
        p = base / relative(name)
        require(p.resolve().is_relative_to(base.resolve()) and not p.is_symlink(), "Artifact escaped archive root")
        if not original:
            required_files.add(str(relative(name)))
        return p

    def mapped(path, *, original=False):
        p, base = Path(path), original_remote if original else remote
        require(p.is_absolute() and p.is_relative_to(base), "Absolute artifact outside its bound root")
        return local(p.relative_to(base), original=original)

    def document(name):
        return read(local(name))

    def inventory(files, *, original=False, job=None):
        require(isinstance(files, dict) and bool(files), "Empty terminal artifact inventory")
        for name, digest in files.items():
            p = relative(name)
            require(job is None or p.is_relative_to(Path("run") / job), "Artifact outside assigned job")
            require(isinstance(digest, str) and len(digest) == 64 and sha(local(name, original=original)) == digest,
                    "Terminal artifact SHA changed: " + name)

    def binding(value, *, original=False):
        p = mapped(value["path"], original=original)
        require(sha(p) == value["sha256"], "Bound receipt/artifact SHA changed")
        return p

    def same_metrics(left, right):
        require(type(left) is type(right), "Metric structure changed")
        if isinstance(left, dict):
            require(left.keys() == right.keys(), "Metric keys changed")
            for key in left:
                same_metrics(left[key], right[key])
        elif isinstance(left, list):
            require(len(left) == len(right), "Metric length changed")
            for a, b in zip(left, right):
                same_metrics(a, b)
        elif isinstance(left, (int, float)) and not isinstance(left, bool):
            require(math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-5),
                    "Validation metrics differ from selected training endpoint")
        else:
            require(left == right, "Metric metadata changed")

    def direct_exit(value, prefix, command, launch=None, *, ticks_required=True):
        require(type(value.get("pid")) is int and value["pid"] > 0 and value.get("command") == command
                and value.get("actual_os_returncode") == 0 and type(value.get("actual_os_returncode")) is int
                and value.get("timeout") is False and value.get("boot_id") == contract["boot_id"]
                and value.get("started_at") and value.get("observed_exit_at"), "Missing actual successful child exit")
        if ticks_required:
            require(type(value.get("start_ticks")) is int and value["start_ticks"] > 0, "Missing launch start ticks")
        if launch is not None:
            require(all(value.get(k) == launch.get(k) for k in ("pid", "start_ticks", "command", "started_at", "boot_id")),
                    "Direct child exit differs from launch identity")
        for stream in ("stdout", "stderr"):
            p = local(prefix + "." + stream + ".log")
            require(type(value.get(stream + "_bytes")) is int and p.stat().st_size == value[stream + "_bytes"]
                    and sha(p) == value.get(stream + "_sha256"), "Direct child log bytes/SHA changed")
        owned.add(value["pid"])

    def origin_copy(value, expected):
        require(value.get("original_path") == expected["path"] and value.get("original_sha256") == expected["sha256"]
                and value.get("copy_sha256") == expected["sha256"], "Original/copy lineage differs")
        binding(expected, original=True)
        p = mapped(value["copy_path"])
        require(sha(p) == expected["sha256"], "Preserved origin copy changed")
        return p

    original_inventory = contract["immutable_originals"]["files"]
    inventory(original_inventory, original=True)
    original_inventory_sha = canonical(original_inventory)
    require(contract["immutable_originals"].get("files_sha256", original_inventory_sha) == original_inventory_sha,
            "Original inventory binding changed")
    science_path = mapped(contract["scientific_contract"]["path"], original=True)
    require(sha(science_path) == contract["scientific_contract"]["raw_sha256"], "Original scientific contract bytes changed")
    science = read(science_path)
    require(canonical(science) == contract["scientific_contract"]["canonical_sha256"], "Original scientific contract changed")
    runtime.validate_contract(contract, science=science)
    require(canonical(document("recovery_contract.json")) == canonical(contract), "Recovery contract archive differs")
    final = document("final_receipt.json")
    counts = {"completed_fit_count": 4, "validation_replay_count": 10, "resumed_fit_count": 3,
              "fresh_fit_count": 1, "stored_raf_reference_count": 6, "test_count": 0}
    require(final.get("schema") == runtime.SCHEMA and final.get("host") == "5090" and final.get("status") == "complete"
            and final.get("contract_sha256") == canonical(contract)
            and final.get("scientific_contract_sha256") == canonical(science)
            and all(type(final.get(k)) is int and final[k] == v for k, v in counts.items())
            and final.get("held_out_test_evaluated") is False and final.get("historical_cnn_changed") is False
            and final.get("owned_worker_pids_absent") is True and final.get("gpu_after", {}).get("compute_pids") == []
            and final.get("dispatcher_own_exit_directly_observed") is False, "Incomplete recovery terminal receipt")
    preserved = final.get("immutable_originals_verification", {})
    require(preserved.get("root") == str(original_remote) and preserved.get("file_count") == len(original_inventory)
            and preserved.get("files_sha256") == original_inventory_sha and preserved.get("original_files_unchanged") is True,
            "Terminal original preservation proof differs")
    contract_path = str(remote / "recovery_contract.json")
    jobs = {j["id"]: j for j in contract["jobs"]}
    endpoint_keys = ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256", "target_identity")
    training = {}
    require(set(final.get("stages", {})) == {"train", "validation"}, "Terminal stages changed")
    for stage in ("train", "validation"):
        expected_ids = set(contract["training_jobs" if stage == "train" else "validation_jobs"])
        saved_stage = document(stage + "_stage_receipt.json")
        require(final["stages"][stage] == saved_stage and saved_stage.get("stage") == stage
                and saved_stage.get("status") == "complete" and saved_stage.get("failure") is None
                and saved_stage.get("pending_job_ids") == [] and set(saved_stage.get("exits", {})) == expected_ids
                and set(saved_stage.get("receipts", {})) == expected_ids, "Terminal stage jobset/inventory differs")
        for key in sorted(expected_ids):
            job, folder = jobs[key], "run/" + key + "/"
            launch, lease = document(folder + stage + "_launch.json"), document(folder + stage + "_lease.json")
            require(launch.get("job_id") == key and launch.get("stage") == stage
                    and lease == {**launch, "contract_sha256": canonical(contract)}, "Worker lease/launch mismatch")
            exit_record = document(folder + stage + "_exit.json")
            require(exit_record == saved_stage["exits"][key] and exit_record.get("job_id") == key
                    and exit_record.get("stage") == stage, "Actual worker exit binding differs")
            command = runtime.worker_command(contract, contract_path, job, stage)
            direct_exit(exit_record, "logs/" + key + "." + stage, command, launch)
            receipt_name = folder + stage + "_receipt.json"
            require(saved_stage["receipts"][key] == {"path": receipt_name, "sha256": sha(local(receipt_name))},
                    "Stage receipt path/SHA differs")
            value = document(receipt_name)
            require(value.get("schema") == runtime.STAGE_SCHEMA and value.get("status") == "complete"
                    and value.get("stage") == stage and value.get("job_id") == key and value.get("contract_sha256") == canonical(contract)
                    and value.get("scientific_contract_sha256") == canonical(science)
                    and value.get("scientific_contract_raw_sha256") == contract["scientific_contract"]["raw_sha256"]
                    and all(value.get(k) == job[k] for k in ("host", "dataset", "arm", "seed", "time_alpha", "loss_mode", "time_input"))
                    and value.get("recovery_mode") == job["recovery_mode"], "Recovery stage identity changed")
            require(value.get("candidate_name") == job["arm"] and value.get("late_path_parameter_count") == 1024
                    and value.get("late_path_mode") == ("observation" if job["arm"] == runtime.ARMS[0] else "hidden")
                    and value.get("time_input_transform") == value.get("quantity_input_transform") == "log1p_nonnegative_float32"
                    and value.get("held_out_test_evaluated") is False and value.get("quantity_only") is True
                    and value.get("time_head_trained") is False and value.get("quantity_loss_normalization") == "none"
                    and value.get("original_files_modified") is False and value.get("scientific_worker_source_unchanged") is True,
                    "Recovery scientific labels changed")
            verify_time_head_proof(value.get("time_head_no_training_verification"))
            inventory(value.get("files"), job=key)
            checkpoint = mapped(value["selected_checkpoint"])
            cp_relative = str(Path(value["selected_checkpoint"]).relative_to(remote))
            require(Path(cp_relative).is_relative_to(Path("run") / key) and value["files"].get(cp_relative) == value["checkpoint_sha256"]
                    and sha(checkpoint) == value["checkpoint_sha256"] and type(value.get("selected_epoch")) is int and value["selected_epoch"] > 0
                    and isinstance(value.get("state_sha256"), str) and len(value["state_sha256"]) == 64
                    and value.get("target_identity") and value.get("metrics"), "Selected endpoint binding differs")
            lineage = value.get("lineage", {})
            if stage == "train":
                nested_path = binding(value["scientific_receipt"])
                require(value["files"].get(str(nested_path.relative_to(root))) == value["scientific_receipt"]["sha256"], "Missing nested scientific receipt inventory")
                selected = read(nested_path)
                require(selected.get("schema") == "titantpp_late_path18_stage_v1" and selected.get("stage") == "train"
                        and selected.get("status") == "complete" and selected.get("job_id") == key
                        and selected.get("contract_sha256") == canonical(science)
                        and all(selected.get(k) == value[k] for k in endpoint_keys), "Recovered scientific receipt endpoint changed")
                inventory(selected.get("files"), job=key)
                require(selected["validation_reference"]["target_identity"] == value["target_identity"], "Training target identity differs")
                same_metrics(value["metrics"], selected["validation_reference"]["metrics"])
                if job["recovery_mode"] == "resume":
                    origin_copy(lineage["resume_checkpoint"], job["origin_resume_checkpoint"])
                    require(lineage.get("resumed_from_epoch") == job["resume_epoch"] and lineage.get("next_epoch") == job["resume_epoch"] + 1
                            and lineage.get("original_started_at") == job["original_started_at"]
                            and lineage.get("resume_validation", {}).get("validated") is True
                            and type(lineage["resume_validation"].get("prefix_optimizer_steps")) is int
                            and lineage["resume_validation"]["prefix_optimizer_steps"] > 0, "Exact-resume lineage missing")
                else:
                    require(lineage.get("fresh_from_original_seed") is True and lineage.get("resumed_from_epoch") is None,
                            "Fresh fit lineage changed")
                training[key] = value
            else:
                require(value.get("validation_replay_passed") is True, "Native Validation replay gate failed")
                if job["recovery_mode"] == "reference":
                    selected = read(origin_copy(lineage["reference_train_receipt"], job["origin_train_receipt"]))
                    copied = origin_copy(lineage["reference_checkpoint"], job["origin_selected_checkpoint"])
                    require(lineage.get("reference_retrained") is False and value["selected_checkpoint"] == lineage["reference_checkpoint"]["copy_path"]
                            and lineage.get("original_selected_checkpoint") == selected["selected_checkpoint"]
                            and value["checkpoint_sha256"] == job["origin_selected_checkpoint"]["sha256"] == sha(copied)
                            and all(value.get(k) == selected.get(k) for k in endpoint_keys if k != "selected_checkpoint"), "RAF selected reference changed")
                    require(selected.get("contract_sha256") == canonical(science) and selected.get("job_id") == key
                            and selected.get("stage") == "train" and selected.get("status") == "complete", "RAF source receipt identity differs")
                else:
                    selected = read(binding(lineage["recovery_scientific_receipt"]))
                    outer = training[key]
                    require(read(binding(lineage["recovery_train_receipt"])) == outer
                            and lineage["recovery_scientific_receipt"] == outer["scientific_receipt"]
                            and all(value.get(k) == outer.get(k) for k in endpoint_keys), "Recovered Validation selection changed")
                require(value["target_identity"] == selected["validation_reference"]["target_identity"], "Replayed targets changed")
                same_metrics(value["metrics"], selected["validation_reference"]["metrics"])
    qualification = document("qualification.json")
    require(qualification.get("status") == "passed" and qualification.get("host") == "5090"
            and qualification.get("contract_sha256") == canonical(contract) and qualification.get("native_worker_exit0_count") == 6
            and qualification.get("maximum_simultaneous_workers") == 3 and qualification.get("owned_worker_pids_absent") is True
            and qualification.get("held_out_test_evaluated") is False and set(qualification.get("per_arm", {})) == set(runtime.ARMS),
            "Incomplete native qualification")
    for arm in runtime.ARMS:
        rows = qualification["per_arm"][arm]
        require(len(rows) == 3, "Qualification must directly observe three children per arm")
        job = next(j for j in contract["jobs"] if j["arm"] == arm)
        for index, row in enumerate(rows):
            prefix = "qualification_" + arm + "_" + str(index)
            record = document(prefix + ".exit.json")
            result = row.get("result")
            require(row == {**record, "result": result} and result == read(local(prefix + ".stdout.log"))
                    and result.get("status") == "passed" and result.get("real_data_loaded") is False
                    and result.get("candidate_name") == arm and set(result.get("cuda_memory_peaks", {})) == {arm}
                    and type(result["cuda_memory_peaks"][arm].get("max_memory_reserved_bytes")) is int
                    and result["cuda_memory_peaks"][arm]["max_memory_reserved_bytes"] > 0, "Native qualification result changed")
            direct_exit(record, prefix, runtime.worker_command(contract, contract_path, job, "qualify"), ticks_required=False)
    qualifier = document("qualification_exit.json")
    qualifier_command = [contract["python"], "-u", "-m", runtime.MODULE, "--contract", contract_path, "--mode", "qualify"]
    direct_exit(qualifier, "qualification", qualifier_command)
    owner = qualification.get("qualification_owner_identity", {})
    require(owner.get("pid") == qualifier["pid"] and owner.get("pgid") == qualifier["pid"]
            and owner.get("start_ticks") == qualifier["start_ticks"] and owner.get("argv") == qualifier_command
            and owner.get("cwd") == contract["source_root"], "Qualification owner/direct-exit binding changed")
    dispatcher = document("dispatcher_exit.json")
    launch, start = document("dispatcher_launch.json"), document("start.json")
    command = [contract["python"], "-u", "-m", runtime.MODULE, "--contract", contract_path, "--mode", "dispatch"]
    require(launch.get("contract_sha256") == start.get("contract_sha256") == canonical(contract)
            and start.get("pid") == launch.get("pid") and start.get("boot_id") == contract["boot_id"]
            and start.get("host") == "5090" and start.get("training_job_count") == 4
            and start.get("validation_job_count") == 10 and start.get("maximum_concurrent_workers") == 3
            and start.get("qualification_sha256") == sha(local("qualification.json")), "Dispatcher launch/start binding changed")
    direct_exit(dispatcher, "dispatcher", command, launch)
    manifest_binding = final["artifact_manifest"]
    require(manifest_binding.get("path") == "artifact_manifest.json" and sha(local("artifact_manifest.json")) == manifest_binding.get("sha256"),
            "Terminal manifest binding changed")
    manifest = document("artifact_manifest.json")
    require(manifest.get("schema") == runtime.SCHEMA and manifest_binding.get("file_count") == len(manifest.get("files", {})),
            "Terminal manifest count/schema differs")
    inventory(manifest.get("files"))
    excluded = {"artifact_manifest.json", "final_receipt.json", "dispatcher.stdout.log", "dispatcher.stderr.log", "dispatcher_exit.json", "supervisor_exit.json"}
    require(set(manifest.get("excluded_lifecycle_files", [])) == excluded and required_files - excluded <= set(manifest["files"]),
            "Terminal manifest omitted required evidence")
    if check_live:
        require(root.resolve() == remote.resolve() and originals.resolve() == original_remote.resolve(), "Fresh process checks require native roots")
        require(all(runtime.proc_identity(pid) is None for pid in sorted(owned)), "A recovery-owned process remains")
        require(runtime.gpu_state(contract).get("compute_pids") == [], "GPU compute process remains")
    return {"status": "verified", **counts, "actual_scientific_worker_exit0_count": 14,
            "actual_qualification_child_exit0_count": 6, "actual_qualification_process_exit0_count": 1,
            "actual_dispatcher_exit0_count": 1, "owned_pids": sorted(owned),
            "fresh_process_absence_checked": check_live, "wrapper_own_os_exit_directly_observed": False,
            "contract_sha256": canonical(contract), "scientific_contract_sha256": canonical(science),
            "final_receipt_sha256": sha(local("final_receipt.json")), "artifact_manifest_sha256": sha(local("artifact_manifest.json")),
            "original_files_sha256": original_inventory_sha, "original_file_count": len(original_inventory)}


def collect():
    contract = verify_local()
    write(REPORT / "collection.intent.json", {"at": now(), "automatic_retry": False})
    code = r'''
import io,json,hashlib,tarfile
from pathlib import Path
from paper.scripts import titantpp_late_path18_5090_recovery_runtime as r
from paper.scripts.manage_titantpp_late_path18_5090_recovery_v2 import verify_terminal_evidence
cp=Path(__CONTRACT__);c=json.loads(cp.read_text());root=Path(c['root'])
r.validate_contract(c);r.verify_files(c);r.verify_originals(c)
final=json.loads((root/'final_receipt.json').read_text());exit=json.loads((root/'dispatcher_exit.json').read_text())
assert final['status']=='complete' and exit['actual_os_returncode']==0
terminal_verification=verify_terminal_evidence(c,check_live=True)
for job in c['jobs']:
 if job['recovery_mode']!='reference':r.verify_receipt(c,job,'train')
 r.verify_receipt(c,job,'validation')
gpu=r.gpu_state(c)
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 for name in files:tar.add(root/name,arcname=name,recursive=False)
archive=buf.getvalue();manifest={'host':'5090','files':files,'archive_sha256':hashlib.sha256(archive).hexdigest(),'contract_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'final':final,'gpu_after':gpu,'terminal_verification':terminal_verification}
meta=json.dumps(manifest,ensure_ascii=False).encode()
import sys
sys.stdout.buffer.write(str(len(meta)).encode()+b'\n'+meta+archive)
'''.replace("__CONTRACT__", repr(contract["root"] + "/recovery_contract.json"))
    command = "cd " + shlex.quote(contract["source_root"]) + " && env PYTHONDONTWRITEBYTECODE=1 " + shlex.quote(contract["python"]) + " -c " + shlex.quote(code)
    raw = transport("collect", ssh(command, timeout=240))
    line, rest = raw.split(b"\n", 1)
    size = int(line)
    metadata, archive = rest[:size], rest[size:]
    manifest = json.loads(metadata)
    require(manifest["files"] and hashlib.sha256(archive).hexdigest() == manifest["archive_sha256"], "Recovered archive differs")
    dest = BUNDLE / "retrieval"
    dest.mkdir(exist_ok=False)
    (dest / "manifest.json").write_bytes(metadata)
    (dest / "originals.tar.gz").write_bytes(archive)
    originals = dest / "originals"
    originals.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts, "Unsafe recovered archive member")
            target = originals / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    for name, digest in manifest["files"].items():
        require(sha(originals / name) == digest, "Recovered original differs: " + name)
    terminal_verification = verify_terminal_evidence(contract, execution_root=originals, origin_root=ORIGIN / "originals")
    require(manifest["terminal_verification"]["fresh_process_absence_checked"] is True,
            "Native collection did not verify process absence")
    for key, value in terminal_verification.items():
        if key != "fresh_process_absence_checked":
            require(manifest["terminal_verification"].get(key) == value, "Native/local terminal verification differs: " + key)
    receipt = {"status": "complete", "at": now(), "host": "5090", "resumed_fit_count": 3, "fresh_fit_count": 1,
        "stored_raf_training_reference_count": 6, "validation_replay_count": 10, "held_out_test_evaluated": False,
        "manifest_sha256": sha(dest / "manifest.json"), "archive_sha256": sha(dest / "originals.tar.gz"), "files": manifest["files"],
        "original_archive_sha256": sha(ORIGIN / "originals.tar.gz"), "original_manifest_sha256": sha(ORIGIN / "manifest.json"),
        "terminal_verification": terminal_verification,
        "original_unknown_exits_preserved": True, "historical_cnn_changed": False}
    write(dest / "final_receipt.json", receipt)
    return receipt


def finish():
    contract = verify_local()
    write(REPORT / "completion.intent.json", {"at": now(), "finite": True, "recurring_automation": False, "automatic_retry": False})
    code = r'''
import pathlib,time,json
from paper.scripts import titantpp_late_path18_5090_recovery_runtime as r
c=json.loads(pathlib.Path(__CONTRACT__).read_text());p=pathlib.Path(c['root']);deadline=c['absolute_host_deadline_unix']
while not (p/'dispatcher_exit.json').exists():
 assert time.time()<deadline,'Inherited deadline expired'
 assert r.boot_id()==c['boot_id'],'Server rebooted during recovery; no automatic retry'
 qualification=p/'qualification_exit.json'
 if qualification.exists():assert json.loads(qualification.read_text())['actual_os_returncode']==0,'Native qualification failed'
 launch=p/'dispatcher_launch.json'
 if launch.exists():
  identity=json.loads(launch.read_text())
  if r.proc_identity(identity['pid']) is None:
   time.sleep(1)
   assert (p/'dispatcher_exit.json').exists(),'Dispatcher disappeared without directly observed exit'
 time.sleep(20)
exit=json.loads((p/'dispatcher_exit.json').read_text());assert exit['actual_os_returncode']==0,'Recovery dispatcher failed'
print(json.dumps({'status':'terminal_observed','actual_os_returncode':0}))
'''.replace("__CONTRACT__", repr(contract["root"] + "/recovery_contract.json"))
    command = "cd " + shlex.quote(contract["source_root"]) + " && env PYTHONDONTWRITEBYTECODE=1 " + shlex.quote(contract["python"]) + " -c " + shlex.quote(code)
    try:
        terminal = json.loads(transport("wait", ssh(command, timeout=max(1, contract["absolute_host_deadline_unix"] - time.time()))))
        receipt = collect()
        write(REPORT / "COMPLETE.json", {"status": "complete", "at": now(), "terminal": terminal, "retrieval_receipt_sha256": sha(BUNDLE / "retrieval/final_receipt.json")})
        return receipt
    except Exception as error:
        write(REPORT / "FAILURE.json", {"status": "action_required", "at": now(), "exception": type(error).__name__, "message": str(error), "automatic_retry": False, "originals_preserved": True})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("prepare", "deploy", "status", "finish", "collect"))
    args = parser.parse_args()
    value = {"prepare": prepare, "deploy": deploy, "status": status, "finish": finish, "collect": collect}[args.mode]()
    if args.mode == "status":
        value = {"path": value["path"], "sha256": value["sha256"]}
    if args.mode in ("finish", "collect"):
        value = {key: item for key, item in value.items() if key != "files"}
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
