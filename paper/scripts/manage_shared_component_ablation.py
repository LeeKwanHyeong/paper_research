"""Exclusive, source-bound controls for the approved A-D component experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import traceback

NAME = "shared_component_ablation_20261022_v1"
SCHEMA = "shared_component_ablation_v1"
ARMS = ("full", "no_prototype", "no_history", "encoder_only")
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / NAME
BUNDLE = ROOT / "search_artifacts" / NAME
PARENT = ROOT / "search_artifacts/titans_mac_accuracy9_20261020_v1"
A_PARENT = ROOT / "search_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1"
NEW_SOURCE = ("models/TPPs/CountAwareComponentAblation.py",
    "paper/scripts/run_shared_component_ablation.py",
    "paper/scripts/profile_shared_component_ablation.py",
    "paper/scripts/manage_shared_component_ablation.py")


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def below(root, relative):
    root = Path(root).resolve(); relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
        "Evidence path must stay below the isolated root")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Evidence symlink escaped the isolated root")
    return path


def validate(c):
    require(c["schema"] == SCHEMA and c["campaign"] == NAME, "Campaign identity changed")
    require(set(c["hosts"]) == {"5080", "5090"}, "Both approved hosts are required")
    expected = {(d, a, s) for d in DATASETS for a in ARMS for s in (42, 52, 62)}
    jobs = c["jobs"]
    require(len(jobs) == 36 and {(j["dataset"], j["arm"], j["seed"]) for j in jobs} == expected,
        "Exactly the 36 matched component jobs are required")
    for j in jobs:
        require(j["id"] == f"{j['dataset']}__{j['arm']}__seed{j['seed']}"
            and j["host"] == ("5080" if j["seed"] == 42 else "5090"), "Fixed job/host assignment changed")
        require(bool(j.get("reuse")) == (j["arm"] == "full" and j["seed"] in (52, 62)),
            "Only the six verified same-host A fits may be reused")
    require(c["common_condition"] == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "Common objective changed")
    require(c["training"] == {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
        "batch_size": 128, "monitor": "validation_raw_quantity_rmse"}, "Training budget changed")
    require(c["source"]["files"] and canonical(c["source"]["files"]) == c["source"]["files_sha256"],
        "Source inventory missing or changed")
    require(len(c["input_files"]) == 9, "All original input files are required")
    for relative in (*c["source"]["files"], *c["input_files"], *c.get("legacy_files", {})):
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Inventory path escaped the source root")


def verify_files(c, source):
    validate(c)
    for inventory in (c["source"]["files"], c["input_files"], c.get("legacy_files", {})):
        for path, digest in inventory.items():
            require(sha(below(source, path)) == digest, "Frozen file SHA changed: " + path)


def prepare():
    spec = read(REPORT / "SCIENCE_SPEC.json")
    require(spec["status"] == "locked", "Scientific contract must be locked before implementation")
    audit = read(REPORT / "A_REUSE_AUDIT.json")
    require(audit["status"] == "PASS_CONDITIONAL_REUSE" and sha(REPORT / "A_REUSE_AUDIT.json")
        == spec["A_reuse_audit"]["sha256"], "Original A reuse audit changed")
    require(not BUNDLE.exists(), "Prepared campaign already exists; preserve it")
    parent = read(PARENT / "execution_contract.json")
    original = read(A_PARENT / "execution_contract.json")
    c = {key: deepcopy(parent[key]) for key in ("datasets", "input_files", "training", "limits", "common_condition")}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    c.update(schema=SCHEMA, campaign=NAME, created_at=datetime.now(timezone.utc).isoformat(),
        scope="execution", science_spec_sha256=sha(REPORT / "SCIENCE_SPEC.json"), science=spec,
        model_role="observed_time_joint_v1", data_variant="original_uncapped", quantity_capping_applied=False,
        evaluation_scope="validation_selection_then_frozen_test", held_out_test_authorized=True,
        maximum_parallel_workers=1, fresh_fit_count=30, reused_fit_count=6, MAC_deferred=True,
        source={"files": deepcopy(original["source"]["files"]), "base_git_revision": revision},
        input_provenance={"path": str(PARENT / "execution_contract.json"),
            "sha256": sha(PARENT / "execution_contract.json"), "purpose": "frozen task data/head conditions only"},
        A_provenance={"path": str(A_PARENT / "execution_contract.json"), "sha256": sha(A_PARENT / "execution_contract.json")},
        cost={key: spec["cost"][key] for key in ("batch_size", "preparation_seed", "warmup_batches", "measured_batches",
            "repetitions", "modes", "fresh_process_per_rep", "same_fit_host")},
        implementation_qualification={"status": "pending", "training_allowed": False},
        native_qualification_required=True, execution_authorization={"real_data_training": True,
            "real_data_inference": True, "remote_execution": True, "test_evaluation": True})
    c["cost"].update(maximum_parallel_workers=1, automatic_retry=False)
    c["architecture"] = {"schema": "shared_component_architecture_v1", "arms": deepcopy(spec["arms"]),
        "retained": deepcopy(spec["retained"]), "A_factory": "unchanged frozen Shared15 factory"}
    c["design_sha256"] = canonical(c["architecture"])
    for relative, digest in c["source"]["files"].items():
        require(sha(ROOT / relative) == digest, "Historical scientific source changed: " + relative)
    c["source"]["files"].update({relative: sha(ROOT / relative) for relative in NEW_SOURCE})
    c["source"]["files_sha256"] = canonical(c["source"]["files"])
    prior5080 = read(ROOT / "reports/titantpp_shared_external_extra42_20261020_v1/SCIENCE_SPEC.json")["native"]
    c["hosts"] = {"5080": deepcopy(prior5080), "5090": deepcopy(parent["hosts"]["5090"])}
    for host, h in c["hosts"].items():
        remote = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + NAME + "_" + host
        h.update(root=remote, source_root=remote + "/source", tmux=NAME + "_" + host,
            absolute_host_deadline_unix=time.time() + c["limits"]["queue_wall_seconds"],
            boot_id=read(REPORT / "preflight" / (host + ".stdout"))["boot_id"])
        h.pop("absolute_host_deadline_binding", None); h.pop("completed_predecessor", None)
        h["environment"].update(SOURCE_REVISION=revision, PYTHONPATH=remote + "/source",
            MPLCONFIGDIR=remote + "/cache/matplotlib", XDG_CACHE_HOME=remote + "/cache",
            TORCHINDUCTOR_CACHE_DIR=remote + "/cache/torchinductor")
    c["jobs"], c["legacy_files"] = [], {}
    rows = {(r["dataset"], r["seed"]): r for r in audit["checkpoint_rows"]}
    BUNDLE.mkdir(parents=True)
    for data in c["datasets"]:
        for seed in (42, 52, 62):
            for arm in ARMS:
                job = {"id": f"{data['dataset_id']}__{arm}__seed{seed}", "dataset": data["dataset_id"],
                    "seed": seed, "arm": arm, "host": "5080" if seed == 42 else "5090",
                    "time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0}
                if arm == "full" and seed in (52, 62):
                    row = rows[data["dataset_id"], seed]
                    legacy = {}
                    items = {key: row[key] for key in ("checkpoint", "train_receipt", "validation_receipt", "history")}
                    items.update({"old_" + stage + "_exit": item for stage, item in row["actual_exits"].items()})
                    for key, item in items.items():
                        source = Path(item["path"])
                        require(sha(source) == item["sha256"], "Audited legacy original changed")
                        relative = "legacy/" + job["id"] + "/" + key + source.suffix
                        dest = BUNDLE / "input_snapshot" / relative; dest.parent.mkdir(parents=True, exist_ok=True)
                        with source.open("rb") as src, dest.open("xb") as dst:
                            shutil.copyfileobj(src, dst)
                        c["legacy_files"][relative] = item["sha256"]
                        legacy[key] = {"path": relative, "sha256": item["sha256"]}
                    legacy.update(selected_epoch=row["checkpoint"]["selected_epoch"],
                        canonical_tensor_sha256=row["checkpoint"]["canonical_tensor_sha256"],
                        original_job_id=row["original_job_id"], original_source_revision=row["source_revision"])
                    job["reuse"] = legacy
                c["jobs"].append(job)
    for relative, digest in c["input_files"].items():
        source = PARENT / "input_snapshot" / relative
        require(sha(source) == digest, "Frozen input changed")
        dest = BUNDLE / "input_snapshot" / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, dest.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    validate(c)
    write(BUNDLE / "prepared_contract.json", c)
    return {"status": "prepared_not_trainable", "fits": 30, "reused": 6}


def seal(tests_receipt):
    c = read(BUNDLE / "prepared_contract.json")
    qa = read(REPORT / "INDEPENDENT_QA.json")
    require(qa.get("status") == "PASS", "Independent implementation qualification has not passed")
    files = qa["files_reviewed_sha256"]
    require(all(name in files for name in NEW_SOURCE) and all(sha(ROOT / name) == digest for name, digest in files.items()),
        "Reviewed implementation changed or review is incomplete")
    tests = read(tests_receipt)
    require(tests.get("actual_os_returncode") == 0, "Focused tests did not pass")
    require(all(tests.get("files_checked_sha256", {}).get(name) == c["source"]["files"][name]
        for name in NEW_SOURCE) and tests.get("source_unchanged_during_tests") is True,
        "Focused tests are not bound to this exact implementation")
    require(c["science_spec_sha256"] == sha(REPORT / "SCIENCE_SPEC.json"), "Locked scientific contract changed")
    require(all(sha(ROOT / name) == digest for name, digest in c["source"]["files"].items()), "Prepared source changed")
    c["implementation_qualification"] = {"status": "passed", "training_allowed": True,
        "independent_qa_sha256": sha(REPORT / "INDEPENDENT_QA.json"), "tests_receipt_sha256": sha(tests_receipt)}
    write(BUNDLE / "execution_contract.json", c)
    with (BUNDLE / "deployment.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(BUNDLE / "execution_contract.json", arcname="execution_contract.json")
        for name in c["source"]["files"]:
            archive.add(ROOT / name, arcname="source/" + name)
        for name, digest in {**c["input_files"], **c["legacy_files"]}.items():
            path = BUNDLE / "input_snapshot" / name
            require(sha(path) == digest, "Prepared data/legacy copy changed")
            archive.add(path, arcname="source/" + name)
        marker = tarfile.TarInfo("source/sample_data"); marker.type = tarfile.DIRTYPE
        archive.addfile(marker)
    value = {"status": "sealed", "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(c), "archive_sha256": sha(BUNDLE / "deployment.tar.gz"),
        "source_files_sha256": c["source"]["files_sha256"], "git_revision": c["source"]["base_git_revision"],
        "science_spec_sha256": c["science_spec_sha256"], "independent_qa_sha256": sha(REPORT / "INDEPENDENT_QA.json"),
        "tests_receipt_sha256": sha(tests_receipt)}
    write(BUNDLE / "integrity.json", value)
    return value


def proc_identity(pid):
    try:
        path = Path("/proc") / str(pid)
        values = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "state": values[0], "start_ticks": int(values[19]), "pgid": int(values[2]),
            "argv": (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0"), "cwd": os.readlink(path / "cwd")}
    except FileNotFoundError:
        return None


def guard(c, host, active=()):
    h = c["hosts"][host]
    require(time.time() < h["absolute_host_deadline_unix"], "Campaign deadline reached")
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == h["boot_id"], "Server boot identity changed")
    paused = h.get("paused_processes", [])
    for expected in paused:
        live = proc_identity(expected["pid"])
        require(live is not None and live["state"] == "T" and all(live[k] == expected[k]
            for k in ("start_ticks", "pgid", "argv", "cwd")), "Historical paused process changed")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15)
    pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    require(pids <= set(active) | {p["pid"] for p in paused}, "Unknown GPU owner; no new work")
    return {"compute_pids": sorted(pids), "paused_preserved": len(paused)}


def verify_native_qualification(c, host, root=None):
    root = Path(root or c["hosts"][host]["root"])
    value = read(root / "qualification.complete.json")
    require(value.get("status") == "complete" and value.get("host") == host
        and value.get("contract_sha256") == canonical(c), "Native qualification identity differs")
    exits = value.get("actual_child_exits", [])
    require(len(exits) == 16 and all(e.get("actual_os_returncode") == 0 and e.get("timeout") is False
        and e.get("supervisor_error") is None and e.get("resource_incident") is None for e in exits),
        "Native qualification requires actual successful exits for twelve integration and four capacity checks")
    jobs = {j["id"]: j for j in c["jobs"]}
    for entry in exits:
        require(entry.get("host") == host and entry.get("contract_sha256") == canonical(c)
            and jobs.get(entry.get("job_id"), {}).get("host") == host
            and entry.get("stage") in {"synthetic", "capacity"}, "Native child scientific identity differs")
        raw = root / "control" / (entry["job_id"] + "." + entry["stage"] + ".exit.json")
        require(read(raw) == entry, "Native qualification differs from the original actual child exit")
    require({jobs[e["job_id"]]["arm"] for e in exits if e["stage"] == "capacity"} == set(ARMS),
        "All four full-batch capacity exits are required")
    require({(jobs[e["job_id"]]["dataset"], jobs[e["job_id"]]["arm"]) for e in exits if e["stage"] == "synthetic"}
        == {(d, a) for d in DATASETS for a in ARMS}, "All dataset/arm integration checks are required")
    for arm, item in value["capacity_receipts"].items():
        path = below(root, item["path"])
        require(sha(path) == item["sha256"], "Native capacity evidence changed")
        receipt = read(path)
        require(receipt["status"] == "qualified" and receipt["arm"] == arm
            and receipt["host"] == host and receipt["contract_sha256"] == canonical(c)
            and receipt["source_files_sha256"] == c["source"]["files_sha256"], "Capacity receipt identity differs")
        require(receipt.get("synthetic") is True and receipt.get("real_data_loaded") is False
            and receipt.get("cost_complete") is False and receipt.get("checkpoint_saved") is False,
            "Capacity qualification must be synthetic and preserve checkpoints")
        rows = receipt.get("rows", [])
        require(len(rows) == 2 and {r.get("mode") for r in rows} == {"train", "eval"}, "Both capacity modes are required")
        for row in rows:
            require(row.get("batch_size") == 128 and row.get("padded_length") == 256
                and row.get("optimizer_steps") == int(row["mode"] == "train"), "Full-batch capacity shape or optimizer step differs")
            elapsed = row.get("elapsed_seconds")
            require(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed > 0, "Capacity time must be finite and positive")
            for kind in ("allocated", "reserved"):
                baseline = row.get("memory_baseline_bytes", {}).get(kind)
                peak = row.get("memory_peak_bytes", {}).get(kind)
                require(type(baseline) is int and type(peak) is int and 0 <= baseline <= peak and peak > 0,
                    "Capacity memory measurement is missing or invalid")
    require(set(value["capacity_receipts"]) == set(ARMS), "Four native capacity receipts are required")
    return value


def verify_phase_receipt(c, host, phase, root=None):
    root = Path(root or c["hosts"][host]["root"])
    complete = read(root / (phase + ".complete.json"))
    terminal = read(root / (phase + ".supervisor_exit.json"))
    require(complete.get("status") == "complete" and all(v.get("host") == host and v.get("phase") == phase
        and v.get("contract_sha256") == canonical(c) for v in (complete, terminal))
        and terminal.get("actual_os_returncode") == 0, "Phase or dispatcher exit identity differs")
    entries = complete.get("actual_child_exits", []); assigned = [j for j in c["jobs"] if j["host"] == host]
    if phase == "qualification":
        verify_native_qualification(c, host, root)
    else:
        expected = {(j["id"], "reuse" if j.get("reuse") else "train") for j in assigned} | {(j["id"], "validation") for j in assigned}
        if phase == "test_cost":
            expected = {(j["id"], "test") for j in assigned} | {(assigned[0]["id"], "prepare_batches")} | {
                (j["id"], f"cost_{mode}_{rep}") for j in assigned for mode in ("train", "eval") for rep in (1, 2, 3)}
        require(len(entries) == len(expected) and {(e.get("job_id"), e.get("stage")) for e in entries} == expected,
            "Phase is missing expected scientific or cost workers")
    for entry in entries:
        require(entry.get("host") == host and entry.get("contract_sha256") == canonical(c)
            and entry.get("actual_os_returncode") == 0 and entry.get("timeout") is False
            and entry.get("supervisor_error") is None and entry.get("resource_incident") is None,
            "Phase has an unsuccessful child exit")
        stem = root / "control" / (entry["job_id"] + "." + entry["stage"])
        require(read(str(stem) + ".exit.json") == entry, "Phase differs from original child exit")
        for stream in ("stdout", "stderr"):
            require(sha(str(stem) + "." + stream + ".log") == entry[stream + "_sha256"], "Actual child output changed")
    return complete, terminal


def verify_global_gate(c, root):
    root = Path(root); gate = read(root / "global_validation_gate.json")
    jobs = {j["id"]: j for j in c["jobs"]}
    require(gate.get("status") == "passed" and gate.get("contract_sha256") == canonical(c)
        and set(gate.get("jobs", {})) == set(jobs), "All 36 individual Validation replays must pass before Test")
    for job_id, evidence in gate["jobs"].items():
        job = jobs[job_id]
        require(evidence["host"] == job["host"], "Global Validation host changed")
        for stage in ("train", "validation"):
            for kind in ("receipt", "exit"):
                item = evidence[f"{stage}_{kind}"]; path = below(root, item["path"])
                require(sha(path) == item["sha256"], "Global original evidence SHA changed")
                value = read(path)
                expected_stage = "reuse" if stage == "train" and kind == "exit" and job.get("reuse") else stage
                require(value.get("job_id") == job_id and value.get("stage") == expected_stage
                    and value.get("host") == job["host"] and value.get("contract_sha256") == canonical(c),
                    "Global scientific identity differs")
                if kind == "exit":
                    require(value.get("actual_os_returncode") == 0 and value.get("timeout") is False
                        and value.get("supervisor_error") is None and value.get("resource_incident") is None,
                        "Actual successful child exits are required")
                else:
                    require(value.get("status") == "complete" and value.get("host") == job["host"], "Receipt is not complete")
                    if stage == "validation":
                        require(value.get("validation_replay_passed") is True
                            and value.get("individual_predictions_match") is True, "Individual Validation replay missing")
    return gate


def terminate_owned(process, command, identity):
    if process.poll() is not None:
        return
    live = proc_identity(process.pid)
    require(identity is not None and live is not None and live["argv"] == command
        and all(live[k] == identity[k] for k in ("pid", "start_ticks", "pgid", "cwd")), "Refuse to terminate unverified process")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)


def stage_child(c, host, job, stage, contract_path):
    h = c["hosts"][host]; root = Path(h["root"]); guard(c, host)
    stem = root / "control" / (job["id"] + "." + stage)
    path = lambda suffix: Path(str(stem) + suffix)
    command = [h["python"], "-u", "-m", "paper.scripts." +
        ("profile_shared_component_ablation" if stage == "capacity" or stage == "prepare_batches" or stage.startswith("cost_")
         else "run_shared_component_ablation"), "--contract", str(contract_path), "--contract-sha", sha(contract_path), "--host", host]
    if stage == "capacity":
        command += ["--job", job["id"], "--mode", "qualification"]
    elif stage == "prepare_batches":
        command += ["--mode", "prepare-batches"]
    elif stage.startswith("cost_"):
        _, mode, rep = stage.split("_"); command += ["--job", job["id"], "--mode", mode, "--rep", rep]
    else:
        command += ["--job", job["id"], "--stage", stage]
    write(path(".intent.json"), {"job": job, "stage": stage, "contract_sha256": canonical(c), "command": command})
    started = time.monotonic(); identity = None; timed_out = False; error = incident = None; last_guard = -1e9
    limit = c["limits"]["fit_wall_seconds" if stage == "train" else "evaluation_wall_seconds"]
    with path(".stdout.log").open("xb") as out, path(".stderr.log").open("xb") as err:
        process = subprocess.Popen(command, cwd=h["source_root"], env={**os.environ, **h["environment"]},
            stdout=out, stderr=err, start_new_session=True)
        try:
            while process.poll() is None:
                live = proc_identity(process.pid)
                if identity is None and live is not None and live["argv"] == command:
                    identity = live; write(path(".launch.json"), identity)
                if time.monotonic() - last_guard >= 30 and incident is None:
                    last_guard = time.monotonic()
                    try:
                        guard(c, host, active=(process.pid,))
                    except RuntimeError as fault:
                        incident = str(fault); write(path(".resource_incident.json"), {"error": incident,
                            "policy": "drain_current_child_preserve_foreign_and_paused_block_next_stage"})
                if time.monotonic() - started >= limit or time.time() >= h["absolute_host_deadline_unix"]:
                    timed_out = True; terminate_owned(process, command, identity); break
                time.sleep(.2)
        except BaseException as fault:
            error = type(fault).__name__ + ": " + str(fault)
        finally:
            if process.poll() is None:
                terminate_owned(process, command, identity)
            code = process.wait()
    value = {"job_id": job["id"], "stage": stage, "host": host, "command": command, "identity": identity,
        "actual_os_returncode": code, "timeout": timed_out, "supervisor_error": error, "resource_incident": incident,
        "contract_sha256": canonical(c), "elapsed_seconds": time.monotonic() - started,
        "stdout_sha256": sha(path(".stdout.log")), "stderr_sha256": sha(path(".stderr.log"))}
    write(path(".exit.json"), value)
    require(code == 0 and not timed_out and error is None and incident is None, "Child or resource guard failed; preserve attempt; no retry")
    guard(c, host)
    return value


def dispatch(c, host, phase, contract_path):
    verify_files(c, c["hosts"][host]["source_root"])
    q = c["implementation_qualification"]
    require(q.get("status") == "passed" and q.get("training_allowed") is True, "Independent implementation gate is pending")
    root = Path(c["hosts"][host]["root"])
    if phase == "fit_validation": verify_native_qualification(c, host, root)
    if phase == "test_cost": verify_global_gate(c, root)
    guard(c, host)
    write(root / (phase + ".claim.json"), {"identity": proc_identity(os.getpid()), "contract_sha256": canonical(c),
        "started_at_unix": time.time()})
    jobs = [j for j in c["jobs"] if j["host"] == host]; exits = []; capacities = {}
    if phase == "qualification":
        unique = {(j["dataset"], j["arm"]): j for j in jobs}
        for job in unique.values(): exits.append(stage_child(c, host, job, "synthetic", contract_path))
        for arm in ARMS:
            job = next(j for j in jobs if j["arm"] == arm)
            exits.append(stage_child(c, host, job, "capacity", contract_path))
            relative = "cost/qualification/" + arm + ".json"
            capacities[arm] = {"path": relative, "sha256": sha(root / relative)}
    elif phase == "fit_validation":
        for job in jobs:
            exits.append(stage_child(c, host, job, "reuse" if job.get("reuse") else "train", contract_path))
        for job in jobs: exits.append(stage_child(c, host, job, "validation", contract_path))
    elif phase == "test_cost":
        for job in jobs: exits.append(stage_child(c, host, job, "test", contract_path))
        exits.append(stage_child(c, host, jobs[0], "prepare_batches", contract_path))
        for job in jobs:
            for mode in ("train", "eval"):
                for rep in (1, 2, 3): exits.append(stage_child(c, host, job, f"cost_{mode}_{rep}", contract_path))
    else:
        raise RuntimeError("Unknown phase")
    write(root / (phase + ".complete.json"), {"status": "complete", "phase": phase, "host": host,
        "contract_sha256": canonical(c), "actual_child_exits": exits, "capacity_receipts": capacities,
        "gpu": guard(c, host), "finished_at_unix": time.time(), "supervisor_own_os_exit_directly_observed": False})


def supervise(c, host, phase, contract_path):
    """A separate parent observes the dispatcher's actual OS return code."""
    h = c["hosts"][host]; root = Path(h["root"])
    verify_files(c, h["source_root"]); guard(c, host)
    command = [h["python"], "-u", "-m", "paper.scripts.manage_shared_component_ablation",
        "--mode", "dispatch", "--contract", str(contract_path), "--contract-sha", sha(contract_path),
        "--host", host, "--phase", phase]
    write(root / (phase + ".supervisor.intent.json"), {"command": command,
        "identity": proc_identity(os.getpid()), "contract_sha256": canonical(c)})
    started = time.time()
    with (root / (phase + ".stdout.log")).open("xb") as out, (root / (phase + ".stderr.log")).open("xb") as err:
        result = subprocess.run(command, cwd=h["source_root"], env={**os.environ, **h["environment"]}, stdout=out, stderr=err)
    value = {"actual_os_returncode": result.returncode, "command": command, "host": host,
        "phase": phase, "contract_sha256": canonical(c), "started_at_unix": started,
        "observed_exit_at_unix": time.time(), "stdout_sha256": sha(root / (phase + ".stdout.log")),
        "stderr_sha256": sha(root / (phase + ".stderr.log")), "parent_own_os_exit_directly_observed": False}
    write(root / (phase + ".supervisor_exit.json"), value)
    return value


def pack(c, host, phase):
    """One immutable transfer archive after the phase has actually exited."""
    h = c["hosts"][host]; root = Path(h["root"])
    verify_files(c, h["source_root"]); gpu = guard(c, host)
    complete, exit_receipt = verify_phase_receipt(c, host, phase, root)
    for identity in [read(root / (phase + ".claim.json"))["identity"],
            *[e.get("identity") for e in complete["actual_child_exits"]]]:
        require(identity is not None, "A scientific child identity was not captured")
        live = proc_identity(identity["pid"])
        require(live is None or any(live[k] != identity[k] for k in ("start_ticks", "argv", "cwd")),
            "Phase or scientific worker is still alive")
    prefix = "collection_" + phase
    write(root / (prefix + ".intent.json"), {"contract_sha256": canonical(c), "host": host, "phase": phase})
    inventory = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        legacy_evidence = relative.parts[:2] == ("source", "legacy")
        if relative.parts[0] == "cache" or (relative.parts[0] == "source" and not legacy_evidence) or relative.name.startswith("collection_"):
            continue
        require(not path.is_symlink(), "Collection refuses symlinks")
        if path.is_file():
            require(path.stat().st_size > 0 or path.suffix == ".log", "Empty scientific evidence file")
            inventory[str(relative)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    require(bool(inventory), "Empty collection inventory")
    manifest = {"host": host, "phase": phase, "contract_sha256": canonical(c), "files": inventory,
        "source_files": c["source"]["files"], "input_files": c["input_files"], "legacy_files": c["legacy_files"],
        "all_source_input_legacy_actual_sha_verified": True, "gpu": gpu}
    manifest_path = root / (prefix + ".manifest.json"); write(manifest_path, manifest)
    archive_path = root / (prefix + ".tar.gz")
    with archive_path.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(manifest_path, arcname=manifest_path.name)
        for relative, item in inventory.items():
            require(sha(root / relative) == item["sha256"], "File changed while collecting")
            archive.add(root / relative, arcname=relative)
    value = {"status": "complete", "host": host, "phase": phase, "archive": str(archive_path),
        "archive_sha256": sha(archive_path), "archive_bytes": archive_path.stat().st_size,
        "manifest_sha256": sha(manifest_path), "files": len(inventory), "contract_sha256": canonical(c)}
    write(root / (prefix + ".receipt.json"), value)
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "seal", "dispatch", "supervise", "pack"), required=True)
    p.add_argument("--tests-receipt", type=Path); p.add_argument("--contract", type=Path)
    p.add_argument("--contract-sha"); p.add_argument("--host", choices=("5080", "5090"))
    p.add_argument("--phase", choices=("qualification", "fit_validation", "test_cost"))
    a = p.parse_args(argv)
    if a.mode == "prepare": result = prepare()
    elif a.mode == "seal": result = seal(a.tests_receipt)
    else:
        require(a.contract_sha and sha(a.contract) == a.contract_sha, "Externally pinned contract SHA is required")
        c = read(a.contract)
        if a.mode == "supervise":
            result = supervise(c, a.host, a.phase, a.contract)
            print(json.dumps(result)); return 0 if result["actual_os_returncode"] == 0 else 1
        if a.mode == "pack":
            print(json.dumps(pack(c, a.host, a.phase))); return 0
        try:
            dispatch(c, a.host, a.phase, a.contract)
        except BaseException as error:
            path = Path(c["hosts"][a.host]["root"]) / (a.phase + ".failure.json")
            if not path.exists(): write(path, {"error": type(error).__name__ + ": " + str(error),
                "traceback": traceback.format_exc(), "contract_sha256": canonical(c), "automatic_retry": False})
            raise
        result = {"status": "complete"}
    print(json.dumps(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
