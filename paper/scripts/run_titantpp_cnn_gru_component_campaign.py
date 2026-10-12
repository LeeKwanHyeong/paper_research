"""One-attempt component campaign operations; importing starts no work.

The sealed frozen_runtime.py supplies the original trainer and checkpoint
replay. This adapter changes only admission, ownership, and four-arm scheduling.
Native qualification and real fits require separately bound user permits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

SCHEMA = "titantpp_cnn_gru_components_20261012_v1"
NONE = "titantpp_cnn_gru54_correction_disabled"
CNN = "titantpp_cnn_gru54_cnn_only"
GRU = "titantpp_cnn_gru54_gru_only"
FULL = "titantpp_cnn_gru54"
ARMS = (NONE, CNN, GRU, FULL)
SEEDS = (42, 52, 62)
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
FROZEN_CLOSURE = "c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a"
FROZEN_REVISION = "57da14a2a7a5687e1bdef7d3164f3e8c0051da59"
FROZEN_RUNTIME_SHA = "e47bea24d796c5eeeff4f4fcc3c702308b461ac4c446cca1aab354d951d4af09"
TRAINING = {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40,
            "patience": 40, "monitor": "validation_raw_quantity_rmse",
            "tie": "strict_earliest_finite_minimum", "warm_start": False}
PARALLELISM = {"maximum_workers": 3,
               "grouping": "same_dataset_seed_four_arms_waves_3_plus_1",
               "per_process_cuda_fraction": .30, "aggregate_gpu_fraction": .8}
GPU_UUIDS = {"5080": "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6",
             "5090": "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"}
PYTHONS = {"5080": "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12",
           "5090": "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12"}
ROOT = None
common = base = runner = frozen = None


class OwnedTermination(RuntimeError):
    """A supervisor signal requests cleanup before this controller exits."""


@contextmanager
def owned_signal_guard():
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        # A repeated supervisor signal must not interrupt owned-child cleanup.
        signal.signal(signum, signal.SIG_IGN)
        raise OwnedTermination("SIGTERM received; stopping owned descendants")
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe relative path")
    resolved = (Path(root) / path).resolve()
    require(resolved.is_relative_to(Path(root).resolve()), "Path escaped owned root")
    return resolved


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode()
    require(len(encoded) < 64 * 1024 ** 2, "JSON exceeds artifact limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as handle:
            handle.write(encoded)
    else:
        temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
        with temporary.open("xb") as handle:
            handle.write(encoded)
        os.replace(temporary, path)


def expected_jobs(c):
    return [{"id": f"{data}__{seed}__{arm}", "host": host,
             "dataset": data, "seed": seed, "arm": arm}
            for host, groups in c["assignments"].items()
            for data, seed in groups for arm in ARMS]


def validate(c, *, source_root=None, operation_root=None, verify_files=True):
    require(c.get("schema") == SCHEMA and c.get("stage") == "components", "Foreign component campaign")
    require(c["arms"] == list(ARMS) and c["seeds"] == list(SEEDS), "Arm or seed drift")
    require(c["training"] == TRAINING and c["parallelism"] == PARALLELISM, "Training or scheduling drift")
    require(set(c["hosts"]) == set(c["assignments"]) == {"5080", "5090"}, "Host scope drift")
    groups = [(data, seed) for rows in c["assignments"].values() for data, seed in rows]
    require(len(groups) == 9 and len(set(groups)) == 9
            and set(groups) == {(d, s) for d in DATASETS for s in SEEDS}, "Dataset/seed ownership drift")
    for data in DATASETS:
        require(sum(any(d == data for d, _ in rows) for rows in c["assignments"].values()) == 1,
                "One dataset's seeds cross GPU/Runtime owners")
    require(c["jobs"] == expected_jobs(c) and len(c["jobs"]) == 36, "Four-arm fit identities drift")
    require(c["evaluation_scope"] == "validation_only" and c["held_out_test_evaluated"] is False,
            "Held-out boundary drift")
    limits = c["limits"]
    require(limits["workers_per_host"] == 3 and limits["per_condition_seconds"] == 36 * 3600
            and limits["total_wall_seconds"] == 168 * 3600 and limits["qualification_seconds"] == 5400
            and limits["automatic_retry"] is False, "Lease or retry drift")
    require(limits["root_bytes"] == 8 * 1024 ** 3 and limits["min_free_bytes"] == 5 * 1024 ** 3
            and limits["cpu_rss_bytes"] == 16 * 1024 ** 3 and limits["gpu_fraction_max"] == .8,
            "Resource ceiling drift")
    old = c["source"]["frozen_science_files"]
    require(len(old) == 129 and sha_json(old) == FROZEN_CLOSURE
            and c["source"]["frozen_science_files_sha256"] == FROZEN_CLOSURE,
            "Frozen science129 drift")
    files = c["source"]["files"]
    require(len(files) == 131 and all(files.get(name) == digest for name, digest in old.items()),
            "Frozen science overwritten or extension count drift")
    require(set(files) - set(old) == {"models/TPPs/CountAwareTitanCNNGRUComponents.py",
                                    "paper/scripts/run_titantpp_cnn_gru_components.py"}, "Unexpected science extension")
    require(c["source"]["git_revision"] == FROZEN_REVISION
            and c["source"]["frozen_science_git_revision"] == FROZEN_REVISION
            and c["source"]["extension_provenance"] == "uncommitted component extension pinned by SHA256",
            "Extension falsely attributed to frozen commit")
    require(sha_json(files) == c["source"]["files_sha256"], "Source closure drift")
    require(sha_json(c["operation"]["files"]) == c["operation"]["files_sha256"], "Operation closure drift")
    require(all(name in c["operation"]["files"] for name in
                ("operation/campaign.py", "operation/frozen_runtime.py", "operation/qualification.py")),
            "Required sealed operation missing")
    require(c["operation"]["files"]["operation/frozen_runtime.py"] == FROZEN_RUNTIME_SHA,
            "Original trainer/operational core replaced")
    require(c["runner_modules"] == {arm: "paper.scripts.run_titantpp_cnn_gru_components" for arm in ARMS},
            "Runner route drift")
    require({d["dataset_id"] for d in c["datasets"]} == set(DATASETS), "Dataset records drift")
    for data in c["datasets"]:
        require(sha_json(data) == c["dataset_sha256"][data["dataset_id"]], "Dataset contract drift")
    for host, spec in c["hosts"].items():
        root = Path(spec["root"])
        require(root == Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts") / (SCHEMA + "_" + host),
                "Foreign owned root")
        require(spec["source_root"] == str(root / "source")
                and spec["operation_root"] == str(root / "operation")
                and spec["output_dir"] == str(root / "run"), "Root isolation drift")
        require(spec["python"] == PYTHONS[host] and spec["gpu_uuid"] == GPU_UUIDS[host], "Pinned Python or GPU drift")
        require(spec["runtime_expected"] == {"cuda": "13.0", "cudnn": 91900 if host == "5080" else 92000,
                                            "numpy": "2.4.4", "polars": "1.39.3", "torch": "2.11.0+cu130"},
                "Pinned native Runtime drift")
        require(spec["environment"]["PYTHONPATH"] == spec["source_root"]
                and spec["environment"]["SOURCE_REVISION"] == FROZEN_REVISION, "Environment source isolation drift")
    if verify_files:
        sr, op = Path(source_root or ROOT), Path(operation_root or Path(__file__).resolve().parent)
        require(os.environ.get("SOURCE_REVISION") == FROZEN_REVISION, "Source environment drift")
        for name, digest in {**files, **c["input_files"]}.items():
            require(sha_file(safe_relative(sr, name)) == digest, "Source/input integrity drift: " + name)
        for name, digest in c["operation"]["files"].items():
            require(sha_file(safe_relative(op.parent, name)) == digest, "Operation integrity drift: " + name)
    return c


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load sealed operation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap(c, host):
    global ROOT, common, base, runner, frozen
    ROOT = Path(c["hosts"][host]["source_root"]).resolve()
    validate(c, source_root=ROOT, operation_root=c["hosts"][host]["operation_root"])
    require(Path(__file__).resolve().parent == Path(c["hosts"][host]["operation_root"]), "Wrong sealed operation cwd")
    sys.path.insert(0, str(ROOT))
    common = importlib.import_module("paper.scripts.observed_slot_parallel_common")
    base = importlib.import_module("paper.scripts.run_multilag_detail_execution")
    prior = importlib.import_module("paper.scripts.run_local_detail_benchmark")
    shared = importlib.import_module("paper.scripts.run_observed_slot_partition")
    runner = importlib.import_module("paper.scripts.run_titantpp_cnn_gru_components")
    runner.install_hooks()
    for module in (common, base, prior, shared, runner):
        require(Path(module.__file__).resolve().is_relative_to(ROOT), "Import escaped sealed source")
    frozen = load_module("component_frozen_runtime", Path(c["hosts"][host]["operation_root"]) / "frozen_runtime.py")
    frozen.ROOT, frozen.common, frozen.base = ROOT, common, base
    frozen.prior, frozen.shared, frozen.engines = prior, shared, {arm: runner for arm in ARMS}
    frozen.validate, frozen.authorization = validate, authorization
    frozen.initial_states = initial_states
    frozen.verify_training_permit, frozen.PulseBudget = verify_training_permit, PulseBudget


def authorization(c, root, *, live=True):
    validate(c)
    require(not any((root / name).exists() for name in ("failure.json", "qualification/failure.json")),
            "Preserved failure; no automatic retry")
    approval, start = read(root / "approval.json"), read(root / "start_permit.json")
    host = next(h for h, spec in c["hosts"].items() if Path(spec["root"]) == root)
    require(approval["approved"] is True and approval["contract_sha256"] == sha_json(c)
            and host in approval["hosts"] and approval["user_instruction"], "User launch approval missing")
    require(start["host"] == host and start["contract_sha256"] == sha_json(c)
            and start["approval_sha256"] == sha_json(approval), "Start permit binding drift")
    require(start["deadline_unix"] - start["started_at_unix"] == c["limits"]["total_wall_seconds"], "Extended host lease")
    if live:
        require(start["started_at_unix"] <= time.time() < start["deadline_unix"], "Expired host lease")
    return start


def initial_states(data, seed):
    return runner.initial_states(data, seed, arms=ARMS)


def verify_qualification_summary(c, host, q):
    require(q["status"] == "passed" and q["host"] == host and q["contract_sha256"] == sha_json(c)
            and q["source_files_sha256"] == c["source"]["files_sha256"], "Foreign native qualification")
    require(q["runtime"]["gpu"]["uuid"] == c["hosts"][host]["gpu_uuid"]
            and q["runtime"]["python"] == "3.12.13"
            and all(q["runtime"][key] == value for key, value in c["hosts"][host]["runtime_expected"].items()),
            "Native GPU/Python/Runtime drift")
    require(set(q["inputs"]) == set(DATASETS) and set(q["initialization"]) == set(DATASETS),
            "Qualified input populations incomplete")
    for data in c["datasets"]:
        native_input = q["inputs"][data["dataset_id"]]
        require(sha_json(native_input["input_receipt"]) == native_input["input_receipt_sha256"]
                and native_input["input_receipt"]["held_out_materialized"] is False
                and native_input["input_receipt"]["populations"] == data["inherited_data_identity"]["populations"]
                and set(native_input["interface_sha256"]) == set(ARMS), "Qualified data identity drift")
        initial = q["initialization"][data["dataset_id"]]
        require(set(initial) == {str(seed) for seed in SEEDS}
                and all(set(initial[str(seed)]) == set(ARMS) for seed in SEEDS), "Native initialization incomplete")
    require(set(q["synthetic"]) == {str(seed) for seed in SEEDS}
            and all(q["synthetic"][str(seed)]["status"] == "passed"
                    and q["synthetic"][str(seed)]["seed"] == seed for seed in SEEDS)
            and q["causality_checks"]["status"] == "passed"
            and q["held_out_test_evaluated"] is False and q["scientific_optimizer_updates"] == 0,
            "Three-seed native synthetic/causality boundary drift")
    native_exit = q["native_qualification_exit"]
    require(native_exit["actual_child_exit_observed"] is True and native_exit["returncode"] == 0
            and native_exit["contract_sha256"] == sha_json(c), "Native qualification actual OS exit missing")
    admission = q["concurrency_admission"]
    require(admission["status"] == "passed" and admission["host"] == host and admission["workers"] == 3
            and admission["arms"] == [FULL, CNN, GRU] and admission["shape"] == [128, 256]
            and admission["scientific_optimizer_updates"] == 0
            and admission["timing_is_scientific_efficiency_evidence"] is False
            and admission["simultaneous_ready_verified"] is True
            and admission["contract_sha256"] == sha_json(c), "Native capacity admission incomplete")
    peak = admission["aggregate_peak_fraction_observed"]
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= .8, "Capacity GPU peak invalid")
    require(len(set(admission["GPU_witness"]["gpu_pids"])) == 3
            and not admission["final_GPU_observation"]["gpu_pids"], "Capacity GPU ownership witness incomplete")
    exits = admission["actual_child_exits"]
    require(set(exits) == {FULL, CNN, GRU} and all(x["returncode"] == 0
            and x["actual_child_exit_observed"] is True and x["contract_sha256"] == sha_json(c)
            for x in exits.values()) and len({x["worker_group_pid"] for x in exits.values()}) == 3,
            "Capacity actual exits incomplete")
    return True


def verify_training_permit(c, root):
    permit, start, q = read(root / "training_permit.json"), read(root / "start_permit.json"), read(root / "qualification/receipt.json")
    require(permit["contract_sha256"] == sha_json(c) and permit["start_permit_sha256"] == sha_json(start)
            and permit["qualification_sha256"] == sha_file(root / "qualification/receipt.json")
            and permit["qualification"] == q, "Training permit binding drift")
    require(q["status"] == "passed" and q["contract_sha256"] == sha_json(c)
            and q["host"] == start["host"], "Native qualification incomplete")
    bridge = permit.get("qualifications", {})
    require(set(bridge) == set(c["hosts"]), "Both native host qualifications required before any real fit")
    for host, binding in bridge.items():
        remote = binding["receipt"]
        require(binding["raw_sha256"] == sha_file(root / "qualification" / ("host_" + host + "_receipt.json"))
                and remote == read(root / "qualification" / ("host_" + host + "_receipt.json"))
                and remote["status"] == "passed" and remote["host"] == host
                and remote["contract_sha256"] == sha_json(c) and binding["os_exit_code"] == 0,
                "Paired-host qualification bridge mismatch")
        verify_qualification_summary(c, host, remote)
    require(bridge[start["host"]]["receipt"] == q, "Local/embedded native receipt mismatch")
    require(q["qualification_files"], "Native qualification raw manifest missing")
    for name, digest in q["qualification_files"].items():
        require(sha_file(safe_relative(root, name)) == digest, "Native admission file drift: " + name)
    return permit


def job_lease_path(root, name):
    require(name and "/" not in name and ".." not in name, "Unsafe job lease id")
    return root / "leases" / (name + ".json")


def owned_root_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


class PulseBudget:
    def __init__(self, root, digest, job, deadline, lease=False):
        self.root, self.digest, self.job, self.deadline, self.lease = root, digest, job, deadline, lease
        self.last = self.storage_check = 0.
        self.monotonic_deadline = time.monotonic() + max(0, deadline - time.time())

    def __call__(self):
        now = time.time()
        require(now < self.deadline and time.monotonic() < self.monotonic_deadline, "Absolute budget exhausted")
        if now - self.last < 2:
            return
        if self.lease:
            lease = read(job_lease_path(self.root, self.job))
            require(lease["contract_sha256"] == self.digest and lease["job"] == self.job
                    and now < lease["expires_unix"] <= self.deadline
                    and lease["worker_group_pid"] == os.getpgrp()
                    and lease["supervisor_pid"] != os.getpid(), "Per-job lease expired or foreign")
        require(shutil.disk_usage(self.root).free > 5 * 1024 ** 3, "Free disk below5GiB")
        rss = int(next(row.split()[1] for row in Path("/proc/self/status").read_text().splitlines()
                       if row.startswith("VmRSS:"))) * 1024
        require(rss < 16 * 1024 ** 3, "Worker CPU RSS over16GiB")
        if now - self.storage_check > 30:
            require(owned_root_bytes(self.root) < 8 * 1024 ** 3, "Owned output over8GiB")
            self.storage_check = now
        import torch
        if torch.cuda.is_initialized():
            fraction = .30 if self.lease else .8
            require(torch.cuda.max_memory_allocated() <= fraction * torch.cuda.get_device_properties(0).total_memory,
                    "CUDA allocation exceeds admitted fraction")
        write_json(self.root / "progress" / (self.job + ".json"), {"job": self.job, "pid": os.getpid(),
                   "updated_unix": now, "deadline_unix": self.deadline,
                   "source": "genuine_phase_or_batch_progress", "mac_required": False})
        self.last = now


def batch_groups(c, host):
    jobs = [j for j in c["jobs"] if j["host"] == host]
    batches = [jobs[i:i + 4] for i in range(0, len(jobs), 4)]
    require(bool(batches) and all(len(batch) == 4
            and len({(j["dataset"], j["seed"]) for j in batch}) == 1
            and [j["arm"] for j in batch] == list(ARMS) for batch in batches), "Four-arm group drift")
    return batches


def group_waves(batch):
    require(len(batch) == 4 and [job["arm"] for job in batch] == list(ARMS)
            and len({(job["host"], job["dataset"], job["seed"]) for job in batch}) == 1,
            "Wave must contain one owned four-arm group")
    return [batch[:3], batch[3:]]


def timeout_command(script, c, host, tail, deadline):
    remaining = int(deadline - time.time())
    require(remaining > 0, "No remaining native timeout budget")
    return ["timeout", "--signal=TERM", "--kill-after=15s", str(remaining),
            c["hosts"][host]["python"], str(script), "--contract",
            str(Path(c["hosts"][host]["root"]) / "execution_contract.json"), "--host", host, *tail]


def stop_owned(child):
    if child.poll() is not None:
        return
    require(os.getpgid(child.pid) == child.pid, "Owned child process group changed")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def observe_exit(child, command, digest, **fields):
    require(child.poll() is not None, "Child is still live")
    return {**fields, "returncode": child.returncode, "worker_group_pid": child.pid,
            "supervisor_pid": os.getpid(), "argv": command, "observed_unix": time.time(),
            "contract_sha256": digest, "actual_child_exit_observed": True}


def verify_gpu_membership(gpu_pids, processes, active, *, script, used_bytes, total_bytes):
    require(total_bytes > 0 and 0 <= used_bytes <= .8 * total_bytes, "Aggregate GPU memory over80 percent")
    for pid in gpu_pids:
        proc = processes.get(pid)
        require(proc is not None, "GPU PID absent from process snapshot")
        owners = [item for item in active.values() if proc["ppid"] == item["child"].pid]
        require(len(owners) == 1 and proc["pgrp"] == owners[0]["child"].pid, "Foreign GPU process group")
        argv, command = proc["argv"], owners[0]["command"]
        require(argv == command[4:] and str(script) in argv, "Foreign GPU worker command")
    return True


def gpu_observation(c, host, active, script):
    output = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=15).stdout
    rows = [row.strip().split(",") for row in output.splitlines() if row.strip()]
    match = [row for row in rows if row[0].strip() == c["hosts"][host]["gpu_uuid"]]
    require(len(match) == 1, "Pinned GPU missing")
    total, used = [float(match[0][i].strip()) * 1024 ** 2 for i in (1, 2)]
    pids, processes = common.gpu_pids(c["hosts"][host]), {}
    for pid in pids:
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            processes[pid] = {"ppid": int(stat[1]), "pgrp": int(stat[2]), "argv": argv}
        except FileNotFoundError:
            require(pid not in common.gpu_pids(c["hosts"][host]), "GPU process identity unavailable")
    verify_gpu_membership(set(processes), processes, active, script=script, used_bytes=used, total_bytes=total)
    return {"gpu_uuid": c["hosts"][host]["gpu_uuid"], "gpu_pids": sorted(processes),
            "processes": processes, "used_bytes": used, "total_bytes": total,
            "observed_unix": time.time()}


def validate_terminal(c, root, job):
    folder = root / "run" / job["id"]
    manifest = read(folder / "terminal_manifest.json")
    require(manifest["scientific_success"] is True and manifest["status"] == "complete"
            and manifest["job"] == job and manifest["contract_sha256"] == sha_json(c)
            and manifest["held_out_test_evaluated"] is False
            and manifest["evaluation_scope"] == "validation_only", "Missing/foreign terminal")
    for name, digest in manifest["files"].items():
        require(sha_file(safe_relative(folder, name)) == digest, "Terminal binary/file integrity mismatch")
    require(any(name.endswith("best_val_qty_rmse_model.pt") for name in manifest["files"])
            and any(name.endswith("last_epoch_state.pt") for name in manifest["files"]), "Selected/final binary missing")
    return {"terminal_manifest_sha256": sha_file(folder / "terminal_manifest.json")}


def write_parallel_pulse(c, host, root, start, active, results):
    now = time.time()
    require(now < start["deadline_unix"], "Campaign absolute deadline reached")
    require(len(active) <= 3, "Parallel worker ceiling exceeded")
    state = {}
    for name, item in active.items():
        require(now < item["deadline"], "Condition absolute deadline reached")
        write_json(job_lease_path(root, name), {"job": name, "contract_sha256": sha_json(c),
                   "issued_unix": now, "expires_unix": min(now + 90, item["deadline"]),
                   "supervisor_pid": os.getpid(), "worker_group_pid": item["child"].pid})
        state[name] = {"job": item["job"], "worker_group_pid": item["child"].pid,
                       "worker_guard_argv": item["command"], "condition_deadline_unix": item["deadline"]}
    write_json(root / "status.json", {"status": "running", "active_jobs": state,
               "completed": results, "supervisor_pid": os.getpid(), "supervisor_argv": [sys.executable, *sys.argv],
               "updated_unix": now, "deadline_unix": start["deadline_unix"], "maximum_concurrency": 3,
               "grouping": PARALLELISM["grouping"], "mac_required": False})


def run_fit(c, host, root, job, deadline):
    require(job in c["jobs"] and job["host"] == host, "Foreign fit")
    lease_path = job_lease_path(root, job["id"])
    until = min(time.time() + 10, deadline)
    while time.time() < until:
        try:
            lease = read(lease_path)
            if lease["worker_group_pid"] == os.getppid():
                break
        except FileNotFoundError:
            pass
        time.sleep(.05)
    else:
        raise ValueError("Matching per-job timeout lease not published")
    parent_stat = Path(f"/proc/{os.getppid()}/stat").read_text().rsplit(")", 1)[1].split()
    tokens = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
    require(os.getppid() == os.getpgrp() == lease["worker_group_pid"]
            and int(parent_stat[1]) == lease["supervisor_pid"], "Timeout PID/PPID ownership mismatch")
    require(len(tokens) == 16 and Path(tokens[0]).name == "timeout"
            and tokens[1:3] == ["--signal=TERM", "--kill-after=15s"]
            and float(tokens[3]) > 0 and tokens[4:] == [c["hosts"][host]["python"],
                str(Path(__file__).resolve()), "--contract", str(root / "execution_contract.json"),
                "--host", host, "--mode", "fit", "--job", job["id"], "--deadline", str(deadline)],
            "Timeout exact command/deadline mismatch")
    require(lease["job"] == job["id"] and lease["contract_sha256"] == sha_json(c), "Foreign fit lease")
    write_json(root / "worker_identities" / (job["id"] + ".json"), {"job": job, "pid": os.getpid(),
               "ppid": os.getppid(), "pgrp": os.getpgrp(), "supervisor_pid": lease["supervisor_pid"],
               "argv": [sys.executable, *sys.argv], "contract_sha256": sha_json(c),
               "started_unix": time.time(), "deadline_unix": deadline}, exclusive=True)
    # Only the legacy operational lease lookup is redirected; trainer, selector,
    # loss, optimizer, loader and endpoint replay remain the sealed function.
    def read_bridge(path):
        return read(lease_path if Path(path) == root / "server_lease.json" else path)
    frozen.read = read_bridge
    import torch
    torch.cuda.set_per_process_memory_fraction(.30, 0)
    common.ENVIRONMENT = {**common.ENVIRONMENT, "PYTHONHASHSEED": str(job["seed"])}
    frozen.run_fit(c, host, root, job, deadline)


def dispatch(c, host, root):
    start = authorization(c, root)
    verify_training_permit(c, root)
    require(shutil.which("timeout") is not None, "Native timeout unavailable")
    require(not common.gpu_pids(c["hosts"][host]), "GPU occupied before real fits")
    script = Path(__file__).resolve()
    with (root / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / "supervisor.claim", {"pid": os.getpid(), "ppid": os.getppid(),
                   "argv": [sys.executable, *sys.argv], "started_unix": time.time(),
                   "contract_sha256": sha_json(c)}, exclusive=True)
        results, active, batch = {}, {}, None
        try:
            for batch in batch_groups(c, host):
                validate(c)
                require(not active, "Prior group not terminal")
                for wave_index, wave in enumerate(group_waves(batch)):
                    gpu_observation(c, host, active, script)
                    for job in wave:
                        require(len(active) < 3 and time.time() < start["deadline_unix"], "Worker/host budget exceeded")
                        require(not (root / "run" / job["id"]).exists(), "Run exists; no implicit restart")
                        deadline = min(start["deadline_unix"], time.time() + c["limits"]["per_condition_seconds"])
                        write_json(root / "claims" / (job["id"] + ".json"), {"job": job,
                                   "contract_sha256": sha_json(c), "deadline_unix": deadline,
                                   "maximum_concurrency": 3, "wave_index": wave_index}, exclusive=True)
                        command = timeout_command(script, c, host, ["--mode", "fit", "--job", job["id"],
                                                  "--deadline", str(deadline)], deadline)
                        log = root / "logs" / (job["id"] + ".log")
                        log.parent.mkdir(exist_ok=True)
                        with log.open("xb") as stream:
                            child = subprocess.Popen(command, cwd=ROOT,
                                env={**os.environ, "PYTHONHASHSEED": str(job["seed"])},
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        active[job["id"]] = {"job": job, "child": child, "deadline": deadline, "command": command}
                        write_parallel_pulse(c, host, root, start, active, results)
                    while active:
                        write_parallel_pulse(c, host, root, start, active, results)
                        gpu_observation(c, host, active, script)
                        finished = []
                        for name, item in active.items():
                            if item["child"].poll() is None:
                                continue
                            exit_receipt = observe_exit(item["child"], item["command"], sha_json(c), job=item["job"])
                            write_json(root / "worker_exits" / (name + ".json"), exit_receipt, exclusive=True)
                            require(exit_receipt["returncode"] == 0, "Worker exited " + str(exit_receipt["returncode"]) + "; no automatic retry")
                            results[name] = {**validate_terminal(c, root, item["job"]),
                                             "worker_exit_sha256": sha_file(root / "worker_exits" / (name + ".json"))}
                            finished.append(name)
                        for name in finished:
                            del active[name]
                        if active:
                            time.sleep(5)
                    gpu_observation(c, host, active, script)
                require(all(j["id"] in results for j in batch), "Four-arm group not terminal")
                for job in batch:
                    exposure = root / "run" / job["id"] / "runs" / job["arm"] / frozen.VARIANT / f"seed_{job['seed']}" / "exposure.json"
                    frozen.audit_host_batch_prefix(c, host, root, job, read(exposure))
            write_json(root / "status.json", {"status": "complete", "completed": results,
                       "completed_conditions": len(results), "endpoint_roles": len(results) * 2,
                       "updated_unix": time.time(), "maximum_concurrency": 3, "mac_required": False,
                       "held_out_test_evaluated": False, "automatic_retry": False})
        except BaseException as exc:
            stopped = []
            for name, item in active.items():
                try:
                    stop_owned(item["child"])
                    stopped.append(observe_exit(item["child"], item["command"], sha_json(c), job=item["job"], cancelled_by_owned_failure=True))
                except BaseException as stop_exc:
                    stopped.append({"job": name, "stop_error": str(stop_exc)})
            failure = {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                       "active_batch": batch, "active_jobs": list(active), "completed": results,
                       "owned_children_stopped": stopped, "updated_unix": time.time(),
                       "traceback": traceback.format_exc(), "automatic_retry": False}
            write_json(root / "failure.json", failure, exclusive=True)
            write_json(root / "status.json", failure)
            raise


def qualify(c, host, root):
    start = authorization(c, root)
    require(not common.gpu_pids(c["hosts"][host]), "GPU occupied before native qualification")
    require(shutil.which("timeout") is not None, "Native timeout unavailable")
    write_json(root / "qualification.claim", {"pid": os.getpid(), "ppid": os.getppid(),
               "contract_sha256": sha_json(c), "started_unix": time.time()}, exclusive=True)
    deadline = min(start["deadline_unix"], time.time() + c["limits"]["qualification_seconds"])
    helper = load_module("component_qualification", root / "operation" / "qualification.py")
    # The controller never initializes CUDA; the completed native child must
    # release its context before the exactly three capacity workers start.
    child = None
    try:
        command = timeout_command(root / "operation" / "qualification.py", c, host,
                                  ["--mode", "native", "--deadline", str(deadline)], deadline)
        log = root / "qualification" / "native.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                require(time.time() < deadline, "Native qualification deadline reached")
                time.sleep(1)
        actual_exit = observe_exit(child, command, sha_json(c), phase="native_qualification")
        write_json(root / "qualification" / "native_exit.json", actual_exit, exclusive=True)
        require(actual_exit["returncode"] == 0, "Native qualification failed; no automatic retry")
        require(not common.gpu_pids(c["hosts"][host]), "Native qualification GPU context still live")
        q = read(root / "qualification" / "native_receipt.json")
        require(q["status"] == "passed" and q["contract_sha256"] == sha_json(c), "Foreign native receipt")
        admission = helper.capacity_admission(c, host, root, deadline, campaign=sys.modules[__name__])
        q["concurrency_admission"], q["native_qualification_exit"] = admission, actual_exit
        q["qualification_files"] = {str(path.relative_to(root)): sha_file(path)
            for folder in (root / "qualification", root / "concurrency_admission")
            for path in sorted(folder.rglob("*")) if path.is_file()}
        q["completed_unix"] = time.time()
        write_json(root / "qualification" / "receipt.json", q, exclusive=True)
    except BaseException as exc:
        if child is not None:
            stop_owned(child)
        write_json(root / "qualification" / "failure.json", {"type": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(), "automatic_retry": False,
                   "contract_sha256": sha_json(c)}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("qualify", "dispatch", "fit"), required=True)
    parser.add_argument("--job")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    c = read(args.contract)
    root = Path(c["hosts"][args.host]["root"])
    require(Path(args.contract).resolve() == root / "execution_contract.json", "Wrong contract path")
    require(Path.cwd().resolve() == Path(c["hosts"][args.host]["source_root"]), "Wrong frozen cwd")
    expected = dict(c["hosts"][args.host]["environment"])
    if args.mode == "fit":
        job = next(j for j in c["jobs"] if j["id"] == args.job)
        require(job["host"] == args.host and args.deadline is not None and math.isfinite(args.deadline), "Foreign fit/deadline")
        expected["PYTHONHASHSEED"] = str(job["seed"])
    require(all(os.environ.get(k) == v for k, v in expected.items()), "Native environment drift")
    bootstrap(c, args.host)
    with owned_signal_guard():
        if args.mode == "fit":
            run_fit(c, args.host, root, job, args.deadline)
        elif args.mode == "qualify":
            qualify(c, args.host, root)
        else:
            dispatch(c, args.host, root)


if __name__ == "__main__":
    main()
