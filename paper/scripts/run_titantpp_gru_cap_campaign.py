"""Isolated operations for the approved nine fixed GRU-residual-cap fits.

The preparation contract is immutable.  This module consumes only the separate
execution overlay and creates fresh, Validation-only candidate outputs.  It
never opens a held-out split, changes an existing CNN/FULL fit, or retries a
failed job.
"""
from __future__ import annotations

import argparse
import fcntl
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


CAPPED = "titantpp_cnn_gru54_residual_cap025"
PREPARATION_RAW = "fae92b1695483ee4c978a6f72ddff7628881fc9bb2af42b246508835bddf06c9"
PREPARATION_CANONICAL = "6b5564173eb6f9feed35bc89142bfdde50944c285c4a402d2a4f5b311e2983f0"
SEEDS = (42, 52, 62)
TRAINING = {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40,
            "patience": 40, "monitor": "validation_raw_quantity_rmse",
            "tie": "strict_earliest_finite_minimum", "warm_start": False}
ROOT = None
common = base = prior = shared = runner = None


class OwnedTermination(RuntimeError):
    """The supervisor was asked to stop its own children."""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
        return
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("x") as stream:
        stream.write(text)
    os.replace(temporary, path)


def safe_relative(root, name):
    root, name = Path(root).resolve(), Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe relative path")
    target = (root / name).resolve()
    require(target.is_relative_to(root), "Path escaped owned root")
    return target


def expected_jobs(contract):
    return [{"id": f"{data['dataset_id']}__{seed}__{CAPPED}", "host": host,
             "dataset": data["dataset_id"], "seed": seed, "arm": CAPPED}
            for host in ("5080", "5090")
            for data in contract["datasets"]
            for seed in SEEDS
            if any(job["host"] == host and job["dataset"] == data["dataset_id"] and job["seed"] == seed
                   and job["arm"] == CAPPED for job in contract["jobs"])]


def validate(contract, *, verify_files=True):
    """Validate only invariant cap9 operational scope, without reading Test."""
    require(contract["schema"] == "titantpp_gru_residual_cap_validation_20261014_v1", "Foreign cap study")
    require(contract["science_preparation"] == {"path": "preparation/execution_contract.json",
        "raw_sha256": PREPARATION_RAW, "canonical_sha256": PREPARATION_CANONICAL}, "Preparation bridge drift")
    require(contract["candidate"]["arm"] == CAPPED and contract["arms"] == [CAPPED]
            and contract["seeds"] == list(SEEDS), "Candidate identity drift")
    require(contract["training"] == TRAINING, "Training selector or budget drift")
    require(contract["evaluation_scope"] == "validation_only"
            and contract["held_out_test_evaluated"] is False, "Held-out scope is forbidden")
    approval = contract["execution_authorization"]
    require(approval["approved"] is True and approval["fresh_fit_count"] == 9
            and approval["Test"] is False and approval["automatic_retry"] is False
            and approval["preserve_original_CNN_FULL"] is True, "Execution authorization drift")
    require(contract["runner_modules"] == {CAPPED: "paper.scripts.run_titantpp_gru_residual_cap"},
            "Candidate runner drift")
    require(contract["parallelism"] == {"mode": "sequential_per_host", "workers_per_host": 1},
            "Concurrency drift")
    require(contract["limits"]["per_condition_seconds"] == 36 * 3600
            and contract["limits"]["total_wall_seconds"] == 168 * 3600
            and contract["limits"]["automatic_retry"] is False, "Lease/retry drift")
    jobs = contract["jobs"]
    require(len(jobs) == 9 and len({job["id"] for job in jobs}) == 9
            and all(job["arm"] == CAPPED and job["seed"] in SEEDS for job in jobs), "Job set drift")
    require({job["host"] for job in jobs} == {"5080", "5090"}, "Host assignment drift")
    require(sum(job["host"] == "5080" for job in jobs) == 7
            and sum(job["host"] == "5090" for job in jobs) == 2, "Approved host counts drift")
    require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source closure drift")
    require(sha_json(contract["operation"]["files"]) == contract["operation"]["files_sha256"], "Operation closure drift")
    for host, spec in contract["hosts"].items():
        root = Path(spec["root"])
        require(root.is_absolute() and spec["source_root"] == str(root / "source")
                and spec["operation_root"] == str(root / "operation"), "Pinned isolated root drift")
    if verify_files:
        source = Path(ROOT or contract["hosts"][next(iter(contract["hosts"]))]["source_root"])
        for name, expected in {**contract["source"]["files"], **contract["input_files"]}.items():
            require(sha_file(safe_relative(source, name)) == expected, "Source/input SHA mismatch: " + name)
        operation = Path(__file__).resolve().parent.parent
        for name, expected in contract["operation"]["files"].items():
            require(sha_file(safe_relative(operation, name)) == expected, "Operation SHA mismatch: " + name)
        bridge = Path(__file__).resolve().parent.parent / "preparation/execution_contract.json"
        require(sha_file(bridge) == PREPARATION_RAW and sha_json(read(bridge)) == PREPARATION_CANONICAL,
                "Frozen preparation bytes/canonical mismatch")
        prepared = read(bridge)
        for key in ("candidate", "arms", "seeds", "datasets", "input_files", "training", "objective_weights",
                    "limits", "jobs", "source", "analysis", "baseline_references", "baseline_references_sha256"):
            require(contract[key] == prepared[key], "Frozen science field changed: " + key)
    return contract


def bootstrap(contract, host):
    global ROOT, common, base, prior, shared, runner
    spec = contract["hosts"][host]
    ROOT = Path(spec["source_root"]).resolve()
    require(Path(__file__).resolve().parent == Path(spec["operation_root"]).resolve(), "Wrong operation path")
    validate(contract)
    sys.path.insert(0, str(ROOT))
    common = importlib.import_module("paper.scripts.observed_slot_parallel_common")
    base = importlib.import_module("paper.scripts.run_multilag_detail_execution")
    prior = importlib.import_module("paper.scripts.run_local_detail_benchmark")
    shared = importlib.import_module("paper.scripts.run_observed_slot_partition")
    runner = importlib.import_module(contract["runner_modules"][CAPPED])
    runner.install_hooks()
    for module in (common, base, prior, shared, runner):
        require(Path(module.__file__).resolve().is_relative_to(ROOT), "Import escaped isolated source")


def authorization(contract, root, *, live=True):
    root = Path(root)
    require(not (root / "failure.json").exists() and not (root / "qualification" / "failure.json").exists(),
            "Preserved failure; automatic retry is forbidden")
    approval, start = read(root / "approval.json"), read(root / "start_permit.json")
    host = next(name for name, spec in contract["hosts"].items() if Path(spec["root"]) == root)
    digest = sha_json(contract)
    require(approval["approved"] is True and approval["contract_sha256"] == digest
            and approval["hosts"] == [host] and approval["new_training_authorized"] is True,
            "Host-specific authorization missing")
    require(start["host"] == host and start["contract_sha256"] == digest
            and start["approval_sha256"] == sha_json(approval), "Start permit binding drift")
    require(start["deadline_unix"] - start["started_at_unix"] == contract["limits"]["total_wall_seconds"],
            "Host deadline changed")
    if live:
        require(start["started_at_unix"] <= time.time() < start["deadline_unix"], "Host lease expired")
    return start


def _host_job(contract, host, job_id):
    matches = [job for job in contract["jobs"] if job["id"] == job_id and job["host"] == host]
    require(len(matches) == 1, "Foreign job")
    return matches[0]


def _runtime(contract, host):
    return common.runtime_check(contract, host)


def _verify_runtime(contract, host, observed, *, seed=None):
    expected = contract["hosts"][host]["runtime_expected"]
    require(observed["python"] == "3.12.13" and observed["gpu"]["uuid"] == contract["hosts"][host]["gpu_uuid"],
            "GPU/Python identity mismatch")
    require(all(observed[key] == value for key, value in expected.items()), "Runtime library mismatch")
    environment = {key: value for key, value in contract["hosts"][host]["environment"].items()
                   if key in observed["environment"]}
    if seed is not None:
        environment["PYTHONHASHSEED"] = str(seed)
    require(observed["environment"] == environment, "Runtime environment mismatch")


def _qualification_data(contract, host, deadline):
    start = authorization(contract, Path(contract["hosts"][host]["root"]))
    require(not common.gpu_pids(contract["hosts"][host]), "GPU occupied before qualification")
    result = {"status": "checking", "host": host, "contract_sha256": sha_json(contract),
        "source_files_sha256": contract["source"]["files_sha256"], "runtime": _runtime(contract, host),
        "inputs": {}, "initialization": {}, "synthetic": {}, "scientific_optimizer_updates": 0,
        "held_out_test_evaluated": False, "real_Validation_predictions": False}
    _verify_runtime(contract, host, result["runtime"])
    native_path = Path(contract["hosts"][host]["operation_root"]) / "native_checks.py"
    native_spec = importlib.util.spec_from_file_location("cap9_native_checks", native_path)
    native = importlib.util.module_from_spec(native_spec)
    sys.modules[native_spec.name] = native
    native_spec.loader.exec_module(native)
    for data in contract["datasets"]:
        dataset = data["dataset_id"]
        if not any(job["host"] == host and job["dataset"] == dataset for job in contract["jobs"]):
            continue
        frame, receipt = base.prepare_admitted_data(data, data_root=ROOT)
        result["inputs"][dataset] = {"input_receipt": receipt, "input_receipt_sha256": sha_json(receipt),
            "interface_sha256": {CAPPED: sha_json(runner.time_interface(data, frame, contract, arm=CAPPED))}}
        result["initialization"][dataset] = {str(seed): {CAPPED: runner.initial_states(data, seed, arms=(CAPPED,))[CAPPED]}
            for seed in SEEDS}
        for seed in SEEDS:
            if any(job["host"] == host and job["dataset"] == dataset and job["seed"] == seed
                   for job in contract["jobs"]):
                result["synthetic"][f"{dataset}__{seed}"] = native.synthetic_check(runner, data, seed, device="cuda:0")
        del frame
    require(time.time() < min(start["deadline_unix"], deadline), "Qualification deadline exhausted")
    result["status"] = "passed"
    result["completed_unix"] = time.time()
    return result


def qualify_worker(contract, host, root):
    root = Path(root)
    start = authorization(contract, root)
    deadline = min(start["deadline_unix"], time.time() + contract["limits"]["qualification_seconds"])
    write_json(root / "qualification.claim", {"pid": os.getpid(), "ppid": os.getppid(),
        "argv": [sys.executable, *sys.argv], "contract_sha256": sha_json(contract)}, exclusive=True)
    receipt = _qualification_data(contract, host, deadline)
    # OS child-exit evidence belongs to the outer pipeline phase receipt, not this still-running process.
    receipt["native_qualification_exit"] = {"pending_outer_phase_exit_observation": True,
        "contract_sha256": sha_json(contract)}
    receipt["qualification_files"] = {str(path.relative_to(root)): sha_file(path)
        for path in sorted((root / "qualification").rglob("*")) if path.is_file()}
    write_json(root / "qualification" / "receipt.json", receipt, exclusive=True)
    return receipt


def qualify(contract, host, root):
    """Run native checks in an owned timeout process; outer process records its exit."""
    root = Path(root)
    start = authorization(contract, root)
    deadline = min(start["deadline_unix"], time.time() + contract["limits"]["qualification_seconds"])
    command = ["timeout", "--signal=TERM", "--kill-after=15s", str(max(1, int(deadline - time.time()))),
        contract["hosts"][host]["python"], str(Path(__file__).resolve()), "--contract", str(root / "execution_contract.json"),
        "--host", host, "--mode", "qualify-worker"]
    child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), start_new_session=True)
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(OwnedTermination("Qualification SIGTERM")))
    try:
        child.wait(timeout=max(1, int(deadline - time.time()) + 20))
    except BaseException:
        _cleanup_qualification_child(root, contract, child, command)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    exit_record = {"actual_child_exit_observed": True, "returncode": child.returncode,
        "worker_group_pid": child.pid, "pid": child.pid, "ppid": os.getpid(), "argv": command,
        "contract_sha256": sha_json(contract), "observed_unix": time.time()}
    write_json(root / "qualification" / "native_exit.json", exit_record, exclusive=True)
    require(child.returncode == 0, "Native qualification worker failed")
    receipt = read(root / "qualification" / "receipt.json")
    receipt["native_qualification_exit"] = exit_record
    receipt["qualification_files"] = {str(path.relative_to(root)): sha_file(path)
        for path in sorted((root / "qualification").rglob("*")) if path.is_file()
        and path.name != "receipt.json"}
    write_json(root / "qualification" / "receipt.json", receipt)
    return receipt


def _cleanup_qualification_child(root, contract, child, command):
    """Preserve the observed guard exit after an owned qualification interruption."""
    try:
        stop_owned(child)
        record = {"actual_child_exit_observed": child.returncode is not None,
            "returncode": child.returncode, "worker_group_pid": child.pid, "pid": child.pid,
            "ppid": os.getpid(), "argv": command, "contract_sha256": sha_json(contract),
            "status": "interrupted", "observed_unix": time.time()}
    except BaseException as exc:
        record = {"actual_child_exit_observed": child.poll() is not None,
            "returncode": child.returncode, "worker_group_pid": child.pid, "pid": child.pid,
            "ppid": os.getpid(), "argv": command, "contract_sha256": sha_json(contract),
            "status": "cleanup_error", "cleanup_error": str(exc), "observed_unix": time.time()}
    write_json(Path(root) / "qualification" / "native_exit.json", record, exclusive=True)
    return record


def verify_training_permit(contract, root):
    root = Path(root)
    permit, start, receipt = read(root / "training_permit.json"), read(root / "start_permit.json"), read(root / "qualification" / "receipt.json")
    require(permit["contract_sha256"] == sha_json(contract) and permit["start_permit_sha256"] == sha_json(start)
            and permit["qualification_sha256"] == sha_file(root / "qualification" / "receipt.json")
            and permit["qualification"] == receipt, "Training permit/qualification binding drift")
    require(receipt["status"] == "passed" and receipt["held_out_test_evaluated"] is False
            and receipt["scientific_optimizer_updates"] == 0, "Qualification scope failed")
    for name, digest in receipt["qualification_files"].items():
        require(sha_file(safe_relative(root, name)) == digest, "Qualification evidence changed")
    return receipt


class PulseBudget:
    def __init__(self, root, digest, job, deadline, *, lease=False):
        self.root, self.digest, self.job, self.deadline, self.lease = Path(root), digest, job, deadline, lease
        self.last = 0.

    def __call__(self):
        now = time.time()
        require(now < self.deadline, "Condition deadline exhausted")
        if now - self.last < 2:
            return
        if self.lease:
            lease = read(self.root / "leases" / (self.job + ".json"))
            require(lease["contract_sha256"] == self.digest and lease["job"] == self.job
                    and lease["worker_group_pid"] == os.getpgrp() and now < lease["expires_unix"] <= self.deadline,
                    "Foreign/expired worker lease")
        require(shutil.disk_usage(self.root).free > 5 * 1024 ** 3, "Free disk below 5GiB")
        self.last = now


def _time_interface(data, frame, contract):
    return runner.time_interface(data, frame, contract, arm=CAPPED)


def run_fit(contract, host, root, job, deadline):
    """Wire the sealed trainer to the cap adapter; no training logic is copied."""
    root = Path(root)
    start = authorization(contract, root)
    receipt = verify_training_permit(contract, root)
    require(job["arm"] == CAPPED and job in contract["jobs"] and job["host"] == host, "Foreign candidate fit")
    common.ENVIRONMENT = {**common.ENVIRONMENT, "PYTHONHASHSEED": str(job["seed"])}
    _verify_runtime(contract, host, _runtime(contract, host), seed=job["seed"])
    claim = read(root / "claims" / (job["id"] + ".json"))
    require(claim["job"] == job and claim["contract_sha256"] == sha_json(contract)
            and claim["deadline_unix"] == deadline <= start["deadline_unix"], "Foreign job claim")
    write_json(root / "worker_identities" / (job["id"] + ".json"), {"job": job,
        "pid": os.getpid(), "ppid": os.getppid(), "pgrp": os.getpgrp(),
        "argv": [sys.executable, *sys.argv], "contract_sha256": sha_json(contract),
        "started_unix": time.time(), "deadline_unix": deadline}, exclusive=True)
    runtime_path = Path(contract["hosts"][host]["operation_root"]) / "frozen_runtime.py"
    spec = importlib.util.spec_from_file_location("cap9_frozen_runtime", runtime_path)
    frozen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = frozen
    spec.loader.exec_module(frozen)
    frozen.ROOT, frozen.common, frozen.base = ROOT, common, base
    frozen.prior, frozen.shared, frozen.engines = prior, shared, {CAPPED: runner}
    frozen.validate, frozen.authorization = lambda c, **_: validate(c, verify_files=False), authorization
    frozen.initial_states = lambda data, seed: runner.initial_states(data, seed, arms=(CAPPED,))
    frozen.verify_training_permit = verify_training_permit
    # The sealed budget preserves root-size, RSS, monotonic, and CUDA guards.
    frozen.PulseBudget = frozen.PulseBudget
    frozen.run_fit(contract, host, root, job, deadline)


def _observe_exit(child, command, contract, job, supervisor_pid):
    try:
        identity = read(Path(contract["hosts"][job["host"]]["root"]) / "worker_identities" / (job["id"] + ".json"))
    except FileNotFoundError:
        return {"returncode": child.returncode, "actual_child_exit_observed": child.returncode is not None,
            "actual_guard_exit_only": True, "contract_sha256": sha_json(contract), "job": job,
            "worker_group_pid": child.pid, "supervisor_pid": supervisor_pid, "argv": command,
            "observed_unix": time.time()}
    return {"returncode": child.returncode, "actual_child_exit_observed": child.returncode is not None,
        "contract_sha256": sha_json(contract), "job": job, "worker_group_pid": child.pid,
        "pid": identity["pid"], "ppid": identity["ppid"],
        "scientific_worker_pid": identity["pid"], "scientific_worker_ppid": identity["ppid"],
        "supervisor_pid": supervisor_pid, "argv": command, "observed_unix": time.time()}


def stop_owned(child):
    if child is None or child.poll() is not None:
        return
    require(os.getpgid(child.pid) == child.pid, "Owned worker process group changed")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)


def dispatch(contract, host, root):
    root = Path(root)
    start = authorization(contract, root)
    verify_training_permit(contract, root)
    require(shutil.which("timeout") is not None, "Native timeout missing")
    require(not common.gpu_pids(contract["hosts"][host]), "GPU occupied before fresh fit")
    script = Path(__file__).resolve()
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(OwnedTermination("Supervisor SIGTERM")))
    try:
      with (root / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / "supervisor.claim", {"pid": os.getpid(), "ppid": os.getppid(),
            "argv": [sys.executable, *sys.argv], "started_unix": time.time(),
            "contract_sha256": sha_json(contract)}, exclusive=True)
        completed, child, current = {}, None, None
        try:
            for current in (job for job in contract["jobs"] if job["host"] == host):
                require(time.time() < start["deadline_unix"] and not (root / "run" / current["id"]).exists(),
                        "Deadline reached or fresh-run rule failed")
                require(not common.gpu_pids(contract["hosts"][host]), "GPU occupied before next fit")
                deadline = min(start["deadline_unix"], time.time() + contract["limits"]["per_condition_seconds"])
                write_json(root / "claims" / (current["id"] + ".json"), {"job": current,
                    "contract_sha256": sha_json(contract), "deadline_unix": deadline,
                    "maximum_concurrency": 1}, exclusive=True)
                command = ["timeout", "--signal=TERM", "--kill-after=15s", str(max(1, int(deadline - time.time()))),
                    contract["hosts"][host]["python"], str(script), "--contract", str(root / "execution_contract.json"),
                    "--host", host, "--mode", "fit", "--job", current["id"], "--deadline", str(deadline)]
                log = root / "logs" / (current["id"] + ".log")
                log.parent.mkdir(exist_ok=True)
                with log.open("x") as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": str(current["seed"])},
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    now = time.time()
                    lease = {"job": current["id"],
                        "contract_sha256": sha_json(contract), "issued_unix": now,
                        "expires_unix": min(now + 90, deadline), "supervisor_pid": os.getpid(),
                        "worker_group_pid": child.pid}
                    write_json(root / "leases" / (current["id"] + ".json"), lease)
                    write_json(root / "server_lease.json", lease)
                    write_json(root / "status.json", {"status": "running", "active_jobs": {current["id"]: {
                        "job": current, "worker_group_pid": child.pid, "worker_guard_argv": command,
                        "condition_deadline_unix": deadline}}, "completed": completed, "supervisor_pid": os.getpid(),
                        "supervisor_argv": [sys.executable, *sys.argv], "updated_unix": now,
                        "deadline_unix": start["deadline_unix"], "maximum_concurrency": 1})
                    time.sleep(5)
                exit_record = _observe_exit(child, command, contract, current, os.getpid())
                write_json(root / "worker_exits" / (current["id"] + ".json"), exit_record, exclusive=True)
                require(child.returncode == 0, "Worker failed; automatic retry forbidden")
                manifest = read(root / "run" / current["id"] / "terminal_manifest.json")
                require(manifest["scientific_success"] is True and manifest["job"] == current
                        and manifest["contract_sha256"] == sha_json(contract), "Terminal manifest failed")
                for name, digest in manifest["files"].items():
                    require(sha_file(safe_relative(root / "run" / current["id"], name)) == digest,
                            "Terminal manifest SHA mismatch")
                completed[current["id"]] = {"worker_exit_sha256": sha_file(root / "worker_exits" / (current["id"] + ".json")),
                    "terminal_manifest_sha256": sha_file(root / "run" / current["id"] / "terminal_manifest.json")}
                child = None
            write_json(root / "status.json", {"status": "complete", "active_jobs": {}, "completed": completed,
                "completed_conditions": len(completed), "held_out_test_evaluated": False,
                "supervisor_pid": os.getpid(), "supervisor_argv": [sys.executable, *sys.argv],
                "updated_unix": time.time(), "maximum_concurrency": 1})
        except BaseException as exc:
            cleanup = None
            if child is not None:
                try:
                    stop_owned(child)
                    cleanup = _observe_exit(child, command, contract, current, os.getpid())
                    write_json(root / "worker_exits" / (current["id"] + ".json"), cleanup, exclusive=True)
                except BaseException as stop_exc:
                    cleanup = {"cleanup_error": str(stop_exc)}
            failure = {"status": "failed", "type": type(exc).__name__, "message": str(exc), "active_job": current,
                "cleanup": cleanup,
                "completed": completed, "traceback": traceback.format_exc(), "automatic_retry": False,
                "updated_unix": time.time(), "contract_sha256": sha_json(contract)}
            write_json(root / "failure.json", failure, exclusive=True)
            write_json(root / "status.json", failure)
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--host", required=True, choices=("5080", "5090"))
    parser.add_argument("--mode", required=True, choices=("qualify", "qualify-worker", "dispatch", "train", "fit"))
    parser.add_argument("--job")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    contract = read(args.contract)
    root = Path(contract["hosts"][args.host]["root"])
    require(Path(args.contract).resolve() == root / "execution_contract.json", "Wrong contract path")
    require(Path.cwd().resolve() == Path(contract["hosts"][args.host]["source_root"]).resolve(), "Wrong source cwd")
    bootstrap(contract, args.host)
    expected = dict(contract["hosts"][args.host]["environment"])
    if args.mode == "fit":
        require(args.job is not None and args.deadline is not None and math.isfinite(args.deadline), "Fit identity missing")
        job = _host_job(contract, args.host, args.job)
        expected["PYTHONHASHSEED"] = str(job["seed"])
    require(all(os.environ.get(key) == value for key, value in expected.items()), "Runtime environment drift")
    try:
        if args.mode == "qualify":
            qualify(contract, args.host, root)
        elif args.mode == "qualify-worker":
            qualify_worker(contract, args.host, root)
        elif args.mode in ("dispatch", "train"):
            dispatch(contract, args.host, root)
        else:
            run_fit(contract, args.host, root, job, args.deadline)
    except BaseException:
        if args.mode == "qualify":
            write_json(root / "qualification" / "failure.json", {"type": traceback.format_exc().splitlines()[-1],
                "automatic_retry": False, "contract_sha256": sha_json(contract)}, exclusive=True)
        raise


if __name__ == "__main__":
    main()
