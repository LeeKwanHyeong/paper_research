"""Explicitly approved, one-shot operational repair of the sealed retrieval pilot.

The scientific source, arguments, optimizer, data, selector and native resume
implementation remain byte-identical.  A new owner root records the one approved
epoch-boundary continuation and the previously unstarted fits. Import is inert.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
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
import statistics
import subprocess
import sys
import time
import traceback

SCHEMA = "titantpp_contextual_retrieval_pilot_repair_20261007_v1"
PARENT_SHA = "5372f2ec93c98e6b45ecf37d2b19ae1375a838fdfd9eb8a9c4f3a908456e7953"
SOURCE_SHA = "918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a"
PARENT_OPERATION_SHA = "e4c25f8b21443f264f70df27fa5f33f28a43df1716ec84b04357c80d88380597"
FIXED_ID = "5090__yellow_trip_hourly__42__titantpp_retrieval_fixed"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_path(root, name):
    name = Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe owned relative path")
    result = (Path(root) / name).resolve()
    require(result.is_relative_to(Path(root).resolve()), "Owned path escapes root")
    return result


def write(path, payload, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(data)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(data)
        os.replace(temporary, path)


def validate(c, *, bundle=None, verify_files=True):
    require(c["schema"] == SCHEMA, "Foreign repair contract")
    parent = c["parent_contract"]
    require(c["parent_contract_sha256"] == PARENT_SHA == sha_json(parent), "Parent scientific contract changed")
    require(parent["source"]["files_sha256"] == SOURCE_SHA
            and sha_json(parent["source"]["files"]) == SOURCE_SHA, "Frozen scientific source changed")
    require(parent["operation"]["files_sha256"] == PARENT_OPERATION_SHA
            and sha_json(parent["operation"]["files"]) == PARENT_OPERATION_SHA, "Parent operation changed")
    if "source" in c:
        require(c["source"] == parent["source"], "Repair source differs from parent")
    if "parent_operation" in c:
        require(c["parent_operation"] == parent["operation"], "Repair parent-operation bridge changed")
    require(c["limits"]["per_condition_seconds"] == 21600
            and c["limits"]["total_wall_seconds"] == 129600
            and c["limits"]["qualification_seconds"] == 1800, "Approved repair wall budget changed")
    require(c["policy"]["automatic_retry"] is False, "Automatic retry prohibited")
    expected = [j for j in parent["jobs"] if j["host"] == "5080"
                or j["arm"] not in ("titantpp_retrieval_mean", "titantpp_retrieval_weighted")]
    require(c["jobs"] == expected and len({j["id"] for j in c["jobs"]}) == 8,
            "Only original five 5080 and three remaining 5090 executions admitted")
    reused = [j for j in parent["jobs"] if j["host"] == "5090"
              and j["arm"] in ("titantpp_retrieval_mean", "titantpp_retrieval_weighted")]
    require(c["reused_jobs"] == reused and set(c["resume"]) == {FIXED_ID}, "Reuse/resume scope changed")
    require(c["resume"][FIXED_ID]["last_epoch"] == 148
            and c["resume"][FIXED_ID]["best_epoch"] == 120, "Approved resume boundary changed")
    require(set(c["hosts"]) == {"5080", "5090"}, "Foreign repair host")
    for host, spec in c["hosts"].items():
        original = parent["hosts"][host]
        require(Path(spec["root"]).is_absolute() and Path(spec["root"]).name == SCHEMA + "_" + host,
                "Repair owner root changed")
        require(Path(spec["source_root"]) == Path(spec["root"]) / "source"
                and Path(spec["operation_root"]) == Path(spec["root"]) / "operation", "Owner isolation changed")
        for key in ("python", "tmux_binary", "gpu_uuid", "runtime_expected", "assigned_datasets"):
            require(spec[key] == original[key], "Qualified runtime changed: " + key)
        require(spec["tmux"] == SCHEMA + "_" + host, "Repair tmux ownership changed")
        relocated_environment = deepcopy(original["environment"])
        relocated_environment.update(PYTHONPATH=spec["source_root"],
            MPLCONFIGDIR=str(Path(spec["root"]) / "cache/matplotlib"),
            XDG_CACHE_HOME=str(Path(spec["root"]) / "cache"))
        require(spec["environment"] == relocated_environment, "Qualified numerical environment changed")
    require(sha_json(c["operation"]["files"]) == c["operation"]["files_sha256"], "Repair operation closure changed")
    if verify_files:
        require(bundle is not None, "Owned bundle required")
        bundle = Path(bundle)
        for name, digest in parent["source"]["files"].items():
            require(sha_file(safe_path(bundle / "source", name)) == digest, "Source SHA mismatch: " + name)
        for name, digest in parent["input_files"].items():
            require(sha_file(safe_path(bundle / "source", name)) == digest, "Input SHA mismatch: " + name)
        for name, digest in parent["operation"]["files"].items():
            require(sha_file(safe_path(bundle / "parent_operation", name)) == digest,
                    "Parent operation SHA mismatch: " + name)
        for name, digest in c["operation"]["files"].items():
            require(sha_file(safe_path(bundle, name)) == digest, "Repair operation SHA mismatch: " + name)
    return c


def bootstrap(c, host, bundle):
    """Load audited parent functions after validating both immutable closures."""
    validate(c, bundle=bundle)
    source = Path(c["hosts"][host]["source_root"]).resolve()
    require(source == (Path(bundle) / "source").resolve() and Path.cwd().resolve() == source,
            "Wrong native frozen source cwd")
    require(Path(__file__).resolve() == (Path(bundle) / "operation/repair_runtime.py").resolve(),
            "Repair adapter outside owner root")
    sys.path.insert(0, str(Path(bundle) / "parent_operation/operation"))
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("sealed_parent_campaign", Path(bundle) / "parent_operation/operation/campaign.py")
    parent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parent
    spec.loader.exec_module(parent)
    parent.validate(c["parent_contract"], verify_files=False)
    parent.ROOT = source
    names = {"common": "observed_slot_parallel_common", "core": "run_titantpp_core_ablation",
             "prior": "run_local_detail_benchmark", "base": "run_multilag_detail_execution",
             "shared": "run_observed_slot_partition", "width": "run_titantpp_history_width",
             "checks": "titantpp_gru_control_checks"}
    for attribute, name in names.items():
        module = importlib.import_module("paper.scripts." + name)
        require(Path(module.__file__).resolve().is_relative_to(source), "Foreign scientific import")
        setattr(parent, attribute, module)
    parent.engine = importlib.import_module(c["parent_contract"]["runner_module"])
    require(Path(parent.engine.__file__).resolve().is_relative_to(source), "Foreign retrieval runner")
    parent.engine.install_hooks()
    parent.common.ENVIRONMENT = {**parent.common.ENVIRONMENT, "PYTHONHASHSEED": "42"}
    return parent


def authorization(c, host, root, *, live=True):
    require(not (root / "failure.json").exists(), "Recorded repair failure; no automatic retry")
    approval, start = read(root / "approval.json"), read(root / "start_permit.json")
    require(approval["approved"] is True and approval["contract_sha256"] == sha_json(c)
            and host in approval["hosts"] and approval["user_instruction"], "Repair approval missing")
    require(start["host"] == host and start["contract_sha256"] == sha_json(c)
            and start["approval_sha256"] == sha_json(approval), "Wrong repair start permit")
    beginning, end = start["started_at_unix"], start["deadline_unix"]
    require(type(beginning) in (int, float) and type(end) in (int, float)
            and math.isfinite(beginning) and math.isfinite(end)
            and end - beginning == c["limits"]["total_wall_seconds"], "Invalid repair lease")
    if live:
        require(beginning <= time.time() < end, "Repair lease not active")
    return start


def qualification_bridge(c, host, root, parent):
    binding = c["qualification_bindings"][host]
    path = safe_path(root, binding["receipt_file"])
    require(sha_file(path) == binding["receipt_sha256"], "Original native qualification raw SHA changed")
    q = read(path)
    parent.verify_qualification(c["parent_contract"], host, q)
    require(q["runtime"]["gpu"]["uuid"] == c["hosts"][host]["gpu_uuid"], "Qualification GPU changed")
    permit = read(root / "training_permit.json")
    require(permit["host"] == host and permit["contract_sha256"] == sha_json(c)
            and permit["start_permit_sha256"] == sha_json(read(root / "start_permit.json"))
            and permit["qualification_sha256"] == binding["receipt_sha256"], "Repair training permit changed")
    require(permit["qualification_parent_contract_sha256"] == PARENT_SHA,
            "Qualification inheritance must explicitly bind the scientific parent")
    return q


def verify_relocated_input_receipt(receipt, qreceipt, original_source, actual_source):
    """Only two resolved paths change; every semantic input field stays exact."""
    mapped = deepcopy(receipt)
    names = {"local_data_path": "inputs/data/yellow_trip_hourly/yellow_trip_hourly_with_split.parquet",
             "local_split_manifest_path": "inputs/data/yellow_trip_hourly/yellow_trip_hourly_split_manifest.json"}
    require(set(mapped["resolved_inputs"]) == set(qreceipt["resolved_inputs"]), "Resolved input keys changed")
    for key, relative in names.items():
        old, new = Path(original_source) / relative, Path(actual_source) / relative
        require(mapped["resolved_inputs"][key] == str(new)
                and qreceipt["resolved_inputs"][key] == str(old), "Unexpected relocated input path")
        require(sha_file(old) == sha_file(new), "Relocated input bytes differ")
        mapped["resolved_inputs"][key] = str(old)
    require(mapped == qreceipt, "Native Train/Validation semantic input receipt changed")
    return True


def validate_prefix(exposure, timing, epoch, data):
    require(set(exposure) == {"train", "validation"}, "Foreign exposure split")
    populations = data["inherited_data_identity"]["populations"]
    for split in ("train", "validation"):
        rows = exposure[split]
        require(len(rows) == epoch, "Resume exposure length does not match epoch boundary")
        for index, row in enumerate(rows, 1):
            require(row["epoch"] == index and row["split"] == split
                    and row["count"] == populations[split]["target_count"]
                    and row["batches"] == math.ceil(row["count"] / data["loader"]["batch_size"])
                    and len(row["batch_order_sha256"]) == 64, "Resume exposure accounting changed")
    require(len(timing["epochs"]) == epoch
            and [r["epoch"] for r in timing["epochs"]] == list(range(1, epoch + 1))
            and all(math.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > 0
                    for r in timing["epochs"]), "Resume completed-epoch timing changed")
    return True


@contextmanager
def audited_training(training, shared, budget, status, *, prefix=None, timing_prefix=None):
    """Use the parent's batch accounting, retaining the completed resume prefix."""
    records = deepcopy(prefix if prefix is not None else {"train": [], "validation": []})
    timings = deepcopy(timing_prefix or [])
    epoch_clock = {}
    original_loader, original_save = training.make_loader, training.atomic_torch_save
    def start_epoch():
        epoch_clock.update(monotonic=time.monotonic(), started_at_unix=time.time())
    def loader(*args, **kwargs):
        split = kwargs["target_split"]
        require(split in records, "Only Train/Validation exposure admitted")
        return shared.ExposureLoader(original_loader(*args, **kwargs), split, records[split], budget,
            on_start=start_epoch if split == "train" else None)
    def save(payload, path):
        budget()
        checkpoint = Path(path).name == "last_epoch_state.pt"
        if checkpoint:
            epoch = int(payload["epoch"])
            require(len(records["train"]) == len(records["validation"]) == epoch,
                    "Saved state lacks completed resumed epoch exposure")
            states = payload["optimizer_state_dict"]["state"]
            steps = sum(r["batches"] for r in records["train"])
            require(states and all(float(v["step"]) == steps for v in states.values()),
                    "Optimizer steps differ from resumed prefix accounting")
            row = payload["history"][-1]
            require(row["train_event_count"] == records["train"][-1]["count"]
                    and row["train_batch_count"] == records["train"][-1]["batches"], "Trainer telemetry mismatch")
        original_save(payload, path)
        budget()
        if checkpoint:
            require(epoch == len(timings) + 1, "Resumed timing discontinuity")
            timings.append({"epoch": epoch, "started_at_unix": epoch_clock["started_at_unix"],
                "finished_at_unix": time.time(), "elapsed_seconds": time.monotonic() - epoch_clock["monotonic"]})
            write(Path(path).parent / "epoch_timing.json", {
                "scope": "train_iteration_through_validation_and_checkpoint_save", "epochs": timings})
            status(epoch, records)
    training.make_loader, training.atomic_torch_save = loader, save
    try:
        yield records
    finally:
        training.make_loader, training.atomic_torch_save = original_loader, original_save


def copy_resume(c, job, run, data):
    binding = c["resume"][job["id"]]
    original = Path(binding["parent_run_dir"])
    expected_original = Path(c["parent_contract"]["hosts"][job["host"]]["root"]) / "run" / job["id"] / "runs" / job["arm"] / "count_only_log_regression/seed_42"
    require(original.resolve() == expected_original.resolve(), "Foreign original resume owner")
    required = {"last_epoch_state.pt", "history.json", "exposure.json", "epoch_timing.json", "train.log"}
    require(required <= set(binding["files"]) and "summary.json" not in binding["files"], "Incomplete or already-terminal resume admission")
    run.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name, digest in binding["files"].items():
        source, destination = safe_path(original, name), safe_path(run, name)
        require(sha_file(source) == digest, "Original resume file changed: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_stream, destination.open("xb") as target:
            shutil.copyfileobj(source_stream, target)
        require(sha_file(destination) == digest, "Resume copy SHA mismatch")
        copied[name] = digest
    exposure, timing = read(run / "exposure.json"), read(run / "epoch_timing.json")
    validate_prefix(exposure, timing, binding["last_epoch"], data)
    history = read(run / "history.json")["history"]
    require(len(history) == binding["last_epoch"]
            and [r["epoch"] for r in history] == list(range(1, binding["last_epoch"] + 1)), "History prefix changed")
    write(run / "resume_copy_receipt.json", {"schema": "retrieval_epoch_boundary_copy_v1",
        "parent_contract_sha256": PARENT_SHA, "repair_contract_sha256": sha_json(c),
        "parent_run_dir": str(original), "last_epoch": binding["last_epoch"],
        "best_epoch": binding["best_epoch"], "copied_raw_sha256": copied,
        "original_files_modified": False, "optimizer_updates": 0}, exclusive=True)
    return exposure, timing["epochs"]


def audit_cross_fit_prefix(c, host, root, job, exposure, parent):
    references = [(root / "run" / j["id"], j, sha_json(c)) for j in c["jobs"]
                  if j["host"] == host and j["id"] != job["id"]]
    references += [(Path(c["parent_contract"]["hosts"][host]["root"]) / "run" / j["id"], j, PARENT_SHA)
                   for j in c["reused_jobs"] if j["host"] == host]
    for folder, ref_job, digest in references:
        manifest_path = folder / "terminal_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read(manifest_path)
        name = f"runs/{ref_job['arm']}/{parent.VARIANT}/seed_42/exposure.json"
        require(manifest["job"] == ref_job and manifest["contract_sha256"] == digest
                and sha_file(safe_path(folder, name)) == manifest["files"][name], "Paired exposure evidence changed")
        ref = read(safe_path(folder, name))
        n = min(len(ref["train"]), len(exposure["train"]))
        require(ref["train"][:n] == exposure["train"][:n], "Actual paired Train batch prefix differs")
        require(len({r["batch_order_sha256"] for rows in (ref["validation"], exposure["validation"])
                     for r in rows}) == 1, "Full Validation batch order differs")


def worker(c, host, root, job, claim_path, parent):
    from paper.scripts.count_aware_tpp_backbone import training
    start = authorization(c, host, root)
    q = qualification_bridge(c, host, root, parent)
    claim = read(claim_path)
    require(claim["job"] == job and claim["contract_sha256"] == sha_json(c)
            and Path(claim_path).resolve() == (root / "claims" / (job["id"] + ".json")).resolve(), "Foreign repair worker claim")
    deadline = claim["deadline_unix"]
    require(deadline <= start["deadline_unix"] and deadline - claim["started_at_unix"] <= c["limits"]["per_condition_seconds"], "Worker budget exceeded")
    runtime_contract = deepcopy(c["parent_contract"])
    runtime_contract["hosts"] = c["hosts"]
    actual = parent.common.runtime_check(runtime_contract, host)
    parent.verify_fit_runtime(c["parent_contract"], host, q, job, actual)
    lease = read(root / "server_lease.json")
    require(os.getpgrp() == lease["worker_group_pid"] and os.getppid() != lease["supervisor_pid"], "Native timeout group missing")
    argv = subprocess.run(["ps", "-p", str(os.getppid()), "-o", "args="], capture_output=True, text=True, check=True).stdout.strip()
    require(argv.startswith("timeout --signal=TERM --kill-after=15s ") and str(claim_path) in argv, "Foreign timeout parent")
    budget = parent.PulseBudget(root, sha_json(c), job["id"], deadline, lease=True)
    budget()
    original = c["parent_contract"]
    data = next(d for d in original["datasets"] if d["dataset_id"] == job["dataset"])
    frame, receipt = parent.base.prepare_admitted_data(data, data_root=parent.ROOT)
    interface = parent.time_interface(data, frame, original, job["arm"])
    verify_relocated_input_receipt(receipt, q["inputs"][job["dataset"]]["input_receipt"],
        original["hosts"][host]["source_root"], parent.ROOT)
    require(sha_json(interface) == q["inputs"][job["dataset"]]["interface_sha256"][job["arm"]], "Scientific interface changed")
    initial = parent.initial_states(data, job["seed"])
    require(initial == q["initialization"][job["dataset"]][str(job["seed"])], "Original initialization changed")
    folder = root / "run" / job["id"]
    folder.mkdir(parents=True, exist_ok=False)
    args = parent.training_args(original, data, folder, job["arm"])
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.seeds, args.execution_role = str(job["seed"]), "fresh_contextual_retrieval_validation"
    quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
                "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(5)]}
    identity = training._resume_identity(args=args, backbone=job["arm"], quantity_variant=parent.VARIANT,
        seed=job["seed"], monitor="validation_raw_quantity_rmse", interface_meta=interface, quantity_contract=quantity)
    write(folder / "input_receipt.json", receipt, exclusive=True)
    write(folder / "initialization.json", initial, exclusive=True)
    run = folder / "runs" / job["arm"] / parent.VARIANT / "seed_42"
    exposure_prefix = timing_prefix = None
    if job["id"] in c["resume"]:
        exposure_prefix, timing_prefix = copy_resume(c, job, run, data)
        payload = training.torch_load_checkpoint(run / "last_epoch_state.pt", map_location="cpu")
        training.validate_checkpoint_route(payload, job["arm"])
        history, best = training._validate_resume_payload(payload, expected_identity=identity,
            expected_initial_state_sha256=initial[job["arm"]], monitor="validation_raw_quantity_rmse")
        binding = c["resume"][job["id"]]
        require(payload["epoch"] == binding["last_epoch"] and best["epoch"] == binding["best_epoch"]
                and history == read(run / "history.json")["history"], "Native checkpoint boundary/history differs")
        require(sum(r["batches"] for r in exposure_prefix["train"]) == 44400,
                "Original optimizer batch exposure changed")
        write(folder / "resume_startup_gate.json", {"status": "passed", "epoch": payload["epoch"],
            "best_epoch": best["epoch"], "native_identity_validated": True,
            "optimizer_rng_loader_and_selected_state_validated": True, "optimizer_updates": 0,
            "parent_checkpoint_sha256": binding["files"]["last_epoch_state.pt"],
            "execution_role_preserved": args.execution_role}, exclusive=True)
        del payload
    def status(epoch, records):
        budget()
        write(run / "exposure.json", records)
        names = ("last_epoch_state.pt", "history.json", "exposure.json", "epoch_timing.json", "train.log")
        write(run / "server_checkpoint_receipt.json", {"epoch": epoch, "contract_sha256": sha_json(c),
            "parent_contract_sha256": PARENT_SHA, "files": {n: sha_file(run / n) for n in names},
            "mac_ack_required": False, "time": time.time()})
        write(folder / "status.json", {"status": "training", "epoch": epoch,
            "global_steps": sum(r["batches"] for r in records["train"]), "job": job,
            "updated_unix": time.time(), "deadline_unix": deadline})
    with parent.strict_training_seed(training), audited_training(training, parent.shared, budget, status,
            prefix=exposure_prefix, timing_prefix=timing_prefix) as exposure:
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone=job["arm"], quantity_variant=parent.VARIANT, seed=job["seed"])
    history = read(run / "history.json")["history"]
    steps = parent.prior.audit_arm(history, summary, exposure, data, original["training"])
    require(summary["initial_state_sha256"] == initial[job["arm"]], "Initial tensor identity differs")
    endpoints = {}
    for label, path in (("selected", Path(summary["checkpoint_path"])), ("last", run / "last_epoch_state.pt")):
        epoch = summary["best_epoch"] if label == "selected" else len(history)
        endpoints[label] = parent.replay_checkpoint(path, data, frame, budget, expected_arm=job["arm"],
            expected_identity=identity, expected_initial=initial[job["arm"]], expected_epoch=epoch, expected_seed=job["seed"])
        parent.verify_full_validation(endpoints[label], data)
        parent.prior.audit_replay_accounting(endpoints[label])
        require(all(math.isclose(endpoints[label][k], history[epoch - 1][v], rel_tol=1e-10, abs_tol=1e-8)
            for k, v in (("qty_mae", "val_qty_mae"), ("qty_rmse", "val_qty_rmse"), ("time_nll", "val_time_nll"))), "Endpoint replay mismatch")
    audit_cross_fit_prefix(c, host, root, job, exposure, parent)
    result = {**endpoints, "status": "complete", "job": job, "best_epoch": summary["best_epoch"],
        "completed_epochs": len(history), "global_steps": steps, "initial_state_sha256": initial[job["arm"]],
        "parameter_count": summary["parameter_count"], "elapsed_seconds": summary["elapsed_seconds"],
        "last30": parent.shared.last30_summary(history),
        "first40": {"mean": statistics.mean(r["val_qty_rmse"] for r in history[:40]),
                    "sd": statistics.stdev(r["val_qty_rmse"] for r in history[:40]), "count": 40},
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "parent_contract_sha256": PARENT_SHA, "resume_parent_epoch": 148 if exposure_prefix else None}
    write(run / "endpoint_replays.json", result, exclusive=True)
    write(run / "exposure.json", exposure)
    write(folder / "status.json", {"status": "complete", "job": job, "updated_unix": time.time()})
    files = {str(p.relative_to(folder)): sha_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    write(folder / "terminal_manifest.json", {"status": "complete", "scientific_success": True,
        "files": files, "contract_sha256": sha_json(c), "parent_contract_sha256": PARENT_SHA,
        "job": job, "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "completed_unix": time.time()}, exclusive=True)


def stop_owned(child):
    if child.poll() is not None:
        return
    require(os.getpgid(child.pid) == child.pid, "Foreign child process group")
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def dispatch(c, host, root, parent):
    start = authorization(c, host, root)
    qualification_bridge(c, host, root, parent)
    require(not parent.common.gpu_pids(c["hosts"][host]), "GPU occupied before repair launch")
    require(shutil.which("timeout") is not None, "Independent native timeout unavailable")
    with (root / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "supervisor.claim", {"pid": os.getpid(), "started_unix": time.time(),
            "contract_sha256": sha_json(c)}, exclusive=True)
        child = job = None
        results = {}
        try:
            for job in (j for j in c["jobs"] if j["host"] == host):
                validate(c, bundle=root)
                require(time.time() < start["deadline_unix"], "Repair host deadline exceeded")
                require(not parent.common.gpu_pids(c["hosts"][host]), "Other GPU owner detected")
                require(not (root / "run" / job["id"]).exists(), "Repair fit already exists; restart forbidden")
                beginning = time.time()
                deadline = min(start["deadline_unix"], beginning + c["limits"]["per_condition_seconds"])
                claim = root / "claims" / (job["id"] + ".json")
                write(claim, {"job": job, "contract_sha256": sha_json(c),
                    "started_at_unix": beginning, "deadline_unix": deadline}, exclusive=True)
                def pulse():
                    now = time.time()
                    require(now < deadline, "Repair condition deadline exceeded")
                    write(root / "server_lease.json", {"job": job["id"], "contract_sha256": sha_json(c),
                        "issued_unix": now, "expires_unix": min(now + 90, deadline),
                        "supervisor_pid": os.getpid(), "worker_group_pid": child.pid if child else None})
                    write(root / "status.json", {"status": "running", "active_job": job,
                        "supervisor_pid": os.getpid(), "worker_group_pid": child.pid if child else None,
                        "completed": results, "updated_unix": now,
                        "deadline_unix": start["deadline_unix"], "condition_deadline_unix": deadline,
                        "mac_required": False})
                pulse()
                command = ["timeout", "--signal=TERM", "--kill-after=15s", str(max(1, int(deadline - time.time()))),
                    sys.executable, str(Path(__file__).resolve()), "--bundle", str(root), "--host", host,
                    "--mode", "worker", "--job", job["id"], "--claim", str(claim)]
                log = root / "logs" / (job["id"] + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("xb") as stream:
                    child = subprocess.Popen(command, cwd=parent.ROOT, env={**os.environ, "PYTHONHASHSEED": "42"},
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    pulse()
                    while child.poll() is None:
                        pulse()
                        time.sleep(5)
                    require(child.returncode == 0, "Worker exited " + str(child.returncode) + "; automatic retry prohibited")
                folder = root / "run" / job["id"]
                manifest = read(folder / "terminal_manifest.json")
                require(manifest["scientific_success"] is True and manifest["job"] == job
                        and manifest["contract_sha256"] == sha_json(c) and manifest["held_out_test_evaluated"] is False,
                        "Successful repair terminal missing")
                for name, digest in manifest["files"].items():
                    require(sha_file(safe_path(folder, name)) == digest, "Terminal SHA mismatch")
                results[job["id"]] = {"terminal_manifest_sha256": sha_file(folder / "terminal_manifest.json")}
                child = None
            write(root / "status.json", {"status": "complete", "completed": results,
                "completed_conditions": len(results), "updated_unix": time.time(), "mac_required": False})
        except BaseException as exc:
            if child is not None:
                stop_owned(child)
            failure = {"status": "failed", "type": type(exc).__name__, "message": str(exc),
                "active_job": job, "completed": results, "traceback": traceback.format_exc(),
                "updated_unix": time.time(), "automatic_retry": False}
            write(root / "failure.json", failure, exclusive=True)
            write(root / "status.json", failure)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("dispatch", "worker"), required=True)
    parser.add_argument("--job")
    parser.add_argument("--claim")
    args = parser.parse_args()
    root = Path(args.bundle).resolve()
    c = read(root / "execution_contract.json")
    require(root == Path(c["hosts"][args.host]["root"]).resolve(), "Wrong repair owner bundle")
    for name, value in c["hosts"][args.host]["environment"].items():
        require(os.environ.get(name) == value, "Numerical environment changed: " + name)
    require(os.environ.get("SOURCE_REVISION") == c["parent_contract"]["source"]["git_revision"], "Frozen source revision changed")
    parent = bootstrap(c, args.host, root)
    if args.mode == "dispatch":
        dispatch(c, args.host, root, parent)
    else:
        jobs = [j for j in c["jobs"] if j["id"] == args.job and j["host"] == args.host]
        require(len(jobs) == 1 and args.claim, "Unapproved repair worker")
        worker(c, args.host, root, jobs[0], Path(args.claim), parent)


if __name__ == "__main__":
    main()
