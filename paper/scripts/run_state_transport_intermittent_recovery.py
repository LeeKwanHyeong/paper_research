#!/usr/bin/env python3
"""One-shot Intermittent completion: preserve B/control and train only elapsed-clock.

Old checkpoint identities stay intact. The fresh candidate records the actual
recovery source and its parent scientific contract; old artifacts are read-only.
Only standard-library modules are imported before authority/source/idle checks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import observed_slot_parallel_common as common
from paper.scripts import run_observed_slot_partition as shared
from paper.scripts import run_state_transport_execution as parent_runner

require = common.require
ARMS = parent_runner.ARMS
VARIANT = parent_runner.VARIANT
DATASET = "intermittent_frozen_5000"
PARENT_SHA = "842aa140e3c6af0e28b6a8d3526768d90e238768a7f72d36bc7205c37f3610ee"
NEW_ROOT = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/state_transport_intermittent_metadata_recovery_20260920_v1"
RECOVERY_CLOCK = {"parent_contract_sha256": PARENT_SHA,
                  "started_at_unix": 1789820694.4010592, "deadline_unix": 1789907094.4010592,
                  "maximum_explicit_recoveries": 1, "automatic_retry": False, "reset_clock": False}
ENTRYPOINT = "paper/scripts/run_state_transport_intermittent_recovery.py"
TEST_FILE = "simple_lab_test/search/tests/test_state_transport_intermittent_recovery.py"
SCOPE = {"dataset": DATASET, "reuse_arms": list(ARMS[:2]), "train_arm": ARMS[2],
         "epochs": 120, "seed": 42, "new_optimizer_steps": 369240,
         "replay_old_control": 2, "replay_new_candidate": 2}
LIMITS = {"total_wall_seconds": 86400, "max_concurrent_gpu_jobs_per_host": 1,
          "per_host_output_bytes": 4 * 1024**3, "min_free_bytes": 5 * 1024**3,
          "max_file_bytes": 64 * 1024**2}


def read(path):
    return json.loads(Path(path).read_text())


def arm_dir(root, arm):
    return Path(root) / "run" / DATASET / "runs" / arm / VARIANT / "seed_42"


def required_preserved_paths(old_root, parent=None):
    root = Path(old_root)
    paths = {root / "frozen_execution/execution_contract.json", root / "qualification/receipt.json",
             root / "approval.json", root / "start_permit.json", root / "training_permit.json",
             root / "run/status.json", root / "run/wrapper_manifest.json",
             root / "run" / DATASET / "input_receipt.json", root / "run" / DATASET / "initialization.json"}
    for arm in ARMS[:2]:
        paths.update(arm_dir(root, arm) / name for name in (
            "summary.json", "history.json", "exposure.json", "epoch_timing.json", "best_val_qty_rmse_model.pt", "last_epoch_state.pt"))
    paths.add(arm_dir(root, ARMS[0]) / "endpoint_replays.json")
    if parent is not None:
        paths.update(root / "source" / name for name in parent["source"]["files"])
    return {str(p) for p in paths}


def validate_contract(contract, *, verify_source=False):
    require(contract.get("schema") == "state_transport_intermittent_completion_v1", "Wrong completion schema")
    require(contract.get("metadata_recovery") == RECOVERY_CLOCK, "Fixed metadata recovery clock/scope changed")
    require(contract.get("required_source_directories") == ["sample_data"], "Required source layout changed")
    parent = contract.get("parent_contract", {})
    require(contract.get("parent_contract_sha256") == PARENT_SHA and common.sha_json(parent) == PARENT_SHA,
            "Wrong immutable parent contract")
    require(contract.get("host") == "5090" and contract.get("scope") == SCOPE
            and contract.get("limits") == LIMITS, "Recovery host/scope/budget changed")
    old_spec = parent["hosts"]["5090"]
    require(contract.get("old_root") == old_spec["root"], "Wrong preserved root")
    expected = {**old_spec, "root": NEW_ROOT, "source_root": NEW_ROOT + "/source",
                "output_dir": NEW_ROOT + "/run", "tmux": Path(NEW_ROOT).name + "_5090"}
    require(contract.get("host_spec") == expected, "Recovery host or isolation changed")
    environment = {**parent["limited_recovery"]["environment_overrides"]["5090"],
                   "XDG_CACHE_HOME": NEW_ROOT + "/cache"}
    require(contract.get("environment_overrides") == environment, "Recovery environment changed")
    source = contract.get("source", {})
    files = source.get("files", {})
    require(source.get("files_sha256") == common.sha_json(files)
            and source.get("base_git_revision") == parent["source"]["base_git_revision"], "Source identity mismatch")
    allowed_new = {ENTRYPOINT, TEST_FILE, "simple_lab_test/search/tests/test_count_aware_titan_state_transport.py"}
    parent_files = parent["source"]["files"]
    require(set(parent_files) <= set(files) and ENTRYPOINT in files
            and set(files) - set(parent_files) <= allowed_new, "Unexpected source addition/removal")
    changed = {name for name in parent_files if files[name] != parent_files[name]}
    require(changed <= {"models/TPPs/CountAwareTitanStateTransport.py"}, "Training/model computation source changed")
    preserved = contract.get("preserved_files", {})
    require(required_preserved_paths(contract["old_root"], parent) <= set(preserved), "Missing preserved artifact binding")
    require(all(preserved[str(Path(contract["old_root"]) / "source" / name)] == digest
                for name, digest in parent_files.items()), "Preserved parent source digest changed")
    for name, digest in preserved.items():
        path = Path(name)
        require(path.is_absolute() and path.is_relative_to(Path(contract["old_root"]))
                and ".." not in path.parts and isinstance(digest, str) and len(digest) == 64,
                "Escaped or malformed preserved artifact")
    if verify_source:
        directory = ROOT / "sample_data"
        require(directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir()), "Empty sample_data sentinel missing or changed")
        for name, digest in files.items():
            path = ROOT / name
            require(not Path(name).is_absolute() and ".." not in Path(name).parts
                    and path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink()
                    and common.sha_file(path) == digest, "Frozen recovery source changed: " + name)
    return expected


def verify_authority(contract, approval, permit, *, now=None):
    spec = validate_contract(contract)
    require(approval.get("explicit_metadata_recovery") is True, "Explicit metadata recovery approval required")
    digest = common.sha_json(contract)
    require(approval.get("approved") is True and approval.get("contract_sha256") == digest
            and approval.get("hosts") == ["5090"]
            and isinstance(approval.get("user_instruction"), str) and approval["user_instruction"].strip(),
            "Explicit recovery approval required")
    require(permit.get("schema") == "state_transport_intermittent_recovery_start_v1"
            and permit.get("contract_sha256") == digest
            and permit.get("approval_sha256") == common.sha_json(approval), "Recovery permit identity mismatch")
    started, deadline = permit.get("started_at_unix"), permit.get("deadline_unix")
    now = time.time() if now is None else now
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started <= now < deadline and deadline - started == LIMITS["total_wall_seconds"],
            "Expired, future, or extended recovery deadline")
    require(started == RECOVERY_CLOCK["started_at_unix"] and deadline == RECOVERY_CLOCK["deadline_unix"], "Recovery clock reset forbidden")
    return spec


def make_permit(contract, approval, *, now=None):
    current = time.time() if now is None else now
    started = RECOVERY_CLOCK["started_at_unix"]
    permit = {"schema": "state_transport_intermittent_recovery_start_v1", "contract_sha256": common.sha_json(contract),
              "approval_sha256": common.sha_json(approval), "started_at_unix": started,
              "deadline_unix": started + LIMITS["total_wall_seconds"]}
    verify_authority(contract, approval, permit, now=current)
    return permit


def verify_preserved(contract):
    for name, digest in contract["preserved_files"].items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(Path(contract["old_root"]).resolve())
                and common.sha_file(path) == digest, "Preserved artifact changed: " + name)
    require(read(Path(contract["old_root"]) / "frozen_execution/execution_contract.json") == contract["parent_contract"],
            "Remote parent contract differs")
    require(not arm_dir(contract["old_root"], ARMS[2]).exists(), "Parent candidate started; recovery duplication forbidden")
    receipt = read(Path(contract["old_root"]) / "qualification/receipt.json")
    parent = contract["parent_contract"]
    require(receipt.get("status") == "passed" and receipt.get("host") == "5090"
            and receipt.get("contract_sha256") == PARENT_SHA
            and receipt.get("source_files_sha256") == parent["source"]["files_sha256"], "Invalid parent qualification")
    # Keep the successful native qualification bound to its original approval,
    # both-host qualification permit, exact libraries and original common clock.
    old_root = Path(contract["old_root"])
    approval = read(old_root / "approval.json")
    start = read(old_root / "start_permit.json")
    training = read(old_root / "training_permit.json")
    parent_runner.verify_authorization(parent, approval, start, "5090")
    parent_runner.verify_authorization(parent, approval, training, "5090", training=True)
    require(start["started_at_unix"] == RECOVERY_CLOCK["started_at_unix"]
            and start["deadline_unix"] == RECOVERY_CLOCK["deadline_unix"]
            and training["qualifications"]["5090"]["receipt"] == receipt,
            "Parent qualification/permit binding differs")
    authority = read(old_root / "run/wrapper_manifest.json")["authority"]
    require(authority.get("contract_sha256") == PARENT_SHA
            and authority.get("approval_sha256") == common.sha_json(approval)
            and authority.get("permit_sha256") == common.sha_json(training)
            and authority.get("host") == "5090" and authority.get("command") == "train",
            "Preserved training authority changed")
    require(read(old_root / "run/status.json").get("status") == "stopped",
            "Parent partition is not stopped")
    return receipt


def runtime_contract(contract):
    adapted = deepcopy(contract["parent_contract"])
    adapted["hosts"]["5090"] = contract["host_spec"]
    adapted["limited_recovery"]["environment_overrides"]["5090"] = contract["environment_overrides"]
    return adapted


def apply_environment(contract):
    parent_runner.apply_recovery_environment(runtime_contract(contract), "5090")


def verify_fresh_output(spec):
    output = Path(spec["output_dir"])
    require(not output.exists() and not output.is_symlink(), "Fresh output required; no retry/resume")
    return output


def audit_history(summary, history, exposure, data, arm, initial, identity):
    require(summary.get("status") == "success" and summary.get("backbone") == arm
            and summary.get("variant") == VARIANT and summary.get("seed") == 42
            and summary.get("completed_epochs") == 120 and summary.get("epochs") == 120
            and summary.get("stopped_early") is False and summary.get("evaluation_scope") == "validation_only"
            and summary.get("held_out_test_evaluated") is False, "Completed arm identity mismatch")
    require(summary.get("initial_state_sha256") == initial and summary.get("resume_identity") == identity
            and summary.get("interface_meta") == identity["interface_meta"], "Saved source/interface/initialization changed")
    require([r.get("epoch") for r in history] == list(range(1, 121)), "Incomplete/noncontinuous history")
    values = [r["val_qty_rmse"] for r in history]
    require(all(type(v) in (int,float) and math.isfinite(v) for v in values), "Invalid RMSE history")
    best = min(range(120), key=lambda i: values[i]) + 1
    require(summary.get("best_epoch") == best and summary.get("checkpoint_monitor") == "validation_raw_quantity_rmse",
            "Strict earliest RMSE selector mismatch")
    require(summary.get("checkpoint_monitor_history_key") == "val_qty_rmse"
            and summary.get("checkpoint_selection") == "best_validation_raw_quantity_rmse", "Selector semantics changed")
    return shared.audit_exposure(exposure, data)


def audit_replays(endpoints, summary, history, data, initial, *, run):
    require(endpoints.get("best_epoch") == summary["best_epoch"]
            and endpoints.get("global_steps") == data["expected_global_steps"]
            and endpoints.get("initial_state_sha256") == initial
            and endpoints.get("last30") == shared.last30_summary(history), "Endpoint summary identity differs")
    for label, epoch in (("selected", summary["best_epoch"]), ("last", 120)):
        value, row = endpoints[label], history[epoch-1]
        path = run / ("best_val_qty_rmse_model.pt" if label == "selected" else "last_epoch_state.pt")
        require(value.get("checkpoint_path") == str(path)
                and value.get("count") == data["inherited_data_identity"]["populations"]["validation"]["target_count"]
                and value.get("evaluation_scope") == "validation_only" and value.get("held_out_test_evaluated") is False
                and value.get("time_metric") == parent_runner.TIME_METRIC
                and value.get("quantity_boundaries") == data["quantity_boundaries_all_train_rows"]
                and value.get("history_boundaries") == data["history_boundaries"], "Endpoint path/split/strata differs")
        require(all(math.isclose(value[metric], row[field], rel_tol=1e-10, abs_tol=1e-8)
                    for metric, field in (("qty_rmse", "val_qty_rmse"), ("qty_mae", "val_qty_mae"), ("time_nll", "val_time_nll"))),
                "Endpoint replay differs from training history")
    require(endpoints["selected"]["state_sha256"] == summary["checkpoint_state_sha256"], "Selected state receipt differs")


def ordered_recovery(load_existing, replay_existing, train_fresh):
    """Only this fresh arm may reach the optimizer; completed arms are read-only."""
    arms = {ARMS[0]: load_existing(ARMS[0]), ARMS[1]: replay_existing(ARMS[1])}
    arms[ARMS[2]] = train_fresh(ARMS[2])
    return arms


class Budget:
    def __init__(self, contract, permit, owner_pid):
        self.contract, self.permit, self.owner_pid = contract, permit, owner_pid
        self.last_scan = -math.inf
        self.monotonic_deadline = shared.fixed_start(permit, wall=time.time, monotonic=time.monotonic,
                                                     total_seconds=LIMITS["total_wall_seconds"])

    def __call__(self):
        require(os.getppid() == self.owner_pid, "Owned supervisor exited")
        require(time.time() < self.permit["deadline_unix"] and time.monotonic() < self.monotonic_deadline,
                "Recovery execution deadline reached")
        if time.monotonic() - self.last_scan >= 10:
            shared.check_storage(self.contract["host_spec"]["output_dir"], LIMITS)
            require(common.gpu_pids(self.contract["host_spec"]) <= {os.getpid()}, "Other GPU job appeared")
            self.last_scan = time.monotonic()


def production(contract, permit, qualification, budget):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    parent = contract["parent_contract"]
    parent_runner.verify_recovery_environment(runtime_contract(contract), "5090", require_active=True)
    runtime = common.runtime_check({"hosts": {"5090": contract["host_spec"]}}, "5090")
    require(runtime == qualification["runtime"], "Runtime changed since successful CUDA qualification")
    data = next(d for d in parent["datasets"] if d["dataset_id"] == DATASET)
    require(data["expected_global_steps"] == contract["scope"]["new_optimizer_steps"], "Candidate optimizer budget changed")
    budget()
    frame, metadata = prepare_quantity_comparison_data(data)
    metadata = bind_frozen_statistics(data, metadata)
    old_dataset = Path(contract["old_root"]) / "run" / DATASET
    require(metadata == read(old_dataset / "input_receipt.json") and metadata["held_out_materialized"] is False,
            "Input statistics/population changed")
    interface = parent_runner.time_interface(data, frame, parent)
    initial = parent_runner.initialization(data)
    require(initial == read(old_dataset / "initialization.json"), "Shared initial state changed")
    output = Path(contract["host_spec"]["output_dir"])
    dest = output / DATASET
    dest.mkdir(exist_ok=False)
    common.write_json(dest / "input_receipt.json", metadata, exclusive=True)
    common.write_json(dest / "initialization.json", initial, exclusive=True)
    q = {"boundaries": data["quantity_boundaries_all_train_rows"],
         "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(5)]}
    args = parent_runner.training_args(parent, data, dest)
    candidate_interface = {**deepcopy(interface), "source_files_sha256": contract["source"]["files_sha256"],
                           "execution_contract_sha256": common.sha_json(contract),
                           "parent_scientific_contract_sha256": PARENT_SHA}
    identities = {arm: training._resume_identity(args=args, backbone=arm, quantity_variant=VARIANT, seed=42,
                    monitor="validation_raw_quantity_rmse",
                    interface_meta=candidate_interface if arm == ARMS[2] else interface,
                    quantity_contract=q) for arm in ARMS}
    saved, exposures = {}, {}
    for arm in ARMS[:2]:
        run = arm_dir(contract["old_root"], arm)
        summary, history, exposure = read(run / "summary.json"), read(run / "history.json")["history"], read(run / "exposure.json")
        audit_history(summary, history, exposure, data, arm, initial[arm], identities[arm])
        require(summary["checkpoint_path"] == str(run / "best_val_qty_rmse_model.pt"), "Preserved checkpoint path differs")
        saved[arm] = (summary, history, run)
        exposures[arm] = exposure
    require(exposures[ARMS[0]] == exposures[ARMS[1]], "Completed arms saw different batches")

    def load_existing(arm):
        budget()
        summary, history, run = saved[arm]
        endpoints = read(run / "endpoint_replays.json")
        audit_replays(endpoints, summary, history, data, initial[arm], run=run)
        return endpoints

    def endpoints_for(arm, summary, history, run):
        result = {}
        for label, epoch in (("selected", summary["best_epoch"]), ("last", 120)):
            budget()
            common.write_json(output / "status.json", {"status": "endpoint_replay", "host": "5090", "dataset": DATASET,
                "backbone": arm, "endpoint": label, "epoch": epoch, "deadline_unix": permit["deadline_unix"]})
            path = run / ("best_val_qty_rmse_model.pt" if label == "selected" else "last_epoch_state.pt")
            result[label] = parent_runner.replay_checkpoint(path, data, frame, budget, expected_arm=arm,
                expected_identity=identities[arm], expected_initial=initial[arm], expected_epoch=epoch)
        result.update(last30=shared.last30_summary(history), global_steps=data["expected_global_steps"],
                      best_epoch=summary["best_epoch"], initial_state_sha256=initial[arm])
        audit_replays(result, summary, history, data, initial[arm], run=run)
        gc.collect(); torch.cuda.empty_cache()
        return result

    def replay_existing(arm):
        result = endpoints_for(arm, *saved[arm])
        target = dest / "preserved_arms" / arm / "endpoint_replays.json"
        common.write_json(target, result, exclusive=True)
        return result

    def train_fresh(arm):
        require(arm == ARMS[2], "Only the previously unstarted candidate may train")
        budget(); validate_contract(contract, verify_source=True); verify_preserved(contract)
        require(common.runtime_check({"hosts": {"5090": contract["host_spec"]}}, "5090") == runtime, "Runtime drift before candidate")
        run = arm_dir(contract["host_spec"]["root"], arm)
        require(not run.exists(), "Fresh candidate required; retry/resume forbidden")
        def status(epoch, records):
            common.write_json(output / "status.json", {"status": "training", "host": "5090", "dataset": DATASET,
                "backbone": arm, "epoch": epoch, "global_steps": sum(r["batches"] for r in records["train"]),
                "preserved_optimizer_steps": 2 * data["expected_global_steps"], "deadline_unix": permit["deadline_unix"]})
            common.write_json(run / "exposure.json", records)
        with shared.audited_training(training, data, budget, status) as exposure:
            summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=q,
                interface_meta=candidate_interface, backbone=arm, quantity_variant=VARIANT, seed=42)
        exposures[arm] = exposure
        common.write_json(run / "exposure.json", exposure)
        history = read(run / "history.json")["history"]
        audit_history(summary, history, exposure, data, arm, initial[arm], identities[arm])
        require(exposure == exposures[ARMS[0]], "Candidate actual batch exposure differs")
        result = endpoints_for(arm, summary, history, run)
        common.write_json(run / "endpoint_replays.json", result, exclusive=True)
        return result

    arms = ordered_recovery(load_existing, replay_existing, train_fresh)
    budget(); verify_preserved(contract)
    require(set(exposures) == set(ARMS) and all(v == exposures[ARMS[0]] for v in exposures.values()), "Actual exposures differ")
    result = {"status": "complete", "dataset": DATASET, "host": "5090", "arms": arms, "exposure_equal": True,
              "acceptance": parent_runner.compare_arms(arms), "time_metric": parent_runner.TIME_METRIC,
              "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    common.write_json(dest / "paired_comparison.json", result, exclusive=True)
    return {"status": "complete", "host": "5090", "datasets": {DATASET: result}, "contract_sha256": common.sha_json(contract),
            "parent_contract_sha256": PARENT_SHA, "actual_source_files_sha256": contract["source"]["files_sha256"],
            "preserved_checkpoint_source_files_sha256": parent["source"]["files_sha256"],
            "candidate_checkpoint_source_files_sha256": contract["source"]["files_sha256"],
            "source_difference": "metadata validation tolerance and completion orchestration only; computation unchanged",
            "deadline_unix": permit["deadline_unix"], "new_optimizer_steps": arms[ARMS[2]]["global_steps"],
            "preserved_optimizer_steps": sum(arms[arm]["global_steps"] for arm in ARMS[:2]),
            "total_optimizer_steps": sum(value["global_steps"] for value in arms.values()), "endpoint_replays": 6, "reused_endpoint_replays": 2, "new_endpoint_replays": 4,
            "preserved_files_unchanged": True, "evaluation_scope": "validation_only", "held_out_test_evaluated": False}


def authority_record(contract, approval, permit, owner_pid):
    return {"contract_sha256": common.sha_json(contract), "approval_sha256": common.sha_json(approval),
            "permit_sha256": common.sha_json(permit), "owner_pid": owner_pid, "command": "complete", "host": "5090"}


def supervisor(contract_path, approval_path, permit_path):
    contract, approval, permit = read(contract_path), read(approval_path), read(permit_path)
    spec = verify_authority(contract, approval, permit)
    validate_contract(contract, verify_source=True); shared.verify_host_paths(spec, __file__); verify_preserved(contract)
    require(not common.gpu_pids(spec), "GPU busy; existing jobs preserved")
    output = verify_fresh_output(spec)
    output.mkdir(parents=True, exist_ok=False)
    apply_environment(contract)
    deadline = shared.fixed_start(permit, wall=time.time, monotonic=time.monotonic, total_seconds=LIMITS["total_wall_seconds"])
    authority = authority_record(contract, approval, permit, os.getpid())
    common.write_json(output / "wrapper_manifest.json", {"authority": authority,
        "started_at_unix": time.time(), "deadline_unix": permit["deadline_unix"], "resume": False, "explicit_metadata_recovery": 1, "automatic_retry": False,
        "actual_source_files_sha256": contract["source"]["files_sha256"], "parent_contract_sha256": PARENT_SHA,
        "preserved_checkpoint_source_files_sha256": contract["parent_contract"]["source"]["files_sha256"],
        "candidate_checkpoint_source_files_sha256": contract["source"]["files_sha256"],
        "preserved_files": contract["preserved_files"]}, exclusive=True)
    read_fd, write_fd = os.pipe(); child = None
    old_signal = signal.signal(signal.SIGTERM, shared._supervisor_terminated)
    try:
        with (output / "worker.log").open("xb") as stream:
            child = subprocess.Popen([spec["python"], str(Path(__file__).resolve()), "_worker", "--contract", str(Path(contract_path).resolve()),
                "--approval", str(Path(approval_path).resolve()), "--permit", str(Path(permit_path).resolve()),
                "--owner-pid", str(os.getpid()), "--authority-fd", str(read_fd)], cwd=spec["source_root"],
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(read_fd,))
            os.close(read_fd); read_fd = None
            with os.fdopen(write_fd, "wb") as pipe:
                write_fd = None; pipe.write(json.dumps(authority).encode())
            while child.poll() is None:
                require(time.time() < permit["deadline_unix"] and time.monotonic() < deadline, "Recovery deadline reached")
                shared.check_storage(output, LIMITS); time.sleep(1.)
            require(child.returncode == 0, f"Owned worker failed: {child.returncode}")
            require(time.time() < permit["deadline_unix"] and time.monotonic() < deadline, "Recovery deadline reached before terminal audit")
            result = read(output / "partition_summary.json")
            require(result.get("status") == "complete" and result.get("contract_sha256") == common.sha_json(contract)
                    and result.get("new_optimizer_steps") == 369240 and result.get("new_endpoint_replays") == 4,
                    "Invalid completion receipt")
            verify_preserved(contract); shared.check_storage(output, LIMITS)
            return result
    except BaseException as error:
        if child is not None: shared.kill_owned_process_group(child)
        common.write_json(output / "status.json", {"status": "stopped", "error": str(error), "automatic_retry": False,
                                                   "deadline_unix": permit["deadline_unix"]})
        raise
    finally:
        signal.signal(signal.SIGTERM, old_signal)
        for fd in (read_fd, write_fd):
            if fd is not None: os.close(fd)


def worker(args):
    contract, approval, permit = read(args.contract), read(args.approval), read(args.permit)
    spec = verify_authority(contract, approval, permit)
    validate_contract(contract, verify_source=True); shared.verify_host_paths(spec, __file__)
    qualification = verify_preserved(contract)
    require(args.owner_pid == os.getppid() and os.getsid(0) == os.getpid(), "Owned process-group worker required")
    with os.fdopen(args.authority_fd, "rb") as pipe: authority = json.loads(pipe.read(4096))
    expected = authority_record(contract, approval, permit, args.owner_pid)
    require(authority == expected, "Inherited authority mismatch")
    output = Path(spec["output_dir"])
    require(read(output / "wrapper_manifest.json")["authority"] == expected, "Output ownership mismatch")
    require(not common.gpu_pids(spec), "GPU became busy; preserving existing jobs")
    apply_environment(contract)
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS["max_file_bytes"],) * 2)
    budget = Budget(contract, permit, args.owner_pid); budget()
    result = production(contract, permit, qualification, budget)
    budget(); common.write_json(output / "partition_summary.json", result, exclusive=True)
    common.write_json(output / "status.json", {"status": "complete", "host": "5090", "deadline_unix": permit["deadline_unix"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start-permit", "complete", "_worker"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--authority-fd", type=int)
    args = parser.parse_args()
    if args.command == "start-permit":
        require(args.permit is not None and args.owner_pid is None and args.authority_fd is None, "Permit output required")
        common.write_json(args.permit, make_permit(read(args.contract), read(args.approval)), exclusive=True)
    elif args.command == "_worker":
        require(args.permit is not None and args.owner_pid is not None and args.authority_fd is not None, "Worker authority required")
        worker(args)
    else:
        require(args.permit is not None and args.owner_pid is None and args.authority_fd is None, "Invalid supervisor arguments")
        supervisor(args.contract, args.approval, args.permit)


if __name__ == "__main__":
    main()
