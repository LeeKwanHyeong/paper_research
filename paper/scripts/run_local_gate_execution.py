#!/usr/bin/env python3
"""Approval-bound observed local-gate CUDA qualification and fresh validation training.

No model/data/CUDA import occurs before approval, source, host and permit checks.
The same common wall-clock origin covers both qualifications, training and replay.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import observed_slot_parallel_common as common
from paper.scripts import prepare_local_gate_execution as proposal
from paper.scripts import run_observed_slot_partition as shared

require = common.require
ARMS = proposal.ARMS
VARIANT = "count_only_log_regression"
TIME_METRIC = "recorded_positive_integer_time_nll"
QUALIFICATION_CHECKS = ("shared_initial_state_rng", "initial_output_objective_common_gradient",
    "local_candidate_initial_tensors", "staged_all_branch_gradients", "prefix_target_padding_causality",
    "pmf_boundary_tail_topcode_gradients", "model_optimizer_rng_exact_resume",
    "projected_reference_forward_backward", "finite_native_updates")
ACCEPTANCE = {
    "overall_raw_rmse_ratio_strictly_less_than": 1.0, "overall_mae_ratio_max": 1.01,
    "body_mae_ratio_max": 1.02, "each_nonempty_middle_bin_mae_and_rmse_ratio_max": 1.02,
    "tail_mae_ratio_max": 1.02, "tail_rmse_ratio_strictly_less_than": 1.0,
    "recorded_time_nll_increase_max": .01, "last30_rmse_mean_ratio_max": 1.05,
    "last30_rmse_sample_sd_ratio_max": 1.05,
}


def read(path):
    return json.loads(Path(path).read_text())


def verify_authorization(contract, approval, permit, host, *, training=False, now=None):
    """Pure pre-runtime boundary; a prepared contract never grants authority."""
    proposal.validate_contract(contract, verify_source=False)
    require(host in proposal.ASSIGNMENTS, "Unrecognized host")
    digest = common.sha_json(contract)
    require(approval.get("approved") is True and approval.get("contract_sha256") == digest
            and set(approval.get("hosts", [])) == set(proposal.ASSIGNMENTS)
            and isinstance(approval.get("user_instruction"), str) and approval["user_instruction"].strip(),
            "Explicit approval bound to this contract and both hosts is required")
    require(permit.get("schema") == "local_gate_start_v1"
            and permit.get("contract_sha256") == digest
            and permit.get("approval_sha256") == common.sha_json(approval), "Permit authority mismatch")
    started, deadline = permit.get("started_at_unix"), permit.get("deadline_unix")
    now = time.time() if now is None else now
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started <= now < deadline
            and deadline - started == contract["limits"]["total_wall_seconds"],
            "Expired, future or extended common deadline")
    fixed = contract.get("launch", {}).get("fixed_started_at_unix")
    if fixed is not None:
        require(started == fixed and deadline == contract["launch"]["fixed_deadline_unix"],
                "Implementation correction must retain the original common deadline")
    if training:
        qualifications = permit.get("qualifications", {})
        require(host in qualifications and set(qualifications) <= set(proposal.ASSIGNMENTS), "This host native qualification required")
        for alias, entry in qualifications.items():
            receipt = entry.get("receipt", {})
            spec = contract["hosts"][alias]
            require(entry.get("receipt_sha256") == common.sha_json(receipt)
                    and receipt.get("status") == "passed" and receipt.get("host") == alias
                    and receipt.get("contract_sha256") == digest
                    and receipt.get("approval_sha256") == common.sha_json(approval)
                    and receipt.get("started_at_unix") == started
                    and receipt.get("deadline_unix") == deadline
                    and receipt.get("source_files_sha256") == contract["source"]["files_sha256"]
                    and receipt.get("synthetic_optimizer_updates") == 48
                    and receipt.get("device") == "cuda:0", "Native qualification identity mismatch")
            runtime = receipt.get("runtime", {})
            require(runtime.get("gpu", {}).get("uuid") == spec["gpu_uuid"], "Qualified GPU mismatch")
            require(all(runtime.get(k) == v for k, v in spec["runtime_expected"].items()),
                    "Qualified runtime mismatch")
            require(receipt.get("process_environment") == contract["process_environment"][alias]
                    and receipt.get("library_sha256") == contract["library_sha256"][alias],
                    "Qualified library/process environment mismatch")
            validate_qualification_measurements(contract, receipt)
    return contract["hosts"][host]


def validate_qualification_measurements(contract, receipt):
    expected = set(QUALIFICATION_CHECKS)
    require(set(receipt.get("checks", {})) == expected and all(v is True for v in receipt["checks"].values()),
            "Qualification correctness checks incomplete")
    require(receipt.get("real_data_loaded") is False and receipt.get("held_out_evaluated") is False,
            "Qualification scope changed")
    rows, gates = receipt.get("costs", []), contract["cost_gates"]
    require(len(rows) == 2 and {r.get("length") for r in rows} == {64,256}, "Qualification lengths changed")
    memory = receipt["runtime"]["gpu"].get("total_memory_bytes")
    require(type(memory) is int and memory > 0, "Missing device memory")
    for row in rows:
        measurements = row.get("measurements", {})
        require(row.get("batch_size") == 128 and set(measurements) == set(ARMS), "Qualification arms/batch changed")
        for value in measurements.values():
            samples = value.get("step_seconds", [])
            require(len(samples) == 5 and all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in samples),
                    "Five positive measured steps required")
            require(value.get("median_step_seconds") == statistics.median(samples)
                    and type(value.get("parameters")) is int and value["parameters"] > 0
                    and type(value.get("peak_allocated_bytes")) is int and value["peak_allocated_bytes"] > 0,
                    "Qualification numerical measurement mismatch")
        base, checks = measurements[ARMS[0]], {}
        require(measurements[ARMS[1]]["parameters"] + 2 == measurements[ARMS[2]]["parameters"], "Expected exact two-parameter gate difference")
        for arm in ARMS[1:]:
            c = measurements[arm]
            checks.update({arm+":step": c["median_step_seconds"] <= base["median_step_seconds"]*gates["median_cuda_step_ratio_max"],
                arm+":memory": c["peak_allocated_bytes"] <= base["peak_allocated_bytes"]*gates["peak_cuda_allocated_ratio_max"],
                arm+":device": c["peak_allocated_bytes"] <= memory*gates["device_memory_fraction_max"],
                arm+":parameters": c["parameters"] <= base["parameters"]*gates["parameter_ratio_max"]})
        require(row.get("checks") == checks and all(checks.values()), "Native CUDA cost gates failed or were misreported")


def source_and_host(contract, host):
    proposal.validate_contract(contract)
    shared.verify_host_paths(contract["hosts"][host], __file__)


def verify_process_environment(contract, host, *, active=False):
    """Verify existing libraries without loading CUDA or mutating packages."""
    env = contract["process_environment"][host]
    require(set(env) == {"LD_LIBRARY_PATH", "XDG_CACHE_HOME"}, "Unexpected process environment")
    root = Path(contract["hosts"][host]["root"])
    require(env["XDG_CACHE_HOME"] == str(root / "cache"), "Cache must remain in the new run root")
    lib = Path(env["LD_LIBRARY_PATH"])
    require(lib.is_absolute() and lib.is_dir(), "Pinned CUDA library directory missing")
    identities = contract["library_sha256"][host]
    require(isinstance(identities, dict) and identities, "Missing installed library identities")
    for name, digest in identities.items():
        path = Path(name)
        require(path.parent == lib and path.is_file() and not path.is_symlink()
                and common.sha_file(path) == digest, "Pinned CUDA library changed: " + name)
    if active:
        require(all(os.environ.get(k) == v for k, v in env.items()), "Process library/cache environment changed")
    return dict(env)


def apply_process_environment(contract, host):
    env = verify_process_environment(contract, host)
    common.apply_environment(contract["hosts"][host])
    cache = Path(env["XDG_CACHE_HOME"])
    require(not cache.is_symlink(), "Cache cannot be a symlink")
    cache.mkdir(mode=0o700, parents=False, exist_ok=True)
    require(cache.is_dir() and os.access(cache, os.W_OK | os.X_OK), "Cache is not writable")
    os.environ.update(env)
    verify_process_environment(contract, host, active=True)


def verify_local_qualification(contract, permit, host):
    local = read(Path(contract["hosts"][host]["root"]) / "qualification/receipt.json")
    require(local == permit["qualifications"][host]["receipt"], "Local/embedded qualification mismatch")
    return local


def storage_limits(contract):
    return {**contract["limits"], "max_file_bytes": 64 * 1024**2}


class Budget:
    def __init__(self, contract, permit, host, owner_pid, output):
        self.contract, self.permit, self.spec = contract, permit, contract["hosts"][host]
        self.owner_pid, self.output, self.last_scan = owner_pid, output, -math.inf
        self.deadline = shared.fixed_start(permit, wall=time.time, monotonic=time.monotonic, total_seconds=contract["limits"]["total_wall_seconds"])

    def __call__(self):
        require(os.getppid() == self.owner_pid, "Owned supervisor exited")
        require(time.time() < self.permit["deadline_unix"] and time.monotonic() < self.deadline,
                "Common execution deadline reached")
        if time.monotonic() - self.last_scan >= 10:
            shared.check_storage(self.output, storage_limits(self.contract))
            common.write_json(self.output / "gpu_observation.json", {"at_unix": time.time(),
                "other_pids": sorted(common.gpu_pids(self.spec) - {os.getpid()}),
                "policy": "observe_without_stopping"})
            self.last_scan = time.monotonic()


class ArmBudget:
    """One arm's training plus both endpoint replays share a wall-time cap."""
    def __init__(self, contract, host, dataset, arm, outer):
        self.outer, self.limit = outer, contract["limits"]["per_arm_wall_seconds"]
        self.started, self.monotonic_started = time.time(), time.monotonic()
        self.path = Path(contract["hosts"][host]["output_dir"]) / "active_arm.json"
        self.record = {"status": "active", "contract_sha256": common.sha_json(contract),
            "host": host, "dataset": dataset, "backbone": arm,
            "started_at_unix": self.started, "deadline_unix": self.started + self.limit,
            "started_monotonic": self.monotonic_started, "deadline_monotonic": self.monotonic_started + self.limit,
            "limit_seconds": self.limit}
        common.write_json(self.path, self.record)
        self()

    def __call__(self):
        self.outer()
        require(time.time() < self.started + self.limit
                and time.monotonic() < self.monotonic_started + self.limit, "Per-arm training/replay deadline reached")

    def finish(self):
        self()
        common.write_json(self.path, {**self.record, "status": "complete", "completed_at_unix": time.time()})


def verify_active_arm_deadline(contract, host):
    """Supervisor watchdog also enforces an arm cap during a blocked GPU call."""
    path = Path(contract["hosts"][host]["output_dir"]) / "active_arm.json"
    if not path.exists():
        return
    active = read(path)
    require(active.get("contract_sha256") == common.sha_json(contract) and active.get("host") == host
            and active.get("backbone") in ARMS
            and active.get("dataset") in contract["hosts"][host]["assigned_datasets"]
            and active.get("limit_seconds") == contract["limits"]["per_arm_wall_seconds"]
            and _finite(active.get("deadline_unix")) and _finite(active.get("started_at_unix"))
            and active["deadline_unix"] - active["started_at_unix"] == active["limit_seconds"]
            and _finite(active.get("deadline_monotonic")) and _finite(active.get("started_monotonic"))
            and active["deadline_monotonic"] - active["started_monotonic"] == active["limit_seconds"],
            "Active-arm watchdog identity changed")
    require(active.get("status") in ("active", "complete"), "Invalid arm watchdog state")
    if active["status"] == "active":
        require(time.time() < active["deadline_unix"] and time.monotonic() < active["deadline_monotonic"],
                "Per-arm training/replay deadline reached")


def build_model(data, backbone):
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {k: v for k, v in data["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    require(backbone in (*ARMS, proposal.UNGATED), "Unexpected local gate arm")
    model, metadata = build_count_aware_model(backbone, **config,
        train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])
    if backbone != ARMS[0]:
        module = model.multilag_detail
        lags = (1, 2, 4, 8, 16, 32, 64, 128)
        require(module.rank == 4 and module.branch_count == 8 and module.residual_divisor == 8
                and module.mode == "local" and tuple(module.availability_lags) == lags
                and tuple(module.lags) == (1,) * 8, "Frozen local-detail body changed")
        if backbone != proposal.UNGATED:
            require(model.local_gate.mode == ("time" if backbone == ARMS[1] else "quantity"), "Gate mode changed")
    return model, metadata


def training_args(contract, data, output):
    from paper.scripts.run_count_aware_tpp_backbone_control import parse_args
    # Obtain maintained defaults without entering the CLI's data execution path.
    old = sys.argv
    try:
        sys.argv = ["multilag_detail", "--data", data["inherited_data_identity"]["data"]["path"],
            "--split-manifest", data["inherited_data_identity"]["split_manifest"]["path"],
            "--output-dir", str(output), "--source-revision", contract["source"]["base_git_revision"],
            "--execution-role", "fresh_local_gate_observed_time_validation"]
        args = parse_args()
    finally:
        sys.argv = old
    model, loader, identity = data["model"], data["loader"], data["inherited_data_identity"]
    require(data["optimizer"] == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.},
            "Frozen optimizer changed")
    require(loader["batch_size"] == 128 and loader["num_workers"] == 0
            and loader["drop_last"] is False and loader["train_shuffle"] is True
            and loader["validation_shuffle"] is False, "Frozen loader changed")
    settings = dict(output_dir=Path(output), force_rerun=False, epochs=120, min_epochs=120,
        early_stopping_patience=120, batch_size=128, lookback_weeks=loader["lookback_weeks"],
        max_seq_len=loader["max_seq_len"], hidden_dim=64, lr=.001, weight_decay=.01, grad_clip=1.,
        lambda_log_qty=1., lambda_tail=0., quantile_adaptive_strength=0., device="cuda:0",
        checkpoint_monitor="validation_raw_quantity_rmse", model_role=contract["model_role"],
        execution_role="fresh_local_gate_observed_time_validation",
        dataset_contract=data["dataset_id"], source_revision=contract["source"]["base_git_revision"],
        data=Path(identity["data"]["path"]), data_sha256=identity["data"]["sha256"],
        split_manifest=Path(identity["split_manifest"]["path"]),
        split_manifest_sha256=identity["split_manifest"]["sha256"], max_series=None,
        max_train_batches=None, max_val_batches=None, allow_partial_contract=False,
        time_head_mode=model["time_head_mode"], time_scale=model["time_scale"],
        time_sigma_floor=model["time_sigma_floor"], time_head_lr_multiplier=1.,
        tail_threshold=float(data["quantity_boundaries_all_train_rows"][2]),
        tail_normalization_scale=float(data["quantity_boundaries_all_train_rows"][2]),
        tail_clip_cap=float(data["quantity_boundaries_all_train_rows"][3]))
    for key, value in settings.items():
        setattr(args, key, value)
    return args


def time_interface(data, frame, contract):
    import numpy as np
    import torch
    from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
    from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    from paper.scripts.count_aware_tpp_backbone.observed_time import validate_recorded_grid
    validate_recorded_grid(frame, data["dataset_id"])
    loader, frozen = data["loader"], data["time_statistics"]
    stats = derive_train_time_contract(frame, lookback_weeks=loader["lookback_weeks"], max_seq_len=loader["max_seq_len"])
    for measured, key in (("time_scale", "train_time_scale"), ("target_log_scaled_mean", "train_log_scaled_mean"),
                          ("target_log_scaled_std", "train_log_scaled_std")):
        require(math.isclose(stats[measured], frozen[key], rel_tol=0, abs_tol=1e-12), "Train time statistics mismatch")
    for split in ("train", "validation"):
        dataset = RMTPPWeekLookbackDataset(frame, lookback_weeks=loader["lookback_weeks"],
            max_seq_len=loader["max_seq_len"], val_ratio=.2, mode="all", split_col="chronological_split", target_splits={split})
        values = torch.tensor(np.asarray([max(1., float(dataset.dt_lists[p][i+1])) for p, i in dataset.index], dtype=np.float64))
        require(values.numel() == frozen[f"expected_{split}_targets"]
                and target_dt_sha256(values) == frozen[f"expected_{split}_target_dt_sha256"], "Duration target identity mismatch")
    model = data["model"]
    return {"train_target_mean": data["statistics"]["train_log_mean"],
        "train_target_std": data["statistics"]["train_log_std"], "statistics": data["statistics"],
        "data_scope": "verified_train_validation_only", "execution_contract_sha256": common.sha_json(contract),
        "source_files_sha256": contract["source"]["files_sha256"],
        "backbone_design": {"schema": "hard_lmm_local_gate_design_v1",
            "design_sha256": contract["design_sha256"],
            "architecture": contract["architecture"],
            "mode_by_arm": {ARMS[0]: None, ARMS[1]: "time", ARMS[2]: "quantity"},
            "train_time_scale": model["time_scale"],
            "time_statistics": data["time_statistics"]},
        "time_head": {"mode": model["time_head_mode"], "time_scale": model["time_scale"],
            "time_initial_location": model["time_initial_location"], "time_initial_scale": model["time_initial_scale"],
            "statistics_source_split": "train", "train_time_statistics": stats,
            "observation_likelihood": model["time_observation_contract"], "reported_metric": TIME_METRIC}}


def initialization(data):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states, rng = {}, None
    with torch.random.fork_rng(devices=[]):
        for arm in (*ARMS, proposal.UNGATED):
            torch.manual_seed(42)
            model, _ = build_model(data, arm)
            states[arm] = model.state_dict()
            if rng is None: rng = torch.get_rng_state().clone()
            require(torch.equal(rng, torch.get_rng_state()), "Initial caller RNG differs")
        require(all(torch.equal(value, states[arm][name]) for arm in (*ARMS[1:], proposal.UNGATED)
                    for name, value in states[ARMS[0]].items()), "Shared initial state differs")
        require(all(torch.equal(value, states[ARMS[2]][name]) for name,value in states[ARMS[1]].items()
                    if name != "local_gate.contract_identity"), "Gate control/candidate common tensors differ")
        return {arm: canonical_state_dict_sha256(state) for arm,state in states.items()}


def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None, expected_identity=None, expected_initial=None, expected_epoch=None):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    payload = torch_load_checkpoint(Path(path), map_location="cpu")
    if expected_arm is not None:
        from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec
        selector = checkpoint_monitor_spec("validation_raw_quantity_rmse")
        require(payload.get("backbone") == expected_arm and payload.get("seed") == 42
                and payload.get("variant") == VARIANT, "Replay arm/seed/objective mismatch")
        require(payload.get("resume_identity") == expected_identity
                and payload.get("interface_meta") == expected_identity["interface_meta"]
                and payload.get("initial_state_sha256") == expected_initial,
                "Replay source/initialization/interface identity mismatch")
        revision = expected_identity["arguments"]["source_revision"] if "source_revision" in expected_identity.get("arguments", {}) else None
        if revision is not None:
            require(payload.get("source_revision") == revision and payload.get("source_revision_history") == [revision], "Replay source revision mismatch")
        require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
                and payload.get("checkpoint_monitor_history_key") == selector["history_key"]
                and payload.get("checkpoint_selection") == selector["selection"], "Replay selector mismatch")
        require(payload.get("epoch", payload.get("best_epoch")) == expected_epoch, "Replay epoch mismatch")
    head = payload.get("interface", {}).get("time_head", {})
    # The shared trainer writes its interface under interface_meta.
    if not head:
        head = payload.get("interface_meta", {}).get("time_head", {})
    require(head.get("observation_likelihood") == data["model"]["time_observation_contract"],
            "Replay observation likelihood metadata mismatch")
    result = _replay_validation(payload, path, data, frame, budget, device=device)
    result["time_metric"] = TIME_METRIC
    return result


def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    """One streaming validation pass with fixed train-derived strata; no records."""
    import numpy as np
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(payload.get("backbone") in ARMS, "Unexpected endpoint backbone")
    validate_checkpoint_route(payload, payload["backbone"])
    require(payload["evaluation_scope"] == "validation_only"
            and payload["held_out_test_evaluated"] is False, "Replay split changed")
    state_digest = canonical_state_dict_sha256(payload["model_state_dict"])
    require(payload.get("model_state_sha256") == state_digest, "Checkpoint state digest differs")
    model, _ = build_model(data, payload["backbone"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = make_loader(frame, target_split="validation", **{
        key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")},
        shuffle=False, generator=None)
    q_bounds, h_bounds = data["quantity_boundaries_all_train_rows"], data["history_boundaries"]
    partitions = [("quantity", q_bounds), ("history", h_bounds)]
    if "additional_history_boundaries" in data:
        partitions.append(("additional_history", data["additional_history_boundaries"]))
    accumulator = {"overall": [0, 0., 0., 0.], "body": [0, 0., 0., 0.], "tail": [0, 0., 0., 0.]}
    for kind, bounds in partitions:
        accumulator.update({f"{kind}_{i}": [0, 0., 0., 0.] for i in range(len(bounds) + 1)})
    with torch.no_grad():
        for _, dts, mask, _, quantities in loader:
            budget_check()
            result = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
            q = result["true_qty"].cpu().numpy().astype(np.float64)
            pred = result["pred_qty"].cpu().numpy().astype(np.float64)
            t = result["time_loss"].cpu().numpy().astype(np.float64)
            h = result["history_length"].cpu().numpy()
            require(np.isfinite(q).all() and np.isfinite(pred).all() and np.isfinite(t).all(),
                    "Nonfinite checkpoint replay")
            masks = {"overall": np.ones(q.shape, dtype=bool), "body": q <= q_bounds[2], "tail": q > q_bounds[3]}
            for kind, bounds in partitions:
                values = q if kind == "quantity" else h
                ids = np.searchsorted(bounds, values, side="left")
                masks.update({f"{kind}_{i}": ids == i for i in range(len(bounds) + 1)})
            error = pred - q
            for key, selected in masks.items():
                acc = accumulator[key]
                acc[0] += int(selected.sum())
                acc[1] += float(np.abs(error[selected]).sum())
                acc[2] += float(np.square(error[selected]).sum())
                acc[3] += float(t[selected].sum())
    budget_check()
    def finish(acc):
        n, absolute, squared, temporal = acc
        return {"count": n, "qty_mae": absolute / n if n else None, "qty_sse": squared,
            "qty_rmse": math.sqrt(squared / n) if n else None,
            "time_nll": temporal / n if n else None}
    result = {**finish(accumulator["overall"]), "body": finish(accumulator["body"]),
        "tail": finish(accumulator["tail"]), "quantity_boundaries": q_bounds,
        "history_boundaries": h_bounds, "quantity_cells": [], "history_cells": [],
        "checkpoint_path": str(path), "state_sha256": state_digest,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    for kind, bounds in partitions:
        result[kind + "_boundaries"] = bounds
        result[kind + "_cells"] = [{"bin": i, **finish(accumulator[f"{kind}_{i}"])}
                                    for i in range(len(bounds) + 1)]
    require(result["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
            "Replay validation target count changed")
    audit_replay_accounting(result)
    return result


def _finite(value, *, nonnegative=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def audit_replay_accounting(result):
    """Disjoint quantity/history partitions each reconstruct the same population."""
    require(type(result.get("count")) is int and result["count"] > 0, "Missing/nonpositive overall count")
    def cell(value):
        require(isinstance(value, dict) and type(value.get("count")) is int and value["count"] >= 0,
                "Missing/invalid cell count")
        require(_finite(value.get("qty_sse"), nonnegative=True), "Missing/nonfinite SSE")
        if value["count"] == 0:
            require(value["qty_sse"] == 0 and value.get("qty_rmse") is None
                    and value.get("qty_mae") is None and value.get("time_nll") is None,
                    "Empty cells must be explicit unavailable")
        else:
            require(_finite(value.get("qty_rmse"), nonnegative=True)
                    and _finite(value.get("qty_mae"), nonnegative=True)
                    and _finite(value.get("time_nll")), "Nonfinite/nonphysical cell metric")
            require(math.isclose(value["qty_rmse"]**2 * value["count"], value["qty_sse"], rel_tol=1e-10, abs_tol=1e-8),
                    "Cell SSE and RMSE disagree")
    cell(result); cell(result["body"]); cell(result["tail"])
    partitions = [("quantity", 5), ("history", 3)]
    if "additional_history_boundaries" in result:
        partitions.append(("additional_history", len(result["additional_history_boundaries"]) + 1))
    for kind, n in partitions:
        cells = result.get(kind + "_cells")
        bounds = result.get(kind + "_boundaries")
        require(isinstance(bounds, list) and len(bounds) == n-1
                and all(_finite(b, nonnegative=True) for b in bounds)
                and bounds == sorted(bounds), "Missing/invalid frozen boundaries")
        require(isinstance(cells, list) and len(cells) == n
                and [v.get("bin") for v in cells] == list(range(n)), "Missing/duplicated disjoint bins")
        for value in cells: cell(value)
        require(sum(v["count"] for v in cells) == result["count"], "Disjoint bin counts do not sum")
        require(math.isclose(sum(v["qty_sse"] for v in cells), result["qty_sse"], rel_tol=1e-10, abs_tol=1e-8),
                "Disjoint SSE does not sum")
        require(math.isclose(sum(v["qty_mae"]*v["count"] for v in cells if v["count"]),
                             result["qty_mae"]*result["count"], rel_tol=1e-10, abs_tol=1e-8), "Disjoint absolute errors do not sum")
    bins = result["quantity_cells"]
    require(result["tail"] == {k: v for k, v in bins[4].items() if k != "bin"}, "Tail differs from its disjoint bin")
    body = result["body"]
    require(body["count"] == sum(v["count"] for v in bins[:3])
            and math.isclose(body["qty_sse"], sum(v["qty_sse"] for v in bins[:3]), rel_tol=1e-10, abs_tol=1e-8),
            "Body accounting differs")


def compare_arms(arms, acceptance=None):
    """Prospective gates use direct inequalities, including zero references.

    Empty, absent or nonfinite required cells make the scientific decision
    unverified; they never count as a pass or a training-process failure.
    """
    gates = ACCEPTANCE if acceptance is None else acceptance
    require(gates == ACCEPTANCE, "Prospective multi-lag acceptance gates changed")
    results = {}
    for reference in (*ARMS[:2], proposal.UNGATED):
        checks, errors = {}, []
        try:
            base, candidate = arms[reference], arms[ARMS[2]]
            b, c = base["selected"], candidate["selected"]
            audit_replay_accounting(b); audit_replay_accounting(c)
            require(b["count"] == c["count"] and b["quantity_boundaries"] == c["quantity_boundaries"]
                    and b["history_boundaries"] == c["history_boundaries"], "Comparison populations differ")
            require(all(b["quantity_cells"][i]["count"] == c["quantity_cells"][i]["count"] for i in range(5)),
                    "Comparison quantity-bin counts differ")
            def check(name, left, right, *, factor=1., additive=0., strict=False, counts=None):
                if (counts is not None and (counts[0] <= 0 or counts[0] != counts[1])) or not (_finite(left) and _finite(right)):
                    checks[name] = None
                    errors.append(name + ": required measurement unavailable")
                    return
                threshold = right * factor + additive
                checks[name] = left < threshold if strict else left <= threshold
            check("raw_rmse_improves", c["qty_rmse"], b["qty_rmse"], strict=True)
            check("raw_mae_guard", c["qty_mae"], b["qty_mae"], factor=gates["overall_mae_ratio_max"])
            for region in ("body", "tail"):
                cc, bb = c[region], b[region]
                check(region+"_mae_guard", cc["qty_mae"], bb["qty_mae"],
                      factor=gates[region+"_mae_ratio_max"], counts=(cc["count"], bb["count"]))
            check("tail_rmse_improves", c["tail"]["qty_rmse"], b["tail"]["qty_rmse"], strict=True,
                  counts=(c["tail"]["count"], b["tail"]["count"]))
            for i in (1, 2, 3):
                cc, bb = c["quantity_cells"][i], b["quantity_cells"][i]
                for metric in ("qty_mae", "qty_rmse"):
                    check(f"middle_bin_{i}_{metric}_guard", cc[metric], bb[metric],
                          factor=gates["each_nonempty_middle_bin_mae_and_rmse_ratio_max"], counts=(cc["count"], bb["count"]))
            check("recorded_time_nll_guard", c["time_nll"], b["time_nll"], additive=gates["recorded_time_nll_increase_max"])
            for metric, name in (("mean", "last30_rmse_mean_ratio_max"), ("sd", "last30_rmse_sample_sd_ratio_max")):
                require(base["last30"].get("count") == candidate["last30"].get("count") == 30
                        and base["last30"].get("standard_deviation") == candidate["last30"].get("standard_deviation") == "sample_ddof_1"
                        and _finite(base["last30"].get(metric), nonnegative=True)
                        and _finite(candidate["last30"].get(metric), nonnegative=True), "Missing/invalid last30 statistics")
                check("last30_rmse_"+metric+"_guard", candidate["last30"][metric], base["last30"][metric], factor=gates[name])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(str(error))
        verified = not errors and len(checks) == 14 and all(v is not None for v in checks.values())
        accepted = verified and all(checks.values())
        results[reference] = {"status": "pass" if accepted else "fail" if verified else "unverified",
            "accepted": accepted, "verified": verified, "checks": checks, "unverified_reasons": errors,
            "scope": "single_seed_same_host_source_bridged_validation_no_universal_claim", "time_metric": TIME_METRIC}
    verified = all(v["verified"] for v in results.values())
    accepted = verified and all(v["accepted"] for v in results.values())
    goal = False
    if accepted:
        b, c = arms[ARMS[0]]["selected"], arms[ARMS[2]]["selected"]
        goal = (all(c["quantity_cells"][i][metric] <= b["quantity_cells"][i][metric]
                    for i in (1,2,3) for metric in ("qty_mae", "qty_rmse"))
                and arms[ARMS[2]]["last30"]["mean"] <= arms[ARMS[0]]["last30"]["mean"]
                and arms[ARMS[2]]["last30"]["sd"] < arms[ARMS[0]]["last30"]["sd"])
    return {"status": "pass" if accepted else "fail" if verified else "unverified",
            "accepted": accepted, "verified": verified, "goal_fully_met": goal, "against": results}


def prepare_admitted_data(data, *, data_root=None):
    """Verify admitted inputs using the frozen statistics, without a model.

    The quantity preparation helper predates the execution contract's nested
    ``statistics`` record. Supply its legacy mean name on a temporary mapping;
    neither the frozen contract nor the initialization values are refitted.
    This is also the CPU-only preflight path used before production training.
    """
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    frozen_mean = data["statistics"]["train_log_mean"]
    require(_finite(frozen_mean) and frozen_mean > 0, "Invalid frozen train log mean")
    if "mu_all_train_rows" in data:
        require(_finite(data["mu_all_train_rows"])
                and math.isclose(data["mu_all_train_rows"], frozen_mean, rel_tol=0, abs_tol=1e-12),
                "Legacy train log mean conflicts with frozen statistics")
    adapted = {**data, "mu_all_train_rows": frozen_mean}
    frame, metadata = prepare_quantity_comparison_data(adapted, data_root=data_root)
    metadata = bind_frozen_statistics(data, metadata)
    require(metadata["held_out_materialized"] is False
            and metadata["populations"] == data["inherited_data_identity"]["populations"],
            "Admitted input population changed")
    return frame, metadata


def verify_start_memory(contract, host):
    spec=contract['hosts'][host]
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    values={parts[0].strip():int(parts[1].strip())*1024**2 for line in rows.splitlines() if (parts:=line.split(','))}
    require(values.get(spec['gpu_uuid'],0)>=contract['policy']['startup_min_free_gpu_bytes'], 'Insufficient free GPU memory; other jobs preserved')


def verify_dependency(contract, host):
    dep=contract.get('dependency',{}).get(host)
    if dep is None:return
    root=Path(dep['root'])
    require(common.sha_json(read(root/'recovery_contract.json'))==dep['contract_sha256'], 'Dependency contract changed')
    require(read(root/'run/status.json').get('status')=='complete', 'Previous 5090 run not complete; preserve and wait')
    terminal=read(root/'run/partition_summary.json')
    require(terminal.get('status')=='complete' and terminal.get('held_out_test_evaluated') is False,
            'Previous completion audit missing')
    # Endpoint counts, continuous 120 epochs and exposure are audited by the old
    # supervisor; also independently check its complete three-arm comparison.
    paired=read(root/'run/insta_market_basket/paired_comparison.json')
    require(paired.get('status')=='complete' and paired.get('exposure_equal') is True
            and set(paired['arms'])=={'titantpp','titantpp_local_detail','titantpp_multilag_detail'}, 'Previous paired audit incomplete')
    data=next(d for d in contract['datasets'] if d['dataset_id']=='insta_market_basket')
    for arm,result in paired['arms'].items():
        for label in ('selected','last'):audit_replay_accounting(result[label])
        run=Path(result['last']['checkpoint_path']).parent
        summary=read(run/'summary.json');history=read(run/'history.json')['history']
        require(summary.get('completed_epochs')==120 and summary.get('status')=='success'
                and [r['epoch'] for r in history]==list(range(1,121)), 'Previous arm unfinished')
        require(shared.audit_exposure(read(run/'exposure.json'),data)==result['global_steps'], 'Previous exposure incomplete')
    old_cmd=Path('/proc')/str(dep['own_worker_pid'])/'cmdline'
    require(not old_cmd.exists() or str(root).encode() not in old_cmd.read_bytes(), 'Previous owned worker still active')
    sessions=subprocess.run([contract['hosts'][host]['tmux_binary'],'has-session','-t',dep['tmux']],capture_output=True)
    require(sessions.returncode!=0, 'Previous owned tmux still active')


def verify_reference_bridge(contract, host):
    root=Path(contract['hosts'][host]['root']);r=read(root/'bridge_receipt.json')
    require(r.get('status')=='passed' and r.get('host')==host and r.get('contract_sha256')==common.sha_json(contract)
            and r.get('source_files_sha256')==contract['source']['files_sha256'], 'Missing source/reference bridge')
    old,new=read(root/'bridge_old.json'),read(root/'bridge_new.json')
    require(r.get('old_sha256')==common.sha_file(root/'bridge_old.json') and r.get('new_sha256')==common.sha_file(root/'bridge_new.json')
            and old==new and set(old)=={d['dataset_id'] for d in contract['datasets']}, 'Source bridge mismatch')
    old_root=Path(proposal.parent()['hosts'][host]['source_root'])
    require(all(common.sha_file(old_root/name)==digest for name,digest in proposal.parent()['source']['files'].items()),
            'Original reference source changed')
    for row in new.values():
        require(set(row)=={'titantpp',proposal.UNGATED} and all(v['optimizer_updates']==3 for v in row.values()), 'Bridge workload changed')
    return r


def load_references(contract, host, data, interface, initial, output):
    from paper.scripts.count_aware_tpp_backbone import training
    arms,exposures={},{}
    q={'boundaries':data['quantity_boundaries_all_train_rows'], 'strata':[{'label':f'frozen_quantity_bin_{i}'} for i in range(5)]}
    for arm,spec in contract['references'][host][data['dataset_id']].items():
        require(arm in ('titantpp',proposal.UNGATED),'Unexpected reused reference')
        path=Path(spec['path'])
        require(all(common.sha_file(path/name)==digest for name,digest in spec['files'].items()), 'Pinned reference changed')
        summary=read(path/'summary.json');history=read(path/'history.json')['history'];endpoints=read(path/'endpoint_replays.json')
        require(summary.get('status')=='success' and summary.get('completed_epochs')==120 and not summary.get('stopped_early')
                and summary.get('evaluation_scope')=='validation_only' and summary.get('held_out_test_evaluated') is False
                and summary.get('seed')==42 and summary.get('backbone')==arm and summary.get('variant')==VARIANT,
                'Reference training scope incomplete')
        require(summary['initial_state_sha256']==spec['initial_state_sha256']==initial[arm]==endpoints['initial_state_sha256'], 'Reference initialization mismatch')
        require([r['epoch'] for r in history]==list(range(1,121)) and all(r['train_all_finite'] for r in history),'Reference history incomplete')
        earliest=training.earliest_strict_minimum(history,metric_key='val_qty_rmse')
        require(summary['best_epoch']==endpoints['best_epoch']==earliest['epoch'],'Reference selector changed')
        require(summary['interface_meta']['execution_contract_sha256']==contract['parent_contract_sha256']
                and summary['interface_meta']['source_files_sha256']==proposal.parent()['source']['files_sha256'], 'Reference scientific lineage changed')
        for key in ('train_target_mean','train_target_std','statistics','data_scope','time_head'):
            require(summary['interface_meta'][key]==interface[key], 'Reference interface differs: '+key)
        args=training_args(contract,data,output)
        expected=training._resume_identity(args=args,backbone=arm,quantity_variant=VARIANT,seed=42,
            monitor='validation_raw_quantity_rmse',interface_meta=interface,quantity_contract=q)
        actual=summary['resume_identity']
        require(actual['quantity_contract']==expected['quantity_contract'] and actual['checkpoint_monitor']==expected['checkpoint_monitor'], 'Reference target/selector differs')
        for key,val in expected['arguments'].items():
            if key not in ('model_role','execution_role'):
                require(actual['arguments'].get(key)==val,'Reference training argument differs: '+key)
        exposures[arm]=read(path/'exposure.json');steps=shared.audit_exposure(exposures[arm],data)
        require(steps==endpoints['global_steps'] and endpoints['last30']==shared.last30_summary(history), 'Reference exposure/stability mismatch')
        for label,index in (('selected',summary['best_epoch']-1),('last',119)):
            value=endpoints[label];audit_replay_accounting(value)
            require(value.get('evaluation_scope')=='validation_only' and value.get('held_out_test_evaluated') is False
                and value['count']==data['inherited_data_identity']['populations']['validation']['target_count']
                and value['quantity_boundaries']==data['quantity_boundaries_all_train_rows']
                and value['history_boundaries']==data['history_boundaries'], 'Reference endpoint scope differs')
            require(all(math.isclose(value[m],history[index][k],rel_tol=1e-10,abs_tol=1e-8)
                for m,k in (('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll'))),'Reference replay/history differs')
        require(endpoints['selected']['state_sha256']==summary['checkpoint_state_sha256'], 'Reference selected state differs')
        arms[arm]=endpoints
    require(set(arms)=={'titantpp',proposal.UNGATED} and exposures[ARMS[0]]==exposures[proposal.UNGATED], 'Reference pair exposure differs')
    return arms,exposures


def production(contract, permit, host, qualification, budget):
    verify_process_environment(contract, host, active=True)
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    verify_reference_bridge(contract, host)
    runtime = common.runtime_check(contract, host)
    require(runtime == qualification["runtime"], "Runtime changed since qualification")
    output = Path(contract["hosts"][host]["output_dir"])
    comparisons = {}
    for dataset_id in contract["hosts"][host]["assigned_datasets"]:
        budget()
        data = next(d for d in contract["datasets"] if d["dataset_id"] == dataset_id)
        frame, metadata = prepare_admitted_data(data)
        interface, initial = time_interface(data, frame, contract), initialization(data)
        dest = output / dataset_id
        dest.mkdir(exist_ok=False)
        common.write_json(dest / "input_receipt.json", metadata, exclusive=True)
        common.write_json(dest / "initialization.json", initial, exclusive=True)
        q = {"boundaries": data["quantity_boundaries_all_train_rows"],
             "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(5)]}
        arms, exposures = load_references(contract, host, data, interface, initial, dest)
        common.write_json(dest / "reference_audit.json", {"status":"passed", "arms":list(arms),
            "source_bridge_sha256":common.sha_file(Path(contract["hosts"][host]["root"])/"bridge_receipt.json")}, exclusive=True)
        for arm in ARMS[1:]:
            budget(); proposal.validate_contract(contract)
            require(common.runtime_check(contract, host) == runtime, "Runtime drift before arm")
            arm_budget = ArmBudget(contract, host, dataset_id, arm, budget)
            args = training_args(contract, data, dest)
            run = dest / "runs" / arm / VARIANT / "seed_42"
            require(not run.exists(), "Fresh arm required; resume/retry forbidden")
            def status(epoch, records):
                common.write_json(output / "status.json", {"status": "training", "host": host,
                    "dataset": dataset_id, "backbone": arm, "epoch": epoch,
                    "global_steps": sum(r["batches"] for r in records["train"]),
                    "deadline_unix": permit["deadline_unix"]})
                common.write_json(run / "exposure.json", records)
            with shared.audited_training(training, data, arm_budget, status) as exposure:
                summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=q,
                    interface_meta=interface, backbone=arm, quantity_variant=VARIANT, seed=42)
            require(summary["status"] == "success" and summary["completed_epochs"] == 120
                    and not summary["stopped_early"] and summary["initial_state_sha256"] == initial[arm],
                    "Incomplete arm or initial state mismatch")
            steps = shared.audit_exposure(exposure, data)
            exposures[arm] = exposure
            common.write_json(run / "exposure.json", exposure)
            history = read(run / "history.json")["history"]
            require([x["epoch"] for x in history] == list(range(1, 121)), "Noncontinuous history")
            earliest = training.earliest_strict_minimum(history, metric_key="val_qty_rmse")
            require(earliest is not None and summary["best_epoch"] == earliest["epoch"], "Selected checkpoint is not the earliest finite RMSE minimum")
            expected_identity = training._resume_identity(args=args, backbone=arm, quantity_variant=VARIANT, seed=42, monitor="validation_raw_quantity_rmse", interface_meta=interface, quantity_contract=q)
            replay = {}
            for label, path in (("selected", Path(summary["checkpoint_path"])), ("last", run / "last_epoch_state.pt")):
                replay[label] = replay_checkpoint(path, data, frame, arm_budget, expected_arm=arm,
                    expected_identity=expected_identity, expected_initial=initial[arm],
                    expected_epoch=summary["best_epoch"] if label == "selected" else 120)
                row = history[summary["best_epoch"]-1] if label == "selected" else history[-1]
                require(all(math.isclose(replay[label][metric], row[field], rel_tol=1e-10, abs_tol=1e-8)
                    for metric, field in (("qty_rmse", "val_qty_rmse"), ("qty_mae", "val_qty_mae"), ("time_nll", "val_time_nll"))),
                    "Endpoint replay differs from history")
            arms[arm] = {**replay, "last30": shared.last30_summary(history), "global_steps": steps,
                "best_epoch": summary["best_epoch"], "initial_state_sha256": initial[arm]}
            common.write_json(run / "endpoint_replays.json", arms[arm], exclusive=True)
            gc.collect(); torch.cuda.empty_cache()
            arm_budget.finish()
        require(all(x == exposures[ARMS[0]] for x in exposures.values()), "Actual batch exposure differs")
        result = {"status": "complete", "dataset": dataset_id, "host": host, "arms": arms,
            "exposure_equal": True, "acceptance": compare_arms(arms, contract["acceptance"]), "time_metric": TIME_METRIC,
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
        common.write_json(dest / "paired_comparison.json", result, exclusive=True)
        comparisons[dataset_id] = result
    return {"status": "complete", "host": host, "datasets": comparisons,
        "contract_sha256": common.sha_json(contract), "deadline_unix": permit["deadline_unix"],
        "total_optimizer_steps": sum(a["global_steps"] for d in comparisons.values() for name,a in d["arms"].items() if name in ARMS[1:]),
        "reused_reference_arms": 2 * len(comparisons),
        "endpoint_replays": 2 * len(ARMS[1:]) * len(comparisons), "time_metric": TIME_METRIC,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}


def make_start_permit(contract, approval, *, started_at_unix=None):
    started = (contract.get("launch", {}).get("fixed_started_at_unix", time.time())
               if started_at_unix is None else started_at_unix)
    permit = {"schema": "local_gate_start_v1", "contract_sha256": common.sha_json(contract),
        "approval_sha256": common.sha_json(approval), "started_at_unix": started,
        "deadline_unix": started + contract["limits"]["total_wall_seconds"]}
    verify_authorization(contract, approval, permit, next(iter(contract["hosts"])), now=started)
    return permit


def make_training_permit(contract, approval, start_permit, receipts):
    verify_authorization(contract, approval, start_permit, next(iter(contract["hosts"])))
    permit = {**start_permit, "qualifications": {host: {"receipt": receipt,
        "receipt_sha256": common.sha_json(receipt)} for host, receipt in receipts.items()}}
    require(receipts, "A host qualification receipt is required")
    for host in receipts: verify_authorization(contract, approval, permit, host, training=True)
    return permit


def native_qualification(contract, approval, permit, host, budget):
    """Run only the frozen synthetic native validation and measured cost workload."""
    verify_authorization(contract, approval, permit, host)
    source_and_host(contract, host)
    verify_process_environment(contract, host, active=True)
    runtime = common.runtime_check(contract, host)
    other_before = sorted(common.gpu_pids(contract["hosts"][host]) - {os.getpid()})
    from paper.scripts import verify_local_gate_cost as cost
    require(tuple(cost.QUALIFICATION_CHECKS) == QUALIFICATION_CHECKS, "Qualification checklist changed")
    result = cost.run(device="cuda:0", batch_size=128, lengths=(64, 256), warmup_steps=2,
        measured_steps=5, max_wall_seconds=contract["limits"]["qualification_seconds_per_host"],
        budget=budget, cost_gates=contract["cost_gates"])
    require(result["synthetic_optimizer_updates"] == 48, "Synthetic update budget changed")
    proposal.validate_contract(contract); budget()
    receipt = {**result, "host": host, "device": "cuda:0", "runtime": runtime,
        "contract_sha256": common.sha_json(contract), "approval_sha256": common.sha_json(approval),
        "source_files_sha256": contract["source"]["files_sha256"],
        "started_at_unix": permit["started_at_unix"], "deadline_unix": permit["deadline_unix"],
        "completed_at_unix": time.time(), "real_data_loaded": False, "held_out_evaluated": False,
        "other_gpu_pids_before": other_before,
        "other_gpu_pids_after": sorted(common.gpu_pids(contract["hosts"][host]) - {os.getpid()}),
        "timing_scope": "same host native screening; sharing allowed, not isolated throughput evidence",
        "process_environment": contract["process_environment"][host],
        "library_sha256": contract["library_sha256"][host]}
    if receipt["status"] == "passed":
        validate_qualification_measurements(contract, receipt)
    return receipt


def supervisor(command, contract_path, approval_path, permit_path, host):
    contract, approval, permit = read(contract_path), read(approval_path), read(permit_path)
    spec = verify_authorization(contract, approval, permit, host, training=command == "train")
    source_and_host(contract, host)
    if command == "train": verify_local_qualification(contract, permit, host)
    verify_dependency(contract, host)
    verify_reference_bridge(contract, host)
    preflight=read(Path(spec['root'])/'preflight_receipt.json')
    require(preflight.get('status')=='passed' and preflight.get('host')==host
        and preflight.get('contract_sha256')==common.sha_json(contract)
        and preflight.get('source_files_sha256')==contract['source']['files_sha256']
        and set(preflight.get('datasets',{}))==set(spec['assigned_datasets']), 'CPU reference preflight missing')
    verify_start_memory(contract, host)
    output = Path(spec["root"]) / ("qualification" if command == "qualify" else "run")
    require(not output.exists(), "Fresh output required; no retry/resume")
    output.mkdir(parents=True, exist_ok=False)
    apply_process_environment(contract, host)
    deadline = shared.fixed_start(permit, wall=time.time, monotonic=time.monotonic, total_seconds=contract["limits"]["total_wall_seconds"])
    if command == "qualify": deadline = min(deadline, time.monotonic()+contract["limits"]["qualification_seconds_per_host"])
    authority = {"contract_sha256": common.sha_json(contract), "approval_sha256": common.sha_json(approval),
        "permit_sha256": common.sha_json(permit), "host": host, "owner_pid": os.getpid(), "command": command}
    common.write_json(output / "wrapper_manifest.json", {"authority": authority,
        "started_at_unix": time.time(), "deadline_unix": permit["deadline_unix"], "resume": False, "retry": False}, exclusive=True)
    read_fd, write_fd = os.pipe(); child = None
    old_signal = signal.signal(signal.SIGTERM, shared._supervisor_terminated)
    try:
        with (output / "worker.log").open("xb") as stream:
            child = subprocess.Popen([spec["python"], str(Path(__file__).resolve()), "_worker",
                "--stage", command, "--contract", str(Path(contract_path).resolve()),
                "--approval", str(Path(approval_path).resolve()), "--permit", str(Path(permit_path).resolve()),
                "--host", host, "--owner-pid", str(os.getpid()), "--authority-fd", str(read_fd)],
                cwd=spec["source_root"], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(read_fd,))
            os.close(read_fd); read_fd = None
            with os.fdopen(write_fd, "wb") as pipe:
                write_fd = None; pipe.write(json.dumps(authority).encode())
            while child.poll() is None:
                require(time.time() < permit["deadline_unix"] and time.monotonic() < deadline, "Execution deadline reached")
                if command == "train": verify_active_arm_deadline(contract, host)
                shared.check_storage(output, storage_limits(contract)); time.sleep(1.)
            require(child.returncode == 0, f"Owned worker failed: {child.returncode}")
            require(time.time() < permit["deadline_unix"] and time.monotonic() < deadline, "Deadline reached before terminal audit")
            result = read(output / ("receipt.json" if command == "qualify" else "partition_summary.json"))
            require(result["status"] == ("passed" if command == "qualify" else "complete")
                    and result["contract_sha256"] == common.sha_json(contract), "Invalid terminal receipt")
            shared.check_storage(output, storage_limits(contract)); return result
    except BaseException as error:
        if child is not None: shared.kill_owned_process_group(child)
        common.write_json(output / "status.json", {"status": "stopped", "error": str(error), "automatic_retry": False})
        raise
    finally:
        signal.signal(signal.SIGTERM, old_signal)
        for fd in (read_fd, write_fd):
            if fd is not None: os.close(fd)


def worker(args):
    contract, approval, permit = read(args.contract), read(args.approval), read(args.permit)
    spec = verify_authorization(contract, approval, permit, args.host, training=args.stage == "train")
    source_and_host(contract, args.host)
    require(args.owner_pid == os.getppid() and os.getsid(0) == os.getpid(), "Owned process-group worker required")
    with os.fdopen(args.authority_fd, "rb") as pipe: authority = json.loads(pipe.read(4096))
    expected = {"contract_sha256": common.sha_json(contract), "approval_sha256": common.sha_json(approval),
        "permit_sha256": common.sha_json(permit), "host": args.host, "owner_pid": args.owner_pid, "command": args.stage}
    require(authority == expected, "Inherited authority mismatch")
    output = Path(spec["root"]) / ("qualification" if args.stage == "qualify" else "run")
    require(read(output / "wrapper_manifest.json")["authority"] == expected, "Output ownership mismatch")
    verify_dependency(contract, args.host)
    verify_reference_bridge(contract, args.host)
    verify_start_memory(contract, args.host)
    apply_process_environment(contract, args.host)
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2,) * 2)
    budget = Budget(contract, permit, args.host, args.owner_pid, output); budget()
    if args.stage == "qualify":
        result = native_qualification(contract, approval, permit, args.host, budget)
        common.write_json(output / "receipt.json", result, exclusive=True)
        require(result["status"] == "passed", "CUDA cost qualification failed")
    else:
        qualification = verify_local_qualification(contract, permit, args.host)
        result = production(contract, permit, args.host, qualification, budget)
        budget(); common.write_json(output / "partition_summary.json", result, exclusive=True)
    common.write_json(output / "status.json", {"status": result["status"], "host": args.host,
        "deadline_unix": permit["deadline_unix"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "train", "_worker"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--host", choices=tuple(proposal.ASSIGNMENTS), required=True)
    parser.add_argument("--stage", choices=("qualify", "train"))
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--authority-fd", type=int)
    args = parser.parse_args()
    if args.command == "_worker":
        require(args.stage is not None and args.owner_pid is not None and args.authority_fd is not None, "Worker authority required")
        worker(args)
    else:
        require(args.stage is None and args.owner_pid is None and args.authority_fd is None, "Unexpected internal authority")
        supervisor(args.command, args.contract, args.approval, args.permit, args.host)


if __name__ == "__main__":
    main()
