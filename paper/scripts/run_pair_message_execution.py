#!/usr/bin/env python3
"""Approval-bound pair-message CUDA qualification and fresh validation training.

No model/data/CUDA import occurs before approval, source, host and permit checks.
The same common wall-clock origin covers both qualifications, training and replay.
"""
from __future__ import annotations

import argparse
import gc
import io
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
from paper.scripts import prepare_pair_message_execution as proposal
from paper.scripts import run_observed_slot_partition as shared

require = common.require
ARMS = proposal.ARMS
VARIANT = "count_only_log_regression"
TIME_METRIC = "recorded_positive_integer_time_nll"


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
    require(permit.get("schema") == "pair_message_start_v1"
            and permit.get("contract_sha256") == digest
            and permit.get("approval_sha256") == common.sha_json(approval), "Permit authority mismatch")
    started, deadline = permit.get("started_at_unix"), permit.get("deadline_unix")
    now = time.time() if now is None else now
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started <= now < deadline
            and deadline - started == contract["limits"]["total_wall_seconds"],
            "Expired, future or extended common deadline")
    if training:
        qualifications = permit.get("qualifications", {})
        require(set(qualifications) == set(proposal.ASSIGNMENTS), "Both native qualifications required")
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
            validate_qualification_measurements(contract, receipt)
    return contract["hosts"][host]


def validate_qualification_measurements(contract, receipt):
    expected = {"pmf_finite_tail_gradients", "initial_output_gradient_rng_match", "causality", "added_gradients", "optimizer_rng_replay"}
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
        require(measurements[ARMS[1]]["parameters"] == measurements[ARMS[2]]["parameters"], "Control/candidate capacity differs")
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
            require(common.gpu_pids(self.spec) <= {os.getpid()}, "Other GPU job appeared; stopping only owned worker")
            self.last_scan = time.monotonic()


def build_model(data, backbone):
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {k: v for k, v in data["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(backbone, **config,
        train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def training_args(contract, data, output):
    from paper.scripts.run_count_aware_tpp_backbone_control import parse_args
    # Obtain maintained defaults without entering the CLI's data execution path.
    old = sys.argv
    try:
        sys.argv = ["pair_message", "--data", data["inherited_data_identity"]["data"]["path"],
            "--split-manifest", data["inherited_data_identity"]["split_manifest"]["path"],
            "--output-dir", str(output), "--source-revision", contract["source"]["base_git_revision"],
            "--execution-role", "fresh_pair_message_observed_time_validation"]
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
        execution_role="fresh_pair_message_observed_time_validation",
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
        "time_head": {"mode": model["time_head_mode"], "time_scale": model["time_scale"],
            "time_initial_location": model["time_initial_location"], "time_initial_scale": model["time_initial_scale"],
            "statistics_source_split": "train", "train_time_statistics": stats,
            "observation_likelihood": model["time_observation_contract"], "reported_metric": TIME_METRIC}}


def initialization(data):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states, rng = {}, None
    with torch.random.fork_rng(devices=[]):
        for arm in ARMS:
            torch.manual_seed(42)
            model, _ = build_model(data, arm)
            states[arm] = model.state_dict()
            if rng is None:
                rng = torch.get_rng_state().clone()
            require(torch.equal(rng, torch.get_rng_state()), "Initial caller RNG differs")
        require(all(torch.equal(value, states[arm][name]) for arm in ARMS[1:] for name, value in states[ARMS[0]].items()),
                "Shared initial state differs")
        require(all(torch.equal(value, states[ARMS[2]][name]) for name, value in states[ARMS[1]].items() if name != "pair_message.routing_code"),
                "Candidate/control initial tensors differ")
        return {arm: canonical_state_dict_sha256(state) for arm, state in states.items()}


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
    original = shared._build_model
    try:
        shared._build_model = build_model
        result = shared.replay_checkpoint(path, data, frame, budget, device=device, allowed_backbones=ARMS)
    finally:
        shared._build_model = original
    result["time_metric"] = TIME_METRIC
    return result


def compare_arms(arms):
    result = {}
    for reference in ARMS[:2]:
        item = shared.paired_acceptance(arms[reference], arms[ARMS[2]])
        item["checks"]["recorded_time_nll_guard"] = item["checks"].pop("legacy_time_guard")
        item["time_metric"] = TIME_METRIC
        result[reference] = item
    return {"accepted": all(x["accepted"] for x in result.values()), "against": result}


def production(contract, permit, host, qualification, budget):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    runtime = common.runtime_check(contract, host)
    require(runtime == qualification["runtime"], "Runtime changed since qualification")
    output = Path(contract["hosts"][host]["output_dir"])
    comparisons = {}
    for dataset_id in contract["hosts"][host]["assigned_datasets"]:
        budget()
        data = next(d for d in contract["datasets"] if d["dataset_id"] == dataset_id)
        frame, metadata = prepare_quantity_comparison_data(data)
        metadata = bind_frozen_statistics(data, metadata)
        require(metadata["held_out_materialized"] is False and metadata["populations"] == data["inherited_data_identity"]["populations"],
                "Admitted input population changed")
        interface, initial = time_interface(data, frame, contract), initialization(data)
        dest = output / dataset_id
        dest.mkdir(exist_ok=False)
        common.write_json(dest / "input_receipt.json", metadata, exclusive=True)
        common.write_json(dest / "initialization.json", initial, exclusive=True)
        q = {"boundaries": data["quantity_boundaries_all_train_rows"],
             "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(5)]}
        arms, exposures = {}, {}
        for arm in ARMS:
            budget(); proposal.validate_contract(contract)
            require(common.runtime_check(contract, host) == runtime, "Runtime drift before arm")
            args = training_args(contract, data, dest)
            run = dest / "runs" / arm / VARIANT / "seed_42"
            require(not run.exists(), "Fresh arm required; resume/retry forbidden")
            def status(epoch, records):
                common.write_json(output / "status.json", {"status": "training", "host": host,
                    "dataset": dataset_id, "backbone": arm, "epoch": epoch,
                    "global_steps": sum(r["batches"] for r in records["train"]),
                    "deadline_unix": permit["deadline_unix"]})
                common.write_json(run / "exposure.json", records)
            with shared.audited_training(training, data, budget, status) as exposure:
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
                replay[label] = replay_checkpoint(path, data, frame, budget, expected_arm=arm,
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
        require(all(x == exposures[ARMS[0]] for x in exposures.values()), "Actual batch exposure differs")
        result = {"status": "complete", "dataset": dataset_id, "host": host, "arms": arms,
            "exposure_equal": True, "acceptance": compare_arms(arms), "time_metric": TIME_METRIC,
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
        common.write_json(dest / "paired_comparison.json", result, exclusive=True)
        comparisons[dataset_id] = result
    return {"status": "complete", "host": host, "datasets": comparisons,
        "contract_sha256": common.sha_json(contract), "deadline_unix": permit["deadline_unix"],
        "total_optimizer_steps": sum(a["global_steps"] for d in comparisons.values() for a in d["arms"].values()),
        "endpoint_replays": 2 * len(ARMS) * len(comparisons), "time_metric": TIME_METRIC,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}


def make_start_permit(contract, approval, *, started_at_unix=None):
    started = time.time() if started_at_unix is None else started_at_unix
    permit = {"schema": "pair_message_start_v1", "contract_sha256": common.sha_json(contract),
        "approval_sha256": common.sha_json(approval), "started_at_unix": started,
        "deadline_unix": started + contract["limits"]["total_wall_seconds"]}
    verify_authorization(contract, approval, permit, "5080", now=started)
    return permit


def make_training_permit(contract, approval, start_permit, receipts):
    verify_authorization(contract, approval, start_permit, "5080")
    permit = {**start_permit, "qualifications": {host: {"receipt": receipt,
        "receipt_sha256": common.sha_json(receipt)} for host, receipt in receipts.items()}}
    verify_authorization(contract, approval, permit, "5080", training=True)
    return permit


def native_qualification(contract, approval, permit, host, budget):
    """48 synthetic optimizer updates on the pinned native CUDA runtime only."""
    verify_authorization(contract, approval, permit, host)
    source_and_host(contract, host)
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from models.TPPs.positive_integer_time import positive_integer_log_mass
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    runtime = common.runtime_check(contract, host)
    require(torch.cuda.is_available(), "Native CUDA required")
    kwargs = dict(hidden_dim=64, train_log_mean=1.5, max_seq_len=256,
        time_head_mode="heteroscedastic_lognormal_duration", time_scale=3.,
        time_initial_location=.1, time_initial_scale=.7,
        time_observation_contract={"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None})
    def build(arm):
        torch.manual_seed(42)
        return build_count_aware_model(arm, **kwargs)[0]
    def batch(size, length):
        generator = torch.Generator().manual_seed(77+length)
        return tuple(t.cuda() for t in (torch.randint(1, 31, (size, length), generator=generator).float(),
            torch.ones(size, length, dtype=torch.bool), torch.randint(0, 64, (size, length), generator=generator).float()))
    def objective(model, data):
        return target_outputs(model, *data, lambda_log_qty=1.)
    updates = 0
    def step(model, optimizer, data):
        nonlocal updates
        budget(); model.train(); optimizer.zero_grad(set_to_none=True)
        loss = objective(model, data)["joint_loss"].mean()
        require(bool(torch.isfinite(loss)), "Nonfinite synthetic PMF joint objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); updates += 1
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite synthetic state")
        budget()
    def activate(model):
        with torch.no_grad():
            model.quantity_head.weight.copy_(torch.linspace(-.1, .1, 64, device="cuda").reshape(1, 64))
            if hasattr(model, "pair_message"):
                p = model.pair_message.output_projection.weight
                p.copy_(torch.linspace(-.05, .05, p.numel(), device="cuda").reshape_as(p))
    # Native PMF boundary and tail gradients are checked independently of heads.
    d = torch.tensor([1., 2., 30., 1000.], device="cuda")
    mu = torch.tensor([-80., 80., 0., 2.], device="cuda", requires_grad=True)
    sigma = torch.ones(4, device="cuda", requires_grad=True)
    logp = positive_integer_log_mass(d, mu, sigma, time_scale=3.)
    (-logp.sum()).backward()
    require(bool(torch.isfinite(logp).all()) and bool(torch.isfinite(mu.grad).all())
            and bool(torch.isfinite(sigma.grad).all()), "CUDA PMF tail gradient failed")
    tiny = batch(4, 8)
    base = build(ARMS[0]); rng = torch.get_rng_state().clone()
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    initial_sha = canonical_state_dict_sha256(base_state)
    base = base.cuda()
    torch.manual_seed(79); expected = objective(base, tiny)
    expected["joint_loss"].mean().backward()
    for arm in ARMS[1:]:
        candidate = build(arm)
        require(torch.equal(rng, torch.get_rng_state()) and all(torch.equal(v, candidate.state_dict()[k]) for k,v in base_state.items()),
                "CUDA qualification shared initialization/RNG differs")
        candidate.cuda(); torch.manual_seed(79)
        result = objective(candidate, tiny)
        require(all(torch.equal(expected[k], result[k]) for k in expected), "Initial output/objective differs from B")
        result["joint_loss"].mean().backward()
        params = dict(candidate.named_parameters())
        require(all((p.grad is None and params[n].grad is None) or
            (p.grad is not None and params[n].grad is not None and torch.equal(p.grad, params[n].grad))
            for n,p in base.named_parameters()), "Initial common preclip gradient differs")
        candidate.zero_grad(set_to_none=True); activate(candidate); candidate.eval()
        prediction = objective(candidate, tiny)["pred_qty"]
        def prediction_state(values):
            dts, mask, quantities = values
            observed = mask.clone(); observed[:, -1] = False
            history_q = quantities.clone(); history_q[:, -1] = 0
            return candidate.encode_task_states(dts, history_q, mask, memory_write_mask=observed)[0][:, -2]
        original_hidden = prediction_state(tiny)
        changed = tuple(t.clone() for t in tiny)
        changed[0][:,-1] += 10000; changed[2][:,-1] += 10000
        changed_hidden = prediction_state(changed)
        require(torch.equal(prediction, objective(candidate, changed)["pred_qty"])
                and torch.equal(original_hidden, changed_hidden)
                and torch.equal(candidate.time_location(original_hidden), candidate.time_location(changed_hidden)), "Future target leaked")
        prediction.sum().backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.count_nonzero(p.grad))
            for p in candidate.pair_message.parameters()), "Active added gradient absent/nonfinite")
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=.001, weight_decay=.01)
        step(candidate, optimizer, tiny)
        stream = io.BytesIO()
        torch.save({"model": candidate.state_dict(), "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, stream)
        step(candidate, optimizer, tiny)
        continuous = {n: v.cpu().clone() for n,v in candidate.state_dict().items()}
        import copy
        optimizer_expected = copy.deepcopy(optimizer.state_dict())
        cpu_expected, cuda_expected = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        stream.seek(0); saved = torch.load(stream, map_location="cpu", weights_only=False)
        candidate.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"]); torch.cuda.set_rng_state(saved["cuda_rng"])
        step(candidate, optimizer, tiny)
        require(all(torch.equal(continuous[n], v.cpu()) for n,v in candidate.state_dict().items()), "Saved model replay differs")
        def same(a, b):
            if isinstance(a, torch.Tensor): return torch.equal(a, b)
            if isinstance(a, dict): return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
            if isinstance(a, (list,tuple)): return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
            return a == b
        require(same(optimizer_expected, optimizer.state_dict()) and torch.equal(cpu_expected, torch.get_rng_state())
                and torch.equal(cuda_expected, torch.cuda.get_rng_state()), "Saved optimizer/RNG replay differs")
        del candidate, optimizer, stream, saved, continuous, result, prediction, changed, params
        del optimizer_expected, cpu_expected, cuda_expected, original_hidden, changed_hidden
    del base, expected, base_state, tiny, d, mu, sigma, logp
    gc.collect(); torch.cuda.empty_cache()
    costs = []
    for length in (64,256):
        data = batch(128, length); measured = {}
        for arm in ARMS:
            model = build(arm).cuda(); activate(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            samples = []; torch.cuda.reset_peak_memory_stats()
            for index in range(7):
                torch.manual_seed(110+index); torch.cuda.synchronize(); start = time.perf_counter()
                step(model, optimizer, data); torch.cuda.synchronize()
                if index >= 2: samples.append(time.perf_counter()-start)
            measured[arm] = {"median_step_seconds": statistics.median(samples), "step_seconds": samples,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "parameters": sum(p.numel() for p in model.parameters())}
            del model, optimizer; gc.collect(); torch.cuda.empty_cache()
        base_cost = measured[ARMS[0]]; gates = contract["cost_gates"]; checks = {}
        for arm in ARMS[1:]:
            cost = measured[arm]
            checks.update({arm+":step": cost["median_step_seconds"] <= base_cost["median_step_seconds"]*gates["median_cuda_step_ratio_max"],
                arm+":memory": cost["peak_allocated_bytes"] <= base_cost["peak_allocated_bytes"]*gates["peak_cuda_allocated_ratio_max"],
                arm+":device": cost["peak_allocated_bytes"] <= runtime["gpu"]["total_memory_bytes"]*gates["device_memory_fraction_max"],
                arm+":parameters": cost["parameters"] <= base_cost["parameters"]*gates["parameter_ratio_max"]})
        costs.append({"length": length, "batch_size": 128, "measurements": measured, "checks": checks})
        del data; gc.collect(); torch.cuda.empty_cache()
    require(updates == 48, "Synthetic update count changed")
    proposal.validate_contract(contract); budget()
    return {"status": "passed" if all(all(c["checks"].values()) for c in costs) else "cost_gate_failed",
        "host": host, "device": "cuda:0", "runtime": runtime,
        "contract_sha256": common.sha_json(contract), "approval_sha256": common.sha_json(approval),
        "source_files_sha256": contract["source"]["files_sha256"],
        "started_at_unix": permit["started_at_unix"], "deadline_unix": permit["deadline_unix"],
        "completed_at_unix": time.time(), "synthetic_optimizer_updates": updates,
        "shared_initial_state_sha256": initial_sha, "real_data_loaded": False, "held_out_evaluated": False,
        "checks": {"pmf_finite_tail_gradients": True, "initial_output_gradient_rng_match": True,
            "causality": True, "added_gradients": True, "optimizer_rng_replay": True}, "costs": costs}


def supervisor(command, contract_path, approval_path, permit_path, host):
    contract, approval, permit = read(contract_path), read(approval_path), read(permit_path)
    spec = verify_authorization(contract, approval, permit, host, training=command == "train")
    source_and_host(contract, host)
    if command == "train": verify_local_qualification(contract, permit, host)
    require(not common.gpu_pids(spec), "GPU busy; existing jobs preserved")
    output = Path(spec["root"]) / ("qualification" if command == "qualify" else "run")
    require(not output.exists(), "Fresh output required; no retry/resume")
    output.mkdir(parents=True, exist_ok=False)
    common.apply_environment(spec)
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
    require(not common.gpu_pids(spec), "GPU became busy; no existing job will be stopped")
    common.apply_environment(spec)
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
