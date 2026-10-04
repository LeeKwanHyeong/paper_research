#!/usr/bin/env python3
"""Fresh, owned-process, deadline-bounded observed-slot/B validation partitions.

The supervisor is standard-library only. Model/data imports occur exclusively
inside its pipe-authorized child after the common two-host qualification permit.
The ordinary shared trainer remains the training and checkpoint-selection path.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSIGNMENT = {"5090": ["insta_market_basket"],
              "5080": ["intermittent_frozen_5000", "yellow_trip_hourly"]}
BASELINE = "titantpp"
CANDIDATE = "titantpp_observed_slot_memory"
VARIANT = "count_only_log_regression"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def common_module():
    from paper.scripts import observed_slot_parallel_common
    return observed_slot_parallel_common


def read_json(path):
    return json.loads(Path(path).read_text())


def write_status(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def validate_assignment(contract, host):
    require(host in ASSIGNMENT and set(contract["hosts"]) == set(ASSIGNMENT),
            "Unknown host assignment")
    require(all(contract["hosts"][name]["assigned_datasets"] == datasets
                for name, datasets in ASSIGNMENT.items()), "Fixed dataset assignment changed")
    require(contract["seed"] == 42 and contract["epochs"] == 120,
            "Fixed seed/epoch exposure changed")
    require(contract["model"] == {"baseline": BASELINE, "candidate": CANDIDATE},
            "Fixed B/candidate pair changed")
    require(contract["limits"]["total_seconds"] == 86400,
            "Common deadline cap changed")
    return contract["hosts"][host]


def fixed_start(permit, *, wall=time.time, monotonic=time.monotonic, total_seconds=86400):
    """Return a monotonic deadline with no reset of the common wall-clock cap."""
    started, deadline, now = permit["started_at_unix"], permit["deadline_unix"], wall()
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started <= now < deadline and deadline - started == total_seconds,
            "Stale, future, or extended common launch deadline")
    return monotonic() + deadline - now


def verify_host_paths(host, entrypoint=__file__):
    require(Path(sys.executable).resolve() == Path(host["python"]).resolve(),
            "Wrong host Python")
    source = Path(host["source_root"]).resolve()
    require(Path.cwd().resolve() == source, "Pinned source working directory required")
    require(Path(entrypoint).resolve() == source / "paper/scripts" / Path(entrypoint).name,
            "Run the pinned source wrapper")
    require(source == Path(host["root"]).resolve() / "source"
            and Path(host["output_dir"]).resolve() == Path(host["root"]).resolve() / "run",
            "Pinned host root layout differs")


def check_storage(output, limits):
    total = 0
    for path in Path(output).rglob("*"):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode), "Output symlinks are forbidden")
        if stat.S_ISREG(info.st_mode):
            require(info.st_size <= limits["max_file_bytes"], "Per-file output budget reached")
            total += info.st_size
    require(total <= limits["per_host_output_bytes"], "Host output budget reached")
    require(shutil.disk_usage(output).free >= limits["min_free_bytes"],
            "Minimum free disk budget reached")
    return total


class WorkerBudget:
    def __init__(self, contract, permit, host, owner_pid, common=None):
        self.contract, self.permit = contract, permit
        self.host = contract["hosts"][host]
        self.owner_pid = owner_pid
        self.common = common or common_module()
        seconds = contract["limits"]["total_seconds"]
        self.deadline = fixed_start(permit) if seconds == 86400 else fixed_start(permit, total_seconds=seconds)
        self.last_scan = -math.inf

    def __call__(self):
        require(os.getppid() == self.owner_pid, "Owned supervisor is no longer the parent")
        require(time.time() < self.permit["deadline_unix"] and time.monotonic() < self.deadline,
                "Common execution deadline reached")
        if time.monotonic() - self.last_scan >= 10:
            check_storage(self.host["output_dir"], self.contract["limits"])
            require(self.common.gpu_pids(self.host) <= {os.getpid()},
                    "Another GPU process appeared; stopping only this worker")
            self.last_scan = time.monotonic()


def _hash_batch(digest, value):
    """Length-frame actual CPU tensor dtype, shape, and bytes in batch order."""
    import torch
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "Exposure hashing requires original CPU batches")
        header = json.dumps(["tensor", str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        for item in (header, raw):
            digest.update(len(item).to_bytes(8, "little")); digest.update(item)
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence" + len(value).to_bytes(8, "little"))
        for item in value:
            _hash_batch(digest, item)
    elif value is None:
        digest.update(b"none")
    else:
        raise ValueError(f"Unsupported batch component: {type(value).__name__}")


class ExposureLoader:
    """Transparent loader wrapper; record only completely consumed iterations."""
    def __init__(self, loader, split, records, budget_check, on_start=None):
        self.loader, self.split, self.records = loader, split, records
        self.budget_check = budget_check
        self.on_start = on_start

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.on_start is not None:
            self.on_start()
        digest, count, batches = hashlib.sha256(), 0, 0
        for batch in self.loader:
            self.budget_check()
            _hash_batch(digest, batch)
            count += int(batch[1].shape[0]); batches += 1
            yield batch
        self.budget_check()
        self.records.append({"epoch": len(self.records) + 1, "split": self.split,
            "count": count, "batches": batches, "batch_order_sha256": digest.hexdigest()})


@contextmanager
def audited_training(training, data, budget_check, status_callback):
    """Wrap the genuine trainer's loaders/save hooks and restore them always."""
    records = {"train": [], "validation": []}
    epoch_clock, epoch_timings = {}, []
    original_loader, original_save = training.make_loader, training.atomic_torch_save

    def start_epoch():
        epoch_clock.update(monotonic=time.monotonic(), started_at_unix=time.time())

    def make_loader(*args, **kwargs):
        split = kwargs["target_split"]
        require(split in records, "Only train/validation loaders are admitted")
        loader = original_loader(*args, **kwargs)
        return ExposureLoader(loader, split, records[split], budget_check,
                              on_start=start_epoch if split == "train" else None)

    def save(payload, path):
        budget_check()
        if Path(path).name == "last_epoch_state.pt":
            epoch = int(payload["epoch"])
            require(len(records["train"]) == len(records["validation"]) == epoch,
                    "Saved checkpoint lacks matching completed epoch exposure")
            expected_steps = sum(row["batches"] for row in records["train"])
            optimizer_states = payload["optimizer_state_dict"]["state"]
            require(bool(optimizer_states) and all(
                float(value["step"]) == expected_steps for value in optimizer_states.values()),
                "Actual optimizer step counters differ from batch exposure")
            row = payload["history"][-1]
            require(row["train_event_count"] == records["train"][-1]["count"]
                    and row["train_batch_count"] == records["train"][-1]["batches"],
                    "Trainer telemetry differs from observed batches")
        original_save(payload, path)
        budget_check()
        if Path(path).name == "last_epoch_state.pt":
            epoch_timings.append({"epoch": epoch,
                "started_at_unix": epoch_clock["started_at_unix"],
                "finished_at_unix": time.time(),
                "elapsed_seconds": time.monotonic() - epoch_clock["monotonic"]})
            write_status(Path(path).parent / "epoch_timing.json", {
                "scope": "train_iteration_through_validation_and_checkpoint_save",
                "epochs": epoch_timings})
            status_callback(epoch, records)

    training.make_loader, training.atomic_torch_save = make_loader, save
    try:
        yield records
    finally:
        training.make_loader, training.atomic_torch_save = original_loader, original_save


def audit_exposure(exposures, data, epochs=120):
    require(len(exposures["train"]) == epochs, "Training epoch exposure is incomplete")
    require(len(exposures["validation"]) == epochs + 1,
            "Expected ordinary validation epochs plus one selected replay")
    populations, batch_size = data["inherited_data_identity"]["populations"], data["loader"]["batch_size"]
    for split in ("train", "validation"):
        expected = populations[split]["target_count"]
        for index, row in enumerate(exposures[split], 1):
            require(row["epoch"] == index and row["split"] == split
                    and row["count"] == expected
                    and row["batches"] == math.ceil(expected / batch_size),
                    "Per-epoch target/batch exposure differs")
    steps = sum(row["batches"] for row in exposures["train"])
    require(steps == data["expected_global_steps"], "Optimizer-step budget differs")
    return steps


def training_args(contract, data, output_dir):
    """Explicit shared-trainer arguments; this does not invoke its generic CLI."""
    model, loader, optimizer = data["model"], data["loader"], data["optimizer"]
    require(optimizer == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.},
            "Frozen AdamW contract changed")
    require(loader["num_workers"] == 0 and loader["drop_last"] is False
            and loader["train_shuffle"] is True and loader["validation_shuffle"] is False,
            "Frozen loader behavior changed")
    require(model["quantity_variant"] == VARIANT and model["lambda_tail"] == 0.
            and model["time_head_mode"] == "legacy_clamped_rmtpp"
            and model["time_intercept_limit"] == 300., "Frozen head/objective changed")
    boundaries = data["quantity_boundaries_all_train_rows"]
    identity = data["inherited_data_identity"]
    return argparse.Namespace(output_dir=Path(output_dir), force_rerun=False,
        epochs=120, min_epochs=120, early_stopping_patience=120,
        batch_size=loader["batch_size"], lookback_weeks=loader["lookback_weeks"],
        max_seq_len=loader["max_seq_len"], hidden_dim=model["hidden_dim"],
        lr=.001, weight_decay=.01, grad_clip=1., lambda_log_qty=1., lambda_tail=0.,
        quantity_sigma_floor=1e-3, lambda_location_huber=1., location_huber_delta=.25,
        tail_threshold=float(boundaries[2]), tail_normalization_scale=float(boundaries[2]),
        tail_clip_cap=float(boundaries[3]), tail_huber_delta=1.,
        time_head_mode=model["time_head_mode"], time_scale=model["time_scale"],
        time_w_max=model["time_w_max"], time_intercept_limit=300.,
        time_initial_intercept=0., time_wd_safety_limit=40., time_head_lr_multiplier=1.,
        time_sigma_floor=1e-3, titans_memory_gradient_clip=None,
        max_train_batches=None, max_val_batches=None,
        quantile_adaptive_strength=0., device="cuda:0",
        checkpoint_monitor="validation_raw_quantity_rmse",
        source_revision=contract["source"]["base_git_revision"],
        execution_role="fresh_observed_slot_parallel_validation", dataset_contract=data["dataset_id"],
        data=Path(identity["data"]["path"]), data_sha256=identity["data"]["sha256"],
        split_manifest=Path(identity["split_manifest"]["path"]),
        split_manifest_sha256=identity["split_manifest"]["sha256"],
        max_series=None, allow_partial_contract=False)


def _build_model(data, backbone):
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = dict(data["model"])
    config.pop("backbone")
    return build_count_aware_model(backbone, **config,
        train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"],
        max_seq_len=data["loader"]["max_seq_len"])


def paired_initialization(data):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline, _ = _build_model(data, BASELINE)
        baseline_rng = torch.get_rng_state()
        torch.manual_seed(42)
        candidate, _ = _build_model(data, CANDIDATE)
        baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
        require(torch.equal(baseline_rng, torch.get_rng_state()), "Paired initialization CPU RNG differs")
        require(all(name in candidate_state and torch.equal(value, candidate_state[name])
                    for name, value in baseline_state.items()), "Paired shared initial parameters differ")
        baseline_sha = canonical_state_dict_sha256(baseline_state)
        return {"shared_initial_state_sha256": baseline_sha,
            "baseline_initial_state_sha256": baseline_sha,
            "candidate_initial_state_sha256": canonical_state_dict_sha256(candidate_state),
            "shared_state_equal": True, "caller_cpu_rng_equal": True,
            "shared_state_keys": list(baseline_state)}


def replay_checkpoint(path, data, frame, budget_check, *, device="cuda:0", allowed_backbones=(BASELINE, CANDIDATE)):
    """One streaming validation pass with fixed train-derived strata; no records."""
    import numpy as np
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import torch_load_checkpoint, canonical_state_dict_sha256
    payload = torch_load_checkpoint(Path(path), map_location="cpu")
    require(payload.get("backbone") in allowed_backbones, "Unexpected endpoint backbone")
    validate_checkpoint_route(payload, payload["backbone"])
    require(payload["evaluation_scope"] == "validation_only"
            and payload["held_out_test_evaluated"] is False, "Replay split changed")
    state_digest = canonical_state_dict_sha256(payload["model_state_dict"])
    require(payload.get("model_state_sha256") == state_digest, "Checkpoint state digest differs")
    model, _ = _build_model(data, payload["backbone"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = make_loader(frame, target_split="validation", **{
        key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")},
        shuffle=False, generator=None)
    q_bounds, h_bounds = data["quantity_boundaries_all_train_rows"], data["history_boundaries"]
    accumulator = {"overall": [0, 0., 0., 0.], "body": [0, 0., 0., 0.], "tail": [0, 0., 0., 0.]}
    for kind, bounds in (("quantity", q_bounds), ("history", h_bounds)):
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
            for kind, values, bounds in (("quantity", q, q_bounds), ("history", h, h_bounds)):
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
        return {"count": n, "qty_mae": absolute / n if n else None,
            "qty_rmse": math.sqrt(squared / n) if n else None,
            "time_nll": temporal / n if n else None}
    result = {**finish(accumulator["overall"]), "body": finish(accumulator["body"]),
        "tail": finish(accumulator["tail"]), "quantity_boundaries": q_bounds,
        "history_boundaries": h_bounds, "quantity_cells": [], "history_cells": [],
        "checkpoint_path": str(path), "state_sha256": state_digest,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    for kind, bounds in (("quantity", q_bounds), ("history", h_bounds)):
        result[kind + "_cells"] = [{"bin": i, **finish(accumulator[f"{kind}_{i}"])}
                                    for i in range(len(bounds) + 1)]
    require(result["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
            "Replay validation target count changed")
    return result


def last30_summary(history):
    """Summarize the last 30 validation RMSE values with sample SD (ddof=1)."""
    require(len(history) >= 30, "Thirty completed epochs are required for the endpoint stability summary")
    values = [row["val_qty_rmse"] for row in history[-30:]]
    require(all(type(value) in (int, float) and math.isfinite(value) for value in values),
            "Last-30 RMSE values must be finite measurements")
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "count": 30, "standard_deviation": "sample_ddof_1"}


def paired_acceptance(baseline, candidate):
    """Quality gates classify the fresh pair; a failed gate is a valid result."""
    for arm in (baseline, candidate):
        require(isinstance(arm, dict) and isinstance(arm.get("selected"), dict)
                and isinstance(arm.get("last30"), dict), "Acceptance metric structure is missing")
        selected = arm["selected"]
        require(all(name in selected for name in ("qty_rmse", "qty_mae", "time_nll"))
                and all(isinstance(selected.get(name), dict) and "qty_mae" in selected[name]
                        for name in ("body", "tail"))
                and all(name in arm["last30"] for name in ("mean", "sd")),
                "Acceptance required metrics are missing")
    b, c = baseline["selected"], candidate["selected"]
    def finite_compare(left, right, *, factor=1., additive=0., strict=False):
        require(type(left) in (int, float) and type(right) in (int, float)
                and math.isfinite(left) and math.isfinite(right),
                "Acceptance requires finite measured metrics")
        threshold = right * factor + additive
        return left < threshold if strict else left <= threshold
    checks = {
        "raw_rmse_improves": finite_compare(c["qty_rmse"], b["qty_rmse"], strict=True),
        "raw_mae_guard": finite_compare(c["qty_mae"], b["qty_mae"], factor=1.01),
        "body_mae_guard": finite_compare(c["body"]["qty_mae"], b["body"]["qty_mae"], factor=1.02),
        "tail_mae_guard": finite_compare(c["tail"]["qty_mae"], b["tail"]["qty_mae"], factor=1.02),
        "legacy_time_guard": finite_compare(c["time_nll"], b["time_nll"], additive=.01),
        "last30_rmse_mean_guard": finite_compare(candidate["last30"]["mean"], baseline["last30"]["mean"], factor=1.05),
        "last30_rmse_sd_guard": finite_compare(candidate["last30"]["sd"], baseline["last30"]["sd"], factor=1.05),
    }
    return {"accepted": all(checks.values()), "checks": checks,
        "scope": "single_seed_fresh_same_host_validation_pair_no_universal_claim"}


def _production_arms(contract, permit, host_name, qualification, budget):
    import gc
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    common = common_module()
    host = contract["hosts"][host_name]
    output, runtime = Path(host["output_dir"]), common.runtime_check(contract, host_name)
    require(runtime == qualification["runtime"], "Runtime differs from CUDA qualification")
    comparisons = {}
    for dataset_id in host["assigned_datasets"]:
        budget()
        data = next(value for value in contract["datasets"] if value["dataset_id"] == dataset_id)
        common.verify_source(contract)
        frame, metadata = prepare_quantity_comparison_data(data)
        metadata = bind_frozen_statistics(data, metadata)
        require(metadata["held_out_materialized"] is False
                and metadata["populations"] == data["inherited_data_identity"]["populations"],
                "Prepared split/population differs")
        initialization = paired_initialization(data)
        dataset_output = output / dataset_id
        dataset_output.mkdir(exist_ok=False)
        common.write_json(dataset_output / "initialization.json", initialization, exclusive=True)
        common.write_json(dataset_output / "input_receipt.json", metadata, exclusive=True)
        boundaries = data["quantity_boundaries_all_train_rows"]
        quantity_contract = {"boundaries": boundaries,
            "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(len(boundaries) + 1)]}
        interface = {"train_target_mean": metadata["train_log_mean"],
            "train_target_std": metadata["train_log_std"], "time_head": {"time_initial_intercept": 0.},
            "data_scope": "verified_train_validation_only", "statistics": data["statistics"],
            "execution_contract_sha256": common.sha_json(contract),
            "source_files_sha256": contract["source"]["files_sha256"]}
        arms, all_exposure = {}, {}
        for backbone in (BASELINE, CANDIDATE):
            common.verify_source(contract)
            require(common.runtime_check(contract, host_name) == runtime, "Runtime changed before arm")
            require(common.gpu_pids(host) <= {os.getpid()}, "Another GPU job appeared; no foreign job will be stopped")
            args = training_args(contract, data, dataset_output)
            args.model_role = "hard_lmm_observed_slot_memory_candidate" if backbone == CANDIDATE else "existing_backbone"
            run_dir = dataset_output / "runs" / backbone / VARIANT / "seed_42"
            require(not run_dir.exists(), "Fresh arm required; reuse/resume/overwrite forbidden")
            def status(epoch, records):
                write_status(output / "status.json", {"status": "training", "host": host_name,
                    "dataset": dataset_id, "backbone": backbone, "epoch": epoch,
                    "global_steps": sum(row["batches"] for row in records["train"]),
                    "deadline_unix": permit["deadline_unix"]})
                write_status(run_dir / "exposure.json", records)
            write_status(output / "status.json", {"status": "training", "host": host_name,
                    "dataset": dataset_id, "backbone": backbone, "epoch": 0,
                    "deadline_unix": permit["deadline_unix"]})
            with audited_training(training, data, budget, status) as exposure:
                summary, _, _ = training.train_one(args=args, frame=frame,
                    quantity_contract=quantity_contract, interface_meta=interface,
                    backbone=backbone, quantity_variant=VARIANT, seed=42)
            require(summary["status"] == "success" and summary["completed_epochs"] == 120
                    and summary["stopped_early"] is False, "Fixed 120-epoch arm incomplete")
            expected_initial = initialization[("candidate" if backbone == CANDIDATE else "baseline") + "_initial_state_sha256"]
            require(summary["initial_state_sha256"] == expected_initial, "Actual trainer initialization differs")
            steps = audit_exposure(exposure, data)
            all_exposure[backbone] = exposure
            write_status(run_dir / "exposure.json", exposure)
            history = read_json(run_dir / "history.json")["history"]
            require(len(history) == 120, "Checkpoint history incomplete")
            replay = {}
            for label, path in (("selected", Path(summary["checkpoint_path"])),
                                ("last", run_dir / "last_epoch_state.pt")):
                write_status(output / "status.json", {"status": "checkpoint_replay", "host": host_name,
                    "dataset": dataset_id, "backbone": backbone, "checkpoint": label,
                    "deadline_unix": permit["deadline_unix"]})
                replay[label] = replay_checkpoint(path, data, frame, budget)
                row = history[summary["best_epoch"] - 1] if label == "selected" else history[-1]
                require(math.isclose(replay[label]["qty_rmse"], row["val_qty_rmse"], rel_tol=1e-10, abs_tol=1e-8)
                        and math.isclose(replay[label]["qty_mae"], row["val_qty_mae"], rel_tol=1e-10, abs_tol=1e-8)
                        and math.isclose(replay[label]["time_nll"], row["val_time_nll"], rel_tol=1e-10, abs_tol=1e-8),
                        "Selected/last checkpoint replay differs from epoch history")
            arms[backbone] = {**replay, "last30": last30_summary(history), "global_steps": steps,
                "best_epoch": summary["best_epoch"], "initial_state_sha256": expected_initial}
            common.write_json(run_dir / "endpoint_replays.json", arms[backbone], exclusive=True)
            gc.collect(); torch.cuda.empty_cache()
        require(all_exposure[BASELINE] == all_exposure[CANDIDATE],
                "Paired actual train/validation batch exposure differs")
        result = {"status": "complete", "dataset": dataset_id, "host": host_name,
            "initialization": initialization, "arms": arms,
            "exposure_equal": True, "epochs": 120,
            "acceptance": paired_acceptance(arms[BASELINE], arms[CANDIDATE]),
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "baseline_provenance": "fresh_same_host_shared_trainer_not_historical_frozen_reproduction"}
        common.write_json(dataset_output / "paired_comparison.json", result, exclusive=True)
        comparisons[dataset_id] = result
        del frame
        gc.collect()
    return {"status": "complete", "host": host_name, "datasets": comparisons,
        "contract_sha256": common.sha_json(contract), "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False, "deadline_unix": permit["deadline_unix"]}


def worker(contract_path, permit_path, host_name, owner_pid, authority_fd, *,
           common=None, entrypoint=__file__, production=None, assignment_validator=None):
    common = common or common_module()
    contract = common.read_contract(Path(contract_path), host_name)
    permit = read_json(permit_path)
    host = (assignment_validator or validate_assignment)(contract, host_name)
    if entrypoint == __file__:
        verify_host_paths(host)
    else:
        verify_host_paths(host, entrypoint)
    common.verify_permit(contract, permit, host_name)
    require(owner_pid == os.getppid() and os.getsid(0) == os.getpid(), "Worker must belong to its new owned session")
    with os.fdopen(authority_fd, "rb") as handle:
        authority = json.loads(handle.read(4096))
    expected = {"contract_sha256": common.sha_json(contract), "permit_sha256": common.sha_json(permit),
                "host": host_name, "owner_pid": owner_pid}
    require(authority == expected, "Inherited worker authority differs")
    require(read_json(Path(host["output_dir"]) / "wrapper_manifest.json")["authority"] == expected,
            "Worker output is not owned by this supervisor")
    common.apply_environment(host)
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (contract["limits"]["max_file_bytes"],) * 2)
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    configure_runtime("cuda:0", threads=4)
    qualification = read_json(permit["qualifications"][host_name]["path"])
    require(qualification["status"] == "passed", "CUDA qualification did not pass")
    budget = WorkerBudget(contract, permit, host_name, owner_pid, common=common)
    budget()
    summary = (production or _production_arms)(contract, permit, host_name, qualification, budget)
    budget()
    common.write_json(Path(host["output_dir"]) / "partition_summary.json", summary, exclusive=True)
    write_status(Path(host["output_dir"]) / "status.json", {"status": "complete", "host": host_name,
        "deadline_unix": permit["deadline_unix"]})


def kill_owned_process_group(child, grace_seconds=15):
    """Signal only the process group created for this supervisor's child."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=grace_seconds)
    except ProcessLookupError:
        pass


def _supervisor_terminated(signum, _frame):
    raise InterruptedError(f"Owned supervisor received signal {signum}")


def supervisor(contract_path, permit_path, host_name, *, common=None, entrypoint=__file__,
               assignment_validator=None):
    common = common or common_module()
    contract = common.read_contract(Path(contract_path), host_name)
    permit = read_json(permit_path)
    host = (assignment_validator or validate_assignment)(contract, host_name)
    if entrypoint == __file__:
        verify_host_paths(host)
    else:
        verify_host_paths(host, entrypoint)
    common.verify_permit(contract, permit, host_name)
    seconds = contract["limits"]["total_seconds"]
    deadline = fixed_start(permit) if seconds == 86400 else fixed_start(permit, total_seconds=seconds)
    require(not common.gpu_pids(host), "GPU is busy; never stop another GPU job")
    output = Path(host["output_dir"])
    require(not output.exists(), "Fresh output required; reuse/resume/retry forbidden")
    output.mkdir(parents=True, exist_ok=False)
    authority = {"contract_sha256": common.sha_json(contract), "permit_sha256": common.sha_json(permit),
                 "host": host_name, "owner_pid": os.getpid()}
    common.write_json(output / "wrapper_manifest.json", {"authority": authority,
        "started_at_unix": time.time(), "common_started_at_unix": permit["started_at_unix"],
        "deadline_unix": permit["deadline_unix"], "assigned_datasets": host["assigned_datasets"],
        "retry": False, "resume": False, "held_out": False}, exclusive=True)
    common.apply_environment(host)
    read_fd, write_fd = os.pipe()
    child = None
    previous_terminate = signal.signal(signal.SIGTERM, _supervisor_terminated)
    try:
        with (output / "worker.log").open("xb") as log:
            child = subprocess.Popen([host["python"], str(Path(entrypoint).resolve()), "_worker",
                "--contract", str(Path(contract_path).resolve()), "--permit", str(Path(permit_path).resolve()),
                "--host", host_name, "--owner-pid", str(os.getpid()), "--authority-fd", str(read_fd)],
                cwd=host["source_root"], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(read_fd,))
            os.close(read_fd); read_fd = None
            with os.fdopen(write_fd, "wb") as handle:
                write_fd = None
                handle.write(json.dumps(authority).encode())
            while child.poll() is None:
                require(time.time() < permit["deadline_unix"] and time.monotonic() < deadline,
                        "Common execution deadline reached")
                check_storage(output, contract["limits"])
                time.sleep(min(2., max(.01, deadline - time.monotonic())))
            require(child.returncode == 0, f"Owned worker failed with return code {child.returncode}")
            require(time.time() < permit["deadline_unix"] and time.monotonic() < deadline,
                    "Common execution deadline reached before terminal validation")
            summary = read_json(output / "partition_summary.json")
            require(summary["status"] == "complete" and summary["contract_sha256"] == common.sha_json(contract)
                    and set(summary["datasets"]) == set(host["assigned_datasets"]), "Worker terminal receipt incomplete")
            check_storage(output, contract["limits"])
            return summary
    except BaseException as error:
        if child is not None:
            kill_owned_process_group(child)
        write_status(output / "status.json", {"status": "stopped", "host": host_name,
            "error": str(error), "deadline_unix": permit["deadline_unix"], "automatic_retry": False})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_terminate)
        for descriptor in (read_fd, write_fd):
            if descriptor is not None:
                os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "_worker"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--host", choices=tuple(ASSIGNMENT), required=True)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--authority-fd", type=int)
    args = parser.parse_args()
    if args.command == "execute":
        require(args.owner_pid is None and args.authority_fd is None, "Supervisor cannot inherit worker arguments")
        supervisor(args.contract, args.permit, args.host)
    else:
        require(args.owner_pid is not None and args.authority_fd is not None, "Internal worker authority required")
        worker(args.contract, args.permit, args.host, args.owner_pid, args.authority_fd)


if __name__ == "__main__":
    main()
