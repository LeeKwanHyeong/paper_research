#!/usr/bin/env python3
"""Explicit, once-only cross-GPU continuation above an immutable A100 engine.

Importing this adapter performs no model, data, GPU or network work. Its three
CLI modes are local operations; callers own retrieval, tmux and host deployment.
All research evaluation is full Validation. The source checkpoint and its A100
history remain immutable, including the original earliest-minimum selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

SCHEMA = "titantpp_cnn_gru_resume_dual_v1"
PARENT_SHA = "46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae"
SOURCE_SHA = "40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd"
WORKER_SHA = "7d961867f612f86e1b78346630764d7624472532f596aba133a3bf3d24da2f26"
VARIANT = "count_only_log_regression"
ARMS = ("titantpp_cnn_mlp16", "titantpp_gru54", "titantpp_cnn_gru54")
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
METRICS = {"qty_rmse": "val_qty_rmse", "qty_mae": "val_qty_mae", "time_nll": "val_time_nll"}
TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
REQUIRED_ARTIFACTS = {"last_epoch_state.pt", "history.json", "exposure.json", "epoch_timing.json", "train.log"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary, path)


def confined(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root), "Path escaped admitted root: " + str(name))
    return path


def validate_contract(c, parent):
    """Only host/path/continuation administration changes; science stays frozen."""
    require(c.get("schema") == SCHEMA, "Foreign resume campaign")
    require(c["parent"]["contract_sha256"] == PARENT_SHA == canonical(parent), "Parent contract changed")
    require(parent["source"]["files_sha256"] == SOURCE_SHA == c["parent"]["source_files_sha256"], "Parent source changed")
    require(parent["worker"]["files_sha256"] == WORKER_SHA == c["parent"]["worker_files_sha256"], "Parent worker changed")
    for key in ("source", "worker", "training"):
        require(c[key] == parent[key], "Frozen scientific/worker contract changed: " + key)
    require(c["evaluation_scope"] == "validation_only" and c["held_out_test_evaluated"] is False,
        "Test access is forbidden")
    require(c["replay_tolerance"] == TOLERANCE, "Cross-GPU replay tolerance changed")
    require(c["limits"]["automatic_retry"] is False and c["limits"]["manual_resume_once"] is True
        and c["limits"]["workers_per_host"] == 1, "Only explicit once-only single-worker continuation is admitted")
    require(0 < c["limits"]["per_condition_seconds"] <= c["limits"]["total_wall_seconds"]
        and 0 < c["limits"]["max_device_memory_fraction"] <= .8, "Enlarged/invalid resource ceilings")
    require(set(c["hosts"]) == {"5080", "5090"}, "Only approved 5080/5090 hosts are admitted")
    jobs = c["jobs"]
    parent_jobs = {j["id"]: j for j in parent["jobs"]}
    require(jobs and len({j["id"] for j in jobs}) == len(jobs), "Empty/duplicated continuation jobs")
    for job in jobs:
        origin = parent_jobs.get(job["id"])
        require(origin is not None and {k: v for k, v in job.items() if k != "host"}
            == {k: v for k, v in origin.items() if k != "host"}, "Job science changed")
        require(job["host"] in c["hosts"] and job["seed"] == 42 and job["arm"] in ARMS
            and job["dataset"] in DATASETS, "Foreign continuation condition")
        a = c["artifacts"][job["id"]]
        require(a["origin_contract_sha256"] == PARENT_SHA and a["terminal_scientific_success"] is False,
            "Already completed/foreign source fit cannot resume")
        require(REQUIRED_ARTIFACTS <= set(a["files"])
            and a["source_checkpoint_sha256"] == a["files"]["last_epoch_state.pt"], "Incomplete checkpoint original seal")
        data = next(d for d in parent["datasets"] if d["dataset_id"] == job["dataset"])
        mappings = c["path_mappings"][job["id"]]
        require(set(mappings) == {"data", "split_manifest"}, "Only data/split paths may map")
        for key, mapping in mappings.items():
            original = data["inherited_data_identity"][key]
            require(mapping["origin_path"] == original["path"] and mapping["sha256"] == original["sha256"]
                and mapping["independently_verified"] is True and Path(mapping["target_path"]).is_absolute(),
                "Unverified or changed input path mapping")
    require(set(c["artifacts"]) == set(c["path_mappings"]) == {j["id"] for j in jobs}, "Continuation job coverage differs")
    return c


def load(contract_path, host, job_id, *, verify_bytes=True):
    contract_path = Path(contract_path).resolve()
    c = read(contract_path)
    require(host in c["hosts"], "Unapproved host")
    root = Path(c["hosts"][host]["root"]).resolve()
    require(contract_path == root / "execution_contract.json", "Wrong host contract root")
    parent = read(confined(root, c["parent"]["contract"]))
    validate_contract(c, parent)
    jobs = [j for j in c["jobs"] if j["id"] == job_id and j["host"] == host]
    require(len(jobs) == 1, "Job is not assigned to this host")
    job = jobs[0]
    if verify_bytes:
        require(sha(__file__) == c["operation_adapter_sha256"], "Operational adapter seal changed")
        source_root = Path(c["hosts"][host]["source_root"]).resolve()
        worker_root = Path(c["hosts"][host]["worker_root"]).resolve()
        require(source_root == root / "source" and worker_root == root / "worker", "Foreign source/worker roots")
        require(len(c["source"]["files"]) == 117 and canonical(c["source"]["files"]) == SOURCE_SHA,
            "Scientific source closure differs")
        require(canonical(c["worker"]["files"]) == WORKER_SHA, "Worker closure differs")
        for name, digest in c["source"]["files"].items():
            require(sha(confined(source_root, name)) == digest, "Scientific source changed: " + name)
        for name, digest in c["worker"]["files"].items():
            require(name.startswith("worker/") and sha(confined(root, name)) == digest, "Parent worker changed: " + name)
        a = c["artifacts"][job_id]
        origin = Path(a["origin_root"]).resolve()
        require(origin.is_relative_to(root / "originals"), "Originals must stay in the new dedicated root")
        for name, digest in a["files"].items():
            require(sha(confined(origin, name)) == digest, "Immutable original SHA changed: " + name)
        for mapping in c["path_mappings"][job_id].values():
            require(sha(mapping["target_path"]) == mapping["sha256"], "Mapped data/split bytes changed")
    return c, parent, root, job


def authorize(c, root, host, *, now=None):
    approval, start = read(root / "approval.json"), read(root / "start_permit.json")
    require(approval.get("approved") is True and approval.get("manual_resume_once") is True
        and approval["contract_sha256"] == canonical(c) and approval["parent_contract_sha256"] == PARENT_SHA
        and set(approval["hosts"]) == set(c["hosts"]) and set(approval["job_ids"]) == {j["id"] for j in c["jobs"]}
        and isinstance(approval.get("user_instruction"), str) and approval["user_instruction"].strip(),
        "Explicit continuation approval is missing or foreign")
    require(start["contract_sha256"] == canonical(c) and start["approval_sha256"] == canonical(approval),
        "Continuation start permit does not bind approval")
    now = time.time() if now is None else now
    require(start["started_at_unix"] <= now < start["deadline_unix"]
        and 0 < start["deadline_unix"] - start["started_at_unix"] <= c["limits"]["total_wall_seconds"],
        "Continuation absolute deadline exhausted/enlarged")
    return start


def mapped_dataset(c, parent, job):
    data = deepcopy(next(d for d in parent["datasets"] if d["dataset_id"] == job["dataset"]))
    for key, mapping in c["path_mappings"][job["id"]].items():
        data["inherited_data_identity"][key]["path"] = mapping["target_path"]
    data["prepared_train_validation_path"] = data["inherited_data_identity"]["data"]["path"]
    return data


def canonicalize_identity(identity, mappings):
    """Normalize exactly two approved absolute paths, never mutate the payload."""
    value = deepcopy(identity)
    for key, mapping in mappings.items():
        current = str(value["arguments"][key])
        require(current in {mapping["target_path"], mapping["origin_path"]}, "Unexpected identity path: " + key)
        require(value["arguments"][key + "_sha256"] == mapping["sha256"], "Identity data digest changed")
        value["arguments"][key] = mapping["origin_path"]
    return value


def load_engine(c, host):
    source = Path(c["hosts"][host]["source_root"]).resolve()
    sys.path.insert(0, str(source))
    engine = importlib.import_module("paper.scripts.run_titantpp_cnn_gru")
    training = importlib.import_module("paper.scripts.count_aware_tpp_backbone.training")
    require(Path(engine.__file__).resolve().is_relative_to(source)
        and Path(training.__file__).resolve().is_relative_to(source), "Engine imported outside sealed source")
    engine.install_hooks()
    return engine, training


def runtime_check(c, host):
    spec = c["hosts"][host]
    require(Path(sys.executable).resolve() == Path(spec["python"]).resolve(), "Wrong pinned Python executable")
    require(Path.cwd().resolve() == Path(spec["source_root"]).resolve(), "Wrong sealed source cwd")
    require(all(os.environ.get(k) == v for k, v in spec["environment"].items()), "Process environment changed")
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    from paper.scripts.run_time_quantity_diagnostic import validate_import_origins
    configure_runtime("cuda:0", threads=2)
    actual = runtime_identity("cuda:0")
    validate_import_origins()
    require(actual["gpu"]["uuid"] == spec["gpu_uuid"] and actual["gpu"]["visible_device_count"] == 1
        and spec["gpu_name_contains"] in actual["gpu"]["name"], "Wrong native GPU")
    require(all(actual.get(k) == v for k, v in spec["runtime_expected"].items()), "Pinned native libraries changed")
    require(actual["deterministic_algorithms"] and not actual["deterministic_warn_only"], "Strict determinism is required")
    return actual


def validate_prefix(payload, history, exposure, timing, data, artifact, *, state_digest):
    epoch = payload["epoch"]
    require(type(epoch) is int and 1 <= epoch <= 300 and payload["history"] == history
        and [r["epoch"] for r in history] == list(range(1, epoch + 1)), "Original checkpoint/history epoch mismatch")
    require(epoch == artifact["source_epoch"] and payload["best_epoch"] == artifact["source_best_epoch"]
        and payload["best_state_sha256"] == artifact["source_best_state_sha256"]
        and payload["model_state_sha256"] == artifact["source_current_state_sha256"]
        and canonical(payload["resume_identity"]) == artifact["source_resume_identity_sha256"], "Recovered checkpoint identity differs")
    require(state_digest(payload["model_state_dict"]) == payload["model_state_sha256"]
        and state_digest(payload["best_state_dict"]) == payload["best_state_sha256"], "Recovered state tensor digest differs")
    require(payload["evaluation_scope"] == "validation_only" and payload["held_out_test_evaluated"] is False,
        "Checkpoint Test scope forbidden")
    records = {}
    for split in ("train", "validation"):
        require(len(exposure[split]) >= epoch, "Checkpoint exposure prefix is incomplete")
        records[split] = deepcopy(exposure[split][:epoch])
        expected = data["inherited_data_identity"]["populations"][split]["target_count"]
        batches = math.ceil(expected / data["loader"]["batch_size"])
        for number, row in enumerate(records[split], 1):
            require(row["epoch"] == number and row["split"] == split and row["count"] == expected
                and row["batches"] == batches and len(row["batch_order_sha256"]) == 64,
                "Original batch exposure changed")
    for row, observed in zip(history, records["train"]):
        require(row["train_event_count"] == observed["count"] and row["train_batch_count"] == observed["batches"],
            "Original train telemetry/exposure differs")
    steps = sum(r["batches"] for r in records["train"])
    states = payload["optimizer_state_dict"]["state"]
    require(states and all(float(v["step"]) == steps for v in states.values()), "Original optimizer steps are not cumulative")
    times = deepcopy(timing["epochs"][:epoch])
    require(len(times) == epoch and [r["epoch"] for r in times] == list(range(1, epoch + 1))
        and all(math.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > 0 for r in times),
        "Original completed epoch timing prefix is incomplete")
    return records, times, steps


def inspect_original(c, parent, root, job, training):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    artifact = c["artifacts"][job["id"]]
    origin = Path(artifact["origin_root"])
    payload = torch_load_checkpoint(origin / "last_epoch_state.pt", map_location="cpu")
    require(payload["backbone"] == job["arm"] and payload["seed"] == job["seed"] and payload["variant"] == VARIANT,
        "Recovered checkpoint belongs to another condition")
    identity = payload["resume_identity"]
    require(identity["arguments"]["source_revision"] == parent["source"]["base_git_revision"], "Checkpoint source revision differs")
    require(canonicalize_identity(identity, c["path_mappings"][job["id"]]) == identity, "Original identity path changed")
    training._validate_resume_payload(payload, expected_identity=identity,
        expected_initial_state_sha256=payload["initial_state_sha256"], monitor="validation_raw_quantity_rmse")
    data = next(d for d in parent["datasets"] if d["dataset_id"] == job["dataset"])
    records, times, steps = validate_prefix(payload, read(origin / "history.json")["history"],
        read(origin / "exposure.json"), read(origin / "epoch_timing.json"), data, artifact,
        state_digest=canonical_state_dict_sha256)
    report = {"status": "verified", "job": job, "source_epoch": payload["epoch"], "next_epoch": payload["epoch"] + 1,
        "source_best_epoch": payload["best_epoch"], "source_best_selection_value": payload["best_selection_value"],
        "source_best_state_sha256": payload["best_state_sha256"], "source_current_state_sha256": payload["model_state_sha256"],
        "source_checkpoint_sha256": artifact["source_checkpoint_sha256"], "source_resume_identity_sha256": canonical(identity),
        "source_global_steps": steps, "optimizer_rng_loader_generator_present": True,
        "already_at_prescribed_stop": training.early_stopping_exhausted(payload["history"], min_epochs=40,
            patience=40, metric_key="val_qty_rmse") or payload["epoch"] == 300,
        "discarded_non_checkpoint_exposure": {s: len(read(origin / "exposure.json")[s]) - len(records[s]) for s in records},
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    return payload, records, times, report


class Budget:
    def __init__(self, root, c, job, deadline):
        self.root, self.c, self.job, self.deadline = root, c, job, deadline
        self.monotonic_deadline = time.monotonic() + max(0., deadline - time.time())
        self.last = 0.

    def __call__(self):
        now = time.time()
        require(now < self.deadline and time.monotonic() < self.monotonic_deadline, "Continuation absolute budget exhausted")
        if now - self.last >= 2:
            require(shutil.disk_usage(self.root).free > 5 * 1024 ** 3, "Free disk below 5 GiB")
            write(self.root / "progress" / (self.job["id"] + ".json"), {"pid": os.getpid(), "job": self.job,
                "contract_sha256": canonical(self.c), "deadline_unix": self.deadline, "updated_unix": now})
            self.last = now


def compare_replay(replay, history_row, *, tolerance=TOLERANCE):
    require(replay["evaluation_scope"] == "validation_only" and replay["held_out_test_evaluated"] is False,
        "Replay scope changed")
    return {name: math.isclose(replay[name], history_row[key], rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"])
        and math.isfinite(replay[name]) and math.isfinite(history_row[key]) for name, key in METRICS.items()}


def gpu_segments(source_epoch, completed_epoch, host):
    require(completed_epoch >= source_epoch, "Continuation lost source epochs")
    segments = [{"host": "a100", "first_epoch": 1, "last_epoch": source_epoch}]
    if completed_epoch > source_epoch:
        segments.append({"host": host, "first_epoch": source_epoch + 1, "last_epoch": completed_epoch})
    return segments


def synthetic_resume_check(engine, data, arm, budget, *, device="cuda:0"):
    """Actual joint-loss updates compare uninterrupted and restored synthetic runs."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256, capture_rng_state, restore_rng_state,
        restore_train_loader_generator_state,
    )
    torch.manual_seed(42)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(42)
    generator = torch.Generator().manual_seed(20261005)
    model, _ = engine.build_model(data, arm)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    losses, checkpoint = [], None
    length = data["loader"]["max_seq_len"]

    def step(current, optim):
        budget()
        dt = torch.randint(1, 8, (128, length), generator=generator).float().to(device)
        qty = torch.randint(1, 20, (128, length), generator=generator).float().to(device)
        mask = torch.ones_like(dt, dtype=torch.bool)
        optim.zero_grad(set_to_none=True)
        loss = target_outputs(current, dt, mask, qty, lambda_log_qty=1.)["joint_loss"].mean()
        require(bool(torch.isfinite(loss)), "Nonfinite synthetic resume objective")
        loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in current.parameters()),
            "Nonfinite synthetic resume gradient")
        torch.nn.utils.clip_grad_norm_(current.parameters(), 1.)
        optim.step()
        return float(loss.detach().cpu())

    for index in range(3):
        losses.append(step(model, optimizer))
        if index == 0:
            checkpoint = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": deepcopy(optimizer.state_dict()), "rng": capture_rng_state(),
                "loader_generator": generator.get_state().clone()}
    uninterrupted = canonical_state_dict_sha256(model.state_dict())
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    restored, _ = engine.build_model(data, arm)
    restored = restored.to(device).train()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001, weight_decay=.01)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint["rng"])
    restore_train_loader_generator_state(generator, checkpoint["loader_generator"])
    repeated_losses = [step(restored, restored_optimizer) for _ in range(2)]
    resumed = canonical_state_dict_sha256(restored.state_dict())
    require(uninterrupted == resumed and losses[1:] == repeated_losses,
        "Native model/optimizer/RNG/loader-generator continuation differs")
    require(all(float(v["step"]) == 3 for v in restored_optimizer.state_dict()["state"].values()),
        "Synthetic resumed optimizer steps restarted")
    del restored, restored_optimizer, checkpoint
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"status": "passed", "uninterrupted_state_sha256": uninterrupted,
        "resumed_state_sha256": resumed, "uninterrupted_losses": losses,
        "resumed_losses": repeated_losses, "restored_after_step": 1, "completed_steps": 3,
        "batch_size": 128, "sequence_length": length, "real_data_loaded": False,
        "held_out_test_evaluated": False}


@contextmanager
def identity_mapping(training, mappings, expected_identity):
    original = training._resume_identity

    def mapped(**kwargs):
        value = canonicalize_identity(original(**kwargs), mappings)
        require(value == expected_identity, "Unapproved training identity drift beyond input paths")
        return value

    training._resume_identity = mapped
    try:
        yield
    finally:
        training._resume_identity = original


@contextmanager
def strict_seed(training):
    """Keep the admitted strict runtime after the historical seeding helper."""
    import torch
    original = training.set_seed

    def seeded(seed):
        generator = original(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        require(torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled(), "Strict runtime was relaxed by seeding")
        return generator

    training.set_seed = seeded
    try:
        yield
    finally:
        training.set_seed = original


@contextmanager
def audited_resume(training, shared, data, budget, status, prefix, timing_prefix, source_epoch):
    """Continue audited exposure numbering and optimizer counters after epoch E."""
    records, times = deepcopy(prefix), deepcopy(timing_prefix)
    original_loader, original_save = training.make_loader, training.atomic_torch_save
    clock = {}

    def start_epoch():
        clock.update(monotonic=time.monotonic(), started_at_unix=time.time())

    def make_loader(*args, **kwargs):
        split = kwargs["target_split"]
        require(split in records, "Only train/Validation loaders are admitted")
        return shared.ExposureLoader(original_loader(*args, **kwargs), split, records[split], budget,
            on_start=start_epoch if split == "train" else None)

    def save(payload, path):
        budget()
        if Path(path).name == "last_epoch_state.pt":
            epoch = payload["epoch"]
            require(epoch > source_epoch and len(records["train"]) == len(records["validation"]) == epoch,
                "Resumed saved epoch lacks complete cumulative exposure")
            require(payload["history"][:source_epoch] == source_history,
                "Immutable A100 history prefix changed")
            steps = sum(r["batches"] for r in records["train"])
            states = payload["optimizer_state_dict"]["state"]
            require(states and all(float(v["step"]) == steps for v in states.values()), "Resumed optimizer steps restarted")
            row = payload["history"][-1]
            require(row["train_event_count"] == records["train"][-1]["count"]
                and row["train_batch_count"] == records["train"][-1]["batches"], "Resumed telemetry differs from batches")
        original_save(payload, path)
        budget()
        if Path(path).name == "last_epoch_state.pt":
            times.append({"epoch": epoch, "started_at_unix": clock["started_at_unix"],
                "finished_at_unix": time.time(), "elapsed_seconds": time.monotonic() - clock["monotonic"]})
            write(Path(path).parent / "epoch_timing.json", {
                "scope": "train_iteration_through_validation_and_checkpoint_save", "epochs": times})
            status(epoch, records)

    # The caller binds the exact source history before entering the wrapper.
    source_history = deepcopy(training._titantpp_resume_source_history)
    require(len(source_history) == source_epoch, "Source history prefix is missing")
    training.make_loader, training.atomic_torch_save = make_loader, save
    try:
        yield records
    finally:
        training.make_loader, training.atomic_torch_save = original_loader, original_save


class SelectedReplayMath:
    """Delegate math unchanged except one qualified cross-GPU final gate.

    This is a module-local proxy, never a patch to Python's global math module.
    Matching values alone cannot activate the exception: the exact train_one
    frame, selected tensor SHA, source epoch/value and qualified replay must match.
    """
    def __init__(self, original, training, qualification, source_payload, events):
        self.original, self.training, self.q, self.source, self.events = original, training, qualification, source_payload, events

    def __getattr__(self, name):
        return getattr(self.original, name)

    def isclose(self, a, b, *, rel_tol=1e-9, abs_tol=0.):
        ordinary = self.original.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        if ordinary:
            return True
        frame = inspect.currentframe().f_back
        if frame.f_code is not self.training.train_one.__code__ or rel_tol != 1e-10 or abs_tol != 1e-8:
            return False
        values = frame.f_locals
        require(self.q["status"] == "passed" and self.q["selected"]["state_sha256"] == self.source["best_state_sha256"],
            "Qualified source selected checkpoint changed")
        if not (values.get("state_digest") == self.source["best_state_sha256"]
                and values.get("best_epoch") == self.source["best_epoch"]
                and values.get("selected_metric_value") == self.source["best_selection_value"]
                and a == values.get("reevaluated_metric_value") and b == values.get("selected_metric_value")):
            return False
        if not (self.original.isclose(a, b, rel_tol=TOLERANCE["rtol"], abs_tol=TOLERANCE["atol"])
                and self.original.isclose(a, self.q["selected"]["qty_rmse"], rel_tol=TOLERANCE["rtol"], abs_tol=TOLERANCE["atol"])):
            return False
        self.events.append({"epoch": values["best_epoch"], "state_sha256": values["state_digest"],
            "source_history_metric": b, "target_replayed_metric": a, "tolerance": TOLERANCE,
            "source_history_overwritten": False, "checkpoint_reselected": False})
        return True


@contextmanager
def selected_replay_gate(training, qualification, payload, events):
    original = training.math
    training.math = SelectedReplayMath(original, training, qualification, payload, events)
    try:
        yield
    finally:
        training.math = original


def prepare_context(c, parent, root, job, engine, training):
    payload, prefix, timing, inspected = inspect_original(c, parent, root, job, training)
    data = mapped_dataset(c, parent, job)
    frame, input_receipt = engine.base.prepare_admitted_data(data)
    # Rebuild the interface with the frozen parent contract, never the operational contract.
    interface = engine.time_interface(data, frame, parent, arm=job["arm"])
    require(interface == payload["interface_meta"], "Recalculated frozen interface changed")
    args = engine.training_args(parent, data, root / "run" / job["id"], job["arm"])
    identity = canonicalize_identity(training._resume_identity(args=args, backbone=job["arm"], quantity_variant=VARIANT,
        seed=42, monitor="validation_raw_quantity_rmse", quantity_contract=payload["resume_identity"]["quantity_contract"],
        interface_meta=interface), c["path_mappings"][job["id"]])
    require(identity == payload["resume_identity"], "Runtime training identity is not the recovered original")
    return payload, prefix, timing, inspected, data, frame, interface, args, input_receipt


def qualify(c, parent, root, job, host, engine, training):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    start = authorize(c, root, host)
    runtime = runtime_check(c, host)
    dest = root / "qualification" / job["id"]
    dest.mkdir(parents=True, exist_ok=False)
    budget = Budget(root, c, job, min(start["deadline_unix"], time.time() + c["limits"]["per_condition_seconds"]))
    payload, prefix, timing, inspected, data, frame, interface, args, inputs = prepare_context(c, parent, root, job, engine, training)
    native_initial = engine.initial_states(data, 42)[job["arm"]]
    require(native_initial == payload["initial_state_sha256"], "Target native initial state differs from checkpoint")
    # Synthetic checks reuse the byte-identical parent code, without its A100-only dispatcher.
    worker = Path(c["hosts"][host]["worker_root"])
    original_runtime = sys.modules.get("runtime")
    spec_r = importlib.util.spec_from_file_location("runtime", worker / "runtime.py")
    module_r = importlib.util.module_from_spec(spec_r)
    sys.modules["runtime"] = module_r
    try:
        spec_r.loader.exec_module(module_r)
        spec_q = importlib.util.spec_from_file_location("_titantpp_native_resume_qualification", worker / "qualify.py")
        module_q = importlib.util.module_from_spec(spec_q)
        spec_q.loader.exec_module(module_q)
    finally:
        if original_runtime is None:
            sys.modules.pop("runtime", None)
        else:
            sys.modules["runtime"] = original_runtime
    repeats = [module_q.synthetic_repeat(engine, module_q.synthetic_data(data["loader"]["max_seq_len"]),
        job["arm"], budget) for _ in range(2)]
    require(repeats[0]["fingerprint"] == repeats[1]["fingerprint"], "Native repeated execution differs")
    total = torch.cuda.get_device_properties(0).total_memory
    require(all(x["peak_reserved_bytes"] <= total * c["limits"]["max_device_memory_fraction"] for x in repeats),
        "Native memory ceiling failed")
    causality = module_q.causality_check(engine, job["arm"], budget)
    resume_probe = synthetic_resume_check(engine, module_q.synthetic_data(data["loader"]["max_seq_len"]),
        job["arm"], budget)
    last = engine._replay_validation(payload, Path(c["artifacts"][job["id"]]["origin_root"]) / "last_epoch_state.pt",
        data, frame, budget)
    selected_payload = {**payload, "model_state_dict": payload["best_state_dict"],
        "model_state_sha256": payload["best_state_sha256"]}
    selected = engine._replay_validation(selected_payload, "embedded_immutable_source_best_state", data, frame, budget)
    checks = {"selected": compare_replay(selected, payload["history"][payload["best_epoch"] - 1]),
        "last": compare_replay(last, payload["history"][-1])}
    require(all(ok for row in checks.values() for ok in row.values()), "Same-checkpoint full Validation replay failed")
    q = {"status": "passed", "contract_sha256": canonical(c), "host": host, "job": job,
        "source_files_sha256": SOURCE_SHA, "worker_files_sha256": WORKER_SHA, "runtime": runtime,
        "source_checkpoint_sha256": inspected["source_checkpoint_sha256"], "source_epoch": payload["epoch"],
        "source_best_epoch": payload["best_epoch"], "source_resume_identity_sha256": canonical(payload["resume_identity"]),
        "native_initial_state_sha256": native_initial, "repetitions": repeats, "causality": causality,
        "model_optimizer_rng_loader_resume": resume_probe,
        "input_receipt": inputs, "interface_sha256": canonical(interface), "selected": selected, "last": last,
        "full_validation_replay": True, "checks": checks, "tolerance": TOLERANCE,
        "source_history_overwritten": False, "new_research_training": False, "held_out_test_evaluated": False,
        "completed_unix": time.time()}
    require(canonical_state_dict_sha256(payload["best_state_dict"]) == selected["state_sha256"], "Qualified selected SHA differs")
    write(dest / "inspection.json", inspected, exclusive=True)
    write(dest / "receipt.json", q, exclusive=True)
    return q


def verify_training_permit(c, root, host, job):
    start = authorize(c, root, host)
    permit = read(root / ("training_permit_" + host + ".json"))
    expected = {j["id"] for j in c["jobs"] if j["host"] == host}
    require(permit["contract_sha256"] == canonical(c) and permit["host"] == host
        and permit["start_permit_sha256"] == canonical(start) and permit["manual_resume_once"] is True
        and permit["automatic_retry"] is False and set(permit["job_ids"]) == expected
        and set(permit["qualification_files"]) == expected, "Foreign/incomplete training permit")
    path = root / "qualification" / job["id"] / "receipt.json"
    require(sha(path) == permit["qualification_files"][job["id"]], "Qualified native receipt changed")
    q = read(path)
    require(q["status"] == "passed" and q["contract_sha256"] == canonical(c) and q["host"] == host
        and q["job"] == job and q["source_files_sha256"] == SOURCE_SHA and q["worker_files_sha256"] == WORKER_SHA
        and q["source_checkpoint_sha256"] == c["artifacts"][job["id"]]["source_checkpoint_sha256"]
        and q["source_epoch"] == c["artifacts"][job["id"]]["source_epoch"]
        and q["source_best_epoch"] == c["artifacts"][job["id"]]["source_best_epoch"]
        and q["source_resume_identity_sha256"] == c["artifacts"][job["id"]]["source_resume_identity_sha256"]
        and q["selected"]["state_sha256"] == c["artifacts"][job["id"]]["source_best_state_sha256"]
        and q["last"]["state_sha256"] == c["artifacts"][job["id"]]["source_current_state_sha256"]
        and q["full_validation_replay"] is True and q["held_out_test_evaluated"] is False
        and q["tolerance"] == TOLERANCE and set(q["checks"]) == {"selected", "last"}
        and all(set(row) == set(METRICS) and all(row.values()) for row in q["checks"].values())
        and q["model_optimizer_rng_loader_resume"]["status"] == "passed"
        and q["model_optimizer_rng_loader_resume"]["uninterrupted_state_sha256"]
            == q["model_optimizer_rng_loader_resume"]["resumed_state_sha256"]
        and q["causality"]["status"] == "passed" and len(q["repetitions"]) == 2
        and q["repetitions"][0]["fingerprint"] == q["repetitions"][1]["fingerprint"],
        "Native qualification does not verify this immutable checkpoint")
    return start, q


def exclusive_claim(c, root, host, job, deadline):
    """An interrupted/failed claim is preserved and cannot be automatically retried."""
    value = {"job": job, "host": host, "contract_sha256": canonical(c), "source_checkpoint_sha256":
        c["artifacts"][job["id"]]["source_checkpoint_sha256"], "deadline_unix": deadline,
        "manual_resume_once": True, "automatic_retry": False, "created_unix": time.time(), "pid": os.getpid()}
    write(root / "claims" / (job["id"] + ".json"), value, exclusive=True)
    return value


def fit(c, parent, root, job, host, engine, training):
    import fcntl
    start, q = verify_training_permit(c, root, host, job)
    require(runtime_check(c, host) == q["runtime"], "Qualified native runtime changed")
    with (root / "host_worker.lock").open("a") as host_lock:
        fcntl.flock(host_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = min(start["deadline_unix"], time.time() + c["limits"]["per_condition_seconds"])
        exclusive_claim(c, root, host, job, deadline)
        budget = Budget(root, c, job, deadline)
        folder = root / "run" / job["id"]
        folder.mkdir(parents=True, exist_ok=False)
        run = folder / "runs" / job["arm"] / VARIANT / "seed_42"
        run.mkdir(parents=True, exist_ok=False)
        try:
            payload, prefix, times, inspected, data, frame, interface, args, inputs = prepare_context(c, parent, root, job, engine, training)
            origin = Path(c["artifacts"][job["id"]]["origin_root"])
            for name in REQUIRED_ARTIFACTS:
                destination = run / name
                with (origin / name).open("rb") as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                require(sha(destination) == c["artifacts"][job["id"]]["files"][name], "Working checkpoint copy changed")
            # Copy immutable originals first; only administrative working records are normalized to the checkpoint prefix.
            write(run / "exposure.json", prefix)
            write(run / "epoch_timing.json", {"scope": "train_iteration_through_validation_and_checkpoint_save", "epochs": times})
            write(folder / "input_receipt.json", inputs, exclusive=True)
            write(folder / "source_continuation.json", {**inspected, "contract_sha256": canonical(c), "host": host,
                "segment_boundary_epoch": payload["epoch"], "source_host": "a100", "target_host": host,
                "GPU_changed": True, "continuation_numerically_identical_to_A100_claim": False}, exclusive=True)

            def status(epoch, records):
                budget()
                write(run / "exposure.json", records)
                names = ("last_epoch_state.pt", "history.json", "exposure.json", "epoch_timing.json", "train.log")
                write(run / "server_checkpoint_receipt.json", {"epoch": epoch, "contract_sha256": canonical(c),
                    "files": {name: sha(run / name) for name in names}, "time": time.time(), "mac_ack_required": False})
                write(folder / "status.json", {"status": "training", "epoch": epoch, "job": job,
                    "global_steps": sum(x["batches"] for x in records["train"]), "deadline_unix": deadline,
                    "source_epoch": payload["epoch"], "updated_unix": time.time()})

            training._titantpp_resume_source_history = deepcopy(payload["history"])
            events = []
            with identity_mapping(training, c["path_mappings"][job["id"]], payload["resume_identity"]), strict_seed(training), \
                    audited_resume(training, engine.shared, data, budget, status, prefix, times, payload["epoch"]) as exposure, \
                    selected_replay_gate(training, q, payload, events):
                summary, _, _ = training.train_one(args=args, frame=frame,
                    quantity_contract=payload["resume_identity"]["quantity_contract"], interface_meta=interface,
                    backbone=job["arm"], quantity_variant=VARIANT, seed=42)
            history = read(run / "history.json")["history"]
            from paper.scripts.run_local_detail_benchmark import audit_arm
            steps = audit_arm(history, summary, exposure, data, parent["training"])
            require(history[:payload["epoch"]] == payload["history"], "Source history prefix changed after fit")
            endpoints, endpoint_checks = {}, {}
            for label, path in (("selected", Path(summary["checkpoint_path"])), ("last", run / "last_epoch_state.pt")):
                epoch = summary["best_epoch"] if label == "selected" else len(history)
                endpoints[label] = engine.replay_checkpoint(path, data, frame, budget, expected_arm=job["arm"],
                    expected_identity=payload["resume_identity"], expected_initial=payload["initial_state_sha256"], expected_epoch=epoch)
                tolerance = TOLERANCE if epoch <= payload["epoch"] else {"rtol": 1e-10, "atol": 1e-8}
                endpoint_checks[label] = compare_replay(endpoints[label], history[epoch - 1], tolerance=tolerance)
                require(all(endpoint_checks[label].values()), "Final selected/last full Validation replay differs")
            result = {**endpoints, "status": "complete", "job": job, "best_epoch": summary["best_epoch"],
                "completed_epochs": len(history), "global_steps": steps, "source_epoch": payload["epoch"],
                "source_global_steps": inspected["source_global_steps"], "new_epochs": len(history) - payload["epoch"],
                "new_optimizer_steps": steps - inspected["source_global_steps"], "replay_checks": endpoint_checks,
                "cross_GPU_selected_replay_gate_events": events, "selected_history_metrics": history[summary["best_epoch"] - 1],
                "parameter_count": summary["parameter_count"], "initial_state_sha256": payload["initial_state_sha256"],
                "GPU_segments": gpu_segments(payload["epoch"], len(history), host),
                "target_host_evaluation_only": len(history) == payload["epoch"],
                "source_history_overwritten": False, "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
            write(run / "endpoint_replays.json", result, exclusive=True)
            write(run / "exposure.json", exposure)
            budget()
            write(folder / "status.json", {"status": "complete", "job": job, "updated_unix": time.time()})
            files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}
            write(folder / "terminal_manifest.json", {"status": "complete", "scientific_success": True, "job": job,
                "files": files, "contract_sha256": canonical(c), "origin_contract_sha256": PARENT_SHA,
                "held_out_test_evaluated": False, "completed_unix": time.time()}, exclusive=True)
            return result
        except BaseException as error:
            write(folder / "failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                "job": job, "automatic_retry": False, "source_originals_preserved": True, "updated_unix": time.time()})
            raise
        finally:
            if hasattr(training, "_titantpp_resume_source_history"):
                del training._titantpp_resume_source_history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("inspect", "qualify", "fit"), required=True)
    parser.add_argument("--job", required=True)
    options = parser.parse_args(argv)
    c, parent, root, job = load(options.contract, options.host, options.job)
    authorize(c, root, options.host)
    engine, training = load_engine(c, options.host)
    if options.mode == "inspect":
        result = inspect_original(c, parent, root, job, training)[-1]
    elif options.mode == "qualify":
        result = qualify(c, parent, root, job, options.host, engine, training)
    else:
        result = fit(c, parent, root, job, options.host, engine, training)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
