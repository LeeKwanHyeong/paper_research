"""Isolated RAF/Taxi seed-42 PHEN pilot with fixed sinusoidal positions.

The frozen full-PHEN factory, loader, joint objective and stopping rule are
reused.  Only the position parameter becomes a deterministic persistent buffer.
Full-population end-of-epoch Train evaluation is observational: its RNG use is
restored before training resumes.  No held-out Test path is provided.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from paper.scripts import run_titantpp_mlp_cap36 as common

SCHEMA = "phen_fixed_position_pilot_v1"
DATASETS = ("raf_spare_parts", "yellow_trip_hourly")
HOST, SEED = "5080", 42
VARIANT, MONITOR = "count_only_log_regression", "validation_raw_quantity_rmse"
POSITION_POLICY = {"kind": "fixed_sinusoidal", "base": 10000.0, "index_origin": 0,
                   "order": "oldest_to_newest_in_observed_window", "trainable": False,
                   "amplitude": 1.0}
IDENTITY_KEY = "fixed_position_pilot_identity"
require, read, sha, canonical, write = (common.require, common.read, common.sha_file,
                                       common.sha_json, common.write_exclusive)


def now():
    return datetime.now(timezone.utc).isoformat()


def job_id(dataset):
    require(dataset in DATASETS, "Only RAF and Taxi are admitted")
    return dataset + "__fixed_sinusoidal__seed42"


def below(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and bool(path.parts), "Unsafe relative path")
    result = Path(root) / path
    require(not any(p.is_symlink() for p in (result, *result.parents)), "Symlink not admitted")
    return common.resolve_under(root, result)


def validate(contract):
    require(contract.get("schema") == SCHEMA, "Wrong experiment schema")
    require(contract.get("seeds") == [SEED] and set(contract.get("hosts", {})) == {HOST}, "Wrong seed/host scope")
    require(contract.get("model_role") == "observed_time_joint_v1"
            and contract.get("evaluation_scope") == "validation_only", "Wrong model/evaluation scope")
    require(contract.get("position_encoding") == POSITION_POLICY, "Position policy changed")
    require(contract.get("jobs") == [{"id": job_id(d), "dataset": d, "seed": SEED, "host": HOST}
                                    for d in DATASETS], "Expected RAF then Taxi, once each")
    datasets = contract.get("datasets", [])
    require([d.get("dataset_id") for d in datasets] == list(DATASETS), "Dataset sequence changed")
    for key, expected in {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
                          "batch_size": 128, "monitor": MONITOR}.items():
        require(contract["training"].get(key) == expected, "Training condition changed: " + key)
    require(not contract["training"].get("warm_start", False), "Warm start forbidden")
    source = contract["source"]
    require(source.get("files") and source.get("files_sha256") == canonical(source["files"]), "Invalid source inventory")
    require(isinstance(source.get("base_git_revision"), str) and len(source["base_git_revision"]) == 40,
            "Missing frozen source revision")
    require(contract.get("input_files"), "Missing input inventory")
    for mapping in (source["files"], contract["input_files"]):
        for path, digest in mapping.items():
            require(not Path(path).is_absolute() and ".." not in Path(path).parts, "Unsafe inventory path")
            require(isinstance(digest, str) and len(digest) == 64
                    and set(digest) <= set("0123456789abcdef"), "Invalid SHA256")
    for data in datasets:
        require(data["loader"]["batch_size"] == 128 and 2 <= data["loader"]["max_seq_len"] <= 256,
                "Loader shape is outside the approved model range")
        model = data["model"]
        require(model["hidden_dim"] == 64 and model["quantity_variant"] == VARIANT
                and model["lambda_log_qty"] == 1.0 and model["lambda_tail"] == 0.0
                and model["time_head_mode"] == "heteroscedastic_lognormal_duration", "PHEN heads/loss changed")
        require(set(data["inherited_data_identity"]["populations"]) == {"train", "validation"}, "Test population forbidden")
        for key in ("data", "split_manifest"):
            entry = data["inherited_data_identity"][key]
            require(contract["input_files"].get(entry["path"]) == entry["sha256"], "Input record not sealed")
    spec = contract["hosts"][HOST]
    require(Path(spec["root"]).is_absolute() and Path(spec["python"]).is_absolute(), "Absolute host paths required")
    require(spec.get("boot_id") and spec.get("gpu_uuid") and isinstance(spec.get("paused_processes"), list),
            "Native identity/protected process list required")


def verify_internal_modules(source):
    source = Path(source).resolve()
    prefixes = {"paper", "models", "utils", "data_loader", "simple_lab_test", "benchmark_data"}
    for name, module in list(sys.modules.items()):
        if name.split(".", 1)[0] not in prefixes or module is None:
            continue
        filename = getattr(module, "__file__", None)
        for path in ([filename] if filename else []) + list(getattr(module, "__path__", [])):
            require(Path(path).resolve().is_relative_to(source), "Internal import escaped frozen source: " + name)


def verify_files(contract, dataset=None):
    validate(contract)
    root = Path(contract["hosts"][HOST]["root"]).resolve()
    source = root / "source"
    require(Path(__file__).resolve().is_relative_to(source), "Runner is not in sealed source")
    require(all((source / name).is_dir() for name in ("models", "utils", "sample_data")), "Root sentinels missing")
    verify_internal_modules(source)
    from simple_lab_test.common.pathing import resolve_project_root
    require(resolve_project_root(__file__) == source, "Project bootstrap escaped source")
    for path, digest in {**contract["source"]["files"], **contract["input_files"]}.items():
        require(sha(below(source, path)) == digest, "Frozen bytes changed: " + path)
    verify_internal_modules(source)
    data = None
    if dataset is not None:
        require(dataset in DATASETS, "Unapproved dataset")
        data = deepcopy(next(d for d in contract["datasets"] if d["dataset_id"] == dataset))
        for key in ("data", "split_manifest"):
            data["inherited_data_identity"][key]["path"] = str(below(source, data["inherited_data_identity"][key]["path"]))
    return data, root


def process_identity(pid):
    proc = Path("/proc") / str(pid)
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "state": fields[0], "pgid": int(fields[2]), "start_ticks": int(fields[19]),
            "argv": (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
            "cwd": str((proc / "cwd").resolve(strict=True))}


def gpu_guard(contract):
    spec = contract["hosts"][HOST]
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(boot == spec["boot_id"], "Host boot changed")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], timeout=15, text=True).splitlines()
    require(gpu == [spec["gpu_uuid"]], "GPU identity changed")
    retained = []
    for expected in spec["paused_processes"]:
        actual = process_identity(expected["pid"])
        require(actual["state"] == "T" and all(actual[k] == expected[k] for k in
                ("pid", "pgid", "start_ticks", "argv", "cwd")), "Protected process identity/state changed")
        retained.append(actual)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], timeout=15, text=True)
    active = {int(line.strip()) for line in apps.splitlines() if line.strip()}
    require(active <= {p["pid"] for p in retained}, "Unexpected GPU process; refusing overlap")
    return {"observed_at": now(), "boot_id": boot, "gpu_uuid": gpu[0],
            "retained_paused_processes": retained, "gpu_processes": sorted(active)}


def runtime(device, source):
    import torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    verify_internal_modules(source)
    configure_runtime(device, 4)
    require(device == "cpu" or (device == "cuda:0" and torch.cuda.is_available()), "Unsupported device")
    return {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None}


def sinusoidal_table(length, width, *, dtype=None, device=None):
    """Compute the standard unscaled sine/cosine table without consuming RNG."""
    import torch
    require(type(length) is int and length > 0 and type(width) is int and width > 0 and width % 2 == 0,
            "Positive length and even width required")
    positions = torch.arange(length, dtype=torch.float64, device="cpu").unsqueeze(1)
    scales = torch.exp(torch.arange(0, width, 2, dtype=torch.float64) * (-math.log(10000.0) / width))
    phase = positions * scales
    table = torch.empty((1, length, width), dtype=torch.float64)
    table[0, :, 0::2] = torch.sin(phase)
    table[0, :, 1::2] = torch.cos(phase)
    return table.to(dtype=dtype or torch.float32, device=device or "cpu")


def tensor_mapping_sha(state):
    """Canonical tensor hash shared with the frozen checkpoint format."""
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    return canonical_state_dict_sha256(state)


def apply_fixed_positions(model):
    import torch
    position = model.encoder.pos_emb
    require(isinstance(position, torch.nn.Parameter) and position.ndim == 3
            and position.shape[0] == 1 and model.encoder.use_pos_emb, "Expected original learned position parameter")
    nonposition = {k: v for k, v in model.state_dict().items() if k != "encoder.pos_emb"}
    before = tensor_mapping_sha(nonposition)
    policy_hash = hashlib.sha256(json.dumps(POSITION_POLICY, sort_keys=True).encode()).digest()
    fixed = sinusoidal_table(position.shape[1], position.shape[2], dtype=position.dtype, device=position.device)
    removed = position.numel()
    delattr(model.encoder, "pos_emb")
    model.encoder.register_buffer("pos_emb", fixed, persistent=True)
    model.register_buffer(IDENTITY_KEY, torch.tensor(list(policy_hash), dtype=torch.uint8, device=fixed.device), persistent=True)
    after = tensor_mapping_sha({k: v for k, v in model.state_dict().items() if k not in ("encoder.pos_emb", IDENTITY_KEY)})
    require(before == after, "Replacing position encoding changed another initialized tensor")
    require("encoder.pos_emb" not in dict(model.named_parameters()) and not model.encoder.pos_emb.requires_grad,
            "Position encoding remains trainable")
    return {"position_encoding": POSITION_POLICY, "position_shape": list(fixed.shape),
            "removed_trainable_parameters": removed, "nonposition_initial_state_sha256": before,
            "fixed_position_sha256": tensor_mapping_sha({"encoder.pos_emb": fixed})}


def build_fixed_model(data):
    from paper.scripts import run_shared_component_ablation as shared
    model, encoder = shared._build_model(data, "full")
    change = apply_fixed_positions(model)
    return model, {**encoder, "position_encoding": dict(POSITION_POLICY)}, change


@contextmanager
def preserve_evaluation_state(model, *, generators=()):
    """Restore all global RNGs, generator states and each module's mode."""
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator_states = [(g, g.get_state().clone()) for g in generators if g is not None]
    modes = [(module, module.training) for module in model.modules()]
    before = tensor_mapping_sha(model.state_dict())
    try:
        model.eval()
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for generator, state in generator_states:
            generator.set_state(state)
        for module, mode in modes:
            module.training = mode
        require(before == tensor_mapping_sha(model.state_dict()), "Monitoring modified model state")


def plain_metrics(result):
    return {"count": int(result["evaluated_count"]), "qty_rmse": float(result["qty_rmse"]),
            "qty_mae": float(result["qty_mae"]), "log_qty_mse": float(result["val_log_qty_mse"]),
            "time_nll": float(result["val_time_nll"]), "joint_objective": float(result["val_joint_objective"])}


@contextmanager
def fixed_training_hooks(data, frame, folder, *, monitor=True):
    """Wrap frozen training hooks without editing or replacing its algorithm."""
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    import torch
    with shared._engine_hooks(data, "full") as tracked:
        old_build, old_epoch, old_evaluate, old_save = (training.build_model, training.train_epoch_with_telemetry,
                                                       training.evaluate, training.atomic_torch_save)
        state = {"epoch": 0}
        curve_dir = Path(folder) / "curve_epochs"
        curve_dir.mkdir(exist_ok=False)
        stream = (Path(folder) / "learning_curves.jsonl").open("x", encoding="utf-8")
        def build(*args, **kwargs):
            model, encoder = old_build(*args, **kwargs)
            state["position_change"] = apply_fixed_positions(model)
            return model, {**encoder, "position_encoding": dict(POSITION_POLICY)}
        def epoch(**kwargs):
            state["epoch"] += 1
            state["train_loader"] = kwargs["loader"]
            started = time.monotonic()
            result = old_epoch(**kwargs)
            state["train_online"] = result
            state["train_online_seconds"] = time.monotonic() - started
            return result
        def evaluate(**kwargs):
            started = time.monotonic()
            validation = old_evaluate(**kwargs)
            validation_seconds = time.monotonic() - started
            if monitor and kwargs["include_breakdowns"] is False:
                require(state["epoch"] > 0 and kwargs["max_batches"] is None, "Expected full epoch Validation")
                model = kwargs["model"]
                train_generator = getattr(state["train_loader"], "generator", None)
                started = time.monotonic()
                tracker = tracked["tracker"]
                previous_observations = tracker.joint_observations
                try:
                    with preserve_evaluation_state(model, generators=(train_generator,)):
                        # A private generator avoids consuming the training sampler's
                        # state.  Constructor and iterator are both inside the guard.
                        train_eval_loader = make_loader(frame, target_split="train", shuffle=False,
                            generator=torch.Generator().manual_seed(SEED),
                            **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")})
                        train_result = old_evaluate(**{**kwargs, "loader": train_eval_loader, "include_breakdowns": False})
                    monitoring_observations = tracker.joint_observations - previous_observations
                finally:
                    # Extra diagnostics must not inflate the native fit proof.
                    tracker.joint_observations = previous_observations
                row = {"schema": SCHEMA, "dataset_id": data["dataset_id"], "seed": SEED,
                       "epoch": state["epoch"], "observed_at": now(),
                       "model_state_sha256": tensor_mapping_sha(model.state_dict()),
                       "train_online": state["train_online"], "train_eval": plain_metrics(train_result),
                       "validation": plain_metrics(validation), "train_eval_seconds": time.monotonic() - started,
                       "validation_seconds": validation_seconds, "train_online_seconds": state["train_online_seconds"],
                       "monitoring_joint_loss_observations": monitoring_observations,
                       "evaluation_mode": "eval", "monitoring_rng_preserved": True, "model_state_unchanged": True}
                for split in ("train", "validation"):
                    key = "train_eval" if split == "train" else "validation"
                    expected = data["inherited_data_identity"]["populations"][split]["target_count"]
                    require(row[key]["count"] == expected, "Incomplete end-of-epoch population: " + split)
                write(curve_dir / f"epoch_{state['epoch']:04d}.json", row)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return validation
        def save(payload, path):
            return old_save({**payload, "position_encoding": dict(POSITION_POLICY),
                             "position_initialization": state["position_change"],
                             "pilot_schema": SCHEMA}, path)
        training.build_model, training.train_epoch_with_telemetry = build, epoch
        training.evaluate, training.atomic_torch_save = evaluate, save
        try:
            yield tracked, state
        finally:
            training.build_model, training.train_epoch_with_telemetry = old_build, old_epoch
            training.evaluate, training.atomic_torch_save = old_evaluate, old_save
            stream.close()


def qualify(contract, dataset, mode, device):
    data, root = verify_files(contract, dataset)
    import torch
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts.count_aware_tpp_backbone import core
    folder = root / "qualification" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / (mode + "_attempt.json"), {"contract_sha256": canonical(contract), "started_at": now()})
    guard = gpu_guard(contract) if device.startswith("cuda") else None
    native = runtime(device, root / "source")
    torch.manual_seed(SEED)
    model, encoder, initialization = build_fixed_model(data)
    model = model.to(device).train()
    tracker = shared.raw9.JointTimeHeadTracker(model)
    optimizer = tracker.bind_optimizer(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01))
    shape = (128, data["loader"]["max_seq_len"]) if mode == "capacity" else (3, min(8, data["loader"]["max_seq_len"]))
    dt = torch.ones(shape, device=device)
    quantities = torch.arange(shape[1], device=device).float().add(1).expand(shape).clone()
    mask = torch.ones(shape, device=device, dtype=torch.bool)
    fixed_before = model.encoder.pos_emb.detach().clone()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        value = core.target_outputs(model, dt, mask, quantities, lambda_log_qty=1.)
        require(torch.equal(value["joint_loss"], value["time_loss"] + value["log_qty_loss"]), "Joint objective changed")
        loss = value["joint_loss"].mean()
        require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
        tracker.joint_observations += 1
        loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "Nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    require(torch.equal(fixed_before, model.encoder.pos_emb), "Optimizer changed fixed positions")
    proof = tracker.proof()
    model.eval()
    digest = tensor_mapping_sha(model.state_dict())
    with torch.no_grad():
        original = core.target_outputs(model, dt, mask, quantities, lambda_log_qty=1.)
        for change_time in (False, True):
            changed_dt, changed_q = dt.clone(), quantities.clone()
            if change_time:
                changed_dt[:, -1] += 100
            else:
                changed_q[:, -1] += 100
            altered = core.target_outputs(model, changed_dt, mask, changed_q, lambda_log_qty=1.)
            torch.testing.assert_close(original["pred_qty"], altered["pred_qty"], rtol=1e-5, atol=1e-5)
        binary = io.BytesIO()
        torch.save(model.state_dict(), binary)
        binary.seek(0)
        restored, second_encoder, _ = build_fixed_model(data)
        require(encoder == second_encoder, "Encoder metadata changed")
        restored.load_state_dict(torch.load(binary, map_location="cpu", weights_only=True), strict=True)
        restored = restored.to(device).eval()
        replay = core.target_outputs(restored, dt, mask, quantities, lambda_log_qty=1.)
        for key in ("pred_qty", "time_loss"):
            torch.testing.assert_close(original[key], replay[key], rtol=1e-5, atol=1e-5)
    require(digest == tensor_mapping_sha(model.state_dict()), "Qualification evaluation modified state")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result = {"schema": SCHEMA, "status": "passed", "mode": mode, "host": HOST, "dataset_id": dataset,
              "contract_sha256": canonical(contract), "source_files_sha256": contract["source"]["files_sha256"],
              "runtime": native, "gpu_guard": guard, "shape": list(shape), "position_initialization": initialization,
              "real_data_loaded": False, "held_out_test_evaluated": False, "parameter_count": sum(p.numel() for p in model.parameters()),
              "time_head_training_verification": proof, "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else None,
              "checks": ["finite_forward_backward", "fixed_position_unchanged", "target_quantity_hidden", "target_time_hidden",
                         "checkpoint_roundtrip", "evaluation_state_unchanged", "joint_time_gradient"], "completed_at": now()}
    optimizer.step = tracker.original_step
    verify_internal_modules(root / "source")
    write(folder / (mode + ".json"), result)
    return result


def verify_qualification(contract, dataset, root):
    data = next(d for d in contract["datasets"] if d["dataset_id"] == dataset)
    result = {}
    for mode, shape in (("synthetic", [3, min(8, data["loader"]["max_seq_len"])]),
                        ("capacity", [128, data["loader"]["max_seq_len"]])):
        row = read(root / "qualification" / dataset / (mode + ".json"))
        require(row["status"] == "passed" and row["host"] == HOST and row["dataset_id"] == dataset
                and row["contract_sha256"] == canonical(contract) and row["source_files_sha256"] == contract["source"]["files_sha256"]
                and row["shape"] == shape and row["runtime"]["device"] == "cuda:0", "Missing matching native qualification")
        result[mode] = row
    require(result["synthetic"]["runtime"] == result["capacity"]["runtime"], "Qualification runtime drift")
    return result


def load_selected(contract, data, folder, summary, proof, device):
    from paper.scripts import run_shared_component_ablation as shared
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    checkpoint = common.resolve_under(folder, summary["checkpoint_path"])
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    require(payload.get("pilot_schema") == SCHEMA and payload.get("position_encoding") == POSITION_POLICY,
            "Checkpoint is not a fixed-position pilot")
    require(payload.get("backbone") == "full" and payload.get("seed") == SEED and payload.get("variant") == VARIANT
            and payload.get("checkpoint_monitor") == MONITOR and payload.get("best_epoch") == summary["best_epoch"]
            and payload.get("evaluation_scope") == "validation_only" and payload.get("held_out_test_evaluated") is False,
            "Selected checkpoint provenance mismatch")
    digest = tensor_mapping_sha(payload["model_state_dict"])
    require(digest == payload["model_state_sha256"] == summary["checkpoint_state_sha256"], "Selected state digest mismatch")
    require(payload["interface_meta"]["source_files_sha256"] == contract["source"]["files_sha256"], "Checkpoint source mismatch")
    history = read(checkpoint.parent / "history.json")["history"]
    require(shared.raw9.prepared.select_best_epoch(history) == summary["best_epoch"], "Not the first strict Validation minimum")
    curve = read(Path(folder) / "curve_epochs" / f"epoch_{summary['best_epoch']:04d}.json")
    require(curve["model_state_sha256"] == digest, "Selected checkpoint differs from its end-of-epoch monitoring state")
    for metric, key in (("qty_rmse", "best_val_qty_rmse"), ("qty_mae", "best_val_qty_mae"),
                        ("time_nll", "best_val_time_nll")):
        require(math.isclose(curve["validation"][metric], summary[key], rel_tol=1e-5, abs_tol=1e-5),
                "Selected checkpoint/curve metric mismatch: " + metric)
    shared.raw9.verify_time_head_proof(payload["time_head_training_verification"], payload["model_state_dict"])
    shared.raw9.verify_checkpoint_fit_proof(payload["time_head_training_verification"], proof)
    model, encoder, _ = build_fixed_model(data)
    require(payload["encoder_config"] == encoder, "Encoder identity mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    expected = sinusoidal_table(model.encoder.pos_emb.shape[1], model.encoder.pos_emb.shape[2], dtype=model.encoder.pos_emb.dtype)
    import torch
    require(torch.equal(model.encoder.pos_emb.cpu(), expected), "Selected fixed position table changed")
    return model.to(device).eval().requires_grad_(False), checkpoint, digest


def evaluate_validation(data, frame, model, output, device):
    import numpy as np
    import torch
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    before = tensor_mapping_sha(model.state_dict())
    loader = make_loader(frame, target_split="validation", shuffle=False, generator=None,
                         **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")})
    truth, identity = common.population_identity(loader.dataset, "validation")
    common.verify_population(identity, data["inherited_data_identity"]["populations"]["validation"])
    accumulator, processed = shared.raw9.MetricAccumulator(data["quantity_boundaries_all_train_rows"][-1]), 0
    with Path(output).open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with torch.no_grad():
            for _, dts, mask, part_ids, quantities in loader:
                value = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
                arrays = {k: value[k].cpu().numpy() for k in ("true_qty", "pred_qty", "time_loss", "history_length")}
                size = len(arrays["true_qty"])
                require(np.array_equal(arrays["true_qty"].astype(np.float64), truth[processed:processed + size]), "Validation target changed")
                accumulator.add(arrays["true_qty"], arrays["pred_qty"], arrays["time_loss"])
                for i in range(size):
                    part, end = loader.dataset.index[processed + i]
                    require(int(part_ids[i]) == int(part), "Validation order changed")
                    row = {"target_index": processed + i, "entity_id": str(loader.dataset.parts[part]),
                           "target_seq": int(loader.dataset.seq_lists[part][end + 1]), "split": "validation",
                           "raw_quantity": float(arrays["true_qty"][i]), "predicted_raw_quantity": float(arrays["pred_qty"][i]),
                           "time_nll": float(arrays["time_loss"][i]), "history_length": int(arrays["history_length"][i])}
                    zipped.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
                processed += size
    require(processed == identity["target_count"] and before == tensor_mapping_sha(model.state_dict()), "Incomplete/mutating Validation")
    return {"metrics": accumulator.finish(), "target_identity": identity, "prediction_rows": processed,
            "prediction_sha256": sha(output), "state_sha256": before, "split": "validation"}


def train(contract, dataset, device):
    require(device == "cuda:0", "Real training requires admitted CUDA device")
    data, root = verify_files(contract, dataset)
    import torch
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.count_aware_tpp_backbone import training
    qualification = verify_qualification(contract, dataset, root)
    guard = gpu_guard(contract)
    folder = root / "jobs" / job_id(dataset)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(folder / "attempt.json", {"dataset_id": dataset, "seed": SEED, "host": HOST, "started_at": now(),
                                   "contract_sha256": canonical(contract), "gpu_guard": guard})
    native = runtime(device, root / "source")
    require(native == qualification["synthetic"]["runtime"], "Training runtime changed")
    frame, input_receipt = base.prepare_admitted_data(data)
    require(set(frame["chronological_split"].unique().to_list()) == {"train", "validation"}, "Test materialized")
    write(folder / "input_receipt.json", input_receipt)
    interface = shared._training_interface(contract, data, frame, "full")
    interface["position_encoding"] = dict(POSITION_POLICY)
    interface["pilot_schema"] = SCHEMA
    args = shared._training_args(contract, data, folder, "full")
    args.device, args.execution_role = device, "phen_fixed_position_pilot_validation"
    require(args.max_train_batches is None and args.max_val_batches is None and not args.force_rerun,
            "Partial batches or rerun not admitted")
    verify_internal_modules(root / "source")
    print(json.dumps({"event": "training_entered", "dataset_id": dataset, "seed": SEED, "host": HOST,
                      "time": now(), "training": contract["training"], "loader": data["loader"], "runtime": native,
                      "position_encoding": POSITION_POLICY}), flush=True)
    with fixed_training_hooks(data, frame, folder) as (tracked, monitoring):
        summary, _, _ = training.train_one(args=args, frame=frame,
            quantity_contract={"boundaries": data["quantity_boundaries_all_train_rows"], "strata": []},
            interface_meta=interface, backbone="full", quantity_variant=VARIANT, seed=SEED)
        proof = tracked["tracker"].proof()
        initialization = monitoring["position_change"]
    fit_seconds = time.monotonic() - started
    curve_files = sorted((folder / "curve_epochs").glob("epoch_*.json"))
    require(len(curve_files) == summary["completed_epochs"], "Missing end-of-epoch curves")
    model, checkpoint, digest = load_selected(contract, data, folder, summary, proof, device)
    selected_bytes = sha(checkpoint)
    reference = evaluate_validation(data, frame, model, folder / "validation_predictions.jsonl.gz", device)
    for metric, key in (("qty_rmse", "best_val_qty_rmse"), ("qty_mae", "best_val_qty_mae"), ("time_nll", "best_val_time_nll")):
        require(math.isclose(reference["metrics"]["overall"][metric], summary[key], rel_tol=1e-5, abs_tol=1e-5),
                "Selected Validation does not replay: " + metric)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, _, replay_digest = load_selected(contract, data, folder, summary, proof, device)
    replay = evaluate_validation(data, frame, model, folder / "validation_replay.jsonl.gz", device)
    count = common.compare_prediction_files(folder / "validation_replay.jsonl.gz", folder / "validation_predictions.jsonl.gz")
    require(digest == replay_digest and sha(checkpoint) == selected_bytes, "Checkpoint changed during replay")
    verify_files(contract, dataset)
    receipt = {"schema": SCHEMA, "status": "complete", "host": HOST, "dataset_id": dataset, "seed": SEED,
               "contract_sha256": canonical(contract), "source_files_sha256": contract["source"]["files_sha256"],
               "selected_checkpoint": str(checkpoint.relative_to(root)), "checkpoint_sha256": selected_bytes,
               "state_sha256": digest, "selected_epoch": summary["best_epoch"], "completed_epochs": summary["completed_epochs"],
               "summary": summary, "position_initialization": initialization, "time_head_training_verification": proof,
               "validation": reference, "validation_replay": replay, "individual_predictions_match": True, "replayed_rows": count,
               "fit_with_preparation_and_monitoring_seconds": fit_seconds,
               "total_seconds": time.monotonic() - started, "curve_epoch_count": len(curve_files),
               "runtime": native, "held_out_test_evaluated": False, "completed_at": now()}
    receipt["files"] = common.file_inventory(root, [p for p in folder.rglob("*") if p.is_file()])
    write(folder / "receipt.json", receipt)
    return receipt


def dispatch(contract, contract_path, contract_sha, device):
    _, root = verify_files(contract)
    for dataset in DATASETS:
        verify_qualification(contract, dataset, root)
    write(root / "dispatch_attempt.json", {"contract_sha256": canonical(contract), "host": HOST, "started_at": now(), "pid": os.getpid()})
    exits = []
    for dataset in DATASETS:
        verify_files(contract, dataset)
        guard = gpu_guard(contract)
        name = job_id(dataset)
        require(not (root / "jobs" / name).exists(), "Existing attempt; no automatic retry or resume")
        command = [contract["hosts"][HOST]["python"], "-u", "-m", "paper.scripts.run_phen_fixed_position_pilot",
                   "--contract", str(Path(contract_path).resolve()), "--contract-sha", contract_sha,
                   "--host", HOST, "--mode", "train", "--dataset", dataset, "--device", device]
        stdout, stderr = root / (name + ".stdout.log"), root / (name + ".stderr.log")
        started = time.monotonic()
        with stdout.open("xb") as out, stderr.open("xb") as err:
            child = subprocess.Popen(command, cwd=root / "source", stdout=out, stderr=err,
                                     env={**os.environ, "PYTHONHASHSEED": str(SEED), "PYTHONDONTWRITEBYTECODE": "1"})
            write(root / (name + ".launch.json"), {"pid": child.pid, "dataset_id": dataset, "seed": SEED, "host": HOST,
                  "command": command, "contract_sha256": canonical(contract), "gpu_guard": guard, "started_at": now()})
            code = child.wait()
        actual = {"dataset_id": dataset, "seed": SEED, "host": HOST, "pid": child.pid, "actual_os_returncode": code,
                  "contract_sha256": canonical(contract), "elapsed_seconds": time.monotonic() - started,
                  "stdout": {"path": stdout.name, "bytes": stdout.stat().st_size, "sha256": sha(stdout)},
                  "stderr": {"path": stderr.name, "bytes": stderr.stat().st_size, "sha256": sha(stderr)}, "observed_at": now()}
        write(root / (name + ".exit.json"), actual)
        exits.append(actual)
        if code != 0:
            write(root / "dispatch_receipt.json", {"status": "failed", "exits": exits, "contract_sha256": canonical(contract)})
            return code
        receipt = read(root / "jobs" / name / "receipt.json")
        require(receipt["status"] == "complete" and receipt["individual_predictions_match"]
                and receipt["contract_sha256"] == canonical(contract), "Child success lacks verified receipt")
        gpu_guard(contract)
    write(root / "dispatch_receipt.json", {"status": "complete", "host": HOST, "exits": exits,
          "contract_sha256": canonical(contract), "supervisor_own_exit_directly_observed": False, "completed_at": now()})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--host", choices=(HOST,), required=True)
    parser.add_argument("--mode", choices=("check", "synthetic", "capacity", "train", "dispatch"), required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    require(sha(args.contract) == args.contract_sha, "Contract bytes changed")
    contract = read(args.contract)
    if args.mode == "check":
        _, root = verify_files(contract, args.dataset)
        result = {"status": "verified", "root": str(root), "contract_sha256": canonical(contract)}
    elif args.mode == "dispatch":
        return dispatch(contract, args.contract, args.contract_sha, args.device)
    else:
        require(args.dataset is not None, "Dataset required")
        result = train(contract, args.dataset, args.device) if args.mode == "train" else qualify(contract, args.dataset, args.mode, args.device)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
