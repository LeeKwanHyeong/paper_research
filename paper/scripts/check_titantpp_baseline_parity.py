"""Bounded parent CNN+GRU54/mean parity checks, never a fit or held-out evaluation.

Scientific imports are delayed until the sealed source gate passes. Models are
independently initialized; only RNG snapshots are restored between branches.
The receipt contains digests and difference diagnostics, not prediction rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import re
import sys
import time

SCHEMA = "titantpp_baseline_parity_20261008_v1"
SCIENCE_REVISION = "4274043717cf24640d9a18fb35231a1f7ba50c8e"
SOURCE126_CLOSURE = "918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a"
SOURCE117_CLOSURE = "40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd"
PARENT = "titantpp_cnn_gru54"
MEAN = "titantpp_retrieval_mean"
IDENTITY_BUFFER = "contextual_retrieval_identity"
OUTPUT_KEYS = {"joint_loss", "time_loss", "quantity_train_loss", "log_qty_loss",
    "quantity_distribution_nll", "quantity_location_huber", "quantity_scale",
    "tail_aux_loss", "tail_indicator", "true_qty", "pred_qty", "history_length"}


class ParityError(RuntimeError):
    """A sealed contract or strict equivalence condition failed."""


def require(condition, message):
    if not condition:
        raise ParityError(message)


def canonical_gpu_uuid(raw):
    """Accept Torch's bare UUID or NVIDIA's GPU-prefixed UUID, strictly."""
    require(isinstance(raw, str) and re.fullmatch(
        r"(?:GPU-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        raw, flags=re.IGNORECASE) is not None, "Malformed GPU UUID")
    value = raw[4:] if raw[:4].upper() == "GPU-" else raw
    return "GPU-" + value.lower()


def require_gpu_uuid_match(left_raw, right_raw):
    normalized = canonical_gpu_uuid(left_raw)
    require(normalized == canonical_gpu_uuid(right_raw), "GPU UUID changed")
    return normalized


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_record(tensor):
    import torch
    require(isinstance(tensor, torch.Tensor), "Expected a tensor")
    require(tensor.layout == torch.strided, "Unsupported tensor layout")
    value = tensor.detach().contiguous().cpu()
    require(not (value.is_floating_point() or value.is_complex())
        or bool(torch.isfinite(value).all()), "Nonfinite tensor")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
        "binary_sha256": hashlib.sha256(raw).hexdigest(), "numel": value.numel()}


def clone_tensors(mapping):
    return {name: None if value is None else value.detach().cpu().clone()
        for name, value in mapping.items()}


def compare_tensors(left, right, *, allowed_extra_right=()):
    """Strict key/shape/dtype/finite/value comparison with bounded diagnostics."""
    import torch
    extra = set(right) - set(left)
    missing = set(left) - set(right)
    expected_extra = set(allowed_extra_right)
    records, differences = {}, []
    max_abs = 0.0
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a is None or b is None:
            equal = a is None and b is None
            records[name] = {"equal": equal, "left_none": a is None, "right_none": b is None}
        else:
            ar, br = tensor_record(a), tensor_record(b)
            compatible = ar["shape"] == br["shape"] and ar["dtype"] == br["dtype"]
            equal = compatible and ar["binary_sha256"] == br["binary_sha256"]
            absolute = None
            differing = None
            if compatible:
                av, bv = a.detach().cpu(), b.detach().cpu()
                differing = int((av != bv).sum())
                if av.numel():
                    absolute = float((av.to(torch.complex128) - bv.to(torch.complex128)).abs().max())
                    max_abs = max(max_abs, absolute)
                else:
                    absolute = 0.0
            records[name] = {"equal": bool(equal), "left": ar, "right": br,
                "numeric_equal": compatible and bool(torch.equal(a.detach().cpu(), b.detach().cpu())),
                "max_abs": absolute, "different_elements": differing}
        if not equal:
            differences.append(name)
    extra_records = {name: tensor_record(right[name]) for name in sorted(extra)}
    return {"passed": not missing and extra == expected_extra and not differences,
        "missing_right": sorted(missing), "extra_right": sorted(extra),
        "allowed_extra_right": sorted(expected_extra), "different_keys": differences,
        "max_abs": max_abs, "tensor_count": len(records), "tensors": records,
        "extra_tensors": extra_records,
        "left_digest": sha_json({name: tensor_record(value) if value is not None else None
            for name, value in sorted(left.items())}),
        "right_digest": sha_json({name: tensor_record(value) if value is not None else None
            for name, value in sorted(right.items())})}


def flatten_tree(value, prefix="root"):
    import torch
    tensors, scalars = {}, {}
    def walk(item, name):
        if isinstance(item, torch.Tensor):
            tensors[name] = item
        elif isinstance(item, dict):
            scalars[name + ":keys"] = sorted(str(key) for key in item)
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                walk(item[key], name + "/" + type(key).__name__ + ":" + str(key))
        elif isinstance(item, (list, tuple)):
            scalars[name + ":container"] = [type(item).__name__, len(item)]
            for index, child in enumerate(item):
                walk(child, name + "/" + str(index))
        else:
            require(item is None or isinstance(item, (str, int, float, bool)), "Unsupported state value")
            require(not isinstance(item, float) or math.isfinite(item), "Nonfinite state scalar")
            scalars[name] = [type(item).__name__, item]
    walk(value, prefix)
    return tensors, scalars


def compare_tree(left, right):
    lt, ls = flatten_tree(left)
    rt, rs = flatten_tree(right)
    result = compare_tensors(lt, rt)
    result["scalars_equal"] = ls == rs
    result["left_scalar_sha256"], result["right_scalar_sha256"] = sha_json(ls), sha_json(rs)
    result["passed"] = result["passed"] and result["scalars_equal"]
    return result


def capture_rng(device):
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if str(device).startswith("cuda") else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_digest(state):
    return {"python": hashlib.sha256(pickle.dumps(state["python"], protocol=4)).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(state["numpy"], protocol=4)).hexdigest(),
        "torch_cpu": tensor_record(state["torch_cpu"])["binary_sha256"],
        "torch_cuda": [{"visible_device_index": index,
            "binary_sha256": tensor_record(item)["binary_sha256"]}
            for index, item in enumerate(state["torch_cuda"])]}


def compare_rng(left, right):
    a, b = rng_digest(left), rng_digest(right)
    return {"passed": a == b, "left": a, "right": b}


def seed_all(seed, device):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def gradients(model):
    return clone_tensors({name: parameter.grad for name, parameter in model.named_parameters()})


def parameters(model):
    return clone_tensors(dict(model.named_parameters()))


def assert_updated(before, after, optimizer_state, *, step):
    import torch
    require(set(before) == set(after), "Update changed parameter keys")
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    require(bool(changed), "Optimizer update changed no parameter")
    states = optimizer_state.get("state", {})
    require(bool(states), "Optimizer state empty after update")
    actual_steps = []
    for item in states.values():
        require("step" in item, "Optimizer state lacks step")
        value = item["step"]
        actual_steps.append(float(value.item() if isinstance(value, torch.Tensor) else value))
    require(all(value == step for value in actual_steps), "Optimizer step counter mismatch")
    return {"changed_parameter_count": len(changed), "changed_parameters": changed,
        "optimizer_state_parameter_count": len(states), "optimizer_step": step}


def memory_row(device, label):
    import resource
    import torch
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row = {"stage": label, "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return row


def checked(receipt, label, result):
    receipt["comparisons"].append({"stage": label, **result})
    require(result["passed"], "Strict parity failed: " + label)


def checked_memory(c, device, label, rows):
    row = memory_row(device, label)
    rows.append(row)
    if "resources" in c:
        require(row["process_peak_rss_bytes"] <= c["resources"]["rss_limit_bytes"], "Process RSS limit exceeded")
        if str(device).startswith("cuda"):
            require(row["cuda_peak_allocated_bytes"] <= c["resources"]["gpu_memory_limit_bytes"]
                and row["cuda_peak_reserved_bytes"] <= c["resources"]["gpu_memory_limit_bytes"],
                "GPU memory limit exceeded")


def branch_step(model, optimizer, batch, target_outputs, *, lambda_log_qty, grad_clip, step):
    """Exactly one inherited loss/backward/clip/AdamW update; no epoch API."""
    import torch
    before = parameters(model)
    model.train()
    output = target_outputs(model, batch["dts"], batch["mask"], batch["quantities"],
        lambda_log_qty=lambda_log_qty)
    require(set(output) == OUTPUT_KEYS, "Frozen output contract changed")
    output_snapshot = clone_tensors(output)
    for value in output_snapshot.values():
        tensor_record(value)
    loss = output["joint_loss"].mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    pre = gradients(model)
    require(any(value is not None and bool((value != 0).any()) for value in pre.values()),
        "Training produced no active gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
    require(bool(torch.isfinite(norm)), "Nonfinite gradient norm")
    post = gradients(model)
    optimizer.step()
    after = parameters(model)
    for value in after.values():
        tensor_record(value)
    opt = optimizer.state_dict()
    # Snapshot optimizer tensors: future updates must not mutate the evidence.
    def clone_tree(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: clone_tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_tree(item) for item in value)
        return value
    return {"outputs": output_snapshot, "loss": loss.detach().cpu().clone(),
        "gradients_pre_clip": pre, "gradients_post_clip": post,
        "pre_clip_norm": norm.detach().cpu().clone(), "parameters": after,
        "parameter_deltas": {name: after[name] - before[name] for name in before},
        "buffers": clone_tensors(dict(model.named_buffers())), "optimizer": clone_tree(opt),
        "update": assert_updated(before, after, opt, step=step)}


def run_pair_case(builders, batch_pair, target_outputs, build_optimizer, c, device, label, receipt,
        canonical_state_digest=None):
    """Fresh seed construction for each case, followed by paired bounded steps."""
    import torch
    models, constructor_rng, pre_constructor_rng, before_transfer_rng = [], [], [], []
    for builder in builders:
        seed_all(c["seed"], device)
        pre_constructor_rng.append(capture_rng(device))
        model, _ = builder()
        before_transfer_rng.append(capture_rng(device))
        model.to(device)
        models.append(model)
        constructor_rng.append(capture_rng(device))
    record = {"case": label, "comparisons": [], "updates": [], "memory": []}
    receipt["cases"].append(record)
    if canonical_state_digest is not None:
        record["initial_state_canonical_sha256"] = [canonical_state_digest(model.state_dict()) for model in models]
        record["mean_initial_state_projected_parent_canonical_sha256"] = canonical_state_digest(
            {name: value for name, value in models[1].state_dict().items() if name != IDENTITY_BUFFER})
    require(all(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        == c["expected_trainable_parameters"] for model in models), "Trainable parameter budget changed")
    require([name for name, _ in models[0].named_parameters()]
        == [name for name, _ in models[1].named_parameters()], "Optimizer parameter traversal order changed")
    checked(record, "before_constructor_rng", compare_rng(*pre_constructor_rng))
    checked(record, "constructor_rng", compare_rng(*before_transfer_rng))
    checked(record, "after_device_transfer_rng", compare_rng(*constructor_rng))
    checked(record, "initial_named_parameters", compare_tensors(*[parameters(model) for model in models]))
    checked(record, "initial_named_buffers", compare_tensors(*[clone_tensors(dict(model.named_buffers()))
        for model in models], allowed_extra_right=(IDENTITY_BUFFER,)))
    identity_initial = tensor_record(dict(models[1].named_buffers())[IDENTITY_BUFFER])
    checked(record, "initial_state", compare_tensors(*[clone_tensors(model.state_dict()) for model in models],
        allowed_extra_right=(IDENTITY_BUFFER,)))
    checked(record, "input_binary", compare_tensors(*batch_pair))
    record["input"] = [{name: tensor_record(value) for name, value in sorted(batch.items())}
        for batch in batch_pair]
    checked_memory(c, device, label + ":constructed", record["memory"])
    shared_rng = constructor_rng[0]
    results, rngs = [], []
    for model, batch in zip(models, batch_pair):
        model.eval()
        restore_rng(shared_rng)
        with torch.no_grad():
            output = target_outputs(model, batch["dts"], batch["mask"], batch["quantities"],
                lambda_log_qty=c["data_spec"]["model"]["lambda_log_qty"])
        require(set(output) == OUTPUT_KEYS, "Frozen output contract changed")
        results.append(clone_tensors(output))
        rngs.append(capture_rng(device))
    checked(record, "eval_outputs", compare_tensors(*results))
    checked(record, "eval_rng", compare_rng(*rngs))
    checked(record, "eval_buffers", compare_tensors(*[clone_tensors(dict(model.named_buffers()))
        for model in models], allowed_extra_right=(IDENTITY_BUFFER,)))
    require(tensor_record(dict(models[1].named_buffers())[IDENTITY_BUFFER]) == identity_initial,
        "Identity buffer changed during eval")
    checked_memory(c, device, label + ":eval", record["memory"])
    shared_rng = rngs[0]
    opts = [build_optimizer(model, lr=c["optimizer"]["lr"],
        time_head_lr_multiplier=c["optimizer"]["time_head_lr_multiplier"]) for model in models]
    for opt in opts:
        require(isinstance(opt, torch.optim.AdamW), "Optimizer implementation changed")
        for group in opt.param_groups:
            for name in ("weight_decay", "eps"):
                require(group[name] == c["optimizer"][name], "Optimizer default changed: " + name)
            require(list(group["betas"]) == c["optimizer"]["betas"], "Optimizer betas changed")
    checked(record, "initial_optimizer", compare_tree(*[opt.state_dict() for opt in opts]))
    updates = c["real_train"]["max_updates"] if label == "real_train_first_batch" else c["synthetic"]["updates_per_length"]
    for index in range(updates):
        outcomes, rngs = [], []
        for model, opt, batch in zip(models, opts, batch_pair):
            restore_rng(shared_rng)
            outcomes.append(branch_step(model, opt, batch, target_outputs,
                lambda_log_qty=c["data_spec"]["model"]["lambda_log_qty"], grad_clip=c["optimizer"]["grad_clip"], step=index + 1))
            rngs.append(capture_rng(device))
            checked_memory(c, device, label + ":update" + str(index + 1) + ":branch" + str(len(outcomes)), record["memory"])
        step_record = {"step": index + 1, "comparisons": [],
            "update": [outcome["update"] for outcome in outcomes]}
        record["updates"].append(step_record)
        for key in ("outputs", "gradients_pre_clip", "gradients_post_clip", "parameters", "parameter_deltas"):
            checked(step_record, key, compare_tensors(*[outcome[key] for outcome in outcomes]))
        for key in ("loss", "pre_clip_norm"):
            checked(step_record, key, compare_tensors(*[{key: outcome[key]} for outcome in outcomes]))
        checked(step_record, "buffers", compare_tensors(*[outcome["buffers"] for outcome in outcomes],
            allowed_extra_right=(IDENTITY_BUFFER,)))
        require(tensor_record(outcomes[1]["buffers"][IDENTITY_BUFFER]) == identity_initial,
            "Identity buffer changed during update")
        checked(step_record, "optimizer", compare_tree(*[outcome["optimizer"] for outcome in outcomes]))
        checked(step_record, "after_update_rng", compare_rng(*rngs))
        shared_rng = rngs[0]
    record["status"] = "passed"
    del models, opts, results


def synthetic_batch(batch_size, length, seed, device):
    import torch
    generator = torch.Generator().manual_seed(seed)
    dts = torch.randint(1, 9, (batch_size, length), generator=generator).float()
    qty = torch.randint(0, 100, (batch_size, length), generator=generator).float()
    positions = torch.arange(length).unsqueeze(0)
    padding = torch.arange(batch_size).unsqueeze(1) % min(17, length - 1)
    mask = positions >= padding
    dts[~mask], qty[~mask] = 0., 0.
    dts[torch.arange(batch_size), padding[:, 0]] = 0.
    return {"dts": dts.to(device), "quantities": qty.to(device), "mask": mask.to(device)}


def verify_source(source, c):
    require(c["schema"] == SCHEMA, "Wrong parity contract schema")
    require(c["science_revision"] == SCIENCE_REVISION, "Science revision changed")
    require(c["seed"] == 42, "Only seed42 is authorized")
    require(c["comparison"]["mode"] == "bitwise" and c["comparison"]["require_finite"] is True
        and c["comparison"]["copy_parameters_between_models"] is False
        and c["comparison"]["require_rng_post_state_equal"] is True, "Strict comparison policy changed")
    require(c["models"] == [PARENT, MEAN], "Only parent/mean comparison is authorized")
    require(c["allowed_extra_mean_buffers"] == [IDENTITY_BUFFER], "Buffer exclusion changed")
    require(all(c["approval"][name] is False for name in ("full_fit", "validation_inference", "test_inference",
        "checkpoint_selection", "expansion_training", "shared_runtime_change", "automatic_retry")), "Execution scope changed")
    require(c["synthetic"] == {"batch_size": 128, "lengths": [64, 256], "eval": True,
        "train": True, "updates_per_length": 1}, "Synthetic scope changed")
    require(1 <= c["real_train"]["max_updates"] <= 3, "Real update budget must be 1..3")
    require(c["real_train"]["split"] == "train" and c["real_train"]["no_other_split_materialization"] is True,
        "Only Train materialization is authorized")
    require(c["real_train"]["max_updates"] + len(c["synthetic"]["lengths"])
        * c["synthetic"]["updates_per_length"] <= c["approval"]["real_optimizer_updates_per_model_max"] == 5,
        "Total update scope changed")
    files = c["source_files"]
    require(len(files) == 126 and sha_json(files) == c["source_closure_sha256"] == SOURCE126_CLOSURE,
        "Source126 manifest changed")
    common = c["shared117_files"]
    require(len(common) == 117 and sha_json(common) == c["shared117_closure_sha256"] == SOURCE117_CLOSURE,
        "Shared117 manifest changed")
    require(all(files.get(name) == digest for name, digest in common.items()), "Shared117 source mismatch")
    for name, digest in files.items():
        path = (source / name).resolve()
        require(path.is_relative_to(source) and path.is_file() and sha_file(path) == digest,
            "Frozen source drift: " + name)
    require(c["expected_trainable_parameters"] == 115167, "Model capacity contract changed")
    require(c["optimizer"] == {"grad_clip": 1., "lr": .001, "name": "AdamW", "weight_decay": .01,
        "time_head_lr_multiplier": 1., "betas": [.9, .999], "eps": 1e-8}, "Frozen optimizer contract changed")
    require(c["data_spec"]["dataset_id"] == "yellow_trip_hourly"
        and c["data_spec"]["model"]["lambda_log_qty"] == 1., "Dataset/objective scope changed")
    require(c["runtime"]["deterministic_algorithms"] is True and c["runtime"]["tf32"] is False
        and c["runtime"]["cudnn_benchmark"] is False and c["runtime"]["cuda_matmul_precision"] == "highest",
        "Arithmetic policy changed")
    require(c["resources"]["gpu_memory_limit_bytes"] == 8 * 1024 ** 3
        and c["resources"]["rss_limit_bytes"] == 16 * 1024 ** 3
        and c["resources"]["process_timeout_seconds"] == 900, "Resource scope changed")
    return {"source126_files_sha256": c["source_closure_sha256"],
        "shared117_files_sha256": c["shared117_closure_sha256"], "files_checked": len(files)}


def load_frozen(source, files):
    from importlib.machinery import ModuleSpec
    from types import ModuleType
    prefixes = ("models", "data_loader", "simple_lab_test", "utils", "paper.scripts")
    for name, module in list(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            origin = getattr(module, "__file__", None)
            require(origin is None or Path(origin).resolve().is_relative_to(source),
                "Preloaded non-frozen scientific module: " + name)
    sys.path.insert(0, str(source))
    # Frozen bootstrap code can prepend a project root. Pin namespace package
    # paths before any scientific import so it cannot select workspace utils.
    for name in ("paper", "paper.scripts", "data_loader", "utils"):
        package_path = source.joinpath(*name.split("."))
        require(package_path.is_dir(), "Frozen package directory missing: " + name)
        if not (package_path / "__init__.py").exists():
            module = sys.modules.get(name, ModuleType(name))
            module.__path__ = [str(package_path)]
            module.__package__ = name
            module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = module
    cnn = importlib.import_module("paper.scripts.run_titantpp_cnn_gru")
    cnn.install_hooks()
    factory = importlib.import_module("models.TPPs.CountAwareFactory")
    parent_factory = factory.build_count_aware_model
    retrieval = importlib.import_module("paper.scripts.run_titantpp_contextual_retrieval")
    retrieval.install_hooks()
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    training = importlib.import_module("paper.scripts.count_aware_tpp_backbone.training")
    loader_module = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
    return cnn, retrieval, parent_factory, core, training, loader_module


def verify_imports(source, files):
    origins = {}
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".")
            for prefix in ("models", "data_loader", "simple_lab_test", "utils", "paper.scripts")):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        path = Path(origin).resolve()
        require(path.is_relative_to(source), "Non-frozen scientific import: " + name)
        relative = path.relative_to(source).as_posix()
        require(relative in files and sha_file(path) == files[relative], "Unsealed scientific import: " + relative)
        origins[name] = relative
    return origins


def first_train_batches(c, loader_module, core, device, contract_root):
    """Only Train rows are materialized; no validation/test loader exists here."""
    import polars as pl
    import numpy as np
    import torch
    from paper.scripts.run_count_aware_tpp_backbone_control import exact_target_population, derive_train_time_contract
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    from paper.scripts.quantity_comparison_data import _all_row_quantity_identity
    path = (contract_root / c["dataset"]["path"]).resolve()
    manifest_path = (contract_root / c["dataset"]["split_manifest_path"]).resolve()
    require(sha_file(path) == c["dataset"]["sha256"], "Raw parquet SHA mismatch")
    require(sha_file(manifest_path) == c["dataset"]["split_manifest_sha256"], "Split manifest SHA mismatch")
    frame = pl.scan_parquet(path).filter(pl.col("chronological_split") == "train").collect().sort(["oper_part_no", "seq"])
    require(set(frame["chronological_split"].unique().to_list()) == {"train"}, "Non-Train rows materialized")
    require(frame.height == c["dataset"]["train_row_count"], "Train row count changed")
    require(frame.select(pl.struct(["oper_part_no", "seq"]).n_unique()).item() == frame.height,
        "Duplicate Train event keys")
    quantity = np.log1p(frame["demand_qty"].to_numpy().astype(np.float64))
    mean, std = float(quantity.mean()), float(quantity.std())
    stats = c["data_spec"]["statistics"]
    require(math.isclose(mean, stats["train_log_mean"], rel_tol=0, abs_tol=1e-12)
        and math.isclose(std, stats["train_log_std"], rel_tol=0, abs_tol=1e-12), "Frozen Train quantity statistics mismatch")
    require(_all_row_quantity_identity(frame["demand_qty"].to_numpy().astype(np.float64))
        == stats["all_train_quantity_sha256"], "All-Train quantity digest changed")
    raw_frame = frame
    frame = core.prepare_count_frame(raw_frame)
    require(frame.height == raw_frame.height and all(frame[name].equals(raw_frame[name]) for name in raw_frame.columns
        if name not in ("mark", "scale_residual")), "Count preparation changed source rows")
    loader_config = c["data_spec"]["loader"]
    require(loader_config["batch_size"] == 128 and loader_config["max_seq_len"] == 256,
        "Native Train loader dimensions changed")
    target_quantities, population = exact_target_population(frame, target_split="train",
        **{name: loader_config[name] for name in ("lookback_weeks", "max_seq_len")})
    expected_population = c["data_spec"]["inherited_data_identity"]["populations"]["train"]
    require({name: population[name] for name in expected_population} == expected_population,
        "Canonical Train target population changed")
    raw_scale = max(1., float(np.sqrt(np.mean(np.square(target_quantities)))))
    require(math.isclose(raw_scale, stats["raw_scale"], rel_tol=0, abs_tol=1e-12), "Train raw scale changed")
    time_stats = derive_train_time_contract(frame,
        **{name: loader_config[name] for name in ("lookback_weeks", "max_seq_len")})
    frozen_time = c["data_spec"]["time_statistics"]
    for measured, frozen in (("time_scale", "train_time_scale"),
        ("target_log_scaled_mean", "train_log_scaled_mean"), ("target_log_scaled_std", "train_log_scaled_std")):
        require(math.isclose(time_stats[measured], frozen_time[frozen], rel_tol=0, abs_tol=1e-12),
            "Frozen Train time statistics mismatch: " + measured)
    batches, receipts = [], []
    for _ in range(2):
        generator = torch.Generator().manual_seed(c["seed"])
        loader = loader_module.make_loader(frame, target_split="train",
            **{name: loader_config[name] for name in ("batch_size", "lookback_weeks", "max_seq_len")},
            shuffle=c["real_train"]["shuffle"], generator=generator)
        dataset = loader.dataset
        order = [[str(dataset.parts[part]), int(dataset.seq_lists[part][context + 1])]
            for part, context in dataset.index]
        order_sha = sha_json(order)
        require(len(dataset) == c["dataset"]["train_target_count"], "Train target count changed")
        durations = torch.tensor([max(1., float(dataset.dt_lists[part][context + 1]))
            for part, context in dataset.index], dtype=torch.float64)
        require(target_dt_sha256(durations) == frozen_time["expected_train_target_dt_sha256"],
            "Train duration target digest changed")
        sampler_state = generator.get_state().clone()
        # Mirror the iterator base-seed draw, then replay this loader's own
        # batch sampler. Restore only its generator, never model tensors.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        first_indices = next(iter(loader.batch_sampler))
        generator.set_state(sampler_state)
        iterator = iter(loader)
        marks, dts, mask, part_ids, qty = next(iterator)
        require(qty is not None and dts.shape == (128, 256), "Native first batch changed")
        replayed = loader_module.collate_week_lookback([dataset[index] for index in first_indices])
        require(all(torch.equal(actual, expected) for actual, expected in zip((marks, dts, mask, part_ids, qty), replayed)),
            "First-batch sampler replay disagrees with frozen DataLoader")
        input_record = {"marks": marks, "dts": dts, "mask": mask, "part_ids": part_ids, "quantities": qty}
        batches.append({name: value.to(device) for name, value in input_record.items()})
        receipts.append({"split": "train", "target_count": len(dataset), "train_rows_materialized": frame.height,
            "train_target_order_sha256": order_sha, "loader_generator_before_sha256": tensor_record(sampler_state)["binary_sha256"],
            "first_batch_target_order_sha256": sha_json([order[index] for index in first_indices]),
            "first_batch_sample_indices_sha256": sha_json(first_indices),
            "loader_generator_after_sha256": tensor_record(generator.get_state())["binary_sha256"],
            "first_batch_binary_sha256": sha_json({name: tensor_record(value) for name, value in sorted(input_record.items())}),
            "quantity_train_log_mean": mean, "quantity_train_log_std": std, "quantity_raw_scale": raw_scale,
            "population": expected_population, "time_statistics": time_stats,
            "train_duration_sha256": target_dt_sha256(durations),
            "count_frame_preparation": {"source_columns_preserved": True, "source_rows_preserved": True,
                "mark": "constant zero Int32", "scale_residual": "demand_qty cast Float64"},
            "frozen_dataset_duration_rule": "delta_t cast Int32 then clamp min1",
            "updates_use": "repeat this same first batch; never advance loader"})
    require(receipts[0] == receipts[1], "Independently generated Train input/order differs")
    return batches, receipts


def actual_runtime(device):
    import numpy as np
    import polars as pl
    import torch
    result = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "polars": pl.__version__, "platform": platform.platform(), "host": platform.node(), "pid": os.getpid(),
        "device": str(device), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "num_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(), "default_dtype": str(torch.get_default_dtype()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if str(device).startswith("cuda"):
        properties = torch.cuda.get_device_properties(device)
        uuid_raw = str(getattr(properties, "uuid", ""))
        uuid_normalized = canonical_gpu_uuid(uuid_raw)
        result["gpu"] = {"name": properties.name, "uuid_raw": uuid_raw,
            "uuid": uuid_normalized, "uuid_normalized": uuid_normalized,
            "total_memory": properties.total_memory, "capability": list(torch.cuda.get_device_capability(device)),
            "device_index": torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()}
        result["cuda_rng_device_mapping"] = []
        for index in range(torch.cuda.device_count()):
            raw = str(getattr(torch.cuda.get_device_properties(index), "uuid", ""))
            normalized = canonical_gpu_uuid(raw)
            result["cuda_rng_device_mapping"].append({"visible_device_index": index,
                "uuid_raw": raw, "uuid": normalized, "uuid_normalized": normalized})
    return result


def input_bytes_gate(c, contract_root):
    result = {}
    for name, digest_name in (("path", "sha256"), ("split_manifest_path", "split_manifest_sha256")):
        path = (contract_root / c["dataset"][name]).resolve()
        require(path.is_relative_to(contract_root), "Input outside sealed bundle")
        require(sha_file(path) == c["dataset"][digest_name], "Input bytes changed: " + name)
        result[digest_name] = c["dataset"][digest_name]
    return result


def execute(source, c, device, receipt, contract_root):
    import torch
    require(str(device) in ("cpu", "cuda", "cuda:0"), "Only CPU or first visible CUDA device is authorized")
    require(not str(device).startswith("cuda") or torch.cuda.is_available(), "CUDA unavailable")
    torch.set_num_threads(c["runtime"]["threads"])
    torch.set_num_interop_threads(c["runtime"]["interop_threads"])
    torch.use_deterministic_algorithms(c["runtime"]["deterministic_algorithms"], warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = c["runtime"]["tf32"]
    torch.backends.cudnn.allow_tf32 = c["runtime"]["tf32"]
    torch.backends.cudnn.benchmark = c["runtime"]["cudnn_benchmark"]
    torch.set_float32_matmul_precision(c["runtime"]["cuda_matmul_precision"])
    receipt["runtime"] = actual_runtime(device)
    require(torch.get_default_dtype() == torch.float32 and c["runtime"]["precision"] == "float32",
        "Only float32 arithmetic is authorized")
    if str(device).startswith("cuda"):
        import subprocess
        lines = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).splitlines()
        uuid = require_gpu_uuid_match(receipt["runtime"]["gpu"]["uuid_raw"], c["runtime"]["gpu_uuid"])
        records = []
        for line in lines:
            fields = line.split(",")
            require(len(fields) == 2, "Malformed GPU/driver inventory row")
            raw, driver = (field.strip() for field in fields)
            records.append({"uuid_raw": raw, "uuid_normalized": canonical_gpu_uuid(raw),
                "driver": driver, "raw_line": line})
        matched = [record for record in records if record["uuid_normalized"] == uuid]
        require(len(matched) == 1, "GPU driver/UUID binding unavailable")
        receipt["runtime"]["driver"] = matched[0]["driver"]
        receipt["runtime"]["nvidia_smi_gpu_binding"] = matched[0]
        for name, expected in c["runtime"]["runtime_expected"].items():
            require(receipt["runtime"].get(name) == expected, "Runtime mismatch: " + name)
        torch.cuda.set_per_process_memory_fraction(c["resources"]["gpu_memory_limit_bytes"]
            / receipt["runtime"]["gpu"]["total_memory"], device)
        receipt["runtime_qualification"] = "native_contract_checked"
    else:
        receipt["runtime_qualification"] = "local_cpu_only; native runtime identity not claimed"
    receipt["input_precheck"] = input_bytes_gate(c, contract_root)
    modules = load_frozen(source, c["source_files"])
    receipt["scientific_imports_precheck"] = verify_imports(source, c["source_files"])
    cnn, retrieval, parent_factory, core, training, loader_module = modules
    data = c["data_spec"]
    config = {name: value for name, value in data["model"].items()
        if name not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    def parent_builder():
        return parent_factory(PARENT, **config, train_log_mean=data["statistics"]["train_log_mean"],
            train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])
    builders = (parent_builder, lambda: retrieval.build_model(data, MEAN))
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    for length in c["synthetic"]["lengths"]:
        batch = synthetic_batch(c["synthetic"]["batch_size"], length, c["seed"] + length, device)
        run_pair_case(builders, (batch, {name: value.clone() for name, value in batch.items()}),
            core.target_outputs, training.build_optimizer, c, device, "synthetic_T" + str(length), receipt,
            canonical_state_dict_sha256)
    if c["real_train"]["enabled"]:
        batches, inputs = first_train_batches(c, loader_module, core, device, contract_root)
        receipt["train_input_receipts"] = inputs
        run_pair_case(builders, batches, core.target_outputs, training.build_optimizer,
            c, device, "real_train_first_batch", receipt, canonical_state_dict_sha256)
    receipt["input_postcheck"] = input_bytes_gate(c, contract_root)
    if str(device).startswith("cuda"):
        process_lines = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True).splitlines()
        owned = [line for line in process_lines if line.split(",")[0].strip() == str(os.getpid())]
        owned_records = [{"uuid_raw": line.split(",")[-1].strip(),
            "uuid_normalized": canonical_gpu_uuid(line.split(",")[-1].strip()), "raw_line": line}
            for line in owned]
        require(any(record["uuid_normalized"] == canonical_gpu_uuid(c["runtime"]["gpu_uuid"])
            for record in owned_records),
            "Current PID is not reported on the approved GPU UUID")
        receipt["native_gpu_process"] = {"pid": os.getpid(), "gpu_uuid": c["runtime"]["gpu_uuid"],
            "gpu_uuid_normalized": canonical_gpu_uuid(c["runtime"]["gpu_uuid"]),
            "owned_nvidia_smi_rows": owned_records, "verified": True}
    receipt["scientific_imports"] = verify_imports(source, c["source_files"])
    receipt["source_postcheck"] = verify_source(source, c)
    require([case["case"] for case in receipt["cases"]] == ["synthetic_T64", "synthetic_T256"]
        + (["real_train_first_batch"] if c["real_train"]["enabled"] else []), "Completed case scope changed")
    require(all(case.get("status") == "passed" for case in receipt["cases"]), "Incomplete parity case")
    receipt["optimizer_updates_per_branch"] = sum(len(case["updates"]) for case in receipt["cases"])
    require(receipt["optimizer_updates_per_branch"] <= 5, "Update budget exceeded")
    receipt["status"] = "passed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cpu")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Receipt already exists; automatic overwrite/retry is forbidden")
    started = time.monotonic()
    receipt = {"schema": SCHEMA, "status": "failed", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv if argv is None else list(argv),
        "evaluation_scope": "train_first_batch_and_synthetic_only", "held_out_test_evaluated": False,
        "validation_evaluated": False, "full_fit": False, "epoch_api_called": False,
        "raw_predictions_written": False, "model_tensors_copied_between_branches": False,
        "historical_native_initialization_equivalence": "not_established_by_this_new_runtime_check", "cases": []}
    exit_code = 1
    try:
        source = args.source.resolve()
        c = json.loads(args.contract.read_text())
        receipt["contract_sha256"] = sha_file(args.contract)
        receipt["contract_canonical_sha256"] = sha_json(c)
        receipt["science_revision"] = c["science_revision"]
        receipt["source_gate"] = verify_source(source, c)
        receipt["source_root"] = str(source)
        execute(source, c, args.device, receipt, args.contract.resolve().parent)
        exit_code = 0
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output.resolve()),
        "output_sha256": sha_file(args.output), "cases": len(receipt["cases"])}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
