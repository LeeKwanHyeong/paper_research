"""Isolated, fixed-budget J/Q/T diagnostic engine; never changes baseline defaults.

Only train and validation loaders are accepted by the caller's frozen contract.
Checkpoints are trusted, local experiment artifacts. ``last_epoch_state.pt`` is
an atomic epoch commit; JSON reports and selector files are repairable mirrors.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    capture_rng_state,
    restore_rng_state,
    torch_load_checkpoint,
)

OBJECTIVES = ("joint", "quantity_only", "time_only")
SELECTORS = ("raw_quantity_rmse", "legacy_time_loss")
SCHEMA_VERSION = "time_quantity_diagnostic_v1"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_model(model: torch.nn.Module, objective: str) -> None:
    _check(objective in OBJECTIVES, f"Unsupported objective: {objective}")
    _check(getattr(model, "quantity_variant", None) == "count_only_log_regression", "Diagnostic requires unweighted count_only_log_regression")
    _check(getattr(model, "time_head_mode", None) == "legacy_clamped_rmtpp", "Diagnostic requires legacy_clamped_rmtpp")
    _check(getattr(model, "time_intercept_limit", None) == 300.0, "Diagnostic requires explicit legacy time intercept cap 300")
    _check(getattr(model, "lambda_tail", None) == 0.0, "Diagnostic excludes auxiliary tail loss")
    _check(getattr(model, "quantile_adaptive_strength", 0.0) == 0.0, "Diagnostic excludes quantity weighting")
    _check(getattr(model, "quantity_memory_gradient_mode", "shared") == "shared", "Diagnostic requires shared quantity gradient route")
    _check(getattr(model, "memory_mode", None) != "hard_lmm_local_time", "Diagnostic excludes local-only time routing")


def _group(name: str) -> str:
    if name in {"b_t", "w_raw"} or name.startswith("v_t."):
        return "time_head"
    if name.startswith("quantity_head."):
        return "quantity_head"
    return "encoder"


def configure_parameters(model: torch.nn.Module, objective: str) -> list[torch.nn.Parameter]:
    """Select trainable parameters explicitly; inactive heads never enter AdamW."""
    _validate_model(model, objective)
    inactive = {"quantity_only": "time_head", "time_only": "quantity_head"}.get(objective)
    active = []
    for name, parameter in model.named_parameters():
        enabled = _group(name) != inactive
        parameter.requires_grad_(enabled)
        # A reused model must not carry stale gradients into the inactive head.
        parameter.grad = None
        if enabled:
            active.append(parameter)
    _check(bool(active), "No active parameters")
    return active


def task_outputs(model, dts, mask, quantities, objective: str) -> dict[str, Any]:
    """Use the baseline causal graph, calling only the objective's active head.

    Inactive losses and predictions are None, not zero-valued approximations.
    Joint arithmetic is identical to core.target_outputs(lambda_log_qty=1).
    """
    _validate_model(model, objective)
    _check(mask.dtype == torch.bool and dts.shape == mask.shape == quantities.shape, "Expected aligned two-dimensional dts/quantity/bool mask")
    _check(mask.ndim == 2 and mask.shape[0] > 0, "Empty or invalid batch")
    _check(bool(torch.isfinite(dts[mask]).all()) and bool((dts[mask] >= 0).all()), "Observed durations must be finite and nonnegative")
    _check(bool(torch.isfinite(quantities[mask]).all()) and bool((quantities[mask] >= 0).all()), "Observed quantities must be finite and nonnegative")
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions, history_positions = lengths - 1, lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    memory_write_mask = mask.clone()
    memory_write_mask[batch_ids, target_positions] = False
    time_encoded, quantity_encoded = model.encode_task_states(
        dts, history_quantities, mask, memory_write_mask=memory_write_mask,
    )
    true_qty = quantities[batch_ids, target_positions].float()
    time_loss = None
    quantity = None
    if objective != "quantity_only":
        time_loss = -model.log_f_dt(time_encoded[batch_ids, history_positions], dts[batch_ids, target_positions].float())
    if objective != "time_only":
        quantity = model.quantity_outputs(quantity_encoded[batch_ids, history_positions], true_qty)
    quantity_loss = quantity["train_loss"] if quantity is not None else None
    if objective == "joint":
        loss = time_loss + 1.0 * quantity_loss
    else:
        loss = quantity_loss if objective == "quantity_only" else time_loss
    return {
        "objective_loss": loss,
        "joint_loss": loss if objective == "joint" else None,
        "time_loss": time_loss,
        "loss_time": time_loss,
        "quantity_train_loss": quantity_loss,
        "loss_quantity": quantity_loss,
        "log_qty_loss": quantity["log_mse"] if quantity is not None else None,
        "true_qty": true_qty,
        "pred_qty": quantity["point_prediction"] if quantity is not None else None,
        "history_length": lengths - 1,
    }


def _grad_norm(gradients) -> float:
    return math.sqrt(sum(float(gradient.detach().double().square().sum().item()) for gradient in gradients if gradient is not None))


def gradient_diagnostics(model, dts, mask, quantities, grad_clip: float = 1.0) -> dict[str, Any]:
    """Read train-batch gradients without changing RNG, flags, state, or .grad.

    Both losses use one stochastic encoder realization. The joint global norm
    includes BOTH exclusive heads as well as encoder/memory parameters, matching
    the optimizer's global clipping mechanism. Caller chooses model train/eval.
    """
    _validate_model(model, "joint")
    _check(math.isfinite(grad_clip) and grad_clip > 0.0, "grad_clip must be finite and positive")
    rng = capture_rng_state()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    parameters = list(model.named_parameters())
    flags = [parameter.requires_grad for _, parameter in parameters]
    try:
        for _, parameter in parameters:
            parameter.requires_grad_(True)
        outputs = task_outputs(model, dts, mask, quantities, "joint")
        _check(all(bool(torch.isfinite(outputs[key]).all()) for key in ("time_loss", "quantity_train_loss", "objective_loss")), "Nonfinite diagnostic objective")
        tensors = [parameter for _, parameter in parameters]
        time_grads = torch.autograd.grad(outputs["time_loss"].mean(), tensors, retain_graph=True, allow_unused=True)
        quantity_grads = torch.autograd.grad(outputs["quantity_train_loss"].mean(), tensors, retain_graph=True, allow_unused=True)
        joint_grads = torch.autograd.grad(outputs["objective_loss"].mean(), tensors, allow_unused=True)
        _check(all(bool(torch.isfinite(g).all()) for grads in (time_grads, quantity_grads, joint_grads) for g in grads if g is not None), "Nonfinite diagnostic gradient")
        groups = {group: [i for i, (name, _) in enumerate(parameters) if _group(name) == group] for group in ("encoder", "time_head", "quantity_head")}
        per_task = {task: {group: _grad_norm(grads[i] for i in ids) for group, ids in groups.items()} for task, grads in (("time", time_grads), ("quantity", quantity_grads), ("joint", joint_grads))}
        dot = sum(float((time_grads[i].detach().double() * quantity_grads[i].detach().double()).sum().item()) for i in groups["encoder"] if time_grads[i] is not None and quantity_grads[i] is not None)
        denominator = per_task["time"]["encoder"] * per_task["quantity"]["encoder"]
        joint_norm = _grad_norm(joint_grads)
        quantity_norm = _grad_norm(quantity_grads)
        time_norm = _grad_norm(time_grads)
        clip_factor = min(1.0, grad_clip / (joint_norm + 1e-6))
        result = {
            "count": int(outputs["true_qty"].numel()),
            "time_loss": float(outputs["time_loss"].mean().detach().item()),
            "quantity_loss": float(outputs["quantity_train_loss"].mean().detach().item()),
            "per_task_gradient_norms": per_task,
            "parameter_groups": {group: [parameters[i][0] for i in ids] for group, ids in groups.items()},
            "encoder_gradient_dot": dot,
            "encoder_cosine": dot / denominator if denominator > 0.0 else None,
            "encoder_cosine_applicable": denominator > 0.0,
            "global_joint_preclip_norm": joint_norm,
            "global_joint_clip_factor": clip_factor,
            "global_quantity_preclip_norm": quantity_norm,
            "global_quantity_clip_factor": min(1.0, grad_clip / (quantity_norm + 1e-6)),
            "global_time_preclip_norm": time_norm,
            "global_time_clip_factor": min(1.0, grad_clip / (time_norm + 1e-6)),
            "time_head_joint_squared_norm_share": per_task["joint"]["time_head"] ** 2 / joint_norm ** 2 if joint_norm > 0.0 else None,
            "global_clip_scaled_quantity_encoder_norm": clip_factor * per_task["quantity"]["encoder"],
            "global_clip_scaled_time_encoder_norm": clip_factor * per_task["time"]["encoder"],
            "model_training": bool(model.training),
        }
    finally:
        for (_, parameter), enabled in zip(parameters, flags, strict=True):
            parameter.requires_grad_(enabled)
        after_hash = canonical_state_dict_sha256(model.state_dict())
        before_hash = canonical_state_dict_sha256(before)
        if after_hash != before_hash:
            model.load_state_dict(before, strict=True)
        restore_rng_state(rng)
    _check(after_hash == before_hash, "Diagnostic model forward mutated persistent state; state restored")
    return result


def _loader_generators(loader) -> dict[str, torch.Generator]:
    found = {}
    for name, obj in (("loader", loader), ("sampler", getattr(loader, "sampler", None))):
        generator = getattr(obj, "generator", None)
        if generator is not None:
            found[name] = generator
    return found


def _generator_hash(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def _generator_aliases(generators):
    """Name object sharing stably across processes, without persisting id()."""
    representatives = {}
    result = {}
    for name, generator in generators.items():
        representative = representatives.setdefault(id(generator), name)
        result[name] = representative
    return result


def _loader_spec(loader, *, training: bool) -> dict[str, Any]:
    _check(getattr(loader, "num_workers", 0) == 0, "Exact epoch resume requires num_workers=0")
    _check(not getattr(loader, "persistent_workers", False), "Persistent loader workers are unsupported")
    _check(not getattr(loader, "drop_last", False), "Fixed sample budget requires drop_last=False")
    generators = _loader_generators(loader)
    _check(not training or bool(generators), "Training requires an explicit deterministic loader generator")
    _check(hasattr(loader, "dataset") and hasattr(loader, "sampler"), "Expected a finite DataLoader")
    _check(type(loader.sampler).__name__ in {"RandomSampler", "SequentialSampler"}, "Unsupported custom sampler")
    _check(not getattr(loader.sampler, "replacement", False), "Replacement sampling is unsupported")
    _check(len(loader.dataset) > 0 and len(loader) > 0, "Empty loader")
    target_splits = getattr(loader.dataset, "target_splits", None)
    if target_splits is not None:
        expected_split = "train" if training else "validation"
        _check(set(target_splits) == {expected_split}, f"Loader targets must be {expected_split} only")
    return {
        "dataset_class": type(loader.dataset).__module__ + "." + type(loader.dataset).__name__,
        "dataset_length": len(loader.dataset),
        "target_splits": sorted(target_splits) if target_splits is not None else None,
        "batches": len(loader),
        "batch_size": loader.batch_size,
        "drop_last": bool(loader.drop_last),
        "num_workers": loader.num_workers,
        "sampler": type(loader.sampler).__name__,
        "sampler_length": len(loader.sampler),
        "generator_initial_states": {name: _generator_hash(generator) for name, generator in generators.items()},
        "collate_fn": getattr(loader.collate_fn, "__module__", "") + "." + getattr(loader.collate_fn, "__qualname__", type(loader.collate_fn).__name__),
    }


def _model_spec(model) -> dict[str, Any]:
    # Include non-state-dict objective/architecture scalars, e.g. dropout and cap.
    scalars = {}
    for module_name, module in model.named_modules():
        values = {}
        for name, value in vars(module).items():
            if name.startswith("_") or name == "training":
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                values[name] = value
            elif isinstance(value, (tuple, list)) and all(v is None or isinstance(v, (str, int, float, bool)) for v in value):
                values[name] = list(value)
        scalars[module_name] = {"class": type(module).__module__ + "." + type(module).__qualname__, "settings": values}
    source_files = {}
    for module in model.modules():
        filename = inspect.getsourcefile(type(module))
        if filename and not type(module).__module__.startswith("torch."):
            source_files[type(module).__module__] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return {"modules": scalars, "source_sha256": source_files}


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _batch_tensors(batch, device):
    if isinstance(batch, dict):
        dts, mask, quantities = batch["dts"], batch["mask"], batch["quantities"]
    elif len(batch) == 5:
        _, dts, mask, _, quantities = batch
    elif len(batch) == 3:
        dts, mask, quantities = batch
    else:
        raise ValueError("Batch must be canonical five-tuple, three tensors, or explicit dictionary")
    _check(quantities is not None, "Batch has no raw quantities")
    return dts.to(device), mask.to(device), quantities.to(device)


def _metric_totals() -> dict[str, float]:
    return {name: 0.0 for name in ("objective", "time", "quantity", "abs", "square", "count", "batches")}


def _accumulate(totals, outputs):
    _check(bool(torch.isfinite(outputs["objective_loss"]).all()), "Nonfinite active objective")
    totals["objective"] += float(outputs["objective_loss"].detach().double().sum().item())
    if outputs["time_loss"] is not None:
        _check(bool(torch.isfinite(outputs["time_loss"]).all()), "Nonfinite time loss")
        totals["time"] += float(outputs["time_loss"].detach().double().sum().item())
    if outputs["pred_qty"] is not None:
        _check(bool(torch.isfinite(outputs["pred_qty"]).all()), "Nonfinite quantity prediction")
        totals["quantity"] += float(outputs["quantity_train_loss"].detach().double().sum().item())
        error = outputs["pred_qty"].detach().double() - outputs["true_qty"].detach().double()
        totals["abs"] += float(error.abs().sum().item())
        totals["square"] += float(error.square().sum().item())
    totals["count"] += outputs["true_qty"].numel()
    totals["batches"] += 1


def _metrics(totals, objective):
    _check(totals["count"] > 0, "No evaluated targets")
    count = totals["count"]
    return {
        "count": int(count), "batches": int(totals["batches"]),
        "objective_loss": totals["objective"] / count,
        "legacy_time_loss": totals["time"] / count if objective != "quantity_only" else None,
        "quantity_train_loss": totals["quantity"] / count if objective != "time_only" else None,
        "quantity_mae": totals["abs"] / count if objective != "time_only" else None,
        "raw_quantity_rmse": math.sqrt(totals["square"] / count) if objective != "time_only" else None,
    }


def update_selectors(selectors: dict[str, Any], metrics: dict[str, Any], *, epoch: int, global_step: int, state: dict[str, torch.Tensor]) -> None:
    """Update only on a strict finite decrease, retaining the earliest tie."""
    for name, selector in selectors.items():
        if not selector["applicable"]:
            continue
        value = metrics[name]
        _check(isinstance(value, (float, int)) and math.isfinite(value), f"Nonfinite selector {name}")
        if selector["best_value"] is None or value < selector["best_value"]:
            saved = {key: tensor.detach().cpu().clone() for key, tensor in state.items()}
            selector.update(best_value=float(value), best_epoch=epoch, global_step=global_step, state_sha256=canonical_state_dict_sha256(saved), model_state_dict=saved)


def _public_selectors(selectors):
    return {name: {key: value for key, value in selector.items() if key != "model_state_dict"} for name, selector in selectors.items()}


def _inactive_state_hash(state, objective):
    inactive = {"quantity_only": "time_head", "time_only": "quantity_head"}.get(objective)
    return canonical_state_dict_sha256({name: value for name, value in state.items() if _group(name) == inactive})


def _optimizer_hash(state):
    tensors = {}
    metadata = {"param_groups": state["param_groups"], "state": {}}
    for parameter_id, values in state["state"].items():
        metadata["state"][str(parameter_id)] = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                tensors[f"{parameter_id}.{name}"] = value
            else:
                metadata["state"][str(parameter_id)][name] = value
    return _sha_json({"metadata": metadata, "tensor_sha256": canonical_state_dict_sha256(tensors)})


def _rng_hash(rng_state, loader_generator_states):
    """Hash RNG leaves and container types without pickle serialization noise."""
    def fingerprint(value):
        if isinstance(value, torch.Tensor):
            return {"type": "torch_tensor", "sha256": canonical_state_dict_sha256({"value": value})}
        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            return {"type": "numpy_array", "dtype": str(contiguous.dtype), "shape": list(contiguous.shape), "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest()}
        if isinstance(value, dict):
            _check(all(isinstance(key, str) for key in value), "RNG dictionary keys must be strings")
            return {"type": "dict", "items": [[key, fingerprint(value[key])] for key in sorted(value)]}
        if isinstance(value, (tuple, list)):
            return {"type": type(value).__name__, "items": [fingerprint(item) for item in value]}
        if value is None or type(value) in {str, int, float, bool}:
            return {"type": type(value).__name__, "value": value}
        raise ValueError(f"Unsupported RNG state leaf: {type(value).__name__}")
    return _sha_json(fingerprint({"global_rng": rng_state, "loader_generators": loader_generator_states}))


def _validate_resume(payload, contract):
    _check(payload.get("contract") == contract, "Resume identity/settings/runtime/initialization mismatch")
    _check(payload.get("contract_sha256") == _sha_json(contract), "Resume contract hash mismatch")
    _check(canonical_state_dict_sha256(payload["model_state_dict"]) == payload["model_state_sha256"], "Resume model state hash mismatch")
    _check(_inactive_state_hash(payload["model_state_dict"], contract["objective"]) == contract["inactive_state_sha256"], "Resume inactive head changed")
    optimizer_state = payload["optimizer_state_dict"]
    _check(_optimizer_hash(optimizer_state) == payload.get("optimizer_state_sha256"), "Resume optimizer state hash mismatch")
    groups = optimizer_state["param_groups"]
    _check(len(groups) == 1, "Resume optimizer group mismatch")
    group = groups[0]
    _check(group["params"] == list(range(len(contract["active_parameter_names"]))), "Resume optimizer active parameter IDs mismatch")
    _check(group["lr"] == contract["lr"] and group["weight_decay"] == contract["weight_decay"] and list(group["betas"]) == contract["optimizer_betas"] and group["eps"] == contract["optimizer_eps"] and not group["amsgrad"] and not group["maximize"], "Resume optimizer settings mismatch")
    epoch = payload["epoch"]
    history = payload["history"]
    _check(type(epoch) is int and 0 < epoch <= contract["epochs"], "Invalid resume epoch")
    _check([row["epoch"] for row in history] == list(range(1, epoch + 1)), "Resume history epoch mismatch")
    expected_steps = 0
    for row in history:
        expected_steps += contract["train_loader"]["batches"]
        _check(row["global_step"] == expected_steps, "Resume equal-step history mismatch")
        _check(row["train_count"] == contract["train_loader"]["dataset_length"] and row["validation_count"] == contract["validation_loader"]["dataset_length"], "Resume target count mismatch")
    _check(payload["global_step"] == expected_steps, "Resume global step mismatch")
    for name, selector in payload["selectors"].items():
        applicable = (name == "legacy_time_loss" and contract["objective"] != "quantity_only") or (name == "raw_quantity_rmse" and contract["objective"] != "time_only")
        _check(selector["applicable"] == applicable, "Resume selector applicability mismatch")
        if not applicable:
            _check(selector["best_epoch"] is None and selector["best_value"] is None and selector["model_state_dict"] is None, "Inactive selector has state")
            continue
        _check(all(isinstance(row[name], (float, int)) and math.isfinite(row[name]) for row in history), "Resume selector history nonfinite")
        selected = min(history, key=lambda row: row[name])
        _check(selector["best_epoch"] == selected["epoch"] and selector["best_value"] == selected[name] and selector["global_step"] == selected["global_step"], "Resume strict earliest selector mismatch")
        _check(canonical_state_dict_sha256(selector["model_state_dict"]) == selector["state_sha256"], "Resume selected state hash mismatch")
    _check(set(payload["selectors"]) == set(SELECTORS), "Resume selector set mismatch")
    _check(payload.get("rng_state") is not None and payload.get("loader_generator_states") is not None, "Resume RNG state missing")
    _check(_rng_hash(payload["rng_state"], payload["loader_generator_states"]) == payload.get("rng_state_sha256"), "Resume RNG/loader generator state hash mismatch")
    saved_generators = payload["loader_generator_states"]
    aliases = contract["loader_generator_aliases"]
    _check(set(saved_generators) == set(aliases), "Resume loader generator set mismatch")
    for name, representative in aliases.items():
        _check(torch.equal(saved_generators[name].cpu(), saved_generators[representative].cpu()), "Resume shared loader generator states disagree")


def _publish(payload, output_dir):
    for name, selector in payload["selectors"].items():
        if selector["applicable"]:
            atomic_torch_save({"schema_version": SCHEMA_VERSION, "contract_sha256": payload["contract_sha256"], "selector": name, **selector}, output_dir / f"best_{name}_model.pt")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "objective": payload["contract"]["objective"],
        "status": "complete" if payload["epoch"] == payload["contract"]["epochs"] else "paused_at_epoch_boundary",
        "epochs_completed": payload["epoch"], "epochs_budget": payload["contract"]["epochs"],
        "global_step": payload["global_step"],
        "initial_state_sha256": payload["contract"]["initial_state_sha256"],
        "last_state_sha256": payload["model_state_sha256"],
        "contract_sha256": payload["contract_sha256"],
        "selectors": _public_selectors(payload["selectors"]),
        "history": payload["history"],
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
    }
    _atomic_json({"history": payload["history"]}, output_dir / "history.json")
    _atomic_json(summary, output_dir / "summary.json")
    return summary


def run_arm(*, model, train_loader, validation_loader, objective: str, output_dir, epochs: int, seed: int, identity: dict[str, Any], device="cpu", lr=0.001, weight_decay=0.01, grad_clip=1.0, resume=False, stop_after_epochs=None) -> dict[str, Any]:
    """Train one static-B arm for equal, fixed epochs; resume only at commits.

    Caller supplies a freshly, identically initialized model and fresh dedicated
    loaders even when resuming. The caller identity must contain frozen source,
    dataset hashes/splits/target identities and execution-contract metadata.
    """
    _validate_model(model, objective)
    _check(type(model).__name__ == "CountAwareTitanTPP" and getattr(model, "memory_mode", None) == "static_hard_lmm", "J/Q/T training is restricted to the static B backbone")
    _check(type(epochs) is int and epochs > 0 and type(seed) is int, "Positive integer epochs and integer seed required")
    _check(math.isfinite(lr) and lr > 0 and math.isfinite(weight_decay) and weight_decay >= 0 and math.isfinite(grad_clip) and grad_clip > 0, "Invalid optimizer settings")
    _check(isinstance(identity, dict) and bool(identity), "Explicit source/data/execution identity required")
    identity = _json_safe(identity)
    _check(stop_after_epochs is None or (type(stop_after_epochs) is int and 0 < stop_after_epochs <= epochs), "stop_after_epochs must be inside the fixed epoch budget")
    device = torch.device(device)
    _check(device.type == "cpu", "J/Q/T training is CPU-only until CUDA replay qualification is completed")
    model.to(device)
    active_parameters = configure_parameters(model, objective)
    initial_hash = canonical_state_dict_sha256(model.state_dict())
    generators = {f"{split}.{name}": generator for split, loader in (("train", train_loader), ("validation", validation_loader)) for name, generator in _loader_generators(loader).items()}
    contract = _json_safe({
        "schema_version": SCHEMA_VERSION, "identity": identity,
        "objective": objective, "epochs": epochs, "seed": seed,
        "lr": float(lr), "weight_decay": float(weight_decay), "grad_clip": float(grad_clip),
        "optimizer": "AdamW", "optimizer_betas": [0.9, 0.999], "optimizer_eps": 1e-8,
        "device": str(device), "initial_state_sha256": initial_hash,
        "inactive_state_sha256": _inactive_state_hash(model.state_dict(), objective),
        "active_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "active_parameter_count": sum(p.numel() for p in active_parameters),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "model": _model_spec(model),
        "train_loader": _loader_spec(train_loader, training=True),
        "validation_loader": _loader_spec(validation_loader, training=False),
        "loader_generator_aliases": _generator_aliases(generators),
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda, "platform": platform.platform(), "torch_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic, "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        "engine_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_rule": "earliest_strict_finite_minimum", "early_stopping": False,
    })
    output_dir = Path(output_dir)
    last_path = output_dir / "last_epoch_state.pt"
    optimizer = torch.optim.AdamW(active_parameters, lr=lr, weight_decay=weight_decay)
    if resume:
        _check(last_path.is_file(), "Resume requires an existing epoch checkpoint")
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        _validate_resume(payload, contract)
        _check(json.loads((output_dir / "contract.json").read_text()) == contract, "Saved contract JSON mismatch")
        _check(set(payload["loader_generator_states"]) == set(generators), "Resume loader RNG set mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        restore_rng_state(payload["rng_state"])
        for name, generator in generators.items():
            generator.set_state(payload["loader_generator_states"][name].cpu())
        history, selectors, global_step = payload["history"], payload["selectors"], payload["global_step"]
        start_epoch = payload["epoch"] + 1
    else:
        _check(not output_dir.exists() or not any(output_dir.iterdir()), "Output exists; use strict resume or a new directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(contract, output_dir / "contract.json")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        history, global_step, start_epoch = [], 0, 1
        selectors = {name: {"applicable": (name == "raw_quantity_rmse" and objective != "time_only") or (name == "legacy_time_loss" and objective != "quantity_only"), "best_epoch": None, "best_value": None, "global_step": None, "state_sha256": None, "model_state_dict": None} for name in SELECTORS}
    end_epoch = epochs if stop_after_epochs is None else stop_after_epochs
    _check(end_epoch >= start_epoch - 1, "Stop boundary is earlier than saved progress")
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        totals = _metric_totals()
        batch_digest = hashlib.sha256()
        clip_norms = []
        for batch in train_loader:
            dts, mask, quantities = _batch_tensors(batch, device)
            batch_digest.update(canonical_state_dict_sha256({"dts": dts, "mask": mask, "quantities": quantities}).encode())
            optimizer.zero_grad(set_to_none=True)
            outputs = task_outputs(model, dts, mask, quantities, objective)
            _accumulate(totals, outputs)
            outputs["objective_loss"].mean().backward()
            norm = torch.nn.utils.clip_grad_norm_(active_parameters, grad_clip, error_if_nonfinite=True)
            clip_norms.append(float(norm.item()))
            optimizer.step()
            global_step += 1
        train_metrics = _metrics(totals, objective)
        _check(train_metrics["count"] == len(train_loader.dataset) and train_metrics["batches"] == len(train_loader), "Training sample/batch budget drift")
        model.eval()
        totals = _metric_totals()
        with torch.no_grad():
            for batch in validation_loader:
                _accumulate(totals, task_outputs(model, *_batch_tensors(batch, device), objective))
        metrics = _metrics(totals, objective)
        _check(metrics["count"] == len(validation_loader.dataset) and metrics["batches"] == len(validation_loader), "Validation sample/batch budget drift")
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        _check(all(bool(torch.isfinite(tensor).all()) for tensor in state.values()), "Nonfinite model state")
        _check(_inactive_state_hash(state, objective) == contract["inactive_state_sha256"], "Inactive head changed during optimization")
        update_selectors(selectors, metrics, epoch=epoch, global_step=global_step, state=state)
        history.append({"epoch": epoch, "global_step": global_step, "train_count": train_metrics["count"], "train_batches": train_metrics["batches"], "validation_count": metrics["count"], "validation_batches": metrics["batches"], "train": train_metrics, "train_batch_order_sha256": batch_digest.hexdigest(), "train_global_preclip_norm_mean": sum(clip_norms) / len(clip_norms), "train_clipped_batch_count": sum(norm > grad_clip for norm in clip_norms), **{key: value for key, value in metrics.items() if key not in {"count", "batches"}}})
        optimizer_state = optimizer.state_dict()
        rng_state = capture_rng_state()
        loader_generator_states = {name: generator.get_state().clone() for name, generator in generators.items()}
        payload = {"schema_version": SCHEMA_VERSION, "contract": contract, "contract_sha256": _sha_json(contract), "epoch": epoch, "global_step": global_step, "model_state_dict": state, "model_state_sha256": canonical_state_dict_sha256(state), "optimizer_state_dict": optimizer_state, "optimizer_state_sha256": _optimizer_hash(optimizer_state), "history": history, "selectors": selectors, "rng_state": rng_state, "loader_generator_states": loader_generator_states, "rng_state_sha256": _rng_hash(rng_state, loader_generator_states)}
        atomic_torch_save(payload, last_path)
        _publish(payload, output_dir)
    return _publish(payload, output_dir)
