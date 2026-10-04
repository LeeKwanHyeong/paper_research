"""Isolated fixed-budget trainer for frozen quantity objective comparisons.

It intentionally has no CLI or remote-launch behavior.  A caller supplies
freshly initialized models/loaders and an execution identity; epoch commits are
the sole resume boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch.utils.data import TensorDataset

from paper.scripts.quantity_objective_comparison import (
    QuantityCase,
    QuantityCaseIdentity,
    QuantityStatistics,
    joint_causal_batch_objective,
)
from paper.scripts.time_quantity_diagnostic import (
    _accumulate,
    _atomic_json,
    _batch_tensors,
    _check,
    _generator_aliases,
    _json_safe,
    _loader_generators,
    _loader_spec,
    _metric_totals,
    _metrics,
    _model_spec,
    _optimizer_hash,
    _rng_hash,
    _sha_json,
    _validate_model,
    update_selectors,
)
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    capture_rng_state,
    restore_rng_state,
    torch_load_checkpoint,
)

if TYPE_CHECKING:
    from paper.scripts.mixed_quantity_objective import MixedQuantityObjective
    from paper.scripts.raw_aux_gradient_control import RawAuxGradientControl

SCHEMA_VERSION = "quantity_comparison_engine_v1"
SELECTORS = ("raw_quantity_rmse", "legacy_time_loss")
BudgetCheck = Callable[[str], None]


def _runtime() -> dict[str, Any]:
    return {"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads()}


def _call_budget(check: BudgetCheck | None, stage: str) -> None:
    if check is not None:
        check(stage)


def _validate_identity(identity: dict[str, Any]) -> dict[str, Any]:
    _check(isinstance(identity, dict), "Explicit source/data/runtime/execution identity required")
    required = {"source", "data", "runtime", "execution_contract_sha256"}
    _check(required <= set(identity), "Identity must include source, data, runtime, and execution_contract_sha256")
    return _json_safe(identity)


def _validate_device(device: torch.device, identity: dict[str, Any], qualification: dict[str, Any] | None, model=None, train_loader=None, validation_loader=None, epochs: int | None = None) -> None:
    if device.type == "cpu":
        return
    _check(str(device) == "cuda:0", "CUDA requires explicit device cuda:0")
    _check(isinstance(qualification, dict), "CUDA requires a passed qualification")
    _check(qualification.get("execution_contract_sha256") == identity["execution_contract_sha256"], "CUDA qualification execution contract mismatch")
    _check(qualification.get("runtime") == identity["runtime"], "CUDA qualification runtime mismatch")
    if qualification.get("passed") is True:
        _check(qualification.get("qualifies_cuda") is True, "CUDA receipt must explicitly qualify cuda")
        _check(isinstance(qualification.get("runtime"), dict) and qualification["runtime"].get("device") == "cuda:0", "CUDA receipt must pin runtime.device=cuda:0")
        from paper.scripts.quantity_comparison_runtime import runtime_identity

        _check(runtime_identity("cuda:0") == identity["runtime"], "Actual CUDA runtime identity mismatch")
        return
    bootstrap = qualification.get("purpose") == "synthetic_probe" and identity.get("purpose") == "synthetic_qualification"
    _check(bootstrap, "CUDA requires a passed qualification")
    _check(type(epochs) is int and 1 <= epochs <= 2, "Synthetic CUDA qualification budget exceeded")
    for loader in (train_loader, validation_loader):
        dataset = getattr(loader, "dataset", None)
        _check(type(dataset) is TensorDataset, "Synthetic CUDA qualification requires TensorDataset loaders")
        _check(1 <= len(dataset) <= 8, "Synthetic CUDA qualification budget exceeded")
        tensors = dataset.tensors
        _check(len(tensors) == 3 and all(tensor.ndim == 2 for tensor in tensors)
               and len({tuple(tensor.shape) for tensor in tensors}) == 1
               and 2 <= tensors[0].shape[1] <= 8,
               "Synthetic CUDA qualification requires three aligned sequence tensors with length <= 8")
    # CountAwareTitanTPP stores max_seq_len on its encoder, not on the model.
    encoder = getattr(model, "encoder", None)
    position = getattr(encoder, "pos_emb", None)
    _check(getattr(model, "hidden_dim", None) == 8
           and getattr(encoder, "max_len", None) == 8
           and isinstance(position, torch.Tensor) and tuple(position.shape) == (1, 8, 8)
           and getattr(getattr(model, "quantity_head", None), "in_features", None) == 8
           and getattr(getattr(model, "v_t", None), "in_features", None) == 8,
           "Synthetic CUDA qualification requires hidden_dim=max_seq_len=8")


def _selector_template() -> dict[str, dict[str, Any]]:
    return {name: {"applicable": True, "best_epoch": None, "best_value": None, "global_step": None, "state_sha256": None, "model_state_dict": None} for name in SELECTORS}


def _public_selectors(selectors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {name: {key: value for key, value in selector.items() if key != "model_state_dict"} for name, selector in selectors.items()}


def _comparison_metric_totals(*, mixed: bool = False) -> dict[str, float]:
    totals = _metric_totals()
    totals["log_quantity"] = 0.0
    if mixed:
        totals["raw_quantity_scaled"] = 0.0
    return totals


def _comparison_accumulate(totals: dict[str, float], outputs: dict[str, torch.Tensor]) -> None:
    _accumulate(totals, outputs)
    log_loss = outputs["log_qty_loss"]
    _check(log_loss is not None and bool(torch.isfinite(log_loss).all()), "Nonfinite log quantity loss")
    totals["log_quantity"] += float(log_loss.detach().double().sum().item())
    if "raw_quantity_scaled" in totals:
        raw_loss = outputs["raw_qty_loss"]
        _check(raw_loss is not None and bool(torch.isfinite(raw_loss).all()), "Nonfinite raw quantity loss")
        totals["raw_quantity_scaled"] += float(raw_loss.detach().double().sum().item())


def _comparison_metrics(totals: dict[str, float]) -> dict[str, Any]:
    metrics = _metrics(totals, "joint")
    metrics["log_quantity_mse"] = totals["log_quantity"] / totals["count"]
    if "raw_quantity_scaled" in totals:
        metrics["raw_quantity_scaled_mse"] = totals["raw_quantity_scaled"] / totals["count"]
    return metrics


def _gradient_norm(gradients) -> float:
    return math.sqrt(sum(float(gradient.detach().double().square().sum().item()) for gradient in gradients if gradient is not None))


def _first_batch_gradient_diagnostic(outputs: dict[str, torch.Tensor], named_parameters, grad_clip: float, *, include_quantity_components: bool = False) -> dict[str, Any]:
    """Measure task gradients without assigning ``.grad`` or consuming the graph."""
    parameters = [parameter for _, parameter in named_parameters]
    time_grads = torch.autograd.grad(outputs["time_loss"].mean(), parameters, retain_graph=True, allow_unused=True)
    quantity_grads = torch.autograd.grad(outputs["quantity_train_loss"].mean(), parameters, retain_graph=True, allow_unused=True)
    joint_grads = torch.autograd.grad(outputs["objective_loss"].mean(), parameters, retain_graph=True, allow_unused=True)
    _check(all(bool(torch.isfinite(gradient).all()) for gradients in (time_grads, quantity_grads, joint_grads) for gradient in gradients if gradient is not None), "Nonfinite task gradient diagnostic")
    groups = {"time_head": [], "quantity_head": [], "encoder": []}
    for index, (name, _) in enumerate(named_parameters):
        groups["time_head" if name in {"b_t", "w_raw"} or name.startswith("v_t.") else "quantity_head" if name.startswith("quantity_head.") else "encoder"].append(index)
    def grouped(grads):
        return {name: _gradient_norm(grads[index] for index in indices) for name, indices in groups.items()}
    time_norms, quantity_norms, joint_norms = grouped(time_grads), grouped(quantity_grads), grouped(joint_grads)
    dot = sum(float((time_grads[index].detach().double() * quantity_grads[index].detach().double()).sum().item()) for index in groups["encoder"] if time_grads[index] is not None and quantity_grads[index] is not None)
    denominator = time_norms["encoder"] * quantity_norms["encoder"]
    global_norm = _gradient_norm(joint_grads)
    result = {"time": time_norms, "quantity": quantity_norms, "joint": joint_norms, "encoder_dot": dot, "encoder_cosine": dot / denominator if denominator > 0.0 else None, "encoder_cosine_applicable": denominator > 0.0, "global_joint_preclip_norm": global_norm, "actual_clip_factor": min(1.0, grad_clip / (global_norm + 1e-6))}
    if include_quantity_components:
        components = {}
        for name, key in (("log", "log_qty_loss"), ("raw_scaled", "raw_qty_loss")):
            grads = torch.autograd.grad(outputs[key].mean(), parameters, retain_graph=True, allow_unused=True)
            _check(all(bool(torch.isfinite(gradient).all()) for gradient in grads if gradient is not None), "Nonfinite quantity component gradient diagnostic")
            components[name] = grouped(grads)
        result["quantity_component_unweighted_norms"] = components
    return result


def _validate_payload(payload: dict[str, Any], contract: dict[str, Any]) -> None:
    _check(payload.get("contract") == contract and payload.get("contract_sha256") == _sha_json(contract), "Resume identity/settings/runtime/initialization mismatch")
    _check(canonical_state_dict_sha256(payload["model_state_dict"]) == payload.get("model_state_sha256"), "Resume model state hash mismatch")
    _check(_optimizer_hash(payload["optimizer_state_dict"]) == payload.get("optimizer_state_sha256"), "Resume optimizer state hash mismatch")
    groups = payload["optimizer_state_dict"].get("param_groups", [])
    _check(len(groups) == 1, "Resume optimizer group mismatch")
    group = groups[0]
    _check(group.get("params") == list(range(len(contract["active_parameter_names"]))), "Resume optimizer active parameter IDs mismatch")
    _check(group.get("lr") == contract["lr"] and group.get("weight_decay") == contract["weight_decay"] and list(group.get("betas", ())) == contract["optimizer_betas"] and group.get("eps") == contract["optimizer_eps"] and not group.get("amsgrad") and not group.get("maximize"), "Resume optimizer settings mismatch")
    epoch = payload.get("epoch")
    history = payload.get("history")
    _check(type(epoch) is int and isinstance(history, list) and [row.get("epoch") for row in history] == list(range(1, epoch + 1)), "Resume history epoch mismatch")
    _check(0 < epoch <= contract["epochs"], "Invalid resume epoch")
    expected_steps = epoch * contract["train_loader"]["batches"]
    _check(payload.get("global_step") == expected_steps, "Resume equal-step history mismatch")
    for position, row in enumerate(history, start=1):
        _check(row.get("global_step") == position * contract["train_loader"]["batches"], "Resume history global step mismatch")
        _check(row.get("train_count") == contract["expected_train_targets"] and row.get("validation_count") == contract["expected_validation_targets"], "Resume target count mismatch")
    _check(set(payload.get("selectors", {})) == set(SELECTORS), "Resume selector set mismatch")
    for name, selector in payload["selectors"].items():
        _check(selector.get("applicable") is True, "Resume selector applicability mismatch")
        values = [row.get(name) for row in history]
        _check(all(isinstance(value, (float, int)) and math.isfinite(value) for value in values), "Resume selector history nonfinite")
        selected = min(history, key=lambda row: row[name])
        _check(selector["best_epoch"] == selected["epoch"] and selector["best_value"] == selected[name] and selector["global_step"] == selected["global_step"], "Resume strict earliest selector mismatch")
        _check(canonical_state_dict_sha256(selector["model_state_dict"]) == selector["state_sha256"], "Resume selected state hash mismatch")
    _check(_rng_hash(payload["rng_state"], payload["loader_generator_states"]) == payload.get("rng_state_sha256"), "Resume RNG/loader generator state hash mismatch")
    aliases = contract["loader_generator_aliases"]
    _check(set(payload["loader_generator_states"]) == set(aliases), "Resume loader generator set mismatch")
    for name, representative in aliases.items():
        _check(torch.equal(payload["loader_generator_states"][name].cpu(), payload["loader_generator_states"][representative].cpu()), "Resume shared loader generator states disagree")


def _publish(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    for name, selector in payload["selectors"].items():
        atomic_torch_save({"schema_version": SCHEMA_VERSION, "contract_sha256": payload["contract_sha256"], "condition": payload["contract"]["condition"], "selector": name, **selector}, output_dir / f"best_{name}_model.pt")
    summary = {"schema_version": SCHEMA_VERSION, "status": "complete" if payload["epoch"] == payload["contract"]["epochs"] else "paused_at_epoch_boundary", "epochs_completed": payload["epoch"], "epochs_budget": payload["contract"]["epochs"], "global_step": payload["global_step"], "initial_state_sha256": payload["contract"]["initial_state_sha256"], "last_state_sha256": payload["model_state_sha256"], "contract_sha256": payload["contract_sha256"], "condition": payload["contract"]["condition"], "selectors": _public_selectors(payload["selectors"]), "history": payload["history"], "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    _atomic_json({"history": payload["history"]}, output_dir / "history.json")
    _atomic_json(summary, output_dir / "summary.json")
    return summary


def run_case(*, model, train_loader, validation_loader, case: QuantityCase, statistics: QuantityStatistics, output_dir, epochs: int, seed: int, identity: dict[str, Any], device="cpu", lr=0.001, weight_decay=0.01, grad_clip=1.0, resume=False, stop_after_epochs=None, budget_check: BudgetCheck | None = None, cuda_qualification: dict[str, Any] | None = None, mixed_objective: MixedQuantityObjective | None = None, raw_aux_control: RawAuxGradientControl | None = None) -> dict[str, Any]:
    """Train one joint condition, with opt-in mixed loss and epoch commits.

    The legacy condition/metric schema is retained when ``mixed_objective`` is
    absent. Mixed coefficients and calibration identity become part of the
    same strict resume contract as initialization, data, runtime, and budget.
    """
    case = QuantityCase(case)
    if mixed_objective is not None:
        from paper.scripts.mixed_quantity_objective import MixedQuantityObjective, mixed_joint_causal_batch_objective

        _check(isinstance(mixed_objective, MixedQuantityObjective), "Mixed objective must be a frozen MixedQuantityObjective")
        _check(case is QuantityCase.B_LOG_ORIGINAL, "Mixed objective requires the original B output and backbone condition")
    if raw_aux_control is not None:
        from paper.scripts.raw_aux_gradient_control import (
            RawAuxGradientControl, apply_raw_aux_control, finite_gradient_norm, summarize_epoch,
        )
        _check(type(raw_aux_control) is RawAuxGradientControl, "Frozen RawAuxGradientControl required")
        _check(mixed_objective is not None and mixed_objective.name == "mixed_original"
               and mixed_objective.quantity_scale == 1.0, "Raw auxiliary control requires mixed_original with quantity_scale=1")
        _check(grad_clip == 1.0, "Raw auxiliary control preserves global grad_clip=1")

    def batch_objective(dts, mask, quantities):
        if mixed_objective is None:
            return joint_causal_batch_objective(model, dts, mask, quantities, statistics=statistics, case=case)
        return mixed_joint_causal_batch_objective(model, dts, mask, quantities, statistics=statistics, objective=mixed_objective)

    _check(type(epochs) is int and epochs > 0 and type(seed) is int, "Positive integer epochs and integer seed required")
    _check(math.isfinite(lr) and lr > 0 and math.isfinite(weight_decay) and weight_decay >= 0 and math.isfinite(grad_clip) and grad_clip > 0, "Invalid optimizer settings")
    _check(stop_after_epochs is None or type(stop_after_epochs) is int and 0 < stop_after_epochs <= epochs, "stop_after_epochs must be inside the fixed epoch budget")
    identity = _validate_identity(identity)
    device = torch.device(device)
    _validate_model(model, "joint")
    _check(type(model).__name__ == "CountAwareTitanTPP" and getattr(model, "memory_mode", None) == "static_hard_lmm", "Quantity comparison is restricted to the static B backbone")
    _loader_spec(train_loader, training=True)
    _loader_spec(validation_loader, training=False)
    _validate_device(device, identity, cuda_qualification, model, train_loader, validation_loader, epochs)
    model.to(device)
    parameters = list(model.parameters())
    _check(bool(parameters) and all(parameter.requires_grad for parameter in parameters), "Comparison requires all joint parameters trainable")
    initial_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    initial_hash = canonical_state_dict_sha256(initial_state)
    generators = {f"{split}.{name}": generator for split, loader in (("train", train_loader), ("validation", validation_loader)) for name, generator in _loader_generators(loader).items()}
    condition = QuantityCaseIdentity(case, statistics).to_json()
    if mixed_objective is not None:
        condition = json.dumps({**json.loads(condition), "mixed_objective": mixed_objective.to_dict()}, allow_nan=False, sort_keys=True)
    if raw_aux_control is not None:
        condition = json.dumps({**json.loads(condition), "raw_aux_control": raw_aux_control.to_dict()}, allow_nan=False, sort_keys=True)
    contract = _json_safe({"schema_version": SCHEMA_VERSION, "identity": identity, "condition": json.loads(condition), "epochs": epochs, "seed": seed, "lr": float(lr), "weight_decay": float(weight_decay), "grad_clip": float(grad_clip), "optimizer": "AdamW", "optimizer_betas": [0.9, .999], "optimizer_eps": 1e-8, "device": str(device), "initial_state_sha256": initial_hash, "parameter_count": sum(parameter.numel() for parameter in parameters), "active_parameter_names": [name for name, _ in model.named_parameters()], "model": _model_spec(model), "train_loader": _loader_spec(train_loader, training=True), "validation_loader": _loader_spec(validation_loader, training=False), "expected_train_targets": len(train_loader.dataset), "expected_validation_targets": len(validation_loader.dataset), "expected_global_steps": epochs * len(train_loader), "loader_generator_aliases": _generator_aliases(generators), "runtime_observed": _runtime(), "engine_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "selection_rule": "earliest_strict_finite_minimum", "early_stopping": False})
    output_dir = Path(output_dir)
    last_path = output_dir / "last_epoch_state.pt"
    timing_path = output_dir / "timing.json"
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if resume:
        _check(last_path.is_file(), "Resume requires an existing epoch checkpoint")
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        _validate_payload(payload, contract)
        _check(json.loads((output_dir / "contract.json").read_text()) == contract, "Saved contract JSON mismatch")
        _check(set(payload["loader_generator_states"]) == set(generators), "Resume loader RNG set mismatch")
        if raw_aux_control is not None:
            for row in payload["history"]:
                record = row.get("raw_aux_control", {})
                expected_file = f"raw_aux_batches_epoch_{row['epoch']:04d}.json"
                _check(record.get("batches_file") == expected_file, "Resume raw auxiliary record path mismatch")
                saved = json.loads((output_dir / expected_file).read_text())
                _check(_sha_json(saved) == record.get("batches_sha256"), "Resume raw auxiliary batch record hash mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        restore_rng_state(payload["rng_state"])
        for name, generator in generators.items():
            generator.set_state(payload["loader_generator_states"][name].cpu())
        history, selectors, global_step, start_epoch = payload["history"], payload["selectors"], payload["global_step"], payload["epoch"] + 1
    else:
        _check(not output_dir.exists() or not any(output_dir.iterdir()), "Output exists; use strict resume or a new directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(contract, output_dir / "contract.json")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        history, selectors, global_step, start_epoch = [], _selector_template(), 0, 1
    if resume and timing_path.exists():
        timing = json.loads(timing_path.read_text())
        _check(timing.get("contract_sha256") == _sha_json(contract) and isinstance(timing.get("epochs"), list), "Saved timing JSON mismatch")
    else:
        timing = {"schema_version": SCHEMA_VERSION, "contract_sha256": _sha_json(contract), "epochs": []}
    end_epoch = epochs if stop_after_epochs is None else stop_after_epochs
    _check(end_epoch >= start_epoch - 1, "Stop boundary is earlier than saved progress")
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_started = datetime.now(UTC).isoformat()
        epoch_clock = time.monotonic()
        model.train()
        train_totals = _comparison_metric_totals(mixed=mixed_objective is not None)
        digest = hashlib.sha256()
        norms = []
        diagnostic = None
        post_first_update_diagnostic = None
        if raw_aux_control is not None:
            raw_aux_records = []
            adaptive_objective_sum = adaptive_quantity_sum = 0.0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        for batch_index, batch in enumerate(train_loader):
            _call_budget(budget_check, "train_batch")
            dts, mask, quantities = _batch_tensors(batch, device)
            digest.update(canonical_state_dict_sha256({"dts": dts, "mask": mask, "quantities": quantities}).encode())
            optimizer.zero_grad(set_to_none=True)
            outputs = batch_objective(dts, mask, quantities)
            _comparison_accumulate(train_totals, outputs)
            if raw_aux_control is not None:
                outputs, raw_aux_record = apply_raw_aux_control(
                    outputs, model.named_parameters(), objective=mixed_objective,
                    control=raw_aux_control, training=model.training)
                adaptive_objective_sum += float(outputs["adaptive_objective_loss"].detach().double().sum().item())
                adaptive_quantity_sum += float(outputs["adaptive_quantity_train_loss"].detach().double().sum().item())
            if batch_index == 0 and epoch in {1, 30, 60, 120}:
                diagnostic = _first_batch_gradient_diagnostic(outputs, list(model.named_parameters()), grad_clip, include_quantity_components=mixed_objective is not None)
            if mixed_objective is not None and epoch == 1 and batch_index == 1:
                post_first_update_diagnostic = _first_batch_gradient_diagnostic(outputs, list(model.named_parameters()), grad_clip, include_quantity_components=True)
            outputs["objective_loss"].mean().backward()
            if raw_aux_control is not None:
                actual_joint_norm = finite_gradient_norm((parameter.grad for parameter in parameters), device=device)
                predicted_norm = raw_aux_record["predicted_composed_joint_norm_preclip"]
                _check(math.isclose(float(actual_joint_norm.item()), predicted_norm, rel_tol=1e-5, abs_tol=1e-7),
                       "Raw auxiliary composed gradient norm audit mismatch")
            norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
            if raw_aux_control is not None:
                actual_norm = float(norm.item())
                factor = min(1.0, grad_clip / (actual_norm + 1e-6))
                raw_aux_record.update(epoch=epoch, batch_index=batch_index, global_step=global_step + 1,
                                      composed_joint_norm_preclip=actual_norm, common_clip_factor=factor,
                                      clipped=factor < 1.0, norm_above_max=actual_norm > grad_clip)
                raw_aux_records.append(raw_aux_record)
            if diagnostic is not None and batch_index == 0 and epoch in {1, 30, 60, 120}:
                actual_norm = float(norm.item())
                diagnostic["actual_global_preclip_norm"] = actual_norm
                diagnostic["actual_clip_factor"] = min(1.0, grad_clip / (actual_norm + 1e-6))
            if post_first_update_diagnostic is not None and epoch == 1 and batch_index == 1:
                actual_norm = float(norm.item())
                post_first_update_diagnostic["actual_global_preclip_norm"] = actual_norm
                post_first_update_diagnostic["actual_clip_factor"] = min(1.0, grad_clip / (actual_norm + 1e-6))
            norms.append(float(norm.item()))
            optimizer.step()
            global_step += 1
        train_metrics = _comparison_metrics(train_totals)
        if raw_aux_control is not None:
            train_metrics["adaptive_objective_loss"] = adaptive_objective_sum / train_metrics["count"]
            train_metrics["adaptive_quantity_train_loss"] = adaptive_quantity_sum / train_metrics["count"]
            _check(all(math.isfinite(value) for value in train_metrics.values()), "Nonfinite capped training metric")
        _check(train_metrics["count"] == len(train_loader.dataset) and train_metrics["batches"] == len(train_loader), "Training sample/batch budget drift")
        model.eval()
        validation_totals = _comparison_metric_totals(mixed=mixed_objective is not None)
        with torch.no_grad():
            for batch in validation_loader:
                _call_budget(budget_check, "validation_batch")
                _comparison_accumulate(validation_totals, batch_objective(*_batch_tensors(batch, device)))
        metrics = _comparison_metrics(validation_totals)
        _check(metrics["count"] == len(validation_loader.dataset) and metrics["batches"] == len(validation_loader), "Validation sample/batch budget drift")
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        _check(all(bool(torch.isfinite(value).all()) for value in state.values()), "Nonfinite model state")
        update_selectors(selectors, metrics, epoch=epoch, global_step=global_step, state=state)
        history.append({"epoch": epoch, "global_step": global_step, "train_count": train_metrics["count"], "train_batches": train_metrics["batches"], "validation_count": metrics["count"], "validation_batches": metrics["batches"], "train": train_metrics, "train_batch_order_sha256": digest.hexdigest(), "train_global_preclip_norm_mean": sum(norms) / len(norms), "train_clipped_batch_count": sum(norm > grad_clip for norm in norms), "gradient_diagnostic": diagnostic, **({"post_first_update_gradient_diagnostic": post_first_update_diagnostic} if mixed_objective is not None else {}), **{key: value for key, value in metrics.items() if key not in {"count", "batches"}}})
        _call_budget(budget_check, "before_epoch_commit")
        if raw_aux_control is not None:
            batch_file = f"raw_aux_batches_epoch_{epoch:04d}.json"
            batch_payload = {"epoch": epoch, "contract_sha256": _sha_json(contract), "batches": raw_aux_records}
            history[-1]["raw_aux_control"] = {
                **summarize_epoch(raw_aux_records), "batches_file": batch_file,
                "batches_sha256": _sha_json(batch_payload),
                "train_reference_objective": "uncapped; adaptive training metrics are separate",
            }
            _atomic_json(batch_payload, output_dir / batch_file)
        optimizer_state = optimizer.state_dict()
        rng_state = capture_rng_state()
        generator_states = {name: generator.get_state().clone() for name, generator in generators.items()}
        payload = {"schema_version": SCHEMA_VERSION, "contract": contract, "contract_sha256": _sha_json(contract), "epoch": epoch, "global_step": global_step, "model_state_dict": state, "model_state_sha256": canonical_state_dict_sha256(state), "optimizer_state_dict": optimizer_state, "optimizer_state_sha256": _optimizer_hash(optimizer_state), "history": history, "selectors": selectors, "rng_state": rng_state, "loader_generator_states": generator_states, "rng_state_sha256": _rng_hash(rng_state, generator_states)}
        atomic_torch_save(payload, last_path)
        _publish(payload, output_dir)
        timing["epochs"].append({"epoch": epoch, "started_at": epoch_started, "finished_at": datetime.now(UTC).isoformat(), "epoch_wall_seconds": time.monotonic() - epoch_clock})
        if raw_aux_control is not None:
            import resource
            import sys

            timing["epochs"][-1]["raw_auxiliary_work"] = {
                "component_gradient_queries": 2 * len(train_loader),
                "sparse_diagnostic_gradient_queries": 5 * (int(epoch in {1, 30, 60, 120})
                                                           + int(epoch == 1 and len(train_loader) > 1)),
                "ordinary_backward_calls": len(train_loader),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                                                  * (1 if sys.platform == "darwin" else 1024),
                "rss_scope": "process lifetime maximum, not epoch increment",
            }
        _atomic_json(timing, timing_path)
    return _publish(payload, output_dir)
