#!/usr/bin/env python3
"""Fit one common conditional CDF calibration layer on frozen aligned-B.

Only the zero-initialized two-shape calibrator is trainable.  The aligned-B
model, cached hidden states, base duration distribution, and quantity path are
immutable.  Train and validation cache rows are admitted; held-out rows are
never loaded.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from models.TPPs.ConditionalCDFCalibrator import (
    ConditionalKumaraswamyCDFCalibrator,
    GlobalKumaraswamyCDFCalibrator,
)
from paper.scripts.run_aligned_causal_duration_scale_adapter import (
    FrozenBaseTimeCache,
    derive_frozen_base_time_cache,
    freeze_source_model,
    load_verified_feature_cache,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    build_candidate_from_selected_checkpoint,
    censor_mask,
    resolve_censor_threshold,
    validate_observation_likelihood_contract,
)
from paper.scripts.run_hard_lmm_time_head_refit import (
    FrozenFeatureCache,
    cached_quantity_predictions,
    finite_nested_tensors,
    quantity_metrics,
    require,
    save_json,
    tensor_sha256,
)
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "aligned_conditional_cdf_calibration_v1"
DEFAULT_CONTRACT = PROJECT_ROOT / "paper/contracts/aligned_conditional_cdf_calibration_v1.json"
DEFAULT_ALIGNED_CONTRACT = PROJECT_ROOT / "paper/contracts/aligned_frozen_lognormal_duration_v1.json"
SUMMARY_NAME = "summary.json"
SELECTED_CHECKPOINT_NAME = "best_validation_primary_nll_model.pt"
LAST_CHECKPOINT_NAME = "last_epoch_state.pt"
ROLES = ("candidate", "global_shape_control", "permuted_hidden_control")
INSTACART_DATASET = "insta_market_basket"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_strict(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(name for name, _ in pairs)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        require(not duplicates, f"Duplicate JSON keys: {duplicates}")
        return dict(pairs)

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    require(isinstance(payload, dict), "JSON root must be an object")
    return payload


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(contract.get("schema_version") == 1, "Contract schema drift")
    require(contract.get("contract_id") == CONTRACT_ID, "Contract ID drift")
    candidate = contract.get("candidate", {})
    require(candidate.get("single_hypothesis") is True, "Candidate count drift")
    require(candidate.get("source_contract_id") == "aligned_frozen_lognormal_duration_v1", "Source contract drift")
    require(candidate.get("backbone_change") is False, "Backbone change admitted")
    require(candidate.get("quantity_path_change") is False, "Quantity change admitted")
    scope = contract.get("scope", {})
    expected_datasets = ["intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket"]
    require(scope.get("datasets") == expected_datasets, "Dataset scope drift")
    require(scope.get("input_splits") == ["train", "validation"], "Split scope drift")
    require(scope.get("held_out_test") is False, "Held-out data admitted")
    require(scope.get("seed") == 42, "Seed drift")
    transform = contract.get("cdf_transform", {})
    require(transform.get("expected_trainable_parameter_count") == 130, "Candidate size drift")
    require(transform.get("minimum_shape") == 0.1 and transform.get("maximum_shape") == 10.0, "Shape bounds drift")
    controls = contract.get("controls", {})
    require(controls.get("history_free_global_shapes", {}).get("trainable_parameter_count") == 2, "Global control drift")
    permuted = controls.get("same_capacity_permuted_hidden", {})
    require(permuted.get("trainable_parameter_count") == 130, "Permuted control size drift")
    require("manual_seed(42)" in str(permuted.get("permutation_formula")), "Permutation seed drift")
    optimization = contract.get("future_optimization_contract", {})
    expected_optimization = {
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 8192,
        "gradient_clip": 1.0,
        "epochs": 100,
        "minimum_epochs": 20,
        "early_stopping_patience": 20,
        "scheduler": None,
        "dataset_specific_hyperparameters": False,
        "result_dependent_retuning": False,
    }
    for key, expected in expected_optimization.items():
        require(optimization.get(key) == expected, f"Optimization drift: {key}")
    selection = contract.get("checkpoint_selection", {})
    require(selection.get("monitor") == "validation primary proper observation NLL", "Selector drift")
    require(selection.get("rule") == "earliest strict finite minimum", "Selector rule drift")
    require(selection.get("fallback") == "epoch 0", "Selector fallback drift")
    require(selection.get("other_metrics_in_selector") is False, "Guardrail entered selector")
    rows = contract.get("source_artifacts")
    require(isinstance(rows, list) and len(rows) == 3, "Expected three source rows")
    datasets = {str(row["dataset"]): dict(row) for row in rows}
    require(list(datasets) == expected_datasets, "Source row order drift")
    for dataset_name, row in datasets.items():
        for key in (
            "aligned_B_checkpoint_file_sha256",
            "aligned_B_model_state_sha256",
            "aligned_B_train_feature_cache_sha256",
            "aligned_B_validation_feature_cache_sha256",
            "aligned_B_quantity_prediction_sha256",
        ):
            require(_is_sha256(row.get(key)), f"Invalid {dataset_name} {key}")
        for key in ("aligned_A_validation_primary_nll", "aligned_B_validation_primary_nll", "aligned_B_validation_continuous_nll"):
            require(math.isfinite(float(row[key])), f"Invalid {dataset_name} {key}")
    return datasets


def validate_aligned_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(contract.get("contract_id") == "aligned_frozen_lognormal_duration_v1", "Wrong aligned contract")
    scope = contract.get("scope", {})
    require(scope.get("input_splits") == ["train", "validation"], "Aligned split drift")
    require(scope.get("held_out_test") is False, "Aligned contract admits held-out")
    rows = contract.get("datasets")
    require(isinstance(rows, list) and len(rows) == 3, "Aligned datasets missing")
    result = {str(row["dataset"]): dict(row) for row in rows}
    for dataset_name, row in result.items():
        for key in ("data_sha256", "split_manifest_sha256", "expected_train_target_identity_sha256", "expected_validation_target_identity_sha256"):
            require(_is_sha256(row.get(key)), f"Invalid aligned {dataset_name} {key}")
        validate_observation_likelihood_contract(row["observation_contract"])
    return result


def deterministic_hidden_permutation(count: int) -> torch.Tensor:
    require(count > 0, "Cannot permute an empty split")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return torch.randperm(count, generator=generator)


def _tensor_digest(label: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _stable_logdiff(log_large: torch.Tensor, log_small: torch.Tensor) -> torch.Tensor:
    require(log_large.shape == log_small.shape, "Log-difference shape drift")
    difference = (log_small - log_large).clamp_max(0.0)
    split = -math.log(2.0)
    low = difference < split
    complement = torch.empty_like(difference)
    if bool(low.any()):
        indices = torch.nonzero(low, as_tuple=False).squeeze(-1)
        complement = complement.index_copy(0, indices, torch.log1p(-torch.exp(difference[indices])))
    high = ~low
    if bool(high.any()):
        indices = torch.nonzero(high, as_tuple=False).squeeze(-1)
        complement = complement.index_copy(0, indices, torch.log(-torch.expm1(difference[indices])))
    result = log_large + complement
    require(bool(torch.isfinite(result).all()), "Non-finite log interval mass")
    return result


def _base_terms_at(
    *, location: torch.Tensor, sigma: torch.Tensor, time_value: torch.Tensor, time_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = time_value.to(torch.float64)
    require(bool((value > 0.0).all()), "Duration boundary must be positive")
    mu = location.to(torch.float64)
    scale = sigma.to(torch.float64)
    z = (torch.log(value) - math.log(time_scale) - mu) / scale
    log_cdf = torch.special.log_ndtr(z)
    log_survival = torch.special.log_ndtr(-z)
    log_density = -torch.log(value) - torch.log(scale) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z.square()
    return log_cdf, log_survival, log_density


def calibrated_observation_log_likelihood(
    *,
    module: ConditionalKumaraswamyCDFCalibrator,
    hidden: torch.Tensor,
    location: torch.Tensor,
    sigma: torch.Tensor,
    target_dt: torch.Tensor,
    time_scale: float,
    is_right_censored: torch.Tensor,
    observation_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return primary observation likelihood and continuous reference."""
    target = target_dt.to(torch.float64)
    target_cdf, target_sf, target_density = _base_terms_at(
        location=location, sigma=sigma, time_value=target, time_scale=time_scale
    )
    calibrated_target_sf = module.transformed_log_cdf_survival(
        base_log_cdf=target_cdf, base_log_survival=target_sf, hidden=hidden
    )[1]
    calibrated_density = module.transformed_log_density(
        base_log_density=target_density,
        base_log_cdf=target_cdf,
        base_log_survival=target_sf,
        hidden=hidden,
    )
    continuous = torch.where(is_right_censored, calibrated_target_sf, calibrated_density)
    if observation_mode == OBSERVATION_LIKELIHOOD_CONTINUOUS:
        return continuous, continuous
    require(observation_mode == OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER, "Unsupported observation mode")
    require(bool(torch.isclose(target, target.round(), atol=1e-8, rtol=0.0).all()), "Non-integer target")
    first_boundary = torch.full_like(target, 1.5)
    lower = (target - 0.5).clamp_min(0.5)
    upper = target + 0.5
    first_base = _base_terms_at(location=location, sigma=sigma, time_value=first_boundary, time_scale=time_scale)
    lower_base = _base_terms_at(location=location, sigma=sigma, time_value=lower, time_scale=time_scale)
    upper_base = _base_terms_at(location=location, sigma=sigma, time_value=upper, time_scale=time_scale)
    first_log_cdf = module.transformed_log_cdf_survival(
        base_log_cdf=first_base[0], base_log_survival=first_base[1], hidden=hidden
    )[0]
    lower_log_cdf, lower_log_sf = module.transformed_log_cdf_survival(
        base_log_cdf=lower_base[0], base_log_survival=lower_base[1], hidden=hidden
    )
    upper_log_cdf, upper_log_sf = module.transformed_log_cdf_survival(
        base_log_cdf=upper_base[0], base_log_survival=upper_base[1], hidden=hidden
    )
    left_mass = _stable_logdiff(upper_log_cdf, lower_log_cdf)
    right_mass = _stable_logdiff(lower_log_sf, upper_log_sf)
    lower_z = (torch.log(lower) - math.log(time_scale) - location.to(torch.float64)) / sigma.to(torch.float64)
    regular_mass = torch.where(lower_z >= 0.0, right_mass, left_mass)
    uncensored = torch.where(target == 1.0, first_log_cdf, regular_mass)
    primary = torch.where(is_right_censored, lower_log_sf, uncensored)
    require(bool(torch.isfinite(primary).all()), "Non-finite calibrated primary likelihood")
    return primary, continuous


@torch.no_grad()
def calibrated_time_median(
    *, module: ConditionalKumaraswamyCDFCalibrator, hidden: torch.Tensor,
    location: torch.Tensor, sigma: torch.Tensor, time_scale: float,
) -> torch.Tensor:
    probability = torch.full((hidden.shape[0],), 0.5, dtype=torch.float64, device=hidden.device)
    base_level = module.base_quantile_level(probability, hidden)
    z = torch.special.ndtri(base_level)
    median = time_scale * torch.exp(location.to(torch.float64) + sigma.to(torch.float64) * z)
    require(bool(torch.isfinite(median).all()) and bool((median > 0.0).all()), "Invalid calibrated median")
    return median


def _role_hidden(
    cache: FrozenFeatureCache, indices: torch.Tensor, permutation: torch.Tensor | None, device: torch.device
) -> torch.Tensor:
    source_indices = indices if permutation is None else permutation[indices]
    return cache.time_hidden[source_indices].to(device)


@torch.no_grad()
def evaluate_module(
    *, module: ConditionalKumaraswamyCDFCalibrator, cache: FrozenFeatureCache,
    base_cache: FrozenBaseTimeCache, count: int, permutation: torch.Tensor | None,
    device: torch.device, batch_size: int, time_scale: float,
    censor_threshold: float | None, observation_mode: str,
) -> dict[str, Any]:
    module.eval()
    primary_total = continuous_total = absolute_total = squared_total = 0.0
    medians: list[torch.Tensor] = []
    shape_values: list[torch.Tensor] = []
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        indices = torch.arange(start, end, dtype=torch.long)
        hidden = _role_hidden(cache, indices, permutation, device)
        location = base_cache.location[start:end].to(device)
        sigma = base_cache.sigma[start:end].to(device)
        target = base_cache.target_dt[start:end].to(device)
        censored = censor_mask(target, threshold=censor_threshold).to(device)
        primary, continuous = calibrated_observation_log_likelihood(
            module=module, hidden=hidden, location=location, sigma=sigma,
            target_dt=target, time_scale=time_scale, is_right_censored=censored,
            observation_mode=observation_mode,
        )
        median = calibrated_time_median(
            module=module, hidden=hidden, location=location, sigma=sigma, time_scale=time_scale
        )
        error = median - target.to(torch.float64)
        primary_total -= float(primary.sum().item())
        continuous_total -= float(continuous.sum().item())
        absolute_total += float(error.abs().sum().item())
        squared_total += float(error.square().sum().item())
        medians.append(median.detach().cpu())
        shape_values.append(module.bounded_log_shapes(hidden).detach().cpu())
    median_tensor = torch.cat(medians).contiguous()
    shapes = torch.cat(shape_values)
    return {
        "count": count,
        "primary_proper_time_nll": primary_total / count,
        "continuous_reference_time_nll": continuous_total / count,
        "time_median_mae": absolute_total / count,
        "time_median_rmse": math.sqrt(squared_total / count),
        "time_median_sha256": _tensor_digest("time_median", median_tensor),
        "bounded_log_shape_min": float(shapes.min().item()),
        "bounded_log_shape_max": float(shapes.max().item()),
        "bounded_log_shape_mean": [float(value) for value in shapes.mean(dim=0).tolist()],
    }


def _finite_optimizer(optimizer: torch.optim.Optimizer) -> None:
    finite_nested_tensors(optimizer.state_dict(), label="Optimizer state")


def train_epoch(
    *, module: ConditionalKumaraswamyCDFCalibrator, cache: FrozenFeatureCache,
    base_cache: FrozenBaseTimeCache, count: int, permutation: torch.Tensor | None,
    optimizer: torch.optim.Optimizer, device: torch.device, batch_size: int,
    time_scale: float, censor_threshold: float | None, observation_mode: str,
    grad_clip: float, seed: int, epoch: int,
) -> dict[str, float]:
    module.train()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1_000_003 + epoch)
    order = torch.randperm(count, generator=generator)
    total = 0.0
    gradient_norms: list[float] = []
    parameters = tuple(module.parameters())
    for start in range(0, count, batch_size):
        indices = order[start : start + batch_size]
        hidden = _role_hidden(cache, indices, permutation, device)
        location = base_cache.location[indices].to(device)
        sigma = base_cache.sigma[indices].to(device)
        target = base_cache.target_dt[indices].to(device)
        censored = censor_mask(target, threshold=censor_threshold).to(device)
        primary, _ = calibrated_observation_log_likelihood(
            module=module, hidden=hidden, location=location, sigma=sigma,
            target_dt=target, time_scale=time_scale, is_right_censored=censored,
            observation_mode=observation_mode,
        )
        loss = -primary.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in module.named_parameters():
            require(parameter.grad is not None, f"Missing calibrator gradient: {name}")
            require(bool(torch.isfinite(parameter.grad).all()), f"Non-finite calibrator gradient: {name}")
        norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        require(bool(torch.isfinite(norm)), "Non-finite calibrator gradient norm")
        optimizer.step()
        _finite_optimizer(optimizer)
        finite_nested_tensors(module.state_dict(), label="Calibrator state")
        total -= float(primary.detach().sum().item())
        gradient_norms.append(float(norm.detach().cpu().item()))
    return {
        "train_primary_proper_time_nll": total / count,
        "pre_clip_gradient_norm_mean": float(np.mean(gradient_norms)),
        "pre_clip_gradient_norm_max": float(np.max(gradient_norms)),
    }


def earliest_strict_minimum(history: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(history), "Empty selection history")
    best = history[0]
    for row in history[1:]:
        require(math.isfinite(float(row["val_primary_proper_time_nll"])), "Non-finite selection metric")
        if float(row["val_primary_proper_time_nll"]) < float(best["val_primary_proper_time_nll"]):
            best = row
    return best


def early_stopping_exhausted(history: list[dict[str, Any]], *, minimum_epochs: int, patience: int) -> bool:
    current = int(history[-1]["epoch"])
    best = int(earliest_strict_minimum(history)["epoch"])
    return current >= minimum_epochs and current - best >= patience


def fit_module(
    *, role: str, module: ConditionalKumaraswamyCDFCalibrator,
    output_dir: Path, train_cache: FrozenFeatureCache,
    validation_cache: FrozenFeatureCache, train_base_cache: FrozenBaseTimeCache,
    validation_base_cache: FrozenBaseTimeCache, train_count: int,
    validation_count: int, train_permutation: torch.Tensor | None,
    validation_permutation: torch.Tensor | None, device: torch.device,
    time_scale: float, censor_threshold: float | None, observation_mode: str,
    settings: Mapping[str, Any], identity: Mapping[str, Any],
    run_epoch_limit: int | None = None,
) -> dict[str, Any]:
    require(role in ROLES, "Unknown calibrator role")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / SELECTED_CHECKPOINT_NAME
    last_path = output_dir / LAST_CHECKPOINT_NAME
    summary_path = output_dir / SUMMARY_NAME
    expected_parameters = 2 if role == "global_shape_control" else 130
    require(sum(parameter.numel() for parameter in module.parameters()) == expected_parameters, "Trainable parameter count drift")
    role_identity = {**identity, "role": role, "train_permutation_sha256": None if train_permutation is None else _tensor_digest("permutation", train_permutation), "validation_permutation_sha256": None if validation_permutation is None else _tensor_digest("permutation", validation_permutation)}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("status") == "success", "Cached role summary incomplete")
        require(summary.get("resume_identity") == role_identity, "Cached role identity drift")
        checkpoint = torch_load_checkpoint(selected_path, map_location="cpu")
        require(checkpoint.get("resume_identity") == role_identity, "Cached selected identity drift")
        module.load_state_dict(checkpoint["module_state_dict"], strict=True)
        replay = evaluate_module(
            module=module, cache=validation_cache, base_cache=validation_base_cache,
            count=validation_count, permutation=validation_permutation, device=device,
            batch_size=int(settings["batch_size"]), time_scale=time_scale,
            censor_threshold=censor_threshold, observation_mode=observation_mode,
        )
        require(math.isclose(float(replay["primary_proper_time_nll"]), float(summary["selected_validation_metrics"]["primary_proper_time_nll"]), rel_tol=0.0, abs_tol=1e-12), "Cached selected replay drift")
        return summary

    optimizer = torch.optim.AdamW(module.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    initial_state = _clone_state(module)
    initial_sha = canonical_state_dict_sha256(initial_state)
    if last_path.exists():
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        require(payload.get("resume_identity") == role_identity, "Resume identity drift")
        module.load_state_dict(payload["module_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload["history"])
        best_state = copy.deepcopy(payload["best_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
    else:
        epoch_zero = evaluate_module(
            module=module, cache=validation_cache, base_cache=validation_base_cache,
            count=validation_count, permutation=validation_permutation, device=device,
            batch_size=int(settings["batch_size"]), time_scale=time_scale,
            censor_threshold=censor_threshold, observation_mode=observation_mode,
        )
        require(epoch_zero["bounded_log_shape_min"] == 0.0 and epoch_zero["bounded_log_shape_max"] == 0.0, "Epoch-zero shape identity failed")
        history = [{
            "epoch": 0,
            "train_primary_proper_time_nll": None,
            "val_primary_proper_time_nll": epoch_zero["primary_proper_time_nll"],
            "val_continuous_reference_time_nll": epoch_zero["continuous_reference_time_nll"],
        }]
        best_state = initial_state
        start_epoch = 1
    planned_epochs = int(settings["epochs"])
    stopped_early = early_stopping_exhausted(history, minimum_epochs=int(settings["minimum_epochs"]), patience=int(settings["early_stopping_patience"]))
    call_last_epoch = planned_epochs
    if run_epoch_limit is not None:
        require(run_epoch_limit >= 0, "run_epoch_limit must be nonnegative")
        call_last_epoch = min(planned_epochs, start_epoch + run_epoch_limit - 1)
    for epoch in range(start_epoch, call_last_epoch + 1):
        if stopped_early:
            break
        train_metrics = train_epoch(
            module=module, cache=train_cache, base_cache=train_base_cache,
            count=train_count, permutation=train_permutation, optimizer=optimizer,
            device=device, batch_size=int(settings["batch_size"]), time_scale=time_scale,
            censor_threshold=censor_threshold, observation_mode=observation_mode,
            grad_clip=float(settings["gradient_clip"]), seed=int(settings["seed"]), epoch=epoch,
        )
        validation_metrics = evaluate_module(
            module=module, cache=validation_cache, base_cache=validation_base_cache,
            count=validation_count, permutation=validation_permutation, device=device,
            batch_size=int(settings["batch_size"]), time_scale=time_scale,
            censor_threshold=censor_threshold, observation_mode=observation_mode,
        )
        history.append({
            "epoch": epoch,
            **train_metrics,
            "val_primary_proper_time_nll": validation_metrics["primary_proper_time_nll"],
            "val_continuous_reference_time_nll": validation_metrics["continuous_reference_time_nll"],
        })
        selected_row = earliest_strict_minimum(history)
        if int(selected_row["epoch"]) == epoch:
            best_state = _clone_state(module)
        current_state = _clone_state(module)
        atomic_torch_save({
            "checkpoint_type": "conditional_cdf_calibrator_resume",
            "checkpoint_schema_version": 1,
            "role": role,
            "epoch": epoch,
            "module_state_dict": current_state,
            "module_state_sha256": canonical_state_dict_sha256(current_state),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_state_dict": best_state,
            "best_state_sha256": canonical_state_dict_sha256(best_state),
            "resume_identity": role_identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }, last_path)
        stopped_early = early_stopping_exhausted(history, minimum_epochs=int(settings["minimum_epochs"]), patience=int(settings["early_stopping_patience"]))
    if int(history[-1]["epoch"]) < planned_epochs and not stopped_early:
        return {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "status": "paused",
            "role": role,
            "completed_epochs": int(history[-1]["epoch"]),
            "planned_epochs": planned_epochs,
            "history": history,
            "resume_identity": role_identity,
            "last_checkpoint_path": str(last_path.resolve()),
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }
    selected_row = earliest_strict_minimum(history)
    module.load_state_dict(best_state, strict=True)
    selected_metrics = evaluate_module(
        module=module, cache=validation_cache, base_cache=validation_base_cache,
        count=validation_count, permutation=validation_permutation, device=device,
        batch_size=int(settings["batch_size"]), time_scale=time_scale,
        censor_threshold=censor_threshold, observation_mode=observation_mode,
    )
    require(math.isclose(float(selected_metrics["primary_proper_time_nll"]), float(selected_row["val_primary_proper_time_nll"]), rel_tol=0.0, abs_tol=1e-12), "Selected state replay drift")
    selected_sha = canonical_state_dict_sha256(best_state)
    atomic_torch_save({
        "checkpoint_type": "selected_conditional_cdf_calibrator",
        "checkpoint_schema_version": 1,
        "role": role,
        "best_epoch": int(selected_row["epoch"]),
        "module_state_dict": best_state,
        "module_state_sha256": selected_sha,
        "validation_metrics": selected_metrics,
        "resume_identity": role_identity,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }, selected_path)
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "success",
        "role": role,
        "best_epoch": int(selected_row["epoch"]),
        "completed_epochs": int(history[-1]["epoch"]),
        "stopped_early": bool(stopped_early),
        "epoch_zero_validation_primary_nll": float(history[0]["val_primary_proper_time_nll"]),
        "selected_validation_metrics": selected_metrics,
        "module_initial_state_sha256": initial_sha,
        "selected_module_state_sha256": selected_sha,
        "trainable_parameter_count": expected_parameters,
        "history": history,
        "resume_identity": role_identity,
        "selected_checkpoint_path": str(selected_path.resolve()),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    save_json(summary_path, summary)
    return summary


def _effective_count(full_count: int, *, max_batches: int | None, batch_size: int) -> int:
    if max_batches is None:
        return full_count
    require(max_batches > 0, "max_batches must be positive")
    return min(full_count, max_batches * batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--aligned-contract", type=Path, default=DEFAULT_ALIGNED_CONTRACT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--run-epoch-limit", type=int, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    contract = load_json_strict(args.contract)
    source_rows = validate_contract(contract)
    require(sha256_file(args.aligned_contract) == contract["candidate"]["source_contract_sha256"], "Aligned contract file digest drift")
    aligned_rows = validate_aligned_contract(load_json_strict(args.aligned_contract))
    require(args.dataset in source_rows and args.dataset in aligned_rows, "Dataset outside contract")
    dataset_spec = {**aligned_rows[args.dataset], **source_rows[args.dataset]}
    partial = any(value is not None for value in (args.max_train_batches, args.max_validation_batches, args.max_epochs, args.run_epoch_limit))
    if partial:
        require(args.allow_partial_contract, "Limits require --allow-partial-contract")
    require(len(args.source_revision) == 40 and set(args.source_revision) <= set("0123456789abcdef"), "source_revision must be a full SHA")
    require(sha256_file(args.data) == dataset_spec["data_sha256"], "Data digest drift")
    require(sha256_file(args.split_manifest) == dataset_spec["split_manifest_sha256"], "Split-manifest digest drift")
    checkpoint_sha = sha256_file(args.checkpoint)
    require(checkpoint_sha == dataset_spec["aligned_B_checkpoint_file_sha256"], "Aligned-B checkpoint digest drift")
    selected_payload = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    require(selected_payload.get("checkpoint_type") == "selected_frozen_lognormal_duration", "Wrong source checkpoint type")
    require(selected_payload.get("model_role") == "B", "Source is not aligned-B")
    require(selected_payload.get("model_state_sha256") == dataset_spec["aligned_B_model_state_sha256"], "Source state digest drift")
    observation_contract = validate_observation_likelihood_contract(dataset_spec["observation_contract"])
    observation_mode = str(observation_contract["mode"])

    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        device_name = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        device_name = str(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_cache = load_verified_feature_cache(
        path=args.feature_cache_dir / "train_features.pt", split="train",
        dataset_name=args.dataset, dataset_spec=dataset_spec,
        selected_payload=selected_payload, require_quantity=False,
    )
    validation_cache = load_verified_feature_cache(
        path=args.feature_cache_dir / "validation_features.pt", split="validation",
        dataset_name=args.dataset, dataset_spec=dataset_spec,
        selected_payload=selected_payload, require_quantity=True,
    )
    source_model = build_candidate_from_selected_checkpoint(selected_payload).to(device)
    source_state_sha, trainable_source = freeze_source_model(source_model)
    require(not trainable_source and source_state_sha == dataset_spec["aligned_B_model_state_sha256"], "Source freeze boundary failed")
    settings = dict(contract["future_optimization_contract"])
    if args.max_epochs is not None:
        require(args.max_epochs >= 0, "max_epochs must be nonnegative")
        settings["epochs"] = min(int(settings["epochs"]), int(args.max_epochs))
    batch_size = int(settings["batch_size"])
    train_count = _effective_count(train_cache.count, max_batches=args.max_train_batches, batch_size=batch_size)
    validation_count = _effective_count(validation_cache.count, max_batches=args.max_validation_batches, batch_size=batch_size)
    full_data = train_count == train_cache.count and validation_count == validation_cache.count
    qualified_full_fit = full_data and args.max_epochs is None and args.run_epoch_limit is None
    train_base = derive_frozen_base_time_cache(model=source_model, cache=train_cache, device=device, batch_size=batch_size)
    validation_base = derive_frozen_base_time_cache(model=source_model, cache=validation_cache, device=device, batch_size=batch_size)
    time_scale = float(dataset_spec["train_time_scale"])
    censor_threshold = resolve_censor_threshold(args.dataset, dataset_spec)

    torch.manual_seed(int(settings["seed"]))
    candidate = ConditionalKumaraswamyCDFCalibrator(hidden_dim=64)
    candidate_initial = _clone_state(candidate)
    permuted_control = ConditionalKumaraswamyCDFCalibrator(hidden_dim=64)
    permuted_control.load_state_dict(candidate_initial, strict=True)
    global_control = GlobalKumaraswamyCDFCalibrator(hidden_dim=64)
    require(canonical_state_dict_sha256(candidate.state_dict()) == canonical_state_dict_sha256(permuted_control.state_dict()), "Candidate/control initialization drift")
    train_permutation = deterministic_hidden_permutation(train_count)
    validation_permutation = deterministic_hidden_permutation(validation_count)

    identity = {
        "contract_sha256": sha256_file(args.contract),
        "aligned_contract_sha256": sha256_file(args.aligned_contract),
        "source_revision": args.source_revision,
        "dataset": args.dataset,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_model_state_sha256": source_state_sha,
        "train_cache_sha256": train_cache.digest(),
        "validation_cache_sha256": validation_cache.digest(),
        "train_count": train_count,
        "validation_count": validation_count,
        "settings": settings,
        "observation_likelihood_contract": observation_contract,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    common = dict(
        train_cache=train_cache, validation_cache=validation_cache,
        train_base_cache=train_base, validation_base_cache=validation_base,
        train_count=train_count, validation_count=validation_count, device=device,
        time_scale=time_scale, censor_threshold=censor_threshold,
        observation_mode=observation_mode, settings=settings, identity=identity,
        run_epoch_limit=args.run_epoch_limit,
    )
    candidate_summary = fit_module(
        role="candidate", module=candidate, output_dir=args.output_dir / "candidate",
        train_permutation=None, validation_permutation=None, **common,
    )
    if candidate_summary.get("status") != "success":
        return candidate_summary
    global_summary = fit_module(
        role="global_shape_control", module=global_control,
        output_dir=args.output_dir / "global_shape_control",
        train_permutation=None, validation_permutation=None, **common,
    )
    if global_summary.get("status") != "success":
        return global_summary
    permuted_summary = fit_module(
        role="permuted_hidden_control", module=permuted_control,
        output_dir=args.output_dir / "permuted_hidden_control",
        train_permutation=train_permutation, validation_permutation=validation_permutation,
        **common,
    )
    if permuted_summary.get("status") != "success":
        return permuted_summary

    require(canonical_state_dict_sha256(source_model.state_dict()) == source_state_sha, "Source state changed")
    require(all(parameter.grad is None for parameter in source_model.parameters()), "Source received gradients")
    assert validation_cache.source_quantity_prediction is not None
    assert validation_cache.target_quantity is not None
    quantity_before = validation_cache.source_quantity_prediction[:validation_count].contiguous()
    runtime_quantity = cached_quantity_predictions(model=source_model, cache=validation_cache, device=device, batch_size=batch_size)[:validation_count].contiguous()
    require(torch.equal(runtime_quantity, quantity_before), "Quantity prediction is not bitwise identical")
    quantity_digest = tensor_sha256("quantity_prediction", runtime_quantity)
    if full_data:
        require(quantity_digest == dataset_spec["aligned_B_quantity_prediction_sha256"], "Quantity prediction digest drift")

    base_model = ConditionalKumaraswamyCDFCalibrator(hidden_dim=64).to(device)
    base_metrics = evaluate_module(
        module=base_model, cache=validation_cache, base_cache=validation_base,
        count=validation_count, permutation=None, device=device, batch_size=batch_size,
        time_scale=time_scale, censor_threshold=censor_threshold, observation_mode=observation_mode,
    )
    if full_data:
        require(math.isclose(float(base_metrics["primary_proper_time_nll"]), float(dataset_spec["aligned_B_validation_primary_nll"]), rel_tol=0.0, abs_tol=1e-6), "Aligned-B primary replay drift")
        require(math.isclose(float(base_metrics["continuous_reference_time_nll"]), float(dataset_spec["aligned_B_validation_continuous_nll"]), rel_tol=0.0, abs_tol=1e-6), "Aligned-B continuous replay drift")
    for role_summary in (candidate_summary, global_summary, permuted_summary):
        require(role_summary["epoch_zero_validation_primary_nll"] == base_metrics["primary_proper_time_nll"], "Epoch-zero base identity drift")
    candidate_metrics = candidate_summary["selected_validation_metrics"]
    global_metrics = global_summary["selected_validation_metrics"]
    permuted_metrics = permuted_summary["selected_validation_metrics"]
    deltas = {
        "candidate_primary_improvement_over_aligned_B": float(base_metrics["primary_proper_time_nll"]) - float(candidate_metrics["primary_proper_time_nll"]),
        "candidate_minus_aligned_A_primary_nll": float(candidate_metrics["primary_proper_time_nll"]) - float(dataset_spec["aligned_A_validation_primary_nll"]),
        "candidate_primary_improvement_over_global_control": float(global_metrics["primary_proper_time_nll"]) - float(candidate_metrics["primary_proper_time_nll"]),
        "candidate_primary_improvement_over_permuted_hidden_control": float(permuted_metrics["primary_proper_time_nll"]) - float(candidate_metrics["primary_proper_time_nll"]),
        "candidate_minus_aligned_B_continuous_nll": float(candidate_metrics["continuous_reference_time_nll"]) - float(base_metrics["continuous_reference_time_nll"]),
        "candidate_time_median_mae_relative_change": float(candidate_metrics["time_median_mae"]) / float(base_metrics["time_median_mae"]) - 1.0,
        "candidate_time_median_rmse_relative_change": float(candidate_metrics["time_median_rmse"]) / float(base_metrics["time_median_rmse"]) - 1.0,
    }
    acceptance = contract["future_acceptance"]
    gates: dict[str, bool | None] = {
        "candidate_improves_aligned_B_by_0_005": None,
        "candidate_within_aligned_A_plus_0_01": None,
        "candidate_beats_global_by_0_005": None,
        "candidate_beats_permuted_by_0_005": None,
        "continuous_reference_within_aligned_B_plus_0_01": None,
        "time_median_mae_worsening_within_2_percent": None,
        "time_median_rmse_worsening_within_2_percent": None,
        "quantity_prediction_bitwise_identical": True,
        "source_model_state_unchanged": True,
    }
    if qualified_full_fit:
        gates.update(
            candidate_improves_aligned_B_by_0_005=deltas["candidate_primary_improvement_over_aligned_B"] >= float(acceptance["minimum_candidate_improvement_over_aligned_B_primary_nll_each_dataset"]),
            candidate_within_aligned_A_plus_0_01=deltas["candidate_minus_aligned_A_primary_nll"] <= float(acceptance["maximum_candidate_minus_aligned_A_primary_nll_each_dataset"]),
            candidate_beats_global_by_0_005=deltas["candidate_primary_improvement_over_global_control"] >= float(acceptance["minimum_candidate_improvement_over_global_control_primary_nll_each_dataset"]),
            candidate_beats_permuted_by_0_005=deltas["candidate_primary_improvement_over_permuted_hidden_control"] >= float(acceptance["minimum_candidate_improvement_over_permuted_hidden_control_primary_nll_each_dataset"]),
            continuous_reference_within_aligned_B_plus_0_01=deltas["candidate_minus_aligned_B_continuous_nll"] <= float(acceptance["maximum_continuous_reference_nll_worsening_from_aligned_B_each_dataset"]),
            time_median_mae_worsening_within_2_percent=deltas["candidate_time_median_mae_relative_change"] <= float(acceptance["maximum_time_median_mae_relative_worsening_from_aligned_B_each_dataset"]),
            time_median_rmse_worsening_within_2_percent=deltas["candidate_time_median_rmse_relative_change"] <= float(acceptance["maximum_time_median_rmse_relative_worsening_from_aligned_B_each_dataset"]),
        )
    performance_gates = [value for name, value in gates.items() if name.startswith("candidate_") or name.startswith("continuous_") or name.startswith("time_median_")]
    acceptance_status = "passed" if qualified_full_fit and all(value is True for value in performance_gates) else "failed" if qualified_full_fit else "not_evaluated_execution_check"
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": sha256_file(args.contract),
        "source_revision": args.source_revision,
        "status": "success",
        "dataset": args.dataset,
        "seed": int(settings["seed"]),
        "observation_likelihood_contract": observation_contract,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_model_state_sha256": source_state_sha,
        "source_model_state_unchanged": True,
        "train_feature_cache_sha256": train_cache.digest(),
        "validation_feature_cache_sha256": validation_cache.digest(),
        "train_count": train_count,
        "validation_count": validation_count,
        "qualified_full_data": full_data,
        "qualified_full_fit": qualified_full_fit,
        "base_aligned_B_validation_metrics": base_metrics,
        "candidate": candidate_summary,
        "global_shape_control": global_summary,
        "permuted_hidden_control": permuted_summary,
        "candidate_permuted_initial_state_identical": True,
        "train_permutation_sha256": _tensor_digest("permutation", train_permutation),
        "validation_permutation_sha256": _tensor_digest("permutation", validation_permutation),
        "quantity_prediction_bitwise_identical": True,
        "quantity_prediction_sha256": quantity_digest,
        "quantity_metrics": quantity_metrics(runtime_quantity, validation_cache.target_quantity[:validation_count]),
        "acceptance_observed_deltas": deltas,
        "acceptance_gates": gates,
        "acceptance_status": acceptance_status,
        "taxi_first_stop_triggered": args.dataset == "yellow_trip_hourly" and qualified_full_fit and acceptance_status != "passed",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "runtime": {
            "requested_device": str(device),
            "device_name": device_name,
            "cuda_available": torch.cuda.is_available(),
            "elapsed_seconds": time.perf_counter() - started_at,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        },
    }
    save_json(args.output_dir / SUMMARY_NAME, summary)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
