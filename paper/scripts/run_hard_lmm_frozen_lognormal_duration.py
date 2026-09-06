#!/usr/bin/env python3
"""Fit a proper heteroscedastic log-normal duration head on frozen B states.

The audited raw-RMSE-selected Hard-LMM B checkpoint supplies the encoder,
memories, and quantity decoder. Those tensors are copied exactly and never
optimized. Only the conditional log-normal location and scale projections are
fit on detached train states and selected by validation proper time NLL.

Instacart delta_t == 30 targets are treated as right censored. No other
dataset has a censoring threshold. Held-out rows are removed before any eager
materialization.
"""

from __future__ import annotations

import argparse
import copy
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
import polars as pl
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
    SharedTimeCountModel,
    inverse_softplus,
)
from paper.scripts.count_aware_tpp_backbone.core import (
    load_train_validation_frame,
    prepare_count_frame,
)
from paper.scripts.run_count_aware_tpp_backbone_control import (
    derive_train_time_contract,
)
from paper.scripts.run_hard_lmm_time_head_refit import (
    FrozenFeatureCache,
    build_source_model,
    cached_quantity_predictions,
    clone_state_dict,
    exact_target_population_contract,
    finite_nested_tensors,
    finite_tensor_mapping,
    load_or_extract_feature_cache,
    quantity_metrics,
    require,
    save_json,
    sha256_file,
    stratified_quantity_metrics,
    tensor_sha256,
    validate_source_checkpoint,
    validate_target_population,
)
from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "hard_lmm_frozen_lognormal_duration_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/hard_lmm_frozen_lognormal_duration_v1.json"
)
SOURCE_BACKBONE = "titantpp"
SOURCE_VARIANT = LOG_MSE_VARIANT
CANDIDATE_TIME_HEAD_MODE = TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION
LEGACY_TIME_HEAD_PARAMETER_NAMES = ("v_t.weight", "b_t", "w_raw")
TIME_HEAD_PARAMETER_NAMES = (
    "v_t.weight",
    "b_t",
    "w_raw",
    "time_scale_weight.weight",
)
SELECTION_RULE = (
    "earliest_strict_finite_minimum_validation_proper_time_nll"
)
SELECTED_CHECKPOINT_NAME = "best_validation_proper_time_nll_model.pt"
LAST_CHECKPOINT_NAME = "last_epoch_state.pt"
SUMMARY_NAME = "summary.json"
INSTACART_DATASET = "insta_market_basket"
INSTACART_RIGHT_CENSOR_THRESHOLD = 30.0


def _state_partition(
    state: Mapping[str, torch.Tensor], *, time_head: bool
) -> dict[str, torch.Tensor]:
    names = set(TIME_HEAD_PARAMETER_NAMES)
    return {
        name: value
        for name, value in state.items()
        if (name in names) is time_head
    }


def state_partition_sha256(
    state: Mapping[str, torch.Tensor], *, time_head: bool
) -> str:
    partition = _state_partition(state, time_head=time_head)
    require(bool(partition), "Requested model-state partition is empty")
    return canonical_state_dict_sha256(partition)


def target_dt_sha256(tensor: torch.Tensor) -> str:
    """Hash durations under the frozen contract's float64 byte convention."""
    value = (
        np.asarray(tensor.detach().cpu().numpy(), dtype="<f8")
        .reshape(-1)
    )
    digest = hashlib.sha256()
    digest.update(b"frozen_lognormal_target_dt_v1\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def censor_mask_sha256(tensor: torch.Tensor) -> str:
    """Hash the censor mask under its pinned uint8 byte convention."""
    value = (
        np.asarray(tensor.detach().cpu().numpy(), dtype=np.uint8)
        .reshape(-1)
    )
    digest = hashlib.sha256()
    digest.update(b"frozen_lognormal_censor_mask_v1\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_admitted_frame(path: Path) -> pl.DataFrame:
    """Materialize train/validation only and reject any scope drift."""
    frame = load_train_validation_frame(path)
    admitted = set(frame["chronological_split"].unique().to_list())
    require(
        admitted <= {"train", "validation"},
        "Held-out rows entered the frame",
    )
    require(bool(admitted), "Train/validation frame is empty")
    return frame


def resolve_censor_threshold(
    dataset: str,
    dataset_spec: Mapping[str, Any] | None = None,
) -> float | None:
    """Enforce the one predeclared observation-censoring rule."""
    declared = (
        None
        if dataset_spec is None
        else dataset_spec.get("right_censor_threshold")
    )
    if dataset == INSTACART_DATASET:
        if dataset_spec is not None:
            require(declared is not None, "Instacart censor threshold is missing")
            require(
                math.isclose(
                    float(declared),
                    INSTACART_RIGHT_CENSOR_THRESHOLD,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                "Instacart censor threshold must be exactly 30",
            )
        return INSTACART_RIGHT_CENSOR_THRESHOLD
    require(
        declared is None,
        f"Right censoring is not permitted for dataset {dataset!r}",
    )
    return None


def censor_mask(
    target_dt: torch.Tensor,
    *,
    threshold: float | None,
) -> torch.Tensor:
    require(target_dt.ndim == 1, "target_dt must be rank one")
    require(
        bool(torch.isfinite(target_dt).all()),
        "Non-finite duration target",
    )
    require(
        bool((target_dt > 0.0).all()),
        "Duration targets must be strictly positive",
    )
    if threshold is None:
        return torch.zeros_like(target_dt, dtype=torch.bool)
    require(
        math.isfinite(threshold) and threshold > 0.0,
        "Invalid censor threshold",
    )
    require(
        bool((target_dt <= threshold).all()),
        "A duration target exceeds the declared censor threshold",
    )
    return target_dt == float(threshold)


def proper_time_log_likelihood(
    model: SharedTimeCountModel,
    hidden: torch.Tensor,
    target_dt: torch.Tensor,
    *,
    is_right_censored: torch.Tensor,
) -> torch.Tensor:
    """Return the censor-aware log likelihood, calculated in float64.

    The candidate model owns the normalized log-normal formula and guarantees
    float64 calculation from its affine maps onward. Cached hidden states stay
    float32 so replay through the frozen quantity head remains bitwise equal to
    B.
    """
    require(
        model.time_head_mode == CANDIDATE_TIME_HEAD_MODE,
        "Expected the heteroscedastic log-normal duration head",
    )
    require(hidden.ndim == 2, "hidden must be rank two")
    require(target_dt.ndim == 1, "target_dt must be rank one")
    require(
        hidden.shape[0] == target_dt.shape[0],
        "Hidden and duration batch sizes differ",
    )
    require(
        is_right_censored.shape == target_dt.shape,
        "Censor mask shape does not match duration targets",
    )
    require(
        is_right_censored.dtype == torch.bool,
        "Censor mask must be boolean",
    )
    require(
        bool(torch.isfinite(hidden).all()),
        "Non-finite cached hidden state",
    )
    require(
        bool(torch.isfinite(target_dt).all())
        and bool((target_dt > 0.0).all()),
        "Duration targets must be finite and strictly positive",
    )
    log_density = model.log_f_dt(hidden, target_dt)
    require(
        log_density.dtype == torch.float64,
        "Proper duration likelihood must be calculated in float64",
    )
    if not bool(is_right_censored.any()):
        return log_density
    log_survival = model.log_survival_dt(hidden, target_dt)
    require(
        log_survival.dtype == torch.float64,
        "Proper duration survival must be calculated in float64",
    )
    return torch.where(is_right_censored, log_survival, log_density)


def predict_time_median_float64(
    model: SharedTimeCountModel,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Compute the conditional log-normal median in contract dtype."""
    prediction = model.predict_time_median(hidden)
    require(
        prediction.dtype == torch.float64,
        "Conditional duration median must be calculated in float64",
    )
    return prediction


def freeze_time_head_only(
    model: SharedTimeCountModel,
) -> tuple[torch.nn.Parameter, ...]:
    """Freeze B's state/quantity path and expose exactly 130 parameters."""
    require(
        model.time_head_mode == CANDIDATE_TIME_HEAD_MODE,
        "Expected the heteroscedastic log-normal duration head",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    named = dict(model.named_parameters())
    require(
        set(TIME_HEAD_PARAMETER_NAMES).issubset(named),
        "Candidate does not expose the complete duration-head boundary",
    )
    for name in TIME_HEAD_PARAMETER_NAMES:
        named[name].requires_grad_(True)
    trainable_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    require(
        set(trainable_names) == set(TIME_HEAD_PARAMETER_NAMES)
        and len(trainable_names) == len(TIME_HEAD_PARAMETER_NAMES),
        f"Unexpected trainable parameters: {trainable_names}",
    )
    parameters = tuple(named[name] for name in TIME_HEAD_PARAMETER_NAMES)
    require(
        sum(parameter.numel() for parameter in parameters) == 130,
        "The frozen-B duration head must contain exactly 130 parameters",
    )
    return parameters


def assert_time_head_gradient_boundary(model: SharedTimeCountModel) -> None:
    named = dict(model.named_parameters())
    for name, parameter in named.items():
        if name in TIME_HEAD_PARAMETER_NAMES:
            require(
                parameter.grad is not None,
                f"Missing time-head gradient: {name}",
            )
            require(
                bool(torch.isfinite(parameter.grad).all()),
                f"Non-finite time-head gradient: {name}",
            )
        else:
            require(
                parameter.grad is None,
                f"Frozen parameter received gradient: {name}",
            )


def train_time_statistics_from_contract(
    dataset_spec: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact train-derived initialization against pinned constants."""
    require(
        observed["statistics_source_split"] == "train",
        "Duration initialization must use train targets only",
    )
    pinned = {
        "time_scale": float(dataset_spec["train_time_scale"]),
        "target_log_scaled_mean": float(
            dataset_spec["train_log_scaled_mean"]
        ),
        "target_log_scaled_std": float(
            dataset_spec["train_log_scaled_std"]
        ),
    }
    for name, expected in pinned.items():
        actual = float(observed[name])
        require(math.isfinite(actual), f"Non-finite train statistic: {name}")
        require(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12),
            f"Pinned train statistic drift: {name}",
        )
    require(pinned["time_scale"] > 0.0, "Train time scale must be positive")
    require(
        pinned["target_log_scaled_std"] > 0.0,
        "Train log-duration standard deviation must be positive",
    )
    return {**dict(observed), **pinned}


def build_frozen_lognormal_candidate(
    source_payload: Mapping[str, Any],
    *,
    train_time_statistics: Mapping[str, Any],
    time_sigma_floor: float,
    source_backbone: str = SOURCE_BACKBONE,
    source_variant: str = SOURCE_VARIANT,
    max_seq_len: int | None = None,
    training_stage: str = "frozen_B_posthoc_duration_refit",
) -> tuple[SharedTimeCountModel, dict[str, Any], dict[str, Any]]:
    """Build Hetero-LN and transplant every non-time source tensor exactly.

    The optional source arguments keep the original Frozen-B contract as the
    default while allowing the matched A/RMTPP/THP audit to reuse the exact
    same 130-parameter duration head and state-copy implementation.
    """
    initial_scale = float(
        train_time_statistics["target_log_scaled_std"]
    )
    require(
        initial_scale > time_sigma_floor,
        "Train log-duration scale must exceed time_sigma_floor",
    )
    encoder = source_payload["encoder_config"]
    interface = source_payload["interface_meta"]
    hidden_dim = encoder.get("d_model", encoder.get("hidden_dim"))
    require(
        type(hidden_dim) is int and int(hidden_dim) > 0,
        "Source hidden dimension is missing",
    )
    source_state = source_payload["model_state_dict"]
    require(isinstance(source_state, Mapping), "Source model state is missing")
    source_location_weight = source_state.get("v_t.weight")
    require(
        isinstance(source_location_weight, torch.Tensor)
        and source_location_weight.ndim == 2
        and tuple(source_location_weight.shape) == (1, int(hidden_dim)),
        "Source hidden dimension disagrees with the time-head state",
    )
    resolved_max_seq_len = encoder.get("max_len", max_seq_len)
    require(
        type(resolved_max_seq_len) is int
        and int(resolved_max_seq_len) > 0,
        "Source maximum sequence length is missing",
    )
    candidate, candidate_encoder = build_count_aware_model(
        source_backbone,
        hidden_dim=int(hidden_dim),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface["train_target_std"]),
        max_seq_len=int(resolved_max_seq_len),
        quantity_variant=source_variant,
        lambda_tail=0.0,
        time_head_mode=CANDIDATE_TIME_HEAD_MODE,
        time_scale=float(train_time_statistics["time_scale"]),
        time_initial_location=float(
            train_time_statistics["target_log_scaled_mean"]
        ),
        time_initial_scale=initial_scale,
        time_sigma_floor=float(time_sigma_floor),
    )
    candidate_state = clone_state_dict(candidate)
    expected_candidate_keys = set(source_state) | {
        "time_scale_weight.weight"
    }
    require(
        set(candidate_state) == expected_candidate_keys,
        "Candidate state schema differs from B plus conditional scale",
    )
    for name, value in source_state.items():
        if name in LEGACY_TIME_HEAD_PARAMETER_NAMES:
            continue
        require(
            candidate_state[name].shape == value.shape
            and candidate_state[name].dtype == value.dtype,
            f"Non-time tensor schema drift: {name}",
        )
        candidate_state[name] = value.detach().cpu().clone()

    candidate_state["v_t.weight"].zero_()
    candidate_state["time_scale_weight.weight"].zero_()
    candidate_state["b_t"].fill_(
        float(train_time_statistics["target_log_scaled_mean"])
    )
    candidate_state["w_raw"].fill_(
        inverse_softplus(initial_scale - float(time_sigma_floor))
    )
    candidate.load_state_dict(candidate_state, strict=True)
    candidate.eval()

    source_non_time_sha256 = state_partition_sha256(
        source_state, time_head=False
    )
    candidate_non_time_sha256 = state_partition_sha256(
        candidate.state_dict(), time_head=False
    )
    require(
        source_non_time_sha256 == candidate_non_time_sha256,
        "Candidate changed a non-time B tensor",
    )
    require(
        torch.equal(
            candidate.v_t.weight,
            torch.zeros_like(candidate.v_t.weight),
        ),
        "Location weight is not zero initialized",
    )
    require(
        torch.equal(
            candidate.time_scale_weight.weight,
            torch.zeros_like(candidate.time_scale_weight.weight),
        ),
        "Scale weight is not zero initialized",
    )
    probe = torch.zeros(2, int(hidden_dim))
    require(
        torch.allclose(
            candidate.positive_time_sigma(probe),
            torch.full((2,), initial_scale),
            rtol=0.0,
            atol=1e-7,
        ),
        "Initial sigma differs from the train log-duration std",
    )

    candidate_interface = copy.deepcopy(interface)
    candidate_interface["time_head"] = {
        **candidate.time_head_contract(),
        "statistics_source_split": "train",
        "train_time_statistics": dict(train_time_statistics),
        "training_stage": training_stage,
    }
    metadata = {
        "encoder_config": candidate_encoder,
        "interface_meta": candidate_interface,
        "source_non_time_state_sha256": source_non_time_sha256,
        "candidate_initial_state_sha256": (
            canonical_state_dict_sha256(candidate.state_dict())
        ),
        "candidate_initial_time_head_state_sha256": (
            state_partition_sha256(candidate.state_dict(), time_head=True)
        ),
    }
    return candidate, candidate_encoder, metadata


def validate_cache_observation_contract(
    cache: FrozenFeatureCache,
    *,
    dataset_spec: Mapping[str, Any],
    split: str,
    censor_threshold: float | None,
) -> dict[str, Any]:
    """Bind the cached targets to the pinned duration and censor identities."""
    require(split in {"train", "validation"}, "Unsupported cache split")
    cache.validate(require_quantity=split == "validation")
    mask = censor_mask(cache.target_dt, threshold=censor_threshold)
    dt_hash = target_dt_sha256(cache.target_dt)
    mask_hash = censor_mask_sha256(mask)
    require(
        dt_hash == dataset_spec[f"expected_{split}_target_dt_sha256"],
        f"{split} duration-target digest drift",
    )
    require(
        mask_hash == dataset_spec[f"expected_{split}_censor_mask_sha256"],
        f"{split} censor-mask digest drift",
    )
    expected_count = int(
        dataset_spec[f"expected_{split}_censored_targets"]
    )
    actual_count = int(mask.sum().item())
    require(
        actual_count == expected_count,
        f"{split} censored-target count drift",
    )
    return {
        "split": split,
        "target_count": cache.count,
        "target_dt_sha256": dt_hash,
        "right_censor_threshold": censor_threshold,
        "right_censored_count": actual_count,
        "censor_mask_sha256": mask_hash,
    }


def _time_metrics_from_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    is_right_censored: torch.Tensor,
) -> dict[str, Any]:
    prediction = prediction.to(torch.float64)
    target = target.to(torch.float64)
    error = prediction - target
    uncensored = ~is_right_censored.cpu()
    coded_mae = float(error.abs().mean().item())
    coded_rmse = float(torch.square(error).mean().sqrt().item())
    result: dict[str, Any] = {
        "time_median_mae_against_recorded_target": coded_mae,
        "time_median_rmse_against_recorded_target": coded_rmse,
        "time_median_observed_target_semantics": (
            "recorded_censor_threshold_is_a_lower_bound_for_"
            "right_censored_rows_not_an_exact_duration"
        ),
        "recorded_target_metric_includes_right_censored_lower_bounds": bool(
            is_right_censored.any()
        ),
        "uncensored_count": int(uncensored.sum().item()),
    }
    if bool(uncensored.any()):
        uncensored_error = error[uncensored]
        uncensored_mae = float(
            uncensored_error.abs().mean().item()
        )
        uncensored_rmse = float(
            torch.square(uncensored_error).mean().sqrt().item()
        )
        result.update(
            time_median_mae=uncensored_mae,
            time_median_rmse=uncensored_rmse,
            time_median_metric_scope="uncensored_targets_only",
            uncensored_time_median_mae=uncensored_mae,
            uncensored_time_median_rmse=uncensored_rmse,
        )
    else:
        result.update(
            time_median_mae=None,
            time_median_rmse=None,
            time_median_metric_scope="uncensored_targets_only",
            uncensored_time_median_mae=None,
            uncensored_time_median_rmse=None,
        )
    censored = ~uncensored
    result["median_below_censor_threshold_rate"] = (
        float((prediction[censored] < target[censored]).double().mean().item())
        if bool(censored.any())
        else None
    )
    return result


@torch.no_grad()
def evaluate_cached_time_metrics(
    *,
    model: SharedTimeCountModel,
    cache: FrozenFeatureCache,
    device: str | torch.device,
    batch_size: int,
    censor_threshold: float | None,
) -> dict[str, Any]:
    cache.validate(require_quantity=False)
    full_censor_mask = censor_mask(
        cache.target_dt, threshold=censor_threshold
    )
    model.eval()
    total_nll = 0.0
    count = 0
    predictions: list[torch.Tensor] = []
    for start in range(0, cache.count, batch_size):
        end = min(start + batch_size, cache.count)
        hidden = cache.time_hidden[start:end].to(device)
        target_dt = cache.target_dt[start:end].to(device)
        batch_censor_mask = full_censor_mask[start:end].to(device)
        log_likelihood = proper_time_log_likelihood(
            model,
            hidden,
            target_dt,
            is_right_censored=batch_censor_mask,
        )
        require(
            bool(torch.isfinite(log_likelihood).all()),
            "Non-finite proper Time NLL",
        )
        prediction = predict_time_median_float64(model, hidden)
        require(
            bool(torch.isfinite(prediction).all()),
            "Non-finite time median",
        )
        require(
            bool((prediction > 0.0).all()),
            "Time medians must be positive",
        )
        total_nll -= float(
            log_likelihood.to(torch.float64).sum().item()
        )
        count += int(log_likelihood.numel())
        predictions.append(prediction.detach().cpu())
    require(count == cache.count, "Time metric cache count drift")
    proper_nll = total_nll / count
    require(
        math.isfinite(proper_nll),
        "Validation proper Time NLL is non-finite",
    )
    result = {
        "proper_time_nll": proper_nll,
        "count": count,
        "right_censored_count": int(full_censor_mask.sum().item()),
        "right_censor_threshold": censor_threshold,
    }
    result.update(
        _time_metrics_from_tensors(
            torch.cat(predictions, dim=0).contiguous(),
            cache.target_dt,
            full_censor_mask,
        )
    )
    return result


def earliest_strict_minimum(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_value = float("inf")
    for row in history:
        value = float(row["val_proper_time_nll"])
        require(
            math.isfinite(value),
            "History contains non-finite validation NLL",
        )
        if value < best_value:
            best = row
            best_value = value
    if best is None:
        raise ValueError(
            "No finite validation proper Time NLL candidate"
        )
    return best


def early_stopping_exhausted(
    history: list[dict[str, Any]],
    *,
    min_epochs: int,
    patience: int,
) -> bool:
    current_epoch = int(history[-1]["epoch"])
    best_epoch = int(earliest_strict_minimum(history)["epoch"])
    return (
        current_epoch >= min_epochs
        and current_epoch - best_epoch >= patience
    )


def _finite_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> None:
    finite_nested_tensors(
        optimizer.state_dict(), label="Optimizer state"
    )


def _train_cached_epoch(
    *,
    model: SharedTimeCountModel,
    cache: FrozenFeatureCache,
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    device: str | torch.device,
    batch_size: int,
    grad_clip: float,
    seed: int,
    epoch: int,
    censor_threshold: float | None,
) -> dict[str, float]:
    require(
        not model.training,
        "Frozen encoder must remain in evaluation mode",
    )
    full_censor_mask = censor_mask(
        cache.target_dt, threshold=censor_threshold
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + int(epoch))
    order = torch.randperm(cache.count, generator=generator)
    total = 0.0
    count = 0
    pre_clip_norms: list[float] = []
    for start in range(0, cache.count, batch_size):
        indices = order[start : start + batch_size]
        hidden = cache.time_hidden[indices].to(device)
        target_dt = cache.target_dt[indices].to(device)
        batch_censor_mask = full_censor_mask[indices].to(device)
        losses = -proper_time_log_likelihood(
            model,
            hidden,
            target_dt,
            is_right_censored=batch_censor_mask,
        )
        require(
            bool(torch.isfinite(losses).all()),
            "Non-finite train proper Time NLL",
        )
        loss = losses.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert_time_head_gradient_boundary(model)
        pre_clip = torch.nn.utils.clip_grad_norm_(
            parameters, grad_clip
        )
        require(
            bool(torch.isfinite(pre_clip)),
            "Non-finite duration-head gradient norm",
        )
        optimizer.step()
        _finite_optimizer_state(optimizer)
        finite_tensor_mapping(model.state_dict(), label="Model state")
        total += float(
            losses.detach().to(torch.float64).sum().item()
        )
        count += int(losses.numel())
        pre_clip_norms.append(
            float(pre_clip.detach().cpu().item())
        )
    require(count == cache.count, "Train cache count drift")
    return {
        "train_proper_time_nll": total / count,
        "pre_clip_gradient_norm_mean": float(
            np.mean(pre_clip_norms)
        ),
        "pre_clip_gradient_norm_max": float(
            np.max(pre_clip_norms)
        ),
    }


def _resume_identity(
    *,
    contract_sha256: str,
    dataset: str,
    source_checkpoint_sha256: str,
    source_state_sha256: str,
    source_non_time_state_sha256: str,
    candidate_initial_state_sha256: str,
    train_cache: FrozenFeatureCache,
    validation_cache: FrozenFeatureCache,
    train_time_statistics: Mapping[str, Any],
    censor_threshold: float | None,
    calibration_source_revision: str,
    seed: int,
    planned_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    quantity_replay_batch_size: int,
    grad_clip: float,
    min_epochs: int,
    patience: int,
    contract_id: str = CONTRACT_ID,
    model_role: str = "B",
    source_backbone: str = SOURCE_BACKBONE,
    source_variant: str = SOURCE_VARIANT,
) -> dict[str, Any]:
    train_mask = censor_mask(
        train_cache.target_dt, threshold=censor_threshold
    )
    validation_mask = censor_mask(
        validation_cache.target_dt, threshold=censor_threshold
    )
    identity = {
        "schema_version": 1,
        "contract_id": contract_id,
        "contract_sha256": contract_sha256,
        "dataset": dataset,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "source_non_time_state_sha256": (
            source_non_time_state_sha256
        ),
        "candidate_initial_state_sha256": (
            candidate_initial_state_sha256
        ),
        "train_cache_sha256": train_cache.digest(),
        "validation_cache_sha256": validation_cache.digest(),
        "train_cache_count": train_cache.count,
        "validation_cache_count": validation_cache.count,
        "train_target_dt_sha256": target_dt_sha256(
            train_cache.target_dt
        ),
        "validation_target_dt_sha256": target_dt_sha256(
            validation_cache.target_dt
        ),
        "train_censor_mask_sha256": censor_mask_sha256(train_mask),
        "validation_censor_mask_sha256": censor_mask_sha256(
            validation_mask
        ),
        "right_censor_threshold": censor_threshold,
        "train_time_statistics": dict(train_time_statistics),
        "calibration_source_revision": calibration_source_revision,
        "seed": int(seed),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "cached_state_batch_size": int(batch_size),
        "quantity_replay_batch_size": int(
            quantity_replay_batch_size
        ),
        "gradient_clip": float(grad_clip),
        "minimum_epochs": int(min_epochs),
        "early_stopping_patience": int(patience),
        "planned_epochs": int(planned_epochs),
        "selection": SELECTION_RULE,
        "trainable_parameter_names": list(
            TIME_HEAD_PARAMETER_NAMES
        ),
        "time_head_mode": CANDIDATE_TIME_HEAD_MODE,
        "likelihood": "censor_aware_proper_lognormal_duration",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    if contract_id != CONTRACT_ID:
        identity["model_role"] = model_role
        identity["source_backbone"] = source_backbone
        identity["source_variant"] = source_variant
    return identity


def _validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    source_non_time_state_sha256: str,
) -> None:
    require(
        payload.get("checkpoint_type")
        == "frozen_lognormal_duration_resume",
        "Wrong resume checkpoint type",
    )
    require(
        payload.get("checkpoint_schema_version") == 1,
        "Wrong resume schema",
    )
    require(
        payload.get("resume_identity") == dict(identity),
        "Resume identity drift",
    )
    require(
        payload.get("evaluation_scope") == "validation_only",
        "Resume evaluation scope drift",
    )
    require(
        payload.get("held_out_test_evaluated") is False,
        "Resume held-out scope drift",
    )
    require(
        payload.get("legacy_nll_compared") is False,
        "Resume improperly compares legacy NLL",
    )
    history = payload.get("history")
    require(
        isinstance(history, list) and bool(history),
        "Resume history is missing",
    )
    require(
        [int(row["epoch"]) for row in history]
        == list(range(len(history))),
        "Resume history epochs are not contiguous",
    )
    require(
        int(payload.get("epoch", -1)) == len(history) - 1,
        "Resume epoch drift",
    )
    selected = earliest_strict_minimum(history)
    require(
        int(payload.get("best_epoch", -1))
        == int(selected["epoch"]),
        "Resume best epoch drift",
    )
    for label, state_name, digest_name in (
        ("current", "model_state_dict", "model_state_sha256"),
        ("best", "best_state_dict", "best_state_sha256"),
    ):
        state = payload.get(state_name)
        require(
            isinstance(state, Mapping),
            f"Resume {label} model state is missing",
        )
        finite_tensor_mapping(state, label=f"Resume {label} state")
        require(
            canonical_state_dict_sha256(state)
            == payload.get(digest_name),
            f"Resume {label} state digest drift",
        )
        require(
            state_partition_sha256(state, time_head=False)
            == source_non_time_state_sha256,
            f"Resume {label} changed a non-time B tensor",
        )
    optimizer_state = payload.get("optimizer_state_dict")
    require(
        isinstance(optimizer_state, Mapping),
        "Resume optimizer state is missing",
    )
    state_rows = optimizer_state.get("state")
    parameter_groups = optimizer_state.get("param_groups")
    require(
        isinstance(state_rows, Mapping)
        and len(state_rows) == len(TIME_HEAD_PARAMETER_NAMES),
        "Resume optimizer does not cover all time-head parameters",
    )
    require(
        isinstance(parameter_groups, list)
        and len(parameter_groups) == 1,
        "Resume optimizer parameter-group drift",
    )
    parameter_ids = parameter_groups[0].get("params", [])
    require(
        parameter_ids == list(range(len(TIME_HEAD_PARAMETER_NAMES))),
        "Resume optimizer parameter list drift",
    )
    require(
        set(state_rows) == set(parameter_ids),
        "Resume optimizer state parameter IDs drift",
    )
    expected_group_values = {
        "lr": float(identity["learning_rate"]),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": float(identity["weight_decay"]),
        "amsgrad": False,
        "maximize": False,
    }
    for name, expected in expected_group_values.items():
        require(
            parameter_groups[0].get(name) == expected,
            f"Resume AdamW parameter-group drift: {name}",
        )
    optional_group_values = {
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    for name, expected in optional_group_values.items():
        if name in parameter_groups[0]:
            require(
                parameter_groups[0][name] == expected,
                f"Resume AdamW parameter-group drift: {name}",
            )
    expected_steps = int(payload["epoch"]) * math.ceil(
        int(identity["train_cache_count"])
        / int(identity["cached_state_batch_size"])
    )
    current_state = payload["model_state_dict"]
    for parameter_id, parameter_name in zip(
        parameter_ids, TIME_HEAD_PARAMETER_NAMES
    ):
        parameter_state = state_rows[parameter_id]
        require(
            isinstance(parameter_state, Mapping)
            and {"step", "exp_avg", "exp_avg_sq"}.issubset(
                parameter_state
            ),
            "Resume AdamW moment state is incomplete",
        )
        expected_parameter = current_state[parameter_name]
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            require(
                isinstance(moment, torch.Tensor)
                and moment.shape == expected_parameter.shape
                and moment.dtype == expected_parameter.dtype,
                (
                    f"Resume AdamW {moment_name} shape or dtype drift: "
                    f"{parameter_name}"
                ),
            )
        step = parameter_state["step"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and step.dtype == torch.float32
            and float(step.item()) == float(expected_steps),
            f"Resume AdamW step drift: {parameter_name}",
        )
    finite_nested_tensors(
        optimizer_state, label="Resume optimizer state"
    )


def fit_time_head_from_cache(
    *,
    model: SharedTimeCountModel,
    train_cache: FrozenFeatureCache,
    validation_cache: FrozenFeatureCache,
    output_dir: Path,
    dataset: str,
    censor_threshold: float | None,
    contract_sha256: str,
    source_checkpoint_sha256: str,
    source_state_sha256: str,
    source_non_time_state_sha256: str,
    candidate_initial_state_sha256: str,
    train_time_statistics: Mapping[str, Any],
    calibration_source_revision: str,
    seed: int,
    planned_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    quantity_replay_batch_size: int,
    grad_clip: float,
    min_epochs: int,
    patience: int,
    device: str | torch.device,
    candidate_metadata: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    run_epoch_limit: int | None = None,
    contract_id: str = CONTRACT_ID,
    model_role: str = "B",
    source_backbone: str = SOURCE_BACKBONE,
    source_variant: str = SOURCE_VARIANT,
) -> dict[str, Any]:
    """Fit the isolated proper duration head with exact deterministic resume."""
    require(planned_epochs >= 0, "planned_epochs must be nonnegative")
    require(batch_size > 0, "batch_size must be positive")
    require(
        quantity_replay_batch_size > 0,
        "quantity replay batch size must be positive",
    )
    require(
        learning_rate > 0.0 and math.isfinite(learning_rate),
        "Invalid learning rate",
    )
    require(
        weight_decay >= 0.0 and math.isfinite(weight_decay),
        "Invalid weight decay",
    )
    require(
        grad_clip > 0.0 and math.isfinite(grad_clip),
        "Invalid gradient clip",
    )
    require(min_epochs >= 0, "min_epochs must be nonnegative")
    require(patience > 0, "patience must be positive")
    require(
        len(calibration_source_revision) == 40,
        "Calibration source revision must be a full Git SHA",
    )
    train_cache.validate(require_quantity=False)
    validation_cache.validate(require_quantity=True)
    censor_mask(train_cache.target_dt, threshold=censor_threshold)
    censor_mask(
        validation_cache.target_dt, threshold=censor_threshold
    )

    parameters = freeze_time_head_only(model)
    initial_state = clone_state_dict(model)
    observed_initial_sha256 = canonical_state_dict_sha256(
        initial_state
    )
    require(
        observed_initial_sha256 == candidate_initial_state_sha256,
        "Loaded candidate differs from its declared initial state",
    )
    require(
        state_partition_sha256(initial_state, time_head=False)
        == source_non_time_state_sha256,
        "Initial candidate changed a non-time B tensor",
    )
    initial_time_head_sha256 = state_partition_sha256(
        initial_state, time_head=True
    )
    identity = _resume_identity(
        contract_sha256=contract_sha256,
        dataset=dataset,
        source_checkpoint_sha256=source_checkpoint_sha256,
        source_state_sha256=source_state_sha256,
        source_non_time_state_sha256=source_non_time_state_sha256,
        candidate_initial_state_sha256=candidate_initial_state_sha256,
        train_cache=train_cache,
        validation_cache=validation_cache,
        train_time_statistics=train_time_statistics,
        censor_threshold=censor_threshold,
        calibration_source_revision=calibration_source_revision,
        seed=seed,
        planned_epochs=planned_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        quantity_replay_batch_size=quantity_replay_batch_size,
        grad_clip=grad_clip,
        min_epochs=min_epochs,
        patience=patience,
        contract_id=contract_id,
        model_role=model_role,
        source_backbone=source_backbone,
        source_variant=source_variant,
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / LAST_CHECKPOINT_NAME
    selected_path = output_dir / SELECTED_CHECKPOINT_NAME
    summary_path = output_dir / SUMMARY_NAME

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(
            summary.get("resume_identity") == identity,
            "Cached summary identity drift",
        )
        require(
            summary.get("evaluation_scope") == "validation_only",
            "Cached summary evaluation scope drift",
        )
        require(
            summary.get("held_out_test_evaluated") is False,
            "Cached summary held-out scope drift",
        )
        require(
            summary.get("legacy_nll_compared") is False,
            "Cached summary compares an incompatible legacy NLL",
        )
        require(
            summary.get("quantity_prediction_bitwise_identical")
            is True,
            "Cached summary did not preserve quantity predictions",
        )
        require(
            selected_path.is_file(),
            "Cached selected checkpoint is missing",
        )
        selected = torch_load_checkpoint(
            selected_path, map_location="cpu"
        )
        require(
            selected.get("resume_identity") == identity,
            "Cached selected checkpoint identity drift",
        )
        selected_state = selected.get("model_state_dict")
        require(
            isinstance(selected_state, Mapping),
            "Cached selected model state is missing",
        )
        require(
            canonical_state_dict_sha256(selected_state)
            == summary.get("selected_state_sha256"),
            "Cached selected state digest drift",
        )
        require(
            state_partition_sha256(
                selected_state, time_head=False
            )
            == source_non_time_state_sha256,
            "Cached selected checkpoint changed non-time state",
        )
        return summary

    history: list[dict[str, Any]]
    best_state: dict[str, torch.Tensor]
    start_epoch: int
    if last_path.exists():
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        _validate_resume_payload(
            payload,
            identity=identity,
            source_non_time_state_sha256=(
                source_non_time_state_sha256
            ),
        )
        model.load_state_dict(
            payload["model_state_dict"], strict=True
        )
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _finite_optimizer_state(optimizer)
        history = list(payload["history"])
        best_state = payload["best_state_dict"]
        start_epoch = int(payload["epoch"]) + 1
    else:
        epoch_zero = evaluate_cached_time_metrics(
            model=model,
            cache=validation_cache,
            device=device,
            batch_size=batch_size,
            censor_threshold=censor_threshold,
        )
        history = [
            {
                "epoch": 0,
                "train_proper_time_nll": None,
                "val_proper_time_nll": epoch_zero[
                    "proper_time_nll"
                ],
                "val_time_median_mae": epoch_zero[
                    "time_median_mae"
                ],
                "val_time_median_rmse": epoch_zero[
                    "time_median_rmse"
                ],
                "pre_clip_gradient_norm_mean": None,
                "pre_clip_gradient_norm_max": None,
            }
        ]
        best_state = clone_state_dict(model)
        start_epoch = 1

    stopped_early = early_stopping_exhausted(
        history, min_epochs=min_epochs, patience=patience
    )
    call_last_epoch = planned_epochs
    if run_epoch_limit is not None:
        require(
            run_epoch_limit >= 0,
            "run_epoch_limit must be nonnegative",
        )
        call_last_epoch = min(
            planned_epochs, start_epoch + run_epoch_limit - 1
        )
    for epoch in range(start_epoch, call_last_epoch + 1):
        if stopped_early:
            break
        train_telemetry = _train_cached_epoch(
            model=model,
            cache=train_cache,
            optimizer=optimizer,
            parameters=parameters,
            device=device,
            batch_size=batch_size,
            grad_clip=grad_clip,
            seed=seed,
            epoch=epoch,
            censor_threshold=censor_threshold,
        )
        require(
            not model.training,
            "Encoder mode changed during duration-head fitting",
        )
        require(
            state_partition_sha256(
                model.state_dict(), time_head=False
            )
            == source_non_time_state_sha256,
            "Non-time B state changed during fitting",
        )
        validation = evaluate_cached_time_metrics(
            model=model,
            cache=validation_cache,
            device=device,
            batch_size=batch_size,
            censor_threshold=censor_threshold,
        )
        history.append(
            {
                "epoch": epoch,
                **train_telemetry,
                "val_proper_time_nll": validation[
                    "proper_time_nll"
                ],
                "val_time_median_mae": validation[
                    "time_median_mae"
                ],
                "val_time_median_rmse": validation[
                    "time_median_rmse"
                ],
            }
        )
        selected_row = earliest_strict_minimum(history)
        if int(selected_row["epoch"]) == epoch:
            best_state = clone_state_dict(model)
        current_state = clone_state_dict(model)
        atomic_torch_save(
            {
                "checkpoint_type": (
                    "frozen_lognormal_duration_resume"
                ),
                "checkpoint_schema_version": 1,
                "epoch": epoch,
                "model_state_dict": current_state,
                "model_state_sha256": (
                    canonical_state_dict_sha256(current_state)
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_epoch": int(selected_row["epoch"]),
                "best_state_dict": best_state,
                "best_state_sha256": (
                    canonical_state_dict_sha256(best_state)
                ),
                "source_non_time_state_sha256": (
                    source_non_time_state_sha256
                ),
                "resume_identity": identity,
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "legacy_nll_compared": False,
            },
            last_path,
        )
        stopped_early = early_stopping_exhausted(
            history, min_epochs=min_epochs, patience=patience
        )

    current_epoch = int(history[-1]["epoch"])
    paused = (
        run_epoch_limit is not None
        and current_epoch < planned_epochs
        and not stopped_early
    )
    if paused:
        return {
            "status": "paused",
            "completed_epochs": current_epoch,
            "resume_identity": identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "legacy_nll_compared": False,
        }

    selected_row = earliest_strict_minimum(history)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    selected_state = clone_state_dict(model)
    selected_state_sha256 = canonical_state_dict_sha256(
        selected_state
    )
    selected_non_time_sha256 = state_partition_sha256(
        selected_state, time_head=False
    )
    require(
        selected_non_time_sha256
        == source_non_time_state_sha256,
        "Selected checkpoint changed a non-time B tensor",
    )
    selected_time_head_sha256 = state_partition_sha256(
        selected_state, time_head=True
    )
    selected_metrics = evaluate_cached_time_metrics(
        model=model,
        cache=validation_cache,
        device=device,
        batch_size=batch_size,
        censor_threshold=censor_threshold,
    )
    require(
        math.isclose(
            float(selected_metrics["proper_time_nll"]),
            float(selected_row["val_proper_time_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Selected validation proper Time NLL replay drift",
    )
    final_quantity_prediction = cached_quantity_predictions(
        model=model,
        cache=validation_cache,
        device=device,
        batch_size=quantity_replay_batch_size,
    )
    assert validation_cache.source_quantity_prediction is not None
    assert validation_cache.target_quantity is not None
    quantity_bitwise_identical = torch.equal(
        final_quantity_prediction,
        validation_cache.source_quantity_prediction,
    )
    require(
        quantity_bitwise_identical,
        "Quantity predictions changed during duration-head fitting",
    )
    source_quantity_sha256 = tensor_sha256(
        "quantity_prediction",
        validation_cache.source_quantity_prediction,
    )
    selected_quantity_sha256 = tensor_sha256(
        "quantity_prediction", final_quantity_prediction
    )
    require(
        source_quantity_sha256 == selected_quantity_sha256,
        "Quantity prediction digest drift",
    )
    epoch_zero_nll = float(history[0]["val_proper_time_nll"])
    selected_nll = float(selected_metrics["proper_time_nll"])
    changed_state_keys = sorted(
        name
        for name in selected_state
        if not torch.equal(initial_state[name], selected_state[name])
    )
    require(
        set(changed_state_keys).issubset(
            TIME_HEAD_PARAMETER_NAMES
        ),
        f"State outside the duration head changed: {changed_state_keys}",
    )

    checkpoint = {
        "checkpoint_type": "selected_frozen_lognormal_duration",
        "checkpoint_schema_version": 1,
        "selection": SELECTION_RULE,
        "selection_formula": (
            "mean negative censor-aware normalized log-normal "
            "duration log likelihood"
        ),
        "best_epoch": int(selected_row["epoch"]),
        "selected_metric_value": selected_nll,
        "model_state_dict": selected_state,
        "model_state_sha256": selected_state_sha256,
        "source_state_sha256": source_state_sha256,
        "source_non_time_state_sha256": (
            source_non_time_state_sha256
        ),
        "candidate_initial_state_sha256": (
            candidate_initial_state_sha256
        ),
        "candidate_initial_time_head_state_sha256": (
            initial_time_head_sha256
        ),
        "selected_time_head_state_sha256": (
            selected_time_head_sha256
        ),
        "backbone": source_backbone,
        "model_role": model_role,
        "variant": source_variant,
        "time_head_mode": CANDIDATE_TIME_HEAD_MODE,
        "encoder_config": (
            candidate_metadata.get("encoder_config")
            if candidate_metadata is not None
            else None
        ),
        "interface_meta": (
            candidate_metadata.get("interface_meta")
            if candidate_metadata is not None
            else None
        ),
        "source_checkpoint_lineage": (
            dict(source_metadata)
            if source_metadata is not None
            else None
        ),
        "train_time_statistics": dict(
            train_time_statistics
        ),
        "right_censor_threshold": censor_threshold,
        "resume_identity": identity,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    atomic_torch_save(checkpoint, selected_path)
    summary = {
        "schema_version": 1,
        "contract_id": contract_id,
        "status": "success",
        "dataset": dataset,
        "model_role": model_role,
        "source_backbone": source_backbone,
        "source_variant": source_variant,
        "seed": int(seed),
        "time_head_mode": CANDIDATE_TIME_HEAD_MODE,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "source_non_time_state_sha256": (
            source_non_time_state_sha256
        ),
        "candidate_initial_state_sha256": (
            candidate_initial_state_sha256
        ),
        "candidate_initial_time_head_state_sha256": (
            initial_time_head_sha256
        ),
        "selected_state_sha256": selected_state_sha256,
        "selected_non_time_state_sha256": (
            selected_non_time_sha256
        ),
        "selected_time_head_state_sha256": (
            selected_time_head_sha256
        ),
        "changed_state_keys": changed_state_keys,
        "trainable_parameter_names": list(
            TIME_HEAD_PARAMETER_NAMES
        ),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "encoder_mode_during_refit": "eval",
        "hidden_state_gradient": "detached_cache",
        "calculation_dtype": "float64",
        "epoch_zero_initialization": (
            "train_only_log_moment_initialization_with_zero_"
            "conditional_weights"
        ),
        "selection": SELECTION_RULE,
        "best_epoch": int(selected_row["epoch"]),
        "completed_epochs": current_epoch,
        "stopped_early": bool(stopped_early),
        "epoch_zero_validation_proper_time_nll": (
            epoch_zero_nll
        ),
        "best_validation_proper_time_nll": selected_nll,
        "proper_time_nll_improvement_from_epoch_zero": (
            epoch_zero_nll - selected_nll
        ),
        "strictly_improves_epoch_zero": (
            selected_nll < epoch_zero_nll
        ),
        "validation_time_metrics": selected_metrics,
        "time_median_mae": selected_metrics[
            "time_median_mae"
        ],
        "time_median_rmse": selected_metrics[
            "time_median_rmse"
        ],
        "right_censor_threshold": censor_threshold,
        "train_right_censored_count": int(
            censor_mask(
                train_cache.target_dt,
                threshold=censor_threshold,
            ).sum().item()
        ),
        "validation_right_censored_count": int(
            censor_mask(
                validation_cache.target_dt,
                threshold=censor_threshold,
            ).sum().item()
        ),
        "quantity_prediction_bitwise_identical": (
            quantity_bitwise_identical
        ),
        "source_quantity_prediction_sha256": (
            source_quantity_sha256
        ),
        "selected_quantity_prediction_sha256": (
            selected_quantity_sha256
        ),
        "quantity_metrics": quantity_metrics(
            final_quantity_prediction,
            validation_cache.target_quantity,
        ),
        "train_time_statistics": dict(
            train_time_statistics
        ),
        "train_cache": {
            "count": train_cache.count,
            "sha256": train_cache.digest(),
            "target_dt_sha256": target_dt_sha256(
                train_cache.target_dt
            ),
            "censor_mask_sha256": censor_mask_sha256(
                censor_mask(
                    train_cache.target_dt,
                    threshold=censor_threshold,
                )
            ),
        },
        "validation_cache": {
            "count": validation_cache.count,
            "sha256": validation_cache.digest(),
            "target_dt_sha256": target_dt_sha256(
                validation_cache.target_dt
            ),
            "censor_mask_sha256": censor_mask_sha256(
                censor_mask(
                    validation_cache.target_dt,
                    threshold=censor_threshold,
                )
            ),
        },
        "history": history,
        "resume_identity": identity,
        "selected_checkpoint_path": str(selected_path),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    save_json(summary_path, summary)
    return summary


def build_candidate_from_selected_checkpoint(
    payload: Mapping[str, Any],
) -> SharedTimeCountModel:
    """Strictly rebuild one selected candidate from its own metadata."""
    require(
        payload.get("checkpoint_type")
        == "selected_frozen_lognormal_duration",
        "Wrong selected candidate checkpoint type",
    )
    require(
        payload.get("time_head_mode") == CANDIDATE_TIME_HEAD_MODE,
        "Selected checkpoint time-head mode drift",
    )
    encoder = payload.get("encoder_config")
    interface = payload.get("interface_meta")
    require(
        isinstance(encoder, Mapping),
        "Selected checkpoint encoder metadata is missing",
    )
    require(
        isinstance(interface, Mapping),
        "Selected checkpoint interface metadata is missing",
    )
    time_head = encoder.get("time_head")
    require(
        isinstance(time_head, Mapping),
        "Selected checkpoint time-head metadata is missing",
    )
    require(
        time_head.get("mode") == CANDIDATE_TIME_HEAD_MODE,
        "Selected encoder time-head mode drift",
    )
    source_backbone = str(payload.get("backbone", SOURCE_BACKBONE))
    hidden_dim = encoder.get("d_model", encoder.get("hidden_dim"))
    require(
        type(hidden_dim) is int and int(hidden_dim) > 0,
        "Selected checkpoint hidden dimension is missing",
    )
    max_seq_len = encoder.get("max_len", 1)
    model, rebuilt_encoder = build_count_aware_model(
        source_backbone,
        hidden_dim=int(hidden_dim),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface["train_target_std"]),
        max_seq_len=int(max_seq_len),
        quantity_variant=str(payload.get("variant", SOURCE_VARIANT)),
        lambda_tail=0.0,
        time_head_mode=CANDIDATE_TIME_HEAD_MODE,
        time_scale=float(time_head["time_scale"]),
        time_initial_location=float(
            time_head["time_initial_location"]
        ),
        time_initial_scale=float(
            time_head["time_initial_scale"]
        ),
        time_sigma_floor=float(time_head["time_sigma_floor"]),
    )
    require(
        rebuilt_encoder == dict(encoder),
        "Rebuilt candidate encoder metadata drift",
    )
    state = payload.get("model_state_dict")
    require(
        isinstance(state, Mapping),
        "Selected checkpoint model state is missing",
    )
    model.load_state_dict(state, strict=True)
    require(
        canonical_state_dict_sha256(model.state_dict())
        == payload.get("model_state_sha256"),
        "Selected checkpoint model-state digest drift",
    )
    model.eval()
    return model


def dataset_by_id(
    contract: Mapping[str, Any], dataset: str
) -> dict[str, Any]:
    rows = [
        row
        for row in contract.get("datasets", [])
        if row.get("dataset") == dataset
    ]
    require(
        len(rows) == 1,
        f"Expected one contract row for dataset {dataset!r}",
    )
    return dict(rows[0])


def validate_contract(
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    require(
        contract.get("contract_id") == CONTRACT_ID,
        "Wrong frozen log-normal contract",
    )
    require(contract.get("schema_version") == 1, "Wrong contract schema")
    candidate = contract.get("candidate")
    require(isinstance(candidate, Mapping), "Candidate contract is missing")
    require(
        candidate.get("time_head_mode") == CANDIDATE_TIME_HEAD_MODE,
        "Candidate time-head mode drift",
    )
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope contract is missing")
    expected_scope = {
        "input_splits": ["train", "validation"],
        "optimization_split": "train",
        "selection_split": "validation",
        "evaluation_scope": "validation_only",
        "held_out_test": False,
        "additional_seeds": False,
        "quantity_predictions_may_change": False,
    }
    for name, expected in expected_scope.items():
        require(scope.get(name) == expected, f"Scope drift: {name}")
    time_head = contract.get("time_head")
    require(
        isinstance(time_head, Mapping),
        "Duration-head contract is missing",
    )
    require(
        time_head.get("trainable_parameter_names")
        == list(TIME_HEAD_PARAMETER_NAMES),
        "Duration-head parameter names drift",
    )
    require(
        int(time_head.get("expected_trainable_parameter_count", -1))
        == 130,
        "Duration-head parameter count drift",
    )
    require(
        time_head.get("duration_clamp") is False,
        "Duration clamp is prohibited",
    )
    require(
        time_head.get("original_unit_jacobian") is True,
        "Original-unit Jacobian is required",
    )
    observation = contract.get("observation_likelihood")
    require(
        isinstance(observation, Mapping),
        "Observation likelihood contract is missing",
    )
    require(
        observation.get("reduction")
        == "arithmetic mean over admitted target events",
        "Observation reduction drift",
    )
    optimization = contract.get("optimization")
    require(
        isinstance(optimization, Mapping),
        "Optimization contract is missing",
    )
    expected_optimization = {
        "shared_across_all_datasets": True,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "encoder_batch_size": 128,
        "cached_state_batch_size": 8192,
        "gradient_clip": 1.0,
        "epochs": 100,
        "minimum_epochs": 20,
        "early_stopping_patience": 20,
        "scheduler": None,
        "dataset_specific_hyperparameters": False,
    }
    for name, expected in expected_optimization.items():
        require(
            optimization.get(name) == expected,
            f"Optimization drift: {name}",
        )
    selection = contract.get("checkpoint_selection")
    require(
        isinstance(selection, Mapping),
        "Checkpoint selection contract is missing",
    )
    require(
        selection.get("monitor") == "validation_proper_time_nll",
        "Checkpoint monitor drift",
    )
    require(
        selection.get("rule") == "earliest strict finite minimum",
        "Checkpoint selection rule drift",
    )
    require(
        selection.get("fallback") == "epoch 0",
        "Epoch-zero fallback drift",
    )
    acceptance = contract.get("acceptance")
    require(
        isinstance(acceptance, Mapping),
        "Acceptance contract is missing",
    )
    require(
        acceptance.get("legacy_comparison_prohibited") is True,
        "Legacy likelihood comparison must remain prohibited",
    )
    rows = contract.get("datasets")
    require(
        isinstance(rows, list) and len(rows) == 3,
        "Expected exactly three dataset contracts",
    )
    datasets = {str(row["dataset"]): dict(row) for row in rows}
    require(
        tuple(datasets)
        == (
            "intermittent_frozen_5000",
            "yellow_trip_hourly",
            INSTACART_DATASET,
        ),
        "Dataset order or scope drift",
    )
    for dataset, row in datasets.items():
        resolve_censor_threshold(dataset, row)
        for split in ("train", "validation"):
            require(
                int(row[f"expected_{split}_targets"]) > 0,
                f"Empty {dataset} {split} target population",
            )
            require(
                int(row[f"expected_{split}_censored_targets"])
                >= 0,
                f"Invalid {dataset} {split} censor count",
            )
            for suffix in (
                "target_identity_sha256",
                "target_quantity_sha256",
                "target_dt_sha256",
                "censor_mask_sha256",
            ):
                require(
                    len(str(row[f"expected_{split}_{suffix}"]))
                    == 64,
                    f"Missing {dataset} {split} {suffix}",
                )
        require(
            float(row["train_time_scale"]) > 0.0,
            f"Invalid {dataset} train time scale",
        )
        require(
            float(row["train_log_scaled_std"]) > 0.001,
            f"Invalid {dataset} train duration scale",
        )
        require(
            len(str(row["B_checkpoint_file_sha256"])) == 64,
            f"Missing {dataset} B checkpoint digest",
        )
        require(
            len(str(row["B_checkpoint_state_sha256"])) == 64,
            f"Missing {dataset} B state digest",
        )
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-source-revision", required=True
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-partial-contract", action="store_true"
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument(
        "--max-validation-batches", type=int, default=None
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but unavailable")
        torch.cuda.reset_peak_memory_stats(requested_device)
        device_name = torch.cuda.get_device_name(requested_device)
    else:
        device_name = str(requested_device)
    require(
        len(args.calibration_source_revision) == 40,
        "Calibration source revision must be a full Git SHA",
    )
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    datasets = validate_contract(contract)
    require(args.dataset in datasets, "Dataset is outside the contract")
    dataset_spec = datasets[args.dataset]
    contract_sha256 = sha256_file(args.contract)
    require(
        sha256_file(args.data) == dataset_spec["data_sha256"],
        "Dataset digest mismatch",
    )
    require(
        sha256_file(args.split_manifest)
        == dataset_spec["split_manifest_sha256"],
        "Split-manifest digest mismatch",
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    require(
        checkpoint_sha256
        == dataset_spec["B_checkpoint_file_sha256"],
        "B checkpoint file digest mismatch",
    )
    source_payload = torch_load_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    validate_source_checkpoint(
        source_payload, dataset_spec=dataset_spec
    )
    source_state_sha256 = str(source_payload["model_state_sha256"])
    source_model = build_source_model(source_payload).to(
        requested_device
    )

    raw_frame = load_admitted_frame(args.data)
    frame = prepare_count_frame(raw_frame)
    lookback = int(dataset_spec["lookback"])
    max_seq_len = int(dataset_spec["max_sequence_length"])
    train_population = exact_target_population_contract(
        frame,
        target_split="train",
        lookback=lookback,
        max_seq_len=max_seq_len,
    )
    validation_population = exact_target_population_contract(
        frame,
        target_split="validation",
        lookback=lookback,
        max_seq_len=max_seq_len,
    )
    validate_target_population(
        train_population, dataset_spec=dataset_spec
    )
    validate_target_population(
        validation_population, dataset_spec=dataset_spec
    )
    observed_train_time_statistics = derive_train_time_contract(
        frame,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
    )
    train_time_statistics = train_time_statistics_from_contract(
        dataset_spec, observed_train_time_statistics
    )
    censor_threshold = resolve_censor_threshold(
        args.dataset, dataset_spec
    )

    settings = dict(contract["optimization"])
    partial_values = {
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "max_epochs": args.max_epochs,
    }
    if any(value is not None for value in partial_values.values()):
        require(
            args.allow_partial_contract,
            "Batch or epoch limits require --allow-partial-contract",
        )
    planned_epochs = int(settings["epochs"])
    if args.max_epochs is not None:
        require(args.max_epochs >= 0, "max epochs must be nonnegative")
        planned_epochs = min(planned_epochs, args.max_epochs)
    encoder_batch_size = int(settings["encoder_batch_size"])
    train_loader = make_loader(
        frame,
        target_split="train",
        batch_size=encoder_batch_size,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        shuffle=False,
        generator=None,
    )
    validation_loader = make_loader(
        frame,
        target_split="validation",
        batch_size=encoder_batch_size,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        shuffle=False,
        generator=None,
    )
    cache_identity_base = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "dataset": args.dataset,
        "data_sha256": dataset_spec["data_sha256"],
        "split_manifest_sha256": dataset_spec[
            "split_manifest_sha256"
        ],
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "encoder_mode": "eval",
        "target_quantity_masked": True,
        "memory_target_write_masked": True,
        "evaluation_scope": "train_and_validation_only",
        "held_out_test_evaluated": False,
    }
    cache_dir = (
        args.feature_cache_dir
        if args.feature_cache_dir is not None
        else args.output_dir / "cache"
    )
    if args.feature_cache_dir is not None:
        require(
            (cache_dir / "train_features.pt").is_file()
            and (cache_dir / "validation_features.pt").is_file(),
            "Requested shared feature cache is incomplete",
        )
    train_cache = load_or_extract_feature_cache(
        path=cache_dir / "train_features.pt",
        identity={
            **cache_identity_base,
            "split": "train",
            "target_count": train_population["target_count"],
            "target_identity_sha256": train_population[
                "target_identity_sha256"
            ],
            "target_quantity_sha256": train_population[
                "target_quantity_sha256"
            ],
            "max_batches": args.max_train_batches,
        },
        model=source_model,
        loader=train_loader,
        device=requested_device,
        include_quantity=False,
        max_batches=args.max_train_batches,
    )
    validation_cache = load_or_extract_feature_cache(
        path=cache_dir / "validation_features.pt",
        identity={
            **cache_identity_base,
            "split": "validation",
            "target_count": validation_population["target_count"],
            "target_identity_sha256": validation_population[
                "target_identity_sha256"
            ],
            "target_quantity_sha256": validation_population[
                "target_quantity_sha256"
            ],
            "max_batches": args.max_validation_batches,
        },
        model=source_model,
        loader=validation_loader,
        device=requested_device,
        include_quantity=True,
        max_batches=args.max_validation_batches,
    )
    qualified_full_data = (
        args.max_train_batches is None
        and args.max_validation_batches is None
    )
    train_observation_contract = None
    validation_observation_contract = None
    if qualified_full_data:
        train_observation_contract = (
            validate_cache_observation_contract(
                train_cache,
                dataset_spec=dataset_spec,
                split="train",
                censor_threshold=censor_threshold,
            )
        )
        validation_observation_contract = (
            validate_cache_observation_contract(
                validation_cache,
                dataset_spec=dataset_spec,
                split="validation",
                censor_threshold=censor_threshold,
            )
        )

    source_non_time_state_sha256 = state_partition_sha256(
        source_payload["model_state_dict"], time_head=False
    )
    candidate, candidate_encoder, candidate_metadata = (
        build_frozen_lognormal_candidate(
            source_payload,
            train_time_statistics=train_time_statistics,
            time_sigma_floor=0.001,
        )
    )
    require(
        candidate_metadata["source_non_time_state_sha256"]
        == source_non_time_state_sha256,
        "Candidate source non-time digest drift",
    )
    del source_model
    if requested_device.type == "cuda":
        torch.cuda.empty_cache()
    candidate = candidate.to(requested_device)
    source_lineage = {
        "B_checkpoint_file_sha256": checkpoint_sha256,
        "B_checkpoint_state_sha256": source_state_sha256,
        "B_training_source_revision": source_payload.get(
            "source_revision"
        ),
        "B_training_source_revision_history": source_payload.get(
            "source_revision_history"
        ),
        "calibration_source_revision": (
            args.calibration_source_revision
        ),
    }
    summary = fit_time_head_from_cache(
        model=candidate,
        train_cache=train_cache,
        validation_cache=validation_cache,
        output_dir=args.output_dir,
        dataset=args.dataset,
        censor_threshold=censor_threshold,
        contract_sha256=contract_sha256,
        source_checkpoint_sha256=checkpoint_sha256,
        source_state_sha256=source_state_sha256,
        source_non_time_state_sha256=(
            source_non_time_state_sha256
        ),
        candidate_initial_state_sha256=candidate_metadata[
            "candidate_initial_state_sha256"
        ],
        train_time_statistics=train_time_statistics,
        calibration_source_revision=args.calibration_source_revision,
        seed=int(settings["seed"]),
        planned_epochs=planned_epochs,
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        batch_size=int(settings["cached_state_batch_size"]),
        quantity_replay_batch_size=encoder_batch_size,
        grad_clip=float(settings["gradient_clip"]),
        min_epochs=int(settings["minimum_epochs"]),
        patience=int(settings["early_stopping_patience"]),
        device=requested_device,
        candidate_metadata=candidate_metadata,
        source_metadata=source_lineage,
    )
    if summary.get("status") != "success":
        return summary
    summary = dict(summary)
    summary["encoder_config"] = candidate_encoder
    summary["interface_meta"] = candidate_metadata["interface_meta"]
    summary["train_target_population"] = train_population
    summary["validation_target_population"] = validation_population
    summary["train_observation_contract"] = (
        train_observation_contract
    )
    summary["validation_observation_contract"] = (
        validation_observation_contract
    )
    summary["qualified_full_data"] = qualified_full_data
    summary["feature_cache_dir"] = str(cache_dir.resolve())
    summary["quantity_metrics"].update(
        stratified_quantity_metrics(
            validation_cache.source_quantity_prediction,
            validation_cache.target_quantity,
            body_max=float(
                dataset_spec["reporting_body_max_train_p95"]
            ),
            tail_min_exclusive=float(
                dataset_spec[
                    "reporting_tail_min_exclusive_train_p99"
                ]
            ),
            require_nonempty=qualified_full_data,
        )
    )
    summary["runtime"] = {
        "requested_device": str(requested_device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(requested_device))
            if requested_device.type == "cuda"
            else 0
        ),
        "peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(requested_device))
            if requested_device.type == "cuda"
            else 0
        ),
    }
    save_json(args.output_dir / SUMMARY_NAME, summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
