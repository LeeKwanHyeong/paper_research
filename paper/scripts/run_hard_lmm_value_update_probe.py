#!/usr/bin/env python3
"""Run the frozen-B, train-only selected-value update diagnostic."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "paper/contracts/hard_lmm_value_update_probe_v1.json"
BASE_CONTRACT = ROOT / "paper/contracts/hard_lmm_credit_assignment_probe_v1.json"
BASE_RUNNER = ROOT / "paper/scripts/run_hard_lmm_credit_assignment_probe.py"
HELPER = ROOT / "paper/scripts/hard_lmm_value_update_probe.py"
RESULT = ROOT / "paper/results/hard_lmm_value_update_probe_20260909"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hard_lmm_value_update")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_hard_lmm_value_update")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("hard_lmm_value_update_base_runner", BASE_RUNNER)
pure = load_module("hard_lmm_value_update_helpers", HELPER)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("contract_id") == "hard_lmm_value_update_probe_v1", "Contract ID drift")
    authorization = contract["authorization"]
    require(authorization["local_train_only_diagnostic"] is True, "Local diagnostic disabled")
    for key in ("candidate_implementation", "gpu_execution", "held_out_test", "model_parameter_updates", "validation_rows"):
        require(authorization[key] is False, f"Forbidden authorization enabled: {key}")
    scope = contract["diagnostic_scope"]
    require(scope["allowed_split"] == "train_only", "Diagnostic is not train-only")
    require(scope["batch_size"] == 128, "Batch size drift")
    require(scope["maximum_targets_per_dataset"] == 4096, "Cohort size drift")
    require(scope["minimum_targets_per_fold"] == 2048, "Fold size drift")
    require(len(scope["datasets"]) == 3, "Dataset scope drift")
    baseline = contract["baseline"]
    require(baseline["source_revision"] == "f75243473adc25d622319dbca9bda7e076d8240f", "B source drift")
    require("no online" in baseline["important_boundary"], "Online-write boundary is ambiguous")
    probe = contract["single_mechanistic_probe"]
    require("64" in probe["batch_formula"]["mean_support"], "Memory size drift")
    require(probe["forward_change"] == "none", "Forward mutation enabled")
    controls = contract["controls"]
    require("63" in controls["cyclic_row_shuffles"], "All cyclic controls are not frozen")
    require(len(contract["prospective_gate"]["directions"]) == 2, "Fold direction drift")
    require(len(contract["prospective_gate"]["scopes"]) == 2, "Gate scope drift")


def verify_reused_evidence(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    reuse = contract["evidence_reuse"]
    checks = (
        ("dataset_and_checkpoint_contract_path", "dataset_and_checkpoint_contract_sha256"),
        ("aggregate_gradient_analysis_path", "aggregate_gradient_analysis_sha256"),
        ("aggregate_gradient_manifest_path", "aggregate_gradient_manifest_sha256"),
        ("aggregate_gradient_decision_path", "aggregate_gradient_decision_sha256"),
    )
    for path_key, hash_key in checks:
        path = ROOT / reuse[path_key]
        require(path.is_file(), f"Missing reused evidence: {path}")
        require(sha256_file(path) == reuse[hash_key], f"Reused evidence drift: {path}")
    base_contract = read_json(ROOT / reuse["dataset_and_checkpoint_contract_path"])
    base.validate_contract(base_contract)
    previous_analysis = read_json(ROOT / reuse["aggregate_gradient_analysis_path"])
    require(set(previous_analysis) == set(contract["diagnostic_scope"]["datasets"]), "Previous dataset scope drift")
    return base_contract, previous_analysis


def complete_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> list[torch.Tensor]:
    require(len(parameters) == len(gradients), "Gradient list length drift")
    result = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ]
    require(all(bool(torch.isfinite(value).all()) for value in result), "Non-finite gradient")
    return result


def flatten(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [value.detach().reshape(-1).to(dtype=torch.float64, device="cpu") for value in values]
    )


def clip_scale(vector: torch.Tensor, max_norm: float = 1.0) -> float:
    norm = float(torch.linalg.vector_norm(vector))
    require(math.isfinite(norm), "Non-finite global gradient norm")
    return min(1.0, float(max_norm) / (norm + 1e-6))


def hidden_credits(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    target_quantity: torch.Tensor,
    target_duration: torch.Tensor,
    body_threshold: float,
) -> dict[str, torch.Tensor]:
    location, prediction = model.predict_quantity(hidden)
    log_losses = (location - torch.log1p(target_quantity.clamp_min(0.0))).square()
    raw_losses = (prediction - target_quantity).square()
    body_mask = target_quantity <= float(body_threshold)
    body_losses = (prediction - target_quantity).abs() * body_mask.to(prediction.dtype)
    time_losses = -model.log_f_dt(hidden, target_duration)
    outputs: dict[str, torch.Tensor] = {}
    for name, losses in (
        ("log", log_losses),
        ("raw", raw_losses),
        ("body", body_losses),
        ("time", time_losses),
    ):
        outputs[name] = torch.autograd.grad(losses.sum(), hidden, retain_graph=True)[0].detach()
        outputs[f"{name}_losses"] = losses
    outputs["prediction"] = prediction
    outputs["location"] = location
    outputs["body_mask"] = body_mask
    return outputs


def cached_hidden_credits(
    model: torch.nn.Module,
    cache: Mapping[str, np.ndarray],
    start: int,
    stop: int,
    *,
    body_threshold: float,
) -> dict[str, torch.Tensor]:
    def numeric(name: str) -> torch.Tensor:
        return torch.from_numpy(cache[name][start:stop]).double()

    def flag(name: str) -> torch.Tensor:
        return torch.from_numpy(cache[name][start:stop]).bool()

    return pure.reconstruct_hidden_credits(
        numeric("B__prediction"),
        numeric("true_quantity"),
        numeric("true_duration"),
        model.quantity_head.weight.detach().double(),
        model.v_t.weight.detach().double(),
        numeric("B__w"),
        numeric("B__time_intercept_raw"),
        numeric("B__time_intercept_clamped"),
        numeric("B__integral_term"),
        flag("B__intercept_saturated"),
        flag("B__wd_saturated"),
        body_threshold=body_threshold,
    )


def zeros_like_parameters(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter, dtype=torch.float64, device="cpu") for parameter in parameters]


def accumulate_parameter_lists(
    destination: list[torch.Tensor],
    source: Sequence[torch.Tensor],
    weight: int,
) -> None:
    for target, value in zip(destination, source):
        target.add_(value.detach().to(dtype=torch.float64, device="cpu"), alpha=float(weight))


def parameter_layout(parameters: Sequence[torch.nn.Parameter], memory_index: int) -> tuple[int, int, int]:
    cursor = 0
    start = end = -1
    for index, parameter in enumerate(parameters):
        next_cursor = cursor + parameter.numel()
        if index == memory_index:
            start, end = cursor, next_cursor
        cursor = next_cursor
    require(start >= 0 and end > start, "Memory parameter layout missing")
    return cursor, start, end


def add_descriptor_batch(
    descriptor: dict[str, Any],
    indices: torch.Tensor,
    credits: Mapping[str, torch.Tensor],
    target_quantity: torch.Tensor,
    boundaries: Sequence[float],
) -> None:
    memory_size = descriptor["selection_counts"].numel()
    topk = int(indices.size(1))
    flat_indices = indices.reshape(-1).cpu()
    descriptor["selection_counts"].add_(torch.bincount(flat_indices, minlength=memory_size).double())
    strata = torch.bucketize(
        target_quantity.detach().cpu().double(),
        torch.tensor(boundaries, dtype=torch.float64),
        right=False,
    )
    descriptor["stratum_target_counts"].add_(torch.bincount(strata, minlength=5).double())
    for stratum in range(5):
        mask = strata == stratum
        if bool(mask.any()):
            stratum_indices = indices.detach().cpu()[mask]
            descriptor["stratum_selection_counts"][stratum].add_(
                torch.bincount(stratum_indices.reshape(-1), minlength=memory_size).double()
            )
    for metric in ("log", "raw", "body", "time"):
        metric_credit = credits[metric].detach().cpu().double()
        repeated = metric_credit[:, None, :].expand(-1, topk, -1).reshape(-1, metric_credit.size(1)) / float(topk)
        descriptor["gradient_sums"][metric].index_add_(0, flat_indices, repeated)
        magnitudes = torch.linalg.vector_norm(repeated, dim=1)
        descriptor["absolute_contribution_sums"][metric].index_add_(0, flat_indices, magnitudes)
        for stratum in range(5):
            mask = strata == stratum
            if bool(mask.any()):
                selected = indices.detach().cpu()[mask]
                selected_credit = metric_credit[mask]
                expanded = selected_credit[:, None, :].expand(-1, topk, -1).reshape(-1, metric_credit.size(1)) / float(topk)
                descriptor["stratum_gradient_sums"][metric][stratum].index_add_(0, selected.reshape(-1), expanded)


def empty_descriptor(memory_size: int, hidden_dim: int) -> dict[str, Any]:
    return {
        "selection_counts": torch.zeros(memory_size, dtype=torch.float64),
        "stratum_target_counts": torch.zeros(5, dtype=torch.float64),
        "stratum_selection_counts": torch.zeros(5, memory_size, dtype=torch.float64),
        "gradient_sums": {
            metric: torch.zeros(memory_size, hidden_dim, dtype=torch.float64)
            for metric in ("log", "raw", "body", "time")
        },
        "absolute_contribution_sums": {
            metric: torch.zeros(memory_size, dtype=torch.float64)
            for metric in ("log", "raw", "body", "time")
        },
        "stratum_gradient_sums": {
            metric: torch.zeros(5, memory_size, hidden_dim, dtype=torch.float64)
            for metric in ("log", "raw", "body", "time")
        },
    }


def fold_probe(
    model: torch.nn.Module,
    frozen: Any,
    loader: DataLoader,
    *,
    parameters: Sequence[torch.nn.Parameter],
    names: Sequence[str],
    body_threshold: float,
    boundaries: Sequence[float],
    cache: Mapping[str, np.ndarray],
    cache_offset: int,
    dataset_name: str,
    fold: int,
) -> dict[str, Any]:
    memory_index = names.index("lmm.mem")
    memory = parameters[memory_index]
    require(tuple(memory.shape) == (1, 64, 64), "B memory shape drift")
    memory_size, hidden_dim = 64, 64
    total_parameters, memory_start, memory_end = parameter_layout(parameters, memory_index)
    hard_bank_sum = torch.zeros(memory_size * hidden_dim, dtype=torch.float64)
    shifted_bank_sums = torch.zeros(memory_size, memory_size * hidden_dim, dtype=torch.float64)
    hard_postclip_sum = torch.zeros(total_parameters, dtype=torch.float64)
    shifted_postclip_sums = torch.zeros(memory_size, total_parameters, dtype=torch.float64)
    metric_sums = {
        metric: zeros_like_parameters(parameters)
        for metric in ("log", "raw", "body", "time")
    }
    descriptor = empty_descriptor(memory_size, hidden_dim)
    counts = {"targets": 0, "body": 0, "candidate_signal_batches": 0, "batches": 0}
    parity = {
        "prediction_max_relative": 0.0,
        "prediction_log1p_max_abs": 0.0,
        "target_quantity_max_abs": 0.0,
        "target_duration_max_abs": 0.0,
        "top4_indices_bitwise_equal": True,
        "cache_hidden_credit_max_relative": {metric: 0.0 for metric in ("log", "raw", "body", "time")},
        "selected_gradient_max_relative": {metric: 0.0 for metric in ("log", "raw", "body", "time")},
        "selected_gradient_cancellation_amplified_relative": {metric: 0.0 for metric in ("log", "raw", "body", "time")},
    }
    batch_audits: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        require(quantities is not None, "Quantity observations are missing")
        batch = base.prepare_batch(model, frozen, dts, mask, quantities)
        hidden = batch["hard_hidden"]
        target_quantity = batch["target_quantity"]
        target_duration = batch["target_duration"]
        indices = batch["top4_indices"]
        credits = hidden_credits(
            model, hidden, target_quantity, target_duration, body_threshold
        )
        body_mask = credits["body_mask"]
        batch_count = int(target_quantity.numel())
        body_count = int(body_mask.sum())
        require(batch_count > 0 and body_count > 0, "Empty diagnostic batch")
        log_loss = credits["log_losses"].mean()
        raw_loss = credits["raw_losses"].mean()
        body_loss = credits["body_losses"].sum() / body_count
        time_loss = credits["time_losses"].mean()
        native: dict[str, list[torch.Tensor]] = {}
        for position, (metric, loss) in enumerate(
            (("log", log_loss), ("raw", raw_loss), ("body", body_loss), ("time", time_loss))
        ):
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=position < 3,
                allow_unused=True,
            )
            native[metric] = complete_gradients(parameters, gradients)
            weight = body_count if metric == "body" else batch_count
            accumulate_parameter_lists(metric_sums[metric], native[metric], weight)
            if metric == "body":
                analytic = pure.aggregate_selected_value_gradient(
                    credits[metric][body_mask],
                    indices[body_mask],
                    memory_size=memory_size,
                )
            else:
                analytic = pure.aggregate_selected_value_gradient(
                    credits[metric], indices, memory_size=memory_size
                )
            observed = native[metric][memory_index].squeeze(0)
            difference = float(torch.linalg.vector_norm(observed.double() - analytic))
            selected_credit = credits[metric][body_mask] if metric == "body" else credits[metric]
            denominator = body_count if metric == "body" else batch_count
            uncancelled_scale = float(
                torch.linalg.vector_norm(selected_credit.detach().double(), dim=1).sum()
                / float(denominator)
            )
            final_scale = max(1e-30, float(torch.linalg.vector_norm(observed)))
            scale = max(
                final_scale,
                float(torch.linalg.vector_norm(analytic)),
                uncancelled_scale,
                1e-30,
            )
            parity["selected_gradient_max_relative"][metric] = max(
                parity["selected_gradient_max_relative"][metric], difference / scale
            )
            parity["selected_gradient_cancellation_amplified_relative"][metric] = max(
                parity["selected_gradient_cancellation_amplified_relative"][metric],
                difference / final_scale,
            )
        hard_joint = [left + right for left, right in zip(native["log"], native["time"])]
        hard_flat = flatten(hard_joint)
        hard_memory = hard_joint[memory_index].squeeze(0).detach().cpu().double()
        transform = pure.usage_normalized_support_confidence(
            credits["log"].detach().cpu(),
            indices.detach().cpu(),
            memory_size=memory_size,
        )
        transformed = transform["transformed_gradient"]
        transform_audit = {
            "mean_support": float(transform["average_usage"]),
            "active_rows": int((transform["selection_counts"] > 0).sum()),
            "hard_norm": float(torch.linalg.vector_norm(transform["current_gradient"])),
            "raw_norm": float(torch.linalg.vector_norm(transform["raw_transformed_gradient"])),
            "transformed_norm": float(torch.linalg.vector_norm(transformed)),
            "candidate_signal": float(torch.linalg.vector_norm(transform["current_gradient"])) > 1e-30,
        }
        current_difference = float(
            torch.linalg.vector_norm(
                transform["current_gradient"]
                - native["log"][memory_index].squeeze(0).detach().cpu().double()
            )
        )
        current_scale = max(
            1e-30,
            float(torch.linalg.vector_norm(native["log"][memory_index])),
        )
        require(current_difference / current_scale <= 2e-6, "Helper current-gradient parity failed")
        if transform_audit["candidate_signal"]:
            counts["candidate_signal_batches"] += 1
        time_memory = native["time"][memory_index].squeeze(0).detach().cpu().double()
        shifted_memory = torch.cat(
            [
                transformed.unsqueeze(0),
                pure.cyclic_row_shifts(transformed),
            ],
            dim=0,
        )
        shifted_memory = (shifted_memory + time_memory.unsqueeze(0)).reshape(memory_size, -1)
        hard_bank_sum.add_(hard_memory.reshape(-1), alpha=float(batch_count))
        shifted_bank_sums.add_(shifted_memory, alpha=float(batch_count))
        hard_scale = clip_scale(hard_flat)
        hard_postclip_sum.add_(hard_flat, alpha=float(batch_count) * hard_scale)
        candidate_full = hard_flat.unsqueeze(0).expand(memory_size, -1).clone()
        candidate_full[:, memory_start:memory_end] = shifted_memory
        arm_norms = torch.linalg.vector_norm(candidate_full, dim=1)
        arm_scales = torch.clamp(1.0 / (arm_norms + 1e-6), max=1.0)
        shifted_postclip_sums.add_(
            candidate_full * arm_scales[:, None], alpha=float(batch_count)
        )
        add_descriptor_batch(descriptor, indices, credits, target_quantity, boundaries)

        observed_prediction = credits["prediction"].detach().cpu().double().numpy()
        start = cache_offset + counts["targets"]
        stop = start + batch_count
        cached_credits = cached_hidden_credits(
            model, cache, start, stop, body_threshold=body_threshold
        )
        cache_credit_names = {
            "log": "log_mse_hidden_credit",
            "raw": "raw_squared_error_hidden_credit",
            "body": "body_mae_hidden_credit",
            "time": "legacy_time_nll_hidden_credit",
        }
        for metric, cache_name in cache_credit_names.items():
            observed_credit = credits[metric].detach().cpu().double()
            cached_credit = cached_credits[cache_name]
            difference_norm = float(torch.linalg.vector_norm(observed_credit - cached_credit))
            observed_norm = max(1e-30, float(torch.linalg.vector_norm(observed_credit)))
            parity["cache_hidden_credit_max_relative"][metric] = max(
                parity["cache_hidden_credit_max_relative"][metric],
                difference_norm / observed_norm,
            )
        cached_prediction = cache["B__prediction"][start:stop].astype(np.float64)
        difference = np.abs(observed_prediction - cached_prediction)
        parity["prediction_max_relative"] = max(
            parity["prediction_max_relative"],
            float(np.max(difference / np.maximum(1.0, np.abs(cached_prediction)))),
        )
        parity["prediction_log1p_max_abs"] = max(
            parity["prediction_log1p_max_abs"],
            float(np.max(np.abs(np.log1p(observed_prediction) - np.log1p(cached_prediction)))),
        )
        parity["top4_indices_bitwise_equal"] = bool(
            parity["top4_indices_bitwise_equal"]
            and np.array_equal(
                indices.detach().cpu().numpy(), cache["B__top4_indices"][start:stop]
            )
        )
        parity["target_quantity_max_abs"] = max(
            parity["target_quantity_max_abs"],
            float(np.max(np.abs(target_quantity.detach().cpu().double().numpy() - cache["true_quantity"][start:stop].astype(np.float64)))),
        )
        parity["target_duration_max_abs"] = max(
            parity["target_duration_max_abs"],
            float(np.max(np.abs(target_duration.detach().cpu().double().numpy() - cache["true_duration"][start:stop].astype(np.float64)))),
        )
        counts["targets"] += batch_count
        counts["body"] += body_count
        counts["batches"] += 1
        batch_audits.append(transform_audit)
        if batch_index % 8 == 0:
            print(json.dumps({"stage": "value_update", "dataset": dataset_name, "fold": fold, "targets": counts["targets"], "seconds": round(time.monotonic() - started, 2)}), flush=True)

    require(counts["targets"] == 2048, f"{dataset_name} fold {fold} target count drift")
    require(parity["top4_indices_bitwise_equal"], "Cached top-4 parity failed")
    require(parity["prediction_log1p_max_abs"] <= 2e-6 and parity["prediction_max_relative"] <= 2e-6, "Cached prediction parity failed")
    require(parity["target_quantity_max_abs"] == 0.0 and parity["target_duration_max_abs"] == 0.0, "Cached target parity failed")
    # Cached predictions were serialized as float32 by the earlier QKV audit.
    # Their allowed ~2e-6 output roundoff is amplified by the raw-scale and
    # exponential time derivatives.  This auxiliary reconstruction check is
    # therefore looser than the decisive native selected-gradient parity above.
    require(max(parity["cache_hidden_credit_max_relative"].values()) <= 1e-2, f"Cached hidden-credit parity failed: {parity}")
    require(max(parity["selected_gradient_max_relative"].values()) <= 2e-6, f"Selected-value analytic parity failed: {parity}")
    for metric, values in metric_sums.items():
        denominator = counts["body"] if metric == "body" else counts["targets"]
        for value in values:
            value.div_(float(denominator))
    hard_bank = hard_bank_sum / float(counts["targets"])
    shifted_bank = shifted_bank_sums / float(counts["targets"])
    hard_postclip = hard_postclip_sum / float(counts["targets"])
    shifted_postclip = shifted_postclip_sums / float(counts["targets"])
    return {
        "counts": counts,
        "parity": parity,
        "batch_transform": {
            "mean_active_rows": float(np.mean([row["active_rows"] for row in batch_audits])),
            "min_active_rows": int(min(row["active_rows"] for row in batch_audits)),
            "max_active_rows": int(max(row["active_rows"] for row in batch_audits)),
            "max_norm_relative_error": float(max(abs(row["transformed_norm"] - row["hard_norm"]) / max(1e-30, row["hard_norm"]) for row in batch_audits)),
        },
        "metric_gradients": metric_sums,
        "preclip_bank": {"hard": hard_bank, "arms": shifted_bank},
        "postclip_full": {"hard": hard_postclip, "arms": shifted_postclip},
        "descriptor": descriptor,
        "layout": {
            "total_parameters": total_parameters,
            "memory_start": memory_start,
            "memory_end": memory_end,
        },
    }


def alignment(source: torch.Tensor, held: torch.Tensor) -> dict[str, Any]:
    source = source.reshape(-1).double()
    held = held.reshape(-1).double()
    dot = float(torch.dot(source, held))
    source_norm = float(torch.linalg.vector_norm(source))
    held_norm = float(torch.linalg.vector_norm(held))
    denominator = source_norm * held_norm
    return {
        "dot": dot,
        "cosine": dot / denominator if denominator > 1e-30 else 0.0,
        "source_norm": source_norm,
        "held_norm": held_norm,
        "predicted_first_order_change_for_negative_step": -dot,
    }


def scope_decision(
    hard: torch.Tensor,
    arms: torch.Tensor,
    held_metrics: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    candidate = arms[0]
    delta = candidate - hard
    result: dict[str, Any] = {"metrics": {}}
    all_checks: list[bool] = []
    for metric in ("log", "raw", "body", "time"):
        held = held_metrics[metric].reshape(-1).double()
        hard_row = alignment(hard, held)
        candidate_row = alignment(candidate, held)
        delta_row = alignment(delta, held)
        epsilon = 1e-12 * float(torch.linalg.vector_norm(held)) * float(torch.linalg.vector_norm(delta))
        shuffled_delta_dots = torch.mv(arms[1:] - hard.unsqueeze(0), held)
        q95 = float(torch.quantile(shuffled_delta_dots, 0.95, interpolation="linear"))
        checks: dict[str, bool] = {
            "preserve_positive_hard_direction": not (hard_row["dot"] > epsilon) or candidate_row["dot"] > epsilon,
        }
        if metric in ("log", "raw"):
            checks.update(
                absolute_positive=candidate_row["dot"] > epsilon,
                incremental_over_b=delta_row["dot"] > epsilon,
                exceeds_shuffle_p95=delta_row["dot"] > q95,
            )
        else:
            checks["incremental_non_worsening"] = delta_row["dot"] >= -epsilon
        all_checks.extend(checks.values())
        result["metrics"][metric] = {
            "hard": hard_row,
            "candidate": candidate_row,
            "candidate_minus_hard": delta_row,
            "epsilon": epsilon,
            "shuffle_incremental_dot_p95": q95,
            "shuffle_incremental_dot_min": float(shuffled_delta_dots.min()),
            "shuffle_incremental_dot_max": float(shuffled_delta_dots.max()),
            "checks": checks,
            "passed": all(checks.values()),
        }
    result["passed"] = all(all_checks)
    return result


def cross_fold_decision(folds: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source, held in ((0, 1), (1, 0)):
        row: dict[str, Any] = {}
        held_full = {metric: flatten(folds[held]["metric_gradients"][metric]) for metric in ("log", "raw", "body", "time")}
        start = int(folds[held]["layout"]["memory_start"])
        end = int(folds[held]["layout"]["memory_end"])
        held_bank = {metric: value[start:end] for metric, value in held_full.items()}
        row["preclip_selected_value_bank"] = scope_decision(
            folds[source]["preclip_bank"]["hard"],
            folds[source]["preclip_bank"]["arms"],
            held_bank,
        )
        row["postclip_full_model_update"] = scope_decision(
            folds[source]["postclip_full"]["hard"],
            folds[source]["postclip_full"]["arms"],
            held_full,
        )
        row["passed"] = all(value["passed"] for value in row.values())
        result[f"fold{source}_to_fold{held}"] = row
    return result


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1).double()
    right = right.reshape(-1).double()
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    return float(torch.dot(left, right)) / denominator if denominator > 1e-30 else 0.0


def summarize_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    counts = descriptor["selection_counts"]
    mass = counts / counts.sum()
    positive = mass > 0
    entropy_effective = float(torch.exp(-(mass[positive] * torch.log(mass[positive])).sum()))
    top4_mass = float(torch.topk(mass, 4).values.sum())
    coherence: dict[str, list[float]] = {}
    for metric in ("log", "raw", "body", "time"):
        numerator = torch.linalg.vector_norm(descriptor["gradient_sums"][metric], dim=1)
        denominator = descriptor["absolute_contribution_sums"][metric]
        coherence[metric] = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(denominator)).tolist()
    conflicts: dict[str, Any] = {}
    for left, right in (("log", "raw"), ("log", "body"), ("log", "time"), ("raw", "time"), ("body", "time")):
        left_values = descriptor["gradient_sums"][left]
        right_values = descriptor["gradient_sums"][right]
        dots = (left_values * right_values).sum(dim=1)
        left_norms = torch.linalg.vector_norm(left_values, dim=1)
        right_norms = torch.linalg.vector_norm(right_values, dim=1)
        denominators = left_norms * right_norms
        eligible = denominators > 0
        conflict = eligible & (dots < 0)
        slot_cosines = torch.zeros_like(dots)
        slot_cosines[eligible] = dots[eligible] / denominators[eligible]
        conflicts[f"{left}_vs_{right}"] = {
            "eligible_rows": int(eligible.sum()),
            "eligible_slot_ids": torch.nonzero(eligible, as_tuple=False).flatten().tolist(),
            "conflicting_rows": int(conflict.sum()),
            "conflicting_slot_ids": torch.nonzero(conflict, as_tuple=False).flatten().tolist(),
            "conflicting_selection_mass": float(mass[conflict].sum()),
            "global_cosine": safe_cosine(left_values, right_values),
            "slot_cosines": [
                float(slot_cosines[index]) if bool(eligible[index]) else None
                for index in range(counts.numel())
            ],
        }
    stratum_counts = descriptor["stratum_target_counts"]
    strata: list[dict[str, Any]] = []
    for index in range(5):
        row = {
            "index": index,
            "target_count": int(stratum_counts[index]),
            "selection_mass": float(descriptor["stratum_selection_counts"][index].sum() / counts.sum()),
            "active_rows": int((descriptor["stratum_selection_counts"][index] > 0).sum()),
            "gradient_norms": {
                metric: float(torch.linalg.vector_norm(descriptor["stratum_gradient_sums"][metric][index]))
                for metric in ("log", "raw", "body", "time")
            },
        }
        strata.append(row)
    return {
        "active_rows": int(positive.sum()),
        "entropy_effective_rows": entropy_effective,
        "top4_selection_mass": top4_mass,
        "maximum_row_selection_mass": float(mass.max()),
        "selection_counts": counts.tolist(),
        "selection_mass": mass.tolist(),
        "coherence": coherence,
        "metric_conflicts": conflicts,
        "quantity_strata": strata,
    }


def compare_previous_gradient_parity(
    fold: Mapping[str, Any], previous: Mapping[str, Any], fold_id: int
) -> dict[str, Any]:
    mapping = {"log": "hard_log_mse", "raw": "hard_raw_squared_error", "body": "hard_body_mae", "time": "hard_time_nll"}
    result: dict[str, Any] = {}
    start = int(fold["layout"]["memory_start"])
    end = int(fold["layout"]["memory_end"])
    for metric, old_name in mapping.items():
        observed = float(torch.linalg.vector_norm(flatten(fold["metric_gradients"][metric])[start:end]))
        expected = float(previous["folds"][str(fold_id)]["gradient_norms"][old_name]["key"])
        relative = abs(observed - expected) / max(1e-30, abs(expected))
        require(relative <= 2e-6, f"Previous selected-value gradient parity failed: {metric} {relative}")
        result[metric] = {"observed_norm": observed, "previous_norm": expected, "relative_error": relative}
    return result


def analyze_dataset(
    row: Mapping[str, Any],
    previous: Mapping[str, Any],
    frozen: Any,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    name = str(row["dataset"])
    for key, hash_key in (
        ("data_path", "data_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("launch_contract_path", "launch_contract_sha256"),
        ("checkpoint_path", "checkpoint_file_sha256"),
        ("summary_path", "summary_sha256"),
    ):
        base.verify_file(ROOT / row[key], row[hash_key], key)
    identities, cache = base.load_cohort(row)
    extended_names = (
        "true_duration",
        "B__time_intercept_raw",
        "B__time_intercept_clamped",
        "B__integral_term",
        "B__w",
        "B__wd_saturated",
        "B__intercept_saturated",
    )
    with np.load(ROOT / row["cohort_cache_path"], allow_pickle=False) as loaded:
        require(all(name in loaded for name in extended_names), "Extended cache schema drift")
        for cache_name in extended_names:
            cache[cache_name] = loaded[cache_name].copy()
    frame = base.load_train_frame(ROOT / row["data_path"])
    require(frame.height == int(row["train_rows"]), f"{name} train row count drift")
    prepared = frozen.prepare_count_frame(frame)
    dataset = frozen.Dataset(
        prepared,
        lookback_weeks=int(row["lookback_weeks"]),
        max_seq_len=int(row["max_seq_len"]),
        mode="all",
        split_col="chronological_split",
        target_splits={"train"},
    )
    require(len(dataset) == int(row["train_targets"]), f"{name} train target count drift")
    base.verify_dataset_identity(dataset, identities["rows"])
    launch = read_json(ROOT / row["launch_contract_path"])
    boundaries = [float(value) for value in launch["quantity_contract"]["boundaries"]]
    require(len(boundaries) == 4 and boundaries == sorted(boundaries), "Quantity boundaries drift")
    model, restore = frozen.restore_b0(ROOT / row["checkpoint_path"], launch, "cpu")
    require(restore["model_state_sha256"] == row["checkpoint_state_sha256"], "B state hash drift")
    model.eval().requires_grad_(True)
    state_before = frozen.canonical_state_sha256(model.state_dict())
    names, parameters = base.probe_parameters(model)
    fold_results: dict[int, dict[str, Any]] = {}
    for fold in (0, 1):
        fold_rows = [item for item in identities["rows"] if int(item["fold"]) == fold]
        loader = DataLoader(
            Subset(dataset, [int(item["dataset_index"]) for item in fold_rows]),
            batch_size=int(contract["diagnostic_scope"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=frozen.collate,
        )
        fold_results[fold] = fold_probe(
            model,
            frozen,
            loader,
            parameters=parameters,
            names=names,
            body_threshold=float(row["body_threshold_train_p95"]),
            boundaries=boundaries,
            cache=cache,
            cache_offset=fold * 2048,
            dataset_name=name,
            fold=fold,
        )
    state_after = frozen.canonical_state_sha256(model.state_dict())
    require(state_after == state_before == row["checkpoint_state_sha256"], "Frozen B state changed")
    require(all(parameter.grad is None for parameter in model.parameters()), "Parameter .grad was populated")
    parity = {
        str(fold): compare_previous_gradient_parity(fold_results[fold], previous, fold)
        for fold in (0, 1)
    }
    cross_fold = cross_fold_decision(fold_results)
    decision_passed = all(direction["passed"] for direction in cross_fold.values())
    summaries = {str(fold): summarize_descriptor(fold_results[fold]["descriptor"]) for fold in (0, 1)}
    per_slot: list[dict[str, Any]] = []
    per_slot_stratum: list[dict[str, Any]] = []
    for fold in (0, 1):
        summary = summaries[str(fold)]
        descriptor = fold_results[fold]["descriptor"]
        for slot in range(64):
            record: dict[str, Any] = {
                "dataset": name,
                "fold": fold,
                "slot": slot,
                "selection_count": int(summary["selection_counts"][slot]),
                "selection_mass": summary["selection_mass"][slot],
            }
            for metric in ("log", "raw", "body", "time"):
                record[f"{metric}_gradient_norm"] = float(torch.linalg.vector_norm(descriptor["gradient_sums"][metric][slot]))
                record[f"{metric}_coherence"] = summary["coherence"][metric][slot]
            per_slot.append(record)
        total_selections = float(descriptor["selection_counts"].sum())
        for stratum in range(5):
            stratum_counts = descriptor["stratum_selection_counts"][stratum]
            stratum_selections = float(stratum_counts.sum())
            lower = None if stratum == 0 else boundaries[stratum - 1]
            upper = boundaries[stratum] if stratum < 4 else None
            if lower is None:
                label = f"<= {upper:g}"
            elif upper is None:
                label = f"> {lower:g}"
            else:
                label = f"({lower:g}, {upper:g}]"
            for slot in range(64):
                record = {
                    "dataset": name,
                    "fold": fold,
                    "stratum": stratum,
                    "stratum_label": label,
                    "slot": slot,
                    "stratum_target_count": int(
                        descriptor["stratum_target_counts"][stratum]
                    ),
                    "selection_count": int(stratum_counts[slot]),
                    "selection_mass_all": (
                        float(stratum_counts[slot]) / total_selections
                    ),
                    "selection_mass_within_stratum": (
                        float(stratum_counts[slot]) / stratum_selections
                        if stratum_selections > 0
                        else 0.0
                    ),
                }
                for metric in ("log", "raw", "body", "time"):
                    record[f"{metric}_gradient_norm"] = float(
                        torch.linalg.vector_norm(
                            descriptor["stratum_gradient_sums"][metric][stratum][slot]
                        )
                    )
                per_slot_stratum.append(record)
    compact_folds = {
        str(fold): {
            "counts": fold_results[fold]["counts"],
            "parity": fold_results[fold]["parity"],
            "batch_transform": fold_results[fold]["batch_transform"],
            "usage_and_conflict": summaries[str(fold)],
            "previous_gradient_parity": parity[str(fold)],
        }
        for fold in (0, 1)
    }
    return {
        "dataset": name,
        "scope": {
            "split": "train",
            "validation_rows_materialized": False,
            "held_out_rows_materialized": False,
            "model_parameter_updates": False,
        },
        "cohort": {
            "target_count": 4096,
            "fold_counts": identities["fold_counts"],
            "fold_unique_series": identities["fold_unique_series"],
            "sample_ids_sha256": row["cohort_sample_ids_sha256"],
            "cache_sha256": row["cohort_cache_sha256"],
        },
        "checkpoint_state_sha256_before_after": state_before,
        "body_threshold_train_p95": float(row["body_threshold_train_p95"]),
        "quantity_boundaries": boundaries,
        "folds": compact_folds,
        "cross_fold": cross_fold,
        "passed": decision_passed,
    }, per_slot, per_slot_stratum


def write_per_slot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "No per-slot rows")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def source_hashes() -> dict[str, str]:
    paths = (
        "paper/contracts/hard_lmm_value_update_probe_v1.json",
        "paper/scripts/hard_lmm_value_update_probe.py",
        "paper/scripts/run_hard_lmm_value_update_probe.py",
        "paper/contracts/hard_lmm_credit_assignment_probe_v1.json",
        "paper/scripts/hard_lmm_credit_assignment_probe.py",
        "paper/scripts/run_hard_lmm_credit_assignment_probe.py",
    )
    return {path: sha256_file(ROOT / path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    args = parser.parse_args()
    require(not RESULT.exists(), f"Refusing to overwrite {RESULT}")
    require(HELPER.is_file(), "Pure helper module is missing")
    contract = read_json(CONTRACT)
    validate_contract(contract)
    base_contract, previous_analysis = verify_reused_evidence(contract)
    source = base.verify_frozen_source(base_contract, args.frozen_source_root)
    frozen = base.import_frozen_modules(args.frozen_source_root)
    source["loaded_module_audit"] = base.audit_loaded_frozen_modules(
        base_contract, args.frozen_source_root.resolve()
    )
    torch.set_num_threads(int(contract["diagnostic_scope"]["torch_threads"]))
    torch.manual_seed(int(contract["diagnostic_scope"]["sample_seed"]))
    RESULT.mkdir(parents=True)
    save_json(RESULT / "execution_contract.json", contract)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "contract_sha256": sha256_file(CONTRACT),
        "source": source,
        "diagnostic_source_hashes": source_hashes(),
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "validation_rows_materialized": False,
        "held_out_rows_materialized": False,
        "model_parameter_updates": False,
    }
    save_json(RESULT / "execution_manifest.json", manifest)
    try:
        results: dict[str, Any] = {}
        per_slot: list[dict[str, Any]] = []
        per_slot_stratum: list[dict[str, Any]] = []
        for row in base_contract["datasets"]:
            result, slot_rows, slot_stratum_rows = analyze_dataset(
                row, previous_analysis[str(row["dataset"])], frozen, contract
            )
            results[str(row["dataset"])] = result
            per_slot.extend(slot_rows)
            per_slot_stratum.extend(slot_stratum_rows)
        save_json(RESULT / "analysis.json", results)
        write_per_slot(RESULT / "per_slot.csv", per_slot)
        write_per_slot(RESULT / "per_slot_stratum.csv", per_slot_stratum)
        passed = bool(results) and all(row["passed"] for row in results.values())
        decision = {
            "contract_id": contract["contract_id"],
            "datasets": {name: {"passed": row["passed"]} for name, row in results.items()},
            "passed": passed,
            "action": (
                "contract_and_implement_usage_normalized_prototype_training_rule"
                if passed
                else "do_not_implement_or_gpu_train_usage_normalized_prototype_rule"
            ),
            "interpretation": "This is a first-order train-only necessary-condition result, not validation performance.",
        }
        save_json(RESULT / "evidence_decision.json", decision)
        require(source_hashes() == manifest["diagnostic_source_hashes"], "Diagnostic source changed during run")
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            analysis_sha256=sha256_file(RESULT / "analysis.json"),
            per_slot_sha256=sha256_file(RESULT / "per_slot.csv"),
            per_slot_stratum_sha256=sha256_file(
                RESULT / "per_slot_stratum.csv"
            ),
            decision_sha256=sha256_file(RESULT / "evidence_decision.json"),
            common_gate_passed=passed,
        )
        save_json(RESULT / "execution_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", failed_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        save_json(RESULT / "execution_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
