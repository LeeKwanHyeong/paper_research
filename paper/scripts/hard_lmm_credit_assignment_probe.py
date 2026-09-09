"""Pure helpers for the Hard-LMM retrieval credit-assignment diagnostic.

The proposed surrogate is zero in the forward pass.  Its only purpose is to
add a continuous cosine-addressing gradient to the quantity graph while the
time graph, selected-value gradient, and every prediction remain those of B.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch.nn import functional as F


EPS = 1e-30


def cyclic_derangement(size: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return the fixed result-independent value permutation used by the sham."""
    if size < 2:
        raise ValueError("A derangement requires at least two prototypes")
    return torch.arange(size, device=device).roll(1)


def cosine_soft_read(
    query: torch.Tensor,
    key_memory: torch.Tensor,
    *,
    value_memory: torch.Tensor | None = None,
    temperature: float = 1.0,
    value_permutation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read every prototype with soft cosine weights and detached values.

    ``key_memory`` stays in autograd because the candidate needs a key-role
    gradient.  Values are always detached so the existing hard selected-value
    gradient is not duplicated by the surrogate.
    """
    if query.ndim != 2:
        raise ValueError("query must have shape [batch, hidden]")
    if key_memory.ndim != 2 or key_memory.size(1) != query.size(1):
        raise ValueError("key_memory must have shape [memory, hidden]")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = key_memory if value_memory is None else value_memory
    if values.shape != key_memory.shape:
        raise ValueError("value_memory must have the same shape as key_memory")
    if value_permutation is not None:
        if value_permutation.shape != (key_memory.size(0),):
            raise ValueError("value_permutation must have shape [memory]")
        if value_permutation.dtype != torch.long:
            raise ValueError("value_permutation must use torch.long indices")
        if not torch.equal(
            value_permutation.detach().cpu().sort().values,
            torch.arange(key_memory.size(0)),
        ):
            raise ValueError("value_permutation must contain each memory row once")
        values = values[value_permutation]
    scores = F.normalize(query, p=2, dim=-1) @ F.normalize(
        key_memory, p=2, dim=-1
    ).transpose(0, 1)
    weights = torch.softmax(scores / float(temperature), dim=-1)
    read = weights @ values.detach()
    return read, scores, weights


def forward_identical_hidden(
    hard_hidden: torch.Tensor,
    soft_read: torch.Tensor,
) -> torch.Tensor:
    """Return B exactly in forward while retaining the soft-read backward."""
    if hard_hidden.shape != soft_read.shape:
        raise ValueError("hard_hidden and soft_read must have identical shapes")
    return hard_hidden + (soft_read - soft_read.detach())


def quantity_credit_signal(
    hard_hidden: torch.Tensor,
    quantity_weight: torch.Tensor,
    quantity_bias: torch.Tensor | None,
    target_quantity: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return B predictions and d(sum log-MSE)/d(hidden).

    The returned credit signal is detached before it is paired with a soft
    read.  This is algebraically the extra gradient induced by the zero-valued
    surrogate, without subtracting nearly equal full-model gradients.
    """
    if quantity_weight.ndim == 2 and quantity_weight.size(0) == 1:
        weight = quantity_weight.squeeze(0)
    elif quantity_weight.ndim == 1:
        weight = quantity_weight
    else:
        raise ValueError("quantity_weight must have shape [1, hidden] or [hidden]")
    if hard_hidden.ndim != 2 or hard_hidden.size(1) != weight.numel():
        raise ValueError("hard_hidden has the wrong shape")
    if target_quantity.shape != (hard_hidden.size(0),):
        raise ValueError("target_quantity must have shape [batch]")
    preactivation = hard_hidden @ weight
    if quantity_bias is not None:
        preactivation = preactivation + quantity_bias.reshape(())
    location = F.softplus(preactivation)
    target_log = torch.log1p(target_quantity.clamp_min(0.0))
    log_error = location - target_log
    log_mse = log_error.square()
    raw_prediction = torch.expm1(location)
    raw_squared_error = (raw_prediction - target_quantity).square()
    scalar = 2.0 * log_error * torch.sigmoid(preactivation)
    hidden_credit = scalar.unsqueeze(-1) * weight.unsqueeze(0)
    return {
        "preactivation": preactivation,
        "location": location,
        "log_mse": log_mse,
        "raw_prediction": raw_prediction,
        "raw_squared_error": raw_squared_error,
        "hidden_credit": hidden_credit.detach(),
    }


def replace_none_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
) -> list[torch.Tensor]:
    if len(parameters) != len(gradients):
        raise ValueError("parameter and gradient counts differ")
    return [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ]


def canonical_vector(
    names: Sequence[str],
    tensors: Sequence[torch.Tensor],
    *,
    block: str = "all",
) -> torch.Tensor:
    """Flatten gradients in their caller-supplied canonical name order."""
    if len(names) != len(tensors):
        raise ValueError("name and tensor counts differ")
    pieces: list[torch.Tensor] = []
    for name, tensor in zip(names, tensors):
        include = (
            block == "all"
            or (block == "query" and name.startswith("encoder."))
            or (block == "key" and name == "lmm.mem")
            or (
                block == "head"
                and not name.startswith("encoder.")
                and name != "lmm.mem"
            )
        )
        if include:
            pieces.append(tensor.detach().reshape(-1).to(dtype=torch.float64, device="cpu"))
    if not pieces:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(pieces)


def directional_alignment(
    source_additional_gradient: torch.Tensor,
    held_metric_gradient: torch.Tensor,
) -> dict[str, float | bool]:
    """Measure whether update ``-source`` lowers the held metric to first order."""
    source = source_additional_gradient.to(dtype=torch.float64, device="cpu").reshape(-1)
    held = held_metric_gradient.to(dtype=torch.float64, device="cpu").reshape(-1)
    if source.shape != held.shape or source.numel() == 0:
        raise ValueError("gradient vectors must be nonempty and have equal shape")
    if not bool(torch.isfinite(source).all() and torch.isfinite(held).all()):
        raise ValueError("gradient vectors must be finite")
    dot = float(torch.dot(source, held))
    source_norm = float(torch.linalg.vector_norm(source))
    held_norm = float(torch.linalg.vector_norm(held))
    denominator = source_norm * held_norm
    cosine = dot / denominator if denominator > EPS else 0.0
    return {
        "dot": dot,
        "cosine": cosine,
        "source_norm": source_norm,
        "held_norm": held_norm,
        "predicted_first_order_change_for_negative_source_step": -dot,
        "improvement_direction": dot > 0.0,
    }


def evaluate_dataset_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    structural = result["structural_audit"]
    structural_pass = bool(
        structural["normal_forward_bitwise_identity"]
        and structural["shuffled_forward_bitwise_identity"]
        and structural["normal_prediction_bitwise_identity"]
        and structural["shuffled_prediction_bitwise_identity"]
        and structural["hard_query_addressing_gradient_max_abs"] <= 1e-12
        and structural["hard_key_addressing_gradient_max_abs"] <= 1e-12
        and structural["normal_query_addressing_gradient_l2"] > 1e-12
        and structural["normal_key_addressing_gradient_l2"] > 1e-12
    )
    decisions: dict[str, Any] = {}
    for direction, row in result["cross_fold"].items():
        normal = row["normal"]
        shuffled = row["shuffled"]
        checks = {
            "log_mse_positive": normal["log_mse"]["dot"] > 0.0,
            "raw_squared_error_positive": normal["raw_squared_error"]["dot"] > 0.0,
            "body_mae_no_conflict": normal["body_mae"]["dot"] >= 0.0,
            "log_mse_beats_shuffled": (
                normal["log_mse"]["cosine"]
                > shuffled["log_mse"]["cosine"]
            ),
            "raw_squared_error_beats_shuffled": (
                normal["raw_squared_error"]["cosine"]
                > shuffled["raw_squared_error"]["cosine"]
            ),
        }
        if "normal_clipped" in row and "shuffled_clipped" in row:
            normal_clipped = row["normal_clipped"]
            shuffled_clipped = row["shuffled_clipped"]
            post_clip_checks = {
                "log_mse_positive": normal_clipped["log_mse"]["dot"] > 0.0,
                "raw_squared_error_positive": (
                    normal_clipped["raw_squared_error"]["dot"] > 0.0
                ),
                "body_mae_no_conflict": normal_clipped["body_mae"]["dot"] >= 0.0,
                "log_mse_beats_shuffled": (
                    normal_clipped["log_mse"]["cosine"]
                    > shuffled_clipped["log_mse"]["cosine"]
                ),
                "raw_squared_error_beats_shuffled": (
                    normal_clipped["raw_squared_error"]["cosine"]
                    > shuffled_clipped["raw_squared_error"]["cosine"]
                ),
            }
        else:
            post_clip_checks = {}
        checks.update(
            {f"post_clip_{name}": value for name, value in post_clip_checks.items()}
        )
        decisions[direction] = {"checks": checks, "passed": all(checks.values())}
    directional_pass = all(row["passed"] for row in decisions.values())
    return {
        "structural_pass": structural_pass,
        "directional": decisions,
        "directional_pass": directional_pass,
        "passed": structural_pass and directional_pass,
    }


def aggregate_common_gate(dataset_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    decisions = {
        dataset: evaluate_dataset_gate(result)
        for dataset, result in dataset_results.items()
    }
    passed = bool(decisions) and all(row["passed"] for row in decisions.values())
    return {
        "datasets": decisions,
        "passed": passed,
        "action": (
            "freeze_separate_candidate_implementation_contract"
            if passed
            else "do_not_implement_or_gpu_train_candidate"
        ),
    }


__all__ = [
    "aggregate_common_gate",
    "canonical_vector",
    "cosine_soft_read",
    "cyclic_derangement",
    "directional_alignment",
    "evaluate_dataset_gate",
    "forward_identical_hidden",
    "quantity_credit_signal",
    "replace_none_gradients",
]
