"""Frozen, observed-history features for the original and separate-key Hard-LMM.

This module does not load data, choose a split, fit a model, or select a candidate.
Callers own the train-only row scope. Target quantities appear only as labels;
neither target quantities nor target gaps enter the representation or statistics.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.TPPs.CountAwareTPP import TITAN_MEMORY_MODE_STATIC_HARD
from models.Titan.common.key_value_memory import KEY_VALUE_MEMORY_MODE, KeyValueLocalMemoryMatcher
from models.Titan.common.memory import HardLocalMemoryMatcher
from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch


@torch.no_grad()
def extract(model, dts: torch.Tensor, mask: torch.Tensor,
            quantities: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return one CPU cache row per next-event target, with no parameter updates.

    ``stats`` contains the mean observed log1p quantity and the last observed
    log1p quantity minus that mean. ``h`` and ``stats`` are the proposed query
    inputs; ``quantity`` and ``log_residual`` are diagnostic labels only.
    ``value_projections`` are individual selected values projected onto the
    quantity-head weight; their actual weighted sum equals ``projection``.
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Diagnostic model must be frozen and in eval mode")
    lmm = getattr(model, "lmm", None)
    mode = getattr(model, "memory_mode", None)
    if not (
        (type(lmm) is HardLocalMemoryMatcher and mode == TITAN_MEMORY_MODE_STATIC_HARD)
        or (type(lmm) is KeyValueLocalMemoryMatcher and mode == KEY_VALUE_MEMORY_MODE)
    ) or lmm.topk != 4 or lmm.mem_size < 4:
        raise ValueError("Diagnostic supports only original or separate-key static top-4 Hard-LMM")
    if dts.ndim != 2 or dts.shape != quantities.shape or dts.shape != mask.shape:
        raise ValueError("dts, quantities and mask must share shape [batch, sequence]")
    if mask.dtype != torch.bool or dts.size(0) == 0:
        raise ValueError("A nonempty batch and boolean mask are required")
    if not torch.isfinite(dts[mask]).all() or not torch.isfinite(quantities[mask]).all():
        raise ValueError("Observed and target inputs must be finite")
    if bool((quantities[mask] < 0).any()):
        raise ValueError("Count quantities must be nonnegative")

    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    rows = torch.arange(len(lengths), device=dts.device)
    target, previous = lengths - 1, lengths - 2
    true_quantity = quantities[rows, target].float().clone()
    history_mask = mask.clone()
    history_mask[rows, target] = False
    history_qty = quantities.masked_fill(~history_mask, 0)
    history_dt = dts.masked_fill(~history_mask, 0)

    local = model._encode_base(
        history_dt, history_qty, mask, memory_write_mask=history_mask
    )
    residual, trace = lmm.retrieve(local)
    h, r = local[rows, previous], residual[rows, previous]
    indices = trace["prototype_indices"][rows, previous]
    scores = trace["topk_similarity"][rows, previous]
    is_separate_key = type(lmm) is KeyValueLocalMemoryMatcher
    weights = (
        trace["retrieval_weights"][rows, previous]
        if is_separate_key else torch.full_like(scores, 0.25)
    )

    # Recompute from the addressing bank, not the value bank of a K/V model.
    keys = lmm.memory_keys if is_separate_key else lmm.mem
    all_scores = F.normalize(h, dim=-1) @ F.normalize(keys[0], dim=-1).T
    selected_scores = all_scores.gather(1, indices)
    torch.testing.assert_close(scores, selected_scores, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        scores, all_scores.topk(4, dim=-1).values, rtol=1e-5, atol=1e-6
    )
    expected_weights = (
        (selected_scores / lmm.temperature).softmax(-1)
        if is_separate_key else torch.full_like(scores, 0.25)
    )
    torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-6)
    selected_values = lmm.mem[0][indices]
    torch.testing.assert_close(
        r, (selected_values * weights.unsqueeze(-1)).sum(1), rtol=1e-5, atol=1e-6
    )

    log_history_qty = history_qty.float().clamp_min(0).log1p()
    history_length = lengths - 1
    mean_log_quantity = log_history_qty.sum(1) / history_length
    stats = torch.stack((
        mean_log_quantity,
        log_history_qty[rows, previous] - mean_log_quantity,
    ), dim=-1)
    z = model.quantity_head(h + r).squeeze(-1)
    log_prediction = F.softplus(z)
    head_weight = model.quantity_head.weight[0]
    value_projections = (selected_values * head_weight).sum(-1)
    projection = (r * head_weight).sum(-1)
    torch.testing.assert_close(
        projection, (value_projections * weights).sum(-1), rtol=1e-5, atol=1e-6
    )
    result = {
        "h": h,
        "stats": stats,
        "log_residual": true_quantity.log1p() - log_prediction,
        "z": z,
        "prediction": log_prediction.expm1(),
        "quantity": true_quantity,
        "projection": projection,
        "h_norm": h.norm(dim=-1),
        "r_norm": r.norm(dim=-1),
        "indices": indices,
        "weights": weights,
        "scores": scores,
        "value_projections": value_projections,
        "history_length": history_length,
    }
    for name, tensor in result.items():
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Non-finite diagnostic tensor: {name}")
    return {name: tensor.detach().cpu() for name, tensor in result.items()}
