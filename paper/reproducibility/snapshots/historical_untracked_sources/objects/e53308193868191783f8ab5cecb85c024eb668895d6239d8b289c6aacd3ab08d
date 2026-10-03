from __future__ import annotations

import torch
import torch.nn.functional as F


def normalized_ranked_probability_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_real_marks: int,
) -> torch.Tensor:
    """Return one normalized ordinal RPS value per target."""
    if logits.ndim != targets.ndim + 1:
        raise ValueError("logits must have exactly one more dimension than targets.")
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets must share their leading dimensions.")

    class_count = int(num_real_marks)
    if class_count < 1 or class_count > logits.size(-1):
        raise ValueError("num_real_marks must be within the logits class dimension.")
    if class_count == 1:
        return logits[..., 0] * 0.0

    real_probs = F.softmax(logits[..., :class_count], dim=-1)
    predicted_cdf = real_probs.cumsum(dim=-1)[..., :-1]
    thresholds = torch.arange(
        class_count - 1,
        device=logits.device,
        dtype=targets.dtype,
    )
    safe_targets = targets.clamp(min=0, max=class_count - 1)
    target_cdf = (safe_targets.unsqueeze(-1) <= thresholds).to(dtype=real_probs.dtype)
    return (predicted_cdf - target_cdf).square().mean(dim=-1)


def masked_normalized_ranked_probability_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_real_marks: int,
) -> torch.Tensor:
    """Average normalized RPS over the same valid transitions as marker CE."""
    if mask.shape != targets.shape:
        raise ValueError("mask and targets must have the same shape.")
    scores = normalized_ranked_probability_score(
        logits,
        targets,
        num_real_marks=num_real_marks,
    )
    weights = mask.to(dtype=scores.dtype)
    return (scores * weights).sum() / weights.sum().clamp_min(1.0)
