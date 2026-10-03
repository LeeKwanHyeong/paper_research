from __future__ import annotations

from typing import Any

import torch


def predict_value_for_marks(
    model: Any,
    h: torch.Tensor,
    marks: torch.Tensor,
) -> torch.Tensor:
    """
    Select residual predictions for explicit next marks.

    TitanTPP V3 exposes one residual per real mark. Shared-head models keep the
    legacy behavior through the `predict_value` fallback.
    """
    predict_by_mark = getattr(model, "predict_value_by_mark", None)
    if not callable(predict_by_mark):
        return model.predict_value(h)

    values_by_mark = predict_by_mark(h)
    if values_by_mark.ndim != h.ndim:
        raise ValueError(
            "predict_value_by_mark must return shape [..., num_real_marks]."
        )
    real_mark_count = int(values_by_mark.size(-1))
    if real_mark_count < 1:
        raise ValueError("predict_value_by_mark returned no real-mark experts.")

    safe_marks = marks.long().clamp(min=0, max=real_mark_count - 1)
    return values_by_mark.gather(-1, safe_marks.unsqueeze(-1)).squeeze(-1)


def restrict_to_last_transition(step_mask: torch.Tensor) -> torch.Tensor:
    """
    Keep only the final valid autoregressive transition in each sample.

    Week-lookback samples are left-padded and end with the target event, so this
    mask aligns train loss with the validation/test "next event only" metric.
    """
    if step_mask.ndim != 2:
        raise ValueError("step_mask must have shape [batch, transitions].")

    target_mask = torch.zeros_like(step_mask, dtype=torch.bool)
    positions = torch.arange(step_mask.size(1), device=step_mask.device).view(1, -1)
    last_positions = torch.where(step_mask, positions, torch.full_like(positions, -1)).max(dim=1).values
    has_target = last_positions >= 0
    if has_target.any():
        target_mask[has_target, last_positions[has_target]] = True
    return target_mask


def apply_transition_loss_scope(step_mask: torch.Tensor, loss_scope: str = "all") -> torch.Tensor:
    """
    Select which transitions contribute to the likelihood/value losses.
    """
    scope = str(loss_scope or "all").strip().lower()
    if scope == "all":
        return step_mask
    if scope == "target_only":
        return restrict_to_last_transition(step_mask)
    raise ValueError(f"Unsupported train_loss_scope='{loss_scope}'. Use 'all' or 'target_only'.")


def mask_appended_target_value(values: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor | None:
    """
    Remove the appended target value from model inputs while keeping history.

    The dataloader appends the target event to the right side of each window.
    Causal encoders should not look ahead, but zeroing the final valid value is
    a defensive guard against accidental non-causal use of target quantity.
    """
    if values is None:
        return None
    if mask is None:
        safe_values = values.clone()
        if safe_values.size(1) > 0:
            safe_values[:, -1] = 0.0
        return safe_values

    safe_values = values.clone()
    positions = torch.arange(mask.size(1), device=mask.device).view(1, -1)
    last_positions = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values
    has_target = last_positions >= 0
    if has_target.any():
        safe_values[has_target, last_positions[has_target]] = 0.0
    return safe_values


def build_value_input_feature(
    *,
    marks: torch.Tensor,
    values: torch.Tensor | None,
    cfg: Any,
    mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    Build the optional continuous quantity-mark input feature.

    Modes:
    - none: disabled, preserves the original mark + dt input.
    - residual: observed scale_residual_t only.
    - log_qty: observed log_base(qty_t) = mark_t + scale_residual_t.
    """
    mode = str(getattr(cfg, "value_input_mode", "none") or "none").strip().lower()
    if mode == "none":
        return None

    if values is None:
        feature = torch.zeros_like(marks, dtype=torch.float32)
    elif mode == "residual":
        feature = values.float()
    elif mode == "log_qty":
        pad_id = int(getattr(cfg, "num_marks", 1) - 1)
        feature = marks.clamp(min=0, max=max(pad_id - 1, 0)).float() + values.float()
    else:
        raise ValueError(
            f"Unsupported value_input_mode='{mode}'. Use 'none', 'residual', or 'log_qty'."
        )

    if mask is not None:
        feature = feature * mask.float()
    return feature
