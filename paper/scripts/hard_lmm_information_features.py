"""Frozen observed-history representations; no data loading or probe fitting.

Every stage is extracted from the same model call. Input/history stages never
contain the appended target. The final target quantity is read only after the
representations and input summaries have been constructed.
"""

from __future__ import annotations

import torch

from models.TPPs.CountAwareTPP import TITAN_MEMORY_MODE_STATIC_HARD
from models.Titan.common.key_value_memory import KEY_VALUE_MEMORY_MODE, KeyValueLocalMemoryMatcher
from models.Titan.common.memory import HardLocalMemoryMatcher
from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch


HISTORY_FEATURE_NAMES = (
    "log_history_length", "mean_log_quantity", "last_log_quantity",
    "std_log_quantity", "recent3_mean_log_quantity", "last_minus_mean_log_quantity",
    "normalized_rank_log_quantity_slope", "mean_internal_log_gap", "last_internal_log_gap",
    "std_internal_log_gap", "log_internal_span", "age_distortion",
)
STAGE_NAMES = (
    "input_last", "input_mean", "layer1_last", "layer1_mean",
    "layer2_last", "layer2_mean", "h", "r", "fused", "concat_hr",
)


def _validate_observed(dts, quantities, mask):
    if dts.ndim != 2 or dts.shape != quantities.shape or dts.shape != mask.shape:
        raise ValueError("dts, quantities and mask must share shape [batch, sequence]")
    if mask.dtype != torch.bool or not dts.size(0):
        raise ValueError("A nonempty batch and boolean mask are required")
    if any(tensor.device != dts.device for tensor in (quantities, mask)):
        raise ValueError("Inputs must share the same device")
    if bool((mask.sum(1) < 1).any()):
        raise ValueError("Every sample requires an observed history")
    for tensor in (dts, quantities):
        if not bool(torch.isfinite(tensor[mask]).all()):
            raise ValueError("Observed inputs must be finite")
        if bool((tensor[mask] < 0).any()):
            raise ValueError("Observed inputs must be nonnegative")


@torch.no_grad()
def observed_history_features(dts, quantities, observed_mask):
    """Return F12 [B,12], with reductions in float64 and final float32.

    Order is the valid-token order, independent of padding placement. Internal
    gap summaries exclude the first retained event's boundary gap. A singleton
    has zero slope/internal-gap summaries/J; J is zero for H<3 or zero span.
    """
    _validate_observed(dts, quantities, observed_mask)
    valid = observed_mask
    gaps = dts.masked_fill(~valid, 0).double()
    logq = quantities.masked_fill(~valid, 0).double().log1p()
    loggap = gaps.log1p()
    length = valid.sum(1)
    count = length.double()
    ranks = valid.long().cumsum(1) - 1
    last = valid & (ranks == length[:, None] - 1)
    meanq = logq.sum(1) / count
    lastq = (logq * last).sum(1)
    stdq = (((logq - meanq[:, None]).square() * valid).sum(1) / count).sqrt()
    recent = valid & (ranks >= (length - 3).clamp_min(0)[:, None])
    recentq = (logq * recent).sum(1) / recent.sum(1)

    normalized_ranks = ranks.double() / (count - 1).clamp_min(1)[:, None]
    rank_mean = (normalized_ranks * valid).sum(1) / count
    centered_rank = (normalized_ranks - rank_mean[:, None]) * valid
    slope = (centered_rank * (logq - meanq[:, None])).sum(1) / centered_rank.square().sum(1).clamp_min(1e-300)

    internal = valid & (ranks > 0)
    internal_count = internal.sum(1).clamp_min(1).double()
    meanloggap = (loggap * internal).sum(1) / internal_count
    lastloggap = (loggap * last * internal).sum(1)
    stdloggap = (((loggap - meanloggap[:, None]).square() * internal).sum(1) / internal_count).sqrt()
    internal_gaps = gaps * internal
    elapsed = internal_gaps.cumsum(1)
    span = internal_gaps.sum(1)
    ages = (span[:, None] - elapsed) / span.clamp_min(1e-300)[:, None]
    event_ages = (count[:, None] - 1 - ranks) / (count - 1).clamp_min(1)[:, None]
    distortion = (((ages - event_ages).square() * valid).sum(1) / count).sqrt()
    distortion = torch.where((length >= 3) & (span > 0), distortion, 0.)
    features = torch.stack((count.log1p(), meanq, lastq, stdq, recentq,
                            lastq - meanq, slope, meanloggap, lastloggap,
                            stdloggap, span.log1p(), distortion), dim=1).float()
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("Nonfinite observed history summary")
    return features


@torch.no_grad()
def extract(model, dts, mask, quantities):
    """Extract paired last/mean states and F12 from frozen original or K/V.

    ``dts,mask,quantities`` follow the loader contract: valid tokens in event
    order, final valid token is the target, padding may be on either side.
    Returned tensors are detached on CPU. Target gap is never consulted.
    """
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Diagnostic model must be frozen and in eval mode")
    lmm, mode = getattr(model, "lmm", None), getattr(model, "memory_mode", None)
    if not ((type(lmm) is HardLocalMemoryMatcher and mode == TITAN_MEMORY_MODE_STATIC_HARD)
            or (type(lmm) is KeyValueLocalMemoryMatcher and mode == KEY_VALUE_MEMORY_MODE)):
        raise ValueError("Only original or separate-key static Hard-LMM is supported")
    if lmm.topk != 4 or len(model.encoder.layers) != 2:
        raise ValueError("The frozen diagnostic requires top4 and two encoder layers")
    if dts.ndim != 2 or dts.shape != quantities.shape or dts.shape != mask.shape:
        raise ValueError("dts, quantities and mask must share shape [batch, sequence]")
    if mask.dtype != torch.bool or not dts.size(0):
        raise ValueError("A nonempty batch and boolean mask are required")
    if any(tensor.device != dts.device for tensor in (quantities, mask)):
        raise ValueError("Inputs must share the same device")
    right_dt, right_qty, right_mask, full_lengths = right_pad_batch(dts, quantities, mask)
    rows = torch.arange(len(full_lengths), device=dts.device)
    targets, previous = full_lengths - 1, full_lengths - 2
    observed_mask = right_mask.clone()
    observed_mask[rows, targets] = False
    history_dt = right_dt.masked_fill(~observed_mask, 0)
    history_qty = right_qty.masked_fill(~observed_mask, 0)
    _validate_observed(history_dt, history_qty, observed_mask)
    summaries = observed_history_features(history_dt, history_qty, observed_mask)

    captures, handles = {}, []
    try:
        def capture_input(_module, args):
            captures["input"] = args[0]

        def capture_layer(name):
            def hook(_module, _args, output):
                captures[name] = output
            return hook

        handles.append(model.encoder.layers[0].register_forward_pre_hook(capture_input))
        for index, layer in enumerate(model.encoder.layers):
            handles.append(layer.register_forward_hook(capture_layer(f"layer{index + 1}")))
        local = model._encode_base(history_dt, history_qty, observed_mask,
                                   memory_write_mask=observed_mask)
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != {"input", "layer1", "layer2"}:
        raise RuntimeError("Incomplete encoder stage captures")
    result = {}
    for name, representation in captures.items():
        if representation.shape != local.shape:
            raise RuntimeError("Encoder stage shape changed")
        result[f"{name}_last"] = representation[rows, previous]
        result[f"{name}_mean"] = (representation * observed_mask.unsqueeze(-1)).sum(1) / (full_lengths - 1).unsqueeze(-1)
    residual, _ = lmm.retrieve(local)
    h, r = local[rows, previous], residual[rows, previous]
    fused = h + r
    z = model.quantity_head(fused).squeeze(-1)
    log_prediction, prediction = model.predict_quantity(fused)
    result.update(h=h, r=r, fused=fused, concat_hr=torch.cat((h, r), dim=-1),
                  history_features=summaries, history_length=full_lengths - 1,
                  z=z, base_log_prediction=log_prediction, base_logpred=log_prediction,
                  prediction=prediction)
    # The target label is read only here, after all input representations exist.
    target_quantity = right_qty[rows, targets].float().clone()
    if not bool(torch.isfinite(target_quantity).all()):
        raise ValueError("Target quantities must be finite")
    if bool((target_quantity < 0).any()):
        raise ValueError("Target quantities must be nonnegative")
    result.update(quantity=target_quantity, log_quantity=target_quantity.log1p())
    for name, tensor in result.items():
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Nonfinite extracted tensor: {name}")
    return {name: tensor.detach().cpu() for name, tensor in result.items()}
