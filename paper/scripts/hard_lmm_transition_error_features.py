"""Causal transition-error features for a static top-k Hard-LMM bank.

The helper is intentionally independent of model and runner code.  A transition
at position ``t`` compares the quantity observed at ``t`` with the base quantity
prediction made by state ``t - 1``.  Its error is written to the prototypes
selected by that preceding state and becomes visible from state ``t`` onward.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def transition_error_features(
    base_pre_softplus_logits: torch.Tensor,
    observed_log1p_quantities: torch.Tensor,
    top4_prototype_indices: torch.Tensor,
    observed_mask: torch.Tensor,
    prototype_count: int,
    memory_write_mask: torch.Tensor | None = None,
    write_top4_prototype_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build causal prefix statistics from observed one-step quantity errors.

    Args:
        base_pre_softplus_logits: Base quantity-head logits with shape ``[B, L]``.
            They are detached before error construction, so this feature path
            cannot train the base prediction through its own residual target.
        observed_log1p_quantities: Observed ``log1p(quantity)`` values, ``[B, L]``.
        top4_prototype_indices: Prototype IDs selected at each state, ``[B, L, 4]``.
        observed_mask: Boolean mask identifying real observed states, ``[B, L]``.
        prototype_count: Total number of prototypes in the static bank.
        memory_write_mask: Optional boolean write mask, ``[B, L]``.  A transition
            is recorded only when both its endpoints are observed and writable.
        write_top4_prototype_indices: Optional ``[B, L, 4]`` source IDs used only
            for error writes.  Reads always use ``top4_prototype_indices``.  This
            permits a diagnostic sham assignment without changing retrieval.

    Returns:
        State-aligned tensors.  Prototype sums and counts have shape
        ``[B, L, prototype_count]``; the remaining values have shape ``[B, L]``.
        Masked positions and states with no prior transition are exactly zero.
    """
    if base_pre_softplus_logits.ndim != 2:
        raise ValueError("base_pre_softplus_logits must have shape [batch, sequence]")
    if observed_log1p_quantities.shape != base_pre_softplus_logits.shape:
        raise ValueError("quantity observations and base logits must share shape [batch, sequence]")
    if observed_mask.shape != base_pre_softplus_logits.shape or observed_mask.dtype != torch.bool:
        raise ValueError("observed_mask must be boolean with shape [batch, sequence]")
    batch_size, sequence_length = base_pre_softplus_logits.shape
    if batch_size == 0 or sequence_length == 0:
        raise ValueError("a nonempty batch and sequence are required")
    if top4_prototype_indices.shape != (batch_size, sequence_length, 4):
        raise ValueError("top4_prototype_indices must have shape [batch, sequence, 4]")
    if top4_prototype_indices.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("top4_prototype_indices must contain integer prototype IDs")
    if not isinstance(prototype_count, int) or isinstance(prototype_count, bool) or prototype_count <= 0:
        raise ValueError("prototype_count must be a positive integer")
    if not base_pre_softplus_logits.is_floating_point() or not observed_log1p_quantities.is_floating_point():
        raise ValueError("base logits and observed log1p quantities must be floating point")

    tensors = (observed_log1p_quantities, top4_prototype_indices, observed_mask)
    if any(tensor.device != base_pre_softplus_logits.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")
    if memory_write_mask is None:
        memory_write_mask = observed_mask
    elif memory_write_mask.shape != observed_mask.shape or memory_write_mask.dtype != torch.bool:
        raise ValueError("memory_write_mask must be boolean with shape [batch, sequence]")
    elif memory_write_mask.device != observed_mask.device:
        raise ValueError("all inputs must be on the same device")
    if bool((memory_write_mask & ~observed_mask).any()):
        raise ValueError("memory_write_mask cannot enable an unobserved position")

    if write_top4_prototype_indices is None:
        write_top4_prototype_indices = top4_prototype_indices
    elif write_top4_prototype_indices.shape != top4_prototype_indices.shape:
        raise ValueError("write_top4_prototype_indices must have shape [batch, sequence, 4]")
    elif write_top4_prototype_indices.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    ):
        raise ValueError("write_top4_prototype_indices must contain integer prototype IDs")
    elif write_top4_prototype_indices.device != observed_mask.device:
        raise ValueError("all inputs must be on the same device")

    relevant_indices = top4_prototype_indices[observed_mask]
    if relevant_indices.numel() and bool(
        ((relevant_indices < 0) | (relevant_indices >= prototype_count)).any()
    ):
        raise ValueError("observed prototype IDs must lie in [0, prototype_count)")
    if not bool(torch.isfinite(base_pre_softplus_logits[observed_mask]).all()):
        raise ValueError("observed base logits must be finite")
    observed_values = observed_log1p_quantities[observed_mask]
    if not bool(torch.isfinite(observed_values).all()) or bool((observed_values < 0).any()):
        raise ValueError("observed log1p quantities must be finite and nonnegative")

    value_dtype = torch.promote_types(
        base_pre_softplus_logits.dtype, observed_log1p_quantities.dtype
    )
    device = base_pre_softplus_logits.device
    valid_transition = torch.zeros_like(observed_mask)
    valid_transition[:, 1:] = (
        observed_mask[:, :-1]
        & observed_mask[:, 1:]
        & memory_write_mask[:, :-1]
        & memory_write_mask[:, 1:]
    )
    write_source_mask = torch.zeros_like(observed_mask)
    write_source_mask[:, :-1] = valid_transition[:, 1:]
    relevant_write_indices = write_top4_prototype_indices[write_source_mask]
    if relevant_write_indices.numel() and bool(
        ((relevant_write_indices < 0) | (relevant_write_indices >= prototype_count)).any()
    ):
        raise ValueError("used write prototype IDs must lie in [0, prototype_count)")

    transition_error = torch.zeros(
        (batch_size, sequence_length), dtype=value_dtype, device=device
    )
    if sequence_length > 1:
        valid_tail = valid_transition[:, 1:]
        previous_logits = torch.where(
            valid_tail,
            base_pre_softplus_logits[:, :-1].to(value_dtype).detach(),
            torch.zeros((), dtype=value_dtype, device=device),
        )
        current_observations = torch.where(
            valid_tail,
            observed_log1p_quantities[:, 1:].to(value_dtype),
            torch.zeros((), dtype=value_dtype, device=device),
        )
        tail_error = current_observations - F.softplus(previous_logits)
        transition_error[:, 1:] = torch.where(valid_tail, tail_error, 0.0)

    # Write e_t to the top-4 IDs selected by state t-1, then prefix-scan time.
    safe_indices = torch.where(
        observed_mask.unsqueeze(-1),
        top4_prototype_indices.to(torch.long),
        torch.zeros((), dtype=torch.long, device=device),
    )
    safe_write_indices = torch.where(
        write_source_mask.unsqueeze(-1),
        write_top4_prototype_indices.to(torch.long),
        torch.zeros((), dtype=torch.long, device=device),
    )
    write_indices = torch.zeros_like(safe_write_indices)
    write_indices[:, 1:] = safe_write_indices[:, :-1]
    sum_increment = torch.zeros(
        (batch_size, sequence_length, prototype_count), dtype=value_dtype, device=device
    )
    count_increment = torch.zeros(
        (batch_size, sequence_length, prototype_count), dtype=torch.long, device=device
    )
    sum_increment.scatter_add_(
        2, write_indices, transition_error.unsqueeze(-1).expand(-1, -1, 4)
    )
    count_increment.scatter_add_(
        2,
        write_indices,
        valid_transition.unsqueeze(-1).expand(-1, -1, 4).to(torch.long),
    )
    prototype_error_sum = sum_increment.cumsum(dim=1)
    prototype_error_count = count_increment.cumsum(dim=1)

    prototype_mean = prototype_error_sum / prototype_error_count.clamp_min(1).to(value_dtype)
    selected_prototype_mean = prototype_mean.gather(2, safe_indices)
    prototype_conditioned_top4_mean = selected_prototype_mean.mean(dim=-1)

    valid_transition_count = valid_transition.to(torch.long).cumsum(dim=1)
    prefix_error_sum = transition_error.cumsum(dim=1)
    unconditioned_prefix_mean = (
        prefix_error_sum / valid_transition_count.clamp_min(1).to(value_dtype)
    )

    # Forward-fill the most recent valid transition without a Python time loop.
    positions = torch.arange(1, sequence_length + 1, device=device).expand(batch_size, -1)
    last_position = torch.where(valid_transition, positions, 0).cummax(dim=1).values
    padded_error = torch.cat((torch.zeros_like(transition_error[:, :1]), transition_error), dim=1)
    last_error = padded_error.gather(1, last_position)

    state_mask = observed_mask
    prototype_error_sum = prototype_error_sum * state_mask.unsqueeze(-1)
    prototype_error_count = prototype_error_count * state_mask.unsqueeze(-1)
    prototype_conditioned_top4_mean = prototype_conditioned_top4_mean * state_mask
    unconditioned_prefix_mean = unconditioned_prefix_mean * state_mask
    last_error = last_error * state_mask
    valid_transition_count = valid_transition_count * state_mask

    result = {
        "prototype_error_sum": prototype_error_sum,
        "prototype_error_count": prototype_error_count,
        "prototype_conditioned_top4_mean": prototype_conditioned_top4_mean,
        "unconditioned_prefix_mean": unconditioned_prefix_mean,
        "last_error": last_error,
        "valid_transition_count": valid_transition_count,
    }
    for name, tensor in result.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"nonfinite transition-error feature: {name}")
    return result
