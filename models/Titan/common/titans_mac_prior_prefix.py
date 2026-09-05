"""Fixed-shape execution of prior-prefix reads with unchanged B1 writes.

The eager encoder remains the reference. This opt-in scan fuses a memory read
before each observed-token write; it never feeds the fused prediction back into
memory. Existing B1 write-only CUDA scans are independent of this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .titans_memory_stability import clip_associative_gradients

if TYPE_CHECKING:
    from .titans_mac import TitansMemoryState, TitansNeuralMemory


PRIOR_PREFIX_SCAN_BATCH_SIZE = 128
PRIOR_PREFIX_SCAN_CHUNK_SIZE = 16


def _scan_titans_prior_prefix_sequence(
    weight_1: torch.Tensor,
    bias_1: torch.Tensor,
    weight_2: torch.Tensor,
    bias_2: torch.Tensor,
    momentum_weight_1: torch.Tensor,
    momentum_bias_1: torch.Tensor,
    momentum_weight_2: torch.Tensor,
    momentum_bias_2: torch.Tensor,
    queries: torch.Tensor,
    segment_start_reads: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    update_rates: torch.Tensor,
    momentum_rates: torch.Tensor,
    forgetting_rates: torch.Tensor,
    write_mask: torch.Tensor,
    seen_writes: torch.Tensor,
    gradient_max_norm: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Read M_(i-1)(q_i), then apply the B1 update from the observed y_i."""
    reads: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    applied_update_rates: list[torch.Tensor] = []
    applied_momentum_rates: list[torch.Tensor] = []
    applied_forgetting_rates: list[torch.Tensor] = []
    applied_writes: list[torch.Tensor] = []
    for position in range(keys.size(1)):
        query = queries[:, position : position + 1]
        read_hidden = F.silu(
            torch.einsum("bhd,bld->blh", weight_1, query) + bias_1.unsqueeze(1)
        )
        prefix_read = (
            torch.einsum("bdh,blh->bld", weight_2, read_hidden)
            + bias_2.unsqueeze(1)
        )
        reads.append(torch.where(
            seen_writes[:, None, None], prefix_read,
            segment_start_reads[:, position : position + 1],
        ))

        key = keys[:, position]
        value = values[:, position]
        theta = update_rates[:, position]
        eta = momentum_rates[:, position]
        alpha = forgetting_rates[:, position]
        valid = write_mask[:, position]
        pre_activation = torch.einsum("bhd,bd->bh", weight_1, key) + bias_1
        hidden = F.silu(pre_activation)
        prediction = torch.einsum("bdh,bh->bd", weight_2, hidden) + bias_2
        error = prediction - value
        output_gradient = 2.0 * error
        grad_weight_2 = output_gradient.unsqueeze(-1) * hidden.unsqueeze(1)
        grad_bias_2 = output_gradient
        hidden_gradient = torch.einsum("bdh,bd->bh", weight_2, output_gradient)
        sigmoid = torch.sigmoid(pre_activation)
        pre_gradient = hidden_gradient * (
            sigmoid * (1.0 + pre_activation * (1.0 - sigmoid))
        )
        grad_weight_1 = pre_gradient.unsqueeze(-1) * key.unsqueeze(1)
        grad_bias_1 = pre_gradient
        grad_weight_1, grad_bias_1, grad_weight_2, grad_bias_2 = (
            clip_associative_gradients(
                (grad_weight_1, grad_bias_1, grad_weight_2, grad_bias_2),
                gradient_max_norm,
            )
        )

        def update_tensor(
            parameter: torch.Tensor,
            momentum: torch.Tensor,
            gradient: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            dimensions = [1] * (parameter.ndim - 1)
            row_theta = theta.view(parameter.size(0), *dimensions)
            row_eta = eta.view(parameter.size(0), *dimensions)
            row_alpha = alpha.view(parameter.size(0), *dimensions)
            row_valid = valid.view(parameter.size(0), *dimensions)
            next_momentum = row_eta * momentum - row_theta * gradient
            next_parameter = (1.0 - row_alpha) * parameter + next_momentum
            return (
                torch.where(row_valid, next_parameter, parameter),
                torch.where(row_valid, next_momentum, momentum),
            )

        weight_1, momentum_weight_1 = update_tensor(
            weight_1, momentum_weight_1, grad_weight_1,
        )
        bias_1, momentum_bias_1 = update_tensor(
            bias_1, momentum_bias_1, grad_bias_1,
        )
        weight_2, momentum_weight_2 = update_tensor(
            weight_2, momentum_weight_2, grad_weight_2,
        )
        bias_2, momentum_bias_2 = update_tensor(
            bias_2, momentum_bias_2, grad_bias_2,
        )
        seen_writes = seen_writes | valid
        valid_values = valid.to(dtype=keys.dtype)
        losses.append(torch.square(error).sum(dim=-1) * valid_values)
        applied_update_rates.append(theta * valid_values)
        applied_momentum_rates.append(eta * valid_values)
        applied_forgetting_rates.append(alpha * valid_values)
        applied_writes.append(valid_values)
    return (
        weight_1, bias_1, weight_2, bias_2,
        momentum_weight_1, momentum_bias_1, momentum_weight_2, momentum_bias_2,
        torch.cat(reads, dim=1), seen_writes,
        torch.stack(losses, dim=1),
        torch.stack(applied_update_rates, dim=1),
        torch.stack(applied_momentum_rates, dim=1),
        torch.stack(applied_forgetting_rates, dim=1),
        torch.stack(applied_writes, dim=1),
    )


_COMPILED_PRIOR_PREFIX_SCAN = (
    torch.compile(
        _scan_titans_prior_prefix_sequence,
        fullgraph=True, dynamic=False, mode="default",
    )
    if hasattr(torch, "compile") else None
)


def _pad_dimension(tensor: torch.Tensor, dimension: int, size: int) -> torch.Tensor:
    missing = size - tensor.size(dimension)
    if missing < 0:
        raise ValueError("Prior-prefix input exceeds the fixed scan shape")
    if missing == 0:
        return tensor
    padding_shape = list(tensor.shape)
    padding_shape[dimension] = missing
    return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=dimension)


def read_prior_prefix_with_fixed_shape_scan(
    memory: TitansNeuralMemory,
    state: TitansMemoryState,
    attention_output: torch.Tensor,
    output_queries: torch.Tensor,
    segment_start_reads: torch.Tensor,
    write_mask: torch.Tensor,
    *,
    chunk_size: int | None = None,
    compiled: bool = True,
) -> tuple[torch.Tensor, TitansMemoryState, dict[str, torch.Tensor]]:
    """Use a 128-row/16-token scan; compiled=False audits it on CPU.

    Short batches and chunks are zero-padded with disabled writes. Padding is
    removed from returned states, reads, and diagnostics. No state is detached,
    and compilation errors propagate instead of silently changing the backend.
    """
    from .titans_mac import TitansMemoryState

    batch_size, length = write_mask.shape
    if not 1 <= batch_size <= PRIOR_PREFIX_SCAN_BATCH_SIZE:
        raise ValueError("Prior-prefix fixed scan requires 1..128 batch rows")
    if attention_output.shape[:2] != (batch_size, length):
        raise ValueError("Prior-prefix write mask shape must match the input")
    if output_queries.shape != attention_output.shape or segment_start_reads.shape != attention_output.shape:
        raise ValueError("Prior-prefix queries and reads must match the input shape")
    step = PRIOR_PREFIX_SCAN_CHUNK_SIZE if chunk_size is None else int(chunk_size)
    if step < 1:
        raise ValueError("chunk_size must be positive")
    step = min(step, PRIOR_PREFIX_SCAN_CHUNK_SIZE)
    if compiled:
        if attention_output.device.type != "cuda":
            raise ValueError("Compiled prior-prefix scan requires CUDA inputs")
        if _COMPILED_PRIOR_PREFIX_SCAN is None:
            raise RuntimeError("Compiled prior-prefix scan is unavailable")
        scan = _COMPILED_PRIOR_PREFIX_SCAN
    else:
        scan = _scan_titans_prior_prefix_sequence
    diagnostic_names = (
        "associative_loss", "update_rate", "momentum_rate", "forgetting_rate",
        "write_applied",
    )
    if length == 0:
        return segment_start_reads, state, {
            name: attention_output.new_zeros(batch_size, 0) for name in diagnostic_names
        }
    keys, values, theta, eta, alpha = memory._project_write(attention_output)
    projected = (
        output_queries, segment_start_reads, keys, values,
        theta.squeeze(-1), eta.squeeze(-1), alpha.squeeze(-1), write_mask,
    )
    seen_writes = torch.zeros(batch_size, device=attention_output.device, dtype=torch.bool)
    collected_reads: list[torch.Tensor] = []
    collected_diagnostics: dict[str, list[torch.Tensor]] = {
        name: [] for name in diagnostic_names
    }
    for start in range(0, length, step):
        end = min(start + step, length)
        state_tensors = tuple(
            _pad_dimension(tensor, 0, PRIOR_PREFIX_SCAN_BATCH_SIZE)
            for tensor in (*state.memory_tensors(), *state.momentum_tensors())
        )
        chunk_inputs = tuple(
            _pad_dimension(
                _pad_dimension(tensor[:, start:end], 1, PRIOR_PREFIX_SCAN_CHUNK_SIZE),
                0, PRIOR_PREFIX_SCAN_BATCH_SIZE,
            )
            for tensor in projected
        )
        scanned = scan(
            *state_tensors, *chunk_inputs,
            _pad_dimension(seen_writes, 0, PRIOR_PREFIX_SCAN_BATCH_SIZE),
            memory.gradient_max_norm,
        )
        state = TitansMemoryState(
            *(tensor[:batch_size] for tensor in scanned[:8]),
            positions=state.positions, series_ids=state.series_ids,
        )
        collected_reads.append(scanned[8][:batch_size, : end - start])
        seen_writes = scanned[9][:batch_size]
        for name, value in zip(diagnostic_names, scanned[10:], strict=True):
            collected_diagnostics[name].append(value[:batch_size, : end - start])
    return torch.cat(collected_reads, dim=1), state, {
        name: torch.cat(values, dim=1) for name, values in collected_diagnostics.items()
    }


__all__ = [
    "PRIOR_PREFIX_SCAN_BATCH_SIZE", "PRIOR_PREFIX_SCAN_CHUNK_SIZE",
    "read_prior_prefix_with_fixed_shape_scan",
]
