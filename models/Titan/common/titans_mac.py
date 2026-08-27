"""Faithful Titans neural memory and Memory-as-Context event encoder.

The neural memory follows equations 11--15 of Behrouz et al. (2025). The
event-domain MAC wrapper keeps a stricter prediction-before-write order: a
segment reads its start state, builds causal prediction states, and only then
writes valid observed events for later segments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TitansMemoryState:
    """Batch-local neural-memory parameters and surprise momentum."""

    weight_1: torch.Tensor
    bias_1: torch.Tensor
    weight_2: torch.Tensor
    bias_2: torch.Tensor
    momentum_weight_1: torch.Tensor
    momentum_bias_1: torch.Tensor
    momentum_weight_2: torch.Tensor
    momentum_bias_2: torch.Tensor
    positions: torch.Tensor
    series_ids: Optional[torch.Tensor] = None

    def memory_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.weight_1, self.bias_1, self.weight_2, self.bias_2)

    def momentum_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.momentum_weight_1,
            self.momentum_bias_1,
            self.momentum_weight_2,
            self.momentum_bias_2,
        )

    def detach(self) -> "TitansMemoryState":
        """Detach online state without changing its values or series scope."""
        return TitansMemoryState(
            *(tensor.detach() for tensor in self.memory_tensors()),
            *(tensor.detach() for tensor in self.momentum_tensors()),
            positions=self.positions.detach(),
            series_ids=(
                None if self.series_ids is None else self.series_ids.detach()
            ),
        )


def _scan_titans_write_sequence(
    weight_1: torch.Tensor,
    bias_1: torch.Tensor,
    weight_2: torch.Tensor,
    bias_2: torch.Tensor,
    momentum_weight_1: torch.Tensor,
    momentum_bias_1: torch.Tensor,
    momentum_weight_2: torch.Tensor,
    momentum_bias_2: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    update_rates: torch.Tensor,
    momentum_rates: torch.Tensor,
    forgetting_rates: torch.Tensor,
    write_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Run the exact Titans write recurrence as one compilable CUDA graph."""
    losses: list[torch.Tensor] = []
    applied_update_rates: list[torch.Tensor] = []
    applied_momentum_rates: list[torch.Tensor] = []
    applied_forgetting_rates: list[torch.Tensor] = []
    applied_writes: list[torch.Tensor] = []

    for position in range(keys.size(1)):
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
            weight_1,
            momentum_weight_1,
            grad_weight_1,
        )
        bias_1, momentum_bias_1 = update_tensor(
            bias_1,
            momentum_bias_1,
            grad_bias_1,
        )
        weight_2, momentum_weight_2 = update_tensor(
            weight_2,
            momentum_weight_2,
            grad_weight_2,
        )
        bias_2, momentum_bias_2 = update_tensor(
            bias_2,
            momentum_bias_2,
            grad_bias_2,
        )
        valid_values = valid.to(dtype=keys.dtype)
        losses.append(torch.square(error).sum(dim=-1) * valid_values)
        applied_update_rates.append(theta * valid_values)
        applied_momentum_rates.append(eta * valid_values)
        applied_forgetting_rates.append(alpha * valid_values)
        applied_writes.append(valid_values)

    return (
        weight_1,
        bias_1,
        weight_2,
        bias_2,
        momentum_weight_1,
        momentum_bias_1,
        momentum_weight_2,
        momentum_bias_2,
        torch.stack(losses, dim=1),
        torch.stack(applied_update_rates, dim=1),
        torch.stack(applied_momentum_rates, dim=1),
        torch.stack(applied_forgetting_rates, dim=1),
        torch.stack(applied_writes, dim=1),
    )


_COMPILED_TITANS_WRITE_SEQUENCE = (
    torch.compile(
        _scan_titans_write_sequence,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead",
    )
    if hasattr(torch, "compile")
    else None
)


class TitansNeuralMemory(nn.Module):
    """Two-layer associative memory updated by surprise, momentum, and decay."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_expansion: int = 2,
        initial_update_rate: float = 0.01,
        initial_momentum: float = 0.9,
        initial_forgetting: float = 0.001,
        compile_cuda_scan: bool = True,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if hidden_expansion < 1:
            raise ValueError("hidden_expansion must be positive")
        for name, value in (
            ("initial_update_rate", initial_update_rate),
            ("initial_momentum", initial_momentum),
            ("initial_forgetting", initial_forgetting),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")

        self.d_model = int(d_model)
        self.memory_hidden_dim = int(d_model * hidden_expansion)
        self.compile_cuda_scan = bool(compile_cuda_scan)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.query_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.update_rate_projection = nn.Linear(self.d_model, 1)
        self.momentum_projection = nn.Linear(self.d_model, 1)
        self.forgetting_projection = nn.Linear(self.d_model, 1)

        self.initial_weight_1 = nn.Parameter(
            torch.empty(self.memory_hidden_dim, self.d_model)
        )
        self.initial_bias_1 = nn.Parameter(torch.zeros(self.memory_hidden_dim))
        self.initial_weight_2 = nn.Parameter(
            torch.empty(self.d_model, self.memory_hidden_dim)
        )
        self.initial_bias_2 = nn.Parameter(torch.zeros(self.d_model))
        nn.init.xavier_uniform_(self.initial_weight_1)
        nn.init.xavier_uniform_(self.initial_weight_2)

        self._initialize_rate(
            self.update_rate_projection,
            initial_update_rate,
        )
        self._initialize_rate(self.momentum_projection, initial_momentum)
        self._initialize_rate(
            self.forgetting_projection,
            initial_forgetting,
        )

    @staticmethod
    def _logit(probability: float) -> float:
        return math.log(probability) - math.log1p(-probability)

    @classmethod
    def _initialize_rate(cls, layer: nn.Linear, probability: float) -> None:
        nn.init.zeros_(layer.weight)
        nn.init.constant_(layer.bias, cls._logit(probability))

    @staticmethod
    def _expand_parameter(parameter: torch.Tensor, batch_size: int) -> torch.Tensor:
        return parameter.unsqueeze(0).expand(batch_size, *parameter.shape).clone()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        series_ids: Optional[torch.Tensor] = None,
    ) -> TitansMemoryState:
        """Create independent online memory for every batch row."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameters = tuple(
            self._expand_parameter(parameter.to(device=device, dtype=dtype), batch_size)
            for parameter in (
                self.initial_weight_1,
                self.initial_bias_1,
                self.initial_weight_2,
                self.initial_bias_2,
            )
        )
        momenta = tuple(torch.zeros_like(parameter) for parameter in parameters)
        normalized_ids = self._normalize_series_ids(
            series_ids,
            batch_size=batch_size,
            device=device,
        )
        return TitansMemoryState(
            *parameters,
            *momenta,
            positions=torch.zeros(batch_size, device=device, dtype=torch.long),
            series_ids=normalized_ids,
        )

    @staticmethod
    def _normalize_series_ids(
        series_ids: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if series_ids is None:
            return None
        if series_ids.shape != (batch_size,):
            raise ValueError(
                f"series_ids must have shape {(batch_size,)}, got "
                f"{tuple(series_ids.shape)}"
            )
        return series_ids.to(device=device, dtype=torch.long)

    @staticmethod
    def _state_batch_size(state: TitansMemoryState) -> int:
        return int(state.weight_1.size(0))

    def prepare_state(
        self,
        state: Optional[TitansMemoryState],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        series_ids: Optional[torch.Tensor],
    ) -> TitansMemoryState:
        """Validate explicit state and reset rows whose series ID changed."""
        normalized_ids = self._normalize_series_ids(
            series_ids,
            batch_size=batch_size,
            device=device,
        )
        if state is None:
            return self.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
                series_ids=normalized_ids,
            )
        if normalized_ids is None:
            raise ValueError("series_ids are required when reusing Titans memory state")
        if self._state_batch_size(state) != batch_size:
            raise ValueError("Memory state batch size does not match the input batch")
        if state.series_ids is None:
            raise ValueError("Reusable Titans memory state must carry series_ids")
        for tensor in (*state.memory_tensors(), *state.momentum_tensors()):
            if tensor.device != device or tensor.dtype != dtype:
                raise ValueError("Memory state device and dtype must match the input")

        changed = state.series_ids.to(device=device) != normalized_ids
        if not bool(changed.any()):
            return replace(state, series_ids=normalized_ids)

        fresh = self.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
            series_ids=normalized_ids,
        )

        def reset_rows(current: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
            row_mask = changed.view(batch_size, *([1] * (current.ndim - 1)))
            return torch.where(row_mask, initial, current)

        memory = tuple(
            reset_rows(current, initial)
            for current, initial in zip(
                state.memory_tensors(),
                fresh.memory_tensors(),
                strict=True,
            )
        )
        momenta = tuple(
            reset_rows(current, initial)
            for current, initial in zip(
                state.momentum_tensors(),
                fresh.momentum_tensors(),
                strict=True,
            )
        )
        positions = torch.where(changed, fresh.positions, state.positions)
        return TitansMemoryState(
            *memory,
            *momenta,
            positions=positions,
            series_ids=normalized_ids,
        )

    def project_query(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(inputs)
        return F.normalize(
            F.silu(self.query_projection(normalized)),
            dim=-1,
            eps=1e-6,
        )

    def _project_write(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.input_norm(inputs)
        keys = F.normalize(
            F.silu(self.key_projection(normalized)),
            dim=-1,
            eps=1e-6,
        )
        values = F.silu(self.value_projection(normalized))
        theta = torch.sigmoid(self.update_rate_projection(normalized))
        eta = torch.sigmoid(self.momentum_projection(normalized))
        alpha = torch.sigmoid(self.forgetting_projection(normalized))
        return keys, values, theta, eta, alpha

    @staticmethod
    def _memory_forward(
        state: TitansMemoryState,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_activation = torch.einsum(
            "bhd,bd->bh",
            state.weight_1,
            inputs,
        ) + state.bias_1
        hidden = F.silu(pre_activation)
        output = torch.einsum(
            "bdh,bh->bd",
            state.weight_2,
            hidden,
        ) + state.bias_2
        return output, pre_activation, hidden

    def read(
        self,
        state: TitansMemoryState,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve without changing neural-memory parameters."""
        if queries.ndim == 2:
            output, _, _ = self._memory_forward(state, queries)
            return output
        if queries.ndim != 3:
            raise ValueError("queries must have shape [B, D] or [B, L, D]")
        pre_activation = torch.einsum(
            "bhd,bld->blh",
            state.weight_1,
            queries,
        ) + state.bias_1.unsqueeze(1)
        hidden = F.silu(pre_activation)
        return torch.einsum(
            "bdh,blh->bld",
            state.weight_2,
            hidden,
        ) + state.bias_2.unsqueeze(1)

    @staticmethod
    def _silu_derivative(inputs: torch.Tensor) -> torch.Tensor:
        sigmoid = torch.sigmoid(inputs)
        return sigmoid * (1.0 + inputs * (1.0 - sigmoid))

    def associative_gradients(
        self,
        state: TitansMemoryState,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return exact gradients of ||M(k)-v||^2 for each batch row."""
        predicted, pre_activation, hidden = self._memory_forward(state, keys)
        error = predicted - values
        output_gradient = 2.0 * error
        grad_weight_2 = output_gradient.unsqueeze(-1) * hidden.unsqueeze(1)
        grad_bias_2 = output_gradient
        hidden_gradient = torch.einsum(
            "bdh,bd->bh",
            state.weight_2,
            output_gradient,
        )
        pre_gradient = hidden_gradient * self._silu_derivative(pre_activation)
        grad_weight_1 = pre_gradient.unsqueeze(-1) * keys.unsqueeze(1)
        grad_bias_1 = pre_gradient
        loss = torch.square(error).sum(dim=-1)
        return (
            grad_weight_1,
            grad_bias_1,
            grad_weight_2,
            grad_bias_2,
        ), loss

    @staticmethod
    def _apply_update(
        parameter: torch.Tensor,
        momentum: torch.Tensor,
        gradient: torch.Tensor,
        *,
        theta: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dimensions = [1] * (parameter.ndim - 1)
        theta = theta.view(parameter.size(0), *dimensions)
        eta = eta.view(parameter.size(0), *dimensions)
        alpha = alpha.view(parameter.size(0), *dimensions)
        row_mask = valid.view(parameter.size(0), *dimensions)
        next_momentum = eta * momentum - theta * gradient
        next_parameter = (1.0 - alpha) * parameter + next_momentum
        return (
            torch.where(row_mask, next_parameter, parameter),
            torch.where(row_mask, next_momentum, momentum),
        )

    def write_token(
        self,
        state: TitansMemoryState,
        inputs: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[TitansMemoryState, dict[str, torch.Tensor]]:
        """Apply one observed-token update using the original Titans equations."""
        if inputs.shape != (state.weight_1.size(0), self.d_model):
            raise ValueError("inputs must have shape [batch, d_model]")
        valid = valid.to(device=inputs.device, dtype=torch.bool)
        keys, values, theta, eta, alpha = self._project_write(inputs)
        gradients, loss = self.associative_gradients(state, keys, values)
        updated = tuple(
            self._apply_update(
                parameter,
                momentum,
                gradient,
                theta=theta.squeeze(-1),
                eta=eta.squeeze(-1),
                alpha=alpha.squeeze(-1),
                valid=valid,
            )
            for parameter, momentum, gradient in zip(
                state.memory_tensors(),
                state.momentum_tensors(),
                gradients,
                strict=True,
            )
        )
        parameters = tuple(item[0] for item in updated)
        momenta = tuple(item[1] for item in updated)
        valid_values = valid.to(dtype=inputs.dtype)
        next_state = TitansMemoryState(
            *parameters,
            *momenta,
            positions=state.positions,
            series_ids=state.series_ids,
        )
        diagnostics = {
            "associative_loss": loss * valid_values,
            "update_rate": theta.squeeze(-1) * valid_values,
            "momentum_rate": eta.squeeze(-1) * valid_values,
            "forgetting_rate": alpha.squeeze(-1) * valid_values,
            "write_applied": valid_values,
        }
        return next_state, diagnostics

    def write_sequence(
        self,
        state: TitansMemoryState,
        inputs: torch.Tensor,
        write_mask: torch.Tensor,
        *,
        chunk_size: Optional[int] = None,
    ) -> tuple[TitansMemoryState, dict[str, torch.Tensor]]:
        """Scan writes exactly; chunking changes scheduling, not numerics."""
        if inputs.ndim != 3 or inputs.size(-1) != self.d_model:
            raise ValueError("inputs must have shape [batch, sequence, d_model]")
        if write_mask.shape != inputs.shape[:2]:
            raise ValueError("write_mask shape must match the input sequence")
        step = inputs.size(1) if chunk_size is None else int(chunk_size)
        if step < 1:
            raise ValueError("chunk_size must be positive")
        use_compiled_scan = (
            inputs.size(1) > 0
            and inputs.device.type == "cuda"
            and self.compile_cuda_scan
            and _COMPILED_TITANS_WRITE_SEQUENCE is not None
        )
        if use_compiled_scan:
            keys, values, theta, eta, alpha = self._project_write(inputs)
            # CUDAGraph reuses output buffers across calls; state must outlive them.
            scanned = tuple(
                tensor.clone()
                for tensor in _COMPILED_TITANS_WRITE_SEQUENCE(
                    *state.memory_tensors(),
                    *state.momentum_tensors(),
                    keys,
                    values,
                    theta.squeeze(-1),
                    eta.squeeze(-1),
                    alpha.squeeze(-1),
                    write_mask,
                )
            )
            next_state = TitansMemoryState(
                *scanned[:8],
                positions=state.positions,
                series_ids=state.series_ids,
            )
            return next_state, {
                "associative_loss": scanned[8],
                "update_rate": scanned[9],
                "momentum_rate": scanned[10],
                "forgetting_rate": scanned[11],
                "write_applied": scanned[12],
            }
        collected: dict[str, list[torch.Tensor]] = {
            "associative_loss": [],
            "update_rate": [],
            "momentum_rate": [],
            "forgetting_rate": [],
            "write_applied": [],
        }
        for chunk_start in range(0, inputs.size(1), step):
            chunk_end = min(chunk_start + step, inputs.size(1))
            for position in range(chunk_start, chunk_end):
                state, diagnostics = self.write_token(
                    state,
                    inputs[:, position],
                    write_mask[:, position],
                )
                for name, value in diagnostics.items():
                    collected[name].append(value)
        if inputs.size(1) == 0:
            empty = inputs.new_zeros(inputs.size(0), 0)
            return state, {name: empty for name in collected}
        return state, {
            name: torch.stack(values, dim=1)
            for name, values in collected.items()
        }


class TitansMACBlock(nn.Module):
    """Pre-norm attention block over persistent, retrieved, and local tokens."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.feedforward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        blocked_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=blocked_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        tokens = tokens * valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        fed = self.feedforward(self.feedforward_norm(tokens))
        tokens = tokens + self.feedforward_dropout(fed)
        return tokens * valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)


class TitansMACEncoder(nn.Module):
    """Count-aware event encoder using faithful neural memory in MAC form."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        persistent_memory_size: int = 16,
        segment_size: int = 16,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if segment_size < 1:
            raise ValueError("segment_size must be positive")
        if persistent_memory_size < 1:
            raise ValueError("persistent_memory_size must be positive")
        if max_len < 1:
            raise ValueError("max_len must be positive")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.segment_size = int(segment_size)
        self.persistent_memory_size = int(persistent_memory_size)
        self.max_len = int(max_len)
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.max_len, self.d_model) * 0.02
        )
        self.persistent_memory = nn.Parameter(
            torch.randn(1, self.persistent_memory_size, self.d_model) * 0.02
        )
        self.neural_memory = TitansNeuralMemory(self.d_model)
        self.layers = nn.ModuleList(
            TitansMACBlock(
                self.d_model,
                n_heads,
                self.d_model * 2 if d_ff is None else int(d_ff),
                dropout,
            )
            for _ in range(int(n_layers))
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.memory_output_projection = nn.Linear(
            self.d_model,
            self.d_model,
            bias=False,
        )
        self.output_gate = nn.Linear(self.d_model, self.d_model)

    @staticmethod
    def _validate_mask(
        mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        if mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
            )
        return mask.to(device=device, dtype=torch.bool)

    def _position_values(
        self,
        positions: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        offsets = torch.arange(seq_len, device=positions.device)
        indices = positions.unsqueeze(1) + offsets.unsqueeze(0)
        if bool((indices >= self.max_len).any()):
            raise ValueError(
                "Titans MAC position exceeds max_len; reset at the series boundary "
                "or construct the encoder with a larger max_len"
            )
        table = self.position_embedding[0]
        return table[indices]

    @staticmethod
    def _mac_attention_mask(
        persistent_size: int,
        segment_length: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        total = persistent_size + 2 * segment_length
        allowed = torch.zeros(total, total, device=device, dtype=torch.bool)
        allowed[:persistent_size, :persistent_size] = True
        for position in range(segment_length):
            memory_query = persistent_size + position
            event_query = persistent_size + segment_length + position
            allowed[memory_query, :persistent_size] = True
            allowed[
                memory_query,
                persistent_size : persistent_size + position + 1,
            ] = True
            allowed[event_query, :persistent_size] = True
            allowed[
                event_query,
                persistent_size : persistent_size + position + 1,
            ] = True
            allowed[
                event_query,
                persistent_size + segment_length : event_query + 1,
            ] = True
        return ~allowed

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        series_ids: Optional[torch.Tensor] = None,
    ) -> TitansMemoryState:
        return self.neural_memory.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
            series_ids=series_ids,
        )

    def forward_with_state(
        self,
        inputs: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        state: Optional[TitansMemoryState] = None,
        series_ids: Optional[torch.Tensor] = None,
        segment_size: Optional[int] = None,
        write_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, TitansMemoryState, dict[str, torch.Tensor]]:
        """Encode one batch and return explicit online state and diagnostics."""
        if inputs.ndim != 3 or inputs.size(-1) != self.input_dim:
            raise ValueError("inputs must have shape [batch, sequence, input_dim]")
        batch_size, seq_len, _ = inputs.shape
        mask = self._validate_mask(
            mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=inputs.device,
        )
        if write_mask is None:
            write_mask = mask
        else:
            write_mask = self._validate_mask(
                write_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                device=inputs.device,
            ) & mask
        state = self.neural_memory.prepare_state(
            state,
            batch_size=batch_size,
            device=inputs.device,
            dtype=inputs.dtype,
            series_ids=series_ids,
        )
        current_segment_size = (
            self.segment_size if segment_size is None else int(segment_size)
        )
        if current_segment_size < 1:
            raise ValueError("segment_size must be positive")

        projected = self.input_projection(inputs)
        projected = projected + self._position_values(state.positions, seq_len)
        projected = projected * mask.unsqueeze(-1).to(dtype=projected.dtype)
        outputs: list[torch.Tensor] = []
        diagnostics: dict[str, list[torch.Tensor]] = {
            "associative_loss": [],
            "update_rate": [],
            "momentum_rate": [],
            "forgetting_rate": [],
            "write_applied": [],
        }

        for start in range(0, seq_len, current_segment_size):
            end = min(start + current_segment_size, seq_len)
            current = projected[:, start:end]
            current_mask = mask[:, start:end]
            current_write_mask = write_mask[:, start:end]
            query = self.neural_memory.project_query(current)
            retrieved = self.neural_memory.read(state, query)
            persistent = self.persistent_memory.to(
                device=inputs.device,
                dtype=inputs.dtype,
            ).expand(batch_size, -1, -1)
            mac_tokens = torch.cat((persistent, retrieved, current), dim=1)
            prefix_valid = torch.ones(
                batch_size,
                self.persistent_memory_size,
                device=inputs.device,
                dtype=torch.bool,
            )
            valid_tokens = torch.cat(
                (prefix_valid, current_mask, current_mask),
                dim=1,
            )
            blocked_mask = self._mac_attention_mask(
                self.persistent_memory_size,
                end - start,
                device=inputs.device,
            )
            for layer in self.layers:
                mac_tokens = layer(
                    mac_tokens,
                    blocked_mask=blocked_mask,
                    key_padding_mask=~valid_tokens,
                    valid_mask=valid_tokens,
                )
            event_start = self.persistent_memory_size + (end - start)
            attention_output = mac_tokens[:, event_start:]

            # The prediction state reads the segment-start memory. Writes occur
            # only after these states are complete, so the target cannot leak.
            output_query = self.neural_memory.project_query(attention_output)
            output_memory = self.neural_memory.read(state, output_query)
            normalized_output = self.output_norm(attention_output)
            gated_memory = (
                torch.sigmoid(self.output_gate(normalized_output))
                * self.memory_output_projection(output_memory)
            )
            prediction_state = attention_output + gated_memory
            prediction_state = prediction_state * current_mask.unsqueeze(-1).to(
                dtype=prediction_state.dtype
            )
            outputs.append(prediction_state)

            state, write_diagnostics = self.neural_memory.write_sequence(
                state,
                attention_output,
                current_write_mask,
                chunk_size=write_chunk_size,
            )
            for name, value in write_diagnostics.items():
                diagnostics[name].append(value)

        encoded = torch.cat(outputs, dim=1) if outputs else projected
        next_positions = state.positions + mask.sum(dim=1)
        state = replace(state, positions=next_positions)
        combined_diagnostics = {
            name: (
                torch.cat(values, dim=1)
                if values
                else inputs.new_zeros(batch_size, 0)
            )
            for name, values in diagnostics.items()
        }
        return encoded, state, combined_diagnostics

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded, _, _ = self.forward_with_state(
            inputs,
            mask=mask,
            write_mask=write_mask,
        )
        return encoded


__all__ = [
    "TitansMACBlock",
    "TitansMACEncoder",
    "TitansMemoryState",
    "TitansNeuralMemory",
]
