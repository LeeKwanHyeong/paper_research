"""TPP-specific causal sparse memory with explicit online state.

This module is a project-specific event-memory design, not an implementation of
the Titans neural long-term memory. Each event reads previously observed slots,
builds its prediction state, and is written only afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TPPGatedMemoryState:
    """Batch-local observed-event memory used by the B2 backbone."""

    keys: torch.Tensor
    values: torch.Tensor
    valid_slots: torch.Tensor
    write_counts: torch.Tensor
    positions: torch.Tensor
    series_ids: Optional[torch.Tensor] = None

    def memory_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.keys, self.values)

    def detach(self) -> "TPPGatedMemoryState":
        """Detach online state while retaining slot and series boundaries."""
        return TPPGatedMemoryState(
            keys=self.keys.detach(),
            values=self.values.detach(),
            valid_slots=self.valid_slots.detach(),
            write_counts=self.write_counts.detach(),
            positions=self.positions.detach(),
            series_ids=(
                None if self.series_ids is None else self.series_ids.detach()
            ),
        )


def _scan_tpp_gated_sequence(
    state_keys: torch.Tensor,
    state_values: torch.Tensor,
    state_valid_slots: torch.Tensor,
    state_write_counts: torch.Tensor,
    encoded: torch.Tensor,
    normalized: torch.Tensor,
    queries: torch.Tensor,
    write_keys: torch.Tensor,
    write_values: torch.Tensor,
    null_logits: torch.Tensor,
    confidence_weight: torch.Tensor,
    confidence_bias: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_norm_bias: torch.Tensor,
    output_projection_weight: torch.Tensor,
    mask: torch.Tensor,
    write_mask: torch.Tensor,
    topk: int,
    temperature: float,
    dropout_probability: float,
    training: bool,
) -> tuple[torch.Tensor, ...]:
    """Run exact sparse read-before-write recurrence as one CUDA graph."""
    outputs: list[torch.Tensor] = []
    null_probabilities: list[torch.Tensor] = []
    retrieval_confidences: list[torch.Tensor] = []
    learned_confidences: list[torch.Tensor] = []
    effective_gates: list[torch.Tensor] = []
    selected_slot_counts: list[torch.Tensor] = []
    applied_writes: list[torch.Tensor] = []
    memory_size = state_keys.size(1)
    d_model = state_keys.size(2)

    for position in range(encoded.size(1)):
        scores = torch.einsum(
            "bd,bmd->bm",
            queries[:, position],
            state_keys,
        ) / temperature
        scores = scores.masked_fill(~state_valid_slots, float("-inf"))
        selected_scores, selected_indices = torch.topk(scores, topk, dim=-1)
        selected_valid = torch.gather(
            state_valid_slots,
            1,
            selected_indices,
        )
        selected_scores = selected_scores.masked_fill(
            ~selected_valid,
            float("-inf"),
        )
        candidate_logits = torch.cat(
            (selected_scores, null_logits[:, position]),
            dim=-1,
        )
        candidate_weights = torch.softmax(candidate_logits, dim=-1)
        selected_weights = candidate_weights[:, :-1]
        null_probability = candidate_weights[:, -1]
        gather_indices = selected_indices.unsqueeze(-1).expand(-1, -1, d_model)
        selected_values = torch.gather(state_values, 1, gather_indices)
        retrieved = torch.sum(
            selected_weights.unsqueeze(-1) * selected_values,
            dim=1,
        )
        retrieval_confidence = selected_weights.sum(dim=-1)
        confidence_input = torch.cat(
            (normalized[:, position], retrieved),
            dim=-1,
        )
        learned_confidence = torch.sigmoid(
            F.linear(confidence_input, confidence_weight, confidence_bias)
        ).squeeze(-1)
        effective_gate = retrieval_confidence * learned_confidence
        normalized_retrieved = F.layer_norm(
            retrieved,
            (d_model,),
            output_norm_weight,
            output_norm_bias,
        )
        projected_retrieved = F.linear(
            normalized_retrieved,
            output_projection_weight,
        )
        residual = effective_gate.unsqueeze(-1) * F.dropout(
            projected_retrieved,
            p=dropout_probability,
            training=training,
        )
        valid = mask[:, position]
        valid_values = valid.to(dtype=encoded.dtype)
        outputs.append(
            (encoded[:, position] + residual) * valid_values.unsqueeze(-1)
        )
        null_probabilities.append(null_probability * valid_values)
        retrieval_confidences.append(retrieval_confidence * valid_values)
        learned_confidences.append(learned_confidence * valid_values)
        effective_gates.append(effective_gate * valid_values)
        selected_slot_counts.append(
            selected_valid.sum(dim=-1).to(encoded.dtype) * valid_values
        )

        write_valid = write_mask[:, position]
        applied_writes.append(write_valid.to(encoded.dtype))
        slot_indices = torch.remainder(state_write_counts, memory_size)
        slot_mask = F.one_hot(slot_indices, num_classes=memory_size).bool()
        write_slots = slot_mask & write_valid.unsqueeze(-1)
        state_keys = torch.where(
            write_slots.unsqueeze(-1),
            write_keys[:, position].unsqueeze(1),
            state_keys,
        )
        state_values = torch.where(
            write_slots.unsqueeze(-1),
            write_values[:, position].unsqueeze(1),
            state_values,
        )
        state_valid_slots = state_valid_slots | write_slots
        state_write_counts = state_write_counts + write_valid.to(torch.long)

    return (
        torch.stack(outputs, dim=1),
        state_keys,
        state_values,
        state_valid_slots,
        state_write_counts,
        torch.stack(null_probabilities, dim=1),
        torch.stack(retrieval_confidences, dim=1),
        torch.stack(learned_confidences, dim=1),
        torch.stack(effective_gates, dim=1),
        torch.stack(selected_slot_counts, dim=1),
        torch.stack(applied_writes, dim=1),
    )


_COMPILED_TPP_GATED_SEQUENCE = (
    torch.compile(
        _scan_tpp_gated_sequence,
        fullgraph=True,
        dynamic=False,
        mode="reduce-overhead",
    )
    if hasattr(torch, "compile")
    else None
)


class TPPSpecificGatedMemory(nn.Module):
    """Sparse observed-event retrieval with null selection and confidence gating."""

    def __init__(
        self,
        d_model: int,
        *,
        memory_size: int = 64,
        topk: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
        initial_null_logit: float = 0.0,
        initial_confidence: float = 0.5,
        compile_cuda_scan: bool = True,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if memory_size < 1:
            raise ValueError("memory_size must be positive")
        if topk < 1:
            raise ValueError("topk must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < initial_confidence < 1.0:
            raise ValueError("initial_confidence must lie strictly between zero and one")

        self.d_model = int(d_model)
        self.memory_size = int(memory_size)
        self.topk = min(int(topk), self.memory_size)
        self.temperature = float(temperature)
        self.compile_cuda_scan = bool(compile_cuda_scan)

        self.input_norm = nn.LayerNorm(self.d_model)
        self.query_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.null_logit_projection = nn.Linear(self.d_model, 1)
        self.confidence_projection = nn.Linear(self.d_model * 2, 1)
        self.dropout = nn.Dropout(float(dropout))

        nn.init.zeros_(self.null_logit_projection.weight)
        nn.init.constant_(self.null_logit_projection.bias, float(initial_null_logit))
        nn.init.zeros_(self.confidence_projection.weight)
        nn.init.constant_(
            self.confidence_projection.bias,
            self._logit(initial_confidence),
        )

    @staticmethod
    def _logit(probability: float) -> float:
        return math.log(probability) - math.log1p(-probability)

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

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        series_ids: Optional[torch.Tensor] = None,
    ) -> TPPGatedMemoryState:
        """Create an empty event-memory bank for every batch row."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        normalized_ids = self._normalize_series_ids(
            series_ids,
            batch_size=batch_size,
            device=device,
        )
        return TPPGatedMemoryState(
            keys=torch.zeros(
                batch_size,
                self.memory_size,
                self.d_model,
                device=device,
                dtype=dtype,
            ),
            values=torch.zeros(
                batch_size,
                self.memory_size,
                self.d_model,
                device=device,
                dtype=dtype,
            ),
            valid_slots=torch.zeros(
                batch_size,
                self.memory_size,
                device=device,
                dtype=torch.bool,
            ),
            write_counts=torch.zeros(batch_size, device=device, dtype=torch.long),
            positions=torch.zeros(batch_size, device=device, dtype=torch.long),
            series_ids=normalized_ids,
        )

    def prepare_state(
        self,
        state: Optional[TPPGatedMemoryState],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        series_ids: Optional[torch.Tensor],
    ) -> TPPGatedMemoryState:
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
            raise ValueError("series_ids are required when reusing TPP memory state")
        if state.keys.shape != (batch_size, self.memory_size, self.d_model):
            raise ValueError("TPP memory state shape does not match the input batch")
        if state.values.shape != state.keys.shape:
            raise ValueError("TPP memory key and value state shapes must match")
        if state.valid_slots.shape != (batch_size, self.memory_size):
            raise ValueError("TPP memory valid-slot state shape is invalid")
        if state.write_counts.shape != (batch_size,) or state.positions.shape != (
            batch_size,
        ):
            raise ValueError("TPP memory counter state shape is invalid")
        if state.series_ids is None:
            raise ValueError("Reusable TPP memory state must carry series_ids")
        if state.keys.device != device or state.values.device != device:
            raise ValueError("TPP memory state device must match the input")
        if state.keys.dtype != dtype or state.values.dtype != dtype:
            raise ValueError("TPP memory state dtype must match the input")
        for tensor in (
            state.valid_slots,
            state.write_counts,
            state.positions,
            state.series_ids,
        ):
            if tensor.device != device:
                raise ValueError("TPP memory metadata device must match the input")

        changed = state.series_ids != normalized_ids
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

        return TPPGatedMemoryState(
            keys=reset_rows(state.keys, fresh.keys),
            values=reset_rows(state.values, fresh.values),
            valid_slots=reset_rows(state.valid_slots, fresh.valid_slots),
            write_counts=reset_rows(state.write_counts, fresh.write_counts),
            positions=reset_rows(state.positions, fresh.positions),
            series_ids=normalized_ids,
        )

    def retrieve_token(
        self,
        state: TPPGatedMemoryState,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read similarity-weighted sparse slots while a null slot competes."""
        if encoded.shape != (state.keys.size(0), self.d_model):
            raise ValueError("encoded must have shape [batch, d_model]")
        normalized = self.input_norm(encoded)
        query = F.normalize(
            self.query_projection(normalized),
            dim=-1,
            eps=1e-6,
        )
        scores = torch.einsum("bd,bmd->bm", query, state.keys)
        scores = scores / self.temperature
        scores = scores.masked_fill(~state.valid_slots, float("-inf"))
        selected_scores, selected_indices = torch.topk(scores, self.topk, dim=-1)
        selected_valid = torch.gather(state.valid_slots, 1, selected_indices)
        selected_scores = selected_scores.masked_fill(
            ~selected_valid,
            float("-inf"),
        )
        null_logits = self.null_logit_projection(normalized)
        candidate_logits = torch.cat((selected_scores, null_logits), dim=-1)
        candidate_weights = torch.softmax(candidate_logits, dim=-1)
        selected_weights = candidate_weights[:, :-1]
        null_probability = candidate_weights[:, -1]

        gather_indices = selected_indices.unsqueeze(-1).expand(
            -1,
            -1,
            self.d_model,
        )
        selected_values = torch.gather(state.values, 1, gather_indices)
        retrieved = torch.sum(
            selected_weights.unsqueeze(-1) * selected_values,
            dim=1,
        )
        retrieval_confidence = selected_weights.sum(dim=-1)
        learned_confidence = torch.sigmoid(
            self.confidence_projection(torch.cat((normalized, retrieved), dim=-1))
        ).squeeze(-1)
        effective_gate = retrieval_confidence * learned_confidence
        residual = effective_gate.unsqueeze(-1) * self.dropout(
            self.output_projection(self.output_norm(retrieved))
        )
        diagnostics = {
            "selected_indices": selected_indices,
            "selected_weights": selected_weights,
            "selected_valid": selected_valid,
            "null_probability": null_probability,
            "retrieval_confidence": retrieval_confidence,
            "learned_confidence": learned_confidence,
            "effective_gate": effective_gate,
        }
        return residual, diagnostics

    def write_token(
        self,
        state: TPPGatedMemoryState,
        encoded: torch.Tensor,
        valid: torch.Tensor,
    ) -> TPPGatedMemoryState:
        """Write one observed event into its row-local circular slot."""
        if encoded.shape != (state.keys.size(0), self.d_model):
            raise ValueError("encoded must have shape [batch, d_model]")
        valid = valid.to(device=encoded.device, dtype=torch.bool)
        normalized = self.input_norm(encoded)
        keys = F.normalize(
            self.key_projection(normalized),
            dim=-1,
            eps=1e-6,
        )
        values = self.value_projection(normalized)
        slot_indices = torch.remainder(state.write_counts, self.memory_size)
        slot_mask = F.one_hot(slot_indices, num_classes=self.memory_size).bool()
        write_slots = slot_mask & valid.unsqueeze(-1)
        next_keys = torch.where(write_slots.unsqueeze(-1), keys.unsqueeze(1), state.keys)
        next_values = torch.where(
            write_slots.unsqueeze(-1),
            values.unsqueeze(1),
            state.values,
        )
        return TPPGatedMemoryState(
            keys=next_keys,
            values=next_values,
            valid_slots=state.valid_slots | write_slots,
            write_counts=state.write_counts + valid.to(dtype=torch.long),
            positions=state.positions,
            series_ids=state.series_ids,
        )

    def forward_with_state(
        self,
        encoded: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
        state: Optional[TPPGatedMemoryState] = None,
        series_ids: Optional[torch.Tensor] = None,
        write_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, TPPGatedMemoryState, dict[str, torch.Tensor]]:
        """Read-before-write scan with explicit state-in/state-out semantics."""
        if encoded.ndim != 3 or encoded.size(-1) != self.d_model:
            raise ValueError("encoded must have shape [batch, sequence, d_model]")
        batch_size, seq_len, _ = encoded.shape
        mask = self._validate_mask(
            mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=encoded.device,
        )
        if write_mask is None:
            write_mask = mask
        else:
            write_mask = self._validate_mask(
                write_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                device=encoded.device,
            ) & mask
        use_compiled_scan = (
            state is None
            and series_ids is None
            and write_chunk_size is None
            and seq_len > 0
            and encoded.device.type == "cuda"
            and self.compile_cuda_scan
            and _COMPILED_TPP_GATED_SEQUENCE is not None
        )
        state = self.prepare_state(
            state,
            batch_size=batch_size,
            device=encoded.device,
            dtype=encoded.dtype,
            series_ids=series_ids,
        )
        chunk_size = seq_len if write_chunk_size is None else int(write_chunk_size)
        if chunk_size < 1:
            raise ValueError("write_chunk_size must be positive")

        if use_compiled_scan:
            normalized = self.input_norm(encoded)
            queries = F.normalize(
                self.query_projection(normalized),
                dim=-1,
                eps=1e-6,
            )
            write_keys = F.normalize(
                self.key_projection(normalized),
                dim=-1,
                eps=1e-6,
            )
            write_values = self.value_projection(normalized)
            # CUDAGraph reuses output buffers across calls; state must outlive them.
            scanned = tuple(
                tensor.clone()
                for tensor in _COMPILED_TPP_GATED_SEQUENCE(
                    state.keys,
                    state.values,
                    state.valid_slots,
                    state.write_counts,
                    encoded,
                    normalized,
                    queries,
                    write_keys,
                    write_values,
                    self.null_logit_projection(normalized),
                    self.confidence_projection.weight,
                    self.confidence_projection.bias,
                    self.output_norm.weight,
                    self.output_norm.bias,
                    self.output_projection.weight,
                    mask,
                    write_mask,
                    self.topk,
                    self.temperature,
                    self.dropout.p,
                    self.training,
                )
            )
            next_state = TPPGatedMemoryState(
                keys=scanned[1],
                values=scanned[2],
                valid_slots=scanned[3],
                write_counts=scanned[4],
                positions=state.positions + mask.sum(dim=1),
                series_ids=state.series_ids,
            )
            diagnostics = {
                "null_probability": scanned[5],
                "retrieval_confidence": scanned[6],
                "learned_confidence": scanned[7],
                "effective_gate": scanned[8],
                "selected_slot_count": scanned[9],
                "write_applied": scanned[10],
            }
            return scanned[0], next_state, diagnostics

        outputs: list[torch.Tensor] = []
        collected: dict[str, list[torch.Tensor]] = {
            "null_probability": [],
            "retrieval_confidence": [],
            "learned_confidence": [],
            "effective_gate": [],
            "selected_slot_count": [],
            "write_applied": [],
        }
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            for position in range(chunk_start, chunk_end):
                residual, read_diagnostics = self.retrieve_token(
                    state,
                    encoded[:, position],
                )
                valid = mask[:, position]
                valid_values = valid.to(dtype=encoded.dtype)
                prediction_state = (encoded[:, position] + residual) * valid_values.unsqueeze(-1)
                outputs.append(prediction_state)
                collected["null_probability"].append(
                    read_diagnostics["null_probability"] * valid_values
                )
                collected["retrieval_confidence"].append(
                    read_diagnostics["retrieval_confidence"] * valid_values
                )
                collected["learned_confidence"].append(
                    read_diagnostics["learned_confidence"] * valid_values
                )
                collected["effective_gate"].append(
                    read_diagnostics["effective_gate"] * valid_values
                )
                collected["selected_slot_count"].append(
                    read_diagnostics["selected_valid"].sum(dim=-1).to(encoded.dtype)
                    * valid_values
                )
                write_valid = write_mask[:, position]
                collected["write_applied"].append(write_valid.to(encoded.dtype))

                # Prediction state is complete before this observed event is stored.
                state = self.write_token(
                    state,
                    encoded[:, position],
                    write_valid,
                )

        if outputs:
            output = torch.stack(outputs, dim=1)
            diagnostics = {
                name: torch.stack(values, dim=1)
                for name, values in collected.items()
            }
        else:
            output = encoded.new_zeros(batch_size, 0, self.d_model)
            diagnostics = {
                name: encoded.new_zeros(batch_size, 0) for name in collected
            }
        state = replace(state, positions=state.positions + mask.sum(dim=1))
        return output, state, diagnostics

    def forward(
        self,
        encoded: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _, _ = self.forward_with_state(
            encoded,
            mask=mask,
            write_mask=write_mask,
        )
        return output


__all__ = ["TPPGatedMemoryState", "TPPSpecificGatedMemory"]
