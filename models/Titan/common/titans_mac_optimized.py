"""Semantics-preserving execution adapter for the frozen Titans-MAC encoder."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .titans_memory_stability import clip_associative_gradients

from .titans_mac import (
    TitansMACEncoder,
    TitansMemoryState,
    TitansNeuralMemory,
    _scan_titans_write_sequence,
)


def _scan_titans_write_sequence_state_only(
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
    gradient_max_norm: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run the frozen recurrence without materializing unused diagnostics."""
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
        output_gradient = 2.0 * (prediction - value)
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
            weight_1, momentum_weight_1, grad_weight_1
        )
        bias_1, momentum_bias_1 = update_tensor(
            bias_1, momentum_bias_1, grad_bias_1
        )
        weight_2, momentum_weight_2 = update_tensor(
            weight_2, momentum_weight_2, grad_weight_2
        )
        bias_2, momentum_bias_2 = update_tensor(
            bias_2, momentum_bias_2, grad_bias_2
        )

    return (
        weight_1,
        bias_1,
        weight_2,
        bias_2,
        momentum_weight_1,
        momentum_bias_1,
        momentum_weight_2,
        momentum_bias_2,
    )


_COMPILED_DIAGNOSTIC_SCAN = (
    torch.compile(
        _scan_titans_write_sequence,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    if hasattr(torch, "compile")
    else None
)
_COMPILED_STATE_ONLY_SCAN = (
    torch.compile(
        _scan_titans_write_sequence_state_only,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    if hasattr(torch, "compile")
    else None
)


class OptimizedTitansNeuralMemory(TitansNeuralMemory):
    """Frozen neural memory with fixed-shape CUDA scans and optional diagnostics."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_expansion: int = 2,
        initial_update_rate: float = 0.01,
        initial_momentum: float = 0.9,
        initial_forgetting: float = 0.001,
        compile_cuda_scan: bool = True,
        gradient_max_norm: float | None = None,
        compiled_scan_batch_size: int = 128,
        compiled_scan_chunk_size: int = 16,
    ) -> None:
        super().__init__(
            d_model,
            hidden_expansion=hidden_expansion,
            initial_update_rate=initial_update_rate,
            initial_momentum=initial_momentum,
            initial_forgetting=initial_forgetting,
            compile_cuda_scan=compile_cuda_scan,
            gradient_max_norm=gradient_max_norm,
        )
        if compiled_scan_batch_size < 1 or compiled_scan_chunk_size < 1:
            raise ValueError("Compiled scan dimensions must be positive")
        self.compiled_scan_batch_size = int(compiled_scan_batch_size)
        self.compiled_scan_chunk_size = int(compiled_scan_chunk_size)
        self.collect_diagnostics_by_default = True

    @staticmethod
    def _pad_dimension(
        tensor: torch.Tensor,
        *,
        dimension: int,
        target_size: int,
    ) -> torch.Tensor:
        current_size = tensor.size(dimension)
        if current_size > target_size:
            raise ValueError("target_size cannot be smaller than the tensor")
        if current_size == target_size:
            return tensor
        padding_shape = list(tensor.shape)
        padding_shape[dimension] = target_size - current_size
        return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=dimension)

    def _compiled_chunk(
        self,
        state: TitansMemoryState,
        keys: torch.Tensor,
        values: torch.Tensor,
        theta: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        write_mask: torch.Tensor,
        *,
        target_sequence_length: int,
        collect_diagnostics: bool,
    ) -> tuple[TitansMemoryState, dict[str, torch.Tensor]]:
        batch_size, sequence_length = write_mask.shape
        if batch_size > self.compiled_scan_batch_size:
            raise ValueError(
                "Compiled Titans-MAC batches cannot exceed "
                f"{self.compiled_scan_batch_size} rows"
            )
        if target_sequence_length != self.compiled_scan_chunk_size:
            raise ValueError("Compiled Titans-MAC scan shape must remain frozen")
        if sequence_length > target_sequence_length:
            raise ValueError("Compiled chunk exceeds the frozen scan length")
        target_batch_size = self.compiled_scan_batch_size
        memory = tuple(
            self._pad_dimension(
                tensor,
                dimension=0,
                target_size=target_batch_size,
            )
            for tensor in state.memory_tensors()
        )
        momenta = tuple(
            self._pad_dimension(
                tensor,
                dimension=0,
                target_size=target_batch_size,
            )
            for tensor in state.momentum_tensors()
        )

        def pad_projected(tensor: torch.Tensor) -> torch.Tensor:
            tensor = self._pad_dimension(
                tensor,
                dimension=1,
                target_size=target_sequence_length,
            )
            return self._pad_dimension(
                tensor,
                dimension=0,
                target_size=target_batch_size,
            )

        scan_inputs = tuple(
            pad_projected(tensor)
            for tensor in (keys, values, theta, eta, alpha, write_mask)
        )
        compiled = (
            _COMPILED_DIAGNOSTIC_SCAN
            if collect_diagnostics
            else _COMPILED_STATE_ONLY_SCAN
        )
        if compiled is None:
            raise RuntimeError("Compiled Titans scan is unavailable")
        scanned = compiled(
            *memory, *momenta, *scan_inputs, self.gradient_max_norm
        )
        next_state = TitansMemoryState(
            *(tensor[:batch_size] for tensor in scanned[:8]),
            positions=state.positions,
            series_ids=state.series_ids,
        )
        if not collect_diagnostics:
            return next_state, {}
        return next_state, {
            "associative_loss": scanned[8][:batch_size, :sequence_length],
            "update_rate": scanned[9][:batch_size, :sequence_length],
            "momentum_rate": scanned[10][:batch_size, :sequence_length],
            "forgetting_rate": scanned[11][:batch_size, :sequence_length],
            "write_applied": scanned[12][:batch_size, :sequence_length],
        }

    def write_sequence(
        self,
        state: TitansMemoryState,
        inputs: torch.Tensor,
        write_mask: torch.Tensor,
        *,
        chunk_size: Optional[int] = None,
        collect_diagnostics: Optional[bool] = None,
    ) -> tuple[TitansMemoryState, dict[str, torch.Tensor]]:
        if inputs.ndim != 3 or inputs.size(-1) != self.d_model:
            raise ValueError("inputs must have shape [batch, sequence, d_model]")
        if write_mask.shape != inputs.shape[:2]:
            raise ValueError("write_mask shape must match the input sequence")
        collect = (
            self.collect_diagnostics_by_default
            if collect_diagnostics is None
            else bool(collect_diagnostics)
        )
        if inputs.size(1) == 0:
            if not collect:
                return state, {}
            empty = inputs.new_zeros(inputs.size(0), 0)
            return state, {
                name: empty
                for name in (
                    "associative_loss",
                    "update_rate",
                    "momentum_rate",
                    "forgetting_rate",
                    "write_applied",
                )
            }
        step = inputs.size(1) if chunk_size is None else int(chunk_size)
        if step < 1:
            raise ValueError("chunk_size must be positive")
        use_compiled = (
            inputs.device.type == "cuda"
            and self.compile_cuda_scan
            and _COMPILED_DIAGNOSTIC_SCAN is not None
            and _COMPILED_STATE_ONLY_SCAN is not None
        )
        if not use_compiled:
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
                    if collect:
                        for name, value in diagnostics.items():
                            collected[name].append(value)
            if not collect:
                return state, {}
            return state, {
                name: torch.stack(values, dim=1)
                for name, values in collected.items()
            }

        keys, values, theta, eta, alpha = self._project_write(inputs)
        scheduling_step = (
            self.compiled_scan_chunk_size if chunk_size is None else int(chunk_size)
        )
        if scheduling_step < 1:
            raise ValueError("chunk_size must be positive")
        collected = {
            "associative_loss": [],
            "update_rate": [],
            "momentum_rate": [],
            "forgetting_rate": [],
            "write_applied": [],
        }
        for schedule_start in range(0, inputs.size(1), scheduling_step):
            schedule_end = min(schedule_start + scheduling_step, inputs.size(1))
            for chunk_start in range(
                schedule_start,
                schedule_end,
                self.compiled_scan_chunk_size,
            ):
                chunk_end = min(
                    chunk_start + self.compiled_scan_chunk_size,
                    schedule_end,
                )
                state, diagnostics = self._compiled_chunk(
                    state,
                    keys[:, chunk_start:chunk_end],
                    values[:, chunk_start:chunk_end],
                    theta[:, chunk_start:chunk_end, 0],
                    eta[:, chunk_start:chunk_end, 0],
                    alpha[:, chunk_start:chunk_end, 0],
                    write_mask[:, chunk_start:chunk_end],
                    target_sequence_length=self.compiled_scan_chunk_size,
                    collect_diagnostics=collect,
                )
                for name, value in diagnostics.items():
                    collected[name].append(value)
        if not collect:
            return state, {}
        return state, {
            name: torch.cat(values, dim=1)
            for name, values in collected.items()
        }


class OptimizedTitansMACEncoder(TitansMACEncoder):
    """Frozen MAC topology using the optimized neural-memory scheduler."""

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        write_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.neural_memory
        if not isinstance(memory, OptimizedTitansNeuralMemory):
            raise TypeError("Optimized encoder requires optimized neural memory")
        previous = memory.collect_diagnostics_by_default
        memory.collect_diagnostics_by_default = False
        try:
            encoded, _, _ = super().forward_with_state(
                inputs,
                mask=mask,
                write_mask=write_mask,
            )
        finally:
            memory.collect_diagnostics_by_default = previous
        return encoded


def _optimized_encoder_from_frozen(
    frozen: TitansMACEncoder,
) -> OptimizedTitansMACEncoder:
    first_parameter = next(frozen.parameters())
    first_layer = frozen.layers[0]
    # Reconstructing an equivalent module must not perturb training RNG state.
    cpu_rng_state = torch.random.get_rng_state()
    try:
        optimized = OptimizedTitansMACEncoder(
            input_dim=frozen.input_dim,
            d_model=frozen.d_model,
            n_layers=len(frozen.layers),
            n_heads=first_layer.attention.num_heads,
            d_ff=first_layer.feedforward[0].out_features,
            persistent_memory_size=frozen.persistent_memory_size,
            segment_size=frozen.segment_size,
            max_len=frozen.max_len,
            dropout=float(first_layer.attention.dropout),
        ).to(device=first_parameter.device, dtype=first_parameter.dtype)
        replacement = OptimizedTitansNeuralMemory(
            frozen.d_model,
            gradient_max_norm=getattr(frozen.neural_memory, "gradient_max_norm", None),
        ).to(
            device=first_parameter.device,
            dtype=first_parameter.dtype,
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
    optimized.neural_memory = replacement
    optimized.load_state_dict(frozen.state_dict(), strict=True)
    optimized.train(frozen.training)
    return optimized


def apply_titantpp_mac_semantic_optimization(model: Any) -> Any:
    """Replace only B1 execution classes while preserving all state-dict keys."""
    frozen = getattr(model, "titans_mac_encoder", None)
    if frozen is None:
        raise TypeError("TitanTPP-MAC optimization requires a B1 model")
    if isinstance(frozen, OptimizedTitansMACEncoder):
        return model
    model.titans_mac_encoder = _optimized_encoder_from_frozen(frozen)
    return model


def optimization_metadata() -> dict[str, Any]:
    return {
        "optimization_contract_id": (
            "count_aware_titantpp_mac_semantic_optimization_v1"
        ),
        "scan_backend": "fixed_shape_compiled_state_only_cuda",
        "compiled_scan_batch_size": 128,
        "compiled_scan_chunk_size": 16,
        "mac_segment_size": 16,
        "training_diagnostics": "disabled_state_only_scan",
        "checkpoint_parameter_keys_changed": False,
    }


__all__ = [
    "OptimizedTitansMACEncoder",
    "OptimizedTitansNeuralMemory",
    "_scan_titans_write_sequence_state_only",
    "apply_titantpp_mac_semantic_optimization",
    "optimization_metadata",
]
