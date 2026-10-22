"""NeurIPS 2025 Titans MAC, for last-observed-prefix prediction.

This implementation follows the final-paper residual GELU neural memory and
RMSNorm/RoPE/SwiGLU token mixer. Inner writes follow sequential surprise,
momentum and adaptive forgetting with an explicit gradient chunk size. MAC segment size
is a separate setting. Online states are functional, per-example and per-call;
outer training gradients propagate through all writes without detachment.

Intermediate segment outputs are NOT causal next-token predictions: the final
read uses the updated memory. Only ``encode_prefix`` is a forecast input, after
the caller has excluded the entire target token from the observed input.

The final paper's concatenation equation and diagram require an explicit
``concat_order`` selection; there is deliberately no inferred default. The
chosen policy must be recorded in the experiment contract. Local choices:
biased input projection; bias-free memory MLP and QKV; affine-free memory LN;
causal kernel-4 depthwise projections reset each segment; residual output gate
``y + sigmoid(Wg RMSNorm(y))*Wo(M_end(y))``; no inner clipping. Hidden memory
expansion is four. Architectural choices are exposed for contract verification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PAPER_EXPERIMENT_REFERENCE = {"persistent_memory_size": 128, "segment_size": 512, "retrieved_memory_size": 256, "gradient_chunk_size": 16,
                              "memory_expansion": 4, "conv_kernel": 4}


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * torch.rsqrt(inputs.square().mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.out(F.silu(self.gate(inputs)) * self.value(inputs))


def apply_rope(inputs: Tensor) -> Tensor:
    """Pairwise RoPE over concatenation positions, independently per segment."""
    width = inputs.shape[-1]
    if width % 2:
        raise ValueError("RoPE requires an even head dimension")
    frequencies = 10000.0 ** (-torch.arange(0, width, 2, device=inputs.device, dtype=inputs.dtype) / width)
    angles = torch.arange(inputs.shape[-2], device=inputs.device, dtype=inputs.dtype)[:, None] * frequencies
    cosine, sine = angles.cos(), angles.sin()
    even, odd = inputs[..., 0::2], inputs[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


@dataclass(frozen=True)
class PaperMemoryState:
    """Functional per-example fast weights; never stored on the module."""

    weights: tuple[Tensor, ...]
    weight_momentum: tuple[Tensor, ...]

    def parameters(self) -> tuple[Tensor, ...]:
        return self.weights


class CausalDepthwiseProjection(nn.Module):
    """Linear -> causal depthwise convolution -> SiLU; no right context."""

    def __init__(self, width: int, kernel_size: int = 4, *, normalize: bool = False):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.linear = nn.Linear(width, width, bias=False)
        self.conv = nn.Conv1d(width, width, kernel_size, groups=width, bias=False)
        self.kernel_size = kernel_size
        self.normalize = normalize

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            inputs = torch.where(mask.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        projected = self.linear(inputs).transpose(1, 2)
        result = F.silu(self.conv(F.pad(projected, (self.kernel_size - 1, 0))).transpose(1, 2))
        result = F.normalize(result, dim=-1, eps=1e-6) if self.normalize else result
        return torch.where(mask.unsqueeze(-1), result, torch.zeros_like(result)) if mask is not None else result


class PaperNeuralMemory(nn.Module):
    """Residual GELU memory: M(x) = x + LN(W_out GELU(W_in x))."""

    def __init__(self, d_model: int, hidden_dim: int, *, depth: int = 2, conv_kernel: int = 4, gradient_chunk_size: int = 16):
        super().__init__()
        if depth != 2:
            raise ValueError("This pinned paper comparison requires memory depth 2")
        if gradient_chunk_size < 1:
            raise ValueError("gradient_chunk_size must be positive")
        self.gradient_chunk_size = gradient_chunk_size
        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.query = CausalDepthwiseProjection(d_model, conv_kernel, normalize=True)
        self.key = CausalDepthwiseProjection(d_model, conv_kernel, normalize=True)
        self.value = CausalDepthwiseProjection(d_model, conv_kernel)
        self.update_rate = nn.ModuleList([nn.Linear(d_model, width) for width in (hidden_dim, d_model)])
        self.momentum_rate = nn.ModuleList([nn.Linear(d_model, width) for width in (hidden_dim, d_model)])
        self.forget_rate = nn.ModuleList([nn.Linear(d_model, width) for width in (hidden_dim, d_model)])
        self.initial_weights = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, d_model)),
            nn.Parameter(torch.empty(d_model, hidden_dim)),
        ])
        for weight in self.initial_weights:
            nn.init.xavier_uniform_(weight)
        for layers, probability in ((self.update_rate, .01), (self.momentum_rate, .9), (self.forget_rate, .001)):
            for layer in layers:
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, math.log(probability / (1 - probability)))

    def initial_state(self, batch_size: int) -> PaperMemoryState:
        weights = tuple(p.unsqueeze(0).expand(batch_size, -1, -1) for p in self.initial_weights)
        return PaperMemoryState(weights, tuple(torch.zeros_like(p) for p in weights))

    @staticmethod
    def read(state: PaperMemoryState, queries: Tensor) -> Tensor:
        value = queries
        for index, weight in enumerate(state.weights):
            value = torch.einsum("b...i,boi->b...o", value, weight)
            if index < len(state.weights) - 1:
                value = F.gelu(value)
        return queries + F.layer_norm(value, (value.shape[-1],), eps=1e-5)

    @staticmethod
    def associative_gradients(state: PaperMemoryState, keys: Tensor, values: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        """Per-example exact d||M(k)-v||²/dM, keeping outer derivatives.

        A general reverse pass through the MLP is independently checked against
        torch.autograd (including mixed outer derivatives) in the regression
        suite. Explicit tensor expressions also work under inference_mode.
        """
        activations = [keys]
        preactivations = []
        value = keys
        for index, weight in enumerate(state.weights):
            pre = torch.einsum("boi,bi->bo", weight, value)
            preactivations.append(pre)
            value = F.gelu(pre) if index < len(state.weights) - 1 else pre
            activations.append(value)
        centered = value - value.mean(-1, keepdim=True)
        inv_std = torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)
        normalized = centered * inv_std
        error = keys + normalized - values
        output_adjoint = 2 * error
        adjoint = inv_std * (output_adjoint - output_adjoint.mean(-1, keepdim=True)
                             - normalized * (output_adjoint * normalized).mean(-1, keepdim=True))
        gradients: list[Tensor] = []
        for index in reversed(range(len(state.weights))):
            if index < len(state.weights) - 1:
                pre = preactivations[index]
                cdf = .5 * (1 + torch.erf(pre / math.sqrt(2)))
                pdf = torch.exp(-.5 * pre.square()) / math.sqrt(2 * math.pi)
                adjoint = adjoint * (cdf + pre * pdf)
            gradients.insert(0, adjoint.unsqueeze(-1) * activations[index].unsqueeze(-2))
            adjoint = torch.einsum("boi,bo->bi", state.weights[index], adjoint)
        return tuple(gradients), error.square().sum(-1)

    def project_query(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.query(self.norm(inputs), mask)

    def write_reference(self, state: PaperMemoryState, inputs: Tensor, mask: Tensor) -> tuple[PaperMemoryState, dict[str, Tensor]]:
        """Gradient at chunk-start memory; b=1 reduces to the sequential equation.

        Token-dependent row-vector gates and momentum/forget recurrences remain
        sequential within a chunk. No claim of b>1 equivalence to b=1 is made.
        """
        # Compact valid tokens BEFORE QKV convolution and chunk assignment.
        # Otherwise h-padding between P/h/S changes both conv history and b16
        # boundaries despite skipped individual writes.
        original_mask = mask
        order = torch.argsort(mask.to(torch.int64), dim=1, descending=True, stable=True)
        safe = torch.where(mask.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        inputs = safe.gather(1, order.unsqueeze(-1).expand_as(safe))
        mask = mask.gather(1, order)
        normalized = self.norm(inputs)
        keys, values = self.key(normalized, mask), self.value(normalized, mask)
        theta = [torch.sigmoid(layer(normalized)) for layer in self.update_rate]
        eta = [torch.sigmoid(layer(normalized)) for layer in self.momentum_rate]
        alpha = [torch.sigmoid(layer(normalized)) for layer in self.forget_rate]
        losses = []
        for index in range(inputs.shape[1]):
            if index % self.gradient_chunk_size == 0:
                gradient_state = state
            gradients, loss = self.associative_gradients(gradient_state, keys[:, index], values[:, index])
            new_weights, new_momentum = [], []
            for layer, (parameter, momentum, gradient) in enumerate(zip(state.weights, state.weight_momentum, gradients)):
                shape = (inputs.shape[0], 1, 1)
                surprise = eta[layer][:, index, :, None] * momentum - theta[layer][:, index, :, None] * gradient
                next_parameter = (1 - alpha[layer][:, index, :, None]) * parameter + surprise
                valid = mask[:, index].reshape(shape)
                new_weights.append(torch.where(valid, next_parameter, parameter))
                new_momentum.append(torch.where(valid, surprise, momentum))
            state = PaperMemoryState(tuple(new_weights), tuple(new_momentum))
            losses.append(torch.where(mask[:, index], loss, torch.zeros_like(loss)))
        return state, {
            "associative_loss": torch.zeros_like(torch.stack(losses, 1)).scatter(1, order, torch.stack(losses, 1)),
            "write_applied": original_mask,
            "theta": [torch.zeros_like(g).scatter(1, order.unsqueeze(-1).expand_as(g), g) for g in theta],
            "eta": [torch.zeros_like(g).scatter(1, order.unsqueeze(-1).expand_as(g), g) for g in eta],
            "alpha": [torch.zeros_like(g).scatter(1, order.unsqueeze(-1).expand_as(g), g) for g in alpha],
        }

    @staticmethod
    def associative_factors(state: PaperMemoryState, keys: Tensor, values: Tensor):
        """Chunk gradients as (output adjoint, input activation) factors.

        Avoid constructing [batch, tokens, out, in] gradients. All operations
        remain differentiable with respect to both the fast state and inputs.
        """
        pre_hidden = torch.einsum("bhi,bti->bth", state.weights[0], keys)
        hidden = F.gelu(pre_hidden)
        output = torch.einsum("boh,bth->bto", state.weights[1], hidden)
        centered = output - output.mean(-1, keepdim=True)
        rstd = torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5)
        normalized = centered * rstd
        error = keys + normalized - values
        g = 2 * error
        gz = rstd * (g - g.mean(-1, keepdim=True) - normalized * (g * normalized).mean(-1, keepdim=True))
        gelu_derivative = .5 * (1 + torch.erf(pre_hidden / math.sqrt(2))) + pre_hidden * torch.exp(-.5 * pre_hidden.square()) / math.sqrt(2 * math.pi)
        gh = torch.einsum("boh,bto->bth", state.weights[1], gz) * gelu_derivative
        return ((gh, keys), (gz, hidden)), error.square().sum(-1)

    @staticmethod
    def recurrence_coefficients(theta: Tensor, eta: Tensor, alpha: Tensor, mask: Tensor):
        """Suffix products of rowwise [[1-alpha, eta], [0, eta]].

        Invalid tokens use the identity transition and zero forcing. The two
        returned coefficient sequences multiply each token's factored gradient
        directly into final memory and final momentum, respectively.
        """
        a = torch.ones_like(theta[:, 0])
        b = torch.zeros_like(a)
        d = torch.ones_like(a)
        memory_coefficients, momentum_coefficients = [], []
        for position in reversed(range(theta.shape[1])):
            valid = mask[:, position, None]
            rate = torch.where(valid, theta[:, position], torch.zeros_like(a))
            memory_coefficients.append(-rate * (a + b))
            momentum_coefficients.append(-rate * d)
            forget = torch.where(valid, 1 - alpha[:, position], torch.ones_like(a))
            off_diagonal = torch.where(valid, eta[:, position], torch.zeros_like(a))
            diagonal = torch.where(valid, eta[:, position], torch.ones_like(a))
            a, b, d = a * forget, a * off_diagonal + b * diagonal, d * diagonal
        return a, b, d, torch.stack(memory_coefficients[::-1], 1), torch.stack(momentum_coefficients[::-1], 1)

    def write(self, state: PaperMemoryState, inputs: Tensor, mask: Tensor):
        """Exact chunk-start writes with factored, batched gradient aggregation."""
        original_mask = mask
        order = torch.argsort(mask.to(torch.int64), dim=1, descending=True, stable=True)
        safe = torch.where(mask.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        packed = safe.gather(1, order.unsqueeze(-1).expand_as(safe))
        mask = mask.gather(1, order)
        normalized = self.norm(packed)
        keys, values = self.key(normalized, mask), self.value(normalized, mask)
        theta = [torch.sigmoid(layer(normalized)) for layer in self.update_rate]
        eta = [torch.sigmoid(layer(normalized)) for layer in self.momentum_rate]
        alpha = [torch.sigmoid(layer(normalized)) for layer in self.forget_rate]
        losses = []
        for start in range(0, inputs.shape[1], self.gradient_chunk_size):
            end = min(start + self.gradient_chunk_size, inputs.shape[1])
            factors, loss = self.associative_factors(state, keys[:, start:end], values[:, start:end])
            weights, momenta = [], []
            for layer, (weight, momentum, (adjoint, activation)) in enumerate(zip(state.weights, state.weight_momentum, factors)):
                a, b, d, memory_coef, momentum_coef = self.recurrence_coefficients(
                    theta[layer][:, start:end], eta[layer][:, start:end], alpha[layer][:, start:end], mask[:, start:end])
                memory_increment = torch.einsum("bto,bti->boi", memory_coef * adjoint, activation)
                momentum_increment = torch.einsum("bto,bti->boi", momentum_coef * adjoint, activation)
                weights.append(a.unsqueeze(-1) * weight + b.unsqueeze(-1) * momentum + memory_increment)
                momenta.append(d.unsqueeze(-1) * momentum + momentum_increment)
            state = PaperMemoryState(tuple(weights), tuple(momenta))
            losses.append(torch.where(mask[:, start:end], loss, torch.zeros_like(loss)))
        losses = torch.cat(losses, 1)
        return state, {
            "associative_loss": torch.zeros_like(losses).scatter(1, order, losses),
            "write_applied": original_mask,
            "theta": [torch.zeros_like(g).scatter(1, order.unsqueeze(-1).expand_as(g), g) for g in theta],
            "eta": [torch.zeros_like(g).scatter(1, order.unsqueeze(-1).expand_as(g), g) for g in eta],
            "alpha": [torch.zeros_like(g).scatter(1, order.unsqueeze(-1).expand_as(g), g) for g in alpha],
        }


class PaperMACAttentionBlock(nn.Module):
    """Full causal attention over the explicitly selected concatenation."""

    def __init__(self, width: int, heads: int, d_ff: int, dropout: float, conv_kernel: int):
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads = heads
        self.norm = RMSNorm(width)
        self.query = CausalDepthwiseProjection(width, conv_kernel, normalize=True)
        self.key = CausalDepthwiseProjection(width, conv_kernel, normalize=True)
        self.value = CausalDepthwiseProjection(width, conv_kernel)
        self.out = nn.Linear(width, width)
        self.ff_norm = RMSNorm(width)
        self.ff = SwiGLU(width, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        batch, length, width = tokens.shape
        normalized = self.norm(tokens)
        normalized = torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))
        def heads(value: Tensor) -> Tensor:
            return value.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
        query, key, value = (heads(project(normalized, valid)) for project in (self.query, self.key, self.value))
        query, key = apply_rope(F.normalize(query, dim=-1)), apply_rope(F.normalize(key, dim=-1))
        allowed = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        allowed = allowed[None, None] & valid[:, None, None, :]
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(width // self.heads)
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
        attended = (self.dropout(weights) @ value).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.dropout(self.out(attended))
        tokens = tokens + self.dropout(self.ff(self.ff_norm(tokens)))
        return torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))


class PaperMACBlock(nn.Module):
    """One complete MAC layer with its own memory, attention and feed-forward."""

    def __init__(self, width, heads, d_ff, dropout, persistent_size, hidden, depth, kernel, chunk, concat_order):
        super().__init__()
        self.concat_order = concat_order
        self.persistent_memory = nn.Parameter(torch.randn(1, persistent_size, width) * .02)
        self.neural_memory = PaperNeuralMemory(width, hidden, depth=depth, conv_kernel=kernel, gradient_chunk_size=chunk)
        self.attention = PaperMACAttentionBlock(width, heads, d_ff, dropout, kernel)
        self.output_norm = RMSNorm(width)
        self.output_gate = nn.Linear(width, width)
        self.memory_output_projection = nn.Linear(width, width, bias=False)

    def forward(self, current: Tensor, mask: Tensor, state: PaperMemoryState):
        retrieved = self.neural_memory.read(state, self.neural_memory.project_query(current, mask))
        persistent = self.persistent_memory.expand(current.shape[0], -1, -1)
        tokens = torch.cat((persistent, retrieved, current) if self.concat_order == "persistent_memory_current" else (persistent, current, retrieved), 1)
        prefix_valid = torch.ones(current.shape[0], persistent.shape[1], dtype=torch.bool, device=current.device)
        valid = torch.cat((prefix_valid, mask, mask), 1)
        # Compact each example without a Python row loop. RoPE positions and
        # causal convolution history then match its unpadded [P,h,S] sequence.
        order = torch.argsort(valid.to(torch.int64), dim=1, descending=True, stable=True)
        packed_tokens = tokens.gather(1, order.unsqueeze(-1).expand_as(tokens))
        packed_valid = valid.gather(1, order)
        y = self.attention(packed_tokens, packed_valid)
        # An all-padding segment must not write persistent tokens either.
        write_mask = packed_valid & mask.any(1, keepdim=True)
        state, diagnostics = self.neural_memory.write(state, y, write_mask)
        read = self.neural_memory.read(state, y)
        output = y + torch.sigmoid(self.output_gate(self.output_norm(y))) * self.memory_output_projection(read)
        output = torch.zeros_like(output).scatter(1, order.unsqueeze(-1).expand_as(output), output)
        event_start = persistent.shape[1] + (current.shape[1] if self.concat_order == "persistent_memory_current" else 0)
        current_output = output[:, event_start:event_start + current.shape[1]]
        return torch.where(mask.unsqueeze(-1), current_output, torch.zeros_like(current_output)), state, diagnostics


class TitansMACPaperEncoder(nn.Module):
    """Fresh state per call; only last observed-prefix output is a forecast input."""

    def __init__(self, input_dim: int = 2, d_model: int = 64, *, n_layers: int = 2, n_heads: int = 4,
                 d_ff: int = 128, memory_depth: int = 2, memory_hidden_dim: int | None = None,
                 persistent_memory_size: int = 128, segment_size: int = 512, max_len: int = 256,
                 dropout: float = .1, conv_kernel: int = 4, gradient_chunk_size: int = 16, concat_order: str):
        super().__init__()
        if min(segment_size, persistent_memory_size, max_len, n_layers) < 1:
            raise ValueError("segment, persistent, max_len and layer counts must be positive")
        if concat_order not in ("persistent_memory_current", "persistent_current_memory"):
            raise ValueError("concat_order must explicitly select the approved paper contract")
        self.concat_order = concat_order
        self.input_dim, self.d_model = input_dim, d_model
        self.segment_size, self.persistent_memory_size, self.max_len = segment_size, persistent_memory_size, max_len
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            PaperMACBlock(d_model, n_heads, d_ff, dropout, persistent_memory_size,
                          d_model * 4 if memory_hidden_dim is None else memory_hidden_dim,
                          memory_depth, conv_kernel, gradient_chunk_size, concat_order)
            for _ in range(n_layers)
        ])

    def implementation_choices(self) -> dict:
        """Record choices, not a claim that the unresolved paper gate passed."""
        return {
            "paper_version": "NeurIPS 2025 final",
            "concat_order": self.concat_order,
            "concat_equation_figure_conflict": "requires explicit experiment-contract resolution",
            "segment_size": self.segment_size,
            "persistent_memory_size": self.persistent_memory_size,
            "d_model": self.d_model,
            "attention_layers": len(self.layers),
            "memory_hidden_dim": self.layers[0].neural_memory.initial_weights[0].shape[0],
            "memory_depth": 2,
            "memory_layernorm_affine": False,
            "memory_bias": False,
            "inner_gradient_chunk_size": self.layers[0].neural_memory.gradient_chunk_size,
            "adaptive_gates": "token-dependent vector per output row of each memory matrix",
            "retrieved_memory_size": min(self.segment_size, self.max_len),
            "memory_count": len(self.layers),
            "write_scope": "all valid concatenation attention outputs before event extraction",
            "retrieval_length_policy": "one query per segment slot; paper experiment used 256",
            "inner_gradient_clipping": None,
            "gradient_semantics": "gradient at chunk-start memory; sequential recurrence within chunk",
            "write_padding_policy": "compact valid tokens before convolution and b16 chunk assignment",
            "attention_padding_policy": "compact valid concatenation before QKV convolution and RoPE; scatter back after read",
            "write_implementation": "factored gradients with exact 2x2 suffix recurrence; batched, no row loops",
            "conv_scope": "memory QKV and attention QKV; reset each segment",
            "conv_kernel": self.layers[0].neural_memory.query.kernel_size,
            "partial_segment_layout": "right-pad to min(segment_size, max_len) before concatenation",
            "physical_segment_slots": min(self.segment_size, self.max_len),
            "post_read": "direct updated-memory read on attention output",
            "output_gate": "y + sigmoid(Wg RMSNorm(y)) * Wo(M_end(y))",
            "forecast_positions": "last valid observed prefix only",
            "paper_experiment_reference": dict(PAPER_EXPERIMENT_REFERENCE),
        }

    def _mask(self, inputs: Tensor, mask: Tensor | None) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim or not 0 < inputs.shape[1] <= self.max_len:
            raise ValueError("inputs must be a nonempty [batch, length <= max_len, input_dim] tensor")
        if mask is None:
            mask = torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.bool)
        if mask.shape != inputs.shape[:2]:
            raise ValueError("mask shape must match [batch, length]")
        mask = mask.to(device=inputs.device, dtype=torch.bool)
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()) or bool((~mask.any(1)).any()):
            raise ValueError("each row must contain a nonempty, right-padded observed prefix")
        return mask

    def forward_with_state(self, inputs: Tensor, *, mask: Tensor | None = None) -> tuple[Tensor, tuple[PaperMemoryState, ...], dict[str, Tensor]]:
        """Return sequence, final per-layer memories and diagnostics, without caching state."""
        mask = self._mask(inputs, mask)
        # Excluded target/padding values (even NaN) cannot enter projections.
        safe = torch.where(mask.unsqueeze(-1), inputs, torch.zeros_like(inputs))
        projected = self.input_projection(safe)
        projected = torch.where(mask.unsqueeze(-1), projected, torch.zeros_like(projected))
        states = [layer.neural_memory.initial_state(inputs.shape[0]) for layer in self.layers]
        outputs, losses, writes = [], [], []
        for start in range(0, inputs.shape[1], self.segment_size):
            current = projected[:, start:start + self.segment_size]
            current_mask = mask[:, start:start + self.segment_size]
            actual_length = current.shape[1]
            # Fixed segment slots prevent batch padding from shifting the RoPE
            # distance between retrieved and current tokens in a partial segment.
            slots = min(self.segment_size, self.max_len)
            if actual_length < slots:
                current = F.pad(current, (0, 0, 0, slots - actual_length))
                current_mask = F.pad(current_mask, (0, slots - actual_length), value=False)
            segment_losses, segment_writes = [], []
            for index, layer in enumerate(self.layers):
                current, states[index], diagnostics = layer(current, current_mask, states[index])
                segment_losses.append(diagnostics["associative_loss"])
                segment_writes.append(diagnostics["write_applied"])
            outputs.append(current[:, :actual_length])
            losses.append(torch.stack(segment_losses, 1))
            writes.append(torch.stack(segment_writes, 1))
        return torch.cat(outputs, 1), tuple(states), {
            "associative_loss": torch.stack(losses, 2),
            "write_applied": torch.stack(writes, 2),
        }

    def forward(self, inputs: Tensor, *, mask: Tensor | None = None) -> Tensor:
        return self.forward_with_state(inputs, mask=mask)[0]

    def encode_prefix(self, inputs: Tensor, observed_mask: Tensor | None = None) -> Tensor:
        """Encode observed events only; the prediction target must be excluded."""
        mask = self._mask(inputs, observed_mask)
        encoded = self.forward(inputs, mask=mask)
        return encoded[torch.arange(inputs.shape[0], device=inputs.device), mask.sum(1) - 1]


__all__ = ["TitansMACPaperEncoder", "PaperNeuralMemory", "PaperMemoryState", "CausalDepthwiseProjection"]
