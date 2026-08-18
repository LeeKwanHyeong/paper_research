from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualMemoryBuffer(nn.Module):
    """
    Simple FIFO contextual memory buffer: keeps last `size` tokens (detached by caller).
    Shape: [B, M, D] or [1, M, D]
    """
    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def update(self, mem: Optional[torch.Tensor], x_new: torch.Tensor) -> torch.Tensor:
        """
        mem: [B, M, D] or None
        x_new: [B, L, D] (usually L=lookback or layer output)
        returns: [B, M', D] where M' == self.size (if enough)
        """
        if self.size <= 0:
            # no contextual memory
            return x_new[:, :0, :]

        # take last tokens from x_new
        take = min(self.size, x_new.size(1))
        tail = x_new[:, -take:, :]  # [B, take, D]

        if mem is None or mem.numel() == 0:
            out = tail
        else:
            out = torch.cat([mem, tail], dim=1)  # [B, M+take, D]
            if out.size(1) > self.size:
                out = out[:, -self.size:, :]

        return out


class MemoryAttention(nn.Module):
    """
    Attention with optional contextual + persistent memory.
    - contextual memory: updated outside (encoder loop) via update_contextual_memory()
    - persistent memory: learnable parameters
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        contextual_mem_size: int,
        persistent_mem_size: int,
        dropout: float = 0.1,
        use_causal: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.use_causal = use_causal
        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.drop = nn.Dropout(float(dropout))

        self.contextual_mem_size = int(contextual_mem_size)
        self.persistent_mem_size = int(persistent_mem_size)

        # persistent (learnable) memory: [1, Mp, D]
        if self.persistent_mem_size > 0:
            self.persistent_mem = nn.Parameter(torch.randn(1, self.persistent_mem_size, self.d_model) * 0.02)
        else:
            self.register_parameter("persistent_mem", None)

        # contextual memory buffer holder (not parameter)
        self._ctx_buf = ContextualMemoryBuffer(self.contextual_mem_size)
        self._ctx_mem: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update_contextual_memory(self, x_detached: torch.Tensor):
        """
        x_detached: [B, L, D] (caller should detach)
        """
        self._ctx_mem = self._ctx_buf.update(self._ctx_mem, x_detached)

    @torch.no_grad()
    def reset_contextual_memory(self) -> None:
        """
        Clear the contextual memory buffer.

        TTM-Lite uses one online memory stream per series. Resetting here keeps
        memory from one part/grid cell from leaking into another series.
        """
        self._ctx_mem = None

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Hd]
        B, T, D = t.shape
        t = t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return t

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Hd] -> [B, T, D]
        B, H, T, Hd = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * Hd)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, L, D]
        is_causal: True for Autoregressive tasks (RMTPPs), False for Seq2Seq Encoder
        """
        B, L, D = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # --- Memory Concatenation (MAC) ---
        mem_list = []
        if self._ctx_mem is not None and self._ctx_mem.numel() > 0:
            mem_list.append(self._ctx_mem.to(device=x.device, dtype=x.dtype))  # [B, M_ctx, D]
        if self.persistent_mem is not None:
            mem_list.append(self.persistent_mem.to(device=x.device, dtype=x.dtype).expand(B, -1, -1))  # [B, M_per, D]

        n_mem = 0
        if len(mem_list) > 0:
            mem = torch.cat(mem_list, dim=1)
            n_mem = mem.size(1)
            k = torch.cat([mem, k], dim=1)  # Key extends to [B, L + n_mem, D]
            v = torch.cat([mem, v], dim=1)  # Value extends

        qh = self._split_heads(q)  # [B, H, L, Hd]
        kh = self._split_heads(k)  # [B, H, L + n_mem, Hd]
        vh = self._split_heads(v)

        # --- Attention Score ---
        # scores: [B, H, L, L + n_mem]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale

        # --- Causal Masking ---
        full_mask: torch.Tensor | None = None
        if self.use_causal:
            # Create mask: [L, L + n_mem]
            # Memory part (left side): 1 (Visible)
            # Sequence part (right side): Triangular (Causal)

            # 1. Memory is always fully visible
            mask_mem = torch.ones(L, n_mem, device=x.device, dtype=torch.bool)

            # 2. Sequence is causal
            mask_seq = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))

            # 3. Concat -> [L, L + n_mem]
            full_mask = torch.cat([mask_mem, mask_seq], dim=1)

        if mask is not None:
            if mask.shape != (B, L):
                raise ValueError(f"Expected attention mask {(B, L)}, got {tuple(mask.shape)}")
            valid_keys = torch.cat(
                [
                    torch.ones(B, n_mem, device=x.device, dtype=torch.bool),
                    mask.to(device=x.device, dtype=torch.bool),
                ],
                dim=1,
            )
            key_mask = valid_keys[:, None, None, :]
            if full_mask is None:
                full_mask = key_mask
            else:
                full_mask = full_mask[None, None, :, :] & key_mask

            # Padded queries are discarded after attention, but an open
            # diagonal prevents all-masked rows and NaNs when no memory exists.
            if n_mem == 0:
                invalid_queries = ~mask.to(device=x.device, dtype=torch.bool)
                if invalid_queries.any():
                    full_mask = full_mask.expand(B, 1, L, L).clone()
                    batch_ids, positions = invalid_queries.nonzero(as_tuple=True)
                    full_mask[batch_ids, 0, positions, positions] = True

        if full_mask is not None:
            scores = scores.masked_fill(~full_mask, float("-inf"))

        att = F.softmax(scores, dim=-1)
        att = self.drop(att)

        out = torch.matmul(att, vh)  # [B, H, L, Hd]
        out = self._merge_heads(out)
        out = self.out_proj(out)
        if mask is not None:
            out = out * mask.to(device=out.device, dtype=out.dtype).unsqueeze(-1)
        return out


class LMM(nn.Module):
    """
    Local Memory Matching:
    - learnable memory bank (persistent) of shape [1, M, D]
    - matches encoded tokens to top-k memory vectors and adds mean(selected)
    """
    def __init__(self, d_model: int, mem_size: int = 128, topk: int = 8):
        super().__init__()
        self.d_model = int(d_model)
        self.mem_size = int(mem_size)
        self.topk = int(topk)

        if self.mem_size > 0:
            self.mem = nn.Parameter(torch.randn(1, self.mem_size, self.d_model) * 0.02)
        else:
            self.register_parameter("mem", None)

    def forward(self, encoded: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        encoded: [B, L, D]
        memory:
          - None -> use self.mem
          - [M, D] or [1, M, D] or [B, M, D]
        """
        B, L, D = encoded.shape

        if memory is None:
            memory = self.mem

        if memory is None or memory.numel() == 0:
            return encoded

        if memory.dim() == 2:
            memory = memory.unsqueeze(0)  # [1, M, D]
        if memory.size(0) == 1:
            memory = memory.expand(B, -1, -1)  # [B, M, D]

        M = memory.size(1)
        k = min(self.topk, M)
        if k <= 0:
            return encoded

        enc_n = F.normalize(encoded, p=2, dim=-1)
        mem_n = F.normalize(memory, p=2, dim=-1)

        sim = torch.matmul(enc_n, mem_n.transpose(-2, -1))  # [B, L, M]
        _, idx = torch.topk(sim, k, dim=-1)

        mem_exp = memory.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, M, D]
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, D)    # [B, L, k, D]
        selected = torch.gather(mem_exp, 2, idx_exp).mean(dim=2)  # [B, L, D]
        return encoded + selected


class GatedSoftMemory(nn.Module):
    """Differentiable static memory retrieval with a zero-init residual gate."""

    def __init__(
        self,
        d_model: int,
        mem_size: int = 64,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if mem_size < 1:
            raise ValueError("mem_size must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.d_model = int(d_model)
        self.mem_size = int(mem_size)
        self.temperature = float(temperature)

        self.input_norm = nn.LayerNorm(self.d_model)
        self.query_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gate_proj = nn.Linear(self.d_model, self.d_model)
        self.memory_keys = nn.Parameter(
            torch.randn(1, self.mem_size, self.d_model) * 0.02
        )
        self.memory_values = nn.Parameter(
            torch.randn(1, self.mem_size, self.d_model) * 0.02
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(float(dropout))

    def retrieve(
        self,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the retrieved values and dense memory attention weights."""
        normalized = self.input_norm(encoded)
        query = self.query_proj(normalized)
        keys = self.key_proj(self.memory_keys).expand(encoded.size(0), -1, -1)
        values = self.value_proj(self.memory_values).expand(
            encoded.size(0), -1, -1
        )
        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = scores / (math.sqrt(self.d_model) * self.temperature)
        weights = torch.softmax(scores, dim=-1)
        retrieved = self.output_proj(torch.matmul(weights, values))
        return retrieved, weights

    def forward(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        retrieved, _ = self.retrieve(encoded)
        gate = torch.sigmoid(self.gate_proj(self.input_norm(encoded)))
        residual = torch.tanh(self.residual_scale) * gate * self.dropout(retrieved)
        output = encoded + residual
        if mask is not None:
            output = output * mask.to(
                device=output.device,
                dtype=output.dtype,
            ).unsqueeze(-1)
        return output


class SurpriseGatedMemory(nn.Module):
    """Causal low-rank fast-weight memory with truncated surprise updates."""

    def __init__(
        self,
        d_model: int,
        memory_rank: int = 16,
        chunk_size: int = 32,
        initial_update_rate: float = 0.01,
        initial_retention: float = 0.99,
        initial_momentum: float = 0.5,
        memory_clip: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if memory_rank < 1:
            raise ValueError("memory_rank must be positive")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        for name, value in (
            ("initial_update_rate", initial_update_rate),
            ("initial_retention", initial_retention),
            ("initial_momentum", initial_momentum),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if memory_clip <= 0.0:
            raise ValueError("memory_clip must be positive")

        self.d_model = int(d_model)
        self.memory_rank = int(memory_rank)
        self.chunk_size = int(chunk_size)
        self.memory_clip = float(memory_clip)

        self.input_norm = nn.LayerNorm(self.d_model)
        self.query_proj = nn.Linear(self.d_model, self.memory_rank, bias=False)
        self.key_proj = nn.Linear(self.d_model, self.memory_rank, bias=False)
        self.value_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.retrieval_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gate_proj = nn.Linear(self.d_model, self.d_model)
        self.update_rate_proj = nn.Linear(self.d_model, 1)
        self.retention_proj = nn.Linear(self.d_model, 1)
        self.momentum_logit = nn.Parameter(
            torch.tensor(self._logit(initial_momentum))
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(float(dropout))

        nn.init.zeros_(self.update_rate_proj.weight)
        nn.init.constant_(
            self.update_rate_proj.bias,
            self._logit(initial_update_rate),
        )
        nn.init.zeros_(self.retention_proj.weight)
        nn.init.constant_(
            self.retention_proj.bias,
            self._logit(initial_retention),
        )

    @staticmethod
    def _logit(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    def process(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Process one independent sequence batch and return memory diagnostics."""
        if encoded.ndim != 3 or encoded.size(-1) != self.d_model:
            raise ValueError(
                "encoded must have shape [batch, sequence, d_model]"
            )
        batch_size, seq_len, _ = encoded.shape
        if mask is None:
            mask = torch.ones(
                batch_size,
                seq_len,
                device=encoded.device,
                dtype=torch.bool,
            )
        elif mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"Expected mask {(batch_size, seq_len)}, got {tuple(mask.shape)}"
            )
        else:
            mask = mask.to(device=encoded.device, dtype=torch.bool)

        memory = encoded.new_zeros(batch_size, self.d_model, self.memory_rank)
        momentum = torch.zeros_like(memory)
        outputs: list[torch.Tensor] = []
        surprise_values: list[torch.Tensor] = []
        update_rates: list[torch.Tensor] = []
        retentions: list[torch.Tensor] = []
        retrieval_gates: list[torch.Tensor] = []
        momentum_rate = torch.sigmoid(self.momentum_logit)
        rank_scale = math.sqrt(self.memory_rank)

        for chunk_start in range(0, seq_len, self.chunk_size):
            if chunk_start:
                memory = memory.detach()
                momentum = momentum.detach()
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            for position in range(chunk_start, chunk_end):
                valid = mask[:, position]
                valid_vector = valid.to(dtype=encoded.dtype).unsqueeze(-1)
                valid_state = valid_vector.unsqueeze(-1)
                current = encoded[:, position]
                normalized = self.input_norm(current)

                query = F.normalize(
                    self.query_proj(normalized),
                    dim=-1,
                    eps=1e-6,
                )
                key = F.normalize(
                    self.key_proj(normalized),
                    dim=-1,
                    eps=1e-6,
                )
                value = torch.tanh(self.value_proj(normalized))

                retrieved = torch.bmm(memory, query.unsqueeze(-1)).squeeze(-1)
                retrieved = self.output_proj(self.retrieval_norm(retrieved))
                retrieval_gate = torch.sigmoid(self.gate_proj(normalized))
                residual = (
                    torch.tanh(self.residual_scale)
                    * retrieval_gate
                    * self.dropout(retrieved)
                )
                outputs.append((current + residual) * valid_vector)

                predicted_value = torch.bmm(
                    memory,
                    key.unsqueeze(-1),
                ).squeeze(-1)
                error = value - predicted_value
                surprise = torch.linalg.vector_norm(error, dim=-1)
                update_rate = torch.sigmoid(self.update_rate_proj(normalized))
                retention = torch.sigmoid(self.retention_proj(normalized))
                gradient_step = torch.bmm(
                    error.unsqueeze(-1),
                    key.unsqueeze(1),
                ) / rank_scale
                next_momentum = (
                    momentum_rate * momentum
                    + update_rate.unsqueeze(-1) * gradient_step
                )
                next_memory = (
                    retention.unsqueeze(-1) * memory + next_momentum
                ).clamp(min=-self.memory_clip, max=self.memory_clip)
                memory = valid_state * next_memory + (1.0 - valid_state) * memory
                momentum = (
                    valid_state * next_momentum + (1.0 - valid_state) * momentum
                )

                surprise_values.append(surprise * valid_vector.squeeze(-1))
                update_rates.append(update_rate.squeeze(-1) * valid_vector.squeeze(-1))
                retentions.append(retention.squeeze(-1) * valid_vector.squeeze(-1))
                retrieval_gates.append(
                    retrieval_gate.mean(dim=-1) * valid_vector.squeeze(-1)
                )

        output = torch.stack(outputs, dim=1)
        diagnostics = {
            "surprise": torch.stack(surprise_values, dim=1),
            "update_rate": torch.stack(update_rates, dim=1),
            "retention": torch.stack(retentions, dim=1),
            "retrieval_gate": torch.stack(retrieval_gates, dim=1),
        }
        return output, diagnostics

    def forward(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _ = self.process(encoded, mask=mask)
        return output
