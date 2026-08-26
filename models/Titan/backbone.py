from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Titan import MemoryAttention


def _summarize_context_update(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Select what a processed chunk contributes to contextual memory.

    `all` reproduces the previous behavior. `last` and `mean` compress each
    chunk to one token so chunked TTM can reduce memory-attention cost too.
    """
    normalized = str(mode or "all").strip().lower()
    if normalized == "all":
        return x
    if normalized == "last":
        return x[:, -1:, :]
    if normalized == "mean":
        return x.mean(dim=1, keepdim=True)
    raise ValueError("context_memory_update must be one of: all, last, mean")


class TitanBackbone(nn.Module):
    """
    Titan block (pre-norm):
      x -> attn -> residual -> ff -> residual
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        contextual_mem_size: int,
        persistent_mem_size: int,
        dropout: float = 0.1,
        use_causal: bool =  True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MemoryAttention(
            d_model=d_model,
            n_heads=n_heads,
            contextual_mem_size=contextual_mem_size,
            persistent_mem_size=persistent_mem_size,
            dropout=dropout,
            use_causal= use_causal
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # attn
        h = self.norm1(x)
        h = self.attn(h, mask=mask)
        x = x + self.drop1(h)
        if mask is not None:
            x = x * mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

        # ff
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.drop2(h)
        if mask is not None:
            x = x * mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return x

    @torch.no_grad()
    def update_contextual_memory(self, x_detached: torch.Tensor) -> None:
        """
        Append observed hidden tokens to this block's contextual memory.
        """
        self.attn.update_contextual_memory(x_detached.detach())

    @torch.no_grad()
    def reset_contextual_memory(self) -> None:
        """
        Clear this block's contextual memory state.
        """
        self.attn.reset_contextual_memory()


class MemoryEncoder(nn.Module):
    """
    Input projection + TitanBackbone x n_layers
    - learnable positional embedding (optional)
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        contextual_mem_size: int,
        persistent_mem_size: int,
        dropout: float = 0.1,
        *,
        use_context_update: bool = False,
        use_pos_emb: bool = True,
        max_len: int = 512,
        use_causal: bool = True
    ):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model))

        self.layers = nn.ModuleList(
            [
                TitanBackbone(
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    contextual_mem_size=int(contextual_mem_size),
                    persistent_mem_size=int(persistent_mem_size),
                    dropout=float(dropout),
                    use_causal = use_causal
                )
                for _ in range(int(n_layers))
            ]
        )

        self.use_context_update = bool(use_context_update)
        self.use_pos_emb = bool(use_pos_emb)
        self.max_len = int(max_len)

        if self.use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, self.max_len, int(d_model)) * 0.02)
        else:
            self.register_parameter("pos_emb", None)

    def _get_pos(
        self,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        offset: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        batch_offsets: torch.Tensor | None = None
        if torch.is_tensor(offset):
            if offset.ndim != 1:
                raise ValueError("position_offset tensor must have shape [batch]")
            batch_offsets = offset.to(device=device, dtype=torch.long)
            if bool((batch_offsets < 0).any()):
                raise ValueError("position_offset values must be nonnegative")

        if self.pos_emb is None:
            batch_size = 1 if batch_offsets is None else batch_offsets.size(0)
            return torch.zeros(
                batch_size,
                L,
                self.input_proj.out_features,
                device=device,
                dtype=dtype,
            )

        if batch_offsets is not None:
            indices = batch_offsets.unsqueeze(1) + torch.arange(
                L,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            end = int(indices.max().item()) + 1 if indices.numel() else 0
            table = self.pos_emb.to(device=device, dtype=dtype)
            if end > table.size(1):
                table = F.interpolate(
                    table.transpose(1, 2),
                    size=end,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            return table[0][indices]

        offset = max(int(offset), 0)
        end = offset + int(L)

        if end <= self.pos_emb.size(1):
            return self.pos_emb[:, offset:end, :].to(device=device, dtype=dtype)

        # interpolate if longer than max_len
        pe = self.pos_emb.to(device=device, dtype=dtype)     # [1, max_len, D]
        pe_t = pe.transpose(1, 2)                            # [1, D, max_len]
        pe_i = F.interpolate(pe_t, size=end, mode="linear", align_corners=False)
        return pe_i.transpose(1, 2)[:, offset:end, :]         # [1, L, D]

    @torch.no_grad()
    def reset_contextual_memory(self) -> None:
        """
        Clear every layer's contextual memory state.
        """
        for layer in self.layers:
            layer.reset_contextual_memory()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        update_context_memory: Optional[bool] = None,
        position_offset: int | torch.Tensor = 0,
        context_memory_update: str = "all",
    ) -> torch.Tensor:
        # x: [B, L, input_dim]
        x = self.input_proj(x)  # [B, L, D]

        if self.use_pos_emb:
            L = x.size(1)
            x = x + self._get_pos(L, x.device, x.dtype, offset=position_offset)
        if mask is not None:
            x = x * mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

        should_update = (
            bool(update_context_memory)
            if update_context_memory is not None
            else bool(self.training and self.use_context_update)
        )

        for layer in self.layers:
            x = layer(x, mask=mask)
            if should_update:
                # TTM-Lite updates memory only after processing observed tokens.
                # Callers must avoid passing future target tokens when this flag
                # is enabled.
                memory_update = _summarize_context_update(x, context_memory_update)
                layer.update_contextual_memory(memory_update.detach())
        return x
