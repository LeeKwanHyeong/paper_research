"""Self-attentive Hawkes encoder for the adapted count-aware SAHP control."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import SharedTimeCountModel


class SAHPEncoderBlock(nn.Module):
    """Causal self-attention and feed-forward block used by adapted SAHP."""

    def __init__(self, hidden_dim: int, *, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=~mask,
            need_weights=False,
        )
        x = self.attention_norm(x + self.attention_dropout(attended))
        x = self.feed_forward_norm(
            x + self.feed_forward_dropout(self.feed_forward(x))
        )
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)


class CountAwareSAHP(SharedTimeCountModel):
    """Causal self-attentive encoder under shared time/count heads.

    This is the adapted SAHP backbone used by the controlled count-aware
    experiment. It retains SAHP-style temporal encoding and continuous decay
    while using the experiment's common time and quantity objectives.
    """

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        **quantity_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        if hidden_dim % 4 != 0:
            raise ValueError("SAHP hidden_dim must be divisible by four")
        self.input_projection = nn.Linear(2, hidden_dim)
        self.input_dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [
                SAHPEncoderBlock(hidden_dim, n_heads=4, dropout=0.1)
                for _ in range(2)
            ]
        )
        self.base_projection = nn.Linear(hidden_dim, hidden_dim)
        self.event_projection = nn.Linear(hidden_dim, hidden_dim)
        self.decay_projection = nn.Linear(hidden_dim, hidden_dim)

    def temporal_encoding(self, dts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=dts.device, dtype=torch.float32)
            * (-math.log(10000.0) / denominator)
        )
        event_times = torch.cumsum(dts.float().clamp_min(0.0), dim=1)
        phase = event_times.unsqueeze(-1) * frequencies
        encoding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return encoding * mask.unsqueeze(-1).to(dtype=encoding.dtype)

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.continuous_features(dts, history_quantities, mask)
        x = self.input_projection(features) + self.temporal_encoding(dts, mask)
        x = self.input_dropout(x) * mask.unsqueeze(-1).to(dtype=x.dtype)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, mask=mask, causal_mask=causal_mask)

        base_state = torch.tanh(self.base_projection(x))
        event_state = torch.tanh(self.event_projection(x))
        decay_rate = F.softplus(self.decay_projection(x)) + 1e-4
        elapsed = dts.to(x.dtype).clamp_min(0.0).unsqueeze(-1)
        encoded = base_state + (event_state - base_state) * torch.exp(
            -decay_rate * elapsed
        )
        return encoded * mask.unsqueeze(-1).to(dtype=encoded.dtype)


__all__ = ["CountAwareSAHP", "SAHPEncoderBlock"]
