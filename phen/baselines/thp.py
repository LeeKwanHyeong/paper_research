"""The existing two-block, post-norm THP variant with common event heads."""

import torch
from torch import nn
from torch.nn import functional as F

from phen.models.heads import NextEventModel


class THPBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.n_heads = 4
        self.self_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)
        self.ffn_1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.ffn_2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, non_pad_mask, blocked_mask):
        batch, length, _ = x.shape
        blocked_mask = blocked_mask.repeat_interleave(self.n_heads, dim=0).view(
            batch * self.n_heads, length, length)
        attended, _ = self.self_attn(x, x, x, attn_mask=blocked_mask, need_weights=False)
        x = self.norm_attn(x + self.dropout(attended)) * non_pad_mask
        feedforward = self.ffn_2(self.dropout(F.gelu(self.ffn_1(x))))
        return self.norm_ffn(x + self.dropout(feedforward)) * non_pad_mask


class THP(NextEventModel):
    def __init__(self, hidden_dim, train_log_mean, **head_kwargs):
        super().__init__(hidden_dim, train_log_mean, **head_kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([THPBlock(hidden_dim) for _ in range(2)])

    @staticmethod
    def blocked_attention_mask(mask):
        batch, length = mask.shape
        future = torch.ones(length, length, device=mask.device, dtype=torch.bool).triu(1).unsqueeze(0)
        blocked = future | (~mask).unsqueeze(1).expand(batch, length, length)
        positions = torch.arange(length, device=mask.device)
        blocked[:, positions, positions] = False
        return blocked

    def encode(self, dts, history_quantities, mask):
        x = self.input_projection(self.continuous_features(dts, history_quantities, mask))
        non_pad = mask.unsqueeze(-1).to(x.dtype)
        blocked = self.blocked_attention_mask(mask)
        for layer in self.layers:
            x = layer(x, non_pad, blocked)
        return x * non_pad
