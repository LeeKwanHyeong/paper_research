"""AttNHP matched-head variant; pinned author kernels and licenses are in phen/vendor/."""

import math

import torch
from torch import nn

from phen.models.heads import NextEventModel
from phen.vendor.attnhp.xfmr import EncoderLayer, MultiHeadAttention


class AttNHP(NextEventModel):
    """Author query/context two-stream attention, queried at observed t_i+.

    One attention head, width64, two layers, time dimension16; no FFN or
    attention residual, original tanh residual on the query stream.
    """
    def __init__(self, hidden_dim, train_log_mean, **kwargs):
        super().__init__(hidden_dim, train_log_mean, **kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.d_time = 16
        self.register_buffer("div_term", torch.exp(torch.arange(0, self.d_time, 2)
                             * -(math.log(10000.) / self.d_time)).reshape(1, 1, -1))
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim + self.d_time,
                         MultiHeadAttention(1, hidden_dim + self.d_time, hidden_dim,
                                            dropout=.1, output_linear=False),
                         use_residual=False, dropout=.1) for _ in range(2)
        ])

    def temporal(self, times):
        phases = times.unsqueeze(-1) * self.div_term
        return torch.stack((phases.sin(), phases.cos()), -1).flatten(-2)

    def encode(self, dts, history_quantities, mask):
        gaps = torch.where(mask, dts.float().clamp_min(0), 0.)
        qty = torch.where(mask, history_quantities, 0.)
        nonpad = mask.unsqueeze(-1)
        temporal = self.temporal(gaps.cumsum(1)) * nonpad
        x = torch.tanh(self.input_projection(self.continuous_features(gaps, qty, mask))) * nonpad
        query = torch.zeros_like(x)
        b, n = mask.shape
        future = torch.ones(n, n, device=mask.device, dtype=torch.bool).triu(1)
        history_mask = future[None].expand(b, -1, -1) | ~mask[:, None, :]
        # Padded context rows have a private dummy self-key and emit zero;
        # no valid query can ever attend to that padded key.
        context_mask = history_mask.clone()
        idx = torch.arange(n, device=mask.device)
        context_mask[:, idx, idx] = False
        only_self = ~torch.eye(n, device=mask.device, dtype=torch.bool)[None].expand(b, -1, -1)
        combined_mask = torch.cat((torch.cat((context_mask, torch.ones_like(only_self)), -1),
                                   torch.cat((history_mask, only_self), -1)), 1)
        enc_input = torch.cat((x, temporal), -1)
        for layer in self.layers:
            z = layer(torch.cat((enc_input, torch.cat((query, temporal), -1)), 1), combined_mask)
            query = (query + torch.tanh(z[:, n:] * nonpad)) * nonpad
            enc_input = torch.cat((z[:, :n] * nonpad, temporal), -1)
        return query
