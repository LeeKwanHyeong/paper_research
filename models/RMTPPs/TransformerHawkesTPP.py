"""
Transformer Hawkes Process baseline adapted for this project.

Source reference:
    SimiaoZuo/Transformer-Hawkes-Process
    https://github.com/SimiaoZuo/Transformer-Hawkes-Process

The official THP implementation uses:
1. event-type embeddings
2. sinusoidal temporal encoding from event timestamps
3. causal Transformer self-attention

This file keeps that encoder design, but exposes the same project-level
interface as RMTPP/TitanTPP:
    - forward(marks, dts) -> hidden states
    - nll(...)
    - mark_head, value_head, log_f_dt, sample_next_dt

That adapter choice is intentional. It lets us compare encoder families under
the same magnitude-factorized decoder and validation metrics instead of mixing
different quantity reconstruction objectives.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RMTPPs.config import RMTPPConfig, THPConfig


class THPEncoderLayer(nn.Module):
    """
    One causal Transformer layer following the official THP block structure.

    The official code uses a custom MultiHeadAttention + PositionwiseFeedForward
    pair. Here we use PyTorch's MultiheadAttention for maintainability while
    preserving the same residual, layernorm, causal-mask, and FFN behavior.
    """

    def __init__(self, cfg: THPConfig):
        super().__init__()
        self.normalize_before = bool(cfg.normalize_before)
        self.n_heads = int(cfg.n_heads)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn_1 = nn.Linear(cfg.d_model, cfg.d_inner)
        self.ffn_2 = nn.Linear(cfg.d_inner, cfg.d_model)
        self.norm_attn = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(cfg.d_model, eps=1e-6)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        non_pad_mask: torch.Tensor,
        blocked_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        attn_input = self.norm_attn(x) if self.normalize_before else x
        batch_size, seq_len, _ = x.shape
        # MultiheadAttention returns NaN if a query has every key masked. This
        # can happen for left-padded query positions, so the encoder-level mask
        # leaves each diagonal open and we expand it per attention head here.
        attn_mask = blocked_mask.repeat_interleave(self.n_heads, dim=0).view(
            batch_size * self.n_heads,
            seq_len,
            seq_len,
        )
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attn_output)
        if not self.normalize_before:
            x = self.norm_attn(x)
        x = x * non_pad_mask

        residual = x
        ffn_input = self.norm_ffn(x) if self.normalize_before else x
        ffn_output = self.ffn_2(self.dropout(F.gelu(self.ffn_1(ffn_input))))
        x = residual + self.dropout(ffn_output)
        if not self.normalize_before:
            x = self.norm_ffn(x)
        return x * non_pad_mask


class THPTemporalEncoder(nn.Module):
    """
    Official-THP-style temporal encoder with causal self-attention.

    Official THP receives absolute timestamps. Our loaders provide inter-event
    times, so we reconstruct absolute event time inside forward via cumulative
    sum. Left-padded positions stay at zero and are removed by `mask`.
    """

    def __init__(self, rmtpp_cfg: RMTPPConfig, thp_cfg: THPConfig):
        super().__init__()
        self.rmtpp_cfg = rmtpp_cfg
        self.thp_cfg = thp_cfg
        self.pad_id = int(rmtpp_cfg.num_marks - 1)

        self.event_emb = nn.Embedding(
            rmtpp_cfg.num_marks,
            thp_cfg.d_model,
            padding_idx=self.pad_id,
        )
        position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / thp_cfg.d_model) for i in range(thp_cfg.d_model)],
            dtype=torch.float32,
        )
        self.register_buffer("position_vec", position_vec, persistent=False)
        self.layer_stack = nn.ModuleList([THPEncoderLayer(thp_cfg) for _ in range(thp_cfg.n_layers)])

        self.use_rnn = bool(thp_cfg.use_rnn)
        if self.use_rnn:
            # The official repository includes an optional recurrent layer after
            # the Transformer. We keep it opt-in because this project's batches
            # are left-padded, while the official code assumes sequence lengths
            # can be packed from the first position.
            self.rnn = nn.LSTM(thp_cfg.d_model, thp_cfg.d_rnn, num_layers=1, batch_first=True)
            self.rnn_projection = nn.Linear(thp_cfg.d_rnn, thp_cfg.d_model)

    def _temporal_encoding(self, event_time: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        angles = event_time.unsqueeze(-1) / self.position_vec.view(1, 1, -1)
        encoding = torch.zeros_like(angles)
        encoding[..., 0::2] = torch.sin(angles[..., 0::2])
        encoding[..., 1::2] = torch.cos(angles[..., 1::2])
        return encoding * mask.unsqueeze(-1).float()

    @staticmethod
    def _blocked_attention_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        Build a safe causal + padding mask.

        True means "do not attend". Diagonal entries stay open even for padded
        queries to prevent all-masked attention rows and the resulting NaNs.
        """
        batch_size, seq_len = mask.shape
        future_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=mask.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        key_pad_mask = (~mask).unsqueeze(1).expand(batch_size, seq_len, seq_len)
        blocked = future_mask | key_pad_mask
        idx = torch.arange(seq_len, device=mask.device)
        blocked[:, idx, idx] = False
        return blocked

    def forward(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = marks.ne(self.pad_id)

        clean_dts = torch.where(mask, dts.float().clamp_min(0.0), torch.zeros_like(dts.float()))
        event_time = torch.cumsum(clean_dts, dim=1)
        non_pad_mask = mask.unsqueeze(-1).float()
        temporal = self._temporal_encoding(event_time, mask)

        x = self.event_emb(marks.clamp(min=0, max=self.pad_id))
        blocked_mask = self._blocked_attention_mask(mask)

        for layer in self.layer_stack:
            if self.thp_cfg.add_temporal_encoding_each_layer:
                x = x + temporal
            x = layer(
                x,
                non_pad_mask=non_pad_mask,
                blocked_mask=blocked_mask,
            )

        if self.use_rnn:
            x, _ = self.rnn(x)
            x = self.rnn_projection(x)
            x = x * non_pad_mask

        return x


class TransformerHawkesTPP(nn.Module):
    """
    THP encoder + project-standard marked TPP decoder.

    The decoder intentionally mirrors RMTPP/TitanTPP so experimental differences
    isolate the history encoder as much as possible.
    """

    def __init__(self, cfg: RMTPPConfig, thp_cfg: THPConfig):
        super().__init__()
        self.cfg = cfg
        self.thp_cfg = thp_cfg
        self.encoder = THPTemporalEncoder(cfg, thp_cfg)

        d_model = thp_cfg.d_model
        self.mark_head = nn.Linear(d_model, cfg.num_marks)
        self.value_head = nn.Linear(d_model, 1)
        self.v_t = nn.Linear(d_model, 1, bias=False)
        self.b_t = nn.Parameter(torch.zeros(1))
        self.w_raw = nn.Parameter(torch.zeros(1))

        self._init_stable()

    def _init_stable(self) -> None:
        nn.init.normal_(self.v_t.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mark_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mark_head.bias)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_head.bias)
        with torch.no_grad():
            self.w_raw.fill_(-3.0)

    def _w_pos(self) -> torch.Tensor:
        return F.softplus(self.w_raw) + self.cfg.w_min

    def _clamped_exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, max=self.cfg.exp_clamp))

    def forward(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(marks, dts, mask=mask)

    def predict_value(self, h_j: torch.Tensor) -> torch.Tensor:
        value_raw = self.value_head(h_j).squeeze(-1)
        if self.cfg.value_head_activation == "sigmoid":
            return torch.sigmoid(value_raw)
        return value_raw

    def reconstruct_log_qty(self, mark: torch.Tensor, value_residual: torch.Tensor) -> torch.Tensor:
        return mark.float() + value_residual.float()

    def reconstruct_qty(self, mark: torch.Tensor, value_residual: torch.Tensor) -> torch.Tensor:
        log_qty = self.reconstruct_log_qty(mark, value_residual)
        scale_base = torch.as_tensor(
            self.cfg.scale_base,
            dtype=log_qty.dtype,
            device=log_qty.device,
        )
        return torch.pow(scale_base, log_qty)

    def log_f_dt(self, h_j: torch.Tensor, dt_next: torch.Tensor) -> torch.Tensor:
        w = self._w_pos()
        a = self.v_t(h_j).squeeze(-1) + self.b_t

        a_c = torch.clamp(a, max=self.cfg.exp_clamp)
        exp_a = torch.exp(a_c)
        wd = w * dt_next
        wd_c = torch.clamp(wd, max=getattr(self.cfg, "wd_clamp", 10.0))
        expm1_wd = torch.expm1(wd_c)

        log_lambda = a_c + wd_c
        return log_lambda - (exp_a / w) * expm1_wd

    def nll(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = marks.shape
        if mask is None:
            mask = marks.ne(int(self.cfg.num_marks - 1))

        h = self.forward(marks, dts, mask=mask)
        h_j = h[:, :-1, :]
        y_next = marks[:, 1:]
        dt_next = dts[:, 1:].float()
        step_mask = mask[:, 1:] & mask[:, :-1]

        logits = self.mark_head(h_j)
        log_y = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_next.reshape(-1),
            reduction="none",
        ).reshape(B, L - 1)
        logf_dt = self.log_f_dt(h_j, dt_next)

        if values is not None and self.cfg.use_value_head:
            value_next = values[:, 1:].float()
            value_hat = self.predict_value(h_j)
            value_sq = F.huber_loss(value_hat, value_next, reduction="none")
            value_loss = (value_sq * step_mask).sum() / step_mask.sum().clamp_min(1)
        else:
            value_hat = None
            value_loss = torch.zeros((), device=marks.device, dtype=torch.float32)

        logp_y = log_y * step_mask
        logf_dt = logf_dt * step_mask
        nll_marker = -logp_y.sum() / step_mask.sum().clamp_min(1)
        nll_time = -logf_dt.sum() / step_mask.sum().clamp_min(1)
        nll_total = nll_marker + nll_time

        return {
            "nll": nll_total,
            "nll_marker": nll_marker,
            "nll_time": nll_time,
            "value_loss": value_loss,
            "value_hat": value_hat,
            "steps": step_mask.sum(),
        }

    @torch.no_grad()
    def sample_next_dt(self, h_j: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self._w_pos()
        a = self.v_t(h_j).squeeze(-1) + self.b_t

        if u is None:
            u = torch.rand_like(a).clamp_min(self.cfg.eps)

        exp_a = self._clamped_exp(a).clamp_min(self.cfg.eps)
        x = 1.0 + (w / exp_a) * (-torch.log(u))
        return (1.0 / w) * torch.log(x.clamp_min(self.cfg.eps))
