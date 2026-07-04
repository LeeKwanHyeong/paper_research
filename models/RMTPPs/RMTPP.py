from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.value_conditioning import (
    apply_transition_loss_scope,
    build_value_input_feature,
    mask_appended_target_value,
)


class RMTPP(nn.Module):
    """
    Recurrent Marked Temporal Point Process (RMTPPs)
    - Marker head: softmax P(y_{j+1} | h_j) (paper equation. 10)
    - Time head: intensity lamgda(t) = exp(v^T h_j + w * (t-t_j) + b)   (paper equation. 11)
        => duration density f(d | h_j) closed-form

    Input Sequence:
        marks: [B, L] (0..k-1)
        dts:   [B, L] (d_j = t_j - t_{j-1})

    Traget
        next_marks: [B, L-1] (y_{j+1})
        next_dts:   [B, L-1] (d_{j+1})
    """
    def __init__(self, cfg: RMTPPConfig):
        super().__init__()

        self.cfg = cfg

        self.emb = nn.Embedding(cfg.num_marks, cfg.mark_emb_dim) # marker embedding

        self.use_value_input = str(getattr(cfg, "value_input_mode", "none")).lower() != "none"
        if self.use_value_input:
            self.value_input_proj = nn.Linear(1, int(cfg.value_input_emb_dim))

        rnn_in_dim = cfg.mark_emb_dim + 1
        if self.use_value_input:
            rnn_in_dim += int(cfg.value_input_emb_dim)
        if cfg.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size = rnn_in_dim,
                hidden_size = cfg.rnn_hidden_dim,
                batch_first = True,
                dropout = cfg.dropout if cfg.dropout > 0.0 else 0.0
            )

        elif cfg.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size = rnn_in_dim,
                hidden_size = cfg.rnn_hidden_dim,
                batch_first = True,
                dropout = cfg.dropout if cfg.dropout > 0.0 else 0.0
            )

        elif cfg.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size = rnn_in_dim,
                hidden_size = cfg.rnn_hidden_dim,
                batch_first = True,
                dropout = cfg.dropout if cfg.dropout > 0.0 else 0.0
            )

        else:
            raise ValueError(f"Unsupported rnn_type: {cfg.rnn_type}")

        # Marker generation head: logits for next mark
        self.mark_head = nn.Linear(cfg.rnn_hidden_dim, cfg.num_marks)
        # Value head predicts the residual part of log10(qty).
        self.value_head = nn.Linear(cfg.rnn_hidden_dim, 1)

        # Time intensity parameters (paper equation. 10)
        # v_t: vector, b_t: scalar, w_t: scalr (>0 enforced by softplus)
        self.v_t = nn.Linear(cfg.rnn_hidden_dim, 1, bias = False) # v*T h
        self.b_t = nn.Parameter(torch.zeros(1))
        self.w_raw = nn.Parameter(torch.zeros(1))

        self._init_stable()

    def _init_stable(self):
        # v_t, mark_head를 너무 크게 시작하지 않게
        nn.init.normal_(self.v_t.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mark_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mark_head.bias)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_head.bias)

        # w_raw를 음수로 시작하면 softplus(w_raw)가 작게 시작 -> wd가 작아져 폭주 억제
        with torch.no_grad():
            self.w_raw.fill_(-3.0)  # softplus(-3) ~ 0.048

    def _w_pos(self) -> torch.Tensor:
        # ensure w > 0 for stability of 1/w and inverse-CDF sampling
        return F.softplus(self.w_raw) + self.cfg.w_min

    def _clamped_exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, max = self.cfg.exp_clamp))

    def predict_value(self, h_j: torch.Tensor) -> torch.Tensor:
        """
        Predict the residual component of log_base(qty).
        Sigmoid is the default because the intended target is in [0, 1).
        """
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

    def expected_qty_from_logits(
        self,
        logits_no_pad: torch.Tensor,
        value_residual: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable expected quantity used by optional hybrid qty loss.
        """
        mark_probs = torch.softmax(logits_no_pad, dim=-1)
        real_mark_count = logits_no_pad.size(-1)
        mark_grid = torch.arange(
            real_mark_count,
            device=logits_no_pad.device,
            dtype=value_residual.dtype,
        ).view(1, 1, real_mark_count)
        log_qty = mark_grid + value_residual.unsqueeze(-1)
        scale_base = torch.as_tensor(
            self.cfg.scale_base,
            dtype=value_residual.dtype,
            device=value_residual.device,
        )
        qty_per_mark = torch.pow(scale_base, log_qty)
        return (mark_probs * qty_per_mark).sum(dim=-1)

    def forward(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        *,
        values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :returns [B, L, H]
        """
        # marks: [B, L], dts: [B, L]
        emb = self.emb(marks)                           # [B, L, E]
        dt_feat = dts.unsqueeze(-1).float()             # [B, L, 1]
        features = [emb, dt_feat]
        value_feat = build_value_input_feature(
            marks=marks,
            values=values,
            cfg=self.cfg,
            mask=mask,
        )
        if value_feat is not None:
            # Value-conditioned ablation: only observed history values should
            # reach this point. nll/eval callers mask the appended target value.
            features.append(self.value_input_proj(value_feat.unsqueeze(-1)))
        x = torch.cat(features, dim = -1)  # [B, L, E + 1 (+ value feature)]

        out, _ = self.rnn(x)                            # [B, L, H]
        return out

    def log_f_dt(self, h_j: torch.Tensor, dt_next: torch.Tensor) -> torch.Tensor:
        w = self._w_pos()  # scalar
        a = self.v_t(h_j).squeeze(-1) + self.b_t  # [B, L-1]

        # clamp a to prevent exp overflow
        a_c = torch.clamp(a, max=self.cfg.exp_clamp)
        exp_a = torch.exp(a_c)

        wd = w * dt_next
        wd_c = torch.clamp(wd, max=getattr(self.cfg, "wd_clamp", 10.0))
        expm1_wd = torch.expm1(wd_c)  # exp(wd)-1 안정 계산

        # log f(d) = (a + wd) - (exp(a)/w) * (exp(wd)-1)
        log_lambda = a_c + wd_c
        log_f = log_lambda - (exp_a / w) * expm1_wd
        return log_f

    def log_S_dt(self, h_j: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        log S(dt | h_j)
        S(d) = exp(-(exp(a)/w) * (exp(w d)-1))
        => logS(d) = - (exp(a)/w) * (exp(w d)-1)
        """
        w = self._w_pos()
        a = self.v_t(h_j).squeeze(-1) + self.b_t

        a_c = torch.clamp(a, max=self.cfg.exp_clamp)
        exp_a = torch.exp(a_c)

        wd = w * dt
        wd_c = torch.clamp(wd, max=getattr(self.cfg, "wd_clamp", 10.0))
        expm1_wd = torch.expm1(wd_c)

        return - (exp_a / w) * expm1_wd

    def nll(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        loss_scope: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute negative log-likelihood for a batch.

        marks:  [B, L]
        dts:    [B, L]
        mask    [B, L] (True for valid positions). If None -> all valid.

        steps j=0..L-2:
            - marker loss uses y_{j+1} predicted from h_j
            - time loss uses d_{j+1} predicted from h_j
        """
        B, L = marks.shape
        if mask is None:
            mask = torch.ones((B, L), device = marks.device, dtype = torch.bool)

        input_values = mask_appended_target_value(values, mask)
        h = self.forward(marks, dts, values=input_values, mask=mask) # [B, L, H]

        h_j = h[:, :-1, :]                      # [B, L-1, H]
        y_next = marks[:, 1:]                   # [B, L-1]
        dt_next = dts[:, 1:].float()            # [B, L-1]
        step_mask = mask[:, 1:] & mask[:, :-1]  # exist target & exist context
        step_mask = apply_transition_loss_scope(
            step_mask,
            loss_scope or getattr(self.cfg, "train_loss_scope", "all"),
        )

        # Marker log-prob (paper equation 10.)
        logits = self.mark_head(h_j)            # [B, L-1, K]
        log_y = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_next.reshape(-1),
            reduction = 'none'
        ).reshape(B, L - 1)                     # [B, L-1] (log P)

        # Time log-density
        logf_dt = self.log_f_dt(h_j, dt_next)   # [B, L-1]

        # Value regression is evaluated on the same autoregressive steps.
        if values is not None and self.cfg.use_value_head:
            value_next = values[:, 1:].float()
            value_hat = self.predict_value(h_j)
            # value_sq = F.smooth_l1_loss(value_hat, value_next, reduction="none")
            value_sq = F.huber_loss(value_hat, value_next, reduction="none")
            value_sq = value_sq * step_mask
            value_loss = value_sq.sum() / step_mask.sum().clamp_min(1)

            pad_id = int(self.cfg.num_marks - 1)
            logits_real = logits[..., :pad_id]
            expected_qty = self.expected_qty_from_logits(logits_real, value_hat)
            true_qty = self.reconstruct_qty(y_next.clamp_max(pad_id - 1), value_next)

            qty_scale_value = torch.as_tensor(
                float(max(getattr(self.cfg, "qty_scale_value", 1.0), 1.0)),
                device=true_qty.device,
                dtype=true_qty.dtype,
            )
            qty_sq = F.huber_loss(
                expected_qty / qty_scale_value,
                true_qty / qty_scale_value,
                reduction="none",
            )
            qty_sq = qty_sq * step_mask
            qty_loss = qty_sq.sum() / step_mask.sum().clamp_min(1)
        else:
            value_hat = None
            value_loss = torch.zeros((), device=marks.device, dtype=torch.float32)
            qty_loss = torch.zeros((), device=marks.device, dtype=torch.float32)

        # apply mask
        logp_y = log_y * step_mask
        logf_dt = logf_dt * step_mask

        # negative log-likelihood
        nll_marker = -logp_y.sum() / (step_mask.sum().clamp_min(1))
        nll_time = -logf_dt.sum() / (step_mask.sum().clamp_min(1))
        nll_total = nll_marker + nll_time

        return {
            'nll': nll_total,
            'nll_marker': nll_marker,
            'nll_time': nll_time,
            'value_loss': value_loss,
            'qty_loss': qty_loss,
            'value_hat': value_hat,
            'steps': step_mask.sum(),
        }

    @torch.no_grad()
    def sample_next_dt(self, h_j: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inverse-CDF sampling for dt when lambda(u) = exp(a + w u).
        Survival: S(d) = exp(-(exp(a)/w) * (exp(w d)-1))
        Sample Uniform(0, 1):
            exp(w d) = 1 + (w/exp(a)) * (-log U)
            d = (1/w) log (1 + (w/exp(a)) * (-log U))
        """
        w = self._w_pos()
        a = self.v_t(h_j).squeeze(-1) + self.b_t

        if u is None:
            u = torch.rand_like(a).clamp_min(self.cfg.eps)

        exp_a = self._clamped_exp(a).clamp_min(self.cfg.eps)
        x = 1.0 + (w/exp_a) * (-torch.log(u))
        dt = (1.0 / w) * torch.log(x.clamp_min(self.cfg.eps))
        return dt
