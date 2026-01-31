from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.recurrent_marked_temporal_point_process.config import RMTPPConfig


class RMTPP(nn.Module):
    """
    Recurrent Marked Temporal Point Process (RMTPP)
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

        rnn_in_dim = cfg.mark_emb_dim + 1
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

        # Time intensity parameters (paper equation. 10)
        # v_t: vector, b_t: scalar, w_t: scalr (>0 enforced by softplus)
        self.v_t = nn.Linear(cfg.rnn_hidden_dim, 1, bias = False) # v*T h
        self.b_t = nn.Parameter(torch.zeros(1))
        self.w_raw = nn.Parameter(torch.zeros(1))

    def _w_pos(self) -> torch.Tensor:
        # ensure w > 0 for stability of 1/w and inverse-CDF sampling
        return F.softplus(self.w_raw) + self.cfg.w_min

    def _clamped_exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, max = self.cfg.exp_clamp))

    def forward_hidden(self, marks: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        """
        :returns [B, L, H]
        """
        # marks: [B, L], dts: [B, L]
        emb = self.emb(marks)                           # [B, L, E]
        dt_feat = dts.unsqueeze(-1).float()             # [B, L, 1]
        x = torch.cat([emb, dt_feat], dim = -1)  # [B, L, E + 1]

        out, _ = self.rnn(x)                            # [B, L, H]
        return out

    def log_f_dt(self, h_j: torch.Tensor, dt_next: torch.Tensor) -> torch.Tensor:
        """
        log f(d_{j+1} | h_j) using closed-form  # 논문 수식 (12)

        Let:
            a = v^T*h_j + b
            w = positive scalar
            d = (t - t_{j})
            log f(d) = a + w d + (1/w)exp(a) - (1/w)exp(a+w d)
        """
        w = self._w_pos()   # scalr
        a = self.v_t(h_j).squeeze(-1) + self.b_t    # [B, ...] + scalar => [B, ...]

        wd = w * dt_next
        term1 = a + wd
        exp_a = self._clamped_exp(a)
        exp_a_wd = self._clamped_exp(a + wd)

        log_f = term1 + (exp_a / w) - (exp_a_wd / w)

        return log_f

    def nll(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
            mask = torch.ones_like((B, L), device = marks.device, dtype = torch.bool)

        h = self.forward_hidden(marks, dts) # [B, L, H]

        h_j = h[:, :-1, :]                      # [B, L-1, H]
        y_next = marks[:, 1:]                   # [B, L-1]
        dt_next = dts[:, 1:].float()            # [B, L-1]
        step_mask = mask[:, 1:] & mask[:, :-1]  # exist target & exist context

        # Marker log-prob (paper equation 10.)
        logits = self.mark_head(h_j)            # [B, L-1, K]
        log_y = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_next.reshape(-1),
            reduction = 'none'
        ).reshape(B, L - 1)                     # [B, L-1] (log P)

        # Time log-density
        logf_dt = self.log_f_dt(h_j, dt_next)   # [B, L-1]

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

