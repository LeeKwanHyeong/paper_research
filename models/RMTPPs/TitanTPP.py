from typing import Optional, Dict

from torch import nn
import torch
from models.RMTPPs.config import RMTPPConfig
from models.Titan import TitanConfig, MemoryEncoder, LMM
import torch.nn.functional as F

class TitanTPP(nn.Module):
    '''
    Titan-Based TPPs Model
    - Backbone: Titan MemoryEncoder (replaces RNN)
    - Memory: Optional LMM (Local Memory Matching) for retrieving past patterns
    - Head: Standard TPPs intensity and marker prediction heads
    '''

    def __init__(self, cfg: RMTPPConfig, titan_cfg: TitanConfig):
        super().__init__()
        self.cfg = cfg
        self.titan_cfg = titan_cfg

        self.emb = nn.Embedding(cfg.num_marks, cfg.mark_emb_dim)    # marker embedding

        # Input dim = Mark Embedding + Log-Time (1 dim)
        input_dim = cfg.mark_emb_dim + 1
        d_model = titan_cfg.d_model

        # Titan Backbone
        # RMTPP는 AutoRegressive하므로, Encoder 내부 Attention에서 Causal Mask 적용 필요.
        self.encoder = MemoryEncoder(
            input_dim = input_dim,
            d_model = d_model,
            n_layers = titan_cfg.n_layers,
            n_heads = titan_cfg.n_heads,
            d_ff = titan_cfg.d_ff,
            contextual_mem_size = titan_cfg.contextual_mem_size,
            persistent_mem_size = titan_cfg.persistent_mem_size,
            dropout = titan_cfg.dropout,
            use_pos_emb = True,
            max_len = getattr(titan_cfg, 'max_len', 512),
            use_causal = titan_cfg.use_causal
        )

        # LMM (Local Memory Matchinig)
        self.use_lmm = getattr(titan_cfg, 'use_lmm', False)
        if self.use_lmm:
            self.lmm = LMM(
                d_model = titan_cfg.d_model,
                mem_size = getattr(titan_cfg, 'mem_size', 512),
                topk = getattr(titan_cfg, 'mem_topk', 8),
            )

        # Prediction Heads (RMTPPs Standards)
        # Next Marker Prediction
        self.mark_head = nn.Linear(d_model, cfg.num_marks)
        # Value head predicts the residual part of log10(qty).
        self.value_head = nn.Linear(d_model, 1)

        # Next Time Intensity Parameters
        # v_t: vector, b_t: scalar, w_t: scalr (>0 enforced by softplus)
        self.v_t = nn.Linear(d_model, 1, bias = False)  # Depends on History H
        self.b_t = nn.Parameter(torch.zeros(1))                   # Base intensity bias
        self.w_raw = nn.Parameter(torch.zeros(1))                 # Time decay parameter

        # Initialize weights
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
        Build a differentiable quantity estimate from mark probabilities.

        We intentionally avoid argmax-mark reconstruction here so direct
        quantity supervision can still update the mark head during training.
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

    def forward(self, marks: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        """
        Process input sequence through Embedding -> Titan Encoder -> LMM
        :returns
            h: [B, L, d_model] (Context vector for each step)
        """
        # marks: [B, L], dts: [B, L]

        # 1. Embeddings
        emb = self.emb(marks)                           # [B, L, E]
        # dt_feat = dts.unsqueeze(-1).float()             # [B, L, 1]
        dt_feat = torch.log1p(dts.clamp_min(0).float()).unsqueeze(-1)
        x = torch.cat([emb, dt_feat], dim = -1)  # [B, L, Input_Dim]

        # 2. Titan Encoder
        # Ensure 'MemoryEncoder' or 'MemoryAttention' applies causal masking
        h = self.encoder(x) # [B, L, D]

        # 3. LMM (Long-term Pattern Retrieval)
        if self.use_lmm:
            h = self.lmm(h)

        return h

    def log_f_dt(self, h_j: torch.Tensor, dt_next: torch.Tensor) -> torch.Tensor:
        '''
        Calculate log density of the next inter-event time.
        log f(d) = log(lambda(d)) - integral_0^d lambda(u) du
        '''
        w = self._w_pos()
        # h_j: [B, L-1, D] -> v_t -> [B, L-1, 1] -> squeeze -> [B, L-1]
        a = self.v_t(h_j).squeeze(-1) + self.b_t

        # Numerical stability
        a_c = torch.clamp(a, max=self.cfg.exp_clamp)
        exp_a = torch.exp(a_c)

        wd = w * dt_next
        wd_c = torch.clamp(wd, max=getattr(self.cfg, "wd_clamp", 10.0))
        expm1_wd = torch.expm1(wd_c)  # exp(wd) - 1

        # log f(d) calculation
        log_lambda = a_c + wd_c
        log_f = log_lambda - (exp_a / w) * expm1_wd
        return log_f

    def nll(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        values: Optional[torch.Tensor] = None,
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
            mask = torch.ones((B, L), device = marks.device, dtype = torch.bool)

        h = self.forward(marks, dts) # [B, L, H]

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

        # Residual regression for quantity reconstruction.
        if values is not None and self.cfg.use_value_head:
            value_next = values[:, 1:].float()
            value_hat = self.predict_value(h_j)
            # value_sq = F.smooth_l1_loss(value_hat, value_next, reduction="none")
            value_sq = F.huber_loss(value_hat, value_next, reduction="none")
            value_sq = value_sq * step_mask
            value_loss = value_sq.sum() / step_mask.sum().clamp_min(1)

            # Direct quantity supervision is kept optional so TitanTPP can
            # stay on the legacy residual-only objective for the paper A/B,
            # while still allowing development-time experiments on qty loss.
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

        loss_mode = getattr(self.cfg, "loss_mode", "residual_only")
        if loss_mode == "residual_only":
            total_loss = nll_total + value_loss
        elif loss_mode == "hybrid":
            total_loss = nll_total + value_loss + getattr(self.cfg, "lambda_qty", 0.25) * qty_loss
        elif loss_mode == "qty_only":
            total_loss = nll_total + getattr(self.cfg, "lambda_qty", 0.25) * qty_loss
        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

        return {
            'nll': nll_total,
            'nll_marker': nll_marker,
            'nll_time': nll_time,
            'value_loss': value_loss,
            'qty_loss': qty_loss,
            'total_loss': total_loss,
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
