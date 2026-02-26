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

        # w_raw를 음수로 시작하면 softplus(w_raw)가 작게 시작 -> wd가 작아져 폭주 억제
        with torch.no_grad():
            self.w_raw.fill_(-3.0)  # softplus(-3) ~ 0.048

    def _w_pos(self) -> torch.Tensor:
        # ensure w > 0 for stability of 1/w and inverse-CDF sampling
        return F.softplus(self.w_raw) + self.cfg.w_min

    def _clamped_exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, max = self.cfg.exp_clamp))

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