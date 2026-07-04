from typing import Optional, Dict

from torch import nn
import torch
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.value_conditioning import (
    apply_transition_loss_scope,
    build_value_input_feature,
    mask_appended_target_value,
)
from models.Titan import TitanConfig, MemoryEncoder, LMM
import torch.nn.functional as F

class TitanTPP(nn.Module):
    '''
    Titan-Based TPPs Model
    - Backbone: Titan MemoryEncoder (replaces RNN)
    - Memory: selectable memory_mode for ablation and final validation
    - Head: Standard TPPs intensity and marker prediction heads
    '''
    VALID_MEMORY_MODES = {
        "none",
        "static_lmm",
        "contextual_ttm",
        "series_lmm",
        "hybrid_lmm_ttm",
    }
    VALID_TTM_MEMORY_UPDATES = {"all", "last", "mean"}

    def __init__(self, cfg: RMTPPConfig, titan_cfg: TitanConfig):
        super().__init__()
        self.cfg = cfg
        self.titan_cfg = titan_cfg
        self.memory_mode = self._resolve_memory_mode(titan_cfg)
        self.ttm_chunk_size = self._resolve_ttm_chunk_size(titan_cfg)
        self.ttm_memory_update = self._resolve_ttm_memory_update(titan_cfg)

        self.emb = nn.Embedding(cfg.num_marks, cfg.mark_emb_dim)    # marker embedding

        self.use_value_input = str(getattr(cfg, "value_input_mode", "none")).lower() != "none"
        if self.use_value_input:
            self.value_input_proj = nn.Linear(1, int(cfg.value_input_emb_dim))

        # Input dim = Mark Embedding + Log-Time (1 dim) + optional value feature
        input_dim = cfg.mark_emb_dim + 1
        if self.use_value_input:
            input_dim += int(cfg.value_input_emb_dim)
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
            use_context_update = bool(getattr(titan_cfg, "use_context_update", False)),
            use_pos_emb = True,
            max_len = getattr(titan_cfg, 'max_len', 512),
            use_causal = titan_cfg.use_causal
        )

        # LMM (Local Memory Matching)
        # static_lmm      : use the learnable LMM memory bank
        # series_lmm      : use caller-provided per-series memory when available
        # hybrid_lmm_ttm  : combine contextual TTM with LMM retrieval
        self.use_lmm = self.memory_mode in {"static_lmm", "series_lmm", "hybrid_lmm_ttm"}
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

    @classmethod
    def _resolve_memory_mode(cls, titan_cfg: TitanConfig) -> str:
        """
        Resolve the official TitanTPP memory mode.

        Older experiment configs only had `use_lmm`; keeping this fallback lets
        legacy candidate names continue to run while new experiments can compare
        memory mechanisms explicitly through `memory_mode`.
        """
        raw_mode = getattr(titan_cfg, "memory_mode", None)
        if raw_mode is None or str(raw_mode).strip().lower() in {"", "auto"}:
            raw_mode = "static_lmm" if bool(getattr(titan_cfg, "use_lmm", False)) else "none"

        mode = str(raw_mode).strip().lower()
        aliases = {
            "no_memory": "none",
            "no_lmm": "none",
            "lmm": "static_lmm",
            "ttm": "contextual_ttm",
            "contextual": "contextual_ttm",
            "hybrid": "hybrid_lmm_ttm",
        }
        mode = aliases.get(mode, mode)
        if mode not in cls.VALID_MEMORY_MODES:
            available = ", ".join(sorted(cls.VALID_MEMORY_MODES))
            raise ValueError(f"Unsupported TitanTPP memory_mode='{raw_mode}'. Available: {available}")
        return mode

    @property
    def uses_contextual_ttm(self) -> bool:
        """
        True when the encoder is allowed to maintain online contextual memory.
        """
        return self.memory_mode in {"contextual_ttm", "hybrid_lmm_ttm"}

    @staticmethod
    def _resolve_ttm_chunk_size(titan_cfg: TitanConfig) -> int:
        """
        Resolve how many tokens are encoded before updating contextual memory.

        A value of 1 is the original exact token-wise TTM path. Larger values
        are chunked/patch-style approximations that improve GPU utilization by
        reducing repeated encoder calls.
        """
        chunk_size = int(getattr(titan_cfg, "ttm_chunk_size", 1) or 1)
        if chunk_size < 1:
            raise ValueError("ttm_chunk_size must be >= 1")
        return chunk_size

    @classmethod
    def _resolve_ttm_memory_update(cls, titan_cfg: TitanConfig) -> str:
        """
        Resolve how each processed TTM chunk is summarized into memory.
        """
        mode = str(getattr(titan_cfg, "ttm_memory_update", "all") or "all").strip().lower()
        if mode not in cls.VALID_TTM_MEMORY_UPDATES:
            available = ", ".join(sorted(cls.VALID_TTM_MEMORY_UPDATES))
            raise ValueError(f"Unsupported ttm_memory_update='{mode}'. Available: {available}")
        return mode

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

    @torch.no_grad()
    def reset_contextual_memory(self) -> None:
        """
        Reset Titan encoder contextual memory for series-wise online evaluation.
        """
        reset = getattr(self.encoder, "reset_contextual_memory", None)
        if callable(reset):
            reset()

    def forward(
        self,
        marks: torch.Tensor,
        dts: torch.Tensor,
        *,
        values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        update_context_memory: Optional[bool] = None,
        series_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        features = [emb, dt_feat]
        value_feat = build_value_input_feature(
            marks=marks,
            values=values,
            cfg=self.cfg,
            mask=mask,
        )
        if value_feat is not None:
            # Value-conditioned marked TPP: append already-observed quantity
            # state without changing the next-event prediction heads.
            features.append(self.value_input_proj(value_feat.unsqueeze(-1)))
        x = torch.cat(features, dim = -1)  # [B, L, Input_Dim]

        # 2. Titan Encoder
        # Contextual TTM is stateful, so we only update memory when the caller
        # explicitly asks for it. This avoids cross-series leakage in shuffled
        # mini-batch training.
        if self.uses_contextual_ttm and update_context_memory is None:
            # Self-contained online pass for ordinary train/validation windows.
            # The memory is reset at window boundaries and updated by chunk.
            # A chunk size of 1 reproduces exact token-wise TTM; larger chunks
            # reduce repeated encoder calls while causal attention still keeps
            # future tokens hidden inside each chunk.
            h = self._encode_window_with_contextual_ttm(x)
        else:
            should_update_context = bool(update_context_memory) and self.uses_contextual_ttm
            h = self.encoder(
                x,
                update_context_memory=should_update_context,
                context_memory_update=self.ttm_memory_update,
            ) # [B, L, D]

        # 3. LMM / retrieved memory branch
        if self.memory_mode == "static_lmm":
            h = self.lmm(h)
        elif self.memory_mode == "series_lmm":
            # Series memory is expected to be built by the runner from past
            # events of the same entity. If absent, this mode degrades safely
            # to the pure encoder path instead of using unrelated static memory.
            if series_memory is not None:
                h = self.lmm(h, memory=series_memory)
        elif self.memory_mode == "hybrid_lmm_ttm":
            # Hybrid mode combines contextual memory in the encoder with LMM.
            # When a series-specific memory is supplied, it takes precedence;
            # otherwise the learnable static LMM bank is used as a fallback.
            h = self.lmm(h, memory=series_memory)

        return h

    def _encode_window_with_contextual_ttm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode one mini-batch window with online contextual memory.

        This path approximates Titan test-time memory during normal training.
        Memory is local to the current window, updated from already-observed
        chunks, and cleared before returning so batches remain independent.
        """
        reset = getattr(self.encoder, "reset_contextual_memory", None)
        if callable(reset):
            reset()

        outputs = []
        chunk_size = min(self.ttm_chunk_size, max(int(x.size(1)), 1))
        for start in range(0, x.size(1), chunk_size):
            end = min(start + chunk_size, x.size(1))
            h_chunk = self.encoder(
                x[:, start:end, :],
                update_context_memory=True,
                position_offset=start,
                context_memory_update=self.ttm_memory_update,
            )
            outputs.append(h_chunk)

        if callable(reset):
            reset()

        if not outputs:
            d_model = int(self.encoder.input_proj.out_features)
            return x.new_empty((x.size(0), 0, d_model))
        return torch.cat(outputs, dim=1)

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
        series_memory: Optional[torch.Tensor] = None,
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
        h = self.forward(
            marks,
            dts,
            values=input_values,
            mask=mask,
            series_memory=series_memory,
        ) # [B, L, H]

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
