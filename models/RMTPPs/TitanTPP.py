import math
from typing import Optional, Dict

from torch import nn
import torch
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.marker_losses import masked_normalized_ranked_probability_score
from models.RMTPPs.magnitude_normalization import (
    MagnitudeContext,
    build_causal_revin_magnitude_context,
    build_causal_shrinkage_revin_magnitude_context,
    build_global_magnitude_context,
    build_raw_global_magnitude_context,
    denormalize_magnitude,
    normalized_magnitude_target,
    reconstruct_log2_quantity,
    reconstruct_raw_quantity,
    safe_exp2,
)
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
    VALID_VALUE_HEAD_MODES = {"shared", "mark_conditioned_experts"}
    VALID_QTY_MARK_GRADIENT_MODES = {"coupled", "detached"}
    VALID_VALUE_ENCODER_GRADIENT_MODES = {"coupled", "detached"}
    VALID_MARKER_LOSS_MODES = {"ce", "ce_rps"}
    VALID_QTY_DECODER_MODES = {"mark_residual", "direct_log_qty", "direct_raw_qty"}
    VALID_MAGNITUDE_NORM_MODES = {
        "global",
        "causal_revin",
        "causal_shrinkage_revin",
    }

    def __init__(self, cfg: RMTPPConfig, titan_cfg: TitanConfig):
        super().__init__()
        self.cfg = cfg
        self.titan_cfg = titan_cfg
        self.memory_mode = self._resolve_memory_mode(titan_cfg)
        self.ttm_chunk_size = self._resolve_ttm_chunk_size(titan_cfg)
        self.ttm_memory_update = self._resolve_ttm_memory_update(titan_cfg)

        self.emb = nn.Embedding(cfg.num_marks, cfg.mark_emb_dim)    # marker embedding
        self.num_real_marks = int(cfg.num_marks - 1)
        if self.num_real_marks < 1:
            raise ValueError("TitanTPP requires at least one real mark plus PAD.")
        self.qty_decoder_mode = self._resolve_qty_decoder_mode(cfg)
        self.magnitude_norm_mode = self._resolve_magnitude_norm_mode(cfg)
        self.use_direct_magnitude = self.qty_decoder_mode in {
            "direct_log_qty",
            "direct_raw_qty",
        }
        self.use_direct_raw_quantity = self.qty_decoder_mode == "direct_raw_qty"

        self.use_value_input = str(getattr(cfg, "value_input_mode", "none")).lower() != "none"
        if self.use_direct_magnitude:
            self.magnitude_input_proj = nn.Linear(1, int(cfg.magnitude_input_emb_dim))
        elif self.use_value_input:
            self.value_input_proj = nn.Linear(1, int(cfg.value_input_emb_dim))

        # Input dim = Mark Embedding + Log-Time (1 dim) + optional value feature
        input_dim = cfg.mark_emb_dim + 1
        if self.use_direct_magnitude:
            input_dim += int(cfg.magnitude_input_emb_dim)
        elif self.use_value_input:
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
        self.value_head_mode = self._resolve_value_head_mode(cfg)
        self.qty_mark_gradient_mode = self._resolve_qty_mark_gradient_mode(cfg)
        self.value_encoder_gradient_mode = self._resolve_value_encoder_gradient_mode(cfg)
        self.marker_loss_mode = self._resolve_marker_loss_mode(cfg)
        self.lambda_ordinal = float(getattr(cfg, "lambda_ordinal", 0.0))
        self._validate_gradient_mode_combination()
        self._validate_marker_loss_combination()
        self._validate_quantity_decoder_combination()
        if self.use_direct_magnitude:
            self.magnitude_head = nn.Linear(d_model, 1)
        else:
            # Legacy head predicts the residual part of log_base(qty).
            self.value_head = nn.Linear(d_model, 1)
        if not self.use_direct_magnitude and self.value_head_mode == "mark_conditioned_experts":
            rng_state = torch.random.get_rng_state()
            self.value_mark_delta_head = nn.Linear(d_model, self.num_real_marks)
            torch.random.set_rng_state(rng_state)

        # Next Time Intensity Parameters
        # v_t: vector, b_t: scalar, w_t: scalr (>0 enforced by softplus)
        self.v_t = nn.Linear(d_model, 1, bias = False)  # Depends on History H
        self.b_t = nn.Parameter(torch.zeros(1))                   # Base intensity bias
        self.w_raw = nn.Parameter(torch.zeros(1))                 # Time decay parameter

        # Initialize weights
        self._init_stable()

    @classmethod
    def _resolve_value_head_mode(cls, cfg: RMTPPConfig) -> str:
        mode = str(getattr(cfg, "value_head_mode", "shared") or "shared").strip().lower()
        if mode not in cls.VALID_VALUE_HEAD_MODES:
            available = ", ".join(sorted(cls.VALID_VALUE_HEAD_MODES))
            raise ValueError(
                f"Unsupported TitanTPP value_head_mode='{mode}'. Available: {available}"
            )
        return mode

    @classmethod
    def _resolve_qty_mark_gradient_mode(cls, cfg: RMTPPConfig) -> str:
        mode = str(
            getattr(cfg, "qty_mark_gradient_mode", "coupled") or "coupled"
        ).strip().lower()
        if mode not in cls.VALID_QTY_MARK_GRADIENT_MODES:
            available = ", ".join(sorted(cls.VALID_QTY_MARK_GRADIENT_MODES))
            raise ValueError(
                "Unsupported TitanTPP qty_mark_gradient_mode="
                f"'{mode}'. Available: {available}"
            )
        return mode

    @classmethod
    def _resolve_value_encoder_gradient_mode(cls, cfg: RMTPPConfig) -> str:
        mode = str(
            getattr(cfg, "value_encoder_gradient_mode", "coupled") or "coupled"
        ).strip().lower()
        if mode not in cls.VALID_VALUE_ENCODER_GRADIENT_MODES:
            available = ", ".join(sorted(cls.VALID_VALUE_ENCODER_GRADIENT_MODES))
            raise ValueError(
                "Unsupported TitanTPP value_encoder_gradient_mode="
                f"'{mode}'. Available: {available}"
            )
        return mode

    def _validate_gradient_mode_combination(self) -> None:
        if self.value_encoder_gradient_mode != "detached":
            return
        if (
            self.value_head_mode != "mark_conditioned_experts"
            or self.qty_mark_gradient_mode != "detached"
        ):
            raise ValueError(
                "value_encoder_gradient_mode='detached' requires "
                "value_head_mode='mark_conditioned_experts' and "
                "qty_mark_gradient_mode='detached'."
            )

    @classmethod
    def _resolve_marker_loss_mode(cls, cfg: RMTPPConfig) -> str:
        mode = str(getattr(cfg, "marker_loss_mode", "ce") or "ce").strip().lower()
        if mode not in cls.VALID_MARKER_LOSS_MODES:
            available = ", ".join(sorted(cls.VALID_MARKER_LOSS_MODES))
            raise ValueError(
                f"Unsupported TitanTPP marker_loss_mode='{mode}'. Available: {available}"
            )
        return mode

    def _validate_marker_loss_combination(self) -> None:
        if not math.isfinite(self.lambda_ordinal) or self.lambda_ordinal < 0.0:
            raise ValueError("lambda_ordinal must be finite and non-negative.")
        if self.marker_loss_mode == "ce" and self.lambda_ordinal != 0.0:
            raise ValueError("marker_loss_mode='ce' requires lambda_ordinal=0.")
        if self.marker_loss_mode == "ce_rps" and self.lambda_ordinal <= 0.0:
            raise ValueError("marker_loss_mode='ce_rps' requires lambda_ordinal>0.")

    @classmethod
    def _resolve_qty_decoder_mode(cls, cfg: RMTPPConfig) -> str:
        mode = str(getattr(cfg, "qty_decoder_mode", "mark_residual") or "mark_residual").strip().lower()
        if mode not in cls.VALID_QTY_DECODER_MODES:
            available = ", ".join(sorted(cls.VALID_QTY_DECODER_MODES))
            raise ValueError(
                f"Unsupported TitanTPP qty_decoder_mode='{mode}'. Available: {available}"
            )
        return mode

    @classmethod
    def _resolve_magnitude_norm_mode(cls, cfg: RMTPPConfig) -> str:
        mode = str(getattr(cfg, "magnitude_norm_mode", "global") or "global").strip().lower()
        if mode not in cls.VALID_MAGNITUDE_NORM_MODES:
            available = ", ".join(sorted(cls.VALID_MAGNITUDE_NORM_MODES))
            raise ValueError(
                f"Unsupported TitanTPP magnitude_norm_mode='{mode}'. Available: {available}"
            )
        return mode

    def _validate_quantity_decoder_combination(self) -> None:
        if not self.use_direct_magnitude:
            return
        decoder_label = self.qty_decoder_mode
        if not bool(getattr(self.cfg, "use_value_head", True)):
            raise ValueError(f"{decoder_label} requires use_value_head=True.")
        if not math.isclose(float(self.cfg.scale_base), 2.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{decoder_label} requires scale_base=2.0.")
        if str(getattr(self.cfg, "train_loss_scope", "all")) != "target_only":
            raise ValueError(f"{decoder_label} requires train_loss_scope='target_only'.")
        if str(getattr(self.cfg, "loss_mode", "residual_only")) != "hybrid":
            raise ValueError(f"{decoder_label} requires loss_mode='hybrid'.")
        if str(getattr(self.cfg, "value_input_mode", "none")) != "none":
            raise ValueError(
                f"{decoder_label} owns the continuous input and requires value_input_mode='none'."
            )
        if self.marker_loss_mode != "ce" or self.lambda_ordinal != 0.0:
            raise ValueError(f"{decoder_label} requires plain marker CE and lambda_ordinal=0.")
        if self.value_head_mode != "shared":
            raise ValueError(f"{decoder_label} does not combine with mark-conditioned value experts.")
        if self.qty_mark_gradient_mode != "coupled" or self.value_encoder_gradient_mode != "coupled":
            raise ValueError(f"{decoder_label} does not combine with detached V3 gradient routes.")
        if self.memory_mode in {"contextual_ttm", "hybrid_lmm_ttm"}:
            raise ValueError(f"{decoder_label} does not support contextual TTM.")
        if int(getattr(self.cfg, "magnitude_input_emb_dim", 0)) <= 0:
            raise ValueError("magnitude_input_emb_dim must be positive.")
        values = (
            float(getattr(self.cfg, "magnitude_global_mean", float("nan"))),
            float(getattr(self.cfg, "magnitude_global_var", float("nan"))),
            float(getattr(self.cfg, "magnitude_global_std", float("nan"))),
            float(getattr(self.cfg, "magnitude_sigma_floor", float("nan"))),
            float(getattr(self.cfg, "lambda_magnitude", float("nan"))),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Direct magnitude constants and lambda must be finite.")
        if values[1] < 0.0 or values[2] <= 0.0 or values[3] <= 0.0 or values[4] <= 0.0:
            raise ValueError("Magnitude std, sigma floor, and lambda must be positive.")

        if self.qty_decoder_mode == "direct_log_qty":
            if self.magnitude_norm_mode != "global":
                raise ValueError("direct_log_qty supports only global normalization.")
        else:
            if str(getattr(self.cfg, "magnitude_center_mode", "mean")) != "mean":
                raise ValueError("direct_raw_qty requires magnitude_center_mode='mean'.")
            if bool(getattr(self.cfg, "magnitude_revin_affine", False)):
                raise ValueError("direct_raw_qty requires magnitude_revin_affine=False.")
            if str(getattr(self.cfg, "magnitude_stat_context_mode", "none")) != "none":
                raise ValueError("direct_raw_qty requires magnitude_stat_context_mode='none'.")
            revin_eps = float(getattr(self.cfg, "magnitude_revin_eps", float("nan")))
            shrinkage_k = float(getattr(self.cfg, "magnitude_shrinkage_k", float("nan")))
            if not math.isfinite(revin_eps) or revin_eps <= 0.0:
                raise ValueError("direct_raw_qty requires positive finite magnitude_revin_eps.")
            if not math.isfinite(shrinkage_k) or shrinkage_k <= 0.0:
                raise ValueError("direct_raw_qty requires positive finite magnitude_shrinkage_k.")

        if self.use_direct_raw_quantity:
            return
        clamp_min = float(getattr(self.cfg, "magnitude_exp_clamp_min", -2.0))
        clamp_max = float(getattr(self.cfg, "magnitude_exp_clamp_max", 15.0))
        if not math.isfinite(clamp_min) or not math.isfinite(clamp_max) or clamp_min >= clamp_max:
            raise ValueError("Magnitude exp2 clamp bounds must be finite and increasing.")

    def _value_branch_hidden(self, h_j: torch.Tensor) -> torch.Tensor:
        if self.value_encoder_gradient_mode == "detached":
            return h_j.detach()
        return h_j

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
        if hasattr(self, "value_head"):
            nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.value_head.bias)
        if hasattr(self, "magnitude_head"):
            nn.init.normal_(self.magnitude_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.magnitude_head.bias)
        if hasattr(self, "value_mark_delta_head"):
            # V3 starts exactly on the V2 shared-head function.
            nn.init.zeros_(self.value_mark_delta_head.weight)
            nn.init.zeros_(self.value_mark_delta_head.bias)

        # w_raw를 음수로 시작하면 softplus(w_raw)가 작게 시작 -> wd가 작아져 폭주 억제
        with torch.no_grad():
            self.w_raw.fill_(-3.0)  # softplus(-3) ~ 0.048

    def _w_pos(self) -> torch.Tensor:
        # ensure w > 0 for stability of 1/w and inverse-CDF sampling
        return F.softplus(self.w_raw) + self.cfg.w_min

    def _clamped_exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, max = self.cfg.exp_clamp))

    def _activate_value(self, value_raw: torch.Tensor) -> torch.Tensor:
        if self.cfg.value_head_activation == "sigmoid":
            return torch.sigmoid(value_raw)
        return value_raw

    def predict_value_by_mark(self, h_j: torch.Tensor) -> torch.Tensor:
        """
        Predict one residual per real next mark.

        Shared mode expands the original V2 value prediction without adding
        parameters. V3 adds zero-initialized per-mark deltas to that shared
        prediction.
        """
        if self.use_direct_magnitude:
            raise RuntimeError("Legacy residual prediction is disabled in direct magnitude mode.")
        shared_raw = self.value_head(h_j)
        if self.value_head_mode == "mark_conditioned_experts":
            value_raw = shared_raw + self.value_mark_delta_head(h_j)
        else:
            value_raw = shared_raw.expand(*shared_raw.shape[:-1], self.num_real_marks)
        return self._activate_value(value_raw)

    def predict_value(
        self,
        h_j: torch.Tensor,
        marks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict one residual for each hidden state.

        Existing shared-head callers preserve V2 behavior. In V3, callers can
        provide explicit marks; otherwise the model uses its predicted mark.
        """
        if self.use_direct_magnitude:
            raise RuntimeError("Legacy residual prediction is disabled in direct magnitude mode.")
        if self.value_head_mode == "shared":
            return self._activate_value(self.value_head(h_j).squeeze(-1))

        if marks is None:
            logits_real = self.mark_head(h_j)[..., :self.num_real_marks]
            marks = torch.argmax(logits_real, dim=-1)
        values_by_mark = self.predict_value_by_mark(h_j)
        safe_marks = marks.long().clamp(min=0, max=self.num_real_marks - 1)
        return values_by_mark.gather(-1, safe_marks.unsqueeze(-1)).squeeze(-1)

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

        We avoid argmax-mark reconstruction so the estimate stays
        differentiable. The configured gradient mode decides whether direct
        quantity supervision can also update the mark-probability gate.
        """
        mark_probs = torch.softmax(logits_no_pad, dim=-1)
        if self.qty_mark_gradient_mode == "detached":
            # V3b keeps the V3a forward estimate but prevents quantity loss
            # from directly optimizing the mark-probability gate.
            mark_probs = mark_probs.detach()
        real_mark_count = logits_no_pad.size(-1)
        mark_grid = torch.arange(
            real_mark_count,
            device=logits_no_pad.device,
            dtype=value_residual.dtype,
        ).view(*([1] * (logits_no_pad.ndim - 1)), real_mark_count)
        if value_residual.ndim == logits_no_pad.ndim - 1:
            value_by_mark = value_residual.unsqueeze(-1)
        elif value_residual.shape == logits_no_pad.shape:
            value_by_mark = value_residual
        else:
            raise ValueError(
                "value_residual must have shape [B, L] or [B, L, num_real_marks]."
            )
        log_qty = mark_grid + value_by_mark
        scale_base = torch.as_tensor(
            self.cfg.scale_base,
            dtype=value_residual.dtype,
            device=value_residual.device,
        )
        qty_per_mark = torch.pow(scale_base, log_qty)
        return (mark_probs * qty_per_mark).sum(dim=-1)

    def build_magnitude_context(
        self,
        marks: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> MagnitudeContext:
        """Build the stateless causal context shared by input and decoder."""
        if not self.use_direct_magnitude:
            raise RuntimeError("Magnitude context is only available in direct decoder mode.")
        if self.qty_decoder_mode == "direct_log_qty":
            return build_global_magnitude_context(
                marks,
                values,
                mask,
                num_real_marks=self.num_real_marks,
                global_mean=float(self.cfg.magnitude_global_mean),
                global_std=float(self.cfg.magnitude_global_std),
                sigma_floor=float(self.cfg.magnitude_sigma_floor),
            )
        if self.magnitude_norm_mode == "global":
            return build_raw_global_magnitude_context(
                marks,
                values,
                mask,
                num_real_marks=self.num_real_marks,
                global_mean=float(self.cfg.magnitude_global_mean),
                global_std=float(self.cfg.magnitude_global_std),
                sigma_floor=float(self.cfg.magnitude_sigma_floor),
            )
        if self.magnitude_norm_mode == "causal_revin":
            return build_causal_revin_magnitude_context(
                marks,
                values,
                mask,
                num_real_marks=self.num_real_marks,
                revin_eps=float(self.cfg.magnitude_revin_eps),
            )
        if self.magnitude_norm_mode == "causal_shrinkage_revin":
            return build_causal_shrinkage_revin_magnitude_context(
                marks,
                values,
                mask,
                num_real_marks=self.num_real_marks,
                global_mean=float(self.cfg.magnitude_global_mean),
                global_var=float(self.cfg.magnitude_global_var),
                sigma_floor=float(self.cfg.magnitude_sigma_floor),
                shrinkage_k=float(self.cfg.magnitude_shrinkage_k),
            )
        raise RuntimeError(f"Unsupported active magnitude norm mode: {self.magnitude_norm_mode}")

    def predict_direct_magnitude(
        self,
        h_j: torch.Tensor,
        *,
        marks: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        context: MagnitudeContext | None = None,
    ) -> dict[str, torch.Tensor | MagnitudeContext]:
        """Predict normalized magnitude and reconstruct its quantity domain."""
        if not self.use_direct_magnitude:
            raise RuntimeError("Direct magnitude prediction requires a direct decoder mode.")
        context = context or self.build_magnitude_context(marks, values, mask)
        normalized = self.magnitude_head(h_j).squeeze(-1)
        denormalized = denormalize_magnitude(normalized, context)
        if self.use_direct_raw_quantity:
            affine_qty = denormalized
            qty = affine_qty.clamp_min(0.0)
            log_qty = torch.log2(qty.clamp_min(float(self.cfg.eps)))
        else:
            log_qty = denormalized
            qty = safe_exp2(
                log_qty,
                clamp_min=float(self.cfg.magnitude_exp_clamp_min),
                clamp_max=float(self.cfg.magnitude_exp_clamp_max),
            )
            affine_qty = qty
        for name, tensor in {
            "normalized magnitude": normalized,
            "affine quantity": affine_qty,
            "quantity": qty,
        }.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Direct {name} prediction contains NaN or Inf.")
        return {
            "normalized": normalized,
            "denormalized": denormalized,
            "log_qty": log_qty,
            "affine_qty": affine_qty,
            "qty": qty,
            "context": context,
        }

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
        magnitude_context: Optional[MagnitudeContext] = None,
    ) -> torch.Tensor:
        """
        Process input sequence through Embedding -> Titan Encoder -> LMM
        :returns
            h: [B, L, d_model] (Context vector for each step)
        """
        # marks: [B, L], dts: [B, L]

        # 1. Embeddings. Canonicalize masked tokens so arbitrary padding values
        # cannot affect direct-magnitude predictions.
        feature_marks = marks
        feature_dts = dts
        if mask is not None:
            feature_marks = marks.masked_fill(~mask, int(self.cfg.num_marks - 1))
            feature_dts = dts.masked_fill(~mask, 0)
        emb = self.emb(feature_marks)                   # [B, L, E]
        # dt_feat = dts.unsqueeze(-1).float()             # [B, L, 1]
        dt_feat = torch.log1p(feature_dts.clamp_min(0).float()).unsqueeze(-1)
        features = [emb, dt_feat]
        if self.use_direct_magnitude:
            if values is None or mask is None:
                raise ValueError("Direct magnitude forward requires values and mask.")
            magnitude_context = magnitude_context or self.build_magnitude_context(
                marks,
                values,
                mask,
            )
            features.append(
                self.magnitude_input_proj(
                    magnitude_context.normalized_history.unsqueeze(-1)
                )
            )
        else:
            value_feat = build_value_input_feature(
                marks=marks,
                values=values,
                cfg=self.cfg,
                mask=mask,
            )
        if not self.use_direct_magnitude and value_feat is not None:
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

        magnitude_context = None
        if self.use_direct_magnitude:
            if values is None:
                raise ValueError("Direct magnitude nll requires residual values.")
            magnitude_context = self.build_magnitude_context(marks, values, mask)
        input_values = mask_appended_target_value(values, mask)
        h = self.forward(
            marks,
            dts,
            values=input_values,
            mask=mask,
            series_memory=series_memory,
            magnitude_context=magnitude_context,
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

        magnitude_loss = torch.zeros((), device=marks.device, dtype=torch.float32)
        log_qty_hat = None
        qty_affine_hat = None
        qty_hat = None

        # Direct magnitude regression bypasses predicted marks for quantity.
        if self.use_direct_magnitude:
            assert values is not None
            assert magnitude_context is not None
            if self.use_direct_raw_quantity:
                magnitude_target = reconstruct_raw_quantity(
                    y_next,
                    values[:, 1:].float(),
                    num_real_marks=self.num_real_marks,
                )
                target_log_qty = torch.log2(magnitude_target.clamp_min(float(self.cfg.eps)))
            else:
                target_log_qty = reconstruct_log2_quantity(
                    y_next,
                    values[:, 1:].float(),
                    num_real_marks=self.num_real_marks,
                )
                magnitude_target = target_log_qty
            direct = self.predict_direct_magnitude(
                h_j,
                marks=marks,
                values=values,
                mask=mask,
                context=magnitude_context,
            )
            normalized_hat = direct["normalized"]
            log_qty_hat = direct["log_qty"]
            qty_affine_hat = direct["affine_qty"]
            qty_hat = direct["qty"]
            assert isinstance(normalized_hat, torch.Tensor)
            assert isinstance(log_qty_hat, torch.Tensor)
            assert isinstance(qty_affine_hat, torch.Tensor)
            assert isinstance(qty_hat, torch.Tensor)
            normalized_target = normalized_magnitude_target(
                magnitude_target,
                magnitude_context,
            )
            magnitude_step = F.huber_loss(
                normalized_hat,
                normalized_target,
                reduction="none",
            )
            magnitude_loss = (magnitude_step * step_mask).sum() / step_mask.sum().clamp_min(1)
            value_hat = None
            value_by_mark = None
            value_loss = torch.zeros((), device=marks.device, dtype=torch.float32)
            if self.use_direct_raw_quantity:
                true_qty = magnitude_target
                expected_qty_for_loss = qty_affine_hat
            else:
                true_qty = safe_exp2(
                    target_log_qty,
                    clamp_min=float(self.cfg.magnitude_exp_clamp_min),
                    clamp_max=float(self.cfg.magnitude_exp_clamp_max),
                )
                expected_qty_for_loss = qty_hat

            qty_scale_value = torch.as_tensor(
                float(max(getattr(self.cfg, "qty_scale_value", 1.0), 1.0)),
                device=true_qty.device,
                dtype=true_qty.dtype,
            )
            qty_sq = F.huber_loss(
                expected_qty_for_loss / qty_scale_value,
                true_qty / qty_scale_value,
                reduction="none",
            )
            qty_loss = (qty_sq * step_mask).sum() / step_mask.sum().clamp_min(1)

        # Legacy residual regression for mark-factorized quantity reconstruction.
        elif values is not None and self.cfg.use_value_head:
            value_next = values[:, 1:].float()
            value_by_mark = self.predict_value_by_mark(self._value_branch_hidden(h_j))
            safe_next_marks = y_next.clamp(min=0, max=self.num_real_marks - 1)
            value_hat = value_by_mark.gather(
                -1,
                safe_next_marks.unsqueeze(-1),
            ).squeeze(-1)
            # value_sq = F.smooth_l1_loss(value_hat, value_next, reduction="none")
            value_sq = F.huber_loss(value_hat, value_next, reduction="none")
            value_sq = value_sq * step_mask
            value_loss = value_sq.sum() / step_mask.sum().clamp_min(1)

            # Direct quantity supervision is kept optional so TitanTPP can
            # stay on the legacy residual-only objective for the paper A/B,
            # while still allowing development-time experiments on qty loss.
            pad_id = int(self.cfg.num_marks - 1)
            logits_real = logits[..., :pad_id]
            expected_qty = self.expected_qty_from_logits(logits_real, value_by_mark)
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
            value_by_mark = None
            value_loss = torch.zeros((), device=marks.device, dtype=torch.float32)
            qty_loss = torch.zeros((), device=marks.device, dtype=torch.float32)

        # apply mask
        logp_y = log_y * step_mask
        logf_dt = logf_dt * step_mask

        # negative log-likelihood
        nll_marker = -logp_y.sum() / (step_mask.sum().clamp_min(1))
        nll_time = -logf_dt.sum() / (step_mask.sum().clamp_min(1))
        nll_total = nll_marker + nll_time
        ordinal_marker_loss = masked_normalized_ranked_probability_score(
            logits,
            y_next,
            step_mask,
            num_real_marks=self.num_real_marks,
        )
        ordinal_weight = self.lambda_ordinal if self.marker_loss_mode == "ce_rps" else 0.0
        marker_train_loss = nll_marker + ordinal_weight * ordinal_marker_loss

        loss_mode = getattr(self.cfg, "loss_mode", "residual_only")
        if self.use_direct_magnitude:
            total_loss = (
                marker_train_loss
                + nll_time
                + float(self.cfg.lambda_magnitude) * magnitude_loss
                + getattr(self.cfg, "lambda_qty", 0.25) * qty_loss
            )
        elif loss_mode == "residual_only":
            total_loss = marker_train_loss + nll_time + value_loss
        elif loss_mode == "hybrid":
            total_loss = (
                marker_train_loss
                + nll_time
                + value_loss
                + getattr(self.cfg, "lambda_qty", 0.25) * qty_loss
            )
        elif loss_mode == "qty_only":
            total_loss = (
                marker_train_loss
                + nll_time
                + getattr(self.cfg, "lambda_qty", 0.25) * qty_loss
            )
        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

        return {
            'nll': nll_total,
            'nll_marker': nll_marker,
            'nll_time': nll_time,
            'ordinal_marker_loss': ordinal_marker_loss,
            'marker_train_loss': marker_train_loss,
            'value_loss': value_loss,
            'magnitude_loss': magnitude_loss,
            'qty_loss': qty_loss,
            'total_loss': total_loss,
            'value_hat': value_hat,
            'value_by_mark': value_by_mark,
            'log_qty_hat': log_qty_hat,
            'qty_affine_hat': qty_affine_hat,
            'qty_hat': qty_hat,
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
