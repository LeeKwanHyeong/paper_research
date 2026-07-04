from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TitanConfig:
    # -------------------------
    # Data / IO
    # -------------------------
    lookback: int = 52
    horizon: int = 27

    # past exogenous
    past_exo_cont_dim: int = 0
    past_exo_cat_dims: Optional[List[int]] = None
    past_exo_cat_embed_dims: Optional[List[int]] = None

    final_clamp_nonneg: bool = False

    # future exogenous
    exo_dim: int = 0  # future exo dim

    # -------------------------
    # Model dims
    # -------------------------
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1

    # -------------------------
    # Memory (attention-side)
    # -------------------------
    # TitanTPP memory behavior is selected explicitly so experiments can
    # compare encoder-only, static memory, contextual TTM, and hybrid variants.
    # Supported values:
    # - "none": pure causal Titan encoder without memory tokens
    # - "static_lmm": learnable persistent/static memory
    # - "contextual_ttm": online contextual memory updated by the caller
    # - "series_lmm": retrieved per-series memory supplied by the caller
    # - "hybrid_lmm_ttm": contextual memory plus LMM retrieval
    memory_mode: str = "static_lmm"
    contextual_mem_size: int = 32
    persistent_mem_size: int = 32
    use_context_update: bool = False
    # Contextual TTM update granularity.
    # - ttm_chunk_size=1 keeps the original token-wise exact update path.
    # - larger chunks reduce repeated encoder calls while preserving causal
    #   attention inside each chunk.
    # - ttm_memory_update controls what each chunk contributes to memory.
    #   Supported values: "all", "last", "mean".
    ttm_chunk_size: int = 1
    ttm_memory_update: str = "all"

    # -------------------------
    # Positional embedding (encoder)
    # -------------------------
    use_pos_emb: bool = True
    max_len: int = 512

    # -------------------------
    # LMM (local memory matching)
    # -------------------------
    use_lmm: bool = True
    mem_size: int = 128
    mem_topk: int = 8

    # -------------------------
    # Causal Masking
    # -------------------------
    use_causal: bool = True


    # -------------------------
    # RevIN
    # -------------------------
    use_revin: bool = True

    # -------------------------
    # Output / head
    # -------------------------
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = None
