from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import torch

from models.RMTPPs.RMTPP import RMTPP
from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.TransformerHawkesTPP import TransformerHawkesTPP
from models.RMTPPs.config import RMTPPConfig, THPConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig, THPCandidate
from simple_lab_test.search.common.experiment_utils import (
    TitanCandidate,
    build_rmtpp_config,
    build_titan_config,
)
from simple_lab_test.search.common.benchmark_utils import make_search_cfg


def canonical_model_name(model_name: str) -> str:
    """
    Normalize user-facing model aliases into internal labels.
    """
    normalized = model_name.strip().lower()
    aliases = {
        "transformer_hawkes_process": "thp",
        "transformer_hawkes": "thp",
        "transformerhawkes": "thp",
        "transformerhawkesprocess": "thp",
        "transformer_hawkes_tpp": "thp",
        "transformerhawkestpp": "thp",
        "transformerhawkes_tpp": "thp",
        "transformerhawkestemporalpointprocess": "thp",
        "transformerhawkesprocess_tpp": "thp",
    }
    return aliases.get(normalized, normalized)


def default_titan_candidates() -> list[TitanCandidate]:
    """
    Curated TitanTPP presets shared by all experiment runners.

    Keeping Titan and THP candidates in this registry makes model comparison
    scripts easier to extend: new candidates should be added here first, then
    selected from CLI arguments such as `--titan-candidates`.
    """
    return [
        TitanCandidate(
            name="small_no_lmm",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=16,
            use_lmm=False,
            mem_size=64,
            mem_topk=4,
            memory_mode="none",
        ),
        TitanCandidate(
            name="small_lmm",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=16,
            use_lmm=True,
            mem_size=64,
            mem_topk=4,
            memory_mode="static_lmm",
        ),
        TitanCandidate(
            name="small_contextual_ttm",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=0,
            use_lmm=False,
            mem_size=64,
            mem_topk=4,
            memory_mode="contextual_ttm",
        ),
        TitanCandidate(
            name="small_contextual_ttm_p8",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=0,
            use_lmm=False,
            mem_size=64,
            mem_topk=4,
            memory_mode="contextual_ttm",
            ttm_chunk_size=8,
        ),
        TitanCandidate(
            name="small_contextual_ttm_p16",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=0,
            use_lmm=False,
            mem_size=64,
            mem_topk=4,
            memory_mode="contextual_ttm",
            ttm_chunk_size=16,
        ),
        TitanCandidate(
            name="small_contextual_ttm_p32",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=0,
            use_lmm=False,
            mem_size=64,
            mem_topk=4,
            memory_mode="contextual_ttm",
            ttm_chunk_size=32,
        ),
        TitanCandidate(
            name="small_hybrid_lmm_ttm",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=0,
            use_lmm=True,
            mem_size=64,
            mem_topk=4,
            memory_mode="hybrid_lmm_ttm",
        ),
        TitanCandidate(
            name="small_series_lmm",
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            contextual_mem_size=0,
            persistent_mem_size=0,
            use_lmm=True,
            mem_size=64,
            mem_topk=4,
            memory_mode="series_lmm",
        ),
        TitanCandidate(
            name="small_deep_lmm",
            d_model=64,
            n_layers=3,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            contextual_mem_size=16,
            persistent_mem_size=16,
            use_lmm=True,
            mem_size=64,
            mem_topk=4,
            memory_mode="static_lmm",
        ),
        TitanCandidate(
            name="mid_no_lmm",
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            contextual_mem_size=32,
            persistent_mem_size=32,
            use_lmm=False,
            mem_size=128,
            mem_topk=8,
            memory_mode="none",
        ),
        TitanCandidate(
            name="mid_lmm",
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            contextual_mem_size=32,
            persistent_mem_size=32,
            use_lmm=True,
            mem_size=128,
            mem_topk=8,
            memory_mode="static_lmm",
        ),
        TitanCandidate(
            name="mid_contextual_ttm",
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            contextual_mem_size=32,
            persistent_mem_size=0,
            use_lmm=False,
            mem_size=128,
            mem_topk=8,
            memory_mode="contextual_ttm",
        ),
        TitanCandidate(
            name="mid_hybrid_lmm_ttm",
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            contextual_mem_size=32,
            persistent_mem_size=0,
            use_lmm=True,
            mem_size=128,
            mem_topk=8,
            memory_mode="hybrid_lmm_ttm",
        ),
        TitanCandidate(
            name="mid_series_lmm",
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            contextual_mem_size=0,
            persistent_mem_size=0,
            use_lmm=True,
            mem_size=128,
            mem_topk=8,
            memory_mode="series_lmm",
        ),
        TitanCandidate(
            name="mid_deep_lmm",
            d_model=128,
            n_layers=3,
            n_heads=8,
            d_ff=512,
            dropout=0.1,
            contextual_mem_size=32,
            persistent_mem_size=32,
            use_lmm=True,
            mem_size=128,
            mem_topk=8,
            memory_mode="static_lmm",
        ),
        TitanCandidate(
            name="mid_dropout_lmm",
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.2,
            contextual_mem_size=32,
            persistent_mem_size=32,
            use_lmm=True,
            mem_size=128,
            mem_topk=8,
            memory_mode="static_lmm",
        ),
    ]


def default_thp_candidates() -> list[THPCandidate]:
    """
    Curated THP presets for quick baseline sweeps.
    """
    return [
        THPCandidate(name="small", d_model=64, d_inner=256, n_layers=2, n_heads=4, dropout=0.1),
        THPCandidate(name="base", d_model=128, d_inner=512, n_layers=3, n_heads=4, dropout=0.1),
        THPCandidate(name="deep", d_model=128, d_inner=512, n_layers=4, n_heads=4, dropout=0.1),
        THPCandidate(name="wide", d_model=256, d_inner=1024, n_layers=3, n_heads=8, dropout=0.1),
    ]


def find_candidate_by_name(candidates: Iterable[Any], name: str) -> Any:
    """
    Recover a preset candidate from a list by stable name.
    """
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    available = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(f"Unknown candidate '{name}'. Available: {available}")


def flatten_candidate(candidate: Any) -> dict[str, Any]:
    """
    Persist candidate architecture fields alongside run metrics.
    """
    if candidate is None:
        return {}
    if hasattr(candidate, "__dataclass_fields__"):
        row = asdict(candidate)
    else:
        row = dict(getattr(candidate, "__dict__", {}))
    row["candidate_name"] = getattr(candidate, "name", "none")
    return row


def build_project_rmtpp_config(
    *,
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
    marked_meta: dict[str, Any],
) -> RMTPPConfig:
    """
    Build the shared decoder/time-intensity config for every encoder family.
    """
    search_cfg = make_search_cfg(cfg, run_cfg.dataset_kind)
    base_rmtpp_cfg = build_rmtpp_config(
        search_cfg,
        num_marks=int(marked_meta["num_marks"]),
        scale_base=run_cfg.scale_base,
    )

    hidden_dim = cfg.rmtpp_hidden_dim
    if hidden_dim is None:
        hidden_dim = int(getattr(run_cfg.candidate, "d_model", base_rmtpp_cfg.rnn_hidden_dim))

    return RMTPPConfig(
        **{
            **asdict(base_rmtpp_cfg),
            "rnn_type": cfg.rmtpp_rnn_type,
            "mark_emb_dim": cfg.rmtpp_mark_emb_dim,
            "rnn_hidden_dim": int(hidden_dim),
            "value_head_mode": cfg.value_head_mode,
            "qty_mark_gradient_mode": cfg.qty_mark_gradient_mode,
            "value_encoder_gradient_mode": cfg.value_encoder_gradient_mode,
            "marker_loss_mode": cfg.marker_loss_mode,
            "lambda_ordinal": float(cfg.lambda_ordinal),
            "qty_decoder_mode": cfg.qty_decoder_mode,
            "magnitude_norm_mode": cfg.magnitude_norm_mode,
            "magnitude_input_emb_dim": int(cfg.magnitude_input_emb_dim),
            "lambda_magnitude": float(cfg.lambda_magnitude),
            "magnitude_encoder_gradient_mode": cfg.magnitude_encoder_gradient_mode,
            "magnitude_aux_loss_mode": cfg.magnitude_aux_loss_mode,
            "lambda_log_qty": float(cfg.lambda_log_qty),
            "log_qty_huber_delta": float(cfg.log_qty_huber_delta),
            "log_qty_floor": float(cfg.log_qty_floor),
            "magnitude_global_mean": float(marked_meta.get("magnitude_global_mean", 0.0)),
            "magnitude_global_var": float(marked_meta.get("magnitude_global_var", 1.0)),
            "magnitude_global_std": float(marked_meta.get("magnitude_global_std", 1.0)),
            "magnitude_sigma_floor": float(
                marked_meta.get("magnitude_sigma_floor", cfg.magnitude_sigma_floor)
            ),
            "magnitude_revin_eps": float(cfg.magnitude_revin_eps),
            "magnitude_shrinkage_k": float(cfg.magnitude_shrinkage_k),
            "magnitude_center_mode": cfg.magnitude_center_mode,
            "magnitude_revin_affine": bool(cfg.magnitude_revin_affine),
            "magnitude_stat_context_mode": cfg.magnitude_stat_context_mode,
            "magnitude_exp_clamp_min": float(cfg.magnitude_exp_clamp_min),
            "magnitude_exp_clamp_max": float(cfg.magnitude_exp_clamp_max),
            "loss_mode": cfg.loss_mode,
            "value_input_mode": cfg.value_input_mode,
            "value_input_emb_dim": int(cfg.value_input_emb_dim),
            "train_loss_scope": cfg.train_loss_scope,
        }
    )


def build_model(
    *,
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
    marked_meta: dict[str, Any],
) -> tuple[torch.nn.Module, RMTPPConfig, Any]:
    """
    Instantiate one model from the model registry.
    """
    model_name = canonical_model_name(run_cfg.model_name)
    rmtpp_cfg = build_project_rmtpp_config(
        cfg=cfg,
        run_cfg=run_cfg,
        marked_meta=marked_meta,
    )
    search_cfg = make_search_cfg(cfg, run_cfg.dataset_kind)

    if model_name == "rmtpp":
        model = RMTPP(rmtpp_cfg).to(cfg.device)
        return model, rmtpp_cfg, None

    if model_name == "titantpp":
        titan_cfg = build_titan_config(search_cfg, run_cfg.candidate)
        model = TitanTPP(rmtpp_cfg, titan_cfg).to(cfg.device)
        return model, rmtpp_cfg, titan_cfg

    if model_name == "thp":
        candidate = run_cfg.candidate
        thp_cfg = THPConfig(
            d_model=int(candidate.d_model),
            d_inner=int(candidate.d_inner),
            n_layers=int(candidate.n_layers),
            n_heads=int(candidate.n_heads),
            dropout=float(candidate.dropout),
            normalize_before=bool(candidate.normalize_before),
            add_temporal_encoding_each_layer=bool(candidate.add_temporal_encoding_each_layer),
            use_rnn=bool(candidate.use_rnn),
            d_rnn=int(candidate.d_rnn),
        )
        model = TransformerHawkesTPP(rmtpp_cfg, thp_cfg).to(cfg.device)
        return model, rmtpp_cfg, thp_cfg

    raise ValueError(f"Unsupported model_name: {run_cfg.model_name}")


def model_run_label(model_name: str, candidate_name: str) -> str:
    """
    Human-readable label for plots and reports.
    """
    model_name = canonical_model_name(model_name)
    if model_name == "rmtpp":
        return "RMTPP"
    if model_name == "titantpp":
        return f"TITAN:{candidate_name}"
    if model_name == "thp":
        return f"THP:{candidate_name}"
    return f"{model_name.upper()}:{candidate_name}"


def make_rmtpp_proxy_candidate(hidden_dim: int, rnn_type: str) -> TitanCandidate:
    """
    Carry RMTPP capacity through the same RunConfig candidate slot.
    """
    return TitanCandidate(
        name=f"rmtpp_{rnn_type}_h{hidden_dim}",
        d_model=int(hidden_dim),
        n_layers=1,
        n_heads=1,
        d_ff=int(hidden_dim),
        dropout=0.1,
        contextual_mem_size=0,
        persistent_mem_size=0,
        use_lmm=False,
        mem_size=0,
        mem_topk=0,
        use_pos_emb=False,
        use_causal=True,
    )
