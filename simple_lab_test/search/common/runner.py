from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from models.RMTPPs.config import RMTPPConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig, RunPaths
from simple_lab_test.search.common.models import (
    build_model,
    canonical_model_name,
    default_thp_candidates,
    default_titan_candidates,
    find_candidate_by_name,
    flatten_candidate,
    make_rmtpp_proxy_candidate,
    model_run_label,
)
from simple_lab_test.search.common.experiment_utils import (
    DatasetSpec,
    build_logger,
    build_training_config,
    ensure_dir,
    save_json,
    sanitize_float_label,
    tee_training_output,
    to_jsonable,
)
from simple_lab_test.search.common.benchmark_utils import (
    build_marked_cache,
    default_profile_map,
    make_dataset_specs,
    make_search_cfg,
    markdown_table_from_df,
    persist_rows,
)
from utils.training import (
    TrainingConfig,
    eval_next_event_week_lookback,
    make_fixed_split_week_lookback_loaders,
    make_week_lookback_loaders,
)
from models.RMTPPs.value_conditioning import (
    mask_appended_target_value,
    predict_value_for_marks,
)


def set_global_seed(seed: int) -> None:
    """
    Keep data loader shuffling and model initialization reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_training_cfg(cfg: ExperimentConfig, dataset_kind: str | None = None) -> TrainingConfig:
    """
    Convert a unified experiment config into the shared trainer config.
    """
    search_cfg = make_search_cfg(cfg, dataset_kind)
    return build_training_config(search_cfg, epochs=cfg.epochs)


def attach_train_global_magnitude_stats(
    *,
    marked_df: pl.DataFrame,
    marked_meta: dict[str, Any],
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
) -> dict[str, Any]:
    """Freeze direct-decoder global moments from fixed-split train events only."""
    effective_meta = dict(marked_meta)
    decoder_mode = str(getattr(cfg, "qty_decoder_mode", "mark_residual"))
    if decoder_mode not in {"direct_log_qty", "direct_raw_qty"}:
        return effective_meta
    if not np.isclose(float(run_cfg.scale_base), 2.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{decoder_mode} requires scale_base=2.0.")

    required = {"chronological_split", "mark", "scale_residual"}
    missing = sorted(required - set(marked_df.columns))
    if missing:
        raise ValueError(f"Direct magnitude train-global statistics require columns: {missing}")
    train_factorized = (
        marked_df.lazy()
        .filter(pl.col("chronological_split") == "train")
        .select(
            (pl.col("mark").cast(pl.Float64) + pl.col("scale_residual").cast(pl.Float64))
            .alias("factorized_qty")
        )
        .collect()["factorized_qty"]
        .to_numpy()
    )
    if train_factorized.size == 0:
        raise ValueError("Direct magnitude train-global statistics found no train events.")
    if not np.isfinite(train_factorized).all():
        raise ValueError("Factorized train quantities must be finite.")

    if decoder_mode == "direct_raw_qty":
        train_magnitude = np.exp2(train_factorized)
        domain = "raw_qty"
    else:
        train_magnitude = train_factorized
        domain = "log2_qty"
    if not np.isfinite(train_magnitude).all():
        raise ValueError(f"Train-global {domain} values must be finite.")

    global_mean = float(train_magnitude.mean())
    global_var = float(train_magnitude.var())
    global_std = float(np.sqrt(global_var))
    if not np.isfinite(global_std) or global_std <= 0.0:
        raise ValueError(f"Train-global {domain} standard deviation must be positive.")
    sigma_floor = float(cfg.magnitude_sigma_floor)
    if decoder_mode == "direct_raw_qty":
        sigma_floor = max(0.001 * global_std, 1e-4)
    effective_meta.update({
        "magnitude_domain": domain,
        "magnitude_norm_mode": str(cfg.magnitude_norm_mode),
        "magnitude_stats_source_split": "train",
        "magnitude_train_event_count": int(train_magnitude.size),
        "magnitude_global_mean": global_mean,
        "magnitude_global_var": global_var,
        "magnitude_global_std": global_std,
        "magnitude_sigma_floor": sigma_floor,
    })
    return effective_meta


def magnitude_artifact_identity(cfg: ExperimentConfig) -> dict[str, Any]:
    """Return direct-magnitude fields that must travel with every artifact row."""
    decoder_mode = str(cfg.qty_decoder_mode)
    if decoder_mode == "direct_raw_qty":
        domain = "raw_qty"
    elif decoder_mode == "direct_log_qty":
        domain = "log2_qty"
    else:
        domain = "mark_residual"
    return {
        "magnitude_domain": domain,
        "magnitude_encoder_gradient_mode": str(cfg.magnitude_encoder_gradient_mode),
        "magnitude_aux_loss_mode": str(cfg.magnitude_aux_loss_mode),
        "lambda_log_qty": float(cfg.lambda_log_qty),
        "log_qty_huber_delta": float(cfg.log_qty_huber_delta),
        "log_qty_floor": float(cfg.log_qty_floor),
        "magnitude_sigma_floor": float(cfg.magnitude_sigma_floor),
        "magnitude_revin_eps": float(cfg.magnitude_revin_eps),
        "magnitude_shrinkage_k": float(cfg.magnitude_shrinkage_k),
        "magnitude_center_mode": str(cfg.magnitude_center_mode),
        "magnitude_revin_affine": bool(cfg.magnitude_revin_affine),
        "magnitude_stat_context_mode": str(cfg.magnitude_stat_context_mode),
    }


def magnitude_artifact_columns(cfg: ExperimentConfig) -> list[pl.Expr]:
    return [pl.lit(value).alias(name) for name, value in magnitude_artifact_identity(cfg).items()]


RAW_MAGNITUDE_DIAGNOSTIC_NAMES = (
    "qty_rmse",
    "qty_wape",
    "preclamp_negative_share",
    "magnitude_center_p01",
    "magnitude_center_p50",
    "magnitude_center_p95",
    "magnitude_center_p99",
    "magnitude_scale_p01",
    "magnitude_scale_p50",
    "magnitude_scale_p95",
    "magnitude_scale_p99",
    "magnitude_scale_floor_share",
    "normalized_target_abs_p95",
    "normalized_target_abs_p99",
    "normalized_target_nonfinite_count",
    "context_1_count",
    "context_1_qty_mae",
    "context_1_log_qty_mae",
    "context_2_4_count",
    "context_2_4_qty_mae",
    "context_2_4_log_qty_mae",
    "context_5_8_count",
    "context_5_8_qty_mae",
    "context_5_8_log_qty_mae",
    "context_9_plus_count",
    "context_9_plus_qty_mae",
    "context_9_plus_log_qty_mae",
)


def build_run_paths(cfg: ExperimentConfig, run_cfg: RunConfig) -> RunPaths:
    """
    Keep output paths stable across all model families.
    """
    base_label = sanitize_float_label(run_cfg.scale_base)
    dataset_path_name = run_cfg.dataset_name
    if run_cfg.dataset_kind in {"yellow_trip_daily", "yellow_trip_hourly"}:
        # Keep yellow-trip resolution variants in separate run folders so raw
        # weekly/daily artifacts cannot be reused as false cache hits.
        dataset_path_name = run_cfg.dataset_kind

    run_dir_base = (
        Path(cfg.base_dir)
        / "runs"
        / dataset_path_name
        / canonical_model_name(run_cfg.model_name)
        / f"lossmode_{cfg.loss_mode}"
    )
    if getattr(cfg, "qty_decoder_mode", "mark_residual") != "mark_residual":
        run_dir_base = (
            run_dir_base
            / f"qtydecoder_{cfg.qty_decoder_mode}"
            / f"magnorm_{cfg.magnitude_norm_mode}"
            / f"magemb_{int(cfg.magnitude_input_emb_dim)}"
            / f"lambdamag_{sanitize_float_label(cfg.lambda_magnitude)}"
            / f"magsigmafloor_{sanitize_float_label(cfg.magnitude_sigma_floor)}"
        )
        if cfg.qty_decoder_mode == "direct_raw_qty":
            run_dir_base = (
                run_dir_base
                / "domain_raw_qty"
                / f"magencgrad_{cfg.magnitude_encoder_gradient_mode}"
                / f"magaux_{cfg.magnitude_aux_loss_mode}"
                / f"lambdalogqty_{sanitize_float_label(cfg.lambda_log_qty)}"
                / f"logqtydelta_{sanitize_float_label(cfg.log_qty_huber_delta)}"
                / f"logqtyfloor_{sanitize_float_label(cfg.log_qty_floor)}"
                / f"center_{cfg.magnitude_center_mode}"
                / f"affine_{str(bool(cfg.magnitude_revin_affine)).lower()}"
                / f"statcontext_{cfg.magnitude_stat_context_mode}"
                / f"revineps_{sanitize_float_label(cfg.magnitude_revin_eps)}"
            )
            if cfg.magnitude_norm_mode == "causal_shrinkage_revin":
                run_dir_base = run_dir_base / (
                    f"k_{sanitize_float_label(cfg.magnitude_shrinkage_k)}"
                )
        else:
            run_dir_base = run_dir_base / (
                "magexpclamp_"
                f"{sanitize_float_label(cfg.magnitude_exp_clamp_min)}_"
                f"{sanitize_float_label(cfg.magnitude_exp_clamp_max)}"
            )
    if getattr(cfg, "split_mode", "internal") != "internal":
        run_dir_base = run_dir_base / f"split_{cfg.split_mode}"
    if getattr(cfg, "value_head_activation", "sigmoid") != "sigmoid":
        run_dir_base = run_dir_base / f"value_{cfg.value_head_activation}"
    if getattr(cfg, "value_head_mode", "shared") != "shared":
        run_dir_base = run_dir_base / f"valuehead_{cfg.value_head_mode}"
    if getattr(cfg, "qty_mark_gradient_mode", "coupled") != "coupled":
        run_dir_base = run_dir_base / f"qtymarkgrad_{cfg.qty_mark_gradient_mode}"
    if getattr(cfg, "value_encoder_gradient_mode", "coupled") != "coupled":
        run_dir_base = run_dir_base / f"valueencgrad_{cfg.value_encoder_gradient_mode}"
    if getattr(cfg, "marker_loss_mode", "ce") != "ce":
        run_dir_base = (
            run_dir_base
            / f"markloss_{cfg.marker_loss_mode}"
            / f"lambdaord_{sanitize_float_label(cfg.lambda_ordinal)}"
        )
    if getattr(cfg, "value_input_mode", "none") != "none":
        run_dir_base = (
            run_dir_base
            / f"valueinput_{cfg.value_input_mode}"
            / f"valueemb_{int(cfg.value_input_emb_dim)}"
        )
    if getattr(cfg, "train_loss_scope", "all") != "all":
        run_dir_base = run_dir_base / f"trainscope_{cfg.train_loss_scope}"
    if abs(float(getattr(cfg, "lambda_dt", 1.0)) - 1.0) > 1e-12:
        run_dir_base = run_dir_base / f"lambdadt_{sanitize_float_label(cfg.lambda_dt)}"
    if cfg.test_time_memory != "none":
        run_dir_base = run_dir_base / f"ttm_{cfg.test_time_memory}"

    run_dir = (
        run_dir_base
        / f"profile_{run_cfg.titan_profile}"
        / f"base_{base_label}"
        / run_cfg.candidate_name
        / f"epochs_{run_cfg.epochs}"
        / f"seed_{run_cfg.seed}"
    )
    return RunPaths(
        run_dir=ensure_dir(run_dir),
        checkpoint_dir=ensure_dir(run_dir / "checkpoints"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        manifest_dir=ensure_dir(run_dir / "manifest"),
        logs_dir=ensure_dir(run_dir / "logs"),
    )


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Store checkpoints on CPU to avoid keeping stale GPU tensors alive.
    """
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """
    Write a torch checkpoint through a temporary file before replacing it.

    Long GPU runs can be interrupted at awkward moments. Saving atomically keeps
    `last_epoch_state.pt` from becoming half-written if the process is killed
    during checkpoint serialization.
    """
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def torch_load_checkpoint(path: Path, *, map_location: str | torch.device) -> dict[str, Any]:
    """
    Load a full training checkpoint across PyTorch versions.

    Recent PyTorch versions may default toward weights-only loading. Resume
    checkpoints contain optimizer/history/RNG state, so we explicitly request a
    full load when the installed version supports that argument.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def capture_rng_state() -> dict[str, Any]:
    """
    Persist RNG states so resumed epochs stay as close as possible to an
    uninterrupted run.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """
    Restore RNG states saved in an epoch checkpoint.
    """
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        # `torch.set_rng_state` only accepts a CPU ByteTensor. Resume
        # checkpoints may be loaded with a CUDA map_location on GPU servers, so
        # keep this defensive conversion in place.
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().cpu()
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            cuda_state.detach().cpu() if isinstance(cuda_state, torch.Tensor) else cuda_state
            for cuda_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def finite_or_default(value: Any, default: float) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value_float):
        return default
    return value_float


def forward_model(
    model: torch.nn.Module,
    marks: torch.Tensor,
    dts: torch.Tensor,
    mask: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Call forward with mask when a model supports it.

    RMTPP/TitanTPP historically accept `(marks, dts)`, while THP benefits from
    the padding mask. This adapter keeps evaluators model-agnostic.
    """
    input_values = mask_appended_target_value(values, mask)
    try:
        return model.forward(marks, dts, values=input_values, mask=mask)
    except TypeError:
        try:
            return model.forward(marks, dts, values=input_values)
        except TypeError:
            return model.forward(marks, dts)


def compute_training_loss(
    *,
    model: torch.nn.Module,
    out: dict[str, torch.Tensor],
    training_cfg: TrainingConfig,
) -> torch.Tensor:
    """
    Project-standard training objective shared by compatible TPP models.
    """
    loss_mode = getattr(model.cfg, "loss_mode", "residual_only")
    marker_train_loss = out.get("marker_train_loss", out["nll_marker"])
    if getattr(model, "use_direct_magnitude", False):
        return (
            marker_train_loss
            + training_cfg.lambda_dt * out["nll_time"]
            + float(model.cfg.lambda_magnitude) * out["magnitude_loss"]
            + float(model.cfg.lambda_qty) * out["qty_loss"]
            + float(model.cfg.lambda_log_qty) * out["log_qty_aux_loss"]
        )
    if loss_mode == "residual_only":
        return (
            marker_train_loss
            + training_cfg.lambda_value * out["value_loss"]
            + training_cfg.lambda_dt * out["nll_time"]
        )
    if loss_mode == "hybrid":
        return (
            marker_train_loss
            + training_cfg.lambda_value * out["value_loss"]
            + training_cfg.lambda_dt * out["nll_time"]
            + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
        )
    if loss_mode == "qty_only":
        return (
            marker_train_loss
            + training_cfg.lambda_dt * out["nll_time"]
            + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
        )
    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


def score_from_metrics(metrics: dict[str, Any]) -> float:
    """
    Shared validation/test score used for checkpoint selection and reporting.
    """
    return float(
        metrics["mark_acc"]
        - 0.01 * metrics["dt_mae"]
        - 0.001 * metrics["qty_mae"]
    )


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Keep both score-based and validation-NLL-based sweet spots.
    """
    if not history:
        raise ValueError("Training history is empty.")

    best_score_row = max(
        history,
        key=lambda row: (
            finite_or_default(row.get("score"), -float("inf")),
            -finite_or_default(row.get("val_nll"), float("inf")),
        ),
    )
    min_nll_row = min(
        history,
        key=lambda row: (
            finite_or_default(row.get("val_nll"), float("inf")),
            -finite_or_default(row.get("score"), -float("inf")),
        ),
    )
    final_row = history[-1]

    summary = {
        "best_score_epoch": int(best_score_row["epoch"]),
        "best_score": float(best_score_row["score"]),
        "best_score_val_nll": float(best_score_row["val_nll"]),
        "best_score_qty_mae": float(best_score_row["qty_mae"]),
        "best_score_dt_mae": float(best_score_row["dt_mae"]),
        "best_score_mark_acc": float(best_score_row["mark_acc"]),
        "best_val_nll_epoch": int(min_nll_row["epoch"]),
        "best_val_nll": float(min_nll_row["val_nll"]),
        "best_val_nll_score": float(min_nll_row["score"]),
        "best_val_nll_qty_mae": float(min_nll_row["qty_mae"]),
        "best_val_nll_dt_mae": float(min_nll_row["dt_mae"]),
        "best_val_nll_mark_acc": float(min_nll_row["mark_acc"]),
        "best_val_nll_value_mae": float(min_nll_row["value_mae"]),
        "best_val_nll_log_qty_mae": float(min_nll_row.get("log_qty_mae", float("nan"))),
        "best_val_nll_log_qty_rmse": float(min_nll_row.get("log_qty_rmse", float("nan"))),
        "best_val_nll_magnitude_loss": float(
            min_nll_row.get("val_magnitude_loss", float("nan"))
        ),
        "best_val_nll_log_qty_aux_loss": float(
            min_nll_row.get("val_log_qty_aux_loss", float("nan"))
        ),
        "best_val_nll_ordinal_marker_loss": float(
            min_nll_row.get("val_ordinal_marker_loss", float("nan"))
        ),
        "best_val_nll_mark_balanced_accuracy": float(
            min_nll_row.get("mark_balanced_accuracy", float("nan"))
        ),
        "best_val_nll_mark_macro_f1": float(min_nll_row.get("mark_macro_f1", float("nan"))),
        "best_val_nll_mark_mae": float(min_nll_row.get("mark_mae", float("nan"))),
        "best_val_nll_mark_adjacent_accuracy": float(
            min_nll_row.get("mark_adjacent_accuracy", float("nan"))
        ),
        "best_val_nll_mark_0_recall": float(min_nll_row.get("mark_0_recall", float("nan"))),
        "best_val_nll_mark_1_recall": float(min_nll_row.get("mark_1_recall", float("nan"))),
        "final_epoch": int(final_row["epoch"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_score": float(final_row["score"]),
        "final_val_nll": float(final_row["val_nll"]),
        "final_qty_mae": float(final_row["qty_mae"]),
        "final_log_qty_mae": float(final_row.get("log_qty_mae", float("nan"))),
        "final_log_qty_rmse": float(final_row.get("log_qty_rmse", float("nan"))),
        "final_magnitude_loss": float(final_row.get("val_magnitude_loss", float("nan"))),
        "final_log_qty_aux_loss": float(
            final_row.get("val_log_qty_aux_loss", float("nan"))
        ),
        "final_dt_mae": float(final_row["dt_mae"]),
        "final_mark_acc": float(final_row["mark_acc"]),
        "final_ordinal_marker_loss": float(
            final_row.get("val_ordinal_marker_loss", float("nan"))
        ),
        "final_mark_mae": float(final_row.get("mark_mae", float("nan"))),
    }
    for metric_name in RAW_MAGNITUDE_DIAGNOSTIC_NAMES:
        if metric_name in min_nll_row:
            summary[f"best_val_nll_{metric_name}"] = float(min_nll_row[metric_name])
        if metric_name in final_row:
            summary[f"final_{metric_name}"] = float(final_row[metric_name])
    return summary


def scale_label(order: int, *, base: float, tail_order: int) -> str:
    low = base ** order
    if order >= tail_order:
        return f">={low:g}"
    high = (base ** (order + 1)) - 1
    if float(base).is_integer():
        return f"{int(round(low))}-{int(round(high))}"
    return f"[{low:g}, {base ** (order + 1):g})"


def empty_scale_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "true_sum": 0.0,
        "pred_sum": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "log_abs_sum": 0.0,
        "dt_abs_sum": 0.0,
        "mark_correct": 0,
        "abs_errors": [],
    }


@torch.no_grad()
def evaluate_scale_wise_qty(
    *,
    model: torch.nn.Module,
    val_loader: Iterable,
    device: str,
    analysis_scale_base: float,
    analysis_tail_order: int,
) -> pl.DataFrame:
    """
    Evaluate quantity reconstruction after grouping by true quantity scale.
    """
    model.eval()
    pad_id = int(model.cfg.num_marks - 1)
    log_base = float(np.log(analysis_scale_base))
    eps = float(getattr(model.cfg, "eps", 1e-8))
    buckets = {order: empty_scale_bucket() for order in range(0, int(analysis_tail_order) + 1)}

    for marks, dts, mask, _, values in val_loader:
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)
        values = values.to(device) if values is not None else None
        if values is None:
            continue

        valid = mask[:, -1] & mask[:, -2]
        h = forward_model(model, marks, dts, mask, values)
        h_prev = h[:, -2, :]
        y_mk = marks[:, -1]
        y_dt = dts[:, -1].float()
        y_val = values[:, -1].float()

        valid = valid & (y_mk != pad_id)
        if valid.sum().item() == 0:
            continue

        if getattr(model, "use_direct_magnitude", False):
            direct = model.predict_direct_magnitude(
                h_prev,
                marks=marks,
                values=values,
                mask=mask,
            )
            qty_hat_all = direct["qty"]
            log_qty_hat_all = direct["log_qty"]
            if not isinstance(qty_hat_all, torch.Tensor) or not isinstance(log_qty_hat_all, torch.Tensor):
                raise TypeError("Direct magnitude decoder must return tensor predictions.")

        h_prev = h_prev[valid]
        y_mk = y_mk[valid]
        y_dt = y_dt[valid]
        y_val = y_val[valid]

        logits = model.mark_head(h_prev)[..., :pad_id]
        pred_mk = torch.argmax(logits, dim=-1)
        if getattr(model, "use_direct_magnitude", False):
            qty_hat = qty_hat_all[valid]
            log_qty_hat = log_qty_hat_all[valid]
            true_log_qty = y_mk.float() + y_val
            if getattr(model, "use_direct_raw_quantity", False):
                qty_true = torch.exp2(true_log_qty)
            else:
                qty_true = torch.exp2(
                    true_log_qty.clamp(
                        min=float(model.cfg.magnitude_exp_clamp_min),
                        max=float(model.cfg.magnitude_exp_clamp_max),
                    )
                )
        else:
            value_hat = predict_value_for_marks(model, h_prev, pred_mk)
            qty_hat = model.reconstruct_qty(pred_mk, value_hat)
            qty_true = model.reconstruct_qty(y_mk, y_val)
            log_qty_hat = torch.log2(qty_hat.clamp_min(eps))
            true_log_qty = torch.log2(qty_true.clamp_min(eps))

        abs_err = (qty_hat - qty_true).abs()
        sq_err = (qty_hat - qty_true) ** 2
        log_abs_err = (log_qty_hat - true_log_qty).abs() * (np.log(2.0) / log_base)

        u = torch.full((y_dt.size(0),), 0.5, device=device).clamp_min(model.cfg.eps)
        dt_hat = model.sample_next_dt(h_prev, u=u).clamp_min(1.0)
        dt_abs_err = (dt_hat - y_dt).abs()

        scale_order = torch.floor(torch.log(qty_true.clamp_min(1.0)) / log_base).long()
        scale_order = scale_order.clamp_min(0).clamp_max(int(analysis_tail_order))

        for order in buckets:
            order_mask = scale_order == order
            if order_mask.sum().item() == 0:
                continue
            count = int(order_mask.sum().item())
            bucket = buckets[order]
            bucket["count"] += count
            bucket["true_sum"] += float(qty_true[order_mask].sum().item())
            bucket["pred_sum"] += float(qty_hat[order_mask].sum().item())
            bucket["abs_sum"] += float(abs_err[order_mask].sum().item())
            bucket["sq_sum"] += float(sq_err[order_mask].sum().item())
            bucket["log_abs_sum"] += float(log_abs_err[order_mask].sum().item())
            bucket["dt_abs_sum"] += float(dt_abs_err[order_mask].sum().item())
            bucket["mark_correct"] += int((pred_mk[order_mask] == y_mk[order_mask]).sum().item())
            bucket["abs_errors"].extend(abs_err[order_mask].detach().cpu().tolist())

    total_count = sum(bucket["count"] for bucket in buckets.values())
    rows: list[dict[str, Any]] = []
    for order, bucket in buckets.items():
        count = int(bucket["count"])
        label = scale_label(order, base=analysis_scale_base, tail_order=analysis_tail_order)
        if count == 0:
            rows.append({
                "scale_order": order,
                "scale_label": label,
                "count": 0,
                "share": 0.0,
                "true_qty_mean": float("nan"),
                "pred_qty_mean": float("nan"),
                "qty_mae": float("nan"),
                "qty_median_ae": float("nan"),
                "qty_rmse": float("nan"),
                "qty_wape": float("nan"),
                "log_abs_error": float("nan"),
                "dt_mae": float("nan"),
                "mark_acc": float("nan"),
            })
            continue
        abs_errors = np.asarray(bucket["abs_errors"], dtype=np.float64)
        rows.append({
            "scale_order": order,
            "scale_label": label,
            "count": count,
            "share": count / max(total_count, 1),
            "true_qty_mean": bucket["true_sum"] / count,
            "pred_qty_mean": bucket["pred_sum"] / count,
            "qty_mae": bucket["abs_sum"] / count,
            "qty_median_ae": float(np.median(abs_errors)) if abs_errors.size else float("nan"),
            "qty_rmse": float(np.sqrt(bucket["sq_sum"] / count)),
            "qty_wape": bucket["abs_sum"] / max(bucket["true_sum"], 1e-12),
            "log_abs_error": bucket["log_abs_sum"] / count,
            "dt_mae": bucket["dt_abs_sum"] / count,
            "mark_acc": bucket["mark_correct"] / count,
        })

    return pl.DataFrame(rows)


@torch.no_grad()
def evaluate_mark_confusion(
    *,
    model: torch.nn.Module,
    val_loader: Iterable,
    device: str,
) -> pl.DataFrame:
    """
    Count predicted mark by true mark on the final next-event target.

    This is especially useful for Instacart, where tail quantity marks can be
    hidden by aggregate accuracy if the model keeps predicting central bins.
    """
    model.eval()
    pad_id = int(model.cfg.num_marks - 1)
    counts: dict[tuple[int, int], int] = {}
    true_totals: dict[int, int] = {}

    for marks, dts, mask, _, values in val_loader:
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)
        values = values.to(device) if values is not None else None

        valid = mask[:, -1] & mask[:, -2]
        if valid.sum().item() == 0:
            continue

        h = forward_model(model, marks, dts, mask, values)
        h_prev = h[:, -2, :]
        y_mk = marks[:, -1]
        valid = valid & (y_mk != pad_id)
        if valid.sum().item() == 0:
            continue

        logits = model.mark_head(h_prev[valid])[..., :pad_id]
        pred_mk = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        true_mk = y_mk[valid].detach().cpu().tolist()

        for true_value, pred_value in zip(true_mk, pred_mk):
            true_int = int(true_value)
            pred_int = int(pred_value)
            counts[(true_int, pred_int)] = counts.get((true_int, pred_int), 0) + 1
            true_totals[true_int] = true_totals.get(true_int, 0) + 1

    rows = []
    for (true_mark, pred_mark), count in sorted(counts.items()):
        true_total = max(int(true_totals.get(true_mark, 0)), 1)
        rows.append({
            "true_mark": int(true_mark),
            "pred_mark": int(pred_mark),
            "count": int(count),
            "share_within_true": float(count / true_total),
        })
    schema = {
        "true_mark": pl.Int64,
        "pred_mark": pl.Int64,
        "count": pl.Int64,
        "share_within_true": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def summarize_mark_confusion(
    confusion_df: pl.DataFrame,
    *,
    num_real_marks: int,
) -> pl.DataFrame:
    """Build support-aware per-class metrics from a confusion artifact."""
    class_count = int(num_real_marks)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for row in confusion_df.iter_rows(named=True):
        true_mark = int(row["true_mark"])
        pred_mark = int(row["pred_mark"])
        if 0 <= true_mark < class_count and 0 <= pred_mark < class_count:
            matrix[true_mark, pred_mark] += int(row["count"])

    total = int(matrix.sum())
    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    correct = np.diag(matrix)
    rows: list[dict[str, Any]] = []
    for mark in range(class_count):
        true_count = int(true_counts[mark])
        pred_count = int(pred_counts[mark])
        precision = float(correct[mark] / pred_count) if pred_count else 0.0
        recall = float(correct[mark] / true_count) if true_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "mark": mark,
            "true_count": true_count,
            "true_share": float(true_count / total) if total else 0.0,
            "pred_count": pred_count,
            "pred_share": float(pred_count / total) if total else 0.0,
            "correct_count": int(correct[mark]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pl.DataFrame(rows)


def _series_list_from_marked_df(marked_df: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Convert marked events into Python lists for series-wise online evaluation.
    """
    grouped = (
        marked_df.sort(["oper_part_no", "seq"])
        .with_columns(pl.col("delta_t").cast(pl.Float32).clip(1, None).alias("delta_t"))
        .group_by("oper_part_no", maintain_order=True)
        .agg([
            pl.col("seq").cast(pl.Int32).alias("seq_list"),
            pl.col("delta_t").cast(pl.Float32).alias("dt_list"),
            pl.col("mark").cast(pl.Int64).alias("mk_list"),
            pl.col("scale_residual").cast(pl.Float32).alias("val_list")
            if "scale_residual" in marked_df.columns
            else pl.lit(None).alias("val_list"),
        ])
    )
    return grouped.to_dicts()


@torch.no_grad()
def _forward_titantpp_observed_tokens(
    *,
    model: torch.nn.Module,
    marks: np.ndarray,
    dts: np.ndarray,
    values: np.ndarray | None = None,
    device: str,
    update_context_memory: bool,
) -> torch.Tensor:
    """
    Run TitanTPP on already-observed tokens with optional memory update.
    """
    marks_t = torch.as_tensor(marks, dtype=torch.long, device=device).view(1, -1)
    dts_t = torch.as_tensor(dts, dtype=torch.float32, device=device).view(1, -1)
    values_t = None
    if values is not None:
        values_t = torch.as_tensor(values, dtype=torch.float32, device=device).view(1, -1)
    mask_t = torch.ones_like(marks_t, dtype=torch.bool)
    return model.forward(
        marks_t,
        dts_t,
        values=values_t,
        mask=mask_t,
        update_context_memory=bool(update_context_memory),
    )


@torch.no_grad()
def evaluate_titantpp_contextual_ttm(
    *,
    model: torch.nn.Module,
    marked_df: pl.DataFrame,
    training_cfg: TrainingConfig,
    selection: str,
) -> dict[str, Any]:
    """
    Evaluate TitanTPP with series-wise online contextual memory.

    This is TTM-Lite: no gradients or parameter adaptation are used at test
    time. For each series, we reset memory, warm it with train-prefix events,
    predict validation events in chronological order, and update memory only
    after the true target event becomes observed.
    """
    if canonical_model_name(type(model).__name__) != "titantpp" and not hasattr(model, "reset_contextual_memory"):
        raise TypeError("TTM-Lite contextual evaluation requires TitanTPP-like contextual memory methods.")

    model.eval()
    device = str(training_cfg.device)
    pad_id = int(model.cfg.num_marks - 1)
    max_len = max(int(training_cfg.max_seq_len), 1)
    lookback = max(int(training_cfg.lookback), 1)
    val_ratio = float(training_cfg.val_ratio)

    total = 0
    correct = 0
    dt_abs = 0.0
    dt_sq = 0.0
    value_abs = 0.0
    qty_abs = 0.0
    sum_nll_time = 0.0
    sum_nll_marker = 0.0
    sum_nll_total = 0.0
    sum_value_loss = 0.0
    warmup_events = 0
    update_events = 0
    evaluated_series = 0

    for series in _series_list_from_marked_df(marked_df):
        seq = np.asarray(series["seq_list"], dtype=np.int32)
        dts = np.asarray(series["dt_list"], dtype=np.float32)
        marks = np.asarray(series["mk_list"], dtype=np.int64)
        val_list = series.get("val_list")
        values = None if val_list is None else np.asarray(val_list, dtype=np.float32)
        n = int(len(seq))
        if n < 2:
            continue

        split_idx = int(np.floor(n * (1.0 - val_ratio)))
        first_target = max(1, split_idx)
        if first_target >= n:
            continue

        reset_memory = getattr(model, "reset_contextual_memory", None)
        if callable(reset_memory):
            reset_memory()

        # Warm the online memory with training-prefix observations only. Chunk
        # long prefixes so position embeddings and memory size remain bounded.
        if split_idx > 0:
            for start in range(0, split_idx, max_len):
                end = min(split_idx, start + max_len)
                if end <= start:
                    continue
                _forward_titantpp_observed_tokens(
                    model=model,
                    marks=marks[start:end],
                    dts=dts[start:end],
                    values=None if values is None else values[start:end],
                    device=device,
                    update_context_memory=True,
                )
                warmup_events += int(end - start)

        series_had_eval = False
        for target_idx in range(first_target, n):
            context_end = target_idx - 1
            left_seq = int(seq[context_end]) - (lookback - 1)
            context_start = int(np.searchsorted(seq, left_seq, side="left"))
            context_idx = np.arange(context_start, context_end + 1, dtype=np.int32)
            if context_idx.size == 0:
                continue
            if context_idx.size > max_len:
                context_idx = context_idx[-max_len:]

            h = _forward_titantpp_observed_tokens(
                model=model,
                marks=marks[context_idx],
                dts=dts[context_idx],
                values=None if values is None else values[context_idx],
                device=device,
                update_context_memory=False,
            )
            h_prev = h[:, -1, :]
            y_mk = torch.tensor([int(marks[target_idx])], dtype=torch.long, device=device)
            if int(y_mk.item()) == pad_id:
                continue
            y_dt = torch.tensor([float(dts[target_idx])], dtype=torch.float32, device=device)

            logits_full = model.mark_head(h_prev)
            log_y = -F.cross_entropy(logits_full, y_mk, reduction="none")
            logf_dt = model.log_f_dt(h_prev, y_dt)
            sum_nll_marker += float((-log_y).sum().item())
            sum_nll_time += float((-logf_dt).sum().item())
            sum_nll_total += float(((-log_y) + (-logf_dt)).sum().item())

            logits = logits_full[..., :pad_id]
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y_mk).sum().item())
            total += 1

            u = torch.full((1,), 0.5, device=device).clamp_min(model.cfg.eps)
            dt_hat = model.sample_next_dt(h_prev, u=u).clamp_min(1.0)
            dt_err = (dt_hat - y_dt).abs()
            dt_abs += float(dt_err.sum().item())
            dt_sq += float(((dt_hat - y_dt) ** 2).sum().item())

            if values is not None and getattr(model.cfg, "use_value_head", False):
                y_val = torch.tensor([float(values[target_idx])], dtype=torch.float32, device=device)
                value_hat = predict_value_for_marks(model, h_prev, y_mk)
                value_loss = F.huber_loss(value_hat, y_val, reduction="none")
                sum_value_loss += float(value_loss.sum().item())
                value_abs += float((value_hat - y_val).abs().sum().item())

                qty_value_hat = predict_value_for_marks(model, h_prev, pred)
                qty_hat = model.reconstruct_qty(pred, qty_value_hat)
                qty_true = model.reconstruct_qty(y_mk, y_val)
                qty_abs += float((qty_hat - qty_true).abs().sum().item())

            # Now the target is observed, so it can be written to memory for
            # future validation targets from the same series.
            _forward_titantpp_observed_tokens(
                model=model,
                marks=marks[target_idx:target_idx + 1],
                dts=dts[target_idx:target_idx + 1],
                values=None if values is None else values[target_idx:target_idx + 1],
                device=device,
                update_context_memory=True,
            )
            update_events += 1
            series_had_eval = True

        if series_had_eval:
            evaluated_series += 1

    reset_memory = getattr(model, "reset_contextual_memory", None)
    if callable(reset_memory):
        reset_memory()

    mark_acc = correct / max(total, 1)
    dt_mae = dt_abs / max(total, 1)
    dt_rmse = float(np.sqrt(dt_sq / max(total, 1)))
    value_mae = value_abs / max(total, 1)
    qty_mae = qty_abs / max(total, 1)
    val_nll_time = sum_nll_time / max(total, 1)
    val_nll_marker = sum_nll_marker / max(total, 1)
    val_nll = sum_nll_total / max(total, 1)
    val_value_loss = sum_value_loss / max(total, 1)
    score = mark_acc - 0.01 * dt_mae - 0.001 * qty_mae

    return {
        "selection": selection,
        "test_time_memory": "contextual",
        "score": float(score),
        "mark_acc": float(mark_acc),
        "dt_mae": float(dt_mae),
        "dt_rmse": float(dt_rmse),
        "value_mae": float(value_mae),
        "qty_mae": float(qty_mae),
        "val_nll_time": float(val_nll_time),
        "val_nll_marker": float(val_nll_marker),
        "val_nll": float(val_nll),
        "val_value_loss": float(val_value_loss),
        "_total": int(total),
        "_correct": int(correct),
        "_nll_steps": int(total),
        "evaluated_series": int(evaluated_series),
        "warmup_events": int(warmup_events),
        "contextual_update_events": int(update_events),
    }


def scale_metric_paths(run_paths: RunPaths, selection: str) -> tuple[Path, Path]:
    stem = f"scale_wise_{selection}"
    return run_paths.metrics_dir / f"{stem}.csv", run_paths.metrics_dir / f"{stem}.parquet"


def test_metric_paths(run_paths: RunPaths, selection: str) -> tuple[Path, Path, Path]:
    """
    Per-checkpoint held-out test metrics for fixed-split experiments.
    """
    stem = f"test_metrics_{selection}"
    return (
        run_paths.metrics_dir / f"{stem}.json",
        run_paths.metrics_dir / f"{stem}.csv",
        run_paths.metrics_dir / f"{stem}.parquet",
    )


def test_scale_metric_paths(run_paths: RunPaths, selection: str) -> tuple[Path, Path]:
    """
    Scale-wise held-out test quantity metrics for fixed-split experiments.
    """
    stem = f"test_scale_wise_{selection}"
    return run_paths.metrics_dir / f"{stem}.csv", run_paths.metrics_dir / f"{stem}.parquet"


def confusion_metric_paths(run_paths: RunPaths, selection: str, eval_split: str) -> tuple[Path, Path]:
    """
    Mark confusion matrix path for validation/test target transitions.
    """
    stem = f"{eval_split}_mark_confusion_{selection}"
    return run_paths.metrics_dir / f"{stem}.csv", run_paths.metrics_dir / f"{stem}.parquet"


def mark_class_metric_paths(run_paths: RunPaths, selection: str, eval_split: str) -> tuple[Path, Path]:
    stem = f"{eval_split}_mark_class_metrics_{selection}"
    return run_paths.metrics_dir / f"{stem}.csv", run_paths.metrics_dir / f"{stem}.parquet"


def ttm_metric_paths(run_paths: RunPaths, selection: str) -> tuple[Path, Path, Path]:
    """
    Per-checkpoint TTM-Lite metric files.
    """
    stem = f"ttm_contextual_{selection}"
    return (
        run_paths.metrics_dir / f"{stem}.json",
        run_paths.metrics_dir / f"{stem}.csv",
        run_paths.metrics_dir / f"{stem}.parquet",
    )


def cached_run_is_complete(
    *,
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
    run_paths: RunPaths,
    effective_marked_meta: dict[str, Any] | None = None,
) -> bool:
    summary_path = run_paths.metrics_dir / "summary.json"
    history_path = run_paths.metrics_dir / "history.json"
    nll_checkpoint_path = run_paths.checkpoint_dir / "best_val_nll_model.pt"
    if not (summary_path.exists() and history_path.exists() and nll_checkpoint_path.exists()):
        return False
    for selection in cfg.eval_selections:
        csv_path, parquet_path = scale_metric_paths(run_paths, selection)
        if not (csv_path.exists() and parquet_path.exists()):
            return False
        if getattr(cfg, "split_mode", "internal") == "fixed":
            test_json_path, test_csv_path, test_parquet_path = test_metric_paths(run_paths, selection)
            test_scale_csv_path, test_scale_parquet_path = test_scale_metric_paths(run_paths, selection)
            if not (
                test_json_path.exists()
                and test_csv_path.exists()
                and test_parquet_path.exists()
                and test_scale_csv_path.exists()
                and test_scale_parquet_path.exists()
            ):
                return False
        if cfg.test_time_memory == "contextual" and canonical_model_name(run_cfg.model_name) == "titantpp":
            json_path, ttm_csv_path, ttm_parquet_path = ttm_metric_paths(run_paths, selection)
            if not (json_path.exists() and ttm_csv_path.exists() and ttm_parquet_path.exists()):
                return False
    with open(summary_path, "r", encoding="utf-8") as f:
        cached_summary = json.load(f)
    magnitude_stats_match = True
    if cfg.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        if effective_marked_meta is None:
            return False
        for summary_key, meta_key in (
            ("magnitude_global_mean", "magnitude_global_mean"),
            ("magnitude_global_var", "magnitude_global_var"),
            ("magnitude_global_std", "magnitude_global_std"),
            ("magnitude_effective_sigma_floor", "magnitude_sigma_floor"),
        ):
            cached_value = cached_summary.get(summary_key)
            current_value = effective_marked_meta.get(meta_key)
            if cached_value is None or current_value is None or not np.isclose(
                float(cached_value),
                float(current_value),
                rtol=1e-12,
                atol=1e-12,
            ):
                magnitude_stats_match = False
                break
    return (
        magnitude_stats_match
        and
        int(cached_summary.get("epochs", -1)) == int(run_cfg.epochs)
        and str(cached_summary.get("test_time_memory", "none")) == str(cfg.test_time_memory)
        and str(cached_summary.get("split_mode", "internal")) == str(getattr(cfg, "split_mode", "internal"))
        and str(cached_summary.get("value_head_activation", "sigmoid")) == str(cfg.value_head_activation)
        and str(cached_summary.get("value_head_mode", "shared")) == str(cfg.value_head_mode)
        and str(cached_summary.get("qty_mark_gradient_mode", "coupled"))
        == str(cfg.qty_mark_gradient_mode)
        and str(cached_summary.get("value_encoder_gradient_mode", "coupled"))
        == str(cfg.value_encoder_gradient_mode)
        and str(cached_summary.get("marker_loss_mode", "ce")) == str(cfg.marker_loss_mode)
        and abs(
            float(cached_summary.get("lambda_ordinal", 0.0)) - float(cfg.lambda_ordinal)
        ) <= 1e-12
        and str(cached_summary.get("qty_decoder_mode", "mark_residual"))
        == str(cfg.qty_decoder_mode)
        and str(cached_summary.get("magnitude_norm_mode", "global"))
        == str(cfg.magnitude_norm_mode)
        and int(cached_summary.get("magnitude_input_emb_dim", 8))
        == int(cfg.magnitude_input_emb_dim)
        and abs(
            float(cached_summary.get("lambda_magnitude", 1.0)) - float(cfg.lambda_magnitude)
        ) <= 1e-12
        and str(cached_summary.get("magnitude_encoder_gradient_mode", "coupled"))
        == str(cfg.magnitude_encoder_gradient_mode)
        and str(cached_summary.get("magnitude_aux_loss_mode", "none"))
        == str(cfg.magnitude_aux_loss_mode)
        and abs(
            float(cached_summary.get("lambda_log_qty", 0.25))
            - float(cfg.lambda_log_qty)
        ) <= 1e-12
        and abs(
            float(cached_summary.get("log_qty_huber_delta", 1.0))
            - float(cfg.log_qty_huber_delta)
        ) <= 1e-12
        and abs(
            float(cached_summary.get("log_qty_floor", 1.0)) - float(cfg.log_qty_floor)
        ) <= 1e-12
        and abs(
            float(cached_summary.get("magnitude_sigma_floor", 0.0014535461338152059))
            - float(cfg.magnitude_sigma_floor)
        ) <= 1e-12
        and abs(
            float(cached_summary.get("magnitude_revin_eps", 1e-5))
            - float(cfg.magnitude_revin_eps)
        ) <= 1e-12
        and abs(
            float(cached_summary.get("magnitude_shrinkage_k", 8.0))
            - float(cfg.magnitude_shrinkage_k)
        ) <= 1e-12
        and str(cached_summary.get("magnitude_center_mode", "mean"))
        == str(cfg.magnitude_center_mode)
        and bool(cached_summary.get("magnitude_revin_affine", False))
        == bool(cfg.magnitude_revin_affine)
        and str(cached_summary.get("magnitude_stat_context_mode", "none"))
        == str(cfg.magnitude_stat_context_mode)
        and abs(
            float(cached_summary.get("magnitude_exp_clamp_min", -2.0))
            - float(cfg.magnitude_exp_clamp_min)
        ) <= 1e-12
        and abs(
            float(cached_summary.get("magnitude_exp_clamp_max", 15.0))
            - float(cfg.magnitude_exp_clamp_max)
        ) <= 1e-12
    )


def save_checkpoint(
    *,
    path: Path,
    model_state: dict[str, torch.Tensor],
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
    training_cfg: TrainingConfig,
    rmtpp_cfg: RMTPPConfig,
    encoder_cfg: Any,
    summary: dict[str, Any],
    selection: str,
) -> None:
    torch.save(
        {
            "selection": selection,
            "model_state_dict": model_state,
            "experiment_config": to_jsonable(cfg),
            "run_config": to_jsonable(run_cfg),
            "training_config": to_jsonable(training_cfg),
            "rmtpp_config": to_jsonable(rmtpp_cfg),
            "encoder_config": to_jsonable(encoder_cfg),
            "summary": summary,
        },
        path,
    )


def save_epoch_resume_checkpoint(
    *,
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    best_score: float,
    best_val_nll: float,
    best_score_state: dict[str, torch.Tensor] | None,
    best_val_nll_state: dict[str, torch.Tensor] | None,
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
    training_cfg: TrainingConfig,
    rmtpp_cfg: RMTPPConfig,
    encoder_cfg: Any,
) -> None:
    """
    Save enough state to resume a partially completed run after the last
    finished epoch.
    """
    atomic_torch_save(
        {
            "checkpoint_type": "epoch_resume",
            "epoch": int(epoch),
            "model_state_dict": clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_score": float(best_score),
            "best_val_nll": float(best_val_nll),
            "best_score_state_dict": best_score_state,
            "best_val_nll_state_dict": best_val_nll_state,
            "experiment_config": to_jsonable(cfg),
            "run_config": to_jsonable(run_cfg),
            "training_config": to_jsonable(training_cfg),
            "rmtpp_config": to_jsonable(rmtpp_cfg),
            "encoder_config": to_jsonable(encoder_cfg),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def validate_resume_magnitude_identity(
    *,
    resume_payload: dict[str, Any],
    current_cfg: RMTPPConfig,
) -> None:
    """Reject a direct-decoder resume checkpoint with stale normalization state."""
    if current_cfg.qty_decoder_mode not in {"direct_log_qty", "direct_raw_qty"}:
        return
    stored_cfg = resume_payload.get("rmtpp_config")
    if not isinstance(stored_cfg, dict):
        raise ValueError("Direct magnitude resume checkpoint has no rmtpp_config identity.")
    identity_fields = (
        "qty_decoder_mode",
        "magnitude_norm_mode",
        "magnitude_global_mean",
        "magnitude_global_var",
        "magnitude_global_std",
        "magnitude_sigma_floor",
        "magnitude_revin_eps",
        "magnitude_shrinkage_k",
        "magnitude_center_mode",
        "magnitude_revin_affine",
        "magnitude_stat_context_mode",
        "magnitude_encoder_gradient_mode",
        "magnitude_aux_loss_mode",
        "lambda_log_qty",
        "log_qty_huber_delta",
        "log_qty_floor",
    )
    current_values = to_jsonable(current_cfg)
    legacy_defaults = {
        "magnitude_encoder_gradient_mode": "coupled",
        "magnitude_aux_loss_mode": "none",
        "lambda_log_qty": 0.25,
        "log_qty_huber_delta": 1.0,
        "log_qty_floor": 1.0,
    }
    mismatched = [
        name
        for name in identity_fields
        if stored_cfg.get(name, legacy_defaults.get(name)) != current_values.get(name)
    ]
    if mismatched:
        raise ValueError(
            "Resume magnitude identity mismatch: " + ", ".join(mismatched)
        )


def train_one_run(
    *,
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
    marked_df: pl.DataFrame,
    marked_meta: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Train one model run and persist checkpoints/metrics.
    """
    run_paths = build_run_paths(cfg, run_cfg)
    summary_path = run_paths.metrics_dir / "summary.json"
    history_json_path = run_paths.metrics_dir / "history.json"
    history_parquet_path = run_paths.metrics_dir / "history.parquet"
    resume_checkpoint_path = run_paths.checkpoint_dir / "last_epoch_state.pt"
    log_path = run_paths.logs_dir / "train.log"
    manifest_path = run_paths.manifest_dir / "run_config.json"
    effective_marked_meta = attach_train_global_magnitude_stats(
        marked_df=marked_df,
        marked_meta=marked_meta,
        cfg=cfg,
        run_cfg=run_cfg,
    )

    if not cfg.force_rerun and cached_run_is_complete(
        cfg=cfg,
        run_cfg=run_cfg,
        run_paths=run_paths,
        effective_marked_meta=effective_marked_meta,
    ):
        with open(summary_path, "r", encoding="utf-8") as f:
            cached_summary = json.load(f)
        cached_summary.setdefault("value_head_mode", "shared")
        cached_summary.setdefault("qty_mark_gradient_mode", "coupled")
        cached_summary.setdefault("value_encoder_gradient_mode", "coupled")
        cached_summary.setdefault("marker_loss_mode", "ce")
        cached_summary.setdefault("lambda_ordinal", 0.0)
        cached_summary.setdefault("qty_decoder_mode", "mark_residual")
        cached_summary.setdefault("magnitude_norm_mode", "global")
        cached_summary.setdefault("lambda_magnitude", 1.0)
        cached_summary.setdefault("magnitude_encoder_gradient_mode", "coupled")
        cached_summary.setdefault("magnitude_aux_loss_mode", "none")
        cached_summary.setdefault("lambda_log_qty", 0.25)
        cached_summary.setdefault("log_qty_huber_delta", 1.0)
        cached_summary.setdefault("log_qty_floor", 1.0)
        cached_summary.setdefault("magnitude_revin_eps", 1e-5)
        cached_summary.setdefault("magnitude_shrinkage_k", 8.0)
        cached_summary.setdefault("magnitude_center_mode", "mean")
        cached_summary.setdefault("magnitude_revin_affine", False)
        cached_summary.setdefault("magnitude_stat_context_mode", "none")
        return cached_summary

    set_global_seed(run_cfg.seed)
    search_cfg = make_search_cfg(cfg, run_cfg.dataset_kind)
    training_cfg = make_training_cfg(cfg, run_cfg.dataset_kind)
    test_loader = None
    if getattr(cfg, "split_mode", "internal") == "fixed":
        train_loader, val_loader, test_loader = make_fixed_split_week_lookback_loaders(
            marked_df,
            training_cfg,
        )
        logger.info(
            "Using fixed chronological splits | dataset=%s train_samples=%s validation_samples=%s test_samples=%s "
            "| batch_size=%s max_seq_len=%s lookback=%s",
            run_cfg.dataset_name,
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(test_loader.dataset),
            training_cfg.batch_size,
            training_cfg.max_seq_len,
            training_cfg.lookback,
        )
    else:
        train_loader, val_loader = make_week_lookback_loaders(marked_df, training_cfg)
    if cfg.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        logger.info(
            "Direct magnitude stats | decoder=%s norm=%s domain=%s split=train "
            "count=%s mean=%.8f std=%.8f floor=%.8f",
            cfg.qty_decoder_mode,
            cfg.magnitude_norm_mode,
            effective_marked_meta["magnitude_domain"],
            effective_marked_meta["magnitude_train_event_count"],
            effective_marked_meta["magnitude_global_mean"],
            effective_marked_meta["magnitude_global_std"],
            effective_marked_meta["magnitude_sigma_floor"],
        )
    model, rmtpp_cfg, encoder_cfg = build_model(
        cfg=cfg,
        run_cfg=run_cfg,
        marked_meta=effective_marked_meta,
    )

    save_json(
        {
            "experiment_config": cfg,
            "effective_search_config": search_cfg,
            "run_config": run_cfg,
            "training_config": training_cfg,
            "rmtpp_config": rmtpp_cfg,
            "encoder_config": encoder_cfg,
            "marked_meta": effective_marked_meta,
            "loader_sample_counts": {
                "train": len(train_loader.dataset),
                "validation": len(val_loader.dataset),
                "test": len(test_loader.dataset) if test_loader is not None else None,
            },
        },
        manifest_path,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.lr)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_val_nll = float("inf")
    best_score_state: dict[str, torch.Tensor] | None = None
    best_val_nll_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1

    if not cfg.force_rerun and resume_checkpoint_path.exists():
        try:
            resume_payload = torch_load_checkpoint(
                resume_checkpoint_path,
                # Load resume payloads on CPU first. Model/optimizer states can
                # be moved by their load_state_dict paths, while RNG states must
                # remain CPU ByteTensors for PyTorch's RNG restore APIs.
                map_location="cpu",
            )
            validate_resume_magnitude_identity(
                resume_payload=resume_payload,
                current_cfg=rmtpp_cfg,
            )
            loaded_epoch = int(resume_payload.get("epoch", 0))
            if loaded_epoch > 0:
                model.load_state_dict(resume_payload["model_state_dict"])
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
                history = list(resume_payload.get("history", []))
                best_score = float(resume_payload.get("best_score", best_score))
                best_val_nll = float(resume_payload.get("best_val_nll", best_val_nll))
                best_score_state = resume_payload.get("best_score_state_dict")
                best_val_nll_state = resume_payload.get("best_val_nll_state_dict")
                restore_rng_state(resume_payload.get("rng_state"))
                start_epoch = min(loaded_epoch + 1, training_cfg.epochs + 1)
                logger.info(
                    "Resuming run from epoch checkpoint | dataset=%s model=%s candidate=%s seed=%s "
                    "last_epoch=%s target_epochs=%s",
                    run_cfg.dataset_name,
                    run_cfg.model_name,
                    run_cfg.candidate_name,
                    run_cfg.seed,
                    loaded_epoch,
                    training_cfg.epochs,
                )
        except Exception as exc:
            logger.warning(
                "Could not load resume checkpoint; restarting this run from epoch 1. path=%s error=%r",
                resume_checkpoint_path,
                exc,
            )

    with tee_training_output(log_path):
        if start_epoch > 1:
            print(
                f"[resume] loaded {resume_checkpoint_path} | "
                f"next_epoch={start_epoch} | target_epochs={training_cfg.epochs}"
            )

        for epoch in range(start_epoch, training_cfg.epochs + 1):
            model.train()
            running = 0.0
            steps = 0
            running_log_qty_aux = 0.0
            log_qty_aux_steps = 0

            for marks, dts, mask, _, values in train_loader:
                marks = marks.to(training_cfg.device)
                dts = dts.to(training_cfg.device)
                mask = mask.to(training_cfg.device)
                values = values.to(training_cfg.device) if values is not None else None
                out = model.nll(
                    marks,
                    dts,
                    values=values,
                    mask=mask,
                    loss_scope=cfg.train_loss_scope,
                )
                loss = compute_training_loss(model=model, out=out, training_cfg=training_cfg)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if training_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.grad_clip)
                optimizer.step()

                running += float(loss.item())
                steps += 1
                if "log_qty_aux_loss" in out:
                    running_log_qty_aux += float(out["log_qty_aux_loss"].item())
                    log_qty_aux_steps += 1

            train_loss = running / max(steps, 1)
            train_log_qty_aux_loss = (
                running_log_qty_aux / log_qty_aux_steps
                if log_qty_aux_steps > 0
                else float("nan")
            )
            val_metrics = eval_next_event_week_lookback(
                model,
                val_loader,
                training_cfg.device,
                target_only_nll=getattr(cfg, "split_mode", "internal") == "fixed",
            )
            score = score_from_metrics(val_metrics)
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_log_qty_aux_loss": float(train_log_qty_aux_loss),
                "score": float(score),
                **{
                    key: float(value) if isinstance(value, (int, float, np.floating)) else value
                    for key, value in val_metrics.items()
                },
            }
            history.append(epoch_record)

            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.8f} | "
                f"score={score:.8f} | val_nll={val_metrics['val_nll']:.8f} | "
                f"val_acc={val_metrics['mark_acc']:.8f} | "
                f"val_dt_mae={val_metrics['dt_mae']:.8f} | "
                f"val_qty_mae={val_metrics['qty_mae']:.8f} | "
                f"train_log_qty_aux={train_log_qty_aux_loss:.8f} | "
                f"val_log_qty_aux={val_metrics['val_log_qty_aux_loss']:.8f}"
            )

            if score > best_score:
                best_score = float(score)
                best_score_state = clone_state_dict(model)

            val_nll = finite_or_default(val_metrics["val_nll"], float("inf"))
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_val_nll_state = clone_state_dict(model)

            # Keep the visible history files fresh for monitoring, and keep a
            # separate full resume checkpoint for interrupted long GPU jobs.
            pl.DataFrame(history).write_parquet(history_parquet_path)
            save_json({"history": history}, history_json_path)
            save_epoch_resume_checkpoint(
                path=resume_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                history=history,
                best_score=best_score,
                best_val_nll=best_val_nll,
                best_score_state=best_score_state,
                best_val_nll_state=best_val_nll_state,
                cfg=cfg,
                run_cfg=run_cfg,
                training_cfg=training_cfg,
                rmtpp_cfg=rmtpp_cfg,
                encoder_cfg=encoder_cfg,
            )

    final_state = clone_state_dict(model)
    best_score_state = best_score_state or final_state
    best_val_nll_state = best_val_nll_state or final_state

    pl.DataFrame(history).write_parquet(history_parquet_path)
    save_json({"history": history}, history_json_path)

    summary = {
        "status": "success",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "model_name": canonical_model_name(run_cfg.model_name),
        "candidate_name": run_cfg.candidate_name,
        # Keep the old column name temporarily so older analysis notebooks do
        # not break while the unified runner is introduced.
        "titan_candidate_name": run_cfg.candidate_name,
        "seed": int(run_cfg.seed),
        "epochs": int(run_cfg.epochs),
        "lr": float(cfg.lr),
        "batch_size": int(training_cfg.batch_size),
        "lookback_weeks": int(training_cfg.lookback),
        "max_seq_len": int(training_cfg.max_seq_len),
        "scale_base": float(run_cfg.scale_base),
        "analysis_scale_base": float(cfg.analysis_scale_base),
        "analysis_tail_order": int(cfg.analysis_tail_order),
        "titan_profile": run_cfg.titan_profile,
        "loss_mode": cfg.loss_mode,
        "split_mode": cfg.split_mode,
        "value_head_activation": cfg.value_head_activation,
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
        "magnitude_sigma_floor": float(cfg.magnitude_sigma_floor),
        "magnitude_effective_sigma_floor": effective_marked_meta.get("magnitude_sigma_floor"),
        "magnitude_revin_eps": float(cfg.magnitude_revin_eps),
        "magnitude_shrinkage_k": float(cfg.magnitude_shrinkage_k),
        "magnitude_center_mode": cfg.magnitude_center_mode,
        "magnitude_revin_affine": bool(cfg.magnitude_revin_affine),
        "magnitude_stat_context_mode": cfg.magnitude_stat_context_mode,
        "magnitude_domain": effective_marked_meta.get("magnitude_domain"),
        "magnitude_exp_clamp_min": float(cfg.magnitude_exp_clamp_min),
        "magnitude_exp_clamp_max": float(cfg.magnitude_exp_clamp_max),
        "magnitude_stats_source_split": effective_marked_meta.get("magnitude_stats_source_split"),
        "magnitude_train_event_count": effective_marked_meta.get("magnitude_train_event_count"),
        "magnitude_global_mean": effective_marked_meta.get("magnitude_global_mean"),
        "magnitude_global_var": effective_marked_meta.get("magnitude_global_var"),
        "magnitude_global_std": effective_marked_meta.get("magnitude_global_std"),
        "value_input_mode": cfg.value_input_mode,
        "value_input_emb_dim": int(cfg.value_input_emb_dim),
        "train_loss_scope": cfg.train_loss_scope,
        "lambda_dt": float(cfg.lambda_dt),
        "test_time_memory": cfg.test_time_memory,
        "run_dir": str(run_paths.run_dir),
        "resume_checkpoint_path": str(resume_checkpoint_path),
        "best_score_checkpoint_path": str(run_paths.checkpoint_dir / "best_score_model.pt"),
        "best_val_nll_checkpoint_path": str(run_paths.checkpoint_dir / "best_val_nll_model.pt"),
        "final_checkpoint_path": str(run_paths.checkpoint_dir / "final_model.pt"),
        "num_marks": int(effective_marked_meta["num_marks"]),
        "max_order": int(effective_marked_meta["max_order"]),
        "series_count": int(effective_marked_meta["series_count"]),
        "rmtpp_rnn_type": cfg.rmtpp_rnn_type,
        "rmtpp_hidden_dim": int(rmtpp_cfg.rnn_hidden_dim),
        "rmtpp_mark_emb_dim": int(rmtpp_cfg.mark_emb_dim),
        **flatten_candidate(run_cfg.candidate),
        **summarize_history(history),
    }

    state_by_selection = {
        "best_score": best_score_state,
        "best_val_nll": best_val_nll_state,
        "final": final_state,
    }
    for selection, state in state_by_selection.items():
        save_checkpoint(
            path=run_paths.checkpoint_dir / f"{selection}_model.pt",
            model_state=state,
            cfg=cfg,
            run_cfg=run_cfg,
            training_cfg=training_cfg,
            rmtpp_cfg=rmtpp_cfg,
            encoder_cfg=encoder_cfg,
            summary=summary,
            selection=selection,
        )

    for selection in cfg.eval_selections:
        selected_state = state_by_selection[selection]
        model.load_state_dict(selected_state)
        scale_df = evaluate_scale_wise_qty(
            model=model,
            val_loader=val_loader,
            device=training_cfg.device,
            analysis_scale_base=cfg.analysis_scale_base,
            analysis_tail_order=cfg.analysis_tail_order,
        ).with_columns([
            pl.lit(run_cfg.dataset_name).alias("dataset_name"),
            pl.lit(run_cfg.dataset_kind).alias("dataset_kind"),
            pl.lit(canonical_model_name(run_cfg.model_name)).alias("model_name"),
            pl.lit(run_cfg.candidate_name).alias("candidate_name"),
            pl.lit(run_cfg.candidate_name).alias("titan_candidate_name"),
            pl.lit(run_cfg.seed).alias("seed"),
            pl.lit(cfg.value_head_mode).alias("value_head_mode"),
            pl.lit(cfg.qty_mark_gradient_mode).alias("qty_mark_gradient_mode"),
            pl.lit(cfg.value_encoder_gradient_mode).alias("value_encoder_gradient_mode"),
            pl.lit(cfg.marker_loss_mode).alias("marker_loss_mode"),
            pl.lit(float(cfg.lambda_ordinal)).alias("lambda_ordinal"),
            pl.lit(cfg.qty_decoder_mode).alias("qty_decoder_mode"),
            pl.lit(cfg.magnitude_norm_mode).alias("magnitude_norm_mode"),
            pl.lit(float(cfg.lambda_magnitude)).alias("lambda_magnitude"),
            *magnitude_artifact_columns(cfg),
            pl.lit(selection).alias("selection"),
            pl.lit("validation").alias("eval_split"),
            pl.lit(run_cfg.scale_base).alias("model_scale_base"),
        ])
        csv_path, parquet_path = scale_metric_paths(run_paths, selection)
        scale_df.write_csv(csv_path)
        scale_df.write_parquet(parquet_path)

        confusion_df = evaluate_mark_confusion(
            model=model,
            val_loader=val_loader,
            device=training_cfg.device,
        ).with_columns([
            pl.lit(run_cfg.dataset_name).alias("dataset_name"),
            pl.lit(run_cfg.dataset_kind).alias("dataset_kind"),
            pl.lit(canonical_model_name(run_cfg.model_name)).alias("model_name"),
            pl.lit(run_cfg.candidate_name).alias("candidate_name"),
            pl.lit(run_cfg.candidate_name).alias("titan_candidate_name"),
            pl.lit(run_cfg.seed).alias("seed"),
            pl.lit(cfg.value_head_mode).alias("value_head_mode"),
            pl.lit(cfg.qty_mark_gradient_mode).alias("qty_mark_gradient_mode"),
            pl.lit(cfg.value_encoder_gradient_mode).alias("value_encoder_gradient_mode"),
            pl.lit(cfg.marker_loss_mode).alias("marker_loss_mode"),
            pl.lit(float(cfg.lambda_ordinal)).alias("lambda_ordinal"),
            pl.lit(cfg.qty_decoder_mode).alias("qty_decoder_mode"),
            pl.lit(cfg.magnitude_norm_mode).alias("magnitude_norm_mode"),
            pl.lit(float(cfg.lambda_magnitude)).alias("lambda_magnitude"),
            pl.lit(selection).alias("selection"),
            pl.lit("validation").alias("eval_split"),
            pl.lit(run_cfg.scale_base).alias("model_scale_base"),
        ])
        confusion_csv_path, confusion_parquet_path = confusion_metric_paths(
            run_paths,
            selection,
            "validation",
        )
        confusion_df.write_csv(confusion_csv_path)
        confusion_df.write_parquet(confusion_parquet_path)
        mark_class_df = summarize_mark_confusion(
            confusion_df,
            num_real_marks=int(rmtpp_cfg.num_marks - 1),
        ).with_columns([
            pl.lit(run_cfg.dataset_name).alias("dataset_name"),
            pl.lit(canonical_model_name(run_cfg.model_name)).alias("model_name"),
            pl.lit(run_cfg.candidate_name).alias("candidate_name"),
            pl.lit(run_cfg.seed).alias("seed"),
            pl.lit(cfg.marker_loss_mode).alias("marker_loss_mode"),
            pl.lit(float(cfg.lambda_ordinal)).alias("lambda_ordinal"),
            pl.lit(cfg.qty_decoder_mode).alias("qty_decoder_mode"),
            pl.lit(cfg.magnitude_norm_mode).alias("magnitude_norm_mode"),
            pl.lit(float(cfg.lambda_magnitude)).alias("lambda_magnitude"),
            pl.lit(selection).alias("selection"),
            pl.lit("validation").alias("eval_split"),
        ])
        mark_class_csv_path, mark_class_parquet_path = mark_class_metric_paths(
            run_paths,
            selection,
            "validation",
        )
        mark_class_df.write_csv(mark_class_csv_path)
        mark_class_df.write_parquet(mark_class_parquet_path)

        if test_loader is not None:
            test_metrics = eval_next_event_week_lookback(
                model,
                test_loader,
                training_cfg.device,
                target_only_nll=True,
            )
            test_score = score_from_metrics(test_metrics)
            test_row = {
                "dataset_name": run_cfg.dataset_name,
                "dataset_kind": run_cfg.dataset_kind,
                "model_name": canonical_model_name(run_cfg.model_name),
                "candidate_name": run_cfg.candidate_name,
                "titan_candidate_name": run_cfg.candidate_name,
                "seed": int(run_cfg.seed),
                "value_head_mode": cfg.value_head_mode,
                "qty_mark_gradient_mode": cfg.qty_mark_gradient_mode,
                "value_encoder_gradient_mode": cfg.value_encoder_gradient_mode,
                "marker_loss_mode": cfg.marker_loss_mode,
                "lambda_ordinal": float(cfg.lambda_ordinal),
                "qty_decoder_mode": cfg.qty_decoder_mode,
                "magnitude_norm_mode": cfg.magnitude_norm_mode,
                "lambda_magnitude": float(cfg.lambda_magnitude),
                **magnitude_artifact_identity(cfg),
                "selection": selection,
                "score": float(test_score),
                **{
                    key: float(value) if isinstance(value, (int, float, np.floating)) else value
                    for key, value in test_metrics.items()
                },
            }
            test_json_path, test_csv_path, test_parquet_path = test_metric_paths(run_paths, selection)
            save_json(test_row, test_json_path)
            test_df = pl.DataFrame([{key: to_jsonable(value) for key, value in test_row.items()}])
            test_df.write_csv(test_csv_path)
            test_df.write_parquet(test_parquet_path)

            # Keep concise test metrics on the run summary so leaderboard rows can
            # directly expose the final held-out performance used in the paper.
            summary[f"test_{selection}_score"] = float(test_score)
            summary[f"test_{selection}_nll"] = float(test_metrics["val_nll"])
            summary[f"test_{selection}_nll_marker"] = float(test_metrics["val_nll_marker"])
            summary[f"test_{selection}_nll_time"] = float(test_metrics["val_nll_time"])
            summary[f"test_{selection}_ordinal_marker_loss"] = float(
                test_metrics["val_ordinal_marker_loss"]
            )
            summary[f"test_{selection}_value_loss"] = float(test_metrics["val_value_loss"])
            summary[f"test_{selection}_magnitude_loss"] = float(
                test_metrics["val_magnitude_loss"]
            )
            summary[f"test_{selection}_log_qty_aux_loss"] = float(
                test_metrics["val_log_qty_aux_loss"]
            )
            summary[f"test_{selection}_log_qty_mae"] = float(test_metrics["log_qty_mae"])
            summary[f"test_{selection}_log_qty_rmse"] = float(test_metrics["log_qty_rmse"])
            summary[f"test_{selection}_qty_mae"] = float(test_metrics["qty_mae"])
            summary[f"test_{selection}_value_mae"] = float(test_metrics["value_mae"])
            summary[f"test_{selection}_dt_mae"] = float(test_metrics["dt_mae"])
            summary[f"test_{selection}_dt_rmse"] = float(test_metrics["dt_rmse"])
            summary[f"test_{selection}_mark_acc"] = float(test_metrics["mark_acc"])
            for metric_name in (
                "mark_balanced_accuracy",
                "mark_macro_f1",
                "mark_mae",
                "mark_adjacent_accuracy",
                "mark_pred_0_share",
                "mark_0_recall",
                "mark_1_recall",
            ):
                summary[f"test_{selection}_{metric_name}"] = float(test_metrics[metric_name])
            summary[f"test_{selection}_total"] = int(test_metrics["_total"])
            summary[f"test_{selection}_nll_steps"] = float(test_metrics["_nll_steps"])

            test_scale_df = evaluate_scale_wise_qty(
                model=model,
                val_loader=test_loader,
                device=training_cfg.device,
                analysis_scale_base=cfg.analysis_scale_base,
                analysis_tail_order=cfg.analysis_tail_order,
            ).with_columns([
                pl.lit(run_cfg.dataset_name).alias("dataset_name"),
                pl.lit(run_cfg.dataset_kind).alias("dataset_kind"),
                pl.lit(canonical_model_name(run_cfg.model_name)).alias("model_name"),
                pl.lit(run_cfg.candidate_name).alias("candidate_name"),
                pl.lit(run_cfg.candidate_name).alias("titan_candidate_name"),
                pl.lit(run_cfg.seed).alias("seed"),
                pl.lit(cfg.value_head_mode).alias("value_head_mode"),
                pl.lit(cfg.qty_mark_gradient_mode).alias("qty_mark_gradient_mode"),
                pl.lit(cfg.value_encoder_gradient_mode).alias("value_encoder_gradient_mode"),
                pl.lit(cfg.marker_loss_mode).alias("marker_loss_mode"),
                pl.lit(float(cfg.lambda_ordinal)).alias("lambda_ordinal"),
                pl.lit(cfg.qty_decoder_mode).alias("qty_decoder_mode"),
                pl.lit(cfg.magnitude_norm_mode).alias("magnitude_norm_mode"),
                pl.lit(float(cfg.lambda_magnitude)).alias("lambda_magnitude"),
                *magnitude_artifact_columns(cfg),
                pl.lit(selection).alias("selection"),
                pl.lit("test").alias("eval_split"),
                pl.lit(run_cfg.scale_base).alias("model_scale_base"),
            ])
            test_scale_csv_path, test_scale_parquet_path = test_scale_metric_paths(run_paths, selection)
            test_scale_df.write_csv(test_scale_csv_path)
            test_scale_df.write_parquet(test_scale_parquet_path)

            test_confusion_df = evaluate_mark_confusion(
                model=model,
                val_loader=test_loader,
                device=training_cfg.device,
            ).with_columns([
                pl.lit(run_cfg.dataset_name).alias("dataset_name"),
                pl.lit(run_cfg.dataset_kind).alias("dataset_kind"),
                pl.lit(canonical_model_name(run_cfg.model_name)).alias("model_name"),
                pl.lit(run_cfg.candidate_name).alias("candidate_name"),
                pl.lit(run_cfg.candidate_name).alias("titan_candidate_name"),
                pl.lit(run_cfg.seed).alias("seed"),
                pl.lit(cfg.value_head_mode).alias("value_head_mode"),
                pl.lit(cfg.qty_mark_gradient_mode).alias("qty_mark_gradient_mode"),
                pl.lit(cfg.value_encoder_gradient_mode).alias("value_encoder_gradient_mode"),
                pl.lit(cfg.marker_loss_mode).alias("marker_loss_mode"),
                pl.lit(float(cfg.lambda_ordinal)).alias("lambda_ordinal"),
                pl.lit(cfg.qty_decoder_mode).alias("qty_decoder_mode"),
                pl.lit(cfg.magnitude_norm_mode).alias("magnitude_norm_mode"),
                pl.lit(float(cfg.lambda_magnitude)).alias("lambda_magnitude"),
                pl.lit(selection).alias("selection"),
                pl.lit("test").alias("eval_split"),
                pl.lit(run_cfg.scale_base).alias("model_scale_base"),
            ])
            test_confusion_csv_path, test_confusion_parquet_path = confusion_metric_paths(
                run_paths,
                selection,
                "test",
            )
            test_confusion_df.write_csv(test_confusion_csv_path)
            test_confusion_df.write_parquet(test_confusion_parquet_path)
            test_mark_class_df = summarize_mark_confusion(
                test_confusion_df,
                num_real_marks=int(rmtpp_cfg.num_marks - 1),
            ).with_columns([
                pl.lit(run_cfg.dataset_name).alias("dataset_name"),
                pl.lit(canonical_model_name(run_cfg.model_name)).alias("model_name"),
                pl.lit(run_cfg.candidate_name).alias("candidate_name"),
                pl.lit(run_cfg.seed).alias("seed"),
                pl.lit(cfg.marker_loss_mode).alias("marker_loss_mode"),
                pl.lit(float(cfg.lambda_ordinal)).alias("lambda_ordinal"),
                pl.lit(cfg.qty_decoder_mode).alias("qty_decoder_mode"),
                pl.lit(cfg.magnitude_norm_mode).alias("magnitude_norm_mode"),
                pl.lit(float(cfg.lambda_magnitude)).alias("lambda_magnitude"),
                pl.lit(selection).alias("selection"),
                pl.lit("test").alias("eval_split"),
            ])
            test_mark_class_csv_path, test_mark_class_parquet_path = mark_class_metric_paths(
                run_paths,
                selection,
                "test",
            )
            test_mark_class_df.write_csv(test_mark_class_csv_path)
            test_mark_class_df.write_parquet(test_mark_class_parquet_path)

        if cfg.test_time_memory == "contextual" and canonical_model_name(run_cfg.model_name) == "titantpp":
            ttm_metrics = evaluate_titantpp_contextual_ttm(
                model=model,
                marked_df=marked_df,
                training_cfg=training_cfg,
                selection=selection,
            )
            json_path, ttm_csv_path, ttm_parquet_path = ttm_metric_paths(run_paths, selection)
            save_json(ttm_metrics, json_path)
            ttm_df = pl.DataFrame([{key: to_jsonable(value) for key, value in ttm_metrics.items()}])
            ttm_df.write_csv(ttm_csv_path)
            ttm_df.write_parquet(ttm_parquet_path)

            prefix = f"ttm_contextual_{selection}"
            for metric_name in (
                "score",
                "val_nll",
                "val_nll_time",
                "val_nll_marker",
                "val_value_loss",
                "mark_acc",
                "dt_mae",
                "dt_rmse",
                "value_mae",
                "qty_mae",
                "_total",
                "_correct",
                "_nll_steps",
                "evaluated_series",
                "warmup_events",
                "contextual_update_events",
            ):
                summary[f"{prefix}_{metric_name}"] = ttm_metrics[metric_name]

    save_json(summary, summary_path)
    logger.info(
        "Finished run | dataset=%s model=%s candidate=%s seed=%s best_val_nll=%.6f epoch=%s",
        run_cfg.dataset_name,
        run_cfg.model_name,
        run_cfg.candidate_name,
        run_cfg.seed,
        summary["best_val_nll"],
        summary["best_val_nll_epoch"],
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def build_error_row(cfg: ExperimentConfig, run_cfg: RunConfig, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "model_name": canonical_model_name(run_cfg.model_name),
        "candidate_name": run_cfg.candidate_name,
        "titan_candidate_name": run_cfg.candidate_name,
        "seed": int(run_cfg.seed),
        "epochs": int(run_cfg.epochs),
        "scale_base": float(run_cfg.scale_base),
        "titan_profile": run_cfg.titan_profile,
        "split_mode": "unavailable_after_failure",
        "value_head_activation": "unavailable_after_failure",
        "value_head_mode": cfg.value_head_mode,
        "qty_mark_gradient_mode": cfg.qty_mark_gradient_mode,
        "value_encoder_gradient_mode": cfg.value_encoder_gradient_mode,
        "marker_loss_mode": cfg.marker_loss_mode,
        "lambda_ordinal": float(cfg.lambda_ordinal),
        "qty_decoder_mode": cfg.qty_decoder_mode,
        "magnitude_norm_mode": cfg.magnitude_norm_mode,
        "lambda_magnitude": float(cfg.lambda_magnitude),
        **magnitude_artifact_identity(cfg),
        "test_time_memory": "unavailable_after_failure",
        "error": repr(exc),
        **flatten_candidate(run_cfg.candidate),
    }


def selected_model_jobs(cfg: ExperimentConfig, profile: dict[str, Any]) -> list[tuple[str, str, Any]]:
    """
    Resolve model/candidate jobs for one dataset profile.
    """
    models = tuple(canonical_model_name(model) for model in cfg.models)
    titan_candidates = default_titan_candidates()
    thp_candidates = default_thp_candidates()
    profile_titan = find_candidate_by_name(titan_candidates, str(profile["candidate_name"]))
    jobs: list[tuple[str, str, Any]] = []

    if "rmtpp" in models:
        hidden = int(cfg.rmtpp_hidden_dim or profile_titan.d_model)
        candidate = make_rmtpp_proxy_candidate(hidden, cfg.rmtpp_rnn_type)
        jobs.append(("rmtpp", candidate.name, candidate))

    if "titantpp" in models:
        names = cfg.titan_candidates or (str(profile["candidate_name"]),)
        for name in names:
            candidate = find_candidate_by_name(titan_candidates, name)
            jobs.append(("titantpp", candidate.name, candidate))

    if "thp" in models:
        names = cfg.thp_candidates or ("base",)
        for name in names:
            candidate = find_candidate_by_name(thp_candidates, name)
            jobs.append(("thp", candidate.name, candidate))

    unsupported = sorted(set(models) - {"rmtpp", "titantpp", "thp"})
    if unsupported:
        raise ValueError(f"Unsupported models: {unsupported}")
    return jobs


def run_long_epoch_benchmark(
    *,
    cfg: ExperimentConfig,
    dataset_specs: list[DatasetSpec],
    profile_map: dict[str, dict[str, Any]],
    marked_cache: dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Run all dataset/model/candidate/seed jobs for long-epoch validation.
    """
    rows: list[dict[str, Any]] = []
    leaderboard_dir = ensure_dir(Path(cfg.base_dir) / "leaderboard")
    path_prefix = leaderboard_dir / "runs"
    total_runs = sum(
        len(cfg.seeds) * len(selected_model_jobs(cfg, profile_map[spec.name]))
        for spec in dataset_specs
    )
    completed = 0

    for spec in dataset_specs:
        profile = profile_map[spec.name]
        scale_base = float(profile["scale_base"])
        marked_df, marked_meta = marked_cache[(spec.name, scale_base)]
        jobs = selected_model_jobs(cfg, profile)
        for seed in cfg.seeds:
            for model_name, candidate_name, candidate in jobs:
                completed += 1
                logger.info(
                    "Run %s/%s | dataset=%s model=%s candidate=%s base=%s seed=%s",
                    completed,
                    total_runs,
                    spec.name,
                    model_name,
                    candidate_name,
                    scale_base,
                    seed,
                )
                run_cfg = RunConfig(
                    dataset_name=spec.name,
                    dataset_kind=spec.kind,
                    model_name=model_name,
                    candidate_name=candidate_name,
                    candidate=candidate,
                    seed=seed,
                    epochs=cfg.epochs,
                    scale_base=scale_base,
                    titan_profile=cfg.titan_profile,
                )
                try:
                    row = train_one_run(
                        cfg=cfg,
                        run_cfg=run_cfg,
                        marked_df=marked_df,
                        marked_meta=marked_meta,
                        logger=logger,
                    )
                except Exception as exc:
                    row = build_error_row(cfg, run_cfg, exc)
                    logger.exception(
                        "Run failed | dataset=%s model=%s candidate=%s seed=%s",
                        spec.name,
                        model_name,
                        candidate_name,
                        seed,
                    )
                    if cfg.stop_on_error:
                        raise
                rows.append(row)
                persist_rows(rows, path_prefix)
    return rows


def load_all_histories(run_rows: list[dict[str, Any]]) -> pl.DataFrame:
    history_rows: list[dict[str, Any]] = []
    for row in run_rows:
        if row.get("status") != "success":
            continue
        history_path = Path(str(row["run_dir"])) / "metrics" / "history.json"
        if not history_path.exists():
            continue
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f).get("history", [])
        for epoch_row in history:
            history_rows.append({
                "dataset_name": row["dataset_name"],
                "model_name": row["model_name"],
                "candidate_name": row["candidate_name"],
                "titan_candidate_name": row["candidate_name"],
                "value_head_mode": row.get("value_head_mode", "shared"),
                "qty_mark_gradient_mode": row.get("qty_mark_gradient_mode", "coupled"),
                "value_encoder_gradient_mode": row.get(
                    "value_encoder_gradient_mode", "coupled"
                ),
                "marker_loss_mode": row.get("marker_loss_mode", "ce"),
                "lambda_ordinal": float(row.get("lambda_ordinal", 0.0)),
                "qty_decoder_mode": row.get("qty_decoder_mode", "mark_residual"),
                "magnitude_norm_mode": row.get("magnitude_norm_mode", "global"),
                "lambda_magnitude": float(row.get("lambda_magnitude", 1.0)),
                "magnitude_encoder_gradient_mode": row.get(
                    "magnitude_encoder_gradient_mode", "coupled"
                ),
                "magnitude_aux_loss_mode": row.get("magnitude_aux_loss_mode", "none"),
                "lambda_log_qty": float(row.get("lambda_log_qty", 0.25)),
                "log_qty_huber_delta": float(row.get("log_qty_huber_delta", 1.0)),
                "log_qty_floor": float(row.get("log_qty_floor", 1.0)),
                "magnitude_domain": row.get("magnitude_domain", "mark_residual"),
                "magnitude_sigma_floor": float(
                    row.get("magnitude_sigma_floor", 0.0014535461338152059)
                ),
                "magnitude_revin_eps": float(row.get("magnitude_revin_eps", 1e-5)),
                "magnitude_shrinkage_k": float(row.get("magnitude_shrinkage_k", 8.0)),
                "magnitude_center_mode": row.get("magnitude_center_mode", "mean"),
                "magnitude_revin_affine": bool(row.get("magnitude_revin_affine", False)),
                "magnitude_stat_context_mode": row.get(
                    "magnitude_stat_context_mode", "none"
                ),
                "seed": int(row["seed"]),
                "epoch": int(epoch_row["epoch"]),
                "score": float(epoch_row["score"]),
                "val_nll": float(epoch_row["val_nll"]),
                "val_nll_marker": float(epoch_row.get("val_nll_marker", float("nan"))),
                "val_nll_time": float(epoch_row.get("val_nll_time", float("nan"))),
                "val_ordinal_marker_loss": float(
                    epoch_row.get("val_ordinal_marker_loss", float("nan"))
                ),
                "qty_mae": float(epoch_row["qty_mae"]),
                "qty_rmse": float(epoch_row.get("qty_rmse", float("nan"))),
                "qty_wape": float(epoch_row.get("qty_wape", float("nan"))),
                "log_qty_mae": float(epoch_row.get("log_qty_mae", float("nan"))),
                "log_qty_rmse": float(epoch_row.get("log_qty_rmse", float("nan"))),
                "val_magnitude_loss": float(
                    epoch_row.get("val_magnitude_loss", float("nan"))
                ),
                "train_log_qty_aux_loss": float(
                    epoch_row.get("train_log_qty_aux_loss", float("nan"))
                ),
                "val_log_qty_aux_loss": float(
                    epoch_row.get("val_log_qty_aux_loss", float("nan"))
                ),
                "dt_mae": float(epoch_row["dt_mae"]),
                "mark_acc": float(epoch_row["mark_acc"]),
                "mark_balanced_accuracy": float(
                    epoch_row.get("mark_balanced_accuracy", float("nan"))
                ),
                "mark_macro_f1": float(epoch_row.get("mark_macro_f1", float("nan"))),
                "mark_mae": float(epoch_row.get("mark_mae", float("nan"))),
                "mark_adjacent_accuracy": float(
                    epoch_row.get("mark_adjacent_accuracy", float("nan"))
                ),
                "mark_0_recall": float(epoch_row.get("mark_0_recall", float("nan"))),
                "mark_1_recall": float(epoch_row.get("mark_1_recall", float("nan"))),
                "train_loss": float(epoch_row["train_loss"]),
            })
    return pl.DataFrame(history_rows) if history_rows else pl.DataFrame()


def load_all_scale_metrics(run_rows: list[dict[str, Any]], selections: Iterable[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for row in run_rows:
        if row.get("status") != "success":
            continue
        metrics_dir = Path(str(row["run_dir"])) / "metrics"
        for selection in selections:
            parquet_path = metrics_dir / f"scale_wise_{selection}.parquet"
            csv_path = metrics_dir / f"scale_wise_{selection}.csv"
            if parquet_path.exists():
                frames.append(pl.read_parquet(parquet_path))
            elif csv_path.exists():
                frames.append(pl.read_csv(csv_path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def load_all_test_metrics(run_rows: list[dict[str, Any]], selections: Iterable[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for row in run_rows:
        if row.get("status") != "success":
            continue
        metrics_dir = Path(str(row["run_dir"])) / "metrics"
        for selection in selections:
            parquet_path = metrics_dir / f"test_metrics_{selection}.parquet"
            csv_path = metrics_dir / f"test_metrics_{selection}.csv"
            if parquet_path.exists():
                frames.append(pl.read_parquet(parquet_path))
            elif csv_path.exists():
                frames.append(pl.read_csv(csv_path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def load_all_test_scale_metrics(run_rows: list[dict[str, Any]], selections: Iterable[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for row in run_rows:
        if row.get("status") != "success":
            continue
        metrics_dir = Path(str(row["run_dir"])) / "metrics"
        for selection in selections:
            parquet_path = metrics_dir / f"test_scale_wise_{selection}.parquet"
            csv_path = metrics_dir / f"test_scale_wise_{selection}.csv"
            if parquet_path.exists():
                frames.append(pl.read_parquet(parquet_path))
            elif csv_path.exists():
                frames.append(pl.read_csv(csv_path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def with_magnitude_identity_defaults(frame: pl.DataFrame) -> pl.DataFrame:
    """Backfill new identity columns when aggregating older artifacts."""
    defaults = {
        "magnitude_encoder_gradient_mode": "coupled",
        "magnitude_aux_loss_mode": "none",
        "lambda_log_qty": 0.25,
        "log_qty_huber_delta": 1.0,
        "log_qty_floor": 1.0,
        "magnitude_sigma_floor": 0.0014535461338152059,
        "magnitude_revin_eps": 1e-5,
        "magnitude_shrinkage_k": 8.0,
        "magnitude_center_mode": "mean",
        "magnitude_revin_affine": False,
        "magnitude_stat_context_mode": "none",
    }
    for name, value in defaults.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(value).alias(name))
    if "magnitude_domain" not in frame.columns:
        frame = frame.with_columns(
            pl.when(pl.col("qty_decoder_mode") == "direct_raw_qty")
            .then(pl.lit("raw_qty"))
            .when(pl.col("qty_decoder_mode") == "direct_log_qty")
            .then(pl.lit("log2_qty"))
            .otherwise(pl.lit("mark_residual"))
            .alias("magnitude_domain")
        )
    return frame


MAGNITUDE_GROUP_COLUMNS = [
    "magnitude_domain",
    "magnitude_encoder_gradient_mode",
    "magnitude_aux_loss_mode",
    "lambda_log_qty",
    "log_qty_huber_delta",
    "log_qty_floor",
    "magnitude_sigma_floor",
    "magnitude_revin_eps",
    "magnitude_shrinkage_k",
    "magnitude_center_mode",
    "magnitude_revin_affine",
    "magnitude_stat_context_mode",
]


def aggregate_test_metrics(test_df: pl.DataFrame) -> pl.DataFrame:
    if test_df.height == 0:
        return pl.DataFrame()
    if "value_head_mode" not in test_df.columns:
        test_df = test_df.with_columns(pl.lit("shared").alias("value_head_mode"))
    if "qty_mark_gradient_mode" not in test_df.columns:
        test_df = test_df.with_columns(
            pl.lit("coupled").alias("qty_mark_gradient_mode")
        )
    if "value_encoder_gradient_mode" not in test_df.columns:
        test_df = test_df.with_columns(
            pl.lit("coupled").alias("value_encoder_gradient_mode")
        )
    if "marker_loss_mode" not in test_df.columns:
        test_df = test_df.with_columns(pl.lit("ce").alias("marker_loss_mode"))
    if "lambda_ordinal" not in test_df.columns:
        test_df = test_df.with_columns(pl.lit(0.0).alias("lambda_ordinal"))
    if "qty_decoder_mode" not in test_df.columns:
        test_df = test_df.with_columns(pl.lit("mark_residual").alias("qty_decoder_mode"))
    if "magnitude_norm_mode" not in test_df.columns:
        test_df = test_df.with_columns(pl.lit("global").alias("magnitude_norm_mode"))
    if "lambda_magnitude" not in test_df.columns:
        test_df = test_df.with_columns(pl.lit(1.0).alias("lambda_magnitude"))
    test_df = with_magnitude_identity_defaults(test_df)
    for metric_name in (
        "val_ordinal_marker_loss",
        "mark_balanced_accuracy",
        "mark_macro_f1",
        "mark_mae",
        "mark_adjacent_accuracy",
        "mark_pred_0_share",
        "mark_0_recall",
        "mark_1_recall",
        "log_qty_mae",
        "log_qty_rmse",
        "val_magnitude_loss",
        "val_log_qty_aux_loss",
        *RAW_MAGNITUDE_DIAGNOSTIC_NAMES,
    ):
        if metric_name not in test_df.columns:
            test_df = test_df.with_columns(pl.lit(float("nan")).alias(metric_name))
    return (
        test_df.group_by([
            "dataset_name",
            "model_name",
            "candidate_name",
            "value_head_mode",
            "qty_mark_gradient_mode",
            "value_encoder_gradient_mode",
            "marker_loss_mode",
            "lambda_ordinal",
            "qty_decoder_mode",
            "magnitude_norm_mode",
            "lambda_magnitude",
            *MAGNITUDE_GROUP_COLUMNS,
            "selection",
        ])
        .agg([
            pl.len().alias("run_count"),
            pl.mean("score").alias("mean_test_score"),
            pl.std("score").fill_null(0.0).alias("std_test_score"),
            pl.mean("val_nll").alias("mean_test_nll"),
            pl.std("val_nll").fill_null(0.0).alias("std_test_nll"),
            pl.mean("val_nll_marker").alias("mean_test_nll_marker"),
            pl.mean("val_nll_time").alias("mean_test_nll_time"),
            pl.mean("val_ordinal_marker_loss").alias("mean_test_ordinal_marker_loss"),
            pl.mean("qty_mae").alias("mean_test_qty_mae"),
            pl.std("qty_mae").fill_null(0.0).alias("std_test_qty_mae"),
            pl.mean("log_qty_mae").alias("mean_test_log_qty_mae"),
            pl.mean("log_qty_rmse").alias("mean_test_log_qty_rmse"),
            pl.mean("val_magnitude_loss").alias("mean_test_magnitude_loss"),
            pl.mean("val_log_qty_aux_loss").alias("mean_test_log_qty_aux_loss"),
            pl.mean("dt_mae").alias("mean_test_dt_mae"),
            pl.mean("mark_acc").alias("mean_test_mark_acc"),
            pl.mean("mark_balanced_accuracy").alias("mean_test_mark_balanced_accuracy"),
            pl.mean("mark_macro_f1").alias("mean_test_mark_macro_f1"),
            pl.mean("mark_mae").alias("mean_test_mark_mae"),
            pl.mean("mark_adjacent_accuracy").alias("mean_test_mark_adjacent_accuracy"),
            pl.mean("mark_pred_0_share").alias("mean_test_mark_pred_0_share"),
            pl.mean("mark_0_recall").alias("mean_test_mark_0_recall"),
            pl.mean("mark_1_recall").alias("mean_test_mark_1_recall"),
            pl.mean("value_mae").alias("mean_test_value_mae"),
            pl.mean("_total").alias("mean_test_total"),
            pl.mean("_nll_steps").alias("mean_test_nll_steps"),
            *[
                pl.mean(metric_name).alias(f"mean_test_{metric_name}")
                for metric_name in RAW_MAGNITUDE_DIAGNOSTIC_NAMES
            ],
        ])
        .sort(["dataset_name", "selection", "model_name", "candidate_name"])
    )


def aggregate_run_rows(rows: list[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    run_df = pl.DataFrame([{key: to_jsonable(value) for key, value in row.items()} for row in rows])
    success_df = run_df.filter(pl.col("status") == "success")
    if success_df.height == 0:
        return run_df, pl.DataFrame()
    if "value_encoder_gradient_mode" not in success_df.columns:
        success_df = success_df.with_columns(
            pl.lit("coupled").alias("value_encoder_gradient_mode")
        )
    if "marker_loss_mode" not in success_df.columns:
        success_df = success_df.with_columns(pl.lit("ce").alias("marker_loss_mode"))
    if "lambda_ordinal" not in success_df.columns:
        success_df = success_df.with_columns(pl.lit(0.0).alias("lambda_ordinal"))
    if "qty_decoder_mode" not in success_df.columns:
        success_df = success_df.with_columns(pl.lit("mark_residual").alias("qty_decoder_mode"))
    if "magnitude_norm_mode" not in success_df.columns:
        success_df = success_df.with_columns(pl.lit("global").alias("magnitude_norm_mode"))
    if "lambda_magnitude" not in success_df.columns:
        success_df = success_df.with_columns(pl.lit(1.0).alias("lambda_magnitude"))
    success_df = with_magnitude_identity_defaults(success_df)

    agg_exprs = [
        pl.first("dataset_kind").alias("dataset_kind"),
        pl.first("scale_base").alias("scale_base"),
        pl.first("lr").alias("lr"),
        pl.first("batch_size").alias("batch_size"),
        pl.first("lookback_weeks").alias("lookback_weeks"),
        pl.first("max_seq_len").alias("max_seq_len"),
        pl.first("analysis_scale_base").alias("analysis_scale_base"),
        pl.first("titan_profile").alias("titan_profile"),
        pl.first("split_mode").alias("split_mode"),
        pl.first("value_head_activation").alias("value_head_activation"),
        pl.first("test_time_memory").alias("test_time_memory"),
        pl.len().alias("run_count"),
        pl.mean("best_val_nll").alias("mean_best_val_nll"),
        pl.std("best_val_nll").fill_null(0.0).alias("std_best_val_nll"),
        pl.mean("best_val_nll_epoch").alias("mean_best_val_nll_epoch"),
        pl.mean("best_val_nll_score").alias("mean_best_val_nll_score"),
        pl.mean("best_val_nll_qty_mae").alias("mean_best_val_nll_qty_mae"),
        pl.std("best_val_nll_qty_mae").fill_null(0.0).alias("std_best_val_nll_qty_mae"),
        pl.mean("best_val_nll_dt_mae").alias("mean_best_val_nll_dt_mae"),
        pl.mean("best_val_nll_mark_acc").alias("mean_best_val_nll_mark_acc"),
        pl.mean("best_score").alias("mean_best_score"),
        pl.std("best_score").fill_null(0.0).alias("std_best_score"),
        pl.mean("best_score_epoch").alias("mean_best_score_epoch"),
        pl.mean("best_score_val_nll").alias("mean_best_score_val_nll"),
        pl.mean("best_score_qty_mae").alias("mean_best_score_qty_mae"),
        pl.mean("final_val_nll").alias("mean_final_val_nll"),
        pl.mean("final_qty_mae").alias("mean_final_qty_mae"),
        pl.mean("final_score").alias("mean_final_score"),
        pl.mean("final_train_loss").alias("mean_final_train_loss"),
    ]
    for metric_name in (
        "best_val_nll_ordinal_marker_loss",
        "best_val_nll_mark_balanced_accuracy",
        "best_val_nll_mark_macro_f1",
        "best_val_nll_mark_mae",
        "best_val_nll_mark_adjacent_accuracy",
        "best_val_nll_mark_0_recall",
        "best_val_nll_mark_1_recall",
        "final_ordinal_marker_loss",
        "final_mark_mae",
        "best_val_nll_log_qty_mae",
        "best_val_nll_log_qty_rmse",
        "best_val_nll_magnitude_loss",
        "best_val_nll_log_qty_aux_loss",
        "final_log_qty_mae",
        "final_log_qty_rmse",
        "final_magnitude_loss",
        "final_log_qty_aux_loss",
    ):
        if metric_name in success_df.columns:
            agg_exprs.append(pl.mean(metric_name).alias(f"mean_{metric_name}"))
    for prefix in ("best_val_nll", "final"):
        for metric_name in RAW_MAGNITUDE_DIAGNOSTIC_NAMES:
            column_name = f"{prefix}_{metric_name}"
            if column_name in success_df.columns:
                agg_exprs.append(pl.mean(column_name).alias(f"mean_{column_name}"))
    for selection in ("best_val_nll", "best_score", "final"):
        for metric in ("score", "val_nll", "qty_mae", "dt_mae", "mark_acc", "evaluated_series"):
            col = f"ttm_contextual_{selection}_{metric}"
            if col in success_df.columns:
                agg_exprs.append(pl.mean(col).alias(f"mean_{col}"))

    summary_df = (
        success_df.group_by([
            "dataset_name",
            "model_name",
            "candidate_name",
            "value_head_mode",
            "qty_mark_gradient_mode",
            "value_encoder_gradient_mode",
            "marker_loss_mode",
            "lambda_ordinal",
            "qty_decoder_mode",
            "magnitude_norm_mode",
            "lambda_magnitude",
            *MAGNITUDE_GROUP_COLUMNS,
        ])
        .agg(agg_exprs)
        .sort(["dataset_name", "model_name", "candidate_name"])
    )
    return success_df, summary_df


def build_delta_table(summary_df: pl.DataFrame, baseline_model: str = "rmtpp") -> pl.DataFrame:
    """
    Compute model minus baseline deltas under the best-validation-NLL view.
    """
    rows: list[dict[str, Any]] = []
    for dataset_name in summary_df["dataset_name"].unique().to_list():
        dataset_rows = summary_df.filter(pl.col("dataset_name") == dataset_name)
        baseline_rows = dataset_rows.filter(pl.col("model_name") == baseline_model).to_dicts()
        compare_rows = dataset_rows.filter(pl.col("model_name") != baseline_model).to_dicts()
        if not baseline_rows or not compare_rows:
            continue
        baseline_row = baseline_rows[0]
        for row in compare_rows:
            rows.append({
                "dataset_name": dataset_name,
                "baseline_model": baseline_row["model_name"],
                "baseline_candidate_name": baseline_row["candidate_name"],
                "model_name": row["model_name"],
                "candidate_name": row["candidate_name"],
                "scale_base": row["scale_base"],
                "delta_best_val_nll": float(row["mean_best_val_nll"] - baseline_row["mean_best_val_nll"]),
                "delta_best_val_nll_qty_mae": float(
                    row["mean_best_val_nll_qty_mae"] - baseline_row["mean_best_val_nll_qty_mae"]
                ),
                "delta_best_val_nll_score": float(
                    row["mean_best_val_nll_score"] - baseline_row["mean_best_val_nll_score"]
                ),
                "delta_best_val_nll_dt_mae": float(
                    row["mean_best_val_nll_dt_mae"] - baseline_row["mean_best_val_nll_dt_mae"]
                ),
                "delta_best_val_nll_mark_acc": float(
                    row["mean_best_val_nll_mark_acc"] - baseline_row["mean_best_val_nll_mark_acc"]
                ),
                "delta_best_epoch": float(
                    row["mean_best_val_nll_epoch"] - baseline_row["mean_best_val_nll_epoch"]
                ),
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def aggregate_scale_metrics(scale_df: pl.DataFrame) -> pl.DataFrame:
    if scale_df.height == 0:
        return pl.DataFrame()
    if "value_head_mode" not in scale_df.columns:
        scale_df = scale_df.with_columns(pl.lit("shared").alias("value_head_mode"))
    if "qty_mark_gradient_mode" not in scale_df.columns:
        scale_df = scale_df.with_columns(
            pl.lit("coupled").alias("qty_mark_gradient_mode")
        )
    if "value_encoder_gradient_mode" not in scale_df.columns:
        scale_df = scale_df.with_columns(
            pl.lit("coupled").alias("value_encoder_gradient_mode")
        )
    if "marker_loss_mode" not in scale_df.columns:
        scale_df = scale_df.with_columns(pl.lit("ce").alias("marker_loss_mode"))
    if "lambda_ordinal" not in scale_df.columns:
        scale_df = scale_df.with_columns(pl.lit(0.0).alias("lambda_ordinal"))
    if "qty_decoder_mode" not in scale_df.columns:
        scale_df = scale_df.with_columns(pl.lit("mark_residual").alias("qty_decoder_mode"))
    if "magnitude_norm_mode" not in scale_df.columns:
        scale_df = scale_df.with_columns(pl.lit("global").alias("magnitude_norm_mode"))
    if "lambda_magnitude" not in scale_df.columns:
        scale_df = scale_df.with_columns(pl.lit(1.0).alias("lambda_magnitude"))
    scale_df = with_magnitude_identity_defaults(scale_df)
    return (
        scale_df.group_by([
            "dataset_name",
            "model_name",
            "candidate_name",
            "value_head_mode",
            "qty_mark_gradient_mode",
            "value_encoder_gradient_mode",
            "marker_loss_mode",
            "lambda_ordinal",
            "qty_decoder_mode",
            "magnitude_norm_mode",
            "lambda_magnitude",
            *MAGNITUDE_GROUP_COLUMNS,
            "selection",
            "scale_order",
            "scale_label",
        ])
        .agg([
            pl.sum("count").alias("total_count"),
            pl.mean("share").alias("mean_share"),
            pl.mean("true_qty_mean").alias("mean_true_qty"),
            pl.mean("pred_qty_mean").alias("mean_pred_qty"),
            pl.mean("qty_mae").alias("mean_qty_mae"),
            pl.std("qty_mae").fill_null(0.0).alias("std_qty_mae"),
            pl.mean("qty_median_ae").alias("mean_qty_median_ae"),
            pl.mean("qty_rmse").alias("mean_qty_rmse"),
            pl.mean("qty_wape").alias("mean_qty_wape"),
            pl.mean("log_abs_error").alias("mean_log_abs_error"),
            pl.mean("dt_mae").alias("mean_dt_mae"),
            pl.mean("mark_acc").alias("mean_mark_acc"),
        ])
        .sort(["dataset_name", "selection", "scale_order", "model_name", "candidate_name"])
    )


def quantity_run_label(row: dict[str, Any]) -> str:
    label = model_run_label(str(row["model_name"]), str(row["candidate_name"]))
    decoder_mode = str(row.get("qty_decoder_mode", "mark_residual"))
    if decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        domain = "log" if decoder_mode == "direct_log_qty" else "raw"
        label += f" [{domain}/{row.get('magnitude_norm_mode', 'global')}]"
    return label


def save_learning_curve_plots(history_df: pl.DataFrame, plots_dir: Path) -> None:
    if history_df.height == 0:
        return
    if "qty_decoder_mode" not in history_df.columns:
        history_df = history_df.with_columns(
            pl.lit("mark_residual").alias("qty_decoder_mode")
        )
    if "magnitude_norm_mode" not in history_df.columns:
        history_df = history_df.with_columns(
            pl.lit("global").alias("magnitude_norm_mode")
        )
    ensure_dir(plots_dir)
    # Total NLL alone hides whether the gain/loss comes from mark prediction or
    # event-time density, so we plot both decomposed likelihood terms as well.
    curve_metrics = [
        ("score", "Validation Score"),
        ("val_nll", "Validation NLL"),
        ("val_nll_marker", "Validation Marker NLL"),
        ("val_nll_time", "Validation Time NLL"),
        ("val_ordinal_marker_loss", "Validation Normalized RPS"),
        ("mark_mae", "Validation Mark MAE"),
        ("qty_mae", "Validation Qty MAE"),
        ("log_qty_mae", "Validation Log2 Qty MAE"),
        ("val_magnitude_loss", "Validation Magnitude Loss"),
    ]
    palette = ["#5DA5DA", "#F17CB0", "#60BD68", "#B276B2", "#F15854", "#DECF3F", "#FAA43A"]

    for dataset_name in history_df["dataset_name"].unique().to_list():
        dataset_df = history_df.filter(pl.col("dataset_name") == dataset_name).with_columns(
            pl.struct([
                "model_name",
                "candidate_name",
                "qty_decoder_mode",
                "magnitude_norm_mode",
            ])
            .map_elements(
                quantity_run_label,
                return_dtype=pl.Utf8,
            )
            .alias("run_label")
        )
        run_labels = dataset_df["run_label"].unique().sort().to_list()
        ncols = 3
        nrows = int(np.ceil(len(curve_metrics) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(18, 4.5 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        for ax, (metric, title) in zip(axes_flat, curve_metrics):
            if metric not in dataset_df.columns:
                ax.set_title(f"{title} (missing)")
                ax.axis("off")
                continue
            for idx, run_label in enumerate(run_labels):
                run_df = dataset_df.filter(pl.col("run_label") == run_label)
                agg_df = (
                    run_df.group_by("epoch")
                    .agg([
                        pl.mean(metric).alias("mean_metric"),
                        pl.std(metric).fill_null(0.0).alias("std_metric"),
                    ])
                    .sort("epoch")
                )
                epochs = agg_df["epoch"].to_list()
                mean_values = agg_df["mean_metric"].to_list()
                std_values = agg_df["std_metric"].to_list()
                color = palette[idx % len(palette)]
                ax.plot(epochs, mean_values, label=run_label, color=color, linewidth=2)
                lower = np.array(mean_values) - np.array(std_values)
                upper = np.array(mean_values) + np.array(std_values)
                ax.fill_between(epochs, lower, upper, color=color, alpha=0.18)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
            ax.legend()
        for ax in axes_flat[len(curve_metrics):]:
            ax.axis("off")
        fig.suptitle(f"{dataset_name}: Learning Curves", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plots_dir / f"{dataset_name}_learning_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_scale_metric_plots(scale_summary_df: pl.DataFrame, plots_dir: Path) -> None:
    if scale_summary_df.height == 0:
        return
    ensure_dir(plots_dir)
    palette = ["#5DA5DA", "#F17CB0", "#60BD68", "#B276B2", "#F15854", "#DECF3F", "#FAA43A"]
    for dataset_name in scale_summary_df["dataset_name"].unique().to_list():
        for selection in scale_summary_df["selection"].unique().to_list():
            dataset_df = scale_summary_df.filter(
                (pl.col("dataset_name") == dataset_name) & (pl.col("selection") == selection)
            ).with_columns(
                pl.struct([
                    "model_name",
                    "candidate_name",
                    "qty_decoder_mode",
                    "magnitude_norm_mode",
                ])
                .map_elements(
                    quantity_run_label,
                    return_dtype=pl.Utf8,
                )
                .alias("run_label")
            )
            if dataset_df.height == 0:
                continue
            scale_labels = (
                dataset_df.select(["scale_order", "scale_label"])
                .unique()
                .sort("scale_order")["scale_label"]
                .to_list()
            )
            x = np.arange(len(scale_labels))
            run_labels = dataset_df["run_label"].unique().sort().to_list()
            width = min(0.8 / max(len(run_labels), 1), 0.28)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for metric, title, ax in [
                ("mean_qty_mae", "Scale-wise Qty MAE", axes[0]),
                ("mean_qty_wape", "Scale-wise WAPE", axes[1]),
            ]:
                for idx, run_label in enumerate(run_labels):
                    offset = (idx - (len(run_labels) - 1) / 2) * width
                    model_df = dataset_df.filter(pl.col("run_label") == run_label).sort("scale_order")
                    value_by_label = {str(row["scale_label"]): float(row[metric]) for row in model_df.to_dicts()}
                    values = [value_by_label.get(str(label), 0.0) for label in scale_labels]
                    ax.bar(
                        x + offset,
                        values,
                        width=width,
                        label=run_label,
                        color=palette[idx % len(palette)],
                        alpha=0.9,
                    )
                ax.set_title(title)
                ax.set_xticks(x)
                ax.set_xticklabels(scale_labels, rotation=30, ha="right")
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
                ax.legend()
            fig.suptitle(f"{dataset_name}: {selection} scale-wise quantity errors", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(
                plots_dir / f"{dataset_name}_{selection}_scale_wise_qty_errors.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def save_outputs(
    *,
    summary_df: pl.DataFrame,
    delta_df: pl.DataFrame,
    scale_summary_df: pl.DataFrame,
    test_summary_df: pl.DataFrame,
    test_scale_summary_df: pl.DataFrame,
    paper_dir: Path,
) -> None:
    ensure_dir(paper_dir)
    if summary_df.width > 0:
        summary_df.write_csv(paper_dir / "paper_table_metrics.csv")
        summary_df.write_parquet(paper_dir / "paper_table_metrics.parquet")
    if delta_df.width > 0:
        delta_df.write_csv(paper_dir / "paper_table_deltas.csv")
        delta_df.write_parquet(paper_dir / "paper_table_deltas.parquet")
    if scale_summary_df.width > 0:
        scale_summary_df.write_csv(paper_dir / "paper_table_scale_wise_mae.csv")
        scale_summary_df.write_parquet(paper_dir / "paper_table_scale_wise_mae.parquet")
    if test_summary_df.width > 0:
        test_summary_df.write_csv(paper_dir / "paper_table_test_metrics.csv")
        test_summary_df.write_parquet(paper_dir / "paper_table_test_metrics.parquet")
    if test_scale_summary_df.width > 0:
        test_scale_summary_df.write_csv(paper_dir / "paper_table_test_scale_wise_mae.csv")
        test_scale_summary_df.write_parquet(paper_dir / "paper_table_test_scale_wise_mae.parquet")

    report_lines = [
        "# Unified TPP Experiment Report",
        "",
        "## Best Validation NLL Summary",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## Model - Baseline Delta",
        "",
        markdown_table_from_df(delta_df),
        "",
        "## Scale-wise Quantity Errors",
        "",
        markdown_table_from_df(scale_summary_df),
        "",
        "## Held-out Test Metrics",
        "",
        markdown_table_from_df(test_summary_df),
        "",
        "## Held-out Test Scale-wise Quantity Errors",
        "",
        markdown_table_from_df(test_scale_summary_df),
        "",
    ]
    (paper_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_long_epoch_experiment(cfg: ExperimentConfig) -> None:
    """
    End-to-end long-epoch experiment entrypoint.
    """
    selected_dataset_names = set(cfg.datasets)
    if not selected_dataset_names:
        raise ValueError("No datasets selected.")

    dataset_specs = make_dataset_specs(cfg, selected_dataset_names)
    if not dataset_specs:
        raise ValueError("No dataset specs resolved.")

    resolved_dataset_names = {spec.name for spec in dataset_specs}
    profile_map = default_profile_map(cfg.titan_profile)
    missing_profiles = sorted(resolved_dataset_names - set(profile_map))
    if missing_profiles:
        raise ValueError(f"No Titan profile found for datasets: {missing_profiles}")
    profile_map = {name: profile_map[name] for name in resolved_dataset_names}

    base_dir = ensure_dir(Path(cfg.base_dir))
    leaderboard_dir = ensure_dir(base_dir / "leaderboard")
    paper_dir = ensure_dir(base_dir / "paper_outputs")
    plots_dir = ensure_dir(paper_dir / "plots")
    logger = build_logger(base_dir / "tpp_experiment.log", "tpp_experiment")

    save_json(
        {
            "experiment_config": cfg,
            "dataset_effective_search_configs": {
                spec.name: make_search_cfg(cfg, spec.kind)
                for spec in dataset_specs
            },
            "dataset_specs": dataset_specs,
            "profile_map": profile_map,
            "titan_candidates": default_titan_candidates(),
            "thp_candidates": default_thp_candidates(),
        },
        base_dir / "experiment_manifest.json",
    )

    logger.info("Preparing marked datasets for profile=%s", cfg.titan_profile)
    marked_cache = build_marked_cache(
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        ab_cfg=cfg,
        logger=logger,
    )
    run_rows = run_long_epoch_benchmark(
        cfg=cfg,
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        marked_cache=marked_cache,
        logger=logger,
    )

    run_df, summary_df = aggregate_run_rows(run_rows)
    delta_df = build_delta_table(summary_df) if summary_df.height > 0 else pl.DataFrame()
    history_df = load_all_histories(run_rows)
    scale_df = load_all_scale_metrics(run_rows, cfg.eval_selections)
    scale_summary_df = aggregate_scale_metrics(scale_df)
    test_df = load_all_test_metrics(run_rows, cfg.eval_selections)
    test_summary_df = aggregate_test_metrics(test_df)
    test_scale_df = load_all_test_scale_metrics(run_rows, cfg.eval_selections)
    test_scale_summary_df = aggregate_scale_metrics(test_scale_df)

    if run_df.width > 0:
        run_df.write_parquet(leaderboard_dir / "runs.parquet")
        run_df.write_csv(leaderboard_dir / "runs.csv")
    if summary_df.width > 0:
        summary_df.write_parquet(leaderboard_dir / "summary.parquet")
        summary_df.write_csv(leaderboard_dir / "summary.csv")
    if delta_df.width > 0:
        delta_df.write_parquet(leaderboard_dir / "deltas.parquet")
        delta_df.write_csv(leaderboard_dir / "deltas.csv")
    if history_df.width > 0:
        history_df.write_parquet(leaderboard_dir / "histories.parquet")
        history_df.write_csv(leaderboard_dir / "histories.csv")
    if scale_df.width > 0:
        scale_df.write_parquet(leaderboard_dir / "scale_wise_metrics.parquet")
        scale_df.write_csv(leaderboard_dir / "scale_wise_metrics.csv")
    if scale_summary_df.width > 0:
        scale_summary_df.write_parquet(leaderboard_dir / "scale_wise_summary.parquet")
        scale_summary_df.write_csv(leaderboard_dir / "scale_wise_summary.csv")
    if test_df.width > 0:
        test_df.write_parquet(leaderboard_dir / "test_metrics.parquet")
        test_df.write_csv(leaderboard_dir / "test_metrics.csv")
    if test_summary_df.width > 0:
        test_summary_df.write_parquet(leaderboard_dir / "test_summary.parquet")
        test_summary_df.write_csv(leaderboard_dir / "test_summary.csv")
    if test_scale_df.width > 0:
        test_scale_df.write_parquet(leaderboard_dir / "test_scale_wise_metrics.parquet")
        test_scale_df.write_csv(leaderboard_dir / "test_scale_wise_metrics.csv")
    if test_scale_summary_df.width > 0:
        test_scale_summary_df.write_parquet(leaderboard_dir / "test_scale_wise_summary.parquet")
        test_scale_summary_df.write_csv(leaderboard_dir / "test_scale_wise_summary.csv")

    save_learning_curve_plots(history_df, plots_dir)
    save_scale_metric_plots(scale_summary_df, plots_dir)
    save_scale_metric_plots(test_scale_summary_df, plots_dir / "test")
    save_outputs(
        summary_df=summary_df,
        delta_df=delta_df,
        scale_summary_df=scale_summary_df,
        test_summary_df=test_summary_df,
        test_scale_summary_df=test_scale_summary_df,
        paper_dir=paper_dir,
    )
    logger.info("Unified long-epoch experiment complete. Summary rows:\n%s", summary_df)
