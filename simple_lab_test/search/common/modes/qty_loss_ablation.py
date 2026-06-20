"""
Quantity-supervision ablation for magnitude-factorized TPP models.

This runner exists to answer one targeted question:

    If quantity MAE is only an indirect validation metric today,
    what changes when we supervise quantity more directly?

The script compares three loss modes on the same marked-event setup:

1. `residual_only`
   - current baseline
   - mark CE + time NLL + residual Huber
2. `hybrid`
   - current baseline + direct quantity loss
3. `qty_only`
   - mark CE + time NLL + direct quantity loss

The direct quantity term is intentionally computed with expected quantity
instead of argmax-mark quantity, so gradients can flow into the mark head.

Outputs:
- run-level metrics / histories / checkpoints
- aggregated leaderboard tables
- residual-vs-direct-loss delta tables
- paper-friendly CSV / markdown tables
- learning-curve and metric-grid plots
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_paper_research")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_paper_research")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F


def _configure_stdio_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()

THIS_FILE = Path(__file__).resolve()
BOOTSTRAP_ROOT = THIS_FILE.parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from simple_lab_test.common.pathing import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(THIS_FILE)

from models.RMTPPs.RMTPP import RMTPP
from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.config import RMTPPConfig
from simple_lab_test.search.common.models import default_titan_candidates
from simple_lab_test.search.common.experiment_utils import (
    DatasetSpec,
    SearchConfig,
    TitanCandidate,
    build_logger,
    build_rmtpp_config,
    build_titan_config,
    build_training_config,
    default_dataset_specs,
    ensure_dir,
    prepare_marked_dataset,
    sanitize_float_label,
    search_config_for_dataset,
    save_json,
    summarize_history,
    tee_training_output,
    to_jsonable,
)
from utils.training import eval_next_event_week_lookback, make_week_lookback_loaders


# ---------------------------------------------------------------------------
# Report-driven Titan defaults
# ---------------------------------------------------------------------------

BEST_TITAN_BY_DATASET = {
    "intermittent": {"scale_base": 2.0, "candidate_name": "small_lmm"},
    "yellow_trip_hourly": {"scale_base": 10.0, "candidate_name": "mid_lmm"},
    "insta_market_basket": {"scale_base": 2.0, "candidate_name": "mid_lmm"},
}

BEST_TITAN_OVERALL = {
    "intermittent": {"scale_base": 2.0, "candidate_name": "small_lmm"},
    "yellow_trip_hourly": {"scale_base": 10.0, "candidate_name": "mid_lmm"},
    "insta_market_basket": {"scale_base": 2.0, "candidate_name": "mid_lmm"},
}

BEST_TITAN_SCORE_PRIORITY = {
    "intermittent": {"scale_base": 2.0, "candidate_name": "small_lmm"},
    "yellow_trip_hourly": {"scale_base": 4.0, "candidate_name": "mid_deep_lmm"},
    "insta_market_basket": {"scale_base": 2.0, "candidate_name": "mid_lmm"},
}

LOSS_MODE_ORDER = ["residual_only", "hybrid", "qty_only"]
LOSS_MODE_LABELS = {
    "residual_only": "Residual Only",
    "hybrid": "Residual + Qty",
    "qty_only": "Qty Only",
}


# ---------------------------------------------------------------------------
# Runtime dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QtyAblationConfig:
    """
    Runtime options shared across all quantity-supervision ablation runs.
    """
    base_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lookback_weeks: int = 52
    max_seq_len: int = 64
    batch_size: int = 128
    lr: float = 3e-4
    val_ratio: float = 0.2
    lambda_value: float = 1.0
    lambda_dt: float = 1.0
    lambda_qty: float = 0.25
    grad_clip: float = 1.0
    epochs: int = 30
    seeds: tuple[int, ...] = (42, 52, 62)
    titan_profile: str = "dataset_best"
    models: tuple[str, ...] = ("titantpp",)
    loss_modes: tuple[str, ...] = ("residual_only", "hybrid", "qty_only")
    qty_scale_mode: str = "p95"
    qty_scale_quantile: float = 0.95
    intermittent_max_series: int | None = None
    yellow_max_series: int | None = None
    insta_max_series: int | None = None
    force_rerun: bool = False
    stop_on_error: bool = False
    rmtpp_rnn_type: str = "gru"
    rmtpp_mark_emb_dim: int = 32
    value_head_activation: str = "sigmoid"


DEFAULT_ABLATION_EPOCHS = int(QtyAblationConfig.__dataclass_fields__["epochs"].default)


@dataclass(frozen=True)
class QtyRunConfig:
    """
    One concrete run in the ablation matrix.
    """
    dataset_name: str
    dataset_kind: str
    model_name: str
    loss_mode: str
    seed: int
    epochs: int
    scale_base: float
    titan_profile: str
    titan_candidate_name: str
    titan_candidate: TitanCandidate


@dataclass(frozen=True)
class QtyRunPaths:
    """
    Canonical filesystem layout for one ablation run.
    """
    run_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    manifest_dir: Path
    logs_dir: Path


# ---------------------------------------------------------------------------
# Small reusable helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """
    Keep train/validation shuffling and weight initialization reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_candidate(candidate: TitanCandidate) -> dict[str, Any]:
    """
    Save Titan architecture fields directly next to metrics for later analysis.
    """
    row = asdict(candidate)
    row["titan_candidate_name"] = candidate.name
    return row


def default_profile_map(profile_name: str) -> dict[str, dict[str, Any]]:
    """
    Map a user-facing profile name to the report-derived Titan defaults.
    """
    if profile_name == "dataset_best":
        return BEST_TITAN_BY_DATASET
    if profile_name == "overall":
        return BEST_TITAN_OVERALL
    if profile_name == "score_priority":
        return BEST_TITAN_SCORE_PRIORITY
    raise ValueError(f"Unsupported titan profile: {profile_name}")


def find_candidate_by_name(candidates: Iterable[TitanCandidate], name: str) -> TitanCandidate:
    """
    Recover a Titan candidate from the curated preset list.
    """
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise KeyError(f"Titan candidate not found: {name}")


def make_search_cfg(cfg: QtyAblationConfig, dataset_kind: str | None = None) -> SearchConfig:
    """
    Reuse the existing SearchConfig so preprocessing/cache utilities stay shared.
    """
    search_cfg = SearchConfig(
        base_dir=cfg.base_dir,
        device=cfg.device,
        lookback_weeks=cfg.lookback_weeks,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        val_ratio=cfg.val_ratio,
        lambda_value=cfg.lambda_value,
        lambda_dt=cfg.lambda_dt,
        grad_clip=cfg.grad_clip,
        value_head_activation=cfg.value_head_activation,
        force_rerun=cfg.force_rerun,
        stop_on_error=cfg.stop_on_error,
    )
    if dataset_kind is None:
        return search_cfg
    return search_config_for_dataset(search_cfg, dataset_kind)


def make_training_cfg(cfg: QtyAblationConfig, dataset_kind: str | None = None) -> Any:
    """
    Build the shared loader/training config used by the existing dataset split.
    """
    return build_training_config(make_search_cfg(cfg, dataset_kind), epochs=cfg.epochs)


def make_dataset_specs(cfg: QtyAblationConfig, selected_names: set[str]) -> list[DatasetSpec]:
    """
    Attach dataset-size overrides while preserving the shared dataset builders.
    """
    specs: list[DatasetSpec] = []
    for spec in default_dataset_specs():
        if spec.name not in selected_names:
            continue
        if spec.name == "intermittent":
            spec = DatasetSpec(**{**asdict(spec), "max_series": cfg.intermittent_max_series})
        elif spec.name == "yellow_trip_hourly":
            spec = DatasetSpec(**{**asdict(spec), "max_series": cfg.yellow_max_series})
        elif spec.name == "insta_market_basket":
            spec = DatasetSpec(**{**asdict(spec), "max_series": cfg.insta_max_series})
        specs.append(spec)
    return specs


def build_marked_cache(
    *,
    dataset_specs: list[DatasetSpec],
    profile_map: dict[str, dict[str, Any]],
    cfg: QtyAblationConfig,
    logger: logging.Logger,
) -> dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]]:
    """
    Prepare only the dataset/base pairs needed by the selected Titan profile.
    """
    search_cfg = make_search_cfg(cfg)
    needed_pairs = {
        (dataset_name, float(profile["scale_base"]))
        for dataset_name, profile in profile_map.items()
    }

    marked_cache: dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]] = {}
    for spec in dataset_specs:
        for dataset_name, scale_base in sorted(needed_pairs):
            if spec.name != dataset_name:
                continue
            marked_df, marked_meta, cache_root = prepare_marked_dataset(
                spec=spec,
                scale_base=scale_base,
                search_cfg=search_cfg,
                logger=logger,
            )
            marked_cache[(spec.name, scale_base)] = (marked_df, marked_meta)
            logger.info(
                "Prepared dataset=%s base=%s | rows=%s | series=%s | num_marks=%s | max_order=%s | cache=%s",
                spec.name,
                scale_base,
                marked_meta["raw_rows"],
                marked_meta["series_count"],
                marked_meta["num_marks"],
                marked_meta["max_order"],
                cache_root,
            )
    return marked_cache


def build_run_paths(cfg: QtyAblationConfig, run_cfg: QtyRunConfig) -> QtyRunPaths:
    """
    Keep loss-mode runs separated so cache reuse never mixes objectives.
    """
    base_label = sanitize_float_label(run_cfg.scale_base)
    lambda_qty_label = sanitize_float_label(cfg.lambda_qty)
    run_dir = (
        Path(cfg.base_dir)
        / "runs"
        / run_cfg.dataset_name
        / run_cfg.model_name
        / f"loss_{run_cfg.loss_mode}"
        / f"profile_{run_cfg.titan_profile}"
        / f"base_{base_label}"
        / run_cfg.titan_candidate_name
        / f"qtyscale_{cfg.qty_scale_mode}"
        / f"lambdaqty_{lambda_qty_label}"
        / f"epochs_{run_cfg.epochs}"
        / f"seed_{run_cfg.seed}"
    )
    return QtyRunPaths(
        run_dir=ensure_dir(run_dir),
        checkpoint_dir=ensure_dir(run_dir / "checkpoints"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        manifest_dir=ensure_dir(run_dir / "manifest"),
        logs_dir=ensure_dir(run_dir / "logs"),
    )


def persist_rows(rows: list[dict[str, Any]], path_prefix: Path) -> None:
    """
    Continuously snapshot aggregated rows so long runs remain inspectable.
    """
    if not rows:
        return
    serializable_rows = [{k: to_jsonable(v) for k, v in row.items()} for row in rows]
    df = pl.DataFrame(serializable_rows)
    ensure_dir(path_prefix.parent)
    df.write_parquet(path_prefix.with_suffix(".parquet"))
    df.write_csv(path_prefix.with_suffix(".csv"))


def infer_qty_scale(marked_df: pl.DataFrame, *, mode: str, quantile: float) -> float:
    """
    Derive a dataset-level scaling constant for the direct quantity loss.

    We keep this intentionally simple and deterministic:
    - `none`: no scaling, use raw quantity space
    - `p95`: divide by the dataset's positive-demand p95 before Huber loss
    """
    if mode == "none":
        return 1.0
    if mode == "p95":
        positive = marked_df.filter(pl.col("demand_qty") > 0)
        if positive.height == 0:
            return 1.0
        scale = float(
            positive.select(pl.col("demand_qty").quantile(quantile, "nearest")).item()
        )
        return max(scale, 1.0)
    raise ValueError(f"Unsupported qty_scale_mode: {mode}")


def instantiate_model(
    *,
    model_name: str,
    cfg: QtyAblationConfig,
    num_marks: int,
    scale_base: float,
    candidate: TitanCandidate,
    loss_mode: str,
    lambda_qty: float,
    qty_scale_value: float,
    dataset_kind: str | None = None,
) -> tuple[torch.nn.Module, RMTPPConfig, Any]:
    """
    Create one RMTPP-family model and the configs that define it.
    """
    search_cfg = make_search_cfg(cfg, dataset_kind)
    rmtpp_cfg = build_rmtpp_config(search_cfg, num_marks=num_marks, scale_base=scale_base)
    rmtpp_cfg = RMTPPConfig(
        **{
            **asdict(rmtpp_cfg),
            "rnn_type": cfg.rmtpp_rnn_type,
            "mark_emb_dim": cfg.rmtpp_mark_emb_dim,
            # When RMTPP joins the ablation, we keep its hidden width aligned
            # with the Titan preset so the main change remains the backbone type.
            "rnn_hidden_dim": candidate.d_model,
            # Persist the active qty-loss setup in the model config so
            # manifests/checkpoints reflect the true ablation condition.
            "loss_mode": loss_mode,
            "lambda_qty": lambda_qty,
            "qty_scale_value": qty_scale_value,
        }
    )
    titan_cfg = build_titan_config(search_cfg, candidate)

    if model_name == "rmtpp":
        model = RMTPP(rmtpp_cfg).to(cfg.device)
    elif model_name == "titantpp":
        model = TitanTPP(rmtpp_cfg, titan_cfg).to(cfg.device)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, rmtpp_cfg, titan_cfg


def expected_qty_from_logits(model, logits_no_pad: torch.Tensor, value_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute differentiable expected quantity from mark probabilities.

    This avoids the `argmax(mark)` trap during training, where quantity loss
    would stop sending gradients into the mark head.
    """
    mark_probs = torch.softmax(logits_no_pad, dim=-1)
    real_mark_count = logits_no_pad.size(-1)

    mark_grid = torch.arange(
        real_mark_count,
        device=logits_no_pad.device,
        dtype=value_hat.dtype,
    ).view(1, 1, real_mark_count)
    log_qty = mark_grid + value_hat.unsqueeze(-1)
    scale_base = torch.as_tensor(
        model.cfg.scale_base,
        device=logits_no_pad.device,
        dtype=value_hat.dtype,
    )
    qty_per_mark = torch.pow(scale_base, log_qty)
    return (mark_probs * qty_per_mark).sum(dim=-1)


def compute_loss_bundle(
    *,
    model,
    marks: torch.Tensor,
    dts: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor | None,
    loss_mode: str,
    lambda_value: float,
    lambda_dt: float,
    lambda_qty: float,
    qty_scale: float,
) -> dict[str, torch.Tensor]:
    """
    Rebuild the week-lookback loss locally so we can swap objectives cleanly.
    """
    h = model.forward(marks, dts)

    h_j = h[:, :-1, :]
    y_next = marks[:, 1:]
    dt_next = dts[:, 1:].float()
    step_mask = mask[:, 1:] & mask[:, :-1]
    denom = step_mask.sum().clamp_min(1)

    logits_full = model.mark_head(h_j)
    log_y = -F.cross_entropy(
        logits_full.reshape(-1, logits_full.size(-1)),
        y_next.reshape(-1),
        reduction="none",
    ).reshape_as(y_next)
    nll_marker = -(log_y * step_mask).sum() / denom

    logf_dt = model.log_f_dt(h_j, dt_next)
    nll_time = -(logf_dt * step_mask).sum() / denom

    if values is not None and getattr(model.cfg, "use_value_head", False):
        value_next = values[:, 1:].float()
        value_hat = model.predict_value(h_j)
        residual_elem = F.huber_loss(value_hat, value_next, reduction="none")
        value_loss = (residual_elem * step_mask).sum() / denom

        # Exclude the final PAD class when building expected quantity.
        pad_id = int(model.cfg.num_marks - 1)
        logits_real = logits_full[..., :pad_id]
        expected_qty = expected_qty_from_logits(model, logits_real, value_hat)
        true_qty = model.reconstruct_qty(y_next.clamp_max(pad_id - 1), value_next)

        qty_scale_tensor = torch.as_tensor(
            float(max(qty_scale, 1.0)),
            device=true_qty.device,
            dtype=true_qty.dtype,
        )
        qty_elem = F.huber_loss(
            expected_qty / qty_scale_tensor,
            true_qty / qty_scale_tensor,
            reduction="none",
        )
        qty_loss = (qty_elem * step_mask).sum() / denom
    else:
        value_hat = None
        value_loss = torch.zeros((), device=marks.device, dtype=torch.float32)
        qty_loss = torch.zeros((), device=marks.device, dtype=torch.float32)

    if loss_mode == "residual_only":
        total_loss = nll_marker + lambda_dt * nll_time + lambda_value * value_loss
    elif loss_mode == "hybrid":
        total_loss = (
            nll_marker
            + lambda_dt * nll_time
            + lambda_value * value_loss
            + lambda_qty * qty_loss
        )
    elif loss_mode == "qty_only":
        total_loss = nll_marker + lambda_dt * nll_time + lambda_qty * qty_loss
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    return {
        "loss": total_loss,
        "nll_marker": nll_marker,
        "nll_time": nll_time,
        "value_loss": value_loss,
        "qty_loss": qty_loss,
        "value_hat": value_hat,
    }


def train_with_qty_supervision(
    *,
    model,
    marked_df: pl.DataFrame,
    training_cfg: Any,
    loss_mode: str,
    lambda_value: float,
    lambda_dt: float,
    lambda_qty: float,
    qty_scale: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Shared trainer for RMTPP/TitanTPP under custom quantity-supervision modes.
    """
    train_loader, val_loader = make_week_lookback_loaders(marked_df, training_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.lr)

    best_val = -1.0
    best_state = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, training_cfg.epochs + 1):
        model.train()
        running_total = 0.0
        running_marker = 0.0
        running_time = 0.0
        running_value = 0.0
        running_qty = 0.0
        steps = 0

        for marks, dts, mask, _, values in train_loader:
            marks = marks.to(training_cfg.device)
            dts = dts.to(training_cfg.device)
            mask = mask.to(training_cfg.device)
            values = values.to(training_cfg.device) if values is not None else None

            out = compute_loss_bundle(
                model=model,
                marks=marks,
                dts=dts,
                mask=mask,
                values=values,
                loss_mode=loss_mode,
                lambda_value=lambda_value,
                lambda_dt=lambda_dt,
                lambda_qty=lambda_qty,
                qty_scale=qty_scale,
            )

            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            if training_cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.grad_clip)
            optimizer.step()

            running_total += float(out["loss"].item())
            running_marker += float(out["nll_marker"].item())
            running_time += float(out["nll_time"].item())
            running_value += float(out["value_loss"].item())
            running_qty += float(out["qty_loss"].item())
            steps += 1

        train_loss = running_total / max(steps, 1)
        val_metrics = eval_next_event_week_lookback(model, val_loader, training_cfg.device)

        score = (
            val_metrics["mark_acc"]
            - 0.01 * val_metrics["dt_mae"]
            - 0.001 * val_metrics["qty_mae"]
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_nll_marker": running_marker / max(steps, 1),
            "train_nll_time": running_time / max(steps, 1),
            "train_value_loss": running_value / max(steps, 1),
            "train_qty_loss": running_qty / max(steps, 1),
            "score": float(score),
            **{
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in val_metrics.items()
            },
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch:02d}] loss_mode={loss_mode} train_loss={train_loss:.8f} | "
            f"train_nll_marker={epoch_record['train_nll_marker']:.6f} "
            f"train_nll_time={epoch_record['train_nll_time']:.6f} "
            f"train_value_loss={epoch_record['train_value_loss']:.6f} "
            f"train_qty_loss={epoch_record['train_qty_loss']:.6f} | "
            f"val_acc={val_metrics['mark_acc']:.8f} "
            f"val_dt_mae={val_metrics['dt_mae']:.8f} | val_dt_rmse={val_metrics['dt_rmse']:.8f} | "
            f"val_value_mae={val_metrics['value_mae']:.8f} | val_qty_mae={val_metrics['qty_mae']:.8f} | "
            f"val_nll_time={val_metrics['val_nll_time']:.6f} "
            f"val_nll_marker={val_metrics['val_nll_marker']:.6f} "
            f"val_value_loss={val_metrics['val_value_loss']:.6f} "
            f"val_nll={val_metrics['val_nll']:.6f} | "
            f"total={val_metrics['_total']} | correct={val_metrics['_correct']} | steps={val_metrics['_nll_steps']:.0f}"
        )

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_score": best_val, "history": history}


def train_one_run(
    *,
    cfg: QtyAblationConfig,
    run_cfg: QtyRunConfig,
    marked_df: pl.DataFrame,
    marked_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Train exactly one model/loss-mode run and persist all artifacts.
    """
    run_paths = build_run_paths(cfg, run_cfg)
    summary_path = run_paths.metrics_dir / "summary.json"
    history_json_path = run_paths.metrics_dir / "history.json"
    history_parquet_path = run_paths.metrics_dir / "history.parquet"
    checkpoint_path = run_paths.checkpoint_dir / "best_model.pt"
    log_path = run_paths.logs_dir / "train.log"
    manifest_path = run_paths.manifest_dir / "run_config.json"

    if (
        not cfg.force_rerun
        and summary_path.exists()
        and history_json_path.exists()
        and checkpoint_path.exists()
    ):
        with open(summary_path, "r", encoding="utf-8") as f:
            cached_summary = json.load(f)
        if int(cached_summary.get("epochs", -1)) == int(run_cfg.epochs):
            return cached_summary

    set_global_seed(run_cfg.seed)
    search_cfg = make_search_cfg(cfg, run_cfg.dataset_kind)
    training_cfg = make_training_cfg(cfg, run_cfg.dataset_kind)
    qty_scale = infer_qty_scale(
        marked_df,
        mode=cfg.qty_scale_mode,
        quantile=cfg.qty_scale_quantile,
    )

    model, rmtpp_cfg, titan_cfg = instantiate_model(
        model_name=run_cfg.model_name,
        cfg=cfg,
        num_marks=int(marked_meta["num_marks"]),
        scale_base=run_cfg.scale_base,
        candidate=run_cfg.titan_candidate,
        loss_mode=run_cfg.loss_mode,
        lambda_qty=cfg.lambda_qty,
        qty_scale_value=qty_scale,
        dataset_kind=run_cfg.dataset_kind,
    )

    save_json(
        {
            "ablation_config": cfg,
            "effective_search_config": search_cfg,
            "run_config": run_cfg,
            "training_config": training_cfg,
            "rmtpp_config": rmtpp_cfg,
            "titan_config": titan_cfg,
            "marked_meta": marked_meta,
            "qty_scale_value": qty_scale,
        },
        manifest_path,
    )

    with tee_training_output(log_path):
        model, info = train_with_qty_supervision(
            model=model,
            marked_df=marked_df,
            training_cfg=training_cfg,
            loss_mode=run_cfg.loss_mode,
            lambda_value=cfg.lambda_value,
            lambda_dt=cfg.lambda_dt,
            lambda_qty=cfg.lambda_qty,
            qty_scale=qty_scale,
        )

    history = info["history"]
    pl.DataFrame(history).write_parquet(history_parquet_path)
    save_json({"history": history}, history_json_path)

    summary = {
        "status": "success",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "model_name": run_cfg.model_name,
        "loss_mode": run_cfg.loss_mode,
        "seed": int(run_cfg.seed),
        "epochs": int(run_cfg.epochs),
        "batch_size": int(training_cfg.batch_size),
        "lookback_weeks": int(training_cfg.lookback),
        "max_seq_len": int(training_cfg.max_seq_len),
        "scale_base": float(run_cfg.scale_base),
        "titan_profile": run_cfg.titan_profile,
        "titan_candidate_name": run_cfg.titan_candidate_name,
        "run_dir": str(run_paths.run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_marks": int(marked_meta["num_marks"]),
        "max_order": int(marked_meta["max_order"]),
        "series_count": int(marked_meta["series_count"]),
        "lambda_qty": float(cfg.lambda_qty),
        "qty_scale_mode": cfg.qty_scale_mode,
        "qty_scale_quantile": float(cfg.qty_scale_quantile),
        "qty_scale_value": float(qty_scale),
        "rmtpp_rnn_type": cfg.rmtpp_rnn_type,
        "rmtpp_hidden_dim": int(rmtpp_cfg.rnn_hidden_dim),
        "rmtpp_mark_emb_dim": int(rmtpp_cfg.mark_emb_dim),
        **flatten_candidate(run_cfg.titan_candidate),
        **summarize_history(history),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ablation_config": to_jsonable(cfg),
            "run_config": to_jsonable(run_cfg),
            "training_config": to_jsonable(training_cfg),
            "rmtpp_config": to_jsonable(rmtpp_cfg),
            "titan_config": to_jsonable(titan_cfg),
            "summary": summary,
        },
        checkpoint_path,
    )

    save_json(summary, summary_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def build_error_row(run_cfg: QtyRunConfig, exc: Exception) -> dict[str, Any]:
    """
    Convert a failed run into a durable row instead of losing that failure.
    """
    return {
        "status": "failed",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "model_name": run_cfg.model_name,
        "loss_mode": run_cfg.loss_mode,
        "seed": int(run_cfg.seed),
        "epochs": int(run_cfg.epochs),
        "scale_base": float(run_cfg.scale_base),
        "titan_profile": run_cfg.titan_profile,
        "titan_candidate_name": run_cfg.titan_candidate_name,
        "error": repr(exc),
        **flatten_candidate(run_cfg.titan_candidate),
    }


def run_ablation(
    *,
    cfg: QtyAblationConfig,
    dataset_specs: list[DatasetSpec],
    profile_map: dict[str, dict[str, Any]],
    marked_cache: dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Run the full model/loss-mode/seed ablation matrix.
    """
    all_candidates = default_titan_candidates()
    rows: list[dict[str, Any]] = []

    leaderboard_dir = ensure_dir(Path(cfg.base_dir) / "leaderboard")
    path_prefix = leaderboard_dir / "qty_loss_runs"

    total_runs = len(dataset_specs) * len(cfg.models) * len(cfg.loss_modes) * len(cfg.seeds)
    completed = 0

    for spec in dataset_specs:
        profile = profile_map[spec.name]
        scale_base = float(profile["scale_base"])
        candidate = find_candidate_by_name(all_candidates, str(profile["candidate_name"]))
        marked_df, marked_meta = marked_cache[(spec.name, scale_base)]

        for model_name in cfg.models:
            for loss_mode in cfg.loss_modes:
                for seed in cfg.seeds:
                    completed += 1
                    logger.info(
                        "Qty-loss run %s/%s | dataset=%s | model=%s | loss_mode=%s | base=%s | titan_candidate=%s | seed=%s",
                        completed,
                        total_runs,
                        spec.name,
                        model_name,
                        loss_mode,
                        scale_base,
                        candidate.name,
                        seed,
                    )
                    run_cfg = QtyRunConfig(
                        dataset_name=spec.name,
                        dataset_kind=spec.kind,
                        model_name=model_name,
                        loss_mode=loss_mode,
                        seed=seed,
                        epochs=cfg.epochs,
                        scale_base=scale_base,
                        titan_profile=cfg.titan_profile,
                        titan_candidate_name=candidate.name,
                        titan_candidate=candidate,
                    )
                    try:
                        row = train_one_run(
                            cfg=cfg,
                            run_cfg=run_cfg,
                            marked_df=marked_df,
                            marked_meta=marked_meta,
                        )
                    except Exception as exc:
                        row = build_error_row(run_cfg, exc)
                        logger.exception(
                            "Qty-loss run failed | dataset=%s model=%s loss_mode=%s base=%s seed=%s",
                            spec.name,
                            model_name,
                            loss_mode,
                            scale_base,
                            seed,
                        )
                        if cfg.stop_on_error:
                            raise
                    rows.append(row)
                    persist_rows(rows, path_prefix)
    return rows


# ---------------------------------------------------------------------------
# Aggregation, paper tables, and plots
# ---------------------------------------------------------------------------

def aggregate_run_rows(rows: list[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build run-level and aggregated summary tables for reporting.
    """
    run_df = pl.DataFrame([{k: to_jsonable(v) for k, v in row.items()} for row in rows])
    success_df = run_df.filter(pl.col("status") == "success")
    if success_df.height == 0:
        return run_df, pl.DataFrame()

    summary_df = (
        success_df.group_by(["dataset_name", "model_name", "loss_mode"])
        .agg([
            pl.first("dataset_kind").alias("dataset_kind"),
            pl.first("scale_base").alias("scale_base"),
            pl.first("batch_size").alias("batch_size"),
            pl.first("lookback_weeks").alias("lookback_weeks"),
            pl.first("max_seq_len").alias("max_seq_len"),
            pl.first("titan_profile").alias("titan_profile"),
            pl.first("titan_candidate_name").alias("titan_candidate_name"),
            pl.first("lambda_qty").alias("lambda_qty"),
            pl.first("qty_scale_mode").alias("qty_scale_mode"),
            pl.first("qty_scale_value").alias("qty_scale_value"),
            pl.len().alias("run_count"),
            pl.mean("best_score").alias("mean_best_score"),
            pl.std("best_score").fill_null(0.0).alias("std_best_score"),
            pl.mean("best_val_nll").alias("mean_best_val_nll"),
            pl.std("best_val_nll").fill_null(0.0).alias("std_best_val_nll"),
            pl.mean("best_qty_mae").alias("mean_best_qty_mae"),
            pl.std("best_qty_mae").fill_null(0.0).alias("std_best_qty_mae"),
            pl.mean("best_dt_mae").alias("mean_best_dt_mae"),
            pl.std("best_dt_mae").fill_null(0.0).alias("std_best_dt_mae"),
            pl.mean("best_mark_acc").alias("mean_best_mark_acc"),
            pl.std("best_mark_acc").fill_null(0.0).alias("std_best_mark_acc"),
            pl.mean("best_value_mae").alias("mean_best_value_mae"),
            pl.std("best_value_mae").fill_null(0.0).alias("std_best_value_mae"),
            pl.mean("best_epoch").alias("mean_best_epoch"),
            pl.mean("final_train_loss").alias("mean_final_train_loss"),
        ])
        .sort(["dataset_name", "model_name", "loss_mode"])
    )
    return success_df, summary_df


def build_delta_vs_residual(summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compare each direct-quantity variant against the residual-only baseline.
    """
    rows: list[dict[str, Any]] = []
    for dataset_name in summary_df["dataset_name"].unique().to_list():
        for model_name in summary_df.filter(pl.col("dataset_name") == dataset_name)["model_name"].unique().to_list():
            group_df = summary_df.filter(
                (pl.col("dataset_name") == dataset_name)
                & (pl.col("model_name") == model_name)
            )
            baseline_rows = group_df.filter(pl.col("loss_mode") == "residual_only").to_dicts()
            if not baseline_rows:
                continue
            baseline = baseline_rows[0]

            for loss_mode in LOSS_MODE_ORDER:
                if loss_mode == "residual_only":
                    continue
                current_rows = group_df.filter(pl.col("loss_mode") == loss_mode).to_dicts()
                if not current_rows:
                    continue
                current = current_rows[0]
                rows.append({
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "loss_mode": loss_mode,
                    "delta_best_score": float(current["mean_best_score"] - baseline["mean_best_score"]),
                    "delta_best_val_nll": float(current["mean_best_val_nll"] - baseline["mean_best_val_nll"]),
                    "delta_best_qty_mae": float(current["mean_best_qty_mae"] - baseline["mean_best_qty_mae"]),
                    "delta_best_dt_mae": float(current["mean_best_dt_mae"] - baseline["mean_best_dt_mae"]),
                    "delta_best_mark_acc": float(current["mean_best_mark_acc"] - baseline["mean_best_mark_acc"]),
                    "delta_best_value_mae": float(current["mean_best_value_mae"] - baseline["mean_best_value_mae"]),
                })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def build_best_mode_table(summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Pick the best loss mode per dataset/model using the same score-first rule.
    """
    rows: list[dict[str, Any]] = []
    for dataset_name in summary_df["dataset_name"].unique().to_list():
        for model_name in summary_df.filter(pl.col("dataset_name") == dataset_name)["model_name"].unique().to_list():
            group_rows = summary_df.filter(
                (pl.col("dataset_name") == dataset_name)
                & (pl.col("model_name") == model_name)
            ).to_dicts()
            if not group_rows:
                continue
            best_row = max(group_rows, key=lambda row: (float(row["mean_best_score"]), -float(row["mean_best_val_nll"])))
            rows.append(best_row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def markdown_table_from_df(df: pl.DataFrame) -> str:
    """
    Render a small DataFrame as markdown without extra dependencies.
    """
    if df.height == 0:
        return "_No rows available._"

    columns = df.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.to_dicts():
        formatted = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def save_paper_tables(
    *,
    summary_df: pl.DataFrame,
    delta_df: pl.DataFrame,
    best_df: pl.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save paper-ready tables in both machine-readable and markdown forms.
    """
    ensure_dir(output_dir)
    if summary_df.width > 0:
        summary_df.write_csv(output_dir / "paper_table_metrics.csv")
        summary_df.write_parquet(output_dir / "paper_table_metrics.parquet")
    if delta_df.width > 0:
        delta_df.write_csv(output_dir / "paper_table_deltas.csv")
        delta_df.write_parquet(output_dir / "paper_table_deltas.parquet")
    if best_df.width > 0:
        best_df.write_csv(output_dir / "paper_table_best_modes.csv")
        best_df.write_parquet(output_dir / "paper_table_best_modes.parquet")

    report_lines = [
        "# Quantity-Supervision Ablation Summary",
        "",
        "## Metrics Table",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## Direct-Loss Delta vs Residual Baseline",
        "",
        markdown_table_from_df(delta_df),
        "",
        "## Best Loss Mode per Dataset/Model",
        "",
        markdown_table_from_df(best_df),
        "",
    ]
    (output_dir / "paper_table_metrics.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def save_metric_bar_plots(summary_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Save one paper-friendly 2x3 metric grid for each dataset/model pair.
    """
    ensure_dir(plots_dir)
    metrics = [
        ("mean_best_score", "std_best_score", "Best Score"),
        ("mean_best_val_nll", "std_best_val_nll", "Best Val NLL"),
        ("mean_best_qty_mae", "std_best_qty_mae", "Best Qty MAE"),
        ("mean_best_dt_mae", "std_best_dt_mae", "Best DT MAE"),
        ("mean_best_mark_acc", "std_best_mark_acc", "Best Mark Acc"),
        ("mean_best_value_mae", "std_best_value_mae", "Best Value MAE"),
    ]
    colors = {
        "residual_only": "#5DA5DA",
        "hybrid": "#60BD68",
        "qty_only": "#F17CB0",
    }

    for dataset_name in summary_df["dataset_name"].unique().to_list():
        for model_name in summary_df.filter(pl.col("dataset_name") == dataset_name)["model_name"].unique().to_list():
            group_df = summary_df.filter(
                (pl.col("dataset_name") == dataset_name)
                & (pl.col("model_name") == model_name)
            )
            rows_by_mode = {row["loss_mode"]: row for row in group_df.to_dicts()}
            ordered_rows = [rows_by_mode[mode] for mode in LOSS_MODE_ORDER if mode in rows_by_mode]
            if not ordered_rows:
                continue

            x_labels = [LOSS_MODE_LABELS[row["loss_mode"]] for row in ordered_rows]
            bar_colors = [colors[row["loss_mode"]] for row in ordered_rows]

            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            axes = axes.ravel()

            for ax, (value_col, std_col, title) in zip(axes, metrics):
                values = [row[value_col] for row in ordered_rows]
                errors = [row[std_col] for row in ordered_rows]
                ax.bar(x_labels, values, yerr=errors, color=bar_colors, capsize=6)
                ax.set_title(title)
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
                ax.tick_params(axis="x", rotation=20)

            fig.suptitle(f"{dataset_name}: {model_name.upper()} qty-loss ablation", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(
                plots_dir / f"{dataset_name}_{model_name}_metric_grid.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def load_all_histories(run_rows: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Expand per-run history files into one long DataFrame for learning curves.
    """
    history_rows: list[dict[str, Any]] = []
    for row in run_rows:
        history_path = Path(str(row["run_dir"])) / "metrics" / "history.json"
        if not history_path.exists():
            continue
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f).get("history", [])
        for epoch_row in history:
            history_rows.append({
                "dataset_name": row["dataset_name"],
                "model_name": row["model_name"],
                "loss_mode": row["loss_mode"],
                "seed": row["seed"],
                "epoch": int(epoch_row["epoch"]),
                "score": float(epoch_row["score"]),
                "val_nll": float(epoch_row["val_nll"]),
                "val_nll_marker": float(epoch_row.get("val_nll_marker", float("nan"))),
                "val_nll_time": float(epoch_row.get("val_nll_time", float("nan"))),
                "qty_mae": float(epoch_row["qty_mae"]),
                "dt_mae": float(epoch_row["dt_mae"]),
                "mark_acc": float(epoch_row["mark_acc"]),
                "train_qty_loss": float(epoch_row.get("train_qty_loss", 0.0)),
            })
    return pl.DataFrame(history_rows) if history_rows else pl.DataFrame()


def save_learning_curve_plots(history_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Plot mean +/- std learning curves across seeds for each dataset/model pair.
    """
    if history_df.height == 0:
        return

    ensure_dir(plots_dir)
    # Total NLL can move for two different reasons: mark classification or
    # event-time likelihood. Keep both terms visible in ablation plots.
    curve_metrics = [
        ("score", "Validation Score"),
        ("val_nll", "Validation NLL"),
        ("val_nll_marker", "Validation Marker NLL"),
        ("val_nll_time", "Validation Time NLL"),
        ("qty_mae", "Validation Qty MAE"),
    ]
    colors = {
        "residual_only": "#5DA5DA",
        "hybrid": "#60BD68",
        "qty_only": "#F17CB0",
    }

    for dataset_name in history_df["dataset_name"].unique().to_list():
        for model_name in history_df.filter(pl.col("dataset_name") == dataset_name)["model_name"].unique().to_list():
            group_df = history_df.filter(
                (pl.col("dataset_name") == dataset_name)
                & (pl.col("model_name") == model_name)
            )

            fig, axes = plt.subplots(2, 3, figsize=(18, 9))
            axes_flat = axes.flatten()
            for ax, (metric, title) in zip(axes_flat, curve_metrics):
                if metric not in group_df.columns:
                    ax.set_title(f"{title} (missing)")
                    ax.axis("off")
                    continue
                for loss_mode in LOSS_MODE_ORDER:
                    mode_df = group_df.filter(pl.col("loss_mode") == loss_mode)
                    if mode_df.height == 0:
                        continue
                    agg_df = (
                        mode_df.group_by("epoch")
                        .agg([
                            pl.mean(metric).alias("mean_metric"),
                            pl.std(metric).fill_null(0.0).alias("std_metric"),
                        ])
                        .sort("epoch")
                    )
                    epochs = agg_df["epoch"].to_list()
                    mean_values = agg_df["mean_metric"].to_list()
                    std_values = agg_df["std_metric"].to_list()
                    ax.plot(
                        epochs,
                        mean_values,
                        label=LOSS_MODE_LABELS[loss_mode],
                        color=colors[loss_mode],
                        linewidth=2,
                    )
                    lower = np.array(mean_values) - np.array(std_values)
                    upper = np.array(mean_values) + np.array(std_values)
                    ax.fill_between(epochs, lower, upper, color=colors[loss_mode], alpha=0.2)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.25)
                ax.legend()
            for ax in axes_flat[len(curve_metrics):]:
                ax.axis("off")

            fig.suptitle(f"{dataset_name}: {model_name.upper()} learning curves", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(
                plots_dir / f"{dataset_name}_{model_name}_learning_curves.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def save_text_summary(
    *,
    summary_df: pl.DataFrame,
    delta_df: pl.DataFrame,
    best_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """
    Produce a short narrative summary for quick report/paper drafting.
    """
    lines = [
        "# Quantity-Supervision Ablation",
        "",
        "## Main Table",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## Direct-Loss Delta vs Residual Baseline",
        "",
        markdown_table_from_df(delta_df),
        "",
        "## Best Loss Mode per Dataset/Model",
        "",
        markdown_table_from_df(best_df),
        "",
    ]

    for row in best_df.to_dicts():
        lines.extend([
            f"### {row['dataset_name']} / {row['model_name']}",
            "",
            (
                f"Best loss mode by score: `{row['loss_mode']}` "
                f"(score={row['mean_best_score']:.6f}, "
                f"val_nll={row['mean_best_val_nll']:.6f}, "
                f"qty_mae={row['mean_best_qty_mae']:.6f})."
            ),
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse benchmark runtime settings.
    """
    parser = argparse.ArgumentParser(
        description="Run quantity-supervision ablations for RMTPP/TitanTPP."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT / "search_artifacts" / "tpp_qty_loss_ablation"),
        help="Directory where ablation artifacts will be written.",
    )
    parser.add_argument(
        "--datasets",
        default="intermittent,yellow_trip_hourly",
        help=(
            "Comma-separated dataset names to evaluate. Supported: "
            "intermittent,yellow_trip_hourly,insta_market_basket. Use "
            "yellow_trip_hourly after simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb."
        ),
    )
    parser.add_argument(
        "--models",
        default="titantpp",
        help="Comma-separated model names. Typical values: titantpp or rmtpp,titantpp.",
    )
    parser.add_argument(
        "--loss-modes",
        default="residual_only,hybrid,qty_only",
        help="Comma-separated loss modes.",
    )
    parser.add_argument(
        "--titan-profile",
        default="dataset_best",
        choices=["dataset_best", "overall", "score_priority"],
        help="Which report-derived Titan default set to use.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_ABLATION_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated random seeds.")
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--lambda-qty", type=float, default=0.25)
    parser.add_argument(
        "--qty-scale-mode",
        default="p95",
        choices=["none", "p95"],
        help="How to normalize direct quantity loss.",
    )
    parser.add_argument(
        "--qty-scale-quantile",
        type=float,
        default=0.95,
        help="Quantile used when qty-scale-mode=p95.",
    )
    parser.add_argument("--intermittent-max-series", type=int, default=None)
    parser.add_argument("--yellow-max-series", type=int, default=None)
    parser.add_argument("--insta-max-series", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Run the full quantity-supervision ablation and export report artifacts.
    """
    args = parse_args()
    selected_dataset_names = {name.strip() for name in args.datasets.split(",") if name.strip()}
    selected_models = tuple(name.strip() for name in args.models.split(",") if name.strip())
    selected_loss_modes = tuple(name.strip() for name in args.loss_modes.split(",") if name.strip())
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())

    invalid_loss_modes = [mode for mode in selected_loss_modes if mode not in LOSS_MODE_ORDER]
    if invalid_loss_modes:
        raise ValueError(f"Unsupported loss modes: {invalid_loss_modes}")

    cfg = QtyAblationConfig(
        base_dir=args.base_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=seeds,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        titan_profile=args.titan_profile,
        models=selected_models,
        loss_modes=selected_loss_modes,
        lambda_qty=args.lambda_qty,
        qty_scale_mode=args.qty_scale_mode,
        qty_scale_quantile=args.qty_scale_quantile,
        intermittent_max_series=args.intermittent_max_series,
        yellow_max_series=args.yellow_max_series,
        insta_max_series=args.insta_max_series,
        force_rerun=args.force_rerun,
        stop_on_error=args.stop_on_error,
    )

    print(f"Qty-loss ablation configuration: {cfg}")

    dataset_specs = make_dataset_specs(cfg, selected_dataset_names)
    if not dataset_specs:
        raise ValueError("No datasets selected for qty-loss ablation.")

    profile_map = default_profile_map(cfg.titan_profile)
    profile_map = {name: profile_map[name] for name in selected_dataset_names}

    base_dir = ensure_dir(Path(cfg.base_dir))
    leaderboard_dir = ensure_dir(base_dir / "leaderboard")
    paper_dir = ensure_dir(base_dir / "paper_outputs")
    plots_dir = ensure_dir(paper_dir / "plots")
    logger = build_logger(base_dir / "qty_loss_ablation.log", "tpp_qty_loss_ablation")

    candidates = default_titan_candidates()
    save_json(
        {
            "ablation_config": cfg,
            "dataset_effective_search_configs": {
                spec.name: make_search_cfg(cfg, spec.kind)
                for spec in dataset_specs
            },
            "dataset_specs": dataset_specs,
            "titan_profile_map": profile_map,
            "candidates": candidates,
        },
        base_dir / "qty_loss_ablation_manifest.json",
    )

    logger.info("Preparing marked datasets for profile=%s", cfg.titan_profile)
    marked_cache = build_marked_cache(
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        cfg=cfg,
        logger=logger,
    )

    run_rows = run_ablation(
        cfg=cfg,
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        marked_cache=marked_cache,
        logger=logger,
    )

    run_df, summary_df = aggregate_run_rows(run_rows)
    delta_df = build_delta_vs_residual(summary_df)
    best_df = build_best_mode_table(summary_df)

    if run_df.width > 0:
        run_df.write_parquet(leaderboard_dir / "qty_loss_runs.parquet")
        run_df.write_csv(leaderboard_dir / "qty_loss_runs.csv")
    if summary_df.width > 0:
        summary_df.write_parquet(leaderboard_dir / "qty_loss_summary.parquet")
        summary_df.write_csv(leaderboard_dir / "qty_loss_summary.csv")
    if delta_df.width > 0:
        delta_df.write_parquet(leaderboard_dir / "qty_loss_deltas.parquet")
        delta_df.write_csv(leaderboard_dir / "qty_loss_deltas.csv")
    if best_df.width > 0:
        best_df.write_parquet(leaderboard_dir / "qty_loss_best_modes.parquet")
        best_df.write_csv(leaderboard_dir / "qty_loss_best_modes.csv")

    save_paper_tables(summary_df=summary_df, delta_df=delta_df, best_df=best_df, output_dir=paper_dir)
    save_metric_bar_plots(summary_df, plots_dir)
    history_df = load_all_histories(run_rows)
    if history_df.height > 0:
        history_df.write_parquet(leaderboard_dir / "qty_loss_histories.parquet")
        history_df.write_csv(leaderboard_dir / "qty_loss_histories.csv")
    save_learning_curve_plots(history_df, plots_dir)
    save_text_summary(
        summary_df=summary_df,
        delta_df=delta_df,
        best_df=best_df,
        output_path=paper_dir / "qty_loss_analysis_summary.md",
    )

    logger.info("Qty-loss ablation complete. Summary rows:\n%s", summary_df)


if __name__ == "__main__":
    main()
