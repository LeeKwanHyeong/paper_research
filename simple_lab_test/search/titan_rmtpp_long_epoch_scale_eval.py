"""
Long-epoch RMTPP vs TitanTPP validation with scale-wise quantity errors.

Why this script exists:
1. answer whether validation NLL keeps improving beyond the 30-epoch run
2. save best-validation-NLL checkpoints so the comparison uses a true sweet spot
3. split quantity MAE by true demand scale so extreme quantities do not hide
   performance on small and medium demand events

This is intentionally separate from `titan_rmtpp_ab_test.py`. The existing
A/B runner remains the main 30-epoch benchmark, while this file is a follow-up
validation runner for the professor's convergence and scale-error questions.
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


def _configure_stdio_utf8() -> None:
    """
    Keep long-running remote logs readable even when the shell locale is not UTF-8.
    """
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
from simple_lab_test.search.titan_hparam_search import (
    DatasetSpec,
    SearchConfig,
    TitanCandidate,
    build_logger,
    build_rmtpp_config,
    build_titan_config,
    build_training_config,
    default_titan_candidates,
    ensure_dir,
    save_json,
    sanitize_float_label,
    tee_training_output,
    to_jsonable,
)
from simple_lab_test.search.titan_rmtpp_ab_test import (
    build_marked_cache,
    default_profile_map,
    find_candidate_by_name,
    flatten_candidate,
    make_dataset_specs,
    make_search_cfg,
    markdown_table_from_df,
    persist_rows,
    save_learning_curve_plots,
)
from utils.training import (
    TrainingConfig,
    eval_next_event_week_lookback,
    make_week_lookback_loaders,
)


# ---------------------------------------------------------------------------
# Runtime dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LongEpochConfig:
    """
    User-facing runtime options for the long-epoch validation run.
    """
    base_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lookback_weeks: int = 52
    max_seq_len: int = 256
    batch_size: int = 128
    lr: float = 1e-3
    val_ratio: float = 0.2
    lambda_value: float = 1.0
    lambda_dt: float = 1.0
    grad_clip: float = 1.0
    epochs: int = 100
    seeds: tuple[int, ...] = (42, 52, 62)
    titan_profile: str = "dataset_best"
    intermittent_max_series: int | None = None
    yellow_max_series: int | None = None
    force_rerun: bool = False
    stop_on_error: bool = False
    rmtpp_rnn_type: str = "gru"
    rmtpp_mark_emb_dim: int = 32
    loss_mode: str = "residual_only"
    analysis_scale_base: float = 10.0
    analysis_tail_order: int = 4
    eval_selections: tuple[str, ...] = ("best_val_nll",)


DEFAULT_LONG_EPOCHS = int(LongEpochConfig.__dataclass_fields__["epochs"].default)


@dataclass(frozen=True)
class LongRunConfig:
    """
    Full identity of one dataset/model/seed long-run experiment.
    """
    dataset_name: str
    dataset_kind: str
    model_name: str
    seed: int
    epochs: int
    scale_base: float
    titan_profile: str
    titan_candidate_name: str
    titan_candidate: TitanCandidate


@dataclass(frozen=True)
class LongRunPaths:
    """
    Canonical per-run output layout.
    """
    run_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    manifest_dir: Path
    logs_dir: Path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """
    Keep data loader shuffling and weight initialization reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_training_cfg(long_cfg: LongEpochConfig, dataset_kind: str | None = None) -> TrainingConfig:
    """
    Convert the long-run config into the shared model training config.
    """
    search_cfg = make_search_cfg(long_cfg, dataset_kind)
    return build_training_config(search_cfg, epochs=long_cfg.epochs)


def build_long_run_paths(long_cfg: LongEpochConfig, run_cfg: LongRunConfig) -> LongRunPaths:
    """
    Keep long-run outputs separate from the 30-epoch A/B benchmark outputs.
    """
    base_label = sanitize_float_label(run_cfg.scale_base)
    run_dir = (
        Path(long_cfg.base_dir)
        / "runs"
        / run_cfg.dataset_name
        / run_cfg.model_name
        / f"lossmode_{long_cfg.loss_mode}"
        / f"profile_{run_cfg.titan_profile}"
        / f"base_{base_label}"
        / run_cfg.titan_candidate_name
        / f"epochs_{run_cfg.epochs}"
        / f"seed_{run_cfg.seed}"
    )
    return LongRunPaths(
        run_dir=ensure_dir(run_dir),
        checkpoint_dir=ensure_dir(run_dir / "checkpoints"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        manifest_dir=ensure_dir(run_dir / "manifest"),
        logs_dir=ensure_dir(run_dir / "logs"),
    )


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Store checkpoints on CPU so long GPU runs do not keep stale tensors alive.
    """
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def finite_or_default(value: Any, default: float) -> float:
    """
    Make ranking robust to occasional NaN validation metrics.
    """
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value_float):
        return default
    return value_float


def build_model(
    *,
    long_cfg: LongEpochConfig,
    run_cfg: LongRunConfig,
    marked_meta: dict[str, Any],
) -> tuple[torch.nn.Module, RMTPPConfig, Any]:
    """
    Instantiate RMTPP or TitanTPP with the same mark vocabulary and scale base.
    """
    search_cfg = make_search_cfg(long_cfg, run_cfg.dataset_kind)
    rmtpp_cfg = build_rmtpp_config(
        search_cfg,
        num_marks=int(marked_meta["num_marks"]),
        scale_base=run_cfg.scale_base,
    )
    rmtpp_cfg = RMTPPConfig(
        **{
            **asdict(rmtpp_cfg),
            "rnn_type": long_cfg.rmtpp_rnn_type,
            "mark_emb_dim": long_cfg.rmtpp_mark_emb_dim,
            "rnn_hidden_dim": run_cfg.titan_candidate.d_model,
            # Keep the main RMTPP-vs-TitanTPP comparison on the legacy
            # residual-only objective. Quantity-loss variants stay in the
            # dedicated ablation script.
            "loss_mode": long_cfg.loss_mode,
        }
    )
    titan_cfg = build_titan_config(search_cfg, run_cfg.titan_candidate)

    if run_cfg.model_name == "rmtpp":
        if long_cfg.loss_mode != "residual_only":
            raise ValueError("RMTPP long-epoch comparison currently supports residual_only only.")
        model = RMTPP(rmtpp_cfg).to(long_cfg.device)
    elif run_cfg.model_name == "titantpp":
        model = TitanTPP(rmtpp_cfg, titan_cfg).to(long_cfg.device)
    else:
        raise ValueError(f"Unsupported model_name: {run_cfg.model_name}")

    return model, rmtpp_cfg, titan_cfg


def compute_training_loss(
    *,
    model: torch.nn.Module,
    out: dict[str, torch.Tensor],
    training_cfg: TrainingConfig,
) -> torch.Tensor:
    """
    Reproduce the shared trainer objective while keeping this file checkpoint-aware.
    """
    loss_mode = getattr(model.cfg, "loss_mode", "residual_only")
    if loss_mode == "residual_only":
        return (
            out["nll_marker"]
            + training_cfg.lambda_value * out["value_loss"]
            + training_cfg.lambda_dt * out["nll_time"]
        )
    if loss_mode == "hybrid":
        return (
            out["nll_marker"]
            + training_cfg.lambda_value * out["value_loss"]
            + training_cfg.lambda_dt * out["nll_time"]
            + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
        )
    if loss_mode == "qty_only":
        return (
            out["nll_marker"]
            + training_cfg.lambda_dt * out["nll_time"]
            + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
        )
    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


def summarize_long_history(history: list[dict[str, Any]]) -> dict[str, Any]:
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

    return {
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
        "final_epoch": int(final_row["epoch"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_score": float(final_row["score"]),
        "final_val_nll": float(final_row["val_nll"]),
        "final_qty_mae": float(final_row["qty_mae"]),
        "final_dt_mae": float(final_row["dt_mae"]),
        "final_mark_acc": float(final_row["mark_acc"]),
    }


def scale_label(order: int, *, base: float, tail_order: int) -> str:
    """
    Human-readable scale labels for tables and plots.
    """
    low = base ** order
    if order >= tail_order:
        return f">={low:g}"
    high = (base ** (order + 1)) - 1
    if float(base).is_integer():
        return f"{int(round(low))}-{int(round(high))}"
    return f"[{low:g}, {base ** (order + 1):g})"


def empty_scale_bucket() -> dict[str, Any]:
    """
    Accumulator for one true-quantity scale bucket.
    """
    return {
        "count": 0,
        "true_sum": 0.0,
        "pred_sum": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "true_sq_sum": 0.0,
        "log_abs_sum": 0.0,
        "dt_abs_sum": 0.0,
        "mark_correct": 0,
        # Median absolute error is useful for heavy-tailed quantities, so keep
        # only per-scale absolute errors rather than the full prediction table.
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

    The scale bucket is based on the true reconstructed quantity, not on the
    predicted mark. This answers "where is the model actually accurate?" rather
    than "which class did the model choose?".
    """
    model.eval()
    pad_id = int(model.cfg.num_marks - 1)
    log_base = float(np.log(analysis_scale_base))
    eps = float(getattr(model.cfg, "eps", 1e-8))

    buckets = {
        order: empty_scale_bucket()
        for order in range(0, int(analysis_tail_order) + 1)
    }

    for marks, dts, mask, _, values in val_loader:
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)
        values = values.to(device) if values is not None else None

        valid = mask[:, -1] & mask[:, -2]
        if values is None:
            continue

        h = model.forward(marks, dts)
        h_prev = h[:, -2, :]
        y_mk = marks[:, -1]
        y_dt = dts[:, -1].float()
        y_val = values[:, -1].float()

        valid = valid & (y_mk != pad_id)
        if valid.sum().item() == 0:
            continue

        h_prev = h_prev[valid]
        y_mk = y_mk[valid]
        y_dt = y_dt[valid]
        y_val = y_val[valid]

        logits = model.mark_head(h_prev)[..., :pad_id]
        pred_mk = torch.argmax(logits, dim=-1)
        value_hat = model.predict_value(h_prev)

        qty_hat = model.reconstruct_qty(pred_mk, value_hat)
        qty_true = model.reconstruct_qty(y_mk, y_val)
        abs_err = (qty_hat - qty_true).abs()
        sq_err = (qty_hat - qty_true) ** 2
        log_abs_err = (
            torch.log(qty_hat.clamp_min(eps))
            - torch.log(qty_true.clamp_min(eps))
        ).abs() / log_base

        u = torch.full((y_dt.size(0),), 0.5, device=device).clamp_min(model.cfg.eps)
        dt_hat = model.sample_next_dt(h_prev, u=u).clamp_min(1.0)
        dt_abs_err = (dt_hat - y_dt).abs()

        # `floor(log_base(qty))` gives the natural order-of-magnitude bucket.
        scale_order = torch.floor(
            torch.log(qty_true.clamp_min(1.0)) / log_base
        ).long()
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
            bucket["true_sq_sum"] += float((qty_true[order_mask] ** 2).sum().item())
            bucket["log_abs_sum"] += float(log_abs_err[order_mask].sum().item())
            bucket["dt_abs_sum"] += float(dt_abs_err[order_mask].sum().item())
            bucket["mark_correct"] += int((pred_mk[order_mask] == y_mk[order_mask]).sum().item())
            bucket["abs_errors"].extend(abs_err[order_mask].detach().cpu().tolist())

    total_count = sum(bucket["count"] for bucket in buckets.values())
    rows: list[dict[str, Any]] = []
    for order, bucket in buckets.items():
        count = int(bucket["count"])
        if count == 0:
            rows.append({
                "scale_order": order,
                "scale_label": scale_label(order, base=analysis_scale_base, tail_order=analysis_tail_order),
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
            "scale_label": scale_label(order, base=analysis_scale_base, tail_order=analysis_tail_order),
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


# ---------------------------------------------------------------------------
# Training and per-run persistence
# ---------------------------------------------------------------------------

def scale_metric_paths(run_paths: LongRunPaths, selection: str) -> tuple[Path, Path]:
    """
    Use one scale-wise file per checkpoint selection.
    """
    stem = f"scale_wise_{selection}"
    return run_paths.metrics_dir / f"{stem}.csv", run_paths.metrics_dir / f"{stem}.parquet"


def cached_run_is_complete(
    *,
    long_cfg: LongEpochConfig,
    run_cfg: LongRunConfig,
    run_paths: LongRunPaths,
) -> bool:
    """
    Avoid repeating expensive long runs when all expected artifacts exist.
    """
    summary_path = run_paths.metrics_dir / "summary.json"
    history_path = run_paths.metrics_dir / "history.json"
    nll_checkpoint_path = run_paths.checkpoint_dir / "best_val_nll_model.pt"
    if not (summary_path.exists() and history_path.exists() and nll_checkpoint_path.exists()):
        return False

    for selection in long_cfg.eval_selections:
        csv_path, parquet_path = scale_metric_paths(run_paths, selection)
        if not (csv_path.exists() and parquet_path.exists()):
            return False

    with open(summary_path, "r", encoding="utf-8") as f:
        cached_summary = json.load(f)
    return int(cached_summary.get("epochs", -1)) == int(run_cfg.epochs)


def save_checkpoint(
    *,
    path: Path,
    model_state: dict[str, torch.Tensor],
    long_cfg: LongEpochConfig,
    run_cfg: LongRunConfig,
    training_cfg: TrainingConfig,
    rmtpp_cfg: RMTPPConfig,
    titan_cfg: Any,
    summary: dict[str, Any],
    selection: str,
) -> None:
    """
    Persist one selected model state with enough metadata to reload it later.
    """
    torch.save(
        {
            "selection": selection,
            "model_state_dict": model_state,
            "long_epoch_config": to_jsonable(long_cfg),
            "run_config": to_jsonable(run_cfg),
            "training_config": to_jsonable(training_cfg),
            "rmtpp_config": to_jsonable(rmtpp_cfg),
            "titan_config": to_jsonable(titan_cfg),
            "summary": summary,
        },
        path,
    )


def train_one_long_run(
    *,
    long_cfg: LongEpochConfig,
    run_cfg: LongRunConfig,
    marked_df: pl.DataFrame,
    marked_meta: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Train one long-run model and save best-score, best-NLL, and final states.
    """
    run_paths = build_long_run_paths(long_cfg, run_cfg)
    summary_path = run_paths.metrics_dir / "summary.json"
    history_json_path = run_paths.metrics_dir / "history.json"
    history_parquet_path = run_paths.metrics_dir / "history.parquet"
    log_path = run_paths.logs_dir / "train.log"
    manifest_path = run_paths.manifest_dir / "run_config.json"

    if not long_cfg.force_rerun and cached_run_is_complete(
        long_cfg=long_cfg,
        run_cfg=run_cfg,
        run_paths=run_paths,
    ):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    set_global_seed(run_cfg.seed)
    search_cfg = make_search_cfg(long_cfg, run_cfg.dataset_kind)
    training_cfg = make_training_cfg(long_cfg, run_cfg.dataset_kind)
    train_loader, val_loader = make_week_lookback_loaders(marked_df, training_cfg)
    model, rmtpp_cfg, titan_cfg = build_model(
        long_cfg=long_cfg,
        run_cfg=run_cfg,
        marked_meta=marked_meta,
    )

    save_json(
        {
            "long_epoch_config": long_cfg,
            "effective_search_config": search_cfg,
            "run_config": run_cfg,
            "training_config": training_cfg,
            "rmtpp_config": rmtpp_cfg,
            "titan_config": titan_cfg,
            "marked_meta": marked_meta,
        },
        manifest_path,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.lr)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_val_nll = float("inf")
    best_score_state: dict[str, torch.Tensor] | None = None
    best_val_nll_state: dict[str, torch.Tensor] | None = None

    with tee_training_output(log_path):
        for epoch in range(1, training_cfg.epochs + 1):
            model.train()
            running = 0.0
            steps = 0

            for marks, dts, mask, _, values in train_loader:
                marks = marks.to(training_cfg.device)
                dts = dts.to(training_cfg.device)
                mask = mask.to(training_cfg.device)
                values = values.to(training_cfg.device) if values is not None else None

                out = model.nll(marks, dts, values=values, mask=mask)
                loss = compute_training_loss(model=model, out=out, training_cfg=training_cfg)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if training_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.grad_clip)
                optimizer.step()

                running += float(loss.item())
                steps += 1

            train_loss = running / max(steps, 1)
            val_metrics = eval_next_event_week_lookback(model, val_loader, training_cfg.device)
            score = (
                val_metrics["mark_acc"]
                - 0.01 * val_metrics["dt_mae"]
                - 0.001 * val_metrics["qty_mae"]
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
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
                f"val_qty_mae={val_metrics['qty_mae']:.8f}"
            )

            if score > best_score:
                best_score = float(score)
                best_score_state = clone_state_dict(model)

            val_nll = finite_or_default(val_metrics["val_nll"], float("inf"))
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_val_nll_state = clone_state_dict(model)

    final_state = clone_state_dict(model)
    if best_score_state is None:
        best_score_state = final_state
    if best_val_nll_state is None:
        best_val_nll_state = final_state

    history_df = pl.DataFrame(history)
    history_df.write_parquet(history_parquet_path)
    save_json({"history": history}, history_json_path)

    summary = {
        "status": "success",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "model_name": run_cfg.model_name,
        "seed": int(run_cfg.seed),
        "epochs": int(run_cfg.epochs),
        "lr": float(long_cfg.lr),
        "batch_size": int(training_cfg.batch_size),
        "lookback_weeks": int(training_cfg.lookback),
        "max_seq_len": int(training_cfg.max_seq_len),
        "scale_base": float(run_cfg.scale_base),
        "analysis_scale_base": float(long_cfg.analysis_scale_base),
        "analysis_tail_order": int(long_cfg.analysis_tail_order),
        "titan_profile": run_cfg.titan_profile,
        "loss_mode": long_cfg.loss_mode,
        "titan_candidate_name": run_cfg.titan_candidate_name,
        "run_dir": str(run_paths.run_dir),
        "best_score_checkpoint_path": str(run_paths.checkpoint_dir / "best_score_model.pt"),
        "best_val_nll_checkpoint_path": str(run_paths.checkpoint_dir / "best_val_nll_model.pt"),
        "final_checkpoint_path": str(run_paths.checkpoint_dir / "final_model.pt"),
        "num_marks": int(marked_meta["num_marks"]),
        "max_order": int(marked_meta["max_order"]),
        "series_count": int(marked_meta["series_count"]),
        "rmtpp_rnn_type": long_cfg.rmtpp_rnn_type,
        "rmtpp_hidden_dim": int(rmtpp_cfg.rnn_hidden_dim),
        "rmtpp_mark_emb_dim": int(rmtpp_cfg.mark_emb_dim),
        **flatten_candidate(run_cfg.titan_candidate),
        **summarize_long_history(history),
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
            long_cfg=long_cfg,
            run_cfg=run_cfg,
            training_cfg=training_cfg,
            rmtpp_cfg=rmtpp_cfg,
            titan_cfg=titan_cfg,
            summary=summary,
            selection=selection,
        )

    # Scale-wise metrics are computed after loading the requested checkpoint
    # state, so the reported table explicitly matches the selected sweet spot.
    for selection in long_cfg.eval_selections:
        selected_state = state_by_selection[selection]
        model.load_state_dict(selected_state)
        scale_df = evaluate_scale_wise_qty(
            model=model,
            val_loader=val_loader,
            device=training_cfg.device,
            analysis_scale_base=long_cfg.analysis_scale_base,
            analysis_tail_order=long_cfg.analysis_tail_order,
        ).with_columns([
            pl.lit(run_cfg.dataset_name).alias("dataset_name"),
            pl.lit(run_cfg.dataset_kind).alias("dataset_kind"),
            pl.lit(run_cfg.model_name).alias("model_name"),
            pl.lit(run_cfg.seed).alias("seed"),
            pl.lit(selection).alias("selection"),
            pl.lit(run_cfg.scale_base).alias("model_scale_base"),
            pl.lit(run_cfg.titan_candidate_name).alias("titan_candidate_name"),
        ])
        csv_path, parquet_path = scale_metric_paths(run_paths, selection)
        scale_df.write_csv(csv_path)
        scale_df.write_parquet(parquet_path)

    save_json(summary, summary_path)
    logger.info(
        "Finished long run | dataset=%s model=%s seed=%s best_val_nll=%.6f epoch=%s",
        run_cfg.dataset_name,
        run_cfg.model_name,
        run_cfg.seed,
        summary["best_val_nll"],
        summary["best_val_nll_epoch"],
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def build_error_row(run_cfg: LongRunConfig, exc: Exception) -> dict[str, Any]:
    """
    Persist failed runs in the leaderboard instead of losing the failure state.
    """
    return {
        "status": "failed",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "model_name": run_cfg.model_name,
        "seed": int(run_cfg.seed),
        "epochs": int(run_cfg.epochs),
        "scale_base": float(run_cfg.scale_base),
        "titan_profile": run_cfg.titan_profile,
        "titan_candidate_name": run_cfg.titan_candidate_name,
        "error": repr(exc),
        **flatten_candidate(run_cfg.titan_candidate),
    }


def run_long_benchmark(
    *,
    long_cfg: LongEpochConfig,
    dataset_specs: list[DatasetSpec],
    profile_map: dict[str, dict[str, Any]],
    marked_cache: dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Run RMTPP and TitanTPP for every dataset/seed pair.
    """
    all_candidates = default_titan_candidates()
    rows: list[dict[str, Any]] = []
    leaderboard_dir = ensure_dir(Path(long_cfg.base_dir) / "leaderboard")
    path_prefix = leaderboard_dir / "long_epoch_runs"

    total_runs = len(dataset_specs) * len(long_cfg.seeds) * 2
    completed = 0

    for spec in dataset_specs:
        profile = profile_map[spec.name]
        scale_base = float(profile["scale_base"])
        candidate = find_candidate_by_name(all_candidates, str(profile["candidate_name"]))
        marked_df, marked_meta = marked_cache[(spec.name, scale_base)]

        for seed in long_cfg.seeds:
            for model_name in ("rmtpp", "titantpp"):
                completed += 1
                logger.info(
                    "Long run %s/%s | dataset=%s | model=%s | base=%s | titan_candidate=%s | seed=%s",
                    completed,
                    total_runs,
                    spec.name,
                    model_name,
                    scale_base,
                    candidate.name,
                    seed,
                )
                run_cfg = LongRunConfig(
                    dataset_name=spec.name,
                    dataset_kind=spec.kind,
                    model_name=model_name,
                    seed=seed,
                    epochs=long_cfg.epochs,
                    scale_base=scale_base,
                    titan_profile=long_cfg.titan_profile,
                    titan_candidate_name=candidate.name,
                    titan_candidate=candidate,
                )
                try:
                    row = train_one_long_run(
                        long_cfg=long_cfg,
                        run_cfg=run_cfg,
                        marked_df=marked_df,
                        marked_meta=marked_meta,
                        logger=logger,
                    )
                except Exception as exc:
                    row = build_error_row(run_cfg, exc)
                    logger.exception(
                        "Long run failed | dataset=%s model=%s base=%s seed=%s",
                        spec.name,
                        model_name,
                        scale_base,
                        seed,
                    )
                    if long_cfg.stop_on_error:
                        raise
                rows.append(row)
                persist_rows(rows, path_prefix)
    return rows


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def load_all_histories(run_rows: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Expand per-run history files into one table for long learning curves.
    """
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
                "seed": int(row["seed"]),
                "epoch": int(epoch_row["epoch"]),
                "score": float(epoch_row["score"]),
                "val_nll": float(epoch_row["val_nll"]),
                "qty_mae": float(epoch_row["qty_mae"]),
                "dt_mae": float(epoch_row["dt_mae"]),
                "mark_acc": float(epoch_row["mark_acc"]),
                "train_loss": float(epoch_row["train_loss"]),
            })
    return pl.DataFrame(history_rows) if history_rows else pl.DataFrame()


def load_all_scale_metrics(run_rows: list[dict[str, Any]], selections: Iterable[str]) -> pl.DataFrame:
    """
    Combine per-run scale-wise metrics into one long table.
    """
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


def aggregate_run_rows(rows: list[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Summarize run-level sweet-spot metrics across seeds.
    """
    run_df = pl.DataFrame([{key: to_jsonable(value) for key, value in row.items()} for row in rows])
    success_df = run_df.filter(pl.col("status") == "success")
    if success_df.height == 0:
        return run_df, pl.DataFrame()

    summary_df = (
        success_df.group_by(["dataset_name", "model_name"])
        .agg([
            pl.first("dataset_kind").alias("dataset_kind"),
            pl.first("scale_base").alias("scale_base"),
            pl.first("lr").alias("lr"),
            pl.first("batch_size").alias("batch_size"),
            pl.first("lookback_weeks").alias("lookback_weeks"),
            pl.first("max_seq_len").alias("max_seq_len"),
            pl.first("analysis_scale_base").alias("analysis_scale_base"),
            pl.first("titan_profile").alias("titan_profile"),
            pl.first("titan_candidate_name").alias("titan_candidate_name"),
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
        ])
        .sort(["dataset_name", "model_name"])
    )
    return success_df, summary_df


def build_long_delta_table(summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute TitanTPP minus RMTPP deltas under the best-validation-NLL view.
    """
    rows: list[dict[str, Any]] = []
    for dataset_name in summary_df["dataset_name"].unique().to_list():
        dataset_rows = summary_df.filter(pl.col("dataset_name") == dataset_name)
        rmtpp_rows = dataset_rows.filter(pl.col("model_name") == "rmtpp").to_dicts()
        titan_rows = dataset_rows.filter(pl.col("model_name") == "titantpp").to_dicts()
        if not rmtpp_rows or not titan_rows:
            continue

        rmtpp_row = rmtpp_rows[0]
        titan_row = titan_rows[0]
        rows.append({
            "dataset_name": dataset_name,
            "titan_profile": titan_row["titan_profile"],
            "scale_base": titan_row["scale_base"],
            "titan_candidate_name": titan_row["titan_candidate_name"],
            "delta_best_val_nll": float(titan_row["mean_best_val_nll"] - rmtpp_row["mean_best_val_nll"]),
            "delta_best_val_nll_qty_mae": float(
                titan_row["mean_best_val_nll_qty_mae"] - rmtpp_row["mean_best_val_nll_qty_mae"]
            ),
            "delta_best_val_nll_score": float(
                titan_row["mean_best_val_nll_score"] - rmtpp_row["mean_best_val_nll_score"]
            ),
            "delta_best_val_nll_dt_mae": float(
                titan_row["mean_best_val_nll_dt_mae"] - rmtpp_row["mean_best_val_nll_dt_mae"]
            ),
            "delta_best_val_nll_mark_acc": float(
                titan_row["mean_best_val_nll_mark_acc"] - rmtpp_row["mean_best_val_nll_mark_acc"]
            ),
            "delta_best_epoch": float(
                titan_row["mean_best_val_nll_epoch"] - rmtpp_row["mean_best_val_nll_epoch"]
            ),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def aggregate_scale_metrics(scale_df: pl.DataFrame) -> pl.DataFrame:
    """
    Average scale-wise errors across seeds for paper-friendly reporting.
    """
    if scale_df.height == 0:
        return pl.DataFrame()

    return (
        scale_df.group_by(["dataset_name", "model_name", "selection", "scale_order", "scale_label"])
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
        .sort(["dataset_name", "selection", "scale_order", "model_name"])
    )


def save_scale_metric_plots(scale_summary_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Save grouped bar plots for scale-wise quantity MAE and WAPE.
    """
    if scale_summary_df.height == 0:
        return

    ensure_dir(plots_dir)
    colors = {"rmtpp": "#5DA5DA", "titantpp": "#F17CB0"}

    for dataset_name in scale_summary_df["dataset_name"].unique().to_list():
        for selection in scale_summary_df["selection"].unique().to_list():
            dataset_df = scale_summary_df.filter(
                (pl.col("dataset_name") == dataset_name)
                & (pl.col("selection") == selection)
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
            width = 0.36

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for metric, title, ax in [
                ("mean_qty_mae", "Scale-wise Qty MAE", axes[0]),
                ("mean_qty_wape", "Scale-wise WAPE", axes[1]),
            ]:
                for offset, model_name in [(-width / 2, "rmtpp"), (width / 2, "titantpp")]:
                    model_df = dataset_df.filter(pl.col("model_name") == model_name).sort("scale_order")
                    values = model_df[metric].to_list() if model_df.height else [0.0] * len(scale_labels)
                    ax.bar(
                        x + offset,
                        values,
                        width=width,
                        label=model_name.upper(),
                        color=colors[model_name],
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


def save_paper_outputs(
    *,
    summary_df: pl.DataFrame,
    delta_df: pl.DataFrame,
    scale_summary_df: pl.DataFrame,
    paper_dir: Path,
) -> None:
    """
    Save CSV/Parquet/Markdown tables for paper and meeting notes.
    """
    ensure_dir(paper_dir)
    if summary_df.width > 0:
        summary_df.write_csv(paper_dir / "paper_table_long_epoch_metrics.csv")
        summary_df.write_parquet(paper_dir / "paper_table_long_epoch_metrics.parquet")
    if delta_df.width > 0:
        delta_df.write_csv(paper_dir / "paper_table_long_epoch_deltas.csv")
        delta_df.write_parquet(paper_dir / "paper_table_long_epoch_deltas.parquet")
    if scale_summary_df.width > 0:
        scale_summary_df.write_csv(paper_dir / "paper_table_scale_wise_mae.csv")
        scale_summary_df.write_parquet(paper_dir / "paper_table_scale_wise_mae.parquet")

    metrics_md = [
        "# Long-Epoch RMTPP vs TitanTPP Tables",
        "",
        "## Best Validation NLL Summary",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## TitanTPP - RMTPP Delta",
        "",
        markdown_table_from_df(delta_df),
        "",
        "## Scale-wise Quantity Errors",
        "",
        markdown_table_from_df(scale_summary_df),
        "",
    ]
    (paper_dir / "paper_table_long_epoch_metrics.md").write_text(
        "\n".join(metrics_md),
        encoding="utf-8",
    )


def save_text_report(
    *,
    summary_df: pl.DataFrame,
    delta_df: pl.DataFrame,
    scale_summary_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a meeting-ready report around the professor's two questions.
    """
    lines = [
        "# Long-Epoch and Scale-wise MAE Report",
        "",
        "## Purpose",
        "",
        "This follow-up experiment answers two validation questions:",
        "",
        "- whether RMTPP and TitanTPP have actually converged, or whether validation NLL keeps improving beyond the 30-epoch benchmark",
        "- whether the large overall quantity MAE is dominated by large-demand events, hiding behavior on smaller demand scales",
        "",
        "## Selection Rule",
        "",
        "The primary model-selection point is `best_val_nll`, not the final epoch. This is the sweet spot requested for checking enough training and possible overfitting.",
        "",
        "## Best Validation NLL Summary",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## TitanTPP - RMTPP Delta",
        "",
        markdown_table_from_df(delta_df),
        "",
        "Interpretation guide: negative `delta_best_val_nll` and negative `delta_best_val_nll_qty_mae` favor TitanTPP, while positive `delta_best_val_nll_score` favors TitanTPP.",
        "",
        "## Scale-wise Quantity MAE",
        "",
        "Scale buckets are computed from the true reconstructed quantity as `floor(log10(true_qty))` by default. The final bucket is a tail bucket.",
        "",
        markdown_table_from_df(scale_summary_df),
        "",
        "## Files to Check",
        "",
        "- `leaderboard/long_epoch_histories.csv`: epoch-by-epoch learning curves",
        "- `paper_outputs/plots/*learning_curves.png`: convergence and overfitting check",
        "- `paper_outputs/paper_table_scale_wise_mae.csv`: scale-wise quantity error table",
        "- `paper_outputs/plots/*scale_wise_qty_errors.png`: scale-wise MAE/WAPE plots",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse long-run validation settings.
    """
    parser = argparse.ArgumentParser(
        description="Run long-epoch RMTPP vs TitanTPP validation with scale-wise MAE."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT / "search_artifacts" / "titan_rmtpp_long_epoch_scale_eval"),
        help="Directory where long-run artifacts will be written.",
    )
    parser.add_argument(
        "--datasets",
        default="intermittent,yellow_trip",
        help="Comma-separated dataset names to evaluate.",
    )
    parser.add_argument(
        "--titan-profile",
        default="dataset_best",
        choices=["dataset_best", "overall", "score_priority"],
        help="Which report-derived Titan default set to use.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=DEFAULT_LONG_EPOCHS)
    parser.add_argument("--lr", type=float, default=float(LongEpochConfig.__dataclass_fields__["lr"].default))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated random seeds.")
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--intermittent-max-series", type=int, default=None)
    parser.add_argument("--yellow-max-series", type=int, default=None)
    parser.add_argument("--analysis-scale-base", type=float, default=10.0)
    parser.add_argument(
        "--analysis-tail-order",
        type=int,
        default=4,
        help="Final scale bucket is >= analysis_scale_base ** analysis_tail_order.",
    )
    parser.add_argument(
        "--eval-selections",
        default="best_val_nll",
        help="Comma-separated checkpoint selections for scale-wise metrics: best_val_nll,best_score,final.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Run long-epoch training, aggregate results, and export meeting-ready files.
    """
    args = parse_args()
    selected_dataset_names = {name.strip() for name in args.datasets.split(",") if name.strip()}
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    eval_selections = tuple(
        selection.strip()
        for selection in args.eval_selections.split(",")
        if selection.strip()
    )
    allowed_selections = {"best_val_nll", "best_score", "final"}
    unsupported = sorted(set(eval_selections) - allowed_selections)
    if unsupported:
        raise ValueError(f"Unsupported eval selections: {unsupported}")

    long_cfg = LongEpochConfig(
        base_dir=args.base_dir,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seeds=seeds,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        titan_profile=args.titan_profile,
        intermittent_max_series=args.intermittent_max_series,
        yellow_max_series=args.yellow_max_series,
        analysis_scale_base=args.analysis_scale_base,
        analysis_tail_order=args.analysis_tail_order,
        eval_selections=eval_selections,
        force_rerun=args.force_rerun,
        stop_on_error=args.stop_on_error,
    )
    print(f"LongEpoch Configuration:: {long_cfg}")

    dataset_specs = make_dataset_specs(long_cfg, selected_dataset_names)
    if not dataset_specs:
        raise ValueError("No datasets selected for long-epoch validation.")

    profile_map = default_profile_map(long_cfg.titan_profile)
    profile_map = {name: profile_map[name] for name in selected_dataset_names}

    base_dir = ensure_dir(Path(long_cfg.base_dir))
    leaderboard_dir = ensure_dir(base_dir / "leaderboard")
    paper_dir = ensure_dir(base_dir / "paper_outputs")
    plots_dir = ensure_dir(paper_dir / "plots")
    logger = build_logger(base_dir / "long_epoch_scale_eval.log", "titan_rmtpp_long_epoch_scale_eval")

    save_json(
        {
            "long_epoch_config": long_cfg,
            "dataset_effective_search_configs": {
                spec.name: make_search_cfg(long_cfg, spec.kind)
                for spec in dataset_specs
            },
            "dataset_specs": dataset_specs,
            "titan_profile_map": profile_map,
            "candidates": default_titan_candidates(),
        },
        base_dir / "long_epoch_manifest.json",
    )

    logger.info("Preparing marked datasets for profile=%s", long_cfg.titan_profile)
    marked_cache = build_marked_cache(
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        ab_cfg=long_cfg,
        logger=logger,
    )

    run_rows = run_long_benchmark(
        long_cfg=long_cfg,
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        marked_cache=marked_cache,
        logger=logger,
    )

    run_df, summary_df = aggregate_run_rows(run_rows)
    delta_df = build_long_delta_table(summary_df) if summary_df.height > 0 else pl.DataFrame()
    history_df = load_all_histories(run_rows)
    scale_df = load_all_scale_metrics(run_rows, long_cfg.eval_selections)
    scale_summary_df = aggregate_scale_metrics(scale_df)

    if run_df.width > 0:
        run_df.write_parquet(leaderboard_dir / "long_epoch_runs.parquet")
        run_df.write_csv(leaderboard_dir / "long_epoch_runs.csv")
    if summary_df.width > 0:
        summary_df.write_parquet(leaderboard_dir / "long_epoch_summary.parquet")
        summary_df.write_csv(leaderboard_dir / "long_epoch_summary.csv")
    if delta_df.width > 0:
        delta_df.write_parquet(leaderboard_dir / "long_epoch_deltas.parquet")
        delta_df.write_csv(leaderboard_dir / "long_epoch_deltas.csv")
    if history_df.width > 0:
        history_df.write_parquet(leaderboard_dir / "long_epoch_histories.parquet")
        history_df.write_csv(leaderboard_dir / "long_epoch_histories.csv")
    if scale_df.width > 0:
        scale_df.write_parquet(leaderboard_dir / "scale_wise_metrics.parquet")
        scale_df.write_csv(leaderboard_dir / "scale_wise_metrics.csv")
    if scale_summary_df.width > 0:
        scale_summary_df.write_parquet(leaderboard_dir / "scale_wise_summary.parquet")
        scale_summary_df.write_csv(leaderboard_dir / "scale_wise_summary.csv")

    save_learning_curve_plots(history_df, plots_dir)
    save_scale_metric_plots(scale_summary_df, plots_dir)
    save_paper_outputs(
        summary_df=summary_df,
        delta_df=delta_df,
        scale_summary_df=scale_summary_df,
        paper_dir=paper_dir,
    )
    save_text_report(
        summary_df=summary_df,
        delta_df=delta_df,
        scale_summary_df=scale_summary_df,
        output_path=paper_dir / "long_epoch_scale_report.md",
    )

    logger.info("Long-epoch validation complete. Summary rows:\n%s", summary_df)


if __name__ == "__main__":
    main()


# validation metric을 어떻게 할건지
# baseline으로 mark를 잘 맞추는 문제로..
# training에 사용하지 않았던 테스트들 갖고...
# 다음 이벤트가 어떤 mark ㄷ
