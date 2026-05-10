"""
Dataset-level A/B benchmark between RMTPP and TitanTPP using the best Titan
configurations discovered by the hyperparameter search report.

Why this script exists:
1. pick the strongest Titan setup discovered so far
2. compare it against the plain RMTPP baseline under the same data split
3. save reusable artifacts for the paper: run tables, summary tables, and plots

The default profile follows the analysis report:
- `intermittent`: `log10 + mid_deep_lmm`
- `yellow_trip`: `log10 + mid_lmm`

This script intentionally keeps the loss definition fixed at
`residual_only` so the headline RMTPP vs TitanTPP comparison stays aligned
with the legacy benchmark setup. Quantity-loss variants are explored in the
separate `tpp_qty_loss_ablation.py` runner.

Alternative profiles are also supported so we can compare:
- `overall`: use the single global best Titan profile for both datasets
- `score_priority`: use the report's score-oriented yellow-trip choice
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

from models.RMTPPs.config import RMTPPConfig
from simple_lab_test.search.titan_hparam_search import (
    DatasetSpec,
    SearchConfig,
    TitanCandidate,
    build_logger,
    build_rmtpp_config,
    build_titan_config,
    build_training_config,
    default_dataset_specs,
    default_titan_candidates,
    ensure_dir,
    prepare_marked_dataset,
    sanitize_float_label,
    save_json,
    summarize_history,
    tee_training_output,
    to_jsonable,
)
from utils.training import train_rmtpp, train_titantpp


# ---------------------------------------------------------------------------
# Report-driven configuration presets
# ---------------------------------------------------------------------------

BEST_TITAN_BY_DATASET = {
    "intermittent": {"scale_base": 10.0, "candidate_name": "mid_deep_lmm"},
    "yellow_trip": {"scale_base": 10.0, "candidate_name": "mid_lmm"},
}

BEST_TITAN_OVERALL = {
    "intermittent": {"scale_base": 10.0, "candidate_name": "mid_lmm"},
    "yellow_trip": {"scale_base": 10.0, "candidate_name": "mid_lmm"},
}

BEST_TITAN_SCORE_PRIORITY = {
    "intermittent": {"scale_base": 10.0, "candidate_name": "mid_deep_lmm"},
    "yellow_trip": {"scale_base": 4.0, "candidate_name": "mid_deep_lmm"},
}


# ---------------------------------------------------------------------------
# Runtime dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ABConfig:
    """
    Runtime options shared by all A/B runs.
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
    grad_clip: float = 1.0
    epochs: int = 30
    seeds: tuple[int, ...] = (42, 52, 62)
    titan_profile: str = "dataset_best"
    intermittent_max_series: int | None = None
    yellow_max_series: int | None = None
    force_rerun: bool = False
    stop_on_error: bool = False
    rmtpp_rnn_type: str = "gru"
    rmtpp_mark_emb_dim: int = 32
    value_head_activation: str = "sigmoid"
    loss_mode: str = "residual_only"


DEFAULT_AB_EPOCHS = int(ABConfig.__dataclass_fields__["epochs"].default)


@dataclass(frozen=True)
class ABRunConfig:
    """
    Full specification for one model/dataset/seed benchmark run.
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
class ABRunPaths:
    """
    Canonical filesystem layout for one benchmark run.
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


def make_search_cfg(ab_cfg: ABConfig) -> SearchConfig:
    """
    Reuse the existing SearchConfig so preprocessing/cache utilities stay shared.
    """
    return SearchConfig(
        base_dir=ab_cfg.base_dir,
        device=ab_cfg.device,
        lookback_weeks=ab_cfg.lookback_weeks,
        max_seq_len=ab_cfg.max_seq_len,
        batch_size=ab_cfg.batch_size,
        lr=ab_cfg.lr,
        val_ratio=ab_cfg.val_ratio,
        lambda_value=ab_cfg.lambda_value,
        lambda_dt=ab_cfg.lambda_dt,
        grad_clip=ab_cfg.grad_clip,
        force_rerun=ab_cfg.force_rerun,
        stop_on_error=ab_cfg.stop_on_error,
    )


def make_training_cfg(ab_cfg: ABConfig) -> Any:
    """
    Build the shared trainer config used by both RMTPP and TitanTPP.
    """
    search_cfg = make_search_cfg(ab_cfg)
    return build_training_config(search_cfg, epochs=ab_cfg.epochs)


def make_dataset_specs(ab_cfg: ABConfig, selected_names: set[str]) -> list[DatasetSpec]:
    """
    Attach dataset-size overrides while preserving the shared dataset builders.
    """
    specs: list[DatasetSpec] = []
    for spec in default_dataset_specs():
        if spec.name not in selected_names:
            continue
        if spec.name == "intermittent":
            spec = DatasetSpec(**{**asdict(spec), "max_series": ab_cfg.intermittent_max_series})
        elif spec.name == "yellow_trip":
            spec = DatasetSpec(**{**asdict(spec), "max_series": ab_cfg.yellow_max_series})
        specs.append(spec)
    return specs


def build_ab_run_paths(ab_cfg: ABConfig, run_cfg: ABRunConfig) -> ABRunPaths:
    """
    Keep model-specific runs organized under a predictable folder tree.
    """
    base_label = sanitize_float_label(run_cfg.scale_base)
    run_dir = (
        Path(ab_cfg.base_dir)
        / "runs"
        / run_cfg.dataset_name
        / run_cfg.model_name
        / f"lossmode_{ab_cfg.loss_mode}"
        / f"profile_{run_cfg.titan_profile}"
        / f"base_{base_label}"
        / run_cfg.titan_candidate_name
        / f"epochs_{run_cfg.epochs}"
        / f"seed_{run_cfg.seed}"
    )
    return ABRunPaths(
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


def build_marked_cache(
    *,
    dataset_specs: list[DatasetSpec],
    profile_map: dict[str, dict[str, Any]],
    ab_cfg: ABConfig,
    logger: logging.Logger,
) -> dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]]:
    """
    Prepare the exact dataset/base pairs needed by the chosen Titan profile.

    RMTPP uses the same `scale_base` as Titan for a fair A/B comparison on each
    dataset, so we only cache the bases that the selected profile actually uses.
    """
    search_cfg = make_search_cfg(ab_cfg)
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


# ---------------------------------------------------------------------------
# Model execution
# ---------------------------------------------------------------------------

def train_one_model(
    *,
    ab_cfg: ABConfig,
    run_cfg: ABRunConfig,
    marked_df: pl.DataFrame,
    marked_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Train exactly one RMTPP or TitanTPP run and persist its artifacts.
    """
    run_paths = build_ab_run_paths(ab_cfg, run_cfg)
    summary_path = run_paths.metrics_dir / "summary.json"
    history_json_path = run_paths.metrics_dir / "history.json"
    history_parquet_path = run_paths.metrics_dir / "history.parquet"
    checkpoint_path = run_paths.checkpoint_dir / "best_model.pt"
    log_path = run_paths.logs_dir / "train.log"
    manifest_path = run_paths.manifest_dir / "run_config.json"

    if (
        not ab_cfg.force_rerun
        and summary_path.exists()
        and history_json_path.exists()
        and checkpoint_path.exists()
    ):
        with open(summary_path, "r", encoding="utf-8") as f:
            cached_summary = json.load(f)
        if int(cached_summary.get("epochs", -1)) == int(run_cfg.epochs):
            return cached_summary

    set_global_seed(run_cfg.seed)
    search_cfg = make_search_cfg(ab_cfg)
    training_cfg = make_training_cfg(ab_cfg)

    # RMTPP shares the same mark vocabulary and scale base as Titan so the only
    # real change in the A/B test is the sequence encoder architecture.
    rmtpp_cfg = build_rmtpp_config(
        search_cfg,
        num_marks=int(marked_meta["num_marks"]),
        scale_base=run_cfg.scale_base,
    )
    rmtpp_cfg = RMTPPConfig(
        **{
            **asdict(rmtpp_cfg),
            "rnn_type": ab_cfg.rmtpp_rnn_type,
            "mark_emb_dim": ab_cfg.rmtpp_mark_emb_dim,
            "rnn_hidden_dim": run_cfg.titan_candidate.d_model,
            # The main paper benchmark intentionally fixes both models to the
            # legacy residual-only objective. Loss-type exploration lives in
            # the dedicated qty-loss ablation runner.
            "loss_mode": ab_cfg.loss_mode,
        }
    )
    titan_cfg = build_titan_config(search_cfg, run_cfg.titan_candidate)

    save_json(
        {
            "ab_config": ab_cfg,
            "run_config": run_cfg,
            "training_config": training_cfg,
            "rmtpp_config": rmtpp_cfg,
            "titan_config": titan_cfg,
            "marked_meta": marked_meta,
        },
        manifest_path,
    )

    with tee_training_output(log_path):
        if run_cfg.model_name == "rmtpp":
            model, info = train_rmtpp(
                marked_df=marked_df,
                training_config=training_cfg,
                rmtpp_config=rmtpp_cfg,
            )
        elif run_cfg.model_name == "titantpp":
            model, info = train_titantpp(
                marked_df=marked_df,
                training_config=training_cfg,
                rmtpp_config=rmtpp_cfg,
                titan_config=titan_cfg,
            )
        else:
            raise ValueError(f"Unsupported model_name: {run_cfg.model_name}")

    history = info["history"]
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
        "scale_base": float(run_cfg.scale_base),
        "titan_profile": run_cfg.titan_profile,
        "loss_mode": ab_cfg.loss_mode,
        "titan_candidate_name": run_cfg.titan_candidate_name,
        "run_dir": str(run_paths.run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_marks": int(marked_meta["num_marks"]),
        "max_order": int(marked_meta["max_order"]),
        "series_count": int(marked_meta["series_count"]),
        "rmtpp_rnn_type": ab_cfg.rmtpp_rnn_type,
        "rmtpp_hidden_dim": int(rmtpp_cfg.rnn_hidden_dim),
        "rmtpp_mark_emb_dim": int(rmtpp_cfg.mark_emb_dim),
        **flatten_candidate(run_cfg.titan_candidate),
        **summarize_history(history),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ab_config": to_jsonable(ab_cfg),
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


def build_error_row(run_cfg: ABRunConfig, exc: Exception) -> dict[str, Any]:
    """
    Convert a failed run into a durable row instead of losing that failure.
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
        "loss_mode": ab_cfg.loss_mode,
        "titan_candidate_name": run_cfg.titan_candidate_name,
        "error": repr(exc),
        **flatten_candidate(run_cfg.titan_candidate),
    }


def run_benchmark(
    *,
    ab_cfg: ABConfig,
    dataset_specs: list[DatasetSpec],
    profile_map: dict[str, dict[str, Any]],
    marked_cache: dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Run RMTPP and TitanTPP on every dataset/seed pair under the chosen profile.
    """
    all_candidates = default_titan_candidates()
    rows: list[dict[str, Any]] = []

    leaderboard_dir = ensure_dir(Path(ab_cfg.base_dir) / "leaderboard")
    path_prefix = leaderboard_dir / "ab_runs"

    total_runs = len(dataset_specs) * len(ab_cfg.seeds) * 2
    completed = 0

    for spec in dataset_specs:
        profile = profile_map[spec.name]
        scale_base = float(profile["scale_base"])
        candidate = find_candidate_by_name(all_candidates, str(profile["candidate_name"]))
        marked_df, marked_meta = marked_cache[(spec.name, scale_base)]

        for seed in ab_cfg.seeds:
            for model_name in ("rmtpp", "titantpp"):
                completed += 1
                logger.info(
                    "A/B run %s/%s | dataset=%s | model=%s | base=%s | titan_candidate=%s | seed=%s",
                    completed,
                    total_runs,
                    spec.name,
                    model_name,
                    scale_base,
                    candidate.name,
                    seed,
                )
                run_cfg = ABRunConfig(
                    dataset_name=spec.name,
                    dataset_kind=spec.kind,
                    model_name=model_name,
                    seed=seed,
                    epochs=ab_cfg.epochs,
                    scale_base=scale_base,
                    titan_profile=ab_cfg.titan_profile,
                    titan_candidate_name=candidate.name,
                    titan_candidate=candidate,
                )
                try:
                    row = train_one_model(
                        ab_cfg=ab_cfg,
                        run_cfg=run_cfg,
                        marked_df=marked_df,
                        marked_meta=marked_meta,
                    )
                except Exception as exc:
                    row = build_error_row(run_cfg, exc)
                    logger.exception(
                        "A/B run failed | dataset=%s model=%s base=%s seed=%s",
                        spec.name,
                        model_name,
                        scale_base,
                        seed,
                    )
                    if ab_cfg.stop_on_error:
                        raise
                rows.append(row)
                persist_rows(rows, path_prefix)
    return rows


# ---------------------------------------------------------------------------
# Aggregation, paper tables, and plots
# ---------------------------------------------------------------------------

METRIC_COLUMNS = [
    "best_score",
    "best_val_nll",
    "best_qty_mae",
    "best_dt_mae",
    "best_mark_acc",
    "best_value_mae",
]


def aggregate_run_rows(rows: list[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build run-level and aggregated model-level tables for reporting.
    """
    run_df = pl.DataFrame([{k: to_jsonable(v) for k, v in row.items()} for row in rows])
    success_df = run_df.filter(pl.col("status") == "success")
    if success_df.height == 0:
        return run_df, pl.DataFrame()

    summary_df = (
        success_df.group_by(["dataset_name", "model_name"])
        .agg([
            pl.first("dataset_kind").alias("dataset_kind"),
            pl.first("scale_base").alias("scale_base"),
            pl.first("titan_profile").alias("titan_profile"),
            pl.first("titan_candidate_name").alias("titan_candidate_name"),
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
        .sort(["dataset_name", "model_name"])
    )
    return success_df, summary_df


def build_delta_table(summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Summarize TitanTPP's gain/loss against RMTPP for each dataset.
    """
    rows: list[dict[str, Any]] = []
    for dataset_name in summary_df["dataset_name"].unique().to_list():
        dataset_rows = summary_df.filter(pl.col("dataset_name") == dataset_name)
        if dataset_rows.height < 2:
            continue

        rmtpp_row = dataset_rows.filter(pl.col("model_name") == "rmtpp").to_dicts()
        titan_row = dataset_rows.filter(pl.col("model_name") == "titantpp").to_dicts()
        if not rmtpp_row or not titan_row:
            continue

        rmtpp_row = rmtpp_row[0]
        titan_row = titan_row[0]
        rows.append({
            "dataset_name": dataset_name,
            "titan_profile": titan_row["titan_profile"],
            "scale_base": titan_row["scale_base"],
            "titan_candidate_name": titan_row["titan_candidate_name"],
            "delta_best_score": float(titan_row["mean_best_score"] - rmtpp_row["mean_best_score"]),
            "delta_best_val_nll": float(titan_row["mean_best_val_nll"] - rmtpp_row["mean_best_val_nll"]),
            "delta_best_qty_mae": float(titan_row["mean_best_qty_mae"] - rmtpp_row["mean_best_qty_mae"]),
            "delta_best_dt_mae": float(titan_row["mean_best_dt_mae"] - rmtpp_row["mean_best_dt_mae"]),
            "delta_best_mark_acc": float(titan_row["mean_best_mark_acc"] - rmtpp_row["mean_best_mark_acc"]),
            "delta_best_value_mae": float(titan_row["mean_best_value_mae"] - rmtpp_row["mean_best_value_mae"]),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def markdown_table_from_df(df: pl.DataFrame) -> str:
    """
    Render a small DataFrame as markdown without introducing extra dependencies.
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

    report_lines = [
        "# RMTPP vs TitanTPP A/B Summary",
        "",
        "## Metrics Table",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## TitanTPP - RMTPP Delta",
        "",
        markdown_table_from_df(delta_df),
        "",
    ]
    (output_dir / "paper_table_metrics.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def save_metric_bar_plots(summary_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Save one paper-friendly 2x3 metric grid for each dataset.
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

    for dataset_name in summary_df["dataset_name"].unique().to_list():
        dataset_df = summary_df.filter(pl.col("dataset_name") == dataset_name).sort("model_name")
        models = dataset_df["model_name"].to_list()

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.ravel()

        for ax, (value_col, std_col, title) in zip(axes, metrics):
            values = dataset_df[value_col].to_list()
            errors = dataset_df[std_col].to_list()
            ax.bar(models, values, yerr=errors, color=["#5DA5DA", "#F17CB0"], capsize=6)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

        fig.suptitle(f"{dataset_name}: RMTPP vs TitanTPP", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plots_dir / f"{dataset_name}_metric_grid.png", dpi=200, bbox_inches="tight")
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
                "seed": row["seed"],
                "epoch": int(epoch_row["epoch"]),
                "score": float(epoch_row["score"]),
                "val_nll": float(epoch_row["val_nll"]),
                "qty_mae": float(epoch_row["qty_mae"]),
                "dt_mae": float(epoch_row["dt_mae"]),
                "mark_acc": float(epoch_row["mark_acc"]),
            })
    return pl.DataFrame(history_rows) if history_rows else pl.DataFrame()


def save_learning_curve_plots(history_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Plot mean +/- std learning curves across seeds for both models.
    """
    if history_df.height == 0:
        return

    ensure_dir(plots_dir)
    curve_metrics = [
        ("score", "Validation Score"),
        ("val_nll", "Validation NLL"),
        ("qty_mae", "Validation Qty MAE"),
    ]

    for dataset_name in history_df["dataset_name"].unique().to_list():
        dataset_df = history_df.filter(pl.col("dataset_name") == dataset_name)
        fig, axes = plt.subplots(1, len(curve_metrics), figsize=(18, 5))

        for ax, (metric, title) in zip(axes, curve_metrics):
            for model_name, color in (("rmtpp", "#5DA5DA"), ("titantpp", "#F17CB0")):
                model_df = dataset_df.filter(pl.col("model_name") == model_name)
                if model_df.height == 0:
                    continue
                agg_df = (
                    model_df.group_by("epoch")
                    .agg([
                        pl.mean(metric).alias("mean_metric"),
                        pl.std(metric).fill_null(0.0).alias("std_metric"),
                    ])
                    .sort("epoch")
                )
                epochs = agg_df["epoch"].to_list()
                mean_values = agg_df["mean_metric"].to_list()
                std_values = agg_df["std_metric"].to_list()
                ax.plot(epochs, mean_values, label=model_name.upper(), color=color, linewidth=2)
                lower = np.array(mean_values) - np.array(std_values)
                upper = np.array(mean_values) + np.array(std_values)
                ax.fill_between(epochs, lower, upper, color=color, alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
            ax.legend()

        fig.suptitle(f"{dataset_name}: Learning Curves", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plots_dir / f"{dataset_name}_learning_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_text_summary(
    *,
    summary_df: pl.DataFrame,
    delta_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """
    Produce a short narrative summary for quick paper/report drafting.
    """
    lines = [
        "# RMTPP vs TitanTPP A/B Analysis",
        "",
        "## Main Table",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## TitanTPP Gain/Loss vs RMTPP",
        "",
        markdown_table_from_df(delta_df),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse benchmark runtime settings.
    """
    parser = argparse.ArgumentParser(
        description="Run RMTPP vs TitanTPP A/B benchmarks using report-derived Titan configs."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT / "search_artifacts" / "titan_rmtpp_ab_test"),
        help="Directory where benchmark artifacts will be written.",
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
    parser.add_argument("--epochs", type=int, default=DEFAULT_AB_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated random seeds.")
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--intermittent-max-series", type=int, default=None)
    parser.add_argument("--yellow-max-series", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Run the full A/B benchmark and export paper-friendly artifacts.
    """
    args = parse_args()
    selected_dataset_names = {name.strip() for name in args.datasets.split(",") if name.strip()}
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())

    ab_cfg = ABConfig(
        base_dir=args.base_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=seeds,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        titan_profile=args.titan_profile,
        intermittent_max_series=args.intermittent_max_series,
        yellow_max_series=args.yellow_max_series,
        force_rerun=args.force_rerun,
        stop_on_error=args.stop_on_error,
    )

    print(f'AB Configuration:: {ab_cfg}')

    dataset_specs = make_dataset_specs(ab_cfg, selected_dataset_names)
    if not dataset_specs:
        raise ValueError("No datasets selected for A/B testing.")

    profile_map = default_profile_map(ab_cfg.titan_profile)
    profile_map = {name: profile_map[name] for name in selected_dataset_names}

    base_dir = ensure_dir(Path(ab_cfg.base_dir))
    leaderboard_dir = ensure_dir(base_dir / "leaderboard")
    paper_dir = ensure_dir(base_dir / "paper_outputs")
    plots_dir = ensure_dir(paper_dir / "plots")
    logger = build_logger(base_dir / "ab_test.log", "titan_rmtpp_ab_test")

    candidates = default_titan_candidates()
    save_json(
        {
            "ab_config": ab_cfg,
            "dataset_specs": dataset_specs,
            "titan_profile_map": profile_map,
            "candidates": candidates,
        },
        base_dir / "ab_manifest.json",
    )

    logger.info("Preparing marked datasets for profile=%s", ab_cfg.titan_profile)
    marked_cache = build_marked_cache(
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        ab_cfg=ab_cfg,
        logger=logger,
    )

    run_rows = run_benchmark(
        ab_cfg=ab_cfg,
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        marked_cache=marked_cache,
        logger=logger,
    )

    run_df, summary_df = aggregate_run_rows(run_rows)
    delta_df = build_delta_table(summary_df)

    if run_df.width > 0:
        run_df.write_parquet(leaderboard_dir / "ab_runs.parquet")
        run_df.write_csv(leaderboard_dir / "ab_runs.csv")
    if summary_df.width > 0:
        summary_df.write_parquet(leaderboard_dir / "ab_summary.parquet")
        summary_df.write_csv(leaderboard_dir / "ab_summary.csv")
    if delta_df.width > 0:
        delta_df.write_parquet(leaderboard_dir / "ab_deltas.parquet")
        delta_df.write_csv(leaderboard_dir / "ab_deltas.csv")

    save_paper_tables(summary_df=summary_df, delta_df=delta_df, output_dir=paper_dir)
    save_metric_bar_plots(summary_df, plots_dir)
    history_df = load_all_histories(run_rows)
    if history_df.height > 0:
        history_df.write_parquet(leaderboard_dir / "ab_histories.parquet")
        history_df.write_csv(leaderboard_dir / "ab_histories.csv")
    save_learning_curve_plots(history_df, plots_dir)
    save_text_summary(
        summary_df=summary_df,
        delta_df=delta_df,
        output_path=paper_dir / "ab_analysis_summary.md",
    )

    logger.info("A/B benchmark complete. Summary rows:\n%s", summary_df)


if __name__ == "__main__":
    main()
