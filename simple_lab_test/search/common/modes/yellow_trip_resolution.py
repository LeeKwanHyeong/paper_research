"""
Yellow-trip daily/hourly RMTPP vs TitanTPP benchmark.

Why this script exists:
1. the previous yellow-trip benchmark used weekly grid-cell counts
2. weekly preprocessing produced very short sequences, so RMTPP was already a
   strong baseline and TitanTPP had little temporal context to exploit
3. this runner rebuilds yellow-trip as daily/hourly marked event sequences and
   checks whether longer, more heterogeneous sequences make TitanTPP useful

The training loop intentionally reuses the long-epoch helper hosted in
`simple_lab_test.search.common.modes.long_epoch_legacy`.
That keeps checkpoints, validation-NLL decomposition, and scale-wise quantity
metrics identical to the long-epoch validation script.
"""

from __future__ import annotations

import argparse
import json
import os
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
    Keep remote SSH/Jupyter logs readable even under a non-UTF-8 locale.
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

from simple_lab_test.search.titan_hparam_search import (
    TitanCandidate,
    build_logger,
    default_titan_candidates,
    ensure_dir,
    sanitize_float_label,
    save_json,
    to_jsonable,
)
from simple_lab_test.search.titan_rmtpp_ab_test import (
    find_candidate_by_name,
    markdown_table_from_df,
    persist_rows,
)
from simple_lab_test.search.common.modes.long_epoch_legacy import (
    LongEpochConfig,
    LongRunConfig,
    aggregate_scale_metrics,
    build_error_row,
    train_one_long_run,
)
from utils.magnitude_pipeline import build_magnitude_marked_df


# ---------------------------------------------------------------------------
# Experiment dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YellowResolutionSpec:
    """
    Dataset construction knobs for one yellow-trip temporal resolution.

    `seq` is created from sorted global time buckets, so `delta_t` later means
    elapsed day/hour buckets between two positive events in the same grid cell.
    """
    resolution: str
    parquet_path: str
    grid_size_deg: float = 0.02
    min_active_buckets: int = 20
    max_series: int | None = 1000
    vendor: int = 0
    min_lon: float = -75.0
    max_lon: float = -72.0
    min_lat: float = 40.0
    max_lat: float = 42.0
    max_raw_rows: int | None = None


@dataclass(frozen=True)
class ResolutionRuntimeConfig:
    """
    Runtime options shared across all resolution experiments.
    """
    base_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scale_base: float = 10.0
    epochs: int = 100
    seeds: tuple[int, ...] = (42, 52, 62)
    batch_size: int = 128
    lr: float = 1e-3
    val_ratio: float = 0.2
    lambda_value: float = 1.0
    lambda_dt: float = 1.0
    grad_clip: float = 1.0
    max_seq_len: int = 256
    daily_lookback_buckets: int = 90
    hourly_lookback_buckets: int = 168
    rmtpp_rnn_type: str = "gru"
    rmtpp_hidden_dim: int = 128
    rmtpp_mark_emb_dim: int = 32
    loss_mode: str = "residual_only"
    analysis_scale_base: float = 10.0
    analysis_tail_order: int = 4
    eval_selections: tuple[str, ...] = ("best_val_nll",)
    force_rerun: bool = False
    stop_on_error: bool = False


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------

def parse_csv(value: str) -> tuple[str, ...]:
    """
    Parse comma-separated CLI values while dropping empty tokens.
    """
    return tuple(token.strip() for token in value.split(",") if token.strip())


def parse_seeds(value: str) -> tuple[int, ...]:
    """
    Convert `42,52,62` into a reproducible seed tuple.
    """
    seeds = tuple(int(token) for token in parse_csv(value))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def zero_to_none(value: int | None) -> int | None:
    """
    Interpret CLI value 0 as "do not cap".
    """
    if value is None:
        return None
    return None if int(value) <= 0 else int(value)


def lookback_for_resolution(runtime_cfg: ResolutionRuntimeConfig, resolution: str) -> int:
    """
    Choose a context window in the same unit as the temporal resolution.
    """
    if resolution == "daily":
        return int(runtime_cfg.daily_lookback_buckets)
    if resolution == "hourly":
        return int(runtime_cfg.hourly_lookback_buckets)
    raise ValueError(f"Unsupported resolution: {resolution}")


def make_rmtpp_proxy_candidate(runtime_cfg: ResolutionRuntimeConfig) -> TitanCandidate:
    """
    Reuse the long-run builder, which reads hidden size from TitanCandidate.

    This proxy is not a Titan architecture. It only carries RMTPP hidden-dim
    metadata through the shared run config without adding another helper layer.
    """
    hidden_dim = int(runtime_cfg.rmtpp_hidden_dim)
    return TitanCandidate(
        name=f"rmtpp_{runtime_cfg.rmtpp_rnn_type}_h{hidden_dim}",
        d_model=hidden_dim,
        n_layers=1,
        n_heads=1,
        d_ff=hidden_dim,
        dropout=0.1,
        contextual_mem_size=0,
        persistent_mem_size=0,
        use_lmm=False,
        mem_size=0,
        mem_topk=0,
        use_pos_emb=False,
        use_causal=True,
    )


# ---------------------------------------------------------------------------
# Yellow-trip event construction
# ---------------------------------------------------------------------------

def resolution_every(resolution: str) -> str:
    """
    Polars truncate interval for the requested temporal bucket.
    """
    if resolution == "daily":
        return "1d"
    if resolution == "hourly":
        return "1h"
    raise ValueError(f"Unsupported resolution: {resolution}")


def resolution_label(spec: YellowResolutionSpec) -> str:
    """
    Stable label used in run paths and leaderboards.
    """
    grid_label = sanitize_float_label(spec.grid_size_deg)
    series_label = "all" if spec.max_series is None else str(spec.max_series)
    raw_label = "allraw" if spec.max_raw_rows is None else f"raw{spec.max_raw_rows}"
    return (
        f"yellow_trip_{spec.resolution}"
        f"_vendor_{spec.vendor}"
        f"_grid_{grid_label}"
        f"_minb_{spec.min_active_buckets}"
        f"_entities_{series_label}"
        f"_{raw_label}"
    )


def load_yellow_trip_raw(spec: YellowResolutionSpec) -> pl.DataFrame:
    """
    Read the raw parquet, optionally limiting rows for fast smoke tests.
    """
    path = Path(spec.parquet_path)
    if spec.max_raw_rows is not None:
        return pl.scan_parquet(path).head(int(spec.max_raw_rows)).collect()
    return pl.read_parquet(path)


def pickup_datetime_expr(raw_df: pl.DataFrame) -> pl.Expr:
    """
    Parse pickup timestamps robustly for local/server parquet variations.
    """
    dtype = raw_df.schema.get("tpep_pickup_datetime")
    expr = pl.col("tpep_pickup_datetime")
    if dtype == pl.Utf8:
        return expr.str.strptime(pl.Datetime, strict=False)
    return expr.cast(pl.Datetime, strict=False)


def prepare_yellow_resolution_events(
    spec: YellowResolutionSpec,
    *,
    logger,
) -> pl.DataFrame:
    """
    Convert raw trips into positive-demand grid-cell event sequences.

    Each row means: one grid cell had `demand_qty` pickups in one day/hour
    bucket. Buckets with zero pickups are not events; their absence is captured
    by the next positive event's `delta_t`.
    """
    raw_df = load_yellow_trip_raw(spec)
    required_cols = {
        "tpep_pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
    }
    missing = sorted(required_cols - set(raw_df.columns))
    if missing:
        raise ValueError(f"yellow_trip parquet is missing required columns: {missing}")

    if spec.vendor in (1, 2) and "VendorID" in raw_df.columns:
        raw_df = raw_df.filter(pl.col("VendorID") == int(spec.vendor))

    bucket_every = resolution_every(spec.resolution)
    logger.info(
        "Aggregating yellow_trip | resolution=%s | grid=%s | min_active=%s | max_series=%s",
        spec.resolution,
        spec.grid_size_deg,
        spec.min_active_buckets,
        spec.max_series,
    )

    events = (
        raw_df.with_columns([
            pickup_datetime_expr(raw_df).alias("pickup_dt"),
            pl.col("pickup_longitude").cast(pl.Float64).alias("pickup_lon"),
            pl.col("pickup_latitude").cast(pl.Float64).alias("pickup_lat"),
        ])
        .filter(
            pl.col("pickup_dt").is_not_null()
            & pl.col("pickup_lon").is_not_null()
            & pl.col("pickup_lat").is_not_null()
            & pl.col("pickup_lon").is_between(spec.min_lon, spec.max_lon)
            & pl.col("pickup_lat").is_between(spec.min_lat, spec.max_lat)
        )
        .with_columns([
            (pl.col("pickup_lon") / spec.grid_size_deg).floor().cast(pl.Int64).alias("gx"),
            (pl.col("pickup_lat") / spec.grid_size_deg).floor().cast(pl.Int64).alias("gy"),
            pl.col("pickup_dt").dt.truncate(bucket_every).alias("time_bucket"),
        ])
        .with_columns(
            (
                pl.col("gx").cast(pl.String)
                + pl.lit("_")
                + pl.col("gy").cast(pl.String)
            ).alias("oper_part_no")
        )
        .group_by(["oper_part_no", "time_bucket"], maintain_order=True)
        .agg(pl.len().cast(pl.Float64).alias("demand_qty"))
    )

    if events.height == 0:
        raise ValueError("yellow_trip preprocessing produced no events after filtering.")

    # Global bucket index keeps `seq` comparable across grid cells.
    bucket_map = (
        events.select("time_bucket")
        .unique()
        .sort("time_bucket")
        .with_row_index("seq", offset=1)
        .with_columns(
            pl.col("time_bucket").dt.strftime("%Y%m%d%H%M%S").cast(pl.Int64).alias("demand_dt")
        )
    )

    events = (
        events.join(bucket_map, on="time_bucket", how="left")
        .select(["oper_part_no", "demand_dt", "seq", "demand_qty"])
        .sort(["oper_part_no", "seq"])
    )

    series_stats = (
        events.group_by("oper_part_no")
        .agg([
            pl.len().alias("active_buckets"),
            pl.col("demand_qty").sum().alias("total_demand"),
        ])
        .sort(["active_buckets", "total_demand"], descending=[True, True])
    )

    max_active_buckets = int(series_stats.select(pl.col("active_buckets").max()).item())
    effective_min_active = min(int(spec.min_active_buckets), max_active_buckets)
    if effective_min_active < int(spec.min_active_buckets):
        logger.warning(
            "Requested min_active_buckets=%s but max active buckets=%s. Falling back to %s.",
            spec.min_active_buckets,
            max_active_buckets,
            effective_min_active,
        )

    eligible_series = series_stats.filter(pl.col("active_buckets") >= effective_min_active)
    if spec.max_series is not None:
        eligible_series = eligible_series.head(int(spec.max_series))

    events = events.join(eligible_series.select("oper_part_no"), on="oper_part_no", how="inner")
    if events.height == 0:
        raise ValueError("No eligible yellow_trip series remained after active-bucket filtering.")

    return events.sort(["oper_part_no", "seq"])


def summarize_raw_events(raw_events: pl.DataFrame) -> dict[str, Any]:
    """
    Save compact dataset-shape metadata for later experiment interpretation.
    """
    lengths = (
        raw_events.group_by("oper_part_no")
        .agg([
            pl.len().alias("event_count"),
            pl.col("demand_qty").sum().alias("total_demand"),
        ])
    )
    return {
        "rows": int(raw_events.height),
        "series_count": int(lengths.height),
        "seq_min": int(raw_events.select(pl.col("seq").min()).item()),
        "seq_max": int(raw_events.select(pl.col("seq").max()).item()),
        "event_count_mean": float(lengths.select(pl.col("event_count").mean()).item()),
        "event_count_median": float(lengths.select(pl.col("event_count").median()).item()),
        "event_count_max": int(lengths.select(pl.col("event_count").max()).item()),
        "total_demand_sum": float(raw_events.select(pl.col("demand_qty").sum()).item()),
        "demand_qty_mean": float(raw_events.select(pl.col("demand_qty").mean()).item()),
        "demand_qty_p95": float(raw_events.select(pl.col("demand_qty").quantile(0.95)).item()),
        "demand_qty_max": float(raw_events.select(pl.col("demand_qty").max()).item()),
    }


def has_positive_events(raw_events: pl.DataFrame) -> bool:
    """
    Guard against stale or over-filtered caches.
    """
    if raw_events.height == 0 or "demand_qty" not in raw_events.columns:
        return False
    positive_count = raw_events.select((pl.col("demand_qty") > 0).sum()).item()
    return bool(positive_count and positive_count > 0)


def prepare_marked_resolution_dataset(
    *,
    spec: YellowResolutionSpec,
    runtime_cfg: ResolutionRuntimeConfig,
    logger,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Build/load raw and magnitude-marked yellow-trip tables for one resolution.
    """
    variant = resolution_label(spec)
    cache_root = ensure_dir(Path(runtime_cfg.base_dir) / "cache" / variant)
    base_label = sanitize_float_label(runtime_cfg.scale_base)
    raw_cache_path = cache_root / "raw_events.parquet"
    marked_cache_path = cache_root / f"marked_base_{base_label}.parquet"
    meta_json_path = cache_root / f"meta_base_{base_label}.json"
    raw_dist_path = cache_root / f"raw_dist_base_{base_label}.parquet"
    marked_dist_path = cache_root / f"marked_dist_base_{base_label}.parquet"
    raw_summary_path = cache_root / "raw_summary.json"

    if raw_cache_path.exists() and not runtime_cfg.force_rerun:
        raw_events = pl.read_parquet(raw_cache_path)
        if not has_positive_events(raw_events):
            logger.warning("Cached raw events at %s are invalid. Rebuilding.", raw_cache_path)
            raw_events = prepare_yellow_resolution_events(spec, logger=logger)
            raw_events.write_parquet(raw_cache_path)
    else:
        raw_events = prepare_yellow_resolution_events(spec, logger=logger)
        raw_events.write_parquet(raw_cache_path)

    if not has_positive_events(raw_events):
        raise ValueError(f"No positive events found for {variant}.")

    raw_summary = summarize_raw_events(raw_events)
    save_json(raw_summary, raw_summary_path)

    if marked_cache_path.exists() and meta_json_path.exists() and not runtime_cfg.force_rerun:
        marked_df = pl.read_parquet(marked_cache_path)
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return marked_df, meta, raw_summary

    logger.info("Building magnitude-marked dataset | variant=%s | base=%s", variant, runtime_cfg.scale_base)
    marked_df, meta = build_magnitude_marked_df(
        raw_events,
        scale_base=runtime_cfg.scale_base,
    )
    marked_df.write_parquet(marked_cache_path)

    raw_distribution = meta.pop("raw_distribution")
    marked_distribution = meta.pop("marked_distribution")
    raw_distribution.write_parquet(raw_dist_path)
    marked_distribution.write_parquet(marked_dist_path)

    meta.update({
        "dataset_name": variant,
        "dataset_kind": f"yellow_trip_{spec.resolution}",
        "resolution": spec.resolution,
        "grid_size_deg": float(spec.grid_size_deg),
        "min_active_buckets": int(spec.min_active_buckets),
        "max_series": None if spec.max_series is None else int(spec.max_series),
        "vendor": int(spec.vendor),
        "raw_rows": int(raw_events.height),
        "series_count": int(raw_events["oper_part_no"].n_unique()),
        "raw_distribution_path": str(raw_dist_path),
        "marked_distribution_path": str(marked_dist_path),
        "raw_summary_path": str(raw_summary_path),
        "cache_root": str(cache_root),
    })
    save_json(meta, meta_json_path)
    return marked_df, meta, raw_summary


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def build_long_cfg_for_resolution(
    *,
    runtime_cfg: ResolutionRuntimeConfig,
    resolution: str,
) -> LongEpochConfig:
    """
    Convert this runner's config into the shared long-run config.
    """
    lookback = lookback_for_resolution(runtime_cfg, resolution)
    profile_label = (
        f"yellow_resolution_{resolution}"
        f"_lb{lookback}"
        f"_maxlen{runtime_cfg.max_seq_len}"
    )
    return LongEpochConfig(
        base_dir=runtime_cfg.base_dir,
        device=runtime_cfg.device,
        lookback_weeks=lookback,
        max_seq_len=runtime_cfg.max_seq_len,
        batch_size=runtime_cfg.batch_size,
        lr=runtime_cfg.lr,
        val_ratio=runtime_cfg.val_ratio,
        lambda_value=runtime_cfg.lambda_value,
        lambda_dt=runtime_cfg.lambda_dt,
        grad_clip=runtime_cfg.grad_clip,
        epochs=runtime_cfg.epochs,
        seeds=runtime_cfg.seeds,
        titan_profile=profile_label,
        force_rerun=runtime_cfg.force_rerun,
        stop_on_error=runtime_cfg.stop_on_error,
        rmtpp_rnn_type=runtime_cfg.rmtpp_rnn_type,
        rmtpp_mark_emb_dim=runtime_cfg.rmtpp_mark_emb_dim,
        loss_mode=runtime_cfg.loss_mode,
        analysis_scale_base=runtime_cfg.analysis_scale_base,
        analysis_tail_order=runtime_cfg.analysis_tail_order,
        eval_selections=runtime_cfg.eval_selections,
    )


def run_resolution_benchmark(
    *,
    spec: YellowResolutionSpec,
    runtime_cfg: ResolutionRuntimeConfig,
    models: tuple[str, ...],
    titan_candidates: tuple[TitanCandidate, ...],
    logger,
) -> list[dict[str, Any]]:
    """
    Train all requested models for one temporal resolution.
    """
    marked_df, marked_meta, raw_summary = prepare_marked_resolution_dataset(
        spec=spec,
        runtime_cfg=runtime_cfg,
        logger=logger,
    )
    long_cfg = build_long_cfg_for_resolution(runtime_cfg=runtime_cfg, resolution=spec.resolution)
    rows: list[dict[str, Any]] = []
    leaderboard_dir = ensure_dir(Path(runtime_cfg.base_dir) / "leaderboard")
    path_prefix = leaderboard_dir / "resolution_runs"

    model_jobs: list[tuple[str, TitanCandidate]] = []
    if "rmtpp" in models:
        model_jobs.append(("rmtpp", make_rmtpp_proxy_candidate(runtime_cfg)))
    if "titantpp" in models:
        model_jobs.extend(("titantpp", candidate) for candidate in titan_candidates)

    total_runs = len(model_jobs) * len(runtime_cfg.seeds)
    completed = 0
    for model_name, candidate in model_jobs:
        for seed in runtime_cfg.seeds:
            completed += 1
            logger.info(
                "Resolution run %s/%s | resolution=%s | model=%s | candidate=%s | seed=%s",
                completed,
                total_runs,
                spec.resolution,
                model_name,
                candidate.name,
                seed,
            )
            run_cfg = LongRunConfig(
                dataset_name=str(marked_meta["dataset_name"]),
                dataset_kind=str(marked_meta["dataset_kind"]),
                model_name=model_name,
                seed=int(seed),
                epochs=int(runtime_cfg.epochs),
                scale_base=float(runtime_cfg.scale_base),
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
                    "Resolution run failed | resolution=%s model=%s candidate=%s seed=%s",
                    spec.resolution,
                    model_name,
                    candidate.name,
                    seed,
                )
                if runtime_cfg.stop_on_error:
                    raise

            # Add resolution-specific metadata outside the shared trainer so
            # generic long-run code can stay unchanged.
            row.update({
                "resolution": spec.resolution,
                "resolution_variant": resolution_label(spec),
                "grid_size_deg": float(spec.grid_size_deg),
                "min_active_buckets": int(spec.min_active_buckets),
                "max_series": None if spec.max_series is None else int(spec.max_series),
                "lookback_buckets": int(long_cfg.lookback_weeks),
                "raw_rows": int(raw_summary["rows"]),
                "raw_series_count": int(raw_summary["series_count"]),
                "raw_event_count_mean": float(raw_summary["event_count_mean"]),
                "raw_event_count_median": float(raw_summary["event_count_median"]),
                "raw_event_count_max": int(raw_summary["event_count_max"]),
            })
            rows.append(row)
            persist_rows(rows, path_prefix)

    return rows


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def load_resolution_histories(run_rows: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Expand per-run history files and keep resolution/candidate labels.
    """
    rows: list[dict[str, Any]] = []
    for run_row in run_rows:
        if run_row.get("status") != "success":
            continue
        history_path = Path(str(run_row["run_dir"])) / "metrics" / "history.json"
        if not history_path.exists():
            continue
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f).get("history", [])

        model_name = str(run_row["model_name"])
        candidate_name = str(run_row["titan_candidate_name"])
        run_label = model_name.upper() if model_name == "rmtpp" else f"TITANTPP:{candidate_name}"
        for epoch_row in history:
            rows.append({
                "dataset_name": run_row["dataset_name"],
                "resolution": run_row["resolution"],
                "resolution_variant": run_row["resolution_variant"],
                "model_name": model_name,
                "titan_candidate_name": candidate_name,
                "run_label": run_label,
                "seed": int(run_row["seed"]),
                "epoch": int(epoch_row["epoch"]),
                "train_loss": float(epoch_row.get("train_loss", float("nan"))),
                "score": float(epoch_row.get("score", float("nan"))),
                "val_nll": float(epoch_row.get("val_nll", float("nan"))),
                "val_nll_time": float(epoch_row.get("val_nll_time", float("nan"))),
                "val_nll_marker": float(epoch_row.get("val_nll_marker", float("nan"))),
                "val_value_loss": float(epoch_row.get("val_value_loss", float("nan"))),
                "mark_acc": float(epoch_row.get("mark_acc", float("nan"))),
                "dt_mae": float(epoch_row.get("dt_mae", float("nan"))),
                "qty_mae": float(epoch_row.get("qty_mae", float("nan"))),
                "value_mae": float(epoch_row.get("value_mae", float("nan"))),
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def load_resolution_scale_metrics(
    run_rows: list[dict[str, Any]],
    selections: Iterable[str],
) -> pl.DataFrame:
    """
    Combine scale-wise metric files while preserving resolution/candidate labels.
    """
    frames: list[pl.DataFrame] = []
    for run_row in run_rows:
        if run_row.get("status") != "success":
            continue
        metrics_dir = Path(str(run_row["run_dir"])) / "metrics"
        model_name = str(run_row["model_name"])
        candidate_name = str(run_row["titan_candidate_name"])
        run_label = model_name.upper() if model_name == "rmtpp" else f"TITANTPP:{candidate_name}"

        for selection in selections:
            parquet_path = metrics_dir / f"scale_wise_{selection}.parquet"
            csv_path = metrics_dir / f"scale_wise_{selection}.csv"
            if parquet_path.exists():
                df = pl.read_parquet(parquet_path)
            elif csv_path.exists():
                df = pl.read_csv(csv_path)
            else:
                continue
            frames.append(
                df.with_columns([
                    pl.lit(run_row["resolution"]).alias("resolution"),
                    pl.lit(run_row["resolution_variant"]).alias("resolution_variant"),
                    pl.lit(candidate_name).alias("titan_candidate_name"),
                    pl.lit(run_label).alias("run_label"),
                ])
            )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def aggregate_resolution_runs(run_rows: list[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Summarize run-level sweet-spot metrics across seeds.
    """
    if not run_rows:
        return pl.DataFrame(), pl.DataFrame()

    run_df = pl.DataFrame([{key: to_jsonable(value) for key, value in row.items()} for row in run_rows])
    success_df = run_df.filter(pl.col("status") == "success")
    if success_df.height == 0:
        return run_df, pl.DataFrame()

    summary_df = (
        success_df.group_by(["resolution", "model_name", "titan_candidate_name"])
        .agg([
            pl.first("dataset_name").alias("dataset_name"),
            pl.first("dataset_kind").alias("dataset_kind"),
            pl.first("resolution_variant").alias("resolution_variant"),
            pl.first("scale_base").alias("scale_base"),
            pl.first("lr").alias("lr"),
            pl.first("batch_size").alias("batch_size"),
            pl.first("lookback_buckets").alias("lookback_buckets"),
            pl.first("max_seq_len").alias("max_seq_len"),
            pl.first("grid_size_deg").alias("grid_size_deg"),
            pl.first("min_active_buckets").alias("min_active_buckets"),
            pl.first("max_series").alias("max_series"),
            pl.first("raw_rows").alias("raw_rows"),
            pl.first("raw_series_count").alias("raw_series_count"),
            pl.first("raw_event_count_mean").alias("raw_event_count_mean"),
            pl.first("raw_event_count_median").alias("raw_event_count_median"),
            pl.first("raw_event_count_max").alias("raw_event_count_max"),
            pl.len().alias("run_count"),
            pl.mean("best_val_nll").alias("mean_best_val_nll"),
            pl.std("best_val_nll").fill_null(0.0).alias("std_best_val_nll"),
            pl.mean("best_val_nll_epoch").alias("mean_best_val_nll_epoch"),
            pl.mean("best_val_nll_score").alias("mean_best_val_nll_score"),
            pl.mean("best_val_nll_qty_mae").alias("mean_best_val_nll_qty_mae"),
            pl.std("best_val_nll_qty_mae").fill_null(0.0).alias("std_best_val_nll_qty_mae"),
            pl.mean("best_val_nll_dt_mae").alias("mean_best_val_nll_dt_mae"),
            pl.mean("best_val_nll_mark_acc").alias("mean_best_val_nll_mark_acc"),
            pl.mean("best_val_nll_value_mae").alias("mean_best_val_nll_value_mae"),
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
        .sort(["resolution", "model_name", "mean_best_val_nll"])
    )
    return run_df, summary_df


def build_resolution_delta_table(summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compare every Titan candidate against the RMTPP baseline for each resolution.
    """
    if summary_df.height == 0:
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    for resolution in summary_df["resolution"].unique().to_list():
        resolution_df = summary_df.filter(pl.col("resolution") == resolution)
        rmtpp_rows = resolution_df.filter(pl.col("model_name") == "rmtpp").to_dicts()
        titan_rows = resolution_df.filter(pl.col("model_name") == "titantpp").to_dicts()
        if not rmtpp_rows or not titan_rows:
            continue
        rmtpp_row = rmtpp_rows[0]
        for titan_row in titan_rows:
            rows.append({
                "resolution": resolution,
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


def aggregate_resolution_scale_metrics(scale_df: pl.DataFrame) -> pl.DataFrame:
    """
    Average scale-wise quantity errors without mixing Titan candidates.
    """
    if scale_df.height == 0:
        return pl.DataFrame()

    # Start from the generic aggregation shape, then keep candidate labels by
    # aggregating with a richer key here.
    return (
        scale_df.group_by([
            "resolution",
            "model_name",
            "titan_candidate_name",
            "selection",
            "scale_order",
            "scale_label",
        ])
        .agg([
            pl.first("dataset_name").alias("dataset_name"),
            pl.first("run_label").alias("run_label"),
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
        .sort(["resolution", "selection", "scale_order", "model_name", "titan_candidate_name"])
    )


def save_resolution_learning_curve_plots(history_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Save convergence plots by resolution and candidate.
    """
    if history_df.height == 0:
        return

    ensure_dir(plots_dir)
    curve_metrics = [
        ("train_loss", "Train Loss"),
        ("val_nll", "Validation NLL"),
        ("val_nll_marker", "Marker NLL"),
        ("val_nll_time", "Time NLL"),
        ("mark_acc", "Mark Accuracy"),
        ("qty_mae", "Qty MAE"),
    ]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for resolution in history_df["resolution"].unique().to_list():
        resolution_df = history_df.filter(pl.col("resolution") == resolution)
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        axes_flat = axes.flatten()
        labels = resolution_df["run_label"].unique().to_list()
        color_map = {
            label: color_cycle[idx % len(color_cycle)]
            for idx, label in enumerate(labels)
        }

        for ax, (metric, title) in zip(axes_flat, curve_metrics):
            for label in labels:
                label_df = resolution_df.filter(pl.col("run_label") == label)
                agg_df = (
                    label_df.group_by("epoch")
                    .agg([
                        pl.mean(metric).alias("mean_metric"),
                        pl.std(metric).fill_null(0.0).alias("std_metric"),
                    ])
                    .sort("epoch")
                )
                if agg_df.height == 0:
                    continue
                epochs = np.asarray(agg_df["epoch"].to_list(), dtype=float)
                mean_values = np.asarray(agg_df["mean_metric"].to_list(), dtype=float)
                std_values = np.asarray(agg_df["std_metric"].to_list(), dtype=float)
                ax.plot(epochs, mean_values, label=label, color=color_map[label], linewidth=2)
                ax.fill_between(
                    epochs,
                    mean_values - std_values,
                    mean_values + std_values,
                    color=color_map[label],
                    alpha=0.15,
                )
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)

        fig.suptitle(f"yellow_trip {resolution}: resolution benchmark learning curves", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plots_dir / f"yellow_trip_{resolution}_learning_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_resolution_summary_plots(summary_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Save bar charts for best-validation-NLL, mark accuracy, and quantity MAE.
    """
    if summary_df.height == 0:
        return

    ensure_dir(plots_dir)
    metrics = [
        ("mean_best_val_nll", "Best Validation NLL", "lower"),
        ("mean_best_val_nll_mark_acc", "Best Mark Accuracy", "higher"),
        ("mean_best_val_nll_qty_mae", "Best Qty MAE", "lower"),
        ("mean_best_val_nll_dt_mae", "Best DT MAE", "lower"),
    ]

    for resolution in summary_df["resolution"].unique().to_list():
        resolution_df = summary_df.filter(pl.col("resolution") == resolution)
        labels = [
            row["model_name"].upper()
            if row["model_name"] == "rmtpp"
            else f"TITAN:{row['titan_candidate_name']}"
            for row in resolution_df.to_dicts()
        ]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for ax, (metric, title, direction) in zip(axes.flatten(), metrics):
            values = resolution_df[metric].to_list()
            ax.bar(x, values, color="#72B7B2", alpha=0.9)
            ax.set_title(f"{title} ({direction} is better)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
        fig.suptitle(f"yellow_trip {resolution}: RMTPP vs TitanTPP summary", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(plots_dir / f"yellow_trip_{resolution}_summary_grid.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_resolution_scale_plots(scale_summary_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Save scale-wise quantity MAE plots for each resolution/selection.
    """
    if scale_summary_df.height == 0:
        return

    ensure_dir(plots_dir)
    for resolution in scale_summary_df["resolution"].unique().to_list():
        for selection in scale_summary_df["selection"].unique().to_list():
            df = scale_summary_df.filter(
                (pl.col("resolution") == resolution)
                & (pl.col("selection") == selection)
            )
            if df.height == 0:
                continue
            labels = (
                df.select(["scale_order", "scale_label"])
                .unique()
                .sort("scale_order")["scale_label"]
                .to_list()
            )
            run_labels = df["run_label"].unique().to_list()
            x = np.arange(len(labels))
            width = min(0.8 / max(len(run_labels), 1), 0.28)

            fig, ax = plt.subplots(figsize=(14, 6))
            for idx, run_label in enumerate(run_labels):
                run_df = df.filter(pl.col("run_label") == run_label).sort("scale_order")
                values = run_df["mean_qty_mae"].to_list()
                offset = (idx - (len(run_labels) - 1) / 2) * width
                ax.bar(x + offset, values, width=width, label=run_label, alpha=0.9)
            ax.set_title(f"yellow_trip {resolution}: scale-wise Qty MAE ({selection})")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"yellow_trip_{resolution}_{selection}_scale_wise_qty_mae.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def save_paper_outputs(
    *,
    summary_df: pl.DataFrame,
    delta_df: pl.DataFrame,
    history_df: pl.DataFrame,
    scale_df: pl.DataFrame,
    scale_summary_df: pl.DataFrame,
    runtime_cfg: ResolutionRuntimeConfig,
    specs: list[YellowResolutionSpec],
    paper_dir: Path,
) -> None:
    """
    Persist paper/notion-friendly tables and a narrative report.
    """
    ensure_dir(paper_dir)
    if summary_df.width > 0:
        summary_df.write_csv(paper_dir / "paper_table_resolution_metrics.csv")
        summary_df.write_parquet(paper_dir / "paper_table_resolution_metrics.parquet")
    if delta_df.width > 0:
        delta_df.write_csv(paper_dir / "paper_table_resolution_deltas.csv")
        delta_df.write_parquet(paper_dir / "paper_table_resolution_deltas.parquet")
    if history_df.width > 0:
        history_df.write_csv(paper_dir / "paper_table_resolution_histories.csv")
    if scale_df.width > 0:
        scale_df.write_csv(paper_dir / "paper_table_scale_wise_metrics.csv")
    if scale_summary_df.width > 0:
        scale_summary_df.write_csv(paper_dir / "paper_table_scale_wise_summary.csv")
        scale_summary_df.write_parquet(paper_dir / "paper_table_scale_wise_summary.parquet")

    spec_lines = [
        f"- `{spec.resolution}`: grid={spec.grid_size_deg}, "
        f"min_active_buckets={spec.min_active_buckets}, "
        f"max_series={'all' if spec.max_series is None else spec.max_series}, "
        f"lookback={lookback_for_resolution(runtime_cfg, spec.resolution)} buckets"
        for spec in specs
    ]
    report_lines = [
        "# Yellow-trip Daily/Hourly RMTPP vs TitanTPP Report",
        "",
        "## Purpose",
        "",
        "The previous yellow-trip benchmark used weekly grid-cell pickup counts. "
        "That made each sequence very short, so this experiment rebuilds the same "
        "raw taxi data at daily/hourly resolution and tests whether TitanTPP "
        "benefits from longer and more heterogeneous event histories.",
        "",
        "## Event Definition",
        "",
        "- `entity`: pickup longitude/latitude grid cell",
        "- `time_bucket`: day or hour",
        "- `demand_qty`: pickup count in that grid/time bucket",
        "- `event`: positive-demand bucket only",
        "- `delta_t`: elapsed bucket count between two positive events in the same grid cell",
        "- `mark`: `floor(log_base(demand_qty))` with upper-tail merge from `build_magnitude_marked_df`",
        "- `scale_residual`: fractional transformed-scale residual used for quantity reconstruction",
        "",
        "## Runtime Configuration",
        "",
        f"- scale_base: `{runtime_cfg.scale_base}`",
        f"- epochs: `{runtime_cfg.epochs}`",
        f"- seeds: `{runtime_cfg.seeds}`",
        f"- lr: `{runtime_cfg.lr}`",
        f"- batch_size: `{runtime_cfg.batch_size}`",
        f"- max_seq_len: `{runtime_cfg.max_seq_len}`",
        f"- loss_mode: `{runtime_cfg.loss_mode}`",
        "",
        "## Resolution Specs",
        "",
        *spec_lines,
        "",
        "## Best Validation NLL Summary",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## TitanTPP - RMTPP Delta",
        "",
        "Negative `delta_best_val_nll`, `delta_best_val_nll_qty_mae`, and "
        "`delta_best_val_nll_dt_mae` favor TitanTPP. Positive "
        "`delta_best_val_nll_mark_acc` and `delta_best_val_nll_score` favor TitanTPP.",
        "",
        markdown_table_from_df(delta_df),
        "",
        "## Scale-wise Quantity Metrics",
        "",
        markdown_table_from_df(scale_summary_df),
        "",
        "## Files To Check",
        "",
        "- `leaderboard/resolution_runs.csv`: run-level rows and failures",
        "- `leaderboard/resolution_summary.csv`: seed-averaged model/candidate comparison",
        "- `leaderboard/resolution_histories.csv`: epoch-by-epoch curves",
        "- `leaderboard/scale_wise_summary.csv`: quantity error by true demand scale",
        "- `paper_outputs/plots/`: learning curves, summary grids, scale-wise quantity plots",
        "",
    ]
    (paper_dir / "resolution_ab_report.md").write_text("\n".join(report_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse resolution benchmark settings.
    """
    parser = argparse.ArgumentParser(
        description="Run yellow-trip daily/hourly RMTPP vs TitanTPP benchmark."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT / "search_artifacts" / "yellow_trip_resolution_ab_test"),
        help="Directory where artifacts will be written.",
    )
    parser.add_argument(
        "--parquet-path",
        default=str(PROJECT_ROOT / "sample_data" / "yellow_trip.parquet"),
        help="Path to raw yellow_trip parquet.",
    )
    parser.add_argument("--resolutions", default="daily,hourly", help="Comma-separated: daily,hourly")
    parser.add_argument("--models", default="rmtpp,titantpp", help="Comma-separated: rmtpp,titantpp")
    parser.add_argument(
        "--titan-candidates",
        default="mid_lmm,mid_deep_lmm",
        help="Comma-separated TitanCandidate names from titan_hparam_search.py.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scale-base", type=float, default=10.0)
    parser.add_argument("--grid-size-deg", type=float, default=0.02)
    parser.add_argument("--max-series", type=int, default=1000, help="0 means all eligible series.")
    parser.add_argument("--max-raw-rows", type=int, default=0, help="0 means read all raw rows.")
    parser.add_argument("--daily-min-active-buckets", type=int, default=20)
    parser.add_argument("--hourly-min-active-buckets", type=int, default=72)
    parser.add_argument("--daily-lookback-buckets", type=int, default=90)
    parser.add_argument("--hourly-lookback-buckets", type=int, default=168)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--rmtpp-rnn-type", default="gru", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--rmtpp-hidden-dim", type=int, default=128)
    parser.add_argument("--rmtpp-mark-emb-dim", type=int, default=32)
    parser.add_argument(
        "--loss-mode",
        default="residual_only",
        choices=["residual_only", "hybrid", "qty_only"],
        help="RMTPP baseline only supports residual_only in this shared runner.",
    )
    parser.add_argument("--analysis-scale-base", type=float, default=10.0)
    parser.add_argument("--analysis-tail-order", type=int, default=4)
    parser.add_argument(
        "--eval-selections",
        default="best_val_nll",
        help="Comma-separated checkpoint selections: best_val_nll,best_score,final.",
    )
    parser.add_argument("--vendor", type=int, default=0, help="0 means all vendors.")
    parser.add_argument("--min-lon", type=float, default=-75.0)
    parser.add_argument("--max-lon", type=float, default=-72.0)
    parser.add_argument("--min-lat", type=float, default=40.0)
    parser.add_argument("--max-lat", type=float, default=42.0)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Run preprocessing, training, aggregation, and paper-output export.
    """
    args = parse_args()
    resolutions = parse_csv(args.resolutions)
    models = parse_csv(args.models)
    candidate_names = parse_csv(args.titan_candidates)
    eval_selections = parse_csv(args.eval_selections)

    supported_resolutions = {"daily", "hourly"}
    unsupported_resolutions = sorted(set(resolutions) - supported_resolutions)
    if unsupported_resolutions:
        raise ValueError(f"Unsupported resolutions: {unsupported_resolutions}")

    supported_models = {"rmtpp", "titantpp"}
    unsupported_models = sorted(set(models) - supported_models)
    if unsupported_models:
        raise ValueError(f"Unsupported models: {unsupported_models}")

    supported_selections = {"best_val_nll", "best_score", "final"}
    unsupported_selections = sorted(set(eval_selections) - supported_selections)
    if unsupported_selections:
        raise ValueError(f"Unsupported eval selections: {unsupported_selections}")

    if "rmtpp" in models and args.loss_mode != "residual_only":
        raise ValueError(
            "RMTPP baseline is intentionally kept on residual_only in this runner. "
            "Use --models titantpp if you want to explore Titan-only hybrid/qty_only."
        )

    all_candidates = default_titan_candidates()
    titan_candidates = tuple(find_candidate_by_name(all_candidates, name) for name in candidate_names)

    runtime_cfg = ResolutionRuntimeConfig(
        base_dir=args.base_dir,
        device=args.device,
        scale_base=args.scale_base,
        epochs=args.epochs,
        seeds=parse_seeds(args.seeds),
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        max_seq_len=args.max_seq_len,
        daily_lookback_buckets=args.daily_lookback_buckets,
        hourly_lookback_buckets=args.hourly_lookback_buckets,
        rmtpp_rnn_type=args.rmtpp_rnn_type,
        rmtpp_hidden_dim=args.rmtpp_hidden_dim,
        rmtpp_mark_emb_dim=args.rmtpp_mark_emb_dim,
        loss_mode=args.loss_mode,
        analysis_scale_base=args.analysis_scale_base,
        analysis_tail_order=args.analysis_tail_order,
        eval_selections=eval_selections,
        force_rerun=args.force_rerun,
        stop_on_error=args.stop_on_error,
    )

    max_series = zero_to_none(args.max_series)
    max_raw_rows = zero_to_none(args.max_raw_rows)
    spec_by_resolution = {
        "daily": YellowResolutionSpec(
            resolution="daily",
            parquet_path=args.parquet_path,
            grid_size_deg=args.grid_size_deg,
            min_active_buckets=args.daily_min_active_buckets,
            max_series=max_series,
            vendor=args.vendor,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            max_raw_rows=max_raw_rows,
        ),
        "hourly": YellowResolutionSpec(
            resolution="hourly",
            parquet_path=args.parquet_path,
            grid_size_deg=args.grid_size_deg,
            min_active_buckets=args.hourly_min_active_buckets,
            max_series=max_series,
            vendor=args.vendor,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            max_raw_rows=max_raw_rows,
        ),
    }
    specs = [spec_by_resolution[resolution] for resolution in resolutions]

    base_dir = ensure_dir(Path(runtime_cfg.base_dir))
    leaderboard_dir = ensure_dir(base_dir / "leaderboard")
    paper_dir = ensure_dir(base_dir / "paper_outputs")
    plots_dir = ensure_dir(paper_dir / "plots")
    logger = build_logger(base_dir / "yellow_trip_resolution_ab_test.log", "yellow_trip_resolution_ab_test")

    save_json(
        {
            "runtime_config": runtime_cfg,
            "specs": specs,
            "models": models,
            "titan_candidates": [asdict(candidate) for candidate in titan_candidates],
        },
        base_dir / "resolution_ab_manifest.json",
    )
    logger.info("Resolution runtime config: %s", runtime_cfg)
    logger.info("Resolution specs: %s", specs)

    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows = run_resolution_benchmark(
            spec=spec,
            runtime_cfg=runtime_cfg,
            models=models,
            titan_candidates=titan_candidates,
            logger=logger,
        )
        all_rows.extend(rows)
        persist_rows(all_rows, leaderboard_dir / "resolution_runs")

    run_df, summary_df = aggregate_resolution_runs(all_rows)
    delta_df = build_resolution_delta_table(summary_df)
    history_df = load_resolution_histories(all_rows)
    scale_df = load_resolution_scale_metrics(all_rows, runtime_cfg.eval_selections)
    scale_summary_df = aggregate_resolution_scale_metrics(scale_df)

    # Keep the generic scale aggregation available too, useful for quick
    # RMTPP-vs-Titan views when only one Titan candidate is used.
    generic_scale_summary_df = aggregate_scale_metrics(scale_df)

    if run_df.width > 0:
        run_df.write_parquet(leaderboard_dir / "resolution_runs.parquet")
        run_df.write_csv(leaderboard_dir / "resolution_runs.csv")
    if summary_df.width > 0:
        summary_df.write_parquet(leaderboard_dir / "resolution_summary.parquet")
        summary_df.write_csv(leaderboard_dir / "resolution_summary.csv")
    if delta_df.width > 0:
        delta_df.write_parquet(leaderboard_dir / "resolution_deltas.parquet")
        delta_df.write_csv(leaderboard_dir / "resolution_deltas.csv")
    if history_df.width > 0:
        history_df.write_parquet(leaderboard_dir / "resolution_histories.parquet")
        history_df.write_csv(leaderboard_dir / "resolution_histories.csv")
    if scale_df.width > 0:
        scale_df.write_parquet(leaderboard_dir / "scale_wise_metrics.parquet")
        scale_df.write_csv(leaderboard_dir / "scale_wise_metrics.csv")
    if scale_summary_df.width > 0:
        scale_summary_df.write_parquet(leaderboard_dir / "scale_wise_summary.parquet")
        scale_summary_df.write_csv(leaderboard_dir / "scale_wise_summary.csv")
    if generic_scale_summary_df.width > 0:
        generic_scale_summary_df.write_parquet(leaderboard_dir / "scale_wise_summary_generic.parquet")
        generic_scale_summary_df.write_csv(leaderboard_dir / "scale_wise_summary_generic.csv")

    save_resolution_learning_curve_plots(history_df, plots_dir)
    save_resolution_summary_plots(summary_df, plots_dir)
    save_resolution_scale_plots(scale_summary_df, plots_dir)
    save_paper_outputs(
        summary_df=summary_df,
        delta_df=delta_df,
        history_df=history_df,
        scale_df=scale_df,
        scale_summary_df=scale_summary_df,
        runtime_cfg=runtime_cfg,
        specs=specs,
        paper_dir=paper_dir,
    )

    logger.info("Resolution benchmark complete. Summary rows:\n%s", summary_df)


if __name__ == "__main__":
    main()
