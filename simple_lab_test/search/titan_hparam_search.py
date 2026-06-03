"""
Automated TitanTPP hyperparameter search for the magnitude-factorized TPP setup.

This runner is designed to be a durable experiment script rather than a
notebook-only utility. The full flow is:

1. materialize dataset-specific raw event tables
2. build cached magnitude-marked datasets for several log bases
3. run a short coarse search over curated Titan presets
4. refine the best coarse-search combinations with more epochs and seeds
5. persist histories, checkpoints, leaderboards, and best-summary files

The module is intentionally verbose because this file is expected to be read and
reused as an experiment blueprint.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import polars as pl
import torch

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


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
from models.Titan import TitanConfig
from utils.magnitude_pipeline import build_magnitude_marked_df, train_magnitude_titantpp
from utils.training import TrainingConfig


SECS_PER_WEEK = 7 * 24 * 60 * 60


# ---------------------------------------------------------------------------
# Experiment configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """
    Dataset-specific knobs for event-table construction.

    `marked_target` uses the already episode-collapsed and magnitude-marked
    intermittent target table, while `yellow_trip` first needs to be aggregated
    into weekly pickup-count series.
    """
    name: str
    parquet_path: str
    kind: str
    max_series: Optional[int] = None
    yellow_grid_size_deg: float = 0.01
    yellow_min_active_weeks: int = 20
    yellow_vendor: int = 0
    yellow_min_lon: float = -75.0
    yellow_max_lon: float = -72.0
    yellow_min_lat: float = 40.0
    yellow_max_lat: float = 42.0


@dataclass(frozen=True)
class TitanCandidate:
    """
    Compact preset describing one Titan architecture choice in the search grid.
    """
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    contextual_mem_size: int
    persistent_mem_size: int
    use_lmm: bool
    mem_size: int
    mem_topk: int
    use_pos_emb: bool = True
    use_causal: bool = True


@dataclass(frozen=True)
class SearchConfig:
    """
    Search-level runtime settings shared by all runs in this script.
    """
    base_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_bases: tuple[float, ...] = (10.0, 4.0, 2.0)
    lookback_weeks: int = 52
    max_seq_len: int = 64
    val_ratio: float = 0.2
    batch_size: int = 128
    lr: float = 3e-4
    lambda_value: float = 1.0
    lambda_dt: float = 1.0
    grad_clip: float = 1.0
    stage1_epochs: int = 3
    stage2_epochs: int = 8
    stage1_seeds: tuple[int, ...] = (42,)
    stage2_seeds: tuple[int, ...] = (42, 52, 62)
    stage1_top_k: int = 4
    rmtpp_mark_emb_dim: int = 32
    value_head_activation: str = "sigmoid"
    force_rerun: bool = False
    stop_on_error: bool = False


@dataclass(frozen=True)
class RunConfig:
    """
    Immutable description of one concrete training run.
    """
    dataset_name: str
    dataset_kind: str
    stage: str
    scale_base: float
    candidate_name: str
    seed: int
    training_epochs: int
    titan_candidate: TitanCandidate


@dataclass(frozen=True)
class RunPaths:
    """
    Centralized filesystem layout for one concrete run.
    """
    run_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    manifest_dir: Path
    logs_dir: Path


class TeeIO(io.TextIOBase):
    """
    Mirror training stdout/stderr into both the console and a run log file.

    This keeps the epoch-level prints from `utils.training` while also storing
    them on disk for long hyperparameter searches.
    """

    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# ---------------------------------------------------------------------------
# Generic filesystem / serialization helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=True, indent=2)


def build_logger(log_path: Path, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8", errors="replace")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


@contextlib.contextmanager
def tee_training_output(log_path: Path):
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
        tee = TeeIO(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            yield


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_float_label(value: float) -> str:
    return str(value).replace(".", "p")


# ---------------------------------------------------------------------------
# Dataset-specific runtime defaults
# ---------------------------------------------------------------------------

MARKED_TARGET_KIND = "marked_target"
MARKED_TARGET_DEFAULT_SCALE_BASE = 2.0
MARKED_TARGET_LOOKBACK_WEEKS = 52
MARKED_TARGET_MAX_SEQ_LEN = 16
MARKED_TARGET_BATCH_SIZE = 64
MARKED_TARGET_TITAN_CANDIDATES = frozenset({"small_no_lmm", "small_lmm"})
MARKED_TARGET_REQUIRED_COLUMNS = frozenset(
    {
        "oper_part_no",
        "demand_dt",
        "seq",
        "demand_qty",
        "delta_t",
    }
)


def is_marked_target_kind(dataset_kind: str) -> bool:
    """
    Identify the episode-level intermittent target dataset.
    """
    return dataset_kind == MARKED_TARGET_KIND


def search_config_for_dataset(search_cfg: SearchConfig, dataset_kind: str) -> SearchConfig:
    """
    Apply dataset-specific runtime overrides without changing yellow-trip runs.

    The marked target table has only about 21k episode-level events, so the
    model and loader should stay intentionally small. Yellow-trip keeps whatever
    was passed through CLI/defaults.
    """
    if not is_marked_target_kind(dataset_kind):
        return search_cfg

    return replace(
        search_cfg,
        lookback_weeks=MARKED_TARGET_LOOKBACK_WEEKS,
        max_seq_len=MARKED_TARGET_MAX_SEQ_LEN,
        batch_size=MARKED_TARGET_BATCH_SIZE,
    )


def scale_bases_for_dataset(search_cfg: SearchConfig, spec: DatasetSpec) -> tuple[float, ...]:
    """
    Return the scale bases that make sense for a dataset.
    """
    if is_marked_target_kind(spec.kind):
        return (MARKED_TARGET_DEFAULT_SCALE_BASE,)
    return search_cfg.log_bases


def scale_base_allowed_for_dataset(spec: DatasetSpec, scale_base: float) -> bool:
    """
    Prevent a pre-marked target table from being treated as if it can be re-binned.
    """
    if not is_marked_target_kind(spec.kind):
        return True
    return abs(float(scale_base) - MARKED_TARGET_DEFAULT_SCALE_BASE) < 1e-9


def candidate_allowed_for_dataset(spec: DatasetSpec, candidate: TitanCandidate) -> bool:
    """
    Keep Titan small on the marked target data, while leaving yellow-trip intact.
    """
    if not is_marked_target_kind(spec.kind):
        return True
    return candidate.name in MARKED_TARGET_TITAN_CANDIDATES


# ---------------------------------------------------------------------------
# Search-space definitions
# ---------------------------------------------------------------------------

def default_dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="intermittent",
            parquet_path=str(PROJECT_ROOT / "sample_data" / "marked_target_df.parquet"),
            kind=MARKED_TARGET_KIND,
        ),
        DatasetSpec(
            name="yellow_trip",
            parquet_path=str(PROJECT_ROOT / "sample_data" / "yellow_trip.parquet"),
            kind="yellow_trip",
        ),
    ]


def default_titan_candidates() -> list[TitanCandidate]:
    """
    Curated preset search space for TitanTPP.

    Instead of a huge cartesian product, we start with a compact but diverse
    preset family. This keeps the search tractable while still varying the most
    important Titan dimensions and LMM usage.
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
        ),
    ]


# ---------------------------------------------------------------------------
# Dataset preparation and caching
# ---------------------------------------------------------------------------

def dataset_variant_key(spec: DatasetSpec) -> str:
    """
    Build a cache-key-friendly label so preprocessing variants do not collide.
    """
    if spec.kind == MARKED_TARGET_KIND:
        if spec.max_series is None:
            return "marked_parts_all"
        return f"marked_parts_{spec.max_series}"

    if spec.kind == "intermittent":
        if spec.max_series is None:
            return "parts_all"
        return f"parts_{spec.max_series}"

    grid_label = sanitize_float_label(spec.yellow_grid_size_deg)
    series_label = "all" if spec.max_series is None else str(spec.max_series)
    return (
        f"vendor_{spec.yellow_vendor}_grid_{grid_label}"
        f"_minweeks_{spec.yellow_min_active_weeks}"
        f"_entities_{series_label}"
    )


def filter_top_series(raw_df: pl.DataFrame, *, key_col: str, max_series: Optional[int]) -> pl.DataFrame:
    """
    Keep only the most active series when we want a faster subset experiment.
    """
    if max_series is None:
        return raw_df

    top_keys = (
        raw_df.filter(pl.col("demand_qty") > 0)
        .group_by(key_col)
        .agg([
            pl.len().alias("active_points"),
            pl.col("demand_qty").sum().alias("total_demand"),
        ])
        .sort(["active_points", "total_demand"], descending=[True, True])
        .head(max_series)
        .select(key_col)
    )
    return raw_df.join(top_keys, on=key_col, how="inner")


def prepare_intermittent_events(spec: DatasetSpec) -> pl.DataFrame:
    """
    Load the intermittent-demand event table with the schema expected downstream.
    """
    raw_df = pl.read_parquet(spec.parquet_path).select([
        "oper_part_no",
        "demand_dt",
        "seq",
        "demand_qty",
    ])
    return filter_top_series(raw_df, key_col="oper_part_no", max_series=spec.max_series)


def prepare_marked_target_events(
    spec: DatasetSpec,
    *,
    scale_base: float = MARKED_TARGET_DEFAULT_SCALE_BASE,
) -> pl.DataFrame:
    """
    Load the episode-level marked target table as-is.

    `marked_target_df.parquet` is expected to come from the notebook pipeline
    that collapses consecutive burst weeks into one episode. This loader keeps
    those episode-level rows and `delta_t` values, then normalizes
    `mark/scale_residual` from `demand_qty` for the configured scale base.
    It never rebuilds rows from raw weekly positives.
    """
    marked_df = pl.read_parquet(spec.parquet_path)
    missing = sorted(MARKED_TARGET_REQUIRED_COLUMNS - set(marked_df.columns))
    if missing:
        raise ValueError(
            "marked_target_df.parquet is missing required columns: "
            f"{missing}. Regenerate it from the episode-level intermittent "
            "pipeline with scale_base=2.0."
        )

    log_base = math.log(float(scale_base))
    safe_qty = pl.max_horizontal(
        pl.col("demand_qty").cast(pl.Float64),
        pl.lit(1.0),
    )
    marked_df = (
        marked_df.select([
            "oper_part_no",
            "demand_dt",
            "seq",
            "delta_t",
            "demand_qty",
        ])
        .with_columns((safe_qty.log() / log_base).alias("log_qty"))
        .with_columns([
            pl.col("log_qty").floor().cast(pl.Int32).alias("mark"),
            (pl.col("log_qty") - pl.col("log_qty").floor()).cast(pl.Float64).alias("scale_residual"),
            pl.col("log_qty").alias("z"),
        ])
    )
    marked_df = filter_top_series(marked_df, key_col="oper_part_no", max_series=spec.max_series)
    return marked_df.sort(["oper_part_no", "seq"])


def prepare_yellow_trip_events(spec: DatasetSpec) -> pl.DataFrame:
    """
    Convert raw taxi trips into weekly event-count sequences.

    We spatially bucket pickups into grid cells, count weekly demand per grid
    cell, and then map those counts into the common event-table schema used by
    RMTPP/TitanTPP training.
    """
    raw_df = pl.read_parquet(spec.parquet_path)

    if spec.yellow_vendor in (1, 2):
        raw_df = raw_df.filter(pl.col("VendorID") == int(spec.yellow_vendor))

    weekly = (
        raw_df.with_columns([
            pl.col("tpep_pickup_datetime")
            .str.strptime(pl.Datetime, strict=False)
            .alias("pickup_dt"),
            pl.col("pickup_longitude").cast(pl.Float64).alias("pickup_lon"),
            pl.col("pickup_latitude").cast(pl.Float64).alias("pickup_lat"),
        ])
        .filter(
            pl.col("pickup_dt").is_not_null()
            & pl.col("pickup_lon").is_not_null()
            & pl.col("pickup_lat").is_not_null()
            & pl.col("pickup_lon").is_between(spec.yellow_min_lon, spec.yellow_max_lon)
            & pl.col("pickup_lat").is_between(spec.yellow_min_lat, spec.yellow_max_lat)
        )
        .with_columns([
            (pl.col("pickup_lon") / spec.yellow_grid_size_deg).floor().cast(pl.Int64).alias("gx"),
            (pl.col("pickup_lat") / spec.yellow_grid_size_deg).floor().cast(pl.Int64).alias("gy"),
            pl.col("pickup_dt").dt.truncate("1w").alias("week_start"),
        ])
        .with_columns(
            (pl.col("gx").cast(pl.String) + pl.lit("_") + pl.col("gy").cast(pl.String)).alias("oper_part_no")
        )
        .group_by(["oper_part_no", "week_start"], maintain_order=True)
        .agg(pl.len().cast(pl.Float64).alias("demand_qty"))
    )

    week_map = (
        weekly.select("week_start")
        .unique()
        .sort("week_start")
        .with_row_index("seq", offset=1)
        .with_columns(
            pl.col("week_start").dt.strftime("%Y%m%d").cast(pl.Int64).alias("demand_dt")
        )
    )

    weekly = (
        weekly.join(week_map, on="week_start", how="left")
        .select(["oper_part_no", "demand_dt", "seq", "demand_qty"])
        .sort(["oper_part_no", "seq"])
    )

    series_stats = (
        weekly.group_by("oper_part_no")
        .agg([
            pl.len().alias("active_weeks"),
            pl.col("demand_qty").sum().alias("total_demand"),
        ])
        .sort(["active_weeks", "total_demand"], descending=[True, True])
    )

    if series_stats.height == 0:
        raise ValueError(
            "yellow_trip preprocessing produced no valid weekly entities. "
            "Please check the coordinate/time filters."
        )

    max_active_weeks = int(series_stats.select(pl.col("active_weeks").max()).item())
    requested_min_active_weeks = int(spec.yellow_min_active_weeks)
    effective_min_active_weeks = min(requested_min_active_weeks, max_active_weeks)

    if requested_min_active_weeks > max_active_weeks:
        print(
            "[yellow_trip][WARN] "
            f"requested yellow_min_active_weeks={requested_min_active_weeks} "
            f"but max active weeks in this parquet is only {max_active_weeks}. "
            f"Falling back to effective_min_active_weeks={effective_min_active_weeks}."
        )

    eligible_series = (
        series_stats
        .filter(pl.col("active_weeks") >= effective_min_active_weeks)
    )

    if spec.max_series is not None:
        eligible_series = eligible_series.head(spec.max_series)

    weekly = weekly.join(eligible_series.select("oper_part_no"), on="oper_part_no", how="inner")
    return weekly.sort(["oper_part_no", "seq"])


def prepare_raw_events(spec: DatasetSpec) -> pl.DataFrame:
    """
    Route each dataset spec to the appropriate raw-event builder.
    """
    if spec.kind == MARKED_TARGET_KIND:
        return prepare_marked_target_events(spec)
    if spec.kind == "intermittent":
        return prepare_intermittent_events(spec)
    if spec.kind == "yellow_trip":
        return prepare_yellow_trip_events(spec)
    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def has_positive_events(raw_events: pl.DataFrame) -> bool:
    """
    Validate cached raw-event tables before reusing them.

    This protects the search flow from stale caches created by an earlier
    preprocessing bug or by stricter filtering thresholds.
    """
    if raw_events.height == 0:
        return False
    if "demand_qty" not in raw_events.columns:
        return False
    positive_count = raw_events.select((pl.col("demand_qty") > 0).sum()).item()
    return bool(positive_count and positive_count > 0)


def build_premarked_target_meta(
    *,
    marked_df: pl.DataFrame,
    spec: DatasetSpec,
    scale_base: float,
    raw_dist_path: Path,
    marked_dist_path: Path,
) -> dict[str, Any]:
    """
    Build the same metadata shape used by freshly marked datasets.
    """
    if marked_df.height == 0:
        raise ValueError("marked_target_df.parquet is empty after dataset filtering.")

    max_order = int(marked_df.select(pl.col("mark").max()).item())
    min_order = int(marked_df.select(pl.col("mark").min()).item())
    distribution = (
        marked_df.group_by("mark")
        .agg(pl.len().alias("count"))
        .sort("mark")
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("ratio"))
    )
    distribution.write_parquet(marked_dist_path)
    distribution.write_parquet(raw_dist_path)

    return {
        "scale_base": float(scale_base),
        "min_order": min_order,
        "max_order": max_order,
        "num_marks": max_order + 2,
        "dataset_name": spec.name,
        "dataset_kind": spec.kind,
        "source_path": spec.parquet_path,
        "raw_rows": int(marked_df.height),
        "series_count": int(marked_df["oper_part_no"].n_unique()),
        "raw_distribution_path": str(raw_dist_path),
        "marked_distribution_path": str(marked_dist_path),
        "premarked": True,
    }


def prepare_marked_dataset(
    spec: DatasetSpec,
    scale_base: float,
    search_cfg: SearchConfig,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, dict[str, Any], Path]:
    """
    Build or load the cached magnitude-marked dataset for one dataset/base pair.

    This function also persists distribution snapshots so we can later inspect
    how each log base reshaped the class balance.
    """
    cache_root = ensure_dir(Path(search_cfg.base_dir) / "cache" / spec.name / dataset_variant_key(spec))
    raw_cache_path = cache_root / "raw_events.parquet"
    marked_cache_path = cache_root / f"marked_base_{sanitize_float_label(scale_base)}.parquet"
    meta_json_path = cache_root / f"meta_base_{sanitize_float_label(scale_base)}.json"
    raw_dist_path = cache_root / f"raw_dist_base_{sanitize_float_label(scale_base)}.parquet"
    marked_dist_path = cache_root / f"marked_dist_base_{sanitize_float_label(scale_base)}.parquet"

    if is_marked_target_kind(spec.kind):
        if not scale_base_allowed_for_dataset(spec, scale_base):
            raise ValueError(
                f"marked_target_df.parquet is pre-marked for scale_base="
                f"{MARKED_TARGET_DEFAULT_SCALE_BASE}; got scale_base={scale_base}."
            )

        if marked_cache_path.exists() and meta_json_path.exists() and not search_cfg.force_rerun:
            marked_df = pl.read_parquet(marked_cache_path)
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return marked_df, meta, cache_root

        logger.info("Loading pre-marked target dataset=%s from %s", spec.name, spec.parquet_path)
        marked_df = prepare_marked_target_events(spec, scale_base=scale_base)
        marked_df.write_parquet(raw_cache_path)
        marked_df.write_parquet(marked_cache_path)
        meta = build_premarked_target_meta(
            marked_df=marked_df,
            spec=spec,
            scale_base=scale_base,
            raw_dist_path=raw_dist_path,
            marked_dist_path=marked_dist_path,
        )
        save_json(meta, meta_json_path)
        return marked_df, meta, cache_root

    if raw_cache_path.exists() and not search_cfg.force_rerun:
        raw_events = pl.read_parquet(raw_cache_path)
        if not has_positive_events(raw_events):
            logger.warning(
                "Cached raw events are empty or have no positive demand for dataset=%s. "
                "Rebuilding raw cache at %s",
                spec.name,
                raw_cache_path,
            )
            raw_events = prepare_raw_events(spec)
            raw_events.write_parquet(raw_cache_path)
    else:
        logger.info("Preparing raw events for dataset=%s", spec.name)
        raw_events = prepare_raw_events(spec)
        raw_events.write_parquet(raw_cache_path)

    if not has_positive_events(raw_events):
        raise ValueError(
            f"Raw events for dataset={spec.name} are empty after preprocessing. "
            "Please check the dataset-specific filters or cached preprocessing outputs."
        )

    if marked_cache_path.exists() and meta_json_path.exists() and not search_cfg.force_rerun:
        marked_df = pl.read_parquet(marked_cache_path)
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return marked_df, meta, cache_root

    logger.info("Building magnitude-marked dataset for dataset=%s, base=%s", spec.name, scale_base)
    marked_df, meta = build_magnitude_marked_df(
        raw_events,
        scale_base=scale_base,
    )
    marked_df.write_parquet(marked_cache_path)
    meta["dataset_name"] = spec.name
    meta["dataset_kind"] = spec.kind
    meta["raw_rows"] = int(raw_events.height)
    meta["series_count"] = int(raw_events["oper_part_no"].n_unique())

    raw_distribution = meta.pop("raw_distribution")
    marked_distribution = meta.pop("marked_distribution")
    raw_distribution.write_parquet(raw_dist_path)
    marked_distribution.write_parquet(marked_dist_path)
    meta["raw_distribution_path"] = str(raw_dist_path)
    meta["marked_distribution_path"] = str(marked_dist_path)

    save_json(meta, meta_json_path)
    return marked_df, meta, cache_root


def build_run_paths(search_cfg: SearchConfig, run_cfg: RunConfig) -> RunPaths:
    """
    Materialize the on-disk layout for a single training run.
    """
    base_label = sanitize_float_label(run_cfg.scale_base)
    run_dir = (
        Path(search_cfg.base_dir)
        / "runs"
        / run_cfg.stage
        / run_cfg.dataset_name
        / f"base_{base_label}"
        / run_cfg.candidate_name
        / f"seed_{run_cfg.seed}"
    )
    return RunPaths(
        run_dir=ensure_dir(run_dir),
        checkpoint_dir=ensure_dir(run_dir / "checkpoints"),
        metrics_dir=ensure_dir(run_dir / "metrics"),
        manifest_dir=ensure_dir(run_dir / "manifest"),
        logs_dir=ensure_dir(run_dir / "logs"),
    )


def build_training_config(search_cfg: SearchConfig, *, epochs: int) -> TrainingConfig:
    """
    Translate search-level knobs into the shared trainer config object.
    """
    return TrainingConfig(
        lookback=search_cfg.lookback_weeks,
        max_seq_len=search_cfg.max_seq_len,
        batch_size=search_cfg.batch_size,
        lr=search_cfg.lr,
        epochs=epochs,
        val_ratio=search_cfg.val_ratio,
        device=search_cfg.device,
        lambda_value=search_cfg.lambda_value,
        lambda_dt=search_cfg.lambda_dt,
        grad_clip=search_cfg.grad_clip,
    )


def build_rmtpp_config(search_cfg: SearchConfig, *, num_marks: int, scale_base: float) -> RMTPPConfig:
    """
    RMTPPConfig is still needed because TitanTPP reuses the same TPP heads.
    """
    return RMTPPConfig(
        num_marks=num_marks,
        mark_emb_dim=search_cfg.rmtpp_mark_emb_dim,
        dropout=0.1,
        scale_base=scale_base,
        use_value_head=True,
        value_head_activation=search_cfg.value_head_activation,
    )


def build_titan_config(search_cfg: SearchConfig, candidate: TitanCandidate) -> TitanConfig:
    """
    Expand a compact Titan candidate preset into the full model config.
    """
    return TitanConfig(
        lookback=search_cfg.lookback_weeks,
        horizon=27,
        d_model=candidate.d_model,
        n_layers=candidate.n_layers,
        n_heads=candidate.n_heads,
        d_ff=candidate.d_ff,
        dropout=candidate.dropout,
        contextual_mem_size=candidate.contextual_mem_size,
        persistent_mem_size=candidate.persistent_mem_size,
        use_pos_emb=candidate.use_pos_emb,
        max_len=search_cfg.max_seq_len,
        use_lmm=candidate.use_lmm,
        mem_size=candidate.mem_size,
        mem_topk=candidate.mem_topk,
        use_causal=candidate.use_causal,
    )


def summarize_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Reduce full epoch histories into leaderboard-friendly best/final metrics.
    """
    if not history:
        raise ValueError("Training history is empty.")

    best_row = max(
        history,
        key=lambda row: (float(row["score"]), -float(row["val_nll"])),
    )
    min_nll_row = min(history, key=lambda row: float(row["val_nll"]))
    return {
        "best_epoch": int(best_row["epoch"]),
        "best_score": float(best_row["score"]),
        "best_mark_acc": float(best_row["mark_acc"]),
        "best_dt_mae": float(best_row["dt_mae"]),
        "best_dt_rmse": float(best_row["dt_rmse"]),
        "best_value_mae": float(best_row["value_mae"]),
        "best_qty_mae": float(best_row["qty_mae"]),
        "best_val_nll": float(best_row["val_nll"]),
        "best_val_nll_time": float(best_row["val_nll_time"]),
        "best_val_nll_marker": float(best_row["val_nll_marker"]),
        "best_val_value_loss": float(best_row["val_value_loss"]),
        "min_val_nll_epoch": int(min_nll_row["epoch"]),
        "min_val_nll": float(min_nll_row["val_nll"]),
        "final_epoch": int(history[-1]["epoch"]),
        "final_train_loss": float(history[-1]["train_loss"]),
    }


def flatten_candidate(candidate: TitanCandidate) -> dict[str, Any]:
    """
    Persist candidate architecture fields directly alongside metrics.
    """
    flat = asdict(candidate)
    flat["candidate_name"] = candidate.name
    return flat


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def run_single_experiment(
    *,
    search_cfg: SearchConfig,
    run_cfg: RunConfig,
    marked_df: pl.DataFrame,
    marked_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one TitanTPP run, persist artifacts, and return a summary row.
    """
    run_paths = build_run_paths(search_cfg, run_cfg)
    summary_path = run_paths.metrics_dir / "summary.json"
    history_json_path = run_paths.metrics_dir / "history.json"
    history_parquet_path = run_paths.metrics_dir / "history.parquet"
    checkpoint_path = run_paths.checkpoint_dir / "best_model.pt"
    log_path = run_paths.logs_dir / "train.log"
    manifest_path = run_paths.manifest_dir / "run_config.json"

    if (
        not search_cfg.force_rerun
        and summary_path.exists()
        and checkpoint_path.exists()
        and history_json_path.exists()
    ):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    set_global_seed(run_cfg.seed)
    dataset_search_cfg = search_config_for_dataset(search_cfg, run_cfg.dataset_kind)
    training_cfg = build_training_config(dataset_search_cfg, epochs=run_cfg.training_epochs)
    rmtpp_cfg = build_rmtpp_config(
        dataset_search_cfg,
        num_marks=int(marked_meta["num_marks"]),
        scale_base=run_cfg.scale_base,
    )
    titan_cfg = build_titan_config(dataset_search_cfg, run_cfg.titan_candidate)

    save_json(
        {
            "run_config": run_cfg,
            "effective_search_config": dataset_search_cfg,
            "training_config": training_cfg,
            "rmtpp_config": rmtpp_cfg,
            "titan_config": titan_cfg,
            "marked_meta": marked_meta,
        },
        manifest_path,
    )

    with tee_training_output(log_path):
        model, info = train_magnitude_titantpp(
            marked_df,
            training_config=training_cfg,
            rmtpp_config=rmtpp_cfg,
            titan_config=titan_cfg,
        )

    history = info["history"]
    history_df = pl.DataFrame(history)
    history_df.write_parquet(history_parquet_path)
    save_json({"history": history}, history_json_path)

    summary = {
        "status": "success",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "stage": run_cfg.stage,
        "scale_base": float(run_cfg.scale_base),
        "candidate_name": run_cfg.candidate_name,
        "seed": int(run_cfg.seed),
        "training_epochs": int(run_cfg.training_epochs),
        "batch_size": int(training_cfg.batch_size),
        "lookback_weeks": int(training_cfg.lookback),
        "max_seq_len": int(training_cfg.max_seq_len),
        "run_dir": str(run_paths.run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_marks": int(marked_meta["num_marks"]),
        "max_order": int(marked_meta["max_order"]),
        "series_count": int(marked_meta["series_count"]),
        **flatten_candidate(run_cfg.titan_candidate),
        **summarize_history(history),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
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


def persist_rows(rows: list[dict[str, Any]], path_prefix: Path) -> None:
    """
    Continuously checkpoint tabular run outputs so long searches are resumable.
    """
    if not rows:
        return

    serializable_rows = [{k: to_jsonable(v) for k, v in row.items()} for row in rows]
    df = pl.DataFrame(serializable_rows)
    ensure_dir(path_prefix.parent)
    df.write_parquet(path_prefix.with_suffix(".parquet"))
    df.write_csv(path_prefix.with_suffix(".csv"))


def build_run_row_from_error(run_cfg: RunConfig, candidate: TitanCandidate, exc: Exception) -> dict[str, Any]:
    """
    Convert a failed run into a serializable row so the search can continue.
    """
    return {
        "status": "failed",
        "dataset_name": run_cfg.dataset_name,
        "dataset_kind": run_cfg.dataset_kind,
        "stage": run_cfg.stage,
        "scale_base": float(run_cfg.scale_base),
        "candidate_name": run_cfg.candidate_name,
        "seed": int(run_cfg.seed),
        "training_epochs": int(run_cfg.training_epochs),
        "error": repr(exc),
        **flatten_candidate(candidate),
    }


def build_combo_leaderboard(run_rows: list[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Aggregate run rows into dataset-level and overall combo leaderboards.

    Dataset ranking blends score rank and NLL rank so that we do not pick a
    configuration that only looks good under one metric.
    """
    if not run_rows:
        return pl.DataFrame(), pl.DataFrame()

    run_df = pl.DataFrame([{k: to_jsonable(v) for k, v in row.items()} for row in run_rows])
    success_df = run_df.filter(pl.col("status") == "success")
    if success_df.height == 0:
        return success_df, pl.DataFrame()

    dataset_combo_df = (
        success_df.group_by(["stage", "dataset_name", "scale_base", "candidate_name"])
        .agg([
            pl.mean("best_score").alias("mean_best_score"),
            pl.std("best_score").fill_null(0.0).alias("std_best_score"),
            pl.mean("best_val_nll").alias("mean_best_val_nll"),
            pl.mean("best_qty_mae").alias("mean_best_qty_mae"),
            pl.mean("best_dt_mae").alias("mean_best_dt_mae"),
            pl.mean("best_mark_acc").alias("mean_best_mark_acc"),
            pl.len().alias("run_count"),
            pl.first("d_model").alias("d_model"),
            pl.first("n_layers").alias("n_layers"),
            pl.first("n_heads").alias("n_heads"),
            pl.first("d_ff").alias("d_ff"),
            pl.first("dropout").alias("dropout"),
            pl.first("contextual_mem_size").alias("contextual_mem_size"),
            pl.first("persistent_mem_size").alias("persistent_mem_size"),
            pl.first("use_lmm").alias("use_lmm"),
            pl.first("mem_size").alias("mem_size"),
            pl.first("mem_topk").alias("mem_topk"),
        ])
        .with_columns([
            pl.col("mean_best_score").rank(method="dense", descending=True).over("dataset_name").alias("score_rank"),
            pl.col("mean_best_val_nll").rank(method="dense", descending=False).over("dataset_name").alias("nll_rank"),
        ])
        .with_columns(
            (0.7 * pl.col("score_rank") + 0.3 * pl.col("nll_rank")).alias("dataset_rank_score")
        )
        .sort(["dataset_name", "dataset_rank_score", "mean_best_val_nll"])
    )

    combo_df = (
        dataset_combo_df.group_by(["stage", "scale_base", "candidate_name"])
        .agg([
            pl.mean("dataset_rank_score").alias("mean_dataset_rank"),
            pl.mean("mean_best_score").alias("mean_best_score"),
            pl.mean("mean_best_val_nll").alias("mean_best_val_nll"),
            pl.mean("mean_best_qty_mae").alias("mean_best_qty_mae"),
            pl.mean("mean_best_dt_mae").alias("mean_best_dt_mae"),
            pl.mean("mean_best_mark_acc").alias("mean_best_mark_acc"),
            pl.len().alias("dataset_count"),
            pl.first("d_model").alias("d_model"),
            pl.first("n_layers").alias("n_layers"),
            pl.first("n_heads").alias("n_heads"),
            pl.first("d_ff").alias("d_ff"),
            pl.first("dropout").alias("dropout"),
            pl.first("contextual_mem_size").alias("contextual_mem_size"),
            pl.first("persistent_mem_size").alias("persistent_mem_size"),
            pl.first("use_lmm").alias("use_lmm"),
            pl.first("mem_size").alias("mem_size"),
            pl.first("mem_topk").alias("mem_topk"),
        ])
        .sort(["mean_dataset_rank", "mean_best_val_nll"], descending=[False, False])
    )

    return dataset_combo_df, combo_df


def select_top_stage1_combos(combo_df: pl.DataFrame, top_k: int) -> list[tuple[float, str]]:
    """
    Extract the stage-1 winners that should be re-evaluated in stage 2.
    """
    if combo_df.height == 0:
        return []
    top_rows = combo_df.head(top_k).select(["scale_base", "candidate_name"]).to_dicts()
    return [(float(row["scale_base"]), str(row["candidate_name"])) for row in top_rows]


def find_candidate_by_name(candidates: Iterable[TitanCandidate], name: str) -> TitanCandidate:
    """
    Recover the candidate object after only saving its leaderboard label.
    """
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise KeyError(f"Candidate not found: {name}")


def write_final_summary(
    *,
    search_cfg: SearchConfig,
    dataset_combo_df: pl.DataFrame,
    combo_df: pl.DataFrame,
) -> None:
    """
    Persist the concise "best by dataset" and "best overall" JSON summaries.
    """
    leaderboard_dir = ensure_dir(Path(search_cfg.base_dir) / "leaderboard")
    if dataset_combo_df.height > 0:
        best_by_dataset = (
            dataset_combo_df.sort(["dataset_name", "dataset_rank_score", "mean_best_val_nll"])
            .group_by("dataset_name", maintain_order=True)
            .first()
        )
        save_json(
            {"best_by_dataset": best_by_dataset.to_dicts()},
            leaderboard_dir / "best_by_dataset.json",
        )

    if combo_df.height > 0:
        save_json(
            {"best_overall": combo_df.head(1).to_dicts()},
            leaderboard_dir / "best_overall.json",
        )


def run_stage(
    *,
    stage_name: str,
    search_cfg: SearchConfig,
    dataset_specs: list[DatasetSpec],
    candidate_pairs: list[tuple[float, TitanCandidate]],
    seeds: tuple[int, ...],
    epochs: int,
    marked_cache: dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Run one full stage (coarse or refinement) across datasets, bases, and seeds.
    """
    rows: list[dict[str, Any]] = []
    total_runs = sum(
        1
        for dataset_spec in dataset_specs
        for scale_base, candidate in candidate_pairs
        if scale_base_allowed_for_dataset(dataset_spec, scale_base)
        and candidate_allowed_for_dataset(dataset_spec, candidate)
        for _seed in seeds
    )
    completed = 0

    leaderboard_dir = ensure_dir(Path(search_cfg.base_dir) / "leaderboard")
    path_prefix = leaderboard_dir / f"{stage_name}_runs"

    for dataset_spec in dataset_specs:
        for scale_base, candidate in candidate_pairs:
            if not scale_base_allowed_for_dataset(dataset_spec, scale_base):
                continue
            if not candidate_allowed_for_dataset(dataset_spec, candidate):
                continue
            marked_df, marked_meta = marked_cache[(dataset_spec.name, scale_base)]
            for seed in seeds:
                completed += 1
                logger.info(
                    "[%s] Run %s/%s | dataset=%s | base=%s | candidate=%s | seed=%s",
                    stage_name,
                    completed,
                    total_runs,
                    dataset_spec.name,
                    scale_base,
                    candidate.name,
                    seed,
                )
                run_cfg = RunConfig(
                    dataset_name=dataset_spec.name,
                    dataset_kind=dataset_spec.kind,
                    stage=stage_name,
                    scale_base=scale_base,
                    candidate_name=candidate.name,
                    seed=seed,
                    training_epochs=epochs,
                    titan_candidate=candidate,
                )
                try:
                    row = run_single_experiment(
                        search_cfg=search_cfg,
                        run_cfg=run_cfg,
                        marked_df=marked_df,
                        marked_meta=marked_meta,
                    )
                except Exception as exc:
                    row = build_run_row_from_error(run_cfg, candidate, exc)
                    logger.exception(
                        "Run failed | stage=%s dataset=%s base=%s candidate=%s seed=%s",
                        stage_name,
                        dataset_spec.name,
                        scale_base,
                        candidate.name,
                        seed,
                    )
                    if search_cfg.stop_on_error:
                        raise
                rows.append(row)
                persist_rows(rows, path_prefix)

    return rows


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for search/runtime control.
    """
    parser = argparse.ArgumentParser(
        description="Automatic TitanTPP hyperparameter search over scale bases and TitanConfig presets."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT / "search_artifacts" / "titan_hparam_search"),
        help="Directory where caches, checkpoints, and leaderboards will be saved.",
    )
    parser.add_argument(
        "--datasets",
        default="intermittent,yellow_trip",
        help=(
            "Comma-separated dataset names to run. Available: intermittent,yellow_trip. "
            "The intermittent label loads sample_data/marked_target_df.parquet."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--stage1-top-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--intermittent-max-series", type=int, default=None)
    parser.add_argument("--yellow-max-series", type=int, default=None)
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Orchestrate the full search workflow from cache prep to final summaries.
    """
    args = parse_args()
    selected_dataset_names = {name.strip() for name in args.datasets.split(",") if name.strip()}

    dataset_specs = []
    for spec in default_dataset_specs():
        if spec.name not in selected_dataset_names:
            continue
        if spec.name == "intermittent":
            spec = DatasetSpec(**{**asdict(spec), "max_series": args.intermittent_max_series})
        elif spec.name == "yellow_trip":
            spec = DatasetSpec(**{**asdict(spec), "max_series": args.yellow_max_series})
        dataset_specs.append(spec)

    if not dataset_specs:
        raise ValueError("No datasets selected.")

    search_cfg = SearchConfig(
        base_dir=args.base_dir,
        device=args.device,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_top_k=args.stage1_top_k,
        force_rerun=args.force_rerun,
        stop_on_error=args.stop_on_error,
    )

    base_dir = ensure_dir(Path(search_cfg.base_dir))
    leaderboard_dir = ensure_dir(base_dir / "leaderboard")
    logger = build_logger(base_dir / "search.log", "titan_hparam_search")

    candidates = default_titan_candidates()
    save_json(
        {
            "search_config": search_cfg,
            "dataset_effective_search_configs": {
                spec.name: search_config_for_dataset(search_cfg, spec.kind)
                for spec in dataset_specs
            },
            "dataset_specs": dataset_specs,
            "candidates": candidates,
        },
        base_dir / "search_manifest.json",
    )

    logger.info("Preparing marked dataset caches for %s datasets", len(dataset_specs))
    marked_cache: dict[tuple[str, float], tuple[pl.DataFrame, dict[str, Any]]] = {}
    for spec in dataset_specs:
        for scale_base in scale_bases_for_dataset(search_cfg, spec):
            marked_df, marked_meta, cache_root = prepare_marked_dataset(
                spec=spec,
                scale_base=scale_base,
                search_cfg=search_cfg,
                logger=logger,
            )
            marked_cache[(spec.name, scale_base)] = (marked_df, marked_meta)
            logger.info(
                "Cached dataset=%s base=%s | rows=%s | series=%s | num_marks=%s | max_order=%s | cache=%s",
                spec.name,
                scale_base,
                marked_meta["raw_rows"],
                marked_meta["series_count"],
                marked_meta["num_marks"],
                marked_meta["max_order"],
                cache_root,
            )

    stage1_pairs = [(scale_base, candidate) for scale_base in search_cfg.log_bases for candidate in candidates]
    logger.info(
        "Stage 1 search over %s global candidate/base pairs; marked_target runs are filtered to base=%s and %s",
        len(stage1_pairs),
        MARKED_TARGET_DEFAULT_SCALE_BASE,
        sorted(MARKED_TARGET_TITAN_CANDIDATES),
    )
    stage1_rows = run_stage(
        stage_name="stage1",
        search_cfg=search_cfg,
        dataset_specs=dataset_specs,
        candidate_pairs=stage1_pairs,
        seeds=search_cfg.stage1_seeds,
        epochs=search_cfg.stage1_epochs,
        marked_cache=marked_cache,
        logger=logger,
    )

    stage1_dataset_combo_df, stage1_combo_df = build_combo_leaderboard(stage1_rows)
    if stage1_dataset_combo_df.height > 0:
        stage1_dataset_combo_df.write_parquet(leaderboard_dir / "stage1_dataset_combo.parquet")
        stage1_dataset_combo_df.write_csv(leaderboard_dir / "stage1_dataset_combo.csv")
    if stage1_combo_df.height > 0:
        stage1_combo_df.write_parquet(leaderboard_dir / "stage1_combo.parquet")
        stage1_combo_df.write_csv(leaderboard_dir / "stage1_combo.csv")

    logger.info("Stage 1 complete. Top combos:\n%s", stage1_combo_df.head(search_cfg.stage1_top_k))

    if args.skip_stage2:
        write_final_summary(
            search_cfg=search_cfg,
            dataset_combo_df=stage1_dataset_combo_df,
            combo_df=stage1_combo_df,
        )
        logger.info("Skipped stage 2 by request.")
        return

    top_stage1 = select_top_stage1_combos(stage1_combo_df, search_cfg.stage1_top_k)
    stage2_pairs = [
        (scale_base, find_candidate_by_name(candidates, candidate_name))
        for scale_base, candidate_name in top_stage1
    ]
    if any(is_marked_target_kind(spec.kind) for spec in dataset_specs):
        marked_pair = (
            MARKED_TARGET_DEFAULT_SCALE_BASE,
            find_candidate_by_name(candidates, "small_lmm"),
        )
        if not any(
            abs(scale_base - marked_pair[0]) < 1e-9 and candidate.name == marked_pair[1].name
            for scale_base, candidate in stage2_pairs
        ):
            stage2_pairs.append(marked_pair)

    logger.info("Stage 2 refinement over %s combos", len(stage2_pairs))
    stage2_rows = run_stage(
        stage_name="stage2",
        search_cfg=search_cfg,
        dataset_specs=dataset_specs,
        candidate_pairs=stage2_pairs,
        seeds=search_cfg.stage2_seeds,
        epochs=search_cfg.stage2_epochs,
        marked_cache=marked_cache,
        logger=logger,
    )

    stage2_dataset_combo_df, stage2_combo_df = build_combo_leaderboard(stage2_rows)
    if stage2_dataset_combo_df.height > 0:
        stage2_dataset_combo_df.write_parquet(leaderboard_dir / "stage2_dataset_combo.parquet")
        stage2_dataset_combo_df.write_csv(leaderboard_dir / "stage2_dataset_combo.csv")
    if stage2_combo_df.height > 0:
        stage2_combo_df.write_parquet(leaderboard_dir / "stage2_combo.parquet")
        stage2_combo_df.write_csv(leaderboard_dir / "stage2_combo.csv")

    write_final_summary(
        search_cfg=search_cfg,
        dataset_combo_df=stage2_dataset_combo_df,
        combo_df=stage2_combo_df,
    )

    logger.info("Stage 2 complete. Best overall combo:\n%s", stage2_combo_df.head(1))


if __name__ == "__main__":
    main()
