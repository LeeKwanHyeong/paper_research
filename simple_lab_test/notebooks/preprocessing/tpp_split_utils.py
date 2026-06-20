from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import polars as pl


REQUIRED_EVENT_COLUMNS = {"oper_part_no", "demand_dt", "seq", "demand_qty"}
SPLIT_VALUES = ("train", "validation", "test")


def resolve_project_root(start: Path | None = None) -> Path:
    """
    Resolve the paper_research project root from a notebook or script context.

    Jupyter kernels are often launched from the home directory or a nested
    notebook folder, so this helper walks upward until the project layout is
    found. If the project was copied to the Linux server, the common server path
    is checked as a final fallback.
    """
    candidates: list[Path] = []
    anchor = (start or Path.cwd()).expanduser().resolve()
    base = anchor if anchor.is_dir() else anchor.parent
    candidates.extend([base, *base.parents])
    candidates.append(Path("~/workspace/paper_research").expanduser())
    candidates.append(Path("/home/workspace/paper_research"))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if all((candidate / name).exists() for name in ("sample_data", "models", "utils")):
            return candidate

    raise RuntimeError("Could not locate the paper_research project root.")


def ensure_project_root_on_path(project_root: Path | None = None) -> Path:
    """
    Put the project root on sys.path so notebooks can import project utilities.
    """
    root = resolve_project_root(project_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for one quantity-reconstruction TPP split artifact.

    The split is chronological within each `entity_col`. The magnitude labels
    are fitted from train rows only, then applied to all splits with the same
    `scale_base` and upper-order cap.
    """
    dataset_name: str
    input_path: Path
    output_dir: Path
    output_prefix: str
    scale_base: float
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    entity_col: str = "oper_part_no"
    order_col: str = "seq"
    clip_min_qty: float = 1.0
    min_count: int = 100
    min_coverage: float = 0.999


def _validate_ratios(cfg: SplitConfig) -> None:
    ratio_sum = cfg.train_ratio + cfg.validation_ratio + cfg.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
    if min(cfg.train_ratio, cfg.validation_ratio, cfg.test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")
    if cfg.scale_base <= 1.0:
        raise ValueError(f"scale_base must be > 1.0, got {cfg.scale_base}")


def load_event_table(path: Path) -> pl.DataFrame:
    """
    Load and minimally validate an event-level quantity table.
    """
    df = pl.read_parquet(path)
    missing = sorted(REQUIRED_EVENT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    return (
        df
        .filter(pl.col("demand_qty") > 0)
        .with_columns([
            pl.col("oper_part_no").cast(pl.String),
            pl.col("demand_dt").cast(pl.Int64),
            pl.col("seq").cast(pl.Int64),
            pl.col("demand_qty").cast(pl.Float64),
        ])
        .sort(["oper_part_no", "seq"])
    )


def add_chronological_split(df: pl.DataFrame, cfg: SplitConfig) -> pl.DataFrame:
    """
    Assign train/validation/test per entity using row order within each series.

    Very short series may not contribute to all three splits, but they are kept
    rather than dropped so the full event table remains auditable.
    """
    entity = cfg.entity_col
    order = cfg.order_col
    ranked = (
        df.sort([entity, order])
        .with_columns([
            # Ordinal rank gives a deterministic 1..n position inside each
            # entity even if two events share the same order value.
            (pl.col(order).rank("ordinal").over(entity) - 1)
            .cast(pl.Int64)
            .alias("_event_rank"),
            pl.len().over(entity).cast(pl.Int64).alias("_n_events"),
        ])
        .with_columns([
            pl.min_horizontal(
                pl.max_horizontal(
                    (pl.col("_n_events") * cfg.train_ratio).floor().cast(pl.Int64),
                    pl.lit(1),
                ),
                pl.col("_n_events"),
            ).alias("_train_cut")
        ])
        .with_columns([
            # Keep at least one validation row when the sequence is long enough,
            # while still allowing very short series to remain mostly train.
            pl.min_horizontal(
                pl.max_horizontal(
                    (pl.col("_n_events") * (cfg.train_ratio + cfg.validation_ratio))
                    .floor()
                    .cast(pl.Int64),
                    pl.col("_train_cut") + 1,
                ),
                pl.col("_n_events"),
            ).alias("_validation_cut")
        ])
        .with_columns([
            pl.when(pl.col("_event_rank") < pl.col("_train_cut"))
            .then(pl.lit("train"))
            .when(pl.col("_event_rank") < pl.col("_validation_cut"))
            .then(pl.lit("validation"))
            .otherwise(pl.lit("test"))
            .alias("chronological_split")
        ])
        .drop(["_event_rank", "_n_events", "_train_cut", "_validation_cut"])
    )
    return ranked


def fit_max_order_from_train(df_with_split: pl.DataFrame, cfg: SplitConfig) -> int:
    """
    Select the upper magnitude class using train rows only.

    This avoids looking at validation/test tail information when defining the
    mark space for quantity reconstruction.
    """
    from utils.mark_utils import suggest_max_order

    train_df = df_with_split.filter(pl.col("chronological_split") == "train")
    if train_df.height == 0:
        raise ValueError(f"{cfg.dataset_name}: train split is empty.")

    return suggest_max_order(
        train_df,
        clip_min_qty=cfg.clip_min_qty,
        min_count=cfg.min_count,
        min_coverage=cfg.min_coverage,
        log_base=cfg.scale_base,
    )


def apply_magnitude_labels(
    df_with_split: pl.DataFrame,
    cfg: SplitConfig,
    *,
    max_order: int,
) -> pl.DataFrame:
    """
    Rebuild mark and residual labels while preserving split metadata.

    Existing `mark`, `z`, or `scale_residual` columns are intentionally replaced
    so all three quantity datasets use the same magnitude-factorized definition.
    """
    label_cols = {
        "mark",
        "z",
        "scale_residual",
        "log_qty",
        "log10_qty",
        "raw_order",
        "demand_qty_clipped",
    }
    drop_cols = [col for col in label_cols if col in df_with_split.columns]
    df = df_with_split.drop(drop_cols) if drop_cols else df_with_split

    return (
        df.sort([cfg.entity_col, cfg.order_col])
        .with_columns([
            pl.col("demand_qty")
            .clip(cfg.clip_min_qty, None)
            .cast(pl.Float64)
            .alias("demand_qty_clipped"),
            (
                pl.col(cfg.order_col) - pl.col(cfg.order_col).shift(1).over(cfg.entity_col)
            )
            .fill_null(0)
            .cast(pl.Int32)
            .alias("delta_t"),
        ])
        .with_columns([
            pl.col("demand_qty_clipped").log(base=cfg.scale_base).alias("log_qty"),
            pl.col("demand_qty_clipped").log(base=10.0).alias("log10_qty"),
        ])
        .with_columns([
            pl.col("log_qty").floor().cast(pl.Int32).alias("raw_order"),
        ])
        .with_columns([
            pl.col("raw_order").clip(0, int(max_order)).cast(pl.Int32).alias("mark"),
        ])
        .with_columns([
            (pl.col("log_qty") - pl.col("mark"))
            .cast(pl.Float64)
            .alias("scale_residual"),
            pl.col("log_qty").alias("z"),
        ])
        .drop("demand_qty_clipped")
    )


def summarize_split_table(df: pl.DataFrame, cfg: SplitConfig) -> dict[str, Any]:
    """
    Build a compact manifest-friendly summary for audit and reporting.
    """
    split_counts = (
        df.group_by("chronological_split")
        .agg([
            pl.len().alias("rows"),
            pl.col(cfg.entity_col).n_unique().alias("series"),
            pl.col("demand_qty").median().alias("qty_median"),
            pl.col("demand_qty").quantile(0.95).alias("qty_p95"),
            pl.col("demand_qty").max().alias("qty_max"),
        ])
        .sort("chronological_split")
    )
    mark_counts = (
        df.group_by(["chronological_split", "mark"])
        .len()
        .sort(["chronological_split", "mark"])
    )
    seq_stats = (
        df.group_by(cfg.entity_col)
        .len()
        .select([
            pl.len().alias("series"),
            pl.col("len").mean().alias("seq_len_mean"),
            pl.col("len").median().alias("seq_len_median"),
            pl.col("len").quantile(0.95).alias("seq_len_p95"),
            pl.col("len").max().alias("seq_len_max"),
        ])
    )

    return {
        "split_counts": split_counts.to_dicts(),
        "mark_counts": mark_counts.to_dicts(),
        "sequence_length_summary": seq_stats.to_dicts()[0] if seq_stats.height else {},
    }


def write_split_artifacts(
    df: pl.DataFrame,
    cfg: SplitConfig,
    *,
    max_order: int,
    overwrite: bool = True,
) -> dict[str, Path]:
    """
    Save with-split and individual split parquet files plus a JSON manifest.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "with_split": cfg.output_dir / f"{cfg.output_prefix}_with_split.parquet",
        "train": cfg.output_dir / f"{cfg.output_prefix}_train.parquet",
        "validation": cfg.output_dir / f"{cfg.output_prefix}_validation.parquet",
        "test": cfg.output_dir / f"{cfg.output_prefix}_test.parquet",
        "manifest": cfg.output_dir / f"{cfg.output_prefix}_split_manifest.json",
    }

    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing artifacts: {existing}")

    df.write_parquet(paths["with_split"])
    for split in SPLIT_VALUES:
        df.filter(pl.col("chronological_split") == split).write_parquet(paths[split])

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            **asdict(cfg),
            "input_path": str(cfg.input_path),
            "output_dir": str(cfg.output_dir),
        },
        "magnitude_rule": {
            "scale_base": cfg.scale_base,
            "min_order": 0,
            "max_order": int(max_order),
            "fitted_on": "train",
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
        "summary": summarize_split_table(df, cfg),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def build_and_save_quantity_splits(
    cfg: SplitConfig,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    End-to-end helper used by the preprocessing notebooks.
    """
    _validate_ratios(cfg)
    raw_df = load_event_table(cfg.input_path)
    split_df = add_chronological_split(raw_df, cfg)
    max_order = fit_max_order_from_train(split_df, cfg)
    labeled_df = apply_magnitude_labels(split_df, cfg, max_order=max_order)
    paths = write_split_artifacts(
        labeled_df,
        cfg,
        max_order=max_order,
        overwrite=overwrite,
    )
    return {
        "raw_df": raw_df,
        "split_df": split_df,
        "labeled_df": labeled_df,
        "max_order": max_order,
        "paths": paths,
        "summary": summarize_split_table(labeled_df, cfg),
    }
