from __future__ import annotations

from dataclasses import asdict

import polars as pl
import torch

from models.RMTPPs.config import RMTPPConfig
from models.Titan import TitanConfig
from utils.mark_utils import (
    compare_log_base_distributions,
    make_marks_log_magnitude,
    suggest_max_order,
    summarize_log_magnitude_distribution,
)
from utils.tpp_simulation import simulate_horizon_week_grid
from utils.training import TrainingConfig, train_titantpp


def build_magnitude_marked_df(
    df: pl.DataFrame,
    *,
    min_order: int = 0,
    max_order: int | None = None,
    clip_min_qty: float = 1.0,
    min_count: int = 100,
    min_coverage: float = 0.999,
    scale_base: float = 10.0,
):
    """
    Build the professor-style magnitude-factorized event table.

    Workflow:
    1. inspect raw log-base order distribution
    2. choose an upper order cap if the caller did not provide one
    3. create mark/order + residual labels for training
    """
    raw_dist = summarize_log_magnitude_distribution(
        df,
        clip_min_qty=clip_min_qty,
        log_base=scale_base,
    )

    if max_order is None:
        max_order = suggest_max_order(
            df,
            clip_min_qty=clip_min_qty,
            min_count=min_count,
            min_coverage=min_coverage,
            log_base=scale_base,
        )

    marked_df = make_marks_log_magnitude(
        df,
        min_order=min_order,
        max_order=max_order,
        clip_min_qty=clip_min_qty,
        log_base=scale_base,
    )

    meta = {
        "scale_base": float(scale_base),
        "min_order": int(min_order),
        "max_order": int(max_order),
        "num_marks": infer_num_marks(marked_df),
        "raw_distribution": raw_dist,
        "marked_distribution": marked_df.group_by("mark").len().sort("mark"),
    }
    return marked_df, meta


def infer_num_marks(marked_df: pl.DataFrame) -> int:
    """
    Keep config creation deterministic from the materialized training table.

    The week-lookback dataset uses PAD = K_real, so the model vocabulary must be
    K_real + 1 to include the padding token.
    """
    return int(marked_df.select(pl.col("mark").max()).item()) + 2


def train_magnitude_titantpp(
    marked_df: pl.DataFrame,
    *,
    training_config: TrainingConfig,
    rmtpp_config: RMTPPConfig,
    titan_config: TitanConfig,
):
    """
    Thin convenience wrapper that auto-fills num_marks from the prepared table.
    """
    cfg = RMTPPConfig(**{**asdict(rmtpp_config), "num_marks": infer_num_marks(marked_df)})
    return train_titantpp(
        marked_df=marked_df,
        training_config=training_config,
        rmtpp_config=cfg,
        titan_config=titan_config,
    )


def get_part_history_tensors(
    marked_df: pl.DataFrame,
    *,
    oper_part_no: str,
    history_len: int = 32,
):
    """
    Extract the most recent event history for one part in tensor form.
    """
    part_df = (
        marked_df
        .filter(pl.col("oper_part_no") == oper_part_no)
        .sort("seq")
        .tail(history_len)
    )
    if part_df.height == 0:
        raise ValueError(f"No rows found for oper_part_no={oper_part_no}")

    marks = torch.tensor(part_df["mark"].to_list(), dtype=torch.long)
    dts = torch.tensor(part_df["delta_t"].to_list(), dtype=torch.float32)
    return marks, dts, part_df


def forecast_part_week_grid(
    model,
    marked_df: pl.DataFrame,
    *,
    oper_part_no: str,
    history_len: int = 32,
    horizon_weeks: int = 13,
    n_sims: int = 100,
    sample_mark: bool = False,
    qty_cap: float | None = None,
):
    """
    End-to-end helper:
    part history -> autoregressive event simulation -> weekly demand grid.
    """
    history_marks, history_dts, part_df = get_part_history_tensors(
        marked_df,
        oper_part_no=oper_part_no,
        history_len=history_len,
    )
    mean_grid, all_grids = simulate_horizon_week_grid(
        model,
        history_marks=history_marks,
        history_dts=history_dts,
        horizon_weeks=horizon_weeks,
        n_sims=n_sims,
        sample_mark=sample_mark,
        qty_cap=qty_cap,
    )
    return {
        "oper_part_no": oper_part_no,
        "history_df": part_df,
        "mean_grid": mean_grid,
        "all_grids": all_grids,
    }


def compare_scale_bases(
    df: pl.DataFrame,
    *,
    log_bases: tuple[float, ...] = (10.0, 4.0, 2.0),
    clip_min_qty: float = 1.0,
    min_count: int = 100,
    min_coverage: float = 0.999,
) -> dict[str, object]:
    """
    Summarize how log-base choices change class spread and top-tail merging.

    The returned dictionary is notebook-friendly so we can print the stacked
    distributions and a compact recommendation table side by side.
    """
    raw_dist = compare_log_base_distributions(
        df,
        log_bases=log_bases,
        clip_min_qty=clip_min_qty,
    )

    summaries: list[dict[str, float | int]] = []
    for scale_base in log_bases:
        dist = raw_dist.filter(pl.col("log_base") == float(scale_base)).sort("raw_order")
        max_order = suggest_max_order(
            df,
            clip_min_qty=clip_min_qty,
            min_count=min_count,
            min_coverage=min_coverage,
            log_base=scale_base,
        )
        summaries.append({
            "log_base": float(scale_base),
            "num_raw_classes": int(dist.height),
            "head_ratio": float(dist["ratio"][0]) if dist.height > 0 else 0.0,
            "suggested_max_order": int(max_order),
            "tail_count": int(
                dist.filter(pl.col("raw_order") > max_order)["len"].sum() or 0
            ),
        })

    return {
        "raw_distribution": raw_dist,
        "summary": pl.DataFrame(summaries).sort("log_base"),
    }
