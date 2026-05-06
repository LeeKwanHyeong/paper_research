"""
Minimal end-to-end example for the professor's new direction.

Flow:
1. load intermittent weekly demand data
2. rebuild mark as log-base order class
3. train TitanTPP with value residual head
4. simulate future weekly demand grid without rep_qty tables
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from simple_lab_test.common.pathing import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(__file__)

from models.RMTPPs.config import RMTPPConfig
from models.Titan import TitanConfig
from utils.magnitude_pipeline import (
    build_magnitude_marked_df,
    compare_scale_bases,
    forecast_part_week_grid,
    train_magnitude_titantpp,
)
from utils.tpp_simulation import grid_to_int_list
from utils.training import TrainingConfig


def main():
    raw_df = pl.read_parquet(PROJECT_ROOT / "sample_data" / "intermittent_df.parquet")

    # Compare several candidate bases first so we can see whether log10 is too
    # coarse and whether log4/log2 spreads the head class more evenly.
    compare_info = compare_scale_bases(raw_df, log_bases=(10.0, 4.0, 2.0))
    print("base comparison summary")
    print(compare_info["summary"])
    print("stacked raw distributions")
    print(compare_info["raw_distribution"])

    scale_base = 4.0
    marked_df, meta = build_magnitude_marked_df(raw_df, scale_base=scale_base)
    print("raw order distribution")
    print(meta["raw_distribution"])
    print("marked distribution")
    print(meta["marked_distribution"])
    print("suggested max_order", meta["max_order"])
    print("scale_base", meta["scale_base"])

    training_cfg = TrainingConfig(
        lookback=52,
        max_seq_len=64,
        batch_size=128,
        epochs=2,
        lambda_value=1.0,
        lambda_dt=1.0,
    )
    rmtpp_cfg = RMTPPConfig(
        num_marks=meta["num_marks"],
        rnn_type="gru",
        rnn_hidden_dim=64,
        scale_base=scale_base,
    )
    titan_cfg = TitanConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        contextual_mem_size=16,
        persistent_mem_size=16,
        use_lmm=True,
        mem_size=64,
        mem_topk=4,
        use_causal=True,
    )

    model, info = train_magnitude_titantpp(
        marked_df,
        training_config=training_cfg,
        rmtpp_config=rmtpp_cfg,
        titan_config=titan_cfg,
    )
    print("best_score", info["best_score"])

    part_id = marked_df["oper_part_no"][0]
    forecast = forecast_part_week_grid(
        model,
        marked_df,
        oper_part_no=part_id,
        history_len=32,
        horizon_weeks=13,
        n_sims=32,
        sample_mark=False,
    )
    print("forecast part", part_id)
    print("mean_grid", forecast["mean_grid"])
    print("rounded_grid", grid_to_int_list(forecast["mean_grid"], rounding="round"))


if __name__ == "__main__":
    main()
