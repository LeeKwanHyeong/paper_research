"""
Quick utility script for inspecting how the mark distribution changes when we
replace log10 binning with log4 or log2 binning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.magnitude_pipeline import compare_scale_bases


def main():
    raw_df = pl.read_parquet("sample_data/intermittent_df.parquet")

    # The summary table is compact enough to paste into notes or slides, while
    # the raw stacked table keeps the per-class counts available for inspection.
    compare_info = compare_scale_bases(
        raw_df,
        log_bases=(10.0, 4.0, 2.0),
        min_count=100,
        min_coverage=0.999,
    )

    print("=== base comparison summary ===")
    print(compare_info["summary"])
    print()
    print("=== stacked raw distributions ===")
    print(compare_info["raw_distribution"])


if __name__ == "__main__":
    main()
