from __future__ import annotations

import json
from pathlib import Path
import sys

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.run_taxi_quantity_interface_ablation import make_loader


DATA_ROOT = (
    PROJECT_ROOT / "benchmark_data" / "data" / "candidates" / "raf_spare_parts"
)


def main() -> None:
    manifest = json.loads(
        (DATA_ROOT / "raf_spare_parts_split_manifest.json").read_text()
    )
    frame = pl.read_parquet(DATA_ROOT / "raf_spare_parts_with_split.parquet").sort(
        ["oper_part_no", "seq"]
    )
    frame = frame.with_columns(
        [
            pl.lit(0, dtype=pl.Int32).alias("mark"),
            pl.col("demand_qty").cast(pl.Float64).alias("scale_residual"),
        ]
    )

    expected_targets = manifest["next_event_target_counts"]
    for split in ("train", "validation"):
        loader = make_loader(
            frame,
            target_split=split,
            batch_size=128,
            lookback_weeks=84,
            max_seq_len=84,
            shuffle=False,
            generator=None,
        )
        if len(loader.dataset) != expected_targets[split]:
            raise AssertionError(
                f"{split}: expected {expected_targets[split]} targets, "
                f"found {len(loader.dataset)}"
            )
        marks, dts, mask, _, quantities = next(iter(loader))
        if quantities is None or len({marks.shape, dts.shape, mask.shape, quantities.shape}) != 1:
            raise AssertionError(f"{split}: malformed model batch")
        if int(mask.sum(dim=1).min()) < 2:
            raise AssertionError(f"{split}: target has no history event")
        if float(quantities[mask].min()) <= 0:
            raise AssertionError(f"{split}: nonpositive quantity entered the event batch")

        print(
            f"{split}: targets={len(loader.dataset)}, "
            f"batch_shape={tuple(marks.shape)}, valid_tokens={int(mask.sum())}"
        )

    print("RAF model input: PASS")


if __name__ == "__main__":
    main()
