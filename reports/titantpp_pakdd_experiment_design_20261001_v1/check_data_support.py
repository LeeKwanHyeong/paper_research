"""Inspect distribution support of train/validation only; no predictions or fitting."""
import datetime
import hashlib
import json
from pathlib import Path

import polars as pl

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    manifest = json.loads((ROOT / "reports/titantpp_independent_evaluation_protocol_20261001_v1/dataset_manifest.json").read_text())
    records = []
    for dataset in manifest["datasets"]:
        identity = dataset["identities"].get("deployed_data", dataset["identities"]["data"])
        path = ROOT / identity["path"]
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        assert digest == identity["sha256"]
        # The filter precedes materialization. Only permitted columns are selected.
        frame = (pl.scan_parquet(path)
                 .filter(pl.col("chronological_split").is_in(["train", "validation"]))
                 .select("oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split")
                 .with_columns(pl.col("seq").min().over("oper_part_no").alias("first_seq")))
        duplicate_count = frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().sum()).collect().item()
        first_nontrain = frame.filter(pl.col("seq") == pl.col("first_seq")).select((pl.col("chronological_split") != "train").sum()).collect().item()
        assert duplicate_count == first_nontrain == 0
        groups = frame.group_by("chronological_split").agg(
            pl.len().alias("rows"),
            pl.col("oper_part_no").n_unique().alias("entities"),
            (pl.col("seq") > pl.col("first_seq")).sum().alias("next_event_targets"),
            (pl.col("delta_t") == 0).sum().alias("raw_gap_zero"),
            ((pl.col("delta_t") == 0) & (pl.col("seq") != pl.col("first_seq"))).sum().alias("raw_gap_zero_nonfirst"),
            (pl.col("delta_t") == 30).sum().alias("recorded_gap_30"),
            *[expression for column in ["demand_qty", "delta_t"] for expression in [
                pl.col(column).min().alias(column + "_min"),
                pl.col(column).max().alias(column + "_max"),
                (pl.col(column).is_null() | ~pl.col(column).is_finite() |
                 (pl.col(column) != pl.col(column).round()) |
                 (pl.col(column) < (1 if column == "demand_qty" else 0))).sum().alias(column + "_invalid"),
            ]],
        ).sort("chronological_split").collect().to_dicts()
        assert all(g["demand_qty_invalid"] == g["delta_t_invalid"] == g["raw_gap_zero_nonfirst"] == 0 for g in groups)
        records.append({"dataset": dataset["dataset"], "path": identity["path"], "sha256": digest,
                        "groups": groups, "duplicate_entity_seq": duplicate_count,
                        "first_entity_rows_outside_train": first_nontrain,
                        "loader_policy": "existing clip_dt_min1=True: initial history gap 0 becomes 1; entity-first rows are not targets",
                        "native_shifted_nb_support_compatible_after_existing_loader": True,
                        "top_code": 30 if dataset["dataset"] == "insta_market_basket" else None})
    report = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "passed_support_only", "scope": "train_validation_only",
              "heldout_content_read": False, "gpu_or_fit": False,
              "new_rounding_filtering_or_target_exclusion": False, "datasets": records}
    (OUT / "data_support.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "datasets": len(records), "heldout_content_read": False}))


if __name__ == "__main__":
    main()
