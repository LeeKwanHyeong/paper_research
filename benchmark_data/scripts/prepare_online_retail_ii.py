from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, artifact_record, write_json


DEFAULT_SOURCE = ROOT / "data" / "main" / "online_retail_ii" / "raw" / "online_retail_II.csv"
DEFAULT_OUTPUT = ROOT / "data" / "main" / "online_retail_ii"
STOCK_PATTERN = r"^[0-9]{4,}[A-Z0-9]*$"
SPLIT_NAMES = ("train", "validation", "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train-events", type=int, default=20)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        source,
        encoding="latin1",
        dtype={"Invoice": str, "StockCode": str},
        low_memory=False,
    )
    expected = {
        "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
        "Price", "Customer ID", "Country",
    }
    if not expected.issubset(raw.columns):
        raise ValueError(f"missing columns: {sorted(expected - set(raw.columns))}")

    raw["Invoice"] = raw["Invoice"].str.strip()
    raw["StockCode"] = raw["StockCode"].str.strip().str.upper()
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")

    valid_timestamp = raw["InvoiceDate"].notna()
    positive_quantity = raw["Quantity"] > 0
    positive_price = raw["Price"] > 0
    non_cancelled = ~raw["Invoice"].str.startswith("C", na=False)
    product_code = raw["StockCode"].str.fullmatch(STOCK_PATTERN, na=False)
    keep = valid_timestamp & positive_quantity & positive_price & non_cancelled & product_code
    cleaned = raw.loc[keep].copy()

    cleaned["event_hour"] = cleaned["InvoiceDate"].dt.floor("h")
    events = (
        cleaned.groupby(["StockCode", "event_hour"], as_index=False, sort=True)
        .agg(
            demand_qty=("Quantity", "sum"),
            invoice_rows=("Invoice", "size"),
            country=("Country", lambda values: sorted(values.dropna().astype(str))[0] if values.notna().any() else ""),
        )
        .sort_values(["StockCode", "event_hour"], kind="stable")
    )
    unique_timestamps = np.sort(events["event_hour"].unique())
    train_boundary = pd.Timestamp(unique_timestamps[int(np.floor(len(unique_timestamps) * 0.70)) - 1])
    validation_boundary = pd.Timestamp(unique_timestamps[int(np.floor(len(unique_timestamps) * 0.85)) - 1])
    events["chronological_split"] = np.select(
        [events["event_hour"] <= train_boundary, events["event_hour"] <= validation_boundary],
        ["train", "validation"],
        default="test",
    )

    train_counts = (
        events.loc[events["chronological_split"] == "train"]
        .groupby("StockCode").size().rename("train_event_count")
    )
    eligible = train_counts.loc[train_counts >= args.min_train_events]
    events = events.merge(eligible, on="StockCode", how="inner", validate="many_to_one")
    events["event_index"] = events.groupby("StockCode").cumcount() + 1
    events["delta_t_hours"] = (
        events.groupby("StockCode")["event_hour"].diff().dt.total_seconds().div(3600)
    )

    origin = pd.Timestamp(events["event_hour"].min())
    model_events = events.assign(
        oper_part_no=events["StockCode"].map(lambda value: f"ORII::{value}"),
        demand_dt=events["event_hour"].dt.strftime("%Y%m%d%H").astype("int64"),
        seq=(
            events["event_hour"].sub(origin).dt.total_seconds().div(3600).astype("int64")
            + 1
        ),
        demand_qty=events["demand_qty"].astype("float64"),
        delta_t=events["delta_t_hours"].fillna(0).round().astype("int32"),
    )
    model_events["log_qty"] = np.log2(model_events["demand_qty"])
    model_events["log10_qty"] = np.log10(model_events["demand_qty"])
    model_events["raw_order"] = np.floor(model_events["log_qty"]).astype("int32")
    train_max_order = int(
        model_events.loc[
            model_events["chronological_split"] == "train", "raw_order"
        ].max()
    )
    model_events["mark"] = model_events["raw_order"].clip(0, train_max_order).astype("int32")
    model_events["scale_residual"] = (
        model_events["log_qty"] - model_events["mark"]
    ).astype("float64")
    model_events["z"] = model_events["log_qty"]
    model_events = model_events[
        [
            "oper_part_no",
            "demand_dt",
            "seq",
            "demand_qty",
            "chronological_split",
            "delta_t",
            "log_qty",
            "log10_qty",
            "raw_order",
            "mark",
            "scale_residual",
            "z",
        ]
    ].sort_values(["oper_part_no", "seq"], kind="stable")

    event_path = output / "online_retail_ii_positive_events.parquet"
    eligibility_path = output / "online_retail_ii_train_eligible_skus.csv"
    model_path = output / "online_retail_ii_with_split.parquet"
    events.to_parquet(event_path, index=False)
    eligible.rename_axis("StockCode").reset_index().to_csv(eligibility_path, index=False)
    model_events.to_parquet(model_path, index=False)
    split_paths = {
        split: output / f"online_retail_ii_{split}.parquet" for split in SPLIT_NAMES
    }
    for split, path in split_paths.items():
        model_events.loc[model_events["chronological_split"] == split].to_parquet(
            path, index=False
        )

    quantity = events["demand_qty"]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "online_retail_ii",
        "source": artifact_record(source),
        "filter_audit": {
            "raw_rows": int(len(raw)),
            "invalid_timestamp_rows": int((~valid_timestamp).sum()),
            "nonpositive_quantity_rows": int((~positive_quantity).sum()),
            "nonpositive_price_rows": int((~positive_price).sum()),
            "cancelled_invoice_rows": int((~non_cancelled).sum()),
            "nonproduct_stock_code_rows": int((~product_code).sum()),
            "kept_rows_before_aggregation": int(keep.sum()),
        },
        "event_profile": {
            "events": int(len(events)),
            "eligible_skus": int(events["StockCode"].nunique()),
            "date_start": events["event_hour"].min().isoformat(),
            "date_end": events["event_hour"].max().isoformat(),
            "quantity": {
                "p50": float(quantity.quantile(0.50)),
                "p90": float(quantity.quantile(0.90)),
                "p95": float(quantity.quantile(0.95)),
                "p99": float(quantity.quantile(0.99)),
                "max": float(quantity.max()),
            },
        },
        "split": {
            "train_boundary": train_boundary.isoformat(),
            "validation_boundary": validation_boundary.isoformat(),
            "minimum_train_events": args.min_train_events,
            "eligibility_fit_scope": "train only",
            "counts": {
                key: int(value)
                for key, value in events["chronological_split"].value_counts().sort_index().items()
            },
        },
        "artifacts": {
            "positive_events": artifact_record(event_path),
            "train_eligible_skus": artifact_record(eligibility_path),
            "with_split": artifact_record(model_path),
            **{split: artifact_record(path) for split, path in split_paths.items()},
        },
        "qualification": {
            "structural_status": "qualified",
            "tpp_convertible": True,
            "publication_status": "qualified",
            "quantity_provenance": "native transaction quantity",
        },
    }
    manifest_path = ROOT / "manifests" / "online_retail_ii_v1.json"
    write_json(manifest_path, manifest)
    prior_event = model_events.groupby("oper_part_no").cumcount() > 0
    split_manifest = {
        "schema_version": 1,
        "dataset_id": "online_retail_ii",
        "split_method": "global chronological hourly boundary",
        "time_unit": "hour",
        "aggregation_unit": "StockCode x event hour",
        "entity_column": "oper_part_no",
        "order_column": "seq",
        "quantity_column": "demand_qty",
        "target_schema": list(model_events.columns),
        "origin": origin.isoformat(),
        "boundaries": {
            "train_end": train_boundary.isoformat(),
            "validation_end": validation_boundary.isoformat(),
        },
        "split_counts": {
            split: int((model_events["chronological_split"] == split).sum())
            for split in SPLIT_NAMES
        },
        "next_event_target_counts": {
            split: int(
                (prior_event & (model_events["chronological_split"] == split)).sum()
            )
            for split in SPLIT_NAMES
        },
        "series_counts": {
            split: int(
                model_events.loc[
                    model_events["chronological_split"] == split, "oper_part_no"
                ].nunique()
            )
            for split in SPLIT_NAMES
        },
        "eligibility": {
            "minimum_train_events": args.min_train_events,
            "fit_scope": "train only",
            "eligible_skus": int(len(eligible)),
        },
        "magnitude_compatibility": {
            "scale_base": 2.0,
            "max_order": train_max_order,
            "fitted_on": "train",
            "note": "Compatibility columns are not used by mark-free count-aware models.",
        },
        "held_out_policy": {
            "test_used_for_filtering": False,
            "test_used_for_threshold_selection": False,
            "test_used_for_model_selection": False,
        },
        "artifacts": {
            "with_split": artifact_record(model_path),
            **{split: artifact_record(path) for split, path in split_paths.items()},
        },
    }
    split_manifest_path = output / "online_retail_ii_split_manifest.json"
    write_json(split_manifest_path, split_manifest)
    print(manifest_path)
    print(split_manifest_path)


if __name__ == "__main__":
    main()
