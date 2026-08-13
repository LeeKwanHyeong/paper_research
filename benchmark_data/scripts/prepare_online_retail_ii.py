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

    events = (
        cleaned.groupby(["StockCode", "InvoiceDate"], as_index=False, sort=True)
        .agg(
            demand_qty=("Quantity", "sum"),
            invoice_rows=("Invoice", "size"),
            country=("Country", lambda values: sorted(values.dropna().astype(str))[0] if values.notna().any() else ""),
        )
        .sort_values(["StockCode", "InvoiceDate"], kind="stable")
    )
    unique_timestamps = np.sort(events["InvoiceDate"].unique())
    train_boundary = pd.Timestamp(unique_timestamps[int(np.floor(len(unique_timestamps) * 0.70)) - 1])
    validation_boundary = pd.Timestamp(unique_timestamps[int(np.floor(len(unique_timestamps) * 0.85)) - 1])
    events["chronological_split"] = np.select(
        [events["InvoiceDate"] <= train_boundary, events["InvoiceDate"] <= validation_boundary],
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
        events.groupby("StockCode")["InvoiceDate"].diff().dt.total_seconds().div(3600)
    )

    event_path = output / "online_retail_ii_positive_events.parquet"
    eligibility_path = output / "online_retail_ii_train_eligible_skus.csv"
    events.to_parquet(event_path, index=False)
    eligible.rename_axis("StockCode").reset_index().to_csv(eligibility_path, index=False)

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
            "date_start": events["InvoiceDate"].min().isoformat(),
            "date_end": events["InvoiceDate"].max().isoformat(),
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
    print(manifest_path)


if __name__ == "__main__":
    main()
