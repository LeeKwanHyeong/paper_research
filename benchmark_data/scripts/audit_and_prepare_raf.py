from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from common import ROOT, artifact_record, write_json


DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "candidates"
    / "raf_spare_parts"
    / "raw"
    / "RAF data - 7 years demand - 5000 items.xls"
)
DEFAULT_OUTPUT = ROOT / "data" / "candidates" / "raf_spare_parts"
MONTH_PATTERN = re.compile(r"^[A-Z]{3}\d{2}$")


def parse_month(label: str) -> pd.Timestamp:
    return pd.to_datetime(label, format="%b%y")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        workbook = pd.ExcelFile(source, engine="xlrd")
    except ImportError as exc:
        raise SystemExit(
            "xlrd is required for RAF .xls audit. Install it outside the repository "
            "or set PYTHONPATH to a local dependency directory."
        ) from exc

    if len(workbook.sheet_names) != 1:
        raise ValueError(f"expected one RAF sheet, found {workbook.sheet_names}")
    sheet = workbook.sheet_names[0]
    frame = pd.read_excel(source, sheet_name=sheet, engine="xlrd")
    month_columns = [str(column) for column in frame.columns if MONTH_PATTERN.match(str(column))]
    if len(month_columns) != 84:
        raise ValueError(f"expected 84 monthly columns, found {len(month_columns)}")

    quantity = frame[month_columns].apply(pd.to_numeric, errors="coerce")
    if quantity.isna().any().any() or (quantity < 0).any().any():
        raise ValueError("RAF monthly demand must be complete and non-negative")
    if frame["Item Ref no"].duplicated().any():
        raise ValueError("RAF item references must be unique")

    long = frame[["Item Ref no", "DESCRIPTION", "Lead Time (months)", "PRICE (£)"]].join(
        quantity
    ).melt(
        id_vars=["Item Ref no", "DESCRIPTION", "Lead Time (months)", "PRICE (£)"],
        value_vars=month_columns,
        var_name="month_label",
        value_name="demand_qty",
    )
    long["event_month"] = long["month_label"].map(parse_month)
    long = long.sort_values(["Item Ref no", "event_month"], kind="stable")

    train_end = parse_month(month_columns[57])
    validation_end = parse_month(month_columns[70])
    long["chronological_split"] = np.select(
        [long["event_month"] <= train_end, long["event_month"] <= validation_end],
        ["train", "validation"],
        default="test",
    )
    events = long.loc[long["demand_qty"] > 0].copy()
    events["delta_t"] = (
        events.groupby("Item Ref no")["event_month"]
        .diff()
        .map(lambda value: np.nan if pd.isna(value) else value.days)
        .div(30.4375)
        .round()
    )
    events["event_index"] = events.groupby("Item Ref no").cumcount() + 1
    events["event_month"] = events["event_month"].dt.strftime("%Y-%m-01")

    full_path = output / "raf_monthly_panel.parquet"
    events_path = output / "raf_positive_events.parquet"
    long.to_parquet(full_path, index=False)
    events.to_parquet(events_path, index=False)

    positive = quantity.where(quantity > 0).stack()
    events_per_item = (quantity > 0).sum(axis=1)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "raf_spare_parts",
        "source": artifact_record(source),
        "audit": {
            "sheet_names": workbook.sheet_names,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "items": int(frame["Item Ref no"].nunique()),
            "months": len(month_columns),
            "date_start": parse_month(month_columns[0]).strftime("%Y-%m"),
            "date_end": parse_month(month_columns[-1]).strftime("%Y-%m"),
            "missing_monthly_values": int(quantity.isna().sum().sum()),
            "negative_monthly_values": int((quantity < 0).sum().sum()),
            "zero_monthly_values": int((quantity == 0).sum().sum()),
            "positive_events": int((quantity > 0).sum().sum()),
            "zero_rate": float((quantity == 0).sum().sum() / quantity.size),
            "positive_quantity": {
                "mean": float(positive.mean()),
                "p50": float(positive.quantile(0.50)),
                "p90": float(positive.quantile(0.90)),
                "p95": float(positive.quantile(0.95)),
                "p99": float(positive.quantile(0.99)),
                "max": float(positive.max()),
            },
            "positive_events_per_item": {
                "min": int(events_per_item.min()),
                "p50": float(events_per_item.quantile(0.50)),
                "p95": float(events_per_item.quantile(0.95)),
                "max": int(events_per_item.max()),
            },
            "lead_time_months": {
                "min": float(frame["Lead Time (months)"].min()),
                "p50": float(frame["Lead Time (months)"].median()),
                "max": float(frame["Lead Time (months)"].max()),
            },
        },
        "split_counts": {
            key: int(value)
            for key, value in events["chronological_split"].value_counts().sort_index().items()
        },
        "artifacts": {
            "monthly_panel": artifact_record(full_path),
            "positive_events": artifact_record(events_path),
        },
        "qualification": {
            "structural_status": "qualified",
            "tpp_convertible": True,
            "publication_status": "conditional",
            "blocker": "No explicit repository license was found; confirm publication reuse permission.",
        },
    }
    manifest_path = ROOT / "manifests" / "raf_spare_parts_v1.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
