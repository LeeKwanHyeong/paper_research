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
SPLIT_NAMES = ("train", "validation", "test")


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

    month_index = {label: index + 1 for index, label in enumerate(month_columns)}
    model_events = events.assign(
        oper_part_no=events["Item Ref no"].map(lambda value: f"RAF::{int(value):05d}"),
        demand_dt=pd.to_datetime(events["event_month"]).dt.strftime("%Y%m%d").astype("int64"),
        seq=events["month_label"].map(month_index).astype("int64"),
        demand_qty=events["demand_qty"].astype("float64"),
        delta_t=events["delta_t"].fillna(0).astype("int32"),
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

    full_path = output / "raf_monthly_panel.parquet"
    events_path = output / "raf_positive_events.parquet"
    model_path = output / "raf_spare_parts_with_split.parquet"
    long.to_parquet(full_path, index=False)
    events.to_parquet(events_path, index=False)
    model_events.to_parquet(model_path, index=False)
    split_paths = {
        split: output / f"raf_spare_parts_{split}.parquet" for split in SPLIT_NAMES
    }
    for split, path in split_paths.items():
        model_events.loc[model_events["chronological_split"] == split].to_parquet(
            path, index=False
        )

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
            "with_split": artifact_record(model_path),
            **{split: artifact_record(path) for split, path in split_paths.items()},
        },
        "qualification": {
            "structural_status": "qualified",
            "tpp_convertible": True,
            "publication_status": "research_use_with_citation",
            "redistribution_status": "not_cleared",
            "usage_note": (
                "Use for non-commercial academic analysis and report aggregate results "
                "with repository and provenance citations. Do not redistribute the raw "
                "workbook because the repository provides no explicit data license."
            ),
        },
    }
    manifest_path = ROOT / "manifests" / "raf_spare_parts_v1.json"
    write_json(manifest_path, manifest)
    split_manifest = {
        "schema_version": 1,
        "dataset_id": "raf_spare_parts",
        "split_method": "global chronological month boundary",
        "time_unit": "month",
        "entity_column": "oper_part_no",
        "order_column": "seq",
        "quantity_column": "demand_qty",
        "target_schema": list(model_events.columns),
        "boundaries": {
            "train": "1996-01 through 2000-10",
            "validation": "2000-11 through 2001-11",
            "test": "2001-12 through 2002-12",
        },
        "split_counts": {
            split: int((model_events["chronological_split"] == split).sum())
            for split in SPLIT_NAMES
        },
        "next_event_target_counts": {
            split: int(
                (
                    (model_events.groupby("oper_part_no").cumcount() > 0)
                    & (model_events["chronological_split"] == split)
                ).sum()
            )
            for split in SPLIT_NAMES
        },
        "target_count_note": (
            "The first positive event of each item supplies history but cannot be a "
            "next-event target. Validation and test events retain all earlier events "
            "as context."
        ),
        "series_counts": {
            split: int(
                model_events.loc[
                    model_events["chronological_split"] == split, "oper_part_no"
                ].nunique()
            )
            for split in SPLIT_NAMES
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
    split_manifest_path = output / "raf_spare_parts_split_manifest.json"
    write_json(split_manifest_path, split_manifest)
    print(manifest_path)
    print(split_manifest_path)


if __name__ == "__main__":
    main()
