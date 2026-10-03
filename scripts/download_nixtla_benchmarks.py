#!/usr/bin/env python3
"""Download Nixtla datasetsforecast benchmarks and save parquet copies.

The script keeps raw Nixtla downloads under ``_nixtla_raw`` and writes
analysis-ready parquet files in one subdirectory per dataset family/group.
It records success/failure metadata in JSON and CSV manifests so partial
downloads remain inspectable.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class DatasetJob:
    family: str
    group: str
    component_names: tuple[str, ...]
    loader: Callable[[Path, str], Any]


def _m3_loader(raw_dir: Path, group: str) -> Any:
    from datasetsforecast.m3 import M3

    return M3.load(str(raw_dir), group)


def _m4_loader(raw_dir: Path, group: str) -> Any:
    from datasetsforecast.m4 import M4

    if group == "Other":
        parts = [M4.load(str(raw_dir), part, cache=True) for part in ("Weekly", "Daily", "Hourly")]
        y_frames = [part[0] for part in parts if part[0] is not None]
        x_frames = [part[1] for part in parts if len(part) > 1 and part[1] is not None]
        s_frames = [part[2] for part in parts if len(part) > 2 and part[2] is not None]
        return (
            pd.concat(y_frames, ignore_index=True) if y_frames else None,
            pd.concat(x_frames, ignore_index=True) if x_frames else None,
            pd.concat(s_frames, ignore_index=True) if s_frames else None,
        )
    return M4.load(str(raw_dir), group, cache=True)


def _m5_loader(raw_dir: Path, group: str) -> Any:
    from datasetsforecast.m5 import M5

    return M5.load(str(raw_dir), cache=True)


def _long_horizon_loader(raw_dir: Path, group: str) -> Any:
    from datasetsforecast.long_horizon import LongHorizon

    return LongHorizon.load(str(raw_dir), group, cache=True)


def _long_horizon2_loader(raw_dir: Path, group: str) -> Any:
    from datasetsforecast.long_horizon2 import LongHorizon2

    return LongHorizon2.load(str(raw_dir), group, normalize=True)


def _favorita_loader(raw_dir: Path, group: str) -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    from datasetsforecast.favorita import FavoritaData, FavoritaRawData

    FavoritaRawData.download(str(raw_dir))
    train_feather = raw_dir / "train.feather"
    if train_feather.exists():
        train_df = pd.read_feather(train_feather)
        if not pd.api.types.is_datetime64_any_dtype(train_df["date"]):
            train_df["date"] = pd.to_datetime(pd.to_numeric(train_df["date"]), unit="ns")
            train_df.to_feather(train_feather)
    y_df, s_df, _tags = FavoritaData.load(str(raw_dir), group, cache=True, verbose=False)
    return y_df, s_df


def _hierarchical_loader(raw_dir: Path, group: str) -> Any:
    from datasetsforecast.hierarchical import HierarchicalData

    y_df, s_df, _tags = HierarchicalData.load(str(raw_dir), group, cache=True)
    return y_df, s_df


def _phm2008_loader(raw_dir: Path, group: str) -> Any:
    from datasetsforecast.phm2008 import PHM2008

    return PHM2008.load(str(raw_dir), group, clip_rul=True)


def build_jobs() -> list[DatasetJob]:
    jobs: list[DatasetJob] = []
    for group in ("Yearly", "Quarterly", "Monthly", "Other"):
        jobs.append(DatasetJob("m3", group, ("y", "x", "s"), _m3_loader))
    for group in ("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly", "Other"):
        jobs.append(DatasetJob("m4", group, ("y", "x", "s"), _m4_loader))
    jobs.append(DatasetJob("m5", "complete", ("y", "x", "s"), _m5_loader))
    for group in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "Exchange", "ILI", "TrafficL", "Weather"):
        jobs.append(DatasetJob("long_horizon", group, ("y", "x", "s"), _long_horizon_loader))
    for group in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "TrafficL", "Weather"):
        jobs.append(DatasetJob("long_horizon2", group, ("data",), _long_horizon2_loader))
    for group in ("Favorita200", "Favorita500", "FavoritaComplete"):
        jobs.append(DatasetJob("favorita", group, ("y", "s"), _favorita_loader))
    for group in ("TourismSmall", "TourismLarge", "Traffic", "Labour", "Wiki2", "OldTourismLarge", "OldTraffic"):
        jobs.append(DatasetJob("hierarchical", group, ("y", "s"), _hierarchical_loader))
    for group in ("FD001", "FD002", "FD003", "FD004"):
        jobs.append(DatasetJob("phm2008", group, ("train", "test"), _phm2008_loader))
    return jobs


def normalize_slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def as_components(data: Any, names: tuple[str, ...]) -> list[tuple[str, Any]]:
    if isinstance(data, tuple):
        values = list(data)
    else:
        values = [data]
    if len(names) < len(values):
        names = names + tuple(f"component_{idx}" for idx in range(len(names), len(values)))
    return [(names[idx], value) for idx, value in enumerate(values)]


def summarize_frame(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    if "unique_id" in df.columns:
        summary["unique_id_count"] = int(df["unique_id"].nunique(dropna=True))
    if "ds" in df.columns:
        ds = df["ds"]
        summary["ds_min"] = str(ds.min())
        summary["ds_max"] = str(ds.max())
    if "y" in df.columns and pd.api.types.is_numeric_dtype(df["y"]):
        y = df["y"]
        summary["y_nulls"] = int(y.isna().sum())
        summary["y_min"] = float(y.min(skipna=True)) if len(y) else None
        summary["y_max"] = float(y.max(skipna=True)) if len(y) else None
        summary["y_mean"] = float(y.mean(skipna=True)) if len(y) else None
    return summary


def write_component(value: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if value is None:
        return {"status": "skipped_none"}
    if isinstance(value, pd.Series):
        df = value.to_frame()
    elif isinstance(value, pd.DataFrame):
        df = value
    else:
        df = pd.DataFrame(value)
    df.to_parquet(path, index=False, compression="snappy")
    summary = summarize_frame(df)
    summary["status"] = "success"
    summary["path"] = str(path)
    summary["bytes"] = int(path.stat().st_size)
    return summary


def run(output_dir: Path, include_families: set[str] | None, skip_existing: bool) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "_nixtla_raw"
    parquet_root = output_dir / "parquet"
    manifest_rows: list[dict[str, Any]] = []

    for job in build_jobs():
        if include_families and job.family not in include_families:
            continue
        group_slug = normalize_slug(job.group)
        out_dir = parquet_root / job.family / group_slug
        raw_dir = raw_root / job.family
        if skip_existing and out_dir.exists() and list(out_dir.glob("*.parquet")):
            manifest_rows.append(
                {
                    "family": job.family,
                    "group": job.group,
                    "status": "skipped_existing",
                    "output_dir": str(out_dir),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"[skip] {job.family}/{job.group}: existing parquet files found", flush=True)
            continue

        print(f"[start] {job.family}/{job.group}", flush=True)
        row: dict[str, Any] = {
            "family": job.family,
            "group": job.group,
            "output_dir": str(out_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            data = job.loader(raw_dir, job.group)
            component_summaries = {}
            for component_name, component_value in as_components(data, job.component_names):
                component_path = out_dir / f"{component_name}.parquet"
                component_summaries[component_name] = write_component(component_value, component_path)
            row["status"] = "success"
            row["components"] = component_summaries
            print(f"[done] {job.family}/{job.group}", flush=True)
        except Exception as exc:  # noqa: BLE001 - manifest needs full failure context.
            row["status"] = "failed"
            row["error"] = repr(exc)
            row["traceback"] = traceback.format_exc()
            print(f"[failed] {job.family}/{job.group}: {exc!r}", flush=True)
        manifest_rows.append(row)

        manifest_json = output_dir / "nixtla_download_manifest.json"
        manifest_json.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        write_manifest_csv(manifest_rows, output_dir / "nixtla_download_manifest.csv")

    return manifest_rows


def write_manifest_csv(rows: list[dict[str, Any]], path: Path) -> None:
    flat_rows = []
    for row in rows:
        components = row.get("components")
        if isinstance(components, dict):
            for name, summary in components.items():
                flat = {
                    "family": row.get("family"),
                    "group": row.get("group"),
                    "status": row.get("status"),
                    "component": name,
                    "path": summary.get("path") if isinstance(summary, dict) else None,
                    "rows": summary.get("rows") if isinstance(summary, dict) else None,
                    "unique_id_count": summary.get("unique_id_count") if isinstance(summary, dict) else None,
                    "bytes": summary.get("bytes") if isinstance(summary, dict) else None,
                    "ds_min": summary.get("ds_min") if isinstance(summary, dict) else None,
                    "ds_max": summary.get("ds_max") if isinstance(summary, dict) else None,
                    "error": row.get("error"),
                }
                flat_rows.append(flat)
        else:
            flat_rows.append(
                {
                    "family": row.get("family"),
                    "group": row.get("group"),
                    "status": row.get("status"),
                    "component": None,
                    "path": None,
                    "rows": None,
                    "unique_id_count": None,
                    "bytes": None,
                    "ds_min": None,
                    "ds_max": None,
                    "error": row.get("error"),
                }
            )
    pd.DataFrame(flat_rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/igwanhyeong/data/neural_forecast_benchmark"),
        help="Directory where parquet files and manifests will be written.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional family filter, e.g. m3 m4 long_horizon.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip groups that already have parquet outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_families = set(args.families) if args.families else None
    rows = run(args.output_dir, include_families, args.skip_existing)
    status_counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    print(json.dumps({"output_dir": str(args.output_dir), "status_counts": status_counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
