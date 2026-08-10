from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import polars as pl

from simple_lab_test.notebooks.preprocessing.tpp_split_utils import (
    SplitConfig,
    build_and_save_quantity_splits,
)


DEFAULT_SOURCE = Path("/Users/igwanhyeong/data/demand_engine/data_v2/intermittent.parquet")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_score(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def proportional_quotas(
    sizes: dict[str, int],
    total: int,
    *,
    minimum: int = 0,
) -> dict[str, int]:
    if total < 0 or minimum < 0:
        raise ValueError("total and minimum must be non-negative")
    if not sizes:
        if total == 0:
            return {}
        raise ValueError("cannot allocate a positive total across no groups")
    if any(size < minimum for size in sizes.values()):
        raise ValueError("a group is smaller than the required minimum")
    if total < minimum * len(sizes) or total > sum(sizes.values()):
        raise ValueError("requested total is outside the feasible allocation range")

    quotas = {name: minimum for name in sizes}
    remaining = total - sum(quotas.values())
    capacities = {name: sizes[name] - quotas[name] for name in sizes}

    while remaining:
        active = {name: cap for name, cap in capacities.items() if cap > 0}
        if not active:
            raise ValueError("allocation exhausted before reaching requested total")
        weight_sum = sum(active.values())
        raw = {name: remaining * cap / weight_sum for name, cap in active.items()}
        floors = {
            name: min(capacities[name], int(math.floor(value)))
            for name, value in raw.items()
        }
        assigned = sum(floors.values())
        for name, count in floors.items():
            quotas[name] += count
            capacities[name] -= count
        remaining -= assigned
        if remaining == 0:
            break
        order = sorted(
            active,
            key=lambda name: (-(raw[name] - math.floor(raw[name])), name),
        )
        for name in order:
            if remaining == 0:
                break
            if capacities[name] <= 0:
                continue
            quotas[name] += 1
            capacities[name] -= 1
            remaining -= 1
    return quotas


def quantile_thresholds(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return []
    return [
        ordered[min(len(ordered) - 1, math.ceil(len(ordered) * i / bins) - 1)]
        for i in range(1, bins)
    ]


def bin_value(value: float, thresholds: Iterable[float]) -> int:
    return sum(value > threshold for threshold in thresholds)


def select_series(
    stats: pl.DataFrame,
    *,
    sample_size: int,
    seed: int,
    min_per_site: int,
    bins: int,
) -> pl.DataFrame:
    rows = stats.sort(["site_cd", "oper_part_no"]).to_dicts()
    by_site: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_site[str(row["site_cd"])].append(row)

    site_quotas = proportional_quotas(
        {site: len(site_rows) for site, site_rows in by_site.items()},
        sample_size,
        minimum=min_per_site,
    )
    selected: list[dict] = []
    for site in sorted(by_site):
        site_rows = by_site[site]
        event_thresholds = quantile_thresholds(
            [float(row["event_count"]) for row in site_rows], bins
        )
        qty_thresholds = quantile_thresholds(
            [float(row["train_qty_p95"]) for row in site_rows], bins
        )
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in site_rows:
            event_bin = bin_value(float(row["event_count"]), event_thresholds)
            qty_bin = bin_value(float(row["train_qty_p95"]), qty_thresholds)
            strata[f"{event_bin}:{qty_bin}"].append(row)
        stratum_quotas = proportional_quotas(
            {name: len(group) for name, group in strata.items()},
            site_quotas[site],
        )
        for name in sorted(strata):
            ranked = sorted(
                strata[name],
                key=lambda row: stable_score(seed, str(row["oper_part_no"])),
            )
            for row in ranked[: stratum_quotas[name]]:
                selected.append({
                    **row,
                    "event_bin": int(name.split(":")[0]),
                    "quantity_bin": int(name.split(":")[1]),
                    "sampling_seed": seed,
                })
    if len(selected) != sample_size:
        raise RuntimeError(f"selected {len(selected)} series, expected {sample_size}")
    return pl.DataFrame(selected).sort(["site_cd", "oper_part_no"])


def build_positive_events(source: Path, *, min_events: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    positive = (
        pl.scan_parquet(source)
        .filter(pl.col("order_qty") > 0)
        .select(["site_cd", "part_no", "order_dt", "order_qty"])
        .with_columns(
            (pl.col("site_cd") + pl.lit("::") + pl.col("part_no")).alias("oper_part_no"),
            pl.col("order_dt")
            .cast(pl.String)
            .str.strptime(pl.Date, "%Y%m%d")
            .alias("event_date"),
        )
        .sort(["oper_part_no", "event_date"])
        .collect()
    )
    positive = positive.with_columns(
        (
            (
                (pl.col("event_date") - pl.col("event_date").min().over("oper_part_no"))
                .dt.total_days()
                / 7
            ).floor()
            + 1
        )
        .cast(pl.Int64)
        .alias("seq")
    )
    counts = positive.group_by(["site_cd", "oper_part_no"]).agg(
        pl.len().alias("event_count")
    )
    eligible = counts.filter(pl.col("event_count") >= min_events)
    positive = positive.join(
        eligible.select(["site_cd", "oper_part_no"]),
        on=["site_cd", "oper_part_no"],
        how="inner",
    )

    provisional = positive.join(
        eligible, on=["site_cd", "oper_part_no"], how="left"
    ).with_columns(
        pl.int_range(pl.len()).over("oper_part_no").add(1).alias("event_rank")
    )
    train_stats = (
        provisional
        .filter(pl.col("event_rank") <= (pl.col("event_count") * 0.70).floor())
        .group_by(["site_cd", "oper_part_no"])
        .agg(pl.col("order_qty").quantile(0.95).alias("train_qty_p95"))
    )
    stats = eligible.join(train_stats, on=["site_cd", "oper_part_no"], how="inner")
    return positive, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-events", type=int, default=20)
    parser.add_argument("--min-per-site", type=int, default=40)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--prefix", default="intermittent_frozen_5000")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    positive, stats = build_positive_events(source, min_events=args.min_events)
    selected = select_series(
        stats,
        sample_size=args.sample_size,
        seed=args.seed,
        min_per_site=args.min_per_site,
        bins=args.bins,
    )
    event_table = (
        positive.join(
            selected.select("oper_part_no"), on="oper_part_no", how="inner"
        )
        .select(
            pl.col("oper_part_no"),
            pl.col("order_dt").cast(pl.Int64).alias("demand_dt"),
            pl.col("seq").cast(pl.Int64),
            pl.col("order_qty").cast(pl.Float64).alias("demand_qty"),
        )
        .sort(["oper_part_no", "seq"])
    )
    event_path = output_dir / f"{args.prefix}_events.parquet"
    selected_path = output_dir / f"{args.prefix}_selected_series.parquet"
    event_table.write_parquet(event_path)
    selected.write_parquet(selected_path)

    split_cfg = SplitConfig(
        dataset_name="intermittent",
        input_path=event_path,
        output_dir=output_dir,
        output_prefix=args.prefix,
        scale_base=2.0,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        min_count=100,
        min_coverage=0.999,
    )
    split_result = build_and_save_quantity_splits(split_cfg)
    paths = split_result["paths"]
    artifact_paths = {"events": event_path, "selected_series": selected_path, **paths}
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "sampling": {
            "sample_size": args.sample_size,
            "seed": args.seed,
            "min_events": args.min_events,
            "min_per_site": args.min_per_site,
            "bins": args.bins,
            "site_count": int(selected["site_cd"].n_unique()),
        },
        "split_config": {**asdict(split_cfg), "input_path": str(event_path), "output_dir": str(output_dir)},
        "max_order": int(split_result["max_order"]),
        "artifacts": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
    }
    manifest_path = output_dir / f"{args.prefix}_sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "events": event_table.height,
        "series": selected.height,
        "sites": int(selected["site_cd"].n_unique()),
        "max_order": int(split_result["max_order"]),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
