"""Create a separate Train-fitted Q3-capped dataset; never modify source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "intermittent": ("main/intermittent_v2", "intermittent_frozen_5000", 2.0),
    "taxi": ("auxiliary/yellow_trip_hourly", "yellow_trip_hourly", 10.0),
    "raf": ("candidates/raf_spare_parts", "raf_spare_parts", 2.0),
}
SPLITS = ("train", "validation", "test")
QUANTITY_COLUMNS = {
    "demand_qty", "log_qty", "log10_qty", "raw_order", "mark", "scale_residual", "z"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_source(table: pa.Table) -> pd.DataFrame:
    required = {"oper_part_no", "seq", "demand_dt", "chronological_split", "demand_qty"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Missing required columns: {sorted(required - set(table.column_names))}")
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError("Empty source")
    if frame[list(required)].isna().any().any():
        raise ValueError("Null required fields")
    qty = frame["demand_qty"].to_numpy(dtype=np.float64)
    if not np.isfinite(qty).all() or (qty <= 0).any():
        raise ValueError("Quantity must be finite and strictly positive")
    if not set(frame["chronological_split"]).issubset(SPLITS):
        raise ValueError("Unknown split")
    if not (frame["chronological_split"] == "train").any():
        raise ValueError("No Train events")
    if frame.duplicated(["oper_part_no", "seq"]).any():
        raise ValueError("Duplicate entity/event key")
    return frame


def fit_max_order(train_orders: np.ndarray, parent: dict) -> int:
    """Use the parent rule without its regrouping/time reconstruction functions."""
    config = parent.get("config", {})
    if "min_count" not in config or "min_coverage" not in config:
        return int(train_orders.max())
    orders, counts = np.unique(train_orders, return_counts=True)
    cumulative = 0
    for order, count in zip(orders, counts):
        cumulative += int(count)
        if cumulative / len(train_orders) >= float(config["min_coverage"]) and count < int(config["min_count"]):
            return int(order)
    return int(orders[-1])


def stats(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(np.mean(values)),
        "std_ddof0": float(np.std(values)), "min": float(np.min(values)),
        "raw_rms": float(np.sqrt(np.mean(np.square(values)))),
        "log1p_mean": float(np.log1p(values).mean()),
        "log1p_std_ddof0": float(np.log1p(values).std()),
        "q25": float(np.quantile(values, .25, method="linear")),
        "median": float(np.quantile(values, .5, method="linear")),
        "q75": float(np.quantile(values, .75, method="linear")),
        "q99": float(np.quantile(values, .99, method="linear")),
        "max": float(np.max(values)),
    }


def transform(table: pa.Table, base: float, parent: dict) -> tuple[pa.Table, dict]:
    frame = validate_source(table)
    qty = frame["demand_qty"].to_numpy(dtype=np.float64)
    train = frame["chronological_split"].eq("train").to_numpy()
    threshold = float(np.quantile(qty[train], .75, method="linear"))
    capped = np.minimum(qty, threshold)
    log_qty = np.log2(capped) if base == 2 else np.log10(capped)
    raw_order = np.floor(log_qty).astype(np.int32)
    max_order = fit_max_order(raw_order[train], parent)
    mark = np.clip(raw_order, 0, max_order).astype(np.int32)
    values = {
        "demand_qty": capped, "log_qty": log_qty, "log10_qty": np.log10(capped),
        "raw_order": raw_order, "mark": mark,
        "scale_residual": log_qty - mark, "z": log_qty,
    }
    transformed = table
    for column in table.column_names:
        if column in values:
            field = table.schema.field(column)
            transformed = transformed.set_column(table.schema.get_field_index(column), field,
                                                 pa.array(values[column], type=field.type, safe=True))
    if not transformed.schema.equals(table.schema, check_metadata=True):
        raise ValueError("Source schema/metadata changed")
    for column in table.column_names:
        if column not in QUANTITY_COLUMNS and not table[column].equals(transformed[column]):
            raise ValueError(f"Non-quantity column changed: {column}")
    # A sequence's first event is context, not a next-event target. Do not remove it.
    first = frame.groupby("oper_part_no", sort=False)["seq"].transform("min").eq(frame["seq"]).to_numpy()
    per_split = {}
    for split in SPLITS:
        mask = frame["chronological_split"].eq(split).to_numpy()
        changed = int(np.sum(qty[mask] != capped[mask]))
        item = {
            "rows": int(mask.sum()), "series": int(frame.loc[mask, "oper_part_no"].nunique()),
            "next_event_target_count": int((mask & ~first).sum()),
            "changed_rows": changed,
            "changed_fraction": float(changed / mask.sum()) if mask.any() else 0.0,
            "at_cap_rows": int(np.sum(capped[mask] == threshold)),
        }
        # Test is transformed mechanically with the Train threshold; no Test distribution report.
        if split != "test":
            item["before"] = stats(qty[mask])
            item["after"] = stats(capped[mask])
            item["next_event_targets_before"] = stats(qty[mask & ~first])
            item["next_event_targets_after"] = stats(capped[mask & ~first])
        per_split[split] = item
    target_train = train & ~first
    report = {
        "cap_value": threshold, "quantile": .75, "quantile_method": "linear",
        "quantile_fit_population": "all_train_events_including_first_event",
        "fitted_on": "train", "test_used_for_threshold_selection": False,
        "target_train_q3_reference_only": float(np.quantile(qty[target_train], .75, method="linear")) if target_train.any() else None,
        "scale_base": base, "magnitude_rule": {"scale_base": base, "min_order": 0,
            "max_order": max_order, "fitted_on": "train", "parent_method_preserved": True},
        "splits": per_split, "rows_preserved": True, "row_order_preserved": True,
        "all_nonquantity_columns_equal": True, "schema_and_metadata_preserved": True,
        "new_columns_added": [], "quantity_columns_updated": sorted(set(table.column_names) & QUANTITY_COLUMNS),
    }
    return transformed, report


def build(source_root: Path, output_root: Path, datasets: dict | None = None) -> dict:
    datasets = DATASETS if datasets is None else datasets
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output must be outside the original data tree")
    planned = [output_root / name for name in ("manifest.json", "COMPARISON.json", "COMPARISON.md")]
    for dataset, (_, prefix, _) in datasets.items():
        folder = output_root / "data" / dataset
        planned += [folder / f"{prefix}_{suffix}.parquet" for suffix in ("with_split", *SPLITS, "train_validation")]
        planned.append(folder / f"{prefix}_split_manifest.json")
    existing = [str(path) for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    # Validate every input before creating any outputs.
    prepared = []
    for dataset, (relative, prefix, base) in datasets.items():
        original = source_root / relative / f"{prefix}_with_split.parquet"
        parent_path = source_root / relative / f"{prefix}_split_manifest.json"
        source_record, parent_record = artifact(original), artifact(parent_path)
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        original_table = pq.read_table(original)
        table, comparison = transform(original_table, base, parent)
        comparison.update(dataset_id=dataset, canonical_dataset_id=parent.get("dataset_id", parent.get("config", {}).get("dataset_name", dataset)), source=source_record, parent_manifest=parent_record)
        prepared.append((dataset, prefix, table, comparison, parent))

    output_root.mkdir(parents=True, exist_ok=True)
    results, inventories = {}, []
    created_at = datetime.now(timezone.utc).isoformat()
    for dataset, prefix, table, comparison, parent in prepared:
        folder = output_root / "data" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        split_labels = np.array(table["chronological_split"].to_pylist())
        masks = {"with_split": np.ones(len(table), dtype=bool),
                 **{split: split_labels == split for split in SPLITS},
                 "train_validation": np.isin(split_labels, ["train", "validation"])}
        artifacts = {}
        for suffix, mask in masks.items():
            path = folder / f"{prefix}_{suffix}.parquet"
            subset = table.filter(pa.array(mask))
            with path.open("xb") as stream:
                pq.write_table(subset, stream)
            restored = pq.read_table(path)
            if not restored.equals(subset, check_metadata=True):
                raise ValueError(f"Parquet round-trip mismatch: {path}")
            artifacts[suffix] = artifact(path)
            inventories.append(artifacts[suffix])
        split_manifest = {
            "schema_version": 1, "created_at": created_at, "dataset_id": dataset,
            "canonical_dataset_id": comparison["canonical_dataset_id"],
            "quantity_column": "demand_qty", "entity_column": "oper_part_no", "order_column": "seq",
            "transformation": {key: comparison[key] for key in ("cap_value", "quantile", "quantile_method", "quantile_fit_population", "fitted_on", "test_used_for_threshold_selection")},
            "magnitude_rule": comparison["magnitude_rule"],
            "parent_manifest": comparison["parent_manifest"], "source": comparison["source"],
            "artifacts": artifacts,
            "split_counts": {split: comparison["splits"][split]["rows"] for split in SPLITS},
            "next_event_target_counts": {split: comparison["splits"][split]["next_event_target_count"] for split in SPLITS},
            "split_labels_and_boundaries_preserved": True,
            "parent_split_method": parent.get("split_method", "parent chronological split labels preserved"),
            "parent_boundaries": parent.get("boundaries"),
            "held_out_test_evaluated": False, "model_training_performed": False,
        }
        manifest_path = folder / f"{prefix}_split_manifest.json"
        write_json(manifest_path, split_manifest)
        inventories.append(artifact(manifest_path))
        for record in (comparison["source"], comparison["parent_manifest"]):
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError(f"Original input changed: {record['path']}")
        comparison["source_and_parent_manifest_sha_unchanged"] = True
        comparison["artifacts"] = artifacts
        comparison["split_manifest"] = artifact(manifest_path)
        results[dataset] = comparison

    comparison_path = output_root / "COMPARISON.json"
    report = {"schema_version": 1, "created_at": created_at, "status": "complete",
              "scope": "data_transformation_only_no_training_or_performance_evaluation", "datasets": results}
    write_json(comparison_path, report)
    lines = ["# Q3 cap 데이터 비교", "", "Train 전체 이벤트에서 구한 Q3로 모든 분할의 수량을 제한했습니다. 첫 이벤트도 Q3 계산에 포함합니다.", "",
             "| 데이터셋 | Train Q3 | Train 변경/전체 | Validation 변경/전체 | Test 변경/전체 |",
             "|---|---:|---:|---:|---:|"]
    for dataset, item in results.items():
        cells = [f"{item['splits'][split]['changed_rows']:,}/{item['splits'][split]['rows']:,}" for split in SPLITS]
        lines.append(f"| {dataset} | {item['cap_value']:g} | " + " | ".join(cells) + " |")
    lines += ["", "원본·행 순서·시간 간격·분할·수량 외 컬럼·Arrow 자료형/메타데이터는 보존했습니다.",
              "Test는 고정된 Train 상한 적용 및 변경 행 수 확인만 수행했습니다. 모델 학습·예측·성능 평가는 수행하지 않았습니다.",
              "Train·Validation 분포의 전후 비교는 COMPARISON.json에 기록했습니다.",
              "첫 이벤트를 제외한 Train 예측 대상의 Q3는 참고 통계이며 상한 계산에 사용하지 않았습니다.", ""]
    markdown_path = output_root / "COMPARISON.md"
    with markdown_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    inventories += [artifact(comparison_path), artifact(markdown_path)]
    final = {"schema_version": 1, "created_at": created_at, "status": "complete", "dataset_count": len(results),
             "source_files_unchanged": True, "model_training_performed": False, "held_out_test_evaluated": False,
             "datasets": {key: {"cap_value": item["cap_value"], "source": item["source"], "parent_manifest": item["parent_manifest"]} for key, item in results.items()},
             "files": inventories, "builder": artifact(Path(__file__))}
    write_json(output_root / "manifest.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "benchmark_data/data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmark_data/Cap")
    args = parser.parse_args()
    result = build(args.source_root, args.output_root)
    print(json.dumps({"status": result["status"], "dataset_count": result["dataset_count"],
                      "manifest": str((args.output_root / "manifest.json").resolve())}))


if __name__ == "__main__":
    main()
