#!/usr/bin/env python3
"""Train-only data preparation for the frozen quantity comparison.

This module deliberately prepares no model, optimizer, loader, checkpoint, or
prediction.  It verifies the immutable input bytes and derives the two
different train-only populations needed by the 2x2 quantity comparison:
``mu`` from all train rows and the raw-loss scale from canonical loader
next-event targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame  # noqa: E402
from paper.scripts.run_count_aware_tpp_backbone_control import exact_target_population  # noqa: E402


REQUIRED_COLUMNS = {
    "oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split",
}
POPULATION_KEYS = ("target_count", "target_identity_sha256", "target_quantity_sha256")


def sha256_file(path: Path) -> str:
    """Hash an admitted input file without interpreting its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_local_input(recorded_path: str | Path, *, data_root: Path | None = None) -> Path:
    """Map a frozen remote project path to the same relative local input."""
    path = Path(recorded_path)
    if path.is_file():
        return path.resolve()
    if data_root is not None:
        root = Path(data_root).resolve()
        parts = path.parts
        if "sample_data" in parts:
            candidate = root / Path(*parts[parts.index("sample_data") + 1:])
            if candidate.is_file():
                return candidate
    try:
        relative = path.relative_to("/home/leekwanhyeong/workspace/paper_research")
    except ValueError as exc:
        raise ValueError(f"Cannot resolve input outside the frozen project: {path}") from exc
    local = PROJECT_ROOT / relative
    if not local.is_file():
        raise ValueError(f"Missing local frozen input: {local}")
    return local.resolve()


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _validate_frame(frame: pl.DataFrame, *, allowed_splits: set[str]) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    _require(not missing, f"Input schema is missing required columns: {missing}")
    observed_splits = set(frame["chronological_split"].unique().to_list())
    _require(observed_splits == allowed_splits, f"Unexpected materialized splits: {sorted(observed_splits)}")
    for column in ("demand_qty", "delta_t", "seq"):
        values = frame[column].to_numpy()
        _require(np.isfinite(values).all(), f"Nonfinite observed {column}")
    _require(not bool((frame["demand_qty"].to_numpy() < 0).any()), "Negative demand_qty is not allowed")
    _require(not bool((frame["delta_t"].to_numpy() < 0).any()), "Negative delta_t is not allowed")
    identities = frame.select(pl.struct(["oper_part_no", "seq"]).n_unique()).item()
    _require(identities == frame.height, "Duplicate series/sequence identities")
    if {"train", "validation"}.issubset(allowed_splits):
        # Train targets may use preceding observations, so a train event after a
        # validation event would make the train-only raw-scale path disagree
        # with the actual admitted runtime context.
        bad_order = frame.with_columns(
            (pl.col("chronological_split") == "validation")
            .cum_max()
            .over("oper_part_no")
            .alias("seen_validation")
        ).filter((pl.col("chronological_split") == "train") & pl.col("seen_validation"))
        _require(bad_order.height == 0, "Validation precedes a training target")


def _scan_splits(data_path: Path, splits: list[str]) -> pl.DataFrame:
    """Push a split predicate into parquet before collecting permitted rows."""
    return (
        pl.scan_parquet(data_path)
        .filter(pl.col("chronological_split").is_in(splits))
        .collect()
        .sort(["oper_part_no", "seq"])
    )


def _all_row_quantity_identity(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"quantity_comparison_all_train_row_quantity_v1\0")
    digest.update(values.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _population_view(population: Mapping[str, Any]) -> dict[str, Any]:
    return {key: population[key] for key in POPULATION_KEYS}


def _require_population_match(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, split: str,
) -> None:
    compact = _population_view(observed)
    if compact != dict(expected):
        raise ValueError(f"{split} canonical target population mismatch: {compact}")


def prepare_dataset_statistics(dataset: Mapping[str, Any], *, data_root: Path | None = None) -> dict[str, Any]:
    """Verify one comparison dataset and return only train-derived statistics.

    Validation is read only to verify its canonical target identity.  No
    validation values contribute to ``mu``, standard deviation, raw RMS, or
    raw-loss scale.
    """
    identity = dataset["inherited_data_identity"]
    data_path = resolve_local_input(identity["data"]["path"], data_root=data_root)
    manifest_path = resolve_local_input(identity["split_manifest"]["path"], data_root=data_root)
    _require(sha256_file(data_path) == identity["data"]["sha256"], "data file digest mismatch")
    _require(sha256_file(manifest_path) == identity["split_manifest"]["sha256"], "split manifest digest mismatch")

    # Validate the admitted chronology before deriving the train target scale.
    # This does not contribute any validation values to train statistics.
    admitted_raw = _scan_splits(data_path, ["train", "validation"])
    _validate_frame(admitted_raw, allowed_splits={"train", "validation"})

    # This collection is intentionally train-only: it is the complete
    # population used for B's original quantity-head initialization.
    train_raw = _scan_splits(data_path, ["train"])
    _validate_frame(train_raw, allowed_splits={"train"})
    all_train_quantities = train_raw["demand_qty"].to_numpy().astype(np.float64, copy=False)
    train_log = np.log1p(all_train_quantities)
    _require(train_log.size > 0 and np.isfinite(train_log).all(), "Train log quantities must be nonempty and finite")
    mu = float(train_log.mean())
    std = float(train_log.std())
    _require(mu > 0.0 and std > 0.0, "Train quantity initialization must be positive and nondegenerate")

    # Train targets need only train history.  Keeping this frame train-only is
    # the enforcement point for the raw-scale statistic.
    train_frame = prepare_count_frame(train_raw)
    loader = dataset["loader"]
    train_targets, train_population = exact_target_population(
        train_frame,
        target_split="train",
        lookback_weeks=int(loader["lookback_weeks"]),
        max_seq_len=int(loader["max_seq_len"]),
    )
    _require_population_match(train_population, identity["populations"]["train"], split="train")
    raw_rms = float(math.sqrt(float(np.mean(np.square(train_targets, dtype=np.float64)))))
    raw_scale = max(1.0, raw_rms)
    _require(math.isfinite(raw_rms) and math.isfinite(raw_scale), "Canonical train target RMS must be finite")

    # Validation remains locked for metrics, but its identity must be checked
    # against the frozen contract.  Parquet pushdown excludes held-out rows.
    admitted_frame = prepare_count_frame(admitted_raw)
    _, validation_population = exact_target_population(
        admitted_frame,
        target_split="validation",
        lookback_weeks=int(loader["lookback_weeks"]),
        max_seq_len=int(loader["max_seq_len"]),
    )
    _require_population_match(validation_population, identity["populations"]["validation"], split="validation")

    expected_mu = float(dataset["mu_all_train_rows"])
    _require(math.isclose(mu, expected_mu, rel_tol=0.0, abs_tol=1e-12), "All-train log mean mismatch")
    return {
        "dataset_id": str(dataset["dataset_id"]),
        "input": {
            "local_data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "local_split_manifest_path": str(manifest_path),
            "split_manifest_sha256": sha256_file(manifest_path),
        },
        "all_train_rows": {
            "count": int(all_train_quantities.size),
            "quantity_sha256": _all_row_quantity_identity(all_train_quantities),
            "log1p_mean": mu,
            "log1p_std": std,
            "statistic_scope": "all original train rows only",
        },
        "canonical_train_targets": {
            **_population_view(train_population),
            "raw_rms": raw_rms,
            "raw_scale": raw_scale,
            "statistic_scope": "canonical next-event train targets only",
        },
        "canonical_validation_targets": _population_view(validation_population),
        "held_out_materialized": False,
    }


def prepare_quantity_comparison_data(
    dataset_entry: Mapping[str, Any], *, data_root: Path | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return the admitted loader frame plus immutable execution metadata.

    ``prepared_train_validation_frame`` contains only train and validation rows,
    sorted identically to the frozen loader path.  The metadata's train
    statistics were calculated before that frame was materialized, from a
    separate train-only scan.  ``data_root`` may point at a local
    ``sample_data`` directory when the contract records a remote project path.
    """
    receipt = prepare_dataset_statistics(dataset_entry, data_root=data_root)
    data_path = Path(receipt["input"]["local_data_path"])
    admitted_raw = _scan_splits(data_path, ["train", "validation"])
    _validate_frame(admitted_raw, allowed_splits={"train", "validation"})
    metadata = {
        "dataset_id": receipt["dataset_id"],
        "input_identity": {
            "data_sha256": receipt["input"]["data_sha256"],
            "split_manifest_sha256": receipt["input"]["split_manifest_sha256"],
        },
        "resolved_inputs": {
            "local_data_path": receipt["input"]["local_data_path"],
            "local_split_manifest_path": receipt["input"]["local_split_manifest_path"],
        },
        "all_train_rows": receipt["all_train_rows"],
        "train_log_mean": receipt["all_train_rows"]["log1p_mean"],
        "train_log_std": receipt["all_train_rows"]["log1p_std"],
        "raw_scale": receipt["canonical_train_targets"]["raw_scale"],
        "raw_rms": receipt["canonical_train_targets"]["raw_rms"],
        "populations": {
            "train": {key: receipt["canonical_train_targets"][key] for key in POPULATION_KEYS},
            "validation": receipt["canonical_validation_targets"],
        },
        "held_out_materialized": False,
    }
    return prepare_count_frame(admitted_raw), metadata


def prepare_comparison_contract(contract_path: Path) -> dict[str, Any]:
    """Prepare every dataset in a frozen comparison design, without training."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(contract.get("schema") == "quantity_objective_output_comparison_v1", "Unexpected comparison contract schema")
    return {
        "schema": "quantity_comparison_preparation_v1",
        "comparison_contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
        "status": "prepared_train_only_statistics_and_identities",
        "held_out_materialized": False,
        "datasets": [prepare_dataset_statistics(dataset) for dataset in contract["datasets"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / "paper/contracts/quantity_objective_output_comparison_v1.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "search_artifacts/quantity_comparison_preparation_v1/train_statistics.json")
    args = parser.parse_args()
    result = prepare_comparison_contract(args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for dataset in result["datasets"]:
        train = dataset["all_train_rows"]
        targets = dataset["canonical_train_targets"]
        print(
            f"{dataset['dataset_id']} mu={train['log1p_mean']:.15g} std={train['log1p_std']:.15g} "
            f"raw_scale={targets['raw_scale']:.15g} target_count={targets['target_count']} "
            f"target_sha256={targets['target_quantity_sha256']}"
        )


if __name__ == "__main__":
    main()
