#!/usr/bin/env python3
"""Validate the frozen B0/B1/B2 CUDA and four-dataset e1 preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLE_TITAN_B012_SCREENING,
    TITAN_B012_BACKBONES,
    VARIANT,
)
from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS


SMOKE_DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "raf_spare_parts",
    "insta_market_basket",
)
SUMMARY_FINITE_FIELDS = (
    "best_val_joint_objective",
    "best_val_time_nll",
    "best_val_quantity_train_loss",
    "best_val_log_qty_mse",
    "best_val_qty_mae",
    "best_val_qty_rmse",
    "elapsed_seconds",
)
BREAKDOWN_FINITE_FIELDS = (
    "joint_objective",
    "time_nll",
    "quantity_train_loss",
    "log_qty_mse",
    "qty_mae",
    "qty_rmse",
    "qty_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_finite(row: dict[str, str], fields: tuple[str, ...], *, label: str) -> None:
    for field in fields:
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {label}.{field}: {value}")


def validate_dataset(
    artifact_root: Path,
    dataset_id: str,
    *,
    source_revision: str,
) -> dict[str, Any]:
    run_dir = artifact_root / dataset_id / MODEL_ROLE_TITAN_B012_SCREENING
    launch = load_json(run_dir / "launch_contract.json")
    expected_dataset = DATASET_CONTRACTS[dataset_id]
    expected_contract = {
        "status": "complete",
        "model_role": MODEL_ROLE_TITAN_B012_SCREENING,
        "dataset": dataset_id,
        "data_sha256": expected_dataset["data_sha256"],
        "split_manifest_sha256": expected_dataset["split_manifest_sha256"],
        "quantity_variants": [VARIANT],
        "backbones": list(TITAN_B012_BACKBONES),
        "seeds": [42],
        "expected_run_count": 3,
        "completed_run_count": 3,
        "epochs": 1,
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": expected_dataset["lookback"],
        "max_seq_len": expected_dataset["max_seq_len"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": source_revision,
        "partial_smoke": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": launch.get(key)}
        for key, expected in expected_contract.items()
        if launch.get(key) != expected
    }
    if launch.get("time_head", {}).get("mode") != "legacy_clamped_rmtpp":
        mismatches["time_head.mode"] = {
            "expected": "legacy_clamped_rmtpp",
            "observed": launch.get("time_head", {}).get("mode"),
        }
    expected_early_stopping = {
        "min_epochs": 1,
        "patience": 1,
        "restore": "best_validation_joint_objective",
    }
    for key, expected in expected_early_stopping.items():
        observed = launch.get("early_stopping", {}).get(key)
        if observed != expected:
            mismatches[f"early_stopping.{key}"] = {
                "expected": expected,
                "observed": observed,
            }
    if mismatches:
        raise ValueError(f"{dataset_id} launch contract mismatch: {mismatches}")

    summaries = load_csv(run_dir / "run_summaries.csv")
    if len(summaries) != 3:
        raise ValueError(f"{dataset_id} expected three summaries, got {len(summaries)}")
    if tuple(row["backbone"] for row in summaries) != TITAN_B012_BACKBONES:
        raise ValueError(f"{dataset_id} backbone order drifted")
    elapsed = {}
    for row in summaries:
        backbone = row["backbone"]
        if row["status"] != "success" or int(row["seed"]) != 42:
            raise ValueError(f"{dataset_id}/{backbone} did not complete successfully")
        if row["variant"] != VARIANT:
            raise ValueError(f"{dataset_id}/{backbone} variant drifted")
        if row["source_revision"] != source_revision:
            raise ValueError(f"{dataset_id}/{backbone} source revision drifted")
        if row["evaluation_scope"] != "validation_only":
            raise ValueError(f"{dataset_id}/{backbone} is not validation-only")
        if row["held_out_test_evaluated"].lower() != "false":
            raise ValueError(f"{dataset_id}/{backbone} used held-out test")
        assert_finite(row, SUMMARY_FINITE_FIELDS, label=f"{dataset_id}/{backbone}")
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        elapsed[backbone] = float(row["elapsed_seconds"])

    for filename in ("quantity_seed_metrics.csv", "history_seed_metrics.csv"):
        rows = load_csv(run_dir / filename)
        if not rows:
            raise ValueError(f"{dataset_id}/{filename} is empty")
        for row in rows:
            assert_finite(
                row,
                BREAKDOWN_FINITE_FIELDS,
                label=f"{dataset_id}/{filename}/{row['backbone']}/{row['stratum']}",
            )

    b0_elapsed = elapsed["titantpp"]
    elapsed_ratios = {
        backbone: value / b0_elapsed for backbone, value in elapsed.items()
    }
    return {
        "dataset": dataset_id,
        "run_count": len(summaries),
        "all_metrics_finite": True,
        "held_out_test_evaluated": False,
        "elapsed_seconds": elapsed,
        "elapsed_ratio_vs_b0": elapsed_ratios,
    }


def main() -> None:
    args = parse_args()
    if len(args.source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_revision
    ):
        raise ValueError("source-revision must be a 40-character lowercase Git SHA")
    cuda_test = load_json(args.artifact_root / "cuda_model_test.json")
    if cuda_test.get("status") != "complete" or not cuda_test.get(
        "speed_gate_passed"
    ):
        raise ValueError("CUDA model test or B0-relative speed gate did not pass")
    if cuda_test.get("backbones") != list(TITAN_B012_BACKBONES):
        raise ValueError("CUDA model test backbone contract drifted")

    forbidden = [
        path
        for path in args.artifact_root.rglob("*")
        if path.is_file()
        and path.name in {"test_summary.json", "test_metrics.csv", "held_out_test.json"}
    ]
    if forbidden:
        raise ValueError(f"Held-out test artifacts are forbidden: {forbidden}")

    dataset_results = [
        validate_dataset(
            args.artifact_root,
            dataset_id,
            source_revision=args.source_revision,
        )
        for dataset_id in SMOKE_DATASETS
    ]
    payload = {
        "status": "complete",
        "contract_id": "count_aware_titan_b012_screening_v1",
        "source_revision": args.source_revision,
        "execution_server": "5080",
        "cuda_model_test_passed": True,
        "training_step_speed_gate_passed": True,
        "dataset_results": dataset_results,
        "held_out_test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
