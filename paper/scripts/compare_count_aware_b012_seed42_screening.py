#!/usr/bin/env python3
"""Audit and compare B0/B1/B2 seed-42 matched validation screening."""

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


SCREENING_DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "raf_spare_parts",
)
BODY_STRATA = ("le_p50", "p50_p90", "p90_p95")
EXTREME_TAIL_STRATUM = "gt_p99"
PRIMARY_MINIMUM_IMPROVEMENT = 0.05
RMSE_MAXIMUM_REGRESSION = 0.02
P99_MAE_MAXIMUM_REGRESSION = 0.02
TIME_NLL_MAXIMUM_REGRESSION = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def relative_regression(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError(f"Relative regression requires positive baseline, got {baseline}")
    return (candidate - baseline) / baseline


def relative_improvement(candidate: float, baseline: float) -> float:
    return -relative_regression(candidate, baseline)


def weighted_body_mae(rows: list[dict[str, str]]) -> tuple[float, int]:
    selected = [row for row in rows if row["stratum"] in BODY_STRATA]
    observed = {row["stratum"] for row in selected}
    if observed != set(BODY_STRATA):
        raise ValueError(f"Missing <=p95 strata: {sorted(set(BODY_STRATA) - observed)}")
    count = sum(int(row["count"]) for row in selected)
    if count <= 0:
        raise ValueError("<=p95 validation event count must be positive")
    absolute_error_sum = sum(
        int(row["count"]) * float(row["qty_mae"]) for row in selected
    )
    value = absolute_error_sum / count
    if not math.isfinite(value):
        raise ValueError("<=p95 MAE must be finite")
    return value, count


def compute_backbone_metrics(
    summary: dict[str, str],
    quantity_rows: list[dict[str, str]],
) -> dict[str, float | int]:
    body_mae, body_count = weighted_body_mae(quantity_rows)
    tail_rows = [row for row in quantity_rows if row["stratum"] == EXTREME_TAIL_STRATUM]
    if len(tail_rows) != 1:
        raise ValueError(f"Expected one >p99 row, got {len(tail_rows)}")
    metrics: dict[str, float | int] = {
        "body_le_p95_mae": body_mae,
        "body_le_p95_count": body_count,
        "overall_qty_mae": float(summary["best_val_qty_mae"]),
        "overall_qty_rmse": float(summary["best_val_qty_rmse"]),
        "gt_p99_mae": float(tail_rows[0]["qty_mae"]),
        "gt_p99_count": int(tail_rows[0]["count"]),
        "time_nll": float(summary["best_val_time_nll"]),
        "joint_objective": float(summary["best_val_joint_objective"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "best_epoch": int(summary["best_epoch"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
    }
    if not all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if not key.endswith("_count") and key not in {"completed_epochs", "best_epoch"}
    ):
        raise ValueError("Backbone metrics contain non-finite values")
    return metrics


def evaluate_b2_gate(
    baseline: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, Any]:
    body_improvement = relative_improvement(
        float(candidate["body_le_p95_mae"]),
        float(baseline["body_le_p95_mae"]),
    )
    rmse_regression = relative_regression(
        float(candidate["overall_qty_rmse"]),
        float(baseline["overall_qty_rmse"]),
    )
    p99_mae_regression = relative_regression(
        float(candidate["gt_p99_mae"]),
        float(baseline["gt_p99_mae"]),
    )
    time_nll_regression = float(candidate["time_nll"]) - float(
        baseline["time_nll"]
    )
    gates = {
        "body_le_p95_mae_improvement_at_least_5pct": (
            body_improvement >= PRIMARY_MINIMUM_IMPROVEMENT
        ),
        "overall_rmse_regression_at_most_2pct": (
            rmse_regression <= RMSE_MAXIMUM_REGRESSION
        ),
        "gt_p99_mae_regression_at_most_2pct": (
            p99_mae_regression <= P99_MAE_MAXIMUM_REGRESSION
        ),
        "time_nll_regression_at_most_0_01": (
            time_nll_regression <= TIME_NLL_MAXIMUM_REGRESSION
        ),
        "all_values_finite": all(
            math.isfinite(float(value))
            for value in (
                body_improvement,
                rmse_regression,
                p99_mae_regression,
                time_nll_regression,
            )
        ),
    }
    return {
        "body_le_p95_mae_relative_improvement": body_improvement,
        "overall_rmse_relative_regression": rmse_regression,
        "gt_p99_mae_relative_regression": p99_mae_regression,
        "time_nll_absolute_regression": time_nll_regression,
        "gates": gates,
        "passed": all(gates.values()),
    }


def assert_numeric_csv_finite(path: Path) -> None:
    rows = load_csv(path)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    for row_index, row in enumerate(rows, start=2):
        for key, raw in row.items():
            if raw is None or raw == "":
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {path}:{row_index}:{key}={raw}")


def validate_encoder_contract(backbone: str, summary: dict[str, Any]) -> None:
    config = summary["encoder_config"]
    expected_contract_id = {
        "titantpp": "B0",
        "titantpp_titans_mac": "B1",
        "titantpp_tpp_gated_memory": "B2",
    }[backbone]
    if config.get("backbone_contract_id") != expected_contract_id:
        raise ValueError(f"{backbone} backbone contract id drifted")
    if backbone == "titantpp_titans_mac":
        expected = {
            "persistent_mem_size": 16,
            "titans_neural_memory_depth": 2,
            "titans_neural_memory_hidden_expansion": 2,
            "titans_mac_segment_size": 16,
            "titans_scan_backend": "compiled_sequence_cuda",
            "titans_online_update": "surprise_momentum_adaptive_forgetting",
        }
    elif backbone == "titantpp_tpp_gated_memory":
        expected = {
            "persistent_mem_size": 16,
            "tpp_gated_memory_size": 64,
            "tpp_gated_topk": 4,
            "tpp_gated_temperature": 1.0,
            "tpp_gated_state_scope": "explicit_per_series_state",
            "tpp_gated_scan_backend": "compiled_sequence_cuda",
        }
    else:
        expected = {
            "memory_mode": "static_hard_lmm",
            "persistent_mem_size": 16,
            "lmm_mem_size": 64,
            "lmm_topk": 4,
        }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{backbone} encoder contract mismatch: {mismatches}")


def validate_dataset(
    artifact_root: Path,
    dataset_id: str,
    *,
    source_revision: str,
) -> dict[str, Any]:
    run_dir = artifact_root / dataset_id / MODEL_ROLE_TITAN_B012_SCREENING
    launch = load_json(run_dir / "launch_contract.json")
    dataset_contract = DATASET_CONTRACTS[dataset_id]
    expected = {
        "status": "complete",
        "model_role": MODEL_ROLE_TITAN_B012_SCREENING,
        "dataset": dataset_id,
        "data_sha256": dataset_contract["data_sha256"],
        "split_manifest_sha256": dataset_contract["split_manifest_sha256"],
        "quantity_variants": [VARIANT],
        "backbones": list(TITAN_B012_BACKBONES),
        "seeds": [42],
        "expected_run_count": 3,
        "completed_run_count": 3,
        "epochs": 300,
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": dataset_contract["lookback"],
        "max_seq_len": dataset_contract["max_seq_len"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": source_revision,
        "partial_smoke": False,
        "max_series": None,
    }
    mismatches = {
        key: {"expected": value, "observed": launch.get(key)}
        for key, value in expected.items()
        if launch.get(key) != value
    }
    if launch.get("time_head", {}).get("mode") != "legacy_clamped_rmtpp":
        mismatches["time_head.mode"] = {
            "expected": "legacy_clamped_rmtpp",
            "observed": launch.get("time_head", {}).get("mode"),
        }
    for key, value in {
        "min_epochs": 40,
        "patience": 40,
        "restore": "best_validation_joint_objective",
    }.items():
        observed = launch.get("early_stopping", {}).get(key)
        if observed != value:
            mismatches[f"early_stopping.{key}"] = {
                "expected": value,
                "observed": observed,
            }
    if mismatches:
        raise ValueError(f"{dataset_id} launch contract mismatch: {mismatches}")

    summaries = load_csv(run_dir / "run_summaries.csv")
    if len(summaries) != 3:
        raise ValueError(f"{dataset_id} expected three summaries")
    if tuple(row["backbone"] for row in summaries) != TITAN_B012_BACKBONES:
        raise ValueError(f"{dataset_id} backbone order drifted")
    quantity_rows = load_csv(run_dir / "quantity_seed_metrics.csv")
    history_rows_path = run_dir / "history_seed_metrics.csv"
    assert_numeric_csv_finite(run_dir / "run_summaries.csv")
    assert_numeric_csv_finite(run_dir / "quantity_seed_metrics.csv")
    assert_numeric_csv_finite(history_rows_path)

    metrics_by_backbone: dict[str, dict[str, float | int]] = {}
    for summary_row in summaries:
        backbone = summary_row["backbone"]
        if summary_row["status"] != "success" or int(summary_row["seed"]) != 42:
            raise ValueError(f"{dataset_id}/{backbone} is not a successful seed-42 run")
        if summary_row["variant"] != VARIANT:
            raise ValueError(f"{dataset_id}/{backbone} quantity variant drifted")
        if summary_row["source_revision"] != source_revision:
            raise ValueError(f"{dataset_id}/{backbone} source revision drifted")
        if summary_row["evaluation_scope"] != "validation_only":
            raise ValueError(f"{dataset_id}/{backbone} is not validation-only")
        if summary_row["held_out_test_evaluated"].lower() != "false":
            raise ValueError(f"{dataset_id}/{backbone} used held-out test")
        completed_epochs = int(summary_row["completed_epochs"])
        if not 40 <= completed_epochs <= 300:
            raise ValueError(f"{dataset_id}/{backbone} epoch contract failed")
        checkpoint = Path(summary_row["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        run_summary_path = (
            run_dir / "runs" / backbone / VARIANT / "seed_42" / "summary.json"
        )
        run_summary = load_json(run_summary_path)
        validate_encoder_contract(backbone, run_summary)
        history_path = run_summary_path.parent / "history.csv"
        assert_numeric_csv_finite(history_path)
        backbone_rows = [
            row
            for row in quantity_rows
            if row["backbone"] == backbone
            and row["variant"] == VARIANT
            and int(row["seed"]) == 42
        ]
        metrics_by_backbone[backbone] = compute_backbone_metrics(
            summary_row,
            backbone_rows,
        )

    b0 = metrics_by_backbone["titantpp"]
    b1 = metrics_by_backbone["titantpp_titans_mac"]
    b2 = metrics_by_backbone["titantpp_tpp_gated_memory"]
    return {
        "dataset": dataset_id,
        "metrics": metrics_by_backbone,
        "b1_deltas_vs_b0": {
            "body_le_p95_mae_relative_improvement": relative_improvement(
                float(b1["body_le_p95_mae"]), float(b0["body_le_p95_mae"])
            ),
            "overall_rmse_relative_regression": relative_regression(
                float(b1["overall_qty_rmse"]), float(b0["overall_qty_rmse"])
            ),
            "gt_p99_mae_relative_regression": relative_regression(
                float(b1["gt_p99_mae"]), float(b0["gt_p99_mae"])
            ),
            "time_nll_absolute_regression": float(b1["time_nll"])
            - float(b0["time_nll"]),
            "selection_status": "reference_only_not_selectable",
        },
        "b2_gate": evaluate_b2_gate(b0, b2),
        "held_out_test_evaluated": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Count-aware Titan B0/B1/B2 Seed-42 Screening",
        "",
        f"- Decision: `{payload['selected_variant']}`",
        f"- B2 accepted: `{str(payload['b2_accepted']).lower()}`",
        "- Held-out test evaluated: `false`",
        "",
        "| Dataset | Backbone | <=p95 MAE | Overall RMSE | >p99 MAE | Time NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_result in payload["datasets"]:
        for backbone in TITAN_B012_BACKBONES:
            metrics = dataset_result["metrics"][backbone]
            lines.append(
                f"| {dataset_result['dataset']} | {backbone} | "
                f"{float(metrics['body_le_p95_mae']):.6f} | "
                f"{float(metrics['overall_qty_rmse']):.6f} | "
                f"{float(metrics['gt_p99_mae']):.6f} | "
                f"{float(metrics['time_nll']):.6f} |"
            )
    lines.extend(["", "## B2 Dataset Gates", ""])
    for dataset_result in payload["datasets"]:
        gate = dataset_result["b2_gate"]
        lines.append(
            f"- `{dataset_result['dataset']}`: "
            f"{'PASS' if gate['passed'] else 'FAIL'}; "
            f"body improvement={gate['body_le_p95_mae_relative_improvement']:.2%}, "
            f"RMSE regression={gate['overall_rmse_relative_regression']:.2%}, "
            f">p99 MAE regression={gate['gt_p99_mae_relative_regression']:.2%}, "
            f"Time NLL delta={gate['time_nll_absolute_regression']:.6f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if len(args.source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_revision
    ):
        raise ValueError("source-revision must be a 40-character lowercase Git SHA")
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
        for dataset_id in SCREENING_DATASETS
    ]
    b2_accepted = all(result["b2_gate"]["passed"] for result in dataset_results)
    payload = {
        "status": "complete",
        "contract_id": "count_aware_titan_b012_screening_v1",
        "source_revision": args.source_revision,
        "execution_server": "5080",
        "datasets": dataset_results,
        "b1_selection_status": "reference_only_not_selectable",
        "b2_accepted": b2_accepted,
        "selected_variant": (
            "B2_tpp_specific_gated_memory"
            if b2_accepted
            else "B0_current_hard_lmm_control"
        ),
        "held_out_test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decision.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for dataset_result in dataset_results:
        for backbone in TITAN_B012_BACKBONES:
            rows.append(
                {
                    "dataset": dataset_result["dataset"],
                    "backbone": backbone,
                    **dataset_result["metrics"][backbone],
                }
            )
    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "comparison.md").write_text(
        markdown_report(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
