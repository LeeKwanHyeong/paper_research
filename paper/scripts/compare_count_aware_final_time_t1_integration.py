#!/usr/bin/env python3
"""Compare H3 log-normal time against H0 under matched Hard-LMM plus T1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from models.TPPs.CountAwareTPP import (
    TAIL_SHARED_VARIANT,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
    TIME_HEAD_MODE_SCALED_EXACT,
)
from paper.scripts.count_aware_tpp_backbone.reporting import write_csv
from paper.scripts.run_intermittent_log_backbone_control import (
    EXPECTED_DATA_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
)
from paper.scripts.run_taxi_quantity_interface_ablation import save_json


BACKBONE = "titantpp"
SEED = 42
BODY_STRATA = {"le_p50", "p50_p90", "p90_p95"}
TAIL_STRATUM = "gt_p99"
SHARED_FIELDS = (
    "dataset",
    "data_sha256",
    "split_manifest_sha256",
    "epochs",
    "batch_size",
    "lr",
    "lookback_weeks",
    "max_seq_len",
    "hidden_dim",
    "lambda_log_qty",
    "lambda_tail",
    "tail_contract",
    "grad_clip",
    "early_stopping",
    "evaluation_scope",
    "held_out_test_evaluated",
    "source_revision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def exact_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["backbone"] == BACKBONE
        and row["variant"] == TAIL_SHARED_VARIANT
        and int(row["seed"]) == SEED
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one matched summary row, found {len(matches)}")
    return matches[0]


def exact_quantity_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["backbone"] == BACKBONE
        and row["variant"] == TAIL_SHARED_VARIANT
        and int(row["seed"]) == SEED
    ]
    if {row["stratum"] for row in selected} != {
        "le_p50",
        "p50_p90",
        "p90_p95",
        "p95_p99",
        "gt_p99",
    }:
        raise ValueError("Incomplete quantity strata for the integrated comparison")
    return selected


def weighted_mae(rows: list[dict[str, str]], strata: set[str]) -> float:
    selected = [row for row in rows if row["stratum"] in strata]
    if {row["stratum"] for row in selected} != strata:
        raise ValueError(f"Missing strata for weighted MAE: {strata}")
    count = sum(int(row["count"]) for row in selected)
    if count < 1:
        raise ValueError("Cannot aggregate empty quantity strata")
    return sum(float(row["qty_mae"]) * int(row["count"]) for row in selected) / count


def metric_record(
    summary: dict[str, str],
    quantity_rows: list[dict[str, str]],
) -> dict[str, float | int]:
    tail = next(row for row in quantity_rows if row["stratum"] == TAIL_STRATUM)
    return {
        "joint_objective": float(summary["best_val_joint_objective"]),
        "time_nll": float(summary["best_val_time_nll"]),
        "quantity_train_loss": float(summary["best_val_quantity_train_loss"]),
        "qty_mae": float(summary["best_val_qty_mae"]),
        "qty_rmse": float(summary["best_val_qty_rmse"]),
        "le_p95_qty_mae": weighted_mae(quantity_rows, BODY_STRATA),
        "gt_p99_qty_mae": float(tail["qty_mae"]),
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "parameter_count": int(summary["parameter_count"]),
    }


def percent_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("Percentage comparison requires a nonzero reference")
    return 100.0 * (candidate - reference) / reference


def evaluate_gate(
    reference: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, Any]:
    tolerance = 1e-12
    deltas = {
        "time_nll_absolute_regression": (
            float(candidate["time_nll"]) - float(reference["time_nll"])
        ),
        "qty_mae_regression_pct": percent_change(
            float(candidate["qty_mae"]), float(reference["qty_mae"])
        ),
        "qty_rmse_regression_pct": percent_change(
            float(candidate["qty_rmse"]), float(reference["qty_rmse"])
        ),
        "le_p95_qty_mae_regression_pct": percent_change(
            float(candidate["le_p95_qty_mae"]),
            float(reference["le_p95_qty_mae"]),
        ),
        "gt_p99_qty_mae_regression_pct": percent_change(
            float(candidate["gt_p99_qty_mae"]),
            float(reference["gt_p99_qty_mae"]),
        ),
    }
    checks = {
        "finite_contract": all(
            math.isfinite(float(value))
            for value in (*reference.values(), *candidate.values())
        ),
        "time_nll_regression_at_most_0_01": (
            deltas["time_nll_absolute_regression"] <= 0.01 + tolerance
        ),
        "qty_mae_regression_at_most_2pct": (
            deltas["qty_mae_regression_pct"] <= 2.0 + tolerance
        ),
        "qty_rmse_regression_at_most_2pct": (
            deltas["qty_rmse_regression_pct"] <= 2.0 + tolerance
        ),
        "le_p95_qty_mae_regression_at_most_2pct": (
            deltas["le_p95_qty_mae_regression_pct"] <= 2.0 + tolerance
        ),
        "gt_p99_qty_mae_regression_at_most_2pct": (
            deltas["gt_p99_qty_mae_regression_pct"] <= 2.0 + tolerance
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "deltas": deltas,
        "thresholds": {
            "max_time_nll_absolute_regression": 0.01,
            "max_qty_mae_regression_pct": 2.0,
            "max_qty_rmse_regression_pct": 2.0,
            "max_le_p95_qty_mae_regression_pct": 2.0,
            "max_gt_p99_qty_mae_regression_pct": 2.0,
        },
    }


def validate_launch_contracts(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    mismatches: dict[str, Any] = {}
    expected_common = {
        "status": "complete",
        "backbones": [BACKBONE],
        "seeds": [SEED],
        "quantity_variants": [TAIL_SHARED_VARIANT],
        "expected_run_count": 1,
        "completed_run_count": 1,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "data_sha256": EXPECTED_DATA_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
    }
    for label, launch in (("H0", reference), ("H3", candidate)):
        for key, expected in expected_common.items():
            if launch.get(key) != expected:
                mismatches[f"{label}.{key}"] = {
                    "expected": expected,
                    "observed": launch.get(key),
                }
    for key in SHARED_FIELDS:
        if reference.get(key) != candidate.get(key):
            mismatches[f"matched.{key}"] = {
                "expected": reference.get(key),
                "observed": candidate.get(key),
            }

    h0_time = reference.get("time_head", {})
    h3_time = candidate.get("time_head", {})
    expected_h0 = {
        "mode": TIME_HEAD_MODE_SCALED_EXACT,
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_wd_safety_limit": 40.0,
        "density_unit": "original_delta_t_with_jacobian",
        "wd_clamp": 0.0,
    }
    expected_h3 = {
        "mode": TIME_HEAD_MODE_LOGNORMAL_DURATION,
        "time_scale": 3.0,
        "time_sigma_floor": 1e-3,
        "density_unit": "original_delta_t_with_jacobian",
        "wd_clamp": 0.0,
    }
    for label, observed, expected in (
        ("H0.time_head", h0_time, expected_h0),
        ("H3.time_head", h3_time, expected_h3),
    ):
        for key, value in expected.items():
            actual = observed.get(key)
            matches = (
                math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(value, float) and actual is not None
                else actual == value
            )
            if not matches:
                mismatches[f"{label}.{key}"] = {
                    "expected": value,
                    "observed": actual,
                }

    train_stats = h3_time.get("train_time_statistics", {})
    for field, statistic in (
        ("time_initial_location", "target_log_scaled_mean"),
        ("time_initial_scale", "target_log_scaled_std"),
    ):
        observed = h3_time.get(field)
        expected = train_stats.get(statistic)
        if observed is None or expected is None or not math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            mismatches[f"H3.train_only.{field}"] = {
                "expected": expected,
                "observed": observed,
            }
    if mismatches:
        raise ValueError(f"Integrated launch contract mismatch: {mismatches}")


def markdown_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    gate = payload["safety_gate"]
    lines = [
        "# Hard-LMM + T1 Final Time-Head Integration",
        "",
        f"- Status: `{gate['status']}`",
        f"- Selected head: `{payload['selected_time_head']}`",
        "- Evaluation: validation only; held-out test not used",
        "",
        "| Head | Time NLL | Quantity MAE | Quantity RMSE | <=p95 MAE | >p99 MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for head in ("H0", "H3"):
        row = metrics[head]
        lines.append(
            f"| {head} | {row['time_nll']:.6f} | {row['qty_mae']:.6f} | "
            f"{row['qty_rmse']:.6f} | {row['le_p95_qty_mae']:.6f} | "
            f"{row['gt_p99_qty_mae']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    reference_launch = read_json(args.reference_artifact / "launch_contract.json")
    candidate_launch = read_json(args.candidate_artifact / "launch_contract.json")
    validate_launch_contracts(reference_launch, candidate_launch)

    metrics: dict[str, dict[str, float | int]] = {}
    for label, artifact in (
        ("H0", args.reference_artifact),
        ("H3", args.candidate_artifact),
    ):
        summary = exact_summary(read_csv(artifact / "run_summaries.csv"))
        quantity_rows = exact_quantity_rows(
            read_csv(artifact / "quantity_seed_metrics.csv")
        )
        metrics[label] = metric_record(summary, quantity_rows)
    gate = evaluate_gate(metrics["H0"], metrics["H3"])
    payload = {
        "schema_version": 1,
        "status": gate["status"],
        "reference_time_head": "H0_scaled_exact",
        "candidate_time_head": "H3_lognormal_duration",
        "selected_time_head": (
            "H3_lognormal_duration"
            if gate["status"] == "pass"
            else "H0_scaled_exact"
        ),
        "backbone": "TitanTPP_Hard_LMM",
        "quantity_objective": "T1_tail_shared",
        "seed": SEED,
        "metrics": metrics,
        "safety_gate": gate,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "selection_rule": (
            "H3 passes when Time NLL regression is <=0.01 and overall MAE, "
            "RMSE, <=p95 MAE, and >p99 MAE regressions are each <=2%."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "acceptance.json", payload)
    write_csv(
        args.output_dir / "metrics.csv",
        [{"time_head": label, **record} for label, record in metrics.items()],
    )
    (args.output_dir / "acceptance.md").write_text(
        markdown_report(payload),
        encoding="utf-8",
    )
    print(
        f"[complete] status={gate['status']} "
        f"selected={payload['selected_time_head']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
