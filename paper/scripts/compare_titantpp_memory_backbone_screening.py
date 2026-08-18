#!/usr/bin/env python3
"""Apply the frozen TitanTPP memory-backbone validation gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.count_aware_tpp_backbone.constants import VARIANT
from paper.scripts.run_intermittent_log_backbone_control import (
    EXPECTED_DATA_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
)


CONTROL_BACKBONE = "titantpp"
CANDIDATE_BACKBONES = (
    "titantpp_no_memory",
    "titantpp_gated_soft_memory",
    "titantpp_surprise_memory",
)
EXPECTED_BACKBONES = (CONTROL_BACKBONE, *CANDIDATE_BACKBONES)
BODY_STRATA = {"le_p50", "p50_p90", "p90_p95"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percent_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("Percentage comparison requires a nonzero reference")
    return 100.0 * (candidate - reference) / reference


def exact_row(
    rows: list[dict[str, str]],
    *,
    backbone: str,
    stratum: str | None = None,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["backbone"] == backbone
        and row["variant"] == VARIANT
        and int(row["seed"]) == 42
        and (stratum is None or row.get("stratum") == stratum)
    ]
    if len(matches) != 1:
        suffix = f"/{stratum}" if stratum else ""
        raise ValueError(
            f"Expected one {backbone}/{VARIANT}{suffix} row, found {len(matches)}"
        )
    return matches[0]


def weighted_mae(
    rows: list[dict[str, str]],
    *,
    backbone: str,
    strata: set[str],
) -> float:
    selected = [
        row
        for row in rows
        if row["backbone"] == backbone
        and row["variant"] == VARIANT
        and int(row["seed"]) == 42
        and row["stratum"] in strata
    ]
    if {row["stratum"] for row in selected} != strata:
        raise ValueError(f"Incomplete strata for {backbone}: {selected}")
    total = sum(int(row["count"]) for row in selected)
    if total < 1:
        raise ValueError(f"Empty strata for {backbone}")
    return sum(float(row["qty_mae"]) * int(row["count"]) for row in selected) / total


def metric_record(
    summary: dict[str, str],
    quantity_rows: list[dict[str, str]],
    *,
    backbone: str,
) -> dict[str, float | int]:
    return {
        "joint_objective": float(summary["best_val_joint_objective"]),
        "qty_mae": float(summary["best_val_qty_mae"]),
        "qty_rmse": float(summary["best_val_qty_rmse"]),
        "time_nll": float(summary["best_val_time_nll"]),
        "le_p95_qty_mae": weighted_mae(
            quantity_rows,
            backbone=backbone,
            strata=BODY_STRATA,
        ),
        "parameter_count": int(summary["parameter_count"]),
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
    }


def evaluate_gate(
    control: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, Any]:
    tolerance = 1e-12
    mae_improvement = -percent_change(
        float(candidate["qty_mae"]),
        float(control["qty_mae"]),
    )
    rmse_improvement = -percent_change(
        float(candidate["qty_rmse"]),
        float(control["qty_rmse"]),
    )
    body_mae_regression = percent_change(
        float(candidate["le_p95_qty_mae"]),
        float(control["le_p95_qty_mae"]),
    )
    time_nll_regression = float(candidate["time_nll"]) - float(control["time_nll"])
    finite = all(
        math.isfinite(float(value))
        for value in (*control.values(), *candidate.values())
    )
    checks = {
        "finite_contract": finite,
        "mae_or_rmse_improvement_at_least_5pct": (
            mae_improvement + tolerance >= 5.0
            or rmse_improvement + tolerance >= 5.0
        ),
        "le_p95_mae_regression_at_most_2pct": (
            body_mae_regression <= 2.0 + tolerance
        ),
        "time_nll_regression_at_most_0_01": (
            time_nll_regression <= 0.01 + tolerance
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "deltas": {
            "overall_mae_improvement_pct": mae_improvement,
            "overall_rmse_improvement_pct": rmse_improvement,
            "le_p95_mae_regression_pct": body_mae_regression,
            "time_nll_absolute_regression": time_nll_regression,
        },
    }


def select_backbone(
    metrics: dict[str, dict[str, float | int]],
    gates: dict[str, dict[str, Any]],
) -> str:
    passing = [
        backbone
        for backbone in CANDIDATE_BACKBONES
        if gates[backbone]["status"] == "pass"
    ]
    if not passing:
        return CONTROL_BACKBONE
    return min(
        passing,
        key=lambda backbone: (
            float(metrics[backbone]["qty_mae"]),
            float(metrics[backbone]["qty_rmse"]),
            float(metrics[backbone]["joint_objective"]),
            CANDIDATE_BACKBONES.index(backbone),
        ),
    )


def validate_launch_contract(launch: dict[str, Any]) -> None:
    expected = {
        "status": "complete",
        "dataset": "intermittent_frozen_5000",
        "backbones": list(EXPECTED_BACKBONES),
        "seeds": [42],
        "quantity_variants": [VARIANT],
        "expected_run_count": 4,
        "completed_run_count": 4,
        "epochs": 300,
        "batch_size": 128,
        "lookback_weeks": 520,
        "max_seq_len": 256,
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "data_sha256": EXPECTED_DATA_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
    }
    mismatches = {
        key: {"expected": value, "observed": launch.get(key)}
        for key, value in expected.items()
        if launch.get(key) != value
    }
    if not math.isclose(float(launch.get("lr", math.nan)), 1e-3):
        mismatches["lr"] = {"expected": 1e-3, "observed": launch.get("lr")}
    early_stopping = launch.get("early_stopping", {})
    expected_early_stopping = {
        "min_epochs": 40,
        "patience": 40,
        "restore": "best_validation_joint_objective",
    }
    for key, value in expected_early_stopping.items():
        if early_stopping.get(key) != value:
            mismatches[f"early_stopping.{key}"] = {
                "expected": value,
                "observed": early_stopping.get(key),
            }
    if mismatches:
        raise ValueError(f"Screening contract mismatch: {mismatches}")


def write_metrics_csv(
    path: Path,
    metrics: dict[str, dict[str, float | int]],
    gates: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "backbone",
        "role",
        "status",
        "joint_objective",
        "qty_mae",
        "qty_rmse",
        "le_p95_qty_mae",
        "time_nll",
        "parameter_count",
        "best_epoch",
        "completed_epochs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for backbone in EXPECTED_BACKBONES:
            writer.writerow({
                "backbone": backbone,
                "role": "control" if backbone == CONTROL_BACKBONE else "candidate",
                "status": (
                    "control" if backbone == CONTROL_BACKBONE else gates[backbone]["status"]
                ),
                **metrics[backbone],
            })


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# TitanTPP Memory Backbone Seed-42 Validation Gate",
        "",
        f"- Final status: **{payload['status'].upper()}**",
        f"- Selected backbone: `{payload['selected_backbone']}`",
        "- Scope: Intermittent validation only; held-out test not evaluated",
        "",
        "| Backbone | MAE | RMSE | <=p95 MAE | Time NLL | Parameters | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for backbone in EXPECTED_BACKBONES:
        metrics = payload["metrics"][backbone]
        status = "control" if backbone == CONTROL_BACKBONE else payload["gates"][backbone]["status"]
        lines.append(
            f"| {backbone} | {metrics['qty_mae']:.8f} | {metrics['qty_rmse']:.8f} | "
            f"{metrics['le_p95_qty_mae']:.8f} | {metrics['time_nll']:.8f} | "
            f"{metrics['parameter_count']} | {status} |"
        )
    for backbone in CANDIDATE_BACKBONES:
        gate = payload["gates"][backbone]
        lines.extend([
            "",
            f"## {backbone}",
            "",
            f"- Overall MAE improvement: `{gate['deltas']['overall_mae_improvement_pct']:.4f}%`",
            f"- Overall RMSE improvement: `{gate['deltas']['overall_rmse_improvement_pct']:.4f}%`",
            f"- <=p95 MAE regression: `{gate['deltas']['le_p95_mae_regression_pct']:.4f}%`",
            f"- Time NLL regression: `{gate['deltas']['time_nll_absolute_regression']:.8f}`",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = (args.output_dir or artifact_dir / "comparison").resolve()
    launch = json.loads(
        (artifact_dir / "launch_contract.json").read_text(encoding="utf-8")
    )
    validate_launch_contract(launch)

    summaries = read_csv(artifact_dir / "run_summaries.csv")
    quantity_rows = read_csv(artifact_dir / "quantity_seed_metrics.csv")
    if len(summaries) != len(EXPECTED_BACKBONES):
        raise ValueError(
            f"Expected {len(EXPECTED_BACKBONES)} matched runs, found {len(summaries)}"
        )
    if {row["status"] for row in summaries} != {"success"}:
        raise ValueError("Every matched run must finish successfully")
    if {row["source_revision"] for row in summaries} != {launch["source_revision"]}:
        raise ValueError("Source revision mismatch")

    metrics = {
        backbone: metric_record(
            exact_row(summaries, backbone=backbone),
            quantity_rows,
            backbone=backbone,
        )
        for backbone in EXPECTED_BACKBONES
    }
    gates = {
        backbone: evaluate_gate(metrics[CONTROL_BACKBONE], metrics[backbone])
        for backbone in CANDIDATE_BACKBONES
    }
    selected = select_backbone(metrics, gates)
    accepted = [
        backbone
        for backbone in CANDIDATE_BACKBONES
        if gates[backbone]["status"] == "pass"
    ]
    payload = {
        "schema_version": 1,
        "status": "pass" if accepted else "fail",
        "control_backbone": CONTROL_BACKBONE,
        "accepted_backbones": accepted,
        "selected_backbone": selected,
        "selection_rule": (
            "Among passing candidates: lowest validation quantity MAE, then RMSE, "
            "then joint objective; retain hard-LMM when none pass."
        ),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": launch["source_revision"],
        "metrics": metrics,
        "gates": gates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "backbone_metrics.csv", metrics, gates)
    write_markdown(output_dir / "acceptance.md", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
