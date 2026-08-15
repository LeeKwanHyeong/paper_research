#!/usr/bin/env python3
"""Apply the frozen TitanTPP log-normal K=1 validation gate."""

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

from paper.scripts.run_count_aware_tpp_backbone_control import (
    LOGNORMAL_VARIANT,
    VARIANT,
)


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
    variant: str,
    seed: int,
) -> dict[str, str]:
    matches = [
        row for row in rows
        if row["backbone"] == backbone
        and row["variant"] == variant
        and int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for {backbone}/{variant}/seed_{seed}, found {len(matches)}"
        )
    return matches[0]


def p99_row(
    rows: list[dict[str, str]],
    *,
    backbone: str,
    variant: str,
    seed: int,
) -> dict[str, str]:
    matches = [
        row for row in rows
        if row["backbone"] == backbone
        and row["variant"] == variant
        and int(row["seed"]) == seed
        and row["stratum"] == "gt_p99"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one >p99 row for {backbone}/{variant}/seed_{seed}, "
            f"found {len(matches)}"
        )
    return matches[0]


def evaluate_gate(
    control: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, Any]:
    tolerance = 1e-12
    mae_improvement_pct = -percent_change(candidate["qty_mae"], control["qty_mae"])
    rmse_regression_pct = percent_change(candidate["qty_rmse"], control["qty_rmse"])
    p99_mae_regression_pct = percent_change(
        candidate["p99_qty_mae"], control["p99_qty_mae"]
    )
    time_nll_regression = candidate["time_nll"] - control["time_nll"]
    finite = all(math.isfinite(value) for value in (*control.values(), *candidate.values()))
    checks = {
        "finite_contract": finite and candidate["quantity_scale_mean"] > 1e-3,
        "overall_mae_improvement_at_least_5pct": (
            mae_improvement_pct + tolerance >= 5.0
        ),
        "overall_rmse_regression_at_most_2pct": (
            rmse_regression_pct <= 2.0 + tolerance
        ),
        "p99_mae_regression_at_most_2pct": (
            p99_mae_regression_pct <= 2.0 + tolerance
        ),
        "time_nll_regression_at_most_0_01": (
            time_nll_regression <= 0.01 + tolerance
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "deltas": {
            "overall_mae_improvement_pct": mae_improvement_pct,
            "overall_rmse_regression_pct": rmse_regression_pct,
            "p99_mae_regression_pct": p99_mae_regression_pct,
            "time_nll_absolute_regression": time_nll_regression,
        },
    }


def metric_record(
    summary: dict[str, str],
    tail: dict[str, str],
) -> dict[str, float]:
    return {
        "qty_mae": float(summary["best_val_qty_mae"]),
        "qty_rmse": float(summary["best_val_qty_rmse"]),
        "time_nll": float(summary["best_val_time_nll"]),
        "p99_qty_mae": float(tail["qty_mae"]),
        "quantity_scale_mean": float(summary["best_val_quantity_scale_mean"]),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    comparison = payload["comparison"]
    deltas = payload["gate"]["deltas"]
    checks = payload["gate"]["checks"]
    lines = [
        "# TitanTPP Log-normal K=1 Seed-42 Validation Gate",
        "",
        f"- 판정: **{payload['gate']['status'].upper()}**",
        "- 범위: Intermittent validation only, held-out test 미사용",
        "- 비교: fresh matched TitanTPP log-MSE 대 TitanTPP log-normal K=1",
        "",
        "| Variant | MAE | RMSE | >p99 MAE | Time NLL | Mean sigma |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("control", "candidate"):
        row = comparison[key]
        lines.append(
            f"| {key} | {row['qty_mae']:.8f} | {row['qty_rmse']:.8f} | "
            f"{row['p99_qty_mae']:.8f} | {row['time_nll']:.8f} | "
            f"{row['quantity_scale_mean']:.8f} |"
        )
    lines.extend([
        "",
        "| Gate | 관측값 | 통과 |",
        "|---|---:|:---:|",
        f"| 전체 MAE 개선률 >= 5% | {deltas['overall_mae_improvement_pct']:.4f}% | "
        f"{'Y' if checks['overall_mae_improvement_at_least_5pct'] else 'N'} |",
        f"| 전체 RMSE 악화율 <= 2% | {deltas['overall_rmse_regression_pct']:.4f}% | "
        f"{'Y' if checks['overall_rmse_regression_at_most_2pct'] else 'N'} |",
        f"| >p99 MAE 악화율 <= 2% | {deltas['p99_mae_regression_pct']:.4f}% | "
        f"{'Y' if checks['p99_mae_regression_at_most_2pct'] else 'N'} |",
        f"| Time NLL 악화 <= 0.01 | {deltas['time_nll_absolute_regression']:.8f} | "
        f"{'Y' if checks['time_nll_regression_at_most_0_01'] else 'N'} |",
        f"| 수치 안정성 및 sigma 양수 | - | "
        f"{'Y' if checks['finite_contract'] else 'N'} |",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = (args.output_dir or artifact_dir / "comparison").resolve()
    launch = json.loads((artifact_dir / "launch_contract.json").read_text(encoding="utf-8"))
    if launch.get("status") != "complete":
        raise ValueError("Screening artifact is not complete")
    if launch.get("held_out_test_evaluated") is not False:
        raise ValueError("Held-out test scope is not locked")
    if launch.get("seeds") != [42]:
        raise ValueError(f"Expected seed-42 screening, found {launch.get('seeds')}")
    if set(launch.get("backbones", [])) != {"thp", "titantpp"}:
        raise ValueError("Expected matched THP and TitanTPP backbones")
    if set(launch.get("quantity_variants", [])) != {VARIANT, LOGNORMAL_VARIANT}:
        raise ValueError("Expected fresh log-MSE and log-normal K=1 variants")

    summaries = read_csv(artifact_dir / "run_summaries.csv")
    quantity_rows = read_csv(artifact_dir / "quantity_seed_metrics.csv")
    if len(summaries) != 4:
        raise ValueError(f"Expected exactly four matched runs, found {len(summaries)}")
    revisions = {row["source_revision"] for row in summaries}
    if revisions != {launch["source_revision"]}:
        raise ValueError(f"Source revision mismatch: {revisions}")

    control_summary = exact_row(
        summaries, backbone="titantpp", variant=VARIANT, seed=42
    )
    candidate_summary = exact_row(
        summaries, backbone="titantpp", variant=LOGNORMAL_VARIANT, seed=42
    )
    control_tail = p99_row(
        quantity_rows, backbone="titantpp", variant=VARIANT, seed=42
    )
    candidate_tail = p99_row(
        quantity_rows, backbone="titantpp", variant=LOGNORMAL_VARIANT, seed=42
    )
    control = metric_record(control_summary, control_tail)
    candidate = metric_record(candidate_summary, candidate_tail)
    gate = evaluate_gate(control, candidate)
    payload = {
        "schema_version": 1,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": launch["source_revision"],
        "comparison": {"control": control, "candidate": candidate},
        "gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "acceptance.md", payload)
    print(json.dumps(gate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
