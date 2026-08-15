#!/usr/bin/env python3
"""Apply the frozen TitanTPP tail-aware validation gate."""

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
    TAIL_HEAD_ONLY_VARIANT,
    TAIL_SHARED_VARIANT,
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
    variant: str,
    stratum: str | None = None,
) -> dict[str, str]:
    matches = [
        row for row in rows
        if row["backbone"] == "titantpp"
        and row["variant"] == variant
        and int(row["seed"]) == 42
        and (stratum is None or row.get("stratum") == stratum)
    ]
    if len(matches) != 1:
        suffix = f"/{stratum}" if stratum else ""
        raise ValueError(f"Expected one TitanTPP/{variant}{suffix} row, found {len(matches)}")
    return matches[0]


def weighted_mae(rows: list[dict[str, str]], *, variant: str, strata: set[str]) -> float:
    selected = [
        row for row in rows
        if row["backbone"] == "titantpp"
        and row["variant"] == variant
        and int(row["seed"]) == 42
        and row["stratum"] in strata
    ]
    if {row["stratum"] for row in selected} != strata:
        raise ValueError(f"Incomplete strata for {variant}: {selected}")
    total = sum(int(row["count"]) for row in selected)
    return sum(float(row["qty_mae"]) * int(row["count"]) for row in selected) / total


def metric_record(
    summary: dict[str, str],
    quantity_rows: list[dict[str, str]],
    *,
    variant: str,
) -> dict[str, float]:
    return {
        "qty_mae": float(summary["best_val_qty_mae"]),
        "qty_rmse": float(summary["best_val_qty_rmse"]),
        "time_nll": float(summary["best_val_time_nll"]),
        "le_p95_qty_mae": weighted_mae(
            quantity_rows,
            variant=variant,
            strata={"le_p50", "p50_p90", "p90_p95"},
        ),
        "gt_p99_qty_mae": float(
            exact_row(quantity_rows, variant=variant, stratum="gt_p99")["qty_mae"]
        ),
    }


def evaluate_gate(control: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    tolerance = 1e-12
    rmse_improvement = -percent_change(candidate["qty_rmse"], control["qty_rmse"])
    p99_improvement = -percent_change(
        candidate["gt_p99_qty_mae"], control["gt_p99_qty_mae"]
    )
    overall_mae_regression = percent_change(candidate["qty_mae"], control["qty_mae"])
    body_mae_regression = percent_change(
        candidate["le_p95_qty_mae"], control["le_p95_qty_mae"]
    )
    time_nll_regression = candidate["time_nll"] - control["time_nll"]
    finite = all(math.isfinite(value) for value in (*control.values(), *candidate.values()))
    checks = {
        "finite_contract": finite,
        "rmse_or_gt_p99_mae_improvement_at_least_5pct": (
            rmse_improvement + tolerance >= 5.0
            or p99_improvement + tolerance >= 5.0
        ),
        "overall_mae_regression_at_most_2pct": (
            overall_mae_regression <= 2.0 + tolerance
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
            "overall_rmse_improvement_pct": rmse_improvement,
            "gt_p99_mae_improvement_pct": p99_improvement,
            "overall_mae_regression_pct": overall_mae_regression,
            "le_p95_mae_regression_pct": body_mae_regression,
            "time_nll_absolute_regression": time_nll_regression,
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# TitanTPP Log-MSE + Tail-aware Auxiliary Seed-42 Gate",
        "",
        f"- 최종 판정: **{payload['status'].upper()}**",
        f"- 선택 Variant: `{payload['accepted_variant'] or 'none'}`",
        "- 범위: Intermittent validation only, held-out test 미사용",
        "",
        "| Variant | MAE | RMSE | <=p95 MAE | >p99 MAE | Time NLL | 판정 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for variant, metrics in payload["metrics"].items():
        status = "control" if variant == VARIANT else payload["gates"][variant]["status"]
        lines.append(
            f"| {variant} | {metrics['qty_mae']:.8f} | {metrics['qty_rmse']:.8f} | "
            f"{metrics['le_p95_qty_mae']:.8f} | {metrics['gt_p99_qty_mae']:.8f} | "
            f"{metrics['time_nll']:.8f} | {status} |"
        )
    for variant in (TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT):
        gate = payload["gates"][variant]
        lines.extend([
            "",
            f"## {variant}",
            "",
            f"- RMSE 개선: `{gate['deltas']['overall_rmse_improvement_pct']:.4f}%`",
            f"- >p99 MAE 개선: `{gate['deltas']['gt_p99_mae_improvement_pct']:.4f}%`",
            f"- 전체 MAE 악화: `{gate['deltas']['overall_mae_regression_pct']:.4f}%`",
            f"- <=p95 MAE 악화: `{gate['deltas']['le_p95_mae_regression_pct']:.4f}%`",
            f"- Time NLL 악화: `{gate['deltas']['time_nll_absolute_regression']:.8f}`",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = (args.output_dir or artifact_dir / "comparison").resolve()
    launch = json.loads((artifact_dir / "launch_contract.json").read_text(encoding="utf-8"))
    expected_variants = {VARIANT, TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT}
    if launch.get("status") != "complete":
        raise ValueError("Screening artifact is not complete")
    if launch.get("dataset") != "intermittent_frozen_5000":
        raise ValueError(f"Unexpected dataset: {launch.get('dataset')}")
    if launch.get("held_out_test_evaluated") is not False:
        raise ValueError("Held-out test scope is not locked")
    if launch.get("seeds") != [42] or launch.get("backbones") != ["titantpp"]:
        raise ValueError("Expected TitanTPP seed-42 screening")
    if set(launch.get("quantity_variants", [])) != expected_variants:
        raise ValueError("Expected fresh T0/T1/T2 variants")

    summaries = read_csv(artifact_dir / "run_summaries.csv")
    quantity_rows = read_csv(artifact_dir / "quantity_seed_metrics.csv")
    if len(summaries) != 3:
        raise ValueError(f"Expected three matched runs, found {len(summaries)}")
    if {row["source_revision"] for row in summaries} != {launch["source_revision"]}:
        raise ValueError("Source revision mismatch")

    metrics = {
        variant: metric_record(
            exact_row(summaries, variant=variant),
            quantity_rows,
            variant=variant,
        )
        for variant in (VARIANT, TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT)
    }
    gates = {
        variant: evaluate_gate(metrics[VARIANT], metrics[variant])
        for variant in (TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT)
    }
    if gates[TAIL_HEAD_ONLY_VARIANT]["status"] == "pass":
        accepted = TAIL_HEAD_ONLY_VARIANT
    elif gates[TAIL_SHARED_VARIANT]["status"] == "pass":
        accepted = TAIL_SHARED_VARIANT
    else:
        accepted = None
    payload = {
        "schema_version": 1,
        "status": "pass" if accepted else "fail",
        "accepted_variant": accepted,
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
    write_markdown(output_dir / "acceptance.md", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
