#!/usr/bin/env python3
"""Validate the H1 stable exact time head against the matched H0 reference."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any

from models.TPPs.CountAwareTPP import (
    TIME_HEAD_MODE_SCALED_EXACT,
    TIME_HEAD_MODE_SCALED_EXACT_STABLE,
)
from paper.scripts.count_aware_tpp_backbone.constants import VARIANT
from paper.scripts.run_intermittent_log_backbone_control import (
    EXPECTED_DATA_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
)


BACKBONE = "titantpp"
SEED = 42
BODY_STRATA = {"le_p50", "p50_p90", "p90_p95"}
SHARED_LAUNCH_FIELDS = (
    "dataset",
    "epochs",
    "batch_size",
    "lr",
    "lookback_weeks",
    "max_seq_len",
    "hidden_dim",
    "lambda_log_qty",
    "grad_clip",
    "evaluation_scope",
    "held_out_test_evaluated",
    "data_sha256",
    "split_manifest_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--stability-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percent_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("Percentage comparison requires a nonzero reference")
    return 100.0 * (candidate - reference) / reference


def exact_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["backbone"] == BACKBONE
        and row["variant"] == VARIANT
        and int(row["seed"]) == SEED
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {BACKBONE}/{VARIANT}/seed-{SEED} row, found {len(matches)}"
        )
    return matches[0]


def weighted_body_mae(rows: list[dict[str, str]]) -> float:
    selected = [
        row
        for row in rows
        if row["backbone"] == BACKBONE
        and row["variant"] == VARIANT
        and int(row["seed"]) == SEED
        and row["stratum"] in BODY_STRATA
    ]
    if {row["stratum"] for row in selected} != BODY_STRATA:
        raise ValueError("Incomplete <=p95 quantity strata")
    total = sum(int(row["count"]) for row in selected)
    if total < 1:
        raise ValueError("Empty <=p95 quantity strata")
    return sum(float(row["qty_mae"]) * int(row["count"]) for row in selected) / total


def metric_record(
    summary: dict[str, str],
    quantity_rows: list[dict[str, str]],
) -> dict[str, float | int]:
    return {
        "joint_objective": float(summary["best_val_joint_objective"]),
        "time_nll": float(summary["best_val_time_nll"]),
        "log_qty_mse": float(summary["best_val_log_qty_mse"]),
        "qty_mae": float(summary["best_val_qty_mae"]),
        "qty_rmse": float(summary["best_val_qty_rmse"]),
        "le_p95_qty_mae": weighted_body_mae(quantity_rows),
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "parameter_count": int(summary["parameter_count"]),
    }


def evaluate_safety_gate(
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
        },
    }


def validate_stability_decision(decision: dict[str, Any]) -> None:
    h1_gate = decision.get("variant_gates", {}).get("H1", {})
    expected = {
        "selected_variant": "H1",
        "h2_executed": False,
        "selection_source": "train_stability_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
    }
    mismatches = {
        key: {"expected": value, "observed": decision.get(key)}
        for key, value in expected.items()
        if decision.get(key) != value
    }
    if h1_gate.get("passed") is not True:
        mismatches["variant_gates.H1.passed"] = {
            "expected": True,
            "observed": h1_gate.get("passed"),
        }
    if mismatches:
        raise ValueError(f"H1 train-only selection mismatch: {mismatches}")


def validate_launch_contracts(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    candidate_expected = {
        "status": "complete",
        "backbones": [BACKBONE],
        "seeds": [SEED],
        "quantity_variants": [VARIANT],
        "expected_run_count": 1,
        "completed_run_count": 1,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "data_sha256": EXPECTED_DATA_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
    }
    mismatches = {
        key: {"expected": value, "observed": candidate.get(key)}
        for key, value in candidate_expected.items()
        if candidate.get(key) != value
    }
    for key in SHARED_LAUNCH_FIELDS:
        if candidate.get(key) != reference.get(key):
            mismatches[f"matched.{key}"] = {
                "expected": reference.get(key),
                "observed": candidate.get(key),
            }
    expected_early_stopping = {
        "min_epochs": 40,
        "patience": 40,
        "restore": "best_validation_joint_objective",
    }
    for label, launch in (("reference", reference), ("candidate", candidate)):
        early_stopping = launch.get("early_stopping", {})
        for key, value in expected_early_stopping.items():
            if early_stopping.get(key) != value:
                mismatches[f"{label}.early_stopping.{key}"] = {
                    "expected": value,
                    "observed": early_stopping.get(key),
                }
    time_contracts = {
        "reference": (
            reference.get("time_head", {}),
            TIME_HEAD_MODE_SCALED_EXACT,
            10.0 / 3.0,
            30.0,
            40.0,
        ),
        "candidate": (
            candidate.get("time_head", {}),
            TIME_HEAD_MODE_SCALED_EXACT_STABLE,
            2.0 / 3.0,
            6.0,
            8.0,
        ),
    }
    for label, (
        time_head,
        mode,
        w_max,
        intercept_limit,
        wd_limit,
    ) in time_contracts.items():
        expected_time = {
            "mode": mode,
            "time_scale": 3.0,
            "time_w_max": w_max,
            "time_intercept_limit": intercept_limit,
            "density_unit": "original_delta_t_with_jacobian",
            "wd_clamp": 0.0,
        }
        for key, value in expected_time.items():
            observed = time_head.get(key)
            matches = (
                math.isclose(float(observed), value, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(value, float) and observed is not None
                else observed == value
            )
            if not matches:
                mismatches[f"{label}.time_head.{key}"] = {
                    "expected": value,
                    "observed": observed,
                }
        observed_wd_limit = time_head.get(
            "time_wd_safety_limit",
            time_head.get("train_time_statistics", {}).get("wd_safety_limit"),
        )
        if observed_wd_limit is None or not math.isclose(
            float(observed_wd_limit),
            wd_limit,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            mismatches[f"{label}.time_head.time_wd_safety_limit"] = {
                "expected": wd_limit,
                "observed": observed_wd_limit,
            }
    if mismatches:
        raise ValueError(f"H1 validation contract mismatch: {mismatches}")


def validate_summary_contracts(
    reference: dict[str, str],
    candidate: dict[str, str],
    candidate_launch: dict[str, Any],
) -> None:
    for label, row in (("reference", reference), ("candidate", candidate)):
        if row["status"] != "success":
            raise ValueError(f"{label} run did not complete successfully")
        if row["evaluation_scope"] != "validation_only":
            raise ValueError(f"{label} run is not validation-only")
        if row["held_out_test_evaluated"] != "False":
            raise ValueError(f"{label} run evaluated held-out test")
    if candidate["source_revision"] != candidate_launch["source_revision"]:
        raise ValueError("Candidate source revision mismatch")
    shared_fields = (
        "backbone",
        "variant",
        "seed",
        "epochs",
        "parameter_count",
        "evaluation_scope",
        "held_out_test_evaluated",
    )
    mismatches = {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in shared_fields
        if reference.get(key) != candidate.get(key)
    }
    reference_encoder = ast.literal_eval(reference["encoder_config"])
    candidate_encoder = ast.literal_eval(candidate["encoder_config"])
    for key in (
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "memory_mode",
        "persistent_mem_size",
        "lmm_mem_size",
        "lmm_topk",
        "quantity_memory_gradient_mode",
        "max_len",
    ):
        if reference_encoder.get(key) != candidate_encoder.get(key):
            mismatches[f"encoder_config.{key}"] = {
                "reference": reference_encoder.get(key),
                "candidate": candidate_encoder.get(key),
            }
    if mismatches:
        raise ValueError(f"Matched H0/H1 summary mismatch: {mismatches}")


def write_metrics_csv(
    path: Path,
    metrics: dict[str, dict[str, float | int]],
) -> None:
    fieldnames = [
        "time_head",
        "role",
        "joint_objective",
        "time_nll",
        "log_qty_mse",
        "qty_mae",
        "qty_rmse",
        "le_p95_qty_mae",
        "best_epoch",
        "completed_epochs",
        "parameter_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, role in (("H0", "reference"), ("H1", "candidate")):
            writer.writerow({"time_head": name, "role": role, **metrics[name]})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    gate = payload["safety_gate"]
    lines = [
        "# Time Head v2 H1 Validation Safety Gate",
        "",
        f"- Final status: **{payload['status'].upper()}**",
        f"- Selected time head: `{payload['selected_time_head']}`",
        "- Scope: Intermittent seed 42 validation only",
        "- Held-out test evaluated: `False`",
        "",
        "| Head | Joint | Time NLL | MAE | RMSE | <=p95 MAE | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("H0", "H1"):
        metric = payload["metrics"][name]
        lines.append(
            f"| {name} | {metric['joint_objective']:.8f} | "
            f"{metric['time_nll']:.8f} | {metric['qty_mae']:.8f} | "
            f"{metric['qty_rmse']:.8f} | {metric['le_p95_qty_mae']:.8f} | "
            f"{metric['best_epoch']} |"
        )
    lines.extend(
        [
            "",
            "## H1 regressions relative to H0",
            "",
            f"- Time NLL: `{gate['deltas']['time_nll_absolute_regression']:.8f}`",
            f"- Quantity MAE: `{gate['deltas']['qty_mae_regression_pct']:.4f}%`",
            f"- Quantity RMSE: `{gate['deltas']['qty_rmse_regression_pct']:.4f}%`",
            f"- <=p95 MAE: `{gate['deltas']['le_p95_qty_mae_regression_pct']:.4f}%`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_dir = args.candidate_artifact.resolve()
    reference_dir = args.reference_artifact.resolve()
    stability_dir = args.stability_artifact.resolve()
    output_dir = (args.output_dir or candidate_dir / "comparison").resolve()

    candidate_launch = read_json(candidate_dir / "launch_contract.json")
    reference_launch = read_json(reference_dir / "launch_contract.json")
    stability_decision = read_json(stability_dir / "decision.json")
    validate_stability_decision(stability_decision)
    validate_launch_contracts(reference_launch, candidate_launch)

    candidate_summary = exact_summary(read_csv(candidate_dir / "run_summaries.csv"))
    reference_summary = exact_summary(read_csv(reference_dir / "run_summaries.csv"))
    validate_summary_contracts(reference_summary, candidate_summary, candidate_launch)
    metrics = {
        "H0": metric_record(
            reference_summary,
            read_csv(reference_dir / "quantity_seed_metrics.csv"),
        ),
        "H1": metric_record(
            candidate_summary,
            read_csv(candidate_dir / "quantity_seed_metrics.csv"),
        ),
    }
    gate = evaluate_safety_gate(metrics["H0"], metrics["H1"])
    payload = {
        "schema_version": 1,
        "status": gate["status"],
        "selected_time_head": "H1" if gate["status"] == "pass" else "H0_reference",
        "reference_time_head": "H0",
        "candidate_time_head": "H1",
        "selection_rule": (
            "H1 passes when Time NLL regression is <=0.01 and overall MAE, "
            "RMSE, and <=p95 MAE regressions are each <=2%."
        ),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "candidate_source_revision": candidate_launch["source_revision"],
        "reference_source_revision": reference_launch["source_revision"],
        "train_only_selection": {
            "selected_variant": stability_decision["selected_variant"],
            "h1_gate_passed": stability_decision["variant_gates"]["H1"]["passed"],
        },
        "metrics": metrics,
        "safety_gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "time_head_metrics.csv", metrics)
    write_markdown(output_dir / "acceptance.md", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
