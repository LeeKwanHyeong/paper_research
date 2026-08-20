#!/usr/bin/env python3
"""Combine TitanTPP T1 seeds with matched T0 backbone baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 52, 62)
BASELINE_MODELS = ("rmtpp", "thp", "titantpp")
BASE_VARIANT = "count_only_log_regression"
T1_VARIANT = "count_only_log_mse_tail_shared"
MODEL_LABELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "titantpp": "TitanTPP-T0",
    "titantpp_t1": "TitanTPP-T1",
}
RUN_METRICS = (
    "best_val_joint_objective",
    "best_val_time_nll",
    "best_val_log_qty_mse",
    "best_val_qty_mae",
    "best_val_qty_rmse",
    "best_epoch",
    "completed_epochs",
    "elapsed_seconds",
)
SCALE_METRICS = (
    "joint_objective",
    "time_nll",
    "log_qty_mse",
    "qty_mae",
    "qty_rmse",
    "qty_bias",
)
CONTRACT_KEYS = (
    "dataset",
    "data_sha256",
    "split_manifest_sha256",
    "split_rows",
    "epochs",
    "batch_size",
    "lr",
    "lookback_weeks",
    "max_seq_len",
    "hidden_dim",
    "evaluation_scope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--t1-seed42-artifact", type=Path, required=True)
    parser.add_argument("--t1-extension-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for fieldname in row:
            if fieldname not in seen:
                seen.add(fieldname)
                fieldnames.append(fieldname)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def load_contract(path: Path) -> dict[str, Any]:
    return json.loads((path / "launch_contract.json").read_text(encoding="utf-8"))


def validate_contracts(
    baseline: dict[str, Any], seed42: dict[str, Any], extension: dict[str, Any]
) -> None:
    for name, contract in (
        ("baseline", baseline),
        ("t1_seed42", seed42),
        ("t1_extension", extension),
    ):
        if contract.get("status") != "complete":
            raise ValueError(f"{name} artifact is not complete")
        if parse_bool(contract.get("held_out_test_evaluated")):
            raise ValueError(f"{name} evaluated the held-out test")
    for key in CONTRACT_KEYS:
        expected = baseline.get(key)
        for name, contract in (("t1_seed42", seed42), ("t1_extension", extension)):
            if contract.get(key) != expected:
                raise ValueError(
                    f"{name} contract mismatch for {key}: {contract.get(key)!r}"
                )
    if tuple(baseline.get("seeds", ())) != SEEDS:
        raise ValueError("baseline seed contract mismatch")
    if tuple(seed42.get("seeds", ())) != (42,):
        raise ValueError("T1 seed-42 artifact contract mismatch")
    if tuple(extension.get("seeds", ())) != (52, 62):
        raise ValueError("T1 extension seed contract mismatch")


def validate_run_row(row: dict[str, str]) -> None:
    if row.get("status") != "success":
        raise ValueError("non-success run found")
    if row.get("evaluation_scope") != "validation_only":
        raise ValueError("non-validation run found")
    if parse_bool(row.get("held_out_test_evaluated")):
        raise ValueError("held-out test run found")
    if not row.get("checkpoint_state_sha256"):
        raise ValueError("checkpoint digest missing")
    for metric in RUN_METRICS:
        if not math.isfinite(float(row[metric])):
            raise ValueError(f"non-finite run metric: {metric}")


def collect_run_rows(
    baseline_rows: list[dict[str, str]],
    seed42_rows: list[dict[str, str]],
    extension_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in baseline_rows:
        if row["backbone"] in BASELINE_MODELS and row["variant"] == BASE_VARIANT:
            selected.append({**row, "model": row["backbone"]})
    for rows in (seed42_rows, extension_rows):
        for row in rows:
            if row["backbone"] == "titantpp" and row["variant"] == T1_VARIANT:
                selected.append({**row, "model": "titantpp_t1"})
    expected = {(model, seed) for model in MODEL_LABELS for seed in SEEDS}
    observed = {(row["model"], int(row["seed"])) for row in selected}
    if len(selected) != len(expected) or observed != expected:
        raise ValueError(f"combined run grid mismatch: {sorted(observed)}")
    for row in selected:
        validate_run_row(row)
    return selected


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model, label in MODEL_LABELS.items():
        group = [row for row in rows if row["model"] == model]
        record: dict[str, Any] = {"model": model, "model_label": label, "n_seeds": 3}
        for metric in RUN_METRICS:
            mean, std = mean_std([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def collect_scale_rows(
    filename: str,
    baseline_artifact: Path,
    seed42_artifact: Path,
    extension_artifact: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in read_csv(baseline_artifact / filename):
        if row["backbone"] in BASELINE_MODELS and row["variant"] == BASE_VARIANT:
            output.append({**row, "model": row["backbone"]})
    for artifact in (seed42_artifact, extension_artifact):
        for row in read_csv(artifact / filename):
            if row["backbone"] == "titantpp" and row["variant"] == T1_VARIANT:
                output.append({**row, "model": "titantpp_t1"})
    return output


def summarize_scales(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    strata = sorted(
        {(int(row["stratum_order"]), row["stratum"], row["stratum_label"]) for row in rows}
    )
    for order, stratum, label in strata:
        for model, model_label in MODEL_LABELS.items():
            group = [
                row for row in rows
                if row["model"] == model and row["stratum"] == stratum
            ]
            if {int(row["seed"]) for row in group} != set(SEEDS):
                raise ValueError(f"scale seed grid mismatch for {model}/{stratum}")
            record: dict[str, Any] = {
                "model": model,
                "model_label": model_label,
                "stratum_order": order,
                "stratum": stratum,
                "stratum_label": label,
                "count": int(group[0]["count"]),
                "share": float(group[0]["share"]),
                "n_seeds": 3,
            }
            for metric in SCALE_METRICS:
                mean, std = mean_std([float(row[metric]) for row in group])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
            output.append(record)
    return output


def improvement_pct(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def paired_lower_count(
    rows: list[dict[str, Any]], candidate: str, baseline: str, metric: str
) -> int:
    count = 0
    for seed in SEEDS:
        candidate_value = float(next(
            row[metric] for row in rows
            if row["model"] == candidate and int(row["seed"]) == seed
        ))
        baseline_value = float(next(
            row[metric] for row in rows
            if row["model"] == baseline and int(row["seed"]) == seed
        ))
        count += candidate_value < baseline_value
    return count


def build_comparison(
    summaries: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_model = {row["model"]: row for row in summaries}
    candidate = by_model["titantpp_t1"]
    comparisons = {}
    for baseline in BASELINE_MODELS:
        base = by_model[baseline]
        comparisons[baseline] = {
            "mae_improvement_pct": improvement_pct(
                candidate["best_val_qty_mae_mean"], base["best_val_qty_mae_mean"]
            ),
            "rmse_improvement_pct": improvement_pct(
                candidate["best_val_qty_rmse_mean"], base["best_val_qty_rmse_mean"]
            ),
            "time_nll_absolute_regression": (
                candidate["best_val_time_nll_mean"] - base["best_val_time_nll_mean"]
            ),
            "mae_lower_seed_count": paired_lower_count(
                rows, "titantpp_t1", baseline, "best_val_qty_mae"
            ),
            "rmse_lower_seed_count": paired_lower_count(
                rows, "titantpp_t1", baseline, "best_val_qty_rmse"
            ),
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "table_scope": "t0_common_controls_and_t1_incumbent",
        "model_roles": {
            "rmtpp": "t0_common_control",
            "thp": "t0_common_control",
            "titantpp": "t0_common_control",
            "titantpp_t1": "t1_incumbent",
        },
        "excluded_diagnostic_models": ["H0_scaled_exact", "H3_lognormal_duration"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "comparisons": comparisons,
    }


def render_markdown(
    summaries: list[dict[str, Any]], comparison: dict[str, Any]
) -> str:
    lines = [
        "# TitanTPP-T1 Three-seed Validation Comparison",
        "",
        "- Scope: Intermittent validation only",
        "- Seeds: 42, 52, 62",
        "- Held-out test: not evaluated",
        "- Table role: T0 common controls and the T1 incumbent",
        "- H0/H3 time heads: diagnostic-only and excluded from this model table",
        "",
        "| Model | Joint objective | Time NLL | Log quantity MSE | Quantity MAE | Quantity RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['model_label']} | "
            f"{row['best_val_joint_objective_mean']:.6f} +/- {row['best_val_joint_objective_std']:.6f} | "
            f"{row['best_val_time_nll_mean']:.6f} +/- {row['best_val_time_nll_std']:.6f} | "
            f"{row['best_val_log_qty_mse_mean']:.6f} +/- {row['best_val_log_qty_mse_std']:.6f} | "
            f"{row['best_val_qty_mae_mean']:.6f} +/- {row['best_val_qty_mae_std']:.6f} | "
            f"{row['best_val_qty_rmse_mean']:.6f} +/- {row['best_val_qty_rmse_std']:.6f} |"
        )
    lines.extend(["", "## TitanTPP-T1 deltas", ""])
    for baseline, values in comparison["comparisons"].items():
        lines.extend([
            f"### Versus {MODEL_LABELS[baseline]}",
            "",
            f"- MAE improvement: `{values['mae_improvement_pct']:.4f}%`",
            f"- RMSE improvement: `{values['rmse_improvement_pct']:.4f}%`",
            f"- Time NLL absolute regression: `{values['time_nll_absolute_regression']:.8f}`",
            f"- Lower MAE seeds: `{values['mae_lower_seed_count']}/3`",
            f"- Lower RMSE seeds: `{values['rmse_lower_seed_count']}/3`",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = args.baseline_artifact.resolve()
    seed42 = args.t1_seed42_artifact.resolve()
    extension = args.t1_extension_artifact.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    validate_contracts(
        load_contract(baseline), load_contract(seed42), load_contract(extension)
    )
    run_rows = collect_run_rows(
        read_csv(baseline / "run_summaries.csv"),
        read_csv(seed42 / "run_summaries.csv"),
        read_csv(extension / "run_summaries.csv"),
    )
    summaries = summarize_runs(run_rows)
    quantity_rows = collect_scale_rows(
        "quantity_seed_metrics.csv", baseline, seed42, extension
    )
    history_rows = collect_scale_rows(
        "history_seed_metrics.csv", baseline, seed42, extension
    )
    comparison = build_comparison(summaries, run_rows)

    write_csv(output / "combined_seed_metrics.csv", run_rows)
    write_csv(output / "model_summary.csv", summaries)
    write_csv(output / "quantity_scale_summary.csv", summarize_scales(quantity_rows))
    write_csv(output / "history_scale_summary.csv", summarize_scales(history_rows))
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "comparison.md").write_text(
        render_markdown(summaries, comparison) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
