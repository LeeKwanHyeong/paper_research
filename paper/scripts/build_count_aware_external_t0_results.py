#!/usr/bin/env python3
"""Combine matched RMTPP, THP, NHP, and SAHP T0 validation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from paper.scripts.build_count_aware_tail_shared_multiseed_results import write_csv


SEEDS = (42, 52, 62)
MODELS = ("rmtpp", "thp", "nhp", "sahp")
BASE_MODELS = ("rmtpp", "thp")
EXTENSION_MODELS = ("nhp", "sahp")
VARIANT = "count_only_log_regression"
MODEL_LABELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "nhp": "Adapted NHP",
    "sahp": "Adapted SAHP",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--extension-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def validate_contracts(base: dict[str, Any], extension: dict[str, Any]) -> None:
    for name, contract in (("base", base), ("extension", extension)):
        if contract.get("status") != "complete":
            raise ValueError(f"{name} artifact is not complete")
        if parse_bool(contract.get("held_out_test_evaluated")):
            raise ValueError(f"{name} artifact evaluated held-out test")
        if tuple(contract.get("seeds", ())) != SEEDS:
            raise ValueError(f"{name} seed contract mismatch")
    for key in CONTRACT_KEYS:
        if extension.get(key) != base.get(key):
            raise ValueError(f"extension contract mismatch for {key}")
    if not set(BASE_MODELS).issubset(base.get("backbones", ())):
        raise ValueError("base artifact is missing RMTPP or THP")
    if tuple(extension.get("backbones", ())) != EXTENSION_MODELS:
        raise ValueError("extension artifact must contain only NHP and SAHP")
    if extension.get("model_role") != "t0_common_control":
        raise ValueError("extension artifact is not an official T0 control")
    if tuple(extension.get("quantity_variants", ())) != (VARIANT,):
        raise ValueError("extension quantity variant mismatch")
    if extension.get("time_head", {}).get("mode") != "legacy_clamped_rmtpp":
        raise ValueError("extension time-head mismatch")
    early = extension.get("early_stopping", {})
    if early.get("min_epochs") != 40 or early.get("patience") != 40:
        raise ValueError("extension early-stopping mismatch")
    if early.get("restore") != "best_validation_joint_objective":
        raise ValueError("extension checkpoint restore mismatch")


def validate_run(row: dict[str, str]) -> None:
    if row.get("status") != "success":
        raise ValueError("non-success run found")
    if row.get("variant") != VARIANT:
        raise ValueError("non-T0 run found")
    if row.get("evaluation_scope") != "validation_only":
        raise ValueError("non-validation run found")
    if parse_bool(row.get("held_out_test_evaluated")):
        raise ValueError("held-out test run found")
    if not row.get("checkpoint_state_sha256"):
        raise ValueError("checkpoint digest missing")
    for metric in RUN_METRICS:
        if not math.isfinite(float(row[metric])):
            raise ValueError(f"non-finite run metric: {metric}")


def collect_runs(base_rows: list[dict[str, str]], extension_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [
        row
        for source in (base_rows, extension_rows)
        for row in source
        if row.get("backbone") in MODELS and row.get("variant") == VARIANT
    ]
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    observed = {(row["backbone"], int(row["seed"])) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError(f"external T0 run grid mismatch: {sorted(observed)}")
    for row in rows:
        validate_run(row)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        group = [row for row in rows if row["backbone"] == model]
        record: dict[str, Any] = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "model_role": "t0_common_control",
            "n_seeds": len(SEEDS),
        }
        for metric in RUN_METRICS:
            mean, std = mean_std([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def collect_scale_rows(filename: str, artifacts: tuple[Path, Path]) -> list[dict[str, Any]]:
    rows = []
    for artifact in artifacts:
        rows.extend(
            row
            for row in read_csv(artifact / filename)
            if row.get("backbone") in MODELS and row.get("variant") == VARIANT
        )
    return rows


def summarize_scales(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    strata = sorted({(int(row["stratum_order"]), row["stratum"], row["stratum_label"]) for row in rows})
    for order, stratum, label in strata:
        for model in MODELS:
            group = [row for row in rows if row["backbone"] == model and row["stratum"] == stratum]
            if {int(row["seed"]) for row in group} != set(SEEDS):
                raise ValueError(f"scale seed grid mismatch for {model}/{stratum}")
            record: dict[str, Any] = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "stratum_order": order,
                "stratum": stratum,
                "stratum_label": label,
                "count": int(group[0]["count"]),
                "share": float(group[0]["share"]),
                "n_seeds": len(SEEDS),
            }
            for metric in SCALE_METRICS:
                mean, std = mean_std([float(row[metric]) for row in group])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
            output.append(record)
    return output


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Count-aware External T0 Three-seed Validation Comparison",
        "",
        "- Scope: Intermittent validation only",
        "- Models: Adapted RMTPP, THP, NHP, and SAHP",
        "- Shared role: T0 common control",
        "- Held-out test: not evaluated",
        "- TitanTPP-T1 is reported separately as the proposed incumbent.",
        "- H0/H3 are diagnostic-only and excluded.",
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
    return "\n".join(lines) + "\n"


def render_audit(base: dict[str, Any], extension: dict[str, Any]) -> str:
    return "\n".join([
        "# External T0 Contract Audit",
        "",
        "- Status: PASS",
        "- RMTPP/THP source: existing matched T0 artifact",
        "- NHP/SAHP source: fresh official `t0_common_control` artifact",
        "- Dataset, split, epoch, seed, batch, learning rate, lookback, max sequence length: matched",
        "- Quantity loss: direct log-MSE for all four models",
        "- Time head: `legacy_clamped_rmtpp` for all four models",
        "- Checkpoint: minimum validation joint objective",
        "- Held-out test: not evaluated",
        f"- Base source revision: `{base.get('source_revision')}`",
        f"- Extension source revision: `{extension.get('source_revision')}`",
        "",
    ])


def main() -> None:
    args = parse_args()
    base = args.base_artifact.resolve()
    extension = args.extension_artifact.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_contract = read_json(base / "launch_contract.json")
    extension_contract = read_json(extension / "launch_contract.json")
    validate_contracts(base_contract, extension_contract)
    runs = collect_runs(
        read_csv(base / "run_summaries.csv"),
        read_csv(extension / "run_summaries.csv"),
    )
    summaries = summarize_runs(runs)
    quantity = summarize_scales(collect_scale_rows("quantity_seed_metrics.csv", (base, extension)))
    history = summarize_scales(collect_scale_rows("history_seed_metrics.csv", (base, extension)))
    write_csv(output / "combined_seed_metrics.csv", runs)
    write_csv(output / "model_summary.csv", summaries)
    write_csv(output / "quantity_scale_summary.csv", quantity)
    write_csv(output / "history_scale_summary.csv", history)
    (output / "comparison.md").write_text(render_markdown(summaries), encoding="utf-8")
    (output / "contract_audit.md").write_text(render_audit(base_contract, extension_contract), encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "model_role": "t0_common_control",
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "excluded_models": ["TitanTPP-T1", "H0", "H3"],
    }
    (output / "comparison.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
