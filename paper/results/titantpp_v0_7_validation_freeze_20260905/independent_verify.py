#!/usr/bin/env python3
"""Independently verify the TitanTPP v0.7 validation-freeze tables.

This verifier deliberately reconstructs the published tables from the source
CSV exports rather than importing ``build_freeze.py`` or reading per-run
``summary.json`` files.  It never reads a held-out prediction or metric row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEEDS = (42, 52, 62)
MODELS = ("rmtpp", "thp", "titantpp")
REFERENCES = ("rmtpp", "thp")
MODEL_LABELS = {
    "rmtpp": "Count-aware RMTPP",
    "thp": "Count-aware THP",
    "titantpp": "TitanTPP Hard-LMM",
}
DATASETS = {
    "taxi": {
        "label": "Taxi",
        "root": ROOT
        / "search_artifacts/count_aware_taxi_t0_t1_e300_20260824/"
        "yellow_trip_hourly/t0_common_control",
    },
    "instacart": {
        "label": "Instacart",
        "root": ROOT
        / "search_artifacts/count_aware_instacart_t0_e300_20260824_recovery1/"
        "insta_market_basket/t0_common_control",
    },
}
FLOAT_ABS_TOL = 1e-12
FLOAT_REL_TOL = 1e-12


class Verification:
    """Collect compact verification evidence while retaining exact failures."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks: list[dict[str, Any]] = []
        self.rows_compared = 0
        self.cells_compared = 0
        self.numeric_cells_compared = 0
        self.categorical_cells_compared = 0
        self.source_files_hashed = 0
        self.max_abs_error = 0.0

    def check(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.errors.append(f"{name}: {detail}")

    def error(self, message: str) -> None:
        self.errors.append(message)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def sample_std(values: Iterable[float]) -> float:
    return statistics.stdev(list(values))


def relative_improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / abs(reference)


def normalized_bool(value: Any) -> str:
    return str(value).strip().lower()


def require_columns(path: Path, fields: list[str], required: set[str]) -> None:
    missing = sorted(required.difference(fields))
    if missing:
        raise AssertionError(f"{path}: missing columns {missing}")


def unique_index(
    rows: list[dict[str, Any]], key_fields: tuple[str, ...], description: str
) -> dict[tuple[str, ...], dict[str, Any]]:
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        if key in index:
            raise AssertionError(f"{description}: duplicate key {key}")
        index[key] = row
    return index


def source_summary_path(dataset: str, model: str, seed: int) -> str:
    path = (
        DATASETS[dataset]["root"]
        / "runs"
        / model
        / "count_only_log_regression"
        / f"seed_{seed}"
        / "summary.json"
    )
    return str(path.relative_to(ROOT))


def load_validation_sources(
    verification: Verification,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    quantity_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    run_required = {
        "status",
        "backbone",
        "backbone_label",
        "variant",
        "seed",
        "epochs",
        "completed_epochs",
        "best_epoch",
        "best_val_joint_objective",
        "best_val_time_nll",
        "best_val_log_qty_mse",
        "best_val_qty_mae",
        "best_val_qty_rmse",
        "parameter_count",
        "source_revision",
        "evaluation_scope",
        "held_out_test_evaluated",
        "checkpoint_state_sha256",
    }
    stratum_required = {
        "backbone",
        "variant",
        "seed",
        "stratum_order",
        "stratum",
        "stratum_label",
        "share",
        "count",
        "time_nll",
        "log_qty_mse",
        "qty_mae",
        "qty_rmse",
        "qty_bias",
    }

    for dataset, spec in DATASETS.items():
        root: Path = spec["root"]
        contract_path = root / "launch_contract.json"
        contract = read_json(contract_path)
        contract_ok = (
            contract.get("evaluation_scope") == "validation_only"
            and contract.get("held_out_test_evaluated") is False
            and tuple(contract.get("seeds", [])) == SEEDS
            and contract.get("model_role") == "t0_common_control"
        )
        verification.check(
            f"{dataset}_source_contract_validation_only",
            contract_ok,
            (
                f"scope={contract.get('evaluation_scope')}, "
                f"held_out={contract.get('held_out_test_evaluated')}, "
                f"seeds={contract.get('seeds')}, role={contract.get('model_role')}"
            ),
        )

        run_path = root / "run_summaries.csv"
        run_fields, all_run_rows = read_csv(run_path)
        require_columns(run_path, run_fields, run_required)
        selected_runs = [
            row
            for row in all_run_rows
            if row["backbone"] in MODELS
            and row["variant"] == "count_only_log_regression"
            and int(row["seed"]) in SEEDS
        ]
        run_index = unique_index(
            selected_runs, ("backbone", "seed"), f"{dataset} source run summaries"
        )
        expected_run_keys = {(model, str(seed)) for model in MODELS for seed in SEEDS}
        verification.check(
            f"{dataset}_source_run_matrix",
            set(run_index) == expected_run_keys,
            f"observed={len(run_index)} expected={len(expected_run_keys)}",
        )

        run_scope_ok = True
        joint_ok = True
        for model in MODELS:
            for seed in SEEDS:
                source = run_index.get((model, str(seed)))
                if source is None:
                    continue
                run_scope_ok = run_scope_ok and (
                    source["status"] == "success"
                    and source["evaluation_scope"] == "validation_only"
                    and normalized_bool(source["held_out_test_evaluated"]) == "false"
                )
                expected_joint = float(source["best_val_time_nll"]) + float(
                    source["best_val_log_qty_mse"]
                )
                observed_joint = float(source["best_val_joint_objective"])
                joint_ok = joint_ok and math.isclose(
                    observed_joint,
                    expected_joint,
                    rel_tol=0.0,
                    # The source exporter aggregates the three values through
                    # independent validation batches; its recorded drift is
                    # below 1.3e-9.  This check is separate from the stricter
                    # generated-cell comparisons below.
                    abs_tol=1e-8,
                )
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": spec["label"],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": seed,
                        "best_epoch": int(source["best_epoch"]),
                        "completed_epochs": int(source["completed_epochs"]),
                        "joint_objective": observed_joint,
                        "time_nll": float(source["best_val_time_nll"]),
                        "log_qty_mse": float(source["best_val_log_qty_mse"]),
                        "qty_mae": float(source["best_val_qty_mae"]),
                        "qty_rmse": float(source["best_val_qty_rmse"]),
                        "parameter_count": int(source["parameter_count"]),
                        "source_revision": source["source_revision"],
                        "checkpoint_state_sha256": source["checkpoint_state_sha256"],
                        "evaluation_scope": source["evaluation_scope"],
                        "held_out_test_evaluated": normalized_bool(
                            source["held_out_test_evaluated"]
                        ),
                        "source_summary": source_summary_path(dataset, model, seed),
                    }
                )
        verification.check(
            f"{dataset}_run_rows_validation_only",
            run_scope_ok,
            "all selected rows must be successful validation-only rows with held-out=false",
        )
        verification.check(
            f"{dataset}_joint_objective_identity",
            joint_ok,
            "joint objective must equal clamped time loss plus log-quantity MSE",
        )

        for filename, destination in (
            ("quantity_seed_metrics.csv", quantity_rows),
            ("history_seed_metrics.csv", history_rows),
        ):
            path = root / filename
            fields, all_rows = read_csv(path)
            require_columns(path, fields, stratum_required)
            selected = [
                row
                for row in all_rows
                if row["backbone"] in MODELS
                and row["variant"] == "count_only_log_regression"
                and int(row["seed"]) in SEEDS
            ]
            unique_index(
                selected,
                ("backbone", "seed", "stratum"),
                f"{dataset} {filename}",
            )
            for source in selected:
                model = source["backbone"]
                destination.append(
                    {
                        "dataset": dataset,
                        "dataset_label": spec["label"],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "seed": int(source["seed"]),
                        "stratum_order": int(source["stratum_order"]),
                        "stratum": source["stratum"],
                        "stratum_label": source["stratum_label"],
                        "count": int(source["count"]),
                        "share": float(source["share"]),
                        "time_nll": float(source["time_nll"]),
                        "log_qty_mse": float(source["log_qty_mse"]),
                        "qty_mae": float(source["qty_mae"]),
                        "qty_rmse": float(source["qty_rmse"]),
                        "qty_bias": float(source["qty_bias"]),
                    }
                )

    verification.check(
        "source_seed_row_count",
        len(seed_rows) == len(DATASETS) * len(MODELS) * len(SEEDS),
        f"observed={len(seed_rows)} expected=18",
    )
    return seed_rows, quantity_rows, history_rows


def aggregate_validation(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        for model in MODELS:
            rows = sorted(
                (
                    row
                    for row in seed_rows
                    if row["dataset"] == dataset and row["model"] == model
                ),
                key=lambda row: row["seed"],
            )
            if len(rows) != 3:
                raise AssertionError(f"Missing source seed rows: {dataset}/{model}")
            result: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": spec["label"],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_seeds": 3,
            }
            for metric in (
                "joint_objective",
                "time_nll",
                "log_qty_mse",
                "qty_mae",
                "qty_rmse",
            ):
                values = [float(row[metric]) for row in rows]
                result[f"{metric}_mean"] = mean(values)
                result[f"{metric}_std"] = sample_std(values)
            output.append(result)
    return output


def paired_tables(
    seed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (row["dataset"], row["model"], row["seed"]): row for row in seed_rows
    }
    paired: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        for reference in REFERENCES:
            for seed in SEEDS:
                candidate = indexed[(dataset, "titantpp", seed)]
                baseline = indexed[(dataset, reference, seed)]
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "dataset_label": spec["label"],
                    "candidate": "titantpp",
                    "candidate_label": MODEL_LABELS["titantpp"],
                    "reference": reference,
                    "reference_label": MODEL_LABELS[reference],
                    "seed": seed,
                }
                for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                    cvalue = float(candidate[metric])
                    rvalue = float(baseline[metric])
                    row[f"{metric}_candidate_minus_reference"] = cvalue - rvalue
                    row[f"{metric}_relative_improvement_pct"] = relative_improvement(
                        rvalue, cvalue
                    )
                    row[f"{metric}_candidate_better"] = normalized_bool(cvalue < rvalue)
                paired.append(row)

    summary: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        for reference in REFERENCES:
            rows = [
                row
                for row in paired
                if row["dataset"] == dataset and row["reference"] == reference
            ]
            result: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": spec["label"],
                "candidate": "titantpp",
                "candidate_label": MODEL_LABELS["titantpp"],
                "reference": reference,
                "reference_label": MODEL_LABELS[reference],
                "n_paired_seeds": 3,
            }
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                deltas = [
                    float(row[f"{metric}_candidate_minus_reference"]) for row in rows
                ]
                gains = [
                    float(row[f"{metric}_relative_improvement_pct"]) for row in rows
                ]
                result[f"{metric}_mean_delta"] = mean(deltas)
                result[f"{metric}_std_delta"] = sample_std(deltas)
                result[f"{metric}_mean_paired_relative_improvement_pct"] = mean(gains)
                result[f"{metric}_better_seed_count"] = sum(gain > 0.0 for gain in gains)
            summary.append(result)
    return paired, summary


def aggregate_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["stratum"])].append(row)

    output: list[dict[str, Any]] = []
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            key[0],
            key[1],
            min(int(row["stratum_order"]) for row in grouped[key]),
        ),
    )
    for key in ordered_keys:
        selected = sorted(grouped[key], key=lambda row: row["seed"])
        if [row["seed"] for row in selected] != list(SEEDS):
            raise AssertionError(f"Incomplete stratum seed matrix: {key}")
        count_values = {int(row["count"]) for row in selected}
        share_values = {float(row["share"]) for row in selected}
        label_values = {row["stratum_label"] for row in selected}
        order_values = {int(row["stratum_order"]) for row in selected}
        if len(count_values) != 1 or len(share_values) != 1:
            raise AssertionError(f"Count/share varies by seed: {key}")
        if len(label_values) != 1 or len(order_values) != 1:
            raise AssertionError(f"Stratum metadata varies by seed: {key}")
        first = selected[0]
        result: dict[str, Any] = {
            "dataset": first["dataset"],
            "dataset_label": first["dataset_label"],
            "model": first["model"],
            "model_label": first["model_label"],
            "stratum_order": first["stratum_order"],
            "stratum": first["stratum"],
            "stratum_label": first["stratum_label"],
            "count_per_seed": first["count"],
            "share": first["share"],
            "n_seeds": 3,
        }
        for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse", "qty_bias"):
            values = [float(row[metric]) for row in selected]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_std"] = sample_std(values)
        output.append(result)
    return output


def paired_strata_tables(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute within-seed Hard-LMM differences for each stored stratum."""
    indexed = {
        (row["dataset"], row["model"], row["seed"], row["stratum"]): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        strata = sorted(
            {row["stratum"] for row in rows if row["dataset"] == dataset},
            key=lambda stratum: min(
                int(row["stratum_order"])
                for row in rows
                if row["dataset"] == dataset and row["stratum"] == stratum
            ),
        )
        for reference in REFERENCES:
            for stratum in strata:
                for seed in SEEDS:
                    candidate = indexed[(dataset, "titantpp", seed, stratum)]
                    baseline = indexed[(dataset, reference, seed, stratum)]
                    if int(candidate["count"]) != int(baseline["count"]):
                        raise AssertionError(
                            f"Paired stratum count mismatch: {dataset}/{reference}/{seed}/{stratum}"
                        )
                    if not math.isclose(
                        float(candidate["share"]),
                        float(baseline["share"]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise AssertionError(
                            f"Paired stratum share mismatch: {dataset}/{reference}/{seed}/{stratum}"
                        )
                    result: dict[str, Any] = {
                        "dataset": dataset,
                        "dataset_label": spec["label"],
                        "candidate": "titantpp",
                        "candidate_label": MODEL_LABELS["titantpp"],
                        "reference": reference,
                        "reference_label": MODEL_LABELS[reference],
                        "seed": seed,
                        "stratum_order": candidate["stratum_order"],
                        "stratum": stratum,
                        "stratum_label": candidate["stratum_label"],
                        "count": candidate["count"],
                        "share": candidate["share"],
                    }
                    for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                        cvalue = float(candidate[metric])
                        rvalue = float(baseline[metric])
                        result[f"{metric}_candidate_minus_reference"] = cvalue - rvalue
                        result[f"{metric}_relative_improvement_pct"] = relative_improvement(
                            rvalue, cvalue
                        )
                        result[f"{metric}_candidate_better"] = normalized_bool(cvalue < rvalue)
                    paired.append(result)

    summary: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (row["dataset"], row["reference"], row["stratum"])
            for row in paired
        },
        key=lambda key: (
            key[0],
            key[1],
            min(
                int(row["stratum_order"])
                for row in paired
                if row["dataset"] == key[0]
                and row["reference"] == key[1]
                and row["stratum"] == key[2]
            ),
        ),
    )
    for dataset, reference, stratum in group_keys:
        selected = sorted(
            (
                row
                for row in paired
                if row["dataset"] == dataset
                and row["reference"] == reference
                and row["stratum"] == stratum
            ),
            key=lambda row: row["seed"],
        )
        if [row["seed"] for row in selected] != list(SEEDS):
            raise AssertionError(
                f"Incomplete paired stratum matrix: {dataset}/{reference}/{stratum}"
            )
        first = selected[0]
        result: dict[str, Any] = {
            "dataset": dataset,
            "dataset_label": DATASETS[dataset]["label"],
            "candidate": "titantpp",
            "candidate_label": MODEL_LABELS["titantpp"],
            "reference": reference,
            "reference_label": MODEL_LABELS[reference],
            "stratum_order": first["stratum_order"],
            "stratum": stratum,
            "stratum_label": first["stratum_label"],
            "count_per_seed": first["count"],
            "share": first["share"],
            "n_paired_seeds": 3,
        }
        for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
            deltas = [
                float(row[f"{metric}_candidate_minus_reference"]) for row in selected
            ]
            gains = [
                float(row[f"{metric}_relative_improvement_pct"]) for row in selected
            ]
            result[f"{metric}_mean_delta"] = mean(deltas)
            result[f"{metric}_std_delta"] = sample_std(deltas)
            result[f"{metric}_mean_paired_relative_improvement_pct"] = mean(gains)
            result[f"{metric}_better_seed_count"] = sum(gain > 0.0 for gain in gains)
        summary.append(result)
    return paired, summary


def interface_observed_metrics() -> list[dict[str, Any]]:
    categorical_path = (
        ROOT / "paper/results/taxi_quantity_interface_ablation_20260810/source/run_summaries.csv"
    )
    direct_path = (
        ROOT
        / "paper/results/taxi_positive_regression_control_20260811/"
        "log_regression/run_summaries.csv"
    )
    _, categorical = read_csv(categorical_path)
    _, direct = read_csv(direct_path)
    combined = categorical + direct
    output: list[dict[str, Any]] = []
    for variant in ("uniform_categorical", "quantile_categorical", "log_regression"):
        selected = sorted(
            (row for row in combined if row["variant"] == variant),
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in selected] != list(SEEDS):
            raise AssertionError(f"Incomplete legacy interface seed matrix: {variant}")
        if any(
            row["evaluation_scope"] != "validation_only"
            or normalized_bool(row["held_out_test_evaluated"]) != "false"
            for row in selected
        ):
            raise AssertionError(f"Non-validation row in legacy interface source: {variant}")
        for metric in ("best_val_nll", "best_val_qty_mae", "best_val_qty_rmse"):
            values = [float(row[metric]) for row in selected]
            output.append(
                {
                    "dataset": "taxi",
                    "variant": variant,
                    "metric": metric,
                    "n_seeds": 3,
                    "mean": mean(values),
                    "sample_std": sample_std(values),
                    "qualification": "diagnostic_only_contract_fail",
                }
            )
    return output


def compare_csv(
    verification: Verification,
    filename: str,
    expected_rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    numeric_fields: set[str],
) -> None:
    path = OUT / filename
    fields, observed_rows = read_csv(path)
    expected_fields = list(expected_rows[0]) if expected_rows else []
    schema_ok = set(fields) == set(expected_fields) and len(fields) == len(expected_fields)
    verification.check(
        f"{filename}_schema",
        schema_ok,
        f"observed={fields}; expected={expected_fields}",
    )
    expected_index = unique_index(expected_rows, key_fields, f"expected {filename}")
    observed_index = unique_index(observed_rows, key_fields, f"observed {filename}")
    keys_ok = set(expected_index) == set(observed_index)
    verification.check(
        f"{filename}_row_keys",
        keys_ok,
        f"observed_rows={len(observed_index)} expected_rows={len(expected_index)}",
    )

    table_errors_before = len(verification.errors)
    compared_rows = 0
    for key in sorted(set(expected_index).intersection(observed_index)):
        expected = expected_index[key]
        observed = observed_index[key]
        compared_rows += 1
        for field in expected_fields:
            if field not in observed:
                verification.error(f"{filename} {key}: missing field {field}")
                continue
            verification.cells_compared += 1
            if field in numeric_fields:
                verification.numeric_cells_compared += 1
                try:
                    expected_value = float(expected[field])
                    observed_value = float(observed[field])
                except (TypeError, ValueError) as exc:
                    verification.error(f"{filename} {key} {field}: non-numeric value: {exc}")
                    continue
                if not math.isfinite(expected_value) or not math.isfinite(observed_value):
                    verification.error(
                        f"{filename} {key} {field}: non-finite value "
                        f"expected={expected_value} observed={observed_value}"
                    )
                    continue
                absolute_error = abs(expected_value - observed_value)
                verification.max_abs_error = max(
                    verification.max_abs_error, absolute_error
                )
                if not math.isclose(
                    expected_value,
                    observed_value,
                    rel_tol=FLOAT_REL_TOL,
                    abs_tol=FLOAT_ABS_TOL,
                ):
                    verification.error(
                        f"{filename} {key} {field}: expected={expected_value!r} "
                        f"observed={observed_value!r} abs_error={absolute_error!r}"
                    )
            else:
                verification.categorical_cells_compared += 1
                if str(expected[field]) != observed[field]:
                    verification.error(
                        f"{filename} {key} {field}: expected={expected[field]!r} "
                        f"observed={observed[field]!r}"
                    )
    verification.rows_compared += compared_rows
    verification.check(
        f"{filename}_all_cells",
        len(verification.errors) == table_errors_before,
        f"compared_rows={compared_rows}; tolerance=abs {FLOAT_ABS_TOL}, rel {FLOAT_REL_TOL}",
    )


def verify_source_manifest(verification: Verification) -> None:
    path = OUT / "source_manifest.json"
    manifest = read_json(path)
    verification.check(
        "source_manifest_validation_boundary",
        manifest.get("evaluation_scope") == "validation_only"
        and manifest.get("held_out_test_evaluated") is False,
        (
            f"scope={manifest.get('evaluation_scope')}, "
            f"held_out={manifest.get('held_out_test_evaluated')}"
        ),
    )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("source_manifest.json files must be an object")

    required: set[str] = set()
    for spec in DATASETS.values():
        root: Path = spec["root"]
        required.update(
            str((root / filename).relative_to(ROOT))
            for filename in (
                "launch_contract.json",
                "run_summaries.csv",
                "quantity_seed_metrics.csv",
                "history_seed_metrics.csv",
            )
        )
    required.update(
        {
            "paper/tables/T1_dataset_statistics.csv",
            "paper/results/intermittent_log_backbone_control_20260811/source_5080_baselines/launch_contract.json",
            "sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json",
            "paper/results/hard_lmm_key_value_screening_5090_20260904/final_audit.json",
            "paper/results/hard_lmm_key_value_screening_5090_20260904/README.md",
        }
    )
    verification.check(
        "source_manifest_required_inputs",
        required.issubset(files),
        f"missing={sorted(required.difference(files))}",
    )

    root_resolved = ROOT.resolve()
    hash_errors_before = len(verification.errors)
    for relative, expected_hash in sorted(files.items()):
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            verification.error(f"source_manifest path escapes repository: {relative}")
            continue
        if not candidate.is_file():
            verification.error(f"source_manifest missing file: {relative}")
            continue
        actual_hash = sha256_file(candidate)
        verification.source_files_hashed += 1
        if actual_hash != expected_hash:
            verification.error(
                f"source_manifest hash mismatch: {relative}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
    verification.check(
        "source_manifest_all_hashes",
        len(verification.errors) == hash_errors_before,
        f"hashed_files={verification.source_files_hashed}",
    )


def verify_dataset_identity(verification: Verification) -> None:
    audit = read_json(OUT / "dataset_identity_audit.json")
    top_level_ok = (
        audit.get("v0_7_three_dataset_identity_decision")
        == "FAIL_INTERMITTENT_POPULATION_MISMATCH"
        and audit.get("taxi_instacart_integrated_table_decision")
        == "PASS_SAME_ARTIFACT_IDENTITIES_ONLY"
        and audit.get("integrated_validation_table_datasets") == ["taxi", "instacart"]
        and audit.get("intermittent_included_in_integrated_validation_table") is False
    )
    verification.check(
        "dataset_identity_top_level_decision",
        top_level_ok,
        (
            "expected Taxi/Instacart-only integration and an explicit Intermittent "
            "population mismatch"
        ),
    )
    checks = audit.get("checks")
    if not isinstance(checks, list):
        raise TypeError("dataset_identity_audit.json checks must be an array")
    decisions = {
        item.get("dataset"): item.get("decision")
        for item in checks
        if isinstance(item, dict)
    }
    expected = {
        "taxi": "PASS_SAME_ARTIFACT_IDENTITY",
        "instacart": "PASS_SAME_ARTIFACT_IDENTITY",
        "intermittent": "FAIL_POPULATION_IDENTITY_MISMATCH",
    }
    verification.check(
        "dataset_identity_per_dataset_decisions",
        decisions == expected,
        f"observed={decisions}; expected={expected}",
    )


def verify_final_model_evidence(verification: Verification) -> None:
    audit = read_json(OUT / "final_model_evidence_audit.json")
    candidates = audit.get("candidates")
    if not isinstance(candidates, dict):
        raise TypeError("final_model_evidence_audit.json candidates must be an object")
    original = candidates.get("original_mark_free_hard_lmm_t0")
    separate = candidates.get("separate_key_hard_lmm")
    if not isinstance(original, dict) or not isinstance(separate, dict):
        raise TypeError("final model audit is missing candidate objects")

    top_level_ok = (
        audit.get("evaluation_scope") == "validation_only"
        and audit.get("held_out_test_evaluated") is False
        and audit.get("required_common_datasets") == ["taxi", "instacart"]
        and audit.get("required_seeds") == list(SEEDS)
        and audit.get("final_structure_decision")
        == "FREEZE_ORIGINAL_MARK_FREE_HARD_LMM_T0"
        and audit.get("separate_key_in_main_tables") is False
    )
    verification.check(
        "final_model_evidence_top_level_decision",
        top_level_ok,
        "expected original mark-free T0 freeze under a validation-only boundary",
    )
    original_ok = (
        original.get("decision") == "PASS_THREE_SEED_COVERAGE_ON_BOTH_DATASETS"
        and original.get("dataset_seed_coverage")
        == {"taxi": list(SEEDS), "instacart": list(SEEDS)}
        and original.get("eligible_as_common_final_structure") is True
        and original.get("same_model_role") == "t0_common_control"
    )
    verification.check(
        "original_hard_lmm_three_seed_coverage",
        original_ok,
        f"coverage={original.get('dataset_seed_coverage')}",
    )

    source_path = (
        ROOT / "paper/results/hard_lmm_key_value_screening_5090_20260904/final_audit.json"
    )
    source = read_json(source_path)
    source_seed = int(source["contract"]["seed"])
    source_decision = source["decision"]
    separate_ok = (
        separate.get("decision")
        == "FAIL_THREE_SEED_COVERAGE_AND_PROSPECTIVE_CROSS_DATASET_GATE"
        and separate.get("dataset_seed_coverage")
        == {"taxi": [source_seed], "instacart": [source_seed]}
        and separate.get("dataset_gates_passed")
        == source_decision.get("dataset_gates_passed")
        and separate.get("dataset_gates_evaluated")
        == source_decision.get("dataset_gates_evaluated")
        and separate.get("candidate_accepted_for_expansion") is False
        and source_decision.get("candidate_accepted_for_expansion") is False
        and separate.get("eligible_as_common_final_structure") is False
    )
    verification.check(
        "separate_key_single_seed_rejection",
        separate_ok,
        (
            f"coverage={separate.get('dataset_seed_coverage')}, "
            f"accepted={separate.get('candidate_accepted_for_expansion')}"
        ),
    )


def verify_quantity_interface_audit(verification: Verification) -> None:
    audit = read_json(OUT / "quantity_interface_audit.json")
    categorical_path = (
        ROOT
        / "paper/results/taxi_quantity_interface_ablation_20260810/source/"
        "launch_contract.json"
    )
    direct_path = (
        ROOT
        / "paper/results/taxi_positive_regression_control_20260811/"
        "log_regression/launch_contract.json"
    )
    categorical = read_json(categorical_path)
    direct = read_json(direct_path)

    verification.check(
        "quantity_interface_audit_decision",
        audit.get("decision") == "FAIL" and audit.get("paper_use") == "diagnostic_only",
        f"decision={audit.get('decision')}, paper_use={audit.get('paper_use')}",
    )

    common_fields = (
        "dataset",
        "data_sha256",
        "split_rows",
        "seeds",
        "epochs",
        "batch_size",
        "lr",
        "grad_clip",
        "lookback_weeks",
        "max_seq_len",
        "hidden_dim",
        "max_train_batches",
        "max_val_batches",
        "checkpoint_selection",
        "early_stopping",
        "evaluation_scope",
        "held_out_test_evaluated",
    )
    differing_common_fields = [
        field for field in common_fields if categorical.get(field) != direct.get(field)
    ]
    common_contract_ok = (
        not differing_common_fields
        and categorical.get("seeds") == list(SEEDS)
        and categorical.get("epochs") == direct.get("epochs") == 300
        and categorical.get("evaluation_scope")
        == direct.get("evaluation_scope")
        == "validation_only"
        and categorical.get("held_out_test_evaluated") is False
        and direct.get("held_out_test_evaluated") is False
    )
    verification.check(
        "quantity_interface_common_contract_configuration",
        common_contract_ok,
        f"differing_fields={differing_common_fields}",
    )

    revisions_differ = categorical.get("source_revision") != direct.get("source_revision")
    verification.check(
        "quantity_interface_source_revision_mismatch",
        revisions_differ,
        (
            f"categorical={categorical.get('source_revision')}, "
            f"direct={direct.get('source_revision')}"
        ),
    )

    categorical_interfaces = categorical.get("interfaces", {})
    categorical_bins_ok = True
    for variant in ("uniform_categorical", "quantile_categorical"):
        interface = categorical_interfaces.get(variant, {})
        bin_count = interface.get("bin_count")
        categorical_bins_ok = categorical_bins_ok and (
            interface.get("fitted_on") == "train"
            and isinstance(bin_count, int)
            and len(interface.get("representatives", [])) == bin_count
            and len(interface.get("edges", [])) == bin_count - 1
        )
    verification.check(
        "quantity_interface_train_fitted_bins_and_representatives",
        categorical_bins_ok,
        "uniform and quantile bins must be train-fitted with one representative per bin",
    )

    direct_interface = direct.get("interfaces", {}).get("log_regression", {})
    direct_residual = (
        direct_interface.get("history_quantity_input") == "log10_within_mark_residual"
    )
    active_mark_free_features = ["log1p_delta_t", "log1p_raw_quantity"]
    categorical_serialized = json.dumps(categorical_interfaces, sort_keys=True)
    direct_serialized = json.dumps(direct.get("interfaces", {}), sort_keys=True)
    active_mark_free_absent = all(
        feature not in categorical_serialized and feature not in direct_serialized
        for feature in active_mark_free_features
    )
    verification.check(
        "quantity_interface_history_is_not_active_mark_free_token",
        direct_residual and active_mark_free_absent,
        (
            f"direct_history_quantity_input={direct_interface.get('history_quantity_input')}; "
            f"active_features_present={not active_mark_free_absent}"
        ),
    )

    expected_blocking = {
        "single_frozen_source_revision",
        "mark_free_common_history_input",
        "only_quantity_target_head_and_reconstruction_change",
        "exact_log_transformed_categorical_binning_control",
        "same_checkpoint_selector_semantics_and_active_v0_7_selector",
        "supporting_reconstructed_log_quantity_mse_reported",
    }
    blocking = audit.get("blocking_requirements")
    blocking_ok = (
        isinstance(blocking, list)
        and len(blocking) == len(expected_blocking)
        and set(blocking) == expected_blocking
    )
    verification.check(
        "quantity_interface_exact_six_blockers",
        blocking_ok,
        f"observed={blocking}; expected={sorted(expected_blocking)}",
    )

    audit_checks = audit.get("checks")
    if not isinstance(audit_checks, list):
        raise TypeError("quantity_interface_audit.json checks must be an array")
    audit_check_map = {
        item.get("requirement"): item.get("passed")
        for item in audit_checks
        if isinstance(item, dict)
    }
    blocker_flags_ok = all(
        audit_check_map.get(requirement) is False for requirement in expected_blocking
    )
    known_passes = {
        "same_rmtpp_encoder_family_and_dimensions",
        "same_dataset_split_seeds_budget_and_optimizer_settings",
        "train_fitted_bin_edges_and_numeric_representatives",
        "same_checkpoint_selector_identifier",
        "raw_scale_rmse_and_mae_reported",
        "held_out_test_locked",
    }
    pass_flags_ok = all(audit_check_map.get(requirement) is True for requirement in known_passes)
    verification.check(
        "quantity_interface_audit_check_flags",
        blocker_flags_ok and pass_flags_ok,
        "all six blockers must be false and all six established conditions must be true",
    )

    scope = audit.get("scope")
    scope_ok = isinstance(scope, dict) and (
        scope.get("evaluation") == "validation_only"
        and scope.get("held_out_test_evaluated") is False
        and scope.get("categorical_source_revision") == categorical.get("source_revision")
        and scope.get("direct_log_source_revision") == direct.get("source_revision")
    )
    verification.check(
        "quantity_interface_audit_scope_and_revisions",
        scope_ok,
        f"scope={scope}",
    )


def verify_tables(verification: Verification) -> None:
    seed, quantity_seed, history_seed = load_validation_sources(verification)
    validation_aggregate = aggregate_validation(seed)
    paired_seed, paired_summary = paired_tables(seed)
    quantity_aggregate = aggregate_strata(quantity_seed)
    history_aggregate = aggregate_strata(history_seed)
    quantity_paired, quantity_paired_summary = paired_strata_tables(quantity_seed)
    history_paired, history_paired_summary = paired_strata_tables(history_seed)
    interface_metrics = interface_observed_metrics()

    seed_numeric = {
        "seed",
        "best_epoch",
        "completed_epochs",
        "joint_objective",
        "time_nll",
        "log_qty_mse",
        "qty_mae",
        "qty_rmse",
        "parameter_count",
    }
    aggregate_numeric = {
        "n_seeds",
        *{
            f"{metric}_{suffix}"
            for metric in (
                "joint_objective",
                "time_nll",
                "log_qty_mse",
                "qty_mae",
                "qty_rmse",
            )
            for suffix in ("mean", "std")
        },
    }
    paired_numeric = {
        "seed",
        *{
            f"{metric}_{suffix}"
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse")
            for suffix in (
                "candidate_minus_reference",
                "relative_improvement_pct",
            )
        },
    }
    paired_summary_numeric = {
        "n_paired_seeds",
        *{
            f"{metric}_{suffix}"
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse")
            for suffix in (
                "mean_delta",
                "std_delta",
                "mean_paired_relative_improvement_pct",
                "better_seed_count",
            )
        },
    }
    stratum_seed_numeric = {
        "seed",
        "stratum_order",
        "count",
        "share",
        "time_nll",
        "log_qty_mse",
        "qty_mae",
        "qty_rmse",
        "qty_bias",
    }
    stratum_aggregate_numeric = {
        "stratum_order",
        "count_per_seed",
        "share",
        "n_seeds",
        *{
            f"{metric}_{suffix}"
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse", "qty_bias")
            for suffix in ("mean", "std")
        },
    }
    stratum_paired_numeric = {
        "seed",
        "stratum_order",
        "count",
        "share",
        *{
            f"{metric}_{suffix}"
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse")
            for suffix in (
                "candidate_minus_reference",
                "relative_improvement_pct",
            )
        },
    }
    stratum_paired_summary_numeric = {
        "stratum_order",
        "count_per_seed",
        "share",
        "n_paired_seeds",
        *{
            f"{metric}_{suffix}"
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse")
            for suffix in (
                "mean_delta",
                "std_delta",
                "mean_paired_relative_improvement_pct",
                "better_seed_count",
            )
        },
    }

    table_specs = (
        (
            "validation_seed_metrics.csv",
            seed,
            ("dataset", "model", "seed"),
            seed_numeric,
        ),
        (
            "validation_aggregate.csv",
            validation_aggregate,
            ("dataset", "model"),
            aggregate_numeric,
        ),
        (
            "paired_seed_deltas.csv",
            paired_seed,
            ("dataset", "reference", "seed"),
            paired_numeric,
        ),
        (
            "paired_delta_summary.csv",
            paired_summary,
            ("dataset", "reference"),
            paired_summary_numeric,
        ),
        (
            "quantity_strata_seed_metrics.csv",
            quantity_seed,
            ("dataset", "model", "seed", "stratum"),
            stratum_seed_numeric,
        ),
        (
            "quantity_strata_aggregate.csv",
            quantity_aggregate,
            ("dataset", "model", "stratum"),
            stratum_aggregate_numeric,
        ),
        (
            "quantity_strata_paired_seed_deltas.csv",
            quantity_paired,
            ("dataset", "reference", "seed", "stratum"),
            stratum_paired_numeric,
        ),
        (
            "quantity_strata_paired_delta_summary.csv",
            quantity_paired_summary,
            ("dataset", "reference", "stratum"),
            stratum_paired_summary_numeric,
        ),
        (
            "history_strata_seed_metrics.csv",
            history_seed,
            ("dataset", "model", "seed", "stratum"),
            stratum_seed_numeric,
        ),
        (
            "history_strata_aggregate.csv",
            history_aggregate,
            ("dataset", "model", "stratum"),
            stratum_aggregate_numeric,
        ),
        (
            "history_strata_paired_seed_deltas.csv",
            history_paired,
            ("dataset", "reference", "seed", "stratum"),
            stratum_paired_numeric,
        ),
        (
            "history_strata_paired_delta_summary.csv",
            history_paired_summary,
            ("dataset", "reference", "stratum"),
            stratum_paired_summary_numeric,
        ),
        (
            "quantity_interface_observed_metrics.csv",
            interface_metrics,
            ("dataset", "variant", "metric"),
            {"n_seeds", "mean", "sample_std"},
        ),
    )
    for filename, rows, key_fields, numeric_fields in table_specs:
        try:
            compare_csv(
                verification,
                filename,
                rows,
                key_fields,
                numeric_fields,
            )
        except Exception as exc:  # keep the report useful if one table is malformed
            verification.error(f"{filename}: {type(exc).__name__}: {exc}")


def main() -> int:
    verification = Verification()
    try:
        verify_tables(verification)
    except Exception as exc:
        verification.error(f"source_recomputation: {type(exc).__name__}: {exc}")

    try:
        verify_source_manifest(verification)
    except Exception as exc:
        verification.error(f"source_manifest: {type(exc).__name__}: {exc}")

    try:
        verify_dataset_identity(verification)
    except Exception as exc:
        verification.error(f"dataset_identity: {type(exc).__name__}: {exc}")

    try:
        verify_final_model_evidence(verification)
    except Exception as exc:
        verification.error(f"final_model_evidence: {type(exc).__name__}: {exc}")

    try:
        verify_quantity_interface_audit(verification)
    except Exception as exc:
        verification.error(f"quantity_interface_audit: {type(exc).__name__}: {exc}")

    status = "PASS" if not verification.errors else "FAIL"
    report = {
        "verification_id": "titantpp_v0_7_validation_freeze_independent_20260905",
        "status": status,
        "method": (
            "Independent standard-library recomputation from T0 run_summaries.csv, "
            "quantity_seed_metrics.csv and history_seed_metrics.csv; no import from the "
            "builder and no per-run summary JSON metric read."
        ),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "counts": {
            "rows_compared": verification.rows_compared,
            "cells_compared": verification.cells_compared,
            "numeric_cells_compared": verification.numeric_cells_compared,
            "categorical_cells_compared": verification.categorical_cells_compared,
            "source_files_hashed": verification.source_files_hashed,
            "checks": len(verification.checks),
            "failed_checks": sum(not item["passed"] for item in verification.checks),
        },
        "max_abs_error": verification.max_abs_error,
        "tolerance": {"absolute": FLOAT_ABS_TOL, "relative": FLOAT_REL_TOL},
        "checks": verification.checks,
        "errors": verification.errors,
    }
    write_json(OUT / "independent_verification.json", report)
    print(
        f"{status}: {verification.rows_compared} rows / "
        f"{verification.cells_compared} cells compared; "
        f"max_abs_error={verification.max_abs_error:.3g}"
    )
    if verification.errors:
        for error in verification.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
