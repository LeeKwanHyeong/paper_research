#!/usr/bin/env python3
"""Independently verify the frozen TitanTPP v0.7 validation artifacts.

This verifier intentionally does not import the artifact builder. It reconstructs
the 27 validation rows from their two source CSV files, recomputes aggregate and
paired statistics, and reads the frozen parquet files for dataset statistics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any

import polars as pl


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "paper/manifests/titantpp_v0_7_validation_freeze_verification.json"
MANIFEST_PATH = ROOT / "paper/manifests/titantpp_v0_7_validation_freeze_manifest.json"
MANIFEST_CSV_PATH = ROOT / "paper/manifests/titantpp_v0_7_validation_freeze_manifest.csv"

SEEDS = (42, 52, 62)
DATASET_ORDER = ("intermittent", "taxi", "instacart")
MODEL_ORDER = ("rmtpp", "thp", "titantpp")
REFERENCES = ("rmtpp", "thp")
METRICS = ("joint_objective", "time_nll", "log_qty_mse", "qty_mae", "qty_rmse")
PAIRED_METRICS = ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse")

MODEL_LABELS = {
    "rmtpp": "Count-aware RMTPP",
    "thp": "Count-aware THP",
    "titantpp": "Count-aware TitanTPP",
}

SOURCE_REVISIONS = {
    "intermittent": "044add1f3de768d804d9f0269fd0013bd9658a35",
    "taxi": "6a01aea9024db9e3ef6cfdd2c3d0219ceb320856",
    "instacart": "28293c43521615be2ed8fad5b043dc9df8e5e457",
}

DATASETS = {
    "intermittent": {
        "label": "Intermittent-5000",
        "sequence_unit": "part",
        "time_unit": "week",
        "lookback": 520,
        "max_sequence_length": 256,
        "validation_rows": 86285,
        "data_path": "sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet",
        "data_sha256": "85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f",
        "split_path": "sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json",
        "split_sha256": "393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04",
    },
    "taxi": {
        "label": "Taxi",
        "sequence_unit": "0.02-degree pickup grid cell",
        "time_unit": "hour",
        "lookback": 168,
        "max_sequence_length": 256,
        "validation_rows": 8268,
        "data_path": "sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet",
        "data_sha256": "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46",
        "split_path": "sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json",
        "split_sha256": "4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85",
    },
    "instacart": {
        "label": "Instacart",
        "sequence_unit": "user",
        "time_unit": "day",
        "lookback": 52,
        "max_sequence_length": 64,
        "validation_rows": 503733,
        "data_path": "sample_data/insta_market_basket/instacart_marked_target_with_split.parquet",
        "data_sha256": "06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2",
        "split_path": "sample_data/insta_market_basket/instacart_marked_target_split_manifest.json",
        "split_sha256": "6c6cdd41f847878fbb405b73dfa038fbb7a88ad53df6843b0cc9e64531a8b71d",
    },
}

EXPECTED_OUTPUTS = {
    "paper/tables/T1_v0_7_dataset_statistics.csv",
    "paper/tables/T2_v0_7_model_training_contract.csv",
    "paper/tables/T3_v0_7_backbone_validation.csv",
    "paper/tables/T4_v0_7_paired_titan_deltas.csv",
    "paper/tables/T5_v0_7_quantity_strata_validation.csv",
    "paper/tables/T1_v0_7_dataset_statistics.md",
    "paper/tables/T2_v0_7_model_training_contract.md",
    "paper/tables/T3_v0_7_backbone_validation.md",
    "paper/tables/T4_v0_7_paired_titan_deltas.md",
    "paper/tables/T5_v0_7_quantity_strata_validation.md",
    "paper/figures/F1_v0_7_mark_free_architecture.png",
    "paper/figures/F1_v0_7_mark_free_architecture.pdf",
    "paper/figures/F1_v0_7_mark_free_architecture.svg",
    "paper/figures/source_data/F2_v0_7_dataset_validation_errors.csv",
    "paper/figures/F2_v0_7_dataset_validation_errors.png",
    "paper/figures/F2_v0_7_dataset_validation_errors.pdf",
    "paper/figures/F2_v0_7_dataset_validation_errors.svg",
}

REQUIRED_STRATA_SOURCES = {
    "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_seed_metrics.csv",
    "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_summary.csv",
    "paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_seed_metrics.csv",
    "paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_aggregate.csv",
}

EXPECTED_PUBLICATION_DOCUMENTS = {
    "paper/titantpp_short_paper_draft_v0_7_manuscript.md",
    "README.md",
}

STRATA = {
    0: "le_p50",
    1: "p50_p90",
    2: "p90_p95",
    3: "p95_p99",
    4: "gt_p99",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}")


def canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        items = [canonical_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    return value


class Verification:
    def __init__(self) -> None:
        self.checks_total = 0
        self.checks_passed = 0
        self.failures: list[dict[str, str]] = []
        self.numeric_comparisons = 0
        self.max_numeric_abs_error = 0.0
        self.summary: dict[str, Any] = {}

    def check(self, condition: bool, check_id: str, detail: str = "") -> None:
        self.checks_total += 1
        if condition:
            self.checks_passed += 1
        else:
            self.failures.append({"check": check_id, "detail": detail})

    def equal(self, actual: Any, expected: Any, check_id: str) -> None:
        self.check(
            actual == expected,
            check_id,
            json.dumps(
                {
                    "expected": canonical_value(expected),
                    "actual": canonical_value(actual),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def numeric(
        self,
        actual: Any,
        expected: Any,
        check_id: str,
        *,
        abs_tol: float = 1e-12,
        rel_tol: float = 1e-12,
    ) -> None:
        self.numeric_comparisons += 1
        try:
            actual_number = float(actual)
            expected_number = float(expected)
            error = abs(actual_number - expected_number)
            if math.isfinite(error):
                self.max_numeric_abs_error = max(self.max_numeric_abs_error, error)
            ok = (
                math.isfinite(actual_number)
                and math.isfinite(expected_number)
                and math.isclose(actual_number, expected_number, abs_tol=abs_tol, rel_tol=rel_tol)
            )
            detail = (
                f"expected={expected_number:.17g}; actual={actual_number:.17g}; "
                f"abs_error={error:.17g}"
            )
        except (TypeError, ValueError) as error:
            ok = False
            detail = f"numeric conversion failed: {error}"
        self.check(ok, check_id, detail)

    def report(self, fatal_error: str | None = None) -> dict[str, Any]:
        failures = list(self.failures)
        checks_total = self.checks_total
        checks_passed = self.checks_passed
        if fatal_error is not None:
            checks_total += 1
            failures.append({"check": "fatal_error", "detail": fatal_error})
        return {
            "schema_version": 1,
            "status": "PASS" if not failures else "FAIL",
            "verifier": "paper/scripts/verify_v0_7_paper_artifacts.py",
            "target_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "checks_total": checks_total,
            "checks_passed": checks_passed,
            "checks_failed": len(failures),
            "numeric_comparisons": self.numeric_comparisons,
            "max_numeric_abs_error": self.max_numeric_abs_error,
            "validated": self.summary,
            "failures": failures,
        }


def reconstruct_validation_rows(verify: Verification) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intermittent_path = (
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/run_summaries.csv"
    )
    for source in read_csv(intermittent_path):
        if source.get("backbone") not in MODEL_ORDER:
            continue
        rows.append(
            {
                "dataset": "intermittent",
                "dataset_label": "Intermittent-5000",
                "model": source["backbone"],
                "model_label": MODEL_LABELS[source["backbone"]],
                "seed": int(source["seed"]),
                "joint_objective": float(source["best_val_joint_objective"]),
                "time_nll": float(source["best_val_time_nll"]),
                "log_qty_mse": float(source["best_val_log_qty_mse"]),
                "qty_mae": float(source["best_val_qty_mae"]),
                "qty_rmse": float(source["best_val_qty_rmse"]),
                "evaluation_scope": source["evaluation_scope"],
                "held_out_test_evaluated": parse_bool(source["held_out_test_evaluated"]),
                "status": source["status"],
                "source_revision": source["source_revision"],
            }
        )

    external_path = (
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/validation_seed_metrics.csv"
    )
    for source in read_csv(external_path):
        rows.append(
            {
                "dataset": source["dataset"],
                "dataset_label": source["dataset_label"],
                "model": source["model"],
                # The source bundle preserves the historical label "TitanTPP
                # Hard-LMM". The frozen paper table uses the canonical display
                # label, so normalize labels independently from the model key.
                "model_label": MODEL_LABELS[source["model"]],
                "seed": int(source["seed"]),
                "joint_objective": float(source["joint_objective"]),
                "time_nll": float(source["time_nll"]),
                "log_qty_mse": float(source["log_qty_mse"]),
                "qty_mae": float(source["qty_mae"]),
                "qty_rmse": float(source["qty_rmse"]),
                "evaluation_scope": source["evaluation_scope"],
                "held_out_test_evaluated": parse_bool(source["held_out_test_evaluated"]),
                "status": "success",
                "source_revision": source["source_revision"],
            }
        )

    expected_matrix = {
        (dataset, model, seed)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for seed in SEEDS
    }
    actual_keys = [(row["dataset"], row["model"], row["seed"]) for row in rows]
    actual_matrix = set(actual_keys)
    verify.equal(len(rows), 27, "validation.row_count")
    verify.equal(len(actual_keys), len(actual_matrix), "validation.no_duplicate_keys")
    verify.equal(actual_matrix, expected_matrix, "validation.exact_dataset_model_seed_matrix")
    for row in rows:
        key = f"{row['dataset']}.{row['model']}.seed{row['seed']}"
        verify.equal(row["dataset_label"], DATASETS[row["dataset"]]["label"], f"{key}.dataset_label")
        verify.equal(row["model_label"], MODEL_LABELS[row["model"]], f"{key}.model_label")
        verify.equal(row["status"], "success", f"{key}.status")
        verify.equal(row["evaluation_scope"], "validation_only", f"{key}.scope")
        verify.equal(row["held_out_test_evaluated"], False, f"{key}.held_out_false")
        verify.equal(row["source_revision"], SOURCE_REVISIONS[row["dataset"]], f"{key}.source_revision")
        verify.numeric(
            row["joint_objective"],
            row["time_nll"] + row["log_qty_mse"],
            f"{key}.joint_equals_time_plus_log_mse",
            abs_tol=1e-8,
            rel_tol=1e-12,
        )
    verify.summary["validation_rows_recomputed"] = len(rows)
    return rows


def verify_t3(verify: Verification, source_rows: list[dict[str, Any]]) -> None:
    path = ROOT / "paper/tables/T3_v0_7_backbone_validation.csv"
    table_rows = read_csv(path)
    expected_keys = {(dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER}
    actual_keys = [(row["dataset"], row["model"]) for row in table_rows]
    verify.equal(len(table_rows), 9, "t3.row_count")
    verify.equal(len(actual_keys), len(set(actual_keys)), "t3.no_duplicate_keys")
    verify.equal(set(actual_keys), expected_keys, "t3.exact_dataset_model_matrix")

    indexed = {(row["dataset"], row["model"]): row for row in table_rows}
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            key = f"t3.{dataset}.{model}"
            observed = indexed[(dataset, model)]
            selected = [
                row
                for row in source_rows
                if row["dataset"] == dataset and row["model"] == model
            ]
            verify.equal(len(selected), 3, f"{key}.source_seed_count")
            verify.equal(int(observed["n_seeds"]), 3, f"{key}.reported_seed_count")
            verify.equal(observed["dataset_label"], DATASETS[dataset]["label"], f"{key}.dataset_label")
            verify.equal(observed["model_label"], MODEL_LABELS[model], f"{key}.model_label")
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                verify.numeric(
                    observed[f"{metric}_mean"],
                    statistics.mean(values),
                    f"{key}.{metric}.mean",
                )
                verify.numeric(
                    observed[f"{metric}_std"],
                    statistics.stdev(values),
                    f"{key}.{metric}.sample_std",
                )
    verify.summary["t3_aggregate_rows_verified"] = len(table_rows)


def verify_t4(verify: Verification, source_rows: list[dict[str, Any]]) -> None:
    path = ROOT / "paper/tables/T4_v0_7_paired_titan_deltas.csv"
    table_rows = read_csv(path)
    expected_keys = {
        (dataset, reference, seed)
        for dataset in DATASET_ORDER
        for reference in REFERENCES
        for seed in SEEDS
    }
    actual_keys = [(row["dataset"], row["reference"], int(row["seed"])) for row in table_rows]
    verify.equal(len(table_rows), 18, "t4.row_count")
    verify.equal(len(actual_keys), len(set(actual_keys)), "t4.no_duplicate_keys")
    verify.equal(set(actual_keys), expected_keys, "t4.exact_dataset_reference_seed_matrix")

    source_index = {
        (row["dataset"], row["model"], row["seed"]): row for row in source_rows
    }
    for observed in table_rows:
        dataset = observed["dataset"]
        reference = observed["reference"]
        seed = int(observed["seed"])
        key = f"t4.{dataset}.titantpp_vs_{reference}.seed{seed}"
        verify.equal(observed["candidate"], "titantpp", f"{key}.candidate")
        verify.equal(observed["dataset_label"], DATASETS[dataset]["label"], f"{key}.dataset_label")
        candidate_row = source_index[(dataset, "titantpp", seed)]
        reference_row = source_index[(dataset, reference, seed)]
        for metric in PAIRED_METRICS:
            candidate = float(candidate_row[metric])
            baseline = float(reference_row[metric])
            expected_delta = candidate - baseline
            expected_relative = 100.0 * (baseline - candidate) / abs(baseline)
            expected_better = candidate < baseline
            verify.numeric(observed[f"{metric}_delta"], expected_delta, f"{key}.{metric}.delta")
            verify.numeric(
                observed[f"{metric}_relative_improvement_pct"],
                expected_relative,
                f"{key}.{metric}.relative_improvement_pct",
            )
            try:
                observed_better = parse_bool(observed[f"{metric}_better"])
            except ValueError:
                observed_better = None
            verify.equal(observed_better, expected_better, f"{key}.{metric}.better_flag")
    verify.summary["t4_paired_rows_verified"] = len(table_rows)


def reconstruct_quantity_strata() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intermittent_path = (
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_summary.csv"
    )
    for source in read_csv(intermittent_path):
        if source.get("backbone") not in MODEL_ORDER:
            continue
        model = source["backbone"]
        rows.append(
            {
                "dataset": "intermittent",
                "dataset_label": "Intermittent-5000",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "stratum_order": int(source["stratum_order"]),
                "stratum": source["stratum"],
                "stratum_label": source["stratum_label"],
                "count_per_seed": int(source["count"]),
                "share": float(source["share"]),
                "n_seeds": int(source["n_seeds"]),
                "qty_mae_mean": float(source["qty_mae_mean"]),
                "qty_mae_std": float(source["qty_mae_std"]),
                "qty_rmse_mean": float(source["qty_rmse_mean"]),
                "qty_rmse_std": float(source["qty_rmse_std"]),
            }
        )

    external_path = (
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_aggregate.csv"
    )
    for source in read_csv(external_path):
        model = source["model"]
        rows.append(
            {
                "dataset": source["dataset"],
                "dataset_label": source["dataset_label"],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "stratum_order": int(source["stratum_order"]),
                "stratum": source["stratum"],
                "stratum_label": source["stratum_label"],
                "count_per_seed": int(source["count_per_seed"]),
                "share": float(source["share"]),
                "n_seeds": int(source["n_seeds"]),
                "qty_mae_mean": float(source["qty_mae_mean"]),
                "qty_mae_std": float(source["qty_mae_std"]),
                "qty_rmse_mean": float(source["qty_rmse_mean"]),
                "qty_rmse_std": float(source["qty_rmse_std"]),
            }
        )
    return rows


def reconstruct_quantity_strata_seeds() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intermittent_path = (
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_seed_metrics.csv"
    )
    for source in read_csv(intermittent_path):
        if source.get("backbone") not in MODEL_ORDER:
            continue
        rows.append(
            {
                "dataset": "intermittent",
                "model": source["backbone"],
                "seed": int(source["seed"]),
                "stratum_order": int(source["stratum_order"]),
                "stratum": source["stratum"],
                "stratum_label": source["stratum_label"],
                "count": int(source["count"]),
                "share": float(source["share"]),
                "qty_mae": float(source["qty_mae"]),
                "qty_rmse": float(source["qty_rmse"]),
            }
        )

    external_path = (
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_seed_metrics.csv"
    )
    for source in read_csv(external_path):
        rows.append(
            {
                "dataset": source["dataset"],
                "model": source["model"],
                "seed": int(source["seed"]),
                "stratum_order": int(source["stratum_order"]),
                "stratum": source["stratum"],
                "stratum_label": source["stratum_label"],
                "count": int(source["count"]),
                "share": float(source["share"]),
                "qty_mae": float(source["qty_mae"]),
                "qty_rmse": float(source["qty_rmse"]),
            }
        )
    return rows


def verify_t5(verify: Verification) -> None:
    source_rows = reconstruct_quantity_strata()
    expected_keys = {
        (dataset, model, stratum_order)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for stratum_order in STRATA
    }
    source_keys = [
        (row["dataset"], row["model"], row["stratum_order"]) for row in source_rows
    ]
    verify.equal(len(source_rows), 45, "t5.source_row_count")
    verify.equal(len(source_keys), len(set(source_keys)), "t5.source_no_duplicate_keys")
    verify.equal(set(source_keys), expected_keys, "t5.source_exact_matrix")

    seed_rows = reconstruct_quantity_strata_seeds()
    expected_seed_keys = {
        (dataset, model, seed, stratum_order)
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
        for seed in SEEDS
        for stratum_order in STRATA
    }
    seed_keys = [
        (row["dataset"], row["model"], row["seed"], row["stratum_order"])
        for row in seed_rows
    ]
    verify.equal(len(seed_rows), 135, "t5.seed_source_row_count")
    verify.equal(len(seed_keys), len(set(seed_keys)), "t5.seed_source_no_duplicate_keys")
    verify.equal(set(seed_keys), expected_seed_keys, "t5.seed_source_exact_matrix")

    table_path = ROOT / "paper/tables/T5_v0_7_quantity_strata_validation.csv"
    table_rows = read_csv(table_path)
    table_keys = [
        (row["dataset"], row["model"], int(row["stratum_order"])) for row in table_rows
    ]
    verify.equal(len(table_rows), 45, "t5.table_row_count")
    verify.equal(len(table_keys), len(set(table_keys)), "t5.table_no_duplicate_keys")
    verify.equal(set(table_keys), expected_keys, "t5.table_exact_matrix")

    source_index = {
        (row["dataset"], row["model"], row["stratum_order"]): row
        for row in source_rows
    }
    seed_index = {
        (row["dataset"], row["model"], row["seed"], row["stratum_order"]): row
        for row in seed_rows
    }
    table_index = {
        (row["dataset"], row["model"], int(row["stratum_order"])): row
        for row in table_rows
    }
    validation_totals: dict[str, int] = {}
    for dataset in DATASET_ORDER:
        membership_rows: list[dict[str, Any]] = []
        for order, stratum in STRATA.items():
            selected = [
                source_index[(dataset, model, order)] for model in MODEL_ORDER
            ]
            key = f"t5.{dataset}.stratum{order}"
            membership = {
                (row["stratum"], row["stratum_label"], row["count_per_seed"], row["share"])
                for row in selected
            }
            verify.equal(len(membership), 1, f"{key}.membership_identical_across_models")
            verify.equal({row["n_seeds"] for row in selected}, {3}, f"{key}.three_seeds")
            verify.equal({row["stratum"] for row in selected}, {stratum}, f"{key}.stratum_name")
            membership_rows.append(selected[0])

            seed_selected = [
                seed_index[(dataset, model, seed, order)]
                for model in MODEL_ORDER
                for seed in SEEDS
            ]
            seed_membership = {
                (row["stratum"], row["stratum_label"], row["count"], row["share"])
                for row in seed_selected
            }
            verify.equal(
                len(seed_membership),
                1,
                f"{key}.seed_membership_identical_across_models_and_seeds",
            )
            summary_membership = next(iter(membership))
            verify.equal(
                next(iter(seed_membership)),
                summary_membership,
                f"{key}.seed_membership_matches_summary",
            )

        total = sum(int(row["count_per_seed"]) for row in membership_rows)
        validation_totals[dataset] = total
        verify.equal(total, DATASETS[dataset]["validation_rows"], f"t5.{dataset}.count_covers_validation")
        verify.numeric(
            sum(float(row["share"]) for row in membership_rows),
            1.0,
            f"t5.{dataset}.shares_sum_to_one",
        )
        for row in membership_rows:
            verify.numeric(
                row["share"],
                row["count_per_seed"] / total,
                f"t5.{dataset}.stratum{row['stratum_order']}.share_matches_count",
            )

    for key_tuple in sorted(
        expected_keys,
        key=lambda item: (
            DATASET_ORDER.index(item[0]),
            item[2],
            MODEL_ORDER.index(item[1]),
        ),
    ):
        expected = source_index[key_tuple]
        observed = table_index[key_tuple]
        dataset, model, order = key_tuple
        key = f"t5.{dataset}.{model}.stratum{order}"
        selected_seeds = [seed_index[(dataset, model, seed, order)] for seed in SEEDS]
        verify.equal(len(selected_seeds), 3, f"{key}.seed_count")
        for metric in ("qty_mae", "qty_rmse"):
            values = [row[metric] for row in selected_seeds]
            verify.numeric(
                expected[f"{metric}_mean"],
                statistics.mean(values),
                f"{key}.{metric}.source_mean_from_seeds",
            )
            verify.numeric(
                expected[f"{metric}_std"],
                statistics.stdev(values),
                f"{key}.{metric}.source_sample_std_from_seeds",
            )
        for field in ("dataset_label", "model_label", "stratum", "stratum_label"):
            verify.equal(observed[field], expected[field], f"{key}.{field}")
        for field in ("stratum_order", "count_per_seed", "n_seeds"):
            verify.equal(int(observed[field]), int(expected[field]), f"{key}.{field}")
        for field in (
            "share",
            "qty_mae_mean",
            "qty_mae_std",
            "qty_rmse_mean",
            "qty_rmse_std",
        ):
            verify.numeric(observed[field], expected[field], f"{key}.{field}")
    verify.summary["t5_quantity_strata_rows_verified"] = len(table_rows)
    verify.summary["t5_quantity_strata_seed_rows_recomputed"] = len(seed_rows)
    verify.summary["t5_validation_counts_per_dataset"] = validation_totals


def manifest_split_counts(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["chronological_split"]): int(row["rows"])
        for row in payload["summary"]["split_counts"]
    }


def recompute_dataset_statistics(spec: dict[str, Any]) -> dict[str, Any]:
    data_path = ROOT / spec["data_path"]
    scan = pl.scan_parquet(data_path)
    identity = scan.select(["oper_part_no", "chronological_split"]).collect()
    split_counts = {
        str(row["chronological_split"]): int(row["len"])
        for row in identity.group_by("chronological_split").len().to_dicts()
    }
    train_scan = scan.filter(pl.col("chronological_split") == "train")
    sequence_lengths = (
        train_scan.group_by("oper_part_no").len().select("len").collect()["len"]
    )
    quantity = (
        train_scan.select(
            pl.col("demand_qty").median().alias("median"),
            pl.col("demand_qty").quantile(0.95, interpolation="nearest").alias("p95"),
            pl.col("demand_qty").max().alias("max"),
        )
        .collect()
        .row(0, named=True)
    )
    return {
        "sequences": int(identity["oper_part_no"].n_unique()),
        "event_rows": identity.height,
        "train_rows": split_counts.get("train"),
        "validation_rows": split_counts.get("validation"),
        "test_rows": split_counts.get("test"),
        "sequence_length_median": float(sequence_lengths.median()),
        "sequence_length_p95": float(sequence_lengths.quantile(0.95, interpolation="nearest")),
        "sequence_length_max": int(sequence_lengths.max()),
        "quantity_median": float(quantity["median"]),
        "quantity_p95": float(quantity["p95"]),
        "quantity_max": float(quantity["max"]),
        "split_counts": split_counts,
    }


def verify_t1(verify: Verification) -> None:
    path = ROOT / "paper/tables/T1_v0_7_dataset_statistics.csv"
    rows = read_csv(path)
    actual_datasets = [row["dataset"] for row in rows]
    verify.equal(len(rows), 3, "t1.row_count")
    verify.equal(len(actual_datasets), len(set(actual_datasets)), "t1.no_duplicate_datasets")
    verify.equal(set(actual_datasets), set(DATASET_ORDER), "t1.exact_dataset_set")
    indexed = {row["dataset"]: row for row in rows}

    for dataset in DATASET_ORDER:
        spec = DATASETS[dataset]
        observed = indexed[dataset]
        key = f"t1.{dataset}"
        data_path = ROOT / spec["data_path"]
        split_path = ROOT / spec["split_path"]
        data_hash = sha256_file(data_path)
        split_hash = sha256_file(split_path)
        verify.equal(data_hash, spec["data_sha256"], f"{key}.frozen_data_sha256")
        verify.equal(split_hash, spec["split_sha256"], f"{key}.frozen_split_manifest_sha256")
        verify.equal(observed["data_sha256"], data_hash, f"{key}.reported_data_sha256")
        verify.equal(
            observed["split_manifest_sha256"],
            split_hash,
            f"{key}.reported_split_manifest_sha256",
        )
        verify.equal(observed["dataset_label"], spec["label"], f"{key}.dataset_label")
        verify.equal(observed["sequence_unit"], spec["sequence_unit"], f"{key}.sequence_unit")
        verify.equal(observed["time_unit"], spec["time_unit"], f"{key}.time_unit")
        verify.equal(observed["distribution_statistics_scope"], "train_only", f"{key}.stats_scope")
        verify.equal(int(observed["lookback"]), spec["lookback"], f"{key}.lookback")
        verify.equal(
            int(observed["max_sequence_length"]),
            spec["max_sequence_length"],
            f"{key}.max_sequence_length",
        )

        computed = recompute_dataset_statistics(spec)
        verify.equal(set(computed["split_counts"]), {"train", "validation", "test"}, f"{key}.split_labels")
        declared_counts = manifest_split_counts(split_path)
        verify.equal(declared_counts, computed["split_counts"], f"{key}.split_manifest_matches_parquet")
        for field in (
            "sequences",
            "event_rows",
            "train_rows",
            "validation_rows",
            "test_rows",
            "sequence_length_max",
        ):
            verify.equal(int(observed[field]), int(computed[field]), f"{key}.{field}")
        for field in (
            "sequence_length_median",
            "sequence_length_p95",
            "quantity_median",
            "quantity_p95",
            "quantity_max",
        ):
            verify.numeric(observed[field], computed[field], f"{key}.{field}")
    verify.summary["t1_dataset_rows_verified"] = len(rows)


def validate_png(path: Path) -> tuple[bool, str, dict[str, int]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False, "invalid PNG signature", {}
    position = 8
    width = height = 0
    chunks: list[bytes] = []
    try:
        while position + 12 <= len(data):
            length = struct.unpack(">I", data[position : position + 4])[0]
            chunk_type = data[position + 4 : position + 8]
            chunk_data = data[position + 8 : position + 8 + length]
            stored_crc = struct.unpack(">I", data[position + 8 + length : position + 12 + length])[0]
            calculated_crc = zlib.crc32(chunk_type)
            calculated_crc = zlib.crc32(chunk_data, calculated_crc) & 0xFFFFFFFF
            if stored_crc != calculated_crc:
                return False, f"CRC mismatch in {chunk_type!r}", {}
            chunks.append(chunk_type)
            if chunk_type == b"IHDR":
                if length != 13:
                    return False, "invalid IHDR length", {}
                width, height = struct.unpack(">II", chunk_data[:8])
            position += length + 12
            if chunk_type == b"IEND":
                break
    except (IndexError, struct.error) as error:
        return False, f"malformed PNG chunks: {error}", {}
    valid = chunks[:1] == [b"IHDR"] and b"IDAT" in chunks and chunks[-1:] == [b"IEND"]
    valid = valid and width > 0 and height > 0 and position == len(data)
    return valid, f"width={width}; height={height}; chunks={len(chunks)}", {"width": width, "height": height}


def validate_pdf(path: Path) -> tuple[bool, str]:
    data = path.read_bytes()
    page_marker_count = data.count(b"/Type /Page") - data.count(b"/Type /Pages")
    valid = (
        data.startswith(b"%PDF-")
        and b"%%EOF" in data[-2048:]
        and b"xref" in data
        and page_marker_count >= 1
    )
    return valid, f"bytes={len(data)}; page_markers={page_marker_count}"


def validate_svg(path: Path) -> tuple[bool, str]:
    data = path.read_bytes()
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        return False, f"XML parse error: {error}"
    tag = root.tag.rsplit("}", 1)[-1]
    has_extent = "viewBox" in root.attrib or (
        "width" in root.attrib and "height" in root.attrib
    )
    valid = tag == "svg" and has_extent and len(data) > 1024
    return valid, f"bytes={len(data)}; root={tag}; has_extent={has_extent}"


def verify_figures(verify: Verification) -> None:
    dimensions: dict[str, dict[str, int]] = {}
    figure_stems = (
        "paper/figures/F1_v0_7_mark_free_architecture",
        "paper/figures/F2_v0_7_dataset_validation_errors",
    )
    count = 0
    for stem in figure_stems:
        png_path = ROOT / f"{stem}.png"
        pdf_path = ROOT / f"{stem}.pdf"
        svg_path = ROOT / f"{stem}.svg"
        png_ok, png_detail, png_dimensions = validate_png(png_path)
        pdf_ok, pdf_detail = validate_pdf(pdf_path)
        svg_ok, svg_detail = validate_svg(svg_path)
        verify.check(png_ok, f"figure.{Path(stem).name}.png", png_detail)
        verify.check(pdf_ok, f"figure.{Path(stem).name}.pdf", pdf_detail)
        verify.check(svg_ok, f"figure.{Path(stem).name}.svg", svg_detail)
        dimensions[Path(stem).name] = png_dimensions
        count += 3
    verify.summary["figure_files_verified"] = count
    verify.summary["png_dimensions"] = dimensions


def extract_markdown_table(text: str, header_prefix: str) -> list[list[str]]:
    lines = text.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith(header_prefix)
        )
    except StopIteration as error:
        raise ValueError(f"Markdown table header not found: {header_prefix}") from error
    if header_index + 1 >= len(lines) or not lines[header_index + 1].startswith("| ---"):
        raise ValueError(f"Markdown table separator missing after: {header_prefix}")
    rows: list[list[str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def parse_markdown_integer(value: str) -> int:
    return int(value.replace(",", "").strip())


def parse_markdown_metric(value: str) -> tuple[float, float]:
    normalized = value.replace("**", "").replace("$", "")
    normalized = normalized.replace("\\pm", "+/-").replace("±", "+/-")
    pieces = [piece.strip() for piece in normalized.split("+/-")]
    if len(pieces) != 2:
        raise ValueError(f"Expected mean +/- std, received {value!r}")
    return float(pieces[0]), float(pieces[1])


def normalize_root_publication_links(text: str) -> str:
    return re.sub(r"(\]\()paper/(?=(?:figures|tables)/)", r"\1", text)


def verify_publication_content(verify: Verification) -> None:
    manuscript_path = ROOT / "paper/titantpp_short_paper_draft_v0_7_manuscript.md"
    readme_path = ROOT / "README.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    root_readme = readme_path.read_text(encoding="utf-8")

    t1_rows = extract_markdown_table(
        manuscript,
        "| Dataset | Sequences | Event rows | Train / validation / test rows |",
    )
    verify.equal(len(t1_rows), 3, "publication.manuscript.t1.row_count")
    t1_source = {
        row["dataset_label"]: row
        for row in read_csv(ROOT / "paper/tables/T1_v0_7_dataset_statistics.csv")
    }
    seen_t1_labels: list[str] = []
    for row in t1_rows:
        verify.equal(len(row), 6, f"publication.manuscript.t1.columns.row{len(seen_t1_labels)}")
        if len(row) != 6:
            continue
        label = row[0]
        seen_t1_labels.append(label)
        verify.check(label in t1_source, f"publication.manuscript.t1.known_dataset.{label}", label)
        if label not in t1_source:
            continue
        expected = t1_source[label]
        key = f"publication.manuscript.t1.{label}"
        verify.equal(parse_markdown_integer(row[1]), int(expected["sequences"]), f"{key}.sequences")
        verify.equal(parse_markdown_integer(row[2]), int(expected["event_rows"]), f"{key}.event_rows")
        split_values = [parse_markdown_integer(value) for value in row[3].split("/")]
        verify.equal(
            split_values,
            [int(expected["train_rows"]), int(expected["validation_rows"]), int(expected["test_rows"])],
            f"{key}.split_counts",
        )
        length_values = [parse_markdown_integer(value) for value in row[4].split("/")]
        verify.equal(
            length_values,
            [
                int(float(expected["sequence_length_median"])),
                int(float(expected["sequence_length_p95"])),
                int(expected["sequence_length_max"]),
            ],
            f"{key}.train_sequence_statistics",
        )
        quantity_values = [parse_markdown_integer(value) for value in row[5].split("/")]
        verify.equal(
            quantity_values,
            [
                int(float(expected["quantity_median"])),
                int(float(expected["quantity_p95"])),
                int(float(expected["quantity_max"])),
            ],
            f"{key}.train_quantity_statistics",
        )
    verify.equal(
        seen_t1_labels,
        [DATASETS[dataset]["label"] for dataset in DATASET_ORDER],
        "publication.manuscript.t1.dataset_order_and_labels",
    )

    t3_rows = extract_markdown_table(
        manuscript,
        "| Dataset | Model | Clamped time loss",
    )
    verify.equal(len(t3_rows), 9, "publication.manuscript.t3.row_count")
    t3_source = {
        (row["dataset_label"], row["model"]): row
        for row in read_csv(ROOT / "paper/tables/T3_v0_7_backbone_validation.csv")
    }
    display_models = {"RMTPP": "rmtpp", "THP": "thp", "TitanTPP": "titantpp"}
    last_dataset = ""
    seen_t3_keys: list[tuple[str, str]] = []
    metric_columns = (
        ("time_nll", 6),
        ("log_qty_mse", 6),
        ("qty_mae", 4),
        ("qty_rmse", 4),
    )
    for row_number, row in enumerate(t3_rows):
        verify.equal(len(row), 6, f"publication.manuscript.t3.columns.row{row_number}")
        if len(row) != 6:
            continue
        if row[0]:
            last_dataset = row[0]
        model_display = row[1]
        verify.check(
            model_display in display_models,
            f"publication.manuscript.t3.known_model.row{row_number}",
            model_display,
        )
        if model_display not in display_models:
            continue
        model = display_models[model_display]
        source_key = (last_dataset, model)
        seen_t3_keys.append(source_key)
        verify.check(
            source_key in t3_source,
            f"publication.manuscript.t3.known_row.{last_dataset}.{model}",
            str(source_key),
        )
        if source_key not in t3_source:
            continue
        expected = t3_source[source_key]
        for column_index, (metric, decimals) in enumerate(metric_columns, start=2):
            mean, sample_std = parse_markdown_metric(row[column_index])
            tolerance = 0.51 * 10 ** (-decimals)
            verify.numeric(
                mean,
                float(expected[f"{metric}_mean"]),
                f"publication.manuscript.t3.{last_dataset}.{model}.{metric}.mean",
                abs_tol=tolerance,
                rel_tol=0.0,
            )
            verify.numeric(
                sample_std,
                float(expected[f"{metric}_std"]),
                f"publication.manuscript.t3.{last_dataset}.{model}.{metric}.sample_std",
                abs_tol=tolerance,
                rel_tol=0.0,
            )
    verify.equal(set(seen_t3_keys), set(t3_source), "publication.manuscript.t3.exact_rows")
    verify.equal(len(seen_t3_keys), len(set(seen_t3_keys)), "publication.manuscript.t3.no_duplicates")

    t5_link = "[complete three-dataset stratum table](tables/T5_v0_7_quantity_strata_validation.md)"
    verify.equal(manuscript.count(t5_link), 1, "publication.manuscript.t5_link")
    verify.check(
        manuscript.count("Intermittent-5000") >= 5,
        "publication.manuscript.intermittent_5000_label",
        f"occurrences={manuscript.count('Intermittent-5000')}",
    )
    verify.check(
        "On the Intermittent-5000 validation split" in manuscript,
        "publication.manuscript.abstract_uses_frozen_dataset_label",
    )
    verify.equal(
        normalize_root_publication_links(root_readme),
        manuscript,
        "publication.root_readme_matches_manuscript_after_link_normalization",
    )
    verify.summary["publication_t1_rows_verified"] = len(t1_rows)
    verify.summary["publication_t3_rows_verified"] = len(t3_rows)
    verify.summary["publication_documents_content_verified"] = 2


def safe_manifest_path(relative_path: str) -> Path | None:
    candidate = (ROOT / relative_path).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return candidate


def verify_manifest(verify: Verification) -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    verify.equal(payload.get("schema_version"), 1, "manifest.schema_version")
    verify.equal(payload.get("status"), "validation_freeze_complete_held_out_locked", "manifest.status")
    verify.equal(payload.get("model_identity"), "original mark-free Hard-LMM / Count-aware TitanTPP-T0", "manifest.model_identity")
    verify.equal(payload.get("main_models"), list(MODEL_ORDER), "manifest.main_models")
    verify.equal(payload.get("datasets"), list(DATASET_ORDER), "manifest.datasets")
    verify.equal(payload.get("seeds"), list(SEEDS), "manifest.seeds")
    verify.equal(payload.get("validation_run_count"), 27, "manifest.validation_run_count")
    verify.equal(payload.get("evaluation_scope"), "validation_only", "manifest.evaluation_scope")
    verify.equal(payload.get("held_out_test_evaluated"), False, "manifest.held_out_false")
    verify.equal(
        payload.get("quantity_interface_empirical_claim"),
        "excluded_contract_mismatch",
        "manifest.quantity_interface_claim",
    )
    verify.equal(
        payload.get("training_source_revisions"),
        SOURCE_REVISIONS,
        "manifest.training_source_revisions",
    )
    verify.equal(
        payload.get("time_metric"),
        {
            "artifact_field": "time_nll",
            "paper_label": "clamped time loss",
            "executed_intercept_cap": 300.0,
            "wd_cap": 10.0,
            "exact_likelihood": False,
        },
        "manifest.time_metric_contract",
    )

    sources = payload.get("sources", [])
    outputs = payload.get("outputs", [])
    publication_documents = payload.get("publication_documents", [])
    verify.equal(len(outputs), 17, "manifest.output_count")
    output_paths = [str(row.get("path")) for row in outputs]
    verify.equal(len(output_paths), len(set(output_paths)), "manifest.no_duplicate_outputs")
    verify.equal(set(output_paths), EXPECTED_OUTPUTS, "manifest.exact_output_set")
    source_paths = [str(row.get("path")) for row in sources]
    verify.equal(len(source_paths), len(set(source_paths)), "manifest.no_duplicate_sources")
    verify.check(
        REQUIRED_STRATA_SOURCES.issubset(set(source_paths)),
        "manifest.includes_all_t5_provenance",
        f"missing={sorted(REQUIRED_STRATA_SOURCES - set(source_paths))}",
    )

    document_paths = [str(row.get("path")) for row in publication_documents]
    verify.equal(len(publication_documents), 2, "manifest.publication_document_count")
    verify.equal(
        len(document_paths),
        len(set(document_paths)),
        "manifest.no_duplicate_publication_documents",
    )
    verify.equal(
        set(document_paths),
        EXPECTED_PUBLICATION_DOCUMENTS,
        "manifest.exact_publication_document_set",
    )
    verify.equal(
        set(document_paths).isdisjoint(set(source_paths) | set(output_paths)),
        True,
        "manifest.publication_documents_are_separate",
    )

    for kind, rows in (
        ("source", sources),
        ("output", outputs),
        ("document", publication_documents),
    ):
        for index, row in enumerate(rows):
            relative_path = str(row.get("path", ""))
            key = f"manifest.{kind}.{index}.{relative_path}"
            path = safe_manifest_path(relative_path)
            verify.check(path is not None, f"{key}.inside_repository", relative_path)
            if path is None:
                continue
            verify.check(path.is_file(), f"{key}.exists", str(path))
            if not path.is_file():
                continue
            verify.equal(path.stat().st_size, int(row.get("bytes", -1)), f"{key}.bytes")
            verify.equal(sha256_file(path), row.get("sha256"), f"{key}.sha256")

    manifest_csv_rows = read_csv(MANIFEST_CSV_PATH)
    json_rows = [
        {"kind": kind, "path": str(row["path"]), "sha256": str(row["sha256"]), "bytes": str(row["bytes"])}
        for kind, rows in (
            ("source", sources),
            ("output", outputs),
            ("document", publication_documents),
        )
        for row in rows
    ]
    entry_key = lambda row: (row["kind"], row["path"], row["sha256"], row["bytes"])
    verify.equal(
        sorted(manifest_csv_rows, key=entry_key),
        sorted(json_rows, key=entry_key),
        "manifest.csv_matches_json_entries",
    )
    verify.summary["manifest_sources_verified"] = len(sources)
    verify.summary["manifest_outputs_verified"] = len(outputs)
    verify.summary["manifest_publication_documents_verified"] = len(publication_documents)


def run_verification(verify: Verification) -> None:
    source_rows = reconstruct_validation_rows(verify)
    verify_t3(verify, source_rows)
    verify_t4(verify, source_rows)
    verify_t5(verify)
    verify_t1(verify)
    verify_figures(verify)
    verify_publication_content(verify)
    verify_manifest(verify)


def main() -> int:
    verify = Verification()
    fatal_error: str | None = None
    try:
        run_verification(verify)
    except Exception as error:  # The deterministic report records unexpected failures.
        fatal_error = f"{type(error).__name__}: {error}"
    payload = verify.report(fatal_error)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{payload['status']}: {payload['checks_passed']}/{payload['checks_total']} checks; "
        f"numeric comparisons={payload['numeric_comparisons']}; "
        f"max abs error={payload['max_numeric_abs_error']:.17g}"
    )
    if payload["status"] != "PASS":
        for failure in payload["failures"]:
            print(f"- {failure['check']}: {failure['detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
