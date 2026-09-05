#!/usr/bin/env python3
"""Build the frozen v0.7 paper tables, figures, manuscript mirror, and manifest."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

_PLOT_CACHE = Path(tempfile.gettempdir()) / "titantpp-paper-plot-cache"
(_PLOT_CACHE / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE))

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "paper/tables"
FIGURE_DIR = ROOT / "paper/figures"
FIGURE_DATA_DIR = FIGURE_DIR / "source_data"
MANIFEST_DIR = ROOT / "paper/manifests"
MANUSCRIPT_PATH = ROOT / "paper/titantpp_short_paper_draft_v0_7_manuscript.md"
ROOT_README_PATH = ROOT / "README.md"

SEEDS = (42, 52, 62)
MODEL_ORDER = ("rmtpp", "thp", "titantpp")
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
DATASETS = (
    {
        "dataset": "intermittent",
        "label": "Intermittent-5000",
        "path": ROOT / "sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet",
        "expected_sha256": "85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f",
        "split_manifest": ROOT / "sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json",
        "expected_split_sha256": "393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04",
        "sequence_unit": "part",
        "time_unit": "week",
        "lookback": 520,
        "max_sequence_length": 256,
    },
    {
        "dataset": "taxi",
        "label": "Taxi",
        "path": ROOT / "sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet",
        "expected_sha256": "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46",
        "split_manifest": ROOT / "sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json",
        "expected_split_sha256": "4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85",
        "sequence_unit": "0.02-degree pickup grid cell",
        "time_unit": "hour",
        "lookback": 168,
        "max_sequence_length": 256,
    },
    {
        "dataset": "instacart",
        "label": "Instacart",
        "path": ROOT / "sample_data/insta_market_basket/instacart_marked_target_with_split.parquet",
        "expected_sha256": "06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2",
        "split_manifest": ROOT / "sample_data/insta_market_basket/instacart_marked_target_split_manifest.json",
        "expected_split_sha256": "6c6cdd41f847878fbb405b73dfa038fbb7a88ad53df6843b0cc9e64531a8b71d",
        "sequence_unit": "user",
        "time_unit": "day",
        "lookback": 52,
        "max_sequence_length": 64,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sync_root_readme() -> None:
    """Mirror the frozen manuscript at the repository root with root-relative links."""
    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    root_copy = manuscript.replace("](figures/", "](paper/figures/").replace(
        "](tables/", "](paper/tables/"
    )
    if "](figures/" in root_copy or "](tables/" in root_copy:
        raise AssertionError("Root README contains unresolved manuscript-relative artifact links")
    ROOT_README_PATH.write_text(root_copy, encoding="utf-8")


def sample_mean(rows: list[dict[str, Any]], metric: str) -> float:
    return statistics.mean(float(row[metric]) for row in rows)


def sample_std(rows: list[dict[str, Any]], metric: str) -> float:
    return statistics.stdev(float(row[metric]) for row in rows)


def dataset_statistics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in DATASETS:
        data_path = spec["path"]
        split_path = spec["split_manifest"]
        actual_data_hash = sha256_file(data_path)
        actual_split_hash = sha256_file(split_path)
        if actual_data_hash != spec["expected_sha256"]:
            raise ValueError(f"Dataset hash mismatch: {data_path}")
        if actual_split_hash != spec["expected_split_sha256"]:
            raise ValueError(f"Split hash mismatch: {split_path}")

        identity_frame = pl.read_parquet(
            data_path,
            columns=["oper_part_no", "chronological_split"],
        )
        split_counts = {
            str(item["chronological_split"]): int(item["len"])
            for item in identity_frame.group_by("chronological_split").len().to_dicts()
        }
        if set(split_counts) != {"train", "validation", "test"}:
            raise ValueError(f"Unexpected split labels in {data_path}: {split_counts}")
        train_frame = (
            pl.scan_parquet(data_path)
            .filter(pl.col("chronological_split") == "train")
            .select(["oper_part_no", "demand_qty"])
            .collect()
        )
        lengths = train_frame.group_by("oper_part_no").len()["len"]
        rows.append(
            {
                "dataset": spec["dataset"],
                "dataset_label": spec["label"],
                "sequence_unit": spec["sequence_unit"],
                "time_unit": spec["time_unit"],
                "sequences": identity_frame["oper_part_no"].n_unique(),
                "event_rows": identity_frame.height,
                "train_rows": split_counts["train"],
                "validation_rows": split_counts["validation"],
                "test_rows": split_counts["test"],
                "distribution_statistics_scope": "train_only",
                "sequence_length_median": float(lengths.median()),
                "sequence_length_p95": float(lengths.quantile(0.95, interpolation="nearest")),
                "sequence_length_max": int(lengths.max()),
                "quantity_median": float(train_frame["demand_qty"].median()),
                "quantity_p95": float(
                    train_frame["demand_qty"].quantile(0.95, interpolation="nearest")
                ),
                "quantity_max": float(train_frame["demand_qty"].max()),
                "lookback": spec["lookback"],
                "max_sequence_length": spec["max_sequence_length"],
                "data_sha256": actual_data_hash,
                "split_manifest_sha256": actual_split_hash,
            }
        )
    return rows


def validation_seed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intermittent_path = (
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/run_summaries.csv"
    )
    for source in read_csv(intermittent_path):
        if source["backbone"] not in MODEL_ORDER:
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
                "best_epoch": int(source["best_epoch"]),
                "checkpoint_state_sha256": source["checkpoint_state_sha256"],
                "source_revision": source["source_revision"],
                "evaluation_scope": source["evaluation_scope"],
                "held_out_test_evaluated": source["held_out_test_evaluated"].lower(),
            }
        )

    external_path = (
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/validation_seed_metrics.csv"
    )
    for source in read_csv(external_path):
        model = source["model"]
        rows.append(
            {
                "dataset": source["dataset"],
                "dataset_label": source["dataset_label"],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seed": int(source["seed"]),
                "joint_objective": float(source["joint_objective"]),
                "time_nll": float(source["time_nll"]),
                "log_qty_mse": float(source["log_qty_mse"]),
                "qty_mae": float(source["qty_mae"]),
                "qty_rmse": float(source["qty_rmse"]),
                "best_epoch": int(source["best_epoch"]),
                "checkpoint_state_sha256": source["checkpoint_state_sha256"],
                "source_revision": source["source_revision"],
                "evaluation_scope": source["evaluation_scope"],
                "held_out_test_evaluated": source["held_out_test_evaluated"].lower(),
            }
        )

    expected = {
        (dataset, model, seed)
        for dataset in ("intermittent", "taxi", "instacart")
        for model in MODEL_ORDER
        for seed in SEEDS
    }
    actual = {(row["dataset"], row["model"], row["seed"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"Unexpected validation rows. Missing={expected-actual}; extra={actual-expected}")
    for row in rows:
        if row["evaluation_scope"] != "validation_only":
            raise ValueError(f"Non-validation row: {row}")
        if row["held_out_test_evaluated"] != "false":
            raise ValueError(f"Held-out row detected: {row}")
        if row["source_revision"] != SOURCE_REVISIONS[row["dataset"]]:
            raise ValueError(f"Unexpected source revision: {row}")
        if not math.isclose(
            row["joint_objective"], row["time_nll"] + row["log_qty_mse"], abs_tol=1e-8
        ):
            raise ValueError(f"Joint objective mismatch: {row}")
    return sorted(rows, key=lambda row: (row["dataset"], MODEL_ORDER.index(row["model"]), row["seed"]))


def aggregate_validation(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_order = ("intermittent", "taxi", "instacart")
    for dataset in dataset_order:
        for model in MODEL_ORDER:
            selected = [
                row for row in seed_rows if row["dataset"] == dataset and row["model"] == model
            ]
            if len(selected) != 3:
                raise ValueError(f"Expected three seeds for {dataset}/{model}")
            row: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": selected[0]["dataset_label"],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_seeds": 3,
            }
            for metric in ("joint_objective", "time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                row[f"{metric}_mean"] = sample_mean(selected, metric)
                row[f"{metric}_std"] = sample_std(selected, metric)
            rows.append(row)
    return rows


def paired_deltas(seed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    metrics = ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse")
    for dataset in ("intermittent", "taxi", "instacart"):
        for reference in ("rmtpp", "thp"):
            paired: list[dict[str, Any]] = []
            for seed in SEEDS:
                candidate_row = next(
                    row
                    for row in seed_rows
                    if row["dataset"] == dataset
                    and row["model"] == "titantpp"
                    and row["seed"] == seed
                )
                reference_row = next(
                    row
                    for row in seed_rows
                    if row["dataset"] == dataset
                    and row["model"] == reference
                    and row["seed"] == seed
                )
                item: dict[str, Any] = {
                    "dataset": dataset,
                    "dataset_label": candidate_row["dataset_label"],
                    "candidate": "titantpp",
                    "reference": reference,
                    "seed": seed,
                }
                for metric in metrics:
                    candidate = float(candidate_row[metric])
                    baseline = float(reference_row[metric])
                    item[f"{metric}_delta"] = candidate - baseline
                    item[f"{metric}_relative_improvement_pct"] = (
                        100.0 * (baseline - candidate) / abs(baseline)
                    )
                    item[f"{metric}_better"] = candidate < baseline
                detail.append(item)
                paired.append(item)
            aggregate: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": paired[0]["dataset_label"],
                "candidate": "titantpp",
                "reference": reference,
                "n_paired_seeds": 3,
            }
            for metric in metrics:
                aggregate[f"{metric}_mean_delta"] = statistics.mean(
                    float(row[f"{metric}_delta"]) for row in paired
                )
                aggregate[f"{metric}_better_seed_count"] = sum(
                    bool(row[f"{metric}_better"]) for row in paired
                )
            summary.append(aggregate)
    return detail, summary


def quantity_strata_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intermittent_path = (
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_summary.csv"
    )
    for source in read_csv(intermittent_path):
        model = source["backbone"]
        if model not in MODEL_ORDER:
            continue
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

    expected = {
        (dataset, model, stratum_order)
        for dataset in ("intermittent", "taxi", "instacart")
        for model in MODEL_ORDER
        for stratum_order in range(5)
    }
    actual = {
        (row["dataset"], row["model"], row["stratum_order"])
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"Unexpected quantity-strata rows. Missing={expected-actual}; extra={actual-expected}")
    for dataset in ("intermittent", "taxi", "instacart"):
        for order in range(5):
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["stratum_order"] == order
            ]
            if {row["n_seeds"] for row in selected} != {3}:
                raise ValueError(f"Quantity-strata seed count mismatch: {dataset}/{order}")
            if len({(row["count_per_seed"], row["share"]) for row in selected}) != 1:
                raise ValueError(f"Quantity-strata membership mismatch: {dataset}/{order}")
    return sorted(
        rows,
        key=lambda row: (
            ("intermittent", "taxi", "instacart").index(row["dataset"]),
            row["stratum_order"],
            MODEL_ORDER.index(row["model"]),
        ),
    )


def model_contract_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoders = {
        "rmtpp": "one-layer GRU",
        "thp": "two-layer causal Transformer",
        "titantpp": "two-layer causal memory attention + static Hard-LMM",
    }
    rows: list[dict[str, Any]] = []
    for dataset in dataset_rows:
        for model in MODEL_ORDER:
            rows.append(
                {
                    "dataset": dataset["dataset"],
                    "dataset_label": dataset["dataset_label"],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "encoder": encoders[model],
                    "hidden_dimension": 64,
                    "history_input": "log1p(delta_t) + log1p(raw_quantity)",
                    "quantity_target_loss": "log1p(raw_quantity) + direct MSE",
                    "time_head": "legacy_clamped_rmtpp (executed intercept cap 300; w*dt cap 10)",
                    "source_revision": SOURCE_REVISIONS[dataset["dataset"]],
                    "lookback": dataset["lookback"],
                    "time_unit": dataset["time_unit"],
                    "max_sequence_length": dataset["max_sequence_length"],
                    "seeds": "42|52|62",
                    "checkpoint": "minimum validation joint objective",
                    "held_out_test_evaluated": "false",
                }
            )
    return rows


def write_markdown_tables(
    dataset_rows: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    quantity_rows: list[dict[str, Any]],
) -> None:
    t1 = [
        "# T1 v0.7. Frozen dataset statistics",
        "",
        "> Population and split counts come from the frozen split identity. Sequence-length and quantity distributions use train rows only; no held-out target statistic or model performance is included.",
        "",
        "| Dataset | Sequences | Event rows | Train / validation / test | Train sequence length med. / p95 / max | Train quantity med. / p95 / max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_rows:
        t1.append(
            f"| {row['dataset_label']} | {row['sequences']:,} | {row['event_rows']:,} | "
            f"{row['train_rows']:,} / {row['validation_rows']:,} / {row['test_rows']:,} | "
            f"{row['sequence_length_median']:g} / {row['sequence_length_p95']:g} / {row['sequence_length_max']:,} | "
            f"{row['quantity_median']:g} / {row['quantity_p95']:g} / {row['quantity_max']:g} |"
        )
    t1.extend(
        [
            "",
            "The Intermittent row refers to the frozen 5,000-series experiment population. It supersedes the earlier full-corpus row for v0.7 model-result reporting.",
            "",
        ]
    )
    (TABLE_DIR / "T1_v0_7_dataset_statistics.md").write_text("\n".join(t1), encoding="utf-8")

    t2 = [
        "# T2 v0.7. Matched model and training contract",
        "",
        "> Frozen for validation reporting. Models differ only in the history encoder within each dataset.",
        "",
        "| Dataset | Model | Encoder | Lookback / max length | Shared input and quantity loss |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in contract_rows:
        t2.append(
            f"| {row['dataset_label']} | {row['model_label']} | {row['encoder']} | "
            f"{row['lookback']} {row['time_unit']} / {row['max_sequence_length']} | "
            "`log1p(dt)`, `log1p(q)` / direct log-MSE |"
        )
    t2.extend(
        [
            "",
            "All rows use hidden dimension 64, seeds 42/52/62, AdamW at 0.001, batch size 128, gradient clipping 1.0, an e300 ceiling, minimum 40 epochs, patience 40, and minimum validation joint objective checkpoint selection.",
            "The executed legacy time score caps the intercept at 300 and `w * delta_t` at 10. The source-artifact field `time_nll` is reported as clamped time loss, not exact NLL.",
            "",
        ]
    )
    (TABLE_DIR / "T2_v0_7_model_training_contract.md").write_text("\n".join(t2), encoding="utf-8")

    t3 = [
        "# T3 v0.7. Three-seed mark-free backbone validation",
        "",
        "> Mean +/- sample standard deviation. Lower is better. Held-out test data were not evaluated.",
        "",
        "| Dataset | Model | Clamped time loss | Log-count MSE | Quantity MAE | Quantity RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        t3.append(
            f"| {row['dataset_label']} | {row['model_label']} | "
            f"{row['time_nll_mean']:.6f} +/- {row['time_nll_std']:.6f} | "
            f"{row['log_qty_mse_mean']:.6f} +/- {row['log_qty_mse_std']:.6f} | "
            f"{row['qty_mae_mean']:.4f} +/- {row['qty_mae_std']:.4f} | "
            f"{row['qty_rmse_mean']:.4f} +/- {row['qty_rmse_std']:.4f} |"
        )
    t3.append("")
    (TABLE_DIR / "T3_v0_7_backbone_validation.md").write_text("\n".join(t3), encoding="utf-8")

    t4 = [
        "# T4 v0.7. Paired TitanTPP comparisons",
        "",
        "> Candidate minus reference. A negative delta and a larger better-seed count favor TitanTPP.",
        "",
        "| Dataset | Reference | MAE delta | MAE better seeds | RMSE delta | RMSE better seeds | Clamped-time delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired_summary:
        t4.append(
            f"| {row['dataset_label']} | {MODEL_LABELS[row['reference']]} | "
            f"{row['qty_mae_mean_delta']:.6f} | {row['qty_mae_better_seed_count']}/3 | "
            f"{row['qty_rmse_mean_delta']:.6f} | {row['qty_rmse_better_seed_count']}/3 | "
            f"{row['time_nll_mean_delta']:.6f} |"
        )
    t4.append("")
    (TABLE_DIR / "T4_v0_7_paired_titan_deltas.md").write_text("\n".join(t4), encoding="utf-8")

    t5 = [
        "# T5 v0.7. Train-defined quantity-stratified validation errors",
        "",
        "> Mean +/- sample standard deviation over seeds 42, 52, and 62. Stratum thresholds are fitted on each train split and then fixed for validation targets.",
        "",
        "| Dataset | Validation target stratum | Share | Model | Quantity MAE | Quantity RMSE |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in quantity_rows:
        t5.append(
            f"| {row['dataset_label']} | {row['stratum_label']} | {100.0 * row['share']:.2f}% | "
            f"{row['model_label']} | {row['qty_mae_mean']:.4f} +/- {row['qty_mae_std']:.4f} | "
            f"{row['qty_rmse_mean']:.4f} +/- {row['qty_rmse_std']:.4f} |"
        )
    t5.extend(
        [
            "",
            "Target membership and count are identical across the three models within each dataset and stratum. Results remain validation-only.",
            "",
        ]
    )
    (TABLE_DIR / "T5_v0_7_quantity_strata_validation.md").write_text(
        "\n".join(t5), encoding="utf-8"
    )


def make_figure(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_rows: list[dict[str, Any]] = []
    for dataset in ("intermittent", "taxi", "instacart"):
        baseline = next(
            row for row in aggregate_rows if row["dataset"] == dataset and row["model"] == "rmtpp"
        )
        for model in MODEL_ORDER:
            row = next(
                item for item in aggregate_rows if item["dataset"] == dataset and item["model"] == model
            )
            source_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": row["dataset_label"],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "qty_mae_mean": row["qty_mae_mean"],
                    "qty_rmse_mean": row["qty_rmse_mean"],
                    "mae_ratio_to_rmtpp": row["qty_mae_mean"] / baseline["qty_mae_mean"],
                    "rmse_ratio_to_rmtpp": row["qty_rmse_mean"] / baseline["qty_rmse_mean"],
                }
            )
    source_path = FIGURE_DATA_DIR / "F2_v0_7_dataset_validation_errors.csv"
    write_csv(source_path, source_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "svg.hashsalt": "titantpp-v0.7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    x = np.arange(3)
    width = 0.24
    colors = {"rmtpp": "#6B7280", "thp": "#2563EB", "titantpp": "#C56A00"}
    dataset_labels = ["Intermittent\n5000", "Taxi", "Instacart"]
    for axis, (metric, title) in zip(
        axes,
        (("mae_ratio_to_rmtpp", "Quantity MAE"), ("rmse_ratio_to_rmtpp", "Quantity RMSE")),
    ):
        for index, model in enumerate(MODEL_ORDER):
            values = [
                next(
                    row[metric]
                    for row in source_rows
                    if row["dataset"] == dataset and row["model"] == model
                )
                for dataset in ("intermittent", "taxi", "instacart")
            ]
            bars = axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=MODEL_LABELS[model].replace("Count-aware ", ""),
                color=colors[model],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=0,
                )
        axis.axhline(1.0, color="#111827", linewidth=0.9, linestyle="--", zorder=0)
        axis.set_title(title)
        axis.set_xticks(x, dataset_labels)
        axis.set_ylim(0, 1.16)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation error / RMTPP error")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Mark-free quantity error relative to RMTPP", y=1.10, fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "Ratios use three-seed validation means within each dataset. Below 1.0 favors the compared encoder.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=1.6)
    output_stem = FIGURE_DIR / "F2_v0_7_dataset_validation_errors"
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "paper/scripts/build_v0_7_paper_artifacts.py"},
    )
    fixed_time = dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc)
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "paper/scripts/build_v0_7_paper_artifacts.py", "CreationDate": fixed_time, "ModDate": fixed_time},
    )
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "paper/scripts/build_v0_7_paper_artifacts.py", "Date": "2026-09-05"},
    )
    svg_path = output_stem.with_suffix(".svg")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return source_rows


def write_manifest(
    dataset_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    generated_paths: list[Path],
) -> None:
    source_paths = [
        ROOT / "paper/contracts/titantpp_v0_7_final_model_claim_contract_v1.md",
        ROOT / "paper/scripts/build_v0_7_paper_artifacts.py",
        ROOT / "paper/scripts/generate_v0_7_mark_free_figures.py",
        ROOT / "paper/scripts/audit_v0_7_time_head_revisions.py",
        ROOT / "paper/scripts/verify_v0_7_paper_artifacts.py",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/build_freeze.py",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/independent_verify.py",
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/run_summaries.csv",
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_summary.csv",
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_seed_metrics.csv",
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/history_summary.csv",
        ROOT
        / "paper/results/count_aware_tpp_backbone_control_20260812/qualification_briefing.md",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/validation_seed_metrics.csv",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_aggregate.csv",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_strata_seed_metrics.csv",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/quantity_interface_audit.json",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/dataset_identity_audit.json",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/source_manifest.json",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/independent_verification.json",
        ROOT
        / "paper/results/titantpp_v0_7_validation_freeze_20260905/time_head_revision_audit.json",
        ROOT
        / "paper/results/hard_lmm_key_value_screening_5090_20260904/README.md",
        ROOT
        / "paper/results/hard_lmm_elapsed_age_screening_5090_20260905/README.md",
        ROOT
        / "paper/results/hard_lmm_instacart_raw_history_20260905/README.md",
    ]
    source_paths.extend(spec["path"] for spec in DATASETS)
    source_paths.extend(spec["split_manifest"] for spec in DATASETS)
    sources = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in source_paths
    ]
    outputs = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in generated_paths
    ]
    publication_documents = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (MANUSCRIPT_PATH, ROOT_README_PATH)
    ]
    payload = {
        "schema_version": 1,
        "frozen_at": "2026-09-05 KST",
        "status": "validation_freeze_complete_held_out_locked",
        "model_identity": "original mark-free Hard-LMM / Count-aware TitanTPP-T0",
        "main_models": list(MODEL_ORDER),
        "datasets": [row["dataset"] for row in dataset_rows],
        "seeds": list(SEEDS),
        "validation_run_count": len(seed_rows),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "quantity_interface_empirical_claim": "excluded_contract_mismatch",
        "training_source_revisions": SOURCE_REVISIONS,
        "time_metric": {
            "artifact_field": "time_nll",
            "paper_label": "clamped time loss",
            "executed_intercept_cap": 300.0,
            "wd_cap": 10.0,
            "exact_likelihood": False,
        },
        "sources": sources,
        "outputs": outputs,
        "publication_documents": publication_documents,
    }
    json_path = MANIFEST_DIR / "titantpp_v0_7_validation_freeze_manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows = (
        [{"kind": "source", **row} for row in sources]
        + [{"kind": "output", **row} for row in outputs]
        + [{"kind": "document", **row} for row in publication_documents]
    )
    write_csv(MANIFEST_DIR / "titantpp_v0_7_validation_freeze_manifest.csv", csv_rows)

    lines = [
        "# TitanTPP v0.7 validation-freeze artifact manifest",
        "",
        "- Status: `validation_freeze_complete_held_out_locked`",
        "- Frozen model: original mark-free Hard-LMM / Count-aware TitanTPP-T0",
        "- Main comparison: Count-aware RMTPP, THP, and TitanTPP",
        "- Datasets: Intermittent-5000, Taxi, and Instacart",
        "- Seeds: 42, 52, and 62",
        f"- Qualified validation rows: {len(seed_rows)}",
        "- Held-out test evaluated: false",
        "- Quantity-interface superiority claim: excluded because the existing ablation fails the v0.7 matched contract",
        "- Time metric: artifact `time_nll` is reported as clamped time loss; executed caps are intercept 300 and `w * delta_t` 10",
        "- Training source revisions: Intermittent `044add1f`, Taxi `6a01aea9`, Instacart `28293c43`",
        "",
        "## Source identity",
        "",
        "| Path | SHA-256 | Bytes |",
        "| --- | --- | ---: |",
    ]
    for row in sources:
        lines.append(f"| `{row['path']}` | `{row['sha256']}` | {row['bytes']:,} |")
    lines.extend(["", "## Generated artifacts", "", "| Path | SHA-256 | Bytes |", "| --- | --- | ---: |"])
    for row in outputs:
        lines.append(f"| `{row['path']}` | `{row['sha256']}` | {row['bytes']:,} |")
    lines.extend(["", "## Publication documents", "", "| Path | SHA-256 | Bytes |", "| --- | --- | ---: |"])
    for row in publication_documents:
        lines.append(f"| `{row['path']}` | `{row['sha256']}` | {row['bytes']:,} |")
    lines.extend(
        [
            "",
            "This manifest freezes validation evidence only. The one-time held-out evaluation remains a separate approved step after the checkpoint list and execution contract are fixed.",
            "",
        ]
    )
    (MANIFEST_DIR / "titantpp_v0_7_validation_freeze_manifest.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    for directory in (TABLE_DIR, FIGURE_DIR, FIGURE_DATA_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_rows = dataset_statistics()
    seed_rows = validation_seed_rows()
    aggregate_rows = aggregate_validation(seed_rows)
    paired_detail, paired_summary = paired_deltas(seed_rows)
    quantity_rows = quantity_strata_rows()
    contract_rows = model_contract_rows(dataset_rows)

    csv_outputs = {
        TABLE_DIR / "T1_v0_7_dataset_statistics.csv": dataset_rows,
        TABLE_DIR / "T2_v0_7_model_training_contract.csv": contract_rows,
        TABLE_DIR / "T3_v0_7_backbone_validation.csv": aggregate_rows,
        TABLE_DIR / "T4_v0_7_paired_titan_deltas.csv": paired_detail,
        TABLE_DIR / "T5_v0_7_quantity_strata_validation.csv": quantity_rows,
    }
    for path, rows in csv_outputs.items():
        write_csv(path, rows)
    write_markdown_tables(
        dataset_rows,
        contract_rows,
        aggregate_rows,
        paired_summary,
        quantity_rows,
    )
    make_figure(aggregate_rows)
    sync_root_readme()

    generated_paths = [
        *csv_outputs,
        TABLE_DIR / "T1_v0_7_dataset_statistics.md",
        TABLE_DIR / "T2_v0_7_model_training_contract.md",
        TABLE_DIR / "T3_v0_7_backbone_validation.md",
        TABLE_DIR / "T4_v0_7_paired_titan_deltas.md",
        TABLE_DIR / "T5_v0_7_quantity_strata_validation.md",
        FIGURE_DIR / "F1_v0_7_mark_free_architecture.png",
        FIGURE_DIR / "F1_v0_7_mark_free_architecture.pdf",
        FIGURE_DIR / "F1_v0_7_mark_free_architecture.svg",
        FIGURE_DATA_DIR / "F2_v0_7_dataset_validation_errors.csv",
        FIGURE_DIR / "F2_v0_7_dataset_validation_errors.png",
        FIGURE_DIR / "F2_v0_7_dataset_validation_errors.pdf",
        FIGURE_DIR / "F2_v0_7_dataset_validation_errors.svg",
    ]
    write_manifest(dataset_rows, seed_rows, generated_paths)
    print(
        f"Built {len(generated_paths)} v0.7 paper artifacts and 2 publication documents "
        f"from {len(seed_rows)} validation rows."
    )


if __name__ == "__main__":
    main()
