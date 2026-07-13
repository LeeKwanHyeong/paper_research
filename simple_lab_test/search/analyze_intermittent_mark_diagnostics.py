#!/usr/bin/env python3
"""Diagnose Intermittent mark imbalance and V2/V3 confusion behavior."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import polars as pl

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "sample_data/head_office/marked_target_with_split.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "search_artifacts/model_enhancement_inter_mark_diagnostics_0712"
DEFAULT_VARIANTS = {
    "V2": PROJECT_ROOT / "search_artifacts/model_enhancement_v2_inter_short_e50_0710",
    "V3a": PROJECT_ROOT / "search_artifacts/model_enhancement_v3_inter_short_e50_0710",
    "V3b": PROJECT_ROOT / "search_artifacts/model_enhancement_v3b_inter_short_e50_0710",
    "V3c": PROJECT_ROOT / "search_artifacts/model_enhancement_v3c_inter_short_e50_0712",
}
VARIANT_COLORS = {
    "V2": "#2563EB",
    "V3a": "#D97706",
    "V3b": "#6B7C23",
    "V3c": "#C24172",
}
SPLIT_COLORS = {
    "train": "#2563EB",
    "validation": "#D97706",
    "test": "#6B7C23",
}
SPLIT_ORDER = ("train", "validation", "test")
EVAL_SPLITS = ("validation", "test")
SELECTION = "best_val_nll"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    for variant, path in DEFAULT_VARIANTS.items():
        parser.add_argument(f"--{variant.lower()}-dir", type=Path, default=path)
    return parser.parse_args()


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return Jensen-Shannon divergence in nats."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        valid = left > 0
        return float(np.sum(left[valid] * np.log(left[valid] / right[valid])))

    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def load_target_events(path: Path) -> pl.DataFrame:
    """Load the fixed split and retain rows that are valid next-event targets."""
    required = {"oper_part_no", "seq", "mark", "chronological_split"}
    frame = pl.read_parquet(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = frame.sort(["oper_part_no", "seq"]).with_columns(
        pl.col("seq")
        .rank(method="ordinal")
        .over("oper_part_no")
        .alias("_series_event_rank")
    )
    targets = frame.filter(pl.col("_series_event_rank") > 1)
    observed_splits = set(targets["chronological_split"].unique().to_list())
    if observed_splits != set(SPLIT_ORDER):
        raise ValueError(f"Unexpected chronological splits: {sorted(observed_splits)}")
    return targets


def build_distribution_tables(
    targets: pl.DataFrame,
    num_marks: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    counts = targets.group_by(["chronological_split", "mark"]).len(name="count")
    lookup = {
        (str(row["chronological_split"]), int(row["mark"])): int(row["count"])
        for row in counts.iter_rows(named=True)
    }

    distribution_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    split_vectors: dict[str, np.ndarray] = {}

    for split in SPLIT_ORDER:
        vector = np.asarray([lookup.get((split, mark), 0) for mark in range(num_marks)], dtype=np.int64)
        total = int(vector.sum())
        if total == 0:
            raise ValueError(f"Split {split} has no target rows")
        shares = vector.astype(np.float64) / total
        split_vectors[split] = shares
        entropy = float(-np.sum(shares[shares > 0] * np.log(shares[shares > 0])))
        nonzero_counts = vector[vector > 0]

        for mark, (count, share) in enumerate(zip(vector, shares)):
            distribution_rows.append(
                {
                    "split": split,
                    "mark": mark,
                    "count": int(count),
                    "share": float(share),
                }
            )

        summary_rows.append(
            {
                "split": split,
                "target_count": total,
                "observed_classes": int((vector > 0).sum()),
                "majority_mark": int(np.argmax(vector)),
                "majority_share": float(shares.max()),
                "top2_share": float(np.sort(shares)[-2:].sum()),
                "mark_0_2_share": float(shares[:3].sum()),
                "mark_4_plus_share": float(shares[4:].sum()),
                "imbalance_ratio_max_min": float(nonzero_counts.max() / nonzero_counts.min()),
                "entropy_nats": entropy,
                "normalized_entropy": float(entropy / math.log(num_marks)),
                "effective_class_count": float(math.exp(entropy)),
                "mean_mark": float(np.dot(np.arange(num_marks), shares)),
                "rare_class_count_below_1pct": int((shares < 0.01).sum()),
            }
        )

    train_shares = split_vectors["train"]
    for row in summary_rows:
        shares = split_vectors[str(row["split"])]
        row["tv_distance_vs_train"] = float(0.5 * np.abs(shares - train_shares).sum())
        row["js_divergence_vs_train"] = jensen_shannon_divergence(shares, train_shares)

    distribution = pl.DataFrame(distribution_rows).with_columns(
        pl.col("split").replace_strict({name: idx for idx, name in enumerate(SPLIT_ORDER)}).alias("split_order")
    ).sort(["split_order", "mark"]).drop("split_order")
    summary = pl.DataFrame(summary_rows).with_columns(
        pl.col("split").replace_strict({name: idx for idx, name in enumerate(SPLIT_ORDER)}).alias("split_order")
    ).sort("split_order").drop("split_order")
    return distribution, summary


def find_unique_metric_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} below {root}, found {len(matches)}")
    return matches[0]


def confusion_matrix_from_csv(path: Path, num_marks: int) -> np.ndarray:
    frame = pl.read_csv(path)
    matrix = np.zeros((num_marks, num_marks), dtype=np.int64)
    for row in frame.iter_rows(named=True):
        true_mark = int(row["true_mark"])
        pred_mark = int(row["pred_mark"])
        if not 0 <= true_mark < num_marks or not 0 <= pred_mark < num_marks:
            raise ValueError(f"Out-of-range mark in {path}: {(true_mark, pred_mark)}")
        matrix[true_mark, pred_mark] += int(row["count"])
    return matrix


def confusion_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    matrix = np.asarray(matrix, dtype=np.int64)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Confusion matrix is empty")

    marks = np.arange(matrix.shape[0])
    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    correct = np.diag(matrix)
    recall = np.divide(correct, true_counts, out=np.zeros_like(correct, dtype=float), where=true_counts > 0)
    precision = np.divide(correct, pred_counts, out=np.zeros_like(correct, dtype=float), where=pred_counts > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) > 0,
    )

    distance = np.abs(marks[:, None] - marks[None, :])
    signed_distance = marks[None, :] - marks[:, None]
    error_count = total - int(correct.sum())
    true_shares = true_counts / total
    pred_shares = pred_counts / total
    supported = true_counts > 0

    return {
        "total": total,
        "accuracy": float(correct.sum() / total),
        "balanced_accuracy": float(recall[supported].mean()),
        "macro_precision": float(precision[supported].mean()),
        "macro_f1": float(f1[supported].mean()),
        "weighted_f1": float(np.dot(f1, true_shares)),
        "majority_baseline_accuracy": float(true_shares.max()),
        "adjacent_accuracy": float(matrix[distance <= 1].sum() / total),
        "within_two_accuracy": float(matrix[distance <= 2].sum() / total),
        "adjacent_share_of_errors": float(matrix[distance == 1].sum() / max(error_count, 1)),
        "severe_error_rate": float(matrix[distance >= 2].sum() / total),
        "mark_mae": float((matrix * distance).sum() / total),
        "mark_rmse": float(np.sqrt((matrix * distance**2).sum() / total)),
        "signed_mark_error": float((matrix * signed_distance).sum() / total),
        "underprediction_rate": float(matrix[signed_distance < 0].sum() / total),
        "overprediction_rate": float(matrix[signed_distance > 0].sum() / total),
        "true_mark_0_share": float(true_shares[0]),
        "pred_mark_0_share": float(pred_shares[0]),
        "true_mark_0_2_share": float(true_shares[:3].sum()),
        "pred_mark_0_2_share": float(pred_shares[:3].sum()),
        "true_mean_mark": float(np.dot(marks, true_shares)),
        "pred_mean_mark": float(np.dot(marks, pred_shares)),
        "prediction_tv_vs_true": float(0.5 * np.abs(pred_shares - true_shares).sum()),
        "prediction_js_vs_true": jensen_shannon_divergence(pred_shares, true_shares),
    }


def per_class_metrics(matrix: np.ndarray, variant: str, split: str) -> list[dict[str, object]]:
    total = int(matrix.sum())
    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    correct = np.diag(matrix)
    rows: list[dict[str, object]] = []

    for mark in range(matrix.shape[0]):
        true_count = int(true_counts[mark])
        pred_count = int(pred_counts[mark])
        recall = float(correct[mark] / true_count) if true_count else 0.0
        precision = float(correct[mark] / pred_count) if pred_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        distances = np.abs(np.arange(matrix.shape[1]) - mark)
        signed = np.arange(matrix.shape[1]) - mark
        rows.append(
            {
                "variant": variant,
                "eval_split": split,
                "mark": mark,
                "true_count": true_count,
                "true_share": float(true_count / total),
                "pred_count": pred_count,
                "pred_share": float(pred_count / total),
                "correct_count": int(correct[mark]),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "mark_mae_within_true": float(np.dot(matrix[mark], distances) / max(true_count, 1)),
                "underprediction_rate_within_true": float(matrix[mark][signed < 0].sum() / max(true_count, 1)),
                "overprediction_rate_within_true": float(matrix[mark][signed > 0].sum() / max(true_count, 1)),
            }
        )
    return rows


def top_confusion_rows(matrix: np.ndarray, variant: str, split: str) -> list[dict[str, object]]:
    total = int(matrix.sum())
    errors = total - int(np.trace(matrix))
    true_counts = matrix.sum(axis=1)
    rows: list[dict[str, object]] = []
    for true_mark in range(matrix.shape[0]):
        for pred_mark in range(matrix.shape[1]):
            count = int(matrix[true_mark, pred_mark])
            if true_mark == pred_mark or count == 0:
                continue
            rows.append(
                {
                    "variant": variant,
                    "eval_split": split,
                    "true_mark": true_mark,
                    "pred_mark": pred_mark,
                    "distance": abs(pred_mark - true_mark),
                    "count": count,
                    "share_within_true": float(count / true_counts[true_mark]),
                    "share_of_all_errors": float(count / max(errors, 1)),
                    "share_of_all_targets": float(count / total),
                }
            )
    return sorted(rows, key=lambda row: (-int(row["count"]), int(row["true_mark"]), int(row["pred_mark"])))


def accuracy_contribution_rows(
    matrices: dict[str, dict[str, np.ndarray]],
    baseline_variant: str = "V2",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in EVAL_SPLITS:
        baseline = matrices[split][baseline_variant]
        baseline_total = int(baseline.sum())
        baseline_true = baseline.sum(axis=1)
        for variant, matrix in matrices[split].items():
            if variant == baseline_variant:
                continue
            if not np.array_equal(matrix.sum(axis=1), baseline_true):
                raise ValueError(f"True target counts differ for {variant} on {split}")
            for mark in range(matrix.shape[0]):
                baseline_correct = int(baseline[mark, mark])
                variant_correct = int(matrix[mark, mark])
                true_count = int(baseline_true[mark])
                rows.append(
                    {
                        "variant": variant,
                        "eval_split": split,
                        "mark": mark,
                        "true_count": true_count,
                        "baseline_correct_count": baseline_correct,
                        "variant_correct_count": variant_correct,
                        "delta_correct_count": variant_correct - baseline_correct,
                        "delta_recall_pp": 100.0 * (variant_correct - baseline_correct) / max(true_count, 1),
                        "delta_accuracy_pp": 100.0 * (variant_correct - baseline_correct) / baseline_total,
                    }
                )
    return rows


def load_reported_accuracy(root: Path, split: str) -> float:
    if split == "test":
        frame = pl.read_csv(root / "leaderboard/test_summary.csv").filter(pl.col("selection") == SELECTION)
        return float(frame["mean_test_mark_acc"].item())
    frame = pl.read_csv(root / "leaderboard/summary.csv")
    return float(frame["mean_best_val_nll_mark_acc"].item())


def load_confusions(
    variant_roots: dict[str, Path],
    distribution: pl.DataFrame,
    num_marks: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in EVAL_SPLITS}
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []

    for split in EVAL_SPLITS:
        expected = (
            distribution.filter(pl.col("split") == split)
            .sort("mark")["count"]
            .to_numpy()
            .astype(np.int64)
        )
        for variant, root in variant_roots.items():
            filename = f"{split}_mark_confusion_{SELECTION}.csv"
            path = find_unique_metric_file(root, filename)
            matrix = confusion_matrix_from_csv(path, num_marks)
            matrices[split][variant] = matrix

            true_match = np.array_equal(matrix.sum(axis=1), expected)
            if not true_match:
                raise ValueError(f"Confusion true counts do not match fixed split for {variant} {split}")
            metrics = confusion_metrics(matrix)
            reported_accuracy = load_reported_accuracy(root, split)
            accuracy_match = math.isclose(float(metrics["accuracy"]), reported_accuracy, abs_tol=1e-12)
            if not accuracy_match:
                raise ValueError(f"Confusion accuracy does not match leaderboard for {variant} {split}")

            metric_rows.append({"variant": variant, "eval_split": split, **metrics})
            class_rows.extend(per_class_metrics(matrix, variant, split))
            top_rows.extend(top_confusion_rows(matrix, variant, split))
            true_counts = matrix.sum(axis=1)
            pred_counts = matrix.sum(axis=0)
            total = int(matrix.sum())
            for mark in range(num_marks):
                prediction_rows.append(
                    {
                        "variant": variant,
                        "eval_split": split,
                        "mark": mark,
                        "true_count": int(true_counts[mark]),
                        "true_share": float(true_counts[mark] / total),
                        "pred_count": int(pred_counts[mark]),
                        "pred_share": float(pred_counts[mark] / total),
                        "pred_minus_true_pp": float(100 * (pred_counts[mark] - true_counts[mark]) / total),
                    }
                )
            validations.append(
                {
                    "variant": variant,
                    "eval_split": split,
                    "confusion_path": str(path),
                    "target_count": total,
                    "true_counts_match_fixed_split": true_match,
                    "accuracy_matches_leaderboard": accuracy_match,
                    "reported_accuracy": reported_accuracy,
                    "recomputed_accuracy": float(metrics["accuracy"]),
                }
            )

    contribution_rows = accuracy_contribution_rows(matrices)
    return (
        matrices,
        pl.DataFrame(metric_rows),
        pl.DataFrame(class_rows),
        pl.DataFrame(prediction_rows),
        pl.DataFrame(contribution_rows),
        top_rows,
        validations,
    )


def row_for(frame: pl.DataFrame, **filters: object) -> dict[str, object]:
    expression = None
    for name, value in filters.items():
        current = pl.col(name) == value
        expression = current if expression is None else expression & current
    result = frame.filter(expression) if expression is not None else frame
    if result.height != 1:
        raise ValueError(f"Expected one row for {filters}, found {result.height}")
    return result.row(0, named=True)


def plot_split_distribution(distribution: pl.DataFrame, output_path: Path, num_marks: int) -> None:
    marks = np.arange(num_marks)
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for index, split in enumerate(SPLIT_ORDER):
        shares = distribution.filter(pl.col("split") == split).sort("mark")["share"].to_numpy()
        ax.bar(
            marks + (index - 1) * width,
            shares,
            width=width,
            label=split,
            color=SPLIT_COLORS[split],
            edgecolor="#263238",
            linewidth=0.5,
        )
    ax.set_yscale("log")
    ax.set_xticks(marks)
    ax.set_xlabel("True mark")
    ax.set_ylabel("Target share (log scale)")
    ax.set_title("Intermittent target mark distribution by fixed split")
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_recall(per_class: pl.DataFrame, output_path: Path, num_marks: int) -> None:
    marks = np.arange(num_marks)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    test = per_class.filter(pl.col("eval_split") == "test")
    for variant in DEFAULT_VARIANTS:
        values = test.filter(pl.col("variant") == variant).sort("mark")["recall"].to_numpy()
        ax.plot(
            marks,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=variant,
            color=VARIANT_COLORS[variant],
        )
    ax.set_xticks(marks)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("True mark")
    ax.set_ylabel("Recall")
    ax.set_title("Held-out test recall by true mark")
    ax.grid(color="#D7DCE2", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_test_confusions(
    matrices: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    num_marks: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    image = None
    for ax, variant in zip(axes.flat, DEFAULT_VARIANTS):
        matrix = matrices["test"][variant]
        true_counts = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, true_counts, out=np.zeros_like(matrix, dtype=float), where=true_counts > 0)
        image = ax.imshow(normalized * 100, vmin=0, vmax=90, cmap="Blues", aspect="auto")
        for true_mark in range(num_marks):
            for pred_mark in range(num_marks):
                value = normalized[true_mark, pred_mark] * 100
                if true_mark == pred_mark or value >= 10:
                    ax.text(
                        pred_mark,
                        true_mark,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value >= 48 else "#263238",
                    )
        ax.set_title(variant)
        ax.set_xticks(range(num_marks))
        ax.set_yticks(range(num_marks))
        ax.set_xlabel("Predicted mark")
        ax.set_ylabel("True mark")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Row share (%)")
    fig.suptitle("Held-out test row-normalized confusion matrices", y=0.99, fontsize=15)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.06, top=0.94, wspace=0.14, hspace=0.16)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_contributions(contributions: pl.DataFrame, output_path: Path, num_marks: int) -> None:
    marks = np.arange(num_marks)
    variants = ("V3a", "V3b", "V3c")
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 6.2))
    test = contributions.filter(pl.col("eval_split") == "test")
    for index, variant in enumerate(variants):
        values = test.filter(pl.col("variant") == variant).sort("mark")["delta_accuracy_pp"].to_numpy()
        ax.bar(
            marks + (index - 1) * width,
            values,
            width=width,
            label=variant,
            color=VARIANT_COLORS[variant],
            edgecolor="#263238",
            linewidth=0.5,
        )
    ax.axhline(0, color="#263238", linewidth=1.0)
    ax.set_xticks(marks)
    ax.set_xlabel("True mark")
    ax.set_ylabel("Contribution to accuracy delta vs V2 (%p)")
    ax.set_title("Held-out test accuracy delta decomposition by true mark")
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_model_table(metrics: pl.DataFrame) -> list[str]:
    lines = [
        "| Variant | Accuracy | Balanced acc. | Macro F1 | Adjacent acc. | Adjacent share of errors | Mark MAE | Pred mark-0 share | Signed bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in DEFAULT_VARIANTS:
        row = row_for(metrics, variant=variant, eval_split="test")
        lines.append(
            f"| {variant} | {100*float(row['accuracy']):.3f}% | "
            f"{100*float(row['balanced_accuracy']):.3f}% | {float(row['macro_f1']):.4f} | "
            f"{100*float(row['adjacent_accuracy']):.3f}% | "
            f"{100*float(row['adjacent_share_of_errors']):.3f}% | "
            f"{float(row['mark_mae']):.4f} | {100*float(row['pred_mark_0_share']):.3f}% | "
            f"{float(row['signed_mark_error']):+.4f} |"
        )
    return lines


def build_report(
    distribution: pl.DataFrame,
    distribution_summary: pl.DataFrame,
    model_metrics: pl.DataFrame,
    per_class: pl.DataFrame,
    contributions: pl.DataFrame,
    top_confusions: pl.DataFrame,
    matrices: dict[str, dict[str, np.ndarray]],
) -> str:
    test_summary = row_for(distribution_summary, split="test")
    v2 = row_for(model_metrics, variant="V2", eval_split="test")
    v3c = row_for(model_metrics, variant="V3c", eval_split="test")
    v3c_mark0 = row_for(contributions, variant="V3c", eval_split="test", mark=0)
    v3c_mark1 = row_for(contributions, variant="V3c", eval_split="test", mark=1)
    v3c_total_delta = 100 * (float(v3c["accuracy"]) - float(v2["accuracy"]))
    mark1_to_0_v2 = matrices["test"]["V2"][1, 0] / matrices["test"]["V2"][1].sum()
    mark1_to_0_v3c = matrices["test"]["V3c"][1, 0] / matrices["test"]["V3c"][1].sum()
    pred0_delta = 100 * (float(v3c["pred_mark_0_share"]) - float(v2["pred_mark_0_share"]))

    lines = [
        "# Intermittent Mark Imbalance And Confusion Diagnostic",
        "",
        "Status: `completed`  ",
        f"Selection: `{SELECTION}`  ",
        "Comparison: TitanTPP V2, V3a, V3b, V3c; fixed split; seed 42; e50",
        "",
        "## Decision-Useful Answer",
        "",
        f"The held-out target distribution is strongly concentrated: marks 0-2 account for "
        f"`{100*float(test_summary['mark_0_2_share']):.2f}%` of targets and the majority class alone "
        f"accounts for `{100*float(test_summary['majority_share']):.2f}%`. This imbalance is real, but it "
        "does not by itself explain the V3 regression because every variant is evaluated on identical targets.",
        "",
        f"The differentiating failure is a local decision-boundary shift. V3c predicts mark 0 "
        f"`{pred0_delta:+.2f}%p` more often than V2. Its mark-1-to-0 confusion rises from "
        f"`{100*mark1_to_0_v2:.2f}%` to `{100*mark1_to_0_v3c:.2f}%`. The mark-1 loss contributes "
        f"`{float(v3c_mark1['delta_accuracy_pp']):+.3f}%p` to the total V3c accuracy change, while the "
        f"mark-0 gain offsets `{float(v3c_mark0['delta_accuracy_pp']):+.3f}%p`; the net change is "
        f"`{v3c_total_delta:+.3f}%p`.",
        "",
        f"Most errors remain ordinally local: adjacent classes account for "
        f"`{100*float(v3c['adjacent_share_of_errors']):.2f}%` of V3c test errors. The evidence therefore "
        "supports an ordinal or boundary-aware marker objective as the next focused design, rather than more "
        "encoder-gradient detachment. Raw inverse-frequency weighting is not the preferred first move because "
        "the extreme tail has very small support and could dominate gradients.",
        "",
        "## Source And Metric Validation",
        "",
        "- Source of target distribution: `sample_data/head_office/marked_target_with_split.parquet`.",
        "- Target grain: next-event rows; the first event of each series is excluded exactly as in `RMTPPWeekLookbackDataset`.",
        "- Model source: validation/test `mark_confusion_best_val_nll.csv` for each variant.",
        "- Every confusion matrix true-class count matches the fixed split target distribution.",
        "- Recomputed accuracy matches the corresponding leaderboard value for every variant and split.",
        "",
        "## Fixed-Split Target Distribution",
        "",
        "| Split | Targets | Majority share | Marks 0-2 | Marks 4+ | Effective classes | TV vs train |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLIT_ORDER:
        row = row_for(distribution_summary, split=split)
        lines.append(
            f"| {split} | {int(row['target_count']):,} | {100*float(row['majority_share']):.2f}% | "
            f"{100*float(row['mark_0_2_share']):.2f}% | {100*float(row['mark_4_plus_share']):.2f}% | "
            f"{float(row['effective_class_count']):.2f} | {float(row['tv_distance_vs_train']):.4f} |"
        )

    lines.extend(
        [
            "",
            "Held-out test class counts:",
            "",
            "| Mark | Count | Share |",
            "| ---: | ---: | ---: |",
        ]
    )
    for mark in sorted(distribution["mark"].unique().to_list()):
        row = row_for(distribution, split="test", mark=int(mark))
        lines.append(f"| {mark} | {int(row['count']):,} | {100*float(row['share']):.3f}% |")

    lines.extend(["", "## Held-Out Test Model Diagnostics", "", *markdown_model_table(model_metrics)])
    lines.extend(
        [
            "",
            "## Head-Class Recall And Accuracy Contribution",
            "",
            "| Mark | True share | V2 recall | V3a recall | V3b recall | V3c recall | V3c contribution vs V2 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mark in range(min(6, matrices["test"]["V2"].shape[0])):
        dist = row_for(distribution, split="test", mark=mark)
        recalls = [
            100 * float(row_for(per_class, variant=variant, eval_split="test", mark=mark)["recall"])
            for variant in DEFAULT_VARIANTS
        ]
        contribution = row_for(contributions, variant="V3c", eval_split="test", mark=mark)
        lines.append(
            f"| {mark} | {100*float(dist['share']):.3f}% | {recalls[0]:.2f}% | {recalls[1]:.2f}% | "
            f"{recalls[2]:.2f}% | {recalls[3]:.2f}% | {float(contribution['delta_accuracy_pp']):+.3f}%p |"
        )

    lines.extend(
        [
            "",
            "Largest V3c off-diagonal test confusions:",
            "",
            "| True -> Pred | Count | Within true | Share of all errors | Distance |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    v3c_top = top_confusions.filter(
        (pl.col("variant") == "V3c") & (pl.col("eval_split") == "test")
    ).head(8)
    for row in v3c_top.iter_rows(named=True):
        lines.append(
            f"| {row['true_mark']} -> {row['pred_mark']} | {int(row['count']):,} | "
            f"{100*float(row['share_within_true']):.2f}% | "
            f"{100*float(row['share_of_all_errors']):.2f}% | {row['distance']} |"
        )

    lines.extend(
        [
            "",
            "## Confirmed Findings",
            "",
            "- The target distribution is imbalanced and shifts modestly toward lower marks in validation/test.",
            "- V3 variants do not fail because of rare tail classes alone. Most aggregate accuracy movement is generated by marks 0 and 1, which have high support.",
            "- V3a/V3b/V3c increase mark-0 preference. The gain on true mark 0 is offset by a larger loss on true mark 1.",
            "- V3c partially reverses the V3a/V3b mark-0 collapse, but not enough to restore V2 accuracy.",
            "- Most errors are adjacent in ordinal mark space, while distance-two-or-more errors are a minority.",
            "- Validation and test use identical class definitions and the confusion totals reconcile to their fixed-split target counts.",
            "",
            "## Interpretation And Next Design Input",
            "",
            "Inference, not causal proof:",
            "",
            "- Class imbalance is an enabling condition that makes the 0/1 boundary sensitive, but the architecture/optimization path determines how the boundary moves.",
            "- Full class reweighting may improve macro recall but could over-amplify very rare marks. It should not be the first standalone replacement for cross-entropy.",
            "- The first focused ordinal prototype should retain standard marker CE and add a small ordered-distance auxiliary term, with explicit 0/1 recall and macro-recall guardrails.",
            "- A separate capped effective-number or logit-adjustment ablation can test prior correction without mixing it into the first ordinal architecture result.",
            "- V2 remains the Intermittent baseline; V3c multi-seed remains stopped.",
            "",
            "## Caveats",
            "",
            "- Model comparisons are seed-42 e50 screening results, not final multi-seed evidence.",
            "- Confusion artifacts contain hard predictions, not full probability calibration; calibration error and class-conditional confidence remain unresolved.",
            "- The diagnostic shows association and exact error decomposition, not that imbalance alone caused the boundary shift.",
            "- Tail-class metrics have low support and should not be interpreted without counts.",
            "",
            "## Generated Artifacts",
            "",
            "- `data/target_mark_distribution.csv`",
            "- `data/split_distribution_summary.csv`",
            "- `data/model_confusion_metrics.csv`",
            "- `data/per_class_metrics.csv`",
            "- `data/prediction_distribution.csv`",
            "- `data/accuracy_contributions_vs_v2.csv`",
            "- `data/top_confusions.csv`",
            "- `plots/split_target_distribution.png`",
            "- `plots/test_per_class_recall.png`",
            "- `plots/test_confusion_heatmaps.png`",
            "- `plots/test_accuracy_delta_contributions.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    dataset_path: Path,
    variant_roots: dict[str, Path],
) -> None:
    data_dir = output_dir / "data"
    plot_dir = output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    targets = load_target_events(dataset_path)
    num_marks = int(targets["mark"].max()) + 1
    distribution, distribution_summary = build_distribution_tables(targets, num_marks)
    (
        matrices,
        model_metrics,
        per_class,
        prediction_distribution,
        contributions,
        top_rows,
        validations,
    ) = load_confusions(variant_roots, distribution, num_marks)
    top_confusions = pl.DataFrame(top_rows).sort(
        ["eval_split", "variant", "count"], descending=[False, False, True]
    )

    distribution.write_csv(data_dir / "target_mark_distribution.csv")
    distribution_summary.write_csv(data_dir / "split_distribution_summary.csv")
    model_metrics.sort(["eval_split", "variant"]).write_csv(data_dir / "model_confusion_metrics.csv")
    per_class.sort(["eval_split", "variant", "mark"]).write_csv(data_dir / "per_class_metrics.csv")
    prediction_distribution.sort(["eval_split", "variant", "mark"]).write_csv(
        data_dir / "prediction_distribution.csv"
    )
    contributions.sort(["eval_split", "variant", "mark"]).write_csv(
        data_dir / "accuracy_contributions_vs_v2.csv"
    )
    top_confusions.write_csv(data_dir / "top_confusions.csv")

    plot_split_distribution(distribution, plot_dir / "split_target_distribution.png", num_marks)
    plot_per_class_recall(per_class, plot_dir / "test_per_class_recall.png", num_marks)
    plot_test_confusions(matrices, plot_dir / "test_confusion_heatmaps.png", num_marks)
    plot_accuracy_contributions(contributions, plot_dir / "test_accuracy_delta_contributions.png", num_marks)

    report = build_report(
        distribution,
        distribution_summary,
        model_metrics,
        per_class,
        contributions,
        top_confusions,
        matrices,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "status": "completed",
        "analysis": "intermittent_mark_imbalance_and_confusion",
        "dataset_path": str(dataset_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selection": SELECTION,
        "variants": {name: str(path.resolve()) for name, path in variant_roots.items()},
        "num_marks": num_marks,
        "target_counts": {
            split: int(row_for(distribution_summary, split=split)["target_count"])
            for split in SPLIT_ORDER
        },
        "source_validations": validations,
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    variant_roots = {
        "V2": args.v2_dir,
        "V3a": args.v3a_dir,
        "V3b": args.v3b_dir,
        "V3c": args.v3c_dir,
    }
    write_outputs(args.output_dir, args.dataset, variant_roots)
    print(f"[mark-diagnostics] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
