#!/usr/bin/env python3
"""Analyze the TitanTPP V4 Taxi 2x2 validation-only screening artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "search_artifacts/model_enhancement_titantpp_v4_taxi_2x2_seed42_e50_0716"
)
VARIANTS = {
    "v2_shared_value_shared_time": {
        "value_head_mode": "shared",
        "time_head_mode": "shared",
    },
    "v3b_mark_value_shared_time": {
        "value_head_mode": "mark_conditioned_experts",
        "time_head_mode": "shared",
    },
    "v4a_shared_value_mark_time": {
        "value_head_mode": "shared",
        "time_head_mode": "mark_conditioned",
    },
    "v4b_mark_value_mark_time": {
        "value_head_mode": "mark_conditioned_experts",
        "time_head_mode": "mark_conditioned",
    },
}
PAIRS = {
    "v4a_vs_v2": ("v2_shared_value_shared_time", "v4a_shared_value_mark_time"),
    "v4b_vs_v3b": (
        "v3b_mark_value_shared_time",
        "v4b_mark_value_mark_time",
    ),
}
LOWER_HISTORY_METRICS = (
    "val_nll",
    "val_nll_marker",
    "val_nll_time",
    "dt_mae",
    "qty_mae",
    "log_qty_mae",
    "train_loss",
)
HIGHER_HISTORY_METRICS = (
    "mark_acc",
    "mark_balanced_accuracy",
    "mark_macro_f1",
)
EXPECTED_VALIDATION_COUNT = 8_268


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def one_path(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def finite(value: Any) -> bool:
    return math.isfinite(float(value))


def lower_improvement_pct(candidate: float, control: float) -> float:
    if control == 0.0:
        raise ValueError("Control value cannot be zero for a relative comparison")
    return 100.0 * (control - candidate) / abs(control)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def pstdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def validate_held_out_lock(root: Path) -> dict[str, Any]:
    forbidden = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.startswith("test_")
            or "plots/test" in path.as_posix()
            or path.name.startswith("paper_table_test")
        )
    )
    if forbidden:
        raise ValueError(f"Held-out artifacts exist in validation-only root: {forbidden}")
    return {
        "held_out_artifact_count": 0,
        "mixed_report_read": False,
        "held_out_lock": "PASS",
    }


def assert_finite_columns(frame: pl.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        values = frame[column].to_list()
        if any(not finite(value) for value in values):
            raise ValueError(f"Non-finite active metric in {label}: {column}")


def history_row(frame: pl.DataFrame, epoch: int) -> dict[str, Any]:
    rows = frame.filter(pl.col("epoch") == epoch).to_dicts()
    if len(rows) != 1:
        raise ValueError(f"Expected one history row for epoch {epoch}, found {len(rows)}")
    return rows[0]


def load_variant(root: Path, name: str, contract: dict[str, str]) -> dict[str, Any]:
    variant_root = root / "variants" / name
    summary_path = one_path(
        list(variant_root.glob("runs/**/metrics/summary.json")), f"{name} summary"
    )
    scale_path = one_path(
        list(variant_root.glob("runs/**/metrics/scale_wise_best_val_nll.csv")),
        f"{name} validation scale metrics",
    )
    class_path = one_path(
        list(
            variant_root.glob(
                "runs/**/metrics/validation_mark_class_metrics_best_val_nll.csv"
            )
        ),
        f"{name} validation class metrics",
    )
    confusion_path = one_path(
        list(
            variant_root.glob(
                "runs/**/metrics/validation_mark_confusion_best_val_nll.csv"
            )
        ),
        f"{name} validation confusion metrics",
    )
    history_path = variant_root / "leaderboard" / "histories.csv"

    summary = read_json(summary_path)
    history = pl.read_csv(history_path).sort("epoch")
    scale = pl.read_csv(scale_path).sort("scale_order")
    classes = pl.read_csv(class_path).sort("mark")
    confusion = pl.read_csv(confusion_path).sort(["true_mark", "pred_mark"])

    expected_identity: dict[str, Any] = {
        **contract,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "status": "success",
        "epochs": 50,
        "seed": 42,
    }
    for key, expected in expected_identity.items():
        if summary.get(key) != expected:
            raise ValueError(f"{name} identity mismatch: {key}={summary.get(key)!r}")
    if history.height != 50 or history["epoch"].to_list() != list(range(1, 51)):
        raise ValueError(f"{name} history is not exactly epochs 1..50")
    assert_finite_columns(
        history,
        LOWER_HISTORY_METRICS + HIGHER_HISTORY_METRICS,
        f"{name} history",
    )

    for label, frame in {
        "scale": scale,
        "class": classes,
        "confusion": confusion,
    }.items():
        if frame["eval_split"].unique().to_list() != ["validation"]:
            raise ValueError(f"{name} {label} artifact is not validation-only")
        if frame["selection"].unique().to_list() != ["best_val_nll"]:
            raise ValueError(f"{name} {label} artifact selection mismatch")

    nonempty_scale = scale.filter(pl.col("count") > 0)
    scale_count = int(nonempty_scale["count"].sum())
    class_true_count = int(classes["true_count"].sum())
    class_pred_count = int(classes["pred_count"].sum())
    confusion_count = int(confusion["count"].sum())
    counts = {scale_count, class_true_count, class_pred_count, confusion_count}
    if counts != {EXPECTED_VALIDATION_COUNT}:
        raise ValueError(f"{name} validation counts do not reconcile: {counts}")

    best_epoch = int(summary["best_val_nll_epoch"])
    best_row = history_row(history, best_epoch)
    summary_checks = {
        "best_val_nll": "val_nll",
        "best_val_nll_qty_mae": "qty_mae",
        "best_val_nll_dt_mae": "dt_mae",
        "best_val_nll_mark_acc": "mark_acc",
    }
    for summary_key, history_key in summary_checks.items():
        if not math.isclose(
            float(summary[summary_key]),
            float(best_row[history_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name} summary/history mismatch: {summary_key}")

    weighted_qty_mae = float(
        (nonempty_scale["count"] * nonempty_scale["qty_mae"]).sum()
        / nonempty_scale["count"].sum()
    )
    class_accuracy = float(classes["correct_count"].sum() / classes["true_count"].sum())
    confusion_accuracy = float(
        confusion.filter(pl.col("true_mark") == pl.col("pred_mark"))["count"].sum()
        / confusion["count"].sum()
    )
    for label, observed, expected in (
        ("scale weighted quantity MAE", weighted_qty_mae, float(best_row["qty_mae"])),
        ("class accuracy", class_accuracy, float(best_row["mark_acc"])),
        ("confusion accuracy", confusion_accuracy, float(best_row["mark_acc"])),
    ):
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{name} {label} does not reconcile: {observed} vs {expected}")

    return {
        "summary": summary,
        "history": history,
        "scale": scale,
        "classes": classes,
        "confusion": confusion,
        "integrity": {
            "history_rows": history.height,
            "validation_count": scale_count,
            "weighted_qty_mae": weighted_qty_mae,
            "class_accuracy": class_accuracy,
            "confusion_accuracy": confusion_accuracy,
            "gate": "PASS",
        },
    }


def summarize_history(data: dict[str, Any]) -> dict[str, Any]:
    history: pl.DataFrame = data["history"]
    summary = data["summary"]
    best_epoch = int(summary["best_val_nll_epoch"])
    best = history_row(history, best_epoch)
    final = history_row(history, 50)
    min_time = history.sort("val_nll_time").row(0, named=True)
    max_dt = history.sort("dt_mae", descending=True).row(0, named=True)
    last_ten = history.filter(pl.col("epoch") >= 41)

    final_vs_best = {
        metric: 100.0
        * (float(final[metric]) - float(best[metric]))
        / abs(float(best[metric]))
        for metric in (
            "val_nll",
            "val_nll_marker",
            "val_nll_time",
            "dt_mae",
            "qty_mae",
        )
    }
    final_vs_best["mark_acc_delta_pp"] = 100.0 * (
        float(final["mark_acc"]) - float(best["mark_acc"])
    )
    last_ten_stability = {
        metric: {
            "mean": mean([float(value) for value in last_ten[metric].to_list()]),
            "std": pstdev([float(value) for value in last_ten[metric].to_list()]),
        }
        for metric in ("val_nll", "val_nll_time", "dt_mae", "qty_mae")
    }
    return {
        "best_epoch": best_epoch,
        "best_metrics": {
            metric: float(best[metric])
            for metric in LOWER_HISTORY_METRICS + HIGHER_HISTORY_METRICS
        },
        "final_metrics": {
            metric: float(final[metric])
            for metric in LOWER_HISTORY_METRICS + HIGHER_HISTORY_METRICS
        },
        "final_vs_best_regression_pct": final_vs_best,
        "minimum_time_nll": {
            "epoch": int(min_time["epoch"]),
            "value": float(min_time["val_nll_time"]),
        },
        "maximum_dt_mae": {
            "epoch": int(max_dt["epoch"]),
            "value": float(max_dt["dt_mae"]),
        },
        "dt_mae_over_1_epoch_count": int(history.filter(pl.col("dt_mae") > 1.0).height),
        "last_ten": last_ten_stability,
    }


def summarize_pair_history(
    control: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control_history: pl.DataFrame = control["history"]
    candidate_history: pl.DataFrame = candidate["history"]
    metrics = list(LOWER_HISTORY_METRICS + HIGHER_HISTORY_METRICS)
    joined = control_history.select(["epoch", *metrics]).join(
        candidate_history.select(["epoch", *metrics]),
        on="epoch",
        how="inner",
        suffix="_candidate",
    )
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for metric in LOWER_HISTORY_METRICS:
        changes = [
            lower_improvement_pct(float(row[f"{metric}_candidate"]), float(row[metric]))
            for row in joined.iter_rows(named=True)
        ]
        metric_rows = [
            {
                "epoch": int(epoch),
                "metric": metric,
                "candidate_improvement_pct": change,
            }
            for epoch, change in zip(joined["epoch"].to_list(), changes, strict=True)
        ]
        rows.extend(metric_rows)
        last_ten = changes[-10:]
        payload: dict[str, Any] = {
            "improved_epoch_count": sum(change > 0.0 for change in changes),
            "median_improvement_pct": median(changes),
            "mean_improvement_pct": mean(changes),
            "last_ten_mean_improvement_pct": mean(last_ten),
            "last_ten_improved_epoch_count": sum(change > 0.0 for change in last_ten),
        }
        if metric == "val_nll_time":
            payload["epochs_meeting_0p5_pct_gate"] = sum(
                change >= 0.5 for change in changes
            )
            payload["last_ten_epochs_meeting_0p5_pct_gate"] = sum(
                change >= 0.5 for change in last_ten
            )
        summary[metric] = payload

    for metric in HIGHER_HISTORY_METRICS:
        deltas = [
            100.0 * (float(row[f"{metric}_candidate"]) - float(row[metric]))
            for row in joined.iter_rows(named=True)
        ]
        rows.extend(
            {
                "epoch": int(epoch),
                "metric": metric,
                "candidate_delta_pp": delta,
            }
            for epoch, delta in zip(joined["epoch"].to_list(), deltas, strict=True)
        )
        summary[metric] = {
            "improved_epoch_count": sum(delta > 0.0 for delta in deltas),
            "median_delta_pp": median(deltas),
            "mean_delta_pp": mean(deltas),
            "last_ten_mean_delta_pp": mean(deltas[-10:]),
        }

    control_best_epoch = int(control["summary"]["best_val_nll_epoch"])
    candidate_best_epoch = int(candidate["summary"]["best_val_nll_epoch"])
    checkpoint_context: dict[str, Any] = {}
    for label, epoch in {
        "control_best_epoch": control_best_epoch,
        "candidate_best_epoch": candidate_best_epoch,
    }.items():
        row = joined.filter(pl.col("epoch") == epoch).row(0, named=True)
        checkpoint_context[label] = {
            "epoch": epoch,
            "lower_metric_improvement_pct": {
                metric: lower_improvement_pct(
                    float(row[f"{metric}_candidate"]), float(row[metric])
                )
                for metric in LOWER_HISTORY_METRICS
            },
            "higher_metric_delta_pp": {
                metric: 100.0
                * (float(row[f"{metric}_candidate"]) - float(row[metric]))
                for metric in HIGHER_HISTORY_METRICS
            },
        }
    summary["checkpoint_context"] = checkpoint_context
    return summary, rows


def summarize_scale_pair(
    pair_name: str,
    control_name: str,
    candidate_name: str,
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control_scale: pl.DataFrame = control["scale"].filter(pl.col("count") > 0)
    candidate_scale: pl.DataFrame = candidate["scale"].filter(pl.col("count") > 0)
    joined = control_scale.join(
        candidate_scale,
        on=["scale_order", "scale_label", "count"],
        how="inner",
        suffix="_candidate",
    ).sort("scale_order")
    rows: list[dict[str, Any]] = []
    for row in joined.iter_rows(named=True):
        count = int(row["count"])
        control_mae = float(row["qty_mae"])
        candidate_mae = float(row["qty_mae_candidate"])
        rows.append(
            {
                "pair": pair_name,
                "control": control_name,
                "candidate": candidate_name,
                "scale_order": int(row["scale_order"]),
                "scale_label": str(row["scale_label"]),
                "count": count,
                "share": float(row["share"]),
                "control_qty_mae": control_mae,
                "candidate_qty_mae": candidate_mae,
                "qty_mae_improvement_pct": lower_improvement_pct(
                    candidate_mae, control_mae
                ),
                "qty_median_ae_improvement_pct": lower_improvement_pct(
                    float(row["qty_median_ae_candidate"]),
                    float(row["qty_median_ae"]),
                ),
                "qty_wape_improvement_pct": lower_improvement_pct(
                    float(row["qty_wape_candidate"]), float(row["qty_wape"])
                ),
                "log_abs_error_improvement_pct": lower_improvement_pct(
                    float(row["log_abs_error_candidate"]),
                    float(row["log_abs_error"]),
                ),
                "control_prediction_bias": float(row["pred_qty_mean"])
                - float(row["true_qty_mean"]),
                "candidate_prediction_bias": float(row["pred_qty_mean_candidate"])
                - float(row["true_qty_mean"]),
                "mark_accuracy_delta_pp": 100.0
                * (float(row["mark_acc_candidate"]) - float(row["mark_acc"])),
                "absolute_error_reduction": count * (control_mae - candidate_mae),
            }
        )
    total_reduction = sum(float(row["absolute_error_reduction"]) for row in rows)
    for row in rows:
        row["reduction_contribution_pct"] = (
            100.0 * float(row["absolute_error_reduction"]) / total_reduction
            if total_reduction != 0.0
            else 0.0
        )

    control_weighted = float(control["integrity"]["weighted_qty_mae"])
    candidate_weighted = float(candidate["integrity"]["weighted_qty_mae"])
    return {
        "control_weighted_qty_mae": control_weighted,
        "candidate_weighted_qty_mae": candidate_weighted,
        "weighted_qty_mae_improvement_pct": lower_improvement_pct(
            candidate_weighted, control_weighted
        ),
        "all_nonempty_scales_improved": all(
            float(row["qty_mae_improvement_pct"]) > 0.0 for row in rows
        ),
        "largest_reduction_scale": max(
            rows, key=lambda row: float(row["absolute_error_reduction"])
        )["scale_label"],
    }, rows


def confusion_summary(frame: pl.DataFrame) -> dict[str, Any]:
    rows = frame.to_dicts()
    total = sum(int(row["count"]) for row in rows)
    exact = sum(
        int(row["count"])
        for row in rows
        if int(row["true_mark"]) == int(row["pred_mark"])
    )
    adjacent = sum(
        int(row["count"])
        for row in rows
        if abs(int(row["true_mark"]) - int(row["pred_mark"])) == 1
    )
    nonadjacent = sum(
        int(row["count"])
        for row in rows
        if abs(int(row["true_mark"]) - int(row["pred_mark"])) > 1
    )
    upward = sum(
        int(row["count"])
        for row in rows
        if int(row["pred_mark"]) > int(row["true_mark"])
    )
    downward = sum(
        int(row["count"])
        for row in rows
        if int(row["pred_mark"]) < int(row["true_mark"])
    )
    mark_abs_error = sum(
        int(row["count"])
        * abs(int(row["true_mark"]) - int(row["pred_mark"]))
        for row in rows
    )
    error_count = total - exact
    return {
        "total": total,
        "correct": exact,
        "errors": error_count,
        "accuracy": exact / total,
        "adjacent_error_count": adjacent,
        "nonadjacent_error_count": nonadjacent,
        "adjacent_share_of_errors": adjacent / error_count if error_count else 0.0,
        "upward_error_count": upward,
        "downward_error_count": downward,
        "mark_mae": mark_abs_error / total,
    }


def summarize_class_pair(
    pair_name: str,
    control_name: str,
    candidate_name: str,
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    joined = control["classes"].join(
        candidate["classes"], on=["mark", "true_count"], how="inner", suffix="_candidate"
    ).sort("mark")
    rows: list[dict[str, Any]] = []
    for row in joined.iter_rows(named=True):
        control_share_error = abs(float(row["pred_share"]) - float(row["true_share"]))
        candidate_share_error = abs(
            float(row["pred_share_candidate"]) - float(row["true_share"])
        )
        rows.append(
            {
                "pair": pair_name,
                "control": control_name,
                "candidate": candidate_name,
                "mark": int(row["mark"]),
                "support": int(row["true_count"]),
                "true_share": float(row["true_share"]),
                "control_pred_share": float(row["pred_share"]),
                "candidate_pred_share": float(row["pred_share_candidate"]),
                "control_precision": float(row["precision"]),
                "candidate_precision": float(row["precision_candidate"]),
                "control_recall": float(row["recall"]),
                "candidate_recall": float(row["recall_candidate"]),
                "control_f1": float(row["f1"]),
                "candidate_f1": float(row["f1_candidate"]),
                "correct_count_delta": int(row["correct_count_candidate"])
                - int(row["correct_count"]),
                "precision_delta_pp": 100.0
                * (float(row["precision_candidate"]) - float(row["precision"])),
                "recall_delta_pp": 100.0
                * (float(row["recall_candidate"]) - float(row["recall"])),
                "f1_delta_pp": 100.0
                * (float(row["f1_candidate"]) - float(row["f1"])),
                "control_pred_share_error_pp": 100.0 * control_share_error,
                "candidate_pred_share_error_pp": 100.0 * candidate_share_error,
                "pred_share_error_change_pp": 100.0
                * (candidate_share_error - control_share_error),
            }
        )

    control_confusion = confusion_summary(control["confusion"])
    candidate_confusion = confusion_summary(candidate["confusion"])
    control_tv = 50.0 * sum(
        abs(float(row["pred_share"]) - float(row["true_share"]))
        for row in control["classes"].iter_rows(named=True)
    )
    candidate_tv = 50.0 * sum(
        abs(float(row["pred_share"]) - float(row["true_share"]))
        for row in candidate["classes"].iter_rows(named=True)
    )
    return {
        "control": control_confusion,
        "candidate": candidate_confusion,
        "correct_count_delta": candidate_confusion["correct"]
        - control_confusion["correct"],
        "accuracy_delta_pp": 100.0
        * (candidate_confusion["accuracy"] - control_confusion["accuracy"]),
        "adjacent_error_count_delta": candidate_confusion["adjacent_error_count"]
        - control_confusion["adjacent_error_count"],
        "nonadjacent_error_count_delta": candidate_confusion[
            "nonadjacent_error_count"
        ]
        - control_confusion["nonadjacent_error_count"],
        "predicted_share_tv_control_pp": control_tv,
        "predicted_share_tv_candidate_pp": candidate_tv,
        "predicted_share_tv_change_pp": candidate_tv - control_tv,
    }, rows


def confusion_pair_rows(
    pair_name: str,
    control_name: str,
    candidate_name: str,
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    control_counts = {
        (int(row["true_mark"]), int(row["pred_mark"])): int(row["count"])
        for row in control["confusion"].iter_rows(named=True)
    }
    candidate_counts = {
        (int(row["true_mark"]), int(row["pred_mark"])): int(row["count"])
        for row in candidate["confusion"].iter_rows(named=True)
    }
    support = {
        int(row["mark"]): int(row["true_count"])
        for row in control["classes"].iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for true_mark in sorted(support):
        for pred_mark in sorted(support):
            control_count = control_counts.get((true_mark, pred_mark), 0)
            candidate_count = candidate_counts.get((true_mark, pred_mark), 0)
            rows.append(
                {
                    "pair": pair_name,
                    "control": control_name,
                    "candidate": candidate_name,
                    "true_mark": true_mark,
                    "pred_mark": pred_mark,
                    "support": support[true_mark],
                    "control_count": control_count,
                    "candidate_count": candidate_count,
                    "count_delta": candidate_count - control_count,
                    "control_share_within_true": control_count / support[true_mark],
                    "candidate_share_within_true": candidate_count
                    / support[true_mark],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty analysis table: {path}")
    pl.DataFrame(rows).write_csv(path)


def main() -> None:
    args = parse_args()
    root = args.artifact_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "validation_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    held_out = validate_held_out_lock(root)
    screening = read_json(root / "screening_summary.json")
    variants = {
        name: load_variant(root, name, contract) for name, contract in VARIANTS.items()
    }
    variant_history = {
        name: summarize_history(data) for name, data in variants.items()
    }

    pair_history: dict[str, Any] = {}
    pair_scale: dict[str, Any] = {}
    pair_class: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for pair_name, (control_name, candidate_name) in PAIRS.items():
        history_summary, history_detail = summarize_pair_history(
            variants[control_name], variants[candidate_name]
        )
        pair_history[pair_name] = history_summary
        history_rows.extend({"pair": pair_name, **row} for row in history_detail)

        scale_summary, scale_detail = summarize_scale_pair(
            pair_name,
            control_name,
            candidate_name,
            variants[control_name],
            variants[candidate_name],
        )
        pair_scale[pair_name] = scale_summary
        scale_rows.extend(scale_detail)

        class_summary, class_detail = summarize_class_pair(
            pair_name,
            control_name,
            candidate_name,
            variants[control_name],
            variants[candidate_name],
        )
        pair_class[pair_name] = class_summary
        class_rows.extend(class_detail)
        confusion_rows.extend(
            confusion_pair_rows(
                pair_name,
                control_name,
                candidate_name,
                variants[control_name],
                variants[candidate_name],
            )
        )

    result = {
        "status": "PASS",
        "scope": "validation_only",
        "held_out": held_out,
        "source_revision": screening["source_revision"],
        "screening_decision": screening["decision"],
        "screening_pairs": screening["pairs"],
        "variant_integrity": {
            name: data["integrity"] for name, data in variants.items()
        },
        "variant_history": variant_history,
        "pair_history": pair_history,
        "pair_scale": pair_scale,
        "pair_class_confusion": pair_class,
        "final_decision": "retain_v2_and_taxi_v3b_do_not_promote_v4",
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(output_dir / "history_pairwise.csv", history_rows)
    write_csv(output_dir / "scale_pairwise.csv", scale_rows)
    write_csv(output_dir / "class_pairwise.csv", class_rows)
    write_csv(output_dir / "confusion_pairwise.csv", confusion_rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
