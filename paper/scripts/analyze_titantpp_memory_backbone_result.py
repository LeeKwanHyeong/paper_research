#!/usr/bin/env python3
"""Diagnose the completed TitanTPP memory-backbone validation screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_titan_memory_result")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_titan_memory_result")
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame


CONTROL = "titantpp"
BACKBONE_ORDER = (
    "titantpp",
    "titantpp_no_memory",
    "titantpp_gated_soft_memory",
    "titantpp_surprise_memory",
)
BACKBONE_LABELS = {
    "titantpp": "Hard-LMM",
    "titantpp_no_memory": "No memory",
    "titantpp_gated_soft_memory": "Gated soft",
    "titantpp_surprise_memory": "Surprise",
}
COLORS = {
    "titantpp": "#264653",
    "titantpp_no_memory": "#8d99ae",
    "titantpp_gated_soft_memory": "#e9c46a",
    "titantpp_surprise_memory": "#e76f51",
}
HISTORY_SPECS = (
    ("history_le_64", "<=64", 0, 64),
    ("history_65_128", "65-128", 65, 128),
    ("history_gt_128", ">128", 129, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def pct_improvement(control: float, candidate: float) -> float:
    return 100.0 * (control - candidate) / control


def validate_artifact(
    artifact_dir: Path,
    data_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads((artifact_dir / "launch_contract.json").read_text())
    manifest_lines = (artifact_dir / "source_manifest.txt").read_text().splitlines()
    manifest_shas: dict[str, str] = {}
    for line in manifest_lines:
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
        if not match:
            continue
        relative = match.group(2).split("/paper_research/", maxsplit=1)[-1]
        manifest_shas[relative] = match.group(1)

    source_mismatches = []
    for relative, expected in manifest_shas.items():
        source = PROJECT_ROOT / relative
        observed = sha256_file(source) if source.exists() else None
        if observed != expected:
            source_mismatches.append(
                {"path": relative, "expected": expected, "observed": observed}
            )

    log_text = (artifact_dir / "logs" / "launcher.log").read_text()
    error_matches = re.findall(
        r"(?im)^.*(?:traceback|runtimeerror|valueerror|cuda error|\bnan\b|\binf\b).*$",
        log_text,
    )
    summary_rows = read_csv(artifact_dir / "run_summaries.csv")
    test_artifacts = [
        str(path.relative_to(artifact_dir))
        for path in artifact_dir.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    finite_fields = (
        "best_val_joint_objective",
        "best_val_time_nll",
        "best_val_log_qty_mse",
        "best_val_qty_mae",
        "best_val_qty_rmse",
    )
    finite = all(
        math.isfinite(float(row[field]))
        for row in summary_rows
        for field in finite_fields
    )
    checks = {
        "contract_status_complete": contract.get("status") == "complete",
        "run_count_complete": int(contract.get("completed_run_count", -1))
        == int(contract.get("expected_run_count", -2))
        == len(BACKBONE_ORDER),
        "all_run_status_success": all(row["status"] == "success" for row in summary_rows),
        "source_sha_match": not source_mismatches,
        "data_sha_match": sha256_file(data_path) == contract["data_sha256"],
        "held_out_test_unused": contract.get("held_out_test_evaluated") is False,
        "test_artifacts_absent": not test_artifacts,
        "logs_finite_and_error_free": not error_matches and finite,
    }
    if not all(checks.values()):
        raise ValueError(
            "Artifact validation failed: "
            + json.dumps(
                {
                    "checks": checks,
                    "source_mismatches": source_mismatches,
                    "test_artifacts": test_artifacts,
                    "log_errors": error_matches,
                },
                ensure_ascii=False,
            )
        )
    return contract, {
        "checks": checks,
        "source_file_count": len(manifest_shas),
        "summary_run_count": len(summary_rows),
        "source_mismatches": source_mismatches,
        "test_artifacts": test_artifacts,
        "log_errors": error_matches,
    }


def load_histories(artifact_dir: Path) -> dict[str, list[dict[str, Any]]]:
    histories: dict[str, list[dict[str, Any]]] = {}
    for backbone in BACKBONE_ORDER:
        path = (
            artifact_dir
            / "runs"
            / backbone
            / "count_only_log_regression"
            / "seed_42"
            / "history.json"
        )
        histories[backbone] = json.loads(path.read_text())["history"]
    return histories


def summarize_overall(artifact_dir: Path) -> list[dict[str, Any]]:
    rows = {row["backbone"]: row for row in read_csv(artifact_dir / "run_summaries.csv")}
    if set(rows) != set(BACKBONE_ORDER):
        raise ValueError(f"Unexpected backbone set: {sorted(rows)}")
    control = rows[CONTROL]
    output = []
    for backbone in BACKBONE_ORDER:
        row = rows[backbone]
        output.append(
            {
                "backbone": backbone,
                "backbone_label": BACKBONE_LABELS[backbone],
                "best_epoch": int(row["best_epoch"]),
                "completed_epochs": int(row["completed_epochs"]),
                "stopped_early": row["stopped_early"],
                "joint_objective": as_float(row, "best_val_joint_objective"),
                "time_nll": as_float(row, "best_val_time_nll"),
                "log_qty_mse": as_float(row, "best_val_log_qty_mse"),
                "qty_mae": as_float(row, "best_val_qty_mae"),
                "qty_rmse": as_float(row, "best_val_qty_rmse"),
                "parameter_count": int(row["parameter_count"]),
                "time_nll_delta_vs_hard": as_float(row, "best_val_time_nll")
                - as_float(control, "best_val_time_nll"),
                "log_qty_mse_improvement_pct": pct_improvement(
                    as_float(control, "best_val_log_qty_mse"),
                    as_float(row, "best_val_log_qty_mse"),
                ),
                "qty_mae_improvement_pct": pct_improvement(
                    as_float(control, "best_val_qty_mae"),
                    as_float(row, "best_val_qty_mae"),
                ),
                "qty_rmse_improvement_pct": pct_improvement(
                    as_float(control, "best_val_qty_rmse"),
                    as_float(row, "best_val_qty_rmse"),
                ),
            }
        )
    return output


def summarize_breakdown_deltas(
    artifact_dir: Path,
    filename: str,
) -> list[dict[str, Any]]:
    rows = read_csv(artifact_dir / filename)
    lookup = {(row["backbone"], row["stratum"]): row for row in rows}
    strata = sorted(
        {row["stratum"]: int(row["stratum_order"]) for row in rows}.items(),
        key=lambda item: item[1],
    )
    output = []
    for backbone in BACKBONE_ORDER[1:]:
        for stratum, order in strata:
            control = lookup[(CONTROL, stratum)]
            candidate = lookup[(backbone, stratum)]
            share = as_float(candidate, "share")
            time_delta = as_float(candidate, "time_nll_mean") - as_float(
                control, "time_nll_mean"
            )
            output.append(
                {
                    "backbone": backbone,
                    "backbone_label": BACKBONE_LABELS[backbone],
                    "stratum_order": order,
                    "stratum": stratum,
                    "stratum_label": candidate["stratum_label"],
                    "count": int(candidate["count"]),
                    "share": share,
                    "time_nll_delta_vs_hard": time_delta,
                    "weighted_time_nll_delta_contribution": share * time_delta,
                    "log_qty_mse_improvement_pct": pct_improvement(
                        as_float(control, "log_qty_mse_mean"),
                        as_float(candidate, "log_qty_mse_mean"),
                    ),
                    "qty_mae_improvement_pct": pct_improvement(
                        as_float(control, "qty_mae_mean"),
                        as_float(candidate, "qty_mae_mean"),
                    ),
                    "qty_rmse_improvement_pct": pct_improvement(
                        as_float(control, "qty_rmse_mean"),
                        as_float(candidate, "qty_rmse_mean"),
                    ),
                }
            )
    return output


def summarize_trajectories(
    histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for backbone in BACKBONE_ORDER:
        history = histories[backbone]
        by_joint = min(history, key=lambda row: float(row["val_joint_objective"]))
        by_time = min(history, key=lambda row: float(row["val_time_nll"]))
        by_log = min(history, key=lambda row: float(row["val_log_qty_mse"]))
        by_mae = min(history, key=lambda row: float(row["val_qty_mae"]))
        late = history[-40:]
        rows.append(
            {
                "backbone": backbone,
                "backbone_label": BACKBONE_LABELS[backbone],
                "completed_epochs": int(history[-1]["epoch"]),
                "best_joint_epoch": int(by_joint["epoch"]),
                "best_time_epoch": int(by_time["epoch"]),
                "best_time_nll_any_epoch": float(by_time["val_time_nll"]),
                "time_nll_at_best_joint": float(by_joint["val_time_nll"]),
                "best_log_qty_mse_epoch": int(by_log["epoch"]),
                "best_log_qty_mse_any_epoch": float(by_log["val_log_qty_mse"]),
                "best_qty_mae_epoch": int(by_mae["epoch"]),
                "best_qty_mae_any_epoch": float(by_mae["val_qty_mae"]),
                "late40_time_nll_mean": float(
                    np.mean([float(row["val_time_nll"]) for row in late])
                ),
                "late40_log_qty_mse_median": float(
                    np.median([float(row["val_log_qty_mse"]) for row in late])
                ),
                "late40_qty_mae_median": float(
                    np.median([float(row["val_qty_mae"]) for row in late])
                ),
                "late40_qty_rmse_median": float(
                    np.median([float(row["val_qty_rmse"]) for row in late])
                ),
            }
        )
    return rows


def validation_history_profile(
    data_path: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    frame = prepare_count_frame(pl.read_parquet(data_path))
    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=int(contract["lookback_weeks"]),
        max_seq_len=int(contract["max_seq_len"]),
        val_ratio=0.2,
        mode="all",
        split_col="chronological_split",
        target_splits={"validation"},
    )
    samples: list[tuple[int, float, int]] = []
    for part_index, context_end in dataset.index:
        sequence = np.asarray(dataset.seq_lists[part_index], dtype=np.int32)
        left = int(sequence[context_end]) - (dataset.W - 1)
        context_start = int(np.searchsorted(sequence, left, side="left"))
        history_length = min(context_end - context_start + 1, dataset.max_len - 1)
        target_dt = max(1.0, float(dataset.dt_lists[part_index][context_end + 1]))
        samples.append((history_length, target_dt, part_index))

    output = []
    total = len(samples)
    for stratum, label, lower, upper in HISTORY_SPECS:
        selected = [sample for sample in samples if lower <= sample[0] <= upper]
        histories = np.asarray([sample[0] for sample in selected], dtype=np.float64)
        dts = np.asarray([sample[1] for sample in selected], dtype=np.float64)
        output.append(
            {
                "stratum": stratum,
                "stratum_label": label,
                "count": len(selected),
                "share": len(selected) / total,
                "series_count": len({sample[2] for sample in selected}),
                "history_mean": float(histories.mean()),
                "history_p50": float(np.median(histories)),
                "target_dt_min": float(dts.min()),
                "target_dt_mean": float(dts.mean()),
                "target_dt_p50": float(np.median(dts)),
                "target_dt_p90": float(np.quantile(dts, 0.90)),
                "target_dt_p99": float(np.quantile(dts, 0.99)),
                "target_dt_max": float(dts.max()),
                "target_dt_eq_1_share": float(np.mean(dts == 1.0)),
            }
        )
    if sum(row["count"] for row in output) != total:
        raise ValueError("Validation history strata do not cover every target")
    return output


def target_time_profile(
    data_path: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize target times without using the held-out test split."""
    frame = prepare_count_frame(pl.read_parquet(data_path))
    output = []
    for split in ("train", "validation"):
        dataset = RMTPPWeekLookbackDataset(
            frame,
            lookback_weeks=int(contract["lookback_weeks"]),
            max_seq_len=int(contract["max_seq_len"]),
            val_ratio=0.2,
            mode="all",
            split_col="chronological_split",
            target_splits={split},
        )
        target_dts = np.asarray(
            [
                max(1.0, float(dataset.dt_lists[part_index][context_end + 1]))
                for part_index, context_end in dataset.index
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "split": split,
                "target_count": int(target_dts.size),
                "target_dt_min": float(target_dts.min()),
                "target_dt_mean": float(target_dts.mean()),
                "target_dt_p50": float(np.quantile(target_dts, 0.50)),
                "target_dt_p90": float(np.quantile(target_dts, 0.90)),
                "target_dt_p95": float(np.quantile(target_dts, 0.95)),
                "target_dt_p99": float(np.quantile(target_dts, 0.99)),
                "target_dt_max": float(target_dts.max()),
                "target_dt_eq_1_share": float(np.mean(target_dts == 1.0)),
            }
        )
    return output


def checkpoint_diagnostics(
    artifact_dir: Path,
    minimum_target_dt: float,
) -> list[dict[str, Any]]:
    rows = []
    for backbone in BACKBONE_ORDER:
        checkpoint = (
            artifact_dir
            / "runs"
            / backbone
            / "count_only_log_regression"
            / "seed_42"
            / "best_val_joint_objective_model.pt"
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        raw_w = float(state["w_raw"])
        w = float(torch.nn.functional.softplus(state["w_raw"]) + 1e-3)
        saturation_dt = 10.0 / w
        row: dict[str, Any] = {
            "backbone": backbone,
            "backbone_label": BACKBONE_LABELS[backbone],
            "w_raw": raw_w,
            "w_positive": w,
            "wd_clamp": 10.0,
            "saturation_dt": saturation_dt,
            "minimum_validation_target_dt": minimum_target_dt,
            "all_validation_targets_wd_clamped": saturation_dt < minimum_target_dt,
            "time_projection_norm": float(state["v_t.weight"].norm()),
            "time_bias": float(state["b_t"]),
            "quantity_projection_norm": float(state["quantity_head.weight"].norm()),
            "memory_residual_scale_raw": None,
            "memory_residual_scale_tanh": None,
            "surprise_momentum": None,
            "surprise_update_bias_rate": None,
            "surprise_retention_bias_rate": None,
            "surprise_retention_bias_half_life_events": None,
            "surprise_update_weight_norm": None,
            "surprise_retention_weight_norm": None,
        }
        for prefix in ("soft_memory", "surprise_memory"):
            key = f"{prefix}.residual_scale"
            if key in state:
                row["memory_residual_scale_raw"] = float(state[key])
                row["memory_residual_scale_tanh"] = float(torch.tanh(state[key]))
        if "surprise_memory.momentum_logit" in state:
            retention = float(
                torch.sigmoid(state["surprise_memory.retention_proj.bias"])
            )
            row.update(
                {
                    "surprise_momentum": float(
                        torch.sigmoid(state["surprise_memory.momentum_logit"])
                    ),
                    "surprise_update_bias_rate": float(
                        torch.sigmoid(state["surprise_memory.update_rate_proj.bias"])
                    ),
                    "surprise_retention_bias_rate": retention,
                    "surprise_retention_bias_half_life_events": math.log(0.5)
                    / math.log(retention),
                    "surprise_update_weight_norm": float(
                        state["surprise_memory.update_rate_proj.weight"].norm()
                    ),
                    "surprise_retention_weight_norm": float(
                        state["surprise_memory.retention_proj.weight"].norm()
                    ),
                }
            )
        rows.append(row)
    return rows


def rolling_mean(values: np.ndarray, window: int = 7) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    prefix = np.full(window - 1, np.nan)
    return np.concatenate((prefix, np.convolve(values, kernel, mode="valid")))


def plot_learning_curves(
    histories: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    specs = (
        ("val_time_nll", "Validation time NLL", False),
        ("val_time_nll", "Validation time NLL (epoch >= 100)", True),
        ("val_log_qty_mse", "Validation log1p quantity MSE", False),
        ("val_qty_mae", "Validation raw quantity MAE", False),
    )
    for axis, (metric, title, late_only) in zip(axes.flat, specs):
        for backbone in BACKBONE_ORDER:
            rows = histories[backbone]
            epochs = np.asarray([int(row["epoch"]) for row in rows])
            values = np.asarray([float(row[metric]) for row in rows])
            selected = epochs >= 100 if late_only else np.ones_like(epochs, dtype=bool)
            axis.plot(
                epochs[selected],
                values[selected],
                color=COLORS[backbone],
                alpha=0.18,
                linewidth=0.8,
            )
            smoothed = rolling_mean(values)
            axis.plot(
                epochs[selected],
                smoothed[selected],
                color=COLORS[backbone],
                linewidth=1.8,
                label=BACKBONE_LABELS[backbone],
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("TitanTPP memory-backbone validation trajectories (seed 42)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def grouped_bar(
    axis: Any,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    strata = sorted(
        {row["stratum"]: (int(row["stratum_order"]), row["stratum_label"]) for row in rows}.items(),
        key=lambda item: item[1][0],
    )
    positions = np.arange(len(strata), dtype=np.float64)
    width = 0.24
    for index, backbone in enumerate(BACKBONE_ORDER[1:]):
        lookup = {
            row["stratum"]: float(row[metric])
            for row in rows
            if row["backbone"] == backbone
        }
        values = [lookup[stratum] for stratum, _ in strata]
        axis.bar(
            positions + (index - 1) * width,
            values,
            width=width,
            label=BACKBONE_LABELS[backbone],
            color=COLORS[backbone],
        )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(positions, [label for _, (_, label) in strata])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)


def plot_history_tradeoff(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    grouped_bar(
        axes[0],
        rows,
        "time_nll_delta_vs_hard",
        "Time NLL regression is isolated to long histories",
        "Candidate - hard-LMM (lower is better)",
    )
    grouped_bar(
        axes[1],
        rows,
        "qty_mae_improvement_pct",
        "Quantity MAE improves across history lengths",
        "Improvement vs hard-LMM (%)",
    )
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quantity_improvement(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    grouped_bar(
        axis,
        rows,
        "qty_mae_improvement_pct",
        "Quantity MAE improvement by validation quantity stratum",
        "Improvement vs hard-LMM (%)",
    )
    axis.legend(frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    contract, artifact_validation = validate_artifact(artifact_dir, data_path)
    histories = load_histories(artifact_dir)
    overall = summarize_overall(artifact_dir)
    trajectory = summarize_trajectories(histories)
    history_deltas = summarize_breakdown_deltas(artifact_dir, "history_summary.csv")
    quantity_deltas = summarize_breakdown_deltas(artifact_dir, "quantity_summary.csv")
    history_profile = validation_history_profile(data_path, contract)
    time_profile = target_time_profile(data_path, contract)
    minimum_target_dt = min(row["target_dt_min"] for row in history_profile)
    checkpoint_rows = checkpoint_diagnostics(artifact_dir, minimum_target_dt)

    overall_lookup = {row["backbone"]: row for row in overall}
    surprise = overall_lookup["titantpp_surprise_memory"]
    surprise_history = [
        row for row in history_deltas if row["backbone"] == "titantpp_surprise_memory"
    ]
    weighted_delta = sum(
        row["weighted_time_nll_delta_contribution"] for row in surprise_history
    )
    if not math.isclose(
        weighted_delta,
        surprise["time_nll_delta_vs_hard"],
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("History-stratum Time NLL deltas do not reconcile")
    long_contribution = next(
        row["weighted_time_nll_delta_contribution"]
        for row in surprise_history
        if row["stratum"] == "history_gt_128"
    )
    surprise_checkpoint = next(
        row for row in checkpoint_rows if row["backbone"] == "titantpp_surprise_memory"
    )
    train_time = next(row for row in time_profile if row["split"] == "train")
    candidate_time_scale = train_time["target_dt_p50"]
    candidate_w_max = 40.0 / (train_time["target_dt_max"] / candidate_time_scale)
    payload = {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "source_revision": contract["source_revision"],
        "evaluation_scope": contract["evaluation_scope"],
        "held_out_test_evaluated": contract["held_out_test_evaluated"],
        "artifact_validation": artifact_validation,
        "headline": {
            "strict_joint_gate_status": "fail",
            "selected_by_original_gate": "titantpp",
            "quantity_primary_candidate": "titantpp_surprise_memory",
            "surprise_qty_mae_improvement_pct": surprise["qty_mae_improvement_pct"],
            "surprise_qty_rmse_improvement_pct": surprise["qty_rmse_improvement_pct"],
            "surprise_log_qty_mse_improvement_pct": surprise[
                "log_qty_mse_improvement_pct"
            ],
            "surprise_time_nll_delta": surprise["time_nll_delta_vs_hard"],
            "surprise_long_history_time_delta_contribution_share": (
                long_contribution / weighted_delta
            ),
            "time_head_w": surprise_checkpoint["w_positive"],
            "time_head_saturation_dt": surprise_checkpoint["saturation_dt"],
            "all_validation_targets_wd_clamped": surprise_checkpoint[
                "all_validation_targets_wd_clamped"
            ],
            "train_target_dt_p50": candidate_time_scale,
            "train_target_dt_max": train_time["target_dt_max"],
            "candidate_scaled_exact_w_max": candidate_w_max,
        },
        "caveats": [
            "Single-seed validation-only screening; no held-out test was read.",
            "Hard-LMM also retains 16 persistent tokens, while all three candidates remove them.",
            "The common RMTPP-style time head is fully wd-clamped at the selected checkpoints.",
            "All validation targets therefore share the same clamped duration term.",
            "Raw quantity MAE/RMSE vary materially across adjacent late epochs.",
        ],
    }

    write_csv(output_dir / "overall_metrics_and_deltas.csv", overall)
    write_csv(output_dir / "trajectory_diagnostics.csv", trajectory)
    write_csv(output_dir / "history_stratum_deltas.csv", history_deltas)
    write_csv(output_dir / "quantity_stratum_deltas.csv", quantity_deltas)
    write_csv(output_dir / "validation_history_profile.csv", history_profile)
    write_csv(output_dir / "target_time_profile.csv", time_profile)
    write_csv(output_dir / "checkpoint_diagnostics.csv", checkpoint_rows)
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_learning_curves(histories, plots_dir / "validation_learning_curves.png")
    plot_history_tradeoff(history_deltas, plots_dir / "history_tradeoff.png")
    plot_quantity_improvement(
        quantity_deltas,
        plots_dir / "quantity_scale_improvement.png",
    )
    print(json.dumps(payload["headline"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
