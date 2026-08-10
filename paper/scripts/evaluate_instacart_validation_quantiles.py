#!/usr/bin/env python3
"""Evaluate Instacart validation quantity errors by train-derived quantiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/titantpp-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from data_loader.event_seq_data_module import (
    RMTPPWeekLookbackDataset,
    collate_week_lookback,
)
from models.RMTPPs.config import RMTPPConfig, THPConfig
from models.RMTPPs.RMTPP import RMTPP
from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.TransformerHawkesTPP import TransformerHawkesTPP
from models.RMTPPs.value_conditioning import predict_value_for_marks
from models.Titan import TitanConfig
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    forward_model,
    torch_load_checkpoint,
)
from utils.training import TrainingConfig


QUANTILES = (0.50, 0.90, 0.95, 0.99)
MODEL_LABELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "titantpp": "TitanTPP",
}
MODEL_ORDER = tuple(MODEL_LABELS.values())
MODEL_COLORS = {
    "Adapted RMTPP": "#4477AA",
    "Adapted THP": "#228833",
    "TitanTPP": "#CC3311",
}
SEEDS = (42, 52, 62)
METRICS = (
    "qty_mae",
    "qty_rmse",
    "qty_wape",
    "qty_bias",
    "true_qty_mean",
    "pred_qty_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        required=True,
        help="Repeat for the baseline and TitanTPP experiment roots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Smoke-only batch limit. Omit for the qualified full analysis.",
    )
    parser.add_argument(
        "--expected-validation-count",
        type=int,
        default=503733,
    )
    return parser.parse_args()


def dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    return cls(**{name: value for name, value in values.items() if name in allowed})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch_load_checkpoint(path, map_location="cpu")
    if payload.get("selection") != "best_val_nll":
        raise ValueError(f"Unexpected checkpoint selection: {path}")

    experiment = payload.get("experiment_config", {})
    run = payload.get("run_config", {})
    summary = payload.get("summary", {})
    if experiment.get("evaluation_scope") != "validation_only":
        raise ValueError(f"Checkpoint is not validation-only: {path}")
    if bool(summary.get("held_out_test_evaluated", True)):
        raise ValueError(f"Held-out test flag is not locked: {path}")
    if run.get("dataset_name") != "insta_market_basket":
        raise ValueError(f"Unexpected dataset checkpoint: {path}")
    if int(run.get("epochs", -1)) != 300:
        raise ValueError(f"Unexpected epoch budget: {path}")

    rmtpp_cfg = dataclass_from_dict(RMTPPConfig, payload["rmtpp_config"])
    model_name = str(run["model_name"]).strip().lower()
    if model_name == "rmtpp":
        model = RMTPP(rmtpp_cfg)
    elif model_name == "thp":
        encoder_cfg = dataclass_from_dict(THPConfig, payload["encoder_config"])
        model = TransformerHawkesTPP(rmtpp_cfg, encoder_cfg)
    elif model_name == "titantpp":
        encoder_cfg = dataclass_from_dict(TitanConfig, payload["encoder_config"])
        model = TitanTPP(rmtpp_cfg, encoder_cfg)
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}: {path}")

    state = payload["model_state_dict"]
    actual_digest = canonical_state_dict_sha256(state)
    if actual_digest != payload.get("model_state_sha256"):
        raise ValueError(f"Checkpoint state digest mismatch: {path}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    identity = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "state_sha256": actual_digest,
        "source_revision": summary.get("source_revision"),
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
        "seed": int(run["seed"]),
        "training_config": payload["training_config"],
        "best_val_nll": float(summary["best_val_nll"]),
        "expected_qty_mae": float(summary["best_val_nll_qty_mae"]),
        "held_out_test_evaluated": False,
    }
    return model, identity


def discover_checkpoints(roots: Iterable[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for root in roots:
        checkpoints.extend(
            path
            for path in root.rglob("best_val_nll_model.pt")
            if "insta_market_basket" in path.parts
            and "epochs_300" in path.parts
            and any(f"seed_{seed}" in path.parts for seed in SEEDS)
        )
    unique = sorted(set(checkpoints))
    if len(unique) != 9:
        raise ValueError(f"Expected 9 Instacart checkpoints, found {len(unique)}")
    return unique


def train_quantile_contract(marked_df: pl.DataFrame) -> dict[str, Any]:
    train = marked_df.filter(pl.col("chronological_split") == "train")
    boundaries = [
        float(train["demand_qty"].quantile(q, interpolation="nearest"))
        for q in QUANTILES
    ]
    if boundaries != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError(f"Quantile boundaries must be strictly increasing: {boundaries}")

    split_rows = {
        str(row["chronological_split"]): int(row["len"])
        for row in (
            marked_df.group_by("chronological_split")
            .agg(pl.len().alias("len"))
            .iter_rows(named=True)
        )
    }
    labels = [
        f"<= {boundaries[0]:g}",
        f"({boundaries[0]:g}, {boundaries[1]:g}]",
        f"({boundaries[1]:g}, {boundaries[2]:g}]",
        f"({boundaries[2]:g}, {boundaries[3]:g}]",
        f"> {boundaries[3]:g}",
    ]
    keys = ["le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99"]
    return {
        "quantiles": list(QUANTILES),
        "boundaries": boundaries,
        "strata": [
            {"stratum_order": index, "stratum": key, "label": label}
            for index, (key, label) in enumerate(zip(keys, labels))
        ],
        "quantile_source_split": "train",
        "evaluation_split": "validation",
        "split_rows": split_rows,
    }


def make_validation_loader(
    marked_df: pl.DataFrame,
    training_values: dict[str, Any],
    batch_size: int,
) -> DataLoader:
    training_cfg = dataclass_from_dict(TrainingConfig, training_values)
    dataset = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_cfg.lookback,
        max_seq_len=training_cfg.max_seq_len,
        val_ratio=training_cfg.val_ratio,
        mode="all",
        split_col="chronological_split",
        target_splits={"validation"},
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_week_lookback,
        num_workers=0,
    )


def empty_accumulator() -> dict[str, float]:
    return {
        "count": 0,
        "true_sum": 0.0,
        "pred_sum": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "signed_sum": 0.0,
    }


def update_accumulator(
    accumulator: dict[str, float],
    true_qty: np.ndarray,
    pred_qty: np.ndarray,
) -> None:
    error = pred_qty - true_qty
    accumulator["count"] += int(true_qty.size)
    accumulator["true_sum"] += float(true_qty.sum())
    accumulator["pred_sum"] += float(pred_qty.sum())
    accumulator["abs_sum"] += float(np.abs(error).sum())
    accumulator["sq_sum"] += float(np.square(error).sum())
    accumulator["signed_sum"] += float(error.sum())


def finalize_accumulator(accumulator: dict[str, float]) -> dict[str, float]:
    count = int(accumulator["count"])
    if count < 1:
        raise ValueError("Empty quantile stratum")
    return {
        "count": count,
        "true_qty_mean": accumulator["true_sum"] / count,
        "pred_qty_mean": accumulator["pred_sum"] / count,
        "qty_mae": accumulator["abs_sum"] / count,
        "qty_rmse": float(np.sqrt(accumulator["sq_sum"] / count)),
        "qty_wape": accumulator["abs_sum"] / max(accumulator["true_sum"], 1e-12),
        "qty_bias": accumulator["signed_sum"] / count,
    }


@torch.no_grad()
def evaluate_checkpoint(
    *,
    model: torch.nn.Module,
    identity: dict[str, Any],
    loader: DataLoader,
    contract: dict[str, Any],
    device: str,
    max_batches: int | None,
) -> list[dict[str, Any]]:
    boundaries = np.asarray(contract["boundaries"], dtype=np.float64)
    strata = [empty_accumulator() for _ in contract["strata"]]
    overall = empty_accumulator()
    started = time.time()

    for batch_index, (marks, dts, mask, _, values) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if values is None:
            raise ValueError("Quantity residual is missing from validation loader")
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)
        values = values.to(device)

        valid = mask[:, -1] & mask[:, -2]
        h = forward_model(model, marks, dts, mask, values)
        h_prev = h[:, -2, :]
        true_mark = marks[:, -1]
        true_residual = values[:, -1]
        pad_id = int(model.cfg.num_marks - 1)
        valid = valid & (true_mark != pad_id)
        if not bool(valid.any().item()):
            continue

        h_prev = h_prev[valid]
        true_mark = true_mark[valid]
        true_residual = true_residual[valid]
        logits = model.mark_head(h_prev)[..., :pad_id]
        pred_mark = torch.argmax(logits, dim=-1)
        pred_residual = predict_value_for_marks(model, h_prev, pred_mark)
        pred_qty = model.reconstruct_qty(pred_mark, pred_residual)
        true_qty = model.reconstruct_qty(true_mark, true_residual)

        pred_np = pred_qty.detach().float().cpu().numpy().astype(np.float64)
        true_np = true_qty.detach().float().cpu().numpy().astype(np.float64)
        update_accumulator(overall, true_np, pred_np)
        stratum_ids = np.searchsorted(boundaries, true_np, side="left")
        for stratum_index in range(len(strata)):
            selected = stratum_ids == stratum_index
            if selected.any():
                update_accumulator(strata[stratum_index], true_np[selected], pred_np[selected])

    total = int(overall["count"])
    if max_batches is None and total != len(loader.dataset):
        raise ValueError(f"Validation count mismatch: evaluated={total} loader={len(loader.dataset)}")

    rows: list[dict[str, Any]] = []
    overall_metrics = finalize_accumulator(overall)
    rows.append({
        **{key: identity[key] for key in ("model_name", "model_label", "seed")},
        "stratum_order": -1,
        "stratum": "all",
        "stratum_label": "All validation",
        "share": 1.0,
        **overall_metrics,
        "elapsed_seconds": time.time() - started,
    })
    for spec, accumulator in zip(contract["strata"], strata):
        metrics = finalize_accumulator(accumulator)
        rows.append({
            **{key: identity[key] for key in ("model_name", "model_label", "seed")},
            "stratum_order": int(spec["stratum_order"]),
            "stratum": spec["stratum"],
            "stratum_label": spec["label"],
            "share": int(metrics["count"]) / max(total, 1),
            **metrics,
            "elapsed_seconds": time.time() - started,
        })

    if max_batches is None:
        observed = float(overall_metrics["qty_mae"])
        expected = float(identity["expected_qty_mae"])
        if not np.isclose(observed, expected, rtol=5e-5, atol=5e-5):
            raise ValueError(
                f"Leaderboard MAE mismatch for {identity['model_label']} seed {identity['seed']}: "
                f"observed={observed} expected={expected}"
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    strata = sorted(
        {(int(row["stratum_order"]), row["stratum"], row["stratum_label"]) for row in seed_rows}
    )
    for model in MODEL_ORDER:
        for order, key, label in strata:
            group = [
                row
                for row in seed_rows
                if row["model_label"] == model and row["stratum"] == key
            ]
            if {int(row["seed"]) for row in group} != set(SEEDS):
                raise ValueError(f"Seed contract failed for {model}/{key}")
            record: dict[str, Any] = {
                "model_label": model,
                "stratum_order": order,
                "stratum": key,
                "stratum_label": label,
                "count": int(group[0]["count"]),
                "share": float(group[0]["share"]),
                "n_seeds": len(group),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_std"] = statistics.stdev(values)
            output.append(record)
    return output


def paired_deltas(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["model_label"], int(row["seed"]), row["stratum"]): row
        for row in seed_rows
    }
    output: list[dict[str, Any]] = []
    strata = sorted(
        {(int(row["stratum_order"]), row["stratum"], row["stratum_label"]) for row in seed_rows}
    )
    for baseline in ("Adapted RMTPP", "Adapted THP"):
        for order, key, label in strata:
            for metric in ("qty_mae", "qty_rmse", "qty_wape", "qty_bias"):
                deltas = [
                    float(indexed[("TitanTPP", seed, key)][metric])
                    - float(indexed[(baseline, seed, key)][metric])
                    for seed in SEEDS
                ]
                baseline_values = [
                    float(indexed[(baseline, seed, key)][metric]) for seed in SEEDS
                ]
                output.append({
                    "baseline": baseline,
                    "stratum_order": order,
                    "stratum": key,
                    "stratum_label": label,
                    "metric": metric,
                    "delta_mean": statistics.mean(deltas),
                    "delta_std": statistics.stdev(deltas),
                    "relative_delta_pct": (
                        100.0 * statistics.mean(deltas) / statistics.mean(baseline_values)
                        if statistics.mean(baseline_values) != 0.0
                        else float("nan")
                    ),
                    "titan_better_seeds": sum(delta < 0.0 for delta in deltas),
                    "seed_42_delta": deltas[0],
                    "seed_52_delta": deltas[1],
                    "seed_62_delta": deltas[2],
                })
    return output


def mean_std(record: dict[str, Any], metric: str, digits: int = 4) -> str:
    return (
        f"{float(record[f'{metric}_mean']):.{digits}f} +/- "
        f"{float(record[f'{metric}_std']):.{digits}f}"
    )


def write_briefing(
    path: Path,
    contract: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Instacart validation quantity error by train-derived quantiles",
        "",
        "- Quantile source: fixed-split train quantities only",
        "- Evaluation target: validation events only",
        "- Checkpoint: best validation event NLL",
        "- Held-out test evaluated: false",
        "- Seeds: 42, 52, 62",
        "- Boundaries: "
        + ", ".join(
            f"p{int(q * 100)}={value:g}"
            for q, value in zip(contract["quantiles"], contract["boundaries"])
        ),
    ]
    visible_strata = [spec["stratum"] for spec in contract["strata"]]
    for key in visible_strata:
        label = next(row["stratum_label"] for row in summary_rows if row["stratum"] == key)
        count = next(row["count"] for row in summary_rows if row["stratum"] == key)
        share = next(row["share"] for row in summary_rows if row["stratum"] == key)
        lines.extend([
            "",
            f"## {label} (n={count:,}, {100.0 * float(share):.2f}%)",
            "",
            "| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |",
            "|---|---:|---:|---:|---:|",
        ])
        for model in MODEL_ORDER:
            row = next(
                item for item in summary_rows
                if item["model_label"] == model and item["stratum"] == key
            )
            lines.append(
                f"| {model} | {mean_std(row, 'qty_mae')} | "
                f"{mean_std(row, 'qty_rmse')} | {mean_std(row, 'qty_wape')} | "
                f"{mean_std(row, 'qty_bias')} |"
            )

    lines.extend([
        "",
        "## TitanTPP paired MAE deltas",
        "",
        "Negative delta means lower MAE for TitanTPP.",
        "",
        "| Stratum | Baseline | Delta mean | Relative delta | Better seeds |",
        "|---|---|---:|---:|---:|",
    ])
    for key in visible_strata:
        for baseline in ("Adapted RMTPP", "Adapted THP"):
            row = next(
                item for item in delta_rows
                if item["stratum"] == key
                and item["baseline"] == baseline
                and item["metric"] == "qty_mae"
            )
            lines.append(
                f"| {row['stratum_label']} | {baseline} | "
                f"{float(row['delta_mean']):+.4f} | "
                f"{float(row['relative_delta_pct']):+.2f}% | "
                f"{int(row['titan_better_seeds'])}/3 |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    output_dir: Path,
    contract: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    strata = contract["strata"]
    x = np.arange(len(strata), dtype=np.float64)
    fig, (ax_mae, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )

    for model_index, model in enumerate(MODEL_ORDER):
        rows = [
            next(
                row for row in summary_rows
                if row["model_label"] == model and row["stratum"] == spec["stratum"]
            )
            for spec in strata
        ]
        means = np.asarray([float(row["qty_mae_mean"]) for row in rows])
        stds = np.asarray([float(row["qty_mae_std"]) for row in rows])
        ax_mae.errorbar(
            x,
            means,
            yerr=stds,
            color=MODEL_COLORS[model],
            marker=("o", "s", "D")[model_index],
            linewidth=2.0,
            markersize=5.8,
            capsize=3,
            label=model,
        )

    bar_width = 0.34
    for offset, baseline in ((-bar_width / 2, "Adapted RMTPP"), (bar_width / 2, "Adapted THP")):
        values = []
        for spec in strata:
            row = next(
                item for item in delta_rows
                if item["baseline"] == baseline
                and item["stratum"] == spec["stratum"]
                and item["metric"] == "qty_mae"
            )
            values.append(float(row["relative_delta_pct"]))
        ax_delta.bar(
            x + offset,
            values,
            width=bar_width,
            color=MODEL_COLORS[baseline],
            alpha=0.88,
            label=f"vs. {baseline.replace('Adapted ', '')}",
        )

    shares = [
        next(row["share"] for row in summary_rows if row["stratum"] == spec["stratum"])
        for spec in strata
    ]
    tick_labels = [
        f"{spec['label']}\n({100.0 * float(share):.1f}%)"
        for spec, share in zip(strata, shares)
    ]
    ax_mae.set_xticks(x, tick_labels)
    ax_delta.set_xticks(x, tick_labels)
    ax_mae.set_ylabel("Validation quantity MAE")
    ax_mae.set_title("(a) Error by true-quantity stratum")
    ax_delta.set_ylabel("TitanTPP MAE change (%)")
    ax_delta.set_title("(b) Relative to adapted baselines")
    ax_delta.axhline(0.0, color="#333333", linewidth=1.0)
    ax_delta.text(
        0.01,
        0.98,
        "lower is better",
        transform=ax_delta.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    for axis in (ax_mae, ax_delta):
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelsize=8.5)
    ax_mae.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax_delta.legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle(
        "Instacart validation quantity error across train-derived quantiles",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Error bars show mean +/- sample standard deviation over seeds 42, 52, and 62. "
        "Percentages in labels are validation-event shares.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"F2_instacart_validation_quantile_mae.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    marked_df = pl.read_parquet(args.data).sort(["oper_part_no", "seq"])
    contract = train_quantile_contract(marked_df)
    checkpoints = discover_checkpoints(args.checkpoint_root)

    payload = torch_load_checkpoint(checkpoints[0], map_location="cpu")
    loader = make_validation_loader(
        marked_df,
        payload["training_config"],
        batch_size=args.batch_size,
    )
    if args.max_batches is None and len(loader.dataset) != args.expected_validation_count:
        raise ValueError(
            f"Expected {args.expected_validation_count} validation targets, "
            f"found {len(loader.dataset)}"
        )

    all_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        model, identity = load_checkpoint(checkpoint, args.device)
        training = identity["training_config"]
        for key in ("lookback", "max_seq_len"):
            if int(training[key]) != int(payload["training_config"][key]):
                raise ValueError(f"Loader contract mismatch for {checkpoint}: {key}")
        print(
            f"[evaluate] model={identity['model_label']} seed={identity['seed']} "
            f"checkpoint={checkpoint}",
            flush=True,
        )
        rows = evaluate_checkpoint(
            model=model,
            identity=identity,
            loader=loader,
            contract=contract,
            device=args.device,
            max_batches=args.max_batches,
        )
        all_rows.extend(rows)
        identities.append(identity)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if {identity["model_label"] for identity in identities} != set(MODEL_ORDER):
        raise ValueError("Model contract failed")
    for model in MODEL_ORDER:
        if {
            identity["seed"] for identity in identities if identity["model_label"] == model
        } != set(SEEDS):
            raise ValueError(f"Seed contract failed for {model}")
    revisions = {identity["source_revision"] for identity in identities}
    if len(revisions) != 1:
        raise ValueError(f"Source revision mismatch: {revisions}")

    summary_rows = summarize(all_rows)
    delta_rows = paired_deltas(all_rows)
    prefix = "smoke_" if args.max_batches is not None else ""
    write_csv(args.output_dir / f"{prefix}instacart_quantile_seed_metrics.csv", all_rows)
    write_csv(args.output_dir / f"{prefix}instacart_quantile_summary.csv", summary_rows)
    write_csv(args.output_dir / f"{prefix}instacart_quantile_paired_deltas.csv", delta_rows)

    contract.update({
        "schema_version": 1,
        "data_path": str(args.data),
        "data_sha256": sha256_file(args.data),
        "dataset_rows": marked_df.height,
        "validation_target_count": len(loader.dataset),
        "checkpoint_selection": "best_val_nll",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": next(iter(revisions)),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "checkpoints": identities,
    })
    (args.output_dir / f"{prefix}quantile_analysis_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_briefing(
        args.output_dir / f"{prefix}instacart_quantile_briefing.md",
        contract,
        summary_rows,
        delta_rows,
    )
    if args.max_batches is None:
        plot_results(args.output_dir, contract, summary_rows, delta_rows)
    print(f"[complete] output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
