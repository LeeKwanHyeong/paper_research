#!/usr/bin/env python3
"""Run the fair Intermittent log-regression backbone control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RMTPPs.config import RMTPPConfig, THPConfig
from models.RMTPPs.value_conditioning import mask_appended_target_value
from models.Titan import TitanConfig
from paper.scripts.run_taxi_log_backbone_control import (
    LogRegressionTHP,
    LogRegressionTitanTPP,
)
from paper.scripts.run_taxi_quantity_interface_ablation import (
    PositiveRegressionRMTPP,
    clone_state_dict,
    early_stopping_exhausted,
    make_loader,
    parse_int_tuple,
    parse_str_tuple,
    save_json,
    set_seed,
    sha256_file,
    train_quantile_contract,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


BACKBONES = ("rmtpp", "thp", "titantpp")
BACKBONE_LABELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "titantpp": "TitanTPP",
}
SEEDS = (42, 52, 62)
EXPECTED_DATA_SHA256 = "85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f"
EXPECTED_SPLIT_MANIFEST_SHA256 = "393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04"
HISTORY_BOUNDARIES = (64, 128)
HISTORY_STRATA = (
    {"stratum_order": 0, "stratum": "history_le_64", "stratum_label": "History <= 64"},
    {"stratum_order": 1, "stratum": "history_65_128", "stratum_label": "History 65-128"},
    {"stratum_order": 2, "stratum": "history_gt_128", "stratum_label": "History > 128"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execution-role", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback-weeks", type=int, default=520)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lambda-log", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def shared_config(*, num_marks: int, hidden_dim: int) -> RMTPPConfig:
    return RMTPPConfig(
        num_marks=num_marks,
        mark_emb_dim=32,
        rnn_hidden_dim=hidden_dim,
        rnn_type="gru",
        dropout=0.1,
        scale_base=2.0,
        use_value_head=False,
        value_head_activation="identity",
        value_input_mode="residual",
        value_input_emb_dim=8,
        train_loss_scope="target_only",
        loss_mode="residual_only",
        lambda_qty=0.25,
    )


def build_model(
    *,
    backbone: str,
    cfg: RMTPPConfig,
    interface_meta: dict[str, Any],
    max_seq_len: int,
    lookback_weeks: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if backbone == "rmtpp":
        model = PositiveRegressionRMTPP(
            cfg,
            mode="log_regression",
            train_min=float(interface_meta["train_min"]),
            train_max=float(interface_meta["train_max"]),
            train_target_mean=float(interface_meta["train_target_mean"]),
        )
        return model, {
            "candidate_name": "gru_h64",
            "rnn_type": "gru",
            "hidden_dim": cfg.rnn_hidden_dim,
        }
    if backbone == "thp":
        encoder_cfg = THPConfig(
            d_model=64,
            d_inner=256,
            n_layers=2,
            n_heads=4,
            dropout=0.1,
            normalize_before=False,
            add_temporal_encoding_each_layer=True,
            use_rnn=False,
            d_rnn=64,
        )
        model = LogRegressionTHP(
            cfg,
            encoder_cfg,
            train_log_mean=float(interface_meta["train_target_mean"]),
        )
        return model, {"candidate_name": "small", **asdict(encoder_cfg)}
    if backbone == "titantpp":
        encoder_cfg = TitanConfig(
            lookback=lookback_weeks,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            memory_mode="static_lmm",
            contextual_mem_size=0,
            persistent_mem_size=16,
            use_context_update=False,
            use_lmm=True,
            mem_size=64,
            mem_topk=4,
            use_pos_emb=True,
            max_len=max_seq_len,
            use_causal=True,
        )
        model = LogRegressionTitanTPP(
            cfg,
            encoder_cfg,
            train_log_mean=float(interface_meta["train_target_mean"]),
        )
        return model, {"candidate_name": "small_lmm", **asdict(encoder_cfg)}
    raise ValueError(f"Unsupported backbone: {backbone}")


def target_outputs(
    model: nn.Module,
    marks: torch.Tensor,
    dts: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    input_values = mask_appended_target_value(values, mask)
    encoded = model.forward(marks, dts, values=input_values, mask=mask)
    hidden = encoded[:, -2, :]
    true_mark = marks[:, -1]
    true_dt = dts[:, -1].float()
    true_value = values[:, -1]
    valid = mask[:, -1] & mask[:, -2] & (true_mark != int(model.cfg.num_marks - 1))
    if not bool(valid.all()):
        hidden = hidden[valid]
        true_mark = true_mark[valid]
        true_dt = true_dt[valid]
        true_value = true_value[valid]
        history_length = mask[valid].sum(dim=1) - 1
    else:
        history_length = mask.sum(dim=1) - 1

    logits = model.mark_head(hidden)
    marker_loss = F.cross_entropy(logits, true_mark, reduction="none")
    time_loss = -model.log_f_dt(hidden.unsqueeze(1), true_dt.unsqueeze(1)).squeeze(1)
    transformed, pred_qty = model.predict_quantity(hidden)
    true_qty = model.reconstruct_qty(true_mark, true_value)
    log_qty_loss = F.mse_loss(transformed, torch.log1p(true_qty), reduction="none")
    pred_mark = torch.argmax(logits[..., : int(model.cfg.num_marks - 1)], dim=-1)
    return {
        "marker_loss": marker_loss,
        "time_loss": time_loss,
        "log_qty_loss": log_qty_loss,
        "true_qty": true_qty,
        "pred_qty": pred_qty,
        "correct": pred_mark == true_mark,
        "history_length": history_length,
    }


def empty_metric_accumulator() -> dict[str, float]:
    return {
        "count": 0,
        "nll_sum": 0.0,
        "marker_sum": 0.0,
        "time_sum": 0.0,
        "correct": 0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "signed_sum": 0.0,
    }


def update_metrics(
    accumulator: dict[str, float],
    *,
    marker: np.ndarray,
    time_nll: np.ndarray,
    correct: np.ndarray,
    true_qty: np.ndarray,
    pred_qty: np.ndarray,
) -> None:
    error = pred_qty - true_qty
    accumulator["count"] += int(true_qty.size)
    accumulator["marker_sum"] += float(marker.sum())
    accumulator["time_sum"] += float(time_nll.sum())
    accumulator["nll_sum"] += float((marker + time_nll).sum())
    accumulator["correct"] += int(correct.sum())
    accumulator["abs_sum"] += float(np.abs(error).sum())
    accumulator["sq_sum"] += float(np.square(error).sum())
    accumulator["signed_sum"] += float(error.sum())


def finalize_metrics(accumulator: dict[str, float]) -> dict[str, Any]:
    count = int(accumulator["count"])
    if count < 1:
        raise ValueError("Cannot finalize an empty metric accumulator")
    return {
        "count": count,
        "nll": accumulator["nll_sum"] / count,
        "nll_marker": accumulator["marker_sum"] / count,
        "nll_time": accumulator["time_sum"] / count,
        "mark_acc": accumulator["correct"] / count,
        "qty_mae": accumulator["abs_sum"] / count,
        "qty_rmse": float(np.sqrt(accumulator["sq_sum"] / count)),
        "qty_bias": accumulator["signed_sum"] / count,
    }


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader: Any,
    quantity_contract: dict[str, Any],
    device: str,
    max_batches: int | None,
    include_breakdowns: bool,
) -> dict[str, Any]:
    model.eval()
    overall = empty_metric_accumulator()
    quantity_accumulators = [
        empty_metric_accumulator() for _ in quantity_contract["strata"]
    ]
    history_accumulators = [empty_metric_accumulator() for _ in HISTORY_STRATA]

    for batch_index, (marks, dts, mask, _, values) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if values is None:
            raise ValueError("Log-regression evaluation requires quantity values")
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)
        values = values.to(device)
        outputs = target_outputs(model, marks, dts, mask, values)
        marker = outputs["marker_loss"].detach().cpu().numpy().astype(np.float64)
        time_nll = outputs["time_loss"].detach().cpu().numpy().astype(np.float64)
        correct = outputs["correct"].detach().cpu().numpy().astype(bool)
        true_qty = torch.round(outputs["true_qty"]).detach().cpu().numpy().astype(np.float64)
        pred_qty = outputs["pred_qty"].detach().cpu().numpy().astype(np.float64)
        history_length = outputs["history_length"].detach().cpu().numpy().astype(np.int64)
        update_metrics(
            overall,
            marker=marker,
            time_nll=time_nll,
            correct=correct,
            true_qty=true_qty,
            pred_qty=pred_qty,
        )
        if not include_breakdowns:
            continue

        quantity_ids = np.searchsorted(
            np.asarray(quantity_contract["boundaries"], dtype=np.float64),
            true_qty,
            side="left",
        )
        history_ids = np.searchsorted(
            np.asarray(HISTORY_BOUNDARIES, dtype=np.int64),
            history_length,
            side="left",
        )
        for ids, accumulators in (
            (quantity_ids, quantity_accumulators),
            (history_ids, history_accumulators),
        ):
            for index, accumulator in enumerate(accumulators):
                selected = ids == index
                if selected.any():
                    update_metrics(
                        accumulator,
                        marker=marker[selected],
                        time_nll=time_nll[selected],
                        correct=correct[selected],
                        true_qty=true_qty[selected],
                        pred_qty=pred_qty[selected],
                    )

    overall_metrics = finalize_metrics(overall)
    result: dict[str, Any] = {
        "val_nll": overall_metrics["nll"],
        "val_nll_marker": overall_metrics["nll_marker"],
        "val_nll_time": overall_metrics["nll_time"],
        "mark_acc": overall_metrics["mark_acc"],
        "qty_mae": overall_metrics["qty_mae"],
        "qty_rmse": overall_metrics["qty_rmse"],
        "preclamp_negative_share": 0.0,
        "evaluated_count": overall_metrics["count"],
    }
    if not include_breakdowns:
        return result

    quantity_rows = []
    history_rows = []
    for spec, accumulator in zip(quantity_contract["strata"], quantity_accumulators):
        metrics = finalize_metrics(accumulator)
        quantity_rows.append({
            **spec,
            "share": metrics["count"] / overall_metrics["count"],
            **metrics,
        })
    for spec, accumulator in zip(HISTORY_STRATA, history_accumulators):
        metrics = finalize_metrics(accumulator)
        history_rows.append({
            **spec,
            "share": metrics["count"] / overall_metrics["count"],
            **metrics,
        })
    result["quantity_rows"] = quantity_rows
    result["history_rows"] = history_rows
    return result


def train_one(
    *,
    args: argparse.Namespace,
    df: pl.DataFrame,
    quantity_contract: dict[str, Any],
    interface_meta: dict[str, Any],
    backbone: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = args.output_dir / "runs" / backbone / "log_regression" / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_val_nll_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    if summary_path.exists() and best_path.exists() and not args.force_rerun:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        quantity_rows = payload.pop("quantity_rows")
        history_rows = payload.pop("history_rows")
        return payload, quantity_rows, history_rows

    generator = set_seed(seed)
    train_loader = make_loader(
        df,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=generator,
    )
    val_loader = make_loader(
        df,
        target_split="validation",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=False,
        generator=None,
    )
    num_marks = int(df["mark"].max()) + 2
    cfg = shared_config(num_marks=num_marks, hidden_dim=args.hidden_dim)
    model, encoder_config = build_model(
        backbone=backbone,
        cfg=cfg,
        interface_meta=interface_meta,
        max_seq_len=args.max_seq_len,
        lookback_weeks=args.lookback_weeks,
    )
    model.to(args.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, Any]] = []
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    source_revision_history = [args.source_revision]
    start_epoch = 1

    if last_path.exists() and not args.force_rerun:
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload.get("history", []))
        best_nll = float(payload.get("best_val_nll", best_nll))
        best_state = payload.get("best_state_dict")
        start_epoch = int(payload["epoch"]) + 1
        source_revision_history = [
            revision for revision in payload.get("source_revision_history", []) if revision
        ]
        if args.source_revision not in source_revision_history:
            source_revision_history.append(args.source_revision)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    started = time.time()
    stopped_early = early_stopping_exhausted(
        history,
        min_epochs=args.min_epochs,
        patience=args.early_stopping_patience,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        if stopped_early:
            break
        model.train()
        running = 0.0
        batches = 0
        for batch_index, (marks, dts, mask, _, values) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            if values is None:
                raise ValueError("Log-regression training requires quantity values")
            marks = marks.to(args.device)
            dts = dts.to(args.device)
            mask = mask.to(args.device)
            values = values.to(args.device)
            outputs = target_outputs(model, marks, dts, mask, values)
            loss = (
                outputs["marker_loss"].mean()
                + outputs["time_loss"].mean()
                + args.lambda_log * outputs["log_qty_loss"].mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item())
            batches += 1

        validation = evaluate(
            model=model,
            loader=val_loader,
            quantity_contract=quantity_contract,
            device=args.device,
            max_batches=args.max_val_batches,
            include_breakdowns=False,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "val_nll": float(validation["val_nll"]),
            "val_nll_marker": float(validation["val_nll_marker"]),
            "val_nll_time": float(validation["val_nll_time"]),
            "val_qty_mae": float(validation["qty_mae"]),
            "val_qty_rmse": float(validation["qty_rmse"]),
            "mark_acc": float(validation["mark_acc"]),
        }
        history.append(epoch_row)
        line = (
            f"[epoch {epoch:03d}] backbone={backbone} seed={seed} "
            f"train={epoch_row['train_loss']:.8f} nll={epoch_row['val_nll']:.8f} "
            f"time_nll={epoch_row['val_nll_time']:.8f} "
            f"qty_mae={epoch_row['val_qty_mae']:.8f} mark_acc={epoch_row['mark_acc']:.8f}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if float(validation["val_nll"]) < best_nll:
            best_nll = float(validation["val_nll"])
            best_state = clone_state_dict(model)
        save_json(run_dir / "history.json", {"history": history})
        torch.save({
            "epoch": epoch,
            "backbone": backbone,
            "seed": seed,
            "model_state_dict": clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_nll": best_nll,
            "best_state_dict": best_state,
            "rmtpp_config": asdict(cfg),
            "encoder_config": encoder_config,
            "interface_meta": interface_meta,
            "source_revision": args.source_revision,
            "source_revision_history": source_revision_history,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }, last_path)
        stopped_early = early_stopping_exhausted(
            history,
            min_epochs=args.min_epochs,
            patience=args.early_stopping_patience,
        )
        if stopped_early:
            best_epoch = min(history, key=lambda row: float(row["val_nll"]))["epoch"]
            line = (
                f"[early-stop] backbone={backbone} seed={seed} "
                f"current_epoch={epoch} best_epoch={best_epoch}"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for {backbone}/seed_{seed}")
    model.load_state_dict(best_state, strict=True)
    validation = evaluate(
        model=model,
        loader=val_loader,
        quantity_contract=quantity_contract,
        device=args.device,
        max_batches=args.max_val_batches,
        include_breakdowns=True,
    )
    state_digest = canonical_state_dict_sha256(best_state)
    checkpoint = {
        "selection": "best_val_nll",
        "backbone": backbone,
        "seed": seed,
        "model_state_dict": best_state,
        "model_state_sha256": state_digest,
        "rmtpp_config": asdict(cfg),
        "encoder_config": encoder_config,
        "interface_meta": interface_meta,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    torch.save(checkpoint, best_path)
    quantity_rows = [
        {
            "backbone": backbone,
            "backbone_label": BACKBONE_LABELS[backbone],
            "variant": "log_regression",
            "seed": seed,
            **row,
        }
        for row in validation["quantity_rows"]
    ]
    history_rows = [
        {
            "backbone": backbone,
            "backbone_label": BACKBONE_LABELS[backbone],
            "variant": "log_regression",
            "seed": seed,
            **row,
        }
        for row in validation["history_rows"]
    ]
    best_epoch = int(min(history, key=lambda row: float(row["val_nll"]))["epoch"])
    summary = {
        "status": "success",
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": "log_regression",
        "seed": seed,
        "epochs": args.epochs,
        "completed_epochs": int(history[-1]["epoch"]),
        "stopped_early": int(history[-1]["epoch"]) < args.epochs,
        "best_epoch": best_epoch,
        "best_val_nll": float(validation["val_nll"]),
        "best_val_nll_marker": float(validation["val_nll_marker"]),
        "best_val_nll_time": float(validation["val_nll_time"]),
        "best_val_qty_mae": float(validation["qty_mae"]),
        "best_val_qty_rmse": float(validation["qty_rmse"]),
        "mark_acc": float(validation["mark_acc"]),
        "preclamp_negative_share": 0.0,
        "parameter_count": parameter_count,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_path": str(best_path),
        "checkpoint_state_sha256": state_digest,
        "elapsed_seconds": time.time() - started,
        "encoder_config": encoder_config,
        "interface_meta": interface_meta,
        "quantity_rows": quantity_rows,
        "history_rows": history_rows,
    }
    save_json(summary_path, summary)
    returned = dict(summary)
    returned.pop("quantity_rows")
    returned.pop("history_rows")
    return returned, quantity_rows, history_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_breakdowns(
    rows: list[dict[str, Any]],
    *,
    backbones: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    output = []
    strata = sorted({
        (int(row["stratum_order"]), row["stratum"], row["stratum_label"])
        for row in rows
    })
    for backbone in backbones:
        for order, key, label in strata:
            group = [
                row for row in rows
                if row["backbone"] == backbone and row["stratum"] == key
            ]
            if {int(row["seed"]) for row in group} != set(seeds):
                raise ValueError(f"Seed contract failed for {backbone}/{key}")
            record = {
                "backbone": backbone,
                "backbone_label": BACKBONE_LABELS[backbone],
                "variant": "log_regression",
                "stratum_order": order,
                "stratum": key,
                "stratum_label": label,
                "count": int(group[0]["count"]),
                "share": float(group[0]["share"]),
                "n_seeds": len(group),
            }
            for metric in (
                "nll",
                "nll_marker",
                "nll_time",
                "mark_acc",
                "qty_mae",
                "qty_rmse",
                "qty_bias",
            ):
                values = [float(row[metric]) for row in group]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            output.append(record)
    return output


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    backbones = parse_str_tuple(args.backbones)
    seeds = parse_int_tuple(args.seeds)
    if any(backbone not in BACKBONES for backbone in backbones):
        raise ValueError(f"Unsupported backbone selection: {backbones}")
    if not args.allow_partial_contract:
        if set(backbones) != set(BACKBONES) or set(seeds) != set(SEEDS):
            raise ValueError("Qualified run requires all three backbones and seeds 42/52/62")
    if args.hidden_dim != 64:
        raise ValueError("The matched Intermittent contract requires hidden_dim=64")
    if args.max_seq_len != 256:
        raise ValueError("The long-history Intermittent contract requires max_seq_len=256")

    data_sha256 = sha256_file(args.data)
    manifest_sha256 = sha256_file(args.split_manifest)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError(f"Unexpected Intermittent fixed-split SHA-256: {data_sha256}")
    if manifest_sha256 != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise ValueError(f"Unexpected split-manifest SHA-256: {manifest_sha256}")
    df = pl.read_parquet(args.data).sort(["oper_part_no", "seq"])
    required = {
        "oper_part_no",
        "seq",
        "delta_t",
        "demand_qty",
        "mark",
        "scale_residual",
        "chronological_split",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Intermittent fixed split is missing columns: {missing}")
    quantity_contract = train_quantile_contract(df)
    train_qty = df.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy()
    interface_meta = {
        "mode": "log_regression",
        "target": "log1p_demand_qty",
        "loss": "mse_on_log1p_quantity",
        "output_activation": "softplus",
        "inverse_transform": "expm1",
        "train_min": float(train_qty.min()),
        "train_max": float(train_qty.max()),
        "train_target_mean": float(np.log1p(train_qty).mean()),
        "history_quantity_input": "log2_within_mark_residual",
        "support": "nonnegative",
        "fitted_on": "train",
    }
    split_rows = {
        str(row["chronological_split"]): int(row["len"])
        for row in df.group_by("chronological_split").agg(pl.len()).iter_rows(named=True)
    }
    contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "intermittent_log_backbone_control",
        "dataset": "intermittent_frozen_5000",
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha256,
        "split_rows": split_rows,
        "quantity_contract": quantity_contract,
        "history_length_contract": {
            "boundaries": list(HISTORY_BOUNDARIES),
            "strata": list(HISTORY_STRATA),
            "definition": "number of observed events before the validation target",
            "rationale": "pre-registered architecture-oriented context ranges",
        },
        "interface": interface_meta,
        "backbones": list(backbones),
        "seeds": list(seeds),
        "expected_run_count": len(backbones) * len(seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_log": args.lambda_log,
        "grad_clip": args.grad_clip,
        "early_stopping": {
            "monitor": "validation_nll",
            "min_epochs": args.min_epochs,
            "patience": args.early_stopping_patience,
            "restore": "best_val_nll",
        },
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "hidden_dim": args.hidden_dim,
        "backbone_candidates": {
            "rmtpp": "gru_h64",
            "thp": "small",
            "titantpp": "small_lmm",
        },
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "execution_host": os.uname().nodename,
        "execution_role": args.execution_role,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "launch_contract.json", contract)

    summaries: list[dict[str, Any]] = []
    quantity_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for backbone in backbones:
        for seed in seeds:
            summary, run_quantity_rows, run_history_rows = train_one(
                args=args,
                df=df,
                quantity_contract=quantity_contract,
                interface_meta=interface_meta,
                backbone=backbone,
                seed=seed,
            )
            summaries.append(summary)
            quantity_rows.extend(run_quantity_rows)
            history_rows.extend(run_history_rows)
            write_csv(args.output_dir / "run_summaries.csv", summaries)
            write_csv(args.output_dir / "quantity_seed_metrics.csv", quantity_rows)
            write_csv(args.output_dir / "history_seed_metrics.csv", history_rows)

    write_csv(
        args.output_dir / "quantity_summary.csv",
        summarize_breakdowns(quantity_rows, backbones=backbones, seeds=seeds),
    )
    write_csv(
        args.output_dir / "history_summary.csv",
        summarize_breakdowns(history_rows, backbones=backbones, seeds=seeds),
    )
    contract["status"] = "complete"
    contract["completed_run_count"] = len(summaries)
    contract["held_out_test_evaluated"] = False
    save_json(args.output_dir / "launch_contract.json", contract)
    print(f"[complete] output_dir={args.output_dir} runs={len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
