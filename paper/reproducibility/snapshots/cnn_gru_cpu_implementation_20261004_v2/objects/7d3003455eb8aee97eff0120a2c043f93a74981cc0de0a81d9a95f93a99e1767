#!/usr/bin/env python3
"""Run the fair Taxi log-regression head on THP and TitanTPP backbones."""

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

from models.TPPs.TitanTPP import TitanTPP
from models.TPPs.TransformerHawkesTPP import TransformerHawkesTPP
from models.TPPs.config import RMTPPConfig, THPConfig
from models.TPPs.value_conditioning import mask_appended_target_value
from models.Titan import TitanConfig
from paper.scripts.run_taxi_quantity_interface_ablation import (
    QUANTILES,
    SEEDS,
    clone_state_dict,
    early_stopping_exhausted,
    empty_accumulator,
    finalize_accumulator,
    make_loader,
    parse_int_tuple,
    parse_str_tuple,
    save_json,
    set_seed,
    sha256_file,
    train_quantile_contract,
    update_accumulator,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


BACKBONES = ("thp", "titantpp")
BACKBONE_LABELS = {
    "thp": "Adapted THP",
    "titantpp": "TitanTPP",
}
EXPECTED_DATA_SHA256 = "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46"


class LogRegressionHead:
    """Shared log1p quantity head mixed into each history backbone."""

    regression_mode = "log_regression"

    def _init_log_regression_head(self, hidden_dim: int, train_log_mean: float) -> None:
        if not np.isfinite(train_log_mean) or train_log_mean <= 0.0:
            raise ValueError("Train log-quantity mean must be finite and positive.")
        self.register_buffer(
            "train_log_mean",
            torch.tensor(float(train_log_mean), dtype=torch.float32),
        )
        self.direct_qty_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.direct_qty_head.weight)
        initial_bias = float(np.log(np.expm1(train_log_mean)))
        nn.init.constant_(self.direct_qty_head.bias, initial_bias)

    def predict_quantity(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = F.softplus(self.direct_qty_head(hidden).squeeze(-1))
        return transformed, torch.expm1(transformed)


class LogRegressionTHP(LogRegressionHead, TransformerHawkesTPP):
    def __init__(
        self,
        cfg: RMTPPConfig,
        thp_cfg: THPConfig,
        *,
        train_log_mean: float,
    ) -> None:
        super().__init__(cfg, thp_cfg)
        self._init_log_regression_head(thp_cfg.d_model, train_log_mean)


class LogRegressionTitanTPP(LogRegressionHead, TitanTPP):
    def __init__(
        self,
        cfg: RMTPPConfig,
        titan_cfg: TitanConfig,
        *,
        train_log_mean: float,
    ) -> None:
        super().__init__(cfg, titan_cfg)
        self._init_log_regression_head(titan_cfg.d_model, train_log_mean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rmtpp-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback-weeks", type=int, default=168)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lambda-log", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=50)
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
        scale_base=10.0,
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
    train_log_mean: float,
) -> tuple[nn.Module, dict[str, Any]]:
    if backbone == "thp":
        encoder_cfg = THPConfig(
            d_model=128,
            d_inner=512,
            n_layers=3,
            n_heads=4,
            dropout=0.1,
            normalize_before=False,
            add_temporal_encoding_each_layer=True,
            use_rnn=False,
            d_rnn=128,
        )
        model = LogRegressionTHP(
            cfg,
            encoder_cfg,
            train_log_mean=train_log_mean,
        )
        return model, {"candidate_name": "base", **asdict(encoder_cfg)}
    if backbone == "titantpp":
        encoder_cfg = TitanConfig(
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            memory_mode="static_lmm",
            contextual_mem_size=32,
            persistent_mem_size=32,
            use_lmm=True,
            mem_size=128,
            mem_topk=8,
            use_pos_emb=True,
            max_len=512,
            use_causal=True,
        )
        model = LogRegressionTitanTPP(
            cfg,
            encoder_cfg,
            train_log_mean=train_log_mean,
        )
        return model, {"candidate_name": "mid_lmm", **asdict(encoder_cfg)}
    raise ValueError(f"Unsupported backbone: {backbone}")


def log_regression_loss(
    model: LogRegressionHead,
    marks: torch.Tensor,
    dts: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    input_values = mask_appended_target_value(values, mask)
    hidden = model.forward(marks, dts, values=input_values, mask=mask)[:, -2, :]
    true_mark = marks[:, -1]
    true_value = values[:, -1]
    valid = mask[:, -1] & mask[:, -2] & (true_mark != int(model.cfg.num_marks - 1))
    true_qty = model.reconstruct_qty(true_mark[valid], true_value[valid])
    transformed, _ = model.predict_quantity(hidden[valid])
    return F.mse_loss(transformed, torch.log1p(true_qty))


@torch.no_grad()
def evaluate(
    *,
    model: LogRegressionHead,
    loader: Any,
    quantile_contract: dict[str, Any],
    device: str,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    boundaries = np.asarray(quantile_contract["boundaries"], dtype=np.float64)
    strata = [empty_accumulator() for _ in quantile_contract["strata"]]
    overall = empty_accumulator()
    nll_sum = 0.0
    marker_sum = 0.0
    time_sum = 0.0
    steps_sum = 0.0
    correct = 0
    total = 0

    for batch_index, (marks, dts, mask, _, values) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if values is None:
            raise ValueError("Log-regression evaluation requires quantity values.")
        marks = marks.to(device)
        dts = dts.to(device)
        mask = mask.to(device)
        values = values.to(device)
        out = model.nll(marks, dts, values=values, mask=mask, loss_scope="target_only")
        steps = float(out["steps"].item())
        nll_sum += float(out["nll"].item()) * steps
        marker_sum += float(out["nll_marker"].item()) * steps
        time_sum += float(out["nll_time"].item()) * steps
        steps_sum += steps

        input_values = mask_appended_target_value(values, mask)
        hidden = model.forward(marks, dts, values=input_values, mask=mask)[:, -2, :]
        true_mark = marks[:, -1]
        true_value = values[:, -1]
        valid = mask[:, -1] & mask[:, -2] & (true_mark != int(model.cfg.num_marks - 1))
        hidden = hidden[valid]
        true_mark = true_mark[valid]
        true_value = true_value[valid]
        logits = model.mark_head(hidden)[..., : int(model.cfg.num_marks - 1)]
        pred_mark = torch.argmax(logits, dim=-1)
        correct += int((pred_mark == true_mark).sum().item())
        total += int(true_mark.numel())

        true_qty = torch.round(model.reconstruct_qty(true_mark, true_value))
        _, pred_qty = model.predict_quantity(hidden)
        true_np = true_qty.detach().cpu().numpy().astype(np.float64)
        pred_np = pred_qty.detach().cpu().numpy().astype(np.float64)
        update_accumulator(overall, true_np, pred_np)
        stratum_ids = np.searchsorted(boundaries, true_np, side="left")
        for index, accumulator in enumerate(strata):
            selected = stratum_ids == index
            if selected.any():
                update_accumulator(accumulator, true_np[selected], pred_np[selected])

    rows = [{
        "stratum_order": -1,
        "stratum": "all",
        "stratum_label": "All validation",
        "share": 1.0,
        **finalize_accumulator(overall),
    }]
    total_count = int(overall["count"])
    for spec, accumulator in zip(quantile_contract["strata"], strata):
        if int(accumulator["count"]) < 1:
            if max_batches is not None:
                continue
            raise ValueError(f"Empty validation stratum: {spec['stratum']}")
        metrics = finalize_accumulator(accumulator)
        rows.append({
            **spec,
            "share": int(metrics["count"]) / total_count,
            **metrics,
        })
    return {
        "val_nll": nll_sum / max(steps_sum, 1.0),
        "val_nll_marker": marker_sum / max(steps_sum, 1.0),
        "val_nll_time": time_sum / max(steps_sum, 1.0),
        "mark_acc": correct / max(total, 1),
        "preclamp_negative_share": 0.0,
        "rows": rows,
        "evaluated_count": total_count,
    }


def train_one(
    *,
    args: argparse.Namespace,
    df: pl.DataFrame,
    quantile_contract: dict[str, Any],
    interface_meta: dict[str, Any],
    backbone: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = args.output_dir / "runs" / backbone / "log_regression" / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_val_nll_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    if summary_path.exists() and best_path.exists() and not args.force_rerun:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = payload.pop("stratum_rows")
        return payload, rows

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
        train_log_mean=float(interface_meta["train_target_mean"]),
    )
    model.to(args.device)
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
            revision
            for revision in payload.get("source_revision_history", [])
            if revision
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
                raise ValueError("Log-regression training requires quantity values.")
            marks = marks.to(args.device)
            dts = dts.to(args.device)
            mask = mask.to(args.device)
            values = values.to(args.device)
            out = model.nll(marks, dts, values=values, mask=mask, loss_scope="target_only")
            quantity_loss = log_regression_loss(model, marks, dts, mask, values)
            loss = out["nll_marker"] + out["nll_time"] + args.lambda_log * quantity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item())
            batches += 1

        validation = evaluate(
            model=model,
            loader=val_loader,
            quantile_contract=quantile_contract,
            device=args.device,
            max_batches=args.max_val_batches,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "val_nll": float(validation["val_nll"]),
            "val_nll_marker": float(validation["val_nll_marker"]),
            "val_nll_time": float(validation["val_nll_time"]),
            "val_qty_mae": float(validation["rows"][0]["qty_mae"]),
            "val_qty_rmse": float(validation["rows"][0]["qty_rmse"]),
            "mark_acc": float(validation["mark_acc"]),
        }
        history.append(epoch_row)
        line = (
            f"[epoch {epoch:03d}] backbone={backbone} seed={seed} "
            f"train={epoch_row['train_loss']:.8f} nll={epoch_row['val_nll']:.8f} "
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
            line = (
                f"[early-stop] backbone={backbone} seed={seed} current_epoch={epoch} "
                f"best_epoch={min(history, key=lambda row: float(row['val_nll']))['epoch']}"
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
        quantile_contract=quantile_contract,
        device=args.device,
        max_batches=args.max_val_batches,
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
    stratum_rows = [
        {
            "backbone": backbone,
            "backbone_label": BACKBONE_LABELS[backbone],
            "variant": "log_regression",
            "seed": seed,
            **row,
        }
        for row in validation["rows"]
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
        "best_val_qty_mae": float(validation["rows"][0]["qty_mae"]),
        "best_val_qty_rmse": float(validation["rows"][0]["qty_rmse"]),
        "mark_acc": float(validation["mark_acc"]),
        "preclamp_negative_share": 0.0,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_path": str(best_path),
        "checkpoint_state_sha256": state_digest,
        "elapsed_seconds": time.time() - started,
        "encoder_config": encoder_config,
        "interface_meta": interface_meta,
        "stratum_rows": stratum_rows,
    }
    save_json(summary_path, summary)
    returned = dict(summary)
    returned.pop("stratum_rows")
    return returned, stratum_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
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
            for metric in ("qty_mae", "qty_rmse", "qty_bias"):
                values = [float(row[metric]) for row in group]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            output.append(record)
    return output


def validate_rmtpp_reference(path: Path, data_sha256: str) -> dict[str, Any]:
    contract_path = path / "launch_contract.json"
    summary_path = path / "run_summaries.csv"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if contract["data_sha256"] != data_sha256:
        raise ValueError("RMTPP reference uses a different Taxi split.")
    if contract["evaluation_scope"] != "validation_only" or contract["held_out_test_evaluated"]:
        raise ValueError("RMTPP reference is not validation-only.")
    if {int(row["seed"]) for row in rows} != set(SEEDS):
        raise ValueError("RMTPP reference does not contain the required three seeds.")
    return {
        "contract_path": str(contract_path.resolve()),
        "contract_source_revision": contract["source_revision"],
        "run_summary_path": str(summary_path.resolve()),
        "seeds": list(SEEDS),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA.")
    backbones = parse_str_tuple(args.backbones)
    seeds = parse_int_tuple(args.seeds)
    if sorted(backbones) != sorted(BACKBONES) and not args.allow_partial_contract:
        raise ValueError(f"Qualified run requires backbones {BACKBONES}.")
    if set(seeds) != set(SEEDS) and not args.allow_partial_contract:
        raise ValueError(f"Qualified run requires seeds {SEEDS}.")
    if args.hidden_dim != 128:
        raise ValueError("The matched Taxi contract requires hidden_dim=128.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_sha256 = sha256_file(args.data)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError(f"Unexpected Taxi fixed-split SHA-256: {data_sha256}")
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
        raise ValueError(f"Taxi fixed split is missing columns: {missing}")
    quantile_contract = train_quantile_contract(df)
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
        "history_quantity_input": "log10_within_mark_residual",
        "support": "nonnegative",
        "fitted_on": "train",
    }
    rmtpp_reference = validate_rmtpp_reference(args.rmtpp_reference, data_sha256)
    split_rows = {
        str(row["chronological_split"]): int(row["len"])
        for row in df.group_by("chronological_split").agg(pl.len()).iter_rows(named=True)
    }
    contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "taxi_log_backbone_control",
        "dataset": "yellow_trip_hourly",
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "split_rows": split_rows,
        "quantile_contract": quantile_contract,
        "interface": interface_meta,
        "trained_backbones": list(backbones),
        "reference_backbone": "rmtpp",
        "rmtpp_reference": rmtpp_reference,
        "seeds": list(seeds),
        "expected_new_run_count": len(backbones) * len(seeds),
        "expected_combined_run_count": 3 + len(backbones) * len(seeds),
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
        "backbone_candidates": {"thp": "base", "titantpp": "mid_lmm"},
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "execution_host": os.uname().nodename,
    }
    save_json(args.output_dir / "launch_contract.json", contract)

    summaries: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for backbone in backbones:
        for seed in seeds:
            summary, rows = train_one(
                args=args,
                df=df,
                quantile_contract=quantile_contract,
                interface_meta=interface_meta,
                backbone=backbone,
                seed=seed,
            )
            summaries.append(summary)
            seed_rows.extend(rows)
            write_csv(args.output_dir / "run_summaries.csv", summaries)
            write_csv(args.output_dir / "quantity_seed_metrics.csv", seed_rows)

    write_csv(
        args.output_dir / "quantity_summary.csv",
        summarize(seed_rows, backbones=backbones, seeds=seeds),
    )
    contract["status"] = "complete"
    contract["completed_run_count"] = len(summaries)
    contract["held_out_test_evaluated"] = False
    save_json(args.output_dir / "launch_contract.json", contract)
    print(f"[complete] output_dir={args.output_dir} new_runs={len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
