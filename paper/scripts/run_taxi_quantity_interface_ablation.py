#!/usr/bin/env python3
"""Run a validation-only Taxi quantity-interface ablation with RMTPP encoders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.event_seq_data_module import (
    RMTPPWeekLookbackDataset,
    collate_week_lookback,
)
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.RMTPP import RMTPP
from models.RMTPPs.value_conditioning import (
    mask_appended_target_value,
    predict_value_for_marks,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


SEEDS = (42, 52, 62)
QUANTILES = (0.50, 0.90, 0.95, 0.99)
NEW_VARIANTS = (
    "uniform_categorical",
    "quantile_categorical",
    "direct_raw_mse",
    "minmax_sigmoid",
    "log_regression",
)
ALL_VARIANTS = (*NEW_VARIANTS, "mark_residual")
VARIANT_LABELS = {
    "uniform_categorical": "Uniform bins",
    "quantile_categorical": "Quantile bins",
    "direct_raw_mse": "Raw MSE regression",
    "minmax_sigmoid": "Min-max + sigmoid regression",
    "log_regression": "Log-scale regression",
    "mark_residual": "Mark-residual",
}


class DirectRawRMTPP(RMTPP):
    """RMTPP with a raw-scale regression head and unchanged mark/time heads."""

    def __init__(self, cfg: RMTPPConfig, *, raw_mean: float, raw_std: float):
        super().__init__(cfg)
        if not np.isfinite(raw_mean) or not np.isfinite(raw_std) or raw_std <= 0.0:
            raise ValueError("Raw target normalization must be finite with positive std.")
        self.register_buffer("raw_mean", torch.tensor(float(raw_mean), dtype=torch.float32))
        self.register_buffer("raw_std", torch.tensor(float(raw_std), dtype=torch.float32))
        self.direct_qty_head = nn.Linear(cfg.rnn_hidden_dim, 1)
        nn.init.zeros_(self.direct_qty_head.weight)
        nn.init.zeros_(self.direct_qty_head.bias)

    def predict_raw_quantity(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.direct_qty_head(hidden).squeeze(-1)
        affine = self.raw_mean.to(hidden) + self.raw_std.to(hidden) * normalized
        return affine, affine.clamp_min(0.0)


class PositiveRegressionRMTPP(RMTPP):
    """RMTPP with a quantity head whose reconstruction is nonnegative by design."""

    def __init__(
        self,
        cfg: RMTPPConfig,
        *,
        mode: str,
        train_min: float,
        train_max: float,
        train_target_mean: float,
    ):
        super().__init__(cfg)
        if mode not in {"minmax_sigmoid", "log_regression"}:
            raise ValueError(f"Unsupported positive regression mode: {mode}")
        if not np.isfinite(train_min) or not np.isfinite(train_max) or train_max <= train_min:
            raise ValueError("Train quantity range must be finite and non-degenerate.")
        self.regression_mode = mode
        self.register_buffer("train_min", torch.tensor(float(train_min), dtype=torch.float32))
        self.register_buffer("train_max", torch.tensor(float(train_max), dtype=torch.float32))
        self.direct_qty_head = nn.Linear(cfg.rnn_hidden_dim, 1)
        nn.init.zeros_(self.direct_qty_head.weight)
        if mode == "minmax_sigmoid":
            scaled_mean = (train_target_mean - train_min) / (train_max - train_min)
            scaled_mean = float(np.clip(scaled_mean, 1e-4, 1.0 - 1e-4))
            initial_bias = float(np.log(scaled_mean / (1.0 - scaled_mean)))
        else:
            positive_mean = max(float(train_target_mean), 1e-4)
            initial_bias = float(np.log(np.expm1(positive_mean)))
        nn.init.constant_(self.direct_qty_head.bias, initial_bias)

    def predict_quantity(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.direct_qty_head(hidden).squeeze(-1)
        if self.regression_mode == "minmax_sigmoid":
            transformed = torch.sigmoid(raw)
            quantity = self.train_min.to(hidden) + (
                self.train_max.to(hidden) - self.train_min.to(hidden)
            ) * transformed
        else:
            transformed = F.softplus(raw)
            quantity = torch.expm1(transformed)
        return transformed, quantity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback-weeks", type=int, default=168)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--bin-count", type=int, default=4)
    parser.add_argument("--lambda-raw", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--variants", default=",".join(NEW_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--skip-proposal", action="store_true")
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
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


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(",") if token.strip())


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def early_stopping_exhausted(
    history: list[dict[str, Any]],
    *,
    min_epochs: int,
    patience: int,
) -> bool:
    if not history or patience < 1:
        return False
    current_epoch = int(history[-1]["epoch"])
    best_epoch = int(min(history, key=lambda row: float(row["val_nll"]))["epoch"])
    return current_epoch >= min_epochs and current_epoch - best_epoch >= patience


def train_quantile_contract(df: pl.DataFrame) -> dict[str, Any]:
    train = df.filter(pl.col("chronological_split") == "train")
    boundaries = [
        float(train["demand_qty"].quantile(q, interpolation="nearest"))
        for q in QUANTILES
    ]
    if boundaries != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError(f"Quantity quantiles are not strictly increasing: {boundaries}")
    labels = [
        f"<= {boundaries[0]:g}",
        f"({boundaries[0]:g}, {boundaries[1]:g}]",
        f"({boundaries[1]:g}, {boundaries[2]:g}]",
        f"({boundaries[2]:g}, {boundaries[3]:g}]",
        f"> {boundaries[3]:g}",
    ]
    keys = ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")
    return {
        "quantiles": list(QUANTILES),
        "boundaries": boundaries,
        "strata": [
            {"stratum_order": index, "stratum": key, "stratum_label": label}
            for index, (key, label) in enumerate(zip(keys, labels))
        ],
    }


def fit_categorical_interface(
    df: pl.DataFrame,
    *,
    mode: str,
    bin_count: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    train = df.filter(pl.col("chronological_split") == "train")
    train_qty = train["demand_qty"].to_numpy().astype(np.float64)
    if mode == "uniform_categorical":
        minimum = float(train_qty.min())
        maximum = float(train_qty.max())
        edges = np.linspace(minimum, maximum, bin_count + 1, dtype=np.float64)[1:-1]
    elif mode == "quantile_categorical":
        edges = np.asarray([
            float(train["demand_qty"].quantile(index / bin_count, interpolation="nearest"))
            for index in range(1, bin_count)
        ])
    else:
        raise ValueError(f"Unsupported categorical mode: {mode}")
    if edges.size != bin_count - 1 or not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"{mode} produced non-increasing train-only edges: {edges.tolist()}")

    all_qty = df["demand_qty"].to_numpy().astype(np.float64)
    all_marks = np.searchsorted(edges, all_qty, side="left").astype(np.int32)
    train_marks = np.searchsorted(edges, train_qty, side="left").astype(np.int32)
    representatives = []
    counts = []
    for mark in range(bin_count):
        values = train_qty[train_marks == mark]
        if values.size == 0:
            raise ValueError(f"{mode} has an empty train bin: {mark}")
        representatives.append(float(np.median(values)))
        counts.append(int(values.size))

    transformed = df.with_columns([
        pl.Series("mark", all_marks, dtype=pl.Int32),
        pl.col("demand_qty").cast(pl.Float64).alias("scale_residual"),
    ])
    metadata = {
        "mode": mode,
        "bin_count": bin_count,
        "edges": edges.tolist(),
        "representatives": representatives,
        "train_bin_counts": counts,
        "fitted_on": "train",
    }
    return transformed, metadata


def make_loader(
    df: pl.DataFrame,
    *,
    target_split: str,
    batch_size: int,
    lookback_weeks: int,
    max_seq_len: int,
    shuffle: bool,
    generator: torch.Generator | None,
) -> DataLoader:
    dataset = RMTPPWeekLookbackDataset(
        df,
        lookback_weeks=lookback_weeks,
        max_seq_len=max_seq_len,
        val_ratio=0.2,
        mode="all",
        split_col="chronological_split",
        target_splits={target_split},
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_week_lookback,
        num_workers=0,
    )


def base_config(
    *,
    num_marks: int,
    hidden_dim: int,
    scale_base: float,
    use_value_head: bool,
    value_input_mode: str,
) -> RMTPPConfig:
    return RMTPPConfig(
        num_marks=num_marks,
        mark_emb_dim=32,
        rnn_hidden_dim=hidden_dim,
        rnn_type="gru",
        dropout=0.1,
        scale_base=scale_base,
        use_value_head=use_value_head,
        value_head_activation="identity",
        value_input_mode=value_input_mode,
        value_input_emb_dim=8,
        train_loss_scope="target_only",
        loss_mode="hybrid" if use_value_head else "residual_only",
        lambda_qty=0.25,
    )


def empty_accumulator() -> dict[str, float]:
    return {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0, "signed_sum": 0.0}


def update_accumulator(
    accumulator: dict[str, float],
    true_qty: np.ndarray,
    pred_qty: np.ndarray,
) -> None:
    error = pred_qty - true_qty
    accumulator["count"] += int(true_qty.size)
    accumulator["abs_sum"] += float(np.abs(error).sum())
    accumulator["sq_sum"] += float(np.square(error).sum())
    accumulator["signed_sum"] += float(error.sum())


def finalize_accumulator(accumulator: dict[str, float]) -> dict[str, float]:
    count = int(accumulator["count"])
    if count < 1:
        raise ValueError("Cannot finalize an empty quantity stratum")
    return {
        "count": count,
        "qty_mae": accumulator["abs_sum"] / count,
        "qty_rmse": float(np.sqrt(accumulator["sq_sum"] / count)),
        "qty_bias": accumulator["signed_sum"] / count,
    }


@torch.no_grad()
def evaluate(
    *,
    model: RMTPP,
    loader: DataLoader,
    variant: str,
    quantile_contract: dict[str, Any],
    device: str,
    representatives: list[float] | None,
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
    negative_preclamp = 0

    for batch_index, (marks, dts, mask, _, values) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if values is None:
            raise ValueError("Ablation loader must provide quantity values")
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

        valid = mask[:, -1] & mask[:, -2]
        input_values = mask_appended_target_value(values, mask)
        hidden = model.forward(marks, dts, values=input_values, mask=mask)[:, -2, :]
        true_mark = marks[:, -1]
        true_value = values[:, -1]
        pad_id = int(model.cfg.num_marks - 1)
        valid = valid & (true_mark != pad_id)
        if not bool(valid.any().item()):
            continue
        hidden = hidden[valid]
        true_mark = true_mark[valid]
        true_value = true_value[valid]
        logits = model.mark_head(hidden)[..., :pad_id]
        pred_mark = torch.argmax(logits, dim=-1)
        correct += int((pred_mark == true_mark).sum().item())
        total += int(true_mark.numel())

        if variant in {"uniform_categorical", "quantile_categorical"}:
            assert representatives is not None
            representatives_tensor = torch.as_tensor(
                representatives,
                dtype=torch.float32,
                device=device,
            )
            true_qty = true_value
            pred_qty = representatives_tensor[pred_mark]
        elif variant == "direct_raw_mse":
            assert isinstance(model, DirectRawRMTPP)
            true_qty = model.reconstruct_qty(true_mark, true_value)
            affine, pred_qty = model.predict_raw_quantity(hidden)
            negative_preclamp += int((affine < 0.0).sum().item())
        elif variant in {"minmax_sigmoid", "log_regression"}:
            assert isinstance(model, PositiveRegressionRMTPP)
            true_qty = model.reconstruct_qty(true_mark, true_value)
            _, pred_qty = model.predict_quantity(hidden)
        elif variant == "mark_residual":
            true_qty = model.reconstruct_qty(true_mark, true_value)
            pred_value = predict_value_for_marks(model, hidden, pred_mark)
            pred_qty = model.reconstruct_qty(pred_mark, pred_value)
        else:
            raise ValueError(f"Unsupported evaluation variant: {variant}")

        true_qty = torch.round(true_qty)
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
        "preclamp_negative_share": negative_preclamp / max(total, 1),
        "rows": rows,
        "evaluated_count": total_count,
    }


def raw_regression_loss(
    model: DirectRawRMTPP,
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
    hidden = hidden[valid]
    true_qty = model.reconstruct_qty(true_mark[valid], true_value[valid])
    target = (true_qty - model.raw_mean.to(true_qty)) / model.raw_std.to(true_qty)
    normalized = model.direct_qty_head(hidden).squeeze(-1)
    return F.mse_loss(normalized, target)


def positive_regression_loss(
    model: PositiveRegressionRMTPP,
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
    hidden = hidden[valid]
    true_qty = model.reconstruct_qty(true_mark[valid], true_value[valid])
    transformed, _ = model.predict_quantity(hidden)
    if model.regression_mode == "minmax_sigmoid":
        target = (true_qty - model.train_min.to(true_qty)) / (
            model.train_max.to(true_qty) - model.train_min.to(true_qty)
        )
    else:
        target = torch.log1p(true_qty)
    return F.mse_loss(transformed, target)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_variant_seed(
    *,
    args: argparse.Namespace,
    original_df: pl.DataFrame,
    variant_df: pl.DataFrame,
    quantile_contract: dict[str, Any],
    variant: str,
    seed: int,
    interface_meta: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = args.output_dir / "runs" / variant / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_val_nll_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    if summary_path.exists() and best_path.exists() and not args.force_rerun:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = summary.pop("stratum_rows")
        return summary, rows

    generator = set_seed(seed)
    train_loader = make_loader(
        variant_df,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=generator,
    )
    val_loader = make_loader(
        variant_df,
        target_split="validation",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=False,
        generator=None,
    )
    if variant in {"uniform_categorical", "quantile_categorical"}:
        num_marks = int(args.bin_count + 1)
        cfg = base_config(
            num_marks=num_marks,
            hidden_dim=args.hidden_dim,
            scale_base=10.0,
            use_value_head=False,
            value_input_mode="none",
        )
        model: RMTPP = RMTPP(cfg)
        representatives = list(interface_meta["representatives"])
    elif variant == "direct_raw_mse":
        if interface_meta is None:
            raise ValueError("direct_raw_mse requires train-only normalization metadata")
        num_marks = int(original_df["mark"].max()) + 2
        cfg = base_config(
            num_marks=num_marks,
            hidden_dim=args.hidden_dim,
            scale_base=10.0,
            use_value_head=False,
            value_input_mode="residual",
        )
        model = DirectRawRMTPP(
            cfg,
            raw_mean=float(interface_meta["train_mean"]),
            raw_std=float(interface_meta["train_std"]),
        )
        representatives = None
    elif variant in {"minmax_sigmoid", "log_regression"}:
        if interface_meta is None:
            raise ValueError(f"{variant} requires train-only transform metadata")
        num_marks = int(original_df["mark"].max()) + 2
        cfg = base_config(
            num_marks=num_marks,
            hidden_dim=args.hidden_dim,
            scale_base=10.0,
            use_value_head=False,
            value_input_mode="residual",
        )
        model = PositiveRegressionRMTPP(
            cfg,
            mode=variant,
            train_min=float(interface_meta["train_min"]),
            train_max=float(interface_meta["train_max"]),
            train_target_mean=float(interface_meta["train_target_mean"]),
        )
        representatives = None
    else:
        raise ValueError(f"Cannot train unsupported variant: {variant}")
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
        previous_revisions = payload.get("source_revision_history") or [
            payload.get("source_revision")
        ]
        source_revision_history = [
            revision for revision in previous_revisions if revision
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
    if stopped_early:
        best_epoch = min(history, key=lambda row: float(row["val_nll"]))["epoch"]
        print(
            f"[early-stop-resume] variant={variant} seed={seed} "
            f"current_epoch={history[-1]['epoch']} best_epoch={best_epoch}",
            flush=True,
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
                raise ValueError("Training requires quantity values")
            marks = marks.to(args.device)
            dts = dts.to(args.device)
            mask = mask.to(args.device)
            values = values.to(args.device)
            out = model.nll(marks, dts, values=values, mask=mask, loss_scope="target_only")
            loss = out["nll_marker"] + out["nll_time"]
            if variant == "direct_raw_mse":
                assert isinstance(model, DirectRawRMTPP)
                loss = loss + args.lambda_raw * raw_regression_loss(
                    model,
                    marks,
                    dts,
                    mask,
                    values,
                )
            elif variant in {"minmax_sigmoid", "log_regression"}:
                assert isinstance(model, PositiveRegressionRMTPP)
                loss = loss + args.lambda_raw * positive_regression_loss(
                    model,
                    marks,
                    dts,
                    mask,
                    values,
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
            variant=variant,
            quantile_contract=quantile_contract,
            device=args.device,
            representatives=representatives,
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
            f"[epoch {epoch:03d}] variant={variant} seed={seed} "
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
            "variant": variant,
            "seed": seed,
            "model_state_dict": clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_nll": best_nll,
            "best_state_dict": best_state,
            "rmtpp_config": {name: getattr(cfg, name) for name in cfg.__dataclass_fields__},
            "interface_meta": interface_meta,
            "source_revision": args.source_revision,
            "source_revision_history": source_revision_history,
            "early_stopping_patience": args.early_stopping_patience,
            "min_epochs": args.min_epochs,
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
                f"[early-stop] variant={variant} seed={seed} "
                f"current_epoch={epoch} best_epoch={best_epoch} "
                f"patience={args.early_stopping_patience}"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for {variant}/seed_{seed}")
    model.load_state_dict(best_state, strict=True)
    validation = evaluate(
        model=model,
        loader=val_loader,
        variant=variant,
        quantile_contract=quantile_contract,
        device=args.device,
        representatives=representatives,
        max_batches=args.max_val_batches,
    )
    state_digest = canonical_state_dict_sha256(best_state)
    checkpoint = {
        "selection": "best_val_nll",
        "variant": variant,
        "seed": seed,
        "model_state_dict": best_state,
        "model_state_sha256": state_digest,
        "rmtpp_config": {name: getattr(cfg, name) for name in cfg.__dataclass_fields__},
        "interface_meta": interface_meta,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "early_stopping_patience": args.early_stopping_patience,
        "min_epochs": args.min_epochs,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    torch.save(checkpoint, best_path)
    rows = [
        {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "seed": seed,
            **row,
        }
        for row in validation["rows"]
    ]
    best_epoch = min(history, key=lambda row: float(row["val_nll"]))["epoch"]
    summary = {
        "status": "success",
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "seed": seed,
        "epochs": args.epochs,
        "completed_epochs": int(history[-1]["epoch"]),
        "stopped_early": int(history[-1]["epoch"]) < args.epochs,
        "best_epoch": int(best_epoch),
        "best_val_nll": float(validation["val_nll"]),
        "best_val_qty_mae": float(validation["rows"][0]["qty_mae"]),
        "best_val_qty_rmse": float(validation["rows"][0]["qty_rmse"]),
        "mark_acc": float(validation["mark_acc"]),
        "preclamp_negative_share": float(validation["preclamp_negative_share"]),
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_path": str(best_path),
        "checkpoint_state_sha256": state_digest,
        "elapsed_seconds": time.time() - started,
        "interface_meta": interface_meta,
        "stratum_rows": rows,
    }
    save_json(summary_path, summary)
    returned = dict(summary)
    returned.pop("stratum_rows")
    return returned, rows


def discover_proposal_checkpoints(root: Path) -> list[Path]:
    canonical_root = root / "runs" / "yellow_trip_hourly" / "rmtpp"
    checkpoints = sorted(
        path
        for path in canonical_root.rglob("best_val_nll_model.pt")
        if "epochs_300" in path.parts
        and any(f"seed_{seed}" in path.parts for seed in SEEDS)
    )
    if len(checkpoints) != 3:
        raise ValueError(f"Expected 3 mark-residual RMTPP checkpoints, found {len(checkpoints)}")
    return checkpoints


def evaluate_proposal(
    *,
    args: argparse.Namespace,
    df: pl.DataFrame,
    quantile_contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    val_loader = make_loader(
        df,
        target_split="validation",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=False,
        generator=None,
    )
    summaries = []
    rows = []
    for path in discover_proposal_checkpoints(args.proposal_root):
        payload = torch_load_checkpoint(path, map_location="cpu")
        if payload.get("selection") != "best_val_nll":
            raise ValueError(f"Unexpected proposal checkpoint selection: {path}")
        run = payload["run_config"]
        seed = int(run["seed"])
        model = RMTPP(dataclass_from_dict(RMTPPConfig, payload["rmtpp_config"]))
        state = payload["model_state_dict"]
        digest = canonical_state_dict_sha256(state)
        if digest != payload.get("model_state_sha256"):
            raise ValueError(f"Proposal checkpoint digest mismatch: {path}")
        model.load_state_dict(state, strict=True)
        model.to(args.device)
        validation = evaluate(
            model=model,
            loader=val_loader,
            variant="mark_residual",
            quantile_contract=quantile_contract,
            device=args.device,
            representatives=None,
            max_batches=args.max_val_batches,
        )
        expected = float(payload["summary"]["best_val_nll_qty_mae"])
        observed = float(validation["rows"][0]["qty_mae"])
        if args.max_val_batches is None and not np.isclose(
            expected,
            observed,
            rtol=5e-5,
            atol=5e-5,
        ):
            raise ValueError(
                f"Proposal MAE mismatch seed={seed}: observed={observed} expected={expected}"
            )
        summaries.append({
            "status": "success",
            "variant": "mark_residual",
            "variant_label": VARIANT_LABELS["mark_residual"],
            "seed": seed,
            "epochs": 300,
            "completed_epochs": 300,
            "stopped_early": False,
            "best_epoch": int(payload["summary"]["best_val_nll_epoch"]),
            "best_val_nll": float(validation["val_nll"]),
            "best_val_qty_mae": observed,
            "best_val_qty_rmse": float(validation["rows"][0]["qty_rmse"]),
            "mark_acc": float(validation["mark_acc"]),
            "source_revision": payload["summary"]["source_revision"],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "checkpoint_path": str(path),
            "checkpoint_state_sha256": digest,
        })
        rows.extend({
            "variant": "mark_residual",
            "variant_label": VARIANT_LABELS["mark_residual"],
            "seed": seed,
            **row,
        } for row in validation["rows"])
    return summaries, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    seed_rows: list[dict[str, Any]],
    *,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    output = []
    strata = sorted({
        (int(row["stratum_order"]), row["stratum"], row["stratum_label"])
        for row in seed_rows
    })
    for variant in variants:
        for order, key, label in strata:
            group = [
                row for row in seed_rows
                if row["variant"] == variant and row["stratum"] == key
            ]
            if {int(row["seed"]) for row in group} != set(seeds):
                raise ValueError(f"Seed contract failed for {variant}/{key}")
            record = {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
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
                record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output.append(record)
    return output


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    variants = parse_str_tuple(args.variants)
    unsupported = sorted(set(variants) - set(NEW_VARIANTS))
    if unsupported:
        raise ValueError(f"Unsupported train variants: {unsupported}")
    seeds = parse_int_tuple(args.seeds)
    if not args.allow_partial_contract and set(seeds) != set(SEEDS):
        raise ValueError(f"Qualified run requires seeds {SEEDS}, received {seeds}")
    if len(args.source_revision) != 40:
        raise ValueError("source-revision must be a 40-character Git SHA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_df = pl.read_parquet(args.data).sort(["oper_part_no", "seq"])
    required = {
        "oper_part_no",
        "seq",
        "delta_t",
        "demand_qty",
        "mark",
        "scale_residual",
        "chronological_split",
    }
    missing = sorted(required - set(original_df.columns))
    if missing:
        raise ValueError(f"Taxi fixed split is missing columns: {missing}")
    integral_error = (
        original_df["demand_qty"] - original_df["demand_qty"].round(0)
    ).abs().max()
    if float(integral_error) > 1e-9:
        raise ValueError("Taxi quantity-interface ablation requires integral demand quantities")
    quantile_contract = train_quantile_contract(original_df)
    train_qty = original_df.filter(
        pl.col("chronological_split") == "train"
    )["demand_qty"].to_numpy().astype(np.float64)
    variant_data: dict[str, tuple[pl.DataFrame, dict[str, Any] | None]] = {
        "direct_raw_mse": (original_df, {
            "mode": "direct_raw_mse",
            "target": "demand_qty",
            "loss": "mse_on_train_standardized_raw_quantity",
            "train_mean": float(train_qty.mean()),
            "train_std": float(train_qty.std()),
            "history_quantity_input": "log10_within_mark_residual",
            "fitted_on": "train",
        }),
        "minmax_sigmoid": (original_df, {
            "mode": "minmax_sigmoid",
            "target": "train_minmax_scaled_demand_qty",
            "loss": "mse_on_minmax_scaled_quantity",
            "output_activation": "sigmoid",
            "inverse_transform": "train_min_plus_range_times_sigmoid",
            "train_min": float(train_qty.min()),
            "train_max": float(train_qty.max()),
            "train_target_mean": float(train_qty.mean()),
            "history_quantity_input": "log10_within_mark_residual",
            "support": "closed_train_quantity_range",
            "fitted_on": "train",
        }),
        "log_regression": (original_df, {
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
        }),
    }
    for variant in ("uniform_categorical", "quantile_categorical"):
        variant_data[variant] = fit_categorical_interface(
            original_df,
            mode=variant,
            bin_count=args.bin_count,
        )

    split_rows = {
        str(row["chronological_split"]): int(row["len"])
        for row in original_df.group_by("chronological_split").agg(pl.len()).iter_rows(named=True)
    }
    contract = {
        "schema_version": 1,
        "status": "running",
        "dataset": "yellow_trip_hourly",
        "data_path": str(args.data),
        "data_sha256": sha256_file(args.data),
        "split_rows": split_rows,
        "quantile_contract": quantile_contract,
        "interfaces": {
            variant: metadata for variant, (_, metadata) in variant_data.items()
        },
        "variants": [*variants, *([] if args.skip_proposal else ["mark_residual"])],
        "seeds": list(seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_raw": args.lambda_raw,
        "grad_clip": args.grad_clip,
        "early_stopping": {
            "enabled": args.early_stopping_patience > 0,
            "monitor": "val_nll",
            "patience": args.early_stopping_patience,
            "min_epochs": args.min_epochs,
            "restore": "best_val_nll",
        },
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "hidden_dim": args.hidden_dim,
        "checkpoint_selection": "best_val_nll",
        "evaluation_quantity_target": "nearest_integer_demand_qty",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "proposal_source_revisions": [],
        "qualified_seed_contract": set(seeds) == set(SEEDS),
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }
    save_json(args.output_dir / "launch_contract.json", contract)

    summaries: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for variant in variants:
        df, interface_meta = variant_data[variant]
        for seed in seeds:
            summary, rows = train_variant_seed(
                args=args,
                original_df=original_df,
                variant_df=df,
                quantile_contract=quantile_contract,
                variant=variant,
                seed=seed,
                interface_meta=interface_meta,
            )
            summaries.append(summary)
            seed_rows.extend(rows)

    proposal_summaries: list[dict[str, Any]] = []
    if not args.skip_proposal:
        proposal_summaries, proposal_rows = evaluate_proposal(
            args=args,
            df=original_df,
            quantile_contract=quantile_contract,
        )
        summaries.extend(proposal_summaries)
        seed_rows.extend(proposal_rows)
    summary_variants = (*variants, *(() if args.skip_proposal else ("mark_residual",)))
    summary_rows = summarize(seed_rows, variants=summary_variants, seeds=seeds)
    write_csv(args.output_dir / "run_summaries.csv", summaries)
    write_csv(args.output_dir / "quantity_interface_seed_metrics.csv", seed_rows)
    write_csv(args.output_dir / "quantity_interface_summary.csv", summary_rows)

    contract["status"] = "complete"
    contract["proposal_source_revisions"] = sorted({
        row["source_revision"] for row in proposal_summaries
    })
    save_json(args.output_dir / "launch_contract.json", contract)
    print(f"[complete] output_dir={args.output_dir} runs={len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
