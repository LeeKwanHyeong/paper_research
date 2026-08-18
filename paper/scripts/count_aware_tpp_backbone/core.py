"""Data, objective, and evaluation primitives for count-aware TPP runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch

from models.TPPs.CountAwareTPP import SharedTimeCountModel
from paper.scripts.run_intermittent_log_backbone_control import (
    HISTORY_BOUNDARIES,
    HISTORY_STRATA,
)


def prepare_count_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Map raw demand to the loader fields used by count-aware experiments."""
    if frame.filter(pl.col("demand_qty") < 0).height:
        raise ValueError("Count-aware input requires nonnegative demand_qty")
    return frame.with_columns(
        [
            pl.lit(0, dtype=pl.Int32).alias("mark"),
            pl.col("demand_qty").cast(pl.Float64).alias("scale_residual"),
        ]
    )


def right_pad_batch(
    dts: torch.Tensor,
    quantities: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert loader left-padding to right-padding and return valid lengths."""
    batch_size, seq_len = mask.shape
    positions = torch.arange(seq_len, device=mask.device).expand(batch_size, -1)
    sort_key = (~mask).long() * seq_len + positions
    order = torch.argsort(sort_key, dim=1)
    right_dts = torch.gather(dts, 1, order)
    right_quantities = torch.gather(quantities, 1, order)
    right_mask = torch.gather(mask, 1, order)
    lengths = right_mask.sum(dim=1)
    if bool((lengths < 2).any()):
        raise ValueError("Every next-event sample requires at least one history event")
    return right_dts, right_quantities, right_mask, lengths


def target_outputs(
    model: SharedTimeCountModel,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    *,
    lambda_log_qty: float,
) -> dict[str, torch.Tensor]:
    """Evaluate the final next-event target without exposing its quantity."""
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    encoded = model.encode(dts, history_quantities, mask)
    hidden = encoded[batch_ids, history_positions]
    true_dt = dts[batch_ids, target_positions].float()
    true_qty = quantities[batch_ids, target_positions].float()
    time_loss = -model.log_f_dt(hidden, true_dt)
    quantity = model.quantity_outputs(hidden, true_qty)
    return {
        "joint_loss": time_loss + float(lambda_log_qty) * quantity["train_loss"],
        "time_loss": time_loss,
        "quantity_train_loss": quantity["train_loss"],
        "log_qty_loss": quantity["log_mse"],
        "quantity_distribution_nll": quantity["distribution_nll"],
        "quantity_location_huber": quantity["location_huber"],
        "quantity_scale": quantity["scale"],
        "tail_aux_loss": quantity["tail_aux_loss"],
        "tail_indicator": quantity["tail_indicator"],
        "true_qty": true_qty,
        "pred_qty": quantity["point_prediction"],
        "history_length": lengths - 1,
    }


def empty_accumulator() -> dict[str, float]:
    return {
        "count": 0,
        "joint_sum": 0.0,
        "time_sum": 0.0,
        "quantity_train_sum": 0.0,
        "log_qty_sum": 0.0,
        "distribution_nll_sum": 0.0,
        "location_huber_sum": 0.0,
        "tail_aux_sum": 0.0,
        "tail_count": 0,
        "scale_sum": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "signed_sum": 0.0,
    }


def update_accumulator(
    accumulator: dict[str, float],
    *,
    joint: np.ndarray,
    time_nll: np.ndarray,
    quantity_train_loss: np.ndarray,
    log_qty_mse: np.ndarray,
    distribution_nll: np.ndarray,
    location_huber: np.ndarray,
    tail_aux_loss: np.ndarray,
    tail_indicator: np.ndarray,
    scale: np.ndarray,
    true_qty: np.ndarray,
    pred_qty: np.ndarray,
) -> None:
    error = pred_qty - true_qty
    accumulator["count"] += int(true_qty.size)
    accumulator["joint_sum"] += float(joint.sum())
    accumulator["time_sum"] += float(time_nll.sum())
    accumulator["quantity_train_sum"] += float(quantity_train_loss.sum())
    accumulator["log_qty_sum"] += float(log_qty_mse.sum())
    accumulator["distribution_nll_sum"] += float(distribution_nll.sum())
    accumulator["location_huber_sum"] += float(location_huber.sum())
    accumulator["tail_aux_sum"] += float(tail_aux_loss.sum())
    accumulator["tail_count"] += int(tail_indicator.sum())
    accumulator["scale_sum"] += float(scale.sum())
    accumulator["abs_sum"] += float(np.abs(error).sum())
    accumulator["sq_sum"] += float(np.square(error).sum())
    accumulator["signed_sum"] += float(error.sum())


def finalize_accumulator(accumulator: dict[str, float]) -> dict[str, Any]:
    count = int(accumulator["count"])
    if count < 1:
        raise ValueError("Cannot finalize an empty accumulator")
    return {
        "count": count,
        "joint_objective": accumulator["joint_sum"] / count,
        "time_nll": accumulator["time_sum"] / count,
        "quantity_train_loss": accumulator["quantity_train_sum"] / count,
        "log_qty_mse": accumulator["log_qty_sum"] / count,
        "quantity_distribution_nll": accumulator["distribution_nll_sum"] / count,
        "quantity_location_huber": accumulator["location_huber_sum"] / count,
        "tail_aux_loss": accumulator["tail_aux_sum"] / count,
        "tail_count": int(accumulator["tail_count"]),
        "quantity_scale_mean": accumulator["scale_sum"] / count,
        "qty_mae": accumulator["abs_sum"] / count,
        "qty_rmse": float(np.sqrt(accumulator["sq_sum"] / count)),
        "qty_bias": accumulator["signed_sum"] / count,
    }


@torch.no_grad()
def evaluate(
    *,
    model: SharedTimeCountModel,
    loader: Any,
    quantity_contract: dict[str, Any],
    device: str,
    lambda_log_qty: float,
    max_batches: int | None,
    include_breakdowns: bool,
) -> dict[str, Any]:
    """Evaluate shared metrics overall and by quantity/history strata."""
    model.eval()
    overall = empty_accumulator()
    quantity_accumulators = [
        empty_accumulator() for _ in quantity_contract["strata"]
    ]
    history_accumulators = [empty_accumulator() for _ in HISTORY_STRATA]

    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if quantities is None:
            raise ValueError("Count-aware evaluation requires raw quantities")
        outputs = target_outputs(
            model,
            dts.to(device),
            mask.to(device),
            quantities.to(device),
            lambda_log_qty=lambda_log_qty,
        )
        arrays = {
            "joint": outputs["joint_loss"].cpu().numpy().astype(np.float64),
            "time_nll": outputs["time_loss"].cpu().numpy().astype(np.float64),
            "quantity_train_loss": outputs["quantity_train_loss"].cpu().numpy().astype(np.float64),
            "log_qty_mse": outputs["log_qty_loss"].cpu().numpy().astype(np.float64),
            "distribution_nll": (
                outputs["quantity_distribution_nll"]
                .cpu()
                .numpy()
                .astype(np.float64)
            ),
            "location_huber": outputs["quantity_location_huber"].cpu().numpy().astype(np.float64),
            "tail_aux_loss": outputs["tail_aux_loss"].cpu().numpy().astype(np.float64),
            "tail_indicator": outputs["tail_indicator"].cpu().numpy().astype(np.float64),
            "scale": outputs["quantity_scale"].cpu().numpy().astype(np.float64),
            "true_qty": outputs["true_qty"].cpu().numpy().astype(np.float64),
            "pred_qty": outputs["pred_qty"].cpu().numpy().astype(np.float64),
        }
        history_length = outputs["history_length"].cpu().numpy().astype(np.int64)
        update_accumulator(overall, **arrays)
        if not include_breakdowns:
            continue

        quantity_ids = np.searchsorted(
            np.asarray(quantity_contract["boundaries"], dtype=np.float64),
            arrays["true_qty"],
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
                    update_accumulator(
                        accumulator,
                        **{key: value[selected] for key, value in arrays.items()},
                    )

    overall_metrics = finalize_accumulator(overall)
    result: dict[str, Any] = {
        "val_joint_objective": overall_metrics["joint_objective"],
        "val_time_nll": overall_metrics["time_nll"],
        "val_quantity_train_loss": overall_metrics["quantity_train_loss"],
        "val_log_qty_mse": overall_metrics["log_qty_mse"],
        "val_quantity_distribution_nll": overall_metrics["quantity_distribution_nll"],
        "val_quantity_location_huber": overall_metrics["quantity_location_huber"],
        "val_tail_aux_loss": overall_metrics["tail_aux_loss"],
        "val_tail_count": overall_metrics["tail_count"],
        "val_quantity_scale_mean": overall_metrics["quantity_scale_mean"],
        "qty_mae": overall_metrics["qty_mae"],
        "qty_rmse": overall_metrics["qty_rmse"],
        "evaluated_count": overall_metrics["count"],
    }
    if not include_breakdowns:
        return result

    result["quantity_rows"] = [
        {
            **spec,
            "share": int(accumulator["count"]) / overall_metrics["count"],
            **finalize_accumulator(accumulator),
        }
        for spec, accumulator in zip(
            quantity_contract["strata"], quantity_accumulators
        )
        if int(accumulator["count"]) > 0
    ]
    result["history_rows"] = [
        {
            **spec,
            "share": int(accumulator["count"]) / overall_metrics["count"],
            **finalize_accumulator(accumulator),
        }
        for spec, accumulator in zip(HISTORY_STRATA, history_accumulators)
        if int(accumulator["count"]) > 0
    ]
    return result


__all__ = [
    "empty_accumulator",
    "evaluate",
    "finalize_accumulator",
    "prepare_count_frame",
    "right_pad_batch",
    "target_outputs",
    "update_accumulator",
]
