from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.event_seq_data_module import time_split_events, RMTPPDataset, collate_next_event, \
    RMTPPWeekLookbackDataset, collate_week_lookback
from models.RMTPPs.RMTPP import RMTPP
from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.TransformerHawkesTPP import TransformerHawkesTPP
from models.RMTPPs.marker_losses import normalized_ranked_probability_score
from models.RMTPPs.magnitude_normalization import normalized_magnitude_target
from models.RMTPPs.value_conditioning import (
    mask_appended_target_value,
    predict_value_for_marks,
)
import numpy as np
import polars as pl

from models.RMTPPs.config import RMTPPConfig, THPConfig
from models.Titan import TitanConfig


def _mark_metrics_from_confusion(confusion: np.ndarray) -> Dict[str, float]:
    total = int(confusion.sum())
    if total <= 0:
        return {
            "mark_balanced_accuracy": float("nan"),
            "mark_macro_f1": float("nan"),
            "mark_mae": float("nan"),
            "mark_adjacent_accuracy": float("nan"),
            "mark_pred_0_share": float("nan"),
            "mark_0_recall": float("nan"),
            "mark_1_recall": float("nan"),
        }

    true_counts = confusion.sum(axis=1)
    pred_counts = confusion.sum(axis=0)
    correct = np.diag(confusion)
    recall = np.divide(
        correct,
        true_counts,
        out=np.zeros_like(correct, dtype=np.float64),
        where=true_counts > 0,
    )
    precision = np.divide(
        correct,
        pred_counts,
        out=np.zeros_like(correct, dtype=np.float64),
        where=pred_counts > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) > 0,
    )
    supported = true_counts > 0
    marks = np.arange(confusion.shape[0])
    distance = np.abs(marks[:, None] - marks[None, :])

    return {
        "mark_balanced_accuracy": float(recall[supported].mean()),
        "mark_macro_f1": float(f1[supported].mean()),
        "mark_mae": float((confusion * distance).sum() / total),
        "mark_adjacent_accuracy": float(confusion[distance <= 1].sum() / total),
        "mark_pred_0_share": float(pred_counts[0] / total),
        "mark_0_recall": float(recall[0]),
        "mark_1_recall": float(recall[1]) if confusion.shape[0] > 1 else float("nan"),
    }


@dataclass
class TrainingConfig:
    lookback: int = 30
    max_seq_len: int = 64
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 20
    val_ratio: float = 0.2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lambda_value: float = 1.0
    lambda_dt: float = 1.0 # dt loss weight
    grad_clip: float = 1.0


def eval_next_event(model: RMTPP, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    dt_abs = 0
    dt_sq = 0.0

    with torch.no_grad():
        for x_mk, x_dt, y_mk, y_dt, _, _, _ in loader:
            x_mk = x_mk.to(device)
            x_dt = x_dt.to(device)
            y_mk = y_mk.to(device)
            y_dt = y_dt.to(device)

            h = model.forward(x_mk, x_dt)
            h_last = h[:, -1, :]

            logits = model.mark_head(h_last)
            pred = torch.argmax(logits, dim = -1)

            correct += (pred == y_mk).sum().item()
            total += y_mk.numel()

            u = torch.full_like(y_dt, 0.5, device=device).clamp_min(model.cfg.eps)
            dt_hat = model.sample_next_dt(h_last, u=u).clamp_min(1.0)

            err = (dt_hat - y_dt).abs()
            dt_abs += err.sum().item()
            dt_sq += ((dt_hat - y_dt) ** 2).sum().item()

        acc = correct / max(total, 1)
        mae = dt_abs / max(total, 1)
        rmse = np.sqrt(dt_sq / max(total, 1))
        return {'mark_acc': acc, 'dt_mae': mae, 'dt_rmse': rmse}


def eval_next_event_classic(model, loader: DataLoader, device: str) -> Dict[str, float]:
    """
    Classic event-count lookback evaluation.

    Dataset shape:
      - input:  x_mk [B, L], x_dt [B, L]
      - target: y_mk [B],    y_dt [B]

    Each batch contributes exactly one supervised next-event step per sample.
    """
    model.eval()

    total = 0
    correct = 0
    dt_abs = 0.0
    dt_sq = 0.0
    value_abs = 0.0
    qty_abs = 0.0

    sum_nll_time = 0.0
    sum_nll_marker = 0.0
    sum_nll_total = 0.0
    sum_value_loss = 0.0
    sum_ordinal_marker_loss = 0.0
    num_real_marks = int(model.cfg.num_marks - 1)
    mark_confusion = np.zeros((num_real_marks, num_real_marks), dtype=np.int64)

    with torch.no_grad():
        for x_mk, x_dt, y_mk, y_dt, _, _, y_val in loader:
            x_mk = x_mk.to(device)
            x_dt = x_dt.to(device)
            y_mk = y_mk.to(device)
            y_dt = y_dt.to(device).float()
            y_val = y_val.to(device).float() if y_val is not None else None

            h = model.forward(x_mk, x_dt)
            h_last = h[:, -1, :]

            logits = model.mark_head(h_last)
            log_y = -F.cross_entropy(logits, y_mk, reduction='none')
            logf_dt = model.log_f_dt(h_last, y_dt)
            ordinal_scores = normalized_ranked_probability_score(
                logits,
                y_mk,
                num_real_marks=num_real_marks,
            )

            sum_nll_marker += float((-log_y).sum().item())
            sum_nll_time += float((-logf_dt).sum().item())
            sum_nll_total += float(((-log_y) + (-logf_dt)).sum().item())
            sum_ordinal_marker_loss += float(ordinal_scores.sum().item())

            logits = logits[..., : model.cfg.num_marks - 1]
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y_mk).sum().item()
            total += y_mk.numel()
            np.add.at(
                mark_confusion,
                (y_mk.detach().cpu().numpy(), pred.detach().cpu().numpy()),
                1,
            )

            u = torch.full_like(y_dt, 0.5, device=device).clamp_min(model.cfg.eps)
            dt_hat = model.sample_next_dt(h_last, u=u).clamp_min(1.0)

            err = (dt_hat - y_dt).abs()
            dt_abs += err.sum().item()
            dt_sq += ((dt_hat - y_dt) ** 2).sum().item()

            if y_val is not None and getattr(model.cfg, "use_value_head", False):
                value_hat = predict_value_for_marks(model, h_last, y_mk)
                value_loss = F.huber_loss(value_hat, y_val, reduction="none")
                sum_value_loss += float(value_loss.sum().item())
                value_abs += (value_hat - y_val).abs().sum().item()

                qty_value_hat = predict_value_for_marks(model, h_last, pred)
                qty_hat = model.reconstruct_qty(pred, qty_value_hat)
                qty_true = model.reconstruct_qty(y_mk, y_val)
                qty_abs += (qty_hat - qty_true).abs().sum().item()

    acc = correct / max(total, 1)
    mae = dt_abs / max(total, 1)
    rmse = float(np.sqrt(dt_sq / max(total, 1)))
    value_mae = value_abs / max(total, 1)
    qty_mae = qty_abs / max(total, 1)

    return {
        "mark_acc": acc,
        "dt_mae": mae,
        "dt_rmse": rmse,
        "value_mae": value_mae,
        "qty_mae": qty_mae,
        "val_nll_time": sum_nll_time / max(total, 1),
        "val_nll_marker": sum_nll_marker / max(total, 1),
        "val_nll": sum_nll_total / max(total, 1),
        "val_ordinal_marker_loss": sum_ordinal_marker_loss / max(total, 1),
        "val_value_loss": sum_value_loss / max(total, 1),
        **_mark_metrics_from_confusion(mark_confusion),
        "_total": total,
        "_correct": correct,
        "_nll_steps": total,
    }

def _forward_with_optional_mask(
    model,
    marks: torch.Tensor,
    dts: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor | None = None,
):
    """
    THP uses the padding mask, while RMTPP/TitanTPP keep the older two-argument
    forward signature. This small adapter keeps evaluation code shared.
    """
    input_values = mask_appended_target_value(values, mask)
    try:
        return model.forward(marks, dts, values=input_values, mask=mask)
    except TypeError:
        try:
            return model.forward(marks, dts, values=input_values)
        except TypeError:
            return model.forward(marks, dts)


def eval_next_event_week_lookback(
    model,
    loader: DataLoader,
    device: str,
    *,
    target_only_nll: bool = False,
) -> Dict[str, float]:
    """
    Week-lookback + padding/mask 버전 평가.
    Dataset이 (marks, dts, mask, part_idx)를 반환.

    시퀀스 구성:
      [context events within W weeks] + [target event]
      => 각 샘플에서 target은 '마지막 유효 토큰'
      => 예측은 target 직전 토큰(h_{T-1})으로 target 토큰(mark_T, dt_T)을 맞춤
    """
    model.eval()

    total = 0
    correct = 0
    dt_abs = 0.0
    dt_sq = 0.0
    value_abs = 0.0
    qty_abs = 0.0
    qty_sq = 0.0
    qty_true_abs = 0.0
    log_qty_abs = 0.0
    log_qty_sq = 0.0

    sum_nll_time = 0.0
    sum_nll_marker = 0.0
    sum_nll_total = 0.0
    sum_value_loss = 0.0
    sum_magnitude_loss = 0.0
    sum_log_qty_aux_loss = 0.0
    sum_steps = 0.0
    sum_ordinal_marker_loss = 0.0
    sum_ordinal_steps = 0.0

    pad_id = int(model.cfg.num_marks - 1)
    use_direct_magnitude = bool(getattr(model, "use_direct_magnitude", False))
    use_direct_raw_quantity = bool(getattr(model, "use_direct_raw_quantity", False))
    raw_affine_negative = 0
    raw_affine_count = 0
    magnitude_centers: list[float] = []
    magnitude_scales: list[float] = []
    normalized_target_abs: list[float] = []
    normalized_target_nonfinite = 0
    scale_floor_count = 0
    context_buckets = {
        "1": {"count": 0, "qty_abs": 0.0, "log_abs": 0.0},
        "2_4": {"count": 0, "qty_abs": 0.0, "log_abs": 0.0},
        "5_8": {"count": 0, "qty_abs": 0.0, "log_abs": 0.0},
        "9_plus": {"count": 0, "qty_abs": 0.0, "log_abs": 0.0},
    }
    mark_confusion = np.zeros((pad_id, pad_id), dtype=np.int64)

    with torch.no_grad():
        for marks, dts, mask, _, values in loader:
            marks = marks.to(device)  # (B, Lmax)
            dts   = dts.to(device)    # (B, Lmax)
            mask  = mask.to(device)   # (B, Lmax)
            values = values.to(device) if values is not None else None

            # --------------------- NLL -----------------------
            # Legacy validation reports the average NLL across every transition
            # inside the padded window. Fixed train/validation/test splits should
            # instead score only the final target transition; otherwise test NLL
            # can be dominated by repeated train-context transitions.
            if not target_only_nll:
                out = model.nll(marks, dts, values=values, mask=mask, loss_scope="all")
                # out: {"nll", "nll_time", "nll_marker", "steps"...} 형태라고 가정
                steps = float(
                    out.get("steps", mask[:, 1:].sum()).item() if hasattr(out.get("steps", None), "item") else out.get(
                        "steps", mask[:, 1:].sum().item()))
                # steps가 tensor일 수도 있어서 안전 처리
                if steps <= 0:
                    steps = float(mask[:, 1:].sum().item())

                sum_nll_time += float(out["nll_time"].item()) * steps
                sum_nll_marker += float(out["nll_marker"].item()) * steps
                sum_nll_total += float(out["nll"].item()) * steps
                sum_value_loss += float(out["value_loss"].item()) * steps
                if use_direct_magnitude:
                    sum_magnitude_loss += float(out["magnitude_loss"].item()) * steps
                    sum_log_qty_aux_loss += float(out["log_qty_aux_loss"].item()) * steps
                sum_steps += steps
                if "ordinal_marker_loss" in out:
                    sum_ordinal_marker_loss += float(out["ordinal_marker_loss"].item()) * steps
                    sum_ordinal_steps += steps

            # --------------------- Marker -----------------------
            context_counts = (mask.sum(dim=1) - 1).clamp_min(0)
            valid = mask[:, -1] & mask[:, -2]
            if valid.sum().item() == 0:
                continue

            h = _forward_with_optional_mask(model, marks, dts, mask, values)     # (B, Lmax, H)
            h_prev = h[:, -2, :]              # (B, H)
            y_mk   = marks[:, -1]             # (B,)
            y_dt   = dts[:, -1].float()       # (B,)
            y_val  = values[:, -1].float() if values is not None else None

            direct_prediction = None
            direct_magnitude_step = None
            direct_log_qty_aux_step = None
            if use_direct_magnitude:
                if values is None:
                    raise ValueError("Direct magnitude evaluation requires residual values.")
                direct_prediction = model.predict_direct_magnitude(
                    h_prev,
                    marks=marks,
                    values=values,
                    mask=mask,
                )
                target_log_qty_all = y_mk.float() + y_val
                target_magnitude_all = (
                    torch.exp2(target_log_qty_all)
                    if use_direct_raw_quantity
                    else target_log_qty_all
                )
                normalized_target_all = normalized_magnitude_target(
                    target_magnitude_all,
                    direct_prediction["context"],
                )
                direct_magnitude_step = F.huber_loss(
                    direct_prediction["normalized"],
                    normalized_target_all,
                    reduction="none",
                )
                direct_log_qty_aux_step = model.log_qty_auxiliary_step(
                    direct_prediction["affine_qty"],
                    target_magnitude_all,
                )

            # valid 필터 + 혹시 모를 PAD target 제거
            valid = valid & (y_mk != pad_id)
            if valid.sum().item() == 0:
                continue

            h_prev = h_prev[valid]
            y_mk   = y_mk[valid]
            y_dt   = y_dt[valid]
            context_counts = context_counts[valid]
            if y_val is not None:
                y_val = y_val[valid]

            # ---- mark acc ----
            logits = model.mark_head(h_prev)      # (Bv, K_with_pad)
            if target_only_nll:
                log_y = -F.cross_entropy(logits, y_mk, reduction="none")
                logf_dt = model.log_f_dt(h_prev, y_dt)
                ordinal_scores = normalized_ranked_probability_score(
                    logits,
                    y_mk,
                    num_real_marks=pad_id,
                )
                sum_nll_marker += float((-log_y).sum().item())
                sum_nll_time += float((-logf_dt).sum().item())
                sum_nll_total += float(((-log_y) + (-logf_dt)).sum().item())
                sum_ordinal_marker_loss += float(ordinal_scores.sum().item())
                sum_ordinal_steps += float(y_mk.numel())
                if use_direct_magnitude:
                    assert direct_magnitude_step is not None
                    assert direct_log_qty_aux_step is not None
                    sum_magnitude_loss += float(direct_magnitude_step[valid].sum().item())
                    sum_log_qty_aux_loss += float(
                        direct_log_qty_aux_step[valid].sum().item()
                    )
                elif y_val is not None and getattr(model.cfg, "use_value_head", False):
                    value_hat_for_nll = predict_value_for_marks(model, h_prev, y_mk)
                    value_loss = F.huber_loss(value_hat_for_nll, y_val, reduction="none")
                    sum_value_loss += float(value_loss.sum().item())
                sum_steps += float(y_mk.numel())

            # The final class is reserved for PAD, so exclude it from inference.
            logits = logits[..., :pad_id]
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y_mk).sum().item()
            total += y_mk.numel()
            np.add.at(
                mark_confusion,
                (y_mk.detach().cpu().numpy(), pred.detach().cpu().numpy()),
                1,
            )

            # ---- dt point estimate: median(u=0.5) ----
            u = torch.full((y_dt.size(0),), 0.5, device=device).clamp_min(model.cfg.eps)
            dt_hat = model.sample_next_dt(h_prev, u=u).clamp_min(1.0)

            err = (dt_hat - y_dt).abs()
            dt_abs += err.sum().item()
            dt_sq  += ((dt_hat - y_dt) ** 2).sum().item()

            if use_direct_magnitude:
                assert direct_prediction is not None
                qty_hat = direct_prediction["qty"][valid]
                log_qty_hat = direct_prediction["log_qty"][valid]
                true_log_qty = y_mk.float() + y_val
                if use_direct_raw_quantity:
                    qty_true = torch.exp2(true_log_qty)
                else:
                    qty_true = torch.exp2(
                        true_log_qty.clamp(
                            min=float(model.cfg.magnitude_exp_clamp_min),
                            max=float(model.cfg.magnitude_exp_clamp_max),
                        )
                    )
                qty_error = qty_hat - qty_true
                qty_abs += qty_error.abs().sum().item()
                qty_sq += qty_error.square().sum().item()
                qty_true_abs += qty_true.abs().sum().item()
                log_error = log_qty_hat - true_log_qty
                log_qty_abs += log_error.abs().sum().item()
                log_qty_sq += (log_error ** 2).sum().item()

                if use_direct_raw_quantity:
                    affine_qty = direct_prediction["affine_qty"][valid]
                    raw_affine_negative += int((affine_qty < 0.0).sum().item())
                    raw_affine_count += int(affine_qty.numel())
                    context = direct_prediction["context"]
                    centers = context.center.squeeze(-1)[valid]
                    scales = context.scale.squeeze(-1)[valid]
                    counts = context.context_count.squeeze(-1)[valid]
                    context_counts = counts
                    normalized_targets = normalized_target_all[valid]
                    finite_normalized = torch.isfinite(normalized_targets)
                    normalized_target_nonfinite += int((~finite_normalized).sum().item())
                    magnitude_centers.extend(centers.detach().cpu().tolist())
                    magnitude_scales.extend(scales.detach().cpu().tolist())
                    normalized_target_abs.extend(
                        normalized_targets[finite_normalized].abs().detach().cpu().tolist()
                    )
                    floor = float(model.cfg.magnitude_sigma_floor)
                    if getattr(model, "magnitude_norm_mode", "global") != "causal_revin":
                        scale_floor_count += int(
                            (scales <= floor * (1.0 + 1e-6)).sum().item()
                        )
            elif y_val is not None and getattr(model.cfg, "use_value_head", False):
                value_hat = predict_value_for_marks(model, h_prev, y_mk)
                value_abs += (value_hat - y_val).abs().sum().item()

                qty_value_hat = predict_value_for_marks(model, h_prev, pred)
                qty_hat = model.reconstruct_qty(pred, qty_value_hat)
                qty_true = model.reconstruct_qty(y_mk, y_val)
                qty_error = qty_hat - qty_true
                qty_abs += qty_error.abs().sum().item()
                qty_sq += qty_error.square().sum().item()
                qty_true_abs += qty_true.abs().sum().item()
                log_error = torch.log2(qty_hat.clamp_min(model.cfg.eps)) - torch.log2(
                    qty_true.clamp_min(model.cfg.eps)
                )
                log_qty_abs += log_error.abs().sum().item()
                log_qty_sq += (log_error ** 2).sum().item()

            if y_val is not None and (
                use_direct_magnitude or getattr(model.cfg, "use_value_head", False)
            ):
                for bucket_name, bucket_mask in {
                    "1": context_counts == 1,
                    "2_4": (context_counts >= 2) & (context_counts <= 4),
                    "5_8": (context_counts >= 5) & (context_counts <= 8),
                    "9_plus": context_counts >= 9,
                }.items():
                    if not bucket_mask.any():
                        continue
                    bucket = context_buckets[bucket_name]
                    bucket["count"] += int(bucket_mask.sum().item())
                    bucket["qty_abs"] += float(qty_error[bucket_mask].abs().sum().item())
                    bucket["log_abs"] += float(log_error[bucket_mask].abs().sum().item())

    acc = correct / max(total, 1)
    mae = dt_abs / max(total, 1)
    rmse = float(np.sqrt(dt_sq / max(total, 1)))
    value_mae = float("nan") if use_direct_magnitude else value_abs / max(total, 1)
    qty_mae = qty_abs / max(total, 1)
    qty_rmse = float(np.sqrt(qty_sq / max(total, 1)))
    qty_wape = qty_abs / max(qty_true_abs, float(model.cfg.eps))
    log_qty_mae = log_qty_abs / max(total, 1)
    log_qty_rmse = float(np.sqrt(log_qty_sq / max(total, 1)))

    # step-weighted mean nll
    if sum_steps <= 0:
        val_nll_time = float("nan")
        val_nll_marker = float("nan")
        val_nll_total = float("nan")
    else:
        val_nll_time = sum_nll_time / sum_steps
        val_nll_marker = sum_nll_marker / sum_steps
        val_nll_total = sum_nll_total / sum_steps
        val_value_loss = (
            float("nan") if use_direct_magnitude else sum_value_loss / sum_steps
        )
    val_magnitude_loss = (
        sum_magnitude_loss / sum_steps
        if use_direct_magnitude and sum_steps > 0
        else float("nan")
    )
    val_log_qty_aux_loss = (
        sum_log_qty_aux_loss / sum_steps
        if use_direct_magnitude and sum_steps > 0
        else float("nan")
    )
    val_ordinal_marker_loss = (
        sum_ordinal_marker_loss / sum_ordinal_steps
        if sum_ordinal_steps > 0
        else float("nan")
    )

    metrics = {
        "mark_acc": acc,
        "dt_mae": mae,
        "dt_rmse": rmse,
        "value_mae": value_mae,
        "qty_mae": qty_mae,
        "qty_rmse": qty_rmse,
        "qty_wape": qty_wape,
        "log_qty_mae": log_qty_mae,
        "log_qty_rmse": log_qty_rmse,
        "val_nll_time": val_nll_time,
        "val_nll_marker": val_nll_marker,
        "val_nll": val_nll_total,
        "val_ordinal_marker_loss": val_ordinal_marker_loss,
        "val_value_loss": val_value_loss if sum_steps > 0 else float("nan"),
        "val_magnitude_loss": val_magnitude_loss,
        "val_log_qty_aux_loss": val_log_qty_aux_loss,
        **_mark_metrics_from_confusion(mark_confusion),
        "_total": total,
        "_correct": correct,
        "_nll_steps": sum_steps,
    }
    if use_direct_raw_quantity:
        def percentile(values_list: list[float], q: float) -> float:
            if not values_list:
                return float("nan")
            return float(np.percentile(np.asarray(values_list, dtype=np.float64), q))

        metrics.update({
            "preclamp_negative_share": raw_affine_negative / max(raw_affine_count, 1),
            "magnitude_center_p01": percentile(magnitude_centers, 1),
            "magnitude_center_p50": percentile(magnitude_centers, 50),
            "magnitude_center_p95": percentile(magnitude_centers, 95),
            "magnitude_center_p99": percentile(magnitude_centers, 99),
            "magnitude_scale_p01": percentile(magnitude_scales, 1),
            "magnitude_scale_p50": percentile(magnitude_scales, 50),
            "magnitude_scale_p95": percentile(magnitude_scales, 95),
            "magnitude_scale_p99": percentile(magnitude_scales, 99),
            "magnitude_scale_floor_share": scale_floor_count / max(raw_affine_count, 1),
            "normalized_target_abs_p95": percentile(normalized_target_abs, 95),
            "normalized_target_abs_p99": percentile(normalized_target_abs, 99),
            "normalized_target_nonfinite_count": normalized_target_nonfinite,
        })
    for bucket_name, bucket in context_buckets.items():
        count = max(int(bucket["count"]), 1)
        metrics[f"context_{bucket_name}_count"] = int(bucket["count"])
        metrics[f"context_{bucket_name}_qty_mae"] = float(bucket["qty_abs"]) / count
        metrics[f"context_{bucket_name}_log_qty_mae"] = float(bucket["log_abs"]) / count
    return metrics


def _make_week_lookback_loaders(
    marked_df: pl.DataFrame,
    training_config: TrainingConfig,
    *,
    train_generator: torch.Generator | None = None,
):
    """
    Centralize loader construction so RMTPP and TitanTPP share the exact same
    train/validation split and padding behaviour.
    """
    train_ds = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_config.lookback,
        max_seq_len=training_config.max_seq_len,
        val_ratio=training_config.val_ratio,
        mode="train",
    )
    val_ds = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_config.lookback,
        max_seq_len=training_config.max_seq_len,
        val_ratio=training_config.val_ratio,
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_week_lookback,
        drop_last=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_week_lookback,
        num_workers=0,
    )
    return train_loader, val_loader


def make_week_lookback_loaders(
    marked_df: pl.DataFrame,
    training_config: TrainingConfig,
    *,
    train_generator: torch.Generator | None = None,
):
    """
    Public wrapper for notebook/analysis code that needs the exact same
    train/validation split used by the trainers.
    """
    return _make_week_lookback_loaders(
        marked_df,
        training_config,
        train_generator=train_generator,
    )


def make_fixed_split_week_lookback_loaders(
    marked_df: pl.DataFrame,
    training_config: TrainingConfig,
    *,
    split_col: str = "chronological_split",
    train_generator: torch.Generator | None = None,
):
    """
    Build train/validation/test loaders from a pre-split event table.

    Unlike the legacy `val_ratio` path, this uses the full chronological series
    as observed context and selects samples by the target event's split label.
    That keeps validation/test evaluation realistic: a validation target may
    condition on earlier train events, and a test target may condition on
    earlier train/validation/test observations.
    """
    train_ds = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_config.lookback,
        max_seq_len=training_config.max_seq_len,
        val_ratio=training_config.val_ratio,
        mode="all",
        split_col=split_col,
        target_splits={"train"},
    )
    val_ds = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_config.lookback,
        max_seq_len=training_config.max_seq_len,
        val_ratio=training_config.val_ratio,
        mode="all",
        split_col=split_col,
        target_splits={"validation"},
    )
    test_ds = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_config.lookback,
        max_seq_len=training_config.max_seq_len,
        val_ratio=training_config.val_ratio,
        mode="all",
        split_col=split_col,
        target_splits={"test"},
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_week_lookback,
        drop_last=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_week_lookback,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_week_lookback,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def _make_classic_next_event_loaders(marked_df: pl.DataFrame, training_config: TrainingConfig):
    """
    Classic event-count lookback loaders.

    Unlike the week-lookback path, `training_config.lookback` here means the
    number of past events fed into the model.
    """
    train_ds = RMTPPDataset(
        marked_df,
        lookback=training_config.lookback,
        val_ratio=training_config.val_ratio,
        mode="train",
    )
    val_ds = RMTPPDataset(
        marked_df,
        lookback=training_config.lookback,
        val_ratio=training_config.val_ratio,
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_next_event,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_next_event,
    )
    return train_loader, val_loader


def make_classic_next_event_loaders(marked_df: pl.DataFrame, training_config: TrainingConfig):
    """
    Public wrapper for notebooks that want the exact classic split used by the
    classic trainers.
    """
    return _make_classic_next_event_loaders(marked_df, training_config)


def _train_classic_next_event_model(model, train_loader: DataLoader, val_loader: DataLoader, training_config: TrainingConfig):
    """
    Shared trainer for the classic fixed-event-lookback next-event setup.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr)

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for x_mk, x_dt, y_mk, y_dt, _, _, y_val in train_loader:
            x_mk = x_mk.to(training_config.device)
            x_dt = x_dt.to(training_config.device)
            y_mk = y_mk.to(training_config.device)
            y_dt = y_dt.to(training_config.device).float()
            y_val = y_val.to(training_config.device).float() if y_val is not None else None

            h = model.forward(x_mk, x_dt)
            h_last = h[:, -1, :]

            marker_logits = model.mark_head(h_last)
            loss_marker = F.cross_entropy(marker_logits, y_mk)
            if getattr(model.cfg, "marker_loss_mode", "ce") == "ce_rps":
                ordinal_marker_loss = normalized_ranked_probability_score(
                    marker_logits,
                    y_mk,
                    num_real_marks=int(model.cfg.num_marks - 1),
                ).mean()
                loss_marker = (
                    loss_marker
                    + float(getattr(model.cfg, "lambda_ordinal", 0.0)) * ordinal_marker_loss
                )
            loss_time = -model.log_f_dt(h_last, y_dt).mean()

            if y_val is not None and getattr(model.cfg, "use_value_head", False):
                value_hat = predict_value_for_marks(model, h_last, y_mk)
                value_loss = F.huber_loss(value_hat, y_val, reduction="mean")
            else:
                value_loss = torch.zeros((), device=training_config.device, dtype=torch.float32)

            loss = (
                loss_marker
                + training_config.lambda_value * value_loss
                + training_config.lambda_dt * loss_time
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)

            optimizer.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        val_metrics = eval_next_event_classic(model, val_loader, training_config.device)

        score = (
            val_metrics["mark_acc"]
            - 0.01 * val_metrics["dt_mae"]
            - 0.001 * val_metrics["qty_mae"]
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "score": float(score),
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch:02d}] train_loss={train_loss:.8f} | "
            f"val_acc={val_metrics['mark_acc']:.8f} "
            f"val_dt_mae={val_metrics['dt_mae']:.8f} | val_dt_rmse={val_metrics['dt_rmse']:.8f} | "
            f"val_value_mae={val_metrics['value_mae']:.8f} | val_qty_mae={val_metrics['qty_mae']:.8f} | "
            f"val_nll_time={val_metrics['val_nll_time']:.6f} "
            f"val_nll_marker={val_metrics['val_nll_marker']:.6f} "
            f"val_value_loss={val_metrics['val_value_loss']:.6f} "
            f"val_nll={val_metrics['val_nll']:.6f} | "
            f"total={val_metrics['_total']} | correct={val_metrics['_correct']} | steps={val_metrics['_nll_steps']:.0f}"
        )

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {'best_score': best_val, 'history': history}


def _train_week_lookback_model(model, train_loader: DataLoader, val_loader: DataLoader, training_config: TrainingConfig):
    """
    Shared trainer for RMTPP-family models.

    The model is expected to expose:
    - nll(...)
    - mark_head
    - predict_value(...)
    - reconstruct_qty(...)
    - sample_next_dt(...)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr)

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for marks, dts, mask, _, values in train_loader:
            marks = marks.to(training_config.device)
            dts = dts.to(training_config.device)
            mask = mask.to(training_config.device)
            values = values.to(training_config.device) if values is not None else None

            out = model.nll(marks, dts, values=values, mask=mask)
            loss_mode = getattr(model.cfg, "loss_mode", "residual_only")
            marker_train_loss = out.get("marker_train_loss", out["nll_marker"])
            if "total_loss" in out:
                if getattr(model, "use_direct_magnitude", False):
                    loss = (
                        marker_train_loss
                        + training_config.lambda_dt * out["nll_time"]
                        + float(model.cfg.lambda_magnitude) * out["magnitude_loss"]
                        + float(model.cfg.lambda_qty) * out["qty_loss"]
                        + float(model.cfg.lambda_log_qty) * out["log_qty_aux_loss"]
                    )
                elif loss_mode == "residual_only":
                    loss = (
                        marker_train_loss
                        + training_config.lambda_value * out["value_loss"]
                        + training_config.lambda_dt * out["nll_time"]
                    )
                elif loss_mode == "hybrid":
                    loss = (
                        marker_train_loss
                        + training_config.lambda_value * out["value_loss"]
                        + training_config.lambda_dt * out["nll_time"]
                        + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
                    )
                elif loss_mode == "qty_only":
                    loss = (
                        marker_train_loss
                        + training_config.lambda_dt * out["nll_time"]
                        + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
                    )
                else:
                    raise ValueError(f"Unsupported loss_mode: {loss_mode}")
            else:
                loss = (
                    marker_train_loss
                    + training_config.lambda_value * out["value_loss"]
                    + training_config.lambda_dt * out["nll_time"]
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if training_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)

            optimizer.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        val_metrics = eval_next_event_week_lookback(model, val_loader, training_config.device)

        # Quantity reconstruction is now part of the objective, so it is folded
        # into model selection with a small penalty term.
        score = (
            val_metrics["mark_acc"]
            - 0.01 * val_metrics["dt_mae"]
            - 0.001 * val_metrics["qty_mae"]
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "score": float(score),
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch:02d}] train_loss={train_loss:.8f} | "
            f"val_acc={val_metrics['mark_acc']:.8f} "
            f"val_dt_mae={val_metrics['dt_mae']:.8f} | val_dt_rmse={val_metrics['dt_rmse']:.8f} | "
            f"val_value_mae={val_metrics['value_mae']:.8f} | val_qty_mae={val_metrics['qty_mae']:.8f} | "
            f"val_nll_time={val_metrics['val_nll_time']:.6f} "
            f"val_nll_marker={val_metrics['val_nll_marker']:.6f} "
            f"val_value_loss={val_metrics['val_value_loss']:.6f} "
            f"val_nll={val_metrics['val_nll']:.6f} | "
            f"total={val_metrics['_total']} | correct={val_metrics['_correct']} | steps={val_metrics['_nll_steps']:.0f}"
        )

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {'best_score': best_val, 'history': history}

def train_rmtpp(marked_df: pl.DataFrame, training_config: TrainingConfig, rmtpp_config: RMTPPConfig):
    train_loader, val_loader = _make_week_lookback_loaders(marked_df, training_config)
    model = RMTPP(rmtpp_config).to(training_config.device)
    return _train_week_lookback_model(model, train_loader, val_loader, training_config)


def train_titantpp(
    marked_df: pl.DataFrame,
    training_config: TrainingConfig,
    rmtpp_config: RMTPPConfig,
    titan_config: TitanConfig,
):
    """
    TitanTPP trainer that mirrors the RMTPP path exactly.

    This finishes the "TitanTPP 완성하기" direction at the training API layer:
    the model now shares the same dataset, value-head loss, and validation
    metrics as the vanilla RMTPP baseline.
    """
    train_loader, val_loader = _make_week_lookback_loaders(marked_df, training_config)
    model = TitanTPP(rmtpp_config, titan_config).to(training_config.device)
    return _train_week_lookback_model(model, train_loader, val_loader, training_config)


def train_transformer_hawkes_tpp(
    marked_df: pl.DataFrame,
    training_config: TrainingConfig,
    rmtpp_config: RMTPPConfig,
    thp_config: THPConfig,
):
    """
    THP baseline trainer using the same loader/objective as RMTPP/TitanTPP.

    This intentionally keeps the magnitude-factorized decoder identical across
    models, so the comparison focuses on the history encoder family:
    recurrent vs Transformer Hawkes vs Titan.
    """
    train_loader, val_loader = _make_week_lookback_loaders(marked_df, training_config)
    model = TransformerHawkesTPP(rmtpp_config, thp_config).to(training_config.device)
    return _train_week_lookback_model(model, train_loader, val_loader, training_config)


def train_rmtpp_classic(marked_df: pl.DataFrame, training_config: TrainingConfig, rmtpp_config: RMTPPConfig):
    train_loader, val_loader = _make_classic_next_event_loaders(marked_df, training_config)
    model = RMTPP(rmtpp_config).to(training_config.device)
    return _train_classic_next_event_model(model, train_loader, val_loader, training_config)


def train_titantpp_classic(
    marked_df: pl.DataFrame,
    training_config: TrainingConfig,
    rmtpp_config: RMTPPConfig,
    titan_config: TitanConfig,
):
    """
    Classic event-count-lookback trainer for TitanTPP.
    """
    train_loader, val_loader = _make_classic_next_event_loaders(marked_df, training_config)
    model = TitanTPP(rmtpp_config, titan_config).to(training_config.device)
    return _train_classic_next_event_model(model, train_loader, val_loader, training_config)


def train_transformer_hawkes_tpp_classic(
    marked_df: pl.DataFrame,
    training_config: TrainingConfig,
    rmtpp_config: RMTPPConfig,
    thp_config: THPConfig,
):
    """
    Classic fixed-event-lookback trainer for the THP baseline.
    """
    train_loader, val_loader = _make_classic_next_event_loaders(marked_df, training_config)
    model = TransformerHawkesTPP(rmtpp_config, thp_config).to(training_config.device)
    return _train_classic_next_event_model(model, train_loader, val_loader, training_config)
