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
import numpy as np
import polars as pl

from models.RMTPPs.config import RMTPPConfig, THPConfig
from models.Titan import TitanConfig


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

            sum_nll_marker += float((-log_y).sum().item())
            sum_nll_time += float((-logf_dt).sum().item())
            sum_nll_total += float(((-log_y) + (-logf_dt)).sum().item())

            logits = logits[..., : model.cfg.num_marks - 1]
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y_mk).sum().item()
            total += y_mk.numel()

            u = torch.full_like(y_dt, 0.5, device=device).clamp_min(model.cfg.eps)
            dt_hat = model.sample_next_dt(h_last, u=u).clamp_min(1.0)

            err = (dt_hat - y_dt).abs()
            dt_abs += err.sum().item()
            dt_sq += ((dt_hat - y_dt) ** 2).sum().item()

            if y_val is not None and getattr(model.cfg, "use_value_head", False):
                value_hat = model.predict_value(h_last)
                value_loss = F.huber_loss(value_hat, y_val, reduction="none")
                sum_value_loss += float(value_loss.sum().item())
                value_abs += (value_hat - y_val).abs().sum().item()

                qty_hat = model.reconstruct_qty(pred, value_hat)
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
        "val_value_loss": sum_value_loss / max(total, 1),
        "_total": total,
        "_correct": correct,
        "_nll_steps": total,
    }

def eval_next_event_week_lookback(model, loader: DataLoader, device: str) -> Dict[str, float]:
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

    sum_nll_time = 0.0
    sum_nll_marker = 0.0
    sum_nll_total = 0.0
    sum_value_loss = 0.0
    sum_steps = 0.0

    pad_id = int(model.cfg.num_marks - 1)

    with torch.no_grad():
        for marks, dts, mask, _, values in loader:
            marks = marks.to(device)  # (B, Lmax)
            dts   = dts.to(device)    # (B, Lmax)
            mask  = mask.to(device)   # (B, Lmax)
            values = values.to(device) if values is not None else None

            # --------------------- NLL -----------------------
            out = model.nll(marks, dts, values=values, mask=mask)
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
            sum_steps += steps

            # --------------------- Marker -----------------------
            valid = mask[:, -1] & mask[:, -2]
            if valid.sum().item() == 0:
                continue

            h = model.forward(marks, dts)     # (B, Lmax, H)
            h_prev = h[:, -2, :]              # (B, H)
            y_mk   = marks[:, -1]             # (B,)
            y_dt   = dts[:, -1].float()       # (B,)
            y_val  = values[:, -1].float() if values is not None else None

            # valid 필터 + 혹시 모를 PAD target 제거
            valid = valid & (y_mk != pad_id)
            if valid.sum().item() == 0:
                continue

            h_prev = h_prev[valid]
            y_mk   = y_mk[valid]
            y_dt   = y_dt[valid]
            if y_val is not None:
                y_val = y_val[valid]

            # ---- mark acc ----
            logits = model.mark_head(h_prev)      # (Bv, K_with_pad)
            # The final class is reserved for PAD, so exclude it from inference.
            logits = logits[..., :pad_id]
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y_mk).sum().item()
            total += y_mk.numel()

            # ---- dt point estimate: median(u=0.5) ----
            u = torch.full((y_dt.size(0),), 0.5, device=device).clamp_min(model.cfg.eps)
            dt_hat = model.sample_next_dt(h_prev, u=u).clamp_min(1.0)

            err = (dt_hat - y_dt).abs()
            dt_abs += err.sum().item()
            dt_sq  += ((dt_hat - y_dt) ** 2).sum().item()

            if y_val is not None and getattr(model.cfg, "use_value_head", False):
                value_hat = model.predict_value(h_prev)
                value_abs += (value_hat - y_val).abs().sum().item()

                qty_hat = model.reconstruct_qty(pred, value_hat)
                qty_true = model.reconstruct_qty(y_mk, y_val)
                qty_abs += (qty_hat - qty_true).abs().sum().item()

    acc = correct / max(total, 1)
    mae = dt_abs / max(total, 1)
    rmse = float(np.sqrt(dt_sq / max(total, 1)))
    value_mae = value_abs / max(total, 1)
    qty_mae = qty_abs / max(total, 1)

    # step-weighted mean nll
    if sum_steps <= 0:
        val_nll_time = float("nan")
        val_nll_marker = float("nan")
        val_nll_total = float("nan")
    else:
        val_nll_time = sum_nll_time / sum_steps
        val_nll_marker = sum_nll_marker / sum_steps
        val_nll_total = sum_nll_total / sum_steps
        val_value_loss = sum_value_loss / sum_steps

    return {
        "mark_acc": acc,
        "dt_mae": mae,
        "dt_rmse": rmse,
        "value_mae": value_mae,
        "qty_mae": qty_mae,
        "val_nll_time": val_nll_time,
        "val_nll_marker": val_nll_marker,
        "val_nll": val_nll_total,
        "val_value_loss": val_value_loss if sum_steps > 0 else float("nan"),
        "_total": total,
        "_correct": correct,
        "_nll_steps": sum_steps,
    }


def _make_week_lookback_loaders(marked_df: pl.DataFrame, training_config: TrainingConfig):
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_week_lookback,
    )
    return train_loader, val_loader


def make_week_lookback_loaders(marked_df: pl.DataFrame, training_config: TrainingConfig):
    """
    Public wrapper for notebook/analysis code that needs the exact same
    train/validation split used by the trainers.
    """
    return _make_week_lookback_loaders(marked_df, training_config)


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

            loss_marker = F.cross_entropy(model.mark_head(h_last), y_mk)
            loss_time = -model.log_f_dt(h_last, y_dt).mean()

            if y_val is not None and getattr(model.cfg, "use_value_head", False):
                value_hat = model.predict_value(h_last)
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
            if "total_loss" in out:
                if loss_mode == "residual_only":
                    loss = (
                        out["nll_marker"]
                        + training_config.lambda_value * out["value_loss"]
                        + training_config.lambda_dt * out["nll_time"]
                    )
                elif loss_mode == "hybrid":
                    loss = (
                        out["nll_marker"]
                        + training_config.lambda_value * out["value_loss"]
                        + training_config.lambda_dt * out["nll_time"]
                        + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
                    )
                elif loss_mode == "qty_only":
                    loss = (
                        out["nll_marker"]
                        + training_config.lambda_dt * out["nll_time"]
                        + getattr(model.cfg, "lambda_qty", 0.25) * out["qty_loss"]
                    )
                else:
                    raise ValueError(f"Unsupported loss_mode: {loss_mode}")
            else:
                loss = (
                    out["nll_marker"]
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
