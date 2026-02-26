from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader.event_seq_data_module import time_split_events, RMTPPDataset, collate_next_event, \
    RMTPPWeekLookbackDataset, collate_week_lookback
from models.RMTPPs.RMTPP import RMTPP
import numpy as np
import polars as pl

from models.RMTPPs.config import RMTPPConfig


@dataclass
class TrainingConfig:
    lookback: int = 30
    max_seq_len: int = 64
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 20
    val_ratio: float = 0.2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lambda_dt: float = 1.0 # dt loss weight
    grad_clip: float = 1.0


def eval_next_event(model: RMTPP, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    dt_abs = 0
    dt_sq = 0.0

    with torch.no_grad():
        for x_mk, x_dt, y_mk, y_dt, _ in loader:
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

def eval_next_event_weeklookback(model, loader: DataLoader, device: str) -> Dict[str, float]:
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

    pad_id = int(model.cfg.num_marks - 1)

    with torch.no_grad():
        for marks, dts, mask, _ in loader:
            marks = marks.to(device)  # (B, Lmax)
            dts   = dts.to(device)    # (B, Lmax)
            mask  = mask.to(device)   # (B, Lmax)

            # left-pad에서는 target은 항상 끝(-1), prev는 -2
            # 단, 유효 토큰이 2개 이상이어야 함
            valid = mask[:, -1] & mask[:, -2]
            if valid.sum().item() == 0:
                continue

            h = model.forward(marks, dts)     # (B, Lmax, H)
            h_prev = h[:, -2, :]              # (B, H)
            y_mk   = marks[:, -1]             # (B,)
            y_dt   = dts[:, -1].float()       # (B,)

            # valid 필터 + 혹시 모를 PAD target 제거
            valid = valid & (y_mk != pad_id)
            if valid.sum().item() == 0:
                continue

            h_prev = h_prev[valid]
            y_mk   = y_mk[valid]
            y_dt   = y_dt[valid]

            # ---- mark acc ----
            logits = model.mark_head(h_prev)      # (Bv, K_with_pad)
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == y_mk).sum().item()
            total += y_mk.numel()

            # ---- dt point estimate: median(u=0.5) ----
            u = torch.full((y_dt.size(0),), 0.5, device=device).clamp_min(model.cfg.eps)
            dt_hat = model.sample_next_dt(h_prev, u=u).clamp_min(1.0)

            err = (dt_hat - y_dt).abs()
            dt_abs += err.sum().item()
            dt_sq  += ((dt_hat - y_dt) ** 2).sum().item()

    acc = correct / max(total, 1)
    mae = dt_abs / max(total, 1)
    rmse = float(np.sqrt(dt_sq / max(total, 1)))
    return {"mark_acc": acc, "dt_mae": mae, "dt_rmse": rmse, "_total": total, "_correct": correct}

def train_rmtpp(marked_df: pl.DataFrame, training_config: TrainingConfig, rmtpp_config: RMTPPConfig):
    # train_ds = RMTPPDataset(marked_df, lookback=training_config.lookback, val_ratio=training_config.val_ratio,
    #                         mode="train")
    # val_ds = RMTPPDataset(marked_df, lookback=training_config.lookback, val_ratio=training_config.val_ratio, mode="val")

    # print("train_ds len:", len(train_ds), "val_ds len:", len(val_ds))
    #
    # train_loader = DataLoader(train_ds, batch_size = training_config.batch_size, shuffle = True, num_workers = 0, collate_fn = collate_next_event, drop_last = True)
    # val_loader = DataLoader(val_ds, batch_size = training_config.batch_size, shuffle = False, num_workers = 0, collate_fn = collate_next_event)

    train_ds = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_config.lookback,  # 새 파라미터
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

    train_loader = DataLoader(train_ds, batch_size = training_config.batch_size, shuffle=True, collate_fn = collate_week_lookback, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size = training_config.batch_size, shuffle=False, collate_fn = collate_week_lookback)

    model = RMTPP(rmtpp_config).to(training_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = training_config.lr)

    best_val = -1.0
    best_state = None

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        # for x_mk, x_dt, y_mk, y_dt, _ in train_loader:
        #     x_mk = x_mk.to(training_config.device)
        #     x_dt = x_dt.to(training_config.device)
        #
        #     out = model.nll(x_mk, x_dt)
        for marks, dts, mask, _ in train_loader:
            marks = marks.to(training_config.device)
            dts = dts.to(training_config.device)
            mask = mask.to(training_config.device)

            out = model.nll(marks, dts, mask = mask)
            loss = out["nll_marker"] + training_config.lambda_dt * out["nll_time"]

            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            if training_config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)

            optimizer.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        # val_metrics = eval_next_event(model, val_loader, training_config.device)
        val_metrics = eval_next_event_weeklookback(model, val_loader, training_config.device)

        score = val_metrics['mark_acc'] - 0.01 * val_metrics['dt_mae']

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.8f} | "
              f"val_acc={val_metrics['mark_acc']:.8f} val_dt_mae={val_metrics['dt_mae']:.8f} | val_dt_rmse={val_metrics['dt_rmse']:.8f} "
              f"| total={val_metrics['_total']} | correct={val_metrics['_correct']}")

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {'best_score': best_val}

