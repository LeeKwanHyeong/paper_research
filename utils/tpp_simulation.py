from __future__ import annotations

from typing import Literal

import numpy as np
import torch


@torch.no_grad()
def simulate_horizon_week_grid(
    model,
    *,
    history_marks: torch.Tensor,
    history_dts: torch.Tensor,
    horizon_weeks: int = 13,
    n_sims: int = 100,
    sample_mark: bool = False,
    qty_cap: float | None = None,
    device: str | None = None,
):
    """
    Simulate a weekly demand grid directly from (mark, residual, dt) predictions.

    Compared with the old notebook logic, this path does not use rep_qty tables.
    Each sampled event reconstructs quantity via:
        qty = 10 ** (mark + residual)

    Args:
        history_marks:
            [L] or [1, L] integer mark history.
        history_dts:
            [L] or [1, L] inter-event time history in "week" units.
        horizon_weeks:
            Number of future grid cells to accumulate.
        n_sims:
            Number of Monte Carlo rollouts.
        sample_mark:
            False -> argmax mark prediction, True -> categorical sampling.
        qty_cap:
            Optional upper bound to prevent rare reconstructed values from exploding.
    """
    if device is None:
        device = next(model.parameters()).device

    marks = history_marks.to(device)
    dts = history_dts.to(device)
    if marks.dim() == 1:
        marks = marks.unsqueeze(0)
    if dts.dim() == 1:
        dts = dts.unsqueeze(0)

    if marks.size(0) != 1 or dts.size(0) != 1:
        raise ValueError("history_marks and history_dts must describe a single sequence.")

    all_grids = torch.zeros((n_sims, horizon_weeks), dtype=torch.float32, device=device)

    for sim_idx in range(n_sims):
        cur_marks = marks.clone()
        cur_dts = dts.clone()
        grid = torch.zeros((horizon_weeks,), dtype=torch.float32, device=device)
        elapsed = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Keep generating future events until the next event falls beyond horizon.
        for _ in range(max(horizon_weeks * 8, 32)):
            h = model.forward(cur_marks, cur_dts)
            h_last = h[:, -1, :]

            # The final class is reserved for PAD during padded training.
            logits = model.mark_head(h_last)[..., : model.cfg.num_marks - 1]
            prob = torch.softmax(logits, dim=-1)
            if sample_mark:
                mk_next = torch.multinomial(prob.squeeze(0), num_samples=1)
            else:
                mk_next = torch.argmax(prob, dim=-1)

            dt_next = model.sample_next_dt(h_last).clamp_min(1e-6)
            val_next = model.predict_value(h_last)
            qty_next = model.reconstruct_qty(mk_next, val_next)

            if qty_cap is not None:
                qty_next = qty_next.clamp(max=float(qty_cap))

            elapsed = elapsed + dt_next.squeeze(0)

            # Weekly bucket indexing:
            # - dt=1.0 -> week index 0
            # - dt in (1, 2] -> week index 1
            bucket = int(torch.ceil(elapsed).item()) - 1
            if bucket >= horizon_weeks:
                break
            if bucket >= 0:
                grid[bucket] += qty_next.squeeze(0)

            # Append the sampled event so subsequent events condition on it.
            cur_marks = torch.cat([cur_marks, mk_next.view(1, 1)], dim=1)
            cur_dts = torch.cat([cur_dts, dt_next.view(1, 1)], dim=1)

        all_grids[sim_idx] = grid

    mean_grid = all_grids.mean(dim=0)
    return mean_grid.detach().cpu().numpy(), all_grids.detach().cpu().numpy()


def grid_to_int_list(mean_grid: np.ndarray, rounding: Literal["round", "floor", "ceil"] = "round") -> list[int]:
    """
    Convert a float forecast grid into integer demand quantities.
    """
    if rounding == "round":
        out = np.rint(mean_grid)
    elif rounding == "floor":
        out = np.floor(mean_grid)
    elif rounding == "ceil":
        out = np.ceil(mean_grid)
    else:
        raise ValueError(f"Unsupported rounding: {rounding}")
    return out.astype(int).tolist()
