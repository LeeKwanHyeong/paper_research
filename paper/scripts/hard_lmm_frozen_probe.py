"""Isolated, causal readout probes; the historical Hard-LMM is never trained."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch


def require_finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite {name}")


class FrozenResidualProbe(nn.Module):
    """Only this MLP is optimized. Labels are deliberately absent from its API."""

    def __init__(self, feature_dim: int, candidate: str):
        super().__init__()
        if candidate not in ("calibration", "shrinkage"):
            raise ValueError(f"Unknown independent probe: {candidate}")
        self.candidate = candidate
        self.network = nn.Sequential(nn.Linear(feature_dim, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, base_logit, residual_projection):
        score = self.network(features).squeeze(-1)
        gate = torch.ones_like(score)
        if self.candidate == "calibration":
            correction = 0.05 * score.tanh()
        else:
            gate = 1.0 - 0.2 * score.clamp(0.0, 1.0)
            correction = (gate - 1.0) * residual_projection
        return base_logit + correction, gate, correction


@torch.no_grad()
def extract_features(model, dts, mask, quantities):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Base must be frozen and in eval mode")
    if model.lmm is None or model.lmm.topk != 4 or model.lmm.mem_size < 5:
        raise ValueError("Expected original static top-4 prototype memory")
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    rows = torch.arange(len(lengths), device=dts.device)
    target, previous = lengths - 1, lengths - 2
    history_qty = quantities.masked_fill(~mask, 0).clone()
    history_dt = dts.masked_fill(~mask, 0).clone()
    history_qty[rows, target] = 0
    history_dt[rows, target] = 0
    write_mask = mask.clone()
    write_mask[rows, target] = False
    local = model._encode_base(history_dt, history_qty, mask, memory_write_mask=write_mask)
    residual, _ = model.lmm.retrieve(local)
    h, r = local[rows, previous], residual[rows, previous]
    combined = h + r
    z = model.quantity_head(combined).squeeze(-1)
    projection = (r * model.quantity_head.weight.squeeze(0)).sum(-1)
    scores = F.normalize(h, dim=-1) @ F.normalize(model.lmm.mem[0], dim=-1).T
    top5 = scores.topk(5, dim=-1).values
    top4 = top5[:, :4]
    log_prob = top4.log_softmax(dim=-1)
    entropy = -(log_prob.exp() * log_prob).sum(-1)
    history = lengths - 1
    statistics = torch.stack((
        top4[:, 0], top4.mean(-1), top4.std(-1, unbiased=False),
        top5[:, 3] - top5[:, 4], entropy,
        h.norm(dim=-1).log1p().tanh(), r.norm(dim=-1).log1p().tanh(),
        z.tanh(), projection.tanh(), history.float().log1p().tanh(),
    ), dim=-1)
    result = {
        "features": torch.cat((F.normalize(h, dim=-1), F.normalize(r, dim=-1), statistics), -1),
        "z": z, "projection": projection, "history_length": history,
        # These labels never enter the feature tensor or probe.forward.
        "quantity": quantities[rows, target],
        "time_nll": -model.log_f_dt(combined, dts[rows, target]),
    }
    for name, value in result.items():
        require_finite(name, value)
    return {name: value.detach().cpu() for name, value in result.items()}


def sample_indices(count: int, limit: int, seed: int = 42):
    if count < 1 or limit < 1:
        raise ValueError("Nonempty train sample required")
    if count <= limit:
        return torch.arange(count)
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed))[:limit].sort().values


@torch.no_grad()
def predict(probe, cache, batch_size=8192):
    outputs = [[], [], []]
    for start in range(0, len(cache["z"]), batch_size):
        sl = slice(start, start + batch_size)
        batch = probe(cache["features"][sl], cache["z"][sl], cache["projection"][sl])
        for i, tensor in enumerate(batch):
            require_finite("probe output", tensor)
            outputs[i].append(tensor)
    return tuple(torch.cat(tensors) for tensors in outputs)


def metric_values(z, quantity, time_nll):
    require_finite("logits", z)
    log_quantity = F.softplus(z)
    prediction = log_quantity.expm1()
    require_finite("raw quantity prediction", prediction)
    error = prediction.double() - quantity.double()
    mse = (log_quantity - quantity.clamp_min(0).log1p()).square().double().mean().item()
    time = time_nll.double().mean().item()
    result = {"qty_mae": error.abs().mean().item(), "qty_rmse": error.square().mean().sqrt().item(),
              "time_nll": time, "log_qty_mse": mse, "joint_objective": time + mse}
    if not all(math.isfinite(x) for x in result.values()):
        raise FloatingPointError("Non-finite metrics")
    return result


def fit_probe(candidate, train, validation, policy, progress):
    torch.manual_seed(42)
    probe = FrozenResidualProbe(train["features"].shape[-1], candidate)
    optimizer = torch.optim.Adam(probe.parameters(), lr=policy["learning_rate"], weight_decay=0)
    initial, _, _ = predict(probe, validation)
    if not torch.equal(initial, validation["z"]):
        raise AssertionError("Probe initialization is not exactly identity")
    best_metrics = metric_values(initial, validation["quantity"], validation["time_nll"])
    history = [{"epoch": 0, **best_metrics}]
    best_state, best_epoch, stale = copy.deepcopy(probe.state_dict()), 0, 0
    generator = torch.Generator().manual_seed(policy["shuffle_seed"])
    first_gradient_norm = None
    for epoch in range(1, policy["maximum_epochs"] + 1):
        started = time.monotonic()
        permutation = torch.randperm(len(train["z"]), generator=generator)
        train_sum = 0.0
        for indices in permutation.split(policy["batch_size"]):
            z, _, _ = probe(train["features"][indices], train["z"][indices], train["projection"][indices])
            loss = (F.softplus(z) - train["quantity"][indices].clamp_min(0).log1p()).square().mean()
            joint = loss + train["time_nll"][indices].mean()
            require_finite("train objective", joint)
            optimizer.zero_grad(set_to_none=True)
            joint.backward()
            for parameter in probe.parameters():
                if parameter.grad is not None:
                    require_finite("adapter gradient", parameter.grad)
            norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), policy["gradient_clip"], error_if_nonfinite=True)
            if first_gradient_norm is None:
                first_gradient_norm = float(norm)
                if first_gradient_norm == 0:
                    raise AssertionError("Adapter has no gradient at identity")
            optimizer.step()
            for parameter in probe.parameters():
                require_finite("adapter parameter", parameter)
            train_sum += float(joint.detach()) * len(indices)
        z, _, _ = predict(probe, validation)
        metrics = metric_values(z, validation["quantity"], validation["time_nll"])
        if metrics["joint_objective"] < best_metrics["joint_objective"]:
            best_state, best_epoch, best_metrics, stale = copy.deepcopy(probe.state_dict()), epoch, metrics, 0
        else:
            stale += 1
        row = {"epoch": epoch, "train_joint_objective": train_sum / len(train["z"]),
               "best_epoch": best_epoch, "epoch_seconds": time.monotonic() - started, **metrics}
        history.append(row)
        progress(row)
        if epoch >= policy["minimum_epochs"] and stale >= policy["patience"]:
            break
    probe.load_state_dict(best_state)
    return probe.eval(), history, {"best_epoch": best_epoch, "completed_epochs": epoch,
        "selection": "identity_fallback" if best_epoch == 0 else "trained_adapter",
        "first_gradient_norm": first_gradient_norm, "trainable_parameters": sum(p.numel() for p in probe.parameters())}


def scope_masks(cache, boundaries):
    q, h = cache["quantity"], cache["history_length"]
    p50, p90, p95, p99 = boundaries
    return {"overall": torch.ones_like(q, dtype=torch.bool),
        "le_p50": q <= p50, "p50_p90": (q > p50) & (q <= p90),
        "p90_p95": (q > p90) & (q <= p95), "p95_p99": (q > p95) & (q <= p99),
        "gt_p99": q > p99, "body_le_p95": q <= p95, "tail_gt_p95": q > p95,
        "history_le_64": h <= 64, "history_65_128": (h > 64) & (h <= 128),
        "history_gt_128": h > 128}


def summarize(cache, z, gate, correction, boundaries):
    result = {}
    for name, mask in scope_masks(cache, boundaries).items():
        count = int(mask.sum())
        if count == 0:
            result[name] = {"count": 0, "status": "empty"}
            continue
        base = metric_values(cache["z"][mask], cache["quantity"][mask], cache["time_nll"][mask])
        metrics = metric_values(z[mask], cache["quantity"][mask], cache["time_nll"][mask])
        result[name] = {"count": count, "status": "evaluated", "baseline": base, "candidate": metrics,
            "mae_delta": metrics["qty_mae"] - base["qty_mae"],
            "rmse_delta": metrics["qty_rmse"] - base["qty_rmse"],
            "gate_mean": gate[mask].double().mean().item(),
            "gate_p05": torch.quantile(gate[mask], 0.05).item(),
            "correction_abs_p95": torch.quantile(correction[mask].abs(), 0.95).item()}
    return result


def acceptance(scopes, selected_epoch, full_validation):
    if not full_validation:
        return {"status": "not_assessed_smoke", "eligible_for_fresh_training": False}
    if any(scopes[k]["status"] != "evaluated" for k in ("overall", "body_le_p95", "gt_p99")):
        return {"status": "not_assessable_empty_stratum", "eligible_for_fresh_training": False}
    def relative(scope, metric):
        row = scopes[scope]
        base, candidate = row["baseline"][metric], row["candidate"][metric]
        return (candidate - base) / base if base != 0 else (0.0 if candidate == 0 else None)
    body, rmse, tail = relative("body_le_p95", "qty_mae"), relative("overall", "qty_rmse"), relative("gt_p99", "qty_mae")
    time_delta = scopes["overall"]["candidate"]["time_nll"] - scopes["overall"]["baseline"]["time_nll"]
    checks = {"trained_checkpoint_selected": selected_epoch > 0,
        "body_improved_at_least_5pct": body is not None and body <= -0.05,
        "rmse_regression_at_most_2pct": rmse is not None and rmse <= 0.02,
        "p99_regression_at_most_2pct": tail is not None and tail <= 0.02,
        "time_nll_regression_at_most_001": time_delta <= 0.01}
    return {"status": "exploratory_pass" if all(checks.values()) else "exploratory_fail", "checks": checks,
        "body_relative_change": body, "rmse_relative_change": rmse, "p99_relative_change": tail,
        "eligible_for_fresh_training": False, "fresh_training_requires_separate_authorization": True}


def write_event_deltas(path: Path, cache, z, gate, correction):
    base = F.softplus(cache["z"]).expm1().double()
    prediction = F.softplus(z).expm1().double()
    q = cache["quantity"].double()
    pl.DataFrame({
        **{key: cache[key].numpy() for key in ("target_index", "series_index", "context_end", "history_length", "quantity")},
        "baseline_prediction": base.numpy(), "candidate_prediction": prediction.numpy(),
        "absolute_error_delta": ((prediction - q).abs() - (base - q).abs()).numpy(),
        "squared_error_delta": ((prediction - q).square() - (base - q).square()).numpy(),
        "gate": gate.numpy(), "logit_correction": correction.numpy(),
    }).write_parquet(path)
