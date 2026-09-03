"""Read-only post-stop audit: histories, strata, and bounded train gradients.

No optimizer, checkpoint selection change, validation inference, or server use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/weighted-mpl")

import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset, collate_week_lookback
from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, right_pad_batch, target_outputs
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

RAW = ROOT / "search_artifacts/hard_lmm_weighted_static_seed42_20260903"
OUT = ROOT / "paper/results/hard_lmm_weighted_static_20260903/mechanism_audit.json"
CANDIDATE = "titantpp_weighted_static_memory"
VARIANT = "count_only_log_regression"
DATASETS = ("yellow_trip_hourly", "raf_spare_parts")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_path(root, model):
    return root / "runs" / model / VARIANT / "seed_42"


def history_audit(path):
    history = read(path / "history.json")["history"]
    selected = min(history, key=lambda r: r["val_joint_objective"])
    summary = read(path / "summary.json")
    assert selected["epoch"] == summary["best_epoch"]
    for row in history:
        assert abs(row["val_joint_objective"] - row["val_time_nll"] - row["val_log_qty_mse"]) < 1e-6
    keys = ("epoch", "val_joint_objective", "val_time_nll", "val_log_qty_mse", "val_qty_mae", "val_qty_rmse")
    views = {}
    for name, row in {
        "official_joint": selected,
        "posthoc_min_mae": min(history, key=lambda r: r["val_qty_mae"]),
        "posthoc_min_log_mse": min(history, key=lambda r: r["val_log_qty_mse"]),
        "last": history[-1],
    }.items():
        views[name] = {key: row[key] for key in keys}
        views[name]["delta_joint_time_component"] = row["val_time_nll"] - selected["val_time_nll"]
        views[name]["delta_joint_quantity_component"] = row["val_log_qty_mse"] - selected["val_log_qty_mse"]
    clips = [r["train_gradient_clip_fraction"] for r in history if "train_gradient_clip_fraction" in r]
    return {"epochs": len(history), "views": views,
            "mean_train_clipping_fraction": sum(clips) / len(clips) if clips else None,
            "epochs_beating_official_mae_and_rmse": sum(
                r["val_qty_mae"] < selected["val_qty_mae"] and r["val_qty_rmse"] < selected["val_qty_rmse"] for r in history),
            "posthoc_views_are_not_new_official_results": True}


def strata_audit(baseline, candidate):
    def rows(root, model):
        return {r["stratum"]: r for r in pl.read_csv(root / "quantity_seed_metrics.csv").filter(
            (pl.col("backbone") == model) & (pl.col("seed") == 42) & (pl.col("variant") == VARIANT)).to_dicts()}
    old, new = rows(baseline, "titantpp"), rows(candidate, CANDIDATE)
    assert set(old) == set(new)
    result = []
    for name in old:
        a, b = old[name], new[name]
        assert a["count"] == b["count"] and a["stratum_label"] == b["stratum_label"]
        n = a["count"]
        result.append({"stratum": name, "count": n,
                       "baseline_mae": a["qty_mae"], "candidate_mae": b["qty_mae"],
                       "baseline_rmse": a["qty_rmse"], "candidate_rmse": b["qty_rmse"],
                       "baseline_bias": a["qty_bias"], "candidate_bias": b["qty_bias"],
                       "absolute_error_reduction_sum": n * (a["qty_mae"] - b["qty_mae"]),
                       "squared_error_reduction_sum": n * (a["qty_rmse"] ** 2 - b["qty_rmse"] ** 2)})
    total = sum(r["count"] for r in result)
    for side, root, model in (("baseline", baseline, "titantpp"), ("candidate", candidate, CANDIDATE)):
        summary = read(run_path(root, model) / "summary.json")
        assert abs(sum(r["count"] * r[f"{side}_mae"] for r in result) / total - summary["best_val_qty_mae"]) < 1e-6
        assert abs(math.sqrt(sum(r["count"] * r[f"{side}_rmse"] ** 2 for r in result) / total) - summary["best_val_qty_rmse"]) < 1e-6
    return result


def describe(values):
    values = values.detach().double().flatten()
    assert values.numel() and torch.isfinite(values).all()
    return {"mean": values.mean().item(), "p50": values.median().item(),
            "p95": values.quantile(.95).item(), "min": values.min().item(), "max": values.max().item()}


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return (a @ b / denominator).item() if denominator > 0 else None


def restore(path, backbone):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["backbone"] == backbone and payload["seed"] == 42
    assert payload["evaluation_scope"] == "validation_only" and not payload["held_out_test_evaluated"]
    interface, encoder = payload["interface_meta"], payload["encoder_config"]
    assert interface["quantity_loss"] == "mse_on_log1p_quantity"
    time_contract = interface["time_head"]
    assert time_contract["mode"] == "legacy_clamped_rmtpp"
    model, _ = build_count_aware_model(backbone, hidden_dim=encoder["d_model"],
        train_log_mean=interface["train_target_mean"], train_log_std=interface["train_target_std"],
        max_seq_len=encoder["max_len"], quantity_variant=VARIANT,
        time_scale=time_contract["time_scale"], time_w_max=time_contract["time_w_max"])
    state = payload["model_state_dict"]
    if "model_state_sha256" in payload:
        assert canonical_state_dict_sha256(state) == payload["model_state_sha256"]
    model.load_state_dict(state, strict=True)
    return model.eval(), payload


def train_probe(path, backbone, dataset, indices, p95):
    model, payload = restore(path, backbone)
    before = canonical_state_dict_sha256(model.state_dict())
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=64, shuffle=False,
                        collate_fn=collate_week_lookback, num_workers=0)
    arrays, gradient_rows = {}, []
    parameters = [(n, p) for n, p in model.named_parameters() if n.startswith(("encoder.", "lmm."))]
    assert parameters[-1][0] == "lmm.mem"
    memory_start = sum(p.numel() for n, p in parameters if n.startswith("encoder."))
    for batch_number, (_, dts, mask, _, quantities) in enumerate(loader):
        dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
        rows, target, previous = torch.arange(len(lengths)), lengths - 1, lengths - 2
        q, dt = quantities[rows, target].clone(), dts[rows, target].clone()
        history_q, history_dt = quantities.clone(), dts.clone()
        history_q[rows, target] = 0
        history_dt[rows, target] = 0
        write_mask = mask.clone()
        write_mask[rows, target] = False
        local = model._encode_base(history_dt, history_q, mask, memory_write_mask=write_mask)[rows, previous]
        scores = F.normalize(local, dim=-1) @ F.normalize(model.lmm.mem[0], dim=-1).T
        top, idx = scores.topk(4, dim=-1)
        weights = top.softmax(-1)
        selected = model.lmm.mem[0][idx]
        uniform = selected.mean(1)
        weighted = (selected * weights.unsqueeze(-1)).sum(1)
        residual = weighted if backbone == CANDIDATE else uniform
        hidden = local + residual
        logq, pred = model.predict_quantity(hidden)
        time_loss = -model.log_f_dt(hidden, dt)
        log_mse = (logq - q.log1p()).square()
        body = q <= p95
        assert body.any()
        quantity_projection_delta = ((weighted - uniform) * model.quantity_head.weight[0]).sum(-1)
        batch_values = {
            "quantity": q, "is_body": body, "history_length": lengths - 1,
            "score_span": top[:, 0] - top[:, -1], "max_weight": weights.max(-1).values,
            "normalized_entropy": -(weights * weights.log()).sum(-1) / math.log(4),
            "residual_relative_change": (weighted - uniform).norm(dim=-1) / uniform.norm(dim=-1).clamp_min(1e-12),
            "quantity_logit_change_abs": quantity_projection_delta.abs(),
            "quantity_relative_change_abs": quantity_projection_delta.expm1().abs(),
            "prototype_indices": idx,
        }
        with torch.no_grad():
            _, uniform_pred = model.predict_quantity(local + uniform)
            _, weighted_pred = model.predict_quantity(local + weighted)
            batch_values["weighted_vs_uniform_absolute_error_delta"] = (weighted_pred - q).abs() - (uniform_pred - q).abs()
            batch_values["time_nll_memory_minus_local"] = time_loss + model.log_f_dt(local, dt)
            if batch_number == 0:
                official = target_outputs(model, dts, mask, quantities, lambda_log_qty=1)
                torch.testing.assert_close(pred, official["pred_qty"], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(time_loss, official["time_loss"], rtol=1e-5, atol=1e-5)
        for name, value in batch_values.items():
            arrays.setdefault(name, []).append(value.detach())
        gradients = {}
        losses = {"time": time_loss.mean(), "log_quantity": log_mse.mean(), "body_mae": (pred[body] - q[body]).abs().mean()}
        for i, (name, loss) in enumerate(losses.items()):
            parts = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=i < 2, allow_unused=True)
            gradients[name] = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for (_, p), g in zip(parameters, parts)]).double()
            assert torch.isfinite(gradients[name]).all()
        gradients["joint"] = gradients["time"] + gradients["log_quantity"]
        gradient_rows.append({"batch": batch_number, "count": len(q), "body_count": int(body.sum()),
            "time_vs_log_quantity": cosine(gradients["time"], gradients["log_quantity"]),
            "log_quantity_vs_body_mae": cosine(gradients["log_quantity"], gradients["body_mae"]),
            "joint_vs_body_mae": cosine(gradients["joint"], gradients["body_mae"]),
            "encoder_time_vs_log_quantity": cosine(gradients["time"][:memory_start], gradients["log_quantity"][:memory_start]),
            "memory_time_vs_log_quantity": cosine(gradients["time"][memory_start:], gradients["log_quantity"][memory_start:]),
            "memory_time_grad_norm": gradients["time"][memory_start:].norm().item(),
            "memory_quantity_grad_norm": gradients["log_quantity"][memory_start:].norm().item(),
            "time_grad_norm": gradients["time"].norm().item(), "log_quantity_grad_norm": gradients["log_quantity"].norm().item()})
    arrays = {k: torch.cat(v) for k, v in arrays.items()}
    scopes = {}
    for name, chosen in {"all": torch.ones(len(indices), dtype=torch.bool), "body_le_p95": arrays["is_body"]}.items():
        scopes[name] = {k: describe(v[chosen]) for k, v in arrays.items() if k not in ("quantity", "is_body", "prototype_indices")}
        counts = arrays["prototype_indices"][chosen].flatten().bincount(minlength=64).double()
        shares = counts[counts > 0] / counts.sum()
        scopes[name].update(count=int(chosen.sum()), active_prototypes=int((counts > 0).sum()),
                           effective_prototypes=float((-(shares * shares.log()).sum()).exp()))
    after = canonical_state_dict_sha256(model.state_dict())
    assert before == after and all(p.grad is None for p in model.parameters())
    return {"checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": digest(path),
            "state_sha256": before, "state_unchanged": before == after,
            "source_revision": payload["source_revision"], "scopes": scopes, "gradient_batches": gradient_rows,
            "gradient_mode": "eval; train targets; encoder plus prototype parameters; no optimizer step"}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(1)
    start = time.monotonic()
    registry = read(ROOT / "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json")
    result = {"scope": "post-stop mechanism diagnosis, not a new performance experiment", "training_performed": False,
              "held_out_used": False, "validation_inference_performed": False, "torch": torch.__version__,
              "device": "cpu", "sample_seed": 20260903, "train_sample_limit": 1024, "datasets": {}}
    for row in registry["datasets"]:
        name = row["dataset"]
        if name not in DATASETS:
            continue
        print(f"Audit {name}", flush=True)
        baseline, candidate = ROOT / row["artifact_dir"], RAW / name
        paths = {"original_hard": run_path(baseline, "titantpp"), "weighted": run_path(candidate, CANDIDATE), "thp": run_path(baseline, "thp")}
        item = {"histories": {key: history_audit(path) for key, path in paths.items()},
                "strata": strata_audit(baseline, candidate), "train_probes": {}}
        data = ROOT / row["data_path"]
        assert digest(data) == row["data_sha256"]
        frame = prepare_count_frame(pl.scan_parquet(data).filter(pl.col("chronological_split") == "train").collect())
        dataset = RMTPPWeekLookbackDataset(frame, lookback_weeks=row["lookback"], max_seq_len=row["max_seq_len"],
                    mode="all", split_col="chronological_split", target_splits={"train"})
        assert len(dataset) == read(ROOT / "paper/contracts/hard_lmm_weighted_static_v1.json")["train_target_counts"][name]
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(20260903))[:1024].sort().values
        item["sample"] = {"indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                          "available_targets": len(dataset), "selected_targets": len(indices), "data_sha256": digest(data)}
        quantity_contract = read(candidate / "launch_contract.json")["quantity_contract"]
        assert quantity_contract["quantiles"] == [0.5, 0.9, 0.95, 0.99]
        p95 = quantity_contract["boundaries"][2]
        for key, backbone, checkpoint in (
            ("original_selected", "titantpp", paths["original_hard"] / "best_val_joint_objective_model.pt"),
            ("weighted_selected", CANDIDATE, paths["weighted"] / "best_val_joint_objective_model.pt"),
            ("weighted_last", CANDIDATE, paths["weighted"] / "last_epoch_state.pt"),
        ):
            print(f"  {key}: 1024 fixed train targets", flush=True)
            item["train_probes"][key] = train_probe(checkpoint, backbone, dataset, indices, p95)
        item["input_digests"] = {str(path.relative_to(ROOT)): digest(path) for root in paths.values()
                                for path in (root / "history.json", root / "summary.json")}
        result["datasets"][name] = item
    result["elapsed_seconds"] = time.monotonic() - start
    result["analysis_script_sha256"] = digest(Path(__file__))
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Saved {OUT.relative_to(ROOT)} in {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
