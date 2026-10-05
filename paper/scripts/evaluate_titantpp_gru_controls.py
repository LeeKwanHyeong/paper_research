"""Aggregate-only diagnostics for frozen HistoryMLP width checkpoints.

The campaign calls ``evaluate_checkpoint(..., validation_replay=endpoint,
train_sample_contract=...)``: its full-validation endpoint is reused, while
only the original deterministic TRAIN sample receives an additional forward
pass. No optimizer, selection change, held-out loader, or prediction export.
The caller must install the width model routes before loading a width model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAIL_THRESHOLDS = {
    "yellow_trip_hourly": 3449., "raf_spare_parts": 200.,
    "intermittent_frozen_5000": 187., "insta_market_basket": 35.,
}
TRAIN_SAMPLE = {"train_sampling_seed": 20261003, "max_per_bin": 2048}
SCHEMA = "titantpp_history_width_metrics_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_array(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def _bounds(data):
    bounds = np.asarray(data["quantity_boundaries_all_train_rows"], dtype=np.float64)
    require(bounds.shape == (4,) and np.isfinite(bounds).all()
            and np.all(np.diff(bounds) >= 0), "Invalid frozen quantity boundaries")
    require(data["dataset_id"] in TAIL_THRESHOLDS
            and bounds[-1] == TAIL_THRESHOLDS[data["dataset_id"]], "Frozen upper-bin threshold changed")
    return bounds


def _loader(data, frame, split):
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    require(split in ("train", "validation"), "Only train/validation are admitted")
    require(set(frame["chronological_split"].to_list()) <= {"train", "validation"},
            "Held-out rows must be excluded before evaluation")
    return make_loader(frame, target_split=split, shuffle=False, generator=None,
                       **{k: data["loader"][k] for k in ("batch_size", "lookback_weeks", "max_seq_len")})


def population_identity(dataset, split, expected):
    """Match the original diagnostic's complete eligible-target identity."""
    require(split in ("train", "validation"), "Only train/validation are admitted")
    ix = np.asarray(dataset.index, dtype=np.int64).reshape(-1, 2)
    part, position = ix[:, 0], ix[:, 1] + 1
    seq = np.fromiter((dataset.seq_lists[p][t] for p, t in zip(part, position)), dtype=np.int64)
    q = np.fromiter((dataset.val_lists[p][t] for p, t in zip(part, position)), dtype=np.float64)
    require(all(dataset.split_lists[p][t] == split for p, t in zip(part, position)),
            "Target split differs from admitted split")
    require(np.isfinite(q).all() and (q >= 0).all(), "Invalid target quantity")
    h = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
    h.update(json.dumps([str(p) for p in dataset.parts], ensure_ascii=False, separators=(",", ":")).encode())
    for a in (part, position, seq):
        h.update(a.astype("<i8").tobytes())
    qh = hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + q.astype("<f8").tobytes()).hexdigest()
    identity = {"target_count": len(q), "target_identity_sha256": h.hexdigest(), "target_quantity_sha256": qh}
    require(identity == expected, "Eligible target population identity changed")
    return q, identity


def frozen_train_sample(q, bounds, contract=None):
    """Reproduce diagnose.py's sorted, stratified SRS and N_bin/n_bin weights."""
    contract = TRAIN_SAMPLE if contract is None else contract
    require(all(contract.get(k) == v for k, v in TRAIN_SAMPLE.items()), "Frozen TRAIN sampling contract changed")
    q = np.asarray(q, dtype=np.float64)
    groups = np.searchsorted(bounds, q, side="left")  # Equality belongs to the lower bin.
    rng = np.random.default_rng(contract["train_sampling_seed"])
    indices, weights, strata = [], [], []
    for g in range(len(bounds) + 1):
        candidates = np.flatnonzero(groups == g)
        n = min(len(candidates), contract["max_per_bin"])
        strata.append({"bin": g, "population_n": len(candidates), "sample_n": n})
        if n:
            chosen = np.sort(rng.choice(candidates, n, replace=False))
            indices.extend(chosen.tolist())
            weights.extend([len(candidates) / n] * n)
    order = np.argsort(indices)
    indices = np.asarray(indices, dtype=np.int64)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    receipt = {**TRAIN_SAMPLE, "sample_n": len(indices), "represented_n": float(weights.sum()),
               "sample_indices_sha256": _sha_array(indices, "<i8"),
               "weights_sha256": _sha_array(weights, "<f8"), "strata": strata,
               "weight_definition": "N_bin / n_bin; SRS without replacement within fixed TRAIN quantity bins"}
    require(math.isclose(receipt["represented_n"], len(q), rel_tol=1e-12, abs_tol=1e-8),
            "Sample weights do not reconstruct TRAIN population")
    return indices, weights, receipt


class QuantityMetrics:
    """Streaming sufficient statistics, with event-count and represented-count distinct."""

    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        # count, weight, absolute error, squared error, signed error, time loss
        self.sums = np.zeros((len(self.bounds) + 2, 6), dtype=np.float64)

    def add(self, q, prediction, time_nll, weights=None):
        q, p, t = [np.asarray(x, dtype=np.float64) for x in (q, prediction, time_nll)]
        w = np.ones_like(q) if weights is None else np.asarray(weights, dtype=np.float64)
        require(q.ndim == 1 and q.shape == p.shape == t.shape == w.shape, "Metric shapes differ")
        require(all(np.isfinite(a).all() for a in (q, p, t, w))
                and (q >= 0).all() and (p >= 0).all() and (w > 0).all(), "Invalid metric inputs")
        e = p - q
        groups = np.searchsorted(self.bounds, q, side="left")
        for row in range(len(self.sums)):
            keep = np.ones(q.shape, dtype=bool) if row == 0 else groups == row - 1
            ww, ee = w[keep], e[keep]
            self.sums[row] += [int(keep.sum()), ww.sum(), np.sum(ww * abs(ee)),
                               np.sum(ww * ee ** 2), np.sum(ww * ee), np.sum(ww * t[keep])]
        require(np.isfinite(self.sums).all(), "Metric aggregation overflow")

    @staticmethod
    def _finish(a):
        count, mass, absolute, squared, signed, temporal = a
        return {"count": int(count), "represented_n": float(mass), "qty_sse": float(squared),
                "qty_absolute_error_sum": float(absolute), "qty_signed_error_sum": float(signed),
                "time_nll_sum": float(temporal), "qty_rmse": math.sqrt(squared / mass) if mass else None,
                "qty_mae": float(absolute / mass) if mass else None,
                "qty_bias": float(signed / mass) if mass else None,
                "time_nll": float(temporal / mass) if mass else None}

    def finish(self):
        require(np.allclose(self.sums[1:].sum(axis=0), self.sums[0], rtol=1e-10, atol=1e-8),
                "Quantity strata do not reconstruct aggregate sums")
        cells = [{"bin": i, **self._finish(a)} for i, a in enumerate(self.sums[1:])]
        return {**self._finish(self.sums[0]), "quantity_boundaries": self.bounds.tolist(),
                "quantity_cells": cells, "tail": self._finish(self.sums[-1]),
                "tail_definition": "raw_quantity > quantity_boundaries[-1]",
                "bias_definition": "prediction minus target; target-defined strata are descriptive"}


def _payload(path, data):
    from paper.scripts.run_titantpp_gru_controls import SUPPORTED_ARMS as CONTROL_SUPPORTED
    from paper.scripts.run_titantpp_history_width import SUPPORTED_ARMS as WIDTH_SUPPORTED
    SUPPORTED_ARMS=(*CONTROL_SUPPORTED,*WIDTH_SUPPORTED)
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    payload = torch_load_checkpoint(Path(path), map_location="cpu")
    require(payload.get("backbone") in SUPPORTED_ARMS,
            "Checkpoint is outside the width comparison")
    validate_checkpoint_route(payload, payload["backbone"])
    require(payload.get("evaluation_scope") == "validation_only"
            and payload.get("held_out_test_evaluated") is False, "Checkpoint evaluation scope differs")
    head = payload.get("interface_meta", {}).get("time_head", {}) or payload.get("interface", {}).get("time_head", {})
    require(head.get("observation_likelihood") == data["model"]["time_observation_contract"],
            "Checkpoint time observation contract changed")
    state = canonical_state_dict_sha256(payload["model_state_dict"])
    require(payload.get("model_state_sha256") == state, "Checkpoint state digest changed")
    selector = checkpoint_monitor_spec("validation_raw_quantity_rmse")
    require(payload.get("variant") == "count_only_log_regression"
            and payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
            and payload.get("checkpoint_monitor_history_key") == selector["history_key"]
            and payload.get("checkpoint_selection") == selector["selection"],
            "Checkpoint objective/strict-first raw validation RMSE selector changed")
    return payload, state


def _build_model(data, payload, device, model_builder=None):
    from models.TPPs.CountAwareFactory import build_count_aware_model
    if model_builder is None:
        config = {k: v for k, v in data["model"].items()
                  if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
        model, _ = build_count_aware_model(payload["backbone"], **config,
            train_log_mean=data["statistics"]["train_log_mean"],
            train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])
    else:
        model, _ = model_builder(data, payload["backbone"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def _infer(model, dataset, indices, weights, expected_q, bounds, batch_size, device, budget):
    import torch
    from data_loader.event_seq_data_module import collate_week_lookback
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    metrics = QuantityMetrics(bounds)
    residual=[];norm_sums=[0.,0.,0.]
    def hook(module,args,output):
        hidden,valid=args[:2]
        # Evaluation consumes the last observed history event; the withheld target is not eligible.
        positions=torch.arange(valid.size(1),device=valid.device).expand_as(valid)
        last=torch.where(valid,positions,-1).max(1).values-1
        ids=torch.arange(valid.size(0),device=valid.device)
        delta=output[ids,last].norm(dim=-1).detach().cpu().numpy()
        base=hidden[ids,last].norm(dim=-1).detach().cpu().numpy()
        residual[:]=[delta,base]
    handle=model.multilag_detail.register_forward_hook(hook)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            budget()
            ids = indices[start:start + batch_size]
            _, dt, mask, _, quantity = collate_week_lookback([dataset[int(i)] for i in ids])
            outputs = target_outputs(model, dt.to(device), mask.to(device), quantity.to(device), lambda_log_qty=1.)
            q, pred, temporal = [outputs[k].detach().cpu().numpy().astype(np.float64)
                                  for k in ("true_qty", "pred_qty", "time_loss")]
            require(np.array_equal(q, expected_q[ids]), "Inference targets differ from fixed population")
            batch_weights=weights[start:start+len(ids)]
            metrics.add(q,pred,temporal,batch_weights)
            require(len(residual)==2,'Missing residual diagnostic')
            delta,base=residual
            require(np.isfinite(delta).all() and np.isfinite(base).all(),'Nonfinite residual diagnostic')
            norm_sums[0]+=float(np.sum(batch_weights*delta))
            norm_sums[1]+=float(np.sum(batch_weights*delta/np.maximum(base,1e-12)))
            norm_sums[2]+=float(np.sum(batch_weights))
    budget()
    handle.remove()
    result=metrics.finish()
    result['residual_diagnostic']={'mean_l2_delta':norm_sums[0]/norm_sums[2],
        'mean_delta_to_base_l2_ratio':norm_sums[1]/norm_sums[2],
        'scope':'last_history_state; same target weights; descriptive only; fixed alpha never retuned'}
    return result


def _train_sample(payload, state, path, data, frame, budget, device, contract, model_builder):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    bounds = _bounds(data)
    loader = _loader(data, frame, "train")
    q, identity = population_identity(loader.dataset, "train", data["inherited_data_identity"]["populations"]["train"])
    indices, weights, sampling = frozen_train_sample(q, bounds, contract)
    for i in indices:
        p, end = loader.dataset.index[int(i)]
        require(all(s == "train" for s in loader.dataset.split_lists[p][:end + 2]),
                "TRAIN sample context contains another split")
    model = _build_model(data, payload, device, model_builder)
    metrics = _infer(model, loader.dataset, indices, weights, q, bounds, 64, device, budget)
    require(canonical_state_dict_sha256(model.state_dict()) == state, "Evaluation mutated checkpoint parameters")
    return {**metrics, "split": "train", "population": identity, "sampling": sampling,
            "state_sha256": state, "checkpoint_path": str(path), "new_training": False,
            "time_metric": "recorded_positive_integer_time_nll",
            "held_out_test_evaluated": False, "interpretation": "weighted in-sample diagnosis; not capacity or calibration proof"}


def evaluate_train_sample(path, data, frame, budget, *, device="cuda:0"):
    """Frozen seed-20261003 / maximum-2048-per-bin TRAIN diagnostic only."""
    payload, state = _payload(path, data)
    return _train_sample(payload, state, path, data, frame, budget, device, TRAIN_SAMPLE, None)


def reuse_validation_replay(replay, data, population, state, target_quantity):
    """Consume aggregate endpoint evidence without replaying its predictions."""
    from paper.scripts.run_local_detail_benchmark import audit_replay_accounting
    require(replay.get("evaluation_scope") == "validation_only"
            and replay.get("held_out_test_evaluated") is False, "Cached replay is not validation-only")
    require(replay.get("state_sha256") == state, "Cached replay checkpoint state differs")
    bounds = _bounds(data)
    require(replay.get("quantity_boundaries") == bounds.tolist(), "Cached quantity boundaries changed")
    require(replay.get("count") == population["target_count"], "Cached validation population count changed")
    audit_replay_accounting(replay)
    expected_counts = np.bincount(np.searchsorted(bounds, target_quantity, side="left"), minlength=5)
    require([cell["count"] for cell in replay["quantity_cells"]] == expected_counts.tolist(),
            "Cached validation quantity population changed")
    def cell(x):
        return {**x, "represented_n": float(x["count"]), "qty_bias": x.get("qty_bias"),
                "bias_available": "qty_bias" in x}
    return {**cell(replay), "tail": cell(replay["tail"]),
            "quantity_cells": [cell(x) for x in replay["quantity_cells"]],
            "split": "validation", "population": population, "full_population": True,
            "reused_endpoint_replay": True, "additional_validation_forward_pass": False,
            "tail_definition": "raw_quantity > quantity_boundaries[-1]"}


def evaluate_checkpoint(data, checkpoint_path, frame, *, device="cpu", budget_check=lambda: None,
                        validation_replay=None, train_sample_contract=None, model_builder=None):
    """Reuse a validation endpoint and optionally compute its frozen TRAIN sample.

    Passing no validation_replay explicitly requests one full validation pass.
    train_sample_contract must retain TRAIN_SAMPLE's seed and per-bin maximum.
    model_builder, if supplied, has the shared engine's (data, arm)->(model,meta) API.
    """
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    budget_check()
    payload, state = _payload(checkpoint_path, data)
    bounds = _bounds(data)
    loader = _loader(data, frame, "validation")
    q, population = population_identity(loader.dataset, "validation", data["inherited_data_identity"]["populations"]["validation"])
    if validation_replay is not None:
        validation = reuse_validation_replay(validation_replay, data, population, state, q)
    else:
        model = _build_model(data, payload, device, model_builder)
        validation = _infer(model, loader.dataset, np.arange(len(q)), np.ones(len(q)), q, bounds,
                            data["loader"]["batch_size"], device, budget_check)
        require(canonical_state_dict_sha256(model.state_dict()) == state, "Evaluation mutated checkpoint parameters")
        validation.update(split="validation", population=population, full_population=True,
                          state_sha256=state, reused_endpoint_replay=False, additional_validation_forward_pass=True,
                          time_metric="recorded_positive_integer_time_nll")
        del model
    del loader, q
    train = None
    if train_sample_contract is not None:
        train = _train_sample(payload, state, checkpoint_path, data, frame, budget_check, device,
                              train_sample_contract, model_builder)
    budget_check()
    return {"schema": SCHEMA, "dataset": data["dataset_id"], "model": payload["backbone"],
            "seed": payload["seed"], "epoch": payload.get("epoch", payload.get("best_epoch")),
            "validation": validation, "train_sample": train,
            "provenance": {"checkpoint_path": str(checkpoint_path), "state_sha256": state,
                           "device": str(device), "checkpoint_monitor": payload["checkpoint_monitor"],
                           "selection_unchanged": True, "new_training": False,
                           "held_out_test_evaluated": False, "raw_predictions_written": False}}


def compare_evaluations(baseline, candidate):
    """Paired descriptive changes. Refuse different populations or TRAIN samples."""
    require(baseline["dataset"] == candidate["dataset"] and baseline["seed"] == candidate["seed"],
            "Comparison dataset/seed differs")
    require(baseline["validation"]["population"] == candidate["validation"]["population"],
            "Validation population identity differs")
    require((baseline["train_sample"] is None) == (candidate["train_sample"] is None),
            "Both comparisons must include the same splits")
    differences = {}
    for split in ("validation", "train_sample"):
        a, b = baseline[split], candidate[split]
        if a is None:
            continue
        require(a["quantity_boundaries"] == b["quantity_boundaries"], "Comparison boundaries differ")
        require(a["population"] == b["population"], "Comparison population identity differs")
        if split == "train_sample":
            require(a["sampling"] == b["sampling"], "TRAIN sample indices/weights/contract differ")
        differences[split] = {}
        for name in ("overall", "tail"):
            aa, bb = (a, b) if name == "overall" else (a["tail"], b["tail"])
            require(aa["count"] == bb["count"] and aa["represented_n"] == bb["represented_n"],
                    "Comparison stratum counts differ")
            differences[split][name] = {k: bb[k] - aa[k] if aa.get(k) is not None and bb.get(k) is not None else None
                                       for k in ("qty_rmse", "qty_mae", "time_nll", "qty_bias")}
    return {"dataset": baseline["dataset"], "seed": baseline["seed"],
            "baseline": {k: baseline[k] for k in ("model", "epoch", "provenance")},
            "candidate": {k: candidate[k] for k in ("model", "epoch", "provenance")},
            "candidate_minus_baseline": differences, "population_and_sample_identity_verified": True,
            "selection_unchanged": True, "interpretation": "descriptive paired checkpoint comparison; no significance or causal claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(TAIL_THRESHOLDS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-replay", type=Path)
    parser.add_argument("--endpoint", choices=("selected", "last"), default="selected")
    parser.add_argument("--train-sample", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Do not overwrite an existing evaluation")
    from paper.scripts import run_titantpp_gru_controls as engine
    from paper.scripts.run_multilag_detail_execution import prepare_admitted_data
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    engine.install_hooks()
    configure_runtime(args.device, threads=2)
    contract = json.loads(args.contract.read_text())
    data = next(d for d in contract["datasets"] if d["dataset_id"] == args.dataset)
    from copy import deepcopy
    data = deepcopy(data)
    # The immutable contract keeps portable paths relative to its source bundle.
    for ref in data['inherited_data_identity'].values():
        if isinstance(ref, dict) and 'path' in ref and not Path(ref['path']).is_absolute():
            ref['path'] = str(args.contract.parent / 'source' / ref['path'])
    frame, _ = prepare_admitted_data(data)
    replay = None
    if args.validation_replay:
        replay = json.loads(args.validation_replay.read_text())
        replay = replay.get(args.endpoint, replay)
    result = evaluate_checkpoint(data, args.checkpoint, frame, device=args.device, validation_replay=replay,
                                 train_sample_contract=TRAIN_SAMPLE if args.train_sample else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
