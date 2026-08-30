#!/usr/bin/env python3
"""Fail closed on partial, mixed-source, or non-finite MAC stability evidence."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.scripts.validate_count_aware_titantpp_mac_three_seed_validation import (
    load_json, require_finite, save_json_atomic, sha256_file,
)


def check_header(contract, summary, history, *, seed, revision):
    expected = {"status": "complete", "completed_run_count": 1,
                "source_revision": revision, "dataset": "insta_market_basket",
                "seeds": [seed], "backbones": ["titantpp_titans_mac"],
                "epochs": 1, "batch_size": 128, "lr": 0.001,
                "lookback_weeks": 52, "max_seq_len": 64, "hidden_dim": 64,
                "lambda_log_qty": 1., "lambda_tail": 0., "grad_clip": 1.,
                "titans_memory_gradient_clip": 1., "partial_smoke": False,
                "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Preflight contract mismatch: {key}")
    if contract["time_head"]["mode"] != "legacy_clamped_rmtpp":
        raise ValueError("Time head changed")
    if summary.get("status") != "success" or summary.get("completed_epochs") != 1:
        raise ValueError("Full epoch is not complete")
    for payload in (summary,):
        if payload.get("source_revision") != revision or payload.get("source_revision_history") != [revision]:
            raise ValueError("Mixed source revision")
        if payload.get("held_out_test_evaluated") is not False:
            raise ValueError("Held-out test was evaluated")
        if payload["encoder_config"].get("titans_memory_gradient_clip") != 1.:
            raise ValueError("Checkpoint metadata lacks the inner stability policy")
    if len(history) != 1 or history[0]["epoch"] != 1 or not history[0].get("train_all_finite"):
        raise ValueError("Incomplete or non-finite history")
    for name, value in (("contract", contract), ("summary", summary), ("history", history)):
        require_finite(value, location=name)


def validate(args):
    import polars as pl
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, right_pad_batch, target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    manifest = load_json(args.source_manifest)
    if manifest["training_source_revision"] != args.expected_revision:
        raise ValueError("Manifest source revision mismatch")
    if not manifest.get("files"):
        raise ValueError("Empty source digest manifest")
    for path, digest in manifest["files"].items():
        if sha256_file(args.project_root / path) != digest:
            raise ValueError(f"Source changed during preflight: {path}")
    contract = load_json(args.run_root / "launch_contract.json")
    run = args.run_root / "runs/titantpp_titans_mac/count_only_log_regression" / f"seed_{args.seed}"
    summary = load_json(run / "summary.json")
    history = load_json(run / "history.json")["history"]
    check_header(contract, summary, history, seed=args.seed, revision=args.expected_revision)
    if any("test" in p.name.lower() for p in args.run_root.rglob("*") if p.is_file()):
        raise ValueError("Unexpected held-out test artifact")
    if sha256_file(Path(contract["data_path"])) != contract["data_sha256"]:
        raise ValueError("Data digest changed")
    if sha256_file(Path(contract["split_manifest_path"])) != contract["split_manifest_sha256"]:
        raise ValueError("Fixed split digest changed")
    frame = prepare_count_frame(pl.read_parquet(contract["data_path"]))
    loaders = {split: make_loader(frame, target_split=split, batch_size=128,
                                  lookback_weeks=52, max_seq_len=64,
                                  shuffle=False, generator=None)
               for split in ("train", "validation")}
    train_count, val_count = (len(loaders[s].dataset) for s in ("train", "validation"))
    if history[0]["train_event_count"] != train_count or history[0]["train_batch_count"] != math.ceil(train_count / 128):
        raise ValueError("Training skipped targets or batches")
    for key in ("quantity_rows", "history_rows"):
        if sum(row["count"] for row in summary[key]) != val_count:
            raise ValueError(f"Validation coverage incomplete: {key}")

    checkpoint = torch.load(run / "best_val_joint_objective_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["source_revision"] != args.expected_revision or checkpoint["source_revision_history"] != [args.expected_revision]:
        raise ValueError("Checkpoint source mismatch")
    if checkpoint["encoder_config"].get("titans_memory_gradient_clip") != 1.:
        raise ValueError("Checkpoint inner policy mismatch")
    digest = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    if digest != summary["checkpoint_state_sha256"] or digest != checkpoint["model_state_sha256"]:
        raise ValueError("Checkpoint tensor digest mismatch")
    if not all(torch.isfinite(t).all() for t in checkpoint["model_state_dict"].values()):
        raise ValueError("Non-finite checkpoint parameter")
    interface = checkpoint["interface_meta"]
    time_kwargs = {name: contract["time_head"][name] for name in (
        "time_scale", "time_w_max", "time_intercept_limit", "time_initial_intercept",
        "time_wd_safety_limit", "time_initial_location", "time_initial_scale", "time_sigma_floor")}
    def restore(state):
        model, _ = build_count_aware_model(
            "titantpp_titans_mac", hidden_dim=64, max_seq_len=64,
            train_log_mean=interface["train_target_mean"], train_log_std=interface["train_target_std"],
            titans_memory_gradient_clip=1., time_head_mode="legacy_clamped_rmtpp", **time_kwargs)
        model.load_state_dict(state, strict=True)
        return model.to(args.device).eval()
    first = restore(checkpoint["model_state_dict"])
    buffer = io.BytesIO()
    torch.save(first.state_dict(), buffer)
    buffer.seek(0)
    second = restore(torch.load(buffer, map_location="cpu", weights_only=True))
    _, dts, mask, _, qty = next(iter(loaders["validation"]))
    dts, mask, qty = (t.to(args.device) for t in (dts, mask, qty))
    with torch.no_grad():
        a = target_outputs(first, dts, mask, qty, lambda_log_qty=1.)
        b = target_outputs(second, dts, mask, qty, lambda_log_qty=1.)
        for key in a:
            if not torch.isfinite(a[key]).all() or not torch.equal(a[key], b[key]):
                raise ValueError(f"Prediction restore mismatch: {key}")
        right_dt, right_qty, right_mask, lengths = right_pad_batch(dts, qty, mask)
        rows = torch.arange(lengths.numel(), device=lengths.device)
        right_qty = right_qty.clone()
        right_qty[rows, lengths-1] = 0
        writes = right_mask.clone()
        writes[rows, lengths-1] = False
        states = [model.encode_with_memory_state(right_dt, right_qty, right_mask,
                  memory_write_mask=writes)[1] for model in (first, second)]
        for left, right in zip((*states[0].memory_tensors(), *states[0].momentum_tensors()),
                               (*states[1].memory_tensors(), *states[1].momentum_tensors())):
            if not torch.isfinite(left).all() or not torch.equal(left, right):
                raise ValueError("Observed-history memory replay mismatch")
    return {"status": "complete", "seed": args.seed, "training_source_revision": args.expected_revision,
            "verified_source_files": len(manifest["files"]), "train_target_count": train_count,
            "train_batch_count": len(loaders["train"]), "validation_target_count": val_count,
            "checkpoint_state_sha256": digest, "checkpoint_prediction_replay_exact": True,
            "observed_history_memory_replay_exact": True, "all_metrics_finite": True,
            "held_out_test_evaluated": False, "e300_automatically_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        report = validate(args)
    except BaseException as exc:
        save_json_atomic(args.output, {"status": "failed", "error": repr(exc)})
        raise
    save_json_atomic(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
