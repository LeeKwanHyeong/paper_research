#!/usr/bin/env python3
"""Inference-only replay; preserve event evidence even when equivalence fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_loader.event_seq_data_module import collate_week_lookback
from paper.scripts.analyze_count_aware_b0_retrieval import restore_b0, summary_path
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.hard_lmm_frozen_probe import extract_features, metric_values, sample_indices
from paper.scripts.run_hard_lmm_frozen_probe import (
    gpu_preflight, load_json, make_dataset, run_guarded, save_json, sha256_file,
    verify_hashes, visible_frame,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

POLICY = ROOT / "paper/contracts/hard_lmm_seed62_frozen_replay_v1.json"


def indices_for(dataset, phase):
    if phase == "train":
        return sample_indices(len(dataset), 65536, seed=42)[:256]
    if phase == "validation":
        return torch.arange(len(dataset))
    raise ValueError("Only train and validation inference are permitted")


def frozen(model, digest):
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("Inference model must be frozen")
    if canonical_state_dict_sha256(model.state_dict()) != digest:
        raise AssertionError("Original checkpoint state changed")


@torch.no_grad()
def traced_outputs(model, dts, mask, quantities, route):
    """Observe the existing retrieval without changing its input or return value."""
    if route not in ("official", "probe"):
        raise ValueError("Unknown inference route")
    observed = []
    original = model.lmm.retrieve
    rows = torch.arange(len(dts), device=dts.device)
    previous = mask.sum(1).long() - 2

    def retrieve(encoded, memory=None):
        residual, trace = original(encoded, memory)
        observed.append({
            "prototype_indices": trace["prototype_indices"][rows, previous].cpu(),
            "topk_similarity": trace["topk_similarity"][rows, previous].cpu(),
            "trace_z": model.quantity_head((encoded + residual)[rows, previous]).squeeze(-1).cpu(),
        })
        return residual, trace

    with patch.object(model.lmm, "retrieve", side_effect=retrieve):
        if route == "official":
            values = target_outputs(model, dts, mask, quantities, lambda_log_qty=1)
            output = {"prediction": values["pred_qty"].cpu(), "quantity": values["true_qty"].cpu(),
                "time_nll": values["time_loss"].cpu(), "log_loss": values["log_qty_loss"].cpu(),
                "joint_loss": values["joint_loss"].cpu(), "history_length": values["history_length"].cpu()}
        else:
            output = extract_features(model, dts, mask, quantities)
    if len(observed) != 1:
        raise AssertionError("Expected exactly one static-memory retrieval")
    output.update(observed[0])
    if route == "probe" and not torch.equal(output["z"], output["trace_z"]):
        raise AssertionError("Retrieval instrumentation changed the readout")
    if any(not torch.isfinite(v).all() for v in output.values()):
        raise FloatingPointError("Nonfinite inference evidence")
    return output


def official_metrics(events):
    error = events["official_prediction"].double() - events["quantity"].double()
    return {"qty_mae": error.abs().mean().item(), "qty_rmse": error.square().mean().sqrt().item(),
        "time_nll": events["official_time_nll"].double().mean().item(),
        "log_qty_mse": events["official_log_loss"].double().mean().item(),
        "joint_objective": events["official_joint_loss"].double().mean().item()}


def replay_assessment(metrics, reference, tolerance):
    differences = {k: abs(metrics[k] - reference[f"best_val_{k}"])
        for k in ("qty_mae", "qty_rmse", "time_nll", "joint_objective")}
    return {"absolute_differences": differences, "tolerance": tolerance,
        "all_pass": all(v <= tolerance for v in differences.values())}


def execute(args, status):
    policy, manifest = load_json(POLICY), load_json(args.source_manifest)
    verify_hashes(ROOT, manifest["files"])
    allowed = policy["runtime"][args.device]
    if str(torch.__version__) not in allowed:
        raise ValueError(f"Unregistered runtime: {torch.__version__}")
    prior = load_json(ROOT / "paper/contracts/hard_lmm_readout_seed_replication_v1.json")
    row = next(r for r in prior["checkpoints"] if r["seed"] == 62)
    spec = load_json(ROOT / "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json")["datasets"][0]
    inputs = {args.checkpoint: row["checkpoint_file_sha256"], args.data: spec["data_sha256"],
        args.reference_dir / "launch_contract.json": spec["contract_sha256"],
        summary_path(args.reference_dir, 62): row["summary_sha256"]}
    for path, digest in inputs.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Input digest mismatch: {path}")
    if args.phase == "validation":
        pre = load_json(args.train_preflight / "summary.json") if args.train_preflight else {}
        if pre.get("phase") != "train" or pre.get("device") != args.device or not pre.get("route_preflight_passed"):
            raise ValueError("Matching train-only preflight required before validation")
        if load_json(args.train_preflight / "source_manifest.json") != manifest:
            raise ValueError("Preflight source differs")
        if pre["runtime"]["torch"] != str(torch.__version__):
            raise ValueError("Preflight runtime differs")
        verify_hashes(args.train_preflight, load_json(args.train_preflight / "output_digests.json"))
    if args.device == "cuda":
        save_json(args.output / "gpu_preflight.json", gpu_preflight())
    launch = load_json(args.reference_dir / "launch_contract.json")
    model, audit = restore_b0(args.checkpoint, launch, args.device)
    model.requires_grad_(False).eval()
    frozen(model, row["checkpoint_state_sha256"])
    save_json(args.output / "source_manifest.json", manifest)
    save_json(args.output / "launch_contract.json", policy)
    save_json(args.output / "baseline_audit.json", audit)
    save_json(args.output / "input_digests.json", {str(p.resolve()): d for p, d in inputs.items()})
    frame = visible_frame(args.data)
    dataset = make_dataset(frame, spec, args.phase)
    indices = indices_for(dataset, args.phase)
    expected = policy["targets"][args.phase]
    if len(indices) != expected:
        raise ValueError("Wrong target population")
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=128, shuffle=False,
        collate_fn=collate_week_lookback, num_workers=0)
    collected, offset, route_ok = {}, 0, True
    for _, dts, mask, series, quantities in loader:
        dts, mask, quantities = dts.to(args.device), mask.to(args.device), quantities.to(args.device)
        native = traced_outputs(model, dts, mask, quantities, "official")
        probe = traced_outputs(model, dts, mask, quantities, "probe")
        if not torch.equal(native["quantity"], probe["quantity"]) or not torch.equal(native["history_length"], probe["history_length"]):
            raise AssertionError("Inference routes use different targets")
        pred = F.softplus(probe["z"]).expm1()
        route_ok &= torch.allclose(pred, native["prediction"], atol=1e-6, rtol=1e-5)
        route_ok &= torch.allclose(probe["time_nll"], native["time_nll"], atol=1e-6, rtol=1e-5)
        ids = indices[offset:offset + len(dts)]
        values = {"target_index": ids, "series_index": series.cpu(),
            "context_end": torch.tensor([dataset.index[int(i)][1] for i in ids]),
            "quantity": probe["quantity"], "history_length": probe["history_length"],
            **{f"official_{k}": v for k, v in native.items() if k not in ("quantity", "history_length")},
            **{f"probe_{k}": v for k, v in probe.items() if k not in ("quantity", "history_length")}}
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
        offset += len(dts)
        if offset == len(indices) or offset % 12800 == 128:
            status(stage="frozen_inference", phase=args.phase, device=args.device, completed_targets=offset, total_targets=len(indices))
    events = {k: torch.cat(v) for k, v in collected.items()}
    events["probe_prediction"] = F.softplus(events["probe_z"]).expm1()
    events["probe_log_loss"] = (F.softplus(events["probe_z"]) - events["quantity"].log1p()).square()
    frozen(model, row["checkpoint_state_sha256"])
    torch.save(events, args.output / "events.pt")
    metrics = {"official": official_metrics(events),
        "probe": metric_values(events["probe_z"], events["quantity"], events["probe_time_nll"])}
    result = {"status": "complete", "scope": "frozen_inference_only_not_candidate_validation",
        "phase": args.phase, "device": args.device, "targets": offset, "route_preflight_passed": bool(route_ok),
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "python": sys.version,
            "platform": platform.platform(), "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision()},
        "metrics": metrics, "base_unchanged": True, "training_steps": 0, "held_out_test_evaluated": False,
        "events_sha256": sha256_file(args.output / "events.pt")}
    if args.phase == "validation":
        reference = load_json(summary_path(args.reference_dir, 62))
        result["reference_replay"] = {k: replay_assessment(v, reference, policy["replay_absolute_tolerance"])
            for k, v in metrics.items()}
    # A failed equivalence check is a diagnostic result, not a lost-cache exception.
    save_json(args.output / "summary.json", result)
    verify_hashes(ROOT, manifest["files"])
    for path, digest in inputs.items():
        if sha256_file(path) != digest:
            raise AssertionError("Inference input changed")
    save_json(args.output / "output_digests.json", {str(p.relative_to(args.output)): sha256_file(p)
        for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "probe_status.json"})
    status(status="complete", stage="inference_evidence_saved", training_steps=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "checkpoint", "reference-dir", "source-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--phase", choices=("train", "validation"), required=True)
    parser.add_argument("--train-preflight", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_guarded(args.output, lambda status: execute(args, status), {"training_steps": 0, "held_out_test_evaluated": False})
