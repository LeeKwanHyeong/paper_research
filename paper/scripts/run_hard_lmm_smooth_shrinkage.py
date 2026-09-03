#!/usr/bin/env python3
"""Frozen Hard-LMM smooth shrinkage versus matched scalar and identity controls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.hard_lmm_frozen_probe import (
    acceptance, metric_values, predict, require_finite, summarize, write_event_deltas,
)
from paper.scripts.run_hard_lmm_frozen_probe import (
    load_json, run_guarded, save_json, sha256_file, verify_hashes,
)
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json

CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_smooth_shrinkage_v1.json"
KINDS = ("constant_shrinkage", "smooth_shrinkage")
INITIAL_SCORE = math.log(.05 / .95)


class ResidualGate(nn.Module):
    def __init__(self, feature_dim, kind):
        super().__init__()
        self.kind, self.feature_dim = kind, feature_dim
        if kind == "smooth_shrinkage":
            self.network = nn.Sequential(nn.Linear(feature_dim, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.constant_(self.network[-1].bias, INITIAL_SCORE)
        elif kind == "constant_shrinkage":
            self.score_bias = nn.Parameter(torch.tensor(INITIAL_SCORE))
        elif kind != "identity":
            raise ValueError(f"Unknown gate: {kind}")

    def forward(self, features, base_logit, residual_projection):
        if self.kind == "identity":
            return base_logit, torch.ones_like(base_logit), torch.zeros_like(base_logit)
        score = (self.network(features).squeeze(-1) if self.kind == "smooth_shrinkage"
                 else self.score_bias.expand_as(base_logit))
        gate = 1 - .2 * score.sigmoid()
        correction = (gate - 1) * residual_projection
        return base_logit + correction, gate, correction


def new_gate(feature_dim, kind):
    torch.manual_seed(42)
    return ResidualGate(feature_dim, kind)


@torch.no_grad()
def gate_stats(model, cache):
    z, gate, correction = predict(model, cache)
    return {"mean": gate.double().mean().item(), "std": gate.double().std(unbiased=False).item(),
        "min": gate.min().item(), "max": gate.max().item(),
        "near_identity_fraction": (gate >= 1 - 1e-6).double().mean().item(),
        "near_max_shrinkage_fraction": (gate <= .8 + 1e-6).double().mean().item(),
        "relative_residual_reduction_mean": (1 - gate).double().mean().item(),
        "absolute_projected_residual_reduction_mean": correction.abs().double().mean().item(),
        "metrics": metric_values(z, cache["quantity"], cache["time_nll"])}


def train_epoch(model, train, optimizer, generator, policy):
    total, norm_sum, first_norm, zero_batches, batches = 0., 0., None, 0, 0
    for indices in torch.randperm(len(train["z"]), generator=generator).split(policy["batch_size"]):
        z, _, _ = model(train["features"][indices], train["z"][indices], train["projection"][indices])
        quantity_loss = (F.softplus(z) - train["quantity"][indices].clamp_min(0).log1p()).square().mean()
        objective = quantity_loss + train["time_nll"][indices].mean()
        require_finite("train objective", objective)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        for p in model.parameters():
            if p.grad is not None:
                require_finite("gradient", p.grad)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), policy["gradient_clip"], error_if_nonfinite=True))
        if first_norm is None:
            first_norm = norm
        optimizer.step()
        for p in model.parameters():
            require_finite("parameter", p)
        total += float(objective.detach()) * len(indices)
        norm_sum += norm
        zero_batches += norm == 0
        batches += 1
    return {"train_joint_objective": total / len(train["z"]), "batches": batches,
        "first_gradient_norm": first_norm, "gradient_norm_mean": norm_sum / batches,
        "zero_gradient_batches": zero_batches, "zero_gradient_batch_fraction": zero_batches / batches}


def preflight(kind, train, policy):
    model = new_gate(train["features"].shape[-1], kind)
    before = copy.deepcopy(model.state_dict())
    initial = predict(model, train)[1]
    optimizer = torch.optim.Adam(model.parameters(), lr=policy["learning_rate"], weight_decay=0)
    diagnostics = train_epoch(model, train, optimizer, torch.Generator().manual_seed(policy["shuffle_seed"]), policy)
    after = predict(model, train)[1]
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    gate_change = (after - initial).abs().max().item()
    if diagnostics["first_gradient_norm"] <= 0 or not changed or gate_change == 0:
        raise RuntimeError(f"Train-only preflight did not learn: {kind}")
    return {"status": "passed", "split": "train", "targets": len(train["z"]), "epochs": 1,
        "parameters_changed": changed, "maximum_train_gate_change": gate_change,
        "initial_gate_mean": initial.double().mean().item(), "final_train_gate": gate_stats(model, train),
        "diagnostics": diagnostics, "reused_in_main_fit": False}


def fit_gate(kind, train, validation, policy, progress):
    model = new_gate(train["features"].shape[-1], kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=policy["learning_rate"], weight_decay=0)
    generator = torch.Generator().manual_seed(policy["shuffle_seed"])
    baseline = metric_values(validation["z"], validation["quantity"], validation["time_nll"])
    initial = metric_values(predict(model, validation)[0], validation["quantity"], validation["time_nll"])
    initial_train_gate = gate_stats(model, train)
    history = [{"epoch": -1, "role": "identity_baseline", **baseline},
               {"epoch": 0, "role": "candidate_initialization", **initial}]
    best, best_state, best_epoch = baseline, None, -1
    if initial["joint_objective"] < best["joint_objective"]:
        best, best_state, best_epoch = initial, copy.deepcopy(model.state_dict()), 0
    stale = 0
    for epoch in range(1, policy["maximum_epochs"] + 1):
        started = time.monotonic()
        diagnostics = train_epoch(model, train, optimizer, generator, policy)
        metrics = metric_values(predict(model, validation)[0], validation["quantity"], validation["time_nll"])
        if metrics["joint_objective"] < best["joint_objective"]:
            best, best_state, best_epoch, stale = metrics, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        row = {"epoch": epoch, "role": "trained_candidate", **metrics, **diagnostics,
            "train_gate": gate_stats(model, train), "best_epoch": best_epoch,
            "epoch_seconds": time.monotonic() - started}
        finite_json(row)
        history.append(row)
        progress(row)
        if epoch >= policy["minimum_epochs"] and stale >= policy["patience"]:
            break
    selected = ResidualGate(model.feature_dim, "identity" if best_epoch == -1 else kind)
    if best_state is not None:
        selected.load_state_dict(best_state, strict=True)
    return selected.eval(), model.eval(), {"kind": kind, "selected_kind": selected.kind,
        "best_epoch": best_epoch, "completed_epochs": epoch, "history": history,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "initial_train_gate": initial_train_gate, "selected_train_gate": gate_stats(selected, train),
        "final_train_gate": gate_stats(model, train)}


def save_checkpoint(path, model, candidate_kind, base_digest, contract_digest):
    torch.save({"model_kind": model.kind, "candidate_kind": candidate_kind,
        "feature_dim": model.feature_dim, "state_dict": model.state_dict(),
        "base_state_sha256": base_digest, "contract_sha256": contract_digest}, path)


def restore_checkpoint(path, base_digest, contract_digest):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["base_state_sha256"] != base_digest or payload["contract_sha256"] != contract_digest:
        raise ValueError("Checkpoint provenance mismatch")
    model = ResidualGate(payload["feature_dim"], payload["model_kind"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def adaptive_decision(smooth, constant):
    scopes, controls = smooth["scopes"], constant["scopes"]
    if any(rows[k]["status"] != "evaluated" for rows in (scopes, controls) for k in ("body_le_p95", "overall")):
        return {"status": "not_assessable", "passes": False}
    checks = {"parent_body_tail_gate": smooth["decision"]["status"] == "exploratory_pass",
        "body_mae_better_than_constant": scopes["body_le_p95"]["candidate"]["qty_mae"] < controls["body_le_p95"]["candidate"]["qty_mae"],
        "joint_no_worse_than_constant": scopes["overall"]["candidate"]["joint_objective"] <= controls["overall"]["candidate"]["joint_objective"]}
    return {"status": "exploratory_pass" if all(checks.values()) else "exploratory_fail",
        "passes": all(checks.values()), "checks": checks, "fresh_training_authorized": False}


def checked_json(path, inputs):
    inputs[str(path)] = sha256_file(path)
    result = load_json(path)
    finite_json(result)
    return result


def read_cache(root, spec, split, inputs):
    if split not in ("train", "validation"):
        raise ValueError("Held-out or unknown split forbidden")
    directory = root / spec["dataset"]
    audit = checked_json(directory / "baseline_audit.json", inputs)
    if audit["model_state_sha256"] != spec["checkpoint_state_sha256"]:
        raise ValueError("Baseline digest mismatch")
    path = directory / f"{split}_cache.pt"
    digest = sha256_file(path)
    if digest != audit["cache"][split]["sha256"]:
        raise ValueError("Cache checksum mismatch")
    inputs[str(path)] = digest
    cache = torch.load(path, map_location="cpu", weights_only=True)
    n = len(cache["z"])
    available = audit["cache"][split]["available_targets"]
    expected = min(65536, available) if split == "train" else available
    if n < 1 or n != expected or n != audit["cache"][split]["selected_targets"] or torch.unique(cache["target_index"]).numel() != n:
        raise ValueError("Cache target count or uniqueness mismatch")
    required = {"features", "z", "projection", "quantity", "time_nll", "history_length",
                "target_index", "series_index", "context_end"}
    if set(cache) != required or cache["features"].shape != (n, 138):
        raise ValueError("Cache feature schema mismatch")
    for name, value in cache.items():
        if len(value) != n or (name != "features" and value.ndim != 1):
            raise ValueError(f"Cache alignment: {name}")
        require_finite(name, value)
    return cache, audit


def execute(source, output, contract, status):
    inputs = {}
    parent = checked_json(ROOT / contract["parent_contract"], inputs)
    launch = checked_json(source / "launch_contract.json", inputs)
    old_source = checked_json(source / "source_manifest.json", inputs)
    if old_source["source_revision"] != contract["parent_execution_revision"] or launch["partial_smoke"]:
        raise ValueError("Parent source revision/scope mismatch")
    for k, v in parent.items():
        if launch[k] != v:
            raise ValueError(f"Parent contract mismatch: {k}")
    if str(torch.__version__) != contract["runtime"]["torch_version"] or torch.get_num_threads() != 1:
        raise ValueError("Use pinned existing CPU runtime /usr/local/bin/python3 -s")
    verify_hashes(ROOT, old_source["files"])
    files = {str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
        str(CONTRACT_PATH.relative_to(ROOT)): sha256_file(CONTRACT_PATH)}
    for path, digest in files.items():
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted experiment code or contract: {path}")
    save_json(output / "source_manifest.json", {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "parent_execution_revision": old_source["source_revision"], "files": files,
        "parent_files": old_source["files"], "python": sys.executable, "torch_version": str(torch.__version__),
        "torch_path": torch.__file__, "device": "cpu", "threads": 1})
    save_json(output / "launch_contract.json", contract)
    specs = parent["datasets"]
    if [s["dataset"] for s in specs] != contract["datasets"]:
        raise ValueError("Dataset grid mismatch")
    all_preflights = {}
    for spec in specs:
        name = spec["dataset"]
        train, _ = read_cache(source, spec, "train", inputs)
        all_preflights[name] = {}
        for kind in KINDS:
            status(stage="train_only_preflight", dataset=name, candidate=kind)
            all_preflights[name][kind] = preflight(kind, train, contract["training"])
    save_json(output / "train_only_preflight.json", all_preflights)
    rows, datasets = [], {}
    for spec in specs:
        name = spec["dataset"]
        directory = output / name
        directory.mkdir()
        train, _ = read_cache(source, spec, "train", inputs)
        validation, audit = read_cache(source, spec, "validation", inputs)
        baseline = metric_values(validation["z"], validation["quantity"], validation["time_nll"])
        if any(abs(v - audit["baseline"][k]) > 1e-5 for k, v in baseline.items()):
            raise AssertionError("Original validation replay mismatch")
        baseline_path = ROOT / spec["artifact_dir"] / "launch_contract.json"
        if not baseline_path.exists() and name == "intermittent_v2":
            baseline_path = ROOT / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/launch_contract.json"
        if sha256_file(baseline_path) != spec["contract_sha256"]:
            raise ValueError("Baseline contract checksum mismatch")
        boundaries = checked_json(baseline_path, inputs)["quantity_contract"]["boundaries"]
        results = {}
        for kind in KINDS:
            status(stage="matched_fit", dataset=name, candidate=kind)
            history_file = directory / f"{kind}_history.jsonl"
            def progress(row):
                with history_file.open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            selected, final, result = fit_gate(kind, train, validation, contract["training"], progress)
            z, gate, correction = predict(selected, validation)
            for tag, model in (("selected", selected), ("final", final)):
                path = directory / f"{kind}_{tag}.pt"
                save_checkpoint(path, model, kind, spec["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH))
                restored = restore_checkpoint(path, spec["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH))
                replay = predict(restored, validation)
                if not all(torch.equal(a, b) for a, b in zip(predict(model, validation), replay)):
                    raise AssertionError("Gate checkpoint replay mismatch")
                result[f"{tag}_checkpoint_sha256"] = sha256_file(path)
            result.update(scopes=summarize(validation, z, gate, correction, boundaries),
                boundaries=boundaries, validation_targets=len(validation["z"]),
                selected_validation_gate=gate_stats(selected, validation),
                checkpoint_replay_exact=True, held_out_test_evaluated=False,
                baseline_state_sha256=spec["checkpoint_state_sha256"])
            result["decision"] = acceptance(result["scopes"], result["best_epoch"], True)
            finite_json(result)
            write_event_deltas(directory / f"{kind}_validation_events.parquet", validation, z, gate, correction)
            save_json(directory / f"{kind}_summary.json", result)
            for scope, values in result["scopes"].items():
                if values["status"] != "evaluated":
                    continue
                for role in ("baseline", "candidate"):
                    rows.append({"dataset": name, "candidate": kind, "scope": scope, "role": role,
                        "count": values["count"], **values[role]})
            results[kind] = result
        results["adaptive_decision"] = adaptive_decision(results["smooth_shrinkage"], results["constant_shrinkage"])
        datasets[name] = results
        save_json(directory / "summary.json", results)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise AssertionError(f"Input changed: {path}")
    verify_hashes(ROOT, old_source["files"])
    verify_hashes(ROOT, files)
    save_json(output / "input_digests.json", inputs)
    pl.DataFrame(rows).write_csv(output / "scope_metrics.csv")
    save_json(output / "summary.json", {"status": "complete", "datasets": datasets, "completed_fits": 8,
        "original_body_tail_passes": sum(v["smooth_shrinkage"]["decision"]["status"] == "exploratory_pass" for v in datasets.values()),
        "adaptive_value_passes": sum(v["adaptive_decision"]["passes"] for v in datasets.values()),
        "cross_dataset_support": all(v["adaptive_decision"]["passes"] for v in datasets.values()),
        "held_out_test_evaluated": False, "fresh_training_authorized": False})
    save_json(output / "output_digests.json", {str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "probe_status.json"})
    status(status="complete", stage="comparison_complete", completed_fits=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_json(CONTRACT_PATH)
    torch.set_num_threads(contract["runtime"]["threads"])
    source = args.artifact or ROOT / contract["source_artifact"]
    run_guarded(args.output.resolve(), lambda status: execute(source.resolve(), args.output.resolve(), contract, status),
                {"held_out_test_evaluated": False, "fresh_backbone_training": False})
