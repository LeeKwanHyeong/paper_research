#!/usr/bin/env python3
"""Registered head-capacity/objective diagnostic on immutable Hard-LMM caches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

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
from paper.scripts.run_hard_lmm_smooth_shrinkage import checked_json, read_cache
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json

CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_readout_factorial_v1.json"
HEADS = ("constant", "linear", "mlp")
OBJECTIVES = ("log_mse", "raw_mae")
SELECTORS = {"joint": "joint_objective", "body": "body_mae"}


def inputs(features, z, projection):
    return torch.cat((features, z[:, None], projection[:, None]), dim=1).detach()


class Readout(nn.Module):
    def __init__(self, feature_dim, kind):
        super().__init__()
        self.kind, self.feature_dim = kind, feature_dim
        self.register_buffer("center", torch.zeros(feature_dim + 2))
        self.register_buffer("scale", torch.ones(feature_dim + 2))
        if kind == "constant":
            self.offset = nn.Parameter(torch.zeros(()))
        elif kind == "linear":
            self.network = nn.Linear(feature_dim + 2, 1)
            nn.init.zeros_(self.network.weight)
            nn.init.zeros_(self.network.bias)
        elif kind == "mlp":
            self.network = nn.Sequential(nn.Linear(feature_dim + 2, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        else:
            raise ValueError(f"Unknown readout {kind}")

    def forward(self, features, z, projection):
        if self.kind == "constant":
            correction = self.offset.expand_as(z)
        else:
            x = (inputs(features, z, projection) - self.center) / self.scale
            correction = self.network(x).squeeze(-1)
        return z.detach() + correction, torch.ones_like(z), correction


def new_readout(kind, train, seed=42):
    torch.manual_seed(seed)
    model = Readout(train["features"].shape[1], kind)
    x = inputs(train["features"], train["z"], train["projection"])
    model.center.copy_(x.mean(0))
    model.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-6))
    for buffer in model.buffers():
        require_finite("train normalizer", buffer)
    return model


def quantity_loss(z, q, objective):
    if (q < 0).any():
        raise ValueError("Negative quantity is outside the frozen count contract")
    if objective == "log_mse":
        loss = (F.softplus(z) - q.log1p()).square().mean()
    elif objective == "raw_mae":
        loss = (F.softplus(z).expm1() - q).abs().mean()
    else:
        raise ValueError(f"Unknown objective {objective}")
    require_finite("train loss", loss)
    return loss


def train_epoch(model, cache, objective, optimizer, generator, policy):
    total, norms, zero, batches = 0., 0., 0, 0
    first = None
    for idx in torch.randperm(len(cache["z"]), generator=generator).split(policy["batch_size"]):
        z = model(cache["features"][idx], cache["z"][idx], cache["projection"][idx])[0]
        loss = quantity_loss(z, cache["quantity"][idx], objective)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(model.parameters(), policy["gradient_clip"], error_if_nonfinite=True))
        first = norm if first is None else first
        optimizer.step()
        for p in model.parameters():
            require_finite("trained parameter", p)
        total += float(loss.detach()) * len(idx)
        norms += norm
        zero += norm == 0
        batches += 1
    return {"optimization_loss": total / len(cache["z"]), "gradient_norm_mean": norms / batches,
        "first_gradient_norm": first, "zero_gradient_batches": zero, "batches": batches}


@torch.no_grad()
def evaluate(model, cache, p95):
    z = predict(model, cache)[0]
    metrics = metric_values(z, cache["quantity"], cache["time_nll"])
    mask = cache["quantity"] <= p95
    if not mask.any():
        raise ValueError("Empty body stratum")
    metrics["body_mae"] = metric_values(z[mask], cache["quantity"][mask], cache["time_nll"][mask])["qty_mae"]
    return metrics


def optimizer_for(model, policy):
    return torch.optim.Adam(model.parameters(), lr=policy["learning_rate"], weight_decay=policy["weight_decay"])


def preflight(kind, objective, train, policy):
    model = new_readout(kind, train, policy["seed"])
    initial = predict(model, train)[0]
    if not torch.equal(initial, train["z"]):
        raise AssertionError("Nonidentity initialization")
    before = copy.deepcopy(model.state_dict())
    diagnostics = train_epoch(model, train, objective, optimizer_for(model, policy),
        torch.Generator().manual_seed(policy["shuffle_seed"]), policy)
    after = predict(model, train)[0]
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    difference = (after - initial).abs().max().item()
    if diagnostics["first_gradient_norm"] <= 0 or not changed or difference == 0:
        raise RuntimeError(f"Inactive train-only preflight: {kind}/{objective}")
    return {"status": "passed", "split": "train", "targets": len(initial), "epochs": 1,
        "initialization_exact": True, "parameters_changed": changed, "max_logit_change": difference,
        "reused_in_main_fit": False, "diagnostics": diagnostics}


def fit(kind, objective, train, validation, p95, policy, progress):
    model = new_readout(kind, train, policy["seed"])
    if not torch.equal(predict(model, validation)[0], validation["z"]):
        raise AssertionError("Main fit did not reset to identity")
    initial = {"epoch": 0, **evaluate(model, validation, p95), "train": evaluate(model, train, p95)}
    history = [initial]
    best = {tag: initial for tag in SELECTORS}
    states = {tag: copy.deepcopy(model.state_dict()) for tag in SELECTORS}
    optimizer = optimizer_for(model, policy)
    generator = torch.Generator().manual_seed(policy["shuffle_seed"])
    for epoch in range(1, policy["maximum_epochs"] + 1):
        start = time.monotonic()
        diagnostics = train_epoch(model, train, objective, optimizer, generator, policy)
        row = {"epoch": epoch, **evaluate(model, validation, p95), **diagnostics,
            "train": evaluate(model, train, p95), "epoch_seconds": time.monotonic() - start}
        for tag, metric in SELECTORS.items():
            if row[metric] < best[tag][metric]:
                best[tag], states[tag] = row, copy.deepcopy(model.state_dict())
        row["best_epochs"] = {tag: best[tag]["epoch"] for tag in SELECTORS}
        finite_json(row)
        history.append(row)
        progress(row)
    selected = {}
    for tag in SELECTORS:
        selected[tag] = Readout(model.feature_dim, kind).eval()
        selected[tag].load_state_dict(states[tag], strict=True)
    return selected, model.eval(), {"head": kind, "objective": objective, "history": history,
        "completed_epochs": epoch, "best_epochs": {tag: best[tag]["epoch"] for tag in SELECTORS},
        "trainable_parameters": sum(p.numel() for p in model.parameters())}


def save_checkpoint(path, model, base_digest, contract_digest):
    torch.save({"head": model.kind, "feature_dim": model.feature_dim,
        "state_dict": model.state_dict(), "base_state_sha256": base_digest,
        "contract_sha256": contract_digest}, path)


def restore_checkpoint(path, base_digest, contract_digest):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["base_state_sha256"] != base_digest or payload["contract_sha256"] != contract_digest:
        raise ValueError("Checkpoint provenance mismatch")
    model = Readout(payload["feature_dim"], payload["head"])
    model.load_state_dict(payload["state_dict"], strict=True)
    for value in model.state_dict().values():
        require_finite("checkpoint", value)
    if (model.scale <= 0).any():
        raise ValueError("Invalid normalizer scale")
    return model.eval()


def baseline_contract_path(spec):
    path = ROOT / spec["artifact_dir"] / "launch_contract.json"
    if not path.exists() and spec["dataset"] == "intermittent_v2":
        path = ROOT / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/launch_contract.json"
    if sha256_file(path) != spec["contract_sha256"]:
        raise ValueError("Original launch contract changed")
    return path


def execute(source, output, contract, status):
    digests = {}
    parent = checked_json(ROOT / contract["parent_contract"], digests)
    old_launch = checked_json(source / "launch_contract.json", digests)
    old_manifest = checked_json(source / "source_manifest.json", digests)
    if old_manifest["source_revision"] != contract["parent_execution_revision"] or old_launch["partial_smoke"]:
        raise ValueError("Parent revision or scope mismatch")
    if any(old_launch[k] != v for k, v in parent.items()):
        raise ValueError("Parent launch mismatch")
    if str(torch.__version__) != contract["runtime"]["torch_version"] or torch.get_num_threads() != 1:
        raise ValueError("Pinned existing torch2.7.1 CPU runtime required")
    verify_hashes(ROOT, old_manifest["files"])
    own_paths = [CONTRACT_PATH, Path(__file__),
        ROOT / "paper/scripts/validate_hard_lmm_readout_factorial.py",
        ROOT / "paper/scripts/run_hard_lmm_smooth_shrinkage.py",
        ROOT / "paper/scripts/validate_hard_lmm_smooth_shrinkage.py"]
    files = {str(p.relative_to(ROOT)): sha256_file(p) for p in own_paths}
    for path, digest in files.items():
        if hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)).hexdigest() != digest:
            raise ValueError(f"Uncommitted experiment file: {path}")
    save_json(output / "source_manifest.json", {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "files": files,
        "parent_files": old_manifest["files"], "torch_version": str(torch.__version__),
        "python": sys.executable, "torch_path": torch.__file__, "device": "cpu", "threads": 1})
    save_json(output / "launch_contract.json", contract)
    specs = parent["datasets"]
    if [s["dataset"] for s in specs] != contract["datasets"] or list(HEADS) != contract["heads"] or list(OBJECTIVES) != contract["objectives"]:
        raise ValueError("Factorial grid mismatch")
    preflights = {}
    for spec in specs:
        name = spec["dataset"]
        train, _ = read_cache(source, spec, "train", digests)
        preflights[name] = {}
        for kind, objective in itertools.product(HEADS, OBJECTIVES):
            cell = f"{kind}_{objective}"
            status(stage="train_only_preflight", dataset=name, cell=cell)
            preflights[name][cell] = preflight(kind, objective, train, contract["training"])
    save_json(output / "train_only_preflight.json", preflights)
    datasets = {}
    for spec in specs:
        name = spec["dataset"]
        directory = output / name
        directory.mkdir()
        train, _ = read_cache(source, spec, "train", digests)
        validation, audit = read_cache(source, spec, "validation", digests)
        original = metric_values(validation["z"], validation["quantity"], validation["time_nll"])
        if any(abs(v - audit["baseline"][k]) > 1e-5 for k, v in original.items()):
            raise AssertionError("Original replay mismatch")
        boundaries = checked_json(baseline_contract_path(spec), digests)["quantity_contract"]["boundaries"]
        results = {}
        for kind, objective in itertools.product(HEADS, OBJECTIVES):
            cell = f"{kind}_{objective}"
            status(stage="matched_fit", dataset=name, cell=cell)
            print(f"START {name}/{cell}", flush=True)
            def progress(row):
                with (directory / f"{cell}_history.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            selected, final, result = fit(kind, objective, train, validation, boundaries[2], contract["training"], progress)
            result.update(boundaries=boundaries, validation_targets=len(validation["z"]),
                train_targets=len(train["z"]), baseline_state_sha256=spec["checkpoint_state_sha256"],
                held_out_test_evaluated=False, selections={})
            for tag, model in {**selected, "final": final}.items():
                path = directory / f"{cell}_{tag}.pt"
                save_checkpoint(path, model, spec["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH))
                replay = predict(restore_checkpoint(path, spec["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH)), validation)
                if not all(torch.equal(a, b) for a, b in zip(predict(model, validation), replay)):
                    raise AssertionError("Readout restore mismatch")
                result[f"{tag}_checkpoint_sha256"] = sha256_file(path)
                if tag == "final":
                    continue
                z, gate, correction = replay
                scopes = summarize(validation, z, gate, correction, boundaries)
                result["selections"][tag] = {"best_epoch": result["best_epochs"][tag], "scopes": scopes,
                    "decision": acceptance(scopes, result["best_epochs"][tag], True),
                    "train": evaluate(model, train, boundaries[2])}
                write_event_deltas(directory / f"{cell}_{tag}_events.parquet", validation, z, gate, correction)
            finite_json(result)
            save_json(directory / f"{cell}_summary.json", result)
            results[cell] = result
            print(f"COMPLETE {name}/{cell} epochs={result['best_epochs']}", flush=True)
        datasets[name] = results
    for path, digest in digests.items():
        if sha256_file(Path(path)) != digest:
            raise AssertionError(f"Input modified: {path}")
    verify_hashes(ROOT, old_manifest["files"])
    verify_hashes(ROOT, files)
    save_json(output / "input_digests.json", digests)
    save_json(output / "summary.json", {"status": "complete", "datasets": datasets,
        "completed_fits": 24, "held_out_test_evaluated": False, "fresh_training_authorized": False})
    save_json(output / "output_digests.json", {str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "probe_status.json"})
    status(status="complete", stage="comparison_complete", completed_fits=24)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load_json(CONTRACT_PATH)
    torch.set_num_threads(contract["runtime"]["threads"])
    run_guarded(args.output.resolve(), lambda status: execute(ROOT / contract["source_artifact"],
        args.output.resolve(), contract, status), {"held_out_test_evaluated": False, "fresh_backbone_training": False})
