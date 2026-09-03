#!/usr/bin/env python3
"""Cache-only train gate traces and a train-fitted Taxi constant control."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import polars as pl
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.hard_lmm_frozen_probe import (
    FrozenResidualProbe, metric_values, predict, require_finite, scope_masks, summarize,
)
from paper.scripts.run_hard_lmm_frozen_probe import (
    load_json, run_guarded, save_json, sha256_file, verify_hashes,
)
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json

PROTOCOL = ROOT / "paper/contracts/hard_lmm_probe_mechanism_diagnostics_v1.md"
TAXI = "yellow_trip_hourly"
EPOCHS = 10


def load_cache(root, dataset, split, hashes):
    if split not in ("train", "validation") or (split == "validation" and dataset != TAXI):
        raise ValueError("Only train caches and Taxi validation are permitted")
    directory = root / dataset
    audit = load_json(directory / "baseline_audit.json")
    path = directory / f"{split}_cache.pt"
    digest = sha256_file(path)
    if digest != audit["cache"][split]["sha256"]:
        raise ValueError(f"Cache checksum mismatch: {path}")
    hashes[str(path)] = digest
    cache = torch.load(path, map_location="cpu", weights_only=True)
    n = len(cache["z"])
    if n != audit["cache"][split]["selected_targets"]:
        raise ValueError("Cache target count mismatch")
    if len(torch.unique(cache["target_index"])) != n:
        raise ValueError("Duplicate target indices")
    for name, value in cache.items():
        if len(value) != n:
            raise ValueError(f"Cache row alignment: {name}")
        require_finite(name, value)
    return cache


def checked_json(path, hashes):
    hashes[str(path)] = sha256_file(path)
    result = load_json(path)
    finite_json(result)
    return result


def fraction(mask):
    return mask.double().mean().item()


def activity(score, gate, correction):
    for name, value in (("score", score), ("gate", gate), ("correction", correction)):
        require_finite(name, value)
    return {
        "score_min": score.min().item(), "score_mean": score.double().mean().item(),
        "score_max": score.max().item(), "score_negative_fraction": fraction(score < 0),
        "score_zero_fraction": fraction(score == 0),
        "score_interior_fraction": fraction((score > 0) & (score < 1)),
        "score_ge_one_fraction": fraction(score >= 1),
        "gate_min": gate.min().item(), "gate_mean": gate.double().mean().item(),
        "identity_fraction": fraction(gate == 1),
        "relative_residual_norm_reduction_mean": (1 - gate).double().mean().item(),
        "projected_residual_reduction_abs_mean": correction.abs().double().mean().item(),
        "projected_residual_reduction_abs_max": correction.abs().max().item(),
    }


def loss(z, quantity):
    return (F.softplus(z) - quantity.clamp_min(0).log1p()).square().mean()


def gradient_norm(parameters):
    return sum(p.grad.detach().double().square().sum().item()
               for p in parameters if p.grad is not None) ** 0.5


@torch.no_grad()
def train_snapshot(probe, train, boundaries):
    z, gate, correction = predict(probe, train)
    score = torch.cat([probe.network(x).squeeze(-1) for x in train["features"].split(8192)])
    # d(log-MSE)/d(shrink_fraction) at the original prediction; labels are
    # used only for an oracle diagnostic, never supplied to the network.
    derivative = -2 * (F.softplus(train["z"]) - train["quantity"].clamp_min(0).log1p()) * train["z"].sigmoid() * train["projection"]
    scopes = {}
    for name, mask in scope_masks(train, boundaries).items():
        if not bool(mask.any()):
            scopes[name] = {"status": "empty", "count": 0}
            continue
        desired = derivative[mask] < 0
        scopes[name] = {
            "status": "evaluated", "count": int(mask.sum()),
            **activity(score[mask], gate[mask], correction[mask]),
            "baseline": metric_values(train["z"][mask], train["quantity"][mask], train["time_nll"][mask]),
            "candidate": metric_values(z[mask], train["quantity"][mask], train["time_nll"][mask]),
            "oracle_shrink_helpful_fraction": fraction(desired),
            "oracle_shrink_derivative_mean": derivative[mask].double().mean().item(),
            "inactive_and_oracle_shrink_helpful_fraction": fraction(desired & (score[mask] < 0)),
        }
    return scopes


def trace_shrinkage(train, boundaries, policy, path, epochs=EPOCHS):
    torch.manual_seed(42)
    probe = FrozenResidualProbe(train["features"].shape[-1], "shrinkage")
    optimizer = torch.optim.Adam(probe.parameters(), lr=policy["learning_rate"], weight_decay=0)
    generator = torch.Generator().manual_seed(policy["shuffle_seed"])
    snapshots = [{"epoch": 0, "scopes": train_snapshot(probe, train, boundaries)}]
    history, first_steps = [], []
    with path.open("x") as handle:
        for epoch in range(1, epochs + 1):
            total, zero_batches, batches, total_norm = 0., 0, 0, 0.
            for indices in torch.randperm(len(train["z"]), generator=generator).split(policy["batch_size"]):
                features, base, projection = [train[k][indices] for k in ("features", "z", "projection")]
                score = probe.network(features).squeeze(-1)
                z, gate, correction = probe(features, base, projection)
                objective = loss(z, train["quantity"][indices]) + train["time_nll"][indices].mean()
                require_finite("train objective", objective)
                before = activity(score.detach(), gate.detach(), correction.detach())
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                for p in probe.parameters():
                    if p.grad is not None:
                        require_finite("gradient", p.grad)
                hidden_norm = gradient_norm(probe.network[0].parameters())
                output_norm = gradient_norm(probe.network[-1].parameters())
                norm = float(torch.nn.utils.clip_grad_norm_(probe.parameters(), policy["gradient_clip"], error_if_nonfinite=True))
                optimizer.step()
                with torch.no_grad():
                    after_score = probe.network(features).squeeze(-1)
                    _, after_gate, after_correction = probe(features, base, projection)
                row = {"epoch": epoch, "batch": batches, "targets": len(indices),
                    "objective": float(objective.detach()), "gradient_norm": norm,
                    "hidden_gradient_norm": hidden_norm, "output_gradient_norm": output_norm,
                    "before": before, "after": activity(after_score, after_gate, after_correction)}
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                if epoch == 1 and batches < 8:
                    first_steps.append(row)
                total += row["objective"] * len(indices)
                total_norm += norm
                zero_batches += norm == 0
                batches += 1
            snapshots.append({"epoch": epoch, "scopes": train_snapshot(probe, train, boundaries)})
            history.append({"epoch": epoch, "train_joint_objective": total / len(train["z"]),
                "zero_gradient_batch_fraction": zero_batches / batches,
                "gradient_norm_mean": total_norm / batches, "batches": batches})
    return {"epochs": epochs, "train_targets": len(train["z"]), "history": history,
        "first_steps": first_steps, "snapshots": snapshots, "validation_evaluated": False,
        "trace_sha256": sha256_file(path), "checkpoint_replaced": False}


@torch.no_grad()
def fit_constant(z, quantity, points=1001):
    if points < 3 or points % 2 != 1:
        raise ValueError("Odd grid size including zero required")
    offsets = torch.linspace(-0.05, 0.05, points, dtype=torch.float64)
    # linspace can round the midpoint away from exact zero on some runtimes.
    offsets[points // 2] = 0.
    target = quantity.double().clamp_min(0).log1p()
    curve = []
    for block in offsets.split(32):
        objectives = (F.softplus(z.double()[:, None] + block) - target[:, None]).square().mean(0)
        require_finite("constant train objective", objectives)
        curve.extend({"offset": c.item(), "train_log_mse": v.item()} for c, v in zip(block, objectives))
    selected = min(curve, key=lambda x: (x["train_log_mse"], abs(x["offset"])))
    return selected, curve


@torch.no_grad()
def correction_summary(correction):
    return {"mean": correction.double().mean().item(), "std": correction.double().std(unbiased=False).item(),
        "min": correction.min().item(), "max": correction.max().item(),
        "p05": torch.quantile(correction, .05).item(), "p50": torch.quantile(correction, .5).item(),
        "p95": torch.quantile(correction, .95).item(),
        "near_positive_bound_fraction": fraction(correction >= .049),
        "near_negative_bound_fraction": fraction(correction <= -.049)}


@torch.no_grad()
def taxi_comparison(cache, mlp, offset, boundaries):
    mlp_z, gate, correction = predict(mlp, cache)
    constant_correction = torch.full_like(cache["z"], offset)
    constant_z = cache["z"] + constant_correction
    predictions = {"baseline": F.softplus(cache["z"]).expm1().double(),
        "mlp": F.softplus(mlp_z).expm1().double(), "constant": F.softplus(constant_z).expm1().double()}
    for name, pred in predictions.items():
        require_finite(name, pred)
    scopes = {"mlp": summarize(cache, mlp_z, gate, correction, boundaries),
        "constant": summarize(cache, constant_z, torch.ones_like(gate), constant_correction, boundaries)}
    bins = {}
    q = cache["quantity"].double()
    for name, mask in scope_masks(cache, boundaries).items():
        if not bool(mask.any()):
            bins[name] = {"status": "empty", "count": 0}
            continue
        errors = {k: v[mask] - q[mask] for k, v in predictions.items()}
        bins[name] = {"status": "evaluated", "count": int(mask.sum()),
            "mlp_correction": correction_summary(correction[mask]),
            "prediction_gap_abs_mean": (predictions["mlp"][mask] - predictions["constant"][mask]).abs().mean().item(),
            "signed_bias": {k: e.mean().item() for k, e in errors.items()},
            "absolute_error_reduction_sum": {k: (errors["baseline"].abs() - errors[k].abs()).sum().item() for k in ("mlp", "constant")},
            "squared_error_reduction_sum": {k: (errors["baseline"].square() - errors[k].square()).sum().item() for k in ("mlp", "constant")}}
    overall = bins["overall"]
    for key in ("absolute_error_reduction_sum", "squared_error_reduction_sum"):
        disjoint_sum = sum(bins[k][key]["mlp"] for k in ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99") if bins[k]["status"] != "empty")
        if abs(disjoint_sum - overall[key]["mlp"]) > 1e-6 * max(1., abs(disjoint_sum)):
            raise AssertionError("Disjoint contribution reconciliation failed")
    return {"targets": len(q), "scopes": scopes, "bins": bins, "time_route_changed": False,
        "constant_gain_fraction_of_mlp": {k: (overall[k]["constant"] / overall[k]["mlp"] if overall[k]["mlp"] != 0 else None)
            for k in ("absolute_error_reduction_sum", "squared_error_reduction_sum")}}


def execute(root, output, status):
    inputs = {}
    source = checked_json(root / "source_manifest.json", inputs)
    verify_hashes(ROOT, source["files"])
    launch = checked_json(root / "launch_contract.json", inputs)
    if launch["partial_smoke"]:
        raise ValueError("Full original diagnostic required")
    boundary = torch.tensor(0., requires_grad=True)
    boundary.clamp(0, 1).backward()
    if boundary.grad.item() != 1:
        raise ValueError("Unsupported clamp boundary gradient; use pinned original runtime")
    save_json(output / "execution_manifest.json", {"python": sys.executable,
        "torch_version": str(torch.__version__), "torch_path": torch.__file__,
        "platform": platform.platform(), "threads": torch.get_num_threads(),
        "clamp_zero_gradient": boundary.grad.item(), "protocol_sha256": sha256_file(PROTOCOL),
        "diagnostic_source_sha256": sha256_file(Path(__file__)), "source_revision": source["source_revision"],
        "original_fit_torch_version": launch["torch_version"], "device": "cpu"})
    results = {}
    for spec in launch["datasets"]:
        name = spec["dataset"]
        directory = root / name
        dataset_launch = checked_json(directory / "launch_contract.json", inputs)
        audit = checked_json(directory / "baseline_audit.json", inputs)
        original = checked_json(directory / "shrinkage_summary.json", inputs)
        if audit["model_state_sha256"] != spec["checkpoint_state_sha256"]:
            raise ValueError("Baseline state digest mismatch")
        # This per-dataset probe launch repeats the root contract, not the
        # baseline quantity thresholds. Recover thresholds from hash-pinned source.
        baseline_path = ROOT / spec["artifact_dir"] / "launch_contract.json"
        if not baseline_path.exists() and name == "intermittent_v2":
            baseline_path = ROOT / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/launch_contract.json"
        if sha256_file(baseline_path) != spec["contract_sha256"]:
            raise ValueError("Baseline launch checksum mismatch")
        boundaries = checked_json(baseline_path, inputs)["quantity_contract"]["boundaries"]
        if dataset_launch["probe_training"] != launch["probe_training"]:
            raise ValueError("Probe policy mismatch")
        train = load_cache(root, name, "train", inputs)
        status(stage="train_gate_trace", dataset=name)
        result = trace_shrinkage(train, boundaries, launch["probe_training"], output / f"{name}_train_steps.jsonl")
        original_history = {r["epoch"]: r for r in original["history"] if r["epoch"] > 0}
        gaps = [abs(row["train_joint_objective"] - original_history[row["epoch"]]["train_joint_objective"]) for row in result["history"]]
        result["historical_train_objective_max_absolute_difference"] = max(gaps)
        result["historical_train_objective_exact_match"] = max(gaps) == 0
        results[name] = result
        save_json(output / f"{name}_shrinkage.json", result)
        if name == TAXI:
            status(stage="taxi_train_constant_control", dataset=name)
            selected, curve = fit_constant(train["z"], train["quantity"])
            save_json(output / "taxi_constant_train_selection.json", {"selected": selected, "curve": curve,
                "selection_split": "train", "validation_used_for_selection": False})
            saved = checked_json(directory / "calibration_summary.json", inputs)
            checkpoint = directory / "calibration_best.pt"
            digest = sha256_file(checkpoint)
            if digest != saved["adapter_sha256"]:
                raise ValueError("Calibration checkpoint checksum mismatch")
            inputs[str(checkpoint)] = digest
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["baseline_state_sha256"] != spec["checkpoint_state_sha256"]:
                raise ValueError("Adapter base mismatch")
            mlp = FrozenResidualProbe(payload["feature_dim"], "calibration").eval()
            mlp.load_state_dict(payload["state_dict"], strict=True)
            mlp.requires_grad_(False)
            taxi = {"offset": selected["offset"], "constant_selection_split": "train",
                "mlp_selection_split": "historical_validation", "train": taxi_comparison(train, mlp, selected["offset"], boundaries)}
            validation = load_cache(root, name, "validation", inputs)
            taxi["validation"] = taxi_comparison(validation, mlp, selected["offset"], boundaries)
            for metric in ("qty_mae", "qty_rmse", "time_nll", "joint_objective"):
                actual = taxi["validation"]["scopes"]["mlp"]["overall"]["candidate"][metric]
                if abs(actual - saved["scopes"]["overall"]["candidate"][metric]) > 1e-5:
                    raise AssertionError("Historical Taxi MLP replay differs")
            save_json(output / "taxi_constant_comparison.json", taxi)
            flat = []
            for split in ("train", "validation"):
                for candidate, scopes in taxi[split]["scopes"].items():
                    for scope, row in scopes.items():
                        if row["status"] == "empty":
                            continue
                        for role in ("baseline", "candidate"):
                            flat.append({"split": split, "candidate": candidate, "scope": scope, "role": role,
                                "count": row["count"], **row[role]})
            pl.DataFrame(flat).write_csv(output / "taxi_scope_metrics.csv")
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise AssertionError(f"Input changed during diagnostic: {path}")
    verify_hashes(ROOT, source["files"])
    save_json(output / "input_digests.json", inputs)
    summary = {"status": "complete", "shrinkage": results, "taxi": taxi,
        "input_digests_unchanged": True, "held_out_test_evaluated": False,
        "new_candidate_promoted": False, "backbone_training": False}
    finite_json(summary)
    save_json(output / "summary.json", summary)
    save_json(output / "output_digests.json", {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "probe_status.json"})
    status(status="complete", stage="diagnostics_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_guarded(args.output.resolve(), lambda status: execute(args.artifact.resolve(), args.output.resolve(), status),
                {"diagnostic_only": True, "held_out_test_evaluated": False})
