#!/usr/bin/env python3
"""Frozen bank/selection/averaging diagnosis from existing train-only caches."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from models.TPPs.CountAwareFactory import validate_checkpoint_route
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from paper.scripts.analyze_count_aware_b0_retrieval import checkpoint_path, restore_b0
from paper.scripts.diagnose_hard_lmm_weighted_static import restore
from paper.scripts.hard_lmm_bank_geometry import analyze_bank
from paper.scripts.run_hard_lmm_query_diagnostic import read, save, digest, check, numeric_summary
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

CONTRACT = ROOT / "paper/contracts/hard_lmm_bank_diagnostic_v1.json"
RESULT = ROOT / "paper/results/hard_lmm_bank_diagnostic_20260905"


def time_intercept_effect(base, contribution, limit):
    """Match the upper-only intercept clamp in the legacy training likelihood."""
    check(math.isfinite(limit) and limit > 0, "Invalid time intercept limit")
    check(base.shape == contribution.shape, "Time contribution shape mismatch")
    check(bool(torch.isfinite(base).all() and torch.isfinite(contribution).all()), "Nonfinite time input")
    off = base.clamp(max=limit)
    on = (base + contribution).clamp(max=limit)
    return on - off


def verify_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return float((actual - expected).abs().max())


@torch.no_grad()
def verify_cache_and_heads(model, cache, label):
    values = model.lmm.mem[0]
    indices, weights = cache["indices"], cache["weights"]
    qbank = values @ model.quantity_head.weight[0]
    r = (weights[..., None] * values[indices]).sum(1)
    projection = (weights * qbank[indices]).sum(-1)
    parity = {
        "selected_quantity_projection_max_abs_error": verify_close(qbank[indices], cache["value_projections"]),
        "quantity_memory_projection_max_abs_error": verify_close(projection, cache["projection"]),
        "quantity_logit_max_abs_error": verify_close(model.quantity_head(cache["h"] + r).squeeze(-1), cache["z"]),
    }
    keys = values if label == "original" else model.lmm.memory_keys[0]
    scores = F.normalize(cache["h"], dim=-1) @ F.normalize(keys, dim=-1).T
    selected_scores = scores.gather(1, indices)
    parity["selected_scores_max_abs_error"] = verify_close(selected_scores, cache["scores"])
    parity["top4_scores_max_abs_error"] = verify_close(
        selected_scores.sort(-1).values, scores.topk(indices.shape[1], dim=-1).values.sort(-1).values)
    expected_weights = torch.full_like(weights, 1 / weights.shape[1]) if label == "original" else F.softmax(selected_scores / model.lmm.temperature, dim=-1)
    parity["weights_max_abs_error"] = verify_close(expected_weights, weights)
    check(model.time_head_mode == "legacy_clamped_rmtpp", "Unsupported time likelihood")
    base = model.v_t(cache["h"]).squeeze(-1) + model.b_t
    time_projection = (weights * (values @ model.v_t.weight[0])[indices]).sum(-1)
    check(bool(torch.isfinite(base).all() and torch.isfinite(time_projection).all()), "Nonfinite time reconstruction")
    parity["time_linear_path_max_abs_error"] = verify_close(
        model.v_t(cache["h"] + r).squeeze(-1) + model.b_t, base + time_projection)
    effective = time_intercept_effect(base.double(), time_projection.double(), model.time_intercept_limit)
    timing = {
        "likelihood": model.time_head_mode, "upper_limit": model.time_intercept_limit,
        "base_intercept": numeric_summary(base), "raw_memory_contribution": numeric_summary(time_projection),
        "clamp_effective_memory_contribution": numeric_summary(effective),
        "clamp_active_without_memory_fraction": float((base >= model.time_intercept_limit).double().mean()),
        "clamp_active_with_memory_fraction": float((base + time_projection >= model.time_intercept_limit).double().mean()),
        "clamp_changes_contribution_fraction": float((effective != time_projection.double()).double().mean()),
        "time_nll_evaluated": False,
    }
    return parity, timing


def verify_decomposition(summary):
    for axis, numbers in summary["selected_variance_decomposition"].items():
        check(abs(numbers["closure_error"]) <= 1e-12 * max(1., numbers["total"]),
              f"Variance identity failed: {axis}")


def main():
    check(not RESULT.exists(), "Refusing to overwrite previous bank diagnostic")
    contract = read(CONTRACT)
    prior_path = ROOT / contract["prior_manifest"]
    check(digest(prior_path) == contract["prior_manifest_sha256"], "Prior manifest changed")
    prior = read(prior_path)
    query_contract = read(ROOT / contract["prior_contract"])
    verified = {contract["prior_manifest"]: contract["prior_manifest_sha256"],
                contract["prior_contract"]: prior["contract_sha256"],
                **prior["source_files"], **prior["reference_document_hashes"]}
    for name, expected in verified.items():
        check(digest(ROOT / name) == expected, f"Prior source/reference integrity: {name}")
    for path in (CONTRACT, Path(__file__), ROOT / "paper/scripts/hard_lmm_bank_geometry.py",
                 ROOT / "simple_lab_test/search/tests/test_hard_lmm_bank_geometry.py",
                 ROOT / "simple_lab_test/search/tests/test_hard_lmm_bank_diagnostic.py"):
        verified[str(path.relative_to(ROOT))] = digest(path)
    registry = read(ROOT / query_contract["original_registry"])
    kv_audit = read(ROOT / query_contract["separate_key_audit"])
    torch.set_num_threads(contract["runtime"]["threads"])
    manifest = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(), "platform": platform.platform()},
        "verified_hashes": verified, "datasets": {}, "server_accessed": False,
        "target_labels_used": False, "new_dataset_rows_loaded": False, "held_out_or_validation_rows_used": False,
        "model_parameter_updates": False,
    }
    RESULT.mkdir(parents=True)
    save(RESULT / "execution_manifest.json", manifest)
    output, slot_rows = {}, []
    try:
        for dataset in contract["datasets"]:
            row = next(r for r in registry["datasets"] if r["dataset"] == dataset)
            paths = {"original": checkpoint_path(ROOT / row["artifact_dir"], 42),
                     "separate_key": ROOT / query_contract["separate_key_artifact"] / dataset / "runs" / KEY_VALUE_BACKBONE / "count_only_log_regression/seed_42/best_val_joint_objective_model.pt"}
            output[dataset], manifest["datasets"][dataset] = {}, {}
            identities = None
            for label in contract["models"]:
                path = paths[label]
                info = prior["datasets"][dataset]["models"][label]
                for file in (str(path.relative_to(ROOT)), row["artifact_dir"] + "/launch_contract.json"):
                    expected = prior["datasets"][dataset]["input_hashes"][file]
                    check(digest(ROOT / file) == expected, f"Checkpoint/launch integrity: {file}")
                    verified[file] = expected
                verified[info["cache_path"]] = info["cache_sha256"]
                check(digest(ROOT / info["cache_path"]) == info["cache_sha256"], "Cache integrity")
                full_cache = torch.load(ROOT / info["cache_path"], map_location="cpu", weights_only=True)
                cache = {key: full_cache[key] for key in contract["cache_fields_used"]}
                del full_cache
                check(len(cache["h"]) == 8192, "Unexpected sample count")
                check(cache["indices"].shape == (8192, 4), "Unexpected top4 shape")
                identity_fields = ("target_index", "series_index", "fold", "context_end")
                if identities is None:
                    identities = {key: cache[key].clone() for key in identity_fields}
                else:
                    for key in identity_fields:
                        check(torch.equal(identities[key], cache[key]), f"Paired identity mismatch: {key}")
                if label == "original":
                    model, _ = restore_b0(path, read(ROOT / row["artifact_dir"] / "launch_contract.json"), "cpu")
                    expected_state = row["checkpoint_state_sha256"]
                else:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    validate_checkpoint_route(payload, KEY_VALUE_BACKBONE)
                    model, _ = restore(path, KEY_VALUE_BACKBONE)
                    del payload
                    expected_state = kv_audit["runs"][dataset]["checkpoint_state_sha256"]
                model.requires_grad_(False).eval()
                before = canonical_state_dict_sha256(model.state_dict())
                check(before == expected_state, "Canonical state mismatch")
                parity, timing = verify_cache_and_heads(model, cache, label)
                values, qhead, thead = model.lmm.mem[0].detach(), model.quantity_head.weight[0].detach(), model.v_t.weight[0].detach()
                check(values.shape == (64, 64), "Unexpected bank shape")
                geometry, slots = analyze_bank(values, cache["indices"], cache["weights"], qhead, thead)
                verify_decomposition(geometry)
                geometry.update(time_intercept=timing, train_folds={})
                for fold in (0, 1):
                    mask = cache["fold"] == fold
                    summary, _ = analyze_bank(values, cache["indices"][mask], cache["weights"][mask], qhead, thead)
                    verify_decomposition(summary)
                    geometry["train_folds"][str(fold)] = {key: summary[key] for key in ("n_targets", "active_slots", "selected_variance_decomposition")}
                    geometry["train_folds"][str(fold)]["selected_mass"] = summary["distributions"]["selected_mass"]
                output[dataset][label] = geometry
                for slot in range(len(values)):
                    slot_rows.append({"dataset": dataset, "model": label, "slot": slot,
                                      **{key: value[slot].item() for key, value in slots.items()}})
                check(canonical_state_dict_sha256(model.state_dict()) == before, "Frozen state mutated")
                manifest["datasets"][dataset][label] = {"checkpoint": str(path.relative_to(ROOT)), "state_sha256": before,
                    "cache": info["cache_path"], "parity": parity, "checkpoint_state_unchanged": True,
                    "n_targets": len(cache["h"]), "paired_identity_verified": True}
                print(f"Completed {dataset}/{label}: {len(cache['h'])} cached train targets", flush=True)
                del model, cache
        for name, expected in verified.items():
            check(digest(ROOT / name) == expected, f"Source/input changed during diagnosis: {name}")
        save(RESULT / "analysis.json", output)
        with (RESULT / "per_slot.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(slot_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(slot_rows)
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
            output_hashes={name: digest(RESULT / name) for name in ("analysis.json", "per_slot.csv")})
        save(RESULT / "execution_manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(RESULT / "execution_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
