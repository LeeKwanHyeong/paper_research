#!/usr/bin/env python3
"""Blind Instacart design, body eligibility, then conditional frozen inference.

Each phase is a separate invocation. Gate artifacts and immutable hashes must
pass before the next phase can read target labels or load model checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from paper.scripts.hard_lmm_instacart_blind_inputs import select_contexts, read_observed, read_body_labels
from paper.scripts.hard_lmm_instacart_balanced_analysis import design, body_recheck, evaluate
from paper.scripts.run_hard_lmm_query_diagnostic import read, save, digest, check

CONTRACT = ROOT / "paper/contracts/hard_lmm_instacart_balanced_v1.json"
RESULT = ROOT / "paper/results/hard_lmm_instacart_balanced_20260905"
RAW = ROOT / "search_artifacts/hard_lmm_instacart_balanced_20260905"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return str(path.relative_to(ROOT))


def persist_raw(name, value):
    path = RAW / name
    check(not path.exists(), f"Refusing to overwrite {name}")
    torch.save(value, path)
    return relative(path), digest(path)


def verify_hashes(hashes):
    for name, expected in hashes.items():
        check(digest(ROOT / name) == expected, f"Frozen source/input changed: {name}")


def require_gate(marker, expected_stage):
    """Validate the barrier before its caller can access restricted next inputs."""
    check(marker["stage"] == expected_stage, "Wrong gate stage")
    check(marker["passed"] is True, f"{expected_stage} comparability gate did not pass")
    verify_hashes(marker["frozen_hashes"])


def load_inputs():
    manifest = read(RESULT / "design_manifest.json")
    verify_hashes(manifest["verified_hashes"])
    return manifest, torch.load(RAW / "features.pt", weights_only=False), torch.load(RAW / "selection.pt", weights_only=False)


def status(phase, **fields):
    path = RESULT / "execution_manifest.json"
    value = read(path) if path.exists() else {"started_at": timestamp(), "model_parameter_updates": False,
        "server_accessed": False, "validation_or_held_out_rows_used": False,
        "chosen_target_labels_read": False, "checkpoints_restored": False, "prediction_errors_computed": False}
    value.update(status=phase, updated_at=timestamp(), **fields)
    save(path, value)


def design_phase():
    check(not RESULT.exists() and not RAW.exists(), "Refusing to overwrite fixed balanced study")
    contract = read(CONTRACT)
    verified = {relative(CONTRACT): digest(CONTRACT), contract["prior_contract"]: contract["prior_contract_sha256"],
        contract["prior_manifest"]: contract["prior_manifest_sha256"], contract["prior_features"]: contract["prior_features_sha256"],
        contract["data_path"]: contract["data_sha256"]}
    verify_hashes(verified)
    prior_manifest = read(ROOT / contract["prior_manifest"])
    # Only prior source code is read here; old prediction/checkpoint files are not.
    verified.update({name: expected for name, expected in prior_manifest["verified_hashes"].items() if name.endswith(".py")})
    files = [Path(__file__), *(ROOT / "paper/scripts").glob("hard_lmm_instacart_*.py"),
             *(ROOT / "simple_lab_test/search/tests").glob("test_hard_lmm_instacart_*.py")]
    verified.update({relative(path): digest(path) for path in files})
    verify_hashes(verified)
    old_features = torch.load(ROOT / contract["prior_features"], map_location="cpu", weights_only=True)
    exclusion = old_features["series_index"].numpy().astype("<i8")
    exclusion = np.unique(exclusion)
    check(len(exclusion) == contract["excluded_series_count"], "Exclusion count mismatch")
    check(hashlib.sha256(exclusion.tobytes()).hexdigest() == contract["excluded_series_int64_le_sha256"], "Exclusion identity mismatch")
    RESULT.mkdir(parents=True)
    RAW.mkdir(parents=True)
    torch.set_num_threads(contract["runtime"]["threads"])
    status("metadata_selection")
    metadata, selection = select_contexts(ROOT / contract["data_path"], old_features,
                                         maximum_series=contract["population"]["maximum_series"])
    selection_path, selection_hash = persist_raw("selection.pt", selection)
    verified[selection_path] = selection_hash
    save(RESULT / "selection_manifest.json", {"stage": "metadata_only", "completed_at": timestamp(),
        "selection_sha256": selection_hash, "metadata": metadata, "observed_values_read": False,
        "chosen_target_labels_read": False, "checkpoints_restored": False})
    verified[relative(RESULT / "selection_manifest.json")] = digest(RESULT / "selection_manifest.json")
    print("Fixed metadata selection; selected target quantities and gaps remain unread", flush=True)
    features, histories = read_observed(ROOT / contract["data_path"], selection)
    for name, value in (("features.pt", features), ("histories.pt", histories)):
        path, expected = persist_raw(name, value)
        verified[path] = expected
    summary, assignment = design(features)
    path, expected = persist_raw("input_assignment.pt", assignment)
    verified[path] = expected
    save(RESULT / "input_design.json", summary)
    verified[relative(RESULT / "input_design.json")] = digest(RESULT / "input_design.json")
    verify_hashes(verified)
    manifest = {"stage": "input_design_frozen", "completed_at": timestamp(), "verified_hashes": verified,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads()},
        "metadata": metadata, "chosen_target_labels_read": False, "checkpoints_restored": False,
        "prediction_errors_computed": False}
    save(RESULT / "design_manifest.json", manifest)
    passed = bool(summary["passes"])
    save(RESULT / "input_gate.json", {"stage": "input", "passed": passed, "completed_at": timestamp(),
        "frozen_hashes": {**verified, relative(RESULT / "design_manifest.json"): digest(RESULT / "design_manifest.json")},
        "chosen_target_labels_read": False, "checkpoints_restored": False, "prediction_errors_computed": False})
    status("input_gate_passed" if passed else "complete_input_gate_failed", input_gate_passed=passed)
    print(f"Input comparability gate: {passed}", flush=True)


def body_phase():
    check(not (RESULT / "body_gate.json").exists(), "Body gate already executed")
    marker = read(RESULT / "input_gate.json")
    require_gate(marker, "input")
    manifest, features, selection = load_inputs()
    assignment = torch.load(RAW / "input_assignment.pt", weights_only=False)
    summary = read(RESULT / "input_design.json")
    contract = read(CONTRACT)
    positions = np.flatnonzero(assignment["arm"] >= 0)
    # All extremes, including unsupported cells, are required for the original
    # retention denominator. No model state or prediction is admitted here.
    labels = np.full(len(assignment["arm"]), np.nan)
    admitted_labels = read_body_labels(ROOT / contract["data_path"], selection, positions)
    check(np.array_equal(admitted_labels["selection_position"], positions), "Body label position mismatch")
    check(np.array_equal(admitted_labels["target_physical_row_id"],
                         selection["target_physical_row_id"][positions]), "Body label identity mismatch")
    labels[positions] = admitted_labels["quantity"]
    body_summary, body_assignment = body_recheck(features, assignment, labels, summary)
    hashes = {**marker["frozen_hashes"], relative(RESULT / "input_gate.json"): digest(RESULT / "input_gate.json")}
    for name, value in (("body_labels.pt", labels), ("body_assignment.pt", body_assignment)):
        path, expected = persist_raw(name, value)
        hashes[path] = expected
    save(RESULT / "body_design.json", body_summary)
    hashes[relative(RESULT / "body_design.json")] = digest(RESULT / "body_design.json")
    verify_hashes(hashes)
    passed = bool(body_summary["passes"])
    save(RESULT / "body_gate.json", {"stage": "body", "passed": passed, "completed_at": timestamp(),
        "frozen_hashes": hashes, "chosen_target_labels_read": True, "target_quantities_used_only_for_body": True,
        "body_label_count": len(positions), "checkpoints_restored": False, "prediction_errors_computed": False})
    status("body_gate_passed" if passed else "complete_body_gate_failed", chosen_target_labels_read=True,
           body_label_count=len(positions), body_gate_passed=passed)
    print(f"Body comparability gate: {passed}; no checkpoint/prediction access", flush=True)


@torch.no_grad()
def predict_histories(model, histories, positions, labels):
    result = []
    parity = None
    for start in range(0, len(positions), 64):
        index = torch.as_tensor(positions[start:start + 64])
        dts, qty, mask = (histories[key][index] for key in ("dts", "quantities", "mask"))
        local = model._encode_base(dts, qty, mask, memory_write_mask=mask)
        residual, _ = model.lmm.retrieve(local)
        rows, last = torch.arange(len(index)), mask.sum(1) - 1
        z = model.quantity_head(local[rows, last] + residual[rows, last]).squeeze(-1)
        check(bool(torch.isfinite(z).all()), "Nonfinite frozen prediction")
        if start == 0:
            from paper.scripts.hard_lmm_query_diagnostic_features import extract
            full_dt, full_qty, full_mask = dts.clone(), qty.clone(), mask.clone()
            target = last + 1
            full_dt[rows, target] = 0.
            full_qty[rows, target] = torch.as_tensor(labels[positions[:len(index)]], dtype=full_qty.dtype)
            full_mask[rows, target] = True
            reference = extract(model, full_dt, full_mask, full_qty)
            torch.testing.assert_close(z, reference["z"], rtol=1e-5, atol=1e-6)
            parity = {"reference_quantity_logit_max_abs_error": float((z - reference["z"]).abs().max()),
                      "observed_only_vs_original_extraction_path_verified": True}
        result.append(z.cpu())
    return torch.cat(result), parity


def diagnostic_phase():
    check(not (RESULT / "diagnostic_manifest.json").exists(), "Diagnostic already executed")
    require_gate(read(RESULT / "input_gate.json"), "input")
    marker = read(RESULT / "body_gate.json")
    require_gate(marker, "body")
    _, features, _ = load_inputs()
    assignment = torch.load(RAW / "body_assignment.pt", weights_only=False)
    labels = torch.load(RAW / "body_labels.pt", weights_only=False)
    histories = torch.load(RAW / "histories.pt", weights_only=False)
    positions = np.flatnonzero(assignment["rowweight"] > 0)
    check(len(positions) > 0, "Empty frozen evaluation cohort")
    contract = read(CONTRACT)
    torch.set_num_threads(contract["runtime"]["threads"])
    # Checkpoint restore imports/access are deliberately below both barriers.
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
    from paper.scripts.analyze_count_aware_b0_retrieval import checkpoint_path, restore_b0
    from paper.scripts.diagnose_hard_lmm_weighted_static import restore
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    old = read(ROOT / "paper/contracts/hard_lmm_query_diagnostic_v1.json")
    check(digest(ROOT / old["original_registry"]) == old["original_registry_sha256"], "Original registry changed")
    check(digest(ROOT / old["separate_key_audit"]) == old["separate_key_audit_sha256"], "Separate-key registry changed")
    registry, kv = read(ROOT / old["original_registry"]), read(ROOT / old["separate_key_audit"])
    row = next(r for r in registry["datasets"] if r["dataset"] == "insta_market_basket")
    launch = ROOT / row["artifact_dir"] / "launch_contract.json"
    check(digest(launch) == row["contract_sha256"], "Original launch changed")
    hashes = {**marker["frozen_hashes"], relative(RESULT / "body_gate.json"): digest(RESULT / "body_gate.json")}
    models, caches = {}, {}
    for label in ("original", "separate_key"):
        if label == "original":
            path = checkpoint_path(ROOT / row["artifact_dir"], 42)
            expected, expected_state = row["checkpoint_file_sha256"], row["checkpoint_state_sha256"]
            check(digest(path) == expected, "Original checkpoint changed")
            model, _ = restore_b0(path, read(launch), "cpu")
        else:
            path = ROOT / old["separate_key_artifact"] / "insta_market_basket/runs" / KEY_VALUE_BACKBONE / "count_only_log_regression/seed_42/best_val_joint_objective_model.pt"
            expected, expected_state = kv["runs"]["insta_market_basket"]["checkpoint_file_sha256"], kv["runs"]["insta_market_basket"]["checkpoint_state_sha256"]
            check(digest(path) == expected, "Separate-key checkpoint changed")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_route(payload, KEY_VALUE_BACKBONE)
            del payload
            model, _ = restore(path, KEY_VALUE_BACKBONE)
        hashes[relative(path)] = expected
        model.requires_grad_(False).eval()
        check(canonical_state_dict_sha256(model.state_dict()) == expected_state, "Canonical state mismatch")
        z, parity = predict_histories(model, histories, positions, labels)
        log_prediction = F.softplus(z)
        prediction = np.full(len(labels), np.nan)
        residual = np.full(len(labels), np.nan)
        prediction[positions] = log_prediction.expm1().double().numpy()
        residual[positions] = (torch.as_tensor(labels[positions], dtype=torch.float32).log1p() - log_prediction).double().numpy()
        check(np.isfinite(prediction[positions]).all(), "Nonfinite quantity prediction")
        caches[label] = {"prediction": prediction, "quantity": labels.copy(), "log_residual": residual}
        raw_path, expected_hash = persist_raw(f"{label}_predictions.pt", {**caches[label], "positions": positions, "z": z})
        hashes[raw_path] = expected_hash
        check(canonical_state_dict_sha256(model.state_dict()) == expected_state, "Frozen model mutated")
        models[label] = {"state_sha256": expected_state, "checkpoint_state_unchanged": True,
                         "prediction_count": len(positions), "parity": parity}
        print(f"Frozen predictions completed: {label}, {len(positions)} targets", flush=True)
        del model
    outcome = evaluate(features, assignment, caches)
    save(RESULT / "analysis.json", outcome)
    hashes[relative(RESULT / "analysis.json")] = digest(RESULT / "analysis.json")
    verify_hashes(hashes)
    save(RESULT / "diagnostic_manifest.json", {"stage": "error_diagnostic_complete", "completed_at": timestamp(),
        "verified_hashes": hashes, "models": models, "prediction_count_per_model": len(positions),
        "no_post_prediction_design_changes": True, "model_parameter_updates": False, "server_accessed": False})
    status("complete_error_diagnostic", checkpoints_restored=True, prediction_errors_computed=True,
           prediction_count_per_model=len(positions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("design", "body", "diagnose"))
    args = parser.parse_args()
    try:
        {"design": design_phase, "body": body_phase, "diagnose": diagnostic_phase}[args.phase]()
    except Exception as exc:
        if RESULT.exists():
            status("failed", failed_phase=args.phase, error=f"{type(exc).__name__}: {exc}")
        raise
