"""Fixed validation-only gates for the mixed quantity screening experiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from paper.scripts.mixed_quantity_objective import mixed_joint_causal_batch_objective
from paper.scripts.quantity_comparison_engine import _validate_payload
from paper.scripts.time_quantity_diagnostic import _batch_tensors, _sha_json
from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint

CASES = ("B_log_original", "mixed_original", "mixed_matched_original")
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
ACCEPTANCE_POLICY = {
    "primary": {"intermittent_rmse_max_ratio": 1.0, "other_rmse_max_ratio": 0.99,
                "overall_body_lowest_tail_mae_max_ratio": 1.01},
    "last120": {"rmse_overall_body_lowest_mae_max_ratio": 1.01},
    "both_scopes_legacy_time_max_increase": "0.01 * max(1, abs(B_same_checkpoint_time))",
    "absolute_numeric_tolerance": 1e-7,
    "one_common_candidate_for_all_datasets": True,
    "if_both_pass_prefer": "mixed_matched_original",
    "empty_cells": "indeterminate_fail",
    "interpretation": "preregistered engineering screen; not significance or benchmark superiority",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def evaluate_checkpoints(*, model, validation_loader, statistics, objective, arm_dir,
                         quantity_boundaries, device="cpu", budget_check=None, synthetic=False):
    """Replay only newly generated raw-selected and final checkpoints."""
    dataset = validation_loader.dataset
    if synthetic:
        require(type(dataset) is TensorDataset and 1 <= len(dataset) <= 8,
                "Synthetic evaluation must use a bounded TensorDataset")
    else:
        require(getattr(dataset, "target_splits", None) == {"validation"},
                "Validation-only targets required")
    require(len(quantity_boundaries) == 4 and all(math.isfinite(x) for x in quantity_boundaries)
            and list(quantity_boundaries) == sorted(quantity_boundaries), "Invalid quantity boundaries")
    root = Path(arm_dir)
    contract = json.loads((root / "contract.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    require(summary["status"] == "complete" and summary["epochs_completed"] == contract["epochs"],
            "Only completed arms may be evaluated")
    require(summary["contract_sha256"] == _sha_json(contract)
            and summary["condition"] == contract["condition"]
            and contract["condition"]["mixed_objective"] == objective.to_dict(), "Condition identity mismatch")
    require(contract["condition"]["mu"] == statistics.mu
            and contract["condition"]["raw_scale"] == statistics.raw_scale, "Statistics mismatch")
    last = torch_load_checkpoint(root / "last_epoch_state.pt", map_location="cpu")
    _validate_payload(last, contract)
    require(last["history"] == summary["history"] and last["global_step"] == summary["global_step"],
            "Summary/checkpoint history mismatch")
    selected = torch_load_checkpoint(root / "best_raw_quantity_rmse_model.pt", map_location="cpu")
    selector = last["selectors"]["raw_quantity_rmse"]
    require(selected["contract_sha256"] == _sha_json(contract)
            and selected["condition"] == contract["condition"]
            and selected["selector"] == "raw_quantity_rmse", "Selected checkpoint identity mismatch")
    for key in ("best_epoch", "best_value", "global_step", "state_sha256"):
        require(selected[key] == selector[key] == summary["selectors"]["raw_quantity_rmse"][key],
                "Selected checkpoint metadata mismatch")
    model.to(device).eval()
    result = {"evaluation_split": "validation", "held_out_evaluated": False,
              "execution_identity": contract["identity"], "condition": contract["condition"]}
    for scope, payload, epoch, state_hash in (
        ("primary", selected, selected["best_epoch"], selected["state_sha256"]),
        ("last120", last, last["epoch"], last["model_state_sha256"]),
    ):
        require(canonical_state_dict_sha256(payload["model_state_dict"]) == state_hash,
                "Checkpoint state hash mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        totals = {key: 0.0 for key in ("sse", "absolute", "time", "log", "quantity")}
        cells = {key: [0, 0.0] for key in ("body", "lowest", "tail")}
        count, batches, digest = 0, 0, hashlib.sha256()
        with torch.no_grad():
            for batch in validation_loader:
                if budget_check:
                    budget_check("checkpoint_validation")
                dts, mask, quantities = _batch_tensors(batch, torch.device(device))
                digest.update(canonical_state_dict_sha256({"dts": dts, "mask": mask, "quantities": quantities}).encode())
                outputs = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
                                                            statistics=statistics, objective=objective)
                truth, prediction = outputs["true_qty"].double(), outputs["pred_qty"].double()
                error = prediction - truth
                require(bool(torch.isfinite(error).all()), "Nonfinite validation error")
                count += truth.numel()
                batches += 1
                totals["sse"] += error.square().sum().item()
                totals["absolute"] += error.abs().sum().item()
                for key, field in (("time", "time_loss"), ("log", "log_qty_loss"), ("quantity", "quantity_train_loss")):
                    totals[key] += outputs[field].double().sum().item()
                for key, cell_mask in (("body", truth <= quantity_boundaries[2]),
                                       ("lowest", truth <= quantity_boundaries[0]),
                                       ("tail", truth > quantity_boundaries[3])):
                    cells[key][0] += int(cell_mask.sum().item())
                    cells[key][1] += error[cell_mask].abs().sum().item()
        require(count == contract["expected_validation_targets"] and batches == contract["validation_loader"]["batches"],
                "Validation exposure mismatch")
        metrics = {"raw_quantity_rmse": math.sqrt(totals["sse"] / count), "quantity_mae": totals["absolute"] / count,
                   "legacy_time_loss": totals["time"] / count, "log_quantity_mse": totals["log"] / count,
                   "quantity_train_loss": totals["quantity"] / count}
        row = last["history"][epoch - 1]
        for key, value in metrics.items():
            require(math.isfinite(value) and math.isclose(value, row[key], rel_tol=1e-5, abs_tol=1e-5),
                    f"Validation replay mismatch: {key}")
        metrics.update({key + "_mae": total / n if n else None for key, (n, total) in cells.items()})
        result[scope] = {"epoch": epoch, "global_step": row["global_step"], "state_sha256": state_hash,
                         "metrics": metrics, "count": count, "cell_counts": {key: value[0] for key, value in cells.items()},
                         "validation_batch_sha256": digest.hexdigest()}
    return result


def assess_acceptance(dataset_results):
    require(set(dataset_results) == set(DATASETS), "All three datasets required")
    checks, passed = [], {}
    for dataset_id, arms in dataset_results.items():
        require(set(arms) == set(CASES), "Three fixed arms required")
        baseline = arms[CASES[0]]
        for candidate_name in CASES[1:]:
            candidate = arms[candidate_name]
            require(candidate["execution_identity"] == baseline["execution_identity"], "Cross-arm execution identity mismatch")
            for scope in ("primary", "last120"):
                b, c = baseline[scope], candidate[scope]
                require(b["count"] == c["count"] and b["cell_counts"] == c["cell_counts"]
                        and b["validation_batch_sha256"] == c["validation_batch_sha256"], "Cross-arm evaluation population differs")
                require(scope != "last120" or b["epoch"] == c["epoch"] == 120,
                        "Final comparison requires 120 completed epochs")
                keys = ["raw_quantity_rmse", "quantity_mae", "body_mae", "lowest_mae", "legacy_time_loss"]
                if scope == "primary":
                    keys.append("tail_mae")
                for key in keys:
                    bv, cv = b["metrics"][key], c["metrics"][key]
                    valid = all(isinstance(x, (int, float)) and math.isfinite(x) for x in (bv, cv))
                    bound = None
                    if valid:
                        if key == "legacy_time_loss":
                            bound = bv + 0.01 * max(1.0, abs(bv))
                        else:
                            ratio = 1.01
                            if key == "raw_quantity_rmse" and scope == "primary":
                                ratio = 1.0 if dataset_id == DATASETS[0] else 0.99
                            bound = ratio * bv
                    ok = valid and cv <= bound + ACCEPTANCE_POLICY["absolute_numeric_tolerance"]
                    checks.append({"dataset": dataset_id, "candidate": candidate_name, "scope": scope,
                                   "metric": key, "B": bv, "candidate_value": cv, "upper_bound": bound, "passed": ok})
    for name in CASES[1:]:
        passed[name] = all(row["passed"] for row in checks if row["candidate"] == name)
    chosen = next((name for name in reversed(CASES[1:]) if passed[name]), None)
    return {"status": "complete", "candidate_passed": passed, "recommended_candidate": chosen,
            "checks": checks, "policy": ACCEPTANCE_POLICY, "benchmark_superiority_established": False}
