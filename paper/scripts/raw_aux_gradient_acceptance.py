"""Validation-only replay and frozen engineering gates for raw gradient control.

The controller is never evaluated on validation. All loss values are the
uncapped reference objective at the selected weights. No training is launched.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler, TensorDataset

from paper.scripts import intermittent_cell_statistics as cells
from paper.scripts.intermittent_cell_gradients import preserved_model_state, preserved_rng
from paper.scripts.mixed_quantity_objective import MixedQuantityObjective, mixed_joint_causal_batch_objective
from paper.scripts.quantity_comparison_engine import _validate_payload
from paper.scripts.time_quantity_diagnostic import _batch_tensors, _loader_generators, _sha_json
from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint

DESIGN_SHA256 = "94308f2eab435c460b248ce614f3d31f58eb979a8dddd9cc7c2070574ec06145"
DESIGN_PATH = Path(__file__).resolve().parents[1] / "contracts/raw_aux_gradient_control_design_v1.json"
CALIBRATION_SHA256 = "a3e3f9bb6977b5d8bac0fb635c7ba6355a42ec0357362257c59a54e97de0a884"
VALIDATION_INPUT_SHA256 = "d36022747ab2da08a22afaae6aa197a863755bff679dd21100e32c17de869b57"
CASES = ("B_log_original", "mixed_original", "mixed_raw_capped_original")
SCOPES = ("primary", "last120")
PARTITIONS = ("quantity_cells", "history_cells", "cross_cells")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_design(path=None):
    design = json.loads(Path(path or DESIGN_PATH).read_text())
    require(_canonical(design) == DESIGN_SHA256, "Frozen raw auxiliary design hash mismatch")
    return design


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _identity(dataset, synthetic):
    if synthetic:
        n = len(dataset)
        return {"series_ids": np.asarray(["synthetic"] * n),
                "target_position": np.arange(n, dtype=np.int64),
                "target_seq": np.arange(n, dtype=np.int64)}
    return {"series_ids": np.asarray([str(dataset.parts[p]) for p, _ in dataset.index]),
            "target_position": np.asarray([i + 1 for _, i in dataset.index], dtype=np.int64),
            "target_seq": np.asarray([int(dataset.seq_lists[p][i + 1]) for p, i in dataset.index], dtype=np.int64),
            "true_qty": np.asarray([dataset.val_lists[p][i + 1] for p, i in dataset.index], dtype=np.float32)}


def _collect(model, loader, statistics, objective, device, budget_check, synthetic):
    columns, digest = {}, hashlib.sha256()
    batches = 0
    with torch.no_grad():
        for batch in loader:
            if budget_check:
                budget_check("checkpoint_validation")
            dts, mask, quantities = _batch_tensors(batch, torch.device(device))
            digest.update(canonical_state_dict_sha256({"dts": dts, "mask": mask, "quantities": quantities}).encode())
            outputs = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
                                                        statistics=statistics, objective=objective)
            for key in ("true_qty", "pred_qty", "history_length", "time_loss", "log_qty_loss",
                        "raw_qty_loss", "quantity_train_loss", "objective_loss"):
                value = outputs[key]
                require(value.ndim == 1 and value.numel() == dts.shape[0]
                        and bool(torch.isfinite(value).all()), f"Invalid validation vector: {key}")
                columns.setdefault(key, []).append(value.detach().cpu().numpy())
            require(torch.equal(outputs["history_length"].long(), mask.bool().sum(1) - 1),
                    "History must exclude target")
            batches += 1
    require(batches > 0, "Empty validation loader")
    rows = {key: np.concatenate(value) for key, value in columns.items()}
    identity = _identity(loader.dataset, synthetic)
    if "true_qty" in identity:
        require(np.array_equal(identity.pop("true_qty"), rows["true_qty"]), "Validation target order mismatch")
    rows.update(identity)
    # This fingerprints actual ordered target identity, truth and target-excluded history.
    paired = {key: rows[key].tolist() for key in cells.PAIR_FIELDS}
    return rows, batches, digest.hexdigest(), _canonical(paired)


def _metrics(summary):
    all_rows = summary["overall"]
    losses = all_rows["losses"]
    q = summary["quantity_cells"]
    body_n = sum(row["count"] for row in q[:3])
    return {"raw_quantity_rmse": all_rows["RMSE"], "quantity_mae": all_rows["MAE"],
            "legacy_time_loss": losses["time_loss"]["mean"],
            "log_quantity_mse": losses["log_qty_loss"]["mean"],
            "raw_quantity_scaled_mse": losses["raw_qty_loss"]["mean"],
            "quantity_train_loss": losses["quantity_train_loss"]["mean"],
            "reference_uncapped_joint_loss": losses["objective_loss"]["mean"],
            "body_mae": sum(row["absolute_error_sum"] for row in q[:3]) / body_n if body_n else None,
            "lowest_mae": q[0]["MAE"], "tail_mae": q[4]["MAE"]}


def _condition(contract, objective, arm_id, design, synthetic):
    condition = contract["condition"]
    require(arm_id in CASES and condition["case"] == "B_log_original", "Unexpected arm/case")
    require(condition["mixed_objective"] == objective.to_dict(), "Objective identity mismatch")
    expected_name = "B_log_original" if arm_id == CASES[0] else "mixed_original"
    require(objective.name == expected_name and objective.quantity_scale == 1.0,
            "Underlying objective must retain c=1")
    if arm_id == CASES[2]:
        from paper.scripts.raw_aux_gradient_control import RawAuxGradientControl
        require(condition.get("raw_aux_control") == RawAuxGradientControl().to_dict(),
                "Capped arm requires raw_aux_control identity")
    else:
        require("raw_aux_control" not in condition, "Uncapped controls must not contain controller metadata")
    if not synthetic:
        expected_alpha = 0.0 if arm_id == CASES[0] else design["losses"]["alpha"]
        require(objective.alpha == expected_alpha and objective.calibration_sha256 == CALIBRATION_SHA256,
                "Frozen alpha/calibration mismatch")


def evaluate_checkpoints(*, model, validation_loader, statistics, objective, arm_dir, arm_id,
                         quantity_boundaries=(2, 31, 46, 187), history_boundaries=(64, 128),
                         device="cpu", budget_check=None, synthetic=False, design_path=None):
    """Replay raw-selected and final states without controller gradients.

Synthetic mode is explicitly limited to <=8 TensorDataset rows and <=2 CPU
epochs; it cannot produce a production acceptance result.
"""
    design = load_design(design_path)
    require(list(quantity_boundaries) == design["partitions"]["quantity_boundaries"]
            and list(history_boundaries) == design["partitions"]["history_boundaries"], "Frozen boundaries changed")
    dataset = validation_loader.dataset
    require(type(validation_loader.sampler) is SequentialSampler and validation_loader.num_workers == 0
            and not validation_loader.drop_last, "Validation requires sequential complete single-process loader")
    if synthetic:
        require(type(dataset) is TensorDataset and 1 <= len(dataset) <= 8 and str(device) == "cpu",
                "Synthetic evaluation requires <=8 CPU TensorDataset rows")
    else:
        require(getattr(dataset, "target_splits", None) == {"validation"}, "Validation-only targets required")
    root = Path(arm_dir)
    contract = json.loads((root / "contract.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    _condition(contract, objective, arm_id, design, synthetic)
    require(summary["status"] == "complete" and summary["epochs_completed"] == contract["epochs"], "Completed arm required")
    require(summary["contract_sha256"] == _sha_json(contract) and summary["condition"] == contract["condition"],
            "Summary contract identity mismatch")
    require(summary.get("evaluation_scope") == "validation_only" and summary.get("held_out_test_evaluated") is False,
            "Summary split identity mismatch")
    require(contract["condition"]["mu"] == statistics.mu and contract["condition"]["raw_scale"] == statistics.raw_scale,
            "Statistics mismatch")
    if synthetic:
        require(1 <= contract["epochs"] <= 2, "Synthetic replay allows at most two epochs")
    else:
        exposure = design["initialization_and_exposure"]
        require(contract["epochs"] == 120 and contract["seed"] == 42
                and contract["expected_global_steps"] == exposure["steps_per_arm"]
                and contract["initial_state_sha256"] == exposure["initial_state_sha256"]
                and contract["expected_validation_targets"] == 86285
                and contract["expected_train_targets"] == 393824, "Production exposure/initialization mismatch")
        require(statistics.mu == exposure["statistics"]["train_log_mean"]
                and statistics.raw_scale == design["losses"]["raw_scale"], "Frozen train statistics mismatch")
    if budget_check:
        budget_check("checkpoint_validation")
    last = torch_load_checkpoint(root / "last_epoch_state.pt", map_location="cpu")
    _validate_payload(last, contract)
    require(last["history"] == summary["history"] and last["global_step"] == summary["global_step"]
            and last["model_state_sha256"] == summary["last_state_sha256"], "Last checkpoint/summary mismatch")
    # Both selectors are validated by _validate_payload and retained verbatim.
    selectors = {name: {key: value for key, value in selector.items() if key != "model_state_dict"}
                 for name, selector in last["selectors"].items()}
    require(selectors == summary["selectors"], "Both selector metadata must match")
    selected = torch_load_checkpoint(root / "best_raw_quantity_rmse_model.pt", map_location="cpu")
    selector = last["selectors"]["raw_quantity_rmse"]
    require(selected["contract_sha256"] == _sha_json(contract) and selected["condition"] == contract["condition"]
            and selected["selector"] == "raw_quantity_rmse", "Selected checkpoint identity mismatch")
    for key in ("best_epoch", "best_value", "global_step", "state_sha256"):
        require(selected[key] == selector[key], "Selected checkpoint metadata mismatch")
    result = {"schema": "raw_aux_checkpoint_validation_v1", "design_sha256": DESIGN_SHA256,
              "arm_id": arm_id, "synthetic": synthetic, "evaluation_split": "validation", "held_out_evaluated": False,
              "execution_identity": contract["identity"], "condition": contract["condition"],
              "initial_state_sha256": contract["initial_state_sha256"], "selectors": selectors,
              "training_contract_sha256": _sha_json(contract),
              "training_contract_common_sha256": _sha_json({key: value for key, value in contract.items() if key != "condition"}),
              "training_exposure": {key: contract[key] for key in ("epochs", "seed", "expected_global_steps",
                    "expected_train_targets", "expected_validation_targets")},
              "train_batch_order_sha256": [row["train_batch_order_sha256"] for row in last["history"]],
              "reference_loss_semantics": "uncapped validation reference; controller disabled; no validation gradients"}
    original_device = next(model.parameters()).device
    original_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    generators = [(g, g.get_state().clone()) for g in _loader_generators(validation_loader).values()]
    try:
        model.to(device)
        with preserved_rng():
            for scope, payload, epoch, state_hash in (
                ("primary", selected, selected["best_epoch"], selected["state_sha256"]),
                ("last120", last, last["epoch"], last["model_state_sha256"]),
            ):
                require(canonical_state_dict_sha256(payload["model_state_dict"]) == state_hash, "Checkpoint tensor hash mismatch")
                model.load_state_dict(payload["model_state_dict"], strict=True)
                with preserved_model_state(model):
                    model.eval()
                    rows, batches, batch_hash, target_hash = _collect(model, validation_loader, statistics, objective,
                                                                     device, budget_check, synthetic)
                require(len(rows["true_qty"]) == contract["expected_validation_targets"]
                        and batches == contract["validation_loader"]["batches"], "Validation exposure mismatch")
                if not synthetic:
                    require(batch_hash == VALIDATION_INPUT_SHA256, "Frozen production validation input mismatch")
                cell_summary = cells.summarize_rows(rows, quantity_boundaries, history_boundaries)
                metrics = _metrics(cell_summary)
                history = last["history"][epoch - 1]
                for key in ("raw_quantity_rmse", "quantity_mae", "legacy_time_loss", "log_quantity_mse",
                            "raw_quantity_scaled_mse", "quantity_train_loss"):
                    require(_finite(metrics[key]) and math.isclose(metrics[key], history[key], rel_tol=1e-5, abs_tol=1e-5),
                            "Validation replay mismatch: " + key)
                require(math.isclose(metrics["reference_uncapped_joint_loss"], history["objective_loss"], rel_tol=1e-5, abs_tol=1e-5),
                        "Uncapped validation reference replay mismatch")
                result[scope] = {"epoch": epoch, "global_step": history["global_step"], "state_sha256": state_hash,
                                 "count": len(rows["true_qty"]), "batches": batches,
                                 "validation_batch_sha256": batch_hash, "ordered_target_sha256": target_hash,
                                 "metrics": metrics, "partitions": cell_summary, "replay_passed": True}
    finally:
        model.to(original_device)
        model.load_state_dict(original_state, strict=True)
        for module, training in modes:
            module.training = training
        for generator, state in generators:
            generator.set_state(state)
    return result


def _audit_summary(summary):
    require(summary["conservation_audit"]["passed"] is True, "Missing conservation audit")
    require([len(summary[label]) for label in PARTITIONS] == [5, 3, 15], "Incomplete partition cells")
    require(summary["quantity_boundaries"] == [2, 31, 46, 187] and summary["history_boundaries"] == [64, 128],
            "Summary bin boundaries differ")
    for label in PARTITIONS:
        key = "cell_index" if label == "cross_cells" else "bin_index"
        require([row[key] for row in summary[label]] == list(range(len(summary[label]))), "Partition order mismatch")
    overall = summary["overall"]
    require(type(overall["count"]) is int and overall["count"] > 0, "Empty validation population")
    for row in [overall] + [row for label in PARTITIONS for row in summary[label]]:
        require(type(row["count"]) is int and row["count"] >= 0, "Invalid cell count")
        for key in cells.SUM_FIELDS:
            require(_finite(row[key]), "Nonfinite cell accounting")
        for key in cells.COUNT_FIELDS:
            require(type(row[key]) is int and row[key] >= 0, "Invalid sign count")
        require(row["raw_squared_error_sum"] >= 0 and row["absolute_error_sum"] >= 0, "Negative error accounting")
        require(row["under_absolute_error_sum"] >= 0 and row["over_absolute_error_sum"] >= 0,
                "Negative directional absolute error")
        require(math.isclose(row["predicted_quantity_sum"] - row["true_quantity_sum"], row["signed_error_sum"],
                             rel_tol=cells.REL_TOL, abs_tol=cells.ABS_TOL),
                "Predicted/true/signed error accounting mismatch")
        n = row["count"]
        for key, expected in (("RMSE", math.sqrt(row["raw_squared_error_sum"] / n) if n else None),
                              ("MAE", row["absolute_error_sum"] / n if n else None),
                              ("mean_bias", row["signed_error_sum"] / n if n else None)):
            require(row[key] is None if expected is None else _finite(row[key]) and math.isclose(row[key], expected, rel_tol=1e-10, abs_tol=1e-8),
                    "Cell derived metric mismatch")
        for loss in row["losses"].values():
            require(_finite(loss["sum"]), "Nonfinite loss sum")
            require(loss["mean"] is None if not n else _finite(loss["mean"]) and math.isclose(loss["mean"], loss["sum"] / n, rel_tol=1e-10, abs_tol=1e-8),
                    "Loss mean/sum mismatch")
    cells._conservation(overall, {label: summary[label] for label in PARTITIONS})


def _paired(b, c):
    n = b["overall"]["count"]
    def pair(left, right):
        require(left["count"] == right["count"], "Paired cell population mismatch")
        return {"count": left["count"], "B": left, "candidate": right,
                "delta_SSE": right["raw_squared_error_sum"] - left["raw_squared_error_sum"],
                "delta_SSE_divided_by_total_N": (right["raw_squared_error_sum"] - left["raw_squared_error_sum"]) / n,
                "delta_AE": right["absolute_error_sum"] - left["absolute_error_sum"],
                "delta_AE_divided_by_total_N": (right["absolute_error_sum"] - left["absolute_error_sum"]) / n}
    result = {"overall": pair(b["overall"], c["overall"])}
    for label in PARTITIONS:
        result[label] = [pair(x, y) for x, y in zip(b[label], c[label], strict=True)]
        for metric in ("delta_SSE", "delta_AE"):
            require(math.isclose(math.fsum(row[metric] for row in result[label]), result["overall"][metric], rel_tol=1e-10, abs_tol=1e-8),
                    "Paired additive error mismatch")
    return result


def assess_acceptance(arms_by_id, *, synthetic=False, design_path=None):
    """Fail closed on changed identity, missing cells, nonfinite values or gates.

Synthetic inputs never return an eligible production candidate.
"""
    design = load_design(design_path)
    require(set(arms_by_id) == set(CASES), "All three fixed arms required")
    baseline = arms_by_id[CASES[0]]
    for name, arm in arms_by_id.items():
        require(arm["arm_id"] == name and arm["design_sha256"] == DESIGN_SHA256, "Arm/design identity mismatch")
        require(arm["synthetic"] is synthetic and arm["evaluation_split"] == "validation"
                and arm["held_out_evaluated"] is False, "Evaluation authorization/split mismatch")
        objective = MixedQuantityObjective.from_dict(arm["condition"]["mixed_objective"])
        _condition({"condition": arm["condition"]}, objective, name, design, synthetic)
        for key in ("execution_identity", "initial_state_sha256", "training_exposure", "train_batch_order_sha256",
                    "training_contract_common_sha256"):
            require(arm[key] == baseline[key], "Cross-arm identity/exposure mismatch: " + key)
        require(set(arm["selectors"]) == {"raw_quantity_rmse", "legacy_time_loss"}, "Both real selectors required")
        exposure = arm["training_exposure"]
        if not synthetic:
            require(exposure == {"epochs": 120, "seed": 42, "expected_global_steps": 369240,
                                 "expected_train_targets": 393824, "expected_validation_targets": 86285}
                    and arm["initial_state_sha256"] == design["initialization_and_exposure"]["initial_state_sha256"],
                    "Full production scope must match frozen exposure")
        for scope in SCOPES:
            row, b = arm[scope], baseline[scope]
            require(row["replay_passed"] is True and _digest(row["state_sha256"]), "State replay identity invalid")
            for key in ("count", "batches", "validation_batch_sha256", "ordered_target_sha256"):
                require(row[key] == b[key], "Cross-arm target population/order differs: " + key)
            for key in ("validation_batch_sha256", "ordered_target_sha256"):
                require(_digest(row[key]), "Missing population digest")
            if not synthetic:
                require(row["count"] == 86285 and row["batches"] == 675
                        and row["validation_batch_sha256"] == VALIDATION_INPUT_SHA256,
                        "Full production validation population required")
            expected_epochs = arm["training_exposure"]["epochs"]
            require((1 <= expected_epochs <= 2) if synthetic else expected_epochs == 120, "Wrong evaluation epoch scope")
            if scope == "last120":
                require(row["epoch"] == expected_epochs and row["global_step"] == arm["training_exposure"]["expected_global_steps"],
                        "Final comparison requires contract final epoch and equal steps")
            else:
                selected = arm["selectors"]["raw_quantity_rmse"]
                require(row["epoch"] == selected["best_epoch"] and row["global_step"] == selected["global_step"]
                        and row["state_sha256"] == selected["state_sha256"], "Raw selector metadata mismatch")
            _audit_summary(row["partitions"])
            require(row["count"] == row["partitions"]["overall"]["count"], "Count/summary mismatch")
            derived = _metrics(row["partitions"])
            require(row["metrics"] == derived, "Global metrics differ from audited sums")
    checks, comparisons, passed = [], [], {}
    tolerance = design["acceptance"]["numeric_absolute_tolerance"]
    for name in CASES[1:]:
        candidate = arms_by_id[name]
        for scope in SCOPES:
            b, c = baseline[scope], candidate[scope]
            def gate(metric, bv, cv, ratio=1.01, *, count=None, time=False):
                valid = _finite(bv) and _finite(cv) and (count is None or count > 0)
                if not time:
                    valid = valid and bv >= 0 and cv >= 0
                bound = (bv + .01 * max(1., abs(bv)) if time else ratio * bv) + tolerance if valid else None
                checks.append({"candidate": name, "scope": scope, "metric": metric, "B": bv,
                               "candidate_value": cv, "upper_bound_including_tolerance": bound,
                               "passed": bool(valid and cv <= bound)})
            gate("raw_quantity_rmse", b["metrics"]["raw_quantity_rmse"], c["metrics"]["raw_quantity_rmse"],
                 design["acceptance"][scope]["raw_quantity_rmse_max_ratio"])
            for key in ("quantity_mae", "body_mae", "lowest_mae", "tail_mae"):
                gate(key, b["metrics"][key], c["metrics"][key])
            gate("legacy_time_loss", b["metrics"]["legacy_time_loss"], c["metrics"]["legacy_time_loss"], time=True)
            for index, (left, right) in enumerate(zip(b["partitions"]["quantity_cells"], c["partitions"]["quantity_cells"], strict=True)):
                require(left["count"] == right["count"], "Quantity cell population mismatch")
                for key in ("RMSE", "MAE"):
                    gate(f"quantity_bin_{index}_{key}", left[key], right[key], count=left["count"])
            comparisons.append({"baseline": CASES[0], "candidate": name, "scope": scope,
                                "equal_global_step": b["global_step"] == c["global_step"],
                                **_paired(b["partitions"], c["partitions"])})
        passed[name] = all(row["passed"] for row in checks if row["candidate"] == name)
    for scope in SCOPES:
        comparisons.append({"baseline": CASES[1], "candidate": CASES[2], "scope": scope,
                            "equal_global_step": arms_by_id[CASES[1]][scope]["global_step"] == arms_by_id[CASES[2]][scope]["global_step"],
                            **_paired(arms_by_id[CASES[1]][scope]["partitions"], arms_by_id[CASES[2]][scope]["partitions"])})
    return {"schema": "raw_aux_gradient_acceptance_v1", "status": "synthetic_only" if synthetic else "complete",
            "design_sha256": DESIGN_SHA256, "candidate_passed": passed,
            "eligible_candidate": CASES[2] if passed[CASES[2]] and not synthetic else None,
            "checks": checks, "paired_comparisons": comparisons, "policy": deepcopy(design["acceptance"]),
            "benchmark_superiority_established": False, "statistical_significance_established": False}
