"""Three-arm partition audit: B/capped on 5090, uncapped on 5080.

No identity is rewritten to fit the single-host evaluator. Its audited cell
accounting is reused; the authoritative numerical gate is explicitly two-arm.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

from paper.scripts import raw_aux_gradient_acceptance as original
from paper.scripts.mixed_quantity_objective import MixedQuantityObjective
from paper.scripts.time_quantity_diagnostic import _sha_json

CASES = original.CASES
B, U, CAPPED = CASES
OWNERS = {B: "5090", U: "5080", CAPPED: "5090"}
DESIGN_FILE_SHA256 = "34a16247a55c370c5b5dc9c50abd70a6818d8ec2e19ee6addc492425eeb3dde2"
require = original.require


def _partition(contract, design_path):
    require(contract.get("schema") == "raw_aux_gradient_parallel_execution_v1", "Wrong parallel contract schema")
    require(contract.get("design_sha256") == DESIGN_FILE_SHA256, "Partition design file hash differs")
    require(hashlib.sha256(Path(design_path or original.DESIGN_PATH).read_bytes()).hexdigest() == DESIGN_FILE_SHA256,
            "Frozen design file changed")
    require(set(contract["hosts"]) == {"5080", "5090"}, "Exactly two partition hosts required")
    expected = {"5080": [U], "5090": [B, CAPPED]}
    for host, names in expected.items():
        require(contract["hosts"][host]["assigned_arms"] == names, "Fixed unique arm ownership changed")
    source = contract["source"]
    require(isinstance(source["files"], dict) and source["files"]
            and all(isinstance(name, str) and original._digest(value) for name, value in source["files"].items())
            and source["files_sha256"] == original._canonical(source["files"]), "Invalid frozen source closure")


def _runtime_and_contract(arm, partition, start_permit, *, synthetic):
    identity = arm["execution_identity"]
    host = OWNERS[arm["arm_id"]]
    require(identity.get("host_alias") == host, "Actual host identity disagrees with arm owner")
    require(identity.get("execution_contract_sha256") == original._canonical(partition), "Partition execution identity differs")
    require(identity.get("raw_aux_design_sha256") == DESIGN_FILE_SHA256, "Execution design binding differs")
    source = identity["source"]
    require(source["files"] == partition["source"]["files"]
            and source["files_sha256"] == partition["source"]["files_sha256"], "Cross-host source closure differs")
    if synthetic:
        # Synthetic fixtures are never promoted or mistaken for CUDA evidence.
        require(identity.get("runtime") == partition["hosts"][host]["runtime_expected"],
                "Synthetic native runtime differs")
        receipt = None
    else:
        from paper.scripts.raw_aux_parallel_common import validate_permit
        require(isinstance(start_permit, dict), "Historical common start permit required")
        receipt = validate_permit(partition, start_permit, host, now=start_permit["started_at_unix"])
        require(receipt["runtime"] == identity["runtime"]
                and receipt["initial_state_sha256"] == arm["initial_state_sha256"],
                "Observed native qualification runtime/initialization differs")
        if arm.get("runtime_receipt") is not None:
            require(arm["runtime_receipt"] == receipt, "Arm runtime receipt differs from common start permit")
    training = arm["training_contract"]
    require(training["identity"] == identity and training["condition"] == arm["condition"], "Training identity/condition differs")
    require(_sha_json(training) == arm["training_contract_sha256"]
            and _sha_json({k: v for k, v in training.items() if k != "condition"}) == arm["training_contract_common_sha256"],
            "Training contract digest mismatch")
    require(training["engine_source_sha256"] == partition["source"]["files"]["paper/scripts/quantity_comparison_engine.py"],
            "Observed training engine source differs from partition closure")
    if not synthetic:
        require(training["runtime_observed"] == {key: receipt["runtime"][key] for key in ("torch", "numpy", "cuda", "threads")},
                "Actual training runtime differs from native qualification")
    return training


def _validate_arm(arm, name, design, *, synthetic):
    require(arm["arm_id"] == name and arm["design_sha256"] == original.DESIGN_SHA256, "Arm/design identity mismatch")
    require(arm["synthetic"] is synthetic and arm["evaluation_split"] == "validation"
            and arm["held_out_evaluated"] is False, "Evaluation split or synthetic identity differs")
    objective = MixedQuantityObjective.from_dict(arm["condition"]["mixed_objective"])
    original._condition({"condition": arm["condition"]}, objective, name, design, synthetic)
    exposure = arm["training_exposure"]
    epochs = exposure["epochs"]
    require(1 <= epochs <= 2 if synthetic else epochs == 120, "Wrong epoch scope")
    require(len(arm["train_batch_order_sha256"]) == epochs
            and all(original._digest(x) for x in arm["train_batch_order_sha256"]), "Incomplete training batch order evidence")
    if not synthetic:
        require(exposure == {"epochs": 120, "seed": 42, "expected_global_steps": 369240,
                             "expected_train_targets": 393824, "expected_validation_targets": 86285}
                and arm["initial_state_sha256"] == design["initialization_and_exposure"]["initial_state_sha256"],
                "Production exposure/initialization differs")
    selectors = arm["selectors"]
    require(set(selectors) == {"raw_quantity_rmse", "legacy_time_loss"}, "Two actual selectors required")
    require(type(exposure["expected_global_steps"]) is int and exposure["expected_global_steps"] > 0
            and exposure["expected_global_steps"] % epochs == 0, "Invalid step budget")
    steps_per_epoch = exposure["expected_global_steps"] // epochs
    for selected in selectors.values():
        require(type(selected["best_epoch"]) is int and 1 <= selected["best_epoch"] <= epochs
                and selected["global_step"] == selected["best_epoch"] * steps_per_epoch
                and original._digest(selected["state_sha256"]) and original._finite(selected["best_value"]),
                "Invalid actual selector metadata")
    for scope in original.SCOPES:
        row = arm[scope]
        require(row["replay_passed"] is True and original._digest(row["state_sha256"]), "Invalid checkpoint replay identity")
        require(original._digest(row["validation_batch_sha256"]) and original._digest(row["ordered_target_sha256"]),
                "Missing validation population digests")
        require(type(row["epoch"]) is int and 1 <= row["epoch"] <= epochs
                and row["global_step"] == row["epoch"] * steps_per_epoch, "Checkpoint step/epoch differs")
        if scope == "primary":
            selected = selectors["raw_quantity_rmse"]
            require(row["epoch"] == selected["best_epoch"] and row["state_sha256"] == selected["state_sha256"],
                    "Raw-selected state metadata differs")
            require(math.isclose(row["metrics"]["raw_quantity_rmse"], selected["best_value"], rel_tol=1e-5, abs_tol=1e-5),
                    "Raw-selected value differs")
        else:
            require(row["epoch"] == epochs, "Last scope is not completed final epoch")
        if not synthetic:
            require(row["count"] == 86285 and row["batches"] == 675
                    and row["validation_batch_sha256"] == original.VALIDATION_INPUT_SHA256,
                    "Full frozen validation population required")
        else:
            require(type(row["count"]) is int and 1 <= row["count"] <= 8, "Bounded synthetic population required")
        original._audit_summary(row["partitions"])
        require(row["partitions"]["overall"]["count"] == row["count"]
                and row["metrics"] == original._metrics(row["partitions"]), "Audited replay metrics differ")
    for key in ("count", "batches", "validation_batch_sha256", "ordered_target_sha256"):
        require(arm["primary"][key] == arm["last120"][key], "Cross-scope validation population differs")


def _same_population(left, right):
    for scope in original.SCOPES:
        for key in ("count", "batches", "validation_batch_sha256", "ordered_target_sha256"):
            require(left[scope][key] == right[scope][key], "Cross-host validation population/order differs: " + key)


def _scientific_contract(training):
    # Only the explicitly host-dependent identity/runtime may differ. Model,
    # optimizer, all loader settings/generator origins and engine SHA stay exact.
    return {key: value for key, value in training.items() if key not in {"identity", "runtime_observed", "condition"}}


def _gates(baseline, candidate, design):
    checks = []
    tolerance = design["acceptance"]["numeric_absolute_tolerance"]
    for scope in original.SCOPES:
        b, c = baseline[scope], candidate[scope]
        policy = design["acceptance"][scope]
        def gate(metric, bv, cv, ratio=1.01, *, count=None, time=False):
            valid = original._finite(bv) and original._finite(cv) and (count is None or count > 0)
            if not time:
                valid = valid and bv >= 0 and cv >= 0
            bound = (bv + .01 * max(1., abs(bv)) if time else ratio * bv) + tolerance if valid else None
            checks.append({"candidate": CAPPED, "scope": scope, "metric": metric, "B": bv,
                           "candidate_value": cv, "upper_bound_including_tolerance": bound,
                           "passed": bool(valid and cv <= bound)})
        gate("raw_quantity_rmse", b["metrics"]["raw_quantity_rmse"], c["metrics"]["raw_quantity_rmse"],
             policy["raw_quantity_rmse_max_ratio"])
        for key in ("quantity_mae", "body_mae", "lowest_mae", "tail_mae"):
            gate(key, b["metrics"][key], c["metrics"][key])
        gate("legacy_time_loss", b["metrics"]["legacy_time_loss"], c["metrics"]["legacy_time_loss"], time=True)
        for index, (left, right) in enumerate(zip(b["partitions"]["quantity_cells"], c["partitions"]["quantity_cells"], strict=True)):
            require(left["count"] == right["count"], "Quantity cell population differs")
            for metric in ("RMSE", "MAE"):
                gate(f"quantity_bin_{index}_{metric}", left[metric], right[metric], count=left["count"])
    return checks


def _comparison(left, right, *, cross_runtime):
    return [{"baseline": left["arm_id"], "candidate": right["arm_id"], "scope": scope,
             "baseline_host": OWNERS[left["arm_id"]], "candidate_host": OWNERS[right["arm_id"]],
             "cross_runtime": cross_runtime, "descriptive_only": cross_runtime,
             "used_for_candidate_gate": not cross_runtime,
             "equal_global_step": left[scope]["global_step"] == right[scope]["global_step"],
             **original._paired(left[scope]["partitions"], right[scope]["partitions"])} for scope in original.SCOPES]


def assess_partition(arms_by_id, *, partition_contract, start_permit=None, synthetic=False, design_path=None):
    """Assess real B/capped identities; U affects completeness, never the gate.

An omitted/None arm is incomplete. Malformed completed evidence fails closed;
it is not silently relabeled as absent. Synthetic evidence cannot be promoted.
"""
    design = original.load_design(design_path)
    _partition(partition_contract, design_path)
    if not synthetic:
        from paper.scripts.raw_aux_parallel_common import data_entry
        data_entry(partition_contract)
    require(set(arms_by_id) <= set(CASES), "Unexpected fourth arm")
    present = {name: arm for name, arm in arms_by_id.items() if arm is not None}
    training = {}
    for name, arm in present.items():
        _validate_arm(arm, name, design, synthetic=synthetic)
        training[name] = _runtime_and_contract(arm, partition_contract, start_permit, synthetic=synthetic)
        if not synthetic:
            require(arm["execution_identity"]["data"] == design["initialization_and_exposure"]["data_identity"],
                    "Production data identity differs from frozen design")
        require({key: training[name][key] for key in arm["training_exposure"]} == arm["training_exposure"]
                and training[name]["initial_state_sha256"] == arm["initial_state_sha256"],
                "Training contract/exposure evidence differs")
    reference = present.get(B) or next(iter(present.values()), None)
    if reference is not None:
        for name, arm in present.items():
            for key in ("initial_state_sha256", "training_exposure", "train_batch_order_sha256"):
                require(arm[key] == reference[key], "Cross-host scientific invariant differs: " + key)
            require(arm["execution_identity"]["data"] == reference["execution_identity"]["data"], "Cross-host data identity differs")
            require(arm["condition"]["mixed_objective"]["calibration_sha256"] == reference["condition"]["mixed_objective"]["calibration_sha256"],
                    "Cross-host coefficient provenance differs")
            for key in ("mu", "raw_scale"):
                require(arm["condition"][key] == reference["condition"][key], "Cross-host train statistics differ")
            require(_scientific_contract(training[name]) == _scientific_contract(training[reference["arm_id"]]),
                    "Cross-host model/optimizer/loader/source settings differ")
            _same_population(reference, arm)
    ready = B in present and CAPPED in present
    checks, paired = [], []
    passed = None
    if ready:
        for key in ("execution_identity", "training_contract_common_sha256"):
            require(present[B][key] == present[CAPPED][key], "B/capped must share exact native runtime and contract")
        checks = _gates(present[B], present[CAPPED], design)
        passed = all(row["passed"] for row in checks)
        paired.extend(_comparison(present[B], present[CAPPED], cross_runtime=False))
    if U in present:
        if B in present:
            paired.extend(_comparison(present[B], present[U], cross_runtime=True))
        if CAPPED in present:
            paired.extend(_comparison(present[U], present[CAPPED], cross_runtime=True))
    complete = set(present) == set(CASES)
    return {"schema": "raw_aux_gradient_partition_acceptance_v1",
            "status": ("synthetic_only" if synthetic else "complete") if complete else "suite_incomplete",
            "synthetic": synthetic, "partition_contract_sha256": original._canonical(partition_contract),
            "start_permit_sha256": start_permit.get("permit_sha256") if start_permit is not None else None,
            "design_sha256": original.DESIGN_SHA256, "design_file_sha256": DESIGN_FILE_SHA256,
            "completed_arms": [name for name in CASES if name in present],
            "missing_arms": [name for name in CASES if name not in present],
            "primary_gate_status": "complete" if ready else "pending",
            "primary_gate_passed": passed,
            "eligible_candidate": CAPPED if complete and passed and not synthetic else None,
            "uncapped_used_for_candidate_selection": False, "uncapped_gate_status": "descriptive_only",
            "checks": checks, "paired_comparisons": paired,
            "native_execution_identities": {name: deepcopy(arm["execution_identity"]) for name, arm in present.items()},
            "policy": deepcopy(design["acceptance"]),
            "interpretation": "B/capped same-runtime engineering screen; 5080 uncapped comparisons include host effects. No training-equivalence claim.",
            "benchmark_superiority_established": False, "statistical_significance_established": False}
