"""Strict artifact and screening audit for the bounded-Q/K Hard-LMM."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from models.TPPs.CountAwareTitanBoundedQK import (
    BOUNDED_QK_BACKBONE,
    BOUNDED_QK_CONTRACT_ID,
    BOUNDED_QK_KERNEL_KEYS,
    BOUNDED_QK_ROLE,
)
from paper.scripts.audit_hard_lmm_causal_qkv import (
    _finite_tree,
    _require_equal_outputs,
    _require_route_identity,
    _require_same_state,
    _restore_model,
    _state_dict,
    _synthetic_outputs,
)
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    BODY_STRATA,
    DATASETS,
    audit_job,
    read_json,
    require,
    sha256,
    validate_validation_population,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_bounded_qk_screening_v1.json"
CONTRACT_SHA256 = (
    "09464ee99907b7a3f0c3e35637c5bd36e5bb105111f1898d07843a6ca3b72997"
)
VARIANT = "count_only_log_regression"
MONITOR = "validation_raw_quantity_rmse"
SELECTION = "best_validation_raw_quantity_rmse"


def _finite_number(value: Any, *, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    return number


def _require_metric_equal(left: Any, right: float, *, label: str) -> None:
    observed = _finite_number(left, label=label)
    require(
        math.isclose(observed, right, rel_tol=0.0, abs_tol=1e-12),
        f"{label} drift",
    )


def _phase_contract(
    contract: Mapping[str, Any], *, expected_epochs: int,
) -> tuple[int, int]:
    require(expected_epochs in (1, 300), "Only e1 and e300 audits are supported")
    phase_name = "e1" if expected_epochs == 1 else "seed42_screening"
    phase = contract.get("phases", {}).get(phase_name)
    require(isinstance(phase, Mapping), f"Missing {phase_name} phase contract")
    require(phase.get("epochs") == expected_epochs, f"{phase_name} epoch drift")
    minimum_epochs = int(phase.get("minimum_epochs", -1))
    patience = int(phase.get("patience", -1))
    expected_minimum = 1 if expected_epochs == 1 else 40
    require(minimum_epochs == expected_minimum, f"{phase_name} minimum epoch drift")
    require(patience == 40, f"{phase_name} patience drift")
    return minimum_epochs, patience


def _validate_binding(
    *, output: Path, dataset: str, binding: Mapping[str, Any],
    base_audit: Mapping[str, Any], summary: Mapping[str, Any],
) -> None:
    launch = read_json(output / "launch_contract.json")
    require(launch.get("dataset") == dataset, "Launch dataset binding drift")
    population = launch.get("validation_target_population")
    require(
        population == binding.get("validation_target_population"),
        "Validation population differs from bounded-QK binding",
    )
    validate_validation_population(
        dataset=dataset,
        population=population,
        reference=dict(binding["validation_target_population"]),
    )
    for name in ("data_sha256", "split_manifest_sha256"):
        require(launch.get(name) == binding.get(name), f"Launch {name} binding drift")

    quantity_contract = launch.get("quantity_contract")
    require(isinstance(quantity_contract, Mapping), "Launch quantity contract is missing")
    require(
        quantity_contract.get("boundaries") == binding.get("quantity_boundaries")
        and quantity_contract.get("quantiles") == [0.5, 0.9, 0.95, 0.99],
        "Train quantity quantiles differ from bounded-QK binding",
    )
    resume = summary.get("resume_identity")
    require(isinstance(resume, Mapping), "Summary resume identity is missing")
    resume_quantity = resume.get("quantity_contract")
    require(
        isinstance(resume_quantity, Mapping)
        and resume_quantity == quantity_contract,
        "Summary train quantity contract drift",
    )
    metrics = base_audit.get("metrics")
    require(isinstance(metrics, Mapping), "Base audit metrics are missing")
    require(
        metrics.get("stratum_counts") == binding.get("stratum_counts")
        and metrics.get("body_target_count") == binding.get("body_target_count"),
        "Quantity stratum population differs from bounded-QK binding",
    )


def _validate_training_identity(
    summary: Mapping[str, Any], *, dataset: str, binding: Mapping[str, Any],
    source_revision: str, expected_epochs: int, minimum_epochs: int,
    patience: int,
) -> None:
    resume = summary.get("resume_identity")
    require(isinstance(resume, Mapping), "Training resume identity is missing")
    require(
        resume.get("backbone") == BOUNDED_QK_BACKBONE
        and resume.get("variant") == VARIANT
        and resume.get("seed") == 42
        and resume.get("checkpoint_monitor") == MONITOR,
        "Training resume route drift",
    )
    require(
        summary.get("source_revision") == source_revision
        and summary.get("source_revision_history") == [source_revision],
        "Training source revision lineage drift",
    )
    arguments = resume.get("arguments")
    require(isinstance(arguments, Mapping), "Training arguments are missing")
    population = binding["validation_target_population"]
    expected = {
        "dataset_contract": dataset,
        "data_sha256": binding["data_sha256"],
        "split_manifest_sha256": binding["split_manifest_sha256"],
        "lookback_weeks": int(population["lookback_weeks"]),
        "max_seq_len": int(population["max_seq_len"]),
        "max_train_batches": None,
        "max_val_batches": None,
        "max_series": None,
        "hidden_dim": 64,
        "batch_size": 128,
        "lr": 0.001,
        "grad_clip": 1.0,
        "epochs": expected_epochs,
        "min_epochs": minimum_epochs,
        "early_stopping_patience": patience,
        "model_role": BOUNDED_QK_ROLE,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "quantile_adaptive_strength": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 300.0,
        "time_wd_safety_limit": 40.0,
        "time_head_lr_multiplier": 1.0,
        "source_revision": source_revision,
    }
    for name, value in expected.items():
        require(
            name in arguments and arguments[name] == value,
            f"Training argument {name} drift",
        )
    require(
        summary.get("checkpoint_monitor") == MONITOR
        and summary.get("checkpoint_monitor_history_key") == "val_qty_rmse"
        and summary.get("checkpoint_selection") == SELECTION,
        "Training selector metadata drift",
    )


def _first_early_stop(
    history: list[Mapping[str, Any]], *, minimum_epochs: int, patience: int,
) -> int | None:
    best_epoch = int(history[0]["epoch"])
    best_value = float(history[0]["val_qty_rmse"])
    for row in history:
        epoch = int(row["epoch"])
        value = float(row["val_qty_rmse"])
        if value < best_value:
            best_epoch = epoch
            best_value = value
        if epoch >= minimum_epochs and epoch - best_epoch >= patience:
            return epoch
    return None


def _validate_history_and_selection(
    *, history_payload: Mapping[str, Any], summary: Mapping[str, Any],
    best: Mapping[str, Any], last: Mapping[str, Any], dataset: str,
    expected_epochs: int, minimum_epochs: int, patience: int,
) -> dict[str, Any]:
    history = history_payload.get("history")
    require(isinstance(history, list) and bool(history), "Training history is missing")
    require(
        all(isinstance(row, Mapping) for row in history),
        "Training history row is invalid",
    )
    require(
        last.get("history") == history,
        "Last-state checkpoint history differs from history artifact",
    )
    completed = summary.get("completed_epochs")
    require(
        isinstance(completed, int)
        and not isinstance(completed, bool)
        and completed == len(history),
        "Training history length drift",
    )
    epochs = [row.get("epoch") for row in history]
    require(epochs == list(range(1, completed + 1)), "Training history epoch drift")
    values = [
        _finite_number(row.get("val_qty_rmse"), label="history val_qty_rmse")
        for row in history
    ]
    require(
        all(
            row.get("train_event_count") == DATASETS[dataset]["train_targets"]
            for row in history
        ),
        "An epoch did not process the full train population",
    )
    selected_index = min(range(len(history)), key=values.__getitem__)
    selected_epoch = int(history[selected_index]["epoch"])
    selected_value = values[selected_index]
    require(
        summary.get("best_epoch")
        == best.get("best_epoch")
        == last.get("best_epoch")
        == selected_epoch,
        "Earliest raw-RMSE selected epoch drift",
    )
    _require_metric_equal(
        summary.get("best_val_qty_rmse"), selected_value,
        label="Summary selected raw RMSE",
    )
    _require_metric_equal(
        summary.get("selected_metric_value"), selected_value,
        label="Summary selected metric",
    )
    _require_metric_equal(
        best.get("selected_metric_value"), selected_value,
        label="Checkpoint selected metric",
    )

    first_stop = _first_early_stop(
        history, minimum_epochs=minimum_epochs, patience=patience
    )
    if expected_epochs == 1:
        require(
            completed == 1
            and first_stop is None
            and summary.get("stopped_early") is False,
            "e1 completion contract drift",
        )
    elif first_stop is None:
        require(
            completed == expected_epochs
            and summary.get("stopped_early") is False,
            "e300 fit ended before its epoch budget",
        )
    else:
        require(
            completed == first_stop
            and summary.get("stopped_early") is True,
            "e300 fit continued after or stopped before early stopping",
        )
    return {
        "history_length": len(history),
        "completed_epochs": completed,
        "selected_epoch": selected_epoch,
        "selected_val_qty_rmse": selected_value,
        "stopped_early": first_stop is not None,
    }


def _kernel_lag_evidence(
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    evidence: dict[str, dict[str, dict[str, float]]] = {}
    for state_name, state in states.items():
        state_evidence: dict[str, dict[str, float]] = {}
        for key in BOUNDED_QK_KERNEL_KEYS:
            kernel = state.get(key)
            require(
                isinstance(kernel, torch.Tensor)
                and tuple(kernel.shape) == (3, 64)
                and bool(torch.isfinite(kernel).all()),
                f"{state_name} bounded-QK kernel is missing or malformed: {key}",
            )
            lag_values: dict[str, float] = {}
            for lag in range(3):
                lag_l1 = float(kernel[lag].abs().sum().item())
                require(
                    math.isfinite(lag_l1) and lag_l1 > 0.0,
                    f"{state_name} {key} lag{lag} was not trained",
                )
                lag_values[f"lag{lag}_l1"] = lag_l1
            state_evidence[key] = lag_values
        evidence[state_name] = state_evidence
    return evidence


def audit_bounded_qk_job(
    output: Path, *, candidate: dict[str, Any], dataset: str,
    source_revision: str, expected_epochs: int, binding: dict[str, Any],
) -> dict[str, Any]:
    """Audit one full-data bounded-QK e1 or e300 training job."""
    require(sha256(CONTRACT_PATH) == CONTRACT_SHA256, "Bounded-QK contract digest drift")
    contract = read_json(CONTRACT_PATH)
    require(
        contract.get("contract_id") == "hard_lmm_bounded_qk_screening_v1"
        and contract.get("status") == "frozen_before_bounded_qk_gpu_outputs"
        and contract.get("held_out_test_evaluated") is False,
        "Bounded-QK screening contract drift",
    )
    frozen_candidate = contract.get("candidate")
    require(isinstance(frozen_candidate, Mapping), "Frozen bounded-QK candidate is missing")
    require(
        candidate.get("backbone") == frozen_candidate.get("backbone")
        == BOUNDED_QK_BACKBONE
        and candidate.get("model_role") == frozen_candidate.get("model_role")
        == BOUNDED_QK_ROLE,
        "Bounded-QK candidate route drift",
    )
    require(
        contract.get("body_strata") == list(BODY_STRATA),
        "Bounded-QK body strata drift",
    )
    canonical_binding = contract.get("data_bindings", {}).get(dataset)
    require(binding == canonical_binding, "Dataset binding differs from bounded-QK contract")
    minimum_epochs, patience = _phase_contract(
        contract, expected_epochs=expected_epochs
    )

    base_audit = audit_job(
        output,
        candidate=candidate,
        dataset=dataset,
        source_revision=source_revision,
        expected_epochs=expected_epochs,
        population_reference=binding.get("validation_target_population"),
    )
    summary_path = Path(str(base_audit["summary"]))
    run_dir = summary_path.parent
    require(run_dir.name == "seed_42", "Summary is outside the canonical seed route")
    require(run_dir.parent.name == VARIANT, "Summary is outside the canonical variant route")
    require(
        run_dir.parent.parent.name == BOUNDED_QK_BACKBONE,
        "Summary is outside the bounded-QK backbone route",
    )
    best_path = run_dir / "best_val_qty_rmse_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    history_path = run_dir / "history.json"
    require(best_path.is_file(), "Selected checkpoint is missing")
    require(last_path.is_file(), "Last-state checkpoint is missing")
    require(history_path.is_file(), "Training history is missing")

    summary = read_json(summary_path)
    _validate_binding(
        output=output,
        dataset=dataset,
        binding=binding,
        base_audit=base_audit,
        summary=summary,
    )
    _validate_training_identity(
        summary,
        dataset=dataset,
        binding=binding,
        source_revision=source_revision,
        expected_epochs=expected_epochs,
        minimum_epochs=minimum_epochs,
        patience=patience,
    )
    best = torch_load_checkpoint(best_path, map_location="cpu")
    last = torch_load_checkpoint(last_path, map_location="cpu")
    require(isinstance(best, dict), "Selected checkpoint payload is invalid")
    require(isinstance(last, dict), "Last-state checkpoint payload is invalid")
    _require_route_identity(
        best,
        artifact="selected checkpoint",
        candidate=candidate,
        source_revision=source_revision,
        summary=summary,
    )
    _require_route_identity(
        last,
        artifact="last-state checkpoint",
        candidate=candidate,
        source_revision=source_revision,
        summary=summary,
    )
    for artifact, payload in (("selected checkpoint", best), ("last-state checkpoint", last)):
        require(
            payload.get("source_revision_history") == [source_revision],
            f"{artifact} source revision lineage drift",
        )
        require(
            payload.get("selection") == SELECTION
            and payload.get("checkpoint_selection") == SELECTION
            and payload.get("checkpoint_monitor_history_key") == "val_qty_rmse",
            f"{artifact} selector metadata drift",
        )
    require(
        last.get("checkpoint_type") == "epoch_resume"
        and last.get("checkpoint_schema_version") == 2,
        "Last-state checkpoint type drift",
    )
    require(last.get("epoch") == summary.get("completed_epochs"), "Last-state epoch drift")
    selector = _validate_history_and_selection(
        history_payload=read_json(history_path),
        summary=summary,
        best=best,
        last=last,
        dataset=dataset,
        expected_epochs=expected_epochs,
        minimum_epochs=minimum_epochs,
        patience=patience,
    )

    best_state = _state_dict(best, "model_state_dict", artifact="selected checkpoint")
    last_state = _state_dict(last, "model_state_dict", artifact="last-state checkpoint")
    last_best_state = _state_dict(last, "best_state_dict", artifact="last-state checkpoint")
    best_digest = canonical_state_dict_sha256(best_state)
    last_digest = canonical_state_dict_sha256(last_state)
    last_best_digest = canonical_state_dict_sha256(last_best_state)
    require(best.get("model_state_sha256") == best_digest, "Selected state digest drift")
    require(summary.get("checkpoint_state_sha256") == best_digest, "Summary selected state digest drift")
    require(last.get("model_state_sha256") == last_digest, "Last current-state digest drift")
    require(last.get("best_state_sha256") == last_best_digest, "Last best-state digest drift")
    require(last_best_digest == best_digest, "Last best state does not route to selected checkpoint")
    _require_same_state(best_state, last_best_state, label="Selected/last-best")

    best_model = _restore_model(
        best, best_state, candidate=candidate, artifact="selected checkpoint"
    )
    last_best_model = _restore_model(
        last, last_best_state, candidate=candidate, artifact="last best-state"
    )
    last_model = _restore_model(
        last, last_state, candidate=candidate, artifact="last current-state"
    )
    _require_equal_outputs(
        _synthetic_outputs(best_model), _synthetic_outputs(last_best_model)
    )

    optimizer_state = last.get("optimizer_state_dict")
    require(isinstance(optimizer_state, dict), "Last optimizer state is missing")
    require(bool(optimizer_state.get("state")), "Last optimizer state is empty")
    _finite_tree(optimizer_state, path="last.optimizer_state_dict")
    arguments = last["resume_identity"]["arguments"]
    optimizer = build_optimizer(
        last_model,
        lr=float(arguments["lr"]),
        time_head_lr_multiplier=float(arguments["time_head_lr_multiplier"]),
    )
    try:
        optimizer.load_state_dict(optimizer_state)
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise ValueError("Last optimizer strict restore failed") from error
    _finite_tree(optimizer.state_dict(), path="restored_optimizer")

    kernel_evidence = _kernel_lag_evidence(
        {"selected": best_state, "last_current": last_state}
    )
    result = dict(base_audit)
    result["bounded_qk_audit"] = {
        "status": "passed",
        "contract_id": BOUNDED_QK_CONTRACT_ID,
        "screening_contract_sha256": CONTRACT_SHA256,
        "binding_verified": True,
        "full_train_population_per_epoch": DATASETS[dataset]["train_targets"],
        "validation_target_population": binding["validation_target_population"],
        "data_sha256": binding["data_sha256"],
        "split_manifest_sha256": binding["split_manifest_sha256"],
        "train_quantity_boundaries": binding["quantity_boundaries"],
        "selector": selector,
        "best_state_sha256": best_digest,
        "last_state_sha256": last_digest,
        "last_best_state_sha256": last_best_digest,
        "best_checkpoint_file_sha256": sha256(best_path),
        "last_checkpoint_file_sha256": sha256(last_path),
        "history_file_sha256": sha256(history_path),
        "strict_best_restore": True,
        "strict_last_restore": True,
        "optimizer_restore": True,
        "synthetic_target_outputs_roundtrip_equal": True,
        "kernel_lag_evidence": kernel_evidence,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    return result


def evaluate_quantity_gate(
    actual: Mapping[str, Any], B: Mapping[str, Any], FULL: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen B improvement and FULL retention quantity gate."""
    names = ("raw_rmse", "overall_mae", "body_mae", "gt_p99_mae")
    values: dict[str, dict[str, float]] = {}
    for label, metrics in (("actual", actual), ("B", B), ("FULL", FULL)):
        row = {name: _finite_number(metrics.get(name), label=f"{label}.{name}") for name in names}
        require(all(value >= 0.0 for value in row.values()), f"{label} quantity metric is negative")
        if label != "actual":
            require(all(value > 0.0 for value in row.values()), f"{label} quantity denominator is zero")
        values[label] = row
    checks = {
        "B_raw_rmse_strict_improvement": values["actual"]["raw_rmse"] < values["B"]["raw_rmse"],
        "B_overall_mae_within_1_percent": values["actual"]["overall_mae"] <= values["B"]["overall_mae"] * 1.01,
        "B_body_mae_within_2_percent": values["actual"]["body_mae"] <= values["B"]["body_mae"] * 1.02,
        "B_gt_p99_mae_within_2_percent": values["actual"]["gt_p99_mae"] <= values["B"]["gt_p99_mae"] * 1.02,
        "FULL_raw_rmse_within_1_percent": values["actual"]["raw_rmse"] <= values["FULL"]["raw_rmse"] * 1.01,
        "FULL_overall_mae_within_1_percent": values["actual"]["overall_mae"] <= values["FULL"]["overall_mae"] * 1.01,
        "FULL_body_mae_within_2_percent": values["actual"]["body_mae"] <= values["FULL"]["body_mae"] * 1.02,
        "FULL_gt_p99_mae_within_2_percent": values["actual"]["gt_p99_mae"] <= values["FULL"]["gt_p99_mae"] * 1.02,
    }
    ratios = {
        reference: {
            name: values["actual"][name] / values[reference][name]
            for name in names
        }
        for reference in ("B", "FULL")
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "ratios": ratios,
        "values": values,
    }


def evaluate_time_gate(
    candidate_proper: Any, B_proper: Any, FULL_proper: Any, dataset: str,
) -> dict[str, Any]:
    """Apply the common B guardrail and report the separate Taxi/FULL claim."""
    require(dataset in DATASETS, f"Unknown dataset: {dataset}")
    candidate = _finite_number(candidate_proper, label="candidate proper NLL")
    baseline_B = _finite_number(B_proper, label="B proper NLL")
    baseline_FULL = _finite_number(FULL_proper, label="FULL proper NLL")
    common_passed = candidate <= baseline_B + 0.01
    taxi_applicable = dataset == "yellow_trip_hourly"
    taxi_claim_passed = (
        candidate <= baseline_FULL - 0.005 if taxi_applicable else None
    )
    return {
        "status": "passed" if common_passed else "failed",
        "candidate_within_aligned_B_plus_0_01": common_passed,
        "candidate_minus_B": candidate - baseline_B,
        "taxi_FULL_improvement_claim": {
            "applicable": taxi_applicable,
            "passed": taxi_claim_passed,
            "FULL_minus_candidate": baseline_FULL - candidate,
            "minimum_improvement": 0.005,
            "affects_overall_status": False,
        },
        "values": {
            "candidate_proper_nll": candidate,
            "B_proper_nll": baseline_B,
            "FULL_proper_nll": baseline_FULL,
        },
    }


__all__ = [
    "CONTRACT_PATH",
    "CONTRACT_SHA256",
    "audit_bounded_qk_job",
    "evaluate_quantity_gate",
    "evaluate_time_gate",
]
