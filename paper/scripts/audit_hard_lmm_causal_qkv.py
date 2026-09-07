"""Strict artifact audit for the Hard-LMM causal event-Q/K/V candidate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from models.TPPs.CountAwareFactory import (
    build_count_aware_model,
    validate_checkpoint_route,
)
from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_BACKBONE,
    CAUSAL_QKV_KERNEL_KEYS,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    BODY_STRATA,
    audit_job,
    read_json,
    require,
    validate_validation_population,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_causal_qkv_screening_v1.json"
VARIANT = "count_only_log_regression"
MONITOR = "validation_raw_quantity_rmse"


def _finite_tree(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"Non-finite tensor at {path}")
    elif isinstance(value, Mapping):
        for name, child in value.items():
            _finite_tree(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Non-finite value at {path}")


def _state_dict(payload: Mapping[str, Any], key: str, *, artifact: str) -> dict[str, torch.Tensor]:
    state = payload.get(key)
    require(isinstance(state, dict) and bool(state), f"{artifact} {key} is missing")
    require(
        all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()),
        f"{artifact} {key} is not a tensor state dict",
    )
    _finite_tree(state, path=f"{artifact}.{key}")
    return state


def _require_same_state(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    require(set(left) == set(right), f"{label} state keys differ")
    for name in left:
        require(torch.equal(left[name], right[name]), f"{label} differs at {name}")


def _require_route_identity(
    payload: dict[str, Any],
    *,
    artifact: str,
    candidate: Mapping[str, Any],
    source_revision: str,
    summary: Mapping[str, Any],
) -> None:
    validate_checkpoint_route(payload, str(candidate["backbone"]))
    expected = {
        "backbone": candidate["backbone"],
        "variant": VARIANT,
        "seed": 42,
        "source_revision": source_revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_monitor": MONITOR,
    }
    for key, value in expected.items():
        require(payload.get(key) == value, f"{artifact} {key} route drift")
    for key in ("encoder_config", "interface_meta", "resume_identity"):
        require(payload.get(key) == summary.get(key), f"{artifact} {key} drift")


def _model_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    interface = payload.get("interface_meta")
    resume_identity = payload.get("resume_identity")
    require(isinstance(interface, dict), "Training interface metadata is missing")
    require(isinstance(resume_identity, dict), "Training resume identity is missing")
    arguments = resume_identity.get("arguments")
    require(isinstance(arguments, dict), "Training argument identity is missing")
    time_meta = interface.get("time_head")
    require(isinstance(time_meta, dict), "Training time-head metadata is missing")

    def argument(name: str, default: Any = None) -> Any:
        value = arguments.get(name, default)
        require(value is not None or default is None, f"Training argument {name} is missing")
        return value

    return {
        "hidden_dim": int(argument("hidden_dim")),
        "train_log_mean": float(interface["train_target_mean"]),
        "train_log_std": float(interface["train_target_std"]),
        "max_seq_len": int(argument("max_seq_len")),
        "quantity_variant": str(payload.get("variant")),
        "quantity_sigma_floor": float(argument("quantity_sigma_floor", 1e-3)),
        "lambda_location_huber": float(argument("lambda_location_huber", 1.0)),
        "location_huber_delta": float(argument("location_huber_delta", 0.25)),
        "lambda_tail": float(argument("lambda_tail", 0.0)),
        "tail_threshold": float(argument("tail_threshold", 46.0)),
        "tail_normalization_scale": float(argument("tail_normalization_scale", 46.0)),
        "tail_clip_cap": float(argument("tail_clip_cap", 187.0)),
        "tail_huber_delta": float(argument("tail_huber_delta", 1.0)),
        "time_head_mode": str(argument("time_head_mode")),
        "time_scale": float(argument("time_scale")),
        "time_w_max": float(argument("time_w_max")),
        "time_intercept_limit": float(argument("time_intercept_limit")),
        "time_initial_intercept": time_meta.get("time_initial_intercept"),
        "time_wd_safety_limit": float(argument("time_wd_safety_limit", 40.0)),
        "time_initial_location": time_meta.get("time_initial_location"),
        "time_initial_scale": time_meta.get("time_initial_scale"),
        "time_sigma_floor": float(argument("time_sigma_floor", 1e-3)),
        "titans_memory_gradient_clip": argument("titans_memory_gradient_clip", None),
    }


def _restore_model(
    payload: Mapping[str, Any],
    state: dict[str, torch.Tensor],
    *,
    candidate: Mapping[str, Any],
    artifact: str,
) -> torch.nn.Module:
    model, metadata = build_count_aware_model(
        str(candidate["backbone"]),
        **_model_kwargs(payload),
    )
    require(metadata == payload.get("encoder_config"), f"{artifact} rebuilt metadata drift")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{artifact} strict model restore failed") from error
    _finite_tree(model.state_dict(), path=f"{artifact}.restored_model")
    return model


def _synthetic_outputs(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model.eval()
    dts = torch.tensor(
        [[0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 3.0, 6.0, 0.0]],
        dtype=torch.float32,
    )
    quantities = torch.tensor(
        [[2.0, 5.0, 9.0, 12.0, 20.0], [1.0, 7.0, 13.0, 21.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]],
        dtype=torch.bool,
    )
    with torch.no_grad():
        outputs = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
    _finite_tree(outputs, path="synthetic_outputs")
    return outputs


def _require_equal_outputs(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> None:
    require(left.keys() == right.keys(), "Synthetic output keys drift")
    for name in left:
        require(torch.equal(left[name], right[name]), f"Synthetic roundtrip drift at {name}")


def _kernel_lag_evidence(
    states: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    evidence: dict[str, dict[str, dict[str, float]]] = {}
    for state_name, state in states.items():
        state_evidence: dict[str, dict[str, float]] = {}
        for key in CAUSAL_QKV_KERNEL_KEYS:
            kernel = state.get(key)
            require(
                isinstance(kernel, torch.Tensor) and kernel.ndim == 2 and kernel.shape[0] == 3,
                f"{state_name} causal Q/K/V kernel is missing or malformed: {key}",
            )
            lag_values: dict[str, float] = {}
            for lag in (1, 2):
                lag_l1 = float(kernel[lag].abs().sum().item())
                require(
                    math.isfinite(lag_l1) and lag_l1 > 0.0,
                    f"{state_name} {key} lag{lag} was not trained",
                )
                lag_values[f"lag{lag}_l1"] = lag_l1
            state_evidence[key] = lag_values
        evidence[state_name] = state_evidence
    return evidence


def _validate_binding(
    *,
    output: Path,
    dataset: str,
    binding: dict[str, Any],
    base_audit: Mapping[str, Any],
) -> None:
    contract = read_json(CONTRACT_PATH)
    require(contract.get("contract_id") == "hard_lmm_causal_qkv_screening_v1", "Contract ID drift")
    require(contract.get("body_strata") == list(BODY_STRATA), "Contract body strata drift")
    canonical = contract.get("data_bindings", {}).get(dataset)
    require(binding == canonical, "Dataset binding differs from the frozen contract")
    launch = read_json(output / "launch_contract.json")
    population = launch.get("validation_target_population")
    require(
        population == binding.get("validation_target_population"),
        "Validation population differs from causal-QKV binding",
    )
    validate_validation_population(
        dataset=dataset,
        population=population,
        reference=binding["validation_target_population"],
    )
    for key in ("data_sha256", "split_manifest_sha256"):
        require(launch.get(key) == binding.get(key), f"Launch {key} binding drift")
    quantity_contract = launch.get("quantity_contract")
    require(isinstance(quantity_contract, dict), "Launch quantity contract is missing")
    require(
        quantity_contract.get("boundaries") == binding.get("quantity_boundaries"),
        "Quantity boundaries differ from causal-QKV binding",
    )
    summary = read_json(Path(str(base_audit["summary"])))
    resume_identity = summary.get("resume_identity")
    require(isinstance(resume_identity, dict), "Summary resume identity is missing")
    resume_quantity_contract = resume_identity.get("quantity_contract")
    require(
        isinstance(resume_quantity_contract, dict)
        and resume_quantity_contract.get("boundaries")
        == binding.get("quantity_boundaries"),
        "Summary quantity boundaries differ from causal-QKV binding",
    )
    metrics = base_audit.get("metrics")
    require(isinstance(metrics, dict), "Base audit metrics are missing")
    require(
        metrics.get("stratum_counts") == binding.get("stratum_counts"),
        "Quantity stratum counts differ from causal-QKV binding",
    )
    require(
        metrics.get("body_target_count") == binding.get("body_target_count"),
        "Body target count differs from causal-QKV binding",
    )


def audit_causal_qkv_job(
    output: Path,
    *,
    candidate: dict[str, Any],
    dataset: str,
    source_revision: str,
    expected_epochs: int,
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Audit one causal-QKV job, including strict model and optimizer restore."""
    contract = read_json(CONTRACT_PATH)
    frozen_candidate = contract.get("candidate")
    require(isinstance(frozen_candidate, dict), "Frozen causal-QKV candidate is missing")
    for key in ("backbone", "model_role"):
        require(candidate.get(key) == frozen_candidate.get(key), f"Candidate {key} contract drift")
    require(candidate.get("backbone") == CAUSAL_QKV_BACKBONE, "Wrong causal-QKV backbone")

    base_audit = audit_job(
        output,
        candidate=candidate,
        dataset=dataset,
        source_revision=source_revision,
        expected_epochs=expected_epochs,
        population_reference=binding.get("validation_target_population"),
    )
    _validate_binding(
        output=output,
        dataset=dataset,
        binding=binding,
        base_audit=base_audit,
    )

    summary_path = Path(str(base_audit["summary"]))
    run_dir = summary_path.parent
    require(run_dir.name == "seed_42", "Summary is outside the canonical seed route")
    require(run_dir.parent.name == VARIANT, "Summary is outside the canonical variant route")
    require(run_dir.parent.parent.name == candidate["backbone"], "Summary is outside the canonical backbone route")
    best_path = run_dir / "best_val_qty_rmse_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    require(best_path.is_file(), "Selected checkpoint is missing")
    require(last_path.is_file(), "Last-state checkpoint is missing")

    summary = read_json(summary_path)
    require(summary.get("encoder_config") is not None, "Summary encoder metadata is missing")
    require(summary.get("interface_meta") is not None, "Summary interface metadata is missing")
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
    require(last.get("checkpoint_type") == "epoch_resume", "Last-state checkpoint type drift")
    require(last.get("checkpoint_schema_version") == 2, "Last-state checkpoint schema drift")
    require(int(last.get("epoch", -1)) == int(summary["completed_epochs"]), "Last-state epoch drift")
    require(int(best.get("best_epoch", -1)) == int(summary["best_epoch"]), "Selected epoch drift")
    require(int(last.get("best_epoch", -1)) == int(summary["best_epoch"]), "Last best epoch drift")

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
        best,
        best_state,
        candidate=candidate,
        artifact="selected checkpoint",
    )
    last_best_model = _restore_model(
        last,
        last_best_state,
        candidate=candidate,
        artifact="last best-state",
    )
    last_model = _restore_model(
        last,
        last_state,
        candidate=candidate,
        artifact="last current-state",
    )
    best_outputs = _synthetic_outputs(best_model)
    last_best_outputs = _synthetic_outputs(last_best_model)
    _require_equal_outputs(best_outputs, last_best_outputs)

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
        {
            "selected": best_state,
            "last_current": last_state,
        }
    )
    result = dict(base_audit)
    result["causal_qkv_audit"] = {
        "status": "passed",
        "contract_id": contract["contract_id"],
        "binding_verified": True,
        "best_artifact_route_verified": True,
        "last_artifact_route_verified": True,
        "best_state_sha256": best_digest,
        "last_state_sha256": last_digest,
        "last_best_state_sha256": last_best_digest,
        "strict_best_restore": True,
        "strict_last_restore": True,
        "optimizer_restore": True,
        "synthetic_target_outputs_roundtrip_equal": True,
        "kernel_lag_evidence": kernel_evidence,
    }
    return result


__all__ = ["CONTRACT_PATH", "audit_causal_qkv_job"]
