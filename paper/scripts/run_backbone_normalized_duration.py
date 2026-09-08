#!/usr/bin/env python3
"""Fit the aligned K=1 duration head on frozen B, FULL, or BOUNDED states.

The job manifest pins one source checkpoint and its training summary before
this runner sees duration outputs.  The source encoder, Hard-LMM, and quantity
path are cached once in eval mode and remain immutable.  Only the existing
130-parameter heteroscedastic log-normal duration head is fitted on train
states and selected by validation proper observation NLL.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.TPPs.CountAwareFactory import validate_checkpoint_route
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
)
from models.TPPs.CountAwareTitanBoundedQK import (
    BOUNDED_QK_BACKBONE,
    BOUNDED_QK_CONTRACT_ID,
    BOUNDED_QK_KERNEL_KEYS,
    BOUNDED_QK_ROLE,
    validate_bounded_qk_checkpoint,
)
from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_BACKBONE,
    CAUSAL_QKV_CONTRACT_ID,
    CAUSAL_QKV_KERNEL_KEYS,
    CAUSAL_QKV_ROLE,
    validate_causal_qkv_checkpoint,
)
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.run_aligned_frozen_lognormal_duration import (
    DATASETS,
    load_contract_without_duplicate_keys,
    validate_contract as validate_aligned_contract,
)
from paper.scripts.run_count_aware_tpp_backbone_control import (
    derive_train_time_contract,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    SELECTED_CHECKPOINT_NAME,
    build_frozen_lognormal_candidate,
    evaluate_cached_time_metrics,
    fit_time_head_from_cache,
    load_admitted_frame,
    resolve_censor_threshold,
    state_partition_sha256,
    train_time_statistics_from_contract,
    validate_cache_observation_contract,
    validate_observation_likelihood_contract,
)
from paper.scripts.run_hard_lmm_time_head_refit import (
    FrozenFeatureCache,
    exact_target_population_contract,
    finite_tensor_mapping,
    load_or_extract_feature_cache,
    require,
    save_json,
    sha256_file,
    stratified_quantity_metrics,
    validate_target_population,
)
from paper.scripts.run_matched_frozen_lognormal_duration import (
    build_source_model,
)
from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "backbone_normalized_duration_v1"
EXPECTED_EXECUTION_CONTRACT_SHA256 = (
    "ada204083f1389671b4df6285a130079ee76c45132a0f8cdfa65319a2ae447af"
)
JOB_MANIFEST_ID = "backbone_normalized_duration_job_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/backbone_normalized_duration_v1.json"
)
DEFAULT_ALIGNED_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/aligned_frozen_lognormal_duration_v1.json"
)
ALIGNED_CONTRACT_SHA256 = (
    "677cc3af91d84dfea8e3b4e71b38697fb467059f680678f0b8370a50f2366f16"
)
ALIGNED_CALIBRATION_SOURCE_REVISION = (
    "8213dbcdace2b62425236af15aa28f5b57f7b958"
)
DESIGN_CONTRACT_SHA256 = (
    "c2f4e931de69b0be0883aa5b8c022efedc44a243afd92c2861d35c23fe625319"
)
MODEL_ROLES = ("B", "FULL", "BOUNDED")
FIT_MODEL_ROLES = ("FULL", "BOUNDED")
REFERENCE_ONLY_MODEL_ROLES = ("B",)
SOURCE_PLANNED_EPOCHS = 300
SOURCE_MINIMUM_EPOCHS = 40
SOURCE_EARLY_STOPPING_PATIENCE = 40
SUMMARY_NAME = "summary.json"


def load_json_without_duplicate_keys(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(name for name, _ in pairs)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        require(not duplicates, f"Duplicate JSON keys: {duplicates}")
        return dict(pairs)

    require(path.is_file(), f"Missing JSON file: {path}")
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    require(isinstance(payload, dict), "JSON root must be an object")
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def runtime_identity(device: torch.device) -> dict[str, Any]:
    """Return the runtime fields that a frozen per-job manifest must pin."""
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_type": device.type,
        "device_name": device_name,
        "host": platform.node(),
    }


def current_source_revision() -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    require(_is_git_sha(revision), "Current checkout lacks a full Git revision")
    return revision


def _resolved_project_file(relative: Any, *, label: str) -> Path:
    require(
        isinstance(relative, str)
        and bool(relative)
        and relative not in {"TBD", "UNRESOLVED", "null"},
        f"Unresolved {label}",
    )
    logical = Path(relative)
    require(".." not in logical.parts, f"{label} contains parent traversal")
    path = (logical if logical.is_absolute() else PROJECT_ROOT / logical).resolve()
    require(path.is_file(), f"Missing {label}: {path}")
    return path


def validate_execution_contract(contract: Mapping[str, Any]) -> None:
    """Reject any drift from the approved three-backbone evaluation route."""
    require(contract.get("schema_version") == 1, "Execution contract schema drift")
    require(contract.get("contract_id") == CONTRACT_ID, "Execution contract ID drift")
    require(
        contract.get("status")
        == "frozen_before_backbone_normalized_duration_outputs",
        "Execution contract was not frozen before outputs",
    )
    aligned = contract.get("aligned_contract")
    require(isinstance(aligned, Mapping), "Aligned contract binding is missing")
    require(
        aligned.get("path")
        == "paper/contracts/aligned_frozen_lognormal_duration_v1.json"
        and aligned.get("sha256") == ALIGNED_CONTRACT_SHA256
        and aligned.get("calibration_source_revision")
        == ALIGNED_CALIBRATION_SOURCE_REVISION,
        "Aligned K=1 contract binding drift",
    )
    design = contract.get("backbone_design_contract")
    require(isinstance(design, Mapping), "Backbone design binding is missing")
    require(
        design.get("path")
        == "paper/contracts/hard_lmm_bounded_qk_causal_v_v1.md"
        and design.get("sha256") == DESIGN_CONTRACT_SHA256
        and design.get("contract_id") == BOUNDED_QK_CONTRACT_ID,
        "Bounded-QK design binding drift",
    )
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Execution scope is missing")
    expected_scope = {
        "roles": list(MODEL_ROLES),
        "datasets": list(DATASETS),
        "input_splits": ["train", "validation"],
        "optimization_split": "train",
        "selection_split": "validation",
        "evaluation_scope": "validation_only",
        "held_out_test": False,
        "seed": 42,
        "quantity_predictions_may_change": False,
        "fit_roles": list(FIT_MODEL_ROLES),
        "reference_only_roles": list(REFERENCE_ONLY_MODEL_ROLES),
    }
    require(dict(scope) == expected_scope, "Execution scope drift")
    roles = contract.get("roles")
    require(isinstance(roles, Mapping) and tuple(roles) == MODEL_ROLES, "Role scope drift")
    expected_roles = {
        "B": ("titantpp", None, "B0", "reuse_previously_verified_aligned_B"),
        "FULL": (
            CAUSAL_QKV_BACKBONE,
            CAUSAL_QKV_ROLE,
            CAUSAL_QKV_CONTRACT_ID,
            "fit_isolated_K1_head",
        ),
        "BOUNDED": (
            BOUNDED_QK_BACKBONE,
            BOUNDED_QK_ROLE,
            BOUNDED_QK_CONTRACT_ID,
            "fit_isolated_K1_head",
        ),
    }
    for role, (backbone, model_role, routing, action) in expected_roles.items():
        row = roles[role]
        require(
            row.get("backbone") == backbone
            and row.get("model_role") == model_role
            and row.get("routing_contract_id") == routing
            and row.get("source_checkpoint_selection")
            == "best_validation_raw_quantity_rmse"
            and row.get("normalized_duration_action") == action,
            f"Role route drift: {role}",
        )
    manifest = contract.get("source_job_manifest")
    require(
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "frozen_before_normalized_duration_outputs"
        and manifest.get("unresolved_source_fields_fail_closed") is True
        and manifest.get("checkpoint_and_summary_file_sha256_required") is True
        and manifest.get("source_history_file_sha256_required") is True
        and manifest.get("source_history_selector_replay_required") is True
        and manifest.get("checkpoint_canonical_state_sha256_required") is True
        and manifest.get("source_revision_history_required") is True
        and manifest.get("evaluation_source_revision_required") is True,
        "Source job-manifest policy drift",
    )
    cache = contract.get("cache_policy")
    require(
        isinstance(cache, Mapping)
        and all(
            cache.get(name) is expected
            for name, expected in {
                "one_cache_pair_per_dataset_role_source_checkpoint": True,
                "cross_role_cache_reuse": False,
                "identity_includes_job_manifest_sha256": True,
                "held_out_materialization": False,
            }.items()
        ),
        "Cache policy drift",
    )
    fit = contract.get("duration_fit")
    require(isinstance(fit, Mapping), "Duration fit contract is missing")
    expected_fit = {
        "reuse_aligned_k1_builder_likelihood_cache_and_fit_primitives": True,
        "trainable_parameter_names": [
            "v_t.weight", "b_t", "w_raw", "time_scale_weight.weight"
        ],
        "expected_trainable_parameter_count": 130,
        "calculation_dtype": "float64",
        "probability_clamp": None,
        "loss_cap": None,
        "seed": 42,
        "epochs": 100,
        "minimum_epochs": 20,
        "early_stopping_patience": 20,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "encoder_batch_size": 128,
        "cached_state_batch_size": 8192,
        "gradient_clip": 1.0,
        "checkpoint_selection": (
            "earliest_strict_finite_minimum_validation_proper_time_nll"
        ),
        "epoch_zero_is_candidate": True,
        "resume_identity_required": True,
    }
    require(dict(fit) == expected_fit, "Duration fit policy drift")


def job_binding_sha256(
    *, execution_contract_sha256: str, aligned_contract_sha256: str,
    job_manifest_sha256: str,
) -> str:
    require(
        all(
            _is_sha256(value)
            for value in (
                execution_contract_sha256,
                aligned_contract_sha256,
                job_manifest_sha256,
            )
        ),
        "Invalid job-binding digest input",
    )
    payload = json.dumps(
        {
            "aligned_contract_sha256": aligned_contract_sha256,
            "execution_contract_sha256": execution_contract_sha256,
            "job_manifest_sha256": job_manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"backbone_normalized_duration_v1\0" + payload).hexdigest()


def validate_job_manifest(
    manifest: Mapping[str, Any], *, contract: Mapping[str, Any],
    aligned_dataset: Mapping[str, Any], execution_contract_sha256: str,
    aligned_contract_sha256: str, evaluation_source_revision: str,
    expected_runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one immutable dataset/role/source binding."""
    require(manifest.get("schema_version") == 1, "Job manifest schema drift")
    require(manifest.get("manifest_id") == JOB_MANIFEST_ID, "Job manifest ID drift")
    require(
        manifest.get("status") == "frozen_before_normalized_duration_outputs",
        "Job manifest was not frozen before outputs",
    )
    require(
        manifest.get("execution_contract_sha256") == execution_contract_sha256
        and manifest.get("aligned_contract_sha256") == aligned_contract_sha256,
        "Job manifest contract binding drift",
    )
    dataset = manifest.get("dataset")
    role = manifest.get("model_role")
    require(dataset == aligned_dataset.get("dataset"), "Job dataset drift")
    require(role in MODEL_ROLES, "Job model role drift")
    role_contract = contract["roles"][role]
    require(
        manifest.get("calibration_source_revision")
        == ALIGNED_CALIBRATION_SOURCE_REVISION,
        "Calibration source revision drift",
    )
    require(
        _is_git_sha(evaluation_source_revision)
        and manifest.get("evaluation_source_revision")
        == evaluation_source_revision,
        "Evaluation source revision drift",
    )
    require(
        manifest.get("evaluation_runner_file_sha256")
        == sha256_file(Path(__file__)),
        "Evaluation runner file digest drift",
    )
    if expected_runtime_identity is not None:
        require(
            manifest.get("runtime") == dict(expected_runtime_identity),
            "Evaluation runtime identity drift",
        )
    data = manifest.get("data")
    require(isinstance(data, Mapping), "Job data binding is missing")
    require(
        data.get("sha256") == aligned_dataset["data_sha256"]
        and data.get("split_manifest_sha256")
        == aligned_dataset["split_manifest_sha256"],
        "Job data digest binding drift",
    )
    source = manifest.get("source")
    require(isinstance(source, Mapping), "Job source binding is missing")
    required_source_fields = (
        "checkpoint_path", "checkpoint_file_sha256", "checkpoint_state_sha256",
        "source_non_time_state_sha256", "summary_path", "summary_sha256",
        "history_path", "history_sha256",
        "training_source_revision", "training_source_revision_history",
    )
    require(
        all(source.get(name) not in (None, "", "TBD", "UNRESOLVED") for name in required_source_fields),
        "Job source contains unresolved fields",
    )
    for name in (
        "checkpoint_file_sha256", "checkpoint_state_sha256",
        "source_non_time_state_sha256", "summary_sha256",
        "history_sha256",
    ):
        require(_is_sha256(source.get(name)), f"Invalid job source digest: {name}")
    revision = source.get("training_source_revision")
    require(
        _is_git_sha(revision)
        and source.get("training_source_revision_history") == [revision],
        "Job source revision lineage drift",
    )
    require(
        source.get("backbone") == role_contract["backbone"]
        and source.get("model_role") == role_contract["model_role"]
        and source.get("routing_contract_id")
        == role_contract["routing_contract_id"]
        and source.get("quantity_variant") == LOG_MSE_VARIANT
        and source.get("seed") == 42
        and source.get("checkpoint_selection")
        == role_contract["source_checkpoint_selection"]
        and source.get("evaluation_scope") == "validation_only"
        and source.get("held_out_test_evaluated") is False,
        "Job source route/scope drift",
    )
    return {"dataset": dataset, "model_role": role, "data": dict(data), "source": dict(source)}


def validate_source_selection_history(
    source_history: Mapping[str, Any], *, payload: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the frozen raw-RMSE selector and its stopping boundary."""
    history = source_history.get("history")
    require(isinstance(history, list) and bool(history), "Source history is missing")
    require(
        all(isinstance(row, Mapping) for row in history),
        "Source history row is invalid",
    )
    epochs: list[int] = []
    values: list[float] = []
    for row in history:
        epoch = row.get("epoch")
        value = row.get("val_qty_rmse")
        require(
            isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            "Source history contains an invalid raw-RMSE row",
        )
        epochs.append(epoch)
        values.append(float(value))
    require(
        epochs == list(range(1, len(history) + 1)),
        "Source history epochs are not contiguous from one",
    )
    require(
        epochs[-1] <= SOURCE_PLANNED_EPOCHS,
        "Source history exceeds its epoch budget",
    )

    best_index = min(range(len(values)), key=values.__getitem__)
    best_epoch = epochs[best_index]
    best_value = values[best_index]
    first_stop: int | None = None
    running_best_epoch = epochs[0]
    running_best_value = values[0]
    for epoch, value in zip(epochs, values, strict=True):
        if value < running_best_value:
            running_best_epoch = epoch
            running_best_value = value
        if (
            epoch >= SOURCE_MINIMUM_EPOCHS
            and epoch - running_best_epoch >= SOURCE_EARLY_STOPPING_PATIENCE
        ):
            first_stop = epoch
            break
    if first_stop is None:
        require(
            epochs[-1] == SOURCE_PLANNED_EPOCHS
            and summary.get("stopped_early") is False,
            "Source fit ended before its epoch budget",
        )
    else:
        require(
            epochs[-1] == first_stop
            and summary.get("stopped_early") is True,
            "Source fit continued after or stopped before early stopping",
        )
    require(
        summary.get("completed_epochs") == epochs[-1],
        "Source completed epoch drift",
    )
    require(
        payload.get("best_epoch") == summary.get("best_epoch") == best_epoch,
        "Source raw-RMSE selected epoch drift",
    )
    selected_values = (
        payload.get("selected_metric_value"),
        summary.get("selected_metric_value"),
        summary.get("best_val_qty_rmse"),
    )
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and math.isclose(
                float(value), best_value, rel_tol=0.0, abs_tol=1e-12
            )
            for value in selected_values
        ),
        "Source raw-RMSE selected value drift",
    )
    return {
        "history_length": len(history),
        "completed_epochs": epochs[-1],
        "stopped_early": first_stop is not None,
        "selected_epoch": best_epoch,
        "selected_val_qty_rmse": best_value,
    }


def validate_source_artifacts(
    payload: Mapping[str, Any], summary: Mapping[str, Any],
    source_history: Mapping[str, Any], *, role: str,
    source_spec: Mapping[str, Any], checkpoint_file_sha256: str,
    aligned_dataset: Mapping[str, Any],
) -> str:
    """Bind source file, canonical state, training summary, and model route."""
    require(summary.get("status") == "success", "Source summary is not successful")
    required = {
        "backbone": source_spec["backbone"],
        "variant": LOG_MSE_VARIANT,
        "seed": 42,
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection": "best_validation_raw_quantity_rmse",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": source_spec["training_source_revision"],
        "source_revision_history": source_spec["training_source_revision_history"],
    }
    for name, expected in required.items():
        require(payload.get(name) == expected, f"Source checkpoint metadata drift: {name}")
    for name in (
        "backbone", "variant", "seed", "best_epoch", "evaluation_scope",
        "held_out_test_evaluated", "source_revision", "source_revision_history",
        "encoder_config", "interface_meta", "resume_identity",
        "checkpoint_monitor", "checkpoint_monitor_history_key",
        "checkpoint_selection", "selected_metric_value", "selection_formula",
    ):
        require(summary.get(name) == payload.get(name), f"Source summary/checkpoint drift: {name}")
    resume = payload.get("resume_identity")
    require(isinstance(resume, Mapping), "Source resume identity is missing")
    arguments = resume.get("arguments")
    require(isinstance(arguments, Mapping), "Source training arguments are missing")
    expected_arguments = {
        "dataset_contract": aligned_dataset["dataset"],
        "data_sha256": aligned_dataset["data_sha256"],
        "split_manifest_sha256": aligned_dataset["split_manifest_sha256"],
        "lookback_weeks": int(aligned_dataset["lookback"]),
        "max_seq_len": int(aligned_dataset["max_sequence_length"]),
        "max_train_batches": None,
        "max_val_batches": None,
        "max_series": None,
        "hidden_dim": 64,
        "batch_size": 128,
        "lr": 0.001,
        "grad_clip": 1.0,
        "epochs": 300,
        "min_epochs": 40,
        "early_stopping_patience": 40,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "time_head_mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 300.0,
        "time_wd_safety_limit": 40.0,
        "time_head_lr_multiplier": 1.0,
        "source_revision": source_spec["training_source_revision"],
    }
    for name, expected in expected_arguments.items():
        require(
            name in arguments and arguments[name] == expected,
            f"Source training/data contract drift: {name}",
        )
    validate_source_selection_history(
        source_history, payload=payload, summary=summary
    )
    expected_training_role = {
        "B": ("quantile_checkpoint_alignment", 1.0),
        "FULL": (CAUSAL_QKV_ROLE, 0.0),
        "BOUNDED": (BOUNDED_QK_ROLE, 0.0),
    }[role]
    require(
        arguments.get("model_role") == expected_training_role[0]
        and arguments.get("quantile_adaptive_strength")
        == expected_training_role[1],
        "Source quantity-training role drift",
    )
    require(
        summary.get("checkpoint_state_sha256")
        == payload.get("model_state_sha256")
        == source_spec["checkpoint_state_sha256"],
        "Source canonical state binding drift",
    )
    state = payload.get("model_state_dict")
    require(isinstance(state, Mapping), "Source model state is missing")
    finite_tensor_mapping(state, label="Source checkpoint")
    require(
        canonical_state_dict_sha256(state) == payload["model_state_sha256"],
        "Source checkpoint state digest drift",
    )
    encoder = payload.get("encoder_config")
    interface = payload.get("interface_meta")
    require(isinstance(encoder, Mapping), "Source encoder metadata is missing")
    require(isinstance(interface, Mapping), "Source interface metadata is missing")
    require(
        encoder.get("d_model") == 64
        and encoder.get("memory_mode") == "static_hard_lmm"
        and encoder.get("lmm_mem_size") == 64
        and encoder.get("lmm_topk") == 4,
        "Source Hard-LMM route drift",
    )
    time_head = encoder.get("time_head")
    require(
        isinstance(time_head, Mapping)
        and time_head.get("mode") == TIME_HEAD_MODE_LEGACY_CLAMPED,
        "Source legacy time-head route drift",
    )
    expected_legacy_time_head = {
        "mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 300.0,
        "jacobian_correction": False,
        "wd_clamp": 10.0,
    }
    require(
        all(time_head.get(name) == value for name, value in expected_legacy_time_head.items()),
        "Source encoder time-head contract drift",
    )
    require(
        interface.get("mode") == "mark_free_count_aware_log_regression"
        and interface.get("target_quantity_masked_from_history") is True,
        "Source quantity interface drift",
    )
    interface_time_head = interface.get("time_head")
    require(
        isinstance(interface_time_head, Mapping)
        and all(
            interface_time_head.get(name) == value
            for name, value in {
                "mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
                "time_scale": 3.0,
                "time_w_max": 10.0 / 3.0,
                "time_intercept_limit": 300.0,
                "time_wd_safety_limit": 40.0,
                "time_head_lr_multiplier": 1.0,
            }.items()
        ),
        "Source interface time-head contract drift",
    )
    validate_checkpoint_route(dict(payload), str(source_spec["backbone"]))
    if role == "B":
        require(
            encoder.get("backbone_contract_id") == "B0"
            and not any(name in state for name in CAUSAL_QKV_KERNEL_KEYS + BOUNDED_QK_KERNEL_KEYS),
            "B source contains a candidate route",
        )
    elif role == "FULL":
        require(validate_causal_qkv_checkpoint(dict(payload), CAUSAL_QKV_BACKBONE), "FULL route validation failed")
    else:
        require(validate_bounded_qk_checkpoint(dict(payload), BOUNDED_QK_BACKBONE), "BOUNDED route validation failed")
    non_time_sha = state_partition_sha256(state, time_head=False)
    require(
        non_time_sha == source_spec["source_non_time_state_sha256"],
        "Source non-time state digest drift",
    )
    require(_is_sha256(checkpoint_file_sha256), "Invalid checkpoint file digest")
    return non_time_sha


def build_cache_identity(
    *, dataset: str, role: str, source_spec: Mapping[str, Any],
    data_spec: Mapping[str, Any], execution_contract_sha256: str,
    aligned_contract_sha256: str, job_manifest_sha256: str,
    job_binding_sha: str, evaluation_source_revision: str,
) -> dict[str, Any]:
    """Return the common identity shared only by this role's train/val caches."""
    require(role in MODEL_ROLES, "Unknown cache model role")
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "fit_contract_binding_sha256": job_binding_sha,
        "execution_contract_sha256": execution_contract_sha256,
        "aligned_contract_sha256": aligned_contract_sha256,
        "job_manifest_sha256": job_manifest_sha256,
        "evaluation_source_revision": evaluation_source_revision,
        "dataset": dataset,
        "model_role": role,
        "source_backbone": source_spec["backbone"],
        "source_variant": source_spec["quantity_variant"],
        "data_sha256": data_spec["sha256"],
        "split_manifest_sha256": data_spec["split_manifest_sha256"],
        "source_checkpoint_sha256": source_spec["checkpoint_file_sha256"],
        "source_state_sha256": source_spec["checkpoint_state_sha256"],
        "source_non_time_state_sha256": source_spec["source_non_time_state_sha256"],
        "encoder_mode": "eval",
        "target_quantity_masked": True,
        "memory_target_write_masked": True,
        "evaluation_scope": "train_and_validation_only",
        "held_out_test_evaluated": False,
    }


def _selected_fit_audit(
    *, summary: Mapping[str, Any], selected: Mapping[str, Any], role: str,
    source_non_time_sha256: str, observation_contract: Mapping[str, Any],
    source_lineage: Mapping[str, Any],
) -> None:
    require(summary.get("status") == "success", "Duration fit did not complete")
    require(
        summary.get("selection")
        == "earliest_strict_finite_minimum_validation_proper_time_nll",
        "Duration checkpoint selector drift",
    )
    require(
        selected.get("checkpoint_type") == "selected_frozen_lognormal_duration"
        and selected.get("model_role") == role
        and selected.get("resume_identity") == summary.get("resume_identity")
        and selected.get("observation_likelihood_contract")
        == dict(observation_contract),
        "Selected duration checkpoint identity drift",
    )
    require(
        selected.get("source_checkpoint_lineage") == dict(source_lineage),
        "Selected duration checkpoint source lineage drift",
    )
    require(
        canonical_state_dict_sha256(selected["model_state_dict"])
        == selected.get("model_state_sha256"),
        "Selected duration checkpoint state digest drift",
    )
    require(
        state_partition_sha256(selected["model_state_dict"], time_head=False)
        == source_non_time_sha256,
        "Duration refit changed source non-time state",
    )
    require(
        summary.get("quantity_prediction_bitwise_identical") is True
        and summary.get("source_quantity_prediction_sha256")
        == summary.get("selected_quantity_prediction_sha256"),
        "Duration refit changed source quantity predictions",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", type=Path, required=True)
    parser.add_argument("--expected-job-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--aligned-contract", type=Path, default=DEFAULT_ALIGNED_CONTRACT
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        torch.cuda.reset_peak_memory_stats(device)
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)

    contract = load_json_without_duplicate_keys(args.contract)
    validate_execution_contract(contract)
    contract_sha = sha256_file(args.contract)
    require(
        contract_sha == EXPECTED_EXECUTION_CONTRACT_SHA256,
        "Frozen execution contract file digest drift",
    )
    aligned_sha = sha256_file(args.aligned_contract)
    require(aligned_sha == ALIGNED_CONTRACT_SHA256, "Aligned contract file digest drift")
    aligned_contract = load_contract_without_duplicate_keys(args.aligned_contract)
    aligned_datasets = validate_aligned_contract(aligned_contract)
    design_path = _resolved_project_file(
        contract["backbone_design_contract"]["path"], label="backbone design contract"
    )
    require(sha256_file(design_path) == DESIGN_CONTRACT_SHA256, "Backbone design file digest drift")

    manifest = load_json_without_duplicate_keys(args.job_manifest)
    require(_is_git_sha(args.source_revision), "Evaluation source revision must be a full Git SHA")
    require(
        args.source_revision == current_source_revision(),
        "Evaluation source revision differs from the current checkout",
    )
    manifest_sha = sha256_file(args.job_manifest)
    require(
        _is_sha256(args.expected_job_manifest_sha256)
        and manifest_sha == args.expected_job_manifest_sha256,
        "Pre-registered job manifest digest drift",
    )
    dataset_name = manifest.get("dataset")
    require(dataset_name in aligned_datasets, "Job dataset is outside aligned scope")
    job = validate_job_manifest(
        manifest,
        contract=contract,
        aligned_dataset=aligned_datasets[dataset_name],
        execution_contract_sha256=contract_sha,
        aligned_contract_sha256=aligned_sha,
        evaluation_source_revision=args.source_revision,
        expected_runtime_identity=runtime_identity(device),
    )
    dataset_spec = aligned_datasets[dataset_name]
    role = str(job["model_role"])
    data_spec = job["data"]
    source_spec = job["source"]
    partial_limits = (
        args.max_train_batches,
        args.max_validation_batches,
        args.max_epochs,
    )
    strictly_partial_request = (
        args.max_train_batches is not None
        or args.max_validation_batches is not None
        or (
            args.max_epochs is not None
            and 0 <= args.max_epochs < int(contract["duration_fit"]["epochs"])
        )
    )
    require(
        role not in REFERENCE_ONLY_MODEL_ROLES
        or strictly_partial_request,
        "B full fit is forbidden; reuse the previously verified aligned-B result",
    )
    fit_contract_binding_sha = job_binding_sha256(
        execution_contract_sha256=contract_sha,
        aligned_contract_sha256=aligned_sha,
        job_manifest_sha256=manifest_sha,
    )

    data_path = _resolved_project_file(data_spec["path"], label="dataset")
    split_manifest_path = _resolved_project_file(
        data_spec["split_manifest_path"], label="split manifest"
    )
    checkpoint_path = _resolved_project_file(
        source_spec["checkpoint_path"], label="source checkpoint"
    )
    source_summary_path = _resolved_project_file(
        source_spec["summary_path"], label="source summary"
    )
    source_history_path = _resolved_project_file(
        source_spec["history_path"], label="source history"
    )
    require(sha256_file(data_path) == data_spec["sha256"], "Dataset file digest drift")
    require(
        sha256_file(split_manifest_path) == data_spec["split_manifest_sha256"],
        "Split-manifest file digest drift",
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    require(
        checkpoint_sha == source_spec["checkpoint_file_sha256"],
        "Source checkpoint file digest drift",
    )
    require(
        sha256_file(source_summary_path) == source_spec["summary_sha256"],
        "Source summary file digest drift",
    )
    require(
        sha256_file(source_history_path) == source_spec["history_sha256"],
        "Source history file digest drift",
    )
    source_payload = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    source_summary = load_json_without_duplicate_keys(source_summary_path)
    source_history = load_json_without_duplicate_keys(source_history_path)
    source_non_time_sha = validate_source_artifacts(
        source_payload,
        source_summary,
        source_history,
        role=role,
        source_spec=source_spec,
        checkpoint_file_sha256=checkpoint_sha,
        aligned_dataset=dataset_spec,
    )
    source_model = build_source_model(
        source_payload, max_seq_len=int(dataset_spec["max_sequence_length"])
    ).to(device)
    source_model_state_before = canonical_state_dict_sha256(
        source_model.state_dict()
    )
    source_model_non_time_before = state_partition_sha256(
        source_model.state_dict(), time_head=False
    )
    require(
        source_model_state_before == source_spec["checkpoint_state_sha256"]
        and source_model_non_time_before == source_non_time_sha,
        "Restored source model state differs from its pinned checkpoint",
    )

    frame = prepare_count_frame(load_admitted_frame(data_path))
    require(
        set(frame["chronological_split"].unique().to_list())
        <= {"train", "validation"},
        "Held-out rows entered the admitted frame",
    )
    lookback = int(dataset_spec["lookback"])
    max_seq_len = int(dataset_spec["max_sequence_length"])
    train_population = exact_target_population_contract(
        frame, target_split="train", lookback=lookback, max_seq_len=max_seq_len
    )
    validation_population = exact_target_population_contract(
        frame,
        target_split="validation",
        lookback=lookback,
        max_seq_len=max_seq_len,
    )
    validate_target_population(train_population, dataset_spec=dataset_spec)
    validate_target_population(validation_population, dataset_spec=dataset_spec)
    train_statistics = train_time_statistics_from_contract(
        dataset_spec,
        derive_train_time_contract(
            frame, lookback_weeks=lookback, max_seq_len=max_seq_len
        ),
    )
    censor_threshold = resolve_censor_threshold(dataset_name, dataset_spec)
    observation_contract = validate_observation_likelihood_contract(
        dataset_spec["observation_contract"]
    )

    fit_settings = contract["duration_fit"]
    if any(value is not None for value in partial_limits):
        require(
            args.allow_partial_contract,
            "Batch or epoch limits require --allow-partial-contract",
        )
    planned_epochs = int(fit_settings["epochs"])
    if args.max_epochs is not None:
        require(args.max_epochs >= 0, "max epochs must be nonnegative")
        planned_epochs = min(planned_epochs, args.max_epochs)
    encoder_batch_size = int(fit_settings["encoder_batch_size"])
    train_loader = make_loader(
        frame,
        target_split="train",
        batch_size=encoder_batch_size,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        shuffle=False,
        generator=None,
    )
    validation_loader = make_loader(
        frame,
        target_split="validation",
        batch_size=encoder_batch_size,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        shuffle=False,
        generator=None,
    )
    cache_base = build_cache_identity(
        dataset=dataset_name,
        role=role,
        source_spec=source_spec,
        data_spec=data_spec,
        execution_contract_sha256=contract_sha,
        aligned_contract_sha256=aligned_sha,
        job_manifest_sha256=manifest_sha,
        job_binding_sha=fit_contract_binding_sha,
        evaluation_source_revision=args.source_revision,
    )
    cache_dir = args.feature_cache_dir or (args.output_dir / "cache")
    if args.feature_cache_dir is not None:
        require(
            (cache_dir / "train_features.pt").is_file()
            and (cache_dir / "validation_features.pt").is_file(),
            "Requested role-specific feature cache is incomplete",
        )
    train_cache = load_or_extract_feature_cache(
        path=cache_dir / "train_features.pt",
        identity={
            **cache_base,
            "split": "train",
            "target_count": train_population["target_count"],
            "target_identity_sha256": train_population["target_identity_sha256"],
            "target_quantity_sha256": train_population["target_quantity_sha256"],
            "max_batches": args.max_train_batches,
        },
        model=source_model,
        loader=train_loader,
        device=device,
        include_quantity=False,
        max_batches=args.max_train_batches,
    )
    validation_cache = load_or_extract_feature_cache(
        path=cache_dir / "validation_features.pt",
        identity={
            **cache_base,
            "split": "validation",
            "target_count": validation_population["target_count"],
            "target_identity_sha256": validation_population[
                "target_identity_sha256"
            ],
            "target_quantity_sha256": validation_population[
                "target_quantity_sha256"
            ],
            "max_batches": args.max_validation_batches,
        },
        model=source_model,
        loader=validation_loader,
        device=device,
        include_quantity=True,
        max_batches=args.max_validation_batches,
    )
    require(
        canonical_state_dict_sha256(source_model.state_dict())
        == source_model_state_before
        and state_partition_sha256(source_model.state_dict(), time_head=False)
        == source_model_non_time_before,
        "Feature extraction changed the frozen source model state",
    )
    full_data = (
        args.max_train_batches is None and args.max_validation_batches is None
    )
    qualified_full_fit = full_data and args.max_epochs is None
    train_observation = validation_observation = None
    if full_data:
        require(
            train_cache.count == int(train_population["target_count"])
            and validation_cache.count
            == int(validation_population["target_count"]),
            "Complete cache target count drift",
        )
        train_observation = validate_cache_observation_contract(
            train_cache,
            dataset_spec=dataset_spec,
            split="train",
            censor_threshold=censor_threshold,
        )
        validation_observation = validate_cache_observation_contract(
            validation_cache,
            dataset_spec=dataset_spec,
            split="validation",
            censor_threshold=censor_threshold,
        )

    candidate, candidate_encoder, candidate_metadata = (
        build_frozen_lognormal_candidate(
            source_payload,
            train_time_statistics=train_statistics,
            time_sigma_floor=0.001,
            source_backbone=str(source_spec["backbone"]),
            source_variant=LOG_MSE_VARIANT,
            max_seq_len=max_seq_len,
            training_stage="backbone_normalized_duration_refit",
        )
    )
    require(
        candidate_metadata["source_non_time_state_sha256"]
        == source_non_time_sha,
        "Normalized duration candidate changed source non-time state",
    )
    del source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    candidate = candidate.to(device)
    source_lineage = {
        "job_manifest_sha256": manifest_sha,
        "execution_contract_sha256": contract_sha,
        "aligned_contract_sha256": aligned_sha,
        "fit_contract_binding_sha256": fit_contract_binding_sha,
        "evaluation_source_revision": args.source_revision,
        "model_role": role,
        "backbone": source_spec["backbone"],
        "checkpoint_file_sha256": checkpoint_sha,
        "checkpoint_state_sha256": source_payload["model_state_sha256"],
        "source_non_time_state_sha256": source_non_time_sha,
        "source_summary_sha256": source_spec["summary_sha256"],
        "source_history_sha256": source_spec["history_sha256"],
        "checkpoint_selection": source_spec["checkpoint_selection"],
        "training_source_revision": source_payload["source_revision"],
        "training_source_revision_history": source_payload[
            "source_revision_history"
        ],
        "calibration_source_revision": ALIGNED_CALIBRATION_SOURCE_REVISION,
    }
    summary = fit_time_head_from_cache(
        model=candidate,
        train_cache=train_cache,
        validation_cache=validation_cache,
        output_dir=args.output_dir,
        dataset=dataset_name,
        censor_threshold=censor_threshold,
        contract_sha256=fit_contract_binding_sha,
        source_checkpoint_sha256=checkpoint_sha,
        source_state_sha256=str(source_payload["model_state_sha256"]),
        source_non_time_state_sha256=source_non_time_sha,
        candidate_initial_state_sha256=candidate_metadata[
            "candidate_initial_state_sha256"
        ],
        train_time_statistics=train_statistics,
        calibration_source_revision=ALIGNED_CALIBRATION_SOURCE_REVISION,
        seed=int(fit_settings["seed"]),
        planned_epochs=planned_epochs,
        learning_rate=float(fit_settings["learning_rate"]),
        weight_decay=float(fit_settings["weight_decay"]),
        batch_size=int(fit_settings["cached_state_batch_size"]),
        quantity_replay_batch_size=encoder_batch_size,
        grad_clip=float(fit_settings["gradient_clip"]),
        min_epochs=int(fit_settings["minimum_epochs"]),
        patience=int(fit_settings["early_stopping_patience"]),
        device=device,
        candidate_metadata=candidate_metadata,
        source_metadata=source_lineage,
        contract_id=CONTRACT_ID,
        model_role=role,
        source_backbone=str(source_spec["backbone"]),
        source_variant=LOG_MSE_VARIANT,
        observation_likelihood_contract=observation_contract,
    )
    if summary.get("status") != "success":
        return summary

    selected_path = args.output_dir / SELECTED_CHECKPOINT_NAME
    selected = torch_load_checkpoint(selected_path, map_location="cpu")
    _selected_fit_audit(
        summary=summary,
        selected=selected,
        role=role,
        source_non_time_sha256=source_non_time_sha,
        observation_contract=observation_contract,
        source_lineage=source_lineage,
    )
    candidate.load_state_dict(selected["model_state_dict"], strict=True)
    candidate.eval()
    continuous_metrics = evaluate_cached_time_metrics(
        model=candidate,
        cache=validation_cache,
        device=device,
        batch_size=int(fit_settings["cached_state_batch_size"]),
        censor_threshold=censor_threshold,
        observation_likelihood_mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
    )
    summary = dict(summary)
    summary.update(
        {
            "contract_id": CONTRACT_ID,
            "execution_contract_sha256": contract_sha,
            "aligned_contract_sha256": aligned_sha,
            "job_manifest_sha256": manifest_sha,
            "evaluation_source_revision": args.source_revision,
            "fit_contract_binding_sha256": fit_contract_binding_sha,
            "model_role": role,
            "encoder_config": candidate_encoder,
            "interface_meta": candidate_metadata["interface_meta"],
            "train_target_population": train_population,
            "validation_target_population": validation_population,
            "train_observation_contract": train_observation,
            "validation_observation_contract": validation_observation,
            "qualified_full_data": full_data,
            "qualified_full_fit": qualified_full_fit,
            "feature_cache_dir": str(cache_dir.resolve()),
            "selected_checkpoint_file_sha256": sha256_file(selected_path),
            "validation_continuous_reference_metrics": continuous_metrics,
        }
    )
    assert validation_cache.source_quantity_prediction is not None
    assert validation_cache.target_quantity is not None
    summary["quantity_metrics"].update(
        stratified_quantity_metrics(
            validation_cache.source_quantity_prediction,
            validation_cache.target_quantity,
            body_max=float(dataset_spec["reporting_body_max_train_p95"]),
            tail_min_exclusive=float(
                dataset_spec["reporting_tail_min_exclusive_train_p99"]
            ),
            require_nonempty=full_data,
        )
    )
    summary["runtime"] = {
        "requested_device": str(device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        ),
    }
    save_json(args.output_dir / SUMMARY_NAME, summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
