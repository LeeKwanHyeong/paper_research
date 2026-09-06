#!/usr/bin/env python3
"""Run matched frozen proper duration heads on one idle RTX 5090.

The controller is deliberately fail closed.  It pins source, contract, data, split manifests, eight available source
checkpoints, and the previously audited B result before work.  Eight new rows
then run serially as one full-data epoch followed by a full fit that reuses the
same immutable feature cache.  Every artifact is independently replayed; the
known missing Intermittent RMTPP row keeps the four-way claim incomplete.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTRACT_REL = Path("paper/contracts/matched_frozen_lognormal_duration_v1.json")
RUNNER_REL = Path("paper/scripts/run_matched_frozen_lognormal_duration.py")
TEST_REL = Path("simple_lab_test/search/tests/test_matched_frozen_lognormal_duration.py")
BASE_RUNNER_TEST_REL = Path(
    "simple_lab_test/search/tests/test_hard_lmm_frozen_lognormal_duration.py"
)
CONTROLLER_TEST_REL = Path(
    "simple_lab_test/search/tests/test_matched_frozen_lognormal_duration_5090.py"
)
DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)
MODEL_ROLES = ("A", "B", "rmtpp", "thp")
NEW_EXECUTION_ORDER = (
    ("intermittent_frozen_5000", "A"),
    ("intermittent_frozen_5000", "thp"),
    ("yellow_trip_hourly", "A"),
    ("yellow_trip_hourly", "rmtpp"),
    ("yellow_trip_hourly", "thp"),
    ("insta_market_basket", "A"),
    ("insta_market_basket", "rmtpp"),
    ("insta_market_basket", "thp"),
)
MISSING_ROW = ("intermittent_frozen_5000", "rmtpp")
MODEL_TEST_REL = Path(
    "simple_lab_test/search/tests/"
    "test_count_aware_heteroscedastic_lognormal_time_head.py"
)
TIME_KEYS = (
    "v_t.weight",
    "b_t",
    "w_raw",
    "time_scale_weight.weight",
)
EXPECTED_CONTRACT_SHA256 = (
    "1152303afbe9152844038efc725164333eaec1ce1fcc3a02c539efd5437bf476"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            finite_tree(child, path=f"{path}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Non-finite value at {path}")


def finite_tensor_tree(value: Any, *, path: str = "root") -> None:
    import torch

    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"Non-finite tensor at {path}")
    elif isinstance(value, Mapping):
        for name, child in value.items():
            finite_tensor_tree(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tensor_tree(child, path=f"{path}[{index}]")


def validate_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate every policy that can affect the matched comparison."""

    require(
        contract.get("contract_id") == "matched_frozen_lognormal_duration_v1",
        "Wrong matched frozen log-normal duration contract",
    )
    scope = contract.get("scope")
    require(isinstance(scope, dict), "Contract scope is missing")
    require(scope.get("input_splits") == ["train", "validation"], "Split scope drift")
    require(scope.get("optimization_split") == "train", "Optimization split drift")
    require(scope.get("selection_split") == "validation", "Selection split drift")
    require(scope.get("evaluation_scope") == "validation_only", "Evaluation scope drift")
    require(scope.get("held_out_test") is False, "Held-out access is enabled")
    require(scope.get("seed") == 42, "Seed drift")
    require(scope.get("additional_seeds") is False, "Additional seeds are enabled")
    require(scope.get("datasets") == list(DATASETS), "Dataset scope drift")
    require(scope.get("model_roles") == list(MODEL_ROLES), "Model-role scope drift")

    policy = contract.get("comparison_policy")
    require(isinstance(policy, dict), "Comparison policy is missing")
    expected_policy = {
        "same_duration_head_for_every_role": True,
        "same_optimization_for_every_new_fit": True,
        "B_result_policy": "reuse the audited c04d32b full result without refitting",
        "cross_dataset_nll_aggregation": False,
        "legacy_nll_comparison": False,
        "four_way_complete_requires_all_twelve_rows": True,
    }
    for name, expected in expected_policy.items():
        require(policy.get(name) == expected, f"Comparison policy drift: {name}")

    role_specs = contract.get("model_roles")
    require(isinstance(role_specs, dict), "Model-role contract is missing")
    require(tuple(role_specs) == MODEL_ROLES, "Model-role order drift")
    expected_roles = {
        "A": ("titantpp", "fit_new"),
        "B": ("titantpp", "reuse_c04d32b"),
        "rmtpp": ("rmtpp", "fit_new_when_available"),
        "thp": ("thp", "fit_new"),
    }
    for role, (backbone, execution) in expected_roles.items():
        require(role_specs[role].get("backbone") == backbone, f"Backbone drift: {role}")
        require(role_specs[role].get("execution") == execution, f"Execution drift: {role}")
    time_head = contract.get("time_head")
    require(isinstance(time_head, dict), "Time-head contract is missing")
    require(
        time_head.get("trainable_parameter_names") == list(TIME_KEYS),
        "Time-head parameter boundary drift",
    )
    require(
        time_head.get("expected_trainable_parameter_count") == 130,
        "Parameter count drift",
    )
    expected_time_head = {
        "location": "mu(h) = v_t(h) + b_t",
        "scale": (
            "sigma(h) = 0.001 + softplus(time_scale_weight(h) + w_raw)"
        ),
        "time_scale": "exact train-target median",
        "location_initialization": (
            "zero weight and float32 parameter initialized from the exact train "
            "mean of log(dt / train median)"
        ),
        "scale_initialization": (
            "zero weight and float32 parameter initialized from the exact train "
            "standard deviation of log(dt / train median)"
        ),
        "original_unit_jacobian": True,
        "duration_clamp": False,
        "calculation_dtype": "float64",
    }
    for name, expected in expected_time_head.items():
        require(time_head.get(name) == expected, f"Time-head drift: {name}")
    boundary = contract.get("parameter_boundary")
    require(isinstance(boundary, dict), "Parameter boundary is missing")
    require(boundary.get("encoder_mode") == "eval", "Encoder eval contract drift")
    require(boundary.get("hidden_state_gradient") == "detached", "Detach contract drift")
    require(boundary.get("dropout_during_cache") is False, "Dropout contract drift")
    require(
        boundary.get("target_quantity_masked_before_encoding") is True,
        "Target-quantity masking drift",
    )
    require(boundary.get("target_memory_write_masked") is True, "Memory masking drift")
    optimization = contract.get("optimization")
    require(isinstance(optimization, dict), "Optimization contract is missing")
    expected_optimization = {
        "shared_across_all_datasets": True,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "encoder_batch_size": 128,
        "cached_state_batch_size": 8192,
        "gradient_clip": 1.0,
        "epochs": 100,
        "minimum_epochs": 20,
        "early_stopping_patience": 20,
        "dataset_specific_hyperparameters": False,
        "training_objective": (
            "mean negative censor-aware normalized log-normal duration log likelihood"
        ),
        "scheduler": None,
    }
    for name, expected in expected_optimization.items():
        require(optimization.get(name) == expected, f"Optimization drift: {name}")
    selection = contract.get("checkpoint_selection")
    require(isinstance(selection, dict), "Selection contract is missing")
    require(
        selection.get("monitor") == "validation_proper_time_nll",
        "Monitor drift",
    )
    require(selection.get("rule") == "earliest strict finite minimum", "Rule drift")
    require(selection.get("fallback") == "epoch 0", "Epoch-zero fallback drift")
    require(selection.get("other_metrics_in_selector") is False, "Selector metric drift")
    stability = contract.get("identity_and_stability")
    require(isinstance(stability, dict), "Stability contract is missing")
    expected_stability = {
        "quantity_mae_and_rmse_required": "calculation-identical to each pinned source checkpoint",
        "proper_time_nll_replay_absolute_tolerance": 1e-6,
        "reported_quantity_metric_absolute_tolerance": 1e-6,
        "reported_quantity_metric_relative_tolerance": 1e-5,
        "finite_train_loss": True,
        "finite_validation_loss": True,
        "finite_gradient": True,
        "finite_model_and_optimizer_state": True,
        "selected_checkpoint_replay_required": True,
        "density_integrates_to_one_required": True,
        "survival_monotonic_required": True,
        "median_survival_equals_one_half_required": True,
        "target_dt_sha256_required": True,
        "censor_mask_sha256_required": True,
    }
    for name, expected in expected_stability.items():
        require(stability.get(name) == expected, f"Stability contract drift: {name}")
    rows = contract.get("datasets")
    require(isinstance(rows, list), "Dataset contract is missing")
    datasets = {str(row["dataset"]): dict(row) for row in rows}
    require(tuple(datasets) == DATASETS, "Dataset order or scope drift")
    for name, row in datasets.items():
        require(int(row["expected_train_targets"]) > 0, f"Empty train target set: {name}")
        require(
            int(row["expected_validation_targets"]) > 0,
            f"Empty validation target set: {name}",
        )
        for hash_name in (
            "expected_train_target_identity_sha256",
            "expected_train_target_quantity_sha256",
            "expected_validation_target_identity_sha256",
            "expected_validation_target_quantity_sha256",
            "expected_train_target_dt_sha256",
            "expected_validation_target_dt_sha256",
            "expected_train_censor_mask_sha256",
            "expected_validation_censor_mask_sha256",
        ):
            require(
                len(str(row.get(hash_name))) == 64,
                f"Pinned hash is missing: {name}.{hash_name}",
            )
        require(
            float(row["train_time_scale"]) > 0.0
            and math.isfinite(float(row["train_log_scaled_mean"]))
            and float(row["train_log_scaled_std"]) > 0.0,
            f"Invalid train-only duration statistics: {name}",
        )
        is_instacart = name == "insta_market_basket"
        require(
            row.get("right_censor_threshold") == (30.0 if is_instacart else None),
            f"Censor threshold drift: {name}",
        )
        expected_train_censored = 189607 if is_instacart else 0
        expected_validation_censored = 63923 if is_instacart else 0
        require(
            int(row["expected_train_censored_targets"])
            == expected_train_censored,
            f"Train censor count drift: {name}",
        )
        require(
            int(row["expected_validation_censored_targets"])
            == expected_validation_censored,
            f"Validation censor count drift: {name}",
        )
    likelihood = contract.get("observation_likelihood")
    require(isinstance(likelihood, dict), "Observation likelihood is missing")
    require(
        likelihood.get("uncensored") == "log f(dt | h) in the original time unit",
        "Uncensored likelihood drift",
    )
    require(
        likelihood.get("right_censored") == "log S(threshold | h)",
        "Censored likelihood drift",
    )
    reuse = contract.get("B_reuse")
    require(isinstance(reuse, dict), "B reuse contract is missing")
    require(
        reuse.get("required_source_revision")
        == "c04d32b0ba80a463097d91c4128ea966f22ec660",
        "B result revision drift",
    )
    require(
        reuse.get("artifact_contract_id") == "hard_lmm_frozen_lognormal_duration_v1",
        "B artifact contract drift",
    )
    require(reuse.get("required_decision") == "accepted", "B decision policy drift")
    for key in ("decision_file_sha256",):
        require(len(str(reuse.get(key))) == 64, f"B reuse hash is missing: {key}")

    execution = contract.get("execution_order")
    require(isinstance(execution, list), "Execution order is missing")
    parsed_order = tuple(
        (str(item.get("dataset")), str(item.get("model_role")))
        for item in execution
    )
    require(parsed_order == NEW_EXECUTION_ORDER, "Execution order drift")
    require(
        all(item.get("phase_order") == ["e1", "full"] for item in execution),
        "Per-row phase order drift",
    )

    new_count = 0
    missing_count = 0
    for dataset, row in datasets.items():
        sources = row.get("sources")
        require(isinstance(sources, dict), f"Source map is missing: {dataset}")
        require(tuple(sources) == MODEL_ROLES, f"Source role order drift: {dataset}")
        for role in MODEL_ROLES:
            source = sources[role]
            require(source.get("backbone") == expected_roles[role][0], f"Source backbone drift: {dataset}/{role}")
            require(source.get("quantity_variant") == "count_only_log_regression", f"Source variant drift: {dataset}/{role}")
            require(source.get("seed") == 42, f"Source seed drift: {dataset}/{role}")
            require(source.get("evaluation_scope") == "validation_only", f"Source scope drift: {dataset}/{role}")
            require(source.get("held_out_test_evaluated") is False, f"Source held-out drift: {dataset}/{role}")
            if role == "B":
                require(source.get("available") is True and source.get("execution") == "reuse_c04d32b", f"B reuse drift: {dataset}")
                for key in (
                    "source_checkpoint_file_sha256",
                    "source_checkpoint_state_sha256",
                    "matched_summary_file_sha256",
                    "matched_selected_checkpoint_file_sha256",
                    "matched_selected_state_sha256",
                ):
                    require(len(str(source.get(key))) == 64, f"B hash missing: {dataset}.{key}")
                require(source.get("quantity_prediction_bitwise_identical") is True, f"B quantity identity drift: {dataset}")
            elif (dataset, role) == MISSING_ROW:
                missing_count += 1
                require(source.get("available") is False, "Intermittent RMTPP unexpectedly marked available")
                require(source.get("execution") == "blocked_missing_checkpoint", "Missing row execution drift")
                require(source.get("checkpoint_path") is None and source.get("checkpoint_file_sha256") is None, "Missing row pins a checkpoint binary")
                require(len(str(source.get("checkpoint_state_sha256"))) == 64, "Missing row state identity absent")
                require(bool(source.get("missing_reason")), "Missing row reason absent")
            else:
                new_count += 1
                require(source.get("available") is True and source.get("execution") == "fit_new", f"New-row availability drift: {dataset}/{role}")
                for key in ("checkpoint_file_sha256", "checkpoint_state_sha256"):
                    require(len(str(source.get(key))) == 64, f"Source hash missing: {dataset}/{role}.{key}")
                require(bool(source.get("checkpoint_path")), f"Source checkpoint path missing: {dataset}/{role}")
                require(len(str(source.get("training_source_revision"))) == 40, f"Training revision missing: {dataset}/{role}")
    require(new_count == 8, f"Expected eight new fits, got {new_count}")
    require(missing_count == 1, f"Expected one missing row, got {missing_count}")

    acceptance = contract.get("acceptance")
    require(isinstance(acceptance, dict), "Acceptance contract is missing")
    require(
        acceptance.get("four_way_claim")
        == "prohibited until all twelve dataset-role rows are present and audited",
        "Four-way claim policy drift",
    )
    require(
        acceptance.get("current_known_blocker")
        == "Intermittent RMTPP full-e300 checkpoint binary is unavailable",
        "Known blocker drift",
    )
    return datasets


def verify_source(project: Path, revision: str) -> dict[str, Any]:
    require(
        len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
        "A full lowercase source revision is required",
    )
    manifest = read_json(project / "source_manifest.json")
    require(manifest.get("source_revision") == revision, "Source revision mismatch")
    require(manifest.get("held_out_test_evaluated") is False, "Source manifest opened held-out")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Source manifest is empty")
    for relative_name, expected in files.items():
        relative = Path(relative_name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        path = project / relative
        require(path.is_file(), f"Manifest file is missing: {relative_name}")
        require(sha256_file(path) == expected, f"Manifest checksum drift: {relative_name}")
    for critical in (
        CONTRACT_REL,
        RUNNER_REL,
        TEST_REL,
        BASE_RUNNER_TEST_REL,
        MODEL_TEST_REL,
        CONTROLLER_TEST_REL,
        Path(__file__).resolve().relative_to(PROJECT_ROOT.resolve()),
    ):
        require(str(critical) in files, f"Critical source is not manifested: {critical}")
    for path in project.rglob("*.py"):
        relative_name = str(path.relative_to(project))
        require(
            relative_name in files,
            f"Python source is not manifested: {relative_name}",
        )
    return manifest


def _safe_relative_path(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe {label} path")
    require(
        not any("held_out" in part.lower() or part.lower().startswith("test_") for part in path.parts),
        f"Held-out path prohibited for {label}",
    )
    return path


def source_checkpoint_path(
    input_root: Path,
    row: Mapping[str, Any],
    model_role: str,
) -> Path:
    source = row["sources"][model_role]
    require(source.get("available") is True, f"Unavailable source row: {row['dataset']}/{model_role}")
    return input_root / _safe_relative_path(
        source["checkpoint_path"], label=f"{row['dataset']}/{model_role} checkpoint"
    )


def verify_inputs(
    input_root: Path,
    datasets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    import torch

    from paper.scripts.run_matched_frozen_lognormal_duration import validate_source_checkpoint
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
    )

    verified_checkpoints: list[dict[str, Any]] = []
    for name, row in datasets.items():
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            path = input_root / _safe_relative_path(row[path_key], label=f"{name}.{path_key}")
            require(path.is_file(), f"Missing {name} input: {path_key}")
            require(sha256_file(path) == row[hash_key], f"{name} {path_key} checksum drift")
        for model_role in MODEL_ROLES:
            source = row["sources"][model_role]
            if model_role == "B" or source.get("available") is not True:
                continue
            checkpoint = source_checkpoint_path(input_root, row, model_role)
            require(checkpoint.is_file(), f"Missing source checkpoint: {name}/{model_role}")
            file_sha = sha256_file(checkpoint)
            require(
                file_sha == source["checkpoint_file_sha256"],
                f"Source checkpoint checksum drift: {name}/{model_role}",
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict")
            require(isinstance(state, Mapping), f"Missing source model state: {name}/{model_role}")
            finite_tensor_tree(state, path=f"source_model_state.{name}.{model_role}")
            state_sha = canonical_state_dict_sha256(state)
            require(
                state_sha == source["checkpoint_state_sha256"],
                f"Source model-state checksum drift: {name}/{model_role}",
            )
            validate_source_checkpoint(payload, source_spec=source)
            verified_checkpoints.append(
                {
                    "dataset": name,
                    "model_role": model_role,
                    "path": str(checkpoint),
                    "file_sha256": file_sha,
                    "state_sha256": state_sha,
                }
            )
    require(len(verified_checkpoints) == 8, "Exactly eight source checkpoints must verify")
    require(
        [(item["dataset"], item["model_role"]) for item in verified_checkpoints]
        == list(NEW_EXECUTION_ORDER),
        "Verified checkpoint order drift",
    )
    return {
        "input_root": str(input_root),
        "verified_checkpoint_count": len(verified_checkpoints),
        "verified_checkpoints": verified_checkpoints,
        "missing_row": {
            "dataset": MISSING_ROW[0],
            "model_role": MISSING_ROW[1],
            "status": "blocked_missing_checkpoint",
        },
        "held_out_test_evaluated": False,
    }


def verify_b_reuse(
    b_result_root: Path,
    *,
    input_root: Path,
    contract: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate and semantically re-audit the immutable c04d32b result."""

    import torch

    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
        state_partition_sha256,
    )
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    reuse = contract["B_reuse"]
    b_manifest_path = b_result_root / _safe_relative_path(
        reuse["source_manifest_path"], label="B source manifest"
    )
    require(b_manifest_path.is_file(), "B source manifest is missing")
    require(
        sha256_file(b_manifest_path) == reuse["source_manifest_file_sha256"],
        "B source manifest checksum drift",
    )
    b_manifest = read_json(b_manifest_path)
    require(
        b_manifest.get("source_revision") == reuse["required_source_revision"],
        "B source revision drift",
    )
    require(
        b_manifest.get("contract_sha256")
        == "5316ccd3d3c041f80393a4357308e5bbaeddb08a2c5873e81d5b2d7b2026157b",
        "B source contract checksum drift",
    )
    require(b_manifest.get("held_out_test_evaluated") is False, "B source manifest used held-out test")
    decision_path = b_result_root / _safe_relative_path(
        reuse["decision_path"], label="B decision"
    )
    require(decision_path.is_file(), "B decision is missing")
    require(
        sha256_file(decision_path) == reuse["decision_file_sha256"],
        "B decision checksum drift",
    )
    decision = read_json(decision_path)
    require(decision.get("status") == "complete", "B result is incomplete")
    require(
        decision.get("candidate_acceptance") == reuse["required_decision"],
        "B result was not accepted",
    )
    require(
        decision.get("all_runtime_and_audit_contracts_passed") is True,
        "B result audit did not pass",
    )
    require(
        decision.get("quantity_exactly_preserved_on_all_datasets") is True,
        "B result changed quantity predictions",
    )
    require(decision.get("evaluation_scope") == "validation_only", "B scope drift")
    require(decision.get("held_out_test_evaluated") is False, "B used held-out test")
    require(decision.get("legacy_nll_compared") is False, "B compared legacy NLL")
    require(decision.get("cross_dataset_nll_aggregated") is False, "B aggregated NLL across datasets")
    require(decision.get("additional_seeds_executed") is False, "B used extra seeds")
    decision_rows = decision.get("datasets")
    require(isinstance(decision_rows, list), "B decision rows are missing")
    require([item.get("dataset") for item in decision_rows] == list(DATASETS), "B decision dataset order drift")

    verified: list[dict[str, Any]] = []
    for dataset in DATASETS:
        b = datasets[dataset]["sources"]["B"]
        summary_path = b_result_root / _safe_relative_path(
            b["matched_summary_path"], label=f"B {dataset} summary"
        )
        selected_path = b_result_root / _safe_relative_path(
            b["matched_selected_checkpoint_path"], label=f"B {dataset} selected checkpoint"
        )
        require(summary_path.is_file(), f"B summary is missing: {dataset}")
        require(selected_path.is_file(), f"B selected checkpoint is missing: {dataset}")
        require(
            sha256_file(summary_path) == b["matched_summary_file_sha256"],
            f"B summary checksum drift: {dataset}",
        )
        require(
            sha256_file(selected_path) == b["matched_selected_checkpoint_file_sha256"],
            f"B selected checkpoint checksum drift: {dataset}",
        )
        summary = read_json(summary_path)
        finite_tree(summary)
        expected_summary = {
            "contract_id": reuse["artifact_contract_id"],
            "status": "success",
            "dataset": dataset,
            "seed": 42,
            "time_head_mode": "heteroscedastic_lognormal_duration",
            "calculation_dtype": "float64",
            "trainable_parameter_names": list(TIME_KEYS),
            "trainable_parameter_count": 130,
            "encoder_mode_during_refit": "eval",
            "hidden_state_gradient": "detached_cache",
            "selection": "earliest_strict_finite_minimum_validation_proper_time_nll",
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "legacy_nll_compared": False,
            "qualified_full_data": True,
            "quantity_prediction_bitwise_identical": True,
            "source_checkpoint_sha256": b["source_checkpoint_file_sha256"],
            "source_state_sha256": b["source_checkpoint_state_sha256"],
            "selected_state_sha256": b["matched_selected_state_sha256"],
            "best_epoch": b["best_epoch"],
        }
        for key, expected in expected_summary.items():
            require(summary.get(key) == expected, f"B summary policy drift: {dataset}.{key}")
        _assert_close(
            summary["best_validation_proper_time_nll"],
            b["best_validation_proper_time_nll"],
            label=f"B selected proper Time NLL: {dataset}",
            absolute_tolerance=1e-12,
        )
        history = summary.get("history")
        require(isinstance(history, list) and bool(history), f"B history is missing: {dataset}")
        selected_row = min(
            history,
            key=lambda item: (float(item["val_proper_time_nll"]), int(item["epoch"])),
        )
        require(int(selected_row["epoch"]) == int(b["best_epoch"]), f"B selector drift: {dataset}")
        _assert_close(
            selected_row["val_proper_time_nll"],
            b["best_validation_proper_time_nll"],
            label=f"B history selected NLL: {dataset}",
            absolute_tolerance=1e-12,
        )
        resume_identity = summary.get("resume_identity")
        require(isinstance(resume_identity, Mapping), f"B resume identity missing: {dataset}")
        for key, expected in (
            ("seed", 42),
            ("optimizer", "AdamW"),
            ("learning_rate", contract["optimization"]["learning_rate"]),
            ("weight_decay", contract["optimization"]["weight_decay"]),
            ("cached_state_batch_size", contract["optimization"]["cached_state_batch_size"]),
            ("minimum_epochs", contract["optimization"]["minimum_epochs"]),
            ("early_stopping_patience", contract["optimization"]["early_stopping_patience"]),
            ("trainable_parameter_names", list(TIME_KEYS)),
            ("evaluation_scope", "validation_only"),
            ("held_out_test_evaluated", False),
            ("legacy_nll_compared", False),
        ):
            require(resume_identity.get(key) == expected, f"B optimization drift: {dataset}.{key}")

        checkpoint = torch.load(selected_path, map_location="cpu", weights_only=False)
        require(checkpoint.get("evaluation_scope") == "validation_only", f"B checkpoint scope drift: {dataset}")
        require(checkpoint.get("held_out_test_evaluated") is False, f"B checkpoint held-out drift: {dataset}")
        require(checkpoint.get("legacy_nll_compared") is False, f"B checkpoint legacy drift: {dataset}")
        require(checkpoint.get("time_head_mode") == "heteroscedastic_lognormal_duration", f"B checkpoint head drift: {dataset}")
        state = checkpoint.get("model_state_dict")
        require(isinstance(state, Mapping), f"B selected state missing: {dataset}")
        finite_tensor_tree(state, path=f"B_selected_state.{dataset}")
        require(
            all(name in state for name in TIME_KEYS)
            and sum(int(state[name].numel()) for name in TIME_KEYS) == 130,
            f"B selected duration-head capacity drift: {dataset}",
        )
        state_sha = canonical_state_dict_sha256(state)
        require(
            state_sha == checkpoint.get("model_state_sha256") == b["matched_selected_state_sha256"],
            f"B selected state checksum drift: {dataset}",
        )
        source_path = input_root / _safe_relative_path(
            b["source_checkpoint_path"], label=f"B {dataset} source checkpoint"
        )
        require(source_path.is_file(), f"B source checkpoint is missing: {dataset}")
        require(
            sha256_file(source_path) == b["source_checkpoint_file_sha256"],
            f"B source checkpoint checksum drift: {dataset}",
        )
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
        source_state = source_payload.get("model_state_dict")
        require(isinstance(source_state, Mapping), f"B source state missing: {dataset}")
        finite_tensor_tree(source_state, path=f"B_source_state.{dataset}")
        source_state_sha = canonical_state_dict_sha256(source_state)
        require(
            source_state_sha == b["source_checkpoint_state_sha256"],
            f"B source state checksum drift: {dataset}",
        )
        require(
            {name for name in source_state if name not in TIME_KEYS}
            == {name for name in state if name not in TIME_KEYS},
            f"B source/selected non-time key-set drift: {dataset}",
        )
        for name, tensor in source_state.items():
            if name not in TIME_KEYS:
                require(
                    torch.equal(tensor, state[name]),
                    f"B selected non-time tensor drift: {dataset}.{name}",
                )
        source_non_time_sha = state_partition_sha256(source_state, time_head=False)
        require(
            source_non_time_sha
            == state_partition_sha256(state, time_head=False)
            == summary.get("source_non_time_state_sha256")
            == summary.get("selected_non_time_state_sha256"),
            f"B non-time state partition drift: {dataset}",
        )
        quantity_metrics = summary.get("quantity_metrics")
        require(isinstance(quantity_metrics, Mapping), f"B quantity metrics missing: {dataset}")
        for metric in ("mae", "rmse", "body_mae", "gt_p99_mae"):
            require(
                math.isfinite(float(quantity_metrics[metric])),
                f"B non-finite quantity metric: {dataset}.{metric}",
            )
        source_quantity_sha = summary.get("source_quantity_prediction_sha256")
        require(
            isinstance(source_quantity_sha, str)
            and len(source_quantity_sha) == 64
            and source_quantity_sha == summary.get("selected_quantity_prediction_sha256"),
            f"B source/selected quantity identity drift: {dataset}",
        )
        verified.append(
            {
                "dataset": dataset,
                "model_role": "B",
                "status": "reused_audited",
                "best_epoch": int(b["best_epoch"]),
                "epoch_zero_validation_proper_time_nll": float(
                    summary["epoch_zero_validation_proper_time_nll"]
                ),
                "selected_validation_proper_time_nll": float(b["best_validation_proper_time_nll"]),
                "summary_sha256": b["matched_summary_file_sha256"],
                "selected_checkpoint_sha256": b["matched_selected_checkpoint_file_sha256"],
                "selected_state_sha256": state_sha,
                "quantity_prediction_bitwise_identical": True,
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "legacy_nll_compared": False,
            }
        )

    forbidden = [
        str(path)
        for path in b_result_root.rglob("*")
        if path.is_file()
        and (path.name.lower().startswith("test_") or "held_out" in path.name.lower())
    ]
    require(not forbidden, f"Held-out artifact found in B result: {forbidden}")
    return {
        "status": "verified",
        "required_source_revision": reuse["required_source_revision"],
        "decision_sha256": reuse["decision_file_sha256"],
        "rows": verified,
        "same_duration_head_policy_verified": True,
        "same_optimization_policy_verified": True,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def command_output(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode not in allowed:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def gpu_preflight() -> dict[str, Any]:
    rows = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.free",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    require(len(rows) == 1, "Expected exactly one GPU")
    gpu_name, raw_free = (part.strip() for part in rows[0].split(",", 1))
    free_mib = int(raw_free)
    require("RTX 5090" in gpu_name, f"Wrong GPU: {gpu_name}")
    require(free_mib >= 12000, f"Insufficient free VRAM: {free_mib} MiB")
    compute = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    require(not compute, f"GPU already has a compute process: {compute}")
    gdm = command_output(["systemctl", "is-active", "gdm"], allowed=(0, 3))
    require(gdm == "inactive", f"GDM must be inactive, got {gdm!r}")
    return {
        "gpu_name": gpu_name,
        "free_vram_mib": free_mib,
        "compute_processes": [],
        "gdm": gdm,
    }


def deterministic_env(project: Path, *, input_root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(project),
        "HARD_LMM_FROZEN_LOGNORMAL_REQUIRE_CUDA": "1",
        "MATCHED_FROZEN_LOGNORMAL_REQUIRE_CUDA": "1",
        "MATCHED_FROZEN_LOGNORMAL_INPUT_ROOT": str(input_root),
    }


def runner_command(
    *,
    python: str,
    project: Path,
    input_root: Path,
    output: Path,
    revision: str,
    row: Mapping[str, Any],
    model_role: str,
    phase: str,
) -> list[str]:
    require(phase in {"e1", "full"}, f"Unknown phase: {phase}")
    require((str(row["dataset"]), model_role) in NEW_EXECUTION_ORDER, "Unscheduled row")
    command = [
        python,
        "-s",
        str(project / RUNNER_REL),
        "--dataset",
        str(row["dataset"]),
        "--model-role",
        model_role,
        "--data",
        str(input_root / _safe_relative_path(row["data_path"], label="data")),
        "--split-manifest",
        str(input_root / _safe_relative_path(row["split_manifest_path"], label="split manifest")),
        "--checkpoint",
        str(source_checkpoint_path(input_root, row, model_role)),
        "--output-dir",
        str(output),
        "--calibration-source-revision",
        revision,
        "--contract",
        str(project / CONTRACT_REL),
        "--device",
        "cuda",
    ]
    if phase == "e1":
        command.extend(
            [
                "--allow-partial-contract",
                "--max-epochs",
                "1",
            ]
        )
    else:
        command.extend(
            [
                "--feature-cache-dir",
                str(
                    output.parents[2]
                    / "e1"
                    / str(row["dataset"])
                    / model_role
                    / "cache"
                ),
            ]
        )
    return command


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    project: Path,
    env: Mapping[str, str],
    child_slot: dict[str, subprocess.Popen[str] | None],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(
            command,
            cwd=project,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        child_slot["child"] = child
        return_code = child.wait()
        child_slot["child"] = None
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def audit_pytest(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    cases = list(root.iter("testcase"))
    require(bool(cases), "Contract suite executed no tests")
    require(not list(root.iter("failure")), "Contract suite failed")
    require(not list(root.iter("error")), "Contract suite errored")
    require(not list(root.iter("skipped")), "Contract suite skipped a test")
    return {
        "test_count": len(cases),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "xml_sha256": sha256_file(xml_path),
    }


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> None:
    require(
        math.isclose(
            float(actual),
            float(expected),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ),
        f"{label} drift: expected {expected}, got {actual}",
    )


def _expected_cache_identity(
    row: Mapping[str, Any], *, model_role: str, split: str
) -> dict[str, Any]:
    require(split in {"train", "validation"}, f"Unknown cache split: {split}")
    source = row["sources"][model_role]
    return {
        "schema_version": 1,
        "contract_id": "matched_frozen_lognormal_duration_v1",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "dataset": row["dataset"],
        "model_role": model_role,
        "source_backbone": source["backbone"],
        "source_variant": source["quantity_variant"],
        "data_sha256": row["data_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "source_checkpoint_sha256": source["checkpoint_file_sha256"],
        "source_state_sha256": source["checkpoint_state_sha256"],
        "encoder_mode": "eval",
        "target_quantity_masked": True,
        "memory_target_write_masked": True,
        "evaluation_scope": "train_and_validation_only",
        "held_out_test_evaluated": False,
        "split": split,
        "target_count": row[f"expected_{split}_targets"],
        "target_identity_sha256": row[
            f"expected_{split}_target_identity_sha256"
        ],
        "target_quantity_sha256": row[
            f"expected_{split}_target_quantity_sha256"
        ],
        "max_batches": None,
    }


def _audit_optimizer(
    last: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    current_state: Mapping[str, Any],
    completed_epochs: int,
    train_count: int,
) -> None:
    import torch

    optimizer_state = last.get("optimizer_state_dict")
    require(isinstance(optimizer_state, Mapping), "Last optimizer state is missing")
    finite_tensor_tree(optimizer_state, path="optimizer_state_dict")
    optimizer_rows = optimizer_state.get("state")
    optimizer_groups = optimizer_state.get("param_groups")
    require(
        isinstance(optimizer_rows, Mapping)
        and len(optimizer_rows) == len(TIME_KEYS),
        "Last optimizer parameter coverage drift",
    )
    require(
        isinstance(optimizer_groups, list) and len(optimizer_groups) == 1,
        "Last optimizer parameter-group drift",
    )
    parameter_ids = optimizer_groups[0].get("params", [])
    require(
        parameter_ids == list(range(len(TIME_KEYS))),
        "Last optimizer parameter list drift",
    )
    expected_group = {
        "lr": float(contract["optimization"]["learning_rate"]),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": float(contract["optimization"]["weight_decay"]),
        "amsgrad": False,
        "maximize": False,
    }
    for name, expected in expected_group.items():
        require(
            optimizer_groups[0].get(name) == expected,
            f"Last AdamW parameter-group drift: {name}",
        )
    optional_group = {
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    for name, expected in optional_group.items():
        if name in optimizer_groups[0]:
            require(
                optimizer_groups[0][name] == expected,
                f"Last AdamW parameter-group drift: {name}",
            )
    require(
        set(optimizer_rows) == set(parameter_ids),
        "Last optimizer state parameter IDs drift",
    )
    batch_size = int(contract["optimization"]["cached_state_batch_size"])
    expected_steps = completed_epochs * math.ceil(train_count / batch_size)
    for parameter_id, parameter_name in zip(parameter_ids, TIME_KEYS):
        parameter_state = optimizer_rows[parameter_id]
        require(
            isinstance(parameter_state, Mapping)
            and {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state),
            f"Last AdamW moment state is incomplete: {parameter_name}",
        )
        expected_parameter = current_state[parameter_name]
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            require(
                isinstance(moment, torch.Tensor)
                and moment.shape == expected_parameter.shape
                and moment.dtype == expected_parameter.dtype,
                f"Last AdamW {moment_name} shape or dtype drift: {parameter_name}",
            )
        step = parameter_state["step"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and step.dtype == torch.float32
            and float(step.item()) == float(expected_steps),
            f"Last AdamW step drift: {parameter_name}",
        )


def audit_run(
    output: Path,
    *,
    contract: Mapping[str, Any],
    row: Mapping[str, Any],
    model_role: str,
    phase: str,
    source_checkpoint: Path,
    source_revision: str,
) -> dict[str, Any]:
    import torch

    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
        FrozenFeatureCache,
        build_candidate_from_selected_checkpoint,
        build_frozen_lognormal_candidate,
        cached_quantity_predictions,
        censor_mask,
        censor_mask_sha256,
        evaluate_cached_time_metrics,
        quantity_metrics,
        state_partition_sha256,
        stratified_quantity_metrics,
        target_dt_sha256,
        tensor_sha256,
        train_time_statistics_from_contract,
    )
    from paper.scripts.run_matched_frozen_lognormal_duration import (
        build_source_model,
        validate_source_checkpoint,
    )
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    require(phase in {"e1", "full"}, f"Unknown audit phase: {phase}")
    require((str(row["dataset"]), model_role) in NEW_EXECUTION_ORDER, "Unscheduled audit row")
    source = row["sources"][model_role]
    require(source_checkpoint.is_file(), "Source checkpoint disappeared before audit")
    require(
        sha256_file(source_checkpoint) == source["checkpoint_file_sha256"],
        "Source checkpoint file checksum drift during audit",
    )
    summary = read_json(output / "summary.json")
    finite_tree(summary)
    require(summary.get("schema_version") == 1, "Wrong summary schema")
    require(
        summary.get("contract_id")
        == "matched_frozen_lognormal_duration_v1",
        "Summary contract drift",
    )
    require(summary.get("status") == "success", "Duration fit did not complete")
    require(summary.get("dataset") == row["dataset"], "Dataset identity drift")
    require(summary.get("model_role") == model_role, "Model-role identity drift")
    require(summary.get("source_backbone") == source["backbone"], "Source backbone drift")
    require(summary.get("source_variant") == source["quantity_variant"], "Source variant drift")
    require(summary.get("seed") == 42, "Seed drift")
    require(
        summary.get("time_head_mode")
        == "heteroscedastic_lognormal_duration",
        "Duration-head mode drift",
    )
    require(summary.get("calculation_dtype") == "float64", "Calculation dtype drift")
    require(summary.get("evaluation_scope") == "validation_only", "Scope drift")
    require(summary.get("held_out_test_evaluated") is False, "Held-out was evaluated")
    require(summary.get("legacy_nll_compared") is False, "Legacy NLL was compared")
    require(
        summary.get("source_checkpoint_sha256")
        == source["checkpoint_file_sha256"],
        "Summary source checkpoint digest drift",
    )
    require(
        summary.get("source_state_sha256") == source["checkpoint_state_sha256"],
        "Summary source model-state digest drift",
    )
    require(
        summary.get("trainable_parameter_names") == list(TIME_KEYS)
        and int(summary.get("trainable_parameter_count", -1)) == 130,
        "Trainable duration-head boundary drift",
    )
    require(summary.get("encoder_mode_during_refit") == "eval", "Encoder mode drift")
    require(
        summary.get("hidden_state_gradient") == "detached_cache",
        "Hidden states were not detached",
    )
    require(summary.get("qualified_full_data") is True, "Run was not full data")

    expected_cache_dir = (
        output / "cache"
        if phase == "e1"
        else output.parents[2] / "e1" / str(row["dataset"]) / model_role / "cache"
    ).resolve()
    require(
        Path(summary["feature_cache_dir"]).resolve() == expected_cache_dir,
        "Feature-cache route drift",
    )
    caches: dict[str, FrozenFeatureCache] = {}
    for split in ("train", "validation"):
        cache_payload = torch_load_checkpoint(
            expected_cache_dir / f"{split}_features.pt", map_location="cpu"
        )
        expected_identity = _expected_cache_identity(row, model_role=model_role, split=split)
        require(
            cache_payload.get("identity") == expected_identity,
            f"{split} feature-cache identity drift",
        )
        cache = FrozenFeatureCache.from_payload(
            cache_payload,
            expected_identity=expected_identity,
            require_quantity=split == "validation",
        )
        caches[split] = cache
        expected_count = int(row[f"expected_{split}_targets"])
        expected_dt_hash = row[f"expected_{split}_target_dt_sha256"]
        expected_mask_hash = row[f"expected_{split}_censor_mask_sha256"]
        expected_censored = int(row[f"expected_{split}_censored_targets"])
        threshold = row["right_censor_threshold"]
        observed_mask = censor_mask(cache.target_dt, threshold=threshold)
        require(cache.count == expected_count, f"{split} target count drift")
        require(
            target_dt_sha256(cache.target_dt) == expected_dt_hash,
            f"{split} target-dt digest drift",
        )
        require(
            censor_mask_sha256(observed_mask) == expected_mask_hash,
            f"{split} censor-mask digest drift",
        )
        require(
            int(observed_mask.sum().item()) == expected_censored,
            f"{split} right-censored count drift",
        )
        cache_summary = summary.get(f"{split}_cache")
        require(isinstance(cache_summary, Mapping), f"{split} cache summary is missing")
        require(int(cache_summary.get("count", -1)) == expected_count, f"{split} summary count drift")
        require(cache_summary.get("sha256") == cache.digest(), f"{split} cache digest drift")
        require(cache_summary.get("target_dt_sha256") == expected_dt_hash, f"{split} summary dt digest drift")
        require(cache_summary.get("censor_mask_sha256") == expected_mask_hash, f"{split} summary censor digest drift")
        population = summary.get(f"{split}_target_population")
        require(isinstance(population, Mapping), f"{split} target population is missing")
        require(int(population.get("target_count", -1)) == expected_count, f"{split} population count drift")
        require(
            population.get("target_identity_sha256")
            == row[f"expected_{split}_target_identity_sha256"],
            f"{split} target identity drift",
        )
        require(
            population.get("target_quantity_sha256")
            == row[f"expected_{split}_target_quantity_sha256"],
            f"{split} target quantity digest drift",
        )
        observation = summary.get(f"{split}_observation_contract")
        require(isinstance(observation, Mapping), f"{split} observation contract is missing")
        expected_observation = {
            "split": split,
            "target_count": expected_count,
            "target_dt_sha256": expected_dt_hash,
            "right_censor_threshold": threshold,
            "right_censored_count": expected_censored,
            "censor_mask_sha256": expected_mask_hash,
        }
        for name, expected in expected_observation.items():
            require(
                observation.get(name) == expected,
                f"{split} observation contract drift: {name}",
            )

    train_cache = caches["train"]
    validation_cache = caches["validation"]
    threshold = row["right_censor_threshold"]
    require(summary.get("right_censor_threshold") == threshold, "Summary censor threshold drift")
    require(
        int(summary.get("train_right_censored_count", -1))
        == int(row["expected_train_censored_targets"]),
        "Summary train censor count drift",
    )
    require(
        int(summary.get("validation_right_censored_count", -1))
        == int(row["expected_validation_censored_targets"]),
        "Summary validation censor count drift",
    )

    source_payload = torch_load_checkpoint(source_checkpoint, map_location="cpu")
    source_state = source_payload.get("model_state_dict")
    require(isinstance(source_state, Mapping), "Source model state is missing")
    finite_tensor_tree(source_state, path="source_model_state")
    require(
        canonical_state_dict_sha256(source_state)
        == source["checkpoint_state_sha256"],
        "Source model-state digest drift",
    )
    validate_source_checkpoint(source_payload, source_spec=source)
    source_non_time_sha = state_partition_sha256(source_state, time_head=False)
    require(
        summary.get("source_non_time_state_sha256") == source_non_time_sha,
        "Summary source non-time digest drift",
    )

    train_statistics = summary.get("train_time_statistics")
    require(isinstance(train_statistics, Mapping), "Train-only time statistics are missing")
    train_statistics = train_time_statistics_from_contract(row, train_statistics)
    for summary_name, row_name in (
        ("time_scale", "train_time_scale"),
        ("target_log_scaled_mean", "train_log_scaled_mean"),
        ("target_log_scaled_std", "train_log_scaled_std"),
    ):
        _assert_close(
            train_statistics[summary_name],
            row[row_name],
            label=f"train-only statistic {summary_name}",
            absolute_tolerance=1e-12,
        )
    initial_model, _, initial_metadata = build_frozen_lognormal_candidate(
        source_payload,
        train_time_statistics=train_statistics,
        time_sigma_floor=0.001,
        source_backbone=str(source["backbone"]),
        source_variant=str(source["quantity_variant"]),
        max_seq_len=int(row["max_sequence_length"]),
        training_stage="matched_frozen_representation_duration_refit",
    )
    initial_state = initial_model.state_dict()
    initial_state_sha = canonical_state_dict_sha256(initial_state)
    require(
        initial_state_sha
        == initial_metadata["candidate_initial_state_sha256"]
        == summary.get("candidate_initial_state_sha256"),
        "Candidate initial-state lineage drift",
    )
    require(
        summary.get("candidate_initial_time_head_state_sha256")
        == initial_metadata["candidate_initial_time_head_state_sha256"],
        "Candidate initial duration-head digest drift",
    )

    selected_path = output / "best_validation_proper_time_nll_model.pt"
    checkpoint = torch_load_checkpoint(selected_path, map_location="cpu")
    require(
        checkpoint.get("checkpoint_type")
        == "selected_frozen_lognormal_duration",
        "Wrong selected checkpoint type",
    )
    require(checkpoint.get("checkpoint_schema_version") == 1, "Wrong selected schema")
    require(checkpoint.get("evaluation_scope") == "validation_only", "Selected scope drift")
    require(checkpoint.get("held_out_test_evaluated") is False, "Selected held-out drift")
    require(checkpoint.get("legacy_nll_compared") is False, "Selected legacy NLL drift")
    require(checkpoint.get("model_role") == model_role, "Selected model-role drift")
    require(checkpoint.get("backbone") == source["backbone"], "Selected backbone drift")
    require(checkpoint.get("variant") == source["quantity_variant"], "Selected variant drift")
    require(
        checkpoint.get("time_head_mode")
        == "heteroscedastic_lognormal_duration",
        "Selected duration-head mode drift",
    )
    require(
        checkpoint.get("selection")
        == "earliest_strict_finite_minimum_validation_proper_time_nll",
        "Selected checkpoint rule drift",
    )
    require(
        checkpoint.get("selection_formula")
        == "mean negative censor-aware normalized log-normal duration log likelihood",
        "Selected checkpoint objective drift",
    )
    require(checkpoint.get("right_censor_threshold") == threshold, "Selected censor threshold drift")
    selected_state = checkpoint.get("model_state_dict")
    require(isinstance(selected_state, Mapping), "Selected model state is missing")
    finite_tensor_tree(selected_state, path="selected_model_state")
    require(
        all(name in selected_state for name in TIME_KEYS)
        and sum(int(selected_state[name].numel()) for name in TIME_KEYS) == 130,
        "Selected duration-head parameter count drift",
    )
    require(set(selected_state) == set(initial_state), "Selected model key set drift")
    selected_state_sha = canonical_state_dict_sha256(selected_state)
    require(
        checkpoint.get("model_state_sha256")
        == selected_state_sha
        == summary.get("selected_state_sha256"),
        "Selected model-state digest drift",
    )
    require(
        checkpoint.get("source_state_sha256")
        == source["checkpoint_state_sha256"],
        "Selected source-state lineage drift",
    )
    require(
        checkpoint.get("source_non_time_state_sha256")
        == source_non_time_sha,
        "Selected source non-time lineage drift",
    )
    require(
        checkpoint.get("candidate_initial_state_sha256") == initial_state_sha,
        "Selected initial-state lineage drift",
    )
    require(
        checkpoint.get("candidate_initial_time_head_state_sha256")
        == initial_metadata["candidate_initial_time_head_state_sha256"],
        "Selected initial duration-head lineage drift",
    )
    require(
        checkpoint.get("train_time_statistics") == dict(train_statistics),
        "Selected train-only statistics drift",
    )
    selected_non_time_sha = state_partition_sha256(selected_state, time_head=False)
    require(
        selected_non_time_sha
        == source_non_time_sha
        == summary.get("selected_non_time_state_sha256"),
        "Selected non-time state drift",
    )
    selected_time_sha = state_partition_sha256(selected_state, time_head=True)
    require(
        selected_time_sha
        == checkpoint.get("selected_time_head_state_sha256")
        == summary.get("selected_time_head_state_sha256"),
        "Selected duration-head digest drift",
    )
    changed_state_keys = sorted(
        name
        for name in initial_state
        if not torch.equal(initial_state[name], selected_state[name])
    )
    require(
        set(changed_state_keys).issubset(TIME_KEYS),
        f"State outside the duration head changed: {changed_state_keys}",
    )
    require(
        summary.get("changed_state_keys") == changed_state_keys,
        "Summary changed-state key drift",
    )
    for name in source_state:
        if name not in TIME_KEYS:
            require(
                torch.equal(source_state[name], selected_state[name]),
                f"Selected non-time tensor changed: {name}",
            )

    lineage = checkpoint.get("source_checkpoint_lineage")
    require(isinstance(lineage, Mapping), "Selected source lineage is missing")
    expected_lineage = {
        "model_role": model_role,
        "backbone": source["backbone"],
        "checkpoint_file_sha256": source["checkpoint_file_sha256"],
        "checkpoint_state_sha256": source["checkpoint_state_sha256"],
        "checkpoint_selection": source["checkpoint_selection"],
        "training_source_revision": source["training_source_revision"],
        "training_source_revision_history": source["training_source_revision_history"],
        "calibration_source_revision": source_revision,
    }
    for name, expected in expected_lineage.items():
        require(lineage.get(name) == expected, f"Selected source lineage drift: {name}")

    history = summary.get("history")
    require(isinstance(history, list) and bool(history), "History is missing")
    require(
        [int(item["epoch"]) for item in history] == list(range(len(history))),
        "History epochs are not contiguous",
    )
    for index, item in enumerate(history):
        require(
            math.isfinite(float(item["val_proper_time_nll"])),
            f"Non-finite validation proper NLL at epoch {index}",
        )
        if index == 0:
            require(item.get("train_proper_time_nll") is None, "Epoch zero has train loss")
        else:
            require(
                math.isfinite(float(item["train_proper_time_nll"])),
                f"Non-finite train proper NLL at epoch {index}",
            )
            require(
                math.isfinite(float(item["pre_clip_gradient_norm_mean"]))
                and math.isfinite(float(item["pre_clip_gradient_norm_max"])),
                f"Non-finite gradient telemetry at epoch {index}",
            )
    selected_row = min(
        history,
        key=lambda item: (
            float(item["val_proper_time_nll"]),
            int(item["epoch"]),
        ),
    )
    completed_epochs = int(summary["completed_epochs"])
    require(completed_epochs == len(history) - 1, "Completed epoch/history drift")
    if phase == "e1":
        require(completed_epochs == 1, "e1 did not complete exactly one epoch")
    else:
        require(1 <= completed_epochs <= int(contract["optimization"]["epochs"]), "Full fit epoch range drift")
        if summary.get("stopped_early") is False:
            require(
                completed_epochs == int(contract["optimization"]["epochs"]),
                "Non-early-stopped full fit did not reach the epoch limit",
            )
    require(int(summary["best_epoch"]) == int(selected_row["epoch"]), "Selector epoch drift")
    epoch_zero_nll = float(history[0]["val_proper_time_nll"])
    selected_nll = float(selected_row["val_proper_time_nll"])
    _assert_close(
        summary["epoch_zero_validation_proper_time_nll"],
        epoch_zero_nll,
        label="epoch-zero validation proper NLL",
        absolute_tolerance=1e-12,
    )
    _assert_close(
        summary["best_validation_proper_time_nll"],
        selected_nll,
        label="selected validation proper NLL",
        absolute_tolerance=1e-12,
    )
    _assert_close(
        checkpoint["selected_metric_value"],
        selected_nll,
        label="selected checkpoint metric",
        absolute_tolerance=1e-12,
    )
    require(int(checkpoint["best_epoch"]) == int(selected_row["epoch"]), "Selected checkpoint epoch drift")
    strict_improvement = selected_nll < epoch_zero_nll
    require(
        summary.get("strictly_improves_epoch_zero") is strict_improvement,
        "Strict epoch-zero improvement marker drift",
    )
    _assert_close(
        summary["proper_time_nll_improvement_from_epoch_zero"],
        epoch_zero_nll - selected_nll,
        label="proper NLL improvement",
        absolute_tolerance=1e-12,
    )

    resume_identity = summary.get("resume_identity")
    require(isinstance(resume_identity, Mapping), "Resume identity is missing")
    expected_planned_epochs = 1 if phase == "e1" else int(contract["optimization"]["epochs"])
    expected_resume_values = {
        "schema_version": 1,
        "contract_id": "matched_frozen_lognormal_duration_v1",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "dataset": row["dataset"],
        "model_role": model_role,
        "source_backbone": source["backbone"],
        "source_variant": source["quantity_variant"],
        "source_checkpoint_sha256": source["checkpoint_file_sha256"],
        "source_state_sha256": source["checkpoint_state_sha256"],
        "source_non_time_state_sha256": source_non_time_sha,
        "candidate_initial_state_sha256": initial_state_sha,
        "train_cache_sha256": train_cache.digest(),
        "validation_cache_sha256": validation_cache.digest(),
        "train_cache_count": train_cache.count,
        "validation_cache_count": validation_cache.count,
        "train_target_dt_sha256": row["expected_train_target_dt_sha256"],
        "validation_target_dt_sha256": row["expected_validation_target_dt_sha256"],
        "train_censor_mask_sha256": row["expected_train_censor_mask_sha256"],
        "validation_censor_mask_sha256": row["expected_validation_censor_mask_sha256"],
        "right_censor_threshold": threshold,
        "calibration_source_revision": source_revision,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": float(contract["optimization"]["learning_rate"]),
        "weight_decay": float(contract["optimization"]["weight_decay"]),
        "cached_state_batch_size": int(contract["optimization"]["cached_state_batch_size"]),
        "quantity_replay_batch_size": int(contract["optimization"]["encoder_batch_size"]),
        "gradient_clip": float(contract["optimization"]["gradient_clip"]),
        "minimum_epochs": int(contract["optimization"]["minimum_epochs"]),
        "early_stopping_patience": int(contract["optimization"]["early_stopping_patience"]),
        "planned_epochs": expected_planned_epochs,
        "selection": "earliest_strict_finite_minimum_validation_proper_time_nll",
        "trainable_parameter_names": list(TIME_KEYS),
        "time_head_mode": "heteroscedastic_lognormal_duration",
        "likelihood": "censor_aware_proper_lognormal_duration",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    for name, expected in expected_resume_values.items():
        require(resume_identity.get(name) == expected, f"Resume identity drift: {name}")
    require(
        resume_identity.get("train_time_statistics") == dict(train_statistics),
        "Resume train-only statistics drift",
    )
    require(checkpoint.get("resume_identity") == resume_identity, "Selected resume identity drift")
    require(
        Path(summary["selected_checkpoint_path"]).resolve() == selected_path.resolve(),
        "Selected checkpoint path drift",
    )

    last = torch_load_checkpoint(output / "last_epoch_state.pt", map_location="cpu")
    require(
        last.get("checkpoint_type") == "frozen_lognormal_duration_resume",
        "Wrong last-state type",
    )
    require(last.get("checkpoint_schema_version") == 1, "Wrong last-state schema")
    require(last.get("evaluation_scope") == "validation_only", "Last-state scope drift")
    require(last.get("held_out_test_evaluated") is False, "Last-state held-out drift")
    require(last.get("legacy_nll_compared") is False, "Last-state legacy NLL drift")
    require(last.get("resume_identity") == resume_identity, "Last-state identity drift")
    require(last.get("history") == history, "Last-state history drift")
    require(int(last.get("epoch", -1)) == completed_epochs, "Last-state epoch drift")
    require(int(last.get("best_epoch", -1)) == int(selected_row["epoch"]), "Last-state selected epoch drift")
    current_state = last.get("model_state_dict")
    best_state = last.get("best_state_dict")
    require(isinstance(current_state, Mapping), "Last current model state is missing")
    require(isinstance(best_state, Mapping), "Last best model state is missing")
    finite_tensor_tree(current_state, path="last_current_model_state")
    finite_tensor_tree(best_state, path="last_best_model_state")
    require(set(current_state) == set(initial_state), "Last current model key set drift")
    require(set(best_state) == set(selected_state), "Last best model key set drift")
    require(
        canonical_state_dict_sha256(current_state) == last.get("model_state_sha256"),
        "Last current-state digest drift",
    )
    require(
        canonical_state_dict_sha256(best_state) == last.get("best_state_sha256"),
        "Last best-state digest drift",
    )
    for name in selected_state:
        require(
            torch.equal(best_state[name], selected_state[name]),
            f"Last best state differs from selected checkpoint: {name}",
        )
        if name not in TIME_KEYS:
            require(
                torch.equal(current_state[name], source_state[name])
                and torch.equal(best_state[name], source_state[name]),
                f"Last non-time state drift: {name}",
            )
    _audit_optimizer(
        last,
        contract=contract,
        current_state=current_state,
        completed_epochs=completed_epochs,
        train_count=train_cache.count,
    )

    replay_model = build_candidate_from_selected_checkpoint(checkpoint).to("cpu")
    replay_metrics = evaluate_cached_time_metrics(
        model=replay_model,
        cache=validation_cache,
        device="cpu",
        batch_size=int(contract["optimization"]["cached_state_batch_size"]),
        censor_threshold=threshold,
    )
    time_tolerance = float(
        contract["identity_and_stability"][
            "proper_time_nll_replay_absolute_tolerance"
        ]
    )
    _assert_close(
        replay_metrics["proper_time_nll"],
        selected_nll,
        label="selected checkpoint proper Time NLL replay",
        absolute_tolerance=time_tolerance,
    )
    reported_time_metrics = summary.get("validation_time_metrics")
    require(isinstance(reported_time_metrics, Mapping), "Validation time metrics are missing")
    expected_time_metadata = {
        "count": int(row["expected_validation_targets"]),
        "right_censored_count": int(
            row["expected_validation_censored_targets"]
        ),
        "right_censor_threshold": threshold,
        "uncensored_count": int(row["expected_validation_targets"])
        - int(row["expected_validation_censored_targets"]),
        "time_median_observed_target_semantics": (
            "recorded_censor_threshold_is_a_lower_bound_for_"
            "right_censored_rows_not_an_exact_duration"
        ),
        "recorded_target_metric_includes_right_censored_lower_bounds": (
            int(row["expected_validation_censored_targets"]) > 0
        ),
        "time_median_metric_scope": "uncensored_targets_only",
    }
    for name, expected in expected_time_metadata.items():
        require(
            reported_time_metrics.get(name) == expected,
            f"Reported validation time metadata drift: {name}",
        )
    require(
        int(replay_metrics["count"]) == int(row["expected_validation_targets"]),
        "Replayed validation count drift",
    )
    require(
        int(replay_metrics["right_censored_count"])
        == int(row["expected_validation_censored_targets"]),
        "Replayed validation censor count drift",
    )
    require(
        int(replay_metrics["uncensored_count"])
        == int(row["expected_validation_targets"])
        - int(row["expected_validation_censored_targets"]),
        "Replayed validation uncensored count drift",
    )
    for metric in (
        "proper_time_nll",
        "time_median_mae",
        "time_median_rmse",
        "uncensored_time_median_mae",
        "uncensored_time_median_rmse",
        "time_median_mae_against_recorded_target",
        "time_median_rmse_against_recorded_target",
    ):
        _assert_close(
            reported_time_metrics[metric],
            replay_metrics[metric],
            label=f"reported {metric}",
            absolute_tolerance=time_tolerance,
        )
    if replay_metrics["median_below_censor_threshold_rate"] is None:
        require(
            reported_time_metrics.get("median_below_censor_threshold_rate")
            is None,
            "Reported censored-median rate drift",
        )
    else:
        _assert_close(
            reported_time_metrics["median_below_censor_threshold_rate"],
            replay_metrics["median_below_censor_threshold_rate"],
            label="reported median_below_censor_threshold_rate",
            absolute_tolerance=time_tolerance,
        )
    _assert_close(
        summary["time_median_mae"],
        replay_metrics["time_median_mae"],
        label="top-level time median MAE",
        absolute_tolerance=time_tolerance,
    )
    _assert_close(
        summary["time_median_rmse"],
        replay_metrics["time_median_rmse"],
        label="top-level time median RMSE",
        absolute_tolerance=time_tolerance,
    )
    initial_metrics = evaluate_cached_time_metrics(
        model=initial_model.to("cpu"),
        cache=validation_cache,
        device="cpu",
        batch_size=int(contract["optimization"]["cached_state_batch_size"]),
        censor_threshold=threshold,
    )
    _assert_close(
        initial_metrics["proper_time_nll"],
        epoch_zero_nll,
        label="epoch-zero marginal proper Time NLL replay",
        absolute_tolerance=time_tolerance,
    )

    source_replay_model = build_source_model(
        source_payload, max_seq_len=int(row["max_sequence_length"])
    ).to("cpu")
    source_replay_quantity = cached_quantity_predictions(
        model=source_replay_model,
        cache=validation_cache,
        device="cpu",
        batch_size=int(contract["optimization"]["encoder_batch_size"]),
    )
    selected_quantity = cached_quantity_predictions(
        model=replay_model,
        cache=validation_cache,
        device="cpu",
        batch_size=int(contract["optimization"]["encoder_batch_size"]),
    )
    source_quantity = validation_cache.source_quantity_prediction
    target_quantity = validation_cache.target_quantity
    require(source_quantity is not None, "Source quantity predictions are missing")
    require(target_quantity is not None, "Validation quantity targets are missing")
    require(
        torch.equal(selected_quantity, source_replay_quantity),
        "Quantity predictions changed under matched CPU replay",
    )
    quantity_tolerance = contract["identity_and_stability"]
    quantity_abs = float(
        quantity_tolerance["reported_quantity_metric_absolute_tolerance"]
    )
    quantity_rel = float(
        quantity_tolerance["reported_quantity_metric_relative_tolerance"]
    )
    require(
        torch.allclose(
            source_replay_quantity,
            source_quantity,
            rtol=quantity_rel,
            atol=quantity_abs,
        ),
        "Cached CUDA source quantity predictions drift from CPU replay",
    )
    source_quantity_sha = tensor_sha256("quantity_prediction", source_quantity)
    source_replay_quantity_sha = tensor_sha256(
        "quantity_prediction", source_replay_quantity
    )
    selected_replay_quantity_sha = tensor_sha256(
        "quantity_prediction", selected_quantity
    )
    require(
        summary.get("quantity_prediction_bitwise_identical") is True
        and source_quantity_sha
        == summary.get("source_quantity_prediction_sha256")
        == summary.get("selected_quantity_prediction_sha256"),
        "Cached CUDA quantity prediction identity drift",
    )
    require(
        source_replay_quantity_sha == selected_replay_quantity_sha,
        "Matched CPU quantity prediction identity drift",
    )
    replay_quantity_metrics = quantity_metrics(selected_quantity, target_quantity)
    replay_quantity_metrics.update(
        stratified_quantity_metrics(
            selected_quantity,
            target_quantity,
            body_max=float(row["reporting_body_max_train_p95"]),
            tail_min_exclusive=float(row["reporting_tail_min_exclusive_train_p99"]),
            require_nonempty=True,
        )
    )
    reported_quantity_metrics = summary.get("quantity_metrics")
    require(isinstance(reported_quantity_metrics, Mapping), "Quantity metrics are missing")
    for metric in ("mae", "rmse", "body_mae", "gt_p99_mae"):
        _assert_close(
            reported_quantity_metrics[metric],
            replay_quantity_metrics[metric],
            label=f"replayed quantity {metric}",
            absolute_tolerance=quantity_abs,
            relative_tolerance=quantity_rel,
        )
    for metric in ("body_count", "gt_p99_count"):
        require(
            int(reported_quantity_metrics[metric])
            == int(replay_quantity_metrics[metric]),
            f"Quantity stratum count drift: {metric}",
        )

    runtime = summary.get("runtime")
    require(isinstance(runtime, Mapping), "CUDA runtime telemetry is missing")
    require(runtime.get("requested_device") == "cuda", "Run did not request CUDA")
    require(runtime.get("cuda_available") is True, "CUDA was unavailable during run")
    require("RTX 5090" in str(runtime.get("device_name")), "Wrong execution GPU")
    require(
        int(runtime.get("peak_memory_allocated_bytes", 0)) > 0
        and int(runtime.get("peak_memory_reserved_bytes", 0)) > 0,
        "CUDA peak-memory telemetry is empty",
    )
    require(float(runtime.get("elapsed_seconds", 0.0)) > 0.0, "Runtime telemetry is empty")

    forbidden = [
        str(path)
        for path in output.rglob("*")
        if path.is_file()
        and (path.name.startswith("test_") or "held_out" in path.name.lower())
    ]
    require(not forbidden, f"Held-out artifact found: {forbidden}")
    return {
        "status": "passed",
        "phase": phase,
        "dataset": row["dataset"],
        "model_role": model_role,
        "source_backbone": source["backbone"],
        "source_variant": source["quantity_variant"],
        "best_epoch": int(selected_row["epoch"]),
        "completed_epochs": completed_epochs,
        "epoch_zero_validation_proper_time_nll": epoch_zero_nll,
        "selected_validation_proper_time_nll": selected_nll,
        "proper_time_nll_improvement_from_epoch_zero": epoch_zero_nll - selected_nll,
        "strictly_improves_epoch_zero": strict_improvement,
        "quantity_prediction_bitwise_identical": True,
        "matched_cpu_quantity_prediction_sha256": selected_replay_quantity_sha,
        "cached_cuda_quantity_prediction_sha256": source_quantity_sha,
        "cached_cuda_vs_cpu_source_close": True,
        "source_state_sha256": summary["source_state_sha256"],
        "selected_state_sha256": selected_state_sha,
        "selected_time_head_state_sha256": selected_time_sha,
        "changed_state_keys": changed_state_keys,
        "selected_checkpoint_replayed_proper_time_nll": replay_metrics["proper_time_nll"],
        "epoch_zero_replayed_proper_time_nll": initial_metrics["proper_time_nll"],
        "train_target_count": train_cache.count,
        "validation_target_count": validation_cache.count,
        "train_right_censored_count": int(row["expected_train_censored_targets"]),
        "validation_right_censored_count": int(row["expected_validation_censored_targets"]),
        "target_dt_and_censor_identity_verified": True,
        "resume_and_checkpoint_replay_verified": True,
        "runtime": dict(runtime),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }



def build_comparison(
    *,
    full_records: list[Mapping[str, Any]],
    b_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a dataset-local comparison without aggregating unlike NLL units."""

    require(
        [(item.get("dataset"), item.get("model_role")) for item in full_records]
        == list(NEW_EXECUTION_ORDER),
        "Full-fit comparison row order drift",
    )
    new_by_key = {
        (str(item["dataset"]), str(item["model_role"])): dict(item)
        for item in full_records
    }
    b_rows = b_verification.get("rows")
    require(isinstance(b_rows, list), "Verified B comparison rows are missing")
    b_by_dataset = {str(item["dataset"]): dict(item) for item in b_rows}
    require(tuple(b_by_dataset) == DATASETS, "B comparison row order drift")

    dataset_results: list[dict[str, Any]] = []
    preserved: list[bool] = []
    for dataset in DATASETS:
        role_rows: list[dict[str, Any]] = []
        for role in MODEL_ROLES:
            if role == "B":
                record = b_by_dataset[dataset]
                role_rows.append(
                    {
                        "model_role": role,
                        "status": "reused_audited",
                        "selected_validation_proper_time_nll": float(
                            record["selected_validation_proper_time_nll"]
                        ),
                        "epoch_zero_validation_proper_time_nll": float(
                            record["epoch_zero_validation_proper_time_nll"]
                        ),
                        "best_epoch": int(record["best_epoch"]),
                        "quantity_prediction_bitwise_identical": True,
                    }
                )
            elif (dataset, role) == MISSING_ROW:
                role_rows.append(
                    {
                        "model_role": role,
                        "status": "blocked_missing_checkpoint",
                        "selected_validation_proper_time_nll": None,
                        "epoch_zero_validation_proper_time_nll": None,
                        "best_epoch": None,
                    }
                )
            else:
                record = new_by_key[(dataset, role)]
                role_rows.append(
                    {
                        "model_role": role,
                        "status": "audited",
                        "selected_validation_proper_time_nll": float(
                            record["selected_validation_proper_time_nll"]
                        ),
                        "epoch_zero_validation_proper_time_nll": float(
                            record["epoch_zero_validation_proper_time_nll"]
                        ),
                        "best_epoch": int(record["best_epoch"]),
                        "quantity_prediction_bitwise_identical": bool(
                            record["quantity_prediction_bitwise_identical"]
                        ),
                    }
                )
        available = [
            item for item in role_rows
            if item["selected_validation_proper_time_nll"] is not None
        ]
        epoch_zero_values = [
            float(item["epoch_zero_validation_proper_time_nll"])
            for item in available
        ]
        epoch_zero_spread = max(epoch_zero_values) - min(epoch_zero_values)
        require(
            epoch_zero_spread <= 1e-6,
            f"Matched epoch-zero proper Time NLL drift: {dataset}",
        )
        ranking = [
            item["model_role"]
            for item in sorted(
                available,
                key=lambda item: (
                    float(item["selected_validation_proper_time_nll"]),
                    MODEL_ROLES.index(str(item["model_role"])),
                ),
            )
        ]
        a_nll = float(new_by_key[(dataset, "A")]["selected_validation_proper_time_nll"])
        b_nll = float(b_by_dataset[dataset]["selected_validation_proper_time_nll"])
        b_preserved = b_nll <= a_nll + 0.01
        preserved.append(b_preserved)
        dataset_results.append(
            {
                "dataset": dataset,
                "rows": role_rows,
                "available_role_count": len(available),
                "four_way_complete": len(available) == 4,
                "epoch_zero_proper_time_nll_max_absolute_spread": epoch_zero_spread,
                "epoch_zero_proper_time_nll_matched_within_1e_6": True,
                "validation_proper_time_nll_ranking": ranking,
                "B_minus_A_validation_proper_time_nll": b_nll - a_nll,
                "B_at_most_A_plus_0_01": b_preserved,
            }
        )

    present_count = sum(item["available_role_count"] for item in dataset_results)
    require(present_count == 11, f"Expected eleven present rows, got {present_count}")
    return {
        "schema_version": 1,
        "contract_id": "matched_frozen_lognormal_duration_v1",
        "status": "complete_with_known_missing_row",
        "datasets": dataset_results,
        "present_audited_row_count": present_count,
        "required_four_way_row_count": 12,
        "missing_rows": [
            {
                "dataset": MISSING_ROW[0],
                "model_role": MISSING_ROW[1],
                "status": "blocked_missing_checkpoint",
            }
        ],
        "four_way_complete": False,
        "four_way_claim_allowed": False,
        "B_vs_A_preservation_target_met_on_all_datasets": all(preserved),
        "cross_dataset_nll_aggregated": False,
        "legacy_nll_compared": False,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve()
    require(
        project == PROJECT_ROOT.resolve(),
        "Controller must execute from its manifested project",
    )
    input_root = args.input_root.resolve()
    b_result_root = args.b_result_root.resolve()
    output = args.output_root.resolve()
    require(input_root.is_dir(), "Input root is missing")
    require(b_result_root.is_dir(), "B result root is missing")
    require(
        sha256_file(project / CONTRACT_REL) == EXPECTED_CONTRACT_SHA256,
        "Pinned matched duration contract digest drift",
    )
    contract = read_json(project / CONTRACT_REL)
    datasets = validate_contract(contract)
    source_manifest = verify_source(project, args.source_revision)
    input_verification = verify_inputs(input_root, datasets)
    b_verification = verify_b_reuse(
        b_result_root, input_root=input_root, contract=contract, datasets=datasets
    )
    if args.verify_only:
        return {
            "status": "verified",
            "source_revision": args.source_revision,
            "verified_checkpoint_count": input_verification["verified_checkpoint_count"],
            "B_reused_row_count": len(b_verification["rows"]),
            "planned_new_rows": [
                {"dataset": dataset, "model_role": role}
                for dataset, role in NEW_EXECUTION_ORDER
            ],
            "missing_row": input_verification["missing_row"],
            "four_way_complete": False,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }

    output.mkdir(parents=True, exist_ok=True)
    launch_identity = {
        "schema_version": 1,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "source_revision": args.source_revision,
        "source_manifest_sha256": sha256_file(project / "source_manifest.json"),
        "input_root": str(input_root),
        "B_result_root": str(b_result_root),
        "B_decision_sha256": b_verification["decision_sha256"],
        "new_execution_order": [list(item) for item in NEW_EXECUTION_ORDER],
        "missing_row": list(MISSING_ROW),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
    }
    launch_identity_path = output / "control" / "launch_identity.json"
    copied_manifest_path = output / "control" / "source_manifest.json"
    if launch_identity_path.exists():
        require(
            read_json(launch_identity_path) == launch_identity,
            "Existing output launch identity drift",
        )
        require(copied_manifest_path.is_file(), "Existing source manifest is missing")
        require(
            sha256_file(copied_manifest_path)
            == launch_identity["source_manifest_sha256"],
            "Existing source manifest drift",
        )
    else:
        require(not any(output.iterdir()), "Unidentified pre-populated output")
        launch_identity_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(launch_identity_path, launch_identity)
        shutil.copy2(project / "source_manifest.json", copied_manifest_path)

    existing_status_path = output / "status.json"
    if existing_status_path.exists():
        existing_status = read_json(existing_status_path)
        for key, expected in (
            ("source_revision", args.source_revision),
            ("input_root", str(input_root)),
            ("B_result_root", str(b_result_root)),
        ):
            require(
                existing_status.get(key) == expected,
                f"Existing output identity drift: {key}",
            )

    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "source_revision": args.source_revision,
        "source_manifest_sha256": launch_identity["source_manifest_sha256"],
        "source_manifest_file_count": len(source_manifest["files"]),
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "comparison": "matched_frozen_heteroscedastic_lognormal_duration",
        "new_execution_order": [list(item) for item in NEW_EXECUTION_ORDER],
        "input_root": str(input_root),
        "B_result_root": str(b_result_root),
        "input_verification": input_verification,
        "B_reuse_verification": b_verification,
        "completed_e1": [],
        "completed_full": [],
        "completed_new_rows": [],
        "four_way_complete": False,
        "missing_row": input_verification["missing_row"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
    }
    child_slot: dict[str, subprocess.Popen[str] | None] = {"child": None}

    def update(**values: Any) -> None:
        state.update(values)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "status.json", state)

    def stop(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = deterministic_env(project, input_root=input_root)
    with (output / "controller.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            update(
                status="running",
                phase="contract_tests",
                contract_test_device="cuda",
                gpu_preflight=gpu_preflight(),
            )
            xml_path = output / "contract_tests.xml"
            run_logged(
                [
                    args.python,
                    "-s",
                    "-m",
                    "pytest",
                    "-q",
                    str(project / MODEL_TEST_REL),
                    str(project / BASE_RUNNER_TEST_REL),
                    str(project / TEST_REL),
                    str(project / CONTROLLER_TEST_REL),
                    f"--junitxml={xml_path}",
                ],
                output / "contract_tests.log",
                project=project,
                env=environment,
                child_slot=child_slot,
            )
            contract_test_audit = audit_pytest(xml_path)
            contract_test_audit["required_device"] = "cuda"
            update(contract_tests=contract_test_audit)

            for row_index, (dataset, model_role) in enumerate(
                NEW_EXECUTION_ORDER, start=1
            ):
                row = datasets[dataset]
                for phase in ("e1", "full"):
                    verify_source(project, args.source_revision)
                    verify_inputs(input_root, datasets)
                    verify_b_reuse(
                        b_result_root,
                        input_root=input_root,
                        contract=contract,
                        datasets=datasets,
                    )
                    preflight = gpu_preflight()
                    run_output = output / phase / dataset / model_role
                    update(
                        phase=phase,
                        current_dataset=dataset,
                        current_model_role=model_role,
                        current_row_index=row_index,
                        total_new_rows=len(NEW_EXECUTION_ORDER),
                        stage="refit",
                        gpu_preflight=preflight,
                    )
                    command = runner_command(
                        python=args.python,
                        project=project,
                        input_root=input_root,
                        output=run_output,
                        revision=args.source_revision,
                        row=row,
                        model_role=model_role,
                        phase=phase,
                    )
                    run_logged(
                        command,
                        output / phase / dataset / f"{model_role}.log",
                        project=project,
                        env=environment,
                        child_slot=child_slot,
                    )
                    update(stage="audit")
                    audit = audit_run(
                        run_output,
                        contract=contract,
                        row=row,
                        model_role=model_role,
                        phase=phase,
                        source_checkpoint=source_checkpoint_path(
                            input_root, row, model_role
                        ),
                        source_revision=args.source_revision,
                    )
                    completed_key = (
                        "completed_e1" if phase == "e1" else "completed_full"
                    )
                    state[completed_key].append(
                        {
                            "dataset": dataset,
                            "model_role": model_role,
                            "audit": audit,
                        }
                    )
                    update(**{completed_key: state[completed_key]}, stage="complete")
                state["completed_new_rows"].append(
                    {"dataset": dataset, "model_role": model_role}
                )
                update(completed_new_rows=state["completed_new_rows"])

            expected = list(NEW_EXECUTION_ORDER)
            for key in ("completed_e1", "completed_full"):
                observed = [
                    (item["dataset"], item["model_role"])
                    for item in state[key]
                ]
                require(observed == expected, f"{key} completion order drift")
            require(
                [
                    (item["dataset"], item["model_role"])
                    for item in state["completed_new_rows"]
                ]
                == expected,
                "Per-row e1-to-full completion order drift",
            )
            full_records = [item["audit"] for item in state["completed_full"]]
            require(
                all(item["status"] == "passed" for item in full_records),
                "A full-fit audit failed",
            )
            require(
                all(item["quantity_prediction_bitwise_identical"] for item in full_records),
                "A matched head changed quantity predictions",
            )
            comparison = build_comparison(
                full_records=full_records,
                b_verification=b_verification,
            )
            save_json(output / "comparison.json", comparison)
            decision = {
                "schema_version": 1,
                "contract_id": "matched_frozen_lognormal_duration_v1",
                "status": "complete_with_known_missing_row",
                "new_rows_audited": True,
                "new_row_count": 8,
                "B_rows_reused_and_audited": True,
                "B_row_count": 3,
                "present_audited_row_count": 11,
                "required_four_way_row_count": 12,
                "four_way_complete": False,
                "four_way_claim_allowed": False,
                "four_way_decision": "prohibited_missing_intermittent_rmtpp_checkpoint",
                "missing_rows": comparison["missing_rows"],
                "B_vs_A_preservation_target_met_on_all_datasets": comparison[
                    "B_vs_A_preservation_target_met_on_all_datasets"
                ],
                "all_runtime_and_audit_contracts_passed": True,
                "same_duration_head_for_every_present_row": True,
                "same_optimization_for_every_new_fit": True,
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "legacy_nll_compared": False,
                "cross_dataset_nll_aggregated": False,
                "additional_seeds_executed": False,
                "comparison_path": str((output / "comparison.json").resolve()),
            }
            save_json(output / "decision.json", decision)
            update(
                status="complete_with_known_missing_row",
                phase="complete",
                current_dataset=None,
                current_model_role=None,
                stage="complete",
                comparison=comparison,
                decision=decision,
            )
            return state
        except BaseException as error:
            child = child_slot["child"]
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            update(status="failed", stage="failed", error=repr(error))
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--b-result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = execute(parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
