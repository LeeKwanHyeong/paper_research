#!/usr/bin/env python3
"""Build the validation-only A/B decision for aligned K=1 duration heads.

The decision is deliberately fail-closed. A summary can enter the comparison
only when its pinned source, complete target population, cache identities,
resume identity, and selected checkpoint all agree with the experiment
contract and with one another.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from paper.scripts.run_aligned_frozen_lognormal_duration import (
    CONTRACT_ID,
    DATASETS,
    DEFAULT_CONTRACT,
    load_contract_without_duplicate_keys,
    validate_contract,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    CANDIDATE_TIME_HEAD_MODE,
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    SELECTED_CHECKPOINT_NAME,
    SELECTION_RULE,
    TIME_HEAD_PARAMETER_NAMES,
    earliest_strict_minimum,
    first_early_stopping_epoch,
    state_partition_sha256,
)
from paper.scripts.run_hard_lmm_time_head_refit import (
    require,
    save_json,
    sha256_file,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


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


def _finite_number(value: Any, *, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} is invalid",
    )
    return float(value)


def _same_number(
    observed: Any,
    expected: Any,
    *,
    label: str,
    absolute_tolerance: float = 0.0,
) -> None:
    left = _finite_number(observed, label=label)
    right = _finite_number(expected, label=f"expected {label}")
    require(
        math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ),
        f"{label} drift",
    )


def _load_summary(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Aligned summary is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "Aligned summary root must be an object")
    return payload


def _expected_target_population(
    dataset_spec: Mapping[str, Any], *, split: str
) -> dict[str, Any]:
    return {
        "split": split,
        "target_count": int(dataset_spec[f"expected_{split}_targets"]),
        "target_identity_sha256": dataset_spec[
            f"expected_{split}_target_identity_sha256"
        ],
        "target_quantity_sha256": dataset_spec[
            f"expected_{split}_target_quantity_sha256"
        ],
    }


def _expected_observation_population(
    dataset_spec: Mapping[str, Any], *, split: str
) -> dict[str, Any]:
    return {
        "split": split,
        "target_count": int(dataset_spec[f"expected_{split}_targets"]),
        "target_dt_sha256": dataset_spec[
            f"expected_{split}_target_dt_sha256"
        ],
        "right_censor_threshold": dataset_spec["right_censor_threshold"],
        "right_censored_count": int(
            dataset_spec[f"expected_{split}_censored_targets"]
        ),
        "censor_mask_sha256": dataset_spec[
            f"expected_{split}_censor_mask_sha256"
        ],
    }


def _validate_metric_block(
    metrics: Any,
    *,
    dataset_spec: Mapping[str, Any],
    observation_mode: str,
    label: str,
) -> float:
    require(isinstance(metrics, Mapping), f"{label} is missing")
    require(
        metrics.get("observation_likelihood_mode") == observation_mode,
        f"{label} likelihood drift",
    )
    require(
        int(metrics.get("count", -1))
        == int(dataset_spec["expected_validation_targets"]),
        f"{label} population drift",
    )
    require(
        metrics.get("right_censor_threshold")
        == dataset_spec["right_censor_threshold"],
        f"{label} censor threshold drift",
    )
    require(
        int(metrics.get("right_censored_count", -1))
        == int(dataset_spec["expected_validation_censored_targets"]),
        f"{label} censored population drift",
    )
    return _finite_number(
        metrics.get("proper_time_nll"), label=f"{label} proper Time NLL"
    )


def _validate_resume_identity(
    identity: Any,
    *,
    summary: Mapping[str, Any],
    dataset: str,
    role: str,
    dataset_spec: Mapping[str, Any],
    source_spec: Mapping[str, Any],
    observation_contract: Mapping[str, Any],
    contract_sha256: str,
    contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    require(isinstance(identity, Mapping), "Resume identity is missing")
    optimization = contract["optimization"]
    expected_scalars = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "dataset": dataset,
        "model_role": role,
        "source_backbone": source_spec["backbone"],
        "source_variant": source_spec["quantity_variant"],
        "source_checkpoint_sha256": source_spec["checkpoint_file_sha256"],
        "source_state_sha256": source_spec["checkpoint_state_sha256"],
        "source_non_time_state_sha256": source_spec[
            "source_non_time_state_sha256"
        ],
        "right_censor_threshold": dataset_spec["right_censor_threshold"],
        "seed": optimization["seed"],
        "optimizer": optimization["optimizer"],
        "learning_rate": optimization["learning_rate"],
        "weight_decay": optimization["weight_decay"],
        "cached_state_batch_size": optimization["cached_state_batch_size"],
        "quantity_replay_batch_size": optimization["encoder_batch_size"],
        "gradient_clip": optimization["gradient_clip"],
        "minimum_epochs": optimization["minimum_epochs"],
        "early_stopping_patience": optimization[
            "early_stopping_patience"
        ],
        "planned_epochs": optimization["epochs"],
        "selection": SELECTION_RULE,
        "time_head_mode": CANDIDATE_TIME_HEAD_MODE,
        "likelihood": observation_contract["mode"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
        "observation_likelihood_contract": dict(observation_contract),
    }
    for name, expected in expected_scalars.items():
        require(identity.get(name) == expected, f"Resume identity drift: {name}")

    expected_names = list(contract["time_head"]["trainable_parameter_names"])
    require(
        identity.get("trainable_parameter_names") == expected_names,
        "Resume trainable-parameter boundary drift",
    )
    require(
        identity.get("candidate_initial_state_sha256")
        == summary.get("candidate_initial_state_sha256"),
        "Resume initial-state digest drift",
    )
    require(
        _is_git_sha(identity.get("calibration_source_revision")),
        "Resume calibration revision is invalid",
    )
    require(
        identity.get("train_time_statistics")
        == summary.get("train_time_statistics"),
        "Resume train-time statistics drift",
    )

    for split in ("train", "validation"):
        cache = summary.get(f"{split}_cache")
        require(isinstance(cache, Mapping), f"{split} cache identity is missing")
        require(
            identity.get(f"{split}_cache_sha256") == cache.get("sha256"),
            f"Resume {split} cache digest drift",
        )
        require(
            int(identity.get(f"{split}_cache_count", -1))
            == int(cache.get("count", -2)),
            f"Resume {split} cache count drift",
        )
        require(
            identity.get(f"{split}_target_dt_sha256")
            == cache.get("target_dt_sha256"),
            f"Resume {split} duration digest drift",
        )
        require(
            identity.get(f"{split}_censor_mask_sha256")
            == cache.get("censor_mask_sha256"),
            f"Resume {split} censor-mask digest drift",
        )
    return identity


def _validate_selected_checkpoint(
    checkpoint_path: Path,
    *,
    summary: Mapping[str, Any],
    role: str,
    source_spec: Mapping[str, Any],
    observation_contract: Mapping[str, Any],
    expected_resume_identity: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        checkpoint_path.is_file(),
        f"Selected checkpoint is missing: {checkpoint_path}",
    )
    checkpoint_file_sha256 = sha256_file(checkpoint_path)
    require(
        checkpoint_file_sha256
        == summary.get("selected_checkpoint_file_sha256"),
        "Selected checkpoint file digest drift",
    )
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    require(isinstance(checkpoint, dict), "Selected checkpoint root is invalid")
    require(
        checkpoint.get("checkpoint_type")
        == "selected_frozen_lognormal_duration"
        and checkpoint.get("checkpoint_schema_version") == 1,
        "Selected checkpoint type drift",
    )
    require(checkpoint.get("selection") == SELECTION_RULE, "Selector drift")
    require(
        checkpoint.get("selection_formula")
        == observation_contract["selection_formula"],
        "Selected checkpoint likelihood formula drift",
    )
    require(checkpoint.get("model_role") == role, "Checkpoint role drift")
    require(
        checkpoint.get("backbone") == source_spec["backbone"],
        "Checkpoint backbone drift",
    )
    require(
        checkpoint.get("variant") == source_spec["quantity_variant"],
        "Checkpoint quantity variant drift",
    )
    require(
        checkpoint.get("time_head_mode") == CANDIDATE_TIME_HEAD_MODE,
        "Checkpoint time-head mode drift",
    )
    require(
        checkpoint.get("source_state_sha256")
        == source_spec["checkpoint_state_sha256"],
        "Checkpoint source-state digest drift",
    )
    require(
        checkpoint.get("source_non_time_state_sha256")
        == source_spec["source_non_time_state_sha256"],
        "Checkpoint source non-time digest drift",
    )
    require(
        checkpoint.get("candidate_initial_state_sha256")
        == summary.get("candidate_initial_state_sha256"),
        "Checkpoint initial-state digest drift",
    )
    require(
        checkpoint.get("candidate_initial_time_head_state_sha256")
        == summary.get("candidate_initial_time_head_state_sha256"),
        "Checkpoint initial time-head digest drift",
    )
    require(
        checkpoint.get("train_time_statistics")
        == summary.get("train_time_statistics"),
        "Checkpoint train-time statistics drift",
    )
    require(
        checkpoint.get("right_censor_threshold")
        == summary.get("right_censor_threshold"),
        "Checkpoint censor threshold drift",
    )
    lineage = checkpoint.get("source_checkpoint_lineage")
    require(isinstance(lineage, Mapping), "Checkpoint source lineage is missing")
    expected_lineage = {
        "model_role": role,
        "backbone": source_spec["backbone"],
        "checkpoint_file_sha256": source_spec["checkpoint_file_sha256"],
        "checkpoint_state_sha256": source_spec["checkpoint_state_sha256"],
        "checkpoint_selection": source_spec["checkpoint_selection"],
        "training_source_revision": source_spec["training_source_revision"],
        "training_source_revision_history": source_spec[
            "training_source_revision_history"
        ],
        "calibration_source_revision": expected_resume_identity[
            "calibration_source_revision"
        ],
    }
    require(dict(lineage) == expected_lineage, "Checkpoint source lineage drift")
    require(
        checkpoint.get("resume_identity") == dict(expected_resume_identity),
        "Selected checkpoint resume identity drift",
    )
    require(
        checkpoint.get("observation_likelihood_contract")
        == dict(observation_contract),
        "Selected checkpoint observation contract drift",
    )
    require(
        checkpoint.get("evaluation_scope") == "validation_only"
        and checkpoint.get("held_out_test_evaluated") is False
        and checkpoint.get("legacy_nll_compared") is False,
        "Selected checkpoint evaluation scope drift",
    )
    require(
        int(checkpoint.get("best_epoch", -1))
        == int(summary.get("best_epoch", -2)),
        "Selected checkpoint epoch drift",
    )
    _same_number(
        checkpoint.get("selected_metric_value"),
        summary.get("best_validation_proper_time_nll"),
        label="Selected checkpoint primary NLL",
    )

    state = checkpoint.get("model_state_dict")
    require(isinstance(state, Mapping) and bool(state), "Selected state is missing")
    require(
        all(isinstance(value, torch.Tensor) for value in state.values()),
        "Selected state contains a non-tensor entry",
    )
    require(
        all(bool(torch.isfinite(value).all()) for value in state.values()),
        "Selected state contains a non-finite tensor",
    )
    state_dict = dict(state)
    state_sha256 = canonical_state_dict_sha256(state_dict)
    require(
        state_sha256 == checkpoint.get("model_state_sha256")
        == summary.get("selected_state_sha256"),
        "Selected state digest drift",
    )
    non_time_sha256 = state_partition_sha256(state_dict, time_head=False)
    require(
        non_time_sha256 == source_spec["source_non_time_state_sha256"]
        == checkpoint.get("source_non_time_state_sha256")
        == summary.get("selected_non_time_state_sha256"),
        "Selected state changed a pinned non-time tensor",
    )
    time_head_sha256 = state_partition_sha256(state_dict, time_head=True)
    require(
        time_head_sha256 == checkpoint.get("selected_time_head_state_sha256")
        == summary.get("selected_time_head_state_sha256"),
        "Selected time-head state digest drift",
    )

    trainable_names = list(contract["time_head"]["trainable_parameter_names"])
    require(
        all(name in state_dict for name in trainable_names),
        "Selected state is missing a duration-head tensor",
    )
    require(
        sum(int(state_dict[name].numel()) for name in trainable_names)
        == int(contract["time_head"]["expected_trainable_parameter_count"]),
        "Selected state duration-head capacity drift",
    )
    return {
        "file_sha256": checkpoint_file_sha256,
        "state_sha256": state_sha256,
    }


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    dataset: str,
    role: str,
    dataset_spec: Mapping[str, Any],
    source_spec: Mapping[str, Any],
    observation_contract: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    require(summary.get("schema_version") == 1, "Summary schema drift")
    require(summary.get("contract_id") == CONTRACT_ID, "Summary contract drift")
    require(
        summary.get("contract_sha256") == contract_sha256,
        "Summary contract digest drift",
    )
    require(summary.get("status") == "success", "Aligned fit did not succeed")
    require(summary.get("dataset") == dataset, "Summary dataset drift")
    require(summary.get("model_role") == role, "Summary model-role drift")
    require(summary.get("seed") == 42, "Summary seed drift")
    require(
        summary.get("qualified_full_data") is True
        and summary.get("qualified_full_fit") is True,
        "A partial run cannot enter the aligned decision",
    )
    require(
        summary.get("evaluation_scope") == "validation_only"
        and summary.get("held_out_test_evaluated") is False
        and summary.get("legacy_nll_compared") is False,
        "Summary evaluation scope drift",
    )
    require(
        summary.get("source_backbone") == source_spec["backbone"]
        and summary.get("source_variant") == source_spec["quantity_variant"],
        "Summary source architecture drift",
    )
    require(
        summary.get("source_checkpoint_sha256")
        == source_spec["checkpoint_file_sha256"],
        "Summary source checkpoint file digest drift",
    )
    require(
        summary.get("source_state_sha256")
        == source_spec["checkpoint_state_sha256"],
        "Summary source checkpoint state digest drift",
    )
    require(
        summary.get("source_non_time_state_sha256")
        == source_spec["source_non_time_state_sha256"],
        "Summary source non-time digest drift",
    )
    require(
        summary.get("observation_likelihood_contract")
        == dict(observation_contract)
        and summary.get("observation_likelihood_mode")
        == observation_contract["mode"],
        "Observation likelihood contract drift",
    )
    require(
        summary.get("time_head_mode") == CANDIDATE_TIME_HEAD_MODE
        and summary.get("calculation_dtype") == "float64",
        "Summary duration-head implementation drift",
    )
    require(
        summary.get("right_censor_threshold")
        == dataset_spec["right_censor_threshold"],
        "Summary censor threshold drift",
    )
    require(
        int(summary.get("train_right_censored_count", -1))
        == int(dataset_spec["expected_train_censored_targets"])
        and int(summary.get("validation_right_censored_count", -1))
        == int(dataset_spec["expected_validation_censored_targets"]),
        "Summary censored population drift",
    )
    require(
        summary.get("encoder_mode_during_refit") == "eval"
        and summary.get("hidden_state_gradient") == "detached_cache",
        "Frozen representation boundary drift",
    )

    expected_names = list(contract["time_head"]["trainable_parameter_names"])
    require(
        summary.get("trainable_parameter_names") == expected_names,
        "Summary trainable-parameter boundary drift",
    )
    require(
        int(summary.get("trainable_parameter_count", -1))
        == int(contract["time_head"]["expected_trainable_parameter_count"]),
        "Summary trainable-parameter count drift",
    )
    changed_keys = summary.get("changed_state_keys")
    require(
        isinstance(changed_keys, list)
        and len(changed_keys) == len(set(changed_keys))
        and set(changed_keys).issubset(TIME_HEAD_PARAMETER_NAMES),
        "Summary changed-state boundary drift",
    )
    for name in (
        "candidate_initial_state_sha256",
        "candidate_initial_time_head_state_sha256",
        "selected_state_sha256",
        "selected_non_time_state_sha256",
        "selected_time_head_state_sha256",
        "source_quantity_prediction_sha256",
        "selected_quantity_prediction_sha256",
        "selected_checkpoint_file_sha256",
    ):
        require(_is_sha256(summary.get(name)), f"Invalid summary digest: {name}")
    require(
        summary.get("selected_non_time_state_sha256")
        == source_spec["source_non_time_state_sha256"],
        "Summary selected non-time state drift",
    )
    require(
        summary.get("quantity_prediction_bitwise_identical") is True
        and summary.get("source_quantity_prediction_sha256")
        == summary.get("selected_quantity_prediction_sha256"),
        "Quantity predictions changed",
    )

    for split in ("train", "validation"):
        require(
            summary.get(f"{split}_target_population")
            == _expected_target_population(dataset_spec, split=split),
            f"{split} target population drift",
        )
        expected_observation = _expected_observation_population(
            dataset_spec, split=split
        )
        require(
            summary.get(f"{split}_observation_contract")
            == expected_observation,
            f"{split} observation population drift",
        )
        cache = summary.get(f"{split}_cache")
        require(isinstance(cache, Mapping), f"{split} cache identity is missing")
        require(_is_sha256(cache.get("sha256")), f"Invalid {split} cache digest")
        require(
            int(cache.get("count", -1)) == expected_observation["target_count"]
            and cache.get("target_dt_sha256")
            == expected_observation["target_dt_sha256"]
            and cache.get("censor_mask_sha256")
            == expected_observation["censor_mask_sha256"],
            f"{split} cache population drift",
        )

    statistics = summary.get("train_time_statistics")
    require(isinstance(statistics, Mapping), "Train-time statistics are missing")
    require(
        statistics.get("statistics_source_split") == "train"
        and int(statistics.get("target_count", -1))
        == int(dataset_spec["expected_train_targets"]),
        "Train-time statistics population drift",
    )
    require(
        all(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or math.isfinite(float(value))
            for value in statistics.values()
        ),
        "Train-time statistics contain a non-finite value",
    )
    for observed_name, contract_name in (
        ("time_scale", "train_time_scale"),
        ("target_log_scaled_mean", "train_log_scaled_mean"),
        ("target_log_scaled_std", "train_log_scaled_std"),
    ):
        _same_number(
            statistics.get(observed_name),
            dataset_spec[contract_name],
            label=f"Train-time statistic {observed_name}",
        )

    history = summary.get("history")
    require(isinstance(history, list) and bool(history), "Fit history is missing")
    require(
        [int(row.get("epoch", -1)) for row in history]
        == list(range(len(history))),
        "Fit history epochs are not contiguous",
    )
    require(
        int(summary.get("completed_epochs", -1)) == len(history) - 1,
        "Summary completed epoch drift",
    )
    completed_epochs = int(summary["completed_epochs"])
    require(
        completed_epochs <= int(contract["optimization"]["epochs"]),
        "Summary exceeded the fixed epoch budget",
    )
    selected = earliest_strict_minimum(history)
    require(
        int(summary.get("best_epoch", -1)) == int(selected["epoch"]),
        "Summary earliest-minimum selector drift",
    )
    first_stop = first_early_stopping_epoch(
        history,
        min_epochs=int(contract["optimization"]["minimum_epochs"]),
        patience=int(contract["optimization"]["early_stopping_patience"]),
    )
    stopped_early = first_stop is not None
    require(
        summary.get("stopped_early") is stopped_early,
        "Summary early-stopping audit drift",
    )
    if first_stop is None:
        require(
            completed_epochs == int(contract["optimization"]["epochs"]),
            "Successful fit ended before its epoch budget",
        )
    else:
        require(
            completed_epochs == first_stop,
            "Fit history continued after early stopping",
        )
    _same_number(
        summary.get("best_validation_proper_time_nll"),
        selected["val_proper_time_nll"],
        label="Summary selected primary NLL",
    )
    primary_nll = _validate_metric_block(
        summary.get("validation_time_metrics"),
        dataset_spec=dataset_spec,
        observation_mode=str(observation_contract["mode"]),
        label="Primary validation metrics",
    )
    _same_number(
        primary_nll,
        summary.get("best_validation_proper_time_nll"),
        label="Primary validation metric replay",
        absolute_tolerance=1e-12,
    )
    continuous_nll = _validate_metric_block(
        summary.get("validation_continuous_reference_metrics"),
        dataset_spec=dataset_spec,
        observation_mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
        label="Continuous reference validation metrics",
    )

    resume_identity = _validate_resume_identity(
        summary.get("resume_identity"),
        summary=summary,
        dataset=dataset,
        role=role,
        dataset_spec=dataset_spec,
        source_spec=source_spec,
        observation_contract=observation_contract,
        contract_sha256=contract_sha256,
        contract=contract,
    )
    checkpoint_path = summary_path.parent / SELECTED_CHECKPOINT_NAME
    checkpoint_identity = _validate_selected_checkpoint(
        checkpoint_path,
        summary=summary,
        role=role,
        source_spec=source_spec,
        observation_contract=observation_contract,
        expected_resume_identity=resume_identity,
        contract=contract,
    )
    return {
        "primary_nll": primary_nll,
        "continuous_nll": continuous_nll,
        "summary_file_sha256": sha256_file(summary_path),
        "selected_checkpoint_file_sha256": checkpoint_identity["file_sha256"],
        "selected_checkpoint_state_sha256": checkpoint_identity["state_sha256"],
        "calibration_source_revision": resume_identity[
            "calibration_source_revision"
        ],
    }


def _validate_prior_B_reference(
    summary: Mapping[str, Any],
    *,
    dataset_spec: Mapping[str, Any],
    observation_contract: Mapping[str, Any],
    replay_absolute_tolerance: float,
) -> tuple[float, float]:
    prior = summary.get("prior_B_reference")
    require(isinstance(prior, Mapping), "Prior B replay is missing")
    reference_spec = dataset_spec["prior_B_reference"]
    require(
        reference_spec["source_non_time_state_sha256"]
        == dataset_spec["sources"]["B"]["source_non_time_state_sha256"],
        "Prior B source non-time contract drift",
    )
    require(
        prior.get("checkpoint_file_sha256")
        == reference_spec["checkpoint_file_sha256"]
        and prior.get("checkpoint_state_sha256")
        == reference_spec["checkpoint_state_sha256"],
        "Prior B checkpoint identity drift",
    )
    require(
        prior.get("evaluation_scope") == "validation_only"
        and prior.get("held_out_test_evaluated") is False,
        "Prior B replay scope drift",
    )
    primary_nll = _validate_metric_block(
        prior.get("primary_observation_metrics"),
        dataset_spec=dataset_spec,
        observation_mode=str(observation_contract["mode"]),
        label="Prior B primary replay",
    )
    continuous_nll = _validate_metric_block(
        prior.get("continuous_reference_metrics"),
        dataset_spec=dataset_spec,
        observation_mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
        label="Prior B continuous replay",
    )
    _same_number(
        continuous_nll,
        reference_spec["validation_continuous_nll"],
        label="Prior B pinned continuous NLL",
        absolute_tolerance=replay_absolute_tolerance,
    )
    return primary_nll, continuous_nll


def build_decision(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    results_root: Path,
) -> dict[str, Any]:
    """Evaluate the fixed validation gates without opening held-out data."""
    require(_is_sha256(contract_sha256), "Actual contract digest is invalid")
    datasets = validate_contract(contract)
    integer_improvement = float(
        contract["acceptance"][
            "minimum_aligned_B_improvement_over_prior_B_primary_nll"
        ]
    )
    continuous_tolerance = float(
        contract["acceptance"][
            "maximum_continuous_objective_replay_worsening"
        ]
    )
    A_margin = float(
        contract["acceptance"][
            "maximum_aligned_B_minus_aligned_A_primary_nll"
        ]
    )
    prior_replay_tolerance = float(
        contract["comparison_policy"][
            "maximum_prior_B_continuous_replay_absolute_drift"
        ]
    )
    rows: list[dict[str, Any]] = []
    all_pass = True
    calibration_revisions: set[str] = set()
    for dataset in DATASETS:
        dataset_spec = datasets[dataset]
        observation_contract = dataset_spec["observation_contract"]
        summaries: dict[str, dict[str, Any]] = {}
        evidence: dict[str, dict[str, Any]] = {}
        for role in ("A", "B"):
            summary_path = results_root / dataset / role / "summary.json"
            summary = _load_summary(summary_path)
            summaries[role] = summary
            evidence[role] = _validate_summary(
                summary,
                summary_path=summary_path,
                dataset=dataset,
                role=role,
                dataset_spec=dataset_spec,
                source_spec=dataset_spec["sources"][role],
                observation_contract=observation_contract,
                contract=contract,
                contract_sha256=contract_sha256,
            )
            calibration_revisions.add(
                str(evidence[role]["calibration_source_revision"])
            )
        require(
            summaries["A"]["candidate_initial_time_head_state_sha256"]
            == summaries["B"]["candidate_initial_time_head_state_sha256"],
            "A/B initial duration heads differ",
        )
        require(
            summaries["A"]["train_time_statistics"]
            == summaries["B"]["train_time_statistics"],
            "A/B train-only initialization statistics differ",
        )

        A_primary = evidence["A"]["primary_nll"]
        B_primary = evidence["B"]["primary_nll"]
        B_continuous = evidence["B"]["continuous_nll"]
        prior_primary, prior_continuous = _validate_prior_B_reference(
            summaries["B"],
            dataset_spec=dataset_spec,
            observation_contract=observation_contract,
            replay_absolute_tolerance=prior_replay_tolerance,
        )
        if observation_contract["mode"] == OBSERVATION_LIKELIHOOD_CONTINUOUS:
            _same_number(
                evidence["A"]["primary_nll"],
                evidence["A"]["continuous_nll"],
                label="Aligned A continuous primary/reference NLL",
                absolute_tolerance=1e-12,
            )
            _same_number(
                B_primary,
                B_continuous,
                label="Aligned B continuous primary/reference NLL",
                absolute_tolerance=1e-12,
            )
            _same_number(
                prior_primary,
                prior_continuous,
                label="Prior B continuous primary/reference NLL",
                absolute_tolerance=1e-12,
            )
        improvement = prior_primary - B_primary
        B_minus_A = B_primary - A_primary
        if (
            observation_contract["mode"]
            == OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER
        ):
            primary_gate_name = "aligned_B_improves_prior_B_by_0_005"
            minimum_improvement = integer_improvement
        else:
            primary_gate_name = (
                "aligned_B_replays_prior_continuous_objective_within_1e_6"
            )
            minimum_improvement = -continuous_tolerance
        gates = {
            primary_gate_name: improvement >= minimum_improvement,
            "aligned_B_within_aligned_A_plus_0_01": B_minus_A <= A_margin,
            "A_quantity_bitwise_identity": True,
            "B_quantity_bitwise_identity": True,
        }
        within_B = summaries["B"].get("within_B_gates")
        require(isinstance(within_B, Mapping), "Runner B gate audit is missing")
        expected_requirement = (
            "improve_by_at_least_0.005"
            if observation_contract["mode"]
            == OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER
            else "replay_nonworse_within_1e-6"
        )
        require(
            within_B.get("primary_requirement") == expected_requirement,
            "Runner B primary requirement drift",
        )
        _same_number(
            within_B.get("primary_nll_improvement"),
            improvement,
            label="Runner B primary improvement",
        )
        _same_number(
            within_B.get("minimum_accepted_primary_nll_improvement"),
            minimum_improvement,
            label="Runner B accepted improvement",
        )
        require(
            within_B.get("passes_primary_requirement")
            is gates[primary_gate_name],
            "Runner B primary gate drift",
        )
        _same_number(
            within_B.get("continuous_nll_delta"),
            B_continuous - prior_continuous,
            label="Runner B continuous report delta",
        )
        require(
            within_B.get("continuous_nll_is_report_only") is True,
            "Continuous reference metric entered the runner gate",
        )

        passed = all(gates.values())
        all_pass = all_pass and passed
        rows.append(
            {
                "dataset": dataset,
                "observation_likelihood_mode": observation_contract["mode"],
                "aligned_A_primary_nll": A_primary,
                "aligned_B_primary_nll": B_primary,
                "prior_B_primary_nll_same_likelihood": prior_primary,
                "aligned_B_improvement_over_prior_B": improvement,
                "minimum_applicable_improvement": minimum_improvement,
                "aligned_B_minus_aligned_A": B_minus_A,
                "aligned_B_continuous_nll_report_only": B_continuous,
                "prior_B_continuous_nll_report_only": prior_continuous,
                "evidence": evidence,
                "gates": gates,
                "passed": passed,
            }
        )
    require(
        len(calibration_revisions) == 1,
        "Aligned rows used different calibration source revisions",
    )
    calibration_source_revision = next(iter(calibration_revisions))
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "calibration_source_revision": calibration_source_revision,
        "status": "complete",
        "evidence_integrity": "verified",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "datasets": rows,
        "all_datasets_pass": all_pass,
        "decision": (
            "adopt_aligned_B_without_adapter"
            if all_pass
            else "aligned_head_insufficient_continue_to_adapter_gate"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract_without_duplicate_keys(args.contract)
    contract_sha256 = sha256_file(args.contract)
    decision = build_decision(
        contract=contract,
        contract_sha256=contract_sha256,
        results_root=args.results_root,
    )
    save_json(args.output, decision)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
