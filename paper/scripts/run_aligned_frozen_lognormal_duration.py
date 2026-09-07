#!/usr/bin/env python3
"""Fit observation-aligned K=1 duration heads on frozen A or B states.

The source encoder, Hard-LMM memory, and quantity path are immutable. Only
the four 130-parameter heteroscedastic log-normal duration-head tensors are
optimized. Intermittent uses the continuous observation density requested by
the contract; Taxi and Instacart use a normalized positive-integer observation
law, with Instacart code 30 represented by survival from 29.5.

Only train and validation rows are admitted. Held-out rows are removed before
eager materialization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.run_count_aware_tpp_backbone_control import (
    derive_train_time_contract,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    SELECTED_CHECKPOINT_NAME,
    build_candidate_from_selected_checkpoint,
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
    load_or_extract_feature_cache,
    require,
    save_json,
    sha256_file,
    stratified_quantity_metrics,
    validate_target_population,
)
from paper.scripts.run_matched_frozen_lognormal_duration import (
    build_source_model,
    validate_source_checkpoint,
)
from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "aligned_frozen_lognormal_duration_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "paper/contracts/aligned_frozen_lognormal_duration_v1.json"
)
MODEL_ROLES = ("A", "B")
DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)
SUMMARY_NAME = "summary.json"


def load_contract_without_duplicate_keys(path: Path) -> dict[str, Any]:
    """Load JSON while rejecting duplicate keys before validation."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(name for name, _ in pairs)
        duplicates = sorted(
            name for name, count in counts.items() if count > 1
        )
        require(not duplicates, f"Duplicate contract keys: {duplicates}")
        return dict(pairs)

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    require(isinstance(payload, dict), "Contract root must be an object")
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


def validate_contract(
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Fail closed on experiment, observation, and comparison drift."""
    require(contract.get("schema_version") == 1, "Contract schema drift")
    require(contract.get("contract_id") == CONTRACT_ID, "Contract ID drift")
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope is missing")
    require(tuple(scope.get("datasets", [])) == DATASETS, "Dataset scope drift")
    require(
        tuple(scope.get("model_roles", [])) == MODEL_ROLES,
        "Model-role scope drift",
    )
    require(
        scope.get("input_splits") == ["train", "validation"],
        "Input split scope drift",
    )
    require(scope.get("held_out_test") is False, "Held-out test admitted")
    require(
        scope.get("evaluation_scope") == "validation_only",
        "Evaluation scope drift",
    )
    require(scope.get("seed") == 42, "Seed drift")
    require(scope.get("additional_seeds") is False, "Additional seeds admitted")
    require(
        scope.get("quantity_predictions_may_change") is False,
        "Quantity changes were admitted",
    )

    time_head = contract.get("time_head")
    require(isinstance(time_head, Mapping), "Time-head contract is missing")
    require(
        time_head.get("expected_trainable_parameter_count") == 130,
        "Time-head parameter count drift",
    )
    require(
        time_head.get("calculation_dtype") == "float64",
        "Time-head calculation dtype drift",
    )
    require(time_head.get("duration_clamp") is False, "Duration clamp admitted")

    optimization = contract.get("optimization")
    require(isinstance(optimization, Mapping), "Optimization contract is missing")
    expected_optimization = {
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
        "scheduler": None,
        "dataset_specific_hyperparameters": False,
        "shared_across_all_datasets": True,
    }
    for name, expected in expected_optimization.items():
        require(
            optimization.get(name) == expected,
            f"Optimization drift: {name}",
        )

    selection = contract.get("checkpoint_selection")
    require(isinstance(selection, Mapping), "Checkpoint selection is missing")
    require(
        selection.get("rule") == "earliest strict finite minimum",
        "Selector rule drift",
    )
    require(selection.get("fallback") == "epoch 0", "Epoch-zero fallback drift")
    require(
        selection.get("other_metrics_in_selector") is False,
        "A guardrail entered checkpoint selection",
    )

    comparison = contract.get("comparison_policy")
    require(isinstance(comparison, Mapping), "Comparison policy is missing")
    require(
        comparison.get("same_time_head_architecture_for_A_and_B") is True,
        "A/B head parity drift",
    )
    require(
        comparison.get("same_initialization_optimizer_batch_order_and_epoch_budget")
        is True,
        "A/B fitting parity drift",
    )
    require(
        comparison.get("same_calibration_source_revision_for_all_rows")
        is True,
        "Calibration implementation parity drift",
    )
    require(
        comparison.get("maximum_prior_B_continuous_replay_absolute_drift")
        == 1e-6,
        "Prior B replay tolerance drift",
    )
    require(
        comparison.get("legacy_time_nll_comparison") is False,
        "Legacy Time NLL comparison admitted",
    )
    require(
        comparison.get("cross_dataset_nll_aggregation") is False,
        "Cross-dataset NLL aggregation admitted",
    )

    acceptance = contract.get("acceptance")
    require(isinstance(acceptance, Mapping), "Acceptance contract is missing")
    require(
        acceptance.get(
            "minimum_aligned_B_improvement_over_prior_B_primary_nll"
        )
        == 0.005,
        "B improvement margin drift",
    )
    require(
        acceptance.get("minimum_aligned_B_improvement_scope")
        == "datasets using positive_integer_round_clamp_lognormal only",
        "B improvement scope drift",
    )
    require(
        acceptance.get("maximum_continuous_objective_replay_worsening")
        == 1e-6,
        "Continuous-objective replay tolerance drift",
    )
    require(
        acceptance.get("maximum_aligned_B_minus_aligned_A_primary_nll")
        == 0.01,
        "A preservation margin drift",
    )
    require(acceptance.get("held_out_claim") is False, "Held-out claim admitted")

    rows = contract.get("datasets")
    require(
        isinstance(rows, list) and len(rows) == len(DATASETS),
        "Dataset contracts are incomplete",
    )
    datasets = {str(row.get("dataset")): dict(row) for row in rows}
    require(tuple(datasets) == DATASETS, "Dataset contract order drift")
    for dataset, row in datasets.items():
        for name in (
            "data_sha256",
            "split_manifest_sha256",
            "expected_train_target_identity_sha256",
            "expected_train_target_quantity_sha256",
            "expected_validation_target_identity_sha256",
            "expected_validation_target_quantity_sha256",
            "expected_train_target_dt_sha256",
            "expected_validation_target_dt_sha256",
            "expected_train_censor_mask_sha256",
            "expected_validation_censor_mask_sha256",
        ):
            require(_is_sha256(row.get(name)), f"Invalid dataset digest: {name}")
        require(
            int(row.get("expected_train_targets", 0)) > 0
            and int(row.get("expected_validation_targets", 0)) > 0,
            "Target population is empty",
        )
        observation = validate_observation_likelihood_contract(
            row.get("observation_contract", {})
        )
        if dataset == "intermittent_frozen_5000":
            require(
                observation["mode"] == OBSERVATION_LIKELIHOOD_CONTINUOUS,
                "Intermittent observation mode drift",
            )
        else:
            require(
                observation["mode"]
                == "positive_integer_round_clamp_lognormal",
                "Integer dataset observation mode drift",
            )
        if dataset == "insta_market_basket":
            require(
                row.get("right_censor_threshold") == 30.0
                and observation.get("top_code") == "P(T > 29.5)",
                "Instacart top-code contract drift",
            )
        else:
            require(
                row.get("right_censor_threshold") is None
                and observation.get("top_code") is None,
                "Unexpected censoring contract",
            )

        sources = row.get("sources")
        require(isinstance(sources, Mapping), "Source map is missing")
        require(tuple(sources) == MODEL_ROLES, "A/B source order drift")
        for role in MODEL_ROLES:
            source = sources[role]
            require(source.get("available") is True, "Source is unavailable")
            require(source.get("execution") == "fit_new", "Source execution drift")
            require(source.get("backbone") == "titantpp", "Backbone drift")
            require(source.get("seed") == 42, "Source seed drift")
            require(
                source.get("checkpoint_selection")
                == contract["model_roles"][role]["source_checkpoint_selection"],
                "Source selector drift",
            )
            require(
                _is_sha256(source.get("checkpoint_file_sha256"))
                and _is_sha256(source.get("checkpoint_state_sha256"))
                and _is_sha256(source.get("source_non_time_state_sha256")),
                "Source checkpoint digest is invalid",
            )
            require(
                _is_git_sha(source.get("training_source_revision")),
                "Source revision is invalid",
            )
            require(
                source.get("training_source_revision_history")
                == [source.get("training_source_revision")],
                "Source revision history drift",
            )
            require(
                source.get("evaluation_scope") == "validation_only"
                and source.get("held_out_test_evaluated") is False,
                "Source evaluation scope drift",
            )
        reference = row.get("prior_B_reference")
        require(isinstance(reference, Mapping), "Prior B reference is missing")
        for name in (
            "checkpoint_file_sha256",
            "checkpoint_state_sha256",
            "source_non_time_state_sha256",
        ):
            require(_is_sha256(reference.get(name)), f"Prior B digest invalid: {name}")
        require(
            reference.get("primary_nll_replay_required") is True,
            "Prior B replay was disabled",
        )
    expected_execution = [
        (dataset, role)
        for dataset in DATASETS
        for role in MODEL_ROLES
    ]
    observed_execution = [
        (row.get("dataset"), row.get("model_role"))
        for row in contract.get("execution_order", [])
    ]
    require(observed_execution == expected_execution, "Execution order drift")
    return datasets


def dataset_by_id(
    contract: Mapping[str, Any], dataset: str
) -> dict[str, Any]:
    datasets = validate_contract(contract)
    require(dataset in datasets, "Dataset is outside the aligned contract")
    return datasets[dataset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model-role", choices=MODEL_ROLES, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior-b-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-source-revision", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_prior_B_reference(
    *,
    checkpoint_path: Path,
    reference_spec: Mapping[str, Any],
    source_non_time_state_sha256: str,
    validation_cache: FrozenFeatureCache,
    device: torch.device,
    batch_size: int,
    censor_threshold: float | None,
    observation_likelihood_mode: str,
) -> dict[str, Any]:
    """Replay continuous-selected B under the aligned likelihood."""
    require(checkpoint_path.is_file(), "Prior B reference checkpoint is missing")
    require(
        sha256_file(checkpoint_path)
        == reference_spec["checkpoint_file_sha256"],
        "Prior B checkpoint file digest drift",
    )
    payload = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    require(
        canonical_state_dict_sha256(payload["model_state_dict"])
        == reference_spec["checkpoint_state_sha256"],
        "Prior B checkpoint state digest drift",
    )
    require(payload.get("model_role", "B") == "B", "Prior reference is not B")
    require(
        state_partition_sha256(payload["model_state_dict"], time_head=False)
        == source_non_time_state_sha256
        == reference_spec["source_non_time_state_sha256"],
        "Prior B non-time state differs from source B",
    )
    model = build_candidate_from_selected_checkpoint(payload).to(device)
    primary = evaluate_cached_time_metrics(
        model=model,
        cache=validation_cache,
        device=device,
        batch_size=batch_size,
        censor_threshold=censor_threshold,
        observation_likelihood_mode=observation_likelihood_mode,
    )
    continuous = evaluate_cached_time_metrics(
        model=model,
        cache=validation_cache,
        device=device,
        batch_size=batch_size,
        censor_threshold=censor_threshold,
        observation_likelihood_mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
    )
    return {
        "checkpoint_file_sha256": reference_spec["checkpoint_file_sha256"],
        "checkpoint_state_sha256": reference_spec["checkpoint_state_sha256"],
        "primary_observation_metrics": primary,
        "continuous_reference_metrics": continuous,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but unavailable")
        torch.cuda.reset_peak_memory_stats(device)
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)
    require(
        _is_git_sha(args.calibration_source_revision),
        "Calibration revision must be a full Git SHA",
    )

    contract = load_contract_without_duplicate_keys(args.contract)
    dataset_spec = dataset_by_id(contract, args.dataset)
    source_spec = dataset_spec["sources"][args.model_role]
    observation_contract = validate_observation_likelihood_contract(
        dataset_spec["observation_contract"]
    )
    observation_mode = str(observation_contract["mode"])
    contract_sha = sha256_file(args.contract)
    if args.model_role == "B":
        require(
            args.prior_b_checkpoint is not None
            and args.prior_b_checkpoint.is_file(),
            "B alignment requires the prior B reference checkpoint",
        )
        require(
            sha256_file(args.prior_b_checkpoint)
            == dataset_spec["prior_B_reference"][
                "checkpoint_file_sha256"
            ],
            "Prior B checkpoint file digest drift",
        )
    else:
        require(
            args.prior_b_checkpoint is None,
            "A alignment must not receive a prior B checkpoint",
        )
    require(
        sha256_file(args.data) == dataset_spec["data_sha256"],
        "Dataset digest mismatch",
    )
    require(
        sha256_file(args.split_manifest)
        == dataset_spec["split_manifest_sha256"],
        "Split-manifest digest mismatch",
    )
    checkpoint_sha = sha256_file(args.checkpoint)
    require(
        checkpoint_sha == source_spec["checkpoint_file_sha256"],
        "Source checkpoint file digest mismatch",
    )
    source_payload = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    validate_source_checkpoint(source_payload, source_spec=source_spec)
    source_state_sha = str(source_payload["model_state_sha256"])
    source_non_time_sha = state_partition_sha256(
        source_payload["model_state_dict"], time_head=False
    )
    require(
        source_non_time_sha == source_spec["source_non_time_state_sha256"],
        "Source non-time state digest mismatch",
    )
    source_model = build_source_model(
        source_payload,
        max_seq_len=int(dataset_spec["max_sequence_length"]),
    ).to(device)

    frame = prepare_count_frame(load_admitted_frame(args.data))
    require(
        set(frame["chronological_split"].unique().to_list())
        <= {"train", "validation"},
        "Held-out rows entered the frame",
    )
    lookback = int(dataset_spec["lookback"])
    max_seq_len = int(dataset_spec["max_sequence_length"])
    train_population = exact_target_population_contract(
        frame,
        target_split="train",
        lookback=lookback,
        max_seq_len=max_seq_len,
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
            frame,
            lookback_weeks=lookback,
            max_seq_len=max_seq_len,
        ),
    )
    censor_threshold = resolve_censor_threshold(args.dataset, dataset_spec)

    settings = dict(contract["optimization"])
    partial_limits = (
        args.max_train_batches,
        args.max_validation_batches,
        args.max_epochs,
    )
    if any(value is not None for value in partial_limits):
        require(
            args.allow_partial_contract,
            "Batch or epoch limits require --allow-partial-contract",
        )
    planned_epochs = int(settings["epochs"])
    if args.max_epochs is not None:
        require(args.max_epochs >= 0, "max epochs must be nonnegative")
        planned_epochs = min(planned_epochs, args.max_epochs)
    encoder_batch_size = int(settings["encoder_batch_size"])
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
    cache_base = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha,
        "dataset": args.dataset,
        "model_role": args.model_role,
        "source_backbone": source_spec["backbone"],
        "source_variant": source_spec["quantity_variant"],
        "data_sha256": dataset_spec["data_sha256"],
        "split_manifest_sha256": dataset_spec["split_manifest_sha256"],
        "source_checkpoint_sha256": checkpoint_sha,
        "source_state_sha256": source_state_sha,
        "encoder_mode": "eval",
        "target_quantity_masked": True,
        "memory_target_write_masked": True,
        "evaluation_scope": "train_and_validation_only",
        "held_out_test_evaluated": False,
    }
    cache_dir = args.feature_cache_dir or (args.output_dir / "cache")
    if args.feature_cache_dir is not None:
        require(
            (cache_dir / "train_features.pt").is_file()
            and (cache_dir / "validation_features.pt").is_file(),
            "Requested feature cache is incomplete",
        )
    train_cache = load_or_extract_feature_cache(
        path=cache_dir / "train_features.pt",
        identity={
            **cache_base,
            "split": "train",
            "target_count": train_population["target_count"],
            "target_identity_sha256": train_population[
                "target_identity_sha256"
            ],
            "target_quantity_sha256": train_population[
                "target_quantity_sha256"
            ],
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
    full_data = (
        args.max_train_batches is None
        and args.max_validation_batches is None
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
            source_variant=str(source_spec["quantity_variant"]),
            max_seq_len=max_seq_len,
            training_stage="observation_aligned_frozen_duration_refit",
        )
    )
    require(
        candidate_metadata["source_non_time_state_sha256"]
        == source_non_time_sha,
        "Candidate non-time copy drift",
    )
    del source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    candidate = candidate.to(device)
    source_lineage = {
        "model_role": args.model_role,
        "backbone": source_spec["backbone"],
        "checkpoint_file_sha256": checkpoint_sha,
        "checkpoint_state_sha256": source_state_sha,
        "checkpoint_selection": source_spec["checkpoint_selection"],
        "training_source_revision": source_payload.get("source_revision"),
        "training_source_revision_history": source_payload.get(
            "source_revision_history"
        ),
        "calibration_source_revision": args.calibration_source_revision,
    }
    summary = fit_time_head_from_cache(
        model=candidate,
        train_cache=train_cache,
        validation_cache=validation_cache,
        output_dir=args.output_dir,
        dataset=args.dataset,
        censor_threshold=censor_threshold,
        contract_sha256=contract_sha,
        source_checkpoint_sha256=checkpoint_sha,
        source_state_sha256=source_state_sha,
        source_non_time_state_sha256=source_non_time_sha,
        candidate_initial_state_sha256=candidate_metadata[
            "candidate_initial_state_sha256"
        ],
        train_time_statistics=train_statistics,
        calibration_source_revision=args.calibration_source_revision,
        seed=int(settings["seed"]),
        planned_epochs=planned_epochs,
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        batch_size=int(settings["cached_state_batch_size"]),
        quantity_replay_batch_size=encoder_batch_size,
        grad_clip=float(settings["gradient_clip"]),
        min_epochs=int(settings["minimum_epochs"]),
        patience=int(settings["early_stopping_patience"]),
        device=device,
        candidate_metadata=candidate_metadata,
        source_metadata=source_lineage,
        contract_id=CONTRACT_ID,
        model_role=args.model_role,
        source_backbone=str(source_spec["backbone"]),
        source_variant=str(source_spec["quantity_variant"]),
        observation_likelihood_contract=observation_contract,
    )
    if summary.get("status") != "success":
        return summary

    selected_payload = torch_load_checkpoint(
        args.output_dir / SELECTED_CHECKPOINT_NAME,
        map_location="cpu",
    )
    candidate.load_state_dict(selected_payload["model_state_dict"], strict=True)
    candidate.eval()
    continuous_metrics = evaluate_cached_time_metrics(
        model=candidate,
        cache=validation_cache,
        device=device,
        batch_size=int(settings["cached_state_batch_size"]),
        censor_threshold=censor_threshold,
        observation_likelihood_mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
    )
    summary = dict(summary)
    summary.update(
        {
            "encoder_config": candidate_encoder,
            "interface_meta": candidate_metadata["interface_meta"],
            "train_target_population": train_population,
            "validation_target_population": validation_population,
            "train_observation_contract": train_observation,
            "validation_observation_contract": validation_observation,
            "qualified_full_data": full_data,
            "qualified_full_fit": qualified_full_fit,
            "contract_sha256": contract_sha,
            "selected_checkpoint_file_sha256": sha256_file(
                args.output_dir / SELECTED_CHECKPOINT_NAME
            ),
            "feature_cache_dir": str(cache_dir.resolve()),
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
    if args.model_role == "B":
        assert args.prior_b_checkpoint is not None
        prior = evaluate_prior_B_reference(
            checkpoint_path=args.prior_b_checkpoint,
            reference_spec=dataset_spec["prior_B_reference"],
            source_non_time_state_sha256=source_non_time_sha,
            validation_cache=validation_cache,
            device=device,
            batch_size=int(settings["cached_state_batch_size"]),
            censor_threshold=censor_threshold,
            observation_likelihood_mode=observation_mode,
        )
        candidate_primary = float(summary["best_validation_proper_time_nll"])
        prior_primary = float(
            prior["primary_observation_metrics"]["proper_time_nll"]
        )
        candidate_continuous = float(continuous_metrics["proper_time_nll"])
        prior_continuous = float(
            prior["continuous_reference_metrics"]["proper_time_nll"]
        )
        improvement = prior_primary - candidate_primary
        if observation_mode == OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER:
            primary_requirement = "improve_by_at_least_0.005"
            required_improvement = float(
                contract["acceptance"][
                    "minimum_aligned_B_improvement_over_prior_B_primary_nll"
                ]
            )
            passes_primary = improvement >= required_improvement
        else:
            primary_requirement = "replay_nonworse_within_1e-6"
            allowed_worsening = float(
                contract["acceptance"][
                    "maximum_continuous_objective_replay_worsening"
                ]
            )
            required_improvement = -allowed_worsening
            passes_primary = improvement >= required_improvement
        summary["prior_B_reference"] = prior
        summary["within_B_gates"] = {
            "primary_requirement": primary_requirement,
            "primary_nll_improvement": improvement,
            "minimum_accepted_primary_nll_improvement": required_improvement,
            "passes_primary_requirement": passes_primary,
            "continuous_nll_delta": candidate_continuous - prior_continuous,
            "continuous_nll_is_report_only": True,
        }
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
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
