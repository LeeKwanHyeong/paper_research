#!/usr/bin/env python3
"""Fit the matched proper duration head on one frozen comparison backbone.

This runner is intentionally limited to the eight available seed-42 A,
RMTPP, and THP rows in ``matched_frozen_lognormal_duration_v1``.  The already
audited B results are consumed by the 5090 controller and are never refit here.
Only train and validation histories are admitted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    SharedTimeCountModel,
)
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.run_count_aware_tpp_backbone_control import (
    derive_train_time_contract,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    SUMMARY_NAME,
    build_frozen_lognormal_candidate,
    fit_time_head_from_cache,
    load_admitted_frame,
    resolve_censor_threshold,
    state_partition_sha256,
    train_time_statistics_from_contract,
    validate_cache_observation_contract,
)
from paper.scripts.run_hard_lmm_time_head_refit import (
    exact_target_population_contract,
    finite_tensor_mapping,
    load_or_extract_feature_cache,
    require,
    save_json,
    sha256_file,
    stratified_quantity_metrics,
    validate_target_population,
)
from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "matched_frozen_lognormal_duration_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/matched_frozen_lognormal_duration_v1.json"
)
FIT_ROLES = ("A", "rmtpp", "thp")
ALL_ROLES = ("A", "B", "rmtpp", "thp")


def _hidden_dim(encoder: Mapping[str, Any]) -> int:
    value = encoder.get("d_model", encoder.get("hidden_dim"))
    require(type(value) is int and int(value) > 0, "Invalid source hidden dimension")
    return int(value)


def _legacy_time_contract(
    encoder: Mapping[str, Any], interface: Mapping[str, Any]
) -> dict[str, Any]:
    """Recover the legacy head arguments, including old THP checkpoints.

    The earliest Intermittent THP checkpoint predates serialized ``time_head``
    metadata.  Its pinned binary and state digest establish identity, and the
    historical defaults below reproduce its strict state schema.
    """
    value = encoder.get("time_head")
    if not isinstance(value, Mapping):
        value = interface.get("time_head")
    if not isinstance(value, Mapping):
        value = {}
    return {
        "mode": str(value.get("mode", TIME_HEAD_MODE_LEGACY_CLAMPED)),
        "time_scale": float(value.get("time_scale", 3.0)),
        "time_w_max": float(value.get("time_w_max", 10.0 / 3.0)),
        "time_intercept_limit": float(value.get("time_intercept_limit", 30.0)),
        "time_initial_intercept": float(value.get("time_initial_intercept", 0.0)),
        "time_wd_safety_limit": float(value.get("time_wd_safety_limit", 40.0)),
        "time_sigma_floor": float(value.get("time_sigma_floor", 1e-3)),
    }


def validate_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong matched contract")
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Matched scope is missing")
    require(scope.get("model_roles") == list(ALL_ROLES), "Model-role scope drift")
    require(scope.get("input_splits") == ["train", "validation"], "Split scope drift")
    require(scope.get("optimization_split") == "train", "Optimization scope drift")
    require(scope.get("selection_split") == "validation", "Selection scope drift")
    require(scope.get("evaluation_scope") == "validation_only", "Evaluation scope drift")
    require(scope.get("held_out_test") is False, "Held-out access is enabled")
    require(scope.get("seed") == 42, "Seed drift")
    require(scope.get("additional_seeds") is False, "Additional seeds are enabled")
    policy = contract.get("comparison_policy")
    require(isinstance(policy, Mapping), "Comparison policy is missing")
    require(policy.get("same_duration_head_for_every_role") is True, "Head parity drift")
    require(policy.get("same_optimization_for_every_new_fit") is True, "Optimization parity drift")
    require(
        policy.get("B_result_policy")
        == "reuse the audited c04d32b full result without refitting",
        "B reuse policy drift",
    )
    require(
        policy.get("cross_dataset_nll_aggregation") is False,
        "Cross-dataset NLL aggregation enabled",
    )
    require(policy.get("legacy_nll_comparison") is False, "Legacy NLL comparison enabled")
    require(
        policy.get("four_way_complete_requires_all_twelve_rows") is True,
        "Four-way completeness policy drift",
    )
    head = contract.get("time_head")
    require(isinstance(head, Mapping), "Duration-head contract is missing")
    require(
        head.get("trainable_parameter_names")
        == ["v_t.weight", "b_t", "w_raw", "time_scale_weight.weight"],
        "Duration-head parameter boundary drift",
    )
    require(head.get("expected_trainable_parameter_count") == 130, "Head capacity drift")
    for name, expected in {
        "location": "mu(h) = v_t(h) + b_t",
        "scale": "sigma(h) = 0.001 + softplus(time_scale_weight(h) + w_raw)",
        "time_scale": "exact train-target median",
        "original_unit_jacobian": True,
        "duration_clamp": False,
        "calculation_dtype": "float64",
    }.items():
        require(head.get(name) == expected, f"Duration-head drift: {name}")
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
        "dataset_specific_hyperparameters": False,
    }
    for name, expected in expected_optimization.items():
        require(optimization.get(name) == expected, f"Optimization drift: {name}")
    selection = contract.get("checkpoint_selection")
    require(isinstance(selection, Mapping), "Selection contract is missing")
    require(selection.get("monitor") == "validation_proper_time_nll", "Selector drift")
    require(selection.get("rule") == "earliest strict finite minimum", "Selector rule drift")
    require(selection.get("other_metrics_in_selector") is False, "Selector leakage")
    stability = contract.get("identity_and_stability")
    require(isinstance(stability, Mapping), "Identity contract is missing")
    require(
        stability.get("quantity_mae_and_rmse_required")
        == "calculation-identical to each pinned source checkpoint",
        "Quantity identity contract drift",
    )
    rows = contract.get("datasets")
    require(isinstance(rows, list) and len(rows) == 3, "Dataset scope drift")
    datasets = {str(row["dataset"]): dict(row) for row in rows}
    require(list(datasets) == scope.get("datasets"), "Dataset order drift")
    planned = []
    for dataset, row in datasets.items():
        require(int(row["expected_train_targets"]) > 0, f"Empty train row: {dataset}")
        require(int(row["expected_validation_targets"]) > 0, f"Empty validation row: {dataset}")
        for split in ("train", "validation"):
            for suffix in (
                "target_identity_sha256",
                "target_quantity_sha256",
                "target_dt_sha256",
                "censor_mask_sha256",
            ):
                require(
                    len(str(row.get(f"expected_{split}_{suffix}"))) == 64,
                    f"Missing pinned identity: {dataset}.{split}.{suffix}",
                )
        threshold = row.get("right_censor_threshold")
        require(
            (dataset == "insta_market_basket" and threshold == 30.0)
            or (dataset != "insta_market_basket" and threshold is None),
            f"Censor contract drift: {dataset}",
        )
        expected_train_censored = 189607 if dataset == "insta_market_basket" else 0
        expected_validation_censored = 63923 if dataset == "insta_market_basket" else 0
        require(
            int(row["expected_train_censored_targets"])
            == expected_train_censored,
            f"Train censor count drift: {dataset}",
        )
        require(
            int(row["expected_validation_censored_targets"])
            == expected_validation_censored,
            f"Validation censor count drift: {dataset}",
        )
        sources = row.get("sources")
        require(isinstance(sources, Mapping) and list(sources) == list(ALL_ROLES), f"Source roles drift: {dataset}")
        for role in FIT_ROLES:
            source = sources[role]
            if source.get("available") is True:
                require(source.get("execution") == "fit_new", f"Execution drift: {dataset}/{role}")
                require(len(str(source.get("checkpoint_file_sha256"))) == 64, f"Checkpoint hash missing: {dataset}/{role}")
                require(len(str(source.get("checkpoint_state_sha256"))) == 64, f"State hash missing: {dataset}/{role}")
                planned.append({"dataset": dataset, "model_role": role, "phase_order": ["e1", "full"]})
        if dataset == "intermittent_frozen_5000":
            missing = sources["rmtpp"]
            require(missing.get("available") is False, "Missing RMTPP was silently admitted")
            require(missing.get("execution") == "blocked_missing_checkpoint", "Missing RMTPP status drift")
    require(contract.get("execution_order") == planned, "Eight-row execution order drift")
    require(len(planned) == 8, "Matched execution must contain exactly eight new rows")
    likelihood = contract.get("observation_likelihood")
    require(isinstance(likelihood, Mapping), "Observation likelihood is missing")
    require(
        likelihood.get("uncensored") == "log f(dt | h) in the original time unit",
        "Uncensored likelihood drift",
    )
    require(
        likelihood.get("right_censored") == "log S(threshold | h)",
        "Right-censored likelihood drift",
    )
    return datasets


def validate_source_checkpoint(
    payload: Mapping[str, Any],
    *,
    source_spec: Mapping[str, Any],
) -> None:
    require(source_spec.get("available") is True, "Unavailable source checkpoint")
    for name in ("backbone", "quantity_variant", "seed"):
        payload_name = "variant" if name == "quantity_variant" else name
        require(payload.get(payload_name) == source_spec.get(name), f"Source metadata drift: {name}")
    require(payload.get("selection") == source_spec.get("checkpoint_selection"), "Source selector drift")
    require(payload.get("evaluation_scope") == "validation_only", "Source scope drift")
    require(payload.get("held_out_test_evaluated") is False, "Source used held-out test")
    require(payload.get("source_revision") == source_spec.get("training_source_revision"), "Source revision drift")
    require(payload.get("source_revision_history") == source_spec.get("training_source_revision_history"), "Source revision history drift")
    state = payload.get("model_state_dict")
    require(isinstance(state, Mapping), "Source model state is missing")
    finite_tensor_mapping(state, label="Source model state")
    observed = canonical_state_dict_sha256(state)
    require(observed == payload.get("model_state_sha256"), "Payload state digest drift")
    require(observed == source_spec.get("checkpoint_state_sha256"), "Pinned state digest drift")
    encoder = payload.get("encoder_config")
    require(isinstance(encoder, Mapping), "Source encoder metadata is missing")
    require(_hidden_dim(encoder) == 64, "Matched head requires hidden dimension 64")
    interface = payload.get("interface_meta")
    require(isinstance(interface, Mapping), "Source interface metadata is missing")
    require(
        interface.get("mode") == "mark_free_count_aware_log_regression",
        "Source mark-free interface drift",
    )
    require(interface.get("target_quantity_masked_from_history") is True, "Target quantity masking drift")
    require(payload.get("variant") == LOG_MSE_VARIANT, "Quantity interface drift")


def build_source_model(
    payload: Mapping[str, Any],
    *,
    max_seq_len: int,
) -> SharedTimeCountModel:
    encoder = payload["encoder_config"]
    interface = payload["interface_meta"]
    time = _legacy_time_contract(encoder, interface)
    require(time["mode"] == TIME_HEAD_MODE_LEGACY_CLAMPED, "Source time-head mode drift")
    model, _ = build_count_aware_model(
        str(payload["backbone"]),
        hidden_dim=_hidden_dim(encoder),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface["train_target_std"]),
        max_seq_len=int(encoder.get("max_len", max_seq_len)),
        quantity_variant=str(payload["variant"]),
        lambda_tail=0.0,
        time_head_mode=time["mode"],
        time_scale=time["time_scale"],
        time_w_max=time["time_w_max"],
        time_intercept_limit=time["time_intercept_limit"],
        time_initial_intercept=time["time_initial_intercept"],
        time_wd_safety_limit=time["time_wd_safety_limit"],
        time_sigma_floor=time["time_sigma_floor"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    require(
        canonical_state_dict_sha256(model.state_dict()) == payload["model_state_sha256"],
        "Strict source replay digest drift",
    )
    model.eval()
    return model


def dataset_by_id(contract: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    datasets = validate_contract(contract)
    require(dataset in datasets, "Dataset is outside the matched contract")
    return datasets[dataset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-role", choices=FIT_ROLES, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-source-revision", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but unavailable")
        torch.cuda.reset_peak_memory_stats(device)
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)
    require(len(args.calibration_source_revision) == 40, "Calibration revision must be a full Git SHA")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dataset_spec = dataset_by_id(contract, args.dataset)
    source_spec = dataset_spec["sources"][args.model_role]
    require(source_spec.get("available") is True, "Requested source is unavailable")
    require(source_spec.get("execution") == "fit_new", "Requested role is not a new-fit row")
    contract_sha = sha256_file(args.contract)
    require(sha256_file(args.data) == dataset_spec["data_sha256"], "Dataset digest mismatch")
    require(sha256_file(args.split_manifest) == dataset_spec["split_manifest_sha256"], "Split-manifest digest mismatch")
    checkpoint_sha = sha256_file(args.checkpoint)
    require(checkpoint_sha == source_spec["checkpoint_file_sha256"], "Source checkpoint file digest mismatch")
    source_payload = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    validate_source_checkpoint(source_payload, source_spec=source_spec)
    source_state_sha = str(source_payload["model_state_sha256"])
    source_model = build_source_model(
        source_payload, max_seq_len=int(dataset_spec["max_sequence_length"])
    ).to(device)

    frame = prepare_count_frame(load_admitted_frame(args.data))
    admitted = set(frame["chronological_split"].unique().to_list())
    require(admitted <= {"train", "validation"}, "Held-out rows entered the frame")
    lookback = int(dataset_spec["lookback"])
    max_seq_len = int(dataset_spec["max_sequence_length"])
    train_population = exact_target_population_contract(
        frame, target_split="train", lookback=lookback, max_seq_len=max_seq_len
    )
    validation_population = exact_target_population_contract(
        frame, target_split="validation", lookback=lookback, max_seq_len=max_seq_len
    )
    validate_target_population(train_population, dataset_spec=dataset_spec)
    validate_target_population(validation_population, dataset_spec=dataset_spec)
    observed_statistics = derive_train_time_contract(
        frame, lookback_weeks=lookback, max_seq_len=max_seq_len
    )
    train_statistics = train_time_statistics_from_contract(dataset_spec, observed_statistics)
    censor_threshold = resolve_censor_threshold(args.dataset, dataset_spec)

    settings = dict(contract["optimization"])
    partial = (args.max_train_batches, args.max_validation_batches, args.max_epochs)
    if any(value is not None for value in partial):
        require(args.allow_partial_contract, "Batch or epoch limits require --allow-partial-contract")
    planned_epochs = int(settings["epochs"])
    if args.max_epochs is not None:
        require(args.max_epochs >= 0, "max epochs must be nonnegative")
        planned_epochs = min(planned_epochs, args.max_epochs)
    encoder_batch_size = int(settings["encoder_batch_size"])
    train_loader = make_loader(
        frame, target_split="train", batch_size=encoder_batch_size,
        lookback_weeks=lookback, max_seq_len=max_seq_len, shuffle=False, generator=None,
    )
    validation_loader = make_loader(
        frame, target_split="validation", batch_size=encoder_batch_size,
        lookback_weeks=lookback, max_seq_len=max_seq_len, shuffle=False, generator=None,
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
            "Requested shared feature cache is incomplete",
        )
    train_cache = load_or_extract_feature_cache(
        path=cache_dir / "train_features.pt",
        identity={
            **cache_base, "split": "train",
            "target_count": train_population["target_count"],
            "target_identity_sha256": train_population["target_identity_sha256"],
            "target_quantity_sha256": train_population["target_quantity_sha256"],
            "max_batches": args.max_train_batches,
        },
        model=source_model, loader=train_loader, device=device,
        include_quantity=False, max_batches=args.max_train_batches,
    )
    validation_cache = load_or_extract_feature_cache(
        path=cache_dir / "validation_features.pt",
        identity={
            **cache_base, "split": "validation",
            "target_count": validation_population["target_count"],
            "target_identity_sha256": validation_population["target_identity_sha256"],
            "target_quantity_sha256": validation_population["target_quantity_sha256"],
            "max_batches": args.max_validation_batches,
        },
        model=source_model, loader=validation_loader, device=device,
        include_quantity=True, max_batches=args.max_validation_batches,
    )
    full_data = args.max_train_batches is None and args.max_validation_batches is None
    train_observation = validation_observation = None
    if full_data:
        train_observation = validate_cache_observation_contract(
            train_cache, dataset_spec=dataset_spec, split="train",
            censor_threshold=censor_threshold,
        )
        validation_observation = validate_cache_observation_contract(
            validation_cache, dataset_spec=dataset_spec, split="validation",
            censor_threshold=censor_threshold,
        )

    source_non_time_sha = state_partition_sha256(
        source_payload["model_state_dict"], time_head=False
    )
    candidate, candidate_encoder, candidate_metadata = build_frozen_lognormal_candidate(
        source_payload,
        train_time_statistics=train_statistics,
        time_sigma_floor=0.001,
        source_backbone=str(source_spec["backbone"]),
        source_variant=str(source_spec["quantity_variant"]),
        max_seq_len=max_seq_len,
        training_stage="matched_frozen_representation_duration_refit",
    )
    require(candidate_metadata["source_non_time_state_sha256"] == source_non_time_sha, "Candidate non-time copy drift")
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
        "training_source_revision_history": source_payload.get("source_revision_history"),
        "calibration_source_revision": args.calibration_source_revision,
    }
    summary = fit_time_head_from_cache(
        model=candidate, train_cache=train_cache, validation_cache=validation_cache,
        output_dir=args.output_dir, dataset=args.dataset,
        censor_threshold=censor_threshold, contract_sha256=contract_sha,
        source_checkpoint_sha256=checkpoint_sha, source_state_sha256=source_state_sha,
        source_non_time_state_sha256=source_non_time_sha,
        candidate_initial_state_sha256=candidate_metadata["candidate_initial_state_sha256"],
        train_time_statistics=train_statistics,
        calibration_source_revision=args.calibration_source_revision,
        seed=int(settings["seed"]), planned_epochs=planned_epochs,
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        batch_size=int(settings["cached_state_batch_size"]),
        quantity_replay_batch_size=encoder_batch_size,
        grad_clip=float(settings["gradient_clip"]),
        min_epochs=int(settings["minimum_epochs"]),
        patience=int(settings["early_stopping_patience"]), device=device,
        candidate_metadata=candidate_metadata, source_metadata=source_lineage,
        contract_id=CONTRACT_ID, model_role=args.model_role,
        source_backbone=str(source_spec["backbone"]),
        source_variant=str(source_spec["quantity_variant"]),
    )
    if summary.get("status") != "success":
        return summary
    summary = dict(summary)
    summary.update({
        "encoder_config": candidate_encoder,
        "interface_meta": candidate_metadata["interface_meta"],
        "train_target_population": train_population,
        "validation_target_population": validation_population,
        "train_observation_contract": train_observation,
        "validation_observation_contract": validation_observation,
        "qualified_full_data": full_data,
        "feature_cache_dir": str(cache_dir.resolve()),
    })
    assert validation_cache.source_quantity_prediction is not None
    assert validation_cache.target_quantity is not None
    summary["quantity_metrics"].update(
        stratified_quantity_metrics(
            validation_cache.source_quantity_prediction,
            validation_cache.target_quantity,
            body_max=float(dataset_spec["reporting_body_max_train_p95"]),
            tail_min_exclusive=float(dataset_spec["reporting_tail_min_exclusive_train_p99"]),
            require_nonempty=full_data,
        )
    )
    summary["runtime"] = {
        "requested_device": str(device), "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    }
    save_json(args.output_dir / SUMMARY_NAME, summary)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
