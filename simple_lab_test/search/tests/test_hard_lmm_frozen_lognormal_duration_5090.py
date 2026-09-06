from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import paper.scripts.run_hard_lmm_frozen_lognormal_duration_5090 as controller
from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    FrozenFeatureCache,
    build_frozen_lognormal_candidate,
    censor_mask,
    censor_mask_sha256,
    evaluate_cached_time_metrics,
    freeze_time_head_only,
    proper_time_log_likelihood,
    quantity_metrics,
    state_partition_sha256,
    stratified_quantity_metrics,
    target_dt_sha256,
    tensor_sha256,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration_5090 import (
    EXPECTED_CONTRACT_SHA256,
    TIME_KEYS,
    audit_run,
    gpu_preflight,
    runner_command,
    sha256_file,
    validate_contract,
    verify_inputs,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


CONTRACT_PATH = (
    ROOT / "paper/contracts/hard_lmm_frozen_lognormal_duration_v1.json"
)
B_TRAINING_REVISION = "f75243473adc25d622319dbca9bda7e076d8240f"
CALIBRATION_REVISION = "c" * 40


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_runner_e1_is_one_full_data_epoch_and_full_reuses_e1_cache(tmp_path):
    row = {
        "dataset": "synthetic",
        "data_path": "data.parquet",
        "split_manifest_path": "split.json",
    }
    e1 = runner_command(
        python="python",
        project=tmp_path,
        b_root=tmp_path / "B",
        output=tmp_path / "out" / "e1" / "synthetic",
        revision="a" * 40,
        row=row,
        phase="e1",
    )
    assert "--calibration-source-revision" in e1
    assert "--source-revision" not in e1
    assert e1[e1.index("--max-epochs") + 1] == "1"
    assert "--max-train-batches" not in e1
    assert "--max-validation-batches" not in e1

    full = runner_command(
        python="python",
        project=tmp_path,
        b_root=tmp_path / "B",
        output=tmp_path / "out" / "full" / "synthetic",
        revision="a" * 40,
        row=row,
        phase="full",
    )
    assert "--max-epochs" not in full
    assert full[full.index("--feature-cache-dir") + 1] == str(
        tmp_path / "out" / "e1" / "synthetic" / "cache"
    )


def test_checked_in_contract_pins_common_policy_and_censor_identities():
    assert sha256_file(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256
    datasets = validate_contract(load_contract())
    assert tuple(datasets) == (
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    )
    assert datasets["insta_market_basket"]["right_censor_threshold"] == 30.0
    assert datasets["insta_market_basket"]["expected_train_censored_targets"] == 189607
    assert datasets["insta_market_basket"]["expected_validation_censored_targets"] == 63923
    for name in ("intermittent_frozen_5000", "yellow_trip_hourly"):
        assert datasets[name]["right_censor_threshold"] is None
        assert datasets[name]["expected_train_censored_targets"] == 0
        assert datasets[name]["expected_validation_censored_targets"] == 0


def test_contract_validation_rejects_tolerance_masking_and_censor_drift():
    contract = load_contract()
    weakened = copy.deepcopy(contract)
    weakened["identity_and_stability"][
        "proper_time_nll_replay_absolute_tolerance"
    ] = 1.0
    with pytest.raises(ValueError, match="proper_time_nll_replay_absolute_tolerance"):
        validate_contract(weakened)

    unmasked = copy.deepcopy(contract)
    unmasked["parameter_boundary"]["target_quantity_masked_before_encoding"] = False
    with pytest.raises(ValueError, match="masking drift"):
        validate_contract(unmasked)

    uncensored = copy.deepcopy(contract)
    uncensored["datasets"][2]["right_censor_threshold"] = None
    with pytest.raises(ValueError, match="Censor threshold drift"):
        validate_contract(uncensored)

    wrong_count = copy.deepcopy(contract)
    wrong_count["datasets"][2]["expected_validation_censored_targets"] -= 1
    with pytest.raises(ValueError, match="Validation censor count drift"):
        validate_contract(wrong_count)


def test_gpu_preflight_checks_gpu_workload_without_generic_process_matching(monkeypatch):
    commands: list[tuple[str, ...]] = []

    def fake_output(command, *, allowed=(0,)):
        commands.append(tuple(command))
        if "--query-gpu=name,memory.free" in command:
            return "NVIDIA GeForce RTX 5090, 32000"
        if "--query-compute-apps=pid" in command:
            return ""
        if command == ["systemctl", "is-active", "gdm"]:
            return "inactive"
        raise AssertionError(command)

    monkeypatch.setattr(controller, "command_output", fake_output)
    result = gpu_preflight()
    assert result["compute_processes"] == []
    assert result["gdm"] == "inactive"
    assert all(command[0] != "pgrep" for command in commands)

    def busy_output(command, *, allowed=(0,)):
        if "--query-gpu=name,memory.free" in command:
            return "NVIDIA GeForce RTX 5090, 32000"
        if "--query-compute-apps=pid" in command:
            return "12345"
        return "inactive"

    monkeypatch.setattr(controller, "command_output", busy_output)
    with pytest.raises(ValueError, match="compute process"):
        gpu_preflight()


def test_required_device_duration_head_contract_is_finite_and_isolated():
    require_cuda = os.environ.get(
        "HARD_LMM_FROZEN_LOGNORMAL_REQUIRE_CUDA"
    ) == "1"
    if require_cuda:
        assert torch.cuda.is_available(), "Controller-required CUDA is unavailable"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, _ = build_count_aware_model(
        "titantpp",
        hidden_dim=64,
        train_log_mean=1.0,
        train_log_std=0.5,
        max_seq_len=8,
        quantity_variant=LOG_MSE_VARIANT,
        lambda_tail=0.0,
        time_head_mode="heteroscedastic_lognormal_duration",
        time_scale=1.0,
        time_initial_location=0.1,
        time_initial_scale=0.8,
    )
    model = model.to(device)
    parameters = freeze_time_head_only(model)
    hidden = torch.randn(4, 64, device=device)
    target_dt = torch.tensor([0.5, 1.0, 2.0, 30.0], device=device)
    is_censored = torch.tensor(
        [False, False, False, True], device=device
    )
    frozen_before = state_partition_sha256(
        model.state_dict(), time_head=False
    )
    quantity_before = model.predict_quantity(hidden)[1].detach().clone()
    log_likelihood = proper_time_log_likelihood(
        model,
        hidden,
        target_dt,
        is_right_censored=is_censored,
    )
    assert torch.isfinite(log_likelihood).all()
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    (-log_likelihood.mean()).backward()
    for name, parameter in model.named_parameters():
        if name in TIME_KEYS:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    optimizer.step()
    assert state_partition_sha256(model.state_dict(), time_head=False) == frozen_before
    assert torch.equal(model.predict_quantity(hidden)[1], quantity_before)


def _clone_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def write_synthetic_audit_artifacts(tmp_path: Path):
    torch.manual_seed(7)
    source_model, encoder = build_count_aware_model(
        "titantpp",
        hidden_dim=64,
        train_log_mean=1.0,
        train_log_std=0.5,
        max_seq_len=8,
        quantity_variant=LOG_MSE_VARIANT,
        lambda_tail=0.0,
        time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED,
        time_intercept_limit=300.0,
    )
    source_state = _clone_state(source_model.state_dict())
    source_state_sha = canonical_state_dict_sha256(source_state)
    interface = {
        "mode": "mark_free_count_aware_log_regression",
        "train_target_mean": 1.0,
        "train_target_std": 0.5,
        "target_quantity_masked_from_history": True,
    }

    quantity_hidden = torch.randn(2, 64)
    with torch.no_grad():
        source_quantity_prediction = source_model.predict_quantity(quantity_hidden)[1].cpu()
    target_quantity = torch.tensor([2.0, 5.0])
    b_metrics = quantity_metrics(source_quantity_prediction, target_quantity)
    b_metrics.update(
        stratified_quantity_metrics(
            source_quantity_prediction,
            target_quantity,
            body_max=3.0,
            tail_min_exclusive=4.0,
            require_nonempty=True,
        )
    )
    source_payload = {
        "backbone": "titantpp",
        "variant": "count_only_log_regression",
        "seed": 42,
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection": "best_validation_raw_quantity_rmse",
        "selected_metric_value": b_metrics["rmse"],
        "source_revision": B_TRAINING_REVISION,
        "source_revision_history": [B_TRAINING_REVISION],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "encoder_config": encoder,
        "interface_meta": interface,
        "model_state_dict": source_state,
        "model_state_sha256": source_state_sha,
    }
    source_checkpoint = tmp_path / "source.pt"
    torch.save(source_payload, source_checkpoint)

    train_dt = torch.tensor([1.0, 30.0, 2.0])
    validation_dt = torch.tensor([30.0, 2.0])
    train_mask = censor_mask(train_dt, threshold=30.0)
    validation_mask = censor_mask(validation_dt, threshold=30.0)
    row = {
        "dataset": "insta_market_basket",
        "data_sha256": "d" * 64,
        "split_manifest_sha256": "s" * 64,
        "max_sequence_length": 8,
        "expected_train_targets": 3,
        "expected_validation_targets": 2,
        "expected_train_target_identity_sha256": "i" * 64,
        "expected_train_target_quantity_sha256": "j" * 64,
        "expected_validation_target_identity_sha256": "v" * 64,
        "expected_validation_target_quantity_sha256": "u" * 64,
        "expected_train_target_dt_sha256": target_dt_sha256(train_dt),
        "expected_validation_target_dt_sha256": target_dt_sha256(validation_dt),
        "expected_train_censor_mask_sha256": censor_mask_sha256(train_mask),
        "expected_validation_censor_mask_sha256": censor_mask_sha256(validation_mask),
        "right_censor_threshold": 30.0,
        "expected_train_censored_targets": 1,
        "expected_validation_censored_targets": 1,
        "train_time_scale": 1.0,
        "train_log_scaled_mean": 0.1,
        "train_log_scaled_std": 0.8,
        "reporting_body_max_train_p95": 3.0,
        "reporting_tail_min_exclusive_train_p99": 4.0,
        "B_checkpoint_file_sha256": sha256_file(source_checkpoint),
        "B_checkpoint_state_sha256": source_state_sha,
        "B_metrics": {
            "raw_rmse": b_metrics["rmse"],
            "overall_mae": b_metrics["mae"],
            "body_mae": b_metrics["body_mae"],
            "gt_p99_mae": b_metrics["gt_p99_mae"],
        },
    }
    train_statistics = {
        "statistics_source_split": "train",
        "time_scale": 1.0,
        "target_log_scaled_mean": 0.1,
        "target_log_scaled_std": 0.8,
    }
    candidate, candidate_encoder, metadata = build_frozen_lognormal_candidate(
        source_payload,
        train_time_statistics=train_statistics,
        time_sigma_floor=0.001,
    )
    selected_state = _clone_state(candidate.state_dict())
    selected_state_sha = canonical_state_dict_sha256(selected_state)
    source_non_time_sha = state_partition_sha256(source_state, time_head=False)
    selected_time_sha = state_partition_sha256(selected_state, time_head=True)

    train_cache = FrozenFeatureCache(
        time_hidden=torch.randn(3, 64),
        target_dt=train_dt,
    )
    validation_cache = FrozenFeatureCache(
        time_hidden=torch.randn(2, 64),
        target_dt=validation_dt,
        quantity_hidden=quantity_hidden,
        target_quantity=target_quantity,
        source_quantity_prediction=source_quantity_prediction,
    )
    output = tmp_path / "run"
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True)
    for split, cache in (("train", train_cache), ("validation", validation_cache)):
        identity = controller._expected_cache_identity(row, split=split)
        torch.save(
            cache.to_payload(identity=identity),
            cache_dir / f"{split}_features.pt",
        )

    initial_metrics = evaluate_cached_time_metrics(
        model=candidate,
        cache=validation_cache,
        device="cpu",
        batch_size=8192,
        censor_threshold=30.0,
    )
    epoch_zero_nll = initial_metrics["proper_time_nll"]
    history = [
        {
            "epoch": 0,
            "train_proper_time_nll": None,
            "val_proper_time_nll": epoch_zero_nll,
            "val_time_median_mae": initial_metrics["time_median_mae"],
            "val_time_median_rmse": initial_metrics["time_median_rmse"],
            "pre_clip_gradient_norm_mean": None,
            "pre_clip_gradient_norm_max": None,
        },
        {
            "epoch": 1,
            "train_proper_time_nll": epoch_zero_nll + 0.2,
            "val_proper_time_nll": epoch_zero_nll + 0.1,
            "val_time_median_mae": initial_metrics["time_median_mae"],
            "val_time_median_rmse": initial_metrics["time_median_rmse"],
            "pre_clip_gradient_norm_mean": 0.5,
            "pre_clip_gradient_norm_max": 0.75,
        },
    ]
    resume_identity = {
        "schema_version": 1,
        "contract_id": "hard_lmm_frozen_lognormal_duration_v1",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "dataset": row["dataset"],
        "source_checkpoint_sha256": row["B_checkpoint_file_sha256"],
        "source_state_sha256": source_state_sha,
        "source_non_time_state_sha256": source_non_time_sha,
        "candidate_initial_state_sha256": selected_state_sha,
        "train_cache_sha256": train_cache.digest(),
        "validation_cache_sha256": validation_cache.digest(),
        "train_cache_count": train_cache.count,
        "validation_cache_count": validation_cache.count,
        "train_target_dt_sha256": row["expected_train_target_dt_sha256"],
        "validation_target_dt_sha256": row["expected_validation_target_dt_sha256"],
        "train_censor_mask_sha256": row["expected_train_censor_mask_sha256"],
        "validation_censor_mask_sha256": row["expected_validation_censor_mask_sha256"],
        "right_censor_threshold": 30.0,
        "train_time_statistics": train_statistics,
        "calibration_source_revision": CALIBRATION_REVISION,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "cached_state_batch_size": 8192,
        "quantity_replay_batch_size": 128,
        "gradient_clip": 1.0,
        "minimum_epochs": 20,
        "early_stopping_patience": 20,
        "planned_epochs": 1,
        "selection": "earliest_strict_finite_minimum_validation_proper_time_nll",
        "trainable_parameter_names": list(TIME_KEYS),
        "time_head_mode": "heteroscedastic_lognormal_duration",
        "likelihood": "censor_aware_proper_lognormal_duration",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    selected_checkpoint = {
        "checkpoint_type": "selected_frozen_lognormal_duration",
        "checkpoint_schema_version": 1,
        "selection": "earliest_strict_finite_minimum_validation_proper_time_nll",
        "selection_formula": "mean negative censor-aware normalized log-normal duration log likelihood",
        "best_epoch": 0,
        "selected_metric_value": epoch_zero_nll,
        "model_state_dict": selected_state,
        "model_state_sha256": selected_state_sha,
        "source_state_sha256": source_state_sha,
        "source_non_time_state_sha256": source_non_time_sha,
        "candidate_initial_state_sha256": selected_state_sha,
        "candidate_initial_time_head_state_sha256": metadata[
            "candidate_initial_time_head_state_sha256"
        ],
        "selected_time_head_state_sha256": selected_time_sha,
        "backbone": "titantpp",
        "variant": "count_only_log_regression",
        "time_head_mode": "heteroscedastic_lognormal_duration",
        "encoder_config": candidate_encoder,
        "interface_meta": metadata["interface_meta"],
        "source_checkpoint_lineage": {
            "B_checkpoint_file_sha256": row["B_checkpoint_file_sha256"],
            "B_checkpoint_state_sha256": source_state_sha,
            "B_training_source_revision": B_TRAINING_REVISION,
            "B_training_source_revision_history": [B_TRAINING_REVISION],
            "calibration_source_revision": CALIBRATION_REVISION,
        },
        "train_time_statistics": train_statistics,
        "right_censor_threshold": 30.0,
        "resume_identity": resume_identity,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    selected_path = output / "best_validation_proper_time_nll_model.pt"
    torch.save(selected_checkpoint, selected_path)

    current_state = _clone_state(selected_state)
    current_state["b_t"] += 0.01
    optimizer_state = {
        "state": {
            index: {
                "step": torch.tensor(1.0, dtype=torch.float32),
                "exp_avg": torch.zeros_like(current_state[name]),
                "exp_avg_sq": torch.zeros_like(current_state[name]),
            }
            for index, name in enumerate(TIME_KEYS)
        },
        "param_groups": [
            {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "params": [0, 1, 2, 3],
            }
        ],
    }
    torch.save(
        {
            "checkpoint_type": "frozen_lognormal_duration_resume",
            "checkpoint_schema_version": 1,
            "epoch": 1,
            "model_state_dict": current_state,
            "model_state_sha256": canonical_state_dict_sha256(current_state),
            "optimizer_state_dict": optimizer_state,
            "history": history,
            "best_epoch": 0,
            "best_state_dict": selected_state,
            "best_state_sha256": selected_state_sha,
            "source_non_time_state_sha256": source_non_time_sha,
            "resume_identity": resume_identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "legacy_nll_compared": False,
        },
        output / "last_epoch_state.pt",
    )
    quantity_summary = dict(b_metrics)
    summary = {
        "schema_version": 1,
        "contract_id": "hard_lmm_frozen_lognormal_duration_v1",
        "status": "success",
        "dataset": row["dataset"],
        "seed": 42,
        "time_head_mode": "heteroscedastic_lognormal_duration",
        "source_checkpoint_sha256": row["B_checkpoint_file_sha256"],
        "source_state_sha256": source_state_sha,
        "source_non_time_state_sha256": source_non_time_sha,
        "candidate_initial_state_sha256": selected_state_sha,
        "candidate_initial_time_head_state_sha256": metadata[
            "candidate_initial_time_head_state_sha256"
        ],
        "selected_state_sha256": selected_state_sha,
        "selected_non_time_state_sha256": source_non_time_sha,
        "selected_time_head_state_sha256": selected_time_sha,
        "changed_state_keys": [],
        "trainable_parameter_names": list(TIME_KEYS),
        "trainable_parameter_count": 130,
        "encoder_mode_during_refit": "eval",
        "hidden_state_gradient": "detached_cache",
        "selection": "earliest_strict_finite_minimum_validation_proper_time_nll",
        "best_epoch": 0,
        "completed_epochs": 1,
        "stopped_early": False,
        "epoch_zero_validation_proper_time_nll": epoch_zero_nll,
        "best_validation_proper_time_nll": epoch_zero_nll,
        "proper_time_nll_improvement_from_epoch_zero": 0.0,
        "strictly_improves_epoch_zero": False,
        "validation_time_metrics": initial_metrics,
        "time_median_mae": initial_metrics["time_median_mae"],
        "time_median_rmse": initial_metrics["time_median_rmse"],
        "right_censor_threshold": 30.0,
        "train_right_censored_count": 1,
        "validation_right_censored_count": 1,
        "quantity_prediction_bitwise_identical": True,
        "source_quantity_prediction_sha256": tensor_sha256(
            "quantity_prediction", source_quantity_prediction
        ),
        "selected_quantity_prediction_sha256": tensor_sha256(
            "quantity_prediction", source_quantity_prediction
        ),
        "quantity_metrics": quantity_summary,
        "train_time_statistics": train_statistics,
        "train_cache": {
            "count": 3,
            "sha256": train_cache.digest(),
            "target_dt_sha256": row["expected_train_target_dt_sha256"],
            "censor_mask_sha256": row["expected_train_censor_mask_sha256"],
        },
        "validation_cache": {
            "count": 2,
            "sha256": validation_cache.digest(),
            "target_dt_sha256": row["expected_validation_target_dt_sha256"],
            "censor_mask_sha256": row["expected_validation_censor_mask_sha256"],
        },
        "train_target_population": {
            "target_count": 3,
            "target_identity_sha256": row["expected_train_target_identity_sha256"],
            "target_quantity_sha256": row["expected_train_target_quantity_sha256"],
        },
        "validation_target_population": {
            "target_count": 2,
            "target_identity_sha256": row[
                "expected_validation_target_identity_sha256"
            ],
            "target_quantity_sha256": row[
                "expected_validation_target_quantity_sha256"
            ],
        },
        "train_observation_contract": {
            "split": "train",
            "target_count": 3,
            "target_dt_sha256": row["expected_train_target_dt_sha256"],
            "right_censor_threshold": 30.0,
            "right_censored_count": 1,
            "censor_mask_sha256": row["expected_train_censor_mask_sha256"],
        },
        "validation_observation_contract": {
            "split": "validation",
            "target_count": 2,
            "target_dt_sha256": row["expected_validation_target_dt_sha256"],
            "right_censor_threshold": 30.0,
            "right_censored_count": 1,
            "censor_mask_sha256": row["expected_validation_censor_mask_sha256"],
        },
        "qualified_full_data": True,
        "history": history,
        "resume_identity": resume_identity,
        "selected_checkpoint_path": str(selected_path),
        "feature_cache_dir": str(cache_dir.resolve()),
        "runtime": {
            "requested_device": "cuda",
            "device_name": "NVIDIA GeForce RTX 5090",
            "cuda_available": True,
            "peak_memory_allocated_bytes": 1,
            "peak_memory_reserved_bytes": 1,
            "elapsed_seconds": 1.0,
        },
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return output, row, source_checkpoint, summary


def test_audit_replays_proper_likelihood_quantity_and_resume(tmp_path):
    output, row, source_checkpoint, _ = write_synthetic_audit_artifacts(tmp_path)
    audit = audit_run(
        output,
        contract=load_contract(),
        row=row,
        phase="e1",
        source_checkpoint=source_checkpoint,
        source_revision=CALIBRATION_REVISION,
    )
    assert audit["strictly_improves_epoch_zero"] is False
    assert audit["target_dt_and_censor_identity_verified"] is True
    assert audit["resume_and_checkpoint_replay_verified"] is True
    assert audit["changed_state_keys"] == []


def test_audit_matches_source_and_candidate_on_same_device(tmp_path):
    output, row, source_checkpoint, summary = write_synthetic_audit_artifacts(
        tmp_path
    )
    cache_path = output / "cache" / "validation_features.pt"
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached_prediction = payload["source_quantity_prediction"].clone()
    cached_prediction[0] = torch.nextafter(
        cached_prediction[0], torch.tensor(float("inf"))
    )
    assert not torch.equal(
        cached_prediction, payload["source_quantity_prediction"]
    )
    cache = FrozenFeatureCache(
        time_hidden=payload["time_hidden"],
        target_dt=payload["target_dt"],
        quantity_hidden=payload["quantity_hidden"],
        target_quantity=payload["target_quantity"],
        source_quantity_prediction=cached_prediction,
    )
    torch.save(cache.to_payload(identity=payload["identity"]), cache_path)
    cached_sha = tensor_sha256("quantity_prediction", cached_prediction)
    cache_sha = cache.digest()
    summary["validation_cache"]["sha256"] = cache_sha
    summary["resume_identity"]["validation_cache_sha256"] = cache_sha
    summary["source_quantity_prediction_sha256"] = cached_sha
    summary["selected_quantity_prediction_sha256"] = cached_sha
    (output / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    for checkpoint_name in (
        "best_validation_proper_time_nll_model.pt",
        "last_epoch_state.pt",
    ):
        checkpoint_path = output / checkpoint_name
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint["resume_identity"]["validation_cache_sha256"] = cache_sha
        torch.save(checkpoint, checkpoint_path)

    audit = audit_run(
        output,
        contract=load_contract(),
        row=row,
        phase="e1",
        source_checkpoint=source_checkpoint,
        source_revision=CALIBRATION_REVISION,
    )

    assert audit["quantity_prediction_bitwise_identical"] is True
    assert audit["cached_cuda_vs_cpu_source_close"] is True
    assert audit["cached_cuda_quantity_prediction_sha256"] == cached_sha
    assert (
        audit["matched_cpu_quantity_prediction_sha256"] != cached_sha
    )


def test_audit_rejects_non_time_state_mutation(tmp_path):
    output, row, source_checkpoint, summary = write_synthetic_audit_artifacts(tmp_path)
    path = output / "best_validation_proper_time_nll_model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutable_name = next(name for name in payload["model_state_dict"] if name not in TIME_KEYS)
    payload["model_state_dict"][mutable_name].view(-1)[0] += 1.0
    payload["model_state_sha256"] = canonical_state_dict_sha256(
        payload["model_state_dict"]
    )
    summary["selected_state_sha256"] = payload["model_state_sha256"]
    torch.save(payload, path)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="non-time state drift"):
        audit_run(
            output,
            contract=load_contract(),
            row=row,
            phase="e1",
            source_checkpoint=source_checkpoint,
            source_revision=CALIBRATION_REVISION,
        )


def test_audit_rejects_duration_or_censor_cache_drift(tmp_path):
    output, row, source_checkpoint, _ = write_synthetic_audit_artifacts(tmp_path)
    path = output / "cache" / "validation_features.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["target_dt"][0] = 29.0
    cache = FrozenFeatureCache(
        time_hidden=payload["time_hidden"],
        target_dt=payload["target_dt"],
        quantity_hidden=payload["quantity_hidden"],
        target_quantity=payload["target_quantity"],
        source_quantity_prediction=payload["source_quantity_prediction"],
    )
    payload["cache_sha256"] = cache.digest()
    torch.save(payload, path)

    with pytest.raises(ValueError, match="target-dt digest drift"):
        audit_run(
            output,
            contract=load_contract(),
            row=row,
            phase="e1",
            source_checkpoint=source_checkpoint,
            source_revision=CALIBRATION_REVISION,
        )


def test_verify_inputs_checks_b_file_and_model_state_hashes(tmp_path):
    output, row, source_checkpoint, _ = write_synthetic_audit_artifacts(tmp_path)
    del output
    project = tmp_path / "project"
    b_root = tmp_path / "B"
    project.mkdir()
    data = project / "data.parquet"
    split = project / "split.json"
    data.write_bytes(b"data")
    split.write_bytes(b"split")
    destination = controller.checkpoint_path(b_root, row["dataset"])
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source_checkpoint.read_bytes())
    checked_row = {
        **row,
        "data_path": "data.parquet",
        "data_sha256": sha256_file(data),
        "split_manifest_path": "split.json",
        "split_manifest_sha256": sha256_file(split),
        "B_checkpoint_path": str(
            Path("artifact")
            / destination.relative_to(b_root)
        ),
        "B_checkpoint_file_sha256": sha256_file(destination),
    }
    verify_inputs(project, b_root, {row["dataset"]: checked_row})

    wrong_state = dict(checked_row)
    wrong_state["B_checkpoint_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model-state checksum drift"):
        verify_inputs(project, b_root, {row["dataset"]: wrong_state})
