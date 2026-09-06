from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.run_hard_lmm_time_head_refit import (
    CONTRACT_ID,
    SELECTION_RULE,
    TIME_HEAD_PARAMETER_NAMES,
    FrozenFeatureCache,
    _train_cached_epoch,
    cached_quantity_predictions,
    extract_frozen_features,
    fit_time_head_from_cache,
    freeze_time_head_only,
    state_partition_sha256,
    validate_source_checkpoint,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_time_head_refit_v1.json"


def build_small_model(*, hidden_dim: int = 8, max_seq_len: int = 16):
    torch.manual_seed(1234)
    model, metadata = build_count_aware_model(
        "titantpp",
        hidden_dim=hidden_dim,
        train_log_mean=1.2,
        train_log_std=0.7,
        max_seq_len=max_seq_len,
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
        time_scale=3.0,
        time_w_max=10.0 / 3.0,
        time_intercept_limit=300.0,
        time_initial_intercept=0.0,
        time_wd_safety_limit=40.0,
    )
    return model, metadata


def synthetic_cache(model, *, count: int, include_quantity: bool) -> FrozenFeatureCache:
    generator = torch.Generator().manual_seed(91 + count)
    hidden_dim = model.v_t.weight.shape[1]
    time_hidden = torch.randn(count, hidden_dim, generator=generator)
    target_dt = 0.25 + 2.0 * torch.rand(count, generator=generator)
    if not include_quantity:
        return FrozenFeatureCache(time_hidden=time_hidden, target_dt=target_dt)
    quantity_hidden = torch.randn(count, hidden_dim, generator=generator)
    target_quantity = torch.arange(count, dtype=torch.float32).remainder(7)
    with torch.no_grad():
        prediction = model.predict_quantity(quantity_hidden)[1]
    return FrozenFeatureCache(
        time_hidden=time_hidden,
        target_dt=target_dt,
        quantity_hidden=quantity_hidden,
        target_quantity=target_quantity,
        source_quantity_prediction=prediction,
    )


def fit_kwargs(model, train, validation, output_dir, *, planned_epochs: int):
    state_sha256 = canonical_state_dict_sha256(model.state_dict())
    return {
        "model": model,
        "train_cache": train,
        "validation_cache": validation,
        "output_dir": output_dir,
        "dataset": "synthetic",
        "contract_sha256": "c" * 64,
        "source_checkpoint_sha256": "f" * 64,
        "source_state_sha256": state_sha256,
        "calibration_source_revision": "a" * 40,
        "seed": 42,
        "planned_epochs": planned_epochs,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 8,
        "quantity_replay_batch_size": 8,
        "grad_clip": 1.0,
        "min_epochs": planned_epochs + 1,
        "patience": 20,
        "device": "cpu",
    }


def loader_with_target_quantity(target_quantity: float):
    dts = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    quantities = torch.tensor(
        [[0.0, 2.0, 4.0, target_quantity]], dtype=torch.float32
    )
    mask = torch.tensor([[False, True, True, True]])
    marks = torch.zeros_like(dts, dtype=torch.long)
    return [(marks, dts, mask, None, quantities)]


def test_target_quantity_is_hidden_and_encoder_stays_in_eval_mode():
    model, _ = build_small_model()
    model.train()
    first = extract_frozen_features(
        model=model,
        loader=loader_with_target_quantity(5.0),
        device="cpu",
        include_quantity=True,
    )
    second = extract_frozen_features(
        model=model,
        loader=loader_with_target_quantity(5000.0),
        device="cpu",
        include_quantity=True,
    )
    assert not model.training
    assert all(not module.training for module in model.modules())
    assert torch.equal(first.time_hidden, second.time_hidden)
    assert torch.equal(first.quantity_hidden, second.quantity_hidden)
    assert torch.equal(
        first.source_quantity_prediction,
        second.source_quantity_prediction,
    )
    assert not torch.equal(first.target_quantity, second.target_quantity)


def test_one_step_changes_only_the_three_time_head_parameters(tmp_path):
    model, _ = build_small_model()
    train = synthetic_cache(model, count=23, include_quantity=False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    non_time_sha256 = state_partition_sha256(before, time_head=False)
    parameters = freeze_time_head_only(model)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.0)

    telemetry = _train_cached_epoch(
        model=model,
        cache=train,
        optimizer=optimizer,
        parameters=parameters,
        device="cpu",
        batch_size=8,
        grad_clip=1.0,
        seed=42,
        epoch=1,
    )

    assert set(name for name, value in model.named_parameters() if value.requires_grad) == set(
        TIME_HEAD_PARAMETER_NAMES
    )
    assert state_partition_sha256(model.state_dict(), time_head=False) == non_time_sha256
    assert any(
        not torch.equal(before[name], model.state_dict()[name])
        for name in TIME_HEAD_PARAMETER_NAMES
    )
    assert telemetry["pre_clip_gradient_norm_max"] > 0.0
    assert not model.training


def test_epoch_zero_is_a_real_fallback_and_quantity_is_bitwise_identical(tmp_path):
    model, _ = build_small_model()
    train = synthetic_cache(model, count=19, include_quantity=False)
    validation = synthetic_cache(model, count=13, include_quantity=True)
    expected_quantity = validation.source_quantity_prediction.clone()
    kwargs = fit_kwargs(model, train, validation, tmp_path, planned_epochs=0)

    summary = fit_time_head_from_cache(**kwargs)

    assert summary["status"] == "success"
    assert summary["selection"] == SELECTION_RULE
    assert summary["best_epoch"] == 0
    assert summary["history"] == [
        {
            "epoch": 0,
            "train_time_nll": None,
            "val_time_nll": summary["epoch_zero_val_time_nll"],
            "pre_clip_gradient_norm_mean": None,
            "pre_clip_gradient_norm_max": None,
        }
    ]
    assert summary["time_nll_non_worse_than_B"] is True
    assert summary["quantity_prediction_bitwise_identical"] is True
    assert summary["evaluation_scope"] == "validation_only"
    assert summary["held_out_test_evaluated"] is False
    assert torch.equal(
        cached_quantity_predictions(
            model=model,
            cache=validation,
            device="cpu",
            batch_size=8,
        ),
        expected_quantity,
    )
    checkpoint = torch.load(
        tmp_path / "best_validation_time_nll_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["best_epoch"] == 0
    assert checkpoint["held_out_test_evaluated"] is False


def test_quantity_replay_uses_the_original_encoder_batch_shape(tmp_path, monkeypatch):
    import paper.scripts.run_hard_lmm_time_head_refit as refit

    model, _ = build_small_model()
    train = synthetic_cache(model, count=19, include_quantity=False)
    validation = synthetic_cache(model, count=17, include_quantity=True)
    time_batches = []
    quantity_batches = []
    original_time = refit.evaluate_cached_time_nll
    original_quantity = refit.cached_quantity_predictions

    def record_time(**kwargs):
        time_batches.append(kwargs["batch_size"])
        return original_time(**kwargs)

    def record_quantity(**kwargs):
        quantity_batches.append(kwargs["batch_size"])
        return original_quantity(**kwargs)

    monkeypatch.setattr(refit, "evaluate_cached_time_nll", record_time)
    monkeypatch.setattr(refit, "cached_quantity_predictions", record_quantity)
    kwargs = fit_kwargs(model, train, validation, tmp_path, planned_epochs=0)
    kwargs["batch_size"] = 8
    kwargs["quantity_replay_batch_size"] = 5
    refit.fit_time_head_from_cache(**kwargs)

    assert time_batches == [8, 8]
    assert quantity_batches == [5]


def test_resume_replays_the_same_trajectory_as_uninterrupted_training(tmp_path):
    source_model, _ = build_small_model()
    source_state = copy.deepcopy(source_model.state_dict())
    train = synthetic_cache(source_model, count=29, include_quantity=False)
    validation = synthetic_cache(source_model, count=17, include_quantity=True)

    full_model, _ = build_small_model()
    full_model.load_state_dict(source_state, strict=True)
    full = fit_time_head_from_cache(
        **fit_kwargs(full_model, train, validation, tmp_path / "full", planned_epochs=4)
    )

    partial_model, _ = build_small_model()
    partial_model.load_state_dict(source_state, strict=True)
    partial_kwargs = fit_kwargs(
        partial_model,
        train,
        validation,
        tmp_path / "resumed",
        planned_epochs=4,
    )
    paused = fit_time_head_from_cache(**partial_kwargs, run_epoch_limit=2)
    assert paused["status"] == "paused"
    assert paused["completed_epochs"] == 2

    resumed_model, _ = build_small_model()
    resumed_model.load_state_dict(source_state, strict=True)
    resumed = fit_time_head_from_cache(
        **fit_kwargs(
            resumed_model,
            train,
            validation,
            tmp_path / "resumed",
            planned_epochs=4,
        )
    )

    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["selected_state_sha256"] == full["selected_state_sha256"]
    assert resumed["quantity_prediction_bitwise_identical"] is True


def test_resume_rejects_missing_adamw_moment_state(tmp_path):
    source_model, _ = build_small_model()
    source_state = copy.deepcopy(source_model.state_dict())
    train = synthetic_cache(source_model, count=29, include_quantity=False)
    validation = synthetic_cache(source_model, count=17, include_quantity=True)
    output = tmp_path / "resume"
    partial_model, _ = build_small_model()
    partial_model.load_state_dict(source_state, strict=True)
    partial_kwargs = fit_kwargs(
        partial_model,
        train,
        validation,
        output,
        planned_epochs=4,
    )
    paused = fit_time_head_from_cache(**partial_kwargs, run_epoch_limit=1)
    assert paused["status"] == "paused"

    last_path = output / "last_epoch_state.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last["optimizer_state_dict"]["state"] = {}
    torch.save(last, last_path)

    resumed_model, _ = build_small_model()
    resumed_model.load_state_dict(source_state, strict=True)
    with pytest.raises(ValueError, match="does not cover all time-head parameters"):
        fit_time_head_from_cache(
            **fit_kwargs(
                resumed_model,
                train,
                validation,
                output,
                planned_epochs=4,
            )
        )


def test_resume_rejects_adamw_parameter_id_and_moment_shape_drift(tmp_path):
    source_model, _ = build_small_model()
    source_state = copy.deepcopy(source_model.state_dict())
    train = synthetic_cache(source_model, count=29, include_quantity=False)
    validation = synthetic_cache(source_model, count=17, include_quantity=True)
    output = tmp_path / "resume"
    partial_model, _ = build_small_model()
    partial_model.load_state_dict(source_state, strict=True)
    paused = fit_time_head_from_cache(
        **fit_kwargs(partial_model, train, validation, output, planned_epochs=4),
        run_epoch_limit=1,
    )
    assert paused["status"] == "paused"
    last_path = output / "last_epoch_state.pt"
    original = torch.load(last_path, map_location="cpu", weights_only=False)

    wrong_ids = copy.deepcopy(original)
    wrong_ids["optimizer_state_dict"]["state"] = {
        int(parameter_id) + 10: value
        for parameter_id, value in wrong_ids["optimizer_state_dict"]["state"].items()
    }
    torch.save(wrong_ids, last_path)
    resumed_model, _ = build_small_model()
    resumed_model.load_state_dict(source_state, strict=True)
    with pytest.raises(ValueError, match="parameter IDs drift"):
        fit_time_head_from_cache(
            **fit_kwargs(
                resumed_model,
                train,
                validation,
                output,
                planned_epochs=4,
            )
        )

    wrong_order = copy.deepcopy(original)
    wrong_order["optimizer_state_dict"]["param_groups"][0]["params"] = [0, 2, 1]
    torch.save(wrong_order, last_path)
    resumed_model, _ = build_small_model()
    resumed_model.load_state_dict(source_state, strict=True)
    with pytest.raises(ValueError, match="parameter list drift"):
        fit_time_head_from_cache(
            **fit_kwargs(
                resumed_model,
                train,
                validation,
                output,
                planned_epochs=4,
            )
        )

    wrong_shape = copy.deepcopy(original)
    first_parameter_id = wrong_shape["optimizer_state_dict"]["param_groups"][0][
        "params"
    ][0]
    wrong_shape["optimizer_state_dict"]["state"][first_parameter_id][
        "exp_avg"
    ] = torch.zeros(1)
    torch.save(wrong_shape, last_path)
    resumed_model, _ = build_small_model()
    resumed_model.load_state_dict(source_state, strict=True)
    with pytest.raises(ValueError, match="shape or dtype drift"):
        fit_time_head_from_cache(
            **fit_kwargs(
                resumed_model,
                train,
                validation,
                output,
                planned_epochs=4,
            )
        )

    wrong_learning_rate = copy.deepcopy(original)
    wrong_learning_rate["optimizer_state_dict"]["param_groups"][0]["lr"] = 1.0
    torch.save(wrong_learning_rate, last_path)
    resumed_model, _ = build_small_model()
    resumed_model.load_state_dict(source_state, strict=True)
    with pytest.raises(ValueError, match="parameter-group drift: lr"):
        fit_time_head_from_cache(
            **fit_kwargs(
                resumed_model,
                train,
                validation,
                output,
                planned_epochs=4,
            )
        )

    wrong_step = copy.deepcopy(original)
    for parameter_state in wrong_step["optimizer_state_dict"]["state"].values():
        parameter_state["step"] += 1.0
    torch.save(wrong_step, last_path)
    resumed_model, _ = build_small_model()
    resumed_model.load_state_dict(source_state, strict=True)
    with pytest.raises(ValueError, match="AdamW step drift"):
        fit_time_head_from_cache(
            **fit_kwargs(
                resumed_model,
                train,
                validation,
                output,
                planned_epochs=4,
            )
        )


def test_cached_summary_rejects_selected_non_time_state_drift(tmp_path):
    model, _ = build_small_model()
    train = synthetic_cache(model, count=9, include_quantity=False)
    validation = synthetic_cache(model, count=7, include_quantity=True)
    kwargs = fit_kwargs(model, train, validation, tmp_path, planned_epochs=0)
    fit_time_head_from_cache(**kwargs)
    checkpoint_path = tmp_path / "best_validation_time_nll_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frozen_name = next(
        name for name in checkpoint["model_state_dict"] if name not in TIME_HEAD_PARAMETER_NAMES
    )
    checkpoint["model_state_dict"][frozen_name].view(-1)[0] += 1.0
    checkpoint["model_state_sha256"] = canonical_state_dict_sha256(
        checkpoint["model_state_dict"]
    )
    torch.save(checkpoint, checkpoint_path)

    fresh_model, _ = build_small_model()
    with pytest.raises(ValueError, match="non-time state drift"):
        fit_time_head_from_cache(
            **fit_kwargs(fresh_model, train, validation, tmp_path, planned_epochs=0)
        )


def test_strict_source_checkpoint_contract_rejects_selector_drift():
    model, encoder = build_count_aware_model(
        "titantpp",
        hidden_dim=64,
        train_log_mean=1.2,
        train_log_std=0.7,
        max_seq_len=64,
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
        time_scale=3.0,
        time_w_max=10.0 / 3.0,
        time_intercept_limit=300.0,
        time_initial_intercept=0.0,
        time_wd_safety_limit=40.0,
    )
    state = model.state_dict()
    digest = canonical_state_dict_sha256(state)
    dataset_spec = {
        "max_sequence_length": 64,
        "B_checkpoint_state_sha256": digest,
        "B_metrics": {"raw_rmse": 1.0},
    }
    payload = {
        "backbone": "titantpp",
        "variant": "count_only_log_regression",
        "seed": 42,
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection": "best_validation_raw_quantity_rmse",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "selected_metric_value": 1.0,
        "source_revision": "f75243473adc25d622319dbca9bda7e076d8240f",
        "source_revision_history": [
            "f75243473adc25d622319dbca9bda7e076d8240f"
        ],
        "model_state_dict": state,
        "model_state_sha256": digest,
        "encoder_config": encoder,
        "interface_meta": {
            "mode": "mark_free_count_aware_log_regression",
            "target_quantity_masked_from_history": True,
        },
    }
    validate_source_checkpoint(payload, dataset_spec=dataset_spec)
    payload["checkpoint_monitor"] = "validation_joint_objective"
    with pytest.raises(ValueError, match="checkpoint_monitor mismatch"):
        validate_source_checkpoint(payload, dataset_spec=dataset_spec)


def test_contract_pins_all_three_B_checkpoints_and_common_refit_policy():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["scope"]["held_out_test"] is False
    assert contract["parameter_boundary"]["trainable_parameter_names"] == list(
        TIME_HEAD_PARAMETER_NAMES
    )
    assert contract["parameter_boundary"]["expected_trainable_parameter_count"] == 66
    assert contract["optimization"]["shared_across_all_datasets"] is True
    assert contract["optimization"]["dataset_specific_hyperparameters"] is False
    assert contract["checkpoint_selection"]["fallback"] == "epoch 0"
    rows = contract["datasets"]
    assert {row["dataset"] for row in rows} == {
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    }
    assert all(len(row["B_checkpoint_file_sha256"]) == 64 for row in rows)
    assert all(len(row["B_checkpoint_state_sha256"]) == 64 for row in rows)
    assert all(
        row["time_nll_target_max"] == pytest.approx(row["A_time_nll"] + 0.01)
        for row in rows
    )
