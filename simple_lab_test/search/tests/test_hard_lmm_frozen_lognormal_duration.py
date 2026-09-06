from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    CANDIDATE_TIME_HEAD_MODE,
    CONTRACT_ID,
    LAST_CHECKPOINT_NAME,
    SELECTED_CHECKPOINT_NAME,
    TIME_HEAD_PARAMETER_NAMES,
    FrozenFeatureCache,
    _train_cached_epoch,
    build_frozen_lognormal_candidate,
    censor_mask,
    censor_mask_sha256,
    fit_time_head_from_cache,
    freeze_time_head_only,
    load_admitted_frame,
    proper_time_log_likelihood,
    resolve_censor_threshold,
    state_partition_sha256,
    target_dt_sha256,
    validate_contract,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
)


CONTRACT_PATH = (
    ROOT / "paper/contracts/hard_lmm_frozen_lognormal_duration_v1.json"
)


def build_legacy_B_source():
    torch.manual_seed(7401)
    model, encoder = build_count_aware_model(
        "titantpp",
        hidden_dim=64,
        train_log_mean=2.0,
        train_log_std=0.7,
        max_seq_len=16,
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
        time_scale=3.0,
        time_w_max=10.0 / 3.0,
        time_intercept_limit=300.0,
        time_initial_intercept=0.0,
        time_wd_safety_limit=40.0,
    )
    model.eval()
    state = copy.deepcopy(model.state_dict())
    payload = {
        "model_state_dict": state,
        "model_state_sha256": canonical_state_dict_sha256(state),
        "encoder_config": encoder,
        "interface_meta": {
            "mode": "mark_free_count_aware_log_regression",
            "target_quantity_masked_from_history": True,
            "train_target_mean": 2.0,
            "train_target_std": 0.7,
        },
    }
    return model, payload


def train_time_statistics() -> dict[str, float | int | str]:
    return {
        "statistics_source_split": "train",
        "target_count": 17,
        "target_dt_min": 1.0,
        "target_dt_mean": 4.0,
        "target_dt_p50": 3.0,
        "target_dt_p99": 9.0,
        "target_dt_max": 10.0,
        "time_scale": 3.0,
        "wd_safety_limit": 40.0,
        "time_w_max": 12.0,
        "time_initial_intercept": math.log(0.75),
        "target_log_scaled_mean": 0.25,
        "target_log_scaled_std": 0.8,
    }


def build_candidate():
    source, payload = build_legacy_B_source()
    candidate, encoder, metadata = build_frozen_lognormal_candidate(
        payload,
        train_time_statistics=train_time_statistics(),
        time_sigma_floor=0.001,
    )
    return source, payload, candidate, encoder, metadata


def synthetic_cache(
    source,
    *,
    count: int,
    include_quantity: bool,
) -> FrozenFeatureCache:
    generator = torch.Generator().manual_seed(991 + count)
    time_hidden = torch.randn(count, 64, generator=generator)
    target_dt = torch.exp(
        0.25 + 0.35 * time_hidden[:, 0]
    ).to(torch.float32)
    if not include_quantity:
        return FrozenFeatureCache(
            time_hidden=time_hidden,
            target_dt=target_dt,
        )
    quantity_hidden = torch.randn(count, 64, generator=generator)
    target_quantity = torch.arange(count, dtype=torch.float32).remainder(9)
    with torch.no_grad():
        source_prediction = source.predict_quantity(quantity_hidden)[1]
    return FrozenFeatureCache(
        time_hidden=time_hidden,
        target_dt=target_dt,
        quantity_hidden=quantity_hidden,
        target_quantity=target_quantity,
        source_quantity_prediction=source_prediction,
    )


def fit_kwargs(
    *,
    source_payload,
    candidate,
    metadata,
    train,
    validation,
    output_dir: Path,
    planned_epochs: int,
):
    source_state = source_payload["model_state_dict"]
    return {
        "model": candidate,
        "train_cache": train,
        "validation_cache": validation,
        "output_dir": output_dir,
        "dataset": "synthetic",
        "censor_threshold": None,
        "contract_sha256": "c" * 64,
        "source_checkpoint_sha256": "f" * 64,
        "source_state_sha256": source_payload["model_state_sha256"],
        "source_non_time_state_sha256": state_partition_sha256(
            source_state, time_head=False
        ),
        "candidate_initial_state_sha256": metadata[
            "candidate_initial_state_sha256"
        ],
        "train_time_statistics": train_time_statistics(),
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
        "candidate_metadata": metadata,
        "source_metadata": {"calibration_source_revision": "a" * 40},
    }


def test_candidate_copies_state_and_replays_quantity_bitwise() -> None:
    source, payload, candidate, encoder, metadata = build_candidate()
    source_state = payload["model_state_dict"]
    candidate_state = candidate.state_dict()

    assert candidate.time_head_mode == CANDIDATE_TIME_HEAD_MODE
    assert set(candidate_state) == set(source_state) | {
        "time_scale_weight.weight"
    }
    for name, value in source_state.items():
        if name not in {"v_t.weight", "b_t", "w_raw"}:
            assert torch.equal(candidate_state[name], value), name
    assert metadata["source_non_time_state_sha256"] == state_partition_sha256(
        candidate_state, time_head=False
    )
    assert encoder == metadata["encoder_config"]

    dts = torch.tensor(
        [[1.0, 2.0, 1.0, 3.0], [1.0, 1.0, 4.0, 2.0]]
    )
    quantities = torch.tensor(
        [[2.0, 3.0, 1.0, 0.0], [1.0, 4.0, 2.0, 0.0]]
    )
    mask = torch.ones_like(dts, dtype=torch.bool)
    memory_write_mask = mask.clone()
    memory_write_mask[:, -1] = False
    with torch.no_grad():
        source_states = source.encode_task_states(
            dts,
            quantities,
            mask,
            memory_write_mask=memory_write_mask,
        )
        candidate_states = candidate.encode_task_states(
            dts,
            quantities,
            mask,
            memory_write_mask=memory_write_mask,
        )
    for source_hidden, candidate_hidden in zip(
        source_states, candidate_states
    ):
        assert torch.equal(source_hidden, candidate_hidden)
        assert torch.equal(
            source.predict_quantity(source_hidden)[1],
            candidate.predict_quantity(candidate_hidden)[1],
        )


def test_initialization_and_130_parameter_gradient_boundary() -> None:
    source, _, candidate, _, _ = build_candidate()
    statistics = train_time_statistics()
    assert torch.count_nonzero(candidate.v_t.weight) == 0
    assert torch.count_nonzero(candidate.time_scale_weight.weight) == 0
    assert candidate.b_t.item() == pytest.approx(
        statistics["target_log_scaled_mean"], abs=1e-7
    )
    assert torch.equal(
        candidate.positive_time_sigma(torch.zeros(3, 64)),
        torch.full((3,), statistics["target_log_scaled_std"]),
    )

    before = copy.deepcopy(candidate.state_dict())
    non_time_sha256 = state_partition_sha256(before, time_head=False)
    parameters = freeze_time_head_only(candidate)
    assert sum(parameter.numel() for parameter in parameters) == 130
    assert {
        name
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    } == set(TIME_HEAD_PARAMETER_NAMES)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.0)
    telemetry = _train_cached_epoch(
        model=candidate,
        cache=synthetic_cache(source, count=23, include_quantity=False),
        optimizer=optimizer,
        parameters=parameters,
        device="cpu",
        batch_size=8,
        grad_clip=1.0,
        seed=42,
        epoch=1,
        censor_threshold=None,
    )
    assert state_partition_sha256(
        candidate.state_dict(), time_head=False
    ) == non_time_sha256
    assert telemetry["pre_clip_gradient_norm_max"] > 0.0
    assert all(
        candidate.get_parameter(name).grad is not None
        for name in TIME_HEAD_PARAMETER_NAMES
    )


def test_censor_aware_likelihood_uses_survival_only_at_instacart_cap() -> None:
    _, _, candidate, _, _ = build_candidate()
    hidden = torch.randn(4, 64, dtype=torch.float32)
    target_dt = torch.tensor([1.0, 30.0, 7.0, 30.0])
    mask = censor_mask(target_dt, threshold=30.0)
    actual = proper_time_log_likelihood(
        candidate,
        hidden,
        target_dt,
        is_right_censored=mask,
    )
    density = candidate.log_f_dt(hidden, target_dt)
    survival = candidate.log_survival_dt(hidden, target_dt)
    assert actual.dtype == torch.float64
    assert torch.equal(actual[~mask], density[~mask])
    assert torch.equal(actual[mask], survival[mask])
    assert not torch.equal(actual[mask], density[mask])
    assert resolve_censor_threshold(
        "insta_market_basket", {"right_censor_threshold": 30.0}
    ) == 30.0
    assert resolve_censor_threshold(
        "yellow_trip_hourly", {"right_censor_threshold": None}
    ) is None
    with pytest.raises(ValueError, match="not permitted"):
        resolve_censor_threshold(
            "yellow_trip_hourly", {"right_censor_threshold": 30.0}
        )


def test_target_and_censor_hashes_follow_the_pinned_byte_contract() -> None:
    target = torch.tensor([1.0, 7.5, 30.0], dtype=torch.float32)
    mask = torch.tensor([False, False, True])
    expected_target = hashlib.sha256(
        b"frozen_lognormal_target_dt_v1\0"
        + np.asarray(target.numpy(), dtype="<f8").reshape(-1).tobytes()
    ).hexdigest()
    expected_mask = hashlib.sha256(
        b"frozen_lognormal_censor_mask_v1\0"
        + np.asarray(mask.numpy(), dtype=np.uint8).reshape(-1).tobytes()
    ).hexdigest()
    assert target_dt_sha256(target) == expected_target
    assert censor_mask_sha256(mask) == expected_mask


def test_resume_matches_uninterrupted_training_exactly(tmp_path: Path) -> None:
    source, payload, candidate, _, metadata = build_candidate()
    train = synthetic_cache(source, count=29, include_quantity=False)
    validation = synthetic_cache(source, count=17, include_quantity=True)
    full = fit_time_head_from_cache(
        **fit_kwargs(
            source_payload=payload,
            candidate=candidate,
            metadata=metadata,
            train=train,
            validation=validation,
            output_dir=tmp_path / "full",
            planned_epochs=4,
        )
    )

    _, _, partial_candidate, _, partial_metadata = build_candidate()
    resumed_dir = tmp_path / "resumed"
    partial = fit_time_head_from_cache(
        **fit_kwargs(
            source_payload=payload,
            candidate=partial_candidate,
            metadata=partial_metadata,
            train=train,
            validation=validation,
            output_dir=resumed_dir,
            planned_epochs=4,
        ),
        run_epoch_limit=2,
    )
    assert partial["status"] == "paused"
    assert partial["completed_epochs"] == 2
    assert (resumed_dir / LAST_CHECKPOINT_NAME).is_file()

    _, _, resumed_candidate, _, resumed_metadata = build_candidate()
    resumed = fit_time_head_from_cache(
        **fit_kwargs(
            source_payload=payload,
            candidate=resumed_candidate,
            metadata=resumed_metadata,
            train=train,
            validation=validation,
            output_dir=resumed_dir,
            planned_epochs=4,
        )
    )
    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["selected_state_sha256"] == full[
        "selected_state_sha256"
    ]
    assert resumed["quantity_prediction_bitwise_identical"] is True
    assert (resumed_dir / SELECTED_CHECKPOINT_NAME).is_file()
    selected = torch.load(
        resumed_dir / SELECTED_CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=False,
    )
    assert selected["held_out_test_evaluated"] is False
    assert selected["legacy_nll_compared"] is False


def test_held_out_rows_are_excluded_before_admitted_frame(tmp_path: Path) -> None:
    path = tmp_path / "splits.parquet"
    pl.DataFrame(
        {
            "oper_part_no": ["a", "a", "a", "a"],
            "seq": [1, 2, 3, 4],
            "chronological_split": [
                "train",
                "validation",
                "test",
                "heldout_test",
            ],
            "sentinel": [10.0, 20.0, 1.0e20, -1.0e20],
        }
    ).write_parquet(path)
    admitted = load_admitted_frame(path)
    assert admitted["chronological_split"].to_list() == [
        "train",
        "validation",
    ]
    assert admitted["sentinel"].to_list() == [10.0, 20.0]


def test_contract_pins_common_frozen_B_refit() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    datasets = validate_contract(contract)
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["time_head"]["calculation_dtype"] == "float64"
    assert contract["scope"]["held_out_test"] is False
    assert contract["time_head"]["trainable_parameter_names"] == list(
        TIME_HEAD_PARAMETER_NAMES
    )
    assert contract["time_head"]["expected_trainable_parameter_count"] == 130
    assert contract["acceptance"]["legacy_comparison_prohibited"] is True
    assert set(datasets) == {
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    }
    assert datasets["insta_market_basket"]["right_censor_threshold"] == 30.0
    assert all(
        row["right_censor_threshold"] is None
        for name, row in datasets.items()
        if name != "insta_market_basket"
    )
