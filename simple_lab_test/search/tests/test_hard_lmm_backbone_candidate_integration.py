from __future__ import annotations

import copy

import pytest
import torch

from models.TPPs.CountAwareFactory import (
    build_count_aware_model,
    validate_checkpoint_route,
)
from models.TPPs.CountAwareTitanInterLayerMemory import (
    INTERLAYER_MEMORY_BACKBONE,
    INTERLAYER_MEMORY_ROLE,
    CountAwareTitanInterLayerMemoryTPP,
)
from models.TPPs.CountAwareTitanMemoryFiLM import (
    MEMORY_FILM_BACKBONE,
    MEMORY_FILM_ROLE,
    CountAwareTitanMemoryFiLM,
)
from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_BACKBONE, CAUSAL_QKV_ROLE, CountAwareTitanCausalQKVTPP,
)
from models.TPPs.CountAwareTitanBoundedQK import (
    BOUNDED_QK_BACKBONE, BOUNDED_QK_ROLE, CountAwareTitanBoundedQKTPP,
)
from models.TPPs.CountAwareTitanLevelHistoryQKV import (
    LEVEL_HISTORY_QKV_BACKBONE,
    LEVEL_HISTORY_QKV_ROLE,
    CountAwareTitanLevelHistoryQKVTPP,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLE_INTERLAYER_MEMORY,
    MODEL_ROLE_MEMORY_FILM,
    SUPPORTED_BACKBONES,
    validate_model_role_contract,
)


def _build(backbone: str):
    return build_count_aware_model(
        backbone,
        hidden_dim=64,
        train_log_mean=2.0,
        max_seq_len=16,
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
        time_intercept_limit=300.0,
    )


@pytest.mark.parametrize(
    ("backbone", "model_type", "role", "added"),
    [
        (INTERLAYER_MEMORY_BACKBONE, CountAwareTitanInterLayerMemoryTPP, INTERLAYER_MEMORY_ROLE, 1),
        (MEMORY_FILM_BACKBONE, CountAwareTitanMemoryFiLM, MEMORY_FILM_ROLE, 128),
        (CAUSAL_QKV_BACKBONE, CountAwareTitanCausalQKVTPP, CAUSAL_QKV_ROLE, 576),
        (BOUNDED_QK_BACKBONE, CountAwareTitanBoundedQKTPP, BOUNDED_QK_ROLE, 576),
        (
            LEVEL_HISTORY_QKV_BACKBONE,
            CountAwareTitanLevelHistoryQKVTPP,
            LEVEL_HISTORY_QKV_ROLE,
            384,
        ),
    ],
)
def test_factory_and_role_bind_each_candidate(backbone, model_type, role, added) -> None:
    model, metadata = _build(backbone)
    assert isinstance(model, model_type)
    assert backbone in SUPPORTED_BACKBONES
    assert metadata["model_role"] == role
    assert metadata["additional_parameter_count"] == added
    assert metadata["time_head"]["mode"] == "legacy_clamped_rmtpp"
    validate_model_role_contract(
        model_role=role,
        backbones=(backbone,),
        quantity_variants=("count_only_log_regression",),
        time_head_mode="legacy_clamped_rmtpp",
        lambda_tail=0.0,
    )


@pytest.mark.parametrize(
    ("role", "backbone"),
    [
        (MODEL_ROLE_INTERLAYER_MEMORY, INTERLAYER_MEMORY_BACKBONE),
        (MODEL_ROLE_MEMORY_FILM, MEMORY_FILM_BACKBONE),
        (CAUSAL_QKV_ROLE, CAUSAL_QKV_BACKBONE),
        (BOUNDED_QK_ROLE, BOUNDED_QK_BACKBONE),
        (LEVEL_HISTORY_QKV_ROLE, LEVEL_HISTORY_QKV_BACKBONE),
    ],
)
def test_candidate_role_rejects_selector_or_route_drift(role, backbone) -> None:
    with pytest.raises(ValueError):
        validate_model_role_contract(
            model_role=role,
            backbones=("titantpp",),
            quantity_variants=("count_only_log_regression",),
            time_head_mode="legacy_clamped_rmtpp",
            lambda_tail=0.0,
        )
    with pytest.raises(ValueError):
        validate_model_role_contract(
            model_role=role,
            backbones=(backbone,),
            quantity_variants=("count_only_log_regression",),
            time_head_mode="legacy_clamped_rmtpp",
            lambda_tail=0.1,
        )


def _artifact(backbone: str) -> dict:
    model, metadata = _build(backbone)
    return {
        "backbone": backbone,
        "variant": "count_only_log_regression",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "encoder_config": metadata,
        "model_state_dict": model.state_dict(),
    }


@pytest.mark.parametrize("backbone", [
    INTERLAYER_MEMORY_BACKBONE,
    MEMORY_FILM_BACKBONE,
    CAUSAL_QKV_BACKBONE,
    BOUNDED_QK_BACKBONE,
    LEVEL_HISTORY_QKV_BACKBONE,
])
def test_checkpoint_route_roundtrip_and_summary(backbone: str) -> None:
    payload = _artifact(backbone)
    validate_checkpoint_route(payload, backbone)
    summary = {key: value for key, value in payload.items() if key != "model_state_dict"}
    summary.update(checkpoint_state_sha256="0" * 64, best_epoch=1)
    validate_checkpoint_route(summary, backbone)


def test_candidates_cannot_be_relabelled_as_each_other_or_b() -> None:
    for payload, other in (
        (_artifact(INTERLAYER_MEMORY_BACKBONE), MEMORY_FILM_BACKBONE),
        (_artifact(MEMORY_FILM_BACKBONE), INTERLAYER_MEMORY_BACKBONE),
        (_artifact(CAUSAL_QKV_BACKBONE), INTERLAYER_MEMORY_BACKBONE),
        (_artifact(CAUSAL_QKV_BACKBONE), MEMORY_FILM_BACKBONE),
        (_artifact(CAUSAL_QKV_BACKBONE), BOUNDED_QK_BACKBONE),
        (_artifact(BOUNDED_QK_BACKBONE), CAUSAL_QKV_BACKBONE),
        (_artifact(LEVEL_HISTORY_QKV_BACKBONE), BOUNDED_QK_BACKBONE),
        (_artifact(BOUNDED_QK_BACKBONE), LEVEL_HISTORY_QKV_BACKBONE),
    ):
        with pytest.raises(ValueError):
            validate_checkpoint_route(payload, other)
        with pytest.raises(ValueError):
            validate_checkpoint_route(payload, "titantpp")


def test_candidate_state_marker_cannot_be_removed() -> None:
    payload = _artifact(MEMORY_FILM_BACKBONE)
    corrupted = copy.deepcopy(payload)
    corrupted["model_state_dict"].pop("film_shift_gain")
    with pytest.raises(ValueError):
        validate_checkpoint_route(corrupted, MEMORY_FILM_BACKBONE)
    assert torch.isfinite(payload["model_state_dict"]["film_shift_gain"]).all()


def test_bounded_checkpoint_cannot_be_relabelled_by_removing_metadata() -> None:
    payload = _artifact(BOUNDED_QK_BACKBONE)
    payload["backbone"] = "titantpp"
    payload["encoder_config"] = {"d_model": 64}
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, "titantpp")


@pytest.mark.parametrize("backbone,role", [
    (CAUSAL_QKV_BACKBONE, CAUSAL_QKV_ROLE),
    (BOUNDED_QK_BACKBONE, BOUNDED_QK_ROLE),
    (LEVEL_HISTORY_QKV_BACKBONE, LEVEL_HISTORY_QKV_ROLE),
])
def test_frozen_duration_rebuild_does_not_allow_joint_head_change(backbone, role) -> None:
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=64, train_log_mean=2.0, max_seq_len=16,
        quantity_variant="count_only_log_regression", lambda_tail=0.0,
        time_head_mode="heteroscedastic_lognormal_duration",
        time_initial_location=0.0, time_initial_scale=1.0,
    )
    assert model is not None
    assert metadata["time_head"]["mode"] == "heteroscedastic_lognormal_duration"
    with pytest.raises(ValueError):
        validate_model_role_contract(
            model_role=role, backbones=(backbone,),
            quantity_variants=("count_only_log_regression",),
            time_head_mode="heteroscedastic_lognormal_duration", lambda_tail=0.0,
        )
