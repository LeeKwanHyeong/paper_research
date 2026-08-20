from __future__ import annotations

import pytest

from models.TPPs.CountAwareTPP import (
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    FROZEN_TAIL_LAMBDA,
    MODEL_ROLE_T0_COMMON_CONTROL,
    MODEL_ROLE_T1_BACKBONE_COMPARISON,
    MODEL_ROLE_T1_INCUMBENT,
    MODEL_ROLE_TIME_HEAD_DIAGNOSTIC,
    TAIL_SHARED_VARIANT,
    VARIANT,
    validate_model_role_contract,
)


def validate(**overrides) -> None:
    values = {
        "model_role": MODEL_ROLE_T0_COMMON_CONTROL,
        "backbones": ("rmtpp", "thp", "nhp", "sahp"),
        "quantity_variants": (VARIANT,),
        "time_head_mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
        "lambda_tail": 0.0,
    }
    values.update(overrides)
    validate_model_role_contract(**values)


def test_t0_common_control_accepts_only_common_head_and_loss() -> None:
    validate()
    with pytest.raises(ValueError, match="direct log-MSE"):
        validate(quantity_variants=(TAIL_SHARED_VARIANT,))
    with pytest.raises(ValueError, match="legacy_clamped_rmtpp"):
        validate(time_head_mode=TIME_HEAD_MODE_LOGNORMAL_DURATION)


def test_t1_incumbent_freezes_tail_loss_and_legacy_time_head() -> None:
    validate(
        model_role=MODEL_ROLE_T1_INCUMBENT,
        backbones=("titantpp",),
        quantity_variants=(TAIL_SHARED_VARIANT,),
        lambda_tail=FROZEN_TAIL_LAMBDA,
    )
    with pytest.raises(ValueError, match="lambda_tail"):
        validate(
            model_role=MODEL_ROLE_T1_INCUMBENT,
            backbones=("titantpp",),
            quantity_variants=(TAIL_SHARED_VARIANT,),
            lambda_tail=0.1,
        )


def test_t1_backbone_comparison_requires_fresh_incumbent() -> None:
    validate(
        model_role=MODEL_ROLE_T1_BACKBONE_COMPARISON,
        backbones=("titantpp", "titantpp_gated_soft_memory"),
        quantity_variants=(TAIL_SHARED_VARIANT,),
        lambda_tail=FROZEN_TAIL_LAMBDA,
    )
    with pytest.raises(ValueError, match="fresh titantpp"):
        validate(
            model_role=MODEL_ROLE_T1_BACKBONE_COMPARISON,
            backbones=("titantpp_gated_soft_memory",),
            quantity_variants=(TAIL_SHARED_VARIANT,),
            lambda_tail=FROZEN_TAIL_LAMBDA,
        )


def test_time_head_diagnostic_is_not_an_incumbent_role() -> None:
    validate(
        model_role=MODEL_ROLE_TIME_HEAD_DIAGNOSTIC,
        backbones=("titantpp",),
        quantity_variants=(TAIL_SHARED_VARIANT,),
        time_head_mode=TIME_HEAD_MODE_LOGNORMAL_DURATION,
        lambda_tail=FROZEN_TAIL_LAMBDA,
    )
