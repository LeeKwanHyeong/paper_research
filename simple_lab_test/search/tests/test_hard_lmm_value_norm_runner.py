from __future__ import annotations

import sys

import pytest

from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TIME_HEAD_MODE_SCALED_EXACT,
)
from models.TPPs.CountAwareTitanBoundedQK import BOUNDED_QK_ROLE
from models.TPPs.CountAwareTitanValueNorm import (
    VALUE_NORM_BACKBONE,
    VALUE_NORM_ROLE,
)
from paper.scripts import run_count_aware_tpp_backbone_control as runner
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONE_LABELS,
    MODEL_ROLES,
    MODEL_ROLE_EXPERIMENTAL,
    MODEL_ROLE_VALUE_NORM,
    SUPPORTED_BACKBONES,
    TITAN_MEMORY_BACKBONES,
    validate_model_role_contract,
)


def _validate(
    *,
    role: str = VALUE_NORM_ROLE,
    backbones: tuple[str, ...] = (VALUE_NORM_BACKBONE,),
    variants: tuple[str, ...] = (LOG_MSE_VARIANT,),
    time_head: str = TIME_HEAD_MODE_LEGACY_CLAMPED,
    lambda_tail: float = 0.0,
) -> None:
    validate_model_role_contract(
        model_role=role,
        backbones=backbones,
        quantity_variants=variants,
        time_head_mode=time_head,
        lambda_tail=lambda_tail,
    )


def test_value_norm_route_is_supported_named_and_dedicated() -> None:
    assert VALUE_NORM_BACKBONE in TITAN_MEMORY_BACKBONES
    assert VALUE_NORM_BACKBONE in SUPPORTED_BACKBONES
    assert VALUE_NORM_ROLE in MODEL_ROLES
    assert MODEL_ROLE_VALUE_NORM == VALUE_NORM_ROLE
    assert BACKBONE_LABELS[VALUE_NORM_BACKBONE] == (
        "Hard-LMM Prototype Value-Norm Consistency"
    )
    _validate()


@pytest.mark.parametrize(
    "changes",
    [
        {"role": MODEL_ROLE_EXPERIMENTAL},
        {"role": BOUNDED_QK_ROLE},
        {"backbones": (VALUE_NORM_BACKBONE, "titantpp")},
        {"backbones": ("titantpp",)},
        {"variants": ("count_only_lognormal",)},
        {"time_head": TIME_HEAD_MODE_SCALED_EXACT},
        {"lambda_tail": 0.1},
    ],
)
def test_value_norm_route_rejects_role_structure_or_objective_drift(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _validate(**changes)


def test_training_cli_parser_accepts_exact_value_norm_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--data",
            str(tmp_path / "data.parquet"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--source-revision",
            "1" * 40,
            "--execution-role",
            "local_contract_test",
            "--device",
            "cpu",
            "--backbones",
            VALUE_NORM_BACKBONE,
            "--model-role",
            VALUE_NORM_ROLE,
            "--quantity-variants",
            LOG_MSE_VARIANT,
            "--time-head-mode",
            TIME_HEAD_MODE_LEGACY_CLAMPED,
            "--lambda-tail",
            "0",
            "--checkpoint-monitor",
            "validation_raw_quantity_rmse",
            "--quantile-adaptive-strength",
            "0",
            "--time-intercept-limit",
            "300",
            "--allow-partial-contract",
        ],
    )

    args = runner.parse_args()

    assert args.backbones == VALUE_NORM_BACKBONE
    assert args.model_role == VALUE_NORM_ROLE
    assert args.quantity_variants == LOG_MSE_VARIANT
    assert args.time_head_mode == TIME_HEAD_MODE_LEGACY_CLAMPED
    assert args.lambda_tail == 0.0
    assert args.checkpoint_monitor == "validation_raw_quantity_rmse"
    assert args.quantile_adaptive_strength == 0.0
    assert args.time_intercept_limit == 300.0


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("checkpoint_monitor", "validation_joint_objective", "raw-RMSE checkpoint"),
        ("quantile_adaptive_strength", 0.1, "adaptive strength=0"),
        ("time_intercept_limit", 30.0, "intercept cap=300"),
    ],
)
def test_training_run_rejects_value_norm_alignment_drift_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    attribute: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--data",
            str(tmp_path / "must_not_be_read.parquet"),
            "--split-manifest",
            str(tmp_path / "must_not_be_read.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--source-revision",
            "1" * 40,
            "--execution-role",
            "local_contract_test",
            "--device",
            "cpu",
            "--backbones",
            VALUE_NORM_BACKBONE,
            "--model-role",
            VALUE_NORM_ROLE,
            "--quantity-variants",
            LOG_MSE_VARIANT,
            "--time-head-mode",
            TIME_HEAD_MODE_LEGACY_CLAMPED,
            "--lambda-tail",
            "0",
            "--checkpoint-monitor",
            "validation_raw_quantity_rmse",
            "--quantile-adaptive-strength",
            "0",
            "--time-intercept-limit",
            "300",
            "--allow-partial-contract",
        ],
    )
    args = runner.parse_args()
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=message):
        runner.run(args)
