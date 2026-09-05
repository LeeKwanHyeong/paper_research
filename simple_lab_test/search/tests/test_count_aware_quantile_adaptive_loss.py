from __future__ import annotations

import math

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    QUANTILE_ADAPTIVE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONES,
    CHECKPOINT_HISTORY_RAW_QUANTITY_RMSE,
    CHECKPOINT_MONITOR_JOINT,
    CHECKPOINT_MONITOR_RAW_QUANTITY_RMSE,
    MODEL_ROLE_QUANTILE_CHECKPOINT_ALIGNMENT,
    QUANTITY_VARIANT_ALIASES,
    QUANTILE_ADAPTIVE_QUANTILES,
    QUANTILE_ADAPTIVE_RAW_WEIGHTS,
    QUANTILE_ADAPTIVE_STRENGTH,
    validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


BOUNDARIES = (10.0, 20.0, 30.0, 40.0)
NORMALIZED_WEIGHTS = (0.5, 0.75, 1.0, 1.5, 2.0)


def build_model(
    backbone: str,
    variant: str,
    *,
    strength: float = 0.0,
    boundaries: tuple[float, ...] | None = BOUNDARIES,
    weights: tuple[float, ...] | None = NORMALIZED_WEIGHTS,
):
    return build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
        quantity_variant=variant,
        quantile_adaptive_strength=strength,
        quantile_adaptive_boundaries=boundaries,
        quantile_adaptive_weights=weights,
    )[0]


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor(
        [[0.0, 1.0, 2.0, 1.0], [0.0, 2.0, 1.0, 3.0]],
        dtype=torch.float32,
    )
    mask = torch.ones_like(dts, dtype=torch.bool)
    quantities = torch.tensor(
        [[2.0, 3.0, 4.0, 50.0], [1.0, 5.0, 6.0, 100.0]],
        dtype=torch.float32,
    )
    return dts, mask, quantities


def assert_matching_gradients(
    control: torch.nn.Module,
    candidate: torch.nn.Module,
) -> None:
    control_parameters = dict(control.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    assert control_parameters.keys() == candidate_parameters.keys()
    for name, control_parameter in control_parameters.items():
        candidate_parameter = candidate_parameters[name]
        if control_parameter.grad is None:
            assert candidate_parameter.grad is None, name
        else:
            assert candidate_parameter.grad is not None, name
            assert torch.equal(
                control_parameter.grad,
                candidate_parameter.grad,
            ), name


def test_frozen_identifiers_aliases_role_and_monitors() -> None:
    assert QUANTILE_ADAPTIVE_VARIANT == (
        "count_only_quantile_adaptive_log_regression"
    )
    assert QUANTITY_VARIANT_ALIASES["quantile_adaptive"] == (
        QUANTILE_ADAPTIVE_VARIANT
    )
    assert QUANTILE_ADAPTIVE_QUANTILES == (0.5, 0.9, 0.95, 0.99)
    assert QUANTILE_ADAPTIVE_RAW_WEIGHTS == (1.0, 1.0, 1.5, 2.0, 3.0)
    assert QUANTILE_ADAPTIVE_STRENGTH == 1.0
    assert CHECKPOINT_MONITOR_JOINT == "validation_joint_objective"
    assert CHECKPOINT_MONITOR_RAW_QUANTITY_RMSE == (
        "validation_raw_quantity_rmse"
    )
    assert CHECKPOINT_HISTORY_RAW_QUANTITY_RMSE == "val_qty_rmse"

    validate_model_role_contract(
        model_role=MODEL_ROLE_QUANTILE_CHECKPOINT_ALIGNMENT,
        backbones=("titantpp",),
        quantity_variants=(LOG_MSE_VARIANT, QUANTILE_ADAPTIVE_VARIANT),
        time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED,
        lambda_tail=0.0,
    )
    with pytest.raises(ValueError, match="ordered unweighted"):
        validate_model_role_contract(
            model_role=MODEL_ROLE_QUANTILE_CHECKPOINT_ALIGNMENT,
            backbones=("titantpp",),
            quantity_variants=(QUANTILE_ADAPTIVE_VARIANT, LOG_MSE_VARIANT),
            time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED,
            lambda_tail=0.0,
        )


@pytest.mark.parametrize("backbone", BACKBONES)
def test_zero_strength_is_bitwise_legacy_loss_and_all_gradient_identity(
    backbone: str,
) -> None:
    torch.manual_seed(20260906)
    control = build_model(
        backbone,
        LOG_MSE_VARIANT,
        boundaries=None,
        weights=None,
    ).eval()
    control_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260906)
    candidate = build_model(
        backbone,
        QUANTILE_ADAPTIVE_VARIANT,
        strength=0.0,
    ).eval()
    candidate_rng = torch.random.get_rng_state().clone()

    assert torch.equal(control_rng, candidate_rng)
    assert control.state_dict().keys() == candidate.state_dict().keys()
    for name, value in control.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name

    dts, mask, quantities = sample_batch()
    control_outputs = target_outputs(
        control,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    candidate_outputs = target_outputs(
        candidate,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    assert control_outputs.keys() == candidate_outputs.keys()
    for name, value in control_outputs.items():
        assert torch.equal(value, candidate_outputs[name]), name

    control_outputs["joint_loss"].mean().backward()
    candidate_outputs["joint_loss"].mean().backward()
    assert_matching_gradients(control, candidate)


def test_equal_boundaries_stay_in_lower_bin() -> None:
    model = build_model(
        "rmtpp",
        QUANTILE_ADAPTIVE_VARIANT,
        strength=1.0,
    )
    hidden = torch.zeros(9, model.hidden_dim)
    quantities = torch.tensor(
        [0.0, 10.0, 10.001, 20.0, 20.001, 30.0, 30.001, 40.0, 40.001]
    )
    expected_weights = torch.tensor(
        [0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.5, 1.5, 2.0]
    )

    outputs = model.quantity_outputs(hidden, quantities)

    assert torch.all(outputs["log_mse"] > 0.0)
    assert torch.equal(
        outputs["train_loss"],
        outputs["log_mse"] * expected_weights,
    )


def test_caller_population_mean_normalization_is_preserved_without_batch_renorm() -> None:
    counts = (50, 40, 5, 4, 1)
    raw = torch.tensor(QUANTILE_ADAPTIVE_RAW_WEIGHTS, dtype=torch.float64)
    population_raw_mean = sum(
        count * weight for count, weight in zip(counts, raw.tolist())
    ) / sum(counts)
    normalized = tuple((raw / population_raw_mean).tolist())
    quantities = torch.cat(
        [
            torch.full((counts[0],), 5.0),
            torch.full((counts[1],), 15.0),
            torch.full((counts[2],), 25.0),
            torch.full((counts[3],), 35.0),
            torch.full((counts[4],), 45.0),
        ]
    )
    model = build_model(
        "rmtpp",
        QUANTILE_ADAPTIVE_VARIANT,
        strength=1.0,
        weights=normalized,
    )
    hidden = torch.zeros(quantities.numel(), model.hidden_dim)

    outputs = model.quantity_outputs(hidden, quantities)
    observed_weights = outputs["train_loss"] / outputs["log_mse"]

    assert math.isclose(
        float(observed_weights.detach().to(torch.float64).mean()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    # A batch containing only the upper tail retains its train-population
    # normalized weight; the model performs no outcome-dependent batch scaling.
    assert math.isclose(
        float(observed_weights[-1].detach()),
        normalized[-1],
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_active_loss_changes_gradients_but_not_predictions_or_parameters() -> None:
    torch.manual_seed(73)
    control = build_model(
        "titantpp",
        LOG_MSE_VARIANT,
        boundaries=None,
        weights=None,
    )
    torch.manual_seed(73)
    candidate = build_model(
        "titantpp",
        QUANTILE_ADAPTIVE_VARIANT,
        strength=1.0,
    )
    assert control.state_dict().keys() == candidate.state_dict().keys()
    for name, value in control.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
    torch.nn.init.constant_(control.quantity_head.weight, 0.05)
    torch.nn.init.constant_(candidate.quantity_head.weight, 0.05)

    hidden_control = torch.randn(5, control.hidden_dim, requires_grad=True)
    hidden_candidate = hidden_control.detach().clone().requires_grad_(True)
    quantities = torch.tensor([5.0, 15.0, 25.0, 35.0, 45.0])
    control_outputs = control.quantity_outputs(hidden_control, quantities)
    candidate_outputs = candidate.quantity_outputs(hidden_candidate, quantities)

    for name in ("location", "point_prediction", "log_mse"):
        assert torch.equal(control_outputs[name], candidate_outputs[name]), name
    assert not torch.equal(
        control_outputs["train_loss"], candidate_outputs["train_loss"]
    )

    control_outputs["train_loss"].mean().backward()
    candidate_outputs["train_loss"].mean().backward()
    assert not torch.equal(hidden_control.grad, hidden_candidate.grad)
    assert not torch.equal(
        control.quantity_head.weight.grad,
        candidate.quantity_head.weight.grad,
    )


def test_active_loss_and_gradients_remain_finite_at_raw_quantity_extremes() -> None:
    model = build_model(
        "rmtpp",
        QUANTILE_ADAPTIVE_VARIANT,
        strength=1.0,
        boundaries=(1.0, 10.0, 1.0e10, 1.0e20),
    )
    hidden = torch.zeros(3, model.hidden_dim, requires_grad=True)
    quantities = torch.tensor([0.0, 1.0e20, torch.finfo(torch.float32).max])

    outputs = model.quantity_outputs(hidden, quantities)
    outputs["train_loss"].mean().backward()

    for name, value in outputs.items():
        assert torch.isfinite(value).all(), name
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"strength": float("nan")}, "strength"),
        ({"strength": -0.1}, "strength"),
        ({"strength": 0.5}, "strength"),
        ({"strength": 1.1}, "strength"),
        ({"boundaries": (1.0, 2.0, 3.0)}, "four values"),
        ({"boundaries": (1.0, 2.0, 2.0, 4.0)}, "strictly increasing"),
        ({"boundaries": (1.0, 2.0, 3.0, float("inf"))}, "finite"),
        ({"weights": (1.0, 1.0, 1.0, 1.0)}, "five values"),
        ({"weights": (1.0, 1.0, 0.0, 1.0, 1.0)}, "positive"),
        ({"weights": (1.0, 1.0, 1.0, 1.0, float("nan"))}, "finite"),
    ],
)
def test_quantile_contract_rejects_invalid_settings(change, message: str) -> None:
    kwargs = {
        "strength": 1.0,
        "boundaries": BOUNDARIES,
        "weights": NORMALIZED_WEIGHTS,
    }
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        build_model("rmtpp", QUANTILE_ADAPTIVE_VARIANT, **kwargs)


def test_active_contract_requires_both_boundaries_and_weights() -> None:
    with pytest.raises(ValueError, match="provided together"):
        build_model(
            "rmtpp",
            QUANTILE_ADAPTIVE_VARIANT,
            strength=1.0,
            boundaries=BOUNDARIES,
            weights=None,
        )
    with pytest.raises(ValueError, match="requires boundaries"):
        build_model(
            "rmtpp",
            QUANTILE_ADAPTIVE_VARIANT,
            strength=1.0,
            boundaries=None,
            weights=None,
        )
    with pytest.raises(ValueError, match="require the quantile-adaptive variant"):
        build_model(
            "rmtpp",
            LOG_MSE_VARIANT,
            strength=1.0,
            boundaries=BOUNDARIES,
            weights=NORMALIZED_WEIGHTS,
        )
