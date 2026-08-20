from __future__ import annotations

import pytest
import torch

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TAIL_SHARED_VARIANT,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
)
from paper.scripts.audit_count_aware_h0_h3_gradient_attribution import (
    classify_clipping,
    classify_quantity_damage,
    gradient_attribution,
    mixed_quantity_state,
    parameter_groups,
)


def make_model() -> CountAwareTitanTPP:
    return CountAwareTitanTPP(
        hidden_dim=16,
        train_log_mean=1.5,
        train_log_std=0.8,
        max_seq_len=8,
        quantity_variant=TAIL_SHARED_VARIANT,
        lambda_tail=0.1,
        time_head_mode=TIME_HEAD_MODE_LOGNORMAL_DURATION,
        time_scale=3.0,
        time_initial_location=-0.2,
        time_initial_scale=0.7,
    )


def test_parameter_groups_partition_trainable_parameters() -> None:
    model = make_model()
    groups = parameter_groups(model)

    selected = {name for values in groups.values() for name, _ in values}
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert selected == expected
    assert {name for name, _ in groups["time_head"]} == {
        "v_t.weight",
        "b_t",
        "w_raw",
    }
    assert {name for name, _ in groups["quantity_head"]} == {
        "quantity_head.weight",
        "quantity_head.bias",
    }


def test_gradient_attribution_identifies_time_head_clipping_driver() -> None:
    model = make_model()
    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    time_gradients = tuple(
        torch.full_like(parameter, 10.0) if name == "v_t.weight" else None
        for name, parameter in named
    )
    quantity_gradients = tuple(
        torch.full_like(parameter, 0.01)
        if name.startswith("quantity_head.")
        else None
        for name, parameter in named
    )

    result = gradient_attribution(
        named,
        time_gradients,
        quantity_gradients,
        grad_clip=1.0,
    )

    assert result["clipped"] == 1.0
    assert result["dominant_joint_gradient_group"] == "time_head"
    assert 0.0 < result["clip_scale"] < 1.0
    assert result["time_head_joint_sq_norm_share"] > 0.99


def test_quantity_crossing_classifies_encoder_damage() -> None:
    result = classify_quantity_damage(
        [
            {"combination": "H0_encoder_H0_head", "qty_mae": 1.0},
            {"combination": "H3_encoder_H3_head", "qty_mae": 1.5},
            {"combination": "H3_encoder_H0_head", "qty_mae": 1.4},
            {"combination": "H0_encoder_H3_head", "qty_mae": 1.01},
        ]
    )

    assert result["quantity_damage_location"] == "encoder_dominant"
    assert result["full_h3_mae_change_pct"] == pytest.approx(50.0)


def test_mixed_quantity_state_replaces_only_quantity_head() -> None:
    model = make_model()
    encoder_state = model.state_dict()
    head_state = {
        key: value.detach().clone() for key, value in encoder_state.items()
    }
    head_state["quantity_head.bias"] = torch.full_like(
        head_state["quantity_head.bias"], 3.0
    )

    mixed = mixed_quantity_state(encoder_state, head_state)

    assert torch.equal(mixed["quantity_head.bias"], head_state["quantity_head.bias"])
    assert torch.equal(mixed["v_t.weight"], encoder_state["v_t.weight"])


def test_clipping_classification_preserves_h0_risk_boundary() -> None:
    summary = {
        "clipped_mean": 1.0,
        "clip_scale_median": 0.25,
        "shared_encoder_joint_sq_norm_share_median": 0.2,
        "time_head_joint_sq_norm_share_median": 0.7,
        "quantity_head_joint_sq_norm_share_median": 0.1,
    }

    result = classify_clipping(summary)

    assert result["clipping_persistent"] is True
    assert result["dominant_gradient_group"] == "time_head"
    assert result["dominant_group_contract_met"] is True
