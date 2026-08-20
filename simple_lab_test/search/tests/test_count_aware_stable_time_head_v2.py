from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    TIME_HEAD_MODE_SCALED_EXACT_STABLE,
    CountAwareRMTPP,
    inverse_sigmoid,
)
from paper.scripts.count_aware_tpp_backbone.training import (
    build_optimizer,
    optimizer_group_contract,
    train_epoch_with_telemetry,
)
from paper.scripts.run_count_aware_time_head_stability import (
    evaluate_stability_gate,
    load_train_only_frame,
    select_stable_variant,
    should_run_h2,
)
from paper.scripts.run_count_aware_tpp_backbone_control import (
    derive_train_time_contract,
    prepare_count_frame,
)


TRAIN_MEAN_DT = 2.9969021695
STABLE_TIME_KWARGS = {
    "time_head_mode": TIME_HEAD_MODE_SCALED_EXACT_STABLE,
    "time_scale": 3.0,
    "time_w_max": 2.0 / 3.0,
    "time_intercept_limit": 6.0,
    "time_initial_intercept": math.log(3.0 / TRAIN_MEAN_DT),
    "time_wd_safety_limit": 8.0,
}


def build_stable_rmtpp(*, hidden_dim: int = 8) -> CountAwareRMTPP:
    torch.manual_seed(17)
    return CountAwareRMTPP(
        hidden_dim,
        train_log_mean=1.5,
        **STABLE_TIME_KWARGS,
    )


@pytest.mark.parametrize("backbone", ["rmtpp", "thp", "titantpp"])
def test_stable_exact_contract_is_shared_by_primary_backbones(backbone: str) -> None:
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
        **STABLE_TIME_KWARGS,
    )

    assert model.time_head_contract() == metadata["time_head"]
    assert metadata["time_head"] == {
        "mode": TIME_HEAD_MODE_SCALED_EXACT_STABLE,
        "time_scale": 3.0,
        "time_w_max": 2.0 / 3.0,
        "time_intercept_limit": 6.0,
        "jacobian_correction": True,
        "wd_clamp": 0.0,
        "time_initial_intercept": math.log(3.0 / TRAIN_MEAN_DT),
        "time_intercept_transform": "scaled_tanh",
        "time_wd_safety_limit": 8.0,
    }


def test_stable_constants_and_initial_intercept_are_train_derived() -> None:
    frame = pl.DataFrame(
        {
            "oper_part_no": ["a", "a", "a", "a", "b", "b", "b"],
            "seq": [0, 1, 2, 3, 0, 1, 2],
            "delta_t": [999, 1, 3, 100, 999, 5, 100],
            "demand_qty": [1.0] * 7,
            "chronological_split": [
                "train",
                "train",
                "train",
                "validation",
                "train",
                "train",
                "test",
            ],
        }
    )

    contract = derive_train_time_contract(
        prepare_count_frame(frame),
        lookback_weeks=520,
        max_seq_len=8,
        wd_safety_limit=8.0,
    )

    assert contract["target_count"] == 3
    assert contract["target_dt_mean"] == 3.0
    assert contract["time_scale"] == 3.0
    assert contract["target_dt_max"] == 5.0
    assert contract["time_w_max"] == pytest.approx(4.8)
    assert contract["time_initial_intercept"] == pytest.approx(0.0)


def test_stable_initial_hazard_matches_inverse_train_mean() -> None:
    model = build_stable_rmtpp().double()
    model.v_t.weight.data.zero_()
    hidden = torch.zeros(1, model.hidden_dim, dtype=torch.float64)
    zero_dt = torch.zeros(1, dtype=torch.float64)

    original_time_hazard = torch.exp(model.log_f_dt(hidden, zero_dt))

    assert torch.allclose(
        original_time_hazard,
        original_time_hazard.new_tensor([1.0 / TRAIN_MEAN_DT]),
        atol=1e-10,
        rtol=1e-10,
    )


def test_stable_intercept_uses_smooth_bound() -> None:
    model = build_stable_rmtpp().double()
    model.v_t.weight.data.fill_(2.0)
    hidden = torch.tensor(
        [[100.0] * model.hidden_dim, [-100.0] * model.hidden_dim],
        dtype=torch.float64,
    )

    intercept = model.bounded_time_intercept(hidden)

    assert torch.all(intercept.abs() <= model.time_intercept_limit)
    assert intercept[0] > 0.0
    assert intercept[1] < 0.0


def test_stable_exact_density_integrates_to_one_without_duration_clamp() -> None:
    model = build_stable_rmtpp().double()
    model.v_t.weight.data.zero_()
    model.w_raw.data.fill_(inverse_sigmoid(0.15 / model.time_w_max))
    grid = torch.linspace(0.0, 36.0, 40_001, dtype=torch.float64)
    hidden = torch.zeros(grid.numel(), model.hidden_dim, dtype=torch.float64)

    density = torch.exp(model.log_f_dt(hidden, grid))
    integral = torch.trapezoid(density, grid)

    assert torch.isclose(integral, integral.new_tensor(1.0), atol=2e-6, rtol=0.0)


def test_stable_exact_forward_and_backward_are_finite_on_extreme_range() -> None:
    model = build_stable_rmtpp()
    hidden = torch.randn(4, model.hidden_dim, requires_grad=True)
    delta_t = torch.tensor([1.0, 36.0, 120.0, 360.0])

    loss = -model.log_f_dt(hidden, delta_t).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert model.w_raw.grad is not None and torch.isfinite(model.w_raw.grad).all()
    assert model.v_t.weight.grad is not None
    assert torch.isfinite(model.v_t.weight.grad).all()


def test_lower_time_lr_routes_only_time_head_parameters() -> None:
    model = build_stable_rmtpp()
    optimizer = build_optimizer(model, lr=1e-3, time_head_lr_multiplier=0.1)
    groups = optimizer_group_contract(optimizer)
    time_ids = {id(parameter) for _, parameter in model.time_head_named_parameters()}

    assert [group["group_name"] for group in groups] == [
        "backbone_and_quantity",
        "time_head",
    ]
    assert [group["lr"] for group in groups] == pytest.approx([1e-3, 1e-4])
    assert {id(parameter) for parameter in optimizer.param_groups[1]["params"]} == time_ids
    assert not (
        {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        & time_ids
    )


def test_train_epoch_reports_finite_preclip_telemetry() -> None:
    model = build_stable_rmtpp()
    optimizer = build_optimizer(model, lr=1e-3)
    dts = torch.tensor(
        [[0.0, 1.0, 3.0], [0.0, 2.0, 5.0], [0.0, 1.0, 36.0]]
    )
    quantities = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 4.0, 2.0], [2.0, 3.0, 8.0]]
    )
    mask = torch.ones_like(dts, dtype=torch.bool)
    loader = [(None, dts, mask, None, quantities)]

    telemetry = train_epoch_with_telemetry(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device="cpu",
        lambda_log_qty=1.0,
        grad_clip=1.0,
        max_batches=None,
    )

    assert telemetry["train_event_count"] == 3
    assert telemetry["train_batch_count"] == 1
    assert telemetry["train_all_finite"] is True
    assert math.isfinite(telemetry["train_joint_objective"])
    assert math.isfinite(telemetry["train_max_per_event_time_nll"])
    assert math.isfinite(telemetry["train_pre_clip_grad_norm_mean"])
    assert 0.0 <= telemetry["train_gradient_clip_fraction"] <= 1.0


def passing_history() -> list[dict[str, float | int | bool]]:
    return [
        {
            "train_joint_objective": 2.0,
            "train_time_nll": 1.0,
            "train_quantity_loss": 1.0,
            "train_batch_joint_p99": 3.0,
            "train_max_per_event_time_nll": 20.0,
            "train_pre_clip_grad_norm_mean": 0.5,
            "train_pre_clip_grad_norm_max": 0.9,
            "train_gradient_clip_fraction": 0.0,
            "train_gradient_clip_count": 0,
            "train_batch_count": 4,
            "train_time_slope": 0.1,
            "train_all_finite": True,
        }
    ]


def test_train_gate_and_conditional_h2_selection() -> None:
    passing_gate = evaluate_stability_gate(passing_history(), run_status="success")
    failing_history = passing_history()
    failing_history[0]["train_batch_joint_p99"] = 101.0
    failing_gate = evaluate_stability_gate(failing_history, run_status="success")
    h1_pass = {"stability_gate": passing_gate}
    h1_fail = {"stability_gate": failing_gate}
    h2_pass = {"stability_gate": passing_gate}

    assert passing_gate["passed"] is True
    assert failing_gate["passed"] is False
    assert should_run_h2(h1_pass) is False
    assert should_run_h2(h1_fail) is True
    assert select_stable_variant({"H1": h1_pass}) == "H1"
    assert select_stable_variant({"H1": h1_fail, "H2": h2_pass}) == "H2"
    assert select_stable_variant({"H1": h1_fail}) is None


def test_train_only_loader_excludes_validation_and_test_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.parquet"
    pl.DataFrame(
        {
            "oper_part_no": ["a", "a", "a"],
            "seq": [0, 1, 2],
            "delta_t": [0, 1, 2],
            "demand_qty": [1.0, 2.0, 3.0],
            "chronological_split": ["train", "validation", "test"],
        }
    ).write_parquet(path)

    train = load_train_only_frame(path)

    assert train.height == 1
    assert train["chronological_split"].to_list() == ["train"]
