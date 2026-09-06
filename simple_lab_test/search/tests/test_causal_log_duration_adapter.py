from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CausalLogDurationAdapter import CausalLogDurationAdapter


def build_adapter(*, double: bool = False) -> CausalLogDurationAdapter:
    torch.manual_seed(20260906)
    adapter = CausalLogDurationAdapter(
        log_duration_mean=math.log1p(3.0),
        log_duration_std=0.75,
        sigma_floor=1e-3,
    )
    return adapter.double() if double else adapter


def base_log_density(
    location: torch.Tensor,
    scale: torch.Tensor,
    target_delta_t: torch.Tensor,
    *,
    time_scale: float,
) -> torch.Tensor:
    log_delta_t = torch.log(target_delta_t.to(torch.float64))
    standardized = (
        log_delta_t - math.log(time_scale) - location.to(torch.float64)
    ) / scale.to(torch.float64)
    return (
        -0.5 * torch.square(standardized)
        - torch.log(scale.to(torch.float64))
        - log_delta_t
        - 0.5 * math.log(2.0 * math.pi)
    )


def base_log_survival(
    location: torch.Tensor,
    scale: torch.Tensor,
    target_delta_t: torch.Tensor,
    *,
    time_scale: float,
) -> torch.Tensor:
    standardized = (
        torch.log(target_delta_t.to(torch.float64))
        - math.log(time_scale)
        - location.to(torch.float64)
    ) / scale.to(torch.float64)
    return torch.special.log_ndtr(-standardized)


def test_architecture_and_train_only_scalar_normalization_contract() -> None:
    adapter = build_adapter(double=True)
    history = torch.tensor(
        [[0.0, 3.0, 8.0, 999.0], [3.0, float("nan"), -7.0, float("inf")]],
        dtype=torch.float64,
    )
    lengths = torch.tensor([3, 1], dtype=torch.long)

    normalized = adapter.normalized_history(history, lengths).squeeze(-1)
    expected = torch.zeros_like(normalized)
    expected[0, :3] = (
        torch.log1p(history[0, :3]) - math.log1p(3.0)
    ) / 0.75
    expected[1, 0] = 0.0

    assert adapter.gru.input_size == 1
    assert adapter.gru.hidden_size == 8
    assert adapter.gru.num_layers == 1
    assert adapter.gru.dropout == 0.0
    assert torch.equal(adapter.scale_projection.weight, torch.zeros(1, 8))
    assert torch.equal(adapter.scale_projection.bias, torch.zeros(1))
    assert torch.equal(normalized, expected)
    assert set(adapter.state_dict()) == {
        "log_duration_mean",
        "log_duration_std",
        "sigma_floor",
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.bias_ih_l0",
        "gru.bias_hh_l0",
        "scale_projection.weight",
        "scale_projection.bias",
    }


def test_lengths_exclude_target_and_padding_and_preserve_prefix_causality() -> None:
    adapter = build_adapter(double=True)
    with torch.no_grad():
        adapter.scale_projection.weight.fill_(0.2)
        adapter.scale_projection.bias.fill_(0.1)

    same_prefix_different_suffix = torch.tensor(
        [
            [1.0, 2.0, 4.0, 17.0, 31.0],
            [1.0, 2.0, 4.0, float("nan"), float("inf")],
        ],
        dtype=torch.float64,
    )
    lengths = torch.tensor([3, 3])
    prefix_multiplier = adapter(same_prefix_different_suffix, lengths)

    changed_past = same_prefix_different_suffix.clone()
    changed_past[1, 1] = 12.0
    changed_multiplier = adapter(changed_past, lengths)

    assert torch.equal(prefix_multiplier[0], prefix_multiplier[1])
    assert not torch.isclose(changed_multiplier[0], changed_multiplier[1])

    base_location = torch.tensor([0.1, 0.1], dtype=torch.float64)
    base_scale = torch.tensor([0.6, 0.6], dtype=torch.float64)
    location, scale = adapter.adjusted_location_scale(
        base_location,
        base_scale,
        same_prefix_different_suffix,
        lengths,
    )
    # A next-event target is accepted only by likelihood evaluation. It cannot
    # enter the recurrent adapter or change its adjusted parameters.
    target_a = torch.tensor([3.0, 3.0], dtype=torch.float64)
    target_b = torch.tensor([30.0, 30.0], dtype=torch.float64)
    assert location.data_ptr() == base_location.data_ptr()
    assert torch.equal(scale[0], scale[1])
    assert not torch.equal(
        adapter.proper_log_density(
            base_location,
            base_scale,
            target_a,
            same_prefix_different_suffix,
            lengths,
            time_scale=3.0,
        ),
        adapter.proper_log_density(
            base_location,
            base_scale,
            target_b,
            same_prefix_different_suffix,
            lengths,
            time_scale=3.0,
        ),
    )


def test_zero_initialization_exactly_matches_base_density_and_survival() -> None:
    adapter = build_adapter(double=True)
    history = torch.tensor(
        [[1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    lengths = torch.tensor([2, 3, 0])
    location = torch.tensor([-0.4, 0.2, 1.1], dtype=torch.float64)
    scale = torch.tensor([0.3, 0.7, 1.4], dtype=torch.float64)
    target = torch.tensor([0.125, 3.0, 1000.0], dtype=torch.float64)

    adjusted_location, adjusted_scale = adapter.adjusted_location_scale(
        location,
        scale,
        history,
        lengths,
    )
    density = adapter.proper_log_density(
        location,
        scale,
        target,
        history,
        lengths,
        time_scale=3.0,
    )
    survival = adapter.proper_log_survival(
        location,
        scale,
        target,
        history,
        lengths,
        time_scale=3.0,
    )

    assert adjusted_location.data_ptr() == location.data_ptr()
    assert torch.equal(adjusted_location, location)
    assert torch.equal(adjusted_scale, scale)
    assert torch.equal(adapter(history, lengths), torch.ones(3, dtype=torch.float64))
    assert torch.equal(
        density,
        base_log_density(location, scale, target, time_scale=3.0),
    )
    assert torch.equal(
        survival,
        base_log_survival(location, scale, target, time_scale=3.0),
    )


def test_first_step_trains_projection_then_second_step_reaches_gru() -> None:
    adapter = build_adapter(double=True)
    history = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 5.0, 0.0], [8.0, 1.0, 4.0]],
        dtype=torch.float64,
    )
    lengths = torch.tensor([3, 2, 3])
    location = torch.tensor([-0.2, 0.1, 0.3], dtype=torch.float64)
    scale = torch.tensor([0.35, 0.55, 0.8], dtype=torch.float64)
    target = torch.tensor([12.0, 0.4, 30.0], dtype=torch.float64)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)

    first_loss = -adapter.proper_log_density(
        location,
        scale,
        target,
        history,
        lengths,
        time_scale=3.0,
    ).mean()
    first_loss.backward()

    assert torch.isfinite(first_loss)
    assert adapter.scale_projection.weight.grad is not None
    assert torch.isfinite(adapter.scale_projection.weight.grad).all()
    assert torch.count_nonzero(adapter.scale_projection.weight.grad) > 0
    assert adapter.scale_projection.bias.grad is not None
    assert torch.count_nonzero(adapter.scale_projection.bias.grad) > 0
    for name, parameter in adapter.gru.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) == 0, name

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_loss = -adapter.proper_log_density(
        location,
        scale,
        target,
        history,
        lengths,
        time_scale=3.0,
    ).mean()
    second_loss.backward()

    assert torch.isfinite(second_loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in adapter.gru.parameters()
    )


def test_scale_multiplier_and_adjusted_sigma_are_bounded_and_finite() -> None:
    adapter = build_adapter(double=True)
    history = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=torch.float64
    )
    lengths = torch.tensor([2, 2, 0])
    base_scale = torch.tensor([0.4, 1.2, 0.7], dtype=torch.float64)
    floor = adapter.sigma_floor.item()

    with torch.no_grad():
        adapter.scale_projection.weight.zero_()
        adapter.scale_projection.bias.fill_(1.0e6)
    high_multiplier = adapter(history, lengths)
    high_scale = adapter.adjusted_scale(base_scale, history, lengths)

    with torch.no_grad():
        adapter.scale_projection.bias.fill_(-1.0e6)
    low_multiplier = adapter(history, lengths)
    low_scale = adapter.adjusted_scale(base_scale, history, lengths)

    assert torch.isfinite(high_multiplier).all()
    assert torch.isfinite(low_multiplier).all()
    assert torch.all(high_multiplier[:2] <= 10.0)
    assert torch.all(low_multiplier[:2] >= 0.1)
    assert high_multiplier[2] == 1.0
    assert low_multiplier[2] == 1.0
    assert torch.allclose(
        high_scale[:2],
        floor + (base_scale[:2] - floor) * 10.0,
    )
    assert torch.allclose(
        low_scale[:2],
        floor + (base_scale[:2] - floor) * 0.1,
    )
    assert high_scale[2] == base_scale[2]
    assert low_scale[2] == base_scale[2]
    assert torch.all(low_scale > floor)


def test_median_is_unchanged_even_after_scale_adapter_changes() -> None:
    adapter = build_adapter(double=True)
    location = torch.tensor([-2.0, 0.0, 1.5], dtype=torch.float64)
    expected = 3.0 * torch.exp(location)
    before = adapter.predict_median(location, time_scale=3.0)

    with torch.no_grad():
        for parameter in adapter.gru.parameters():
            parameter.fill_(0.7)
        adapter.scale_projection.weight.fill_(-4.0)
        adapter.scale_projection.bias.fill_(5.0)
    after = adapter.predict_median(location, time_scale=3.0)

    assert torch.equal(before, expected)
    assert torch.equal(after, expected)


def test_extreme_proper_likelihood_and_survival_gradients_stay_finite() -> None:
    adapter = build_adapter(double=True)
    with torch.no_grad():
        adapter.scale_projection.weight.fill_(0.05)
    history = torch.tensor(
        [[0.0, 1.0e-12, 1.0e30], [1.0e20, 1.0, 0.0]],
        dtype=torch.float64,
    )
    lengths = torch.tensor([3, 2])
    location = torch.tensor([-0.5, 0.75], dtype=torch.float64)
    scale = torch.tensor([0.2, 1.5], dtype=torch.float64)
    target = torch.tensor([1.0e-30, 1.0e30], dtype=torch.float64)

    density = adapter.proper_log_density(
        location,
        scale,
        target,
        history,
        lengths,
        time_scale=3.0,
    )
    survival = adapter.proper_log_survival(
        location,
        scale,
        target,
        history,
        lengths,
        time_scale=3.0,
    )
    loss = -density.mean() - 0.01 * survival.mean()
    loss.backward()

    assert torch.isfinite(density).all()
    assert torch.isfinite(survival).all()
    assert torch.isfinite(loss)
    for parameter in adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_invalid_active_values_are_rejected_but_invalid_padding_is_ignored() -> None:
    adapter = build_adapter()
    padded_invalid = torch.tensor([[1.0, 2.0, float("nan"), -3.0]])
    assert torch.isfinite(adapter(padded_invalid, torch.tensor([2]))).all()

    with pytest.raises(ValueError, match="finite"):
        adapter(padded_invalid, torch.tensor([3]))
    with pytest.raises(ValueError, match="nonnegative"):
        adapter(torch.tensor([[1.0, -2.0]]), torch.tensor([2]))
    with pytest.raises(ValueError, match="padded time axis"):
        adapter(torch.ones(1, 2), torch.tensor([3]))
    with pytest.raises(TypeError, match="integer"):
        adapter(torch.ones(1, 2), torch.tensor([2.0]))
