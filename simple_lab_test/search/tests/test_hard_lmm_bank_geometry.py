"""Synthetic checks of label-free bank geometry; no real artifacts loaded."""

import json

import pytest
import torch

from paper.scripts.hard_lmm_bank_geometry import analyze_bank


def example():
    values = torch.tensor([[-2., 0.], [2., 0.], [0., 3.], [0., -3.]], dtype=torch.float64)
    indices = torch.tensor([[0, 1], [1, 2], [2, 3]])
    weights = torch.tensor([[.5, .5], [.25, .75], [.8, .2]], dtype=torch.float64)
    return values, indices, weights, torch.tensor([2., 0.]), torch.tensor([0., 3.])


def test_known_cancellation_has_within_variance_but_no_between_variance():
    values, _, _, q, t = example()
    summary, slots = analyze_bank(values, torch.tensor([[0, 1], [0, 1]]),
                                 torch.full((2, 2), .5), q, t)
    variance = summary["selected_variance_decomposition"]["quantity"]
    assert variance["total"] == 16.0
    assert variance["mean_within_query"] == 16.0
    assert variance["between_query"] == 0.0
    assert variance["between_over_total"] == 0.0
    assert summary["averaging"]["quantity"]["cancellation_about_selected_mean_average"] == 1.0
    assert slots["selection_count"].tolist() == [2, 2, 0, 0]
    assert slots["selection_mass"].tolist() == [.5, .5, 0., 0.]


def test_constant_values_and_zero_head_have_defined_safe_output():
    values, indices, weights, q, t = example()
    values[:] = torch.tensor([1., 2.])
    q.zero_()
    summary, _ = analyze_bank(values, indices, weights, q, t)
    assert summary["head_direction_cosine"] is None
    assert summary["distributions"]["full_uniform"]["axes"]["quantity"]["unit_head"] is None
    assert summary["distributions"]["full_uniform"]["covariance_effective_rank"] == 0.0
    assert summary["selected_variance_decomposition"]["quantity"]["between_over_total"] is None
    json.dumps(summary, allow_nan=False)


def test_constant_nonbinary_values_weights_and_bank_size_remain_exactly_zero():
    values = torch.tensor([[.1, .3]] * 7, dtype=torch.float64)
    indices = torch.tensor([[0, 1, 2], [2, 4, 6], [1, 3, 5]])
    weights = torch.tensor([[.1, .2, .7], [.3, .3, .4], [.2, .7, .1]], dtype=torch.float64)
    summary, _ = analyze_bank(values, indices, weights, torch.tensor([.3, .7]), torch.tensor([.9, -.2]))
    for distribution in summary["distributions"].values():
        assert distribution["covariance_trace"] == 0.0
        assert distribution["covariance_effective_rank"] == 0.0
    for decomposition in summary["selected_variance_decomposition"].values():
        assert decomposition["total"] == 0.0
        assert decomposition["between_query"] == 0.0
        assert decomposition["mean_within_query"] == 0.0
        assert decomposition["between_over_total"] is None


def test_head_scaling_changes_raw_projection_but_not_unit_direction_geometry():
    values, indices, weights, q, t = example()
    base, _ = analyze_bank(values, indices, weights, q, t)
    scaled, _ = analyze_bank(values, indices, weights, q * 7, t * 0.2)
    for distribution in base["distributions"]:
        original = base["distributions"][distribution]["axes"]["quantity"]
        changed = scaled["distributions"][distribution]["axes"]["quantity"]
        assert original["unit_head"] == changed["unit_head"]
        assert original["unit_axis_variance_over_vector_trace"] == changed["unit_axis_variance_over_vector_trace"]
        assert changed["raw"]["std"] == pytest.approx(original["raw"]["std"] * 7)


def test_mixture_identity_matches_independent_scalar_and_vector_loops():
    values, indices, weights, q, t = example()
    summary, _ = analyze_bank(values, indices, weights, q, t)
    items = [(int(slot), float(weight) / len(indices)) for row, row_weights in zip(indices, weights)
             for slot, weight in zip(row, row_weights)]
    for name, head in (("quantity", q), ("time", t), ("vector_trace", None)):
        projected = values if head is None else (values @ head.double())[:, None]
        center = sum(weight * projected[slot] for slot, weight in items)
        total = sum(weight * float((projected[slot] - center).square().sum()) for slot, weight in items)
        within = between = 0.0
        for row, row_weights in zip(indices, weights):
            mean = sum(float(weight) * projected[int(slot)] for slot, weight in zip(row, row_weights))
            between += float((mean - center).square().sum()) / len(indices)
            within += sum(float(weight) * float((projected[int(slot)] - mean).square().sum())
                          for slot, weight in zip(row, row_weights)) / len(indices)
        result = summary["selected_variance_decomposition"][name]
        assert result["total"] == pytest.approx(total, abs=1e-12)
        assert result["between_query"] == pytest.approx(between, abs=1e-12)
        assert result["mean_within_query"] == pytest.approx(within, abs=1e-12)
        assert abs(result["closure_error"]) < 1e-12


def test_full_union_and_empirical_support_are_distinct():
    values, _, _, q, t = example()
    summary, slots = analyze_bank(values, torch.tensor([[0, 1], [0, 1]]),
                                 torch.tensor([[.9, .1], [.9, .1]], dtype=torch.float64), q, t)
    distributions = summary["distributions"]
    assert summary["active_slots"] == [0, 1]
    assert summary["unused_slots"] == [2, 3]
    assert distributions["full_uniform"]["covariance_trace"] == 6.5
    assert distributions["active_union_uniform"]["covariance_trace"] == 4.0
    assert distributions["selected_mass"]["covariance_trace"] == pytest.approx(1.44)
    assert slots["selected"].tolist() == [True, True, False, False]


def test_storage_roundoff_is_disclosed_and_mixture_mass_is_normalized():
    values, indices, weights, q, t = example()
    weights[0] *= 1.0000001
    summary, slots = analyze_bank(values, indices, weights, q, t)
    assert 0 < summary["input_weight_sum_max_error"] < 1e-6
    assert float(slots["selection_mass"].sum()) == pytest.approx(1.0)
    assert abs(summary["selected_variance_decomposition"]["quantity"]["closure_error"]) < 1e-12


@pytest.mark.parametrize("fault", ["negative", "sum", "bounds", "nan", "shape", "float_indices"])
def test_invalid_input_fails_closed(fault):
    values, indices, weights, q, t = example()
    if fault == "negative":
        weights[0] = torch.tensor([-1., 2.])
    elif fault == "sum":
        weights *= .5
    elif fault == "bounds":
        indices[0, 0] = 4
    elif fault == "nan":
        values[0, 0] = float("nan")
    elif fault == "shape":
        q = q[None]
    elif fault == "float_indices":
        indices = indices.double()
    with pytest.raises(ValueError):
        analyze_bank(values, indices, weights, q, t)
