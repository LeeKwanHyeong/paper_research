"""Synthetic tests of observed-window time geometry and intervention boundaries."""

import math

import pytest
import torch

from paper.scripts.hard_lmm_temporal_features import observed_features, reverse_interior


def test_equal_spacing_and_quantity_changes_have_known_values():
    result = observed_features(torch.tensor([99., 2., 2., 2.]), torch.tensor([0., 1., 3., 7.]))
    assert result["history_length"] == 4
    assert result["age_distortion"] == 0
    assert result["internal_span"] == 6
    assert result["log_mean_internal_gap"] == pytest.approx(math.log(3))
    assert result["mean_log_quantity"] == pytest.approx(1.5 * math.log(2))
    assert result["latest_deviation"] == pytest.approx(1.5 * math.log(2))
    assert result["count_change_rms"] == pytest.approx(math.log(2))
    assert result["first_gap"] == 99 and result["last_gap"] == 2
    assert all(value is None or isinstance(value, (float, int)) for value in result.values())


def test_known_irregular_age_includes_last_zero_and_excludes_boundary_gap():
    # Ages [4, 3, 0]/4 versus event ages [1, .5, 0]: RMS is .25/sqrt(3).
    qty = torch.tensor([1., 2., 3.])
    result = observed_features(torch.tensor([91., 1., 3.]), qty)
    assert result["age_distortion"] == pytest.approx(.25 / math.sqrt(3), abs=1e-15)
    assert result["internal_span"] == 4
    changed = observed_features(torch.tensor([1e30, 1., 3.]), qty)
    for name in result.keys() - {"first_gap"}:
        assert result[name] == changed[name]


def test_age_distortion_is_invariant_to_common_time_scale():
    gaps = torch.tensor([7., 1., 4., 2., 8.], dtype=torch.float64)
    qty = torch.arange(5.)
    original = observed_features(gaps, qty)
    for scale in (.01, 1000.):
        scaled = observed_features(gaps * scale, qty)
        assert scaled["age_distortion"] == pytest.approx(original["age_distortion"], abs=1e-15)
        assert scaled["internal_span"] == pytest.approx(original["internal_span"] * scale)


def test_singleton_two_event_and_zero_span_are_explicitly_unavailable():
    one = observed_features(torch.tensor([8.]), torch.tensor([5.]))
    assert one["history_length"] == 1 and one["latest_deviation"] == 0
    for name in ("count_change_rms", "log_mean_internal_gap", "internal_span", "age_distortion"):
        assert one[name] is None
    two = observed_features(torch.tensor([8., 2.]), torch.tensor([1., 3.]))
    assert two["internal_span"] == 2
    assert two["log_mean_internal_gap"] == pytest.approx(math.log(3))
    assert two["count_change_rms"] == pytest.approx(math.log(2))
    assert two["age_distortion"] is None
    zero = observed_features(torch.tensor([8., 0., 0.]), torch.ones(3))
    assert zero["internal_span"] == 0 and zero["log_mean_internal_gap"] == 0
    assert zero["age_distortion"] is None


@pytest.mark.parametrize("modality", ["gap", "quantity"])
def test_reversal_preserves_endpoints_multisets_span_and_other_modality(modality):
    gaps = torch.tensor([90., 1., 2., 4., 8., 16.])
    qty = torch.tensor([3., 4., 9., 1., 8., 7.])
    gap_copy, qty_copy = gaps.clone(), qty.clone()
    new_gaps, new_qty = reverse_interior(gaps, qty, modality)
    assert torch.equal(gaps, gap_copy) and torch.equal(qty, qty_copy)
    assert new_gaps.data_ptr() != gaps.data_ptr() and new_qty.data_ptr() != qty.data_ptr()
    for old, new in ((gaps, new_gaps), (qty, new_qty)):
        assert torch.equal(old[[0, -1]], new[[0, -1]])
        assert torch.equal(old.sort().values, new.sort().values)
    assert torch.equal(new_qty, qty) if modality == "gap" else torch.equal(new_gaps, gaps)
    changed = new_gaps if modality == "gap" else new_qty
    source = gaps if modality == "gap" else qty
    assert torch.equal(changed[1:-1], source[1:-1].flip(0))
    before, after = observed_features(gaps, qty), observed_features(new_gaps, new_qty)
    for name in ("history_length", "first_gap", "last_gap", "internal_span",
                 "mean_log_quantity", "latest_deviation", "log_mean_internal_gap"):
        assert after[name] == pytest.approx(before[name])
    restored = reverse_interior(new_gaps, new_qty, modality)
    assert torch.equal(restored[0], gaps) and torch.equal(restored[1], qty)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_short_histories_are_unchanged_clones(length):
    gaps, qty = torch.arange(length).float(), torch.arange(length).float() + 1
    for modality in ("gap", "quantity"):
        changed = reverse_interior(gaps, qty, modality)
        assert torch.equal(changed[0], gaps) and torch.equal(changed[1], qty)
        assert changed[0].data_ptr() != gaps.data_ptr() and changed[1].data_ptr() != qty.data_ptr()


@pytest.mark.parametrize("gaps,qty", [
    (torch.tensor([]), torch.tensor([])),
    (torch.ones(2, 1), torch.ones(2, 1)),
    (torch.ones(2), torch.ones(3)),
    (torch.tensor([-1., 2.]), torch.ones(2)),
    (torch.ones(2), torch.tensor([1., -2.])),
    (torch.tensor([float("nan"), 2.]), torch.ones(2)),
    (torch.ones(2), torch.tensor([1., float("inf")])),
    (torch.tensor([1 + 2j]), torch.ones(1)),
])
def test_invalid_inputs_are_rejected_by_both_functions(gaps, qty):
    with pytest.raises(ValueError):
        observed_features(gaps, qty)
    with pytest.raises(ValueError):
        reverse_interior(gaps, qty)


def test_unknown_modality_is_rejected():
    with pytest.raises(ValueError, match="modality"):
        reverse_interior(torch.ones(4), torch.ones(4), "both")
