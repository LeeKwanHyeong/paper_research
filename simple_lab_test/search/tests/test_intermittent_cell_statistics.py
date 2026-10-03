"""Meaningful synthetic-only checks for paired validation accounting."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from paper.scripts.intermittent_cell_statistics import (
    SUM_FIELDS, cell_indices, compare_rows, safe_cosine, summarize_rows,
)

Q = [2, 31, 46, 187]
H = [64, 128]
BOUNDS = {"quantity_boundaries": Q, "history_boundaries": H}


def rows(truth, errors, history):
    truth = np.asarray(truth, dtype=np.float64)
    n = len(truth)
    return {"true_qty": truth, "pred_qty": truth + np.asarray(errors, dtype=np.float64),
            "history_length": np.asarray(history, dtype=np.int64),
            "series_ids": np.asarray(["series" + str(i % 2) for i in range(n)]),
            "target_position": np.arange(n, dtype=np.int64),
            "target_seq": np.arange(n, dtype=np.int64) + 100}


def test_boundaries_equal_lower_bin_and_all_fifteen_cells():
    q_values = np.asarray([0, 2, 3, 31, 32, 46, 47, 187, 188], dtype=np.float64)
    q_expected = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4])
    for history, h_index in [(0, 0), (64, 0), (65, 1), (128, 1), (129, 2), (255, 2)]:
        found = cell_indices(q_values, np.full(len(q_values), history), BOUNDS)
        np.testing.assert_array_equal(found, q_expected * 3 + h_index)
        assert found.dtype == np.int64
    q = np.repeat([2, 31, 46, 187, 188], 3)
    history = np.tile([64, 128, 255], 5)
    np.testing.assert_array_equal(cell_indices(q, history, BOUNDS), np.arange(15))


@pytest.mark.parametrize("truth,history", [([-1], [0]), ([float("nan")], [0]), ([0], [float("inf")]),
                                              ([0], [-1]), ([0], [256]), ([0], [64.5]), ([0, 1], [0])])
def test_invalid_truth_or_history_rejected(truth, history):
    with pytest.raises(ValueError):
        cell_indices(truth, history, BOUNDS)


@pytest.mark.parametrize("q,h", [([2, 31, 31, 187], H), ([2, 31, 46], H), (Q, [64, 128, 255]),
                                 (Q, [128, 64]), (Q, [64.5, 128]), ([2, 31, float("inf"), 187], H)])
def test_invalid_boundaries_rejected(q, h):
    with pytest.raises(ValueError):
        cell_indices([1], [1], {"quantity_boundaries": q, "history_boundaries": h})


def test_uneven_sign_partition_exact_sums_empty_cells_and_losses():
    data = rows([1, 2, 31, 46, 100, 200], [-1, 2, 0, -4, 5, -6], [0, 0, 64, 65, 129, 255])
    data["time_loss"] = np.array([-2., 0., 1., 2., 3., 4.])
    result = summarize_rows(data, Q, H)
    overall = result["overall"]
    expected = {"count": 6, "true_quantity_sum": 380., "predicted_quantity_sum": 376.,
                "signed_error_sum": -4., "absolute_error_sum": 18., "raw_squared_error_sum": 82.,
                "under_count": 3, "over_count": 2, "tie_count": 1,
                "under_absolute_error_sum": 11., "over_absolute_error_sum": 7.}
    for key, value in expected.items():
        assert overall[key] == value
    assert overall["MAE"] == 3. and overall["RMSE"] == math.sqrt(82 / 6)
    assert overall["mean_bias"] == -4 / 6 and overall["under_rate"] == .5
    assert overall["losses"]["time_loss"] == {"sum": 8., "mean": 8/6}
    assert result["conservation_audit"]["passed"]
    assert [len(result[k]) for k in ("quantity_cells", "history_cells", "cross_cells")] == [5, 3, 15]
    empty = result["cross_cells"][1]
    assert all(empty[key] == 0 for key in SUM_FIELDS)
    assert all(empty[key] is None for key in ("mean_bias", "MAE", "RMSE", "under_rate", "over_rate", "tie_rate"))
    assert empty["losses"]["time_loss"] == {"sum": 0., "mean": None}
    json.dumps(result, allow_nan=False)


def test_partial_batches_are_combined_by_rows_not_mean_of_batch_means():
    data = rows([1, 1, 1, 1, 1], [0, 0, 0, 0, 10], [0, 0, 0, 0, 0])
    batch_a = {key: value[:4] for key, value in data.items()}
    batch_b = {key: value[4:] for key, value in data.items()}
    combined = {key: np.concatenate([batch_a[key], batch_b[key]]) for key in data}
    result = summarize_rows(combined, Q, H)
    a, b = [summarize_rows(batch, Q, H)["overall"] for batch in (batch_a, batch_b)]
    assert result["overall"]["MAE"] == 2.
    assert result["overall"]["MAE"] != (a["MAE"] + b["MAE"]) / 2
    for key in SUM_FIELDS:
        assert result["overall"][key] == a[key] + b[key]


def test_opposing_cell_changes_report_signed_shares_above_one_without_causal_claim():
    base = rows([1, 100], [2, 0], [0, 129])
    candidate = rows([1, 100], [1, 2], [0, 129])
    result = compare_rows(base, candidate, Q, H)
    total = result["overall"]
    assert total["delta_SSE"] == 1 and total["delta_MSE_contribution"] == .5
    assert total["delta_AE"] == 1 and total["delta_MAE_contribution"] == .5
    assert total["delta_RMSE"] == math.sqrt(5 / 2) - math.sqrt(4 / 2)
    assert total["delta_RMSE"] != total["delta_MSE_contribution"]
    accounting = result["signed_cell_change_accounting"]["quantity_cells"]["delta_SSE"]
    assert accounting == {"positive_sum": 4., "negative_sum": -3., "net_sum": 1.}
    first, fourth = result["quantity_cells"][0], result["quantity_cells"][3]
    assert first["shares"]["delta_SSE"] == {"net": -3., "positive": 0., "negative": 1.}
    assert fourth["shares"]["delta_SSE"] == {"net": 4., "positive": 1., "negative": 0.}
    assert result["conservation_audit"]["passed"]
    empty = result["cross_cells"][1]
    assert empty["count"] == 0 and empty["fraction"] is None
    assert empty["delta_SSE"] == empty["delta_MSE_contribution"] == empty["delta_AE"] == empty["delta_MAE_contribution"] == 0
    assert all(value is None for shares in empty["shares"].values() for value in shares.values())
    assert all("delta_RMSE" not in cell for cell in result["cross_cells"])
    assert "not causal percentages" in result["definitions"]["interpretation"]
    json.dumps(result, allow_nan=False)


def test_exact_offsetting_net_shares_null_but_signed_sums_remain_visible():
    base = rows([1, 100], [2, 0], [0, 129])
    candidate = rows([1, 100], [0, 2], [0, 129])
    result = compare_rows(base, candidate, Q, H)
    assert result["overall"]["delta_SSE"] == 0
    assert result["overall"]["delta_MAE_contribution"] == 0
    for label in ("quantity_cells", "history_cells", "cross_cells"):
        assert result["signed_cell_change_accounting"][label]["delta_SSE"] == {"positive_sum": 4., "negative_sum": -4., "net_sum": 0.}
        assert all(cell["shares"]["delta_SSE"]["net"] is None for cell in result[label])
    identical = compare_rows(base, deepcopy(base), Q, H)
    assert all(all(value is None for value in cell["shares"]["delta_SSE"].values()) for cell in identical["cross_cells"])


def test_row_and_cell_positive_negative_changes_are_distinct_when_one_cell_cancels():
    base = rows([1, 1], [2, 0], [0, 0])
    candidate = rows([1, 1], [0, 2], [0, 0])
    result = compare_rows(base, candidate, Q, H)
    assert result["signed_row_change_accounting"]["delta_SSE"] == {"positive_sum": 4., "negative_sum": -4., "net_sum": 0.}
    assert result["signed_cell_change_accounting"]["cross_cells"]["delta_SSE"] == {"positive_sum": 0., "negative_sum": 0., "net_sum": 0.}


@pytest.mark.parametrize("column", ["series_ids", "target_position", "target_seq", "true_qty", "history_length"])
def test_all_pairing_fields_are_exact_and_mismatches_fail(column):
    base = rows([1, 3, 40], [1, 1, 1], [0, 100, 200])
    candidate = deepcopy(base)
    if column == "series_ids": candidate[column] = np.array(["changed", "series1", "series0"])
    else: candidate[column][0] += 1
    with pytest.raises(ValueError, match="Paired"):
        compare_rows(base, candidate, Q, H)


def test_order_mismatch_duplicate_ids_and_partial_columns_rejected():
    base = rows([1, 3, 40], [1, 1, 1], [0, 100, 200])
    candidate = {key: values[::-1] for key, values in base.items()}
    with pytest.raises(ValueError, match="order"):
        compare_rows(base, candidate, Q, H)
    duplicate = {key: np.concatenate([values, values[:1]]) for key, values in base.items()}
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_rows(duplicate, Q, H)
    truncated = deepcopy(base); truncated["pred_qty"] = truncated["pred_qty"][:-1]
    with pytest.raises(ValueError, match="length"):
        summarize_rows(truncated, Q, H)
    empty = {key: values[:0] for key, values in base.items()}
    with pytest.raises(ValueError, match="Empty validation"):
        summarize_rows(empty, Q, H)


@pytest.mark.parametrize("column", ["true_qty", "pred_qty", "time_loss", "log_qty_loss", "raw_qty_loss", "objective_loss", "quantity_train_loss"])
def test_any_nonfinite_prediction_truth_or_optional_loss_fails(column):
    data = rows([1, 3], [1, 1], [0, 100])
    data[column] = np.asarray([0., float("nan")])
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize_rows(data, Q, H)


def test_overflow_and_noninteger_identity_rejected():
    data = rows([1, 3], [1, 1], [0, 100])
    data["pred_qty"][0] = 1e308
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize_rows(data, Q, H)
    data = rows([1, 3], [1, 1], [0, 100])
    data["target_position"] = np.array([0., 1.])
    with pytest.raises(ValueError, match="integer identities"):
        summarize_rows(data, Q, H)


def test_zero_norm_cosine_is_none_and_large_small_vectors_are_stable():
    assert safe_cosine([0, 0], [1, 2]) is None
    assert safe_cosine([], []) is None
    assert safe_cosine([1e300, 1e300], [-1e300, -1e300]) == pytest.approx(-1.)
    assert safe_cosine([1e-300, 0], [0, 1e-300]) == 0.
    with pytest.raises(ValueError, match="Nonfinite"):
        safe_cosine([float("nan")], [0])


def test_noninteger_values_accumulate_float64_and_conserve_across_all_cells():
    rng = np.random.default_rng(42)
    truth = rng.uniform(0, 500, 133)
    history = rng.integers(0, 256, 133)
    base = rows(truth, rng.normal(0, 4, 133), history)
    candidate = rows(truth, rng.normal(0, 3, 133), history)
    before = {key: value.copy() for key, value in base.items()}
    result = compare_rows(base, candidate, Q, H)
    assert result["accumulator_dtype"] == "float64"
    assert result["conservation_audit"]["passed"]
    assert result["pairing_audit"]["count"] == 133
    for label in ("quantity_cells", "history_cells", "cross_cells"):
        assert sum(cell["count"] for cell in result[label]) == 133
        assert math.fsum(cell["delta_SSE"] for cell in result[label]) == pytest.approx(result["overall"]["delta_SSE"])
        assert math.fsum(cell["delta_MAE_contribution"] for cell in result[label]) == pytest.approx(result["overall"]["delta_MAE_contribution"])
    for key in before:
        np.testing.assert_array_equal(base[key], before[key])
    json.dumps(result, allow_nan=False)
