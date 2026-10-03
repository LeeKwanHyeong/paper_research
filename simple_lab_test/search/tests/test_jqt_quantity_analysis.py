import numpy as np
import pytest

from paper.scripts.jqt_quantity_analysis import analyze_quantity_pair, verify_quantity_replay


def analysis(**overrides):
    values = dict(dataset_id="synthetic", truth=[1, 2, 10, 100], pred_j=[0, 4, 8, 102], pred_q=[1, 3, 9, 101], log_loss_j=[4, 4, 4, 4], log_loss_q=[1, 1, 1, 1], history_length=[1, 3, 7, 15], series_ids=["a", "a", "b", "c"], quantity_boundaries=[2, 31, 46, 187], history_boundaries=[3, 7])
    values.update(overrides)
    return analyze_quantity_pair(**values)


def test_fixed_left_boundaries_empty_cells_and_ties_are_explicit():
    report = analysis(truth=[2, 31, 46, 187], pred_j=[2, 31, 46, 187], pred_q=[2, 31, 46, 187])
    assert [cell["n"] for cell in report["quantity_cells"]] == [1, 1, 1, 1, 0]
    assert report["quantity_cells"][-1]["j"]["rmse"] is None
    assert report["overall"]["j"]["exact_tie_proportion"] == 1.0
    assert report["binning"]["searchsorted_side"] == "left"
    assert all(value["passed"] for value in report["audit"].values())


def test_hand_calculated_weighted_and_additive_paired_decomposition():
    report = analysis(truth=[1, 10], pred_j=[0, 10], pred_q=[1, 12], log_loss_j=[5, 7], log_loss_q=[3, 9], history_length=[1, 3], series_ids=["a", "b"])
    overall = report["overall"]
    assert overall["delta"]["mse"] == pytest.approx(1.5)
    assert overall["delta"]["mae"] == pytest.approx(0.5)
    assert overall["delta"]["log_mse"] == pytest.approx(0.0)
    decomp = overall["paired_sse_decomposition"]
    assert decomp["cross_term"] == pytest.approx(-1.0)
    assert decomp["shift_squared"] == pytest.approx(2.5)
    assert decomp["total_delta_mse"] == pytest.approx(overall["delta"]["mse"])
    assert sum(cell["weighted_contribution"]["mse"] or 0 for cell in report["quantity_cells"]) == pytest.approx(1.5)


def test_series_cancellation_is_retained_without_causal_label():
    report = analysis(truth=[1, 1], pred_j=[0, 3], pred_q=[3, 2], log_loss_j=[1, 1], log_loss_q=[1, 1], history_length=[1, 1], series_ids=["worse", "better"])
    concentration = report["series_concentration"]
    assert concentration["numbers"] == {"improve": 1, "worsen": 1, "tie": 0}
    assert concentration["net_delta_sse"] == pytest.approx(0.0)
    assert concentration["positive_gross_delta_sse"] == pytest.approx(3.0)
    assert concentration["negative_gross_delta_sse"] == pytest.approx(-3.0)
    assert concentration["top_contribution_to_positive_gross"]["top_1_percent"]["denominator"] == "all_series_count"


def test_series_concentration_preserves_id_tie_breaking_after_vectorized_grouping():
    report = analysis(truth=[1, 1], pred_j=[0, 0], pred_q=[3, 3], log_loss_j=[1, 1], log_loss_q=[1, 1], history_length=[1, 1], series_ids=["z", "a"])
    assert report["series_concentration"]["top_10_worsening_ids"] == ["a", "z"]


def test_replay_rejects_small_delta_distortion_despite_per_arm_tolerance():
    expected_j = {"raw_quantity_rmse": 1.0, "quantity_mae": 1.0, "quantity_train_loss": 1.0}
    expected_q = {"raw_quantity_rmse": 1.00001, "quantity_mae": 1.00001, "quantity_train_loss": 1.00001}
    actual_j = {"raw_quantity_rmse": 1.000009, "quantity_mae": 1.000009, "quantity_train_loss": 1.000009}
    actual_q = {"raw_quantity_rmse": 1.000011, "quantity_mae": 1.000011, "quantity_train_loss": 1.000011}
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_quantity_replay(actual_j, actual_q, expected_j, expected_q)
    assert verify_quantity_replay(expected_j, expected_q, expected_j, expected_q)["passed"]


def test_replay_applies_delta_contract_to_rmse_and_mae_but_not_train_loss():
    expected_j = {"raw_quantity_rmse": 1.0, "quantity_mae": 1.0, "quantity_train_loss": 1.0}
    expected_q = {"raw_quantity_rmse": 1.0, "quantity_mae": 1.0, "quantity_train_loss": 1.00001}
    actual_j = {"raw_quantity_rmse": 1.0, "quantity_mae": 1.0, "quantity_train_loss": 1.000009}
    actual_q = {"raw_quantity_rmse": 1.0, "quantity_mae": 1.0, "quantity_train_loss": 1.000011}
    checks = verify_quantity_replay(actual_j, actual_q, expected_j, expected_q)["checks"]
    assert checks["raw_quantity_rmse"]["delta_contract_applied"] is True
    assert checks["quantity_mae"]["delta_contract_applied"] is True
    assert checks["quantity_train_loss"]["delta_contract_applied"] is False


@pytest.mark.parametrize("keyword", [
    {"truth": [1, np.nan, 2]},
    {"pred_j": [0, -1, 2, 3]},
    {"truth": [[1, 2], [3, 4]]},
    {"history_length": [1, 0, 3, 4]},
    {"series_ids": ["a", "b", 3, "d"]},
])
def test_invalid_input_is_rejected(keyword):
    with pytest.raises(ValueError):
        analysis(**keyword)
