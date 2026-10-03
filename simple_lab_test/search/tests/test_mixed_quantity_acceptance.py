"""Validation-only replay and preregistered screening gate checks."""

from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts.mixed_quantity_acceptance import (
    CASES,
    DATASETS,
    assess_acceptance,
    evaluate_checkpoints,
)
from simple_lab_test.search.tests.test_mixed_quantity_engine import (
    make_model,
    objectives,
    run,
    statistics,
    tensors,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def passing_results():
    """Small explicit gate inputs; these are not experimental results."""
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for case in CASES:
            arm = {"execution_identity": {"source": "synthetic", "dataset": dataset},
                   "evaluation_split": "validation", "held_out_evaluated": False}
            for scope in ("primary", "last120"):
                arm[scope] = {
                    "epoch": 30 if scope == "primary" else 120,
                    "count": 100,
                    "cell_counts": {"body": 90, "lowest": 40, "tail": 2},
                    "validation_batch_sha256": "same-population",
                    "metrics": {"raw_quantity_rmse": 10. if case == CASES[0] else 9.8,
                                "quantity_mae": 2., "body_mae": 2., "lowest_mae": 1.,
                                "tail_mae": 3., "legacy_time_loss": -2.},
                }
            results[dataset][case] = arm
    return results


def test_gate_requires_one_common_candidate_and_prefers_matched_when_both_pass():
    results = passing_results()
    accepted = assess_acceptance(results)
    assert accepted["candidate_passed"] == {CASES[1]: True, CASES[2]: True}
    assert accepted["recommended_candidate"] == CASES[2]
    assert accepted["benchmark_superiority_established"] is False
    results[DATASETS[0]][CASES[1]]["primary"]["metrics"]["body_mae"] = 2.1
    results[DATASETS[1]][CASES[2]]["primary"]["metrics"]["raw_quantity_rmse"] = 10.
    rejected = assess_acceptance(results)
    assert rejected["candidate_passed"] == {CASES[1]: False, CASES[2]: False}
    assert rejected["recommended_candidate"] is None


@pytest.mark.parametrize("baseline_time,bound", [(-2., -1.98), (-.1, -.09), (0., .01), (2., 2.02)])
def test_legacy_time_gate_uses_additive_bound_for_negative_and_near_zero_losses(baseline_time, bound):
    results = passing_results()
    for arms in results.values():
        for scope in ("primary", "last120"):
            arms[CASES[0]][scope]["metrics"]["legacy_time_loss"] = baseline_time
            for case in CASES[1:]:
                arms[case][scope]["metrics"]["legacy_time_loss"] = bound
    assert assess_acceptance(results)["candidate_passed"][CASES[1]]
    results[DATASETS[0]][CASES[1]]["primary"]["metrics"]["legacy_time_loss"] = bound + 1e-4
    rejected = assess_acceptance(results)
    assert rejected["candidate_passed"][CASES[1]] is False
    assert rejected["candidate_passed"][CASES[2]] is True


def test_empty_tail_cell_is_indeterminate_failure_and_final_requires_120():
    results = passing_results()
    for case in CASES:
        results[DATASETS[0]][case]["primary"]["cell_counts"]["tail"] = 0
        results[DATASETS[0]][case]["primary"]["metrics"]["tail_mae"] = None
    rejected = assess_acceptance(results)
    assert not any(rejected["candidate_passed"].values())
    assert any(row["metric"] == "tail_mae" and row["upper_bound"] is None
               and row["passed"] is False for row in rejected["checks"])
    results = passing_results()
    results[DATASETS[0]][CASES[2]]["last120"]["epoch"] = 119
    with pytest.raises(ValueError, match="120 completed"):
        assess_acceptance(results)


@pytest.mark.parametrize("change", ["population", "identity"])
def test_gate_rejects_incomparable_population_or_execution_identity(change):
    results = passing_results()
    candidate = results[DATASETS[0]][CASES[1]]
    if change == "population":
        candidate["primary"]["validation_batch_sha256"] = "other-population"
    else:
        candidate["execution_identity"] = {"source": "another-source"}
    with pytest.raises(ValueError, match="population differs|identity mismatch"):
        assess_acceptance(results)


@pytest.fixture
def completed_arm(tmp_path):
    objective = objectives()[2]
    summary = run(tmp_path, objective)
    return tmp_path, objective, summary


def replay(arm, *, batch=None, synthetic=True, budget_check=None):
    path, objective, _ = arm
    loader = DataLoader(TensorDataset(*(tensors() if batch is None else batch)), batch_size=2)
    return evaluate_checkpoints(
        model=make_model(), validation_loader=loader, statistics=statistics(),
        objective=objective, arm_dir=path, quantity_boundaries=[2., 3., 4., 4.5],
        synthetic=synthetic, budget_check=budget_check)


def test_selected_and_final_replay_preserve_checkpoint_scope_and_validation_counts(completed_arm):
    stages = []
    result = replay(completed_arm, budget_check=stages.append)
    _, objective, summary = completed_arm
    assert result["evaluation_split"] == "validation"
    assert result["held_out_evaluated"] is False
    assert result["condition"]["mixed_objective"] == objective.to_dict()
    primary = summary["selectors"]["raw_quantity_rmse"]
    assert result["primary"]["epoch"] == primary["best_epoch"]
    assert result["primary"]["state_sha256"] == primary["state_sha256"]
    # Synthetic integration uses three epochs; production acceptance separately
    # requires 120 and cannot treat this smoke run as a qualifying result.
    assert result["last120"]["epoch"] == 3
    assert result["last120"]["state_sha256"] == summary["last_state_sha256"]
    for scope in ("primary", "last120"):
        state = result[scope]
        history = summary["history"][state["epoch"] - 1]
        assert state["count"] == 4
        assert state["cell_counts"] == {"body": 3, "lowest": 1, "tail": 1}
        for metric in ("raw_quantity_rmse", "quantity_mae", "legacy_time_loss",
                       "log_quantity_mse", "quantity_train_loss"):
            assert state["metrics"][metric] == pytest.approx(history[metric], rel=1e-5, abs=1e-5)
    assert result["primary"]["validation_batch_sha256"] == result["last120"]["validation_batch_sha256"]
    assert stages == ["checkpoint_validation"] * 4


def test_replay_rejects_tampered_selected_checkpoint_metadata(completed_arm):
    path, _, _ = completed_arm
    selected_path = path / "best_raw_quantity_rmse_model.pt"
    selected = torch.load(selected_path, weights_only=False)
    selected["global_step"] += 1
    torch.save(selected, selected_path)
    with pytest.raises(ValueError, match="metadata mismatch"):
        replay(completed_arm)


def test_replay_rejects_changed_targets_and_unidentified_split(completed_arm):
    batch = list(deepcopy(tensors()))
    batch[2][:, -1] += .25
    with pytest.raises(ValueError, match="Validation replay mismatch"):
        replay(completed_arm, batch=tuple(batch))
    with pytest.raises(ValueError, match="Validation-only targets"):
        replay(completed_arm, synthetic=False)
