import math

import polars as pl

from simple_lab_test.search.analyze_raw_quantity_revin_audit import (
    REVIN_EPS,
    choose_q2_constants,
    q1_statistics,
    q2_statistics,
)


def make_contexts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "context_count": [1, 2],
            "history_mean_qty": [2.0, 3.0],
            "history_var_qty": [0.0, 1.0],
            "target_qty": [4.0, 5.0],
        }
    )


def test_q1_one_event_uses_canonical_epsilon_scale() -> None:
    center, scale, target_norm = q1_statistics(make_contexts())

    assert center[0] == 2.0
    assert math.isclose(scale[0], math.sqrt(REVIN_EPS), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        target_norm[0],
        (4.0 - 2.0) / math.sqrt(REVIN_EPS),
        rel_tol=0.0,
        abs_tol=1e-9,
    )


def test_q2_mixes_first_and_second_moments() -> None:
    contexts = make_contexts().head(1)
    center, scale, target_norm, alpha = q2_statistics(
        contexts,
        global_mean=4.0,
        global_var=4.0,
        sigma_floor=0.01,
        shrinkage_k=1.0,
    )

    assert alpha[0] == 0.5
    assert center[0] == 3.0
    assert math.isclose(scale[0], math.sqrt(3.0), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(target_norm[0], 1.0 / math.sqrt(3.0), rel_tol=0.0, abs_tol=1e-12)


def test_q2_sigma_floor_is_applied_after_moment_mixing() -> None:
    contexts = pl.DataFrame(
        {
            "context_count": [2],
            "history_mean_qty": [4.0],
            "history_var_qty": [0.0],
            "target_qty": [4.0],
        }
    )
    _, scale, target_norm, _ = q2_statistics(
        contexts,
        global_mean=4.0,
        global_var=0.0,
        sigma_floor=0.25,
        shrinkage_k=2.0,
    )

    assert scale[0] == 0.25
    assert target_norm[0] == 0.0


def test_choose_q2_constants_ranks_eligible_candidate_by_p99_then_k() -> None:
    candidates = pl.DataFrame(
        {
            "shrinkage_k": [1.0, 2.0, 4.0],
            "target_abs_norm_p99": [1.5, 1.2, 1.2],
            "eligible": [True, True, True],
        }
    )
    constants = choose_q2_constants(
        candidates,
        global_mean=3.0,
        global_var=4.0,
        global_std=2.0,
        sigma_floor=0.002,
    )

    assert constants["status"] == "frozen"
    assert constants["shrinkage_k"] == 2.0
    assert constants["selection_scope"] == "fixed_split_train_only"


def test_choose_q2_constants_blocks_when_no_candidate_is_eligible() -> None:
    candidates = pl.DataFrame(
        {
            "shrinkage_k": [1.0, 2.0],
            "target_abs_norm_p99": [2.0, 1.5],
            "eligible": [False, False],
        }
    )
    constants = choose_q2_constants(
        candidates,
        global_mean=3.0,
        global_var=4.0,
        global_std=2.0,
        sigma_floor=0.002,
    )

    assert constants["status"] == "blocked"
    assert constants["shrinkage_k"] is None
    assert constants["decision"] == "block_q2_no_train_only_eligible_k"
