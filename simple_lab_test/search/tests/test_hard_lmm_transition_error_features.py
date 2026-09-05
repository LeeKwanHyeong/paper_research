"""Unit contracts for causal Hard-LMM transition-error features."""

import torch
from torch.nn import functional as F

from paper.scripts.hard_lmm_transition_error_features import transition_error_features


def _constant_prediction_inputs(errors: list[float]):
    """Use constant logits so even the -2 hand residual has a valid label."""
    logits = torch.full((1, len(errors)), 2.0, dtype=torch.float64)
    log_quantities = torch.tensor(errors, dtype=torch.float64)[None] + F.softplus(logits)
    return logits, log_quantities


def test_hand_calculated_prototype_prefixes_and_scalar_features():
    logits, log_quantities = _constant_prediction_inputs([0.0, 1.0, 3.0, -2.0])
    indices = torch.tensor([[
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [0, 2, 3, 4],
        [0, 1, 2, 4],
    ]])
    mask = torch.ones((1, 4), dtype=torch.bool)

    result = transition_error_features(logits, log_quantities, indices, mask, 5)

    expected_sums = torch.tensor([[
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 4, 4, 4, 3],
        [-1, 4, 2, 2, 1],
    ]], dtype=torch.float64)
    expected_counts = torch.tensor([[
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 2, 2, 2, 1],
        [2, 2, 3, 3, 2],
    ]])
    torch.testing.assert_close(result["prototype_error_sum"], expected_sums)
    assert torch.equal(result["prototype_error_count"], expected_counts)
    torch.testing.assert_close(
        result["prototype_conditioned_top4_mean"],
        torch.tensor([[0.0, 0.75, 2.0, 2.0 / 3.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["unconditioned_prefix_mean"],
        torch.tensor([[0.0, 1.0, 2.0, 2.0 / 3.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["last_error"], torch.tensor([[0.0, 1.0, 3.0, -2.0]], dtype=torch.float64)
    )
    assert torch.equal(result["valid_transition_count"], torch.tensor([[0, 1, 2, 3]]))
    # A one-event history has no completed transition and every correction is zero.
    for key, value in result.items():
        assert value[:, 0].count_nonzero() == 0, key


def test_prefix_causality_and_masked_values_indices_cannot_leak():
    logits, log_quantities = _constant_prediction_inputs([0.0, 0.5, 1.5, 2.5, 3.5])
    indices = torch.tensor([[
        [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5],
        [0, 3, 4, 5], [0, 1, 4, 5],
    ]])
    mask = torch.tensor([[True, True, True, True, False]])
    before = transition_error_features(logits, log_quantities, indices, mask, 6)

    changed_logits, changed_quantities, changed_indices = logits.clone(), log_quantities.clone(), indices.clone()
    changed_logits[:, 3:] = torch.tensor([[-9e5, float("nan")]])
    changed_quantities[:, 3:] = torch.tensor([[9e5, float("nan")]])
    changed_indices[:, 3] = torch.tensor([5, 4, 1, 0])
    changed_indices[:, 4] = torch.tensor([-99, 700, -3, 99])
    after = transition_error_features(changed_logits, changed_quantities, changed_indices, mask, 6)

    # State 2 can only contain transitions 1 and 2.  Changes at 3+ cannot alter it.
    for key in before:
        torch.testing.assert_close(before[key][:, :3], after[key][:, :3])
    # Padding values and even out-of-range padding IDs are ignored and return zeros.
    for key, value in after.items():
        assert value[:, 4].count_nonzero() == 0, key


def test_write_disabled_target_quantity_cannot_enter_any_prefix_bucket():
    logits, log_quantities = _constant_prediction_inputs([0.0, 1.0, 2.0, 3.0])
    indices = torch.tensor([[
        [0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4],
    ]])
    observed = torch.ones((1, 4), dtype=torch.bool)
    writes = torch.tensor([[True, True, True, False]])
    before = transition_error_features(
        logits, log_quantities, indices, observed, 5, memory_write_mask=writes
    )
    changed_quantities = log_quantities.clone()
    changed_quantities[0, 3] = 1e12
    after = transition_error_features(
        logits, changed_quantities, indices, observed, 5, memory_write_mask=writes
    )

    for key in before:
        torch.testing.assert_close(before[key], after[key])
    assert torch.equal(before["valid_transition_count"], torch.tensor([[0, 1, 2, 2]]))


def test_observed_and_write_masks_require_both_transition_endpoints():
    logits = torch.zeros((1, 6), dtype=torch.float64)
    log_quantities = torch.tensor([[float("nan"), 0.2, 1.2, 9.2, 4.2, float("nan")]])
    log_quantities[:, 1:5] += F.softplus(torch.zeros((), dtype=torch.float64))
    indices = torch.tensor([[
        [-20, -20, -20, -20],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [0, 2, 3, 4],
        [0, 1, 3, 4],
        [100, 100, 100, 100],
    ]])
    observed = torch.tensor([[False, True, True, True, True, False]])
    writable = torch.tensor([[False, True, True, False, True, False]])

    result = transition_error_features(
        logits, log_quantities, indices, observed, 5, memory_write_mask=writable
    )

    # Only transition t=2 has two observed, writable endpoints.  t=3 and t=4
    # are blocked by the write-disabled shared endpoint at position 3.
    assert torch.equal(result["valid_transition_count"], torch.tensor([[0, 0, 1, 1, 1, 0]]))
    torch.testing.assert_close(
        result["last_error"],
        torch.tensor([[0.0, 0.0, 1.2, 1.2, 1.2, 0.0]], dtype=torch.float64),
    )
    expected_final_sum = torch.tensor([1.2, 1.2, 1.2, 1.2, 0.0], dtype=torch.float64)
    torch.testing.assert_close(result["prototype_error_sum"][0, 4], expected_final_sum)
    assert torch.equal(result["prototype_error_count"][0, 4], torch.tensor([1, 1, 1, 1, 0]))
    assert result["prototype_conditioned_top4_mean"][0, 1] == 0


def test_rows_are_batch_isolated_and_match_separate_calls():
    logits = torch.zeros((2, 4), dtype=torch.float64)
    base_prediction = F.softplus(logits)
    log_quantities = base_prediction + torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 10.0, 20.0, 30.0],
    ])
    indices = torch.tensor([
        [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4]],
        [[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4]],
    ])
    mask = torch.ones((2, 4), dtype=torch.bool)
    together = transition_error_features(logits, log_quantities, indices, mask, 5)

    for row in range(2):
        separate = transition_error_features(
            logits[row : row + 1], log_quantities[row : row + 1],
            indices[row : row + 1], mask[row : row + 1], 5,
        )
        for key in together:
            torch.testing.assert_close(together[key][row : row + 1], separate[key])


def test_optional_sham_changes_only_write_buckets_and_keeps_actual_read_indices():
    logits, log_quantities = _constant_prediction_inputs([0.0, 2.0, 4.0])
    actual_indices = torch.tensor([[
        [0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4],
    ]])
    # A fixed cyclic ID permutation uses only each source state's own IDs, so it
    # is a future-free sham.  Actual IDs above remain the reads at every state.
    sham_write_indices = (actual_indices + 1) % 5
    mask = torch.ones((1, 3), dtype=torch.bool)
    original = transition_error_features(logits, log_quantities, actual_indices, mask, 5)
    sham = transition_error_features(
        logits, log_quantities, actual_indices, mask, 5,
        write_top4_prototype_indices=sham_write_indices,
    )

    # Scalar prefix summaries depend only on errors, so the sham cannot alter them.
    for key in ("unconditioned_prefix_mean", "last_error", "valid_transition_count"):
        torch.testing.assert_close(original[key], sham[key])
    assert not torch.equal(original["prototype_error_sum"], sham["prototype_error_sum"])
    # At state 1 the actual read set is [1,2,3,4].  The original write touched
    # three of these four IDs; the sham write touched all four.
    torch.testing.assert_close(
        original["prototype_conditioned_top4_mean"][0, 1], torch.tensor(1.5, dtype=torch.float64)
    )
    torch.testing.assert_close(
        sham["prototype_conditioned_top4_mean"][0, 1], torch.tensor(2.0, dtype=torch.float64)
    )


def test_base_residual_target_is_detached_but_observations_remain_differentiable():
    logits = torch.zeros((1, 4), dtype=torch.float64, requires_grad=True)
    log_quantities = torch.full(
        (1, 4), F.softplus(torch.zeros((), dtype=torch.float64)).item() + 1.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    indices = torch.tensor([[
        [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
    ]])
    mask = torch.tensor([[True, True, True, False]])
    result = transition_error_features(logits, log_quantities, indices, mask, 4)
    loss = result["unconditioned_prefix_mean"].sum() + result["last_error"].sum()
    loss.backward()

    assert logits.grad is None
    assert log_quantities.grad is not None
    assert log_quantities.grad[0, 0] == 0
    assert torch.all(log_quantities.grad[0, 1:3] > 0)
    assert log_quantities.grad[0, 3] == 0


def test_zero_quantity_error_is_finite_and_matches_log_mse_residual():
    logits = torch.tensor([[12.0, -50.0]], dtype=torch.float64)
    observed_log1p = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    indices = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
    mask = torch.ones((1, 2), dtype=torch.bool)

    result = transition_error_features(logits, observed_log1p, indices, mask, 4)
    expected = -F.softplus(logits[0, 0])
    assert torch.isfinite(result["last_error"]).all()
    torch.testing.assert_close(result["last_error"][0, 1], expected)
