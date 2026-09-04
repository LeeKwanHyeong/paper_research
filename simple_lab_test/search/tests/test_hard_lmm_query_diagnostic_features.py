"""Synthetic causal and reconstruction checks; no dataset or optimizer runs."""

import pytest
import torch
from torch.nn import functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.hard_lmm_query_diagnostic_features import extract


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def build(backbone):
    torch.manual_seed(19)
    model, _ = build_count_aware_model(
        backbone, hidden_dim=16, train_log_mean=1.5, max_seq_len=6,
        quantity_variant="count_only_log_regression",
    )
    with torch.no_grad():
        model.quantity_head.weight.normal_(0, .1)
        model.lmm.mem.normal_(0, .3)
        if hasattr(model.lmm, "memory_keys"):
            # Deliberately untie keys so a value-as-key regression cannot hide.
            model.lmm.memory_keys.normal_(0, .5)
    return model.requires_grad_(False).eval()


def batch():
    # One left-padded history of length 3 and one of length 1.
    dt = torch.tensor([[0., 0., 1., 2., 4., 9.], [0., 0., 0., 0., 3., 7.]])
    qty = torch.tensor([[0., 0., 1., 3., 7., 20.], [0., 0., 0., 0., 5., 30.]])
    return dt, dt > 0, qty


@pytest.mark.parametrize("backbone", ["titantpp", KEY_VALUE_BACKBONE])
def test_observed_stats_exact_retrieval_official_prediction_and_no_mutation(backbone):
    model = build(backbone)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    dt, mask, qty = batch()
    result = extract(model, dt, mask, qty)
    expected_stats = torch.tensor([[torch.log(torch.tensor(4.)), torch.log(torch.tensor(2.))],
                                   [torch.log(torch.tensor(6.)), 0.]])
    torch.testing.assert_close(result["stats"], expected_stats)
    assert torch.equal(result["history_length"], torch.tensor([3, 1]))
    with torch.no_grad():
        official = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
    torch.testing.assert_close(result["prediction"], official["pred_qty"])
    torch.testing.assert_close(result["quantity"], official["true_qty"])
    torch.testing.assert_close(result["log_residual"], qty[:, -1].log1p() - F.softplus(result["z"]))

    keys = model.lmm.memory_keys if backbone == KEY_VALUE_BACKBONE else model.lmm.mem
    scores = F.normalize(result["h"], dim=-1) @ F.normalize(keys[0], dim=-1).T
    expected_scores, expected_indices = scores.topk(4, -1)
    assert torch.equal(result["indices"], expected_indices)
    torch.testing.assert_close(result["scores"], expected_scores)
    expected_weights = expected_scores.softmax(-1) if backbone == KEY_VALUE_BACKBONE else torch.full((2, 4), .25)
    torch.testing.assert_close(result["weights"], expected_weights)
    projected_values = (model.lmm.mem[0][expected_indices] * model.quantity_head.weight[0]).sum(-1)
    torch.testing.assert_close(result["value_projections"], projected_values)
    torch.testing.assert_close(result["projection"], (projected_values * expected_weights).sum(-1))
    for key, value in result.items():
        assert value.device.type == "cpu" and not value.requires_grad and torch.isfinite(value).all(), key
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize("backbone", ["titantpp", KEY_VALUE_BACKBONE])
def test_target_padding_and_layout_cannot_change_features_or_routing(backbone):
    model = build(backbone)
    dt, mask, qty = batch()
    original = extract(model, dt, mask, qty)
    changed_dt, changed_qty = dt.clone(), qty.clone()
    changed_dt[:, -1] = torch.tensor([1e6, 2e6])
    changed_qty[:, -1] = torch.tensor([3e6, 4e6])
    changed_dt[~mask] = float("nan")
    changed_qty[~mask] = float("nan")
    changed = extract(model, changed_dt, mask, changed_qty)
    for key in original.keys() - {"quantity", "log_residual"}:
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
    assert not torch.equal(original["log_residual"], changed["log_residual"])

    # Different padding positions preserve observed order and therefore the cache.
    order = torch.tensor([[2, 0, 3, 1, 4, 5], [4, 5, 0, 1, 2, 3]])
    permuted = extract(model, dt.gather(1, order), mask.gather(1, order), qty.gather(1, order))
    for key in original:
        torch.testing.assert_close(original[key], permuted[key], rtol=0, atol=0)


def test_separate_keys_independently_verified_and_used_in_routing():
    model = build(KEY_VALUE_BACKBONE)
    dt, mask, qty = batch()
    original = extract(model, dt, mask, qty)
    with torch.no_grad():
        model.lmm.memory_keys.copy_(model.lmm.memory_keys.roll(1, dims=1))
    changed = extract(model, dt, mask, qty)
    torch.testing.assert_close(original["h"], changed["h"], rtol=0, atol=0)
    torch.testing.assert_close(original["stats"], changed["stats"], rtol=0, atol=0)
    assert not torch.equal(original["indices"], changed["indices"])
    assert not torch.equal(original["prediction"], changed["prediction"])


def test_requires_frozen_eval_and_rejects_invalid_or_unsupported_input():
    model = build("titantpp")
    dt, mask, qty = batch()
    model.train()
    with pytest.raises(ValueError, match="frozen"):
        extract(model, dt, mask, qty)
    model.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        extract(model, dt, mask, qty)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="shape"):
        extract(model, dt[:, :-1], mask, qty)
    with pytest.raises(ValueError, match="boolean"):
        extract(model, dt, mask.float(), qty)
    with pytest.raises(ValueError, match="history"):
        extract(model, dt, torch.tensor([[False] * 5 + [True]] * 2), qty)
    bad_qty = qty.clone()
    bad_qty[0, -1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        extract(model, dt, mask, bad_qty)
    bad_qty[0, -1] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        extract(model, dt, mask, bad_qty)
    weighted = build("titantpp_weighted_static_memory")
    with pytest.raises(ValueError, match="original or separate-key"):
        extract(weighted, dt, mask, qty)
