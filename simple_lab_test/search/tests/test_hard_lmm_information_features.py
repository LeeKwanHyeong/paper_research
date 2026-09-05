"""Synthetic stage/cause boundary checks; no actual data or probe fitting."""

import math

import numpy as np
import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch, target_outputs
from paper.scripts.hard_lmm_information_features import HISTORY_FEATURE_NAMES, STAGE_NAMES, extract, observed_history_features
from paper.scripts.hard_lmm_query_diagnostic_features import extract as prior_extract
from paper.scripts.hard_lmm_temporal_features import observed_features


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(backbone="titantpp"):
    torch.manual_seed(19)
    model, _ = build_count_aware_model(backbone, hidden_dim=16, train_log_mean=1.5,
                                      max_seq_len=6, quantity_variant="count_only_log_regression")
    with torch.no_grad():
        model.quantity_head.weight.normal_(0, .1)
        model.lmm.mem.normal_(0, .3)
        if hasattr(model.lmm, "memory_keys"):
            model.lmm.memory_keys.normal_(0, .5)
    return model.requires_grad_(False).eval()


def batch():
    dt = torch.tensor([[0., 0., 1., 2., 4., 9.], [0., 0., 0., 0., 3., 7.]])
    qty = torch.tensor([[0., 0., 1., 3., 7., 20.], [0., 0., 0., 0., 5., 30.]])
    return dt, dt > 0, qty


def hook_counts(model):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]


@pytest.mark.parametrize("backbone", ["titantpp", KEY_VALUE_BACKBONE])
def test_stages_pooling_previous_cache_and_official_prediction(backbone):
    model = build(backbone)
    dt, mask, qty = batch()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    hooks_before = hook_counts(model)
    actual = extract(model, dt, mask, qty)
    assert hooks_before == hook_counts(model)
    old = prior_extract(model, dt, mask, qty)
    with torch.no_grad():
        official = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
    torch.testing.assert_close(actual["prediction"], official["pred_qty"])
    for key in ("prediction", "quantity", "h", "z", "history_length"):
        torch.testing.assert_close(actual[key], old[key])
    torch.testing.assert_close(actual["h"], actual["layer2_last"], rtol=0, atol=0)
    torch.testing.assert_close(actual["fused"], actual["h"] + actual["r"], rtol=0, atol=0)
    torch.testing.assert_close(actual["concat_hr"], torch.cat((actual["h"], actual["r"]), 1), rtol=0, atol=0)
    torch.testing.assert_close(actual["base_logpred"], actual["base_log_prediction"], rtol=0, atol=0)

    canonical_dt, canonical_q, canonical_mask, lengths = right_pad_batch(dt, qty, mask)
    rows = torch.arange(len(lengths))
    canonical_mask[rows, lengths - 1] = False
    canonical_dt.masked_fill_(~canonical_mask, 0)
    canonical_q.masked_fill_(~canonical_mask, 0)
    with torch.no_grad():
        x = model.encoder.input_proj(model.continuous_features(canonical_dt, canonical_q, canonical_mask))
        x = (x + model.encoder._get_pos(x.size(1), x.device, x.dtype)) * canonical_mask.unsqueeze(-1)
        for name, layer in (("input", None), ("layer1", model.encoder.layers[0]), ("layer2", model.encoder.layers[1])):
            if layer is not None:
                x = layer(x, mask=canonical_mask)
            torch.testing.assert_close(actual[f"{name}_last"], x[rows, lengths - 2])
            torch.testing.assert_close(actual[f"{name}_mean"], (x * canonical_mask.unsqueeze(-1)).sum(1) / (lengths - 1).unsqueeze(-1))
    for key in STAGE_NAMES:
        assert actual[key].shape == (2, 32 if key == "concat_hr" else 16)
    assert actual["history_features"].shape == (2, 12)
    for key, tensor in actual.items():
        assert tensor.device.type == "cpu" and not tensor.requires_grad and torch.isfinite(tensor).all(), key
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize("backbone", ["titantpp", KEY_VALUE_BACKBONE])
def test_target_gap_quantity_and_poison_padding_do_not_enter_features(backbone):
    model = build(backbone)
    dt, mask, qty = batch()
    original = extract(model, dt, mask, qty)
    changed_dt, changed_qty = dt.clone(), qty.clone()
    changed_dt[:, -1] = float("nan")
    changed_qty[:, -1] = torch.tensor([2e6, 3e6])
    changed_dt[~mask] = float("inf")
    changed_qty[~mask] = float("nan")
    changed = extract(model, changed_dt, mask, changed_qty)
    for key in original.keys() - {"quantity", "log_quantity"}:
        torch.testing.assert_close(original[key], changed[key], rtol=0, atol=0)
    assert not torch.equal(original["quantity"], changed["quantity"])
    assert not torch.equal(original["log_quantity"], changed["log_quantity"])
    right_dt, right_q, right_mask, _ = right_pad_batch(dt, qty, mask)
    right = extract(model, right_dt, right_mask, right_q)
    for key in original:
        torch.testing.assert_close(original[key], right[key], rtol=0, atol=0)


def test_independent_f12_formulas_and_boundary_cases():
    histories = [([9.], [5.]), ([100., 3.], [1., 7.]),
                 ([999., 2., 5., 3.], [0., 2., 8., 3.]), ([2., 0., 0.], [1., 1., 1.])]
    dt, qty = torch.zeros(4, 6), torch.zeros(4, 6)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    expected = []
    for index, (gap_values, qty_values) in enumerate(histories):
        length = len(gap_values)
        dt[index, :length], qty[index, :length], mask[index, :length] = torch.tensor(gap_values), torch.tensor(qty_values), True
        logq = np.log1p(qty_values)
        internal_loggap = np.log1p(gap_values[1:])
        slope = np.polyfit(np.linspace(0, 1, length), logq, 1)[0] if length > 1 else 0.
        temporal = observed_features(torch.tensor(gap_values), torch.tensor(qty_values))["age_distortion"]
        expected.append([math.log1p(length), np.mean(logq), logq[-1], np.std(logq), np.mean(logq[-3:]),
                         logq[-1] - np.mean(logq), slope,
                         np.mean(internal_loggap) if length > 1 else 0., internal_loggap[-1] if length > 1 else 0.,
                         np.std(internal_loggap) if length > 1 else 0., math.log1p(sum(gap_values[1:])), temporal or 0.])
    actual = observed_history_features(dt, qty, mask)
    assert len(HISTORY_FEATURE_NAMES) == 12
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32), rtol=1e-5, atol=1e-6)
    assert torch.equal(actual[0, 6:], torch.zeros(6))
    assert actual[1, 9] == actual[1, 11] == 0
    changed_dt = dt.clone()
    changed_dt[:, 0] += 1e6
    torch.testing.assert_close(actual, observed_history_features(changed_dt, qty, mask), rtol=0, atol=0)
    # Masked pooling follows event order even with interior padding.
    order = torch.tensor([4, 0, 5, 1, 2, 3])
    torch.testing.assert_close(actual, observed_history_features(dt[:, order], qty[:, order], mask[:, order]), rtol=0, atol=0)


def test_temporary_hooks_are_removed_when_a_layer_raises(monkeypatch):
    model = build()
    dt, mask, qty = batch()
    existing = model.encoder.layers[0].register_forward_hook(lambda *_args: None)
    before = hook_counts(model)
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic layer failure")
    monkeypatch.setattr(model.encoder.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic layer failure"):
        extract(model, dt, mask, qty)
    assert hook_counts(model) == before
    existing.remove()


def test_frozen_and_input_contracts():
    model = build()
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
    invalid = qty.clone()
    invalid[0, -2] = float("nan")
    with pytest.raises(ValueError, match="Observed inputs must be finite"):
        extract(model, dt, mask, invalid)
    invalid = qty.clone()
    invalid[0, -1] = -1
    before = hook_counts(model)
    with pytest.raises(ValueError, match="Target quantities must be nonnegative"):
        extract(model, dt, mask, invalid)
    assert before == hook_counts(model)
    with pytest.raises(ValueError, match="original or separate-key"):
        extract(build("titantpp_weighted_static_memory"), dt, mask, qty)
