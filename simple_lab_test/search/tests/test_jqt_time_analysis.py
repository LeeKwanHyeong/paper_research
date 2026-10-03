"""CPU-only synthetic tests for per-sample legacy time diagnostics."""
import math
import json

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import SharedTimeCountModel
from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch
from paper.scripts.jqt_time_analysis import inspect_legacy_time_batch, summarize_time_rows
from paper.scripts.time_quantity_diagnostic import task_outputs


def batch(dtype=torch.float32):
    return (
        torch.tensor([[0., 0., 1., 2.], [0., 1., 1., 3.]], dtype=dtype),
        torch.tensor([[False, False, True, True], [False, True, True, True]]),
        torch.tensor([[0., 0., 2., 4.], [0., 1., 2., 3.]], dtype=dtype),
    )


def actual_model():
    torch.manual_seed(42)
    model, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=1.0, max_seq_len=8, time_intercept_limit=300.0)
    return model.eval()


def test_loss_parity_and_inactive_quantity_head():
    model = actual_model()
    dts, mask, quantities = batch()
    expected = task_outputs(model, dts, mask, quantities, "time_only")["time_loss"]
    model.quantity_outputs = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inactive quantity head called"))
    result = inspect_legacy_time_batch(model, dts, mask, quantities, "time_only")
    assert torch.equal(result["original_legacy_loss"], expected)
    assert torch.equal(result["history_length"], torch.tensor([1, 2]))
    assert torch.max(torch.abs(result["residual_reconstruction_error"])).item() < 1e-6


def _time_hidden(model, dts, mask, quantities):
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    ids = torch.arange(dts.size(0))
    target = lengths - 1
    history_quantities = quantities.clone()
    history_quantities[ids, target] = 0.0
    write_mask = mask.clone()
    write_mask[ids, target] = False
    time_encoded, _ = model.encode_task_states(
        dts, history_quantities, mask, memory_write_mask=write_mask,
    )
    return time_encoded[ids, lengths - 2]


def test_target_quantity_padding_and_duration_are_causally_excluded_and_read_only():
    model = actual_model()
    dts, mask, quantities = batch()
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    grads_before = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    baseline_hidden = _time_hidden(model, dts, mask, quantities)
    baseline = inspect_legacy_time_batch(model, dts, mask, quantities, "time_only")

    changed_quantity_padding = quantities.clone()
    changed_quantity_padding[0, :2] = torch.tensor([99.0, 77.0])
    changed_quantity_padding[:, -1] = torch.tensor([123.0, 456.0])
    quantity_hidden = _time_hidden(model, dts, mask, changed_quantity_padding)
    quantity_result = inspect_legacy_time_batch(model, dts, mask, changed_quantity_padding, "time_only")
    assert torch.equal(quantity_hidden, baseline_hidden)
    assert torch.equal(quantity_result["raw_intercept"], baseline["raw_intercept"])
    assert torch.equal(quantity_result["original_legacy_loss"], baseline["original_legacy_loss"])

    changed_duration = dts.clone()
    changed_duration[:, -1] = torch.tensor([8.0, 9.0])
    duration_hidden = _time_hidden(model, changed_duration, mask, quantities)
    duration_result = inspect_legacy_time_batch(model, changed_duration, mask, quantities, "time_only")
    assert torch.equal(duration_hidden, baseline_hidden)
    assert torch.equal(duration_result["raw_intercept"], baseline["raw_intercept"])
    assert not torch.equal(duration_result["original_legacy_loss"], baseline["original_legacy_loss"])
    assert model.training is False
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name]), name
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.grad, grads_before[name]), name


class ToyLegacy(SharedTimeCountModel):
    def __init__(self):
        super().__init__(hidden_dim=1, train_log_mean=1.0, time_intercept_limit=300.0)

    def encode(self, dts, history_quantities, mask):
        return torch.zeros((*dts.shape, 1), dtype=dts.dtype, device=dts.device)


def toy(bias, w_raw=0.0):
    model = ToyLegacy().double().eval()
    with torch.no_grad():
        model.v_t.weight.zero_()
        model.b_t.fill_(bias)
        model.w_raw.fill_(w_raw)
    return model


def test_intercept_and_wd_clamp_boundaries_in_float64():
    dts, mask, quantities = batch(torch.float64)
    dts = dts.clone()
    dts[0, -1], dts[1, -1] = 10.0, 11.0
    result = inspect_legacy_time_batch(toy(300.0, math.log(math.expm1(1.0 - 1e-3))), dts, mask, quantities, "joint")
    assert result["intercept_cap_exact_boundary"].all()
    assert not result["intercept_cap_hit"].any()
    assert result["w_dt_cap_exact_boundary"].any()
    assert result["w_dt_cap_hit"].any()
    below = inspect_legacy_time_batch(toy(299.0), dts, mask, quantities, "time_only")
    assert not below["intercept_cap_hit"].any()
    above = inspect_legacy_time_batch(toy(301.0), dts, mask, quantities, "time_only")
    assert above["intercept_cap_hit"].all()


def test_fp32_boundary_overflow_is_rejected_and_wd_clamp_keeps_w_sensitivity():
    dts, mask, quantities = batch()
    with pytest.raises(ValueError, match="Nonfinite"):
        inspect_legacy_time_batch(toy(300.0).float(), dts, mask, quantities, "time_only")
    w_raw = torch.tensor(0.0, requires_grad=True)
    duration = torch.tensor(100.0)
    capped_wd = torch.clamp((torch.nn.functional.softplus(w_raw) + 1e-3) * duration, max=10.0)
    integral = torch.exp(torch.tensor(0.0)) / (torch.nn.functional.softplus(w_raw) + 1e-3) * torch.expm1(capped_wd)
    integral.backward()
    assert capped_wd.item() == 10.0 and w_raw.grad.item() != 0.0


def test_summary_preserves_empty_bins_and_separates_signed_negative_loss():
    rows = {
        "original_legacy_loss": torch.tensor([-2.0, 3.0, 7.0]), "duration": torch.tensor([1.0, 2.0, 200.0]),
        "intercept_cap_hit": torch.tensor([False, True, False]), "intercept_cap_exact_boundary": torch.tensor([False, False, True]),
        "w_dt_cap_hit": torch.tensor([False, False, True]), "w_dt_cap_exact_boundary": torch.tensor([False, True, False]),
        "w": torch.tensor([1.0, 1.0, 1.0]),
        "raw_intercept": torch.tensor([1.0, 2.0, 4.0]), "raw_w_dt": torch.tensor([.1, .2, .4]),
        "term_linear": torch.tensor([-1.1, -2.2, -4.4]), "term_integral": torch.tensor([.1, .2, .4]),
        "residual_reconstruction_error": torch.tensor([0.0, -0.25, 0.5]),
    }
    summary = summarize_time_rows(rows)
    assert summary["signed_loss_sum"] == 8.0 and summary["negative_loss_sum"] == -2.0
    assert summary["duration_bins"][2]["n"] == 0
    assert sum(row["mean_loss_contribution"] for row in summary["duration_bins"]) == pytest.approx(summary["mean_loss"])
    assert summary["term_statistics"]["raw_intercept"]["p50"] == 2.0
    assert summary["term_statistics"]["term_integral"]["p99"] > .39
    assert summary["residual_reconstruction_error_max_abs"] == .5
    json.dumps(summary, allow_nan=False)
