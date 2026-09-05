"""Synthetic geometry, causal routing and training checks; no dataset access."""

import io
import math
import os

import pytest
import torch
from torch.nn import functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.Titan.common.elapsed_age import (
    ELAPSED_AGE_BACKBONE as CANDIDATE, ELAPSED_AGE_BETA_KEY, causal_elapsed_age_geometry,
)
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from models.Titan.common.memory import MemoryAttention
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
from paper.scripts.hard_lmm_temporal_features import observed_features


DEVICE = os.environ.get("HARD_ELAPSED_AGE_TEST_DEVICE", "cpu")
TOL = 1e-6 if DEVICE.startswith("cuda") else 0.
VARIANT = "count_only_log_regression"


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(backbone=CANDIDATE, *, dim=16, max_len=8):
    torch.manual_seed(42)
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=dim, train_log_mean=1.5, max_seq_len=max_len,
        quantity_variant=VARIANT, lambda_tail=0., time_head_mode="legacy_clamped_rmtpp",
    )
    return model.to(DEVICE), metadata


def batch():
    dts = torch.tensor([[9., 1., 4., 2., 3., 1.], [8., 5., 1., 3., 2., 0.]], device=DEVICE)
    qty = torch.tensor([[1., 5., 3., 8., 2., 12.], [2., 6., 3., 1., 7., 0.]], device=DEVICE)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=torch.bool, device=DEVICE)
    return dts, qty, mask


def outputs(model, data=None):
    dts, qty, mask = batch() if data is None else data
    return target_outputs(model, dts, mask, qty, lambda_log_qty=1.)


def independent_geometry(dts, observed):
    result = torch.zeros(*dts.shape, dts.size(1), dtype=torch.float64)
    for b in range(dts.size(0)):
        positions = observed[b].nonzero().flatten().tolist()
        for rank, i in enumerate(positions):
            gaps = [max(float(dts[b, k]), 0.) for k in positions[1:rank + 1]]
            span = math.fsum(gaps)
            if rank < 2 or span <= 0:
                continue
            for key_rank, j in enumerate(positions[:rank + 1]):
                result[b, i, j] = math.fsum(gaps[key_rank:]) / span - (rank - key_rank) / rank
    return result


def test_geometry_matches_independent_formula_and_existing_j():
    dts, qty, mask = batch()
    actual = causal_elapsed_age_geometry(dts, mask)
    expected = independent_geometry(dts.cpu(), mask.cpu())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-6, atol=1e-7)
    for b, length in enumerate(mask.sum(-1).tolist()):
        j = observed_features(dts[b, :length].cpu(), qty[b, :length].cpu())["age_distortion"]
        torch.testing.assert_close(actual[b, length - 1, :length].square().mean().sqrt().cpu(),
                                   torch.tensor(j), rtol=1e-6, atol=1e-7)
    small = torch.tensor([[99., 1., 3.]], device=DEVICE)
    actual = causal_elapsed_age_geometry(small, torch.ones_like(small, dtype=torch.bool))
    torch.testing.assert_close(actual[0, -1], torch.tensor([0., .25, 0.], device=DEVICE), rtol=0, atol=0)


@pytest.mark.parametrize("length", [1, 2, 3, 8])
def test_uniform_zero_span_and_short_histories_are_neutral(length):
    dts = torch.full((2, length), 3., device=DEVICE)
    dts[1].zero_()
    mask = torch.ones_like(dts, dtype=torch.bool)
    assert torch.count_nonzero(causal_elapsed_age_geometry(dts, mask)) == 0
    assert torch.count_nonzero(causal_elapsed_age_geometry(dts, mask & False)) == 0


def test_geometry_scale_first_gap_target_padding_and_prefix_invariants():
    dts, _, mask = batch()
    observed = mask.clone()
    observed[0, 5] = observed[1, 4] = False
    expected = causal_elapsed_age_geometry(dts, mask, observed_mask=observed)
    changed = dts * 1000.
    changed[:, 0] = 8e20
    changed[~observed] = float("nan")
    actual = causal_elapsed_age_geometry(changed, mask, observed_mask=observed)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(actual.triu(1)) == 0
    assert torch.count_nonzero(actual[0, 5]) == 0
    assert torch.count_nonzero(actual[1, :, 4:]) == 0
    for length in (2, 3, 4):
        prefix = causal_elapsed_age_geometry(dts[:, :length], mask[:, :length])
        torch.testing.assert_close(prefix, expected[:, :length, :length], rtol=0, atol=0)
    # Ranks ignore padding; the first observed gap remains excluded.
    holes = torch.tensor([[False, True, False, True, True]], device=DEVICE)
    gaps = torch.tensor([[float("nan"), 99., float("inf"), 1., 3.]], device=DEVICE)
    torch.testing.assert_close(causal_elapsed_age_geometry(gaps, holes)[0, 4, holes[0]],
                               torch.tensor([0., .25, 0.], device=DEVICE), rtol=0, atol=0)


def test_finite_extreme_gaps_and_negative_gap_clamping():
    gaps = torch.tensor([[1e38, 3e38, 1e-30, 3e38], [1., -2., 3., 0.]], device=DEVICE)
    mask = torch.ones_like(gaps, dtype=torch.bool)
    actual = causal_elapsed_age_geometry(gaps, mask)
    assert torch.isfinite(actual).all() and actual.abs().max() <= 1.
    torch.testing.assert_close(actual, causal_elapsed_age_geometry(gaps.clamp_min(0), mask), rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu().double(), independent_geometry(gaps.cpu(), mask.cpu()),
                               rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("case", ["shape", "empty", "mask_dtype", "observed_padding", "nan", "inf"])
def test_invalid_geometry_inputs_fail_closed(case):
    dts = torch.ones(1, 3, device=DEVICE)
    mask = torch.ones_like(dts, dtype=torch.bool)
    kwargs = {}
    if case == "shape":
        dts = dts.flatten()
    elif case == "empty":
        dts, mask = dts[:, :0], mask[:, :0]
    elif case == "mask_dtype":
        mask = mask.float()
    elif case == "observed_padding":
        kwargs["observed_mask"] = mask.clone()
        mask[0, -1] = False
    else:
        dts[0, 1] = float(case)
    with pytest.raises(ValueError):
        causal_elapsed_age_geometry(dts, mask, **kwargs)


@pytest.mark.parametrize("training", [False, True])
def test_zero_beta_common_parameters_rng_and_full_outputs_equal(training):
    reference, _ = build(KEY_VALUE_BACKBONE)
    rng = torch.get_rng_state()
    candidate, metadata = build()
    assert torch.equal(rng, torch.get_rng_state())
    assert set(candidate.state_dict()) - set(reference.state_dict()) == {ELAPSED_AGE_BETA_KEY}
    for name, tensor in reference.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name
    assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in reference.parameters()) == 8
    assert metadata["elapsed_age_parameter_count"] == 8
    assert candidate.encoder.elapsed_age_beta.shape == (2, 4)
    assert torch.count_nonzero(candidate.encoder.elapsed_age_beta) == 0
    for model in (reference, candidate):
        model.train(training)
        with torch.no_grad():
            model.quantity_head.weight.fill_(.05)
    torch.manual_seed(391)
    a = outputs(reference)
    torch.manual_seed(391)
    b = outputs(candidate)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=TOL, atol=TOL)


def test_beta_gradients_and_actual_encoder_retrieval_prediction_response():
    model, _ = build()
    model.eval()
    outputs(model)["joint_loss"].mean().backward()
    gradient = model.encoder.elapsed_age_beta.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    dts, qty, mask = batch()
    with torch.no_grad():
        model.quantity_head.weight.normal_(0., .1)
        before_h = model._encode_base(dts, qty, mask)
        before_r, before_trace = model.lmm.retrieve(before_h)
        before = outputs(model)["pred_qty"]
        model.encoder.elapsed_age_beta.fill_(4.)
        after_h = model._encode_base(dts, qty, mask)
        after_r, after_trace = model.lmm.retrieve(after_h)
        after = outputs(model)["pred_qty"]
    assert not torch.allclose(before_h, after_h)
    assert not torch.allclose(before_r, after_r)
    assert not torch.allclose(before_trace["retrieval_weights"], after_trace["retrieval_weights"])
    assert not torch.allclose(before, after)


def test_open_beta_uniform_geometry_preserves_reference_full_output():
    reference, _ = build(KEY_VALUE_BACKBONE)
    candidate, _ = build()
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        candidate.encoder.elapsed_age_beta.fill_(5.)
        for model in (reference, candidate):
            model.quantity_head.weight.fill_(.05)
    dts, qty, mask = batch()
    dts.fill_(3.)
    dts[:, 0] = 99.
    a, b = outputs(reference, (dts, qty, mask)), outputs(candidate, (dts, qty, mask))
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=TOL, atol=TOL)


def test_open_beta_extreme_finite_full_model_forward_backward():
    model, _ = build()
    with torch.no_grad():
        model.encoder.elapsed_age_beta.fill_(4.)
    dts, qty, mask = batch()
    out = outputs(model, (dts * 1e8, qty * 1e8, mask))
    assert all(torch.isfinite(value).all() for value in out.values())
    out["joint_loss"].mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_open_beta_target_future_padding_series_and_eval_state_isolation():
    model, _ = build()
    model.eval()
    with torch.no_grad():
        model.encoder.elapsed_age_beta.fill_(3.)
        model.quantity_head.weight.normal_(0., .1)
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    dts, qty, mask = batch()
    original = outputs(model)
    changed_dt, changed_q = dts.clone(), qty.clone()
    changed_dt[0, 5], changed_dt[1, 4:] = 1e5, 1e6
    changed_q[0, 5], changed_q[1, 4:] = 1e6, 1e7
    changed = outputs(model, (changed_dt, changed_q, mask))
    torch.testing.assert_close(original["pred_qty"], changed["pred_qty"], rtol=TOL, atol=TOL)
    before, after = model.encode(dts, qty, mask), model.encode(changed_dt, changed_q, mask)
    torch.testing.assert_close(before[:, :4], after[:, :4], rtol=TOL, atol=TOL)
    assert torch.count_nonzero(before[1, -1]) == 0
    for row in range(2):
        alone = model.encode(dts[row:row+1], qty[row:row+1], mask[row:row+1])
        torch.testing.assert_close(before[row:row+1], alone, rtol=1e-5, atol=1e-6)
    left = tuple(t.clone() for t in (dts, qty, mask))
    for tensor in left:
        tensor[1] = tensor[1].roll(1)
    torch.testing.assert_close(outputs(model, left)["pred_qty"], original["pred_qty"], rtol=TOL, atol=TOL)
    assert all(torch.equal(value, snapshot[name]) for name, value in model.state_dict().items())


def test_attention_preserves_persistent_scores_and_adds_only_event_bias(monkeypatch):
    torch.manual_seed(11)
    attention = MemoryAttention(16, 4, 0, 3, dropout=0.).to(DEVICE).eval()
    x = torch.randn(2, 5, 16, device=DEVICE)
    bias = torch.randn(2, 4, 5, 5, device=DEVICE)
    scores = []
    original_softmax = F.softmax

    def capture(input, *args, **kwargs):
        scores.append(input.detach().clone())
        return original_softmax(input, *args, **kwargs)

    monkeypatch.setattr(F, "softmax", capture)
    attention(x)
    attention(x, event_attention_bias=bias)
    torch.testing.assert_close(scores[0][..., :3], scores[1][..., :3], rtol=0, atol=0)
    valid = torch.ones(5, 5, dtype=torch.bool, device=DEVICE).tril().expand(2, 4, -1, -1)
    torch.testing.assert_close((scores[1][..., 3:] - scores[0][..., 3:])[valid], bias[valid],
                               rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_preserves_zero_beta_score_dtype_and_finite_gradients(dtype, monkeypatch):
    if DEVICE == "cpu" and dtype == torch.float16:
        pytest.skip("CPU autocast coverage uses bfloat16")
    if DEVICE.startswith("cuda") and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    reference, _ = build(KEY_VALUE_BACKBONE)
    candidate, _ = build()
    reference.eval()
    candidate.eval()
    captured = []
    original_softmax = F.softmax

    def capture(input, *args, **kwargs):
        if input.ndim == 4:
            captured.append(input.dtype)
        return original_softmax(input, *args, **kwargs)

    monkeypatch.setattr(F, "softmax", capture)
    with torch.autocast(device_type=torch.device(DEVICE).type, dtype=dtype):
        a, b = outputs(reference), outputs(candidate)
    assert captured and set(captured) == {dtype}
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        assert torch.isfinite(b[name]).all()
    b["joint_loss"].mean().backward()
    assert torch.isfinite(candidate.encoder.elapsed_age_beta.grad).all()
    assert candidate.encoder.elapsed_age_beta.grad.abs().sum() > 0


def test_checkpoint_optimizer_moments_and_next_training_step_replay():
    model, metadata = build()
    optimizer = build_optimizer(model, lr=.001)

    def step(m, opt):
        m.train()
        opt.zero_grad(set_to_none=True)
        outputs(m)["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        opt.step()

    step(model, optimizer)
    beta = model.encoder.elapsed_age_beta
    assert len(optimizer.param_groups) == 1
    assert beta.abs().sum() > 0
    for key in ("exp_avg", "exp_avg_sq"):
        moment = optimizer.state[beta][key]
        assert torch.isfinite(moment).all() and moment.abs().sum() > 0
    payload = {"backbone": CANDIDATE, "variant": VARIANT, "encoder_config": metadata,
               "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True, map_location=DEVICE)
    validate_checkpoint_route(saved, CANDIDATE)
    restored, _ = build()
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    restored_optimizer = build_optimizer(restored, lr=.001)
    restored_optimizer.load_state_dict(saved["optimizer_state_dict"])
    # map_location moves the restored scalar counter to CUDA, while fresh
    # noncapturable AdamW keeps it on CPU. Compare its exact value and dtype.
    torch.testing.assert_close(optimizer.state[beta]["step"].cpu(),
                               restored_optimizer.state[restored.encoder.elapsed_age_beta]["step"].cpu(),
                               rtol=0, atol=0)
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(optimizer.state[beta][key],
                                   restored_optimizer.state[restored.encoder.elapsed_age_beta][key], rtol=0, atol=0)
    torch.manual_seed(501)
    step(model, optimizer)
    torch.manual_seed(501)
    step(restored, restored_optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=TOL, atol=TOL)


@pytest.mark.parametrize("length", [64, 256])
def test_h64_training_shaped_forward_backward_adamw_step(length):
    size = 128 if DEVICE.startswith("cuda") else 2
    model, _ = build(dim=64, max_len=length)
    model.train()
    optimizer = build_optimizer(model, lr=.001)
    i = torch.arange(length, device=DEVICE).expand(size, -1)
    row = torch.arange(size, device=DEVICE).unsqueeze(1)
    dts = ((i * 7 + row * 3) % 13 + 1).float()
    qty = ((i * 3 + row) % 11 + 1).float()
    mask = torch.ones_like(dts, dtype=torch.bool)
    out = outputs(model, (dts, qty, mask))
    assert all(torch.isfinite(value).all() for value in out.values())
    out["joint_loss"].mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    beta = model.encoder.elapsed_age_beta
    assert beta.grad.abs().sum() > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    assert beta.abs().sum() > 0
    assert optimizer.state[beta]["step"].item() == 1
    assert torch.isfinite(optimizer.state[beta]["exp_avg"]).all()
