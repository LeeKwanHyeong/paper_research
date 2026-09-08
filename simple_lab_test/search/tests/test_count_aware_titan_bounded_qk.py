"""Local synthetic contracts for bounded causal Q/K and unchanged causal V."""

from __future__ import annotations

import copy
import io

import pytest
import torch

from models.TPPs.CountAwareTPP import CountAwareTitanTPP, TITAN_MEMORY_MODE_STATIC_HARD
from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_BACKBONE, CAUSAL_QKV_KERNEL_KEYS,
    CausalQKVMemoryAttention, CountAwareTitanCausalQKVTPP,
)
from models.TPPs.CountAwareTitanBoundedQK import (
    BOUNDED_QK_BACKBONE, BOUNDED_QK_CONTRACT_ID, BOUNDED_QK_EPSILON,
    BOUNDED_QK_KERNEL_KEYS, BOUNDED_QK_ROLE,
    BoundedQKMemoryAttention, CountAwareTitanBoundedQKTPP,
    bounded_qk_metadata, validate_bounded_qk_checkpoint,
)
from models.Titan.common.memory import MemoryAttention
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_kwargs(hidden_dim=16):
    return dict(
        hidden_dim=hidden_dim, train_log_mean=1.5, max_seq_len=10,
        quantity_variant="count_only_log_regression", lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
    )


def build_triplet(seed=42, device="cpu", hidden_dim=16):
    torch.manual_seed(seed)
    baseline = CountAwareTitanTPP(
        **model_kwargs(hidden_dim), memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
    )
    expected_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    full = CountAwareTitanCausalQKVTPP(**model_kwargs(hidden_dim))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.manual_seed(seed)
    candidate = CountAwareTitanBoundedQKTPP(**model_kwargs(hidden_dim))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    return tuple(model.to(device) for model in (baseline, full, candidate))


def sample_batch(device="cpu"):
    return (
        torch.tensor([[0., 1., 2., 4., 8.], [0., .5, 3., 6., 0.]], device=device),
        torch.tensor([[2., 5., 9., 12., 20.], [1., 7., 13., 21., 0.]], device=device),
        torch.tensor([[True] * 5, [True] * 4 + [False]], device=device),
    )


def kernels(model):
    attention = model.bounded_qk_attention
    return attention.bounded_q_kernel, attention.bounded_k_kernel, attention.bounded_v_kernel


def objective(model, dts, quantities, mask):
    return target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hidden_dim", [16, 64])
def test_zero_initialization_preserves_b_full_outputs_common_and_kernel_gradients(device, hidden_dim):
    baseline, full, candidate = build_triplet(device=device, hidden_dim=hidden_dim)
    baseline_state, candidate_state = baseline.state_dict(), candidate.state_dict()
    assert set(candidate_state) == set(baseline_state) | set(BOUNDED_QK_KERNEL_KEYS)
    for name, tensor in baseline_state.items():
        assert torch.equal(tensor, candidate_state[name]), name
    batch = sample_batch(device)
    results = []
    for model in (baseline, full, candidate):
        model.train()
        torch.manual_seed(719)
        outputs = objective(model, *batch)
        outputs["joint_loss"].sum().backward()
        results.append(outputs)
    for name in results[0]:
        assert torch.equal(results[0][name], results[1][name]), name
        assert torch.equal(results[0][name], results[2][name]), name
    for name, parameter in baseline.named_parameters():
        for other in (full, candidate):
            actual = dict(other.named_parameters())[name].grad
            if parameter.grad is None:
                assert actual is None, name
            else:
                assert torch.equal(parameter.grad, actual), name
    full_parameters, candidate_parameters = dict(full.named_parameters()), dict(candidate.named_parameters())
    for old_name, new_name in zip(CAUSAL_QKV_KERNEL_KEYS, BOUNDED_QK_KERNEL_KEYS):
        expected, actual = full_parameters[old_name].grad, candidate_parameters[new_name].grad
        assert torch.equal(expected, actual), new_name
        assert torch.isfinite(actual).all()
        assert (actual.abs().sum(dim=1) > 0).all(), new_name


def test_parameter_count_second_layer_and_static_memory_structure():
    baseline, full, candidate = build_triplet(hidden_dim=64)
    count = lambda model: sum(p.numel() for p in model.parameters())
    assert count(candidate) == count(full) == count(baseline) + 576
    assert candidate.additional_parameter_count == 576
    assert isinstance(candidate.encoder.layers[0].attn, BoundedQKMemoryAttention)
    assert type(candidate.encoder.layers[1].attn) is MemoryAttention
    assert tuple(candidate.lmm.mem.shape) == (1, 64, 64)
    assert [tuple(p.shape) for p in kernels(candidate)] == [(3, 64)] * 3
    assert all(torch.count_nonzero(p) == 0 for p in kernels(candidate))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_per_head_bound_zero_identity_and_gradient(dtype):
    _, _, candidate = build_triplet()
    attention = candidate.bounded_qk_attention
    torch.manual_seed(805)
    p = (torch.randn(2, 5, 16, dtype=dtype) * 3).requires_grad_()
    r = (torch.randn_like(p) * 1e4).requires_grad_()
    bounded = attention.bound_qk_residual(p, r)
    p_head, bounded_head = p.reshape(2, 5, 4, 4), bounded.reshape(2, 5, 4, 4)
    limit = (p_head.square().mean(-1) + BOUNDED_QK_EPSILON).sqrt()
    assert (bounded_head.square().mean(-1).sqrt() <= limit * (1 + 1e-6)).all()
    # Alter a different head: this head's normalization cannot change.
    altered = p.detach().clone()
    altered[..., 4:] *= 1e6
    assert torch.equal(bounded[..., :4], attention.bound_qk_residual(altered, r)[..., :4])
    zero = torch.zeros_like(p, requires_grad=True)
    initial = p + attention.bound_qk_residual(p, zero)
    assert torch.equal(initial, p)
    initial.sum().backward()
    assert torch.equal(p.grad, torch.ones_like(p))
    assert torch.equal(zero.grad, torch.ones_like(zero))
    bounded.sum().backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(r.grad).all()
    # The projection scale remains differentiable when the residual is nonzero.
    assert not torch.equal(p.grad, torch.ones_like(p))


@pytest.mark.parametrize("p_dtype,r_dtype", [
    (torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32), (torch.float32, torch.float64),
])
def test_promoted_addition_dtype_and_low_precision_norm(p_dtype, r_dtype):
    _, _, candidate = build_triplet()
    attention = candidate.bounded_qk_attention
    p = torch.full((1, 2, 16), 1000., dtype=p_dtype, requires_grad=True)
    r = torch.full((1, 2, 16), 3000., dtype=r_dtype, requires_grad=True)
    result = attention.bound_qk_residual(p, r)
    assert result.dtype == torch.result_type(p, r)
    assert torch.isfinite(result).all()
    assert result.float().abs().max() < 1001.
    result.sum().backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(r.grad).all()
    zero = torch.zeros_like(r)
    assert torch.equal(p + attention.bound_qk_residual(p, zero), p + zero)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v_operation_is_bitwise_full_and_projection_mask_dtype_is_preserved(masked, dtype):
    _, full, candidate = build_triplet()
    full_attention, attention = full.causal_qkv_attention, candidate.bounded_qk_attention
    for old, new in zip(
        (full_attention.causal_q_kernel, full_attention.causal_k_kernel, full_attention.causal_v_kernel),
        kernels(candidate),
    ):
        old.data.normal_(0, .3)
        new.data.copy_(old)
    projections = tuple(torch.randn(2, 5, 16).to(dtype) for _ in range(3))
    mask = torch.tensor([[True] * 5, [False, False, True, True, True]]) if masked else None
    expected = full_attention._adapt_event_projections(*projections, mask, input_dtype=torch.float32)
    actual = attention._adapt_event_projections(*projections, mask, input_dtype=torch.float32)
    assert torch.equal(actual[2], expected[2])
    assert actual[2].dtype == expected[2].dtype
    # Reconstruct the pre-hook FULL formula; mask was in input x dtype.
    causal_v = projections[2] if mask is None else projections[2] * mask.float().unsqueeze(-1)
    reference = projections[2] + CausalQKVMemoryAttention.causal_depthwise_residual(
        causal_v, full_attention.causal_v_kernel,
    )
    assert torch.equal(reference, actual[2])


def test_persistent_kv_are_appended_without_adaptation():
    _, _, candidate = build_triplet()
    attention = candidate.bounded_qk_attention
    attention.eval()
    for parameter in kernels(candidate):
        parameter.data.fill_(10.)
    captured = []
    original = attention._split_heads

    def capture(value):
        captured.append(value.detach().clone())
        return original(value)

    attention._split_heads = capture
    try:
        attention(torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool))
    finally:
        attention._split_heads = original
    expected = attention.persistent_mem.expand(2, -1, -1)
    assert torch.equal(captured[1][:, :16], expected)
    assert torch.equal(captured[2][:, :16], expected)
    assert captured[0].shape[1] == 5


def test_open_route_changes_h1_h2_attention_and_quantity_prediction():
    _, _, candidate = build_triplet(seed=73)
    candidate.eval()
    candidate.quantity_head.weight.data.fill_(.1)
    batch = sample_batch()
    captured = {}

    def hook(name):
        def record(_module, _args, output):
            captured[name] = output.detach().clone()
        return record

    handles = [
        candidate.encoder.layers[0].register_forward_hook(hook("h1")),
        candidate.encoder.layers[1].register_forward_hook(hook("h2")),
        candidate.bounded_qk_attention.drop.register_forward_hook(hook("attention")),
    ]
    try:
        with torch.no_grad():
            initial = objective(candidate, *batch)
            closed = copy.deepcopy(captured)
            for index, parameter in enumerate(kernels(candidate), start=1):
                parameter.copy_(torch.linspace(-.2 * index, .2 * index, parameter.numel()).reshape_as(parameter))
            opened = objective(candidate, *batch)
        for name in ("h1", "h2", "attention"):
            assert not torch.equal(closed[name], captured[name]), name
        assert not torch.equal(initial["pred_qty"], opened["pred_qty"])
        assert torch.count_nonzero(captured["h1"][~batch[2]]) == 0
        assert torch.count_nonzero(captured["h2"][~batch[2]]) == 0
    finally:
        for handle in handles:
            handle.remove()


def test_prefix_future_target_padding_and_batch_independence():
    _, _, candidate = build_triplet(seed=91)
    candidate.eval()
    for parameter in kernels(candidate):
        parameter.data.fill_(.2)
    dts, quantities, mask = sample_batch()
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    changed_dts[:, 3:] = torch.tensor([99., 999.])
    changed_quantities[:, 3:] = torch.tensor([777., 7777.])
    with torch.no_grad():
        reference = candidate.encode(dts, quantities, mask)
        changed = candidate.encode(changed_dts, changed_quantities, mask)
        torch.testing.assert_close(reference[:, :3], changed[:, :3], rtol=0, atol=1e-6)
        padded_dts, padded_qty = dts.clone(), quantities.clone()
        padded_dts[~mask], padded_qty[~mask] = 1e10, 1e12
        assert torch.equal(reference, candidate.encode(padded_dts, padded_qty, mask))
        alone = candidate.encode(dts[:1], quantities[:1], mask[:1])
        torch.testing.assert_close(reference[:1], alone, rtol=0, atol=1e-6)
        expected = objective(candidate, dts, quantities, mask)
        target_dts, target_qty = dts.clone(), quantities.clone()
        target_positions = mask.sum(1) - 1
        target_dts[torch.arange(2), target_positions] = 333.
        target_qty[torch.arange(2), target_positions] = 77777.
        actual = objective(candidate, target_dts, target_qty, mask)
        assert torch.equal(expected["pred_qty"], actual["pred_qty"])
        assert not torch.equal(expected["true_qty"], actual["true_qty"])


def test_gradient_learned_kernels_change_gain_search_scores_and_prediction():
    _, _, candidate = build_triplet(seed=203)
    batch = sample_batch()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    candidate.train()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        objective(candidate, *batch)["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
        optimizer.step()
    candidate.eval()
    captured = {}

    def record_projection(_module, _inputs, output):
        captured["projection"] = output.detach().clone()

    def record_memory(_module, inputs):
        captured["memory_input"] = inputs[0].detach().clone()

    attention = candidate.bounded_qk_attention
    handles = [
        attention.qkv.register_forward_hook(record_projection),
        candidate.lmm.register_forward_pre_hook(record_memory),
    ]
    try:
        with torch.no_grad():
            opened = objective(candidate, *batch)
            _, opened_trace = candidate.lmm.retrieve(captured["memory_input"])
            q, k, _ = captured["projection"].chunk(3, dim=-1)
            for projection, kernel in ((q, attention.bounded_q_kernel), (k, attention.bounded_k_kernel)):
                residual = attention.causal_depthwise_residual(
                    projection * batch[2].unsqueeze(-1), kernel,
                )
                bounded = attention.bound_qk_residual(projection, residual)
                assert not torch.equal(bounded, residual)
                assert torch.isfinite(bounded).all()
                limited_rms = bounded.reshape(2, 5, 4, 4).square().mean(-1).sqrt()
                limit = (projection.reshape(2, 5, 4, 4).square().mean(-1) + BOUNDED_QK_EPSILON).sqrt()
                assert (limited_rms <= limit * (1 + 1e-6)).all()
            candidate.reset_bounded_qk_identity()
            closed = objective(candidate, *batch)
            _, closed_trace = candidate.lmm.retrieve(captured["memory_input"])
        assert not torch.equal(opened_trace["topk_similarity"], closed_trace["topk_similarity"])
        assert not torch.equal(opened["pred_qty"], closed["pred_qty"])
    finally:
        for handle in handles:
            handle.remove()


def payload(candidate, include_state=True):
    result = dict(
        backbone=BOUNDED_QK_BACKBONE, variant="count_only_log_regression",
        evaluation_scope="validation_only", held_out_test_evaluated=False,
        encoder_config={**bounded_qk_metadata(candidate.hidden_dim), "time_head": candidate.time_head_contract()},
    )
    if include_state:
        result["model_state_dict"] = candidate.state_dict()
    else:
        result.update(best_epoch=17, checkpoint_state_sha256="a" * 64, checkpoint_file_sha256="b" * 64)
    return result


def test_checkpoint_routing_metadata_and_distinct_keys():
    baseline, full, candidate = build_triplet()
    expected = payload(candidate)
    assert candidate.contract_id == BOUNDED_QK_CONTRACT_ID
    assert expected["encoder_config"]["model_role"] == BOUNDED_QK_ROLE
    assert validate_bounded_qk_checkpoint(expected, BOUNDED_QK_BACKBONE)
    assert validate_bounded_qk_checkpoint(payload(candidate, False), BOUNDED_QK_BACKBONE)
    for other_route in ("titantpp", CAUSAL_QKV_BACKBONE):
        with pytest.raises(ValueError, match="cannot be relabelled"):
            validate_bounded_qk_checkpoint(expected, other_route)
    for other in (baseline, full):
        with pytest.raises(RuntimeError):
            candidate.load_state_dict(other.state_dict(), strict=True)
        forged = payload(candidate)
        forged["model_state_dict"] = other.state_dict()
        with pytest.raises(ValueError, match="kernels are missing or invalid"):
            validate_bounded_qk_checkpoint(forged, BOUNDED_QK_BACKBONE)
    for field, wrong in (("bounded_qk_epsilon", 1e-6), ("bounded_qk_rho", .5), ("bounded_qk_detach_projection_rms", True)):
        changed = copy.deepcopy(expected)
        changed["encoder_config"][field] = wrong
        with pytest.raises(ValueError, match="routing metadata mismatch"):
            validate_bounded_qk_checkpoint(changed, BOUNDED_QK_BACKBONE)
    invalid = copy.deepcopy(expected)
    invalid["model_state_dict"][BOUNDED_QK_KERNEL_KEYS[0]][0, 0] = float("nan")
    with pytest.raises(ValueError, match="kernels are missing or invalid"):
        validate_bounded_qk_checkpoint(invalid, BOUNDED_QK_BACKBONE)


def test_explicit_b_initialization_is_strict_and_failures_are_atomic():
    baseline, full, candidate = build_triplet(seed=103)
    for parameter in kernels(candidate):
        parameter.data.fill_(.5)
    before = copy.deepcopy(candidate.state_dict())
    incomplete = dict(baseline.state_dict())
    incomplete.pop("quantity_head.bias")
    for invalid in (incomplete, full.state_dict(), candidate.state_dict()):
        with pytest.raises(RuntimeError):
            candidate.load_hard_lmm_state_dict(invalid)
        for name, tensor in candidate.state_dict().items():
            assert torch.equal(tensor, before[name]), name
    candidate.load_hard_lmm_state_dict(baseline.state_dict())
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(candidate.state_dict()[name], tensor)
    assert all(torch.count_nonzero(parameter) == 0 for parameter in kernels(candidate))


@pytest.mark.parametrize("device", DEVICES)
def test_operating_extremes_finite_and_optimizer_roundtrip_next_step(device):
    _, _, candidate = build_triplet(seed=117, device=device)
    batch = (
        torch.tensor([[0., 1e-8, 1e4, 1e8]], device=device),
        torch.tensor([[0., 1., 1e6, 1e10]], device=device),
        torch.ones(1, 4, dtype=torch.bool, device=device),
    )
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)

    def step(model, optim):
        model.train()
        torch.manual_seed(951)
        optim.zero_grad(set_to_none=True)
        outputs = objective(model, *batch)
        outputs["joint_loss"].mean().backward()
        assert all(torch.isfinite(value).all() for value in outputs.values())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())

    step(candidate, optimizer)
    assert all(torch.count_nonzero(p) > 0 for p in kernels(candidate))
    buffer = io.BytesIO()
    torch.save(dict(model=candidate.state_dict(), optimizer=optimizer.state_dict()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location=device, weights_only=True)
    _, _, restored = build_triplet(seed=999, device=device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer.load_state_dict(saved["optimizer"])
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name
    step(candidate, optimizer)
    step(restored, restored_optimizer)
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name
    original_state, restored_state = optimizer.state_dict(), restored_optimizer.state_dict()
    assert original_state["param_groups"] == restored_state["param_groups"]
    for parameter_id, state in original_state["state"].items():
        for name, value in state.items():
            actual = restored_state["state"][parameter_id][name]
            assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
