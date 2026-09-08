"""Contracts for the level-preserving history-confidence Hard-LMM route."""

from __future__ import annotations

import copy
import io

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import CountAwareTitanTPP, TITAN_MEMORY_MODE_STATIC_HARD
from models.TPPs.CountAwareTitanBoundedQK import BOUNDED_QK_BACKBONE
from models.TPPs.CountAwareTitanCausalQKV import CAUSAL_QKV_BACKBONE
from models.TPPs.CountAwareTitanLevelHistoryQKV import (
    LEVEL_HISTORY_QKV_BACKBONE,
    LEVEL_HISTORY_QKV_CONTRACT_ID,
    LEVEL_HISTORY_QKV_EPSILON,
    LEVEL_HISTORY_QKV_FOREIGN_STATE_KEYS,
    LEVEL_HISTORY_QKV_KERNEL_KEYS,
    LEVEL_HISTORY_QKV_ROLE,
    CountAwareTitanLevelHistoryQKVTPP,
    LevelHistoryQKVMemoryAttention,
    level_history_qkv_metadata,
    validate_level_history_qkv_checkpoint,
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
        hidden_dim=hidden_dim,
        train_log_mean=1.5,
        max_seq_len=10,
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
    )


def build_pair(seed=42, device="cpu", hidden_dim=16):
    torch.manual_seed(seed)
    baseline = CountAwareTitanTPP(
        **model_kwargs(hidden_dim), memory_mode=TITAN_MEMORY_MODE_STATIC_HARD
    )
    expected_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    candidate = CountAwareTitanLevelHistoryQKVTPP(**model_kwargs(hidden_dim))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    return baseline.to(device), candidate.to(device)


def sample_batch(device="cpu"):
    return (
        torch.tensor(
            [[0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 3.0, 6.0, 0.0]],
            device=device,
        ),
        torch.tensor(
            [[2.0, 5.0, 9.0, 12.0, 20.0], [1.0, 7.0, 13.0, 21.0, 0.0]],
            device=device,
        ),
        torch.tensor(
            [[True] * 5, [True] * 4 + [False]], device=device
        ),
    )


def kernels(model):
    attention = model.level_history_qkv_attention
    return (
        attention.level_history_q_kernel,
        attention.level_history_k_kernel,
        attention.level_history_v_kernel,
    )


def objective(model, dts, quantities, mask):
    return target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hidden_dim", [16, 64])
def test_zero_initialization_preserves_b_outputs_gradients_and_rng(device, hidden_dim):
    baseline, candidate = build_pair(device=device, hidden_dim=hidden_dim)
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert set(candidate_state) == set(baseline_state) | set(
        LEVEL_HISTORY_QKV_KERNEL_KEYS
    )
    for name, tensor in baseline_state.items():
        assert torch.equal(tensor, candidate_state[name]), name

    results = []
    for model in (baseline, candidate):
        model.train()
        torch.manual_seed(719)
        outputs = objective(model, *sample_batch(device))
        outputs["joint_loss"].sum().backward()
        results.append(outputs)
    for name in results[0]:
        assert torch.equal(results[0][name], results[1][name]), name
    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in baseline.named_parameters():
        actual = candidate_parameters[name].grad
        if parameter.grad is None:
            assert actual is None, name
        else:
            assert torch.equal(parameter.grad, actual), name
    for name in LEVEL_HISTORY_QKV_KERNEL_KEYS:
        gradient = candidate_parameters[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        assert (gradient.abs().sum(dim=1) > 0).all(), name


def test_parameter_count_second_layer_and_static_memory_are_preserved():
    baseline, candidate = build_pair(hidden_dim=64)
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())
    assert count(candidate) == count(baseline) + 384
    assert candidate.additional_parameter_count == 384
    assert isinstance(
        candidate.encoder.layers[0].attn, LevelHistoryQKVMemoryAttention
    )
    assert type(candidate.encoder.layers[1].attn) is MemoryAttention
    assert tuple(candidate.lmm.mem.shape) == (1, 64, 64)
    assert [tuple(parameter.shape) for parameter in kernels(candidate)] == [
        (2, 64)
    ] * 3
    assert all(torch.count_nonzero(parameter) == 0 for parameter in kernels(candidate))


def test_zero_identity_preserves_hidden_lmm_input_memory_and_top4_trace():
    baseline, candidate = build_pair(seed=75)
    baseline.eval()
    candidate.eval()
    dts, quantities, mask = sample_batch()
    captured = {}

    def record(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach().clone()

        return hook

    handles = [
        baseline.lmm.register_forward_pre_hook(record("baseline")),
        candidate.lmm.register_forward_pre_hook(record("candidate")),
    ]
    try:
        with torch.no_grad():
            baseline_states = baseline.encode_task_states(dts, quantities, mask)
            candidate_states = candidate.encode_task_states(dts, quantities, mask)
            baseline_residual, baseline_trace = baseline.lmm.retrieve(
                captured["baseline"]
            )
            candidate_residual, candidate_trace = candidate.lmm.retrieve(
                captured["candidate"]
            )
        for left, right in zip(baseline_states, candidate_states):
            assert torch.equal(left, right)
        assert torch.equal(captured["baseline"], captured["candidate"])
        assert torch.equal(baseline.lmm.mem, candidate.lmm.mem)
        assert torch.equal(baseline_residual, candidate_residual)
        for name in ("prototype_indices", "topk_similarity"):
            assert torch.equal(baseline_trace[name], candidate_trace[name])
    finally:
        for handle in handles:
            handle.remove()


def test_contiguous_and_gapped_history_confidence_and_available_lags():
    projection = torch.arange(7.0).reshape(1, 7, 1).expand(-1, -1, 4)
    contiguous = LevelHistoryQKVMemoryAttention.transition_components(
        projection, None
    )
    torch.testing.assert_close(
        contiguous[-1],
        torch.tensor([[0.0, 0.5, 2 / 3, 0.75, 0.8, 5 / 6, 6 / 7]]),
        rtol=0,
        atol=1e-7,
    )
    long_projection = torch.zeros(1, 15, 4)
    long_confidence = LevelHistoryQKVMemoryAttention.transition_components(
        long_projection, None
    )[-1]
    assert long_confidence[0, 7].item() == pytest.approx(7 / 8)
    assert long_confidence[0, 14].item() == pytest.approx(14 / 15)

    mask = torch.tensor([[False, True, True, False, True, True, True]])
    lag1, lag2, available1, available2, confidence = (
        LevelHistoryQKVMemoryAttention.transition_components(projection, mask)
    )
    assert torch.equal(
        available1,
        torch.tensor([[False, False, True, False, False, True, True]]),
    )
    assert torch.equal(
        available2,
        torch.tensor([[False, False, False, False, False, False, True]]),
    )
    torch.testing.assert_close(
        confidence,
        torch.tensor([[0.0, 0.0, 0.5, 0.5, 1 / 3, 0.5, 0.6]]),
        rtol=0,
        atol=1e-7,
    )
    assert torch.count_nonzero(lag1[~available1]) == 0
    assert torch.count_nonzero(lag2[~available2]) == 0
    assert torch.equal(lag1[available1], torch.ones_like(lag1[available1]))
    assert torch.equal(lag2[available2], torch.ones_like(lag2[available2]))


@pytest.mark.parametrize("padding", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_padding_is_sanitized_before_differences(padding):
    projection = torch.tensor(
        [[[padding] * 4, [padding] * 4, [2.0] * 4, [5.0] * 4]]
    )
    mask = torch.tensor([[False, False, True, True]])
    lag1, lag2, available1, available2, _ = (
        LevelHistoryQKVMemoryAttention.transition_components(projection, mask)
    )
    assert torch.isfinite(lag1).all() and torch.isfinite(lag2).all()
    assert torch.equal(lag1[available1], torch.full((1, 4), 3.0))
    assert torch.count_nonzero(lag1[~available1]) == 0
    assert torch.count_nonzero(lag2) == 0
    assert not available2.any()


def test_constant_level_translation_preserves_the_complete_added_residual():
    _, candidate = build_pair(seed=81)
    attention = candidate.level_history_qkv_attention
    for index, parameter in enumerate(kernels(candidate), start=1):
        parameter.data.normal_(0.0, 0.2 * index)
    mask = torch.tensor(
        [[False, True, True, True, False, True], [True, True, True, True, True, True]]
    )
    projections = tuple(torch.randn(2, 6, 16, dtype=torch.float64) for _ in range(3))
    shifts = tuple(torch.randn(1, 1, 16, dtype=torch.float64) for _ in range(3))
    translated = tuple(
        projection + shift * mask.unsqueeze(-1)
        for projection, shift in zip(projections, shifts)
    )
    before = attention._adapt_event_projections(*projections, mask)
    after = attention._adapt_event_projections(*translated, mask)
    for original, shifted, adapted_original, adapted_shifted in zip(
        projections, translated, before, after
    ):
        torch.testing.assert_close(
            adapted_original - original,
            adapted_shifted - shifted,
            rtol=0,
            atol=1e-12,
        )

    one_event = torch.tensor(
        [[False, False, True, False, False, False]] * projections[0].size(0)
    )
    single = attention._adapt_event_projections(*projections, one_event)
    for projection, adapted in zip(projections, single):
        assert torch.equal(projection, adapted)


def test_constant_projection_has_exact_zero_dc_residual_with_open_kernels():
    _, candidate = build_pair(seed=82)
    attention = candidate.level_history_qkv_attention
    for parameter in kernels(candidate):
        parameter.data.fill_(7.0)
    projection = torch.full((2, 6, 16), 8.0)
    mask = torch.tensor(
        [[False, True, True, True, True, True], [True, True, False, True, True, True]]
    )
    for kernel in kernels(candidate):
        residual, lag1, lag2, available1, available2, _ = (
            attention.transition_residual(projection, kernel, mask)
        )
        assert torch.count_nonzero(lag1) == 0
        assert torch.count_nonzero(lag2) == 0
        assert torch.count_nonzero(residual) == 0
        bounded = attention.bound_transition_residual(
            residual, lag1, lag2, available1, available2
        )
        assert torch.count_nonzero(bounded) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_qk_bound_uses_headwise_transition_rms_and_is_finite(dtype):
    _, candidate = build_pair()
    attention = candidate.level_history_qkv_attention
    projection = torch.randn(2, 6, 16, dtype=dtype) * 4
    mask = torch.tensor([[True] * 6, [False, True, True, True, False, True]])
    kernel = torch.full((2, 16), 1e4, dtype=dtype)
    residual, lag1, lag2, available1, available2, confidence = (
        attention.transition_residual(projection, kernel, mask)
    )
    bounded = attention.bound_transition_residual(
        residual, lag1, lag2, available1, available2
    )
    shape = (2, 6, 4, 4)
    first = lag1.reshape(shape)
    second = lag2.reshape(shape)
    a1 = available1[..., None, None].to(dtype)
    a2 = available2[..., None, None].to(dtype)
    scale_squared = (
        a1 * first.square().mean(-1, keepdim=True)
        + a2 * second.square().mean(-1, keepdim=True)
    ) / (a1 + a2).clamp_min(1) + LEVEL_HISTORY_QKV_EPSILON
    added = bounded.reshape(shape) * confidence[..., None, None]
    actual_rms = added.square().mean(-1, keepdim=True).sqrt()
    limit = scale_squared.sqrt() * confidence[..., None, None]
    assert torch.isfinite(bounded).all()
    assert bool((actual_rms <= limit * (1 + 1e-6)).all())
    residual_squared = residual.reshape(shape).square().mean(-1, keepdim=True)
    expected = (
        residual.reshape(shape)
        * scale_squared.sqrt()
        / (scale_squared + residual_squared).sqrt()
    ).reshape_as(residual)
    torch.testing.assert_close(bounded, expected, rtol=1e-12, atol=1e-12)

    altered_lag1 = lag1.clone()
    altered_lag2 = lag2.clone()
    altered_residual = residual.clone()
    altered_lag1[..., 4:] *= 1e6
    altered_lag2[..., 4:] *= 1e6
    altered_residual[..., 4:] *= 1e6
    isolated = attention.bound_transition_residual(
        altered_residual,
        altered_lag1,
        altered_lag2,
        available1,
        available2,
    )
    assert torch.equal(bounded[..., :4], isolated[..., :4])


def test_unavailable_lag_has_no_gradient_but_available_lag_learns():
    projection = torch.tensor([[[1.0], [3.0], [0.0]]]).expand(-1, -1, 4)
    kernel = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    residual, *_ = LevelHistoryQKVMemoryAttention.transition_residual(
        projection, kernel, mask
    )
    residual.sum().backward()
    assert torch.count_nonzero(kernel.grad[0]) == 4
    assert torch.count_nonzero(kernel.grad[1]) == 0


@pytest.mark.parametrize(
    ("length", "expected_nonzero_rows"),
    [(1, (False, False)), (2, (True, False)), (3, (True, True))],
)
def test_qkv_kernel_gradient_matches_available_history(length, expected_nonzero_rows):
    _, candidate = build_pair(seed=84)
    attention = candidate.level_history_qkv_attention
    base = torch.arange(1, length + 1, dtype=torch.float32).reshape(1, length, 1)
    projections = tuple(
        (base * scale).expand(-1, -1, 16).clone().requires_grad_()
        for scale in (1.0, 2.0, 3.0)
    )
    mask = torch.ones(1, length, dtype=torch.bool)
    adapted = attention._adapt_event_projections(*projections, mask)
    weights = torch.arange(1, length + 1, dtype=torch.float32).reshape(1, length, 1)
    sum((value * weights).sum() for value in adapted).backward()
    for parameter in kernels(candidate):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        observed = tuple(bool(torch.count_nonzero(row)) for row in parameter.grad)
        assert observed == expected_nonzero_rows


def test_persistent_key_values_are_appended_without_transition_adaptation():
    _, candidate = build_pair(seed=85)
    attention = candidate.level_history_qkv_attention
    attention.eval()
    for parameter in kernels(candidate):
        parameter.data.fill_(10.0)
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


@pytest.mark.parametrize("opened_kind", ["q", "k", "v"])
def test_each_projection_branch_is_independently_wired(opened_kind):
    _, candidate = build_pair(seed=86)
    attention = candidate.level_history_qkv_attention
    projections = tuple(torch.randn(2, 5, 16) for _ in range(3))
    mask = torch.ones(2, 5, dtype=torch.bool)
    for kind, parameter in zip(("q", "k", "v"), kernels(candidate)):
        parameter.data.fill_(0.2 if kind == opened_kind else 0.0)
    adapted = attention._adapt_event_projections(*projections, mask)
    for kind, original, actual in zip(("q", "k", "v"), projections, adapted):
        if kind == opened_kind:
            assert not torch.equal(original, actual)
        else:
            assert torch.equal(original, actual)


def test_prefix_future_target_padding_and_batch_independence():
    _, candidate = build_pair(seed=91)
    candidate.eval()
    for parameter in kernels(candidate):
        parameter.data.fill_(0.2)
    dts, quantities, mask = sample_batch()
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    changed_dts[:, 3:] = torch.tensor([99.0, 999.0])
    changed_quantities[:, 3:] = torch.tensor([777.0, 7777.0])
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
        target_dts[torch.arange(2), target_positions] = 333.0
        target_qty[torch.arange(2), target_positions] = 77777.0
        actual = objective(candidate, target_dts, target_qty, mask)
        assert torch.equal(expected["pred_qty"], actual["pred_qty"])
        assert not torch.equal(expected["true_qty"], actual["true_qty"])


def test_left_and_right_padding_normalize_to_identical_target_outputs():
    _, candidate = build_pair(seed=92)
    candidate.eval()
    for parameter in kernels(candidate):
        parameter.data.fill_(0.15)
    left = (
        torch.tensor([[1e10, 1e10, 1.0, 2.0, 3.0]]),
        torch.tensor([[1e12, 1e12, 2.0, 4.0, 8.0]]),
        torch.tensor([[False, False, True, True, True]]),
    )
    right = (
        torch.tensor([[1.0, 2.0, 3.0, 1e10, 1e10]]),
        torch.tensor([[2.0, 4.0, 8.0, 1e12, 1e12]]),
        torch.tensor([[True, True, True, False, False]]),
    )
    with torch.no_grad():
        left_output = objective(candidate, *left)
        right_output = objective(candidate, *right)
    for name in left_output:
        assert torch.equal(left_output[name], right_output[name]), name


def test_open_route_changes_attention_hidden_search_and_quantity_prediction():
    _, candidate = build_pair(seed=203)
    batch = sample_batch()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    candidate.train()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        objective(candidate, *batch)["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
    candidate.eval()
    captured = {}

    def record_output(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().clone()

        return hook

    def record_memory(_module, inputs):
        captured["memory_input"] = inputs[0].detach().clone()

    handles = [
        candidate.level_history_qkv_attention.register_forward_hook(
            record_output("attention")
        ),
        candidate.encoder.layers[0].register_forward_hook(record_output("layer0")),
        candidate.encoder.layers[1].register_forward_hook(record_output("layer1")),
        candidate.lmm.register_forward_pre_hook(record_memory),
    ]
    try:
        with torch.no_grad():
            opened = objective(candidate, *batch)
            opened_path = copy.deepcopy(captured)
            _, opened_trace = candidate.lmm.retrieve(captured["memory_input"])
            candidate.reset_level_history_qkv_identity()
            closed = objective(candidate, *batch)
            closed_path = copy.deepcopy(captured)
            _, closed_trace = candidate.lmm.retrieve(captured["memory_input"])
        for name in ("attention", "layer0", "layer1", "memory_input"):
            assert not torch.equal(opened_path[name], closed_path[name]), name
        assert not torch.equal(
            opened_trace["topk_similarity"], closed_trace["topk_similarity"]
        )
        assert not torch.equal(opened["pred_qty"], closed["pred_qty"])
    finally:
        for handle in handles:
            handle.remove()


def test_open_route_can_change_selected_top4_and_memory_value_contribution():
    _, candidate = build_pair(seed=203)
    candidate.eval()
    dts, quantities, mask = sample_batch()
    captured = []
    handle = candidate.lmm.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            candidate.encode_task_states(dts, quantities, mask)
            closed_hidden = captured[-1]
            attention = candidate.level_history_qkv_attention
            attention.level_history_q_kernel.copy_(
                torch.stack(
                    [torch.linspace(-2, 2, 16), torch.linspace(2, -2, 16)]
                )
            )
            attention.level_history_k_kernel.copy_(
                torch.stack(
                    [torch.linspace(1, -1, 16), torch.linspace(-1, 1, 16)]
                )
            )
            attention.level_history_v_kernel.copy_(
                torch.stack(
                    [torch.linspace(-8, 8, 16), torch.linspace(8, -8, 16)]
                )
            )
            candidate.encode_task_states(dts, quantities, mask)
            opened_hidden = captured[-1]
            closed_direction = torch.nn.functional.normalize(
                closed_hidden[0, 3], dim=0
            )
            opened_direction = torch.nn.functional.normalize(
                opened_hidden[0, 3], dim=0
            )
            bank = (-closed_direction).repeat(64, 1)
            bank[:4] = closed_direction
            bank[4:8] = opened_direction * 2
            candidate.lmm.mem.copy_(bank.unsqueeze(0))
            closed_residual, closed_trace = candidate.lmm.retrieve(closed_hidden)
            opened_residual, opened_trace = candidate.lmm.retrieve(opened_hidden)
        closed_indices = closed_trace["prototype_indices"][0, 3]
        opened_indices = opened_trace["prototype_indices"][0, 3]
        assert bool((closed_indices < 4).all())
        assert bool(((opened_indices >= 4) & (opened_indices < 8)).all())
        assert not torch.equal(closed_residual[0, 3], opened_residual[0, 3])
    finally:
        handle.remove()


def payload(candidate, include_state=True):
    result = dict(
        backbone=LEVEL_HISTORY_QKV_BACKBONE,
        variant="count_only_log_regression",
        evaluation_scope="validation_only",
        held_out_test_evaluated=False,
        encoder_config={
            **level_history_qkv_metadata(candidate.hidden_dim),
            "time_head": candidate.time_head_contract(),
        },
    )
    if include_state:
        result["model_state_dict"] = candidate.state_dict()
    else:
        result.update(
            best_epoch=17,
            checkpoint_state_sha256="a" * 64,
            checkpoint_file_sha256="b" * 64,
        )
    return result


def test_factory_checkpoint_metadata_and_route_isolation():
    _, candidate = build_pair()
    built, metadata = build_count_aware_model(
        LEVEL_HISTORY_QKV_BACKBONE, **model_kwargs()
    )
    assert isinstance(built, CountAwareTitanLevelHistoryQKVTPP)
    assert metadata["model_role"] == LEVEL_HISTORY_QKV_ROLE
    assert metadata["max_len"] == 10
    expected = payload(candidate)
    assert candidate.contract_id == LEVEL_HISTORY_QKV_CONTRACT_ID
    assert validate_level_history_qkv_checkpoint(
        expected, LEVEL_HISTORY_QKV_BACKBONE
    )
    assert validate_level_history_qkv_checkpoint(
        payload(candidate, False), LEVEL_HISTORY_QKV_BACKBONE
    )
    for other_route in ("titantpp", CAUSAL_QKV_BACKBONE, BOUNDED_QK_BACKBONE):
        with pytest.raises(ValueError, match="cannot be relabelled"):
            validate_level_history_qkv_checkpoint(expected, other_route)
    invalid = copy.deepcopy(expected)
    invalid["model_state_dict"][LEVEL_HISTORY_QKV_KERNEL_KEYS[0]][0, 0] = float(
        "nan"
    )
    with pytest.raises(ValueError, match="kernels are missing or invalid"):
        validate_level_history_qkv_checkpoint(invalid, LEVEL_HISTORY_QKV_BACKBONE)


def test_checkpoint_tamper_matrix_is_rejected_fail_closed():
    _, candidate = build_pair(seed=104)
    expected = payload(candidate)
    corruptions = []

    missing = copy.deepcopy(expected)
    missing["model_state_dict"].pop(LEVEL_HISTORY_QKV_KERNEL_KEYS[0])
    corruptions.append(missing)
    wrong_shape = copy.deepcopy(expected)
    wrong_shape["model_state_dict"][LEVEL_HISTORY_QKV_KERNEL_KEYS[1]] = torch.zeros(
        3, 16
    )
    corruptions.append(wrong_shape)
    wrong_metadata = copy.deepcopy(expected)
    wrong_metadata["encoder_config"]["level_history_qk_epsilon"] = 1e-6
    corruptions.append(wrong_metadata)
    wrong_lmm = copy.deepcopy(expected)
    wrong_lmm["model_state_dict"]["lmm.mem"] = torch.zeros(1, 63, 16)
    corruptions.append(wrong_lmm)
    for marker in LEVEL_HISTORY_QKV_FOREIGN_STATE_KEYS:
        hybrid = copy.deepcopy(expected)
        hybrid["model_state_dict"][marker] = torch.zeros(1)
        corruptions.append(hybrid)
    for old_key in (
        "encoder.layers.0.attn.causal_q_kernel",
        "encoder.layers.0.attn.bounded_q_kernel",
    ):
        hybrid = copy.deepcopy(expected)
        hybrid["model_state_dict"][old_key] = torch.zeros(3, 16)
        corruptions.append(hybrid)

    for corrupted in corruptions:
        with pytest.raises(ValueError):
            validate_level_history_qkv_checkpoint(
                corrupted, LEVEL_HISTORY_QKV_BACKBONE
            )

    for field, wrong in (
        ("variant", "count_only_lognormal_k1"),
        ("evaluation_scope", "test"),
        ("held_out_test_evaluated", True),
    ):
        corrupted = copy.deepcopy(expected)
        corrupted[field] = wrong
        with pytest.raises(ValueError, match="scope/head/objective"):
            validate_level_history_qkv_checkpoint(
                corrupted, LEVEL_HISTORY_QKV_BACKBONE
            )
    no_state = payload(candidate, False)
    no_state["checkpoint_state_sha256"] = "bad"
    with pytest.raises(ValueError, match="requires epoch"):
        validate_level_history_qkv_checkpoint(no_state, LEVEL_HISTORY_QKV_BACKBONE)


def test_explicit_b_initialization_is_strict_atomic_and_exact():
    baseline, candidate = build_pair(seed=103)
    for parameter in kernels(candidate):
        parameter.data.fill_(0.5)
    before = copy.deepcopy(candidate.state_dict())
    incomplete = dict(baseline.state_dict())
    incomplete.pop("quantity_head.bias")
    nonfinite = dict(baseline.state_dict())
    nonfinite["quantity_head.bias"] = torch.full_like(
        nonfinite["quantity_head.bias"], float("nan")
    )
    wrong_dtype = dict(baseline.state_dict())
    wrong_dtype["quantity_head.bias"] = wrong_dtype["quantity_head.bias"].double()
    for invalid in (incomplete, candidate.state_dict(), nonfinite, wrong_dtype):
        with pytest.raises(RuntimeError):
            candidate.load_hard_lmm_state_dict(invalid)
        for name, tensor in candidate.state_dict().items():
            assert torch.equal(tensor, before[name]), name
    candidate.load_hard_lmm_state_dict(baseline.state_dict())
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(candidate.state_dict()[name], tensor)
    assert all(torch.count_nonzero(parameter) == 0 for parameter in kernels(candidate))


@pytest.mark.parametrize("device", DEVICES)
def test_extremes_finite_and_optimizer_roundtrip_next_step(device):
    _, candidate = build_pair(seed=117, device=device)
    batch = (
        torch.tensor([[0.0, 1e-8, 1e4, 1e8]], device=device),
        torch.tensor([[0.0, 1.0, 1e6, 1e10]], device=device),
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
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())

    step(candidate, optimizer)
    assert all(torch.count_nonzero(parameter) > 0 for parameter in kernels(candidate))
    buffer = io.BytesIO()
    torch.save(
        {"model": candidate.state_dict(), "optimizer": optimizer.state_dict()}, buffer
    )
    buffer.seek(0)
    saved = torch.load(buffer, map_location=device, weights_only=True)
    _, restored = build_pair(seed=999, device=device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer.load_state_dict(saved["optimizer"])
    step(candidate, optimizer)
    step(restored, restored_optimizer)
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name
    original_state = optimizer.state_dict()
    restored_state = restored_optimizer.state_dict()
    assert original_state["param_groups"] == restored_state["param_groups"]
    for parameter_id, state in original_state["state"].items():
        for name, value in state.items():
            actual = restored_state["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.cpu(), actual.cpu())
            else:
                assert value == actual
