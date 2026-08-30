"""Regression coverage for the Instacart inner-loop divergence incident."""

import copy

import pytest
import torch

from models.Titan.common.titans_mac import TitansNeuralMemory, _scan_titans_write_sequence
from models.Titan.common.titans_mac_optimized import _scan_titans_write_sequence_state_only
from models.Titan.common.titans_memory_stability import clip_associative_gradients


def test_clipping_is_per_series_joint_and_preserves_small_gradients():
    first = torch.tensor([[0.1, 0.2], [3.0, 4.0], [1e30, -1e30]], requires_grad=True)
    second = torch.tensor([[0.1], [12.0], [1e30]], requires_grad=True)
    clipped = clip_associative_gradients((first, second), 1.0)
    joined = torch.cat(clipped, dim=1)
    assert torch.isfinite(joined).all()
    assert (joined.norm(dim=1) <= 1.000001).all()
    assert torch.equal(clipped[0][0], first[0])
    assert torch.equal(clipped[1][0], second[0])
    assert torch.allclose(joined[1], torch.tensor([3., 4., 12.]) / 13.)
    joined.sum().backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
    assert clip_associative_gradients((first, second), None)[0] is first


def test_zero_gradient_and_higher_order_derivatives_remain_finite():
    values = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    clipped, = clip_associative_gradients((values,), 1.0)
    clipped.square().sum().backward()
    assert torch.equal(values.grad, torch.zeros_like(values))
    x = torch.tensor([[2., 3.], [0.1, -0.2]], dtype=torch.float64, requires_grad=True)
    function = lambda value: clip_associative_gradients((value,), 1.0)[0]
    assert torch.autograd.gradcheck(function, (x,))
    assert torch.autograd.gradgradcheck(function, (x,))


def unstable_memory():
    torch.manual_seed(71)
    memory = TitansNeuralMemory(4, initial_update_rate=0.9, initial_momentum=0.99,
                                compile_cuda_scan=False)
    return memory


def test_unbounded_update_reproduces_failure_and_bounded_update_survives_64_writes():
    memory = unstable_memory()
    inputs = torch.randn(2, 64, 4)
    mask = torch.ones(2, 64, dtype=torch.bool)
    mask[1, 19:] = False
    initial = memory.initial_state(2, device=inputs.device, dtype=inputs.dtype)
    with torch.no_grad():
        broken, _ = memory.write_sequence(initial, inputs, mask)
    assert any(not torch.isfinite(t).all() for t in broken.memory_tensors())

    memory.gradient_max_norm = 1.0
    stable, diagnostics = memory.write_sequence(initial, inputs, mask)
    assert all(torch.isfinite(t).all() for t in (*stable.memory_tensors(), *stable.momentum_tensors()))
    assert all(torch.isfinite(t).all() for t in diagnostics.values())
    sum(t.square().mean() for t in stable.memory_tensors()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in memory.parameters())
    with torch.no_grad():
        short, _ = memory.write_sequence(initial, inputs[:, :19], mask[:, :19])
    for full, partial in zip(stable.memory_tensors(), short.memory_tensors()):
        assert torch.equal(full[1], partial[1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_token_scan_and_state_only_scan_share_the_stability_policy(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(72)
    memory = TitansNeuralMemory(4, compile_cuda_scan=False, gradient_max_norm=1.0).to(device)
    inputs = torch.randn(2, 17, 4, device=device)
    mask = torch.ones(2, 17, dtype=torch.bool, device=device)
    mask[1, 9:] = False
    initial = memory.initial_state(2, device=inputs.device, dtype=inputs.dtype)
    reference, diag = memory.write_sequence(initial, inputs, mask, chunk_size=1)
    keys, values, theta, eta, alpha = memory._project_write(inputs)
    args = (*initial.memory_tensors(), *initial.momentum_tensors(), keys, values,
            theta.squeeze(-1), eta.squeeze(-1), alpha.squeeze(-1), mask, 1.0)
    scanned = _scan_titans_write_sequence(*args)
    state_only = _scan_titans_write_sequence_state_only(*args)
    for expected, actual, compact in zip((*reference.memory_tensors(), *reference.momentum_tensors()), scanned[:8], state_only):
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
        assert torch.equal(actual, compact)
    if device == "cuda":
        compiled_memory = copy.deepcopy(memory)
        compiled_memory.compile_cuda_scan = True
        compiled_state, _ = compiled_memory.write_sequence(initial, inputs, mask)
        for expected, actual in zip(reference.memory_tensors(), compiled_state.memory_tensors()):
            assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("limit", [0., -1., float("nan"), float("inf")])
def test_invalid_inner_gradient_limit_rejected(limit):
    with pytest.raises(ValueError, match="gradient_max_norm"):
        TitansNeuralMemory(4, gradient_max_norm=limit)
