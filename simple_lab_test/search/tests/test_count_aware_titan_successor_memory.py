"""Independent CPU contracts for completed-event association memory."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanSuccessorMemory import (
    CountAwareTitanSameEventMemoryTPP,
    CountAwareTitanSuccessorMemoryTPP,
    EpisodeMemory,
    completed_pairs,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def kwargs(hidden_dim=16):
    return {
        "hidden_dim": hidden_dim, "train_log_mean": 1.5, "max_seq_len": 12,
        "quantity_variant": "count_only_log_regression", "lambda_tail": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp", "time_intercept_limit": 300.0,
    }


def build_three(seed=41, hidden_dim=16):
    models = []
    rng_states = []
    for cls in (CountAwareTitanTPP, CountAwareTitanSameEventMemoryTPP, CountAwareTitanSuccessorMemoryTPP):
        torch.manual_seed(seed)
        models.append(cls(**kwargs(hidden_dim)))
        rng_states.append(torch.get_rng_state().clone())
    assert all(torch.equal(rng_states[0], value) for value in rng_states[1:])
    return models


def sample_batch():
    dts = torch.tensor([[0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 3.0, 6.0, 0.0]])
    quantities = torch.tensor([[2.0, 5.0, 9.0, 12.0, 20.0], [1.0, 7.0, 13.0, 19.0, 0.0]])
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, True, False]])
    return dts, quantities, mask


def pair_oracle(mask, write):
    pairs = []
    for row in range(len(mask)):
        indices = [i for i in range(mask.shape[1]) if bool(mask[row, i])]
        pairs.append([
            (j, k) for j, k in zip(indices, indices[1:])
            if bool(write[row, j]) and bool(write[row, k])
        ])
    return pairs


def read_oracle(memory, hidden, features, mask, write, association):
    rows = []
    width = hidden.shape[-1]
    pairs = pair_oracle(mask, write)
    for row, records in enumerate(pairs):
        outputs = []
        for t in range(hidden.shape[1]):
            available = [(j, k) for j, k in records if k <= t]
            if not bool(mask[row, t] & write[row, t]) or not available:
                outputs.append(features.new_zeros(2))
                continue
            query = F.linear(F.layer_norm(hidden[row, t], (width,), eps=1e-5), memory.query_projection.weight)
            indices = [j if association == "successor" else k for j, k in available]
            contexts = torch.stack([hidden[row, i] for i in indices])
            keys = F.linear(F.layer_norm(contexts, (width,), eps=1e-5), memory.key_projection.weight)
            scores = (keys * query[None]).sum(dim=1) / math.sqrt(16)
            weights = torch.softmax(scores, dim=0)
            values = torch.stack([features[row, k] for _, k in available])
            outputs.append((weights[:, None] * values).sum(dim=0))
        rows.append(torch.stack(outputs))
    return torch.stack(rows)


def test_pair_indices_skip_padding_without_bridging_a_nonwriting_valid_event():
    mask = torch.tensor([[True, False, True, True, True, False, True], [False, True, False, True, False, False, False]])
    write = torch.tensor([[True, True, True, False, True, True, True], [True, False, True, True, True, True, True]])
    predecessor, pair_valid, observed = completed_pairs(mask, memory_write_mask=write)
    assert torch.equal(observed, mask & write)
    expected_valid = torch.zeros_like(mask)
    for row, records in enumerate(pair_oracle(mask, write)):
        for j, k in records:
            expected_valid[row, k] = True
            assert predecessor[row, k].item() == j
    assert torch.equal(pair_valid, expected_valid)
    assert pair_oracle(mask, write)[0] == [(0, 2), (4, 6)]

    memory = EpisodeMemory(8)
    with torch.no_grad():
        memory.query_projection.weight.zero_()
        memory.key_projection.weight.zero_()
    hidden = torch.randn(2, 7, 8)
    features = torch.arange(28, dtype=torch.float32).reshape(2, 7, 2)
    read = memory.read(hidden, features, mask, memory_write_mask=write)
    assert torch.equal(read[0, 2], features[0, 2])
    assert torch.equal(read[0, 4], features[0, 2])
    assert torch.equal(read[0, 6], features[0, [2, 6]].mean(dim=0))
    assert torch.count_nonzero(read[~observed]) == 0


@pytest.mark.parametrize("association", ["successor", "same_event"])
def test_vectorized_reads_and_gradients_match_independent_prefix_oracle(association):
    torch.manual_seed(59)
    memory = EpisodeMemory(12, association=association).double()
    reference_memory = copy.deepcopy(memory)
    hidden = torch.randn(3, 7, 12, dtype=torch.float64, requires_grad=True)
    features = torch.rand(3, 7, 2, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_features = features.detach().clone().requires_grad_()
    mask = torch.tensor([
        [False, True, True, False, True, True, True],
        [True, True, False, True, True, False, True],
        [False, False, False, False, False, False, False],
    ])
    write = torch.tensor([
        [True, True, True, True, False, True, True],
        [True, True, True, True, False, True, True],
        [True, True, True, True, True, True, True],
    ])
    actual = memory.read(hidden, features, mask, memory_write_mask=write)
    expected = read_oracle(reference_memory, reference_hidden, reference_features, mask, write, association)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(features.grad, reference_features.grad, rtol=2e-10, atol=2e-11)
    for name in ("query_projection.weight", "key_projection.weight"):
        actual_parameter = dict(memory.named_parameters())[name]
        expected_parameter = dict(reference_memory.named_parameters())[name]
        assert actual_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("observations", [0, 1])
def test_empty_record_rows_return_finite_zero_correction_and_safe_backward(observations):
    memory = EpisodeMemory(8)
    with torch.no_grad():
        memory.output_projection.weight.fill_(0.3)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    features = torch.randn(2, 5, 2, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    write = torch.zeros_like(mask)
    if observations:
        write[:, 2] = True
    correction = memory(hidden, features, mask, memory_write_mask=write)
    assert torch.isfinite(correction).all()
    assert torch.count_nonzero(correction) == 0
    correction.sum().backward()
    for value in (hidden, features, *memory.parameters()):
        if value.grad is not None:
            assert torch.isfinite(value.grad).all()
            assert torch.count_nonzero(value.grad) == 0


def test_fixed_contexts_and_value_set_expose_pair_permutation_and_same_event_control():
    successor = EpisodeMemory(4, association="successor").double()
    control = EpisodeMemory(4, association="same_event").double()
    with torch.no_grad():
        successor.query_projection.weight.zero_()
        successor.key_projection.weight.zero_()
        successor.query_projection.weight[0, 0] = 4.0
        successor.key_projection.weight[0, 0] = 4.0
        successor.output_projection.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., -1.]]))
        for source, target in zip(successor.parameters(), control.parameters(), strict=True):
            target.copy_(source)
    hidden = torch.tensor([[[3., -1., -1., -1.], [-1., 3., -1., -1.], [-1., -1., 3., -1.], [3., -1., -1., -1.]]], dtype=torch.float64)
    features = torch.tensor([[[0., 0.], [1., 4.], [7., 2.], [3., 9.]]], dtype=torch.float64)
    permuted = features[:, [0, 2, 3, 1]]
    mask = torch.ones(1, 4, dtype=torch.bool)
    first = successor.read(hidden, features, mask)
    reassociated = successor.read(hidden, permuted, mask)
    same_event = control.read(hidden, features, mask)
    torch.testing.assert_close(first[:, -1], features[:, 1], rtol=0., atol=1e-5)
    torch.testing.assert_close(same_event[:, -1], features[:, 3], rtol=0., atol=1e-5)
    torch.testing.assert_close(reassociated[:, -1], features[:, 2], rtol=0., atol=1e-5)
    assert not torch.allclose(successor(hidden, features, mask)[:, -1], successor(hidden, permuted, mask)[:, -1])
    assert not torch.allclose(successor(hidden, features, mask)[:, -1], control(hidden, features, mask)[:, -1])


def test_masked_padding_latents_are_ignored_without_changing_event_adjacency():
    torch.manual_seed(83)
    memory = EpisodeMemory(8).double()
    hidden = torch.randn(1, 4, 8, dtype=torch.float64)
    features = torch.rand(1, 4, 2, dtype=torch.float64)
    valid_positions = torch.tensor([1, 2, 4, 6])
    padded_hidden = torch.full((1, 8, 8), float("nan"), dtype=torch.float64)
    padded_features = torch.full((1, 8, 2), float("inf"), dtype=torch.float64)
    padded_hidden[:, valid_positions] = hidden
    padded_features[:, valid_positions] = features
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, valid_positions] = True
    with torch.no_grad():
        original = memory.read(hidden, features, torch.ones(1, 4, dtype=torch.bool))
        padded = memory.read(padded_hidden, padded_features, mask)
    torch.testing.assert_close(padded[:, valid_positions], original, rtol=1e-13, atol=1e-13)
    assert torch.count_nonzero(padded[~mask]) == 0
    assert torch.isfinite(padded).all()


def test_module_prefix_batch_and_repeated_call_isolation_preserve_parameters():
    torch.manual_seed(89)
    memory = EpisodeMemory(8).double()
    hidden = torch.randn(3, 6, 8, dtype=torch.float64)
    features = torch.randn(3, 6, 2, dtype=torch.float64)
    mask = torch.ones(3, 6, dtype=torch.bool)
    write = mask.clone()
    write[1, 3] = False
    snapshot = {name: value.clone() for name, value in memory.state_dict().items()}
    with torch.no_grad():
        original = memory.read(hidden, features, mask, memory_write_mask=write)
        single = torch.cat([memory.read(hidden[i:i+1], features[i:i+1], mask[i:i+1], memory_write_mask=write[i:i+1]) for i in range(3)])
        order = torch.tensor([2, 0, 1])
        permuted = memory.read(hidden[order], features[order], mask[order], memory_write_mask=write[order])
        memory.read(hidden * 10, features * 20, mask)
        repeated = memory.read(hidden, features, mask, memory_write_mask=write)
        for end in range(1, 7):
            prefix = memory.read(hidden[:, :end], features[:, :end], mask[:, :end], memory_write_mask=write[:, :end])
            torch.testing.assert_close(prefix, original[:, :end], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(single, original, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(permuted, original[order], rtol=2e-12, atol=2e-12)
    assert torch.equal(repeated, original)
    for name, value in memory.state_dict().items():
        assert torch.equal(snapshot[name], value), name


def test_output_map_learns_first_then_both_address_projections_receive_gradients():
    torch.manual_seed(97)
    memory = EpisodeMemory(8).double()
    hidden = torch.randn(2, 6, 8, dtype=torch.float64)
    features = torch.rand(2, 6, 2, dtype=torch.float64) * 4
    target = torch.randn(2, 6, 8, dtype=torch.float64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    optimizer = torch.optim.SGD(memory.parameters(), lr=0.01)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (memory(hidden, features, mask) - target).square().sum()
        loss.backward()
        for name, parameter in memory.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            if step == 0 and name != "output_projection.weight":
                assert torch.count_nonzero(parameter.grad) == 0, name
            else:
                assert torch.count_nonzero(parameter.grad) > 0, name
        optimizer.step()


def test_h64_parameter_budget_and_same_initial_new_weights():
    baseline, control, candidate = build_three(hidden_dim=64)
    baseline_count = sum(value.numel() for value in baseline.parameters())
    for model in (control, candidate):
        assert sum(value.numel() for value in model.parameters()) - baseline_count == 2176
        assert torch.count_nonzero(model.episode_memory.output_projection.weight) == 0
        for name, value in baseline.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
    for name, value in control.episode_memory.named_parameters():
        assert torch.equal(value, dict(candidate.episode_memory.named_parameters())[name]), name
    assert control.episode_memory.association_code.item() == 0
    assert candidate.episode_memory.association_code.item() == 1


@pytest.mark.parametrize("nonconstant_quantity_head", [False, True])
def test_zero_output_map_matches_b_train_rng_outputs_objective_and_shared_preclip_gradients(nonconstant_quantity_head):
    models = build_three(seed=101)
    dts, quantities, mask = sample_batch()
    outputs, rng_states = [], []
    for model in models:
        model.train()
        if nonconstant_quantity_head:
            with torch.no_grad():
                model.quantity_head.weight.copy_(torch.linspace(-0.3, 0.4, 16)[None])
        torch.manual_seed(902)
        outputs.append(target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0))
        rng_states.append(torch.get_rng_state().clone())
        outputs[-1]["joint_loss"].mean().backward()
    for output, rng in zip(outputs[1:], rng_states[1:], strict=True):
        assert torch.equal(rng, rng_states[0])
        for name, value in outputs[0].items():
            assert torch.equal(value, output[name]), name
    for name, baseline_parameter in models[0].named_parameters():
        for model in models[1:]:
            gradient = dict(model.named_parameters())[name].grad
            if baseline_parameter.grad is None:
                assert gradient is None, name
            else:
                assert torch.equal(gradient, baseline_parameter.grad), name
    for model in models[1:]:
        for name, parameter in model.episode_memory.named_parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            if name == "output_projection.weight":
                assert torch.count_nonzero(parameter.grad) > 0
            else:
                assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("cls", [CountAwareTitanSameEventMemoryTPP, CountAwareTitanSuccessorMemoryTPP])
@pytest.mark.parametrize("objective", ["time_loss", "quantity_train_loss"])
def test_activated_memory_is_trainable_through_each_existing_head(cls, objective):
    torch.manual_seed(103)
    model = cls(**kwargs()).eval()
    with torch.no_grad():
        model.episode_memory.output_projection.weight.copy_(torch.linspace(-0.2, 0.3, 32).reshape(16, 2))
        model.quantity_head.weight.copy_(torch.linspace(-0.3, 0.4, 16)[None])
    dts, quantities, mask = sample_batch()
    outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    outputs[objective].mean().backward()
    for name, parameter in model.episode_memory.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_direct_state_loading_cannot_change_the_fixed_association_even_when_nonstrict():
    _, control, successor = build_three()
    with pytest.raises(RuntimeError, match="association"):
        successor.load_state_dict(control.state_dict(), strict=False)
    with pytest.raises(RuntimeError, match="association"):
        control.load_state_dict(successor.state_dict(), strict=False)


@pytest.mark.parametrize("cls", [CountAwareTitanSameEventMemoryTPP, CountAwareTitanSuccessorMemoryTPP])
def test_nonzero_memory_actual_target_outputs_excludes_future_dt_and_quantity(cls):
    torch.manual_seed(107)
    model = cls(**kwargs()).eval()
    with torch.no_grad():
        model.episode_memory.output_projection.weight.copy_(torch.linspace(-0.2, 0.3, 32).reshape(16, 2))
        model.quantity_head.weight.copy_(torch.linspace(-0.3, 0.4, 16)[None])
    dts, quantities, mask = sample_batch()
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    changed_dts[:, 3:] += 10000.0
    changed_quantities[:, 3:] += 20000.0
    with torch.no_grad():
        initial = model.encode(dts, quantities, mask)
        future_changed = model.encode(changed_dts, changed_quantities, mask)
    assert torch.equal(initial[:, :3], future_changed[:, :3])
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    rows, target = torch.arange(len(dts)), mask.sum(dim=1) - 1
    changed_dts[rows, target] += 10000.0
    changed_quantities[rows, target] += 20000.0
    with torch.no_grad():
        expected = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        actual = target_outputs(model, changed_dts, mask, changed_quantities, lambda_log_qty=1.0)
    assert torch.equal(actual["pred_qty"], expected["pred_qty"])
    assert not torch.equal(actual["time_loss"], expected["time_loss"])
    assert not torch.equal(actual["quantity_train_loss"], expected["quantity_train_loss"])
    dts.requires_grad_()
    quantities.requires_grad_()
    target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)["pred_qty"].sum().backward()
    for value in (dts, quantities):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[rows, target]) == 0
        assert torch.count_nonzero(value.grad[:, 1:3]) > 0


def test_runner_canonical_padding_and_model_calls_are_isolated():
    _, _, model = build_three(seed=113)
    model.eval()
    with torch.no_grad():
        model.episode_memory.output_projection.weight.copy_(torch.linspace(-0.2, 0.3, 32).reshape(16, 2))
        model.quantity_head.weight.copy_(torch.linspace(-0.3, 0.4, 16)[None])
    dts, quantities, mask = sample_batch()
    positions = torch.tensor([1, 3, 4, 6, 8])
    padded_dts = torch.full((2, 9), 99999.0)
    padded_quantities = torch.full((2, 9), 99999.0)
    padded_mask = torch.zeros(2, 9, dtype=torch.bool)
    padded_dts[:, positions] = dts
    padded_quantities[:, positions] = quantities
    padded_mask[:, positions] = mask
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        padded = target_outputs(model, padded_dts, padded_mask, padded_quantities, lambda_log_qty=1.0)
        model.encode(dts * 10, quantities * 20, mask)
        repeated = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    for name in expected:
        torch.testing.assert_close(padded[name], expected[name], rtol=1e-6, atol=1e-6)
        assert torch.equal(repeated[name], expected[name]), name
    for name, value in model.state_dict().items():
        assert torch.equal(snapshot[name], value), name
