"""Task-query routing invariants with independent causal oracles; synthetic CPU only."""
from __future__ import annotations

import copy
import hashlib
import io
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanTaskRoutedMemory import (
    CountAwareTitanTaskSharedMemoryTPP, CountAwareTitanTaskSplitMemoryTPP,
    TASK_ROUTED_MEMORY_DESIGN_SHA256, TaskRoutedEpisodeMemory, task_routed_memory_metadata,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs

CLASSES = (CountAwareTitanTaskSharedMemoryTPP, CountAwareTitanTaskSplitMemoryTPP)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def model_kwargs(hidden_dim=16, max_len=12):
    return dict(hidden_dim=hidden_dim, train_log_mean=1.5, max_seq_len=max_len,
                quantity_variant="count_only_log_regression", lambda_tail=0.,
                time_head_mode="legacy_clamped_rmtpp", time_intercept_limit=300.)


def batch():
    return (
        torch.tensor([[0., 1., 2., 4., 8.], [0., .5, 3., 6., 0.]]),
        torch.tensor([[2., 5., 9., 12., 20.], [1., 7., 13., 19., 0.]]),
        torch.tensor([[True, True, True, True, True], [True, True, True, True, False]]),
    )


def activate_memory(memory):
    with torch.no_grad():
        size = memory.hidden_dim * memory.value_width
        memory.time_output_projection.weight.copy_(torch.linspace(-.2, .3, size).reshape(memory.hidden_dim, -1))
        memory.quantity_output_projection.weight.copy_(torch.linspace(.3, -.1, size).reshape(memory.hidden_dim, -1))
        memory.value_projection.bias.copy_(torch.linspace(-.4, .3, memory.value_width))


def activate_model(model):
    activate_memory(model.task_routed_episode_memory)
    with torch.no_grad():
        model.quantity_head.weight.copy_(torch.linspace(-.3, .4, model.hidden_dim)[None])


def scalar_oracle(memory, hidden, features, mask, write):
    """Enumerate adjacency and each query prefix independently of vectorized helpers."""
    branches = [[], []]
    width = hidden.shape[-1]
    for row in range(hidden.shape[0]):
        valid = [pos for pos in range(hidden.shape[1]) if bool(mask[row, pos])]
        records = [(j, k) for j, k in zip(valid, valid[1:]) if bool(write[row, j] & write[row, k])]
        per_branch = [[], []]
        for t in range(hidden.shape[1]):
            available = [(j, k) for j, k in records if k <= t]
            if not bool(mask[row, t] & write[row, t]) or not available:
                reads = [hidden.new_zeros(16), hidden.new_zeros(16)]
            else:
                reads = []
                for projection in (memory.time_query_projection, memory.quantity_query_projection):
                    query = F.linear(F.layer_norm(hidden[row, t], (width,), eps=1e-5), projection.weight)
                    scores, values = [], []
                    for j, k in available:
                        key = F.linear(F.layer_norm(hidden[row, j], (width,), eps=1e-5), memory.key_projection.weight)
                        scores.append((query * key).sum() / math.sqrt(8))
                        values.append(F.gelu(F.linear(features[row, k], memory.value_projection.weight,
                                                      memory.value_projection.bias), approximate="none"))
                    weights = torch.softmax(torch.stack(scores), 0)
                    reads.append(sum(weight * value for weight, value in zip(weights, values)))
                if memory.routing == "shared":
                    reads = [(reads[0] + reads[1]) / 2] * 2
            for branch in (0, 1):
                per_branch[branch].append(reads[branch])
        for branch in (0, 1):
            branches[branch].append(torch.stack(per_branch[branch]))
    return tuple(torch.stack(rows) for rows in branches)


@pytest.mark.parametrize("routing", ["shared", "split"])
def test_scalar_prefix_oracle_matches_both_reads_corrections_and_every_gradient(routing):
    torch.manual_seed(59)
    memory = TaskRoutedEpisodeMemory(12, routing).double()
    activate_memory(memory)
    reference = copy.deepcopy(memory)
    hidden = torch.randn(3, 7, 12, dtype=torch.float64, requires_grad=True)
    features = torch.rand(3, 7, 2, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_features = features.detach().clone().requires_grad_()
    mask = torch.tensor([[False, True, True, False, True, True, True],
                         [True, True, False, True, True, False, True], [False] * 7])
    write = torch.tensor([[True, True, True, True, False, True, True],
                          [True, True, True, True, False, True, True], [True] * 7])
    actual_reads = memory.read(hidden, features, mask, memory_write_mask=write)
    oracle_reads = scalar_oracle(reference, reference_hidden, reference_features, mask, write)
    for actual, expected in zip(actual_reads, oracle_reads):
        torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    actual = memory(hidden, features, mask, memory_write_mask=write)
    expected = tuple(F.linear(context, projection.weight) for context, projection in zip(
        oracle_reads, (reference.time_output_projection, reference.quantity_output_projection)))
    probes = [torch.randn_like(value) for value in actual]
    sum((value * probe).sum() for value, probe in zip(actual, probes)).backward()
    sum((value * probe).sum() for value, probe in zip(expected, probes)).backward()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=3e-12, atol=3e-12)
    for a, b in ((hidden, reference_hidden), (features, reference_features)):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-9, atol=2e-11)
    for name, parameter in memory.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                   rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("routing", ["shared", "split"])
def test_padding_skips_positions_but_unobserved_valid_event_breaks_adjacency(routing):
    memory = TaskRoutedEpisodeMemory(8, routing).double()
    with torch.no_grad():
        memory.time_query_projection.weight.zero_()
        memory.quantity_query_projection.weight.zero_()
    hidden = torch.randn(1, 7, 8, dtype=torch.float64)
    features = torch.tensor([[[0., 0.], [99., 99.], [1., 3.], [200., 500.],
                              [17., 21.], [99., 99.], [4., 5.]]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True, True, True, False, True]])
    write = torch.tensor([[True, True, True, False, True, True, True]])
    # Only (0,2) and (4,6) are complete. The unobserved valid event 3
    # excludes (2,3) and (3,4) without inventing a bridging (2,4) pair.
    phi = lambda values: F.gelu(memory.value_projection(values), approximate="none")
    for context in memory.read(hidden, features, mask, memory_write_mask=write):
        torch.testing.assert_close(context[0, 2], phi(features[0, 2]), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(context[0, 4], phi(features[0, 2]), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(context[0, 6], phi(features[0, [2, 6]]).mean(0), rtol=1e-12, atol=1e-12)
        assert torch.count_nonzero(context[~(mask & write)]) == 0


@pytest.mark.parametrize("routing", ["shared", "split"])
@pytest.mark.parametrize("observations", [0, 1])
def test_nonzero_value_bias_cannot_leak_to_empty_or_single_observation(routing, observations):
    memory = TaskRoutedEpisodeMemory(8, routing)
    activate_memory(memory)
    with torch.no_grad():
        memory.value_projection.bias.fill_(2.)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    features = torch.randn(2, 5, 2, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    write = torch.zeros_like(mask)
    if observations:
        write[:, 2] = True
    corrections = memory(hidden, features, mask, memory_write_mask=write)
    assert all(torch.isfinite(value).all() and torch.count_nonzero(value) == 0 for value in corrections)
    sum(value.sum() for value in corrections).backward()
    for value in (hidden, features, *memory.parameters()):
        if value.grad is not None:
            assert torch.isfinite(value.grad).all() and torch.count_nonzero(value.grad) == 0


@pytest.mark.parametrize("routing", ["shared", "split"])
def test_invalid_nan_features_and_hidden_are_sanitized_before_nonlin(routing):
    torch.manual_seed(83)
    memory = TaskRoutedEpisodeMemory(8, routing).double()
    hidden = torch.randn(1, 4, 8, dtype=torch.float64)
    features = torch.rand(1, 4, 2, dtype=torch.float64)
    positions = torch.tensor([1, 2, 4, 6])
    padded_hidden = torch.full((1, 8, 8), float("nan"), dtype=torch.float64)
    padded_features = torch.full((1, 8, 2), float("inf"), dtype=torch.float64)
    padded_hidden[:, positions], padded_features[:, positions] = hidden, features
    padded_hidden.requires_grad_()
    padded_features.requires_grad_()
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, positions] = True
    expected = memory.read(hidden, features, torch.ones(1, 4, dtype=torch.bool))
    padded = memory.read(padded_hidden, padded_features, mask)
    for actual, reference in zip(padded, expected):
        torch.testing.assert_close(actual[:, positions], reference, rtol=1e-12, atol=1e-12)
        assert torch.count_nonzero(actual[~mask]) == 0
    sum(value.sum() for value in padded).backward()
    for tensor in (padded_hidden, padded_features):
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad[~mask]) == 0
    # A valid but unobserved event is sanitized too, while retaining its gap.
    mask[:, 3] = True
    write = mask.clone()
    write[:, 3] = False
    clean_hidden, clean_features = padded_hidden.detach().clone(), padded_features.detach().clone()
    clean_hidden[:, 3], clean_features[:, 3] = 0., 0.
    for a, b in zip(memory.read(padded_hidden, padded_features, mask, memory_write_mask=write),
                    memory.read(clean_hidden, clean_features, mask, memory_write_mask=write)):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="finite"):
        memory.read(padded_hidden, padded_features, mask)


def paired_memories():
    torch.manual_seed(73)
    shared = TaskRoutedEpisodeMemory(8, "shared").double()
    split = TaskRoutedEpisodeMemory(8, "split").double()
    activate_memory(shared)
    with torch.no_grad():
        for name, parameter in shared.named_parameters():
            dict(split.named_parameters())[name].copy_(parameter)
    return shared, split


def test_query_swap_preserves_shared_mixture_and_exchanges_split_reads():
    shared, split = paired_memories()
    hidden = torch.randn(2, 7, 8, dtype=torch.float64)
    features = torch.rand(2, 7, 2, dtype=torch.float64) * 10
    mask = torch.ones(2, 7, dtype=torch.bool)
    shared_before, split_before = shared.read(hidden, features, mask), split.read(hidden, features, mask)
    assert torch.equal(shared_before[0], shared_before[1])
    assert not torch.allclose(split_before[0][:, -1], split_before[1][:, -1])
    torch.testing.assert_close(shared_before[0], (split_before[0] + split_before[1]) / 2, rtol=0., atol=0.)
    for memory in (shared, split):
        with torch.no_grad():
            previous = memory.time_query_projection.weight.clone()
            memory.time_query_projection.weight.copy_(memory.quantity_query_projection.weight)
            memory.quantity_query_projection.weight.copy_(previous)
    shared_after, split_after = shared.read(hidden, features, mask), split.read(hidden, features, mask)
    assert all(torch.equal(a, b) for a, b in zip(shared_before, shared_after))
    assert torch.equal(split_before[0], split_after[1]) and torch.equal(split_before[1], split_after[0])


@pytest.mark.parametrize("case", ["equal_queries", "single_record"])
def test_expected_degeneracies_collapse_to_identical_reads(case):
    shared, split = paired_memories()
    length = 2 if case == "single_record" else 6
    if case == "equal_queries":
        for memory in (shared, split):
            with torch.no_grad():
                memory.quantity_query_projection.weight.copy_(memory.time_query_projection.weight)
    hidden = torch.randn(2, length, 8, dtype=torch.float64)
    features = torch.rand(2, length, 2, dtype=torch.float64)
    mask = torch.ones(2, length, dtype=torch.bool)
    contexts = (*shared.read(hidden, features, mask), *split.read(hidden, features, mask))
    assert all(torch.equal(value, contexts[0]) for value in contexts[1:])


@pytest.mark.parametrize("routing", ["shared", "split"])
@pytest.mark.parametrize("branch", [0, 1])
def test_each_task_only_reaches_its_query_in_split_and_both_queries_in_shared(routing, branch):
    torch.manual_seed(97)
    memory = TaskRoutedEpisodeMemory(12, routing).double()
    activate_memory(memory)
    hidden = torch.randn(2, 8, 12, dtype=torch.float64)
    features = torch.rand(2, 8, 2, dtype=torch.float64)
    output = memory(hidden, features, torch.ones(2, 8, dtype=torch.bool))[branch]
    (output * torch.randn_like(output)).sum().backward()
    expected_active = {"key_projection.weight", "value_projection.weight", "value_projection.bias",
                       ("time_output_projection.weight", "quantity_output_projection.weight")[branch]}
    queries = ("time_query_projection.weight", "quantity_query_projection.weight")
    expected_active.update(queries if routing == "shared" else (queries[branch],))
    for name, parameter in memory.named_parameters():
        if name in expected_active:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad) > 0, name
        else:
            assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0, name


@pytest.mark.parametrize("routing", ["shared", "split"])
def test_core_future_prefix_causality_and_batch_independence(routing):
    torch.manual_seed(99)
    memory = TaskRoutedEpisodeMemory(8, routing).double()
    hidden, features = torch.randn(2, 7, 8, dtype=torch.float64), torch.rand(2, 7, 2, dtype=torch.float64)
    mask = torch.ones(2, 7, dtype=torch.bool)
    expected = memory.read(hidden, features, mask)
    altered_hidden, altered_features = hidden.clone(), features.clone()
    altered_hidden[:, 4:] *= 100
    altered_features[:, 4:] += 500
    altered = memory.read(altered_hidden, altered_features, mask)
    for a, b in zip(expected, altered):
        assert torch.equal(a[:, :4], b[:, :4])
    separate = [memory.read(hidden[i:i + 1], features[i:i + 1], mask[i:i + 1]) for i in range(2)]
    for branch in (0, 1):
        torch.testing.assert_close(expected[branch], torch.cat([row[branch] for row in separate]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nonconstant_quantity_head", [False, True])
def test_B_initial_state_outputs_objective_common_gradients_and_rng_are_exact(nonconstant_quantity_head):
    models, initial_rng = [], []
    for cls in (CountAwareTitanTPP, *CLASSES):
        torch.manual_seed(101)
        models.append(cls(**model_kwargs()))
        initial_rng.append(torch.get_rng_state().clone())
    assert all(torch.equal(initial_rng[0], state) for state in initial_rng[1:])
    baseline_state = models[0].state_dict()
    for model in models[1:]:
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in baseline_state.items())
        memory = model.task_routed_episode_memory
        assert not torch.equal(memory.time_query_projection.weight, memory.quantity_query_projection.weight)
        assert torch.count_nonzero(memory.time_output_projection.weight) == 0
        assert torch.count_nonzero(memory.quantity_output_projection.weight) == 0
    for name, parameter in models[1].named_parameters():
        assert torch.equal(parameter, dict(models[2].named_parameters())[name])
    dts, quantities, mask = batch()
    outputs, rng_states = [], []
    for model in models:
        model.train()
        if nonconstant_quantity_head:
            with torch.no_grad():
                model.quantity_head.weight.copy_(torch.linspace(-.3, .4, 16)[None])
        torch.manual_seed(902)
        outputs.append(target_outputs(model, dts, mask, quantities, lambda_log_qty=1.))
        rng_states.append(torch.get_rng_state().clone())
        outputs[-1]["joint_loss"].mean().backward()
    for output, rng in zip(outputs[1:], rng_states[1:]):
        assert torch.equal(rng, rng_states[0])
        assert all(torch.equal(value, output[key]) for key, value in outputs[0].items())
    for name, parameter in models[0].named_parameters():
        for model in models[1:]:
            other = dict(model.named_parameters())[name].grad
            assert (parameter.grad is None and other is None) or torch.equal(parameter.grad, other), name
    for model in models[1:]:
        for name, parameter in model.task_routed_episode_memory.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            active = name == "time_output_projection.weight" or nonconstant_quantity_head and name == "quantity_output_projection.weight"
            assert bool(torch.count_nonzero(parameter.grad)) == active, name


@pytest.mark.parametrize("cls", CLASSES)
def test_activated_joint_objective_reaches_every_new_parameter(cls):
    torch.manual_seed(103)
    model = cls(**model_kwargs()).eval()
    activate_model(model)
    dts, quantities, mask = batch()
    target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)["joint_loss"].mean().backward()
    for name, parameter in model.task_routed_episode_memory.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


@pytest.mark.parametrize("cls", CLASSES)
def test_activated_model_does_not_read_future_time_or_target_quantity(cls):
    torch.manual_seed(107)
    model = cls(**model_kwargs()).eval()
    activate_model(model)
    dts, quantities, mask = batch()
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    changed_dts[:, 3:] += 10000.
    changed_quantities[:, 3:] += 20000.
    with torch.no_grad():
        initial = model.encode_task_states(dts, quantities, mask)
        changed = model.encode_task_states(changed_dts, changed_quantities, mask)
    assert all(torch.equal(a[:, :3], b[:, :3]) for a, b in zip(initial, changed))
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    rows, target = torch.arange(len(dts)), mask.sum(1) - 1
    changed_dts[rows, target] += 10000.
    changed_quantities[rows, target] += 20000.
    with torch.no_grad():
        expected = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
        actual = target_outputs(model, changed_dts, mask, changed_quantities, lambda_log_qty=1.)
    assert torch.equal(actual["pred_qty"], expected["pred_qty"])
    assert not torch.equal(actual["time_loss"], expected["time_loss"])
    assert not torch.equal(actual["quantity_train_loss"], expected["quantity_train_loss"])
    dts.requires_grad_()
    quantities.requires_grad_()
    target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)["pred_qty"].sum().backward()
    for value in (dts, quantities):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[rows, target]) == 0
        assert torch.count_nonzero(value.grad[:, 1:3]) > 0


@pytest.mark.parametrize("cls", CLASSES)
def test_runner_compacts_padding_and_model_read_restore_has_no_external_state(cls):
    torch.manual_seed(113)
    model = cls(**model_kwargs()).eval()
    activate_model(model)
    dts, quantities, mask = batch()
    positions = torch.tensor([1, 3, 4, 6, 8])
    padded_dts, padded_quantities = torch.full((2, 9), 99999.), torch.full((2, 9), 99999.)
    padded_mask = torch.zeros(2, 9, dtype=torch.bool)
    padded_dts[:, positions], padded_quantities[:, positions], padded_mask[:, positions] = dts, quantities, mask
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    with torch.no_grad():
        expected = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
        padded = target_outputs(model, padded_dts, padded_mask, padded_quantities, lambda_log_qty=1.)
        model.encode_task_states(dts * 10, quantities * 20, mask)
        repeated = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
    stream.seek(0)
    restored = cls(**model_kwargs()).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        replay = target_outputs(restored, dts, mask, quantities, lambda_log_qty=1.)
    for name in expected:
        torch.testing.assert_close(padded[name], expected[name], rtol=1e-6, atol=1e-6)
        assert torch.equal(repeated[name], expected[name]), name
        assert torch.equal(replay[name], expected[name]), name


@pytest.mark.parametrize("strict", [True, False])
def test_direct_load_rejects_wrong_missing_or_noninteger_routing_identity(strict):
    shared = CountAwareTitanTaskSharedMemoryTPP(**model_kwargs())
    split = CountAwareTitanTaskSplitMemoryTPP(**model_kwargs())
    for recipient, source in ((shared, split), (split, shared)):
        with pytest.raises(RuntimeError, match="routing identity"):
            recipient.load_state_dict(source.state_dict(), strict=strict)
    state = shared.state_dict()
    del state["task_routed_episode_memory.routing_code"]
    with pytest.raises(RuntimeError, match="routing identity"):
        shared.load_state_dict(state, strict=strict)
    state["task_routed_episode_memory.routing_code"] = torch.tensor(0.)
    with pytest.raises(RuntimeError, match="routing identity"):
        shared.load_state_dict(state, strict=strict)


@pytest.mark.parametrize("name,value", [
    ("quantity_variant", "lognormal"), ("lambda_tail", .1), ("memory_mode", "none"),
    ("time_head_mode", "lognormal"), ("quantity_memory_gradient_mode", "detached"),
    ("quantile_adaptive_strength", .1), ("routing", "split"), ("association", "same_event"),
])
def test_constructor_rejects_changed_head_objective_or_route(name, value):
    kwargs = {**model_kwargs(), name: value}
    for cls in CLASSES:
        with pytest.raises(ValueError):
            cls(**kwargs)


@pytest.mark.parametrize("max_len,baseline_count", [(64, 77507), (256, 89795)])
def test_fixed_capacity_and_design_fingerprint_match_frozen_contract(max_len, baseline_count):
    baseline = CountAwareTitanTPP(**model_kwargs(64, max_len))
    assert sum(parameter.numel() for parameter in baseline.parameters()) == baseline_count
    for cls, routing in zip(CLASSES, ("shared", "split")):
        model = cls(**model_kwargs(64, max_len))
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count == baseline_count + 3632 and count / baseline_count < 1.05
        assert task_routed_memory_metadata(64, routing)["additional_parameter_count"] == 3632
    design = Path(__file__).resolve().parents[3] / "paper/contracts/hard_lmm_task_routed_episode_design_v1.json"
    assert hashlib.sha256(design.read_bytes()).hexdigest() == TASK_ROUTED_MEMORY_DESIGN_SHA256
