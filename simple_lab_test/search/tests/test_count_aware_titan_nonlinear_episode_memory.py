"""Causal and algebraic contracts for nonlinear episode values, CPU fixtures only."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanNonlinearEpisodeMemory import (
    CountAwareTitanNonlinearPostPoolTPP,
    CountAwareTitanNonlinearPrePoolTPP,
    NonlinearEpisodeMemory,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


CLASSES = (CountAwareTitanNonlinearPostPoolTPP, CountAwareTitanNonlinearPrePoolTPP)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def model_kwargs(hidden_dim=16):
    return dict(hidden_dim=hidden_dim, train_log_mean=1.5, max_seq_len=12,
                quantity_variant="count_only_log_regression", lambda_tail=0.0,
                time_head_mode="legacy_clamped_rmtpp", time_intercept_limit=300.0)


def models_from_same_seed(seed=41, hidden_dim=16):
    models, states = [], []
    for cls in (CountAwareTitanTPP, *CLASSES):
        torch.manual_seed(seed)
        models.append(cls(**model_kwargs(hidden_dim)))
        states.append(torch.get_rng_state().clone())
    assert all(torch.equal(states[0], state) for state in states[1:])
    return models


def batch():
    return (
        torch.tensor([[0., 1., 2., 4., 8.], [0., .5, 3., 6., 0.]]),
        torch.tensor([[2., 5., 9., 12., 20.], [1., 7., 13., 19., 0.]]),
        torch.tensor([[True, True, True, True, True], [True, True, True, True, False]]),
    )


def activate(model):
    memory = model.nonlinear_episode_memory
    with torch.no_grad():
        memory.output_projection.weight.copy_(torch.linspace(-.2, .3, 256).reshape(16, 16))
        memory.value_projection.bias.copy_(torch.linspace(-.4, .3, 16))
        model.quantity_head.weight.copy_(torch.linspace(-.3, .4, 16)[None])


def oracle(memory, hidden, features, mask, write, pooling):
    """Scalar query loops enumerate valid-event adjacency without model helpers."""
    rows = []
    width = hidden.shape[-1]
    for row in range(hidden.shape[0]):
        valid = [i for i in range(hidden.shape[1]) if bool(mask[row, i])]
        records = [(j, k) for j, k in zip(valid, valid[1:])
                   if bool(write[row, j]) and bool(write[row, k])]
        outputs = []
        for t in range(hidden.shape[1]):
            available = [(j, k) for j, k in records if k <= t]
            if not bool(mask[row, t] & write[row, t]) or not available:
                outputs.append(features.new_zeros(16))
                continue
            query = F.linear(F.layer_norm(hidden[row, t], (width,), eps=1e-5),
                             memory.query_projection.weight)
            scores = []
            values = []
            for j, k in available:
                key = F.linear(F.layer_norm(hidden[row, j], (width,), eps=1e-5),
                               memory.key_projection.weight)
                scores.append((query * key).sum() / math.sqrt(16))
                values.append(features[row, k])
            weights = torch.softmax(torch.stack(scores), dim=0)
            if pooling == "pre":
                values = [F.gelu(F.linear(value, memory.value_projection.weight,
                                         memory.value_projection.bias), approximate="none")
                          for value in values]
                result = sum(weight * value for weight, value in zip(weights, values))
            else:
                mean = sum(weight * value for weight, value in zip(weights, values))
                result = F.gelu(F.linear(mean, memory.value_projection.weight,
                                        memory.value_projection.bias), approximate="none")
            outputs.append(result)
        rows.append(torch.stack(outputs))
    return torch.stack(rows)


@pytest.mark.parametrize("pooling", ["pre", "post"])
def test_scalar_prefix_oracle_values_and_all_gradients_with_sparse_observations(pooling):
    torch.manual_seed(59)
    memory = NonlinearEpisodeMemory(12, pooling=pooling).double()
    with torch.no_grad():
        memory.value_projection.bias.uniform_(-.5, .5)
        memory.output_projection.weight.normal_(std=.2)
    reference = copy.deepcopy(memory)
    hidden = torch.randn(3, 7, 12, dtype=torch.float64, requires_grad=True)
    features = torch.rand(3, 7, 2, dtype=torch.float64, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_features = features.detach().clone().requires_grad_()
    mask = torch.tensor([[False, True, True, False, True, True, True],
                         [True, True, False, True, True, False, True], [False] * 7])
    write = torch.tensor([[True, True, True, True, False, True, True],
                          [True, True, True, True, False, True, True], [True] * 7])
    actual = memory(hidden, features, mask, memory_write_mask=write)
    expected = F.linear(oracle(reference, reference_hidden, reference_features, mask, write, pooling),
                        reference.output_projection.weight)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-12)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for actual_input, expected_input in ((hidden, reference_hidden), (features, reference_features)):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=2e-9, atol=2e-11)
    for name, parameter in memory.named_parameters():
        expected_parameter = dict(reference.named_parameters())[name]
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("pooling", ["pre", "post"])
def test_padding_skips_positions_but_unobserved_valid_event_breaks_the_pair(pooling):
    memory = NonlinearEpisodeMemory(8, pooling=pooling).double()
    with torch.no_grad():
        memory.query_projection.weight.zero_()
        memory.key_projection.weight.zero_()
    hidden = torch.randn(1, 7, 8, dtype=torch.float64)
    features = torch.tensor([[[0., 0.], [99., 99.], [1., 3.], [200., 500.],
                              [17., 21.], [99., 99.], [4., 5.]]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True, True, True, False, True]])
    write = torch.tensor([[True, True, True, False, True, True, True]])
    # The only completed pairs are (0,2) and (4,6); a false write flag at
    # valid event 3 must not turn (2,4) into a new completed pair.
    result = memory.read(hidden, features, mask, memory_write_mask=write)
    phi = lambda x: F.gelu(memory.value_projection(x), approximate="none")
    torch.testing.assert_close(result[0, 2], phi(features[0, 2]), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(result[0, 4], phi(features[0, 2]), rtol=1e-12, atol=1e-12)
    expected = phi(features[0, [2, 6]]).mean(0) if pooling == "pre" else phi(features[0, [2, 6]].mean(0))
    torch.testing.assert_close(result[0, 6], expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(result[~(mask & write)]) == 0


@pytest.mark.parametrize("pooling", ["pre", "post"])
@pytest.mark.parametrize("observations", [0, 1])
def test_nonzero_value_bias_cannot_leak_into_empty_or_unobserved_queries(pooling, observations):
    memory = NonlinearEpisodeMemory(8, pooling=pooling)
    with torch.no_grad():
        memory.value_projection.bias.fill_(2.)
        memory.output_projection.weight.fill_(.3)
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


def paired_memories():
    torch.manual_seed(73)
    pre = NonlinearEpisodeMemory(8, pooling="pre").double()
    post = NonlinearEpisodeMemory(8, pooling="post").double()
    with torch.no_grad():
        pre.output_projection.weight.normal_(std=.2)
        pre.value_projection.bias.uniform_(-.4, .4)
        for name, parameter in pre.named_parameters():
            dict(post.named_parameters())[name].copy_(parameter)
    return pre, post


def test_equal_log_feature_means_hide_different_outcomes_only_after_pooling():
    pre, post = paired_memories()
    with torch.no_grad():
        for memory in (pre, post):
            memory.query_projection.weight.zero_()
            memory.key_projection.weight.zero_()
            memory.value_projection.weight.zero_()
            memory.value_projection.weight[:, 1] = torch.linspace(.2, 1.5, 16)
            memory.value_projection.bias.zero_()
            memory.output_projection.weight.zero_()
            memory.output_projection.weight[0, 0] = 1.
    hidden = torch.zeros(1, 3, 8, dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    varied = torch.tensor([[[0., 0.], [0., math.log(2.)], [0., math.log(8.)]]], dtype=torch.float64)
    constant = varied.clone()
    constant[0, 1:, 1] = math.log(4.)
    torch.testing.assert_close(varied[:, 1:].mean(1), constant[:, 1:].mean(1), rtol=0., atol=1e-15)
    torch.testing.assert_close(post(hidden, varied, mask)[:, -1], post(hidden, constant, mask)[:, -1], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(pre(hidden, varied, mask)[:, -1], pre(hidden, constant, mask)[:, -1], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("case", ["single", "identical"])
def test_nonlinear_pool_orders_agree_for_single_or_identical_record_values(case):
    pre, post = paired_memories()
    length = 2 if case == "single" else 6
    hidden = torch.randn(2, length, 8, dtype=torch.float64)
    features = torch.randn(2, length, 2, dtype=torch.float64)
    if case == "identical":
        features[:, 1:] = features[:, 1:2].clone()
    mask = torch.ones(2, length, dtype=torch.bool)
    torch.testing.assert_close(pre(hidden, features, mask), post(hidden, features, mask), rtol=2e-12, atol=2e-12)


def test_affine_value_map_commutes_with_weighted_pooling(monkeypatch):
    pre, post = paired_memories()
    hidden = torch.randn(2, 6, 8, dtype=torch.float64)
    features = torch.randn(2, 6, 2, dtype=torch.float64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    # This counterfactual replaces only GELU. Nonzero bias ensures the
    # property includes affine maps and the empty-row normalization rule.
    monkeypatch.setattr(F, "gelu", lambda value, **_: value)
    torch.testing.assert_close(pre(hidden, features, mask), post(hidden, features, mask), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("pooling", ["pre", "post"])
def test_invalid_values_are_sanitized_before_nonlin_and_do_not_bridge_records(pooling):
    torch.manual_seed(83)
    memory = NonlinearEpisodeMemory(8, pooling=pooling).double()
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
    original = memory.read(hidden, features, torch.ones(1, 4, dtype=torch.bool))
    padded = memory.read(padded_hidden, padded_features, mask)
    torch.testing.assert_close(padded[:, positions], original, rtol=2e-12, atol=2e-12)
    assert torch.count_nonzero(padded[~mask]) == 0
    assert torch.isfinite(padded).all()
    padded.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in memory.parameters())
    for value in (padded_hidden, padded_features):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[~mask]) == 0


@pytest.mark.parametrize("pooling", ["pre", "post"])
def test_prefix_calls_and_batch_permutations_do_not_carry_memory_or_mutate_state(pooling):
    torch.manual_seed(89)
    memory = NonlinearEpisodeMemory(8, pooling=pooling).double()
    hidden = torch.randn(3, 6, 8, dtype=torch.float64)
    features = torch.randn(3, 6, 2, dtype=torch.float64)
    mask = torch.ones(3, 6, dtype=torch.bool)
    write = mask.clone()
    write[1, 3] = False
    snapshot = copy.deepcopy(memory.state_dict())
    original = memory.read(hidden, features, mask, memory_write_mask=write)
    single = torch.cat([memory.read(hidden[i:i+1], features[i:i+1], mask[i:i+1], memory_write_mask=write[i:i+1]) for i in range(3)])
    order = torch.tensor([2, 0, 1])
    permuted = memory.read(hidden[order], features[order], mask[order], memory_write_mask=write[order])
    memory.read(hidden * 10, features * 20, mask)
    assert torch.equal(memory.read(hidden, features, mask, memory_write_mask=write), original)
    torch.testing.assert_close(single, original, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(permuted, original[order], rtol=2e-12, atol=2e-12)
    for end in range(1, 7):
        prefix = memory.read(hidden[:, :end], features[:, :end], mask[:, :end], memory_write_mask=write[:, :end])
        torch.testing.assert_close(prefix, original[:, :end], rtol=2e-12, atol=2e-12)
    for name, value in memory.state_dict().items():
        assert torch.equal(snapshot[name], value), name


@pytest.mark.parametrize("pooling", ["pre", "post"])
def test_output_learns_first_then_address_and_value_weights_and_bias(pooling):
    torch.manual_seed(97)
    memory = NonlinearEpisodeMemory(8, pooling=pooling).double()
    hidden = torch.randn(2, 6, 8, dtype=torch.float64)
    features = torch.rand(2, 6, 2, dtype=torch.float64) * 4
    target = torch.randn(2, 6, 8, dtype=torch.float64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    optimizer = torch.optim.SGD(memory.parameters(), lr=.01)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (memory(hidden, features, mask) - target).square().sum().backward()
        for name, parameter in memory.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert bool(torch.count_nonzero(parameter.grad)) == (step > 0 or name == "output_projection.weight"), name
        optimizer.step()


def test_h64_budget_and_initial_shared_state_rng_and_new_weights():
    baseline, post, pre = models_from_same_seed(hidden_dim=64)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    for model in (post, pre):
        assert sum(p.numel() for p in model.parameters()) - baseline_count == 3120
        assert torch.count_nonzero(model.nonlinear_episode_memory.output_projection.weight) == 0
        assert torch.count_nonzero(model.nonlinear_episode_memory.value_projection.bias) == 0
        for name, value in baseline.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
    for name, value in post.nonlinear_episode_memory.named_parameters():
        assert torch.equal(value, dict(pre.nonlinear_episode_memory.named_parameters())[name]), name
    assert post.nonlinear_episode_memory.pooling_code.item() == 0
    assert pre.nonlinear_episode_memory.pooling_code.item() == 1


@pytest.mark.parametrize("nonconstant_quantity_head", [False, True])
def test_initial_b_outputs_losses_preclip_shared_gradients_and_dropout_rng_are_exact(nonconstant_quantity_head):
    models = models_from_same_seed(seed=101)
    dts, quantities, mask = batch()
    outputs, rng_states = [], []
    for model in models:
        model.train()
        if nonconstant_quantity_head:
            with torch.no_grad():
                model.quantity_head.weight.copy_(torch.linspace(-.3, .4, 16)[None])
        torch.manual_seed(902)
        outputs.append(target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0))
        rng_states.append(torch.get_rng_state().clone())
        outputs[-1]["joint_loss"].mean().backward()
    for output, rng in zip(outputs[1:], rng_states[1:]):
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
        for name, parameter in model.nonlinear_episode_memory.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert bool(torch.count_nonzero(parameter.grad)) == (name == "output_projection.weight")


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("objective", ["time_loss", "quantity_train_loss"])
def test_active_memory_reaches_all_parameters_through_each_existing_head(cls, objective):
    torch.manual_seed(103)
    model = cls(**model_kwargs()).eval()
    activate(model)
    dts, quantities, mask = batch()
    target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)[objective].mean().backward()
    for name, parameter in model.nonlinear_episode_memory.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


@pytest.mark.parametrize("cls", CLASSES)
def test_nonzero_memory_cannot_read_future_time_or_target_quantity(cls):
    torch.manual_seed(107)
    model = cls(**model_kwargs()).eval()
    activate(model)
    dts, quantities, mask = batch()
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    changed_dts[:, 3:] += 10000.
    changed_quantities[:, 3:] += 20000.
    with torch.no_grad():
        initial = model.encode(dts, quantities, mask)
        changed = model.encode(changed_dts, changed_quantities, mask)
    assert torch.equal(initial[:, :3], changed[:, :3])
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    rows, target = torch.arange(len(dts)), mask.sum(1) - 1
    changed_dts[rows, target] += 10000.
    changed_quantities[rows, target] += 20000.
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
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[rows, target]) == 0
        assert torch.count_nonzero(value.grad[:, 1:3]) > 0


@pytest.mark.parametrize("cls", CLASSES)
def test_actual_runner_compacts_padding_and_repeated_calls_preserve_predictions(cls):
    torch.manual_seed(113)
    model = cls(**model_kwargs()).eval()
    activate(model)
    dts, quantities, mask = batch()
    positions = torch.tensor([1, 3, 4, 6, 8])
    padded_dts, padded_quantities = torch.full((2, 9), 99999.), torch.full((2, 9), 99999.)
    padded_mask = torch.zeros(2, 9, dtype=torch.bool)
    padded_dts[:, positions], padded_quantities[:, positions], padded_mask[:, positions] = dts, quantities, mask
    snapshot = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        expected = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
        padded = target_outputs(model, padded_dts, padded_mask, padded_quantities, lambda_log_qty=1.)
        model.encode(dts * 10, quantities * 20, mask)
        repeated = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
    for name in expected:
        torch.testing.assert_close(padded[name], expected[name], rtol=1e-6, atol=1e-6)
        assert torch.equal(repeated[name], expected[name]), name
    for name, value in model.state_dict().items():
        assert torch.equal(snapshot[name], value), name


def test_direct_load_rejects_opposite_pooling_route_even_nonstrict():
    _, post, pre = models_from_same_seed()
    with pytest.raises(RuntimeError, match="[Pp]ooling|[Ii]dentity"):
        pre.load_state_dict(post.state_dict(), strict=False)
    with pytest.raises(RuntimeError, match="[Pp]ooling|[Ii]dentity"):
        post.load_state_dict(pre.state_dict(), strict=False)
