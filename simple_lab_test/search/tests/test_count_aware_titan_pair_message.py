"""Synthetic CPU contracts for event-wise versus pooled nonlinear messages."""
from __future__ import annotations

import copy
import io
import math

import pytest
import torch

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanPairMessage import (
    CountAwareTitanPairMessagePrePoolTPP, CountAwareTitanPairMessagePostPoolTPP,
    PairMessage, aggregate_pair_messages, pair_message_metadata, validate_pair_message_checkpoint,
    PAIR_MESSAGE_PRE_POOL_BACKBONE, PAIR_MESSAGE_POST_POOL_BACKBONE,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs

CLASSES = (CountAwareTitanPairMessagePrePoolTPP, CountAwareTitanPairMessagePostPoolTPP)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def kwargs():
    return dict(hidden_dim=16, train_log_mean=1.5, max_seq_len=12,
                quantity_variant="count_only_log_regression", lambda_tail=0.0,
                time_head_mode="heteroscedastic_lognormal_duration", time_scale=3.0,
                time_initial_location=0.1, time_initial_scale=0.7,
                time_observation_contract=dict(mode="positive_integer_round_clamp_v1", top_code=None, unit="week"))


def inputs():
    return (torch.tensor([[0., 1., 3., 2., 7., 4.], [0., 2., 1., 3., 4., 0.]]),
            torch.tensor([[1., 2., 6., 3., 12., 5.], [2., 1., 4., 8., 3., 0.]]),
            torch.tensor([[True] * 6, [True] * 5 + [False]]))


def activate(model):
    with torch.no_grad():
        model.pair_message.output_projection.weight.copy_(
            torch.linspace(-.4, .3, model.hidden_dim * 8).reshape(model.hidden_dim, 8))
        model.quantity_head.weight.copy_(torch.linspace(-.3, .4, model.hidden_dim).unsqueeze(0))


def objective(model, dts, quantities, mask, component="joint_loss"):
    return target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)[component].mean()


def test_scalar_oracle_distinguishes_nonlinear_order_and_preserves_mass():
    q = torch.zeros(1, 1, 1, dtype=torch.float64)
    k = torch.tensor([[[-2.], [2.]]], dtype=torch.float64)
    v = torch.tensor([[[-1.], [1.]]], dtype=torch.float64)
    w = torch.tensor([[[.5, .5]]], dtype=torch.float64)
    before = aggregate_pair_messages(q, k, v, w, routing="pre_pool")
    after = aggregate_pair_messages(q, k, v, w, routing="post_pool")
    torch.testing.assert_close(before, torch.full_like(before, math.tanh(1.)), rtol=1e-14, atol=1e-14)
    assert torch.equal(after, torch.zeros_like(after))
    for route in ("pre_pool", "post_pool"):
        half = aggregate_pair_messages(q, k, v, w * .5, routing=route)
        full = aggregate_pair_messages(q, k, v, w, routing=route)
        torch.testing.assert_close(half, full * .5, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("case", ["single", "equal_keys", "empty"])
def test_order_reductions_match(case):
    torch.manual_seed(33)
    q, k, v = (torch.randn(2, 4, 8, dtype=torch.float64) for _ in range(3))
    w = torch.rand(2, 4, 4, dtype=torch.float64) * .1
    if case == "single":
        w[:, :, 1:] = 0.
    elif case == "equal_keys":
        k[:] = k[:, :1]
    else:
        w.zero_()
    a = aggregate_pair_messages(q, k, v, w, routing="pre_pool")
    b = aggregate_pair_messages(q, k, v, w, routing="post_pool")
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("route", ["pre_pool", "post_pool"])
def test_independent_loop_oracle_and_every_gradient(route):
    torch.manual_seed(41)
    module = PairMessage(12, route, chunk_size=2).double()
    reference = copy.deepcopy(module)
    hidden = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_()
    weights = (torch.rand(2, 5, 5, dtype=torch.float64) * .06).requires_grad_()
    other_weights = weights.detach().clone().requires_grad_()
    mask = torch.tensor([[True, False, True, True, True], [False] * 5])
    observed = torch.tensor([[True, True, True, False, True], [True] * 5])
    actual = module.read(hidden, weights, mask, memory_write_mask=observed)
    rows = []
    for b in range(2):
        positions = []
        for i in range(5):
            available = [j for j in range(i + 1) if bool(mask[b, j] & observed[b, j])]
            if not bool(mask[b, i] & observed[b, i]) or not available:
                positions.append(other_hidden[b, i] @ reference.query_projection.weight.T * 0.)
                continue
            q = reference.query_projection(other_hidden[b, i])
            kk = [reference.key_projection(other_hidden[b, j]) for j in available]
            vv = [reference.value_projection(other_hidden[b, j]) for j in available]
            ww = [other_weights[b, i, j] for j in available]
            if route == "pre_pool":
                value = sum(w * 2 * torch.sigmoid(q + k) * v for w, k, v in zip(ww, kk, vv))
            else:
                mass = sum(ww)
                key = sum(w * k for w, k in zip(ww, kk)) / mass
                val = sum(w * v for w, v in zip(ww, vv)) / mass
                value = mass * 2 * torch.sigmoid(q + key) * val
            positions.append(value)
        rows.append(torch.stack(positions))
    expected = torch.stack(rows)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for a, b in ((hidden, other_hidden), (weights, other_weights)):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-11, atol=2e-12)
    for name in ("query_projection", "key_projection", "value_projection"):
        torch.testing.assert_close(getattr(module, name).weight.grad, getattr(reference, name).weight.grad,
                                   rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_chunked_forward_and_all_input_gradients_equal_dense(checkpointed):
    torch.manual_seed(31)
    values = [torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    values.append(torch.rand(2, 7, 7, dtype=torch.float64, requires_grad=True))
    references = [v.detach().clone().requires_grad_() for v in values]
    actual = aggregate_pair_messages(*values, routing="pre_pool", chunk_size=2, recompute=checkpointed)
    expected = aggregate_pair_messages(*references, routing="pre_pool", chunk_size=7, recompute=False)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    for a, b in zip(values, references):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("training", [False, True])
def test_initial_B_parameters_predictions_objective_common_gradients_and_rng_match(cls, training):
    torch.manual_seed(123)
    baseline = CountAwareTitanTPP(**kwargs())
    base_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    model = cls(**kwargs())
    assert torch.equal(torch.get_rng_state(), base_rng)
    for name, param in baseline.named_parameters():
        assert torch.equal(param, dict(model.named_parameters())[name]), name
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == 4 * 16 * 8
    baseline.train(training)
    model.train(training)
    dts, quantities, mask = inputs()
    rng = torch.get_rng_state().clone()
    out_base = target_outputs(baseline, dts, mask, quantities, lambda_log_qty=1.)
    next_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    out_model = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
    assert torch.equal(next_rng, torch.get_rng_state())
    for name in out_base:
        assert torch.equal(out_base[name], out_model[name]), name
    out_base["joint_loss"].mean().backward()
    out_model["joint_loss"].mean().backward()
    assert torch.equal(next_rng, torch.get_rng_state())
    for name, param in baseline.named_parameters():
        other = dict(model.named_parameters())[name]
        if param.grad is None:
            assert other.grad is None, name
        else:
            torch.testing.assert_close(param.grad, other.grad, rtol=0., atol=0., msg=name)
    assert model.pair_message.output_projection.weight.grad.abs().sum() > 0
    for name in ("query_projection", "key_projection", "value_projection"):
        assert torch.count_nonzero(getattr(model.pair_message, name).weight.grad) == 0


def test_candidate_control_have_identical_parameter_initialization():
    torch.manual_seed(51)
    a = CLASSES[0](**kwargs())
    torch.manual_seed(51)
    b = CLASSES[1](**kwargs())
    assert dict(a.named_parameters()).keys() == dict(b.named_parameters()).keys()
    for name, p in a.named_parameters():
        assert torch.equal(p, dict(b.named_parameters())[name]), name


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("component", ["time_loss", "quantity_train_loss"])
def test_activated_route_receives_both_objective_gradients(cls, component):
    torch.manual_seed(64)
    model = cls(**kwargs()).eval()
    activate(model)
    dts, quantities, mask = inputs()
    objective(model, dts, quantities, mask, component).backward()
    for name, param in model.pair_message.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, name
    for name in ("encoder.input_proj.weight", "encoder.layers.0.attn.qkv.weight",
                 "encoder.layers.1.attn.qkv.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, name


@pytest.mark.parametrize("cls", CLASSES)
def test_future_padding_unobserved_gaps_and_batch_independence(cls):
    torch.manual_seed(74)
    model = cls(**kwargs()).eval()
    activate(model)
    dts, quantities, mask = inputs()
    expected = model.encode(dts, quantities, mask)
    future_dts, future_qty = dts.clone(), quantities.clone()
    future_dts[:, 4:], future_qty[:, 4:] = 500., 10000.
    changed = model.encode(future_dts, future_qty, mask)
    assert torch.equal(expected[:, :4], changed[:, :4])
    poison_dts, poison_qty = dts.clone(), quantities.clone()
    poison_dts[~mask], poison_qty[~mask] = float("nan"), float("inf")
    assert torch.equal(expected, model.encode(poison_dts, poison_qty, mask))
    for b in range(2):
        torch.testing.assert_close(expected[b:b+1], model.encode(dts[b:b+1], quantities[b:b+1], mask[b:b+1]),
                                   rtol=2e-6, atol=2e-6)
    # An unobserved middle event must not enter later states indirectly via block 1.
    write = mask.clone()
    write[:, 2] = False
    clean = model.encode_task_states(dts, quantities, mask, memory_write_mask=write)[0]
    poison_dts[:, 2], poison_qty[:, 2] = float("inf"), float("nan")
    poisoned = model.encode_task_states(poison_dts, poison_qty, mask, memory_write_mask=write)[0]
    assert torch.equal(clean, poisoned)
    assert torch.count_nonzero(poisoned[~write]) == 0
    mask_gap = mask.clone()
    mask_gap[:, 2] = False
    assert torch.equal(clean, model.encode(poison_dts, poison_qty, mask_gap))


@pytest.mark.parametrize("route", ["pre_pool", "post_pool"])
def test_message_masks_invalid_values_before_transforms_and_empty_gradients(route):
    module = PairMessage(8, route).double()
    hidden = torch.randn(2, 4, 8, dtype=torch.float64)
    mask = torch.tensor([[True, False, True, False], [False] * 4])
    hidden[~mask] = float("nan")
    hidden.requires_grad_()
    weights = torch.full((2, 4, 4), float("nan"), dtype=torch.float64)
    allowed = mask.unsqueeze(-1) & mask.unsqueeze(1) & torch.ones(4, 4, dtype=torch.bool).tril()
    weights[allowed] = .1
    weights.requires_grad_()
    read = module.read(hidden, weights, mask)
    assert torch.isfinite(read).all() and torch.count_nonzero(read[~mask]) == 0
    read.sum().backward()
    assert torch.isfinite(hidden.grad).all() and torch.count_nonzero(hidden.grad[~mask]) == 0
    assert torch.isfinite(weights.grad).all() and torch.count_nonzero(weights.grad[~allowed]) == 0


@pytest.mark.parametrize("cls", CLASSES)
def test_time_and_quantity_target_gradients_cannot_flow_into_predictor(cls):
    torch.manual_seed(77)
    model = cls(**kwargs()).eval()
    activate(model)
    dts, quantities, mask = inputs()
    dts.requires_grad_()
    quantities.requires_grad_()
    write = mask.clone()
    write[0, 5] = write[1, 4] = False
    states = model.encode_task_states(dts, quantities, mask, memory_write_mask=write)[0]
    prediction = model.quantity_outputs(states[torch.arange(2), torch.tensor([4, 3])], torch.ones(2))["point_prediction"]
    prediction.sum().backward()
    assert torch.isfinite(dts.grad).all() and torch.isfinite(quantities.grad).all()
    assert torch.count_nonzero(dts.grad[~write]) == 0
    assert torch.count_nonzero(quantities.grad[~write]) == 0
    changed_dt, changed_qty = dts.detach().clone(), quantities.detach().clone()
    changed_dt[~write], changed_qty[~write] = 900., 90000.
    actual = model.encode_task_states(changed_dt, changed_qty, mask, memory_write_mask=write)[0]
    assert torch.equal(states.detach(), actual)


@pytest.mark.parametrize("cls", CLASSES)
def test_save_restore_optimizer_rng_and_next_update_equal(cls):
    torch.manual_seed(81)
    model = cls(**kwargs()).train()
    activate(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    dts, quantities, mask = inputs()
    optimizer.zero_grad(set_to_none=True)
    objective(model, dts, quantities, mask).backward()
    optimizer.step()
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    clone = cls(**kwargs()).train()
    clone.load_state_dict(saved["model"])
    other_optimizer = torch.optim.Adam(clone.parameters(), lr=.001)
    other_optimizer.load_state_dict(saved["optimizer"])
    losses = []
    for current, opt in ((model, optimizer), (clone, other_optimizer)):
        torch.set_rng_state(saved["rng"])
        opt.zero_grad(set_to_none=True)
        loss = objective(current, dts, quantities, mask)
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    assert torch.equal(*losses)
    for name, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[name]), name


@pytest.mark.parametrize("strict", [False, True])
def test_wrong_or_absent_route_identity_rejected_before_mutation(strict):
    first, second = (cls(**kwargs()) for cls in CLASSES)
    before = copy.deepcopy(first.state_dict())
    with pytest.raises(RuntimeError, match="routing identity"):
        first.load_state_dict(second.state_dict(), strict=strict)
    for name, value in before.items():
        assert torch.equal(value, first.state_dict()[name])
    absent = copy.deepcopy(before)
    absent.pop("pair_message.routing_code")
    with pytest.raises(RuntimeError, match="routing identity"):
        first.load_state_dict(absent, strict=strict)


@pytest.mark.parametrize("cls,backbone,route", zip(CLASSES,
                         [PAIR_MESSAGE_PRE_POOL_BACKBONE, PAIR_MESSAGE_POST_POOL_BACKBONE], ["pre_pool", "post_pool"]))
def test_checkpoint_identity_and_relabelling_guard(cls, backbone, route):
    model = cls(**kwargs())
    head = {**model.time_head_contract(), "statistics_source_split": "train",
            "train_time_statistics": {"time_scale": 3.0, "target_log_scaled_mean": .1,
                                      "target_log_scaled_std": .7}}
    interface = {"time_head": head}
    payload = dict(backbone=backbone, variant="count_only_log_regression", evaluation_scope="validation_only",
                   held_out_test_evaluated=False, model_state_dict=model.state_dict(),
                   encoder_config={**pair_message_metadata(16, route), "time_head": model.time_head_contract()},
                   interface_meta=interface, resume_identity={"backbone": backbone,
                       "variant": "count_only_log_regression", "checkpoint_monitor": "validation_raw_quantity_rmse",
                       "interface_meta": copy.deepcopy(interface), "arguments": {
                           "dataset_contract": "intermittent_frozen_5000", "model_role": "observed_time_pair_message_v1",
                           "time_head_mode": head["mode"], "time_scale": 3.0}})
    assert validate_pair_message_checkpoint(payload, backbone)
    with pytest.raises(ValueError, match="relabelled"):
        validate_pair_message_checkpoint(payload, "titantpp")
    corrupt = copy.deepcopy(payload)
    corrupt["encoder_config"]["pair_message_routing"] = "unknown"
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_pair_message_checkpoint(corrupt, backbone)
    corrupt = copy.deepcopy(payload)
    corrupt["model_state_dict"]["pair_message.routing_code"].fill_(9)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_pair_message_checkpoint(corrupt, backbone)
    for field, value in (("observation_likelihood", None), ("mode", "legacy_clamped_rmtpp"),
                          ("time_scale", -1.), ("time_initial_location", float("nan"))):
        corrupt = copy.deepcopy(payload)
        corrupt["encoder_config"]["time_head"][field] = value
        with pytest.raises(ValueError):
            validate_pair_message_checkpoint(corrupt, backbone)
    corrupt = copy.deepcopy(payload)
    corrupt["resume_identity"]["interface_meta"]["time_head"]["time_initial_location"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_pair_message_checkpoint(corrupt, backbone)


def test_attention_opt_in_uses_actual_pre_dropout_weights_and_excludes_memory():
    model = CountAwareTitanTPP(**kwargs()).train()
    attention = model.encoder.layers[1].attn
    hidden = torch.randn(2, 5, 16)
    mask = torch.tensor([[True] * 5, [True, True, False, False, False]])
    captured = []
    handle = attention.drop.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    rng = torch.get_rng_state()
    legacy = attention(hidden, mask)
    next_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual, events = attention(hidden, mask, return_event_attention=True)
    handle.remove()
    assert len(captured) == 2 and torch.equal(captured[0], captured[1])
    assert torch.equal(legacy, actual) and torch.equal(next_rng, torch.get_rng_state())
    assert torch.equal(events, captured[1][..., 16:].mean(dim=1))
    assert torch.all(events.sum(dim=-1) < 1.)
