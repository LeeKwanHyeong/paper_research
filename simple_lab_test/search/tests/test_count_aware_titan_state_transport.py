"""Synthetic CPU contracts for elapsed/event-clock interlayer state transport."""
from __future__ import annotations

import copy
import io

import pytest
import torch

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanStateTransport import (
    CountAwareTitanStateTransport, BoundedStateTransport, affine_state_scan,
    state_transport_metadata, validate_state_transport_checkpoint,
    EVENT_STATE_BACKBONE, ELAPSED_STATE_BACKBONE, STATE_TRANSPORT_ROLE,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


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
        model.state_transport.output_projection.weight.copy_(
            torch.linspace(-.4, .3, model.hidden_dim * 8).reshape(model.hidden_dim, 8))
        model.quantity_head.weight.copy_(torch.linspace(-.3, .4, model.hidden_dim).unsqueeze(0))


@pytest.mark.parametrize("clock", ["event", "elapsed"])
@pytest.mark.parametrize("training", [False, True])
def test_initial_B_outputs_objective_common_preclip_gradients_rng(clock, training):
    torch.manual_seed(123)
    baseline = CountAwareTitanTPP(**kwargs())
    base_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    model = CountAwareTitanStateTransport(**kwargs(), state_transport_clock=clock)
    assert torch.equal(torch.get_rng_state(), base_rng)
    for name, p in baseline.named_parameters():
        assert torch.equal(p, dict(model.named_parameters())[name]), name
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == 400
    baseline.train(training)
    model.train(training)
    dts, qty, mask = inputs()
    rng = torch.get_rng_state().clone()
    a = target_outputs(baseline, dts, mask, qty, lambda_log_qty=1.)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    b = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    assert torch.equal(after, torch.get_rng_state())
    for key in a:
        assert torch.equal(a[key], b[key]), key
    a["joint_loss"].mean().backward()
    b["joint_loss"].mean().backward()
    for name, p in baseline.named_parameters():
        q = dict(model.named_parameters())[name]
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
    assert model.state_transport.output_projection.weight.grad.abs().sum() > 0
    for name, p in model.state_transport.named_parameters():
        if name != "output_projection.weight":
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0


def test_both_arms_have_identical_added_parameters_and_rank64_cost():
    torch.manual_seed(32)
    a = CountAwareTitanStateTransport(**kwargs(), state_transport_clock="event")
    torch.manual_seed(32)
    b = CountAwareTitanStateTransport(**kwargs(), state_transport_clock="elapsed")
    for name, p in a.named_parameters():
        assert torch.equal(p, dict(b.named_parameters())[name])
    block = BoundedStateTransport(64, clock="elapsed")
    assert sum(p.numel() for p in block.parameters()) == 1552
    torch.testing.assert_close(torch.sigmoid(block.gate_projection.bias), torch.full((8,), 1 / 16))
    torch.testing.assert_close(torch.nn.functional.softplus(block.rho),
                              torch.log(torch.tensor(2.)) / 2 ** torch.arange(8))


@pytest.mark.parametrize("length", [0, 1, 2, 5, 17, 256])
def test_affine_scan_matches_sequential_values_and_gradients(length):
    torch.manual_seed(25)
    a = torch.rand(2, length, 8, dtype=torch.float64, requires_grad=True)
    b = torch.randn(2, length, 8, dtype=torch.float64, requires_grad=True)
    ar, br = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    actual, expected = affine_state_scan(a, b), affine_state_scan(ar, br, scan="sequential")
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for x, y in ((a, ar), (b, br)):
        if x.grad is None:
            assert y.grad is None or torch.count_nonzero(y.grad) == 0
        else:
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-12, atol=1e-12)


def test_independent_scalar_oracle_state_bounds_and_masks():
    torch.manual_seed(101)
    block = BoundedStateTransport(12, clock="elapsed", time_scale=2.).double()
    hidden = torch.randn(2, 7, 12, dtype=torch.float64, requires_grad=True)
    dt = torch.tensor([[0., 1., 2., 3000., 3., 0., 8.], [0., 1., 2., 3., 4., 5., 6.]], dtype=torch.float64)
    valid = torch.tensor([[True, True, False, True, True, True, True], [False] * 7])
    write = valid.clone()
    write[0, 4] = False
    observed = valid & write
    actual = block.read(hidden, dt, valid, memory_write_mask=write)
    rows = []
    for batch in range(2):
        state, positions = torch.zeros(8, dtype=torch.float64), []
        for i in range(7):
            if observed[batch, i]:
                z = hidden[batch, i]
                u = (z - z.mean()) / torch.sqrt(z.var(unbiased=False) + 1e-5)
                v = torch.tanh(block.value_projection.weight @ u)
                g = torch.sigmoid(block.gate_projection.weight @ u + block.gate_projection.bias)
                decay = torch.exp(-torch.nn.functional.softplus(block.rho) * dt[batch, i] / 2.)
                state = (1 - g) * decay * state + g * v
            elif valid[batch, i]:
                state = torch.zeros_like(state)
            positions.append(state)
        rows.append(torch.stack(positions))
    torch.testing.assert_close(actual, torch.stack(rows), rtol=1e-12, atol=1e-12)
    assert (actual.abs() <= 1).all()
    assert torch.equal(actual[0, 1], actual[0, 2])
    assert torch.count_nonzero(actual[0, 4]) == 0
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.count_nonzero(actual[0, 0]) > 0  # zero interval still writes


def test_clock_identity_scale_invariance_and_elapsed_effect():
    torch.manual_seed(9)
    event = BoundedStateTransport(12, clock="event", time_scale=3.).double()
    torch.manual_seed(9)
    elapsed = BoundedStateTransport(12, clock="elapsed", time_scale=3.).double()
    torch.manual_seed(9)
    scaled = BoundedStateTransport(12, clock="elapsed", time_scale=30.).double()
    h = torch.randn(2, 6, 12, dtype=torch.float64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    dt = torch.full((2, 6), 3., dtype=torch.float64)
    assert torch.equal(event.read(h, dt, mask), elapsed.read(h, dt, mask))
    assert torch.equal(elapsed.read(h, dt, mask), scaled.read(h, dt * 10, mask))
    assert not torch.equal(elapsed.read(h, dt, mask), elapsed.read(h, dt * 100, mask))


@pytest.mark.parametrize("clock", ["event", "elapsed"])
def test_state_read_scan_parameter_and_input_gradients_with_resets(clock):
    torch.manual_seed(124)
    parallel = BoundedStateTransport(12, clock=clock, scan="parallel").double()
    torch.manual_seed(124)
    sequential = BoundedStateTransport(12, clock=clock, scan="sequential").double()
    hidden = torch.randn(2, 17, 12, dtype=torch.float64, requires_grad=True)
    durations = torch.rand(2, 17, dtype=torch.float64, requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_()
    other_durations = durations.detach().clone().requires_grad_()
    valid = torch.ones(2, 17, dtype=torch.bool)
    valid[:, 3] = False
    write = valid.clone()
    write[0, 7] = False
    actual = parallel.read(hidden, durations, valid, memory_write_mask=write)
    expected = sequential.read(other_hidden, other_durations, valid, memory_write_mask=write)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for a, b in [(hidden, other_hidden), (durations, other_durations),
                 *zip(parallel.parameters(), sequential.parameters())]:
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("clock", ["event", "elapsed"])
@pytest.mark.parametrize("component", ["time_loss", "quantity_train_loss"])
def test_activated_gradient_paths(clock, component):
    torch.manual_seed(64)
    model = CountAwareTitanStateTransport(**kwargs(), state_transport_clock=clock).eval()
    activate(model)
    dt, qty, mask = inputs()
    target_outputs(model, dt, mask, qty, lambda_log_qty=1.)[component].mean().backward()
    for name, p in model.state_transport.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    for name in ("encoder.input_proj.weight", "encoder.layers.0.attn.qkv.weight", "encoder.layers.1.attn.qkv.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("clock", ["event", "elapsed"])
def test_activated_target_padding_future_independence_and_no_persistent_state(clock):
    torch.manual_seed(72)
    model = CountAwareTitanStateTransport(**kwargs(), state_transport_clock=clock).eval()
    activate(model)
    dt, qty, valid = inputs()
    expected = model.encode(dt, qty, valid)
    future_dt, future_qty = dt.clone(), qty.clone()
    future_dt[:, 4:], future_qty[:, 4:] = 999., 50000.
    assert torch.equal(expected[:, :4], model.encode(future_dt, future_qty, valid)[:, :4])
    poisoned_dt, poisoned_qty = dt.clone(), qty.clone()
    poisoned_dt[~valid], poisoned_qty[~valid] = float("nan"), float("inf")
    assert torch.equal(expected, model.encode(poisoned_dt, poisoned_qty, valid))
    write = valid.clone()
    write[:, 2] = False
    masked = model.encode_task_states(dt, qty, valid, memory_write_mask=write)[0]
    poisoned_dt[:, 2], poisoned_qty[:, 2] = float("nan"), float("inf")
    assert torch.equal(masked, model.encode_task_states(poisoned_dt, poisoned_qty, valid, memory_write_mask=write)[0])
    for batch in range(2):
        torch.testing.assert_close(expected[batch:batch+1], model.encode(dt[batch:batch+1], qty[batch:batch+1], valid[batch:batch+1]),
                                   rtol=2e-6, atol=2e-6)
    assert torch.equal(expected, model.encode(dt, qty, valid))
    dt, qty = dt.requires_grad_(), qty.requires_grad_()
    state = model.encode_task_states(dt, qty, valid, memory_write_mask=write)[0]
    state.square().sum().backward()
    assert (dt.grad[~write] == 0).all() and (qty.grad[~write] == 0).all()
    # A final withheld next-event target never enters any predicted state.
    write = valid.clone()
    for batch in range(2):
        write[batch, valid[batch].sum() - 1] = False
    clean = model.encode_task_states(dt.detach(), qty.detach(), valid, memory_write_mask=write)[0]
    pdt, pqty = dt.detach().clone(), qty.detach().clone()
    pdt[valid & ~write], pqty[valid & ~write] = float("inf"), float("nan")
    assert torch.equal(clean, model.encode_task_states(pdt, pqty, valid, memory_write_mask=write)[0])


def test_padding_insertion_and_withheld_reset_are_state_path_properties():
    torch.manual_seed(18)
    block = BoundedStateTransport(12, clock="elapsed").double()
    h, dt = torch.randn(1, 4, 12, dtype=torch.float64), torch.tensor([[0., 2., 6., 1.]], dtype=torch.float64)
    valid = torch.ones(1, 4, dtype=torch.bool)
    expected = block.read(h, dt, valid)
    padded_h = torch.cat((h[:, :2], torch.full((1, 1, 12), float("nan")), h[:, 2:]), dim=1)
    padded_dt = torch.cat((dt[:, :2], torch.tensor([[float("nan")]]), dt[:, 2:]), dim=1)
    padded_valid = torch.tensor([[True, True, False, True, True]])
    actual = block.read(padded_h, padded_dt, padded_valid)
    torch.testing.assert_close(expected, actual[:, [0, 1, 3, 4]], rtol=1e-14, atol=1e-14)
    # A valid withheld event resets rather than bridging state history.
    reset_valid = torch.ones_like(padded_valid)
    reset = block.read(padded_h, padded_dt, reset_valid, memory_write_mask=padded_valid)
    fresh = block.read(h[:, 2:], dt[:, 2:], valid[:, 2:])
    torch.testing.assert_close(reset[:, 3:], fresh, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("mismatch", ["clock", "scale", "scan", "identity_missing", "code"])
def test_wrong_identity_rejected_before_any_parameter_mutation(mismatch):
    torch.manual_seed(41)
    model = CountAwareTitanStateTransport(**kwargs())
    source_kwargs = kwargs()
    extra = {}
    if mismatch == "clock":
        extra["state_transport_clock"] = "event"
    if mismatch == "scale":
        source_kwargs["time_scale"] = 1.
    if mismatch == "scan":
        extra["state_transport_scan"] = "sequential"
    source = CountAwareTitanStateTransport(**source_kwargs, **extra).state_dict()
    if mismatch == "identity_missing":
        del source["state_transport.contract_identity"]
    if mismatch == "code":
        source["state_transport.clock_code"] = torch.tensor(0)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="identity mismatch"):
        model.load_state_dict(source, strict=False)
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])


@pytest.mark.parametrize("clock", ["event", "elapsed"])
def test_checkpoint_optimizer_rng_exact_next_update_restore(clock):
    torch.manual_seed(59)
    model = CountAwareTitanStateTransport(**kwargs(), state_transport_clock=clock).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dt, qty, mask = inputs()
    def step(m, opt):
        opt.zero_grad(set_to_none=True)
        loss = target_outputs(m, dt, mask, qty, lambda_log_qty=1.)["joint_loss"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        opt.step()
        return loss.detach().clone()
    step(model, optimizer)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}
    stream = io.BytesIO()
    torch.save(payload, stream)
    expected = step(model, optimizer)
    expected_rng = torch.get_rng_state().clone()
    restored = CountAwareTitanStateTransport(**kwargs(), state_transport_clock=clock).train()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng"])
    actual = step(restored, restored_optimizer)
    assert torch.equal(actual, expected) and torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    for k, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            assert torch.equal(value, restored_optimizer.state_dict()["state"][k][field])


@pytest.mark.parametrize("field", ["duration", "quantity"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_invalid_observed_values_rejected(field, value):
    model = CountAwareTitanStateTransport(**kwargs())
    dt, qty, mask = inputs()
    (dt if field == "duration" else qty)[0, 1] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        model.encode(dt, qty, mask)


def checkpoint_payload(clock="elapsed", *, summary=False):
    model = CountAwareTitanStateTransport(**kwargs(), state_transport_clock=clock)
    head = model.time_head_contract()
    metadata = state_transport_metadata(16, clock, time_scale=3.) | {"time_head": head}
    interface_head = dict(head, statistics_source_split="train", train_time_statistics={
        "time_scale": 3., "target_log_scaled_mean": .1, "target_log_scaled_std": .7})
    interface = {"time_head": interface_head}
    backbone = ELAPSED_STATE_BACKBONE if clock == "elapsed" else EVENT_STATE_BACKBONE
    identity = dict(backbone=backbone, variant="count_only_log_regression", interface_meta=interface,
                    checkpoint_monitor="validation_raw_quantity_rmse", arguments={
                        "dataset_contract": "intermittent_frozen_5000", "model_role": STATE_TRANSPORT_ROLE,
                        "time_head_mode": head["mode"], "time_scale": 3.})
    payload = dict(backbone=backbone, variant="count_only_log_regression", encoder_config=metadata,
                   evaluation_scope="validation_only", held_out_test_evaluated=False,
                   interface_meta=interface, resume_identity=identity)
    if summary:
        payload.update(checkpoint_state_sha256="a" * 64, best_epoch=1)
    else:
        payload["model_state_dict"] = model.state_dict()
    return payload


@pytest.mark.parametrize("summary", [False, True])
def test_checkpoint_metadata_and_summary_validation(summary):
    payload = checkpoint_payload(summary=summary)
    assert validate_state_transport_checkpoint(payload, ELAPSED_STATE_BACKBONE)
    for wrong in ("titantpp", EVENT_STATE_BACKBONE):
        with pytest.raises(ValueError):
            validate_state_transport_checkpoint(payload, wrong)
    bad = copy.deepcopy(payload)
    bad["encoder_config"]["state_transport_time_scale"] = 1.
    with pytest.raises(ValueError):
        validate_state_transport_checkpoint(bad, ELAPSED_STATE_BACKBONE)


@pytest.mark.parametrize("clock", ["event", "elapsed"])
@pytest.mark.parametrize("summary", [False, True])
def test_checkpoint_accepts_observed_intermittent_statistics_roundoff(clock, summary):
    # Actual frozen encoder/interface value and recomputed train statistic from
    # the Intermittent run differ by 1.11e-16; runner already accepts abs_tol=1e-12.
    payload = checkpoint_payload(clock, summary=summary)
    payload["encoder_config"]["time_head"]["time_initial_location"] = -0.21389036220110552
    interface = payload["interface_meta"]["time_head"]
    interface["time_initial_location"] = -0.21389036220110552
    interface["train_time_statistics"]["target_log_scaled_mean"] = -0.2138903622011054
    assert validate_state_transport_checkpoint(payload, payload["backbone"])


_STATISTIC_FIELDS = (("time_scale", "time_scale"),
                     ("time_initial_location", "target_log_scaled_mean"),
                     ("time_initial_scale", "target_log_scaled_std"))


@pytest.mark.parametrize("field,statistic", _STATISTIC_FIELDS)
@pytest.mark.parametrize("offset,accepted", [(1e-13, True), (1e-10, False)])
def test_checkpoint_train_statistics_absolute_tolerance(field, statistic, offset, accepted):
    payload = checkpoint_payload(summary=True)
    value = payload["encoder_config"]["time_head"][field]
    payload["interface_meta"]["time_head"]["train_time_statistics"][statistic] = value + offset
    if accepted:
        assert validate_state_transport_checkpoint(payload, payload["backbone"])
    else:
        with pytest.raises(ValueError, match="train-only initialization mismatch"):
            validate_state_transport_checkpoint(payload, payload["backbone"])


@pytest.mark.parametrize("field,statistic", _STATISTIC_FIELDS)
@pytest.mark.parametrize("invalid", [None, "0.1", True, False, float("nan"), float("inf"), -float("inf")])
def test_checkpoint_rejects_invalid_train_statistics(field, statistic, invalid):
    payload = checkpoint_payload(summary=True)
    payload["interface_meta"]["time_head"]["train_time_statistics"][statistic] = invalid
    with pytest.raises(ValueError, match="train-only initialization mismatch"):
        validate_state_transport_checkpoint(payload, payload["backbone"])


@pytest.mark.parametrize("field,statistic", _STATISTIC_FIELDS)
def test_checkpoint_rejects_missing_train_statistic(field, statistic):
    payload = checkpoint_payload(summary=True)
    del payload["interface_meta"]["time_head"]["train_time_statistics"][statistic]
    with pytest.raises(ValueError, match="train-only initialization mismatch"):
        validate_state_transport_checkpoint(payload, payload["backbone"])


@pytest.mark.parametrize("field,statistic", _STATISTIC_FIELDS)
def test_checkpoint_interface_initialization_stays_exact(field, statistic):
    payload = checkpoint_payload(summary=True)
    payload["interface_meta"]["time_head"][field] += 1e-13
    with pytest.raises(ValueError, match="train-only initialization mismatch"):
        validate_state_transport_checkpoint(payload, payload["backbone"])
