"""Contracts for observed transition memory; all inputs are synthetic."""
from __future__ import annotations

import copy
import io

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanDualTimescale import (
    CountAwareTitanDualTimescaleTPP, DualTimescaleTransitionMemory,
    DUAL_TIMESCALE_BACKBONE, DUAL_TIMESCALE_ROLE, DUAL_TIMESCALE_CONTRACT_ID,
    DUAL_TIMESCALE_EPSILON, DUAL_TIMESCALE_RANK,
    dual_timescale_metadata,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    # Match the actual full-fit runner's run_taxi_quantity_interface_ablation
    # set_seed policy, including deterministic gather/scatter backward on CUDA.
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
    torch.set_num_threads(previous)


def kwargs(dim=16):
    return dict(hidden_dim=dim, train_log_mean=1.5, max_seq_len=32,
                quantity_variant="count_only_log_regression", lambda_tail=0.,
                time_head_mode="legacy_clamped_rmtpp", time_intercept_limit=300.)


def batch(device="cpu"):
    torch.manual_seed(144)
    dts = torch.rand(3, 24, device=device) * 3
    quantities = torch.rand(3, 24, device=device) * 20
    mask = torch.ones(3, 24, device=device, dtype=torch.bool)
    mask[1, 20:] = False
    mask[2, 3] = False
    return dts, quantities, mask


def outputs(model, sample):
    dt, q, mask = sample
    return target_outputs(model, dt, mask, q, lambda_log_qty=1.)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim", [16, 64])
def test_same_seed_zero_alpha_rng_shared_state_output_and_gradient(device, dim):
    torch.manual_seed(42)
    baseline = CountAwareTitanTPP(**kwargs(dim))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    candidate = CountAwareTitanDualTimescaleTPP(**kwargs(dim))
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert candidate.additional_parameter_count == dual_timescale_metadata(dim)["additional_parameter_count"]
    for key, value in baseline.state_dict().items():
        assert torch.equal(candidate.state_dict()[key], value), key
    baseline.to(device).train()
    candidate.to(device).train()
    values = []
    sample = batch(device)
    for model in (baseline, candidate):
        torch.manual_seed(71)
        values.append(outputs(model, sample))
    for key in values[0]:
        assert torch.equal(values[0][key], values[1][key]), key
    values[0]["joint_loss"].sum().backward()
    values[1]["joint_loss"].sum().backward()
    actual = dict(candidate.named_parameters())
    for key, value in baseline.named_parameters():
        if value.grad is None:
            assert actual[key].grad is None, key
        else:
            assert torch.equal(value.grad, actual[key].grad), key
    assert candidate.transition_memory.alpha_raw.grad.abs() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_active_quantity_gradient_to_every_memory_rule(device):
    torch.manual_seed(42)
    model = CountAwareTitanDualTimescaleTPP(**kwargs()).to(device).eval()
    # B starts with a zero quantity-head weight. Let the actual quantity-only
    # optimizer open that head and then alpha before checking every memory rule.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = batch(device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs(model, sample)["log_qty_loss"].sum().backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    result = outputs(model, sample)
    result["log_qty_loss"].sum().backward()
    assert model.transition_memory.alpha_raw.abs() > 0
    for name in ("query.weight", "key.weight", "value.weight", "write.weight", "mix.weight"):
        gradient = dict(model.transition_memory.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def slow_reference(parts):
    """Independently enumerate disjoint admitted writes and direct similarities."""
    q, k, v, trust = (parts[x] for x in ("queries", "keys", "values", "trust"))
    result = {key: torch.zeros_like(parts[key]) for key in (
        "local_read", "global_read", "local_mass", "global_mass")}
    for b in range(q.size(0)):
        for t in range(q.size(1)):
            indices = torch.where(parts["admitted"][b, :t + 1])[0]
            for scope, chosen in (("local", indices[-8:]), ("global", indices[:-8])):
                if chosen.numel():
                    weights = (q[b, t] * k[b, chosen]).sum(-1) * trust[b, chosen, 0]
                    mass = weights.sum()
                    result[scope + "_mass"][b, t] = mass
                    if mass > DUAL_TIMESCALE_EPSILON:
                        normalized = (weights[:, None] * v[b, chosen]).sum(0) / mass
                        result[scope + "_read"][b, t] = normalized * (-torch.expm1(-DUAL_TIMESCALE_RANK * mass))
    return result


@pytest.mark.parametrize("device", DEVICES)
def test_vectorized_prefix_matches_explicit_disjoint_transition_reference(device):
    torch.manual_seed(514)
    memory = DualTimescaleTransitionMemory(16).to(device).double()
    hidden = torch.randn(3, 29, 16, device=device, dtype=torch.float64)
    mask = torch.ones(3, 29, device=device, dtype=torch.bool)
    mask[1, [4, 7, 8, 19]] = False
    mask[2, 13:] = False
    observed = mask.clone()
    observed[0, [5, 15]] = False
    parts = memory.residual_components(hidden, mask, observed)
    reference = slow_reference(parts)
    for key, expected in reference.items():
        torch.testing.assert_close(parts[key], expected, atol=1e-12, rtol=1e-11)
    assert torch.equal(parts["global_count"], (parts["write_count"] - 8).clamp_min(0))
    assert not parts["admitted"][:, 0].any()
    assert not parts["admitted"][0, [5, 6, 15, 16]].any()
    assert (parts["local_weight"][parts["write_count"] > 0] > 0).all()
    assert not parts["global_weight"][parts["global_count"] == 0].any()


@pytest.mark.parametrize("device", DEVICES)
def test_memory_causality_padding_write_mask_and_no_cross_batch_state(device):
    torch.manual_seed(42)
    memory = DualTimescaleTransitionMemory(16).to(device).eval()
    hidden = torch.randn(2, 24, 16, device=device)
    mask = torch.ones(2, 24, dtype=torch.bool, device=device)
    observed = mask.clone()
    observed[:, 16:] = False
    original = memory.residual_components(hidden, mask, observed)
    changed = hidden.clone()
    changed[:, 16:] += 1000
    future = memory.residual_components(changed, mask, observed)
    torch.testing.assert_close(original["residual"][:, :16], future["residual"][:, :16], atol=0, rtol=0)
    assert torch.equal(original["write_count"], future["write_count"])
    # A blocked event can query memory but cannot enter a value or either endpoint.
    for key in ("values", "trust", "keys"):
        left = original[key] * original["admitted"].unsqueeze(-1)
        right = future[key] * future["admitted"].unsqueeze(-1)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    mask[:, 4] = False
    padded = hidden.clone()
    padded[:, 4] = float("nan")
    a = memory.residual_components(hidden, mask)["residual"]
    b = memory.residual_components(padded, mask)["residual"]
    assert torch.equal(a, b) and torch.isfinite(b).all()
    memory(torch.randn_like(hidden), mask)
    isolated = memory.residual_components(hidden[:1], mask[:1])["residual"]
    torch.testing.assert_close(a[:1], isolated, atol=1e-6, rtol=1e-6)
    empty = memory.residual_components(hidden, mask, torch.zeros_like(mask))["residual"]
    assert torch.equal(empty, torch.zeros_like(empty))
    short = memory.residual_components(hidden[:, :1], mask[:, :1])["residual"]
    assert torch.equal(short, torch.zeros_like(short))


@pytest.mark.parametrize("device", DEVICES)
def test_active_model_prefix_and_future_target_identity(device):
    model = CountAwareTitanDualTimescaleTPP(**kwargs()).to(device).eval()
    model.transition_memory.alpha_raw.data.fill_(.3)
    model.quantity_head.weight.data.fill_(.02)
    dt, q, mask = batch(device)
    a = model.encode_task_states(dt, q, mask)[1]
    q2, dt2 = q.clone(), dt.clone()
    q2[:, 15:] += 100
    dt2[:, 15:] += 100
    b = model.encode_task_states(dt2, q2, mask)[1]
    torch.testing.assert_close(a[:, :15], b[:, :15], atol=0, rtol=0)
    base = outputs(model, (dt, q, mask))
    q2, dt2 = q.clone(), dt.clone()
    positions = mask.long().sum(1) - 1
    # core target_outputs uses actual last valid target positions, including holes.
    positions = torch.where(mask, torch.arange(mask.size(1), device=device), -1).amax(1)
    for row, pos in enumerate(positions):
        q2[row, pos] += 100
        dt2[row, pos] += 50
    modified = outputs(model, (dt2, q2, mask))
    assert torch.equal(base["pred_qty"], modified["pred_qty"])


@pytest.mark.parametrize("device", DEVICES)
def test_extreme_memory_finite_bounded_loss_and_gradients(device):
    torch.manual_seed(9)
    memory = DualTimescaleTransitionMemory(16).to(device)
    memory.alpha_raw.data.fill_(.5)
    hidden = (torch.randn(2, 32, 16, device=device) * 1e30).requires_grad_()
    mask = torch.ones(2, 32, dtype=torch.bool, device=device)
    parts = memory.residual_components(hidden, mask)
    assert torch.isfinite(parts["residual"]).all()
    assert parts["residual"].abs().max() <= 1
    parts["residual"].square().sum().backward()
    assert torch.isfinite(hidden.grad).all()
    for parameter in memory.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_checkpoint_optimizer_restores_exact_next_step(device):
    torch.manual_seed(20)
    model = CountAwareTitanDualTimescaleTPP(**kwargs()).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = batch(device)
    def step(target, opt):
        torch.manual_seed(651)
        opt.zero_grad(set_to_none=True)
        outputs(target, sample)["joint_loss"].mean().backward()
        opt.step()
    step(model, optimizer)
    stream = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True, map_location=device)
    restored = CountAwareTitanDualTimescaleTPP(**kwargs()).to(device).train()
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(saved["optimizer"])
    step(model, optimizer)
    step(restored, restored_optimizer)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key


def payload():
    model, metadata = build_count_aware_model(DUAL_TIMESCALE_BACKBONE, **kwargs())
    return {"backbone": DUAL_TIMESCALE_BACKBONE, "encoder_config": metadata,
            "model_state_dict": model.state_dict(), "variant": "count_only_log_regression",
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False}


def test_checkpoint_route_rejects_relabelling_foreign_and_malformed_state():
    p = payload()
    validate_checkpoint_route(p, DUAL_TIMESCALE_BACKBONE)
    with pytest.raises(ValueError, match="relabelled"):
        validate_checkpoint_route(p, "titantpp")
    for mutate in (
        lambda p: p["model_state_dict"].pop("transition_memory.query.weight"),
        lambda p: p["model_state_dict"].update({"foreign": torch.zeros(1)}),
        lambda p: p["encoder_config"].update({"local_transition_count": 4}),
        lambda p: p.update({"held_out_test_evaluated": True}),
        lambda p: p["model_state_dict"].update({"transition_memory.alpha_raw": torch.ones(1)}),
    ):
        bad = copy.deepcopy(p)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_checkpoint_route(bad, DUAL_TIMESCALE_BACKBONE)
    stripped = copy.deepcopy(p)
    stripped["backbone"] = "titantpp"
    stripped["encoder_config"] = {}
    with pytest.raises(ValueError, match="relabelled"):
        validate_checkpoint_route(stripped, "titantpp")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("support", ["local_only", "global_only", "none", "below_epsilon"])
def test_write_underflow_uses_actual_mass_and_preserves_sole_supported_path(device, support):
    torch.manual_seed(43)
    memory = DualTimescaleTransitionMemory(16).to(device).eval()
    # 17 admitted transitions: last eight local, first nine global.
    small = torch.arange(18, device=device, dtype=torch.float32) * .25
    large = torch.where(torch.arange(18, device=device) % 2 == 0, 10000., -10000.)
    if support == "local_only":
        levels = large.clone()
        levels[10:] = levels[9] + torch.arange(1, 9, device=device) * .25
        # Force fusion toward the missing global path as well.
        memory.mix.bias.data.fill_(-1000.)
    elif support == "global_only":
        levels = large.clone()
        levels[:10] = small[:10]
        memory.mix.bias.data.fill_(1000.)
    elif support == "none":
        levels = large
    else:
        levels = small
        memory.write.weight.data.zero_()
        memory.write.bias.data.fill_(-30.)
    hidden = levels[None, :, None].expand(1, -1, 16).clone()
    mask = torch.ones(1, 18, device=device, dtype=torch.bool)
    parts = memory.residual_components(hidden, mask)
    assert parts["write_count"][0, -1] == 17
    assert parts["global_count"][0, -1] == 9
    if support in ("local_only", "global_only"):
        available = support.split("_")[0]
        absent = "global" if available == "local" else "local"
        assert parts[available + "_mass"][0, -1] > DUAL_TIMESCALE_EPSILON
        assert parts[absent + "_mass"][0, -1] <= DUAL_TIMESCALE_EPSILON
        assert parts[available + "_weight"][0, -1] == 1
        assert parts[absent + "_weight"][0, -1] == 0
        assert parts[available + "_read"][0, -1].abs().sum() > 0
        assert torch.equal(parts["residual"][0, -1], parts[available + "_read"][0, -1])
        assert torch.count_nonzero(parts[absent + "_read"][0, -1]) == 0
    else:
        for scope in ("local", "global"):
            mass = parts[scope + "_mass"][0, -1]
            assert mass <= DUAL_TIMESCALE_EPSILON
            if support == "below_epsilon":
                assert mass > 0
            assert parts[scope + "_weight"][0, -1] == 0
            assert torch.count_nonzero(parts[scope + "_read"][0, -1]) == 0
        assert torch.count_nonzero(parts["residual"][0, -1]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_single_transition_absolute_confidence_controls_residual_and_has_gradient(device):
    torch.manual_seed(73)
    memory = DualTimescaleTransitionMemory(16).to(device).eval()
    hidden = torch.randn(1, 2, 16, device=device)
    mask = torch.ones(1, 2, device=device, dtype=torch.bool)
    memory.write.weight.data.zero_()
    residual_norms = []
    for bias in (-10., -3., 0., 3., 10.):
        memory.write.bias.data.fill_(bias)
        parts = memory.residual_components(hidden, mask)
        mass = parts["local_mass"][0, 1]
        support = -torch.expm1(-DUAL_TIMESCALE_RANK * mass)
        assert parts["local_weight"][0, 1] == 1
        assert parts["global_weight"][0, 1] == 0
        torch.testing.assert_close(parts["residual"][0, 1], parts["values"][0, 1] * support,
                                   atol=1e-7, rtol=1e-6)
        assert 0 < support < 1
        residual_norms.append(parts["residual"][0, 1].norm().item())
    assert all(earlier < later for earlier, later in zip(residual_norms, residual_norms[1:]))
    memory.write.bias.data.zero_()
    parts = memory.residual_components(hidden, mask)
    parts["residual"].square().sum().backward()
    # Absolute trust can now train even with exactly one observed transition.
    assert memory.write.bias.grad is not None and memory.write.bias.grad.abs().sum() > 0
    assert memory.write.weight.grad is not None and memory.write.weight.grad.abs().sum() > 0
    assert torch.isfinite(memory.write.bias.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_support_grows_with_repeated_evidence_and_is_bounded(device):
    torch.manual_seed(104)
    memory = DualTimescaleTransitionMemory(16).to(device).eval()
    hidden = torch.randn(1, 1, 16, device=device).expand(1, 24, 16).clone()
    mask = torch.ones(1, 24, device=device, dtype=torch.bool)
    parts = memory.residual_components(hidden, mask)
    for scope in ("local", "global"):
        mass, support = parts[scope + "_mass"], parts[scope + "_support"]
        expected = torch.where(mass > DUAL_TIMESCALE_EPSILON,
                               -torch.expm1(-DUAL_TIMESCALE_RANK * mass), torch.zeros_like(mass))
        torch.testing.assert_close(support, expected, atol=0, rtol=0)
        assert ((support >= 0) & (support <= 1)).all()
        assert (support[:, 1:] >= support[:, :-1] - 1e-6).all()
        assert (parts[scope + "_read"].abs() <= support + 1e-6).all()
    assert parts["local_support"][0, 8] > parts["local_support"][0, 1] > 0
    assert parts["global_support"][0, -1] > parts["global_support"][0, 9] > 0
