"""CPU-only formula and same-graph checks, including the real static Titan."""
import random

import numpy as np
import pytest
import torch

from paper.scripts.mixed_quantity_objective import MixedQuantityObjective, mixed_joint_causal_batch_objective
from paper.scripts.quantity_objective_comparison import QuantityStatistics
from paper.scripts.raw_aux_gradient_control import RawAuxGradientControl, apply_raw_aux_control, finite_gradient_norm, summarize_epoch
from paper.scripts.run_quantity_comparison import synthetic_inputs
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def objective(alpha=1.769152228020146):
    return MixedQuantityObjective("mixed_original", alpha, 1., "c" * 64)


def statistics():
    return QuantityStatistics(1., 5.)


def add_gradients(left, right, scalar):
    return [None if b is None and r is None else (0 if b is None else b) + scalar * (0 if r is None else r)
            for b, r in zip(left, right, strict=True)]


def assert_gradients(first, second, *, exact=False):
    for a, b in zip(first, second, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b.to(a.dtype), rtol=0 if exact else 1e-5, atol=0 if exact else 1e-7)


def test_actual_titan_same_forward_gradient_queries_preserve_rng_state_and_none(monkeypatch):
    import paper.scripts.mixed_quantity_objective as module
    model, train, _ = synthetic_inputs()
    model.train()
    # Exercise quantity-to-encoder gradients, rather than the initial zero head.
    with torch.no_grad():
        model.quantity_head.weight.fill_(.02)
        model.v_t.weight.fill_(.015)
    calls = []
    original_forward = module.b_task_outputs
    def forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)
    monkeypatch.setattr(module, "b_task_outputs", forward)
    outputs = module.mixed_joint_causal_batch_objective(model, *train.dataset.tensors,
        statistics=statistics(), objective=objective())
    state = canonical_state_dict_sha256(model.state_dict())
    parameters = tuple(model.parameters())
    parameters[0].grad = torch.full_like(parameters[0], .123)
    saved_grads = [(p.grad, None if p.grad is None else p.grad.clone()) for p in parameters]
    contexts = [(m, getattr(m, "_ctx_mem", None), getattr(m, "contextual_mem_size", None)) for m in model.modules()]
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    queries = []
    original_grad = torch.autograd.grad
    def grad(*args, **kwargs):
        queries.append(kwargs)
        return original_grad(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", grad)
    adjusted, diagnostic = apply_raw_aux_control(outputs, model.named_parameters(), objective=objective(),
        control=RawAuxGradientControl(), training=model.training)
    assert len(queries) == 2 and all(q["retain_graph"] and not q["create_graph"] and q["allow_unused"] for q in queries)
    monkeypatch.setattr(torch.autograd, "grad", original_grad)
    assert calls == [1] and canonical_state_dict_sha256(model.state_dict()) == state
    assert random.getstate() == python_rng and np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for p, (identity, value) in zip(parameters, saved_grads, strict=True):
        assert p.grad is identity and (value is None or torch.equal(p.grad, value))
    for m, context, size in contexts:
        assert getattr(m, "_ctx_mem", None) is context and getattr(m, "contextual_mem_size", None) == size
    assert "adaptive_objective_loss" not in outputs
    base = original_grad((outputs["time_loss"] + outputs["log_qty_loss"]).mean(), parameters,
                         retain_graph=True, allow_unused=True)
    raw = original_grad((objective().alpha * outputs["raw_qty_loss"]).mean(), parameters,
                        retain_graph=True, allow_unused=True)
    expected = add_gradients(base, raw, diagnostic["s"])
    model.zero_grad(set_to_none=True)
    adjusted["objective_loss"].mean().backward()
    assert_gradients([p.grad for p in parameters], expected)
    assert canonical_state_dict_sha256(model.state_dict()) == state
    assert torch.equal(torch.get_rng_state(), torch_rng) and calls == [1]
    assert diagnostic["effective_raw_norm"] <= diagnostic["base_norm"] * (1 + 1e-5) + 1e-8
    before = [None if p.grad is None else p.grad.clone() for p in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    k = min(1., 1. / (float(norm) + 1e-6))
    assert_gradients([p.grad for p in parameters], [None if g is None else k * g for g in before])


@pytest.mark.parametrize("base,raw,expected", [([0., 0.], [0., 0.], 1.), ([0., 0.], [3., 4.], 0.),
    ([3., 4.], [0., 0.], 1.), ([3., 4.], [.1, .1], 1.), ([3., 4.], [30., 40.], 5./(50.+1e-12)),
    ([3., 4.], [-30., -40.], 5./(50.+1e-12))])
def test_zero_inactive_and_opposing_linear_gradients(base, raw, expected):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    b, r = torch.tensor(base, dtype=torch.float64), torch.tensor(raw, dtype=torch.float64)
    outputs = {"time_loss": (parameter * b).sum().reshape(1), "log_qty_loss": (0 * parameter).sum().reshape(1),
               "raw_qty_loss": (parameter * r).sum().reshape(1)}
    adjusted, record = apply_raw_aux_control(outputs, [("p", parameter), ("unused", unused)],
        objective=objective(1.), control=RawAuxGradientControl(), training=True)
    assert record["s"] == expected
    adjusted["objective_loss"].mean().backward()
    torch.testing.assert_close(parameter.grad, b + expected * r, rtol=1e-12, atol=1e-12)
    assert unused.grad is None


def test_nonlinear_multiplier_is_detached_and_base_time_log_coefficients_are_one():
    p = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    outputs = {"time_loss": (.2*p.square()).reshape(1), "log_qty_loss": (.3*p.square()).reshape(1),
               "raw_qty_loss": (.5*p.pow(4)).reshape(1)}
    adjusted, record = apply_raw_aux_control(outputs, [("p", p)], objective=objective(1.),
        control=RawAuxGradientControl(), training=True)
    adjusted["objective_loss"].mean().backward()
    assert float(p.grad) == pytest.approx(2. + record["s"] * 16., abs=1e-12)
    assert float(p.grad) == pytest.approx(4., abs=1e-11)


@pytest.mark.parametrize("kind", ["loss", "gradient", "norm", "validation", "no_grad", "duplicate", "matched"])
def test_invalid_inputs_rejected(kind):
    p = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    outputs = {"time_loss": p.square().reshape(1), "log_qty_loss": p.square().reshape(1), "raw_qty_loss": (p + 1).reshape(1)}
    params, obj = [("p", p)], objective()
    if kind == "loss":
        outputs["raw_qty_loss"] = (p * float("nan")).reshape(1)
    elif kind == "gradient":
        outputs["raw_qty_loss"] = p.sqrt().reshape(1)
    elif kind == "norm":
        outputs["raw_qty_loss"] = (p * 1e307).reshape(1)
    elif kind == "duplicate":
        params.append(("alias", p))
    elif kind == "matched":
        obj = MixedQuantityObjective("mixed_matched_original", .2, .8, "c" * 64)
    with torch.set_grad_enabled(kind != "no_grad"):
        with pytest.raises(ValueError):
            apply_raw_aux_control(outputs, params, objective=obj, control=RawAuxGradientControl(), training=kind != "validation")
    assert p.grad is None


def test_actual_titan_inactive_cap_matches_uncapped_gradients_within_frozen_tolerance():
    model, train, _ = synthetic_inputs()
    obj = objective(1e-8)
    outputs = mixed_joint_causal_batch_objective(model, *train.dataset.tensors, statistics=statistics(), objective=obj)
    adjusted, record = apply_raw_aux_control(outputs, model.named_parameters(), objective=obj,
        control=RawAuxGradientControl(), training=True)
    assert record["s"] == 1.
    parameters = tuple(model.parameters())
    # Effective formula uses (time+log)+raw rather than legacy time+(log+raw).
    # The corresponding FP32 gradients agree within the frozen reconstruction tolerance.
    baseline = torch.autograd.grad(outputs["objective_loss"].mean(), parameters, retain_graph=True, allow_unused=True)
    actual = torch.autograd.grad(adjusted["objective_loss"].mean(), parameters, allow_unused=True)
    assert_gradients(actual, baseline)


def test_policy_frozen_and_epoch_means_are_sample_weighted():
    with pytest.raises(ValueError):
        RawAuxGradientControl(cap_ratio=.5)
    with pytest.raises(ValueError):
        RawAuxGradientControl(denominator_epsilon=0.)
    rows = []
    for count, s, clipped in ((3, .25, True), (1, 1., False)):
        rows.append(dict(count=count, base_norm=1., weighted_raw_norm=4., effective_raw_norm=4*s,
            s=s, effective_alpha=2*s, composed_joint_norm_preclip=2., common_clip_factor=.5,
            s_less_than_one=s < 1, norm_bound_passed=True, clipped=clipped,
            norm_above_max=clipped, component_gradient_queries=2))
    summary = summarize_epoch(rows)
    assert summary["sample_weighted"]["s"]["mean"] == .4375
    assert summary["capped_batch_rate"] == summary["clipped_batch_rate"] == .5
    assert summary["component_gradient_queries"] == 4
