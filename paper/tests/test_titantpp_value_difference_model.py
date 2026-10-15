"""Synthetic-only tests: never load a scientific dataset or trained CNN."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
import torch

from models.TPPs import CountAwareTitanValueDifferenceCNN as candidate
from models.TPPs.CountAwareTitanCNNGRU import observed_context, causal_depthwise_residual
from paper.scripts import run_titantpp_value_difference_cnn as route
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def forbid_real_data():
    def no(*args, **kwargs):
        raise AssertionError("No real parquet inputs")
    with patch.object(pl, "read_parquet", no), patch.object(pl, "scan_parquet", no):
        yield


def build(arm=candidate.VALUE_DIFFERENCE, seed=42, context_length=None):
    torch.manual_seed(seed)
    data = route.fabricated_data()
    if context_length is not None:
        data["loader"]["max_seq_len"] = context_length
    return route.build_model(data, arm)


def batch():
    dt = torch.tensor([[0., 1., 3., 2., 4., 2., 3., 5.], [0., 0., 0., 2., 5., 1., 4., 2.]])
    quantity = torch.tensor([[0., 0., 5., 2., 7., 4., 3., 6.], [0., 0., 0., 2., 4., 8., 1., 5.]])
    mask = torch.tensor([[False] + [True] * 7, [False] * 3 + [True] * 5])
    return dt, quantity, mask


def activate(model):
    with torch.no_grad():
        attention = model.encoder.layers[0].attn
        attention.causal_q_kernel.normal_(0, .1)
        attention.causal_k_kernel.normal_(0, .1)
        if hasattr(attention, "value_difference_kernel"):
            attention.value_difference_kernel.normal_(0, .1)
        else:
            attention.causal_v_kernel.normal_(0, .1)
        model.quantity_head.weight.normal_(0, .1)
        model.time_scale_weight.weight.normal_(0, .01)
    model.eval()


def payload(arm=candidate.VALUE_DIFFERENCE, context_length=None):
    model, meta = build(arm, context_length=context_length)
    interface = {"time_head": meta["time_head"]}
    return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "variant": route.VARIANT, "interface_meta": interface,
        "resume_identity": {"backbone": arm, "interface_meta": interface,
            "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": route.role_for_arm(arm)}}}


@pytest.mark.parametrize("seed", (42, 52, 62))
@pytest.mark.parametrize("context_length", (32, 84, 256))
def test_common_initial_tensors_rng_physical_counts_and_output(seed, context_length):
    data = route.fabricated_data(); data["loader"]["max_seq_len"] = context_length
    torch.manual_seed(seed)
    baseline, _ = route.original.build_model(data, candidate.LEGACY_CNN)
    rng = torch.get_rng_state().clone()
    model, meta = build(seed=seed, context_length=context_length)
    assert torch.equal(rng, torch.get_rng_state())
    state = model.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in baseline.state_dict().items() if name != candidate.REMOVED_KEY)
    assert candidate.REMOVED_KEY not in state
    assert tuple(state[candidate.DIFFERENCE_KEY].shape) == (2, 64)
    assert bool((state[candidate.DIFFERENCE_KEY] == 0).all())
    assert set(state) == (set(baseline.state_dict()) - {candidate.REMOVED_KEY}) | {candidate.DIFFERENCE_KEY, candidate.IDENTITY_BUFFER}
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters()) - 64
    assert sum(p.numel() for p in model.parameters()) == meta["registered_parameter_count"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == meta["active_trainable_parameter_count"]
    assert not any(p.requires_grad for p in model.multilag_detail.parameters())
    attention = model.encoder.layers[0].attn
    assert attention.qkv.weight.requires_grad
    assert attention.causal_q_kernel.requires_grad and attention.causal_k_kernel.requires_grad
    assert sum(p.numel() for n, p in attention.named_parameters() if n.endswith("kernel")) == 512
    if context_length == 84:
        assert (meta["active_trainable_parameter_count"], meta["registered_parameter_count"]) == (79363, 104095)
    baseline.eval(); model.eval()
    dt, quantity, mask = batch()
    with torch.no_grad():
        a = core.target_outputs(baseline, dt, mask, quantity, lambda_log_qty=1.)
        b = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_hand_computed_boundary_and_interior_difference():
    values = torch.tensor([[[1.], [3.], [8.], [10.]]])
    kernel = torch.tensor([[2.], [4.]])
    _, sources, available = observed_context(torch.ones(1, 4, dtype=torch.bool))
    result = candidate.observed_difference_residual(values, kernel, sources, available)
    assert torch.equal(result, torch.tensor([[[0.], [4.], [18.], [24.]]]))
    effective = torch.stack((kernel[0], kernel[1] - kernel[0], -kernel[1]))
    free = causal_depthwise_residual(values, effective, sources, available)
    assert torch.equal(result[:, 2:], free[:, 2:])
    # Free zero-sum taps with zero padding are not the stipulated boundary rule.
    assert not torch.equal(result[:, :2], free[:, :2])


@pytest.mark.parametrize("length", (0, 1, 2, 7))
def test_constant_latent_translation_invariance_and_short_history(length):
    values = torch.randn(2, 1, 64, dtype=torch.float64).expand(2, length, 64).clone()
    kernel = torch.randn(2, 64, dtype=torch.float64)
    mask = torch.ones(2, length, dtype=torch.bool)
    _, sources, available = observed_context(mask)
    result = candidate.observed_difference_residual(values, kernel, sources, available)
    assert torch.equal(result, torch.zeros_like(result))
    values = torch.randn(2, length, 64, dtype=torch.float64)
    shift = torch.randn(2, 1, 64, dtype=torch.float64)
    a = candidate.observed_difference_residual(values, kernel, sources, available)
    b = candidate.observed_difference_residual(values + shift, kernel, sources, available)
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    if length:
        assert torch.equal(a[:, 0], torch.zeros_like(a[:, 0]))


def test_padding_skip_withheld_reset_and_nan_gradient_isolation():
    values = torch.randn(2, 8, 64)
    valid = torch.tensor([[True, True, False, True, True, False, True, True]] * 2)
    _, sources, available = observed_context(valid)
    kernel = torch.randn(2, 64, requires_grad=True)
    masked = values.clone(); masked[~valid] = float("nan"); masked.requires_grad_()
    actual = candidate.observed_difference_residual(masked, kernel, sources, available)
    keep = [0, 1, 3, 4, 6, 7]
    _, cs, ca = observed_context(valid[:, keep])
    compact = candidate.observed_difference_residual(values[:, keep], kernel, cs, ca)
    torch.testing.assert_close(actual[:, keep], compact, rtol=0, atol=0)
    assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))
    actual.square().sum().backward()
    assert torch.isfinite(kernel.grad).all() and torch.isfinite(masked.grad).all()
    assert torch.equal(masked.grad[~valid], torch.zeros_like(masked.grad[~valid]))
    write = torch.ones_like(valid); write[:, 3] = False
    observed, rs, ra = observed_context(torch.ones_like(valid), memory_write_mask=write)
    values2 = values.clone(); values2[:, 3] = float("nan"); values2.requires_grad_()
    reset = candidate.observed_difference_residual(values2, kernel, rs, ra)
    _, fs, fa = observed_context(torch.ones_like(valid[:, 4:]))
    fresh = candidate.observed_difference_residual(values[:, 4:], kernel, fs, fa)
    assert torch.equal(reset[:, 4:], fresh)
    assert torch.equal(reset[:, 3:5], torch.zeros_like(reset[:, 3:5]))
    reset.square().sum().backward()
    assert torch.isfinite(values2.grad).all() and torch.isfinite(kernel.grad).all()
    assert torch.equal(values2.grad[:, 3], torch.zeros_like(values2.grad[:, 3]))


def test_difference_channel_independence_causality_and_invalid_context():
    values = torch.randn(1, 8, 64)
    kernel = torch.randn(2, 64)
    _, sources, available = observed_context(torch.ones(1, 8, dtype=torch.bool))
    result = candidate.observed_difference_residual(values, kernel, sources, available)
    altered = values.clone(); altered[:, 5:] += 1000.
    assert torch.equal(result[:, :5], candidate.observed_difference_residual(altered, kernel, sources, available)[:, :5])
    altered = values.clone(); altered[:, 2, 9] += 2.
    changed = candidate.observed_difference_residual(altered, kernel, sources, available)
    assert torch.equal(result[..., 7], changed[..., 7]) and not torch.equal(result[..., 9], changed[..., 9])
    bad = values.clone(); bad[:, 1] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite observed"):
        candidate.observed_difference_residual(bad, kernel, sources, available)
    with pytest.raises(ValueError, match="kernel"):
        candidate.observed_difference_residual(values, torch.zeros(3, 64), sources, available)
    with pytest.raises(ValueError, match="context"):
        candidate.observed_difference_residual(values, kernel, sources.float(), available)


def test_qk_original_arithmetic_v_formula_and_memory_untouched():
    model, _ = build(); activate(model)
    attention = model.encoder.layers[0].attn
    attention._ctx_mem = torch.randn(2, 3, 64)
    x = torch.randn(2, 7, 64)
    mask = torch.ones(2, 7, dtype=torch.bool)
    _, sources, available = observed_context(mask)
    projected = dict(zip(("q", "k", "v"), attention.qkv(x).chunk(3, dim=-1)))
    expected = {name: projected[name] + causal_depthwise_residual(projected[name], getattr(attention, "causal_" + name + "_kernel"), sources, available) for name in ("q", "k")}
    expected["v"] = projected["v"] + candidate.observed_difference_residual(projected["v"], attention.value_difference_kernel, sources, available)
    captured = []
    original_split = attention._split_heads
    def capture(value):
        captured.append(value.detach().clone())
        return original_split(value)
    before = deepcopy(model.state_dict()); context_before = attention._ctx_mem.clone()
    with torch.no_grad(), patch.object(attention, "_split_heads", side_effect=capture):
        _, event_weights = attention(x, mask, observed_sources=sources, observed_available=available, return_event_attention=True)
    assert event_weights.shape == (2, 7, 7) and len(captured) == 3
    memory = torch.cat((context_before, attention.persistent_mem.expand(2, -1, -1)), dim=1)
    for index, name in enumerate(("q", "k", "v")):
        actual = captured[index]
        if name != "q":
            assert torch.equal(actual[:, :memory.shape[1]], memory)
            actual = actual[:, memory.shape[1]:]
        assert torch.equal(actual, expected[name])
    assert torch.equal(attention._ctx_mem, context_before)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_all_difference_parameters_receive_gradients_and_adamw_updates():
    model, _ = build(); attention = model.encoder.layers[0].attn
    values = torch.randn(2, 9, 64)
    _, sources, available = observed_context(torch.ones(2, 9, dtype=torch.bool))
    target = torch.randn_like(values)
    parameter = attention.value_difference_kernel
    optimizer = torch.optim.AdamW([parameter], lr=.001, weight_decay=.01)
    loss = (candidate.observed_difference_residual(values, parameter, sources, available) - target).square().mean()
    loss.backward()
    assert parameter.grad.shape == (2, 64)
    assert torch.isfinite(parameter.grad).all() and bool((parameter.grad != 0).all())
    optimizer.step()
    assert bool((parameter != 0).all())


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
def test_target_padding_future_batch_and_evaluation_state(arm):
    model, _ = build(arm); activate(model)
    dt, quantity, mask = batch()
    before = deepcopy(model.state_dict())
    with torch.no_grad(), patch.object(model.multilag_detail, "forward", side_effect=AssertionError("GRU used")):
        output = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
        changed = quantity.clone(); changed[:, -1] = float("nan"); changed[~mask] = float("nan")
        changed_dt = dt.clone(); changed_dt[~mask] = float("nan")
        altered = core.target_outputs(model, changed_dt, mask, changed, lambda_log_qty=1.)
        assert torch.equal(output["pred_qty"], altered["pred_qty"])
        changed_dt = dt.clone(); changed_dt[:, -1] += 1000.
        altered = core.target_outputs(model, changed_dt, mask, quantity, lambda_log_qty=1.)
        assert torch.equal(output["pred_qty"], altered["pred_qty"])
        encoded = model._encode_base(dt, quantity, mask)
        quantity2 = quantity.clone(); quantity2[:, 5:] += 500.
        dt2 = dt.clone(); dt2[:, 5:] += 500.
        later = model._encode_base(dt2, quantity2, mask)
        assert torch.equal(encoded[:, :5], later[:, :5])
        assert torch.equal(encoded[~mask], torch.zeros_like(encoded[~mask]))
        for index in range(2):
            one = core.target_outputs(model, dt[index:index+1], mask[index:index+1], quantity[index:index+1], lambda_log_qty=1.)
            torch.testing.assert_close(one["pred_qty"], output["pred_qty"][index:index+1], rtol=1e-5, atol=1e-6)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
def test_checkpoint_route_selection_and_replay_fidelity(arm):
    data = payload(arm, context_length=84)
    assert candidate.validate_checkpoint(data, arm)
    rebuilt, _ = build(arm, context_length=84)
    rebuilt.load_state_dict(data["model_state_dict"], strict=False)
    assert all(torch.equal(value, rebuilt.state_dict()[name]) for name, value in data["model_state_dict"].items())
    other = candidate.CNN if arm in candidate.ARMS else candidate.VALUE_DIFFERENCE
    with pytest.raises(ValueError):
        candidate.validate_checkpoint(data, other)
    bad = deepcopy(data); bad["resume_identity"]["checkpoint_monitor"] = "test_raw_quantity_rmse"
    with pytest.raises(ValueError):
        candidate.validate_checkpoint(bad, arm)
    if arm in candidate.ARMS:
        bad = deepcopy(data); bad["model_state_dict"][candidate.DIFFERENCE_KEY][0, 0] = float("nan")
        with pytest.raises(ValueError, match="replacement tensor"):
            candidate.validate_checkpoint(bad, arm)
        for mutation in ("old_v", "missing", "identity", "extra"):
            state = deepcopy(data["model_state_dict"])
            if mutation == "old_v": state[candidate.REMOVED_KEY] = torch.zeros(3, 64)
            elif mutation == "missing": del state[candidate.DIFFERENCE_KEY]
            elif mutation == "identity": state[candidate.IDENTITY_BUFFER][0] ^= 1
            else: state["unused_dummy"] = torch.zeros(64)
            with pytest.raises((ValueError, RuntimeError)):
                rebuilt.load_state_dict(state, strict=False)
        with pytest.raises(ValueError, match="relabelled"):
            candidate.validate_checkpoint(data, candidate.LEGACY_CNN)
        with pytest.raises(ValueError, match="identity"):
            rebuilt.load_state_dict(payload(candidate.CNN)["model_state_dict"])


def test_initialization_hash_api_and_bad_arms():
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    data = route.fabricated_data()
    hashes = route.initial_states(data, 42)
    assert hashes[candidate.CNN] == route.baseline_reference_initial_state_sha(data, 42)
    model, _ = build()
    common = {k: v for k, v in model.state_dict().items() if k not in (candidate.DIFFERENCE_KEY, candidate.IDENTITY_BUFFER)}
    assert canonical_state_dict_sha256(common) == route.common_reference_initial_state_sha(data, 42)
    assert route.initial_states(data, 42, arms=candidate.ARMS) == {candidate.VALUE_DIFFERENCE: hashes[candidate.VALUE_DIFFERENCE]}
    for arms in ((), (candidate.CNN, candidate.CNN), ("wrong",)):
        with pytest.raises(ValueError):
            route.initial_states(data, 42, arms=arms)


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_training_arguments_keep_common_loss_head_budget_optimizer(arm, tmp_path):
    contract = {"training": route.TRAINING.copy(), "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False, "test_evaluation": False}
    stub = SimpleNamespace(force_rerun=False, checkpoint_monitor="validation_raw_quantity_rmse")
    with patch.object(route.base, "training_args", return_value=stub) as mocked:
        args = route.training_args(contract, route.fabricated_data(), tmp_path, arm)
    assert (args.epochs, args.min_epochs, args.early_stopping_patience) == (300, 40, 40)
    assert mocked.call_args.args[0]["model_role"] == route.role_for_arm(arm)
    bad = deepcopy(contract); bad["training"]["patience"] = 20
    with pytest.raises(ValueError, match="budget"):
        route.training_args(bad, route.fabricated_data(), tmp_path, arm)
    for group, key, value in (("model", "lambda_log_qty", .5), ("model", "time_head_mode", "other"), ("optimizer", "lr", .1)):
        bad_data = route.fabricated_data(); bad_data[group][key] = value
        with pytest.raises(ValueError):
            route.training_args(contract, bad_data, tmp_path, arm)


def test_existing_cnn_alias_cannot_start_a_new_fit(tmp_path):
    with pytest.raises(ValueError, match="Only the new value-difference"):
        route.training_args({}, route.fabricated_data(), tmp_path, candidate.CNN)


def test_synthetic_qualification_covers_all_512_active_coefficients():
    result = route.synthetic_check("cpu")
    assert result["status"] == "passed"
    assert result["active_correction_gradient_parameter_count"] == {candidate.CNN: 576, candidate.VALUE_DIFFERENCE: 512}
    assert result["scientific_optimizer_updates"] == 0
    assert result["initial_gradient_nonzero_elements"][candidate.VALUE_DIFFERENCE] == {candidate.DIFFERENCE_KEY: 128}
