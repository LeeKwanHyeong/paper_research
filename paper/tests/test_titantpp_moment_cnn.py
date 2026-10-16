"""Synthetic CPU contract tests; real data, optimizers and remote work prohibited."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import torch

from models.TPPs import CountAwareTitanMomentCNN as candidate
from models.TPPs.CountAwareTitanCNNGRU import observed_context
from paper.scripts import prepare_titantpp_moment_cnn as route
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def forbid_real_work():
    def no(*args, **kwargs):
        raise AssertionError("Synthetic preparation only: no data, checkpoint loading, optimization or remote calls")
    with patch.object(torch, "load", no), patch.object(pl, "read_parquet", no), patch.object(pl, "scan_parquet", no), \
         patch.object(torch.optim, "AdamW", no), patch.object(torch.optim, "Adam", no), patch.object(torch.optim, "SGD", no), \
         patch("subprocess.Popen", no), patch("socket.create_connection", no):
        yield


def build(arm, seed=42):
    torch.manual_seed(seed)
    # Frozen configuration metadata only; no dataset rows or selected weights.
    parent = Path(__file__).resolve().parents[2] / "search_artifacts/titantpp_cnn_history_adaptation_matched_20261015_v1/contract.json"
    data = json.loads(parent.read_text())["dataset"]
    return route.build_model(data, arm)


def batch():
    dt = torch.tensor([[0., 1., 3., 2., 4., 2., 3., 5.], [0., 0., 0., 2., 5., 1., 4., 2.]])
    q = torch.tensor([[0., 0., 5., 2., 7., 4., 3., 6.], [0., 0., 0., 2., 4., 8., 1., 5.]])
    mask = torch.tensor([[False] + [True] * 7, [False] * 3 + [True] * 5])
    return dt, q, mask


def activate(model):
    with torch.no_grad():
        attn = model.encoder.layers[0].attn
        for name in ("q", "k", "v"):
            getattr(attn, "causal_" + name + "_kernel").normal_(0., .1)
        for p in attn.moment_networks.parameters():
            p.normal_(0., .15)
        model.quantity_head.weight.normal_(0., .1)
        model.time_scale_weight.weight.normal_(0., .01)
    model.eval()


def make_filter(arm=candidate.REVIN):
    torch.manual_seed(12)
    values = torch.randn(2, 9, 5)
    valid = torch.tensor([[True, True, False, True, True, True, True, True, True], [True] * 9])
    write = valid.clone(); write[0, 5] = False
    observed, sources, available = observed_context(valid, memory_write_mask=write)
    mean, std, count = candidate.prefix_moments(values, valid, observed)
    kernel = torch.randn(3, 5)
    moments = torch.randn(2, 9, 2)
    network = None if arm == candidate.REVIN else candidate.MomentController()
    return values, valid, observed, sources, available, mean, std, count, kernel, moments, network


@pytest.mark.parametrize("arm", candidate.ARMS)
@pytest.mark.parametrize("seed", [42, 52, 62])
def test_original_parameters_rng_heads_and_neutral_output(arm, seed):
    baseline, _ = build(candidate.CNN, seed); rng = torch.get_rng_state().clone()
    model, _ = build(arm, seed)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in baseline.state_dict().items())
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert count(model) - count(baseline) == candidate.EXTRA_PARAMETERS[arm]
    assert not any(p.requires_grad for p in model.multilag_detail.parameters())
    # Inherit the exact functions, not a reimplementation of the heads/loss.
    for name in ("predict_quantity", "quantity_outputs", "log_observation_dt", "encode_task_states"):
        assert getattr(model, name).__func__ is getattr(baseline, name).__func__
    dt, q, mask = batch(); baseline.eval(); model.eval()
    with torch.no_grad():
        expected = core.target_outputs(baseline, dt, mask, q, lambda_log_qty=1.)
        actual = core.target_outputs(model, dt, mask, q, lambda_log_qty=1.)
    for name in expected:
        if isinstance(expected[name], torch.Tensor):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_prefix_statistics_manual_padding_and_withheld_reset():
    values = torch.tensor([[[2.], [4.], [float("nan")], [8.], [float("nan")], [10.], [14.]]])
    valid = torch.tensor([[True, True, False, True, True, True, True]])
    observed = valid.clone(); observed[0, 4] = False
    mean, std, n = candidate.prefix_moments(values, valid, observed)
    torch.testing.assert_close(mean[0, :, 0], torch.tensor([2., 3., 0., 14 / 3, 0., 10., 12.]))
    assert n.tolist() == [[1, 2, 0, 3, 0, 1, 2]]
    assert std[0, 2, 0] == std[0, 4, 0] == 1.
    torch.testing.assert_close(std[0, 3, 0].square(), torch.tensor(56 / 9 + candidate.EPS))
    torch.testing.assert_close(std[0, 6, 0].square(), torch.tensor(4. + candidate.EPS))


def test_statistics_constant_short_empty_detached_and_invalid_input():
    values = torch.full((2, 4, 3), 7., requires_grad=True)
    valid = torch.ones(2, 4, dtype=torch.bool)
    mean, std, _ = candidate.prefix_moments(values, valid, valid)
    assert not mean.requires_grad and not std.requires_grad
    torch.testing.assert_close(mean, values)
    torch.testing.assert_close(std, torch.full_like(std, candidate.EPS ** .5))
    empty = values[:, :0]; m, s, n = candidate.prefix_moments(empty, valid[:, :0], valid[:, :0])
    assert m.shape == s.shape == empty.shape and n.shape == (2, 0)
    bad = values.detach().clone(); bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        candidate.prefix_moments(bad, valid, valid)
    with pytest.raises(ValueError, match="observed"):
        candidate.prefix_moments(values, ~valid, valid)


def test_revin_formula_roundtrip_centered_equivalence_and_latent_scale_cancellation():
    x, valid, obs, sources, available, mean, std, _, kernel, _, _ = make_filter()
    residual = candidate.reversible_cnn_residual(x, kernel, mean, std, sources, available, arm=candidate.REVIN)
    taps = candidate.normalized_taps(x, mean, std, sources, available)
    normalized_output = taps[..., 0, :] + sum(taps[..., i, :] * kernel[i] for i in range(3))
    explicit = torch.where(obs[..., None], mean + std * normalized_output, 0.)
    torch.testing.assert_close(torch.where(obs[..., None], x + residual, 0.), explicit, rtol=1e-5, atol=1e-6)
    centered = torch.zeros_like(x)
    for i in range(3):
        lag = torch.where(obs[..., None], x, 0.).gather(1, sources[..., i, None].expand_as(x))
        centered += torch.where(available[..., i, None], lag - mean, 0.) * kernel[i]
    torch.testing.assert_close(residual, centered, rtol=1e-5, atol=1e-6)
    for factor in (.1, 10.):
        alternate = candidate.reversible_cnn_residual(x, kernel, mean, std * factor, sources, available, arm=candidate.REVIN)
        torch.testing.assert_close(alternate, residual, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("arm", [candidate.CONDITIONED, candidate.CAPACITY])
def test_zero_controller_matches_plain_revin_with_nonzero_cnn(arm):
    x, _, _, sources, available, mean, std, _, kernel, moments, network = make_filter(arm)
    actual = candidate.reversible_cnn_residual(x, kernel, mean, std, sources, available,
        arm=arm, controller=network, moments=moments)
    plain = candidate.reversible_cnn_residual(x, kernel, mean, std, sources, available, arm=candidate.REVIN)
    torch.testing.assert_close(actual, plain, rtol=0, atol=0)


def test_conditioned_weights_bound_mean_and_single_observation():
    valid = torch.tensor([[True, True, False, True, True]])
    write = valid.clone(); write[0, 3] = False
    _, _, available = observed_context(valid, memory_write_mask=write)
    for value in (-1e9, 0., 1e9):
        weights = candidate.tap_weights(torch.full((1, 5, 1), value), available)
        assert weights.min() >= .5 and weights.max() <= 1.5
        torch.testing.assert_close((weights * available[..., None]).sum(-2).squeeze(-1), available.sum(-1).float())
        assert torch.equal(weights[:, 0], torch.ones_like(weights[:, 0]))
        assert torch.equal(weights[:, 4], torch.ones_like(weights[:, 4]))


def test_mean_and_std_each_change_conditioned_filter_not_just_parameter_count():
    x, _, _, sources, available, mean, std, _, kernel, moments, network = make_filter(candidate.CONDITIONED)
    with torch.no_grad():
        network.network[0].weight.fill_(.1); network.network[0].bias.zero_()
        network.network[2].weight.fill_(.1)
    zero = torch.zeros_like(moments)
    reference = candidate.reversible_cnn_residual(x, kernel, mean, std, sources, available,
        arm=candidate.CONDITIONED, controller=network, moments=zero)
    for channel in (0, 1):
        changed = zero.clone(); changed[..., channel] = 2.
        actual = candidate.reversible_cnn_residual(x, kernel, mean, std, sources, available,
            arm=candidate.CONDITIONED, controller=network, moments=changed)
        assert not torch.allclose(actual, reference)


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_unchanged_attention_and_memory_when_normalization_is_identity(arm):
    model, _ = build(arm); baseline, _ = build(candidate.CNN)
    with torch.no_grad():
        for name in ("q", "k", "v"):
            getattr(model.encoder.layers[0].attn, "causal_" + name + "_kernel").normal_(0., .1)
        model.quantity_head.weight.normal_(0., .1)
    baseline.load_state_dict({k: v for k, v in model.state_dict().items() if k in baseline.state_dict()})
    model.eval(); baseline.eval(); dt, q, mask = batch()
    def neutral(values, valid, observed):
        return torch.zeros_like(values), torch.ones_like(values), observed.long().cumsum(1)
    with torch.no_grad(), patch.object(candidate, "prefix_moments", neutral):
        expected = core.target_outputs(baseline, dt, mask, q, lambda_log_qty=1.)
        actual = core.target_outputs(model, dt, mask, q, lambda_log_qty=1.)
    for name in expected:
        if isinstance(expected[name], torch.Tensor):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_target_padding_batch_future_and_eval_state_invariance(arm):
    model, _ = build(arm); activate(model)
    dt, q, mask = batch(); before = deepcopy(model.state_dict())
    with torch.no_grad(), patch.object(model.multilag_detail, "forward", side_effect=AssertionError("GRU called")):
        original = core.target_outputs(model, dt, mask, q, lambda_log_qty=1.)
        changed = q.clone(); changed[:, -1] = float("nan"); changed[~mask] = float("nan")
        d = dt.clone(); d[:, -1] += 1000.; d[~mask] = float("nan")
        altered = core.target_outputs(model, d, mask, changed, lambda_log_qty=1.)
        torch.testing.assert_close(original["pred_qty"], altered["pred_qty"], rtol=0, atol=0)
        for i in range(2):
            one = core.target_outputs(model, dt[i:i+1], mask[i:i+1], q[i:i+1], lambda_log_qty=1.)
            torch.testing.assert_close(one["pred_qty"], original["pred_qty"][i:i+1], rtol=1e-5, atol=1e-6)
        a = model._encode_base(dt, q, mask)
        q[:, 5:] += 999.; dt[:, 5:] += 444.
        b = model._encode_base(dt, q, mask)
        torch.testing.assert_close(a[:, :5], b[:, :5], rtol=0, atol=0)
        assert torch.equal(a[~mask], torch.zeros_like(a[~mask]))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("arm", [candidate.CONDITIONED, candidate.CAPACITY])
def test_gradient_path_all_parameters_after_neutral_initialization(arm):
    x, _, _, sources, available, mean, std, _, kernel, moments, network = make_filter(arm)
    x.requires_grad_(True); kernel.requires_grad_(True)
    # The last layer is zero initially; exercise non-neutral gradients without an optimizer step.
    with torch.no_grad(): network.network[2].weight.normal_(0., .2)
    output = candidate.reversible_cnn_residual(x, kernel, mean, std, sources, available,
        arm=arm, controller=network, moments=moments)
    output.square().mean().backward()
    for parameter in network.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and (parameter.grad != 0).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(kernel.grad).all()


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_complete_checkpoint_identity_selector_and_state_validation(arm):
    model, meta = build(arm); rebuilt, _ = build(arm)
    rebuilt.load_state_dict(model.state_dict())
    interface = {"time_head": meta["time_head"]}
    payload = {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False, "variant": "count_only_log_regression",
        "interface_meta": interface, "resume_identity": {"backbone": arm, "interface_meta": interface,
        "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": candidate.role_for_arm(arm)}}}
    assert candidate.validate_checkpoint(payload, arm)
    with pytest.raises(ValueError, match="relabelled"):
        candidate.validate_checkpoint(payload, candidate.CNN)
    other = candidate.CONDITIONED if arm != candidate.CONDITIONED else candidate.CAPACITY
    with pytest.raises(ValueError, match="route"):
        candidate.validate_checkpoint(payload, other)
    wrong, _ = build(other)
    with pytest.raises(ValueError, match="identity"):
        wrong.load_state_dict(model.state_dict())
    bad = deepcopy(payload); bad["resume_identity"]["checkpoint_monitor"] = "test_raw_quantity_rmse"
    with pytest.raises(ValueError): candidate.validate_checkpoint(bad, arm)
    bad = deepcopy(payload); bad["encoder_config"]["normalization_eps"] *= 10
    with pytest.raises(ValueError): candidate.validate_checkpoint(bad, arm)
    bad = deepcopy(model.state_dict()); bad["encoder.layers.0.attn.causal_q_kernel"][0, 0] = float("nan")
    with pytest.raises(RuntimeError): rebuilt.load_state_dict(bad)
    if arm != candidate.REVIN:
        bad = deepcopy(payload); bad["model_state_dict"][candidate.EXTRA_PREFIX + "q.network.0.weight"][0, 0] = float("nan")
        with pytest.raises(ValueError, match="network state"): candidate.validate_checkpoint(bad, arm)


def test_no_training_or_global_candidate_route():
    with pytest.raises(PermissionError, match="Preparation only"):
        route.run_fit()
    build(candidate.REVIN)
    from paper.scripts.count_aware_tpp_backbone import constants
    assert all(arm not in constants.SUPPORTED_BACKBONES for arm in candidate.ARMS)


def test_preparation_integrity_failure_precedes_output(tmp_path):
    output = tmp_path / "not_created"
    project = Path(__file__).resolve().parents[2]
    with patch.object(route, "raw_sha", return_value="0" * 64):
        with pytest.raises(ValueError, match="parent preparation contract mismatch"):
            route.prepare_bundle(project, output)
    assert not output.exists()


def test_preparation_refuses_existing_directory_without_overwrite(tmp_path):
    output = tmp_path / "existing"; output.mkdir()
    sentinel = output / "contract.json"; sentinel.write_text("preserve me")
    project = Path(__file__).resolve().parents[2]
    with pytest.raises(FileExistsError):
        route.prepare_bundle(project, output)
    assert sentinel.read_text() == "preserve me"
    assert list(output.iterdir()) == [sentinel]
