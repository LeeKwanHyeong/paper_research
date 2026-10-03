import math

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.quantity_objective_comparison import (
    QuantityCase,
    QuantityCaseIdentity,
    QuantityStatistics,
    fit_train_quantity_statistics,
    joint_causal_batch_objective,
    quantity_loss,
    quantity_prediction,
)


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def make_model(seed=17):
    torch.manual_seed(seed)
    result, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=statistics().mu, max_seq_len=8, time_intercept_limit=300.0)
    return result


def batch():
    return (
        torch.tensor([[0., 0., 1., 2.], [0., 1., 1., 3.]]),
        torch.tensor([[False, False, True, True], [False, True, True, True]]),
        torch.tensor([[0., 0., 2., 4.], [0., 1., 2., 3.]]),
    )


def statistics():
    return fit_train_quantity_statistics(torch.tensor([0., 2., 4., 1., 2., 3.]), torch.tensor([4., 3.]), split="train")


def test_hand_calculated_links_losses_and_autograd():
    stats = QuantityStatistics(mu=math.log(3.0), raw_scale=5.0)
    logits = torch.tensor([0., 1.], requires_grad=True)
    targets = torch.tensor([1., 4.])
    raw = quantity_loss(logits, targets, stats, QuantityCase.RAW_ORIGINAL)
    expected_prediction = torch.expm1(torch.nn.functional.softplus(logits))
    assert torch.equal(raw["point_prediction"], expected_prediction)
    assert torch.allclose(raw["train_loss"], ((expected_prediction - targets) / 5.).square())
    raw["train_loss"].sum().backward()
    assert torch.isfinite(logits.grad).all() and not torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_original_log_loss_uses_b_arithmetic_without_round_trip():
    stats = QuantityStatistics(mu=1.0, raw_scale=2.0)
    logits = torch.tensor([-5.0, 0.3, 4.0])
    targets = torch.tensor([0.0, 3.0, 25.0])
    actual = quantity_loss(logits, targets, stats, QuantityCase.B_LOG_ORIGINAL)["log_mse"]
    expected = torch.nn.functional.mse_loss(
        torch.nn.functional.softplus(logits),
        torch.log1p(targets.clamp_min(0.0)),
        reduction="none",
    )
    assert torch.equal(actual, expected)
    sensitive = torch.tensor([-16.41912078857422], dtype=torch.float32)
    direct = torch.nn.functional.softplus(sensitive)
    round_trip = torch.log1p(torch.expm1(direct))
    assert not torch.equal(direct, round_trip)


@pytest.mark.parametrize("case", list(QuantityCase))
def test_all_cases_have_finite_cpu_optimizer_steps(case):
    candidate = make_model()
    optimizer = torch.optim.SGD(candidate.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = joint_causal_batch_objective(candidate, *batch(), statistics=statistics(), case=case)
        outputs["objective_loss"].mean().backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in candidate.parameters())
        optimizer.step()


def test_new_link_initial_value_and_slope_match_original_at_b_initialization():
    stats = QuantityStatistics(mu=1.0, raw_scale=2.0)
    z = torch.tensor([stats.z0], dtype=torch.float64, requires_grad=True)
    original = quantity_prediction(z, stats, QuantityCase.RAW_ORIGINAL)
    new = quantity_prediction(z, stats, QuantityCase.RAW_SOFTPLUS)
    original_slope, = torch.autograd.grad(original.sum(), z, retain_graph=True)
    new_slope, = torch.autograd.grad(new.sum(), z)
    assert new.item() == pytest.approx(stats.q0, abs=1e-12)
    assert original.item() == pytest.approx(stats.q0, abs=1e-12)
    assert original_slope.item() == pytest.approx(stats.q0, abs=1e-12)
    assert new_slope.item() == pytest.approx(stats.q0, abs=1e-12)


def test_b_control_is_bitwise_joint_objective_and_gradient_identity():
    first, second = make_model(), make_model()
    arguments = batch()
    rng = torch.get_rng_state()
    baseline = target_outputs(first, *arguments, lambda_log_qty=1.0)
    baseline["joint_loss"].mean().backward()
    torch.set_rng_state(rng)
    candidate = joint_causal_batch_objective(second, *arguments, statistics=statistics(), case=QuantityCase.B_LOG_ORIGINAL)
    candidate["objective_loss"].mean().backward()
    assert torch.equal(baseline["joint_loss"], candidate["objective_loss"])
    for (name, before), (_, after) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        assert (before.grad is None and after.grad is None) or torch.equal(before.grad, after.grad), name


def test_target_and_padding_are_causal_and_model_state_is_unchanged():
    candidate = make_model().eval()
    state = {name: value.clone() for name, value in candidate.state_dict().items()}
    count = sum(parameter.numel() for parameter in candidate.parameters())
    original = joint_causal_batch_objective(candidate, *batch(), statistics=statistics(), case=QuantityCase.LOG_SOFTPLUS)
    dts, mask, quantities = batch()
    dts[~mask] = 999.
    quantities[~mask] = 999.
    dts[:, -1] = 111.
    quantities[:, -1] = 222.
    changed = joint_causal_batch_objective(candidate, dts, mask, quantities, statistics=statistics(), case=QuantityCase.LOG_SOFTPLUS)
    assert torch.allclose(original["pred_qty"], changed["pred_qty"], atol=1e-6, rtol=0)
    assert count == sum(parameter.numel() for parameter in candidate.parameters())
    assert all(torch.equal(value, candidate.state_dict()[name]) for name, value in state.items())


def test_initial_quantity_encoder_gradient_is_zero_when_head_weight_is_zero():
    candidate = make_model()
    outputs = joint_causal_batch_objective(candidate, *batch(), statistics=statistics(), case=QuantityCase.LOG_SOFTPLUS)
    quantity_gradients = torch.autograd.grad(outputs["quantity_train_loss"].mean(), tuple(candidate.parameters()), allow_unused=True)
    encoder = [(name, grad) for (name, _), grad in zip(candidate.named_parameters(), quantity_gradients, strict=True) if not name.startswith("quantity_head.")]
    assert all(grad is None or torch.equal(grad, torch.zeros_like(grad)) for _, grad in encoder)


def test_train_statistics_units_identity_and_rejections():
    stats = fit_train_quantity_statistics(torch.tensor([0., 3.]), torch.tensor([3., 4.]), split="train")
    assert stats.mu == pytest.approx((math.log1p(0.) + math.log1p(3.)) / 2)
    assert stats.raw_scale == pytest.approx(math.sqrt((9. + 16.) / 2))
    with pytest.raises(ValueError, match="at least one"):
        QuantityStatistics(mu=1., raw_scale=.5)
    identity = QuantityCaseIdentity(QuantityCase.RAW_SOFTPLUS, stats)
    identity.require_match(identity.to_json())
    with pytest.raises(ValueError, match="identity"):
        identity.require_match(QuantityCaseIdentity(QuantityCase.LOG_SOFTPLUS, stats).to_json())
    with pytest.raises(ValueError, match="train split"):
        fit_train_quantity_statistics(torch.tensor([1.]), torch.tensor([1.]), split="validation")
    with pytest.raises(ValueError, match="finite"):
        fit_train_quantity_statistics(torch.tensor([float("nan")]), torch.tensor([1.]), split="train")
    with pytest.raises(ValueError, match="nonnegative"):
        quantity_loss(torch.tensor([0.]), torch.tensor([-1.]), stats, QuantityCase.RAW_ORIGINAL)
    with pytest.raises(ValueError, match="nonfinite"):
        quantity_loss(torch.tensor([1000.]), torch.tensor([1.]), stats, QuantityCase.RAW_ORIGINAL)
    # The link can be finite while raw squared error overflows; fail closed.
    with pytest.raises(ValueError, match="loss is nonfinite"):
        quantity_loss(torch.tensor([60.]), torch.tensor([1.]), QuantityStatistics(mu=1., raw_scale=1.), QuantityCase.RAW_ORIGINAL)
    assert fit_train_quantity_statistics(torch.tensor([1.]), torch.tensor([0.]), split="train").raw_scale == 1.0
    large = torch.tensor([16_777_217, 16_777_219], dtype=torch.int64)
    large_stats = fit_train_quantity_statistics(large, large, split="train")
    assert large_stats.mu == pytest.approx(sum(math.log1p(int(value)) for value in large) / 2)
    assert large_stats.raw_scale == pytest.approx(math.sqrt(sum(int(value) ** 2 for value in large) / 2))


def test_joint_guards_invalid_variant_empty_batch_and_nonfinite_time_loss():
    candidate = make_model()
    candidate.quantity_variant = "not_b"
    with pytest.raises(ValueError, match="count_only_log_regression"):
        joint_causal_batch_objective(candidate, *batch(), statistics=statistics(), case=QuantityCase.RAW_ORIGINAL)
    candidate = make_model()
    with pytest.raises(ValueError, match="Empty batch"):
        joint_causal_batch_objective(candidate, torch.empty((0, 2)), torch.empty((0, 2), dtype=torch.bool), torch.empty((0, 2)), statistics=statistics(), case=QuantityCase.RAW_ORIGINAL)
    candidate = make_model()
    with torch.no_grad():
        candidate.b_t.fill_(1000.)
    with pytest.raises(ValueError, match="joint objective is nonfinite"):
        joint_causal_batch_objective(candidate, *batch(), statistics=statistics(), case=QuantityCase.RAW_ORIGINAL)


def test_b_batch_rejects_nonbaseline_cap_while_standalone_raw_kernel_is_cap_independent():
    candidate = make_model()
    candidate.time_intercept_limit = 30.0
    with pytest.raises(ValueError, match="cap 300"):
        joint_causal_batch_objective(candidate, *batch(), statistics=statistics(), case=QuantityCase.B_LOG_ORIGINAL)
    raw = quantity_loss(torch.tensor([0.]), torch.tensor([1.]), statistics(), QuantityCase.RAW_ORIGINAL)
    assert torch.isfinite(raw["train_loss"]).all()
