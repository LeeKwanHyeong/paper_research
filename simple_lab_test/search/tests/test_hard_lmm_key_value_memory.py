"""Local synthetic contracts, not a performance or real-data training run."""

import copy
import io
import os

import pytest
import torch
from torch.nn import functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.Titan.common.key_value_memory import (
    KEY_VALUE_BACKBONE as CANDIDATE,
    KEY_VALUE_CONTRACT,
    KeyValueLocalMemoryMatcher,
)
from models.Titan.common.memory import HardLocalMemoryMatcher, LMM, SimilarityWeightedLocalMemoryMatcher
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


DEVICE = os.environ.get("HARD_KEY_VALUE_TEST_DEVICE", "cpu")
WEIGHTED = "titantpp_weighted_static_memory"
VARIANT = "count_only_log_regression"


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(backbone=CANDIDATE, *, dim=16, max_len=8):
    torch.manual_seed(42)
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=dim, train_log_mean=1.5, max_seq_len=max_len,
        quantity_variant=VARIANT, lambda_tail=0., time_head_mode="legacy_clamped_rmtpp",
    )
    return model.to(DEVICE), metadata


def batch():
    dts = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 0.]], device=DEVICE)
    qty = torch.tensor([[1., 5., 9., 20.], [2., 6., 12., 0.]], device=DEVICE)
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]], device=DEVICE)
    return dts, qty, mask


def outputs(model, data=None):
    dts, qty, mask = batch() if data is None else data
    return target_outputs(model, dts, mask, qty, lambda_log_qty=1.)


def manual(x, keys, values, topk):
    all_scores = F.normalize(x, dim=-1) @ F.normalize(keys, dim=-1).transpose(-1, -2)
    scores, indices = all_scores.topk(topk, dim=-1)
    selected = values[0][indices]
    weights = scores.softmax(-1)
    return (weights.unsqueeze(-1) * selected).sum(-2), indices, weights


def test_formula_and_gradients_match_independent_reference():
    torch.manual_seed(7)
    module = KeyValueLocalMemoryMatcher(8, 12, 4).to(device=DEVICE, dtype=torch.float64)
    with torch.no_grad():
        module.memory_keys.normal_()
        module.mem.normal_()
    x = torch.randn(2, 5, 8, device=DEVICE, dtype=torch.float64, requires_grad=True)
    actual, trace = module.retrieve(x)
    expected, indices, weights = manual(x, module.memory_keys, module.mem, 4)
    # Independent broadcasting/gather layouts may differ by float64 roundoff.
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(trace["retrieval_weights"], weights, rtol=1e-12, atol=1e-12)
    assert torch.equal(trace["prototype_indices"], indices)
    assert not indices.requires_grad
    assert trace["topk_similarity"].requires_grad
    parameters = (x, module.memory_keys, module.mem)
    actual_grads = torch.autograd.grad((x + actual).square().sum(), parameters, retain_graph=True)
    expected_grads = torch.autograd.grad((x + expected).square().sum(), parameters)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        assert torch.isfinite(a).all() and a.abs().sum() > 0


def test_tied_initialization_preserves_weighted_output_rng_and_gradient_sum():
    torch.manual_seed(12)
    weighted = SimilarityWeightedLocalMemoryMatcher(8, 12, 4).to(DEVICE)
    weighted_rng = torch.get_rng_state()
    torch.manual_seed(12)
    candidate = KeyValueLocalMemoryMatcher(8, 12, 4).to(DEVICE)
    assert torch.equal(weighted_rng, torch.get_rng_state())
    assert candidate.mem.data_ptr() != candidate.memory_keys.data_ptr()
    assert torch.equal(candidate.mem, candidate.memory_keys)
    assert torch.equal(candidate.mem, weighted.mem)
    x = torch.randn(2, 4, 8, device=DEVICE, requires_grad=True)
    a, b = weighted(x), candidate(x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward(retain_graph=True)
    reference_x_grad = x.grad.clone()
    x.grad = None
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x_grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weighted.mem.grad, candidate.mem.grad + candidate.memory_keys.grad,
                               rtol=1e-6, atol=1e-6)


def test_keys_change_addressing_values_do_not_and_unselected_keys_get_no_loss_gradient():
    torch.manual_seed(9)
    module = KeyValueLocalMemoryMatcher(4, 9, 4).to(DEVICE)
    x = torch.tensor([[[1., .2, -.1, .3]]], device=DEVICE)
    initial_values = module.mem.detach().clone()
    before, trace = module.retrieve(x)
    before.sum().backward()
    selected = trace["prototype_indices"].unique()
    unselected = torch.ones(9, dtype=torch.bool, device=DEVICE)
    unselected[selected] = False
    assert torch.count_nonzero(module.memory_keys.grad[0, unselected]) == 0
    assert module.memory_keys.grad[0, selected].abs().sum() > 0
    with torch.no_grad():
        module.memory_keys.copy_(module.memory_keys.roll(1, dims=1))
    changed, new_trace = module.retrieve(x)
    assert torch.equal(module.mem, initial_values)
    assert not torch.equal(trace["prototype_indices"], new_trace["prototype_indices"])
    assert not torch.allclose(before, changed)
    with torch.no_grad():
        module.mem.add_(.4)
    new_residual, value_trace = module.retrieve(x)
    assert torch.equal(new_trace["prototype_indices"], value_trace["prototype_indices"])
    assert torch.equal(new_trace["retrieval_weights"], value_trace["retrieval_weights"])
    torch.testing.assert_close(new_residual, changed + .4)


def test_key_only_synthetic_fit_changes_retrieval_without_changing_values():
    torch.manual_seed(27)
    module = KeyValueLocalMemoryMatcher(4, 8, 4).to(DEVICE)
    with torch.no_grad():
        module.memory_keys.normal_()
        module.mem.normal_()
    module.mem.requires_grad_(False)
    values = module.mem.detach().clone()
    initial_keys = module.memory_keys.detach().clone()
    x = torch.randn(3, 2, 4, device=DEVICE)
    target = torch.zeros_like(x)
    optimizer = torch.optim.SGD([module.memory_keys], lr=.01)
    initial_loss = F.mse_loss(module.retrieve(x)[0], target).item()
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(module.retrieve(x)[0], target)
        loss.backward()
        assert module.memory_keys.grad.abs().sum() > 0
        optimizer.step()
    assert F.mse_loss(module.retrieve(x)[0], target).item() < initial_loss
    assert torch.equal(values, module.mem)
    assert not torch.equal(initial_keys, module.memory_keys)


@pytest.mark.parametrize("topk", [1, 4])
def test_known_zero_addressing_gradient_cases_are_not_hidden(topk):
    module = KeyValueLocalMemoryMatcher(4, 8, topk).to(DEVICE)
    if topk == 4:
        with torch.no_grad():
            module.mem.fill_(1.)
    module(torch.ones(1, 2, 4, device=DEVICE)).sum().backward()
    torch.testing.assert_close(module.memory_keys.grad, torch.zeros_like(module.memory_keys),
                               rtol=0, atol=1e-6)


@pytest.mark.parametrize("dim,mem_size,topk", [(0, 4, 4), (8, 0, 1), (8, 4, 0), (8, 4, 5)])
def test_invalid_configuration_rejected(dim, mem_size, topk):
    with pytest.raises(ValueError):
        KeyValueLocalMemoryMatcher(dim, mem_size, topk)


def test_external_bank_override_and_wrong_input_shape_rejected():
    module = KeyValueLocalMemoryMatcher(8, 8, 4).to(DEVICE)
    with pytest.raises(ValueError, match="value-only"):
        module(torch.randn(2, 3, 8, device=DEVICE), memory=module.mem)
    for x in (torch.zeros(2, 8, device=DEVICE), torch.zeros(2, 3, 4, device=DEVICE)):
        with pytest.raises(ValueError, match="shape"):
            module(x)


@pytest.mark.parametrize("magnitude", [0., 1e10])
def test_extreme_finite_retrieval_forward_backward(magnitude):
    torch.manual_seed(31)
    module = KeyValueLocalMemoryMatcher(8, 12, 4).to(DEVICE)
    x = (torch.randn(2, 5, 8, device=DEVICE) * magnitude).requires_grad_()
    y = module(x)
    y.square().mean().backward()
    for t in (y, x.grad, module.memory_keys.grad, module.mem.grad):
        assert torch.isfinite(t).all()


@pytest.mark.parametrize("training", [False, True])
def test_full_model_weighted_identity_original_parameters_and_rng(training):
    original, original_meta = build("titantpp")
    old_rng = torch.get_rng_state()
    weighted, weighted_meta = build(WEIGHTED)
    candidate, metadata = build()
    assert torch.equal(old_rng, torch.get_rng_state())
    assert LMM is HardLocalMemoryMatcher and type(original.lmm) is HardLocalMemoryMatcher
    assert type(weighted.lmm) is SimilarityWeightedLocalMemoryMatcher
    assert set(candidate.state_dict()) - set(original.state_dict()) == {"lmm.memory_keys"}
    for name, value in original.state_dict().items():
        assert torch.equal(value, weighted.state_dict()[name]), name
        assert torch.equal(value, candidate.state_dict()[name]), name
    assert metadata["additional_parameter_count"] == 64 * 16
    assert (sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in original.parameters())) == 64 * 16
    assert original_meta["backbone_contract_id"] == "B0"
    assert weighted_meta["backbone_contract_id"] == "W0"
    for key in ("d_model", "n_layers", "n_heads", "d_ff", "persistent_mem_size", "lmm_mem_size", "lmm_topk", "time_head"):
        assert original_meta[key] == weighted_meta[key] == metadata[key]
    assert candidate.soft_memory is None and candidate.surprise_memory is None
    assert candidate.tpp_gated_memory is None and candidate.titans_mac_encoder is None
    assert metadata["static_key_value_tied"] is False
    candidate.train(training)
    weighted.train(training)
    for model in (weighted, candidate):
        with torch.no_grad():
            model.quantity_head.weight.fill_(.05)
    torch.manual_seed(391)
    a = outputs(weighted)
    torch.manual_seed(391)
    b = outputs(candidate)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


@pytest.mark.parametrize("max_len,original_count", [(256, 89795), (84, 78787), (64, 77507)])
def test_official_h64_parameter_count(max_len, original_count):
    model, metadata = build(dim=64, max_len=max_len)
    assert sum(p.numel() for p in model.parameters()) == original_count + 4096
    assert metadata["additional_parameter_count"] == 4096


@pytest.mark.parametrize("task", ["quantity_train_loss", "time_loss"])
def test_task_gradients_reach_keys_values_and_encoder(task):
    model, _ = build()
    model.eval()
    with torch.no_grad():
        model.quantity_head.weight.normal_(0., .1)
    outputs(model)[task].mean().backward()
    for parameter in (model.lmm.mem, model.lmm.memory_keys):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_real_synthetic_optimizer_steps_activate_quantity_key_gradient():
    model, _ = build()
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    initial_keys = model.lmm.memory_keys.detach().clone()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = outputs(model)
        key_gradient = torch.autograd.grad(out["quantity_train_loss"].mean(), model.lmm.memory_keys,
                                           retain_graph=True)[0]
        if step == 0:
            assert torch.count_nonzero(key_gradient) == 0
        else:
            assert key_gradient.abs().sum() > 0
        out["joint_loss"].mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    assert not torch.equal(initial_keys, model.lmm.memory_keys)
    assert not torch.equal(model.lmm.memory_keys, model.lmm.mem)


def test_key_only_intervention_changes_quantity_prediction():
    model, _ = build()
    model.eval()
    with torch.no_grad():
        model.quantity_head.weight.normal_(0., .1)
    before = outputs(model)["pred_qty"].detach()
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.lmm.memory_keys.copy_(model.lmm.memory_keys.roll(1, dims=1))
    after = outputs(model)["pred_qty"].detach()
    assert not torch.allclose(before, after, rtol=1e-7, atol=1e-7)
    for name, value in model.state_dict().items():
        if name != "lmm.memory_keys":
            assert torch.equal(value, snapshot[name]), name


def test_target_future_padding_series_isolation_and_no_eval_mutation():
    model, _ = build()
    model.eval()
    with torch.no_grad():
        model.quantity_head.weight.normal_(0., .1)
    dts, qty, mask = batch()
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    original = outputs(model)
    changed_qty, changed_dts = qty.clone(), dts.clone()
    changed_qty[0, 3], changed_qty[1, 2], changed_qty[1, 3] = 9e6, 8e6, 7e6
    changed_dts[0, 3], changed_dts[1, 2], changed_dts[1, 3] = 6e6, 5e6, 4e6
    changed = outputs(model, (changed_dts, changed_qty, mask))
    torch.testing.assert_close(original["pred_qty"], changed["pred_qty"], rtol=0, atol=0)
    states = model.encode(dts, qty, mask)
    altered = model.encode(changed_dts, changed_qty, mask)
    torch.testing.assert_close(states[:, :2], altered[:, :2], rtol=0, atol=0)
    assert torch.count_nonzero(states[1, 3]) == 0
    for row in range(2):
        alone = model.encode(dts[row:row+1], qty[row:row+1], mask[row:row+1])
        torch.testing.assert_close(states[row:row+1], alone, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(states, model.encode(dts.flip(0), qty.flip(0), mask.flip(0)).flip(0),
                               rtol=0, atol=0)
    # Equivalent left padding goes through the production target preparation.
    left_dts, left_qty, left_mask = dts.clone(), qty.clone(), mask.clone()
    for t in (left_dts, left_qty, left_mask):
        t[1] = t[1].roll(1)
    left = outputs(model, (left_dts, left_qty, left_mask))
    torch.testing.assert_close(left["pred_qty"], original["pred_qty"], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        assert torch.equal(value, snapshot[name]), name


def test_extreme_finite_full_model():
    model, _ = build()
    dts, qty, mask = batch()
    qty = qty * 1e8
    out = outputs(model, (dts * 1e8, qty, mask))
    assert all(torch.isfinite(value).all() for value in out.values())
    out["joint_loss"].mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_checkpoint_and_next_optimizer_step_replay():
    model, metadata = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)

    def step(m, opt):
        m.train()
        opt.zero_grad(set_to_none=True)
        outputs(m)["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        opt.step()

    step(model, optimizer)
    payload = {"backbone": CANDIDATE, "variant": VARIANT, "encoder_config": metadata,
               "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True, map_location=DEVICE)
    validate_checkpoint_route(saved, CANDIDATE)
    restored, _ = build()
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_optimizer.load_state_dict(saved["optimizer_state_dict"])
    assert restored.lmm.mem.data_ptr() != restored.lmm.memory_keys.data_ptr()
    model.eval()
    restored.eval()
    for name, value in outputs(model).items():
        torch.testing.assert_close(value, outputs(restored)[name], rtol=0, atol=0)
    torch.manual_seed(501)
    step(model, optimizer)
    torch.manual_seed(501)
    step(restored, restored_optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_identity_and_strict_state_reject_mislabeling():
    model, metadata = build()
    payload = {"backbone": CANDIDATE, "variant": VARIANT, "encoder_config": metadata,
               "model_state_dict": model.state_dict()}
    validate_checkpoint_route(payload, CANDIDATE)
    mutations = [
        {"backbone": "titantpp"}, {"variant": "tail_shared"},
        {"encoder_config": {}}, {"model_state_dict": {}},
    ]
    for mutation in mutations:
        with pytest.raises(ValueError):
            validate_checkpoint_route(payload | mutation, CANDIDATE)
    for field, value in (("static_key_value_tied", True), ("lmm_topk", 3),
                         ("static_retrieval_temperature", 2.), ("additional_parameter_count", 0),
                         ("time_memory_route", "local"), ("time_head", {"mode": "scaled_exact"})):
        bad = copy.deepcopy(payload)
        bad["encoder_config"][field] = value
        with pytest.raises(ValueError):
            validate_checkpoint_route(bad, CANDIDATE)
    for backbone in ("titantpp", WEIGHTED, "thp"):
        with pytest.raises(ValueError):
            validate_checkpoint_route(payload, backbone)
    for backbone in ("titantpp", WEIGHTED):
        old, _ = build(backbone)
        with pytest.raises(RuntimeError):
            model.load_state_dict(old.state_dict(), strict=True)
        with pytest.raises(RuntimeError):
            old.load_state_dict(model.state_dict(), strict=True)
    assert metadata["backbone_contract_id"] == KEY_VALUE_CONTRACT


@pytest.mark.parametrize("override", [
    {"quantity_variant": "count_only_lognormal_k1"},
    {"time_head_mode": "scaled_exact_rmtpp"}, {"lambda_tail": .1},
])
def test_factory_rejects_objective_drift(override):
    with pytest.raises(ValueError, match="Separate-key"):
        build_count_aware_model(CANDIDATE, hidden_dim=16, train_log_mean=1.5, max_seq_len=8, **override)
