"""The local-time ablation must change routing, not the quantity computation."""

import copy
import io
import os
from unittest.mock import patch

import pytest
import torch

from models.TPPs.CountAwareFactory import (
    HARD_LOCAL_TIME_BACKBONE as CANDIDATE,
    build_count_aware_model,
    validate_checkpoint_route,
)
from models.Titan.common.memory import HardLocalMemoryMatcher
from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLE_HARD_LOCAL_TIME, VARIANT, validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer

DEVICE = os.environ.get("HARD_LOCAL_TIME_TEST_DEVICE", "cpu")
TOL = dict(rtol=1e-6, atol=1e-6) if DEVICE == "cuda" else dict(rtol=0, atol=0)


def build(backbone=CANDIDATE, hidden=16, length=8):
    torch.manual_seed(42)
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=hidden, train_log_mean=1.5, max_seq_len=length,
        quantity_variant=VARIANT, lambda_tail=0., time_head_mode="legacy_clamped_rmtpp",
    )
    return model.to(DEVICE), metadata


def batch():
    dts = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 0.]], device=DEVICE)
    qty = torch.tensor([[1., 5., 9., 20.], [2., 6., 12., 0.]], device=DEVICE)
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]], device=DEVICE)
    return dts, qty, mask


def nonconstant_head(model):
    with torch.no_grad():
        model.quantity_head.weight.copy_(torch.linspace(
            -.2, .2, model.hidden_dim, device=DEVICE).reshape(1, -1))


@pytest.mark.parametrize("length,count", [(256, 89795), (84, 78787), (64, 77507)])
def test_parameters_initialization_rng_and_legacy_metadata(length, count):
    old, old_meta = build("titantpp", hidden=64, length=length)
    old_rng = torch.get_rng_state().clone()
    old_cuda_rng = torch.cuda.get_rng_state().clone() if DEVICE == "cuda" else None
    new, meta = build(hidden=64, length=length)
    assert torch.equal(old_rng, torch.get_rng_state())
    if old_cuda_rng is not None:
        assert torch.equal(old_cuda_rng, torch.cuda.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert sum(p.numel() for p in new.parameters()) == count
    assert all(p.requires_grad for p in new.parameters())
    assert type(old.lmm) is type(new.lmm) is HardLocalMemoryMatcher
    assert old_meta["backbone_contract_id"] == "B0"
    assert "routing_contract_id" not in old_meta
    assert old_meta["time_memory_route"] == "hard_local_memory_matcher"
    assert meta["time_memory_route"] == "local_encoder_with_persistent_tokens"
    assert meta["quantity_memory_route"] == "hard_local_memory_matcher"
    assert meta["additional_parameter_count"] == 0
    for key in ("persistent_mem_size", "lmm_mem_size", "lmm_topk", "n_layers", "n_heads", "d_model"):
        assert old_meta[key] == meta[key]


@pytest.mark.parametrize("training", [False, True])
def test_quantity_identity_including_nondegenerate_gradients(training):
    old, _ = build("titantpp")
    new, _ = build()
    nonconstant_head(old)
    new.load_state_dict(old.state_dict(), strict=True)
    dts, qty, mask = batch()
    states, outputs, gradients = [], [], []
    for model in (old, new):
        model.train(training)
        torch.manual_seed(123)
        states.append(model.encode_task_states(dts, qty, mask)[1])
        torch.manual_seed(123)
        out = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
        outputs.append(out)
        out["quantity_train_loss"].mean().backward()
        gradients.append({name: None if p.grad is None else p.grad.clone()
                          for name, p in model.named_parameters()})
        assert model.lmm.mem.grad.abs().sum() > 0
    torch.testing.assert_close(states[0], states[1], **TOL)
    for key in ("pred_qty", "quantity_train_loss", "log_qty_loss"):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], **TOL)
    for name, value in gradients[0].items():
        actual = gradients[1][name]
        if value is None:
            assert actual is None, name
        else:
            torch.testing.assert_close(value, actual, **TOL)


def test_single_base_call_exact_routes_and_masking():
    model, _ = build()
    model.eval()
    dts, qty, mask = batch()
    base = model._encode_base(dts, qty, mask)
    with patch.object(model, "_encode_base", wraps=model._encode_base) as encode:
        time, quantity = model.encode_task_states(dts, qty, mask)
        assert encode.call_count == 1
    torch.testing.assert_close(time, base, **TOL)
    torch.testing.assert_close(quantity, model.lmm(base) * mask.unsqueeze(-1), **TOL)
    torch.testing.assert_close(model.encode(dts, qty, mask), time, **TOL)
    assert time[1, 3].count_nonzero() == quantity[1, 3].count_nonzero() == 0


@pytest.mark.parametrize("task", ["time_loss", "quantity_train_loss"])
def test_task_gradients_and_persistent_memory_are_live(task):
    model, _ = build()
    model.eval()
    nonconstant_head(model)
    dts, qty, mask = batch()
    target_outputs(model, dts, mask, qty, lambda_log_qty=1.)[task].mean().backward()
    gradient = model.lmm.mem.grad
    if task == "time_loss":
        assert gradient is None or gradient.count_nonzero() == 0
    else:
        assert gradient is not None and gradient.abs().sum() > 0
    persistent = [p for name, p in model.named_parameters() if "persistent_mem" in name]
    assert len(persistent) == 2
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in persistent)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if name.startswith("encoder."))


def test_prototype_intervention_changes_only_quantity():
    model, _ = build()
    model.eval()
    nonconstant_head(model)
    dts, qty, mask = batch()
    before = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    with torch.no_grad():
        model.lmm.mem.add_(torch.linspace(-2, 2, model.hidden_dim, device=DEVICE))
    after = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    torch.testing.assert_close(before["time_loss"], after["time_loss"], **TOL)
    assert not torch.allclose(before["pred_qty"], after["pred_qty"])


def test_target_future_padding_and_series_isolation():
    model, _ = build()
    model.eval()
    nonconstant_head(model)
    original_parameters = {k: v.clone() for k, v in model.state_dict().items()}
    dts, qty, mask = batch()
    changed_dts, changed_qty = dts.clone(), qty.clone()
    changed_dts[0, 3], changed_dts[1, 2:] = 60000., 50000.
    changed_qty[0, 3], changed_qty[1, 2:] = 90000., 80000.
    before = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    with patch.object(model, "encode_task_states", wraps=model.encode_task_states) as encode:
        after = target_outputs(model, changed_dts, mask, changed_qty, lambda_log_qty=1.)
        inputs = encode.call_args
        assert inputs.args[1][0, 3] == inputs.args[1][1, 2] == 0
        assert not inputs.kwargs["memory_write_mask"][0, 3]
        assert not inputs.kwargs["memory_write_mask"][1, 2]
    torch.testing.assert_close(before["pred_qty"], after["pred_qty"], **TOL)
    states = model.encode_task_states(dts, qty, mask)
    altered = model.encode_task_states(changed_dts, changed_qty, mask)
    for actual, other in zip(states, altered):
        torch.testing.assert_close(actual[:, :2], other[:, :2], **TOL)
    # Changing the evaluated duration changes NLL; compare density at a fixed query.
    fixed_dt = torch.ones(2, device=DEVICE)
    torch.testing.assert_close(model.log_f_dt(states[0][:, 1], fixed_dt),
                               model.log_f_dt(altered[0][:, 1], fixed_dt), **TOL)
    for row in range(2):
        isolated = model.encode_task_states(dts[row:row+1], qty[row:row+1], mask[row:row+1])
        for actual, separate in zip(states, isolated):
            torch.testing.assert_close(actual[row:row+1], separate, rtol=1e-5, atol=1e-6)
    reordered = model.encode_task_states(dts.flip(0), qty.flip(0), mask.flip(0))
    for actual, reverse in zip(states, reordered):
        torch.testing.assert_close(actual, reverse.flip(0), rtol=1e-5, atol=1e-6)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_parameters[name]), name


@pytest.mark.parametrize("extreme", [False, True])
def test_finite_adamw_steps_and_strict_checkpoint_replay(extreme):
    model, meta = build()
    nonconstant_head(model)
    dts, qty, mask = batch()
    if extreme:
        dts, qty = dts * 1e8, qty * 1e8
    optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
    assert isinstance(optimizer, torch.optim.AdamW) and len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == .01
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
        assert all(torch.isfinite(v).all() for v in outputs.values())
        outputs["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval()
    original = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    payload = {"backbone": CANDIDATE, "variant": VARIANT, "encoder_config": meta,
               "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored_payload = torch.load(stream, map_location=DEVICE, weights_only=True)
    validate_checkpoint_route(restored_payload, CANDIDATE)
    restored, _ = build()
    restored.load_state_dict(restored_payload["model_state_dict"], strict=True)
    restored_optimizer = build_optimizer(restored, lr=.001, time_head_lr_multiplier=1.)
    restored_optimizer.load_state_dict(restored_payload["optimizer_state_dict"])
    assert len(restored_optimizer.state) == len(optimizer.state)
    restored.eval()
    replay = target_outputs(restored, dts, mask, qty, lambda_log_qty=1.)
    for key in original:
        torch.testing.assert_close(original[key], replay[key], **TOL)


def test_checkpoint_route_rejects_wrong_or_missing_identity():
    _, meta = build()
    valid = {"backbone": CANDIDATE, "variant": VARIANT, "encoder_config": meta}
    validate_checkpoint_route(valid, CANDIDATE)
    for update in ({"backbone": "titantpp"}, {"variant": "tail_shared"},
                   {"encoder_config": {}}, {"encoder_config": None}):
        with pytest.raises(ValueError):
            validate_checkpoint_route(valid | update, CANDIDATE)
    for key, value in {"time_memory_route": "hard_local_memory_matcher", "lmm_topk": 8,
                       "quantity_memory_route": "local_only", "routing_contract_id": None,
                       "time_head": None}.items():
        changed = copy.deepcopy(valid)
        changed["encoder_config"][key] = value
        with pytest.raises(ValueError):
            validate_checkpoint_route(changed, CANDIDATE)
    with pytest.raises(ValueError):
        validate_checkpoint_route(valid, "titantpp")
    validate_checkpoint_route({"backbone": "titantpp"}, "titantpp")


def test_factory_and_role_fail_closed():
    kwargs = dict(model_role=MODEL_ROLE_HARD_LOCAL_TIME, backbones=(CANDIDATE,),
                  quantity_variants=(VARIANT,), time_head_mode="legacy_clamped_rmtpp", lambda_tail=0.)
    validate_model_role_contract(**kwargs)
    for change in ({"model_role": "experimental"}, {"backbones": ("titantpp",)},
                   {"backbones": (CANDIDATE, "titantpp")}, {"quantity_variants": ("tail_shared",)},
                   {"time_head_mode": "scaled_exact"}, {"lambda_tail": .1}):
        with pytest.raises(ValueError):
            validate_model_role_contract(**(kwargs | change))
    base = dict(hidden_dim=16, train_log_mean=1.5, max_seq_len=8, quantity_variant=VARIANT,
                time_head_mode="legacy_clamped_rmtpp", lambda_tail=0.)
    for change in ({"quantity_variant": "tail_shared"}, {"lambda_tail": .1},
                   {"time_head_mode": "scaled_exact"}):
        with pytest.raises(ValueError):
            build_count_aware_model(CANDIDATE, **(base | change))
