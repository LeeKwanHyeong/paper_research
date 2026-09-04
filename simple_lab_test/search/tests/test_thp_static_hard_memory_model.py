"""Executable local contracts for THP plus the original static hard bank."""

import copy
import hashlib
import io
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.TPPs.CountAwareTHPStaticMemory import (
    CountAwareTHPStaticMemory,
    THP_STATIC_MEMORY_BACKBONE as CANDIDATE,
    THP_STATIC_MEMORY_ROLE as ROLE,
)
from models.TPPs.CountAwareTPP import CountAwareTHP
from models.Titan.common.memory import HardLocalMemoryMatcher
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONES, SUPPORTED_BACKBONES, VARIANT, validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer


ROOT = Path(__file__).resolve().parents[3]
DEVICE = os.environ.get("THP_STATIC_MEMORY_TEST_DEVICE", "cpu")
TOL = dict(rtol=1e-5, atol=1e-6) if DEVICE == "cuda" else dict(rtol=0., atol=0.)
FROZEN_FACTORY_SHA256 = "0937ccf3219f5c75f5d8ec01cee6a5716e2472cdf546734029f021f9bfc8c588"


def frozen_factory_source():
    # Isolated deployment snapshots have no Git database. The exported reference
    # must still be byte-identical to the pre-implementation contract revision.
    exported = os.environ.get("THP_STATIC_MEMORY_REFERENCE_FACTORY")
    source = Path(exported).read_bytes() if exported else subprocess.check_output(
        ["git", "show", "7f0bf8d:models/TPPs/CountAwareFactory.py"], cwd=ROOT)
    if hashlib.sha256(source).hexdigest() != FROZEN_FACTORY_SHA256:
        raise ValueError("Frozen reference factory checksum mismatch")
    return source.decode()


def build(backbone=CANDIDATE, length=84):
    torch.manual_seed(42)
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=64, train_log_mean=1.5, max_seq_len=length,
        quantity_variant=VARIANT, lambda_tail=0., time_head_mode="legacy_clamped_rmtpp",
    )
    return model.to(DEVICE), metadata


def batch():
    dts = torch.tensor([[1., 2., 3., 4.], [1., 2., 0., 0.]], device=DEVICE)
    qty = torch.tensor([[1., 5., 9., 20.], [2., 6., 0., 0.]], device=DEVICE)
    return dts, qty, dts != 0


def nonconstant_heads(model):
    with torch.no_grad():
        model.quantity_head.weight.copy_(torch.linspace(-.2, .2, 64, device=DEVICE)[None])
        model.v_t.weight.copy_(torch.linspace(.1, -.1, 64, device=DEVICE)[None])


def payload_for(model, metadata):
    return {"backbone": CANDIDATE, "variant": VARIANT, "encoder_config": metadata,
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "model_state_dict": model.state_dict()}


@pytest.mark.parametrize("length", [84, 256])
def test_original_thp_parameters_rng_and_exactly_one_bank(length):
    reference, reference_meta = build("thp", length)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if DEVICE == "cuda" else None
    candidate, meta = build(length=length)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert isinstance(candidate, CountAwareTHP) and type(candidate.lmm) is HardLocalMemoryMatcher
    assert set(candidate.state_dict()) - set(reference.state_dict()) == {"lmm.mem"}
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    assert sum(p.numel() for p in candidate.parameters()) == 104387
    assert candidate.lmm.mem.numel() == 4096 and candidate.lmm.topk == 4
    assert all(p.requires_grad for p in candidate.parameters())
    assert not any("persistent" in name for name, _ in candidate.named_parameters())
    assert not list(candidate.named_buffers())
    for key, value in reference_meta.items():
        if key != "candidate_name":
            assert meta[key] == value


@pytest.mark.parametrize("training", [False, True])
def test_zero_bank_exact_thp_states_predictions_and_shared_gradients(training):
    reference, _ = build("thp")
    candidate, _ = build()
    for model in (reference, candidate):
        nonconstant_heads(model)
        model.train(training)
    with torch.no_grad():
        candidate.lmm.mem.zero_()
    dts, qty, mask = batch()
    states, outputs, gradients = [], [], []
    for model in (reference, candidate):
        torch.manual_seed(123)
        states.append(model.encode_task_states(dts, qty, mask))
        torch.manual_seed(123)
        out = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
        outputs.append(out)
        out["joint_loss"].mean().backward()
        gradients.append({name: None if p.grad is None else p.grad.clone()
                          for name, p in model.named_parameters()})
    for before, after in zip(states[0], states[1]):
        torch.testing.assert_close(before, after, **TOL)
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], **TOL)
    for name, gradient in gradients[0].items():
        if gradient is None:
            assert gradients[1][name] is None
        else:
            torch.testing.assert_close(gradient, gradients[1][name], **TOL)


def test_single_encoder_matcher_call_and_shared_masked_state():
    model, _ = build()
    model.eval()
    dts, qty, mask = batch()
    original_encode = CountAwareTHP.encode
    with patch.object(CountAwareTHP, "encode", autospec=True, side_effect=original_encode) as encoder:
        with patch.object(model.lmm, "forward", wraps=model.lmm.forward) as matcher:
            time, quantity = model.encode_task_states(dts, qty, mask)
    assert encoder.call_count == matcher.call_count == 1
    assert time is quantity and time.requires_grad
    base = original_encode(model, dts, qty, mask)
    torch.testing.assert_close(time, model.lmm(base) * mask[..., None], **TOL)
    torch.testing.assert_close(model.encode(dts, qty, mask), time, **TOL)
    assert time[~mask].count_nonzero() == 0


def test_exact_cosine_top4_raw_value_mean_and_selected_value_gradients():
    model, _ = build()
    model.eval()
    dts, qty, mask = batch()
    base = CountAwareTHP.encode(model, dts, qty, mask)
    bank = model.lmm.mem
    scores = torch.matmul(F.normalize(base, p=2, dim=-1), F.normalize(bank, p=2, dim=-1).transpose(-1, -2))
    indices = scores.topk(4, dim=-1).indices
    residual = bank[0][indices].mean(dim=-2)
    torch.testing.assert_close(model.encode(dts, qty, mask), (base + residual) * mask[..., None], **TOL)
    assert residual.abs().sum() > 0
    residual[mask].sum().backward()
    selected = indices[mask].unique()
    active = bank.grad[0].abs().sum(dim=-1).nonzero().flatten()
    assert torch.equal(selected.sort().values, active)


@pytest.mark.parametrize("task", ["time_loss", "quantity_train_loss"])
def test_both_tasks_reach_original_encoder_and_selected_bank(task):
    model, _ = build()
    model.eval()
    nonconstant_heads(model)
    dts, qty, mask = batch()
    target_outputs(model, dts, mask, qty, lambda_log_qty=1.)[task].mean().backward()
    assert model.lmm.mem.grad is not None and model.lmm.mem.grad.abs().sum() > 0
    assert model.input_projection.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layers.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_nonzero_memory_intervention_changes_both_outputs_not_only_quantity():
    model, _ = build()
    model.eval()
    nonconstant_heads(model)
    dts, qty, mask = batch()
    before = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    with torch.no_grad():
        model.lmm.mem.add_(torch.linspace(-.5, .5, 64, device=DEVICE))
    after = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    assert not torch.allclose(before["pred_qty"], after["pred_qty"])
    assert not torch.allclose(before["time_loss"], after["time_loss"])


def test_target_future_padding_perturbations_and_no_cross_series_state():
    model, _ = build()
    model.eval()
    nonconstant_heads(model)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    dts, qty, mask = batch()
    changed_dts, changed_qty = dts.clone(), qty.clone()
    changed_dts[0, 3], changed_dts[1, 1:] = 6e7, 5e7
    changed_qty[0, 3], changed_qty[1, 1:] = 9e7, 8e7
    before = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    with patch.object(model, "encode_task_states", wraps=model.encode_task_states) as encode:
        after = target_outputs(model, changed_dts, mask, changed_qty, lambda_log_qty=1.)
        assert encode.call_args.args[1][0, 3] == encode.call_args.args[1][1, 1] == 0
        assert not encode.call_args.kwargs["memory_write_mask"][0, 3]
        assert not encode.call_args.kwargs["memory_write_mask"][1, 1]
    torch.testing.assert_close(before["pred_qty"], after["pred_qty"], **TOL)
    base = model.encode(dts, qty, mask)
    changed = model.encode(changed_dts, changed_qty, mask)
    for row, end in ((0, 3), (1, 1)):
        torch.testing.assert_close(base[row, :end], changed[row, :end], **TOL)
        fixed_dt = torch.ones(end, device=DEVICE)
        torch.testing.assert_close(model.log_f_dt(base[row, :end], fixed_dt),
                                   model.log_f_dt(changed[row, :end], fixed_dt), **TOL)
        separate = model.encode(dts[row:row+1], qty[row:row+1], mask[row:row+1])
        torch.testing.assert_close(base[row:row+1], separate, rtol=1e-5, atol=1e-6)
    reverse = model.encode(dts.flip(0), qty.flip(0), mask.flip(0))
    torch.testing.assert_close(base, reverse.flip(0), rtol=1e-5, atol=1e-6)
    for key, value in state.items():
        assert torch.equal(value, model.state_dict()[key])
    assert not list(model.buffers())


def test_all_padding_is_zero_and_prediction_requires_observed_history():
    model, _ = build()
    model.eval()
    empty = torch.zeros(2, 4, device=DEVICE)
    encoded = model.encode(empty, empty, empty.bool())
    assert torch.isfinite(encoded).all() and encoded.count_nonzero() == 0
    with pytest.raises(ValueError, match="history event"):
        target_outputs(model, empty, empty.bool(), empty, lambda_log_qty=1.)
    mask = empty.bool()
    mask[:, -1] = True
    with pytest.raises(ValueError, match="history event"):
        target_outputs(model, empty, mask, empty, lambda_log_qty=1.)


@pytest.mark.parametrize("extreme", [False, True])
def test_finite_synthetic_optimizer_steps_and_full_checkpoint_replay(extreme):
    model, metadata = build()
    nonconstant_heads(model)
    dts, qty, mask = batch()
    if extreme:
        dts, qty = dts * 1e8, qty * 1e8
    optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
    assert isinstance(optimizer, torch.optim.AdamW) and len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == .01
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
        assert all(torch.isfinite(value).all() for value in out.values())
        out["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(value).all() for value in model.parameters())
    model.eval()
    before = target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
    stream = io.BytesIO()
    torch.save(payload_for(model, metadata) | {"optimizer_state_dict": optimizer.state_dict()}, stream)
    stream.seek(0)
    restored_payload = torch.load(stream, map_location=DEVICE, weights_only=True)
    validate_checkpoint_route(restored_payload, CANDIDATE)
    restored, _ = build()
    restored.load_state_dict(restored_payload["model_state_dict"], strict=True)
    restored.eval()
    restored_optimizer = build_optimizer(restored, lr=.001, time_head_lr_multiplier=1.)
    restored_optimizer.load_state_dict(restored_payload["optimizer_state_dict"])
    assert len(restored_optimizer.state) == len(optimizer.state)
    after = target_outputs(restored, dts, mask, qty, lambda_log_qty=1.)
    for key in before:
        torch.testing.assert_close(before[key], after[key], **TOL)
    # Optimizer state must reproduce the next update, not just its dictionary size.
    for net, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        target_outputs(net, dts, mask, qty, lambda_log_qty=1.)["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], **TOL)


@pytest.mark.parametrize("field,value", [
    ("routing_contract_id", "wrong"), ("routing_contract_sha256", "wrong"),
    ("time_memory_route", "local_only"), ("quantity_memory_route", "local_only"),
    ("lmm_topk", 8), ("persistent_mem_size", 16), ("d_inner", 128),
    ("normalize_before", True), ("model_role", "t0_common_control"), ("time_head", None),
])
def test_checkpoint_metadata_drift_is_rejected(field, value):
    model, metadata = build()
    payload = payload_for(model, copy.deepcopy(metadata))
    payload["encoder_config"][field] = value
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, CANDIDATE)


def test_checkpoint_identity_scope_and_tensor_payload_fail_closed():
    model, metadata = build()
    valid = payload_for(model, metadata)
    validate_checkpoint_route(valid, CANDIDATE)
    for update in ({"backbone": "thp"}, {"variant": "tail_shared"}, {"encoder_config": {}},
                   {"evaluation_scope": "test"}, {"held_out_test_evaluated": True},
                   {"model_state_dict": {}}, {"model_state_dict": {"lmm.mem": torch.zeros(1, 64, 32)}},
                   {"model_state_dict": {"lmm.mem": torch.full((1, 64, 64), float("nan"))}}):
        with pytest.raises(ValueError):
            validate_checkpoint_route(valid | update, CANDIDATE)
    for other in ("thp", "titantpp", "titantpp_hard_memory_local_time"):
        with pytest.raises(ValueError):
            validate_checkpoint_route(valid, other)
    state = dict(model.state_dict())
    state.pop("input_projection.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True)
    reference, _ = build("thp")
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(reference.state_dict(), strict=True)


@pytest.mark.parametrize("update", [{"hidden_dim": 32}, {"quantity_variant": "count_only_lognormal_k1"},
    {"lambda_tail": .1}, {"time_head_mode": "scaled_exact_rmtpp"}, {"time_scale": 1.},
    {"time_w_max": .3}, {"time_initial_intercept": 1.}, {"time_wd_safety_limit": 8.}])
def test_model_cannot_silently_drift_from_frozen_architecture_and_heads(update):
    with pytest.raises(ValueError):
        CountAwareTHPStaticMemory(**(dict(hidden_dim=64, train_log_mean=1.5) | update))


def test_role_requires_only_the_new_candidate_not_a_default_benchmark():
    assert CANDIDATE in SUPPORTED_BACKBONES and CANDIDATE not in BACKBONES
    kwargs = dict(model_role=ROLE, backbones=(CANDIDATE,), quantity_variants=(VARIANT,),
                  time_head_mode="legacy_clamped_rmtpp", lambda_tail=0.)
    validate_model_role_contract(**kwargs)
    for update in ({"model_role": "experimental"}, {"model_role": "t0_common_control"},
                   {"backbones": ("thp",)}, {"backbones": (CANDIDATE, "thp")},
                   {"lambda_tail": .1}, {"quantity_variants": ("tail_shared",)},
                   {"time_head_mode": "scaled_exact_rmtpp"}):
        with pytest.raises(ValueError):
            validate_model_role_contract(**(kwargs | update))


@pytest.mark.parametrize("backbone", ["thp", "titantpp"])
def test_legacy_factory_metadata_parameters_and_outputs_unchanged(backbone):
    old_source = frozen_factory_source()
    namespace = {"__name__": "frozen_reference_factory"}
    exec(compile(old_source, "frozen_reference_factory", "exec"), namespace)
    torch.manual_seed(42)
    old, old_meta = namespace["build_count_aware_model"](
        backbone, hidden_dim=64, train_log_mean=1.5, max_seq_len=84)
    current, meta = build(backbone)
    old = old.to(DEVICE).eval()
    current.eval()
    assert old_meta == meta
    for key, value in old.state_dict().items():
        assert torch.equal(value, current.state_dict()[key])
    dts, qty, mask = batch()
    torch.testing.assert_close(old.encode(dts, qty, mask), current.encode(dts, qty, mask), **TOL)
