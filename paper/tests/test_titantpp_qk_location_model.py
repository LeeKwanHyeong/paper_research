"""Synthetic-only tests: never load a scientific dataset or trained CNN."""
from copy import deepcopy
import random
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
import torch

from models.TPPs import CountAwareTitanQKLocationCNN as candidate
from models.TPPs.CountAwareTitanCNNGRU import observed_context, causal_depthwise_residual
from paper.scripts import run_titantpp_qk_location_cnn as route
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def forbid_real_data():
    def no(*args, **kwargs):
        raise AssertionError("No real parquet inputs")
    with patch.object(pl, "read_parquet", no), patch.object(pl, "scan_parquet", no):
        yield


def build(arm=candidate.Q_ONLY, seed=42, context_length=None):
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
        for name in ("q", "k", "v"):
            if hasattr(attention, "causal_" + name + "_kernel"):
                getattr(attention, "causal_" + name + "_kernel").normal_(0, .1)
        model.quantity_head.weight.normal_(0, .1)
        model.time_scale_weight.weight.normal_(0, .01)
    model.eval()


def payload(arm=candidate.Q_ONLY, context_length=None):
    model, meta = build(arm, context_length=context_length)
    interface = {"time_head": meta["time_head"]}
    return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "variant": route.VARIANT, "interface_meta": interface,
        "resume_identity": {"backbone": arm, "interface_meta": interface,
            "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": route.role_for_arm(arm)}}}


@pytest.mark.parametrize("arm", candidate.ARMS)
@pytest.mark.parametrize("seed", (42, 52, 62))
@pytest.mark.parametrize("context_length", (32, 84, 256))
def test_common_initial_tensors_rng_physical_counts_and_output(arm, seed, context_length):
    data = route.fabricated_data(); data["loader"]["max_seq_len"] = context_length
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    baseline, _ = route.original.build_model(data, candidate.LEGACY_CNN)
    rng = torch.get_rng_state().clone(); py_rng = random.getstate(); np_rng = np.random.get_state()
    random.seed(seed); np.random.seed(seed)
    model, meta = build(arm, seed=seed, context_length=context_length)
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py_rng
    actual_np = np.random.get_state()
    assert actual_np[0] == np_rng[0] and np.array_equal(actual_np[1], np_rng[1]) and actual_np[2:] == np_rng[2:]
    state = model.state_dict()
    assert all(torch.equal(value, state[name]) for name, value in baseline.state_dict().items() if name not in candidate.REMOVED_KEYS_BY_ARM[arm])
    assert not set(candidate.REMOVED_KEYS_BY_ARM[arm]) & set(state)
    assert set(state) == (set(baseline.state_dict()) - set(candidate.REMOVED_KEYS_BY_ARM[arm])) | {candidate.IDENTITY_BUFFER}
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters()) - 384
    assert sum(p.numel() for p in model.parameters()) == meta["registered_parameter_count"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == meta["active_trainable_parameter_count"]
    assert not any(p.requires_grad for p in model.multilag_detail.parameters())
    attention = model.encoder.layers[0].attn
    assert attention.qkv.weight.requires_grad
    assert getattr(attention, "causal_" + candidate.LOCATION_BY_ARM[arm] + "_kernel").requires_grad
    assert sum(p.numel() for n, p in attention.named_parameters() if n.endswith("kernel")) == 192
    if context_length == 84:
        assert (meta["active_trainable_parameter_count"], meta["registered_parameter_count"]) == (79043, 103775)
    baseline.eval(); model.eval()
    dt, quantity, mask = batch()
    with torch.no_grad():
        a = core.target_outputs(baseline, dt, mask, quantity, lambda_log_qty=1.)
        b = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_qk_padding_skip_withheld_reset_and_nan_gradient_isolation():
    values = torch.randn(2, 8, 64)
    valid = torch.tensor([[True, True, False, True, True, False, True, True]] * 2)
    _, sources, available = observed_context(valid)
    kernel = torch.randn(3, 64, requires_grad=True)
    masked = values.clone(); masked[~valid] = float("nan"); masked.requires_grad_()
    actual = causal_depthwise_residual(masked, kernel, sources, available)
    keep = [0, 1, 3, 4, 6, 7]
    _, cs, ca = observed_context(valid[:, keep])
    compact = causal_depthwise_residual(values[:, keep], kernel, cs, ca)
    torch.testing.assert_close(actual[:, keep], compact, rtol=0, atol=0)
    assert torch.equal(actual[~valid], torch.zeros_like(actual[~valid]))
    actual.square().sum().backward()
    assert torch.isfinite(kernel.grad).all() and torch.isfinite(masked.grad).all()
    assert torch.equal(masked.grad[~valid], torch.zeros_like(masked.grad[~valid]))
    write = torch.ones_like(valid); write[:, 3] = False
    observed, rs, ra = observed_context(torch.ones_like(valid), memory_write_mask=write)
    values2 = values.clone(); values2[:, 3] = float("nan"); values2.requires_grad_()
    reset = causal_depthwise_residual(values2, kernel, rs, ra)
    _, fs, fa = observed_context(torch.ones_like(valid[:, 4:]))
    fresh = causal_depthwise_residual(values[:, 4:], kernel, fs, fa)
    assert torch.equal(reset[:, 4:], fresh)
    assert torch.equal(reset[:, 3], torch.zeros_like(reset[:, 3]))
    reset.square().sum().backward()
    assert torch.isfinite(values2.grad).all() and torch.isfinite(kernel.grad).all()
    assert torch.equal(values2.grad[:, 3], torch.zeros_like(values2.grad[:, 3]))


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_replacement_reuses_original_objects_and_deepcopy_is_independent(arm):
    base_model, _ = build(candidate.CNN)
    base = base_model.encoder.layers[0].attn
    state_before = deepcopy(base.state_dict()); rng = torch.get_rng_state().clone()
    replacement = candidate.QKLocationAttention(base, arm=arm)
    assert torch.equal(rng, torch.get_rng_state())
    for name in ('qkv', 'out_proj', 'causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel', 'persistent_mem', 'drop'):
        assert getattr(replacement, name) is getattr(base, name)
    assert not hasattr(replacement, 'causal_v_kernel') and not hasattr(replacement, 'value_difference_kernel')
    assert set(replacement.state_dict()) == set(state_before) - {k.rsplit('.',1)[1] for k in candidate.REMOVED_KEYS_BY_ARM[arm]}
    duplicated = deepcopy(replacement)
    for name, value in replacement.state_dict().items():
        assert torch.equal(value, duplicated.state_dict()[name])
    key='causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel'
    with torch.no_grad():getattr(duplicated,key).add_(1.)
    assert not torch.equal(getattr(duplicated,key), getattr(replacement,key))


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_qk_original_arithmetic_v_unmixed_and_memory_untouched(arm):
    model, _ = build(arm); activate(model)
    attention = model.encoder.layers[0].attn
    attention._ctx_mem = torch.randn(2, 3, 64)
    x = torch.randn(2, 7, 64)
    mask = torch.ones(2, 7, dtype=torch.bool)
    _, sources, available = observed_context(mask)
    projected = dict(zip(("q", "k", "v"), attention.qkv(x).chunk(3, dim=-1)))
    expected = projected.copy()
    name = candidate.LOCATION_BY_ARM[arm]
    expected[name] = projected[name] + causal_depthwise_residual(projected[name], getattr(attention, "causal_" + name + "_kernel"), sources, available)
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


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_retained_qk_and_common_value_projection_receive_gradients(arm):
    model, _ = build(arm); activate(model); model.train()
    attention = model.encoder.layers[0].attn
    assert set(n for n, _ in attention.named_parameters() if n.endswith('_kernel')) == {'causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel'}
    dt, quantity, mask = batch()
    before = {n:p.detach().clone() for n,p in attention.named_parameters()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01)
    loss = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)['joint_loss'].mean()
    loss.backward()
    for name in ('causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel',):
        grad = getattr(attention,name).grad
        assert grad.shape == (3,64) and torch.isfinite(grad).all() and bool((grad != 0).all())
    assert torch.isfinite(attention.qkv.weight.grad).all() and bool((attention.qkv.weight.grad[128:] != 0).any())
    assert all(p.grad is None for p in model.multilag_detail.parameters())
    optimizer.step()
    assert bool((before['qkv.weight'][128:] != attention.qkv.weight[128:]).any())
    assert all(bool((before[n] != getattr(attention,n)).any()) for n in ('causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel',))


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
    other = candidate.K_ONLY if arm == candidate.Q_ONLY else candidate.Q_ONLY
    with pytest.raises(ValueError):
        candidate.validate_checkpoint(data, other)
    bad = deepcopy(data); bad["resume_identity"]["checkpoint_monitor"] = "test_raw_quantity_rmse"
    with pytest.raises(ValueError):
        candidate.validate_checkpoint(bad, arm)
    if arm in candidate.ARMS:
        bad = deepcopy(data); bad["model_state_dict"][candidate.LOCAL_KEYS_BY_ARM[arm][0]][0, 0] = float("nan")
        with pytest.raises(ValueError, match="retained correction tensor"):
            candidate.validate_checkpoint(bad, arm)
        for mutation in ("old_v", "missing", "identity", "extra"):
            state = deepcopy(data["model_state_dict"])
            if mutation == "old_v": state[candidate.REMOVED_KEYS_BY_ARM[arm][0]] = torch.zeros(3, 64)
            elif mutation == "missing": del state[candidate.LOCAL_KEYS_BY_ARM[arm][0]]
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
    for arm in candidate.ARMS:
        model, _ = build(arm)
        common = {k: v for k, v in model.state_dict().items() if k != candidate.IDENTITY_BUFFER}
        assert canonical_state_dict_sha256(common) == route.common_reference_initial_state_sha(data, 42, arm=arm)
    assert route.initial_states(data, 42, arms=candidate.ARMS) == {arm:hashes[arm] for arm in candidate.ARMS}
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
    with pytest.raises(ValueError, match="Only the new Q/K-location"):
        route.training_args({}, route.fabricated_data(), tmp_path, candidate.CNN)


def test_synthetic_qualification_covers_all_192_active_coefficients():
    result = route.synthetic_check("cpu")
    assert result["status"] == "passed"
    assert result["active_correction_gradient_parameter_count"] == {candidate.CNN:576,candidate.Q_ONLY:192,candidate.K_ONLY:192}
    assert result["scientific_optimizer_updates"] == 0
    assert result["active_local_cnn_parameters"] == 192 and result["active_value_cnn_parameters"] == 0
    assert result["original_value_kernel_absent"] is True
    assert result["value_operation"] == "unmixed_common_value_projection"
    assert result["common_value_projection_gradient_verified"][candidate.Q_ONLY] is True


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_checkpoint_rejects_wrong_head_split_and_candidate_tensor_keys(arm):
    original = payload(arm, context_length=84)
    for key,value in [('held_out_test_evaluated',True),('evaluation_scope','test'),('variant','other')]:
        bad=deepcopy(original);bad[key]=value
        with pytest.raises(ValueError):candidate.validate_checkpoint(bad,arm)
    for key in (*candidate.REMOVED_KEYS_BY_ARM[arm], 'encoder.layers.0.attn.value_difference_kernel'):
        bad=deepcopy(original);bad['model_state_dict'][key]=torch.zeros(3,64)
        with pytest.raises(ValueError):candidate.validate_checkpoint(bad,arm)


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_attention_mask_and_event_bias_guard(arm):
    model,_=build(arm);attention=model.encoder.layers[0].attn
    x=torch.randn(2,4,64);valid=torch.ones(2,4,dtype=torch.bool);_,sources,available=observed_context(valid)
    for mask in (None,valid.float(),~valid):
        with pytest.raises(ValueError):attention(x,mask,observed_sources=sources,observed_available=available)
    with pytest.raises(ValueError):attention(x,valid,observed_sources=sources,observed_available=available,event_attention_bias=torch.zeros(2,4,4))


@pytest.mark.parametrize("arm", candidate.ARMS)
@pytest.mark.parametrize("memory_enabled", (False, True))
@pytest.mark.parametrize("history_case", ("full", "padding", "withheld"))
def test_original_attention_equivalence_for_nonzero_retained_kernel(arm, memory_enabled, history_case):
    """Synthetic reference keeps unwanted kernels at zero; no stored CNN is loaded."""
    base_model, _ = build(candidate.CNN)
    base = base_model.encoder.layers[0].attn
    base.eval()
    if memory_enabled:
        base._ctx_mem = torch.randn(2, 3, 64)
    else:
        base.persistent_mem = None
        base._ctx_mem = None
    retained = 'causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel'
    with torch.no_grad():
        getattr(base, retained).normal_(0, .2)
    chosen = candidate.QKLocationAttention(deepcopy(base), arm=arm)
    valid = torch.ones(2, 7, dtype=torch.bool)
    write = None
    if history_case == 'padding':valid[:, (0, 3)] = False
    if history_case == 'withheld':
        write = torch.ones_like(valid);write[:, 3] = False
    observed, sources, available = observed_context(valid, memory_write_mask=write)
    x = torch.randn(2, 7, 64);x[~observed] = 0
    bias = torch.randn(2, base.n_heads, 7, 7) * .03
    with torch.no_grad():
        expected = base(x, observed, observed_sources=sources, observed_available=available,
            event_attention_bias=bias, return_event_attention=True)
        actual = chosen(x, observed, observed_sources=sources, observed_available=available,
            event_attention_bias=bias, return_event_attention=True)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))


@pytest.mark.parametrize('arm', candidate.ARMS)
@pytest.mark.parametrize('length', (1, 2, 3))
def test_short_history_uses_only_available_original_taps(arm, length):
    model,_ = build(arm);attention = model.encoder.layers[0].attn;attention.eval()
    with torch.no_grad():getattr(attention,'causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel').fill_(.2)
    x = torch.randn(2, length, 64)
    valid = torch.ones(2, length, dtype=torch.bool)
    _,sources,available = observed_context(valid)
    output = attention(x,valid,observed_sources=sources,observed_available=available)
    assert bool(torch.isfinite(output).all())
    output.square().sum().backward()
    grad=getattr(attention,'causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel').grad
    assert bool(torch.isfinite(grad).all())
    assert bool((grad[:length] != 0).any())
    assert torch.equal(grad[length:], torch.zeros_like(grad[length:]))


def test_query_key_endpoint_asymmetry_changes_memory_competition():
    """Q modifies memory scores; K preserves memory scores but changes the denominator."""
    model,_=build(candidate.CNN);base=model.encoder.layers[0].attn;base.eval()
    torch.manual_seed(905)
    base._ctx_mem=torch.randn(1,3,64)
    x=torch.randn(1,5,64)
    valid=torch.ones(1,5,dtype=torch.bool);_,sources,available=observed_context(valid)
    memory_length=3 + base.persistent_mem.shape[1]
    softmax=candidate.F.softmax
    scores={};outputs={};weights={}
    for name in ('none',*candidate.ARMS):
        attention=deepcopy(base) if name=='none' else candidate.QKLocationAttention(deepcopy(base),arm=name)
        if name!='none':
            with torch.no_grad():getattr(attention,'causal_' + candidate.LOCATION_BY_ARM[name] + '_kernel').fill_(.4)
        captured=[]
        def capture(value, *args, **kwargs):
            captured.append(value.detach().clone())
            return softmax(value,*args,**kwargs)
        with torch.no_grad(),patch.object(candidate.F,'softmax',side_effect=capture):
            outputs[name]=attention(x,valid,observed_sources=sources,observed_available=available)
        assert len(captured)==1
        scores[name]=captured[0][...,-1,:]
        weights[name]=softmax(scores[name],dim=-1)
    q,k=candidate.ARMS
    assert not torch.equal(scores[q][...,:memory_length],scores['none'][...,:memory_length])
    assert torch.equal(scores[k][...,:memory_length],scores['none'][...,:memory_length])
    assert not torch.equal(scores[k][...,memory_length:],scores['none'][...,memory_length:])
    for name in candidate.ARMS:
        assert not torch.equal(weights[name][...,:memory_length].sum(-1),weights['none'][...,:memory_length].sum(-1))
        assert not torch.equal(outputs[name][:,-1],outputs['none'][:,-1])
    assert not torch.equal(outputs[q][:,-1],outputs[k][:,-1])


@pytest.mark.parametrize('arm',candidate.ARMS)
def test_same_seed_original_qkonly_to_single_location_preserves_state_and_rng(arm):
    from models.TPPs.CountAwareTitanQKOnlyCNN import QKOnlyAttention
    model,_=build(candidate.CNN)
    original=model.encoder.layers[0].attn
    rng=torch.get_rng_state().clone()
    joint=QKOnlyAttention(deepcopy(original))
    via_joint=candidate.QKLocationAttention(joint,arm=arm)
    direct=candidate.QKLocationAttention(deepcopy(original),arm=arm)
    assert torch.equal(rng,torch.get_rng_state())
    assert set(via_joint.state_dict())==set(direct.state_dict())
    assert all(torch.equal(v,direct.state_dict()[k]) for k,v in via_joint.state_dict().items())
    retained='causal_' + candidate.LOCATION_BY_ARM[arm] + '_kernel'
    assert getattr(via_joint,retained) is getattr(joint,retained)


@pytest.mark.parametrize('source_arm,target_arm',((candidate.Q_ONLY,candidate.K_ONLY),(candidate.K_ONLY,candidate.Q_ONLY)))
def test_relabelled_selected_and_terminal_states_cannot_cross_candidate_arms(source_arm,target_arm):
    source=payload(source_arm,context_length=84)
    source['best_state_dict']=deepcopy(source['model_state_dict'])
    forged=payload(target_arm,context_length=84)
    for state_name in ('model_state_dict','best_state_dict'):
        bad=deepcopy(forged)
        bad[state_name]=deepcopy(source[state_name])
        # Even replacing the identity cannot hide the wrong physical local kernel.
        bad[state_name][candidate.IDENTITY_BUFFER]=candidate.identity(arm=target_arm)
        with pytest.raises(ValueError,match='local filter'):
            candidate.validate_checkpoint(bad,target_arm)
    source_model,_=build(source_arm)
    target_model,_=build(target_arm)
    with pytest.raises(ValueError,match='identity'):
        target_model.load_state_dict(source_model.state_dict(),strict=False)
