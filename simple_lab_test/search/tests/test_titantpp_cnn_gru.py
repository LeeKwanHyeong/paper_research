"""Fabricated-input CNN/GRU contracts and shared-trainer Validation replay.

No research data, remote command, GPU experiment, or held-out result is read.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from models.TPPs.CountAwareTitanCNNGRU import (
    ARMS, BASELINE, ROLE_BY_ARM, GRUHistoryCorrection, ObservedCausalQKVMemoryAttention,
    causal_depthwise_residual, identity, metadata, observed_context,
    validate_checkpoint,
)
from models.TPPs.CountAwareTitanHistoryWidth import identity as width_identity
from paper.scripts import run_titantpp_cnn_gru as runner
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract, cpu_training


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def isolated_process_hooks():
    """Restore the shared route and role globals after every candidate test."""
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts import run_titantpp_history_width as width_runner
    from paper.scripts.count_aware_tpp_backbone import constants, observed_time, training
    changes = {
        factory: ("build_count_aware_model", "validate_checkpoint_route"),
        training: ("build_count_aware_model", "build_model", "validate_checkpoint_route", "OBSERVED_TIME_ROLES"),
        observed_time: ("ROLES", "validate_launch"),
        constants: ("BACKBONE_LABELS", "MODEL_ROLES", "SUPPORTED_BACKBONES", "validate_model_role_contract"),
        runner: ("_installed",), width_runner: ("_installed",),
    }
    snapshots = {(module, name): (value, dict(value) if isinstance(value, dict) else None)
        for module, names in changes.items() for name in names
        for value in (getattr(module, name),)}
    yield
    for (module, name), (value, contents) in snapshots.items():
        if contents is not None:
            value.clear(); value.update(contents)
        setattr(module, name, value)


def data(max_length=256):
    return {"model": {"hidden_dim": 64, "quantity_variant": runner.VARIANT,
        "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 7.,
        "time_initial_location": .2, "time_initial_scale": .8,
        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None}},
        "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": max_length}}


def payload(arm):
    model, meta = runner.build_model(data(), arm)
    interface = {"time_head": meta["time_head"]}
    return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "variant": runner.VARIANT, "interface_meta": interface,
        "resume_identity": {"backbone": arm, "interface_meta": interface,
            "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": ROLE_BY_ARM[arm]}}}


def masks():
    valid = torch.ones(2, 14, dtype=torch.bool)
    valid[0, [1, 4, 11]] = False
    observed = valid.clone()
    observed[0, 7] = False
    observed[1, [3, 10]] = False
    return valid, observed


def manual_context(valid, write):
    sources = torch.zeros((*valid.shape, 3), dtype=torch.long)
    available = torch.zeros_like(sources, dtype=torch.bool)
    for batch in range(valid.shape[0]):
        history = []
        for position in range(valid.shape[1]):
            if not valid[batch, position]:
                continue
            if not write[batch, position]:
                history = []
                continue
            history.append(position)
            for lag in range(min(3, len(history))):
                sources[batch, position, lag] = history[-1 - lag]
                available[batch, position, lag] = True
    return sources, available


def test_cnn_observed_event_context_skips_padding_and_resets_withheld():
    valid, write = masks()
    observed, sources, available = observed_context(valid, memory_write_mask=write)
    expected_sources, expected_available = manual_context(valid, write)
    assert torch.equal(observed, valid & write)
    assert torch.equal(sources, expected_sources)
    assert torch.equal(available, expected_available)
    assert available[0, 8].tolist() == [True, False, False]
    assert sources[0, 6].tolist() == [6, 5, 3]
    assert not available[~observed].any()


def test_cnn_convolution_matches_independent_reference_and_gradient():
    torch.manual_seed(8)
    valid, write = masks()
    observed, sources, available = observed_context(valid, memory_write_mask=write)
    values = torch.randn(2, 14, 64, requires_grad=True)
    kernel = torch.randn(3, 64, requires_grad=True)
    actual = causal_depthwise_residual(values, kernel, sources, available)
    rows = []
    for batch in range(2):
        output = []
        for position in range(14):
            correction = torch.zeros(64)
            for lag in range(3):
                if available[batch, position, lag]:
                    correction = correction + values[batch, sources[batch, position, lag]] * kernel[lag]
            output.append(correction)
        rows.append(torch.stack(output))
    reference = torch.stack(rows)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    gradients = torch.autograd.grad(actual.square().sum(), (values, kernel), retain_graph=True)
    expected_gradients = torch.autograd.grad(reference.square().sum(), (values, kernel))
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-6, atol=1e-5)
    assert not actual[~observed].any()


def test_cnn_future_nonobserved_and_cross_segment_values_cannot_leak():
    torch.manual_seed(9)
    valid, write = masks()
    observed, sources, available = observed_context(valid, memory_write_mask=write)
    values = torch.randn(2, 14, 64)
    kernel = torch.randn(3, 64)
    reference = causal_depthwise_residual(values, kernel, sources, available)
    future = values.clone(); future[:, 9:] += 5000
    changed = causal_depthwise_residual(future, kernel, sources, available)
    assert torch.equal(reference[:, :9], changed[:, :9])
    ignored = values.clone(); ignored[~observed] = float("nan")
    torch.testing.assert_close(reference, causal_depthwise_residual(ignored, kernel, sources, available))
    reset = values.clone(); reset[0, :7] += 1000
    changed = causal_depthwise_residual(reset, kernel, sources, available)
    assert torch.equal(reference[0, 8:], changed[0, 8:])
    for batch in range(2):
        individual = causal_depthwise_residual(values[batch:batch + 1], kernel,
            sources[batch:batch + 1], available[batch:batch + 1])
        assert torch.equal(reference[batch], individual[0])


@pytest.mark.parametrize("kind", ["dtype", "shape"])
def test_context_requires_explicit_boolean_same_shape_masks(kind):
    valid, write = masks()
    changed = write.float() if kind == "dtype" else write[:, :-1]
    with pytest.raises(ValueError):
        observed_context(valid, memory_write_mask=changed)


def test_cnn_filters_only_event_qkv_and_reuses_persistent_attention_parameters(monkeypatch):
    from models.Titan.common.memory import MemoryAttention
    from models.TPPs import CountAwareTitanCNNGRU as candidate
    torch.manual_seed(42)
    base = MemoryAttention(64, 4, 0, 16, dropout=0.)
    before = torch.get_rng_state().clone()
    layer = ObservedCausalQKVMemoryAttention(base)
    assert torch.equal(before, torch.get_rng_state())
    assert layer.persistent_mem is base.persistent_mem
    assert layer.qkv is base.qkv and layer.out_proj is base.out_proj
    inputs = torch.randn(2, 9, 64)
    mask = torch.ones(2, 9, dtype=torch.bool); mask[0, 4] = False
    _, sources, available = observed_context(mask)
    with torch.no_grad():
        initial = layer(inputs, mask, observed_sources=sources, observed_available=available)
        assert torch.equal(initial, base(inputs, mask))
    seen = []
    original = candidate.causal_depthwise_residual
    def capture(values, kernel, supplied_sources, supplied_available):
        seen.append((values.shape, supplied_sources.shape, supplied_available.shape))
        return original(values, kernel, supplied_sources, supplied_available)
    monkeypatch.setattr(candidate, "causal_depthwise_residual", capture)
    for name in ("causal_q_kernel", "causal_k_kernel", "causal_v_kernel"):
        torch.nn.init.normal_(getattr(layer, name), std=.05)
    with torch.no_grad():
        actual, event_attention = layer(inputs, mask, observed_sources=sources,
            observed_available=available, return_event_attention=True)
    assert seen == [(torch.Size([2, 9, 64]), torch.Size([2, 9, 3]), torch.Size([2, 9, 3]))] * 3
    assert event_attention.shape == (2, 9, 9)
    assert torch.isfinite(actual).all() and not actual[~mask].any()
    assert not event_attention[0, :, 4].any()
    # Persistent token keys and values still affect output, with no convolution call on them.
    memory = layer.persistent_mem.detach().clone()
    with torch.no_grad():
        layer.persistent_mem.add_(3.)
        changed = layer(inputs, mask, observed_sources=sources, observed_available=available)
        layer.persistent_mem.copy_(memory)
    assert not torch.equal(actual, changed)


def manual_gru(layer, hidden, valid, write):
    rows = []
    for batch in range(hidden.shape[0]):
        state = torch.zeros(1, 1, layer.gru.hidden_size)
        count = 0
        output = []
        for position in range(hidden.shape[1]):
            correction = torch.zeros(hidden.shape[-1])
            if valid[batch, position]:
                if write[batch, position]:
                    _, state = layer.gru(layer.input_projection(hidden[batch, position]).reshape(1, 1, -1), state)
                    count += 1
                    if count >= 2:
                        correction = layer.output_projection(state[0, 0])
                else:
                    state = torch.zeros_like(state); count = 0
            output.append(correction)
        rows.append(torch.stack(output))
    return torch.stack(rows)


def test_gru_manual_recurrence_padding_reset_first_event_and_gradients():
    torch.manual_seed(42)
    layer = GRUHistoryCorrection()
    assert sum(p.numel() for p in layer.parameters()) == 24732
    assert layer.gru.input_size == layer.gru.hidden_size == 54
    assert layer.gru.num_layers == 1 and not layer.gru.bidirectional
    assert layer.gru.bias and layer.gru.dropout == 0
    assert layer.input_projection.bias is layer.output_projection.bias is None
    hidden = torch.randn(2, 14, 64, requires_grad=True)
    valid, write = masks()
    assert torch.equal(layer(hidden, valid, memory_write_mask=write), torch.zeros_like(hidden))
    torch.nn.init.normal_(layer.output_projection.weight, std=.02)
    actual = layer(hidden, valid, memory_write_mask=write)
    reference = manual_gru(layer, hidden, valid, write)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(actual.sum(), hidden, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference.sum(), hidden)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=2e-6)
    assert not actual[~write].any()
    for batch, first in ((0, 0), (0, 8), (1, 0), (1, 4), (1, 11)):
        assert not actual[batch, first].any()


def test_gru_no_future_padding_cross_segment_or_sample_state_carry():
    torch.manual_seed(4)
    layer = GRUHistoryCorrection()
    torch.nn.init.normal_(layer.output_projection.weight, std=.02)
    valid, write = masks()
    hidden = torch.randn(2, 14, 64)
    original = layer(hidden, valid, memory_write_mask=write)
    changed = hidden.clone(); changed[:, 9:] += 5000
    torch.testing.assert_close(original[:, :9], layer(changed, valid, memory_write_mask=write)[:, :9])
    ignored = hidden.clone(); ignored[~write] = float("nan")
    torch.testing.assert_close(original, layer(ignored, valid, memory_write_mask=write))
    reset = hidden.clone(); reset[0, :7] += 5000
    torch.testing.assert_close(original[0, 8:], layer(reset, valid, memory_write_mask=write)[0, 8:])
    # A separate intervening call cannot persist hidden state into the next prefix.
    layer(torch.randn_like(hidden) * 10, valid, memory_write_mask=write)
    assert torch.equal(original, layer(hidden, valid, memory_write_mask=write))
    for batch in range(2):
        individual = layer(hidden[batch:batch + 1], valid[batch:batch + 1],
            memory_write_mask=write[batch:batch + 1])
        torch.testing.assert_close(original[batch], individual[0], rtol=2e-5, atol=2e-6)


def test_gru_zero_output_initialization_delays_inner_gradient_one_update():
    torch.manual_seed(52)
    before = torch.get_rng_state().clone()
    layer = GRUHistoryCorrection()
    assert torch.equal(before, torch.get_rng_state())
    hidden = torch.randn(2, 8, 64)
    valid = torch.ones(2, 8, dtype=torch.bool)
    target = torch.randn_like(hidden)
    optimizer = torch.optim.SGD(layer.parameters(), lr=.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (layer(hidden, valid) - target).square().mean()
        loss.backward()
        inner = [p.grad for name, p in layer.named_parameters() if not name.startswith("output_projection.")]
        assert all(g is not None and torch.isfinite(g).all() for g in inner)
        assert layer.output_projection.weight.grad.abs().sum() > 0
        if step == 0:
            assert all(not g.any() for g in inner)
        else:
            assert all(g.abs().sum() > 0 for g in inner)
        optimizer.step()


@pytest.mark.parametrize("seed", [42, 52, 62])
@pytest.mark.parametrize("max_length", [84, 256])
def test_actual_parameters_common_initial_tensors_rng_and_zero_initial_outputs(seed, max_length):
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    models, states, rngs, counts = {}, {}, {}, {}
    for arm in (BASELINE, *ARMS):
        torch.manual_seed(seed)
        models[arm], _ = runner.build_model(data(max_length), arm)
        states[arm] = models[arm].state_dict()
        rngs[arm] = torch.get_rng_state().clone()
        counts[arm] = sum(p.numel() for p in models[arm].parameters())
    expected_corrections = dict(zip(ARMS, (25152, 24732, 25308)))
    expected_baseline = 114435 - (256 - max_length) * 64
    assert counts[BASELINE] == expected_baseline
    for arm in ARMS:
        assert counts[arm] == expected_baseline + expected_corrections[arm] - 24576
        assert torch.equal(rngs[arm], rngs[BASELINE])
        for name, value in states[BASELINE].items():
            if name.startswith("multilag_detail.") or name.endswith("identity"):
                continue
            assert torch.equal(value, states[arm][name]), (arm, name)
    for name, value in states[ARMS[1]].items():
        if name.startswith("multilag_detail."):
            assert torch.equal(value, states[ARMS[2]][name])
    for arm in (ARMS[0], ARMS[2]):
        model = models[arm]
        kernels = {name: parameter for name, parameter in model.named_parameters() if "causal_" in name and name.endswith("_kernel")}
        assert len(kernels) == 3 and sum(p.numel() for p in kernels.values()) == 576
        assert all(not p.any() for p in kernels.values())
        assert all("encoder.layers.0." in name for name in kernels)
    dt = torch.randint(1, 8, (2, 12)).float()
    quantity = torch.randint(1, 20, dt.shape).float()
    valid = torch.ones_like(dt, dtype=torch.bool)
    with torch.no_grad():
        outputs = {arm: target_outputs(model.eval(), dt, valid, quantity, lambda_log_qty=1.)
            for arm, model in models.items()}
    for arm in ARMS:
        for key in outputs[BASELINE]:
            assert torch.equal(outputs[BASELINE][key], outputs[arm][key]), (arm, key)
    before = torch.get_rng_state().clone()
    initial = runner.initial_states(data(max_length), seed)
    assert torch.equal(before, torch.get_rng_state())
    assert len(set(initial.values())) == 4


def activate(model):
    for name, parameter in model.named_parameters():
        if name.endswith("_kernel") and "causal_" in name:
            torch.nn.init.normal_(parameter, std=.015)
    correction = model.multilag_detail
    if hasattr(correction, "output_projection"):
        torch.nn.init.normal_(correction.output_projection.weight, std=.02)
    else:
        for projection in correction.output_projections:
            torch.nn.init.normal_(projection.weight, std=.02)


@pytest.mark.parametrize("arm", [ARMS[0], ARMS[2]])
def test_zero_initialized_cnn_all_kernel_rows_receive_whole_model_gradient(arm):
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    torch.manual_seed(42)
    model, _ = runner.build_model(data(), arm)
    dt = torch.randint(1, 8, (2, 12)).float()
    quantity = torch.randint(1, 20, dt.shape).float()
    output = target_outputs(model, dt, torch.ones_like(dt, dtype=torch.bool), quantity, lambda_log_qty=1.)
    output["joint_loss"].sum().backward()
    kernels = [p for name, p in model.named_parameters() if name.endswith("_kernel") and "causal_" in name]
    assert len(kernels) == 3
    for parameter in kernels:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert (parameter.grad.abs().sum(dim=1) > 0).all()


@pytest.mark.parametrize("arm", ARMS)
def test_full_model_nonzero_modules_hide_target_and_padding(arm):
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    torch.manual_seed(62)
    model, _ = runner.build_model(data(), arm)
    activate(model); model.eval()
    dt = torch.randint(1, 8, (2, 12)).float()
    quantity = torch.randint(1, 20, dt.shape).float()
    valid = torch.ones_like(dt, dtype=torch.bool)
    with torch.no_grad():
        original = target_outputs(model, dt, valid, quantity, lambda_log_qty=1.)
        changed_qty = quantity.clone(); changed_qty[:, -1] = 4000
        changed_dt = dt.clone(); changed_dt[:, -1] = 999
        altered = target_outputs(model, changed_dt, valid, changed_qty, lambda_log_qty=1.)
        for key in ("pred_qty", "pred_log_qty", "pred_time_mu", "pred_time_sigma"):
            if key in original:
                assert torch.equal(original[key], altered[key]), key
        observed = valid.clone(); observed[:, -1] = False
        states = model.encode_task_states(dt, quantity, valid, memory_write_mask=observed)
        changed_states = model.encode_task_states(changed_dt, changed_qty, valid, memory_write_mask=observed)
        assert all(torch.equal(old, new) for old, new in zip(states, changed_states))
        future_dt = dt.clone(); future_dt[:, 8:11] += 400
        future_q = quantity.clone(); future_q[:, 8:11] += 400
        future_states = model.encode_task_states(future_dt, future_q, valid, memory_write_mask=observed)
        for old, new in zip(states, future_states):
            torch.testing.assert_close(old[:, :8], new[:, :8], rtol=1e-5, atol=1e-5)
        padded_dt = torch.cat((torch.full((2, 2), float("nan")), dt), dim=1)
        padded_q = torch.cat((torch.full((2, 2), float("nan")), quantity), dim=1)
        padded_valid = torch.cat((torch.zeros((2, 2), dtype=torch.bool), valid), dim=1)
        padded = target_outputs(model, padded_dt, padded_valid, padded_q, lambda_log_qty=1.)
        torch.testing.assert_close(original["pred_qty"], padded["pred_qty"], rtol=1e-5, atol=1e-5)
    assert not any("sources" in name or "available" in name for name in model.state_dict())


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("strict", [True, False])
def test_state_load_rejects_cross_arm_and_incomplete_state_before_mutation(arm, strict):
    model, _ = runner.build_model(data(), arm)
    before = deepcopy(model.state_dict())
    other = BASELINE if arm == ARMS[0] else ARMS[0]
    unrelated, _ = runner.build_model(data(), other)
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(unrelated.state_dict(), strict=strict)
    broken = deepcopy(before); broken["cnn_gru_identity"][0] ^= 1
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(broken, strict=strict)
    broken = deepcopy(before); broken.pop(next(k for k in broken if k.endswith("weight")))
    with pytest.raises((ValueError, RuntimeError)):
        model.load_state_dict(broken, strict=strict)
    for key, value in before.items():
        assert torch.equal(value, model.state_dict()[key])
    model.load_state_dict(before, strict=strict)


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("mutation", ["arm", "role", "head", "split", "monitor", "identity", "shape", "state_absent"])
def test_checkpoint_metadata_label_split_and_architecture_drift_rejected(arm, mutation):
    value = payload(arm)
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    assert validate_checkpoint(value, arm)
    validate_checkpoint_route(value, arm)
    for other in (BASELINE, *ARMS):
        if other != arm:
            with pytest.raises(ValueError):
                validate_checkpoint_route(value, other)
    if mutation == "arm": value["backbone"] = BASELINE
    elif mutation == "role": value["resume_identity"]["arguments"]["model_role"] = "observed_time_history_mlp_width16_v1"
    elif mutation == "head": value["encoder_config"]["time_head"]["mode"] = "legacy_clamped"
    elif mutation == "split": value["held_out_test_evaluated"] = True
    elif mutation == "monitor": value["resume_identity"]["checkpoint_monitor"] = "validation_time_nll"
    elif mutation == "identity": value["model_state_dict"]["cnn_gru_identity"][0] ^= 1
    elif mutation == "state_absent": value.pop("model_state_dict")
    else:
        key = next(k for k in value["model_state_dict"] if k.startswith("multilag_detail.") and k.endswith("weight"))
        value["model_state_dict"][key] = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        validate_checkpoint_route(value, arm)


@pytest.mark.parametrize("arm", ARMS)
def test_process_hooks_idempotent_and_frozen_role_loss_scope_remain(arm):
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import constants, observed_time, training
    runner.install_hooks(); build = factory.build_count_aware_model; runner.install_hooks()
    assert build is factory.build_count_aware_model is training.build_model
    args = SimpleNamespace(model_role=ROLE_BY_ARM[arm], time_head_mode="heteroscedastic_lognormal_duration",
        lambda_tail=0., lambda_log_qty=1., quantile_adaptive_strength=0.,
        checkpoint_monitor="validation_raw_quantity_rmse", dataset_contract="intermittent_frozen_5000")
    observed_time.validate_launch(args, (arm,), (runner.VARIANT,))
    for field, value in (("lambda_log_qty", 2.), ("model_role", ROLE_BY_ARM[next(a for a in ARMS if a != arm)])):
        changed = deepcopy(args); setattr(changed, field, value)
        with pytest.raises(ValueError):
            observed_time.validate_launch(changed, (arm,), (runner.VARIANT,))
    constants.validate_model_role_contract(model_role=args.model_role, backbones=(arm,),
        quantity_variants=(runner.VARIANT,), time_head_mode=args.time_head_mode, lambda_tail=0.)


def test_frozen_mlp16_identity_remains_exact():
    assert bytes(width_identity().tolist()).hex() == "1309671f1eb463992c1ec3b37209d73cc81501cc81d9e513029566e84bd8277a"


def test_optin_new_routes_preserve_legacy_causal_qkv_checkpoint_route():
    from models.TPPs import CountAwareFactory as factory
    from models.TPPs.CountAwareTitanCausalQKV import CAUSAL_QKV_BACKBONE
    runner.install_hooks()
    model, meta = factory.build_count_aware_model(CAUSAL_QKV_BACKBONE,
        hidden_dim=64, train_log_mean=1.2, max_seq_len=16,
        quantity_variant=runner.VARIANT, time_head_mode="legacy_clamped_rmtpp")
    value = {"backbone": CAUSAL_QKV_BACKBONE, "encoder_config": meta,
        "model_state_dict": model.state_dict(), "variant": runner.VARIANT,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    assert validate_checkpoint(value, CAUSAL_QKV_BACKBONE) is False
    factory.validate_checkpoint_route(value, CAUSAL_QKV_BACKBONE)
    for arm in ARMS:
        with pytest.raises(ValueError):
            factory.validate_checkpoint_route(value, arm)


def test_synthetic_optimizer_checkpoint_roundtrip():
    result = runner.synthetic_check("cpu")
    assert result["status"] == "passed"
    assert result["real_data_loaded"] is False
    assert result["held_out_test_evaluated"] is False
    assert result["synthetic_optimizer_updates"] == 8


@pytest.mark.parametrize("arm", ARMS)
def test_actual_shared_trainer_selected_last_validation_replay(cpu_training, admitted_contract, tmp_path, arm):
    from paper.scripts.count_aware_tpp_backbone import training
    contract = admitted_contract
    contract["training"] = {"maximum_epochs": 300, "minimum_epochs": 40,
                            "patience": 40, "monitor": "validation_raw_quantity_rmse"}
    item = next(d for d in contract["datasets"] if d["dataset_id"] == "intermittent_frozen_5000")
    frame, _ = runner.base.prepare_admitted_data(item)
    interface = runner.time_interface(item, frame, contract, arm=arm)
    initial = runner.initial_states(item, 42)
    args = runner.training_args(contract, item, tmp_path / "train", arm)
    args.device = "cpu"; args.epochs = args.min_epochs = args.early_stopping_patience = 2
    quantity = {"boundaries": item["quantity_boundaries_all_train_rows"],
                "strata": [{"label": f"bin_{i}"} for i in range(5)]}
    with runner.shared.audited_training(training, item, lambda: None, lambda *_: None):
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone=arm, quantity_variant=runner.VARIANT, seed=42)
    assert summary["initial_state_sha256"] == initial[arm]
    assert summary["held_out_test_evaluated"] is False
    directory = args.output_dir / "runs" / arm / runner.VARIANT / "seed_42"
    history = runner.read(directory / "history.json")["history"]
    resume = training._resume_identity(args=args, backbone=arm, quantity_variant=runner.VARIANT,
        seed=42, monitor=args.checkpoint_monitor, quantity_contract=quantity, interface_meta=interface)
    for filename, epoch in (("best_val_qty_rmse_model.pt", summary["best_epoch"]), ("last_epoch_state.pt", 2)):
        replay = runner.replay_checkpoint(directory / filename, item, frame, lambda: None, device="cpu",
            expected_arm=arm, expected_identity=resume, expected_initial=initial[arm], expected_epoch=epoch)
        assert replay["qty_rmse"] == pytest.approx(history[epoch - 1]["val_qty_rmse"], rel=1e-10, abs=1e-8)
        assert replay["qty_mae"] == pytest.approx(history[epoch - 1]["val_qty_mae"], rel=1e-10, abs=1e-8)
        assert replay["time_nll"] == pytest.approx(history[epoch - 1]["val_time_nll"], rel=1e-10, abs=1e-8)
        assert replay["evaluation_scope"] == "validation_only"
