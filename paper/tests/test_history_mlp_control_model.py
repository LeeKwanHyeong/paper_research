"""CPU-only synthetic invariants for the fixed 9,600-parameter MLP control."""
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs import CountAwareComponentAblation as old
from models.TPPs.CountAwareHistoryMLPControl import (
    ARMS, BACKBONE_BY_ARM, CORRECTION_PARAMETER_COUNT, PROJECTION_SHAPES,
    CountAwareHistoryMLPControl, SingleHistoryCorrection, build_model,
    identity, metadata, validate_checkpoint,
)
from models.TPPs.CountAwareTitanMultiLagDetail import AVAILABILITY_LAGS, lag_source_indices
from paper.scripts import run_titantpp_shared_history_cap as prepared
from paper.tests.test_shared_component_model import KWARGS


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def make(seed=42):
    torch.manual_seed(seed)
    return CountAwareHistoryMLPControl(64, 1.2, 256, **deepcopy(KWARGS)).eval()


def active_correction(seed=42):
    torch.manual_seed(seed)
    model = SingleHistoryCorrection()
    with torch.no_grad():
        model.output_projection.weight.normal_(std=.15)
    return model


def payload_for(arm):
    torch.manual_seed(42)
    model, config = build_model(prepared.fabricated_data(), arm)
    return model, {"backbone": BACKBONE_BY_ARM[arm], "encoder_config": config,
        "model_state_dict": deepcopy(model.state_dict())}


@pytest.mark.parametrize("seed", (42, 52, 62))
def test_all_three_arms_keep_common_tensors_rng_heads_and_total_budget(seed):
    models, states, rngs = {}, {}, {}
    for arm in ARMS:
        torch.manual_seed(seed)
        models[arm], _ = build_model(prepared.fabricated_data(), arm)
        states[arm] = models[arm].state_dict()
        rngs[arm] = torch.get_rng_state().clone()
    assert all(torch.equal(rngs["shared15"], value) for value in rngs.values())
    common = set.intersection(*(set(value) for value in states.values()))
    assert any(name.startswith("encoder.") for name in common)
    assert all(torch.equal(states["shared15"][name], states[arm][name]) for name in common for arm in ARMS)
    counts = {arm: sum(value.numel() for value in model.parameters()) for arm, model in models.items()}
    assert counts["single_mlp"] == counts["shared15"] == counts["encoder_only"] + 9600
    assert all(isinstance(model.lmm, nn.Identity) and not any(name.startswith("lmm.") for name in model.state_dict())
        for model in models.values())
    assert all(models[arm].time_head_contract() == models["shared15"].time_head_contract() for arm in ARMS)


@pytest.mark.parametrize("arm", ("shared15", "encoder_only"))
def test_frozen_factory_and_metadata_are_exact_delegations(arm):
    data = prepared.fabricated_data()
    torch.manual_seed(13)
    actual, actual_meta = build_model(data, arm)
    actual_rng = torch.get_rng_state().clone()
    torch.manual_seed(13)
    expected, expected_meta = old.build_model(data, BACKBONE_BY_ARM[arm])
    assert type(actual) is type(expected)
    assert actual_meta == expected_meta and metadata(arm=arm) == old.metadata(arm=BACKBONE_BY_ARM[arm])
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert actual.state_dict().keys() == expected.state_dict().keys()
    assert all(torch.equal(actual.state_dict()[key], expected.state_dict()[key]) for key in actual.state_dict())


def test_single_correction_exact_shapes_bias_activation_zero_init_and_cpu_rng():
    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    correction = SingleHistoryCorrection()
    assert torch.equal(before, torch.get_rng_state())
    assert sum(value.numel() for value in correction.parameters()) == CORRECTION_PARAMETER_COUNT == (136 + 64) * 48
    assert correction.input_projection.weight.shape == (48, 136)
    assert correction.output_projection.weight.shape == (64, 48)
    assert correction.input_projection.bias is correction.output_projection.bias is None
    assert correction.input_projection.weight.device.type == "cpu"
    assert torch.count_nonzero(correction.input_projection.weight) > 0
    assert torch.count_nonzero(correction.output_projection.weight) == 0
    hidden = torch.randn(2, 4, 64)
    assert torch.equal(correction(hidden, torch.ones(2, 4, dtype=torch.bool)), torch.zeros_like(hidden))
    active = active_correction()
    valid = torch.ones(2, 4, dtype=torch.bool)
    sources, available = lag_source_indices(valid, mode="local")
    previous = hidden.gather(1, sources[..., 0, None].expand_as(hidden))
    previous = torch.where(available[..., 0, None], previous, 0.)
    expected = active.output_projection(F.gelu(active.input_projection(
        torch.cat((hidden, previous, available.to(hidden.dtype)), dim=-1)), approximate="none")) / 8
    expected = torch.where(available[..., 0, None], expected, 0.)
    assert torch.equal(active(hidden, valid), expected)


def test_all_availability_thresholds_padding_skip_and_same_immediate_predecessor():
    correction = SingleHistoryCorrection()
    hidden = torch.arange(132 * 64, dtype=torch.float32).reshape(1, 132, 64)
    valid = torch.ones(1, 132, dtype=torch.bool)
    valid[:, [2, 9]] = False
    hidden[:, [2, 9]] = float("nan")
    captured = []
    hook = correction.input_projection.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    output = correction(hidden, valid)
    hook.remove()
    sources, available = lag_source_indices(valid, mode="local")
    assert correction.availability_lags == AVAILABILITY_LAGS == (1, 2, 4, 8, 16, 32, 64, 128)
    assert torch.equal(captured[0][..., 128:].bool(), available)
    assert not available[0, 129, 7] and available[0, 130, 7]
    assert torch.equal(captured[0][0, 3, 64:128], hidden[0, 1])
    assert torch.equal(captured[0][0, 10, 64:128], hidden[0, 8])
    assert torch.count_nonzero(captured[0][0, [0, 2, 9], 64:128]) == 0
    assert torch.equal(output, torch.zeros_like(output)) and torch.isfinite(captured[0]).all()
    counts = valid.long().cumsum(1)
    expected = valid[..., None] & (counts[..., None] > torch.tensor(AVAILABILITY_LAGS))
    assert torch.equal(available, expected)
    assert torch.equal(sources[..., 0][available[..., 0]], sources[..., 1][available[..., 0]])


def test_withheld_reset_unknown_nan_and_gradients_are_observation_safe():
    correction = active_correction()
    values = torch.randn(1, 7, 64)
    values[:, [2, 3]] = float("nan")
    values.requires_grad_()
    valid = torch.tensor([[True, True, False, True, True, True, True]])
    write = torch.tensor([[True, True, False, False, True, True, True]])
    out = correction(values, valid, memory_write_mask=write)
    assert torch.isfinite(out).all()
    assert torch.equal(out[:, [0, 2, 3, 4]], torch.zeros_like(out[:, [0, 2, 3, 4]]))
    assert torch.count_nonzero(out[:, [1, 5, 6]]) > 0
    changed = values.detach().clone()
    changed[:, :2] = 100.
    assert torch.equal(out[:, 4:], correction(changed, valid, memory_write_mask=write)[:, 4:])
    out.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert torch.equal(values.grad[:, [2, 3]], torch.zeros_like(values.grad[:, [2, 3]]))
    broken = values.detach().clone()
    broken[:, 4] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite observed"):
        correction(broken, valid, memory_write_mask=write)


def test_no_future_dependence_or_cross_sample_memory():
    correction = active_correction()
    hidden = torch.randn(2, 10, 64, requires_grad=True)
    valid = torch.ones(2, 10, dtype=torch.bool)
    original = correction(hidden, valid)
    changed = hidden.detach().clone()
    changed[:, 6:] += 1000
    assert torch.equal(original[:, :6], correction(changed, valid)[:, :6])
    original[:, 5].sum().backward()
    assert torch.equal(hidden.grad[:, 6:], torch.zeros_like(hidden.grad[:, 6:]))
    assert torch.count_nonzero(hidden.grad[:, 4]) > 0
    assert torch.equal(original[:1], correction(hidden[:1].detach(), valid[:1]))
    correction(torch.randn_like(hidden), valid)
    assert torch.equal(original, correction(hidden.detach(), valid))


def test_availability_can_create_length_only_correction_and_metadata_admits_limit():
    correction = SingleHistoryCorrection()
    with torch.no_grad():
        correction.input_projection.weight.zero_()
        correction.input_projection.weight[0, 129] = 1.
        correction.output_projection.weight.zero_()
        correction.output_projection.weight[:, 0] = 1.
    output = correction(torch.zeros(1, 5, 64), torch.ones(1, 5, dtype=torch.bool))
    assert torch.count_nonzero(output[:, :2]) == 0
    assert torch.allclose(output[0, 2], F.gelu(torch.ones(64), approximate="none") / 8)
    meta = metadata(arm="single_mlp")
    assert meta["history_length_only_correction_possible"] is True
    assert meta["parameter_matching"] == "equal_total_correction_parameters_not_active_prefix_or_function_class"
    assert "active_prefix_parameter_counts" not in meta
    assert meta["history_relation_parameter_count"] == meta["additional_parameter_count"] == 9600
    assert meta["multilag_detail_projection_shapes"] == {key: list(shape) for key, shape in PROJECTION_SHAPES.items()}


def test_fresh_first_backward_flows_to_output_then_history_input_path_is_live():
    correction = SingleHistoryCorrection()
    hidden = torch.randn(2, 6, 64, requires_grad=True)
    valid = torch.ones(2, 6, dtype=torch.bool)
    correction(hidden, valid).sum().backward()
    assert torch.count_nonzero(correction.output_projection.weight.grad) > 0
    assert torch.count_nonzero(correction.input_projection.weight.grad) == 0
    with torch.no_grad():
        correction.output_projection.weight.add_(correction.output_projection.weight.grad, alpha=-.01)
    correction.zero_grad(set_to_none=True)
    hidden.grad = None
    correction(hidden, valid)[:, 4].sum().backward()
    assert torch.count_nonzero(correction.input_projection.weight.grad) > 0
    assert torch.count_nonzero(hidden.grad[:, 3]) > 0
    assert torch.equal(hidden.grad[:, 5:], torch.zeros_like(hidden.grad[:, 5:]))


def test_full_model_target_nan_and_padding_have_exact_zero_influence_and_gradient():
    model = make().eval()
    with torch.no_grad():
        model.multilag_detail.output_projection.weight.fill_(.01)
    valid = torch.tensor([[True, True, True, False, True]])
    write = torch.tensor([[True, True, False, False, False]])
    dts = torch.tensor([[1., 2., float("nan"), float("nan"), float("nan")]], requires_grad=True)
    quantities = torch.tensor([[2., 3., float("nan"), float("nan"), float("nan")]], requires_grad=True)
    encoded, _ = model.encode_task_states(dts, quantities, valid, memory_write_mask=write)
    zeros, _ = model.encode_task_states(torch.nan_to_num(dts), torch.nan_to_num(quantities), valid, memory_write_mask=write)
    assert torch.isfinite(encoded).all() and torch.equal(encoded, zeros)
    assert torch.count_nonzero(encoded[:, 2:]) == 0
    encoded.sum().backward()
    assert torch.equal(dts.grad[:, 2:], torch.zeros_like(dts.grad[:, 2:]))
    assert torch.equal(quantities.grad[:, 2:], torch.zeros_like(quantities.grad[:, 2:]))
    with pytest.raises(ValueError):
        model.encode_task_states(torch.tensor([[float("nan")]]), torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool))


def test_fresh_three_arm_forward_and_shared_losses_are_identical():
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    dt, qty, valid = prepared.fabricated_batch()
    outputs = {}
    for arm in ARMS:
        torch.manual_seed(52)
        model, _ = build_model(prepared.fabricated_data(), arm)
        outputs[arm] = target_outputs(model.eval(), dt, valid, qty, lambda_log_qty=1.)
    common = set.intersection(*(set(values) for values in outputs.values()))
    assert "joint_loss" in common
    for field in common:
        if isinstance(outputs["shared15"][field], torch.Tensor):
            assert all(torch.equal(outputs["shared15"][field], outputs[arm][field]) for arm in ARMS), field


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_validates_only_actual_backbone_and_rejects_other_arms(arm):
    _, payload = payload_for(arm)
    assert validate_checkpoint(payload, arm) is True
    for wrong in set(ARMS) - {arm}:
        with pytest.raises(ValueError):
            validate_checkpoint(payload, wrong)
    if arm == "shared15":
        payload["backbone"] = "shared15"
        with pytest.raises(ValueError, match="arm mismatch"):
            validate_checkpoint(payload, arm)


@pytest.mark.parametrize("fault", ("metadata", "identity", "shape", "dtype", "nonfinite", "extra_projection", "missing_projection", "foreign_identity", "prototype", "missing_state", "metadata_alias"))
def test_new_checkpoint_rejects_metadata_identity_parameter_shape_and_state_corruption(fault):
    _, payload = payload_for("single_mlp")
    state = payload["model_state_dict"]
    if fault == "metadata":
        payload["encoder_config"]["history_relation_parameter_count"] -= 1
    elif fault == "identity":
        state["control_identity"] = identity().roll(1)
    elif fault == "shape":
        state["multilag_detail.input_projection.weight"] = torch.zeros(48, 128)
    elif fault == "dtype":
        state["multilag_detail.input_projection.weight"] = state["multilag_detail.input_projection.weight"].double()
    elif fault == "nonfinite":
        state["multilag_detail.output_projection.weight"][0, 0] = float("nan")
    elif fault == "extra_projection":
        state["multilag_detail.input_projection.bias"] = torch.zeros(48)
    elif fault == "missing_projection":
        del state["multilag_detail.output_projection.weight"]
    elif fault == "foreign_identity":
        state["component_ablation_identity"] = old.identity(arm="encoder_only")
    elif fault == "prototype":
        state["lmm.memory"] = torch.zeros(64, 64)
    elif fault == "missing_state":
        del payload["model_state_dict"]
    elif fault == "metadata_alias":
        payload["model_metadata"] = deepcopy(payload["encoder_config"])
        payload["model_metadata"]["control_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_checkpoint(payload, "single_mlp")


@pytest.mark.parametrize("fault", ("missing", "extra", "dtype", "shape", "nonfinite", "identity", "foreign"))
@pytest.mark.parametrize("strict", (True, False))
def test_complete_state_preflight_rejects_every_fault_before_mutating_any_tensor(fault, strict):
    model = make()
    original = deepcopy(model.state_dict())
    incoming = deepcopy(original)
    early = next(key for key, value in incoming.items() if value.is_floating_point())
    incoming[early].add_(17.)
    key = "multilag_detail.output_projection.weight"
    if fault == "missing":
        del incoming[key]
    elif fault == "extra":
        incoming["unapproved_extra"] = torch.ones(1)
    elif fault == "dtype":
        incoming[key] = incoming[key].double()
    elif fault == "shape":
        incoming[key] = incoming[key][:63]
    elif fault == "nonfinite":
        incoming[key][0, 0] = float("inf")
    elif fault == "identity":
        incoming["control_identity"] = incoming["control_identity"].roll(1)
    elif fault == "foreign":
        incoming["component_ablation_identity"] = old.identity(arm="encoder_only")
    with pytest.raises((RuntimeError, ValueError)):
        model.load_state_dict(incoming, strict=strict)
    assert all(torch.equal(original[name], value) for name, value in model.state_dict().items())


def test_exact_round_trip_and_correction_partial_strict_false_rejection():
    model, other = make(42), make(62)
    other.load_state_dict(model.state_dict(), strict=False)
    assert all(torch.equal(value, other.state_dict()[name]) for name, value in model.state_dict().items())
    correction = SingleHistoryCorrection()
    before = deepcopy(correction.state_dict())
    with pytest.raises(RuntimeError, match="complete exact"):
        correction.load_state_dict({"input_projection.weight": before["input_projection.weight"]}, strict=False)
    assert all(torch.equal(before[name], value) for name, value in correction.state_dict().items())
    assert "control_identity" in model.state_dict()
    assert not any(name in model.state_dict() for name in ("component_ablation_identity", "shared_history_identity", "core_ablation_identity"))


@pytest.mark.parametrize("factory", (SingleHistoryCorrection, lambda hidden: metadata(hidden, arm="single_mlp")))
@pytest.mark.parametrize("hidden", (True, 32, 128))
def test_fixed_hidden_dimension_rejects_foreign_architecture(factory, hidden):
    with pytest.raises(ValueError, match="hidden_dim=64"):
        factory(hidden)


def test_masks_shapes_and_empty_sequences_are_checked_without_nan_leakage():
    correction = SingleHistoryCorrection()
    for hidden, valid, write in (
        (torch.zeros(1, 2, 63), torch.ones(1, 2, dtype=torch.bool), None),
        (torch.zeros(1, 2, 64), torch.ones(1, 2), None),
        (torch.zeros(1, 2, 64), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 1, dtype=torch.bool)),
    ):
        with pytest.raises(ValueError):
            correction(hidden, valid, memory_write_mask=write)
    empty = correction(torch.empty(2, 0, 64), torch.empty(2, 0, dtype=torch.bool))
    assert empty.shape == (2, 0, 64)
