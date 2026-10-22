"""CPU-only component-ablation invariants; no data files or training jobs."""
from copy import deepcopy

import pytest
import torch

from models.TPPs.CountAwareComponentAblation import (
    ARMS, CountAwareComponentAblation, HistoryZeroCorrection, build_model,
    identity, metadata,
)
from models.TPPs.CountAwareTitanSharedHistoryMLP import CountAwareTitanSharedHistoryMLP
from paper.scripts import run_titantpp_shared_history_cap as prepared


KWARGS = {
    "quantity_variant": "count_only_log_regression",
    "time_head_mode": "heteroscedastic_lognormal_duration",
    "train_log_std": .8, "time_scale": 1.,
    "time_initial_location": .2, "time_initial_scale": .8,
    "time_observation_contract": {
        "mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None,
    },
}


def make(arm, seed=77):
    torch.manual_seed(seed)
    return CountAwareComponentAblation(64, 1.2, 32, component_arm=arm, **deepcopy(KWARGS)).eval()


def test_full_factory_is_raw9_and_ablation_removes_real_parameter_sets():
    data = prepared.fabricated_data()
    torch.manual_seed(15)
    full, _ = build_model(data, "full")
    torch.manual_seed(15)
    raw, _ = __import__("paper.scripts.run_titantpp_shared_history_raw9", fromlist=["build_model"]).build_model(data, "titantpp_history_mlp_shared15_v1")
    assert type(full) is CountAwareTitanSharedHistoryMLP
    assert full.state_dict().keys() == raw.state_dict().keys()
    assert all(torch.equal(full.state_dict()[key], raw.state_dict()[key]) for key in full.state_dict())
    torch.manual_seed(77)
    reference = CountAwareTitanSharedHistoryMLP(64, 1.2, 32, history_sharing_mode="shared", **deepcopy(KWARGS)).eval()
    no_prototype, no_history, encoder_only = (make(arm) for arm in ARMS[1:])
    assert sum(p.numel() for p in no_prototype.parameters()) == sum(p.numel() for p in reference.parameters()) - 4096
    assert sum(p.numel() for p in no_history.parameters()) == sum(p.numel() for p in reference.parameters()) - 9600
    assert sum(p.numel() for p in encoder_only.parameters()) == sum(p.numel() for p in reference.parameters()) - 4096 - 9600
    assert not any(key.startswith("lmm.") for key in no_prototype.state_dict())
    assert not any(key.startswith("multilag_detail.") for key in no_history.state_dict())


def test_common_parameters_have_identical_initial_values_for_each_arm():
    states = {arm: make(arm, seed=91).state_dict() for arm in ARMS[1:]}
    common = set.intersection(*(set(state) for state in states.values())) - {"component_ablation_identity"}
    assert common
    assert all(torch.equal(states["no_prototype"][key], states["no_history"][key])
               and torch.equal(states["no_prototype"][key], states["encoder_only"][key])
               for key in common)


def test_full_delegation_preserves_rng_output_and_gradient():
    data = prepared.fabricated_data()
    torch.manual_seed(313)
    full, _ = build_model(data, "full")
    full_rng = torch.get_rng_state().clone()
    torch.manual_seed(313)
    raw, _ = __import__("paper.scripts.run_titantpp_shared_history_raw9", fromlist=["build_model"]).build_model(data, "titantpp_history_mlp_shared15_v1")
    assert torch.equal(full_rng, torch.get_rng_state())
    dt, qty, valid = prepared.fabricated_batch()
    dt_a, qty_a = dt.clone().requires_grad_(), qty.clone().requires_grad_()
    dt_b, qty_b = dt.clone().requires_grad_(), qty.clone().requires_grad_()
    out_a, _ = full.eval().encode_task_states(dt_a, qty_a, valid)
    out_b, _ = raw.eval().encode_task_states(dt_b, qty_b, valid)
    assert torch.equal(out_a, out_b)
    out_a.sum().backward(); out_b.sum().backward()
    assert torch.equal(dt_a.grad, dt_b.grad) and torch.equal(qty_a.grad, qty_b.grad)


@pytest.mark.parametrize("arm", ARMS[1:])
def test_target_padding_nan_is_causal_but_observed_nan_is_rejected_and_gradients_masked(arm):
    model = make(arm).train()
    dt = torch.tensor([[1., 2., float("nan"), float("nan")]], requires_grad=True)
    qty = torch.tensor([[2., 3., float("nan"), float("nan")]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    write = torch.tensor([[True, True, False, False]])
    encoded, _ = model.encode_task_states(dt, qty, valid, memory_write_mask=write)
    assert torch.isfinite(encoded).all()
    encoded.sum().backward()
    assert torch.equal(dt.grad[0, 2:], torch.zeros_like(dt.grad[0, 2:]))
    assert torch.equal(qty.grad[0, 2:], torch.zeros_like(qty.grad[0, 2:]))
    with pytest.raises(ValueError):
        model.encode_task_states(torch.tensor([[float("nan")]]), torch.ones(1, 1),
                                 torch.ones(1, 1, dtype=torch.bool))


def test_zero_correction_has_no_parameters_and_rejects_observed_nan():
    correction = HistoryZeroCorrection(64)
    assert not list(correction.parameters())
    with pytest.raises(ValueError):
        correction(torch.full((1, 1, 64), float("nan")), torch.ones(1, 1, dtype=torch.bool))


def test_strict_loading_rejects_foreign_arm_and_tamper():
    first, second = make("no_prototype"), make("no_history")
    state = deepcopy(first.state_dict())
    with pytest.raises(ValueError):
        second.load_state_dict(state)
    state["component_ablation_identity"] = identity(arm="encoder_only")
    with pytest.raises(ValueError):
        first.load_state_dict(state)
    state = deepcopy(first.state_dict())
    state.pop(next(key for key in state if key != "component_ablation_identity"))
    with pytest.raises(RuntimeError):
        first.load_state_dict(state)


def test_metadata_records_only_actual_components():
    for arm in ARMS[1:]:
        value = metadata(arm=arm)
        assert value["component_ablation_arm"] == arm
        assert value["prototype_component_enabled"] is (arm == "no_history")
        assert value["history_correction_enabled"] is (arm == "no_prototype")
