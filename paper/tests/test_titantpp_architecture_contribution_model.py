"""Scientific intervention and checkpoint guards; synthetic tensors only."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from models.TPPs.CountAwareFactory import with_time_metadata
from models.TPPs.CountAwareTitanArchitectureContribution import (
    ARMS, CountAwareTitanArchitectureContribution, backbone_for_arm,
    expected_intervention_shapes, metadata, role_for_arm, validate_checkpoint,
)
from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth

torch.set_num_threads(1)
KWARGS = {"time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 1.,
          "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None}}


def build(arm, seed=42):
    torch.manual_seed(seed)
    return CountAwareTitanArchitectureContribution(64, 2., 256, architecture_arm=arm, **KWARGS)


def payload(arm):
    model = build(arm)
    _, meta = with_time_metadata(model, {**metadata(arm=arm), "max_len": 256})
    interface = {"time_head": {"observation_likelihood": meta["time_head"]["observation_likelihood"]}}
    backbone = backbone_for_arm(arm)
    return {"backbone": backbone, "encoder_config": meta, "interface_meta": interface,
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "variant": "count_only_log_regression", "model_state_dict": deepcopy(model.state_dict()),
            "resume_identity": {"backbone": backbone, "interface_meta": deepcopy(interface),
                                "checkpoint_monitor": "validation_raw_quantity_rmse",
                                "arguments": {"model_role": role_for_arm(arm)}}}


@pytest.mark.parametrize("arm", ARMS)
def test_real_parameter_budget_and_exact_checkpoint_roundtrip(arm):
    m = build(arm)
    optional = {name: tuple(p.shape) for name, p in m.named_parameters()
                if name.startswith(("lmm.", "multilag_detail."))}
    assert optional == expected_intervention_shapes(arm)
    assert sum(p.numel() for n, p in m.named_parameters() if n in optional) == metadata(arm=arm)["intervention_parameter_count"]
    p = payload(arm)
    assert validate_checkpoint(p, backbone_for_arm(arm))
    target = build(arm, seed=52)
    target.load_state_dict(p["model_state_dict"], strict=False)
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in m.state_dict().items())


@pytest.mark.parametrize("change", ["label", "arm", "budget", "role", "scope", "test", "selector", "identity", "missing", "extra", "shape", "nan"])
def test_changed_scientific_identity_rejected_before_loading(change):
    p = payload("P1_H1")
    if change == "label": p["backbone"] = backbone_for_arm("PF_H1")
    if change == "arm": p["encoder_config"]["architecture_arm"] = "PF_H1"
    if change == "budget": p["encoder_config"]["intervention_parameter_count"] = 1
    if change == "role": p["resume_identity"]["arguments"]["model_role"] = "other"
    if change == "scope": p["evaluation_scope"] = "test"
    if change == "test": p["held_out_test_evaluated"] = True
    if change == "selector": p["resume_identity"]["checkpoint_monitor"] = "time_nll"
    if change == "identity": p["model_state_dict"]["architecture_contribution_identity"][0] ^= 1
    if change == "missing": p["model_state_dict"].pop("lmm.mem")
    if change == "extra": p["model_state_dict"]["lmm.dummy"] = torch.zeros(4096)
    if change == "shape": p["model_state_dict"]["lmm.mem"] = torch.zeros(1, 32, 128)
    if change == "nan": p["model_state_dict"]["lmm.mem"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_checkpoint(p, backbone_for_arm("P1_H1"))


def test_identity_prevents_cross_arm_or_legacy_relabelling():
    p = payload("P1_H1")
    with pytest.raises(ValueError, match="relabelled"):
        validate_checkpoint(p, "titantpp_history_mlp_width8")
    target = build("P1_HC")
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="identity"):
        target.load_state_dict(p["model_state_dict"])
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())


def test_incomplete_or_nonfinite_common_state_rejected_before_mutation():
    m = build("P0_H0")
    before = deepcopy(m.state_dict())
    state = deepcopy(before)
    name = next(n for n, p in m.named_parameters())
    state[name].flatten()[0] = float("nan")
    with pytest.raises(RuntimeError, match="tensor mismatch"):
        m.load_state_dict(state)
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    state = deepcopy(before)
    state.pop(name)
    with pytest.raises(RuntimeError, match="complete exact"):
        m.load_state_dict(state, strict=False)


def test_anchor_initialization_and_forward_are_native_width8():
    m = build("P1_H1").eval()
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    frozen = CountAwareTitanHistoryWidth(64, 2., 256, history_mlp_width=8, **KWARGS).eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in m.state_dict().items():
        if name != "architecture_contribution_identity":
            assert torch.equal(value, frozen.state_dict()[name])
    x = torch.ones(2, 132)
    q = torch.arange(132).float().repeat(2, 1)
    valid = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(m.encode_task_states(x, q, valid)[0], frozen.encode_task_states(x, q, valid)[0])


@pytest.mark.parametrize("bad", ["unknown_arm", "hidden32", "legacy_time"])
def test_contract_refuses_unsupported_architecture(bad):
    with pytest.raises(ValueError):
        if bad == "unknown_arm": build("P2_H1")
        elif bad == "hidden32": CountAwareTitanArchitectureContribution(32, 2., 256, **KWARGS)
        else: CountAwareTitanArchitectureContribution(64, 2., 256, time_head_mode="legacy_clamped")


def test_cpu_synthetic_qualification_checks_two_update_participation_and_causality():
    path = Path(__file__).resolve().parents[1] / "scripts/qualify_titantpp_architecture_contribution.py"
    spec = importlib.util.spec_from_file_location("architecture_native_qualification", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    result = native.qualify("cpu")
    assert result["status"] == "passed" and result["real_training_updates"] == 0
    assert len(result["initialization"]) == 3 and len(result["arms"]) == 6
    assert all(a["causal"] and a["padding_invariant"] and a["deterministic_eval"] for a in result["arms"])
    assert next(a for a in result["arms"] if a["arm"] == "P1_H1")["P1_H1_frozen_width8_exact_after_updates"]
    assert all(a["RAF_branch128_ineligible"] for a in result["correction_reset"])
