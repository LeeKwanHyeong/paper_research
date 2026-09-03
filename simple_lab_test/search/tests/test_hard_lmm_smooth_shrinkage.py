import copy
import inspect
import json

import pytest
import torch

from paper.scripts.hard_lmm_frozen_probe import acceptance, predict
from paper.scripts.run_hard_lmm_frozen_probe import sha256_file
from paper.scripts.run_hard_lmm_smooth_shrinkage import (
    CONTRACT_PATH, KINDS, ResidualGate, adaptive_decision, fit_gate, new_gate,
    preflight, read_cache, restore_checkpoint, save_checkpoint,
)


def cache(quantity=0., feature_dim=3):
    n = 16
    return {"features": torch.ones(n, feature_dim), "z": torch.ones(n),
        "projection": torch.ones(n), "quantity": torch.full((n,), quantity),
        "time_nll": torch.full((n,), 2.), "history_length": torch.full((n,), 12),
        "target_index": torch.arange(n), "series_index": torch.arange(n),
        "context_end": torch.full((n,), 12)}


POLICY = {"learning_rate": .001, "shuffle_seed": 42, "batch_size": 4,
          "gradient_clip": 1, "maximum_epochs": 3, "minimum_epochs": 2, "patience": 2}


@pytest.mark.parametrize("kind", KINDS)
def test_initialization_is_matched_near_identity_not_exact_identity(kind):
    data = cache()
    z, g, correction = predict(new_gate(3, kind), data)
    assert torch.allclose(g, torch.full_like(g, .99), atol=1e-7, rtol=0)
    assert not torch.equal(z, data["z"])
    assert torch.equal(correction, (g - 1) * data["projection"])
    original = predict(ResidualGate(3, "identity"), data)
    assert torch.equal(original[0], data["z"])
    assert torch.equal(original[1], torch.ones(16))
    assert torch.equal(original[2], torch.zeros(16))


@pytest.mark.parametrize("kind", KINDS)
def test_negative_score_keeps_gradient_and_extreme_finite_input_is_bounded(kind):
    model = new_gate(3, kind)
    parameter = model.score_bias if kind == "constant_shrinkage" else model.network[-1].bias
    with torch.no_grad():
        parameter.fill_(-5)
    data = cache()
    z, _, _ = model(data["features"], data["z"], data["projection"])
    z.sum().backward()
    assert parameter.grad.abs().sum() > 0
    for score in (-1000., 1000.):
        with torch.no_grad():
            parameter.fill_(score)
        z, g, correction = model(torch.full((16, 3), 1e20), data["z"], torch.full((16,), 1e20))
        assert all(torch.isfinite(v).all() for v in (z, g, correction))
        assert bool(((g >= .8) & (g <= 1)).all())


def test_scalar_ignores_features_and_adaptive_can_use_them_without_target_input():
    data = cache()
    scalar = new_gate(3, "constant_shrinkage")
    adaptive = new_gate(3, "smooth_shrinkage")
    assert list(inspect.signature(ResidualGate.forward).parameters) == [
        "self", "features", "base_logit", "residual_projection"]
    with torch.no_grad():
        adaptive.network[0].weight.fill_(.2)
        adaptive.network[0].bias.zero_()
        adaptive.network[-1].weight.fill_(.1)
    def output(model, features):
        return model(features, data["z"], data["projection"])[1]
    assert torch.equal(output(scalar, data["features"]), output(scalar, -data["features"]))
    assert not torch.equal(output(adaptive, data["features"]), output(adaptive, -data["features"]))


@pytest.mark.parametrize("kind", KINDS)
def test_preflight_learns_and_main_fit_resets_state(kind):
    initial = copy.deepcopy(new_gate(3, kind).state_dict())
    result = preflight(kind, cache(), POLICY)
    assert result["status"] == "passed" and not result["reused_in_main_fit"]
    assert result["diagnostics"]["first_gradient_norm"] > 0
    assert result["maximum_train_gate_change"] > 0
    assert "validation" not in inspect.signature(preflight).parameters
    assert all(torch.equal(v, new_gate(3, kind).state_dict()[k]) for k, v in initial.items())
    selected, _, fit = fit_gate(kind, cache(), cache(), POLICY, lambda row: None)
    again, _, repeated = fit_gate(kind, cache(), cache(), POLICY, lambda row: None)
    assert torch.equal(predict(selected, cache())[0], predict(again, cache())[0])
    assert fit["best_epoch"] == repeated["best_epoch"] > 0


@pytest.mark.parametrize("kind", KINDS)
def test_preflight_rejects_no_effect(kind):
    data = cache()
    data["projection"].zero_()
    with pytest.raises(RuntimeError, match="did not learn"):
        preflight(kind, data, POLICY)


@pytest.mark.parametrize("kind", KINDS)
def test_identity_fallback_selection_and_checkpoint_replay(tmp_path, kind):
    data = cache(100.)
    selected, final, result = fit_gate(kind, data, data, POLICY, lambda row: None)
    assert result["best_epoch"] == -1 and selected.kind == "identity"
    assert result["completed_epochs"] == POLICY["minimum_epochs"]
    assert torch.equal(predict(selected, data)[0], data["z"])
    assert final.kind == kind
    for model in (selected, final):
        path = tmp_path / f"{model.kind}.pt"
        save_checkpoint(path, model, kind, "base", "contract")
        restored = restore_checkpoint(path, "base", "contract")
        assert all(torch.equal(a, b) for a, b in zip(predict(model, data), predict(restored, data)))
        with pytest.raises(ValueError, match="provenance"):
            restore_checkpoint(path, "other", "contract")


def test_cache_scope_digest_full_validation_and_finite_checks(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    spec = {"dataset": "dataset", "checkpoint_state_sha256": "base"}
    with pytest.raises(ValueError, match="forbidden"):
        read_cache(tmp_path, spec, "test", {})
    path = folder / "validation_cache.pt"
    audit_path = folder / "baseline_audit.json"
    def prepare(data, available=16):
        torch.save(data, path)
        audit_path.write_text(json.dumps({"model_state_sha256": "base", "cache": {
            "validation": {"sha256": sha256_file(path), "selected_targets": 16,
                           "available_targets": available}}}))
    prepare(cache(feature_dim=138))
    assert len(read_cache(tmp_path, spec, "validation", {})[0]["z"]) == 16
    torch.save(cache(2., 138), path)
    with pytest.raises(ValueError, match="checksum"):
        read_cache(tmp_path, spec, "validation", {})
    prepare(cache(feature_dim=138), 17)
    with pytest.raises(ValueError, match="count"):
        read_cache(tmp_path, spec, "validation", {})
    bad = cache(feature_dim=138)
    bad["features"][0, 0] = float("nan")
    prepare(bad)
    with pytest.raises(FloatingPointError):
        read_cache(tmp_path, spec, "validation", {})
    bad = cache(feature_dim=138)
    bad["z"] = bad["z"].unsqueeze(1)
    prepare(bad)
    with pytest.raises(ValueError, match="alignment"):
        read_cache(tmp_path, spec, "validation", {})


def test_adaptive_gate_keeps_parent_thresholds_and_requires_scalar_advantage():
    def result(body):
        scopes = {scope: {"status": "evaluated", "baseline": {
            "qty_mae": 10., "qty_rmse": 10., "time_nll": 2., "joint_objective": 3.},
            "candidate": {"qty_mae": body if scope == "body_le_p95" else 10.,
            "qty_rmse": 10., "time_nll": 2., "joint_objective": 2.9}}
            for scope in ("overall", "body_le_p95", "gt_p99")}
        return {"scopes": scopes, "decision": acceptance(scopes, 1, True)}
    smooth = result(9.4)
    assert adaptive_decision(smooth, result(9.5))["passes"]
    assert not adaptive_decision(smooth, result(9.3))["passes"]
    assert not adaptive_decision(result(9.6), result(9.7))["passes"]
    smooth["decision"] = acceptance(smooth["scopes"], 0, True)
    assert not adaptive_decision(smooth, result(9.5))["passes"]
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["candidates"] == list(KINDS)
    assert contract["architecture"]["initial_gate"] == .99
    assert contract["architecture"]["maximum_shrinkage"] == .2
    assert not contract["decision"]["automatic_fresh_training"]
