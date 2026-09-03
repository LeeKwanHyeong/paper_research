import copy
import itertools

import pytest
import torch
from torch.nn import functional as F

from paper.scripts.run_hard_lmm_readout_factorial import (
    HEADS, OBJECTIVES, SELECTORS, Readout, fit, new_readout, predict, preflight,
    quantity_loss, restore_checkpoint, save_checkpoint,
)
from paper.scripts.run_hard_lmm_smooth_shrinkage import read_cache
from paper.scripts.validate_hard_lmm_readout_factorial import check_history


def cache():
    generator = torch.Generator().manual_seed(123)
    return {"features": torch.randn(40, 3, generator=generator),
        "z": torch.randn(40, generator=generator), "projection": torch.randn(40, generator=generator),
        "quantity": torch.arange(40).float() / 4, "time_nll": torch.full((40,), 2.)}


POLICY = {"seed": 42, "shuffle_seed": 42, "batch_size": 16, "maximum_epochs": 3,
    "learning_rate": .001, "weight_decay": 0, "gradient_clip": 1}


@pytest.mark.parametrize("kind,objective", itertools.product(HEADS, OBJECTIVES))
def test_preflight_identity_finite_and_fresh(kind, objective):
    train = cache()
    before = copy.deepcopy(train)
    model = new_readout(kind, train)
    assert torch.equal(predict(model, train)[0], train["z"])
    result = preflight(kind, objective, train, POLICY)
    assert result["status"] == "passed" and not result["reused_in_main_fit"]
    assert torch.equal(predict(new_readout(kind, train), train)[0], train["z"])
    assert all(torch.equal(before[k], train[k]) for k in train)


@pytest.mark.parametrize("kind", HEADS)
def test_targets_never_enter_inputs_or_normalization(kind):
    original, changed = cache(), cache()
    changed["quantity"].fill_(99999)
    changed["time_nll"].fill_(-99999)
    a, b = new_readout(kind, original), new_readout(kind, changed)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    with torch.no_grad():
        for p in a.parameters():
            p.fill_(.1)
    assert torch.equal(predict(a, original)[0], predict(a, changed)[0])


@pytest.mark.parametrize("kind,objective", itertools.product(HEADS, OBJECTIVES))
def test_fixed_budget_dual_selection_reproducible_and_checkpoint(tmp_path, kind, objective):
    train = cache()
    validation = cache()
    validation["features"] = validation["features"] + .2
    selected, final, result = fit(kind, objective, train, validation, 8., POLICY, lambda _: None)
    result["selections"] = {s: {"best_epoch": result["best_epochs"][s]} for s in SELECTORS}
    check_history(result, POLICY, len(train["z"]))
    assert len(result["history"]) == 4
    for tag, model in {**selected, "final": final}.items():
        file = tmp_path / f"{tag}.pt"
        save_checkpoint(file, model, "base", "contract")
        restored = restore_checkpoint(file, "base", "contract")
        assert torch.equal(predict(model, validation)[0], predict(restored, validation)[0])
        with pytest.raises(ValueError, match="provenance"):
            restore_checkpoint(file, "wrong", "contract")
    second, _, again = fit(kind, objective, train, validation, 8., POLICY, lambda _: None)
    assert again["best_epochs"] == result["best_epochs"]
    for tag in SELECTORS:
        assert torch.equal(predict(selected[tag], validation)[0], predict(second[tag], validation)[0])


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_loss_and_gradient_definition(objective):
    z, q = torch.tensor([-.5, 1.5], requires_grad=True), torch.tensor([1., 4.])
    expected = ((F.softplus(z) - q.log1p()).square().mean() if objective == "log_mse"
        else (F.softplus(z).expm1() - q).abs().mean())
    observed = quantity_loss(z, q, objective)
    assert torch.equal(observed, expected)
    assert torch.equal(torch.autograd.grad(observed, z, retain_graph=True)[0], torch.autograd.grad(expected, z)[0])


def test_fail_fast_nonfinite_and_forbidden_splits(tmp_path):
    with pytest.raises(FloatingPointError):
        quantity_loss(torch.tensor([1000.]), torch.tensor([1.]), "raw_mae")
    with pytest.raises(FloatingPointError):
        quantity_loss(torch.tensor([float("nan")]), torch.tensor([1.]), "log_mse")
    with pytest.raises(ValueError, match="Negative"):
        quantity_loss(torch.zeros(1), torch.tensor([-1.]), "raw_mae")
    with pytest.raises(ValueError, match="forbidden"):
        read_cache(tmp_path, {}, "test", {})
    with pytest.raises(ValueError):
        Readout(3, "unknown")


def test_unbounded_correction_and_parameter_capacity():
    train = cache()
    for kind, parameters in (("constant", 1), ("linear", 6), ("mlp", 113)):
        model = new_readout(kind, train)
        assert sum(p.numel() for p in model.parameters()) == parameters
        with torch.no_grad():
            if kind == "constant":
                model.offset.fill_(2.)
            elif kind == "linear":
                model.network.bias.fill_(2.)
            else:
                model.network[-1].bias.fill_(2.)
        assert torch.equal(predict(model, train)[2], torch.full_like(train["z"], 2.))


def test_selection_ties_prefer_identity_and_validator_rejects_wrong_epoch():
    metrics = {"time_nll": 1., "joint_objective": 3., "log_qty_mse": 2., "body_mae": 5.}
    history = [{"epoch": 0, **metrics, "train": metrics}]
    for epoch in (1, 2, 3):
        history.append({"epoch": epoch, **metrics, "train": metrics, "batches": 3,
            "zero_gradient_batches": 0, "best_epochs": {"joint": 0, "body": 0}})
    result = {"completed_epochs": 3, "history": history, "best_epochs": {"joint": 0, "body": 0},
        "selections": {s: {"best_epoch": 0} for s in SELECTORS}}
    check_history(result, POLICY, 40)
    result["best_epochs"]["body"] = 1
    with pytest.raises(AssertionError):
        check_history(result, POLICY, 40)
