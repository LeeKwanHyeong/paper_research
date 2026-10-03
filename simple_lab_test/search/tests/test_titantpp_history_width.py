"""Synthetic checks of the isolated width intervention; no real-data evaluation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from models.TPPs.CountAwareTitanHistoryWidth import (
    ARM, ROLE, WIDTH, HistoryWidthCorrection, metadata, validate_checkpoint,
)
from models.TPPs.CountAwareTitanCoreAblation import HistoryCorrection
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices
from paper.scripts import run_titantpp_history_width as runner
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract, cpu_training


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def data():
    return {"model": {"hidden_dim": 64, "quantity_variant": runner.VARIANT,
        "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 7.,
        "time_initial_location": .2, "time_initial_scale": .8,
        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None}},
        "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": 256}}


def payload():
    model, meta = runner.build_model(data(), ARM)
    interface = {"time_head": meta["time_head"]}
    return {"backbone": ARM, "encoder_config": meta, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "variant": runner.VARIANT, "interface_meta": interface,
        "resume_identity": {"backbone": ARM, "interface_meta": interface,
            "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": ROLE}}}


def test_width4_reference_bitwise_outputs_gradients_and_rng():
    torch.manual_seed(8)
    before = torch.get_rng_state().clone()
    old, diagnostic = HistoryCorrection(64, "mlp"), HistoryWidthCorrection(width=4)
    assert torch.equal(before, torch.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, diagnostic.state_dict()[key])
    for projection in old.output_projections:
        torch.nn.init.normal_(projection.weight)
    diagnostic.load_state_dict(old.state_dict())
    hidden = torch.randn(2, 15, 64, requires_grad=True)
    valid = torch.ones(2, 15, dtype=torch.bool); valid[0, 3] = False
    write = valid.clone(); write[1, 5] = False
    a = old(hidden, valid, memory_write_mask=write)
    b = diagnostic(hidden, valid, memory_write_mask=write)
    assert torch.equal(a, b)
    assert torch.equal(torch.autograd.grad(a.sum(), hidden)[0], torch.autograd.grad(b.sum(), hidden)[0])


def test_width16_manual_formula_mask_reset_padding_and_causality():
    torch.manual_seed(42)
    layer = HistoryWidthCorrection()
    hidden = torch.randn(2, 18, 64, requires_grad=True)
    valid = torch.ones(2, 18, dtype=torch.bool); valid[0, [2, 7]] = False
    write = valid.clone(); write[1, 6] = False
    assert torch.equal(layer(hidden, valid, memory_write_mask=write), torch.zeros_like(hidden))
    for projection in layer.output_projections:
        torch.nn.init.normal_(projection.weight)
    reference = torch.zeros_like(hidden)
    # Independent list-based eligibility: padding skips; a withheld valid event resets.
    for batch in range(2):
        previous = []
        for position in range(18):
            if not valid[batch, position]:
                continue
            if not write[batch, position]:
                previous = []
                continue
            for branch, threshold in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
                if len(previous) >= threshold:
                    pair = torch.cat((hidden[batch, position], hidden[batch, previous[-1]]))
                    reference[batch, position] += layer.output_projections[branch](
                        F.gelu(layer.input_projections[branch](pair))) / 8
            previous.append(position)
    actual = layer(hidden, valid, memory_write_mask=write)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=3e-6)
    grad = torch.autograd.grad(actual.sum(), hidden, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(reference.sum(), hidden)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=3e-5, atol=3e-6)
    changed = hidden.detach().clone(); changed[:, 12:] += 4000; changed[~write] = float("nan")
    torch.testing.assert_close(layer(changed, valid, memory_write_mask=write)[:, :12], actual[:, :12])
    assert not actual[~write].any()
    assert not actual[1, 7].any()  # First observed state after reset has no predecessor.
    changed[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        layer(changed, valid, memory_write_mask=write)
    with pytest.raises(ValueError):
        layer(hidden, valid, memory_write_mask=write.float())


@pytest.mark.parametrize("seed", [42, 52, 62])
def test_common_initialization_rng_full_outputs_and_parameter_counts(seed):
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    states, rngs, models, sizes = {}, {}, {}, {}
    for arm in runner.ALL_ARMS:
        torch.manual_seed(seed); models[arm], meta = runner.build_model(data(), arm)
        states[arm] = models[arm].state_dict(); rngs[arm] = torch.get_rng_state().clone()
        sizes[arm] = sum(p.numel() for p in models[arm].parameters())
    assert torch.equal(rngs[runner.BASELINE], rngs[ARM])
    for key, value in states[runner.BASELINE].items():
        if not key.startswith("multilag_detail."):
            assert torch.equal(value, states[ARM][key])
    assert sizes[ARM] - sizes[runner.BASELINE] == 18432
    assert sum(p.numel() for p in models[ARM].multilag_detail.parameters()) == 24576
    assert metadata()["additional_parameter_count"] == 24576
    dt = torch.randint(1, 8, (2, 12)).float(); quantity = torch.randint(1, 20, dt.shape).float()
    mask = torch.ones_like(dt, dtype=torch.bool)
    with torch.no_grad():
        outputs = {arm: target_outputs(model.eval(), dt, mask, quantity, lambda_log_qty=1.)
                   for arm, model in models.items()}
    assert all(torch.equal(outputs[runner.BASELINE][k], outputs[ARM][k]) for k in outputs[ARM])
    initial = runner.initial_states(data(), seed)
    assert initial[ARM] != initial[runner.BASELINE]


def test_full_model_nonzero_correction_hides_target_and_padding():
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    torch.manual_seed(4); model, _ = runner.build_model(data(), ARM); model.eval()
    for projection in model.multilag_detail.output_projections:
        torch.nn.init.normal_(projection.weight, std=.02)
    dt = torch.randint(1, 8, (2, 12)).float(); quantity = torch.randint(1, 20, dt.shape).float()
    mask = torch.ones_like(dt, dtype=torch.bool)
    with torch.no_grad():
        original = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
        changed_quantity = quantity.clone(); changed_quantity[:, -1] = 4000
        changed_dt = dt.clone(); changed_dt[:, -1] = 90
        altered = target_outputs(model, changed_dt, mask, changed_quantity, lambda_log_qty=1.)
        assert torch.equal(original["pred_qty"], altered["pred_qty"])
        padded_dt = torch.cat((torch.full((2, 2), float("nan")), dt), dim=1)
        padded_q = torch.cat((torch.full((2, 2), float("nan")), quantity), dim=1)
        padded_mask = torch.cat((torch.zeros((2, 2), dtype=torch.bool), mask), dim=1)
        padded = target_outputs(model, padded_dt, padded_mask, padded_q, lambda_log_qty=1.)
        torch.testing.assert_close(original["pred_qty"], padded["pred_qty"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("strict", [True, False])
def test_state_load_cannot_cross_width_or_ignore_missing_identity(strict):
    narrow, _ = runner.build_model(data(), runner.BASELINE); wide, _ = runner.build_model(data(), ARM)
    with pytest.raises(ValueError, match="identity"):
        wide.load_state_dict(narrow.state_dict(), strict=strict)
    with pytest.raises(RuntimeError):
        narrow.load_state_dict(wide.state_dict(), strict=strict)
    changed = deepcopy(wide.state_dict()); changed["history_width_identity"][0] ^= 1
    with pytest.raises(ValueError, match="identity"):
        wide.load_state_dict(changed, strict=strict)
    changed = deepcopy(wide.state_dict()); changed["multilag_detail.input_projections.0.weight"] = torch.zeros(4, 128)
    with pytest.raises(RuntimeError, match="tensor mismatch"):
        wide.load_state_dict(changed, strict=strict)
    wide.load_state_dict(wide.state_dict(), strict=strict)


@pytest.mark.parametrize("mutation", ["width", "projection", "role", "head", "split", "monitor", "identity", "shape"])
def test_metadata_route_drift_fails_closed(mutation):
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    value = payload(); validate_checkpoint_route(value, ARM)
    with pytest.raises(ValueError):
        validate_checkpoint_route(value, runner.BASELINE)
    if mutation == "width": value["encoder_config"]["history_mlp_width"] = 4
    elif mutation == "projection": value["encoder_config"]["multilag_detail_rank"] = 4
    elif mutation == "role": value["resume_identity"]["arguments"]["model_role"] = "observed_time_core_ablation_v1"
    elif mutation == "head": value["encoder_config"]["time_head"]["mode"] = "legacy_clamped"
    elif mutation == "split": value["held_out_test_evaluated"] = True
    elif mutation == "monitor": value["resume_identity"]["checkpoint_monitor"] = "validation_joint_objective"
    elif mutation == "identity": value["model_state_dict"]["history_width_identity"][0] ^= 1
    else: value["model_state_dict"]["multilag_detail.input_projections.7.weight"] = torch.zeros(4, 128)
    with pytest.raises(ValueError): validate_checkpoint_route(value, ARM)


def test_hooks_are_idempotent_and_role_loss_constraints_remain():
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import constants, observed_time, training
    runner.install_hooks(); build = factory.build_count_aware_model; runner.install_hooks()
    assert build is factory.build_count_aware_model is training.build_model
    args = SimpleNamespace(model_role=ROLE, time_head_mode="heteroscedastic_lognormal_duration",
        lambda_tail=0., lambda_log_qty=1., quantile_adaptive_strength=0.,
        checkpoint_monitor="validation_raw_quantity_rmse", dataset_contract="intermittent_frozen_5000")
    observed_time.validate_launch(args, (ARM,), (runner.VARIANT,))
    for field, value in (("lambda_log_qty", 2.), ("model_role", "observed_time_core_ablation_v1")):
        changed = deepcopy(args); setattr(changed, field, value)
        with pytest.raises(ValueError): observed_time.validate_launch(changed, (ARM,), (runner.VARIANT,))
    constants.validate_model_role_contract(model_role=ROLE, backbones=(ARM,), quantity_variants=(runner.VARIANT,),
        time_head_mode=args.time_head_mode, lambda_tail=0.)


def test_synthetic_optimizer_check():
    result = runner.synthetic_check("cpu")
    assert result["status"] == "passed" and result["synthetic_optimizer_updates"] == 4
    assert result["real_data_loaded"] is False


def test_synthetic_shared_trainer_checkpoint_and_endpoint_replay(cpu_training, admitted_contract, tmp_path):
    from paper.scripts.count_aware_tpp_backbone import training
    contract = admitted_contract
    contract["training"] = {"maximum_epochs": 300, "minimum_epochs": 40,
                            "patience": 40, "monitor": "validation_raw_quantity_rmse"}
    item = next(d for d in contract["datasets"] if d["dataset_id"] == "intermittent_frozen_5000")
    frame, _ = runner.base.prepare_admitted_data(item)
    interface = runner.time_interface(item, frame, contract)
    initial = runner.initial_states(item, 42)
    args = runner.training_args(contract, item, tmp_path / "train", ARM)
    args.device = "cpu"; args.epochs = args.min_epochs = args.early_stopping_patience = 2
    quantity = {"boundaries": item["quantity_boundaries_all_train_rows"],
                "strata": [{"label": f"bin_{i}"} for i in range(5)]}
    with runner.shared.audited_training(training, item, lambda: None, lambda *_: None):
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone=ARM, quantity_variant=runner.VARIANT, seed=42)
    assert summary["initial_state_sha256"] == initial[ARM]
    directory = args.output_dir / "runs" / ARM / runner.VARIANT / "seed_42"
    history = runner.read(directory / "history.json")["history"]
    resume = training._resume_identity(args=args, backbone=ARM, quantity_variant=runner.VARIANT,
        seed=42, monitor=args.checkpoint_monitor, quantity_contract=quantity, interface_meta=interface)
    for filename, epoch in (("best_val_qty_rmse_model.pt", summary["best_epoch"]), ("last_epoch_state.pt", 2)):
        replay = runner.replay_checkpoint(directory / filename, item, frame, lambda: None, device="cpu",
            expected_arm=ARM, expected_identity=resume, expected_initial=initial[ARM], expected_epoch=epoch)
        assert replay["qty_rmse"] == pytest.approx(history[epoch - 1]["val_qty_rmse"], rel=1e-10, abs=1e-8)
        assert replay["time_nll"] == pytest.approx(history[epoch - 1]["val_time_nll"], rel=1e-10, abs=1e-8)
