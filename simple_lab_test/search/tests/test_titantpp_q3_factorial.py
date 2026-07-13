from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.value_conditioning import apply_transition_loss_scope
from models.Titan import TitanConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.runner import (
    build_run_paths,
    compute_training_loss,
    magnitude_artifact_identity,
    validate_resume_magnitude_identity,
)
from simple_lab_test.search.tpp_experiment import build_long_epoch_config, build_parser
from utils.training import TrainingConfig, eval_next_event_week_lookback


RAW_GLOBAL_MEAN = 6.8458560663480394
RAW_GLOBAL_VAR = 3026.3645310228494
RAW_GLOBAL_STD = 55.0124034288891
RAW_SIGMA_FLOOR = 0.0550124034288891


def make_q3_model(
    gradient_mode: str = "coupled",
    aux_mode: str = "none",
    **overrides,
) -> TitanTPP:
    config = {
        "num_marks": 5,
        "mark_emb_dim": 8,
        "scale_base": 2.0,
        "value_head_activation": "identity",
        "qty_decoder_mode": "direct_raw_qty",
        "magnitude_norm_mode": "causal_shrinkage_revin",
        "magnitude_input_emb_dim": 4,
        "magnitude_encoder_gradient_mode": gradient_mode,
        "magnitude_aux_loss_mode": aux_mode,
        "lambda_log_qty": 0.25,
        "log_qty_huber_delta": 1.0,
        "log_qty_floor": 1.0,
        "magnitude_global_mean": RAW_GLOBAL_MEAN,
        "magnitude_global_var": RAW_GLOBAL_VAR,
        "magnitude_global_std": RAW_GLOBAL_STD,
        "magnitude_sigma_floor": RAW_SIGMA_FLOOR,
        "magnitude_revin_eps": 1e-5,
        "magnitude_shrinkage_k": 8.0,
        "magnitude_center_mode": "mean",
        "magnitude_revin_affine": False,
        "magnitude_stat_context_mode": "none",
        "lambda_magnitude": 1.0,
        "lambda_qty": 0.25,
        "loss_mode": "hybrid",
        "train_loss_scope": "target_only",
        "value_input_mode": "none",
    }
    config.update(overrides)
    titan_cfg = TitanConfig(
        d_model=16,
        n_layers=1,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
        memory_mode="none",
        contextual_mem_size=0,
        persistent_mem_size=0,
        use_lmm=False,
        use_causal=True,
        max_len=32,
    )
    return TitanTPP(RMTPPConfig(**config), titan_cfg)


def example_batch():
    marks = torch.tensor([[4, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.long)
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0], [1.0, 3.0, 1.0, 2.0]])
    values = torch.zeros((2, 4), dtype=torch.float32)
    mask = torch.tensor([[False, True, True, True], [True, True, True, True]])
    return marks, dts, values, mask


def module_grad_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def build_variant(gradient_mode: str, aux_mode: str) -> TitanTPP:
    torch.manual_seed(101)
    return make_q3_model(gradient_mode, aux_mode)


def test_q2_q3_variants_match_parameters_initialization_and_forward() -> None:
    variants = {
        "q2": build_variant("coupled", "none"),
        "q3a": build_variant("detached", "none"),
        "q3b": build_variant("coupled", "log_huber"),
        "q3c": build_variant("detached", "log_huber"),
    }
    reference = variants["q2"]
    reference_state = reference.state_dict()
    marks, dts, values, mask = example_batch()

    with torch.no_grad():
        reference_hidden = reference(marks, dts, values=values, mask=mask)
        reference_direct = reference.predict_direct_magnitude(
            reference_hidden[:, :-1], marks=marks, values=values, mask=mask
        )

        for model in variants.values():
            state = model.state_dict()
            assert state.keys() == reference_state.keys()
            assert sum(tensor.numel() for tensor in state.values()) == sum(
                tensor.numel() for tensor in reference_state.values()
            )
            for name, tensor in reference_state.items():
                assert torch.equal(state[name], tensor), name

            hidden = model(marks, dts, values=values, mask=mask)
            direct = model.predict_direct_magnitude(
                hidden[:, :-1], marks=marks, values=values, mask=mask
            )
            assert torch.equal(hidden, reference_hidden)
            for name in ("normalized", "denormalized", "affine_qty", "qty", "log_qty"):
                assert torch.equal(direct[name], reference_direct[name]), name


@pytest.mark.parametrize(
    "left,right",
    [
        (("coupled", "none"), ("detached", "none")),
        (("coupled", "log_huber"), ("detached", "log_huber")),
    ],
)
def test_factorial_pairs_have_identical_scalar_losses(left, right) -> None:
    marks, dts, values, mask = example_batch()
    left_out = build_variant(*left).nll(marks, dts, values=values, mask=mask)
    right_out = build_variant(*right).nll(marks, dts, values=values, mask=mask)

    for name in (
        "nll",
        "nll_marker",
        "nll_time",
        "magnitude_loss",
        "qty_loss",
        "log_qty_aux_loss",
        "total_loss",
    ):
        assert torch.equal(left_out[name], right_out[name]), name


@pytest.mark.parametrize("loss_name", ["magnitude_loss", "qty_loss", "log_qty_aux_loss"])
def test_detached_magnitude_losses_update_only_magnitude_head(loss_name) -> None:
    model = build_variant("detached", "log_huber")
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)

    model.zero_grad(set_to_none=True)
    out[loss_name].backward()

    assert module_grad_norm(model.magnitude_head) > 0.0
    assert module_grad_norm(model.encoder) == 0.0
    assert module_grad_norm(model.magnitude_input_proj) == 0.0
    assert module_grad_norm(model.mark_head) == 0.0
    assert module_grad_norm(model.v_t) == 0.0


def test_coupled_magnitude_losses_reach_encoder_and_input_projection() -> None:
    model = build_variant("coupled", "log_huber")
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)

    model.zero_grad(set_to_none=True)
    (out["magnitude_loss"] + out["qty_loss"] + out["log_qty_aux_loss"]).backward()

    assert module_grad_norm(model.magnitude_head) > 0.0
    assert module_grad_norm(model.encoder) > 0.0
    assert module_grad_norm(model.magnitude_input_proj) > 0.0
    assert module_grad_norm(model.mark_head) == 0.0
    assert module_grad_norm(model.v_t) == 0.0


def test_marker_time_nll_route_is_unchanged_when_magnitude_is_detached() -> None:
    model = build_variant("detached", "log_huber")
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)

    model.zero_grad(set_to_none=True)
    (out["nll_marker"] + out["nll_time"]).backward()

    assert module_grad_norm(model.mark_head) > 0.0
    assert module_grad_norm(model.v_t) > 0.0
    assert module_grad_norm(model.encoder) > 0.0
    assert module_grad_norm(model.magnitude_input_proj) > 0.0
    assert module_grad_norm(model.magnitude_head) == 0.0


def test_log_auxiliary_matches_formula_and_floor_gradient() -> None:
    model = make_q3_model("coupled", "log_huber")
    affine = torch.tensor([-2.0, 0.5, 2.0, 8.0], requires_grad=True)
    target = torch.tensor([1.0, 2.0, 4.0, 16.0])

    actual = model.log_qty_auxiliary_step(affine, target)
    expected = F.huber_loss(
        torch.log2(affine.clamp_min(1.0)),
        torch.log2(target.clamp_min(1.0)),
        reduction="none",
        delta=1.0,
    )
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    assert affine.grad is not None
    assert affine.grad[0].item() == 0.0
    assert affine.grad[1].item() == 0.0
    assert affine.grad[2].abs().item() > 0.0
    assert affine.grad[3].abs().item() > 0.0


def test_nll_log_auxiliary_uses_the_target_only_transition_mask() -> None:
    model = build_variant("coupled", "log_huber")
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)

    true_qty = torch.exp2(marks[:, 1:].float() + values[:, 1:])
    per_step = model.log_qty_auxiliary_step(out["qty_affine_hat"], true_qty)
    step_mask = apply_transition_loss_scope(
        mask[:, 1:] & mask[:, :-1],
        "target_only",
    )
    expected = (per_step * step_mask).sum() / step_mask.sum().clamp_min(1)

    torch.testing.assert_close(out["log_qty_aux_loss"], expected)
    assert step_mask.sum().item() == marks.size(0)


def test_negative_affine_predictions_still_receive_raw_loss_gradient() -> None:
    model = make_q3_model("coupled", "log_huber")
    marks, dts, values, mask = example_batch()
    with torch.no_grad():
        model.magnitude_head.weight.zero_()
        model.magnitude_head.bias.fill_(-10.0)

    out = model.nll(marks, dts, values=values, mask=mask)
    assert torch.all(out["qty_affine_hat"] < 1.0)

    model.zero_grad(set_to_none=True)
    out["log_qty_aux_loss"].backward()
    assert module_grad_norm(model.magnitude_head) == 0.0

    out = model.nll(marks, dts, values=values, mask=mask)
    model.zero_grad(set_to_none=True)
    (out["magnitude_loss"] + out["qty_loss"]).backward()
    assert module_grad_norm(model.magnitude_head) > 0.0


def test_total_loss_and_evaluator_keep_auxiliary_separate_from_nll() -> None:
    model = build_variant("coupled", "log_huber")
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)
    expected_total = (
        out["marker_train_loss"]
        + out["nll_time"]
        + model.cfg.lambda_magnitude * out["magnitude_loss"]
        + model.cfg.lambda_qty * out["qty_loss"]
        + model.cfg.lambda_log_qty * out["log_qty_aux_loss"]
    )

    torch.testing.assert_close(out["nll"], out["nll_marker"] + out["nll_time"])
    torch.testing.assert_close(out["total_loss"], expected_total)
    torch.testing.assert_close(
        compute_training_loss(
            model=model,
            out=out,
            training_cfg=TrainingConfig(device="cpu", lambda_dt=1.0),
        ),
        expected_total,
    )

    loader = [(marks, dts, mask, torch.arange(marks.size(0)), values)]
    metrics = eval_next_event_week_lookback(
        model,
        loader,
        "cpu",
        target_only_nll=True,
    )
    assert metrics["val_log_qty_aux_loss"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["val_log_qty_aux_loss"]))


def test_q3_cli_path_and_artifact_identity_are_distinct(tmp_path) -> None:
    args = build_parser().parse_args([
        "long-epoch",
        "--datasets",
        "intermittent",
        "--models",
        "titantpp",
        "--qty-decoder-mode",
        "direct_raw_qty",
        "--magnitude-norm-mode",
        "causal_shrinkage_revin",
        "--magnitude-encoder-gradient-mode",
        "detached",
        "--magnitude-aux-loss-mode",
        "log_huber",
        "--split-mode",
        "fixed",
        "--train-loss-scope",
        "target_only",
        "--loss-mode",
        "hybrid",
    ])
    cfg = replace(build_long_epoch_config(args), base_dir=str(tmp_path))
    run_cfg = RunConfig(
        dataset_name="intermittent",
        dataset_kind="marked_target",
        model_name="titantpp",
        candidate_name="small_lmm",
        candidate=SimpleNamespace(name="small_lmm"),
        seed=42,
        epochs=1,
        scale_base=2.0,
        titan_profile="dataset_best",
    )
    run_path = str(build_run_paths(cfg, run_cfg).run_dir)
    identity = magnitude_artifact_identity(cfg)

    assert "magencgrad_detached" in run_path
    assert "magaux_log_huber" in run_path
    assert "lambdalogqty_0p25" in run_path
    assert "logqtydelta_1" in run_path
    assert "logqtyfloor_1" in run_path
    assert identity["magnitude_encoder_gradient_mode"] == "detached"
    assert identity["magnitude_aux_loss_mode"] == "log_huber"
    assert identity["lambda_log_qty"] == pytest.approx(0.25)


def test_legacy_q2_state_and_resume_identity_remain_loadable() -> None:
    q2 = build_variant("coupled", "none")
    q3c = build_variant("detached", "log_huber")
    q3c.load_state_dict(q2.state_dict(), strict=True)

    legacy_cfg = asdict(q2.cfg)
    for name in (
        "magnitude_encoder_gradient_mode",
        "magnitude_aux_loss_mode",
        "lambda_log_qty",
        "log_qty_huber_delta",
        "log_qty_floor",
    ):
        legacy_cfg.pop(name)
    payload = {"rmtpp_config": legacy_cfg}

    validate_resume_magnitude_identity(resume_payload=payload, current_cfg=q2.cfg)
    with pytest.raises(ValueError, match="magnitude_encoder_gradient_mode"):
        validate_resume_magnitude_identity(resume_payload=payload, current_cfg=q3c.cfg)


@pytest.mark.parametrize(
    "gradient_mode,aux_mode,overrides,match",
    [
        (
            "detached",
            "none",
            {"qty_decoder_mode": "direct_log_qty", "magnitude_norm_mode": "global"},
            "direct_raw_qty",
        ),
        ("detached", "none", {"magnitude_norm_mode": "global"}, "causal_shrinkage_revin"),
        ("detached", "none", {"lambda_log_qty": 0.5}, "frozen"),
        ("coupled", "log_huber", {"lambda_log_qty": 0.5}, "frozen"),
        ("coupled", "log_huber", {"log_qty_floor": 0.5}, "frozen"),
    ],
)
def test_invalid_q3_mixed_contracts_fail_fast(
    gradient_mode,
    aux_mode,
    overrides,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        make_q3_model(gradient_mode, aux_mode, **overrides)
