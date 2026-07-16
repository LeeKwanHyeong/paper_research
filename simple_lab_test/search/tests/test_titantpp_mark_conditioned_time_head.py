from types import SimpleNamespace

import pytest
import torch

from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.config import RMTPPConfig
from models.Titan import TitanConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.runner import build_run_paths
from simple_lab_test.search.tpp_experiment import build_long_epoch_config, build_parser


def make_model(
    time_head_mode: str,
    *,
    value_head_mode: str = "shared",
    qty_mark_gradient_mode: str = "coupled",
    value_encoder_gradient_mode: str = "coupled",
    qty_decoder_mode: str = "mark_residual",
    marker_loss_mode: str = "ce",
    lambda_ordinal: float = 0.0,
) -> TitanTPP:
    rmtpp_cfg = RMTPPConfig(
        num_marks=5,
        mark_emb_dim=8,
        rnn_hidden_dim=16,
        scale_base=10.0,
        value_head_activation="identity",
        value_head_mode=value_head_mode,
        time_head_mode=time_head_mode,
        qty_mark_gradient_mode=qty_mark_gradient_mode,
        value_encoder_gradient_mode=value_encoder_gradient_mode,
        qty_decoder_mode=qty_decoder_mode,
        marker_loss_mode=marker_loss_mode,
        lambda_ordinal=lambda_ordinal,
        value_input_mode="residual",
        loss_mode="hybrid",
        train_loss_scope="target_only",
    )
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
        mem_size=8,
        mem_topk=2,
        use_causal=True,
        max_len=32,
    )
    return TitanTPP(rmtpp_cfg, titan_cfg)


def example_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    marks = torch.tensor(
        [[4, 4, 0, 1, 2], [4, 1, 0, 2, 3]],
        dtype=torch.long,
    )
    dts = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0, 1.0], [0.0, 3.0, 1.0, 2.0, 4.0]],
    )
    values = torch.tensor(
        [[0.0, 0.0, 0.1, 0.2, 0.3], [0.0, 0.2, 0.1, 0.3, 0.4]],
    )
    mask = torch.tensor(
        [[False, False, True, True, True], [False, True, True, True, True]],
    )
    return marks, dts, values, mask


def module_grad_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_zero_initialized_v4_matches_shared_time_control_exactly() -> None:
    torch.manual_seed(7)
    shared = make_model("shared")
    torch.manual_seed(7)
    conditioned = make_model("mark_conditioned")

    conditioned_state = conditioned.state_dict()
    for name, value in shared.state_dict().items():
        assert name in conditioned_state
        assert torch.equal(value, conditioned_state[name]), name

    marks, dts, values, mask = example_batch()
    shared_out = shared.nll(marks, dts, values=values, mask=mask)
    conditioned_out = conditioned.nll(marks, dts, values=values, mask=mask)
    for key in (
        "nll",
        "nll_marker",
        "nll_time",
        "value_loss",
        "qty_loss",
        "total_loss",
    ):
        assert torch.equal(shared_out[key], conditioned_out[key]), key

    hidden = shared.forward(marks, dts, values=values, mask=mask)[:, -2]
    u = torch.full((hidden.size(0),), 0.5)
    target_marks = marks[:, -1]
    assert torch.equal(
        shared.sample_next_dt(hidden, u=u),
        conditioned.sample_next_dt(hidden, u=u, marks=target_marks),
    )


def test_conditional_log_density_requires_observed_real_marks() -> None:
    model = make_model("mark_conditioned")
    hidden = torch.randn(2, 16)
    delta_time = torch.ones(2)

    with pytest.raises(ValueError, match="requires observed marks"):
        model.log_f_dt(hidden, delta_time)
    with pytest.raises(ValueError, match="cannot include PAD"):
        model.log_f_dt(hidden, delta_time, marks=torch.tensor([0, 4]))


def test_time_loss_updates_only_selected_delta_rows_and_not_mark_head() -> None:
    torch.manual_seed(11)
    model = make_model("mark_conditioned")
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)

    model.zero_grad(set_to_none=True)
    out["nll_time"].backward()

    delta_grad = model.time_mark_delta_head.weight.grad
    assert delta_grad is not None
    assert torch.count_nonzero(delta_grad[0]) == 0
    assert torch.count_nonzero(delta_grad[1]) == 0
    assert torch.count_nonzero(delta_grad[2]) > 0
    assert torch.count_nonzero(delta_grad[3]) > 0
    assert module_grad_norm(model.mark_head) == 0.0
    assert module_grad_norm(model.v_t) > 0.0
    assert module_grad_norm(model.encoder) > 0.0


def test_left_padding_and_all_masked_batches_are_finite() -> None:
    model = make_model("mark_conditioned")
    marks, dts, values, mask = example_batch()

    left_padded = model.nll(
        marks,
        dts,
        values=values,
        mask=mask,
        loss_scope="all",
    )
    all_masked = model.nll(
        marks,
        dts,
        values=values,
        mask=torch.zeros_like(mask),
        loss_scope="all",
    )

    for output in (left_padded, all_masked):
        for key in ("nll", "nll_marker", "nll_time", "total_loss"):
            assert torch.isfinite(output[key]), key
    assert int(all_masked["steps"].item()) == 0
    assert model.time_mark_delta_head.out_features == model.num_real_marks


def test_conditional_density_matches_closed_form_rmtpp() -> None:
    model = make_model("mark_conditioned")
    with torch.no_grad():
        model.v_t.weight.zero_()
        model.b_t.fill_(0.2)
        model.time_mark_delta_head.weight.zero_()
        model.time_mark_delta_head.bias.copy_(torch.tensor([0.0, 0.4, -0.2, 0.1]))

    hidden = torch.zeros(2, 16)
    delta_time = torch.tensor([1.5, 2.0])
    marks = torch.tensor([1, 2])
    actual = model.log_f_dt(hidden, delta_time, marks=marks)

    w = model._w_pos()
    intercept = torch.tensor([0.6, 0.0])
    wd = torch.clamp(w * delta_time, max=10.0)
    expected = intercept + wd - (torch.exp(intercept) / w) * torch.expm1(wd)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_sampling_uses_explicit_or_predicted_real_mark_and_is_monotonic() -> None:
    model = make_model("mark_conditioned")
    with torch.no_grad():
        model.v_t.weight.zero_()
        model.b_t.zero_()
        model.time_mark_delta_head.weight.zero_()
        model.time_mark_delta_head.bias.copy_(torch.tensor([0.0, 0.8, -0.2, 0.1]))
        model.mark_head.weight.zero_()
        model.mark_head.bias.copy_(torch.tensor([0.0, 5.0, 0.0, 0.0, 100.0]))

    hidden = torch.zeros(2, 16)
    u = torch.full((2,), 0.5)
    explicit = model.sample_next_dt(hidden, u=u, marks=torch.tensor([0, 1]))
    predicted = model.sample_next_dt(hidden, u=u)
    mark_one = model.sample_next_dt(hidden, u=u, marks=torch.ones(2, dtype=torch.long))

    assert explicit[1] < explicit[0]
    assert torch.equal(predicted, mark_one)


def test_v4b_route_is_supported_and_unplanned_combinations_fail() -> None:
    v4b = make_model(
        "mark_conditioned",
        value_head_mode="mark_conditioned_experts",
        qty_mark_gradient_mode="detached",
    )
    assert v4b.time_head_mode == "mark_conditioned"

    with pytest.raises(ValueError, match="supports only the V4a"):
        make_model(
            "mark_conditioned",
            value_head_mode="mark_conditioned_experts",
            qty_mark_gradient_mode="coupled",
        )
    with pytest.raises(ValueError, match="requires qty_decoder_mode"):
        make_model("mark_conditioned", qty_decoder_mode="direct_log_qty")


def test_cli_and_run_identity_separate_v4_validation_only(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "long-epoch",
            "--models",
            "titantpp",
            "--split-mode",
            "fixed",
            "--evaluation-scope",
            "validation_only",
            "--time-head-mode",
            "mark_conditioned",
        ]
    )
    cfg = build_long_epoch_config(args)
    assert cfg.time_head_mode == "mark_conditioned"
    assert cfg.evaluation_scope == "validation_only"

    run_cfg = RunConfig(
        dataset_name="yellow_trip_hourly",
        dataset_kind="yellow_trip_hourly",
        model_name="titantpp",
        candidate_name="mid_lmm",
        candidate=SimpleNamespace(name="mid_lmm"),
        seed=42,
        epochs=50,
        scale_base=10.0,
        titan_profile="dataset_best",
    )
    path = build_run_paths(
        ExperimentConfig(
            base_dir=str(tmp_path),
            split_mode="fixed",
            evaluation_scope="validation_only",
            time_head_mode="mark_conditioned",
        ),
        run_cfg,
    ).run_dir
    assert "evalscope_validation_only" in str(path)
    assert "timehead_mark_conditioned" in str(path)


def test_cli_rejects_non_titan_and_unlocked_internal_validation_only() -> None:
    non_titan = build_parser().parse_args(
        [
            "long-epoch",
            "--models",
            "rmtpp",
            "--time-head-mode",
            "mark_conditioned",
        ]
    )
    with pytest.raises(ValueError, match="TitanTPP-only"):
        build_long_epoch_config(non_titan)

    internal = build_parser().parse_args(
        [
            "long-epoch",
            "--models",
            "titantpp",
            "--evaluation-scope",
            "validation_only",
        ]
    )
    with pytest.raises(ValueError, match="requires --split-mode fixed"):
        build_long_epoch_config(internal)
