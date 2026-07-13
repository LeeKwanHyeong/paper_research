import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.magnitude_normalization import (
    build_causal_revin_magnitude_context,
    build_causal_shrinkage_revin_magnitude_context,
    build_raw_global_magnitude_context,
    denormalize_magnitude,
    normalized_magnitude_target,
    reconstruct_raw_quantity,
)
from models.Titan import TitanConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.runner import (
    attach_train_global_magnitude_stats,
    build_run_paths,
    cached_run_is_complete,
    scale_metric_paths,
    test_metric_paths as runner_test_metric_paths,
    test_scale_metric_paths as runner_test_scale_metric_paths,
    validate_resume_magnitude_identity,
)
from simple_lab_test.search.tpp_experiment import build_long_epoch_config, build_parser
from utils.training import eval_next_event_week_lookback


RAW_GLOBAL_MEAN = 6.8458560663480394
RAW_GLOBAL_VAR = 3026.3645310228494
RAW_GLOBAL_STD = 55.0124034288891
RAW_SIGMA_FLOOR = 0.0550124034288891


def make_model(
    norm_mode: str = "global",
    *,
    memory_mode: str = "none",
    **overrides,
) -> TitanTPP:
    config = {
        "num_marks": 5,
        "mark_emb_dim": 8,
        "scale_base": 2.0,
        "value_head_activation": "identity",
        "qty_decoder_mode": "direct_raw_qty",
        "magnitude_norm_mode": norm_mode,
        "magnitude_input_emb_dim": 4,
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
        memory_mode=memory_mode,
        contextual_mem_size=4 if "ttm" in memory_mode else 0,
        persistent_mem_size=0,
        use_lmm=memory_mode in {"static_lmm", "hybrid_lmm_ttm"},
        mem_size=8,
        mem_topk=2,
        use_causal=True,
        max_len=32,
    )
    return TitanTPP(RMTPPConfig(**config), titan_cfg)


def example_batch():
    marks = torch.tensor([[4, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.long)
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0], [1.0, 3.0, 1.0, 2.0]])
    values = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[False, True, True, True], [True, True, True, True]])
    return marks, dts, values, mask


def module_grad_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_raw_reconstruction_and_q0_round_trip() -> None:
    marks = torch.tensor([[0, 1, 2]])
    values = torch.tensor([[0.0, 0.0, 0.0]])
    mask = torch.ones_like(marks, dtype=torch.bool)
    raw_qty = reconstruct_raw_quantity(marks, values, num_real_marks=3)
    context = build_raw_global_magnitude_context(
        marks,
        values,
        mask,
        num_real_marks=3,
        global_mean=3.0,
        global_std=2.0,
        sigma_floor=0.1,
    )
    normalized = normalized_magnitude_target(raw_qty, context)

    assert torch.equal(raw_qty, torch.tensor([[1.0, 2.0, 4.0]]))
    assert torch.allclose(context.center, torch.tensor([[3.0]]))
    assert torch.allclose(context.scale, torch.tensor([[2.0]]))
    assert torch.allclose(denormalize_magnitude(normalized, context), raw_qty)


def test_q1_masked_population_moments_and_one_event_scale() -> None:
    marks = torch.tensor([[1, 2, 4], [1, 2, 3]])
    values = torch.zeros_like(marks, dtype=torch.float32)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    context = build_causal_revin_magnitude_context(
        marks,
        values,
        mask,
        num_real_marks=4,
        revin_eps=1e-5,
    )

    assert torch.equal(context.context_count, torch.tensor([[1], [2]]))
    assert context.center[0, 0].item() == pytest.approx(2.0)
    assert context.scale[0, 0].item() == pytest.approx(1e-5 ** 0.5)
    assert context.center[1, 0].item() == pytest.approx(3.0)
    assert context.scale[1, 0].item() == pytest.approx((1.0 + 1e-5) ** 0.5)


def test_q2_mixes_first_and_second_moments_and_has_expected_limits() -> None:
    marks = torch.tensor([[1, 2, 3]])
    values = torch.zeros_like(marks, dtype=torch.float32)
    mask = torch.ones_like(marks, dtype=torch.bool)
    context = build_causal_shrinkage_revin_magnitude_context(
        marks,
        values,
        mask,
        num_real_marks=4,
        global_mean=10.0,
        global_var=9.0,
        sigma_floor=0.1,
        shrinkage_k=2.0,
    )

    assert context.center.item() == pytest.approx(6.5)
    assert context.scale.item() == pytest.approx(17.25 ** 0.5)

    history_limit = build_causal_shrinkage_revin_magnitude_context(
        marks,
        values,
        mask,
        num_real_marks=4,
        global_mean=10.0,
        global_var=9.0,
        sigma_floor=1e-6,
        shrinkage_k=1e-7,
    )
    global_limit = build_causal_shrinkage_revin_magnitude_context(
        marks,
        values,
        mask,
        num_real_marks=4,
        global_mean=10.0,
        global_var=9.0,
        sigma_floor=1e-6,
        shrinkage_k=1e9,
    )
    assert history_limit.center.item() == pytest.approx(3.0, abs=1e-5)
    assert history_limit.scale.item() == pytest.approx(1.0, abs=1e-5)
    assert global_limit.center.item() == pytest.approx(10.0, abs=1e-5)
    assert global_limit.scale.item() == pytest.approx(3.0, abs=1e-5)


@pytest.mark.parametrize("norm_mode", ["global", "causal_revin", "causal_shrinkage_revin"])
def test_target_and_padding_mutations_do_not_change_context_or_prediction(norm_mode) -> None:
    torch.manual_seed(7)
    model = make_model(norm_mode).eval()
    marks, dts, values, mask = example_batch()
    changed_marks = marks.clone()
    changed_dts = dts.clone()
    changed_values = values.clone()
    changed_marks[:, -1] = torch.tensor([0, 1])
    changed_values[:, -1] = torch.tensor([0.9, 0.8])
    changed_marks[0, 0] = 2
    changed_dts[0, 0] = 999.0
    changed_values[0, 0] = 20.0

    context = model.build_magnitude_context(marks, values, mask)
    changed_context = model.build_magnitude_context(changed_marks, changed_values, mask)
    assert torch.equal(context.history_mask, changed_context.history_mask)
    assert torch.equal(context.context_count, changed_context.context_count)
    assert torch.allclose(context.center, changed_context.center)
    assert torch.allclose(context.scale, changed_context.scale)
    assert torch.allclose(context.normalized_history, changed_context.normalized_history)

    with torch.no_grad():
        hidden = model(marks, dts, values=values, mask=mask)[:, -2]
        changed_hidden = model(
            changed_marks,
            changed_dts,
            values=changed_values,
            mask=mask,
        )[:, -2]
        prediction = model.predict_direct_magnitude(
            hidden, marks=marks, values=values, mask=mask
        )["qty"]
        changed_prediction = model.predict_direct_magnitude(
            changed_hidden,
            marks=changed_marks,
            values=changed_values,
            mask=mask,
        )["qty"]
    assert torch.allclose(hidden, changed_hidden, atol=1e-6)
    assert torch.allclose(prediction, changed_prediction, atol=1e-6)


@pytest.mark.parametrize("norm_mode", ["global", "causal_revin", "causal_shrinkage_revin"])
def test_left_and_right_padding_have_equivalent_context_statistics(norm_mode) -> None:
    model = make_model(norm_mode)
    left_marks = torch.tensor([[4, 1, 2, 3]])
    right_marks = torch.tensor([[1, 2, 3, 4]])
    values = torch.zeros((1, 4))
    left_mask = torch.tensor([[False, True, True, True]])
    right_mask = torch.tensor([[True, True, True, False]])

    left = model.build_magnitude_context(left_marks, values, left_mask)
    right = model.build_magnitude_context(right_marks, values, right_mask)
    left_history = left.normalized_history[left.history_mask]
    right_history = right.normalized_history[right.history_mask]

    assert torch.equal(left.context_count, right.context_count)
    assert torch.allclose(left.center, right.center)
    assert torch.allclose(left.scale, right.scale)
    assert torch.allclose(left_history, right_history)


def test_q0_q1_q2_have_identical_parameters_and_seeded_initialization() -> None:
    states = {}
    for norm_mode in ("global", "causal_revin", "causal_shrinkage_revin"):
        torch.manual_seed(101)
        states[norm_mode] = {
            name: value.detach().clone()
            for name, value in make_model(norm_mode).state_dict().items()
        }

    reference = states["global"]
    for state in states.values():
        assert state.keys() == reference.keys()
        assert sum(tensor.numel() for tensor in state.values()) == sum(
            tensor.numel() for tensor in reference.values()
        )
        for name in reference:
            assert torch.equal(state[name], reference[name]), name


def test_raw_prediction_is_independent_of_marker_head() -> None:
    torch.manual_seed(17)
    model = make_model("causal_shrinkage_revin").eval()
    marks, dts, values, mask = example_batch()
    with torch.no_grad():
        hidden = model(marks, dts, values=values, mask=mask)[:, -2]
        before = model.predict_direct_magnitude(
            hidden, marks=marks, values=values, mask=mask
        )["qty"]
        model.mark_head.weight.fill_(1000.0)
        model.mark_head.bias.fill_(-1000.0)
        after = model.predict_direct_magnitude(
            hidden, marks=marks, values=values, mask=mask
        )["qty"]
    assert torch.equal(before, after)


def test_raw_losses_and_likelihood_heads_have_expected_gradient_routes() -> None:
    torch.manual_seed(23)
    model = make_model("causal_shrinkage_revin")
    marks, dts, values, mask = example_batch()

    out = model.nll(marks, dts, values=values, mask=mask)
    model.zero_grad(set_to_none=True)
    out["magnitude_loss"].backward()
    assert module_grad_norm(model.magnitude_head) > 0.0
    assert module_grad_norm(model.encoder) > 0.0
    assert module_grad_norm(model.magnitude_input_proj) > 0.0
    assert module_grad_norm(model.mark_head) == 0.0
    assert module_grad_norm(model.v_t) == 0.0

    out = model.nll(marks, dts, values=values, mask=mask)
    model.zero_grad(set_to_none=True)
    (out["nll_marker"] + out["nll_time"]).backward()
    assert module_grad_norm(model.mark_head) > 0.0
    assert module_grad_norm(model.v_t) > 0.0
    assert module_grad_norm(model.magnitude_head) == 0.0


def test_negative_affine_quantity_keeps_quantity_loss_gradient() -> None:
    model = make_model("global")
    marks, dts, values, mask = example_batch()
    with torch.no_grad():
        model.magnitude_head.weight.zero_()
        model.magnitude_head.bias.fill_(-10.0)
    out = model.nll(marks, dts, values=values, mask=mask)

    assert torch.all(out["qty_affine_hat"] < 0.0)
    assert torch.all(out["qty_hat"] == 0.0)
    model.zero_grad(set_to_none=True)
    out["qty_loss"].backward()
    assert module_grad_norm(model.magnitude_head) > 0.0


def test_raw_train_global_stats_use_train_only_and_freeze_floor() -> None:
    frame = pl.DataFrame({
        "chronological_split": ["train", "train", "validation", "test"],
        "mark": [0, 1, 3, 3],
        "scale_residual": [0.0, 0.0, 0.0, 0.0],
    })
    cfg = ExperimentConfig(
        base_dir="/tmp/q-raw-test",
        qty_decoder_mode="direct_raw_qty",
        magnitude_norm_mode="causal_shrinkage_revin",
    )
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
    meta = attach_train_global_magnitude_stats(
        marked_df=frame,
        marked_meta={"num_marks": 5},
        cfg=cfg,
        run_cfg=run_cfg,
    )

    assert meta["magnitude_domain"] == "raw_qty"
    assert meta["magnitude_train_event_count"] == 2
    assert meta["magnitude_global_mean"] == pytest.approx(1.5)
    assert meta["magnitude_global_var"] == pytest.approx(0.25)
    assert meta["magnitude_global_std"] == pytest.approx(0.5)
    assert meta["magnitude_sigma_floor"] == pytest.approx(0.0005)


def test_q2_cli_and_run_path_preserve_full_artifact_identity(tmp_path) -> None:
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
        "--magnitude-shrinkage-k",
        "8",
        "--split-mode",
        "fixed",
        "--train-loss-scope",
        "target_only",
        "--loss-mode",
        "hybrid",
    ])
    cfg = build_long_epoch_config(args)
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
    cfg = replace(cfg, base_dir=str(tmp_path))
    run_path = str(build_run_paths(cfg, run_cfg).run_dir)

    assert cfg.qty_decoder_mode == "direct_raw_qty"
    assert cfg.magnitude_norm_mode == "causal_shrinkage_revin"
    assert cfg.magnitude_shrinkage_k == pytest.approx(8.0)
    assert "qtydecoder_direct_raw_qty" in run_path
    assert "magnorm_causal_shrinkage_revin" in run_path
    assert "domain_raw_qty" in run_path
    assert "k_8" in run_path


def test_cache_and_resume_reject_stale_raw_normalization_identity(tmp_path) -> None:
    cfg = ExperimentConfig(
        base_dir=str(tmp_path),
        qty_decoder_mode="direct_raw_qty",
        magnitude_norm_mode="causal_shrinkage_revin",
        loss_mode="hybrid",
        split_mode="fixed",
        train_loss_scope="target_only",
    )
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
    paths = build_run_paths(cfg, run_cfg)
    summary = {
        "epochs": 1,
        "test_time_memory": "none",
        "split_mode": "fixed",
        "value_head_activation": "sigmoid",
        "value_head_mode": "shared",
        "qty_mark_gradient_mode": "coupled",
        "value_encoder_gradient_mode": "coupled",
        "marker_loss_mode": "ce",
        "lambda_ordinal": 0.0,
        "qty_decoder_mode": "direct_raw_qty",
        "magnitude_norm_mode": "causal_shrinkage_revin",
        "magnitude_input_emb_dim": 8,
        "lambda_magnitude": 1.0,
        "magnitude_sigma_floor": cfg.magnitude_sigma_floor,
        "magnitude_effective_sigma_floor": RAW_SIGMA_FLOOR,
        "magnitude_revin_eps": 1e-5,
        "magnitude_shrinkage_k": 8.0,
        "magnitude_center_mode": "mean",
        "magnitude_revin_affine": False,
        "magnitude_stat_context_mode": "none",
        "magnitude_exp_clamp_min": -2.0,
        "magnitude_exp_clamp_max": 15.0,
        "magnitude_global_mean": RAW_GLOBAL_MEAN,
        "magnitude_global_var": RAW_GLOBAL_VAR,
        "magnitude_global_std": RAW_GLOBAL_STD,
    }
    (paths.metrics_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (paths.metrics_dir / "history.json").write_text('{"history": []}', encoding="utf-8")
    (paths.checkpoint_dir / "best_val_nll_model.pt").touch()
    for selection in cfg.eval_selections:
        for path in scale_metric_paths(paths, selection):
            path.touch()
        for path in runner_test_metric_paths(paths, selection):
            path.touch()
        for path in runner_test_scale_metric_paths(paths, selection):
            path.touch()
    effective_meta = {
        "magnitude_global_mean": RAW_GLOBAL_MEAN,
        "magnitude_global_var": RAW_GLOBAL_VAR,
        "magnitude_global_std": RAW_GLOBAL_STD,
        "magnitude_sigma_floor": RAW_SIGMA_FLOOR,
    }
    assert cached_run_is_complete(
        cfg=cfg,
        run_cfg=run_cfg,
        run_paths=paths,
        effective_marked_meta=effective_meta,
    )
    assert not cached_run_is_complete(
        cfg=cfg,
        run_cfg=run_cfg,
        run_paths=paths,
        effective_marked_meta={**effective_meta, "magnitude_global_mean": RAW_GLOBAL_MEAN + 1.0},
    )

    model = make_model("causal_shrinkage_revin")
    payload = {"rmtpp_config": asdict(model.cfg)}
    validate_resume_magnitude_identity(resume_payload=payload, current_cfg=model.cfg)
    payload["rmtpp_config"]["magnitude_shrinkage_k"] = 4.0
    with pytest.raises(ValueError, match="magnitude_shrinkage_k"):
        validate_resume_magnitude_identity(resume_payload=payload, current_cfg=model.cfg)


def test_raw_weekly_evaluator_exports_contract_metrics() -> None:
    torch.manual_seed(31)
    model = make_model("causal_shrinkage_revin").eval()
    marks, dts, values, mask = example_batch()
    loader = [(marks, dts, mask, torch.arange(marks.size(0)), values)]
    metrics = eval_next_event_week_lookback(
        model,
        loader,
        "cpu",
        target_only_nll=True,
    )

    for name in (
        "qty_mae",
        "qty_rmse",
        "qty_wape",
        "log_qty_mae",
        "log_qty_rmse",
        "preclamp_negative_share",
        "magnitude_center_p50",
        "magnitude_scale_p50",
        "normalized_target_abs_p99",
        "context_1_qty_mae",
        "context_2_4_qty_mae",
    ):
        assert torch.isfinite(torch.tensor(metrics[name])), name
    assert metrics["normalized_target_nonfinite_count"] == 0
    assert metrics["context_1_count"] == 0
    assert metrics["context_2_4_count"] == 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"train_loss_scope": "all"}, "target_only"),
        ({"marker_loss_mode": "ce_rps", "lambda_ordinal": 0.1}, "plain marker CE"),
        ({"magnitude_revin_affine": True}, "magnitude_revin_affine=False"),
        ({"magnitude_stat_context_mode": "stats"}, "magnitude_stat_context_mode='none'"),
    ],
)
def test_invalid_raw_contracts_fail_fast(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        make_model("global", **kwargs)


def test_contextual_ttm_is_rejected_for_direct_raw_quantity() -> None:
    with pytest.raises(ValueError, match="contextual TTM"):
        make_model("causal_revin", memory_mode="contextual_ttm")
