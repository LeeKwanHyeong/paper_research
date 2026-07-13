from types import SimpleNamespace

import polars as pl
import pytest
import torch

from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.magnitude_normalization import (
    build_global_magnitude_context,
    denormalize_magnitude,
    normalized_magnitude_target,
    reconstruct_log2_quantity,
)
from models.Titan import TitanConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.runner import (
    aggregate_test_metrics,
    attach_train_global_magnitude_stats,
    build_run_paths,
    compute_training_loss,
)
from simple_lab_test.search.tpp_experiment import build_long_epoch_config, build_parser
from utils.training import TrainingConfig, eval_next_event_week_lookback


def make_model(**overrides) -> TitanTPP:
    config = {
        "num_marks": 5,
        "mark_emb_dim": 8,
        "scale_base": 2.0,
        "value_head_activation": "identity",
        "qty_decoder_mode": "direct_log_qty",
        "magnitude_norm_mode": "global",
        "magnitude_input_emb_dim": 4,
        "magnitude_global_mean": 1.5,
        "magnitude_global_std": 0.75,
        "magnitude_sigma_floor": 1e-3,
        "lambda_magnitude": 1.25,
        "loss_mode": "hybrid",
        "train_loss_scope": "target_only",
        "value_input_mode": "none",
    }
    config.update(overrides)
    rmtpp_cfg = RMTPPConfig(**config)
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


def example_batch():
    marks = torch.tensor([[4, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.long)
    dts = torch.tensor([[0.0, 1.0, 2.0, 1.0], [1.0, 3.0, 1.0, 2.0]])
    values = torch.tensor([[0.0, 0.1, 0.2, 0.3], [0.2, 0.1, 0.3, 0.4]])
    mask = torch.tensor([[False, True, True, True], [True, True, True, True]])
    return marks, dts, values, mask


def module_grad_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.abs().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_log2_factorization_and_global_round_trip() -> None:
    marks = torch.tensor([[0, 1, 2]])
    values = torch.tensor([[0.25, 0.5, 0.75]])
    mask = torch.ones_like(marks, dtype=torch.bool)
    log_qty = reconstruct_log2_quantity(marks, values, num_real_marks=3)
    context = build_global_magnitude_context(
        marks,
        values,
        mask,
        num_real_marks=3,
        global_mean=1.0,
        global_std=2.0,
        sigma_floor=1e-3,
    )
    normalized = normalized_magnitude_target(log_qty, context)

    assert torch.equal(log_qty, torch.tensor([[0.25, 1.5, 2.75]]))
    assert torch.allclose(denormalize_magnitude(normalized, context), log_qty)


def test_magnitude_context_excludes_appended_target_and_padding() -> None:
    model = make_model().eval()
    marks, _, values, mask = example_batch()
    changed_marks = marks.clone()
    changed_values = values.clone()
    changed_marks[:, -1] = torch.tensor([0, 1])
    changed_values[:, -1] = torch.tensor([0.91, 0.82])

    original = model.build_magnitude_context(marks, values, mask)
    changed = model.build_magnitude_context(changed_marks, changed_values, mask)

    assert torch.equal(original.history_mask, changed.history_mask)
    assert torch.equal(original.context_count, torch.tensor([[2], [3]]))
    assert torch.equal(original.normalized_history, changed.normalized_history)
    assert original.normalized_history[0, 0].item() == 0.0
    assert torch.all(original.normalized_history[:, -1] == 0.0)


def test_target_mutation_does_not_change_next_quantity_prediction() -> None:
    torch.manual_seed(7)
    model = make_model().eval()
    marks, dts, values, mask = example_batch()
    changed_marks = marks.clone()
    changed_values = values.clone()
    changed_marks[:, -1] = torch.tensor([0, 1])
    changed_values[:, -1] = torch.tensor([0.91, 0.82])

    with torch.no_grad():
        hidden = model.forward(marks, dts, values=values, mask=mask)[:, -2]
        changed_hidden = model.forward(
            changed_marks,
            dts,
            values=changed_values,
            mask=mask,
        )[:, -2]
        prediction = model.predict_direct_magnitude(
            hidden,
            marks=marks,
            values=values,
            mask=mask,
        )["qty"]
        changed_prediction = model.predict_direct_magnitude(
            changed_hidden,
            marks=changed_marks,
            values=changed_values,
            mask=mask,
        )["qty"]

    assert torch.allclose(hidden, changed_hidden, atol=1e-6)
    assert torch.allclose(prediction, changed_prediction, atol=1e-6)


def test_direct_loss_is_finite_and_composed_exactly_once() -> None:
    torch.manual_seed(11)
    model = make_model()
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)
    training_cfg = TrainingConfig(lambda_dt=0.75)
    loss = compute_training_loss(model=model, out=out, training_cfg=training_cfg)
    expected = (
        out["marker_train_loss"]
        + 0.75 * out["nll_time"]
        + model.cfg.lambda_magnitude * out["magnitude_loss"]
        + model.cfg.lambda_qty * out["qty_loss"]
    )

    for name in ("nll", "magnitude_loss", "qty_loss", "total_loss", "log_qty_hat"):
        assert torch.isfinite(out[name]).all(), name
    assert torch.allclose(loss, expected)
    assert not hasattr(model, "value_head")


def test_magnitude_loss_routes_only_to_direct_head_and_encoder() -> None:
    torch.manual_seed(13)
    model = make_model()
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)

    model.zero_grad(set_to_none=True)
    out["magnitude_loss"].backward()

    assert module_grad_norm(model.magnitude_head) > 0.0
    assert module_grad_norm(model.encoder) > 0.0
    assert module_grad_norm(model.magnitude_input_proj) > 0.0
    assert module_grad_norm(model.mark_head) == 0.0
    assert module_grad_norm(model.v_t) == 0.0


def test_quantity_prediction_does_not_depend_on_marker_head() -> None:
    torch.manual_seed(17)
    model = make_model().eval()
    marks, dts, values, mask = example_batch()
    with torch.no_grad():
        hidden = model.forward(marks, dts, values=values, mask=mask)[:, -2]
        before = model.predict_direct_magnitude(
            hidden,
            marks=marks,
            values=values,
            mask=mask,
        )["qty"]
        model.mark_head.weight.fill_(1000.0)
        model.mark_head.bias.fill_(-1000.0)
        after = model.predict_direct_magnitude(
            hidden,
            marks=marks,
            values=values,
            mask=mask,
        )["qty"]

    assert torch.equal(before, after)


def test_train_global_stats_ignore_validation_and_test() -> None:
    frame = pl.DataFrame({
        "chronological_split": ["train", "train", "validation", "test"],
        "mark": [0, 1, 3, 3],
        "scale_residual": [0.25, 0.5, 0.9, 0.8],
    })
    cfg = ExperimentConfig(base_dir="/tmp/m0-test", qty_decoder_mode="direct_log_qty")
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

    assert meta["magnitude_stats_source_split"] == "train"
    assert meta["magnitude_train_event_count"] == 2
    assert meta["magnitude_global_mean"] == pytest.approx(0.875)
    assert meta["magnitude_global_var"] == pytest.approx(0.390625)
    assert meta["magnitude_global_std"] == pytest.approx(0.625)


def test_m0_run_path_is_distinct_from_legacy(tmp_path) -> None:
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
    legacy = build_run_paths(ExperimentConfig(base_dir=str(tmp_path)), run_cfg).run_dir
    direct = build_run_paths(
        ExperimentConfig(
            base_dir=str(tmp_path),
            qty_decoder_mode="direct_log_qty",
            loss_mode="hybrid",
            train_loss_scope="target_only",
        ),
        run_cfg,
    ).run_dir

    assert legacy != direct
    assert "qtydecoder_direct_log_qty" in str(direct)
    assert "magnorm_global" in str(direct)
    assert "qtydecoder_" not in str(legacy)


def test_long_epoch_cli_propagates_m0_contract() -> None:
    args = build_parser().parse_args([
        "long-epoch",
        "--datasets",
        "intermittent",
        "--models",
        "titantpp",
        "--qty-decoder-mode",
        "direct_log_qty",
        "--split-mode",
        "fixed",
        "--train-loss-scope",
        "target_only",
        "--loss-mode",
        "hybrid",
    ])

    cfg = build_long_epoch_config(args)

    assert cfg.models == ("titantpp",)
    assert cfg.qty_decoder_mode == "direct_log_qty"
    assert cfg.magnitude_norm_mode == "global"
    assert cfg.lambda_magnitude == pytest.approx(1.0)


def test_long_epoch_cli_rejects_direct_decoder_for_mixed_models() -> None:
    args = build_parser().parse_args([
        "long-epoch",
        "--datasets",
        "intermittent",
        "--models",
        "rmtpp,titantpp",
        "--qty-decoder-mode",
        "direct_log_qty",
        "--split-mode",
        "fixed",
        "--train-loss-scope",
        "target_only",
        "--loss-mode",
        "hybrid",
    ])

    with pytest.raises(ValueError, match="TitanTPP-only"):
        build_long_epoch_config(args)


def test_artifact_aggregation_keeps_direct_and_legacy_separate() -> None:
    rows = []
    for decoder in ("mark_residual", "direct_log_qty"):
        rows.append({
            "dataset_name": "intermittent",
            "model_name": "titantpp",
            "candidate_name": "small_lmm",
            "selection": "best_val_nll",
            "qty_decoder_mode": decoder,
            "magnitude_norm_mode": "global",
            "lambda_magnitude": 1.0,
            "score": 0.1,
            "val_nll": 5.0,
            "val_nll_marker": 1.0,
            "val_nll_time": 4.0,
            "qty_mae": 3.0,
            "log_qty_mae": 0.5,
            "log_qty_rmse": 0.6,
            "val_magnitude_loss": 0.2,
            "dt_mae": 2.0,
            "mark_acc": 0.5,
            "value_mae": 0.2,
            "_total": 100,
            "_nll_steps": 100.0,
        })

    summary = aggregate_test_metrics(pl.DataFrame(rows))

    assert summary.height == 2
    assert set(summary["qty_decoder_mode"].to_list()) == {
        "mark_residual",
        "direct_log_qty",
    }


def test_weekly_evaluator_exports_direct_log_metrics() -> None:
    torch.manual_seed(19)
    model = make_model().eval()
    marks, dts, values, mask = example_batch()
    loader = [(marks, dts, mask, torch.arange(marks.size(0)), values)]

    metrics = eval_next_event_week_lookback(
        model,
        loader,
        "cpu",
        target_only_nll=True,
    )

    assert metrics["_total"] == 2
    assert metrics["_nll_steps"] == 2
    assert metrics["qty_mae"] >= 0.0
    assert metrics["log_qty_mae"] >= 0.0
    assert metrics["log_qty_rmse"] >= 0.0
    assert metrics["val_magnitude_loss"] >= 0.0
    assert metrics["context_1_count"] == 0
    assert metrics["context_2_4_count"] == 2
    assert torch.isnan(torch.tensor(metrics["value_mae"]))
    assert torch.isnan(torch.tensor(metrics["val_value_loss"]))


@pytest.mark.parametrize(
    "override,match",
    [
        ({"scale_base": 10.0}, "scale_base=2.0"),
        ({"loss_mode": "residual_only"}, "loss_mode='hybrid'"),
        ({"train_loss_scope": "all"}, "train_loss_scope='target_only'"),
        ({"value_input_mode": "residual"}, "value_input_mode='none'"),
    ],
)
def test_invalid_m0_contract_is_rejected(override, match) -> None:
    with pytest.raises(ValueError, match=match):
        make_model(**override)
