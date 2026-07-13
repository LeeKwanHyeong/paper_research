from types import SimpleNamespace

import matplotlib
import polars as pl
import pytest
import torch
import torch.nn.functional as F

matplotlib.use("Agg", force=True)

from models.RMTPPs.TitanTPP import TitanTPP
from models.RMTPPs.config import RMTPPConfig
from models.RMTPPs.marker_losses import (
    masked_normalized_ranked_probability_score,
    normalized_ranked_probability_score,
)
from models.RMTPPs.value_conditioning import apply_transition_loss_scope
from models.Titan import TitanConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.runner import (
    aggregate_test_metrics,
    build_run_paths,
    compute_training_loss,
    save_learning_curve_plots,
    summarize_mark_confusion,
)
from simple_lab_test.search.tpp_experiment import build_long_epoch_config, build_parser
from utils.training import TrainingConfig


def make_model(
    *,
    marker_loss_mode: str = "ce",
    lambda_ordinal: float = 0.0,
    loss_mode: str = "residual_only",
) -> TitanTPP:
    rmtpp_cfg = RMTPPConfig(
        num_marks=5,
        mark_emb_dim=8,
        scale_base=2.0,
        value_head_activation="identity",
        marker_loss_mode=marker_loss_mode,
        lambda_ordinal=lambda_ordinal,
        loss_mode=loss_mode,
        train_loss_scope="all",
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


def test_deterministic_rps_is_normalized_ordinal_distance() -> None:
    class_count = 11
    predictions = torch.tensor([1, 0, 3, 10])
    targets = torch.tensor([1, 1, 1, 1])
    logits = torch.full((4, class_count + 1), -100.0)
    logits[torch.arange(4), predictions] = 100.0
    logits[:, -1] = 50.0

    scores = normalized_ranked_probability_score(
        logits,
        targets,
        num_real_marks=class_count,
    )

    assert torch.allclose(scores, torch.tensor([0.0, 0.1, 0.2, 0.9]))
    assert torch.all((scores >= 0.0) & (scores <= 1.0))


def test_rps_excludes_pad_logit_while_categorical_ce_keeps_it() -> None:
    target = torch.tensor([1])
    low_pad_logits = torch.tensor([[0.2, 1.2, -0.4, 0.5, -20.0]])
    high_pad_logits = low_pad_logits.clone()
    high_pad_logits[:, -1] = 20.0

    low_rps = normalized_ranked_probability_score(
        low_pad_logits,
        target,
        num_real_marks=4,
    )
    high_rps = normalized_ranked_probability_score(
        high_pad_logits,
        target,
        num_real_marks=4,
    )

    assert torch.equal(low_rps, high_rps)
    assert not torch.allclose(
        F.cross_entropy(low_pad_logits, target),
        F.cross_entropy(high_pad_logits, target),
    )


def test_masked_rps_uses_only_selected_targets_and_handles_one_class() -> None:
    logits = torch.tensor(
        [[[4.0, 0.0, -1.0], [0.0, 4.0, -1.0], [0.0, 0.0, -1.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 0, 1]])
    mask = torch.tensor([[True, False, True]])
    per_target = normalized_ranked_probability_score(
        logits,
        targets,
        num_real_marks=2,
    )

    masked = masked_normalized_ranked_probability_score(
        logits,
        targets,
        mask,
        num_real_marks=2,
    )
    one_class = masked_normalized_ranked_probability_score(
        logits[..., :1],
        torch.zeros_like(targets),
        mask,
        num_real_marks=1,
    )

    assert torch.allclose(masked, per_target[mask].mean())
    assert one_class.item() == 0.0
    one_class.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("loss_scope", ["all", "target_only"])
def test_titan_ce_and_rps_share_the_transition_mask(loss_scope: str) -> None:
    torch.manual_seed(7)
    model = make_model(marker_loss_mode="ce_rps", lambda_ordinal=0.1).eval()
    marks, dts, values, mask = example_batch()

    out = model.nll(
        marks,
        dts,
        values=values,
        mask=mask,
        loss_scope=loss_scope,
    )
    with torch.no_grad():
        hidden = model.forward(marks, dts, values=values, mask=mask)
        logits = model.mark_head(hidden[:, :-1])
        targets = marks[:, 1:]
        step_mask = apply_transition_loss_scope(
            mask[:, 1:] & mask[:, :-1],
            loss_scope,
        )
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        expected_ce = (ce * step_mask).sum() / step_mask.sum()
        expected_rps = masked_normalized_ranked_probability_score(
            logits,
            targets,
            step_mask,
            num_real_marks=4,
        )

    assert torch.allclose(out["nll_marker"], expected_ce)
    assert torch.allclose(out["ordinal_marker_loss"], expected_rps)
    assert int(out["steps"].item()) == int(step_mask.sum().item())


def test_v2_and_v5a_have_identical_parameters_and_predictions_before_training() -> None:
    torch.manual_seed(11)
    v2 = make_model()
    torch.manual_seed(11)
    v5a = make_model(marker_loss_mode="ce_rps", lambda_ordinal=0.1)
    marks, dts, values, mask = example_batch()

    assert v2.state_dict().keys() == v5a.state_dict().keys()
    for name, value in v2.state_dict().items():
        assert torch.equal(value, v5a.state_dict()[name]), name

    assert torch.equal(
        v2.forward(marks, dts, values=values, mask=mask),
        v5a.forward(marks, dts, values=values, mask=mask),
    )
    v2_out = v2.nll(marks, dts, values=values, mask=mask)
    v5a_out = v5a.nll(marks, dts, values=values, mask=mask)
    for key in (
        "nll",
        "nll_marker",
        "nll_time",
        "ordinal_marker_loss",
        "value_loss",
        "qty_loss",
        "value_hat",
        "value_by_mark",
    ):
        assert torch.equal(v2_out[key], v5a_out[key]), key

    weighted_rps = 0.1 * v5a_out["ordinal_marker_loss"]
    assert torch.allclose(v2_out["marker_train_loss"], v2_out["nll_marker"])
    assert torch.allclose(
        v5a_out["marker_train_loss"],
        v5a_out["nll_marker"] + weighted_rps,
    )
    assert torch.allclose(v5a_out["total_loss"] - v2_out["total_loss"], weighted_rps)

    torch.manual_seed(11)
    legacy_formula_model = make_model()
    legacy_out = legacy_formula_model.nll(marks, dts, values=values, mask=mask)
    legacy_loss = (
        legacy_out["nll_marker"]
        + legacy_out["nll_time"]
        + legacy_out["value_loss"]
    )
    v2.zero_grad(set_to_none=True)
    legacy_formula_model.zero_grad(set_to_none=True)
    v2_out["total_loss"].backward()
    legacy_loss.backward()
    for (name, parameter), (legacy_name, legacy_parameter) in zip(
        v2.named_parameters(),
        legacy_formula_model.named_parameters(),
    ):
        assert name == legacy_name
        if parameter.grad is None or legacy_parameter.grad is None:
            assert parameter.grad is None and legacy_parameter.grad is None, name
            continue
        assert torch.allclose(parameter.grad, legacy_parameter.grad), name


def test_isolated_rps_updates_only_marker_path_and_shared_encoder() -> None:
    torch.manual_seed(13)
    model = make_model(marker_loss_mode="ce_rps", lambda_ordinal=0.1)
    marks, dts, values, mask = example_batch()
    out = model.nll(marks, dts, values=values, mask=mask)

    model.zero_grad(set_to_none=True)
    out["ordinal_marker_loss"].backward()

    assert module_grad_norm(model.mark_head) > 0.0
    assert module_grad_norm(model.encoder) > 0.0
    assert module_grad_norm(model.v_t) == 0.0
    assert module_grad_norm(model.value_head) == 0.0
    pad_grad = model.mark_head.weight.grad[-1]
    assert torch.count_nonzero(pad_grad) == 0


@pytest.mark.parametrize("loss_mode", ["residual_only", "hybrid", "qty_only"])
def test_shared_loss_composer_adds_weighted_rps_exactly_once(loss_mode: str) -> None:
    marker_ce = torch.tensor(2.0, dtype=torch.float64)
    ordinal = torch.tensor(0.5, dtype=torch.float64)
    marker_train = marker_ce + 0.1 * ordinal
    out = {
        "nll_marker": marker_ce,
        "ordinal_marker_loss": ordinal,
        "marker_train_loss": marker_train,
        "nll_time": torch.tensor(3.0, dtype=torch.float64),
        "value_loss": torch.tensor(4.0, dtype=torch.float64),
        "qty_loss": torch.tensor(5.0, dtype=torch.float64),
    }
    model = SimpleNamespace(cfg=SimpleNamespace(loss_mode=loss_mode, lambda_qty=0.25))
    training_cfg = TrainingConfig(lambda_value=2.0, lambda_dt=3.0)

    loss = compute_training_loss(model=model, out=out, training_cfg=training_cfg)
    expected = marker_train + 3.0 * out["nll_time"]
    if loss_mode in {"residual_only", "hybrid"}:
        expected = expected + 2.0 * out["value_loss"]
    if loss_mode in {"hybrid", "qty_only"}:
        expected = expected + 0.25 * out["qty_loss"]

    expected_without_ordinal = expected - 0.1 * ordinal
    assert torch.equal(loss, expected)
    assert torch.allclose(loss - expected_without_ordinal, 0.1 * ordinal)


@pytest.mark.parametrize(
    ("marker_loss_mode", "lambda_ordinal", "message"),
    [
        ("ce", 0.1, "requires lambda_ordinal=0"),
        ("ce_rps", 0.0, "requires lambda_ordinal>0"),
        ("ce_rps", float("nan"), "finite and non-negative"),
        ("unknown", 0.0, "Unsupported TitanTPP marker_loss_mode"),
    ],
)
def test_invalid_marker_loss_configs_fail_fast(
    marker_loss_mode: str,
    lambda_ordinal: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_model(
            marker_loss_mode=marker_loss_mode,
            lambda_ordinal=lambda_ordinal,
        )


def test_v5a_uses_distinct_run_identity_without_changing_v2_path(tmp_path) -> None:
    run_cfg = RunConfig(
        dataset_name="intermittent",
        dataset_kind="marked_target",
        model_name="titantpp",
        candidate_name="small_lmm",
        candidate=SimpleNamespace(name="small_lmm"),
        seed=42,
        epochs=50,
        scale_base=2.0,
        titan_profile="dataset_best",
    )
    v2_path = build_run_paths(ExperimentConfig(base_dir=str(tmp_path)), run_cfg).run_dir
    v5a_path = build_run_paths(
        ExperimentConfig(
            base_dir=str(tmp_path),
            marker_loss_mode="ce_rps",
            lambda_ordinal=0.1,
        ),
        run_cfg,
    ).run_dir

    assert "markloss_" not in str(v2_path)
    assert "lambdaord_" not in str(v2_path)
    assert "markloss_ce_rps/lambdaord_0p1" in str(v5a_path)
    assert v2_path != v5a_path


def test_long_epoch_cli_propagates_v5a_contract() -> None:
    args = build_parser().parse_args(
        [
            "long-epoch",
            "--models",
            "titantpp",
            "--marker-loss-mode",
            "ce_rps",
            "--lambda-ordinal",
            "0.10",
        ]
    )

    cfg = build_long_epoch_config(args)

    assert cfg.models == ("titantpp",)
    assert cfg.marker_loss_mode == "ce_rps"
    assert cfg.lambda_ordinal == pytest.approx(0.1)


def test_long_epoch_cli_rejects_ce_rps_for_mixed_models() -> None:
    args = build_parser().parse_args(
        [
            "long-epoch",
            "--models",
            "rmtpp,titantpp",
            "--marker-loss-mode",
            "ce_rps",
            "--lambda-ordinal",
            "0.10",
        ]
    )

    with pytest.raises(ValueError, match="TitanTPP-only"):
        build_long_epoch_config(args)


def test_per_class_metrics_include_zero_support_classes() -> None:
    confusion = pl.DataFrame(
        {
            "true_mark": [0, 0, 1, 1],
            "pred_mark": [0, 1, 0, 1],
            "count": [8, 2, 1, 9],
            "share_within_true": [0.8, 0.2, 0.1, 0.9],
        }
    )

    metrics = summarize_mark_confusion(confusion, num_real_marks=3)
    mark_0 = metrics.filter(pl.col("mark") == 0).row(0, named=True)
    mark_2 = metrics.filter(pl.col("mark") == 2).row(0, named=True)

    assert metrics.height == 3
    assert mark_0["true_count"] == 10
    assert mark_0["recall"] == pytest.approx(0.8)
    assert mark_0["precision"] == pytest.approx(8 / 9)
    assert mark_2["true_count"] == 0
    assert mark_2["precision"] == 0.0
    assert mark_2["recall"] == 0.0


def test_legacy_test_metrics_receive_v2_marker_defaults() -> None:
    legacy_metrics = pl.DataFrame(
        {
            "dataset_name": ["intermittent"],
            "model_name": ["titantpp"],
            "candidate_name": ["small_lmm"],
            "selection": ["best_val_nll"],
            "score": [0.1],
            "val_nll": [5.0],
            "val_nll_marker": [1.0],
            "val_nll_time": [4.0],
            "qty_mae": [3.0],
            "dt_mae": [2.0],
            "mark_acc": [0.5],
            "value_mae": [0.2],
            "_total": [100],
            "_nll_steps": [100.0],
        }
    )

    summary = aggregate_test_metrics(legacy_metrics).row(0, named=True)

    assert summary["marker_loss_mode"] == "ce"
    assert summary["lambda_ordinal"] == 0.0
    assert summary["value_head_mode"] == "shared"
    assert summary["qty_mark_gradient_mode"] == "coupled"


def test_learning_curve_plot_keeps_all_v5_metrics(tmp_path) -> None:
    history = pl.DataFrame(
        {
            "dataset_name": ["intermittent", "intermittent"],
            "model_name": ["titantpp", "titantpp"],
            "candidate_name": ["small_lmm", "small_lmm"],
            "epoch": [1, 2],
            "score": [0.1, 0.2],
            "val_nll": [5.0, 4.9],
            "val_nll_marker": [1.0, 0.9],
            "val_nll_time": [4.0, 4.0],
            "val_ordinal_marker_loss": [0.2, 0.19],
            "mark_mae": [0.5, 0.48],
            "qty_mae": [3.0, 2.9],
        }
    )

    save_learning_curve_plots(history, tmp_path)

    plot_path = tmp_path / "intermittent_learning_curves.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
