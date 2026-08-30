from __future__ import annotations

import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from paper.scripts.analyze_count_aware_titantpp_mac import (
    add_surprise_strata,
    b1_counterfactual_outputs,
    causal_surprise_features,
    historical_cost_row,
    verify_full_equivalence,
)
import polars as pl


def test_causal_surprise_excludes_current_prediction_segment_writes() -> None:
    diagnostics = {
        "associative_loss": torch.tensor(
            [[1.0, 2.0, 100.0, 200.0], [3.0, 4.0, 5.0, 6.0]]
        ),
        "write_applied": torch.ones(2, 4),
    }
    result = causal_surprise_features(
        diagnostics,
        torch.tensor([2, 1]),
        segment_size=2,
    )

    assert result["causal_surprise_count"].tolist() == [2, 0]
    assert result["causal_surprise_mean"].tolist() == [1.5, 0.0]
    assert result["causal_surprise_latest"].tolist() == [2.0, 0.0]
    assert result["causal_surprise_max"].tolist() == [2.0, 0.0]


def test_same_checkpoint_counterfactual_preserves_official_full_path() -> None:
    torch.manual_seed(123)
    model, _ = build_count_aware_model(
        "titantpp_titans_mac",
        hidden_dim=8,
        train_log_mean=1.0,
        max_seq_len=8,
    )
    assert isinstance(model, CountAwareTitanTPP)
    model.eval()
    dts = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 2.0]])
    quantities = torch.tensor([[1.0, 2.0, 4.0, 8.0], [0.0, 3.0, 6.0, 12.0]])
    mask = torch.tensor([[True, True, True, True], [False, True, True, True]])

    diagnostic = b1_counterfactual_outputs(model, dts, mask, quantities)

    assert verify_full_equivalence(model, dts, mask, quantities, diagnostic) <= 1e-6
    assert diagnostic["pred_full"].shape == (2,)
    assert torch.isfinite(diagnostic["long_term_residual_norm"]).all()
    assert torch.isfinite(diagnostic["online_update_residual_norm"]).all()


def test_surprise_strata_keep_no_prior_write_separate() -> None:
    frame = pl.DataFrame(
        {"causal_surprise_mean": [0.0, 1.0, 2.0, 3.0, 100.0]}
    )

    stratified, boundaries = add_surprise_strata(frame)

    assert stratified["causal_surprise_stratum"][0] == "no_prior_visible_write"
    assert boundaries["p50"] > 0.0
    assert stratified["causal_surprise_stratum"][-1] == "surprise_ge_p99"


def test_historical_epoch_cost_uses_completed_epochs() -> None:
    row = historical_cost_row(
        "dataset",
        {"elapsed_seconds": 100.0, "completed_epochs": 20},
        {"elapsed_seconds": 300.0, "completed_epochs": 30},
    )

    assert row["b0_seconds_per_completed_epoch"] == 5.0
    assert row["b1_seconds_per_completed_epoch"] == 10.0
    assert row["b1_b0_epoch_cost_ratio"] == 2.0
