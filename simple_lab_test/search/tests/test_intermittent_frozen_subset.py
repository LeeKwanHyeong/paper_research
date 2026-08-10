from __future__ import annotations

import polars as pl

from paper.scripts.build_intermittent_frozen_subset import (
    proportional_quotas,
    select_series,
)
from simple_lab_test.search.common.experiment_utils import SearchConfig, search_config_for_dataset
from simple_lab_test.search.common.runner import validation_nll_early_stop


def test_proportional_quotas_preserves_total_and_minimum() -> None:
    quotas = proportional_quotas({"A": 100, "B": 200, "C": 300}, 120, minimum=10)
    assert sum(quotas.values()) == 120
    assert all(value >= 10 for value in quotas.values())
    assert quotas["C"] >= quotas["B"] >= quotas["A"]


def test_select_series_is_deterministic_and_site_balanced() -> None:
    rows = []
    for site in ("V100", "V101"):
        for idx in range(100):
            rows.append({
                "site_cd": site,
                "oper_part_no": f"{site}::P{idx:03d}",
                "event_count": 20 + idx,
                "train_qty_p95": float(1 + idx % 17),
            })
    stats = pl.DataFrame(rows)
    first = select_series(stats, sample_size=40, seed=7, min_per_site=10, bins=5)
    second = select_series(stats, sample_size=40, seed=7, min_per_site=10, bins=5)
    assert first.equals(second)
    assert first.height == 40
    assert first.group_by("site_cd").len().sort("site_cd")["len"].to_list() == [20, 20]


def test_long_runtime_profile_preserves_explicit_sequence_limits() -> None:
    long_cfg = SearchConfig(
        base_dir="/tmp/test",
        lookback_weeks=520,
        max_seq_len=96,
        intermittent_runtime_profile="long",
    )
    effective = search_config_for_dataset(long_cfg, "marked_target")
    assert effective.lookback_weeks == 520
    assert effective.max_seq_len == 96

    legacy = search_config_for_dataset(SearchConfig(base_dir="/tmp/test"), "marked_target")
    assert legacy.lookback_weeks == 52
    assert legacy.max_seq_len == 16


def test_validation_nll_early_stop_respects_minimum_and_patience() -> None:
    history = [
        {"epoch": epoch, "val_nll": value}
        for epoch, value in enumerate([5.0, 4.0, 3.0, 3.1, 3.2], start=1)
    ]
    assert not validation_nll_early_stop(history, min_epochs=6, patience=2)
    assert validation_nll_early_stop(history, min_epochs=5, patience=2)
    assert not validation_nll_early_stop(history, min_epochs=1, patience=0)
