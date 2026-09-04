import copy
import json

import polars as pl
import pytest
import torch

from paper.scripts.analyze_hard_lmm_query_diagnostic import decide, independent_mse_check
from paper.scripts.run_hard_lmm_query_diagnostic import CONTRACT, fold_for_series, load_train_frame, paired_summary


def _analyses():
    model = {"pooled": {"two_stat_relative_improvement_vs_h_only": .02, "mse": {"two_stat": .98, "constant": 1.1}},
             "folds": {str(i): {"two_stat_relative_improvement_vs_h_only": .02} for i in (0, 1)},
             "cluster_bootstrap": {"paired_delta_lower_p05": .001}}
    return {name: {"separate_key": copy.deepcopy(model), "original": copy.deepcopy(model)}
            for name in ("yellow_trip_hourly", "insta_market_basket")}


def test_gate_requires_instacart_and_taxi_with_no_original_fallback():
    policy = json.loads(CONTRACT.read_text())["evidence_gate"]
    data = _analyses()
    assert decide(data, policy)["evidence_passed"]
    data["insta_market_basket"]["separate_key"]["folds"]["1"]["two_stat_relative_improvement_vs_h_only"] = 0
    assert not decide(data, policy)["evidence_passed"]
    data = _analyses()
    data["yellow_trip_hourly"]["separate_key"]["pooled"]["two_stat_relative_improvement_vs_h_only"] = -.01001
    assert not decide(data, policy)["evidence_passed"]
    data = _analyses()
    data["insta_market_basket"]["separate_key"]["cluster_bootstrap"]["paired_delta_lower_p05"] = 0
    assert not decide(data, policy)["evidence_passed"]


def test_lazy_train_filter_excludes_validation_and_test(tmp_path):
    path = tmp_path / "splits.parquet"
    pl.DataFrame({"chronological_split": ["train", "validation", "test"],
                  "demand_qty": [3., float("nan"), float("inf")]}).write_parquet(path)
    result = load_train_frame(path)
    assert result.height == 1 and result["demand_qty"].item() == 3
    assert {fold_for_series(i) for i in range(100)} == {0, 1}
    assert fold_for_series(5) == fold_for_series(torch.tensor(5))


def test_pairing_rejects_same_values_in_wrong_target_order():
    cache = {key: torch.tensor([0, 1]) for key in
             ("target_index", "series_index", "fold", "context_end", "quantity", "history_length")}
    cache["stats"] = torch.zeros(2, 2)
    other = copy.deepcopy(cache)
    other["target_index"] = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="target_index"):
        paired_summary(cache, other)
