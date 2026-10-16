import math

import polars as pl
import pytest
from paper.scripts import analyze_titantpp_cnn_matched_history as m


def histories(split="train"):
    return pl.DataFrame(dict(entity_id=["a","a","a","b","b","c","a","a"],seq=list(range(8)),
        split=[split]*8,last_quantity=[100.,100.,100.,100.,100.,100.,100.,100.],
        last_observed_log_change=[-.3,.3,.31,-.3,.3,.3,0.,-.3],
        previous_observation_available=[True]*7+[False]))


def test_matching_is_outcome_blind_and_balances_cells():
    a=m.match_weights(histories(),[2.,7.,65.])
    b=m.match_weights(histories().with_columns(pl.lit(-9999).alias("prediction_1"),pl.lit(123).alias("raw_quantity")),[2.,7.,65.])
    assert a.equals(b)
    assert set(a["entity_id"])=={"a","b"}
    sums=a.group_by(m.KEYS+["direction"]).agg(pl.col("weight").sum()).sort(m.KEYS+["direction"])
    for _,g in sums.group_by(m.KEYS):
        assert g.height==2
        assert g["weight"][0]==pytest.approx(g["weight"][1])
    assert not {5,6,7}&set(a["seq"])


def test_entity_and_magnitude_cannot_cross_cells():
    f=histories().head(2).with_columns(pl.Series("entity_id",["a","b"]))
    assert m.match_weights(f,[2.,7.,65.]).height==0
    f=histories().head(2).with_columns(pl.Series("last_observed_log_change",[-.3,.9]))
    assert m.match_weights(f,[2.,7.,65.]).height==0
    f=histories().head(2).with_columns(pl.Series("last_quantity",[100.,200.]))
    assert m.match_weights(f,[2.,7.,65.]).height==0


def test_test_split_rejected():
    with pytest.raises(Exception,match="train/validation"):
        m.match_weights(histories("test"),[2.,7.,65.])


def test_common_entity_mass_equal_across_splits():
    t=m.match_weights(histories(),[2.,7.,65.])
    v=m.match_weights(histories("validation").filter(pl.col("entity_id")!="b"),[2.,7.,65.])
    a,b=m.common_entity_weights(t,v)
    for f in (a,b):
        assert set(f["entity_id"])=={"a"}
        assert f.group_by("direction").agg(pl.col("weight").sum())["weight"].to_list()==pytest.approx([1.,1.])


def test_metrics_hand_calculation():
    f=pl.DataFrame(dict(entity_id=["a","b"],weight=[1.,3.],error=[2.,-4.],time_nll_1=[1.,5.]))
    r=m.describe(f)
    assert r["bias"]==pytest.approx(-2.5)
    assert r["mae"]==pytest.approx(3.5)
    assert r["rmse"]==pytest.approx(math.sqrt(13))
    assert r["ess"]==pytest.approx(1.6)
    assert r["time_nll"]==pytest.approx(4.)


def test_empty_cells_are_not_zero_error():
    f=pl.DataFrame(schema={"entity_id":pl.String,"weight":pl.Float64,"error":pl.Float64,"time_nll_1":pl.Float64})
    assert m.describe(f)["rmse"] is None
