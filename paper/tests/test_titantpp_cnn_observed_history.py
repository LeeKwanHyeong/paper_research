"""Synthetic-only tests of pre-target grouping and evidence progression rules."""
import math
import polars as pl
import pytest
from paper.scripts import analyze_titantpp_cnn_observed_history as a


def fixture(split="train"):
    return pl.DataFrame(dict(entity_id=["a","b","c","d"],seq=[2,2,2,1],dataset=["synthetic"]*4,
        seed=[42]*4,split=[split]*4,history_length=[2,2,2,1],last_quantity=[1.,2.,3.,4.],
        mean_quantity=[2.,2.,2.,4.],last_log_deviation=[-1.,0.,1.,0.],history_log_std=[.5,0.,.5,0.],
        last_observed_log_change=[-1.,0.,1.,0.],previous_observation_available=[True,True,True,False],
        raw_quantity=[10.,10.,10.,10.],prediction_1=[12.,9.,14.,10.],time_nll_1=[1.,2.,3.,4.]))


def test_target_perturbation_never_changes_direction_or_train_cuts():
    f=fixture(); altered=f.with_columns(pl.lit(99999.).alias("raw_quantity"),pl.lit(0.).alias("prediction_1"))
    x,y=a.annotate(f),a.annotate(altered)
    assert x["observed_direction"].to_list()==y["observed_direction"].to_list()==["falling","flat","rising","unavailable"]
    assert a.cuts_from_train(x)==a.cuts_from_train(y)


def test_validation_does_not_refit_cuts():
    train=a.annotate(fixture()); cuts=a.cuts_from_train(train)
    val=a.annotate(fixture("validation").with_columns(pl.lit(100.).alias("last_quantity")))
    result=a.cells(val,cuts,200.)
    levels=[r for r in result if r["kind"]=="level_direction"]
    assert sum(r["count"] for r in levels if r["level_quartile"]==3)==4
    assert sum(r["count"] for r in levels if r["level_quartile"]!=3)==0
    with pytest.raises(ValueError,match="Train"):
        a.cuts_from_train(val)


def test_metric_denominators_and_empty_cells():
    f=a.annotate(fixture()); rows=a.cells(f,a.cuts_from_train(f),200.)
    overall=rows[0]
    assert math.isclose(overall["rmse"],math.sqrt(21/4))
    assert overall["mae"]==7/4 and overall["bias"]==5/4
    assert overall["over_rate"]==.5 and overall["under_rate"]==.25
    assert rows[1]["count"]==0 and rows[1]["rmse"] is None
    direction=[r for r in rows if r["kind"]=="direction"]
    assert sum(r["count"] for r in direction)==4
    assert sum(r["mse_contribution"] for r in direction)==21/4
    assert sum(r["sse_share"] for r in direction)==1


def test_missing_predecessor_is_not_flat_and_tied_cuts_preserve_empty_bins():
    f=a.annotate(fixture().with_columns(pl.lit(0.).alias("last_quantity")))
    cuts=a.cuts_from_train(f)
    assert cuts["last_quantity"]==[0.,0.,0.]
    assert cuts["absolute_observed_log_change"]==[.5,1.,1.]
    rows=a.cells(f,cuts,200.)
    assert next(r for r in rows if r["kind"]=="direction" and r["group"]=="unavailable")["count"]==1
    assert next(r for r in rows if r["kind"]=="feature_quartile" and r["feature"]=="last_quantity" and r["group"]=="1")["count"]==0


def support_fixture():
    return [dict(dataset="yellow_trip_hourly",seed=seed,split=split,kind="level_direction",
                 level_quartile=q,group=direction,count=100,entity_count=10,mae=10.,
                 bias=2. if direction=="falling" else -2.)
            for q in range(4) for seed in (42,52,62) for split in a.SPLITS for direction in ("falling","rising")]


def test_support_requires_replication_in_both_splits_and_multiple_seeds():
    rows=support_fixture()
    assert a.implementation_support(rows)["implementation_supported"]
    for row in rows:
        if row["seed"] in (52,62) and row["split"]=="validation": row["bias"] *= -1
    assert not a.implementation_support(rows)["implementation_supported"]


def test_tiny_or_single_entity_groups_do_not_support_progression():
    for field,value in (("count",49),("entity_count",4),("bias",0.)):
        rows=support_fixture()
        for row in rows: row[field]=value
        assert not a.implementation_support(rows)["implementation_supported"]


def test_no_held_out_split_and_no_partial_seed_averages():
    with pytest.raises(ValueError,match="train/validation"): a.checked_split("test")
    with pytest.raises(ValueError,match="train/validation"):
        f=a.annotate(fixture("test")); a.cells(f,{k:[0.,1.,2.] for k in a.FEATURES},200.)
    rows=[]
    for seed in (42,52,62):
        f=a.annotate(fixture())
        rows += [dict(dataset="synthetic",seed=seed,split="train",**r) for r in a.cells(f,a.cuts_from_train(f),200.)]
    result=a.aggregate(rows)
    assert result[0]["metrics"]["rmse"]["sample_sd"]==0
    with pytest.raises(ValueError,match="Duplicate/missing"): a.aggregate(rows[:-1])
