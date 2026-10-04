"""Synthetic boundary and reconstruction checks without model or real data access."""
import importlib.util
import math
from pathlib import Path
import sys

import polars as pl

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('deep_strata_analyzer',HERE/'analyze_results.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def test_strict_tail_equality_lower_bin_and_empty_history_bin():
    f=pl.DataFrame({'raw_quantity':[2.,3.,7.,8.,9.], 'predicted_raw_quantity':[1.,5.,6.,8.,10.],
                    'time_nll':[.1,.2,.3,.4,.5],'history_length':[1,1,1,1,1]})
    rows={r['view']:r for r in a.stratify(f,[3,7],[1,3])}
    assert rows['tail']['count']==2 and math.isclose(rows['tail']['qty_rmse'],math.sqrt(.5))
    assert rows['quantity_bin_0']['count']==2 and rows['quantity_bin_1']['count']==1
    assert rows['history_bin_1']['count']==0 and rows['history_bin_1']['qty_rmse'] is None
    assert rows['history_bin_2']['count']==0 and rows['history_bin_2']['time_nll'] is None
    assert rows['overall']['qty_sse']==7
    assert rows['tail']['qty_sse']+rows['body']['qty_sse']==rows['overall']['qty_sse']

def test_empty_tail_is_null_and_does_not_change_full_population():
    f=pl.DataFrame({'raw_quantity':[1.,3.],'predicted_raw_quantity':[2.,3.],
                    'time_nll':[.25,.75],'history_length':[1,2]})
    rows={r['view']:r for r in a.stratify(f,[3],[1])}
    assert rows['tail']['count']==0 and rows['tail']['qty_rmse'] is None
    assert rows['tail']['time_nll'] is None
    assert rows['body']['count']==rows['overall']['count']==2
    assert rows['body']['qty_sse']==rows['overall']['qty_sse']==1

def test_raw_quantity_metrics_are_event_weighted_not_bin_averages():
    f=pl.DataFrame({'raw_quantity':[1.,2.,20.],'predicted_raw_quantity':[0.,0.,10.],
                    'time_nll':[1.,2.,3.],'history_length':[1,2,3]})
    rows={r['view']:r for r in a.stratify(f,[2],[1,2])}
    assert rows['overall']['qty_sse']==105
    assert math.isclose(rows['overall']['qty_rmse'],math.sqrt(105/3))
    assert math.isclose(rows['overall']['qty_mae'],13/3)
    assert not math.isclose(rows['overall']['qty_rmse'],
        (rows['quantity_bin_0']['qty_rmse']+rows['quantity_bin_1']['qty_rmse'])/2)
