from copy import deepcopy
from pathlib import Path
import json
import pytest
from paper.scripts import advance_titantpp_history_taxi6 as m

def test_preserves_and_reads_known_supervisor_suffix():
    assert m.parse_control(b'{"returncode":0}\n')==m.parse_control(b'{"returncode":0}\\n')
    with pytest.raises(ValueError):m.parse_control(b'{}evil')

def test_wire_bytes_and_process_delimiter():
    import ast
    constants=[n.value for n in ast.walk(ast.parse(m.REMOTE_CODE)) if isinstance(n,ast.Constant)]
    assert b'\n' in constants and '\x00' in constants
    assert b'\\n' not in constants and '\\x00' not in constants

def test_eta_never_borrows_other_fit_or_promotes_unconfirmed():
    j={'job':{'arm':'a','seed':42},'state':'unconfirmed','history':{'history':[{'epoch':5,'val_qty_rmse':10.}]},'timing':{'epochs':[{'elapsed_seconds':10.}]}}
    q={'jobs':[j],'observed_unix':100,'all_training_terminal_verified':False,'failure':False}
    assert 'conditional_stop_seconds' not in m.brief_round(q)['rows'][0]
    j['state']='running';row=m.brief_round(q)['rows'][0]
    assert row['conditional_stop_epoch']==45 and row['conditional_stop_seconds']==400
    j['timing']={};assert 'conditional_stop_seconds' not in m.brief_round(q)['rows'][0]

def endpoint(value):
    return {'qty_rmse':value,'qty_mae':value/2,'time_nll':1.,'count':8268,'tail':{'qty_rmse':2*value,'qty_mae':value,'time_nll':2.}}

def test_collection_aggregation_three_seeds_and_sample_sd(tmp_path,monkeypatch):
    report=tmp_path/'report';report.mkdir();monkeypatch.setattr(m,'REPORT',report)
    c={'jobs':[],'baseline_jobs':[]};local=tmp_path/'originals'
    for s,v in [(42,10.),(52,20.),(62,30.)]:
        c['baseline_jobs'].append({'seed':s,'original':{'selected_epoch':1,'validation_reference':endpoint(v)}})
        for arm,delta in [('titantpp_cnn_history_adaptive_v1',-2),('titantpp_cnn_history_capacity_v1',1)]:
            j={'id':str(s)+arm,'arm':arm,'seed':s};c['jobs'].append(j)
            f=local/'run'/j['id'];p=f/'runs'/arm/'count_only_log_regression'/f'seed_{s}';p.mkdir(parents=True)
            m.write(p/'history.json',{'history':[{'epoch':1,'val_qty_rmse':v+delta}]})
            m.write(p/'endpoint_replays.json',{'best_epoch':1,'selected':endpoint(v+delta)})
            files={str(x.relative_to(f)):m.sha(x) for x in p.iterdir()}
            m.write(f/'terminal_manifest.json',{'job':j,'held_out_test_evaluated':False,'files':files})
    r=m.analyze(c,local)
    assert r['aggregate']['CNN']['RMSE']=={'mean':20.,'sample_sd':10.}
    assert r['aggregate']['Adaptive']['RMSE']['mean']==18.
    assert r['aggregate']['Capacity']['RMSE']['mean']==21.
    assert len(r['seed_differences'])==9
    assert r['seed_differences'][0]['RMSE']['absolute']==-2.
    assert '\n|' in (report/'THREE_SEED_VALIDATION.md').read_text()
