from copy import deepcopy
import math
import pytest
from paper.scripts import compare_titantpp_time_zero_mlp9 as c

def fixture_records():
    rows=[]
    for d in c.DATASETS:
        for i,s in enumerate(c.SEEDS,1):
            for a in c.ALPHAS:
                metrics={k:{'count':9,'qty_rmse':i+(a/10),'qty_mae':i/2+(a/10),'time_nll':99. if a==0 else i} for k in ('overall','cap_hit')}
                metrics['tail']={'count':0,**{k:None for k in c.METRICS}}
                rows.append({'dataset':d,'seed':s,'time_alpha':a,'host':'5080' if d=='yellow_trip_hourly' or (d=='intermittent_frozen_5000' and s==42) else '5090',
                    'target_identity':{'count':9,'hash':d},'time_head_trained':a!=0,'metrics':metrics})
    return rows

def test_exact_matrix_means_sample_sd_and_diagnostic_time():
    r=c.aggregate(fixture_records())
    assert len(r['summary'])==108 and len(r['seed_differences'])==486 and len(r['decisions'])==9
    s=next(x for x in r['summary'] if x['dataset']==c.DATASETS[0] and x['time_alpha']==0 and x['cohort']=='overall' and x['metric']=='qty_rmse')
    assert s['mean']==2 and s['sample_sd_ddof1']==1
    a=next(x for x in r['decisions'] if x['dataset']==c.DATASETS[0] and x['time_alpha']==0)
    assert a['quantity_both_means_improved_vs1'] and a['three_metric_joint_model_improvement'] is None
    assert all(x['mean'] is None and x['sample_sd_ddof1'] is None for x in r['summary'] if x['cohort']=='tail')

@pytest.mark.parametrize('mutation',('missing','duplicate','target','host','head_flag','nonfinite','empty_metric','cohort_count'))
def test_invalid_scope_or_population_rejected(mutation):
    rows=fixture_records()
    if mutation=='missing':rows.pop()
    if mutation=='duplicate':rows[-1]=deepcopy(rows[0])
    if mutation=='target':rows[0]['target_identity']={}
    if mutation=='host':rows[0]['host']='other'
    if mutation=='head_flag':rows[0]['time_head_trained']=True
    if mutation=='nonfinite':rows[0]['metrics']['overall']['qty_rmse']=float('nan')
    if mutation=='empty_metric':rows[0]['metrics']['tail']['qty_mae']=0.
    if mutation=='cohort_count':rows[0]['metrics']['overall']['count']=8
    with pytest.raises(ValueError):c.aggregate(rows)

def test_zero_reference_relative_difference_is_na():
    rows=fixture_records()
    for r in rows:
        if r['time_alpha']==1:r['metrics']['overall']['time_nll']=0.
    out=c.aggregate(rows)
    assert all(x['relative_difference_percent'] is None for x in out['seed_differences'] if x['cohort']=='overall' and x['metric']=='time_nll' and x['right_time_alpha']==1)

def proof():
    return {'quantity_only':True,'time_head_trained':False,'time_head_no_training_verification':{'initial_time_head_sha256':'a'*64,'final_time_head_sha256':'a'*64,
        'optimizer_steps_checked':4,'time_head_parameter_names':['time.weight'],'all_time_head_gradients_none':True,'optimizer_state_absent':True,'time_head_parameters_unchanged':True}}

def test_actual_time_head_proof_required():
    c.verify_zero_proof(proof())
    for key in ('optimizer_steps_checked','all_time_head_gradients_none','optimizer_state_absent','time_head_parameters_unchanged','final_time_head_sha256'):
        p=proof();p['time_head_no_training_verification'].pop(key)
        with pytest.raises(ValueError):c.verify_zero_proof(p)
