import copy,math
import json
import numpy as np
import pytest
from paired_statistics import align_panel,sufficient_statistics,metrics_from_weights,draw_weights,paired_bootstrap
from sample_contract import nonlearned_predictions, target_metadata, SIMPLE

MODELS=['titantpp_history_mlp','baseline'];SEEDS=[42,52,62]

def fixture():
    rows=[]
    for m in MODELS:
        for si,s in enumerate(SEEDS):
            for i,(entity,q) in enumerate([('a',1.),('a',3.),('b',7.),('c',2.)]):
                rows.append({'dataset':'synthetic','model':m,'seed':s,'target_id':str(i),'entity_id':entity,'site_id':entity,'seq':i,'split':'validation','raw_quantity':q,'predicted_raw_quantity':q+(1 if m==MODELS[0] else 2)*(si+1)*(i+1),'recorded_gap':1.,'time_nll':float(i+si),'time_bucket':f'2020-01-01T{(i*2):02d}:00:00','evaluation_scope':'synthetic'})
    return rows

def test_weighted_cluster_matches_expanded_targets_and_seed_RMSE():
    meta,loss=align_panel(fixture(),MODELS,SEEDS)
    labels,count,sums=sufficient_statistics(meta,loss,'cluster')
    got=metrics_from_weights(count,sums,np.array([2,0,1]))
    # Independent expansion: a twice, c once, b omitted.
    for mi,scale in enumerate([1,2]):
        for si in range(3):
            errors=np.array([1,2,1,2,4])*scale*(si+1)
            assert got[mi,si,0]==np.abs(errors).mean()
            assert got[mi,si,1]==pytest.approx(math.sqrt(np.mean(errors**2)))
    assert got[:, :,1].mean(1)[0]!=pytest.approx(math.sqrt((got[0,:,1]**2).mean()))

@pytest.mark.parametrize('mutation',['missing_target','duplicate','truth','scope','nonfinite','missing_seed'])
def test_pairing_rejects_invalid_records(mutation):
    rows=fixture()
    if mutation=='missing_target':rows.pop()
    elif mutation=='duplicate':rows.append(dict(rows[0]))
    elif mutation=='truth':rows[-1]['raw_quantity']+=1
    elif mutation=='scope':rows[0]['evaluation_scope']='test'
    elif mutation=='nonfinite':rows[0]['predicted_raw_quantity']=float('nan')
    else:rows=[r for r in rows if not(r['model']=='baseline' and r['seed']==62)]
    with pytest.raises(ValueError):align_panel(rows,MODELS,SEEDS)

def test_fixed_rng_reproducible_and_identical_models_pair_exactly():
    rows=fixture()
    for r in rows:
        if r['model']=='baseline':r['predicted_raw_quantity']=r['raw_quantity']+(r['predicted_raw_quantity']-r['raw_quantity'])/2
    a=paired_bootstrap(rows,MODELS,SEEDS,draws=1000,anchor='baseline')
    b=paired_bootstrap(rows,MODELS,SEEDS,draws=1000,anchor='baseline')
    assert a==b
    assert a['intervals']['baseline']['qty_rmse']['pointwise_95_percentile']==[0.,0.]
    assert a['intervals']['baseline']['relative_RMSE_improvement_percent']['pointwise_95_percentile']==[0.,0.]

def test_calendar_weights_include_empty_hours_share_all_cells():
    meta,loss=align_panel(fixture(),MODELS,SEEDS)
    labels,count,sums=sufficient_statistics(meta,loss,'calendar')
    assert len(labels)==7 and list(count)==[1,0,1,0,1,0,1]
    got=draw_weights(np.random.Generator(np.random.PCG64(9)),7,'calendar',3)
    rng=np.random.Generator(np.random.PCG64(9));starts=[int(x) for x in rng.integers(0,7,size=3)]
    manual=([((s+j)%7) for s in starts for j in range(3)])[:7]
    assert list(got)==[manual.count(i) for i in range(7)]
    two=copy.deepcopy(meta)+copy.deepcopy(meta);loss2=np.concatenate([loss,loss])
    _,count2,sums2=sufficient_statistics(two,loss2,'calendar')
    np.testing.assert_allclose(metrics_from_weights(count,sums,got),metrics_from_weights(count2,sums2,got))

def test_short_calendar_span_and_missing_timestamp_do_not_fallback():
    rows=fixture();r=paired_bootstrap(rows,MODELS,SEEDS,kind='calendar',block_hours=168,draws=10)
    assert r['draws_completed']==0 and r['intervals'] is None
    for x in rows:x['time_bucket']=None
    with pytest.raises(ValueError):paired_bootstrap(rows,MODELS,SEEDS,kind='calendar',draws=10)

def test_zero_comparator_RMSE_keeps_absolute_difference_only():
    rows=fixture()
    for r in rows:
        if r['model']=='baseline':r['predicted_raw_quantity']=r['raw_quantity']
    result=paired_bootstrap(rows,MODELS,SEEDS,draws=100)
    assert result['intervals']['baseline']['relative_RMSE_improvement_percent']['point'] is None
    assert result['intervals']['baseline']['qty_rmse']['difference']>0

def test_site_clusters_combine_multiple_entities():
    rows=fixture()
    for r in rows:r['site_id']='s1' if r['entity_id']in ['a','b'] else 's2'
    meta,loss=align_panel(rows,MODELS,SEEDS)
    labels,count,sums=sufficient_statistics(meta,loss,'cluster','site_id')
    assert labels==['s1','s2'] and list(count)==[3,1]

def test_seed_SD_is_ddof1_and_order_does_not_change_pairing():
    rows=fixture();a=paired_bootstrap(rows,MODELS,SEEDS,draws=50)
    b=paired_bootstrap(list(reversed(rows)),MODELS,SEEDS,draws=50)
    assert a==b
    values=[a['seed_metrics'][MODELS[0]][str(s)]['qty_rmse'] for s in SEEDS]
    assert a['sample_sd'][MODELS[0]]['qty_rmse']==pytest.approx(np.std(values,ddof=1))

def test_non_hour_timestamp_is_not_silently_rounded():
    rows=fixture()
    for r in rows:
        if r['target_id']=='0':r['time_bucket']='2020-01-01T00:15:00'
    with pytest.raises(ValueError):paired_bootstrap(rows,MODELS,SEEDS,kind='calendar',block_hours=1,draws=10)


def deterministic_fixture():
    rows=fixture()
    refs=[]
    for row in rows:
        if row['model']==MODELS[0] and row['seed']==42:
            refs.append({**row,'model':'last_quantity','seed':None,
                         'predicted_raw_quantity':0.,'time_nll':None})
    return rows+refs


def test_deterministic_reference_is_one_vector_with_no_NLL_or_seed_SD():
    rows=deterministic_fixture();models=MODELS+['last_quantity']
    result=paired_bootstrap(rows,models,SEEDS,draws=1000,deterministic_models=['last_quantity'])
    assert result['count']==4
    assert list(result['seed_metrics']['last_quantity'])==['deterministic']
    assert result['sample_sd']['last_quantity']==dict.fromkeys(('qty_mae','qty_rmse','time_nll'))
    assert result['means']['last_quantity']['qty_rmse']==pytest.approx(math.sqrt((1+9+49+4)/4))
    assert result['means']['last_quantity']['time_nll'] is None
    assert result['intervals']['last_quantity']['time_nll'] is None
    json.dumps(result,allow_nan=False)
    # Deterministic references do not alter any learned pair or bootstrap draw.
    old=paired_bootstrap(fixture(),MODELS,SEEDS,draws=1000)
    assert result['resampling_weights_sha256']==old['resampling_weights_sha256']
    assert result['intervals']['baseline']==old['intervals']['baseline']


@pytest.mark.parametrize('mutation',['seed','time_nll','missing_target','duplicate'])
def test_deterministic_reference_rejects_ambiguous_or_incomplete_vectors(mutation):
    rows=deterministic_fixture()
    if mutation=='seed':rows[-1]['seed']=42
    elif mutation=='time_nll':rows[-1]['time_nll']=0.
    elif mutation=='missing_target':rows.pop()
    else:rows.append(dict(rows[-1]))
    with pytest.raises(ValueError):align_panel(rows,MODELS+['last_quantity'],SEEDS,['last_quantity'])


@pytest.mark.parametrize('split',['test','train','validation/test',None])
def test_scope_label_cannot_disguise_locked_split(split):
    rows=fixture();rows[0]['split']=split;rows[0]['evaluation_scope']='validation_smoke'
    with pytest.raises(ValueError):align_panel(rows,MODELS,SEEDS)


@pytest.mark.parametrize('field',['recorded_event_key','data_file_sha256'])
def test_prepared_row_identity_and_data_bytes_must_match(field):
    rows=fixture();rows[-1][field]='different'
    with pytest.raises(ValueError):align_panel(rows,MODELS,SEEDS)


def test_simple_predictions_exclude_target_and_masked_padding():
    result=nonlearned_predictions([[float('nan'),1.,3.,100.],[9.,0.,4.,999.]],
                                  [[False,True,True,True],[True,False,True,True]])
    assert result==[{SIMPLE[0]:3.,SIMPLE[1]:2.},{SIMPLE[0]:4.,SIMPLE[1]:6.5}]
    changed=nonlearned_predictions([[2.,1.,3.,100000.]],[[False,True,True,True]])
    assert changed[0]==result[0]


@pytest.mark.parametrize('q,mask',[([1.],[True]),([1.,2.],[True]),([1.,2.],[1,1]),([float('inf'),1.],[True,True])])
def test_simple_predictions_reject_invalid_history(q,mask):
    with pytest.raises(ValueError):nonlearned_predictions([q],[mask])


def test_recorded_event_key_binds_date_and_data_but_not_quantity():
    row={'chronological_split':'validation','demand_dt':'2015-01-25T01:00:00','demand_qty':12.}
    first=target_metadata('taxi','a',1,row,'a'*64)
    changed=target_metadata('taxi','a',1,{**row,'demand_qty':500.},'a'*64)
    assert first==changed
    assert target_metadata('taxi','a',1,row,'b'*64)['target_id']!=first['target_id']
    assert target_metadata('taxi','a',1,{**row,'demand_dt':'2015-01-26T01:00:00'},'a'*64)['target_id']!=first['target_id']
    assert first['event_key_scope']=='prepared_dataset_row_not_upstream_raw_lineage'
    with pytest.raises(ValueError):target_metadata('taxi','a',1,{**row,'chronological_split':'test'},'a'*64)
