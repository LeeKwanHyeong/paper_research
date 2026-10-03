import copy,math
import numpy as np
import pytest
from paired_statistics import align_panel,sufficient_statistics,metrics_from_weights,draw_weights,paired_bootstrap

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
