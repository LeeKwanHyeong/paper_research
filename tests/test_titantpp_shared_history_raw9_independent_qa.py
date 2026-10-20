"""Independent local-only raw9 contract, predecessor, and aggregation review."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import math
import statistics
import pytest
from paper.scripts import prepare_titantpp_shared_history_raw9 as p
from paper.scripts import titantpp_shared_history_raw9_runtime as r
from paper.scripts import manage_titantpp_shared_history_raw9 as m
from paper.scripts import compare_titantpp_shared_history_raw9 as c


def fixture_contract():
    value = deepcopy(p.read(p.PARENT / 'execution_contract.json'))
    raw = p.read(p.RAW_CONTRACT)
    value.update(schema='titantpp_shared_history_raw9_v1', data_variant='original_uncapped',
                 quantity_capping_applied=False, common_condition=deepcopy(p.CONDITION))
    value['datasets'] = [deepcopy(d) for d in raw['datasets'] if d['dataset_id'] in p.DATASETS]
    for d in value['datasets']:
        d['inherited_data_identity']['populations'] = {k:v for k,v in d['inherited_data_identity']['populations'].items() if k in ('train','validation')}
    value['jobs'] = [{'id':f'{d}__{p.ARM}__seed{s}__losslog_mse', 'dataset':d,'arm':p.ARM,'seed':s,'host':'5080', **p.CONDITION} for d in p.DATASETS for s in (42,52,62)]
    value['architecture'].update(arms=[p.ARM],time_alpha=1.0,quantity_loss_mode='log_mse')
    value['baseline_validation_references'] = deepcopy(p.read(p.BASELINES)['references'])
    for ref in value['baseline_validation_references']:
        ref['time_head_trained'] = True
    value['predecessor'] = {'contract_raw_sha256':p.PARENT_SHA,'expected_count':18,
        'contract_path':str(p.PARENT/'execution_contract.json'),'receipt_path':str(p.PARENT/'retrieval/final_receipt.json')}
    return value


def test_runtime_admits_shape_emitted_by_prepare():
    contract=fixture_contract()
    assert len(r.validate_contract(contract,'5080')) == 9
    assert r.STAGES == ('train','validation')
    assert r._impl.WORKER_MODULE == 'paper.scripts.run_titantpp_shared_history_raw9'
    assert r._impl.validate_contract is r.validate_contract
    assert r._impl.verify_receipt is r.verify_receipt
    assert r._impl.predecessor_gate is r.predecessor_gate


@pytest.mark.parametrize('case', ['alpha0','cap','data_cap','job_alpha0','test','host5090','seed32','duplicates','new_arm','deadline','optimizer','budget','baselinealpha0','empty_source','empty_input','wrong_predecessor'])
def test_mutated_runtime_contract_is_rejected(case):
    x=fixture_contract()
    if case=='alpha0':x['common_condition']['time_alpha']=0.
    elif case=='cap':x['quantity_capping_applied']=True
    elif case=='data_cap':x['datasets'][0]['cap_value']=10.
    elif case=='job_alpha0':x['jobs'][0]['time_alpha']=0.
    elif case=='test':x['held_out_test_authorized']=True
    elif case=='host5090':x['jobs'][0]['host']='5090'
    elif case=='seed32':x['jobs'][0]['seed']=32
    elif case=='duplicates':x['jobs'][1]=deepcopy(x['jobs'][0])
    elif case=='new_arm':x['jobs'][0]['arm']='titantpp_history_mlp_width16'
    elif case=='deadline':x['hosts']['5080']['absolute_host_deadline_unix']+=1
    elif case=='optimizer':x['datasets'][0]['optimizer']['lr']=.01
    elif case=='budget':x['training']['maximum_epochs']=301
    elif case=='baselinealpha0':x['baseline_validation_references'][0]['time_alpha']=0.
    elif case=='empty_source':x['source']['files']={}
    elif case=='empty_input':x['input_files']={}
    elif case=='wrong_predecessor':x['predecessor']['contract_raw_sha256']='0'*64
    with pytest.raises((ValueError,KeyError)):
        r.validate_contract(x,'5080')


def test_deploy_stops_before_any_remote_operation_when_cap_incomplete():
    with patch.object(m,'verify_local',side_effect=ValueError('Cap18 incomplete')) as check, patch.object(m._impl,'ssh',side_effect=AssertionError('SSH forbidden')) as remote:
        with pytest.raises(ValueError,match='Cap18 incomplete'):m.deploy()
        check.assert_called_once_with(archive=True, require_predecessor_complete=True)
        remote.assert_not_called()


def test_deploy_duplicate_attempt_cannot_reach_ssh(tmp_path):
    x=fixture_contract()
    (tmp_path/'execution_contract.json').write_text(json.dumps(x))
    (tmp_path/'attempts').mkdir()
    (tmp_path/'attempts/5080.json').write_text('{}')
    with patch.object(m,'verify_local',return_value=x),patch.object(m,'BUNDLE',tmp_path),patch.object(m._impl,'ssh',side_effect=AssertionError('SSH forbidden')) as remote:
        with pytest.raises(FileExistsError):m.deploy()
        remote.assert_not_called()


def test_finite_queue_refuses_existing_queue(tmp_path):
    (tmp_path/'QUEUE_STARTED.json').write_text('{}')
    with patch.object(m,'REPORT',tmp_path),patch.object(m,'verify_local',return_value={}),patch.object(m.subprocess,'Popen',side_effect=AssertionError('Spawn forbidden')):
        with pytest.raises(ValueError,match='already started'):m.start_queue()


def fabricated_records():
    rows=[]
    for d in c.DATASETS:
        for s,i in zip(c.SEEDS,(0,1,2)):
            for model in c.MODELS:
                v=2.+i+(0.25 if model==c.SHARED else 0.)
                rows.append({'dataset':d,'seed':s,'model':model,'host':'5080',**c.CONDITION,
                    'data_variant':'original_uncapped','held_out_test_evaluated':False,'selected_epoch':3,
                    'target_identity':{'target_count':5,'target_identity_sha256':'a'*64,'target_quantity_sha256':'b'*64},
                    'metrics':{g:{'count':5 if g=='overall' else 2,'qty_rmse':v,'qty_mae':v/2,'time_nll':-v} for g in c.COHORTS}})
    return rows


def test_aggregate_independent_arithmetic_and_all_deteriorations():
    out=c.aggregate(fabricated_records())
    assert len(out['summary'])==36 and len(out['seed_differences'])==54
    for row in out['summary']:
        factor={'qty_rmse':1.,'qty_mae':.5,'time_nll':-1.}[row['metric']]
        shift=.25 if row['model']==c.SHARED else 0.
        assert row['mean']==pytest.approx((3.+shift)*factor)
        assert row['sample_sd_ddof1']==pytest.approx(abs(factor))
    for row in out['seed_differences']:
        factor={'qty_rmse':1.,'qty_mae':.5,'time_nll':-1.}[row['metric']]
        assert row['absolute_difference']==pytest.approx(.25*factor)
        assert row['relative_difference_percent']==pytest.approx(.25*factor/abs(row['reference_value'])*100)
        assert row['worsened'] is (factor>0)
    assert out['all_datasets_both_quantity_means_improved'] is False


@pytest.mark.parametrize('case',['missing','duplicate','cap','alpha0','test','target','qtytruth','nan','negative','count','seedbool'])
def test_aggregate_rejects_mixed_or_invalid_endpoints(case):
    rows=fabricated_records()
    if case=='missing':rows.pop()
    elif case=='duplicate':rows.append(deepcopy(rows[0]))
    elif case=='cap':rows[0]['data_variant']='cap'
    elif case=='alpha0':rows[0]['time_alpha']=0.
    elif case=='test':rows[0]['held_out_test_evaluated']=True
    elif case=='target':rows[0]['target_identity']['target_identity_sha256']='c'*64
    elif case=='qtytruth':rows[0]['target_identity']['target_quantity_sha256']='c'*64
    elif case=='nan':rows[0]['metrics']['overall']['qty_rmse']=float('nan')
    elif case=='negative':rows[0]['metrics']['overall']['qty_mae']=-1.
    elif case=='count':rows[0]['metrics']['overall']['count']=4
    elif case=='seedbool':rows[0]['seed']=True
    with pytest.raises(ValueError):c.aggregate(rows)


def test_zero_baseline_relative_and_empty_tail_are_na():
    rows=fabricated_records()
    for row in rows:
        row['metrics']['tail']={'count':0, **{k:None for k in c.METRICS}}
        if row['model']==c.BASELINE:row['metrics']['overall']['qty_mae']=0.
    out=c.aggregate(rows)
    assert all(x['mean'] is None for x in out['summary'] if x['cohort']=='tail')
    assert all(x['relative_difference_percent'] is None for x in out['seed_differences'] if x['metric']=='qty_mae' or x['cohort']=='tail')
