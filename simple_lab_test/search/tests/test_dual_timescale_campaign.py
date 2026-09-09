"""Fail-closed scheduling evidence and fixed metric/cost decisions."""
import copy
import json
from pathlib import Path
import pytest

from paper.scripts.run_dual_timescale_campaign import (
    CONTRACT, BACKBONE, checked_cuda_xml, evaluate_gate, read_json, verify_manifest,
)
from paper.scripts.profile_dual_timescale import summarize


def test_frozen_quantity_gate_keeps_strict_rmse_and_all_guardrails():
    c=read_json(CONTRACT)
    for b in c['B_validation_reference']['datasets'].values():
        m=dict(b)
        assert evaluate_gate(m,b)['status']=='failed'
        m['raw_rmse']*=.99
        assert evaluate_gate(m,b)['status']=='passed'
        for key,factor in [('overall_mae',1.01),('body_mae',1.02),('gt_p99_mae',1.02)]:
            changed=dict(m);changed[key]=b[key]*factor+1e-8
            assert evaluate_gate(changed,b)['status']=='failed'
        changed=dict(m);changed['clamped_time_loss']=b['clamped_time_loss']+.01000001
        assert evaluate_gate(changed,b)['status']=='failed'


def test_cuda_evidence_rejects_skip_fail_and_absent_cuda(tmp_path):
    xml=tmp_path/'tests.xml'
    for body in ['<testcase name="cpu"/>','<testcase name="x[cuda]"><skipped/></testcase>',
                 '<testcase name="x[cuda]"><failure/></testcase>']:
        xml.write_text('<testsuite>'+body+'</testsuite>')
        with pytest.raises(ValueError): checked_cuda_xml(xml)
    xml.write_text('<testsuite><testcase name="x[cuda]"/></testsuite>')
    assert checked_cuda_xml(xml)['executed_cuda_cases']==1


def test_cost_gate_rejects_incomplete_grid_and_excess_cost():
    c=read_json(CONTRACT)['cost_gate'];rows=[]
    for b in ['titantpp',BACKBONE]:
        for n in c['sequence_lengths']:
            for r in range(c['repeats']):
                rows.append({'backbone':b,'length':n,'repeat':r,'step_seconds':[1.]*c['measured_steps'],
                             'median_step_seconds':1.,'peak_allocated_bytes':100,'parameter_count':100,'finite':True})
    assert summarize(rows,c)['status']=='passed'
    with pytest.raises(ValueError):summarize(rows[:-1],c)
    with pytest.raises(ValueError):summarize(rows+[rows[0]],c)
    for x in rows:
        if x['backbone']==BACKBONE:x['median_step_seconds']=2.01;x['step_seconds']=[2.01]*c['measured_steps']
    assert summarize(rows,c)['status']=='failed'


def test_deployment_rejects_wrong_host_or_contract_before_dataset_access(tmp_path):
    path=tmp_path/'manifest.json'
    value={'source_revision':'a'*40,'host_role':'5080','evaluation_scope':'validation_only',
           'contract_sha256':'bad','files':{}}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Host'):verify_manifest(path,'a'*40)
    value['host_role']='5090';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Contract'):verify_manifest(path,'a'*40)


def test_actual_artifact_restore_rejects_incompatible_optimizer_moments():
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from paper.scripts.run_dual_timescale_campaign import restore_checkpoint
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, _ = build_count_aware_model(BACKBONE, hidden_dim=64, train_log_mean=2.,
                                          max_seq_len=64, time_intercept_limit=300.)
        optimizer = build_optimizer(model, lr=.001)
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        payload = {'model_state_dict': model.state_dict(),
                   'model_state_sha256': canonical_state_dict_sha256(model.state_dict()),
                   'interface_meta': {'train_target_mean':2.,'train_target_std':1.},
                   'optimizer_state_dict':optimizer.state_dict()}
        receipt = restore_checkpoint(payload, read_json(CONTRACT), 'insta_market_basket', optimizer_required=True)
        assert receipt['strict_model_restore'] and receipt['optimizer_parameters_restored'] > 0
        broken = copy.deepcopy(payload)
        first = next(iter(broken['optimizer_state_dict']['state'].values()))
        first['exp_avg'] = torch.zeros(123)
        with pytest.raises(ValueError, match='moment shape'):
            restore_checkpoint(broken, read_json(CONTRACT), 'insta_market_basket', optimizer_required=True)
    finally:
        torch.set_num_threads(previous)
