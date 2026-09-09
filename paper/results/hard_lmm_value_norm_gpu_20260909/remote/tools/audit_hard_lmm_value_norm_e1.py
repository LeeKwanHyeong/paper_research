#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, hashlib, json, math, subprocess, sys
from pathlib import Path
import xml.etree.ElementTree as ET

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def finite_tree(value):
    import torch
    if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_tree(v) for v in value)
    if isinstance(value,float): return math.isfinite(value)
    return True

def require(condition,message):
    if not condition: raise AssertionError(message)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-root',type=Path,required=True); ap.add_argument('--artifact-root',type=Path,required=True); a=ap.parse_args()
    sys.path.insert(0,str(a.source_root))
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    from models.TPPs.CountAwareTitanValueNorm import VALUE_NORM_ALPHA_KEY
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from paper.scripts.run_hard_lmm_backbone_candidate_campaign import audit_job, save_json
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    rev='956603f16ca3540e2012a461a494ea1ec905102d'; backbone='titantpp_hard_memory_value_norm'; role='hard_lmm_value_norm_candidate'
    archive=Path('/home/leekwanhyeong/workspace/tmp/paper_research_vnc_956603f.tar.gz')
    manifest=Path('/home/leekwanhyeong/workspace/tmp/paper_research_vnc_956603f_source_manifest.sha256')
    require(sha256(archive)=='effaa12f18b7e6ef126218c8b2e6cbeee91a8a1c2159df677955ccc8e6ee404d','archive digest drift')
    require(sha256(manifest)=='177286e0e405e21885214ee0c6acb0fada158465a1715743f399facfd5a9af9f','manifest digest drift')
    with gzip.open(archive,'rb') as stream:
        cp=subprocess.run(['git','get-tar-commit-id'],stdin=stream,text=True,capture_output=True,check=True)
    require(cp.stdout.strip()==rev,'embedded commit drift')
    manifest_check=(a.artifact_root/'source_manifest_check.txt').read_text().splitlines()
    require(len(manifest_check)==2130 and all(line.endswith(': OK') for line in manifest_check),'source manifest check incomplete')
    expected_data={
      'intermittent_frozen_5000':('sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet','85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f','sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json','393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04'),
      'yellow_trip_hourly':('sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet','b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46','sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json','4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85'),
      'insta_market_basket':('sample_data/insta_market_basket/instacart_marked_target_with_split.parquet','06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2','sample_data/insta_market_basket/instacart_marked_target_split_manifest.json','6c6cdd41f847878fbb405b73dfa038fbb7a88ad53df6843b0cc9e64531a8b71d')}
    data_proof={}
    for dataset,(dp,dh,mp,mh) in expected_data.items():
        actual=(sha256(a.source_root/dp),sha256(a.source_root/mp)); require(actual==(dh,mh),dataset+' data checksum drift'); data_proof[dataset]={'data_sha256':actual[0],'split_manifest_sha256':actual[1]}
    suite=ET.parse(a.artifact_root/'cuda_qualification.xml').getroot(); cases=list(suite.iter('testcase'))
    failures=sum(1 for c in cases if c.find('failure') is not None); errors=sum(1 for c in cases if c.find('error') is not None); skipped=sum(1 for c in cases if c.find('skipped') is not None)
    cuda_cases=[c.attrib.get('name','') for c in cases if 'cuda' in (c.attrib.get('name','')+' '+c.attrib.get('classname','')).lower()]
    require(len(cases)==71 and failures==errors==skipped==0 and len(cuda_cases)>=3,'CUDA qualification drift')
    profile_path=a.artifact_root/'cost_profile_v2/profile.json'; profile=json.loads(profile_path.read_text())
    require(profile.get('status')=='passed' and profile.get('cost_gate',{}).get('status')=='passed','cost gate failed')
    for row in profile['rows']:
        if row['backbone']==backbone:
            g=float(row['gradient_audit']['alpha_raw_gradient']); lp=row['learning_path']; require(math.isfinite(g) and g!=0 and lp['status']=='passed' and lp['state_digest_restored'] and lp['zero_alpha_quantity_max_abs_delta']>0,'candidate profile learning path failed')
    jobs={'insta_market_basket':a.artifact_root/'jobs/e1_insta_market_basket','yellow_trip_hourly':a.artifact_root/'jobs/e1_yellow_trip_hourly','intermittent_frozen_5000':a.artifact_root/'jobs/e1_intermittent_frozen_5000'}
    candidate={'backbone':backbone,'model_role':role}; result={'status':'running','contract_id':'hard_lmm_value_norm_consistent_v1','source_revision':rev,'source_archive_sha256':sha256(archive),'source_manifest_sha256':sha256(manifest),'source_file_count':len(manifest_check),'held_out_test_evaluated':False,'data':data_proof,'cuda_qualification':{'status':'passed','test_count':len(cases),'failures':failures,'errors':errors,'skipped':skipped,'cuda_cases':cuda_cases,'xml_sha256':sha256(a.artifact_root/'cuda_qualification.xml')},'cost_profile':{'status':'passed','profile_sha256':sha256(profile_path),'gate':profile['cost_gate']},'jobs':{}}
    for dataset,output in jobs.items():
        run=output/'runs'/backbone/'count_only_log_regression'/'seed_42'; summary=json.loads((run/'summary.json').read_text()); launch=json.loads((output/'launch_contract.json').read_text()); history=json.loads((run/'history.json').read_text())['history']
        require(launch['model_role']==role and launch['backbones']==[backbone] and launch['source_revision']==rev and launch['status']=='complete','launch route/status drift')
        require(launch['held_out_test_evaluated'] is False and summary['held_out_test_evaluated'] is False,'held-out scope drift')
        require(len(history)==1 and history[0]['train_all_finite'] is True and history[0]['train_pre_clip_grad_norm_mean']>0 and history[0]['train_pre_clip_grad_norm_max']>0,'training gradient/finite proof failed')
        best_path=run/'best_val_qty_rmse_model.pt'; last_path=run/'last_epoch_state.pt'; best=torch_load_checkpoint(best_path,map_location='cpu'); last=torch_load_checkpoint(last_path,map_location='cpu')
        for payload in (summary,best,last): validate_checkpoint_route(payload,backbone)
        best_sha=canonical_state_dict_sha256(best['model_state_dict']); require(best_sha==best['model_state_sha256']==summary['checkpoint_state_sha256']==last['best_state_sha256']==canonical_state_dict_sha256(last['best_state_dict']),'best state digest mismatch')
        require(last['model_state_sha256']==canonical_state_dict_sha256(last['model_state_dict']),'last state digest mismatch')
        require(summary['initial_state_sha256']==best['initial_state_sha256']==last['initial_state_sha256'],'initial state mismatch')
        require(summary['resume_identity']==best['resume_identity']==last['resume_identity'],'resume identity mismatch')
        require(isinstance(last.get('optimizer_state_dict'),dict) and isinstance(last.get('rng_state'),dict) and isinstance(last.get('train_loader_generator_state'),torch.Tensor),'resume payload incomplete')
        interface=launch['interfaces']['count_only_log_regression']; th=launch['time_head']
        model,_=build_count_aware_model(backbone,hidden_dim=launch['hidden_dim'],train_log_mean=interface['train_target_mean'],train_log_std=interface['train_target_std'],max_seq_len=launch['max_seq_len'],quantity_variant='count_only_log_regression',lambda_tail=0.0,time_head_mode=th['mode'],time_scale=th['time_scale'],time_w_max=th['time_w_max'],time_intercept_limit=th['time_intercept_limit'],time_initial_intercept=th['time_initial_intercept'],time_wd_safety_limit=th['time_wd_safety_limit'])
        model.load_state_dict(best['model_state_dict'],strict=True); require(canonical_state_dict_sha256(model.state_dict())==best_sha,'strict best restore mismatch')
        model.load_state_dict(last['model_state_dict'],strict=True); require(canonical_state_dict_sha256(model.state_dict())==last['model_state_sha256'],'strict last restore mismatch')
        optimizer=build_optimizer(model,lr=launch['lr'],time_head_lr_multiplier=th['time_head_lr_multiplier']); optimizer.load_state_dict(last['optimizer_state_dict']); require(finite_tree(optimizer.state_dict()),'optimizer restore nonfinite')
        alpha_parameter=dict(model.named_parameters())[VALUE_NORM_ALPHA_KEY]; alpha_state=optimizer.state.get(alpha_parameter,{}); require(alpha_state and finite_tree(alpha_state) and float(alpha_state['exp_avg'].abs().max())>0,'alpha optimizer gradient state missing')
        alpha=best['model_state_dict'][VALUE_NORM_ALPHA_KEY]; require(alpha.shape==() and alpha.is_floating_point() and torch.isfinite(alpha) and float(alpha)!=0.0,'learned alpha missing')
        audit=audit_job(output,candidate=candidate,dataset=dataset,source_revision=rev,expected_epochs=1)
        audit.update({'best_checkpoint_file_sha256':sha256(best_path),'last_checkpoint_file_sha256':sha256(last_path),'checkpoint_route_verified':True,'strict_model_restore_verified':True,'optimizer_rng_loader_restore_verified':True,'alpha_optimizer_state_verified':True,'alpha_raw':float(alpha),'train_pre_clip_grad_norm_mean':float(history[0]['train_pre_clip_grad_norm_mean']),'train_gradient_clip_fraction':float(history[0]['train_gradient_clip_fraction']),'elapsed_seconds':float(summary['elapsed_seconds'])})
        save_json(output/'audit.json',audit); result['jobs'][dataset]=audit
    result['status']='passed'; save_json(a.artifact_root/'e1_audit.json',result); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
