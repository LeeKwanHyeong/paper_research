"""Build a continuation-only contract from SHA-verified immutable originals."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
B = PROJECT / 'search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1'
P = PROJECT / 'search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1'
R = B / 'recovery/ssh_info_attempt2/original'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v): return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def inventory():
    import torch
    c=read(P/'execution_contract.json')
    receipt=read(R.parent/'retrieval_receipt.json')
    assert receipt['complete_originals'] and receipt['status']=='sha_verified'
    files=read(R/'manual_recovery_manifest.json')['files']
    assert canonical(files)==receipt['manifest_files_sha256']
    assert all(sha(R/name)==digest for name,digest in files.items())
    assert sha(R.parent/'original.tar.gz')==receipt['archive_sha256']
    for category,prefix in [('source','source/'),('worker','')]:
        assert all(sha(R/(prefix+name))==digest for name,digest in c[category]['files'].items())
        assert canonical(c[category]['files'])==c[category]['files_sha256']
    result={'schema':'titantpp_cnn_gru_recovered_inventory_v1','parent_contract_sha256':canonical(c),
        'archive_sha256':receipt['archive_sha256'],'original_file_count':len(files),
        'original_completed_jobs':[],'remaining_jobs':[],
        'binary_cpu_inference_audit':'not_performed','held_out_test_evaluated':False}
    for j in c['jobs']:
        folder=R/'run'/j['id'];cp=next(folder.rglob('last_epoch_state.pt'))
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        history=read(cp.parent/'history.json')['history']
        assert payload['history']==history and payload['epoch']==len(history)
        assert payload['seed']==42 and payload['backbone']==j['arm']
        best=min(history,key=lambda x:x['val_qty_rmse'])
        assert payload['best_epoch']==best['epoch'] and payload['best_selection_value']==best['val_qty_rmse']
        term=folder/'terminal_manifest.json'
        row={'job':j,'saved_epoch':payload['epoch'],'best_epoch':payload['best_epoch'],
            'source_checkpoint_sha256':sha(cp),'observed_unix':receipt['observed_unix'],
            'selected_history_validation':{k:best[k] for k in ('val_qty_rmse','val_qty_mae','val_time_nll')}}
        if term.exists():
            t=read(term);assert t['scientific_success'] and t['status']=='complete' and t['contract_sha256']==canonical(c)
            assert all(sha(folder/n)==digest for n,digest in t['files'].items())
            e=read(cp.parent/'endpoint_replays.json')
            assert e['status']=='complete' and e['completed_epochs']==payload['epoch'] and e['best_epoch']==payload['best_epoch']
            for label,epoch in [('selected',payload['best_epoch']),('last',payload['epoch'])]:
                q=e[label];assert q['evaluation_scope']=='validation_only' and q['held_out_test_evaluated'] is False
                for metric in ('qty_rmse','qty_mae','time_nll'):
                    assert math.isclose(q[metric],history[epoch-1]['val_'+metric],rel_tol=1e-10,abs_tol=1e-8)
            row.update(terminal_scientific_success=True,terminal_manifest_sha256=sha(term),small_record_SHA_verified=True,
                selected_last_validation_replay_verified=True,completed_unix=t['completed_unix'],
                selected_validation={k:e['selected'][k] for k in ('qty_rmse','qty_mae','time_nll','tail')})
            result['original_completed_jobs'].append(row)
        else:
            row.update(source_epoch=payload['epoch'],source_best_epoch=payload['best_epoch'],
                source_best_state_sha256=payload['best_state_sha256'],source_current_state_sha256=payload['model_state_sha256'],
                source_resume_identity_sha256=canonical(payload['resume_identity']),
                origin_run_relative=str(cp.parent.relative_to(R)),
                terminal_scientific_success=False,host='5080' if j['arm']=='titantpp_gru54' else '5090')
            result['remaining_jobs'].append(row)
    assert len(result['original_completed_jobs'])==7 and len(result['remaining_jobs'])==2
    assert all(j['job']['dataset']=='intermittent_frozen_5000' for j in result['remaining_jobs'])
    write(B/'recovery_inventory.json',result)
    print(json.dumps({'completed':7,'remaining':[(x['job']['id'],x['source_epoch'],x['host']) for x in result['remaining_jobs']]}))

def prepare():
    inventory=read(B/'recovery_inventory.json');parent=read(P/'execution_contract.json')
    adapter=PROJECT/'paper/scripts/titantpp_cnn_gru_resume_runtime.py'
    previous=read(PROJECT/'search_artifacts/titantpp_history_capacity_handoff_20261004_v1/execution_contract.json')
    c={'schema':'titantpp_cnn_gru_resume_dual_v1','parent':{'contract':'parent_execution_contract.json',
        'contract_sha256':canonical(parent),'source_files_sha256':parent['source']['files_sha256'],
        'worker_files_sha256':parent['worker']['files_sha256']},
        **{k:parent[k] for k in ('source','worker','training')},'evaluation_scope':'validation_only',
        'held_out_test_evaluated':False,'replay_tolerance':{'rtol':1e-5,'atol':1e-5},
        'limits':{'automatic_retry':False,'manual_resume_once':True,'workers_per_host':1,'total_wall_seconds':604800,
            'per_condition_seconds':129600,'max_device_memory_fraction':.8},'operation_adapter_sha256':sha(adapter),
        'operation_supervisor_sha256':sha(PROJECT/'paper/scripts/titantpp_cnn_gru_resume_supervisor.py'),
        'hosts':{},'jobs':[],'artifacts':{},'path_mappings':{}}
    for host in ('5080','5090'):
        pre=read(PROJECT/f'reports/titantpp_cnn_gru_resume_dual_20261005_v1/preflight_{host}.json')
        root=pre['proposed_root'];env=dict(previous['hosts'][host]['environment'])
        env.update(PYTHONPATH=root+'/source',SOURCE_REVISION=parent['source']['base_git_revision'])
        c['hosts'][host]={'root':root,'source_root':root+'/source','worker_root':root+'/worker',
            'python':pre['python_executable'],'gpu_uuid':pre['gpu']['uuid'],'gpu_name_contains':host,
            'runtime_expected':{k:v for k,v in pre['runtime'].items() if k!='python'},'environment':env,
            'tmux_binary':previous['hosts'][host]['tmux_binary'],'tmux':B.name+'_'+host}
    for item in inventory['remaining_jobs']:
        job={**item['job'],'host':item['host']};host=job['host'];root=c['hosts'][host]['root']
        c['jobs'].append(job);old=R/item['origin_run_relative']
        c['artifacts'][job['id']]={'origin_root':root+'/originals/'+job['id'],
            'files':{p.name:sha(p) for p in old.iterdir() if p.is_file()},
            **{k:item[k] for k in ('source_checkpoint_sha256','source_epoch','source_best_epoch',
                'source_best_state_sha256','source_current_state_sha256','source_resume_identity_sha256')},
            'origin_contract_sha256':canonical(parent),'terminal_scientific_success':False}
        data=next(d for d in parent['datasets'] if d['dataset_id']==job['dataset'])
        maps={}
        for kind in ('data','split_manifest'):
            original=data['inherited_data_identity'][kind]
            relative=str(Path(original['path']).relative_to('/workspace/'+P.name))
            assert sha(R/relative)==original['sha256']
            maps[kind]={'origin_path':original['path'],'target_path':root+'/'+relative,
                'sha256':original['sha256'],'independently_verified':True}
        c['path_mappings'][job['id']]=maps
    write(B/'execution_contract.json',c);write(B/'parent_execution_contract.json',parent)
    oldapproval=read(B/'approval.json')
    write(B/'user_authorization.json',oldapproval)
    approval={**oldapproval,'manual_resume_once':True,'contract_sha256':canonical(c),
        'parent_contract_sha256':canonical(parent),'hosts':['5080','5090'],
        'job_ids':[j['id'] for j in c['jobs']],'user_instruction':'진행하자: A100 원본 회수와 환경 검증 후 미완료 조건만 5080·5090에서 이어 학습'}
    # Initial authorization is preserved verbatim before its contract-bound administrative form.
    (B/'approval.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2)+'\n')
    start=time.time();permit={'contract_sha256':canonical(c),'approval_sha256':canonical(approval),
        'started_at_unix':start,'deadline_unix':start+c['limits']['total_wall_seconds']}
    write(B/'start_permit.json',permit)
    for host in ('5080','5090'):
        local=B/'deployment'/host;local.mkdir(parents=True,exist_ok=False)
        for name,value in [('execution_contract.json',c),('parent_execution_contract.json',parent),
            ('approval.json',approval),('start_permit.json',permit),('recovery_inventory.json',inventory)]:write(local/name,value)
        for name in parent['source']['files']:
            dst=local/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/'source'/name,dst)
        for name in parent['worker']['files']:
            dst=local/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,dst)
        # Frozen import bootstrap also requires the original empty sample_data sentinel.
        dst=local/'source/sample_data/.keep';dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(R/'source/sample_data/.keep',dst)
        job=next(j for j in c['jobs'] if j['host']==host);item=next(i for i in inventory['remaining_jobs'] if i['host']==host)
        shutil.copytree(R/item['origin_run_relative'],local/'originals'/job['id'])
        for mapping in c['path_mappings'][job['id']].values():
            relative=str(Path(mapping['origin_path']).relative_to('/workspace/'+P.name))
            dst=local/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/relative,dst)
        for name in ('titantpp_cnn_gru_resume_runtime.py','titantpp_cnn_gru_resume_supervisor.py'):
            dst=local/'operation'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/'paper/scripts'/name,dst)
        files={str(p.relative_to(local)):sha(p) for p in sorted(local.rglob('*')) if p.is_file()}
        write(local/'package_manifest.json',{'files':files,'files_sha256':canonical(files),'contract_sha256':canonical(c)})
        archive=B/'deployment'/('package_'+host+'.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for name in [*files,'package_manifest.json']:t.add(local/name,arcname=name,recursive=False)
        write(B/'deployment'/('package_'+host+'_receipt.json'),{'archive_sha256':sha(archive),'files':len(files),'contract_sha256':canonical(c)})
    print(json.dumps({'contract_sha256':canonical(c),'jobs':c['jobs']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('inventory','prepare'),required=True)
    args=parser.parse_args();inventory() if args.mode=='inventory' else prepare()
