"""Prepare an isolated recovery bundle from verified parent originals (local only)."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1'
SCHEMA = 'titantpp_contribution_history_recovery_20261010_v1'
REMOTE = '/workspace/' + SCHEMA + '_a100'
PARENT_SHA = '4fcf1923c2bfbec8ad7b4bf7af506ee4f9b254dd8695e6b77035dd913b5bc12a'
MANIFEST_SHA = 'd25ca3393c1c9a295ebf6b37d0695f8b700adf8af68b777a4e09b57c245f2091'
ARCHIVE_SHA = '83d0d3f4fabd89554a0cc5acc86dc447a9919b22d5247e27a1aaea65ecc01235'
SOURCE_SHA = 'c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a'
OP_SHA = 'bec6a925503b43e083dcc02bc03b2fa25dc43b98be0803d9e135f97990a97a33'
PARENT_MIN_BILLED = 30.124985997681506

def require(ok, msg):
    if not ok: raise ValueError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: f.write(value)
def write_json(path,value): write(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def replace_node(text,name,replacement):
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])
def function(text,name):
    return ast.get_source_segment(text,next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name))

RECOVERY_FUNCTIONS = r'''
def parent_folder(c,root,job):
    return root/'recovery_parent'/'run'/job['id']

def result_folder(c,root,job):
    return parent_folder(c,root,job) if job['id'] in c['recovery']['reused_jobs'] else root/'run'/job['id']

def result_contract(c,job):
    return c['recovery']['parent_contract_sha256'] if job['id'] in c['recovery']['reused_jobs'] else sha_json(c)

def validate_recovery(c,root):
    recovery=c['recovery']
    require(recovery['parent_contract_sha256']=='4fcf1923c2bfbec8ad7b4bf7af506ee4f9b254dd8695e6b77035dd913b5bc12a','Recovery parent changed')
    require(len(recovery['reused_jobs'])==10 and len(recovery['resumed_jobs'])==2 and len(recovery['fresh_jobs'])==15,'Recovery counts changed')
    all_ids=recovery['reused_jobs']+list(recovery['resumed_jobs'])+recovery['fresh_jobs']
    require(len(set(all_ids))==27 and set(all_ids)=={j['id'] for j in c['jobs']},'Recovery execution ownership changed')
    require(recovery['physical_gpu_migration'] is True and recovery['automatic_retry'] is False,'Recovery authority changed')
    for name,digest in recovery['parent_files'].items():
        require(sha_file(safe_relative(root/'recovery_parent',name))==digest,'Preserved parent original changed: '+name)
    return recovery

def prepare_resume(c,root,job,folder):
    entry=c['recovery']['resumed_jobs'].get(job['id'])
    if entry is None:
        folder.mkdir(parents=True,exist_ok=False)
        return None
    validate_recovery(c,root)
    source=parent_folder(c,root,job)
    require(not folder.exists(),'Recovery output already exists; explicit attempt only')
    run=source/'runs'/job['arm']/VARIANT/f"seed_{job['seed']}"
    receipt=read(run/'server_checkpoint_receipt.json')
    require(sha_file(run/'server_checkpoint_receipt.json')==entry['receipt_sha256'] and receipt['epoch']==entry['epoch']
        and receipt['contract_sha256']==c['recovery']['parent_contract_sha256'],'Resume checkpoint receipt mismatch')
    for name,digest in receipt['files'].items():require(sha_file(safe_relative(run,name))==digest,'Resume sidecar SHA mismatch '+name)
    require(not (source/'terminal_manifest.json').exists(),'Completed job cannot resume')
    shutil.copytree(source,folder)
    common.write_json(folder/'checkpointreceipt_parent.json',{'parent':entry,'parent_contract_sha256':c['recovery']['parent_contract_sha256'],
        'recovery_contract_sha256':sha_json(c),'physical_gpu_migration':True,'source_gpu_uuid':c['recovery']['source_gpu_uuid'],
        'destination_gpu_uuid':c['hosts']['a100']['gpu_uuid'],'same_GPU_identity':False},exclusive=True)
    return entry

def verify_input_root_bridge(parent,current,old_root,new_root):
    expected=deepcopy(parent)
    keys={'local_data_path','local_split_manifest_path'}
    require(set(parent['resolved_inputs'])==set(current['resolved_inputs'])==keys,'Input path fields changed')
    mapping={}
    for key in sorted(keys):
        old=Path(parent['resolved_inputs'][key]);old_base=Path(old_root);new_base=Path(new_root)
        require(old.is_absolute() and old_base.is_absolute() and new_base.is_absolute() and '..' not in old.parts,'Unsafe input root bridge')
        relative=old.relative_to(old_base)
        require(relative.parts and current['resolved_inputs'][key]==str(new_base/relative),'Unauthorized input path mapping')
        expected['resolved_inputs'][key]=str(new_base/relative)
        mapping[key]={'parent':str(old),'recovery':str(new_base/relative)}
    require(expected==current,'Resume input statistics/identity changed')
    return mapping

def bridge_resume_interface(current,parent,parent_sha,recovery_sha):
    require(parent.get('execution_contract_sha256')==parent_sha and current.get('execution_contract_sha256')==recovery_sha,'Interface contract bridge not bound')
    bridged=deepcopy(current)
    bridged['execution_contract_sha256']=parent_sha
    require(bridged==parent,'Resume scientific interface changed')
    return bridged

def migrate_resume_inputs(c,root,folder,current):
    parent_contract=read(root/'recovery_parent/parent_contract.json')
    require(sha_json(parent_contract)==c['recovery']['parent_contract_sha256'],'Parent input root binding changed')
    original=read(folder/'input_receipt.json')
    mapping=verify_input_root_bridge(original,current,parent_contract['hosts']['a100']['source_root'],c['hosts']['a100']['source_root'])
    common.write_json(folder/'input_root_bridge.json',{'parent_input_receipt_raw_sha256':sha_file(folder/'input_receipt.json'),
        'parent_contract_sha256':sha_json(parent_contract),'recovery_contract_sha256':sha_json(c),
        'qualified_recovery_input_receipt_sha256':sha_json(current),'allowed_path_mapping':mapping,
        'all_other_receipt_fields_equal':True,'parent_receipt_preserved':True},exclusive=True)

def migrate_resume_interface(training,c,job,run,folder,current,entry):
    if entry is None:return current
    checkpoint=run/'last_epoch_state.pt'
    require(sha_file(checkpoint)==entry['last_checkpoint_sha256'],'Resume binary bridge binding changed')
    payload=training.torch_load_checkpoint(checkpoint,map_location='cpu')
    parent=payload['interface_meta']
    require(payload['resume_identity']['interface_meta']==parent,'Parent interface identity differs')
    bridged=bridge_resume_interface(current,parent,c['recovery']['parent_contract_sha256'],sha_json(c))
    common.write_json(folder/'interface_contract_bridge.json',{'job':job,'parent_checkpoint_raw_sha256':entry['last_checkpoint_sha256'],
        'parent_contract_sha256':c['recovery']['parent_contract_sha256'],'recovery_contract_sha256':sha_json(c),
        'qualified_recovery_interface_sha256':sha_json(current),'continued_scientific_interface_sha256':sha_json(bridged),
        'allowed_field':'execution_contract_sha256','all_other_interface_fields_equal':True,
        'scope':'resumed fit and both endpoint identities only; qualification remains recovery-bound',
        'parent_checkpoint_preserved':True},exclusive=True)
    return bridged

def verify_resume_before_fit(training,c,job,run,args,quantity,interface,initial,entry):
    if entry is None:return
    payload=training.torch_load_checkpoint(run/'last_epoch_state.pt',map_location='cpu')
    identity=training._resume_identity(args=args,backbone=job['arm'],quantity_variant=variant_for(job['arm']),seed=job['seed'],
        monitor='validation_raw_quantity_rmse',quantity_contract=quantity,interface_meta=interface)
    history,best=training._validate_resume_payload(payload,expected_identity=identity,expected_initial_state_sha256=initial[job['arm']],
        monitor='validation_raw_quantity_rmse')
    exposure=read(run/'exposure.json');timing=read(run/'epoch_timing.json')['epochs']
    require(payload['epoch']==entry['epoch'] and payload['best_epoch']==entry['best_epoch'],'Resume epoch/best continuity mismatch')
    require(read(run/'history.json')['history']==history and len(timing)==len(exposure['train'])==len(exposure['validation'])==len(history),'Resume history/exposure/timing mismatch')
    require([row['epoch'] for row in timing]==list(range(1,entry['epoch']+1)),'Resume timing epochs not consecutive')
    steps=sum(row['batches'] for row in exposure['train'])
    require(all(float(value['step'])==steps for value in payload['optimizer_state_dict']['state'].values()),'Resume optimizer/exposure step mismatch')
    require(payload['source_revision']==c['source']['git_revision'] and payload['held_out_test_evaluated'] is False,'Resume source/split changed')


def batch_groups(c,host):
    batches=[]
    jobs=[j for j in c['jobs'] if j['host']==host]
    require(len(jobs)==27,'All original execution IDs must remain present')
    for i in range(0,len(jobs),3):
        triplet=jobs[i:i+3]
        require(len(triplet)==3 and len({(j['dataset'],j['seed']) for j in triplet})==1
            and [j['arm'] for j in triplet]==c['arms'],'Parallel group scope changed')
        pending=[j for j in triplet if j['id'] not in c['recovery']['reused_jobs']]
        if pending:batches.append(pending)
    require([len(b) for b in batches]==[2,3,3,3,3,3],'Recovery grouping changed')
    return batches

def check_active_gpu(c,host,root,active):
    import importlib.util
    name='_titantpp_recovery_operations'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,root/'operation/recovery_operations.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].check_active_gpu(sys.modules[__name__],c,host,root,active)
'''


def build_runtime(original, shared_text):
    text=original.replace('titantpp_contribution_history_a100_parallel3_20261008_v1',SCHEMA)
    # Budget remains USD50 across both physical Pods, including storage and cleanup reserve.
    scope=function(text,'validate_a100_scope')
    scope=scope.replace("duration/3600*budget['confirmed_gpu_hourly_usd']+budget['reserved_non_compute_usd']<=50",
        "duration/3600*budget['confirmed_running_hourly_usd']+budget['reserved_non_compute_usd']+budget['parent_billed_usd']+budget['prior_network_storage_usd']<=50")
    scope=scope.replace("require(c.get('deployment_ready') is True,'Draft cannot execute')",
        "require(type(budget['parent_billed_usd']) in (int,float) and math.isfinite(budget['parent_billed_usd']) and 30.124985997681506<=budget['parent_billed_usd']<50,'Unbound parent spend')\n    require(budget['confirmed_running_hourly_usd']>=budget['confirmed_gpu_hourly_usd']>0 and budget['recovery_reserve_usd']==5,'Unbound all-in rate/reserve')\n    require(math.isfinite(budget['prior_network_storage_usd']) and budget['prior_network_storage_usd']>=0.0009722222222222222,'Unbound prior network storage')\n    require(c.get('deployment_ready') is True,'Draft cannot execute')")
    text=replace_node(text,'validate_a100_scope',scope)
    terminal=function(text,'validate_terminal').replace("folder=root/'run'/job['id']",'folder=result_folder(c,root,job)').replace("manifest['contract_sha256']==sha_json(c)","manifest['contract_sha256']==result_contract(c,job)")
    text=replace_node(text,'validate_terminal',terminal)
    audit=function(text,'audit_host_batch_prefix').replace("folder=root/'run'/previous['id']",'folder=result_folder(c,root,previous)').replace("manifest['contract_sha256']==sha_json(c)","manifest['contract_sha256']==result_contract(c,previous)")
    text=replace_node(text,'audit_host_batch_prefix',audit)
    batchaudit=function(text,'audit_completed_batch').replace("path=root/'run'/job['id']/'runs'", "path=result_folder(c,root,job)/'runs'")
    text=replace_node(text,'audit_completed_batch',batchaudit)
    dispatch=function(text,'dispatch').replace("results={};active={};batch=None", "validate_recovery(c,root)\n        results={j['id']:validate_terminal(c,root,j) for j in c['jobs'] if j['id'] in c['recovery']['reused_jobs']};active={};batch=None")
    dispatch=dispatch.replace('audit_completed_batch(c,host,root,batch)',"audit_completed_batch(c,host,root,[j for j in c['jobs'] if (j['dataset'],j['seed'])==(batch[0]['dataset'],batch[0]['seed'])])")
    text=replace_node(text,'dispatch',dispatch)
    fit=function(text,'run_fit')
    fit=fit.replace("folder=root/'run'/job['id'];folder.mkdir(parents=True,exist_ok=False)","folder=root/'run'/job['id'];resume_entry=prepare_resume(c,root,job,folder)")
    fit=fit.replace("common.write_json(folder/'input_receipt.json',receipt,exclusive=True)\n    common.write_json(folder/'initialization.json',initial,exclusive=True)",
        "if resume_entry is None:\n        common.write_json(folder/'input_receipt.json',receipt,exclusive=True)\n        common.write_json(folder/'initialization.json',initial,exclusive=True)\n    else:\n        migrate_resume_inputs(c,root,folder,receipt)\n        require(read(folder/'initialization.json')==initial,'Resume initial state changed')")
    fit=fit.replace('with strict_training_seed(training), shared.audited_training(training,data,budget,status) as exposure:',
        "interface=migrate_resume_interface(training,c,job,run,folder,interface,resume_entry)\n    verify_resume_before_fit(training,c,job,run,args,quantity,interface,initial,resume_entry)\n    with strict_training_seed(training), recovery_audited_training(training,data,budget,status,run,resume_entry) as exposure:")
    fit=fit.replace("common.write_json(run/'endpoint_replays.json',result,exclusive=True)",
        "result['recovery_provenance']={'mode':'resume' if resume_entry else 'fresh','parent_epoch':resume_entry['epoch'] if resume_entry else None,'physical_gpu_migration':bool(resume_entry),'elapsed_seconds_scope':'current_physical_GPU_segment_only'}\n    common.write_json(run/'endpoint_replays.json',result,exclusive=True)")
    text=replace_node(text,'run_fit',fit)
    # Preserve audited scientific loader/save checks, adding only verified prefix records.
    audited=function(shared_text,'audited_training')
    audited=audited.replace('def audited_training(training, data, budget_check, status_callback):',
        'def recovery_audited_training(training, data, budget_check, status_callback, run, resume_entry):')
    audited=audited.replace('records = {"train": [], "validation": []}',
        'records = read(run/"exposure.json") if resume_entry else {"train": [], "validation": []}')
    audited=audited.replace('epoch_clock, epoch_timings = {}, []',
        'epoch_clock, epoch_timings = {}, read(run/"epoch_timing.json")["epochs"] if resume_entry else []\n    ExposureLoader=shared.ExposureLoader\n    write_status=common.write_json')
    extra=RECOVERY_FUNCTIONS+'\n@contextmanager\n'+audited+'\n'
    # Replace existing functions with the recovery variants; insert only genuinely new helpers.
    nodes=[n for n in ast.parse(RECOVERY_FUNCTIONS).body if isinstance(n,ast.FunctionDef)]
    existing={n.name for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    additions=[]
    for node in nodes:
        body=ast.get_source_segment(RECOVERY_FUNCTIONS,node)
        if node.name in existing:text=replace_node(text,node.name,body)
        else:additions.append(body)
    text=text.replace('if __name__ == "__main__":main()', '\n\n'.join(additions)+'\n\n@contextmanager\n'+audited+'\n\nif __name__ == "__main__":main()')
    require(text!=original and 'recovery_audited_training' in text,'Recovery transform failed')
    ast.parse(text)
    # Source/optimizer/model/qualification/selector functions stay byte-identical.
    changed={'validate','validate_a100_scope','validate_terminal','audit_host_batch_prefix','audit_completed_batch','dispatch','run_fit','batch_groups','check_active_gpu'}
    for node in ast.parse(original).body:
        if isinstance(node,ast.FunctionDef) and node.name not in changed:
            require(function(text,node.name)==ast.get_source_segment(original,node),'Unexpected function changed '+node.name)
    return text


def prepare(args):
    old=read(PARENT/'execution_contract.json');require(canonical(old)==PARENT_SHA,'Parent contract changed')
    require(old['source']['files_sha256']==SOURCE_SHA and canonical(old['source']['files'])==SOURCE_SHA,'Source closure changed')
    require(canonical(old['operation']['files'])==OP_SHA,'Parent operation closure changed')
    for name,digest in old['operation']['files'].items():require(sha(PARENT/name)==digest,'Parent operation changed')
    partial=PARENT/'originals/partial';verified=partial/'verified';originals=verified/'originals'
    require(sha(verified/'manifest.json')==MANIFEST_SHA and sha(partial/'originals.tar.gz')==ARCHIVE_SHA,'Preserved originals archive changed')
    manifest=read(verified/'manifest.json')
    for name,record in manifest['files'].items():require(sha(originals/name)==record['sha256'],'Parent original changed '+name)
    require(read(PARENT/'control/cleanup_receipt.json')['deleted_confirmed'] is True,'Parent physical Pod not confirmed deleted')
    require(args.pod_id and args.pod_id!='qq2uk2dpfi2rum','New physical Pod binding required')
    require(args.gpu_uuid.startswith('GPU-') and len(args.gpu_uuid)==40 and args.gpu_uuid!=old['hosts']['a100']['gpu_uuid'],'New physical GPU UUID required')
    require(all(math.isfinite(v) for v in [args.billing_started,args.gpu_hourly,args.total_hourly,args.parent_billed,args.prior_network_storage_usd]),'Nonfinite monetary binding')
    require(args.total_hourly>=args.gpu_hourly>0 and PARENT_MIN_BILLED<=args.parent_billed<45,'Invalid rate/remaining budget')
    require(args.prior_network_storage_usd>=0.0009722222222222222,'Pre-Pod network storage upper bound required')
    remaining=50-args.parent_billed-args.prior_network_storage_usd;work_seconds=math.floor((remaining-5)/args.total_hourly*3600)
    deadline=args.billing_started+work_seconds
    require(args.billing_started<=time.time()<deadline,'New billing window not live')
    destination=ROOT/'search_artifacts'/SCHEMA
    require(not destination.exists(),'Recovery bundle already exists; no implicit overwrite/retry')
    c=deepcopy(old);c['schema']=SCHEMA;spec=c['hosts']['a100']
    for key in ('root','source_root','operation_root','output_dir','tmux'):spec[key]=spec[key].replace(old['schema'],SCHEMA)
    spec['environment']={key:value.replace(old['schema'],SCHEMA) for key,value in spec['environment'].items()}
    spec['gpu_uuid']=args.gpu_uuid
    c['budget']={'maximum_total_usd':50,'independent_controller_required':True,'confirmed_gpu_hourly_usd':args.gpu_hourly,
        'confirmed_running_hourly_usd':args.total_hourly,'parent_billed_usd':args.parent_billed,'prior_network_storage_usd':args.prior_network_storage_usd,'billing_started_at_unix':args.billing_started,
        'hard_deadline_unix':deadline,'absolute_stop_deadline_unix':args.billing_started+math.floor(remaining/args.total_hourly*3600),
        'reserved_non_compute_usd':5,'recovery_reserve_usd':5}
    c['limits']['total_wall_seconds']=work_seconds;c['limits']['per_condition_seconds']=min(36*3600,work_seconds)
    reused=[];resumed={};fresh=[]
    for job in c['jobs']:
        folder=originals/'run'/job['id'];terminal=folder/'terminal_manifest.json'
        if terminal.exists():
            term=read(terminal);require(term['scientific_success'] and term['contract_sha256']==PARENT_SHA,'Parent terminal mismatch')
            reused.append(job['id'])
        elif folder.exists():
            run=folder/'runs'/job['arm']/'count_only_log_regression'/f"seed_{job['seed']}";receipt=read(run/'server_checkpoint_receipt.json')
            require(job['dataset']=='raf_spare_parts' and job['seed']==42 and receipt['epoch'] in (23,46),'Unexpected partial identity')
            hist=read(run/'history.json')['history'];best=min(hist,key=lambda row:row['val_qty_rmse'])
            resumed[job['id']]={'epoch':receipt['epoch'],'best_epoch':best['epoch'],'receipt_sha256':sha(run/'server_checkpoint_receipt.json'),
                'last_checkpoint_sha256':receipt['files']['last_epoch_state.pt']}
        else:fresh.append(job['id'])
    require((len(reused),len(resumed),len(fresh))==(10,2,15),'Recovery execution counts mismatch')
    parent_files={name:record['sha256'] for name,record in manifest['files'].items() if name.startswith('run/')}
    c['recovery']={'parent_contract_sha256':PARENT_SHA,'parent_archive_sha256':ARCHIVE_SHA,'parent_manifest_sha256':MANIFEST_SHA,
        'source_gpu_uuid':old['hosts']['a100']['gpu_uuid'],'destination_pod_id':args.pod_id,'physical_gpu_migration':True,'automatic_retry':False,
        'reused_jobs':reused,'resumed_jobs':resumed,'fresh_jobs':fresh,'parent_files':parent_files,
        'limitations':['New physical A100 GPU; no same-UUID or uninterrupted arithmetic equivalence claim','Timing retained per physical GPU segment'],
        'user_instruction':'문제 확인해서 수정하고 남은 학습 진행하자'}
    c['operational_replacement']={'parent_contract_sha256':PARENT_SHA,'reason':'Explicitly authorized recovery after ownership inspection failure',
        'completed_reused':10,'resumed':2,'fresh':15,'same_physical_GPU':False,'automatic_retry':False}
    for name,digest in {**old['source']['files'],**old['input_files']}.items():
        src=PARENT/'source'/name;require(sha(src)==digest,'Frozen source/input changed '+name)
        dst=destination/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        require(sha(dst)==digest,'Frozen copy mismatch')
    for name,digest in parent_files.items():
        dst=destination/'recovery_parent'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(originals/name,dst)
        require(sha(dst)==digest,'Parent recovery copy mismatch')
    shutil.copyfile(verified/'manifest.json',destination/'recovery_parent/parent_manifest.json')
    shutil.copyfile(PARENT/'execution_contract.json',destination/'recovery_parent/parent_contract.json')
    shutil.copyfile(partial/'originals.tar.gz',destination/'recovery_parent/parent_archive.tar.gz')
    for name,digest in [('parent_manifest.json',MANIFEST_SHA),('parent_contract.json',sha(PARENT/'execution_contract.json')),('parent_archive.tar.gz',ARCHIVE_SHA)]:
        require(sha(destination/'recovery_parent'/name)==digest,'Copied preservation binding mismatch')
        c['recovery']['parent_files'][name]=digest
    shared_text=(PARENT/'source/paper/scripts/run_observed_slot_partition.py').read_text()
    operation=build_runtime((PARENT/'operation/campaign.py').read_text(),shared_text)
    write(destination/'operation/campaign.py',operation)
    shutil.copyfile(ROOT/'paper/scripts/titantpp_history_recovery_operations.py',destination/'operation/recovery_operations.py')
    c['operation']={'files':{name:sha(destination/name) for name in ['operation/campaign.py','operation/recovery_operations.py']}}
    c['operation']['files_sha256']=canonical(c['operation']['files']);c['deployment_ready']=True
    write_json(destination/'execution_contract.json',c)
    approval={'approved':True,'hosts':['a100'],'contract_sha256':canonical(c),'user_instruction':c['recovery']['user_instruction'],
        'maximum_combined_usd':50,'new_physical_GPU_migration':True,'held_out_test_authorized':False}
    write_json(destination/'approval.json',approval)
    write_json(destination/'current.json',{'bundle':str(destination),'contract_sha256':canonical(c),'status':'prepared_not_launched','remote_calls':0})
    print(json.dumps({'bundle':str(destination),'contract_sha256':canonical(c),'operation_sha256':c['operation']['files_sha256'],
        'deadline_unix':deadline,'absolute_stop_deadline_unix':c['budget']['absolute_stop_deadline_unix'],'remaining_usd':remaining,'counts':[10,2,15],'remote_calls':0}))
    return c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pod-id',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--billing-started',type=float,required=True);p.add_argument('--gpu-hourly',type=float,required=True)
    p.add_argument('--total-hourly',type=float,required=True);p.add_argument('--parent-billed',type=float,required=True)
    p.add_argument('--prior-network-storage-usd',type=float,required=True)
    prepare(p.parse_args())

if __name__=='__main__':main()
