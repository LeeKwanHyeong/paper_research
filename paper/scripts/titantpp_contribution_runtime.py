"""Sealed, one-attempt staged Validation campaigns. No work at import time.

The frozen trainer/loader/endpoint path is retained. Only owned model routes
and operational scope are additive; no shared runtime or historical data edits.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

VARIANT = 'count_only_log_regression'
TRAINING = {'batch_size':128,'maximum_epochs':300,'minimum_epochs':40,
    'patience':40,'monitor':'validation_raw_quantity_rmse',
    'tie':'strict_earliest_finite_minimum','warm_start':False}
STAGES = {
 'instacart': ('titantpp_history_mlp_width8','titantpp_history_mlp_width12','titantpp_cnn_gru54'),
 'history': ('titantpp_cnn_gru54','titantpp_cnn_gru54_correction_disabled','titantpp_cnn_gru54_current_only'),
 'retrieval': ('titantpp_retrieval_weighted','titantpp_retrieval_fixed','titantpp_retrieval_generic512')}
SEEDS = (42,52,62)
PARENT_CLOSURE = '918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a'
ROOT = None
common = prior = base = shared = engine = None
engines = {}

def require(condition,message):
    if not condition: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text())

def sha_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''): h.update(b)
    return h.hexdigest()

def safe_relative(root,name):
    p=Path(name); require(not p.is_absolute() and '..' not in p.parts,'Unsafe path')
    target=(Path(root)/p).resolve()
    require(target.is_relative_to(Path(root).resolve()),'Path escape')
    return target

def expected_jobs(c):
    return [{'id':f'{d}__{s}__{arm}','host':h,'dataset':d,'seed':s,'arm':arm}
        for h,groups in c['assignments'].items() for d,s in groups for arm in STAGES[c['stage']]]

def validate(c,*,source_root=None,operation_root=None,verify_files=True):
    require(c['stage'] in STAGES and c['schema']=='titantpp_contribution_'+c['stage']+'_20261008_v1','Foreign stage')
    require(c['arms']==list(STAGES[c['stage']]) and c['training']==TRAINING,'Scientific scope drift')
    require(c['jobs']==expected_jobs(c) and len({j['id'] for j in c['jobs']})==len(c['jobs']),'Job ownership drift')
    require(len(c['jobs'])==(9 if c['stage']=='instacart' else 27),'Fit count drift')
    groups=[(d,s) for gs in c['assignments'].values() for d,s in gs]
    require(len(groups)==len(set(groups)) and all(s in SEEDS for d,s in groups),'Duplicate seed owner')
    expected_data={'insta_market_basket'} if c['stage']=='instacart' else {'yellow_trip_hourly','intermittent_frozen_5000','raf_spare_parts'}
    require({d for d,s in groups}==expected_data and set(groups)=={(d,s) for d in expected_data for s in SEEDS},'Dataset/seed scope drift')
    require(c['evaluation_scope']=='validation_only' and c['held_out_test_evaluated'] is False,'Held-out boundary drift')
    require(c['limits']['total_wall_seconds']==168*3600 and c['limits']['per_condition_seconds']==36*3600
        and c['limits']['workers_per_host']==1 and not c['limits']['automatic_retry'],'Budget drift')
    parent=c['source']['parent_files']
    require(len(parent)==126 and sha_json(parent)==PARENT_CLOSURE,'Frozen parent126 drift')
    require(all(c['source']['files'].get(p)==v for p,v in parent.items()),'Frozen scientific source overwritten')
    require(sha_json(c['source']['files'])==c['source']['files_sha256'],'Source closure drift')
    require(sha_json(c['operation']['files'])==c['operation']['files_sha256'],'Operation closure drift')
    require({d['dataset_id'] for d in c['datasets']}==expected_data,'Dataset records drift')
    for data in c['datasets']:
        require(sha_json(data)==c['dataset_sha256'][data['dataset_id']],'Data contract drift')
    for h,spec in c['hosts'].items():
        root=Path(spec['root']);require(root.is_absolute() and root.name==c['schema']+'_'+h,'Foreign owned root')
        require(spec['source_root']==str(root/'source') and spec['operation_root']==str(root/'operation'),'Pinned root drift')
    if verify_files:
        sr=Path(source_root or ROOT);op=Path(operation_root or Path(__file__).resolve().parent)
        require(os.environ['SOURCE_REVISION']==c['source']['git_revision'],'Source revision environment drift')
        for p,v in {**c['source']['files'],**c['input_files']}.items():require(sha_file(safe_relative(sr,p))==v,'Source/input drift '+p)
        for p,v in c['operation']['files'].items():require(sha_file(safe_relative(op.parent,p))==v,'Operation drift '+p)
    return c

def bootstrap(c,host):
    global ROOT,common,prior,base,shared
    ROOT=Path(c['hosts'][host]['source_root']).resolve()
    validate(c,source_root=ROOT,operation_root=c['hosts'][host]['operation_root'])
    require(Path(__file__).resolve().parent==Path(c['hosts'][host]['operation_root']).resolve(),'Wrong pinned operation')
    sys.path.insert(0,str(ROOT))
    common=importlib.import_module('paper.scripts.observed_slot_parallel_common')
    prior=importlib.import_module('paper.scripts.run_local_detail_benchmark')
    base=importlib.import_module('paper.scripts.run_multilag_detail_execution')
    shared=importlib.import_module('paper.scripts.run_observed_slot_partition')
    for arm,module in c['runner_modules'].items():
        engines[arm]=importlib.import_module(module);engines[arm].install_hooks()
    for module in (common,prior,base,shared,*engines.values()):
        require(Path(module.__file__).resolve().is_relative_to(ROOT),'Import escaped frozen source')

def authorization(c,root,*,live=True):
    validate(c)
    require(not (root/'failure.json').exists() and not (root/'qualification/failure.json').exists(),'Preserved failure; no automatic retry')
    a,s=read(root/'approval.json'),read(root/'start_permit.json')
    h=next(h for h,v in c['hosts'].items() if Path(v['root'])==root)
    require(a['approved'] is True and a['contract_sha256']==sha_json(c) and h in a['hosts'] and a['user_instruction'],'Missing approval')
    require(s['host']==h and s['contract_sha256']==sha_json(c) and s['approval_sha256']==sha_json(a),'Foreign permit')
    require(s['deadline_unix']-s['started_at_unix']==c['limits']['total_wall_seconds'],'Extended lease')
    if live: require(s['started_at_unix']<=time.time()<s['deadline_unix'],'Expired lease')
    return s

def build_model(data,arm):return engines[arm].build_model(data,arm)
def training_args(c,data,output,arm):return engines[arm].training_args(c,data,output,arm)
def time_interface(data,frame,c,arm):
    m=engines[arm]
    return m.time_interface(data,frame,c,arm=arm)
def replay_checkpoint(path,data,frame,budget,**kwargs):
    return engines[kwargs['expected_arm']].replay_checkpoint(path,data,frame,budget,**kwargs)
def variant_for(arm):return VARIANT

def initial_states(data,seed):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    result={};states={};rngs={}
    with torch.random.fork_rng(devices=[]):
        for arm in engines:
            torch.manual_seed(seed);model,_=build_model(data,arm)
            states[arm]={k:v.clone() for k,v in model.state_dict().items()};rngs[arm]=torch.get_rng_state().clone()
            result[arm]=canonical_state_dict_sha256(states[arm]);del model
        first=next(iter(states))
        for arm in states:
            require(torch.equal(rngs[first],rngs[arm]),'Common constructor RNG changed')
            for k,v in states[first].items():
                if k.startswith('multilag_detail.') or k.endswith('identity') or (k.startswith('lmm.') and k!='lmm.mem'):continue
                require(k in states[arm] and torch.equal(v,states[arm][k]),'Common initial tensor changed '+k)
    return result

def verify_fit_runtime(c,host,q,job,actual):
    expected=deepcopy(q['runtime']);expected['environment']['PYTHONHASHSEED']=str(job['seed'])
    require(q['status']=='passed' and expected==actual,'Qualified Runtime changed')

def verify_training_permit(c,root):
    p=read(root/'training_permit.json');s=read(root/'start_permit.json');q=read(root/'qualification/receipt.json')
    require(p['contract_sha256']==sha_json(c) and p['start_permit_sha256']==sha_json(s)
        and p['qualification_sha256']==sha_file(root/'qualification/receipt.json') and p['qualification']==q,'Training binding drift')
    require(q['status']=='passed' and q['contract_sha256']==sha_json(c),'Native qualification missing')
    return p

def verify_full_validation(endpoint,data):
    require(endpoint['evaluation_scope']=='validation_only' and endpoint['held_out_test_evaluated'] is False,'Endpoint split drift')
    require(endpoint['count']==data['inherited_data_identity']['populations']['validation']['target_count'],'Not full Validation')
    require(endpoint['quantity_boundaries']==data['quantity_boundaries_all_train_rows'],'Tail boundary drift')
    for cell in (endpoint,endpoint['tail']):
        require(cell['count']>0 and all(math.isfinite(cell[k]) for k in ('qty_rmse','qty_mae','time_nll')),'Nonfinite endpoint')
    return True

def audit_host_batch_prefix(c,host,root,job,exposure):
    # Compare only the same dataset and seed; different seeds deliberately shuffle differently.
    for previous in c['jobs']:
        if previous==job or (previous['host'],previous['dataset'],previous['seed'])!=(host,job['dataset'],job['seed']):continue
        folder=root/'run'/previous['id'];mp=folder/'terminal_manifest.json'
        if not mp.exists():continue
        manifest=read(mp);name=f"runs/{previous['arm']}/{VARIANT}/seed_{job['seed']}/exposure.json"
        require(manifest['job']==previous and manifest['contract_sha256']==sha_json(c) and sha_file(folder/name)==manifest['files'][name],'Previous exposure drift')
        ref=read(folder/name);n=min(len(ref['train']),len(exposure['train']))
        require(ref['train'][:n]==exposure['train'][:n],'Paired train order drift')
        require(len({r['batch_order_sha256'] for rows in (ref['validation'],exposure['validation']) for r in rows})==1,'Validation order drift')

def qualify(c,host,root):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    authorization(c,root);require(not common.gpu_pids(c['hosts'][host]),'GPU occupied')
    common.write_json(root/'qualification.claim',{'pid':os.getpid(),'contract_sha256':sha_json(c)},exclusive=True)
    runtime=common.runtime_check(c,host)
    deadline=min(read(root/'start_permit.json')['deadline_unix'],time.time()+c['limits']['qualification_seconds'])
    budget=PulseBudget(root,sha_json(c),'qualification',deadline)
    result={'status':'checking','host':host,'runtime':runtime,'contract_sha256':sha_json(c),
        'inputs':{},'initialization':{},'measurements':{},'causality_checks':{},'synthetic_optimizer_updates':0,'scientific_optimizer_updates':0,
        'held_out_test_evaluated':False,'real_Validation_predictions':False}
    for module in set(engines.values()):
        if hasattr(module,'check_synthetic_causality'):
            budget();result['causality_checks'][module.__name__]=module.check_synthetic_causality('cuda:0')
    for data in c['datasets']:
        d=data['dataset_id']
        if not any(j['host']==host and j['dataset']==d for j in c['jobs']):continue
        budget();frame,receipt=base.prepare_admitted_data(data,data_root=ROOT)
        result['inputs'][d]={'input_receipt_sha256':sha_json(receipt),'input_receipt':receipt,
            'interface_sha256':{a:sha_json(time_interface(data,frame,c,a)) for a in c['arms']}}
        result['initialization'][d]={str(s):initial_states(data,s) for s in SEEDS};del frame;gc.collect()
        measure={}
        for length in sorted({min(64,data['loader']['max_seq_len']),data['loader']['max_seq_len']}):
            samples={}
            dt=torch.ones((128,length),device='cuda:0');qty=torch.arange(length,device='cuda:0').remainder(17).add(1).float().expand(128,-1)
            mask=torch.ones_like(dt,dtype=torch.bool);mask[:32,:max(1,length//3)]=False
            for arm in c.get('measurement_arms',c['arms']):
                budget();torch.manual_seed(42);model,_=build_model(data,arm);model=model.cuda()
                before={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
                model.eval();torch.cuda.reset_peak_memory_stats();infer=[]
                with torch.no_grad():
                    for i in range(25):
                        budget();torch.cuda.synchronize();t=time.perf_counter();out=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
                        require(all(torch.isfinite(v).all().item() for v in out.values() if hasattr(v,'is_floating_point') and v.is_floating_point()),'Nonfinite native output')
                        torch.cuda.synchronize()
                        if i>=5:infer.append(time.perf_counter()-t)
                inference_peak=torch.cuda.max_memory_allocated();model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);steps=[]
                torch.cuda.reset_peak_memory_stats();active=set()
                for i in range(25):
                    budget();torch.cuda.synchronize();t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    out=target_outputs(model,dt,mask,qty,lambda_log_qty=1.);loss=out['joint_loss'].mean()
                    require(torch.isfinite(loss).item(),'Nonfinite loss');loss.backward()
                    for k,p in model.named_parameters():
                        if p.grad is not None:
                            require(torch.isfinite(p.grad).all().item(),'Nonfinite gradient')
                            if p.grad.ne(0).any().item():active.add(k)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);require(torch.isfinite(norm).item(),'Nonfinite clip');optimizer.step()
                    torch.cuda.synchronize();result['synthetic_optimizer_updates']+=1
                    if i>=5:steps.append(time.perf_counter()-t)
                require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite parameter')
                changed=[k for k,p in model.named_parameters() if k in before and not torch.equal(p.detach(),before[k])]
                require(changed,'No actual native updates')
                samples[arm]={'inference_median_seconds':statistics.median(infer),'training_step_median_seconds':statistics.median(steps),
                    'inference_peak_bytes':inference_peak,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'parameters_total':sum(p.numel() for p in model.parameters()),'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'active_gradient_parameter_names':sorted(active),'changed_parameter_names':changed,'warmup':5,'measure':20}
                del model,optimizer,before,out;gc.collect();torch.cuda.empty_cache()
            measure[str(length)]=samples
        result['measurements'][d]=measure
    result.update(status='passed',completed_unix=time.time(),cost_gate_is_training_launch_gate=False)
    common.write_json(root/'qualification/receipt.json',result,exclusive=True)
    print(json.dumps({'status':'passed','host':host,'synthetic_updates':result['synthetic_optimizer_updates']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--host',choices=('5080','5090'),required=True)
    p.add_argument('--mode',choices=('qualify','dispatch','fit'),required=True);p.add_argument('--job');p.add_argument('--deadline',type=float)
    a=p.parse_args();c=read(a.contract);root=Path(c['hosts'][a.host]['root'])
    require(Path(a.contract).resolve()==root/'execution_contract.json','Wrong contract path')
    require(Path.cwd().resolve()==Path(c['hosts'][a.host]['source_root']),'Wrong frozen cwd')
    expected={**c['hosts'][a.host]['environment']}
    if a.mode=='fit':expected['PYTHONHASHSEED']=str(next(j for j in c['jobs'] if j['id']==a.job)['seed'])
    require(all(os.environ.get(k)==v for k,v in expected.items()),'Runtime environment drift')
    bootstrap(c,a.host)
    if a.mode=='fit':common.ENVIRONMENT={**common.ENVIRONMENT,'PYTHONHASHSEED':expected['PYTHONHASHSEED']}
    try:
        if a.mode=='qualify':qualify(c,a.host,root)
        elif a.mode=='dispatch':dispatch(c,a.host,root)
        else:run_fit(c,a.host,root,next(j for j in c['jobs'] if j['id']==a.job),a.deadline)
    except BaseException as exc:
        if a.mode=='qualify':common.write_json(root/'qualification/failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},exclusive=True)
        raise


@contextmanager
def strict_training_seed(training):
    """Preserve frozen seed/RNG behavior and retain the qualified strict flag."""
    import torch
    original = training.set_seed
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    def strict_seed(*args, **kwargs):
        result = original(*args, **kwargs)
        torch.use_deterministic_algorithms(True, warn_only=False)
        return result
    training.set_seed = strict_seed
    try:
        yield
    finally:
        training.set_seed = original
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

class PulseBudget:
    def __init__(self,root,digest,job,deadline,lease=False):
        self.root,self.digest,self.job,self.deadline,self.lease=root,digest,job,deadline,lease
        self.last=0.;self.last_storage_check=0.;self.monotonic_deadline=time.monotonic()+max(0,deadline-time.time())
    def __call__(self):
        now=time.time()
        require(now<self.deadline and time.monotonic()<self.monotonic_deadline,'Absolute time budget exhausted')
        if now-self.last<2:return
        if self.lease:
            r=read(self.root/'server_lease.json')
            require(r['contract_sha256']==self.digest and r['job']==self.job and now<r['expires_unix']<=self.deadline,
                'Server-owned supervisor lease expired')
        require(shutil.disk_usage(self.root).free>5*1024**3,'Server free disk below5GiB')
        common.write_json(self.root/'progress.json',{'job':self.job,'pid':os.getpid(),'updated_unix':now,
            'deadline_unix':self.deadline,'source':'genuine_phase_or_batch_progress','mac_required':False})
        rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
        require(rss<16*1024**3,'Owned process CPU RSS exceeds16GiB')
        if now-self.last_storage_check>30:
            require(sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())<8*1024**3,'Owned-root bytes exceed8GiB')
            self.last_storage_check=now
        import torch
        if torch.cuda.is_initialized():
            require(torch.cuda.max_memory_allocated() <= .8*torch.cuda.get_device_properties(0).total_memory, 'Owned GPU allocation exceeds80 percent')
        self.last=now

def run_fit(c,host,root,job,deadline):
    from paper.scripts.count_aware_tpp_backbone import training
    start=authorization(c,root);q=read(root/'qualification/receipt.json');verify_training_permit(c,root)
    verify_fit_runtime(c,host,q,job,common.runtime_check(c,host))
    require(job in c['jobs'] and job['host']==host and job['arm'] in c['arms'],'Unapproved fit')
    claim=read(root/'claims'/f"{job['id']}.json")
    require(claim['job']==job and claim['contract_sha256']==common.sha_json(c) and claim['deadline_unix']==deadline and deadline<=start['deadline_unix'],'Foreign fit claim/deadline')
    budget=PulseBudget(root,common.sha_json(c),job['id'],min(deadline,start['deadline_unix']),lease=True)
    lease=read(root/'server_lease.json')
    require(os.getpgrp()==lease['worker_group_pid'] and os.getppid()!=lease['supervisor_pid'],'Fit requires native timeout process group')
    parent=subprocess.run(['ps','-p',str(os.getppid()),'-o','args='],capture_output=True,text=True,check=True).stdout
    require(parent.strip().startswith('timeout --signal=TERM --kill-after=15s ') and str(deadline) in parent,'Native timeout parent required')
    budget();data=next(d for d in c['datasets'] if d['dataset_id']==job['dataset'])
    arm,seed=job['arm'],job['seed']
    frame,receipt=base.prepare_admitted_data(data,data_root=ROOT);interface=time_interface(data,frame,c,arm)
    require(common.sha_json(receipt)==q['inputs'][data['dataset_id']]['input_receipt_sha256'],'Qualified Train/Validation identity changed')
    require(common.sha_json(interface)==q['inputs'][data['dataset_id']]['interface_sha256'][arm],'Qualified interface changed')
    initial=initial_states(data,seed)
    require(initial==q['initialization'][data['dataset_id']][str(seed)],'Native initialization changed')
    folder=root/'run'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    args=training_args(c,data,folder,arm)
    args.epochs=300;args.min_epochs=40;args.early_stopping_patience=40;args.seeds=str(seed)
    args.execution_role='fresh_contribution_validation'
    common.write_json(folder/'input_receipt.json',receipt,exclusive=True)
    common.write_json(folder/'initialization.json',initial,exclusive=True)
    run=folder/'runs'/arm/variant_for(arm)/f'seed_{seed}'
    quantity={'boundaries':data['quantity_boundaries_all_train_rows'],'strata':[{'label':f'frozen_quantity_bin_{i}'} for i in range(5)]}
    def status(epoch,records):
        budget();common.write_json(run/'exposure.json',records)
        names=('last_epoch_state.pt','history.json','exposure.json','epoch_timing.json','train.log')
        common.write_json(run/'server_checkpoint_receipt.json',{'epoch':epoch,'contract_sha256':common.sha_json(c),
            'files':{n:common.sha_file(run/n) for n in names},'mac_ack_required':False,'time':time.time()})
        common.write_json(folder/'status.json',{'status':'training','epoch':epoch,'global_steps':sum(x['batches'] for x in records['train']),
            'job':job,'updated_unix':time.time(),'deadline_unix':deadline})
        if epoch==2:
            times=[x['elapsed_seconds'] for x in read(run/'epoch_timing.json')['epochs']]
            remaining=max(times)*298*1.2+600
            common.write_json(folder/'two_epoch_cost_receipt.json',{'status':'warning_only','termination_gate':False,'same_scientific_fit':True,
                'measured_epoch_seconds':times,'projected_remaining_seconds':remaining,'not_a_completion_guarantee':True})
    with strict_training_seed(training), shared.audited_training(training,data,budget,status) as exposure:
        summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,interface_meta=interface,
            backbone=arm,quantity_variant=variant_for(arm),seed=seed)
    history=read(run/'history.json')['history'];steps=prior.audit_arm(history,summary,exposure,data,c['training'])
    require(summary['initial_state_sha256']==initial[arm],'Initial tensor identity mismatch')
    identity=training._resume_identity(args=args,backbone=arm,quantity_variant=variant_for(arm),seed=seed,
        monitor='validation_raw_quantity_rmse',interface_meta=interface,quantity_contract=quantity)
    endpoints={}
    for label,path in (('selected',Path(summary['checkpoint_path'])),('last',run/'last_epoch_state.pt')):
        epoch=summary['best_epoch'] if label=='selected' else len(history)
        endpoints[label]=replay_checkpoint(path,data,frame,budget,expected_arm=arm,expected_identity=identity,
            expected_initial=initial[arm],expected_epoch=epoch,expected_seed=seed)
        verify_full_validation(endpoints[label],data)
        engines[arm].audit_replay_accounting(endpoints[label])
        require(all(math.isclose(endpoints[label][k],history[epoch-1][v],rel_tol=1e-10,abs_tol=1e-8)
            for k,v in (('qty_mae','val_qty_mae'),('qty_rmse','val_qty_rmse'),('time_nll','val_time_nll'))),'Endpoint replay mismatch')
    audit_host_batch_prefix(c,host,root,job,exposure)
    result={**endpoints,'status':'complete','job':job,'best_epoch':summary['best_epoch'],'completed_epochs':len(history),
        'global_steps':steps,'initial_state_sha256':initial[arm],'parameter_count':summary['parameter_count'],
        'elapsed_seconds':summary['elapsed_seconds'],'last30':shared.last30_summary(history),
        'first40':{'mean':statistics.mean(x['val_qty_rmse'] for x in history[:40]),
            'sd':statistics.stdev(x['val_qty_rmse'] for x in history[:40]),'count':40},
        'evaluation_scope':'validation_only','held_out_test_evaluated':False}
    common.write_json(run/'endpoint_replays.json',result,exclusive=True);common.write_json(run/'exposure.json',exposure)
    common.write_json(folder/'status.json',{'status':'complete','job':job,'updated_unix':time.time()})
    files={str(p.relative_to(folder)):common.sha_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    common.write_json(folder/'terminal_manifest.json',{'status':'complete','scientific_success':True,'files':files,
        'contract_sha256':common.sha_json(c),'job':job,'held_out_test_evaluated':False,'evaluation_scope':'validation_only','completed_unix':time.time()},exclusive=True)

def stop_owned(child):
    if child.poll() is not None:return
    require(os.getpgid(child.pid)==child.pid,'Owned child process-group identity changed')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)

def dispatch(c,host,root):
    start=authorization(c,root);permit=verify_training_permit(c,root)
    require(read(root/'qualification/receipt.json')==permit['qualification'],'Local qualification changed')
    require(not common.gpu_pids(c['hosts'][host]),'GPU occupied before launch')
    require(shutil.which('timeout') is not None,'Independent native timeout guard unavailable')
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        common.write_json(root/'supervisor.claim',{'pid':os.getpid(),'started_unix':time.time(),'contract_sha256':common.sha_json(c)},exclusive=True)
        results={};child=None;job=None
        try:
            # Each approved dataset/seed group has one owner and fresh fits only.
            for job in (j for j in c['jobs'] if j['host']==host):
                validate(c);require(time.time()<start['deadline_unix'],'Campaign budget expired')
                require(not common.gpu_pids(c['hosts'][host]),'Other GPU owner detected')
                require(not (root/'run'/job['id']).exists(),'Run exists; no implicit restart')
                deadline=min(start['deadline_unix'],time.time()+c['limits']['per_condition_seconds'],
                    c.get('retained_condition_deadlines',{}).get(job['id'],float('inf')))
                common.write_json(root/'claims'/f"{job['id']}.json",{'job':job,'contract_sha256':common.sha_json(c),'deadline_unix':deadline},exclusive=True)
                command=None
                def pulse():
                    now=time.time();require(now<deadline,'Absolute deadline reached')
                    common.write_json(root/'server_lease.json',{'job':job['id'],'contract_sha256':common.sha_json(c),
                        'issued_unix':now,'expires_unix':min(now+90,deadline),'supervisor_pid':os.getpid(),'worker_group_pid':None if child is None else child.pid})
                    common.write_json(root/'status.json',{'status':'running','active_job':job,'supervisor_pid':os.getpid(),
                        'worker_group_pid':None if child is None else child.pid,'completed':results,'updated_unix':now,
                        'deadline_unix':start['deadline_unix'],'condition_deadline_unix':deadline,'mac_required':False,
                        'worker_guard_argv':command,'supervisor_argv':[sys.executable,*sys.argv]})
                pulse();log=root/'logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
                command=['timeout','--signal=TERM','--kill-after=15s',str(max(1,int(deadline-time.time()))),
                    sys.executable,str(Path(__file__).resolve()),'--contract',str(root/'execution_contract.json'),
                    '--host',host,'--mode','fit','--job',job['id'],'--deadline',str(deadline)]
                with log.open('xb') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,env={**os.environ, 'PYTHONHASHSEED':str(job['seed'])},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    pulse()
                    while child.poll() is None:pulse();time.sleep(5)
                    require(child.returncode==0,'Worker exited '+str(child.returncode)+'; no automatic retry')
                manifest=read(root/'run'/job['id']/'terminal_manifest.json')
                require(manifest['scientific_success'] is True and manifest['status']=='complete' and manifest['job']==job and manifest['contract_sha256']==common.sha_json(c) and manifest['held_out_test_evaluated'] is False,'Missing/foreign successful terminal receipt')
                for p,s in manifest['files'].items():require(common.sha_file(safe_relative(root/'run'/job['id'],p))==s,'Terminal integrity mismatch')
                results[job['id']]={'terminal_manifest_sha256':common.sha_file(root/'run'/job['id']/'terminal_manifest.json')}
                child=None
            common.write_json(root/'status.json',{'status':'complete','completed':results,'completed_conditions':len(results),
                'endpoint_roles':len(results)*2,'updated_unix':time.time(),'mac_required':False})
        except BaseException as exc:
            if child is not None:stop_owned(child)
            failure={'status':'failed','type':type(exc).__name__,'message':str(exc),'active_job':job,
                'traceback':traceback.format_exc(),'completed':results,'updated_unix':time.time(),'automatic_retry':False}
            common.write_json(root/'failure.json',failure,exclusive=True);common.write_json(root/'status.json',failure)
            raise

if __name__ == "__main__":main()
