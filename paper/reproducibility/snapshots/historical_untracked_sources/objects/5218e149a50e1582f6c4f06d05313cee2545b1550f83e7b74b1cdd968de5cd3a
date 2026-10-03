"""Approved RTX4090 active-branch normalization; pilot three then budgeted expansion."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import gc
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
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from paper.scripts import observed_slot_parallel_common as common
from paper.scripts import run_titantpp_core_ablation as core
from paper.scripts import run_local_detail_benchmark as prior
from paper.scripts import run_multilag_detail_execution as base
from paper.scripts import run_observed_slot_partition as shared
from models.TPPs.CountAwareTitanActiveBranchNorm import ARM, ARMS, ROLE, metadata
read=lambda p:json.loads(Path(p).read_text())
require=common.require
VARIANT='count_only_log_regression'
TIME_METRIC=prior.TIME_METRIC
ALL_ARMS=ARMS
audit_replay_accounting=prior.audit_replay_accounting
SCHEMA='active_branch_norm_runpod4090_20260930_v1'


def validate(c):
    require(c['schema']==SCHEMA and c['arms']==list(ARMS),'Foreign normalization contract')
    require(c['training']=={'batch_size':128,'maximum_epochs':300,'minimum_epochs':40,'patience':40,
        'monitor':'validation_raw_quantity_rmse','tie':'strict_earliest_finite_minimum','warm_start':False},'Training drift')
    require(c['cost']['absolute_cap_usd']==40 and c['cost']['work_cap_usd']<=25,'Unapproved budget')
    require(c['limits']['per_condition_seconds']==36*3600 and c['limits']['total_wall_seconds']<=32*3600,'Time cap drift')
    expected=[{'id':f'{d}__{seed}__{ARM}','host':'4090','dataset':d,'seed':seed,'arm':ARM}
        for seed in (42,52,62) for d in ('yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket')]
    require(c['jobs']==expected and list(c['hosts'])==['4090'],'Scope/order changed')
    require(c['source']['files_sha256']==common.sha_json(c['source']['files']),'Invalid source closure')
    for p,digest in c['source']['files'].items():require(common.sha_file(ROOT/p)==digest,'Source drift '+p)
    for d in c['datasets']:require(common.sha_json(d)==c['dataset_sha256'][d['dataset_id']],'Dataset drift')
    return c


def authorization(c,root):
    validate(c);approval=read(root/'approval.json');start=read(root/'start_permit.json')
    require(approval['approved'] is True and approval['contract_sha256']==common.sha_json(c)
        and approval['hosts']==['4090'] and approval['user_instruction'],'Missing approval')
    require(start['contract_sha256']==common.sha_json(c) and start['approval_sha256']==common.sha_json(approval),'Foreign start')
    require(start['deadline_unix']-start['started_at_unix']<=c['limits']['total_wall_seconds']
        and start['started_at_unix']<=time.time()<start['deadline_unix'],'Expired/enlarged budget')
    require(start['work_rate_usd_hour']<=c['cost']['maximum_running_rate_usd_hour'],'Price exceeds reservation')
    return start


def native_runtime(c,host):
    from paper.scripts.quantity_comparison_runtime import configure_runtime,runtime_identity
    import platform
    require(Path(sys.executable).absolute()==Path(c['hosts'][host]['python']),'Wrong Python')
    configure_runtime('cuda:0',threads=4);r=runtime_identity('cuda:0')
    for k,v in c['hosts'][host]['runtime_expected'].items():require(r.get(k)==v,'Runtime mismatch '+k)
    require(platform.python_version().startswith('3.12.') and r['gpu']['name']=='NVIDIA GeForce RTX 4090'
        and r['gpu']['visible_device_count']==1,'Wrong Python/GPU')
    return r


def idle_gpu():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    return not p.stdout.strip()


def build_model(data,arm):
    from models.TPPs.CountAwareFactory import build_count_aware_model
    require(arm in (*ARMS,'titantpp_history_mlp'),'Wrong model')
    config={k:v for k,v in data['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
    return build_count_aware_model(arm,**config,train_log_mean=data['statistics']['train_log_mean'],
        train_log_std=data['statistics']['train_log_std'],max_seq_len=data['loader']['max_seq_len'])


def initial_states(data,seed):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    result={};states={};rngs={}
    with torch.random.fork_rng(devices=[]):
        for arm in (*ARMS,'titantpp_history_mlp'):
            torch.manual_seed(seed);model,_=build_model(data,arm);states[arm]=model.state_dict();rngs[arm]=torch.get_rng_state()
            result[arm]=canonical_state_dict_sha256(states[arm])
        baseline=states['titantpp_history_mlp'];candidate=states[ARM]
        require(set(candidate)==set(baseline)|{'active_branch_norm_identity'},'Unexpected state layout change')
        require(all(torch.equal(v,candidate[k]) for k,v in baseline.items()),'Baseline initial tensors changed')
        require(torch.equal(rngs[ARM],rngs['titantpp_history_mlp']),'Initial RNG changed')
    return result


def time_interface(data,frame,c):
    interface=base.time_interface(data,frame,c)
    interface['backbone_design']={'schema':SCHEMA,'design_sha256':c['design_sha256'],
        'architecture':c['architecture'],'train_time_scale':data['model']['time_scale'],
        'time_statistics':data['time_statistics']}
    return interface


class PulseBudget:
    def __init__(self,root,digest,job,deadline,lease=False):
        self.root,self.digest,self.job,self.deadline,self.lease=root,digest,job,deadline,lease
        self.last=0.;self.monotonic_deadline=time.monotonic()+max(0,deadline-time.time())
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
        self.last=now


def qualify(c,host,root):
    import torch
    from paper.scripts.verify_active_branch_norm import check
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    start=authorization(c,root);require(idle_gpu(),'GPU is occupied')
    common.write_json(root/'qualification.claim',{'pid':os.getpid(),'time':time.time(),'contract_sha256':common.sha_json(c)},exclusive=True)
    runtime=native_runtime(c,host)
    budget=PulseBudget(root,common.sha_json(c),'qualification',min(start['deadline_unix'],time.time()+5400))
    result={'status':'checking','host':host,'runtime':runtime,'contract_sha256':common.sha_json(c),
        'source_files_sha256':c['source']['files_sha256'],'initialization':{},'inputs':{},'baselines':{},'measurements':{}}
    result['correctness']=check('cuda:0')
    data=next(d for d in c['datasets'] if d['dataset_id']==c['hosts'][host]['assigned_datasets'][0])
    for arm in ARMS:
        result['measurements'][arm]={}
        for length in (64,256):
            budget();torch.manual_seed(42);model,_=build_model(data,arm);model=model.cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            dts=torch.randint(1,8,(128,length),device='cuda:0').float();qty=torch.rand_like(dts)*8
            mask=torch.ones_like(dts,dtype=torch.bool);mask[:32,:length//3]=False
            torch.cuda.reset_peak_memory_stats();times=[]
            for step in range(30):
                budget();torch.cuda.synchronize();t=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)['joint_loss'].mean()
                require(torch.isfinite(loss).item(),'Nonfinite synthetic loss');loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);require(torch.isfinite(norm).item(),'Nonfinite synthetic gradient')
                optimizer.step();require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite synthetic parameter')
                torch.cuda.synchronize();times.append(time.perf_counter()-t)
            peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
            require(0<peak<.8*total,'Native memory exceeds80%')
            result['measurements'][arm][str(length)]={'batch_size':128,'optimizer_updates':30,'step_seconds':times,
                'median_last10_seconds':statistics.median(times[-10:]),'peak_allocated_bytes':peak,'total_memory_bytes':total}
            del model,optimizer;gc.collect();torch.cuda.empty_cache()
    result['synthetic_optimizer_updates']=60
    for data in c['datasets']:
        if data['dataset_id'] not in c['hosts'][host]['assigned_datasets']:continue
        budget();frame,receipt=base.prepare_admitted_data(data);interface=time_interface(data,frame,c)
        result['inputs'][data['dataset_id']]={'receipt':receipt,'interface_sha256':common.sha_json(interface)}
        result['initialization'][data['dataset_id']]={str(s):initial_states(data,s) for s in (42,52,62)}
        for seed in (42,52,62):
            require(result['initialization'][data['dataset_id']][str(seed)]['titantpp_history_mlp']
                ==c['baseline_exposures'][data['dataset_id']][str(seed)]['initial_state_sha256'],
                'Original baseline initialization mismatch')
        # Re-evaluate a selected frozen MLP on validation, no fitting/held-out.
        ref=c['baseline_replays'][data['dataset_id']];path=Path(ref['checkpoint'])
        require(common.sha_file(path)==ref['checkpoint_sha256'],'Baseline checkpoint changed')
        endpoint=core.replay_checkpoint(path,data,frame,budget)
        require(all(math.isclose(endpoint[k],ref['metrics'][k],rel_tol=c['baseline_replay_tolerance']['relative'],abs_tol=c['baseline_replay_tolerance']['absolute'])
            for k in ('qty_rmse','qty_mae','time_nll')),'Frozen MLP replay mismatch')
        result['baselines'][data['dataset_id']]={'status':'passed','checkpoint_sha256':ref['checkpoint_sha256'],
            'qty_mae':endpoint['qty_mae'],'qty_rmse':endpoint['qty_rmse'],'time_nll':endpoint['time_nll']}
        del frame;gc.collect()
    canary=root/'qualification/storage_canary.bin';canary.parent.mkdir(exist_ok=True)
    token=os.urandom(4096);canary.write_bytes(token);require(canary.read_bytes()==token,'Server storage roundtrip failed')
    # Real scoped stop test; do not use any unrelated server PID.
    children=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True) for _ in range(2)]
    try:
        stop_owned(children[0]);require(children[0].poll() is not None and children[1].poll() is None,'Owned stop scope failed')
    finally:
        for child in children:stop_owned(child)
    result.update(status='passed',completed_unix=time.time(),server_storage_sha256=common.sha_file(canary),owned_stop_test='passed')
    common.write_json(root/'qualification/receipt.json',result,exclusive=True)
    print(json.dumps({'status':'passed','host':host,'qualification':str(root/'qualification/receipt.json')}),flush=True)


def run_fit(c,host,root,job,deadline):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    start=authorization(c,root);q=read(root/'qualification/receipt.json');verify_training_permit(c,root)
    require(q['status']=='passed' and q['runtime']==native_runtime(c,host),'Qualified runtime drift')
    require(job in c['jobs'] and job['host']==host,'Unapproved fit')
    budget=PulseBudget(root,common.sha_json(c),job['id'],min(deadline,start['deadline_unix']),lease=True)
    budget();data=next(d for d in c['datasets'] if d['dataset_id']==job['dataset'])
    frame,receipt=base.prepare_admitted_data(data);interface=time_interface(data,frame,c)
    arm,seed=job['arm'],job['seed'];initial=initial_states(data,seed)
    require(initial==q['initialization'][data['dataset_id']][str(seed)],'Native initialization changed')
    folder=root/'run'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    args=base.training_args({**c,'model_role':ROLE},data,folder)
    args.epochs=300;args.min_epochs=40;args.early_stopping_patience=40;args.seeds=str(seed)
    args.execution_role='fresh_active_branch_norm_validation'
    common.write_json(folder/'input_receipt.json',receipt,exclusive=True)
    common.write_json(folder/'initialization.json',initial,exclusive=True)
    run=folder/'runs'/arm/VARIANT/f'seed_{seed}'
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
            require(remaining<deadline-time.time(),'Full-epoch projected300-epoch cost exceeds condition budget')
            common.write_json(folder/'two_epoch_cost_receipt.json',{'status':'passed','same_scientific_fit':True,
                'measured_epoch_seconds':times,'projected_remaining_seconds':remaining,'not_a_completion_guarantee':True})
    with shared.audited_training(training,data,budget,status) as exposure:
        summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,interface_meta=interface,
            backbone=arm,quantity_variant=VARIANT,seed=seed)
    history=read(run/'history.json')['history'];steps=prior.audit_arm(history,summary,exposure,data,c['training'])
    require(summary['initial_state_sha256']==initial[arm],'Initial tensor identity mismatch')
    identity=training._resume_identity(args=args,backbone=arm,quantity_variant=VARIANT,seed=seed,
        monitor='validation_raw_quantity_rmse',interface_meta=interface,quantity_contract=quantity)
    endpoints={}
    for label,path in (('selected',Path(summary['checkpoint_path'])),('last',run/'last_epoch_state.pt')):
        epoch=summary['best_epoch'] if label=='selected' else len(history)
        endpoints[label]=replay_checkpoint(path,data,frame,budget,expected_arm=arm,expected_identity=identity,
            expected_initial=initial[arm],expected_epoch=epoch,expected_seed=seed)
        prior.audit_replay_accounting(endpoints[label])
        require(all(math.isclose(endpoints[label][k],history[epoch-1][v],rel_tol=1e-10,abs_tol=1e-8)
            for k,v in (('qty_mae','val_qty_mae'),('qty_rmse','val_qty_rmse'),('time_nll','val_time_nll'))),'Endpoint replay mismatch')
    ref=c['baseline_exposures'][data['dataset_id']][str(seed)]
    require(common.sha_file(ref['path'])==ref['sha256'],'Reference exposure changed')
    baseline=read(ref['path'])
    # Common shuffle seed and loader: compare every recorded overlap, no synthetic replacement.
    n=min(len(exposure['train']),len(baseline['train']))
    require(n>0 and exposure['train'][:n]==baseline['train'][:n],'Reference train batch prefix differs')
    require(exposure['validation'][0]==baseline['validation'][0],'Reference validation population differs')
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
        'contract_sha256':common.sha_json(c),'job':job,'completed_unix':time.time()},exclusive=True)


def verify_training_permit(c,root):
    p=read(root/'training_permit.json');start=read(root/'start_permit.json')
    require(p['contract_sha256']==common.sha_json(c) and p['start_permit_sha256']==common.sha_json(start),'Foreign training permit')
    require(set(p['qualifications'])==set(c['hosts']),'Both native qualifications required')
    for h,r in p['qualifications'].items():
        require(r['status']=='passed' and r['host']==h and r['contract_sha256']==common.sha_json(c)
            and r['source_files_sha256']==c['source']['files_sha256'],'Native qualification not passed')
    return p


def stop_owned(child):
    if child.poll() is not None:return
    require(os.getpgid(child.pid)==child.pid,'Owned child process-group identity changed')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def dispatch(c,host,root):
    start=authorization(c,root);permit=verify_training_permit(c,root)
    require(read(root/'qualification/receipt.json')==permit['qualifications'][host],'Local qualification changed')
    require(idle_gpu(),'GPU occupied before launch')
    require(shutil.which('timeout') is not None,'Independent native timeout guard unavailable')
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        common.write_json(root/'supervisor.claim',{'pid':os.getpid(),'started_unix':time.time(),'contract_sha256':common.sha_json(c)},exclusive=True)
        results={};child=None;job=None
        try:
            for job in (j for j in c['jobs'] if j['host']==host):
                validate(c);require(time.time()<start['deadline_unix'],'Campaign budget expired')
                if job['seed']!=42 and not (root/'expansion_receipt.json').exists():
                    pilot=[j for j in c['jobs'] if j['seed']==42]
                    require(all(j['id'] in results for j in pilot),'Pilot incomplete')
                    estimates={}
                    for j in pilot:
                        r=root/'run'/j['id']/'runs'/ARM/VARIANT/'seed_42'
                        times=[x['elapsed_seconds'] for x in read(r/'epoch_timing.json')['epochs']]
                        estimates[j['dataset']]=max(times)*300*1.2+600
                    remaining=sum(estimates[j['dataset']] for j in c['jobs'] if j['seed']!=42)
                    receipt={'pilot_complete':True,'performance_not_used_for_admission':True,
                        'remaining_max300_seconds':remaining,'available_seconds':start['deadline_unix']-time.time(),
                        'expanded':remaining<start['deadline_unix']-time.time(),'estimated_seconds':estimates}
                    common.write_json(root/'expansion_receipt.json',receipt,exclusive=True)
                    if not receipt['expanded']:
                        common.write_json(root/'status.json',{'status':'pilot_complete_budget_limited','completed':results,
                            'not_started':[j for j in c['jobs'] if j['seed']!=42],'updated_unix':time.time()})
                        return
                require(idle_gpu(),'Other GPU owner detected')
                require(not (root/'run'/job['id']).exists(),'Run exists; no implicit restart')
                deadline=min(start['deadline_unix'],time.time()+36*3600,
                    c.get('retained_condition_deadlines',{}).get(job['id'],float('inf')))
                common.write_json(root/'claims'/f"{job['id']}.json",{'job':job,'contract_sha256':common.sha_json(c),'deadline_unix':deadline},exclusive=True)
                def pulse():
                    now=time.time();require(now<deadline,'Absolute deadline reached')
                    common.write_json(root/'server_lease.json',{'job':job['id'],'contract_sha256':common.sha_json(c),
                        'issued_unix':now,'expires_unix':min(now+90,deadline),'supervisor_pid':os.getpid()})
                    common.write_json(root/'status.json',{'status':'running','active_job':job,'supervisor_pid':os.getpid(),
                        'worker_group_pid':None if child is None else child.pid,'completed':results,'updated_unix':now,
                        'deadline_unix':start['deadline_unix'],'condition_deadline_unix':deadline,'mac_required':False})
                pulse();log=root/'logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
                command=['timeout','--signal=TERM','--kill-after=15s',str(max(1,int(deadline-time.time()))),
                    sys.executable,str(Path(__file__).resolve()),'--contract',str(root/'execution_contract.json'),
                    '--host',host,'--mode','fit','--job',job['id'],'--deadline',str(deadline)]
                with log.open('xb') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:pulse();time.sleep(5)
                    require(child.returncode==0,'Worker exited '+str(child.returncode)+'; no automatic retry')
                manifest=read(root/'run'/job['id']/'terminal_manifest.json')
                require(manifest['scientific_success'] is True,'Missing successful terminal receipt')
                for p,s in manifest['files'].items():require(common.sha_file(root/'run'/job['id']/p)==s,'Terminal integrity mismatch')
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--host',choices=('4090',),required=True)
    p.add_argument('--mode',choices=('qualify','fit','dispatch'),required=True);p.add_argument('--job');p.add_argument('--deadline',type=float)
    a=p.parse_args();c=read(a.contract);root=Path(c['hosts'][a.host]['root'])
    require(Path(a.contract).resolve()==root/'execution_contract.json','Wrong contract root')
    require(Path.cwd().resolve()==Path(c['hosts'][a.host]['source_root']),'Wrong pinned checkout')
    for k,v in c['hosts'][a.host]['environment'].items():require(os.environ.get(k)==v,'Process environment mismatch '+k)
    if a.mode=='qualify':qualify(c,a.host,root)
    elif a.mode=='dispatch':dispatch(c,a.host,root)
    else:run_fit(c,a.host,root,next(j for j in c['jobs'] if j['id']==a.job),a.deadline)

# Existing frozen endpoint evaluation, copied unchanged; build_model routes new encoders.
def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None, expected_identity=None, expected_initial=None, expected_epoch=None, expected_seed=42):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    payload = torch_load_checkpoint(Path(path), map_location="cpu")
    if expected_arm is not None:
        from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec
        selector = checkpoint_monitor_spec("validation_raw_quantity_rmse")
        require(payload.get("backbone") == expected_arm and payload.get("seed") == expected_seed
                and payload.get("variant") == VARIANT, "Replay arm/seed/objective mismatch")
        require(payload.get("resume_identity") == expected_identity
                and payload.get("interface_meta") == expected_identity["interface_meta"]
                and payload.get("initial_state_sha256") == expected_initial,
                "Replay source/initialization/interface identity mismatch")
        revision = expected_identity["arguments"]["source_revision"] if "source_revision" in expected_identity.get("arguments", {}) else None
        if revision is not None:
            require(payload.get("source_revision") == revision and payload.get("source_revision_history") == [revision], "Replay source revision mismatch")
        require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
                and payload.get("checkpoint_monitor_history_key") == selector["history_key"]
                and payload.get("checkpoint_selection") == selector["selection"], "Replay selector mismatch")
        require(payload.get("epoch", payload.get("best_epoch")) == expected_epoch, "Replay epoch mismatch")
    head = payload.get("interface", {}).get("time_head", {})
    # The shared trainer writes its interface under interface_meta.
    if not head:
        head = payload.get("interface_meta", {}).get("time_head", {})
    require(head.get("observation_likelihood") == data["model"]["time_observation_contract"],
            "Replay observation likelihood metadata mismatch")
    result = _replay_validation(payload, path, data, frame, budget, device=device)
    result["time_metric"] = TIME_METRIC
    return result

def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    """One streaming validation pass with fixed train-derived strata; no records."""
    import numpy as np
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(payload.get("backbone") in ALL_ARMS, "Unexpected endpoint backbone")
    validate_checkpoint_route(payload, payload["backbone"])
    require(payload["evaluation_scope"] == "validation_only"
            and payload["held_out_test_evaluated"] is False, "Replay split changed")
    state_digest = canonical_state_dict_sha256(payload["model_state_dict"])
    require(payload.get("model_state_sha256") == state_digest, "Checkpoint state digest differs")
    model, _ = build_model(data, payload["backbone"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = make_loader(frame, target_split="validation", **{
        key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")},
        shuffle=False, generator=None)
    q_bounds, h_bounds = data["quantity_boundaries_all_train_rows"], data["history_boundaries"]
    partitions = [("quantity", q_bounds), ("history", h_bounds)]
    if "additional_history_boundaries" in data:
        partitions.append(("additional_history", data["additional_history_boundaries"]))
    accumulator = {"overall": [0, 0., 0., 0.], "body": [0, 0., 0., 0.], "tail": [0, 0., 0., 0.]}
    for kind, bounds in partitions:
        accumulator.update({f"{kind}_{i}": [0, 0., 0., 0.] for i in range(len(bounds) + 1)})
    with torch.no_grad():
        for _, dts, mask, _, quantities in loader:
            budget_check()
            result = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
            q = result["true_qty"].cpu().numpy().astype(np.float64)
            pred = result["pred_qty"].cpu().numpy().astype(np.float64)
            t = result["time_loss"].cpu().numpy().astype(np.float64)
            h = result["history_length"].cpu().numpy()
            require(np.isfinite(q).all() and np.isfinite(pred).all() and np.isfinite(t).all(),
                    "Nonfinite checkpoint replay")
            masks = {"overall": np.ones(q.shape, dtype=bool), "body": q <= q_bounds[2], "tail": q > q_bounds[3]}
            for kind, bounds in partitions:
                values = q if kind == "quantity" else h
                ids = np.searchsorted(bounds, values, side="left")
                masks.update({f"{kind}_{i}": ids == i for i in range(len(bounds) + 1)})
            error = pred - q
            for key, selected in masks.items():
                acc = accumulator[key]
                acc[0] += int(selected.sum())
                acc[1] += float(np.abs(error[selected]).sum())
                acc[2] += float(np.square(error[selected]).sum())
                acc[3] += float(t[selected].sum())
    budget_check()
    def finish(acc):
        n, absolute, squared, temporal = acc
        return {"count": n, "qty_mae": absolute / n if n else None, "qty_sse": squared,
            "qty_rmse": math.sqrt(squared / n) if n else None,
            "time_nll": temporal / n if n else None}
    result = {**finish(accumulator["overall"]), "body": finish(accumulator["body"]),
        "tail": finish(accumulator["tail"]), "quantity_boundaries": q_bounds,
        "history_boundaries": h_bounds, "quantity_cells": [], "history_cells": [],
        "checkpoint_path": str(path), "state_sha256": state_digest,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    for kind, bounds in partitions:
        result[kind + "_boundaries"] = bounds
        result[kind + "_cells"] = [{"bin": i, **finish(accumulator[f"{kind}_{i}"])}
                                    for i in range(len(bounds) + 1)]
    require(result["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
            "Replay validation target count changed")
    audit_replay_accounting(result)
    return result


if __name__=="__main__":main()
