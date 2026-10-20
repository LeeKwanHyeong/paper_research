"""Isolated original-data Shared15/MAC cost profile, never a scientific fit.

Copied to an approved experiment root as ``benchmark.py``.  The frozen source
closure lives in ``source/``; this module deliberately imports no ML packages
until a verified child enters preparation, synthetic qualification or profiling.
"""
import argparse, hashlib, json, math, os, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
sys.path.insert(0,str(SOURCE))
MLP='titantpp_history_mlp_shared15_v1'
MAC='titantpp_titans_mac'
DATASETS={'raf_spare_parts':84,'yellow_trip_hourly':256,'intermittent_frozen_5000':256}
GPU_UUID='GPU-c9adc246-ef66-7906-bc07-6e152fa9952f'
WARMUP=5
MEASURED=32
BATCH_SIZE=128
TOTAL_BATCHES=WARMUP+MEASURED
TARGETS=TOTAL_BATCHES*BATCH_SIZE
COMMON_HEAD_KEYS=('b_t','w_raw','v_t.weight','time_scale_weight.weight','quantity_head.weight','quantity_head.bias')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def require(x,msg):
    if not x:raise RuntimeError(msg)
def validate_contract(c):
    require(c['models']==[MLP,MAC],'Exactly Shared15 and frozen MAC required')
    require(c['host']['gpu_uuid']==GPU_UUID,'Only the approved native5090 GPU is allowed')
    expected=c['host']['runtime_expected']
    require(all(expected.get(k)==v for k,v in {'python':'3.12.13','torch':'2.11.0+cu130','cudnn':92000}.items()),'Native5090 runtime contract')
    require(len(c['datasets'])==3 and {d['dataset_id'] for d in c['datasets']}==set(DATASETS),'Original three datasets required')
    for d in c['datasets']:
        require(d['loader']['max_seq_len']==DATASETS[d['dataset_id']] and d['loader']['batch_size']==BATCH_SIZE,'Loader exposure changed')
        require(d['model']['lambda_log_qty']==1. and d['model']['lambda_tail']==0. and d['model']['time_head_lr_multiplier']==1.,'Shared objective changed')
        require(d['model']['quantity_variant']=='count_only_log_regression' and d['model']['time_head_mode']=='heteroscedastic_lognormal_duration','Shared head changed')
        require(d['optimizer']['lr']==.001 and d['optimizer']['grad_clip']==1. and d['optimizer']['weight_decay']==.01 and d['optimizer']['name']=='AdamW','Optimizer changed')
    scope=c['scope']
    require(scope.get('train_only') is True and all(scope.get(k) is False for k in ('accuracy_evaluation','checkpoint_writes','full_epochs','full_fits','held_out','validation')),'Train-only scope required')
    p=c['protocol']
    require(p['seed']==42 and p['batch_size']==BATCH_SIZE and p['repetitions']==3 and p['modes']==['train','eval'] and p['warmup_batches_per_worker']==WARMUP and p['measured_batches_per_worker']==MEASURED and p['workers']==36 and p['parallel_workers']==1 and p['automatic_retry'] is False,'Paired exposure contract changed')
    seconds=c.get('max_seconds',p.get('maximum_seconds_including_remote_preparation',1800))
    require(type(seconds) in (int,float) and math.isfinite(seconds) and 0<seconds<=1800,'Total budget must be in (0,1800] seconds')
    child=c.get('child_timeout_seconds',p.get('child_timeout_seconds',seconds))
    require(type(child) in (int,float) and math.isfinite(child) and 0<child<=seconds,'Child timeout exceeds total budget')
    return float(seconds),float(child)
def verify():
    c=read(ROOT/'contract.json'); a=read(ROOT/'approval.json')
    require(a['contract_sha256']==canonical(c) and a['approved'] is True,'Approval identity')
    validate_contract(c)
    if 'source_closure_sha256' in c:
        require(c['source_closure_sha256']==canonical(c['source_files']),'Source closure identity')
    for p,h in c['source_files'].items():require(sha(SOURCE/p)==h,'Frozen source mismatch: '+p)
    for p,h in c['operational_files'].items():require(sha(ROOT/p)==h,'Operational source mismatch: '+p)
    return c

def model_for(d,arm):
    require(arm in (MLP,MAC),'Unknown arm')
    if arm==MAC:
        from paper.scripts.titans_mac_observed_time_runtime import build_model
        return build_model(d)[0]
    from paper.scripts.run_titantpp_shared_history_raw9 import build_model
    return build_model(d,arm)[0]

def assert_owner(c):
    claim=read(ROOT/'claim.json')
    require(claim['contract_sha256']==canonical(c),'Owner contract changed')
    require(os.getppid()==claim['pid'],'Wrong supervisor')
    require(time.time()<claim['deadline_unix'],'Deadline expired')
    return claim

def native_runtime(c):
    from paper.scripts.quantity_comparison_runtime import configure_runtime,runtime_identity
    require(Path(sys.executable).resolve()==Path(c['host']['python']).resolve(),'Python path changed')
    for k,v in c['environment'].items():require(os.environ.get(k)==v,'Environment changed: '+k)
    configure_runtime('cuda:0',threads=4)
    runtime=runtime_identity('cuda:0')
    for k,v in c['host']['runtime_expected'].items():require(runtime.get(k)==v,'Runtime changed: '+k)
    require(runtime['gpu']['uuid']==c['host']['gpu_uuid'],'Wrong GPU')
    return runtime

def train_only_frame(pl,path):
    # The lazy predicate is applied BEFORE collect; held-out rows are not materialized.
    return pl.scan_parquet(path).filter(pl.col('chronological_split')=='train').collect().sort(['oper_part_no','seq'])

def jobs(c):
    return [(d['dataset_id'],arm,mode,rep) for d in c['datasets'] for mode in ('train','eval') for rep in (1,2,3) for arm in ((MLP,MAC) if rep!=2 else (MAC,MLP))]

def gpu_snapshot():
    def query(args):return subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    return {'unix':time.time(),'gpu':query(['--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm']),
            'compute':query(['--query-compute-apps=pid,process_name,used_memory'])}

def qualification_path(arm,device):
    return ROOT/'qualification'/f"{arm}__{device.replace(':','_')}.json"

def qualify(arm,device):
    """Synthetic-only numerical/state checks on this runtime before real data."""
    c=verify();assert_owner(c)
    require(arm in (MLP,MAC) and device in ('cpu','cuda:0'),'Qualification identity')
    import gc,torch
    from paper.scripts.quantity_comparison_runtime import configure_runtime,runtime_identity
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    if device=='cuda:0':runtime=native_runtime(c)
    else:configure_runtime('cpu',threads=4);runtime=runtime_identity('cpu')
    if arm==MAC:
        from paper.scripts.titans_mac_observed_time_runtime import configure_dynamo
        configure_dynamo()
    rows=[]
    for d in c['datasets']:
        assert_owner(c);torch.manual_seed(42)
        model=model_for(d,arm).to(device).eval()
        length=d['loader']['max_seq_len'] if device=='cuda:0' else 8
        batch=BATCH_SIZE if device=='cuda:0' else 2
        dt=torch.ones(batch,length,device=device)
        qty=(torch.arange(batch*length,device=device).reshape(batch,length)%9+1).float()
        mask=torch.ones_like(dt,dtype=torch.bool);mask[-1,max(2,length//2):]=False
        initial=canonical_state_dict_sha256(model.state_dict())
        captured=[];original=model.log_observation_dt
        def capture(h,t):captured.append(h.detach().clone());return original(h,t)
        model.log_observation_dt=capture
        with torch.no_grad():
            before=target_outputs(model,dt,mask,qty,lambda_log_qty=d['model']['lambda_log_qty'])
            changed_qty=qty.clone();changed_dt=dt.clone()
            idx=torch.arange(batch,device=device);target=mask.sum(1)-1
            changed_qty[idx,target]+=100;changed_qty[~mask]=999
            changed_dt[idx,target]=2;changed_dt[~mask]=999
            changed=target_outputs(model,changed_dt,mask,changed_qty,lambda_log_qty=d['model']['lambda_log_qty'])
            require(torch.equal(captured[0],captured[1]) and torch.equal(before['pred_qty'],changed['pred_qty']),'Synthetic target/padding leak')
            again=target_outputs(model,dt,mask,qty,lambda_log_qty=d['model']['lambda_log_qty'])
            require(all(torch.equal(before[k],again[k]) for k in ('pred_qty','time_loss','joint_loss')),'Synthetic state persisted across forwards')
            require(initial==canonical_state_dict_sha256(model.state_dict()),'Synthetic eval modified state')
        model.log_observation_dt=original;model.train()
        opt=build_optimizer(model,lr=d['optimizer']['lr'],time_head_lr_multiplier=d['model']['time_head_lr_multiplier'])
        opt.zero_grad(set_to_none=True)
        loss=target_outputs(model,dt,mask,qty,lambda_log_qty=d['model']['lambda_log_qty'])['joint_loss'].mean()
        require(torch.isfinite(loss).item(),'Nonfinite synthetic loss');loss.backward()
        require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),'Nonfinite synthetic gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),d['optimizer']['grad_clip']);opt.step()
        if device=='cuda:0':torch.cuda.synchronize()
        require(torch.isfinite(norm).item() and all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite synthetic update')
        require(initial!=canonical_state_dict_sha256(model.state_dict()),'Synthetic optimizer did not update model')
        rows.append({'dataset':d['dataset_id'],'batch_size':batch,'length':length,'optimizer_steps':1,'finite':True,'causality':True,'padding':True,'fresh_state':True})
        del model,opt,loss,before,changed,again,captured,original,dt,qty,mask
        gc.collect()
        if device=='cuda:0':torch.cuda.empty_cache()
    write(qualification_path(arm,device),{'status':'passed','contract_sha256':canonical(c),'arm':arm,'device':device,'pid':os.getpid(),'runtime':runtime,'rows':rows,'real_data_loaded':False,'accuracy_evaluated':False,'checkpoint_saved':False})

def verify_qualifications(c):
    evidence={}
    for device in ('cpu','cuda:0'):
        for arm in (MLP,MAC):
            path=qualification_path(arm,device);q=read(path)
            require(q['status']=='passed' and q['contract_sha256']==canonical(c) and q['arm']==arm and q['device']==device and q['real_data_loaded'] is False,'Synthetic qualification missing or stale')
            require({r['dataset'] for r in q['rows']}==set(DATASETS),'Synthetic dataset coverage')
            evidence[str(path.relative_to(ROOT))]=sha(path)
    return evidence

def prepare():
    c=verify();assert_owner(c);qualification=verify_qualifications(c)
    import polars as pl, torch, numpy as np
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    for d in c['datasets']:
        ident=d['inherited_data_identity']; path=ident['data']['path']
        require(sha(path)==ident['data']['sha256'],'Parquet identity')
        require(sha(ident['split_manifest']['path'])==ident['split_manifest']['sha256'],'Split identity')
        # Predicate before materialization; no validation or held-out records accessed.
        frame=train_only_frame(pl,path)
        require(frame['chronological_split'].unique().to_list()==['train'],'Train-only')
        frame=prepare_count_frame(frame)
        gen=torch.Generator().manual_seed(42)
        loader=make_loader(frame,target_split='train',**{k:d['loader'][k] for k in ('batch_size','lookback_weeks','max_seq_len')},shuffle=True,generator=gen)
        require(len(loader.dataset)==ident['populations']['train']['target_count'],'Train target population changed')
        # Explicit sampled permutation, shared by every model/repetition/mode.
        # Record target IDs instead of depending on DataLoader base_seed side effects.
        require(len(loader.dataset)>=TARGETS,'Insufficient distinct train targets')
        order=torch.randperm(len(loader.dataset),generator=gen).tolist()[:TARGETS]
        from data_loader.event_seq_data_module import collate_week_lookback
        batches=[]; manifest=[]
        for n in range(TOTAL_BATCHES):
            assert_owner(c)
            ids=order[n*128:(n+1)*128]
            raw=collate_week_lookback([loader.dataset[i] for i in ids]); _,dt,mask,_,qty=raw
            batches.append({'dts':dt,'mask':mask,'quantities':qty})
            targets=[]
            for i in ids:
                p,j=loader.dataset.index[i]
                targets.append([loader.dataset.parts[p],loader.dataset.seq_lists[p][j+1]])
            lens=mask.sum(1).tolist()
            manifest.append({'batch':n,'phase':'warmup' if n<5 else 'measured','targets':targets,'valid_lengths':lens,'padded_length':int(mask.shape[1]),'tensor_sha256':canonical_state_dict_sha256(batches[-1])})
        require(len({str(x) for m in manifest for x in m['targets']})==4736,'Duplicate targets')
        states={};heads={}
        for arm in (MLP,MAC):
            torch.manual_seed(42);m=model_for(d,arm);state=m.state_dict()
            states[arm]={'state_sha256':canonical_state_dict_sha256(state),'parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad)}
            heads[arm]={k:state[k] for k in COMMON_HEAD_KEYS}
            del m
        require(all(torch.equal(heads[MLP][k],heads[MAC][k]) for k in heads[MLP]),'Common head initial values differ')
        p=ROOT/'batches'/f"{d['dataset_id']}.pt";p.parent.mkdir(exist_ok=True);torch.save(batches,p)
        write(p.with_suffix('.json'),{'split':'train','population':len(loader.dataset),'seed':42,'sampling':'first 4736 of torch.randperm with dedicated seed42; shared frozen tensors; not original fit prefix','file_sha256':sha(p),'batches':manifest,'initial':states,'common_head_equal':True})
    write(ROOT/'preflight.json',{'status':'passed','pid':os.getpid(),'unix':time.time(),'contract_sha256':canonical(c),'source_closure':canonical(c['source_files']),'qualification_files':qualification,'gpu':gpu_snapshot(),'data_scope':'train only; no GPU optimizer steps in preparation'})

def worker(dataset,arm,mode,rep):
    c=verify(); t0=time.perf_counter(); start=time.time(); out=ROOT/'results'/f'{dataset}__{arm}__{mode}__r{rep}.json'
    claim=assert_owner(c)
    require((dataset,arm,mode,rep) in jobs(c),'Unapproved worker')
    require(not out.exists(),'Existing worker result; automatic retry forbidden')
    d=next(d for d in c['datasets'] if d['dataset_id']==dataset)
    import torch
    from paper.scripts.run_taxi_quantity_interface_ablation import set_seed
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    runtime=native_runtime(c)
    if arm==MAC:
        from paper.scripts.titans_mac_observed_time_runtime import configure_dynamo
        configure_dynamo()
    set_seed(42)
    torch.use_deterministic_algorithms(True,warn_only=False)
    model=model_for(d,arm)
    p=ROOT/'batches'/f'{dataset}.pt';bm=read(p.with_suffix('.json'))
    initial=canonical_state_dict_sha256(model.state_dict())
    require(initial==bm['initial'][arm]['state_sha256'],'Initial tensors changed')
    require(sha(p)==bm['file_sha256'],'Shared batch cache changed')
    batches=torch.load(p,map_location='cpu',weights_only=True)
    require(bm['split']=='train' and bm['common_head_equal'] is True and len(batches)==TOTAL_BATCHES and len(bm['batches'])==TOTAL_BATCHES,'Shared batch exposure changed')
    for b,m in zip(batches,bm['batches']):require(canonical_state_dict_sha256(b)==m['tensor_sha256'],'Batch tensors changed')
    model.to('cuda:0');model.train(mode=='train')
    opt=build_optimizer(model,lr=d['optimizer']['lr'],time_head_lr_multiplier=d['model']['time_head_lr_multiplier']) if mode=='train' else None
    rows=[]; baseline=None;low_baseline=None;setup_seconds=None
    from torch._dynamo.utils import counters
    for n,b in enumerate(batches):
        require(time.time()<claim['deadline_unix'] and os.getppid()==claim['pid'],'Parent/deadline lost')
        if n==5:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            baseline={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
            low_baseline=dict(baseline)
        if n>=WARMUP:
            for k,f in (('allocated',torch.cuda.memory_allocated),('reserved',torch.cuda.memory_reserved)):
                low_baseline[k]=min(low_baseline[k],f())
        torch.cuda.synchronize();ts=time.perf_counter()
        if n==0:setup_seconds=ts-t0
        dt=b['dts'].to('cuda:0');mask=b['mask'].to('cuda:0');qty=b['quantities'].to('cuda:0')
        torch.cuda.synchronize();tc=time.perf_counter()
        with torch.set_grad_enabled(mode=='train'):
            if opt is not None:opt.zero_grad(set_to_none=True)
            outputs=target_outputs(model,dt,mask,qty,lambda_log_qty=d['model']['lambda_log_qty'])
            loss=outputs['joint_loss'].mean()
            if opt is not None:
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),d['optimizer']['grad_clip']);opt.step()
        torch.cuda.synchronize();te=time.perf_counter()
        # Finite checks and evidence serialization are outside the timing interval.
        require(torch.isfinite(loss).item(),'Nonfinite loss')
        if opt is not None:require(torch.isfinite(norm).item(),'Nonfinite gradient')
        row={'batch':n,'phase':'warmup' if n<5 else 'measured','transfer_seconds':tc-ts,'compute_seconds':te-tc,'transfer_compute_seconds':te-ts,'targets':128,'valid_events':int(b['mask'].sum()),'tensor_sha256':bm['batches'][n]['tensor_sha256'],'unique_compiled_graphs':int(counters['stats']['unique_graphs'])}
        rows.append(row)
        del outputs,loss,dt,mask,qty
        write(ROOT/'progress.json',{'pid':os.getpid(),'dataset':dataset,'arm':arm,'mode':mode,'rep':rep,'completed_batches':n+1,'unix':time.time()})
    peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite parameters')
    steps=sorted({int(s['step'].item()) for s in opt.state.values() if 'step' in s}) if opt else []
    require(mode!='train' or steps==[37],'Optimizer exposure mismatch')
    if mode=='eval':require(canonical_state_dict_sha256(model.state_dict())==initial,'Eval persisted parameter updates')
    measured_graphs=rows[-1]['unique_compiled_graphs']-rows[WARMUP-1]['unique_compiled_graphs']
    write(out,{'status':'success','dataset':dataset,'arm':arm,'mode':mode,'rep':rep,'seed':42,'pid':os.getpid(),'owner_pid':claim['pid'],'started_unix':start,'finished_unix':time.time(),'process_seconds':time.perf_counter()-t0,'runtime':runtime,'contract_sha256':canonical(c),'source_closure_sha256':canonical(c['source_files']),'batch_manifest_sha256':sha(p.with_suffix('.json')),'input_file_sha256':d['inherited_data_identity']['data']['sha256'],'parameters':bm['initial'][arm]['parameters'],'initial_state_sha256':initial,'optimizer_steps':37 if opt else 0,'eval_batches':37 if not opt else 0,'targets_total':TARGETS,'targets_measured':MEASURED*BATCH_SIZE,'common_head_equal':True,'memory_baseline_bytes':baseline,'memory_low_baseline_bytes':low_baseline,'memory_peak_bytes':peak,'cold_first_batch_seconds':rows[0]['transfer_compute_seconds'],'process_setup_seconds':setup_seconds,'compiled_graphs_after_warmup':rows[WARMUP-1]['unique_compiled_graphs'],'compiled_graphs_during_measured':measured_graphs,'rows':rows,'compiler_counters':{str(k):{str(a):int(b) for a,b in v.items() if isinstance(b,(int,float))} for k,v in counters.items()},'accuracy_evaluated':False,'checkpoint_saved':False,'scientific_fit':False})

def terminate_owned(p):
    # Popen's owned process group only; never signal a process found by pattern.
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)

def process_absent(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return True
    except PermissionError:return False
    return False


def reject_existing_execution():
    markers=('claim.exclusive','claim.json','status.json','terminal_manifest.json','preflight.json','error.json')
    require(not any((ROOT/name).exists() for name in markers),'Existing root execution claim/output; automatic retry forbidden')
    for folder in ('results','logs','gpu','batches','qualification','child_exits'):
        path=ROOT/folder
        require(not path.exists() or (path.is_dir() and not any(path.iterdir())),'Existing execution output: '+folder)


def controller():
    c=verify();require(str(ROOT)==c['host']['root'],'Must execute in isolated remote root')
    require(str(SOURCE)==c['host']['source_root'],'Frozen source root changed')
    reject_existing_execution()
    seconds,child_timeout=validate_contract(c)
    first=gpu_snapshot();require(c['host']['gpu_uuid'] in first['gpu'] and not first['compute'],'GPU occupied/wrong GPU')
    start=time.time();mono=time.monotonic();deadline=mono+seconds
    claim={'pid':os.getpid(),'started_unix':start,'deadline_unix':start+seconds,'contract_sha256':canonical(c),'max_seconds':seconds}
    # Never remove either claim, including after failure.  Every retry needs a new root.
    with (ROOT/'claim.exclusive').open('x') as f:json.dump(claim,f,indent=2)
    with (ROOT/'claim.json').open('x') as f:json.dump(claim,f,indent=2)
    write(ROOT/'status.json',{'status':'qualifying',**claim});write(ROOT/'gpu_initial.json',first)
    p=None;completed=[];exits=[];status='failed_no_automatic_retry';error=None
    def launch(cmd,label):
        nonlocal p
        require(time.monotonic()<deadline,'Absolute experiment budget exhausted')
        obs=gpu_snapshot();write(ROOT/'gpu'/f'{label}_before.json',obs);require(not obs['compute'],'Foreign CUDA activity')
        log=ROOT/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
        child_started=time.time();child_mono=time.monotonic()
        child_deadline=min(deadline,child_mono+child_timeout)
        timeout=False;launch_error=None
        with log.open('x') as f:
            p=subprocess.Popen(cmd,cwd=SOURCE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                last_observe=time.monotonic()
                while p.poll() is None:
                    if time.monotonic()>=child_deadline:
                        timeout=True;raise TimeoutError('Bounded child/absolute experiment deadline reached')
                    if time.monotonic()-last_observe>=10:
                        obs=gpu_snapshot();write(ROOT/'gpu'/f'{label}_during.json',obs)
                        others={int(line.split(',')[0]) for line in obs['compute'].splitlines() if line.strip()}-{p.pid}
                        require(not others,'Foreign CUDA process appeared')
                        last_observe=time.monotonic()
                    time.sleep(min(1,max(0.01,child_deadline-time.monotonic())))
                # wait() observes the real child OS exit, including nonzero exits.
                p.wait(timeout=1)
                require(p.returncode==0,f'{label} failed with returncode {p.returncode}')
            except BaseException as exc:
                launch_error=repr(exc);terminate_owned(p);raise
            finally:
                absent=process_absent(p.pid)
                record={'label':label,'argv':cmd,'pid':p.pid,'owner_pid':os.getpid(),'returncode':p.returncode,
                        'actual_child_exit_observed':p.returncode is not None,'pid_absent_after_wait':absent,
                        'timed_out':timeout,'error':launch_error,'started_unix':child_started,
                        'finished_unix':time.time(),'wall_seconds':time.monotonic()-child_mono,'log':str(log.relative_to(ROOT))}
                write(ROOT/'child_exits'/f'{label}.json',record);exits.append(record)
        require(record['actual_child_exit_observed'] and record['pid_absent_after_wait'],'Child exit/absence unconfirmed')
        obs=gpu_snapshot();write(ROOT/'gpu'/f'{label}_after.json',obs);require(not obs['compute'],'Unexpected remaining CUDA process')
    def stop(signum,frame):raise RuntimeError('Supervisor interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for device in ('cpu','cuda:0'):
            for arm in (MLP,MAC):
                label=f"qualify__{arm}__{device.replace(':','_')}"
                launch([sys.executable,str(ROOT/'benchmark.py'),'qualify','--arm',arm,'--device',device],label)
        launch([sys.executable,str(ROOT/'benchmark.py'),'prepare'],'prepare')
        for dataset,arm,mode,rep in jobs(c):
            label=f'{dataset}__{arm}__{mode}__r{rep}'
            write(ROOT/'status.json',{'status':'profiling','current':label,'completed':completed,**claim})
            launch([sys.executable,str(ROOT/'benchmark.py'),'worker','--dataset',dataset,'--arm',arm,'--mode',mode,'--rep',str(rep)],label)
            result=read(ROOT/'results'/f'{label}.json')
            require(result['status']=='success' and result['contract_sha256']==canonical(c),'Worker result identity')
            completed.append(label)
        require(len(completed)==36 and len(exits)==41,'Incomplete child exposure')
        require(all(x['actual_child_exit_observed'] and x['returncode']==0 and x['pid_absent_after_wait'] for x in exits),'Unconfirmed child exit')
        status='complete'
    except BaseException as exc:
        error=repr(exc);write(ROOT/'error.json',{'error':error,'traceback':traceback.format_exc()})
    finally:
        if p:terminate_owned(p)
        final_gpu=None
        try:
            final_gpu=gpu_snapshot();write(ROOT/'gpu_final.json',final_gpu)
            require(not final_gpu['compute'],'Final GPU compute process remains')
        except BaseException as exc:
            status='failed_no_automatic_retry';error=repr(exc)
            write(ROOT/'error_final_gpu.json',{'error':error,'traceback':traceback.format_exc()})
        write(ROOT/'status.json',{'status':status,'error':error,'completed':completed,'child_exit_count':len(exits),'worker_exit_count':sum(x['label'] in completed for x in exits),'finished_unix':time.time(),'elapsed_seconds':time.monotonic()-mono,'gpu_final_empty':final_gpu is not None and not final_gpu['compute'],**claim})
        files={str(path.relative_to(ROOT)):sha(path) for folder in ('results','logs','gpu','batches','qualification','child_exits') for path in (ROOT/folder).glob('*') if path.is_file()}
        files.update({n:sha(ROOT/n) for n in ('contract.json','approval.json','benchmark.py','status.json','claim.exclusive','claim.json','gpu_initial.json','gpu_final.json','preflight.json','error.json','error_final_gpu.json') if (ROOT/n).exists()})
        files.update({n:sha(ROOT/n) for n in c['operational_files'] if (ROOT/n).is_file()})
        write(ROOT/'terminal_manifest.json',{'status':status,'files':files,'controller_pid':os.getpid(),'controller_os_exit_observed':False,'controller_absence_requires_external_observer':True,'actual_child_exits':exits,'accuracy_evaluated':False,'checkpoint_saved':False,'automatic_retry':False})
    return 0 if status=='complete' else 1


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['qualify','prepare','worker','controller'])
    ap.add_argument('--dataset');ap.add_argument('--arm',choices=[MLP,MAC])
    ap.add_argument('--mode',choices=['train','eval']);ap.add_argument('--rep',type=int)
    ap.add_argument('--device',choices=['cpu','cuda:0'])
    a=ap.parse_args(argv)
    if a.command=='qualify':qualify(a.arm,a.device)
    elif a.command=='prepare':prepare()
    elif a.command=='worker':worker(a.dataset,a.arm,a.mode,a.rep)
    else:return controller()
    return 0


if __name__=='__main__':
    sys.exit(main())
