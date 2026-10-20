"""MAC-only model construction and owned, bounded execution support.

The shared mathematical model and trainer remain byte-identical to the parent.
The builder override is scoped to one fresh worker and always restored.
"""
from contextlib import contextmanager
from pathlib import Path
import gc
import math
import os
import time
from paper.scripts.titans_mac_observed_time_contract import ARM, ROLE, common, require
from paper.scripts import run_local_detail_benchmark as shared_runner


def configure_dynamo():
    import torch._dynamo.config as cfg
    from models.Titan.common import titans_mac_optimized as optimized
    cfg.recompile_limit=64
    cfg.accumulated_recompile_limit=512
    cfg.suppress_errors=False
    require(optimized._COMPILED_DIAGNOSTIC_SCAN is not None and optimized._COMPILED_STATE_ONLY_SCAN is not None,
        'Required compiled scan unavailable; no eager fallback')


def factory(backbone, **kwargs):
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from models.Titan.common.titans_mac_optimized import apply_titantpp_mac_semantic_optimization
    require(backbone==ARM and kwargs.get('titans_memory_gradient_clip')==1., 'MAC clip1 builder required')
    model,metadata=build_count_aware_model(backbone,**kwargs)
    model=apply_titantpp_mac_semantic_optimization(model)
    require(model.titans_mac_encoder.neural_memory.gradient_max_norm==1., 'Inner clip changed')
    return model,metadata


def build_model(data,backbone=ARM):
    config={k:v for k,v in data['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
    return factory(backbone,**config,titans_memory_gradient_clip=1.,
        train_log_mean=data['statistics']['train_log_mean'],train_log_std=data['statistics']['train_log_std'],
        max_seq_len=data['loader']['max_seq_len'])


def initialization(data,seed):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model,_=build_model(data)
        config={k:v for k,v in data['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
        torch.manual_seed(seed);baseline,_=build_count_aware_model('titantpp',**config,
            train_log_mean=data['statistics']['train_log_mean'],train_log_std=data['statistics']['train_log_std'],
            max_seq_len=data['loader']['max_seq_len'])
        keys=('b_t','w_raw','v_t.weight','time_scale_weight.weight','quantity_head.weight','quantity_head.bias')
        require(all(torch.equal(model.state_dict()[k],baseline.state_dict()[k]) for k in keys), 'Common head initialization changed')
        return {ARM:canonical_state_dict_sha256(model.state_dict())}


def training_args(c,data,output,arm):
    require(arm==ARM, 'Only MAC fit authorized')
    args=shared_runner.base.training_args({**c,'model_role':ROLE},data,output)
    args.epochs=300;args.min_epochs=40;args.early_stopping_patience=40
    args.titans_memory_gradient_clip=1.
    args.execution_role='fresh_titans_mac_observed_time_validation'
    # Resume identity binds this exact copied source, not merely the historical Git revision.
    args.source_revision=c['source']['files_sha256']
    return args


def first_epoch_gate(payload, data, args):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(payload['epoch']==1 and len(payload['history'])==1, 'First epoch gate identity')
    row=payload['history'][0]
    require(row['train_all_finite'] and all(math.isfinite(row[k]) for k in
        ('val_qty_rmse','val_qty_mae','val_time_nll','val_joint_objective')), 'Nonfinite first epoch')
    require(row['train_batch_count']==data['steps_per_epoch'] and
        row['train_event_count']==data['inherited_data_identity']['populations']['train']['target_count'], 'First epoch exposure')
    require(payload['backbone']==ARM and payload['checkpoint_monitor']=='validation_raw_quantity_rmse' and
        payload['evaluation_scope']=='validation_only' and payload['held_out_test_evaluated'] is False and
        payload['source_revision']==args.source_revision and
        payload['model_state_sha256']==canonical_state_dict_sha256(payload['model_state_dict']), 'First epoch checkpoint identity')
    identity=payload['resume_identity']['arguments']
    require(identity['titans_memory_gradient_clip']==1. and identity['model_role']==ROLE, 'First epoch clipping/role')
    return {'status':'passed','epoch':1,'global_steps':data['steps_per_epoch'],
        'model_state_sha256':payload['model_state_sha256'], 'partial_updates_not_counted_as_extra_fit':True}


@contextmanager
def mac_training(training,data,args,receipt):
    old_builder,old_save=training.build_model,training.atomic_torch_save
    def save(payload,path):
        if Path(path).name=='last_epoch_state.pt' and payload['epoch']==1:
            result=first_epoch_gate(payload,data,args)
            # audited_training checks true optimizer counters and train/validation counts before this call.
            common.write_json(receipt,result,exclusive=True)
        old_save(payload,path)
    training.build_model,training.atomic_torch_save=factory,save
    try:
        yield
    finally:
        training.build_model,training.atomic_torch_save=old_builder,old_save


class Budget:
    def __init__(self,c,permit,host,owner_pid,output):
        self.contract,self.permit,self.spec=c,permit,c['hosts'][host]
        self.owner_pid,self.output,self.last_scan=owner_pid,Path(self.spec['root']),-math.inf
        self.deadline=time.monotonic()+permit['deadline_unix']-time.time()
    def __call__(self):
        require(os.getppid()==self.owner_pid,'Owned supervisor exited')
        require(time.time()<self.permit['deadline_unix'] and time.monotonic()<self.deadline,'Host/common deadline reached')
        if time.monotonic()-self.last_scan>=10:
            shared_runner.shared.check_storage(self.output,self.contract['limits'])
            common.write_json(self.output/'gpu_observation.json',{'at_unix':time.time(),
                'other_pids':sorted(common.gpu_pids(self.spec)-{os.getpid()}),'policy':'observe_without_stopping'})
            self.last_scan=time.monotonic()


class ArmBudget(shared_runner.ArmBudget):
    def __init__(self,c,host,dataset,arm,outer,seed=42):
        self.outer,self.limit=outer,c['limits']['per_dataset_arm_wall_seconds'][dataset]
        self.started,self.monotonic_started=time.time(),time.monotonic()
        self.path=Path(c['hosts'][host]['output_dir'])/'active_arm.json'
        self.record={'status':'active','contract_sha256':common.sha_json(c),'host':host,
            'dataset':dataset,'backbone':arm,'seed':seed,'started_at_unix':self.started,
            'deadline_unix':self.started+self.limit,'started_monotonic':self.monotonic_started,
            'deadline_monotonic':self.monotonic_started+self.limit,'limit_seconds':self.limit}
        common.write_json(self.path,self.record)
        self()


def verify_start_memory(c,host,allowed_pids=()):
    import csv
    import subprocess
    spec=c['hosts'][host]
    inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    rows=list(csv.reader(inventory.splitlines()))
    matches=[r for r in rows if r[0].strip()==spec['gpu_uuid']]
    require(len(matches)==1 and int(matches[0][1])>=c['limits']['minimum_free_vram_mib'],'Insufficient free VRAM')
    require(not(common.gpu_pids(spec)-set(allowed_pids)), 'Foreign CUDA process; wait before launch')


def qualify(c,host,*,device='cpu',batch_size=2,budget=lambda:None,cpu_length=34):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    native=device=='cuda:0'
    require(device in ('cpu','cuda:0') and (not native or batch_size==128), 'Qualification device/batch')
    configure_dynamo(); rows=[]; total=0
    datasets=[d for d in c['datasets'] if d['dataset_id'] in c['hosts'][host]['assigned_datasets']]
    for d in datasets:
        budget(); torch.manual_seed(42)
        model,meta=build_model(d);model.to(device).eval()
        length=d['loader']['max_seq_len'] if native else min(cpu_length,d['loader']['max_seq_len'])
        dt=torch.ones(batch_size,length,device=device);qty=(torch.arange(batch_size*length,device=device).reshape(batch_size,length)%9+1).float()
        mask=torch.ones_like(dt,dtype=torch.bool);mask[-1,max(2,length//2):]=False
        initial=canonical_state_dict_sha256(model.state_dict())
        if native: torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            before=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
            changed_qty=qty.clone();changed_dt=dt.clone();idx=torch.arange(batch_size,device=device);target=mask.sum(1)-1
            changed_qty[idx,target]+=100;changed_qty[~mask]=999;changed_dt[idx,target]=2;changed_dt[~mask]=999
            captured=[];fn=model.log_observation_dt
            def capture(h,t): captured.append(h.clone());return fn(h,t)
            model.log_observation_dt=capture
            target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
            changed=target_outputs(model,changed_dt,mask,changed_qty,lambda_log_qty=1.)
            model.log_observation_dt=fn
            require(torch.equal(captured[0],captured[1]) and torch.equal(before['pred_qty'],changed['pred_qty']), 'Target/padding leak')
            again=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
            require(all(torch.equal(before[k],again[k]) for k in ('pred_qty','time_loss')) and
                initial==canonical_state_dict_sha256(model.state_dict()), 'State persisted across forwards')
        optimizer=build_optimizer(model,lr=.001,time_head_lr_multiplier=1.)
        model.train();times=[];updates=60//len(datasets)
        for step in range(updates):
            budget(); start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            value=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)['joint_loss'].mean()
            require(torch.isfinite(value).item(),'Nonfinite synthetic loss');value.backward()
            require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),'Nonfinite synthetic gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if native:torch.cuda.synchronize()
            times.append(time.perf_counter()-start);total+=1
        import copy
        model.eval();saved=copy.deepcopy(model.state_dict());saved_optimizer=copy.deepcopy(optimizer.state_dict())
        with torch.no_grad():
            ref=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
            next(model.parameters()).add_(1.)
            model.load_state_dict(saved,strict=True);optimizer.load_state_dict(saved_optimizer)
            restored=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
            require(all(torch.equal(ref[k],restored[k]) for k in ('pred_qty','time_loss','joint_loss')),'Restore failed')
        rows.append({'dataset':d['dataset_id'],'batch_size':batch_size,'length':length,'updates':updates,
            'step_seconds':times,'parameters':sum(p.numel() for p in model.parameters()),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if native else None})
        del model,optimizer,value,saved,saved_optimizer,ref,restored,before,again,changed,captured,fn
        gc.collect()
        if native:torch.cuda.empty_cache()
    initializations={d['dataset_id']:{str(s):initialization(d,s) for s in (42,52,62)} for d in datasets}
    return {'status':'passed' if native else 'cpu_observation_complete','device':device,'host':host,
        'synthetic_optimizer_updates':total,'real_data_loaded':False,'held_out_evaluated':False,
        'checks':dict.fromkeys(('finite_updates','causality','padding','fresh_state','restore','common_head'),True),
        'costs':rows,'initialization':initializations,'timing_scope':'synthetic; not real epoch ETA'}
