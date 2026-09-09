#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, statistics, subprocess, sys, time
from pathlib import Path

BASELINE='titantpp'
CANDIDATE='titantpp_hard_memory_value_norm'
BACKBONES=(BASELINE,CANDIDATE)
LENGTHS=(8,64,256)
BATCH_SIZE=128
HIDDEN_DIM=64
WARMUP_STEPS=5
MEASURED_STEPS=15
REPEATS=3
STEP_LIMIT=1.25
PEAK_LIMIT=1.10
PROFILE_CONTRACT='hard_lmm_value_norm_cuda_cost_v1'
COMMON_GRADIENT_KEYS=(
    'encoder.layers.0.attn.qkv.weight',
    'encoder.layers.1.attn.qkv.weight',
    'lmm.mem',
    'quantity_head.weight',
    'v_t.weight',
)

def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)

def finite_tree(v):
    import torch
    if isinstance(v, torch.Tensor): return bool(torch.isfinite(v).all())
    if isinstance(v, dict): return all(finite_tree(x) for x in v.values())
    if isinstance(v, (list,tuple)): return all(finite_tree(x) for x in v)
    if isinstance(v,float): return math.isfinite(v)
    return True

def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        cpu=tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(cpu.dtype).encode()); h.update(str(tuple(cpu.shape)).encode()); h.update(cpu.numpy().tobytes())
    return h.hexdigest()

def positive_ratio(a,b):
    a=float(a); b=float(b)
    if not math.isfinite(a) or not math.isfinite(b) or a<=0 or b<=0: raise ValueError('invalid positive ratio')
    return a/b

def order(repeat):
    if repeat not in range(REPEATS): raise ValueError('invalid repeat')
    return BACKBONES if repeat%2==0 else tuple(reversed(BACKBONES))

def require_5090():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda:0')
    if 'RTX 5090' not in torch.cuda.get_device_name(device): raise RuntimeError(torch.cuda.get_device_name(device))
    return device

def synthetic_batch(length,device):
    import torch
    g=torch.Generator(device='cpu').manual_seed(98173+length)
    dts=torch.randint(1,12,(BATCH_SIZE,length),generator=g).float()
    qty=torch.randint(1,35,(BATCH_SIZE,length),generator=g).float()
    qty[:,-1]=(0.6*qty[:,-2]+0.25*qty[:,-3]+dts[:,-2].remainder(3.0)).round().clamp_min(1.0)
    mask=torch.ones_like(dts,dtype=torch.bool)
    return dts.to(device),qty.to(device),mask.to(device)

def gradient_audit(model,backbone):
    import torch
    p=dict(model.named_parameters())
    shared={}
    for name in COMMON_GRADIENT_KEYS:
        q=p.get(name)
        if q is None or q.grad is None or not bool(torch.isfinite(q.grad).all()): raise ValueError('missing/nonfinite shared gradient '+name)
        n=float(q.grad.detach().norm())
        if not math.isfinite(n) or n<=0: raise ValueError('zero shared gradient '+name)
        shared[name]=n
    out={'status':'passed','shared_gradient_norms':shared}
    if backbone==CANDIDATE:
        q=p.get('lmm.alpha_raw')
        if q is None or q.numel()!=1 or q.grad is None or not bool(torch.isfinite(q.grad).all()): raise ValueError('missing/nonfinite alpha gradient')
        g=float(q.grad.detach())
        if g==0.0: raise ValueError('zero alpha gradient')
        out['alpha_raw_gradient']=g
    return out

def output_with_alpha(model,batch,alpha):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    p=dict(model.named_parameters())['lmm.alpha_raw']
    saved=p.detach().clone()
    training=model.training
    model.eval()
    try:
        with torch.no_grad():
            p.fill_(alpha)
            dts,quantities,mask=batch
            out=target_outputs(model,dts,mask,quantities,lambda_log_qty=1.0)
            return out['pred_qty'].detach().clone()
    finally:
        with torch.no_grad(): p.copy_(saved)
        model.train(training)

def worker(source_root,backbone,length,repeat):
    sys.path.insert(0,str(source_root))
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    if backbone not in BACKBONES or length not in LENGTHS: raise ValueError('grid identity drift')
    device=require_5090(); torch.set_num_threads(1)
    torch.manual_seed(42+repeat); torch.cuda.manual_seed_all(42+repeat)
    torch.backends.cuda.matmul.allow_tf32=False
    model,metadata=build_count_aware_model(backbone,hidden_dim=HIDDEN_DIM,train_log_mean=2.0,max_seq_len=256,quantity_variant='count_only_log_regression',lambda_tail=0.0,time_head_mode='legacy_clamped_rmtpp',time_intercept_limit=300.0)
    params=dict(model.named_parameters())
    if backbone==CANDIDATE:
        alpha=params.get('lmm.alpha_raw')
        if alpha is None or alpha.numel()!=1 or float(alpha.detach())!=0.0: raise ValueError('alpha init contract drift')
    model.to(device).train(); opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    batch=synthetic_batch(length,device); first_grad={}
    def step(inspect=False):
        nonlocal first_grad
        dts,quantities,mask=batch
        out=target_outputs(model,dts,mask,quantities,lambda_log_qty=1.0)
        loss=out['joint_loss'].mean(); opt.zero_grad(set_to_none=True); loss.backward()
        if inspect: first_grad=gradient_audit(model,backbone)
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True); opt.step()
        return loss.detach(),gn.detach()
    for i in range(WARMUP_STEPS):
        if not finite_tree(step(i==0)): raise FloatingPointError('nonfinite warmup')
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    times=[]
    for _ in range(MEASURED_STEPS):
        torch.cuda.synchronize(device); start=time.perf_counter(); observed=step(); torch.cuda.synchronize(device)
        times.append(time.perf_counter()-start)
        if not finite_tree(observed): raise FloatingPointError('nonfinite measured step')
    peak=int(torch.cuda.max_memory_allocated(device))
    if not finite_tree((model.state_dict(),opt.state_dict())): raise FloatingPointError('nonfinite state')
    learning={'status':'passed'}
    if backbone==CANDIDATE:
        alpha=float(dict(model.named_parameters())['lmm.alpha_raw'].detach())
        if not math.isfinite(alpha) or alpha==0.0: raise ValueError('alpha did not learn')
        digest=state_digest(model); learned=output_with_alpha(model,batch,alpha); base=output_with_alpha(model,batch,0.0)
        delta=float((learned-base).abs().max())
        if not math.isfinite(delta) or delta<=0 or state_digest(model)!=digest: raise ValueError('learned alpha path/restoration failed')
        learning={'status':'passed','learned_alpha_raw':alpha,'zero_alpha_quantity_max_abs_delta':delta,'state_digest_restored':True}
    return {
      'status':'passed','profile_contract':PROFILE_CONTRACT,'backbone':backbone,'sequence_length':length,'repeat':repeat,
      'observed_events':length-1,'target_events':1,'batch_size':BATCH_SIZE,'hidden_dim':HIDDEN_DIM,'seed':42+repeat,
      'warmup_steps':WARMUP_STEPS,'measured_steps':MEASURED_STEPS,'step_seconds':times,'median_step_seconds':statistics.median(times),
      'peak_allocated_bytes':peak,'finite_model_optimizer':True,'gradient_audit':first_grad,'learning_path':learning,
      'model_metadata':metadata,'parameter_count':sum(p.numel() for p in model.parameters()),'device_name':torch.cuda.get_device_name(device),
      'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'tf32_matmul':False,'amp_enabled':False,'measurement_dtype':'float32',
      'measurement_scope':'train_forward_backward_clip_AdamW_synchronized_wall_time','model_state_after_profile_sha256':state_digest(model)
    }

def summarize(rows):
    expected={(l,r,b) for l in LENGTHS for r in range(REPEATS) for b in BACKBONES}
    got={(x['sequence_length'],x['repeat'],x['backbone']) for x in rows}
    if len(rows)!=len(expected) or got!=expected: raise ValueError('incomplete cost grid')
    lengths=[]
    for x in rows:
        if x['status']!='passed' or x['profile_contract']!=PROFILE_CONTRACT or len(x['step_seconds'])!=MEASURED_STEPS or not x['finite_model_optimizer']: raise ValueError('invalid worker')
        if x['gradient_audit'].get('status')!='passed' or x['learning_path'].get('status')!='passed': raise ValueError('missing path proof')
        if 'RTX 5090' not in x['device_name'] or x['amp_enabled'] or x['measurement_dtype']!='float32': raise ValueError('runtime drift')
    counts={b:{x['parameter_count'] for x in rows if x['backbone']==b} for b in BACKBONES}
    if any(len(v)!=1 for v in counts.values()) or next(iter(counts[CANDIDATE]))!=next(iter(counts[BASELINE]))+1: raise ValueError('parameter count drift')
    for length in LENGTHS:
        med={}
        for b in BACKBONES:
            xs=[x for x in rows if x['sequence_length']==length and x['backbone']==b]
            med[b]={'step_seconds':statistics.median(x['median_step_seconds'] for x in xs),'peak_allocated_bytes':statistics.median(x['peak_allocated_bytes'] for x in xs)}
        sr=positive_ratio(med[CANDIDATE]['step_seconds'],med[BASELINE]['step_seconds']); mr=positive_ratio(med[CANDIDATE]['peak_allocated_bytes'],med[BASELINE]['peak_allocated_bytes'])
        lengths.append({'sequence_length':length,'models':med,'candidate_over_B_step':sr,'candidate_over_B_peak_allocated':mr,'step_passed':sr<=STEP_LIMIT,'peak_passed':mr<=PEAK_LIMIT})
    return {'status':'passed' if all(x['step_passed'] and x['peak_passed'] for x in lengths) else 'failed','step_limit':STEP_LIMIT,'peak_limit':PEAK_LIMIT,'aggregation':'ratio_of_medians_over_three_independent_paired_repeats','lengths':lengths}

def parent(source_root,output):
    workers=output.parent/(output.stem+'_workers')
    if output.exists() or workers.exists(): raise FileExistsError(output)
    workers.mkdir(parents=True)
    payload={'status':'running','profile_contract':PROFILE_CONTRACT,'source_file_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_revision':'956603f16ca3540e2012a461a494ea1ec905102d','data_scope':'synthetic_only','worker_process_isolation':True,'measurement_orders':[list(order(r)) for r in range(REPEATS)],'rows':[]}
    save_json(output,payload)
    for length in LENGTHS:
        for repeat in range(REPEATS):
            for backbone in order(repeat):
                out=workers/f'L{length}_repeat{repeat}_{backbone}.json'
                cmd=[sys.executable,'-s',str(Path(__file__).resolve()),'--source-root',str(source_root),'--worker','--backbone',backbone,'--sequence-length',str(length),'--repeat',str(repeat),'--output',str(out)]
                cp=subprocess.run(cmd,cwd=source_root,text=True,capture_output=True)
                if cp.returncode: raise RuntimeError({'cmd':cmd,'stdout':cp.stdout,'stderr':cp.stderr})
                row=json.loads(out.read_text()); payload['rows'].append(row); save_json(output,payload)
    payload['cost_gate']=summarize(payload['rows']); payload['status']='passed' if payload['cost_gate']['status']=='passed' else 'failed'; save_json(output,payload)
    if payload['status']!='passed': raise RuntimeError('cost gate failed')
    return payload

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--worker',action='store_true'); ap.add_argument('--backbone',choices=BACKBONES); ap.add_argument('--sequence-length',type=int); ap.add_argument('--repeat',type=int)
    a=ap.parse_args()
    if a.worker: save_json(a.output,worker(a.source_root,a.backbone,a.sequence_length,a.repeat))
    else:
        p=parent(a.source_root,a.output); print(json.dumps(p['cost_gate'],indent=2))
if __name__=='__main__': main()
