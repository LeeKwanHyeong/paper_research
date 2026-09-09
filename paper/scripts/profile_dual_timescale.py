#!/usr/bin/env python3
"""Paired CUDA cost measurements with isolated workers; no benchmark targets."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paper.scripts.run_dual_timescale_campaign import CONTRACT, BACKBONE, read_json, require, save, sha256
from paper.scripts.profile_hard_lmm_causal_qkv import synthetic_batch, tensor_tree_finite


def worker(backbone,length,repeat):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    c=read_json(CONTRACT)['cost_gate']
    require(torch.cuda.is_available() and 'RTX 5090' in torch.cuda.get_device_name(), '5090 required')
    torch.set_num_threads(1)
    torch.manual_seed(42+repeat)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=build_count_aware_model(backbone,hidden_dim=64,train_log_mean=2.,max_seq_len=256,
                                    time_intercept_limit=300.)
    model=model.cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    batch=synthetic_batch(length,torch.device('cuda'))
    def step():
        optimizer.zero_grad(set_to_none=True)
        dt,q,mask=batch
        result=target_outputs(model,dt,mask,q,lambda_log_qty=1.)
        loss=result['joint_loss'].mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    for _ in range(c['warmup_steps']): step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed=[]
    for _ in range(c['measured_steps']):
        torch.cuda.synchronize(); started=time.perf_counter()
        step()
        torch.cuda.synchronize(); elapsed.append(time.perf_counter()-started)
    peak=torch.cuda.max_memory_allocated()
    require(tensor_tree_finite((model.state_dict(),optimizer.state_dict())), 'Nonfinite state')
    gradients={}
    if backbone==BACKBONE:
        # After warmup alpha and the formerly zero-initialized quantity head
        # have moved. Test the actual quantity-only gradient at that state.
        model.eval(); optimizer.zero_grad(set_to_none=True)
        dt,q,mask=batch
        outputs=target_outputs(model,dt,mask,q,lambda_log_qty=1.)
        outputs['log_qty_loss'].mean().backward()
        for name,p in model.named_parameters():
            if name.startswith('transition_memory.'):
                if p.requires_grad:
                    require(p.grad is not None and bool(torch.isfinite(p.grad).all()), f'Missing/invalid gradient: {name}')
                    gradients[name]=float(p.grad.norm())
        require(gradients and all(value>0 for value in gradients.values()), 'Inactive memory learning path')
    return {'backbone':backbone,'length':length,'repeat':repeat,'step_seconds':elapsed,
            'median_step_seconds':statistics.median(elapsed),'peak_allocated_bytes':peak,
            'parameter_count':sum(p.numel() for p in model.parameters()),'quantity_gradients':gradients,
            'finite':True,'device':torch.cuda.get_device_name(),'torch_version':torch.__version__,
            'cuda_version':torch.version.cuda,'python':sys.version}


def summarize(rows,c):
    expected={(b,n,r) for b in ('titantpp',BACKBONE) for n in c['sequence_lengths'] for r in range(c['repeats'])}
    require(len(rows)==len(expected) and {(x['backbone'],x['length'],x['repeat']) for x in rows}==expected,'Incomplete cost grid')
    summaries=[]
    for length in c['sequence_lengths']:
        measures={}
        for backbone in ('titantpp',BACKBONE):
            subset=[x for x in rows if x['length']==length and x['backbone']==backbone]
            for x in subset:
                require(x['finite'] and len(x['step_seconds'])==c['measured_steps'], 'Invalid cost row')
                require(all(t>0 and __import__('math').isfinite(t) for t in x['step_seconds']), 'Invalid timing')
            measures[backbone]={key:statistics.median(x[key] for x in subset) for key in ('median_step_seconds','peak_allocated_bytes','parameter_count')}
        ratios={key:measures[BACKBONE][key]/measures['titantpp'][key] for key in measures[BACKBONE]}
        passed=(ratios['median_step_seconds']<=c['training_step_ratio_max'] and
                ratios['peak_allocated_bytes']<=c['peak_allocated_memory_ratio_max'] and
                ratios['parameter_count']<=c['parameter_ratio_max'])
        summaries.append({'length':length,'models':measures,'ratios':ratios,'passed':passed})
    return {'status':'passed' if all(x['passed'] for x in summaries) else 'failed',
            'lengths':summaries,'rows':rows,'contract_sha256':sha256(CONTRACT)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['titantpp',BACKBONE])
    p.add_argument('--length',type=int)
    p.add_argument('--repeat',type=int)
    a=p.parse_args()
    if a.worker:
        save(a.output,worker(a.worker,a.length,a.repeat));return
    c=read_json(CONTRACT)['cost_gate']; rows=[]
    for length in c['sequence_lengths']:
        for repeat in range(c['repeats']):
            order=('titantpp',BACKBONE) if repeat%2==0 else (BACKBONE,'titantpp')
            for b in order:
                path=a.output.parent/'cost_workers'/f'{length}_{repeat}_{b}.json'
                subprocess.run([sys.executable,'-s',__file__,'--output',str(path),'--worker',b,
                                '--length',str(length),'--repeat',str(repeat)],cwd=ROOT,check=True)
                rows.append(read_json(path))
    save(a.output,summarize(rows,c))


if __name__=='__main__':main()
