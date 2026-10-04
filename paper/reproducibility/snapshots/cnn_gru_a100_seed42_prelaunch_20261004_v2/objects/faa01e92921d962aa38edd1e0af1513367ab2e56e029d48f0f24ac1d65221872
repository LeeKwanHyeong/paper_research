"""Synthetic CPU checks/costs; native calls are owned by the approved supervisor."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import io
import json
import statistics
import time
from pathlib import Path

from paper.scripts import local_detail_benchmark_contract as proposal
from paper.scripts.observed_slot_parallel_common import require, write_json


def qualify(*, device="cpu", batch_size=4, lengths=(8,20), budget=lambda:None, seed=42):
    import torch
    from paper.scripts import run_local_detail_benchmark as runner
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    native = device == "cuda:0"
    require(device in ("cpu","cuda:0"), "Explicit CPU or qualified CUDA device required")
    require((not native) or (batch_size==128 and tuple(lengths)==(64,256)), "Native workload changed")
    previous_threads=torch.get_num_threads()
    if not native: torch.set_num_threads(1)
    rows=[]; total_updates=0; configs={}; parameters={}
    try:
        for length in lengths:
            data=deepcopy(proposal.protocol()["datasets"][2 if length<=64 else 1])
            data["loader"]["max_seq_len"]=length
            count=batch_size*length
            dts=(torch.arange(count,device=device).reshape(batch_size,length)%7+1).float()
            qty=(torch.arange(count,device=device).reshape(batch_size,length)%19+1).float()
            mask=torch.ones_like(dts,dtype=torch.bool)
            mask[::2,-2:]=False;dts[~mask]=0;qty[~mask]=0
            measurements={}; original_output=None
            for arm in proposal.ARMS:
                budget();torch.manual_seed(seed)
                model,meta=runner.build_model(data,arm);model.to(device).eval()
                configs[arm]=meta["time_head"]
                parameters[arm]=sum(p.numel() for p in model.parameters() if p.requires_grad)
                captured=[]; original_log=model.log_observation_dt
                def capture(hidden,target):
                    captured.append(hidden.detach().clone());return original_log(hidden,target)
                model.log_observation_dt=capture
                with torch.no_grad():
                    out=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                    changed_dt=dts.clone();changed_qty=qty.clone()
                    target=mask.sum(1)-1;idx=torch.arange(batch_size,device=device)
                    changed_dt[idx,target]=2;changed_qty[idx,target]+=100
                    changed_dt[~mask]=999;changed_qty[~mask]=999
                    changed=target_outputs(model,changed_dt,mask,changed_qty,lambda_log_qty=1.)
                    require(torch.equal(captured[0],captured[1]) and torch.equal(out["pred_qty"],changed["pred_qty"]),
                            "Target or padding leaked into prediction")
                model.log_observation_dt=original_log
                if arm=="titantpp":
                    original_output={k:out[k].detach().clone() for k in ("pred_qty","time_loss","joint_loss")}
                if arm=="titantpp_local_detail":
                    require(all(torch.equal(out[k],v) for k,v in original_output.items()),"B/local initial outputs differ")
                require(meta["time_head"]["observation_likelihood"]==data["model"]["time_observation_contract"],
                        "Shared time head observation changed")
                optimizer=build_optimizer(model,lr=.001,time_head_lr_multiplier=1.)
                model.train();times=[]
                if native:torch.cuda.reset_peak_memory_stats()
                for step in range(5):
                    budget()
                    if native:torch.cuda.synchronize()
                    start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    value=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)["joint_loss"].mean()
                    require(torch.isfinite(value).item(),"Nonfinite synthetic objective")
                    value.backward()
                    require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                            "Nonfinite synthetic gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    if native:torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    if step>=2:times.append(elapsed)
                    total_updates+=1
                model.eval()
                with torch.no_grad():before_restore=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                saved=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},saved);saved.seek(0)
                payload=torch.load(saved,map_location=device,weights_only=False)
                with torch.no_grad():
                    next(model.parameters()).add_(1.)
                optimizer.param_groups[0]['lr']=.123
                model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
                model.eval()
                with torch.no_grad():restored=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                require(all(torch.equal(before_restore[k],restored[k]) for k in ('pred_qty','time_loss','joint_loss')),
                        "Restored model predictions changed")
                require(optimizer.param_groups[0]['lr']==payload['optimizer']['param_groups'][0]['lr'],
                        "Optimizer state restore failed")
                measurements[arm]={"parameters":parameters[arm],"step_seconds":times,
                    "median_step_seconds":statistics.median(times),
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated() if native else None}
                del optimizer,model,payload,saved,out,changed,restored,before_restore,value;gc.collect()
                if native:torch.cuda.empty_cache()
            rows.append({"length":length,"batch_size":batch_size,"measurements":measurements})
    finally:
        if not native:torch.set_num_threads(previous_threads)
    return {"status":"passed" if native else "cpu_observation_complete","device":device,"seed":seed,
        "synthetic_optimizer_updates":total_updates,"real_data_loaded":False,"held_out_evaluated":False,
        "checks":{"finite_joint_updates":True,"B_local_initial_output_equal":True,"shared_head_contract":True,
                  "causality":True,"state_restore":True},"costs":rows,
        "interpretation":"CPU step times are not GPU epoch estimates" if not native else "native synthetic cost, not measured real-data epoch time"}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    write_json(a.output,qualify(),exclusive=True)


if __name__=='__main__':main()
