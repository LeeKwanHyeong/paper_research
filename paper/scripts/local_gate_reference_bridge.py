#!/usr/bin/env python3
"""Same-host CPU-only source bridge and admitted train/validation preflight.

The probe runs in fresh processes against each source tree. It never loads a
checkpoint or held-out rows, initializes CUDA, or changes an old experiment.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random,subprocess,sys
from pathlib import Path


def digest_tree(value):
    import numpy as np
    import torch
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):h.update(str(key).encode());walk(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:walk(v)
        else:h.update(repr(x).encode())
    walk(value);return h.hexdigest()


def probe(source_root,contract):
    sys.path.insert(0,str(source_root))
    import numpy as np
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    results={}
    for data in contract['datasets']:
        rows={}
        config={k:v for k,v in data['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
        for arm in ('titantpp','titantpp_local_detail'):
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            model,meta=build_count_aware_model(arm,**config,train_log_mean=data['statistics']['train_log_mean'],
                train_log_std=data['statistics']['train_log_std'],max_seq_len=data['loader']['max_seq_len'])
            initial=canonical_state_dict_sha256(model.state_dict());rng_initial=digest_tree(torch.get_rng_state())
            model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            x=torch.arange(3*130,dtype=torch.float32).reshape(3,130)
            dt=(x%29)+1;qty=(x*7%201)+1;mask=torch.arange(130)[None,:]<torch.tensor([130,65,32])[:,None]
            dt=dt*mask;qty=qty*mask;steps=[]
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                out=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
                out['joint_loss'].mean().backward()
                gradient=digest_tree({n:p.grad for n,p in model.named_parameters()})
                output=digest_tree(out)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                steps.append({'outputs':output,'preclip_gradients':gradient,'model':canonical_state_dict_sha256(model.state_dict()),
                    'optimizer':digest_tree(optimizer.state_dict()),'rng':digest_tree((torch.get_rng_state(),np.random.get_state(),random.getstate()))})
            rows[arm]={'initial_state_sha256':initial,'initial_rng':rng_initial,'metadata':meta,'steps':steps,'optimizer_updates':3}
        results[data['dataset_id']]=rows
    return results


def preflight(contract_path,host):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from paper.scripts import run_local_gate_execution as r
    c=r.read(contract_path);r.source_and_host(c,host);spec=c['hosts'][host];dest=Path(spec['root'])
    # The old CPU source probe is allowed while its separate CUDA worker runs.
    paths={'old':Path(r.proposal.parent()['hosts'][host]['source_root']),'new':root}
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    env.pop('PYTHONPATH',None)
    for label,source in paths.items():
        subprocess.run([spec['python'],str(Path(__file__).resolve()),'probe','--source-root',str(source),
            '--contract',str(Path(contract_path).resolve()),'--output',str(dest/f'bridge_{label}.json')],
            cwd=source,env=env,check=True,timeout=240)
    old,new=r.read(dest/'bridge_old.json'),r.read(dest/'bridge_new.json')
    r.require(old==new,'Original/current source CPU bridge differs')
    receipt={'status':'passed','host':host,'contract_sha256':r.common.sha_json(c),'source_files_sha256':c['source']['files_sha256'],
        'old_sha256':r.common.sha_file(dest/'bridge_old.json'),'new_sha256':r.common.sha_file(dest/'bridge_new.json'),
        'cpu_only':True,'synthetic_updates_per_source':18,'held_out_loaded':False}
    r.common.write_json(dest/'bridge_receipt.json',receipt,exclusive=True)
    r.verify_reference_bridge(c,host)
    rows={}
    for dataset_id in spec['assigned_datasets']:
        data=next(d for d in c['datasets'] if d['dataset_id']==dataset_id)
        frame,meta=r.prepare_admitted_data(data);interface=r.time_interface(data,frame,c);initial=r.initialization(data)
        arms,exposure=r.load_references(c,host,data,interface,initial,dest)
        rows[dataset_id]={'initialization':initial,'data':meta,'reference_arms':list(arms),
            'reference_exposure_equal':exposure['titantpp']==exposure[r.proposal.UNGATED]}
    r.common.write_json(dest/'preflight_receipt.json',{'status':'passed','host':host,'contract_sha256':r.common.sha_json(c),
        'source_files_sha256':c['source']['files_sha256'],'datasets':rows,'held_out_loaded':False,'cuda_initialized':False},exclusive=True)
    print(json.dumps({'status':'passed','host':host,'datasets':list(rows)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('probe','preflight'))
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--source-root',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--host',choices=('5080','5090'))
    a=parser.parse_args()
    if a.command=='probe':
        assert a.source_root and a.output
        result=probe(a.source_root.resolve(),json.loads(a.contract.read_text()))
        with a.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    else:
        assert a.host;preflight(a.contract,a.host)
if __name__=='__main__':main()
