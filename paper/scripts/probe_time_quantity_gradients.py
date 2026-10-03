#!/usr/bin/env python3
"""CPU train-only frozen checkpoint gradient probe; never performs optimizer updates."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def population(ds, split):
    import numpy as np
    identity = hashlib.sha256(b'hard_lmm_target_identity_v1\0')
    identity.update(split.encode() + b'\0')
    identity.update(json.dumps([str(x) for x in ds.parts], ensure_ascii=False, separators=(',', ':')).encode())
    for values in [(i for i,j in ds.index), (j+1 for i,j in ds.index), (int(ds.seq_lists[i][j+1]) for i,j in ds.index)]:
        identity.update(np.fromiter(values, dtype='<i8', count=len(ds)).tobytes())
    qty = hashlib.sha256(b'hard_lmm_target_quantity_v1\0')
    qty.update(np.fromiter((float(ds.val_lists[i][j+1]) for i,j in ds.index), dtype='<f8', count=len(ds)).tobytes())
    return {'target_count':len(ds), 'target_identity_sha256':identity.hexdigest(), 'target_quantity_sha256':qty.hexdigest()}

def sample_indices(total, seed, batches, batch_size):
    import numpy as np
    require(all(type(x) is int for x in (total, seed, batches, batch_size)) and seed >= 0 and batches > 0 and batch_size > 0, 'Invalid sample budget')
    require(total >= batches*batch_size, 'Insufficient train targets')
    return np.random.default_rng(seed).choice(total, size=batches*batch_size, replace=False).astype('<i8')

def train_frame(data):
    import polars as pl
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    # Apply the predicate to the scan, before materializing any rows.
    return prepare_count_frame(pl.scan_parquet(data).filter(pl.col('chronological_split')=='train').collect().sort(['oper_part_no','seq']))

def reset_batch_rng(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def joint_parity(model, batch, engine, seed):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import capture_rng_state, restore_rng_state
    _, dts, mask, _, quantities = batch
    rng = capture_rng_state()
    try:
        reset_batch_rng(seed)
        expected = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        expected_grads = torch.autograd.grad(expected['joint_loss'].mean(), tuple(model.parameters()), allow_unused=True)
        reset_batch_rng(seed)
        actual = engine.task_outputs(model, dts, mask, quantities, 'joint')
        actual_grads = torch.autograd.grad(actual['objective_loss'].mean(), tuple(model.parameters()), allow_unused=True)
        require(torch.equal(expected['joint_loss'], actual['objective_loss']), 'Pinned joint objective parity failed')
        require(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(expected_grads,actual_grads,strict=True)), 'Pinned joint gradient parity failed')
    finally:
        restore_rng_state(rng)
    return {'objective_exact': True, 'all_parameter_gradients_exact': True, 'batch': 0}

def run(contract_path, output_dir):
    contract_path, output_dir = Path(contract_path).resolve(), Path(output_dir).resolve()
    c = json.loads(contract_path.read_text())
    require(c['contract_id'] == 'frozen_B_dual_time_quantity_gradient_v1', 'Wrong probe contract')
    require(c['scope'] == {'split':'train', 'device':'cpu', 'optimizer_updates':0, 'held_out_test':False}, 'Invalid probe scope')
    require(c['model_mode']=='train' and type(c['sampling']['dropout_seed']) is int, 'Train mode and explicit dropout seed required')
    require(not output_dir.exists(), 'Refusing to overwrite probe output')
    source = Path(c['source']['root']).resolve()
    revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    require(revision == c['source']['revision'], 'Frozen source revision drift')
    for file, expected in c['source']['files'].items():
        require(digest(source/file) == expected, f'Frozen source file drift: {file}')
    engine_path = ROOT/'paper/scripts/time_quantity_diagnostic.py'
    require(digest(engine_path) == c['engine_sha256'], 'Engine drift')
    require(digest(Path(__file__)) == c['probe_sha256'], 'Probe drift')
    sys.path.insert(0, str(source))
    import numpy as np
    import polars as pl
    import torch
    torch.set_num_threads(c['runtime']['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for k,v in {'python':sys.version.split()[0], 'torch':torch.__version__, 'numpy':np.__version__, 'polars':pl.__version__}.items():
        require(v == c['runtime'][k], f'Runtime drift: {k}')
    spec = importlib.util.spec_from_file_location('experimental_jqt_engine', engine_path)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from paper.scripts.run_matched_frozen_lognormal_duration import build_source_model
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    from torch.utils.data import DataLoader, Subset
    output_dir.mkdir(parents=True)
    save(output_dir/'contract.json', c)
    imported_source={}
    def verify_imports():
        for name,module in tuple(sys.modules.items()):
            if name.split('.')[0] not in {'models','paper','data_loader','simple_lab_test','utils'} or not getattr(module,'__file__',None):
                continue
            path=Path(module.__file__).resolve()
            require(path.is_relative_to(source), f'Import escaped pinned source: {name}')
            relative=str(path.relative_to(source))
            require(relative in c['source']['files'] and digest(path)==c['source']['files'][relative], f'Unpinned imported source: {name}')
            imported_source[name]=relative
    verify_imports()
    result={'status':'running','contract_sha256':digest(contract_path),'model_mode':'train; per-batch RNG reset; seed equality does not imply identical masks across architectures','clipping_scope':'hypothetical coefficient of whole-model joint gradient, no optimizer/clipping update executed','datasets':[], 'optimizer_updates':0, 'held_out_test_evaluated':False}
    save(output_dir/'status.json',result)
    started=time.monotonic()
    for d in c['datasets']:
        begin=time.monotonic()
        data = Path(d['data_path'])
        require(digest(data)==d['data_sha256'], 'Data SHA drift')
        require(digest(d['split_manifest_path'])==d['split_manifest_sha256'],'Split SHA drift')
        # Predicate applied before collect: validation/test are never materialized.
        frame=train_frame(data)
        require(frame['chronological_split'].unique().to_list()==['train'],'Non-train rows admitted')
        loader=make_loader(frame,target_split='train',batch_size=c['sampling']['batch_size'],lookback_weeks=d['lookback_weeks'],max_seq_len=d['max_seq_len'],shuffle=False,generator=torch.Generator().manual_seed(c['sampling']['seed']))
        ds=loader.dataset
        pop=population(ds,'train')
        require(pop==d['population'],'Train population drift')
        indices=sample_indices(len(ds),c['sampling']['seed'],c['sampling']['batches'],c['sampling']['batch_size'])
        index_digest=hashlib.sha256(indices.tobytes()).hexdigest()
        sampled=DataLoader(Subset(ds,indices.tolist()),batch_size=c['sampling']['batch_size'],shuffle=False,num_workers=0,collate_fn=loader.collate_fn,generator=torch.Generator().manual_seed(c['sampling']['seed']))
        batches=list(sampled)
        row={'dataset':d['dataset'],'population':pop,'sample_count':len(indices),'sample_index_sha256':index_digest,'sample_indices':indices.tolist(),'models':{}}
        for role in ['B','candidate']:
            m=d['models'][role]
            path=Path(m['checkpoint'])
            require(digest(path)==m['checkpoint_sha256'],f'{role} checkpoint SHA drift')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            require(payload['model_state_sha256']==m['state_sha256'],'Checkpoint state receipt drift')
            require(payload['backbone']==m['backbone'] and payload['variant']=='count_only_log_regression','Model route drift')
            require(payload['best_epoch']==m['best_epoch'],'Selected epoch drift')
            model=build_source_model(payload,max_seq_len=d['max_seq_len']).cpu().train()
            verify_imports()
            before=canonical_state_dict_sha256(model.state_dict())
            require(before==m['state_sha256'],'Loaded state drift')
            parity=joint_parity(model,batches[0],engine,c['sampling']['dropout_seed'])
            records=[]
            for number,(_,dts,mask,_,quantities) in enumerate(batches):
                reset_batch_rng(c['sampling']['dropout_seed']+number)
                record=engine.gradient_diagnostics(model,dts,mask,quantities,grad_clip=c['grad_clip'])
                record['batch']=number
                record['targets']=int(dts.shape[0])
                records.append(record)
            require(canonical_state_dict_sha256(model.state_dict())==before,'Probe mutated model state')
            require(all(p.grad is None for p in model.parameters()),'Probe populated parameter gradients')
            require(digest(path)==m['checkpoint_sha256'],'Probe mutated checkpoint')
            row['models'][role]={'records':records,'pinned_joint_parity':parity,'state_unchanged':True,'checkpoint_sha256':m['checkpoint_sha256'],'state_sha256':before,'best_epoch':m['best_epoch']}
            del model,payload
            print(json.dumps({'dataset':d['dataset'],'model':role,'batches':len(records),'elapsed_seconds':time.monotonic()-begin}),flush=True)
        row['elapsed_seconds']=time.monotonic()-begin
        save(output_dir/(d['dataset']+'.json'),row)
        result['datasets'].append({k:v for k,v in row.items() if k not in ('sample_indices','models')})
        save(output_dir/'status.json',result)
        del batches,sampled,loader,ds,frame
    result.update(status='passed',elapsed_seconds=time.monotonic()-started)
    save(output_dir/'imported_source.json',imported_source)
    save(output_dir/'status.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(run(args.contract,args.output_dir)),flush=True)
    except Exception as error:
        status_path=args.output_dir/'status.json'
        if status_path.is_file():
            status=json.loads(status_path.read_text())
            if status.get('status')=='running' and status.get('contract_sha256')==digest(args.contract):
                status.update(status='failed',error_type=type(error).__name__,error=str(error))
                save(status_path,status)
        raise

if __name__=='__main__':
    main()
