"""Frozen-weight, train/validation-only location/scale diagnosis.

Every scientific source bundle is imported in a fresh process. No optimizer,
checkpoint selection, held-out loading, or source/model mutation is allowed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/titantpp_width_time_adoption_20261004_v1'
REGISTRIES = {
    4: ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json',
    16: ROOT / 'reports/titantpp_completed_validation_test_20261004_v1/width16/evaluation_registry.json',
}
MANIFEST = ROOT / 'reports/titantpp_independent_evaluation_protocol_20261001_v1/dataset_manifest.json'
DATASETS = ('yellow_trip_hourly', 'intermittent_frozen_5000', 'raf_spare_parts')
SEEDS = (42, 52, 62)
GAP_BOUNDS = (1., 4., 13., 52.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def stratified_sample(q, dt, quantity_bounds, *, seed, cap):
    """SRS within fixed quantity × duration cells; equality stays in lower bin."""
    import numpy as np
    q, dt = np.asarray(q), np.asarray(dt)
    require(q.ndim == 1 and q.shape == dt.shape and len(q), 'Invalid population')
    require(np.isfinite(q).all() and (q >= 0).all() and np.isfinite(dt).all()
            and (dt >= 1).all() and (dt == np.rint(dt)).all(), 'Invalid targets')
    require(isinstance(cap, int) and cap > 0, 'Invalid sample cap')
    qb = np.searchsorted(quantity_bounds, q, side='left')
    gb = np.searchsorted(GAP_BOUNDS, dt, side='left')
    rng = np.random.default_rng(seed)
    ids, weights, cells = [], [], []
    for qi in range(len(quantity_bounds) + 1):
        for gi in range(len(GAP_BOUNDS) + 1):
            candidates = np.flatnonzero((qb == qi) & (gb == gi))
            n = min(len(candidates), cap)
            cells.append({'quantity_bin': qi, 'gap_bin': gi, 'population_n': len(candidates), 'sample_n': n})
            if n:
                chosen = np.sort(rng.choice(candidates, n, replace=False))
                ids.extend(chosen.tolist()); weights.extend([len(candidates) / n] * n)
    order = np.argsort(ids)
    ids = np.asarray(ids, dtype=np.int64)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    require(math.isclose(weights.sum(), len(q), rel_tol=1e-12), 'Weights do not reconstruct population')
    return ids, weights, cells


def stratified_mean_design_se(values, weights, cells, keep=None):
    """Finite-population SRS standard error; not model/seed/generalization uncertainty."""
    import numpy as np
    values=np.asarray(values,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64);cells=np.asarray(cells)
    require(values.ndim==1 and values.shape==weights.shape==cells.shape,'SE shapes differ')
    keep=np.ones(len(values),dtype=bool) if keep is None else np.asarray(keep,dtype=bool)
    require(keep.shape==values.shape and keep.any(),'SE has no population')
    variance=0.; mass=float(weights[keep].sum())
    for cell in np.unique(cells[keep]):
        mask=keep & (cells==cell); n=int(mask.sum()); w=weights[mask]
        require(np.allclose(w,w[0]),'Sampling cell weights differ')
        N=float(w.sum())
        if N>n and n<2:return None
        if N>n:variance += N*N*(1-n/N)*float(np.var(values[mask],ddof=1))/n
    return math.sqrt(max(0.,variance))/mass


def width_parameter_decomposition(actual4, actual16, mu16_sigma4, mu4_sigma16):
    """Pointwise algebra, not an intervention on training or unique attribution."""
    import numpy as np
    a, b, c, d = [np.asarray(v, dtype=np.float64) for v in
                   (actual4, actual16, mu16_sigma4, mu4_sigma16)]
    require(a.shape == b.shape == c.shape == d.shape and np.isfinite([a, b, c, d]).all(), 'Invalid NLL arrays')
    result = {'total_delta': b - a, 'location_at_sigma4': c - a,
              'scale_at_mu4': d - a, 'interaction': b - c - d + a}
    require(np.allclose(result['total_delta'], result['location_at_sigma4'] + result['scale_at_mu4']
                        + result['interaction'], rtol=1e-12, atol=1e-10), 'Decomposition identity failed')
    return result


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / 'contract.json').exists(), 'Do not overwrite a frozen contract')
    registries = {w: read(p) for w, p in REGISTRIES.items()}
    identities = {d['dataset']: d['identities'] for d in read(MANIFEST)['datasets'] if d['dataset'] in DATASETS}
    jobs = []
    for width, reg in registries.items():
        arm = 'titantpp_history_mlp' if width == 4 else 'titantpp_history_mlp_width16'
        rows = [r for r in reg['rows'] if r['model'] == arm and r['dataset'] in DATASETS and r['seed'] in SEEDS]
        require(len(rows) == 9, 'Expected nine selected checkpoints per width')
        for row in rows:
            b = reg['bundles'][row['evaluator_source_bundle']]
            spec = next(d for d in b['datasets'] if d['dataset_id'] == row['dataset'])
            identity = identities[row['dataset']].get('deployed_data', identities[row['dataset']]['data'])
            source = ROOT / b['source_root']
            for rel, expected in b['source_files'].items():
                require(sha(source / rel) == expected, 'Frozen source changed: ' + rel)
            require(sha(ROOT / row['checkpoint_path']) == row['checkpoint_file_sha256'], 'Checkpoint changed')
            require(sha(ROOT / identity['path']) == identity['sha256'], 'Dataset changed')
            jobs.append({'width': width, 'row': row, 'source_root': b['source_root'],
                         'source_files': b['source_files'], 'source_closure_sha256': b['source_closure_sha256'],
                         'dataset_spec': spec, 'data': identity})
    for ds in DATASETS:
        relevant = [j['dataset_spec'] for j in jobs if j['row']['dataset'] == ds]
        for key in ('loader', 'model', 'quantity_boundaries_all_train_rows', 'statistics', 'time_statistics'):
            require(all(s[key] == relevant[0][key] for s in relevant), 'Width comparison contract differs: ' + key)
    write(OUT / 'contract.json', {
        'schema': 'titantpp_width_time_adoption_v1', 'created_unix': time.time(),
        'scope': 'fixed Validation-selected weights; train/validation only',
        'followup_baseline_width': 16, 'preserved_comparator_width': 4,
        'datasets': list(DATASETS), 'deferred_dataset': 'insta_market_basket', 'seeds': list(SEEDS),
        'splits': ['train', 'validation'], 'new_training': False, 'heldout_read': False,
        'sampling': {'kind': 'SRS without replacement in original TRAIN quantity bins × fixed duration bins',
                     'quantity_bounds': 'original frozen TRAIN-derived boundaries',
                     'gap_bounds_in_dataset_units': list(GAP_BOUNDS), 'cap_per_cell': 64,
                     'train_seed': 20261004, 'validation_seed': 20261005,
                     'weights': 'N_cell / n_cell; identical sample across model seeds and widths'},
        'runtime': {'device': 'cpu', 'threads': 2, 'batch_size': 32},
        'registries': {str(w): {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for w, p in REGISTRIES.items()},
        'dataset_manifest_sha256': sha(MANIFEST), 'jobs': jobs,
        'interpretation': 'bounded paired diagnostic, not full-population performance or causal training attribution; hybrid location/scale decomposition includes interaction',
    })
    print(json.dumps({'phase': 'frozen', 'jobs': len(jobs), 'contract_sha256': sha(OUT / 'contract.json')}))


def one(dataset, width, seed):
    start = time.monotonic(); c = read(OUT / 'contract.json')
    require(dataset in DATASETS and width in (4, 16) and seed in SEEDS, 'Outside approved diagnostic')
    j = next(j for j in c['jobs'] if j['width'] == width and j['row']['dataset'] == dataset and j['row']['seed'] == seed)
    row, spec = j['row'], j['dataset_spec']; source = ROOT / j['source_root']
    for rel, expected in j['source_files'].items(): require(sha(source / rel) == expected, 'Frozen source changed: ' + rel)
    require(sha(ROOT / row['checkpoint_path']) == row['checkpoint_file_sha256'], 'Checkpoint changed')
    require(sha(ROOT / j['data']['path']) == j['data']['sha256'], 'Dataset changed')
    sys.dont_write_bytecode = True
    sys.path[:] = [str(source)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
    cache = OUT / 'cache'; cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='', POLARS_MAX_THREADS='2', MPLCONFIGDIR=str(cache), XDG_CACHE_HOME=str(cache))
    import numpy as np
    import polars as pl
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, right_pad_batch
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from data_loader.event_seq_data_module import collate_week_lookback
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    if width == 16:
        from paper.scripts.run_titantpp_history_width import install_hooks
        install_hooks()
        from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    cp = torch.load(ROOT / row['checkpoint_path'], map_location='cpu', weights_only=False)
    validate_checkpoint_route(cp, row['model'])
    require(cp['best_epoch'] == row['selected_epoch'] and cp['evaluation_scope'] == 'validation_only'
            and cp['held_out_test_evaluated'] is False and cp['checkpoint_monitor'] == 'validation_raw_quantity_rmse', 'Selection/scope changed')
    config = {k: v for k, v in spec['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
    model, _ = build_count_aware_model(row['model'], **config, train_log_mean=spec['statistics']['train_log_mean'],
                   train_log_std=spec['statistics']['train_log_std'], max_seq_len=spec['loader']['max_seq_len'])
    model.load_state_dict(cp['model_state_dict'], strict=True); model.eval().requires_grad_(False)
    state = canonical_state_dict_sha256(model.state_dict())
    require(state == row['state_tensor_sha256'], 'Parameter digest differs')
    frame = pl.scan_parquet(ROOT / j['data']['path']).filter(pl.col('chronological_split').is_in(['train','validation'])).collect().sort(['oper_part_no','seq'])
    require(set(frame['chronological_split']) <= {'train','validation'}, 'Held-out rows exposed')
    frame = prepare_count_frame(frame)
    job = OUT / 'runs' / f'{dataset}__width{width}__seed{seed}'; job.mkdir(parents=True,exist_ok=False)
    receipt = {'dataset':dataset,'width':width,'seed':seed,'selected_epoch':row['selected_epoch'],
               'checkpoint_sha256':row['checkpoint_file_sha256'],'state_tensor_sha256':state,
               'source_root':j['source_root'],'source_closure_sha256':j['source_closure_sha256'],
               'runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':2},
               'new_training':False,'heldout_read':False,'splits':{}}
    bounds = np.array(spec['quantity_boundaries_all_train_rows'])
    def forward(dt, mask, qty):
        rd,rq,rm,lengths=right_pad_batch(dt,qty,mask); bi=torch.arange(len(lengths)); target=lengths-1; hist=lengths-2
        hq=rq.clone(); hq[bi,target]=0.; observed=rm.clone(); observed[bi,target]=False
        th,qh=model.encode_task_states(rd,hq,rm,memory_write_mask=observed)
        th=th[bi,hist]; qh=qh[bi,hist]; true_dt=rd[bi,target]; true_q=rq[bi,target]
        mu,sigma,logdt=model._lognormal_time_terms(th,true_dt)
        pred=model.quantity_outputs(qh,true_q)['point_prediction']
        nll=-model.log_observation_dt(th,true_dt)
        return true_dt,true_q,pred,nll,mu,sigma,(logdt-math.log(model.time_scale)-mu)/sigma,lengths-1
    for split in c['splits']:
        loader=make_loader(frame,target_split=split,shuffle=False,generator=None,**{k:spec['loader'][k] for k in ('batch_size','lookback_weeks','max_seq_len')}); data=loader.dataset
        ix=np.asarray(data.index,dtype=np.int64); pi,pos=ix[:,0],ix[:,1]+1
        q=np.fromiter((data.val_lists[p][t] for p,t in zip(pi,pos)),dtype=np.float64)
        dt=np.fromiter((data.dt_lists[p][t] for p,t in zip(pi,pos)),dtype=np.float64)
        seq=np.fromiter((data.seq_lists[p][t] for p,t in zip(pi,pos)),dtype=np.int64)
        require(all(data.split_lists[p][t]==split for p,t in zip(pi,pos)), 'Target split differs')
        h=hashlib.sha256(b'hard_lmm_target_identity_v1\0'+split.encode()+b'\0');h.update(json.dumps([str(p) for p in data.parts],ensure_ascii=False,separators=(',',':')).encode())
        for a in (pi,pos,seq):h.update(a.astype('<i8').tobytes())
        pop={'target_count':len(q),'target_identity_sha256':h.hexdigest(),'target_quantity_sha256':hashlib.sha256(b'hard_lmm_target_quantity_v1\0'+q.astype('<f8').tobytes()).hexdigest()}
        require(pop==spec['inherited_data_identity']['populations'][split], 'Eligible population changed')
        sample,weights,cells=stratified_sample(q,dt,bounds,seed=c['sampling'][split+'_seed'],cap=c['sampling']['cap_per_cell'])
        result=[];causal=False
        with torch.inference_mode():
            for begin in range(0,len(sample),c['runtime']['batch_size']):
                ids=sample[begin:begin+c['runtime']['batch_size']]; _,bd,bm,_,bq=collate_week_lookback([data[int(i)] for i in ids]); values=forward(bd,bm,bq)
                require(np.array_equal(values[1].double().numpy(),q[ids]),'Inference targets differ')
                if begin==0:
                    changed_d,changed_q=bd.clone(),bq.clone();changed_q[:,-1]+=12345.;changed_d[:,-1]=torch.where(changed_d[:,-1]==1.,2.,1.)
                    other=forward(changed_d,bm,changed_q)
                    for k in (2,4,5):require(torch.allclose(values[k],other[k],atol=1e-6,rtol=1e-6),'Target affects prediction/parameters')
                    causal=True
                arrays=[v.double().numpy() for v in values]
                require(all(np.isfinite(a).all() for a in arrays),'Non-finite diagnostic outputs')
                for k,target_id in enumerate(ids):
                    if split=='train': require(all(s=='train' for s in data.split_lists[pi[target_id]][:pos[target_id]+1]),'Non-train context')
                    result.append({'target_index':int(target_id),'split':split,'quantity':float(arrays[1][k]),'dt':float(arrays[0][k]),
                                   'prediction':float(arrays[2][k]),'time_nll':float(arrays[3][k]),'location':float(arrays[4][k]),
                                   'sigma':float(arrays[5][k]),'standardized_log_residual':float(arrays[6][k]),
                                   'history_length':int(arrays[7][k]),'weight':float(weights[begin+k])})
        f=pl.DataFrame(result); path=job/(split+'.parquet');f.write_parquet(path)
        receipt['splits'][split]={'population':pop,'sample_n':len(sample),'sample_indices_sha256':hashlib.sha256(sample.astype('<i8').tobytes()).hexdigest(),
                                 'weights_sha256':hashlib.sha256(weights.astype('<f8').tobytes()).hexdigest(),'cells':cells,
                                 'prediction_file':path.name,'prediction_sha256':sha(path),'target_hidden_invariance':causal}
        print(json.dumps({'phase':'diagnostic_done','dataset':dataset,'width':width,'seed':seed,'split':split,'sample_n':len(sample),'elapsed_seconds':round(time.monotonic()-start,2)}),flush=True)
    require(canonical_state_dict_sha256(model.state_dict())==state,'Evaluation mutated parameters')
    receipt['state_unchanged']=True;receipt['elapsed_seconds']=time.monotonic()-start;write(job/'receipt.json',receipt)


def validation_endpoints():
    """Reuse only SHA/state-bound validation endpoint evidence; never mixed tables."""
    import numpy as np
    import polars as pl
    c=read(OUT/'contract.json'); rows=[]; bindings=[]
    for j in c['jobs']:
        r=j['row']; parent=(ROOT/r['checkpoint_path']).parent
        endpoint=parent/'endpoint_replays.json';history_path=parent/'history.json'
        replay=read(endpoint)['selected'];history=read(history_path)['history']
        require(replay['evaluation_scope']=='validation_only' and replay['held_out_test_evaluated'] is False,
                'Endpoint is outside validation scope')
        require(replay['state_sha256']==r['state_tensor_sha256'] and replay['count']==r['validation_count'], 'Endpoint binding changed')
        finite=[h for h in history if math.isfinite(h['val_qty_rmse'])]
        selected=min(finite,key=lambda h:h['val_qty_rmse'])
        require(selected['epoch']==r['selected_epoch'], 'First raw validation minimum differs')
        require(math.isclose(selected['val_qty_rmse'],replay['qty_rmse'],rel_tol=1e-5,abs_tol=1e-5), 'History/replay metrics differ')
        rows.append({'dataset':r['dataset'],'seed':r['seed'],'width':j['width'],'split':'validation',
                     'selected_epoch':r['selected_epoch'],'count':replay['count'],'qty_rmse':replay['qty_rmse'],
                     'qty_mae':replay['qty_mae'],'tail_qty_rmse':replay['tail']['qty_rmse'],'time_nll':replay['time_nll'],
                     'epoch_training_pass_time_nll':selected['train_time_nll'],
                     'training_pass_interpretation':'parameters change during epoch; not fixed-checkpoint train inference'})
        bindings.append({'dataset':r['dataset'],'seed':r['seed'],'width':j['width'],
                         'checkpoint_sha256':r['checkpoint_file_sha256'],'state_tensor_sha256':r['state_tensor_sha256'],
                         'endpoint_path':str(endpoint.relative_to(ROOT)),'endpoint_sha256':sha(endpoint),
                         'history_path':str(history_path.relative_to(ROOT)),'history_sha256':sha(history_path),
                         'selected_epoch_first_minimum_verified':True})
    pl.DataFrame(rows).write_csv(OUT/'full_validation_metrics.csv')
    summary=[]
    for ds in DATASETS:
        for metric in ('qty_rmse','qty_mae','tail_qty_rmse','time_nll'):
            vals={w:np.array([next(r[metric] for r in rows if r['dataset']==ds and r['width']==w and r['seed']==seed) for seed in SEEDS]) for w in (4,16)}
            summary.append({'dataset':ds,'metric':metric,'width4_mean':float(vals[4].mean()),'width4_sample_sd':float(vals[4].std(ddof=1)),
                            'width16_mean':float(vals[16].mean()),'width16_sample_sd':float(vals[16].std(ddof=1)),
                            'delta_width16_minus4_mean':float((vals[16]-vals[4]).mean()),
                            'width16_improved_seed_count':int((vals[16]<vals[4]).sum()),
                            'worst_seed_delta':float((vals[16]-vals[4]).max()),'seed_count':3})
    pl.DataFrame(summary).write_csv(OUT/'full_validation_seed_summary.csv');write(OUT/'validation_source_bindings.json',{'rows':bindings,'scope':'validation_only','heldout_read':False})


def summarize():
    import numpy as np
    import polars as pl
    import torch
    c=read(OUT/'contract.json')
    probability_module='models/TPPs/positive_integer_time.py'
    require(len({j['source_files'][probability_module] for j in c['jobs']})==1,'Probability module differs between sources')
    source=ROOT/c['jobs'][0]['source_root']
    require(sha(source/probability_module)==c['jobs'][0]['source_files'][probability_module],'Probability source changed')
    sys.path.insert(0,str(source))
    from models.TPPs.positive_integer_time import positive_integer_log_mass
    validation_endpoints()
    output=[]; verification=[]
    for ds in DATASETS:
        spec=next(j['dataset_spec'] for j in c['jobs'] if j['row']['dataset']==ds);bounds=np.array(spec['quantity_boundaries_all_train_rows']);scale=spec['model']['time_scale']
        for seed in SEEDS:
            dirs={w:OUT/'runs'/f'{ds}__width{w}__seed{seed}' for w in (4,16)}; receipts={w:read(p/'receipt.json') for w,p in dirs.items()}
            for split in c['splits']:
                fs={w:pl.read_parquet(p/(split+'.parquet')) for w,p in dirs.items()}
                for w in (4,16):require(sha(dirs[w]/(split+'.parquet'))==receipts[w]['splits'][split]['prediction_sha256'],'Prediction changed')
                a,b=fs[4],fs[16]
                for name in ('target_index','quantity','dt','weight','history_length'):require(np.array_equal(a[name].to_numpy(),b[name].to_numpy()),'Paired target/sample differs')
                require(receipts[4]['splits'][split]['population']==receipts[16]['splits'][split]['population'],'Population differs')
                v=lambda f,col:torch.tensor(f[col].to_numpy(),dtype=torch.float64)
                with torch.inference_mode():
                    crossed1=(-positive_integer_log_mass(v(a,'dt'),v(b,'location'),v(a,'sigma'),time_scale=scale)).numpy()
                    crossed2=(-positive_integer_log_mass(v(a,'dt'),v(a,'location'),v(b,'sigma'),time_scale=scale)).numpy()
                    for f in (a,b):require(np.allclose((-positive_integer_log_mass(v(f,'dt'),v(f,'location'),v(f,'sigma'),time_scale=scale)).numpy(),f['time_nll'].to_numpy(),rtol=1e-10,atol=1e-10),'Independent recorded mass replay differs')
                dec=width_parameter_decomposition(a['time_nll'].to_numpy(),b['time_nll'].to_numpy(),crossed1,crossed2)
                q=a['quantity'].to_numpy();dt=a['dt'].to_numpy();wt=a['weight'].to_numpy()
                groups=[('overall',np.ones(len(a),dtype=bool)),('quantity_body',q<=bounds[-1]),('quantity_tail',q>bounds[-1])]
                gapgroup=np.searchsorted(GAP_BOUNDS,dt,side='left')
                groups += [(f'gap_bin_{g}',gapgroup==g) for g in range(len(GAP_BOUNDS)+1)]
                for label,keep in groups:
                    if not keep.any():continue
                    avg=lambda x:float(np.average(np.asarray(x)[keep],weights=wt[keep]))
                    record={'dataset':ds,'seed':seed,'split':split,'group':label,'sample_n':int(keep.sum()),'represented_n':float(wt[keep].sum())}
                    for width,f in fs.items():
                        err=f['prediction'].to_numpy()-q
                        record.update({f'width{width}_{name}':value for name,value in {
                            'qty_rmse':math.sqrt(avg(err**2)),'qty_mae':avg(abs(err)),'qty_bias':avg(err),
                            'time_nll':avg(f['time_nll'].to_numpy()),'location_mean':avg(f['location'].to_numpy()),
                            'sigma_mean':avg(f['sigma'].to_numpy()),'abs_standardized_log_residual_mean':avg(abs(f['standardized_log_residual'].to_numpy())),
                            'sigma_below_0_1_share':avg(f['sigma'].to_numpy()<.1),
                            'time_nll_over_20_share':avg(f['time_nll'].to_numpy()>20),
                        }.items()})
                    record.update({k:avg(val) for k,val in dec.items()})
                    sampling_cells=np.searchsorted(bounds,q,side='left')*(len(GAP_BOUNDS)+1)+gapgroup
                    for width,f in fs.items():
                        record[f'width{width}_time_nll_design_se']=stratified_mean_design_se(f['time_nll'].to_numpy(),wt,sampling_cells,keep)
                    record['paired_time_delta_design_se']=stratified_mean_design_se(dec['total_delta'],wt,sampling_cells,keep)
                    output.append(record)
                verification.append({'dataset':ds,'seed':seed,'split':split,'paired_targets_weights_history_equal':True,'recorded_mass_float64_replayed':True,'decomposition_identity_passed':True})
    pl.DataFrame(output).write_csv(OUT/'paired_diagnostics.csv')
    write(OUT/'verification.json',{'status':'passed','checks':verification,'heldout_read':False,'new_training':False,'source_scope':'frozen checkpoints/source bundles','contract_sha256':sha(OUT/'contract.json'),'paired_diagnostics_sha256':sha(OUT/'paired_diagnostics.csv')})
    print(json.dumps({'phase':'summarized','rows':len(output),'paired_checks':len(verification)}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','one','summarize'));parser.add_argument('--dataset',choices=DATASETS);parser.add_argument('--width',type=int,choices=(4,16));parser.add_argument('--seed',type=int,choices=SEEDS);args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='one':one(args.dataset,args.width,args.seed)
    else:summarize()


if __name__=='__main__':main()
