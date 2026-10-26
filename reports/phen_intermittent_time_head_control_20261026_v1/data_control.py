"""Hash-bound immutable feature preparation and head-only experiment driver."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
P=HERE.parent/'phen_intermittent_full_observations_20261026_v1'
CONTRACT_SHA='8bc478500eb57ff402e1c7a65b779c19ef0b26691c16c975d479a59ddb0d03b7'
FITTER_SHA='e19c5d059cf4877493fcd725051b9736ba2c414bf5131b22526cbdf1b39f8857'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def rec(p):
    p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def verify(r):
    p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    return p
def save(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def contract():
    assert sha(HERE/'SCIENCE_SPEC.json')==CONTRACT_SHA
    s=json.loads((HERE/'SCIENCE_SPEC.json').read_text())
    for r in s['bindings'].values():verify(r)
    return s
def fitter():
    assert sha(HERE/'head/fitter.py')==FITTER_SHA
    spec=importlib.util.spec_from_file_location('headcontrol_fitter',HERE/'head/fitter.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod
def load_checkpoint(seed,kind,s):
    p=P/f'execution_v{6 if seed==42 else 8}/bundle/checkpoints/seed{seed}/{kind}.pt'
    expected=next(x for x in s['checkpoints'] if (x['seed'],x['checkpoint'])==(seed,kind))
    assert sha(p)==expected['sha256']
    permitted={ 'numpy.dtype','numpy.ndarray','numpy._core.multiarray._reconstruct'}
    unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(p));assert unsafe<=permitted
    # Last checkpoints include an original NumPy uint32 RNG state. Only these
    # known NumPy constructors are allowlisted; unrestricted pickle is not used.
    with torch.serialization.safe_globals([np.dtype,np.ndarray,np._core.multiarray._reconstruct,np.dtypes.UInt32DType]):
        cp=torch.load(p,map_location='cpu',weights_only=True)
    state=cp['model_state_dict'];assert isinstance(state,dict)
    return p,state
def check_parity(actual,expected,atol,rtol):
    actual=np.asarray(actual);expected=np.asarray(expected)
    assert actual.shape==expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    delta=np.abs(actual-expected);limit=atol+rtol*np.abs(expected)
    report=dict(rows=len(delta),max_abs_error=float(delta.max()),violations=int((delta>limit).sum()),atol=atol,rtol=rtol)
    assert report['violations']==0,report
    return report
def write_cohort(seed,kind,split,chunks,source_refs,s):
    out=HERE/'features'/f'seed{seed}_{kind}_{split}'
    assert not out.exists(),'Existing feature preparation preserved'
    out.mkdir(parents=True)
    try:
        arrays=[];frames=[];inputs=[]
        for c in chunks:
            hp=verify(c['head_hidden']);op=verify(c['observations'])
            inputs.extend([c['head_hidden'],c['observations']])
            with np.load(hp,allow_pickle=False) as n:
                assert set(n.files)=={'head_hidden','target_index'}
                x=n['head_hidden'];ids=n['target_index']
                assert x.dtype==np.float32 and x.shape==(len(ids),64) and np.isfinite(x).all()
                f=pd.read_parquet(op);assert np.array_equal(ids,f.target_index)
                arrays.append(x);frames.append(f)
        obs=pd.concat(frames,ignore_index=True);x=np.concatenate(arrays)
        assert not obs.target_index.duplicated().any()
        order=np.argsort(obs.target_index.to_numpy());obs=obs.iloc[order].reset_index(drop=True);x=x[order]
        meta=pd.read_parquet(s['bindings']['metadata']['path']);meta=meta.loc[meta.split.eq(split)].sort_values('target_index').reset_index(drop=True)
        for key in ['target_index','target_seq','history_length','target_gap','raw_quantity']:
            assert np.array_equal(obs[key],meta[key]),key
        assert np.array_equal(obs.entity_id.astype(str),meta.entity_id.astype(str))
        cp,state=load_checkpoint(seed,kind,s);fit=fitter()
        head=fit.validate_head_state(state)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        predicted=fit.predict(head,x,meta.target_gap.to_numpy())
        timegate=check_parity(predicted['time_nll'],obs.time_nll.to_numpy(),**s['parity']['time_nll'])
        with torch.no_grad():
            q=torch.expm1(F.softplus(F.linear(torch.from_numpy(x),state['quantity_head.weight'],state['quantity_head.bias']).squeeze(-1))).numpy()
        qgate=check_parity(q,obs.pred_quantity.to_numpy(),**s['parity']['quantity'])
        np.save(out/'hidden.npy',x,allow_pickle=False)
        np.save(out/'target_index.npy',meta.target_index.to_numpy(),allow_pickle=False)
        obs[['target_index','pred_quantity','time_nll']].to_parquet(out/'native_predictions.parquet',index=False)
        torch.save(head,out/'original_time_head.pt')
        torch.save({k:state[k].detach().cpu() for k in ['quantity_head.weight','quantity_head.bias']},out/'original_quantity_head.pt')
        assert sha(cp)==next(c['sha256'] for c in s['checkpoints'] if(c['seed'],c['checkpoint'])==(seed,kind))
        save(out/'COMPLETE.json',dict(seed=seed,checkpoint=kind,split=split,rows=len(x),contract_sha256=CONTRACT_SHA,
             checkpoint_file=rec(cp),head_state_sha256=fit.state_sha(head),inputs=inputs,source_refs=source_refs,
             epoch0_time_gate=timegate,epoch0_quantity_gate=qgate,
             files={p.name:rec(p) for p in out.iterdir() if p.is_file()},training_steps=0))
        print(json.dumps(dict(status='features_verified',seed=seed,checkpoint=kind,split=split,rows=len(x),time_gate=timegate,quantity_gate=qgate)),flush=True)
    except Exception:
        save(out/'FAILURE.json',dict(traceback=traceback.format_exc(),training_steps=0));raise
def prepare_best(seed,s):
    f=json.loads(verify(s['bindings']['collection']).read_text())
    c=next(c for c in f['cohorts'] if(c['seed'],c['checkpoint'],c['split'])==(seed,'best','train'))
    assert c['hidden_rows']==c['rows']==393824 and c['actual_child_exit']==0
    for key in ['manifest','job_complete','child_exit','qa']:verify(c[key])
    write_cohort(seed,'best','train',c['chunks'],{k:c[k] for k in ['manifest','job_complete','child_exit','qa']},s)
def audit_math(seed,kind,split,s):
    """Verify same-input native mu/sigma arithmetic without fitting or changing features."""
    directory=HERE/'features'/f'seed{seed}_{kind}_{split}'
    result_path=directory/'NATIVE_MATH_QA.json'
    assert not result_path.exists(),'Existing native-math audit preserved'
    feature_receipt=rec(directory/'COMPLETE.json')
    try:
        c=json.loads(verify(feature_receipt).read_text())
        expected_rows=s['split']['full_train_rows' if split=='train' else 'validation_rows']
        assert c['contract_sha256']==CONTRACT_SHA
        assert (c['seed'],c['checkpoint'],c['split'],c['rows'])==(seed,kind,split,expected_rows)
        for r in c['files'].values():verify(r)
        for r in c['inputs']:verify(r)
        helper=verify(s['bindings']['native_probability_helper'])
        cp,state=load_checkpoint(seed,kind,s);fit=fitter()
        x=np.load(directory/'hidden.npy',allow_pickle=False)
        ids=np.load(directory/'target_index.npy',allow_pickle=False)
        assert x.dtype==np.float32 and x.shape==(expected_rows,64) and np.isfinite(x).all()
        meta=pd.read_parquet(verify(s['bindings']['metadata'])).loc[lambda v:v.split.eq(split)].sort_values('target_index')
        assert np.array_equal(ids,meta.target_index.to_numpy())
        initial=fit.validate_head_state(state)
        saved=torch.load(directory/'original_time_head.pt',weights_only=True,map_location='cpu')
        assert fit.state_sha(saved)==c['head_state_sha256']==fit.state_sha(initial)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        batch_size=s['fit_config']['evaluation_batch_size'];assert batch_size==8192
        predicted=fit.predict(initial,x,meta.target_gap.to_numpy(),batch_size=batch_size)
        direct_mu=[];direct_sigma=[]
        # Independent expression from original checkpoint tensors, using the same
        # batch size so BLAS accumulation order cannot create false discrepancies.
        with torch.no_grad():
            for start in range(0,len(x),batch_size):
                h=torch.from_numpy(x[start:start+batch_size]).to(torch.float64)
                mu=F.linear(h,state['v_t.weight'].to(torch.float64)).squeeze(-1)+state['b_t'].to(torch.float64)
                raw=F.linear(h,state['time_scale_weight.weight'].to(torch.float64)).squeeze(-1)+state['w_raw'].to(torch.float64)
                sigma=s['head']['sigma_min']+F.softplus(raw)
                direct_mu.append(mu.numpy());direct_sigma.append(sigma.numpy())
        direct={'mu':np.concatenate(direct_mu),'sigma':np.concatenate(direct_sigma)}
        exact={}
        for name in ['mu','sigma']:
            equal=np.array_equal(predicted[name],direct[name]);assert equal,'Independent native '+name+' differs'
            exact[name]=dict(exact_equal=equal,max_abs_error=float(np.abs(predicted[name]-direct[name]).max()))
        observation_refs=[r for r in c['inputs'] if Path(r['path']).suffix=='.parquet']
        assert observation_refs,'Native source observations missing from feature receipt'
        source=pd.concat([pd.read_parquet(verify(r),columns=['target_index','time_location_float64','time_sigma_float64'])
                          for r in observation_refs],ignore_index=True).sort_values('target_index')
        assert not source.target_index.duplicated().any() and np.array_equal(source.target_index.to_numpy(),ids)
        diagnostics={}
        for name,column in [('mu','time_location_float64'),('sigma','time_sigma_float64')]:
            value=source[column].to_numpy(dtype=np.float64);assert np.isfinite(value).all()
            difference=direct[name]-value
            diagnostics[name]=dict(source_column=column,source_dtype=str(source[column].dtype),
                mean_signed_difference=float(difference.mean()),max_abs_difference=float(np.abs(difference).max()),
                exact_equal_events=int(np.count_nonzero(difference==0)),threshold_applied=False)
        verify(feature_receipt)
        for r in c['files'].values():verify(r)
        assert sha(cp)==c['checkpoint_file']['sha256']
        assert sha(helper)==s['bindings']['native_probability_helper']['sha256']
        result=dict(status='PASS',seed=seed,checkpoint=kind,split=split,rows=len(x),training_steps=0,
            contract_sha256=CONTRACT_SHA,feature_receipt=feature_receipt,feature_files=c['files'],
            checkpoint_file=rec(cp),driver=rec(__file__),fitter=rec(HERE/'head/fitter.py'),
            probability_helper=rec(helper),batch_size=batch_size,independent_same_input=exact,
            native_source_diagnostics=diagnostics,source_observation_refs=observation_refs,
            diagnostic_note='Source mu/sigma differences are diagnostic only; frozen quantity/NLL gates are unchanged.')
        save(result_path,result)
        print(json.dumps(dict(status='native_math_verified',seed=seed,checkpoint=kind,split=split,
                              rows=len(x),independent_same_input=exact,native_source_diagnostics=diagnostics)),flush=True)
    except Exception:
        save(directory/'NATIVE_MATH_QA.FAILURE.json',dict(status='FAIL',feature_receipt=feature_receipt,
             driver=rec(__file__),training_steps=0,traceback=traceback.format_exc()));raise
def fit_cohort(seed,kind,s):
    # A technical success receipt, not an improvement criterion, gates seed42.
    if seed==42:
        for k in ['best','last']:
            success=HERE/'fits'/f'seed52_{k}'/'COHORT_COMPLETE.json'
            assert success.is_file(),'Seed52 technical fit gate pending'
            for r in json.loads(success.read_text())['arm_bindings']:verify(r)
            assert json.loads((HERE/'fit_processes'/f'seed52_{k}'/'ACTUAL_EXIT.json').read_text())['actual_exit']==0
    qa=json.loads((HERE/'qa/PRELAUNCH_QA.json').read_text());assert qa['status']=='PASS'
    for r in qa['reviewed_files']:verify(r)
    directory=HERE/'features'/f'seed{seed}_{kind}_train'
    c=json.loads((directory/'COMPLETE.json').read_text());assert c['contract_sha256']==CONTRACT_SHA
    assert (c['seed'],c['checkpoint'],c['split'],c['rows'])==(seed,kind,'train',393824)
    for r in c['files'].values():verify(r)
    math_qa_path=directory/'NATIVE_MATH_QA.json'
    math_qa=json.loads(math_qa_path.read_text())
    assert math_qa['status']=='PASS' and math_qa['contract_sha256']==CONTRACT_SHA
    assert (math_qa['seed'],math_qa['checkpoint'],math_qa['split'],math_qa['rows'])==(seed,kind,'train',393824)
    assert Path(math_qa['feature_receipt']['path']).resolve()==(directory/'COMPLETE.json').resolve()
    for key in ['feature_receipt','checkpoint_file','driver','fitter','probability_helper']:verify(math_qa[key])
    for r in math_qa['feature_files'].values():verify(r)
    cp,original_state=load_checkpoint(seed,kind,s)
    x=np.load(directory/'hidden.npy',allow_pickle=False)
    ids=np.load(directory/'target_index.npy',allow_pickle=False)
    roles=pd.read_parquet(s['bindings']['train_roles']['path']).sort_values('target_index')
    meta=pd.read_parquet(s['bindings']['metadata']['path']);meta=meta.loc[meta.split.eq('train')].sort_values('target_index')
    assert np.array_equal(ids,roles.target_index) and np.array_equal(ids,meta.target_index)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    initial=torch.load(directory/'original_time_head.pt',weights_only=True,map_location='cpu')
    fit=fitter();config=fit.FitConfig(**s['fit_config'])
    assert fit.state_sha(initial)==c['head_state_sha256']==fit.state_sha(fit.validate_head_state(original_state))
    for arm in ['original','widthonly','joint']:
        out=HERE/'fits'/f'seed{seed}_{kind}'/arm
        result=fit.fit_arm(x,meta.target_gap.to_numpy(),roles.role.eq('fit').to_numpy(),roles.role.eq('dev').to_numpy(),
            initial,arm,config,out,lambda row:print(json.dumps(dict(seed=seed,checkpoint=kind,arm=arm,**row)),flush=True))
        save(out/'INPUT_BINDING.json',dict(contract=rec(HERE/'SCIENCE_SPEC.json'),feature_receipt=rec(directory/'COMPLETE.json'),
             native_math_qa=rec(math_qa_path),
             checkpoint=rec(cp),split=s['bindings']['train_roles'],fitter=rec(HERE/'head/fitter.py'),
             feature_files=c['files'],quantity_head_unchanged=True,torch_num_threads=torch.get_num_threads(),
             deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
        assert sha(cp)==c['checkpoint_file']['sha256']
        print(json.dumps(dict(status='fit_complete',seed=seed,checkpoint=kind,**result['receipt'])),flush=True)
    save(HERE/'fits'/f'seed{seed}_{kind}'/'COHORT_COMPLETE.json',dict(seed=seed,checkpoint=kind,status='COMPLETE',
        arm_bindings=[rec(HERE/'fits'/f'seed{seed}_{kind}'/arm/name) for arm in ['original','widthonly','joint']
                      for name in ['FIT_RECEIPT.json','INPUT_BINDING.json','selected_head.pt']],checkpoint_file=rec(cp)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare-best','audit-math','fit']);a.add_argument('--seed',type=int,choices=[52,42],required=True);a.add_argument('--kind',choices=['best','last'],default='best');a.add_argument('--split',choices=['train','validation'],default='train');args=a.parse_args()
    s=contract()
    if args.action=='audit-math':audit_math(args.seed,args.kind,args.split,s)
    elif args.action=='prepare-best':
        assert args.split=='train';prepare_best(args.seed,s)
    else:
        assert args.split=='train';fit_cohort(args.seed,args.kind,s)
