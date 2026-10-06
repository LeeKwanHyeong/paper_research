"""Read-only CPU diagnostics of SHA-admitted cached Train/Validation time heads.

No encoder, optimizer, GPU, target data loader, or held-out evaluation runs here.
All exported values are aggregate summaries; native_nll returns arrays only in memory.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import json
import math
import platform
import sys

PARAMETERS = ('v_t.weight','b_t','w_raw','time_scale_weight.weight')
CONTRACT_SHA = 'f47310a5d2cafa97510f401e1484653023223bf3fa8ec4550dfecbb8ef806187'
ARCHIVE_SHA = '1dbf6f033948813b17b8c004f8e43beffc0d276af258b30734199d7b54eb7348'
SOURCE_SHA = '4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0'
PROJECT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
DEFAULT_BUNDLE = PROJECT/'search_artifacts/titantpp_time_head_full_refit_5080_20261006_v1'
DEFAULT_SOURCE = PROJECT/'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1/source'


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_checked(root, relative, expected):
    root=Path(root).resolve(); rel=Path(relative); p=root/rel
    require(not rel.is_absolute() and '..' not in rel.parts and not p.is_symlink() and p.resolve().is_relative_to(root),'Unsafe immutable path')
    require(p.is_file() and sha(p)==expected,'Immutable file SHA mismatch: '+str(p))
    return p


def tensor_sha(state):
    """Same tensor serialization definition as frozen runner.py:524–549."""
    import torch
    h=hashlib.sha256()
    for name in sorted(state):
        value=state[name].detach().cpu().contiguous()
        meta=json.dumps({'name':name,'dtype':str(value.dtype),'shape':list(value.shape)},sort_keys=True,separators=(',',':')).encode()
        h.update(len(meta).to_bytes(8,'big'));h.update(meta)
        raw=value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b''
        h.update(len(raw).to_bytes(8,'big'));h.update(raw)
    return h.hexdigest()


def audit_bundle(bundle=DEFAULT_BUNDLE, source_root=DEFAULT_SOURCE):
    bundle=Path(bundle).resolve(); source_root=Path(source_root).resolve()
    c=read(bundle/'execution_contract.json');receipt=read(bundle/'analysis/completion_receipt.json')
    require(canonical(c)==CONTRACT_SHA==receipt['contract_sha256'],'Wrong approved full130 contract')
    require(receipt['local_originals_SHA_verified'] is True and receipt['held_out_test_evaluated'] is False,'Unverified archive receipt')
    require(receipt['archive_sha256']==ARCHIVE_SHA and sha(bundle/'retrieved/originals.tar.gz')==ARCHIVE_SHA,'Archive SHA mismatch')
    require(c['source']['files_sha256']==SOURCE_SHA and canonical(c['source']['files'])==SOURCE_SHA and len(c['source']['files'])==123,'Source closure mismatch')
    source_hashes={rel:sha(file_checked(source_root,rel,value)) for rel,value in c['source']['files'].items()}
    original=bundle/'retrieved/original'
    require(read(original/'execution_contract.json')==c,'Recovered contract mismatch')
    sealed=read(bundle/'retrieved/remote_archive_receipt.json')
    require(sealed['contract_sha256']==CONTRACT_SHA and sealed['archive_sha256']==ARCHIVE_SHA,'Remote archive receipt mismatch')
    for rel,value in c['operation']['files'].items():file_checked(original,rel,value)
    require(len(c['jobs'])==6 and {(j['dataset'],j['seed']) for j in c['jobs']}=={(d,s) for d in ('yellow_trip_hourly','raf_spare_parts') for s in (42,52,62)},'Unexpected analysis jobs')
    return {'contract':c,'original_root':original,'source_root':source_root,'archive_files':sealed['archive_files'],
        'provenance':{'contract_sha256':CONTRACT_SHA,'archive_sha256':ARCHIVE_SHA,'source_files_sha256':SOURCE_SHA,
            'source_123_files_verified':True,'operation_files_verified':True,'source_root':str(source_root),
            'native_likelihood_files':{k:v for k,v in source_hashes.items() if k in ('models/TPPs/CountAwareTPP.py','models/TPPs/positive_integer_time.py')},
            'receipt_path':str(bundle/'analysis/completion_receipt.json'),'receipt_sha256':sha(bundle/'analysis/completion_receipt.json')}}


def load_job_inputs(audit, job):
    import torch
    c=audit['contract'];root=audit['original_root'];d=root/'run'/job['id']
    require(job in c['jobs'],'Unadmitted job')
    terminal_path=file_checked(root,'run/'+job['id']+'/terminal_manifest.json',audit['archive_files']['run/'+job['id']+'/terminal_manifest.json'])
    terminal=read(terminal_path)
    require(terminal['status']=='complete' and terminal['scientific_success'] is True and terminal['job']==job and terminal['contract_sha256']==CONTRACT_SHA and terminal['held_out_test_evaluated'] is False,'Wrong terminal')
    require(terminal['trainable_scalar_count']==130 and terminal['original_checkpoint_sha_preserved'] is True,'Wrong head scope')
    file_hashes={}
    for rel,value in terminal['files'].items():
        p=file_checked(d,rel,value)
        require(audit['archive_files']['run/'+job['id']+'/'+rel]==value,'Archive/worker manifest mismatch')
        file_hashes[rel]=sha(p)
    supervisor=read(file_checked(root,'run/'+job['id']+'/supervisor_terminal_receipt.json',audit['archive_files']['run/'+job['id']+'/supervisor_terminal_receipt.json']))
    require(supervisor['scientific_success'] is True and supervisor['status']=='complete' and supervisor['contract_sha256']==CONTRACT_SHA,'Supervisor terminal mismatch')
    require(supervisor['files']['terminal_manifest.json']==sha(terminal_path),'Terminal SHA mismatch')
    source_rel='source_checkpoints/'+job['id']+'/source_selected.pt'
    source_path=file_checked(root,source_rel,job['checkpoint']['sha256'])
    require(audit['archive_files'][source_rel]==job['checkpoint']['sha256'],'Source checkpoint archive mismatch')
    # The original source checkpoint contains native RNG/optimizer metadata. Only
    # deserialize this locally recovered checkpoint after its immutable SHA check.
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    state=source['model_state_dict']
    require(tensor_sha(state)==job['checkpoint']['state_sha256']==source['model_state_sha256'],'Source tensor SHA mismatch')
    require(source.get('epoch',source.get('best_epoch'))==job['checkpoint']['epoch'],'Source selected epoch mismatch')
    require(source['checkpoint_monitor']=='validation_raw_quantity_rmse','Wrong original selector')
    frozen=tensor_sha({k:v for k,v in state.items() if k not in PARAMETERS})
    require(frozen==terminal['frozen_state_sha256'],'Frozen state mismatch')
    documents={name:read(d/(name+'.json')) for name in ('result','history','endpoint_replays','input_receipt','startup_gate')}
    result=documents['result']
    require(result['contract_sha256']==CONTRACT_SHA and result['job']==job['id'] and result['original_checkpoint']==job['checkpoint'],'Result binding mismatch')
    require(result['held_out_test_evaluated'] is False and result['quantity_output_unchanged'] is True,'Wrong scope')
    require(result['quantity_prediction_sha256_before']==result['quantity_prediction_sha256_after'],'Quantity bytehash changed')
    require(result['trainable_tensors']==list(PARAMETERS) and result['trainable_scalar_count']==130,'Wrong head names')
    data=next(v for v in c['datasets'] if v['dataset_id']==job['dataset'])
    require(documents['input_receipt']['held_out_materialized'] is False and documents['input_receipt']['populations']==data['inherited_data_identity']['populations'],'Population binding mismatch')
    heads={}
    for label,epoch in [('selected',result['best_epoch']),('last',result['completed_epochs'])]:
        payload=torch.load(d/(label+'_refit.pt'),map_location='cpu',weights_only=True)
        require(payload['contract_sha256']==CONTRACT_SHA and payload['job']==job['id'] and payload['source_checkpoint']==job['checkpoint'],'Head checkpoint binding mismatch')
        require(payload['epoch']==epoch and payload['frozen_state_sha256']==frozen and payload['held_out_test_evaluated'] is False,'Head epoch/scope mismatch')
        require(set(payload['head_state'])==set(PARAMETERS) and sum(v.numel() for v in payload['head_state'].values())==130,'Wrong head checkpoint tensors')
        heads[label+'_checkpoint']=payload
    caches={}
    for split in ('train','validation'):
        payload=torch.load(d/(split+'_feature_cache.pt'),map_location='cpu',weights_only=True)
        require(payload['scope']==split and payload['source_checkpoint_sha256']==job['checkpoint']['sha256'] and payload['frozen_state_sha256']==frozen,'Cache identity mismatch')
        cache=payload['cache'];n=data['inherited_data_identity']['populations'][split]['target_count']
        require(set(cache)=={'hidden','dt','qty','prediction','native_nll'},'Unexpected cache tensors')
        require(all(v.device.type=='cpu' and len(v)==n and bool(torch.isfinite(v).all()) for v in cache.values()),'Invalid cache')
        require(cache['hidden'].shape==(n,64) and bool((cache['dt']>=1).all()) and torch.equal(cache['dt'],cache['dt'].round()),'Invalid duration/hidden shape')
        require(tensor_sha({'prediction':cache['prediction']})==result['quantity_prediction_sha256_before'][split],'Cached quantity SHA mismatch')
        caches[split]=cache
    return {'source_checkpoint':source,**heads,'caches':caches,'documents':documents,'data':data,
        'provenance':{'job_id':job['id'],'directory':str(d),'terminal_sha256':sha(terminal_path),'input_files_sha256':file_hashes,
            'source_checkpoint_sha256':job['checkpoint']['sha256'],'source_tensor_sha256':job['checkpoint']['state_sha256'],'frozen_tensor_sha256':frozen}}


def load_native_class(source_root=DEFAULT_SOURCE):
    root=Path(source_root).resolve()
    for name,module in list(sys.modules.items()):
        if name=='models' or name.startswith('models.'):
            path=getattr(module,'__file__',None)
            require(path is None or Path(path).resolve().is_relative_to(root),'Already loaded foreign model module: '+name)
    sys.path.insert(0,str(root))
    module=importlib.import_module('models.TPPs.CountAwareTPP')
    require(Path(module.__file__).resolve()==root/'models/TPPs/CountAwareTPP.py','Wrong native likelihood class')
    return module.SharedTimeCountModel


def head_view(source_checkpoint, model_config, head_state=None, native_class=None):
    """Create only the native time decoder; never construct/run an encoder."""
    import torch
    native_class=native_class or load_native_class()
    state=source_checkpoint['model_state_dict']
    selected={name:state[name] for name in PARAMETERS} if head_state is None else head_state
    require(set(selected)==set(PARAMETERS),'Wrong time head names')
    require([tuple(selected[k].shape) for k in PARAMETERS]==[(1,64),(1,),(1,),(1,64)],'Wrong time head shapes')
    require(all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in selected.values()),'Invalid CPU head')
    # Use original methods unchanged on a head-only view with precisely their
    # required attributes. No quantity/encoder tensors are instantiated.
    head=native_class.__new__(native_class);torch.nn.Module.__init__(head)
    head.hidden_dim=64
    for key in ('time_head_mode','time_scale','time_sigma_floor','time_observation_contract'):
        setattr(head,key,copy.deepcopy(model_config[key]))
    require(head.time_head_mode=='heteroscedastic_lognormal_duration' and head.time_observation_contract['mode']=='positive_integer_round_clamp_v1','Wrong observation law')
    for module_name,weight_name in [('v_t','v_t.weight'),('time_scale_weight','time_scale_weight.weight')]:
        layer=torch.nn.Linear.__new__(torch.nn.Linear);torch.nn.Module.__init__(layer)
        layer.in_features=64;layer.out_features=1;layer.register_parameter('bias',None)
        layer.register_parameter('weight',torch.nn.Parameter(selected[weight_name].detach().clone(),requires_grad=False))
        setattr(head,module_name,layer)
    for name in ('b_t','w_raw'):head.register_parameter(name,torch.nn.Parameter(selected[name].detach().clone(),requires_grad=False))
    head.eval();require(sum(p.numel() for p in head.parameters())==130,'Wrong head dimension')
    return head


def native_nll(cache, head, batch_size=128):
    import torch
    values=[]
    with torch.no_grad():
        for start in range(0,len(cache['dt']),batch_size):
            value=-head.log_observation_dt(cache['hidden'][start:start+batch_size],cache['dt'][start:start+batch_size])
            require(value.device.type=='cpu' and bool(torch.isfinite(value).all()),'Nonfinite CPU native likelihood')
            values.append(value.detach().double())
    return torch.cat(values)


def native_mu_sigma(cache, head, batch_size=128):
    import torch
    mus=[];sigmas=[]
    with torch.no_grad():
        for start in range(0,len(cache['dt']),batch_size):
            mu,sigma,_=head._lognormal_time_terms(cache['hidden'][start:start+batch_size],cache['dt'][start:start+batch_size])
            mus.append(mu.detach().double());sigmas.append(sigma.detach().double())
    return torch.cat(mus),torch.cat(sigmas)


def distribution(values):
    import torch
    x=values.detach().double()
    require(len(x)>0 and bool(torch.isfinite(x).all()),'Invalid distribution')
    levels=[0.,.01,.05,.25,.5,.75,.95,.99,1.]
    quantiles=torch.quantile(x,torch.tensor(levels,dtype=torch.float64)).tolist()
    return {'count':len(x),'mean':float(x.mean()),'population_std':float(x.std(unbiased=False)),
        'quantiles':dict(zip(('min','p01','p05','p25','p50','p75','p95','p99','max'),quantiles))}


def cohort_masks(dt):
    return {'dt_1':dt==1,'dt_2_3':(dt>=2)&(dt<=3),'dt_4_7':(dt>=4)&(dt<=7),'dt_8_plus':dt>=8}


def summarize_losses(loss,dt,mu,sigma):
    import torch
    n=len(loss);total=float(loss.sum())
    require(n>0 and bool((loss>=-1e-12).all()),'Invalid positive-integer NLL')
    masks=cohort_masks(dt);require(torch.equal(sum(m.to(torch.int64) for m in masks.values()),torch.ones(n,dtype=torch.int64)),'Nonpartitioned duration cohorts')
    cohorts={}
    for name,mask in masks.items():
        count=int(mask.sum());subtotal=float(loss[mask].sum())
        cohorts[name]={'count':count,'population_share':count/n,'mean_nll':subtotal/count if count else None,
            'contribution_to_overall_mean_nll':subtotal/n,'share_of_total_nll':subtotal/total if total>0 else None,
            'mu':distribution(mu[mask]) if count else None,'sigma':distribution(sigma[mask]) if count else None}
    ordered=torch.sort(loss,descending=True).values
    concentration={}
    for fraction,label in ((.01,'top_1_percent'),(.05,'top_5_percent')):
        k=max(1,math.ceil(n*fraction));subtotal=float(ordered[:k].sum())
        concentration[label]={'count':k,'population_share':k/n,'mean_nll':subtotal/k,'contribution_to_overall_mean_nll':subtotal/n,
            'share_of_total_nll':subtotal/total if total>0 else None}
    return {'count':n,'mean_nll':total/n,'sum_nll':total,'zero_nll_count':int((loss==0).sum()),'loss_distribution':distribution(loss),
        'cohorts':cohorts,'loss_concentration':concentration,'mu':distribution(mu),'sigma':distribution(sigma),
        'sigma_fraction_below':{str(v):float((sigma<v).double().mean()) for v in (.01,.05,.1,.2)}}


def replay_difference(actual, reference):
    import torch
    actual=actual.double();reference=reference.double();delta=actual-reference;absolute=delta.abs()
    tolerance=1e-8+1e-7*reference.abs()
    return {'mean_signed_difference':float(delta.mean()),'mean_absolute_difference':float(absolute.mean()),
        'max_absolute_difference':float(absolute.max()),'p99_absolute_difference':float(torch.quantile(absolute,.99)),
        'diagnostic_atol':1e-8,'diagnostic_rtol':1e-7,'outside_diagnostic_tolerance_count':int((absolute>tolerance).sum()),
        'native_mean':float(reference.mean()),'cpu_mean':float(actual.mean()),
        'campaign_endpoint_atol':1e-5,'campaign_endpoint_rtol':1e-5,
        'within_campaign_endpoint_tolerance':math.isclose(float(actual.mean()),float(reference.mean()),rel_tol=1e-5,abs_tol=1e-5),
        'tolerance_basis':'Campaign verify_baseline uses atol=rtol=1e-5. Tighter per-row 1e-8+1e-7*abs(reference) is disclosed diagnostic flag, not a new scientific acceptance rule.'}


def head_changes(original, changed):
    import torch
    output={}
    for name in PARAMETERS:
        a=original[name].double().reshape(-1);b=changed[name].double().reshape(-1);delta=b-a
        norm_a=float(torch.linalg.vector_norm(a));norm_b=float(torch.linalg.vector_norm(b))
        output[name]={'numel':a.numel(),'before_mean':float(a.mean()),'after_mean':float(b.mean()),
            'before_l2':norm_a,'after_l2':norm_b,'delta_l2':float(torch.linalg.vector_norm(delta)),
            'max_absolute_delta':float(delta.abs().max()),'cosine':float(torch.dot(a,b))/(norm_a*norm_b) if norm_a*norm_b else None}
    return output


def parameter_block_decomposition(loss00, loss10, loss01, loss11, dt):
    """Symmetric two-block arithmetic at fixed representations, not causality."""
    import torch
    location=.5*((loss10-loss00)+(loss11-loss01))
    scale=.5*((loss01-loss00)+(loss11-loss10))
    delta=loss11-loss00
    residual=location+scale-delta
    require(bool(torch.allclose(location+scale,delta,rtol=1e-12,atol=1e-12)), 'Block contributions do not reconcile')
    def pack(mask):
        n=int(mask.sum())
        return {'count':n,'E0_location_E0_scale_mean_nll':float(loss00[mask].mean()) if n else None,
            'last_location_E0_scale_mean_nll':float(loss10[mask].mean()) if n else None,
            'E0_location_last_scale_mean_nll':float(loss01[mask].mean()) if n else None,
            'last_location_last_scale_mean_nll':float(loss11[mask].mean()) if n else None,
            'mean_nll_delta':float(delta[mask].mean()) if n else None,
            'location_symmetric_contribution_to_mean_delta':float(location[mask].mean()) if n else None,
            'scale_symmetric_contribution_to_mean_delta':float(scale[mask].mean()) if n else None,
            'location_contribution_to_overall_mean_delta':float(location[mask].sum())/len(dt),
            'scale_contribution_to_overall_mean_delta':float(scale[mask].sum())/len(dt)}
    return {'scope':'Fixed cached representations; parameter-block arithmetic only, not a causal explanation of training.',
        'candidate_checkpoint_saved':False,'new_fitting_or_selection':False,
        'location_block':['v_t.weight','b_t'],'scale_block':['w_raw','time_scale_weight.weight'],
        'maximum_absolute_row_reconciliation_residual':float(residual.abs().max()),
        'overall':pack(torch.ones(len(dt),dtype=torch.bool)),
        'cohorts':{name:pack(mask) for name,mask in cohort_masks(dt).items()},
        'dt_ge_2':pack(dt>=2)}


def analyze_job(audit,job,native_class):
    import torch
    inputs=load_job_inputs(audit,job);source=inputs['source_checkpoint'];data=inputs['data'];result=inputs['documents']['result']
    original={k:source['model_state_dict'][k] for k in PARAMETERS}
    states={'E0':original,'selected':inputs['selected_checkpoint']['head_state'],'last':inputs['last_checkpoint']['head_state']}
    output={'job':job['id'],'dataset':job['dataset'],'seed':job['seed'],'original_epoch':job['checkpoint']['epoch'],
        'selected_epoch':result['best_epoch'],'last_epoch':result['completed_epochs'],'provenance':inputs['provenance'],
        'native_time_coordinate':{'dt_unit':data['model']['time_observation_contract']['unit'],'time_scale':data['model']['time_scale'],
            'mu_meaning':'Log-normal location of log(dt/time_scale); unchanged native observation likelihood.',
            'sigma_meaning':'Event-conditioned log-normal scale, not zero-hidden bias telemetry.','sigma_floor':data['model']['time_sigma_floor']},
        'head_parameter_changes':{label:head_changes(original,states[label]) for label in ('selected','last')},'splits':{}}
    losses={}
    for split,cache in inputs['caches'].items():
        summary={'population_binding':data['inherited_data_identity']['populations'][split],'stages':{}};losses[split]={}
        for label,state in states.items():
            head=head_view(source,data['model'],state,native_class);before=tensor_sha(head.state_dict())
            loss=native_nll(cache,head);mu,sigma=native_mu_sigma(cache,head)
            require(before==tensor_sha(head.state_dict()),'Diagnostic mutated head state')
            losses[split][label]=loss;summary['stages'][label]=summarize_losses(loss,cache['dt'],mu,sigma)
        last_location={k:(states['last'][k] if k in ('v_t.weight','b_t') else original[k]) for k in PARAMETERS}
        last_scale={k:(states['last'][k] if k in ('w_raw','time_scale_weight.weight') else original[k]) for k in PARAMETERS}
        loss10=native_nll(cache,head_view(source,data['model'],last_location,native_class))
        loss01=native_nll(cache,head_view(source,data['model'],last_scale,native_class))
        summary['parameter_block_decomposition']=parameter_block_decomposition(losses[split]['E0'],loss10,loss01,losses[split]['last'],cache['dt'])
        summary['E0_cpu_vs_cached_native_nll']=replay_difference(losses[split]['E0'],cache['native_nll'])
        delta=losses[split]['last']-losses[split]['E0'];n=len(delta)
        summary['E0_to_last']={'overall_mean_nll_delta':float(delta.mean()),'cohorts':{name:{'count':int(mask.sum()),
            'mean_nll_delta':float(delta[mask].mean()) if bool(mask.any()) else None,
            'contribution_to_overall_mean_delta':float(delta[mask].sum())/n} for name,mask in cohort_masks(cache['dt']).items()}}
        if split=='validation':
            summary['cpu_vs_reported_endpoint_mean_nll']={label:{'reported':result[label]['time_nll'],
                'cpu':summary['stages'][label]['mean_nll'],'signed_delta':summary['stages'][label]['mean_nll']-result[label]['time_nll'],
                'within_campaign_tolerance':math.isclose(summary['stages'][label]['mean_nll'],result[label]['time_nll'],rel_tol=1e-5,abs_tol=1e-5)} for label in states}
        output['splits'][split]=summary
    history=inputs['documents']['history']['history']
    output['training_loss_distinction']={'reported_last_epoch_train_time_nll':history[-1]['train_time_nll'],
        'fixed_last_head_train_mean_nll':output['splits']['train']['stages']['last']['mean_nll'],
        'reported_loss_is_online_average_under_changing_weights':True,'endpoint_loss_uses_one_fixed_head_on_all_cached_Train_targets':True}
    output['generalization_gap_validation_minus_train']={label:output['splits']['validation']['stages'][label]['mean_nll']-output['splits']['train']['stages'][label]['mean_nll'] for label in states}
    for rel,expected in inputs['provenance']['input_files_sha256'].items():file_checked(inputs['provenance']['directory'],rel,expected)
    file_checked(audit['original_root'],'source_checkpoints/'+job['id']+'/source_selected.pt',job['checkpoint']['sha256'])
    output['provenance']['immutable_inputs_reverified_after_analysis']=True
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE);parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results.json'))
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    audit=audit_bundle(args.bundle,args.source_root);native=load_native_class(args.source_root)
    rows=[]
    for job in audit['contract']['jobs']:
        rows.append(analyze_job(audit,job,native));print('analyzed '+job['id'],flush=True)
    for rel,value in audit['contract']['source']['files'].items():file_checked(args.source_root,rel,value)
    require(sha(args.bundle/'retrieved/originals.tar.gz')==ARCHIVE_SHA,'Archive changed during diagnostic')
    report={'schema':'cached_time_head_generalization_diagnostics_v1','scope':'Train_and_Validation_cached_time_head_CPU_only',
        'held_out_test_evaluated':False,'optimizer_steps':0,'GPU_used':False,'encoder_reexecuted':False,'row_or_prediction_arrays_exported':False,
        'runtime':{'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'device':'cpu','threads':torch.get_num_threads(),
            'original_native_runtime':audit['contract']['runtime']['runtime_expected']},'provenance':audit['provenance'],
        'cohort_contract':'Native original-unit positive integer dt: 1, 2–3, 4–7, >=8; mutually exclusive and exhaustive.',
        'interpretation_limits':['CPU diagnostic replay on frozen GPU-produced representations; not an encoder CPU/GPU equivalence test.',
            'Existing full Train was already used for fitting; Train/Validation gap is descriptive, not independent calibration evidence.',
            'Actual event-conditioned mu/sigma summaries are distinct from zero-hidden intercept telemetry.',
            'Cohort loss shares describe this model, not causal explanations or external-model gap decompositions.'],
        'jobs':rows,'script_sha256':sha(__file__)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(args.output)
    print('output '+str(args.output)+' sha256 '+sha(args.output))


if __name__=='__main__':main()
