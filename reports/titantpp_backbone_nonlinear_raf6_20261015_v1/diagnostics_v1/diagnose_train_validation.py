"""Read-only bounded activation diagnostic; no optimizer, Test or checkpoint edits."""
import hashlib,json,math,platform,sys,time
from pathlib import Path
import statistics
PROJECT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
BUNDLE=PROJECT/'search_artifacts/titantpp_backbone_nonlinear_raf6_20261015_v1'
SOURCE=BUNDLE/'source'
OUT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,s):
 if not v:raise ValueError(s)
def write(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
contract=read(BUNDLE/'execution_contract.json')
require(contract['held_out_test_evaluated'] is False,'Training scope changed')
seals={}
for relative,expected in contract['source']['files'].items():
 path=SOURCE/relative;require(sha(path)==expected,'Frozen source drift: '+relative);seals[str(path)]=expected
for relative,expected in contract['input_files'].items():
 path=SOURCE/relative;require(sha(path)==expected,'Input drift: '+relative);seals[str(path)]=expected
# Exact selected originals only. Record all files used for convergence/bin analysis.
manifest=read(BUNDLE/'retrieval/manifest.json');rows=[]
for job in contract['jobs']:
 root=BUNDLE/'retrieval/originals/run'/job['id']
 terminal=read(root/'terminal_manifest.json')
 require(terminal['scientific_success'] is True and terminal['held_out_test_evaluated'] is False,'Nonterminal training record')
 run=root/'runs'/job['arm']/'count_only_log_regression'/('seed_'+str(job['seed']))
 for name in ('best_val_qty_rmse_model.pt','last_epoch_state.pt','history.json','endpoint_replays.json','summary.json'):
  path=run/name;expected=terminal['files'][path.relative_to(root).as_posix()]
  require(sha(path)==expected,'Original drift: '+str(path));seals[str(path)]=expected
 rows.append((job,run))
sys.path=[str(SOURCE)]+[p for p in sys.path if Path(p or '.').resolve()!=PROJECT and Path(p or '.').resolve()!=SOURCE]
import numpy as np,polars as pl,torch
from paper.scripts import run_titantpp_nonlinear_local_cnn as engine
from paper.scripts.count_aware_tpp_backbone import core
from paper.scripts import run_taxi_quantity_interface_ablation as loader_module
from models.TPPs.CountAwareTitanCNNGRU import causal_depthwise_residual
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
engine.install_hooks();torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
def imported():
 result={}
 for name,module in tuple(sys.modules.items()):
  if name.split('.')[0] not in {'models','paper','data_loader','simple_lab_test','utils'}:continue
  f=getattr(module,'__file__',None)
  if f:
   p=Path(f).resolve();require(p.is_relative_to(SOURCE),'Nonfrozen import '+name)
   rel=p.relative_to(SOURCE).as_posix();require(rel in contract['source']['files'],'Unsealed import '+name);result[name]=rel
 return result
imported()
data=contract['datasets'][0]
frame=pl.read_parquet(SOURCE/data['prepared_train_validation_path']).sort(['oper_part_no','seq'])
require(set(frame['chronological_split'].unique().to_list())=={'train','validation'},'Non-Train/Validation rows')
frame=core.prepare_count_frame(frame)
samples={};sample_proofs={}
for split in ('train','validation'):
 loader=loader_module.make_loader(frame,target_split=split,batch_size=128,shuffle=False,generator=None,lookback_weeks=data['loader']['lookback_weeks'],max_seq_len=data['loader']['max_seq_len'])
 ds=loader.dataset;require(len(ds)==data['inherited_data_identity']['populations'][split]['target_count'],'Population count mismatch')
 # Deterministic evenly spaced subset, decided independently of predictions/errors/seed.
 ids=np.linspace(0,len(ds)-1,128,dtype=np.int64).tolist();samples[split]=loader.collate_fn([ds[i] for i in ids])
 identity=[]
 for i in ids:
  part,end=ds.index[i];pos=end+1
  require(ds.split_lists[part][pos]==split,'Wrong target split')
  identity.append([str(ds.parts[part]),int(ds.seq_lists[part][pos])])
 sample_proofs[split]={'count':len(ids),'full_population_count':len(ds),'indices':ids,'target_identity_sha256':hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest(),'selection':'128 evenly spaced ordered loader target indices; no prediction/error/seed-based selection'}

def stat(x):
 x=x.detach().double().flatten();require(x.numel()>0 and bool(torch.isfinite(x).all()),'Nonfinite diagnostic')
 return {'count':x.numel(),'mean':float(x.mean()),'median':float(x.median()),'p95':float(torch.quantile(x,.95)),'min':float(x.min()),'max':float(x.max())}

def attention_diagnostic(module,args,kwargs):
 x=args[0];mask=kwargs['mask'];sources=kwargs['observed_sources'];available=kwargs['observed_available']
 bs,length,_=x.shape;raw=dict(zip(('q','k','v'),module.qkv(x).chunk(3,dim=-1)))
 old={name:raw[name]+causal_depthwise_residual(raw[name],getattr(module,'causal_'+name+'_kernel'),sources,available) for name in raw}
 delta={name:module.nonlinear_local_branches[name](raw[name],sources,available) for name in raw}
 new={name:old[name]+delta[name] for name in raw}
 # Focus on final observed query, which feeds the next-event prediction.
 last=mask.sum(1)-1;bid=torch.arange(bs);gather=lambda a:module._split_heads(a)[bid,:,last,:]
 norms={}
 for name in raw:
  a,b,c=gather(old[name]),gather(delta[name]),gather(new[name]);an=a.norm(dim=-1);bn=b.norm(dim=-1);cn=c.norm(dim=-1)
  require(bool((an>1e-8).all()),'Degenerate base norm')
  norms[name]={'added_over_old_norm':stat(bn/an),'new_over_old_norm':stat(cn/an),'cosine_old_new':stat(torch.nn.functional.cosine_similarity(a,c,dim=-1))}
 mem=[]
 if module._ctx_mem is not None and module._ctx_mem.numel()>0:mem.append(module._ctx_mem.to(x))
 if module.persistent_mem is not None:mem.append(module.persistent_mem.to(x).expand(bs,-1,-1))
 memory=torch.cat(mem,1) if mem else x.new_empty(bs,0,x.shape[-1]);ml=memory.shape[1]
 valid=torch.cat([torch.ones(bs,ml,dtype=torch.bool),mask],dim=1)[:,None,:].expand(bs,module.n_heads,-1)
 def score(q,k):
  keys=torch.cat([memory,k],dim=1)
  return torch.einsum('bhd,bhkd->bhk',gather(q),module._split_heads(keys))*module.scale
 def normalize(v,ref):
  vh=module._split_heads(v);rh=module._split_heads(ref)
  return module._merge_heads(vh*(rh.norm(dim=-1,keepdim=True)/vh.norm(dim=-1,keepdim=True).clamp_min(1e-8)))
 scores={'without_added_qk_branch':score(old['q'],old['k']),'actual_selected':score(new['q'],new['k']),'same_direction_old_per_head_norm':score(normalize(new['q'],old['q']),normalize(new['k'],old['k']))}
 probs={k:s.masked_fill(~valid,float('-inf')).softmax(-1) for k,s in scores.items()}
 probs_sums=[p.sum(-1) for p in probs.values()];require(all(torch.allclose(z,torch.ones_like(z)) for z in probs_sums),'Softmax normalization failure')
 ent={k:stat(-(p*p.clamp_min(1e-30).log()).sum(-1)) for k,p in probs.items()}
 comparison={}
 for variant in ('actual_selected','same_direction_old_per_head_norm'):
  dif=scores[variant]-scores['without_added_qk_branch'];p=probs[variant];base=probs['without_added_qk_branch']
  comparison[variant]={'valid_logit_abs_difference':stat(dif[valid].abs()),'attention_total_variation':stat(.5*(p-base).abs().sum(-1)),'entropy_difference':stat((-(p*p.clamp_min(1e-30).log()).sum(-1))-(-(base*base.clamp_min(1e-30).log()).sum(-1)))}
 return {'heads':module.n_heads,'memory_key_count':ml,'observed_history_lengths':stat(mask.sum(1)),'norms':norms,'entropy':ent,'comparison':comparison,'interpretation':'Same trained candidate and same block input; diagnostic arithmetic only, not historical CNN and not a new model performance evaluation. Norm-preserved direction computes only attention scores; no replacement forward or optimizer.'}
results=[];start=time.time()
for job,run in rows:
 p=torch.load(run/'best_val_qty_rmse_model.pt',map_location='cpu',weights_only=False);endpoint=read(run/'endpoint_replays.json');h=read(run/'history.json')['history'];summary=read(run/'summary.json')
 require(canonical_state_dict_sha256(p['model_state_dict'])==p['model_state_sha256']==endpoint['selected']['state_sha256'],'Selected tensor SHA mismatch')
 engine.validate_checkpoint(p,job['arm']);model,_=engine.build_model(data,job['arm']);model.load_state_dict(p['model_state_dict'],strict=True);model.eval().requires_grad_(False)
 before=canonical_state_dict_sha256(model.state_dict());last=torch.load(run/'last_epoch_state.pt',map_location='cpu',weights_only=False);ls=last['model_state_dict']
 weights={name:{'selected_norm':float(value.norm()),'selected_nonzero':int(torch.count_nonzero(value)),'numel':value.numel(),'last_minus_selected_norm':float((ls[name]-value).norm())} for name,value in p['model_state_dict'].items() if name.startswith(engine.EXTRA_PREFIX)}
 bestrow=next(z for z in h if z['epoch']==p['best_epoch'])
 out={'arm':job['arm'],'seed':job['seed'],'selected_epoch':p['best_epoch'],'last_epoch':h[-1]['epoch'],'checkpoint_state_sha256':before,'weights':weights,'samples':{},'convergence':{'all_train_rows_finite':all(z['train_all_finite'] for z in h),'train_log_quantity_loss_first_best_last':[h[0]['train_quantity_loss'],bestrow['train_quantity_loss'],h[-1]['train_quantity_loss']],'validation_raw_rmse_best_last':[bestrow['val_qty_rmse'],h[-1]['val_qty_rmse']],'minimum_validation_mae_epoch':min(h,key=lambda z:z['val_qty_mae'])['epoch'],'minimum_validation_log_mse_epoch':min(h,key=lambda z:z['val_log_qty_mse'])['epoch'],'mean_global_gradient_clip_fraction':statistics.mean(z['train_gradient_clip_fraction'] for z in h),'per_branch_gradients_logged':False,'train_raw_errors_logged':False},'selected_validation_quantity_rows':summary['quantity_rows']}
 for split,batch in samples.items():
  captures=[]
  handle=model.encoder.layers[0].attn.register_forward_pre_hook(lambda module,args,kwargs:captures.append(attention_diagnostic(module,args,kwargs)),with_kwargs=True)
  _,dts,mask,_,quantities=batch
  with torch.no_grad():value=core.target_outputs(model,dts,mask,quantities,lambda_log_qty=1.)
  handle.remove();require(len(captures)==1,'Unexpected attention call count')
  error=value['pred_qty']-value['true_qty'];out['samples'][split]={'activation':captures[0],'sample_raw_rmse':float(error.square().mean().sqrt()),'sample_raw_mae':float(error.abs().mean()),'sample_tail_count':int((value['true_qty']>200).sum()),'sample_count':len(error)}
 require(canonical_state_dict_sha256(model.state_dict())==before,'Model state mutated')
 out['model_tensor_unchanged']=True;results.append(out)
 print(job['id'],'completed',flush=True)
refs={j['seed']:j['original']['validation_reference'] for j in contract['baseline_jobs']};decomp=[]
for arm in engine.ARMS:
 selected={job['seed']:read(run/'endpoint_replays.json')['selected'] for job,run in rows if job['arm']==arm}
 cells=[]
 for i in range(5):
  n=refs[42]['quantity_cells'][i]['count'];ds=statistics.mean(selected[s]['quantity_cells'][i]['qty_sse']-refs[s]['quantity_cells'][i]['qty_sse'] for s in (42,52,62));da=statistics.mean(n*(selected[s]['quantity_cells'][i]['qty_mae']-refs[s]['quantity_cells'][i]['qty_mae']) for s in (42,52,62));cells.append({'bin':i,'count':n,'mean_seed_delta_sse':ds,'mean_seed_delta_absolute_error_sum':da})
 decomp.append({'arm':arm,'reference':'historical CNN3, native runtime differs','cells':cells})
receipt={'schema':'raf_nonlinear_train_validation_diagnostic_v1','held_out_test_evaluated':False,'test_files_read':False,'optimizer_steps':0,'checkpoint_writes':0,'baseline_reevaluation_count':0,'models_evaluated':6,'source_closure':contract['source']['files_sha256'],'contract_raw_sha256':sha(BUNDLE/'execution_contract.json'),'diagnostic_script_sha256':sha(Path(__file__)),'input_and_original_raw_sha256':seals,'imported_project_modules':imported(),'runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','platform':platform.platform(),'threads':torch.get_num_threads(),'native_training_torch':'2.11.0+cu130','native_training_cudnn':92000,'native_equivalence_claimed':False},'sample_proofs':sample_proofs,'results':results,'validation_error_decomposition':decomp,'elapsed_seconds':time.time()-start,'limitations':['A small deterministic ordered subset is not the full Train/Validation population and may not represent tail conditions.','Local CPU torch differs from native CUDA torch; activation diagnostics are exploratory and do not replace native recorded evaluation.','Attention arithmetic counterfactual does not establish prediction improvement or causal explanation.','No per-branch gradient histories, Train raw full-population errors or paired row predictions were recorded during training.']}
write('RESULTS.json',receipt)
print('RESULTS_SHA256',sha(OUT/'RESULTS.json'),flush=True)
