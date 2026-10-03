"""CPU-only strict loading of six frozen legacy source bundles, in isolated processes."""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ART=ROOT/'search_artifacts/titantpp_independent_evaluation_preparation_20261001_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(j):return hashlib.sha256(json.dumps(j,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def one(host,bundle):
 original=ART/host/'attempt1/original';root=original/bundle
 c=read(root/'frozen_execution/execution_contract.json')
 collection=read(original/'collection_manifest.json')
 request=read(original.parent/'request.json')
 spec=next(b for b in request['bundles'] if b['name']==bundle)
 assert canonical(c)==spec['canonical_sha256']
 work=OUT/'audit_workspace'/bundle
 for rel,h in c['source']['files'].items():
  assert sha(root/'source'/rel)==h
  dest=work/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(root/'source'/rel,dest)
  assert sha(dest)==h
 (work/'sample_data').mkdir(exist_ok=True)
 sys.path.insert(0,str(work));os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch,numpy
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 from paper.scripts.run_local_detail_benchmark import build_model,audit_arm,audit_replay_accounting
 from models.TPPs.CountAwareFactory import validate_checkpoint_route
 from simple_lab_test.search.common.runner import canonical_state_dict_sha256
 results=[]
 for r in spec['rows']:
  rec=next(x for x in collection['checkpoints'] if (x['dataset'],x['model'],x['seed'])==(r['dataset'],r['model'],r['seed']))
  cpfile=original/rec['relative_path'];run=cpfile.parent
  assert sha(cpfile)==rec['sha256']
  cp=torch.load(cpfile,map_location='cpu',weights_only=False)
  summary=read(run/'summary.json');history=read(run/'history.json')['history'];exposure=read(run/'exposure.json');replay=read(run/'endpoint_replays.json')
  data=next(d for d in c['datasets'] if d['dataset_id']==r['dataset'])
  model,_=build_model(data,r['model'])
  validate_checkpoint_route(cp,r['model']);model.load_state_dict(cp['model_state_dict'],strict=True)
  assert all(t.device.type=='cpu' and torch.isfinite(t).all() for t in cp['model_state_dict'].values())
  state=canonical_state_dict_sha256(cp['model_state_dict'])
  assert state==r['state_tensor_sha256']==cp['model_state_sha256']==summary['checkpoint_state_sha256']==replay['selected']['state_sha256']
  assert cp['best_epoch']==r['selected_epoch']==min(history,key=lambda x:x['val_qty_rmse'])['epoch']
  assert cp['seed']==r['seed'] and cp['backbone']==r['model'] and cp['variant']=='count_only_log_regression'
  assert cp['evaluation_scope']=='validation_only' and cp['held_out_test_evaluated'] is False
  assert cp['interface_meta']['execution_contract_sha256']==canonical(c)
  assert cp['interface_meta']['source_files_sha256']==c['source']['files_sha256']
  assert cp['source_revision']==c['source']['base_git_revision']
  for k in ['encoder_config','initial_state_sha256','checkpoint_monitor','checkpoint_monitor_history_key','checkpoint_selection','selection_formula','resume_identity','optimizer_group_contract','interface_meta']:
   assert cp[k]==summary[k],(r['model'],k)
  assert cp['checkpoint_monitor']=='validation_raw_quantity_rmse' and cp['checkpoint_monitor_history_key']=='val_qty_rmse'
  assert sum(t.numel() for t in model.parameters())==summary['parameter_count']
  steps=audit_arm(history,summary,exposure,data,c['training'])
  audit_replay_accounting(replay['selected'])
  olddir=(ROOT/r['validation_replay_path']).parent
  for f in ['summary.json','endpoint_replays.json','history.json','exposure.json']:
   assert sha(run/f)==sha(olddir/f),('old/new record mismatch',f)
  results.append({'dataset':r['dataset'],'model':r['model'],'seed':r['seed'],'selected_epoch':r['selected_epoch'],'checkpoint_path':str(cpfile.relative_to(ROOT)),'checkpoint_file_sha256':sha(cpfile),'state_tensor_sha256':state,'status':'passed','strict_cpu_loading':True,'finite_tensors':True,'first_strict_validation_RMSE_selection_verified':True,'historical_json_records_unchanged':True,'recorded_exposure_audit':steps,'parameter_count':summary['parameter_count'],'source_bundle':bundle,'source_closure_sha256':c['source']['files_sha256'],'contract_canonical_sha256':canonical(c),'initialization_evidence':'saved_native_hash_chain_reused_not_regenerated','optimizer_state_audit':'not_applicable_selected_checkpoint_contains_weights_and_group_contract_only','new_forward_calls':0})
 for rel,v in collection['files'].items():
  if rel.startswith(bundle+'/'):assert sha(original/rel)==v['sha256']
 result={'status':'passed','bundle':bundle,'host':host,'source_files':len(c['source']['files']),'source_closure_sha256':c['source']['files_sha256'],'rows':results,'cpu_runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':numpy.__version__},'new_forward_calls':0,'raw_data_rows_read':False}
 p=OUT/'audits'/f'{bundle}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'bundle':bundle,'checkpoints':len(results),'status':'passed'}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--host');ap.add_argument('--bundle');a=ap.parse_args()
 if a.bundle:return one(a.host,a.bundle)
 results=[]
 for host in ['5080','5090']:
  request=read(ART/host/'attempt1/request.json')
  for b in request['bundles']:
   subprocess.run([sys.executable,__file__,'--host',host,'--bundle',b['name']],check=True)
   results.append(read(OUT/'audits'/f"{b['name']}.json"))
 rows=[r for b in results for r in b['rows']];assert len(rows)==36
 result={'status':'passed','audited_unix':time.time(),'selected_checkpoints':36,'source_files_per_bundle':{b['bundle']:b['source_files'] for b in results},'rows':rows,'new_forward_calls':0,'optimizer_state_audit_scope':'selected weight identity only; no last-state/optimizer binaries requested or loaded','prior_48_CPU_audits_reused':True}
 (OUT/'checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'selected_checkpoints':36,'status':'passed'}))
if __name__=='__main__':main()
