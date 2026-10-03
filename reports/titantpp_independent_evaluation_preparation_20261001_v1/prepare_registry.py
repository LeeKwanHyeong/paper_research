"""Bind all 84 original selections to locally verified binaries/frozen import roots."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(j):return hashlib.sha256(json.dumps(j,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 prior=read(ROOT/'reports/titantpp_independent_evaluation_protocol_20261001_v1/checkpoint_manifest.json')
 audit=read(OUT/'checkpoint_audit.json');assert audit['status']=='passed'
 replacements={(r['dataset'],r['model'],r['seed']):r for r in audit['rows']}
 roots={}
 for host in ['5080','5090']:
  p=ROOT/f'search_artifacts/titantpp_independent_evaluation_preparation_20261001_v1/{host}/attempt1/original'
  for b in p.iterdir():
   if b.is_dir():roots[b.name]=(b/'frozen_execution/execution_contract.json',b/'source')
 core=ROOT/'search_artifacts/titantpp_core_ablation_20260928_v1/final_audit_20260930_v1/terminal_5090'
 add=ROOT/'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5080_20261001_v1/original'
 raf=ROOT/'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/original'
 roots.update({'core':(core/'frozen_execution/execution_contract.json',core/'source'),
               'additional':(add/'execution_contract.json',add/'source'),'raf':(raf/'execution_contract.json',raf/'source')})
 bundles={}
 for name,(contract,source) in roots.items():
  c=read(contract);work=OUT/'evaluator_sources'/name
  for rel,h in c['source']['files'].items():
   assert sha(source/rel)==h,(name,rel)
   dst=work/rel;dst.parent.mkdir(parents=True,exist_ok=True)
   if not dst.exists():shutil.copy2(source/rel,dst)
   assert sha(dst)==h
  (work/'sample_data').mkdir(exist_ok=True)
  bundles[name]={'contract_path':str(contract.relative_to(ROOT)),'canonical_sha256':canonical(c),'source_root':str(work.relative_to(ROOT)),'source_files':c['source']['files'],'source_closure_sha256':c['source']['files_sha256'],'datasets':c['datasets']}
 rows=[]
 for r0 in prior['rows']:
  r=dict(r0);key=(r['dataset'],r['model'],r['seed'])
  if key in replacements:
   a=replacements[key];r.update({k:a[k] for k in ['checkpoint_path','checkpoint_file_sha256']});name=a['source_bundle']
   r.update({'local_binary_present':True,'binary_status':'passed_new_CPU_strict_loading_and_SHA','current_audit_path':str((OUT/'checkpoint_audit.json').relative_to(ROOT))})
  else:name='raf' if r['dataset']=='raf_spare_parts' else 'core' if r['model']=='titantpp_history_mlp' else 'additional'
  r['evaluator_source_bundle']=name
  assert sha(ROOT/r['checkpoint_path'])==r['checkpoint_file_sha256']
  assert bundles[name]['canonical_sha256']==r['contract_canonical_sha256']
  rows.append(r)
 assert len(rows)==84
 result={'status':'all_84_selected_binary_identities_verified','rows':rows,'bundles':bundles,'original_selection_manifest_sha256':sha(ROOT/'reports/titantpp_independent_evaluation_protocol_20261001_v1/checkpoint_manifest.json'),'execution_scope':'validation_preparation_only','heldout_execution_authorized':False}
 (OUT/'evaluation_registry.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'selected_binaries':len(rows),'frozen_source_bundles':len(bundles)}))
if __name__=='__main__':main()
