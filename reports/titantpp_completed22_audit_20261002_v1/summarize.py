"""Combine verified validation records; do not reevaluate or inspect held-out data."""
import csv, hashlib, json, statistics, time
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];REPORT=Path(__file__).resolve().parent
NEW=ROOT/'search_artifacts/titantpp_completed22_audit_20261002_v1/retrieved'
OLD=ROOT/'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/retrieved/completed13_20261002_v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(REPORT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
M=('qty_mae','qty_rmse','time_nll');rows=[];sources=[];retrievals=[]
for base,hosts,reused in [(OLD,['5080','5090'],True),(NEW,['5080','5090','pro4500','a100'],False)]:
 for host in hosts:
  d=base/host;a=read(d/'terminal_audit.json');assert a['status']=='passed' and a['originals_unchanged']
  manifest=read(d/'original/collection_manifest.json');rr=read(d/'retrieval_receipt.json')
  assert sha(d/'original.tar')==rr['archive_sha256']==a['archive_sha256']
  for rel,v in manifest['files'].items():assert sha(d/'original'/rel)==v['sha256'],(host,rel)
  sources.append({'path':str(d/'terminal_audit.json'),'sha256':sha(d/'terminal_audit.json'),'reused':reused})
  retrievals.append({'host':host,'reused':reused,'files':rr['files'],'bytes':rr['bytes'],'source_files':rr['source_files'],'conditions':len(a['conditions']),'checkpoints':a['verified_terminal_checkpoints']})
  for x in a['conditions']:
   j=x['job'];rows.append({'campaign':'a100_candidates' if host=='a100' else 'extension36','dataset':j['dataset'],'model':j['arm'],'seed':j['seed'],'host':host,'job_id':j['id'],'training_state':'complete','binary_CPU_audit':'passed','selected_epoch':x['selected_epoch'],'completed_epochs':x['completed_epochs'],**x['selected_validation'],'audit':str(d/'terminal_audit.json'),'original_job_path':str(d/'original/run'/j['id']),'contract_sha256':a['contract_sha256'],'source_closure_sha256':a['source_closure_sha256'],'reused_prior13_audit':reused,'paper_structural_scope':j['arm'] in ['titantpp_current_only_param_matched','titantpp_all_available_history_mlp']})
assert len(rows)==35 and len({x['job_id'] for x in rows})==35
assert Counter(x['campaign'] for x in rows)=={'extension36':31,'a100_candidates':4}
assert sum(x['paper_structural_scope'] for x in rows)==21
base_csv=ROOT/'reports/titantpp_dataset_appendix_20261002_v1/selected_conditions.csv'
base_rows=list(csv.DictReader(base_csv.open()));base=[]
for x in base_rows:
 if x['model']!='titantpp_history_mlp':continue
 assert sha(ROOT/x['source'])==x['source_sha256']
 base.append({'campaign':'reused_MLP','dataset':x['dataset'],'model':x['model'],'seed':int(x['seed']),'selected_epoch':int(x['selected_epoch']),**{k:float(x[k]) for k in M},'source':x['source'],'source_sha256':x['source_sha256']})
assert len(base)==12
sources.append({'path':str(base_csv),'sha256':sha(base_csv),'reused':True})
groups=defaultdict(list)
for x in base+rows:groups[x['dataset'],x['model']].append(x)
aggregates=[]
for (dataset,model),g in groups.items():
 seeds=sorted(x['seed'] for x in g);complete=seeds==[42,52,62]
 out={'dataset':dataset,'model':model,'seeds':seeds,'three_seed_complete':complete}
 if complete:
  out['mean']={k:statistics.mean(x[k] for x in g) for k in M};out['sample_sd']={k:statistics.stdev(x[k] for x in g) for k in M}
 if model!='titantpp_history_mlp':
  paired=[next(b for b in base if b['dataset']==dataset and b['seed']==x['seed']) for x in g]
  out['candidate_lower_paired_count']={k:sum(x[k]<b[k] for x,b in zip(g,paired)) for k in M}
  if complete:out['candidate_vs_MLP_percent']={k:(out['mean'][k]/statistics.mean(b[k] for b in paired)-1)*100 for k in M}
  out['paired']=[{'seed':x['seed'],'candidate':{k:x[k] for k in M},'MLP':{k:b[k] for k in M},'candidate_vs_MLP_percent':{k:(x[k]/b[k]-1)*100 for k in M}} for x,b in zip(g,paired)]
 aggregates.append(out)
# Full original 36-condition registry, including transferred ownership and pending jobs.
E=ROOT/'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2';A=ROOT/'search_artifacts/titantpp_mlp_candidates_a100_20261002_v1_retry1'
scope=read(REPORT/'scope.json');snapshots={h:read(v['snapshot']) for h,v in scope['hosts'].items()}
completed={(x['campaign'],x['job_id']):x for x in rows};registry=[]
for campaign,contract in [('extension36',read(E/'execution_contract.json')),('a100_candidates',read(A/'execution_contract.json'))]:
 for job in contract['jobs']:
  key=(campaign,job['id'])
  if key in completed:registry.append(completed[key]);continue
  host=job['host']
  if job['dataset']=='intermittent_frozen_5000' and job['seed']==62 and campaign=='extension36':host='pro4500'
  snapshot=snapshots[host]
  rr=next(r for r in snapshot['runs'] if r['job']['id']==job['id'])
  if host=='a100':
   rr=next(r for r in snapshot['resume_runs'] if r['job']['id']==job['id'])
  status=rr.get('status') or {};state=status.get('status','not_started')
  registry.append({'campaign':campaign,'dataset':job['dataset'],'model':job['arm'],'seed':job['seed'],'host':host,'job_id':job['id'],'training_state':state,'binary_CPU_audit':'not_yet_terminal_at_cutoff','observation_source':scope['hosts'][host]['snapshot'],'observed_unix':snapshot['observed_unix'],'paper_structural_scope':job['arm'] in ['titantpp_current_only_param_matched','titantpp_all_available_history_mlp']})
assert len(registry)==42
fields=['campaign','dataset','model','seed','host','job_id','training_state','binary_CPU_audit','selected_epoch','completed_epochs',*M,'reused_prior13_audit','paper_structural_scope','audit','original_job_path','contract_sha256','source_closure_sha256','observation_source','observed_unix']
with (REPORT/'condition_registry.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(registry)
save('comparison.json',{'cutoff_kst':scope['cutoff_kst'],'scope_sha256':sha(REPORT/'scope.json'),'evaluation_scope':'validation_only','completed':rows,'reused_MLP':base,'groups':aggregates,'counts':{'extension_complete_audited':31,'a100_complete_audited':4,'newly_audited':22,'reused_audited':13,'manuscript_structural_complete':21,'extension_incomplete_at_cutoff':5,'a100_incomplete_at_cutoff':2},'sources':sources})
save('verification.json',{'status':'passed','verified_unix':time.time(),'retrievals':retrievals,'new_checkpoints':44,'new_conditions':22,'reused_conditions':13,'all_collected_file_SHAs_rechecked':True,'prior_audits_reused_without_new_forward':True,'scope_sha256':sha(REPORT/'scope.json'),'sources':sources,'new_training_or_model_forward':False,'heldout_read':False,'excluded_from_paper_but_preserved':'deep_renewal_event_native_nb','running_or_later_completions_excluded':True})
for g in aggregates:
 if g['model']=='deep_renewal_event_native_nb':continue
 print(g['dataset'],g['model'],g['seeds'])
 if g['three_seed_complete']:print('mean',g['mean'],'SD',g['sample_sd'],'vsMLP%',g.get('candidate_vs_MLP_percent'),'wins',g.get('candidate_lower_paired_count'))
 elif g.get('paired'):print('paired',g['paired'])
print('REGISTRY',Counter((x['campaign'],x['training_state']) for x in registry))
