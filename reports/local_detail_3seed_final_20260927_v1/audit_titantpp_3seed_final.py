from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from collections import Counter
import hashlib, json, math, statistics, sys

ROOT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
BASE=ROOT/'search_artifacts/local_detail_instacart_remaining_5090_20260926_v1'
TERM=BASE/'monitor/20260927T130927Z/terminal_5090_recovery'
OUT=ROOT/'reports/local_detail_3seed_final_20260927_v1'
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canonical=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
close=lambda a,b:math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
ARMS=['titantpp','titantpp_local_detail','rmtpp','thp','nhp','sahp']
SOURCES={
 'taxi42':('search_artifacts/local_detail_benchmark_execution_20260922_v2/monitor/20260922T234557Z/taxi_complete','local_detail_benchmark_observed_time_v2','0dfb94f82f6b3b88bce779deca3987fbc8cbb96ba9ef66c8b6247bbe7c5afca8'),
 'inter42':('search_artifacts/local_detail_benchmark_intermittent_recovery_20260923_v4/monitor/20260923T235525Z/terminal_5080','local_detail_benchmark_observed_time_v4','45c8b03a23a2c2aa5f0889708c2a0f68d7f958636b5e8e5ebd12c0b327180da5'),
 'insta42':('search_artifacts/local_detail_benchmark_execution_20260923_v3/monitor/20260924T122457Z/terminal_5090_seed42','local_detail_benchmark_observed_time_v3','6220992f788973bc3adc12224060fe4fa8cc0e5ecd0ad0dcf2128abb9108e18a'),
 '5080new':('search_artifacts/local_detail_replication_5080_20260924_v1/monitor/20260925T204245Z/terminal_5080','local_detail_benchmark_replication_v1','0e96e72970b244b4320809336f8e26286b8eecf7803897a2113ff27f3874ffc8'),
 '5090old':('search_artifacts/local_detail_replication_5090_20260924_v1/monitor/20260926T060901Z/pre_recovery','local_detail_instacart_replication_v1','f03afd26bf0085d6f9f6b601ae4b2b25072bf6fc67897c2173653536ac209aa7'),
 'recovery':(str(TERM.relative_to(ROOT)),'local_detail_instacart_remaining_recovery_v1','a4a4f8bb11a86e0866dc2b1bb35bb9dd736a492b13ab7c60b656238aff6ad6a5')}
contracts={};manifests={};source_receipts={}
for key,(folder,name,expected) in SOURCES.items():
 root=ROOT/folder;c=read(ROOT/'paper/contracts'/f'{name}.json');assert canonical(c)==expected,key
 assert canonical(c['source']['files'])==c['source']['files_sha256']
 m=read(root/'collection_manifest.json')
 for rel,info in m['files'].items():
  assert digest(root/rel)==(info if isinstance(info,str) else info['sha256']),(key,rel)
 contracts[key]=c;manifests[key]=m
 source_receipts[key]={'root':folder,'contract_sha256':expected,'source_files_sha256':c['source']['files_sha256'],
                       'collection_manifest_sha256':digest(root/'collection_manifest.json'),'collected_files_verified':len(m['files'])}

# Reused evidence is validated against frozen lineage, not selected by favorable scores.
for item in read(ROOT/'paper/contracts/local_detail_replication_seed42_lineage_v1.json')['datasets']:
 key='taxi42' if item['dataset']=='yellow_trip_hourly' else 'inter42';root=ROOT/SOURCES[key][0]
 for rel,expected in item['files'].items():assert digest(root/rel)==expected
c=contracts['recovery'];m=manifests['recovery'];oldroot=ROOT/SOURCES['5090old'][0]
assert m['predecessor_lineage_sha256']==c['remaining_recovery']['completed_files_sha256']
for rel,expected in c['remaining_recovery']['completed_files_sha256'].items():
 actual=manifests['5090old']['checkpoint_file_hashes'][rel]['sha256'] if rel.endswith('.pt') else digest(oldroot/rel)
 assert actual==expected,rel
assert len(m['checkpoint_file_hashes'])==8
approval,permit,tp=[read(TERM/n) for n in ['approval.json','start_permit.json','training_permit.json']]
assert approval['approved'] and canonical(approval)==permit['approval_sha256']==tp['approval_sha256']
assert permit['contract_sha256']==tp['contract_sha256']==canonical(c)
assert permit['deadline_unix']==tp['deadline_unix']==c['launch']['fixed_deadline_unix']
w=read(TERM/'run/wrapper_manifest.json');q=read(TERM/'qualification/receipt.json');ds=read(TERM/'dispatch_status.json')
assert w['contract_sha256']==canonical(c) and w['permit_sha256']==canonical(tp) and w['approval_sha256']==canonical(approval)
assert w['isolation']=='fresh_exec_process_per_arm' and w['automatic_retry'] is False
assert w['arms']==c['execution_arms'] and w['deadline_unix']==permit['deadline_unix']
assert q['status']=='passed' and q['synthetic_optimizer_updates']==60 and all(q['checks'].values())
assert all(q['runtime'][k]==v for k,v in c['hosts']['5090']['runtime_expected'].items())
assert q['runtime']['gpu']['uuid']==c['hosts']['5090']['gpu_uuid'] and q['runtime']['environment']==c['hosts']['5090']['environment']
assert tp['qualifications']['5090']['receipt']==q and tp['qualifications']['5090']['receipt_sha256']==canonical(q)
assert ds['status']==read(TERM/'run/status.json')['status']=='complete' and ds['completed_at_unix']<permit['deadline_unix']
snapshot=read(ROOT/m['terminal_process_snapshot'])
assert snapshot['tmux_exit']==1 and len(snapshot['owned_processes'].strip().splitlines())==1
pids=[]
for arm in c['execution_arms']:
 worker=read(TERM/f'run/workers/{arm}.json');receipt=read(TERM/f'run/arm_receipts/{arm}.json');progress=read(TERM/f'run/progress/{arm}.json')
 assert receipt['status']=='complete' and worker['pid']==receipt['pid']==progress['pid']
 assert worker['authority']['contract_sha256']==receipt['contract_sha256']==progress['contract_sha256']==canonical(c)
 assert worker['authority']['permit_sha256']==canonical(tp) and progress['calls']>0
 pids.append(worker['pid'])
assert len(set(pids))==4

def audit_endpoint(e, expected_n):
 assert e['evaluation_scope']=='validation_only' and e['held_out_test_evaluated'] is False
 assert e['count']==expected_n and e['time_metric']=='recorded_positive_integer_time_nll'
 assert all(math.isfinite(e[k]) for k in ['qty_rmse','qty_mae','time_nll'])
 assert close(e['qty_rmse']**2*e['count'],e['qty_sse'])
 for kind in ['quantity','history','additional_history']:
  cells=e[kind+'_cells'];bounds=e[kind+'_boundaries']
  assert len(cells)==len(bounds)+1 and [x['bin'] for x in cells]==list(range(len(cells)))
  assert sum(x['count'] for x in cells)==expected_n
  assert close(sum(x['qty_sse'] for x in cells),e['qty_sse'])
  for metric in ['qty_mae','time_nll']:
   assert close(sum(x[metric]*x['count'] for x in cells if x['count'])/expected_n,e[metric])
  for cell in cells:
   if cell['count']:
    assert close(cell['qty_rmse']**2*cell['count'],cell['qty_sse'])
   else:assert cell['qty_sse']==0 and all(cell[k] is None for k in ['qty_rmse','qty_mae','time_nll'])
 assert e['tail']=={k:v for k,v in e['quantity_cells'][4].items() if k!='bin'}
 assert e['body']['count']==sum(x['count'] for x in e['quantity_cells'][:3])
 assert close(e['body']['qty_sse'],sum(x['qty_sse'] for x in e['quantity_cells'][:3]))

def gates(b,l,g):
 x,y=b['selected'],l['selected']
 checks={'raw_rmse_improves':y['qty_rmse']<x['qty_rmse'],'raw_mae_guard':y['qty_mae']<=x['qty_mae']*g['overall_mae_ratio_max']}
 for region in ['body','tail']:checks[region+'_mae_guard']=y[region]['qty_mae']<=x[region]['qty_mae']*g[region+'_mae_ratio_max']
 checks['tail_rmse_improves']=y['tail']['qty_rmse']<x['tail']['qty_rmse']
 for i in [1,2,3]:
  for metric in ['qty_mae','qty_rmse']:
   checks[f'middle_bin_{i}_{metric}_guard']=y['quantity_cells'][i][metric]<=x['quantity_cells'][i][metric]*g['each_nonempty_middle_bin_mae_and_rmse_ratio_max']
 checks['recorded_time_nll_guard']=y['time_nll']<=x['time_nll']+g['recorded_time_nll_increase_max']
 for metric,key in [('mean','last30_rmse_mean_ratio_max'),('sd','last30_rmse_sample_sd_ratio_max')]:
  checks['last30_rmse_'+metric+'_guard']=l['last30'][metric]<=b['last30'][metric]*g[key]
 return checks

jobs=[]
for key,dataset in [('taxi42','yellow_trip_hourly'),('inter42','intermittent_frozen_5000'),('insta42','insta_market_basket')]:jobs.append((dataset,42,key))
for dataset in ['yellow_trip_hourly','intermittent_frozen_5000']:
 for seed in [52,62]:jobs.append((dataset,seed,'5080new'))
jobs += [('insta_market_basket',52,'5090old'),('insta_market_basket',62,'recovery')]
rows=[];job_results={};all_endpoints={};selected_file_hashes={};identity_by_dataset={}
for dataset,seed,key in jobs:
 jobkey=f'{dataset}/seed_{seed}';contract=contracts[key];root=ROOT/SOURCES[key][0]
 data=next(x for x in contract['datasets'] if x['dataset_id']==dataset)
 identity={k:data[k] for k in ['model','loader','optimizer','quantity_boundaries_all_train_rows','history_boundaries','statistics','time_statistics']}
 # Expected sample/hash identities and scientific settings must agree across seeds.
 if dataset in identity_by_dataset:assert identity==identity_by_dataset[dataset],jobkey
 else:identity_by_dataset[dataset]=identity
 jobfolder=root/'run'/dataset
 if seed!=42:jobfolder=jobfolder/f'seed_{seed}'
 inputs=read(jobfolder/'input_receipt.json');init=read(jobfolder/'initialization.json');paired=read(jobfolder/'paired_comparison.json')
 assert inputs['held_out_materialized'] is False and inputs['populations']==data['inherited_data_identity']['populations']
 assert inputs['input_identity']['data_sha256']==data['inherited_data_identity']['data']['sha256']
 assert inputs['input_identity']['split_manifest_sha256']==data['inherited_data_identity']['split_manifest']['sha256']
 assert set(init)==set(paired['arms'])==set(ARMS) and paired['status']=='complete'
 assert paired['evaluation_scope']=='validation_only' and paired['held_out_test_evaluated'] is False
 exps={};eps={};stepsjob=0
 for arm in ARMS:
  actualkey='5090old' if key=='recovery' and arm in ARMS[:2] else key
  actualroot=ROOT/SOURCES[actualkey][0];actualcontract=contracts[actualkey]
  folder=actualroot/'run'/dataset
  if seed!=42:folder=folder/f'seed_{seed}'
  run=folder/'runs'/arm/'count_only_log_regression'/f'seed_{seed}'
  summary,hfile,exp,ep,timing=[read(run/n) for n in ['summary.json','history.json','exposure.json','endpoint_replays.json','epoch_timing.json']]
  for filename in ['summary.json','history.json','exposure.json','endpoint_replays.json','epoch_timing.json']:
   selected_file_hashes[str((run/filename).relative_to(ROOT))]=digest(run/filename)
  h=hfile['history'];n=len(h);assert 40<=n<=300 and [v['epoch'] for v in h]==list(range(1,n+1)),(jobkey,arm)
  assert summary['status']=='success' and summary['seed']==seed and summary['backbone']==arm
  assert summary['evaluation_scope']=='validation_only' and summary['held_out_test_evaluated'] is False
  assert summary['interface_meta']['execution_contract_sha256']==canonical(actualcontract)
  assert summary['interface_meta']['source_files_sha256']==actualcontract['source']['files_sha256']
  assert summary['checkpoint_monitor']=='validation_raw_quantity_rmse'
  best=h[0];firststop=None
  for row in h:
   assert row['train_all_finite'] and all(math.isfinite(row[f'val_{x}']) for x in ['qty_rmse','qty_mae','time_nll'])
   if row['val_qty_rmse']<best['val_qty_rmse']:best=row
   if firststop is None and row['epoch']>=40 and row['epoch']-best['epoch']>=40:firststop=row['epoch']
  assert n==(firststop if firststop is not None else 300)
  assert summary['stopped_early']==ep['stopped_early']==(firststop is not None)
  assert summary['best_epoch']==ep['best_epoch']==best['epoch'] and summary['completed_epochs']==ep['completed_epochs']==n
  assert summary['initial_state_sha256']==ep['initial_state_sha256']==init[arm]
  assert len(exp['train'])==n and len(exp['validation'])==n+1
  for split in ['train','validation']:
   count=inputs['populations'][split]['target_count'];batches=math.ceil(count/128)
   for i,row in enumerate(exp[split],1):assert row['epoch']==i and row['split']==split and row['count']==count and row['batches']==batches and len(row['batch_order_sha256'])==64
  for row in h:assert row['train_event_count']==inputs['populations']['train']['target_count'] and row['train_batch_count']==data['steps_per_epoch']
  steps=sum(x['batches'] for x in exp['train']);assert steps==ep['global_steps']==n*data['steps_per_epoch']
  assert paired['arms'][arm]==ep and ep['parameter_count']==summary['parameter_count']
  assert ep['elapsed_seconds']==summary['elapsed_seconds']<actualcontract['limits']['per_arm_wall_seconds']
  assert len(timing['epochs'])==n
  for endpoint,row in [('selected',best),('last',h[-1])]:
   e=ep[endpoint];audit_endpoint(e,inputs['populations']['validation']['target_count'])
   assert e['quantity_boundaries']==data['quantity_boundaries_all_train_rows'] and e['history_boundaries']==data['history_boundaries']
   for metric in ['qty_rmse','qty_mae','time_nll']:assert close(e[metric],row['val_'+metric]),(jobkey,arm,metric)
  assert ep['selected']['state_sha256']==summary['checkpoint_state_sha256']
  for window,values in [('first40',[v['val_qty_rmse'] for v in h[:40]]),('last30',[v['val_qty_rmse'] for v in h[-30:]])]:
   assert ep[window]['count']==len(values) and close(ep[window]['mean'],statistics.mean(values)) and close(ep[window]['sd'],statistics.stdev(values))
  if actualkey=='recovery':assert read(TERM/f'run/arm_receipts/{arm}.json')['result']==ep
  if key=='recovery':
   prov=paired['provenance'][arm]
   assert prov['reused']==(actualkey=='5090old') and prov['contract_sha256']==canonical(actualcontract)
   assert summary['initial_state_sha256']==c['remaining_recovery']['initial_state_sha256'][arm]
  group='seed42_reused' if seed==42 else '5080_new' if actualkey=='5080new' else '5090_original_completed' if actualkey=='5090old' else '5090_recovery'
  rows.append({'dataset':dataset,'seed':seed,'model':arm,'group':group,'completed_epochs':n,'best_epoch':best['epoch'],'stopped_early':summary['stopped_early'],
   'optimizer_steps':steps,'train_targets_per_epoch':inputs['populations']['train']['target_count'],'validation_targets_per_replay':ep['selected']['count'],'endpoint_replays':2,
   'initial_state_sha256':summary['initial_state_sha256'],'selected':ep['selected'],'last':ep['last'],'first40':ep['first40'],'last30':ep['last30'],
   'parameters':summary['parameter_count'],'elapsed_seconds':summary['elapsed_seconds'],'mean_epoch_seconds':statistics.mean(x['elapsed_seconds'] for x in timing['epochs']),
   'peak_allocated_bytes':summary['cuda_peak_memory_allocated_bytes'],'contract_sha256':canonical(actualcontract),'source_files_sha256':actualcontract['source']['files_sha256'],
   'source':str(run.relative_to(ROOT))})
  exps[arm]=exp;eps[arm]=ep;stepsjob+=steps
 for a,b in combinations(ARMS,2):
  n=min(len(exps[a]['train']),len(exps[b]['train']));assert exps[a]['train'][:n]==exps[b]['train'][:n]
 assert len({x['batch_order_sha256'] for exp in exps.values() for x in exp['validation']})==1 and paired['batch_prefix_equal']
 comparisons={}
 for ref in ARMS:
  if ref=='titantpp_local_detail':continue
  checks=gates(eps[ref],eps['titantpp_local_detail'],contract['comparison']['acceptance'])
  assert checks==paired['comparisons'][ref]['checks'],(jobkey,ref)
  assert paired['comparisons'][ref]['verified'] and paired['comparisons'][ref]['accepted']==all(checks.values())
  comparisons[ref]={'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'accepted':all(checks.values())}
 job_results[jobkey]={'execution_audit':'passed','batch_prefix_equal':True,'optimizer_steps':stepsjob,'comparisons':comparisons}
 all_endpoints[jobkey]=eps

assert len(rows)==54 and len({(x['dataset'],x['seed'],x['model']) for x in rows})==54
counts=Counter(r['group'] for r in rows)
assert counts=={'seed42_reused':18,'5080_new':24,'5090_original_completed':8,'5090_recovery':4}
partition=read(TERM/'run/partition_summary.json')
assert partition['datasets']['insta_market_basket/seed_62']==read(TERM/'run/insta_market_basket/seed_62/paired_comparison.json')
assert partition['new_training_arms']==4 and partition['new_endpoint_replays']==partition['endpoint_replays']==8
assert partition['reused_seed62_arms']==2 and partition['reused_seed62_replays']==4 and partition['combined_seed62_endpoint_replays']==12
assert partition['total_optimizer_steps']==sum(r['optimizer_steps'] for r in rows if r['group']=='5090_recovery')==4635986
assert partition['combined_seed62_optimizer_steps']==job_results['insta_market_basket/seed_62']['optimizer_steps']==6969536
assert partition['held_out_test_evaluated'] is False and partition['contract_sha256']==canonical(c)
baseline5080=read(ROOT/SOURCES['5080new'][0]/'terminal_audit.json')
assert baseline5080['status']=='passed' and baseline5080['optimizer_steps']==sum(r['optimizer_steps'] for r in rows if r['group']=='5080_new')

def stat(values):
 assert len(values)==3 and all(math.isfinite(x) for x in values)
 return {'n':3,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'by_seed':dict(zip([42,52,62],values))}
aggregates=[];paired_changes=[]
for dataset in identity_by_dataset:
 for arm in ARMS:
  group=sorted([r for r in rows if r['dataset']==dataset and r['model']==arm],key=lambda r:r['seed'])
  assert [r['seed'] for r in group]==[42,52,62]
  v={'dataset':dataset,'model':arm,'metrics':{k:stat([r['selected'][k] for r in group]) for k in ['qty_rmse','qty_mae','time_nll']},
     'completed_epochs':{r['seed']:r['completed_epochs'] for r in group},'best_epochs':{r['seed']:r['best_epoch'] for r in group},
     'parameters':stat([r['parameters'] for r in group]),'mean_epoch_seconds':stat([r['mean_epoch_seconds'] for r in group]),
     'training_elapsed_seconds':stat([r['elapsed_seconds'] for r in group]),'peak_allocated_bytes':stat([r['peak_allocated_bytes'] for r in group]),
     'stability':{w:{k:stat([r[w][k] for r in group]) for k in ['mean','sd']} for w in ['first40','last30']},'strata':{}}
  for kind in ['quantity','history','additional_history']:
   cells=group[0]['selected'][kind+'_cells'];v['strata'][kind]=[]
   for i,cell in enumerate(cells):
    assert all(r['selected'][kind+'_cells'][i]['count']==cell['count'] for r in group)
    v['strata'][kind].append({'bin':i,'count':cell['count'],'boundaries':group[0]['selected'][kind+'_boundaries'],
      'metrics':{k:stat([r['selected'][kind+'_cells'][i][k] for r in group]) for k in ['qty_rmse','qty_mae','time_nll']} if cell['count'] else None})
  aggregates.append(v)
 for seed in [42,52,62]:
  endpoints=all_endpoints[f'{dataset}/seed_{seed}'];b=endpoints['titantpp']['selected'];l=endpoints['titantpp_local_detail']['selected']
  paired_changes.append({'dataset':dataset,'seed':seed,'rmse_change_percent':100*(l['qty_rmse']/b['qty_rmse']-1),
   'mae_change_percent':100*(l['qty_mae']/b['qty_mae']-1),'time_nll_change':l['time_nll']-b['time_nll'],
   'quantity_rank_among_six':1+sum(x['selected']['qty_rmse']<l['qty_rmse'] for x in endpoints.values()),
   'failed_prespecified_checks':job_results[f'{dataset}/seed_{seed}']['comparisons']['titantpp']['failed_checks']})
audit={'status':'passed','scope':'54 retained fits and 108 selected/last validation replays; JSON evidence audited without data/checkpoint loading',
 'checked_at_utc':datetime.now(timezone.utc).isoformat(),'source_receipts':source_receipts,'conditions':54,'endpoint_replays':108,
 'groups':{g:{'conditions':n,'endpoint_replays':2*n,'optimizer_steps':sum(r['optimizer_steps'] for r in rows if r['group']==g)} for g,n in counts.items()},
 'total_recorded_optimizer_steps':sum(r['optimizer_steps'] for r in rows),'recovery_terminal':{'completed_at_unix':ds['completed_at_unix'],'tmux_and_owned_workers_exited':True,'fresh_arm_pids':pids,'predecessor_lineage_files_verified':61,'original_checkpoint_hashes_preserved':16,'new_checkpoint_hashes_recorded':8},
 'validation_only':True,'execution_success_is_performance_gate_pass':False,'jobs':job_results,'evidence_sha256':selected_file_hashes,
 'limitations':['Three seeds; sample SD is run-to-run spread, not a confidence interval or proof of significance.',
 'No held-out evaluation; no hyperparameter tuning or new workload was performed.',
 'B/local shared-initialization checks come from frozen qualification and initialization receipts; this audit verifies SHA lineage and does not deserialize checkpoints.',
 'The original RMTPP seed62 attempt stalled before any saved epoch/checkpoint; partial unsaved updates and native internal cause remain unknown. It is not an additional completed fit and is not erased.',
 'Earlier stopped predecessor wrappers are retained; completed admitted partitions are reused. The old Intermittent deadline issue and the RMTPP native stall are different incidents.',
 'Process isolation is containment, not a proven native-library bug fix. THP frozen-runtime nondeterministic attention warning is retained; endpoint replay agreement passes.',
 'B/local contrast changes parameter capacity; no causal separation of history design and extra capacity.',
 'Elapsed training and mean epoch timing exclude external queuing, failed attempts and separate endpoint replay time. Compare costs only within the same dataset/host.']}
OUT.mkdir(parents=True,exist_ok=True)
if '--verify-only' in sys.argv:
 for name,obj in [('terminal_audit.json',audit),('condition_results.json',rows),('three_seed_summary.json',aggregates),('paired_seed_changes.json',paired_changes)]:
  saved=read(OUT/name)
  if name=='terminal_audit.json':saved.pop('checked_at_utc');obj=dict(obj);obj.pop('checked_at_utc')
  assert saved==json.loads(json.dumps(obj,ensure_ascii=False,allow_nan=False)),name
 print(json.dumps({'status':'recomputed_outputs_match','conditions':54,'replays':108}))
 sys.exit(0)
for name,obj in [('terminal_audit.json',audit),('condition_results.json',rows),('three_seed_summary.json',aggregates),('paired_seed_changes.json',paired_changes)]:
 with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
with (TERM/'terminal_audit.json').open('x') as f:json.dump({'status':'passed','new_conditions':4,'new_endpoint_replays':8,'new_optimizer_steps':4635986,'combined_seed62_conditions':6,'combined_seed62_endpoint_replays':12,'combined_seed62_optimizer_steps':6969536,'campaign_audit':str((OUT/'terminal_audit.json').relative_to(ROOT)),'predecessor_lineage_verified':61,'held_out_evaluated':False},f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','groups':audit['groups'],'recorded_optimizer_steps':audit['total_recorded_optimizer_steps'],'paired_changes':paired_changes,'output':str(OUT)},ensure_ascii=False,indent=2))
