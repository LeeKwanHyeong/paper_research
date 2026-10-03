from pathlib import Path
import csv,json,statistics
from paper.scripts.observed_slot_parallel_common import sha_file,write_json
from paper.scripts.local_detail_benchmark_contract import ARMS,ASSIGNMENTS
base=Path('reports/local_detail_benchmark_preparation_20260922_v1')
sources={};old={}
def load_summary(path):
 p=Path(path);d=json.loads(p.read_text())
 assert d['evaluation_scope']=='validation_only' and d['held_out_test_evaluated'] is False
 sources[str(p)]={'sha256':sha_file(p),'completed_epochs':d['completed_epochs'],'elapsed_seconds':d['elapsed_seconds'],'old_head':'legacy_clamped_rmtpp','seconds_per_epoch':d['elapsed_seconds']/d['completed_epochs']}
 return d['elapsed_seconds']/d['completed_epochs']
for ds,root in [('yellow_trip_hourly','search_artifacts/count_aware_taxi_t0_t1_e300_20260824/yellow_trip_hourly/t0_common_control'),('insta_market_basket','search_artifacts/count_aware_instacart_t0_e300_20260824_recovery1/insta_market_basket/t0_common_control')]:
 old[ds]={a:load_summary(Path(root)/'runs'/a/'count_only_log_regression/seed_42/summary.json') for a in ARMS if a!='titantpp_local_detail'}
inter={}
p=Path('paper/results/intermittent_log_backbone_control_20260811/source_5080_baselines/run_summaries.csv')
for r in csv.DictReader(p.open()):
 if r['seed']=='42' and r['backbone'] in ('rmtpp','thp'):
  inter[r['backbone']]=float(r['elapsed_seconds'])/int(r['completed_epochs'])
sources[str(p)]={'sha256':sha_file(p),'scope':'historical validation-only timing; old source/runtime'}
for arm,root in [('nhp','count_aware_external_t0_nhp_sahp_e300_20260820'),('sahp','count_aware_external_t0_sahp_all_e300_20260821_5090'),('titantpp','count_aware_titan_memory_backbone_screening_e300_20260818')]:
 inter[arm]=load_summary(Path('search_artifacts')/root/'runs'/arm/'count_only_log_regression/seed_42/summary.json')
old['intermittent_frozen_5000']=inter
proxy={}
p=Path('search_artifacts/local_gate_execution_20260922_v1/monitor/20260922T094459Z/terminal_5080/run')
for ds in ('yellow_trip_hourly','intermittent_frozen_5000'):
 vals=[]
 for arm in ('titantpp_local_time_gate','titantpp_local_quantity_gate'):
  f=p/ds/'runs'/arm/'count_only_log_regression/seed_42/summary.json';d=json.loads(f.read_text())
  vals.append(d['elapsed_seconds']/d['completed_epochs'])
  sources[str(f)]={'sha256':sha_file(f),'completed_epochs':d['completed_epochs'],'elapsed_seconds':d['elapsed_seconds'],'head':'current_recorded_positive_integer','scope':'nearby local-gate runtime proxy, not exact B/local measurement'}
 proxy[ds]=statistics.mean(vals)
p=Path('search_artifacts/local_gate_execution_20260922_v1/monitor/20260922T110622Z/5090.json')
d=json.loads(p.read_text())['datasets']['insta_market_basket']['arms']['titantpp_local_time_gate']['summary.json']
proxy['insta_market_basket']=d['elapsed_seconds']/d['completed_epochs'];sources[str(p)]={'sha256':sha_file(p),'field':'datasets.insta_market_basket.arms.titantpp_local_time_gate.summary.json','scope':'completed current-head nearby local-gate runtime proxy'}
rows=[]
for host,datasets in ASSIGNMENTS.items():
 for ds in datasets:
  ref=old[ds]['titantpp'];extra=max(0.,proxy[ds]-ref);ratio=proxy[ds]/ref
  for arm in ARMS:
   if arm in ('titantpp','titantpp_local_detail'): low=high=proxy[ds];historical=None
   else:
    historical=old[ds][arm];low,high=sorted((historical+extra,historical*ratio))
   rows.append({'host':host,'dataset':ds,'arm':arm,'historical_seconds_per_epoch':historical,
    'proxy_seconds_per_epoch':proxy[ds],'projected_seconds_per_epoch_low':low,'projected_seconds_per_epoch_high':high,
    'hours_if_all_300_epochs':[low*300/3600,high*300/3600]})
scenarios=[]
for epochs in [41,100,300]:
 hosts={h:[sum(r['projected_seconds_per_epoch_'+bound]*epochs/3600 for r in rows if r['host']==h) for bound in ('low','high')] for h in ASSIGNMENTS}
 scenarios.append({'completed_epochs_assumed_per_arm':epochs,'host_hours':hosts,'parallel_wall_hours':[max(v[i] for v in hosts.values()) for i in range(2)]})
result={'schema':'local_detail_benchmark_cost_estimate_v1','native_current_head_benchmark_timing_measured':False,
 'method':'Two sensitivity scenarios: legacy per-epoch cost plus current-proxy minus legacy-Titan cost; or legacy cost scaled by current-proxy/legacy-Titan. B/local use nearby measured local-gate cost. These are scenarios, NOT confidence intervals or measured new-backbone ETA.',
 'limits':'excludes qualification, deployment, validation replays, queue waiting and filesystem slowdown; runtimes/hardware and head implementations changed; early stopping unknown',
 'sources':sources,'rows':rows,'scenarios':scenarios,
 'proposed_deadline_hours':240,'deadline_is_estimate':False,'native_qualification_required_before_training':True}
write_json(base/'cost_estimate.json',result)
print(json.dumps(scenarios,indent=2))
