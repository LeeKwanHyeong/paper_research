"""Local source/equation/cost audit. No GPU, training, replay or raw data access."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SOURCES = {}

def digest(b):
    return hashlib.sha256(b).hexdigest()

def read(path):
    p = ROOT / path
    b = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = digest(b)
    return b

def load(path):
    return json.loads(read(path))

def canonical(x):
    return digest(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())

def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def table(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def stats(xs):
    return {'n':len(xs), 'mean':statistics.mean(xs),
            'sample_sd':statistics.stdev(xs) if len(xs)>1 else None}

contract = load('search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json')
assert canonical(contract) == 'eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d'
assert len(contract['source']['files']) == 97
assert canonical(contract['source']['files']) == contract['source']['files_sha256'] == 'e21c45b5f095d258e62998ce818b07bef38ff2cfff6bc408b3594bd1adba9265'
paths = ['models/TPPs/CountAwareTitanCoreAblation.py', 'models/TPPs/CountAwareTitanMultiLagDetail.py',
         'models/TPPs/CountAwareTPP.py', 'models/Titan/backbone.py', 'models/Titan/common/memory.py',
         'models/TPPs/positive_integer_time.py', 'paper/scripts/count_aware_tpp_backbone/core.py',
         'paper/scripts/count_aware_tpp_backbone/training.py']
source_checks = []
for p in paths:
    actual = digest(read(p)); expected = contract['source']['files'][p]
    source_checks.append({'path':p, 'expected':expected, 'actual':actual, 'match':actual==expected})
assert all(x['match'] for x in source_checks)

# Small CPU algebra check of the paper equation against the exact frozen module.
# Neither checkpoint nor model training is involved.
import torch
from torch.nn import functional as F
from models.TPPs.CountAwareTitanCoreAblation import HistoryCorrection
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices
torch.set_num_threads(1)
torch.manual_seed(20260930)
before = torch.random.get_rng_state().clone()
module = HistoryCorrection(64, 'mlp').double()
assert torch.equal(before, torch.random.get_rng_state())
assert sum(p.numel() for p in module.parameters()) == 6144
hidden = torch.randn(2,140,64,dtype=torch.float64)
valid = torch.ones(2,140,dtype=torch.bool); valid[0,3] = False
write_mask = valid.clone(); write_mask[0,20] = False; write_mask[1,-1] = False
observed = valid & write_mask
with torch.no_grad():
    assert torch.count_nonzero(module(hidden,valid,memory_write_mask=write_mask)) == 0
    for layer in module.output_projections: layer.weight.copy_(torch.randn_like(layer.weight)*0.01)
    source, available = lag_source_indices(valid,memory_write_mask=write_mask,mode='local')
    reference = torch.zeros_like(hidden)
    thresholds = (1,2,4,8,16,32,64,128)
    for row in range(2):
        count = 0; predecessor = None
        for i in range(140):
            if valid[row,i] and not observed[row,i]: count=0
            if not observed[row,i]: continue
            count += 1
            for branch, threshold in enumerate(thresholds):
                eligible = count > threshold
                assert bool(available[row,i,branch]) == eligible
                if eligible:
                    assert int(source[row,i,branch]) == predecessor
                    pair = torch.cat((hidden[row,i],hidden[row,predecessor]))
                    reference[row,i] += module.output_projections[branch](F.gelu(module.input_projections[branch](pair),approximate='none'))/8
            predecessor = i
    actual = module(hidden,valid,memory_write_mask=write_mask)
    error = float((actual-reference).abs().max())
    assert torch.allclose(actual,reference,atol=1e-12,rtol=1e-12)
    changed = hidden.clone(); changed[~observed] = float('nan')
    assert torch.equal(actual,module(changed,valid,memory_write_mask=write_mask))
    future = hidden.clone(); future[:,80:] *= 11
    assert torch.equal(actual[:,:80],module(future,valid,memory_write_mask=write_mask)[:,:80])

analysis = load('reports/titantpp_first_analysis_20260930_v1/analysis.json')
assert analysis['evaluation_scope'] == 'validation_only'
rows = []
for x in analysis['core_and_external_conditions']:
    if x['arm'] != 'titantpp_history_mlp': continue
    assert x['selected']['evaluation_scope']=='validation_only' and x['selected']['held_out_test_evaluated'] is False
    summary = load(x['source']+'/summary.json')
    assert summary['evaluation_scope']=='validation_only' and summary['held_out_test_evaluated'] is False
    assert math.isclose(summary['elapsed_seconds'],x['fit_elapsed_seconds'],abs_tol=1e-9)
    eligible = not x['timing_interference_recorded']
    reason = '' if eligible else ('recovery_post_resume_timer_and_parallel' if x['seed']==62 else 'parallel_execution')
    rows.append({'dataset':x['dataset'], 'seed':x['seed'], 'host':x['host'], 'parameters':x['parameters'],
                 'completed_epochs':x['completed_epochs'], 'recorded_fit_seconds':x['fit_elapsed_seconds'],
                 'standalone_eligible':eligible, 'exclusion_reason':reason,
                 'amortized_seconds_per_epoch':x['fit_elapsed_seconds']/x['completed_epochs'] if eligible else None,
                 'peak_allocated_mib':x['peak_allocated_bytes']/2**20, 'source':x['source']})
assert len(rows)==9
table('current_mlp_costs.csv',rows)
summary_costs = []
for dataset in ('yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket'):
    r=[x for x in rows if x['dataset']==dataset and x['standalone_eligible']]
    summary_costs.append({'dataset':dataset,'eligible_seeds':[x['seed'] for x in r],
                          'amortized_seconds_per_epoch':stats([x['amortized_seconds_per_epoch'] for x in r]),
                          'peak_allocated_mib':stats([x['peak_allocated_mib'] for x in r])})

historic_base='paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/'
historic_csv=list(csv.DictReader(read(historic_base+'historical_cost.csv').decode().splitlines()))
runtime=load(historic_base+'runtime_cost.json')
manifest=load(historic_base+'source_manifest.json')
history=[]
for x in historic_csv:
    if x['dataset'] not in ('yellow_trip_hourly','intermittent_frozen_5000'):continue
    a=float(x['b0_elapsed_seconds'])/int(x['b0_completed_epochs'])
    b=float(x['b1_elapsed_seconds'])/int(x['b1_completed_epochs'])
    assert math.isclose(b/a,float(x['b1_b0_epoch_cost_ratio']),rel_tol=1e-12)
    v=next(r for r in runtime if r['dataset']==x['dataset'])
    history.append({'dataset':x['dataset'],'old_b0_seconds_per_epoch':a,'old_mac_seconds_per_epoch':b,
                    'old_mac_over_b0_ratio':b/a, 'old_b0_target_outputs_median_ms':v['b0']['steady_forward_median_seconds']*1000,
                    'old_mac_target_outputs_median_ms':v['b1']['steady_forward_median_seconds']*1000,
                    'old_b0_peak_allocated_mib':v['b0']['peak_allocated_mib'],
                    'old_mac_peak_allocated_mib':v['b1']['peak_allocated_mib']})
table('historical_costs.csv',history)
historical_runs=[]
old_root=Path('search_artifacts/count_aware_b012_seed42_screening_e300_20260828_recovery1')
for dataset in ('yellow_trip_hourly','intermittent_frozen_5000'):
    for arm, column in [('titantpp','b0'),('titantpp_titans_mac','b1')]:
        base=old_root/'shards'/dataset/arm/'titan_b012_screening'
        if not (ROOT/base).exists(): base=old_root/dataset/'titan_b012_screening'
        old_contract=load(base/'launch_contract.json')
        old_summary=load(base/'runs'/arm/'count_only_log_regression/seed_42/summary.json')
        assert old_summary['evaluation_scope']=='validation_only' and old_summary['held_out_test_evaluated'] is False
        csv_row=next(x for x in historic_csv if x['dataset']==dataset)
        assert math.isclose(old_summary['elapsed_seconds'],float(csv_row[column+'_elapsed_seconds']),abs_tol=1e-9)
        assert old_summary['completed_epochs']==int(csv_row[column+'_completed_epochs'])
        historical_runs.append({'dataset':dataset,'arm':arm,'source':str(base),
          'declared_execution_host':old_contract.get('execution_host'),
          'execution_role':old_contract.get('execution_role'),
          'source_revision':old_summary.get('source_revision'),
          'physical_gpu_uuid_proven_by_this_record':False})
read('reports/titans_mac_reuse_audit_20260928_v1/mac_current_compatibility.csv')
git_evidence=[]
for rev,path in [(manifest['git_revision'],'paper/scripts/analyze_count_aware_titantpp_mac.py'),
                  ('08e59880cd61cbd27cec40aa04636452b87bebfc','paper/scripts/count_aware_tpp_backbone/training.py')]:
    b=subprocess.check_output(['git','show',rev+':'+path],cwd=ROOT)
    sha=digest(b)
    git_evidence.append({'revision':rev,'path':path,'sha256':sha})
    if 'analyze_' in path:
        assert sha==next(x['sha256'] for x in manifest['files'] if x['path']==path)

effects=[]
for d in ('yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket'):
    idx={x['model']:x for x in analysis['aggregates'] if x['dataset']==d}
    for other in ('titantpp','titantpp_local_detail'):
        for metric in ('qty_mae','qty_rmse','time_nll'):
            a=idx['titantpp_history_mlp']['selected'][metric]['mean']; b=idx[other]['selected'][metric]['mean']
            effects.append({'dataset':d,'comparison':other,'metric':metric,'mlp_mean':a,'other_mean':b,'reduction_percent':100*(1-a/b)})
table('design_effects.csv',effects)
write('efficiency_audit.json',{'evaluation_scope':'validation_only', 'current_mlp_costs':summary_costs,
      'historical_costs':history, 'historical_run_metadata':historical_runs, 'direct_current_mlp_vs_mac_speedup_established':False,
      'direct_memory_saving_established':False,'historical_source_evidence':git_evidence,
      'short_paired_gpu_profile':{'necessary_for_new_direct_speed_claim':True,'approved':False,'executed':False},
      'full_mac_accuracy_training':{'needed_for_matched_accuracy_claim':True,'not_needed_for_step_latency_only':True,'approved':False}})
write('verification.json',{'status':'passed','contract_sha256':canonical(contract),
      'source_closure_sha256':contract['source']['files_sha256'],'source_closure_entry_count':97,
      'locally_verified_method_source_files':source_checks,'full_remote_binary_audit_performed':False,
      'cpu_equation_checks':{'rng_preserved':True,'zero_initial_correction':True,'parameter_count':6144,
        'explicit_equation_max_absolute_error':error,'all_branches_read_immediate_predecessor':True,
        'fixed_divisor_eight':True,'padding_skipped_and_withheld_resets_eligibility':True,
        'invalid_input_nan_isolation':True,'future_input_invariance':True},
      'current_mlp_cost_rows':len(rows),'new_gpu_work':False,'new_training':False,'checkpoint_replay':False,
      'raw_data_read':False,'held_out_read':False,'scheduler_created':False,'scientific_source_edited':False,
      'historical_git_evidence':git_evidence})
write('sources.json',SOURCES)
print(json.dumps({'status':'passed','current_mlp_costs':summary_costs,'cpu_equation_max_error':error},ensure_ascii=False,indent=2))
