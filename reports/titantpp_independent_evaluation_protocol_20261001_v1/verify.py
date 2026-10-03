"""Verify the design inventory and metric arithmetic without held-out inference."""
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    checks = {}
    p = load(OUT / 'protocol.json')
    m = load(OUT / 'checkpoint_manifest.json')
    rows = m['rows']
    checks['84_unique_fixed_selected_conditions'] = len(rows) == len({(r['dataset'],r['model'],r['seed']) for r in rows}) == 84
    checks['complete_cartesian_panel'] = {(r['dataset'],r['model'],r['seed']) for r in rows} == set(itertools.product(p['datasets'],p['models'],p['seeds']))
    checks['no_exploratory_variant_substitution'] = all('active_norm' not in r['model'] and r['endpoint']=='selected' for r in rows)
    checks['tensor_identifiers_bound'] = all(re.fullmatch('[0-9a-f]{64}',r['state_tensor_sha256']) for r in rows)
    checks['48_local_36_recorded_path_missing'] = (sum(r['local_binary_present'] for r in rows),sum(not r['local_binary_present'] for r in rows)) == (48,36)
    checks['missing_only_original_three_datasets_four_legacy_models'] = all(r['model'] in ['rmtpp','thp','nhp','sahp'] and r['dataset']!='raf_spare_parts' for r in rows if not r['local_binary_present'])
    checks['present_file_hashes_match'] = all(sha(ROOT/r['checkpoint_path'])==r['checkpoint_file_sha256'] for r in rows if r['local_binary_present'])
    checks['absent_files_not_assigned_invented_hash'] = all(r['checkpoint_file_sha256'] is None and not (ROOT/r['checkpoint_path']).exists() for r in rows if not r['local_binary_present'])
    checks['csv_json_rows_match'] = [(r['dataset'],r['model'],int(r['seed']),int(r['selected_epoch'])) for r in csv.DictReader((OUT/'checkpoint_manifest.csv').open())] == [(r['dataset'],r['model'],r['seed'],r['selected_epoch']) for r in rows]
    stats = {}
    identity_checks = []
    for r in rows:
        j=load(ROOT/r['validation_replay_path']);v=j['selected']
        identity_checks.append(v['evaluation_scope']=='validation_only' and v['held_out_test_evaluated'] is False and j['best_epoch']==r['selected_epoch'] and v['state_sha256']==r['state_tensor_sha256'])
        stats.setdefault((r['dataset'],r['model']),[]).append(v['qty_rmse'])
    checks['all_reused_endpoints_validation_and_selected_identity_match'] = all(identity_checks)
    anchors = {ds:min((model for model in p['models'] if model!=p['representative']),key=lambda model:statistics.mean(stats[(ds,model)])) for ds in p['datasets']}
    checks['anchors_match_existing_validation_three_seed_means'] = anchors == p['fixed_validation_selected_anchors']
    d=load(OUT/'dataset_manifest.json')
    checks['split_and_data_byte_identity_match_frozen_contracts'] = all(sha(ROOT/i['path'])==i['sha256'] for ds in d['datasets'] for i in ds['identities'].values())
    sources=load(OUT/'sources.json')['local_files_sha256']
    checks['181_source_records_unchanged'] = len(sources)==181 and all(sha(ROOT/f)==s for f,s in sources.items())
    preserved=load(OUT/'preserved_inputs.json')
    checks['manuscript_protocol_and_existing_scientific_edits_preserved'] = all(sha(ROOT/f)==s for f,s in preserved.items())
    access=load(OUT/'split_access_ledger.json')
    checks['all_four_access_statuses_explicit'] = len(access['datasets'])==4 and all(x['status']!='untouched_independent_test_verified' for x in access['datasets'])
    checks['no_evaluation_authorization_inferred'] = p['execution_authorized'] is False and p['status']=='design_complete_execution_not_ready'
    checks['bootstrap_draws_fixed_and_seeds_not_resampled'] = p['resampling']['draws']==10000 and not p['resampling']['seed_resampling']
    checks['taxi_time_blocks_not_target_count_blocks'] = p['resampling']['taxi_block_hours']==168 and p['resampling']['taxi_sensitivity_block_hours']==[24,336] and p['resampling']['unavailable_timestamp']=='no_target_order_fallback'
    checks['four_anchor_bonferroni_quantiles'] = p['resampling']['fixed_four_RMSE_anchor_interval_quantiles']==[0.05/4/2,1-0.05/4/2]
    # Synthetic arithmetic examples; these are not an implementation/test of a real evaluator.
    errors_a=[1.,1.];errors_b=[5.]
    expanded=errors_a*2+errors_b
    weighted_sse=sum(x*x for x in errors_a)*2+sum(x*x for x in errors_b)
    weighted_n=len(errors_a)*2+len(errors_b)
    checks['synthetic_cluster_multiplicity_preserves_event_weighting'] = math.isclose(math.sqrt(weighted_sse/weighted_n),math.sqrt(statistics.mean(x*x for x in expanded))) and not math.isclose(math.sqrt(weighted_sse/weighted_n),statistics.mean([1.,1.,5.]))
    rmses=[1.,2.,3.]
    checks['synthetic_seed_mean_RMSE_not_sqrt_mean_MSE'] = statistics.mean(rmses)==2 and not math.isclose(statistics.mean(rmses),math.sqrt(statistics.mean(x*x for x in rmses)))
    checks['synthetic_sample_SD_not_SE'] = statistics.stdev(rmses)==1 and not math.isclose(statistics.stdev(rmses),statistics.stdev(rmses)/math.sqrt(3))
    docs=[OUT/'protocol.md',OUT/'README.md',ROOT/'reports/titantpp_baseline_reset_20261001_v1/README.md',ROOT/'reports/titantpp_manuscript_integration_20261001_v1/README.md']
    broken=[]
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if target.startswith(('https://','http://','#')):continue
            target=target.split('#',1)[0]
            if (doc.parent/target).resolve()==OUT/'verification.json':continue  # Generated by this invocation.
            if not (doc.parent/target).exists():broken.append({'document':str(doc.relative_to(ROOT)),'target':target})
    checks['local_document_links_resolve'] = not broken
    text=(OUT/'protocol.md').read_text()
    checks['document_numbers_and_boundary_match_design'] = all(s in text for s in ['84','48','36','10,000','168','0.625/99.375','실행 준비 미완료','실행·성능 열람 승인 필요'])
    report={'status':'passed' if all(checks.values()) else 'failed','checked_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'passed_checks':sum(checks.values()),'total_checks':len(checks),'broken_links':broken,'inventory':m['summary'],'execution_readiness':False,'independence_eligibility':'not_established','real_data_confidence_intervals_computed':False,'checkpoint_deserialization_this_work':False,'heldout_content_read':False,'new_gpu_or_forward_calls':False,'new_remote_observation':False,'scope':'local identity/byte verification, design consistency, synthetic metric arithmetic; prior CPU audits reused', 'outputs_sha256':{n:sha(OUT/n) for n in ['protocol.md','protocol.json','checkpoint_manifest.json','checkpoint_manifest.csv','dataset_manifest.json','split_access_ledger.json','sources.json','build_inventory.py','verify.py','README.md']}}
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':f"{sum(checks.values())}/{len(checks)}",'failed':[k for k,v in checks.items() if not v],'inventory':m['summary']},ensure_ascii=False))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
