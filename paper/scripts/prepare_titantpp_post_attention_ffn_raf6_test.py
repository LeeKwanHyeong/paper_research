"""Local preparation and pure RAF Current/Temporal FFN frozen-result comparison.

The pure comparison performs no inference, model loading, training or registry generation.
Test is used for eligibility, never for checkpoint selection; the prepare CLI delegates only local preparation.
The calling operational helper must verify every native receipt/worker/gate SHA.
"""
from __future__ import annotations
import math
import statistics
from paper.scripts.compare_titantpp_backbone_loop import (
    METRICS, SEEDS, SPLITS, _validated_group, require, stats, verify_registry,
)

CURRENT = 'titantpp_cnn_ffn_current_v1'
TEMPORAL = 'titantpp_cnn_ffn_temporal_v1'
ARMS=(CURRENT,TEMPORAL)
SELECTION_RULE = 'passing_candidates:validation_rmse_then_mae_then_arm'
NATIVE_RUNTIME = {'host':'5090','gpu_uuid':'GPU-c9adc246-ef66-7906-bc07-6e152fa9952f',
                  'python':'3.12.13','torch':'2.11.0+cu130','cudnn':92000,'threads':4}
INTERVENTION_LIMITATION = ('Original QKV576 and FFN W1/W2 remain; each arm adds1152 parameters in the final FFN. '
    'Temporal uses current/previous/second-previous observed prefix content, Current repeats current prefix with the same availability. '
    'Current already contains attention history; equal parameter counts do not imply equal function space or optimization. '
    'Common weights train jointly and downstream memory retrieval may change; no isolated causal attribution.')
RUNTIME_LIMITATION = ('Historical CNN/external:5080 cuDNN91900; new candidate:5090 cuDNN92000. '
    'Source/Runtime differences remain; no fresh same-host CNN fit or reevaluation.')


def normalized_native_row(receipt, host):
    require(receipt.get('status') == 'complete' and receipt.get('native_runtime_checked') is True,
            'Native completed receipt required')
    require(host == '5090' and receipt['arm'] in ARMS, 'Foreign RAF6 result')
    from paper.scripts.evaluate_titantpp_post_attention_ffn_frozen import verify_physical_counts
    verify_physical_counts(receipt)
    counts = receipt.get('physical_parameter_counts', {})
    require(counts.get('active_local_cnn_parameters')==576 and counts.get('active_value_cnn_parameters')==192
        and counts.get('post_attention_ffn_added_parameters')==1152
        and counts.get('actual_registered_parameters')==105311
        and counts.get('actual_active_parameters_before_evaluation_freeze')==80579
        and counts.get('original_qkv_filters_retained') is True, 'Candidate physical parameter identity missing')
    runtime = receipt['runtime']
    return {'dataset':receipt['dataset'],'model':receipt['arm'],'split':receipt['split'],
        'seed':receipt['seed'],'status':'complete','validation_selected_checkpoint_verified':True,
        'selected_epoch':receipt['selected_epoch'],'checkpoint_sha256':receipt['checkpoint_sha256'],
        'metrics':{m:receipt['metrics'][m] for m in METRICS},'count':receipt['metrics']['count'],
        'population':{k:receipt[k] for k in ('target_identity_sha256','truth_sha256','time_sha256')},
        'runtime':{'host':host,'gpu_uuid':runtime['gpu']['uuid'],'python':runtime['python'],
                   'torch':runtime['torch'],'cudnn':runtime['cudnn'],'threads':runtime['threads']},
        'local_cnn_parameter_count':576,'original_local_cnn_parameter_count':576,'ffn_added_parameter_count':1152}


def _compare_arm(registry, candidate_rows, arm):
    require(registry.get('status') == 'verified' and registry.get('dataset') == 'raf_spare_parts'
            and registry.get('seeds') == list(SEEDS), 'Verified RAF three-seed registry required')
    candidate = _validated_group(candidate_rows, registry)
    require(all(r['model'] == arm and r['runtime'] == NATIVE_RUNTIME
            and r.get('local_cnn_parameter_count') == 576 and r.get('original_local_cnn_parameter_count') == 576
            and r.get('ffn_added_parameter_count') == 1152
            for r in candidate.values()), 'Candidate identity/Runtime/parameter count differs')
    history_rows = [r for r in registry['rows'] if r['model'] == 'historical_CNN']
    history = {(r['split'],r['seed']):r for r in history_rows}
    expected = {(split,seed) for split in SPLITS for seed in SEEDS}
    require(len(history_rows) == len(history) == 6 and set(history) == expected,
            'Historical CNN3 required on both splits')
    external_models = sorted({r['model'] for r in registry['rows'] if r['model'] != 'historical_CNN'})
    require(bool(external_models), 'External references missing')
    # Reference checkpoint identity is identical across splits; the registry seal
    # additionally binds native source/input/population proof before this function.
    for model in ('historical_CNN',*external_models):
        rows = [r for r in registry['rows'] if r['model'] == model]
        indexed = {(r['split'],r['seed']):r for r in rows}
        require(len(rows) == len(indexed) == 6 and set(indexed) == expected, 'Reference seed set differs')
        for seed in SEEDS:
            v,t = indexed[('validation',seed)],indexed[('test',seed)]
            require((v['selected_epoch'],v['checkpoint_sha256']) == (t['selected_epoch'],t['checkpoint_sha256']),
                    'Reference selected checkpoint differs across splits')
    summaries,differences,checks = {},[],[]
    for split in SPLITS:
        summaries[split] = {}
        for metric in METRICS:
            external_stats = {model:stats([r['metrics'][metric] for r in registry['rows']
                if r['model'] == model and r['split'] == split]) for model in external_models}
            best = min(v['mean'] for v in external_stats.values())
            winners = sorted(m for m,v in external_stats.items() if v['mean'] == best)
            frontier = registry['external_frontier'][split][metric]
            require(frontier['mean'] == best and sorted(frontier['models']) == winners
                and len(frontier['models']) == len(winners)
                and frontier['sample_sd'] == {m:external_stats[m]['sample_sd'] for m in winners},
                'External metricwise frontier differs from seed rows')
            historical = registry['historical_cnn'][split][metric]
            require(historical == stats([history[(split,s)]['metrics'][metric] for s in SEEDS]),
                    'Historical aggregate differs from seed rows')
            cs = stats([candidate[(split,s)]['metrics'][metric] for s in SEEDS])
            passed = cs['mean'] < historical['mean'] and cs['mean'] < best
            checks.append(passed)
            summaries[split][metric] = {'candidate':cs,'historical_CNN':historical,
                'external_frontier':frontier,'user_performance_pass':passed}
            for seed in SEEDS:
                refs = {'historical_CNN':history[(split,seed)]['metrics'][metric]}
                for model in winners:
                    matched = [r for r in registry['rows'] if (r['model'],r['split'],r['seed']) == (model,split,seed)]
                    require(len(matched) == 1, 'External seed reference missing/duplicate')
                    refs['external_frontier:'+model] = matched[0]['metrics'][metric]
                value = candidate[(split,seed)]['metrics'][metric]
                differences.append({'split':split,'seed':seed,'metric':metric,'candidate':value,
                    'differences':{name:{'reference':ref,'absolute_candidate_minus_reference':value-ref,
                        'relative_percent_candidate_minus_reference':100*(value-ref)/ref if ref else None}
                        for name,ref in refs.items()}})
    passed = all(checks)
    require(len(checks) == 4, 'Exactly four user criteria required')
    return {'status':'complete','dataset':registry['dataset'],'candidate_model':arm,
        'candidate_fit_count':3,'historical_cnn_reference_count':3,'summaries':summaries,
        'seed_differences':differences,'user_performance_gate':passed,'user_performance_check_count':4,
        'candidate_selection':{'selection_rule':SELECTION_RULE,'selected_candidate':arm if passed else None,
            'test_used_for_eligibility':True,'test_used_for_checkpoint_selection':False,
            'multiple_candidate_tie_break':False},
        'historical_cnn_retrained':False,'historical_cnn_reevaluated':False,'fresh_same_runtime_cnn_fit_count':0,
        'runtime_confound':RUNTIME_LIMITATION,'intervention_limitation':INTERVENTION_LIMITATION,
        'test_independent':False,'statistical_significance_established':False,'causal_effect_established':False}


def difference_stats(values):
    """Signed paired differences require finite values, not nonnegative errors."""
    require(len(values)==3 and all(type(v) in (float,int) and math.isfinite(v) for v in values),
            "Exactly three finite paired differences required")
    return {"mean":statistics.mean(values),"sample_sd":statistics.stdev(values)}


def compare_historical_rows(registry, candidate_rows):
    expected={(arm,split,seed) for arm in ARMS for split in SPLITS for seed in SEEDS}
    indexed={(r.get('model'),r.get('split'),r.get('seed')):r for r in candidate_rows}
    require(len(candidate_rows)==len(indexed)==12 and set(indexed)==expected, 'Exactly two candidates × three seeds × two splits required')
    result_by_arm={arm:_compare_arm(registry,[r for r in candidate_rows if r['model']==arm],arm) for arm in ARMS}
    passing=[arm for arm in ARMS if result_by_arm[arm]['user_performance_gate']]
    ranking=lambda arm:(result_by_arm[arm]['summaries']['validation']['qty_rmse']['candidate']['mean'],
                        result_by_arm[arm]['summaries']['validation']['qty_mae']['candidate']['mean'],arm)
    passing.sort(key=ranking)
    contrasts=[]
    for split in SPLITS:
        for seed in SEEDS:
            for metric in METRICS:
                left=indexed[(TEMPORAL,split,seed)]['metrics'][metric]
                right=indexed[(CURRENT,split,seed)]['metrics'][metric]
                contrasts.append({'split':split,'seed':seed,'metric':metric,'candidate':TEMPORAL,'reference':CURRENT,
                    'candidate_value':left,'reference_value':right,'absolute_candidate_minus_reference':left-right,
                    'relative_percent_candidate_minus_reference':100*(left-right)/right if right else None})
    mean_contrasts={split:{metric:difference_stats([indexed[(TEMPORAL,split,seed)]['metrics'][metric]
        - indexed[(CURRENT,split,seed)]['metrics'][metric] for seed in SEEDS]) for metric in METRICS} for split in SPLITS}
    both_lower={split:all(mean_contrasts[split][metric]['mean']<0 for metric in METRICS) for split in SPLITS}
    selected=passing[0] if passing else None
    attribution={'validation_both_metrics_lower':both_lower['validation'],
        'test_both_metrics_lower':both_lower['test'],
        'both_splits_both_metrics_lower':all(both_lower.values()),
        'mean_temporal_minus_current':mean_contrasts,
        'selected_candidate_interpretation':('pointwise_FFN_expansion_not_explicit_temporal_content_success'
            if selected==CURRENT else 'temporal_candidate_selected; see_separate_temporal_minus_current_evidence'
            if selected==TEMPORAL else 'no_candidate_passed_user_gate'),
        'causal_attribution_established':False}
    return {'status':'complete','dataset':registry['dataset'],'candidate_models':list(ARMS),
        'candidate_fit_count':6,'historical_cnn_reference_count':3,'comparison_model_count':9,
        'summaries':{arm:result_by_arm[arm]['summaries'] for arm in ARMS},
        'seed_differences':[{**r,'candidate_model':arm} for arm in ARMS for r in result_by_arm[arm]['seed_differences']],
        'primary_contrast':'Temporal−Current','primary_contrast_seed_differences':contrasts,
        'temporal_content_evidence':attribution,
        'user_performance_gate':{arm:result_by_arm[arm]['user_performance_gate'] for arm in ARMS},
        'user_performance_check_count_per_candidate':4,'user_performance_check_count':8,
        'candidate_selection':{'selection_rule':SELECTION_RULE,'selected_candidate':passing[0] if passing else None,
            'eligible_candidates':passing,'test_used_for_eligibility':True,'test_used_for_checkpoint_selection':False,
            'multiple_candidate_tie_break':True},
        'historical_cnn_retrained':False,'historical_cnn_reevaluated':False,'fresh_same_runtime_cnn_fit_count':0,
        'runtime_confound':RUNTIME_LIMITATION,'intervention_limitation':INTERVENTION_LIMITATION,
        'test_independent':False,'statistical_significance_established':False,'causal_effect_established':False}


def prepare(training_integrity_sha, registry_sha):
    from paper.scripts.advance_titantpp_post_attention_ffn_raf6_test import prepare as prepare_bundle
    return prepare_bundle(training_integrity_sha, registry_sha)


if __name__ == '__main__':
    import argparse
    import json
    p=argparse.ArgumentParser(description='Prepare isolated frozen RAF6 evaluation after verified training collection')
    p.add_argument('--mode',choices=('prepare',),default='prepare')
    p.add_argument('--training-integrity-sha',required=True)
    p.add_argument('--registry-sha',required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.training_integrity_sha,args.registry_sha),ensure_ascii=False,allow_nan=False))
