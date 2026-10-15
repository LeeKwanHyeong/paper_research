"""Fabricated RAF6 comparison fixtures: no real result or checkpoint reads."""
from copy import deepcopy
import unittest

from paper.scripts import compare_titantpp_location_loop_historical as c


def fixture():
    population={"count":5,"target_identity_sha256":"a"*64,"truth_sha256":"b"*64,"time_sha256":"c"*64}
    rows=[]
    for model,value in (("historical_CNN",4.),("external_one",3.)):
        for split in c.SPLITS:
            for seed in c.SEEDS:
                rows.append({"model":model,"split":split,"seed":seed,"selected_epoch":5,
                    "checkpoint_sha256":str(seed)[0]*64,"metrics":{m:value for m in c.METRICS},
                    "training_host":"5080","evaluation_host":"5080"})
    registry={"status":"verified","dataset":"raf_spare_parts","seeds":list(c.SEEDS),
        "expected_population":{s:deepcopy(population) for s in c.SPLITS},"rows":rows,
        "historical_cnn":{s:{m:c.stats([4.,4.,4.]) for m in c.METRICS} for s in c.SPLITS},
        "external_frontier":{s:{m:{"models":["external_one"],"mean":3.,"sample_sd":{"external_one":0.}}
            for m in c.METRICS} for s in c.SPLITS}}
    runtime=deepcopy(c.NATIVE_RUNTIME)
    def group(model,value):
        return [{"model":model,"split":s,"seed":seed,"dataset":"raf_spare_parts","status":"complete",
            "validation_selected_checkpoint_verified":True,"selected_epoch":8,"checkpoint_sha256":str(seed)[0]*64,
            "metrics":{m:value for m in c.METRICS},"count":5,"population":deepcopy(population),
            "runtime":deepcopy(runtime),"added_parameters":1024} for s in c.SPLITS for seed in c.SEEDS]
    return registry,group(c.VALUE,2.),group(c.QUERY,2.5)


class HistoricalComparisonTests(unittest.TestCase):
    def test_strict_four_user_checks_and_separate_attribution(self):
        registry,value,query=fixture()
        r=c.compare_historical_rows(registry,value,query)
        self.assertTrue(r['user_performance_gate']);self.assertEqual(r['user_performance_check_count'],4)
        self.assertTrue(r['value_minus_query_attribution_gate'])
        self.assertFalse(r['same_host_control_performed']);self.assertIsNone(r['same_host_control_gate'])
        self.assertFalse(r['historical_cnn_retrained']);self.assertFalse(r['historical_cnn_reevaluated'])
        self.assertEqual(len(r['seed_differences']),12)
        self.assertIn('5080',r['runtime_confound']);self.assertIn('5090',r['runtime_confound'])

    def test_user_gate_does_not_silently_require_attribution(self):
        registry,value,query=fixture()
        for row in query:row['metrics']={m:1. for m in c.METRICS}
        r=c.compare_historical_rows(registry,value,query)
        self.assertTrue(r['user_performance_gate']);self.assertFalse(r['value_minus_query_attribution_gate'])

    def test_equal_external_frontier_fails_strict_gate(self):
        registry,value,query=fixture()
        for row in value:
            if row['split']=='test':row['metrics']['qty_mae']=3.
        r=c.compare_historical_rows(registry,value,query)
        self.assertFalse(r['user_performance_gate'])
        self.assertFalse(r['summaries']['test']['qty_mae']['user_performance_pass'])

    def test_any_one_of_four_criteria_failure_is_retained(self):
        for split in c.SPLITS:
            for metric in c.METRICS:
                registry,value,_=fixture()
                for row in value:
                    if row['split']==split:row['metrics'][metric]=5.
                self.assertFalse(c.compare_historical_rows(registry,value)['user_performance_gate'])

    def test_adverse_seed_and_relative_difference_are_retained(self):
        registry,value,_=fixture();value[0]['metrics']['qty_rmse']=5.
        r=c.compare_historical_rows(registry,value)
        d=r['seed_differences'][0]['differences']['historical_CNN']
        self.assertEqual(d['absolute_candidate_minus_reference'],1.)
        self.assertEqual(d['relative_percent_candidate_minus_reference'],25.)

    def test_runtime_confounded_history_is_not_called_same_host(self):
        registry,value,_=fixture()
        r=c.compare_historical_rows(registry,value)
        self.assertIsNone(r['same_host_control_gate']);self.assertIsNone(r['value_minus_query_attribution_gate'])
        self.assertFalse(r['test_independent']);self.assertFalse(r['statistical_significance_established'])

    def test_missing_duplicate_population_or_checkpoint_drift_rejected(self):
        registry,value,_=fixture()
        examples=[value[:-1],value[:-1]+[value[0]]]
        wrong=deepcopy(value);wrong[-1]['population']['truth_sha256']='wrong';examples.append(wrong)
        wrong=deepcopy(value);wrong[-1]['checkpoint_sha256']='d'*64;examples.append(wrong)
        for rows in examples:
            with self.assertRaises(ValueError):c.compare_historical_rows(registry,rows)

    def test_value_query_runtime_difference_blocks_attribution(self):
        registry,value,query=fixture();query[0]['runtime']['cudnn']=91900
        with self.assertRaisesRegex(ValueError,'Runtime identity mismatch'):c.compare_historical_rows(registry,value,query)

    def test_wrong_candidate_and_reversed_attribution_rejected(self):
        registry,value,query=fixture()
        with self.assertRaisesRegex(ValueError,'value minus query'):c.compare_historical_rows(registry,query,value)
        for row in value:row['model']='titantpp_cnn'
        with self.assertRaisesRegex(ValueError,'approved RAF6'):c.compare_historical_rows(registry,value)

    def test_history_aggregate_and_checkpoint_identity_are_rechecked(self):
        registry,value,_=fixture();registry['historical_cnn']['validation']['qty_mae']['mean']=1.
        with self.assertRaisesRegex(ValueError,'aggregate'):c.compare_historical_rows(registry,value)
        registry,value,_=fixture();registry['rows'][3]['checkpoint_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'checkpoint'):c.compare_historical_rows(registry,value)

    def test_external_frontier_seed_source_required(self):
        registry,value,_=fixture();registry['rows']=registry['rows'][:-1]
        with self.assertRaisesRegex(ValueError,'frontier seed'):c.compare_historical_rows(registry,value)

    def test_inputs_are_not_mutated(self):
        registry,value,query=fixture();before=deepcopy((registry,value,query))
        c.compare_historical_rows(registry,value,query)
        self.assertEqual((registry,value,query),before)

    def test_stored_channel_3072_reference_is_explicitly_auxiliary(self):
        registry,value,query=fixture();stored=deepcopy(query)
        for row in stored:
            row.update(model=c.STORED_CHANNEL,reference_role='stored_loop1_channel_immutable_reference',
                retrained=False,reevaluated=False,added_parameters=3072,original_evidence_verified=True)
        r=c.compare_historical_rows(registry,value,query,stored)
        self.assertTrue(r['stored_channel_reused'])
        self.assertEqual(r['campaign_new_fit_count'],6);self.assertEqual(r['candidate_fit_count'],3)
        self.assertEqual(r['historical_cnn_reference_count'],3)
        self.assertEqual(r['stored_channel_reference_count'],3)
        self.assertEqual(r['candidate_added_parameters'],1024)
        self.assertEqual(r['stored_channel_added_parameters'],3072)
        self.assertFalse(r['stored_channel_retrained']);self.assertFalse(r['stored_channel_reevaluated'])
        self.assertIn('not a matched capacity attribution',r['stored_channel_confound'])
        self.assertIn('stored_QKV_Channel_auxiliary',r['seed_differences'][0]['differences'])
        self.assertEqual(r['summaries']['validation']['qty_rmse']['candidate']['ddof'],1)

    def test_stored_channel_provenance_identity_and_capacity_are_required(self):
        for field,value in [('reference_role','fresh'),('retrained',True),('reevaluated',True),
                            ('added_parameters',1024),('original_evidence_verified',False)]:
            registry,primary,query=fixture();stored=deepcopy(query)
            for row in stored:
                row.update(model=c.STORED_CHANNEL,reference_role='stored_loop1_channel_immutable_reference',
                    retrained=False,reevaluated=False,added_parameters=3072,original_evidence_verified=True)
            stored[0][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'provenance/parameter'):
                c.compare_historical_rows(registry,primary,query,stored)

    def test_candidate_parameter_budget_mismatch_rejected(self):
        registry,value,query=fixture();query[0]['added_parameters']=3072
        with self.assertRaisesRegex(ValueError,'1024'):c.compare_historical_rows(registry,value,query)

    def test_missing_duplicate_foreign_or_nonfinite_any_group_rejected(self):
        for which in ('candidate','query','stored'):
            for change in ('missing','duplicate','target','count','nan','inf','negative','runtime'):
                registry,value,query=fixture();stored=deepcopy(query)
                for row in stored:
                    row.update(model=c.STORED_CHANNEL,reference_role='stored_loop1_channel_immutable_reference',
                        retrained=False,reevaluated=False,added_parameters=3072,original_evidence_verified=True)
                groups={'candidate':value,'query':query,'stored':stored};target=groups[which]
                if change=='missing':target.pop()
                elif change=='duplicate':target[-1]=deepcopy(target[0])
                elif change=='target':target[0]['population']['target_identity_sha256']='0'*64
                elif change=='count':target[0]['count']=4
                elif change=='nan':target[0]['metrics']['qty_mae']=float('nan')
                elif change=='inf':target[0]['metrics']['qty_rmse']=float('inf')
                elif change=='negative':target[0]['metrics']['qty_mae']=-1.
                elif change=='runtime':target[0]['runtime']['gpu_uuid']='wrong'
                with self.subTest(group=which,change=change),self.assertRaises(ValueError):
                    c.compare_historical_rows(registry,value,query,stored)

    def test_external_frontier_is_recomputed_from_all_seed_rows(self):
        registry,value,_=fixture();registry['external_frontier']['test']['qty_mae']['mean']=20.
        with self.assertRaisesRegex(ValueError,'metricwise frontier'):c.compare_historical_rows(registry,value)
        registry,value,_=fixture();registry['external_frontier']['test']['qty_mae']['models']*=2
        with self.assertRaisesRegex(ValueError,'metricwise frontier'):c.compare_historical_rows(registry,value)

    def test_external_frontier_tied_models_all_keep_seed_differences(self):
        registry,value,_=fixture()
        other=[{**deepcopy(r),'model':'external_two'} for r in registry['rows'] if r['model']=='external_one']
        registry['rows']+=other
        for split in c.SPLITS:
            for metric in c.METRICS:
                registry['external_frontier'][split][metric]['models'].append('external_two')
                registry['external_frontier'][split][metric]['sample_sd']['external_two']=0.
        r=c.compare_historical_rows(registry,value)
        self.assertIn('external_frontier:external_one',r['seed_differences'][0]['differences'])
        self.assertIn('external_frontier:external_two',r['seed_differences'][0]['differences'])

    def test_metricwise_best_external_models_can_differ(self):
        registry,value,_=fixture()
        other=[{**deepcopy(r),'model':'external_two','metrics':{'qty_rmse':4.,'qty_mae':2.5}}
               for r in registry['rows'] if r['model']=='external_one']
        registry['rows']+=other
        for split in c.SPLITS:
            registry['external_frontier'][split]['qty_mae']={'models':['external_two'],'mean':2.5,'sample_sd':{'external_two':0.}}
        r=c.compare_historical_rows(registry,value)
        self.assertEqual(r['summaries']['test']['qty_mae']['external_frontier']['models'],['external_two'])
        self.assertEqual(r['summaries']['test']['qty_rmse']['external_frontier']['models'],['external_one'])

    def test_zero_reference_relative_difference_is_na(self):
        registry,value,_=fixture()
        for row in registry['rows']:
            if row['model']=='external_one':row['metrics']['qty_mae']=0.
        for split in c.SPLITS:registry['external_frontier'][split]['qty_mae']['mean']=0.
        r=c.compare_historical_rows(registry,value)
        changes=[d for d in r['seed_differences'] if d['metric']=='qty_mae']
        self.assertTrue(all(d['differences']['external_frontier:external_one']['relative_percent_candidate_minus_reference'] is None for d in changes))
        self.assertFalse(r['user_performance_gate'])

    def test_normalization_preserves_actual_identity_and_marks_parameter_budget(self):
        registry,value,_=fixture();row=value[0]
        receipt={'status':'complete','native_runtime_checked':True,'arm':c.VALUE,'dataset':row['dataset'],
            'split':row['split'],'seed':row['seed'],'selected_epoch':row['selected_epoch'],
            'checkpoint_sha256':row['checkpoint_sha256'],'metrics':{**row['metrics'],'count':5},
            **{k:row['population'][k] for k in ('target_identity_sha256','truth_sha256','time_sha256')},
            'runtime':{**c.NATIVE_RUNTIME,'gpu':{'uuid':c.NATIVE_RUNTIME['gpu_uuid']}}}
        normalized=c.normalized_native_row(receipt,'5090')
        self.assertEqual(normalized['model'],c.VALUE);self.assertEqual(normalized['added_parameters'],1024)
        self.assertEqual(normalized['runtime'],c.NATIVE_RUNTIME)
        receipt['arm']=c.STORED_CHANNEL
        self.assertEqual(c.normalized_native_row(receipt,'5090',stored=True)['added_parameters'],3072)
        with self.assertRaisesRegex(ValueError,'Foreign'):c.normalized_native_row(receipt,'5090')
        receipt['status']='failed'
        with self.assertRaisesRegex(ValueError,'completed receipt'):c.normalized_native_row(receipt,'5090',stored=True)


class CandidateSelectionTests(unittest.TestCase):
    def comparisons(self,query_values=(2.5,2.5,2.5,2.5),value_values=(2.,2.,2.,2.)):
        registry,value,query=fixture()
        for rows,values in ((query,query_values),(value,value_values)):
            for row in rows:
                offset=0 if row['split']=='validation' else 2
                row['metrics']={'qty_rmse':values[offset],'qty_mae':values[offset+1]}
        return {c.QUERY:c.compare_historical_rows(registry,query),
                c.VALUE:c.compare_historical_rows(registry,value,query)}

    def test_passing_candidates_order_by_validation_not_test(self):
        results=self.comparisons(query_values=(2.2,2.2,.1,.1),value_values=(2.,2.,2.8,2.8))
        s=c.select_passing_candidate(results)
        self.assertEqual(s['selected_candidate'],c.VALUE)
        self.assertFalse(s['test_used_for_tie_break'])
        self.assertFalse(s['value_hypothesis_supported'])
        self.assertEqual(s['selection_rule'],c.SELECTION_RULE)

    def test_validation_mae_breaks_rmse_tie_then_arm_name(self):
        s=c.select_passing_candidate(self.comparisons(query_values=(2.,2.1,1.,1.),value_values=(2.,2.,2.,2.)))
        self.assertEqual(s['selected_candidate'],c.VALUE)
        s=c.select_passing_candidate(self.comparisons(query_values=(2.,2.,2.8,2.8),value_values=(2.,2.,1.,1.)))
        self.assertEqual(s['selected_candidate'],min(c.QUERY,c.VALUE))

    def test_query_only_success_is_not_value_hypothesis_success(self):
        s=c.select_passing_candidate(self.comparisons(query_values=(2.,2.,2.,2.),value_values=(1.,1.,3.,3.)))
        self.assertEqual(s['selected_candidate'],c.QUERY)
        self.assertFalse(s['value_hypothesis_supported'])
        self.assertTrue(s['query_only_pass_is_not_value_success'])

    def test_no_passing_candidate_is_not_promoted(self):
        s=c.select_passing_candidate(self.comparisons(query_values=(3.,3.,3.,3.),value_values=(4.,4.,4.,4.)))
        self.assertIsNone(s['selected_candidate']);self.assertEqual(s['passing_candidates'],[])

    def test_value_user_and_attribution_both_required(self):
        s=c.select_passing_candidate(self.comparisons())
        self.assertTrue(s['value_user_criteria_pass']);self.assertTrue(s['value_minus_query_attribution_gate'])
        self.assertTrue(s['value_hypothesis_supported'])

    def test_forged_gates_and_nonfinite_summary_rejected(self):
        for change in ('flag','mean','nan','attribution','identity','count','control','no_attribution'):
            results=self.comparisons();target=results[c.VALUE]
            if change=='flag':target['user_performance_gate']=False
            elif change=='mean':target['summaries']['test']['qty_mae']['candidate']['mean']=5.
            elif change=='nan':target['summaries']['validation']['qty_rmse']['candidate']['mean']=float('nan')
            elif change=='attribution':target['value_minus_query_attribution_gate']=False
            elif change=='identity':target['candidate_model']=c.QUERY
            elif change=='count':target['user_performance_check_count']=3
            elif change=='control':target['summaries']['test']['qty_mae']['query_control']['mean']=2.9
            elif change=='no_attribution':target['value_minus_query_attribution_gate']=None
            with self.subTest(change=change),self.assertRaises(ValueError):c.select_passing_candidate(results)

    def test_missing_candidate_comparison_rejected(self):
        results=self.comparisons();results.pop(c.QUERY)
        with self.assertRaisesRegex(ValueError,'Both approved'):c.select_passing_candidate(results)


if __name__=='__main__':unittest.main()
