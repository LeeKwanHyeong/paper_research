"""Fabricated RAF6 comparison fixtures: no real result or checkpoint reads."""
from copy import deepcopy
import unittest

from paper.scripts import compare_titantpp_backbone_loop_historical as c


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
    runtime={"host":"5090","gpu_uuid":"fixed5090","python":"3.12.13","torch":"2.11.0+cu130","cudnn":92000,"threads":4}
    def group(model,value):
        return [{"model":model,"split":s,"seed":seed,"dataset":"raf_spare_parts","status":"complete",
            "validation_selected_checkpoint_verified":True,"selected_epoch":8,"checkpoint_sha256":str(seed)[0]*64,
            "metrics":{m:value for m in c.METRICS},"count":5,"population":deepcopy(population),
            "runtime":deepcopy(runtime)} for s in c.SPLITS for seed in c.SEEDS]
    return registry,group(c.JOINT,2.),group(c.CHANNEL,2.5)


class HistoricalComparisonTests(unittest.TestCase):
    def test_strict_four_user_checks_and_separate_attribution(self):
        registry,joint,channel=fixture()
        r=c.compare_historical_rows(registry,joint,channel)
        self.assertTrue(r['user_performance_gate']);self.assertEqual(r['user_performance_check_count'],4)
        self.assertTrue(r['channel_attribution_gate'])
        self.assertFalse(r['same_host_control_performed']);self.assertIsNone(r['same_host_control_gate'])
        self.assertFalse(r['historical_cnn_retrained']);self.assertFalse(r['historical_cnn_reevaluated'])
        self.assertEqual(len(r['seed_differences']),12)
        self.assertIn('5080',r['runtime_confound']);self.assertIn('5090',r['runtime_confound'])

    def test_user_gate_does_not_silently_require_attribution(self):
        registry,joint,channel=fixture()
        for row in channel:row['metrics']={m:1. for m in c.METRICS}
        r=c.compare_historical_rows(registry,joint,channel)
        self.assertTrue(r['user_performance_gate']);self.assertFalse(r['channel_attribution_gate'])

    def test_equal_external_frontier_fails_strict_gate(self):
        registry,joint,channel=fixture()
        for row in joint:
            if row['split']=='test':row['metrics']['qty_mae']=3.
        r=c.compare_historical_rows(registry,joint,channel)
        self.assertFalse(r['user_performance_gate'])
        self.assertFalse(r['summaries']['test']['qty_mae']['user_performance_pass'])

    def test_any_one_of_four_criteria_failure_is_retained(self):
        for split in c.SPLITS:
            for metric in c.METRICS:
                registry,joint,_=fixture()
                for row in joint:
                    if row['split']==split:row['metrics'][metric]=5.
                self.assertFalse(c.compare_historical_rows(registry,joint)['user_performance_gate'])

    def test_adverse_seed_and_relative_difference_are_retained(self):
        registry,joint,_=fixture();joint[0]['metrics']['qty_rmse']=5.
        r=c.compare_historical_rows(registry,joint)
        d=r['seed_differences'][0]['differences']['historical_CNN']
        self.assertEqual(d['absolute_candidate_minus_reference'],1.)
        self.assertEqual(d['relative_percent_candidate_minus_reference'],25.)

    def test_runtime_confounded_history_is_not_called_same_host(self):
        registry,joint,_=fixture()
        r=c.compare_historical_rows(registry,joint)
        self.assertIsNone(r['same_host_control_gate']);self.assertIsNone(r['channel_attribution_gate'])
        self.assertFalse(r['test_independent']);self.assertFalse(r['statistical_significance_established'])

    def test_missing_duplicate_population_or_checkpoint_drift_rejected(self):
        registry,joint,_=fixture()
        examples=[joint[:-1],joint[:-1]+[joint[0]]]
        wrong=deepcopy(joint);wrong[-1]['population']['truth_sha256']='wrong';examples.append(wrong)
        wrong=deepcopy(joint);wrong[-1]['checkpoint_sha256']='d'*64;examples.append(wrong)
        for rows in examples:
            with self.assertRaises(ValueError):c.compare_historical_rows(registry,rows)

    def test_joint_channel_runtime_difference_blocks_attribution(self):
        registry,joint,channel=fixture();channel[0]['runtime']['cudnn']=91900
        with self.assertRaisesRegex(ValueError,'Runtime mismatch'):c.compare_historical_rows(registry,joint,channel)

    def test_wrong_candidate_and_reversed_attribution_rejected(self):
        registry,joint,channel=fixture()
        with self.assertRaisesRegex(ValueError,'joint minus channel'):c.compare_historical_rows(registry,channel,joint)
        for row in joint:row['model']='titantpp_cnn'
        with self.assertRaisesRegex(ValueError,'approved RAF6'):c.compare_historical_rows(registry,joint)

    def test_history_aggregate_and_checkpoint_identity_are_rechecked(self):
        registry,joint,_=fixture();registry['historical_cnn']['validation']['qty_mae']['mean']=1.
        with self.assertRaisesRegex(ValueError,'aggregate'):c.compare_historical_rows(registry,joint)
        registry,joint,_=fixture();registry['rows'][3]['checkpoint_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'checkpoint'):c.compare_historical_rows(registry,joint)

    def test_external_frontier_seed_source_required(self):
        registry,joint,_=fixture();registry['rows']=registry['rows'][:-1]
        with self.assertRaisesRegex(ValueError,'frontier seed'):c.compare_historical_rows(registry,joint)

    def test_inputs_are_not_mutated(self):
        registry,joint,channel=fixture();before=deepcopy((registry,joint,channel))
        c.compare_historical_rows(registry,joint,channel)
        self.assertEqual((registry,joint,channel),before)


if __name__=='__main__':unittest.main()
