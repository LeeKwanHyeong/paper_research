"""Hand-calculated statistics and negative provenance gates; synthetic only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

PROJECT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('_components36_merge',PROJECT/'reports/titantpp_components_taxi_raf24_test_20261014_v1/merge_components36_results.py')
merge=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(merge)


def rows_fixture():
    rows=[]
    for d in merge.DATASETS:
        for split in ('validation','test'):
            for seed,base in zip(merge.SEEDS,(1.,3.,5.)):
                for arm,offset in zip(merge.ARMS,(3.,2.,1.,0.)):
                    # FULL is best at Validation but loses to CNN at Test.
                    if split=='test' and arm=='FULL':offset=3.
                    max_len,registered=(84,104159) if d=='raf_spare_parts' else (256,115167)
                    active=dict(zip(merge.ARMS,(78851,79427,103583,104159) if d=='raf_spare_parts' else (89859,90435,114591,115167)))[arm]
                    raw_arm=next(k for k,v in merge.LABELS.items() if v==arm)
                    rows.append({'dataset':d,'split':split,'seed':seed,'arm':arm,'id':f'{d}__{seed}__{raw_arm}',
                        'metrics':{name:base+offset for name,_ in merge.METRICS},
                        'simple_quantity_baselines':{'last':{'qty_rmse':2.},'mean':{'qty_rmse':3.}},
                        'population':{'target_identity_sha256':d+split,'truth_sha256':'b','time_sha256':'c'},
                        'count':100,'tail_count':10,'parameter_counts':{'actual_registered_parameters':registered,
                          'actual_active_parameters_before_evaluation_freeze':active,'maximum_sequence_length':max_len}})
    return rows


class Components36MergeTests(unittest.TestCase):
    def test_three_seed_mean_and_sample_sd_hand_calculation(self):
        result=merge.summarize(rows_fixture())
        cells=result['summary_mean_sample_sd']['yellow_trip_hourly']
        self.assertEqual(cells['validation']['FULL']['전체 RMSE'],{'mean':3.,'sample_sd':2.,'n':3})
        self.assertEqual(cells['test']['FULL']['전체 RMSE'],{'mean':6.,'sample_sd':2.,'n':3})
        self.assertEqual(len(result['seed_paired_differences']),54)

    def test_paired_seed_wins_and_val_test_reversal_hand_calculation(self):
        result=merge.summarize(rows_fixture())
        one=lambda split:next(x for x in result['FULL_metric_comparisons'] if x['dataset']=='yellow_trip_hourly' and x['split']==split and x['comparison']=='CNN' and x['metric']=='전체 RMSE')
        self.assertEqual(one('validation')['strict_wins'],3)
        self.assertEqual(one('test')['strict_losses'],3)
        flips=[x for x in result['Validation_Test_direction_comparisons'] if x['dataset']=='yellow_trip_hourly' and x['comparison']=='CNN' and x['metric']=='전체 RMSE']
        self.assertEqual(len(flips),3);self.assertTrue(all(x['strict_direction_reversal'] for x in flips))

    def test_missing_duplicate_and_foreign_campaign_id_rejected(self):
        good=rows_fixture()
        for bad in (good[:-1],good[:-1]+[good[0]],good+[good[0]]):
            with self.assertRaises(ValueError):merge.summarize(bad)
        bad=deepcopy(good);bad[0]['id']='representative54__42__FULL'
        with self.assertRaises(ValueError):merge.summarize(bad)

    def test_nonfinite_metrics_and_population_or_baseline_mismatch_rejected(self):
        for field,changed in (('metrics',{'전체 RMSE':float('nan')}),('population',{'truth_sha256':'changed'}),('simple_quantity_baselines',{'mean':{'qty_rmse':99.}})):
            bad=rows_fixture();bad[0][field].update(changed)
            with self.assertRaises(ValueError):merge.summarize(bad)

    def test_physical_parameter_budget_change_rejected(self):
        bad=rows_fixture();bad[0]['parameter_counts']['actual_active_parameters_before_evaluation_freeze']+=1
        with self.assertRaises(ValueError):merge.summarize(bad)

    def test_manifest_hash_mismatch_and_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as name:
            root=Path(name);(root/'data').write_text('sealed')
            merge.verify_manifest(root,{'files':{'data':merge.sha(root/'data')}})
            with self.assertRaises(ValueError):merge.verify_manifest(root,{'files':{'data':'a'*64}})
            with self.assertRaises(ValueError):merge.verify_manifest(root,{'files':{'../foreign':'a'*64}})

    def test_unfinished_native_results_only_report_readiness(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as name:
            root=Path(name);before=list(root.iterdir());value=merge.ready(root)
            self.assertEqual(value['status'],'waiting_for_native_results')
            self.assertFalse(value['outputs_written']);self.assertEqual(list(root.iterdir()),before)

    def test_original_manifest_paths_are_relative_to_bundle_not_originals_twice(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as name:
            root=Path(name);(root/'originals').mkdir();files={}
            for index in range(360):
                relative=f'originals/synthetic_{index:03d}.bin'
                (root/relative).write_bytes(str(index).encode());files[relative]=merge.sha(root/relative)
            (root/'originals_archive_verified.tar.gz').write_bytes(b'synthetic archive')
            (root/'originals_manifest.json').write_text(json.dumps({'files':files,
                'all_file_SHA_verified':True,'source131_SHA_verified':True,
                'checkpoint_selected_last_pairs':24,'remote_originals_changed':False,
                'archive_raw_sha256':merge.sha(root/'originals_archive_verified.tar.gz')}))
            self.assertEqual(merge.verify_originals24(root)['file_count'],360)

    def test_nonzero_pipeline_actual_exit_cannot_publish(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as name:
            root=Path(name)
            def write(path,value):(root/path).write_text(json.dumps(value))
            write('pipeline_actual_exit.json',{'actual_child_exit_observed':True,'exit_code':1})
            write('evaluation_results_manifest.json',{'contract_sha256':'c','files':{'pipeline_actual_exit.json':merge.sha(root/'pipeline_actual_exit.json')}})
            (root/'evaluation_results_archive.tar.gz').write_bytes(b'synthetic archive')
            write('evaluation_results_recovery_receipt.json',{'status':'complete','archive_sha256':merge.sha(root/'evaluation_results_archive.tar.gz'),'manifest_sha256':merge.sha(root/'evaluation_results_manifest.json')})
            evaluator=SimpleNamespace(verify_contract=lambda root:{'jobs':[{}],'runtimes':{'5080':{}},'sources':{'components':{'closure':merge.SOURCE}},'contract_sha256':'c'})
            with self.assertRaisesRegex(ValueError,'pipeline exit0'):merge.verify_bundle(root,evaluator,1,'5080',1)
            self.assertFalse((root/'COMPONENTS36_ANALYSIS.json').exists())

if __name__=='__main__':unittest.main()
