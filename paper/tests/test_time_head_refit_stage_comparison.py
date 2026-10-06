"""Synthetic paired-stage Validation comparison; no real artifacts or inference."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

DEFAULT = Path(__file__).resolve().parents[2]/'search_artifacts/titantpp_time_head_full_refit_5080_20261006_v1/control/compare_stages.py'
SCRIPT = Path(os.environ.get('TIME_HEAD_REFIT_COMPARISON_SCRIPT', DEFAULT))
spec = importlib.util.spec_from_file_location('refit_stage_comparison_under_test', SCRIPT)
comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)


class StageComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.full = self.project/'search_artifacts/full'
        self.bias = self.project/'search_artifacts/bias'
        self.full.mkdir(parents=True); self.bias.mkdir(parents=True)
        self.full_jobs=[]; self.bias_jobs=[]; self.full_rows=[]; self.bias_rows=[]
        for dataset,multiplier in [('yellow_trip_hourly',2.),('raf_spare_parts',3.)]:
            for i,seed in enumerate((42,52,62),1):
                canonical=f'{dataset}__{seed}__titantpp_cnn_gru54'
                checkpoint={'path':'/original/'+canonical+'/best_val_qty_rmse_model.pt',
                    'sha256':hashlib.sha256(canonical.encode()).hexdigest(),'state_sha256':'b'*64,'epoch':i+3}
                e0={'count':100,'time_nll':multiplier*i,'qty_rmse':10.*i,'qty_mae':4.*i,
                    'tail':{'count':10,'time_nll':multiplier*i*2,'qty_rmse':20.*i,'qty_mae':8.*i}}
                quantity={'train':hashlib.sha256((canonical+'train').encode()).hexdigest(),
                    'validation':hashlib.sha256((canonical+'validation').encode()).hexdigest()}
                for stage,count,jobs,rows in [('full130',130,self.full_jobs,self.full_rows),('bias2',2,self.bias_jobs,self.bias_rows)]:
                    j={'id':canonical+'__'+stage,'dataset':dataset,'seed':seed,'canonical_original_id':canonical,'checkpoint':deepcopy(checkpoint)}
                    selected=deepcopy(e0);factor=.5 if count==130 else .75
                    selected['time_nll']*=factor;selected['tail']['time_nll']*=factor
                    r={'schema':'titantpp_time_head_'+('full' if count==130 else 'bias')+'_refit_v1',
                        'status':'complete','job':j['id'],'evaluation_scope':'validation_only','held_out_test_evaluated':False,
                        'source_files_sha256':'a'*64,'trainable_scalar_count':count,'original_checkpoint':deepcopy(checkpoint),
                        'E0':deepcopy(e0),'selected':selected,'last':deepcopy(selected),'best_epoch':1 if count==130 else 2,
                        'completed_epochs':11 if count==130 else 12,'selected_is_identity':False,'quantity_output_unchanged':True,
                        'quantity_prediction_sha256_before':deepcopy(quantity),'quantity_prediction_sha256_after':deepcopy(quantity),
                        'frozen_state_sha256_before':('c' if count==130 else 'd')*64,'frozen_state_sha256_after':('c' if count==130 else 'd')*64}
                    jobs.append(j);rows.append({'job':deepcopy(j),'result':r,'all_owned_originals_SHA_verified':True})
        self.prior_c={'schema':'titantpp_time_head_bias_refit_v1','source':{'files_sha256':'a'*64},'jobs':self.bias_jobs}
        self.c={'schema':'titantpp_time_head_full_refit_v1','source':{'files_sha256':'a'*64},'jobs':self.full_jobs,
            'predecessor_bias2':{'local_bundle':'search_artifacts/bias','canonical_sha256':comparison.canonical(self.prior_c),'archive_sha256':'e'*64}}
        self.full_doc=self.document(self.full_rows,130)
        self.bias_doc=self.document(self.bias_rows,2)
        for row in self.full_rows:row['result']['contract_sha256']=comparison.canonical(self.c)
        for row in self.bias_rows:row['result']['contract_sha256']=comparison.canonical(self.prior_c)
        self.full_receipt={'contract_sha256':comparison.canonical(self.c),'local_originals_SHA_verified':True,'archive_sha256':'f'*64}
        self.bias_receipt={'contract_sha256':comparison.canonical(self.prior_c),'local_originals_SHA_verified':True,'archive_sha256':'e'*64}
        self.save()

    @staticmethod
    def document(rows,count):
        return {'scope':'Validation_development_selection','rows':rows,'new_head_fits':6,'trainable_scalars_per_fit':count,
            'held_out_test_evaluated':False,'independent_calibration':False,'S2P2_matching_refit':'not_performed'}

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')

    def save(self):
        for root,c,doc,receipt in [(self.full,self.c,self.full_doc,self.full_receipt),(self.bias,self.prior_c,self.bias_doc,self.bias_receipt)]:
            self.write(root/'execution_contract.json',c)
            path=root/'analysis/Validation_refit_comparison.json';self.write(path,doc)
            receipt['comparison_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            self.write(root/'analysis/completion_receipt.json',receipt)

    def run_comparison(self):
        with mock.patch.object(comparison,'BUNDLE',self.full),mock.patch.object(comparison,'PROJECT',self.project),mock.patch('builtins.print'):
            comparison.main()
        return json.loads((self.full/'analysis/Validation_three_stage_comparison.json').read_text())

    def rejected(self):
        self.save()
        with self.assertRaises((AssertionError,ValueError,KeyError)):self.run_comparison()
        self.assertFalse((self.full/'analysis/Validation_three_stage_comparison.json').exists())

    def test_identical_e0_six_pairs_and_three_seed_summary(self):
        out=self.run_comparison()
        self.assertEqual(len(out['rows']),6);self.assertEqual(len(out['aggregates']),2)
        self.assertEqual({row['canonical_original_id'] for row in out['rows']},{j['canonical_original_id'] for j in self.full_jobs})
        for a in out['aggregates']:
            multiplier=2. if a['dataset']=='yellow_trip_hourly' else 3.
            self.assertEqual(a['n_seeds'],3);self.assertEqual(sorted(a['full130_improved_seeds']),[42,52,62])
            self.assertAlmostEqual(a['original_time_nll_mean'],2*multiplier)
            self.assertAlmostEqual(a['original_time_nll_sample_std'],multiplier)
            self.assertAlmostEqual(a['bias2_time_nll_mean'],1.5*multiplier)
            self.assertAlmostEqual(a['bias2_time_nll_sample_std'],.75*multiplier)
            self.assertAlmostEqual(a['full130_time_nll_mean'],multiplier)
            self.assertAlmostEqual(a['full130_time_nll_sample_std'],.5*multiplier)
            self.assertAlmostEqual(a['full130_relative_mean_change_percent'],-50.)
            self.assertAlmostEqual(a['full130_vs_bias2_relative_mean_change_percent'],-100/3)
            for stage in ('original','bias2','full130'):
                self.assertEqual(a[stage+'_qty_rmse_mean'],20.)
                self.assertEqual(a[stage+'_qty_mae_mean'],8.)
                self.assertEqual(a[stage+'_tail_qty_rmse_mean'],40.)
                self.assertEqual(a[stage+'_tail_qty_mae_mean'],16.)
        self.assertFalse(out['held_out_Test_evaluated']);self.assertFalse(out['independent_calibration'])
        self.assertEqual(out['S2P2_matching_refit'],'not_performed')

    def test_checkpoint_pair_mismatch_rejected(self):
        self.bias_rows[0]['job']['checkpoint']['sha256']='0'*64;self.rejected()

    def test_result_original_checkpoint_mismatch_rejected(self):
        self.full_rows[0]['result']['original_checkpoint']['epoch']=999;self.rejected()

    def test_original_e0_mismatch_rejected(self):
        self.bias_rows[0]['result']['E0']['time_nll']+=.1;self.rejected()

    def test_within_full_quantity_change_rejected(self):
        self.full_rows[0]['result']['quantity_prediction_sha256_after']['validation']='0'*64;self.rejected()

    def test_cross_stage_quantity_mismatch_rejected(self):
        r=self.bias_rows[0]['result'];r['quantity_prediction_sha256_before']['train']='0'*64;r['quantity_prediction_sha256_after']['train']='0'*64;self.rejected()

    def test_full_test_scope_rejected(self):
        self.full_rows[0]['result']['held_out_test_evaluated']=True;self.rejected()

    def test_prior_test_scope_rejected(self):
        self.bias_rows[0]['result']['held_out_test_evaluated']=True;self.rejected()

    def test_stage_level_test_scope_rejected(self):
        self.full_doc['held_out_test_evaluated']=True;self.rejected()

    def test_prior_receipt_contract_mismatch_rejected(self):
        self.bias_receipt['contract_sha256']='0'*64;self.rejected()

    def test_prior_archive_mismatch_rejected(self):
        self.bias_receipt['archive_sha256']='0'*64;self.rejected()

    def test_source_closure_mismatch_rejected(self):
        self.full_rows[0]['result']['source_files_sha256']='0'*64;self.rejected()

    def test_full_frozen_weight_change_rejected(self):
        self.full_rows[0]['result']['frozen_state_sha256_after']='0'*64;self.rejected()

    def test_duplicate_prior_canonical_job_rejected(self):
        self.bias_rows[1]['job']['canonical_original_id']=self.bias_rows[0]['job']['canonical_original_id'];self.rejected()

    def test_duplicate_full_canonical_job_rejected(self):
        # Keep six rows and valid dataset/seed triplets; dictionary-overwrite
        # matching must still reject duplicate canonical source identities.
        self.full_rows[1]['job']['canonical_original_id']=self.full_rows[0]['job']['canonical_original_id']
        self.full_rows[1]['job']['checkpoint']=deepcopy(self.full_rows[0]['job']['checkpoint'])
        self.full_rows[1]['result']=deepcopy(self.full_rows[0]['result'])
        self.rejected()

    def test_full_result_contract_mismatch_rejected(self):
        self.full_rows[0]['result']['contract_sha256']='0'*64;self.rejected()

    def test_prior_result_contract_mismatch_rejected(self):
        self.bias_rows[0]['result']['contract_sha256']='0'*64;self.rejected()

    def test_full_result_job_identity_mismatch_rejected(self):
        self.full_rows[0]['result']['job']='wrong-job';self.rejected()

    def test_prior_result_job_identity_mismatch_rejected(self):
        self.bias_rows[0]['result']['job']='wrong-job';self.rejected()

    def test_paired_seed_mismatch_with_valid_overall_grid_rejected(self):
        self.full_rows[0]['job']['seed'],self.full_rows[1]['job']['seed']=52,42
        self.rejected()

    def test_comparison_file_change_after_receipt_rejected(self):
        self.full_doc['rows'][0]['result']['selected']['time_nll']+=.01
        self.write(self.full/'analysis/Validation_refit_comparison.json',self.full_doc)
        with self.assertRaises((AssertionError,ValueError)):self.run_comparison()

    def test_duplicate_seed_rejected(self):
        self.full_rows[1]['job']['seed']=42;self.rejected()


if __name__=='__main__':unittest.main()
