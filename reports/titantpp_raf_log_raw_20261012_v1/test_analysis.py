"""Fabricated stdlib-only aggregation regressions; no inference or real data."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

A=load(ROOT/'paper/scripts/analyze_titantpp_raf_log_raw.py','fixture_raf_analysis')
E=load(ROOT/'paper/scripts/evaluate_titantpp_raf_log_raw.py','fixture_raf_statistics')

def aggregate(scale=1.):
    collector=E.LogRawAggregates([2.,30.,60.,200.])
    for truth in (1.,2.,3.,30.,60.,200.,201.,500.):
        prediction=truth*scale
        lp,lt=math.log1p(prediction),math.log1p(truth)
        logs={'log_prediction':lp,'log_truth':lt,'log_error':lp-lt,'log_native_loss':(lp-lt)**2}
        baseline={'log_prediction':lt,'log_truth':lt,'log_error':0.,'log_native_loss':0.,'raw_prediction':truth}
        collector.add(truth,prediction,logs,{'last':baseline,'mean':baseline})
    return collector.finish()


def rows():
    output=[]
    for split in A.SPLITS:
        for seed in A.SEEDS:
            for index,arm in enumerate(A.ARMS):
                output.append({'id':f'fixture_{arm}_{seed}','arm':arm,'seed':seed,'split':split,
                    'target_identity_sha256':'fixture_'+split,'truth_history_sha256':'fixture_truth_'+split,
                    'log_raw':aggregate(.5+.1*index+.002*(seed-42))})
    return output


class SufficientStatisticsTests(unittest.TestCase):
    def test_native_bins_preserve_equal_boundary_lower_interval(self):
        value=aggregate()
        self.assertTrue(A.verify_sums(value))
        self.assertEqual([value['cells'][str(i)]['model']['count'] for i in range(5)],[2,2,1,1,2])
        self.assertEqual(value['cells']['4']['model']['count'],2)
        self.assertEqual(value['cells']['all']['model']['log_mse_native'],0.)
        self.assertEqual(value['cells']['all']['model']['raw_rmse'],0.)

    def test_negative_count_nonfinite_moment_and_raw_error_rejected(self):
        for mutate in ('count','finite','moment','raw_error'):
            value=aggregate(.7)
            cell=value['cells']['0']['model']
            if mutate=='count':cell['count']+=1
            if mutate=='finite':cell['log_bias']=float('nan')
            if mutate=='moment':cell['log_prediction_mean']+=1.
            if mutate=='raw_error':cell['raw_error_sum']+=10.
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):A.verify_sums(value)

    def test_empty_bin_does_not_report_zero_performance(self):
        collector=E.LogRawAggregates([2.,30.,60.,200.])
        logs={'log_prediction':math.log(2.),'log_truth':math.log(2.),'log_error':0.,'log_native_loss':0.}
        baseline={**logs,'raw_prediction':1.}
        collector.add(1.,1.,logs,{'last':baseline,'mean':baseline})
        value=collector.finish();self.assertTrue(A.verify_sums(value))
        self.assertIsNone(value['cells']['4']['model']['raw_rmse'])
        value['cells']['4']['model']['log_mse_native']=0.
        with self.assertRaisesRegex(ValueError,'Empty RAF performance'):A.verify_sums(value)

    def test_native_rounding_is_not_float64_error_square_identity(self):
        value=aggregate(.7)
        cell=value['cells']['0']['model']
        adjustment=1e-7
        cell['log_error_sum']+=adjustment
        cell['log_bias']+=adjustment/cell['count'];cell['log_error_mean']+=adjustment/cell['count']
        allcell=value['cells']['all']['model'];allcell['log_error_sum']+=adjustment
        allcell['log_bias']+=adjustment/allcell['count'];allcell['log_error_mean']+=adjustment/allcell['count']
        self.assertTrue(A.verify_sums(value))

    def test_tail_contribution_is_quantity_loss_only_and_zero_error_not_fabricated(self):
        value=A.loss_contributions(aggregate(.7))
        self.assertEqual(value['tail_count'],2)
        self.assertEqual(value['tail_target_share'],.25)
        self.assertTrue(0<value['tail_share_of_quantity_log_loss']<1)
        self.assertTrue(0<value['tail_share_of_raw_quantity_sse']<1)
        self.assertTrue(value['quantity_loss_only_not_joint_time_quantity_objective'])
        zero=A.loss_contributions(aggregate())
        self.assertIsNone(zero['tail_share_of_quantity_log_loss'])
        self.assertIsNone(zero['tail_share_of_raw_quantity_sse'])


class ComparisonTests(unittest.TestCase):
    def test_all_seeds_controls_and_both_scopes_retained(self):
        pairs=A.contrasts(rows())
        self.assertEqual(len(pairs),24)
        self.assertEqual({p['scope'] for p in pairs},{'all','quantity_gt_200'})
        self.assertEqual({p['seed'] for p in pairs},set(A.SEEDS))
        self.assertTrue(all(p['full_minus_control']['raw_rmse']>0 for p in pairs))
        self.assertTrue(all(p['full_minus_control']['log_mse_native']>0 for p in pairs))
        self.assertFalse(any(p['raw_worse_log_better'] for p in pairs))

    def test_matrix_or_truth_history_drift_rejected(self):
        value=rows()
        with self.assertRaisesRegex(ValueError,'matrix'):A.contrasts(value[:-1])
        value[1]['truth_history_sha256']='drift'
        with self.assertRaisesRegex(ValueError,'population'):A.contrasts(value)

    def test_seed_summary_has_sample_sd_and_baseline_invariance(self):
        summary=A.seed_summary(rows())
        self.assertEqual(len(summary),6)
        self.assertTrue(all(s['seed_count']==3 for s in summary))
        self.assertTrue(all(s['scopes']['quantity_gt_200']['model']['raw_rmse']['sample_sd']>0 for s in summary))
        self.assertTrue(all(s['scopes']['all']['last']['raw_rmse']['sample_sd']==0 for s in summary))
        report=A.render_report({},rows(),A.contrasts(rows()),summary)
        self.assertIn('실제 per-target float32',report)
        self.assertIn('Test 평가0',report)
        self.assertIn('조건',report)

    def test_changed_contract_pin_rejects_before_model_or_original_read(self):
        with tempfile.TemporaryDirectory(prefix='raf_analysis_negative_') as directory:
            root=Path(directory)
            (root/'contract.json').write_text(json.dumps({'schema':A.KIND,'contract_sha256':'drift'}))
            with self.assertRaisesRegex(ValueError,'contract pin'):A.collect(root,'unavailable')
            self.assertFalse((root/'analysis_claim.json').exists())

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory(prefix='raf_safe_negative_') as directory:
            for relative in ('../out','/tmp/out'):
                with self.assertRaisesRegex(ValueError,'escapes'):A.safe(Path(directory),relative)


if __name__=='__main__':unittest.main()
