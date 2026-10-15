import copy
import math
import random
import unittest
from paper.scripts import analyze_titantpp_qk_probe_error_tradeoff as t
from paper.scripts import analyze_titantpp_value_difference_rows as pure


def row(e,o,entity='a',seq=0,q=10.,d1=.1,d2=-.1,history_length=3):
    return {'entity_id':entity,'seq':seq,'raw_quantity':q,'cnn_error':e,'candidate_error':o,
            'cnn_prediction':q+e,'candidate_prediction':q+o,'cnn_time_nll':2.,'candidate_time_nll':3.,
            'd1':d1,'d2':d2,'history_length':history_length,
            'delta_ae':abs(o)-abs(e),'delta_se':o*o-e*e,'delta_signed':o-e}


class TradeoffArithmetic(unittest.TestCase):
    def test_identity_and_direction(self):
        x=t.row_terms(row(-5.,-2.))
        self.assertEqual((x['shift'],x['cross_term'],x['squared_shift'],x['delta_ae'],x['delta_se']),(3.,-30.,9.,-3.,-21.))
        self.assertEqual(x['effect'],'both_improve')
        self.assertEqual(t.row_terms(row(1.,3.))['effect'],'both_worsen')
    def test_crossing_partition_exact_zero(self):
        examples=[(-1,1,'negative_to_positive'),(1,-1,'positive_to_negative'),(1,0,'arrives_zero'),(-1,0,'arrives_zero'),(0,1,'leaves_zero'),(0,-1,'leaves_zero'),(0,0,'zero_to_zero'),(1,2,'no_crossing'),(-1,-2,'no_crossing')]
        for e,o,expected in examples:
            with self.subTest(e=e,o=o):self.assertEqual(t.row_terms(row(e,o))['zero_crossing'],expected)
        self.assertEqual(t.row_terms(row(-1,1))['effect'],'equal')
    def test_signs_and_no_synthetic_opposite_outcomes(self):
        for e in (-10.,-1.,0.,1.,10.):
            for o in (-10.,-1.,0.,1.,10.):
                x=t.row_terms(row(e,o));self.assertEqual(t.sign(x['delta_ae']),t.sign(x['delta_se']))
                self.assertEqual(x['on_error_sign'],'negative' if e<0 else 'positive' if e>0 else 'zero')
    def test_nonfinite_and_shift_inconsistent_fail(self):
        for key in ('cnn_error','candidate_error','cnn_time_nll','candidate_prediction'):
            bad=row(1.,2.);bad[key]=float('nan')
            with self.subTest(key=key),self.assertRaises(ValueError):t.row_terms(bad)
        bad=row(1.,2.);bad['candidate_prediction']=18.
        with self.assertRaises(ValueError):t.row_terms(bad)
    def test_aggregate_mae_sse_disagreement_different_rows(self):
        rows=[row(-10.,-9.,seq=0),row(0.,2.,seq=1)]
        x=t.one_cell(rows,[1.,1.]);s=x['total']['sums']
        self.assertEqual((s['delta_ae'],s['delta_se']),(1.,-15.))
        self.assertEqual(x['partitions']['effect']['both_improve']['count'],1)
        self.assertEqual(x['partitions']['effect']['both_worsen']['count'],1)
        self.assertEqual(x['total']['numerical_sign_mismatch_count'],0)
    def test_subpartition_preserves_parent_weights(self):
        rows=[row(2.,1.,'a',0),row(1.,3.,'a',1),row(1.,3.,'a',2),row(2.,1.,'b',0)]
        x=t.one_cell(rows,[1/3,1/3,1/3,1.])
        improve=x['partitions']['effect']['both_improve']
        self.assertAlmostEqual(improve['weight_sum'],4/3)
        self.assertAlmostEqual(improve['sums']['delta_ae'],-4/3)
        self.assertAlmostEqual(improve['contribution_per_parent_weight']['delta_ae'],-2/3)
        self.assertAlmostEqual(x['total']['weight_sum'],2.)
    def test_tail_boundary_and_empty(self):
        rows=[row(1,2,q=200),row(1,2,q=201)]
        x=t.one_cell(rows,[1.,1.]);self.assertEqual([v['count'] for v in x['partitions']['quantity'].values()],[1,1])
        empty=t.one_cell([],[]);self.assertEqual(empty['total']['count'],0);self.assertIsNone(empty['total']['on_rmse'])
        self.assertTrue(all(c['passed'] for c in empty['closures']))
    def test_complete15_and_frozen_common_entities(self):
        cohorts={}
        for seed in t.SEEDS:
            for split in t.SPLITS:
                cohorts[seed,split]=[row(2.,1.,'a',0),row(1.,3.,'a',1),row(2.,1.,'b',0),row(1.,1.,split+'Only',0)]
        x=t.decompose(cohorts,(.2,1.),pure);prior=pure.summarize(cohorts,(.2,1.))
        self.assertEqual(len(x['per_seed']),180);self.assertEqual(x['support'],prior['support'])
        for cell in x['per_seed']:
            old=next(r for r in prior['per_seed'] if all(r[k]==cell[k] for k in ('cell','seed','split','weighting')))
            if old['metrics'] is None:self.assertEqual(cell['total']['count'],0);continue
            for k,j in [('delta_ae','ae_sum'),('delta_se','se_sum'),('delta_signed','signed_sum')]:
                self.assertAlmostEqual(cell['total']['sums'][k],old['metrics']['difference'][j])
    def test_incomplete_cohort_rejected(self):
        with self.assertRaises(ValueError):t.decompose({},(.2,1.),pure)
    def test_near_zero_is_not_tolerance_reclassified(self):
        x=t.row_terms(row(1.,1.+2**-40,q=0.))
        self.assertEqual(x['effect'],'both_worsen')
        self.assertLess(x['delta_ae'],t.ATOL)
    def test_random_scalar_reference_and_permutation(self):
        for trial in range(30):
            rng=random.Random(trial);rows=[row(rng.uniform(-100,100),rng.uniform(-100,100),str(i%4),i,q=200.+(i%2)) for i in range(40)];weights=[1./(1+i%5) for i in range(40)]
            x=t.one_cell(rows,weights)
            self.assertAlmostEqual(x['total']['sums']['delta_se'],math.fsum(w*(r['candidate_error']**2-r['cnn_error']**2) for w,r in zip(weights,rows)),places=7)
            self.assertAlmostEqual(x['total']['sums']['delta_ae'],math.fsum(w*(abs(r['candidate_error'])-abs(r['cnn_error'])) for w,r in zip(weights,rows)),places=9)
            for r in rows:
                a=t.row_terms(r);self.assertEqual(t.sign(a['delta_ae']),t.sign(a['delta_se']))
            order=list(range(40));rng.shuffle(order);y=t.one_cell([rows[i] for i in order],[weights[i] for i in order]);self.assertEqual(x,y)
    def test_no_write_or_native_in_pure_functions(self):
        from unittest.mock import patch
        with patch('builtins.open',side_effect=AssertionError('Unexpected I/O')):
            t.one_cell([row(2,1)],[1.])
    def test_bad_weights_rejected(self):
        for w in (0.,-1.,float('nan')):
            with self.subTest(w=w),self.assertRaises(ValueError):t.one_cell([row(1,2)],[w])

if __name__=='__main__':unittest.main()
