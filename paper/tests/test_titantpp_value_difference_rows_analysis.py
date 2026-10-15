import copy
import unittest
from paper.scripts import analyze_titantpp_value_difference_rows as a

class PairedAnalysis(unittest.TestCase):
    def fixtures(self, split='validation', seed=42):
        h={'entity_id':'a','seq':4,'split':split,'raw_quantity':5.,'history_length':3,'last_quantity':4.,'mean_quantity':3.,'history_log_mean':1.,'history_log_std':.2,'last_log_deviation':.3,'last_observed_log_change':.2,'previous_observation_available':True,'d1':.2,'d2':-.1}
        b={**h,'dataset':'raf_spare_parts','seed':seed,'prediction_1':4.,'time_nll_1':2.}
        c={**h,'dataset':'raf_spare_parts','seed':seed,'prediction':7.,'time_nll':3.,'signed_error':2.,'absolute_error':2.,'squared_error':4.}
        return c,b,h
    def test_join_hand_calculation(self):
        c,b,h=self.fixtures();r=a.paired_rows([c],[b],[h],seed=42,split='validation',expected_count=1)[0]
        self.assertEqual((r['delta_ae'],r['delta_se'],r['delta_signed']),(1.,3.,3.))
        s=a.scores([r],[1]);self.assertEqual(s['candidate']['mae'],2.);self.assertEqual(s['difference']['rmse'],1.)
    def test_fail_closed(self):
        for field,value in [('split','test'),('seed',43),('dataset','taxi'),('raw_quantity',6.),('last_quantity',9.),('squared_error',3.),('prediction',float('nan'))]:
            c,b,h=self.fixtures();c[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):a.paired_rows([c],[b],[h],seed=42,split='validation',expected_count=1)
    def test_duplicates_rejected(self):
        c,b,h=self.fixtures()
        with self.assertRaises(ValueError):a.paired_rows([c,c],[b,b],[h,h],seed=42,split='validation',expected_count=2)
    def test_fixed_patterns(self):
        _,_,h=self.fixtures();self.assertEqual(a.fixed_features(h,(.2,1.)),('reversal','low'))
        h['d2']=0.;self.assertEqual(a.fixed_features(h,(.1,1.)),('stagnation','middle'))
        h['d2']=.1;self.assertEqual(a.fixed_features(h,(.1,.15)),('same_sign','high'))
        h['history_length']=1;self.assertEqual(a.fixed_features(h,(.1,.15)),('insufficient_history','unavailable'))
    def test_full_cohort_closure(self):
        cohort={}
        for seed in a.SEEDS:
            for split in a.SPLITS:
                c,b,h=self.fixtures(split,seed);cohort[seed,split]=a.paired_rows([c],[b],[h],seed=seed,split=split,expected_count=1)
        result=a.summarize(cohort,(.2,1.));self.assertEqual(len(result['per_seed']),180);self.assertEqual(len(result['additive_closure']),12)
        self.assertTrue(all(x['passed'] for x in result['additive_closure']))
        self.assertEqual(len(result['support']),90)
    def test_entity_weights(self):
        c,b,h=self.fixtures();r=a.paired_rows([c],[b],[h],seed=42,split='validation',expected_count=1)[0]
        second=copy.deepcopy(r);second.update(entity_id='b',candidate_error=0.,cnn_error=0.,delta_ae=0.,delta_se=0.,delta_signed=0.)
        s=a.scores([r,r,second],[.5,.5,1.]);self.assertEqual(s['candidate']['mae'],1.);self.assertEqual(s['difference']['ae_mean'],.5)
    def test_zero_reference_relative(self):
        c,b,h=self.fixtures();b['prediction_1']=5.;r=a.paired_rows([c],[b],[h],seed=42,split='validation',expected_count=1)[0]
        self.assertIsNone(a.scores([r],[1.])['difference']['mae_relative_percent'])

if __name__=='__main__':unittest.main()
