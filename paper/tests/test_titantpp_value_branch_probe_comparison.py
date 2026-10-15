import copy
import statistics
import unittest
from unittest.mock import patch
from paper.scripts import compare_titantpp_value_branch_probe as c
from paper.scripts import analyze_titantpp_value_difference_rows as pure

class ProbeComparison(unittest.TestCase):
    def row(self,pred=7.,seed=42,split='validation'):
        h={'entity_id':'a','seq':4,'split':split,'raw_quantity':5.,'history_length':3,'last_quantity':4.,'mean_quantity':3.,'history_log_mean':1.,'history_log_std':.2,'last_log_deviation':.3,'last_observed_log_change':.2,'previous_observation_available':True,'d1':.2,'d2':-.1}
        e=pred-5.;r={**h,'dataset':'raf_spare_parts','seed':seed,'prediction':pred,'time_nll':2.,'signed_error':e,'absolute_error':abs(e),'squared_error':e*e,'target_id':'fixed','recorded_gap':2.,'recorded_date':'2020-01-01','observed_history_sha256':'history'}
        return r,h
    def test_off_minus_on_sign(self):
        off,h=self.row();on,_=self.row(4.)
        with patch.dict(c.COUNTS,validation=1):r=c.paired([off],[on],[h],pure,42,'validation')[0]
        self.assertEqual((r['delta_ae'],r['delta_se'],r['delta_signed']),(1.,3.,3.))
        score=c.rename_roles(pure.scores([r],[1.]))
        self.assertEqual(score['off']['rmse'],2.);self.assertEqual(score['on']['rmse'],1.)
        self.assertNotIn('cnn',score);self.assertNotIn('candidate',score)
    def test_equal_conditions_zero_delta(self):
        row,h=self.row()
        with patch.dict(c.COUNTS,validation=1):r=c.paired([row],[row],[h],pure,42,'validation')[0]
        self.assertEqual((r['delta_ae'],r['delta_se'],r['delta_signed']),(0.,0.,0.))
    def test_join_ignores_order_but_rejects_duplicate(self):
        a,_=self.row();b=copy.deepcopy(a);b.update(target_id='second',seq=5)
        self.assertTrue(c.identical_targets([b,a],[a,b]))
        with self.assertRaises(ValueError):c.identical_targets([a,a],[a,b])
    def test_truth_and_history_mismatch_rejected(self):
        a,h=self.row();b=copy.deepcopy(a)
        for field,value in [('target_id','other'),('seed',52),('split','train'),('raw_quantity',6.),('recorded_gap',3.),('recorded_date','2021'),('observed_history_sha256','changed')]:
            bad={**a,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):c.identical_targets([bad],[b])
        for field in pure.COMMON:
            if field=='raw_quantity':continue
            bad={**a,field:float(a[field])+1.}
            with self.subTest(field=field),patch.dict(c.COUNTS,validation=1),self.assertRaises(ValueError):c.paired([bad],[b],[h],pure,42,'validation')
    def test_both_condition_error_integrity(self):
        a,_=self.row()
        for role in ('on','off'):
            for field in ('signed_error','absolute_error','squared_error'):
                bad={**a,field:float('nan')}
                with self.subTest(role=role,field=field),self.assertRaises(ValueError):c.identical_targets([bad] if role=='off' else [a],[bad] if role=='on' else [a])
    def test_complete_cohort_and_no_role_confusion(self):
        cohort={}
        for seed in c.SEEDS:
            for split in c.SPLITS:
                off,h=self.row(seed=seed,split=split);on,_=self.row(4.,seed,split)
                with patch.dict(c.COUNTS,{split:1}):cohort[seed,split]=c.paired([off],[on],[h],pure,seed,split)
        summary=c.rename_roles(pure.summarize(cohort,(.2,1.)))
        self.assertEqual(len(summary['per_seed']),180);self.assertEqual(len(summary['additive_closure']),12)
        self.assertTrue(all(x['passed'] for x in summary['additive_closure']))
        whole=next(x for x in summary['three_seed'] if x['cell']=='whole|None|None' and x['weighting']=='row')
        self.assertEqual(whole['metrics']['off']['mae']['mean'],2.)
        self.assertEqual(whole['metrics']['on']['mae']['mean'],1.)
    def test_metric_mean_sd_and_paired_differences(self):
        evidence=[]
        for split in c.SPLITS:
            for i,seed in enumerate(c.SEEDS):
                on={'qty_rmse':10.+i,'qty_mae':5.+i,'time_nll':2.+i,'tail':{'qty_rmse':20.+i,'qty_mae':10.+i,'time_nll':4.+i}}
                off=copy.deepcopy(on)
                for target in (off,off['tail']):
                    for metric in ('qty_rmse','qty_mae','time_nll'):target[metric]+=i
                evidence.append({'seed':seed,'split':split,'on_metrics':on,'off_metrics':off})
        groups=c.metric_groups(evidence);self.assertEqual(len(groups),12)
        self.assertTrue(all(r['off_minus_on']=={'mean':1.,'sample_sd':1.} for r in groups))
        with self.assertRaises(ValueError):c.metric_groups(evidence[:-1])
    def test_empty_tail_unavailable(self):
        evidence=[]
        for split in c.SPLITS:
            for seed in c.SEEDS:
                m={'qty_rmse':1.,'qty_mae':1.,'time_nll':1.,'tail':{'qty_rmse':None,'qty_mae':None,'time_nll':None}}
                evidence.append({'seed':seed,'split':split,'on_metrics':m,'off_metrics':m})
        rows=c.metric_groups(evidence)
        self.assertTrue(all(r['off_minus_on'] is None for r in rows if r['population']=='tail'))

if __name__=='__main__':unittest.main()
