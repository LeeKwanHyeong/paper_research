import unittest
from types import SimpleNamespace
import numpy as np
import torch
from history_context_diagnostics import context_arrays, align_context, decompose, symmetric_mix


class ContextTests(unittest.TestCase):
    def test_truncation_includes_one_target_slot(self):
        ds=SimpleNamespace(seq_lists=[[1,2,3,4,5]],dt_lists=[[1,1,2,1,3]],
            val_lists=[[10,20,30,40,50]],index=[(0,3)],W=100,max_len=3)
        c=context_arrays(ds)
        self.assertEqual(c['history_count'].tolist(),[2])
        self.assertEqual(c['history_mean_qty'].tolist(),[35])
        self.assertEqual(c['qty'].tolist(),[50])
        self.assertEqual(c['previous_qty'].tolist(),[40])

    def test_sparse_window_is_sequence_not_elapsed_duration(self):
        ds=SimpleNamespace(seq_lists=[[1,4,8,10]],dt_lists=[[1,3,4,2]],
            val_lists=[[10,20,30,40]],index=[(0,2)],W=5,max_len=8)
        c=context_arrays(ds)
        self.assertEqual(c['history_count'].tolist(),[2])
        self.assertEqual(c['history_mean_qty'].tolist(),[25])

    def test_order_mismatch_rejected(self):
        c={'dt':np.array([1,2]),'qty':np.array([10,20]),'history_count':np.array([1,2])}
        with self.assertRaises(ValueError):
            align_context(c,{'dt':torch.tensor([2.,1.]),'qty':torch.tensor([20.,10.])})

    def test_contributions_use_population_weight(self):
        d=decompose([1,3,2],[2,5,2],[('a',np.array([True,True,False])),('b',np.array([False,False,True]))])
        self.assertAlmostEqual(d['rows'][0]['delta_contribution_to_overall_mean'],1)
        self.assertEqual(d['rows'][1]['delta_contribution_to_overall_mean'],0)
        self.assertLess(max(abs(v) for v in d['reconciliation_residuals'].values()),1e-12)

    def test_overlapping_cohorts_rejected(self):
        with self.assertRaises(ValueError):
            decompose([1],[2],[('a',np.array([True])),('b',np.array([True]))])

    def test_mix_only_when_within_group_losses_fixed(self):
        train=[{'label':'a','population_fraction':.8,'e0_mean_nll':1,'e0_contribution_to_overall_mean':.8},
               {'label':'b','population_fraction':.2,'e0_mean_nll':3,'e0_contribution_to_overall_mean':.6}]
        val=[{'label':'a','population_fraction':.2,'e0_mean_nll':1,'e0_contribution_to_overall_mean':.2},
             {'label':'b','population_fraction':.8,'e0_mean_nll':3,'e0_contribution_to_overall_mean':2.4}]
        m=symmetric_mix(train,val)
        self.assertAlmostEqual(m['mix_component'],1.2)
        self.assertAlmostEqual(m['within_stratum_component'],0)
        self.assertAlmostEqual(m['sum'],1.2)

    def test_unmatched_stratum_kept_separate(self):
        m=symmetric_mix([{'label':'a','population_fraction':0,'e0_mean_nll':None,'e0_contribution_to_overall_mean':0}],
                        [{'label':'a','population_fraction':1,'e0_mean_nll':3,'e0_contribution_to_overall_mean':3}])
        self.assertEqual(m['unmatched_component'],3)


if __name__=='__main__':
    unittest.main()
