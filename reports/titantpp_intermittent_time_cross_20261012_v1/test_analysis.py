"""Stdlib fabricated sufficient-statistic accounting regressions."""
from pathlib import Path
import copy
import importlib.util
import math
import unittest
ROOT=Path(__file__).resolve().parents[2]

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=load(ROOT/'paper/scripts/analyze_titantpp_intermittent_time_cross.py','time_cross_analysis_fixture')

class AnalysisTests(unittest.TestCase):
    def pair(self):
        a=A.E.CrossAccumulator([1.,3.,5.,7.]);a.add([1.,2.,8.],{'ff':[1.,3.,9.],'fc':[2.,2.,4.],'cf':[4.,5.,6.],'cc':[5.,3.,3.]})
        return {'full_arm':A.FULL,'control_arm':A.ARMS[1],'seed':42,'split':'train','cells':a.finish()}

    def test_actual_cross_aggregate_identity_and_negative_controls(self):
        p=self.pair();self.assertEqual(len(A.score_cells(p,3)),4)
        for mutate,message in ((lambda p:p['cells'][0]['means'].__setitem__('scale_effect',1.),'Reconstruction'),
                                 (lambda p:p['cells'][0]['squares'].__setitem__('ff',0.),'second moment'),
                                 (lambda p:p['cells'].append(copy.deepcopy(p['cells'][0])),'Duplicate'),
                                 (lambda p:p['cells'][1].__setitem__('count',8),'coverage'),
                                 (lambda p:p.__setitem__('split','test'),'Foreign')):
            altered=copy.deepcopy(p);mutate(altered)
            with self.assertRaisesRegex(ValueError,message):A.score_cells(altered,3)

    def test_three_seed_sample_sd_not_population_sd(self):
        r=A.mean_sd([1.,2.,3.]);self.assertEqual(r,{'mean':2.,'sample_sd':1.,'seed_count':3})
        with self.assertRaisesRegex(ValueError,'Three-seed'):A.mean_sd([1.,2.])

    def test_actual_interpreter_and_shared_joint_pid_bridge(self):
        c={'contract_sha256':'f'*64,'runtime':{'python':'/pinned/python'}}
        receipt={'pid':100,'ppid':50,'contract_sha256':'f'*64,'argv':['/isolated/evaluate.py','--seed','42','--split','train'],'completed_unix':20.}
        actual={**receipt,'argv':['/pinned/python',*receipt['argv']],'exit_code':0,'completed_unix':21.}
        A.process_bridge(actual,receipt,c)
        altered=copy.deepcopy(actual);altered['argv']=receipt['argv']
        with self.assertRaisesRegex(ValueError,'argv/interpreter'):A.process_bridge(altered,receipt,c)
        altered=copy.deepcopy(actual);altered['ppid']=99
        with self.assertRaisesRegex(ValueError,'PID/PPID'):A.process_bridge(altered,receipt,c)

    def test_render_contains_counterfactual_and_no_new_test_claim(self):
        p=self.pair();rows=[{'seed':42,'split':'train','control_arm':A.ARMS[1],**c} for c in p['cells']]
        report=A.render({'rows':rows})
        self.assertIn('counterfactual',report);self.assertIn('새 Test추론0',report)
        self.assertIn('해결됐다고 말할 수 없',report);self.assertIn('≤1',report)
        self.assertNotIn('Test62.572',report)

if __name__=='__main__':unittest.main()
