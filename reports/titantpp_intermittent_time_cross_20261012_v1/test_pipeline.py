"""Actual tiny local CPU child supervision; no GPU/model/data inference."""
from pathlib import Path
from unittest import mock
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('joint_pipeline_fixture',ROOT/'paper/scripts/run_titantpp_intermittent_time_cross_pipeline.py')
P=importlib.util.module_from_spec(s);s.loader.exec_module(P)

class PipelineTests(unittest.TestCase):
    def fixture(self,tmp,code):
        root=Path(tmp);(root/'control').mkdir();(root/'source').mkdir()
        (root/'control/evaluate.py').write_text(code)
        contract={'contract_sha256':'1'*64,'jobs':[{'id':'fixture_inter42','study':'history','seed':42}],
                  'runtime':{'python':sys.executable,'environment':{}},'sources':{'history':{'revision':'fixture','root':'source'}},
                  'execution':{'per_job_timeout_seconds':10,'total_timeout_seconds':30}}
        return root,contract,{'active':[],'completed_joint_workers':[]}

    def test_actual_one_child_exit_shared_pass_metadata_and_repeat_block(self):
        with tempfile.TemporaryDirectory(prefix='joint_supervision_fixture_') as tmp:
            root,c,state=self.fixture(tmp,"print('fabricated CPU worker; no inference')\n")
            P.run_joint(c,root,42,'train',time.monotonic(),state)
            dest=root/'results/seed42/train_joint';launch=P.read(dest/'launch.json');exit=P.read(dest/'worker_exit.json')
            self.assertEqual(state['active'],[]);self.assertEqual(state['completed_joint_workers'],['seed42__train'])
            self.assertEqual(exit['pid'],launch['pid']);self.assertEqual(exit['ppid'],os.getpid());self.assertEqual(exit['exit_code'],0)
            self.assertTrue(exit['actual_child_exit_observed']);self.assertEqual(exit['argv'],launch['argv'])
            self.assertEqual(exit['argv'][0],sys.executable);self.assertEqual(exit['model_split_pass_count_expected'],3)
            self.assertEqual(launch['resident_models_max'],3)
            with self.assertRaises(FileExistsError):P.run_joint(c,root,42,'train',time.monotonic(),state)

    def test_timeout_stops_only_its_actual_owned_child_and_preserves_exit(self):
        with tempfile.TemporaryDirectory(prefix='joint_timeout_fixture_') as tmp:
            root,c,state=self.fixture(tmp,'import time\ntime.sleep(15)\n')
            c['execution']['per_job_timeout_seconds']=0
            with self.assertRaisesRegex(ValueError,'no automatic retry'):P.run_joint(c,root,42,'validation',time.monotonic(),state)
            r=P.read(root/'results/seed42/validation_joint/worker_exit.json')
            self.assertNotEqual(r['exit_code'],0);self.assertTrue(r['timed_out']);self.assertEqual(state['completed_joint_workers'],[])
            with self.assertRaises(ProcessLookupError):os.kill(r['pid'],0)

    def test_foreign_active_slot_launches_no_extra_child(self):
        with tempfile.TemporaryDirectory(prefix='joint_active_fixture_') as tmp:
            root,c,state=self.fixture(tmp,'print(1)\n');state['active']=[{'pid':99999}]
            with mock.patch.object(P.subprocess,'Popen') as popen:
                with self.assertRaisesRegex(ValueError,'one joint'):P.run_joint(c,root,42,'train',time.monotonic(),state)
                popen.assert_not_called()

    def test_process_sysargv_vs_launchargv_and_ppid_bridge(self):
        c={'contract_sha256':'f'*64,'runtime':{'python':'/pinned/python'}}
        receipt={'pid':100,'ppid':50,'contract_sha256':'f'*64,'argv':['evaluate.py','--seed','42'],'completed_unix':20.}
        x={**receipt,'exit_code':0,'actual_child_exit_observed':True,'argv':['/pinned/python',*receipt['argv']],'completed_unix':21.}
        P.process_bridge(x,receipt,c)
        x['ppid']=1
        with self.assertRaisesRegex(ValueError,'bridge'):P.process_bridge(x,receipt,c)

if __name__=='__main__':unittest.main()
