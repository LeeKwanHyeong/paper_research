import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'scripts/deploy_titantpp_retrieval_repair.py'
spec=importlib.util.spec_from_file_location('repair_deploy_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RepairDeploymentTests(unittest.TestCase):
    def test_preserved_attempt_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'B',Path(d)),patch.object(m,'verify',return_value={}):
            m.write(Path(d)/'deployment/transfer_intent_5080.json',{'automatic_retry':False})
            with patch.object(m,'ssh') as network,self.assertRaisesRegex(ValueError,'no automatic retry'):
                m.activate('5080')
            network.assert_not_called()

    def test_completed_launch_has_no_remote_call(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'B',Path(d)),patch.object(m,'verify',return_value={}):
            m.write(Path(d)/'deployment/launch_receipt_5090.json',{'host':'5090'})
            with patch.object(m,'ssh') as network:m.activate('5090')
            network.assert_not_called()

    def test_clock_skew_uses_local_receive_and_retains_evidence(self):
        class Reply:
            returncode=0;stderr='';stdout=json.dumps({'observed_unix':102.0,'host':'5080'})
        with tempfile.TemporaryDirectory() as d,patch.object(m,'B',Path(d)),patch.object(m.subprocess,'run',return_value=Reply()),patch.object(m.time,'time_ns',side_effect=[99_000_000_000,100_000_000_000]):
            got=m.ssh('5080','print(1)','probe')
            self.assertEqual(got['local_received_unix_ns'],100_000_000_000)
            self.assertEqual(len(list((Path(d)/'deployment').glob('*.json'))),1)

    def test_large_clock_skew_blocks_and_preserves_result(self):
        class Reply:
            returncode=0;stderr='';stdout=json.dumps({'observed_unix':120.0})
        with tempfile.TemporaryDirectory() as d,patch.object(m,'B',Path(d)),patch.object(m.subprocess,'run',return_value=Reply()),patch.object(m.time,'time_ns',side_effect=[99_000_000_000,100_000_000_000]):
            with self.assertRaisesRegex(ValueError,'clock skew'):m.ssh('5080','print(1)','probe')
            self.assertEqual(len(list((Path(d)/'deployment').glob('*.json'))),1)

    def test_remote_prefix_is_valid_python_and_exact_timeout_only(self):
        code=m.remote_prefix({'hosts':{},'parent_contract':{}},'5090')
        compile(code,'remote_gate','exec')
        self.assertIn("Worker exited 124; no automatic retry",code)
        self.assertIn("read(old/'supervisor_process_exit.json')['returncode']==1",code)
        self.assertNotIn('kill(',code)

if __name__=='__main__':unittest.main()
