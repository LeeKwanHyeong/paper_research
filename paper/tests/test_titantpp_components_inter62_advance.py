"""Synthetic waiting/argv/evidence guards; actual SSH/data/model evaluation0."""
import ast,hashlib,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from paper.scripts import advance_titantpp_components_inter62_test as advance
from paper.scripts.titantpp_inter62_transport_evidence import recorded_run

class Guards(unittest.TestCase):
 def test_waiting_round_never_reads_training_snapshot_or_calls_remote(self):
  with tempfile.TemporaryDirectory() as tmp:
   monitor=Path(tmp).resolve();(monitor/'rounds').mkdir();p=monitor/'rounds/x.json'
   p.write_text(json.dumps({'monitor_integrity_raw_sha256':'4f2ab78970d862cf146365de6a75999950964c2d809c27183d1a970f37fedb40','all_training_terminal_verified':False,'groups':{'handoff_5080':{'observed_KST':'synthetic','snapshot_path':'does_not_exist'}}}))
   with patch.object(advance,'MONITOR',monitor),patch.object(advance,'transport',side_effect=AssertionError('Unexpected SSH')):
    r,g,s=advance.terminal_gate(p,advance.sha(p));self.assertIsNone(s)
 def test_tampered_round_rejected_before_remote(self):
  with tempfile.TemporaryDirectory() as tmp:
   monitor=Path(tmp).resolve();(monitor/'rounds').mkdir();p=monitor/'rounds/x.json';p.write_text('{}')
   with patch.object(advance,'MONITOR',monitor):
    with self.assertRaisesRegex(ValueError,'SHA'):advance.terminal_gate(p,'0'*64)
 def test_native_ai_env_argv_contains_only_one_python_and_no_audit_arguments(self):
  path=Path('reports/titantpp_components_inter62_test_20261014_v1/run_native_evaluation.py');tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluation_argv');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
  python='/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12';root='/synthetic/inter62';argv=ns['evaluation_argv']({'runtimes':{'5080':{'python':python}}},{'hosts':{'5080':{'environment':{'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':'oldsource'}}}},root)
  self.assertEqual(argv.count(python),1);self.assertNotIn('--audit-attempt',argv);self.assertEqual(argv[-4:],[python,root+'/operation/pipeline.py','--root',root]);self.assertIn('PYTHONPATH='+root+'/source',argv)
 def test_zero_and_nonzero_transport_preserve_raw_and_actual_exit(self):
  for code in (0,1):
   with self.subTest(code=code),tempfile.TemporaryDirectory() as tmp:
    result=subprocess.CompletedProcess(['synthetic'],code,b'raw-out\x00',b'raw-error\xff')
    with patch('paper.scripts.titantpp_inter62_transport_evidence.subprocess.run',return_value=result):actual=recorded_run(tmp,'probe',['synthetic'])
    p=Path(tmp)/'control';r=json.loads((p/'probe_exit.json').read_text());self.assertEqual(actual.returncode,code);self.assertTrue(r['actual_transport_exit_observed']);self.assertFalse(r['timeout']);self.assertEqual((p/'probe_stderr.bin').read_bytes(),result.stderr)
 def test_timeout_preserves_partial_raw_and_never_claims_remote_exit(self):
  with tempfile.TemporaryDirectory() as tmp:
   error=subprocess.TimeoutExpired(['synthetic'],10,output=b'partial out',stderr=b'partial error')
   with patch('paper.scripts.titantpp_inter62_transport_evidence.subprocess.run',side_effect=error):
    with self.assertRaises(subprocess.TimeoutExpired):recorded_run(tmp,'probe',['synthetic'],timeout=10)
   p=Path(tmp)/'control';r=json.loads((p/'probe_exit.json').read_text());self.assertTrue(r['timeout']);self.assertFalse(r['actual_transport_exit_observed']);self.assertFalse(r['remote_exit_inferred_from_timeout']);self.assertIsNone(r['actual_transport_exit']);self.assertEqual((p/'probe_stdout.bin').read_bytes(),b'partial out');self.assertEqual((p/'probe_stderr.bin').read_bytes(),b'partial error');self.assertEqual(r['stdout_sha256'],hashlib.sha256(b'partial out').hexdigest())

if __name__=='__main__':unittest.main()
