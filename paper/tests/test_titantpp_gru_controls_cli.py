"""Standalone CLI must register GRU controls without campaign hooks."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = "import json,tempfile,pathlib,sys\nfrom unittest.mock import patch\nfrom paper.scripts import evaluate_titantpp_gru_controls as evaluation\nfrom paper.scripts import run_multilag_detail_execution as base\nfrom paper.scripts import quantity_comparison_runtime as runtime\nfrom paper.scripts import titantpp_gru_control_checks as checks\nfrom models.TPPs.CountAwareFactory import build_count_aware_model\n# Fresh process: no installed GRU hooks before main().\ndata=checks.synthetic_data(84);data['dataset_id']='raf_spare_parts';data['inherited_data_identity']={'data':{'path':'inputs/data/raf.parquet','sha256':'x'},'split_manifest':{'path':'inputs/data/split.json','sha256':'y'}}\nwith tempfile.TemporaryDirectory() as folder:\n p=pathlib.Path(folder);contract=p/'execution_contract.json';contract.write_text(json.dumps({'datasets':[data]}));out=p/'out.json'\n def prepare(d):\n  assert d['inherited_data_identity']['data']['path']==str(p/'source/inputs/data/raf.parquet')\n  return None,None\n def evaluate(d,*a,**kw):\n  from models.TPPs import CountAwareFactory as factory\n  model,meta=factory.build_count_aware_model('titantpp_gru54_pair',**d['model'],train_log_mean=1.2,train_log_std=.8,max_seq_len=84)\n  assert model.candidate_arm=='titantpp_gru54_pair'\n  return {'route_passed':True,'new_training':False}\n with patch.object(sys,'argv',['evaluate','--contract',str(contract),'--dataset','raf_spare_parts','--checkpoint',str(p/'dummy.pt'),'--device','cpu','--output',str(out)]),patch.object(base,'prepare_admitted_data',side_effect=prepare),patch.object(runtime,'configure_runtime'),patch.object(evaluation,'evaluate_checkpoint',side_effect=evaluate):evaluation.main()\n assert json.loads(out.read_text())['route_passed']\nprint('standalone CLI routes pair GRU in a fresh process and resolves immutable inputs')\n"

class ControlCliTest(unittest.TestCase):
    def test_fresh_cli_routes_pair_and_resolves_portable_inputs(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-c', SCRIPT], cwd=root,
                                env={**os.environ, 'PYTHONPATH': str(root)},
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('standalone CLI routes pair GRU', result.stdout)

if __name__ == '__main__':
    unittest.main()
