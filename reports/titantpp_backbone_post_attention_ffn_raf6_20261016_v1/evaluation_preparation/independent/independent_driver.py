from pathlib import Path
import unittest,sys,json,torch,polars,socket,subprocess
from unittest.mock import patch
import paper.scripts.run_titantpp_post_attention_ffn as engine
import paper.scripts.run_titantpp_adaptive_history_cnn as inherited
import paper.scripts.evaluate_titantpp_gru_controls as controls
engine.install_hooks();torch.set_num_threads(1)
root=Path(__file__).resolve().parent
for module in (engine,inherited,controls):assert Path(module.__file__).resolve().is_relative_to(root)
def blocked(*args,**kwargs):raise AssertionError('Independent synthetic review prohibits data/checkpoint/network/child processes')
print(json.dumps({'python':sys.version,'torch':torch.__version__,'guarded':True,'module_files':{m.__name__:m.__file__ for m in (engine,inherited,controls)}}),flush=True)
with patch.object(polars,'read_parquet',side_effect=blocked),patch.object(polars,'scan_parquet',side_effect=blocked),patch.object(torch,'load',side_effect=blocked),patch.object(socket.socket,'connect',side_effect=blocked),patch.object(subprocess,'Popen',side_effect=blocked):
 suite=unittest.defaultTestLoader.loadTestsFromName('paper.tests.test_titantpp_post_attention_ffn_raf6_test')
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 print(json.dumps({'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'successful':result.wasSuccessful(),'real_data_loaded':False,'checkpoint_loads':0,'remote_calls':0,'scientific_training':0,'historical_cnn_inference':0}),flush=True)
 sys.exit(0 if result.wasSuccessful() else 1)
