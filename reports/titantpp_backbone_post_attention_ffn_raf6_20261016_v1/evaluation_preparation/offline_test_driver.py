from pathlib import Path
import unittest,sys,json,torch,polars,socket,subprocess
from unittest.mock import patch
import paper.scripts.run_titantpp_post_attention_ffn as engine
import paper.scripts.run_titantpp_adaptive_history_cnn as historical
import paper.scripts.evaluate_titantpp_gru_controls as controls
import models.TPPs.CountAwareTitanPostAttentionFFN as model
import importlib
model=importlib.import_module("models.TPPs.CountAwareTitanPostAttentionFFN")
engine.install_hooks()
root=Path(__file__).resolve().parent
for module in (engine,historical,controls,model):
 path=Path(module.__file__).resolve();assert path.is_relative_to(root),path
print(json.dumps({"runtime":{"python":sys.version,"torch":torch.__version__},"module_files":{m.__name__:m.__file__ for m in (engine,historical,controls,model)},"real_data_read_guard":True,"torch_load_guard":True,"remote_guard":True}),flush=True)
def blocked(*a,**kw):raise AssertionError("Real data/checkpoint/remote/child process access forbidden in offline evaluation tests")
with patch.object(polars,"read_parquet",side_effect=blocked),patch.object(torch,"load",side_effect=blocked),patch.object(socket.socket,"connect",side_effect=blocked),patch.object(subprocess,"Popen",side_effect=blocked):
 suite=unittest.defaultTestLoader.loadTestsFromName("paper.tests.test_titantpp_post_attention_ffn_raf6_test")
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 print(json.dumps({"tests_run":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),"successful":result.wasSuccessful()}),flush=True)
 sys.exit(0 if result.wasSuccessful() else 1)
