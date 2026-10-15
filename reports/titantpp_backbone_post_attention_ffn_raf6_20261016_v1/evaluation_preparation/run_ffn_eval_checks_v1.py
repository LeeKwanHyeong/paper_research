from pathlib import Path
import shutil,json,hashlib,subprocess,sys,ast,os,time
project=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
base=Path('/private/tmp/ffn_model_frozen147_v1')
root=Path('/private/tmp/ffn_eval_frozen147_v1')
logs=Path('/private/tmp/ffn_eval_runs_v1');logs.mkdir(exist_ok=True)
files=['paper/scripts/evaluate_titantpp_post_attention_ffn_frozen.py','paper/scripts/run_titantpp_post_attention_ffn_test_pipeline.py','paper/scripts/prepare_titantpp_post_attention_ffn_raf6_test.py','paper/scripts/advance_titantpp_post_attention_ffn_raf6_test.py','paper/tests/test_titantpp_post_attention_ffn_raf6_test.py']
ops=['paper/scripts/compare_titantpp_backbone_loop.py','paper/scripts/advance_titantpp_post_attention_ffn_raf6.py']
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not root.exists():shutil.copytree(base,root)
for name in files+ops:
 ast.parse((project/name).read_text());dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(project/name,dst)
raw=(base/'PARENT_FILE_SHA.json').read_text();parent=json.loads(raw)
(root/'_eval_test_driver.py').write_text('''from pathlib import Path
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
''')
attempt=len(list(logs.glob('attempt*.json')))+1
before={str(p.relative_to(root)):h(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
command=['/usr/local/bin/python3','-B',str(root/'_eval_test_driver.py')]
started=time.time()
with (logs/f'attempt{attempt}.stdout').open('xb') as out,(logs/f'attempt{attempt}.stderr').open('xb') as err:
 result=subprocess.run(command,cwd=root,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':'/private/tmp/ffn_eval_mpl','XDG_CACHE_HOME':'/private/tmp/ffn_eval_cache'},stdout=out,stderr=err)
after={str(p.relative_to(root)):h(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
receipt={'command':command,'cwd':str(root),'actual_exit':result.returncode,'started_unix':started,'completed_unix':time.time(),'project_code_sha':{n:h(project/n) for n in files},'ops_dependency_sha':{n:h(project/n) for n in ops},'overlay_before':before,'overlay_after':after,'overlay_unchanged':before==after,'stdout':{'path':str(logs/f'attempt{attempt}.stdout'),'sha256':h(logs/f'attempt{attempt}.stdout')},'stderr':{'path':str(logs/f'attempt{attempt}.stderr'),'sha256':h(logs/f'attempt{attempt}.stderr')}}
(logs/f'attempt{attempt}.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'attempt':attempt,'actual_exit':result.returncode,'overlay_unchanged':before==after,'stdout':receipt['stdout'],'stderr':receipt['stderr']}))
print((logs/f'attempt{attempt}.stdout').read_text());print((logs/f'attempt{attempt}.stderr').read_text())
