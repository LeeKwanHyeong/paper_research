import hashlib,json,shlex,subprocess,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper.scripts.titantpp_inter62_transport_evidence import recorded_run
b=Path('search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1')
c=json.loads((b/'contract.json').read_text());old=json.loads((b/'control/native_audit_intent.json').read_text())
r=c['evaluation_root_mapping']['authorized_remote_root']
def evaluation_argv(contract,training,root):
 env={**training['hosts']['5080']['environment'],'PYTHONPATH':root+'/source','MPLCONFIGDIR':root+'/evaluation_runtime_cache/matplotlib','XDG_CACHE_HOME':root+'/evaluation_runtime_cache'}
 return ['env',*[k+'='+v for k,v in sorted(env.items())],contract['runtimes']['5080']['python'],root+'/operation/pipeline.py','--root',root]
a=evaluation_argv(c,json.loads((b/'provenance/training_contract.json').read_text()),r)
p={'root':r,'argv':a,'contract_sha256':c['contract_sha256'],'approval_sha256':c['partial_approval_sha256'],'pipeline_sha256':c['evaluator_integrity']['files']['operation/pipeline.py']}
code="""import hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['root']);assert str(r)=="/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_taxi_raf24_test_20261014_v1_5080"
assert hashlib.sha256((r/'partial_approval.json').read_bytes()).hexdigest()==p['approval_sha256'];assert hashlib.sha256((r/'operation/pipeline.py').read_bytes()).hexdigest()==p['pipeline_sha256']
assert json.loads((r/'contract.json').read_text())['contract_sha256']==p['contract_sha256']
assert not (r/'pipeline.claim').exists() and not (r/'pipeline_actual_exit.json').exists()
spec=importlib.util.spec_from_file_location('inter62_controller',r/'operation/pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
start=time.time();child=None;outcome=None
with (r/'pipeline_supervision_claim.json').open('x') as f:json.dump({'pid':os.getpid(),'argv':p['argv'],'started_unix':start,'automatic_retry':False},f)
def term(signum,frame):raise KeyboardInterrupt('wrapper received signal '+str(signum))
signal.signal(signal.SIGTERM,term);signal.signal(signal.SIGHUP,term)
try:
 child=subprocess.Popen(p['argv'],cwd=r,start_new_session=True)
 code=child.wait(timeout=43200);outcome={'actual_child_exit_observed':True,'exit_code':code,'cleanup_actions':[]}
except BaseException:
 signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGHUP,signal.SIG_IGN)
 if child is not None:outcome=m.stop_owned_child(child)
 raise
finally:
 if outcome is not None:
  receipt={'contract_sha256':p['contract_sha256'],'pipeline_pid':child.pid,'supervisor_pid':os.getpid(),'argv':p['argv'],'started_unix':start,'completed_unix':time.time(),'automatic_retry':False,**outcome}
  with (r/'pipeline_actual_exit.json').open('x') as f:json.dump(receipt,f)
  print(json.dumps({'stage':'pipeline_actual_exit',**receipt}),flush=True)
sys.exit(code)
"""
with (b/'control/evaluation_execution_intent.json').open('x') as f:json.dump({**p,'started_unix':time.time(),'new_training':False,'automatic_retry':False},f,indent=2)
with (b/'control/evaluation_execution_stdout.bin').open('xb') as out:
 x=recorded_run(b,'evaluation_execution',['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','5080','/usr/bin/python3 -c '+shlex.quote(code)],input=json.dumps(p).encode(),stdout=out,timeout=43320)
print(json.dumps(json.loads((b/'control/evaluation_execution_exit.json').read_text())),flush=True)
raise SystemExit(x.returncode)
