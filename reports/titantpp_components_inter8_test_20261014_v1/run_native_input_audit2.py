import hashlib,json,shlex,subprocess,time
from pathlib import Path
b=Path('search_artifacts/titantpp_components_inter8_test_20261014_v1')
r='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090'
old=json.loads((b/'control/native_audit_intent.json').read_text())
a=[]
for v in old['argv']:
 v=v.replace('/operation/audit_inputs.py','/operation/audit_inputs_v2.py').replace('/audit/runtime_cache','/audit/attempt2/runtime_cache')
 a.append(v)
a += ['--audit-attempt','attempt2']
with (b/'control/native_audit2_intent.json').open('x') as f:json.dump({'argv':a,'started_unix':time.time(),'first_failed_before_checkpoint_or_data':True,'model_predictions':0,'automatic_scientific_retry':False},f)
x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5090',shlex.join(a)],capture_output=True,timeout=900)
for n,v in [('native_audit2_stdout.bin',x.stdout),('native_audit2_stderr.bin',x.stderr)]:
 with (b/'control'/n).open('xb') as f:f.write(v)
with (b/'control/native_audit2_exit.json').open('x') as f:json.dump({'actual_transport_exit':x.returncode,'actual_transport_exit_observed':True,'stdout_sha256':hashlib.sha256(x.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(x.stderr).hexdigest(),'completed_unix':time.time(),'model_predictions':0},f)
if x.returncode==0:
 value=json.loads(x.stdout)
 with (b/'native_input_receipt.json').open('xb') as f:f.write(x.stdout)
 print(json.dumps({k:value[k] for k in ('status','model_predictions','populations','runtime')}))
else:print(x.stderr.decode())
raise SystemExit(x.returncode)
