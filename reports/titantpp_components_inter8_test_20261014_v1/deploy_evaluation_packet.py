import base64,hashlib,json,shlex,subprocess,time
from pathlib import Path
b=Path('search_artifacts/titantpp_components_inter8_test_20261014_v1');r='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090'
c=json.loads((b/'contract.json').read_text())
assert c['expected_jobs']==8 and c['evaluation_scope']=='test_partial_followup' and c['new_training'] is False
assert c['original36_global_gate_modified'] is False
names=['contract.json','operation/evaluate.py','operation/pipeline.py','permits/validation.json','permits/test.json','native_input_receipt.json','control/native_audit2_exit.json']
files={n:(b/n).read_bytes() for n in names}
p={'root':r,'files':{n:base64.b64encode(v).decode() for n,v in files.items()},'digests':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},'approval_sha256':c['partial_approval_sha256'],'contract_sha256':c['contract_sha256']}
code="""import base64,hashlib,json,sys,time
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['root']);assert str(r)=="/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090"
assert hashlib.sha256((r/'partial_approval.json').read_bytes()).hexdigest()==p['approval_sha256']
a=json.loads((r/'audit/attempt2/input_receipt.json').read_text());assert a['status']=='complete' and a['model_predictions']==0
assert set(p['files'])=={'contract.json','operation/evaluate.py','operation/pipeline.py','permits/validation.json','permits/test.json','native_input_receipt.json','control/native_audit2_exit.json'}
for n,v in p['files'].items():
 t=r/n;assert not t.exists() and t.resolve().is_relative_to(r.resolve());assert hashlib.sha256(base64.b64decode(v)).hexdigest()==p['digests'][n]
c=json.loads(base64.b64decode(p['files']['contract.json']));identity=dict(c);identity.pop('contract_sha256');canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
assert canonical(identity)==p['contract_sha256']==c['contract_sha256'];assert c['expected_jobs']==8 and c['evaluation_scope']=='test_partial_followup' and not c['new_training'] and not c['original36_global_gate_modified']
for n,v in c['evaluator_integrity']['files'].items():
 assert (p['digests'][n] if n in p['digests'] else hashlib.sha256((r/n).read_bytes()).hexdigest())==v
for j in c['jobs']:
 assert hashlib.sha256((r/j['checkpoint_metadata_path']).read_bytes()).hexdigest()==j['checkpoint_metadata_sha256']
for n,v in p['files'].items():
 t=r/n;t.parent.mkdir(parents=True,exist_ok=True)
 with t.open('xb') as f:f.write(base64.b64decode(v))
receipt={'status':'evaluation_packet_deployed','files':p['digests'],'contract_sha256':p['contract_sha256'],'new_training':False,'original_training_root_writes':0,'completed_unix':time.time()}
with (r/'evaluation_deployment_receipt.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
"""
with (b/'control/evaluation_deploy_intent.json').open('x') as f:json.dump({k:v for k,v in p.items() if k!='files'},f)
x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5090','/usr/bin/python3 -c '+shlex.quote(code)],input=json.dumps(p).encode(),capture_output=True,timeout=120)
for n,v in [('evaluation_deploy_stdout.bin',x.stdout),('evaluation_deploy_stderr.bin',x.stderr)]:
 with (b/'control'/n).open('xb') as f:f.write(v)
with (b/'control/evaluation_deploy_exit.json').open('x') as f:json.dump({'actual_transport_exit':x.returncode,'stdout_sha256':hashlib.sha256(x.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(x.stderr).hexdigest()},f)
print(x.stdout.decode());print(x.stderr.decode());raise SystemExit(x.returncode)
