import base64,hashlib,json,shlex,subprocess,time
from pathlib import Path
b=Path('search_artifacts/titantpp_components_inter8_test_20261014_v1')
r='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090'
a=Path('paper/scripts/audit_titantpp_components_inter8_inputs.py').read_bytes()
files={'source/sample_data/.frozen_root_sentinel':b'','operation/audit_inputs_v2.py':a}
p={'root':r,'files':{k:base64.b64encode(v).decode() for k,v in files.items()},'digests':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},'approval_sha256':'1f253432d34f47872fe3973ae8c1039d959b48b18bc4285d822ac534776f1b5c'}
code='''import base64,hashlib,json,sys,time\nfrom pathlib import Path\np=json.load(sys.stdin);r=Path(p['root'])\nassert str(r)=="/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090"\nassert hashlib.sha256((r/'partial_approval.json').read_bytes()).hexdigest()==p['approval_sha256']\nc=json.loads((r/'provenance/training_contract.json').read_text())\nassert p['digests']['source/sample_data/.frozen_root_sentinel']==c['input_files']['sample_data/.frozen_root_sentinel']\nassert set(p['files'])=={'source/sample_data/.frozen_root_sentinel','operation/audit_inputs_v2.py'}\nfor n,v in p['files'].items():\n t=r/n;assert not t.exists() and t.resolve().is_relative_to(r.resolve());assert hashlib.sha256(base64.b64decode(v)).hexdigest()==p['digests'][n]\nfor n,v in p['files'].items():\n t=r/n;t.parent.mkdir(parents=True,exist_ok=True)\n with t.open('xb') as f:f.write(base64.b64decode(v))\nreceipt={'status':'supplement_deployed','files':p['digests'],'old_audit_claim_preserved':True,'source131_modified':False,'original_training_root_writes':0,'model_predictions':0,'completed_unix':time.time()}\nwith (r/'input_supplement_receipt.json').open('x') as f:json.dump(receipt,f)\nprint(json.dumps(receipt))\n'''
with (b/'control/input_supplement_intent.json').open('x') as f:json.dump({k:v for k,v in p.items() if k!='files'},f)
x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5090','/usr/bin/python3 -c '+shlex.quote(code)],input=json.dumps(p).encode(),capture_output=True,timeout=120)
for name,value in [('input_supplement_stdout.bin',x.stdout),('input_supplement_stderr.bin',x.stderr)]:
 with (b/'control'/name).open('xb') as f:f.write(value)
with (b/'control/input_supplement_exit.json').open('x') as f:json.dump({'actual_transport_exit':x.returncode,'stdout_sha256':hashlib.sha256(x.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(x.stderr).hexdigest()},f)
print(x.stdout.decode());print(x.stderr.decode());raise SystemExit(x.returncode)
