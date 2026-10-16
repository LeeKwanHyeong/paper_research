import base64,hashlib,json,shlex,subprocess
from pathlib import Path
b=Path('search_artifacts/titantpp_components_inter8_test_20261014_v1');a=json.loads((b/'native_input_receipt.json').read_text())
r='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090'
refs={v['metadata_path']:v['metadata_sha256'] for v in a['metadata_refs'].values()}
code="""import base64,hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['root']);assert root.name=='titantpp_components_inter8_test_20261014_v1_5090'
out={}
assert len(p['refs'])==8
for name,digest in p['refs'].items():
 path=root/name;assert Path(name).parts[0]=='metadata' and '..' not in Path(name).parts
 raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest;out[name]=base64.b64encode(raw).decode()
print(json.dumps(out))
"""
x=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5090','/usr/bin/python3 -c '+shlex.quote(code)],input=json.dumps({'root':r,'refs':refs}).encode(),capture_output=True,timeout=120)
with (b/'control/metadata_retrieval_exit.json').open('x') as f:json.dump({'actual_transport_exit':x.returncode,'stdout_sha256':hashlib.sha256(x.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(x.stderr).hexdigest()},f)
assert x.returncode==0,x.stderr.decode()
p=json.loads(x.stdout);assert set(p)==set(refs)
for name,value in p.items():
 raw=base64.b64decode(value);assert hashlib.sha256(raw).hexdigest()==refs[name]
 path=b/name;path.parent.mkdir(exist_ok=True)
 with path.open('xb') as f:f.write(raw)
print('metadata8 recovered; all rawSHA matched')
