"""Read-only retrieval of the 36 legacy selected binaries and their frozen sources."""
import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'search_artifacts/titantpp_independent_evaluation_preparation_20261001_v1'
BUNDLES=['local_detail_benchmark_execution_20260922_v2','local_detail_benchmark_intermittent_recovery_20260923_v4','local_detail_benchmark_execution_20260923_v3','local_detail_replication_5080_20260924_v1','local_detail_replication_5090_20260924_v1','local_detail_instacart_remaining_5090_20260926_v1']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def canonical(j):return hashlib.sha256(json.dumps(j,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

REMOTE=r'''
import hashlib,io,json,os,sys,tarfile,time
from pathlib import Path
p=PAYLOAD
os.nice(10)
def sha(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def canonical(j):return hashlib.sha256(json.dumps(j,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
files={};cp_rows=[]
def add(q,name):
 assert q.is_file() and not q.is_symlink(),str(q)
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 assert q.stat().st_size<128*1024**2
 files[name]=q
for b in p['bundles']:
 root=Path(b['remote_root'])
 contract=root/'frozen_execution/execution_contract.json'
 c=json.loads(contract.read_text());assert canonical(c)==b['canonical_sha256']
 assert c['source']['files']==b['source_files']
 add(contract,b['name']+'/frozen_execution/execution_contract.json')
 for rel,expected in b['source_files'].items():
  q=root/'source'/rel;assert sha(q)==expected,(str(q),'source SHA')
  add(q,b['name']+'/source/'+rel)
 for r in b['rows']:
  q=Path(r['recorded_remote_checkpoint_path']);assert q.is_relative_to(root)
  run=q.parent
  summary=json.loads((run/'summary.json').read_text())
  replay=json.loads((run/'endpoint_replays.json').read_text())
  assert summary['evaluation_scope']=='validation_only' and summary['held_out_test_evaluated'] is False
  assert summary['checkpoint_state_sha256']==r['state_tensor_sha256']
  assert replay['selected']['state_sha256']==r['state_tensor_sha256']
  assert replay['best_epoch']==r['selected_epoch']
  for filename in ['best_val_qty_rmse_model.pt','summary.json','endpoint_replays.json','history.json','exposure.json']:
   f=run/filename
   add(f,b['name']+'/'+str(f.relative_to(root)))
  cp_rows.append({'dataset':r['dataset'],'model':r['model'],'seed':r['seed'],'remote_path':str(q),'relative_path':b['name']+'/'+str(q.relative_to(root)),'sha256':sha(q),'bytes':q.stat().st_size,'selected_epoch':r['selected_epoch'],'tensor_sha256':r['state_tensor_sha256']})
manifest={rel:{'sha256':sha(q),'bytes':q.stat().st_size,'remote_path':str(q)} for rel,q in files.items()}
assert sum(v['bytes'] for v in manifest.values())<1024**3
receipt={'host':p['host'],'observed_unix':time.time(),'files':manifest,'checkpoints':cp_rows,'remote_writes':False,'gpu_calls':False,'data_rows_or_test_metrics_read':False}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for rel,q in sorted(files.items()):
  tar.add(q,arcname=rel,recursive=False)
  assert sha(q)==manifest[rel]['sha256'],('changed during copy',str(q))
 raw=json.dumps(receipt,indent=2).encode();info=tarfile.TarInfo('collection_manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
'''

def main():
 ap=argparse.ArgumentParser();ap.add_argument('host',choices=['5080','5090']);ap.add_argument('--attempt',default='attempt1');a=ap.parse_args()
 allrows=json.loads((ROOT/'reports/titantpp_independent_evaluation_protocol_20261001_v1/checkpoint_manifest.json').read_text())['rows']
 rows=[r for r in allrows if not r['local_binary_present'] and (r['dataset']=='insta_market_basket')==(a.host=='5090')]
 payload={'host':a.host,'bundles':[]}
 for b in BUNDLES:
  archive=ROOT/'search_artifacts'/b/'source_bundle.tar.gz'
  with tarfile.open(archive) as t:
   cn=next(n for n in t.getnames() if n.endswith('/execution_contract.json'))
   c=json.load(t.extractfile(cn))
  selected=[r for r in rows if r['contract_canonical_sha256']==canonical(c)]
  if not selected:continue
  remote=selected[0]['recorded_remote_checkpoint_path'].split('/run/',1)[0]
  assert all(r['recorded_remote_checkpoint_path'].startswith(remote+'/run/') for r in selected)
  payload['bundles'].append({'name':b,'archive_sha256':sha(archive),'canonical_sha256':canonical(c),'source_files':c['source']['files'],'remote_root':remote,'rows':selected})
 assert sum(len(b['rows']) for b in payload['bundles'])==len(rows)==(12 if a.host=='5090' else 24)
 out=OUT/a.host/a.attempt;out.mkdir(parents=True,exist_ok=False)
 script=REMOTE.replace('PAYLOAD',repr(payload),1)
 (out/'remote_readonly.py').write_text(script)
 (out/'request.json').write_text(json.dumps(payload,indent=2)+'\n')
 with (out/'original.tar').open('xb') as f:
  proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',a.host,'python3 -'],input=script.encode(),stdout=f,stderr=subprocess.PIPE,timeout=300)
 (out/'stderr.txt').write_bytes(proc.stderr)
 if proc.returncode:
  print(json.dumps({'host':a.host,'status':'retrieval_failed','stderr':proc.stderr.decode(errors='replace')[-2000:]}));raise SystemExit(proc.returncode)
 original=out/'original';original.mkdir()
 with tarfile.open(out/'original.tar') as t:
  members=t.getmembers();assert len(members)==len({m.name for m in members})
  assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
  t.extractall(original,filter='data')
 receipt=json.loads((original/'collection_manifest.json').read_text())
 assert {m.name for m in members}==set(receipt['files'])|{'collection_manifest.json'}
 for rel,v in receipt['files'].items():assert sha(original/rel)==v['sha256']
 result={'status':'all_retrieved_bytes_verified','host':a.host,'checkpoints':len(receipt['checkpoints']),'files':len(receipt['files']),'bytes':sum(v['bytes'] for v in receipt['files'].values()),'archive_sha256':sha(out/'original.tar'),'verified_unix':time.time(),'cpu_audit':'pending','remote_writes':False}
 (out/'retrieval_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
