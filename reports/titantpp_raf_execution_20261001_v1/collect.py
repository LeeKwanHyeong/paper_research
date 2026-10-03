"""Read-only SSH streaming collection of the completed RAF campaign."""
import argparse, hashlib, json, subprocess, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = {'raf': ('search_artifacts/titantpp_raf_5080_20260930_v1',
    'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2',
    '46beb267a55ad97e257ff8f7cc63025d924a5c9a433d449fac1cd453fda58932')}

def canonical(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

REMOTE=r'''
import hashlib,io,json,os,subprocess,sys,tarfile,time
from pathlib import Path
p=PAYLOAD; root=Path(p['root']); os.nice(10)
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(q):return json.loads(q.read_text())
c=read(root/'execution_contract.json');assert canonical(c)==p['contract_sha256']
assert c['evaluation_scope']=='validation_only' and c['held_out_test_evaluated'] is False
assert canonical(c['source']['files'])==c['source']['files_sha256']
s=read(root/'status.json');assert s['status']=='complete'
jobs=[j for j in c['jobs'] if j.get('host','5080')=='5080'];assert len(jobs)==p['expected_jobs']
owned={}
for q in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  raw=q.read_bytes()
  if str(root).encode() in raw:owned[q.parent.name]=raw.replace(b'\0',b' ').decode(errors='replace')
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not owned,owned
files={};reference_map={}
def add(q,rel=None):
 q=Path(q);name=rel or str(q.relative_to(root))
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 assert q.is_file() and not q.is_symlink()
 assert name.startswith('source/') or q.suffix == '.py' or ('test' not in q.name.lower() and q.suffix in ('.json','.py','.log','.sh','.txt','.pt','.claim'))
 files[name]=q
for rel,h in c['source']['files'].items():
 q=root/'source'/rel;assert sha(q)==h,rel;add(q)
for rel,h in c.get('operational_files',{}).items():assert sha(root/rel)==h;add(root/rel)
for q in root.iterdir():
 if q.is_file() and (q.suffix in ('.json','.py','.log','.sh','.claim')):add(q)
for folder in ('qualification','claims','logs'):
 for q in (root/folder).glob('*'):
  if q.is_file() and q.suffix in ('.json','.log'):add(q)
for j in jobs:
 d=root/'run'/j['id'];mp=d/'terminal_manifest.json';m=read(mp)
 assert m['scientific_success'] is True and m['job']==j and m['contract_sha256']==p['contract_sha256']
 assert s['completed'][j['id']]['terminal_manifest_sha256']==sha(mp)
 add(mp)
 for rel,h in m['files'].items():
  q=d/rel;assert q.resolve().is_relative_to(d.resolve());assert sha(q)==h,(j['id'],rel);add(q)
if p['kind']=='additional':
 for ref in c['reuse']:
  if ref['host']!='5080':continue
  q=Path(ref['remote_run'])/'exposure.json';assert sha(q)==ref['file_sha256']['exposure.json']
  dest='reference_exposure/'+ref['dataset']+'__'+str(ref['seed'])+'__'+ref['arm']+'.json';add(q,dest)
  reference_map[dest]={'original_path':str(q),'sha256':sha(q)}
elif p['kind']=='normalization':
 for rel,h in c['recovery']['files'].items():
  q=root/'recovery_seed'/rel;assert sha(q)==h;add(q)
 add(root/c['recovery']['baseline_exposure'])
for q in (root/'recovery_20261001_v1').rglob('*'):
 if q.is_file() and q.suffix in ('.json','.py','.log','.sh','.claim','.txt'):add(q)
manifest={rel:{'sha256':sha(q),'bytes':q.stat().st_size,'remote_path':str(q)} for rel,q in files.items()}
assert all(v['bytes']<128*1024**2 for v in manifest.values())
assert sum(v['bytes'] for v in manifest.values())<1024**3
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
receipt={'collected_unix':time.time(),'host':'5080','root':str(root),'contract_sha256':canonical(c),'source_closure_sha256':c['source']['files_sha256'],'source_files':len(c['source']['files']),'jobs':len(jobs),'status':s,'owned_processes':owned,'other_gpu_processes_allowed':True,'gpu_compute_at_collection':gpu.stdout,'remote_writes':False,'torch_imported_on_server':False,'reference_map':reference_map,'files':manifest}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for rel,q in sorted(files.items()):
  tar.add(q,arcname=rel,recursive=False)
  assert sha(q)==manifest[rel]['sha256'],('File changed while streaming',rel)
 raw=json.dumps(receipt,ensure_ascii=False,indent=2).encode();info=tarfile.TarInfo('collection_manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('campaign',choices=SPECS);a=ap.parse_args()
    bundle,destination,expected=SPECS[a.campaign];c=json.loads((ROOT/bundle/'execution_contract.json').read_text())
    assert canonical(c)==expected
    spec=c['hosts']['5080'] if a.campaign=='additional' else c['host']
    payload={'root':spec['root'],'contract_sha256':expected,'expected_jobs':24,'kind':a.campaign}
    out=ROOT/destination;out.mkdir(parents=True,exist_ok=False)
    script=REMOTE.replace('PAYLOAD',repr(payload),1)
    (out/'remote_readonly.py').write_text(script)
    (out/'collection_request.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (out/'original.tar').open('xb') as f:
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','python3 -'],input=script.encode(),stdout=f,stderr=subprocess.PIPE,timeout=300)
    (out/'ssh_stderr.txt').write_bytes(proc.stderr)
    assert proc.returncode==0,'Collection failed; preserve partial archive/stderr, do not overwrite.'
    original=out/'original';original.mkdir()
    with tarfile.open(out/'original.tar') as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        tar.extractall(original,filter='data')
    receipt=json.loads((original/'collection_manifest.json').read_text())
    assert {m.name for m in members}==set(receipt['files'])|{'collection_manifest.json'}
    for rel,v in receipt['files'].items():assert sha(original/rel)==v['sha256'] and (original/rel).stat().st_size==v['bytes'],rel
    result={'status':'retrieved_all_file_sha_verified','collected_unix':receipt['collected_unix'],'verified_unix':time.time(),'files':len(receipt['files']),'bytes':sum(v['bytes'] for v in receipt['files'].values()),'source_files':receipt['source_files'],'jobs':receipt['jobs'],'archive_sha256':sha(out/'original.tar'),'binary_cpu_audit':'pending','remote_writes':False,'owned_processes_absent':True,'other_gpu_work_untouched':True,'script_sha256':sha(__file__)}
    (out/'retrieval_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(out),**result}))
if __name__=='__main__':main()
