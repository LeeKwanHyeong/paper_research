"""Read-only collection of 13 pinned terminal jobs; running jobs remain untouched."""
import argparse, hashlib, json, subprocess, tarfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
BUNDLE = ROOT/'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2'
DEST = BUNDLE/'retrieved/completed13_20261002_v1'
EXPECTED = '08f7e9ed402ba74fde099321f039b78de3a11b0bd93cba27492b1a84454f6a6c'
CLOSURE = '989153f9eea2dac6a790835ab1b7c502fc9c0461829800915a6c602166b6ea92'
SNAPSHOT = BUNDLE/'hourly_monitor/20261002T015157805227Z'
def canonical(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def scope():
    c=read(BUNDLE/'execution_contract.json'); assert canonical(c)==EXPECTED
    result={'scope':'completed13_validation_only','contract_sha256':EXPECTED,'source_closure_sha256':CLOSURE,'snapshot':str(SNAPSHOT.relative_to(ROOT)),'hosts':{}}
    for host in ('5080','5090'):
        s=read(SNAPSHOT/host/'snapshot.json'); jobs=[]
        for r in s['runs']:
            m=r.get('terminal_manifest')
            if not m or m.get('scientific_success') is not True:continue
            assert r['small_record_SHA_passed'] is True and m['status']=='complete'
            j=r['job'];assert j in c['jobs']
            rel='run/'+j['id']+'/terminal_manifest.json'
            jobs.append({'job':j,'terminal_manifest_sha256':s['file_sha256'][rel]})
        assert len(jobs)==(11 if host=='5080' else 2)
        result['hosts'][host]={'observed_unix':s['observed_unix'],'snapshot_sha256':sha(SNAPSHOT/host/'snapshot.json'),'jobs':jobs}
    return result
REMOTE=r'''
import hashlib,io,json,os,sys,tarfile,time
from pathlib import Path
p=PAYLOAD;root=Path(p['root']);os.nice(10)
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(q):return json.loads(q.read_text())
c=read(root/'execution_contract.json');assert canonical(c)==p['contract_sha256']
assert c['evaluation_scope']=='validation_only' and c['held_out_test_evaluated'] is False
assert canonical(c['source']['files'])==c['source']['files_sha256']==p['source_closure_sha256']
s=read(root/'status.json')
files={};reference_map={}
def add(q,rel=None):
 q=Path(q);name=rel or str(q.relative_to(root))
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 assert q.is_file() and not q.is_symlink()
 assert name.startswith('source/') or ('test' not in q.name.lower() and q.suffix in ('.json','.py','.log','.sh','.txt','.pt'))
 assert name not in files or files[name]==q
 files[name]=q
for rel,h in c['source']['files'].items():
 q=root/'source'/rel;assert sha(q)==h,rel;add(q)
for rel in ('execution_contract.json','approval.json','start_permit.json','training_permit.json','qualification/receipt.json'):
 add(root/rel)
for pinned in p['jobs']:
 j=pinned['job'];assert j in c['jobs'] and j['host']==p['host']
 d=root/'run'/j['id'];mp=d/'terminal_manifest.json';m=read(mp)
 assert sha(mp)==pinned['terminal_manifest_sha256'],j['id']
 assert m['scientific_success'] is True and m['status']=='complete' and m['job']==j and m['contract_sha256']==p['contract_sha256']
 assert s['completed'][j['id']]['terminal_manifest_sha256']==sha(mp)
 add(mp);add(root/'claims'/(j['id']+'.json'))
 for rel,h in m['files'].items():
  q=d/rel;assert q.resolve().is_relative_to(d.resolve());assert sha(q)==h,(j['id'],rel);add(q)
needed={(x['job']['dataset'],x['job']['seed']) for x in p['jobs']}
for ref in c['reuse']:
 if ref['host']!=p['host'] or (ref['dataset'],ref['seed']) not in needed:continue
 q=Path(ref['remote_run'])/'exposure.json';assert sha(q)==ref['file_sha256']['exposure.json']
 dest='reference_exposure/'+ref['dataset']+'__'+str(ref['seed'])+'__'+ref['arm']+'.json';add(q,dest)
 reference_map[dest]={'original_path':str(q),'sha256':sha(q)}
assert len(reference_map)==len(needed)
manifest={rel:{'sha256':sha(q),'bytes':q.stat().st_size,'remote_path':str(q)} for rel,q in files.items()}
assert all(v['bytes']<128*1024**2 for v in manifest.values())
assert sum(v['bytes'] for v in manifest.values())<1024**3
receipt={'collected_unix':time.time(),'host':p['host'],'root':str(root),'contract_sha256':canonical(c),'source_closure_sha256':c['source']['files_sha256'],'source_files':len(c['source']['files']),'jobs':len(p['jobs']),'pinned_jobs':p['jobs'],'host_status_at_collection':s,'scope':'terminal_subset_on_active_server','running_jobs_excluded':True,'remote_writes':False,'torch_imported_on_server':False,'reference_map':reference_map,'files':manifest}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for rel,q in sorted(files.items()):
  tar.add(q,arcname=rel,recursive=False)
  assert sha(q)==manifest[rel]['sha256'],('File changed while streaming',rel)
 raw=json.dumps(receipt,ensure_ascii=False,indent=2).encode();info=tarfile.TarInfo('collection_manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
'''
def main():
    ap=argparse.ArgumentParser();ap.add_argument('host',choices=('5080','5090'));a=ap.parse_args()
    sc=scope();sp=REPORT/'scope.json';assert sp.exists() and read(sp)==sc
    c=read(BUNDLE/'execution_contract.json')
    payload={'root':c['hosts'][a.host]['root'],'contract_sha256':EXPECTED,'source_closure_sha256':CLOSURE,'host':a.host,'jobs':sc['hosts'][a.host]['jobs']}
    out=DEST/a.host;out.mkdir(parents=True,exist_ok=False)
    script=REMOTE.replace('PAYLOAD',repr(payload),1)
    (out/'remote_readonly.py').write_text(script)
    (out/'collection_request.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (out/'original.tar').open('xb') as f:
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',a.host,'python3 -'],input=script.encode(),stdout=f,stderr=subprocess.PIPE,timeout=600)
    (out/'ssh_stderr.txt').write_bytes(proc.stderr)
    assert proc.returncode==0,'Collection failed; preserve partial archive/stderr, do not overwrite.'
    original=out/'original';original.mkdir()
    with tarfile.open(out/'original.tar') as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        tar.extractall(original,filter='data')
    receipt=read(original/'collection_manifest.json')
    assert {m.name for m in members}==set(receipt['files'])|{'collection_manifest.json'}
    for rel,v in receipt['files'].items():assert sha(original/rel)==v['sha256'] and (original/rel).stat().st_size==v['bytes'],rel
    result={'status':'retrieved_all_file_sha_verified','host':a.host,'collected_unix':receipt['collected_unix'],'verified_unix':time.time(),'files':len(receipt['files']),'bytes':sum(v['bytes'] for v in receipt['files'].values()),'source_files':receipt['source_files'],'jobs':receipt['jobs'],'archive_sha256':sha(out/'original.tar'),'binary_cpu_audit':'pending','remote_writes':False,'running_jobs_excluded':True,'script_sha256':sha(__file__),'scope_sha256':sha(sp)}
    (out/'retrieval_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
