"""Read-only SHA-preserving retrieval of aggregate evaluation evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def main(bundle):
    bundle = Path(bundle).resolve()
    c = json.loads((bundle/'execution_contract.json').read_text())
    remote = c['resources']['root']
    code = '''
import hashlib,json,pathlib,sys,tarfile
root=pathlib.Path(REMOTE_ROOT)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'execution_contract.json')==CONTRACT_SHA
m=json.loads((root/'deployment_manifest.json').read_text())
for name,digest in m['files'].items():assert sha(root/name)==digest,name
files=[]
for name in ['runs','logs','failures','launches','claims']:
 files += [p for p in (root/name).rglob('*') if p.is_file()]
for name in ['campaign.log','pipeline_status.json','validation_gate.json','inference_completion.json','launch_receipt.json','campaign.lock','worker.lock']:
 if (root/name).is_file():files.append(root/name)
assert all(p.suffix not in ['.parquet','.pt','.csv'] and 'predictions' not in p.parts for p in files)
before={str(p.relative_to(root)):sha(p) for p in files}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for p in files:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
assert all(sha(root/name)==digest for name,digest in before.items()),'Evidence changed during retrieval'
print(json.dumps({'files_SHA':before,'input_manifest_verified':True}),file=sys.stderr)
'''
    code=code.replace('REMOTE_ROOT',repr(remote)).replace('CONTRACT_SHA',repr(sha(bundle/'execution_contract.json')))
    target=bundle/'retrieved'
    target.mkdir(exist_ok=False)
    archive=target/'evidence.tar.gz'
    with archive.open('xb') as stream:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','python3 -c '+shlex.quote(code)],stdout=stream,stderr=subprocess.PIPE,timeout=60)
    (target/'transport_stderr.log').write_bytes(result.stderr)
    assert result.returncode==0,result.stderr.decode(errors='replace')[-3000:]
    assert archive.stat().st_size<100*1024**2
    receipt=json.loads(result.stderr.decode().strip().splitlines()[-1])
    with tarfile.open(archive) as stream:
        entries=stream.getmembers()
        assert len({e.name for e in entries})==len(entries) and sum(e.size for e in entries)<100*1024**2
        assert all(e.isfile() and not Path(e.name).is_absolute() and '..' not in Path(e.name).parts for e in entries)
        stream.extractall(target,filter='data')
    assert all(sha(target/p)==digest for p,digest in receipt['files_SHA'].items())
    receipt.update(archive_sha256=sha(archive),remote_root=remote,contract_sha256=sha(bundle/'execution_contract.json'),
        readonly_remote_calls=1,recorded_unix=time.time())
    (target/'retrieval_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'SHA_verified','files':len(receipt['files_SHA']),'archive_sha256':receipt['archive_sha256']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    main(p.parse_args().bundle)
