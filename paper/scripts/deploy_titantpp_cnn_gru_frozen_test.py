"""Deploy and launch the approved sealed Test9 batch once on its dedicated root."""
from __future__ import annotations
import hashlib
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'search_artifacts/titantpp_cnn_gru_test_20261005_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main(bundle=BUNDLE):
    bundle = Path(bundle).resolve()
    global BUNDLE
    BUNDLE = bundle
    manifest = json.loads((BUNDLE / 'deployment_manifest.json').read_text())
    contract = json.loads((BUNDLE / 'execution_contract.json').read_text())
    for relative, digest in manifest['files'].items():
        assert sha(BUNDLE / relative) == digest, relative
    intent = BUNDLE / 'launch_intent.json'
    with intent.open('x') as stream:
        json.dump({'host':'5080', 'remote_root':manifest['remote_root'], 'manifest_sha256':sha(BUNDLE/'deployment_manifest.json'),
                   'contract_sha256':sha(BUNDLE/'execution_contract.json'), 'automatic_retry':False, 'new_training':False,
                   'test_inference_authorized':True, 'created_unix':time.time()}, stream, indent=2)
    archive = BUNDLE / 'deployment.tar.gz'
    with tarfile.open(archive, 'x:gz') as stream:
        for relative in [*manifest['files'], 'deployment_manifest.json']:
            stream.add(BUNDLE / relative, arcname=relative, recursive=False)
    remote_code = '''
import hashlib,json,pathlib,shlex,shutil,subprocess,sys,tarfile,time
root=pathlib.Path(REMOTE_ROOT)
root.mkdir(mode=0o700,parents=False,exist_ok=False)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as archive:
 for member in archive:
  relative=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
  target=root/relative
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as output:shutil.copyfileobj(archive.extractfile(member),output)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'deployment_manifest.json')==MANIFEST_SHA
m=json.loads((root/'deployment_manifest.json').read_text())
for relative,digest in m['files'].items():assert sha(root/relative)==digest,relative
c=json.loads((root/'execution_contract.json').read_text())
assert c['resources']['root']==str(root)
assert shutil.which('timeout') and shutil.which('tmux')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
assert gpu.split(',')[0].strip()==c['resources']['gpu_uuid'],gpu
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert not compute,'GPU already has a compute process: '+compute
probe='import torch,numpy,polars,json;print(json.dumps(dict(torch=str(torch.__version__),numpy=numpy.__version__,polars=polars.__version__,cuda=torch.version.cuda)))'
runtime=json.loads(subprocess.check_output([c['resources']['python'],'-c',probe],text=True))
assert all(runtime[k]==v for k,v in c['runtime_expected'].items()),runtime
subprocess.run([c['resources']['python'],str(root/'operation/evaluate_titantpp_cnn_gru_frozen_test.py'),'--campaign',str(root),'--mode','inspect'],check=True)
command=['timeout','--signal=TERM','--kill-after=15s',str(c['resources']['campaign_timeout_seconds']),c['resources']['python'],str(root/'operation/evaluate_titantpp_cnn_gru_frozen_test.py'),'--campaign',str(root),'--mode','pipeline','--device','cuda']
shell='exec '+shlex.join(command)+' > '+shlex.quote(str(root/'campaign.log'))+' 2>&1'
session='titantpp_cnn_gru_test_20261005_v1'
subprocess.run(['tmux','new-session','-d','-s',session,'sh','-c',shell],check=True)
receipt=dict(status='launched',host='5080',root=str(root),session=session,command=command,runtime=runtime,gpu=gpu,prelaunch_compute_empty=True,manifest_sha256=MANIFEST_SHA,contract_sha256=sha(root/'execution_contract.json'),started_unix=time.time(),new_training=False,shared_runtime_changed=False,automatic_retry=False)
(root/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    remote_code = remote_code.replace('REMOTE_ROOT', repr(manifest['remote_root'])).replace('MANIFEST_SHA', repr(sha(BUNDLE/'deployment_manifest.json')))
    with archive.open('rb') as payload:
        result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080',
            'python3 -c '+shlex.quote(remote_code)], stdin=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    (BUNDLE/'deployment_output.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-3000:]
    receipt = json.loads(result.stdout.decode().strip().splitlines()[-1])
    with (BUNDLE/'launch_receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    main(parser.parse_args().bundle)
