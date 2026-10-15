"""One-shot install of the authorized diagnostic into two fresh native roots."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from paper.scripts.prepare_titantpp_cnn_memory_sensitivity import PROJECT, NAME, sha

INSTALL = r'''
import hashlib,io,json,os,pathlib,shlex,subprocess,sys,tarfile,time
root=pathlib.Path(sys.argv[1]);expected=sys.argv[2];archive_sha=sys.argv[3];host=sys.argv[4]
assert host in ('5080','5090') and root.name=='titantpp_cnn_memory_sensitivity_20261015_v1_'+host
assert root.parent==pathlib.Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts')
assert not root.exists(), 'Existing native root; no retry or overwrite'
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==archive_sha
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 cbytes=tar.extractfile('contract.json').read();assert hashlib.sha256(cbytes).hexdigest()==expected
 c=json.loads(cbytes);assert c['remote_roots'][host]==str(root) and time.time()<c['deadline_unix']
 assert c['allowed_splits']==['train','validation'] and c['new_training'] is False
 files={**c['input_files'],**c['operation_files']}
 assert not set(c['input_files'])&set(c['operation_files'])
 assert {m.name for m in tar.getmembers()}==set(files)|{'contract.json'}
 for m in tar.getmembers():
  p=pathlib.Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  if m.name!='contract.json':assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==files[m.name]
 spec=c['runtimes'][host];env={**os.environ,**c['execution'][host]['environment']}
 assert pathlib.Path(spec['python']).is_file() and pathlib.Path(c['execution'][host]['tmux_binary']).is_file()
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],timeout=20).decode().strip().splitlines()
 assert gpu==[spec['gpu_uuid']], 'Wrong GPU identity'
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],timeout=20).decode().strip()
 assert not busy, 'Existing compute process; no changes made'
 probe="import json,platform,torch,numpy,polars;print(json.dumps(dict(python_version=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=numpy.__version__,polars=polars.__version__)))"
 actual=json.loads(subprocess.check_output([spec['python'],'-c',probe],env=env,timeout=45))
 assert all(actual[k]==spec[k] for k in actual), 'Native Runtime differs'
 root.mkdir()
 for m in tar.getmembers():
  path=root/m.name;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(tar.extractfile(m).read())
 for name,digest in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 supervisor="""import json,os,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]);host=sys.argv[2];digest=sys.argv[3]
c=json.loads((root/'contract.json').read_text());d=root/'launch';d.mkdir(exist_ok=True)
argv=[sys.executable,str(root/'operation/dispatch.py'),'--root',str(root),'--contract-sha',digest,'--host',host]
with (d/'supervisor_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),argv=argv,started_unix=time.time()),f)
with (d/'dispatcher.stdout').open('xb') as out,(d/'dispatcher.stderr').open('xb') as err:
 p=subprocess.Popen(argv,stdout=out,stderr=err,stdin=subprocess.DEVNULL);code=p.wait()
with (d/'dispatcher_exit.json').open('x') as f:json.dump(dict(actual_dispatcher_os_exit=code,dispatcher_pid=p.pid,finished_unix=time.time(),supervisor_own_os_exit_observed=False),f)
sys.exit(code)
"""
 argv=[spec['python'],'-c',supervisor,str(root),host,expected]
 shell='cd '+shlex.quote(str(root))+' && exec env '+' '.join(shlex.quote(k+'='+v) for k,v in c['execution'][host]['environment'].items())+' '+' '.join(shlex.quote(x) for x in argv)
 subprocess.run([c['execution'][host]['tmux_binary'],'new-session','-d','-s',c['execution'][host]['tmux'],shell],check=True,capture_output=True,timeout=20)
 print(json.dumps(dict(status='started',host=host,root=str(root),contract_sha256=expected,archive_sha256=archive_sha,runtime=actual,started_unix=time.time(),max_parallel=3)))
'''


def launch(bundle, expected):
    assert sha(bundle/'contract.json') == expected
    c = json.loads((bundle/'contract.json').read_text())
    report = PROJECT/'reports'/NAME/'deployment'
    report.mkdir(parents=True, exist_ok=False)
    archive = report/'bundle.tar.gz'
    files = {**c['input_files'], **c['operation_files']}
    for name, digest in files.items():
        assert sha(bundle/name) == digest, name
    with tarfile.open(archive, 'x:gz') as tar:
        for name in ['contract.json', *sorted(files)]:
            tar.add(bundle/name, arcname=name, recursive=False)
    digest = sha(archive)
    (report/'install.py').write_text(INSTALL)
    def one(host):
        folder = report/host; folder.mkdir()
        (folder/'intent.json').write_text(json.dumps(dict(host=host,contract_sha256=expected,
             archive_sha256=digest,install_sha256=sha(report/'install.py'),automatic_retry=False), indent=2))
        cmd = ['/usr/bin/python3','-c',INSTALL,c['remote_roots'][host],expected,digest,host]
        with archive.open('rb') as stream:
            r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
                ' '.join(shlex.quote(x) for x in cmd)], stdin=stream,capture_output=True,timeout=180)
        for name, value in [('stdout',r.stdout),('stderr',r.stderr)]:
            with (folder/(name+'.bin')).open('xb') as f:f.write(value)
        record = dict(host=host,actual_transport_exit=r.returncode,finished_unix=time.time(),
            stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest())
        (folder/'exit.json').write_text(json.dumps(record,indent=2))
        return dict(**record,stdout=r.stdout.decode(),stderr=r.stderr.decode())
    with ThreadPoolExecutor(2) as pool:
        for result in pool.map(one, ['5080','5090']):
            print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract-sha',required=True)
    a=p.parse_args();launch(PROJECT/'search_artifacts'/NAME,a.contract_sha)
