"""Exclusive deployment: verified transport plus copies of staged immutable inputs."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_intermittent_time_cross_20261012_v1_5080')
PARENT = Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_train_fit_20261012_v1_5080')

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')

def safe(root, relative):
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts,'Escaping manifest path')
    target = (root/relative).resolve()
    require(target.is_relative_to(root),'Escaping manifest target')
    return target

def main():
    archive, archive_sha, seal_sha, contract_sha = Path(sys.argv[1]),*sys.argv[2:]
    require(not ROOT.exists(),'New diagnostic root already exists; no deployment retry')
    require(sha(archive)==archive_sha,'Transport archive SHA mismatch')
    require(not Path('/var/run/reboot-required').exists(),'Host reboot is pending')
    gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    require(len(gpu.splitlines())==1 and gpu.split(',')[0].strip()=='GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6','Pinned GPU changed')
    compute = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    require(not compute,'GPU busy; deployment not started')
    packages = subprocess.run(['pgrep','-af',r'(^|/)(apt|apt-get|dpkg|unattended-upgrade|do-release-upgrade)( |$)'],capture_output=True,text=True)
    require(packages.returncode==1 and not packages.stdout.strip(),'Package manager active or query failed')
    require(sha(PARENT/'contract.json')=='db4096cae1c1889eb1f7f7b248acc21a66f36dc1e8e023acac76a68507b6f342','Parent contract rawSHA changed')
    previous = read(PARENT/'control/pipeline_complete.json')
    previous_exit = read(PARENT/'control/pipeline_exit.json')
    require(previous['status']=='complete' and previous['train_count']==27 and previous['actual_all_worker_exit0'] is True
        and previous_exit['exit_code']==0 and previous_exit['actual_child_exit_observed'] is True,'Parent evaluation completion differs')
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers()
        require(len({m.name for m in members})==len(members),'Duplicate transport members')
        for m in members:
            require(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts,'Unsafe transport member')
        ROOT.mkdir(parents=True,exist_ok=False)
        t.extractall(ROOT,filter='data')
    try:
        require(sha(ROOT/'operation_integrity.json')==seal_sha,'Operation seal rawSHA changed')
        seal=read(ROOT/'operation_integrity.json')
        inputs=read(ROOT/'input_integrity.json')
        c=read(ROOT/'contract.json')
        require(seal['contract_sha256']==inputs['contract_sha256']==c['contract_sha256']==contract_sha,'Deployment contract differs')
        import importlib.util
        spec=importlib.util.spec_from_file_location('prerequisite',ROOT/'control/prerequisite.py')
        gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate);gate.verify(ROOT,c)
        reuse=read(ROOT/'deployment/reuse.json')
        require(reuse['parent_root']==str(PARENT) and reuse['contract_sha256']==contract_sha,'Reuse source root differs')
        for rel, expected in reuse['files'].items():
            src,dest=safe(PARENT,rel),safe(ROOT,rel)
            require(not dest.exists() and src.is_file() and not src.is_symlink() and sha(src)==expected,'Staged immutable input changed: '+rel)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
            require(sha(dest)==expected and sha(src)==expected,'Immutable copy verification failed: '+rel)
        for rel,expected in {**inputs['files'],**seal['files']}.items():
            require(sha(safe(ROOT,rel))==expected,'Deployed sealed file differs: '+rel)
        receipt={'status':'verified','contract_sha256':contract_sha,'operation_integrity_sha256':seal_sha,
            'archive_sha256':archive_sha,'remote_root':str(ROOT),'parent_root':str(PARENT),
            'transport_member_count':len(members),'copied_immutable_file_count':len(reuse['files']),
            'gpu':gpu,'gpu_idle_observed':True,'python_executable':sys.executable,'reboot_required':False,
            'training_started':False,'inference_started':False,'originals_modified':False,'automatic_retry':False,'observed_unix':time.time()}
        write(ROOT/'control/deployment_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)
    except BaseException as error:
        write(ROOT/'control/deployment_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),
            'observed_unix':time.time(),'automatic_retry':False})
        raise

if __name__=='__main__':main()
