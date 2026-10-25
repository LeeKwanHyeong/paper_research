"""Launch the fixed three fresh fits, only after observed native qualification."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent / "execution_v2"
C = json.loads((OUT / "EXECUTION_CONTRACT.json").read_text())
B = json.loads((OUT / "BUNDLE.json").read_text())

SUPERVISOR = r'''
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parent
contract=root/'EXECUTION_CONTRACT.json';host=sys.argv[1];digest=sys.argv[2]
assert hashlib.sha256(contract.read_bytes()).hexdigest()==digest
c=json.loads(contract.read_text());spec=c['hosts'][host]
assert str(root)==spec['root']
env={**os.environ,'PYTHONPATH':str(root/'source'),'PYTHONDONTWRITEBYTECODE':'1',
     'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONHASHSEED':'42',
     'CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8',
     'NVIDIA_TF32_OVERRIDE':'0','MPLCONFIGDIR':str(root/'cache/matplotlib'),'XDG_CACHE_HOME':str(root/'cache')}
if host=='5080': env['LD_LIBRARY_PATH']='/home/leekwanhyeong/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib'
command=[spec['python'],'-u','-m','paper.scripts.run_phen_instacart_3seed','--contract',str(contract),
         '--contract-sha',digest,'--host',host,'--mode','dispatch','--device','cuda:0']
stdout=root/'dispatcher.stdout.log';stderr=root/'dispatcher.stderr.log';start=time.monotonic()
with stdout.open('xb') as out,stderr.open('xb') as err:
    child=subprocess.Popen(command,cwd=root/'source',env=env,stdout=out,stderr=err)
    with (root/'dispatcher.launch.json').open('x') as f:
        json.dump({'host':host,'pid':child.pid,'parent_pid':os.getpid(),'command':command,
                   'contract_raw_sha256':digest,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()},f,indent=2)
    code=child.wait()
record={'host':host,'pid':child.pid,'actual_os_returncode':code,'elapsed_seconds':time.monotonic()-start,
        'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor_own_exit_directly_observed':False}
for key,path in [('stdout',stdout),('stderr',stderr)]:
    record[key]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'path':path.name}
with (root/'dispatcher.exit.json').open('x') as f:json.dump(record,f,indent=2)
sys.exit(code if 0<=code<256 else 1)
'''

REMOTE = r'''
import hashlib,json,pathlib,shlex,subprocess,sys
root=pathlib.Path(sys.argv[1]);host=sys.argv[2];digest=sys.argv[3]
contract=root/'EXECUTION_CONTRACT.json'
assert hashlib.sha256(contract.read_bytes()).hexdigest()==digest
c=json.loads(contract.read_text());spec=c['hosts'][host]
for mode in ['check','synthetic','capacity']:
    observed=json.loads((root/(mode+'.exit.json')).read_text())
    assert observed['actual_os_returncode']==0
    for channel in ['stdout','stderr']:
        path=root/(mode+'.'+channel)
        assert path.stat().st_size==observed[channel+'_bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==observed[channel+'_sha256']
for mode in ['synthetic','capacity']:
    receipt=json.loads((root/'qualification'/(mode+'.json')).read_text())
    assert receipt['status']=='passed' and receipt['runtime']['device']=='cuda:0'
assert not (root/'dispatch_attempt.json').exists() and not (root/'jobs').exists()
session='phen_instacart_3seed_v2_'+host
assert subprocess.run([spec['tmux'],'has-session','-t',session],capture_output=True).returncode!=0
payload=sys.stdin.buffer.read()
with (root/'supervise.py').open('xb') as f:f.write(payload)
command=' '.join(shlex.quote(x) for x in ['/usr/bin/python3','-u',str(root/'supervise.py'),host,digest])
command+=' > '+shlex.quote(str(root/'supervisor.stdout.log'))+' 2> '+shlex.quote(str(root/'supervisor.stderr.log'))
result=subprocess.run([spec['tmux'],'new-session','-d','-s',session,command],capture_output=True,text=True)
record={'host':host,'tmux_session':session,'actual_tmux_exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'supervisor_sha256':hashlib.sha256(payload).hexdigest(),'contract_raw_sha256':digest}
with (root/'tmux_launch.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
sys.exit(result.returncode)
'''


def launch(host):
    local = json.loads((OUT / f"{host}_deploy_qualify.receipt.json").read_text())
    assert local["actual_ssh_exit"] == 0
    detail = json.loads((OUT / f"{host}_deploy_qualify.stdout").read_text())
    assert detail["status"] == "qualified" and all(x["actual_os_returncode"] == 0 for x in detail["results"])
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host,
               "python3 -c " + shlex.quote(REMOTE) + " " + " ".join(map(shlex.quote,
                  [C["hosts"][host]["root"], host, B["contract_raw_sha256"]]))]
    result = subprocess.run(command, input=SUPERVISOR.encode(), capture_output=True, timeout=45)
    record = {"host": host, "actual_ssh_exit": result.returncode, "observed_at": datetime.now(timezone.utc).isoformat()}
    for name, data in [("stdout", result.stdout), ("stderr", result.stderr)]:
        path = OUT / (host + "_launch." + name)
        with path.open("xb") as f:
            f.write(data)
        record[name] = {"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    with (OUT / (host + "_launch.receipt.json")).open("x") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record), flush=True)
    print(result.stdout.decode(), result.stderr.decode(), flush=True)
    return result.returncode


if __name__ == "__main__":
    # Both qualification gates must have passed before either fit is admitted.
    for host in C["hosts"]:
        assert json.loads((OUT / f"{host}_deploy_qualify.receipt.json").read_text())["actual_ssh_exit"] == 0
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        codes = list(pool.map(launch, ["5080", "5090"]))
    raise SystemExit(0 if codes == [0, 0] else 1)
