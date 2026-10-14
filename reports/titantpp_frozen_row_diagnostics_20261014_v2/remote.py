"""Transport exclusively to new diagnostic roots; retain stdout/stderr bytes."""
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "search_artifacts/titantpp_frozen_row_diagnostics_20261014_v2"
PREFIX = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_frozen_row_diagnostics_20261014_v2_"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ssh(host, label, code, payload=None, timeout=180):
    out = B / "control" / f"{host}_{label}.stdout"
    err = out.with_suffix(".stderr")
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", host,
        "/usr/bin/python3 -c " + shlex.quote(code) + " " + shlex.quote(PREFIX + host)]
    started = time.time()
    timed_out, returncode = False, None
    try:
        with out.open("xb") as f, err.open("xb") as e:
            completed = subprocess.run(argv, input=payload, stdout=f, stderr=e, timeout=timeout)
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    receipt = {"host": host, "action": label, "argv": argv, "actual_transport_exit": returncode,
        "timeout": timed_out, "timeout_seconds": timeout,
        "started_unix": started, "completed_unix": time.time(), "stdout_sha256": sha(out),
        "stderr_sha256": sha(err), "stdout_bytes": out.stat().st_size, "stderr_bytes": err.stat().st_size}
    with out.with_suffix(".exit.json").open("x") as f:
        f.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:v for k,v in receipt.items() if k != "argv"}), flush=True)
    if timed_out:
        raise subprocess.TimeoutExpired(argv, timeout)
    if returncode:
        print(err.read_text(errors="replace")[-5000:])
        raise SystemExit(returncode)
    return out


HEALTH = '''import json,subprocess,sys
from pathlib import Path
r=Path(sys.argv[1]); assert r.parent.name=='paper_research_experiment_artifacts'
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True)
processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
print(json.dumps({'root_exists':r.exists(),'gpu':gpu.stdout,'processes':processes.stdout,'read_only':True}))
'''

DEPLOY = '''import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]); assert r.parent.name=='paper_research_experiment_artifacts' and r.name.startswith('titantpp_frozen_row_diagnostics_20261014_v2_'); assert not r.exists()
data=sys.stdin.buffer.read();t=tarfile.open(fileobj=io.BytesIO(data),mode='r:gz')
members=t.getmembers(); assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
manifest=json.load(t.extractfile('transport_manifest.json'));assert set(manifest)=={m.name for m in members}-{'transport_manifest.json'}
for m in members:
 if m.name!='transport_manifest.json':assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==manifest[m.name]
r.mkdir()
for m in members:
 p=r/m.name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(t.extractfile(m).read())
print(json.dumps({'status':'deployed','files':len(manifest),'root':str(r),'original_root_writes':0}))
'''

LAUNCH = '''import hashlib,json,shlex,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);c=json.loads((r/'contract.json').read_text());assert str(r)==c['authorized_remote_root'];assert hashlib.sha256((r/'operation/evaluate.py').read_bytes()).hexdigest()==c['evaluator_sha256']
session='frozen-rows-'+c['host']+'-20261014-v1';assert not (r/'launch_receipt.json').exists()
argv=[c['runtime']['python'],str(r/'operation/supervise.py'),str(r)]
command='env PYTHONDONTWRITEBYTECODE=1 '+shlex.join(argv)+' > '+shlex.quote(str(r/'supervisor.log'))+' 2>&1'
tmux='/usr/bin/tmux' if c['host']=='5080' else '/opt/miniconda3/envs/ai_env/bin/tmux'
x=subprocess.run([tmux,'new-session','-d','-s',session,command],capture_output=True,text=True);assert x.returncode==0,x.stderr
v={'status':'launched','session':session,'argv':argv,'started_unix':time.time(),'automatic_retry':False}
with (r/'launch_receipt.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))
'''

STATUS = '''import json,sys
from pathlib import Path
r=Path(sys.argv[1]);files=list((r/'results').glob('*/*_worker_exit.json'))
workers=[json.loads(p.read_text()) for p in files]
v={'finished_workers':len(workers),'failed_workers':[x for x in workers if x['exit_code']!=0],'complete':(r/'pipeline_actual_exit.json').exists()}
for n in ['pipeline_actual_exit.json','pipeline_failure.json','pipeline_completion.json']:
 if (r/n).exists():
  obj=json.loads((r/n).read_text());obj.pop('worker_exits',None);v[n]=obj
v['recent_log']=(r/'supervisor.log').read_text(errors='replace')[-2500:] if (r/'supervisor.log').exists() else ''
print(json.dumps(v))
'''

RETRIEVE = '''import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]);assert (r/'pipeline_actual_exit.json').exists()
names=[p for p in r.rglob('*') if p.is_file() and (p.relative_to(r).parts[0] in ['results','logs'] or len(p.relative_to(r).parts)==1) and 'runtime_cache' not in p.parts]
manifest={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in names}
t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')
for p in names:t.add(p,arcname=p.relative_to(r).as_posix(),recursive=False)
data=json.dumps(manifest,sort_keys=True).encode();info=tarfile.TarInfo('retrieval_manifest.json');info.size=len(data);t.addfile(info,io.BytesIO(data));t.close()
'''


def main():
    action, host = sys.argv[1:3]
    assert host in {"5080", "5090"}
    if action == "health":
        print(ssh(host, "health", HEALTH).read_text())
    elif action == "deploy":
        print(ssh(host, "deploy", DEPLOY, (B / f"packet_{host}.tar.gz").read_bytes(), 300).read_text())
    elif action == "launch":
        print(ssh(host, "launch", LAUNCH).read_text())
    elif action == "status":
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        print(ssh(host, "status_" + stamp, STATUS).read_text())
    elif action == "retrieve":
        archive = ssh(host, "retrieve", RETRIEVE, timeout=900)
        dest = B / "native" / host
        dest.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive, "r:gz") as tar:
            manifest = json.load(tar.extractfile("retrieval_manifest.json"))
            for m in tar.getmembers():
                assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                raw = tar.extractfile(m).read()
                if m.name != "retrieval_manifest.json":
                    assert hashlib.sha256(raw).hexdigest() == manifest[m.name]
                p = dest / m.name
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('xb') as f:f.write(raw)
        print(json.dumps({"retrieved":len(manifest),"host":host,"archive_sha256":sha(archive)}))
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
