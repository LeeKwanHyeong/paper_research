"""One-shot transport and native qualification of the approved isolated bundle."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

OUT = Path(__file__).resolve().parent / "execution_v3"
CONTRACT = json.loads((OUT / "EXECUTION_CONTRACT.json").read_text())
BUNDLE = json.loads((OUT / "BUNDLE.json").read_text())

REMOTE = r'''
import hashlib,json,os,pathlib,subprocess,sys,tarfile,time
root=pathlib.Path(sys.argv[1]); host=sys.argv[2]
bundle_sha=sys.argv[3]; contract_sha=sys.argv[4]
root.mkdir(exist_ok=False)
payload=sys.stdin.buffer.read()
assert hashlib.sha256(payload).hexdigest()==bundle_sha, 'Transfer digest mismatch'
archive=root/'source_inputs.tar.gz'
with archive.open('xb') as f: f.write(payload)
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    assert len({m.name for m in members})==len(members)
    assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in members)
    tar.extractall(root,filter='data')
contract_path=root/'EXECUTION_CONTRACT.json'
assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==contract_sha
c=json.loads(contract_path.read_text()); spec=c['hosts'][host]
assert str(root)==spec['root']
env={**os.environ,'PYTHONPATH':str(root/'source'),'PYTHONDONTWRITEBYTECODE':'1',
     'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONHASHSEED':'42',
     'CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUBLAS_WORKSPACE_CONFIG':':4096:8',
     'NVIDIA_TF32_OVERRIDE':'0','MPLCONFIGDIR':str(root/'cache/matplotlib'),'XDG_CACHE_HOME':str(root/'cache')}
env.update(spec['process_environment'])
assert pathlib.Path(env['LD_LIBRARY_PATH'],'libnvrtc-builtins.so.13.0').is_file()
results=[]
for mode in ['check','synthetic','capacity']:
    command=[spec['python'],'-u','-m','paper.scripts.run_phen_instacart_3seed','--contract',str(contract_path),
             '--contract-sha',contract_sha,'--host',host,'--mode',mode,'--device','cuda:0']
    started=time.monotonic()
    out=root/(mode+'.stdout'); err=root/(mode+'.stderr')
    with out.open('xb') as o,err.open('xb') as e:
        child=subprocess.Popen(command,cwd=root/'source',env=env,stdout=o,stderr=e)
        try: code=child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            child.kill();code=child.wait()
    result={'mode':mode,'actual_os_returncode':code,'elapsed_seconds':time.monotonic()-started,'pid':child.pid,
            'process_environment':spec['process_environment'],
            'command':command,'stdout_bytes':out.stat().st_size,'stderr_bytes':err.stat().st_size,
            'stdout_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256(err.read_bytes()).hexdigest()}
    with (root/(mode+'.exit.json')).open('x') as f: json.dump(result,f,indent=2)
    results.append(result)
    if code:
        print(json.dumps({'host':host,'status':'failed','results':results,'stderr':err.read_text()[-6000:]}),flush=True)
        sys.exit(code if 0<code<256 else 1)
receipts={m:json.loads((root/'qualification'/(m+'.json')).read_text()) for m in ['synthetic','capacity']}
print(json.dumps({'host':host,'status':'qualified','results':results,'qualification':receipts}),flush=True)
'''


def deploy(host):
    spec = CONTRACT["hosts"][host]
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=2", host,
               "python3 -c " + shlex.quote(REMOTE) + " " + " ".join(map(shlex.quote, [spec["root"], host,
                    BUNDLE["bundle_sha256"], BUNDLE["contract_raw_sha256"]]))]
    with (OUT / "source_inputs.tar.gz").open("rb") as archive:
        result = subprocess.run(command, stdin=archive, capture_output=True, timeout=590)
    record = {"host": host, "observed_at": datetime.now(timezone.utc).isoformat(), "actual_ssh_exit": result.returncode}
    for name, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        path = OUT / f"{host}_deploy_qualify.{name}"
        with path.open("xb") as stream:
            stream.write(value)
        record[name] = {"path": path.name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    with (OUT / f"{host}_deploy_qualify.receipt.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)
    if result.returncode:
        print(result.stdout.decode(errors="replace")[-6500:], result.stderr.decode(errors="replace"), flush=True)
    else:
        detail = json.loads(result.stdout)
        print(json.dumps({"host":host,"status":detail["status"],"actual_exits":[x['actual_os_returncode'] for x in detail['results']],
                          "capacity_peak_allocated_bytes":detail['qualification']['capacity']['peak_allocated_bytes'],
                          "capacity_peak_reserved_bytes":detail['qualification']['capacity']['peak_reserved_bytes']}),flush=True)
    return result.returncode


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        codes = list(pool.map(deploy, ["5080", "5090"]))
    raise SystemExit(0 if codes == [0, 0] else 1)
