"""Finite CPU-only diagnostic package/run/collection; never resumes Cap27."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_time_quantity_balance_20261018_v1"
BUNDLE = ROOT / "search_artifacts" / NAME
REPORT = ROOT / "reports" / NAME
PROBE = "paper/scripts/probe_titantpp_time_quantity_balance.py"
MODULE = "paper.scripts.probe_titantpp_time_quantity_balance"
CAP_BUNDLE = ROOT / "search_artifacts/titantpp_mlp_cap36_20261017_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


# One bounded CPU controller per host. It never sends signals to paused jobs.
CONTROLLER = r'''import concurrent.futures,hashlib,json,os,pathlib,signal,subprocess,sys,threading,time,traceback
root=pathlib.Path(__file__).resolve().parent
host=sys.argv[1]
c=json.loads((root/'operations.json').read_text())
h=c['hosts'][host]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
 if not ok: raise ValueError(message)
def write(path,value):
 with pathlib.Path(path).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def identity(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rpartition(')')[2].split()
  return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'pgid':int(fields[2]),'start_ticks':int(fields[19]),'argv':(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),'cwd':os.readlink(p/'cwd')}
 except (FileNotFoundError,ProcessLookupError): return None
def paused():
 expected=h['paused_processes']; observed=[]
 for old in expected:
  live=identity(old['pid']);require(live is not None,'Paused process disappeared')
  require(all(live[k]==old[k] for k in ('pid','start_ticks','pgid','argv','cwd')) and live['state'] in ('T','t'),'Paused process identity/state changed')
  observed.append(live)
 dispatcher=expected[0]['pid']; known={x['pid'] for x in expected}; descendants={dispatcher}
 ppid={}
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    fields=(p/'stat').read_text().rpartition(')')[2].split();ppid[int(p.name)]=int(fields[1])
   except (FileNotFoundError,ProcessLookupError,PermissionError): pass
 while True:
  more={pid for pid,parent in ppid.items() if parent in descendants}
  if more <= descendants: break
  descendants.update(more)
 require(descendants==known,'Unverified descendant of paused campaign')
 return observed
require(str(root)==h['root'],'Diagnostic root mismatch')
require(not (root/'attempt.json').exists(),'Existing diagnostic attempt; no retry')
write(root/'attempt.json',{'host':host,'started_at':time.time(),'automatic_retry':False})
for name,d in c['files'].items():require(digest(root/name)==d,'Package SHA changed: '+name)
source_before={name:digest(root/name) for name in c['files']}
paused_before=paused()
(root/'logs').mkdir();(root/'results').mkdir()
failed=threading.Event();started=time.monotonic();lock=threading.Lock();active={}
env={**os.environ,**h['environment'],'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4','PYTHONPATH':str(root/'source')}
def stop_own(process):
 require(process.pid in active,'Unowned diagnostic child')
 live=identity(process.pid)
 if live is not None:require(live['pgid']==process.pid and live['argv']==active[process.pid],'Owned diagnostic process identity changed')
 if process.poll() is None:
  try:os.killpg(process.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:
   try:os.killpg(process.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   process.wait(timeout=5)
def one(job):
 if failed.is_set():return {'id':job['id'],'status':'not_started_after_failure'}
 command=[h['python'],'-u','-m',c['worker_module'],'--contract',str(root/'reference_contract.json'),'--contract-sha',c['reference_contract_sha256'],'--input-root',str(root),'--dataset',job['dataset'],'--seed',str(job['seed']),'--initialization',job['initialization'],'--device','cpu','--num-batches','3','--output',str(root/'results'/ (job['id']+'.json'))]
 if job['initialization']=='selected':command+=['--checkpoint',str(root/job['checkpoint']),'--checkpoint-sha',job['checkpoint_sha256'],'--checkpoint-state-sha',job['checkpoint_state_sha256'],'--selected-epoch',str(job['selected_epoch'])]
 with (root/'logs'/(job['id']+'.stdout')).open('xb') as out,(root/'logs'/(job['id']+'.stderr')).open('xb') as err:
  process=subprocess.Popen(command,cwd=root/'source',env={**env,'PYTHONHASHSEED':str(job['seed'])},stdout=out,stderr=err,start_new_session=True)
  with lock:active[process.pid]=command
  timed_out=False
  try:code=process.wait(timeout=min(300,max(1,900-(time.monotonic()-started))))
  except subprocess.TimeoutExpired:
   timed_out=True;stop_own(process);code=process.returncode
  finally:
   with lock:active.pop(process.pid,None)
 receipt={'id':job['id'],'pid':process.pid,'command':command,'actual_os_returncode':code,'timeout':timed_out,'observed_exit_at':time.time(),'status':'complete' if code==0 and not timed_out else 'failed'}
 if receipt['status']=='complete':
  try:
   value=json.loads((root/'results'/(job['id']+'.json')).read_text())
   require(value['status']=='complete' and value['dataset']==job['dataset'] and value['seed']==job['seed'] and value['initialization']==job['initialization'],'Diagnostic result mismatch')
   require(value['optimizer_steps']==0 and value['test_records_opened']==0 and value['validation_records_selected']==0 and value['model_parameters_buffers_context_unchanged'] and value['process_rng_restored'],'Diagnostic scope/state invariant failed')
   require(value['runtime']['device_type']=='cpu' and value['runtime']['torch']=='2.11.0+cu130','Expected pinned native CPU runtime')
  except Exception:receipt.update(status='failed',traceback=traceback.format_exc())
 if receipt['status']!='complete':failed.set()
 write(root/'logs'/(job['id']+'.exit.json'),receipt)
 return receipt
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,h['jobs']))
 require(all(r['status']=='complete' for r in results),'At least one diagnostic failed or was not started')
 # The two initializations must see exactly the same three Train batches.
 for selected in [j for j in h['jobs'] if j['initialization']=='selected']:
  fresh_id=selected['id'].removesuffix('__selected')+'__fresh'
  a=json.loads((root/'results'/(fresh_id+'.json')).read_text());b=json.loads((root/'results'/(selected['id']+'.json')).read_text())
  require(a['fresh_initial_state_sha256']==b['fresh_initial_state_sha256'],'Fresh baseline initialization differs')
  require(a['batch_count']==b['batch_count']==3,'Batch count mismatch')
  for x,y in zip(a['batches'],b['batches']):require(all(x[k]==y[k] for k in ('target_ids_sha256','batch_tensors_sha256')),'Fresh/selected Train batch mismatch')
 paused_after=paused()
 require(not active,'Diagnostic child still registered')
 require(all(digest(root/name)==d for name,d in source_before.items()),'Package/input/checkpoint changed')
 final={'status':'complete','host':host,'condition_count':len(results),'actual_worker_exit0_count':len(results),'results':results,'paused_before':paused_before,'paused_after':paused_after,'prior_campaign_resumed':False,'diagnostic_child_pids_absent':all(identity(r['pid']) is None for r in results),'optimizer_steps':0,'held_out_test_evaluated':False,'completed_at':time.time()}
 require(final['diagnostic_child_pids_absent'],'Diagnostic child PID still exists')
except Exception:
 final={'status':'failed','host':host,'traceback':traceback.format_exc(),'prior_campaign_resumed':False,'completed_at':time.time()}
final['files']={str(p.relative_to(root)):digest(p) for d in ('results','logs') for p in sorted((root/d).glob('*')) if p.is_file()}
write(root/'final_receipt.json',final)
print(json.dumps(final,ensure_ascii=False,allow_nan=False),flush=True)
sys.exit(0 if final['status']=='complete' else 1)
'''


def prepare():
    require(not BUNDLE.exists(), "Existing diagnostic bundle; no overwrite/retry")
    inputs = read(REPORT / "PROBE_INPUTS.json")
    for path, digest in inputs["input_files_sha256"].items():
        require(sha(path) == digest, "Bound diagnostic input changed")
    committed = subprocess.check_output(["git", "ls-files", "--", PROBE, "paper/scripts/manage_titantpp_time_quantity_balance.py"], cwd=ROOT, text=True).splitlines()
    require(set(committed) == {PROBE, "paper/scripts/manage_titantpp_time_quantity_balance.py"}, "Commit new diagnostic source before prepare")
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *committed], cwd=ROOT, text=True)
    require(not changed.strip(), "Commit changed diagnostic source before prepare")
    reference = Path(inputs["cap36_lineage"]["execution_contract_path"])
    old = read(reference)
    sources = {**old["source"]["files"], PROBE: sha(ROOT / PROBE)}
    for name, digest in sources.items():
        require(sha(ROOT / name) == digest, "Frozen source changed: " + name)
    files = {"reference_contract.json": reference, "PROBE_INPUTS.json": REPORT / "PROBE_INPUTS.json"}
    for d in inputs["datasets"]:
        files["inputs/data/" + d["dataset"] + "/train_validation.parquet"] = Path(d["train_validation_input_path"])
        files["inputs/data/" + d["dataset"] + "/split_manifest.json"] = Path(d["split_manifest_path"])
    for cp in inputs["checkpoints"]:
        files["inputs/checkpoints/" + cp["dataset"] + "__seed" + str(cp["seed"]) + ".pt"] = Path(cp["checkpoint_path"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hosts = {}
    for host in ("5080", "5090"):
        pause_path, qa_path = REPORT / f"PAUSE_RECEIPT_{host}.json", REPORT / f"PAUSE_COMPLETENESS_QA_{host}.json"
        pause, qa = read(pause_path), read(qa_path)
        require(pause["status"] == "paused" and qa["all_live_direct_children_stopped"] is True, "Prior campaign is not verified paused")
        paused = [pause["dispatcher"], *[w["after"] for w in pause["workers"]]]
        require({p["pid"] for p in paused} == {p["pid"] for p in qa["stopped"]} and all(p["state"] in ("T", "t") for p in paused), "Paused lineage mismatch")
        jobs = []
        for cp in inputs["checkpoints"]:
            if cp["training_host"] != host:
                continue
            for mode in ("fresh", "selected"):
                jobs.append({"id": f"{cp['dataset']}__seed{cp['seed']}__{mode}", "dataset": cp["dataset"], "seed": cp["seed"], "initialization": mode,
                    "checkpoint": "inputs/checkpoints/" + cp["dataset"] + "__seed" + str(cp["seed"]) + ".pt", "checkpoint_sha256": cp["checkpoint_sha256"], "checkpoint_state_sha256": cp["checkpoint_state_sha256"], "selected_epoch": cp["selected_epoch"]})
        h = old["hosts"][host]
        remote_root = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{NAME}_{host}"
        environment = {**h["environment"], "SOURCE_REVISION": revision, "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": remote_root + "/source", "MPLCONFIGDIR": remote_root + "/cache/matplotlib",
            "XDG_CACHE_HOME": remote_root + "/cache"}
        hosts[host] = {"root": remote_root, "python": h["python"], "environment": environment, "paused_processes": paused, "jobs": jobs,
            "pause_receipt_sha256": sha(pause_path), "pause_completeness_sha256": sha(qa_path)}
    require(sum(len(h["jobs"]) for h in hosts.values()) == 18, "Expected18 diagnostic conditions")
    BUNDLE.mkdir(parents=True)
    staging = BUNDLE / "package"
    staging.mkdir()
    for name, origin in files.items():
        destination = staging / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with origin.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
    for name in sources:
        destination = staging / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    (staging / "source/sample_data").mkdir(exist_ok=True)
    (staging / "controller.py").write_text(CONTROLLER)
    inventory = {str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}
    operations = {"schema": "titantpp_cpu_gradient_probe_operations_v1", "hosts": hosts, "worker_module": MODULE,
        "reference_contract_sha256": sha(reference), "maximum_cpu_children": 3, "child_timeout_seconds": 300, "host_timeout_seconds": 900,
        "optimizer_steps": 0, "new_training": False, "native_gpu_used": False, "existing_campaign_signals_sent": False, "files": inventory,
        "source_revision": revision, "model_science_source_revision": old["source"]["base_git_revision"],
        "model_science_source_files_sha256": old["source"]["files_sha256"]}
    write(staging / "operations.json", operations)
    archive_path = BUNDLE / "deployment.tar.gz"
    with archive_path.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as archive:
        for name in sorted(staging.iterdir()):
            archive.add(name, arcname=name.name)
    receipt = {"status": "prepared", "created_at": datetime.now(timezone.utc).isoformat(), "archive_sha256": sha(archive_path),
        "operations_sha256": sha(staging / "operations.json"), "source_files": len(sources), "input_files": len(files),
        "conditions_per_host": {h:len(v["jobs"]) for h,v in hosts.items()}, "optimizer_steps": 0, "held_out_test_evaluated": False}
    write(REPORT / "PROBE_PREPARED.json", receipt)
    return receipt


def transport(host, phase, command, *, data=None, timeout=930):
    prefix = BUNDLE / "transport" / f"{host}_{phase}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command], input=data, capture_output=True, timeout=timeout)
        code, stdout, stderr, timed_out = result.returncode, result.stdout, result.stderr, False
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr, timed_out = None, error.stdout or b"", error.stderr or b"", True
    for suffix, content in ((".stdout", stdout), (".stderr", stderr)):
        with prefix.with_suffix(suffix).open("xb") as f:
            f.write(content)
    write(prefix.with_suffix(".json"), {"actual_transport_exit": code, "timeout": timed_out, "stdout_bytes": len(stdout), "stdout_sha256": sha(prefix.with_suffix(".stdout")), "stderr_bytes": len(stderr), "stderr_sha256": sha(prefix.with_suffix(".stderr"))})
    require(code == 0, f"{phase} transport failed; raw evidence preserved; no automatic retry")
    return stdout


def run(host):
    prepared = read(REPORT / "PROBE_PREPARED.json")
    archive = BUNDLE / "deployment.tar.gz"
    require(sha(archive) == prepared["archive_sha256"], "Prepared archive changed")
    operations = read(BUNDLE / "package/operations.json")
    require(sha(BUNDLE / "package/operations.json") == prepared["operations_sha256"], "Prepared operations changed")
    h = operations["hosts"][host]
    write(BUNDLE / "attempts" / f"{host}.json", {"host": host, "automatic_retry": False})
    transport(host, "install", "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"]), data=archive.read_bytes(), timeout=180)
    environment = {**h["environment"], "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    command = shlex.join(["env", *[f"{k}={v}" for k,v in environment.items()], h["python"], "-u", h["root"] + "/controller.py", host])
    result = json.loads(transport(host, "run", command))
    require(result["status"] == "complete", "Native CPU diagnostic did not complete")
    write(REPORT / f"PROBE_RUN_{host}.json", result)
    return {"host": host, "status": result["status"], "condition_count": result["condition_count"]}


def collect(host):
    operations = read(BUNDLE / "package/operations.json")
    h = operations["hosts"][host]
    destination = BUNDLE / "retrieval" / host
    destination.mkdir(parents=True, exist_ok=False)
    code = "import pathlib,json,tarfile,sys; root=pathlib.Path(" + repr(h["root"]) + "); r=json.loads((root/'final_receipt.json').read_text()); assert r['status']=='complete' and r['diagnostic_child_pids_absent']; a=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz'); [a.add(root/n,arcname=n) for n in ['final_receipt.json','attempt.json','results','logs']]; a.close()"
    command = shlex.join([h["python"], "-c", code])
    raw = transport(host, "collect", command, timeout=180)
    archive_path = destination / "original.tar.gz"
    with archive_path.open("xb") as f:
        f.write(raw)
    original = destination / "originals"
    original.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members = archive.getmembers()
        require(all(not Path(m.name).is_absolute() and ".." not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in members), "Unsafe diagnostic archive")
        archive.extractall(original, filter="data")
    final = read(original / "final_receipt.json")
    require(final["status"] == "complete" and final["condition_count"] == len(h["jobs"]), "Collected diagnostic incomplete")
    for name,digest in final["files"].items():
        require(sha(original / name) == digest, "Collected diagnostic SHA changed")
    write(destination / "receipt.json", {"status": "complete", "host": host, "archive_sha256": sha(archive_path), "final_receipt_sha256": sha(original/'final_receipt.json'), "condition_count": final["condition_count"], "raw_inventory": final["files"], "new_training": False, "optimizer_steps": 0})
    return {"host": host, "status": "complete", "condition_count": final["condition_count"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "run", "collect"))
    parser.add_argument("--host", choices=("5080", "5090"))
    args = parser.parse_args(argv)
    require((args.mode == 'prepare') == (args.host is None), "Only prepare omits host")
    result = prepare() if args.mode == 'prepare' else globals()[args.mode](args.host)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
