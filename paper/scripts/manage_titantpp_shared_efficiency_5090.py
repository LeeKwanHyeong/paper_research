"""One isolated, paired Shared15/MAC cost measurement; never starts scientific fits."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import textwrap

REPO = Path(__file__).resolve().parents[2]
NAME = "titantpp_shared_efficiency_5090_20261020_v1"
BUNDLE = REPO / "search_artifacts" / NAME
REPORT = REPO / "reports" / NAME
PARENT = REPO / "search_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1"
OLD = REPO / "search_artifacts/titantpp_efficiency_5080_20260930_v1"
REMOTE = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + NAME
ARMS = ["titantpp_history_mlp_shared15_v1", "titantpp_titans_mac"]


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
def require(x, message):
    if not x: raise ValueError(message)


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write("\n")


SUPERVISOR = '''import datetime,json,os,pathlib,subprocess,sys,time,traceback
root=pathlib.Path(__file__).resolve().parent
def write(name,value):
 with (root/name).open('x') as f:json.dump(value,f,indent=2);f.write('\\n')
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (root/'controller.stdout').open('xb') as out,(root/'controller.stderr').open('xb') as err:
 child=subprocess.Popen([sys.executable,str(root/'benchmark.py'),'controller'],cwd=root/'source',stdout=out,stderr=err,start_new_session=True)
 write('controller_start.json',{'supervisor_pid':os.getpid(),'controller_pid':child.pid,'started_at':started,'command':child.args})
 result=child.wait()
write('controller_exit.json',{'controller_pid':child.pid,'pid':child.pid,'actual_os_returncode':result,'returncode':result,'actual_controller_exit_observed':True,'pid_absent_after_wait':not pathlib.Path('/proc',str(child.pid)).exists(),'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor_own_os_exit_observed':False})
sys.exit(result)
'''


def prepare():
    require(not BUNDLE.exists(), "Benchmark bundle already exists; do not replace or restart")
    parent = read(PARENT / "execution_contract.json")
    old = read(OLD / "contract.json")
    source = dict(parent["source"]["files"])
    locations = {name: REPO / name for name in source}
    for name in ("paper/scripts/titans_mac_observed_time_runtime.py", "paper/scripts/titans_mac_observed_time_contract.py"):
        source[name] = old["source_files"][name]; locations[name] = OLD / "source" / name
    require(len(source) == 985 and canonical(source) == "7905cb02aa6b51c3be5754260e48709f9d4f486276fc85c3ff460166b7032d7f", "Frozen source lineage changed")
    for name, digest in source.items(): require(sha(locations[name]) == digest, "Frozen source changed: " + name)
    for name, digest in parent["input_files"].items(): require(sha(PARENT / "input_snapshot" / name) == digest, "Frozen input changed: " + name)
    runtime = REPO / "paper/scripts/titantpp_shared_efficiency_benchmark.py"
    require(runtime.is_file(), "New benchmark runtime not ready")
    BUNDLE.mkdir(parents=True); REPORT.mkdir(parents=True, exist_ok=True)
    for name, origin in locations.items():
        dest = BUNDLE / "source" / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(origin, dest)
    for name in parent["input_files"]:
        dest = BUNDLE / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(PARENT / "input_snapshot" / name, dest)
    shutil.copyfile(runtime, BUNDLE / "benchmark.py")
    (BUNDLE / "supervise.py").write_text(SUPERVISOR)
    contract = deepcopy(old)
    host = parent["hosts"]["5090"]
    contract.update(schema="titantpp_shared_paired_efficiency_v1", models=ARMS, source_files=source,
                    source_closure_sha256=canonical(source), created_at=now(), max_seconds=1800,
                    child_timeout_seconds=600, scientific_fit_count=0, scientific_checkpoint_saved=False,
                    inherited_raw9_contract_sha256=sha(PARENT / "execution_contract.json"))
    contract["host"].update(assigned_datasets=[d["dataset_id"] for d in parent["datasets"]],
        gpu_uuid=host["gpu_uuid"], root=REMOTE, source_root=REMOTE + "/source", output_dir=REMOTE + "/results",
        python=host["python"], tmux="shared_efficiency_5090_20261020", tmux_binary=host["tmux_binary"],
        runtime_expected={**old["host"]["runtime_expected"], **host["runtime_expected"]})
    contract["environment"] = {**host["environment"], "PYTHONPATH": REMOTE + "/source",
        "MPLCONFIGDIR": REMOTE + "/cache/matplotlib", "XDG_CACHE_HOME": REMOTE + "/cache",
        "TORCHINDUCTOR_CACHE_DIR": REMOTE + "/cache/torchinductor"}
    contract["host"]["environment"] = contract["environment"]
    contract["datasets"] = deepcopy(parent["datasets"])
    for d in contract["datasets"]:
        for kind in ("data", "split_manifest"):
            item = d["inherited_data_identity"][kind]; item["path"] = REMOTE + "/" + item["path"]
        d["prepared_train_validation_path"] = d["inherited_data_identity"]["data"]["path"]
    contract["protocol"].update(repetitions=3, warmup_batches_per_worker=5, measured_batches_per_worker=32,
        workers=36, maximum_optimizer_steps=666, maximum_evaluation_batches=666,
        maximum_seconds_including_remote_preparation=1800,
        order="dataset,mode,rep then Shared15-MAC except rep2 MAC-Shared15",
        runtime="existing native5090; no install/redeploy")
    contract["scope_description"] = "original uncapped Train-only paired cost; no Validation/Test performance or scientific checkpoint updates"
    contract["operational_files"] = {name: sha(BUNDLE / name) for name in ("benchmark.py", "supervise.py")}
    write(BUNDLE / "contract.json", contract)
    write(BUNDLE / "approval.json", {"approved": True, "contract_sha256": canonical(contract),
        "approved_at": now(), "user_instruction": "효율관련해서 세 데이터셋을 다시 측정하는 방법에 대해 5090에서 지금 즉시 진행하자. 기존 MAC의 경우 5080에서 했다면 5090에서 다시 한번 테스트하는걸로하자.",
        "host": "5090", "additional_5080_training_unchanged": True})
    files = {str(p.relative_to(BUNDLE)): sha(p) for p in BUNDLE.rglob("*") if p.is_file()}
    write(BUNDLE / "deployment_manifest.json", {"files": files, "contract_sha256": canonical(contract)})
    with tarfile.open(BUNDLE / "payload.tar.gz", "w:gz") as tf:
        for name in files: tf.add(BUNDLE / name, arcname=name, recursive=False)
        tf.add(BUNDLE / "deployment_manifest.json", arcname="deployment_manifest.json")
    write(REPORT / "PREPARED.json", {"status": "prepared", "contract_sha256": canonical(contract),
        "source_file_count": len(source), "input_file_count": len(parent["input_files"]),
        "payload_sha256": sha(BUNDLE / "payload.tar.gz"), "native_measurement_started": False})
    return {"status": "prepared", "bundle": str(BUNDLE), "contract_sha256": canonical(contract)}


def verify():
    c = read(BUNDLE / "contract.json"); m = read(BUNDLE / "deployment_manifest.json")
    require(canonical(c) == m["contract_sha256"] == read(BUNDLE / "approval.json")["contract_sha256"], "Contract/approval mismatch")
    require(m["files"], "Empty inventory")
    for name, digest in m["files"].items(): require(sha(BUNDLE / name) == digest, "Local deployment changed: " + name)
    return c


def transport(phase, command, *, stdin=None, timeout=60):
    folder = BUNDLE / "transport"; folder.mkdir(exist_ok=True)
    result = subprocess.run(command, input=stdin, capture_output=True, timeout=timeout)
    for suffix, data in (("stdout", result.stdout), ("stderr", result.stderr)):
        with (folder / (phase + "." + suffix)).open("xb") as f: f.write(data)
    write(folder / (phase + ".json"), {"observed_at": now(), "actual_transport_exit": result.returncode,
        "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr),
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()})
    require(result.returncode == 0, f"{phase} transport exit {result.returncode}; original preserved; no automatic retry")
    return result.stdout


def ssh(phase, program, c, *, timeout=60):
    return transport(phase, ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5090",
        shlex.join([c["host"]["python"], "-"])], stdin=program.encode(), timeout=timeout)


def deploy():
    c = verify(); write(REPORT / "DEPLOYMENT_INTENT.json", {"created_at": now(), "contract_sha256": canonical(c), "root": REMOTE})
    program = f'''import json,pathlib,subprocess
root=pathlib.Path({REMOTE!r})
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
assert not apps.stdout.strip(),'5090 compute GPU occupied'
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
assert gpu.stdout.strip()=={c['host']['gpu_uuid']!r},'Wrong5090 GPU'
root.mkdir(exist_ok=False)
print(json.dumps({{'root':str(root),'compute_pids':[],'new_root':True}}))
'''
    admission = json.loads(ssh("admission", program, c))
    transport("upload", ["scp", "-q", str(BUNDLE / "payload.tar.gz"), "5090:" + REMOTE + "/incoming.tar.gz"], timeout=180)
    program = f'''import hashlib,json,pathlib,shlex,subprocess,tarfile
root=pathlib.Path({REMOTE!r})
assert hashlib.sha256((root/'incoming.tar.gz').read_bytes()).hexdigest()=={sha(BUNDLE / 'payload.tar.gz')!r}
with tarfile.open(root/'incoming.tar.gz','r:gz') as tf:tf.extractall(root,filter='data')
m=json.loads((root/'deployment_manifest.json').read_text())
for name,digest in m['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
c=json.loads((root/'contract.json').read_text())
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
assert not apps.stdout.strip(),'5090 became occupied'
command=shlex.join(['env',*[k+'='+v for k,v in c['environment'].items()],c['host']['python'],str(root/'supervise.py')])
r=subprocess.run([c['host']['tmux_binary'],'new-session','-d','-s',c['host']['tmux'],command],capture_output=True,text=True)
assert r.returncode==0,r.stderr
print(json.dumps({{'status':'launch_requested','root':str(root),'tmux':c['host']['tmux'],'tmux_returncode':r.returncode,'stderr':r.stderr,'scientific_fit_started':False}}))
'''
    launch = json.loads(ssh("launch", program, c))
    write(REPORT / "LAUNCH_RECEIPT.json", {**launch, "at": now(), "admission": admission})
    return launch


def finish():
    c = verify(); write(REPORT / "COLLECTION_INTENT.json", {"at": now(), "contract_sha256": canonical(c)})
    retrieval = BUNDLE / "retrieved"; require(not retrieval.exists(), "Retrieved originals already exist")
    program = f'''import json,pathlib,sys,tarfile,time
root=pathlib.Path({REMOTE!r});deadline=time.monotonic()+1900
while not (root/'controller_exit.json').exists():
 assert time.monotonic()<deadline,'Bounded benchmark wait expired'
 time.sleep(1)
exclude={{'incoming.tar.gz'}}
names=[p for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0] not in ('source','inputs','cache') and p.name not in exclude]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tf:
 for p in sorted(names):tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
'''
    out = transport("collection", ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5090", shlex.join([c["host"]["python"], "-"])], stdin=program.encode(), timeout=1930)
    archive = BUNDLE / "original.tar.gz"
    with archive.open("xb") as f: f.write(out)
    retrieval.mkdir()
    with tarfile.open(archive, "r:gz") as tf: tf.extractall(retrieval, filter="data")
    require(read(retrieval / "controller_exit.json")["actual_os_returncode"] == 0, "Native controller failed; originals preserved")
    manifest = read(retrieval / "terminal_manifest.json")
    require(manifest["status"] == "complete" and manifest["files"], "Native benchmark incomplete; originals preserved")
    for name, digest in manifest["files"].items(): require(sha(retrieval / name) == digest, "Original changed: " + name)
    require(sha(retrieval / "contract.json") == sha(BUNDLE / "contract.json"), "Collected contract mismatch")
    receipt = {"status": "verified", "archive_sha256": sha(archive), "manifest_sha256": sha(retrieval / "terminal_manifest.json"),
        "file_count": len(manifest["files"]), "native_controller_exit": 0, "scientific_checkpoint_saved": False, "at": now()}
    write(REPORT / "COLLECTION_RECEIPT.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--mode", choices=("prepare", "verify", "deploy", "finish"), required=True)
    a = p.parse_args(); value = {"prepare": prepare, "verify": verify, "deploy": deploy, "finish": finish}[a.mode]()
    print(json.dumps({k: v for k, v in value.items() if k not in ("source_files", "datasets")} if a.mode == "verify" else value, ensure_ascii=False))


if __name__ == "__main__": main()
