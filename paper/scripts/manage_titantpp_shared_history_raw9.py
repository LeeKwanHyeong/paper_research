"""Finite raw9 follow-up: wait for Cap18, then train, replay and collect once.

No recurring automation is created. Existing Cap18 completion processes and
every old artifact remain untouched. A failed/partial attempt is never retried.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from paper.scripts import titantpp_shared_history_raw9_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_history_mlp_shared_raw9_20261020_v1"
BUNDLE = ROOT / "search_artifacts" / NAME
REPORT = ROOT / "reports" / NAME
MODULE = "paper.scripts.titantpp_shared_history_raw9_runtime"
read, sha, canonical, require, now = runtime.read, runtime.sha_file, runtime.sha_json, runtime.require, runtime.now
_BASE = Path(__file__).with_name("manage_titantpp_quantity_loss_mlp18.py")
_BASE_SHA = "331aa17a98532217bb4cdda8865cf09530cc54b5c31f4968456655e93be27e8d"
require(sha(_BASE) == _BASE_SHA, "Preserved transport/archive manager changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_transport", _BASE)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)


def write(path, value):
    runtime.write(path, value, exclusive=True)


def verify_local(*, archive=False, require_predecessor_complete=False):
    preparation = importlib.import_module("paper.scripts.prepare_titantpp_shared_history_raw9")
    contract = preparation.verify(archive=archive, require_predecessor_complete=require_predecessor_complete)
    runtime.validate_contract(contract, "5080")
    return contract


def verify_collected(contract, host, originals):
    originals = Path(originals)
    require(sha(originals / "execution_contract.json") == sha(BUNDLE / "execution_contract.json"), "Collected raw9 contract changed")
    jobs = runtime.validate_contract(contract, host)
    ids = {job["id"] for job in jobs}
    final = read(originals / "final_receipt.json")
    require(final["status"] == "complete" and final["host"] == host and final["contract_sha256"] == canonical(contract)
        and final["completed_fit_count"] == final["validation_replay_count"] == 9 and final["test_count"] == 0
        and final["owned_worker_pids_absent"] is True and final["held_out_test_evaluated"] is False
        and final["prior_campaign_resumed"] is False, "Raw9 terminal counts/state incomplete")
    require(read(originals / "dispatcher_exit.json")["actual_os_returncode"] == 0, "Raw9 dispatcher actual exit0 missing")
    observed = read(originals / "collection_observation.json")
    expected = contract["hosts"][host]["paused_processes"]
    require(observed["owned_pids_absent"] is True and observed["prior_campaign_still_stopped"] is True
        and len(observed["paused_processes"]) == len(expected) and all(
            live["state"] in ("T", "t") and all(live[key] == old[key] for key in ("pid", "start_ticks", "pgid", "argv", "cwd"))
            for live, old in zip(observed["paused_processes"], expected)), "Collection owner/stopped state differs")
    manifest = final["artifact_manifest"]
    path = runtime.local_path(originals, manifest["path"])
    require(sha(path) == manifest["sha256"], "Raw9 terminal manifest changed")
    inventory = read(path)["files"]
    require(inventory and len(inventory) == manifest["file_count"], "Raw9 terminal inventory empty/incomplete")
    for name, digest in inventory.items():
        require(sha(runtime.local_path(originals, name)) == digest, "Collected raw9 original changed: " + name)
    require(set(final["stages"]) == set(runtime.STAGES), "Raw9 stage inventory differs")
    for stage in runtime.STAGES:
        result = final["stages"][stage]
        require(result["status"] == "complete" and set(result["exits"]) == set(result["receipts"]) == ids, "Raw9 terminal stage incomplete")
        for job in jobs:
            child, binding = result["exits"][job["id"]], result["receipts"][job["id"]]
            require(child["actual_os_returncode"] == 0 and child["timeout"] is False, "Raw9 scientific worker failed")
            path = runtime.local_path(originals, binding["path"])
            require(sha(path) == binding["sha256"], "Raw9 stage receipt changed")
            runtime.verify_receipt(contract, host, job, stage, originals=originals)
    return final


def collection_program(contract, host):
    h = contract["hosts"][host]
    return f'''import io,json,pathlib,sys,tarfile
from paper.scripts import titantpp_shared_history_raw9_runtime as r
root=pathlib.Path({h['root']!r});c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=={canonical(contract)!r},'Raw9 contract mismatch')
jobs=r.validate_contract(c,{host!r});ids={{j['id'] for j in jobs}}
f=r.read(root/'final_receipt.json');x=r.read(root/'dispatcher_exit.json')
r.require(f['status']=='complete' and f['host']=={host!r} and f['contract_sha256']==r.sha_json(c),'Incomplete raw9')
r.require(f['completed_fit_count']==f['validation_replay_count']==len(jobs)==9 and f['test_count']==0,'Raw9 counts differ')
r.require(x['actual_os_returncode']==0,'Raw9 dispatcher failed');pids=[x['pid']]
for stage in r.STAGES:
 s=f['stages'][stage]
 r.require(s['status']=='complete' and set(s['exits'])==set(s['receipts'])==ids,'Raw9 stage incomplete')
 for job in jobs:
  e=s['exits'][job['id']];r.require(e['actual_os_returncode']==0 and not e['timeout'],'Raw9 worker failed')
  pids.append(e['pid']);r.verify_receipt(c,{host!r},job,stage)
gpu,paused=r.gpu_state(c,{host!r})
r.require(all(r.proc_identity(pid) is None for pid in pids) and not set(pids).intersection(gpu['compute_pids']),'Raw9 owner remains')
observation={{'host':{host!r},'observed_at':r.now(),'owned_pids':pids,'owned_pids_absent':True,'gpu':gpu,'paused_processes':paused,'prior_campaign_still_stopped':True,'supervisor_own_exit_directly_observed':False}}
blob=(json.dumps(observation,ensure_ascii=False,indent=2)+chr(10)).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in sorted(root.rglob('*')):
  rel=path.relative_to(root)
  if rel.parts[0] in ('source','cache'):continue
  r.require(not path.is_symlink(),'Unexpected raw9 symlink')
  if path.is_file():archive.add(path,arcname=str(rel),recursive=False)
 info=tarfile.TarInfo('collection_observation.json');info.size=len(blob);archive.addfile(info,io.BytesIO(blob))
'''


def fresh_preflight(contract):
    """Read current Cap18 terminal owners/GPU before installing the new root."""
    host, h = "5080", contract["hosts"]["5080"]
    old_root = runtime._CAP_ROOT
    program = ("import pathlib,json\nfrom paper.scripts import titantpp_shared_history_cap_execution_v3_runtime as r\n"
        f"root=pathlib.Path({old_root!r});new=pathlib.Path({h['root']!r})\n"
        "r.require(not new.exists(),'Raw9 root already exists')\n"
        f"r.require(r.sha_file(root/'execution_contract.json')=={runtime._CAP_RAW_SHA!r},'Cap18 contract bytes differ')\n"
        "c=r.read(root/'execution_contract.json');jobs=r.validate_contract(c,'5080')\n"
        "f=r.read(root/'final_receipt.json');x=r.read(root/'dispatcher_exit.json')\n"
        "r.require(f['status']=='complete' and f['completed_fit_count']==f['validation_replay_count']==len(jobs)==18 and f['test_count']==0 and x['actual_os_returncode']==0,'Cap18 terminal incomplete')\n"
        "pids=[x['pid']]\n"
        "for stage in r.STAGES:\n"
        " for job in jobs:\n"
        "  e=f['stages'][stage]['exits'][job['id']];r.require(e['actual_os_returncode']==0 and not e['timeout'],'Cap18 worker failed');pids.append(e['pid'])\n"
        "r.require(all(r.proc_identity(pid) is None for pid in pids),'Cap18 owner remains')\n"
        "gpu,paused=r.gpu_state(c,'5080')\n"
        "print(json.dumps({'status':'passed','observed_at':r.now(),'gpu':gpu,'paused_processes':paused,'cap18_owned_pids_absent':True,'new_root_absent':True}))\n")
    command = "cd " + shlex.quote(old_root + "/source") + " && " + shlex.join([h["python"], "-c", program])
    result = json.loads(_impl.save_transport(host, "fresh_preflight", _impl.ssh(host, command)))
    require(result.get("status") == "passed" and result.get("cap18_owned_pids_absent") is True, "Raw9 fresh GPU admission failed")
    write(REPORT / "FRESH_PREFLIGHT.json", result)
    return result


def deploy():
    contract = verify_local(archive=True, require_predecessor_complete=True)
    h = contract["hosts"]["5080"]
    write(BUNDLE / "attempts/5080.json", {"at": now(), "host": "5080", "automatic_retry": False,
        "contract_sha256": sha(BUNDLE / "execution_contract.json")})
    fresh_preflight(contract)
    install = "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"])
    _impl.save_transport("5080", "install", _impl.ssh("5080", install, stdin=(BUNDLE / "deployment.tar.gz").read_bytes(), timeout=300))
    command = [h["python"], "-u", "-m", MODULE, "--contract", h["root"] + "/execution_contract.json",
        "--host", "5080", "--mode", "launch_dispatch"]
    supervisor = ("import os,subprocess,json,pathlib,datetime,sys\n"
        f"p=subprocess.Popen({command!r},cwd={h['source_root']!r},env={{**os.environ,**{h['environment']!r}}})\n"
        "code=p.wait()\n"
        f"with (pathlib.Path({h['root']!r})/'dispatcher_exit.json').open('x') as handle:\n"
        " json.dump({'pid':p.pid,'actual_os_returncode':code,'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor_own_exit_directly_observed':False},handle);handle.write(chr(10))\n"
        "sys.exit(code)\n")
    shell = shlex.join([h["python"], "-u", "-c", supervisor]) + " > " + shlex.quote(h["root"] + "/dispatcher.log") + " 2>&1"
    _impl.save_transport("5080", "launch", _impl.ssh("5080", shlex.join([h["tmux_binary"], "new-session", "-d", "-s", h["tmux"], shell])))
    result = {"status": "launch_requested_not_yet_verified", "host": "5080", "at": now(), "planned_fit_count": 9}
    write(REPORT / "LAUNCHED_5080.json", result)
    return result


def finish():
    require(not any(path.exists() for path in (BUNDLE / "retrieval/5080", BUNDLE / "retrieval/final_receipt.json",
        BUNDLE / "final_receipt.json", REPORT / "QUEUE_FAILURE.json", REPORT / "QUEUE_COMPLETE.json",
        REPORT / "FINISH_STARTED.json")), "Existing raw9 finish/retrieval/failure/completion; no repeated remote work")
    contract = verify_local(require_predecessor_complete=True)
    write(REPORT / "FINISH_STARTED.json", {"pid": os.getpid(), "started_at": now(), "automatic_retry": False,
        "purpose": "One finite raw9 terminal wait, immutable collection and comparison"})
    h = contract["hosts"]["5080"]
    program = ("import pathlib,time,json,sys\n"
        f"root=pathlib.Path({h['root']!r});deadline={h['absolute_host_deadline_unix']!r}\n"
        "while not (root/'dispatcher_exit.json').exists():\n"
        " if time.time()>=deadline:raise TimeoutError('Inherited raw9 deadline expired')\n"
        " time.sleep(min(30.,max(.01,deadline-time.time())))\n"
        "value=json.loads((root/'dispatcher_exit.json').read_text());print(json.dumps(value));sys.exit(0 if value['actual_os_returncode']==0 else 1)\n")
    result = _impl.ssh("5080", shlex.join([h["python"], "-c", program]),
        timeout=max(1., h["absolute_host_deadline_unix"] - time.time() + 60.))
    _impl.save_transport("5080", "finite_wait_complete", result)
    collected = _impl.collect("5080")
    importlib.import_module("paper.scripts.compare_titantpp_shared_history_raw9").main()
    return {"status": "complete", "collection": collected, "final_receipt": str(BUNDLE / "retrieval/final_receipt.json")}


def _finisher_alive(path):
    marker = read(path)
    require(type(marker.get("pid")) is int and marker["pid"] > 0 and marker.get("command"), "Cap18 finite finisher identity missing")
    observed = subprocess.run(["/bin/ps", "-p", str(marker["pid"]), "-o", "command="], capture_output=True, text=True, timeout=20)
    return observed.returncode == 0 and all(str(word) in observed.stdout for word in marker["command"])


def wait_predecessor():
    """One bounded local wait; the existing Cap18 finisher is never re-run."""
    contract = verify_local(archive=True)
    write(REPORT / "WAIT_STARTED.json", {"at": now(), "pid": os.getpid(), "recurring_automation": False,
        "purpose": "Wait existing Cap18 completion then one raw9 lifecycle", "automatic_retry": False})
    predecessor = contract["predecessor"]
    receipt_path = Path(predecessor["receipt_path"])
    finisher = Path(predecessor["finisher_receipt_path"])
    failure = Path(predecessor["failure_path"])
    deadline = contract["hosts"]["5080"]["absolute_host_deadline_unix"]
    while not receipt_path.exists():
        require(time.time() < deadline, "Inherited deadline expired while waiting for Cap18")
        require(not failure.exists(), "Existing Cap18 finisher failed; originals preserved")
        require(finisher.exists() and _finisher_alive(finisher), "Cap18 finite finisher stopped before verified completion")
        time.sleep(min(30., max(.01, deadline - time.time())))
    require(read(receipt_path).get("status") == "complete", "Cap18 completion receipt is not complete")
    verify_local(archive=True, require_predecessor_complete=True)
    launched = deploy()
    completed = finish()
    result = {"status": "complete", "launch": launched, "completion": completed, "recurring_automation": False}
    write(REPORT / "QUEUE_COMPLETE.json", result)
    return result


def start_queue():
    verify_local(archive=True)
    require(not (REPORT / "QUEUE_STARTED.json").exists() and not (REPORT / "WAIT_STARTED.json").exists(), "Raw9 finite queue already started")
    command = [sys.executable, "-u", "-m", "paper.scripts.manage_titantpp_shared_history_raw9", "--mode", "wait-predecessor"]
    with (REPORT / "queue.stdout.log").open("xb") as out, (REPORT / "queue.stderr.log").open("xb") as err:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out, stderr=err, start_new_session=True)
    result = {"pid": child.pid, "command": command, "started_at": now(), "recurring_automation": False,
        "own_os_exit_directly_observed": False, "purpose": "One Cap18 wait then raw9 train/Validation/collection"}
    write(REPORT / "QUEUE_STARTED.json", result)
    return result


_impl.ROOT, _impl.NAME, _impl.BUNDLE, _impl.REPORT, _impl.MODULE, _impl.runtime = ROOT, NAME, BUNDLE, REPORT, MODULE, runtime
_impl.verify_local, _impl.verify_collected, _impl.collection_program = verify_local, verify_collected, collection_program
status, collect, safe_extract = _impl.status, _impl.collect, _impl.safe_extract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "deploy", "status", "collect", "finish", "wait-predecessor", "start-queue"), required=True)
    args = parser.parse_args(argv)
    functions = {"verify": verify_local, "deploy": deploy, "status": lambda: status("5080"), "collect": lambda: collect("5080"),
        "finish": finish, "wait-predecessor": wait_predecessor, "start-queue": start_queue}
    try:
        value = functions[args.mode]()
    except Exception as error:
        if args.mode in ("finish", "wait-predecessor") and not (REPORT / "QUEUE_FAILURE.json").exists():
            write(REPORT / "QUEUE_FAILURE.json", {"status": "failed", "at": now(), "error": str(error), "automatic_retry": False})
        raise
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
