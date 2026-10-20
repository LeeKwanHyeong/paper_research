"""One MAC9 deployment, finite completion wait and immutable original collection."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from paper.scripts import prepare_titans_mac_accuracy9 as prep
from paper.scripts import titans_mac_accuracy9_runtime as runtime

ROOT, NAME, BUNDLE, REPORT = prep.ROOT, prep.NAME, prep.BUNDLE, prep.REPORT
MODULE = "paper.scripts.titans_mac_accuracy9_runtime"
read, sha, canonical, require, write = prep.read, prep.sha, prep.canonical, prep.require, prep.write
_BASE = Path(__file__).with_name("manage_titantpp_quantity_loss_mlp18.py")
_BASE_SHA = "331aa17a98532217bb4cdda8865cf09530cc54b5c31f4968456655e93be27e8d"
require(sha(_BASE) == _BASE_SHA, "Preserved bounded transport/archive manager changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_transport", _BASE)
_transport = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_transport)
_transport.BUNDLE, _transport.REPORT = BUNDLE, REPORT


def verify_local(integrity_sha, *, archive=False): return prep.verify(integrity_sha, archive=archive)


def fresh_preflight(c):
    h = c["hosts"]["5090"]
    minimal = {"hosts": {"5090": {"gpu_uuid": h["gpu_uuid"], "paused_processes": h["paused_processes"]}}, "prior_owned_pids": c["prior_owned_pids"]}
    program = ("import pathlib,json\nfrom paper.scripts import titantpp_quantity_loss_mlp18_runtime as r\n"
        f"c={minimal!r};h=c['hosts']['5090']\nr.require(not pathlib.Path({h['root']!r}).exists(),'New root already exists')\n"
        "r.require(h['paused_processes']==[],'Unexpected paused process');gpu=r.gpu_snapshot(h['gpu_uuid']);paused=[]\n"
        "r.require(not gpu['compute_pids'],'GPU is not idle; no installation or fit')\n"
        "r.require(all(r.proc_identity(pid) is None for pid in c['prior_owned_pids']),'Prior owner remains')\n"
        "print(json.dumps({'status':'passed','observed_at':r.now(),'new_root_absent':True,'gpu':gpu,'paused_processes':paused,'prior_owned_pids_absent':True}))\n")
    command = "cd " + shlex.quote(h["preflight_source_root"]) + " && " + shlex.join([h["python"], "-c", program])
    raw = _transport.save_transport("5090", "fresh_preflight", _transport.ssh("5090", command))
    result = json.loads(raw); require(result["status"] == "passed" and result["prior_owned_pids_absent"] is True, "Native admission failed")
    write(REPORT / "FRESH_PREFLIGHT.json", result); return result


def deploy(integrity_sha):
    c = verify_local(integrity_sha, archive=True); h = c["hosts"]["5090"]
    require(not any((REPORT / p).exists() for p in ("LAUNCHED_5090.json", "QUEUE_FAILURE.json", "QUEUE_COMPLETE.json")), "Existing launch/failure; no retry")
    write(BUNDLE / "attempts/5090.json", {"at": runtime.now(), "contract_sha256": sha(BUNDLE / "execution_contract.json"), "automatic_retry": False})
    fresh_preflight(c)
    _transport.save_transport("5090", "install", _transport.ssh("5090", "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"]),
        stdin=(BUNDLE / "deployment.tar.gz").read_bytes(), timeout=300))
    command = [h["python"], "-u", "-m", MODULE, "--contract", h["root"] + "/execution_contract.json", "--host", "5090", "--mode", "launch_dispatch"]
    supervisor = ("import os,subprocess,pathlib,json,sys,datetime,time,signal\nfrom paper.scripts import titans_mac_accuracy9_runtime as r\n"
        f"command={command!r};root=pathlib.Path({h['root']!r});deadline={h['absolute_host_deadline_unix']!r}\n"
        f"child=subprocess.Popen(command,cwd={h['source_root']!r},env={{**os.environ,**{h['environment']!r}}},start_new_session=True)\n"
        "identity=r.proc_identity(child.pid);r.require(identity is not None and identity['argv']==command and identity['cwd']==str(root/'source') and identity['pgid']==child.pid,'Dispatcher actual identity differs')\n"
        "launch={'pid':child.pid,'command':command,'identity':identity};r.write(root/'dispatcher_launch.json',launch,exclusive=True);timed_out=False\n"
        "try:code=child.wait(timeout=max(.001,deadline-time.time()))\n"
        "except subprocess.TimeoutExpired:\n timed_out=True;live=r.proc_identity(child.pid);r.require(live is not None and all(live[k]==identity[k] for k in ('pid','start_ticks','argv','cwd','pgid')),'Dispatcher identity changed; no signal')\n os.killpg(child.pid,signal.SIGTERM)\n try:code=child.wait(timeout=120)\n except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)\n"
        f"p=pathlib.Path({h['root']!r})/'dispatcher_exit.json'\n"
        "p.open('x').write(json.dumps({'pid':child.pid,'start_ticks':identity['start_ticks'],'command':command,'launch_sha256':r.sha_file(root/'dispatcher_launch.json'),'actual_os_returncode':code,'timeout':timed_out,'actual_child_exit_observed':True,'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor_own_exit_directly_observed':False})+chr(10))\nsys.exit(code if not timed_out else 1)\n")
    shell = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(["env", *[f"{k}={v}" for k,v in h["environment"].items()],h["python"], "-u", "-c", supervisor]) + " > " + shlex.quote(h["root"] + "/dispatcher.log") + " 2>&1"
    _transport.save_transport("5090", "launch", _transport.ssh("5090", shlex.join([h["tmux_binary"], "new-session", "-d", "-s", h["tmux"], shell])))
    result = {"status": "launch_requested_not_yet_verified", "at": runtime.now(), "new_fit_count": 9, "host": "5090", "integrity_sha256": integrity_sha}
    write(REPORT / "LAUNCHED_5090.json", result); return result


def status(integrity_sha):
    c = verify_local(integrity_sha); h = c["hosts"]["5090"]
    command = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(["env", *[f"{k}={v}" for k,v in h["environment"].items()],
        h["python"], "-m", MODULE, "--contract", h["root"] + "/execution_contract.json", "--host", "5090", "--mode", "status"])
    return json.loads(_transport.save_transport("5090", "status", _transport.ssh("5090", command)))


def verify_collected(c, originals):
    root = Path(originals); jobs = runtime.validate_contract(c, "5090"); ids = {j["id"] for j in jobs}
    require(sha(root / "execution_contract.json") == sha(BUNDLE / "execution_contract.json"), "Collected contract differs")
    final, exited, observed = read(root / "final_receipt.json"), read(root / "dispatcher_exit.json"), read(root / "collection_observation.json")
    launched, started = read(root / "dispatcher_launch.json"), read(root / "start.json")
    require(exited["pid"] == launched["pid"] == started["pid"] and exited["command"] == launched["command"] == launched["identity"]["argv"]
        and exited["start_ticks"] == launched["identity"]["start_ticks"] == started["identity"]["start_ticks"]
        and exited["launch_sha256"] == sha(root / "dispatcher_launch.json") and exited["timeout"] is False
        and exited["actual_child_exit_observed"] is True, "Dispatcher launch/current/actual-exit identity differs")
    require(final["status"] == "complete" and final["contract_sha256"] == canonical(c)
        and final["completed_fit_count"] == final["validation_replay_count"] == final["test_count"] == 9
        and final["owned_worker_pids_absent"] is True and final["held_out_test_evaluated"] is True
        and exited["actual_os_returncode"] == 0 and observed["owned_pids_absent"] is True
        and observed["historical_campaigns_changed"] is False, "Incomplete MAC9 termination")
    require(c["hosts"]["5090"]["paused_processes"] == observed["paused_processes"] == [], "Unexpected paused predecessor")
    bound = final["artifact_manifest"]; path = runtime.local_path(root, bound["path"]); require(sha(path) == bound["sha256"], "Terminal manifest changed")
    files = read(path)["files"]; require(files and len(files) == bound["file_count"], "Empty terminal inventory")
    for name,digest in files.items(): require(sha(runtime.local_path(root,name)) == digest, "Collected original changed")
    require(set(final["stages"]) == set(runtime.STAGES), "Missing terminal stages")
    for stage in runtime.STAGES:
        value = final["stages"][stage]
        require(value["status"] == "complete" and set(value["exits"]) == set(value["receipts"]) == ids, "Incomplete stage matrix")
        for job in jobs:
            e,b = value["exits"][job["id"]],value["receipts"][job["id"]]
            require(e["actual_os_returncode"] == 0 and e["timeout"] is False and sha(runtime.local_path(root,b["path"])) == b["sha256"], "Actual stage exit/receipt differs")
            runtime.verify_receipt(c,"5090",job,stage,originals=root)
    return final


def collection_program(c):
    h = c["hosts"]["5090"]
    return f'''import io,json,pathlib,sys,tarfile
from paper.scripts import titans_mac_accuracy9_runtime as r
root=pathlib.Path({h['root']!r});c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=={canonical(c)!r},'Contract changed');jobs=r.validate_contract(c,'5090');ids={{j['id'] for j in jobs}}
f=r.read(root/'final_receipt.json');x=r.read(root/'dispatcher_exit.json')
l=r.read(root/'dispatcher_launch.json');s=r.read(root/'start.json')
r.require(x['pid']==l['pid']==s['pid'] and x['command']==l['command']==l['identity']['argv'] and x['start_ticks']==l['identity']['start_ticks']==s['identity']['start_ticks'] and x['launch_sha256']==r.sha_file(root/'dispatcher_launch.json'),'Dispatcher identity differs')
r.require(f['status']=='complete' and f['completed_fit_count']==f['validation_replay_count']==f['test_count']==9 and x['actual_os_returncode']==0 and not x['timeout'],'Terminal counts/exit differ')
pids=[x['pid']]
for stage in r.STAGES:
 s=f['stages'][stage];r.require(s['status']=='complete' and set(s['exits'])==set(s['receipts'])==ids,'Stage incomplete')
 for j in jobs:
  e=s['exits'][j['id']];r.require(e['actual_os_returncode']==0 and not e['timeout'],'Worker failed');pids.append(e['pid']);r.verify_receipt(c,'5090',j,stage)
gpu,paused=r.gpu_state(c,'5090')
r.require(all(r.proc_identity(pid) is None for pid in pids) and not set(pids).intersection(gpu['compute_pids']),'Owner remains')
observation={{'observed_at':r.now(),'owned_pids':pids,'owned_pids_absent':True,'gpu':gpu,'paused_processes':paused,'historical_campaigns_changed':False,'supervisor_own_exit_directly_observed':False}}
blob=(json.dumps(observation,ensure_ascii=False,indent=2)+chr(10)).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root)
  if rel.parts[0] in ('source','cache'):continue
  r.require(not p.is_symlink(),'Unexpected symlink')
  if p.is_file():archive.add(p,arcname=str(rel),recursive=False)
 info=tarfile.TarInfo('collection_observation.json');info.size=len(blob);archive.addfile(info,io.BytesIO(blob))
'''


def collect(c):
    dest = BUNDLE / "retrieval/5090"; dest.mkdir(parents=True,exist_ok=False); h = c["hosts"]["5090"]
    command = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join([h["python"], "-c", collection_program(c)])
    raw = _transport.ssh("5090",command,timeout=900)
    (dest / "original.tar.gz").write_bytes(raw.stdout); _transport.save_transport("5090","collection",raw)
    _transport.safe_extract(dest / "original.tar.gz",dest / "originals"); final = verify_collected(c,dest / "originals")
    files = {str(p.relative_to(dest / "originals")):sha(p) for p in (dest / "originals").rglob("*") if p.is_file()}
    write(dest / "manifest.json",{"files":files})
    result = {"status":"complete","new_fit_count":9,"validation_replay_count":9,"candidate_test_count":9,
        "actual_scientific_worker_exit0_count":27,"actual_dispatcher_exit0_count":1,"archive_sha256":sha(dest / "original.tar.gz"),
        "manifest_sha256":sha(dest / "manifest.json"),"file_count":len(files),"held_out_test_evaluated":True,"prior_campaign_resumed":False}
    write(dest / "receipt.json",result)
    records = [{"dataset":j["dataset"],"model":j["arm"],"seed":j["seed"],"stage":stage,
        "receipt":read(dest / "originals/run" / j["id"] / f"{stage}_receipt.json")} for stage in runtime.STAGES for j in c["jobs"]]
    write(REPORT / "NEW_MAC_RESULTS.json",{"data_variant":"original_uncapped","seeds":[42,52,62],"records":records,"historical_results_merged":False})
    write(BUNDLE / "retrieval/final_receipt.json",{**result,"report_files":{str(REPORT / "NEW_MAC_RESULTS.json"):sha(REPORT / "NEW_MAC_RESULTS.json")}})
    return result


def finish(integrity_sha):
    require(not any(p.exists() for p in (REPORT / "FINISH_STARTED.json", REPORT / "QUEUE_FAILURE.json", REPORT / "QUEUE_COMPLETE.json",
        BUNDLE / "retrieval/5090", BUNDLE / "retrieval/final_receipt.json")), "Existing completion/retrieval/failure; no repeated remote work")
    c = verify_local(integrity_sha); h = c["hosts"]["5090"]
    write(REPORT / "FINISH_STARTED.json",{"pid":os.getpid(),"at":runtime.now(),"automatic_retry":False})
    program = ("import pathlib,time,json,sys\n" + f"root=pathlib.Path({h['root']!r});deadline={h['absolute_host_deadline_unix']!r}\n"
        "while not (root/'dispatcher_exit.json').exists():\n if time.time()>=deadline:raise TimeoutError('Inherited host deadline expired')\n time.sleep(min(30.,max(.01,deadline-time.time())))\n"
        "v=json.loads((root/'dispatcher_exit.json').read_text());print(json.dumps(v))\n"
        "if v['actual_os_returncode']!=0 or v.get('timeout',False):\n"
        " for name in ('dispatcher.log','qualification.stderr','dispatch_failure.json'):\n  p=root/name\n  if p.is_file():print(json.dumps({'failure_source':name,'original_tail':p.read_bytes()[-65536:].decode('utf8','replace')}))\n"
        "sys.exit(0 if v['actual_os_returncode']==0 and not v.get('timeout',False) else 1)\n")
    _transport.save_transport("5090","finite_completion_wait",_transport.ssh("5090",shlex.join([h["python"],"-c",program]),timeout=max(1.,h["absolute_host_deadline_unix"]-time.time()+60)))
    result = collect(c); write(REPORT / "QUEUE_COMPLETE.json",result); return result


def start_finisher(integrity_sha):
    verify_local(integrity_sha,archive=True)
    require((REPORT / "LAUNCHED_5090.json").exists() and not any((REPORT / p).exists() for p in
        ("FINISHER_STARTED.json","FINISH_STARTED.json","QUEUE_FAILURE.json","QUEUE_COMPLETE.json")),"Finite completion already exists/not launched")
    command = [sys.executable,"-u","-m","paper.scripts.manage_titans_mac_accuracy9","--mode","finish","--integrity-sha",integrity_sha]
    with (REPORT / "finish.stdout.log").open("xb") as out,(REPORT / "finish.stderr.log").open("xb") as err:
        child = subprocess.Popen(command,cwd=ROOT,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"},stdout=out,stderr=err,start_new_session=True)
    result = {"pid":child.pid,"command":command,"started_at":runtime.now(),"recurring_automation":False,"own_os_exit_directly_observed":False}
    write(REPORT / "FINISHER_STARTED.json",result); return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--mode",choices=("verify","deploy","status","finish","start-finisher"),required=True)
    parser.add_argument("--integrity-sha",required=True);a=parser.parse_args(argv)
    functions={"verify":lambda:verify_local(a.integrity_sha,archive=True),"deploy":lambda:deploy(a.integrity_sha),"status":lambda:status(a.integrity_sha),
        "finish":lambda:finish(a.integrity_sha),"start-finisher":lambda:start_finisher(a.integrity_sha)}
    try:value=functions[a.mode]()
    except Exception as error:
        if a.mode=="finish" and not (REPORT / "QUEUE_FAILURE.json").exists():write(REPORT / "QUEUE_FAILURE.json",{"status":"failed","at":runtime.now(),"error":str(error),"automatic_retry":False})
        raise
    print(json.dumps(value,ensure_ascii=False,allow_nan=False));return 0


if __name__=="__main__":raise SystemExit(main())
