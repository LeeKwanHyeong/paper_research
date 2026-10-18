"""One authorized SIGSTOP per host, preserving the running Cap27 campaign."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import manage_titantpp_external_cap27 as manager

REPORT = ROOT / "reports/titantpp_time_quantity_balance_20261018_v1"


def program(contract, host, expected):
    h = contract["hosts"][host]
    # The parent dispatcher is frozen before discovering children, including
    # a child forked just before its launch record could be persisted.
    return f'''import json,os,pathlib,signal,time
from datetime import datetime,timezone
from paper.scripts import titantpp_external_cap27_runtime as r
root=pathlib.Path({h["root"]!r})
c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=={manager.canonical(contract)!r},'Contract identity changed')
jobs=r.validate_contract(c,{host!r})
expected={expected!r}
evidence=root.parent/('titantpp_external_cap27_pause_20261018_v1_'+{host!r})
evidence.mkdir(exist_ok=False)
def state(pid):
    raw=pathlib.Path('/proc',str(pid),'stat').read_text()
    tail=raw[raw.rfind(')')+2:].split()
    return {{'state':tail[0],'ppid':int(tail[1]),'pgid':int(tail[2])}}
def identity(pid):
    ident=r.proc_identity(pid)
    if ident is not None:
        ident.update(state(pid))
    return ident
def same(a,b):
    return a is not None and all(a.get(k)==b.get(k) for k in ('pid','start_ticks','argv','cwd','pgid'))
def await_stopped(before):
    until=time.monotonic()+5
    while time.monotonic()<until:
        current=identity(before['pid'])
        r.require(same(current,before),'Process identity disappeared/changed during pause')
        if current['state'] in ('T','t'): return current
        time.sleep(.02)
    raise RuntimeError('SIGSTOP did not reach stopped state')
intent={{'at':datetime.now(timezone.utc).isoformat(),'host':{host!r},'expected_dispatcher':expected,
        'authorized_action':'SIGSTOP_only','automatic_resume':False}}
r.write(evidence/'intent.json',intent,exclusive=True)
dispatcher=identity(expected['pid'])
r.require(same(dispatcher,expected),'Dispatcher identity changed; signal refused')
os.kill(dispatcher['pid'],signal.SIGSTOP)
dispatcher_after=await_stopped(dispatcher)
allowed={{tuple(r.worker_command(c,{host!r},root/'execution_contract.json',j,stage)):(j['id'],stage)
         for j in jobs for stage in r.STAGES}}
workers=[]
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try: ident=identity(int(entry.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError): continue
    if ident is None or tuple(ident['argv']) not in allowed: continue
    r.require(ident['cwd']==str(root/'source') and ident['ppid']==dispatcher['pid']
              and ident['pgid']==ident['pid'],'Worker ownership differs; dispatcher remains paused')
    job,stage=allowed[tuple(ident['argv'])]
    if ident['state']=='Z':
        workers.append({{'job_id':job,'stage':stage,'already_exited_unreaped':True,'identity':ident}})
        continue
    # A process group is exclusive to this worker (new session at launch).
    r.require(same(identity(ident['pid']),ident),'Worker identity changed before signal')
    os.killpg(ident['pgid'],signal.SIGSTOP)
    workers.append({{'job_id':job,'stage':stage,'before':ident,'after':await_stopped(ident)}})
known={{x.get('after',x.get('identity'))['pid'] for x in workers}}
child_ids=set()
for task in pathlib.Path('/proc',str(dispatcher['pid']),'task').iterdir():
    child_ids.update(int(x) for x in (task/'children').read_text().split())
for pid in child_ids:
    child=identity(pid)
    r.require(child is None or child['state']=='Z' or
              (pid in known and child['state'] in ('T','t')),
              'Unidentified/running direct child; pause completeness unknown')
status=r.read(root/'status.json')
gpu=r.gpu_snapshot(c['hosts'][{host!r}]['gpu_uuid'])
result={{'status':'paused','observed_at':datetime.now(timezone.utc).isoformat(),'host':{host!r},
        'campaign_root':str(root),'contract_sha256':r.sha_file(root/'execution_contract.json'),
        'dispatcher':dispatcher_after,'workers':workers,'campaign_status_at_pause':status,
        'gpu':gpu,'automatic_resume':False,'cuda_contexts_and_vram_retained':True,
        'wall_clock_deadlines_continue':True,'checkpoint_or_source_modified':False}}
r.write(evidence/'receipt.json',result,exclusive=True)
print(json.dumps(result,ensure_ascii=False))
'''


def pause(host):
    contract = manager.verify_local()
    obs = sorted((manager.REPORT / "observations").glob(f"*_{host}.json"))[-1]
    snapshot = manager.read(obs)
    expected = snapshot.get("dispatcher_current_identity")
    manager.require(expected is not None, "No verified live dispatcher; do not signal")
    h = contract["hosts"][host]
    source = program(contract, host, expected)
    (REPORT / f"PAUSE_PROGRAM_{host}.py").write_text(source)
    words = ["env", *[f"{k}={v}" for k, v in h["environment"].items()], h["python"], "-"]
    command = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(words)
    response = manager.ssh(host, command, stdin=source.encode(), timeout=45)
    prefix = REPORT / f"PAUSE_TRANSPORT_{host}"
    for suffix, data in ((".stdout", response.stdout), (".stderr", response.stderr)):
        with prefix.with_suffix(suffix).open("xb") as out:
            out.write(data)
    manager.write(prefix.with_suffix(".json"), {
        "actual_transport_exit": response.returncode,
        "transport_timeout": getattr(response, "timed_out", False),
        "stdout_sha256": manager.sha(prefix.with_suffix(".stdout")),
        "stderr_sha256": manager.sha(prefix.with_suffix(".stderr")),
        "pre_pause_observation": str(obs), "pre_pause_observation_sha256": manager.sha(obs),
    })
    manager.require(response.returncode == 0, "Pause failed; inspect preserved transport; no automatic retry")
    receipt = json.loads(response.stdout)
    manager.write(REPORT / f"PAUSE_RECEIPT_{host}.json", receipt)
    return {"host": host, "status": receipt["status"], "observed_at": receipt["observed_at"],
            "paused_worker_count": sum("after" in x for x in receipt["workers"]),
            "jobs": [x["job_id"] for x in receipt["workers"]],
            "completed_at_pause": len(receipt["campaign_status_at_pause"].get("actual_worker_exits", {})),
            "pending_at_pause": len(receipt["campaign_status_at_pause"].get("pending_job_ids", []))}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(pause, ("5080", "5090")):
            print(json.dumps(result, ensure_ascii=False))
