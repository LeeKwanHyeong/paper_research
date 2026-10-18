"""One finite completion job for the already-started quantity-loss experiment.

One connection per host waits for its existing dispatcher exit record. It does
not poll GPUs or launch workers. After successful termination, the established
collector reads originals once, then the local comparator runs once. No retry,
resume, recurring automation, signals, or new scientific evaluation is created.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def wait_program(contract, host, canonical, expected_dispatcher):
    target = contract["hosts"][host]
    return f'''import json,pathlib,time
from paper.scripts import titantpp_quantity_loss_mlp18_runtime as r
root=pathlib.Path({target["root"]!r})
c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=={canonical!r},'Completion wait contract mismatch')
r.validate_contract(c,{host!r})
started=time.monotonic()
deadline=c['hosts'][{host!r}]['absolute_host_deadline_unix']
expected={expected_dispatcher!r}
while not (root/'dispatcher_exit.json').exists():
 r.require(time.time()<deadline,'Inherited absolute deadline expired while awaiting completion')
 r.require(not (root/'dispatch_failure.json').exists(),'Dispatcher failure; preserve originals, no retry')
 r.require(not (root/'wait_failure.json').exists(),'Admission failure; preserve originals, no retry')
 r.require((root/'start.json').exists(),'Dispatcher start identity missing')
 start=r.read(root/'start.json'); live=r.proc_identity(start['pid'])
 if live is None:
  # Exit-record creation follows waitpid; permit only that short publication race.
  time.sleep(1)
  r.require((root/'dispatcher_exit.json').exists(),'Dispatcher absent without actual exit record')
  break
 r.require(isinstance(expected,dict) and start['pid']==expected['pid'],'Missing observed dispatcher identity')
 r.require(all(live[k]==expected[k] for k in ('pid','start_ticks','argv','cwd','pgid')),'Dispatcher identity changed')
 time.sleep(min(30,max(.01,deadline-time.time())))
x=r.read(root/'dispatcher_exit.json'); f=r.read(root/'final_receipt.json')
r.require(x['actual_os_returncode']==0,'Dispatcher failed; no collection/retry')
r.require(f['status']=='complete' and f['contract_sha256']==r.sha_json(c),'Incomplete scientific pipeline')
print(json.dumps({{'status':'terminal_ready_for_collection','host':{host!r},'observed_at':r.now(),
 'actual_dispatcher_exit':x['actual_os_returncode'],'wait_elapsed_seconds':time.monotonic()-started,
 'new_training_started':False,'signals_sent':0}}))
'''


def finish_host(manager, contract, host, output, expected_dispatcher):
    target = contract["hosts"][host]
    manager.require(not (manager.BUNDLE / "retrieval" / host).exists(),
                    "Existing retrieval or attempt; preserve and diagnose instead of retrying")
    remaining = float(target["absolute_host_deadline_unix"]) - time.time()
    manager.require(remaining > 0., "Inherited host deadline expired; no automatic retry")
    program = wait_program(contract, host, manager.canonical(contract), expected_dispatcher)
    command = "cd " + shlex.quote(target["source_root"]) + " && " + shlex.join([target["python"], "-c", program])
    stdout, stderr = output / f"{host}.wait.stdout", output / f"{host}.wait.stderr"
    timed_out = False
    with stdout.open("xb") as out, stderr.open("xb") as err:
        try:
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                     "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=2", host, command],
                                    stdout=out, stderr=err, timeout=remaining + 120.)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, None
    write(output / f"{host}.wait.transport.json", {
        "host": host, "observed_at": now(), "actual_transport_exit": returncode, "timeout": timed_out,
        "stdout_bytes": stdout.stat().st_size, "stdout_sha256": manager.sha(stdout),
        "stderr_bytes": stderr.stat().st_size, "stderr_sha256": manager.sha(stderr),
        "automatic_retry": False, "signals_to_experiment": 0})
    manager.require(returncode == 0, "Completion connection failed; originals preserved; no automatic retry")
    payload = json.loads(stdout.read_text())
    manager.require(payload["status"] == "terminal_ready_for_collection" and payload["host"] == host,
                    "Terminal readiness result mismatched")
    receipt = manager.collect(host)
    write(output / f"{host}.complete.json", receipt)
    print(json.dumps({"host": host, "status": "collected_and_verified", "at": now()}), flush=True)
    return receipt


def main():
    from paper.scripts import manage_titantpp_quantity_loss_mlp18 as manager
    from paper.scripts.compare_titantpp_quantity_loss_mlp18 import main as compare
    contract = manager.verify_local(archive=True)
    start_path = manager.REPORT / "START_RECEIPT.json"
    start = manager.read(start_path)
    manager.require(start["contract_sha256"] == manager.sha(manager.BUNDLE / "execution_contract.json")
                    and start["status"] == "started_and_observed", "Missing verified experiment start")
    output = manager.REPORT / "completion_v1"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "attempt.json", {"started_at": now(), "mode": "finite_existing_experiment_completion",
        "contract_sha256": manager.sha(manager.BUNDLE / "execution_contract.json"),
        "start_receipt_sha256": manager.sha(start_path),
        "completion_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_training": 0, "new_test": 0, "automatic_retry": False, "recurring_automation": False})
    failures, receipts = [], {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(finish_host, manager, contract, host, output,
                               start["hosts"][host]["dispatcher_identity"]): host for host in ("5080", "5090")}
        for future in as_completed(futures):
            host = futures[future]
            try:
                receipts[host] = future.result()
            except Exception as exc:
                failures.append({"host": host, "exception_type": type(exc).__name__, "message": str(exc), "at": now()})
                write(output / f"{host}.failure.json", failures[-1])
                print(json.dumps(failures[-1]), flush=True)
    if failures:
        write(output / "failure.json", {"status": "action_required", "failures": failures,
            "completed_hosts": sorted(receipts), "automatic_retry": False})
        raise RuntimeError("Completion did not pass; no automatic retry or partial comparison")
    compare()
    final = manager.read(manager.BUNDLE / "retrieval/final_receipt.json")
    manager.require(final["status"] == "complete" and final["new_fit_count"] == 18
                    and final["comparison_model_count"] == 27 and final["held_out_test_evaluated"] is False,
                    "Final comparison receipt mismatch")
    write(output / "complete.json", {"status": "complete", "completed_at": now(),
        "final_receipt_path": str(manager.BUNDLE / "retrieval/final_receipt.json"),
        "final_receipt_sha256": manager.sha(manager.BUNDLE / "retrieval/final_receipt.json"),
        "new_fit_count": 18, "validation_replay_count": 18, "stored_reference_count": 9,
        "comparison_model_count": 27, "held_out_test_evaluated": False, "recurring_automation": False})


if __name__ == "__main__":
    main()
