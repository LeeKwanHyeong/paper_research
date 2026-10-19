"""One finite follow-up: await the current loss comparison, then raw-time27.

No recurring automation is created. The existing completion process, experiment
roots and stopped external jobs are never modified, resumed or signalled.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_raw_time_mlp27_20261019_v1"
REPORT = ROOT / "reports" / NAME
PARENT_REPORT = ROOT / "reports/titantpp_quantity_loss_mlp18_20261019_v2"
PARENT_BUNDLE = ROOT / "search_artifacts/titantpp_quantity_loss_mlp18_20261019_v2"
READY = REPORT / "LOCAL_READY.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_ready(path=READY):
    ready = read(path)
    require(ready["status"] == "locally_verified_waiting_for_parent"
            and ready["new_fit_count"] == 27 and ready["held_out_test_authorized"] is False,
            "Raw-time preparation scope changed")
    require(bool(ready["files"]), "Empty preparation inventory")
    for name, digest in ready["files"].items():
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT) and sha(target) == digest,
                "Prepared source/evidence changed: " + name)
    require(sha(PARENT_BUNDLE / "execution_contract.json") == ready["parent_contract_sha256"],
            "Running predecessor contract changed")
    return ready


def parent_process_alive(start):
    result = subprocess.run(["ps", "-p", str(start["pid"]), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return result.stdout.split() == start["local_process_identity"].split()


def wait_for_parent(ready, output, *, sleep=time.sleep):
    """Read local completion evidence only; never launch another parent collector."""
    start = read(PARENT_REPORT / "COMPLETION_STARTED.json")
    deadline = min(ready["absolute_host_deadlines"].values())
    completion = PARENT_REPORT / "completion_v1/complete.json"
    write(output / "waiting_parent.json", {"status": "waiting_for_existing_parent_completion",
        "at": now(), "parent_pid": start["pid"], "parent_start_sha256":
        sha(PARENT_REPORT / "COMPLETION_STARTED.json"), "new_training_started": 0,
        "new_remote_calls": 0, "recurring_automation": False})
    while not completion.exists():
        require(time.time() < deadline, "Inherited deadline expired; no new training")
        failures = list((PARENT_REPORT / "completion_v1").glob("*failure.json"))
        require(not failures, "Parent completion failed; preserve evidence and do not retry")
        if not parent_process_alive(start):
            sleep(2.)  # Allow only the terminal-file publication boundary.
            require(completion.exists(), "Existing parent completion exited without success; no retry")
            break
        sleep(min(30., max(.01, deadline - time.time())))
    # The manager verifies final receipts, both archives/manifests, all originals,
    # selected endpoints and the pure 27-record comparison before prepare.
    from paper.scripts import manage_titantpp_raw_time_mlp27 as manager
    references = manager.baseline_validation_references()
    require(len(references) == 27, "Parent paired reference count changed")
    result = read(completion)
    require(result["status"] == "complete" and result["new_fit_count"] == 18
            and result["comparison_model_count"] == 27 and result["held_out_test_evaluated"] is False,
            "Parent completion scope mismatch")
    write(output / "parent_gate.json", {"status": "passed", "at": now(),
        "parent_completion_sha256": sha(completion),
        "parent_final_receipt_sha256": sha(PARENT_BUNDLE / "retrieval/final_receipt.json"),
        "stored_log_time_reference_count": 27, "parent_reexecuted": False,
        "parent_recollected": False, "new_remote_calls": 0})
    return references


def start_wait_program(contract, host, canonical):
    """Wait for the new dispatcher identity; take exactly one status snapshot."""
    target = contract["hosts"][host]
    return f'''import json,pathlib,time
from paper.scripts import titantpp_raw_time_mlp27_runtime as r
root=pathlib.Path({target["root"]!r}); c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=={canonical!r},'Start observation contract changed')
r.validate_contract(c,{host!r})
deadline=min(time.time()+c['limits']['evaluation_wall_seconds'],c['hosts'][{host!r}]['absolute_host_deadline_unix'])
while not (root/'start.json').exists():
 r.require(time.time()<deadline,'New qualification/start observation deadline expired')
 r.require(not any((root/n).exists() for n in ('dispatch_failure.json','wait_failure.json','dispatcher_exit.json')),'New launch failed; preserve original logs, no retry')
 time.sleep(2)
s=r.read(root/'start.json'); identity=r.proc_identity(s['pid'])
r.require(identity is not None,'New dispatcher not observed alive')
snapshot=r.status(c,{host!r})
r.require(snapshot['dispatcher_current_identity']==identity,'Dispatcher identity changed')
print(json.dumps({{'status':'qualified_dispatcher_observed','host':{host!r},'observed_at':r.now(),
 'dispatcher_identity':identity,'snapshot':snapshot,'training_worker_gpu_start_not_asserted':True}}))
'''


def observe_new_dispatcher(manager, contract, host, output):
    target = contract["hosts"][host]
    command = "cd " + shlex.quote(target["source_root"]) + " && " + shlex.join([
        target["python"], "-c", start_wait_program(contract, host, manager.canonical(contract))])
    result = manager.ssh(host, command, timeout=contract["limits"]["evaluation_wall_seconds"] + 30.)
    blob = manager.save_transport(host, "initial_dispatcher_observation", result)
    payload = json.loads(blob)
    require(payload["status"] == "qualified_dispatcher_observed" and payload["host"] == host,
            "New dispatcher observation mismatch")
    write(output / f"{host}.start.json", payload)
    return payload


def completion_adapter():
    """Reuse the checked read-only host wait/collector in a private namespace."""
    path = ROOT / "paper/scripts/finish_titantpp_quantity_loss_mlp18.py"
    require(sha(path) == "97dad83a27fceedb7f73098a6043a32e6ddea2b5dce254a04a5fee5ac1ee31e2",
            "Preserved host completion source changed")
    spec = importlib.util.spec_from_file_location("paper.scripts._raw_time_completion_private", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    old_wait = adapter.wait_program
    def wait_program(*args, **kwargs):
        program = old_wait(*args, **kwargs)
        original = "from paper.scripts import titantpp_quantity_loss_mlp18_runtime as r"
        require(program.count(original) == 1, "Completion runtime adaptation is not unique")
        return program.replace(original, "from paper.scripts import titantpp_raw_time_mlp27_runtime as r", 1)
    adapter.wait_program = wait_program
    return adapter


def complete_new_experiment(manager, contract, starts, output):
    adapter = completion_adapter()
    failures, receipts = [], {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(adapter.finish_host, manager, contract, host, output,
                               starts[host]["dispatcher_identity"]): host for host in ("5080", "5090")}
        for future in as_completed(futures):
            host = futures[future]
            try:
                receipts[host] = future.result()
            except Exception as exc:
                failures.append({"host": host, "exception_type": type(exc).__name__,
                                 "message": str(exc), "at": now()})
    require(not failures, "Raw-time completion failed; no retry: " + json.dumps(failures))
    from paper.scripts.compare_titantpp_raw_time_mlp27 import main as compare
    compare()
    final_path = manager.BUNDLE / "retrieval/final_receipt.json"
    final = read(final_path)
    require(final["status"] == "complete" and final["new_fit_count"] == 27
            and final["baseline_reference_count"] == 27 and final["comparison_model_count"] == 54
            and final["held_out_test_evaluated"] is False, "Raw-time final comparison mismatch")
    for name, digest in {**final["source_bindings"], **final["archive_bindings"], **final["report_files"]}.items():
        require(sha(name) == digest, "Final raw-time evidence changed: " + name)
    write(output / "complete.json", {"status": "complete", "completed_at": now(),
        "final_receipt_path": str(final_path), "final_receipt_sha256": sha(final_path),
        "new_fit_count": 27, "baseline_reference_count": 27, "comparison_model_count": 54,
        "held_out_test_evaluated": False, "recurring_automation": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "advance"), required=True)
    args = parser.parse_args(argv)
    ready = verify_ready()
    if args.mode == "check":
        print(json.dumps({"status": "local_ready_verified", "planned_new_fits": 27,
                          "remote_calls": 0, "new_training": 0}))
        return
    output = REPORT / "followup_v1"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "attempt.json", {"status": "started", "at": now(),
        "ready_sha256": sha(READY), "code_sha256": sha(__file__),
        "automatic_retry": False, "recurring_automation": False})
    try:
        wait_for_parent(ready, output)
        verify_ready()
        from paper.scripts import manage_titantpp_raw_time_mlp27 as manager
        require(not manager.BUNDLE.exists(), "Existing raw-time bundle/attempt; no retry")
        prepared = manager.prepare()
        write(output / "prepared.json", prepared)
        contract = manager.verify_local(archive=True)
        # Independent servers may install/qualify in parallel; scientific
        # dispatchers each enforce their own three-worker limit.
        with ThreadPoolExecutor(max_workers=2) as pool:
            launched = {host: pool.submit(manager.deploy, host) for host in ("5080", "5090")}
            for host, future in launched.items():
                write(output / f"{host}.launch.json", future.result())
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {host: pool.submit(observe_new_dispatcher, manager, contract, host, output)
                       for host in ("5080", "5090")}
            starts = {host: future.result() for host, future in futures.items()}
        write(REPORT / "START_RECEIPT.json", {"status": "qualified_dispatchers_observed",
            "at": now(), "contract_sha256": sha(manager.BUNDLE / "execution_contract.json"),
            "hosts": {host: {"dispatcher_identity": row["dispatcher_identity"],
                             "observation_path": str(output / f"{host}.start.json"),
                             "observation_sha256": sha(output / f"{host}.start.json")}
                      for host, row in starts.items()},
            "new_fit_count": 27, "training_worker_gpu_start_not_asserted": True,
            "held_out_test_evaluated": False})
        complete_new_experiment(manager, contract, starts, output)
    except Exception as exc:
        write(output / "failure.json", {"status": "action_required", "at": now(),
            "exception_type": type(exc).__name__, "message": str(exc),
            "automatic_retry": False, "originals_preserved": True})
        raise


if __name__ == "__main__":
    main()
