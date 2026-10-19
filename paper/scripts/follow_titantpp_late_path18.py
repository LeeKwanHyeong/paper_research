"""One finite continuation: finish prior comparisons, select once, then late-path18."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_mlp16_late_path18_20261019_v1"
REPORT = ROOT / "reports" / NAME
READY = REPORT / "LOCAL_READY.json"
PARENT_REPORT = ROOT / "reports/titantpp_raw_time_mlp27_20261019_v1"
PARENT_READY_SHA = "59ee0a1b1e2a2a261093a7766cf49a49861922c5814b99c5219da1eda0b03937"
PYTHON_EXECUTABLES = {
    "/usr/local/bin/python3",
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
    "/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
}


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
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_ready(path=READY):
    ready = read(path)
    require(ready["status"] == "locally_verified_waiting_for_raw_time27"
            and ready["new_fit_count"] == 18 and ready["baseline_reference_count"] == 9
            and ready["held_out_test_authorized"] is False, "Late-path scope changed")
    require(ready["files"], "Empty new preparation bindings")
    for name, digest in ready["files"].items():
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT) and sha(target) == digest,
                "Prepared evidence/source changed: " + name)
    require(sha(PARENT_REPORT / "LOCAL_READY.json") == PARENT_READY_SHA,
            "Existing raw-time preparation changed")
    require(sha(PARENT_REPORT / "FOLLOWUP_STARTED.json") == ready["parent_followup_start_sha256"],
            "Existing finite parent process binding changed")
    return ready


def same_local_process_identity(start, actual_identity):
    """Accept only the known Python launcher/runtime transition, same start/argv."""
    expected, actual = start["local_process_identity"].split(), actual_identity.split()
    if len(expected) < 9 or len(actual) != len(expected):
        return False
    return (actual[:5] == expected[:5] and actual[5] in PYTHON_EXECUTABLES
            and expected[5] in PYTHON_EXECUTABLES and actual[6:] == expected[6:]
            and actual[6:] == start["command"][1:])


def parent_process_alive(start):
    result = subprocess.run(["ps", "-p", str(start["pid"]), "-o", "lstart=", "-o", "command="],
                            capture_output=True, text=True)
    return result.returncode == 0 and same_local_process_identity(start, result.stdout)


def wait_for_parent(ready, output, *, sleep=time.sleep):
    """Wait on local terminal evidence, never start/recollect/change the parent."""
    start = read(PARENT_REPORT / "FOLLOWUP_STARTED.json")
    require(start["command"] == ["/usr/local/bin/python3", "-u", "-m",
        "paper.scripts.follow_titantpp_raw_time_mlp27", "--mode", "advance"], "Unexpected parent command")
    deadline = min(ready["absolute_host_deadlines"].values())
    completion = PARENT_REPORT / "followup_v1/complete.json"
    write(output / "waiting_parent.json", {"status": "waiting_for_both_existing_comparisons",
        "at": now(), "parent_pid": start["pid"], "parent_start_sha256": sha(PARENT_REPORT / "FOLLOWUP_STARTED.json"),
        "new_remote_calls": 0, "new_training_started": 0, "recurring_automation": False})
    while not completion.exists():
        require(time.time() < deadline, "Inherited deadline expired; no new work")
        require(not list((PARENT_REPORT / "followup_v1").glob("*failure.json")),
                "Existing raw-time follow-up failed; no automatic retry")
        if not parent_process_alive(start):
            sleep(2.)
            require(completion.exists(), "Parent exited without successful completion; no retry")
            break
        sleep(min(30., max(.01, deadline - time.time())))
    require(time.time() < deadline, "Inherited deadline expired; no new training")
    from paper.scripts import manage_titantpp_late_path18 as manager
    verified = manager.verify_parent_completion()
    write(output / "parent_gate.json", {"status": "passed", "at": now(),
        "parent_completion_sha256": sha(completion), "verified_parent_endpoint_count": len(verified[2]),
        "selected_condition": verified[3]["selected_condition"], "parent_reexecuted": False,
        "parent_recollected": False, "new_remote_calls": 0})


def start_adapter():
    path = ROOT / "paper/scripts/follow_titantpp_raw_time_mlp27.py"
    require(sha(path) == "8b2b4206c4a035456ff1be3a5dff5073e9d037b88109a70c0c9edf55f8e6892a",
            "Preserved finite start observer changed")
    spec = importlib.util.spec_from_file_location("paper.scripts._late18_start_private", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    original = adapter.start_wait_program
    def program(*args, **kwargs):
        source = original(*args, **kwargs)
        needle = "from paper.scripts import titantpp_raw_time_mlp27_runtime as r"
        require(source.count(needle) == 1, "Unexpected initial-observer adaptation")
        return source.replace(needle, "from paper.scripts import titantpp_late_path18_runtime as r", 1)
    adapter.start_wait_program = program
    return adapter


def completion_adapter():
    adapter = start_adapter().completion_adapter()
    # That private adapter yields the raw-time runtime. Adapt only the final
    # emitted program; parent source and function globals remain untouched.
    original = adapter.wait_program
    def program(*args, **kwargs):
        source = original(*args, **kwargs)
        needle = "from paper.scripts import titantpp_raw_time_mlp27_runtime as r"
        require(source.count(needle) == 1, "Unexpected completion-wait adaptation")
        return source.replace(needle, "from paper.scripts import titantpp_late_path18_runtime as r", 1)
    adapter.wait_program = program
    return adapter


def complete_new_experiment(manager, contract, starts, output):
    adapter = completion_adapter()
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(adapter.finish_host, manager, contract, host, output,
                               starts[host]["dispatcher_identity"]): host for host in ("5080", "5090")}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append({"host": futures[future], "exception_type": type(error).__name__,
                                 "message": str(error), "at": now()})
    require(not failures, "Late-path completion failed; no retry: " + json.dumps(failures))
    from paper.scripts.select_compare_titantpp_late_path18 import main as compare
    compare()
    fp = manager.BUNDLE / "retrieval/final_receipt.json"
    final = read(fp)
    expected = {"new_fit_count": 18, "validation_replay_count": 18, "baseline_reference_count": 9,
                "comparison_model_count": 27, "actual_scientific_child_exit0_count": 36,
                "actual_dispatcher_exit0_count": 2}
    require(final["status"] == "complete" and final["held_out_test_evaluated"] is False
            and all(final.get(k) == v for k, v in expected.items()), "New final counts/scope mismatch")
    for group in ("source_bindings", "archive_bindings", "report_files"):
        require(final[group], "Empty final evidence")
        for name, digest in final[group].items():
            require(sha(name) == digest, "Final new evidence changed: " + name)
    write(output / "complete.json", {"status": "complete", "at": now(),
        "final_receipt_path": str(fp), "final_receipt_sha256": sha(fp), **expected,
        "held_out_test_evaluated": False, "recurring_automation": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "advance"), required=True)
    args = parser.parse_args(argv)
    ready = verify_ready()
    if args.mode == "check":
        print(json.dumps({"status": "local_ready_verified", "planned_new_fits": 18,
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
        from paper.scripts import manage_titantpp_late_path18 as manager
        prepared = manager.prepare()
        write(output / "prepared.json", prepared)
        contract = manager.verify_local(archive=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {host: pool.submit(manager.deploy, host) for host in ("5080", "5090")}
            for host, future in futures.items():
                write(output / f"{host}.launch.json", future.result())
        observer = start_adapter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {host: pool.submit(observer.observe_new_dispatcher, manager, contract, host, output)
                       for host in ("5080", "5090")}
            starts = {host: future.result() for host, future in futures.items()}
        write(REPORT / "START_RECEIPT.json", {"status": "qualified_dispatchers_observed", "at": now(),
            "contract_sha256": sha(manager.BUNDLE / "execution_contract.json"),
            "common_condition": contract["common_condition"], "new_fit_count": 18,
            "hosts": {host: {"dispatcher_identity": row["dispatcher_identity"],
                                "observation_path": str(output / f"{host}.start.json"),
                                "observation_sha256": sha(output / f"{host}.start.json")}
                      for host, row in starts.items()},
            "training_worker_gpu_start_not_asserted": True, "held_out_test_evaluated": False})
        complete_new_experiment(manager, contract, starts, output)
    except Exception as error:
        write(output / "failure.json", {"status": "action_required", "at": now(),
            "exception_type": type(error).__name__, "message": str(error),
            "automatic_retry": False, "originals_preserved": True})
        raise


if __name__ == "__main__":
    main()
