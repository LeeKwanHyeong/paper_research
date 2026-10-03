#!/usr/bin/env python3
"""Owned, deadline-bounded native CUDA qualification for the parallel comparison."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import successor_episode_parallel_common as common
from paper.scripts.verify_successor_episode_cost import verify, cost_gates
from paper.scripts.run_observed_slot_partition import kill_owned_process_group, _supervisor_terminated


def qualify(contract, host_name, started):
    host = contract["hosts"][host_name]
    common.require(not common.gpu_pids(host), "GPU busy; existing work preserved")
    runtime = common.runtime_check(contract, host_name)
    receipt = verify("cuda:0")
    gates = cost_gates(receipt, contract, runtime)
    passed = all(all(row[key] for key in ("step", "relative_memory", "device_memory", "parameters")) for row in gates)
    receipt.update(status="passed" if passed else "cost_gate_failed", runtime=runtime, host=host_name,
        contract_sha256=common.sha_json(contract), cost_gate_checks=gates,
        source_files_sha256=contract["source"]["files_sha256"], started_at_unix=started)
    common.verify_source(contract)
    common.require(common.gpu_pids(host) <= {os.getpid()}, "Another GPU job appeared during qualification")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    contract = common.read_contract(args.contract, args.host)
    common.verify_approval(contract, json.loads(args.approval.read_text()))
    host = contract["hosts"][args.host]
    common.require(ROOT == Path(host["source_root"]).resolve() and Path.cwd().resolve() == ROOT
        and Path(sys.executable).resolve() == Path(host["python"]).resolve(), "Pinned host source/Python required")
    common.apply_environment(host)
    output = Path(host["root"]) / "qualification"
    if args.child:
        common.require(os.environ.get("EPISODE_PARALLEL_QUAL_OWNER") == str(os.getppid()), "Qualification requires its supervising parent")
        started = float(os.environ["EPISODE_PARALLEL_QUAL_STARTED"])
        common.require(started <= time.time() < started + contract["limits"]["qualification_seconds"], "Qualification deadline expired")
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (contract["limits"]["max_file_bytes"],) * 2)
        receipt = qualify(contract, args.host, started)
        common.write_json(output / "receipt.json", receipt, exclusive=True)
        raise SystemExit(0 if receipt["status"] == "passed" else 2)
    common.require(not common.gpu_pids(host), "GPU busy; existing work preserved")
    output.mkdir(exist_ok=False)
    started = time.time()
    common.write_json(output / "start.json", {"host": args.host, "contract_sha256": common.sha_json(contract),
        "started_at_unix": started, "qualification_deadline_unix": started + contract["limits"]["qualification_seconds"]}, exclusive=True)
    environment = {**os.environ, "EPISODE_PARALLEL_QUAL_OWNER": str(os.getpid()), "EPISODE_PARALLEL_QUAL_STARTED": str(started)}
    command = [sys.executable, str(Path(__file__).resolve()), "--contract", str(args.contract.resolve()),
        "--approval", str(args.approval.resolve()), "--host", args.host, "--child"]
    child = None
    previous = signal.signal(signal.SIGTERM, _supervisor_terminated)
    try:
        with (output / "qualification.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            code = child.wait(timeout=contract["limits"]["qualification_seconds"])
            common.require(code == 0, f"Native qualification failed: {code}")
    except BaseException as error:
        if child is not None:
            kill_owned_process_group(child)
        common.write_json(output / "failure.json", {"host": args.host, "status": "qualification_stopped",
            "error": str(error), "automatic_retry": False}, exclusive=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
