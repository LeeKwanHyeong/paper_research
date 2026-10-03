#!/usr/bin/env python3
"""Contract-bound, owned native CUDA qualification with a 15-minute ceiling."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import nonlinear_episode_parallel_common as common
from paper.scripts.verify_nonlinear_episode_cost import (
    OWNER_ENV, STARTED_ENV, cost_gates, require_native_owner, verify,
)
from paper.scripts.run_observed_slot_partition import kill_owned_process_group, _supervisor_terminated


def qualify(contract, host_name, started):
    """Called by the owned child after contract, approval, host and source checks."""
    now = time.time()
    ceiling = contract["limits"]["qualification_seconds"]
    common.require(type(started) in (int, float) and math.isfinite(started)
                   and 0 < started <= now < started + ceiling,
                   "Qualification start/deadline is invalid or expired")
    common.verify_source(contract)
    host = contract["hosts"][host_name]
    common.require(not common.gpu_pids(host), "GPU busy; existing work preserved")
    runtime = common.runtime_check(contract, host_name)
    receipt = verify("cuda:0")
    gates = cost_gates(receipt, contract, runtime)
    passed = all(all(row[key] for key in ("step", "relative_memory", "device_memory", "parameters"))
                 for row in gates)
    common.verify_source(contract)
    common.require(common.gpu_pids(host) <= {os.getpid()}, "Another GPU job appeared during qualification")
    completed = time.time()
    common.require(started <= completed <= started + ceiling,
                   "Qualification completed after its deadline")
    receipt.update(
        status="passed" if passed else "cost_gate_failed", runtime=runtime, host=host_name,
        contract_sha256=common.sha_json(contract), cost_gate_checks=gates,
        source_files_sha256=contract["source"]["files_sha256"],
        started_at_unix=started, completed_at_unix=completed,
    )
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
                   and Path(sys.executable).resolve() == Path(host["python"]).resolve(),
                   "Pinned host source/Python required")
    common.apply_environment(host)
    output = Path(host["root"]) / "qualification"
    ceiling = contract["limits"]["qualification_seconds"]
    common.require(ceiling == 900, "Native qualification is limited to 15 minutes")
    if args.child:
        started = require_native_owner()
        start_receipt = json.loads((output / "start.json").read_text())
        common.require(start_receipt.get("host") == args.host
                       and start_receipt.get("contract_sha256") == common.sha_json(contract)
                       and start_receipt.get("started_at_unix") == started
                       and start_receipt.get("qualification_deadline_unix") == started + ceiling
                       and start_receipt.get("supervisor_pid") == os.getppid(),
                       "Qualification start receipt differs from its owning supervisor")
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (contract["limits"]["max_file_bytes"],) * 2)
        receipt = qualify(contract, args.host, started)
        common.write_json(output / "receipt.json", receipt, exclusive=True)
        raise SystemExit(0 if receipt["status"] == "passed" else 2)
    common.require(not common.gpu_pids(host), "GPU busy; existing work preserved")
    output.mkdir(exist_ok=False)
    started = time.time()
    common.write_json(output / "start.json", {
        "host": args.host, "contract_sha256": common.sha_json(contract),
        "supervisor_pid": os.getpid(), "started_at_unix": started,
        "qualification_deadline_unix": started + ceiling,
    }, exclusive=True)
    environment = {**os.environ, OWNER_ENV: str(os.getpid()), STARTED_ENV: str(started)}
    command = [sys.executable, str(Path(__file__).resolve()), "--contract", str(args.contract.resolve()),
               "--approval", str(args.approval.resolve()), "--host", args.host, "--child"]
    child = None
    previous = signal.signal(signal.SIGTERM, _supervisor_terminated)
    try:
        with (output / "qualification.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=environment, start_new_session=True)
            remaining = started + ceiling - time.time()
            common.require(remaining > 0, "Qualification deadline expired before child wait")
            code = child.wait(timeout=remaining)
            common.require(code == 0, f"Native qualification failed: {code}")
    except BaseException as error:
        if child is not None:
            kill_owned_process_group(child)
        common.write_json(output / "failure.json", {
            "host": args.host, "status": "qualification_stopped", "error": str(error),
            "contract_sha256": common.sha_json(contract), "automatic_retry": False,
        }, exclusive=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
