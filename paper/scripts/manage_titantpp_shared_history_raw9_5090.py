"""Run raw shared9 on freshly idle5090 without waiting for Cap5080."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from paper.scripts import titantpp_shared_history_raw9_5090_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_history_mlp_shared_raw9_20261020_v1"
BUNDLE = ROOT / "search_artifacts" / NAME / "execution_5090_v1"
REPORT = ROOT / "reports" / NAME / "execution_5090_v1"
MODULE = "paper.scripts.titantpp_shared_history_raw9_5090_runtime"
_PATH = Path(__file__).with_name("manage_titantpp_shared_history_raw9.py")
_SHA = "d387fd00d0ebd1b026e7524577e1add45c1ddf85482f44d85db878650a91a8c4"
runtime.require(runtime.sha_file(_PATH) == _SHA, "Frozen raw5080 manager changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_raw", _PATH)
_raw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_raw)
read, sha, canonical, require, now = runtime.read, runtime.sha_file, runtime.sha_json, runtime.require, runtime.now
write = _raw.write


def verify_local(*, archive=False, require_predecessor_complete=False):
    preparation = importlib.import_module("paper.scripts.prepare_titantpp_shared_history_raw9_5090")
    contract = preparation.verify(archive=archive)
    runtime.validate_contract(contract, "5090")
    return contract


def fresh_preflight(contract):
    """One read-only current GPU inventory; final admission is checked natively again."""
    h = contract["hosts"]["5090"]
    program = ("import pathlib,subprocess,json\n"
        f"root=pathlib.Path({h['root']!r});expected={h['gpu_uuid']!r}\n"
        "assert not root.exists(),'New5090 root already exists'\n"
        "gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=20)\n"
        "assert gpu.stdout.strip()==expected,'GPU UUID mismatch'\n"
        "apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)\n"
        "assert not apps.stdout.strip(),'5090 has an active compute process'\n"
        "print(json.dumps({'status':'passed','host':'5090','gpu_uuid':expected,'compute_pids':[],'new_root_absent':True,'cap5080_dependency':False}))\n")
    result = json.loads(_raw._impl.save_transport("5090", "fresh_preflight", _raw._impl.ssh(
        "5090", shlex.join([h["python"], "-c", program]))))
    require(result.get("status") == "passed" and result.get("host") == "5090"
        and result.get("compute_pids") == [] and result.get("gpu_uuid") == h["gpu_uuid"], "Fresh5090 preflight failed")
    write(REPORT / "FRESH_PREFLIGHT.json", result)
    return result


_raw.ROOT, _raw.BUNDLE, _raw.REPORT, _raw.MODULE, _raw.runtime = ROOT, BUNDLE, REPORT, MODULE, runtime
_raw.verify_local, _raw.fresh_preflight = verify_local, fresh_preflight
_raw._impl.ROOT, _raw._impl.BUNDLE, _raw._impl.REPORT = ROOT, BUNDLE, REPORT
_raw._impl.MODULE, _raw._impl.runtime = MODULE, runtime
_raw._impl.verify_local = verify_local

# Only the host and comparison entrypoint change. The scientific worker,
# immutable collection checks and finite-wait/no-repeat guards stay identical.
for _function in ("deploy", "finish"):
    _source = inspect.getsource(getattr(_raw, _function))
    require('"5080"' in _source, "Unexpected frozen host adaptation")
    _source = _source.replace("5080", "5090")
    if _function == "finish":
        require(_source.count('"paper.scripts.compare_titantpp_shared_history_raw9"') == 1, "Unexpected comparison adaptation")
        _source = _source.replace('"paper.scripts.compare_titantpp_shared_history_raw9"',
            '"paper.scripts.compare_titantpp_shared_history_raw9_5090"')
    exec(compile(_source, str(_PATH), "exec"), _raw.__dict__)

_collection = inspect.getsource(_raw.collection_program)
require(_collection.count("from paper.scripts import titantpp_shared_history_raw9_runtime as r") == 1, "Unexpected frozen collector adaptation")
_collection = _collection.replace("from paper.scripts import titantpp_shared_history_raw9_runtime as r",
    "from paper.scripts import titantpp_shared_history_raw9_5090_runtime as r")
exec(compile(_collection, str(_PATH), "exec"), _raw.__dict__)
_raw._impl.collection_program = _raw.collection_program
_raw._impl.verify_collected = _raw.verify_collected
deploy, finish = _raw.deploy, _raw.finish
collection_program, verify_collected = _raw.collection_program, _raw.verify_collected
status, collect, safe_extract = _raw._impl.status, _raw._impl.collect, _raw._impl.safe_extract


def start_finisher():
    verify_local(archive=True)
    require(not any((REPORT / name).exists() for name in ("FINISHER_STARTED.json", "FINISH_STARTED.json", "QUEUE_FAILURE.json", "QUEUE_COMPLETE.json")), "5090 completion already started or failed")
    command = [sys.executable, "-u", "-m", "paper.scripts.manage_titantpp_shared_history_raw9_5090", "--mode", "finish"]
    with (REPORT / "finish.stdout.log").open("xb") as out, (REPORT / "finish.stderr.log").open("xb") as err:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out, stderr=err, start_new_session=True)
    result = {"pid": child.pid, "command": command, "started_at": now(), "recurring_automation": False,
        "own_os_exit_directly_observed": False, "purpose": "One raw9_5090 terminal wait, collection and comparison"}
    write(REPORT / "FINISHER_STARTED.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify", "deploy", "status", "collect", "finish", "start-finisher"), required=True)
    args = parser.parse_args(argv)
    functions = {"verify": verify_local, "deploy": deploy, "status": lambda: status("5090"),
        "collect": lambda: collect("5090"), "finish": finish, "start-finisher": start_finisher}
    try:
        value = functions[args.mode]()
    except Exception as error:
        if args.mode == "finish" and not (REPORT / "QUEUE_FAILURE.json").exists():
            write(REPORT / "QUEUE_FAILURE.json", {"status": "failed", "at": now(), "error": str(error), "automatic_retry": False})
        raise
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
