"""Reproduce new-directory startup importer caching using local subprocesses only."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROGRAM = """import importlib,json,pathlib,sys
source=pathlib.Path(sys.argv[1]);invalidate=sys.argv[2]=='invalidate'
before=repr(sys.path_importer_cache.get(str(source),'missing'))
(source/'fresh_probe_package').mkdir(parents=True)
(source/'fresh_probe_package/__init__.py').write_text('VALUE=42\\n')
sys.path.insert(0,str(source))
if invalidate:importlib.invalidate_caches()
try:
 import fresh_probe_package
 print(json.dumps({'status':'imported','value':fresh_probe_package.VALUE,'startup_cache':before,'invalidation':invalidate}),flush=True)
except ModuleNotFoundError:
 print(json.dumps({'status':'module_not_found','startup_cache':before,'invalidation':invalidate}),flush=True)
 raise
"""


def main():
    results = {}
    for mode in ("without_invalidation", "with_invalidation"):
        folder = ROOT / mode
        folder.mkdir(exist_ok=False)
        source = folder / "new_source"
        command = [
            sys.executable,
            "-u",
            "-c",
            PROGRAM,
            str(source),
            "invalidate" if mode == "with_invalidation" else "preserve",
        ]
        result = subprocess.run(
            command,
            cwd=folder,
            env={
                **os.environ,
                "PYTHONPATH": str(source),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            timeout=15,
        )
        for name, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
            (folder / name).write_bytes(raw)
        receipt = {
            "command": command,
            "actual_os_returncode": result.returncode,
            "timeout": False,
            "remote_calls": 0,
            **{
                name + "_bytes": len(raw)
                for name, raw in (("stdout", result.stdout), ("stderr", result.stderr))
            },
            **{
                name + "_sha256": hashlib.sha256(raw).hexdigest()
                for name, raw in (("stdout", result.stdout), ("stderr", result.stderr))
            },
        }
        (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        results[mode] = {
            "actual_os_returncode": result.returncode,
            "stdout": result.stdout.decode(),
            "stderr": result.stderr.decode(),
        }
    if (
        results["without_invalidation"]["actual_os_returncode"] != 1
        or results["with_invalidation"]["actual_os_returncode"] != 0
    ):
        raise RuntimeError(
            "Expected importer cache failure/recovery was not reproduced"
        )
    original = ROOT.parent / "initial_execution_v1/5080/deploy"
    evidence = {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in original.iterdir()
        if p.is_file()
    }
    (ROOT / "RESULT.json").write_text(
        json.dumps(
            {
                "status": "REPRODUCED",
                "remote_calls": 0,
                "diagnosis": "Python startup cached a nonexistent PYTHONPATH as an invalid finder; a later sys.path.insert alone leaves the cached None entry. importlib.invalidate_caches removes it.",
                "original_deployment_actual_exit": 1,
                "connection_failure_exit255_observed": False,
                "original_evidence": evidence,
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
