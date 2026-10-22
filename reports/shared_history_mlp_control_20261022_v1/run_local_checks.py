"""Preserve actual focused test/lint exits, streams and unchanged source hashes."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    tests = [
        p
        for p in sorted(ROOT.glob("paper/tests/test_*history_mlp*.py"))
        if not p.name.startswith("test_titantpp_")
    ]
    sources = (
        [ROOT / "models/TPPs/CountAwareHistoryMLPControl.py"]
        + sorted(ROOT.glob("paper/scripts/*shared_history_mlp_control.py"))
        + [REPORT / "initial_execution.py"]
        + tests
    )
    assert len(tests) >= 8 and all(p.is_file() for p in sources)
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    folder = REPORT / (
        "local_checks_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    )
    folder.mkdir()
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *[str(p.relative_to(ROOT)) for p in tests],
        ],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *[str(p.relative_to(ROOT)) for p in sources],
        ],
    ]
    receipts = []
    for index, command in enumerate(commands):
        stem = folder / ("pytest" if index == 0 else "ruff")
        started = time.monotonic()
        with (
            Path(str(stem) + ".stdout").open("xb") as out,
            Path(str(stem) + ".stderr").open("xb") as err,
        ):
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=out,
                stderr=err,
                timeout=300,
            )
        row = {
            "command": command,
            "actual_os_returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
        }
        for stream in ("stdout", "stderr"):
            path = Path(str(stem) + "." + stream)
            row[stream + "_path"] = str(path)
            row[stream + "_bytes"] = path.stat().st_size
            row[stream + "_sha256"] = sha(path)
        receipts.append(row)
    after = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    value = {
        "actual_os_returncode": 0
        if all(r["actual_os_returncode"] == 0 for r in receipts) and before == after
        else 1,
        "files_checked_sha256": before,
        "source_unchanged_during_tests": before == after,
        "files_after_sha256": after,
        "checks": receipts,
        "remote_calls": 0,
        "real_data_inference": 0,
        "new_training": 0,
    }
    with (folder / "receipt.json").open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "receipt": str(folder / "receipt.json"),
                "actual_os_returncode": value["actual_os_returncode"],
                "checks": [r["actual_os_returncode"] for r in receipts],
            }
        )
    )
    return value["actual_os_returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
