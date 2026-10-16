"""Retrieve immutable RTX training originals once; never train or evaluate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(host):
    spec = importlib.util.spec_from_file_location("_immutable_original_helpers", ROOT / "paper/scripts/finalize_titantpp_capacity_test.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    current = read(BUNDLE / "current.json")
    meta = current["host_contracts"][host]
    contract = read(BUNDLE / meta["path"])
    assert helper.canon(contract) == meta["contract_sha256"]
    latest = read(BUNDLE / f"hourly_monitor/latest_{host}.json")
    assert latest["server_terminal_verified"] is True and latest["contract_sha256"] == meta["contract_sha256"]
    assert sha(latest["snapshot"]) == latest["snapshot_sha256"]
    snapshot = read(latest["snapshot"])
    analysis = read(latest["analysis"])
    assert not snapshot["processes"] and not snapshot["compute"]["rows"]
    jobs = [j for j in contract["jobs"] if j["host"] == host]
    assert len(jobs) == 1
    rows = {r["job"]["id"]: r for r in analysis["rows"]}
    proof = {}
    for job in jobs:
        row = rows[job["id"]]
        assert row["terminal_verified"] and row["small_record_SHA_verified"] and row["selected_last_validation_replay_verified"]
        name = "run/" + job["id"] + "/terminal_manifest.json"
        proof[job["id"]] = {"terminal_manifest_sha256": snapshot["file_sha256"][name]}
    state = BUNDLE / "retrieved" / host
    state.mkdir(parents=True, exist_ok=True)
    destination = state / "run"
    receipt_path = state / "retrieval_receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        assert receipt["contract_sha256"] == meta["contract_sha256"] and receipt["status"] == "SHA_verified"
        for job in jobs:
            helper.verify_original(destination / job["id"], job, meta["contract_sha256"], proof[job["id"]])
        print(json.dumps({"host": host, "status": "SHA_verified", "reused": True, "remote_calls": 0}))
        return
    write(state / "retrieval_intent.json", {"host": host, "contract_sha256": meta["contract_sha256"],
        "remote_root": contract["hosts"][host]["root"], "job_ids": [j["id"] for j in jobs],
        "snapshot_sha256": latest["snapshot_sha256"], "starts_training": False,
        "evaluates_test": False, "automatic_retry": False, "time": time.time()}, exclusive=True)
    archive = state / "originals.tar.gz"
    try:
        with archive.open("xb") as output:
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3 -"],
                input=helper.retrieval_script(contract["hosts"][host]["root"], jobs, meta["contract_sha256"], proof).encode(),
                stdout=output, stderr=subprocess.PIPE, timeout=180)
        assert result.returncode == 0, result.stderr.decode(errors="replace")[-2000:]
        assert archive.stat().st_size < helper.MAX_TOTAL
        stage = state / "unpacked"
        stage.mkdir()
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            assert len({m.name for m in members}) == len(members)
            assert sum(m.size for m in members) < helper.MAX_TOTAL
            for entry in members:
                relative = helper.safe_relative(entry.name)
                assert entry.isfile() and not entry.issym() and not entry.islnk()
                assert entry.size < helper.MAX_FILE and relative.parts[0] in {j["id"] for j in jobs}
            stream.extractall(stage, filter="data")
        verified = []
        for job in jobs:
            folder = stage / job["id"]
            manifest = helper.verify_original(folder, job, meta["contract_sha256"], proof[job["id"]])
            assert {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()} == {*manifest["files"], "terminal_manifest.json"}
            verified.append({"job": job, "terminal_manifest_sha256": proof[job["id"]]["terminal_manifest_sha256"],
                "original_file_count": len(manifest["files"]) + 1, "files": manifest["files"]})
        shutil.move(str(stage), destination)
        write(receipt_path, {"status": "SHA_verified", "host": host, "contract_sha256": meta["contract_sha256"],
            "snapshot_sha256": latest["snapshot_sha256"], "archive_sha256": sha(archive),
            "jobs": verified, "original_root": str(destination), "remote_calls": 1,
            "binary_CPU_reinference_audit": "not_performed", "held_out_test_evaluated": False,
            "completed_unix": time.time()}, exclusive=True)
        print(json.dumps({"host": host, "status": "SHA_verified", "jobs": len(verified), "archive_sha256": sha(archive)}))
    except Exception as error:
        write(state / "retrieval_failure.json", {"host": host, "type": type(error).__name__, "message": str(error),
            "automatic_retry": False, "time": time.time()}, exclusive=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=["5080", "5090"], required=True)
    main(parser.parse_args().host)
