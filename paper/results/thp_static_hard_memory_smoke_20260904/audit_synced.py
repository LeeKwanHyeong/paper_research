"""Reconcile the synced e1 evidence; no real-data forward or optimizer step."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import run_thp_static_hard_memory_smoke as smoke

ARTIFACT = ROOT / "search_artifacts/count_aware_thp_static_memory_cuda_e1_20260904"
REVISION = "093629b96455568f702fc651146664eb176ae13c"


def main():
    source = smoke.read(ARTIFACT / "source_manifest.json")
    assert source["source_revision"] == REVISION
    for name, expected in source["files"].items():
        git_path = f"{REVISION}:{name}"
        if name == source["reference_file"]:
            git_path = f"{source['reference_revision']}:models/TPPs/CountAwareFactory.py"
        content = subprocess.check_output(["git", "show", git_path], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected, name
    evidence = smoke.read(HERE / "count_aware_thp_static_memory_cuda_e1_20260904.final_checksums.json")
    assert evidence["source_files_verified"] == len(source["files"])
    assert set(evidence["files"]) == {str(p.relative_to(ARTIFACT)) for p in ARTIFACT.rglob("*") if p.is_file()}
    for name, expected in evidence["files"].items():
        assert smoke.digest(ARTIFACT / name) == expected, name
    launcher = (HERE / "count_aware_thp_static_memory_cuda_e1_20260904.launcher.log").read_text()
    assert launcher.count("RUN ") == 3 and "Traceback" not in launcher
    status = smoke.read(ARTIFACT / "status.json")
    assert status["status"] == "complete" and len(status["completed"]) == 2
    assert status["performance_training_authorized"] is False
    suite = ET.parse(ARTIFACT / "cuda_contract_tests.xml").getroot().find("testsuite")
    assert {k: int(suite.get(k)) for k in ("tests", "failures", "errors", "skipped")} == {
        "tests": 35, "failures": 0, "errors": 0, "skipped": 0}
    contract, registry = smoke.frozen_documents()
    assert smoke.read(ARTIFACT / "frozen_contract.json") == contract
    assert smoke.read(ARTIFACT / "baseline_registry.json") == registry
    results = []
    for row in registry["datasets"]:
        output = ARTIFACT / row["dataset"]
        log = (ARTIFACT / (row["dataset"] + ".log")).read_text()
        assert "[complete]" in log and not any(token in log for token in ("Traceback", "OutOfMemoryError", "Dynamo error"))
        reference = smoke.baseline(row, ROOT)
        result = smoke.audit_run(output, row, reference, REVISION, write_audit=False)
        assert result == smoke.read(output / "audit.json")
        summary = smoke.read(output / smoke.SUMMARY)
        for path in output.rglob("*.json"):
            smoke.finite(smoke.read(path))
        for path in output.glob("*.csv"):
            with path.open() as stream:
                for record in csv.DictReader(stream):
                    for value in record.values():
                        try:
                            number = float(value)
                        except (ValueError, TypeError):
                            continue
                        assert math.isfinite(number), (path, value)
        for axis in ("quantity", "history"):
            with (output / f"{axis}_seed_metrics.csv").open() as stream:
                records = list(csv.DictReader(stream))
            assert len(records) == len(summary[f"{axis}_rows"])
            for record, metric in zip(records, summary[f"{axis}_rows"]):
                assert record["stratum"] == metric["stratum"] and int(record["count"]) == metric["count"]
                for key in ("qty_mae", "qty_rmse", "time_nll", "joint_objective"):
                    assert math.isclose(float(record[key]), metric[key], rel_tol=1e-12, abs_tol=1e-10)
        results.append(result | {"elapsed_seconds": summary["elapsed_seconds"],
            "metrics_e1_not_performance_acceptance": {key: summary[f"best_val_{key}"]
                for key in ("qty_mae", "qty_rmse", "time_nll", "joint_objective")}})
    assert evidence["cuda_processes"]["returncode"] == 0 and evidence["cuda_processes"]["stdout"] == ""
    assert evidence["gdm"]["stdout"] == "inactive" and evidence["kernel"]["returncode"] == 0
    assert not any(s in evidence["kernel"]["stdout"].lower() for s in ("nvrm: xid", "out of memory", "oom-kill"))
    report = {"status": "cuda_and_full_e1_passed", "audited_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": REVISION, "source_files_verified": len(source["files"]),
        "artifact_files_verified": len(evidence["files"]), "cuda_tests_passed": 35,
        "started_at": status["started_at"], "completed_at": status["completed_at"],
        "results": results, "held_out_test_evaluated": False, "performance_acceptance": "not_evaluated",
        "actual_runtime": {"python": "3.12.13", "torch": "2.11.0+cu130", "cuda": "13.0",
                           "polars": "1.39.3", "pytest": "9.0.3", "driver": "595.84"},
        "limitations": ["Memory-efficient attention backward warns of non-determinism; no bitwise training replay claim.",
                        "Checkpoint replay is synthetic CUDA in contracts and exact CPU best/last replay after real e1.",
                        "Full e1 is feasibility only, not a benchmark ranking or screening acceptance.",
                        "Peak training VRAM was not instrumented; pre/postflight snapshots do not measure peak.",
                        "No plots generated by this e1 runner; JSON histories and quantity/history CSVs reconciled.",
                        "Initial root-sentinel bootstrap failure preceded model creation/training; logs preserved."],
        "next": "Separate approval for fresh Taxi/RAF seed42 e300 screening; do not reuse e1 weights."}
    smoke.save(HERE / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
