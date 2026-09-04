#!/usr/bin/env python3
"""Approved candidate-only seed42 e300, gated by completed CUDA/full e1 evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import run_hard_lmm_key_value_smoke as smoke
from paper.scripts.run_hard_lmm_weighted_static import compare

CONTRACT = ROOT / "paper/contracts/hard_lmm_key_value_e300_5090_v1.json"
EXECUTION_ROLE = "key_value_static_e300_5090"


def training_files(manifest):
    prefixes = ("models/", "data_loader/", "utils/", "simple_lab_test/common/",
                "simple_lab_test/search/common/", "paper/scripts/count_aware_tpp_backbone/")
    extra = {"paper/scripts/run_count_aware_tpp_backbone_control.py",
             "paper/scripts/run_taxi_quantity_interface_ablation.py"}
    return {p: sha for p, sha in manifest["files"].items() if p.startswith(prefixes) or p in extra}


def verify_smoke(spec, rows, references):
    root = Path(spec["smoke_snapshot"])
    artifact = root / spec["smoke_artifact"]
    require = smoke.require
    require(smoke.digest(root / "source_manifest.json") == spec["smoke_source_manifest_sha256"], "Smoke manifest changed")
    old_manifest = smoke.read(root / "source_manifest.json")
    require(old_manifest["source_revision"] == spec["smoke_source_revision"], "Smoke revision mismatch")
    require(training_files(old_manifest) == training_files(smoke.read(ROOT / "source_manifest.json")),
            "Training source differs from CUDA/e1 validated source")
    for path, sha in old_manifest["files"].items():
        require(smoke.digest(root / path) == sha, f"Smoke source changed: {path}")
    status = smoke.read(artifact / "status.json")
    require(status["status"] == "complete" and status["cuda_contract_tests"] == "passed" and
            status["source_revision"] == spec["smoke_source_revision"] and
            status["source_manifest_sha256"] == spec["smoke_source_manifest_sha256"], "Smoke incomplete/changed")
    require([r["dataset"] for r in status["completed"]] == list(smoke.DATASETS), "Missing full e1 run")
    xml = ET.parse(artifact / "cuda_contract_tests.xml")
    cases = xml.findall(".//testcase")
    require(len(cases) == 29 and not xml.findall(".//failure") and not xml.findall(".//error") and
            not xml.findall(".//skipped"), "CUDA test report incomplete")
    audits = [smoke.audit_run(artifact / row["dataset"], row, references[row["dataset"]],
                             spec["smoke_source_revision"], write_audit=False) for row in rows]
    for audit, recorded in zip(audits, status["completed"]):
        for key in ("summary_sha256", "history_sha256", "checkpoint_file_sha256", "checkpoint_state_sha256"):
            require(audit[key] == recorded[key], f"Smoke evidence changed: {key}")
    return {"status": "passed", "source_manifest_sha256": smoke.digest(root / "source_manifest.json"),
            "smoke_status_sha256": smoke.digest(artifact / "status.json"),
            "cuda_report_sha256": smoke.digest(artifact / "cuda_contract_tests.xml"), "audits": audits}


def command(row, project, output, revision):
    cmd = smoke.command(row, project, output, revision)
    for flag, value in (("--epochs", "300"), ("--min-epochs", "40"),
                        ("--early-stopping-patience", "40"), ("--execution-role", EXECUTION_ROLE)):
        cmd[cmd.index(flag) + 1] = value
    return cmd


def execute(args):
    args.output_root.mkdir(parents=True, exist_ok=False)
    status_path = args.output_root / "status.json"
    spec = smoke.read(CONTRACT)
    status = {"status": "running", "phase": "candidate_only_seed42_e300", "source_revision": args.source_revision,
              "frozen_model_revision": spec["frozen_model_revision"], "started_at": datetime.now(timezone.utc).isoformat(),
              "completed": [], "current_dataset": None, "held_out_test_evaluated": False}
    smoke.save(status_path, status)
    try:
        import torch
        import polars
        status["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "cuda": torch.version.cuda, "polars": polars.__version__}
        smoke.require(status["runtime"] == spec["runtime"], "Runtime differs from CUDA/e1 validation")
        status["source_manifest_sha256"] = smoke.verify_source(args.source_revision)
        manifest = smoke.read(ROOT / "source_manifest.json")
        smoke.require(manifest.get("performance_training_authorized") is True and
                      manifest["scope"] == spec["scope"], "Screening source package not authorized")
        _, rows = smoke.frozen_documents()
        smoke.require([r["dataset"] for r in rows] == spec["dataset_order"], "Dataset scope changed")
        references = {r["dataset"]: smoke.baseline(r, args.project_root) for r in rows}
        status["smoke_gate"] = verify_smoke(spec, rows, references)
        smoke.save(args.output_root / "execution_contract.json", spec)
        smoke.save(args.output_root / "source_manifest.json", smoke.read(ROOT / "source_manifest.json"))
        comparisons = {}
        for row in rows:
            name = row["dataset"]
            status["current_dataset"] = name
            status["preflight"] = smoke.preflight(12000)
            smoke.verify_source(args.source_revision)
            smoke.baseline(row, args.project_root)
            cmd = command(row, args.project_root, args.output_root / name, args.source_revision)
            status.update(command=cmd, current_started_at=datetime.now(timezone.utc).isoformat())
            smoke.save(status_path, status)
            start = time.monotonic()
            smoke.run_logged(cmd, args.output_root / f"{name}.log")
            audit = smoke.audit_run(args.output_root / name, row, references[name], args.source_revision, screening=True)
            candidate = smoke.read(args.output_root / name / smoke.SUMMARY)
            comparisons[name] = compare(references[name][1], candidate, spec["per_dataset_gate"])
            smoke.save(args.output_root / "comparison.json", comparisons)
            status["completed"].append({**audit, "elapsed_seconds": time.monotonic() - start,
                                        "gate_passed": comparisons[name]["passed"]})
            smoke.save(status_path, status)
        status["postflight"] = smoke.preflight(12000)
        status.update(status="complete", current_dataset=None, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=datetime.now(timezone.utc).isoformat())
        smoke.save(status_path, status)
        raise
    smoke.save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    execute(parser.parse_args())
