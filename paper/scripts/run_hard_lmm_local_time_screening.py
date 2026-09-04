#!/usr/bin/env python3
"""Approved Taxi/RAF fresh screening against an immutable, smoke-tested model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# The child must import every training module from the frozen tree, not this tree.
if __name__ != "__main__" or sys.argv[1:2] != ["child"]:
    sys.path.insert(0, str(ROOT))
    from paper.scripts import run_hard_lmm_local_time_smoke as smoke

TRAINING_REVISION = "b9d0ac0b5052102d32a7bce57bff294d75576bdb"
TRAINING_MANIFEST_SHA256 = "54b913e67cdf7042e20754fb9a20ebf369ae7cef7965c825db6cade20b6b31bd"
SMOKE_REPORT = ROOT / "paper/results/hard_lmm_local_time_20260903/smoke_verification.json"
SMOKE_REPORT_SHA256 = "f261d79c7a8cc881c4ec099adaa1d41ea9146b2aa94df442d43282f676fa1173"


def verify_training(root):
    path = root / "source_manifest.json"
    smoke.require(smoke.digest(path) == TRAINING_MANIFEST_SHA256, "Frozen training manifest changed")
    manifest = smoke.read(path)
    smoke.require(manifest["source_revision"] == TRAINING_REVISION, "Wrong training revision")
    for name, expected in manifest["files"].items():
        smoke.require(smoke.digest(root / name) == expected, f"Frozen source changed: {name}")


def command(row, project, output, training_root):
    smoke.require(not output.exists(), "Fresh dataset artifact required; no resume")
    args = smoke.command(row, project, output, TRAINING_REVISION)
    args[3] = str(training_root / "paper/scripts/run_count_aware_tpp_backbone_control.py")
    for key, old, new in (("epochs", "1", "300"), ("min-epochs", "1", "40"),
                          ("early-stopping-patience", "1", "40"),
                          ("execution-role", "hard_local_time_smoke_5080", "hard_local_time_screening_5080")):
        index = args.index("--" + key) + 1
        smoke.require(args[index] == old, f"Unexpected source command: {key}")
        args[index] = new
    return args


def compare(reference, candidate, gate):
    def metrics(summary):
        smoke.finite(summary)
        rows = summary["quantity_rows"]
        body = [r for r in rows if r["stratum"] in {"le_p50", "p50_p90", "p90_p95"}]
        tail = [r for r in rows if r["stratum"] == "gt_p99"]
        smoke.require({r["stratum"] for r in body} == {"le_p50", "p50_p90", "p90_p95"}
                      and len(body) == 3 and len(tail) == 1, "Missing/duplicate body or tail strata")
        smoke.require(all(r["count"] > 0 for r in body + tail), "Empty evaluation stratum")
        return {"body_mae": sum(r["count"] * r["qty_mae"] for r in body) / sum(r["count"] for r in body),
                "qty_mae": summary["best_val_qty_mae"], "qty_rmse": summary["best_val_qty_rmse"],
                "gt_p99_mae": tail[0]["qty_mae"], "time_nll": summary["best_val_time_nll"]}
    try:
        before, after = metrics(reference), metrics(candidate)
        smoke.require(all(before[k] > 0 for k in ("body_mae", "qty_mae", "qty_rmse", "gt_p99_mae")),
                      "Zero/negative reference metric")
        relative = {key: after[key] / before[key] - 1 for key in ("body_mae", "qty_mae", "qty_rmse", "gt_p99_mae")}
        time_delta = after["time_nll"] - before["time_nll"]
        checks = {"body": after["body_mae"] <= before["body_mae"] * (1 - gate["body_le_p95_mae_improvement_min"]),
                  "rmse": after["qty_rmse"] <= before["qty_rmse"] * (1 + gate["overall_rmse_regression_max"]),
                  "tail": after["gt_p99_mae"] <= before["gt_p99_mae"] * (1 + gate["gt_p99_mae_regression_max"]),
                  "time": time_delta <= gate["time_nll_absolute_increase_max"]}
        return {"status": "passed" if all(checks.values()) else "rejected", "baseline": before,
                "candidate": after, "relative_changes": relative, "time_nll_delta": time_delta, "checks": checks}
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        return {"status": "not_evaluable", "reason": str(exc)}


def child(entrypoint, report, arguments):
    """Read peak counters after the unchanged trainer; no RNG/gradient changes."""
    started = time.monotonic()
    record = {"status": "failed", "entrypoint": str(entrypoint)}
    try:
        sys.path[0] = str(entrypoint.resolve().parents[2])
        sys.argv = [str(entrypoint), *arguments]
        try:
            runpy.run_path(str(entrypoint), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        record["status"] = "complete"
    finally:
        record["elapsed_seconds_including_startup"] = time.monotonic() - started
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report.parent.mkdir(parents=True, exist_ok=True)
        temporary = report.with_suffix(report.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(report)


def run_logged(cmd, output, name):
    report = output / f"{name}_resources.json"
    wrapped = [sys.executable, "-s", "-u", str(Path(__file__).resolve()), "child",
               "--entrypoint", cmd[3], "--report", str(report), "--", *cmd[4:]]
    epoch_records = []
    started = previous = time.monotonic()
    with (output / f"{name}.log").open("w") as log:
        with subprocess.Popen(wrapped, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("[epoch "):
                    now = time.monotonic()
                    epoch_records.append({"epoch": int(line.split("]", 1)[0].split()[1]),
                                          "observed_seconds": now - previous,
                                          "elapsed_seconds": now - started,
                                          "observed_at": datetime.now(timezone.utc).isoformat()})
                    previous = now
                    smoke.save(output / f"{name}_epoch_timing.json", epoch_records)
            if process.wait() != 0:
                raise RuntimeError(f"Training child failed: {name}; see log")
    resources = smoke.read(report)
    smoke.require(resources["status"] == "complete" and resources.get("peak_allocated_bytes", 0) > 0,
                  "Missing CUDA resource telemetry")


def readiness(args):
    smoke.require(smoke.digest(smoke.CONTRACT) == smoke.CONTRACT_SHA256, "Contract changed")
    smoke.require(smoke.digest(SMOKE_REPORT) == SMOKE_REPORT_SHA256, "Smoke report changed")
    smoke.verify_source(args.orchestration_revision)
    verify_training(args.training_root)
    report, contract = smoke.read(SMOKE_REPORT), smoke.read(smoke.CONTRACT)
    smoke.require(report["status"] == "passed" and report["source_revision"] == TRAINING_REVISION,
                  "Unverified model smoke")
    import torch
    smoke.require(platform.python_version() == report["runtime"]["server_python"] and
                  torch.__version__ == report["runtime"]["server_torch"] and
                  torch.version.cuda == report["runtime"]["server_cuda"], "Smoke-tested runtime changed")
    registry = ROOT / contract["baseline"]["registry"]
    smoke.require(smoke.digest(registry) == contract["baseline"]["registry_sha256"], "Registry changed")
    rows = {r["dataset"]: r for r in smoke.read(registry)["datasets"]}
    specs = {r["dataset"]: r for r in contract["datasets"]}
    references = {}
    for name in smoke.DATASETS:
        references[name] = smoke.baseline(rows[name], args.project_root)[1]
        evidence = report["dataset_runs"][name]
        output = args.project_root / evidence["artifact"]
        smoke.require(smoke.digest(output / smoke.SUMMARY) == evidence["summary_sha256"], "Smoke summary changed")
        smoke.require(smoke.digest((output / smoke.SUMMARY).parent / "best_val_joint_objective_model.pt") ==
                      evidence["checkpoint_file_sha256"], "Smoke checkpoint changed")
        # Re-audit, but never load smoke weights into a training command.
        smoke.require(smoke.digest((output / smoke.SUMMARY).parent / "history.json") ==
                      evidence["history_sha256"], "Smoke history changed")
        smoke.audit_run(output, rows[name], references[name], TRAINING_REVISION, specs[name], write_audit=False)
    return contract, rows, specs, references


def execute(args):
    args.output_root.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "phase": "fresh_e300_screening", "training_revision": TRAINING_REVISION,
              "orchestration_revision": args.orchestration_revision, "started_at": datetime.now(timezone.utc).isoformat(),
              "current_dataset": None, "completed": [], "held_out_test_evaluated": False,
              "authorization": "2026-09-04 user approved Taxi/RAF seed42 only", "automatic_resume": False}
    path = args.output_root / "status.json"
    smoke.save(path, status)
    try:
        contract, rows, specs, references = readiness(args)
        smoke.save(args.output_root / "frozen_contract.json", contract)
        smoke.save(args.output_root / "baseline_registry.json", [rows[name] for name in smoke.DATASETS])
        smoke.save(args.output_root / "training_source_manifest.json", smoke.read(args.training_root / "source_manifest.json"))
        smoke.save(args.output_root / "orchestration_source_manifest.json", smoke.read(ROOT / "source_manifest.json"))
        decisions = {}
        for name in smoke.DATASETS:
            status["current_dataset"] = name
            status["preflight"] = smoke.preflight(12000)
            verify_training(args.training_root)
            smoke.verify_source(args.orchestration_revision)
            cmd = command(rows[name], args.project_root, args.output_root / name, args.training_root)
            status["command"] = cmd
            smoke.save(path, status)
            run_logged(cmd, args.output_root, name)
            status["postflight"] = smoke.preflight(12000)
            audit = smoke.audit_run(args.output_root / name, rows[name], references[name],
                                    TRAINING_REVISION, specs[name], screening=True)
            summary = smoke.read(args.output_root / name / smoke.SUMMARY)
            decisions[name] = compare(references[name], summary, contract["performance_gate"])
            smoke.save(args.output_root / "comparison.json", decisions)
            status["completed"].append(audit)
            smoke.save(path, status)
        status.update(status="complete", current_dataset=None, completed_at=datetime.now(timezone.utc).isoformat(),
                      candidate_decision="eligible_for_separate_followup" if all(
                          d["status"] == "passed" for d in decisions.values()) else "hold")
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=datetime.now(timezone.utc).isoformat())
        smoke.save(path, status)
        raise
    smoke.save(path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run")
    for name in ("training-root", "project-root", "output-root"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--orchestration-revision", required=True)
    wrapped = modes.add_parser("child")
    wrapped.add_argument("--entrypoint", type=Path, required=True)
    wrapped.add_argument("--report", type=Path, required=True)
    wrapped.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "child":
        child(args.entrypoint, args.report, args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    else:
        execute(args)


if __name__ == "__main__":
    main()
