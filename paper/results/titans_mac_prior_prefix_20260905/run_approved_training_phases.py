#!/usr/bin/env python3
"""Run the already approved, finite e1/screening/conditional-confirm sequence.

This controller stays outside the immutable GPU source. It neither trains a
model itself nor changes the source, runtime, service configuration or gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


REVISION = "add02baae27c25da02008873c578f11926f8d125"
MANIFEST_SHA256 = "62ad41b97ea73cfab82b00f0f93ad2689cb50ac4e2a6066bd4b6a469f5d20d3c"
CONTRACT_SHA256 = "81f874eaf7568c0daf262d2cf49371c425a43352789ba227124d6d0b2152ebf7"
LAUNCHER_SHA256 = "c1ff96fbb07098969d879feb378705e690a3011354d63f20d8c29d1030ca7416"
LAUNCHER = "paper/scripts/run_titans_mac_prior_prefix_5090.py"
CONTRACT = "paper/contracts/titans_mac_prior_prefix_v1.json"
PHASES = ("e1", "screening", "confirm")
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket")
BACKBONES = ("titantpp_titans_mac", "titantpp_titans_mac_prior_prefix")
PYTHON = "/opt/miniconda3/envs/ai_env/bin/python"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def save(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_frozen_launcher(args):
    require(args.source_revision == REVISION, "Only the approved add02ba source is allowed")
    root = args.source_root
    require(not Path(__file__).resolve().is_relative_to(root), "Controller must stay outside frozen source")
    require(digest(root / "source_manifest.json") == MANIFEST_SHA256, "Frozen source manifest changed")
    require(digest(root / CONTRACT) == CONTRACT_SHA256, "Frozen training contract changed")
    require(digest(root / LAUNCHER) == LAUNCHER_SHA256, "Frozen phase launcher changed")
    spec = importlib.util.spec_from_file_location("approved_prior_prefix_launcher", root / LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    # These imported helpers use only stdlib. Torch is imported inside the
    # frozen launcher's execution functions, which run in a separate process.
    contract = read(root / CONTRACT)
    launcher.validate_contract(contract)
    require(launcher.verify_source(root / "source_manifest.json", REVISION, root / CONTRACT,
                                   root=root) == MANIFEST_SHA256, "Source archive verification failed")
    require(Path(sys.executable).resolve() == Path(PYTHON).resolve(), "Use the approved existing 5090 Python")
    return launcher


def phase_command(args, phase):
    require(phase in PHASES, "Unknown approved phase")
    command = [PYTHON, "-s", "-B", str(args.source_root / LAUNCHER), "--phase", phase,
               "--source-revision", REVISION, "--source-manifest", str(args.source_root / "source_manifest.json"),
               "--contract", str(args.source_root / CONTRACT), "--cuda-proof", str(args.cuda_proof),
               "--data-project", str(args.data_project), "--output-root", str(args.output_root / phase)]
    if phase in ("screening", "confirm"):
        command.extend(["--e1-proof", str(args.output_root / "e1/status.json")])
    if phase == "confirm":
        command.extend(["--screening-proof", str(args.output_root / "screening/status.json")])
    return command


def verify_evidence(files, allowed_roots):
    require(isinstance(files, dict) and bool(files), "Phase lacks immutable evidence hashes")
    for name, expected in files.items():
        path = Path(name)
        require(path.is_absolute() and any(path.resolve().is_relative_to(root) for root in allowed_roots),
                f"Evidence is outside the approved artifact roots: {name}")
        require(path.is_file() and not path.is_symlink() and digest(path) == expected,
                f"Evidence changed or is missing: {name}")


def verify_phase(path, phase, args, launcher):
    """Accept completed analysis with a false performance result, but no errors."""
    proof = read(path)
    require(proof.get("status") == "PASS" and proof.get("phase") == phase, f"{phase} execution did not pass")
    require(proof.get("source_revision") == REVISION
            and proof.get("source_manifest_sha256") == MANIFEST_SHA256
            and proof.get("contract_sha256") == CONTRACT_SHA256, "Phase source or contract mismatch")
    require(proof.get("held_out_test_evaluated") is False, "Held-out test scope changed")
    require(proof.get("run_root") == str(Path(path).resolve().parent), "Phase artifact root mismatch")
    phase_root = Path(path).resolve().parent
    verify_evidence(proof.get("evidence_files"), [phase_root])
    seeds = (52, 62) if phase == "confirm" else (42,)
    expected = {(d, b, s) for d in DATASETS for b in BACKBONES for s in seeds}
    rows = proof.get("runs", [])
    require(len(rows) == len(expected) and proof.get("completed_run_count") == len(expected)
            and {(row.get("dataset"), row.get("backbone"), row.get("seed")) for row in rows} == expected
            and all(row.get("status") == "PASS" and row.get("held_out_test_evaluated") is False for row in rows),
            "Incomplete, duplicate or failed paired phase runs")
    if phase == "e1":
        launcher.verify_proof(path, phase, REVISION, CONTRACT_SHA256, MANIFEST_SHA256)
        require(proof.get("performance_acceptance") == "not_evaluated_e1_only", "e1 cannot decide performance")
        return proof, None
    audit = proof.get("audit", {})
    require(audit.get("status") == "PASS" and audit.get("phase") == phase
            and audit.get("source_revision") == REVISION and audit.get("contract_sha256") == CONTRACT_SHA256
            and audit.get("held_out_test_evaluated") is False and audit.get("run_root") == str(phase_root),
            "Comparison analysis or identity did not pass")
    verify_evidence(audit.get("evidence_files"), [args.source_root, args.output_root])
    key = "screening_pass" if phase == "screening" else "confirmation_pass"
    accepted = audit.get(key)
    require(type(accepted) is bool and proof.get("performance_acceptance") is accepted,
            "Performance decision is missing or inconsistent")
    if phase == "screening" and accepted:
        launcher.verify_proof(path, phase, REVISION, CONTRACT_SHA256, MANIFEST_SHA256)
    if phase == "confirm":
        require(audit.get("screening_pass") is True and audit.get("screening_run_root")
                == str(args.output_root / "screening"), "Confirmation did not preserve the approved seed42 evidence")
    return proof, accepted


def stop_child(child):
    """TERM the phase group; its frozen handler terminates the training group."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=10)
        return
    try:
        child.wait(timeout=40)  # frozen launcher allows 20s TERM + 10s KILL
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def execute(args):
    for name in ("source_root", "cuda_proof", "data_project", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    require(not args.output_root.exists(), "Fresh controller output required; retry/resume/overwrite are disabled")
    require(not args.output_root.is_relative_to(args.data_project), "Controller output must be outside the data project")
    if args.output_root.is_relative_to(args.source_root):
        relative = args.output_root.relative_to(args.source_root)
        require(len(relative.parts) >= 2 and relative.parts[0] == "outputs",
                "Within the snapshot, only a fresh outputs subdirectory is allowed")
    launcher = load_frozen_launcher(args)
    launcher.verify_proof(args.cuda_proof, "cuda", REVISION, CONTRACT_SHA256, MANIFEST_SHA256)
    args.output_root.mkdir(parents=True, exist_ok=False)
    status_path = args.output_root / "status.json"
    state = {"status": "RUNNING", "pipeline": "approved_e1_screening_conditional_confirm_once",
             "source_root": str(args.source_root), "source_revision": REVISION,
             "source_manifest_sha256": MANIFEST_SHA256, "contract_sha256": CONTRACT_SHA256,
             "controller_sha256": digest(__file__), "run_root": str(args.output_root),
             "held_out_test_evaluated": False, "active_phase": None, "child_pid": None,
             "completed_proofs": {"cuda": {"path": str(args.cuda_proof), "sha256": digest(args.cuda_proof)}},
             "final_performance_status": "not_evaluated", "started_at": datetime.now(timezone.utc).isoformat()}
    child = None
    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        save(status_path, state)
        print(json.dumps({k: state[k] for k in ("status", "active_phase", "child_pid", "final_performance_status")}), flush=True)
    def interrupted(signum, frame):
        raise InterruptedError(f"Controller received signal {signum}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    def revalidate():
        require(launcher.verify_source(args.source_root / "source_manifest.json", REVISION,
                                       args.source_root / CONTRACT, root=args.source_root) == MANIFEST_SHA256,
                "Frozen manifest changed after controller startup")
        for phase, receipt in state["completed_proofs"].items():
            path = Path(receipt["path"])
            require(digest(path) == receipt["sha256"], f"Completed {phase} proof changed")
            if phase == "cuda":
                launcher.verify_proof(path, phase, REVISION, CONTRACT_SHA256, MANIFEST_SHA256)
            else:
                verify_phase(path, phase, args, launcher)
    try:
        update()
        for phase in PHASES:
            revalidate()
            require(not (args.output_root / phase).exists(), "Phase output exists; resume/retry is forbidden")
            command = phase_command(args, phase)
            log = args.output_root / f"{phase}.log"
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=args.source_root, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                update(active_phase=phase, child_pid=child.pid, command=command, log_path=str(log))
                code = child.wait()
                child = None
            require(code == 0, f"{phase} exited {code}; automatic retry is disabled")
            path = args.output_root / phase / "status.json"
            _, accepted = verify_phase(path, phase, args, launcher)
            state["completed_proofs"][phase] = {"path": str(path), "sha256": digest(path),
                                                "performance_acceptance": accepted}
            update(child_pid=None)
            revalidate()
            if phase == "screening" and not accepted:
                update(status="PASS", active_phase=None, final_performance_status="screening_failed",
                       stopped_reason="Screening completed normally but did not pass; confirm was not started")
                return state
            if phase == "confirm":
                update(status="PASS", active_phase=None,
                       final_performance_status="confirmed" if accepted else "confirmation_failed")
                return state
    except BaseException as exc:
        # Ignore a repeated terminal signal while the child launcher performs
        # its own bounded cleanup of the separately grouped training process.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop_child(child)
        finally:
            update(status="FAIL", child_pid=None, error=f"{type(exc).__name__}: {exc}",
                   final_performance_status="execution_error")
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--cuda-proof", type=Path, required=True)
    parser.add_argument("--data-project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        print(json.dumps(execute(parse_args()), indent=2, sort_keys=True))
    except (Exception, KeyboardInterrupt) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
