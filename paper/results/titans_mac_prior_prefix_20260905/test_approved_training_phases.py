"""Finite pipeline tests use fake phase processes and no model/GPU runtime."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).with_name("run_approved_training_phases.py")
SPEC = importlib.util.spec_from_file_location("approved_training_phases", MODULE_PATH)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class FakeLauncher:
    def verify_source(self, *args, **kwargs):
        return controller.MANIFEST_SHA256

    def verify_proof(self, path, phase, revision, contract, manifest):
        proof = controller.read(path)
        assert proof["status"] == "PASS" and proof["phase"] == phase
        assert (revision, contract, manifest) == (controller.REVISION, controller.CONTRACT_SHA256,
                                                  controller.MANIFEST_SHA256)
        return proof


@pytest.fixture
def args(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cuda = source / "outputs/cuda/status.json"
    cuda.parent.mkdir(parents=True)
    controller.save(cuda, {"status": "PASS", "phase": "cuda"})
    return argparse.Namespace(source_root=source, source_revision=controller.REVISION,
                              cuda_proof=cuda, data_project=tmp_path / "data",
                              output_root=source / "outputs/full_validation")


def write_proof(args, phase, accepted=True):
    root = args.output_root / phase
    root.mkdir(parents=True)
    evidence = root / "evidence.json"
    controller.save(evidence, {"complete": True})
    seeds = (52, 62) if phase == "confirm" else (42,)
    proof = {"status": "PASS", "phase": phase, "source_revision": controller.REVISION,
             "source_manifest_sha256": controller.MANIFEST_SHA256,
             "contract_sha256": controller.CONTRACT_SHA256,
             "held_out_test_evaluated": False, "run_root": str(root),
             "evidence_files": {str(evidence): controller.digest(evidence)},
             "runs": [{"dataset": d, "backbone": b, "seed": s, "status": "PASS",
                       "held_out_test_evaluated": False} for d in controller.DATASETS
                      for b in controller.BACKBONES for s in seeds]}
    proof["completed_run_count"] = len(proof["runs"])
    if phase == "e1":
        proof["performance_acceptance"] = "not_evaluated_e1_only"
    else:
        proof["performance_acceptance"] = accepted
        proof["audit"] = {"status": "PASS", "phase": phase, "source_revision": controller.REVISION,
                          "contract_sha256": controller.CONTRACT_SHA256, "held_out_test_evaluated": False,
                          "run_root": str(root), "evidence_files": proof["evidence_files"],
                          "screening_pass": True if phase == "confirm" else accepted}
        if phase == "confirm":
            proof["audit"].update(confirmation_pass=accepted,
                                  screening_run_root=str(args.output_root / "screening"))
    path = root / "status.json"
    controller.save(path, proof)
    return path, proof


def fake_processes(monkeypatch, args, *, decisions=None, failed_phase=None, mutation=None):
    calls = []
    monkeypatch.setattr(controller, "load_frozen_launcher", lambda _args: FakeLauncher())
    class Process:
        def __init__(self, command, **kwargs):
            self.phase = command[command.index("--phase") + 1]
            calls.append(self.phase)
            self.pid = 900000 + len(calls)
            self.returncode = None
            assert kwargs["start_new_session"] is True
            assert kwargs["cwd"] == args.source_root
            if self.phase != failed_phase:
                write_proof(args, self.phase, (decisions or {}).get(self.phase, True))
        def wait(self, **kwargs):
            if mutation:
                mutation(self.phase)
            self.returncode = 1 if self.phase == failed_phase else 0
            return self.returncode
        def poll(self):
            return self.returncode
    monkeypatch.setattr(controller.subprocess, "Popen", Process)
    return calls


@pytest.mark.parametrize("screening,confirm,phases,result", [
    (False, True, ["e1", "screening"], "screening_failed"),
    (True, False, ["e1", "screening", "confirm"], "confirmation_failed"),
    (True, True, ["e1", "screening", "confirm"], "confirmed"),
])
def test_finite_pipeline_expands_only_after_screening_pass(args, monkeypatch, screening, confirm, phases, result):
    calls = fake_processes(monkeypatch, args, decisions={"screening": screening, "confirm": confirm})
    state = controller.execute(args)
    assert calls == phases
    assert state["status"] == "PASS" and state["final_performance_status"] == result
    assert state["active_phase"] is None and state["child_pid"] is None
    assert set(state["completed_proofs"]) == {"cuda", *phases}
    for receipt in state["completed_proofs"].values():
        assert receipt["sha256"] == controller.digest(receipt["path"])
    if not screening:
        assert not (args.output_root / "confirm").exists()


@pytest.mark.parametrize("failed_phase,phases", [("e1", ["e1"]), ("screening", ["e1", "screening"]),
                                                ("confirm", ["e1", "screening", "confirm"])])
def test_execution_failure_stops_without_retry(args, monkeypatch, failed_phase, phases):
    calls = fake_processes(monkeypatch, args, failed_phase=failed_phase)
    with pytest.raises(ValueError, match="automatic retry"):
        controller.execute(args)
    state = controller.read(args.output_root / "status.json")
    assert calls == phases and state["status"] == "FAIL"
    assert state["final_performance_status"] == "execution_error"
    with pytest.raises(ValueError, match="Fresh controller"):
        controller.execute(args)
    assert calls == phases


def test_completed_proof_mutation_prevents_next_phase(args, monkeypatch):
    def mutation(phase):
        if phase == "e1":
            args.cuda_proof.write_text('{"status":"PASS","phase":"cuda","changed":true}')
    calls = fake_processes(monkeypatch, args, mutation=mutation)
    with pytest.raises(ValueError, match="Completed cuda proof changed"):
        controller.execute(args)
    assert calls == ["e1"]


@pytest.mark.parametrize("phase", controller.PHASES)
def test_phase_command_has_fixed_source_and_required_predecessor_proofs(args, phase):
    cmd = controller.phase_command(args, phase)
    assert cmd[:3] == [controller.PYTHON, "-s", "-B"]
    assert cmd[3] == str(args.source_root / controller.LAUNCHER)
    assert cmd[cmd.index("--source-revision") + 1] == controller.REVISION
    assert cmd[cmd.index("--output-root") + 1] == str(args.output_root / phase)
    assert ("--e1-proof" in cmd) == (phase != "e1")
    assert ("--screening-proof" in cmd) == (phase == "confirm")
    assert not set(cmd) & {"--resume", "--force-rerun", "--retry", "--max-train-batches"}


@pytest.mark.parametrize("key,bad", [("source_revision", "f" * 40), ("contract_sha256", "f" * 64),
                                    ("source_manifest_sha256", "f" * 64), ("held_out_test_evaluated", True),
                                    ("status", "FAIL"), ("run_root", "/wrong"), ("completed_run_count", 5)])
def test_phase_proof_rejects_identity_scope_and_count_drift(args, key, bad):
    path, proof = write_proof(args, "screening")
    controller.save(path, {**proof, key: bad})
    with pytest.raises(ValueError):
        controller.verify_phase(path, "screening", args, FakeLauncher())


def test_phase_proof_rejects_duplicate_runs_and_evidence_tamper(args):
    path, proof = write_proof(args, "screening")
    bad = copy.deepcopy(proof)
    bad["runs"][-1] = bad["runs"][0]
    controller.save(path, bad)
    with pytest.raises(ValueError, match="Incomplete, duplicate"):
        controller.verify_phase(path, "screening", args, FakeLauncher())
    controller.save(path, proof)
    Path(next(iter(proof["evidence_files"]))).write_text("changed")
    with pytest.raises(ValueError, match="Evidence changed"):
        controller.verify_phase(path, "screening", args, FakeLauncher())


def test_controller_rejects_wrong_source_before_import(args):
    args.source_revision = "a" * 40
    with pytest.raises(ValueError, match="Only the approved"):
        controller.load_frozen_launcher(args)


def test_controller_refuses_source_file_namespace(args, monkeypatch):
    args.output_root = args.source_root / "models/new_outputs"
    with pytest.raises(ValueError, match="only a fresh outputs"):
        controller.execute(args)


def test_termination_reaches_separately_grouped_training_child(tmp_path):
    # Match the frozen launcher: phase and training each have their own group;
    # the phase's TERM handler terminates and reaps its training group.
    script = '''import os,signal,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],start_new_session=True)
def stop(signum,frame):
    raise InterruptedError('stop')
signal.signal(signal.SIGTERM,stop)
print(child.pid,flush=True)
try:
    child.wait()
except InterruptedError:
    os.killpg(child.pid,signal.SIGTERM)
    child.wait(timeout=5)
    sys.exit(0)
'''
    child = subprocess.Popen([sys.executable, "-c", script], start_new_session=True,
                             stdout=subprocess.PIPE, text=True)
    grandchild_pid = int(child.stdout.readline())
    try:
        controller.stop_child(child)
        assert child.returncode == 0
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        try:
            os.kill(grandchild_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
