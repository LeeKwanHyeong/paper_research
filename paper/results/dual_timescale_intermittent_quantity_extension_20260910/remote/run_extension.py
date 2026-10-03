#!/usr/bin/env python3
"""Run the explicitly authorized, unchanged Intermittent fit exactly once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REVISION = "ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c"
DATASET = "intermittent_frozen_5000"
PHASE = {"epochs": 300, "minimum_epochs": 40, "patience": 40}
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def validate_scope(c):
    require(c["source_revision"] == REVISION, "Source revision changed")
    require(c["datasets"] == [DATASET] and c["seeds"] == [42], "One-job scope changed")
    require(c["phase"] == PHASE, "Epoch/patience contract changed")
    require(c["evaluation_scope"] == "validation_only", "Evaluation scope changed")
    require(c["additional_jobs_authorized"] is False, "Unauthorized follow-up jobs")
    require(c["reuse_e1"] is True, "Completed e1 must be reused")
    require(c["checkpoint_monitor"] == "validation_raw_quantity_rmse", "Selector changed")


def validate_resume_identity(previous, saved_plan, plan, runtime, job):
    for key in ("command", "source_revision", "dataset", "phase",
                "extension_contract_sha256", "runner_sha256"):
        require(saved_plan[key] == plan[key], "Resume execution identity changed: " + key)
    require(previous["runtime"] == runtime and previous["job_directory"] == str(job),
            "Resume runtime or output drift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only an existing execution with exactly the same identity")
    args = parser.parse_args()
    c = read(args.contract)
    validate_scope(c)
    source = Path(c["source_root"]).resolve()
    out = Path(c["output_root"]).resolve()
    prior = Path(c["prior_campaign_root"]).resolve()
    require(out == args.contract.resolve().parent, "Operational bundle/output drift")
    require(out != source and source not in out.parents and out not in source.parents,
            "Source and output must be disjoint")
    require(out != prior and prior not in out.parents and out not in prior.parents,
            "Prior campaign must remain separate")
    require(digest(__file__) == c["runner_sha256"], "Operational runner changed")
    sys.path.insert(0, str(source))
    from paper.scripts import run_dual_timescale_campaign as original

    manifest = Path(c["source_manifest"])
    require(digest(manifest) == c["source_manifest_sha256"], "Manifest changed")
    frozen = original.verify_manifest(manifest, REVISION)
    require({key: frozen["phases"]["seed42_screening"][key] for key in PHASE} == PHASE,
            "Frozen training phase changed")
    require(digest(original.CONTRACT) == c["original_contract_sha256"], "Original contract changed")
    for name, sha in c["prior_evidence_sha256"].items():
        require(digest(prior / name) == sha, "Prior evidence changed: " + name)
    prior_state = read(prior / "status.json")
    require(prior_state["status"] == "stopped_seed42_gate_failed", "Prior terminal state changed")
    proof = read(prior / "e1_intermittent_frozen_5000_audit.json")
    require(proof["status"] == "passed" and proof["completed_epochs"] == 1,
            "Missing successful e1 proof")
    require(proof["source_revision"] == REVISION, "e1 source differs")
    folder = Path(proof["summary"]).parent
    for name, sha in [("summary.json", proof["summary_sha256"]),
                      ("best_val_qty_rmse_model.pt", proof["checkpoint_sha256"]),
                      ("last_epoch_state.pt", proof["last_checkpoint_sha256"])]:
        require(digest(folder / name) == sha, "Reused e1 artifact changed: " + name)
    require(proof["last_restore"]["optimizer_restore"] and
            proof["selected_restore"]["strict_model_restore"], "Incomplete e1 restore proof")
    require(original.checked_cuda_xml(prior / "cuda_tests.xml") ==
            read(prior / "cuda_tests_receipt.json"), "CUDA evidence changed")
    require(read(prior / "cost.json")["status"] == "passed", "Cost gate not passed")
    reference = out / "benchmark_reference.json"
    require(digest(reference) == c["benchmark_reference_sha256"], "Benchmark evidence changed")
    job = out / "jobs" / "seed42_intermittent_frozen_5000"
    command = original.job_command(python=sys.executable, candidate=frozen["candidate"],
        dataset=DATASET, output=job, source_revision=REVISION, phase=PHASE, host_role="5090")
    plan = {"command": command, "source_revision": REVISION, "dataset": DATASET,
            "phase": PHASE, "reused_e1": proof, "extension_contract_sha256": digest(args.contract),
            "runner_sha256": digest(__file__), "created_at": now()}
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False))
        return

    import torch
    torch.set_num_threads(1)
    runtime = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
               "host": platform.node(), "python_executable": str(Path(sys.executable).resolve())}
    require(runtime == read(prior / "identity.json")["runtime"], "Runtime identity changed")
    previous = None
    if (out / "status.json").exists():
        require(args.resume, "Existing extension; use explicit --resume after inspection")
        previous = read(out / "status.json")
        saved_plan = read(out / "execution_plan.json")
        validate_resume_identity(previous, saved_plan, plan, runtime, job)
        if previous["status"] == "completed":
            require(digest(out / "audit.json") == previous["audit_sha256"], "Completed audit drift")
            print(json.dumps({"status": "already_completed", "audit_sha256": previous["audit_sha256"]}))
            return
        require(previous["status"] in ("preflight", "running", "auditing", "execution_failed"),
                "Unsupported resume state")
    else:
        require(not args.resume, "Nothing to resume")
        require(not job.exists(), "Unregistered existing job; do not overwrite")
    lock = open("/tmp/paper_research_dual_timescale_gpu0.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "preflight", "started_at": previous["started_at"] if previous else now(), "updated_at": now(),
             "controller_pid": os.getpid(), "dataset": DATASET, "source_revision": REVISION,
             "job_directory": str(job), "runtime": runtime, "evaluation_scope": "validation_only",
             "quantity_focused_extension": True, "held_out_test_evaluated": False,
             "additional_jobs_authorized": False}
    state["resume_count"] = int(previous.get("resume_count", 0)) + 1 if previous else 0
    try:
        if not previous:
            save(out / "execution_plan.json", plan)
        state["gpu_preflight"] = original.gpu_preflight("RTX 5090")
        save(out / "status.json", state)
        job.mkdir(parents=True, exist_ok=previous is not None)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="42",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source))
        with (out / "training.log").open("a" if previous else "x") as stream:
            child = subprocess.Popen(command, cwd=source, env=env, stdout=stream,
                stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            state.update(status="running", child_pid=child.pid, updated_at=now())
            save(out / "status.json", state)
            code = child.wait()
        require(code == 0, "Training process failed with exit " + str(code))
        state.update(status="auditing", updated_at=now())
        save(out / "status.json", state)
        original.verify_manifest(manifest, REVISION)
        for name, sha in c["prior_evidence_sha256"].items():
            require(digest(prior / name) == sha, "Prior evidence changed during execution")
        receipt = original.full_audit(job, frozen, DATASET, REVISION, PHASE)
        baseline = frozen["B_validation_reference"]["datasets"][DATASET]
        receipt["original_gate_for_transparency"] = original.evaluate_gate(receipt["metrics"], baseline)
        checks = dict(receipt["original_gate_for_transparency"]["checks"])
        checks.pop("clamped_time_loss")
        receipt["quantity_components_of_original_gate"] = checks
        receipt["comparisons"] = []
        for row in read(reference)["rows"]:
            if row["dataset"] != DATASET:
                continue
            require(row["validation_target_identity_sha256"] ==
                    receipt["validation_target_identity_sha256"], "Benchmark population mismatch")
            metrics = {}
            for key in ("raw_rmse", "overall_mae", "body_mae", "gt_p99_mae"):
                value = receipt["metrics"][key]
                metrics[key] = {"reference": row[key], "candidate": value,
                                "relative_change_percent": 100 * (value / row[key] - 1)}
            receipt["comparisons"].append({"model": row["model"], "metrics": metrics})
        require(len(receipt["comparisons"]) == 3, "Incomplete comparator table")
        receipt["interpretation"] = "Exploratory quantity extension; prior Taxi failure unchanged; no model adoption."
        save(out / "audit.json", receipt)
        state.update(status="completed", updated_at=now(), completed_at=now(),
                     best_epoch=receipt["best_epoch"], completed_epochs=receipt["completed_epochs"],
                     audit_sha256=digest(out / "audit.json"), metrics=receipt["metrics"],
                     final_model_unchanged="TitanTPP(B)")
        save(out / "status.json", state)
    except BaseException as error:
        state.update(status="execution_failed", error=repr(error), updated_at=now())
        save(out / "status.json", state)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
