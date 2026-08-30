#!/usr/bin/env python3
"""5080-only full-context qualification followed by four fresh seed62 runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper.scripts.validate_count_aware_titantpp_mac_three_seed_validation import (
    load_json, save_json_atomic, sha256_file, verify_source,
)
from paper.scripts.validate_titantpp_mac_stability_preflight import CONTEXTS

REVISION = "c4dbf856c32e6502acc660ffac23c3e2f68e5375"
VALIDATION_ORDER = ("insta_market_basket", "raf_spare_parts", "yellow_trip_hourly",
                    "intermittent_frozen_5000")
VALIDATOR = "paper/scripts/validate_titantpp_mac_stability_preflight.py"
POLICY = "paper/scripts/run_with_titantpp_mac_dynamo_policy.py"
DIAGNOSTIC = "paper/scripts/diagnose_titantpp_mac_nonfinite.py"
TRAINING = "paper/scripts/run_count_aware_tpp_backbone_control.py"


def validate_contract(contract):
    if contract["training_source_revision"] != REVISION:
        raise ValueError("Training revision drift")
    auth = contract["authorization"]
    if (auth["execution_server"] != "5080" or auth["seed"] != 62
            or auth["other_hosts_authorized"] or auth["other_seeds_authorized"]
            or not auth["long_validation_after_all_preflights"]):
        raise ValueError("Only the 5080 seed62 stage is authorized")
    if tuple(contract["validation_order"]) != VALIDATION_ORDER or contract["run_count"] != 4:
        raise ValueError("Unexpected validation grid")
    if tuple(contract["context_preflight_order"]) != VALIDATION_ORDER[1:]:
        raise ValueError("Every remaining context requires a full preflight")
    if set(contract["datasets"]) != set(CONTEXTS):
        raise ValueError("Dataset grid drift")
    for name, (lookback, length) in CONTEXTS.items():
        ds = contract["datasets"][name]
        if (ds["lookback"], ds["max_seq_len"]) != (lookback, length):
            raise ValueError(f"Context drift: {name}")
    expected = {"epochs": 300, "min_epochs": 40, "patience": 40, "batch_size": 128,
                "lr": .001, "hidden_dim": 64, "lambda_log_qty": 1., "lambda_tail": 0.,
                "outer_gradient_clip": 1., "inner_gradient_clip": 1.,
                "time_head": "legacy_clamped_rmtpp", "time_scale": 3.,
                "time_w_max": 10./3., "time_intercept_limit": 30.,
                "checkpoint_selection": "minimum_validation_joint_objective",
                "evaluation_scope": "validation_only"}
    if contract["training"] != expected:
        raise ValueError("Frozen training policy drift")
    runtime = contract["runtime"]
    if (runtime["automatic_retry"] or runtime["resume_failed_checkpoint"]
            or runtime["overwrite_existing_artifact"] or not runtime["require_gdm_inactive"]
            or runtime["minimum_free_mib"] != 12000
            or runtime["dynamo_recompile_limit"] != 64
            or runtime["dynamo_accumulated_recompile_limit"] != 512):
        raise ValueError("Safety policy drift")
    if contract["result_boundary"]["held_out_test"] != "locked":
        raise ValueError("Held-out test must remain locked")


def verify_inputs(project, contract):
    validate_contract(contract)
    source = verify_source(project, contract)
    for name, ds in contract["datasets"].items():
        for path_key, hash_key in (("data_path", "data_sha256"),
                                    ("split_manifest_path", "split_manifest_sha256")):
            if sha256_file(project / ds[path_key]) != ds[hash_key]:
                raise ValueError(f"Data/split checksum mismatch: {name}")
    prior = load_json(project / contract["existing_instacart_gate"])
    if (prior.get("status") != "complete" or prior.get("seed") != 62
            or prior.get("training_source_revision") != REVISION
            or prior.get("train_batch_count") != 15557
            or prior.get("train_target_count") != 1991192
            or prior.get("validation_target_count") != 503733
            or prior.get("held_out_test_evaluated") is not False
            or not all(prior.get(k) is True for k in (
                "all_metrics_finite", "checkpoint_prediction_replay_exact",
                "observed_history_memory_replay_exact"))):
        raise ValueError("Completed Instacart full-epoch gate is missing")
    return source


def check_gpu(gpu_csv, compute_csv, gdm, process_names, minimum=12000):
    rows = [line for line in gpu_csv.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("Expected exactly one RTX 5080")
    name, free = (field.strip() for field in rows[0].split(","))
    if "RTX 5080" not in name or int(free) < minimum:
        raise ValueError("Wrong GPU or insufficient free VRAM")
    if compute_csv.strip() or gdm.strip() != "inactive":
        raise ValueError("GPU is busy or GDM is not inactive")
    if {"gnome-shell", "Xwayland"} & set(process_names.split()):
        raise ValueError("Desktop process is still present")


def gpu_preflight():
    def read(cmd, allowed=(0,)):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode not in allowed:
            raise RuntimeError(f"Preflight command failed: {cmd}: {result.stderr}")
        return result.stdout
    gpu = read(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"])
    compute = read(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    gdm = read(["systemctl", "is-active", "gdm"], allowed=(0, 3))
    names = read(["ps", "-eo", "comm="])
    check_gpu(gpu, compute, gdm, names)
    return {"gpu": gpu.strip(), "compute": compute.strip(), "gdm": gdm.strip()}


def training_command(project, output, contract, dataset, phase, python):
    if phase not in ("context_e1", "validation_e300"):
        raise ValueError("Unknown training phase")
    ds = contract["datasets"][dataset]
    epochs, minimum = (1, 1) if phase == "context_e1" else (300, 40)
    run = output / phase / dataset / "seed_62"
    cmd = [python, DIAGNOSTIC, "--snapshot-dir", str(run.parent / "diagnostic"),
           POLICY, "--recompile-limit", "64", "--accumulated-recompile-limit", "512",
           TRAINING, "--data", str(project / ds["data_path"]), "--split-manifest",
           str(project / ds["split_manifest_path"]), "--output-dir", str(run),
           "--source-revision", REVISION, "--execution-role", f"stable_5080_{phase}_seed62",
           "--dataset-contract", dataset, "--model-role", "experimental", "--device", "cuda",
           "--epochs", str(epochs), "--min-epochs", str(minimum),
           "--early-stopping-patience", "40", "--batch-size", "128", "--lr", "0.001",
           "--hidden-dim", "64", "--lookback-weeks", str(ds["lookback"]),
           "--max-seq-len", str(ds["max_seq_len"]), "--lambda-log-qty", "1",
           "--lambda-tail", "0", "--quantity-variants", "log_mse",
           "--backbones", "titantpp_titans_mac", "--seeds", "62",
           "--time-head-mode", "legacy_clamped_rmtpp", "--time-scale", "3",
           "--time-w-max", str(10./3.), "--time-intercept-limit", "30", "--grad-clip", "1",
           "--titans-memory-gradient-clip", "1", "--allow-partial-contract"]
    return run, cmd


def ensure_fresh(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite/restart an existing artifact: {output}")
    output.mkdir(parents=True, exist_ok=True)


def execute(args):
    project, output = args.project_root.resolve(), args.output_root.resolve()
    contract = load_json(args.contract)
    verify_inputs(project, contract)
    if args.verify_only:
        print("Source, data, contract and prior Instacart gate verified", flush=True)
        return
    if not re.fullmatch(r"[0-9a-f]{40}", args.orchestration_revision):
        raise ValueError("Full orchestration revision is required")
    ensure_fresh(output)
    with (project / ".titantpp_mac_stable_5080.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "starting", "execution_server": "5080", "seed": 62,
                 "training_source_revision": REVISION,
                 "orchestration_revision": args.orchestration_revision,
                 "context_preflights_complete": 0, "completed_run_count": 0,
                 "expected_run_count": 4, "held_out_test_evaluated": False}
        child = None
        def write(**updates):
            state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
            save_json_atomic(output / "status.json", state)
            print(json.dumps(state), flush=True)
        def stop(signum, frame):
            raise InterruptedError(f"Launcher received signal {signum}")
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        def run(cmd, logfile):
            nonlocal child
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "42",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                   "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            with logfile.open("w") as log:
                child = subprocess.Popen(cmd, cwd=project, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                write(child_pid=child.pid, command=cmd, log_path=str(logfile))
                code = child.wait()
                child = None
            if code:
                raise RuntimeError(f"Stage exited {code}; evidence preserved at {logfile}")
        try:
            write(state="running", phase="source_preflight")
            owned_files = [Path(__file__).resolve(), project / VALIDATOR,
                           project / "paper/scripts/validate_count_aware_titantpp_mac_three_seed_validation.py",
                           args.contract.resolve(), project / contract["existing_instacart_gate"]]
            hashes = {str(p): sha256_file(p) for p in owned_files}
            manifest = {"training_source_revision": REVISION,
                        "files": contract["frozen_training_file_sha256"],
                        "orchestration_revision": args.orchestration_revision,
                        "orchestration_file_sha256": hashes,
                        "inner_gradient_clip": 1., "held_out_test": False}
            save_json_atomic(output / "source_manifest.json", manifest)
            save_json_atomic(output / "execution_contract.json", contract)
            for phase, datasets in (("context_e1", contract["context_preflight_order"]),
                                     ("validation_e300", contract["validation_order"])):
                for dataset in datasets:
                    write(phase=phase, current_dataset=dataset, stage="runtime_preflight", child_pid=None)
                    if any(sha256_file(Path(p)) != h for p, h in hashes.items()):
                        raise ValueError("Orchestration changed during execution")
                    verify_inputs(project, contract)
                    gpu = gpu_preflight()
                    runroot, cmd = training_command(project, output, contract, dataset, phase, args.python)
                    runroot.parent.mkdir(parents=True, exist_ok=True)
                    write(phase=phase, current_dataset=dataset, stage="training", gpu_preflight=gpu)
                    run(cmd, runroot.parent / "train.log")
                    gpu_preflight()
                    write(stage="artifact_validation")
                    report = runroot.parent / "validation_report.json"
                    epochs, minimum = (1, 1) if phase == "context_e1" else (300, 40)
                    run([args.python, VALIDATOR, "--project-root", str(project),
                         "--run-root", str(runroot), "--source-manifest", str(output / "source_manifest.json"),
                         "--expected-revision", REVISION, "--seed", "62", "--dataset", dataset,
                         "--epochs", str(epochs), "--min-epochs", str(minimum),
                         "--output", str(report)], runroot.parent / "validator.log")
                    evidence = load_json(report)
                    if evidence.get("status") != "complete":
                        raise ValueError("Artifact gate did not complete")
                    key = "context_preflights_complete" if phase == "context_e1" else "completed_run_count"
                    write(**{key: state[key] + 1}, child_pid=None)
            write(state="complete", stage="complete", child_pid=None)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            write(state="failed", error=repr(exc), child_pid=None)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--orchestration-revision", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--verify-only", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
