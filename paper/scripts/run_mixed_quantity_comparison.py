#!/usr/bin/env python3
"""Freeze, qualify and explicitly execute the three-arm mixed quantity study.

The freeze/validate commands are local metadata operations. GPU execution needs
a receipt bound to this exact contract; old training approvals are not reused.
"""
from __future__ import annotations

import argparse
import copy
import gc
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts import run_quantity_comparison as base  # noqa: E402

SCHEMA = "mixed_quantity_execution_v1"
PARENT_SHA256 = "e7f892c0b6f14c2a9b4f2fcfa9692d75e3cc7498e8208fb02da3cb750d37dad1"
SCOPE = "5090_mixed_quantity_qualification_calibration_9_fresh_arms_18_validation_replays"
CASES = ("B_log_original", "mixed_original", "mixed_matched_original")
LIMITS = {"max_wall_seconds": 172800, "qualification_seconds": 900,
          "calibration_seconds_per_dataset": 600, "max_concurrent_gpu_jobs": 1,
          "output_bytes": 2 * 1024**3, "minimum_free_bytes": 5 * 1024**3,
          "cpu_threads": 4, "per_file_bytes": 64 * 1024**2, "termination_grace_seconds": 15}
POLICY = {"automatic_retry": False, "automatic_resume": False, "held_out": False,
          "budget_extension": False, "cpu_fallback": False}
REMOTE = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/mixed_quantity_seed42_v1"
EXECUTION = {"server_alias": "5090", "python": "/opt/miniconda3/envs/ai_env/bin/python3.12",
             "source_dir": REMOTE + "/source", "output_dir": REMOTE + "/run",
             "tmux_session": "mixed_quantity_seed42_5090_v1", "tmux_binary": "/opt/miniconda3/envs/ai_env/bin/tmux"}
require, read_json, write_json = base.require, base.read_json, base.write_json
sha_json, sha_file = base.sha_json, base.sha_file


class Budget(base.Budget):
    """Keep old aggregate limits and enforce the new per-file limit too."""

    def __call__(self, stage=None):
        previous_scan = self.last_scan
        super().__call__(stage)
        if stage == "before_epoch_commit" or self.last_scan != previous_scan:
            for path in self.output.rglob("*"):
                try:
                    if path.is_file():
                        require(path.stat().st_size <= self.contract["limits"]["per_file_bytes"],
                                "Individual artifact file ceiling reached")
                except FileNotFoundError:
                    continue  # Concurrent atomic rename by the owned worker.


class CalibrationDeadline:
    """Supervisor deadline continues even when a calibration kernel stalls."""

    def __init__(self, *, clock=time.time, monotonic=time.monotonic):
        self.clock, self.monotonic = clock, monotonic
        self.phase, self.deadline = None, None

    def remaining(self, status):
        if status.get("status") != "calibrating":
            return float("inf")
        phase = (status["dataset_id"], status["phase_started_at_unix"])
        if phase != self.phase:
            elapsed = max(0, self.clock() - status["phase_started_at_unix"])
            self.deadline = self.monotonic() + max(0, LIMITS["calibration_seconds_per_dataset"] - elapsed)
            self.phase = phase
        return self.deadline - self.monotonic()


def source_hashes(root=ROOT):
    # The existing engine's static import closure contains the mixed primitive.
    # These two new entrypoint-only modules depend solely on that same closure.
    files = base.source_hashes(root)
    for name in ("run_mixed_quantity_comparison.py", "mixed_quantity_acceptance.py"):
        relative = "paper/scripts/" + name
        files[relative] = sha_file(Path(root) / relative)
    return dict(sorted(files.items()))


def validate_contract(contract, *, check_source=True):
    from paper.scripts.mixed_quantity_acceptance import ACCEPTANCE_POLICY
    from paper.scripts.mixed_quantity_objective import CALIBRATION_POLICY
    require(contract["schema"] == SCHEMA and contract["status"] == "frozen_pending_explicit_approval", "Execution schema/status mismatch")
    require(contract["cases"] == list(CASES) and contract["epochs"] == 120 and contract["seed"] == 42,
            "Fixed cases/epoch/seed mismatch")
    require(contract["limits"] == LIMITS and contract["policy"] == POLICY, "Execution limits/policy mismatch")
    require(contract["execution"] == EXECUTION, "Pinned 5090 execution paths required")
    require(contract["calibration_policy"] == CALIBRATION_POLICY and contract["acceptance_policy"] == ACCEPTANCE_POLICY,
            "Calibration or acceptance policy changed")
    parent = contract["basis"]["contract"]
    require(sha_json(parent) == contract["basis"]["contract_sha256"] == PARENT_SHA256, "Parent contract hash mismatch")
    base.validate_contract(parent, check_source=False)
    require(contract["datasets"] == parent["datasets"], "Frozen B data/model/optimizer settings changed")
    require(contract["total_optimizer_steps"] == sum(d["expected_global_steps"] for d in contract["datasets"]) * 3 == 6816240,
            "Nine-arm step budget mismatch")
    require(contract["validation_replays"] == 18, "Exactly two validation replays per arm required")
    require(contract["approval"] == {"approved": False, "required_scope": SCOPE}, "Pending approval metadata required")
    runtime = contract["runtime_expected"]
    require(runtime == contract["runtime_evidence"]["runtime"], "Runtime evidence mismatch")
    require(runtime["device"] == "cuda:0" and runtime["gpu"]["uuid"] == "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f",
            "Pinned 5090 GPU required")
    require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source manifest hash mismatch")
    for name, digest in parent["source"]["files"].items():
        if name != "paper/scripts/quantity_comparison_engine.py":
            require(contract["source"]["files"].get(name) == digest, "Original B dependency changed: " + name)
    if check_source:
        require(source_hashes() == contract["source"]["files"], "Source closure or contents changed")
    return {"status": "validated_metadata_only", "contract_sha256": sha_json(contract),
            "arms": 9, "optimizer_steps": 6816240, "gpu_started": False}


def validate_approval(contract, approval):
    require(approval.get("status") == "approved_by_user" and approval.get("execution_contract_sha256") == sha_json(contract),
            "Explicit approval bound to this contract required")
    require(approval.get("scope") == SCOPE and bool(approval.get("user_instruction")), "Approval scope/instruction mismatch")


def freeze(parent_path, runtime_path, destination):
    from paper.scripts.mixed_quantity_acceptance import ACCEPTANCE_POLICY
    from paper.scripts.mixed_quantity_objective import CALIBRATION_POLICY
    require(not Path(destination).exists(), "Refusing to replace a frozen contract")
    parent, runtime_evidence = read_json(parent_path), read_json(runtime_path)
    base.validate_contract(parent, check_source=False)
    files = source_hashes()
    digest = sha_json(files)
    preparation = ROOT / "search_artifacts/mixed_quantity_preparation_v1"
    folder = preparation / "snapshots" / digest
    snapshot = folder / "source"
    snapshot.mkdir(parents=True, exist_ok=False)
    for sentinel in ("models", "utils", "sample_data"):
        (snapshot / sentinel).mkdir(exist_ok=True)
    for name, file_hash in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        require(sha_file(target) == file_hash, "Source copy mismatch")
    archive = folder / "source.tar.gz"
    with archive.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
            for sentinel in ("models", "utils", "sample_data"):
                info = tarfile.TarInfo("source/" + sentinel)
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                bundle.addfile(info)
            for name in files:
                info = tarfile.TarInfo("source/" + name)
                info.size, info.mode = (snapshot / name).stat().st_size, 0o644
                with (snapshot / name).open("rb") as handle:
                    bundle.addfile(info, handle)
    contract = {"schema": SCHEMA, "status": "frozen_pending_explicit_approval", "cases": list(CASES),
                "seed": 42, "epochs": 120, "datasets": copy.deepcopy(parent["datasets"]),
                "basis": {"path": str(Path(parent_path).resolve()), "contract_sha256": sha_json(parent), "contract": parent},
                "source": {"files": files, "files_sha256": digest, "local_snapshot": str(snapshot),
                           "archive": str(archive), "archive_sha256": sha_file(archive),
                           "git_head_provenance": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                           "identity_authority": "content-addressed import closure; includes uncommitted implementation"},
                "runtime_evidence": {"path": str(Path(runtime_path).resolve()), "sha256": sha_file(runtime_path),
                                     "runtime": runtime_evidence["runtime"]},
                "runtime_expected": runtime_evidence["runtime"], "execution": EXECUTION,
                "calibration_policy": CALIBRATION_POLICY, "acceptance_policy": ACCEPTANCE_POLICY,
                "limits": LIMITS, "policy": POLICY, "total_optimizer_steps": 6816240, "validation_replays": 18,
                "approval": {"approved": False, "required_scope": SCOPE},
                "coefficients_state": "rho fixed; dataset alpha/c determined once from approved train-only calibration before all arms; no optimizer updates",
                "runtime_state": "prior read-only evidence; exact runtime recheck and fresh CUDA qualification mandatory at launch",
                "budget_interpretation": "48h total ceiling including qualification/calibration/training/validation; not a measured ETA",
                "result_scope": "single-seed validation engineering screen; no benchmark superiority claim"}
    validate_contract(contract)
    require(source_hashes(snapshot) == files, "Frozen snapshot import closure differs")
    write_json(destination, contract)
    return validate_contract(contract)


def configured_runtime(contract, device):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    from paper.scripts.run_time_quantity_diagnostic import validate_import_origins
    validate_import_origins()
    configure_runtime(device, threads=4)
    runtime = runtime_identity(device)
    if device == "cuda:0":
        require(runtime == contract["runtime_expected"], "Pinned 5090 runtime changed")
    return runtime


def execution_identity(contract, runtime, data, **extra):
    return {"execution_contract_sha256": sha_json(contract), "runtime": runtime,
            "source": contract["source"], "data": data, **extra}


def audit_pairs(summaries):
    require(len(summaries) == 3, "Three completed cases required")
    baseline = summaries[0]
    for name, candidate in zip(CASES, summaries, strict=True):
        require(candidate["condition"]["mixed_objective"]["name"] == name, "Case order mismatch")
        require(candidate["status"] == "complete" and candidate["initial_state_sha256"] == baseline["initial_state_sha256"]
                and candidate["global_step"] == baseline["global_step"]
                and len(candidate["history"]) == len(baseline["history"]), "Paired state/budget mismatch")
        for left, right in zip(baseline["history"], candidate["history"], strict=True):
            for key in ("epoch", "global_step", "train_count", "train_batches", "validation_count",
                        "validation_batches", "train_batch_order_sha256"):
                require(left[key] == right[key], "Paired exposure mismatch: " + key)
    return {"status": "passed", "arms": 3, "epochs": len(baseline["history"]), "global_steps_per_case": baseline["global_step"]}


def probe_worker(contract, case, stage, started, device):
    from paper.scripts.mixed_quantity_objective import calibrate_mixed_objective, objectives_from_calibration
    from paper.scripts.quantity_comparison_engine import run_case
    from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics
    runtime = configured_runtime(contract, device)
    model, train, validation = base.synthetic_inputs()
    model.to(device)
    identity = execution_identity(contract, runtime, {"kind": "synthetic", "train_targets": 8, "validation_targets": 8},
                                  purpose="synthetic_qualification")
    statistics = QuantityStatistics(1.0, 4.0)
    calibration = calibrate_mixed_objective(model, train.dataset, statistics=statistics, identity=identity,
                                          device=device, synthetic=True)
    objective = next(o for o in objectives_from_calibration(calibration) if o.name == case)
    output = Path(contract["execution"]["output_dir"]) / "qualification" / case / ("full" if stage == "full" else "split")
    return run_case(model=model, train_loader=train, validation_loader=validation, case=QuantityCase.B_LOG_ORIGINAL,
                    mixed_objective=objective, statistics=statistics, output_dir=output, epochs=2, seed=42,
                    identity=identity, device=device, resume=stage == "resume", stop_after_epochs=1 if stage == "first" else None,
                    budget_check=Budget(contract, started),
                    cuda_qualification={"purpose": "synthetic_probe", "execution_contract_sha256": sha_json(contract), "runtime": runtime})


def shape_checks(contract, started, device):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.mixed_quantity_objective import MixedQuantityObjective, mixed_joint_causal_batch_objective
    from paper.scripts.quantity_objective_comparison import QuantityStatistics
    from paper.scripts.run_time_quantity_diagnostic import reset_seed
    from paper.scripts.time_quantity_diagnostic import task_outputs
    from simple_lab_test.search.common.runner import capture_rng_state, restore_rng_state
    configured_runtime(contract, device)
    budget = Budget(contract, started)
    for data in contract["datasets"]:
        for name in CASES:
            budget()
            reset_seed(42)
            stats = QuantityStatistics(data["statistics"]["train_log_mean"], data["statistics"]["raw_scale"])
            model, _ = build_count_aware_model(**data["model"], train_log_mean=stats.mu,
                         train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])
            model.to(device)
            state_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
            require(9 * (32 * state_bytes + 4 * 1024**2) < LIMITS["output_bytes"] // 2,
                    "Projected artifacts exceed budget")
            require(10 * state_bytes + 4 * 1024**2 < LIMITS["per_file_bytes"], "Projected checkpoint exceeds file limit")
            objective = MixedQuantityObjective(name, 0.0 if name == CASES[0] else 0.1,
                                               0.95 if name == CASES[2] else 1.0, "0" * 64)
            dts = torch.ones((128, data["loader"]["max_seq_len"]), device=device)
            mask = torch.ones_like(dts, dtype=torch.bool)
            mask[::2, :4] = False
            quantities = torch.full_like(dts, stats.q0)
            rng = capture_rng_state()
            output = mixed_joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, objective=objective)
            if name == CASES[0]:
                restore_rng_state(rng)
                original = task_outputs(model, dts, mask, quantities, "joint")
                require(torch.equal(output["objective_loss"], original["objective_loss"]), "B shape objective changed")
            output["objective_loss"].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01).step()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite shape step")
            model.eval()
            with torch.no_grad():
                before = mixed_joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, objective=objective)["pred_qty"]
                dts[~mask], quantities[~mask] = 999., 999.
                dts[:, -1], quantities[:, -1] = 31., 42.
                after = mixed_joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, objective=objective)["pred_qty"]
                require(torch.equal(before, after), "Target/padding causality failed")
            del model, output
            gc.collect()


def qualify(contract, contract_path, started, device="cuda:0"):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    output = Path(contract["execution"]["output_dir"])
    runtime = configured_runtime(contract, device)
    began, summaries = time.monotonic(), []
    for case in CASES:
        for stage in ("full", "first", "resume"):
            remaining = LIMITS["qualification_seconds"] - (time.monotonic() - began)
            require(remaining > 0, "Qualification deadline reached")
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "_probe", "--contract", str(contract_path),
                            "--case", case, "--stage", stage, "--started", str(started), "--device", device],
                           check=True, timeout=remaining)
        folder = output / "qualification" / case
        full, split = read_json(folder / "full/summary.json"), read_json(folder / "split/summary.json")
        base.compare_replay(full, split)
        a = torch_load_checkpoint(folder / "full/last_epoch_state.pt", map_location="cpu")
        b = torch_load_checkpoint(folder / "split/last_epoch_state.pt", map_location="cpu")
        for key in ("optimizer_state_sha256", "rng_state_sha256", "model_state_sha256"):
            require(a[key] == b[key], "Fresh-process replay mismatch: " + key)
        summaries.append(full)
    audit_pairs(summaries)
    remaining = LIMITS["qualification_seconds"] - (time.monotonic() - began)
    require(remaining > 0, "Qualification deadline reached")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "_shapes", "--contract", str(contract_path),
                    "--started", str(started), "--device", device], check=True, timeout=remaining)
    receipt = {"passed": True, "qualifies_cuda": device == "cuda:0", "runtime": runtime,
               "execution_contract_sha256": sha_json(contract), "elapsed_seconds": time.monotonic() - began,
               "checks": ["three_fresh_process_replays", "optimizer_rng_identity", "paired_batches",
                          "production_shapes", "B_identity", "target_padding_causality", "synthetic_calibration"]}
    require(configured_runtime(contract, device) == runtime, "Runtime drift during qualification")
    write_json(output / "qualification/receipt.json", receipt)
    return receipt


def run_suite(contract, contract_path, started):
    from paper.scripts.mixed_quantity_acceptance import assess_acceptance, evaluate_checkpoints
    from paper.scripts.mixed_quantity_objective import calibrate_mixed_objective, objectives_from_calibration
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.quantity_comparison_engine import run_case
    from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics
    from paper.scripts.run_time_quantity_diagnostic import build_arm_inputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    output, budget = Path(contract["execution"]["output_dir"]), Budget(contract, started)
    receipt = qualify(contract, contract_path, started)
    require(receipt["qualifies_cuda"], "CPU qualification cannot authorize GPU training")
    completed, evaluations = [], {}
    for data in contract["datasets"]:
        dataset_id = data["dataset_id"]
        write_json(output / "status.json", {"status": "calibrating", "dataset_id": dataset_id,
                   "phase_started_at_unix": time.time(), "completed_arms": len(completed)})
        budget()
        frame, metadata = prepare_quantity_comparison_data(data)
        metadata = base.bind_frozen_statistics(data, metadata)
        identity = execution_identity(contract, receipt["runtime"], data["inherited_data_identity"], dataset_id=dataset_id)
        stats = QuantityStatistics(metadata["train_log_mean"], metadata["raw_scale"])
        model, train, validation, _ = build_arm_inputs({**data, "seed": 42}, frame, metadata)
        model.to("cuda:0")
        calibration_start = time.monotonic()

        def calibration_budget(stage=None):
            budget(stage)
            require(time.monotonic() - calibration_start < LIMITS["calibration_seconds_per_dataset"], "Calibration deadline reached")

        calibration = calibrate_mixed_objective(model, train.dataset, statistics=stats, identity=identity,
                                              device="cuda:0", budget_check=calibration_budget)
        calibration_budget()
        write_json(output / dataset_id / "calibration.json", calibration)
        objectives = objectives_from_calibration(calibration)
        del model, train, validation
        summaries, evaluations[dataset_id] = [], {}
        for objective in objectives:
            budget()
            require(configured_runtime(contract, "cuda:0") == receipt["runtime"], "Runtime drift before arm")
            write_json(output / "status.json", {"status": "running", "dataset_id": dataset_id, "case": objective.name,
                       "completed_arms": len(completed), "started_at_unix": started})
            model, train, validation, _ = build_arm_inputs({**data, "seed": 42}, frame, metadata)
            require(canonical_state_dict_sha256(model.state_dict()) == calibration["initial_model_state_sha256"],
                    "Calibration/arm initialization mismatch")
            arm_dir = output / dataset_id / objective.name
            summary = run_case(model=model, train_loader=train, validation_loader=validation,
                      case=QuantityCase.B_LOG_ORIGINAL, mixed_objective=objective, statistics=stats,
                      output_dir=arm_dir, epochs=120, seed=42, identity=identity, device="cuda:0",
                      lr=.001, weight_decay=.01, grad_clip=1., budget_check=budget, cuda_qualification=receipt)
            require(summary["global_step"] == data["expected_global_steps"], "Arm step budget incomplete")
            summaries.append(summary)
            completed.append({"dataset_id": dataset_id, "case": objective.name})
            write_json(output / "status.json", {"status": "validating_checkpoints", "dataset_id": dataset_id,
                       "case": objective.name, "completed_arms": len(completed)})
            evaluation = evaluate_checkpoints(model=model, validation_loader=validation, statistics=stats, objective=objective,
                         arm_dir=arm_dir, quantity_boundaries=data["quantity_boundaries_all_train_rows"],
                         device="cuda:0", budget_check=budget)
            evaluations[dataset_id][objective.name] = evaluation
            write_json(arm_dir / "validation_diagnosis.json", evaluation)
            del model, train, validation
            gc.collect()
        write_json(output / dataset_id / "paired_comparison.json", audit_pairs(summaries))
        del frame
        gc.collect()
    write_json(output / "acceptance.json", assess_acceptance(evaluations))
    write_json(output / "status.json", {"status": "complete", "completed_arms": 9, "optimizer_steps": 6816240,
               "validation_replays": 18, "arms": completed, "held_out_evaluated": False,
               "started_at_unix": started, "elapsed_seconds": time.time() - started})


def require_launch_receipt(contract, started):
    receipt = read_json(Path(contract["execution"]["output_dir"]) / "execution_receipt.json")
    validate_approval(contract, receipt["approval"])
    require(receipt["contract_sha256"] == sha_json(contract) and receipt["started_at_unix"] == started
            and receipt["deadline_unix"] == started + LIMITS["max_wall_seconds"], "Launch receipt mismatch")
    require(time.time() < receipt["deadline_unix"], "Launch deadline already reached")


def execute(contract, contract_path, approval):
    import fcntl
    validate_contract(contract)
    validate_approval(contract, approval)
    require(str(Path(sys.executable).resolve()) == EXECUTION["python"] and str(ROOT) == EXECUTION["source_dir"],
            "Run from the pinned 5090 Python/source snapshot")
    require(bool(os.environ.get("TMUX")), "Long training must run inside the specified tmux session")
    session = subprocess.check_output([EXECUTION["tmux_binary"], "display-message", "-p", "#S"], text=True).strip()
    require(session == EXECUTION["tmux_session"], "Unexpected tmux session")
    output = Path(EXECUTION["output_dir"])
    require(not output.exists(), "Fresh output required; automatic resume prohibited")
    require(shutil.disk_usage(output.parent).free >= LIMITS["minimum_free_bytes"], "Insufficient free disk")
    with (output.parent / "mixed_quantity_gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = subprocess.check_output(["nvidia-smi", "--id=" + contract["runtime_expected"]["gpu"]["uuid"],
                   "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        require(not running, "GPU has another compute process")
        output.mkdir()
        started = time.time()
        write_json(output / "execution_receipt.json", {"contract_sha256": sha_json(contract), "approval": approval,
                   "started_at_unix": started, "deadline_unix": started + LIMITS["max_wall_seconds"]})
        environment = os.environ.copy()
        for key, value in contract["runtime_expected"]["environment"].items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        process, handlers = None, {}

        def interrupted(signum, frame):
            raise InterruptedError(f"Launcher received signal {signum}")

        try:
            for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                handlers[signum] = signal.signal(signum, interrupted)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_suite", "--contract", str(contract_path),
                                        "--started", str(started)], env=environment, start_new_session=True)
            began, budget = time.monotonic(), Budget(contract, started)
            calibration_deadline = CalibrationDeadline()
            while True:
                budget()
                qualified = (output / "qualification/receipt.json").is_file()
                ceiling = LIMITS["max_wall_seconds"] if qualified else LIMITS["qualification_seconds"]
                remaining = ceiling - (time.monotonic() - began)
                status_path = output / "status.json"
                if qualified and status_path.exists():
                    status = read_json(status_path)
                    remaining = min(remaining, calibration_deadline.remaining(status))
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, ceiling)
                try:
                    code = process.wait(timeout=min(5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(code == 0, f"Suite failed with exit code {code}; no retry")
        except BaseException as error:
            base.stop_owned_process_group(process)
            previous = read_json(output / "status.json") if (output / "status.json").exists() else {}
            write_json(output / "status.json", {**previous,
                       "status": "timeout" if isinstance(error, (subprocess.TimeoutExpired, TimeoutError)) else "failed",
                       "error": str(error), "automatic_retry": False})
            raise
        finally:
            for signum, previous in handlers.items():
                signal.signal(signum, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "execute", "_suite", "_probe", "_shapes"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--stage", choices=("full", "first", "resume"))
    parser.add_argument("--started", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    if args.command == "freeze":
        require(args.parent and args.runtime_evidence, "Parent and runtime evidence required")
        print(json.dumps(freeze(args.parent, args.runtime_evidence, args.contract)))
        return
    contract = read_json(args.contract)
    if args.command == "validate":
        print(json.dumps(validate_contract(contract)))
    elif args.command == "execute":
        require(args.approval is not None, "An explicit approval receipt is required")
        execute(contract, args.contract.resolve(), read_json(args.approval))
    else:
        require(args.started is not None, "Internal command requires a launch start time")
        if args.device == "cuda:0" or args.command == "_suite":
            validate_contract(contract)
            require(str(ROOT) == EXECUTION["source_dir"] and str(Path(sys.executable).resolve()) == EXECUTION["python"],
                    "Internal GPU command requires frozen 5090 source/Python")
            require_launch_receipt(contract, args.started)
        if args.command == "_suite":
            run_suite(contract, args.contract.resolve(), args.started)
        elif args.command == "_shapes":
            shape_checks(contract, args.started, args.device)
        else:
            require(args.case and args.stage, "Synthetic probe case/stage required")
            probe_worker(contract, args.case, args.stage, args.started, args.device)


if __name__ == "__main__":
    main()
