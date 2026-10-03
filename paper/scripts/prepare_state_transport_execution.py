#!/usr/bin/env python3
"""Freeze the bounded two-dataset GPU proposal; never connect or launch work.

The generated contract fixes the scientific and resource conditions. Native
qualification and an approval-bound supervisor remain required before launch;
the raw shared-training CLI is not a substitute for that supervisor.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts.observed_slot_parallel_common import require, sha_file, sha_json, source_manifest, write_json

ARMS = ("titantpp", "titantpp_event_state_transport", "titantpp_elapsed_state_transport")
ASSIGNMENTS = {"5080": ["yellow_trip_hourly"], "5090": ["intermittent_frozen_5000"]}
RUN_NAME = "state_transport_observed_time_seed42_20260919_v1"
PRIOR = "paper/contracts/pair_message_observed_time_execution_v1.json"
TIME = "paper/contracts/hard_lmm_frozen_lognormal_duration_v1.json"
DESIGN = "paper/contracts/hard_lmm_state_transport_design_v1.json"
OBSERVATION = "paper/contracts/observed_time_joint_training_v1.json"
READINESS = "search_artifacts/state_transport_preparation_20260919_v1/host_readiness.json"
ENTRYPOINTS = ("paper/scripts/run_count_aware_tpp_backbone_control.py",
               "paper/scripts/verify_state_transport_cost.py",
               "paper/scripts/prepare_state_transport_execution.py",
               "paper/scripts/run_state_transport_execution.py")
LIMITS = {"qualification_seconds_per_host": 900, "total_wall_seconds": 86400,
          "max_concurrent_gpu_jobs_per_host": 1, "max_gpu_hours_aggregate": 48,
          "per_host_output_bytes": 4 * 1024**3, "min_free_bytes": 5 * 1024**3}
ACCEPTANCE = {"raw_rmse_ratio_strictly_less_than": 1.0, "mae_ratio_max": 1.01,
              "body_mae_ratio_max": 1.02, "tail_mae_ratio_max": 1.02,
              "recorded_time_nll_increase_max": 0.01,
              "last30_rmse_mean_ratio_max": 1.05, "last30_rmse_std_ratio_max": 1.05}
COST_GATES = {"parameter_ratio_max": 1.05, "median_cuda_step_ratio_max": 2.5,
              "peak_cuda_allocated_ratio_max": 2.5, "device_memory_fraction_max": 0.8}
TIME_KEYS = ("expected_train_target_dt_sha256", "expected_validation_target_dt_sha256",
             "expected_train_targets", "expected_validation_targets", "train_time_scale",
             "train_log_scaled_mean", "train_log_scaled_std")


def read(path):
    return json.loads(Path(path).read_text())


def build_contract(cpu_path: Path, implementation_path: Path) -> dict:
    cost, receipt = read(cpu_path), read(implementation_path)
    require(cost["status"] == "complete" and receipt["verification"]["failed"] == 0,
            "Completed CPU cost and passing implementation verification required")
    require(receipt["verification"]["passed"] > 0, "Missing verification evidence")
    require(receipt.get("native_execution_verification", {}).get("status") == "passed_synthetic_cpu",
            "Synthetic verification of the approval-bound executor is required before freezing")
    old, time_contract = read(ROOT / PRIOR), read(ROOT / TIME)
    readiness = read(ROOT / READINESS)
    require(readiness.get("mode") == "read_only_ssh_no_gpu_workload"
            and set(readiness.get("hosts", {})) == set(ASSIGNMENTS), "Missing host readiness snapshot")
    datasets = []
    for dataset_id in ("yellow_trip_hourly", "intermittent_frozen_5000"):
        data = deepcopy(next(d for d in old["datasets"] if d["dataset_id"] == dataset_id))
        duration = next(d for d in time_contract["datasets"] if d["dataset"] == dataset_id)
        require(data["inherited_data_identity"]["data"]["sha256"] == duration["data_sha256"]
                and data["loader"]["lookback_weeks"] == duration["lookback"]
                and data["loader"]["max_seq_len"] == duration["max_sequence_length"],
                "Duration statistics do not match the frozen data/loader")
        data["model"] = {"hidden_dim": 64, "quantity_variant": "count_only_log_regression",
            "lambda_log_qty": 1.0, "lambda_tail": 0.0,
            "time_head_mode": "heteroscedastic_lognormal_duration",
            "time_scale": duration["train_time_scale"],
            "time_initial_location": duration["train_log_scaled_mean"],
            "time_initial_scale": duration["train_log_scaled_std"], "time_sigma_floor": 0.001,
            "time_head_lr_multiplier": 1.0,
            "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": duration["time_unit"], "top_code": None}}
        data["time_statistics"] = {k: duration[k] for k in TIME_KEYS}
        data["time_statistics"]["recompute_absolute_tolerance"] = 1e-12
        data["time_statistics"]["recompute_before_training"] = True
        data.pop("validation_diagnosis", None)
        datasets.append(data)
    hosts = deepcopy(old["hosts"])
    for alias, host in hosts.items():
        observed = readiness["hosts"][alias]
        require(observed["gpu_uuid"] == host["gpu_uuid"]
                and observed["python"] == host["python"]
                and all(observed["runtime"].get(k) == v for k, v in host["runtime_expected"].items()),
                "Read-only host snapshot differs from frozen runtime")
        host["assigned_datasets"] = ASSIGNMENTS[alias]
        host["root"] = str(Path(host["root"]).parent / RUN_NAME)
        host["source_root"] = host["root"] + "/source"
        host["output_dir"] = host["root"] + "/run"
        host["tmux"] = RUN_NAME + "_" + alias
    files = source_manifest(ROOT, entrypoints=ENTRYPOINTS, extra_files=(DESIGN, OBSERVATION, PRIOR, TIME))
    return {
        "schema": "state_transport_observed_time_execution_v1", "date": "2026-09-19",
        "status": "conditions_frozen_not_approved_not_cuda_qualified",
        "approval_required": True, "approved": False,
        "objective": "Test bounded elapsed-time state propagation against fresh B and an identical-capacity event-clock control under one recorded-time probability law.",
        "model_role": "observed_time_state_transport_v1", "seed": 42, "epochs": 120,
        "architecture": {"hidden_dim": 64, "rank": 8, "scan_mode": "parallel", "time_scale": "frozen_train_median"},
        "arms": list(ARMS), "datasets": datasets, "hosts": hosts,
        "total_arms": 6, "total_optimizer_steps": 1215720, "endpoint_replays": 12,
        "source": {"base_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "working_tree": "dirty; source-file hashes and bundle are authoritative", "files": files, "files_sha256": sha_json(files)},
        "evidence": {"cpu_cost": str(cpu_path.relative_to(ROOT)), "cpu_cost_sha256": sha_file(cpu_path),
                     "implementation": str(implementation_path.relative_to(ROOT)), "implementation_sha256": sha_file(implementation_path),
                     "host_readiness": READINESS, "host_readiness_sha256": sha_file(ROOT / READINESS),
                     "host_readiness_checked_at_utc": readiness["checked_at_utc"]},
        "limits": deepcopy(LIMITS), "acceptance": deepcopy(ACCEPTANCE),
        "cost_gates": deepcopy(COST_GATES),
        "design_sha256": sha_file(ROOT / DESIGN),
        "comparison": {"initialization": "same fresh B tensors and caller RNG; new tensors equal except clock identity; output projection zero",
            "same_batch_exposure": "record and compare each epoch's train/validation content+order hashes and counts",
            "arm_order": list(ARMS), "same_dataset_same_host": True,
            "state_transport": "rank8 parallel affine scan; event clock=1 vs elapsed clock=observed dt/frozen train scale; unchanged common head and objective",
            "selector": "strict earliest finite validation raw quantity RMSE minimum",
            "metrics": "selected checkpoint RMSE/MAE/body/tail and recorded-time NLL together; last120 and last30 stability separately",
            "body_tail": "quantity<=all-train p95; quantity>all-train p99; preserve all other quantity/history strata",
            "pass": "candidate satisfies every gate against BOTH fresh B and event-clock control on BOTH datasets",
            "scope": "single-seed reused-validation exploration; not statistical significance or benchmark superiority",
            "excluded": "Instacart, new seeds, held-out, NB head and auxiliary losses require separate future decisions"},
        "qualification": {"executed": False, "approval_bound": True,
            "entrypoint": "paper/scripts/run_state_transport_execution.py qualify",
            "workload": "per host batch128 lengths64/256; each arm 2 warmup+5 measured updates, plus 3 restore updates per added arm; 48 synthetic updates",
            "checks": ["PMF boundaries/tails and finite forward/backward/optimizer", "sequential/parallel scan forward and backward", "unit-clock candidate/control state equality", "initial B outputs/objective/common preclip gradients/RNG", "activated causal predictions and all added gradients", "exact saved optimizer/RNG next-update replay", "measured CUDA cost gates"],
            "cpu_cannot_qualify_cuda": True},
        "launch": {"current_server_availability": "read-only snapshot recorded in evidence; verify idle device and frozen runtime again immediately before launch",
            "root_must_be_new": True, "source_sync": "new isolated roots; verified bundle only; no shared runtime changes or deletes",
            "native_supervisor_binding": "required before execution: contract+approval SHA, exclusive output, owned process group, inherited worker authority, wall+monotonic deadline, storage and GPU exclusivity checks",
            "supervisor_entrypoint": "paper/scripts/run_state_transport_execution.py",
            "supervisor_adaptation_status": "implemented; see separate synthetic CPU execution verification receipt",
            "training_core": ENTRYPOINTS[0], "training_cli_direct_launch_allowed": False,
            "budget_origin": "earliest newly approved native qualification start; both hosts share one 24h deadline including replay/audit work",
            "absolute_deadline": None, "estimated_completion_hours": None,
            "time_cap_rationale": "24 hours is a shared termination ceiling, not an ETA. Previous same-dataset three-arm training took approximately 8.49h on 5090 and 1.38h on 5080; the new CUDA cost and actual epoch durations remain unmeasured.",
            "deadline_action": "terminate only owned process group; mark incomplete comparison inconclusive; no automatic extension",
            "approval_scope": "new native qualification plus six fresh arms on listed 5080/5090 assignments; existing experiment approval is not reusable"},
        "policy": {"retry": False, "resume": False, "server_transfer": False, "additional_arms": False,
                   "additional_seeds": False, "held_out": False, "runtime_mutation": False,
                   "scheduler_changes": False, "commit_push": False, "budget_extension": False},
    }


def validate_contract(contract: dict, *, verify_source: bool = True) -> None:
    require(contract["schema"] == "state_transport_observed_time_execution_v1", "Wrong contract")
    require(contract["arms"] == list(ARMS) and contract["seed"] == 42 and contract["epochs"] == 120, "Frozen arms/budget changed")
    require({h: x["assigned_datasets"] for h, x in contract["hosts"].items()} == ASSIGNMENTS, "Host allocation changed")
    require(contract["limits"] == LIMITS and contract["acceptance"] == ACCEPTANCE, "Resource or performance gate changed")
    require(contract["cost_gates"] == COST_GATES, "Frozen CUDA cost gates changed")
    require(len(contract["datasets"]) == 2 and {d["dataset_id"] for d in contract["datasets"]} == {x[0] for x in ASSIGNMENTS.values()}, "Dataset scope changed")
    steps = sum(math.ceil(d["inherited_data_identity"]["populations"]["train"]["target_count"] / d["loader"]["batch_size"]) * contract["epochs"] * len(ARMS) for d in contract["datasets"])
    require(steps == contract["total_optimizer_steps"] == 1215720 and contract["endpoint_replays"] == 12, "Step/replay budget changed")
    require(contract["approval_required"] is True and contract["approved"] is False
            and contract["total_arms"] == 6 and contract["policy"]["held_out"] is False,
            "Approval, arm count or split boundary changed")
    require(contract["model_role"] == "observed_time_state_transport_v1"
            and contract["architecture"] == {"hidden_dim": 64, "rank": 8, "scan_mode": "parallel", "time_scale": "frozen_train_median"}, "Model role or architecture changed")
    require(not any(contract["policy"].values()), "Execution policy changed")
    frozen_time = read(ROOT / TIME)["datasets"]
    prior = read(ROOT / PRIOR)
    frozen_data = prior["datasets"]
    for alias, spec in contract["hosts"].items():
        expected = deepcopy(prior["hosts"][alias])
        expected["assigned_datasets"] = ASSIGNMENTS[alias]
        expected["root"] = str(Path(expected["root"]).parent / RUN_NAME)
        expected["source_root"], expected["output_dir"] = expected["root"] + "/source", expected["root"] + "/run"
        expected["tmux"] = RUN_NAME + "_" + alias
        require(spec == expected, "Frozen host path, GPU, runtime or environment changed")
    for data in contract["datasets"]:
        original = next(d for d in frozen_data if d["dataset_id"] == data["dataset_id"])
        duration = next(d for d in frozen_time if d["dataset"] == data["dataset_id"])
        require(data["time_statistics"] == {**{k: duration[k] for k in TIME_KEYS},
            "recompute_absolute_tolerance": 1e-12, "recompute_before_training": True}, "Frozen duration target identity changed")
        for key in ("loader", "optimizer", "inherited_data_identity", "statistics", "quantity_boundaries_all_train_rows", "expected_global_steps"):
            require(data[key] == original[key], "Frozen data/training settings changed: " + key)
        model = data["model"]
        require(model == {"hidden_dim": 64, "quantity_variant": "count_only_log_regression",
            "lambda_log_qty": 1.0, "lambda_tail": 0.0, "time_head_mode": "heteroscedastic_lognormal_duration",
            "time_scale": duration["train_time_scale"], "time_initial_location": duration["train_log_scaled_mean"],
            "time_initial_scale": duration["train_log_scaled_std"], "time_sigma_floor": 0.001,
            "time_head_lr_multiplier": 1.0, "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": duration["time_unit"], "top_code": None}}, "Frozen objective or train-only initialization changed")
        require(data["epochs"] == 120, "Dataset epoch budget changed")
    require(contract.get("design_sha256") == sha_file(ROOT / DESIGN), "Design contract changed")
    if verify_source:
        actual = source_manifest(ROOT, entrypoints=ENTRYPOINTS, extra_files=(DESIGN, OBSERVATION, PRIOR, TIME))
        require(contract["source"]["files"] == actual, "Frozen source closure changed")
        require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source manifest digest mismatch")
        for name, digest in contract["source"]["files"].items():
            path = (ROOT / name).resolve()
            require(path.is_relative_to(ROOT) and path.is_file() and sha_file(path) == digest, "Frozen source changed: " + name)


def freeze(contract_path: Path, output: Path, cpu_path: Path, implementation_path: Path):
    require(not contract_path.exists() and not output.exists(), "New contract and output required")
    contract = build_contract(cpu_path.resolve(), implementation_path.resolve())
    validate_contract(contract)
    output.mkdir(parents=True)
    write_json(contract_path, contract, exclusive=True)
    write_json(output / "approval_template_NOT_APPROVED.json", {"approved": False,
        "contract_sha256": sha_json(contract), "hosts": list(ASSIGNMENTS), "user_instruction": ""}, exclusive=True)
    commands = {}
    for alias, host in contract["hosts"].items():
        base_args = [host["python"], "paper/scripts/run_state_transport_execution.py"]
        shared_args = ["--contract", host["root"] + "/frozen_execution/execution_contract.json",
                       "--approval", host["root"] + "/approval.json", "--host", alias]
        commands[alias] = {"cwd": host["source_root"], "environment": host["environment"],
            "tmux": host["tmux"], "tmux_binary": host["tmux_binary"],
            "qualification_argv": [*base_args, "qualify", *shared_args, "--permit", host["root"] + "/start_permit.json"],
            "training_argv": [*base_args, "train", *shared_args, "--permit", host["root"] + "/training_permit.json"]}
    write_json(output / "prepared_commands_NOT_EXECUTED.json", commands, exclusive=True)
    with (output / "source_bundle.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        sentinel = tarfile.TarInfo("source/sample_data")
        sentinel.type, sentinel.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(sentinel)
        for name in contract["source"]["files"]:
            raw = (ROOT / name).read_bytes(); info = tarfile.TarInfo("source/" + name)
            info.size, info.mode = len(raw), 0o644; archive.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo("frozen_execution/execution_contract.json")
        info.size, info.mode = len(raw), 0o644; archive.addfile(info, io.BytesIO(raw))
    validate_contract(contract)
    result = {"status": "contract_frozen_not_launched", "contract": str(contract_path),
        "contract_sha256": sha_json(contract), "source_files_sha256": contract["source"]["files_sha256"],
        "bundle_sha256": sha_file(output / "source_bundle.tar.gz"), "source_file_count": len(contract["source"]["files"]),
        "approved": False, "source_uploaded": False, "gpu_executed": False,
        "real_data_loaded": False, "held_out_accessed": False, "scheduler_changed": False,
        "remaining_before_execution": ["explicit execution approval", "native CUDA qualification on both assigned hosts"]}
    write_json(output / "preparation_receipt.json", result, exclusive=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("freeze", "start-permit", "training-permit"), default="freeze")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-cost", type=Path)
    parser.add_argument("--implementation", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--qualification-5080", type=Path)
    parser.add_argument("--qualification-5090", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.cpu_cost is None or args.implementation is None:
            parser.error("freeze requires --cpu-cost and --implementation")
        result = freeze(args.contract, args.output, args.cpu_cost, args.implementation)
    else:
        if args.approval is None:
            parser.error("permit creation requires --approval")
        from paper.scripts.run_state_transport_execution import make_start_permit, make_training_permit
        contract, approval = read(args.contract), read(args.approval)
        validate_contract(contract)
        require(not args.output.exists(), "Fresh permit output required")
        if args.action == "start-permit":
            result = make_start_permit(contract, approval)
        else:
            if args.permit is None or args.qualification_5080 is None or args.qualification_5090 is None:
                parser.error("training-permit requires --permit and both --qualification-* receipts")
            result = make_training_permit(contract, approval, read(args.permit),
                {"5080": read(args.qualification_5080), "5090": read(args.qualification_5090)})
        write_json(args.output, result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
