#!/usr/bin/env python3
"""Freeze the three-dataset multi-lag comparison; preparation never launches it."""
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

SCHEMA = "multilag_detail_observed_time_execution_v1"
ARMS = ("titantpp", "titantpp_local_detail", "titantpp_multilag_detail")
ASSIGNMENTS = {"5080": ["yellow_trip_hourly", "intermittent_frozen_5000"], "5090": ["insta_market_basket"]}
RUN_NAME = "multilag_detail_observed_time_seed42_20260921_data_v3"
CORRECTION = {'parent_contract_sha256': 'a79970d201174b05b0b2ad7c471675a5cb1af59afea15cd1aea61be9896c2cf8', 'parent_source_files_sha256': '7dcc577836ce9cf3f7158e431de4113b7c126c4bb6970045fde41980f456c3dd', 'original_started_at_unix': 1789947057.99564, 'original_deadline_unix': 1790119857.99564, 'qualification_optimizer_updates_consumed': 192, 'real_training_optimizer_updates': 0, 'kind': 'data_adapter_integration_fix_before_first_optimizer_step', 'new_native_qualification_rounds_max': 1, 'new_real_training_dispatch_rounds_max': 1, 'real_data_cpu_preflight_required': True, 'architecture_objective_data_gates_unchanged': True, 'shared_deadline_extended': False}
DESIGN = "paper/contracts/hard_lmm_multilag_detail_design_v1.json"
PRIOR = "paper/contracts/nonlinear_episode_parallel_execution_v1.json"
HOST_PRIOR = "paper/contracts/state_transport_observed_time_execution_v1.json"
TIME = "paper/contracts/hard_lmm_frozen_lognormal_duration_v1.json"
OBSERVATION = "paper/contracts/observed_time_joint_training_v1.json"
READINESS = "search_artifacts/multilag_detail_preparation_20260921_v1/host_readiness.json"
ENTRYPOINTS = ("paper/scripts/run_count_aware_tpp_backbone_control.py",
               "paper/scripts/verify_multilag_detail_cost.py",
               "paper/scripts/prepare_multilag_detail_execution.py",
               "paper/scripts/run_multilag_detail_execution.py")
LIMITS = {"qualification_seconds_per_host": 900, "total_wall_seconds": 172800,
          "per_arm_wall_seconds": 86400, "max_concurrent_gpu_jobs_per_host": 1,
          "max_gpu_hours_aggregate": 96, "per_host_output_bytes": 8 * 1024**3,
          "min_free_bytes": 10 * 1024**3}
COST_GATES = {"parameter_ratio_max": 1.10, "median_cuda_step_ratio_max": 1.5,
              "peak_cuda_allocated_ratio_max": 1.25, "device_memory_fraction_max": 0.8}
POLICY = dict.fromkeys(("retry", "resume", "server_transfer", "additional_arms", "additional_seeds",
                       "held_out", "runtime_mutation", "scheduler_changes", "commit_push", "budget_extension"), False)
TIME_KEYS = ("expected_train_target_dt_sha256", "expected_validation_target_dt_sha256",
             "expected_train_targets", "expected_validation_targets", "train_time_scale",
             "train_log_scaled_mean", "train_log_scaled_std")
ARCHITECTURE = {"hidden_dim": 64, "rank": 4, "candidate_lags": [1,2,4,8,16,32,64,128],
                "control_lags": [1]*8, "residual_divisor": 8, "gelu_approximate": "none"}


def read(path):
    return json.loads(Path(path).read_text())


def dataset_specs():
    old, durations = read(ROOT / PRIOR), read(ROOT / TIME)["datasets"]
    result = []
    for dataset_id in ("yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket"):
        original = deepcopy(next(d for d in old["datasets"] if d["dataset_id"] == dataset_id))
        duration = next(d for d in durations if d["dataset"] == dataset_id)
        require(original["inherited_data_identity"]["data"]["sha256"] == duration["data_sha256"]
                and original["loader"]["lookback_weeks"] == duration["lookback"]
                and original["loader"]["max_seq_len"] == duration["max_sequence_length"], "Duration/data/loader identities differ")
        data = {k: original[k] for k in ("dataset_id", "epochs", "expected_global_steps", "inherited_data_identity",
                    "loader", "optimizer", "statistics", "quantity_boundaries_all_train_rows")}
        data["history_boundaries"] = [64, 128]
        data["additional_history_boundaries"] = original["history_boundaries"]
        data["model"] = {"hidden_dim": 64, "quantity_variant": "count_only_log_regression",
            "lambda_log_qty": 1.0, "lambda_tail": 0.0, "time_head_mode": "heteroscedastic_lognormal_duration",
            "time_scale": duration["train_time_scale"], "time_initial_location": duration["train_log_scaled_mean"],
            "time_initial_scale": duration["train_log_scaled_std"], "time_sigma_floor": 0.001,
            "time_head_lr_multiplier": 1.0, "time_observation_contract": {"mode": "positive_integer_round_clamp_v1",
                "unit": duration["time_unit"], "top_code": 30 if dataset_id == "insta_market_basket" else None}}
        data["time_statistics"] = {**{k: duration[k] for k in TIME_KEYS},
            "recompute_absolute_tolerance": 1e-12, "recompute_before_training": True}
        data["potentially_active_lags"] = [d for d in ARCHITECTURE["candidate_lags"] if d < data["loader"]["max_seq_len"] - 1]
        data["window_limitation"] = "Existing maximum sequence includes the target. Availability still depends on each actual observed history; unavailable branches remain inactive in BOTH new arms."
        result.append(data)
    return result


def host_specs():
    hosts, readiness = deepcopy(read(ROOT / HOST_PRIOR)["hosts"]), read(ROOT / READINESS)
    require(readiness["mode"] == "read_only_ssh_no_gpu_workload" and set(readiness["hosts"]) == set(ASSIGNMENTS), "Missing host readiness")
    process_environment, libraries = {}, {}
    for alias, spec in hosts.items():
        observed = readiness["hosts"][alias]
        require(spec["gpu_uuid"] == observed["gpu_uuid"] and spec["python"] == observed["python"]
                and all(observed["runtime"].get(k) == v for k,v in spec["runtime_expected"].items()), "Host identity changed")
        require(not observed["compute_processes"] and observed["disk_available_bytes"] >= LIMITS["min_free_bytes"], "Host not ready")
        spec["assigned_datasets"] = ASSIGNMENTS[alias]
        spec["root"] = str(Path(spec["root"]).parent / RUN_NAME)
        spec["source_root"], spec["output_dir"] = spec["root"] + "/source", spec["root"] + "/run"
        spec["tmux"] = RUN_NAME + "_" + alias
        process_environment[alias] = {"LD_LIBRARY_PATH": observed["nvrtc_library_dir"], "XDG_CACHE_HOME": spec["root"] + "/cache"}
        libraries[alias] = observed["library_sha256"]
        require(len(libraries[alias]) == 2 and all(len(s) == 64 for s in libraries[alias].values()), "NVRTC libraries missing")
        for dataset_id in ASSIGNMENTS[alias]:
            data = next(d for d in dataset_specs() if d["dataset_id"] == dataset_id)
            for key in ("data", "split_manifest"):
                identity = data["inherited_data_identity"][key]
                require(observed["file_sha256"].get(identity["path"]) == identity["sha256"], "Host data digest mismatch")
    return hosts, process_environment, libraries


def build_contract(cpu_path: Path, implementation_path: Path):
    cost, receipt = read(cpu_path), read(implementation_path)
    require(cost["status"] in ("complete", "passed", "cpu_observation_complete") and receipt["verification"]["failed"] == 0
            and receipt["verification"]["passed"] > 0, "Passing local implementation and cost evidence required")
    require(cost.get("device") == "cpu" and cost.get("cuda_qualification") is False
            and cost.get("synthetic_optimizer_updates") == 48
            and len(cost.get("checks", {})) == 9 and all(v is True for v in cost["checks"].values())
            and cost.get("verifier_source_sha256") == sha_file(ROOT / "paper/scripts/verify_multilag_detail_cost.py"),
            "CPU verifier evidence is incomplete or its source changed")
    require(receipt.get("native_execution_verification", {}).get("status") == "passed_synthetic_cpu", "Executor CPU verification missing")
    design, data = read(ROOT / DESIGN), dataset_specs()
    hosts, process_environment, libraries = host_specs()
    files = source_manifest(ROOT, entrypoints=ENTRYPOINTS, extra_files=(DESIGN, PRIOR, HOST_PRIOR, TIME, OBSERVATION, READINESS))
    return {"schema": SCHEMA, "date": "2026-09-21", "status": "conditions_frozen_pending_cuda_qualification",
        "approval_required": True, "approved": False,
        "authorization_note": "User explicitly approved implementation through GPU comparison in this session; bind that instruction in a separate contract-hashed approval receipt before native execution.",
        "model_role": "observed_time_multilag_detail_v1", "seed": 42, "epochs": 120,
        "architecture": deepcopy(ARCHITECTURE), "arms": list(ARMS), "datasets": data, "hosts": hosts,
        "total_arms": 9, "total_optimizer_steps": sum(d["expected_global_steps"] * 3 for d in data), "endpoint_replays": 18,
        "source": {"base_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree": "dirty; file closure and verified isolated bundle are authoritative", "files": files, "files_sha256": sha_json(files)},
        "evidence": {"cpu_cost": str(cpu_path.relative_to(ROOT)), "cpu_cost_sha256": sha_file(cpu_path),
            "implementation": str(implementation_path.relative_to(ROOT)), "implementation_sha256": sha_file(implementation_path),
            "host_readiness": READINESS, "host_readiness_sha256": sha_file(ROOT / READINESS)},
        "design_sha256": sha_file(ROOT / DESIGN), "limits": deepcopy(LIMITS), "policy": deepcopy(POLICY),
        "cost_gates": deepcopy(COST_GATES),
        "acceptance": design["prospective_comparison"]["screening_against_each_of_B_and_local_control_on_each_dataset"],
        "comparison": {"scope": "single-seed reused-validation exploration; no universal or benchmark superiority claim",
            "same_dataset_same_host": True, "same_batch_exposure": "identical per-epoch content/order hashes and counts",
            "arm_order": list(ARMS), "initialization": "same common B and caller RNG, equal new tensors, zero U",
            "selector": "strict earliest finite validation raw RMSE minimum; selected quantity/time/strata from one checkpoint",
            "goal_fully_met_rule": design["prospective_comparison"]["goal_fully_met"],
            "Instacart": "Includes all three fresh arms; original max_seq_len64 and day top_code30 retained. Lag64/128 cannot be active. No input-window expansion.",
            "empty_bins": "unavailable, not pass; short Instacart history makes >64 groups empty; primary quantity bins are separately audited"},
        "process_environment": process_environment, "library_sha256": libraries, "implementation_correction": deepcopy(CORRECTION),
        "qualification": {"executed": False, "entrypoint": ENTRYPOINTS[-1], "native_cuda_required": True,
            "workload": "synthetic correctness plus batch128 lengths64/256; exact update count bound by verifier source",
            "real_data_loaded": False},
        "launch": {"root_must_be_new": True, "source_sync": "new isolated roots, checksum verified; no deletion/shared-runtime mutation",
            "budget_origin": "original v1 native qualification start, unchanged by corrections; shared 48-hour termination ceiling, not an ETA",
            "absolute_deadline": CORRECTION["original_deadline_unix"], "estimated_completion_hours": None,
            "fixed_started_at_unix": CORRECTION["original_started_at_unix"], "fixed_deadline_unix": CORRECTION["original_deadline_unix"],
            "deadline_action": "stop owned process group only; preserve outputs; incomplete is inconclusive; no automatic retry/resume/extension",
            "supervisor_entrypoint": ENTRYPOINTS[-1], "training_cli_direct_launch_allowed": False,
            "server_assignment": ASSIGNMENTS}}


def validate_contract(contract, *, verify_source=True):
    require(contract.get("schema") == SCHEMA and contract.get("model_role") == "observed_time_multilag_detail_v1", "Wrong contract/role")
    require(contract.get("arms") == list(ARMS) and contract.get("architecture") == ARCHITECTURE
            and contract.get("seed") == 42 and contract.get("epochs") == 120, "Frozen architecture/training changed")
    require(contract.get("limits") == LIMITS and contract.get("policy") == POLICY
            and contract.get("cost_gates") == COST_GATES, "Resource policy changed")
    require(contract.get("datasets") == dataset_specs(), "Frozen dataset/training/statistics changed")
    require(contract.get("implementation_correction") == CORRECTION
            and contract.get("launch", {}).get("fixed_started_at_unix") == CORRECTION["original_started_at_unix"]
            and contract.get("launch", {}).get("fixed_deadline_unix") == CORRECTION["original_deadline_unix"],
            "Correction scope or original deadline changed")
    hosts, process_environment, libraries = host_specs()
    require(contract.get("hosts") == hosts and contract.get("process_environment") == process_environment
            and contract.get("library_sha256") == libraries, "Host/environment/library identities changed")
    require(contract.get("acceptance") == read(ROOT / DESIGN)["prospective_comparison"]["screening_against_each_of_B_and_local_control_on_each_dataset"], "Prospective gates changed")
    steps = sum(math.ceil(d["inherited_data_identity"]["populations"]["train"]["target_count"] / d["loader"]["batch_size"]) * 120 * 3 for d in contract["datasets"])
    require(steps == contract.get("total_optimizer_steps") == 6816240 and contract.get("total_arms") == 9
            and contract.get("endpoint_replays") == 18, "Step/arm/replay scope changed")
    require(contract.get("approved") is False and contract.get("approval_required") is True
            and contract.get("design_sha256") == sha_file(ROOT / DESIGN), "Approval or design provenance changed")
    require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source digest inconsistent")
    if verify_source:
        actual = source_manifest(ROOT, entrypoints=ENTRYPOINTS, extra_files=(DESIGN, PRIOR, HOST_PRIOR, TIME, OBSERVATION, READINESS))
        require(actual == contract["source"]["files"], "Frozen source changed")
        require(sha_file(ROOT / READINESS) == contract["evidence"]["host_readiness_sha256"], "Readiness evidence changed")


def freeze(contract_path, output, cpu_path, implementation_path):
    require(not contract_path.exists() and not output.exists(), "Fresh contract/output required")
    contract = build_contract(cpu_path.resolve(), implementation_path.resolve())
    validate_contract(contract)
    output.mkdir(parents=True)
    write_json(contract_path, contract, exclusive=True)
    with (output / "source_bundle.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        sentinel = tarfile.TarInfo("source/sample_data"); sentinel.type, sentinel.mode = tarfile.DIRTYPE, 0o755; archive.addfile(sentinel)
        for name in contract["source"]["files"]:
            raw = (ROOT / name).read_bytes(); info = tarfile.TarInfo("source/" + name)
            info.size, info.mode = len(raw), 0o644; archive.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode()
        info = tarfile.TarInfo("frozen_execution/execution_contract.json"); info.size, info.mode = len(raw), 0o644
        archive.addfile(info, io.BytesIO(raw))
    result = {"status": "contract_frozen_not_launched", "contract_sha256": sha_json(contract),
        "source_files_sha256": contract["source"]["files_sha256"], "bundle_sha256": sha_file(output / "source_bundle.tar.gz"),
        "gpu_executed": False, "held_out_accessed": False, "source_uploaded": False}
    write_json(output / "preparation_receipt.json", result, exclusive=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-cost", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.contract, args.output, args.cpu_cost, args.implementation)))


if __name__ == "__main__":
    main()
