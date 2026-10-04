"""Frozen six-encoder comparison. Preparing a contract grants no GPU authority."""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tarfile

from paper.scripts.observed_slot_parallel_common import require, sha_file, sha_json, source_manifest, write_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "local_detail_benchmark_observed_time_v4"
PARENT = "paper/contracts/local_gate_observed_time_execution_v1.json"
PARENT_SHA = "958467da6d55b362fe7965c01056abe2532e8ffba4dede996b46a6162f8ead22"
DESIGN = "paper/contracts/hard_lmm_multilag_detail_design_v1.json"
RUN_NAME = "local_detail_benchmark_intermittent_recovery_seed42_e300_20260923_v4"
RECOVERY = "paper/contracts/local_detail_benchmark_intermittent_recovery_lineage_v4.json"
ARMS = ("titantpp", "titantpp_local_detail", "rmtpp", "thp", "nhp", "sahp")
ASSIGNMENTS = {"5080": ["intermittent_frozen_5000"], "5090": []}
TRAINING = {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "monitor": "validation_raw_quantity_rmse", "tie": "strict_earliest_finite_minimum",
            "seed": 42, "batch_size": 128, "warm_start": False,
            "early_stopping": "epoch >= 40 and epoch - earliest_best_epoch >= 40"}
LIMITS = {"total_wall_seconds": 864000, "per_arm_wall_seconds": 432000,
          "qualification_seconds_per_host": 1800, "max_concurrent_gpu_jobs_per_host": 1,
          "max_gpu_hours_aggregate": 480, "per_host_output_bytes": 16 * 1024**3,
          "min_free_bytes": 20 * 1024**3, "max_file_bytes": 64 * 1024**2}
POLICY = {"retry": False, "resume": False, "additional_seeds": False, "held_out": False,
          "runtime_mutation": False, "budget_extension": False, "server_transfer": False,
          "scheduler_changes": False, "commit_push": False,
          "other_gpu_processes": "observe_without_stopping",
          "startup_min_free_gpu_bytes": 2 * 1024**3}
ENTRYPOINTS = ("paper/scripts/run_local_detail_benchmark.py",
               "paper/scripts/local_detail_benchmark_contract.py",
               "paper/scripts/verify_local_detail_benchmark.py")
EXTRA = (PARENT, DESIGN, RECOVERY)


def read(path):
    return json.loads(Path(path).read_text())


def parent():
    value = read(ROOT / PARENT)
    require(sha_json(value) == PARENT_SHA, "Frozen parent contract changed")
    return value


def protocol():
    p = parent()
    datasets = deepcopy(p["datasets"])
    for d in datasets:
        batches = (d["inherited_data_identity"]["populations"]["train"]["target_count"] + 127) // 128
        d["epochs"] = 300
        d["expected_global_steps"] = batches * 300
        d["steps_per_epoch"] = batches
    hosts, environment = deepcopy(p["hosts"]), deepcopy(p["process_environment"])
    for host, spec in hosts.items():
        spec["root"] = str(Path(spec["root"]).parent / RUN_NAME)
        spec["source_root"] = spec["root"] + "/source"
        spec["output_dir"] = spec["root"] + "/run"
        spec["tmux"] = RUN_NAME + "_" + host
        spec["assigned_datasets"] = list(ASSIGNMENTS[host])
        environment[host]["XDG_CACHE_HOME"] = spec["root"] + "/cache"
    return {
        "schema": SCHEMA, "approved": False, "approval_required": True,
        "model_role": "per_arm_observed_time_roles", "arms": list(ARMS), "seed": 42,
        "training": TRAINING, "datasets": datasets, "hosts": hosts,
        "process_environment": environment, "library_sha256": p["library_sha256"],
        "limits": LIMITS, "policy": POLICY, "total_arms": 18, "endpoint_replays": 36,
        "totals_scope": "original three-dataset campaign; only recovery.new_training_arms execute in this contract",
        "recovery": read(ROOT / RECOVERY),
        "maximum_optimizer_steps": sum(d["expected_global_steps"] for d in datasets) * len(ARMS),
        "actual_steps": "sum(completed_epochs * steps_per_epoch); early stopping permits unequal epoch counts",
        "parent_contract_sha256": PARENT_SHA, "design_sha256": sha_file(ROOT / DESIGN),
        "architecture": {"candidate": "titantpp_local_detail", "baseline": "titantpp",
                         "hidden_dim": 64, "rank": 4, "lags": [1]*8,
                         "availability_lags": [1,2,4,8,16,32,64,128], "residual_divisor": 8,
                         "additional_gate": False, "parameter_matched": False},
        "comparison": {
            "primary": "local detail versus B; benchmark encoder comparisons under shared heads",
            "selector": "all selected quantity/time/stratum metrics use the SAME raw-RMSE-selected checkpoint",
            "batch_rule": "identical train batch prefix through the minimum completed epoch of each pair",
            "initialization": "B/local shared tensors and post-build RNG equal; other encoders seeded separately, not tensor-matched",
            "historical_results": "exploratory context only; legacy head and 120-epoch runs not final-table rows",
            "scope": "seed42 validation; neither native full-model benchmark nor universal superiority",
            "capacity": "report parameters/memory/step time; B/local contrast alone does not isolate extra capacity",
            "stability": "last30 separately plus fixed first40-epoch window; differing stop epochs disclosed",
            "acceptance": p["acceptance"],
            "failed_gates": "retain each dataset/metric failure; do not exclude datasets or revise thresholds",
            "tuning": "no model-specific hyperparameter search in this controlled comparison"},
        "launch": {"clock_origin": "retain original v2 start permit clock; explicit 5080 Intermittent recovery only",
                   "fixed_started_at_unix": 1790080128.1799538,
                   "fixed_deadline_unix": 1790944128.1799538,
                   "allowed_execution_hosts": ["5080"],
                   "supersedes_contract_sha256": "0dfb94f82f6b3b88bce779deca3987fbc8cbb96ba9ef66c8b6247bbe7c5afca8",
                   "correction": "validate stored deadlines by the original addition, not floating-point subtraction; expired deadlines still rejected",
                   "requires_both_native_qualifications": False,
                   "qualification_scope": "each host must pass its own qualification before its assigned training",
                   "staggered_start": "5080 Intermittent only; preserve completed Taxi and running 5090 v3 Instacart",
                   "root_must_be_new": True,
                   "preserve_existing_experiments": True,
                   "old_5090_dependency_root": p["hosts"]["5090"]["root"],
                   "old_5090_dependency_contract_sha256": PARENT_SHA,
                   "deadline_action": "stop only this campaign's owned worker group; preserve files; no automatic retry"}}


def validate_contract(c, *, verify_source=True):
    for key, value in protocol().items():
        require(c.get(key) == value, "Frozen benchmark protocol changed: " + key)
    source = c["source"]
    require(len(source["base_git_revision"]) == 40 and sha_json(source["files"]) == source["files_sha256"],
            "Source closure inconsistent")
    if verify_source:
        require(source_manifest(ROOT, entrypoints=ENTRYPOINTS, extra_files=EXTRA) == source["files"],
                "Frozen benchmark source changed")
    require(c["verification"]["status"] == "passed_synthetic_cpu" and c["verification"]["failed"] == 0,
            "Local integration verification missing")
    require(c["cost_estimate"]["native_current_head_benchmark_timing_measured"] is False,
            "Preparation must distinguish historical estimates from native measurements")


def freeze(contract_path, output, verification_path, cost_path):
    require(not contract_path.exists() and not output.exists(), "New contract and bundle paths required")
    verification, cost = read(verification_path), read(cost_path)
    files = source_manifest(ROOT, entrypoints=ENTRYPOINTS, extra_files=EXTRA)
    require(verification.get("tested_source_files_sha256") == sha_json(files),
            "Verification predates the current source closure")
    require(verification.get("passed", 0) > 0, "Passing tests required")
    c = {**protocol(), "status": "frozen_pending_execution_approval_and_native_qualification",
         "verification": verification, "cost_estimate": cost,
         "source": {"base_git_revision": subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
                    "working_tree": "isolated content-hashed closure; existing dirty worktree preserved",
                    "files": files, "files_sha256": sha_json(files)},
         "evidence_sha256": {"verification": sha_file(verification_path), "cost": sha_file(cost_path)}}
    validate_contract(c)
    output.mkdir(parents=True)
    write_json(contract_path, c, exclusive=True)
    with tarfile.open(output / "source_bundle.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("source/sample_data"); entry.type = tarfile.DIRTYPE; entry.mode = 0o755
        archive.addfile(entry)
        for name in files:
            raw = (ROOT / name).read_bytes()
            entry = tarfile.TarInfo("source/" + name); entry.size = len(raw); entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(raw))
        raw = (json.dumps(c, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
        entry = tarfile.TarInfo("frozen_execution/execution_contract.json"); entry.size = len(raw)
        archive.addfile(entry, io.BytesIO(raw))
    receipt = {"status": "prepared_not_launched", "contract_sha256": sha_json(c),
               "source_files_sha256": sha_json(files), "bundle_sha256": sha_file(output / "source_bundle.tar.gz"),
               "gpu_executed": False, "deployment_performed": False,
               "new_training_arms": sum(len(c["hosts"][h]["assigned_datasets"]) * len(ARMS)
                   for h in c["launch"].get("allowed_execution_hosts", c["hosts"]))}
    write_json(output / "preparation_receipt.json", receipt, exclusive=True)
    return receipt


def main():
    p = argparse.ArgumentParser()
    for name in ("contract", "output", "verification", "cost"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(freeze(a.contract, a.output, a.verification, a.cost)))


if __name__ == "__main__":
    main()
