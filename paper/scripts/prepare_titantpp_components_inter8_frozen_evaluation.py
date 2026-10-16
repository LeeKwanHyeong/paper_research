"""Bind the explicitly approved native5090 Intermittent42/52 subset.

The original36 design and global completion gate are preserved. This adapter
reads immutable evidence metadata and hashes opaque bytes; it does not retrieve
files, deserialize data/models, infer predictions, or create execution permits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from paper.scripts import prepare_titantpp_components_frozen_evaluation as base
from paper.scripts.evaluate_titantpp_components_frozen import validate_runtime_spec


KIND = "titantpp_components_inter8_frozen_evaluation_v1"
COHORT = "components_inter42_52_source131"
DATASET = "intermittent_frozen_5000"
DATASETS = (DATASET,)
SEEDS = (42, 52)
ARMS = base.ARMS
SOURCE = base.SOURCE
PARENT = base.PARENT
REVISION = base.REVISION
ORIGINAL_DESIGN_CLOSURE = "33aea609d14b796997d874b7ea82d03046aa48b5ab4e2747ef73de292ff6fdcd"
AUTHORIZED_LOCAL_ROOT = "/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_components_inter8_test_20261014_v1"
AUTHORIZED_REMOTE_ROOT = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090"
require, read, sha, canonical, checked = base.require, base.read, base.sha, base.canonical, base.checked


def expected_ids():
    return {f"{DATASET}__{seed}__{arm}" for seed in SEEDS for arm in ARMS}


def check_approval(root, bindings):
    approval = read(checked(root, bindings["partial_approval_path"], bindings["partial_approval_sha256"]))
    require(approval.get("schema") == "titantpp_components_inter8_partial_approval_v1"
            and approval.get("partial_cohort_explicitly_authorized") is True
            and approval.get("evaluation_scope") == "test_partial_followup"
            and approval.get("cohort") == COHORT and approval.get("host") == "5090"
            and type(approval.get("expected_condition_count")) is int
            and approval["expected_condition_count"] == 8
            and approval.get("job_ids") == sorted(expected_ids())
            and approval.get("parent_contract_sha256") == PARENT
            and approval.get("frozen_source_closure") == SOURCE
            and approval.get("new_training") is False
            and approval.get("original36_global_gate_modified") is False,
            "Inter8 explicit partial approval missing or changed")
    instruction = approval.get("user_instruction")
    require(isinstance(instruction, str) and bool(instruction.strip())
            and approval.get("user_instruction_sha256") == hashlib.sha256(instruction.encode()).hexdigest(),
            "Original user instruction SHA differs")
    require(approval.get("authorization_root") == AUTHORIZED_REMOTE_ROOT
            and approval.get("allowed_local_binding_root") == AUTHORIZED_LOCAL_ROOT
            and str(Path(root).resolve()) in {AUTHORIZED_LOCAL_ROOT, AUTHORIZED_REMOTE_ROOT},
            "Partial approval root is outside the authorized local/remote pair")
    return approval


def verify_partial_training(root, bindings, training, jobs):
    """Preserve the incomplete35/36 report while requiring all native8 originals."""
    monitor = read(checked(root, bindings["training_monitor_path"], bindings["training_monitor_sha256"]))
    require(monitor.get("parent_contract_sha256") == PARENT
            and monitor.get("all_training_terminal_verified") is False
            and monitor.get("held_out_test_evaluated") is False
            and monitor.get("totals") == {"total": 36, "completed": 35, "running": 1,
                                           "waiting": 0, "unconfirmed": 0,
                                           "all_training_terminal_verified": False},
            "Original36 incomplete observation must remain35/1, never global completion")
    parent = monitor["groups"]["parent_5090"]
    require(parent.get("status") == "planned_queue_exclusion_terminal_verified"
            and parent.get("host") == "5090" and parent.get("contract_sha256") == PARENT
            and parent.get("completed") == 8 and parent.get("total") == 8
            and parent.get("identity_verified") is True
            and parent.get("owner_process_absence_verified") is True
            and parent.get("administrative_dispatcher_actual_exit") == 1
            and parent.get("parent_original12_complete") is False
            and parent.get("pipeline_wrapper_OS_exit_directly_observed") is False
            and parent.get("original_failure_and_claim_preserved") is True
            and parent.get("scientific_failure_inferred") is False,
            "Native5090 partial terminal8 is not verified")
    rows = parent.get("jobs", [])
    require(len(rows) == 8 and {row["job"]["id"] for row in rows} == expected_ids(),
            "Parent terminal evidence must contain exactly8 selected fits")
    lookup = {job["id"]: job for job in jobs}
    for row in rows:
        job = lookup[row["job"]["id"]]
        expected = {"id": job["id"], "host": "5090", "dataset": DATASET,
                    "seed": job["seed"], "arm": job["arm"]}
        require(row["job"] == expected and row.get("status") == "complete"
                and row.get("selected_checkpoint_file_present") is True
                and row.get("remote_all_terminal_file_hashes_verified") is True
                and row.get("terminal_manifest_raw_sha256") == job["terminal_manifest_sha256"]
                and row.get("worker_exit_raw_sha256") == job["worker_exit_sha256"],
                "Partial worker/binary proof differs from recovered original")
    snapshot = read(checked(root, bindings["parentterminal_evidence_path"], bindings["parentterminal_evidence_sha256"]))
    require(bindings["parentterminal_evidence_sha256"] == parent.get("snapshot_raw_sha256")
            and snapshot.get("host") == "5090" and snapshot.get("contract_sha256") == PARENT
            and snapshot.get("source_files_sha256") == SOURCE
            and snapshot.get("root") == training["hosts"]["5090"]["root"] == parent.get("root")
            and snapshot.get("gpu_uuid") == base.GPU_UUIDS["5090"]
            and type(snapshot.get("observed_unix")) in (int, float)
            and math.isfinite(snapshot["observed_unix"])
            and snapshot["observed_unix"] == parent.get("observed_unix"),
            "Native parent snapshot byte/identity/time differs")
    # This is pure stdlib evidence validation; no observer or remote call occurs.
    from paper.scripts.titantpp_components_parallel_monitor_v2 import original, verify_owner_absence
    files = original.decode_snapshot(snapshot, training, "5090")
    phase_exit = json.loads(files["phases/train_exit.json"])
    require(type(phase_exit.get("os_exit_code")) is int and phase_exit["os_exit_code"] == 1
            and phase_exit.get("phase") == "train" and phase_exit.get("contract_sha256") == PARENT,
            "Original administrative dispatcher actual exit1 missing")
    for job in jobs:
        worker_bytes = files["worker_exits/" + job["id"] + ".json"]
        terminal_bytes = files["run/" + job["id"] + "/terminal_manifest.json"]
        require(hashlib.sha256(worker_bytes).hexdigest() == job["worker_exit_sha256"]
                and hashlib.sha256(terminal_bytes).hexdigest() == job["terminal_manifest_sha256"],
                "Recovered worker/terminal bytes differ from native snapshot")
        terminal = json.loads(terminal_bytes)
        witness = snapshot.get("terminal_hashes", {}).get(job["id"], {})
        require(witness.get("all_files_verified") is True
                and set(witness.get("files", {})) == set(terminal["files"])
                and all(value is True for value in witness["files"].values()),
                "Native terminal binary witness is incomplete")
    probe = verify_owner_absence(snapshot, training, "5090", [row["job"] for row in rows])
    require(probe["owner_process_probe_count"] == parent.get("owner_process_probe_count") == 10
            and probe["owner_process_observations_sha256"] == parent.get("owner_process_observations_sha256"),
            "Native owner absence proof differs")
    return monitor


def bind_job(root, job):
    require(job.get("cohort") == COHORT and job.get("study") == "components"
            and job.get("id") in expected_ids() and job.get("dataset") == DATASET
            and type(job.get("seed")) is int and job["seed"] in SEEDS,
            "Foreign condition cannot enter Inter8 partial cohort")
    # Only the operational cohort tag adapts; source/selector/files stay exact.
    bound = base.bind_job(root, {**job, "cohort": base.COHORT})
    return {**bound, "cohort": COHORT}


def build_contract(root, design, bindings, runtimes):
    require(design.get("schema") == "titantpp_components_final_Test_design_v1"
            and design.get("fresh_Test_condition_count") == 36
            and canonical(design) == ORIGINAL_DESIGN_CLOSURE,
            "Original36 frozen selection/metric design changed")
    require(bindings.get("schema") == "titantpp_components_inter8_checkpoint_bindings_v1",
            "Foreign Inter8 bindings schema")
    approval = check_approval(root, bindings)
    training = read(checked(root, bindings["training_contract_path"], bindings["training_contract_sha256"]))
    require(canonical(training) == PARENT, "Original source131 training contract changed")
    definitions = {definition["dataset_id"]: definition for definition in training["datasets"]}
    require(set(definitions) == set(base.DATASETS), "Original three-dataset training lineage changed")
    source = bindings["sources"]["components"]
    require(set(bindings["sources"]) == {"components"}
            and source["closure"] == SOURCE == canonical(source["files"])
            and source["revision"] == REVISION and len(source["files"]) == 131,
            "Frozen source131 manifest changed")
    source_root = base.confined(root, source["root"])
    for name, value in source["files"].items():
        checked(source_root, name, value)
    jobs = bindings["jobs"]
    require(len(jobs) == 8 and {job["id"] for job in jobs} == expected_ids(),
            "Need exactly8 unique Intermittent42/52 conditions")
    bound = [bind_job(root, job) for job in jobs]
    monitor = verify_partial_training(root, bindings, training, bound)
    require(set(bindings["datasets"]) == {DATASET}, "Foreign partial dataset set")
    spec = bindings["datasets"][DATASET]
    definition, fixed = spec["definition"], design["datasets"][DATASET]
    require(canonical(definition) == canonical(definitions[DATASET]),
            "Frozen dataset statistics/model/preprocessing differ from training contract")
    inherited = definition["inherited_data_identity"]
    require(spec["data_sha256"] == fixed["full_data_byte_sha256"] == inherited["data"]["sha256"]
            and spec["training_data_sha256"] == inherited["data"]["sha256"]
            and spec["split_manifest_sha256"] == fixed["split_manifest_sha256"] == inherited["split_manifest"]["sha256"]
            and type(spec["expected_test_count"]) is int
            and spec["expected_test_count"] == fixed["expected_Test_target_count"] == base.TEST_COUNTS[DATASET]
            and spec["tail_threshold"] == 187,
            "Partial data identity/count/tail differs from frozen lineage")
    for path_key, sha_key in (("data_path", "data_sha256"), ("training_data_path", "training_data_sha256"),
                              ("split_manifest_path", "split_manifest_sha256")):
        checked(root, spec[path_key], spec[sha_key])
    for split in ("validation", "test"):
        population = spec[split + "_population"]
        require(type(population.get("count")) is int and population["count"] > 0, "Population not bound")
        for key in ("target_identity_sha256", "truth_sha256", "time_sha256"):
            base.digest(population.get(key), split + ":" + key)
    require(spec["test_population"]["count"] == spec["expected_test_count"]
            and spec["validation_population"]["count"] == inherited["populations"]["validation"]["target_count"]
            and all(job["validation_reference"]["count"] == spec["validation_population"]["count"] for job in bound),
            "Partial target population differs")
    require(set(runtimes) == {"5090"}, "Only native5090 Runtime is approved")
    validate_runtime_spec(runtimes["5090"], "5090")
    integrity = bindings["evaluator_integrity"]
    require(integrity.get("files") and integrity["closure"] == canonical(integrity["files"]),
            "Operational Inter8 evaluator seal missing")
    for name, value in integrity["files"].items():
        checked(root, name, value)
    contract = {"schema": KIND, "cohort": COHORT, "evaluation_scope": "test_partial_followup",
                "metrics_schema": "quantity_time_tail_v1", "expected_seeds": list(SEEDS),
                "partial_cohort_explicitly_authorized": True,
                "source_training_cohort_count": 36, "expected_condition_count": 8, "expected_jobs": 8,
                "original36_all_training_terminal_verified": monitor["all_training_terminal_verified"],
                "original36_global_gate_modified": False, "original36_design_sha256": canonical(design),
                "sources": bindings["sources"], "datasets": bindings["datasets"], "jobs": bound,
                "runtimes": runtimes, "evaluator_integrity": integrity,
                "validation_replay_tolerance": {"absolute": 1e-5, "relative": 1e-5},
                "training_monitor_path": bindings["training_monitor_path"],
                "training_monitor_sha256": bindings["training_monitor_sha256"],
                "training_contract_path": bindings["training_contract_path"],
                "training_contract_sha256": bindings["training_contract_sha256"],
                "parentterminal_evidence_path": bindings["parentterminal_evidence_path"],
                "parentterminal_evidence_sha256": bindings["parentterminal_evidence_sha256"],
                "partial_approval_path": bindings["partial_approval_path"],
                "partial_approval_sha256": bindings["partial_approval_sha256"],
                "evaluation_root_mapping": {"local_preparation_root": AUTHORIZED_LOCAL_ROOT,
                                             "authorized_remote_root": AUTHORIZED_REMOTE_ROOT},
                "test_is_independent": False, "heldout_exposure_history_retained": True,
                "automatic_retry": False, "new_training": False, "checkpoint_reselection": False,
                "authorization": {"permit_required": True, "Test_authorized_by_binding": False,
                                  "partial_cohort_user_authorized": True,
                                  "user_instruction": approval["user_instruction"],
                                  "user_instruction_sha256": approval["user_instruction_sha256"]}}
    contract["contract_sha256"] = canonical(contract)
    return contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--runtimes", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = build_contract(args.root, read(args.design), read(args.bindings), read(args.runtimes))
    destination = args.root / "contract.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"contract_path": str(destination), "contract_sha256": contract["contract_sha256"],
                      "raw_sha256": sha(destination), "job_count": 8,
                      "original36_all_training_terminal_verified": False, "Test_permit_created": False}))


if __name__ == "__main__":
    main()
