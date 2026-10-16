"""Bind the explicitly approved native5080 Taxi/RAF24 subset.

The original36 design and global evaluation gate are preserved. This adapter
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


KIND = "titantpp_components_taxi_raf24_frozen_evaluation_v1"
COHORT = "components_taxi_raf24_source131"
DATASETS = ("yellow_trip_hourly", "raf_spare_parts")
SEEDS = (42, 52, 62)
ARMS = base.ARMS
SOURCE = base.SOURCE
PARENT = base.PARENT
REVISION = base.REVISION
ORIGINAL_DESIGN_CLOSURE = "33aea609d14b796997d874b7ea82d03046aa48b5ab4e2747ef73de292ff6fdcd"
AUTHORIZED_LOCAL_ROOT = "/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1"
AUTHORIZED_REMOTE_ROOT = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_taxi_raf24_test_20261014_v1_5080"
require, read, sha, canonical, checked = base.require, base.read, base.sha, base.canonical, base.checked


def expected_ids():
    return {f"{dataset}__{seed}__{arm}" for dataset in DATASETS for seed in SEEDS for arm in ARMS}


def check_approval(root, bindings):
    approval = read(checked(root, bindings["partial_approval_path"], bindings["partial_approval_sha256"]))
    require(approval.get("schema") == "titantpp_components_taxi_raf24_partial_approval_v1"
            and approval.get("partial_cohort_explicitly_authorized") is True
            and approval.get("evaluation_scope") == "test_taxi_raf24_followup"
            and approval.get("cohort") == COHORT and approval.get("host") == "5080"
            and type(approval.get("expected_condition_count")) is int
            and approval["expected_condition_count"] == 24
            and approval.get("job_ids") == sorted(expected_ids())
            and approval.get("parent_contract_sha256") == PARENT
            and approval.get("frozen_source_closure") == SOURCE
            and approval.get("new_training") is False
            and approval.get("original36_global_gate_modified") is False,
            "Taxi/RAF24 explicit partial approval missing or changed")
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
    """Verify historical original5080 closure without inventing new PID probes."""
    from paper.scripts import evaluate_titantpp_components_taxi_raf24_frozen as evaluator
    c = {key: bindings[key] for key in (
        "training_contract_path", "training_contract_sha256",
        "training_monitor_path", "training_monitor_sha256",
        "parentterminal_evidence_path", "parentterminal_evidence_sha256",
        "partial_approval_path", "partial_approval_sha256")}
    c.update(original36_all_training_terminal_verified=True,
             original36_global_gate_modified=False, source_training_cohort_count=36,
             expected_jobs=24, partial_cohort_explicitly_authorized=True)
    evaluator.verify_partial_metadata(root, c)
    snapshot = read(checked(root, bindings["parentterminal_evidence_path"],
                            bindings["parentterminal_evidence_sha256"]))
    for job in jobs:
        require(snapshot["files"]["worker_exits/" + job["id"] + ".json"]["raw_sha256"]
                == job["worker_exit_sha256"]
                and snapshot["files"]["run/" + job["id"] + "/terminal_manifest.json"]["raw_sha256"]
                == job["terminal_manifest_sha256"],
                "Recovered worker/terminal bytes differ from native snapshot")
    return read(checked(root, bindings["training_monitor_path"], bindings["training_monitor_sha256"]))


def bind_job(root, job):
    require(job.get("cohort") == COHORT and job.get("study") == "components"
            and job.get("id") in expected_ids() and job.get("dataset") in DATASETS
            and type(job.get("seed")) is int and job["seed"] in SEEDS,
            "Foreign condition cannot enter Taxi/RAF24 partial cohort")
    # Only the operational cohort tag adapts; source/selector/files stay exact.
    bound = base.bind_job(root, {**job, "cohort": base.COHORT})
    return {**bound, "cohort": COHORT}


def build_contract(root, design, bindings, runtimes):
    require(design.get("schema") == "titantpp_components_final_Test_design_v1"
            and design.get("fresh_Test_condition_count") == 36
            and canonical(design) == ORIGINAL_DESIGN_CLOSURE,
            "Original36 frozen selection/metric design changed")
    require(bindings.get("schema") == "titantpp_components_taxi_raf24_checkpoint_bindings_v1",
            "Foreign Taxi/RAF24 bindings schema")
    approval = check_approval(root, bindings)
    training = read(checked(root, bindings["training_contract_path"], bindings["training_contract_sha256"]))
    require(canonical(training) == PARENT, "Original source131 training contract changed")
    definitions = {definition["dataset_id"]: definition for definition in training["datasets"]}
    require(set(definitions) == set(base.DATASETS), "Original three-dataset training lineage changed")
    require(len(training.get("jobs", [])) == 36
            and {job["id"] for job in training["jobs"] if job.get("host") == "5080"} == expected_ids(),
            "Original training registry must bind all36 and exact24 native5080 fits")
    source = bindings["sources"]["components"]
    require(set(bindings["sources"]) == {"components"}
            and source["closure"] == SOURCE == canonical(source["files"])
            and source["revision"] == REVISION and len(source["files"]) == 131,
            "Frozen source131 manifest changed")
    source_root = base.confined(root, source["root"])
    for name, value in source["files"].items():
        checked(source_root, name, value)
    jobs = bindings["jobs"]
    require(len(jobs) == 24 and {job["id"] for job in jobs} == expected_ids(),
            "Need exactly24 unique Taxi/RAF conditions")
    bound = [bind_job(root, job) for job in jobs]
    monitor = verify_partial_training(root, bindings, training, bound)
    require(set(bindings["datasets"]) == set(DATASETS), "Foreign dataset set")
    for name, spec in bindings["datasets"].items():
        fixed = design["datasets"][name]
        require(spec["data_sha256"] == fixed["full_data_byte_sha256"]
                and spec["tail_threshold"] == base.TAILS[name]
                and spec["expected_test_count"] == fixed["expected_Test_target_count"]
                and spec["split_manifest_sha256"] == fixed["split_manifest_sha256"], "Data identity changed")
        # Byte hashing does not deserialize or reveal the full chronological data.
        checked(root, spec["data_path"], spec["data_sha256"])
        checked(root, spec["training_data_path"], spec["training_data_sha256"])
        checked(root, spec["split_manifest_path"], spec["split_manifest_sha256"])
        definition = spec["definition"]
        require(canonical(definition) == canonical(definitions[name]),
                "Train statistics, architecture or preprocessing changed")
        inherited = definition["inherited_data_identity"]
        full_identity = definition["parent_data_identity"] if name == "raf_spare_parts" else inherited
        require(spec["data_sha256"] == full_identity["data"]["sha256"]
                and spec["training_data_sha256"] == inherited["data"]["sha256"]
                and spec["split_manifest_sha256"] == inherited["split_manifest"]["sha256"]
                and type(spec["expected_test_count"]) is int
                and spec["expected_test_count"] == base.TEST_COUNTS[name],
                "Dataset bytes/count differ from frozen training lineage")
        require(definition["dataset_id"] == name
                and definition["model"]["time_observation_contract"] == fixed["time_observation_contract"]
                and definition["model"]["time_scale"] == fixed["time_scale"]
                and definition["quantity_boundaries_all_train_rows"][-1] == base.TAILS[name]
                and definition["loader"]["batch_size"] == 128
                and definition["loader"]["max_seq_len"] == fixed["max_seq_len_including_target"]
                and definition["loader"]["lookback_weeks"] == fixed["lookback_seq_units"], "Frozen preprocessing changed")
        for split in ("validation", "test"):
            population = spec[split + "_population"]
            require(type(population.get("count")) is int and population["count"] > 0, "Population not bound")
            for key in ("target_identity_sha256", "truth_sha256", "time_sha256"):
                base.digest(population.get(key), split + ":" + key)
        require(spec["test_population"]["count"] == spec["expected_test_count"], "Test count changed")
        require(spec["validation_population"]["count"]
                == definitions[name]["inherited_data_identity"]["populations"]["validation"]["target_count"],
                "Original Validation target count changed")
        require(all(j["validation_reference"]["count"] == spec["validation_population"]["count"]
                    for j in bound if j["dataset"] == name), "Cross-arm Validation population mismatch")
    require(set(runtimes) == {"5080"}, "Only native5080 Runtime is approved")
    validate_runtime_spec(runtimes["5080"], "5080")
    integrity = bindings["evaluator_integrity"]
    require(integrity.get("files") and integrity["closure"] == canonical(integrity["files"]),
            "Operational Taxi/RAF24 evaluator seal missing")
    for name, value in integrity["files"].items():
        checked(root, name, value)
    contract = {"schema": KIND, "cohort": COHORT, "evaluation_scope": "test_taxi_raf24_followup",
                "metrics_schema": "quantity_time_tail_v1", "expected_seeds": list(SEEDS),
                "partial_cohort_explicitly_authorized": True,
                "source_training_cohort_count": 36, "expected_condition_count": 24, "expected_jobs": 24,
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
                      "raw_sha256": sha(destination), "job_count": 24,
                      "original36_all_training_terminal_verified": True, "Test_permit_created": False}))


if __name__ == "__main__":
    main()
