"""Prepare a local immutable TaxiRAF24 packet from recovered/native audit evidence.

Default mode only checks readiness. --build requires root-finalized operation
copies already present; this helper never copies operational code, issues a
permit, deserializes parquet/checkpoints, infers predictions, or uses SSH.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from paper.scripts import prepare_titantpp_components_taxi_raf24_frozen_evaluation as prepare
from paper.scripts import evaluate_titantpp_components_taxi_raf24_frozen as evaluator


DEFAULT_BUNDLE = PROJECT / "search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1"
OUTPUTS = ("bindings.json", "runtimes.json", "contract.json", "binding_preparation_receipt.json")
INPUTS = ("partial_approval.json", "originals_manifest.json", "originals_archive_verified.tar.gz",
          "provenance/training_contract.json", "provenance/original36_evaluation_design.json",
          "provenance/training_monitor.json", "provenance/training_terminal_snapshot.json",
          "native_input_receipt.json", "control/native_audit_exit.json", "control/native_audit_stdout.bin",
          "operation/evaluate.py", "operation/pipeline.py", "source/sample_data/.frozen_root_sentinel")


def write_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def readiness(root):
    root = Path(root).resolve()
    missing = [name for name in INPUTS if not (root / name).is_file()]
    metadata_missing = []
    native = root / "native_input_receipt.json"
    if native.is_file():
        value = prepare.read(native)
        for identity in sorted(prepare.expected_ids()):
            reference = value.get("metadata_refs", {}).get(identity)
            if not reference or not prepare.base.confined(root, reference["metadata_path"]).is_file():
                metadata_missing.append(identity)
    else:
        metadata_missing = sorted(prepare.expected_ids())
    existing = [name for name in OUTPUTS if (root / name).exists()]
    return {"status": "inputs_present_not_yet_bound" if not missing and not metadata_missing else "waiting_for_inputs",
            "ready_to_attempt_binding": not missing and not metadata_missing and not existing,
            "missing_paths": missing, "missing_metadata_ids": metadata_missing,
            "existing_immutable_outputs": existing,
            "metadata_readiness_is_not_execution_permission": True,
            "operations_must_be_finalized_and_copied_by_root": True,
            "model_predictions": 0, "remote_calls": 0, "Test_permits_created": 0}


def create_packet(root):
    """Validate available evidence in memory before any exclusive output write."""
    root = Path(root).resolve()
    state = readiness(root)
    prepare.require(state["ready_to_attempt_binding"], "Missing input or existing immutable output: " + json.dumps(state))
    prepare.require(str(root) == prepare.AUTHORIZED_LOCAL_ROOT, "Builder is restricted to the approved local binding root")
    approval_path, training_path = "partial_approval.json", "provenance/training_contract.json"
    approval, training = prepare.read(root / approval_path), prepare.read(root / training_path)
    prepare.require(prepare.canonical(training) == prepare.PARENT, "Original source131 training contract canonical changed")
    definitions = {item["dataset_id"]: item for item in training["datasets"]}
    
    design = prepare.read(root / "provenance/original36_evaluation_design.json")
    prepare.require(prepare.canonical(design) == prepare.ORIGINAL_DESIGN_CLOSURE,
                    "Original36 design changed")
    originals = prepare.read(root / "originals_manifest.json")
    files = originals["files"]
    prepare.require(originals.get("all_file_SHA_verified") is True
                    and originals.get("source131_SHA_verified") is True
                    and originals.get("checkpoint_selected_last_pairs") == 24
                    and originals.get("remote_originals_changed") is False
                    and isinstance(files, dict) and len(files) == 360,
                    "Original24 retrieval manifest is incomplete")
    prepare.require(prepare.sha(root / "originals_archive_verified.tar.gz") == originals["archive_raw_sha256"],
                    "Original archive raw SHA differs")
    for name, digest in files.items():
        prepare.checked(root, name, digest)
    audit = prepare.read(root / "native_input_receipt.json")
    audit_exit = prepare.read(root / "control/native_audit_exit.json")
    prepare.require(audit_exit.get("actual_transport_exit_observed") is True
                    and type(audit_exit.get("actual_transport_exit")) is int and audit_exit["actual_transport_exit"] == 0
                    and audit_exit.get("model_predictions") == 0
                    and audit_exit.get("stdout_sha256") == prepare.sha(root / "control/native_audit_stdout.bin")
                    == prepare.sha(root / "native_input_receipt.json"),
                    "Successful native input audit receipt is not bound to observed transport exit0")
    prepare.require(audit.get("status") == "complete" and audit.get("model_predictions") == 0
                    and audit.get("new_training") is False and audit.get("source131_closure") == prepare.SOURCE
                    and audit.get("data_sha256") == {dataset: design["datasets"][dataset]["full_data_byte_sha256"] for dataset in prepare.DATASETS}
                    and audit.get("approval_sha256") == prepare.sha(root / approval_path)
                    and audit.get("runtime_observed") == audit.get("runtime_after")
                    and set(audit.get("metadata_refs", {})) == prepare.expected_ids()
                    and set(audit.get("populations", {})) == set(prepare.DATASETS) and all(set(value) == {"validation", "test"} for value in audit["populations"].values()),
                    "Native no-prediction input audit differs")
    source = {"root": "source", "files": training["source"]["files"],
              "closure": training["source"]["files_sha256"], "revision": prepare.REVISION}
    prepare.require(source["closure"] == prepare.SOURCE == prepare.canonical(source["files"])
                    and len(source["files"]) == 131, "Original scientific source131 seal changed")
    sentinel = "source/sample_data/.frozen_root_sentinel"
    prepare.checked(root, sentinel, training["input_files"]["sample_data/.frozen_root_sentinel"])
    bindings = {"schema": "titantpp_components_taxi_raf24_checkpoint_bindings_v1",
                "sources": {"components": source}, "jobs": [],
                "datasets": {}, "training_contract_path": training_path,
                "training_contract_sha256": prepare.sha(root / training_path),
                "training_monitor_path": "provenance/training_monitor.json",
                "training_monitor_sha256": prepare.sha(root / "provenance/training_monitor.json"),
                "parentterminal_evidence_path": "provenance/training_terminal_snapshot.json",
                "parentterminal_evidence_sha256": prepare.sha(root / "provenance/training_terminal_snapshot.json"),
                "partial_approval_path": approval_path, "partial_approval_sha256": prepare.sha(root / approval_path)}
    selected_jobs = [job for job in training["jobs"] if job["id"] in prepare.expected_ids()]
    prepare.require(len(selected_jobs) == 24 and {job["id"] for job in selected_jobs} == prepare.expected_ids(),
                    "Original training job registry differs from the authorized exact24")
    expected_files = {}
    for job in sorted(selected_jobs, key=lambda row: row["id"]):
        jid = job["id"]
        prefix = "originals/run/" + jid + "/"
        terminal_path, exit_path = prefix + "terminal_manifest.json", "originals/worker_exits/" + jid + ".json"
        terminal = prepare.read(prepare.checked(root, terminal_path, files[terminal_path]))
        expected_files[terminal_path], expected_files[exit_path] = files[terminal_path], files[exit_path]
        for name, digest in terminal["files"].items():
            expected_files[prefix + name] = digest
        original_run = "runs/" + job["arm"] + "/count_only_log_regression/seed_" + str(job["seed"])
        cp, last = original_run + "/best_val_qty_rmse_model.pt", original_run + "/last_epoch_state.pt"
        history, endpoint = original_run + "/history.json", original_run + "/endpoint_replays.json"
        reference = audit["metadata_refs"][jid]
        metadata = prepare.read(prepare.checked(root, reference["metadata_path"], reference["metadata_sha256"]))
        endpoints = prepare.read(prepare.checked(root, prefix + endpoint, terminal["files"][endpoint]))
        prepare.require(endpoints.get("evaluation_scope") == "validation_only"
                        and endpoints.get("held_out_test_evaluated") is False,
                        "Endpoint is not a Validation-only original")
        item = {"cohort": prepare.COHORT, "study": "components", "id": jid,
                "dataset": job["dataset"], "seed": job["seed"], "arm": job["arm"],
                "training_host": job["host"], "evaluation_host": job["host"],
                "frozen_source_closure": prepare.SOURCE,
                "source_revision_history": metadata["source_revision_history"],
                "checkpoint_state_sha256": reference["checkpoint_state_sha256"],
                "initial_state_sha256": reference["initial_state_sha256"],
                "selected_epoch": reference["selected_epoch"], "selected_metric_value": reference["selected_metric_value"],
                "checkpoint": {"relative_path": prefix + cp, "sha256": terminal["files"][cp]},
                "checkpoint_original_relative_path": cp,
                "last_checkpoint": {"relative_path": prefix + last, "sha256": terminal["files"][last]},
                "last_original_relative_path": last,
                "checkpoint_metadata_path": reference["metadata_path"], "checkpoint_metadata_sha256": reference["metadata_sha256"],
                "history_path": prefix + history, "history_sha256": terminal["files"][history],
                "history_original_relative_path": history,
                "validation_endpoint_path": prefix + endpoint, "validation_endpoint_sha256": terminal["files"][endpoint],
                "validation_endpoint_original_relative_path": endpoint,
                "terminal_manifest_path": terminal_path, "terminal_manifest_sha256": files[terminal_path],
                "worker_exit_path": exit_path, "worker_exit_sha256": files[exit_path],
                "validation_reference": {**deepcopy(endpoints["selected"]), "split": "validation"}}
        bindings["jobs"].append(item)
    prepare.require(expected_files == files, "Original manifest has missing or unrelated conditions")
    for dataset in prepare.DATASETS:
        definition = definitions[dataset]
        data_path = "data/" + dataset + ".parquet"
        split_path = "data/" + dataset + "_split_manifest.json"
        training_path = data_path if dataset == "yellow_trip_hourly" else "data/raf_spare_parts_train_validation.parquet"
        data_sha = design["datasets"][dataset]["full_data_byte_sha256"]
        training_sha = definition["inherited_data_identity"]["data"]["sha256"]
        split_sha = definition["inherited_data_identity"]["split_manifest"]["sha256"]
        bindings["datasets"][dataset] = {
            "data_path": data_path, "data_sha256": data_sha,
            "training_data_path": training_path, "training_data_sha256": training_sha,
            "split_manifest_path": split_path, "split_manifest_sha256": split_sha,
            "tail_threshold": prepare.base.TAILS[dataset],
            "expected_test_count": design["datasets"][dataset]["expected_Test_target_count"],
            "definition": deepcopy(definition),
            "validation_population": audit["populations"][dataset]["validation"],
            "test_population": audit["populations"][dataset]["test"]}
    integrity_files = {name: prepare.sha(root / name) for name in
                       ("operation/evaluate.py", "operation/pipeline.py", sentinel, "native_input_receipt.json")}
    bindings["evaluator_integrity"] = {"files": integrity_files, "closure": prepare.canonical(integrity_files)}
    runtimes = {"5080": audit["runtime"]}
    contract = prepare.build_contract(root, design, bindings, runtimes)
    contract["native_input_audit"] = {"path": "native_input_receipt.json", "raw_sha256": prepare.sha(root / "native_input_receipt.json"),
        "actual_exit_path": "control/native_audit_exit.json", "actual_exit_raw_sha256": prepare.sha(root / "control/native_audit_exit.json"),
        "model_predictions": 0, "new_training": False}
    contract["immutable_originals"] = {"manifest_path": "originals_manifest.json",
        "manifest_raw_sha256": prepare.sha(root / "originals_manifest.json"), "file_count": len(files),
        "archive_path": "originals_archive_verified.tar.gz", "archive_raw_sha256": originals["archive_raw_sha256"],
        "archive_local_only": True, "remote_archive_presence_claimed": False}
    contract["contract_sha256"] = prepare.canonical({key: value for key, value in contract.items() if key != "contract_sha256"})
    return bindings, runtimes, contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--build", action="store_true", help="Root must first finalize/copy operation/evaluate.py and pipeline.py")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not args.build:
        print(json.dumps(readiness(root), ensure_ascii=False, allow_nan=False))
        return
    bindings, runtimes, contract = create_packet(root)
    write_exclusive(root / "bindings.json", bindings)
    write_exclusive(root / "runtimes.json", runtimes)
    write_exclusive(root / "contract.json", contract)
    # Static verification only: no model/scientific import, permit, or inference.
    actual = evaluator.verify_contract(root, require_bound=True)
    prepare.require(actual["contract_sha256"] == contract["contract_sha256"], "Prepared evaluator contract changed")
    receipt = {"status": "local_immutable_binding_verified", "cohort": prepare.COHORT, "job_count": 24,
               "contract_sha256": contract["contract_sha256"], "contract_raw_sha256": prepare.sha(root / "contract.json"),
               "bindings_raw_sha256": prepare.sha(root / "bindings.json"), "runtimes_raw_sha256": prepare.sha(root / "runtimes.json"),
               "model_predictions": 0, "remote_calls": 0, "Test_permits_created": 0,
               "original36_all_training_terminal_verified": True,
               "operations_copied_or_modified_by_builder": False,
               "validation24_replay_and_Test24_not_executed_by_builder": True}
    write_exclusive(root / "binding_preparation_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
