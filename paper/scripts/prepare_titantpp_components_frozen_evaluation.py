"""Bind recovered source131 originals without inference or execution permission.

This is an operational adapter, outside the frozen science manifest. All paths
in a bindings packet are relative to an immutable evaluation workspace. It does
not retrieve artifacts, read parquet data, load torch checkpoints, or issue
validation/Test permits. A partially completed campaign cannot be bound.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


KIND = "titantpp_components_frozen_evaluation_v1"
COHORT = "components36_source131"
SOURCE = "15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53"
REVISION = "57da14a2a7a5687e1bdef7d3164f3e8c0051da59"
PARENT = "e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074"
HANDOFF = "81972d7b774c3b924d4bf6517252a0d0bed8291da9165100ab5b8932128ef1c1"
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
ARMS = ("titantpp_cnn_gru54_correction_disabled", "titantpp_cnn_gru54_cnn_only",
        "titantpp_cnn_gru54_gru_only", "titantpp_cnn_gru54")
SEEDS = (42, 52, 62)
TAILS = dict(zip(DATASETS, (3449, 187, 200)))
TEST_COUNTS = dict(zip(DATASETS, (8327, 88019, 5226)))
GPU_UUIDS = {"5080": "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6",
             "5090": "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def confined(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(),
            "Artifact path must be relative")
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Artifact escapes immutable root")
    return path


def digest(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), f"Missing SHA: {label}")
    return value


def checked(root, name, expected):
    path = confined(root, name)
    require(sha(path) == digest(expected, name), f"Artifact SHA mismatch: {name}")
    return path


def expected_ids():
    return {f"{d}__{s}__{a}" for d in DATASETS for s in SEEDS for a in ARMS}


def host_for(dataset, seed):
    return "5090" if dataset == "intermittent_frozen_5000" and seed != 62 else "5080"


def history_rows(history_path):
    path = Path(history_path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [{**row, "epoch": int(row["epoch"]),
                     "val_qty_rmse": float(row["val_qty_rmse"])}
                    for row in csv.DictReader(handle)]
    require(path.suffix.lower() == ".json", "History must be original CSV or JSON")
    value = read(path)
    rows = value.get("history") if isinstance(value, dict) else value
    require(isinstance(rows, list), "Original history rows missing")
    return rows


def first_minimum(history_path):
    """Replay the original strict selector over completed saved epochs."""
    best = None
    last = 0
    for row in history_rows(history_path):
        epoch = row["epoch"]
        require(type(epoch) is int and epoch == last + 1 and 1 <= epoch <= 300,
                "Incomplete or unordered history")
        value = row["val_qty_rmse"]
        require(type(value) in (int, float), "Original selector must be numeric")
        if math.isfinite(value) and (best is None or value < best[1]):
            best = (epoch, value)
        last = epoch
    require(best is not None and last >= 40, "No completed finite Validation history")
    return best


def bind_job(root, job):
    require(job.get("cohort") == COHORT and job.get("study") == "components", "Foreign cohort")
    require(job.get("id") == f"{job['dataset']}__{job['seed']}__{job['arm']}"
            and job["id"] in expected_ids(), "Foreign condition identity")
    host = host_for(job["dataset"], job["seed"])
    require(job.get("evaluation_host") == job.get("training_host") == host,
            "Native host reassignment")
    require(job.get("frozen_source_closure") == SOURCE
            and job.get("source_revision_history") == [REVISION], "Scientific lineage changed")
    for key in ("checkpoint_state_sha256", "initial_state_sha256"):
        digest(job.get(key), key)
    require(type(job.get("selected_epoch")) is int and job["selected_epoch"] >= 1
            and type(job.get("selected_metric_value")) in (int, float)
            and math.isfinite(job["selected_metric_value"]), "Missing finite selected epoch/value")
    checkpoint = job["checkpoint"]
    checked(root, checkpoint["relative_path"], checkpoint["sha256"])
    require(Path(checkpoint["relative_path"]).name == "best_val_qty_rmse_model.pt",
            "Last checkpoint cannot replace selected checkpoint")
    metadata_path = checked(root, job["checkpoint_metadata_path"], job["checkpoint_metadata_sha256"])
    metadata = read(metadata_path)
    for key, expected in (("backbone", job["arm"]), ("seed", job["seed"]),
                          ("best_epoch", job["selected_epoch"]),
                          ("model_state_sha256", job["checkpoint_state_sha256"]),
                          ("initial_state_sha256", job["initial_state_sha256"]),
                          ("source_revision", REVISION), ("source_revision_history", [REVISION]),
                          ("evaluation_scope", "validation_only"), ("held_out_test_evaluated", False),
                          ("checkpoint_monitor_history_key", "val_qty_rmse"),
                          ("checkpoint_monitor", "validation_raw_quantity_rmse"),
                          ("checkpoint_selection", "best_validation_raw_quantity_rmse")):
        require(metadata.get(key) == expected, f"Selected metadata mismatch: {key}")
    history = checked(root, job["history_path"], job["history_sha256"])
    epoch, value = first_minimum(history)
    require(job["selected_epoch"] == epoch and job["selected_metric_value"] == value
            == metadata.get("selected_metric_value"), "Checkpoint is not original first finite minimum")
    terminal = read(checked(root, job["terminal_manifest_path"], job["terminal_manifest_sha256"]))
    origin = HANDOFF if job["dataset"] == "intermittent_frozen_5000" and job["seed"] == 62 else PARENT
    require(terminal.get("status") == "complete" and terminal.get("scientific_success") is True
            and terminal.get("held_out_test_evaluated") is False
            and terminal.get("contract_sha256") == origin
            and terminal.get("job") == {"id": job["id"], "host": host,
                                         "dataset": job["dataset"], "seed": job["seed"], "arm": job["arm"]},
            "Scientific terminal not verified")
    require(terminal["files"].get(job["checkpoint_original_relative_path"]) == checkpoint["sha256"],
            "Selected binary not bound by original terminal")
    require(Path(job["checkpoint_original_relative_path"]).name == "best_val_qty_rmse_model.pt",
            "Last checkpoint cannot replace selected checkpoint")
    require(terminal["files"].get(job.get("history_original_relative_path")) == job["history_sha256"],
            "Selection history not bound by original terminal")
    last = job["last_checkpoint"]
    checked(root, last["relative_path"], last["sha256"])
    require(Path(job["last_original_relative_path"]).name == "last_epoch_state.pt"
            and terminal["files"].get(job["last_original_relative_path"]) == last["sha256"],
            "Last checkpoint audit binding differs")
    exited = read(checked(root, job["worker_exit_path"], job["worker_exit_sha256"]))
    require(exited.get("actual_child_exit_observed") is True and exited.get("returncode") == 0
            and exited.get("contract_sha256") == origin
            and exited.get("job") == terminal["job"], "Actual scientific worker exit0 missing")
    reference = job["validation_reference"]
    require(reference.get("split") == "validation" and reference.get("evaluation_scope") == "validation_only"
            and reference.get("held_out_test_evaluated") is False
            and type(reference.get("qty_rmse")) in (int, float)
            and math.isfinite(reference["qty_rmse"])
            and math.isclose(reference["qty_rmse"], value, rel_tol=1e-10, abs_tol=1e-8),
            "Foreign Validation endpoint")
    for group in (reference, reference["tail"]):
        require(type(group.get("count")) is int and group["count"] > 0, "Invalid Validation population")
        for key in ("qty_rmse", "qty_mae", "time_nll"):
            require(type(group.get(key)) in (int, float) and math.isfinite(group[key]),
                    "Validation six metrics unavailable")
    endpoint_path = checked(root, job["validation_endpoint_path"], job["validation_endpoint_sha256"])
    require(terminal["files"].get(job.get("validation_endpoint_original_relative_path"))
            == job["validation_endpoint_sha256"], "Validation endpoint not bound by original terminal")
    endpoints = read(endpoint_path)
    require(endpoints.get("status") == "complete" and endpoints.get("job") == terminal["job"]
            and endpoints.get("best_epoch") == epoch
            and endpoints.get("completed_epochs") == len(history_rows(history))
            and endpoints.get("initial_state_sha256") == job["initial_state_sha256"]
            and endpoints.get("evaluation_scope") == "validation_only"
            and endpoints.get("held_out_test_evaluated") is False,
            "Original selected endpoint identity/epoch differs")
    selected = endpoints["selected"]
    require(selected.get("state_sha256") == job["checkpoint_state_sha256"]
            and selected.get("evaluation_scope") == "validation_only"
            and selected.get("held_out_test_evaluated") is False,
            "Original selected endpoint tensor/split differs")
    original_parts = Path(job["checkpoint_original_relative_path"]).parts
    remote_checkpoint = Path(selected["checkpoint_path"])
    require(remote_checkpoint.is_absolute() and remote_checkpoint.parts[-len(original_parts):] == original_parts,
            "Original selected endpoint binary differs")
    for group in (None, "tail"):
        left = selected if group is None else selected[group]
        right = reference if group is None else reference[group]
        require(all(left.get(key) == right.get(key) for key in ("count", "qty_rmse", "qty_mae", "time_nll")),
                "Validation reference differs from original selected endpoint")
    return {**job, "selection_history_path": job["history_path"],
            "selection_history_sha256": job["history_sha256"],
            "binding_status": "bound_terminal_immutable_originals"}


def build_contract(root, design, bindings, runtimes):
    """Seal fully recovered metadata; runtime authorization stays separate."""
    require(design.get("schema") == "titantpp_components_final_Test_design_v1"
            and design.get("fresh_Test_condition_count") == 36, "Foreign design")
    require(bindings.get("schema") == "titantpp_components_checkpoint_bindings_v1",
            "Foreign bindings")
    monitor = read(checked(root, bindings["training_monitor_path"], bindings["training_monitor_sha256"]))
    require(monitor.get("all_training_terminal_verified") is True
            and monitor.get("totals") == {"total": 36, "completed": 36, "running": 0,
                                         "waiting": 0, "unconfirmed": 0,
                                         "all_training_terminal_verified": True}
            and monitor.get("parent_contract_sha256") == PARENT
            and monitor.get("destination_contract_sha256") == HANDOFF,
            "Training terminal proof is incomplete")
    training = read(checked(root, bindings["training_contract_path"],
                            bindings["training_contract_sha256"]))
    require(canonical(training) == PARENT, "Original scientific training contract changed")
    definitions = {d["dataset_id"]: d for d in training["datasets"]}
    source = bindings["sources"]["components"]
    require(source["closure"] == SOURCE == canonical(source["files"])
            and source["revision"] == REVISION and len(source["files"]) == 131,
            "Frozen source131 manifest changed")
    source_root = confined(root, source["root"])
    for name, value in source["files"].items():
        checked(source_root, name, value)
    jobs = bindings["jobs"]
    require(len(jobs) == 36 and {j["id"] for j in jobs} == expected_ids(), "Need exactly36 unique bindings")
    bound = [bind_job(root, job) for job in jobs]
    require(set(bindings["datasets"]) == set(DATASETS), "Foreign dataset set")
    for name, spec in bindings["datasets"].items():
        fixed = design["datasets"][name]
        require(spec["data_sha256"] == fixed["full_data_byte_sha256"]
                and spec["tail_threshold"] == TAILS[name]
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
                and spec["expected_test_count"] == TEST_COUNTS[name],
                "Dataset bytes/count differ from frozen training lineage")
        require(definition["dataset_id"] == name
                and definition["model"]["time_observation_contract"] == fixed["time_observation_contract"]
                and definition["model"]["time_scale"] == fixed["time_scale"]
                and definition["quantity_boundaries_all_train_rows"][-1] == TAILS[name]
                and definition["loader"]["batch_size"] == 128
                and definition["loader"]["max_seq_len"] == fixed["max_seq_len_including_target"]
                and definition["loader"]["lookback_weeks"] == fixed["lookback_seq_units"], "Frozen preprocessing changed")
        for split in ("validation", "test"):
            population = spec[split + "_population"]
            require(type(population.get("count")) is int and population["count"] > 0, "Population not bound")
            for key in ("target_identity_sha256", "truth_sha256", "time_sha256"):
                digest(population.get(key), split + ":" + key)
        require(spec["test_population"]["count"] == spec["expected_test_count"], "Test count changed")
        require(spec["validation_population"]["count"]
                == definitions[name]["inherited_data_identity"]["populations"]["validation"]["target_count"],
                "Original Validation target count changed")
        require(all(j["validation_reference"]["count"] == spec["validation_population"]["count"]
                    for j in bound if j["dataset"] == name), "Cross-arm Validation population mismatch")
    require(set(runtimes) == {"5080", "5090"}, "Both native runtimes must be bound")
    for host, runtime in runtimes.items():
        require(runtime.get("host") == host and runtime.get("gpu_uuid") == GPU_UUIDS[host]
                and runtime.get("cudnn") == (91900 if host == "5080" else 92000), "Native Runtime mismatch")
        require(all(key in runtime for key in ("python", "torch", "numpy", "polars", "cuda",
                                              "python_version", "torch_num_threads", "interop_threads",
                                              "dtype", "deterministic_algorithms", "TF32")),
                "Incomplete native Runtime fingerprint")
        require(isinstance(runtime["python"], str) and Path(runtime["python"]).is_absolute()
                and all(isinstance(runtime[key], str) and runtime[key]
                        for key in ("python_version", "torch", "numpy", "polars", "cuda"))
                and len(runtime["python_version"].split(".")) == 3
                and all(part.isdigit() for part in runtime["python_version"].split("."))
                and type(runtime["torch_num_threads"]) is int and runtime["torch_num_threads"] == 1
                and type(runtime["interop_threads"]) is int and runtime["interop_threads"] == 1,
                "Invalid native Runtime fingerprint")
        require(runtime["dtype"] == "float32" and runtime["deterministic_algorithms"] is True
                and runtime["TF32"] is False, "Native evaluation precision contract changed")
    integrity = bindings["evaluator_integrity"]
    require(integrity["closure"] == canonical(integrity["files"])
            and integrity["files"], "Operational evaluator seal missing")
    for name, value in integrity["files"].items():
        checked(root, name, value)
    contract = {"schema": KIND, "cohort": COHORT, "evaluation_scope": "test_final_comparison",
                "metrics_schema": "quantity_time_tail_v1", "expected_seeds": list(SEEDS),
                "sources": bindings["sources"], "datasets": bindings["datasets"], "jobs": bound,
                "runtimes": runtimes, "evaluator_integrity": integrity,
                "validation_replay_tolerance": {"absolute": 1e-5, "relative": 1e-5},
                "training_monitor_sha256": bindings["training_monitor_sha256"],
                "training_contract_path": bindings["training_contract_path"],
                "training_contract_sha256": bindings["training_contract_sha256"],
                "test_is_independent": False, "heldout_exposure_history_retained": True,
                "automatic_retry": False, "new_training": False, "checkpoint_reselection": False,
                "authorization": {"permit_required": True, "Test_authorized_by_binding": False}}
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
                      "raw_sha256": sha(destination), "job_count": 36, "Test_permit_created": False}))


if __name__ == "__main__":
    main()
