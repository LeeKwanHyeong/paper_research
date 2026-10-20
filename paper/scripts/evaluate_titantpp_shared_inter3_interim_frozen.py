"""Immutable ongoing Inter3 best-state snapshots: native5080 Validation gate, then Test3.

The raw epoch-resume binaries remain unchanged. Only their best_state_dict is
restored in memory. Recorded history supports three overall Validation metrics;
new tail metrics and individual predictions have no historical replay reference.
"""
_INTERIM_ADAPTER_DOC = __doc__
from pathlib import Path as _Path
import hashlib as _hashlib

_INTERIM_REVIEWED_PATH = _Path(__file__).with_name("evaluate_titantpp_shared_raf_taxi6_frozen.py")
INTERIM_REVIEWED_WORKER_SHA256 = "3d796732555f2d97b0a8e74f1b30cbe22310a49e0b380ad061d3f92e983f70a9"
_interim_reviewed_bytes = _INTERIM_REVIEWED_PATH.read_bytes()
if _hashlib.sha256(_interim_reviewed_bytes).hexdigest() != INTERIM_REVIEWED_WORKER_SHA256:
    raise ValueError("Reviewed selected-six evaluator changed; interim adapter refused")
_interim_reviewed_adapter = _interim_reviewed_bytes.decode("utf-8")
_main = '\nif __name__ == "__main__":\n    main()\n'
if _interim_reviewed_adapter.count(_main) != 1:
    raise ValueError("Reviewed selected-six entry point changed")
exec(compile(_interim_reviewed_adapter.replace(_main, "\n"), str(_INTERIM_REVIEWED_PATH) + " [interim adapter]", "exec"), globals())
__doc__ = _INTERIM_ADAPTER_DOC
import math
from copy import deepcopy

SCHEMA = "titantpp_shared_inter3_interim_frozen_v1"
GATE_SCHEMA = "titantpp_shared_inter3_interim_validation_gate_v1"
SNAPSHOT_SCHEMA = "titantpp_shared_inter3_interim_snapshot_job_v1"
DATASET = "intermittent_frozen_5000"
DATASETS = (DATASET,)
SEEDS = (42, 52, 62)
TAILS = {DATASET: 187.0}
ORIGINAL_TRAINING_CONTRACT_SHA256 = "4ed35f0054995f51dadb9b1d6227fc9c09194f9414a9f832bb723dc1d1c058f1"
HISTORY_METRICS = {"qty_rmse": "val_qty_rmse", "qty_mae": "val_qty_mae", "time_nll": "val_time_nll"}
_METADATA_KEYS = (
    "checkpoint_type", "checkpoint_schema_version", "epoch", "backbone", "variant", "seed",
    "checkpoint_monitor", "checkpoint_monitor_history_key", "checkpoint_selection", "selection",
    "selection_formula", "resume_identity", "initial_state_sha256", "model_state_sha256",
    "best_selection_value", "best_epoch", "best_val_joint_objective", "best_state_sha256",
    "encoder_config", "interface_meta", "optimizer_group_contract", "source_revision",
    "source_revision_history", "evaluation_scope", "held_out_test_evaluated",
    "time_head_training_verification", *frozen.input_identity())


def selected_job(contract, job_id):
    require(contract.get("schema") == SCHEMA and contract.get("host") == "5080"
            and contract.get("evaluation_authorized") is True,
            "Only explicitly approved5080 interim Inter3 evaluation is admitted")
    jobs = contract.get("jobs", [])
    require(len(jobs) == 3 and len({job.get("id") for job in jobs}) == 3
            and {job.get("seed") for job in jobs} == set(SEEDS), "Exactly Inter42/52/62 snapshots required")
    for job in jobs:
        require(job.get("dataset") == DATASET and job.get("arm") == ARM
                and type(job.get("seed")) is int and type(job.get("selected_epoch")) is int
                and type(job.get("snapshot_epoch")) is int
                and 1 <= job["selected_epoch"] <= job["snapshot_epoch"]
                and job.get("id") == f"{DATASET}__{ARM}__seed{job['seed']}__losslog_mse",
                "Interim frozen model, seed or snapshot epoch changed")
    matches = [job for job in jobs if job["id"] == job_id]
    require(len(matches) == 1, "Unknown interim Inter3 job")
    return matches[0]


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def checkpoint_snapshot_metadata(payload):
    """Derived JSON identity only; nonfinite diagnostic metadata becomes null.

    This never creates a completed selected checkpoint or changes the raw file.
    Optimizer/RNG tensors and both weight dictionaries stay only in that file.
    """
    require(isinstance(payload, dict) and all(key in payload for key in _METADATA_KEYS)
            and isinstance(payload.get("history"), list), "Incomplete raw resume metadata")
    result = {"schema": "titantpp_shared_inter3_interim_checkpoint_metadata_v1",
              "metadata": {key: deepcopy(payload[key]) for key in _METADATA_KEYS},
              "history": deepcopy(payload["history"]),
              "nonfinite_diagnostic_metadata_representation": "null"}
    result = _json_safe(result)
    json.dumps(result, allow_nan=False)
    return result


def _history_row(payload):
    history = payload["history"]
    require(type(payload.get("epoch")) is int and payload["epoch"] > 0
            and [row.get("epoch") for row in history] == list(range(1, payload["epoch"] + 1)),
            "Embedded snapshot history must be complete and contiguous")
    epoch = frozen.prepared.select_best_epoch(history)
    require(epoch == payload.get("best_epoch"), "Snapshot best state is not the first strict finite RMSE minimum")
    selected = history[epoch - 1]
    require(all(type(selected.get(key)) in (int, float) and math.isfinite(selected[key])
                for key in HISTORY_METRICS.values()), "Selected epoch lacks finite recorded Validation metrics")
    require(math.isclose(payload["best_selection_value"], selected["val_qty_rmse"], rel_tol=1e-10, abs_tol=1e-8),
            "Raw snapshot selected score differs from embedded history")
    return selected


def verify_snapshot_payload(payload, job, definition, source_files_sha256):
    """Validate current proof honestly and restore only the independently bound best tensors."""
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    frozen.validate_checkpoint(payload, ARM)
    require(payload.get("checkpoint_type") == "epoch_resume" and payload.get("checkpoint_schema_version") == 2
            and payload.get("seed") == job["seed"] and payload.get("best_epoch") == job["selected_epoch"]
            and payload.get("epoch") == job["snapshot_epoch"] and payload.get("checkpoint_monitor") == frozen.MONITOR,
            "Raw interim snapshot type, model, seed, epoch or monitor changed")
    require(all(payload.get(key) == value for key, value in frozen.input_identity().items()),
            "Interim checkpoint input or joint objective identity changed")
    arguments = payload["resume_identity"]["arguments"]
    require(arguments.get("dataset_contract") == DATASET
            and arguments.get("data_sha256") == definition["inherited_data_identity"]["data"]["sha256"]
            and arguments.get("split_manifest_sha256") == definition["inherited_data_identity"]["split_manifest"]["sha256"]
            and payload["interface_meta"].get("source_files_sha256") == source_files_sha256,
            "Interim checkpoint original data, split or source binding changed")
    current, best = payload["model_state_dict"], payload["best_state_dict"]
    require(canonical_state_dict_sha256(current) == payload["model_state_sha256"]
            and canonical_state_dict_sha256(best) == payload["best_state_sha256"] == job["state_sha256"],
            "Raw current/best tensor digest changed")
    proof = payload["time_head_training_verification"]
    frozen.verify_time_head_proof(proof, current)
    best_time = frozen._time_head_sha(best, proof["time_head_parameter_names"])
    require(best_time != proof["initial_time_head_sha256"], "Frozen best time head equals untrained initialization")
    row = _history_row(payload)
    return row, {"original_current_state_proof_verified": True,
                 "current_state_sha256": payload["model_state_sha256"],
                 "selected_best_state_sha256": payload["best_state_sha256"],
                 "selected_best_time_head_sha256": best_time,
                 "selected_best_time_head_differs_from_initial": True,
                 "selected_epoch_individual_gradient_proof_available": False}


def validate_checkpoint_metadata(payload, metadata, job):
    require(metadata == checkpoint_snapshot_metadata(payload), "Derived snapshot metadata differs from original raw resume")
    require(payload.get("best_epoch") == job["selected_epoch"] and payload.get("epoch") == job["snapshot_epoch"],
            "Derived snapshot epoch binding changed")


def verify_contract(root, contract=None):
    root = Path(root).resolve()
    contract = read(root / "evaluation_contract.json") if contract is None else contract
    require(contract.get("jobs"), "Missing frozen interim Inter3 jobs")
    selected_job(contract, contract["jobs"][0]["id"])
    training_path = bound_path(root, contract["training_contract"])
    require(sha(training_path) == ORIGINAL_TRAINING_CONTRACT_SHA256, "Original raw9 execution contract changed")
    training = read(training_path)
    source, source_root = training["source"], frozen.resolve_under(root, contract["source_root"])
    require(len(source["files"]) == 983 and canonical(source["files"]) == source["files_sha256"],
            "Original5090 source983 closure changed")
    for relative, digest in source["files"].items():
        require(sha(frozen.resolve_under(source_root, relative)) == digest, "Original scientific source changed: " + relative)
    require(frozen.ROOT.resolve() == source_root, "Foreign or shadowed scientific source imported")
    frozen.verify_frozen_dependencies()
    require(set(contract["data"]) == {DATASET}, "Only original Intermittent data are authorized")
    for job in contract["jobs"]:
        original, definition, _, _ = frozen.select_job(training, "5090", job["id"])
        spec = contract["data"][DATASET]
        require(original["dataset"] == DATASET and definition["quantity_boundaries_all_train_rows"][-1] == 187.0
                and spec["training"]["sha256"] == definition["inherited_data_identity"]["data"]["sha256"]
                and spec["populations"]["validation"] == definition["inherited_data_identity"]["populations"]["validation"],
                "Original Intermittent data, head or Validation target identity changed")
        require(set(spec["populations"]) == {"validation", "test"}, "Unexpected evaluation split")
        for population in spec["populations"].values():
            require(type(population.get("target_count")) is int and population["target_count"] > 0
                    and all(isinstance(population.get(key), str) and len(population[key]) == 64
                            for key in ("target_identity_sha256", "target_quantity_sha256")), "Missing exact population identity")
        bound_path(root, spec["training"])
        bound_path(root, spec["full"])
    native = contract.get("native", {})
    require(native.get("gpu_uuid") == GPU_UUID and native.get("runtime_expected", {}).get("cudnn") == 91900
            and training["hosts"]["5090"]["runtime_expected"]["cudnn"] == 92000,
            "Approved evaluation5080/training5090 runtime binding changed")
    return contract, training


def verify_job_artifacts(root, contract, training, job):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    _, definition, _, _ = frozen.select_job(training, "5090", job["id"])
    checkpoint = bound_path(root, job["checkpoint"])
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    row, proof = verify_snapshot_payload(payload, job, definition, training["source"]["files_sha256"])
    metadata = read(bound_path(root, job["checkpoint_metadata"]))
    validate_checkpoint_metadata(payload, metadata, job)
    history = read(bound_path(root, job["selection_history"]))["history"]
    require(len(history) >= len(payload["history"])
            and _json_safe(history[:len(payload["history"])]) == _json_safe(payload["history"]),
            "Captured external history does not contain the exact embedded snapshot prefix")
    snapshot = read(bound_path(root, job["snapshot_receipt"]))
    expected = {"schema": SNAPSHOT_SCHEMA, "status": "frozen_interim_selected", "job_id": job["id"],
        "dataset": DATASET, "arm": ARM, "seed": job["seed"], "selected_epoch": job["selected_epoch"],
        "snapshot_epoch": job["snapshot_epoch"], "checkpoint_sha256": job["checkpoint"]["sha256"],
        "state_sha256": job["state_sha256"], "source_files_sha256": training["source"]["files_sha256"],
        "target_identity": contract["data"][DATASET]["populations"]["validation"],
        "original_training_terminal_required": False, "held_out_test_evaluated": False,
        "historical_tail_metrics_available": False, "historical_individual_predictions_available": False,
        "predictions_available": False}
    require(all(snapshot.get(key) == value for key, value in expected.items()), "Frozen interim snapshot receipt binding changed")
    historical = snapshot["history_reference"]
    require(historical.get("split") == "validation" and historical.get("selected_epoch") == job["selected_epoch"]
            and historical.get("source") == "embedded_checkpoint_history"
            and historical.get("metrics") == {"overall": {metric: row[key] for metric, key in HISTORY_METRICS.items()}},
            "Snapshot history reference differs from selected embedded epoch")
    require("validation_predictions" not in job, "Interim snapshot has no original individual prediction reference")
    relocated = {**snapshot, "selected_checkpoint": str(checkpoint), "time_head_snapshot_verification": proof}
    return definition, relocated, historical


def _load_interim_selected(definition, job, snapshot, root, device):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    path = frozen.resolve_under(root, snapshot["selected_checkpoint"])
    require(sha(path) == snapshot["checkpoint_sha256"], "Frozen raw resume binary changed")
    payload = torch_load_checkpoint(path, map_location="cpu")
    verify_snapshot_payload(payload, job, definition, snapshot["source_files_sha256"])
    model, encoder = frozen.build_model(definition, ARM)
    require(encoder == payload["encoder_config"], "Reconstructed frozen encoder metadata changed")
    model.load_state_dict(payload["best_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), path, payload["best_state_sha256"]


# Preserve the reviewed inference, target ordering, causal checks and metrics.
# Only loading changes from a completed selected binary to raw resume best tensors.
_frame_source = _reviewed_source.split("def evaluate_frame(", 1)[1].split("\ndef verify_validation_receipt(", 1)[0]
_frame_source = "def evaluate_frame(" + _frame_source
_loader_call = "frozen._load_selected(definition, job, train_receipt, root, device)"
require(_frame_source.count(_loader_call) == 1, "Reviewed selected loader call changed")
exec(compile(_frame_source.replace(_loader_call, "_load_interim_selected(definition, job, train_receipt, root, device)"),
             str(_REVIEWED_PATH) + " [interim best-state loader]", "exec"), globals())


def compare_recorded_history(metrics, reference):
    """Only the three historical overall metrics exist; no tail replay is claimed."""
    require(metrics.get("tail_threshold") == 187.0
            and metrics.get("tail_rule") == "q > frozen_Train_large_quantity_threshold"
            and metrics.get("time_metric") == "recorded_positive_integer_time_nll", "Intermittent metric definitions changed")
    require(type(metrics["overall"].get("count")) is int and metrics["overall"]["count"] > 0
            and type(metrics["tail"].get("count")) is int
            and 0 <= metrics["tail"]["count"] <= metrics["overall"]["count"], "Invalid Validation metric support")
    for group in ("overall", "tail"):
        for metric in HISTORY_METRICS:
            value = metrics[group][metric]
            require((metrics[group]["count"] == 0 and value is None)
                    or (metrics[group]["count"] > 0 and type(value) in (int, float) and math.isfinite(value)),
                    "New full Validation metrics must be finite on their evaluated population")
    for metric in HISTORY_METRICS:
        require(math.isclose(metrics["overall"][metric], reference["metrics"]["overall"][metric],
                             rel_tol=frozen.REPLAY_RTOL, abs_tol=frozen.REPLAY_ATOL),
                "Selected embedded history Validation replay changed: " + metric)


def verify_validation_receipt(root, contract, job, receipt):
    snapshot = read(bound_path(root, job["snapshot_receipt"]))
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "complete"
            and receipt.get("split") == "validation" and receipt.get("job_id") == job["id"]
            and receipt.get("contract_sha256") == canonical(contract)
            and receipt.get("selected_epoch") == job["selected_epoch"]
            and receipt.get("snapshot_epoch") == job["snapshot_epoch"]
            and receipt.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
            and receipt.get("state_sha256") == job["state_sha256"]
            and receipt.get("parameters_unchanged") is True and receipt.get("context_unchanged") is True
            and receipt.get("validation_replay_passed") is True
            and receipt.get("recorded_history_metric_replay_count") == 3
            and receipt.get("historical_tail_metrics_available") is False
            and receipt.get("historical_individual_predictions_available") is False
            and receipt.get("individual_predictions_match") is None
            and receipt.get("native_runtime_checked") is True
            and receipt.get("held_out_test_evaluated") is False
            and set(receipt.get("qualification", {})) == QUALIFICATION_KEYS
            and all(value is True for value in receipt["qualification"].values()),
            "Fresh5080 interim Validation replay or explicit historical-reference limits missing")
    compare_recorded_history(receipt["metrics"], snapshot["history_reference"])
    require(receipt["source_files_sha256"] == snapshot["source_files_sha256"]
            and receipt["target_identity"] == snapshot["target_identity"]
            == contract["data"][DATASET]["populations"]["validation"]
            and receipt["metrics"]["overall"]["count"] == receipt["prediction_rows"]
            == receipt["target_identity"]["target_count"], "Validation source/target support changed")
    bound_path(root, receipt["predictions"])
    return receipt


def verify_validation_gate(root, contract):
    root = Path(root).resolve()
    gate = read(root / "validation_gate.json")
    require(gate.get("schema") == GATE_SCHEMA and gate.get("status") == "PASS"
            and gate.get("contract_sha256") == canonical(contract) and gate.get("job_count") == 3,
            "All three fresh5080 interim Validation replays must pass before Test")
    ids = {job["id"] for job in contract["jobs"]}
    require(set(gate.get("validation_receipts", {})) == ids
            and set(gate.get("validation_worker_exits", {})) == ids, "Incomplete interim Inter3 gate")
    for job in contract["jobs"]:
        folder = root / "results" / job["id"]
        path, exit_path = folder / "validation_receipt.json", folder / "validation_exit.json"
        require(sha(path) == gate["validation_receipts"][job["id"]]
                and sha(exit_path) == gate["validation_worker_exits"][job["id"]], "Validation receipt/exit SHA changed")
        verify_validation_receipt(root, contract, job, read(path))
        actual = read(exit_path)
        require(actual.get("job_id") == job["id"] and actual.get("split") == "validation"
                and actual.get("contract_sha256") == canonical(contract)
                and type(actual.get("actual_os_returncode")) is int and actual["actual_os_returncode"] == 0
                and actual.get("timeout") is False and actual.get("actual_child_exit_observed") is True,
                "Fresh5080 interim Validation actual child exit0 missing")
    return gate


def evaluate_stage(root, job_id, split, *, device="cuda:0"):
    root = Path(root).resolve()
    require(split in ("validation", "test"), "Unsupported evaluation stage")
    contract, training = verify_contract(root)
    job = selected_job(contract, job_id)
    if split == "test":
        verify_validation_gate(root, contract)  # Before raw checkpoint load or Test materialization.
    verify_synthetic_qualifications(root, contract)
    folder = root / "results" / job_id
    require(not (folder / (split+"_receipt.json")).exists()
            and not (folder / (split+"_failure.json")).exists(), "Existing interim evaluation cannot be repeated")
    write(folder / (split+".intent.json"), {"contract_sha256": canonical(contract),
        "job_id": job_id, "split": split, "started_unix": time.time()})
    try:
        definition, snapshot, reference = verify_job_artifacts(root, contract, training, job)
        runtime = runtime_check(contract, device)
        import polars as pl
        spec = contract["data"][DATASET]
        data_ref = spec["training" if split == "validation" else "full"]
        data_path = bound_path(root, data_ref)
        lazy = pl.scan_parquet(data_path)
        if split == "validation":
            lazy = lazy.filter(pl.col("chronological_split").is_in(["train", "validation"]))
        frame = lazy.collect().sort(["oper_part_no", "seq"])
        if split == "test":
            training_frame = pl.read_parquet(bound_path(root, spec["training"])).sort(["oper_part_no", "seq"])
            require(set(training_frame.columns) <= set(frame.columns)
                    and frame.filter(pl.col("chronological_split").is_in(["train", "validation"]))
                    .select(training_frame.columns).equals(training_frame), "Full raw parent differs from frozen training export")
        output = folder / (split+"_predictions.jsonl.gz")
        result = evaluate_frame(definition, job, snapshot, root, frame, split, output,
                                spec["populations"][split], device=device)
        if split == "validation":
            compare_recorded_history(result["metrics"], reference)
        after_runtime = runtime_check(contract, device)
        require(runtime == after_runtime and sha(data_path) == data_ref["sha256"], "Evaluation Runtime/input changed")
        verify_contract(root, contract)
        verify_job_artifacts(root, contract, training, job)
        if split == "test":
            verify_validation_gate(root, contract)
        receipt = {"schema": SCHEMA, "status": "complete", "contract_sha256": canonical(contract),
            "job_id": job_id, "dataset": DATASET, "arm": ARM, "seed": job["seed"],
            "selected_epoch": job["selected_epoch"], "snapshot_epoch": job["snapshot_epoch"], "split": split, **result,
            "checkpoint_sha256": job["checkpoint"]["sha256"], "snapshot_receipt_sha256": job["snapshot_receipt"]["sha256"],
            "source_files_sha256": training["source"]["files_sha256"],
            "time_head_snapshot_verification": snapshot["time_head_snapshot_verification"],
            "retrained": False, "reselected": False, "held_out_test_evaluated": split == "test",
            "validation_replay_passed": split == "validation", "recorded_history_metric_replay_count": 3 if split == "validation" else 0,
            "historical_tail_metrics_available": False, "historical_individual_predictions_available": False,
            "individual_predictions_match": None, "native_runtime_checked": True, "runtime": runtime, "runtime_after": after_runtime,
            "training_host": "5090", "evaluation_host": "5080", "whole_raw9_complete_asserted": False,
            "interim_snapshot": True, "original_training_terminal_required": False,
            "host_difference_limitation": "Original5090/cuDNN92000 snapshot replayed on5080/cuDNN91900; fixed1e-5 tolerance is unchanged",
            "historical_reference_limitation": "Only selected-epoch overall RMSE,MAE,timeNLL exist; tail metrics and individual predictions are new outputs",
            "predictions": {"path": str(output.relative_to(root)), "sha256": sha(output)}, "completed_unix": time.time()}
        write(folder / (split+"_receipt.json"), receipt)
        return receipt
    except BaseException as error:
        write(folder / (split+"_failure.json"), {"schema": SCHEMA, "status": "failed",
            "job_id": job_id, "split": split, "error": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "contract_sha256": canonical(contract)})
        raise


if __name__ == "__main__":
    main()
