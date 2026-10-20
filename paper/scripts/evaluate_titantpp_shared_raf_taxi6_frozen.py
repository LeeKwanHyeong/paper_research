"""Selected completed RAF/Taxi Shared15 fits: fresh5080 replay gate, then Test6.

The reviewed RAF evaluator is preserved byte-for-byte. This separate worker
adapts only the selected six-fit scope, training-reference binding, dataset
threshold and explicitly authorized native evaluation host. Training remains
bound to original5090/source983; no raw9-wide completion is asserted.
"""
_ADAPTER_DOC = __doc__
from pathlib import Path as _Path
import hashlib as _hashlib

_REVIEWED_PATH = _Path(__file__).with_name("evaluate_titantpp_shared_raw_raf3_frozen.py")
REVIEWED_WORKER_SHA256 = "2fdd0f56757c4330e3e140ca800bbfbb5d13060f8b663d7156a1e40a7e3225c5"
_reviewed_bytes = _REVIEWED_PATH.read_bytes()
if _hashlib.sha256(_reviewed_bytes).hexdigest() != REVIEWED_WORKER_SHA256:
    raise ValueError("Reviewed RAF evaluator changed; selected-six adapter refused")
_reviewed_source = _reviewed_bytes.decode("utf-8")
_replacements = (
    ('frozen.MetricAccumulator(200.0)', 'frozen.MetricAccumulator(definition["quantity_boundaries_all_train_rows"][-1])', 2),
    ('"All three fresh native synthetic qualifications must pass"',
     '"All six fresh native synthetic qualifications must pass"', 1),
    ('\nif __name__ == "__main__":\n    main()\n', '\n', 1),
)
for _old, _new, _count in _replacements:
    if _reviewed_source.count(_old) != _count:
        raise ValueError("Unexpected reviewed evaluator adaptation count")
    _reviewed_source = _reviewed_source.replace(_old, _new)
exec(compile(_reviewed_source, str(_REVIEWED_PATH) + " [selected RAF/Taxi6 adapter]", "exec"), globals())
__doc__ = _ADAPTER_DOC

SCHEMA = "titantpp_shared_raf_taxi6_frozen_v1"
GATE_SCHEMA = "titantpp_shared_raf_taxi6_validation_gate_v1"
SELECTED_EPOCHS = {
    "raf_spare_parts": {42: 9, 52: 5, 62: 19},
    "yellow_trip_hourly": {42: 112, 52: 148, 62: 127},
}
DATASETS = tuple(SELECTED_EPOCHS)
TAILS = {"raf_spare_parts": 200.0, "yellow_trip_hourly": 3449.0}
GPU_UUID = "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"


def selected_job(contract, job_id):
    require(contract.get("schema") == SCHEMA and contract.get("host") == "5080"
            and contract.get("evaluation_authorized") is True,
            "Only explicitly approved5080 RAF/Taxi6 evaluation is admitted")
    jobs = contract.get("jobs", [])
    expected = {(dataset, seed) for dataset in DATASETS for seed in SELECTED_EPOCHS[dataset]}
    require(len(jobs) == 6 and len({job.get("id") for job in jobs}) == 6
            and {(job.get("dataset"), job.get("seed")) for job in jobs} == expected,
            "Exactly completed RAF/Taxi seed42/52/62 fits required")
    for job in jobs:
        require(job.get("arm") == ARM and type(job.get("seed")) is int
                and job.get("selected_epoch") == SELECTED_EPOCHS[job["dataset"]][job["seed"]]
                and job.get("id") == f"{job['dataset']}__{ARM}__seed{job['seed']}__losslog_mse",
                "Selected model, dataset, seed or epoch changed")
    selected = [job for job in jobs if job["id"] == job_id]
    require(len(selected) == 1, "Unknown selected RAF/Taxi job")
    return selected[0]


def verify_training_completion(root, contract):
    receipt = read(bound_path(root, contract["selected_training_receipt"]))
    require(receipt.get("schema") == "titantpp_shared_raf_taxi6_selected_training_v1"
            and receipt.get("status") == "complete_selected_subset"
            and receipt.get("selected_fit_count") == 6
            and receipt.get("training_validation_reference_count") == 6
            and receipt.get("actual_scientific_worker_exit0_count") == 6
            and receipt.get("all_selected_training_terminal_verified") is True
            and receipt.get("whole_raw9_complete") is False
            and receipt.get("held_out_test_evaluated") is False,
            "Selected six completed fits and exclusive originals must be verified; whole raw9 completion is not required")
    return receipt


def verify_contract(root, contract=None):
    root = Path(root).resolve()
    contract = read(root / "evaluation_contract.json") if contract is None else contract
    require(contract.get("jobs"), "Missing selected completed jobs")
    selected_job(contract, contract["jobs"][0]["id"])
    verify_training_completion(root, contract)
    training = read(bound_path(root, contract["training_contract"]))
    source = training["source"]
    source_root = frozen.resolve_under(root, contract["source_root"])
    require(len(source["files"]) == 983 and canonical(source["files"]) == source["files_sha256"],
            "Original5090 source983 closure changed")
    for relative, digest in source["files"].items():
        require(sha(frozen.resolve_under(source_root, relative)) == digest,
                "Original scientific source changed: " + relative)
    require(frozen.ROOT.resolve() == source_root, "Foreign or shadowed scientific source imported")
    frozen.verify_frozen_dependencies()
    require(set(contract["data"]) == set(DATASETS), "Only original RAF and Taxi data are authorized")
    for job in contract["jobs"]:
        original, definition, _, _ = frozen.select_job(training, "5090", job["id"])
        spec = contract["data"][job["dataset"]]
        require(original["dataset"] == job["dataset"]
                and definition["quantity_boundaries_all_train_rows"][-1] == TAILS[job["dataset"]]
                and spec["training"]["sha256"] == definition["inherited_data_identity"]["data"]["sha256"]
                and spec["populations"]["validation"] == definition["inherited_data_identity"]["populations"]["validation"],
                "Original dataset/head/Validation reference changed")
        require(set(spec["populations"]) == {"validation", "test"}, "Unexpected evaluation split")
        for population in spec["populations"].values():
            require(type(population.get("target_count")) is int and population["target_count"] > 0
                    and all(isinstance(population.get(key), str) and len(population[key]) == 64
                            for key in ("target_identity_sha256", "target_quantity_sha256")),
                    "Missing exact population identity")
        bound_path(root, spec["training"])
        bound_path(root, spec["full"])
    native = contract.get("native", {})
    require(native.get("gpu_uuid") == GPU_UUID
            and native.get("runtime_expected", {}).get("cudnn") == 91900,
            "Evaluation must use the separately approved5080 GPU/Runtime")
    return contract, training


def _command_matches(command, job_id):
    return (isinstance(command, list) and "--job" in command and "--stage" in command
            and command[command.index("--job")+1:command.index("--job")+2] == [job_id]
            and command[command.index("--stage")+1:command.index("--stage")+2] == ["train"])


def _find_matching_record(value, job_id, pid):
    if isinstance(value, dict):
        if value.get("job_id") == job_id and value.get("pid") == pid:
            return value
        for child in value.values():
            found = _find_matching_record(child, job_id, pid)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_matching_record(child, job_id, pid)
            if found is not None:
                return found
    return None


def verify_train_exit(root, job):
    recorded = read(bound_path(root, job["train_exit"]))
    actual = recorded.get("actual_worker_exit", recorded)
    require(type(actual.get("actual_os_returncode")) is int and actual["actual_os_returncode"] == 0
            and actual.get("timeout") is False and type(actual.get("pid")) is int and actual["pid"] > 0,
            "Selected scientific fit actual exit0 is missing")
    if _command_matches(recorded.get("command", actual.get("command")), job["id"]):
        return actual
    require(actual.get("job_id") == job["id"] and actual.get("stage") == "train",
            "Selected actual exit job/stage identity differs")
    observation = read(bound_path(root, recorded["original_observation"]))
    original = _find_matching_record(observation, job["id"], actual["pid"])
    require(original is not None and all(original.get(key) == actual.get(key) for key in
        ("job_id", "stage", "pid", "start_ticks", "actual_os_returncode", "timeout")),
        "Joined scientific exit is not bound to the preserved original observation")
    launch_path = bound_path(root, recorded["launch_record"])
    text = launch_path.read_text()
    try:
        launches = json.loads(text)
    except json.JSONDecodeError:
        launches = [json.loads(line) for line in text.splitlines() if line.strip()]
    launch = _find_matching_record(launches, job["id"], actual["pid"])
    require(launch is not None and launch.get("start_ticks") == actual.get("start_ticks")
            and _command_matches(launch.get("command", launch.get("argv")), job["id"]),
            "Selected observed exit is not linked to the original launched scientific command")
    return actual


def verify_job_artifacts(root, contract, training, job):
    _, definition, _, _ = frozen.select_job(training, "5090", job["id"])
    verify_train_exit(root, job)
    train_path = bound_path(root, job["train_receipt"])
    require(bound_path(root, job["training_validation_reference"]) == train_path,
            "Use the preserved training Validation reference; no synthetic parent Validation receipt")
    receipt = read(train_path)
    expected = frozen.base_receipt(training, "5090", job, "train")
    require(all(receipt.get(key) == value for key, value in expected.items())
            and receipt.get("selected_epoch") == job["selected_epoch"], "Original selected train receipt changed")
    frozen.verify_time_head_proof(receipt.get("time_head_training_verification"))
    reference = receipt["validation_reference"]
    require(reference.get("parameters_unchanged") is True
            and reference.get("state_sha256") == receipt["state_sha256"]
            and reference.get("target_identity") == receipt.get("target_identity")
            == contract["data"][job["dataset"]]["populations"]["validation"],
            "Original training Validation reference state/targets changed")
    checkpoint, history = bound_path(root, job["checkpoint"]), bound_path(root, job["selection_history"])
    require(history == checkpoint.parent / "history.json"
            and frozen.prepared.select_best_epoch(read(history)["history"]) == job["selected_epoch"]
            and sha(checkpoint) == receipt["checkpoint_sha256"], "Fixed checkpoint/history binding changed")
    bound_path(root, job["checkpoint_metadata"])
    prediction = bound_path(root, job["validation_predictions"])
    original_relative = str(Path(reference["prediction_path"]).relative_to(training["hosts"]["5090"]["root"]))
    require(receipt["files"].get(original_relative) == sha(prediction),
            "Training Validation predictions differ from the original inventory")
    require(reference["metrics"]["tail_threshold"] == TAILS[job["dataset"]], "Frozen large-quantity rule changed")
    return definition, {**receipt, "selected_checkpoint": str(checkpoint)}, reference


def runtime_check(contract, device):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    require(device == "cuda:0", "Selected data evaluation requires approved native5080 CUDA")
    configure_runtime(device, 4)
    observed = runtime_identity(device)
    expected = contract["native"]["runtime_expected"]
    require(all(observed.get(key) == value for key, value in expected.items())
            and observed.get("gpu", {}).get("uuid") == GPU_UUID,
            "Native evaluation5080 GPU/Runtime differs; original training remains5090")
    return observed


def verify_validation_receipt(root, contract, job, receipt):
    original = read(bound_path(root, job["train_receipt"]))
    reference = original["validation_reference"]
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "complete"
            and receipt.get("split") == "validation" and receipt.get("job_id") == job["id"]
            and receipt.get("contract_sha256") == canonical(contract)
            and receipt.get("selected_epoch") == job["selected_epoch"]
            and receipt.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
            and receipt.get("parameters_unchanged") is True and receipt.get("context_unchanged") is True
            and receipt.get("validation_replay_passed") is True
            and receipt.get("individual_predictions_match") is True
            and receipt.get("native_runtime_checked") is True
            and receipt.get("held_out_test_evaluated") is False
            and set(receipt.get("qualification", {})) == QUALIFICATION_KEYS
            and all(value is True for value in receipt["qualification"].values()),
            "Missing fresh5080 Validation replay of the fixed5090 selected model")
    frozen.compare_metrics(receipt["metrics"], reference["metrics"])
    require(receipt["state_sha256"] == reference["state_sha256"]
            and receipt["source_files_sha256"] == original["source_files_sha256"]
            and receipt["target_identity"] == reference["target_identity"]
            == contract["data"][job["dataset"]]["populations"]["validation"], "Validation model/source/targets changed")
    frozen.compare_prediction_files(bound_path(root, receipt["predictions"]),
                                    bound_path(root, job["validation_predictions"]))
    return receipt


def verify_validation_gate(root, contract):
    root = Path(root).resolve()
    gate = read(root / "validation_gate.json")
    require(gate.get("schema") == GATE_SCHEMA and gate.get("status") == "PASS"
            and gate.get("contract_sha256") == canonical(contract) and gate.get("job_count") == 6,
            "All six fresh5080 Validation replays must pass before either dataset Test")
    ids = {job["id"] for job in contract["jobs"]}
    require(set(gate.get("validation_receipts", {})) == ids
            and set(gate.get("validation_worker_exits", {})) == ids, "Incomplete selected-six gate")
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
                "Fresh5080 Validation actual child exit0 missing")
    return gate


def synthetic_qualification(root, job_id, *, device="cuda:0"):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    contract, training = verify_contract(root)
    job = selected_job(contract, job_id)
    output = Path(root) / "qualification" / (job_id + ".json")
    require(not output.exists(), "Synthetic qualification cannot be repeated")
    write(Path(root) / "qualification" / (job_id + ".intent.json"),
        {"contract_sha256": canonical(contract), "job_id": job_id, "started_unix": time.time()})
    runtime = runtime_check(contract, device)
    _, data, _, _ = frozen.select_job(training, "5090", job_id)
    torch.manual_seed(job["seed"])
    model, _ = frozen.build_model(data, ARM)
    model = model.to(device).eval().requires_grad_(False)
    before = canonical_state_dict_sha256(model.state_dict())
    dt, quantity, mask = [value.to(device) for value in frozen.prepared.fabricated_batch()]
    with frozen.runtime_hooks(ARM):
        checks = qualification_checks(model, dt, mask, quantity)
    require(canonical_state_dict_sha256(model.state_dict()) == before, "Synthetic inference changed model state")
    result = {"schema": SCHEMA, "status": "passed", "job_id": job_id, "seed": job["seed"],
        "contract_sha256": canonical(contract), "checks": checks, "runtime": runtime,
        "real_data_loaded": False, "checkpoint_loaded": False, "held_out_test_evaluated": False}
    write(output, result)
    return result


def evaluate_stage(root, job_id, split, *, device="cuda:0"):
    root = Path(root).resolve()
    require(split in ("validation", "test"), "Unsupported evaluation stage")
    contract, training = verify_contract(root)
    job = selected_job(contract, job_id)
    if split == "test":
        verify_validation_gate(root, contract)
    verify_synthetic_qualifications(root, contract)
    folder = root / "results" / job_id
    require(not (folder / (split+"_receipt.json")).exists()
            and not (folder / (split+"_failure.json")).exists(), "Existing evaluation cannot be repeated")
    write(folder / (split+".intent.json"), {"contract_sha256": canonical(contract),
        "job_id": job_id, "split": split, "started_unix": time.time()})
    try:
        definition, train_receipt, reference = verify_job_artifacts(root, contract, training, job)
        runtime = runtime_check(contract, device)
        import polars as pl
        spec = contract["data"][job["dataset"]]
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
        checkpoint = bound_path(root, job["checkpoint"])
        from simple_lab_test.search.common.runner import torch_load_checkpoint
        payload = torch_load_checkpoint(checkpoint, map_location="cpu")
        validate_checkpoint_metadata(payload, read(bound_path(root, job["checkpoint_metadata"])), job)
        output = folder / (split+"_predictions.jsonl.gz")
        result = evaluate_frame(definition, job, train_receipt, root, frame, split, output,
                                spec["populations"][split], device=device)
        if split == "validation":
            frozen.compare_metrics(result["metrics"], reference["metrics"])
            frozen.compare_prediction_files(output, bound_path(root, job["validation_predictions"]))
        after_runtime = runtime_check(contract, device)
        require(runtime == after_runtime and sha(data_path) == data_ref["sha256"], "Evaluation Runtime/input changed")
        verify_contract(root, contract)
        verify_job_artifacts(root, contract, training, job)
        if split == "test":
            verify_validation_gate(root, contract)
        receipt = {"schema": SCHEMA, "status": "complete", "contract_sha256": canonical(contract),
            "job_id": job_id, "dataset": job["dataset"], "arm": ARM, "seed": job["seed"],
            "selected_epoch": job["selected_epoch"], "split": split, **result,
            "checkpoint_sha256": job["checkpoint"]["sha256"],
            "source_files_sha256": training["source"]["files_sha256"],
            "retrained": False, "reselected": False, "held_out_test_evaluated": split == "test",
            "validation_replay_passed": split == "validation", "individual_predictions_match": split == "validation",
            "native_runtime_checked": True, "runtime": runtime, "runtime_after": after_runtime,
            "training_host": "5090", "evaluation_host": "5080", "whole_raw9_complete_asserted": False,
            "host_difference_limitation": "Original5090/cuDNN92000 checkpoints replayed on5080/cuDNN91900; fixed1e-5 tolerance is unchanged",
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
