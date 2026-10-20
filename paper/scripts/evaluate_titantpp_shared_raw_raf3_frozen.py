"""Fixed RAF Shared15 checkpoints: native Validation replay gate, then Test3.

This separate evaluator reuses the immutable raw9 model/restoration and metric
code. It has no optimizer, training, checkpoint write or selection route.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gzip
import json
from pathlib import Path
import sys
import time
import traceback

from paper.scripts import run_titantpp_shared_history_raw9_5090 as frozen

SCHEMA = "titantpp_shared_raw_raf3_frozen_v1"
GATE_SCHEMA = "titantpp_shared_raw_raf3_validation_gate_v1"
DATASET = "raf_spare_parts"
ARM = "titantpp_history_mlp_shared15_v1"
SELECTED_EPOCHS = {42: 9, 52: 5, 62: 19}
GPU_UUID = "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"
QUALIFICATION_KEYS = frozenset({
    "target_quantity_blocked_for_prediction", "target_quantity_blocked_for_time_loss",
    "target_gap_blocked_for_quantity", "single_batch_prediction_matches",
    "single_batch_time_loss_matches", "finite_outputs", "parameters_unchanged", "context_unchanged"})
require, read, sha, canonical = frozen.require, frozen.read, frozen.sha_file, frozen.sha_json
write = frozen.write_exclusive


def bound_path(root, reference):
    require(isinstance(reference, dict) and set(reference) >= {"path", "sha256"}, "Missing bound artifact")
    path = frozen.resolve_under(root, reference["path"])
    require(path.is_file() and sha(path) == reference["sha256"], "Bound artifact changed: " + str(path))
    return path


def selected_job(contract, job_id):
    require(contract.get("schema") == SCHEMA and contract.get("host") == "5090"
            and contract.get("evaluation_authorized") is True, "Foreign or unauthorized RAF3 evaluation")
    jobs = contract.get("jobs", [])
    require(len(jobs) == 3 and len({job.get("id") for job in jobs}) == 3
            and {job.get("seed") for job in jobs} == set(SELECTED_EPOCHS), "Exactly RAF42/52/62 required")
    for job in jobs:
        require(job.get("dataset") == DATASET and job.get("arm") == ARM
                and type(job.get("seed")) is int
                and job.get("selected_epoch") == SELECTED_EPOCHS[job["seed"]]
                and job.get("id") == f"{DATASET}__{ARM}__seed{job['seed']}__losslog_mse",
                "Frozen RAF model/seed/epoch changed")
    selected = [job for job in jobs if job["id"] == job_id]
    require(len(selected) == 1, "Unknown RAF3 job")
    return selected[0]


def verify_training_completion(root, contract):
    completion = read(bound_path(root, contract["training_completion"]))
    require(completion.get("status") == "complete" and completion.get("new_fit_count") == 9
            and completion.get("validation_replay_count") == 9
            and completion.get("actual_scientific_worker_exit0_count") == 18
            and completion.get("actual_dispatcher_exit0_count") == 1
            and completion.get("all_training_terminal_verified") is True
            and completion.get("held_out_test_evaluated") is False,
            "All original raw9 fits, Validation replays and collection must complete before evaluation")
    return completion


def verify_contract(root, contract=None):
    root = Path(root).resolve()
    contract = read(root / "evaluation_contract.json") if contract is None else contract
    require(contract.get("jobs"), "Missing frozen RAF jobs")
    selected_job(contract, contract["jobs"][0]["id"])
    verify_training_completion(root, contract)
    training = read(bound_path(root, contract["training_contract"]))
    source_root = frozen.resolve_under(root, contract["source_root"])
    source = training["source"]
    require(len(source["files"]) == 983 and canonical(source["files"]) == source["files_sha256"],
            "Immutable training source983 closure changed")
    for relative, digest in source["files"].items():
        require(sha(frozen.resolve_under(source_root, relative)) == digest,
                "Immutable scientific source changed: " + relative)
    require(frozen.ROOT.resolve() == source_root, "Evaluator imported a foreign scientific source root")
    frozen.verify_frozen_dependencies()
    for job in contract["jobs"]:
        original, definition, _, _ = frozen.select_job(training, "5090", job["id"])
        require(original["dataset"] == DATASET
                and definition["quantity_boundaries_all_train_rows"][-1] == 200.0,
                "RAF original quantity definition changed")
    data = contract["data"]
    require(data["training"]["sha256"] == definition["inherited_data_identity"]["data"]["sha256"]
            and data["populations"]["validation"] == definition["inherited_data_identity"]["populations"]["validation"],
            "Evaluation Train/Validation input differs from the immutable original training contract")
    require(set(data["populations"]) == {"validation", "test"}, "Only RAF Validation/Test populations allowed")
    for split in ("validation", "test"):
        population = data["populations"][split]
        require(type(population.get("target_count")) is int and population["target_count"] > 0
                and all(isinstance(population.get(key), str) and len(population[key]) == 64
                        for key in ("target_identity_sha256", "target_quantity_sha256")), "Population binding missing")
    bound_path(root, data["training"])
    bound_path(root, data["full"])
    return contract, training


def verify_job_artifacts(root, contract, training, job):
    _, definition, _, _ = frozen.select_job(training, "5090", job["id"])
    receipt = read(bound_path(root, job["train_receipt"]))
    reference = read(bound_path(root, job["validation_receipt"]))
    expected = frozen.base_receipt(training, "5090", job, "train")
    require(all(receipt.get(key) == value for key, value in expected.items()), "Original train receipt changed")
    require(reference.get("schema") == frozen.SCHEMA and reference.get("stage") == "validation"
            and reference.get("status") == "complete" and reference.get("validation_replay_passed") is True
            and reference.get("parameters_unchanged") is True
            and reference.get("contract_sha256") == canonical(training)
            and reference.get("job_id") == job["id"]
            and receipt["selected_epoch"] == reference["selected_epoch"] == job["selected_epoch"],
            "Completed original Validation identity changed")
    checkpoint = bound_path(root, job["checkpoint"])
    history = bound_path(root, job["selection_history"])
    require(history == checkpoint.parent / "history.json", "Bound history must accompany the selected binary")
    require(frozen.prepared.select_best_epoch(read(history)["history"]) == job["selected_epoch"],
            "Epoch is not the first strict finite full-Validation raw RMSE minimum")
    require(sha(checkpoint) == receipt["checkpoint_sha256"] == reference["checkpoint_sha256"],
            "Original selected binary changed")
    bound_path(root, job["checkpoint_metadata"])
    bound_path(root, job["validation_predictions"])
    require(job["validation_predictions"]["sha256"] == reference["files"].get(
        str(Path(reference["prediction_path"]).relative_to(Path(training["hosts"]["5090"]["root"])))),
        "Original Validation predictions not bound to the original receipt inventory")
    frozen.compare_metrics(reference["metrics"], receipt["validation_reference"]["metrics"])
    require(reference["target_identity"] == receipt["validation_reference"]["target_identity"]
            == contract["data"]["populations"]["validation"], "Original Validation targets changed")
    # Relocation is an in-memory operational mapping, never an original edit.
    relocated = {**receipt, "selected_checkpoint": str(checkpoint)}
    return definition, relocated, reference


def runtime_check(training, device):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    require(device == "cuda:0", "Real-data evaluation requires the original native5090 CUDA runtime")
    configure_runtime(device, 4)
    observed = runtime_identity(device)
    expected = training["hosts"]["5090"]["runtime_expected"]
    require(all(observed.get(key) == value for key, value in expected.items())
            and observed.get("gpu", {}).get("uuid") == GPU_UUID,
            "Native GPU/Runtime changed")
    return observed


def validate_frame(frame, split):
    import numpy as np
    import polars as pl
    required = {"oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split"}
    permitted = {"train", "validation"} | ({"test"} if split == "test" else set())
    require(required <= set(frame.columns) and frame.height > 0
            and set(frame["chronological_split"].unique().to_list()) <= permitted, "Foreign or empty population")
    require(not frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().any()).item(), "Duplicate target key")
    for name in ("delta_t", "demand_qty", "seq"):
        values = frame[name].to_numpy().astype(np.float64)
        require(bool(np.isfinite(values).all()) and (name == "seq" or bool((values >= 0).all())), "Invalid raw values")
    ranks = frame.with_columns(pl.col("chronological_split").replace_strict(
        {"train": 0, "validation": 1, "test": 2}).alias("_rank")).with_columns(
        pl.col("_rank").cum_max().over("oper_part_no").alias("_max_rank"))
    require(ranks.filter(pl.col("_rank") < pl.col("_max_rank")).height == 0, "Chronological split order changed")


def qualification_checks(model, dt, mask, quantity):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    before = canonical_state_dict_sha256(model.state_dict())
    context_before = context_identity(model)
    require(not any(module.training for module in model.modules())
            and not any(parameter.requires_grad for parameter in model.parameters()),
            "Frozen evaluation must disable training and parameter gradients")
    with torch.no_grad():
        result = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
        changed_q, changed_dt = quantity.clone(), dt.clone()
        changed_q[:, -1] += 12345
        changed_dt[:, -1] += 123
        q = target_outputs(model, dt, mask, changed_q, lambda_log_qty=1.0)
        t = target_outputs(model, changed_dt, mask, quantity, lambda_log_qty=1.0)
        single = target_outputs(model, dt[:1], mask[:1], quantity[:1], lambda_log_qty=1.0)
        checks = {
            "target_quantity_blocked_for_prediction": torch.allclose(result["pred_qty"], q["pred_qty"], rtol=1e-5, atol=1e-5),
            "target_quantity_blocked_for_time_loss": torch.allclose(result["time_loss"], q["time_loss"], rtol=1e-5, atol=1e-5),
            "target_gap_blocked_for_quantity": torch.allclose(result["pred_qty"], t["pred_qty"], rtol=1e-5, atol=1e-5),
            "single_batch_prediction_matches": torch.allclose(result["pred_qty"][:1], single["pred_qty"], rtol=1e-5, atol=1e-5),
            "single_batch_time_loss_matches": torch.allclose(result["time_loss"][:1], single["time_loss"], rtol=1e-5, atol=1e-5),
            "finite_outputs": all(bool(torch.isfinite(result[key]).all()) for key in ("pred_qty", "time_loss")),
            "parameters_unchanged": canonical_state_dict_sha256(model.state_dict()) == before,
            "context_unchanged": context_identity(model) == context_before,
        }
    require(all(checks.values()), "Synthetic or first-batch causal qualification failed")
    return checks


def context_identity(model):
    """Include the inherited transient attention context outside state_dict."""
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "_ctx_mem"):
            value = module._ctx_mem
            result[name] = None if value is None else canonical_state_dict_sha256({"context": value})
    return result


def validate_checkpoint_metadata(payload, summary, job):
    """Bind the unchanged original summary instead of generating new metadata."""
    require(summary.get("status") == "success"
            and summary.get("checkpoint_state_sha256") == payload.get("model_state_sha256")
            and summary.get("best_epoch") == payload.get("best_epoch") == job["selected_epoch"],
            "Original selected checkpoint summary changed")
    keys = ("encoder_config", "interface_meta", "resume_identity", "initial_state_sha256",
            "source_revision", "source_revision_history", "checkpoint_monitor",
            "checkpoint_monitor_history_key", "checkpoint_selection", "selection_formula",
            "selected_metric_value", "seed", "backbone", "variant", "evaluation_scope",
            "held_out_test_evaluated")
    require(all(key in summary and key in payload and summary[key] == payload[key] for key in keys),
            "Selected binary metadata differs from original training summary")


def evaluate_frame(definition, job, train_receipt, root, frame, split, output, expected_population, *, device):
    """Frozen selected-model inference; this helper also accepts fabricated CPU frames."""
    import numpy as np
    import torch
    from paper.scripts.count_aware_tpp_backbone import core
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(split in ("validation", "test"), "Only Validation and Test are supported")
    validate_frame(frame, split)
    frame = core.prepare_count_frame(frame)
    loader = make_loader(frame, target_split=split, shuffle=False, generator=None,
        **{key: definition["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")})
    truth, identity = frozen.population_identity(loader.dataset, split)
    frozen.verify_population(identity, expected_population)
    model, checkpoint, digest = frozen._load_selected(definition, job, train_receipt, root, device)
    model.eval().requires_grad_(False)
    context_before = context_identity(model)
    metrics = frozen.MetricAccumulator(200.0)
    simple = {name: frozen.MetricAccumulator(200.0) for name in ("last", "history_mean")}
    processed, qualification = 0, {}
    with Path(output).open("xb") as binary, gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as compressed:
        with frozen.runtime_hooks(ARM), torch.no_grad():
            for batch_index, (_, dts, mask, part_ids, quantities) in enumerate(loader):
                dts, mask, quantities = (value.to(device) for value in (dts, mask, quantities))
                if batch_index == 0:
                    qualification = qualification_checks(model, dts, mask, quantities)
                value = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
                arrays = {key: value[key].detach().cpu().numpy()
                          for key in ("true_qty", "pred_qty", "time_loss", "history_length")}
                size = len(arrays["true_qty"])
                require(np.array_equal(arrays["true_qty"].astype(np.float64), truth[processed:processed+size]),
                        "Target quantity/order changed")
                metrics.add(arrays["true_qty"], arrays["pred_qty"], arrays["time_loss"])
                observed = mask.clone()
                observed[:, -1] = False
                history_qty = quantities.double()
                history_mean = (torch.where(observed, history_qty, 0.0).sum(1) / observed.sum(1)).cpu().numpy()
                for name, prediction in (("last", history_qty[:, -2].cpu().numpy()), ("history_mean", history_mean)):
                    simple[name].add(arrays["true_qty"], prediction, np.zeros(size))
                for index in range(size):
                    part, end = loader.dataset.index[processed+index]
                    require(int(part_ids[index]) == int(part), "Target part order changed")
                    row = {"target_index": processed+index, "entity_id": str(loader.dataset.parts[part]),
                        "target_seq": int(loader.dataset.seq_lists[part][end+1]), "split": split,
                        "raw_quantity": float(arrays["true_qty"][index]), "predicted_raw_quantity": float(arrays["pred_qty"][index]),
                        "time_nll": float(arrays["time_loss"][index]), "history_length": int(arrays["history_length"][index])}
                    compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode())
                processed += size
    require(processed == identity["target_count"]
            and canonical_state_dict_sha256(model.state_dict()) == digest
            and context_identity(model) == context_before, "Incomplete inference or model/context state mutation")
    require(sha(checkpoint) == train_receipt["checkpoint_sha256"], "Inference changed selected checkpoint binary")
    baselines = {name: accumulator.finish() for name, accumulator in simple.items()}
    for result in baselines.values():
        result["time_metric"] = "N/A_no_time_model"
        for group in ("overall", "tail"):
            result[group]["time_nll"] = result[group]["time_nll_sum"] = None
    return {"metrics": metrics.finish(), "simple_quantity_baselines": baselines,
            "target_identity": identity, "prediction_rows": processed,
            "state_sha256": digest, "parameters_unchanged": True, "context_unchanged": True,
            "qualification": qualification}


def verify_validation_receipt(root, contract, job, receipt):
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "complete"
            and receipt.get("split") == "validation" and receipt.get("job_id") == job["id"]
            and receipt.get("contract_sha256") == canonical(contract)
            and receipt.get("selected_epoch") == job["selected_epoch"]
            and receipt.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
            and receipt.get("parameters_unchanged") is True
            and receipt.get("context_unchanged") is True
            and receipt.get("validation_replay_passed") is True
            and receipt.get("individual_predictions_match") is True
            and receipt.get("native_runtime_checked") is True
            and receipt.get("held_out_test_evaluated") is False
            and set(receipt.get("qualification", {})) == QUALIFICATION_KEYS
            and all(value is True for value in receipt["qualification"].values()),
            "Missing or foreign native Validation replay receipt")
    reference = read(bound_path(root, job["validation_receipt"]))
    frozen.compare_metrics(receipt["metrics"], reference["metrics"])
    require(receipt["state_sha256"] == reference["state_sha256"]
            and receipt["source_files_sha256"] == reference["source_files_sha256"]
            and receipt["target_identity"] == reference["target_identity"]
            == contract["data"]["populations"]["validation"], "Replay state/targets changed")
    prediction = bound_path(root, receipt["predictions"])
    frozen.compare_prediction_files(prediction, bound_path(root, job["validation_predictions"]))
    return receipt


def verify_validation_gate(root, contract):
    root = Path(root).resolve()
    gate = read(root / "validation_gate.json")
    require(gate.get("schema") == GATE_SCHEMA and gate.get("status") == "PASS"
            and gate.get("contract_sha256") == canonical(contract) and gate.get("job_count") == 3,
            "All RAF3 native Validation replays must pass before Test")
    ids = {job["id"] for job in contract["jobs"]}
    require(set(gate.get("validation_receipts", {})) == ids
            and set(gate.get("validation_worker_exits", {})) == ids, "Incomplete RAF3 gate")
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
                "Validation actual worker exit0 not verified")
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
    runtime = runtime_check(training, device)
    _, data, _, _ = frozen.select_job(training, "5090", job_id)
    torch.manual_seed(job["seed"])
    model, _ = frozen.build_model(data, ARM)
    model = model.to(device).eval().requires_grad_(False)
    before = canonical_state_dict_sha256(model.state_dict())
    dt, quantity, mask = [value.to(device) for value in frozen.prepared.fabricated_batch()]
    with frozen.runtime_hooks(ARM):
        checks = qualification_checks(model, dt, mask, quantity)
    require(canonical_state_dict_sha256(model.state_dict()) == before, "Synthetic inference mutated model")
    result = {"schema": SCHEMA, "status": "passed", "job_id": job_id, "seed": job["seed"],
        "contract_sha256": canonical(contract), "checks": checks, "runtime": runtime,
        "real_data_loaded": False, "checkpoint_loaded": False, "held_out_test_evaluated": False}
    write(output, result)
    return result


def verify_synthetic_qualifications(root, contract):
    for job in contract["jobs"]:
        result = read(Path(root) / "qualification" / (job["id"] + ".json"))
        require(result.get("schema") == SCHEMA and result.get("status") == "passed"
                and result.get("contract_sha256") == canonical(contract)
                and result.get("job_id") == job["id"] and result.get("seed") == job["seed"]
                and result.get("real_data_loaded") is False and result.get("checkpoint_loaded") is False
                and result.get("held_out_test_evaluated") is False
                and set(result.get("checks", {})) == QUALIFICATION_KEYS
                and all(value is True for value in result["checks"].values()),
                "All three fresh native synthetic qualifications must pass")


def evaluate_stage(root, job_id, split, *, device="cuda:0"):
    root = Path(root).resolve()
    require(split in ("validation", "test"), "Unsupported evaluation stage")
    contract, training = verify_contract(root)
    job = selected_job(contract, job_id)
    if split == "test":
        verify_validation_gate(root, contract)  # Before checkpoint load or Test materialization.
    verify_synthetic_qualifications(root, contract)
    folder = root / "results" / job_id
    require(not (folder / (split+"_receipt.json")).exists()
            and not (folder / (split+"_failure.json")).exists(), "Previous evaluation cannot be repeated")
    write(folder / (split+".intent.json"), {"contract_sha256": canonical(contract),
        "job_id": job_id, "split": split, "started_unix": time.time()})
    try:
        definition, train_receipt, reference = verify_job_artifacts(root, contract, training, job)
        runtime = runtime_check(training, device)
        import polars as pl
        data_ref = contract["data"]["training" if split == "validation" else "full"]
        data_path = bound_path(root, data_ref)
        lazy = pl.scan_parquet(data_path)
        if split == "validation":
            lazy = lazy.filter(pl.col("chronological_split").is_in(["train", "validation"]))
        frame = lazy.collect().sort(["oper_part_no", "seq"])
        if split == "test":
            training_frame = pl.read_parquet(bound_path(root, contract["data"]["training"])).sort(["oper_part_no", "seq"])
            require(set(training_frame.columns) <= set(frame.columns)
                    and frame.filter(pl.col("chronological_split").is_in(["train", "validation"]))
                    .select(training_frame.columns).equals(training_frame), "RAF full parent differs from frozen Train/Validation export")
        checkpoint = bound_path(root, job["checkpoint"])
        from simple_lab_test.search.common.runner import torch_load_checkpoint
        payload = torch_load_checkpoint(checkpoint, map_location="cpu")
        validate_checkpoint_metadata(payload, read(bound_path(root, job["checkpoint_metadata"])), job)
        output = folder / (split+"_predictions.jsonl.gz")
        result = evaluate_frame(definition, job, train_receipt, root, frame, split, output,
            contract["data"]["populations"][split], device=device)
        if split == "validation":
            frozen.compare_metrics(result["metrics"], reference["metrics"])
            frozen.compare_prediction_files(output, bound_path(root, job["validation_predictions"]))
        after_runtime = runtime_check(training, device)
        require(runtime == after_runtime and sha(data_path) == data_ref["sha256"], "Runtime/input changed during inference")
        verify_contract(root, contract)
        verify_job_artifacts(root, contract, training, job)
        if split == "test":
            verify_validation_gate(root, contract)
        receipt = {"schema": SCHEMA, "status": "complete", "contract_sha256": canonical(contract),
            "job_id": job_id, "dataset": DATASET, "arm": ARM, "seed": job["seed"],
            "selected_epoch": job["selected_epoch"], "split": split, **result,
            "checkpoint_sha256": job["checkpoint"]["sha256"],
            "source_files_sha256": training["source"]["files_sha256"],
            "retrained": False, "reselected": False, "held_out_test_evaluated": split == "test",
            "validation_replay_passed": split == "validation", "individual_predictions_match": split == "validation",
            "native_runtime_checked": True, "runtime": runtime, "runtime_after": after_runtime,
            "predictions": {"path": str(output.relative_to(root)), "sha256": sha(output)},
            "completed_unix": time.time()}
        write(folder / (split+"_receipt.json"), receipt)
        return receipt
    except BaseException as error:
        write(folder / (split+"_failure.json"), {"schema": SCHEMA, "status": "failed",
            "job_id": job_id, "split": split, "error": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "contract_sha256": canonical(contract)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--mode", choices=("evaluate", "synthetic"), default="evaluate")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args(argv)
    with redirect_stdout(sys.stderr):
        if args.mode == "synthetic":
            result = synthetic_qualification(args.root, args.job, device=args.device)
        else:
            require(args.split is not None, "Validation or Test split required")
            result = evaluate_stage(args.root, args.job, args.split, device=args.device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
