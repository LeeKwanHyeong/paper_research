"""Approved fresh shared-history Cap fits and fixed-checkpoint Validation replay.

The preparation source remains immutable.  This separate worker accepts only
the explicit 5080 execution contract, two candidates and Train/Validation data.
One invocation owns one exclusive stage; the separate supervisor admits CUDA
resources and limits the queue to three workers.  No Test route is exposed.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import torch

from models.TPPs.CountAwareTitanSharedHistoryMLP import (
    ARMS, MODE_BY_ARM, metadata, validate_checkpoint,
)
from paper.scripts import run_titantpp_shared_history_cap as prepared
from paper.scripts import run_titantpp_mlp_cap36 as metrics_parent


SCHEMA = "titantpp_shared_history_cap_execution_stage_v1"
CONTRACT_SCHEMA = "titantpp_shared_history_cap_execution_v1"
DATASETS, SEEDS = prepared.DATASETS, prepared.SEEDS
VARIANT, MONITOR = prepared.VARIANT, prepared.MONITOR
ROOT = Path(__file__).resolve().parents[2]
PREPARATION_WORKER_SHA256 = "508cd9f26d75c911f6c5c10f341679c271c4a2cea5846206dcae6bcc66b5e55f"
MODEL_SHA256 = "eee18d3abe199f0c71d80da35acf01cf19acd8f56291ff7cfd33fd86d8bc82a2"
REPLAY_RTOL = REPLAY_ATOL = 1e-5
INACTIVE_TAIL = metrics_parent.INACTIVE_TAIL
require, read, sha_file, sha_json = (
    metrics_parent.require, metrics_parent.read, metrics_parent.sha_file, metrics_parent.sha_json)
write_exclusive, resolve_under = metrics_parent.write_exclusive, metrics_parent.resolve_under
file_inventory, verify_inventory = metrics_parent.file_inventory, metrics_parent.verify_inventory
population_identity, verify_population = metrics_parent.population_identity, metrics_parent.verify_population
MetricAccumulator = metrics_parent.MetricAccumulator
compare_metrics, compare_prediction_files = metrics_parent.compare_metrics, metrics_parent.compare_prediction_files
TimeHeadGuard = prepared.TimeHeadGuard


def verify_frozen_dependencies():
    require(sha_file(Path(prepared.__file__)) == PREPARATION_WORKER_SHA256,
            "Immutable preparation worker changed")
    require(sha_file(ROOT / "models/TPPs/CountAwareTitanSharedHistoryMLP.py") == MODEL_SHA256,
            "Immutable shared-history model changed")


def candidate_labels(arm):
    require(arm in ARMS, "Only the two approved candidates are admitted")
    return {"candidate_name": arm, "history_relation_sharing": MODE_BY_ARM[arm],
            "history_relation_parameter_count": 9600, "correction_parameter_count": 9600,
            "architecture_metadata": metadata(mode=MODE_BY_ARM[arm]), **prepared.input_identity(),
            "quantity_only": True, "time_nll_is_diagnostic": True,
            "stored_baselines_retrained_or_reevaluated": False}


def select_job(contract, host, job_id):
    """Reject unsupported conditions before resolving or loading any input."""
    require(isinstance(contract, dict) and contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("scope") == "execution",
            "A separate approved shared-history execution contract is required")
    permission = contract.get("execution_authorization", {})
    require(permission == {"real_data_training": True, "real_data_inference": True,
                           "remote_execution": True, "test_evaluation": False},
            "Execution authorization must allow only approved Train/Validation")
    require(host == "5080" and set(contract.get("hosts", {})) == {"5080"},
            "Only the approved 5080 host is admitted")
    require(contract.get("common_condition") == {
        "time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.0},
        "Common input or objective changed")
    jobs = contract.get("jobs", [])
    expected = {(dataset, arm, seed) for dataset in DATASETS for arm in ARMS for seed in SEEDS}
    require(len(jobs) == 18 and {(j.get("dataset"), j.get("arm"), j.get("seed"))
            for j in jobs} == expected, "The approved 18-job matrix changed")
    for job in jobs:
        require(job.get("host") == host and type(job.get("seed")) is int
                and job.get("time_input") == "log1p" and job.get("loss_mode") == "log_mse"
                and type(job.get("time_alpha")) in (int, float) and job["time_alpha"] == 0.0,
                "A job changed host, seed, input or loss")
        expected_id = f"{job['dataset']}__{job['arm']}__seed{job['seed']}__losslog_mse"
        require(job.get("id") == expected_id, "A job ID changed its scientific identity")
    selected = [j for j in jobs if j["id"] == job_id]
    require(len(selected) == 1, "Job ID must select one approved condition")
    job = selected[0]
    datasets = contract.get("datasets", [])
    require(len(datasets) == 3 and {d.get("dataset_id") for d in datasets} == set(DATASETS),
            "Cap dataset matrix changed")
    data = next(d for d in datasets if d["dataset_id"] == job["dataset"])
    require(data["optimizer"] == {"name": "AdamW", "lr": .001,
            "weight_decay": .01, "grad_clip": 1.0}, "Common optimizer changed")
    for key, expected_value in (("maximum_epochs", 300), ("minimum_epochs", 40),
                               ("patience", 40), ("batch_size", 128), ("monitor", MONITOR)):
        require(contract["training"].get(key) == expected_value,
                "Common training setting changed: " + key)
    require(contract["training"].get("warm_start", False) is False,
            "Warm-start/resume training is not admitted")
    model, loader = data["model"], data["loader"]
    require(model["hidden_dim"] == 64 and model["quantity_variant"] == VARIANT
            and model["lambda_log_qty"] == 1.0 and model["lambda_tail"] == 0.0
            and model["time_head_lr_multiplier"] == 1.0
            and model["time_head_mode"] == "heteroscedastic_lognormal_duration",
            "Common head or loss configuration changed")
    require(loader["batch_size"] == 128 and loader["num_workers"] == 0
            and loader["drop_last"] is False and loader["train_shuffle"] is True
            and loader["validation_shuffle"] is False, "Common loader changed")
    require(data["cap_value"] == {DATASETS[0]: 65.0, DATASETS[1]: 10.0,
                                  DATASETS[2]: 20.0}[data["dataset_id"]], "Cap value changed")
    bounds = data["quantity_boundaries_all_train_rows"]
    require(len(bounds) == 4 and all(math.isfinite(float(v)) for v in bounds)
            and all(a <= b for a, b in zip(bounds, bounds[1:]))
            and bounds[-1] == data["cap_value"], "Frozen quantity boundaries changed")
    identity = data["inherited_data_identity"]
    require(set(identity["populations"]) == {"train", "validation"},
            "Only Train/Validation input populations are allowed")
    root = Path(contract["hosts"][host]["root"]).resolve()
    return job, data, root, root / "run" / job_id


def training_args(contract, data, output, arm):
    from paper.scripts import run_multilag_detail_execution as base
    from models.TPPs.CountAwareTitanSharedHistoryMLP import role_for_arm
    args = base.training_args({**contract, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.time_alpha = 0.0
    args.execution_role = "fresh_shared_history_cap_validation"
    for key, value in {**INACTIVE_TAIL, **prepared.input_identity()}.items():
        setattr(args, key, value)
    return args


def training_interface(contract, data, frame, arm):
    from paper.scripts import run_multilag_detail_execution as base
    interface = base.time_interface(data, frame, contract)
    interface.update(prepared.input_identity())
    interface["backbone_design"] = {
        "schema": CONTRACT_SCHEMA, "arm": arm, "architecture": contract["architecture"],
        "design_sha256": contract["design_sha256"], "training": contract["training"],
        "cap_value": data["cap_value"], "correction": metadata(mode=MODE_BY_ARM[arm])}
    return interface


def base_receipt(contract, host, job, stage):
    require(stage in ("train", "validation"), "Only Train/Validation is admitted")
    return {"schema": SCHEMA, "status": "complete", "stage": stage, "host": host,
        "job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
        "contract_sha256": sha_json(contract), "held_out_test_evaluated": False,
        "source_files_sha256": contract["source"]["files_sha256"],
        "source_revision": contract["source"]["base_git_revision"],
        "design_sha256": contract["design_sha256"],
        **candidate_labels(job["arm"])}


def verify_train_receipt(contract, host, job, root, receipt):
    require(all(receipt.get(key) == value for key, value in
                base_receipt(contract, host, job, "train").items()), "Training receipt binding changed")
    verify_inventory(root, receipt.get("files"))
    proof = prepared.verify_time_head_proof(receipt.get("time_head_no_training_verification"))
    require(receipt["summary"].get("time_head_no_training_verification") == proof,
            "Fit and summary time-head proofs differ")


def _load_selected(data, job, receipt, root, device):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    path = resolve_under(root, receipt["selected_checkpoint"])
    require(sha_file(path) == receipt["checkpoint_sha256"], "Selected binary changed")
    checkpoint = torch_load_checkpoint(path, map_location="cpu")
    validate_checkpoint(checkpoint, job["arm"])
    require(checkpoint["seed"] == job["seed"] and checkpoint["best_epoch"] == receipt["selected_epoch"]
            and checkpoint["checkpoint_monitor"] == MONITOR, "Selected seed/epoch/monitor changed")
    require(all(checkpoint.get(k) == v for k, v in prepared.input_identity().items()),
            "Checkpoint input or objective identity changed")
    arguments = checkpoint["resume_identity"]["arguments"]
    require(arguments.get("dataset_contract") == job["dataset"]
            and arguments.get("data_sha256") == data["inherited_data_identity"]["data"]["sha256"]
            and arguments.get("split_manifest_sha256") == data["inherited_data_identity"]["split_manifest"]["sha256"]
            and checkpoint["interface_meta"].get("source_files_sha256") == receipt["source_files_sha256"],
            "Selected checkpoint data or source binding changed")
    proof = checkpoint["time_head_no_training_verification"]
    prepared.verify_time_head_proof(proof, checkpoint["model_state_dict"])
    prepared.frozen_zero.verify_checkpoint_fit_proof(proof, receipt["time_head_no_training_verification"])
    digest = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(digest == checkpoint["model_state_sha256"] == receipt["state_sha256"],
            "Selected tensor digest changed")
    history_path = path.parent / "history.json"
    history = read(history_path)["history"]
    require(prepared.select_best_epoch(history) == checkpoint["best_epoch"],
            "Selected epoch differs from first strict finite full-Validation minimum")
    model, encoder = prepared.build_model(data, job["arm"])
    require(encoder == checkpoint["encoder_config"], "Reconstructed encoder metadata changed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), path, digest


def evaluate_selected(data, job, receipt, root, frame, split, output, device="cuda:0"):
    """Persist exact targets and six metrics with the frozen Validation loader."""
    import gzip
    import numpy as np
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(split == "validation", "No held-out evaluation path exists")
    model, checkpoint, digest = _load_selected(data, job, receipt, root, device)
    loader = make_loader(frame, target_split="validation", shuffle=False, generator=None,
        **{key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")})
    truth, identity = population_identity(loader.dataset, "validation")
    verify_population(identity, data["inherited_data_identity"]["populations"]["validation"])
    metrics, processed = MetricAccumulator(data["cap_value"]), 0
    with Path(output).open("xb") as binary, gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as compressed:
        with torch.no_grad():
            for _, dts, mask, part_ids, quantities in loader:
                value = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.0)
                arrays = {key: value[key].detach().cpu().numpy()
                          for key in ("true_qty", "pred_qty", "time_loss", "history_length")}
                size = len(arrays["true_qty"])
                require(np.array_equal(arrays["true_qty"].astype(np.float64), truth[processed:processed + size]),
                        "Validation target truth changed")
                metrics.add(arrays["true_qty"], arrays["pred_qty"], arrays["time_loss"])
                for index in range(size):
                    part, end = loader.dataset.index[processed + index]
                    require(int(part_ids[index]) == int(part), "Validation target order changed")
                    row = {"target_index": processed + index, "entity_id": str(loader.dataset.parts[part]),
                        "target_seq": int(loader.dataset.seq_lists[part][end + 1]), "split": "validation",
                        "raw_quantity": float(arrays["true_qty"][index]),
                        "predicted_raw_quantity": float(arrays["pred_qty"][index]),
                        "time_nll": float(arrays["time_loss"][index]),
                        "history_length": int(arrays["history_length"][index])}
                    compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
                processed += size
    require(processed == identity["target_count"], "Incomplete Validation population")
    require(canonical_state_dict_sha256(model.state_dict()) == digest, "Replay changed model state")
    require(sha_file(checkpoint) == receipt["checkpoint_sha256"], "Replay changed checkpoint binary")
    result = {"metrics": metrics.finish(), "target_identity": identity,
        "prediction_path": str(Path(output).resolve()), "prediction_rows": processed,
        "parameters_unchanged": True, "state_sha256": digest}
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_stage(contract, host, job_id, stage):
    require(stage in ("train", "validation"), "New Test execution is forbidden")
    verify_frozen_dependencies()
    job, data, root, folder = select_job(contract, host, job_id)
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / f"{stage}_receipt.json"
    require(not receipt_path.exists(), "Completed stages cannot be repeated")
    if stage == "train":
        require(not (folder / "runs").exists(), "Only fresh fits; resume/retry is unavailable")
    write_exclusive(folder / f"{stage}.intent.json", {"contract_sha256": sha_json(contract),
        "job_id": job_id, "stage": stage, "started_unix": time.time()})
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts import run_multilag_detail_execution as base
    configure_runtime("cuda:0", 4)
    if stage == "train":
        from paper.scripts.count_aware_tpp_backbone import training
        frame, input_receipt = base.prepare_admitted_data(data)
        write_exclusive(folder / "input_receipt.json", input_receipt)
        with prepared._engine_hooks(job["arm"]) as tracked:
            args = training_args(contract, data, folder, job["arm"])
            args.seeds = str(job["seed"])
            interface = training_interface(contract, data, frame, job["arm"])
            quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
                "strata": [{"stratum": f"quantity_{index}", "stratum_order": index,
                            "stratum_label": f"Cap Train bin {index}"} for index in range(5)]}
            summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
                interface_meta=interface, backbone=job["arm"], quantity_variant=VARIANT, seed=job["seed"])
            proof = tracked["guard"].proof()
        summary = {**summary, **prepared.input_identity(), "time_head_no_training_verification": proof}
        checkpoint = resolve_under(root, summary["checkpoint_path"])
        receipt = {**base_receipt(contract, host, job, stage), "selected_checkpoint": str(checkpoint),
            "selected_epoch": summary["best_epoch"], "checkpoint_sha256": sha_file(checkpoint),
            "state_sha256": summary["checkpoint_state_sha256"], "summary": summary,
            "time_head_no_training_verification": proof}
        prediction = folder / "training_validation_predictions.jsonl.gz"
        with prepared.runtime_hooks(job["arm"]):
            reference = evaluate_selected(data, job, receipt, root, frame, "validation", prediction)
        for key, expected in (("qty_rmse", summary["best_val_qty_rmse"]),
                              ("qty_mae", summary["best_val_qty_mae"]),
                              ("time_nll", summary["best_val_time_nll"])):
            require(math.isclose(reference["metrics"]["overall"][key], expected,
                    rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL), "Selected Validation differs from training summary")
        receipt.update(validation_reference=reference, target_identity=reference["target_identity"])
        files = [path for path in (folder / "runs").rglob("*") if path.is_file()]
        receipt["files"] = file_inventory(root, [*files, prediction, folder / "input_receipt.json"])
    else:
        train_path = folder / "train_receipt.json"
        train_receipt = read(train_path)
        verify_train_receipt(contract, host, job, root, train_receipt)
        frame, _ = base.prepare_admitted_data(data)
        prediction = folder / "validation_predictions.jsonl.gz"
        with prepared.runtime_hooks(job["arm"]):
            evaluated = evaluate_selected(data, job, train_receipt, root, frame, "validation", prediction)
        compare_metrics(evaluated["metrics"], train_receipt["validation_reference"]["metrics"])
        require(evaluated["target_identity"] == train_receipt["validation_reference"]["target_identity"],
                "Validation replay target identity changed")
        compare_prediction_files(prediction, train_receipt["validation_reference"]["prediction_path"])
        receipt = {**base_receipt(contract, host, job, stage), **evaluated,
            "selected_checkpoint": train_receipt["selected_checkpoint"],
            "selected_epoch": train_receipt["selected_epoch"],
            "checkpoint_sha256": train_receipt["checkpoint_sha256"],
            "time_head_no_training_verification": train_receipt["time_head_no_training_verification"],
            "validation_replay_passed": True,
            "files": file_inventory(root, [prediction, train_path,
                     resolve_under(root, train_receipt["selected_checkpoint"])])}
    write_exclusive(receipt_path, receipt)
    return receipt


def synthetic_check(contract=None, host=None, *, device=None, arm=None):
    """Fabricated admission on each candidate, including full training shape."""
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import core
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    verify_frozen_dependencies()
    require(arm is None or arm in ARMS, "Only the two approved synthetic candidates are admitted")
    selected_device = device or ("cuda:0" if contract is not None else "cpu")
    configure_runtime(selected_device, 4)
    if contract is not None:
        select_job(contract, host, contract["jobs"][0]["id"])
    lengths = [data["loader"]["max_seq_len"] for data in contract["datasets"]] if contract else [256]
    peaks, counts, states, proofs = {}, {}, {}, {}
    selected_arms = ARMS if arm is None else (arm,)
    for arm in selected_arms:
        if str(selected_device).startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        model, encoder = prepared.build_model(prepared.fabricated_data(), arm)
        states[arm] = canonical_state_dict_sha256(model.state_dict())
        model = model.to(selected_device).train()
        guard = TimeHeadGuard(model)
        optimizer = guard.bind_optimizer(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01))
        dt, quantity, mask = [value.to(selected_device) for value in prepared.fabricated_batch()]
        with prepared.runtime_hooks(arm):
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
                require(output["joint_loss"] is output["quantity_train_loss"], "Time loss re-entered training graph")
                output["joint_loss"].mean().backward()
                require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()), "Nonfinite fabricated gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            before = canonical_state_dict_sha256(model.state_dict())
            with torch.no_grad():
                original = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
                changed_quantity, changed_dt = quantity.clone(), dt.clone()
                changed_quantity[:, -1] += 100
                changed_dt[:, -1] += 100
                for altered in (core.target_outputs(model, dt, mask, changed_quantity, lambda_log_qty=1.0),
                                core.target_outputs(model, changed_dt, mask, quantity, lambda_log_qty=1.0)):
                    torch.testing.assert_close(original["pred_qty"], altered["pred_qty"], rtol=1e-5, atol=1e-5)
                single = core.target_outputs(model, dt[:1], mask[:1], quantity[:1], lambda_log_qty=1.0)
                for key in ("pred_qty", "time_loss"):
                    torch.testing.assert_close(original[key][:1], single[key], rtol=1e-5, atol=1e-5)
                binary = io.BytesIO()
                torch.save(model.state_dict(), binary)
                binary.seek(0)
                restored, restored_encoder = prepared.build_model(prepared.fabricated_data(), arm)
                require(encoder == restored_encoder, "Fabricated encoder metadata changed")
                restored.load_state_dict(torch.load(binary, map_location="cpu", weights_only=True), strict=True)
                restored = restored.to(selected_device).eval()
                replay = core.target_outputs(restored, dt, mask, quantity, lambda_log_qty=1.0)
                for key in ("pred_qty", "time_loss"):
                    torch.testing.assert_close(original[key], replay[key], rtol=1e-5, atol=1e-5)
            require(before == canonical_state_dict_sha256(model.state_dict()), "Synthetic evaluation changed state")
            del restored
            full_shape = [128, max(lengths)] if str(selected_device).startswith("cuda") else None
            if full_shape:
                model.train()
                full_dt = torch.ones(full_shape, device=selected_device)
                full_quantity = torch.randint(1, 11, full_shape, device=selected_device).float()
                full_mask = torch.ones(full_shape, device=selected_device, dtype=torch.bool)
                optimizer.zero_grad(set_to_none=True)
                loss = core.target_outputs(model, full_dt, full_mask, full_quantity, lambda_log_qty=1.0)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite full-shape loss")
                loss.backward()
                require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()), "Nonfinite full-shape gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                del full_dt, full_quantity, full_mask, loss
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        proofs[arm] = guard.proof()
        optimizer.step = guard.original_step
        peaks[arm] = {"max_memory_reserved_bytes": torch.cuda.max_memory_reserved() if full_shape else 0,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if full_shape else 0,
            "full_training_shape": full_shape, "includes_optimizer_step": bool(full_shape)}
        del model, optimizer, guard
        gc.collect()
    return {"schema": SCHEMA, "status": "passed", "device": str(selected_device), "host": host,
        "arms": list(selected_arms), "cuda_memory_peaks": peaks, "parameter_count": counts,
        "initial_state_sha256": states, "time_head_no_training_verification": proofs,
        "real_data_loaded": False, "held_out_test_evaluated": False,
        "checks": {"finite_forward_backward": True, "checkpoint_roundtrip": True,
            "target_quantity_blocked": True, "target_time_blocked_for_quantity": True,
            "batch_independence": True, "state_unchanged_during_evaluation": True,
            "time_loss_graph_excluded": True, "frozen_preparation_source_unchanged": True}}


def synthetic_condition(contract, host, loss_mode="log_mse", *, arm, device=None):
    require(loss_mode == "log_mse", "Only the approved log-MSE condition can be qualified")
    result = synthetic_check(contract, host, device=device, arm=arm)
    return {**result, **candidate_labels(arm), "arm": arm,
            "cuda_memory_peak": result["cuda_memory_peaks"][arm]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080",), required=True)
    parser.add_argument("--job")
    parser.add_argument("--stage", choices=("train", "validation", "synthetic"), required=True)
    args = parser.parse_args(argv)
    contract = read(args.contract)
    with redirect_stdout(sys.stderr):
        if args.stage == "synthetic":
            result = synthetic_check(contract, args.host)
        else:
            require(args.job is not None, "A scientific job is required")
            result = run_stage(contract, args.host, args.job, args.stage)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
