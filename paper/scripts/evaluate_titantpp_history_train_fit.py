#!/usr/bin/env python3
"""Train-only read-only diagnostics for the same immutable 27 selected checkpoints.

The sealed original evaluator supplies model/data/runtime/metric primitives.
This adapter owns its separate Train contract, read-only telemetry and receipts.
Validation/Test diagnostics are reused by SHA elsewhere, never inferred here. Original scientific files and selected checkpoint operations remain
unchanged. Importing this file performs no scientific imports or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

KIND = "titantpp_history_train_fit_20261012_v1"
PARENT = "70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d"
HISTORY_ARMS = ("titantpp_cnn_gru54", "titantpp_cnn_gru54_correction_disabled", "titantpp_cnn_gru54_current_only")
HISTORY_DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
SEEDS = (42, 52, 62)
PROJECT_PREFIXES = {"models", "paper", "data_loader", "simple_lab_test"}
TRUTH_FIELDS = ("target_id", "entity_id", "seq", "split", "recorded_date", "raw_quantity", "recorded_gap", "history_length", "last_quantity", "mean_quantity")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_module():
    here = Path(__file__).resolve().parent
    path = here / "original_evaluate.py"
    if not path.exists():
        path = here / "evaluate_titantpp_history_instacart_frozen_test.py"
    return load_module(path, "_history_original_evaluator")


original = original_module()
require, read, sha, json_bytes, canonical = original.require, original.read, original.sha, original.json_bytes, original.canonical
confined, write_exclusive, route_for, verify_source = original.confined, original.write_exclusive, original.route_for, original.verify_source
verify_imports = original.verify_imports
validate_payload, validate_frame = original.validate_payload, original.validate_frame
population_identity = original.population_identity
qualify, check_runtime = original.qualify, original.check_runtime


def load_telemetry():
    return load_module(Path(__file__).resolve().with_name("titantpp_history_diagnostic_telemetry.py"), "_read_only_diagnostic_telemetry")


def calendar_key(value):
    digits = "".join(x for x in str(value) if x.isdigit())
    require(len(digits) >= 6, "Recorded calendar metadata is malformed")
    return digits[:4] + "-" + digits[4:6]


def verify_contract(root):
    c = read(Path(root) / "contract.json")
    require(c.get("contract_sha256") == canonical({k: v for k, v in c.items() if k != "contract_sha256"}),
            "Train contract canonical SHA mismatch")
    require(c.get("schema") == KIND and c.get("parent_contract_sha256") == PARENT,
            "Foreign Train diagnostic contract")
    require(c.get("expected_seeds") == list(SEEDS), "Train diagnostic seed set differs")
    expected = {(d, a, s) for d in HISTORY_DATASETS for a in HISTORY_ARMS for s in SEEDS}
    require(len(c["jobs"]) == 27 and len({j["id"] for j in c["jobs"]}) == 27
            and {(j["dataset"], j["arm"], j["seed"]) for j in c["jobs"]} == expected
            and all(j["study"] == "history" for j in c["jobs"]),
            "Train contract must bind exactly the selected history27")
    require(set(c["sources"]) == {"history"} and set(c["datasets"]) == set(HISTORY_DATASETS),
            "Foreign scientific source/dataset")
    fit = c["train_fit"]
    require(fit["evaluation_splits"] == ["train"] and fit["new_fits"] == 0
            and fit["validation_test_inference"] == 0, "Train-only inference authorization differs")
    require(set(fit["expected_train_populations"]) == set(HISTORY_DATASETS),
            "Train population contract is incomplete")
    for job in c["jobs"]:
        require(job["id"] and all(x not in job["id"] for x in ("/", "\\", "..")),
                "Unsafe Train job ID")
        require(job["selected_epoch"] >= 1 and job["validation_reference"]["split"] == "validation",
                "Original selector binding differs")
        spec = c["datasets"][job["dataset"]]
        definition = spec["definition"]
        require(job["frozen_source_closure"] == c["sources"]["history"]["closure"], "Source closure differs")
        require(float(spec["tail_threshold"]) == float(definition["quantity_boundaries_all_train_rows"][-1])
                == original.TAILS[job["dataset"]], "Train tail threshold differs")
        loader = definition["loader"]
        require(loader["batch_size"] == 128 and loader["num_workers"] == 0 and loader["drop_last"] is False,
                "Frozen Train loader differs")
        population = definition["inherited_data_identity"]["populations"]["train"]
        require(fit["expected_train_populations"][job["dataset"]] == population
                and population["target_count"] > 0, "Train population binding differs")
        require(spec.get("training_data_path") and spec.get("training_data_sha256"),
                "Original training input export is required")
    return c


def verify_operations(root, c):
    seal = read(Path(root) / "operation_integrity.json")
    require(seal.get("contract_sha256") == c["contract_sha256"], "Foreign Train diagnostic operation seal")
    files = seal["files"]
    for required in ("control/evaluate.py", "control/original_evaluate.py", "control/titantpp_history_diagnostic_telemetry.py"):
        require(required in files, "Operation seal misses diagnostic module: " + required)
    for relative, expected in files.items():
        require(sha(confined(root, relative)) == expected, "Train diagnostic operation bytes changed: " + relative)
    return seal


def verify_train_receipt(receipt, c, job):
    """Bind a completed Train evaluation to its frozen scientific job."""
    require(receipt.get("status") == "complete" and receipt.get("schema") == KIND
            and receipt.get("contract_sha256") == c["contract_sha256"]
            and receipt.get("split") == "train"
            and receipt.get("evaluation_scope") == "train_selected_checkpoint_fit_diagnostics",
            "Foreign or incomplete Train receipt")
    for key in ("id", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256"):
        require(receipt.get(key) == job[key], "Train selected-model identity differs: " + key)
    spec, source = c["datasets"][job["dataset"]], c["sources"]["history"]
    population = c["train_fit"]["expected_train_populations"][job["dataset"]]
    require(receipt.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
            and receipt.get("state_sha256") == receipt.get("state_after_sha256") == job["checkpoint_state_sha256"]
            and receipt.get("source_closure") == receipt.get("frozen_source_closure") == source["closure"]
            and receipt.get("source_revision") == source["revision"]
            and receipt.get("source_revision_history") == job["source_revision_history"],
            "Train checkpoint or scientific lineage differs")
    require(receipt.get("original_population_identity") == population
            and receipt.get("expected_target_count") == receipt.get("prediction_rows") == population["target_count"]
            and receipt.get("full_population") is True
            and receipt.get("data_sha256") == spec["training_data_sha256"]
            and receipt.get("data_identity") == {"raw_sha256": spec["training_data_sha256"],
                                                 "path": spec["training_data_path"]},
            "Train population/input binding differs")
    require(receipt.get("train_reference") == "no_prior_selected_checkpoint_train_evaluation"
            and receipt.get("prior_train_metric_reproduction_claimed") is False
            and receipt.get("train_state_eval_not_training_loss") is True,
            "Train evaluation cannot claim prior training-loss reproduction")
    for key in ("new_fits", "interventions", "optimizer_updates", "validation_test_inference"):
        require(type(receipt.get(key)) is int and receipt[key] == 0, "Unapproved operation: " + key)
    for key in ("retrained", "reselected", "held_out_test_evaluated", "raw_predictions_written", "gradients_enabled"):
        require(receipt.get(key) is False, "Unapproved Train flag: " + key)
    for key in ("inference_mode_enabled", "observational_telemetry_only", "no_forward_replacement"):
        require(receipt.get(key) is True, "Train inference integrity flag differs: " + key)
    for key in ("train_population_match", "diagnostics_match", "aggregate_own_six_metrics_match"):
        require(receipt.get(key) == "PASS", "Train integrity check did not pass: " + key)
    require(set(receipt.get("qualification", {})) == original.QUALIFICATION_KEYS
            and all(value is True for value in receipt["qualification"].values()),
            "Train causality/noninterference qualification is incomplete")
    require(receipt.get("batch_size") == 128 and receipt.get("shuffle") is False
            and receipt.get("num_workers") == 0 and receipt.get("drop_last") is False
            and receipt.get("loader") == spec["definition"]["loader"], "Train inference loader differs")
    metrics = receipt["metrics"]
    require(metrics.get("split") == "train" and metrics["count"] == population["target_count"]
            and metrics.get("quantity_boundaries") == spec["definition"]["quantity_boundaries_all_train_rows"],
            "Train metric population or frozen bins differ")
    for cell in (metrics, metrics["tail"]):
        for key in ("qty_rmse", "qty_mae", "time_nll", "qty_bias"):
            require((cell["count"] == 0 and cell[key] is None)
                    or (cell["count"] > 0 and type(cell[key]) in (int, float) and math.isfinite(cell[key])),
                    "Train metrics are not finite: " + key)
    for key in ("target_identity_sha256", "truth_sha256", "telemetry_sha256"):
        digest = receipt.get(key)
        require(isinstance(digest, str) and len(digest) == 64
                and all(x in "0123456789abcdef" for x in digest), "Train digest is malformed: " + key)
    return receipt


def verify_aggregate_metrics(telemetry, native_metrics):
    all_cells = telemetry["strata"]["all"]
    require(len(all_cells) == 1 and all_cells[0]["key"] == "all", "All-target telemetry cell differs")
    bins = telemetry["strata"]["quantity_bin"]
    tail_key = str(len(telemetry["quantity_boundaries"]))
    tail = next((x for x in bins if x["key"] == tail_key), None)
    tail_reference = native_metrics["tail"]
    require(tail is not None or tail_reference["count"] == 0, "Native nonempty tail telemetry cell is missing")
    pairs = [(all_cells[0], native_metrics)]
    if tail is not None:
        pairs.append((tail, tail_reference))
    else:
        require(all(tail_reference[key] is None for key in ("qty_mae", "qty_rmse", "time_nll")),
                "An empty Train tail cannot carry metrics")
    for cell, reference in pairs:
        require(cell["count"] == reference["count"], "Own aggregate count differs from native metrics")
        for metric in ("qty_mae", "qty_rmse", "time_nll"):
            require(math.isclose(cell[metric], reference[metric], rel_tol=1e-5, abs_tol=1e-5), "Own telemetry aggregate differs: " + metric)
    return True


def prepare_train_population(frame, *, core, loader_module, definition, np):
    """Use the original Train/Validation frame with Train targets only.

    Keeping the frozen frame also retains its complete entity inventory, which
    is part of the original hard_lmm Train population digest. The unchanged
    chronological loader excludes every Validation/Test target and later row
    from each Train target's causal context. Membership metadata remains a set.
    """
    prepared = core.prepare_count_frame(frame)
    loader = loader_module.make_loader(prepared, target_split="train", batch_size=128,
        shuffle=False, generator=None, lookback_weeks=definition["loader"]["lookback_weeks"],
        max_seq_len=definition["loader"]["max_seq_len"])
    dataset = loader.dataset
    validation_entities = {str(dataset.parts[p]) for p, splits in enumerate(dataset.split_lists)
                           if any(value == "validation" for value in splits[1:])}
    require(isinstance(validation_entities, set), "Validation entity metadata must remain a set")
    identity = population_identity(dataset, "train", np)
    metadata = prepared.select(["oper_part_no", "seq", "chronological_split", "demand_dt"])
    offsets, total = [], 0
    for sequences in dataset.seq_lists:
        offsets.append(total)
        total += len(sequences)
    require(total == prepared.height, "Loader frame mapping differs")
    return loader, dataset, identity, metadata, offsets, validation_entities


def worker(root, job_id, split, device):
    root = Path(root).resolve()
    c = verify_contract(root)
    telemetry_module = load_telemetry()
    matching = [r for r in c["jobs"] if r["id"] == job_id]
    require(len(matching) == 1, "Unknown or duplicate scientific job")
    job = matching[0]
    require(split == "train" and device in ("cpu", "cuda"), "Only Train inference is approved")
    started_unix = time.time()
    output = root / "results" / job_id
    name = "train_metrics"
    require(not (output / (name + ".json")).exists() and not (output / (name + "_failure.json")).exists(),
            "Existing receipt/failure prohibits automatic repeat")
    started = time.monotonic()
    write_exclusive(output / (name + "_claim.json"), {"job": job_id, "split": split,
        "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv, "contract_sha256": c["contract_sha256"]})
    phase = "integrity"
    try:
        operation_seal_sha = sha(root / "operation_integrity.json")
        verify_operations(root, c)
        source = c["sources"][job["study"]]
        source_root = verify_source(root, source)
        checkpoint = confined(root, job["checkpoint"]["relative_path"])
        require(sha(checkpoint) == job["checkpoint"]["sha256"], "Checkpoint byte SHA mismatch")
        metadata_path = confined(root, job["checkpoint_metadata_path"])
        require(sha(metadata_path) == job["checkpoint_metadata_sha256"], "Checkpoint metadata byte SHA mismatch")
        expected_metadata = read(metadata_path)
        spec = c["datasets"][job["dataset"]]
        definition = spec["definition"]
        data_key, sha_key = "training_data_path", "training_data_sha256"
        data_path = confined(root, spec[data_key])
        require(sha(data_path) == spec[sha_key], "Frozen input byte SHA mismatch")
        script_sha = sha(__file__)
        sys.dont_write_bytecode = True
        # No checkout fallbacks or previously loaded project packages may be used.
        require(not any(n.split(".")[0] in PROJECT_PREFIXES for n in sys.modules), "Fresh worker process is required")
        sys.path[:] = [str(source_root)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)
                                                and not (Path(p).resolve() / "paper").is_dir()]
        os.environ["MPLCONFIGDIR"] = str(output / "runtime_cache" / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(output / "runtime_cache")
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import numpy as np
        import polars as pl
        import torch
        numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
        execution_device = "cuda:0" if device == "cuda" else "cpu"
        numerical.configure_runtime(execution_device, c["runtime"]["torch_num_threads"])
        runtime = numerical.runtime_identity(execution_device)
        check_runtime(runtime, c["runtime"], device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        engine = importlib.import_module(route_for(job))
        engine.install_hooks()
        factory = importlib.import_module("models.TPPs.CountAwareFactory")
        core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
        loader_module = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
        runner = importlib.import_module("simple_lab_test.search.common.runner")
        metrics_module = importlib.import_module("paper.scripts.evaluate_titantpp_history_width")
        verify_imports(source_root, source)
        phase = "loading_population"
        lazy = pl.scan_parquet(data_path)
        lazy = lazy.filter(pl.col("chronological_split").is_in(["train", "validation"]))
        frame = lazy.collect().sort(["oper_part_no", "seq"])
        validate_frame(frame, definition, "validation", np, pl)
        require(set(frame["chronological_split"].unique().to_list()) <= {"train", "validation"},
                "Train evaluator materialized Test rows")
        # Audit train-only initialization without refitting or replacing stored scalars.
        train_q = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
        stats = definition["statistics"]
        digest = hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0" + train_q.astype("<f8").tobytes()).hexdigest()
        require(digest == stats["all_train_quantity_sha256"], "Frozen all-train quantity identity changed")
        for key, measured in (("train_log_mean", float(np.log1p(train_q).mean())),
                              ("train_log_std", float(np.log1p(train_q).std()))):
            require(math.isclose(measured, stats[key], rel_tol=0, abs_tol=1e-12), "Train-only initialization drift: " + key)
        loader, dataset, identity, metadata, offsets, validation_entities = prepare_train_population(
            frame, core=core, loader_module=loader_module, definition=definition, np=np)
        expected_population = c["train_fit"]["expected_train_populations"][job["dataset"]]
        expected_count = expected_population["target_count"]
        require(len(dataset) == expected_count > 0 and identity == expected_population,
                "Original eligible Train population identity differs")
        del frame
        payload = runner.torch_load_checkpoint(checkpoint, map_location="cpu")
        validate_payload(payload, job, source, definition, expected_metadata)
        factory.validate_checkpoint_route(payload, job["arm"])
        require(runner.canonical_state_dict_sha256(payload["model_state_dict"]) == job["checkpoint_state_sha256"],
                "Checkpoint actual tensor SHA mismatch")
        config = {k: v for k, v in definition["model"].items()
                  if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
        model, _ = factory.build_count_aware_model(job["arm"], **config, train_log_mean=stats["train_log_mean"],
            train_log_std=stats["train_log_std"], max_seq_len=definition["loader"]["max_seq_len"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(execution_device).eval().requires_grad_(False)
        before = runner.canonical_state_dict_sha256(model.state_dict())
        require(before == job["checkpoint_state_sha256"], "Loaded model tensor SHA mismatch")
        del payload
        bounds = definition["quantity_boundaries_all_train_rows"]
        dspec = spec["diagnostics"]
        collector = telemetry_module.AggregateDiagnostics(bounds, dspec["input_history_boundaries"],
            dspec["input_last_gap_boundaries"], dspec["target_gap_boundaries"], dspec["train_history_max"])
        adapter = importlib.import_module("models.TPPs.CountAwareTitanCNNGRU")
        observer = telemetry_module.ReadOnlyTelemetry(model, job["arm"], torch, adapter.causal_depthwise_residual)
        metrics = metrics_module.QuantityMetrics(bounds)
        simple = {key: metrics_module.QuantityMetrics(bounds) for key in ("last_quantity", "mean_quantity")}
        target_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
        qualification, processed, inference_seconds = {}, 0, 0.
        phase = "inference"
        with torch.inference_mode():
            for batch_index, (_, dts, mask, part_indices, quantities) in enumerate(loader):
                dts, mask, quantities = (v.to(execution_device) for v in (dts, mask, quantities))
                if device == "cuda":
                    torch.cuda.synchronize()
                batch_start = time.monotonic()
                observer.begin()
                value = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
                observed_features = observer.snapshot(value["history_length"], dts[:, -1])
                if device == "cuda":
                    torch.cuda.synchronize()
                inference_seconds += time.monotonic() - batch_start
                if batch_index == 0:
                    with observer.suspended():
                        qualification = qualify(torch, model, core.target_outputs, dts, mask, quantities, value,
                            definition["model"]["time_observation_contract"]["top_code"])
                for key in ("true_qty", "pred_qty", "time_loss"):
                    require(bool(torch.isfinite(value[key]).all()), "Nonfinite model output: " + key)
                history_mask = mask.cpu().clone()
                history_mask[:, -1] = False
                history_length = history_mask.sum(dim=1)
                require(bool((history_length >= 1).all()), "Target has no causal history")
                quantity_cpu = quantities.cpu().double()
                last = quantity_cpu[:, -2].numpy()
                mean = ((quantity_cpu * history_mask).sum(dim=1) / history_length).numpy()
                q, prediction, temporal = [value[key].cpu().numpy().astype(np.float64) for key in ("true_qty", "pred_qty", "time_loss")]
                metrics.add(q, prediction, temporal)
                for key, forecast in (("last_quantity", last), ("mean_quantity", mean)):
                    simple[key].add(q, forecast, np.zeros(q.shape, dtype=np.float64))
                gaps = dts[:, -1].cpu().tolist()
                lengths = value["history_length"].cpu().tolist()
                for offset in range(len(q)):
                    index = processed + offset
                    part, end = dataset.index[index]
                    require(int(part_indices[offset]) == part and int(history_length[offset]) == lengths[offset], "Loader order/history differs")
                    raw = metadata.row(offsets[part] + end + 1, named=True)
                    entity, sequence = str(dataset.parts[part]), int(dataset.seq_lists[part][end + 1])
                    require(raw["chronological_split"] == split and str(raw["oper_part_no"]) == entity
                            and int(raw["seq"]) == sequence and raw["demand_dt"] is not None, "Target metadata mapping differs")
                    date = str(raw["demand_dt"])
                    target_id = hashlib.sha256(json_bytes([job["dataset"], spec["data_sha256"], entity, sequence, split, date])).hexdigest()
                    row = {"target_id": target_id, "entity_id": entity, "seq": sequence, "split": split,
                        "recorded_date": date, "raw_quantity": float(q[offset]), "recorded_gap": float(gaps[offset]),
                        "history_length": int(lengths[offset]), "last_quantity": float(last[offset]), "mean_quantity": float(mean[offset])}
                    target_digest.update((target_id + "\n").encode())
                    truth_digest.update(json_bytes([row[key] for key in TRUTH_FIELDS]) + b"\n")
                    previous = metadata.row(offsets[part] + end, named=True)
                    sequences = np.asarray(dataset.seq_lists[part], dtype=np.int64)
                    start = int(np.searchsorted(sequences, sequences[end] - (dataset.W - 1), side="left"))
                    unclipped = int(end + 1 - start)
                    require(unclipped >= int(lengths[offset]), "History truncation derivation differs")
                    collector.add(entity=entity, calendar=calendar_key(date),
                        last_input_calendar=calendar_key(str(previous["demand_dt"])),
                        history_length=int(lengths[offset]), unclipped_history_length=unclipped,
                        truncated=unclipped > dataset.max_len - 1, series_in_validation=entity in validation_entities,
                        true_quantity=float(q[offset]), predicted_quantity=float(prediction[offset]),
                        time_nll=float(temporal[offset]), recorded_gap=float(gaps[offset]),
                        last_input_gap=float(dts[offset, -2].cpu()), last_quantity=float(last[offset]),
                        mean_quantity=float(mean[offset]), features={key: values[offset] for key, values in observed_features.items()})
                processed += len(q)
        require(processed == expected_count, "Incomplete Train population")
        final_metrics = {"split": split, **metrics.finish()}
        telemetry = collector.finish()
        require(telemetry["count"] == expected_count, "Incomplete telemetry population")
        verify_aggregate_metrics(telemetry, final_metrics)
        telemetry.update(job_id=job_id, split=split, contract_sha256=c["contract_sha256"],
                         interpretation=observer.interpretation())
        observer.close()
        simple_results = {key.removesuffix("_quantity"): accumulator.finish() for key, accumulator in simple.items()}
        # Quantity-only baselines have no timing model: never report dummy zero TimeNLL.
        for result in simple_results.values():
            result["split"] = split
            for cell in [result, result["tail"], *result["quantity_cells"]]:
                cell.pop("time_nll", None)
                cell.pop("time_nll_sum", None)
        phase = "final_integrity"
        require(runner.canonical_state_dict_sha256(model.state_dict()) == before, "Model tensors changed during inference")
        require(sha(checkpoint) == job["checkpoint"]["sha256"] and sha(data_path) == spec[sha_key], "Checkpoint/input bytes changed during inference")
        require(sha(metadata_path) == job["checkpoint_metadata_sha256"], "Checkpoint metadata changed during inference")
        if spec.get("training_data_path"):
            require(sha(confined(root, spec["training_data_path"])) == spec["training_data_sha256"], "Frozen Train/Validation export changed")
        verify_source(root, source)
        require(verify_contract(root) == c and sha(__file__) == script_sha, "Contract/evaluator changed during inference")
        verify_operations(root, c)
        require(sha(root / "operation_integrity.json") == operation_seal_sha, "Operation seal changed during inference")
        imports = verify_imports(source_root, source)
        elapsed = time.monotonic() - started
        receipt = {"status": "complete", "schema": KIND, "contract_sha256": c["contract_sha256"],
            **{key: job[key] for key in ("id", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256")},
            "job_id": job["id"], "frozen_source_closure": source["closure"],
            "split": split, "evaluation_scope": "train_selected_checkpoint_fit_diagnostics",
            "checkpoint_sha256": job["checkpoint"]["sha256"], "state_sha256": before, "state_after_sha256": runner.canonical_state_dict_sha256(model.state_dict()),
            "source_closure": source["closure"], "source_revision": source["revision"],
            "source_revision_history": job["source_revision_history"], "full_population": True,
            "expected_target_count": expected_count, "prediction_rows": processed, "metrics": final_metrics,
            "simple_quantity_baselines": simple_results, "original_population_identity": identity,
            "target_identity_sha256": target_digest.hexdigest(), "truth_sha256": truth_digest.hexdigest(),
            "data_identity": {"raw_sha256": spec[sha_key], "path": spec[data_key]},
            "truth_digest_fields": TRUTH_FIELDS, "data_sha256": spec[sha_key], "loader": definition["loader"],
            "batch_size": 128, "shuffle": False, "num_workers": 0, "drop_last": False,
            "qualification": {**qualification, "parameters_unchanged": True, "finite_outputs": True},
            "train_reference": "no_prior_selected_checkpoint_train_evaluation",
            "prior_train_metric_reproduction_claimed": False,
            "train_population_match": "PASS", "train_state_eval_not_training_loss": True,
            "retrained": False, "reselected": False, "test_independent": False,
            "held_out_test_evaluated": False, "raw_predictions_written": False,
            "imported_frozen_modules": imports, "evaluator_sha256": script_sha, "runtime": runtime,
            "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv,
            "observed_unix": time.time(), "started_unix": started_unix, "completed_unix": time.time(),
            "validation_test_inference": 0, "elapsed_seconds": elapsed, "inference_seconds": inference_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
            "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
        telemetry_path = output / (split + "_telemetry.json")
        write_exclusive(telemetry_path, telemetry)
        receipt.update(diagnostics_match="PASS", new_fits=0, interventions=0, optimizer_updates=0,
            gradients_enabled=False, inference_mode_enabled=True,
            telemetry_path=telemetry_path.relative_to(root).as_posix(), telemetry_sha256=sha(telemetry_path),
            telemetry_module_sha256=sha(telemetry_module.__file__), operation_integrity_sha256=operation_seal_sha,
            aggregate_own_six_metrics_match="PASS", observational_telemetry_only=True, no_forward_replacement=True)
        verify_train_receipt(receipt, c, job)
        write_exclusive(output / (name + ".json"), receipt)
        print(json.dumps({key: receipt[key] for key in ("status", "id", "split", "prediction_rows", "elapsed_seconds")}), flush=True)
        return 0
    except BaseException as error:
        write_exclusive(output / (name + "_failure.json"), {"status": "failed", "phase": phase,
            "id": job_id, "split": split, "error": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "contract_sha256": c["contract_sha256"],
            "elapsed_seconds": time.monotonic() - started, "receipt_written": False})
        raise


def native_probe(root, study, arm, device):
    """Synthetic hook noninterference admission; no saved model or data reads."""
    root = Path(root).resolve()
    c = verify_contract(root)
    require(study == "history" and arm in HISTORY_ARMS, "Foreign native diagnostic probe")
    job = next(j for j in c["jobs"] if j["arm"] == arm)
    source = c["sources"]["history"]
    operation_seal_sha = sha(root / "operation_integrity.json")
    verify_operations(root, c)
    source_root = verify_source(root, source)
    require(not any(n.split(".")[0] in PROJECT_PREFIXES for n in sys.modules), "Fresh diagnostic native probe process required")
    telemetry_module = load_telemetry()
    sys.dont_write_bytecode = True
    sys.path[:] = [str(source_root)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)
                                      and not (Path(p).resolve() / "paper").is_dir()]
    cache = root / "control" / "native_probe_runtime_cache" / arm
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
    execution_device = "cuda:0" if device == "cuda" else "cpu"
    numerical.configure_runtime(execution_device, c["runtime"]["torch_num_threads"])
    runtime = numerical.runtime_identity(execution_device)
    if device == "cuda":
        check_runtime(runtime, c["runtime"], device)
    engine = importlib.import_module(route_for(job))
    engine.install_hooks()
    factory = importlib.import_module("models.TPPs.CountAwareFactory")
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    runner = importlib.import_module("simple_lab_test.search.common.runner")
    adapter = importlib.import_module("models.TPPs.CountAwareTitanCNNGRU")
    definition = c["datasets"][job["dataset"]]["definition"]
    config = {k: v for k, v in definition["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    with torch.random.fork_rng(devices=[] if device == "cpu" else [0]):
        torch.manual_seed(42)
        model, _ = factory.build_count_aware_model(arm, **config, train_log_mean=1.2, train_log_std=.8,
            max_seq_len=definition["loader"]["max_seq_len"])
        model.to(execution_device).eval().requires_grad_(False)
        with torch.no_grad():
            # Fixture only; never read or alter a saved selected checkpoint.
            for name, parameter in model.named_parameters():
                if "multilag_detail" in name or "causal_q_kernel" in name or "causal_k_kernel" in name or "causal_v_kernel" in name:
                    parameter.copy_(torch.randn_like(parameter) * .03)
            before = runner.canonical_state_dict_sha256(model.state_dict())
            dts = torch.tensor([[0., 0., 1., 2., 1., 3., 1., 2.], [1., 2., 1., 3., 1., 2., 1., 3.]], device=execution_device)
            quantities = torch.tensor([[0., 0., 2., 4., 1., 3., 2., 5.], [1., 2., 3., 4., 3., 2., 1., 6.]], device=execution_device)
            mask = torch.ones_like(dts, dtype=torch.bool)
            mask[0, :2] = False
            baseline = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
            observer = telemetry_module.ReadOnlyTelemetry(model, arm, torch, adapter.causal_depthwise_residual)
            observer.begin()
            observed = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
            features = observer.snapshot(observed["history_length"], dts[:, -1])
            require(all(torch.equal(baseline[k], observed[k]) for k in baseline), "Read-only hooks changed original synthetic output")
            with observer.suspended():
                checks = qualify(torch, model, core.target_outputs, dts, mask, quantities, observed,
                    definition["model"]["time_observation_contract"]["top_code"])
            observer.close()
            require(all(bool(torch.isfinite(v).all()) for v in observed.values()), "Nonfinite synthetic diagnostic output")
            require(runner.canonical_state_dict_sha256(model.state_dict()) == before, "Synthetic telemetry changed model tensors")
    verify_source(root, source)
    verify_operations(root, c)
    require(sha(root / "operation_integrity.json") == operation_seal_sha, "Native probe operation seal changed")
    imports = verify_imports(source_root, source)
    result = {"status": "PASS", "study": study, "arm": arm, "device": device,
        "runtime": runtime, "native_pinned_contract_checked": device == "cuda", "source_closure": source["closure"],
        "qualification": {**checks, "parameters_unchanged": True, "finite_outputs": True},
        "hooked_and_unhooked_all_original_outputs_bitwise_equal": True,
        "telemetry_finite": True, "telemetry_feature_names": sorted(features),
        "optimizer_updates": 0, "checkpoint_reads": 0, "real_data_loaded": False, "held_out_test_evaluated": False,
        "imported_frozen_modules": imports, "contract_sha256": c["contract_sha256"]}
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--split", choices=("train",))
    parser.add_argument("--native-probe", action="store_true")
    parser.add_argument("--study", choices=("history",))
    parser.add_argument("--arm", choices=HISTORY_ARMS)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    if args.native_probe:
        require(args.study is not None and args.arm is not None and args.job_id is None and args.split is None,
                "Native Train probe takes --study/--arm only")
        return native_probe(args.root, args.study, args.arm, args.device)
    require(args.job_id is not None and args.split is not None and args.study is None and args.arm is None,
            "Train worker takes --job-id/--split only")
    return worker(args.root, args.job_id, args.split, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
