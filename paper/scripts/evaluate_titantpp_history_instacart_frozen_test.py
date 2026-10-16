#!/usr/bin/env python3
"""Aggregate-only final Test comparison of 36 immutable selected checkpoints.

This operational adapter imports one sealed scientific source tree per process.
It does not train, change the model/data implementation, select an epoch, or
write predictions. All 36 full Validation replays must pass before Test data
can be materialized. Importing this module only imports the Python standard
library; the CLI performs the explicitly approved inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

KIND = "titantpp_history_instacart_frozen_test_20261012_v1"
HISTORY_ARMS = ("titantpp_cnn_gru54", "titantpp_cnn_gru54_correction_disabled",
                "titantpp_cnn_gru54_current_only")
INSTACART_ARMS = ("titantpp_history_mlp_width8", "titantpp_history_mlp_width12",
                 "titantpp_cnn_gru54")
HISTORY_DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
TAILS = dict(zip((*HISTORY_DATASETS, "insta_market_basket"), (3449., 187., 200., 35.)))
SEEDS = (42, 52, 62)
METRICS = ("qty_mae", "qty_rmse", "time_nll")
ROUTES = {
    "history": "paper.scripts.run_titantpp_cnn_history_controls",
    "instacart_mlp": "paper.scripts.run_titantpp_history_width",
    "instacart_cnn": "paper.scripts.run_titantpp_cnn_gru",
}
PROJECT_PREFIXES = {"models", "paper", "data_loader", "simple_lab_test"}
TRUTH_FIELDS = ("target_id", "entity_id", "seq", "split", "recorded_date",
                "raw_quantity", "recorded_gap", "history_length", "last_quantity", "mean_quantity")
QUALIFICATION_KEYS = {"target_quantity_perturbation_preserves_prediction", "target_quantity_perturbation_preserves_time_loss",
    "target_gap_perturbation_preserves_quantity_prediction", "single_vs_batch_prediction_matches",
    "single_vs_batch_time_loss_matches", "parameters_unchanged", "finite_outputs"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def canonical(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    require(not Path(relative).is_absolute(), "Absolute artifact path is not permitted")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Artifact path escapes the frozen root")
    return path


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def expected_jobs():
    result = {( "history", d, a, s) for d in HISTORY_DATASETS for a in HISTORY_ARMS for s in SEEDS}
    result.update(("instacart", "insta_market_basket", a, s) for a in INSTACART_ARMS for s in SEEDS)
    return result


def verify_contract(root):
    c = read(Path(root) / "contract.json")
    seal = c.get("contract_sha256")
    require(seal == canonical({k: v for k, v in c.items() if k != "contract_sha256"}),
            "Contract canonical SHA mismatch")
    require(c.get("schema") == KIND and c.get("evaluation_scope") == "test_final_comparison"
            and c.get("metrics_schema") == "quantity_time_tail_v1", "Foreign evaluation contract")
    require(c.get("expected_seeds") == list(SEEDS), "Frozen seed contract differs")
    jobs = c["jobs"]
    actual = {(r["study"], r["dataset"], r["arm"], r["seed"]) for r in jobs}
    require(len(jobs) == 36 and len({r["id"] for r in jobs}) == 36
            and actual == expected_jobs(), "Contract must bind exactly the approved 36 conditions")
    for job in jobs:
        require(job["id"] and all(x not in job["id"] for x in ("/", "\\", "..")), "Unsafe job identifier")
        require(job["validation_reference"]["split"] == "validation", "Checkpoint selection reference is not Validation")
        require(job["selected_epoch"] >= 1, "Invalid selected epoch")
        spec = c["datasets"][job["dataset"]]
        definition = spec["definition"]
        require(definition["dataset_id"] == job["dataset"], "Dataset definition identity differs")
        require(float(spec["tail_threshold"]) == TAILS[job["dataset"]]
                == float(definition["quantity_boundaries_all_train_rows"][-1]), "Frozen tail boundary changed")
        loader = definition["loader"]
        require(loader["batch_size"] == 128 and loader["num_workers"] == 0
                and loader["drop_last"] is False, "Frozen loader contract differs")
        require(job["frozen_source_closure"] == c["sources"][job["study"]]["closure"],
                "Checkpoint scientific source closure differs")
    return c


def route_for(job):
    if job["study"] == "history":
        require(job["dataset"] in HISTORY_DATASETS and job["arm"] in HISTORY_ARMS, "Foreign history route")
        return ROUTES["history"]
    require(job["study"] == "instacart" and job["dataset"] == "insta_market_basket"
            and job["arm"] in INSTACART_ARMS, "Foreign Instacart route")
    return ROUTES["instacart_cnn" if job["arm"] == "titantpp_cnn_gru54" else "instacart_mlp"]


def verify_source(root, source):
    require(canonical(source["files"]) == source["closure"], "Frozen source manifest closure changed")
    base = confined(root, source["root"])
    for relative, expected in source["files"].items():
        path = confined(base, relative)
        require(sha(path) == expected, "Frozen scientific source changed: " + relative)
    return base


def verify_imports(source_root, source):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in PROJECT_PREFIXES:
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        require(path.is_relative_to(source_root), "Nonfrozen project module imported: " + name)
        relative = path.relative_to(source_root).as_posix()
        require(relative in source["files"], "Unsealed project module imported: " + name)
        result[name] = relative
    return result


def compare_validation_metrics(actual, reference):
    for group in (None, "tail"):
        left = actual if group is None else actual[group]
        right = reference if group is None else reference[group]
        require(left["count"] == right["count"], "Validation population count mismatch")
        for key in METRICS:
            require(isinstance(left[key], (int, float)) and math.isfinite(left[key])
                    and math.isfinite(right[key])
                    and math.isclose(left[key], right[key], rel_tol=1e-5, abs_tol=1e-5),
                    "Validation replay mismatch: " + str(group) + "/" + key)


def verify_validation_receipt(receipt, c, job):
    require(receipt.get("status") == "complete" and receipt.get("split") == "validation"
            and receipt.get("contract_sha256") == c["contract_sha256"], "Foreign or incomplete Validation receipt")
    for key in ("id", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256"):
        require(receipt.get(key) == job[key], "Validation receipt identity mismatch: " + key)
    require(receipt.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
            and receipt.get("state_sha256") == job["checkpoint_state_sha256"]
            and receipt.get("source_closure") == job["frozen_source_closure"], "Validation frozen checkpoint binding differs")
    require(receipt.get("retrained") is False and receipt.get("reselected") is False
            and receipt.get("raw_predictions_written") is False
            and receipt.get("full_population") is True
            and receipt.get("validation_match") == "PASS", "Validation qualification receipt is not a full frozen replay")
    require(set(receipt.get("qualification", {})) == QUALIFICATION_KEYS
            and all(value is True for value in receipt["qualification"].values()), "Validation causal qualification failed")
    compare_validation_metrics(receipt["metrics"], job["validation_reference"])
    for name in ("target_identity_sha256", "truth_sha256"):
        require(isinstance(receipt.get(name), str) and len(receipt[name]) == 64, "Missing raw target digest")
    return receipt


def verify_global_validation_gate(root, c):
    gate = read(Path(root) / "validation_gate.json")
    require(gate.get("status") == "PASS" and gate.get("contract_sha256") == c["contract_sha256"]
            and gate.get("job_count") == 36, "All 36 full Validation replays must pass before Test")
    bindings = gate["validation_receipts"]
    require(set(bindings) == {r["id"] for r in c["jobs"]}, "Global Validation gate has missing or foreign jobs")
    created = gate.get("created_unix")
    require(isinstance(created, (int, float)) and math.isfinite(created), "Global Validation gate timestamp is missing")
    identities = {}
    for job in c["jobs"]:
        path = confined(root, "results/" + job["id"] + "/validation_replay.json")
        require(sha(path) == bindings[job["id"]], "Global Validation receipt bytes changed")
        r = verify_validation_receipt(read(path), c, job)
        completed = r.get("completed_unix")
        require(isinstance(completed, (int, float)) and math.isfinite(completed) and completed <= created,
                "Validation receipt completed after global gate")
        exit_path = confined(root, "results/" + job["id"] + "/validation_worker_exit.json")
        actual_exit = read(exit_path)
        exit_completed = actual_exit.get("completed_unix")
        require(actual_exit.get("job_id") == job["id"] and actual_exit.get("contract_sha256") == c["contract_sha256"]
                and type(actual_exit.get("exit_code")) is int and actual_exit["exit_code"] == 0
                and actual_exit.get("actual_child_exit_observed") is True
                and isinstance(exit_completed, (int, float)) and math.isfinite(exit_completed) and exit_completed <= created,
                "Validation actual worker exit0 is not verified before global gate")
        if "validation_worker_exits" in gate:
            require(sha(exit_path) == gate["validation_worker_exits"].get(job["id"]), "Validation actual worker exit receipt SHA changed")
        pair = (r["target_identity_sha256"], r["truth_sha256"])
        old = identities.setdefault(job["dataset"], pair)
        require(old == pair, "Validation arms or seeds do not evaluate identical targets/truth")
    return gate


def validate_payload(payload, job, source, definition, metadata=None):
    require(payload.get("best_epoch") == job["selected_epoch"], "Selected checkpoint epoch mismatch")
    require(payload.get("backbone") == job["arm"] and payload.get("seed") == job["seed"],
            "Checkpoint arm or seed mismatch")
    require(payload.get("variant") == "count_only_log_regression"
            and payload.get("evaluation_scope") == "validation_only"
            and payload.get("held_out_test_evaluated") is False, "Checkpoint objective or original scope changed")
    require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
            and payload.get("checkpoint_monitor_history_key") == "val_qty_rmse"
            and payload.get("checkpoint_selection") == "best_validation_raw_quantity_rmse", "Checkpoint selector changed")
    require(payload.get("initial_state_sha256") == job["initial_state_sha256"], "Checkpoint initialization mismatch")
    require(payload.get("source_revision") == source["revision"]
            and payload.get("source_revision_history") == job["source_revision_history"], "Checkpoint source lineage mismatch")
    require(payload.get("selected_metric_value") == job["selected_metric_value"]
            and payload.get("model_state_sha256") == job["checkpoint_state_sha256"], "Selected checkpoint objective or tensor seal mismatch")
    head = payload.get("interface_meta", {}).get("time_head", {}) or payload.get("interface", {}).get("time_head", {})
    model = definition["model"]
    require(head.get("observation_likelihood") == model["time_observation_contract"]
            and head.get("time_scale") == model["time_scale"]
            and head.get("mode") == model["time_head_mode"], "Frozen recorded time interface changed")
    if metadata is not None:
        actual_metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
        require(canonical(actual_metadata) == canonical(metadata), "Frozen checkpoint complete metadata differs")


def validate_frame(frame, definition, split, np, pl):
    """Apply the frozen recorded-grid checks to the explicitly approved split."""
    required = {"oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split", "demand_dt"}
    require(required <= set(frame.columns) and frame.height > 0, "Frozen input schema is incomplete")
    permitted = {"train", "validation"} if split == "validation" else {"train", "validation", "test"}
    require(set(frame["chronological_split"].unique().to_list()) <= permitted, "Unapproved split materialized")
    require(not frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().any()).item(), "Duplicate target key")
    for key in ("demand_qty", "delta_t", "seq"):
        values = frame[key].to_numpy().astype(np.float64)
        require(np.isfinite(values).all(), "Nonfinite recorded " + key)
        if key != "seq":
            require((values >= 0).all(), "Negative recorded " + key)
        if key != "demand_qty":
            require(np.equal(values, np.round(values)).all(), "Recorded values must be integers: " + key)
    top = definition["model"]["time_observation_contract"]["top_code"]
    require(top is None or (frame["delta_t"].to_numpy() <= top).all(), "Recorded duration exceeds frozen top code")
    # Do not permit a later split to enter an earlier target's context.
    rank = frame["chronological_split"].replace_strict({"train": 0, "validation": 1, "test": 2})
    checked = frame.with_columns(rank.alias("_split_rank")).with_columns(
        pl.col("_split_rank").cum_max().over("oper_part_no").alias("_previous_max"))
    require(checked.filter(pl.col("_split_rank") < pl.col("_previous_max")).height == 0,
            "Frozen chronological splits are out of order")


def require_raf_training_slice(full, training, pl):
    require(set(training.columns) <= set(full.columns), "RAF full parent misses frozen export columns")
    subset = full.filter(pl.col("chronological_split").is_in(["train", "validation"])).select(training.columns)
    require(subset.equals(training), "RAF full parent Train/Validation differs from frozen export")


def population_identity(dataset, split, np):
    """Frozen hard_lmm identity formula, extended operationally to approved Test."""
    index = np.asarray(dataset.index, dtype=np.int64).reshape(-1, 2)
    part, position = index[:, 0], index[:, 1] + 1
    seq = np.fromiter((dataset.seq_lists[p][t] for p, t in zip(part, position)), dtype=np.int64)
    q = np.fromiter((dataset.val_lists[p][t] for p, t in zip(part, position)), dtype=np.float64)
    require(all(dataset.split_lists[p][t] == split for p, t in zip(part, position)), "Loader targets have the wrong split")
    value = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode() + b"\0")
    value.update(json.dumps([str(p) for p in dataset.parts], ensure_ascii=False, separators=(",", ":")).encode())
    for array in (part, position, seq):
        value.update(array.astype("<i8").tobytes())
    return {"target_count": len(q), "target_identity_sha256": value.hexdigest(),
            "target_quantity_sha256": hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + q.astype("<f8").tobytes()).hexdigest()}


def qualify(torch, model, target_outputs, dts, mask, quantities, output, top_code):
    altered_quantity = quantities.clone()
    altered_quantity[:, -1] += 12345.
    q = target_outputs(model, dts, mask, altered_quantity, lambda_log_qty=1.)
    altered_dt = dts.clone()
    altered_dt[:, -1] = (torch.where(dts[:, -1] > 1, torch.ones_like(dts[:, -1]), torch.full_like(dts[:, -1], 2.))
                         if top_code else dts[:, -1] + 123.)
    t = target_outputs(model, altered_dt, mask, quantities, lambda_log_qty=1.)
    single = target_outputs(model, dts[:1], mask[:1], quantities[:1], lambda_log_qty=1.)
    checks = {
        "target_quantity_perturbation_preserves_prediction": torch.allclose(output["pred_qty"], q["pred_qty"], rtol=1e-6, atol=1e-6),
        "target_quantity_perturbation_preserves_time_loss": torch.allclose(output["time_loss"], q["time_loss"], rtol=1e-6, atol=1e-6),
        "target_gap_perturbation_preserves_quantity_prediction": torch.allclose(output["pred_qty"], t["pred_qty"], rtol=1e-6, atol=1e-6),
        "single_vs_batch_prediction_matches": torch.allclose(output["pred_qty"][:1], single["pred_qty"], rtol=1e-5, atol=1e-5),
        "single_vs_batch_time_loss_matches": torch.allclose(output["time_loss"][:1], single["time_loss"], rtol=1e-5, atol=1e-5),
    }
    require(all(checks.values()), "Causality or single/batch qualification failed: " + str(checks))
    return checks


def check_runtime(runtime, expected, device):
    require(runtime["python"] == expected.get("python_version", "3.12.13"), "Pinned Python version mismatch")
    require(Path(sys.executable).resolve() == Path(expected["python"]).resolve(), "Pinned Python executable mismatch")
    for key in ("torch", "cuda", "cudnn", "numpy", "polars"):
        require(runtime[key] == expected[key], "Pinned numerical runtime mismatch: " + key)
    if device == "cuda":
        require(runtime["gpu"]["uuid"] == expected["gpu_uuid"], "Wrong physical GPU UUID")
    for key, value in expected.get("native", {}).items():
        require(runtime[key] == value, "Pinned native runtime mismatch: " + key)
    require(runtime["threads"] == expected["torch_num_threads"] and runtime["interop_threads"] == expected["interop_threads"],
            "Pinned numerical thread counts differ")


def worker(root, job_id, split, device):
    root = Path(root).resolve()
    c = verify_contract(root)
    matching = [r for r in c["jobs"] if r["id"] == job_id]
    require(len(matching) == 1, "Unknown or duplicate scientific job")
    job = matching[0]
    require(split in ("validation", "test") and device in ("cpu", "cuda"), "Invalid evaluation split/device")
    if split == "test":
        verify_global_validation_gate(root, c)  # Before any model, data or CUDA work.
    gate_sha = sha(root / "validation_gate.json") if split == "test" else None
    started_unix = time.time()
    output = root / "results" / job_id
    name = "validation_replay" if split == "validation" else "test_metrics"
    require(not (output / (name + ".json")).exists() and not (output / (name + "_failure.json")).exists(),
            "Existing receipt/failure prohibits automatic repeat")
    started = time.monotonic()
    write_exclusive(output / (name + "_claim.json"), {"job": job_id, "split": split,
        "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv, "contract_sha256": c["contract_sha256"]})
    phase = "integrity"
    try:
        source = c["sources"][job["study"]]
        source_root = verify_source(root, source)
        checkpoint = confined(root, job["checkpoint"]["relative_path"])
        require(sha(checkpoint) == job["checkpoint"]["sha256"], "Checkpoint byte SHA mismatch")
        metadata_path = confined(root, job["checkpoint_metadata_path"])
        require(sha(metadata_path) == job["checkpoint_metadata_sha256"], "Checkpoint metadata byte SHA mismatch")
        expected_metadata = read(metadata_path)
        spec = c["datasets"][job["dataset"]]
        definition = spec["definition"]
        data_key = "training_data_path" if split == "validation" and spec.get("training_data_path") else "data_path"
        sha_key = "training_data_sha256" if data_key == "training_data_path" else "data_sha256"
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
        if split == "validation":
            lazy = lazy.filter(pl.col("chronological_split").is_in(["train", "validation"]))
        frame = lazy.collect().sort(["oper_part_no", "seq"])
        validate_frame(frame, definition, split, np, pl)
        if job["dataset"] == "raf_spare_parts" and split == "test":
            training_path = confined(root, spec["training_data_path"])
            require(sha(training_path) == spec["training_data_sha256"], "RAF frozen export SHA mismatch")
            training = pl.read_parquet(training_path).sort(["oper_part_no", "seq"])
            require_raf_training_slice(frame, training, pl)
        # Audit train-only initialization without refitting or replacing stored scalars.
        train_q = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
        stats = definition["statistics"]
        digest = hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0" + train_q.astype("<f8").tobytes()).hexdigest()
        require(digest == stats["all_train_quantity_sha256"], "Frozen all-train quantity identity changed")
        for key, measured in (("train_log_mean", float(np.log1p(train_q).mean())),
                              ("train_log_std", float(np.log1p(train_q).std()))):
            require(math.isclose(measured, stats[key], rel_tol=0, abs_tol=1e-12), "Train-only initialization drift: " + key)
        frame = core.prepare_count_frame(frame)
        loader = loader_module.make_loader(frame, target_split=split, batch_size=128, shuffle=False, generator=None,
            lookback_weeks=definition["loader"]["lookback_weeks"], max_seq_len=definition["loader"]["max_seq_len"])
        dataset = loader.dataset
        identity = population_identity(dataset, split, np)
        expected_count = job["validation_reference"]["count"] if split == "validation" else spec["expected_test_count"]
        require(len(dataset) == expected_count > 0, "Frozen eligible target population count differs")
        if split == "validation":
            require(identity == definition["inherited_data_identity"]["populations"]["validation"],
                    "Original frozen Validation target identity differs")
        metadata = frame.select(["oper_part_no", "seq", "chronological_split", "demand_dt"])
        offsets, total = [], 0
        for sequences in dataset.seq_lists:
            offsets.append(total)
            total += len(sequences)
        require(total == frame.height, "Loader frame mapping differs")
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
        metrics = metrics_module.QuantityMetrics(bounds)
        simple = {key: metrics_module.QuantityMetrics(bounds) for key in ("last_quantity", "mean_quantity")}
        target_digest, truth_digest = hashlib.sha256(), hashlib.sha256()
        qualification, processed, inference_seconds = {}, 0, 0.
        phase = "inference"
        with torch.no_grad():
            for batch_index, (_, dts, mask, part_indices, quantities) in enumerate(loader):
                dts, mask, quantities = (v.to(execution_device) for v in (dts, mask, quantities))
                if device == "cuda":
                    torch.cuda.synchronize()
                batch_start = time.monotonic()
                value = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
                if device == "cuda":
                    torch.cuda.synchronize()
                inference_seconds += time.monotonic() - batch_start
                if batch_index == 0:
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
                processed += len(q)
        require(processed == expected_count, "Incomplete Test or Validation population")
        final_metrics = {"split": split, **metrics.finish()}
        if split == "validation":
            compare_validation_metrics(final_metrics, job["validation_reference"])
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
        if split == "test":
            verify_global_validation_gate(root, c)
            require(sha(root / "validation_gate.json") == gate_sha, "Global Validation gate changed during Test")
        imports = verify_imports(source_root, source)
        elapsed = time.monotonic() - started
        receipt = {"status": "complete", "schema": KIND, "contract_sha256": c["contract_sha256"],
            **{key: job[key] for key in ("id", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256")},
            "job_id": job["id"], "frozen_source_closure": source["closure"],
            "split": split, "evaluation_scope": "validation_replay" if split == "validation" else "test_final_comparison",
            "checkpoint_sha256": job["checkpoint"]["sha256"], "state_sha256": before,
            "source_closure": source["closure"], "source_revision": source["revision"],
            "source_revision_history": job["source_revision_history"], "full_population": True,
            "expected_target_count": expected_count, "prediction_rows": processed, "metrics": final_metrics,
            "simple_quantity_baselines": simple_results, "original_population_identity": identity,
            "target_identity_sha256": target_digest.hexdigest(), "truth_sha256": truth_digest.hexdigest(),
            "data_identity": {"raw_sha256": spec[sha_key], "path": spec[data_key]},
            "truth_digest_fields": TRUTH_FIELDS, "data_sha256": spec[sha_key], "loader": definition["loader"],
            "batch_size": 128, "shuffle": False, "num_workers": 0, "drop_last": False,
            "qualification": {**qualification, "parameters_unchanged": True, "finite_outputs": True},
            "validation_match": "PASS" if split == "validation" else None,
            "validation_replay_check": {"passed": split == "validation", "tolerance": {"relative": 1e-5, "absolute": 1e-5},
                "rule": "math.isclose", "count_exact": True, "six_metrics": split == "validation"},
            "retrained": False, "reselected": False, "test_independent": False,
            "held_out_test_evaluated": split == "test", "raw_predictions_written": False,
            "imported_frozen_modules": imports, "evaluator_sha256": script_sha, "runtime": runtime,
            "pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv,
            "observed_unix": time.time(), "started_unix": started_unix, "completed_unix": time.time(),
            "test_started_unix": started_unix if split == "test" else None,
            "validation_gate_sha256": gate_sha, "elapsed_seconds": elapsed, "inference_seconds": inference_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
            "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
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
    """Fabricated causal/route/runtime admission, without optimization or real data."""
    root = Path(root).resolve()
    c = verify_contract(root)
    jobs = [j for j in c["jobs"] if j["study"] == study and j["arm"] == arm]
    require(bool(jobs), "Unapproved native probe route")
    job = jobs[0]
    source = c["sources"][study]
    source_root = verify_source(root, source)
    require(not any(n.split(".")[0] in PROJECT_PREFIXES for n in sys.modules), "Fresh native probe process is required")
    sys.dont_write_bytecode = True
    sys.path[:] = [str(source_root)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)
                                            and not (Path(p).resolve() / "paper").is_dir()]
    cache = root / "control" / "native_probe_runtime_cache" / study / arm
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
    execution_device = "cuda:0" if device == "cuda" else "cpu"
    numerical.configure_runtime(execution_device, 4)
    runtime = numerical.runtime_identity(execution_device)
    if device == "cuda":
        check_runtime(runtime, c["runtime"], device)
    engine = importlib.import_module(route_for(job))
    engine.install_hooks()
    factory = importlib.import_module("models.TPPs.CountAwareFactory")
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    runner = importlib.import_module("simple_lab_test.search.common.runner")
    definition = c["datasets"][job["dataset"]]["definition"]
    config = {k: v for k, v in definition["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    with torch.random.fork_rng(devices=[] if device == "cpu" else [0]):
        torch.manual_seed(42)
        model, _ = factory.build_count_aware_model(arm, **config, train_log_mean=1.2, train_log_std=.8,
            max_seq_len=definition["loader"]["max_seq_len"])
        model.to(execution_device).eval().requires_grad_(False)
        with torch.no_grad():
            # Exercise synthetic correction paths past zero-output initialization.
            # Only this newly constructed fixture is altered; no saved model is loaded.
            for name, parameter in model.named_parameters():
                if "multilag_detail" in name or "causal_q_kernel" in name or "causal_k_kernel" in name or "causal_v_kernel" in name:
                    parameter.copy_(torch.randn_like(parameter) * .03)
            before = runner.canonical_state_dict_sha256(model.state_dict())
            dts = torch.tensor([[0., 0., 1., 2., 1., 3., 1., 2.], [1., 2., 1., 3., 1., 2., 1., 3.]], device=execution_device)
            quantities = torch.tensor([[0., 0., 2., 4., 1., 3., 2., 5.], [1., 2., 3., 4., 3., 2., 1., 6.]], device=execution_device)
            mask = torch.ones_like(dts, dtype=torch.bool)
            mask[0, :2] = False
            output = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
            checks = qualify(torch, model, core.target_outputs, dts, mask, quantities, output,
                definition["model"]["time_observation_contract"]["top_code"])
            require(all(bool(torch.isfinite(v).all()) for v in output.values()), "Nonfinite synthetic native output")
            require(runner.canonical_state_dict_sha256(model.state_dict()) == before, "Synthetic inference changed model tensors")
    verify_source(root, source)
    imports = verify_imports(source_root, source)
    result = {"status": "PASS", "study": study, "arm": arm, "device": device,
        "runtime": runtime, "native_pinned_contract_checked": device == "cuda", "source_closure": source["closure"],
        "qualification": {**checks, "parameters_unchanged": True, "finite_outputs": True},
        "optimizer_updates": 0, "checkpoint_reads": 0, "real_data_loaded": False, "held_out_test_evaluated": False,
        "imported_frozen_modules": imports, "contract_sha256": c["contract_sha256"]}
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--native-probe", action="store_true")
    parser.add_argument("--study", choices=("history", "instacart"))
    parser.add_argument("--arm")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    if args.native_probe:
        require(args.study is not None and args.arm is not None and args.job_id is None and args.split is None,
                "Native probe requires only --study/--arm, without a research job/split")
        return native_probe(args.root, args.study, args.arm, args.device)
    require(args.job_id is not None and args.split is not None and args.study is None and args.arm is None,
            "Research worker requires --job-id/--split")
    return worker(args.root, args.job_id, args.split, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
