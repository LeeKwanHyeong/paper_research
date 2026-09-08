#!/usr/bin/env python3
"""Extract paired Instacart validation predictions for frozen Hard-LMM models.

This runner is deliberately diagnostic-only.  It restores the pinned B, FULL,
and BOUNDED-QK checkpoints, presents every model with the same validation batch,
and writes one aligned row per canonical next-event target.  It never trains,
selects a checkpoint, fits a calibration, or materializes the held-out split.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ID = "hard_lmm_bounded_qk_error_diagnostic_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/hard_lmm_bounded_qk_error_diagnostic_v1.json"
)
MODEL_ROLES = ("B", "FULL", "BOUNDED_QK")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong diagnostic contract")
    require(
        contract.get("status") == "frozen_before_row_level_inference",
        "Diagnostic contract is not prospectively frozen",
    )
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope is missing")
    require(scope.get("dataset") == "insta_market_basket", "Dataset scope drift")
    require(scope.get("seed") == 42, "Seed drift")
    require(scope.get("target_split") == "validation", "Target split drift")
    require(scope.get("held_out_test") is False, "Held-out access is enabled")
    for forbidden in ("training", "checkpoint_selection", "parameter_updates", "calibration_fit"):
        require(scope.get(forbidden) is False, f"Forbidden operation enabled: {forbidden}")
    require(scope.get("primary_contrast") == "BOUNDED_QK_minus_B", "Primary contrast drift")
    dataset = contract.get("dataset")
    require(isinstance(dataset, Mapping), "Dataset contract is missing")
    require(dataset.get("loader") == "RMTPPWeekLookbackDataset", "Loader drift")
    require(dataset.get("lookback_weeks") == 52, "Lookback drift")
    require(dataset.get("max_sequence_length") == 64, "Sequence-length drift")
    require(dataset.get("expected_validation_targets") == 503733, "Target-count drift")
    require(dataset.get("train_quantity_boundaries") == [8.0, 20.0, 25.0, 35.0], "Quantity-boundary drift")
    execution = contract.get("execution")
    require(isinstance(execution, Mapping), "Execution contract is missing")
    require(execution.get("frozen_source_revision") == "89cd700c28dd169cd59bfccbd694ef8967bac823", "Frozen revision drift")
    require(execution.get("shuffle") is False, "Inference order must be deterministic")
    require(execution.get("same_dataset_instance_for_every_model") is True, "Dataset parity disabled")
    models = contract.get("models")
    require(isinstance(models, Mapping) and list(models) == list(MODEL_ROLES), "Model-role drift")
    for role, spec in models.items():
        require(spec.get("checkpoint_selection") == "best_validation_raw_quantity_rmse", f"Selector drift: {role}")
        for name in ("checkpoint_file_sha256", "checkpoint_state_sha256", "summary_file_sha256"):
            value = str(spec.get(name, ""))
            require(len(value) == 64 and set(value) <= set("0123456789abcdef"), f"Invalid digest: {role}.{name}")


def verify_frozen_source(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), "Frozen source root is missing")
    expected = contract["execution"]["source_file_sha256"]
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        candidate = (root / relative).resolve()
        require(candidate.is_relative_to(root), f"Source path escapes root: {relative}")
        require(candidate.is_file(), f"Frozen source file is missing: {relative}")
        observed[relative] = sha256_file(candidate)
        require(observed[relative] == digest, f"Frozen source file drift: {relative}")
    return {
        "root": str(root),
        "revision": contract["execution"]["frozen_source_revision"],
        "files": observed,
        "all_file_hashes_verified": True,
    }


def import_frozen_runtime(root: Path) -> dict[str, Any]:
    root_text = str(root.resolve())
    require(not any(name == "models" or name.startswith("models.") for name in sys.modules), "Model modules were imported before the frozen source root")
    sys.path.insert(0, root_text)
    from paper.scripts.count_aware_tpp_backbone.core import (  # type: ignore
        load_train_validation_frame,
        prepare_count_frame,
        target_outputs,
    )
    from paper.scripts.run_count_aware_tpp_backbone_control import (  # type: ignore
        exact_target_population,
    )
    from paper.scripts.run_matched_frozen_lognormal_duration import (  # type: ignore
        build_source_model,
    )
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader  # type: ignore
    from simple_lab_test.search.common.runner import (  # type: ignore
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    return {
        "load_train_validation_frame": load_train_validation_frame,
        "prepare_count_frame": prepare_count_frame,
        "target_outputs": target_outputs,
        "exact_target_population": exact_target_population,
        "build_source_model": build_source_model,
        "make_loader": make_loader,
        "canonical_state_dict_sha256": canonical_state_dict_sha256,
        "torch_load_checkpoint": torch_load_checkpoint,
    }


def history_features(
    quantities: np.ndarray,
    durations: np.ndarray,
) -> dict[str, float]:
    """Summarize an observed prefix; callers must exclude the target."""
    quantities = np.asarray(quantities, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    require(quantities.ndim == durations.ndim == 1, "History arrays must be one-dimensional")
    require(quantities.size == durations.size and quantities.size > 0, "History arrays must be nonempty and aligned")
    require(bool(np.isfinite(quantities).all()) and bool((quantities >= 0.0).all()), "Invalid history quantity")
    require(bool(np.isfinite(durations).all()) and bool((durations >= 1.0).all()), "Invalid history duration")
    logs = np.log1p(quantities)
    recent3 = logs[-3:]
    recent5 = logs[-5:]
    mean_log = float(logs.mean(dtype=np.float64))
    previous_qty = float(quantities[-2]) if quantities.size >= 2 else math.nan
    last_change = float(logs[-1] - logs[-2]) if quantities.size >= 2 else math.nan
    recent3_mean = float(recent3.mean(dtype=np.float64))
    return {
        "last_qty": float(quantities[-1]),
        "previous_qty": previous_qty,
        "history_mean_qty": float(quantities.mean(dtype=np.float64)),
        "history_mean_log_qty": mean_log,
        "recent3_mean_log_qty": recent3_mean,
        "recent5_mean_log_qty": float(recent5.mean(dtype=np.float64)),
        "last_log_qty_change": last_change,
        "last_minus_history_mean_log_qty": float(logs[-1] - mean_log),
        "recent3_minus_history_mean_log_qty": float(recent3_mean - mean_log),
        "last_history_dt": float(durations[-1]),
        "history_mean_log1p_dt": float(np.log1p(durations).mean(dtype=np.float64)),
    }


def build_validation_metadata(dataset: Any, *, lookback_weeks: int, max_seq_len: int) -> dict[str, Any]:
    count = len(dataset)
    integer_fields = {
        "dataset_index": np.arange(count, dtype=np.int64),
        "series_index": np.empty(count, dtype=np.int64),
        "context_end": np.empty(count, dtype=np.int64),
        "target_position": np.empty(count, dtype=np.int64),
        "target_seq": np.empty(count, dtype=np.int64),
        "history_length": np.empty(count, dtype=np.int64),
        "full_series_prior_event_count": np.empty(count, dtype=np.int64),
    }
    float_names = (
        "true_qty",
        "target_dt",
        "last_qty",
        "previous_qty",
        "history_mean_qty",
        "history_mean_log_qty",
        "recent3_mean_log_qty",
        "recent5_mean_log_qty",
        "last_log_qty_change",
        "last_minus_history_mean_log_qty",
        "recent3_minus_history_mean_log_qty",
        "last_history_dt",
        "history_mean_log1p_dt",
    )
    float_fields = {name: np.empty(count, dtype=np.float64) for name in float_names}
    series_ids: list[str] = [""] * count
    for dataset_index, (part_index, context_end) in enumerate(dataset.index):
        target_position = context_end + 1
        seq = np.asarray(dataset.seq_lists[part_index], dtype=np.int64)
        quantities = np.asarray(dataset.val_lists[part_index], dtype=np.float64)
        durations = np.asarray(dataset.dt_lists[part_index], dtype=np.float64)
        left_seq = int(seq[context_end]) - (int(lookback_weeks) - 1)
        history_start = int(np.searchsorted(seq, left_seq, side="left"))
        maximum_history = int(max_seq_len) - 1
        if context_end - history_start + 1 > maximum_history:
            history_start = context_end - maximum_history + 1
        observed_quantities = quantities[history_start : context_end + 1]
        observed_durations = durations[history_start : context_end + 1]
        features = history_features(observed_quantities, observed_durations)
        series_ids[dataset_index] = str(dataset.parts[part_index])
        integer_fields["series_index"][dataset_index] = part_index
        integer_fields["context_end"][dataset_index] = context_end
        integer_fields["target_position"][dataset_index] = target_position
        integer_fields["target_seq"][dataset_index] = int(seq[target_position])
        integer_fields["history_length"][dataset_index] = observed_quantities.size
        integer_fields["full_series_prior_event_count"][dataset_index] = context_end + 1
        float_fields["true_qty"][dataset_index] = float(quantities[target_position])
        float_fields["target_dt"][dataset_index] = float(durations[target_position])
        for name, value in features.items():
            float_fields[name][dataset_index] = value
    return {"series_id": series_ids, **integer_fields, **float_fields}


def checkpoint_paths(args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    return {
        "B": (args.b_checkpoint, args.b_summary),
        "FULL": (args.full_checkpoint, args.full_summary),
        "BOUNDED_QK": (args.bounded_checkpoint, args.bounded_summary),
    }


def restore_models(
    *,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    models: dict[str, torch.nn.Module] = {}
    audit: dict[str, Any] = {}
    for role, (checkpoint_path, summary_path) in checkpoint_paths(args).items():
        spec = contract["models"][role]
        require(checkpoint_path.is_file(), f"Checkpoint is missing: {role}")
        require(summary_path.is_file(), f"Summary is missing: {role}")
        checkpoint_digest = sha256_file(checkpoint_path)
        summary_digest = sha256_file(summary_path)
        require(checkpoint_digest == spec["checkpoint_file_sha256"], f"Checkpoint digest drift: {role}")
        require(summary_digest == spec["summary_file_sha256"], f"Summary digest drift: {role}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = runtime["torch_load_checkpoint"](checkpoint_path, map_location="cpu")
        for field, expected in {
            "backbone": spec["backbone"],
            "variant": "count_only_log_regression",
            "seed": 42,
            "selection": spec["checkpoint_selection"],
            "checkpoint_selection": spec["checkpoint_selection"],
            "source_revision": spec["training_source_revision"],
            "best_epoch": spec["best_epoch"],
            "model_state_sha256": spec["checkpoint_state_sha256"],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }.items():
            require(payload.get(field) == expected, f"Checkpoint metadata drift: {role}.{field}")
        require(summary.get("checkpoint_state_sha256") == spec["checkpoint_state_sha256"], f"Summary state drift: {role}")
        require(summary.get("best_epoch") == spec["best_epoch"], f"Summary epoch drift: {role}")
        require(summary.get("held_out_test_evaluated") is False, f"Summary held-out scope drift: {role}")
        model = runtime["build_source_model"](payload, max_seq_len=64)
        model.requires_grad_(False).eval().to(device)
        restored_digest = runtime["canonical_state_dict_sha256"](model.state_dict())
        require(restored_digest == spec["checkpoint_state_sha256"], f"Strict model replay drift: {role}")
        models[role] = model
        audit[role] = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_file_sha256": checkpoint_digest,
            "checkpoint_state_sha256": restored_digest,
            "summary_path": str(summary_path.resolve()),
            "summary_file_sha256": summary_digest,
            "best_epoch": spec["best_epoch"],
            "backbone": spec["backbone"],
        }
    return models, audit


def quantity_metrics(true_qty: np.ndarray, pred_qty: np.ndarray) -> dict[str, float | int]:
    true_qty = np.asarray(true_qty, dtype=np.float64)
    pred_qty = np.asarray(pred_qty, dtype=np.float64)
    require(true_qty.shape == pred_qty.shape and true_qty.size > 0, "Metric arrays are not aligned")
    error = pred_qty - true_qty
    return {
        "count": int(error.size),
        "qty_mae": float(np.abs(error).mean(dtype=np.float64)),
        "qty_rmse": float(np.sqrt(np.square(error).mean(dtype=np.float64))),
        "qty_bias": float(error.mean(dtype=np.float64)),
        "log_qty_mse": float(np.square(np.log1p(pred_qty) - np.log1p(true_qty)).mean(dtype=np.float64)),
    }


def validate_replay(
    observed: Mapping[str, float | int],
    expected: Mapping[str, float | int],
    *,
    tolerance: float,
    role: str,
) -> dict[str, Any]:
    require(observed["count"] == expected["count"], f"Replay count drift: {role}")
    differences: dict[str, float] = {}
    for name in ("qty_mae", "qty_rmse", "qty_bias", "log_qty_mse"):
        differences[name] = abs(float(observed[name]) - float(expected[name]))
        require(differences[name] <= tolerance, f"Replay metric drift: {role}.{name}={differences[name]}")
    return {"status": "exact_within_tolerance", "absolute_differences": differences}


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    contract_sha = sha256_file(args.contract)
    source_audit = verify_frozen_source(args.frozen_source_root, contract)
    runtime = import_frozen_runtime(args.frozen_source_root)
    dataset_spec = contract["dataset"]
    require(sha256_file(args.data) == dataset_spec["data_sha256"], "Dataset digest drift")
    require(sha256_file(args.split_manifest) == dataset_spec["split_manifest_sha256"], "Split-manifest digest drift")
    device = torch.device(args.device)
    cuda_device_index: int | None = None
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        cuda_device_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        device = torch.device("cuda", cuda_device_index)
        torch.cuda.set_device(cuda_device_index)
        torch.cuda.reset_peak_memory_stats(cuda_device_index)
    frame = runtime["load_train_validation_frame"](args.data)
    admitted_splits = set(frame["chronological_split"].unique().to_list())
    require(admitted_splits == {"train", "validation"}, "Unexpected admitted split set")
    frame = runtime["prepare_count_frame"](frame)
    target_quantities, population = runtime["exact_target_population"](
        frame,
        target_split="validation",
        lookback_weeks=dataset_spec["lookback_weeks"],
        max_seq_len=dataset_spec["max_sequence_length"],
    )
    for field, expected in {
        "target_count": dataset_spec["expected_validation_targets"],
        "target_identity_sha256": dataset_spec["expected_validation_identity_sha256"],
        "target_quantity_sha256": dataset_spec["expected_validation_quantity_sha256"],
    }.items():
        require(population.get(field) == expected, f"Validation population drift: {field}")
    loader = runtime["make_loader"](
        frame,
        target_split="validation",
        batch_size=int(args.batch_size),
        lookback_weeks=dataset_spec["lookback_weeks"],
        max_seq_len=dataset_spec["max_sequence_length"],
        shuffle=False,
        generator=None,
    )
    dataset = loader.dataset
    require(len(dataset) == dataset_spec["expected_validation_targets"], "Loader target-count drift")
    metadata = build_validation_metadata(
        dataset,
        lookback_weeks=dataset_spec["lookback_weeks"],
        max_seq_len=dataset_spec["max_sequence_length"],
    )
    require(np.array_equal(metadata["true_qty"], target_quantities), "Metadata target order drift")
    models, checkpoint_audit = restore_models(
        args=args,
        contract=contract,
        runtime=runtime,
        device=device,
    )
    before_states = {
        role: runtime["canonical_state_dict_sha256"](model.state_dict())
        for role, model in models.items()
    }
    predictions = {
        role: np.empty(len(dataset), dtype=np.float64) for role in MODEL_ROLES
    }
    log_losses = {
        role: np.empty(len(dataset), dtype=np.float64) for role in MODEL_ROLES
    }
    offset = 0
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (_, dts, mask, parts, quantities) in enumerate(loader):
            require(quantities is not None, "Raw quantities are missing")
            batch_count = int(dts.size(0))
            end = offset + batch_count
            require(
                np.array_equal(parts.numpy().astype(np.int64), metadata["series_index"][offset:end]),
                "Loader series order drift",
            )
            dts_device = dts.to(device, non_blocking=device.type == "cuda")
            mask_device = mask.to(device, non_blocking=device.type == "cuda")
            quantities_device = quantities.to(device, non_blocking=device.type == "cuda")
            reference_true: torch.Tensor | None = None
            reference_history: torch.Tensor | None = None
            for role in MODEL_ROLES:
                output = runtime["target_outputs"](
                    models[role],
                    dts_device,
                    mask_device,
                    quantities_device,
                    lambda_log_qty=1.0,
                )
                true_cpu = output["true_qty"].detach().cpu()
                history_cpu = output["history_length"].detach().cpu()
                if reference_true is None:
                    reference_true = true_cpu
                    reference_history = history_cpu
                else:
                    require(torch.equal(true_cpu, reference_true), f"Three-way target drift: {role}")
                    require(torch.equal(history_cpu, reference_history), f"Three-way history drift: {role}")
                predictions[role][offset:end] = output["pred_qty"].detach().cpu().numpy().astype(np.float64)
                log_losses[role][offset:end] = output["log_qty_loss"].detach().cpu().numpy().astype(np.float64)
            assert reference_true is not None and reference_history is not None
            require(
                np.array_equal(reference_true.numpy().astype(np.float64), metadata["true_qty"][offset:end]),
                "Batch target values drift from canonical metadata",
            )
            require(
                np.array_equal(reference_history.numpy().astype(np.int64), metadata["history_length"][offset:end]),
                "Batch history lengths drift from canonical metadata",
            )
            offset = end
            if batch_index % 100 == 0 or offset == len(dataset):
                print(
                    json.dumps(
                        {
                            "stage": "paired_validation_inference",
                            "completed_targets": offset,
                            "total_targets": len(dataset),
                            "elapsed_seconds": time.perf_counter() - inference_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    require(offset == len(dataset), "Incomplete paired inference")
    replay: dict[str, Any] = {}
    for role in MODEL_ROLES:
        require(bool(np.isfinite(predictions[role]).all()) and bool((predictions[role] >= 0.0).all()), f"Invalid predictions: {role}")
        observed = quantity_metrics(metadata["true_qty"], predictions[role])
        official_log_mse = float(log_losses[role].mean(dtype=np.float64))
        require(abs(official_log_mse - float(observed["log_qty_mse"])) <= 1e-6, f"Official log-MSE mismatch: {role}")
        replay[role] = {
            "observed": observed,
            "official_log_qty_mse": official_log_mse,
            **validate_replay(
                observed,
                contract["models"][role]["expected_validation"],
                tolerance=float(contract["execution"]["aggregate_replay_absolute_tolerance"]),
                role=role,
            ),
        }
    after_states = {
        role: runtime["canonical_state_dict_sha256"](model.state_dict())
        for role, model in models.items()
    }
    require(before_states == after_states, "A source model changed during inference")
    require(all(parameter.grad is None for model in models.values() for parameter in model.parameters()), "A frozen model accumulated gradients")
    output_columns: dict[str, Any] = dict(metadata)
    true_qty = metadata["true_qty"]
    for role in MODEL_ROLES:
        key = role.lower()
        prediction = predictions[role]
        error = prediction - true_qty
        output_columns[f"pred_{key}"] = prediction
        output_columns[f"error_{key}"] = error
        output_columns[f"abs_error_{key}"] = np.abs(error)
        output_columns[f"sq_error_{key}"] = np.square(error)
        output_columns[f"log_sq_error_{key}"] = log_losses[role]
    for left, right, label in (
        ("BOUNDED_QK", "B", "bounded_qk_minus_b"),
        ("BOUNDED_QK", "FULL", "bounded_qk_minus_full"),
        ("FULL", "B", "full_minus_b"),
    ):
        left_key, right_key = left.lower(), right.lower()
        output_columns[f"delta_prediction_{label}"] = predictions[left] - predictions[right]
        output_columns[f"delta_abs_error_{label}"] = output_columns[f"abs_error_{left_key}"] - output_columns[f"abs_error_{right_key}"]
        output_columns[f"delta_sq_error_{label}"] = output_columns[f"sq_error_{left_key}"] - output_columns[f"sq_error_{right_key}"]
    paired_frame = pl.DataFrame(output_columns)
    require(paired_frame.height == dataset_spec["expected_validation_targets"], "Paired frame row-count drift")
    require(
        paired_frame.select(pl.struct(["series_id", "target_position", "target_seq"]).n_unique()).item() == paired_frame.height,
        "Canonical target keys are not unique",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "paired_validation_predictions.parquet"
    atomic_write_parquet(paired_frame, prediction_path)
    audit = {
        "schema": "hard_lmm_bounded_qk_error_diagnostic_run_v1",
        "status": "success",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": contract_sha,
            "id": CONTRACT_ID,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "scope": contract["scope"],
        "held_out_test_evaluated": False,
        "training_performed": False,
        "calibration_fit_performed": False,
        "data": {
            "path": str(args.data.resolve()),
            "sha256": sha256_file(args.data),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "admitted_splits": sorted(admitted_splits),
            "population": population,
        },
        "frozen_source": source_audit,
        "checkpoints": checkpoint_audit,
        "state_identity": {
            "before": before_states,
            "after": after_states,
            "unchanged": True,
            "all_gradients_absent": True,
        },
        "aggregate_replay": replay,
        "paired_predictions": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "row_count": paired_frame.height,
            "column_count": paired_frame.width,
            "canonical_identity_unique": True,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "inference_elapsed_seconds": time.perf_counter() - inference_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device_index)) if device.type == "cuda" else 0,
            "pid": os.getpid(),
        },
    }
    save_json(args.output_dir / "run_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--b-checkpoint", type=Path, required=True)
    parser.add_argument("--b-summary", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--bounded-checkpoint", type=Path, required=True)
    parser.add_argument("--bounded-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    require(args.batch_size > 0, "Batch size must be positive")
    return args


if __name__ == "__main__":
    completed = run(parse_args())
    print(json.dumps({"status": completed["status"], "paired_predictions": completed["paired_predictions"]}, sort_keys=True))
