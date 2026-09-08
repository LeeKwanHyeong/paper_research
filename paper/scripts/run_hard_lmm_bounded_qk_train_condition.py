#!/usr/bin/env python3
"""Extract the one missing frozen BOUNDED-QK Instacart train prediction cache.

The runner reuses the prospectively pinned Frozen-B train cache and evaluates
only the already-selected BOUNDED-QK checkpoint on the exact same canonical
train targets.  It never trains, selects, calibrates, or materializes a
validation/test row.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import polars as pl
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ID = "hard_lmm_bounded_qk_train_condition_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/hard_lmm_bounded_qk_train_condition_v1.json"
)
B_CACHE_KEYS = {
    "prediction",
    "quantity",
    "time_nll",
    "history_length",
    "series_index",
    "target_index",
    "context_end",
}
REQUIRED_CONDITION_IDS = (
    "mean_prediction_shift_negative",
    "mean_prediction_shift_material",
    "median_series_mean_prediction_shift_negative",
    "centered_mse_improves",
    "overall_mse_worsens",
    "bias_squared_penalty_increases",
    "bias_penalty_overturns_centered_gain",
    "mae_not_worse",
    "history_2_3_mse_materially_worse",
    "history_2_3_shift_more_negative_than_8_15",
    "history_8_15_mse_improves",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray, *, label: str) -> str:
    """Replay the hash format used by the reusable Frozen-B cache."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"frozen_raw_affine_array_v1\0")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def output_array_sha256(value: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"hard_lmm_bounded_qk_train_array_v1\0")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong train-condition contract")
    require(
        contract.get("status") == "frozen_before_bounded_qk_train_inference",
        "Train-condition contract is not prospectively frozen",
    )
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope is missing")
    require(scope.get("dataset") == "insta_market_basket", "Dataset scope drift")
    require(scope.get("seed") == 42, "Seed drift")
    require(scope.get("target_split") == "train", "Target split drift")
    require(scope.get("input_splits_materialized") == ["train"], "Non-train input admitted")
    require(scope.get("held_out_test") is False, "Held-out access enabled")
    require(scope.get("validation_targets") is False, "Validation target access enabled")
    for forbidden in ("training", "checkpoint_selection", "parameter_updates", "calibration_fit"):
        require(scope.get(forbidden) is False, f"Forbidden operation enabled: {forbidden}")
    dataset = contract.get("dataset")
    require(isinstance(dataset, Mapping), "Dataset contract missing")
    require(dataset.get("expected_train_targets") == 1991192, "Train target count drift")
    require(dataset.get("lookback_weeks") == 52, "Lookback drift")
    require(dataset.get("max_sequence_length") == 64, "Sequence length drift")
    execution = contract.get("execution")
    require(isinstance(execution, Mapping), "Execution contract missing")
    require(execution.get("shuffle") is False, "Inference order must be fixed")
    require(execution.get("numeric_export_dtype") == "float64", "Export dtype drift")
    models = contract.get("models")
    require(isinstance(models, Mapping) and set(models) == {"B", "BOUNDED_QK"}, "Model roles drift")
    analysis = contract.get("analysis")
    require(isinstance(analysis, Mapping), "Analysis contract missing")
    require(analysis.get("fold_count") == 2, "Fold count drift")
    require(
        analysis.get("fold_salt") == "hard_lmm_bounded_qk_train_condition_v1:20260908",
        "Fold salt drift",
    )
    require(
        analysis.get("history_bins") == ["1", "2-3", "4-7", "8-15", "16-31", "32-63"],
        "History-bin drift",
    )
    conditions = analysis.get("required_conditions")
    require(
        isinstance(conditions, Mapping)
        and tuple(conditions) == REQUIRED_CONDITION_IDS,
        "Required-condition drift",
    )
    require(
        analysis.get("acceptance") == "all required conditions must pass in both fixed folds",
        "Acceptance-rule drift",
    )
    require(analysis.get("failure_action") == "stop_bounded_qk_family", "Failure action drift")
    require(
        analysis.get("pass_action")
        == "freeze one common level-preserving history-confidence residual architecture contract before implementation",
        "Pass action drift",
    )


def verify_frozen_source(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), "Frozen source root missing")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    require(revision == contract["execution"]["frozen_source_revision"], "Frozen source revision drift")
    require(not status, "Frozen source checkout is not clean")
    observed: dict[str, str] = {}
    for relative, expected in contract["execution"]["source_file_sha256"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file(), f"Frozen source file missing: {relative}")
        observed[relative] = sha256_file(path)
        require(observed[relative] == expected, f"Frozen source file drift: {relative}")
    return {
        "root": str(root),
        "revision": revision,
        "git_worktree_clean": True,
        "all_file_hashes_verified": True,
        "files": observed,
    }


def import_frozen_runtime(root: Path) -> dict[str, Any]:
    require(
        not any(name == "models" or name.startswith("models.") for name in sys.modules),
        "Model modules imported before frozen source",
    )
    sys.path.insert(0, str(root.resolve()))
    from paper.scripts.count_aware_tpp_backbone.core import (  # type: ignore
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
        "prepare_count_frame": prepare_count_frame,
        "target_outputs": target_outputs,
        "exact_target_population": exact_target_population,
        "build_source_model": build_source_model,
        "make_loader": make_loader,
        "canonical_state_dict_sha256": canonical_state_dict_sha256,
        "torch_load_checkpoint": torch_load_checkpoint,
    }


def load_train_only_frame(path: Path, runtime: Mapping[str, Any]) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(set(frame["chronological_split"].unique().to_list()) == {"train"}, "Train-only scan drift")
    return runtime["prepare_count_frame"](frame)


def load_and_verify_b_cache(
    cache_path: Path,
    result_path: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    spec = contract["models"]["B"]
    require(cache_path.is_file() and result_path.is_file(), "Frozen-B cache evidence missing")
    require(sha256_file(cache_path) == spec["train_cache_file_sha256"], "Frozen-B cache file drift")
    require(sha256_file(result_path) == spec["train_result_file_sha256"], "Frozen-B result file drift")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result.get("status") == "success" and result.get("phase") == "train", "Frozen-B result scope drift")
    require(result.get("dataset") == "insta_market_basket" and result.get("model_role") == "B", "Frozen-B role drift")
    require(result.get("held_out_test_evaluated") is False, "Frozen-B result used held-out test")
    require(result.get("source_checkpoint_file_sha256") == spec["checkpoint_file_sha256"], "Frozen-B checkpoint drift")
    require(result.get("source_model_state_sha256") == spec["checkpoint_state_sha256"], "Frozen-B state drift")
    expected_population = contract["dataset"]
    population = result["target_population"]["train"]
    require(population.get("target_count") == expected_population["expected_train_targets"], "Frozen-B target count drift")
    require(population.get("target_identity_sha256") == expected_population["expected_train_identity_sha256"], "Frozen-B identity drift")
    require(population.get("target_quantity_sha256") == expected_population["expected_train_quantity_sha256"], "Frozen-B target quantity drift")
    with np.load(cache_path, allow_pickle=False) as archive:
        require(set(archive.files) == B_CACHE_KEYS, "Frozen-B cache schema drift")
        cache = {name: archive[name].copy() for name in archive.files}
    count = expected_population["expected_train_targets"]
    require(all(value.shape == (count,) for value in cache.values()), "Frozen-B cache shape drift")
    for name, expected in spec["train_cache_array_sha256"].items():
        require(array_sha256(cache[name], label=name) == expected, f"Frozen-B array drift: {name}")
    require(np.array_equal(cache["target_index"], np.arange(count, dtype=np.int64)), "Frozen-B target order drift")
    require(bool(np.isfinite(cache["prediction"]).all()) and bool((cache["prediction"] >= 0).all()), "Invalid Frozen-B predictions")
    return cache, {
        "cache_path": str(cache_path.resolve()),
        "cache_file_sha256": sha256_file(cache_path),
        "result_path": str(result_path.resolve()),
        "result_file_sha256": sha256_file(result_path),
        "checkpoint_file_sha256": spec["checkpoint_file_sha256"],
        "checkpoint_state_sha256": spec["checkpoint_state_sha256"],
        "all_array_hashes_verified": True,
    }


def restore_bounded_model(
    *,
    checkpoint_path: Path,
    summary_path: Path,
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec = contract["models"]["BOUNDED_QK"]
    require(checkpoint_path.is_file() and summary_path.is_file(), "BOUNDED-QK evidence missing")
    require(sha256_file(checkpoint_path) == spec["checkpoint_file_sha256"], "BOUNDED-QK checkpoint drift")
    require(sha256_file(summary_path) == spec["summary_file_sha256"], "BOUNDED-QK summary drift")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = runtime["torch_load_checkpoint"](checkpoint_path, map_location="cpu")
    expected_metadata = {
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
    }
    for name, expected in expected_metadata.items():
        require(payload.get(name) == expected, f"BOUNDED-QK checkpoint metadata drift: {name}")
    require(summary.get("checkpoint_state_sha256") == spec["checkpoint_state_sha256"], "BOUNDED-QK summary state drift")
    require(summary.get("best_epoch") == spec["best_epoch"], "BOUNDED-QK summary epoch drift")
    require(summary.get("held_out_test_evaluated") is False, "BOUNDED-QK summary scope drift")
    model = runtime["build_source_model"](
        payload,
        max_seq_len=int(contract["dataset"]["max_sequence_length"]),
    ).requires_grad_(False).eval().to(device)
    state = runtime["canonical_state_dict_sha256"](model.state_dict())
    require(state == spec["checkpoint_state_sha256"], "BOUNDED-QK strict restore drift")
    return model, {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_state_sha256": state,
        "summary_path": str(summary_path.resolve()),
        "summary_file_sha256": sha256_file(summary_path),
        "backbone": spec["backbone"],
        "best_epoch": spec["best_epoch"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    require(int(args.batch_size) == int(contract["execution"]["inference_batch_size"]), "Batch-size drift")
    source_audit = verify_frozen_source(args.frozen_source_root, contract)
    runtime = import_frozen_runtime(args.frozen_source_root)
    dataset_spec = contract["dataset"]
    require(sha256_file(args.data) == dataset_spec["data_sha256"], "Dataset digest drift")
    require(sha256_file(args.split_manifest) == dataset_spec["split_manifest_sha256"], "Split manifest drift")
    b_cache, b_audit = load_and_verify_b_cache(args.b_cache, args.b_result, contract)

    device = torch.device(args.device)
    cuda_index: int | None = None
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        cuda_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        device = torch.device("cuda", cuda_index)
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    frame = load_train_only_frame(args.data, runtime)
    require(frame.height == dataset_spec["expected_train_rows"], "Train row count drift")
    require(frame["oper_part_no"].n_unique() == dataset_spec["expected_train_series"], "Train series count drift")
    target_quantity, population = runtime["exact_target_population"](
        frame,
        target_split="train",
        lookback_weeks=dataset_spec["lookback_weeks"],
        max_seq_len=dataset_spec["max_sequence_length"],
    )
    for field, expected in {
        "target_count": dataset_spec["expected_train_targets"],
        "target_identity_sha256": dataset_spec["expected_train_identity_sha256"],
        "target_quantity_sha256": dataset_spec["expected_train_quantity_sha256"],
    }.items():
        require(population.get(field) == expected, f"Canonical train population drift: {field}")
    require(np.array_equal(target_quantity, b_cache["quantity"]), "Frozen-B target values drift")

    loader = runtime["make_loader"](
        frame,
        target_split="train",
        batch_size=int(args.batch_size),
        lookback_weeks=dataset_spec["lookback_weeks"],
        max_seq_len=dataset_spec["max_sequence_length"],
        shuffle=False,
        generator=None,
    )
    dataset = loader.dataset
    count = dataset_spec["expected_train_targets"]
    require(len(dataset) == count, "Train loader target count drift")
    require(len(dataset.parts) == dataset_spec["expected_train_series"], "Dataset part count drift")
    require(np.unique(b_cache["series_index"]).size == dataset_spec["expected_train_target_series"], "Target-series count drift")
    expected_parts = np.fromiter((part for part, _ in dataset.index), dtype=np.int64, count=count)
    expected_context = np.fromiter((end for _, end in dataset.index), dtype=np.int64, count=count)
    require(np.array_equal(expected_parts, b_cache["series_index"]), "Frozen-B series order drift")
    require(np.array_equal(expected_context, b_cache["context_end"]), "Frozen-B context order drift")

    model, checkpoint_audit = restore_bounded_model(
        checkpoint_path=args.bounded_checkpoint,
        summary_path=args.bounded_summary,
        contract=contract,
        runtime=runtime,
        device=device,
    )
    state_before = runtime["canonical_state_dict_sha256"](model.state_dict())
    prediction = np.empty(count, dtype=np.float64)
    time_nll = np.empty(count, dtype=np.float64)
    offset = 0
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (_, dts, mask, parts, quantities) in enumerate(loader):
            require(quantities is not None, "Raw quantities missing")
            batch_count = int(dts.size(0))
            end = offset + batch_count
            require(np.array_equal(parts.numpy().astype(np.int64), b_cache["series_index"][offset:end]), "Batch series alignment drift")
            output = runtime["target_outputs"](
                model,
                dts.to(device, non_blocking=device.type == "cuda"),
                mask.to(device, non_blocking=device.type == "cuda"),
                quantities.to(device, non_blocking=device.type == "cuda"),
                lambda_log_qty=1.0,
            )
            true_qty = output["true_qty"].detach().cpu().numpy().astype(np.float64)
            history = output["history_length"].detach().cpu().numpy().astype(np.int64)
            require(np.array_equal(true_qty, b_cache["quantity"][offset:end]), "Batch target alignment drift")
            require(np.array_equal(history, b_cache["history_length"][offset:end]), "Batch history alignment drift")
            prediction[offset:end] = output["pred_qty"].detach().cpu().numpy().astype(np.float64)
            time_nll[offset:end] = output["time_loss"].detach().cpu().numpy().astype(np.float64)
            offset = end
            if batch_index % 250 == 0 or offset == count:
                print(json.dumps({
                    "stage": "bounded_qk_train_inference",
                    "completed_targets": offset,
                    "total_targets": count,
                    "elapsed_seconds": time.perf_counter() - inference_started,
                }, sort_keys=True), flush=True)
    require(offset == count, "Incomplete BOUNDED-QK train inference")
    require(bool(np.isfinite(prediction).all()) and bool((prediction >= 0).all()), "Invalid BOUNDED-QK predictions")
    require(bool(np.isfinite(time_nll).all()), "Invalid BOUNDED-QK time values")
    state_after = runtime["canonical_state_dict_sha256"](model.state_dict())
    require(state_after == state_before, "BOUNDED-QK state changed during inference")
    require(all(parameter.grad is None for parameter in model.parameters()), "BOUNDED-QK accumulated gradients")

    arrays = {
        "prediction": prediction,
        "quantity": b_cache["quantity"].copy(),
        "time_nll": time_nll,
        "history_length": b_cache["history_length"].copy(),
        "series_index": b_cache["series_index"].copy(),
        "target_index": b_cache["target_index"].copy(),
        "context_end": b_cache["context_end"].copy(),
        "series_parts": np.asarray([str(part) for part in dataset.parts], dtype=np.str_),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "bounded_qk_train_predictions.npz"
    cache_sha = write_npz(cache_path, arrays)
    array_manifest = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": output_array_sha256(value, label=name),
        }
        for name, value in sorted(arrays.items())
    }
    audit = {
        "schema": "hard_lmm_bounded_qk_train_condition_run_v1",
        "status": "success",
        "created_at_utc": utc_now(),
        "contract": {
            "id": CONTRACT_ID,
            "path": str(args.contract.resolve()),
            "sha256": sha256_file(args.contract),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "scope": contract["scope"],
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "calibration_fit_performed": False,
        "validation_targets_evaluated": False,
        "held_out_test_evaluated": False,
        "data": {
            "path": str(args.data.resolve()),
            "sha256": sha256_file(args.data),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "materialized_splits": ["train"],
            "train_rows": frame.height,
            "train_series": frame["oper_part_no"].n_unique(),
            "target_population": population,
        },
        "frozen_source": source_audit,
        "frozen_b_reuse": b_audit,
        "bounded_qk_checkpoint": checkpoint_audit,
        "state_identity": {
            "before": state_before,
            "after": state_after,
            "unchanged": True,
            "all_gradients_absent": True,
        },
        "bounded_qk_predictions": {
            "path": str(cache_path.resolve()),
            "sha256": cache_sha,
            "target_count": count,
            "array_manifest": array_manifest,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "inference_batch_size": int(args.batch_size),
            "elapsed_seconds": time.perf_counter() - started,
            "inference_elapsed_seconds": time.perf_counter() - inference_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_index)) if device.type == "cuda" else 0,
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
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-result", type=Path, required=True)
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
    print(json.dumps({"status": completed["status"], "bounded_qk_predictions": completed["bounded_qk_predictions"]}, sort_keys=True))
