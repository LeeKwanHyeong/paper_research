#!/usr/bin/env python3
"""Fit and audit a train-only raw-quantity affine calibration.

The source TPP is immutable.  This program extracts its next-quantity
predictions from the admitted train and validation populations, fits
``max(0, a * q_hat + b)`` on train predictions only, and reports both a
series-disjoint two-fold train audit and a validation-only screen.  It never
loads held-out targets and it never updates a source-model tensor.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

from data_loader.event_seq_data_module import (
    RMTPPWeekLookbackDataset,
    collate_week_lookback,
)
from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    SharedTimeCountModel,
)
from paper.scripts.count_aware_tpp_backbone.core import (
    prepare_count_frame,
    target_outputs,
)
from paper.scripts.frozen_raw_affine_calibration import (
    FOLD_SALT,
    apply_affine_calibration,
    apply_quantity_only_bundle,
    assign_series_folds,
    compute_quantity_metrics,
    compute_reporting_metrics,
    fit_positive_affine_ols,
    series_clustered_paired_bootstrap_ci,
    two_fold_oof_audit,
)
from paper.scripts.run_intermittent_log_backbone_control import (
    HISTORY_BOUNDARIES,
    HISTORY_STRATA,
)
from paper.scripts.run_hard_lmm_time_head_refit import (
    exact_target_population_contract,
    require,
    sha256_file,
    validate_target_population,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "frozen_raw_affine_calibration_v1"
DEFAULT_CONTRACT = PROJECT_ROOT / "paper/contracts/frozen_raw_affine_calibration_v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_source_manifest(path: Path, *, source_revision: str) -> dict[str, Any]:
    require(path.is_file(), "Committed source manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("schema") == "frozen_raw_affine_source_manifest_v1",
        "Source manifest schema drift",
    )
    require(payload.get("source_revision") == source_revision, "Source manifest revision drift")
    files = payload.get("files")
    require(isinstance(files, Mapping) and bool(files), "Source manifest files are missing")
    require(payload.get("file_count") == len(files), "Source manifest count drift")
    required = {
        "paper/contracts/frozen_raw_affine_calibration_v1.json",
        "paper/scripts/frozen_raw_affine_calibration.py",
        "paper/scripts/build_frozen_raw_affine_source_manifest.py",
        "paper/scripts/control_frozen_raw_affine_5080.py",
        "paper/scripts/run_frozen_raw_affine_calibration.py",
        "paper/scripts/summarize_frozen_raw_affine_calibration.py",
        "paper/scripts/count_aware_tpp_backbone/core.py",
        "paper/scripts/run_hard_lmm_time_head_refit.py",
        "paper/scripts/run_intermittent_log_backbone_control.py",
        "models/TPPs/CountAwareTPP.py",
        "models/TPPs/CountAwareFactory.py",
        "data_loader/event_seq_data_module.py",
        "simple_lab_test/search/common/runner.py",
    }
    require(required <= set(files), "Source manifest omits required execution files")
    root = PROJECT_ROOT.resolve()
    runtime_empty_directories = payload.get("runtime_empty_directories")
    require(
        runtime_empty_directories == ["sample_data"],
        "Runtime empty-directory contract drift",
    )
    for relative in runtime_empty_directories:
        directory = (root / relative).resolve()
        require(directory.is_relative_to(root), f"Runtime directory escapes snapshot: {relative}")
        require(directory.is_dir(), f"Runtime directory is missing: {relative}")
        require(not any(directory.iterdir()), f"Runtime directory must be empty: {relative}")
    for relative, expected in files.items():
        candidate = (root / str(relative)).resolve()
        require(candidate.is_relative_to(root), f"Source path escapes snapshot: {relative}")
        require(candidate.is_file(), f"Source file is missing: {relative}")
        require(sha256_file(candidate) == expected, f"Source file digest drift: {relative}")
    return {
        "file": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "source_revision": source_revision,
        "git_tree": payload.get("git_tree"),
        "file_count": len(files),
        "all_file_hashes_verified": True,
        "runtime_empty_directories_verified": runtime_empty_directories,
        "canonical_contract_sha256": files[
            "paper/contracts/frozen_raw_affine_calibration_v1.json"
        ],
    }


def array_sha256(value: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"frozen_raw_affine_array_v1\0")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _hidden_dim(encoder: Mapping[str, Any]) -> int:
    value = encoder.get("d_model", encoder.get("hidden_dim"))
    require(type(value) is int and int(value) > 0, "Invalid source hidden dimension")
    return int(value)


def _legacy_time_contract(
    encoder: Mapping[str, Any], interface: Mapping[str, Any]
) -> dict[str, Any]:
    value = encoder.get("time_head")
    if not isinstance(value, Mapping):
        value = interface.get("time_head")
    if not isinstance(value, Mapping):
        value = {}
    return {
        "mode": str(value.get("mode", TIME_HEAD_MODE_LEGACY_CLAMPED)),
        "time_scale": float(value.get("time_scale", 3.0)),
        "time_w_max": float(value.get("time_w_max", 10.0 / 3.0)),
        "time_intercept_limit": float(value.get("time_intercept_limit", 30.0)),
        "time_initial_intercept": float(value.get("time_initial_intercept", 0.0)),
        "time_wd_safety_limit": float(value.get("time_wd_safety_limit", 40.0)),
        "time_sigma_floor": float(value.get("time_sigma_floor", 1e-3)),
    }


def validate_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong calibration contract")
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope is missing")
    require(scope.get("input_splits") == ["train", "validation"], "Split scope drift")
    require(scope.get("optimization_split") == "train", "Fit split drift")
    require(scope.get("evaluation_scope") == "validation_only", "Evaluation scope drift")
    require(scope.get("held_out_test") is False, "Held-out access is enabled")
    require(scope.get("seed") == 42, "Seed drift")
    calibration = contract.get("calibration")
    require(isinstance(calibration, Mapping), "Calibration definition is missing")
    require(calibration.get("formula") == "max(0, a * q_base + b)", "Formula drift")
    require(calibration.get("slope_constraint") == "a > 0", "Slope constraint drift")
    require(calibration.get("fit") == "positive-slope ordinary least squares", "Fit rule drift")
    require(calibration.get("fold_salt") == FOLD_SALT, "Fold salt drift")
    require(calibration.get("minimum_slope") == 1e-12, "Minimum slope drift")
    require(calibration.get("numeric_dtype") == "float64", "Calibration dtype drift")
    require(calibration.get("manual_dataset_hyperparameters") is False, "Dataset tuning enabled")
    require(calibration.get("source_model_frozen") is True, "Source model is not frozen")
    runtime = contract.get("runtime")
    require(isinstance(runtime, Mapping), "Runtime contract is missing")
    require(runtime.get("execution_server") == "5080", "Only 5080 execution is allowed")
    require(runtime.get("other_gpu_hosts_allowed") is False, "Another GPU host is enabled")
    require(runtime.get("inference_batch_size") == 128, "Inference batch-size drift")
    require(runtime.get("smoke_targets_per_fold") == 8, "Smoke cohort drift")
    require(runtime.get("paired_bootstrap_seed") == 20260907, "Bootstrap seed drift")
    require(runtime.get("paired_bootstrap_replicates") == 500, "Bootstrap replicate drift")
    require(runtime.get("paired_bootstrap_draw_chunk_size") == 8192, "Bootstrap chunk drift")
    require(
        runtime.get("deterministic_environment")
        == {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "Deterministic environment contract drift",
    )
    acceptance = contract.get("acceptance")
    require(isinstance(acceptance, Mapping), "Acceptance contract is missing")
    require(
        acceptance.get("quantity_boundary_estimator")
        == "numpy.quantile(method='nearest') over canonical train next-event targets",
        "Quantity-boundary estimator drift",
    )
    for name, expected in {
        "minimum_pooled_oof_rmse_improvement_fraction": 0.01,
        "maximum_overall_mae_regression_fraction": 0.02,
        "maximum_body_mae_regression_fraction": 0.02,
        "maximum_tail_mae_regression_fraction": 0.02,
    }.items():
        require(acceptance.get(name) == expected, f"Acceptance threshold drift: {name}")
    rows: dict[str, dict[str, Any]] = {}
    for row in contract.get("datasets", []):
        dataset = str(row.get("dataset"))
        require(dataset not in rows, f"Duplicate dataset row: {dataset}")
        require(isinstance(row.get("sources"), Mapping), f"Missing sources: {dataset}")
        rows[dataset] = dict(row)
    require(
        set(rows) == {"intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket"},
        "Dataset scope drift",
    )
    for dataset, row in rows.items():
        expected_roles = {"B", "rmtpp", "thp"} if dataset == "insta_market_basket" else {"B"}
        require(set(row["sources"]) == expected_roles, f"Source-role scope drift: {dataset}")
        boundaries = row.get("train_quantity_boundaries")
        require(
            isinstance(boundaries, Mapping)
            and set(boundaries) == {"p50", "p90", "p95", "p99"},
            f"Train quantity boundaries missing: {dataset}",
        )
        for role, source in row["sources"].items():
            require(source.get("seed") == 42, f"Source seed drift: {dataset}/{role}")
            require(
                source.get("quantity_variant") == LOG_MSE_VARIANT,
                f"Source quantity interface drift: {dataset}/{role}",
            )
            require(
                source.get("checkpoint_selection")
                == "best_validation_raw_quantity_rmse",
                f"Source selector drift: {dataset}/{role}",
            )
            for digest_name in ("checkpoint_file_sha256", "checkpoint_state_sha256"):
                digest = str(source.get(digest_name, ""))
                require(
                    len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest),
                    f"Source digest drift: {dataset}/{role}/{digest_name}",
                )
    return rows


def validate_source_checkpoint(
    payload: Mapping[str, Any], *, source_spec: Mapping[str, Any]
) -> None:
    require(payload.get("backbone") == source_spec.get("backbone"), "Backbone drift")
    require(payload.get("variant") == source_spec.get("quantity_variant"), "Variant drift")
    require(payload.get("variant") == LOG_MSE_VARIANT, "Quantity interface drift")
    require(payload.get("seed") == source_spec.get("seed"), "Seed drift")
    require(
        payload.get("selection") == source_spec.get("checkpoint_selection"),
        "Checkpoint selector drift",
    )
    require(
        payload.get("source_revision") == source_spec.get("training_source_revision"),
        "Training revision drift",
    )
    require(
        payload.get("source_revision_history") == source_spec.get("training_source_revision_history"),
        "Training revision history drift",
    )
    require(payload.get("evaluation_scope") == "validation_only", "Source scope drift")
    require(payload.get("held_out_test_evaluated") is False, "Source used held-out test")
    state = payload.get("model_state_dict")
    require(isinstance(state, Mapping) and bool(state), "Source model state is missing")
    observed_state = canonical_state_dict_sha256(state)
    require(observed_state == payload.get("model_state_sha256"), "Payload state digest drift")
    require(observed_state == source_spec.get("checkpoint_state_sha256"), "Pinned state digest drift")
    for name, tensor in state.items():
        require(torch.is_tensor(tensor), f"Non-tensor source state: {name}")
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all()), f"Non-finite source state: {name}")
    interface = payload.get("interface_meta")
    require(isinstance(interface, Mapping), "Interface metadata is missing")
    require(interface.get("mode") == "mark_free_count_aware_log_regression", "Interface mode drift")
    require(interface.get("target_quantity_masked_from_history") is True, "Target leakage contract drift")


def build_source_model(
    payload: Mapping[str, Any], *, max_seq_len: int
) -> SharedTimeCountModel:
    encoder = payload["encoder_config"]
    interface = payload["interface_meta"]
    time_contract = _legacy_time_contract(encoder, interface)
    require(time_contract["mode"] == TIME_HEAD_MODE_LEGACY_CLAMPED, "Time-head mode drift")
    model, _ = build_count_aware_model(
        str(payload["backbone"]),
        hidden_dim=_hidden_dim(encoder),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface["train_target_std"]),
        max_seq_len=int(encoder.get("max_len", max_seq_len)),
        quantity_variant=str(payload["variant"]),
        lambda_tail=0.0,
        time_head_mode=time_contract["mode"],
        time_scale=time_contract["time_scale"],
        time_w_max=time_contract["time_w_max"],
        time_intercept_limit=time_contract["time_intercept_limit"],
        time_initial_intercept=time_contract["time_initial_intercept"],
        time_wd_safety_limit=time_contract["time_wd_safety_limit"],
        time_sigma_floor=time_contract["time_sigma_floor"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    require(
        canonical_state_dict_sha256(model.state_dict()) == payload["model_state_sha256"],
        "Strict source replay digest drift",
    )
    return model.requires_grad_(False).eval()


def gpu_preflight(minimum_free_mib: int) -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout.strip()
    rows = query.splitlines()
    require(len(rows) == 1, "Expected exactly one GPU on the 5080 host")
    name, free = (field.strip() for field in rows[0].split(","))
    require("RTX 5080" in name, f"Wrong GPU: {name}")
    require(int(free) >= int(minimum_free_mib), f"Insufficient free VRAM: {free} MiB")
    require(not compute, f"Another CUDA compute process is active: {compute}")
    return {"gpu_name": name, "free_memory_mib": int(free), "compute_processes_before": []}


def deterministic_environment_audit(expected: Mapping[str, str]) -> dict[str, str]:
    observed = {name: os.environ.get(name, "") for name in expected}
    require(observed == dict(expected), f"Deterministic environment drift: {observed}")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return observed


def reset_cuda_peak_memory(device: torch.device) -> int:
    """Reset peak accounting with an integer index for 5080 Runtime compatibility."""
    require(device.type == "cuda", "Peak CUDA reset requires a CUDA device")
    torch.cuda.set_device(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device_index)
    return int(device_index)


def load_admitted_frame(path: Path) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split").is_in(["train", "validation"]))
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(
        set(frame["chronological_split"].unique().to_list()) == {"train", "validation"},
        "Admitted frame must contain exactly train and validation rows",
    )
    return prepare_count_frame(frame)


def load_train_only_frame(path: Path) -> pl.DataFrame:
    """Materialize train rows only; validation is excluded by scan predicate."""
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(
        set(frame["chronological_split"].unique().to_list()) == {"train"},
        "Train-only frame is empty or contains another split",
    )
    return prepare_count_frame(frame)


def make_dataset(
    frame: pl.DataFrame, *, split: str, lookback: int, max_seq_len: int
) -> RMTPPWeekLookbackDataset:
    require(split in {"train", "validation"}, "Held-out target split is prohibited")
    return RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        val_ratio=0.2,
        mode="all",
        split_col="chronological_split",
        target_splits={split},
    )


def compute_train_quantity_boundaries(quantity: np.ndarray) -> dict[str, float]:
    values = np.asarray(quantity, dtype=np.float64)
    require(
        values.ndim == 1 and values.size > 0,
        "Train quantities must be a nonempty one-dimensional array",
    )
    require(bool(np.isfinite(values).all()), "Train quantities contain non-finite values")
    return {
        "p50": float(np.quantile(values, 0.50, method="nearest")),
        "p90": float(np.quantile(values, 0.90, method="nearest")),
        "p95": float(np.quantile(values, 0.95, method="nearest")),
        "p99": float(np.quantile(values, 0.99, method="nearest")),
    }


def smoke_indices(dataset: RMTPPWeekLookbackDataset, per_fold: int) -> np.ndarray:
    part_indices = np.fromiter((part for part, _ in dataset.index), dtype=np.int64)
    folds = assign_series_folds(part_indices, salt=FOLD_SALT)
    selected = [np.flatnonzero(folds == fold)[:per_fold] for fold in (0, 1)]
    require(all(len(values) > 0 for values in selected), "Smoke cohort lacks a series fold")
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


@torch.no_grad()
def extract_predictions(
    *,
    model: SharedTimeCountModel,
    dataset: RMTPPWeekLookbackDataset,
    device: torch.device,
    batch_size: int,
    smoke_targets_per_fold: int | None,
    progress_path: Path,
) -> dict[str, np.ndarray]:
    if smoke_targets_per_fold is None:
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = smoke_indices(dataset, smoke_targets_per_fold)
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_week_lookback,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    values: dict[str, list[np.ndarray]] = {
        "prediction": [],
        "quantity": [],
        "time_nll": [],
        "history_length": [],
        "series_index": [],
    }
    offset = 0
    started = time.monotonic()
    for batch_number, (_, dts, mask, parts, quantities) in enumerate(loader):
        require(quantities is not None, "Raw quantities are required")
        outputs = target_outputs(
            model,
            dts.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
            quantities.to(device, non_blocking=True),
            lambda_log_qty=1.0,
        )
        arrays = {
            "prediction": outputs["pred_qty"].detach().cpu().numpy().astype(np.float64),
            "quantity": outputs["true_qty"].detach().cpu().numpy().astype(np.float64),
            "time_nll": outputs["time_loss"].detach().cpu().numpy().astype(np.float64),
            "history_length": outputs["history_length"].detach().cpu().numpy().astype(np.int64),
            "series_index": parts.numpy().astype(np.int64),
        }
        count = len(arrays["quantity"])
        require(all(len(value) == count for value in arrays.values()), "Batch alignment drift")
        require(all(np.isfinite(value).all() for value in arrays.values()), "Non-finite extraction")
        for name, value in arrays.items():
            values[name].append(value)
        offset += count
        if batch_number % 250 == 0 or offset == len(indices):
            save_json(
                progress_path,
                {
                    "status": "extracting",
                    "completed_targets": offset,
                    "selected_targets": len(indices),
                    "available_targets": len(dataset),
                    "elapsed_seconds": time.monotonic() - started,
                    "updated_at": utc_now(),
                },
            )
    require(offset == len(indices), "Extraction target count drift")
    result = {name: np.concatenate(chunks) for name, chunks in values.items()}
    result["target_index"] = indices
    result["context_end"] = np.asarray(
        [dataset.index[int(index)][1] for index in indices], dtype=np.int64
    )
    require(len({len(value) for value in result.values()}) == 1, "Cache alignment drift")
    return result


def cache_manifest(cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "target_count": len(cache["quantity"]),
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_sha256(value, label=name),
            }
            for name, value in sorted(cache.items())
        },
    }


def write_cache(path: Path, cache: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **cache)
    temporary.replace(path)
    return sha256_file(path)


def source_reference_differences(
    metrics: Mapping[str, float], source_spec: Mapping[str, Any], *, smoke: bool
) -> dict[str, float] | dict[str, str]:
    if smoke:
        return {"status": "partial_smoke_only"}
    reference = source_spec.get("validation_reference_metrics", {})
    differences: dict[str, float] = {}
    key_map = {
        "raw_rmse": "rmse",
        "overall_mae": "mae",
        "log1p_mse": "log1p_mse",
        "time_nll": "time_nll",
    }
    for reference_name, metric_name in key_map.items():
        if reference_name not in reference:
            continue
        difference = abs(float(metrics[metric_name]) - float(reference[reference_name]))
        require(difference <= 1e-5, f"Frozen validation replay drift: {reference_name}={difference}")
        differences[reference_name] = difference
    return differences


def relative_change(candidate: float, baseline: float) -> float:
    require(baseline != 0.0, "Relative change baseline is zero")
    return (candidate - baseline) / abs(baseline)


def optional_relative_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return relative_change(float(candidate), float(baseline))


def validate_prerequisite_decision(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.phase == "train" and args.model_role == "B":
        require(args.prerequisite_decision is None, "B train is the first gated phase")
        return None
    require(args.prerequisite_decision is not None, "A prerequisite decision is required")
    payload = json.loads(args.prerequisite_decision.read_text(encoding="utf-8"))
    expected = (
        ("B_three_dataset_train_gate", "accepted_for_B_validation")
        if args.model_role == "B"
        else (
            "B_three_dataset_validation_gate",
            "accepted_for_instacart_fairness",
        )
    )
    require(payload.get("status") == "success", "Prerequisite summary failed")
    require(payload.get("accepted") is True, "Prerequisite gate did not pass")
    require(payload.get("stage") == expected[0], "Prerequisite stage drift")
    require(payload.get("decision") == expected[1], "Prerequisite decision drift")
    require(
        payload.get("contract_sha256") == sha256_file(args.contract),
        "Prerequisite contract drift",
    )
    require(payload.get("source_revision") == args.source_revision, "Prerequisite source drift")
    require(
        payload.get("source_manifest_file_sha256") == sha256_file(args.source_manifest),
        "Prerequisite source manifest drift",
    )
    require(payload.get("execution_server") == "5080", "Prerequisite host drift")
    require(payload.get("held_out_test_evaluated") is False, "Prerequisite used held-out test")
    identities = payload.get("result_identities")
    require(isinstance(identities, list) and len(identities) == 3, "Prerequisite evidence scope drift")
    require(
        {(row.get("dataset"), row.get("model_role")) for row in identities}
        == {
            ("intermittent_frozen_5000", "B"),
            ("yellow_trip_hourly", "B"),
            ("insta_market_basket", "B"),
        },
        "Prerequisite B evidence identities drift",
    )
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected_checkpoints = {
        row["dataset"]: row["sources"]["B"]["checkpoint_file_sha256"]
        for row in contract["datasets"]
    }
    for identity in identities:
        result_path = Path(str(identity.get("result_file", "")))
        require(result_path.is_file(), "Prerequisite result file is missing")
        require(
            sha256_file(result_path) == identity.get("result_file_sha256"),
            "Prerequisite result digest drift",
        )
        require(
            identity.get("checkpoint_file_sha256")
            == expected_checkpoints[identity["dataset"]],
            "Prerequisite checkpoint identity drift",
        )
    if args.model_role == "B":
        matching_identity = next(
            identity
            for identity in identities
            if identity["dataset"] == args.dataset and identity["model_role"] == "B"
        )
        require(args.train_result is not None, "B validation train result is missing")
        require(
            sha256_file(args.train_result) == matching_identity["result_file_sha256"],
            "B validation train result is outside the admitted train gate",
        )
    return {
        "file": str(args.prerequisite_decision.resolve()),
        "file_sha256": sha256_file(args.prerequisite_decision),
        "stage": payload["stage"],
        "decision": payload["decision"],
    }


def prepare_execution(args: argparse.Namespace) -> dict[str, Any]:
    require(args.output_dir.resolve() != PROJECT_ROOT.resolve(), "Output must be an artifact directory")
    require(not args.output_dir.exists(), "Refusing to overwrite an existing artifact directory")
    args.output_dir.mkdir(parents=True)
    status_path = args.output_dir / "status.json"
    save_json(status_path, {"status": "starting", "started_at": utc_now()})
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    datasets = validate_contract(contract)
    require(args.dataset in datasets, "Dataset is outside the contract")
    dataset_spec = datasets[args.dataset]
    require(args.model_role in dataset_spec["sources"], "Model role is outside the dataset contract")
    source_spec = dataset_spec["sources"][args.model_role]
    require(len(args.source_revision) == 40, "Committed source revision is required")
    source_manifest_audit = verify_source_manifest(
        args.source_manifest, source_revision=args.source_revision
    )
    canonical_contract = PROJECT_ROOT / "paper/contracts/frozen_raw_affine_calibration_v1.json"
    require(args.contract.resolve() == canonical_contract.resolve(), "Noncanonical contract path")
    require(
        sha256_file(args.contract) == source_manifest_audit["canonical_contract_sha256"],
        "Execution contract differs from committed source manifest",
    )
    data_sha256 = sha256_file(args.data)
    split_manifest_sha256 = sha256_file(args.split_manifest)
    require(data_sha256 == dataset_spec["data_sha256"], "Dataset digest mismatch")
    require(
        split_manifest_sha256 == dataset_spec["split_manifest_sha256"],
        "Split-manifest digest mismatch",
    )
    checkpoint_file_sha = sha256_file(args.checkpoint)
    require(
        checkpoint_file_sha == source_spec["checkpoint_file_sha256"],
        "Checkpoint file digest mismatch",
    )
    runtime = contract["runtime"]
    device = torch.device(args.device)
    if device.type == "cuda":
        environment = deterministic_environment_audit(
            runtime["deterministic_environment"]
        )
        runtime_audit = gpu_preflight(int(runtime["minimum_free_vram_mib"]))
        require(torch.cuda.is_available(), "CUDA is unavailable")
        runtime_audit["cuda_device_index"] = reset_cuda_peak_memory(device)
    else:
        require(args.smoke, "CPU execution is allowed only for smoke tests")
        runtime_audit = {"gpu_name": None, "cpu_smoke": True}
        environment = {
            name: os.environ.get(name) for name in runtime["deterministic_environment"]
        }
    runtime_audit["deterministic_environment"] = environment

    payload = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    validate_source_checkpoint(payload, source_spec=source_spec)
    model = build_source_model(
        payload, max_seq_len=int(dataset_spec["max_sequence_length"])
    ).to(device)
    state_before = canonical_state_dict_sha256(model.state_dict())
    return {
        "contract": contract,
        "dataset_spec": dataset_spec,
        "source_spec": source_spec,
        "runtime": runtime,
        "runtime_audit": runtime_audit,
        "payload": payload,
        "model": model,
        "frame": None,
        "device": device,
        "state_before": state_before,
        "checkpoint_file_sha": checkpoint_file_sha,
        "source_manifest_audit": source_manifest_audit,
        "data_sha256": data_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "status_path": status_path,
    }


def extract_split(
    context: Mapping[str, Any], args: argparse.Namespace, *, split: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    dataset_spec = context["dataset_spec"]
    require(context["frame"] is not None, "Split frame was not admitted")
    population = exact_target_population_contract(
        context["frame"],
        target_split=split,
        lookback=int(dataset_spec["lookback"]),
        max_seq_len=int(dataset_spec["max_sequence_length"]),
    )
    validate_target_population(population, dataset_spec=dataset_spec)
    dataset = make_dataset(
        context["frame"],
        split=split,
        lookback=int(dataset_spec["lookback"]),
        max_seq_len=int(dataset_spec["max_sequence_length"]),
    )
    if split == "train":
        train_quantity = np.fromiter(
            (
                float(dataset.val_lists[part_index][context_end + 1])
                for part_index, context_end in dataset.index
            ),
            dtype=np.float64,
            count=len(dataset.index),
        )
        observed_boundaries = compute_train_quantity_boundaries(train_quantity)
        require(
            observed_boundaries == dataset_spec["train_quantity_boundaries"],
            f"Train quantity boundary drift: {observed_boundaries}",
        )
        population = dict(population)
        population["train_quantity_boundaries"] = observed_boundaries
    save_json(
        context["status_path"],
        {
            "status": f"extracting_{split}",
            "phase": args.phase,
            "dataset": args.dataset,
            "model_role": args.model_role,
            "updated_at": utc_now(),
        },
    )
    cache = extract_predictions(
        model=context["model"],
        dataset=dataset,
        device=context["device"],
        batch_size=int(context["runtime"]["inference_batch_size"]),
        smoke_targets_per_fold=(
            int(context["runtime"]["smoke_targets_per_fold"]) if args.smoke else None
        ),
        progress_path=context["status_path"],
    )
    cache_path = args.output_dir / f"{split}_predictions.npz"
    audit = {
        **cache_manifest(cache),
        "file": cache_path.name,
        "file_sha256": write_cache(cache_path, cache),
        "full_population_target_count": int(population["target_count"]),
    }
    return cache, audit, population


def audit_frozen_state(context: Mapping[str, Any]) -> tuple[str, int]:
    state_after = canonical_state_dict_sha256(context["model"].state_dict())
    require(
        context["state_before"] == state_after == context["payload"]["model_state_sha256"],
        "Frozen source state changed",
    )
    require(
        all(parameter.grad is None for parameter in context["model"].parameters()),
        "Frozen source accumulated gradients",
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(context["device"]))
        if context["device"].type == "cuda"
        else 0
    )
    return state_after, peak_memory


def analyze_train(
    train: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    dataset_spec: Mapping[str, Any] | None = None,
    *,
    partial_smoke: bool = False,
) -> dict[str, Any]:
    calibration = contract["calibration"]
    observed_boundaries = compute_train_quantity_boundaries(train["quantity"])
    if dataset_spec is not None and not partial_smoke:
        require(
            observed_boundaries == dataset_spec["train_quantity_boundaries"],
            f"Train quantity boundary drift: {observed_boundaries}",
        )
    train_boundaries = (
        dict(dataset_spec["train_quantity_boundaries"])
        if dataset_spec is not None and partial_smoke
        else observed_boundaries
    )
    quantity_boundaries = [train_boundaries[name] for name in ("p50", "p90", "p95", "p99")]
    history_labels = [str(row["stratum"]) for row in HISTORY_STRATA]
    fold_audit, fold_arrays = two_fold_oof_audit(
        train["prediction"],
        train["quantity"],
        train["series_index"],
        fold_salt=str(calibration["fold_salt"]),
        body_threshold=train_boundaries["p95"],
        extreme_tail_threshold=train_boundaries["p99"],
        quantity_boundaries=quantity_boundaries,
        history_length=train["history_length"],
        history_boundaries=HISTORY_BOUNDARIES,
        history_labels=history_labels,
    )
    acceptance = contract["acceptance"]
    pooled_oof_improvement = float(fold_audit["overall_change"]["rmse_relative_improvement"])
    fold_pass = bool(fold_audit["both_folds_raw_mse_improve"])
    body_regression = optional_relative_change(
        fold_audit["calibrated"]["body_le_p95"]["mae"],
        fold_audit["baseline"]["body_le_p95"]["mae"],
    )
    tail_regression = optional_relative_change(
        fold_audit["calibrated"]["gt_p99"]["mae"],
        fold_audit["baseline"]["gt_p99"]["mae"],
    )
    train_gate = False if partial_smoke else (
        fold_pass
        and pooled_oof_improvement
        >= float(acceptance["minimum_pooled_oof_rmse_improvement_fraction"])
        and body_regression is not None
        and body_regression <= float(acceptance["maximum_body_mae_regression_fraction"])
        and tail_regression is not None
        and tail_regression <= float(acceptance["maximum_tail_mae_regression_fraction"])
    )
    parameters = (
        fit_positive_affine_ols(train["prediction"], train["quantity"])
        if train_gate or partial_smoke
        else None
    )
    return {
        "calibration_parameters": parameters,
        "quantity_boundaries": train_boundaries,
        "two_fold_oof": fold_audit,
        "two_fold_oof_array_hashes": {
            name: array_sha256(value, label=f"oof_{name}")
            for name, value in fold_arrays.items()
        },
        "relative_changes": {
            "pooled_oof_rmse_improvement": pooled_oof_improvement,
            "body_le_p95_mae_regression": body_regression,
            "tail_gt_p99_mae_regression": tail_regression,
        },
        "performance_gate_evaluated": not partial_smoke,
        "gate_passed": bool(train_gate),
    }


def persist_calibration_state(
    *,
    output_dir: Path,
    train: Mapping[str, np.ndarray],
    train_audit: Mapping[str, Any],
    population: Mapping[str, Any],
    contract_sha256: str,
    source_revision: str,
    dataset: str,
    model_role: str,
    checkpoint_file_sha256: str,
    checkpoint_state_sha256: str,
    data_sha256: str,
    split_manifest_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    activated = bool(train_audit["gate_passed"]) or (
        train_audit.get("performance_gate_evaluated") is False
        and train_audit.get("calibration_parameters") is not None
    )
    activation_reason = (
        "train_gate"
        if train_audit["gate_passed"]
        else "smoke_contract_exercise"
        if activated
        else "identity_fallback"
    )
    deployed = (
        dict(train_audit["calibration_parameters"])
        if activated
        else {"slope": 1.0, "intercept": 0.0}
    )
    prediction = apply_affine_calibration(train["prediction"], deployed)
    prediction_hash = array_sha256(prediction, label="restored_train_prediction")
    state = {
        "schema": "frozen_raw_affine_calibration_state_v1",
        "contract_sha256": contract_sha256,
        "source_revision": source_revision,
        "dataset": dataset,
        "model_role": model_role,
        "source_checkpoint_file_sha256": checkpoint_file_sha256,
        "source_checkpoint_state_sha256": checkpoint_state_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "data_sha256": data_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "train_target_identity_sha256": population["target_identity_sha256"],
        "train_target_quantity_sha256": population["target_quantity_sha256"],
        "train_target_count": population["target_count"],
        "train_gate_passed": bool(train_audit["gate_passed"]),
        "calibration_activated": activated,
        "activation_reason": activation_reason,
        "fitted_calibration_parameters": train_audit["calibration_parameters"],
        "deployed_calibration_parameters": deployed,
        "fold_array_sha256": train_audit["two_fold_oof_array_hashes"]["fold"],
        "restored_train_prediction_sha256": prediction_hash,
    }
    path = output_dir / "calibration_state.json"
    save_json(path, state)
    restored = json.loads(path.read_text(encoding="utf-8"))
    require(restored == state, "Calibration state JSON replay drift")
    replay = apply_affine_calibration(
        train["prediction"], restored["deployed_calibration_parameters"]
    )
    require(
        np.array_equal(replay, prediction)
        and array_sha256(replay, label="restored_train_prediction") == prediction_hash,
        "Calibration state prediction replay drift",
    )
    return {
        "file": path.name,
        "file_sha256": sha256_file(path),
        "prediction_replay_exact": True,
        "restored_train_prediction_sha256": prediction_hash,
    }


def validate_train_result(
    payload: Mapping[str, Any], context: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    require(payload.get("status") == "success", "Train phase did not succeed")
    require(payload.get("phase") == "train", "Expected a train-phase result")
    require(payload.get("dataset") == args.dataset, "Train-result dataset drift")
    require(payload.get("model_role") == args.model_role, "Train-result role drift")
    require(payload.get("source_revision") == args.source_revision, "Train-result source drift")
    require(
        payload.get("contract_sha256") == sha256_file(args.contract),
        "Train-result contract drift",
    )
    require(
        payload.get("source_checkpoint_file_sha256") == context["checkpoint_file_sha"],
        "Train-result checkpoint drift",
    )
    require(
        payload.get("source_manifest", {}).get("file_sha256")
        == context["source_manifest_audit"]["file_sha256"],
        "Train-result source manifest drift",
    )
    require(
        payload.get("input_digests", {}).get("data_sha256") == context["data_sha256"],
        "Train-result data digest drift",
    )
    require(
        payload.get("input_digests", {}).get("split_manifest_sha256")
        == context["split_manifest_sha256"],
        "Train-result split-manifest digest drift",
    )
    require(payload.get("source_model_state_unchanged") is True, "Train source changed")
    require(payload.get("source_gradients_absent") is True, "Train source gradients present")
    audit = payload.get("train_audit")
    require(isinstance(audit, Mapping), "Train audit is missing")
    if args.model_role == "B" and not args.smoke:
        require(audit.get("gate_passed") is True, "B train gate failed; validation is prohibited")
    state_spec = audit.get("calibration_state")
    require(isinstance(state_spec, Mapping), "Calibration state reference is missing")
    state_path = args.train_result.parent / str(state_spec["file"])
    require(state_path.is_file(), "Calibration state file is missing")
    require(sha256_file(state_path) == state_spec["file_sha256"], "Calibration state digest drift")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    require(state.get("schema") == "frozen_raw_affine_calibration_state_v1", "State schema drift")
    require(state.get("contract_sha256") == payload["contract_sha256"], "State contract drift")
    require(state.get("source_revision") == args.source_revision, "State source revision drift")
    require(state.get("dataset") == args.dataset, "State dataset drift")
    require(state.get("model_role") == args.model_role, "State role drift")
    require(
        state.get("source_checkpoint_file_sha256") == context["checkpoint_file_sha"],
        "State checkpoint drift",
    )
    require(
        state.get("source_checkpoint_state_sha256") == context["state_before"],
        "State model digest drift",
    )
    require(
        state.get("source_manifest_sha256")
        == context["source_manifest_audit"]["file_sha256"],
        "State source-manifest drift",
    )
    require(state.get("data_sha256") == context["data_sha256"], "State data digest drift")
    require(
        state.get("split_manifest_sha256") == context["split_manifest_sha256"],
        "State split-manifest digest drift",
    )
    train_population = payload.get("target_population", {}).get("train")
    require(isinstance(train_population, Mapping), "Train target population is missing")
    require(
        state.get("train_target_identity_sha256")
        == train_population.get("target_identity_sha256"),
        "State target identity drift",
    )
    require(
        state.get("train_target_quantity_sha256")
        == train_population.get("target_quantity_sha256"),
        "State target quantity drift",
    )
    require(
        state.get("train_target_count") == train_population.get("target_count"),
        "State target count drift",
    )
    require(
        state.get("fold_array_sha256")
        == audit.get("two_fold_oof_array_hashes", {}).get("fold"),
        "State fold digest drift",
    )
    require(
        state.get("fitted_calibration_parameters") == audit.get("calibration_parameters"),
        "State fitted parameters drift",
    )
    require(state.get("train_gate_passed") == audit.get("gate_passed"), "State gate drift")
    require(
        state.get("restored_train_prediction_sha256")
        == state_spec.get("restored_train_prediction_sha256"),
        "State prediction replay digest drift",
    )
    expected_activation = bool(audit.get("gate_passed")) or (
        audit.get("performance_gate_evaluated") is False
        and audit.get("calibration_parameters") is not None
    )
    require(
        state.get("calibration_activated") is expected_activation,
        "State activation drift",
    )
    expected_reason = (
        "train_gate"
        if audit.get("gate_passed") is True
        else "smoke_contract_exercise"
        if expected_activation
        else "identity_fallback"
    )
    require(state.get("activation_reason") == expected_reason, "State activation reason drift")
    expected_deployed = (
        audit.get("calibration_parameters")
        if expected_activation
        else {"slope": 1.0, "intercept": 0.0}
    )
    require(
        state.get("deployed_calibration_parameters") == expected_deployed,
        "State deployed parameters drift",
    )
    result = dict(audit)
    result["source_train_target_population"] = dict(train_population)
    result["source_train_cache_file_sha256"] = payload["cache"]["train"]["file_sha256"]
    result["source_train_result_file_sha256"] = sha256_file(args.train_result)
    return result


def analyze_validation(
    validation: Mapping[str, np.ndarray],
    train_audit: Mapping[str, Any],
    contract: Mapping[str, Any],
    source_spec: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    history_labels = [str(row["stratum"]) for row in HISTORY_STRATA]
    activation = bool(train_audit["gate_passed"]) or (
        smoke and train_audit.get("calibration_parameters") is not None
    )
    parameters = (
        dict(train_audit["calibration_parameters"])
        if activation
        else {"slope": 1.0, "intercept": 0.0}
    )
    candidate_bundle = apply_quantity_only_bundle(
        validation["prediction"], validation["time_nll"], parameters
    )
    validation_calibrated = candidate_bundle["quantity_prediction"]
    baseline_metrics = compute_quantity_metrics(validation["prediction"], validation["quantity"])
    calibrated_metrics = compute_quantity_metrics(validation_calibrated, validation["quantity"])
    baseline_metrics["log1p_mse"] = float(
        np.mean(
            np.square(
                np.log1p(validation["prediction"]) - np.log1p(validation["quantity"])
            ),
            dtype=np.float64,
        )
    )
    calibrated_metrics["log1p_mse"] = float(
        np.mean(
            np.square(
                np.log1p(validation_calibrated) - np.log1p(validation["quantity"])
            ),
            dtype=np.float64,
        )
    )
    baseline_metrics["time_nll"] = float(np.mean(validation["time_nll"], dtype=np.float64))
    calibrated_metrics["time_nll"] = baseline_metrics["time_nll"]
    baseline_time_nll = validation["time_nll"]
    calibrated_time_nll = candidate_bundle["time_nll"]
    require(
        np.array_equal(baseline_time_nll, calibrated_time_nll)
        and array_sha256(baseline_time_nll, label="time_nll")
        == array_sha256(calibrated_time_nll, label="time_nll"),
        "Time NLL identity failed",
    )
    reporting_arguments = {
        "body_threshold": float(train_audit["quantity_boundaries"]["p95"]),
        "extreme_tail_threshold": float(train_audit["quantity_boundaries"]["p99"]),
        "quantity_boundaries": [
            float(train_audit["quantity_boundaries"][name])
            for name in ("p50", "p90", "p95", "p99")
        ],
        "history_length": validation["history_length"],
        "history_boundaries": HISTORY_BOUNDARIES,
        "history_labels": history_labels,
    }
    strata = {
        "baseline": compute_reporting_metrics(
            validation["prediction"], validation["quantity"], **reporting_arguments
        ),
        "calibrated": compute_reporting_metrics(
            validation_calibrated, validation["quantity"], **reporting_arguments
        ),
    }
    acceptance = contract["acceptance"]
    overall_mae_change = relative_change(calibrated_metrics["mae"], baseline_metrics["mae"])
    body_mae_change = optional_relative_change(
        strata["calibrated"]["body_le_p95"]["mae"],
        strata["baseline"]["body_le_p95"]["mae"],
    )
    tail_mae_change = optional_relative_change(
        strata["calibrated"]["gt_p99"]["mae"],
        strata["baseline"]["gt_p99"]["mae"],
    )
    validation_gate = False if smoke else (
        calibrated_metrics["rmse"] < baseline_metrics["rmse"]
        and overall_mae_change <= float(acceptance["maximum_overall_mae_regression_fraction"])
        and body_mae_change is not None
        and body_mae_change <= float(acceptance["maximum_body_mae_regression_fraction"])
        and tail_mae_change is not None
        and tail_mae_change <= float(acceptance["maximum_tail_mae_regression_fraction"])
    )
    identity_prediction = apply_affine_calibration(
        validation["prediction"], {"slope": 1.0, "intercept": 0.0}
    )
    require(np.array_equal(identity_prediction, validation["prediction"]), "Identity output drift")
    reference_replay = source_reference_differences(
        baseline_metrics, source_spec, smoke=smoke
    )
    bootstrap = series_clustered_paired_bootstrap_ci(
        validation["prediction"],
        validation_calibrated,
        validation["quantity"],
        validation["series_index"],
        seed=int(contract["runtime"]["paired_bootstrap_seed"]),
        replicates=int(contract["runtime"]["paired_bootstrap_replicates"]),
        draw_chunk_size=int(contract["runtime"]["paired_bootstrap_draw_chunk_size"]),
    )
    return {
        "baseline": baseline_metrics,
        "calibrated": calibrated_metrics,
        "calibration_activated_by_train_rule": activation,
        "deployed_calibration_parameters": parameters,
        "relative_changes": {
            "rmse": relative_change(calibrated_metrics["rmse"], baseline_metrics["rmse"]),
            "overall_mae": overall_mae_change,
            "body_le_p95_mae": body_mae_change,
            "tail_gt_p99_mae": tail_mae_change,
            "time_nll": 0.0,
        },
        "strata": strata,
        "source_replay_absolute_differences": reference_replay,
        "paired_series_bootstrap": bootstrap,
        "time_nll_identity": {
            "exact": True,
            "evidence_type": "quantity_only_postprocessor_bundle_runtime_comparison",
            "distinct_array_storage": calibrated_time_nll is not baseline_time_nll,
            "vector_sha256": array_sha256(baseline_time_nll, label="time_nll"),
        },
        "gate_passed": bool(validation_gate),
        "performance_gate_evaluated": not smoke,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.phase == "validation":
        require(args.train_result is not None, "Validation requires --train-result")
    else:
        require(args.train_result is None, "--train-result is validation-only")
    prerequisite = validate_prerequisite_decision(args)
    context = prepare_execution(args)
    train_audit: dict[str, Any] | None = None
    train_result_reference: dict[str, Any] | None = None
    target_population: dict[str, Any] = {}
    cache_audit: dict[str, Any] = {}
    if args.phase == "train":
        context["frame"] = load_train_only_frame(args.data)
        train, cache_audit["train"], target_population["train"] = extract_split(
            context, args, split="train"
        )
        train_audit = analyze_train(
            train,
            context["contract"],
            context["dataset_spec"],
            partial_smoke=args.smoke,
        )
        train_audit["calibration_state"] = persist_calibration_state(
            output_dir=args.output_dir,
            train=train,
            train_audit=train_audit,
            population=target_population["train"],
            contract_sha256=sha256_file(args.contract),
            source_revision=args.source_revision,
            dataset=args.dataset,
            model_role=args.model_role,
            checkpoint_file_sha256=context["checkpoint_file_sha"],
            checkpoint_state_sha256=context["state_before"],
            data_sha256=context["data_sha256"],
            split_manifest_sha256=context["split_manifest_sha256"],
            source_manifest_sha256=context["source_manifest_audit"]["file_sha256"],
        )
    else:
        train_payload = json.loads(args.train_result.read_text(encoding="utf-8"))
        train_audit = validate_train_result(train_payload, context, args)
        train_result_reference = {
            "file": str(args.train_result.resolve()),
            "file_sha256": sha256_file(args.train_result),
        }
        context["frame"] = load_admitted_frame(args.data)

    validation_audit = None
    if args.phase == "validation":
        validation, cache_audit["validation"], target_population["validation"] = extract_split(
            context, args, split="validation"
        )
        validation_audit = analyze_validation(
            validation,
            train_audit,
            context["contract"],
            context["source_spec"],
            smoke=args.smoke,
        )

    state_after, peak_memory = audit_frozen_state(context)
    context["model"].cpu()
    if context["device"].type == "cuda":
        torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "status": "success",
        "phase": args.phase,
        "contract_id": CONTRACT_ID,
        "contract_sha256": sha256_file(args.contract),
        "source_revision": args.source_revision,
        "dataset": args.dataset,
        "model_role": args.model_role,
        "source_backbone": context["payload"]["backbone"],
        "source_checkpoint_file_sha256": context["checkpoint_file_sha"],
        "source_model_state_sha256": context["state_before"],
        "source_manifest": context["source_manifest_audit"],
        "input_digests": {
            "data_sha256": context["data_sha256"],
            "split_manifest_sha256": context["split_manifest_sha256"],
        },
        "source_model_state_unchanged": context["state_before"] == state_after,
        "source_gradients_absent": True,
        "identity_control": {"slope": 1.0, "intercept": 0.0, "exact_prediction_identity": True},
        "prerequisite_decision": prerequisite,
        "train_result": train_result_reference,
        "train_audit": train_audit,
        "validation": validation_audit,
        "target_population": target_population,
        "cache": cache_audit,
        "runtime": {
            **context["runtime_audit"],
            "device": str(context["device"]),
            "peak_cuda_allocated_bytes": peak_memory,
            "elapsed_seconds": time.monotonic() - started,
        },
        "smoke": bool(args.smoke),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "completed_at": utc_now(),
    }
    save_json(args.output_dir / "result.json", result)
    save_json(
        context["status_path"],
        {"status": "success", "result": "result.json", "completed_at": utc_now()},
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-role", required=True, choices=("B", "rmtpp", "thp"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("train", "validation"), required=True)
    parser.add_argument("--train-result", type=Path, default=None)
    parser.add_argument("--prerequisite-decision", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except BaseException as error:
        if args.output_dir.exists():
            save_json(
                args.output_dir / "status.json",
                {"status": "failed", "error": f"{type(error).__name__}: {error}", "failed_at": utc_now()},
            )
        raise
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
