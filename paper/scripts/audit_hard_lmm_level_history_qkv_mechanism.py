#!/usr/bin/env python3
"""Audit the preregistered Instacart mechanism gate for LPHC-QKV.

The program restores one already-selected LPHC-QKV checkpoint, performs frozen
validation-only inference, and aligns every prediction to the pinned Hard-LMM B
cache by canonical target identity.  It never trains, selects a checkpoint,
fits a calibration, or reads held-out targets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np
import polars as pl
import torch


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hard_lmm_level_history_qkv_mechanism_audit_v1"
SCREENING_CONTRACT_PATH = (
    ROOT / "paper/contracts/hard_lmm_level_history_qkv_screening_v1.json"
)
SCREENING_CONTRACT_SHA256 = (
    "a4ba58bb69f40496de20869643843798a13af17ca169ee4587815c55b8d1061d"
)
SCREENING_CONTRACT_ID = "hard_lmm_level_history_qkv_screening_v1"
CONTRACT_ID = "hard_lmm_level_preserving_history_confidence_residual_v1"
CANDIDATE_BACKBONE = "titantpp_hard_memory_level_history_qkv"
CANDIDATE_ROLE = "hard_lmm_level_history_qkv_candidate"
VARIANT = "count_only_log_regression"
SELECTION = "best_validation_raw_quantity_rmse"
MONITOR = "validation_raw_quantity_rmse"
IDENTITY_COLUMNS = ("series_id", "target_position", "target_seq")
LEVEL_HISTORY_QKV_KERNEL_KEYS = (
    "encoder.layers.0.attn.level_history_q_kernel",
    "encoder.layers.0.attn.level_history_k_kernel",
    "encoder.layers.0.attn.level_history_v_kernel",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def verify_expected_sha256(path: Path, expected: str, *, label: str) -> str:
    require(path.is_file(), f"{label} is missing")
    require(_is_sha256(expected), f"{label} expected SHA-256 is invalid")
    observed = sha256_file(path)
    require(observed == expected, f"{label} SHA-256 drift")
    return observed


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _safe_root_relative_path(value: object, *, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path is invalid")
    path = Path(value)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"{label} path must be root-relative and cannot escape the source root",
    )
    return path


def validate_screening_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the frozen mechanism reference.

    Every numerical threshold and input identity returned here comes from the
    hash-pinned screening contract.  The runner deliberately has no CLI option
    that can replace any of these values after GPU outputs exist.
    """

    require(contract.get("schema_version") == 1, "Screening contract schema drift")
    require(
        contract.get("contract_id") == SCREENING_CONTRACT_ID,
        "Screening contract ID drift",
    )
    require(
        contract.get("status") == "frozen_before_level_history_qkv_gpu_outputs",
        "Screening contract was not frozen before LPHC GPU outputs",
    )
    require(
        contract.get("evaluation_scope") == "validation_only"
        and contract.get("held_out_test_evaluated") is False,
        "Screening contract evaluation scope drift",
    )
    candidate = contract.get("candidate")
    require(isinstance(candidate, Mapping), "Screening candidate is missing")
    require(
        candidate.get("backbone") == CANDIDATE_BACKBONE
        and candidate.get("model_role") == CANDIDATE_ROLE
        and candidate.get("routing_contract_id") == CONTRACT_ID
        and candidate.get("gpu_name_contains") == "RTX 5090",
        "Screening candidate identity drift",
    )
    design = contract.get("design_contract")
    require(isinstance(design, Mapping), "Design-contract binding is missing")
    design_path = _safe_root_relative_path(
        design.get("path"), label="design contract"
    )
    require(_is_sha256(design.get("sha256")), "Design-contract SHA-256 is invalid")

    reference = contract.get("mechanism_reference")
    require(isinstance(reference, Mapping), "Mechanism reference is missing")
    require(
        reference.get("dataset") == "insta_market_basket"
        and reference.get("baseline_role") == "B",
        "Mechanism reference scope drift",
    )
    cache = reference.get("paired_validation_predictions")
    population = reference.get("validation_target_population")
    baseline = reference.get("B_metrics_from_paired_predictions")
    gate = reference.get("gate")
    require(isinstance(cache, Mapping), "Mechanism B-cache binding is missing")
    require(isinstance(population, Mapping), "Mechanism population binding is missing")
    require(isinstance(baseline, Mapping), "Mechanism B metrics are missing")
    require(isinstance(gate, Mapping), "Mechanism gate is missing")

    cache_path = _safe_root_relative_path(
        cache.get("path"), label="paired B validation cache"
    )
    require(_is_sha256(cache.get("sha256")), "B-cache SHA-256 is invalid")
    required_columns = cache.get("required_columns")
    require(
        isinstance(required_columns, list)
        and set(required_columns)
        == {
            *IDENTITY_COLUMNS,
            "history_length",
            "true_qty",
            "pred_b",
        },
        "B-cache required-column contract drift",
    )
    require(
        isinstance(cache.get("row_count"), int) and cache["row_count"] > 0,
        "B-cache row count is invalid",
    )

    require(
        population.get("split") == "validation"
        and population.get("loader") == "RMTPPWeekLookbackDataset"
        and population.get("lookback_weeks") == 52
        and population.get("max_seq_len") == 64
        and population.get("schema_version") == 1
        and population.get("target_count") == cache.get("row_count"),
        "Mechanism validation-population contract drift",
    )
    require(
        population.get("identity_fields")
        == ["oper_part_no", "target_position", "target_seq"],
        "Mechanism identity fields drift",
    )
    for field in ("target_identity_sha256", "target_quantity_sha256"):
        require(_is_sha256(population.get(field)), f"Mechanism {field} is invalid")

    binding = contract.get("data_bindings", {}).get("insta_market_basket")
    require(isinstance(binding, Mapping), "Instacart data binding is missing")
    require(
        _is_sha256(binding.get("data_sha256"))
        and _is_sha256(binding.get("split_manifest_sha256")),
        "Instacart data-binding digest is invalid",
    )
    require(
        binding.get("validation_target_population") == population,
        "Mechanism population differs from the Instacart data binding",
    )

    for field in (
        "raw_rmse",
        "overall_mae",
        "centered_mse",
        "bias",
        "prediction_mean",
        "target_mean",
    ):
        require(
            isinstance(baseline.get(field), (int, float))
            and math.isfinite(float(baseline[field])),
            f"Mechanism B metric is invalid: {field}",
        )
    for field in ("H2_3", "H8_15"):
        stratum = baseline.get(field)
        require(
            isinstance(stratum, Mapping)
            and isinstance(stratum.get("count"), int)
            and stratum["count"] > 0
            and isinstance(stratum.get("raw_mse"), (int, float))
            and math.isfinite(float(stratum["raw_mse"]))
            and float(stratum["raw_mse"]) >= 0.0,
            f"Mechanism B stratum is invalid: {field}",
        )
    require(
        float(baseline["raw_rmse"]) >= 0.0
        and float(baseline["overall_mae"]) >= 0.0
        and float(baseline["centered_mse"]) >= 0.0,
        "Mechanism B loss metric is negative",
    )
    require(
        abs(
            float(baseline["prediction_mean"])
            - float(baseline["target_mean"])
            - float(baseline["bias"])
        )
        <= 1e-12,
        "Mechanism B bias/mean identity drift",
    )

    require(
        gate.get("raw_rmse_strictly_less_than_B") is True
        and gate.get("overall_mae_strictly_less_than_B") is True
        and gate.get("centered_mse_strictly_less_than_B") is True
        and gate.get("H2_3_raw_mse_ratio_max") == 1.0
        and gate.get("H8_15_raw_mse_ratio_strictly_less_than") == 1.0,
        "Mechanism comparator semantics drift",
    )
    for field in (
        "absolute_mean_prediction_shift_vs_B_max",
        "absolute_candidate_bias_max",
    ):
        require(
            isinstance(gate.get(field), (int, float))
            and math.isfinite(float(gate[field]))
            and float(gate[field]) >= 0.0,
            f"Mechanism gate threshold is invalid: {field}",
        )
    require(
        abs(
            float(gate["absolute_mean_prediction_shift_vs_B_max"])
            - 0.01 * float(baseline["raw_rmse"])
        )
        <= 1e-10,
        "Mechanism mean-shift threshold is not one percent of B RMSE",
    )
    require(
        abs(
            float(gate["absolute_candidate_bias_max"])
            - abs(float(baseline["bias"]))
        )
        <= 1e-15,
        "Mechanism bias threshold differs from absolute B bias",
    )
    screening_phase = contract.get("phases", {}).get("seed42_screening")
    require(
        isinstance(screening_phase, Mapping)
        and screening_phase.get("seed") == 42
        and screening_phase.get("epochs") == 300
        and screening_phase.get("minimum_epochs") == 40
        and screening_phase.get("patience") == 40
        and screening_phase.get("datasets_in_order", [None])[0]
        == "insta_market_basket"
        and screening_phase.get("instacart_mechanism_gate_before_other_datasets")
        is True,
        "Mechanism execution-order contract drift",
    )

    expected_b_metrics: dict[str, float | int] = {
        "count": int(population["target_count"]),
        "raw_mse": float(baseline["raw_rmse"]) ** 2,
        "raw_rmse": float(baseline["raw_rmse"]),
        "overall_mae": float(baseline["overall_mae"]),
        "bias": float(baseline["bias"]),
        "centered_mse": float(baseline["centered_mse"]),
        "prediction_mean": float(baseline["prediction_mean"]),
        "target_mean": float(baseline["target_mean"]),
        "history_2_3_count": int(baseline["H2_3"]["count"]),
        "history_2_3_raw_mse": float(baseline["H2_3"]["raw_mse"]),
        "history_8_15_count": int(baseline["H8_15"]["count"]),
        "history_8_15_raw_mse": float(baseline["H8_15"]["raw_mse"]),
    }
    thresholds = {
        "raw_rmse_strict_max": float(baseline["raw_rmse"]),
        "overall_mae_strict_max": float(baseline["overall_mae"]),
        "centered_mse_strict_max": float(baseline["centered_mse"]),
        "history_2_3_raw_mse_inclusive_max": float(
            baseline["H2_3"]["raw_mse"]
        )
        * float(gate["H2_3_raw_mse_ratio_max"]),
        "history_8_15_raw_mse_strict_max": float(
            baseline["H8_15"]["raw_mse"]
        )
        * float(gate["H8_15_raw_mse_ratio_strictly_less_than"]),
        "absolute_mean_prediction_shift_inclusive_max": float(
            gate["absolute_mean_prediction_shift_vs_B_max"]
        ),
        "absolute_bias_inclusive_max": float(
            gate["absolute_candidate_bias_max"]
        ),
    }
    return {
        "screening_contract_id": SCREENING_CONTRACT_ID,
        "design_contract_id": CONTRACT_ID,
        "design_contract_path": design_path,
        "design_contract_sha256": str(design["sha256"]),
        "cache_path": cache_path,
        "cache_sha256": str(cache["sha256"]),
        "cache_required_columns": list(required_columns),
        "population": dict(population),
        "data_sha256": str(binding["data_sha256"]),
        "split_manifest_sha256": str(binding["split_manifest_sha256"]),
        "expected_b_metrics": expected_b_metrics,
        "thresholds": thresholds,
    }


def load_screening_contract(
    path: Path = SCREENING_CONTRACT_PATH,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    digest = verify_expected_sha256(
        path, SCREENING_CONTRACT_SHA256, label="LPHC screening contract"
    )
    contract = read_json(path, label="LPHC screening contract")
    return contract, validate_screening_contract(contract), digest


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def verify_source_file_map(root: Path, files: Mapping[str, Any]) -> dict[str, str]:
    """Verify an exact, safe map of committed source files."""
    require(bool(files), "Deployment source-file map is empty")
    root = root.resolve()
    observed: dict[str, str] = {}
    for relative, expected in files.items():
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"Unsafe source path: {relative!r}",
        )
        candidate = (root / relative).resolve()
        require(candidate.is_relative_to(root), f"Source path escapes root: {relative}")
        observed[relative] = verify_expected_sha256(
            candidate, str(expected), label=f"source file {relative}"
        )
    return observed


def verify_deployment_source(
    manifest_path: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_digest = verify_expected_sha256(
        manifest_path,
        expected_manifest_sha256,
        label="deployment manifest",
    )
    manifest = read_json(manifest_path, label="deployment manifest")
    require(manifest.get("schema_version") == 1, "Deployment schema drift")
    require(manifest.get("host_role") == "5090", "Deployment is not bound to 5090")
    require(
        manifest.get("status") == "frozen_before_gpu_outputs",
        "Deployment was not frozen before GPU outputs",
    )
    source_root = Path(str(manifest.get("source_root", ""))).resolve()
    require(source_root == ROOT.resolve(), "Deployment source root differs from this runner")
    revision = str(manifest.get("source_revision", ""))
    require(len(revision) == 40, "Deployment source revision is invalid")
    observed_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    require(observed_revision == revision, "Deployment Git revision drift")
    tracked = {
        name
        for name in subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=source_root
        ).decode().split("\0")
        if name
    }
    source_files = manifest.get("source_files")
    require(isinstance(source_files, Mapping), "Deployment source-file map is missing")
    require(set(source_files) == tracked, "Deployment tracked-source population drift")
    observed_files = verify_source_file_map(source_root, source_files)
    for folder in ("models", "paper/scripts", "simple_lab_test"):
        unregistered = {
            str(path.relative_to(source_root))
            for path in (source_root / folder).rglob("*.py")
        } - tracked
        require(not unregistered, f"Unregistered Python source: {sorted(unregistered)}")
    return manifest, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
        "source_root": str(source_root),
        "source_revision": revision,
        "tracked_file_count": len(tracked),
        "all_tracked_file_hashes_verified": True,
        "unregistered_python_source_absent": True,
        "source_files": observed_files,
    }


def import_runtime() -> dict[str, Any]:
    """Import project code only after the deployment source has been verified."""
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import (
        load_train_validation_frame,
        prepare_count_frame,
        target_outputs,
    )
    from paper.scripts.run_count_aware_tpp_backbone_control import (
        exact_target_population,
    )
    from paper.scripts.run_hard_lmm_bounded_qk_error_diagnostic import (
        build_validation_metadata,
    )
    from paper.scripts.run_matched_frozen_lognormal_duration import (
        build_source_model,
    )
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    return {
        "validate_checkpoint_route": validate_checkpoint_route,
        "load_train_validation_frame": load_train_validation_frame,
        "prepare_count_frame": prepare_count_frame,
        "target_outputs": target_outputs,
        "exact_target_population": exact_target_population,
        "build_validation_metadata": build_validation_metadata,
        "build_source_model": build_source_model,
        "make_loader": make_loader,
        "canonical_state_dict_sha256": canonical_state_dict_sha256,
        "torch_load_checkpoint": torch_load_checkpoint,
    }


def _finite_vector(
    values: Any, *, label: str, nonnegative: bool = False
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    require(result.ndim == 1 and result.size > 0, f"{label} must be nonempty and one-dimensional")
    require(bool(np.isfinite(result).all()), f"{label} contains a nonfinite value")
    if nonnegative:
        require(bool((result >= 0.0).all()), f"{label} contains a negative value")
    return result


def quantity_metrics(target: Any, prediction: Any) -> dict[str, float | int]:
    target_array = _finite_vector(target, label="target", nonnegative=True)
    prediction_array = _finite_vector(
        prediction, label="prediction", nonnegative=True
    )
    require(target_array.shape == prediction_array.shape, "Target/prediction shape drift")
    error = prediction_array - target_array
    bias = float(error.mean(dtype=np.float64))
    mse = float(np.square(error).mean(dtype=np.float64))
    centered_mse = float(np.square(error - bias).mean(dtype=np.float64))
    decomposition_residual = mse - centered_mse - bias * bias
    require(
        abs(decomposition_residual) <= 1e-10 * max(1.0, abs(mse)),
        "MSE bias decomposition drift",
    )
    return {
        "count": int(error.size),
        "raw_mse": mse,
        "raw_rmse": math.sqrt(mse),
        "overall_mae": float(np.abs(error).mean(dtype=np.float64)),
        "bias": bias,
        "centered_mse": centered_mse,
        "prediction_mean": float(prediction_array.mean(dtype=np.float64)),
        "target_mean": float(target_array.mean(dtype=np.float64)),
        "mse_decomposition_residual": decomposition_residual,
    }


def mechanism_metrics(
    target: Any,
    candidate_prediction: Any,
    b_prediction: Any,
    history_length: Any,
) -> dict[str, Any]:
    target_array = _finite_vector(target, label="target", nonnegative=True)
    candidate = _finite_vector(
        candidate_prediction, label="LPHC prediction", nonnegative=True
    )
    baseline = _finite_vector(b_prediction, label="B prediction", nonnegative=True)
    history_raw = np.asarray(history_length)
    require(
        history_raw.ndim == 1
        and history_raw.shape == target_array.shape
        and np.issubdtype(history_raw.dtype, np.integer),
        "History length must be an aligned integer vector",
    )
    history = history_raw.astype(np.int64, copy=False)
    require(bool((history >= 1).all()), "History length must be positive")
    require(
        candidate.shape == baseline.shape == target_array.shape,
        "Mechanism arrays are not aligned",
    )
    candidate_overall = quantity_metrics(target_array, candidate)
    baseline_overall = quantity_metrics(target_array, baseline)
    history_rows: dict[str, dict[str, float | int]] = {}
    for name, lower, upper in (("2_3", 2, 3), ("8_15", 8, 15)):
        selected = (history >= lower) & (history <= upper)
        require(bool(selected.any()), f"History {lower}-{upper} stratum is empty")
        candidate_error = candidate[selected] - target_array[selected]
        baseline_error = baseline[selected] - target_array[selected]
        history_rows[name] = {
            "count": int(selected.sum()),
            "candidate_raw_mse": float(
                np.square(candidate_error).mean(dtype=np.float64)
            ),
            "B_raw_mse": float(np.square(baseline_error).mean(dtype=np.float64)),
        }
    shift = candidate - baseline
    return {
        "candidate": candidate_overall,
        "B": baseline_overall,
        "history": history_rows,
        "mean_prediction_shift_candidate_minus_B": float(
            shift.mean(dtype=np.float64)
        ),
        "absolute_mean_prediction_shift_candidate_minus_B": abs(
            float(shift.mean(dtype=np.float64))
        ),
    }


def evaluate_mechanism_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    candidate = metrics.get("candidate")
    history = metrics.get("history")
    require(isinstance(candidate, Mapping), "Candidate metrics are missing")
    require(isinstance(history, Mapping), "History metrics are missing")
    require(isinstance(history.get("2_3"), Mapping), "History 2-3 metrics are missing")
    require(isinstance(history.get("8_15"), Mapping), "History 8-15 metrics are missing")

    values = {
        "raw_rmse": float(candidate["raw_rmse"]),
        "overall_mae": float(candidate["overall_mae"]),
        "centered_mse": float(candidate["centered_mse"]),
        "history_2_3_raw_mse": float(history["2_3"]["candidate_raw_mse"]),
        "history_8_15_raw_mse": float(history["8_15"]["candidate_raw_mse"]),
        "absolute_mean_prediction_shift": float(
            metrics["absolute_mean_prediction_shift_candidate_minus_B"]
        ),
        "absolute_bias": abs(float(candidate["bias"])),
    }
    require(all(math.isfinite(value) and value >= 0.0 for value in values.values()), "Gate metric is invalid")
    checks = {
        "raw_rmse_strictly_better_than_B": values["raw_rmse"]
        < thresholds["raw_rmse_strict_max"],
        "overall_mae_strictly_better_than_B": values["overall_mae"]
        < thresholds["overall_mae_strict_max"],
        "centered_mse_strictly_better_than_B": values["centered_mse"]
        < thresholds["centered_mse_strict_max"],
        "history_2_3_raw_mse_not_worse_than_B": values["history_2_3_raw_mse"]
        <= thresholds["history_2_3_raw_mse_inclusive_max"],
        "history_8_15_raw_mse_strictly_better_than_B": values[
            "history_8_15_raw_mse"
        ]
        < thresholds["history_8_15_raw_mse_strict_max"],
        "absolute_mean_prediction_shift_within_one_percent_B_RMSE": values[
            "absolute_mean_prediction_shift"
        ]
        <= thresholds["absolute_mean_prediction_shift_inclusive_max"],
        "absolute_bias_not_larger_than_B": values["absolute_bias"]
        <= thresholds["absolute_bias_inclusive_max"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "values": values,
        "thresholds": dict(thresholds),
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


def _frame_unique(frame: pl.DataFrame, *, label: str) -> None:
    require(frame.height > 0, f"{label} is empty")
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    require(not missing, f"{label} identity columns are missing: {missing}")
    count = frame.select(pl.struct(list(IDENTITY_COLUMNS)).n_unique()).item()
    require(count == frame.height, f"{label} canonical identities are not unique")


def align_candidate_to_b_cache(
    b_frame: pl.DataFrame,
    candidate_frame: pl.DataFrame,
    *,
    expected_count: int,
) -> pl.DataFrame:
    required_b = {
        *IDENTITY_COLUMNS,
        "split",
        "history_length",
        "true_qty",
        "pred_b",
    }
    required_candidate = {
        *IDENTITY_COLUMNS,
        "candidate_history_length",
        "candidate_true_qty",
        "pred_lphc",
    }
    require(not (required_b - set(b_frame.columns)), "B cache schema is incomplete")
    require(
        not (required_candidate - set(candidate_frame.columns)),
        "Candidate prediction schema is incomplete",
    )
    require(
        b_frame.height == candidate_frame.height == expected_count,
        "Candidate/B target count drift",
    )
    _frame_unique(b_frame, label="B cache")
    _frame_unique(candidate_frame, label="candidate predictions")
    require(
        set(b_frame["split"].cast(pl.String).unique().to_list()) == {"validation"},
        "B cache contains a non-validation split",
    )
    joined = (
        b_frame.select(sorted(required_b))
        .join(
            candidate_frame.select(sorted(required_candidate)),
            on=list(IDENTITY_COLUMNS),
            how="inner",
            validate="1:1",
        )
        .sort(list(IDENTITY_COLUMNS))
    )
    require(joined.height == expected_count, "Candidate/B identity sets differ")
    require(
        np.array_equal(
            joined["true_qty"].cast(pl.Float64).to_numpy(),
            joined["candidate_true_qty"].cast(pl.Float64).to_numpy(),
        ),
        "Candidate/B target quantity differs after identity alignment",
    )
    require(
        np.array_equal(
            joined["history_length"].cast(pl.Int64).to_numpy(),
            joined["candidate_history_length"].cast(pl.Int64).to_numpy(),
        ),
        "Candidate/B history length differs after identity alignment",
    )
    for name in ("true_qty", "pred_b", "pred_lphc"):
        _finite_vector(
            joined[name].cast(pl.Float64).to_numpy(),
            label=name,
            nonnegative=True,
        )
    return joined


def validate_b_cache_metrics(
    metrics: Mapping[str, Any],
    *,
    expected: Mapping[str, float | int],
    tolerance: float = 1e-9,
) -> dict[str, float]:
    observed = {
        "count": metrics["B"]["count"],
        "raw_mse": metrics["B"]["raw_mse"],
        "raw_rmse": metrics["B"]["raw_rmse"],
        "overall_mae": metrics["B"]["overall_mae"],
        "bias": metrics["B"]["bias"],
        "centered_mse": metrics["B"]["centered_mse"],
        "prediction_mean": metrics["B"]["prediction_mean"],
        "target_mean": metrics["B"]["target_mean"],
        "history_2_3_count": metrics["history"]["2_3"]["count"],
        "history_2_3_raw_mse": metrics["history"]["2_3"]["B_raw_mse"],
        "history_8_15_count": metrics["history"]["8_15"]["count"],
        "history_8_15_raw_mse": metrics["history"]["8_15"]["B_raw_mse"],
    }
    differences: dict[str, float] = {}
    for name, expected_value in expected.items():
        if name.endswith("count") or name == "count":
            require(observed[name] == expected_value, f"Pinned B cache {name} drift")
            differences[name] = 0.0
        else:
            difference = abs(float(observed[name]) - float(expected_value))
            require(difference <= tolerance, f"Pinned B cache {name} drift")
            differences[name] = difference
    return differences


def validate_candidate_artifacts(
    *,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    summary_path: Path,
    expected_summary_sha256: str,
    source_revision: str,
    runtime: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint_digest = verify_expected_sha256(
        checkpoint_path,
        expected_checkpoint_sha256,
        label="LPHC selected checkpoint",
    )
    summary_digest = verify_expected_sha256(
        summary_path,
        expected_summary_sha256,
        label="LPHC training summary",
    )
    summary = read_json(summary_path, label="LPHC training summary")
    payload = runtime["torch_load_checkpoint"](checkpoint_path, map_location="cpu")
    require(isinstance(payload, dict), "LPHC checkpoint payload is invalid")
    runtime["validate_checkpoint_route"](payload, CANDIDATE_BACKBONE)
    common_required = {
        "backbone": CANDIDATE_BACKBONE,
        "variant": VARIANT,
        "seed": 42,
        "checkpoint_selection": SELECTION,
        "checkpoint_monitor": MONITOR,
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "source_revision": source_revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for artifact, value in (("checkpoint", payload), ("summary", summary)):
        for name, expected in common_required.items():
            require(value.get(name) == expected, f"LPHC {artifact} metadata drift: {name}")
        require(
            value.get("source_revision_history") == [source_revision],
            f"LPHC {artifact} source lineage drift",
        )
    require(payload.get("selection") == SELECTION, "LPHC checkpoint selector drift")
    require(
        payload.get("resume_identity") == summary.get("resume_identity"),
        "LPHC checkpoint/summary resume identity drift",
    )
    initial_state_digest = payload.get("initial_state_sha256")
    require(
        _is_sha256(initial_state_digest)
        and initial_state_digest == summary.get("initial_state_sha256"),
        "LPHC initial-state identity drift",
    )
    require(summary.get("status") == "success", "LPHC training summary is incomplete")
    require(summary.get("epochs") == 300, "LPHC mechanism audit requires the e300 budget")
    require(
        40 <= int(summary.get("completed_epochs", -1)) <= 300,
        "LPHC completed epoch count violates the screening contract",
    )
    require(
        str(summary.get("training_device", "")).startswith("cuda")
        and int(summary.get("cuda_peak_memory_allocated_bytes", 0)) > 0,
        "LPHC selected fit lacks CUDA execution evidence",
    )
    require(payload.get("best_epoch") == summary.get("best_epoch"), "LPHC best epoch drift")
    state = payload.get("model_state_dict")
    require(isinstance(state, Mapping), "LPHC selected state is missing")
    state_digest = runtime["canonical_state_dict_sha256"](state)
    require(
        state_digest
        == payload.get("model_state_sha256")
        == summary.get("checkpoint_state_sha256"),
        "LPHC selected state digest drift",
    )
    encoder = payload.get("encoder_config")
    require(
        isinstance(encoder, Mapping)
        and encoder.get("model_role") == CANDIDATE_ROLE
        and encoder.get("routing_contract_id") == CONTRACT_ID
        and encoder.get("additional_parameter_count") == 384,
        "LPHC encoder identity drift",
    )
    resume = summary.get("resume_identity")
    require(isinstance(resume, Mapping), "LPHC resume identity is missing")
    arguments = resume.get("arguments")
    require(
        isinstance(arguments, Mapping)
        and arguments.get("dataset_contract") == "insta_market_basket"
        and arguments.get("lookback_weeks") == 52
        and arguments.get("max_seq_len") == 64
        and arguments.get("max_train_batches") is None
        and arguments.get("max_val_batches") is None
        and arguments.get("max_series") is None,
        "LPHC fit is not the full Instacart screening population",
    )
    model = runtime["build_source_model"](payload, max_seq_len=64)
    model.requires_grad_(False).eval().to(device)
    restored_digest = runtime["canonical_state_dict_sha256"](model.state_dict())
    require(restored_digest == state_digest, "LPHC strict model restore drift")
    return model, summary, {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": checkpoint_digest,
        "checkpoint_state_sha256": state_digest,
        "summary_path": str(summary_path.resolve()),
        "summary_file_sha256": summary_digest,
        "initial_state_sha256": initial_state_digest,
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "strict_restore": True,
    }


def validate_candidate_job_audit(
    *,
    path: Path,
    expected_sha256: str,
    checkpoint_audit: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    expected_population: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the independent checkpoint audit and its fresh-init proof."""

    digest = verify_expected_sha256(
        path, expected_sha256, label="LPHC candidate job audit"
    )
    audit = read_json(path, label="LPHC candidate job audit")
    detail = audit.get("level_history_qkv_audit")
    require(
        audit.get("status") == "passed" and isinstance(detail, Mapping),
        "LPHC candidate job audit did not pass",
    )
    require(
        detail.get("status") == "passed"
        and detail.get("contract_id") == CONTRACT_ID
        and detail.get("screening_contract_sha256")
        == SCREENING_CONTRACT_SHA256
        and detail.get("evaluation_scope") == "validation_only"
        and detail.get("held_out_test_evaluated") is False,
        "LPHC candidate job-audit scope or contract drift",
    )
    require(
        audit.get("checkpoint_sha256")
        == detail.get("best_checkpoint_file_sha256")
        == checkpoint_audit["checkpoint_file_sha256"],
        "LPHC candidate job-audit checkpoint file drift",
    )
    require(
        detail.get("best_state_sha256")
        == checkpoint_audit["checkpoint_state_sha256"],
        "LPHC candidate job-audit selected state drift",
    )
    require(
        audit.get("summary_sha256") == checkpoint_audit["summary_file_sha256"],
        "LPHC candidate job-audit summary drift",
    )
    population = detail.get("validation_target_population")
    require(
        population == expected_population
        and audit.get("validation_target_identity_sha256")
        == expected_population["target_identity_sha256"]
        and audit.get("validation_target_quantity_sha256")
        == expected_population["target_quantity_sha256"],
        "LPHC candidate job-audit validation population drift",
    )

    proof = detail.get("fresh_initial_identity")
    require(isinstance(proof, Mapping), "LPHC fresh initial identity proof is missing")
    required_fields = {
        "method",
        "seed",
        "B_backbone",
        "candidate_backbone",
        "fresh_B_state_sha256",
        "candidate_common_state_sha256",
        "candidate_initial_state_sha256",
        "new_parameter_state_keys",
        "new_parameters_exact_zero",
        "additional_parameter_count",
        "inherited_state_bitwise_equal",
        "synthetic_outputs_bitwise_equal",
        "post_construction_RNG_state_bitwise_equal",
        "summary_and_checkpoint_initial_state_sha256_equal",
    }
    require(set(proof) == required_fields, "LPHC fresh initial identity schema drift")
    for field in (
        "fresh_B_state_sha256",
        "candidate_common_state_sha256",
        "candidate_initial_state_sha256",
    ):
        require(_is_sha256(proof.get(field)), f"LPHC fresh identity {field} is invalid")
    require(
        proof.get("method") == "same_seed_fresh_B_identity_v1"
        and proof.get("seed") == 42
        and proof.get("B_backbone") == "titantpp"
        and proof.get("candidate_backbone") == CANDIDATE_BACKBONE
        and proof.get("new_parameter_state_keys")
        == list(LEVEL_HISTORY_QKV_KERNEL_KEYS)
        and proof.get("new_parameters_exact_zero") is True
        and proof.get("additional_parameter_count") == 384
        and proof.get("inherited_state_bitwise_equal") is True
        and proof.get("synthetic_outputs_bitwise_equal") is True
        and proof.get("post_construction_RNG_state_bitwise_equal") is True
        and proof.get("summary_and_checkpoint_initial_state_sha256_equal") is True,
        "LPHC fresh initial identity contract failed",
    )
    require(
        proof["fresh_B_state_sha256"] == proof["candidate_common_state_sha256"],
        "Fresh B and LPHC common initial states differ",
    )
    require(
        proof["candidate_initial_state_sha256"]
        == checkpoint_audit["initial_state_sha256"]
        == candidate_summary.get("initial_state_sha256"),
        "LPHC fresh initial-state digest differs from the trained artifact",
    )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "status": "passed",
        "checkpoint_and_summary_bound": True,
        "validation_population_bound": True,
        "fresh_initial_identity": dict(proof),
    }


def audit_frozen_state(
    model: torch.nn.Module,
    before_sha256: str,
    digest: Callable[[Mapping[str, torch.Tensor]], str],
) -> dict[str, Any]:
    after = digest(model.state_dict())
    all_parameters_frozen = all(
        not parameter.requires_grad for parameter in model.parameters()
    )
    gradients_absent = all(parameter.grad is None for parameter in model.parameters())
    require(after == before_sha256, "LPHC model state changed during inference")
    require(all_parameters_frozen, "LPHC model retains a trainable parameter")
    require(gradients_absent, "LPHC model accumulated gradients during inference")
    return {
        "before": before_sha256,
        "after": after,
        "unchanged": True,
        "all_parameters_frozen": True,
        "all_gradients_absent": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    for expected, label in (
        (args.expected_deployment_manifest_sha256, "deployment manifest"),
        (args.expected_candidate_checkpoint_sha256, "candidate checkpoint"),
        (args.expected_candidate_summary_sha256, "candidate summary"),
        (args.expected_candidate_job_audit_sha256, "candidate job audit"),
    ):
        require(_is_sha256(expected), f"Expected {label} SHA-256 is invalid")
    require(args.batch_size > 0, "Inference batch size must be positive")
    screening_contract, reference, screening_contract_digest = (
        load_screening_contract()
    )
    design_contract_path = ROOT / reference["design_contract_path"]
    design_contract_digest = verify_expected_sha256(
        design_contract_path,
        reference["design_contract_sha256"],
        label="LPHC design contract",
    )
    expected_population = reference["population"]
    expected_count = int(expected_population["target_count"])
    expected_b_cache_path = (ROOT / reference["cache_path"]).resolve()
    require(
        args.b_cache.resolve() == expected_b_cache_path,
        "B cache path differs from the screening mechanism reference",
    )
    output_paths = {
        "paired": args.output_dir / "paired_lphc_b_validation_predictions.parquet",
        "audit": args.output_dir / "audit.json",
        "decision": args.output_dir / "decision.json",
    }
    require(
        not any(path.exists() for path in output_paths.values()),
        "Mechanism audit output already exists; overwrite is forbidden",
    )
    deployment, source_audit = verify_deployment_source(
        args.deployment_manifest, args.expected_deployment_manifest_sha256
    )
    runtime = import_runtime()
    data_digest = verify_expected_sha256(
        args.data, reference["data_sha256"], label="Instacart data"
    )
    split_digest = verify_expected_sha256(
        args.split_manifest,
        reference["split_manifest_sha256"],
        label="Instacart split manifest",
    )
    b_cache_digest = verify_expected_sha256(
        args.b_cache, reference["cache_sha256"], label="pinned B validation cache"
    )
    device = torch.device(args.device)
    cuda_index: int | None = None
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        cuda_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        device = torch.device("cuda", cuda_index)
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    frame = runtime["load_train_validation_frame"](args.data)
    admitted_splits = set(frame["chronological_split"].unique().to_list())
    require(admitted_splits == {"train", "validation"}, "Unexpected admitted split set")
    frame = runtime["prepare_count_frame"](frame)
    target_quantities, population = runtime["exact_target_population"](
        frame,
        target_split="validation",
        lookback_weeks=52,
        max_seq_len=64,
    )
    for name, expected in {
        "target_count": expected_count,
        "target_identity_sha256": expected_population["target_identity_sha256"],
        "target_quantity_sha256": expected_population["target_quantity_sha256"],
    }.items():
        require(population.get(name) == expected, f"Validation population drift: {name}")
    loader = runtime["make_loader"](
        frame,
        target_split="validation",
        batch_size=args.batch_size,
        lookback_weeks=52,
        max_seq_len=64,
        shuffle=False,
        generator=None,
    )
    dataset = loader.dataset
    require(len(dataset) == expected_count, "Validation loader count drift")
    metadata = runtime["build_validation_metadata"](
        dataset, lookback_weeks=52, max_seq_len=64
    )
    require(
        np.array_equal(metadata["true_qty"], target_quantities),
        "Validation metadata target order drift",
    )

    model, summary, checkpoint_audit = validate_candidate_artifacts(
        checkpoint_path=args.candidate_checkpoint,
        expected_checkpoint_sha256=args.expected_candidate_checkpoint_sha256,
        summary_path=args.candidate_summary,
        expected_summary_sha256=args.expected_candidate_summary_sha256,
        source_revision=str(deployment["source_revision"]),
        runtime=runtime,
        device=device,
    )
    candidate_job_audit = validate_candidate_job_audit(
        path=args.candidate_job_audit,
        expected_sha256=args.expected_candidate_job_audit_sha256,
        checkpoint_audit=checkpoint_audit,
        candidate_summary=summary,
        expected_population=expected_population,
    )
    digest = runtime["canonical_state_dict_sha256"]
    before_state = digest(model.state_dict())
    predictions = np.empty(expected_count, dtype=np.float64)
    log_losses = np.empty(expected_count, dtype=np.float64)
    offset = 0
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (_, dts, mask, parts, quantities) in enumerate(loader):
            require(quantities is not None, "Validation raw quantities are missing")
            count = int(dts.size(0))
            end = offset + count
            require(
                np.array_equal(
                    parts.numpy().astype(np.int64),
                    metadata["series_index"][offset:end],
                ),
                "Validation loader series order drift",
            )
            outputs = runtime["target_outputs"](
                model,
                dts.to(device, non_blocking=device.type == "cuda"),
                mask.to(device, non_blocking=device.type == "cuda"),
                quantities.to(device, non_blocking=device.type == "cuda"),
                lambda_log_qty=1.0,
            )
            true_qty = outputs["true_qty"].detach().cpu().numpy().astype(np.float64)
            history = (
                outputs["history_length"].detach().cpu().numpy().astype(np.int64)
            )
            require(
                np.array_equal(true_qty, metadata["true_qty"][offset:end]),
                "LPHC batch target order drift",
            )
            require(
                np.array_equal(history, metadata["history_length"][offset:end]),
                "LPHC batch history order drift",
            )
            predictions[offset:end] = (
                outputs["pred_qty"].detach().cpu().numpy().astype(np.float64)
            )
            log_losses[offset:end] = (
                outputs["log_qty_loss"].detach().cpu().numpy().astype(np.float64)
            )
            offset = end
            if batch_index % 100 == 0 or offset == expected_count:
                print(
                    json.dumps(
                        {
                            "stage": "lphc_validation_inference",
                            "completed_targets": offset,
                            "total_targets": expected_count,
                            "elapsed_seconds": time.perf_counter() - inference_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    require(offset == expected_count, "LPHC validation inference is incomplete")
    _finite_vector(predictions, label="LPHC predictions", nonnegative=True)
    _finite_vector(log_losses, label="LPHC log quantity losses")
    frozen_state = audit_frozen_state(model, before_state, digest)

    candidate_observed = quantity_metrics(metadata["true_qty"], predictions)
    replay_differences = {
        "best_val_qty_mae": abs(
            float(candidate_observed["overall_mae"])
            - float(summary["best_val_qty_mae"])
        ),
        "best_val_qty_rmse": abs(
            float(candidate_observed["raw_rmse"])
            - float(summary["best_val_qty_rmse"])
        ),
        "best_val_log_qty_mse": abs(
            float(log_losses.mean(dtype=np.float64))
            - float(summary["best_val_log_qty_mse"])
        ),
    }
    require(
        all(value <= 1e-5 for value in replay_differences.values()),
        "LPHC selected-checkpoint aggregate replay drift",
    )

    candidate_frame = pl.DataFrame(
        {
            "series_id": metadata["series_id"],
            "target_position": metadata["target_position"],
            "target_seq": metadata["target_seq"],
            "candidate_history_length": metadata["history_length"],
            "candidate_true_qty": metadata["true_qty"],
            "pred_lphc": predictions,
        }
    )
    b_frame = pl.read_parquet(
        args.b_cache,
        columns=[
            "split",
            *IDENTITY_COLUMNS,
            "history_length",
            "true_qty",
            "pred_b",
        ],
    )
    aligned = align_candidate_to_b_cache(
        b_frame, candidate_frame, expected_count=expected_count
    )
    metrics = mechanism_metrics(
        aligned["true_qty"].to_numpy(),
        aligned["pred_lphc"].to_numpy(),
        aligned["pred_b"].to_numpy(),
        aligned["history_length"].to_numpy(),
    )
    b_replay_differences = validate_b_cache_metrics(
        metrics, expected=reference["expected_b_metrics"]
    )
    gate = evaluate_mechanism_gate(metrics, thresholds=reference["thresholds"])
    paired_output = aligned.with_columns(
        (pl.col("pred_lphc") - pl.col("true_qty")).alias("error_lphc"),
        (pl.col("pred_b") - pl.col("true_qty")).alias("error_b"),
        (pl.col("pred_lphc") - pl.col("pred_b")).alias(
            "prediction_shift_lphc_minus_b"
        ),
    ).select(
        "split",
        *IDENTITY_COLUMNS,
        "history_length",
        "true_qty",
        "pred_b",
        "pred_lphc",
        "error_b",
        "error_lphc",
        "prediction_shift_lphc_minus_b",
    )
    atomic_write_parquet(paired_output, output_paths["paired"])
    paired_digest = sha256_file(output_paths["paired"])
    completed_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema": SCHEMA,
        "status": "passed",
        "created_at_utc": completed_at,
        "contract_id": SCREENING_CONTRACT_ID,
        "routing_contract_id": CONTRACT_ID,
        "screening_contract": {
            "path": str(SCREENING_CONTRACT_PATH.resolve()),
            "sha256": screening_contract_digest,
            "status": screening_contract["status"],
            "mechanism_reference_bound": True,
        },
        "design_contract": {
            "path": str(design_contract_path.resolve()),
            "sha256": design_contract_digest,
        },
        "scope": {
            "dataset": "insta_market_basket",
            "seed": 42,
            "target_split": "validation",
            "input_splits_materialized": ["train", "validation"],
            "held_out_test_evaluated": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "parameter_updates_performed": False,
            "calibration_fit_performed": False,
        },
        "source": source_audit,
        "inputs": {
            "data_path": str(args.data.resolve()),
            "data_sha256": data_digest,
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": split_digest,
            "B_cache_path": str(args.b_cache.resolve()),
            "B_cache_sha256": b_cache_digest,
            "candidate": checkpoint_audit,
            "candidate_job_audit": candidate_job_audit,
        },
        "validation_population": population,
        "alignment": {
            "identity_columns": list(IDENTITY_COLUMNS),
            "row_count": paired_output.height,
            "candidate_identity_unique": True,
            "B_identity_unique": True,
            "identity_sets_equal": True,
            "target_quantity_bitwise_equal_after_join": True,
            "history_length_equal_after_join": True,
        },
        "state_identity": frozen_state,
        "candidate_summary_replay": {
            "status": "exact_within_tolerance",
            "absolute_tolerance": 1e-5,
            "absolute_differences": replay_differences,
        },
        "B_cache_replay": {
            "status": "exact_within_tolerance",
            "absolute_tolerance": 1e-9,
            "absolute_differences": b_replay_differences,
        },
        "metrics": metrics,
        "gate": gate,
        "paired_predictions": {
            "path": str(output_paths["paired"].resolve()),
            "sha256": paired_digest,
            "row_count": paired_output.height,
            "column_count": paired_output.width,
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "torch_version": torch.__version__,
            "inference_elapsed_seconds": time.perf_counter() - inference_started,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(cuda_index)
            )
            if device.type == "cuda"
            else 0,
            "pid": os.getpid(),
        },
    }
    decision = {
        "schema": "hard_lmm_level_history_qkv_mechanism_decision_v1",
        "status": gate["status"],
        "contract_id": SCREENING_CONTRACT_ID,
        "routing_contract_id": CONTRACT_ID,
        "screening_contract_sha256": screening_contract_digest,
        "candidate_checkpoint_file_sha256": checkpoint_audit[
            "checkpoint_file_sha256"
        ],
        "candidate_checkpoint_state_sha256": checkpoint_audit[
            "checkpoint_state_sha256"
        ],
        "candidate_summary_file_sha256": checkpoint_audit["summary_file_sha256"],
        "candidate_job_audit_sha256": candidate_job_audit["sha256"],
        "B_cache_sha256": b_cache_digest,
        "paired_predictions_sha256": paired_digest,
        "validation_target_identity_sha256": expected_population[
            "target_identity_sha256"
        ],
        "validation_target_quantity_sha256": expected_population[
            "target_quantity_sha256"
        ],
        "gate": gate,
        "held_out_test_evaluated": False,
        "recorded_at_utc": completed_at,
    }
    save_json(output_paths["audit"], audit)
    save_json(output_paths["decision"], decision)
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--expected-deployment-manifest-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-candidate-checkpoint-sha256", required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--expected-candidate-summary-sha256", required=True)
    parser.add_argument("--candidate-job-audit", type=Path, required=True)
    parser.add_argument("--expected-candidate-job-audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate_status": result["gate"]["status"],
                "audit": str((args.output_dir / "audit.json").resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
