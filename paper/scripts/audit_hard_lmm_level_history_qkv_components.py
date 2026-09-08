#!/usr/bin/env python3
"""Audit frozen LPHC Q/K/V paths on canonical Instacart train targets.

The audit performs no fitting or checkpoint selection.  It rebuilds one fresh
model per predeclared kernel mask from the same selected LPHC checkpoint,
checks that only disabled Q/K/V kernels differ, and evaluates all variants on
the exact Frozen-B train population and series-disjoint folds.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ID = "hard_lmm_level_history_qkv_component_diagnostic_v1"
CONTRACT_SHA256 = "4b8c982450d775ed1b10434f853b45392166b2a3cf95b7e53c62ba8eb1e724d4"
BACKBONE = "titantpp_hard_memory_level_history_qkv"
FOLD_SALT = "hard_lmm_bounded_qk_train_condition_v1:20260908"
VARIANT_ORDER = (
    "residual_off",
    "q_only",
    "k_only",
    "v_only",
    "qk",
    "qv",
    "kv",
    "full_qkv",
)
SINGLETONS = ("q_only", "k_only", "v_only")
HISTORY_BINS = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0])
    require(
        all(list(row) == fieldnames for row in rows),
        f"CSV rows have inconsistent fields: {path}",
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_contract(path: Path) -> dict[str, Any]:
    require(path.is_file(), "Component diagnostic contract is missing")
    require(sha256_file(path) == CONTRACT_SHA256, "Component contract hash drift")
    contract = read_json(path)
    require(contract.get("contract_id") == CONTRACT_ID, "Component contract ID drift")
    require(
        contract.get("status") == "frozen_before_component_inference",
        "Component contract was not prospectively frozen",
    )
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Component scope is missing")
    expected_scope = {
        "dataset": "insta_market_basket",
        "seed": 42,
        "target_split": "train",
        "input_splits_materialized": ["train"],
        "validation_targets": False,
        "held_out_test": False,
        "training": False,
        "checkpoint_selection": False,
        "parameter_updates": False,
        "optimizer_use": False,
        "calibration_fit": False,
    }
    for name, expected in expected_scope.items():
        require(scope.get(name) == expected, f"Component scope drift: {name}")
    require(tuple(contract.get("variants", {})) == VARIANT_ORDER, "Variant order drift")
    require(
        tuple(contract["analysis"]["single_path_attribution"]["eligible_variants"])
        == SINGLETONS,
        "Singleton attribution set drift",
    )
    require(
        tuple(contract["analysis"]["next_candidate_gate"]["eligible_variants"])
        == SINGLETONS,
        "Candidate singleton set drift",
    )
    require(contract["analysis"]["fold_count"] == 2, "Fold-count drift")
    require(contract["analysis"]["fold_salt"] == FOLD_SALT, "Fold-salt drift")
    require(
        contract["execution"]["fresh_strict_restore_per_variant"] is True,
        "Fresh restore was disabled",
    )
    return contract


def _series_token(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        require(math.isfinite(numeric) and numeric.is_integer(), "Invalid numeric series ID")
        return f"int:{int(numeric)}"
    require(isinstance(value, str) and bool(value), "Invalid string series ID")
    return f"str:{value}"


def assign_series_folds(series_ids: Sequence[Any], *, salt: str = FOLD_SALT) -> np.ndarray:
    values = np.asarray(series_ids, dtype=object)
    require(values.ndim == 1, "Series IDs must have shape [N]")
    require(isinstance(salt, str) and bool(salt), "Fold salt must be nonempty")
    cache: dict[str, int] = {}
    folds = np.empty(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        token = _series_token(value)
        fold = cache.get(token)
        if fold is None:
            digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
            fold = int.from_bytes(digest[:8], "big") % 2
            cache[token] = fold
        folds[index] = fold
    return folds


def quantity_metrics(target: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    require(
        target_array.ndim == 1 and prediction_array.shape == target_array.shape,
        "Metric arrays must share shape [N]",
    )
    require(len(target_array) > 0, "Metrics require at least one row")
    require(
        bool(np.isfinite(target_array).all()) and bool(np.isfinite(prediction_array).all()),
        "Metrics require finite arrays",
    )
    error = prediction_array - target_array
    bias = float(error.mean(dtype=np.float64))
    mse = float(np.square(error).mean(dtype=np.float64))
    centered_mse = float(np.square(error - bias).mean(dtype=np.float64))
    bias_squared = bias * bias
    residual = mse - centered_mse - bias_squared
    tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, abs(mse), abs(centered_mse), abs(bias_squared)
    )
    require(abs(residual) <= tolerance, "MSE bias decomposition failed")
    return {
        "count": int(len(target_array)),
        "mae": float(np.abs(error).mean(dtype=np.float64)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias": bias,
        "centered_mse": centered_mse,
        "centered_rmse": math.sqrt(centered_mse),
        "bias_squared": bias_squared,
        "prediction_mean": float(prediction_array.mean(dtype=np.float64)),
        "decomposition_residual": residual,
        "decomposition_tolerance": tolerance,
    }


def compare_predictions(
    target: Sequence[float], candidate: Sequence[float], reference: Sequence[float]
) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    require(
        candidate_array.shape == reference_array.shape == target_array.shape,
        "Comparison arrays must align",
    )
    candidate_metrics = quantity_metrics(target_array, candidate_array)
    reference_metrics = quantity_metrics(target_array, reference_array)
    return {
        "candidate": candidate_metrics,
        "reference": reference_metrics,
        "delta_mae": candidate_metrics["mae"] - reference_metrics["mae"],
        "delta_mse": candidate_metrics["mse"] - reference_metrics["mse"],
        "delta_rmse": candidate_metrics["rmse"] - reference_metrics["rmse"],
        "delta_centered_mse": (
            candidate_metrics["centered_mse"] - reference_metrics["centered_mse"]
        ),
        "mean_prediction_shift": float(
            (candidate_array - reference_array).mean(dtype=np.float64)
        ),
    }


def derive_variant_state(
    source_state: Mapping[str, torch.Tensor],
    *,
    mask: Sequence[int],
    kernel_keys: Mapping[str, str],
    canonical_state_dict_sha256: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    switches = tuple(int(value) for value in mask)
    require(len(switches) == 3 and set(switches) <= {0, 1}, "Invalid Q/K/V mask")
    require(set(kernel_keys) == {"Q", "K", "V"}, "Kernel role set drift")
    require(all(key in source_state for key in kernel_keys.values()), "Kernel state is missing")
    result = {name: value.detach().clone() for name, value in source_state.items()}
    expected_changed: set[str] = set()
    rows: dict[str, Any] = {}
    for role, switch in zip(("Q", "K", "V"), switches):
        key = kernel_keys[role]
        original = source_state[key]
        require(isinstance(original, torch.Tensor), f"{role} kernel is not a tensor")
        require(tuple(original.shape) == (2, 64), f"{role} kernel shape drift")
        require(bool(torch.isfinite(original).all()), f"{role} kernel is not finite")
        if switch == 0:
            result[key].zero_()
            if int(original.count_nonzero()) > 0:
                expected_changed.add(key)
        rows[role] = {
            "enabled": bool(switch),
            "shape": list(original.shape),
            "dtype": str(original.dtype),
            "source_nonzero_count": int(original.count_nonzero()),
            "derived_nonzero_count": int(result[key].count_nonzero()),
            "source_retained_bitwise": bool(torch.equal(result[key], original)) if switch else False,
            "disabled_exact_zero": bool(int(result[key].count_nonzero()) == 0) if not switch else False,
        }
    observed_changed = {
        name for name in source_state if not torch.equal(result[name], source_state[name])
    }
    require(observed_changed == expected_changed, "A variant changed an unexpected state tensor")
    non_kernel = {
        name: tensor for name, tensor in result.items() if name not in set(kernel_keys.values())
    }
    source_non_kernel = {
        name: tensor
        for name, tensor in source_state.items()
        if name not in set(kernel_keys.values())
    }
    require(
        all(torch.equal(non_kernel[name], source_non_kernel[name]) for name in non_kernel),
        "A variant changed non-kernel state",
    )
    return result, {
        "mask": list(switches),
        "kernel_checks": rows,
        "changed_state_keys": sorted(observed_changed),
        "derived_state_sha256": canonical_state_dict_sha256(result),
        "common_non_kernel_state_sha256": canonical_state_dict_sha256(non_kernel),
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def import_frozen_runtime(frozen_root: Path) -> dict[str, Any]:
    frozen_root = frozen_root.resolve()
    require(frozen_root.is_dir() and frozen_root != PROJECT_ROOT, "Frozen source root is invalid")
    guarded = (
        "models",
        "data_loader",
        "simple_lab_test.search.common.runner",
        "paper.scripts.count_aware_tpp_backbone",
    )
    for name, module in tuple(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in guarded):
            module_file = getattr(module, "__file__", None)
            if module_file is not None:
                require(
                    _inside(Path(module_file), frozen_root),
                    f"Mutable project module was imported before frozen activation: {name}",
                )
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or Path(entry).resolve() != PROJECT_ROOT.resolve()
    ]
    sys.path.insert(0, str(frozen_root))
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    control = importlib.import_module("paper.scripts.run_count_aware_tpp_backbone_control")
    head = importlib.import_module("paper.scripts.run_matched_frozen_lognormal_duration")
    loader = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
    runner = importlib.import_module("simple_lab_test.search.common.runner")
    factory = importlib.import_module("models.TPPs.CountAwareFactory")
    causal_audit = importlib.import_module("paper.scripts.audit_hard_lmm_causal_qkv")
    for module in (core, control, head, loader, runner, factory, causal_audit):
        require(
            _inside(Path(module.__file__), frozen_root),
            f"Project import escaped frozen source: {module.__name__}",
        )
    return {
        "prepare_count_frame": core.prepare_count_frame,
        "target_outputs": core.target_outputs,
        "exact_target_population": control.exact_target_population,
        "build_source_model": head.build_source_model,
        "make_loader": loader.make_loader,
        "torch_load_checkpoint": runner.torch_load_checkpoint,
        "canonical_state_dict_sha256": runner.canonical_state_dict_sha256,
        "validate_checkpoint_route": factory.validate_checkpoint_route,
        "restore_model": causal_audit._restore_model,
        "module_files": {
            module.__name__: str(Path(module.__file__).resolve())
            for module in (core, control, head, loader, runner, factory, causal_audit)
        },
    }


def verify_frozen_source(
    frozen_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    source = contract["source"]
    observed: dict[str, str] = {}
    for relative, expected in source["required_source_file_sha256"].items():
        path = (frozen_root / relative).resolve()
        require(path.is_file() and _inside(path, frozen_root), f"Frozen source missing: {relative}")
        observed[relative] = sha256_file(path)
        require(observed[relative] == expected, f"Frozen source drift: {relative}")
    manifest = PROJECT_ROOT / source["deployment_manifest_path"]
    require(
        manifest.is_file()
        and sha256_file(manifest) == source["deployment_manifest_sha256"],
        "Pinned deployment manifest drift",
    )
    payload = read_json(manifest)
    require(
        payload.get("source_revision") == source["frozen_model_revision"],
        "Deployment source revision drift",
    )
    for relative, digest in observed.items():
        require(payload["source_files"].get(relative) == digest, f"Deployment file drift: {relative}")
    return {
        "root": str(frozen_root.resolve()),
        "revision": source["frozen_model_revision"],
        "deployment_manifest_path": str(manifest.resolve()),
        "deployment_manifest_sha256": sha256_file(manifest),
        "verified_files": observed,
    }


def load_b_cache(path: Path, contract: Mapping[str, Any]) -> dict[str, np.ndarray]:
    spec = contract["models"]["B"]
    require(path.is_file() and sha256_file(path) == spec["train_cache_file_sha256"], "B cache drift")
    required = {
        "prediction",
        "quantity",
        "time_nll",
        "history_length",
        "series_index",
        "target_index",
        "context_end",
    }
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == required, "B cache schema drift")
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = int(contract["dataset"]["expected_train_targets"])
    require(all(value.shape == (expected,) for value in arrays.values()), "B cache shape drift")
    for name in ("prediction", "quantity", "time_nll"):
        require(bool(np.isfinite(arrays[name]).all()), f"B cache has non-finite {name}")
    require(bool((arrays["prediction"] >= 0).all()), "B cache has negative predictions")
    return arrays


def restore_variants(
    *,
    checkpoint_path: Path,
    summary_path: Path,
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any], dict[str, Any]]:
    spec = contract["models"]["LPHC"]
    require(
        checkpoint_path.is_file()
        and sha256_file(checkpoint_path) == spec["checkpoint_file_sha256"],
        "LPHC checkpoint file drift",
    )
    require(
        summary_path.is_file() and sha256_file(summary_path) == spec["summary_file_sha256"],
        "LPHC summary drift",
    )
    payload = runtime["torch_load_checkpoint"](checkpoint_path, map_location="cpu")
    require(isinstance(payload, dict), "LPHC checkpoint payload is invalid")
    runtime["validate_checkpoint_route"](payload, BACKBONE)
    summary = read_json(summary_path)
    for artifact, value in (("checkpoint", payload), ("summary", summary)):
        require(value.get("backbone") == BACKBONE, f"{artifact} backbone drift")
        require(value.get("seed") == 42, f"{artifact} seed drift")
        require(
            value.get("checkpoint_selection") == "best_validation_raw_quantity_rmse",
            f"{artifact} selector drift",
        )
        require(value.get("evaluation_scope") == "validation_only", f"{artifact} scope drift")
        require(value.get("held_out_test_evaluated") is False, f"{artifact} held-out drift")
    require(payload.get("best_epoch") == summary.get("best_epoch") == spec["best_epoch"], "Epoch drift")
    require(summary.get("completed_epochs") == spec["completed_epochs"], "Completed-epoch drift")
    source_state = payload.get("model_state_dict")
    require(isinstance(source_state, dict), "Selected LPHC state is missing")
    canonical = runtime["canonical_state_dict_sha256"]
    source_digest = canonical(source_state)
    require(
        source_digest
        == payload.get("model_state_sha256")
        == summary.get("checkpoint_state_sha256")
        == spec["checkpoint_state_sha256"],
        "Selected LPHC state digest drift",
    )
    original_model = runtime["build_source_model"](payload, max_seq_len=64)
    require(canonical(original_model.state_dict()) == source_digest, "Original strict replay drift")
    del original_model
    keys = spec["kernel_state_keys"]
    variants: dict[str, torch.nn.Module] = {}
    variant_audit: dict[str, Any] = {}
    common_digest: str | None = None
    for name in VARIANT_ORDER:
        state, audit = derive_variant_state(
            source_state,
            mask=contract["variants"][name],
            kernel_keys=keys,
            canonical_state_dict_sha256=canonical,
        )
        if common_digest is None:
            common_digest = audit["common_non_kernel_state_sha256"]
        require(
            audit["common_non_kernel_state_sha256"] == common_digest,
            f"Non-kernel state drift: {name}",
        )
        model = runtime["restore_model"](
            payload,
            state,
            candidate={"backbone": BACKBONE},
            artifact=f"LPHC component {name}",
        )
        model.requires_grad_(False).eval().to(device)
        require(
            canonical(model.state_dict()) == audit["derived_state_sha256"],
            f"Variant strict restore digest drift: {name}",
        )
        variants[name] = model
        variant_audit[name] = audit
    return variants, variant_audit, {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "summary_path": str(summary_path.resolve()),
        "summary_file_sha256": sha256_file(summary_path),
        "source_state_sha256": source_digest,
        "common_non_kernel_state_sha256": common_digest,
    }


def _history_label_mask(history: np.ndarray, label: str) -> np.ndarray:
    for name, lower, upper in HISTORY_BINS:
        if name == label:
            return (history >= lower) & (history <= upper)
    raise KeyError(label)


def analyze_predictions(
    *,
    target: np.ndarray,
    history: np.ndarray,
    folds: np.ndarray,
    b_prediction: np.ndarray,
    variants: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = {"B": b_prediction, **variants}
    metrics: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for model, prediction in predictions.items():
        metrics[model] = {"pooled": quantity_metrics(target, prediction), "folds": {}, "history": {}}
        for fold in (0, 1):
            fold_mask = folds == fold
            observed = quantity_metrics(target[fold_mask], prediction[fold_mask])
            metrics[model]["folds"][str(fold)] = observed
            comparison_b = compare_predictions(
                target[fold_mask], prediction[fold_mask], b_prediction[fold_mask]
            )
            comparison_off = compare_predictions(
                target[fold_mask], prediction[fold_mask], variants["residual_off"][fold_mask]
            )
            fold_rows.append(
                {
                    "model": model,
                    "fold": fold,
                    **{key: observed[key] for key in (
                        "count", "mae", "mse", "rmse", "bias", "centered_mse", "prediction_mean"
                    )},
                    "delta_mae_vs_B": comparison_b["delta_mae"],
                    "delta_mse_vs_B": comparison_b["delta_mse"],
                    "delta_centered_mse_vs_B": comparison_b["delta_centered_mse"],
                    "mean_shift_vs_B": comparison_b["mean_prediction_shift"],
                    "delta_mae_vs_residual_off": comparison_off["delta_mae"],
                    "delta_mse_vs_residual_off": comparison_off["delta_mse"],
                    "delta_centered_mse_vs_residual_off": comparison_off["delta_centered_mse"],
                    "mean_shift_vs_residual_off": comparison_off["mean_prediction_shift"],
                }
            )
        for label, _, _ in HISTORY_BINS:
            metrics[model]["history"][label] = {}
            label_mask = _history_label_mask(history, label)
            for scope, scope_mask in (("pooled", np.ones(len(target), dtype=bool)), ("fold_0", folds == 0), ("fold_1", folds == 1)):
                selected = label_mask & scope_mask
                if not bool(selected.any()):
                    metrics[model]["history"][label][scope] = None
                    continue
                observed = quantity_metrics(target[selected], prediction[selected])
                metrics[model]["history"][label][scope] = observed
                comparison_b = compare_predictions(target[selected], prediction[selected], b_prediction[selected])
                comparison_off = compare_predictions(
                    target[selected], prediction[selected], variants["residual_off"][selected]
                )
                history_rows.append(
                    {
                        "model": model,
                        "scope": scope,
                        "history": label,
                        **{key: observed[key] for key in (
                            "count", "mae", "mse", "rmse", "bias", "centered_mse", "prediction_mean"
                        )},
                        "delta_mae_vs_B": comparison_b["delta_mae"],
                        "delta_mse_vs_B": comparison_b["delta_mse"],
                        "delta_centered_mse_vs_B": comparison_b["delta_centered_mse"],
                        "mean_shift_vs_B": comparison_b["mean_prediction_shift"],
                        "delta_mae_vs_residual_off": comparison_off["delta_mae"],
                        "delta_mse_vs_residual_off": comparison_off["delta_mse"],
                        "delta_centered_mse_vs_residual_off": comparison_off["delta_centered_mse"],
                        "mean_shift_vs_residual_off": comparison_off["mean_prediction_shift"],
                    }
                )

    full_fold_gates: dict[str, Any] = {}
    for fold in (0, 1):
        scope = f"fold_{fold}"
        h23_full = metrics["full_qkv"]["history"]["2-3"][scope]["mse"]
        h23_b = metrics["B"]["history"]["2-3"][scope]["mse"]
        h815_full = metrics["full_qkv"]["history"]["8-15"][scope]["mse"]
        h815_b = metrics["B"]["history"]["8-15"][scope]["mse"]
        conditions = {
            "full_qkv_history_2_3_mse_worse_than_B": h23_full > h23_b,
            "full_qkv_history_8_15_mse_better_than_B": h815_full < h815_b,
        }
        full_fold_gates[str(fold)] = {
            "conditions": conditions,
            "passed": all(conditions.values()),
            "delta_history_2_3_mse_vs_B": h23_full - h23_b,
            "delta_history_8_15_mse_vs_B": h815_full - h815_b,
        }
    full_pattern_replicated = all(row["passed"] for row in full_fold_gates.values())

    attribution: dict[str, Any] = {}
    absent_pair = {"q_only": "kv", "k_only": "qv", "v_only": "qk"}
    for variant in SINGLETONS:
        singleton_short = []
        singleton_long = []
        singleton_shift = []
        necessary_short = []
        necessary_long = []
        for fold in (0, 1):
            scope = f"fold_{fold}"
            singleton_short.append(
                metrics[variant]["history"]["2-3"][scope]["mse"]
                > metrics["residual_off"]["history"]["2-3"][scope]["mse"]
            )
            singleton_long.append(
                metrics[variant]["history"]["8-15"][scope]["mse"]
                < metrics["residual_off"]["history"]["8-15"][scope]["mse"]
            )
            singleton_shift.append(
                metrics[variant]["folds"][str(fold)]["prediction_mean"]
                < metrics["residual_off"]["folds"][str(fold)]["prediction_mean"]
            )
            pair = absent_pair[variant]
            necessary_short.append(
                metrics["full_qkv"]["history"]["2-3"][scope]["mse"]
                > metrics[pair]["history"]["2-3"][scope]["mse"]
            )
            necessary_long.append(
                metrics["full_qkv"]["history"]["8-15"][scope]["mse"]
                < metrics[pair]["history"]["8-15"][scope]["mse"]
            )
        attribution[variant] = {
            "consistent_singleton_short_history_harm": all(singleton_short),
            "consistent_singleton_long_history_benefit": all(singleton_long),
            "consistent_singleton_negative_level_shift": all(singleton_shift),
            "consistent_full_leaveout_short_history_harm_contribution": all(necessary_short),
            "consistent_full_leaveout_long_history_benefit_contribution": all(necessary_long),
            "absent_pair_control": absent_pair[variant],
        }

    singleton_gates: dict[str, Any] = {}
    eligible: list[str] = []
    for variant in SINGLETONS:
        fold_gates: dict[str, Any] = {}
        for fold in (0, 1):
            scope = f"fold_{fold}"
            candidate_fold = metrics[variant]["folds"][str(fold)]
            off_fold = metrics["residual_off"]["folds"][str(fold)]
            b_fold = metrics["B"]["folds"][str(fold)]
            h23_candidate = metrics[variant]["history"]["2-3"][scope]
            h23_off = metrics["residual_off"]["history"]["2-3"][scope]
            h23_b = metrics["B"]["history"]["2-3"][scope]
            h815_candidate = metrics[variant]["history"]["8-15"][scope]
            h815_off = metrics["residual_off"]["history"]["8-15"][scope]
            h815_b = metrics["B"]["history"]["8-15"][scope]
            vs_off = {
                "overall_raw_mse_not_worse": candidate_fold["mse"] <= off_fold["mse"],
                "overall_mae_not_worse": candidate_fold["mae"] <= off_fold["mae"],
                "overall_centered_mse_not_worse": candidate_fold["centered_mse"] <= off_fold["centered_mse"],
                "history_2_3_mse_not_worse": h23_candidate["mse"] <= h23_off["mse"],
                "history_8_15_mse_improves": h815_candidate["mse"] < h815_off["mse"],
                "level_shift_bounded": abs(candidate_fold["prediction_mean"] - off_fold["prediction_mean"]) <= 0.01 * b_fold["rmse"],
            }
            vs_b = {
                "overall_raw_mse_not_worse": candidate_fold["mse"] <= b_fold["mse"],
                "overall_mae_not_worse": candidate_fold["mae"] <= b_fold["mae"],
                "overall_centered_mse_not_worse": candidate_fold["centered_mse"] <= b_fold["centered_mse"],
                "history_2_3_mse_not_worse": h23_candidate["mse"] <= h23_b["mse"],
                "history_8_15_mse_improves": h815_candidate["mse"] < h815_b["mse"],
            }
            fold_gates[str(fold)] = {
                "vs_residual_off": vs_off,
                "vs_B": vs_b,
                "passed": all(vs_off.values()) and all(vs_b.values()),
            }
        passed = all(row["passed"] for row in fold_gates.values())
        singleton_gates[variant] = {"folds": fold_gates, "passed": passed}
        if passed:
            eligible.append(variant)

    rank_order = {"q_only": 0, "k_only": 1, "v_only": 2}
    eligible.sort(
        key=lambda name: (
            metrics[name]["pooled"]["mse"],
            metrics[name]["history"]["8-15"]["pooled"]["mse"],
            rank_order[name],
        )
    )
    selected = eligible[0] if full_pattern_replicated and eligible else None
    decision = {
        "full_pattern_replicated": full_pattern_replicated,
        "full_pattern_fold_gates": full_fold_gates,
        "conditional_path_flags": attribution,
        "singleton_candidate_gates": singleton_gates,
        "eligible_single_paths": eligible,
        "selected_path_or_null": selected,
        "continue_or_stop_LPHC_family": (
            "freeze_one_smaller_common_confidence_candidate"
            if selected is not None
            else "stop_LPHC_QKV_family"
        ),
    }
    return {"metrics": metrics, "decision": decision}, fold_rows, history_rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    contract = validate_contract(args.contract)
    device = torch.device(args.device)
    require(device.type == "cpu", "The prospective local diagnostic requires CPU")
    source_audit = verify_frozen_source(args.frozen_source_root, contract)
    runtime = import_frozen_runtime(args.frozen_source_root)
    b_cache = load_b_cache(args.b_cache, contract)
    require(
        args.b_result.is_file()
        and sha256_file(args.b_result) == contract["models"]["B"]["train_result_file_sha256"],
        "B result audit drift",
    )
    require(
        args.data.is_file() and sha256_file(args.data) == contract["dataset"]["data_sha256"],
        "Instacart data drift",
    )
    require(
        args.split_manifest.is_file()
        and sha256_file(args.split_manifest) == contract["dataset"]["split_manifest_sha256"],
        "Instacart split manifest drift",
    )
    frame = (
        pl.scan_parquet(args.data)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(set(frame["chronological_split"].unique().to_list()) == {"train"}, "Non-train row admitted")
    require(frame.height == contract["dataset"]["expected_train_rows"], "Train row-count drift")
    require(frame["oper_part_no"].n_unique() == contract["dataset"]["expected_train_series"], "Train series-count drift")
    frame = runtime["prepare_count_frame"](frame)
    target_quantity, population = runtime["exact_target_population"](
        frame,
        target_split="train",
        lookback_weeks=52,
        max_seq_len=64,
    )
    for name, expected in {
        "target_count": contract["dataset"]["expected_train_targets"],
        "target_identity_sha256": contract["dataset"]["expected_train_identity_sha256"],
        "target_quantity_sha256": contract["dataset"]["expected_train_quantity_sha256"],
    }.items():
        require(population.get(name) == expected, f"Train population drift: {name}")
    require(np.array_equal(target_quantity, b_cache["quantity"]), "B cache target order drift")
    loader = runtime["make_loader"](
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=52,
        max_seq_len=64,
        shuffle=False,
        generator=None,
    )
    dataset = loader.dataset
    count = len(dataset)
    require(count == contract["dataset"]["expected_train_targets"], "Train loader count drift")
    models, variant_audit, checkpoint_audit = restore_variants(
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        contract=contract,
        runtime=runtime,
        device=device,
    )
    canonical = runtime["canonical_state_dict_sha256"]
    state_before = {name: canonical(model.state_dict()) for name, model in models.items()}
    checkpoint_before = sha256_file(args.checkpoint)
    predictions = {name: np.empty(count, dtype=np.float64) for name in VARIANT_ORDER}
    history = np.empty(count, dtype=np.int64)
    row_series = np.empty(count, dtype=np.int64)
    first_batch_invariance: dict[str, bool] = {}
    offset = 0
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (_, dts, mask, parts, quantities) in enumerate(loader):
            require(quantities is not None, "Train quantities are missing")
            batch_count = int(dts.size(0))
            end = offset + batch_count
            parts_numpy = parts.numpy().astype(np.int64)
            require(np.array_equal(parts_numpy, b_cache["series_index"][offset:end]), "Series order drift")
            row_series[offset:end] = parts_numpy
            expected_true = b_cache["quantity"][offset:end]
            expected_history = b_cache["history_length"][offset:end]
            for name in VARIANT_ORDER:
                output = runtime["target_outputs"](
                    models[name], dts, mask, quantities, lambda_log_qty=1.0
                )
                true = output["true_qty"].detach().cpu().numpy().astype(np.float64)
                observed_history = output["history_length"].detach().cpu().numpy().astype(np.int64)
                require(np.array_equal(true, expected_true), f"Target order drift: {name}")
                require(np.array_equal(observed_history, expected_history), f"History order drift: {name}")
                prediction = output["pred_qty"].detach().cpu().numpy().astype(np.float64)
                require(bool(np.isfinite(prediction).all()) and bool((prediction >= 0).all()), f"Invalid prediction: {name}")
                predictions[name][offset:end] = prediction
                if batch_index == 0:
                    changed_dts = dts.clone()
                    changed_quantities = quantities.clone()
                    positions = torch.arange(mask.size(1)).expand_as(mask)
                    target_positions = torch.where(mask, positions, -1).max(dim=1).values
                    rows = torch.arange(mask.size(0))
                    changed_dts[rows, target_positions] += 10007.0
                    changed_quantities[rows, target_positions] += 20011.0
                    changed_dts[~mask] = 30013.0
                    changed_quantities[~mask] = 40009.0
                    changed_output = runtime["target_outputs"](
                        models[name], changed_dts, mask, changed_quantities, lambda_log_qty=1.0
                    )
                    first_batch_invariance[name] = bool(
                        torch.equal(output["pred_qty"], changed_output["pred_qty"])
                    )
                    require(first_batch_invariance[name], f"Target/padding leaked into prediction: {name}")
            history[offset:end] = expected_history
            offset = end
            if batch_index % 50 == 0 or offset == count:
                print(
                    json.dumps(
                        {
                            "stage": "lphc_component_train_inference",
                            "completed_targets": offset,
                            "total_targets": count,
                            "variants": len(VARIANT_ORDER),
                            "elapsed_seconds": time.perf_counter() - inference_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    require(offset == count, "Component inference is incomplete")
    state_after = {name: canonical(model.state_dict()) for name, model in models.items()}
    require(state_after == state_before, "A variant state changed during inference")
    require(
        all(parameter.grad is None for model in models.values() for parameter in model.parameters()),
        "A frozen variant accumulated gradients",
    )
    require(sha256_file(args.checkpoint) == checkpoint_before, "LPHC checkpoint changed")
    series_parts = np.asarray(dataset.parts, dtype=object)
    series_ids = series_parts[row_series]
    folds = assign_series_folds(series_ids)
    expected_fold_targets = contract["analysis"]["expected_fold_target_counts"]
    expected_fold_series = contract["analysis"]["expected_fold_series_counts"]
    for fold in (0, 1):
        selected = folds == fold
        require(int(selected.sum()) == expected_fold_targets[str(fold)], f"Fold {fold} target-count drift")
        require(len(set(series_ids[selected].tolist())) == expected_fold_series[str(fold)], f"Fold {fold} series-count drift")
    require(
        int(history.min()) == contract["dataset"]["expected_history_length_min"]
        and int(history.max()) == contract["dataset"]["expected_history_length_max"],
        "History range drift",
    )
    analysis, fold_rows, history_rows = analyze_predictions(
        target=b_cache["quantity"],
        history=history,
        folds=folds,
        b_prediction=b_cache["prediction"],
        variants=predictions,
        contract=contract,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "component_train_predictions.npz"
    atomic_npz(
        prediction_path,
        {
            "quantity": b_cache["quantity"],
            "history_length": history,
            "series_index": row_series,
            "fold": folds,
            **{f"prediction__{name}": predictions[name] for name in VARIANT_ORDER},
        },
    )
    completed = utc_now()
    analysis_payload = {
        "schema": "hard_lmm_level_history_qkv_component_analysis_v1",
        "status": "complete",
        "created_at_utc": completed,
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "scope": contract["scope"],
        **analysis,
    }
    atomic_json(args.output_dir / "analysis.json", analysis_payload)
    write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    write_csv(args.output_dir / "history_metrics.csv", history_rows)
    audit = {
        "schema": "hard_lmm_level_history_qkv_component_audit_v1",
        "status": "passed",
        "created_at_utc": completed,
        "contract": {
            "path": str(args.contract.resolve()),
            "sha256": sha256_file(args.contract),
            "frozen_commit": "74123e5",
        },
        "scope": contract["scope"],
        "data": {
            "path": str(args.data.resolve()),
            "sha256": sha256_file(args.data),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "materialized_splits": ["train"],
            "train_rows": frame.height,
            "target_population": population,
            "fold_target_counts": {str(fold): int((folds == fold).sum()) for fold in (0, 1)},
            "fold_series_counts": {
                str(fold): len(set(series_ids[folds == fold].tolist())) for fold in (0, 1)
            },
        },
        "frozen_source": source_audit,
        "frozen_runtime_module_files": runtime["module_files"],
        "B_cache": {
            "path": str(args.b_cache.resolve()),
            "sha256": sha256_file(args.b_cache),
            "result_path": str(args.b_result.resolve()),
            "result_sha256": sha256_file(args.b_result),
        },
        "LPHC_checkpoint": checkpoint_audit,
        "variants": variant_audit,
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_unchanged": True,
        "all_gradients_absent": True,
        "first_batch_target_and_padding_prediction_invariance": first_batch_invariance,
        "prediction_cache": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
            "git_tracked": False,
        },
        "outputs": {
            name: {
                "path": str((args.output_dir / name).resolve()),
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in ("analysis.json", "fold_metrics.csv", "history_metrics.csv")
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "batch_size": args.batch_size,
            "elapsed_seconds": time.perf_counter() - started,
            "inference_elapsed_seconds": time.perf_counter() - inference_started,
        },
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "validation_targets_evaluated": False,
        "held_out_test_evaluated": False,
    }
    atomic_json(args.output_dir / "audit.json", audit)
    return {"audit": audit, "analysis": analysis_payload}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "paper/contracts/hard_lmm_level_history_qkv_component_diagnostic_v1.json",
    )
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    require(args.batch_size == 512, "Batch size must match the prospective contract")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["audit"]["status"],
                "decision": result["analysis"]["decision"]["continue_or_stop_LPHC_family"],
                "selected_path": result["analysis"]["decision"]["selected_path_or_null"],
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
