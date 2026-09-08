#!/usr/bin/env python3
"""Extract bounded train-only fixed-checkpoint causal-Q/K/V path evidence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


EXPECTED_CONTRACT_SHA256 = "a4ae310d254ed124fc8be4c340a10d22c462a33fb08d669afd38504cf978ddea"
KERNEL_KEYS = {
    "Q": "encoder.layers.0.attn.causal_q_kernel",
    "K": "encoder.layers.0.attn.causal_k_kernel",
    "V": "encoder.layers.0.attn.causal_v_kernel",
}
INVARIANT_FIELDS = (
    "H1", "H2", "fused_state", "memory_residual", "top4_indices", "prediction",
    "time_intercept_raw", "time_intercept_clamped",
)
VARIANT_FIELDS = (
    "prediction", "log_mse", "absolute_error", "squared_error", "legacy_time_loss",
    "time_intercept_raw", "time_intercept_clamped", "exp_intercept", "w", "wd_raw",
    "wd_clamped", "integral_term", "wd_saturated", "intercept_saturated", "H1", "H2",
    "fused_state", "memory_residual", "top4_indices",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        require(isinstance(tensor, torch.Tensor), f"State entry is not a tensor: {name}")
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, byteorder="big", signed=False))
        digest.update(metadata)
        if value.numel() > 0:
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, byteorder="big", signed=False))
            digest.update(raw)
        else:
            digest.update((0).to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def canonical_series_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and math.isfinite(float(value)):
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".17g")
    text = str(value)
    require(bool(text), "Canonical series ID is empty")
    return text


def fold_for_series(value: Any) -> int:
    payload = ("qkv-path-v1:" + canonical_series_id(value)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2


def validate_contract(contract: Mapping[str, Any], contract_path: Path) -> None:
    require(sha256_file(contract_path) == EXPECTED_CONTRACT_SHA256, "Diagnostic contract hash drift")
    require(contract.get("contract_id") == "hard_lmm_qkv_path_diagnostic_v1", "Contract ID drift")
    scope = contract["scope"]
    require(scope == {
        "split": "train", "lazy_filter_before_collect": True,
        "validation_materialized": False, "held_out_materialized": False,
        "model_or_optimizer_updates": False, "head_refit": False,
        "remote_access": False, "device": "cpu", "torch_threads": 4,
        "batch_size": 16,
    }, "Diagnostic scope drift")
    sampling = contract["sampling"]
    require(sampling["seed"] == 20260908 and sampling["targets_per_dataset"] == 4096,
            "Sampling seed/count drift")
    require(sampling["targets_per_fold"] == 2048 and sampling["fold_count"] == 2,
            "Fold sample contract drift")
    require(sampling["fold_rule"]
            == "sha256('qkv-path-v1:'+canonical_series_id)_first8_big_endian_mod2",
            "Fold hash contract drift")
    expected_variants = {
        "B": "independent frozen B checkpoint, unchanged", "FULL": [1, 1, 1],
        "QK": [1, 1, 0], "QV": [1, 0, 1], "KV": [0, 1, 1],
        "Q": [1, 0, 0], "K": [0, 1, 0], "V": [0, 0, 1], "ZERO": [0, 0, 0],
    }
    require(contract["variants"] == expected_variants, "Variant masks drift")
    require(contract["variant_order"] == list(expected_variants), "Variant order drift")
    require(contract["gradient_diagnostic"]["variants"] == ["B", "FULL"], "Gradient variants drift")
    require(contract["gradient_diagnostic"]["batches_per_fold"] == 4, "Gradient budget drift")
    require(contract["source"]["reject_model_imports_outside_frozen_root"] is True,
            "Frozen import requirement disabled")


def load_train_frame(path: Path) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(frame.height > 0, "Train frame is empty")
    require(set(frame["chronological_split"].unique().to_list()) == {"train"},
            "A non-train row was materialized")
    return frame


@dataclass(frozen=True)
class FrozenModules:
    prepare_count_frame: Any
    right_pad_batch: Any
    target_outputs: Any
    Dataset: Any
    collate: Any
    checkpoint_load: Any
    restore_model: Any


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def import_frozen_modules(repo_root: Path, frozen_root: Path) -> FrozenModules:
    """Defer every project import and reject already-loaded mutable model code."""
    repo_root, frozen_root = repo_root.resolve(), frozen_root.resolve()
    require(frozen_root.is_dir() and frozen_root != repo_root, "Frozen source root is invalid")
    guarded = ("models", "data_loader", "simple_lab_test.search.common.runner",
               "paper.scripts.count_aware_tpp_backbone", "paper.scripts.audit_hard_lmm_causal_qkv")
    for name, module in tuple(sys.modules.items()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in guarded):
            module_file = getattr(module, "__file__", None)
            if module_file is not None:
                require(_inside(Path(module_file), frozen_root),
                        f"Mutable project module was imported before frozen activation: {name}")
    sys.path[:] = [entry for entry in sys.path if not entry or Path(entry).resolve() != repo_root]
    sys.path.insert(0, str(frozen_root))
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    loader = importlib.import_module("data_loader.event_seq_data_module")
    runner = importlib.import_module("simple_lab_test.search.common.runner")
    audit = importlib.import_module("paper.scripts.audit_hard_lmm_causal_qkv")
    for module in (core, loader, runner, audit):
        require(_inside(Path(module.__file__), frozen_root), f"Project import escaped frozen root: {module.__name__}")
    return FrozenModules(
        prepare_count_frame=core.prepare_count_frame,
        right_pad_batch=core.right_pad_batch,
        target_outputs=core.target_outputs,
        Dataset=loader.RMTPPWeekLookbackDataset,
        collate=loader.collate_week_lookback,
        checkpoint_load=runner.torch_load_checkpoint,
        restore_model=audit._restore_model,
    )


def verify_frozen_source(contract: Mapping[str, Any], frozen_root: Path) -> dict[str, Any]:
    source = contract["source"]
    manifest = frozen_root / source["pinned_manifest_path"]
    require(manifest.is_file(), "Pinned Python manifest is missing from frozen source")
    require(sha256_file(manifest) == source["pinned_manifest_sha256"], "Pinned manifest hash drift")
    payload = read_json(manifest)
    files = payload.get("files")
    require(isinstance(files, dict) and len(files) == source["pinned_python_count"],
            "Pinned Python file count drift")
    for relative, expected in files.items():
        path = frozen_root / relative
        require(path.is_file() and sha256_file(path) == expected, f"Frozen Python drift: {relative}")
    return {
        "frozen_implementation_commit": source["frozen_implementation_commit"],
        "archive_sha256": source["archive_sha256"],
        "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest),
        "verified_python_count": len(files),
    }


def dataset_row(contract: Mapping[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in contract["datasets"] if row["dataset"] == name]
    require(len(rows) == 1, f"Dataset row missing or duplicated: {name}")
    return rows[0]


def sample_train_targets(dataset: Any, sampling: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    per_fold = int(sampling["targets_per_fold"])
    selected: dict[int, list[int]] = {0: [], 1: []}
    generator = np.random.default_rng(int(sampling["seed"]))
    for index in generator.permutation(len(dataset)).tolist():
        part_index, _ = dataset.index[int(index)]
        fold = fold_for_series(dataset.parts[part_index])
        if len(selected[fold]) < per_fold:
            selected[fold].append(int(index))
        if all(len(values) == per_fold for values in selected.values()):
            break
    require(all(len(values) == per_fold for values in selected.values()), "Insufficient targets per fold")
    indices = np.asarray(selected[0] + selected[1], dtype=np.int64)
    rows = []
    for index in indices.tolist():
        part_index, context_end = dataset.index[index]
        series = canonical_series_id(dataset.parts[part_index])
        rows.append({
            "dataset_index": index, "series_id": series, "fold": fold_for_series(series),
            "context_end": int(context_end),
            "target_seq": int(dataset.seq_lists[part_index][context_end + 1]),
        })
    for fold in (0, 1):
        fold_rows = [row for row in rows if row["fold"] == fold]
        require(len(fold_rows) == per_fold, f"Fold {fold} target count drift")
        require(len({row["series_id"] for row in fold_rows})
                >= int(sampling["minimum_unique_series_per_fold"]),
                f"Fold {fold} has too few unique series")
    identity_bytes = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 1, "status": "sample_ids_frozen_before_model_outputs",
        "selection_seed": int(sampling["seed"]), "rows": rows,
        "target_count": len(rows),
        "fold_counts": {str(fold): sum(row["fold"] == fold for row in rows) for fold in (0, 1)},
        "fold_unique_series": {
            str(fold): len({row["series_id"] for row in rows if row["fold"] == fold})
            for fold in (0, 1)
        },
        "identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
    }
    return indices, manifest


def duration_quartiles(dataset: Any) -> list[float]:
    values = np.fromiter(
        (float(dataset.dt_lists[p][context + 1]) for p, context in dataset.index),
        dtype=np.float64, count=len(dataset),
    )
    require(values.size == len(dataset) and np.isfinite(values).all(), "Invalid train durations")
    return [float(value) for value in np.quantile(values, [0.25, 0.5, 0.75])]


def legacy_time_telemetry(
    intercept_raw: torch.Tensor,
    w_raw: torch.Tensor,
    duration: torch.Tensor,
    *,
    intercept_limit: float,
) -> dict[str, torch.Tensor]:
    raw = intercept_raw.to(torch.float64)
    dt = duration.to(torch.float64)
    # The production legacy head computes the positive slope in float32.
    # Convert that exact value to float64 for the descriptive algebra below.
    w = (F.softplus(w_raw.to(torch.float32)) + 1e-3).to(torch.float64)
    clamped = torch.clamp(raw, max=float(intercept_limit))
    exp_intercept = torch.exp(clamped)
    wd_raw = w * dt
    wd_clamped = torch.clamp(wd_raw, max=10.0)
    integral = (exp_intercept / w) * torch.expm1(wd_clamped)
    result = {
        "time_intercept_raw": raw, "time_intercept_clamped": clamped,
        "exp_intercept": exp_intercept, "w": w.expand_as(raw), "wd_raw": wd_raw,
        "wd_clamped": wd_clamped, "integral_term": integral,
        "wd_saturated": wd_raw > 10.0,
        "intercept_saturated": raw > float(intercept_limit),
    }
    require(all(bool(torch.isfinite(value).all()) for key, value in result.items()
                if key not in {"wd_saturated", "intercept_saturated"}),
            "Non-finite float64 time telemetry")
    return result


def extract_variant_batch(model: Any, dts: torch.Tensor, mask: torch.Tensor,
                          quantities: torch.Tensor, frozen: FrozenModules) -> dict[str, torch.Tensor]:
    dts, quantities, mask, lengths = frozen.right_pad_batch(dts, quantities, mask)
    rows = torch.arange(dts.size(0))
    target_positions, history_positions = lengths - 1, lengths - 2
    history_quantities = quantities.masked_fill(~mask, 0.0).clone()
    history_quantities[rows, target_positions] = 0.0
    writes = mask.clone()
    writes[rows, target_positions] = False
    features = model.continuous_features(dts, history_quantities, mask)
    encoder = model.encoder
    require(encoder is not None and len(encoder.layers) == 2, "Expected two-layer encoder")
    hidden = encoder.input_proj(features)
    if encoder.use_pos_emb:
        hidden = hidden + encoder._get_pos(hidden.size(1), hidden.device, hidden.dtype)
    hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    H1 = encoder.layers[0](hidden, mask=mask)
    H2 = encoder.layers[1](H1, mask=mask)
    require(model.lmm is not None and model.lmm.topk == 4, "Expected Hard-LMM top-4")
    residual, trace = model.lmm.retrieve(H2)
    fused = (H2 + residual) * mask.unsqueeze(-1).to(H2.dtype)
    selected = lambda value: value[rows, history_positions]
    final = selected(fused)
    prediction = model.predict_quantity(final)[1]
    true_quantity = quantities[rows, target_positions].float()
    true_duration = dts[rows, target_positions].float()
    target_log = torch.log1p(true_quantity.clamp_min(0.0))
    pred_log = model.predict_quantity(final)[0]
    legacy = -model.log_f_dt(final, true_duration)
    intercept32 = model.v_t(final).squeeze(-1) + model.b_t
    telemetry = legacy_time_telemetry(
        intercept32, model.w_raw, true_duration, intercept_limit=float(model.time_intercept_limit),
    )
    result = {
        "prediction": prediction, "true_quantity": true_quantity, "true_duration": true_duration,
        "log_mse": torch.square(pred_log - target_log),
        "absolute_error": torch.abs(prediction - true_quantity),
        "squared_error": torch.square(prediction - true_quantity),
        "legacy_time_loss": legacy,
        "H1": selected(H1), "H2": selected(H2), "fused_state": final,
        "memory_residual": selected(residual),
        "top4_indices": selected(trace["prototype_indices"]),
        **telemetry,
    }
    for name, value in result.items():
        if value.dtype != torch.bool:
            require(bool(torch.isfinite(value).all()), f"Non-finite extraction field: {name}")
    return result


def assert_official_parity(model: Any, dts: torch.Tensor, mask: torch.Tensor,
                           quantities: torch.Tensor, extracted: Mapping[str, torch.Tensor],
                           frozen: FrozenModules) -> dict[str, float]:
    official = frozen.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    comparisons = {
        "prediction": (extracted["prediction"], official["pred_qty"]),
        "log_mse": (extracted["log_mse"], official["log_qty_loss"]),
        "legacy_time_loss": (extracted["legacy_time_loss"], official["time_loss"]),
    }
    differences = {}
    for name, (left, right) in comparisons.items():
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        differences[name] = float((left.double() - right.double()).abs().max())
    return differences


def assert_target_padding_invariance(model: Any, dts: torch.Tensor, mask: torch.Tensor,
                                     quantities: torch.Tensor, extracted: Mapping[str, torch.Tensor],
                                     frozen: FrozenModules) -> dict[str, float]:
    changed_dts, changed_quantities = dts.clone(), quantities.clone()
    row_ids = torch.arange(mask.size(0))
    positions = torch.arange(mask.size(1)).expand_as(mask)
    targets = torch.where(mask, positions, -1).max(dim=1).values
    changed_dts[row_ids, targets] += 10007.0
    changed_quantities[row_ids, targets] += 20011.0
    changed_dts[~mask] = 30013.0
    changed_quantities[~mask] = 40009.0
    changed = extract_variant_batch(model, changed_dts, mask, changed_quantities, frozen)
    differences = {}
    for name in INVARIANT_FIELDS:
        left, right = extracted[name], changed[name]
        require(torch.equal(left, right), f"Forbidden target/padding changed {name}")
        differences[name] = 0.0
    return differences


@contextmanager
def causal_kernel_mask(model: Any, mask: Iterable[int]):
    switches = tuple(int(value) for value in mask)
    require(len(switches) == 3 and set(switches) <= {0, 1}, "Invalid Q/K/V switch")
    parameters = dict(model.named_parameters())
    originals = {name: parameters[key].detach().clone() for name, key in KERNEL_KEYS.items()}
    try:
        with torch.no_grad():
            for switch, name in zip(switches, ("Q", "K", "V")):
                parameter = parameters[KERNEL_KEYS[name]]
                require(tuple(parameter.shape)[0] == 3, f"{name} kernel lacks all three rows")
                if not switch:
                    parameter.zero_()
                require(switch or int(parameter.count_nonzero()) == 0,
                        f"Disabled {name} kernel was not fully zeroed")
        yield
    finally:
        with torch.no_grad():
            for name, value in originals.items():
                parameters[KERNEL_KEYS[name]].copy_(value)


def parameter_groups(model: Any) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups = {name: [] for name in (
        "encoder_layer1", "encoder_layer2", "hard_lmm_bank", "new_Q", "new_K", "new_V",
    )}
    for name, parameter in model.named_parameters():
        if name == KERNEL_KEYS["Q"]:
            groups["new_Q"].append((name, parameter))
        elif name == KERNEL_KEYS["K"]:
            groups["new_K"].append((name, parameter))
        elif name == KERNEL_KEYS["V"]:
            groups["new_V"].append((name, parameter))
        elif name.startswith("encoder.layers.0."):
            groups["encoder_layer1"].append((name, parameter))
        elif name.startswith("encoder.layers.1."):
            groups["encoder_layer2"].append((name, parameter))
        elif name.startswith("lmm."):
            groups["hard_lmm_bank"].append((name, parameter))
    return groups


def head_state_sha256(model: Any) -> str:
    names = ("quantity_head.weight", "quantity_head.bias", "v_t.weight", "b_t", "w_raw")
    state = model.state_dict()
    selected = {name: state[name] for name in names if name in state}
    require(len(selected) == len(names), "Legacy quantity/time head state is incomplete")
    return canonical_state_sha256(selected)


def gradient_records(model: Any, batches_by_fold: Mapping[int, list[tuple]], *, variant: str,
                     frozen: FrozenModules) -> list[dict[str, Any]]:
    model.eval().requires_grad_(True)
    state_before = canonical_state_sha256(model.state_dict())
    groups = parameter_groups(model)
    records = []
    for fold in (0, 1):
        for batch_index, (_, dts, mask, _, quantities) in enumerate(batches_by_fold[fold]):
            require(quantities is not None, "Gradient batch lacks quantities")
            outputs = frozen.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
            named = [(name, parameter) for group in groups.values() for name, parameter in group]
            parameters = [parameter for _, parameter in named]
            time_grad = torch.autograd.grad(outputs["time_loss"].mean(), parameters,
                                            retain_graph=True, allow_unused=True)
            qty_grad = torch.autograd.grad(outputs["log_qty_loss"].mean(), parameters,
                                           allow_unused=True)
            indexed_time = {name: value for (name, _), value in zip(named, time_grad)}
            indexed_qty = {name: value for (name, _), value in zip(named, qty_grad)}
            for group, members in groups.items():
                dot = time_sq = qty_sq = 0.0
                unused = not members
                for name, _ in members:
                    gt, gq = indexed_time[name], indexed_qty[name]
                    if gt is None:
                        unused = True
                    else:
                        require(bool(torch.isfinite(gt).all()), f"Non-finite time gradient: {variant}.{group}")
                        time_sq += float(torch.square(gt.double()).sum())
                    if gq is None:
                        unused = True
                    else:
                        require(bool(torch.isfinite(gq).all()), f"Non-finite quantity gradient: {variant}.{group}")
                        qty_sq += float(torch.square(gq.double()).sum())
                    if gt is not None and gq is not None:
                        dot += float((gt.double() * gq.double()).sum())
                time_norm, quantity_norm = math.sqrt(time_sq), math.sqrt(qty_sq)
                zero = time_norm == 0.0 or quantity_norm == 0.0
                cosine = None if unused or zero else dot / (time_norm * quantity_norm)
                records.append({
                    "variant": variant, "fold": fold, "batch_index": batch_index,
                    "group": group, "dot_product": dot, "cosine": cosine,
                    "time_norm": time_norm, "quantity_norm": quantity_norm,
                    "unused_or_zero_gradient": bool(unused or zero),
                })
            for parameter in model.parameters():
                parameter.grad = None
    require(canonical_state_sha256(model.state_dict()) == state_before,
            f"Gradient diagnostic changed {variant} state")
    require(all(parameter.grad is None for parameter in model.parameters()), "Parameter .grad cleanup failed")
    model.requires_grad_(False)
    return records


def _restore_checkpoint(repo_root: Path, spec: Mapping[str, Any], frozen: FrozenModules,
                        *, role: str) -> tuple[Any, dict[str, Any]]:
    checkpoint_path = repo_root / spec["checkpoint_path"]
    summary_path = repo_root / spec["summary_path"]
    require(checkpoint_path.is_file() and summary_path.is_file(), f"{role} artifact missing")
    require(sha256_file(checkpoint_path) == spec["checkpoint_file_sha256"], f"{role} checkpoint hash drift")
    require(sha256_file(summary_path) == spec["summary_sha256"], f"{role} summary hash drift")
    payload = frozen.checkpoint_load(checkpoint_path, map_location="cpu")
    summary = read_json(summary_path)
    state = payload.get("model_state_dict")
    require(isinstance(state, dict), f"{role} model state missing")
    digest = canonical_state_sha256(state)
    require(digest == spec.get("checkpoint_state_sha256", digest), f"{role} state hash drift")
    require(summary.get("checkpoint_state_sha256") == digest, f"{role} summary state hash drift")
    require(summary.get("backbone") == spec["backbone"] == payload.get("backbone"),
            f"{role} backbone route drift")
    require(summary.get("evaluation_scope") == payload.get("evaluation_scope") == "validation_only",
            f"{role} evaluation scope drift")
    require(summary.get("held_out_test_evaluated") is False
            and payload.get("held_out_test_evaluated") is False, f"{role} held-out route drift")
    require(payload.get("best_epoch") == summary.get("best_epoch"), f"{role} selected epoch drift")
    for key in ("encoder_config", "interface_meta", "resume_identity"):
        require(payload.get(key) == summary.get(key), f"{role} metadata drift: {key}")
    model = frozen.restore_model(payload, state, candidate={"backbone": spec["backbone"]}, artifact=role)
    model.eval().requires_grad_(False)
    return model, {
        "checkpoint_path": str(checkpoint_path), "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
        "checkpoint_state_sha256": digest,
    }


def _batch_identities(dataset: Any, selected_indices: np.ndarray, part_indices: torch.Tensor,
                      offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = selected_indices[offset:offset + len(part_indices)]
    series = []
    folds = []
    for index, returned_part in zip(indices.tolist(), part_indices.tolist()):
        part, _ = dataset.index[index]
        require(part == int(returned_part), "Dataset/sample identity drift")
        value = canonical_series_id(dataset.parts[part])
        series.append(value)
        folds.append(fold_for_series(value))
    return indices, np.asarray(series, dtype=str), np.asarray(folds, dtype=np.int8)


def actual_model_preflight(
    repo_root: Path,
    contract: Mapping[str, Any],
    frozen: FrozenModules,
) -> dict[str, Any]:
    """Restore all six real checkpoints and exercise every synthetic mask safely."""
    dts = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0, 4.0], [0.0, 0.5, 1.5, 3.0, 6.0]],
        dtype=torch.float32,
    )
    quantities = torch.tensor(
        [[0.0, 0.0, 2.0, 7.0, 11.0], [1.0, 3.0, 5.0, 9.0, 13.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[False, False, True, True, True], [True, True, True, True, True]],
        dtype=torch.bool,
    )
    result: dict[str, Any] = {"status": "passed", "datasets": {}}
    for row in contract["datasets"]:
        B_model, B_evidence = _restore_checkpoint(repo_root, row["B"], frozen, role="B")
        candidate_model, candidate_evidence = _restore_checkpoint(
            repo_root, row["candidate"], frozen, role="candidate",
        )
        state_before = canonical_state_sha256(candidate_model.state_dict())
        with torch.no_grad():
            B_output = extract_variant_batch(B_model, dts, mask, quantities, frozen)
            B_parity = assert_official_parity(B_model, dts, mask, quantities, B_output, frozen)
            B_leakage = assert_target_padding_invariance(
                B_model, dts, mask, quantities, B_output, frozen,
            )
            variants = {}
            for name in contract["variant_order"][1:]:
                with causal_kernel_mask(candidate_model, contract["variants"][name]):
                    output = extract_variant_batch(candidate_model, dts, mask, quantities, frozen)
                    variants[name] = {
                        "leakage": assert_target_padding_invariance(
                            candidate_model, dts, mask, quantities, output, frozen,
                        )
                    }
                    if name == "FULL":
                        variants[name]["official_parity"] = assert_official_parity(
                            candidate_model, dts, mask, quantities, output, frozen,
                        )
        require(canonical_state_sha256(candidate_model.state_dict()) == state_before,
                f"Synthetic masks did not restore candidate state: {row['dataset']}")
        require(sha256_file(repo_root / row["B"]["checkpoint_path"])
                == B_evidence["checkpoint_file_sha256"], "B checkpoint changed in preflight")
        require(sha256_file(repo_root / row["candidate"]["checkpoint_path"])
                == candidate_evidence["checkpoint_file_sha256"],
                "Candidate checkpoint changed in preflight")
        result["datasets"][row["dataset"]] = {
            "B": {"checkpoint": B_evidence, "official_parity": B_parity, "leakage": B_leakage},
            "candidate": {"checkpoint": candidate_evidence, "variants": variants,
                          "state_restored_after_all_masks": True},
        }
    return result


def extract_dataset(repo_root: Path, frozen_root: Path, contract_path: Path,
                    contract: Mapping[str, Any], row: Mapping[str, Any], output: Path,
                    frozen: FrozenModules, source_evidence: Mapping[str, Any]) -> dict[str, Any]:
    require(not output.exists(), f"Dataset output already exists: {output}")
    output.mkdir(parents=True)
    checks_path, ids_path, npz_path = output / "checks.json", output / "sample_ids.json", output / "extraction.npz"
    checks: dict[str, Any] = {
        "schema_version": 1, "status": "preparing_train_sample", "dataset": row["dataset"],
        "contract_path": str(contract_path), "contract_sha256": sha256_file(contract_path),
        "source": dict(source_evidence), "validation_materialized": False,
        "held_out_materialized": False, "model_or_optimizer_updates": False,
        "started_at": utc_now(),
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__, "polars": pl.__version__,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "batch_size": contract["scope"]["batch_size"],
        },
    }
    save_json(checks_path, checks)
    checkpoint_hashes_before: dict[str, str] = {}
    runner_sha_before = sha256_file(Path(__file__))
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
    ).strip()
    checks.update(runner_file_sha256=runner_sha_before, diagnostic_source_revision=source_revision)
    try:
        data_path = repo_root / row["data_path"]
        require(data_path.is_file() and sha256_file(data_path) == row["data_sha256"], "Dataset hash drift")
        frame = load_train_frame(data_path)
        prepared = frozen.prepare_count_frame(frame)
        dataset = frozen.Dataset(
            prepared, lookback_weeks=row["lookback_weeks"], max_seq_len=row["max_seq_len"],
            mode="all", split_col="chronological_split", target_splits={"train"},
        )
        require(len(dataset) == row["expected_train_targets"], "Expected train target count drift")
        quartiles = duration_quartiles(dataset)
        indices, ids = sample_train_targets(dataset, contract["sampling"])
        ids.update(dataset=row["dataset"], contract_sha256=sha256_file(contract_path))
        save_json(ids_path, ids)  # Required before checkpoint load or any model output.
        ids_hash_before = sha256_file(ids_path)

        B_model, B_evidence = _restore_checkpoint(repo_root, row["B"], frozen, role="B")
        candidate_model, candidate_evidence = _restore_checkpoint(
            repo_root, row["candidate"], frozen, role="candidate",
        )
        checkpoint_hashes_before = {
            "B": B_evidence["checkpoint_file_sha256"],
            "candidate": candidate_evidence["checkpoint_file_sha256"],
        }
        model_state_before = {
            "B": canonical_state_sha256(B_model.state_dict()),
            "candidate": canonical_state_sha256(candidate_model.state_dict()),
        }
        head_state_before = {"B": head_state_sha256(B_model),
                             "candidate": head_state_sha256(candidate_model)}
        candidate_common_state = {
            name: value for name, value in candidate_model.state_dict().items()
            if name not in set(KERNEL_KEYS.values())
        }
        candidate_common_digest = canonical_state_sha256(candidate_common_state)
        loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=contract["scope"]["batch_size"],
                            shuffle=False, num_workers=0, collate_fn=frozen.collate)
        common: dict[str, list[np.ndarray]] = {key: [] for key in (
            "dataset_index", "series_id", "fold", "true_quantity", "true_duration",
        )}
        variants: dict[str, dict[str, list[np.ndarray]]] = {
            name: {field: [] for field in VARIANT_FIELDS} for name in contract["variant_order"]
        }
        parity: dict[str, Any] = {}
        leakage: dict[str, Any] = {}
        mask_checks: dict[str, Any] = {}
        offset = 0
        first_batches_by_fold: dict[int, list[tuple]] = {0: [], 1: []}
        for batch_index, batch in enumerate(loader):
            _, dts, mask, part_indices, quantities = batch
            require(quantities is not None, "Quantity labels are missing")
            batch_indices, series, folds = _batch_identities(dataset, indices, part_indices, offset)
            common["dataset_index"].append(batch_indices)
            common["series_id"].append(series)
            common["fold"].append(folds)
            outputs: dict[str, dict[str, torch.Tensor]] = {}
            with torch.no_grad():
                outputs["B"] = extract_variant_batch(B_model, dts, mask, quantities, frozen)
                for variant in contract["variant_order"][1:]:
                    switches = contract["variants"][variant]
                    with causal_kernel_mask(candidate_model, switches):
                        require(
                            canonical_state_sha256({
                                name: value for name, value in candidate_model.state_dict().items()
                                if name not in set(KERNEL_KEYS.values())
                            }) == candidate_common_digest,
                            f"Variant {variant} changed a common weight or buffer",
                        )
                        result = extract_variant_batch(candidate_model, dts, mask, quantities, frozen)
                        outputs[variant] = result
                        if batch_index == 0:
                            leakage[variant] = assert_target_padding_invariance(
                                candidate_model, dts, mask, quantities, result, frozen,
                            )
                            parameters = dict(candidate_model.named_parameters())
                            mask_checks[variant] = {
                                name: {
                                    "enabled": bool(switch),
                                    "all_three_rows_zero": bool(
                                        int(parameters[KERNEL_KEYS[name]].detach().count_nonzero()) == 0
                                    ),
                                }
                                for name, switch in zip(("Q", "K", "V"), switches)
                            }
            if batch_index == 0:
                with torch.no_grad():
                    parity["B"] = assert_official_parity(
                        B_model, dts, mask, quantities, outputs["B"], frozen,
                    )
                    parity["FULL"] = assert_official_parity(
                        candidate_model, dts, mask, quantities, outputs["FULL"], frozen,
                    )
                    leakage["B"] = assert_target_padding_invariance(
                        B_model, dts, mask, quantities, outputs["B"], frozen,
                    )
            common["true_quantity"].append(outputs["B"]["true_quantity"].cpu().numpy())
            common["true_duration"].append(outputs["B"]["true_duration"].cpu().numpy())
            for variant, result in outputs.items():
                require(torch.equal(result["true_quantity"], outputs["B"]["true_quantity"]),
                        f"True quantity drift: {variant}")
                require(torch.equal(result["true_duration"], outputs["B"]["true_duration"]),
                        f"True duration drift: {variant}")
                for field in VARIANT_FIELDS:
                    variants[variant][field].append(result[field].detach().cpu().numpy())
            offset += len(part_indices)
            if batch_index % 32 == 0 or offset == len(indices):
                checks.update(status="extracting", processed_targets=offset)
                save_json(checks_path, checks)
                print(json.dumps({
                    "stage": "extract", "dataset": row["dataset"],
                    "processed_targets": offset, "total_targets": len(indices),
                }, sort_keys=True), flush=True)
        require(offset == len(indices), "Incomplete extraction")

        # Canonical gradient batches are constructed independently per frozen fold.
        for fold in (0, 1):
            fold_array = np.asarray([item["fold"] for item in ids["rows"]], dtype=np.int8)
            fold_indices = indices[fold_array == fold]
            fold_loader = DataLoader(
                Subset(dataset, fold_indices.tolist()), batch_size=contract["scope"]["batch_size"],
                shuffle=False, num_workers=0, collate_fn=frozen.collate,
            )
            first_batches_by_fold[fold] = list(
                batch for _, batch in zip(range(contract["gradient_diagnostic"]["batches_per_fold"]), fold_loader)
            )
            require(len(first_batches_by_fold[fold]) == 4, f"Fold {fold} gradient batch count drift")
        gradients = gradient_records(B_model, first_batches_by_fold, variant="B", frozen=frozen)
        gradients += gradient_records(candidate_model, first_batches_by_fold, variant="FULL", frozen=frozen)

        arrays: dict[str, np.ndarray] = {
            key: np.concatenate(values, axis=0) for key, values in common.items()
        }
        for variant in contract["variant_order"]:
            for field in VARIANT_FIELDS:
                arrays[f"{variant}__{field}"] = np.concatenate(variants[variant][field], axis=0)
        n = contract["sampling"]["targets_per_dataset"]
        require(all(value.shape[0] == n for value in arrays.values()), "NPZ row alignment drift")
        require(set(arrays) == set(common) | {
            f"{variant}__{field}" for variant in contract["variant_order"] for field in VARIANT_FIELDS
        }, "NPZ schema drift")
        for key, value in arrays.items():
            if value.dtype.kind not in "biuUS":
                require(np.isfinite(value).all(), f"Non-finite NPZ array: {key}")
        save_npz(npz_path, arrays)

        model_state_after = {
            "B": canonical_state_sha256(B_model.state_dict()),
            "candidate": canonical_state_sha256(candidate_model.state_dict()),
        }
        head_state_after = {"B": head_state_sha256(B_model),
                            "candidate": head_state_sha256(candidate_model)}
        require(model_state_after == model_state_before, "Model state changed across interventions")
        require(head_state_after == head_state_before, "Quantity/time head parameters changed")
        require(all(parameter.grad is None for model in (B_model, candidate_model)
                    for parameter in model.parameters()), "Parameter gradients were not cleaned")
        checkpoint_hashes_after = {
            "B": sha256_file(repo_root / row["B"]["checkpoint_path"]),
            "candidate": sha256_file(repo_root / row["candidate"]["checkpoint_path"]),
        }
        require(checkpoint_hashes_after == checkpoint_hashes_before, "Checkpoint file changed")
        require(sha256_file(ids_path) == ids_hash_before, "Frozen sample identity manifest changed")
        require(sha256_file(contract_path) == EXPECTED_CONTRACT_SHA256, "Contract changed during extraction")
        require(sha256_file(data_path) == row["data_sha256"], "Dataset changed during extraction")
        require(sha256_file(Path(__file__)) == runner_sha_before, "Diagnostic runner changed during extraction")
        source_after = verify_frozen_source(contract, frozen_root)
        require(source_after == source_evidence, "Frozen implementation changed during extraction")
        checks.update(
            status="passed", completed_at=utc_now(), train_rows=frame.height,
            train_targets=len(dataset), sample_targets=n,
            quantity_boundaries=list(row["quantity_boundaries"]),
            duration_train_quartile_boundaries=quartiles,
            sample_ids_path=str(ids_path), sample_ids_sha256=ids_hash_before,
            npz_path=str(npz_path), npz_sha256=sha256_file(npz_path),
            checkpoint_evidence={"B": B_evidence, "candidate": candidate_evidence},
            checkpoint_file_hashes_before=checkpoint_hashes_before,
            checkpoint_file_hashes_after=checkpoint_hashes_after,
            model_state_hashes_before=model_state_before, model_state_hashes_after=model_state_after,
            head_parameter_hashes_before=head_state_before,
            head_parameter_hashes_after=head_state_after,
            official_target_outputs_first_fixed_batch=parity,
            target_duration_quantity_padding_invariance=leakage,
            variant_kernel_mask_checks=mask_checks,
            candidate_common_state_sha256=candidate_common_digest,
            gradient_records=gradients,
            npz_keys=sorted(arrays),
            validation_materialized=False, held_out_materialized=False,
            model_or_optimizer_updates=False, parameter_grad_cleanup=True,
        )
        save_json(checks_path, checks)
        return checks
    except BaseException as error:
        checks.update(status="failed", failed_at=utc_now(), error=f"{type(error).__name__}: {error}")
        if checkpoint_hashes_before:
            checks["checkpoint_file_hashes_before"] = checkpoint_hashes_before
        save_json(checks_path, checks)
        raise


def validate_completed_output(
    output: Path,
    *,
    repo_root: Path,
    contract_path: Path,
    row: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    checks_path = output / "checks.json"
    require(checks_path.is_file(), f"Incomplete existing dataset output: {row['dataset']}")
    checks = read_json(checks_path)
    require(checks.get("status") == "passed", f"Incomplete existing dataset output: {row['dataset']}")
    require(checks.get("dataset") == row["dataset"], "Cached dataset identity drift")
    require(checks.get("contract_sha256") == sha256_file(contract_path) == EXPECTED_CONTRACT_SHA256,
            "Cached contract binding drift")
    require(checks.get("source") == source_evidence, "Cached frozen source binding drift")
    require(checks.get("runner_file_sha256") == sha256_file(Path(__file__)),
            "Cached runner source binding drift")
    require(checks.get("diagnostic_source_revision") == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
    ).strip(), "Cached diagnostic Git revision drift")
    require(sha256_file(repo_root / row["data_path"]) == row["data_sha256"],
            "Cached dataset source drift")
    for role in ("B", "candidate"):
        current = sha256_file(repo_root / row[role]["checkpoint_path"])
        require(current == row[role]["checkpoint_file_sha256"], f"Cached {role} checkpoint drift")
        require(checks["checkpoint_file_hashes_before"][role] == current
                == checks["checkpoint_file_hashes_after"][role],
                f"Cached {role} checkpoint receipt drift")
    for path_key, hash_key in (("npz_path", "npz_sha256"),
                               ("sample_ids_path", "sample_ids_sha256")):
        artifact = Path(checks[path_key])
        require(artifact.is_file() and sha256_file(artifact) == checks[hash_key],
                f"Cached artifact drift: {path_key}")
    require(checks.get("validation_materialized") is False
            and checks.get("held_out_materialized") is False
            and checks.get("model_or_optimizer_updates") is False,
            "Cached scope receipt drift")
    return checks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root, frozen_root, contract_path = (
        args.repo_root.resolve(), args.frozen_source_root.resolve(), args.contract.resolve()
    )
    require(repo_root.is_dir() and contract_path.is_file(), "Repository/contract path is invalid")
    contract = read_json(contract_path)
    validate_contract(contract, contract_path)
    names = [row["dataset"] for row in contract["datasets"]]
    if args.dataset is not None:
        require(args.dataset in names, f"Unknown dataset: {args.dataset}")
        names = [args.dataset]
    source_evidence = verify_frozen_source(contract, frozen_root)
    frozen = import_frozen_modules(repo_root, frozen_root)
    torch.set_num_threads(contract["scope"]["torch_threads"])
    require(not torch.cuda.is_initialized(), "CUDA must not be initialized for the CPU diagnostic")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        output = args.output_dir / name
        if output.exists():
            validate_completed_output(
                output, repo_root=repo_root, contract_path=contract_path,
                row=dataset_row(contract, name), source_evidence=source_evidence,
            )
            continue
        extract_dataset(
            repo_root, frozen_root, contract_path, contract, dataset_row(contract, name),
            output, frozen, source_evidence,
        )


if __name__ == "__main__":
    main()
