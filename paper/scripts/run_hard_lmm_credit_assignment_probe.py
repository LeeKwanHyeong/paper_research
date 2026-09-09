#!/usr/bin/env python3
"""Run the frozen-B, train-only Hard-LMM credit-assignment diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "paper/contracts/hard_lmm_credit_assignment_probe_v1.json"
RESULT = ROOT / "paper/results/hard_lmm_credit_assignment_probe_20260909"
HELPER_PATH = ROOT / "paper/scripts/hard_lmm_credit_assignment_probe.py"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hard_lmm_credit_probe")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_hard_lmm_credit_probe")

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset


def _load_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("hard_lmm_credit_probe_helpers", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load credit-assignment helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


@dataclass(frozen=True)
class FrozenModules:
    Dataset: Any
    canonical_state_sha256: Any
    collate: Any
    prepare_count_frame: Any
    restore_b0: Any
    right_pad_batch: Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def import_frozen_modules(frozen_root: Path) -> FrozenModules:
    frozen_root = frozen_root.resolve()
    require(frozen_root.is_dir(), "Frozen source root is missing")
    project_prefixes = ("models", "data_loader", "paper", "simple_lab_test", "utils")
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in project_prefixes):
            raise ValueError(f"Mutable project module was imported before frozen activation: {name}")
    filtered_path: list[str] = []
    for entry in sys.path:
        resolved = Path.cwd().resolve() if not entry else Path(entry).resolve()
        if resolved == ROOT.resolve() or _inside(resolved, ROOT):
            continue
        filtered_path.append(entry)
    sys.path[:] = filtered_path
    sys.path.insert(0, str(frozen_root))
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    loader = importlib.import_module("data_loader.event_seq_data_module")
    retrieval = importlib.import_module("paper.scripts.analyze_count_aware_b0_retrieval")
    runner = importlib.import_module("simple_lab_test.search.common.runner")
    for module in (core, loader, retrieval, runner):
        require(
            _inside(Path(module.__file__), frozen_root),
            f"Frozen import escaped its root: {module.__name__}",
        )
    return FrozenModules(
        Dataset=loader.RMTPPWeekLookbackDataset,
        canonical_state_sha256=runner.canonical_state_dict_sha256,
        collate=loader.collate_week_lookback,
        prepare_count_frame=core.prepare_count_frame,
        restore_b0=retrieval.restore_b0,
        right_pad_batch=core.right_pad_batch,
    )


def audit_loaded_frozen_modules(
    contract: Mapping[str, Any],
    frozen_root: Path,
) -> dict[str, Any]:
    manifest = read_json(ROOT / contract["source"]["pinned_source_manifest_path"])
    manifest_files = manifest.get("files")
    require(isinstance(manifest_files, dict), "Pinned source file hashes are missing")
    project_prefixes = ("models", "data_loader", "paper", "simple_lab_test", "utils")
    verified: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in project_prefixes
        ):
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if not origin or origin in {"built-in", "frozen"}:
            continue
        path = Path(origin).resolve()
        require(_inside(path, frozen_root), f"Project import escaped frozen root: {name}")
        require(path.suffix == ".py", f"Unexpected frozen module origin: {path}")
        relative = path.relative_to(frozen_root.resolve()).as_posix()
        expected = manifest_files.get(relative)
        require(isinstance(expected, str), f"Loaded source lacks a pinned hash: {relative}")
        observed = sha256_file(path)
        require(observed == expected, f"Loaded frozen source hash drift: {relative}")
        verified[relative] = observed
    require(len(verified) >= 40, "Too few frozen project modules were audited")
    return {
        "policy": "every loaded project Python module must resolve inside the frozen root and match the pinned source manifest",
        "verified_file_count": len(verified),
        "verified_files": verified,
    }


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("contract_id") == "hard_lmm_credit_assignment_probe_v1", "Contract ID drift")
    scope = contract["diagnostic_scope"]
    require(scope["allowed_split"] == "train_only", "Diagnostic is not train-only")
    require(scope["validation_rows_materialized"] is False, "Validation access enabled")
    require(scope["model_parameter_updates"] is False, "Parameter updates enabled")
    require(scope["maximum_targets_per_dataset"] == 4096, "Cohort size drift")
    require(scope["minimum_targets_per_fold"] == 2048, "Fold size drift")
    require(scope["batch_size"] == 128, "Batch size drift")
    require(contract["authorization"]["validation_rows"] is False, "Validation rows authorized")
    require(contract["authorization"]["held_out_test"] is False, "Held-out authorized")
    probe = contract["gradient_probe"]
    require(probe["temperature"] == 1.0, "Surrogate temperature drift")
    require("all-64" in probe["arms"]["normal_similarity"], "All-memory surrogate disabled")
    require("cyclically shifted by one" in probe["arms"]["shuffled_similarity"], "Sham permutation drift")
    require(len(contract["datasets"]) == 3, "Dataset scope drift")


def verify_frozen_source(contract: Mapping[str, Any], frozen_root: Path) -> dict[str, Any]:
    source = contract["source"]
    manifest_path = ROOT / source["pinned_source_manifest_path"]
    require(manifest_path.is_file(), "Pinned source manifest is missing")
    require(
        sha256_file(manifest_path) == source["pinned_source_manifest_sha256"],
        "Pinned source manifest hash drift",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("source_revision") == source["frozen_source_revision"],
        "Frozen source revision drift",
    )
    for directory in ("models", "utils", "sample_data"):
        require((frozen_root / directory).is_dir(), f"Frozen source sentinel is missing: {directory}")
    critical = source.get("frozen_runtime_file_sha256")
    require(isinstance(critical, dict) and critical, "Frozen runtime hashes are missing")
    manifest_files = manifest.get("files")
    require(isinstance(manifest_files, dict), "Pinned source file hashes are missing")
    for relative, expected in critical.items():
        require(
            manifest_files.get(relative) == expected,
            f"Frozen runtime hash is not backed by the pinned manifest: {relative}",
        )
        path = frozen_root / relative
        require(path.is_file(), f"Frozen source file is missing: {relative}")
        require(sha256_file(path) == expected, f"Frozen source hash drift: {relative}")
    return {
        "frozen_source_revision": source["frozen_source_revision"],
        "pinned_source_manifest_path": source["pinned_source_manifest_path"],
        "pinned_source_manifest_sha256": sha256_file(manifest_path),
        "verified_critical_file_count": len(critical),
    }


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


def load_train_frame(path: Path) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(frame.height > 0, "Train population is empty")
    require(
        set(frame["chronological_split"].unique().to_list()) == {"train"},
        "A non-train row was materialized",
    )
    return frame


def verify_file(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"Missing {label}: {path}")
    require(sha256_file(path) == expected, f"Changed {label}: {path}")


def load_cohort(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ids_path = ROOT / row["cohort_sample_ids_path"]
    cache_path = ROOT / row["cohort_cache_path"]
    verify_file(ids_path, row["cohort_sample_ids_sha256"], "cohort identities")
    verify_file(cache_path, row["cohort_cache_sha256"], "cohort cache")
    identities = read_json(ids_path)
    require(identities.get("target_count") == 4096, "Cohort target count drift")
    require(identities.get("fold_counts") == {"0": 2048, "1": 2048}, "Cohort fold count drift")
    with np.load(cache_path, allow_pickle=False) as loaded:
        required = (
            "dataset_index", "series_id", "fold", "true_quantity", "B__prediction",
            "B__H2", "B__memory_residual", "B__top4_indices",
        )
        require(all(name in loaded for name in required), "Cohort cache schema drift")
        cache = {name: loaded[name].copy() for name in required}
    rows = identities["rows"]
    require(len(rows) == 4096, "Cohort identity rows changed")
    require(
        np.array_equal(cache["dataset_index"], np.asarray([item["dataset_index"] for item in rows])),
        "Cohort cache and identity order differ",
    )
    require(
        np.array_equal(cache["fold"], np.asarray([item["fold"] for item in rows])),
        "Cohort fold arrays differ",
    )
    require(
        [canonical_series_id(value) for value in cache["series_id"]]
        == [str(item["series_id"]) for item in rows],
        "Cohort series arrays differ",
    )
    fold_series = {
        fold: {str(item["series_id"]) for item in rows if int(item["fold"]) == fold}
        for fold in (0, 1)
    }
    require(fold_series[0].isdisjoint(fold_series[1]), "Cohort folds share a series")
    require(all(len(values) >= 30 for values in fold_series.values()), "Too few series in a fold")
    require(
        identities.get("fold_unique_series")
        == {str(fold): len(values) for fold, values in fold_series.items()},
        "Cohort unique-series counts drift",
    )
    require(
        np.array_equal(cache["fold"][:2048], np.zeros(2048, dtype=cache["fold"].dtype))
        and np.array_equal(cache["fold"][2048:], np.ones(2048, dtype=cache["fold"].dtype)),
        "Cohort folds are not stored in the contracted order",
    )
    return identities, cache


def verify_dataset_identity(dataset: Any, identity_rows: Sequence[Mapping[str, Any]]) -> None:
    for item in identity_rows:
        index = int(item["dataset_index"])
        part_index, context_end = dataset.index[index]
        require(int(context_end) == int(item["context_end"]), "Cohort context drift")
        require(
            canonical_series_id(dataset.parts[part_index]) == str(item["series_id"]),
            "Cohort series drift",
        )
        target_seq = int(dataset.seq_lists[part_index][context_end + 1])
        require(target_seq == int(item["target_seq"]), "Cohort target sequence drift")


def probe_parameters(model: torch.nn.Module) -> tuple[list[str], list[torch.nn.Parameter]]:
    named = sorted(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        key=lambda item: item[0],
    )
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    require(any(name.startswith("encoder.") for name in names), "Query block is empty")
    require(names.count("lmm.mem") == 1, "Key block is missing or duplicated")
    return names, parameters


def zeros_for(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter, dtype=torch.float64, device="cpu") for parameter in parameters]


def add_scaled(
    accumulator: list[torch.Tensor],
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    scale: float,
) -> None:
    complete = helpers.replace_none_gradients(parameters, gradients)
    for target, gradient in zip(accumulator, complete):
        require(bool(torch.isfinite(gradient).all()), "Non-finite gradient")
        target.add_(gradient.detach().to(dtype=torch.float64, device="cpu"), alpha=float(scale))


def divide_(values: list[torch.Tensor], denominator: int) -> None:
    require(denominator > 0, "Cannot divide a gradient by zero events")
    for value in values:
        value.div_(float(denominator))


def vector_norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.square(value).sum()) for value in values))


def clipped_difference(
    hard_joint: Sequence[torch.Tensor | None],
    additional: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    max_norm: float = 1.0,
) -> list[torch.Tensor]:
    hard = helpers.replace_none_gradients(parameters, hard_joint)
    extra = helpers.replace_none_gradients(parameters, additional)
    candidate = [left + right for left, right in zip(hard, extra)]

    def clip_coefficient(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
        norms = torch.stack(
            [torch.linalg.vector_norm(value.detach(), ord=2.0) for value in gradients]
        )
        total_norm = torch.linalg.vector_norm(norms, ord=2.0)
        return torch.clamp(
            torch.as_tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device)
            / (total_norm + 1e-6),
            max=1.0,
        )

    hard_scale = clip_coefficient(hard)
    candidate_scale = clip_coefficient(candidate)
    return [
        (candidate_value * candidate_scale - hard_value * hard_scale).detach()
        for candidate_value, hard_value in zip(candidate, hard)
    ]


def prepare_batch(
    model: torch.nn.Module,
    frozen: FrozenModules,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dts, quantities, mask, lengths = frozen.right_pad_batch(dts, quantities, mask)
    rows = torch.arange(dts.size(0))
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.masked_fill(~mask, 0.0).clone()
    history_quantities[rows, target_positions] = 0.0
    writes = mask.clone()
    writes[rows, target_positions] = False
    local = model._encode_base(
        dts,
        history_quantities,
        mask,
        memory_write_mask=writes,
    )
    residual, trace = model.lmm.retrieve(local)
    query = local[rows, history_positions]
    hard_residual = residual[rows, history_positions]
    hard_hidden = query + hard_residual
    return {
        "query": query,
        "hard_residual": hard_residual,
        "hard_hidden": hard_hidden,
        "top4_indices": trace["prototype_indices"][rows, history_positions],
        "target_quantity": quantities[rows, target_positions].float(),
        "target_duration": dts[rows, target_positions].float(),
    }


@torch.no_grad()
def invariant_outputs(
    model: torch.nn.Module,
    frozen: FrozenModules,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = prepare_batch(model, frozen, dts, mask, quantities)
    prediction = model.predict_quantity(batch["hard_hidden"])[1]
    return {
        "query": batch["query"],
        "hard_residual": batch["hard_residual"],
        "hard_hidden": batch["hard_hidden"],
        "top4_indices": batch["top4_indices"],
        "prediction": prediction,
    }


def target_padding_invariance(
    model: torch.nn.Module,
    frozen: FrozenModules,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, float]:
    original = invariant_outputs(model, frozen, dts, mask, quantities)
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    positions = torch.arange(mask.size(1)).expand_as(mask)
    target_positions = torch.where(mask, positions, -1).max(dim=1).values
    rows = torch.arange(mask.size(0))
    changed_dts[rows, target_positions] += 10007.0
    changed_quantities[rows, target_positions] += 20011.0
    changed_dts[~mask] = 30013.0
    changed_quantities[~mask] = 40009.0
    changed = invariant_outputs(model, frozen, changed_dts, mask, changed_quantities)
    differences: dict[str, float] = {}
    for name in original:
        left, right = original[name], changed[name]
        require(torch.equal(left, right), f"Forbidden target/padding changed {name}")
        differences[name] = 0.0
    return differences


def structural_audit(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    query = batch["query"].detach().clone().requires_grad_(True)
    key = model.lmm.mem[0].detach().clone().requires_grad_(True)
    values = model.lmm.mem[0].detach().clone()
    credit = helpers.quantity_credit_signal(
        batch["hard_hidden"].detach(),
        model.quantity_head.weight.detach(),
        model.quantity_head.bias.detach(),
        batch["target_quantity"].detach(),
    )["hidden_credit"]
    scores = torch.nn.functional.normalize(query, dim=-1) @ torch.nn.functional.normalize(
        key, dim=-1
    ).transpose(0, 1)
    indices = torch.topk(scores, 4, dim=-1).indices
    hard_read = values[indices].mean(dim=1)
    hard_objective = (hard_read * credit).sum() + 0.0 * (query.sum() + key.sum())
    hard_query, hard_key = torch.autograd.grad(hard_objective, (query, key))
    normal_read, _, _ = helpers.cosine_soft_read(
        query, key, value_memory=values, temperature=1.0
    )
    normal_objective = (normal_read * credit).sum()
    normal_query, normal_key = torch.autograd.grad(normal_objective, (query, key))
    permutation = helpers.cyclic_derangement(key.size(0), device=key.device)
    shuffled_read, _, _ = helpers.cosine_soft_read(
        query,
        key,
        value_memory=values,
        temperature=1.0,
        value_permutation=permutation,
    )
    normal_hidden = helpers.forward_identical_hidden(batch["hard_hidden"], normal_read)
    shuffled_hidden = helpers.forward_identical_hidden(batch["hard_hidden"], shuffled_read)
    return {
        "hard_query_addressing_gradient_max_abs": float(hard_query.abs().max()),
        "hard_key_addressing_gradient_max_abs": float(hard_key.abs().max()),
        "normal_query_addressing_gradient_l2": float(torch.linalg.vector_norm(normal_query)),
        "normal_key_addressing_gradient_l2": float(torch.linalg.vector_norm(normal_key)),
        "normal_forward_bitwise_identity": bool(torch.equal(normal_hidden, batch["hard_hidden"])),
        "shuffled_forward_bitwise_identity": bool(torch.equal(shuffled_hidden, batch["hard_hidden"])),
        "normal_prediction_bitwise_identity": bool(
            torch.equal(model.predict_quantity(normal_hidden)[1], model.predict_quantity(batch["hard_hidden"])[1])
        ),
        "shuffled_prediction_bitwise_identity": bool(
            torch.equal(model.predict_quantity(shuffled_hidden)[1], model.predict_quantity(batch["hard_hidden"])[1])
        ),
        "memory_size": int(key.size(0)),
        "temperature": 1.0,
        "shuffled_value_permutation": permutation.cpu().tolist(),
    }


def fold_gradients(
    model: torch.nn.Module,
    frozen: FrozenModules,
    loader: DataLoader,
    *,
    body_threshold: float,
    names: Sequence[str],
    parameters: Sequence[torch.nn.Parameter],
    parity_cache: Mapping[str, np.ndarray],
    parity_offset: int,
    dataset_name: str,
    fold: int,
) -> tuple[dict[str, Any], dict[str, list[torch.Tensor]], int, dict[str, np.ndarray], dict[str, Any]]:
    accumulated = {
        name: zeros_for(parameters)
        for name in (
            "hard_log_mse", "hard_raw_squared_error", "hard_body_mae", "hard_time_nll",
            "normal_additional", "shuffled_additional",
            "normal_clipped_additional", "shuffled_clipped_additional",
        )
    }
    counts = {"targets": 0, "body": 0}
    sums = {"log_mse": 0.0, "raw_squared_error": 0.0, "body_absolute_error": 0.0, "time_nll": 0.0}
    parity = {"query": [], "hard_residual": [], "top4_indices": [], "prediction": [], "target_quantity": []}
    first_structural: dict[str, Any] | None = None
    first_invariance: dict[str, float] | None = None
    started = time.monotonic()
    permutation = helpers.cyclic_derangement(int(model.lmm.mem.size(1)))
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        require(quantities is not None, "Quantity observations are missing")
        if batch_index == 0:
            first_invariance = target_padding_invariance(model, frozen, dts, mask, quantities)
        batch = prepare_batch(model, frozen, dts, mask, quantities)
        hard_hidden = batch["hard_hidden"]
        target_quantity = batch["target_quantity"]
        target_duration = batch["target_duration"]
        credit = helpers.quantity_credit_signal(
            hard_hidden,
            model.quantity_head.weight,
            model.quantity_head.bias,
            target_quantity,
        )
        normal_read, _, _ = helpers.cosine_soft_read(
            batch["query"], model.lmm.mem[0], temperature=1.0
        )
        shuffled_read, _, _ = helpers.cosine_soft_read(
            batch["query"],
            model.lmm.mem[0],
            temperature=1.0,
            value_permutation=permutation,
        )
        if first_structural is None:
            first_structural = structural_audit(model, batch)
        normal_hidden = helpers.forward_identical_hidden(hard_hidden, normal_read)
        shuffled_hidden = helpers.forward_identical_hidden(hard_hidden, shuffled_read)
        require(torch.equal(normal_hidden, hard_hidden), "Normal forward differs from B")
        require(torch.equal(shuffled_hidden, hard_hidden), "Shuffled forward differs from B")
        normal_objective = (credit["hidden_credit"] * normal_read).sum() / len(target_quantity)
        shuffled_objective = (credit["hidden_credit"] * shuffled_read).sum() / len(target_quantity)
        log_loss = credit["log_mse"].mean()
        raw_loss = credit["raw_squared_error"].mean()
        time_loss = (-model.log_f_dt(hard_hidden, target_duration)).mean()
        body_mask = target_quantity <= float(body_threshold)
        body_count = int(body_mask.sum())
        body_loss = (
            (credit["raw_prediction"][body_mask] - target_quantity[body_mask]).abs().mean()
            if body_count
            else None
        )
        g_log = torch.autograd.grad(log_loss, parameters, retain_graph=True, allow_unused=True)
        g_raw = torch.autograd.grad(raw_loss, parameters, retain_graph=True, allow_unused=True)
        if body_loss is None:
            g_body: Sequence[torch.Tensor | None] = [None] * len(parameters)
        else:
            g_body = torch.autograd.grad(body_loss, parameters, retain_graph=True, allow_unused=True)
        g_time = torch.autograd.grad(time_loss, parameters, retain_graph=True, allow_unused=True)
        g_normal = torch.autograd.grad(normal_objective, parameters, retain_graph=True, allow_unused=True)
        g_shuffled = torch.autograd.grad(shuffled_objective, parameters, allow_unused=True)

        count = len(target_quantity)
        add_scaled(accumulated["hard_log_mse"], g_log, parameters, count)
        add_scaled(accumulated["hard_raw_squared_error"], g_raw, parameters, count)
        if body_count:
            add_scaled(accumulated["hard_body_mae"], g_body, parameters, body_count)
        add_scaled(accumulated["hard_time_nll"], g_time, parameters, count)
        add_scaled(accumulated["normal_additional"], g_normal, parameters, count)
        add_scaled(accumulated["shuffled_additional"], g_shuffled, parameters, count)
        hard_joint = [
            (torch.zeros_like(parameter) if left is None else left)
            + (torch.zeros_like(parameter) if right is None else right)
            for parameter, left, right in zip(parameters, g_log, g_time)
        ]
        normal_clipped = clipped_difference(hard_joint, g_normal, parameters)
        shuffled_clipped = clipped_difference(hard_joint, g_shuffled, parameters)
        add_scaled(accumulated["normal_clipped_additional"], normal_clipped, parameters, count)
        add_scaled(accumulated["shuffled_clipped_additional"], shuffled_clipped, parameters, count)

        counts["targets"] += count
        counts["body"] += body_count
        sums["log_mse"] += float(credit["log_mse"].detach().double().sum())
        sums["raw_squared_error"] += float(
            credit["raw_squared_error"].detach().double().sum()
        )
        sums["body_absolute_error"] += float(
            (credit["raw_prediction"][body_mask] - target_quantity[body_mask])
            .detach()
            .abs()
            .double()
            .sum()
        )
        sums["time_nll"] += float(
            (-model.log_f_dt(hard_hidden.detach(), target_duration))
            .detach()
            .double()
            .sum()
        )
        parity["query"].append(batch["query"].detach().cpu().numpy())
        parity["hard_residual"].append(batch["hard_residual"].detach().cpu().numpy())
        parity["top4_indices"].append(batch["top4_indices"].detach().cpu().numpy())
        parity["prediction"].append(credit["raw_prediction"].detach().cpu().numpy())
        parity["target_quantity"].append(target_quantity.detach().cpu().numpy())
        if batch_index % 8 == 0:
            print(
                json.dumps(
                    {
                        "stage": "gradient",
                        "dataset": dataset_name,
                        "fold": fold,
                        "completed": counts["targets"],
                        "seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
    require(counts["targets"] == 2048, f"{dataset_name} fold {fold} target count drift")
    for name, gradients in accumulated.items():
        divide_(gradients, counts["body"] if name == "hard_body_mae" else counts["targets"])
    arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in parity.items()}
    expected_slice = slice(parity_offset, parity_offset + counts["targets"])
    observed_prediction = arrays["prediction"].astype(np.float64)
    cached_prediction = parity_cache["B__prediction"][expected_slice].astype(np.float64)
    prediction_difference = np.abs(observed_prediction - cached_prediction)
    parity_differences = {
        "query_max_abs": float(np.max(np.abs(arrays["query"].astype(np.float64) - parity_cache["B__H2"][expected_slice].astype(np.float64)))),
        "hard_residual_max_abs": float(np.max(np.abs(arrays["hard_residual"].astype(np.float64) - parity_cache["B__memory_residual"][expected_slice].astype(np.float64)))),
        "prediction_max_abs": float(np.max(prediction_difference)),
        "prediction_max_relative": float(
            np.max(prediction_difference / np.maximum(1.0, np.abs(cached_prediction)))
        ),
        "prediction_log1p_max_abs": float(
            np.max(
                np.abs(
                    np.log1p(observed_prediction) - np.log1p(cached_prediction)
                )
            )
        ),
        "target_quantity_max_abs": float(np.max(np.abs(arrays["target_quantity"].astype(np.float64) - parity_cache["true_quantity"][expected_slice].astype(np.float64)))),
        "top4_indices_bitwise_equal": bool(np.array_equal(arrays["top4_indices"], parity_cache["B__top4_indices"][expected_slice])),
    }
    require(parity_differences["query_max_abs"] <= 1e-6, "Frozen-source H2 parity failed")
    require(parity_differences["hard_residual_max_abs"] <= 1e-6, "Frozen-source residual parity failed")
    require(
        parity_differences["prediction_log1p_max_abs"] <= 2e-6
        and parity_differences["prediction_max_relative"] <= 2e-6,
        f"Frozen-source prediction parity failed: {parity_differences}",
    )
    require(parity_differences["target_quantity_max_abs"] == 0.0, "Frozen cohort target drift")
    require(parity_differences["top4_indices_bitwise_equal"], "Frozen-source top-4 parity failed")
    summary = {
        "counts": counts,
        "metrics": {
            "log_mse": sums["log_mse"] / counts["targets"],
            "raw_rmse": math.sqrt(sums["raw_squared_error"] / counts["targets"]),
            "body_mae": sums["body_absolute_error"] / counts["body"],
            "time_nll": sums["time_nll"] / counts["targets"],
        },
        "gradient_norms": {
            name: {
                block: float(torch.linalg.vector_norm(helpers.canonical_vector(names, values, block=block)))
                for block in ("all", "query", "key")
            }
            for name, values in accumulated.items()
        },
        "parity": parity_differences,
    }
    require(first_structural is not None and first_invariance is not None, "Fold audit was not initialized")
    return summary, accumulated, counts["targets"], arrays, {
        "structural": first_structural,
        "target_padding_invariance": first_invariance,
    }


def cross_fold_results(
    names: Sequence[str],
    folds: Mapping[int, Mapping[str, Sequence[torch.Tensor]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source, held in ((0, 1), (1, 0)):
        row: dict[str, Any] = {}
        source_gradients = folds[source]
        derived_gradients = {
            "hard_top4": source_gradients["hard_log_mse"],
            "normal_total": [
                hard + extra
                for hard, extra in zip(
                    source_gradients["hard_log_mse"],
                    source_gradients["normal_additional"],
                )
            ],
            "shuffled_total": [
                hard + extra
                for hard, extra in zip(
                    source_gradients["hard_log_mse"],
                    source_gradients["shuffled_additional"],
                )
            ],
        }
        for arm, gradient_name in (
            ("normal", "normal_additional"),
            ("shuffled", "shuffled_additional"),
            ("normal_clipped", "normal_clipped_additional"),
            ("shuffled_clipped", "shuffled_clipped_additional"),
        ):
            derived_gradients[arm] = source_gradients[gradient_name]
        for arm, arm_gradients in derived_gradients.items():
            source_vector = helpers.canonical_vector(names, arm_gradients)
            row[arm] = {}
            for metric, held_name in (
                ("log_mse", "hard_log_mse"),
                ("raw_squared_error", "hard_raw_squared_error"),
                ("body_mae", "hard_body_mae"),
                ("time_nll", "hard_time_nll"),
            ):
                held_vector = helpers.canonical_vector(names, folds[held][held_name])
                row[arm][metric] = helpers.directional_alignment(source_vector, held_vector)
            row[arm]["blocks"] = {
                block: {
                    metric: helpers.directional_alignment(
                        helpers.canonical_vector(names, arm_gradients, block=block),
                        helpers.canonical_vector(names, folds[held][held_name], block=block),
                    )
                    for metric, held_name in (
                        ("log_mse", "hard_log_mse"),
                        ("raw_squared_error", "hard_raw_squared_error"),
                        ("body_mae", "hard_body_mae"),
                    )
                }
                for block in ("query", "key")
            }
        result[f"fold{source}_to_fold{held}"] = row
    return result


def analyze_dataset(
    row: Mapping[str, Any],
    frozen: FrozenModules,
) -> dict[str, Any]:
    name = str(row["dataset"])
    for key, label in (
        ("data_path", "data"),
        ("split_manifest_path", "split manifest"),
        ("launch_contract_path", "launch contract"),
        ("checkpoint_path", "checkpoint"),
        ("summary_path", "summary"),
    ):
        hash_key = {
            "data_path": "data_sha256",
            "split_manifest_path": "split_manifest_sha256",
            "launch_contract_path": "launch_contract_sha256",
            "checkpoint_path": "checkpoint_file_sha256",
            "summary_path": "summary_sha256",
        }[key]
        verify_file(ROOT / row[key], row[hash_key], label)
    identities, parity_cache = load_cohort(row)
    frame = load_train_frame(ROOT / row["data_path"])
    require(frame.height == int(row["train_rows"]), f"{name} train row count drift")
    require(frame["oper_part_no"].n_unique() == int(row["train_series"]), f"{name} train series count drift")
    prepared = frozen.prepare_count_frame(frame)
    dataset = frozen.Dataset(
        prepared,
        lookback_weeks=int(row["lookback_weeks"]),
        max_seq_len=int(row["max_seq_len"]),
        mode="all",
        split_col="chronological_split",
        target_splits={"train"},
    )
    require(len(dataset) == int(row["train_targets"]), f"{name} train target count drift")
    verify_dataset_identity(dataset, identities["rows"])
    launch = read_json(ROOT / row["launch_contract_path"])
    model, restore = frozen.restore_b0(ROOT / row["checkpoint_path"], launch, "cpu")
    require(restore["model_state_sha256"] == row["checkpoint_state_sha256"], "B state hash drift")
    model.eval().requires_grad_(True)
    state_before = frozen.canonical_state_sha256(model.state_dict())
    names, parameters = probe_parameters(model)
    require(all(parameter.requires_grad for parameter in parameters), "Probe parameter is frozen to autograd")
    fold_results: dict[int, dict[str, Any]] = {}
    fold_gradients_by_id: dict[int, dict[str, list[torch.Tensor]]] = {}
    structural_records: dict[int, dict[str, Any]] = {}
    for fold in (0, 1):
        fold_rows = [item for item in identities["rows"] if int(item["fold"]) == fold]
        indices = [int(item["dataset_index"]) for item in fold_rows]
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=int(read_json(CONTRACT)["diagnostic_scope"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=frozen.collate,
        )
        summary, gradients, _, _, audits = fold_gradients(
            model,
            frozen,
            loader,
            body_threshold=float(row["body_threshold_train_p95"]),
            names=names,
            parameters=parameters,
            parity_cache=parity_cache,
            parity_offset=fold * 2048,
            dataset_name=name,
            fold=fold,
        )
        fold_results[fold] = summary
        fold_gradients_by_id[fold] = gradients
        structural_records[fold] = audits
    state_after = frozen.canonical_state_sha256(model.state_dict())
    require(state_after == state_before == row["checkpoint_state_sha256"], "Frozen B state changed")
    require(all(parameter.grad is None for parameter in model.parameters()), "Parameter .grad was populated")
    verify_file(ROOT / row["checkpoint_path"], row["checkpoint_file_sha256"], "checkpoint after probe")
    structural = structural_records[0]["structural"]
    for fold in (0, 1):
        require(structural_records[fold]["structural"]["normal_forward_bitwise_identity"], "Forward identity failed")
        require(all(value == 0.0 for value in structural_records[fold]["target_padding_invariance"].values()), "Leakage audit failed")
    return {
        "dataset": name,
        "scope": {
            "split": "train",
            "validation_rows_materialized": False,
            "held_out_rows_materialized": False,
            "model_parameter_updates": False,
        },
        "restore_audit": restore,
        "state_sha256_before_after": state_before,
        "parameter_blocks": {
            "names": names,
            "query_tensor_count": sum(name.startswith("encoder.") for name in names),
            "query_parameter_count": sum(parameter.numel() for name, parameter in zip(names, parameters) if name.startswith("encoder.")),
            "key_tensor_count": sum(name == "lmm.mem" for name in names),
            "key_parameter_count": sum(parameter.numel() for name, parameter in zip(names, parameters) if name == "lmm.mem"),
        },
        "cohort": {
            "target_count": 4096,
            "fold_counts": identities["fold_counts"],
            "fold_unique_series": identities["fold_unique_series"],
            "identity_sha256": identities["identity_sha256"],
            "sample_ids_sha256": row["cohort_sample_ids_sha256"],
            "cache_sha256": row["cohort_cache_sha256"],
        },
        "body_threshold_train_p95": float(row["body_threshold_train_p95"]),
        "structural_audit": structural,
        "fold_audits": structural_records,
        "folds": {str(fold): fold_results[fold] for fold in (0, 1)},
        "cross_fold": cross_fold_results(names, fold_gradients_by_id),
    }


def source_hashes() -> dict[str, str]:
    paths = (
        "paper/contracts/hard_lmm_credit_assignment_probe_v1.json",
        "paper/scripts/hard_lmm_credit_assignment_probe.py",
        "paper/scripts/run_hard_lmm_credit_assignment_probe.py",
        "simple_lab_test/search/tests/test_hard_lmm_credit_assignment_probe.py",
    )
    return {path: sha256_file(ROOT / path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    args = parser.parse_args()
    require(not RESULT.exists(), f"Refusing to overwrite {RESULT}")
    contract = read_json(CONTRACT)
    validate_contract(contract)
    source = verify_frozen_source(contract, args.frozen_source_root)
    frozen = import_frozen_modules(args.frozen_source_root)
    source["loaded_module_audit"] = audit_loaded_frozen_modules(
        contract, args.frozen_source_root.resolve()
    )
    torch.set_num_threads(int(contract["diagnostic_scope"]["torch_threads"]))
    torch.manual_seed(int(contract["diagnostic_scope"]["sample_seed"]))
    RESULT.mkdir(parents=True)
    save_json(RESULT / "execution_contract.json", contract)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "contract_sha256": sha256_file(CONTRACT),
        "source": source,
        "diagnostic_source_hashes": source_hashes(),
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "validation_rows_materialized": False,
        "held_out_rows_materialized": False,
        "model_parameter_updates": False,
        "datasets": {},
    }
    save_json(RESULT / "execution_manifest.json", manifest)
    try:
        results: dict[str, Any] = {}
        for row in contract["datasets"]:
            name = row["dataset"]
            print(json.dumps({"stage": "dataset_start", "dataset": name, "time": utc_now()}), flush=True)
            result = analyze_dataset(row, frozen)
            results[name] = result
            manifest["datasets"][name] = {
                "status": "complete",
                "state_sha256": result["state_sha256_before_after"],
                "cohort": result["cohort"],
            }
            save_json(RESULT / "analysis.json", results)
            save_json(RESULT / "execution_manifest.json", manifest)
        decision = helpers.aggregate_common_gate(results)
        save_json(RESULT / "evidence_decision.json", decision)
        require(source_hashes() == manifest["diagnostic_source_hashes"], "Diagnostic source changed during run")
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            analysis_sha256=sha256_file(RESULT / "analysis.json"),
            decision_sha256=sha256_file(RESULT / "evidence_decision.json"),
            common_gate_passed=decision["passed"],
        )
        save_json(RESULT / "execution_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", failed_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        save_json(RESULT / "execution_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
