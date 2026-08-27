#!/usr/bin/env python3
"""Attribute B0 quantity errors to its static hard-prototype residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_b0_retrieval")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_b0_retrieval")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data_loader.event_seq_data_module import (  # noqa: E402
    RMTPPWeekLookbackDataset,
    collate_week_lookback,
)
from models.TPPs.CountAwareFactory import build_count_aware_model  # noqa: E402
from models.TPPs.CountAwareTPP import (  # noqa: E402
    TITAN_MEMORY_MODE_STATIC_HARD,
    CountAwareTitanTPP,
)
from paper.scripts.count_aware_tpp_backbone.core import (  # noqa: E402
    prepare_count_frame,
    right_pad_batch,
    target_outputs,
)
from simple_lab_test.search.common.runner import (  # noqa: E402
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


VARIANT = "count_only_log_regression"
BACKBONE = "titantpp"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "paper"
    / "configs"
    / "count_aware_b0_retrieval_validation_manifest.json"
)
SOURCE_FILES = (
    "models/Titan/common/memory.py",
    "models/Titan/common/titans_mac.py",
    "models/Titan/common/tpp_gated_memory.py",
    "models/TPPs/CountAwareTPP.py",
    "models/TPPs/CountAwareFactory.py",
    "paper/scripts/count_aware_tpp_backbone/core.py",
    "paper/scripts/analyze_count_aware_b0_retrieval.py",
    "paper/configs/count_aware_b0_retrieval_validation_manifest.json",
    "data_loader/event_seq_data_module.py",
    "simple_lab_test/search/common/runner.py",
)
SCOPE_TYPES = (
    ("overall", None),
    ("quantity", "quantity_stratum"),
    ("quantity_region", "quantity_region"),
    ("history", "history_stratum"),
)
SEED_METRICS = (
    "memory_on_mae",
    "memory_off_mae",
    "mae_delta_on_minus_off",
    "mae_delta_pct_vs_off",
    "memory_on_mse",
    "memory_off_mse",
    "mse_delta_on_minus_off",
    "memory_on_rmse",
    "memory_off_rmse",
    "rmse_delta_on_minus_off",
    "memory_help_share",
    "memory_harm_share",
    "memory_tie_share",
    "retrieval_applied_share",
    "residual_nonzero_share",
    "memory_residual_norm_mean",
    "memory_residual_norm_p50",
    "memory_residual_norm_p95",
    "relative_residual_norm_mean",
    "top1_similarity_mean",
    "topk_similarity_mean",
    "similarity_margin_mean",
    "quantity_logit_residual_mean",
    "quantity_logit_residual_abs_mean",
    "log_location_shift_mean",
    "prediction_shift_mean",
    "prediction_shift_abs_mean",
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    artifact_dir: Path
    data_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--event-chunk-rows", type=int, default=100_000)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(
    path: Path,
    selected_datasets: set[str] | None,
) -> tuple[dict[str, Any], list[DatasetSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_scope") != "validation_only":
        raise ValueError("Retrieval diagnostics are validation-only")
    if payload.get("held_out_test_evaluated") is not False:
        raise ValueError("Manifest must explicitly prohibit held-out test use")

    specs = []
    for row in payload["datasets"]:
        if selected_datasets and row["dataset"] not in selected_datasets:
            continue
        specs.append(
            DatasetSpec(
                dataset=str(row["dataset"]),
                artifact_dir=resolve_path(row["artifact_dir"]),
                data_path=resolve_path(row["data_path"]),
            )
        )
    if not specs:
        raise ValueError("No datasets selected")
    if selected_datasets and {spec.dataset for spec in specs} != selected_datasets:
        missing = sorted(selected_datasets - {spec.dataset for spec in specs})
        raise ValueError(f"Unknown datasets requested: {missing}")
    return payload, specs


def validate_dataset_contract(spec: DatasetSpec) -> dict[str, Any]:
    contract_path = spec.artifact_dir / "launch_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks = {
        "dataset_matches": contract.get("dataset") == spec.dataset,
        "evaluation_scope_validation_only": (
            contract.get("evaluation_scope") == "validation_only"
        ),
        "held_out_test_unused": contract.get("held_out_test_evaluated") is False,
        "t0_role": contract.get("model_role", "t0_common_control")
        == "t0_common_control",
        "direct_log_mse_only": contract.get("quantity_variants") == [VARIANT],
        "legacy_time_head": contract.get("time_head", {}).get("mode")
        == "legacy_clamped_rmtpp",
        "data_exists": spec.data_path.exists(),
    }
    if spec.data_path.exists():
        checks["data_sha256_matches"] = sha256_file(spec.data_path) == contract.get(
            "data_sha256"
        )
    if not all(checks.values()):
        raise ValueError(
            f"Dataset contract failed for {spec.dataset}: "
            f"{json.dumps(checks, sort_keys=True)}"
        )
    return contract


def checkpoint_path(artifact_dir: Path, seed: int) -> Path:
    return (
        artifact_dir
        / "runs"
        / BACKBONE
        / VARIANT
        / f"seed_{seed}"
        / "best_val_joint_objective_model.pt"
    )


def summary_path(artifact_dir: Path, seed: int) -> Path:
    return checkpoint_path(artifact_dir, seed).with_name("summary.json")


def restore_b0(
    checkpoint: Path,
    contract: dict[str, Any],
    device: str,
) -> tuple[CountAwareTitanTPP, dict[str, Any]]:
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    checkpoint_checks = {
        "backbone": payload.get("backbone") == BACKBONE,
        "variant": payload.get("variant") == VARIANT,
        "validation_only": payload.get("evaluation_scope") == "validation_only",
        "held_out_test_unused": payload.get("held_out_test_evaluated") is False,
        "memory_mode": payload.get("encoder_config", {}).get("memory_mode")
        == TITAN_MEMORY_MODE_STATIC_HARD,
    }
    observed_state_sha = canonical_state_dict_sha256(payload["model_state_dict"])
    checkpoint_checks["state_sha256"] = observed_state_sha == payload.get(
        "model_state_sha256"
    )
    if not all(checkpoint_checks.values()):
        raise ValueError(
            f"B0 checkpoint contract failed for {checkpoint}: "
            f"{json.dumps(checkpoint_checks, sort_keys=True)}"
        )

    interface = payload["interface_meta"]
    encoder = payload["encoder_config"]
    time_contract = interface["time_head"]
    tail_contract = contract.get("tail_contract", {})
    model, _ = build_count_aware_model(
        BACKBONE,
        hidden_dim=int(encoder["d_model"]),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface.get("train_target_std", 1.0)),
        max_seq_len=int(encoder["max_len"]),
        quantity_variant=payload["variant"],
        lambda_tail=float(contract.get("lambda_tail", 0.0)),
        tail_threshold=float(tail_contract.get("threshold", 46.0)),
        tail_normalization_scale=float(tail_contract.get("normalization_scale", 46.0)),
        tail_clip_cap=float(tail_contract.get("clip_cap", 187.0)),
        tail_huber_delta=float(tail_contract.get("huber_delta", 1.0)),
        time_head_mode=str(time_contract["mode"]),
        time_scale=float(time_contract.get("time_scale", 3.0)),
        time_w_max=float(time_contract.get("time_w_max", 10.0 / 3.0)),
        time_intercept_limit=float(time_contract.get("time_intercept_limit", 30.0)),
        time_initial_intercept=float(time_contract.get("time_initial_intercept", 0.0)),
        time_wd_safety_limit=float(time_contract.get("time_wd_safety_limit", 40.0)),
        time_initial_location=time_contract.get("time_initial_location"),
        time_initial_scale=time_contract.get("time_initial_scale"),
        time_sigma_floor=float(time_contract.get("time_sigma_floor", 1e-3)),
    )
    if not isinstance(model, CountAwareTitanTPP) or model.lmm is None:
        raise TypeError("Checkpoint did not restore a B0 hard-memory model")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_file_sha256": sha256_file(checkpoint),
        "model_state_sha256": observed_state_sha,
        "checkpoint_source_revision": payload.get("source_revision"),
        "checkpoint_checks": checkpoint_checks,
    }


@torch.no_grad()
def b0_counterfactual_outputs(
    model: CountAwareTitanTPP,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare B0 with and without only its additive hard-memory residual."""
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    memory_write_mask = mask.clone()
    memory_write_mask[batch_ids, target_positions] = False

    local = model._encode_base(
        dts,
        history_quantities,
        mask,
        memory_write_mask=memory_write_mask,
    )
    if model.lmm is None:
        raise RuntimeError("B0 retrieval diagnostics require hard memory")
    residual, trace = model.lmm.retrieve(local)
    valid = mask.unsqueeze(-1).to(dtype=local.dtype)
    memory_on = (local + residual) * valid
    local = local * valid

    local_hidden = local[batch_ids, history_positions]
    residual_hidden = residual[batch_ids, history_positions]
    memory_on_hidden = memory_on[batch_ids, history_positions]
    true_quantity = quantities[batch_ids, target_positions].float()
    on_quantity = model.quantity_outputs(memory_on_hidden, true_quantity)
    off_quantity = model.quantity_outputs(local_hidden, true_quantity)
    raw_on = model.quantity_head(memory_on_hidden).squeeze(-1)
    raw_off = model.quantity_head(local_hidden).squeeze(-1)
    logit_residual = torch.sum(
        residual_hidden * model.quantity_head.weight.squeeze(0),
        dim=-1,
    )
    if not torch.allclose(
        raw_on - raw_off,
        logit_residual,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise AssertionError("Quantity logit shift is not explained by the residual")

    prototype_indices = trace["prototype_indices"][batch_ids, history_positions]
    topk_similarity = trace["topk_similarity"][batch_ids, history_positions]
    if prototype_indices.size(-1) < 1:
        raise AssertionError("B0 must retrieve at least one prototype")
    similarity_margin = (
        topk_similarity[:, 0] - topk_similarity[:, 1]
        if topk_similarity.size(-1) > 1
        else torch.zeros_like(topk_similarity[:, 0])
    )
    pred_on = on_quantity["point_prediction"]
    pred_off = off_quantity["point_prediction"]
    error_on = pred_on - true_quantity
    error_off = pred_off - true_quantity
    abs_on = error_on.abs()
    abs_off = error_off.abs()
    sq_on = error_on.square()
    sq_off = error_off.square()
    local_norm = torch.linalg.vector_norm(local_hidden, dim=-1)
    residual_norm = torch.linalg.vector_norm(residual_hidden, dim=-1)

    outputs = {
        "true_quantity": true_quantity,
        "history_length": lengths - 1,
        "pred_memory_on": pred_on,
        "pred_memory_off": pred_off,
        "prediction_shift": pred_on - pred_off,
        "log_location_memory_on": on_quantity["location"],
        "log_location_memory_off": off_quantity["location"],
        "log_location_shift": (on_quantity["location"] - off_quantity["location"]),
        "quantity_logit_residual": logit_residual,
        "abs_error_memory_on": abs_on,
        "abs_error_memory_off": abs_off,
        "abs_error_delta_on_minus_off": abs_on - abs_off,
        "squared_error_memory_on": sq_on,
        "squared_error_memory_off": sq_off,
        "squared_error_delta_on_minus_off": sq_on - sq_off,
        "local_state_norm": local_norm,
        "memory_residual_norm": residual_norm,
        "relative_residual_norm": residual_norm / local_norm.clamp_min(1e-12),
        "top1_similarity": topk_similarity[:, 0],
        "topk_similarity_mean": topk_similarity.mean(dim=-1),
        "similarity_margin": similarity_margin,
        "prototype_indices": prototype_indices,
        "topk_similarity": topk_similarity,
    }
    for name, value in outputs.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite B0 diagnostic tensor: {name}")
    return outputs


def verify_memory_on_equivalence(
    model: CountAwareTitanTPP,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    diagnostic: dict[str, torch.Tensor],
    lambda_log_qty: float,
) -> float:
    with torch.no_grad():
        official = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=lambda_log_qty,
        )
    maximum_difference = float(
        torch.max(torch.abs(official["pred_qty"] - diagnostic["pred_memory_on"])).item()
    )
    if maximum_difference > 1e-5:
        raise AssertionError(
            "Diagnostic memory-on prediction drifted from official evaluation: "
            f"max_abs_diff={maximum_difference}"
        )
    return maximum_difference


class EventParquetSink:
    def __init__(self, directory: Path, chunk_rows: int) -> None:
        if directory.exists():
            raise FileExistsError(f"Event output already exists: {directory}")
        if chunk_rows < 1:
            raise ValueError("event_chunk_rows must be positive")
        directory.mkdir(parents=True)
        self.directory = directory
        self.chunk_rows = int(chunk_rows)
        self.frames: list[pl.DataFrame] = []
        self.buffered_rows = 0
        self.part = 0

    def append(self, frame: pl.DataFrame) -> None:
        self.frames.append(frame)
        self.buffered_rows += frame.height
        if self.buffered_rows >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        if not self.frames:
            return
        frame = pl.concat(self.frames, how="vertical")
        frame.write_parquet(
            self.directory / f"part_{self.part:05d}.parquet",
            compression="zstd",
            statistics=True,
        )
        self.frames.clear()
        self.buffered_rows = 0
        self.part += 1

    def close(self) -> None:
        self.flush()
        if self.part < 1:
            raise ValueError("No event diagnostics were written")


class PrototypeUsageAccumulator:
    def __init__(self, dataset: str, seed: int, memory_size: int, topk: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.memory_size = int(memory_size)
        self.topk = int(topk)
        self.events: dict[tuple[str, str], int] = {}
        self.selections: dict[tuple[str, str], np.ndarray] = {}
        self.rank1: dict[tuple[str, str], np.ndarray] = {}
        self.similarity_sum: dict[tuple[str, str], np.ndarray] = {}

    def update(
        self,
        prototype_indices: np.ndarray,
        topk_similarity: np.ndarray,
        scopes: dict[str, np.ndarray],
    ) -> None:
        if prototype_indices.shape != topk_similarity.shape:
            raise ValueError("Prototype index and similarity shapes must match")
        if prototype_indices.shape[1] != self.topk:
            raise ValueError("Unexpected B0 top-k width")
        for scope_type, labels in scopes.items():
            for scope in np.unique(labels):
                selected = labels == scope
                key = (scope_type, str(scope))
                event_count = int(selected.sum())
                self.events[key] = self.events.get(key, 0) + event_count
                if key not in self.selections:
                    self.selections[key] = np.zeros(self.memory_size, dtype=np.int64)
                    self.rank1[key] = np.zeros(self.memory_size, dtype=np.int64)
                    self.similarity_sum[key] = np.zeros(
                        self.memory_size,
                        dtype=np.float64,
                    )
                indices = prototype_indices[selected]
                similarities = topk_similarity[selected]
                flat_indices = indices.reshape(-1)
                self.selections[key] += np.bincount(
                    flat_indices,
                    minlength=self.memory_size,
                )
                self.rank1[key] += np.bincount(
                    indices[:, 0],
                    minlength=self.memory_size,
                )
                self.similarity_sum[key] += np.bincount(
                    flat_indices,
                    weights=similarities.reshape(-1),
                    minlength=self.memory_size,
                )

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for (scope_type, scope), event_count in sorted(self.events.items()):
            selection_count = self.selections[(scope_type, scope)]
            rank1_count = self.rank1[(scope_type, scope)]
            similarity_sum = self.similarity_sum[(scope_type, scope)]
            expected_selections = event_count * self.topk
            if int(selection_count.sum()) != expected_selections:
                raise AssertionError("Prototype selection accounting mismatch")
            if int(rank1_count.sum()) != event_count:
                raise AssertionError("Rank-1 prototype accounting mismatch")
            for prototype_id in range(self.memory_size):
                count = int(selection_count[prototype_id])
                rows.append(
                    {
                        "dataset": self.dataset,
                        "seed": self.seed,
                        "scope_type": scope_type,
                        "scope": scope,
                        "prototype_id": prototype_id,
                        "event_count": event_count,
                        "selection_count": count,
                        "selection_share": count / expected_selections,
                        "event_hit_share": count / event_count,
                        "rank1_count": int(rank1_count[prototype_id]),
                        "rank1_share": int(rank1_count[prototype_id]) / event_count,
                        "selected_similarity_mean": (
                            float(similarity_sum[prototype_id]) / count
                            if count
                            else 0.0
                        ),
                        "prototype_used": count > 0,
                    }
                )
        return rows


def quantity_region_labels(
    quantity_labels: np.ndarray,
    region_contract: dict[str, list[str]],
) -> np.ndarray:
    label_to_region = {
        label: region for region, labels in region_contract.items() for label in labels
    }
    missing = sorted(set(quantity_labels.tolist()) - set(label_to_region))
    if missing:
        raise ValueError(f"Quantity strata missing from region contract: {missing}")
    return np.asarray([label_to_region[label] for label in quantity_labels])


def build_event_frame(
    *,
    dataset: str,
    seed: int,
    event_offset: int,
    dataset_index: list[tuple[int, int]],
    part_indices: torch.Tensor,
    diagnostic: dict[str, torch.Tensor],
    quantity_contract: dict[str, Any],
    history_contract: dict[str, Any],
    region_contract: dict[str, list[str]],
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    true_quantity = diagnostic["true_quantity"].cpu().numpy().astype(np.float64)
    history_length = diagnostic["history_length"].cpu().numpy().astype(np.int64)
    count = true_quantity.size
    sample_index = dataset_index[event_offset : event_offset + count]
    if len(sample_index) != count:
        raise AssertionError("Dataset event index ended before the loader")
    expected_parts = np.asarray([row[0] for row in sample_index], dtype=np.int64)
    observed_parts = part_indices.cpu().numpy().astype(np.int64)
    if not np.array_equal(expected_parts, observed_parts):
        raise AssertionError("Validation loader order no longer matches dataset.index")

    quantity_ids = np.searchsorted(
        np.asarray(quantity_contract["boundaries"], dtype=np.float64),
        true_quantity,
        side="left",
    )
    quantity_specs = sorted(
        quantity_contract["strata"],
        key=lambda row: int(row["stratum_order"]),
    )
    quantity_labels = np.asarray(
        [quantity_specs[index]["stratum"] for index in quantity_ids]
    )
    region_labels = quantity_region_labels(quantity_labels, region_contract)
    history_ids = np.searchsorted(
        np.asarray(history_contract["boundaries"], dtype=np.int64),
        history_length,
        side="left",
    )
    history_specs = sorted(
        history_contract["strata"],
        key=lambda row: int(row["stratum_order"]),
    )
    history_labels = np.asarray(
        [history_specs[index]["stratum"] for index in history_ids]
    )
    prototype_indices = diagnostic["prototype_indices"].cpu().numpy().astype(np.int64)
    topk_similarity = diagnostic["topk_similarity"].cpu().numpy().astype(np.float64)

    columns: dict[str, Any] = {
        "dataset": [dataset] * count,
        "seed": np.full(count, seed, dtype=np.int64),
        "event_index": np.arange(
            event_offset,
            event_offset + count,
            dtype=np.int64,
        ),
        "series_index": expected_parts,
        "target_event_index": np.asarray(
            [row[1] + 1 for row in sample_index],
            dtype=np.int64,
        ),
        "true_quantity": true_quantity,
        "history_length": history_length,
        "quantity_stratum": quantity_labels,
        "quantity_region": region_labels,
        "history_stratum": history_labels,
    }
    tensor_columns = (
        "pred_memory_on",
        "pred_memory_off",
        "prediction_shift",
        "log_location_memory_on",
        "log_location_memory_off",
        "log_location_shift",
        "quantity_logit_residual",
        "abs_error_memory_on",
        "abs_error_memory_off",
        "abs_error_delta_on_minus_off",
        "squared_error_memory_on",
        "squared_error_memory_off",
        "squared_error_delta_on_minus_off",
        "local_state_norm",
        "memory_residual_norm",
        "relative_residual_norm",
        "top1_similarity",
        "topk_similarity_mean",
        "similarity_margin",
    )
    for name in tensor_columns:
        columns[name] = diagnostic[name].cpu().numpy().astype(np.float64)
    delta = columns["abs_error_delta_on_minus_off"]
    columns["memory_helped_abs"] = delta < -1e-12
    columns["memory_harmed_abs"] = delta > 1e-12
    columns["memory_tied_abs"] = np.abs(delta) <= 1e-12
    columns["retrieval_applied"] = np.ones(count, dtype=np.bool_)
    columns["residual_nonzero"] = columns["memory_residual_norm"] > 1e-12
    for rank in range(prototype_indices.shape[1]):
        columns[f"prototype_rank_{rank + 1}"] = prototype_indices[:, rank]
        columns[f"similarity_rank_{rank + 1}"] = topk_similarity[:, rank]

    scopes = {
        "overall": np.asarray(["overall"] * count),
        "quantity": quantity_labels,
        "quantity_region": region_labels,
        "history": history_labels,
    }
    return pl.DataFrame(columns), scopes


def aggregate_event_shards(event_dir: Path) -> pl.DataFrame:
    lazy = pl.scan_parquet(str(event_dir / "part_*.parquet"))
    aggregations = [
        pl.len().alias("event_count"),
        pl.col("abs_error_memory_on").mean().alias("memory_on_mae"),
        pl.col("abs_error_memory_off").mean().alias("memory_off_mae"),
        pl.col("abs_error_delta_on_minus_off").mean().alias("mae_delta_on_minus_off"),
        pl.col("squared_error_memory_on").mean().alias("memory_on_mse"),
        pl.col("squared_error_memory_off").mean().alias("memory_off_mse"),
        pl.col("squared_error_delta_on_minus_off")
        .mean()
        .alias("mse_delta_on_minus_off"),
        pl.col("memory_helped_abs").cast(pl.Float64).mean().alias("memory_help_share"),
        pl.col("memory_harmed_abs").cast(pl.Float64).mean().alias("memory_harm_share"),
        pl.col("memory_tied_abs").cast(pl.Float64).mean().alias("memory_tie_share"),
        pl.col("retrieval_applied")
        .cast(pl.Float64)
        .mean()
        .alias("retrieval_applied_share"),
        pl.col("residual_nonzero")
        .cast(pl.Float64)
        .mean()
        .alias("residual_nonzero_share"),
        pl.col("local_state_norm").mean().alias("local_state_norm_mean"),
        pl.col("memory_residual_norm").mean().alias("memory_residual_norm_mean"),
        pl.col("memory_residual_norm").quantile(0.5).alias("memory_residual_norm_p50"),
        pl.col("memory_residual_norm").quantile(0.95).alias("memory_residual_norm_p95"),
        pl.col("relative_residual_norm").mean().alias("relative_residual_norm_mean"),
        pl.col("top1_similarity").mean().alias("top1_similarity_mean"),
        pl.col("topk_similarity_mean").mean().alias("topk_similarity_mean"),
        pl.col("similarity_margin").mean().alias("similarity_margin_mean"),
        pl.col("quantity_logit_residual").mean().alias("quantity_logit_residual_mean"),
        pl.col("quantity_logit_residual")
        .abs()
        .mean()
        .alias("quantity_logit_residual_abs_mean"),
        pl.col("log_location_shift").mean().alias("log_location_shift_mean"),
        pl.col("prediction_shift").mean().alias("prediction_shift_mean"),
        pl.col("prediction_shift").abs().mean().alias("prediction_shift_abs_mean"),
    ]
    scoped_frames = []
    for scope_type, source_column in SCOPE_TYPES:
        scope_expression = (
            pl.lit("overall") if source_column is None else pl.col(source_column)
        )
        scoped = lazy.with_columns(
            pl.lit(scope_type).alias("scope_type"),
            scope_expression.alias("scope"),
        )
        scoped_frames.append(
            scoped.group_by(["dataset", "seed", "scope_type", "scope"]).agg(
                aggregations
            )
        )
    result = pl.concat(scoped_frames, how="vertical").collect()
    return result.with_columns(
        pl.col("memory_on_mse").sqrt().alias("memory_on_rmse"),
        pl.col("memory_off_mse").sqrt().alias("memory_off_rmse"),
        (pl.col("memory_on_mse").sqrt() - pl.col("memory_off_mse").sqrt()).alias(
            "rmse_delta_on_minus_off"
        ),
        (
            100.0
            * pl.col("mae_delta_on_minus_off")
            / pl.col("memory_off_mae").clip(1e-12, None)
        ).alias("mae_delta_pct_vs_off"),
    ).sort(["scope_type", "scope"])


def aggregate_seed_metrics(seed_metrics: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        pl.col("event_count").first().alias("event_count_per_seed")
    ]
    for metric in SEED_METRICS:
        expressions.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).std(ddof=1).fill_null(0.0).alias(f"{metric}_std"),
            ]
        )
    return (
        seed_metrics.group_by(["dataset", "scope_type", "scope"])
        .agg(expressions)
        .sort(["dataset", "scope_type", "scope"])
    )


def aggregate_prototype_usage(seed_usage: pl.DataFrame) -> pl.DataFrame:
    metrics = (
        "selection_share",
        "event_hit_share",
        "rank1_share",
        "selected_similarity_mean",
    )
    expressions = [
        pl.col("event_count").first().alias("event_count_per_seed"),
        pl.col("prototype_used").cast(pl.Float64).mean().alias("seed_use_fraction"),
    ]
    for metric in metrics:
        expressions.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).std(ddof=1).fill_null(0.0).alias(f"{metric}_std"),
            ]
        )
    return (
        seed_usage.group_by(["dataset", "scope_type", "scope", "prototype_id"])
        .agg(expressions)
        .sort(["dataset", "scope_type", "scope", "prototype_id"])
    )


def dataset_frame_and_loader(
    spec: DatasetSpec,
    contract: dict[str, Any],
    batch_size_override: int | None,
) -> tuple[pl.DataFrame, Any]:
    frame = prepare_count_frame(pl.read_parquet(spec.data_path))
    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=int(contract["lookback_weeks"]),
        max_seq_len=int(contract["max_seq_len"]),
        val_ratio=0.2,
        mode="all",
        split_col="chronological_split",
        target_splits={"validation"},
    )
    loader = DataLoader(
        dataset,
        batch_size=(
            int(batch_size_override)
            if batch_size_override is not None
            else int(contract["batch_size"])
        ),
        shuffle=False,
        collate_fn=collate_week_lookback,
        num_workers=0,
    )
    return frame, loader


def run_dataset(
    *,
    spec: DatasetSpec,
    contract: dict[str, Any],
    region_contract: dict[str, list[str]],
    output_dir: Path,
    device: str,
    seeds: list[int],
    batch_size: int | None,
    max_batches: int | None,
    event_chunk_rows: int,
) -> tuple[pl.DataFrame, pl.DataFrame, list[dict[str, Any]]]:
    dataset_dir = output_dir / "datasets" / spec.dataset
    dataset_dir.mkdir(parents=True)
    _, loader = dataset_frame_and_loader(spec, contract, batch_size)
    dataset_index = list(loader.dataset.index)
    seed_frames = []
    prototype_rows: list[dict[str, Any]] = []
    checkpoint_audit = []

    for seed in seeds:
        checkpoint = checkpoint_path(spec.artifact_dir, seed)
        summary_file = summary_path(spec.artifact_dir, seed)
        if not checkpoint.exists() or not summary_file.exists():
            raise FileNotFoundError(
                f"Missing B0 seed {seed} artifact for {spec.dataset}"
            )
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        model, checkpoint_record = restore_b0(checkpoint, contract, device)
        if model.lmm is None or model.lmm.mem is None:
            raise RuntimeError("B0 prototype bank is unavailable")
        sink = EventParquetSink(
            dataset_dir / "events" / f"seed_{seed}",
            event_chunk_rows,
        )
        usage = PrototypeUsageAccumulator(
            spec.dataset,
            seed,
            model.lmm.mem_size,
            min(model.lmm.topk, model.lmm.mem_size),
        )
        event_offset = 0
        equivalence_max_abs_diff = 0.0
        started = time.time()
        for batch_index, (_, dts, mask, part_indices, quantities) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if quantities is None:
                raise ValueError("B0 diagnostics require raw quantities")
            device_dts = dts.to(device)
            device_mask = mask.to(device)
            device_quantities = quantities.to(device)
            diagnostic = b0_counterfactual_outputs(
                model,
                device_dts,
                device_mask,
                device_quantities,
            )
            if batch_index == 0:
                equivalence_max_abs_diff = verify_memory_on_equivalence(
                    model,
                    device_dts,
                    device_mask,
                    device_quantities,
                    diagnostic,
                    float(contract.get("lambda_log_qty", 1.0)),
                )
            event_frame, scopes = build_event_frame(
                dataset=spec.dataset,
                seed=seed,
                event_offset=event_offset,
                dataset_index=dataset_index,
                part_indices=part_indices,
                diagnostic=diagnostic,
                quantity_contract=contract["quantity_contract"],
                history_contract=contract["history_length_contract"],
                region_contract=region_contract,
            )
            sink.append(event_frame)
            usage.update(
                diagnostic["prototype_indices"].cpu().numpy().astype(np.int64),
                diagnostic["topk_similarity"].cpu().numpy().astype(np.float64),
                scopes,
            )
            event_offset += event_frame.height
        sink.close()
        expected_count = (
            min(len(dataset_index), max_batches * loader.batch_size)
            if max_batches is not None
            else len(dataset_index)
        )
        if event_offset != expected_count:
            raise AssertionError(
                f"Unexpected evaluated count for {spec.dataset} seed {seed}: "
                f"{event_offset} != {expected_count}"
            )
        seed_scope = aggregate_event_shards(dataset_dir / "events" / f"seed_{seed}")
        seed_frames.append(seed_scope)
        prototype_rows.extend(usage.rows())
        overall = seed_scope.filter(
            (pl.col("scope_type") == "overall") & (pl.col("scope") == "overall")
        ).row(0, named=True)
        expected_mae = float(summary["best_val_qty_mae"])
        expected_rmse = float(summary["best_val_qty_rmse"])
        full_validation = max_batches is None
        mae_difference = abs(float(overall["memory_on_mae"]) - expected_mae)
        rmse_difference = abs(float(overall["memory_on_rmse"]) - expected_rmse)
        prediction_match = not full_validation or (
            mae_difference <= 1e-5 and rmse_difference <= 1e-5
        )
        if not prediction_match:
            raise AssertionError(
                f"Memory-on metrics do not reproduce {spec.dataset} seed {seed}"
            )
        checkpoint_audit.append(
            {
                "dataset": spec.dataset,
                "seed": seed,
                "evaluated_count": event_offset,
                "full_validation": full_validation,
                "official_memory_on_equivalence_max_abs_diff": (
                    equivalence_max_abs_diff
                ),
                "artifact_qty_mae": expected_mae,
                "diagnostic_memory_on_mae": float(overall["memory_on_mae"]),
                "qty_mae_abs_diff": mae_difference,
                "artifact_qty_rmse": expected_rmse,
                "diagnostic_memory_on_rmse": float(overall["memory_on_rmse"]),
                "qty_rmse_abs_diff": rmse_difference,
                "prediction_contract_match": prediction_match,
                "elapsed_seconds": time.time() - started,
                **checkpoint_record,
            }
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    seed_metrics = pl.concat(seed_frames, how="vertical")
    seed_usage = pl.DataFrame(prototype_rows)
    seed_metrics.write_csv(dataset_dir / "seed_scope_metrics.csv")
    aggregate_seed_metrics(seed_metrics).write_csv(dataset_dir / "scope_summary.csv")
    seed_usage.write_csv(dataset_dir / "prototype_usage_seed.csv")
    aggregate_prototype_usage(seed_usage).write_csv(
        dataset_dir / "prototype_usage_summary.csv"
    )
    write_json(dataset_dir / "checkpoint_audit.json", checkpoint_audit)
    return seed_metrics, seed_usage, checkpoint_audit


def build_decision(
    seed_metrics: pl.DataFrame,
    checkpoint_audit: list[dict[str, Any]],
    full_validation: bool,
) -> dict[str, Any]:
    decisions = []
    for dataset in seed_metrics["dataset"].unique().sort().to_list():
        regions = seed_metrics.filter(
            (pl.col("dataset") == dataset) & (pl.col("scope_type") == "quantity_region")
        )
        body = regions.filter(pl.col("scope") == "body_le_p95")
        extreme = regions.filter(pl.col("scope") == "extreme_tail_gt_p99")
        if body.height < 1 or extreme.height < 1:
            if full_validation:
                raise ValueError(f"Missing body/extreme-tail scopes for {dataset}")
            decisions.append(
                {
                    "dataset": dataset,
                    "seed_count": 0,
                    "body_definition": "validation quantity <= train p95",
                    "extreme_tail_definition": "validation quantity > train p99",
                    "body_mae_delta_on_minus_off_mean": None,
                    "body_mse_delta_on_minus_off_mean": None,
                    "extreme_tail_mae_delta_on_minus_off_mean": None,
                    "extreme_tail_mse_delta_on_minus_off_mean": None,
                    "body_harm_on_both_errors_all_seeds": False,
                    "extreme_tail_improvement_on_both_errors_all_seeds": False,
                    "verdict": "insufficient_scope_coverage",
                }
            )
            continue
        body_mae = body["mae_delta_on_minus_off"].to_numpy()
        extreme_mae = extreme["mae_delta_on_minus_off"].to_numpy()
        body_mse = body["mse_delta_on_minus_off"].to_numpy()
        extreme_mse = extreme["mse_delta_on_minus_off"].to_numpy()
        body_harm_all = bool(np.all(body_mae > 0.0) and np.all(body_mse > 0.0))
        extreme_improve_all = bool(
            np.all(extreme_mae < 0.0) and np.all(extreme_mse < 0.0)
        )
        if not full_validation:
            verdict = "smoke_only"
        elif body_harm_all and extreme_improve_all:
            verdict = "body_harm_extreme_tail_improvement_confirmed"
        elif bool(np.all(body_mae <= 0.0) and np.all(extreme_mae >= 0.0)):
            verdict = "pattern_contradicted"
        else:
            verdict = "mixed_by_seed_or_error_metric"
        decisions.append(
            {
                "dataset": dataset,
                "seed_count": body.height,
                "body_definition": "validation quantity <= train p95",
                "extreme_tail_definition": "validation quantity > train p99",
                "body_mae_delta_on_minus_off_mean": float(body_mae.mean()),
                "body_mse_delta_on_minus_off_mean": float(body_mse.mean()),
                "extreme_tail_mae_delta_on_minus_off_mean": float(extreme_mae.mean()),
                "extreme_tail_mse_delta_on_minus_off_mean": float(extreme_mse.mean()),
                "body_harm_on_both_errors_all_seeds": body_harm_all,
                "extreme_tail_improvement_on_both_errors_all_seeds": (
                    extreme_improve_all
                ),
                "verdict": verdict,
            }
        )
    all_prediction_matches = all(
        bool(row["prediction_contract_match"]) for row in checkpoint_audit
    )
    return {
        "schema_version": 1,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "full_validation": full_validation,
        "intervention": "remove only B0 additive hard-prototype residual",
        "same_checkpoint_local_encoder_and_quantity_head": True,
        "memory_on_reproduces_official_evaluation": all_prediction_matches,
        "inference_time_residual_attribution_valid": all_prediction_matches,
        "training_time_no_memory_effect_identified": False,
        "counterfactual_limit": (
            "The memory-off path reuses a checkpoint trained with memory. It "
            "isolates the inference-time residual, not retraining adaptation."
        ),
        "datasets": decisions,
        "confirmed_dataset_count": sum(
            row["verdict"] == "body_harm_extreme_tail_improvement_confirmed"
            for row in decisions
        ),
    }


def render_analysis(decision: dict[str, Any]) -> str:
    lines = [
        "# B0 hard-prototype retrieval diagnostic",
        "",
        "## Contract",
        "",
        "- Scope: validation only; held-out test was not read or evaluated.",
        "- Counterfactual: one frozen B0 checkpoint, local state versus local state plus the retrieved prototype residual.",
        "- Negative on-minus-off error delta means the residual improved the event; positive means it harmed the event.",
        "- The memory-off path is not a separately retrained no-memory model, so this audit identifies inference-time residual attribution only.",
        "",
        "## Body and extreme-tail result",
        "",
        "| Dataset | Body MAE delta | Body MSE delta | >p99 MAE delta | >p99 MSE delta | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in decision["datasets"]:
        if row["body_mae_delta_on_minus_off_mean"] is None:
            lines.append(
                f"| {row['dataset']} | n/a | n/a | n/a | n/a | {row['verdict']} |"
            )
            continue
        lines.append(
            "| {dataset} | {body_mae:.6f} | {body_mse:.6f} | "
            "{tail_mae:.6f} | {tail_mse:.6f} | {verdict} |".format(
                dataset=row["dataset"],
                body_mae=row["body_mae_delta_on_minus_off_mean"],
                body_mse=row["body_mse_delta_on_minus_off_mean"],
                tail_mae=row["extreme_tail_mae_delta_on_minus_off_mean"],
                tail_mse=row["extreme_tail_mse_delta_on_minus_off_mean"],
                verdict=row["verdict"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Because the two paths share every operation except the additive hard-memory residual, each event-level prediction and error difference is attributable to that residual at inference time. Any claim about how training would adapt without memory requires a separate matched retraining experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def write_source_manifest(output_dir: Path) -> None:
    lines = []
    for relative in SOURCE_FILES:
        source = PROJECT_ROOT / relative
        lines.append(f"{sha256_file(source)}  {relative}")
    (output_dir / "source_manifest.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max_batches must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = set(args.datasets) if args.datasets else None
    manifest, specs = load_manifest(args.manifest, selected)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    launch_record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "source_revision": git_revision(),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "device": args.device,
        "max_batches": args.max_batches,
        "event_chunk_rows": args.event_chunk_rows,
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "datasets": [spec.dataset for spec in specs],
    }
    write_json(args.output_dir / "launch_contract.json", launch_record)
    write_source_manifest(args.output_dir)

    all_seed_metrics = []
    all_seed_usage = []
    all_checkpoint_audit: list[dict[str, Any]] = []
    contract_audit = []
    try:
        for spec in specs:
            contract = validate_dataset_contract(spec)
            available_seeds = [int(seed) for seed in contract["seeds"]]
            seeds = args.seeds if args.seeds else available_seeds
            if not set(seeds).issubset(available_seeds):
                raise ValueError(
                    f"Requested seeds are outside {spec.dataset} contract: {seeds}"
                )
            print(
                f"[{spec.dataset}] validation retrieval diagnostic: seeds={seeds}",
                flush=True,
            )
            seed_metrics, seed_usage, checkpoint_audit = run_dataset(
                spec=spec,
                contract=contract,
                region_contract=manifest["quantity_regions"],
                output_dir=args.output_dir,
                device=args.device,
                seeds=seeds,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                event_chunk_rows=args.event_chunk_rows,
            )
            all_seed_metrics.append(seed_metrics)
            all_seed_usage.append(seed_usage)
            all_checkpoint_audit.extend(checkpoint_audit)
            contract_audit.append(
                {
                    "dataset": spec.dataset,
                    "artifact_dir": str(spec.artifact_dir),
                    "data_path": str(spec.data_path),
                    "data_sha256": contract["data_sha256"],
                    "source_revision": contract.get("source_revision"),
                    "seeds": seeds,
                    "quantity_contract": contract["quantity_contract"],
                    "history_length_contract": contract["history_length_contract"],
                    "lookback_weeks": contract["lookback_weeks"],
                    "max_seq_len": contract["max_seq_len"],
                    "batch_size": args.batch_size or contract["batch_size"],
                    "evaluation_scope": contract["evaluation_scope"],
                    "held_out_test_evaluated": contract["held_out_test_evaluated"],
                }
            )

        seed_metrics = pl.concat(all_seed_metrics, how="vertical")
        seed_usage = pl.concat(all_seed_usage, how="vertical")
        scope_summary = aggregate_seed_metrics(seed_metrics)
        prototype_summary = aggregate_prototype_usage(seed_usage)
        seed_metrics.write_csv(args.output_dir / "seed_scope_metrics.csv")
        scope_summary.write_csv(args.output_dir / "scope_summary.csv")
        seed_usage.write_csv(args.output_dir / "prototype_usage_seed.csv")
        prototype_summary.write_csv(args.output_dir / "prototype_usage_summary.csv")
        write_json(args.output_dir / "checkpoint_audit.json", all_checkpoint_audit)
        write_json(args.output_dir / "dataset_contract_audit.json", contract_audit)
        decision = build_decision(
            seed_metrics,
            all_checkpoint_audit,
            full_validation=args.max_batches is None,
        )
        write_json(args.output_dir / "decision.json", decision)
        (args.output_dir / "analysis.md").write_text(
            render_analysis(decision),
            encoding="utf-8",
        )
        launch_record.update(
            {
                "status": "complete",
                "completed_at_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "dataset_count": len(specs),
                "checkpoint_count": len(all_checkpoint_audit),
                "all_values_finite": all(
                    math.isfinite(float(value))
                    for row in seed_metrics.iter_rows(named=True)
                    for key, value in row.items()
                    if key not in {"dataset", "scope_type", "scope"}
                ),
            }
        )
        write_json(args.output_dir / "launch_contract.json", launch_record)
    except Exception:
        launch_record.update(
            {
                "status": "failed",
                "failed_at_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        write_json(args.output_dir / "launch_contract.json", launch_record)
        raise


if __name__ == "__main__":
    main()
