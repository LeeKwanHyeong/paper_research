#!/usr/bin/env python3
"""Audit B1 neural-memory attribution and cost on frozen validation checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_titantpp_mac_audit")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_titantpp_mac_audit")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from models.TPPs.CountAwareFactory import build_count_aware_model  # noqa: E402
from models.TPPs.CountAwareTPP import (  # noqa: E402
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_MEMORY_MODE_TITANS_MAC,
    CountAwareTitanTPP,
)
from paper.scripts.analyze_count_aware_b0_retrieval import (  # noqa: E402
    DatasetSpec,
    dataset_frame_and_loader,
    quantity_region_labels,
    resolve_path,
    sha256_file,
    write_json,
)
from paper.scripts.count_aware_tpp_backbone.core import (  # noqa: E402
    right_pad_batch,
    target_outputs,
)
from simple_lab_test.search.common.runner import (  # noqa: E402
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


VARIANT = "count_only_log_regression"
B0_BACKBONE = "titantpp"
B1_BACKBONE = "titantpp_titans_mac"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "paper/configs/count_aware_titantpp_mac_audit_manifest.json"
)
SOURCE_FILES = (
    "models/Titan/common/titans_mac.py",
    "models/TPPs/CountAwareTPP.py",
    "models/TPPs/CountAwareFactory.py",
    "paper/contracts/count_aware_titantpp_mac_primary_v1.json",
    "paper/configs/count_aware_titantpp_mac_audit_manifest.json",
    "paper/scripts/analyze_count_aware_titantpp_mac.py",
    "paper/scripts/count_aware_tpp_backbone/core.py",
)
SCOPE_TYPES = (
    ("overall", None),
    ("quantity", "quantity_stratum"),
    ("quantity_region", "quantity_region"),
    ("history", "history_stratum"),
    ("surprise", "causal_surprise_stratum"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--runtime-batches", type=int, default=5)
    return parser.parse_args()


def load_manifest(
    path: Path,
    selected_datasets: set[str] | None,
) -> tuple[dict[str, Any], list[DatasetSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_scope") != "validation_only":
        raise ValueError("TitanTPP-MAC audit must remain validation-only")
    if payload.get("held_out_test_evaluated") is not False:
        raise ValueError("Held-out test must remain locked")
    specs = []
    for row in payload["datasets"]:
        if selected_datasets and row["dataset"] not in selected_datasets:
            continue
        specs.append(
            DatasetSpec(
                dataset=str(row["dataset"]),
                contract_dataset=str(row.get("contract_dataset", row["dataset"])),
                artifact_dir=resolve_path(row["artifact_dir"]),
                data_path=resolve_path(row["data_path"]),
            )
        )
    observed = {spec.dataset for spec in specs}
    if selected_datasets and observed != selected_datasets:
        raise ValueError(
            f"Unknown datasets requested: {sorted(selected_datasets - observed)}"
        )
    if not specs:
        raise ValueError("No audit datasets selected")
    return payload, specs


def validate_dataset_contract(spec: DatasetSpec) -> dict[str, Any]:
    path = spec.artifact_dir / "launch_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "status_complete": contract.get("status") == "complete",
        "dataset_matches": contract.get("dataset") == spec.contract_dataset,
        "model_role": contract.get("model_role") == "titan_b012_screening",
        "validation_only": contract.get("evaluation_scope") == "validation_only",
        "held_out_unused": contract.get("held_out_test_evaluated") is False,
        "direct_log_mse": contract.get("quantity_variants") == [VARIANT],
        "tail_loss_disabled": math.isclose(
            float(contract.get("lambda_tail", math.nan)),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "common_time_head": contract.get("time_head", {}).get("mode")
        == "legacy_clamped_rmtpp",
        "data_exists": spec.data_path.exists(),
    }
    if spec.data_path.exists():
        checks["data_sha256"] = sha256_file(spec.data_path) == contract.get(
            "data_sha256"
        )
    if not all(checks.values()):
        raise ValueError(
            f"Dataset contract failed for {spec.dataset}: "
            f"{json.dumps(checks, sort_keys=True)}"
        )
    return contract


def run_dir(artifact_dir: Path, backbone: str, seed: int) -> Path:
    return artifact_dir / "runs" / backbone / VARIANT / f"seed_{seed}"


def restore_model(
    artifact_dir: Path,
    contract: dict[str, Any],
    backbone: str,
    seed: int,
    device: str,
) -> tuple[CountAwareTitanTPP, dict[str, Any], dict[str, Any]]:
    directory = run_dir(artifact_dir, backbone, seed)
    checkpoint = directory / "best_val_joint_objective_model.pt"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    expected_mode = (
        TITAN_MEMORY_MODE_STATIC_HARD
        if backbone == B0_BACKBONE
        else TITAN_MEMORY_MODE_TITANS_MAC
    )
    checks = {
        "backbone": payload.get("backbone") == backbone,
        "variant": payload.get("variant") == VARIANT,
        "seed": int(payload.get("seed", -1)) == seed,
        "validation_only": payload.get("evaluation_scope") == "validation_only",
        "held_out_unused": payload.get("held_out_test_evaluated") is False,
        "memory_mode": payload.get("encoder_config", {}).get("memory_mode")
        == expected_mode,
        "contract_id": payload.get("encoder_config", {}).get(
            "backbone_contract_id"
        )
        == ("B0" if backbone == B0_BACKBONE else "B1"),
    }
    state_sha = canonical_state_dict_sha256(payload["model_state_dict"])
    checks["state_sha256"] = state_sha == payload.get("model_state_sha256")
    if not all(checks.values()):
        raise ValueError(
            f"Checkpoint contract failed for {checkpoint}: "
            f"{json.dumps(checks, sort_keys=True)}"
        )

    interface = payload["interface_meta"]
    encoder = payload["encoder_config"]
    time_contract = interface["time_head"]
    tail_contract = contract["tail_contract"]
    model, _ = build_count_aware_model(
        backbone,
        hidden_dim=int(encoder["d_model"]),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface.get("train_target_std", 1.0)),
        max_seq_len=int(encoder["max_len"]),
        quantity_variant=VARIANT,
        lambda_tail=0.0,
        tail_threshold=float(tail_contract["threshold"]),
        tail_normalization_scale=float(tail_contract["normalization_scale"]),
        tail_clip_cap=float(tail_contract["clip_cap"]),
        tail_huber_delta=float(tail_contract["huber_delta"]),
        time_head_mode=str(time_contract["mode"]),
        time_scale=float(time_contract["time_scale"]),
        time_w_max=float(time_contract["time_w_max"]),
        time_intercept_limit=float(time_contract["time_intercept_limit"]),
        time_initial_intercept=float(time_contract["time_initial_intercept"]),
        time_wd_safety_limit=float(time_contract["time_wd_safety_limit"]),
        time_initial_location=time_contract.get("time_initial_location"),
        time_initial_scale=time_contract.get("time_initial_scale"),
        time_sigma_floor=float(time_contract.get("time_sigma_floor", 1e-3)),
    )
    if not isinstance(model, CountAwareTitanTPP):
        raise TypeError("Expected CountAwareTitanTPP checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    record = {
        "backbone": backbone,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_file_sha256": sha256_file(checkpoint),
        "model_state_sha256": state_sha,
        "source_revision": payload.get("source_revision"),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checks": checks,
    }
    return model, summary, record


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_validation_forward(
    model: CountAwareTitanTPP,
    loader: Any,
    *,
    device: str,
    maximum_batches: int,
) -> dict[str, Any]:
    if maximum_batches < 1:
        raise ValueError("runtime_batches must be positive")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        if quantities is None:
            raise ValueError("Raw quantities are required")
        synchronize(device)
        started = time.perf_counter()
        target_outputs(
            model,
            dts.to(device),
            mask.to(device),
            quantities.to(device),
            lambda_log_qty=1.0,
        )
        synchronize(device)
        samples.append(time.perf_counter() - started)
    if not samples:
        raise ValueError("Runtime benchmark evaluated no batches")
    steady = samples[1:] if len(samples) > 1 else samples
    return {
        "batch_count": len(samples),
        "cold_forward_seconds": samples[0],
        "steady_forward_seconds": samples[1:],
        "steady_forward_median_seconds": statistics.median(steady),
        "estimated_compile_overhead_seconds": max(
            0.0, samples[0] - statistics.median(steady)
        ),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.startswith("cuda")
            else None
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / (1024**2)
            if device.startswith("cuda")
            else None
        ),
    }


@contextmanager
def neutralize_long_term_read(
    model: CountAwareTitanTPP,
) -> Iterator[None]:
    encoder = model.titans_mac_encoder
    if encoder is None:
        raise TypeError("Long-term read neutralization requires Titans-MAC")
    memory = encoder.neural_memory
    had_instance_attribute = "read" in memory.__dict__
    previous = memory.__dict__.get("read")

    def zero_read(self: Any, state: Any, queries: torch.Tensor) -> torch.Tensor:
        del self, state
        return torch.zeros_like(queries)

    object.__setattr__(memory, "read", types.MethodType(zero_read, memory))
    try:
        yield
    finally:
        if had_instance_attribute:
            object.__setattr__(memory, "read", previous)
        else:
            object.__delattr__(memory, "read")


def causal_surprise_features(
    diagnostics: dict[str, torch.Tensor],
    history_positions: torch.Tensor,
    *,
    segment_size: int,
) -> dict[str, torch.Tensor]:
    """Summarize only writes visible at the target prediction's segment start."""
    losses = diagnostics["associative_loss"]
    writes = diagnostics["write_applied"] > 0
    positions = torch.arange(losses.size(1), device=losses.device).unsqueeze(0)
    segment_starts = torch.div(
        history_positions,
        segment_size,
        rounding_mode="floor",
    ) * segment_size
    visible = positions < segment_starts.unsqueeze(1)
    eligible = visible & writes
    counts = eligible.sum(dim=1)
    visible_losses = torch.where(eligible, losses, torch.zeros_like(losses))
    mean = visible_losses.sum(dim=1) / counts.clamp_min(1).to(losses.dtype)
    maximum = torch.where(
        eligible,
        losses,
        torch.full_like(losses, -torch.inf),
    ).max(dim=1).values
    maximum = torch.where(counts > 0, maximum, torch.zeros_like(maximum))
    latest_indices = (segment_starts - 1).clamp_min(0)
    latest = losses.gather(1, latest_indices.unsqueeze(1)).squeeze(1)
    latest = torch.where(counts > 0, latest, torch.zeros_like(latest))
    return {
        "causal_surprise_count": counts,
        "causal_surprise_latest": latest,
        "causal_surprise_mean": mean,
        "causal_surprise_max": maximum,
        "prediction_segment_start": segment_starts,
    }


@torch.no_grad()
def b1_counterfactual_outputs(
    model: CountAwareTitanTPP,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.titans_mac_encoder is None:
        raise TypeError("B1 diagnostics require a Titans-MAC model")
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    full_write_mask = mask.clone()
    full_write_mask[batch_ids, target_positions] = False
    no_write_mask = torch.zeros_like(mask)

    full, _, diagnostics = model.encode_with_memory_state(
        dts,
        history_quantities,
        mask,
        memory_write_mask=full_write_mask,
    )
    no_update, _, _ = model.encode_with_memory_state(
        dts,
        history_quantities,
        mask,
        memory_write_mask=no_write_mask,
    )
    with neutralize_long_term_read(model):
        no_ltm, _, _ = model.encode_with_memory_state(
            dts,
            history_quantities,
            mask,
            memory_write_mask=no_write_mask,
        )

    true_quantity = quantities[batch_ids, target_positions].float()
    hidden = {
        "full": full[batch_ids, history_positions],
        "no_update": no_update[batch_ids, history_positions],
        "no_ltm": no_ltm[batch_ids, history_positions],
    }
    quantity = {
        name: model.quantity_outputs(value, true_quantity)
        for name, value in hidden.items()
    }
    output: dict[str, torch.Tensor] = {
        "true_quantity": true_quantity,
        "history_length": lengths - 1,
    }
    for name in ("full", "no_update", "no_ltm"):
        prediction = quantity[name]["point_prediction"]
        error = prediction - true_quantity
        output[f"pred_{name}"] = prediction
        output[f"abs_error_{name}"] = error.abs()
        output[f"squared_error_{name}"] = error.square()
        output[f"log_location_{name}"] = quantity[name]["location"]
    output["abs_delta_full_minus_no_update"] = (
        output["abs_error_full"] - output["abs_error_no_update"]
    )
    output["abs_delta_full_minus_no_ltm"] = (
        output["abs_error_full"] - output["abs_error_no_ltm"]
    )
    output["squared_delta_full_minus_no_update"] = (
        output["squared_error_full"] - output["squared_error_no_update"]
    )
    output["squared_delta_full_minus_no_ltm"] = (
        output["squared_error_full"] - output["squared_error_no_ltm"]
    )
    output["online_update_residual_norm"] = torch.linalg.vector_norm(
        hidden["full"] - hidden["no_update"], dim=-1
    )
    output["long_term_residual_norm"] = torch.linalg.vector_norm(
        hidden["full"] - hidden["no_ltm"], dim=-1
    )
    output.update(
        causal_surprise_features(
            diagnostics,
            history_positions,
            segment_size=model.titans_mac_encoder.segment_size,
        )
    )
    for name, value in output.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite B1 diagnostic tensor: {name}")
    return output


def verify_full_equivalence(
    model: CountAwareTitanTPP,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    diagnostic: dict[str, torch.Tensor],
) -> float:
    with torch.no_grad():
        official = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
    difference = float(
        torch.max(torch.abs(official["pred_qty"] - diagnostic["pred_full"])).item()
    )
    if difference > 1e-5:
        raise AssertionError(
            f"Diagnostic full path drifted from official evaluation: {difference}"
        )
    return difference


def build_event_frame(
    *,
    dataset: str,
    seed: int,
    event_offset: int,
    diagnostic: dict[str, torch.Tensor],
    quantity_contract: dict[str, Any],
    history_contract: dict[str, Any],
    region_contract: dict[str, list[str]],
) -> pl.DataFrame:
    true_quantity = diagnostic["true_quantity"].cpu().numpy().astype(np.float64)
    history_length = diagnostic["history_length"].cpu().numpy().astype(np.int64)
    quantity_ids = np.searchsorted(
        np.asarray(quantity_contract["boundaries"], dtype=np.float64),
        true_quantity,
        side="left",
    )
    quantity_specs = sorted(
        quantity_contract["strata"], key=lambda row: int(row["stratum_order"])
    )
    quantity_labels = np.asarray(
        [quantity_specs[index]["stratum"] for index in quantity_ids]
    )
    history_ids = np.searchsorted(
        np.asarray(history_contract["boundaries"], dtype=np.int64),
        history_length,
        side="left",
    )
    history_specs = sorted(
        history_contract["strata"], key=lambda row: int(row["stratum_order"])
    )
    history_labels = np.asarray(
        [history_specs[index]["stratum"] for index in history_ids]
    )
    count = true_quantity.size
    columns: dict[str, Any] = {
        "dataset": [dataset] * count,
        "seed": np.full(count, seed, dtype=np.int64),
        "event_index": np.arange(event_offset, event_offset + count),
        "true_quantity": true_quantity,
        "history_length": history_length,
        "quantity_stratum": quantity_labels,
        "quantity_region": quantity_region_labels(
            quantity_labels, region_contract
        ),
        "history_stratum": history_labels,
    }
    tensor_columns = (
        "pred_full",
        "pred_no_update",
        "pred_no_ltm",
        "abs_error_full",
        "abs_error_no_update",
        "abs_error_no_ltm",
        "squared_error_full",
        "squared_error_no_update",
        "squared_error_no_ltm",
        "abs_delta_full_minus_no_update",
        "abs_delta_full_minus_no_ltm",
        "squared_delta_full_minus_no_update",
        "squared_delta_full_minus_no_ltm",
        "online_update_residual_norm",
        "long_term_residual_norm",
        "causal_surprise_count",
        "causal_surprise_latest",
        "causal_surprise_mean",
        "causal_surprise_max",
        "prediction_segment_start",
    )
    for name in tensor_columns:
        value = diagnostic[name].cpu().numpy()
        columns[name] = (
            value.astype(np.float64)
            if np.issubdtype(value.dtype, np.floating)
            else value.astype(np.int64)
        )
    columns["full_helped_vs_no_update"] = (
        columns["abs_delta_full_minus_no_update"] < -1e-12
    )
    columns["full_harmed_vs_no_update"] = (
        columns["abs_delta_full_minus_no_update"] > 1e-12
    )
    columns["full_helped_vs_no_ltm"] = (
        columns["abs_delta_full_minus_no_ltm"] < -1e-12
    )
    columns["full_harmed_vs_no_ltm"] = (
        columns["abs_delta_full_minus_no_ltm"] > 1e-12
    )
    return pl.DataFrame(columns)


def add_surprise_strata(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    positive = frame.filter(pl.col("causal_surprise_mean") > 0)[
        "causal_surprise_mean"
    ]
    boundaries = {
        "p50": float(positive.quantile(0.5)) if len(positive) else 0.0,
        "p90": float(positive.quantile(0.9)) if len(positive) else 0.0,
        "p99": float(positive.quantile(0.99)) if len(positive) else 0.0,
    }
    values = frame["causal_surprise_mean"].to_numpy()
    labels = np.full(values.size, "no_prior_visible_write", dtype=object)
    labels[(values > 0) & (values <= boundaries["p50"])] = "surprise_le_p50"
    labels[(values > boundaries["p50"]) & (values <= boundaries["p90"])] = (
        "surprise_p50_p90"
    )
    labels[(values > boundaries["p90"]) & (values < boundaries["p99"])] = (
        "surprise_p90_p99"
    )
    labels[(values > 0) & (values >= boundaries["p99"])] = "surprise_ge_p99"
    return frame.with_columns(
        pl.Series("causal_surprise_stratum", labels.tolist())
    ), boundaries


def aggregate_events(frame: pl.DataFrame) -> pl.DataFrame:
    aggregations = [
        pl.len().alias("event_count"),
        *(
            expression
            for name in ("full", "no_update", "no_ltm")
            for expression in (
                pl.col(f"abs_error_{name}").mean().alias(f"mae_{name}"),
                pl.col(f"squared_error_{name}").mean().alias(f"mse_{name}"),
            )
        ),
        pl.col("abs_delta_full_minus_no_update")
        .mean()
        .alias("mae_delta_full_minus_no_update"),
        pl.col("abs_delta_full_minus_no_ltm")
        .mean()
        .alias("mae_delta_full_minus_no_ltm"),
        pl.col("squared_delta_full_minus_no_update")
        .mean()
        .alias("mse_delta_full_minus_no_update"),
        pl.col("squared_delta_full_minus_no_ltm")
        .mean()
        .alias("mse_delta_full_minus_no_ltm"),
        pl.col("full_helped_vs_no_update")
        .cast(pl.Float64)
        .mean()
        .alias("help_share_vs_no_update"),
        pl.col("full_harmed_vs_no_update")
        .cast(pl.Float64)
        .mean()
        .alias("harm_share_vs_no_update"),
        pl.col("full_helped_vs_no_ltm")
        .cast(pl.Float64)
        .mean()
        .alias("help_share_vs_no_ltm"),
        pl.col("full_harmed_vs_no_ltm")
        .cast(pl.Float64)
        .mean()
        .alias("harm_share_vs_no_ltm"),
        pl.col("online_update_residual_norm")
        .mean()
        .alias("online_update_residual_norm_mean"),
        pl.col("long_term_residual_norm")
        .mean()
        .alias("long_term_residual_norm_mean"),
        pl.col("causal_surprise_mean").mean().alias("causal_surprise_mean"),
        pl.col("causal_surprise_max").mean().alias("causal_surprise_max_mean"),
    ]
    frames = []
    for scope_type, source in SCOPE_TYPES:
        scope = pl.lit("overall") if source is None else pl.col(source)
        frames.append(
            frame.with_columns(
                pl.lit(scope_type).alias("scope_type"),
                scope.alias("scope"),
            )
            .group_by(["dataset", "seed", "scope_type", "scope"])
            .agg(aggregations)
        )
    result = pl.concat(frames, how="vertical")
    return result.with_columns(
        pl.col("mse_full").sqrt().alias("rmse_full"),
        pl.col("mse_no_update").sqrt().alias("rmse_no_update"),
        pl.col("mse_no_ltm").sqrt().alias("rmse_no_ltm"),
    ).sort(["dataset", "seed", "scope_type", "scope"])


def historical_cost_row(
    dataset: str,
    b0_summary: dict[str, Any],
    b1_summary: dict[str, Any],
) -> dict[str, Any]:
    b0_epoch = float(b0_summary["elapsed_seconds"]) / int(
        b0_summary["completed_epochs"]
    )
    b1_epoch = float(b1_summary["elapsed_seconds"]) / int(
        b1_summary["completed_epochs"]
    )
    return {
        "dataset": dataset,
        "b0_completed_epochs": int(b0_summary["completed_epochs"]),
        "b1_completed_epochs": int(b1_summary["completed_epochs"]),
        "b0_elapsed_seconds": float(b0_summary["elapsed_seconds"]),
        "b1_elapsed_seconds": float(b1_summary["elapsed_seconds"]),
        "b0_seconds_per_completed_epoch": b0_epoch,
        "b1_seconds_per_completed_epoch": b1_epoch,
        "b1_b0_epoch_cost_ratio": b1_epoch / b0_epoch,
    }


def run_dataset(
    *,
    spec: DatasetSpec,
    contract: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    device: str,
    seed: int,
    batch_size: int | None,
    max_batches: int | None,
    runtime_batches: int,
) -> tuple[pl.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = output_dir / "datasets" / spec.dataset
    dataset_dir.mkdir(parents=True)
    _, loader = dataset_frame_and_loader(spec, contract, batch_size)
    b0_model, b0_summary, b0_record = restore_model(
        spec.artifact_dir, contract, B0_BACKBONE, seed, device
    )
    b0_runtime = benchmark_validation_forward(
        b0_model, loader, device=device, maximum_batches=runtime_batches
    )
    del b0_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()

    b1_model, b1_summary, b1_record = restore_model(
        spec.artifact_dir, contract, B1_BACKBONE, seed, device
    )
    b1_runtime = benchmark_validation_forward(
        b1_model, loader, device=device, maximum_batches=runtime_batches
    )
    frames = []
    event_offset = 0
    equivalence = 0.0
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if quantities is None:
            raise ValueError("Raw quantities are required")
        diagnostic = b1_counterfactual_outputs(
            b1_model,
            dts.to(device),
            mask.to(device),
            quantities.to(device),
        )
        if batch_index == 0:
            equivalence = verify_full_equivalence(
                b1_model,
                dts.to(device),
                mask.to(device),
                quantities.to(device),
                diagnostic,
            )
        frame = build_event_frame(
            dataset=spec.dataset,
            seed=seed,
            event_offset=event_offset,
            diagnostic=diagnostic,
            quantity_contract=contract["quantity_contract"],
            history_contract=contract["history_length_contract"],
            region_contract=manifest["quantity_regions"],
        )
        frames.append(frame)
        event_offset += frame.height
    if not frames:
        raise ValueError(f"No events evaluated for {spec.dataset}")
    events, surprise_boundaries = add_surprise_strata(
        pl.concat(frames, how="vertical")
    )
    events.write_parquet(dataset_dir / "event_diagnostics.parquet")
    scope_metrics = aggregate_events(events)
    scope_metrics.write_csv(dataset_dir / "scope_metrics.csv")
    overall = scope_metrics.filter(
        (pl.col("scope_type") == "overall") & (pl.col("scope") == "overall")
    ).row(0, named=True)
    full_validation = max_batches is None
    mae_diff = abs(float(overall["mae_full"]) - float(b1_summary["best_val_qty_mae"]))
    rmse_diff = abs(
        float(overall["rmse_full"]) - float(b1_summary["best_val_qty_rmse"])
    )
    artifact_match = not full_validation or (mae_diff <= 1e-5 and rmse_diff <= 1e-5)
    if not artifact_match:
        raise AssertionError(f"B1 metrics did not reproduce {spec.dataset}")

    runtime = {
        "dataset": spec.dataset,
        "b0": b0_runtime,
        "b1": b1_runtime,
        "steady_forward_ratio_b1_vs_b0": (
            b1_runtime["steady_forward_median_seconds"]
            / b0_runtime["steady_forward_median_seconds"]
        ),
        "peak_allocated_ratio_b1_vs_b0": (
            b1_runtime["peak_allocated_mib"] / b0_runtime["peak_allocated_mib"]
            if device.startswith("cuda") and b0_runtime["peak_allocated_mib"]
            else None
        ),
    }
    audit = {
        "dataset": spec.dataset,
        "seed": seed,
        "event_count": events.height,
        "full_validation": full_validation,
        "official_prediction_max_abs_diff": equivalence,
        "artifact_qty_mae": float(b1_summary["best_val_qty_mae"]),
        "diagnostic_qty_mae": float(overall["mae_full"]),
        "qty_mae_abs_diff": mae_diff,
        "artifact_qty_rmse": float(b1_summary["best_val_qty_rmse"]),
        "diagnostic_qty_rmse": float(overall["rmse_full"]),
        "qty_rmse_abs_diff": rmse_diff,
        "artifact_match": artifact_match,
        "surprise_boundaries": surprise_boundaries,
        "b0_checkpoint": b0_record,
        "b1_checkpoint": b1_record,
    }
    write_json(dataset_dir / "checkpoint_audit.json", audit)
    write_json(dataset_dir / "runtime_cost.json", runtime)
    del b1_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scope_metrics, [audit], {
        **historical_cost_row(spec.dataset, b0_summary, b1_summary),
        **runtime,
    }


def build_decision(
    metrics: pl.DataFrame,
    checkpoint_audits: list[dict[str, Any]],
    costs: list[dict[str, Any]],
) -> dict[str, Any]:
    overall = metrics.filter(pl.col("scope_type") == "overall").sort("dataset")
    rows = []
    for row in overall.iter_rows(named=True):
        rows.append(
            {
                "dataset": row["dataset"],
                "online_write_mae_delta_full_minus_disabled": row[
                    "mae_delta_full_minus_no_update"
                ],
                "long_term_mechanism_mae_delta_full_minus_neutral": row[
                    "mae_delta_full_minus_no_ltm"
                ],
                "online_write_help_share": row["help_share_vs_no_update"],
                "online_write_harm_share": row["harm_share_vs_no_update"],
            }
        )
    return {
        "status": "complete",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "counterfactual_scope": (
            "same_checkpoint_inference_attribution_not_retrained_ablation"
        ),
        "no_ltm_control_limitation": (
            "retrieved token slots remain present but reads and writes are zeroed"
        ),
        "checkpoint_contracts_passed": all(
            audit["artifact_match"] for audit in checkpoint_audits
        ),
        "dataset_attribution": rows,
        "historical_epoch_cost": [
            {
                "dataset": row["dataset"],
                "b1_b0_epoch_cost_ratio": row["b1_b0_epoch_cost_ratio"],
            }
            for row in costs
        ],
    }


def render_analysis(decision: dict[str, Any]) -> str:
    lines = [
        "# Count-aware TitanTPP-MAC B1 Audit",
        "",
        "This is a validation-only, same-checkpoint attribution audit. It is not a",
        "retrained ablation and does not use held-out test data.",
        "",
        "## Memory attribution",
        "",
        "| Dataset | Full - no update MAE | Help share | Harm share | Full - neutral LTM MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in decision["dataset_attribution"]:
        lines.append(
            "| {dataset} | {update:+.6f} | {help:.2%} | {harm:.2%} | {ltm:+.6f} |".format(
                dataset=row["dataset"],
                update=row["online_write_mae_delta_full_minus_disabled"],
                help=row["online_write_help_share"],
                harm=row["online_write_harm_share"],
                ltm=row["long_term_mechanism_mae_delta_full_minus_neutral"],
            )
        )
    lines.extend(
        [
            "",
            "Negative deltas mean that the enabled memory path reduced MAE. Surprise",
            "strata use only writes completed before the prediction segment begins.",
            "The neutral-LTM control leaves zero-valued retrieved token slots in MAC,",
            "so it isolates the learned long-term content rather than deleting topology.",
            "",
            "## Historical training cost",
            "",
            "| Dataset | B1/B0 seconds per completed epoch |",
            "|---|---:|",
        ]
    )
    for row in decision["historical_epoch_cost"]:
        lines.append(
            f"| {row['dataset']} | {row['b1_b0_epoch_cost_ratio']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "These ratios describe the frozen seed-42 runs before the",
            "semantics-preserving optimization pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_source_manifest(output_dir: Path) -> None:
    rows = []
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        rows.append({"path": relative, "sha256": sha256_file(path)})
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = None
    write_json(
        output_dir / "source_manifest.json",
        {"git_revision": revision, "files": rows},
    )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest, specs = load_manifest(
        args.manifest,
        set(args.datasets) if args.datasets else None,
    )
    seeds = args.seeds if args.seeds else [int(value) for value in manifest["seeds"]]
    if len(seeds) != 1:
        raise ValueError("Current frozen B1 audit expects exactly one available seed")
    args.output_dir.mkdir(parents=True)
    metrics = []
    checkpoint_audits = []
    costs = []
    for spec in specs:
        contract = validate_dataset_contract(spec)
        dataset_metrics, audits, cost = run_dataset(
            spec=spec,
            contract=contract,
            manifest=manifest,
            output_dir=args.output_dir,
            device=args.device,
            seed=seeds[0],
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            runtime_batches=args.runtime_batches,
        )
        metrics.append(dataset_metrics)
        checkpoint_audits.extend(audits)
        costs.append(cost)
    combined = pl.concat(metrics, how="vertical")
    combined.write_csv(args.output_dir / "scope_metrics.csv")
    pl.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"b0", "b1"}}
            for row in costs
        ]
    ).write_csv(args.output_dir / "historical_cost.csv")
    write_json(args.output_dir / "runtime_cost.json", costs)
    write_json(args.output_dir / "checkpoint_audit.json", checkpoint_audits)
    decision = build_decision(combined, checkpoint_audits, costs)
    write_json(args.output_dir / "decision.json", decision)
    (args.output_dir / "analysis.md").write_text(
        render_analysis(decision), encoding="utf-8"
    )
    write_source_manifest(args.output_dir)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
