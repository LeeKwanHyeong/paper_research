#!/usr/bin/env python3
"""Run the frozen-B, train-only shared-block objective-transfer audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "paper/contracts/hard_lmm_shared_block_transfer_probe_v1.json"
BASE_CONTRACT = ROOT / "paper/contracts/hard_lmm_credit_assignment_probe_v1.json"
BASE_RUNNER = ROOT / "paper/scripts/run_hard_lmm_credit_assignment_probe.py"
HELPER = ROOT / "paper/scripts/hard_lmm_shared_block_transfer.py"
DEFAULT_RESULT = ROOT / "paper/results/hard_lmm_shared_block_transfer_probe_20260909"
EXPECTED_CONTRACT_SHA256 = "a87a0f8c7f0f06935392f5ff0c2b80655c42f8cbb5c9eda21c6040fa7900d5fe"
BOUNDARIES = ("H1", "H2", "fused")
METRICS = ("log", "raw", "body", "time")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_hard_lmm_shared_transfer")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_hard_lmm_shared_transfer")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("hard_lmm_shared_transfer_base", BASE_RUNNER)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def save_json(path: Path, value: Any) -> None:
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


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(sha256_file(CONTRACT) == EXPECTED_CONTRACT_SHA256, "Contract hash drift")
    require(
        contract.get("contract_id") == "hard_lmm_shared_block_transfer_probe_v1",
        "Contract ID drift",
    )
    authorization = contract["authorization"]
    require(authorization["local_train_only_diagnostic"] is True, "Local audit disabled")
    require(
        authorization["source_fold_time_head_calibration"] is True,
        "Cross-fit time calibration disabled",
    )
    for name in (
        "backbone_or_quantity_parameter_updates",
        "candidate_implementation",
        "gpu_execution",
        "held_out_test",
        "validation_rows",
    ):
        require(authorization[name] is False, f"Forbidden authorization enabled: {name}")
    scope = contract["diagnostic_scope"]
    require(scope["allowed_split"] == "train_only", "Audit is not train-only")
    require(scope["validation_rows_materialized"] is False, "Validation access enabled")
    require(scope["model_or_quantity_parameter_updates"] is False, "B updates enabled")
    require(scope["maximum_targets_per_dataset"] == 4096, "Cohort count drift")
    require(scope["minimum_targets_per_fold"] == 2048, "Fold count drift")
    require(scope["batch_size"] == 128, "Batch size drift")
    require(scope["device"] == "cpu", "Device drift")
    require(len(scope["fold_directions"]) == 2, "Fold-direction drift")
    reference = contract["frozen_reference"]
    require(
        reference["source_revision"] == "f75243473adc25d622319dbca9bda7e076d8240f",
        "Frozen B revision drift",
    )
    time_fit = contract["normalized_time_measurement"]["fit"]
    require(time_fit["epochs"] == 100, "Time-head epoch budget drift")
    require(time_fit["selection"].startswith("the state after exactly epoch 100"), "Selection drift")
    require(time_fit["cached_state_batch_size"] == 8192, "Time batch size drift")
    require(time_fit["optimizer"] == "AdamW", "Optimizer drift")
    require(time_fit["scheduler"] is None, "Unexpected scheduler")
    settings = contract["normalized_time_measurement"]["dataset_settings"]
    require(
        settings
        == {
            "insta_market_basket": {
                "censor_threshold": 30.0,
                "observation_mode": "positive_integer_round_clamp_lognormal",
            },
            "intermittent_v2": {
                "censor_threshold": None,
                "observation_mode": "continuous_lognormal_density",
            },
            "yellow_trip_hourly": {
                "censor_threshold": None,
                "observation_mode": "positive_integer_round_clamp_lognormal",
            },
        },
        "Observation contract drift",
    )
    probe = contract["probe"]
    require(tuple(probe["boundaries"]) == BOUNDARIES, "Boundary drift")
    require(probe["virtual_adapter_parameter_count"] == 4096, "Adapter size drift")
    require(probe["candidate_eligible_boundaries"] == ["H2", "H1"], "Candidate order drift")
    gate = contract["prospective_gate"]
    require(gate["candidate_order_if_multiple_pass"] == ["H2", "H1"], "Gate order drift")
    require(gate["relative_direction_tolerance"] == 1e-12, "Gate tolerance drift")
    require("63 cyclic shifts" in gate["controls"]["permuted_separated"], "Null drift")


def verify_reused_evidence(contract: Mapping[str, Any]) -> dict[str, str]:
    reuse = contract["evidence_reuse"]
    checks = (
        ("dataset_and_checkpoint_contract_path", "dataset_and_checkpoint_contract_sha256"),
        ("prior_value_update_analysis_path", "prior_value_update_analysis_sha256"),
        ("prior_value_update_decision_path", "prior_value_update_decision_sha256"),
        ("qkv_path_result_path", "qkv_path_result_sha256"),
    )
    verified: dict[str, str] = {}
    for path_key, hash_key in checks:
        path = ROOT / reuse[path_key]
        require(path.is_file(), f"Missing reused evidence: {path}")
        observed = sha256_file(path)
        require(observed == reuse[hash_key], f"Reused evidence drift: {path}")
        verified[reuse[path_key]] = observed
    require(BASE_CONTRACT == ROOT / reuse["dataset_and_checkpoint_contract_path"], "Base path drift")
    base.validate_contract(read_json(BASE_CONTRACT))
    return verified


def source_hashes() -> dict[str, str]:
    paths = (
        "paper/contracts/hard_lmm_shared_block_transfer_probe_v1.json",
        "paper/scripts/hard_lmm_shared_block_transfer.py",
        "paper/scripts/run_hard_lmm_shared_block_transfer_probe.py",
        "paper/contracts/hard_lmm_credit_assignment_probe_v1.json",
        "paper/scripts/run_hard_lmm_credit_assignment_probe.py",
    )
    result = {}
    for relative in paths:
        path = ROOT / relative
        require(path.is_file(), f"Missing diagnostic source: {path}")
        result[relative] = sha256_file(path)
    return result


def load_extended_cache(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    identities, cache = base.load_cohort(row)
    required = (
        "dataset_index",
        "series_id",
        "fold",
        "true_quantity",
        "true_duration",
        "B__prediction",
        "B__H1",
        "B__H2",
        "B__fused_state",
        "B__memory_residual",
        "B__top4_indices",
    )
    with np.load(ROOT / row["cohort_cache_path"], allow_pickle=False) as loaded:
        require(all(name in loaded for name in required), "Extended cache schema drift")
        cache = {name: loaded[name].copy() for name in required}
    require(cache["B__H1"].shape == (4096, 64), "H1 cache shape drift")
    require(cache["B__H2"].shape == (4096, 64), "H2 cache shape drift")
    require(cache["B__fused_state"].shape == (4096, 64), "Fused cache shape drift")
    return identities, cache


def prepare_live_batch(
    model: torch.nn.Module,
    frozen: Any,
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
    features = model.continuous_features(dts, history_quantities, mask)
    encoder = model.encoder
    require(encoder is not None and len(encoder.layers) == 2, "Expected two encoder blocks")
    hidden = encoder.input_proj(features)
    if encoder.use_pos_emb:
        hidden = hidden + encoder._get_pos(hidden.size(1), hidden.device, hidden.dtype)
    valid = mask.unsqueeze(-1).to(hidden.dtype)
    hidden = hidden * valid
    H1 = encoder.layers[0](hidden, mask=mask)
    H2 = encoder.layers[1](H1, mask=mask)
    require(model.lmm is not None and model.lmm.topk == 4, "Expected static Hard-LMM top-4")
    residual, trace = model.lmm.retrieve(H2)
    fused = (H2 + residual) * valid
    memory = model.lmm.mem
    require(memory is not None and tuple(memory.shape) == (1, 64, 64), "Memory shape drift")
    encoded_normalized = torch.nn.functional.normalize(H2, p=2, dim=-1)
    memory_normalized = torch.nn.functional.normalize(memory, p=2, dim=-1)
    similarity = torch.matmul(encoded_normalized, memory_normalized.transpose(-2, -1))
    top5_similarity = torch.topk(similarity, k=5, dim=-1).values
    top4_margin = top5_similarity[..., 3] - top5_similarity[..., 4]
    selected = lambda value: value[rows, history_positions]
    final = selected(fused)
    location, prediction = model.predict_quantity(final)
    target_quantity = quantities[rows, target_positions].float()
    target_duration = dts[rows, target_positions].float()
    return {
        "H1": H1,
        "H2": H2,
        "fused": fused,
        "mask": mask,
        "final": final,
        "location": location,
        "prediction": prediction,
        "target_quantity": target_quantity,
        "target_duration": target_duration,
        "top4_indices": selected(trace["prototype_indices"]),
        "top4_margin": top4_margin,
        "selected_H1": selected(H1),
        "selected_H2": selected(H2),
        "selected_fused": final,
    }


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().double().cpu() - right.detach().double().cpu()).abs().max())


def update_cache_parity(
    audit: dict[str, Any],
    batch: Mapping[str, torch.Tensor],
    cache: Mapping[str, np.ndarray],
    positions: np.ndarray,
    cursor: int,
) -> None:
    batch_count = int(batch["target_quantity"].numel())
    chosen = positions[cursor : cursor + batch_count]
    require(chosen.size == batch_count, "Cache cursor overflow")
    fields = {
        "selected_H1": "B__H1",
        "selected_H2": "B__H2",
        "selected_fused": "B__fused_state",
        "prediction": "B__prediction",
        "target_quantity": "true_quantity",
        "target_duration": "true_duration",
    }
    for actual_name, cache_name in fields.items():
        actual = batch[actual_name].detach().cpu()
        expected = torch.from_numpy(cache[cache_name][chosen])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        audit[actual_name + "_max_abs"] = max(
            audit[actual_name + "_max_abs"], max_abs(actual, expected)
        )
    expected_indices = torch.from_numpy(cache["B__top4_indices"][chosen])
    require(torch.equal(batch["top4_indices"].detach().cpu(), expected_indices), "Top-4 cache drift")
    audit["top4_indices_bitwise_equal"] = True


def official_quantity_parity(
    target_outputs: Any,
    model: torch.nn.Module,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    official = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    torch.testing.assert_close(batch["prediction"], official["pred_qty"], rtol=1e-5, atol=1e-6)
    target_log = torch.log1p(batch["target_quantity"].clamp_min(0.0))
    manual_log = torch.square(batch["location"] - target_log)
    torch.testing.assert_close(manual_log, official["log_qty_loss"], rtol=1e-5, atol=1e-6)
    return {
        "prediction_max_abs": max_abs(batch["prediction"], official["pred_qty"]),
        "log_mse_max_abs": max_abs(manual_log, official["log_qty_loss"]),
    }


def empty_gradient_sums() -> dict[str, dict[str, torch.Tensor]]:
    return {
        boundary: {
            metric: torch.zeros(64, 64, dtype=torch.float64) for metric in METRICS
        }
        for boundary in BOUNDARIES
    }


def probe_fold(
    *,
    model: torch.nn.Module,
    frozen: Any,
    target_outputs: Any,
    loader: DataLoader,
    time_head: torch.nn.Module,
    helper: Any,
    observation_mode: str,
    censor_threshold: float | None,
    body_threshold: float,
    cache: Mapping[str, np.ndarray],
    cache_positions: np.ndarray,
) -> dict[str, Any]:
    require(not model.training, "Frozen B must remain in eval mode")
    require(not time_head.training, "Cross-fit time head must remain in eval mode")
    require(
        all(not parameter.requires_grad for parameter in time_head.parameters()),
        "Cross-fit time head was not frozen before live gradient measurement",
    )
    sums = empty_gradient_sums()
    losses = {metric: 0.0 for metric in METRICS}
    counts = {metric: 0 for metric in METRICS}
    hidden_credit_l2_sum = {boundary: 0.0 for boundary in BOUNDARIES}
    top4_margins: list[torch.Tensor] = []
    cache_audit: dict[str, Any] = {
        "top4_indices_bitwise_equal": True,
        **{
            name + "_max_abs": 0.0
            for name in (
                "selected_H1",
                "selected_H2",
                "selected_fused",
                "prediction",
                "target_quantity",
                "target_duration",
            )
        },
    }
    official_audit: dict[str, float] | None = None
    cursor = 0
    started = time.monotonic()
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        require(quantities is not None, "Quantity observations are missing")
        batch = prepare_live_batch(model, frozen, dts, mask, quantities)
        if official_audit is None:
            official_audit = official_quantity_parity(
                target_outputs, model, dts, mask, quantities, batch
            )
        update_cache_parity(cache_audit, batch, cache, cache_positions, cursor)
        cursor += int(batch["target_quantity"].numel())
        batch_margins = batch["top4_margin"][batch["mask"]].detach().double().cpu()
        require(bool(torch.isfinite(batch_margins).all()), "Non-finite top-4/5 margin")
        require(bool((batch_margins > 0.0).all()), "Exact Hard-LMM top-4/5 tie")
        top4_margins.append(batch_margins)
        target_quantity = batch["target_quantity"]
        target_log = torch.log1p(target_quantity.clamp_min(0.0))
        metric_losses: dict[str, torch.Tensor] = {
            "log": torch.square(batch["location"] - target_log),
            "raw": torch.square(batch["prediction"] - target_quantity),
            "time": time_head.negative_log_likelihood(
                batch["final"],
                batch["target_duration"],
                observation_mode=observation_mode,
                censor_threshold=censor_threshold,
            ),
        }
        body_mask = target_quantity <= float(body_threshold)
        require(bool(body_mask.any()), "A diagnostic batch has no body targets")
        metric_losses["body"] = torch.abs(batch["prediction"] - target_quantity)[body_mask]
        states = (batch["H1"], batch["H2"], batch["fused"])
        for metric_index, metric in enumerate(METRICS):
            values = metric_losses[metric]
            require(bool(torch.isfinite(values).all()), f"Non-finite {metric} loss")
            loss = values.mean()
            credits = torch.autograd.grad(
                loss,
                states,
                retain_graph=metric_index < len(METRICS) - 1,
                allow_unused=False,
            )
            weight = int(values.numel())
            for boundary, state, credit in zip(BOUNDARIES, states, credits):
                gradient = helper.linear_residual_adapter_gradient(
                    state,
                    credit,
                    batch["mask"],
                    denominator=1.0,
                )
                require(gradient.shape == (64, 64), "Adapter gradient shape drift")
                require(bool(torch.isfinite(gradient).all()), "Non-finite adapter gradient")
                sums[boundary][metric].add_(gradient.detach().cpu().double(), alpha=float(weight))
                if metric == "time":
                    selected_credit = credit[batch["mask"]]
                    hidden_credit_l2_sum[boundary] += float(
                        torch.linalg.vector_norm(selected_credit.detach().double(), dim=-1).sum()
                    )
            losses[metric] += float(values.detach().double().sum())
            counts[metric] += weight
    require(cursor == len(cache_positions) == 2048, "Fold target count drift")
    for boundary in BOUNDARIES:
        for metric in METRICS:
            require(counts[metric] > 0, f"Empty metric: {metric}")
            sums[boundary][metric].div_(float(counts[metric]))
    require(official_audit is not None, "Official parity was not measured")
    all_margins = torch.cat(top4_margins)
    for boundary in BOUNDARIES:
        require(
            hidden_credit_l2_sum[boundary] > 1e-12,
            f"Normalized time hidden gradient is zero at {boundary}",
        )
    return {
        "adapter_gradients": sums,
        "counts": counts,
        "mean_losses": {metric: losses[metric] / counts[metric] for metric in METRICS},
        "time_hidden_credit_l2_sum": hidden_credit_l2_sum,
        "cache_parity": cache_audit,
        "official_parity": official_audit,
        "top4_margin": {
            "valid_token_count": int(all_margins.numel()),
            "minimum": float(all_margins.min()),
            "p01": float(torch.quantile(all_margins, 0.01)),
            "median": float(torch.quantile(all_margins, 0.5)),
            "fraction_at_or_below_1e_6": float((all_margins <= 1e-6).double().mean()),
            "exact_tie_count": int((all_margins <= 0.0).sum()),
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def compact_probe(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "counts": result["counts"],
        "mean_losses": result["mean_losses"],
        "gradient_norms": {
            boundary: {
                metric: float(torch.linalg.vector_norm(result["adapter_gradients"][boundary][metric]))
                for metric in METRICS
            }
            for boundary in BOUNDARIES
        },
        "time_hidden_credit_l2_sum": result["time_hidden_credit_l2_sum"],
        "cache_parity": result["cache_parity"],
        "official_parity": result["official_parity"],
        "top4_margin": result["top4_margin"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def build_loader(
    dataset: Any,
    identities: Mapping[str, Any],
    fold: int,
    *,
    batch_size: int,
    collate_fn: Any,
) -> DataLoader:
    rows = [item for item in identities["rows"] if int(item["fold"]) == fold]
    require(len(rows) == 2048, "Fold identity count drift")
    return DataLoader(
        Subset(dataset, [int(item["dataset_index"]) for item in rows]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def analyze_dataset(
    row: Mapping[str, Any],
    *,
    frozen: Any,
    target_outputs: Any,
    helper: Any,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = str(row["dataset"])
    for key, hash_key in (
        ("data_path", "data_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("launch_contract_path", "launch_contract_sha256"),
        ("checkpoint_path", "checkpoint_file_sha256"),
        ("summary_path", "summary_sha256"),
    ):
        base.verify_file(ROOT / row[key], row[hash_key], key)
    identities, cache = load_extended_cache(row)
    frame = base.load_train_frame(ROOT / row["data_path"])
    require(frame.height == int(row["train_rows"]), f"{name} train-row count drift")
    prepared = frozen.prepare_count_frame(frame)
    dataset = frozen.Dataset(
        prepared,
        lookback_weeks=int(row["lookback_weeks"]),
        max_seq_len=int(row["max_seq_len"]),
        mode="all",
        split_col="chronological_split",
        target_splits={"train"},
    )
    require(len(dataset) == int(row["train_targets"]), f"{name} train-target drift")
    base.verify_dataset_identity(dataset, identities["rows"])
    launch = read_json(ROOT / row["launch_contract_path"])
    model, restore = frozen.restore_b0(ROOT / row["checkpoint_path"], launch, "cpu")
    require(restore["model_state_sha256"] == row["checkpoint_state_sha256"], "B state drift")
    model.eval().requires_grad_(True)
    state_before = frozen.canonical_state_sha256(model.state_dict())
    observation = contract["normalized_time_measurement"]["dataset_settings"][name]
    fit = contract["normalized_time_measurement"]["fit"]
    fold_positions = {
        fold: np.flatnonzero(cache["fold"] == fold).astype(np.int64) for fold in (0, 1)
    }
    require(all(len(value) == 2048 for value in fold_positions.values()), "Cache fold drift")
    direction_results: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for source_fold, held_fold in ((0, 1), (1, 0)):
        direction = f"fold{source_fold}_to_fold{held_fold}"
        source_positions = fold_positions[source_fold]
        source_hidden = torch.from_numpy(cache["B__fused_state"][source_positions])
        source_duration = torch.from_numpy(cache["true_duration"][source_positions])
        time_head, fit_audit = helper.fit_crossfit_time_head(
            source_hidden,
            source_duration,
            observation_mode=str(observation["observation_mode"]),
            censor_threshold=observation["censor_threshold"],
            seed=int(fit["seed"]),
            epochs=int(fit["epochs"]),
            learning_rate=float(fit["learning_rate"]),
            weight_decay=float(fit["weight_decay"]),
            grad_clip=float(fit["gradient_clip"]),
            batch_size=int(fit["cached_state_batch_size"]),
        )
        require(fit_audit["target_count"] == 2048, "Time-head fit count drift")
        require(
            fit_audit["final_nll"] < fit_audit["initial_nll"] - 1e-8,
            "Cross-fit time head did not improve its source-fold NLL",
        )
        time_head.eval().requires_grad_(False)
        source_probe = probe_fold(
            model=model,
            frozen=frozen,
            target_outputs=target_outputs,
            loader=build_loader(
                dataset,
                identities,
                source_fold,
                batch_size=int(contract["diagnostic_scope"]["batch_size"]),
                collate_fn=frozen.collate,
            ),
            time_head=time_head,
            helper=helper,
            observation_mode=str(observation["observation_mode"]),
            censor_threshold=observation["censor_threshold"],
            body_threshold=float(row["body_threshold_train_p95"]),
            cache=cache,
            cache_positions=source_positions,
        )
        held_probe = probe_fold(
            model=model,
            frozen=frozen,
            target_outputs=target_outputs,
            loader=build_loader(
                dataset,
                identities,
                held_fold,
                batch_size=int(contract["diagnostic_scope"]["batch_size"]),
                collate_fn=frozen.collate,
            ),
            time_head=time_head,
            helper=helper,
            observation_mode=str(observation["observation_mode"]),
            censor_threshold=observation["censor_threshold"],
            body_threshold=float(row["body_threshold_train_p95"]),
            cache=cache,
            cache_positions=fold_positions[held_fold],
        )
        require(
            math.isclose(
                source_probe["mean_losses"]["time"],
                float(fit_audit["final_nll"]),
                rel_tol=0.0,
                abs_tol=2e-10,
            ),
            "Cached and live source-fold proper NLL differ",
        )
        boundary_results: dict[str, Any] = {}
        for boundary in BOUNDARIES:
            source_mapping = {
                "log_mse": source_probe["adapter_gradients"][boundary]["log"],
                "raw_squared_error": source_probe["adapter_gradients"][boundary]["raw"],
                "body_mae": source_probe["adapter_gradients"][boundary]["body"],
                "normalized_time_nll": source_probe["adapter_gradients"][boundary]["time"],
            }
            held_mapping = {
                "log_mse": held_probe["adapter_gradients"][boundary]["log"],
                "raw_squared_error": held_probe["adapter_gradients"][boundary]["raw"],
                "body_mae": held_probe["adapter_gradients"][boundary]["body"],
                "normalized_time_nll": held_probe["adapter_gradients"][boundary]["time"],
            }
            comparison = helper.compare_shared_and_separated(
                source_mapping,
                held_mapping,
                tolerance=float(
                    contract["prospective_gate"]["relative_direction_tolerance"]
                ),
            )
            boundary_results[boundary] = comparison
            helper_metric_names = {
                "log": "log_mse",
                "raw": "raw_squared_error",
                "body": "body_mae",
                "time": "normalized_time_nll",
            }
            for metric in METRICS:
                row_out = {
                    "dataset": name,
                    "direction": direction,
                    "boundary": boundary,
                    "metric": metric,
                    **comparison["metrics"][helper_metric_names[metric]],
                }
                metric_rows.append(row_out)
        direction_results[direction] = {
            "source_fold": source_fold,
            "held_fold": held_fold,
            "time_head_fit": fit_audit,
            "source": compact_probe(source_probe),
            "held": compact_probe(held_probe),
            "boundaries": boundary_results,
        }
    state_after = frozen.canonical_state_sha256(model.state_dict())
    require(state_after == state_before == row["checkpoint_state_sha256"], "Frozen B changed")
    require(all(parameter.grad is None for parameter in model.parameters()), "B .grad field populated")
    boundary_pass = {
        boundary: all(
            result["boundaries"][boundary]["checks"]["passed"]
            for result in direction_results.values()
        )
        for boundary in BOUNDARIES
    }
    return {
        "dataset": name,
        "scope": {
            "split": "train",
            "validation_rows_materialized": False,
            "held_out_rows_materialized": False,
            "backbone_or_quantity_parameter_updates": False,
            "source_fold_time_head_calibration": True,
        },
        "cohort": {
            "target_count": 4096,
            "fold_counts": identities["fold_counts"],
            "fold_unique_series": identities["fold_unique_series"],
            "sample_ids_sha256": row["cohort_sample_ids_sha256"],
            "cache_sha256": row["cohort_cache_sha256"],
        },
        "observation": observation,
        "body_threshold_train_p95": float(row["body_threshold_train_p95"]),
        "checkpoint_state_sha256_before_after": state_before,
        "directions": direction_results,
        "boundary_pass": boundary_pass,
    }, metric_rows


def write_metric_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "No metric rows")

    def flatten_value(prefix: str, value: Any, output: dict[str, Any]) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                name = f"{prefix}__{key}" if prefix else str(key)
                flatten_value(name, nested, output)
        else:
            output[prefix] = value

    flattened: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            flatten_value(str(key), value, item)
        flattened.append(item)
    identity = ["dataset", "direction", "boundary", "metric"]
    keys = identity + sorted(set().union(*(set(row) for row in flattened)) - set(identity))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in flattened)


def decision_from_results(
    results: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_order = contract["prospective_gate"]["candidate_order_if_multiple_pass"]
    common = {
        boundary: all(result["boundary_pass"][boundary] for result in results.values())
        for boundary in BOUNDARIES
    }
    selected = next((boundary for boundary in candidate_order if common[boundary]), None)
    fused_only = selected is None and common["fused"]
    if selected is not None:
        action = (
            f"freeze_implementation_contract_for_{selected}_zero_init_"
            "task_specific_linear_residual_branch"
        )
    elif fused_only:
        action = "do_not_add_new_backbone_branch_existing_post_lmm_readout_boundary_only"
    else:
        action = (
            "do_not_implement_or_gpu_train_the_zero_init_task_specific_"
            "linear_residual_branch"
        )
    return {
        "contract_id": contract["contract_id"],
        "datasets": {
            name: {"boundary_pass": result["boundary_pass"]} for name, result in results.items()
        },
        "common_boundary_pass": common,
        "selected_candidate_boundary": selected,
        "passed": selected is not None,
        "action": action,
        "interpretation": (
            "An adapter-specific first-order train-only necessary-condition result for a "
            "counterfactual normalized-K1 shared objective using cross-fitted time heads; it "
            "does not measure B's historical legacy update, a trained candidate, or validation "
            "performance."
        ),
    }


def render_readme(results: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen-B shared-block objective-transfer audit",
        "",
        "## 결론",
        "",
    ]
    if decision["passed"]:
        lines.append(
            f"세 데이터셋과 두 fold 방향에서 공통으로 `{decision['selected_candidate_boundary']}` "
            "경계가 사전 기준을 통과했다. 다음 단계는 별도 구현 계약을 고정하는 것이다."
        )
    elif decision["action"].startswith("do_not_add_new_backbone"):
        lines.append(
            "공통 신호는 이미 평가한 post-LMM readout 경계에서만 나타났다. 새로운 Backbone "
            "분기를 구현할 근거는 확보되지 않았다."
        )
    else:
        lines.append(
            "H1 또는 H2에서 공통 task-specific 분기를 구현할 필요조건이 성립하지 않았다. "
            "따라서 이 후보는 구현하거나 GPU에서 학습하지 않는다."
        )
    lines += [
        "",
        "이 판정은 고정 B의 train-only 1차 미분 방향을 비교한 필요조건 검사다. 실제 validation "
        "성능이나 인과적 성능 향상을 뜻하지 않는다.",
        "",
        "## 공통 경계 판정",
        "",
        "| 경계 | Intermittent | Taxi | Instacart | 세 데이터셋 공통 |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    dataset_order = ("intermittent_v2", "yellow_trip_hourly", "insta_market_basket")
    for boundary in BOUNDARIES:
        values = [bool(results[name]["boundary_pass"][boundary]) for name in dataset_order]
        cells = ["통과" if value else "실패" for value in values]
        common = "통과" if decision["common_boundary_pass"][boundary] else "실패"
        lines.append(f"| {boundary} | {cells[0]} | {cells[1]} | {cells[2]} | {common} |")
    lines += [
        "",
        "`fused`는 기존 post-LMM readout 실험과 겹치는 report-only 대조 경계이므로 단독으로 "
        "통과해도 새 Backbone 후보를 열지 않는다.",
        "",
        "## 측정 계약",
        "",
        "- 기존 4,096개 train target과 series-disjoint 2개 fold를 그대로 재사용했다.",
        "- 각 방향의 normalized K=1 시간 head는 source fold 2,048개에서만 100 epoch 적합했고 "
        "validation selector나 held-out test를 사용하지 않았다.",
        "- B encoder, Hard-LMM bank, quantity head와 checkpoint state는 전후 bitwise 동일하다.",
        "- 각 경계에 가상 zero-init `64×64` linear residual을 두고 전체 valid sequence의 "
        "gradient를 계산했다. 실제 forward나 모델 코드는 바꾸지 않았다.",
        "- separated upper bound는 quantity와 time 두 adapter의 합산 update norm을 shared "
        "adapter 하나와 동일하게 맞췄다.",
        "",
        "## 해석 한계",
        "",
        "- B checkpoint 자체는 과거 validation raw RMSE로 선택됐다. 이번 실행은 그 고정 "
        "checkpoint 안의 objective conflict를 재는 감사다.",
        "- shared 비교는 normalized K=1 time objective를 쓰는 counterfactual이며 B가 과거에 "
        "학습한 legacy time update를 재현한 것이 아니다.",
        "- 결과는 infinitesimal one-step 방향이다. 유한 epoch 학습, optimizer dynamics, top-4 "
        "membership 변화는 포함하지 않는다.",
        "- 시간 head는 source-fold measurement instrument이며 최종 모델 후보가 아니다.",
        "",
        "정량 directional dot, shared controls, cyclic alignment control과 fold별 head 적합 기록은 "
        "`analysis.json`과 `metrics.csv`에 저장했다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    require(not result_root.exists(), f"Refusing to overwrite {result_root}")
    require(HELPER.is_file(), "Shared-transfer helper is missing")
    helper = load_module("hard_lmm_shared_block_transfer_helpers", HELPER)
    contract = read_json(CONTRACT)
    validate_contract(contract)
    evidence = verify_reused_evidence(contract)
    base_contract = read_json(BASE_CONTRACT)
    source = base.verify_frozen_source(base_contract, args.frozen_source_root)
    frozen = base.import_frozen_modules(args.frozen_source_root)
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    target_outputs = core.target_outputs
    source["loaded_module_audit"] = base.audit_loaded_frozen_modules(
        base_contract, args.frozen_source_root.resolve()
    )
    torch.set_num_threads(int(contract["diagnostic_scope"]["torch_threads"]))
    torch.manual_seed(int(contract["diagnostic_scope"]["sample_seed"]))
    np.random.seed(int(contract["diagnostic_scope"]["sample_seed"]))
    result_root.mkdir(parents=True)
    save_json(result_root / "execution_contract.json", contract)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "contract_sha256": sha256_file(CONTRACT),
        "source": source,
        "reused_evidence": evidence,
        "diagnostic_source_hashes": source_hashes(),
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "validation_rows_materialized": False,
        "held_out_rows_materialized": False,
        "backbone_or_quantity_parameter_updates": False,
        "source_fold_time_head_calibration": True,
    }
    save_json(result_root / "execution_manifest.json", manifest)
    try:
        results: dict[str, Any] = {}
        metric_rows: list[dict[str, Any]] = []
        for row in base_contract["datasets"]:
            result, rows = analyze_dataset(
                row,
                frozen=frozen,
                target_outputs=target_outputs,
                helper=helper,
                contract=contract,
            )
            results[str(row["dataset"])] = result
            metric_rows.extend(rows)
        save_json(result_root / "analysis.json", results)
        write_metric_csv(result_root / "metrics.csv", metric_rows)
        decision = decision_from_results(results, contract)
        save_json(result_root / "evidence_decision.json", decision)
        (result_root / "README.md").write_text(
            render_readme(results, decision), encoding="utf-8"
        )
        manifest.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "analysis_sha256": sha256_file(result_root / "analysis.json"),
                "metrics_sha256": sha256_file(result_root / "metrics.csv"),
                "decision_sha256": sha256_file(result_root / "evidence_decision.json"),
                "readme_sha256": sha256_file(result_root / "README.md"),
            }
        )
        save_json(result_root / "execution_manifest.json", manifest)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        save_json(result_root / "execution_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
