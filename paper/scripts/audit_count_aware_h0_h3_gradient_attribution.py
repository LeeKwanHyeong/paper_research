#!/usr/bin/env python3
"""Diagnose H0/H3 clipping and quantity damage on train rows only."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import (
    prepare_count_frame,
    right_pad_batch,
    target_outputs,
)
from paper.scripts.count_aware_tpp_backbone.reporting import write_csv
from paper.scripts.run_count_aware_tpp_backbone_control import DATASET_CONTRACTS
from paper.scripts.run_taxi_quantity_interface_ablation import (
    make_loader,
    save_json,
    set_seed,
    sha256_file,
)
from simple_lab_test.search.common.runner import torch_load_checkpoint


SEED = 42
BACKBONE = "titantpp"
VARIANT = "count_only_log_mse_tail_shared"
GRAD_CLIP = 1.0
CLIPPING_FRACTION_THRESHOLD = 0.95
GROUP_DOMINANCE_THRESHOLD = 0.50
QUANTITY_DAMAGE_THRESHOLD_PCT = 2.0
TIME_PARAMETER_NAMES = {"v_t.weight", "b_t", "w_raw"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--integration-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execution-role", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lookback-weeks", type=int, default=520)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--audit-batches", type=int, default=32)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def load_train_only_frame(path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_groups(
    model: torch.nn.Module,
) -> dict[str, tuple[tuple[str, torch.nn.Parameter], ...]]:
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "shared_encoder": [],
        "time_head": [],
        "quantity_head": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in TIME_PARAMETER_NAMES:
            group = "time_head"
        elif name.startswith("quantity_head.") or name.startswith(
            "quantity_scale_head."
        ):
            group = "quantity_head"
        else:
            group = "shared_encoder"
        groups[group].append((name, parameter))
    if any(not values for values in groups.values()):
        raise ValueError("Every diagnostic parameter group must be nonempty")
    selected = [name for values in groups.values() for name, _ in values]
    expected = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if sorted(selected) != sorted(expected):
        raise ValueError("Diagnostic parameter groups must partition the model")
    return {key: tuple(values) for key, values in groups.items()}


def _norm(gradients: Iterable[torch.Tensor | None]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared += torch.square(gradient.detach().double()).sum().cpu()
    return float(torch.sqrt(squared).item())


def _cosine(
    left: Iterable[torch.Tensor | None],
    right: Iterable[torch.Tensor | None],
) -> float:
    dot = torch.zeros((), dtype=torch.float64)
    left_sq = torch.zeros((), dtype=torch.float64)
    right_sq = torch.zeros((), dtype=torch.float64)
    for first, second in zip(left, right, strict=True):
        if first is None or second is None:
            continue
        first64 = first.detach().double()
        second64 = second.detach().double()
        dot += (first64 * second64).sum().cpu()
        left_sq += torch.square(first64).sum().cpu()
        right_sq += torch.square(second64).sum().cpu()
    denominator = float(torch.sqrt(left_sq * right_sq).item())
    return float(dot.item() / denominator) if denominator > 0.0 else 0.0


def gradient_attribution(
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    time_gradients: tuple[torch.Tensor | None, ...],
    quantity_gradients: tuple[torch.Tensor | None, ...],
    *,
    grad_clip: float,
) -> dict[str, float | str]:
    if len(named_parameters) != len(time_gradients) or len(named_parameters) != len(
        quantity_gradients
    ):
        raise ValueError("Gradient tuples must match the named parameter order")
    names = [name for name, _ in named_parameters]
    joint_gradients = tuple(
        (time if quantity is None else quantity)
        if time is None
        else (time if quantity is None else time + quantity)
        for time, quantity in zip(time_gradients, quantity_gradients, strict=True)
    )
    time_global = _norm(time_gradients)
    quantity_global = _norm(quantity_gradients)
    joint_global = _norm(joint_gradients)
    clip_scale = min(1.0, grad_clip / joint_global) if joint_global > 0.0 else 1.0
    output: dict[str, float | str] = {
        "time_global_grad_norm": time_global,
        "quantity_global_grad_norm": quantity_global,
        "joint_global_grad_norm": joint_global,
        "clip_scale": clip_scale,
        "clipped": float(joint_global > grad_clip),
    }
    shares: dict[str, float] = {}
    for group in ("shared_encoder", "time_head", "quantity_head"):
        indices = [
            index
            for index, name in enumerate(names)
            if (
                (group == "time_head" and name in TIME_PARAMETER_NAMES)
                or (
                    group == "quantity_head"
                    and (
                        name.startswith("quantity_head.")
                        or name.startswith("quantity_scale_head.")
                    )
                )
                or (
                    group == "shared_encoder"
                    and name not in TIME_PARAMETER_NAMES
                    and not name.startswith("quantity_head.")
                    and not name.startswith("quantity_scale_head.")
                )
            )
        ]
        joint_norm = _norm(joint_gradients[index] for index in indices)
        output[f"{group}_time_grad_norm"] = _norm(
            time_gradients[index] for index in indices
        )
        output[f"{group}_quantity_grad_norm"] = _norm(
            quantity_gradients[index] for index in indices
        )
        output[f"{group}_joint_grad_norm"] = joint_norm
        output[f"{group}_post_clip_grad_norm"] = joint_norm * clip_scale
        share = (
            joint_norm * joint_norm / (joint_global * joint_global)
            if joint_global > 0.0
            else 0.0
        )
        output[f"{group}_joint_sq_norm_share"] = share
        shares[group] = share
    shared_indices = [
        index
        for index, name in enumerate(names)
        if name not in TIME_PARAMETER_NAMES
        and not name.startswith("quantity_head.")
        and not name.startswith("quantity_scale_head.")
    ]
    output["shared_time_quantity_grad_cosine"] = _cosine(
        (time_gradients[index] for index in shared_indices),
        (quantity_gradients[index] for index in shared_indices),
    )
    output["dominant_joint_gradient_group"] = max(shares, key=shares.get)
    finite = [
        value for value in output.values() if isinstance(value, (int, float))
    ]
    if not all(math.isfinite(float(value)) for value in finite):
        raise FloatingPointError("Non-finite gradient attribution metric")
    return output


def build_model_from_summary(
    summary: dict[str, Any],
    *,
    device: str,
) -> torch.nn.Module:
    interface = summary["interface_meta"]
    time_head = interface["time_head"]
    model, _ = build_count_aware_model(
        BACKBONE,
        hidden_dim=int(summary["encoder_config"]["d_model"]),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface["train_target_std"]),
        max_seq_len=int(summary["encoder_config"]["max_len"]),
        quantity_variant=str(summary["variant"]),
        lambda_tail=float(summary["lambda_tail"]),
        tail_threshold=float(summary["tail_threshold"]),
        tail_normalization_scale=float(summary["tail_normalization_scale"]),
        tail_clip_cap=float(summary["tail_clip_cap"]),
        tail_huber_delta=float(summary["tail_huber_delta"]),
        time_head_mode=str(time_head["mode"]),
        time_scale=float(time_head["time_scale"]),
        time_w_max=float(time_head.get("time_w_max", 10.0 / 3.0)),
        time_intercept_limit=float(time_head.get("time_intercept_limit", 30.0)),
        time_initial_intercept=float(time_head.get("time_initial_intercept", 0.0)),
        time_wd_safety_limit=float(time_head.get("time_wd_safety_limit", 40.0)),
        time_initial_location=time_head.get("time_initial_location"),
        time_initial_scale=time_head.get("time_initial_scale"),
        time_sigma_floor=float(time_head.get("time_sigma_floor", 1e-3)),
    )
    return model.to(device)


def checkpoint_states(run_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    best = torch_load_checkpoint(
        run_dir / "best_val_joint_objective_model.pt", map_location="cpu"
    )
    final = torch_load_checkpoint(run_dir / "last_epoch_state.pt", map_location="cpu")
    if best.get("held_out_test_evaluated") or final.get("held_out_test_evaluated"):
        raise ValueError("Held-out test must remain locked")
    return {
        "best": best["model_state_dict"],
        "final": final["model_state_dict"],
    }


def audit_gradient_state(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: str,
    max_batches: int,
    variant: str,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.train()
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    parameter_groups(model)
    parameters = tuple(parameter for _, parameter in named_parameters)
    rows: list[dict[str, Any]] = []
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if batch_index >= max_batches:
            break
        if quantities is None:
            raise ValueError("Gradient attribution requires raw quantities")
        torch.manual_seed(SEED + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + batch_index)
        outputs = target_outputs(
            model,
            dts.to(device),
            mask.to(device),
            quantities.to(device),
            lambda_log_qty=1.0,
        )
        time_mean = outputs["time_loss"].mean()
        quantity_mean = outputs["quantity_train_loss"].mean()
        time_gradients = torch.autograd.grad(
            time_mean,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        quantity_gradients = torch.autograd.grad(
            quantity_mean,
            parameters,
            allow_unused=True,
        )
        attribution = gradient_attribution(
            named_parameters,
            time_gradients,
            quantity_gradients,
            grad_clip=GRAD_CLIP,
        )
        true_qty = outputs["true_qty"].detach().double()
        pred_qty = outputs["pred_qty"].detach().double()
        errors = pred_qty - true_qty
        rows.append(
            {
                "variant": variant,
                "stage": stage,
                "batch_index": batch_index,
                "event_count": int(true_qty.numel()),
                "time_nll": float(time_mean.detach().cpu().item()),
                "quantity_loss": float(quantity_mean.detach().cpu().item()),
                "quantity_mae": float(errors.abs().mean().cpu().item()),
                "quantity_rmse": float(
                    torch.sqrt(torch.square(errors).mean()).cpu().item()
                ),
                **attribution,
            }
        )
    if not rows:
        raise ValueError("No train batches were audited")
    return summarize_gradient_rows(rows, variant=variant, stage=stage), rows


def summarize_gradient_rows(
    rows: list[dict[str, Any]], *, variant: str, stage: str
) -> dict[str, Any]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "batch_index"
    ]
    summary: dict[str, Any] = {
        "variant": variant,
        "stage": stage,
        "audited_batches": len(rows),
        "audited_events": sum(int(row["event_count"]) for row in rows),
    }
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_median"] = float(np.median(values))
    group_shares = {
        group: float(summary[f"{group}_joint_sq_norm_share_median"])
        for group in ("shared_encoder", "time_head", "quantity_head")
    }
    summary["dominant_joint_gradient_group"] = max(group_shares, key=group_shares.get)
    summary["all_finite"] = True
    return summary


def quantity_predictions(
    model: torch.nn.Module,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    _, quantity_states = model.encode_task_states(dts, history_quantities, mask)
    hidden = quantity_states[batch_ids, history_positions]
    true_qty = quantities[batch_ids, target_positions].float()
    _, pred_qty = model.predict_quantity(hidden)
    return true_qty, pred_qty


def mixed_quantity_state(
    encoder_state: dict[str, torch.Tensor],
    head_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mixed = {key: value.detach().clone() for key, value in encoder_state.items()}
    quantity_keys = [key for key in mixed if key.startswith("quantity_head.")]
    if not quantity_keys:
        raise ValueError("No quantity-head state was found")
    for key in quantity_keys:
        mixed[key] = head_state[key].detach().clone()
    return mixed


@torch.no_grad()
def audit_quantity_crossing(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: str,
    max_batches: int,
    states: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combinations = (
        ("H0_encoder_H0_head", "H0", "H0"),
        ("H3_encoder_H3_head", "H3", "H3"),
        ("H3_encoder_H0_head", "H3", "H0"),
        ("H0_encoder_H3_head", "H0", "H3"),
    )
    for label, encoder_source, head_source in combinations:
        model.load_state_dict(
            mixed_quantity_state(states[encoder_source], states[head_source]),
            strict=True,
        )
        model.eval()
        count = 0
        absolute = 0.0
        squared = 0.0
        log_squared = 0.0
        for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
            if batch_index >= max_batches:
                break
            if quantities is None:
                raise ValueError("Quantity crossing requires raw quantities")
            true_qty, pred_qty = quantity_predictions(
                model,
                dts.to(device),
                mask.to(device),
                quantities.to(device),
            )
            error = pred_qty.double() - true_qty.double()
            count += int(error.numel())
            absolute += float(error.abs().sum().cpu().item())
            squared += float(torch.square(error).sum().cpu().item())
            log_squared += float(
                torch.square(
                    torch.log1p(pred_qty.double()) - torch.log1p(true_qty.double())
                )
                .sum()
                .cpu()
                .item()
            )
        if count < 1:
            raise ValueError("No quantity crossing batches were evaluated")
        rows.append(
            {
                "combination": label,
                "encoder_source": encoder_source,
                "quantity_head_source": head_source,
                "count": count,
                "qty_mae": absolute / count,
                "qty_rmse": math.sqrt(squared / count),
                "log_qty_mse": log_squared / count,
            }
        )
    return rows


def percent_change(candidate: float, reference: float) -> float:
    if reference == 0.0:
        raise ValueError("Percentage comparison requires a nonzero reference")
    return 100.0 * (candidate - reference) / reference


def classify_quantity_damage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {str(row["combination"]): row for row in rows}
    base = float(metrics["H0_encoder_H0_head"]["qty_mae"])
    full = percent_change(float(metrics["H3_encoder_H3_head"]["qty_mae"]), base)
    encoder = percent_change(float(metrics["H3_encoder_H0_head"]["qty_mae"]), base)
    head = percent_change(float(metrics["H0_encoder_H3_head"]["qty_mae"]), base)
    encoder_bad = encoder > QUANTITY_DAMAGE_THRESHOLD_PCT
    head_bad = head > QUANTITY_DAMAGE_THRESHOLD_PCT
    if full <= QUANTITY_DAMAGE_THRESHOLD_PCT:
        location = "no_train_quantity_damage"
    elif encoder_bad and head_bad:
        location = "encoder_and_quantity_head"
    elif encoder_bad:
        location = "encoder_dominant"
    elif head_bad:
        location = "quantity_head_dominant"
    else:
        location = "encoder_head_coupling"
    return {
        "quantity_damage_location": location,
        "full_h3_mae_change_pct": full,
        "h3_encoder_h0_head_mae_change_pct": encoder,
        "h0_encoder_h3_head_mae_change_pct": head,
        "threshold_pct": QUANTITY_DAMAGE_THRESHOLD_PCT,
        "interpretation_limit": "checkpoint_crossing_is_diagnostic_not_causal_proof",
    }


def relative_parameter_drift(
    current: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
    prefix_group: str,
) -> float:
    if prefix_group == "time_head":
        keys = [key for key in current if key in TIME_PARAMETER_NAMES]
    elif prefix_group == "quantity_head":
        keys = [key for key in current if key.startswith("quantity_head.")]
    else:
        keys = [
            key
            for key in current
            if key not in TIME_PARAMETER_NAMES and not key.startswith("quantity_head.")
        ]
    delta_sq = sum(
        float(torch.square(current[key].double() - initial[key].double()).sum().item())
        for key in keys
    )
    initial_sq = sum(
        float(torch.square(initial[key].double()).sum().item()) for key in keys
    )
    return math.sqrt(delta_sq) / max(math.sqrt(initial_sq), 1e-12)


def classify_clipping(summary: dict[str, Any]) -> dict[str, Any]:
    shares = {
        group: float(summary[f"{group}_joint_sq_norm_share_median"])
        for group in ("shared_encoder", "time_head", "quantity_head")
    }
    driver = max(shares, key=shares.get)
    clipping_fraction = float(summary["clipped_mean"])
    return {
        "clipping_persistent": clipping_fraction >= CLIPPING_FRACTION_THRESHOLD,
        "clipping_fraction": clipping_fraction,
        "dominant_gradient_group": driver,
        "dominant_group_share": shares[driver],
        "dominant_group_contract_met": shares[driver] >= GROUP_DOMINANCE_THRESHOLD,
        "median_clip_scale": float(summary["clip_scale_median"]),
        "thresholds": {
            "clipping_fraction": CLIPPING_FRACTION_THRESHOLD,
            "group_sq_norm_share": GROUP_DOMINANCE_THRESHOLD,
        },
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    if args.audit_batches < 1 or args.max_seq_len != 256:
        raise ValueError("Frozen audit requires positive batches and max_seq_len=256")
    status_path = args.output_dir / "status.json"
    if status_path.exists() and not args.force_rerun:
        raise FileExistsError(f"Audit already completed: {status_path}")

    dataset_contract = DATASET_CONTRACTS["intermittent_frozen_5000"]
    data_sha256 = sha256_file(args.data)
    split_sha256 = sha256_file(args.split_manifest)
    if data_sha256 != dataset_contract["data_sha256"]:
        raise ValueError("Unexpected Intermittent data SHA-256")
    if split_sha256 != dataset_contract["split_manifest_sha256"]:
        raise ValueError("Unexpected Intermittent split manifest SHA-256")
    train_raw = load_train_only_frame(args.data)
    if set(train_raw["chronological_split"]) != {"train"}:
        raise ValueError("Audit must load train rows only")
    frame = prepare_count_frame(train_raw)
    loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=False,
        generator=None,
    )

    run_dirs = {
        "H0": args.integration_artifact
        / "h0_scaled_exact_tail_shared/runs/titantpp"
        / VARIANT
        / "seed_42",
        "H3": args.integration_artifact
        / "h3_lognormal_tail_shared/runs/titantpp"
        / VARIANT
        / "seed_42",
    }
    summaries = {
        label: read_json(run_dir / "summary.json")
        for label, run_dir in run_dirs.items()
    }
    if any(summary["held_out_test_evaluated"] for summary in summaries.values()):
        raise ValueError("Held-out test must remain locked")
    if any(summary["evaluation_scope"] != "validation_only" for summary in summaries.values()):
        raise ValueError("Source checkpoints must come from the matched validation screen")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "count_aware_h0_h3_gradient_attribution",
        "dataset": "intermittent_frozen_5000",
        "loaded_split": "train_only",
        "data_sha256": data_sha256,
        "split_manifest_sha256": split_sha256,
        "source_integration_artifact": str(args.integration_artifact),
        "checkpoint_stages": ["initial", "best", "final"],
        "variants": ["H0_scaled_exact", "H3_lognormal_duration"],
        "batch_size": args.batch_size,
        "audit_batches": args.audit_batches,
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "grad_clip": GRAD_CLIP,
        "dropout_replay": "train_mode_seed_42_plus_batch_index",
        "selection_source": "train_only_diagnostic",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "execution_host": os.uname().nodename,
        "execution_role": args.execution_role,
    }
    save_json(args.output_dir / "launch_contract.json", contract)

    gradient_rows: list[dict[str, Any]] = []
    gradient_summaries: list[dict[str, Any]] = []
    states_by_variant: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    initial_states: dict[str, dict[str, torch.Tensor]] = {}
    drift_rows: list[dict[str, Any]] = []
    started = time.time()
    for label in ("H0", "H3"):
        set_seed(SEED)
        model = build_model_from_summary(summaries[label], device=args.device)
        initial_states[label] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        states = checkpoint_states(run_dirs[label])
        states_by_variant[label] = states
        stage_states = {"initial": initial_states[label], **states}
        for stage, state in stage_states.items():
            model.load_state_dict(state, strict=True)
            summary, rows = audit_gradient_state(
                model=model,
                loader=loader,
                device=args.device,
                max_batches=args.audit_batches,
                variant=label,
                stage=stage,
            )
            gradient_summaries.append(summary)
            gradient_rows.extend(rows)
            print(
                f"[audit {label}/{stage}] clip={summary['clipped_mean']:.6f} "
                f"scale={summary['clip_scale_median']:.6f} "
                f"driver={summary['dominant_joint_gradient_group']} "
                f"qty_mae={summary['quantity_mae_mean']:.6f}",
                flush=True,
            )
            if stage != "initial":
                for group in ("shared_encoder", "time_head", "quantity_head"):
                    drift_rows.append(
                        {
                            "variant": label,
                            "stage": stage,
                            "parameter_group": group,
                            "relative_l2_drift": relative_parameter_drift(
                                state, initial_states[label], group
                            ),
                        }
                    )

    best_states = {
        label: states_by_variant[label]["best"] for label in ("H0", "H3")
    }
    crossing_model = build_model_from_summary(summaries["H0"], device=args.device)
    crossing_rows = audit_quantity_crossing(
        model=crossing_model,
        loader=loader,
        device=args.device,
        max_batches=args.audit_batches,
        states=best_states,
    )
    h3_best = next(
        row
        for row in gradient_summaries
        if row["variant"] == "H3" and row["stage"] == "best"
    )
    h3_final = next(
        row
        for row in gradient_summaries
        if row["variant"] == "H3" and row["stage"] == "final"
    )
    decision = {
        "schema_version": 1,
        "status": "complete",
        "h0_model_status": "comparison_incumbent_with_unresolved_instability",
        "h3_best_clipping": classify_clipping(h3_best),
        "h3_final_clipping": classify_clipping(h3_final),
        "quantity_damage": classify_quantity_damage(crossing_rows),
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
    }
    write_csv(args.output_dir / "gradient_batch_metrics.csv", gradient_rows)
    write_csv(args.output_dir / "gradient_stage_summary.csv", gradient_summaries)
    write_csv(args.output_dir / "quantity_checkpoint_crossing.csv", crossing_rows)
    write_csv(args.output_dir / "parameter_drift.csv", drift_rows)
    save_json(args.output_dir / "decision.json", decision)
    contract.update(
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "h0_model_status": decision["h0_model_status"],
        }
    )
    save_json(args.output_dir / "launch_contract.json", contract)
    save_json(
        status_path,
        {
            "status": "complete",
            "validation_evaluated": False,
            "held_out_test_evaluated": False,
            "source_revision": args.source_revision,
        },
    )
    print(
        "[complete] clipping_driver="
        f"{decision['h3_final_clipping']['dominant_gradient_group']} "
        "quantity_damage="
        f"{decision['quantity_damage']['quantity_damage_location']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
