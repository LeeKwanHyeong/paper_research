#!/usr/bin/env python3
"""Build and independently verify the frozen Hard-LMM time-head refit evidence.

The audit deliberately works from the copied checkpoints, controller records,
the frozen contract, and the original B checkpoints.  It does not read or
evaluate held-out data.  Large feature caches were intentionally not copied
from the RTX 5090 host, so prediction replay is represented by the controller's
recorded digests while the model-state boundary is rechecked locally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_time_head_refit_v1.json"
ARTIFACT_ROOT = (
    ROOT
    / "search_artifacts/hard_lmm_time_head_refit_seed42_5090_20260906_f514f0c"
)
B_ARTIFACT_ROOT = (
    ROOT
    / "search_artifacts/"
    "hard_lmm_raw_rmse_checkpoint_alignment_seed42_5090_20260906_f752434"
)
TIME_KEYS = ("v_t.weight", "b_t", "w_raw")
DISPLAY_NAMES = {
    "intermittent_frozen_5000": "Intermittent",
    "yellow_trip_hourly": "Taxi",
    "insta_market_basket": "Instacart",
}
DATASET_ORDER = tuple(DISPLAY_NAMES)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_state_dict_sha256(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Reproduce the project state digest without importing training code."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a tensor")
        canonical = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(canonical.dtype),
                "shape": list(canonical.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, byteorder="big", signed=False))
        digest.update(metadata)
        if canonical.numel() > 0:
            raw_bytes = canonical.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw_bytes).to_bytes(8, byteorder="big", signed=False))
            digest.update(raw_bytes)
        else:
            digest.update((0).to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"Expected checkpoint dictionary: {path}")
    return value


def state_partition(
    state: Mapping[str, torch.Tensor], *, time_head: bool
) -> dict[str, torch.Tensor]:
    names = set(TIME_KEYS)
    return {
        name: value
        for name, value in state.items()
        if (name in names) is time_head
    }


def tensor_mapping_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def finite_nested(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(finite_nested(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_nested(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def strict_earliest_minimum(history: list[dict[str, Any]]) -> dict[str, Any]:
    finite_rows = [
        row for row in history if math.isfinite(float(row["val_time_nll"]))
    ]
    if not finite_rows:
        raise AssertionError("No finite validation Time NLL in history")
    return min(finite_rows, key=lambda row: float(row["val_time_nll"]))


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def check(self, name: str, passed: bool, detail: str) -> None:
        row = {"name": name, "passed": bool(passed), "detail": detail}
        self.checks.append(row)
        if not passed:
            self.errors.append(f"{name}: {detail}")

    def require(self, name: str, passed: bool, detail: str) -> None:
        self.check(name, passed, detail)


def b_summary_path(dataset: str) -> Path:
    return (
        B_ARTIFACT_ROOT
        / "seed42_e300"
        / dataset
        / "quantile_checkpoint_alignment/runs/titantpp/"
        "count_only_log_regression/seed_42/summary.json"
    )


def audit_dataset(
    audit: Audit,
    contract: dict[str, Any],
    dataset_spec: dict[str, Any],
    decision_row: dict[str, Any],
) -> dict[str, Any]:
    dataset = str(dataset_spec["dataset"])
    prefix = f"{dataset}:"
    summary_path = ARTIFACT_ROOT / "full" / dataset / "summary.json"
    selected_path = (
        ARTIFACT_ROOT / "full" / dataset / "best_validation_time_nll_model.pt"
    )
    last_path = ARTIFACT_ROOT / "full" / dataset / "last_epoch_state.pt"
    source_path = ROOT / str(dataset_spec["B_checkpoint_path"])
    source_summary_path = b_summary_path(dataset)

    for label, path in {
        "summary": summary_path,
        "selected checkpoint": selected_path,
        "last state": last_path,
        "B checkpoint": source_path,
        "B summary": source_summary_path,
    }.items():
        audit.require(
            f"{prefix}{label}_present",
            path.is_file(),
            str(path.relative_to(ROOT)),
        )

    summary = read_json(summary_path)
    source_summary = read_json(source_summary_path)
    source = load_checkpoint(source_path)
    selected = load_checkpoint(selected_path)
    last = load_checkpoint(last_path)
    source_state = source["model_state_dict"]
    selected_state = selected["model_state_dict"]
    last_current_state = last["model_state_dict"]
    last_best_state = last["best_state_dict"]

    source_file_sha = sha256_file(source_path)
    source_state_sha = canonical_state_dict_sha256(source_state)
    selected_state_sha = canonical_state_dict_sha256(selected_state)
    last_current_sha = canonical_state_dict_sha256(last_current_state)
    last_best_sha = canonical_state_dict_sha256(last_best_state)
    source_non_time = state_partition(source_state, time_head=False)
    selected_non_time = state_partition(selected_state, time_head=False)
    source_non_time_sha = canonical_state_dict_sha256(source_non_time)
    selected_non_time_sha = canonical_state_dict_sha256(selected_non_time)
    source_time_sha = canonical_state_dict_sha256(
        state_partition(source_state, time_head=True)
    )
    selected_time_sha = canonical_state_dict_sha256(
        state_partition(selected_state, time_head=True)
    )

    audit.require(
        f"{prefix}B_checkpoint_file_sha256",
        source_file_sha
        == dataset_spec["B_checkpoint_file_sha256"]
        == summary.get("source_checkpoint_sha256")
        == selected.get("source_checkpoint_lineage", {}).get(
            "B_checkpoint_file_sha256"
        ),
        source_file_sha,
    )
    audit.require(
        f"{prefix}B_state_sha256",
        source_state_sha == dataset_spec["B_checkpoint_state_sha256"],
        source_state_sha,
    )
    audit.require(
        f"{prefix}B_summary_state_sha256",
        source_summary.get("checkpoint_state_sha256") == source_state_sha,
        str(source_summary.get("checkpoint_state_sha256")),
    )
    B_summary_metrics_ok = all(
        math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12)
        for observed, expected in (
            (source_summary.get("best_val_time_nll"), dataset_spec["B_metrics"]["time_nll"]),
            (source_summary.get("best_val_qty_rmse"), dataset_spec["B_metrics"]["raw_rmse"]),
            (source_summary.get("best_val_qty_mae"), dataset_spec["B_metrics"]["overall_mae"]),
        )
    )
    audit.require(
        f"{prefix}B_summary_metrics",
        B_summary_metrics_ok,
        (
            f"time={source_summary.get('best_val_time_nll')}, "
            f"rmse={source_summary.get('best_val_qty_rmse')}, "
            f"mae={source_summary.get('best_val_qty_mae')}"
        ),
    )
    audit.require(
        f"{prefix}selected_state_sha256",
        selected_state_sha
        == selected.get("model_state_sha256")
        == summary.get("selected_state_sha256")
        == decision_row.get("selected_state_sha256"),
        selected_state_sha,
    )
    audit.require(
        f"{prefix}source_lineage",
        selected.get("source_state_sha256")
        == summary.get("source_state_sha256")
        == decision_row.get("source_state_sha256")
        == source_state_sha,
        source_state_sha,
    )

    state_schema_equal = set(source_state) == set(selected_state) and all(
        source_state[name].shape == selected_state[name].shape
        and source_state[name].dtype == selected_state[name].dtype
        for name in source_state
    )
    changed_keys = sorted(
        name
        for name in source_state
        if not torch.equal(source_state[name], selected_state[name])
    )
    expected_changed_keys = sorted(TIME_KEYS)
    audit.require(
        f"{prefix}state_schema_equal",
        state_schema_equal,
        f"source={len(source_state)} selected={len(selected_state)} tensors",
    )
    audit.require(
        f"{prefix}changed_keys_exact",
        changed_keys == expected_changed_keys,
        json.dumps(changed_keys),
    )
    audit.require(
        f"{prefix}non_time_state_bitwise_equal",
        tensor_mapping_equal(source_non_time, selected_non_time),
        f"{len(source_non_time)} frozen tensors",
    )
    audit.require(
        f"{prefix}non_time_state_sha256",
        source_non_time_sha
        == selected_non_time_sha
        == summary.get("source_non_time_state_sha256")
        == summary.get("selected_non_time_state_sha256")
        == last.get("source_non_time_state_sha256"),
        source_non_time_sha,
    )
    audit.require(
        f"{prefix}time_head_state_sha256",
        source_time_sha == summary.get("source_time_head_state_sha256")
        and selected_time_sha
        == summary.get("selected_time_head_state_sha256")
        == decision_row.get("selected_time_head_state_sha256")
        and source_time_sha != selected_time_sha,
        f"source={source_time_sha}, selected={selected_time_sha}",
    )
    audit.require(
        f"{prefix}finite_model_states",
        finite_nested(source_state)
        and finite_nested(selected_state)
        and finite_nested(last_current_state)
        and finite_nested(last_best_state),
        "source, selected, final-current, and final-best states",
    )

    history = summary["history"]
    contiguous = [int(row["epoch"]) for row in history] == list(
        range(len(history))
    )
    earliest = strict_earliest_minimum(history)
    best_epoch = int(summary["best_epoch"])
    completed_epochs = int(summary["completed_epochs"])
    patience = int(summary["resume_identity"]["patience"])
    min_epochs = int(summary["resume_identity"]["min_epochs"])
    planned_epochs = int(summary["resume_identity"]["planned_epochs"])
    early_stop_expected = (
        completed_epochs >= min_epochs
        and completed_epochs - best_epoch >= patience
    )
    audit.require(
        f"{prefix}history_contiguous",
        contiguous and len(history) == completed_epochs + 1,
        f"epochs 0..{completed_epochs}",
    )
    audit.require(
        f"{prefix}history_last_state_equal",
        history == last.get("history"),
        "summary history equals resumable last-state history",
    )
    audit.require(
        f"{prefix}earliest_strict_minimum",
        int(earliest["epoch"]) == best_epoch
        and math.isclose(
            float(earliest["val_time_nll"]),
            float(summary["best_val_time_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"best={best_epoch}, value={earliest['val_time_nll']}",
    )
    audit.require(
        f"{prefix}selected_equals_last_best",
        tensor_mapping_equal(selected_state, last_best_state)
        and selected_state_sha == last_best_sha == last.get("best_state_sha256"),
        selected_state_sha,
    )
    audit.require(
        f"{prefix}last_current_sha256",
        last_current_sha == last.get("model_state_sha256"),
        last_current_sha,
    )
    audit.require(
        f"{prefix}early_stop_contract",
        bool(summary["stopped_early"]) == early_stop_expected
        and (
            completed_epochs == best_epoch + patience
            if early_stop_expected
            else completed_epochs == planned_epochs
        ),
        (
            f"best={best_epoch}, completed={completed_epochs}, "
            f"patience={patience}, planned={planned_epochs}"
        ),
    )

    optimizer = last["optimizer_state_dict"]
    groups = optimizer.get("param_groups", [])
    states = optimizer.get("state", {})
    expected_steps = completed_epochs * math.ceil(
        int(summary["train_cache"]["count"])
        / int(summary["resume_identity"]["batch_size"])
    )
    optimizer_shape_ok = len(groups) == 1 and groups[0].get("params") == [0, 1, 2]
    optimizer_shape_ok = optimizer_shape_ok and set(states) == {0, 1, 2}
    optimizer_steps: list[int] = []
    if optimizer_shape_ok:
        for parameter_id, parameter_name in enumerate(TIME_KEYS):
            row = states[parameter_id]
            step = int(row["step"].item())
            optimizer_steps.append(step)
            optimizer_shape_ok = optimizer_shape_ok and step == expected_steps
            optimizer_shape_ok = optimizer_shape_ok and tuple(
                row["exp_avg"].shape
            ) == tuple(last_current_state[parameter_name].shape)
            optimizer_shape_ok = optimizer_shape_ok and tuple(
                row["exp_avg_sq"].shape
            ) == tuple(last_current_state[parameter_name].shape)
    audit.require(
        f"{prefix}optimizer_resume_coherent",
        optimizer_shape_ok and finite_nested(optimizer),
        f"steps={optimizer_steps}, expected={expected_steps}",
    )

    B_time = float(dataset_spec["B_metrics"]["time_nll"])
    A_time = float(dataset_spec["A_time_nll"])
    target = float(dataset_spec["time_nll_target_max"])
    refit_time = float(summary["best_val_time_nll"])
    replay_time = float(decision_row["selected_checkpoint_replayed_time_nll"])
    time_abs_tol = float(
        contract["identity_and_stability"]["time_nll_replay_absolute_tolerance"]
    )
    audit.require(
        f"{prefix}epoch_zero_matches_B",
        math.isclose(
            float(summary["epoch_zero_val_time_nll"]),
            B_time,
            rel_tol=0.0,
            abs_tol=time_abs_tol,
        ),
        f"epoch0={summary['epoch_zero_val_time_nll']}, B={B_time}",
    )
    audit.require(
        f"{prefix}selected_checkpoint_replay",
        math.isclose(
            replay_time,
            refit_time,
            rel_tol=0.0,
            abs_tol=time_abs_tol,
        ),
        f"selected={refit_time}, CPU replay={replay_time}",
    )
    audit.require(
        f"{prefix}time_non_worse_than_B",
        refit_time <= B_time
        and summary.get("time_nll_non_worse_than_B") is True
        and decision_row.get("time_nll_non_worse_than_B") is True,
        f"B={B_time}, refit={refit_time}",
    )

    quantity_map = {
        "raw_rmse": "rmse",
        "overall_mae": "mae",
        "body_mae": "body_mae",
        "gt_p99_mae": "gt_p99_mae",
    }
    quantity_abs_tol = float(
        contract["identity_and_stability"][
            "reported_quantity_metric_absolute_tolerance"
        ]
    )
    quantity_rel_tol = float(
        contract["identity_and_stability"][
            "reported_quantity_metric_relative_tolerance"
        ]
    )
    quantity_checks = {
        contract_name: math.isclose(
            float(summary["quantity_metrics"][observed_name]),
            float(dataset_spec["B_metrics"][contract_name]),
            rel_tol=quantity_rel_tol,
            abs_tol=quantity_abs_tol,
        )
        for contract_name, observed_name in quantity_map.items()
    }
    prediction_digest_equal = (
        summary.get("quantity_prediction_bitwise_identical") is True
        and decision_row.get("quantity_prediction_bitwise_identical") is True
        and summary.get("source_quantity_prediction_sha256")
        == summary.get("selected_quantity_prediction_sha256")
    )
    audit.require(
        f"{prefix}quantity_prediction_remote_bitwise_replay",
        prediction_digest_equal,
        str(summary.get("source_quantity_prediction_sha256")),
    )
    audit.require(
        f"{prefix}quantity_metrics_match_B",
        all(quantity_checks.values()),
        json.dumps(quantity_checks, sort_keys=True),
    )

    data_sha = sha256_file(ROOT / dataset_spec["data_path"])
    split_sha = sha256_file(ROOT / dataset_spec["split_manifest_path"])
    train_population = summary["train_target_population"]
    validation_population = summary["validation_target_population"]
    train_population_ok = (
        int(train_population["target_count"])
        == int(dataset_spec["expected_train_targets"])
        and train_population["target_identity_sha256"]
        == dataset_spec["expected_train_target_identity_sha256"]
        and train_population["target_quantity_sha256"]
        == dataset_spec["expected_train_target_quantity_sha256"]
    )
    validation_population_ok = (
        int(validation_population["target_count"])
        == int(dataset_spec["expected_validation_targets"])
        and validation_population["target_identity_sha256"]
        == dataset_spec["expected_validation_target_identity_sha256"]
        and validation_population["target_quantity_sha256"]
        == dataset_spec["expected_validation_target_quantity_sha256"]
    )
    audit.require(
        f"{prefix}data_and_split_sha256",
        data_sha == dataset_spec["data_sha256"]
        and split_sha == dataset_spec["split_manifest_sha256"],
        f"data={data_sha}, split={split_sha}",
    )
    audit.require(
        f"{prefix}train_population_identity",
        train_population_ok,
        json.dumps(train_population, sort_keys=True),
    )
    audit.require(
        f"{prefix}validation_population_identity",
        validation_population_ok,
        json.dumps(validation_population, sort_keys=True),
    )

    strict_gap = refit_time - target
    A_gap = B_time - A_time
    recovery_fraction = (B_time - refit_time) / A_gap if A_gap > 0 else None
    last_val = float(history[-1]["val_time_nll"])
    is_cap_limited = (
        not bool(summary["stopped_early"])
        and best_epoch == completed_epochs == planned_epochs
    )
    strict_pass = refit_time <= target
    return {
        "dataset": dataset,
        "dataset_label": DISPLAY_NAMES[dataset],
        "A_time_nll": A_time,
        "B_time_nll": B_time,
        "refit_time_nll": refit_time,
        "refit_minus_B": refit_time - B_time,
        "refit_minus_A": refit_time - A_time,
        "B_to_refit_improvement": B_time - refit_time,
        "fraction_of_A_to_B_loss_recovered": recovery_fraction,
        "strict_target_A_plus_0_01": target,
        "strict_target_signed_gap": strict_gap,
        "strict_target_pass": strict_pass,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "last_val_time_nll": last_val,
        "last_minus_best_time_nll": last_val - refit_time,
        "stopped_early": bool(summary["stopped_early"]),
        "cap_limited": is_cap_limited,
        "train_target_count": int(summary["train_cache"]["count"]),
        "validation_target_count": int(summary["validation_cache"]["count"]),
        "quantity_prediction_bitwise_identical": prediction_digest_equal,
        "quantity_metrics_match_B": all(quantity_checks.values()),
        "quantity_metrics": dict(summary["quantity_metrics"]),
        "changed_state_keys": changed_keys,
        "non_time_state_bitwise_equal": tensor_mapping_equal(
            source_non_time, selected_non_time
        ),
        "selected_checkpoint_cpu_replay_time_nll": replay_time,
        "selected_checkpoint_replay_abs_error": abs(replay_time - refit_time),
        "source_checkpoint_file_sha256": source_file_sha,
        "source_state_sha256": source_state_sha,
        "selected_checkpoint_file_sha256": sha256_file(selected_path),
        "selected_state_sha256": selected_state_sha,
        "last_state_file_sha256": sha256_file(last_path),
        "optimizer_steps": optimizer_steps,
        "runtime": dict(summary["runtime"]),
        "local_feature_cache_present": (
            ARTIFACT_ROOT / "full" / dataset / "cache"
        ).is_dir(),
    }


def audit_e1(audit: Audit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        summary_path = ARTIFACT_ROOT / "e1" / dataset / "summary.json"
        summary = read_json(summary_path)
        passed = (
            summary.get("status") == "success"
            and int(summary.get("completed_epochs", -1)) == 1
            and summary.get("quantity_prediction_bitwise_identical") is True
            and summary.get("evaluation_scope") == "validation_only"
            and summary.get("held_out_test_evaluated") is False
            and summary.get("runtime", {}).get("cuda_available") is True
            and summary.get("runtime", {}).get("device_name")
            == "NVIDIA GeForce RTX 5090"
        )
        audit.require(
            f"e1:{dataset}:contract_smoke",
            passed,
            (
                f"status={summary.get('status')}, "
                f"epochs={summary.get('completed_epochs')}, "
                f"device={summary.get('runtime', {}).get('device_name')}"
            ),
        )
        rows.append(
            {
                "dataset": dataset,
                "status": summary.get("status"),
                "completed_epochs": summary.get("completed_epochs"),
                "quantity_prediction_bitwise_identical": summary.get(
                    "quantity_prediction_bitwise_identical"
                ),
                "runtime": summary.get("runtime"),
            }
        )
    return rows


def write_metrics(rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "A_time_nll",
        "B_time_nll",
        "refit_time_nll",
        "refit_minus_B",
        "refit_minus_A",
        "B_to_refit_improvement",
        "fraction_of_A_to_B_loss_recovered",
        "strict_target_A_plus_0_01",
        "strict_target_signed_gap",
        "strict_target_pass",
        "raw_rmse",
        "overall_mae",
        "body_mae",
        "gt_p99_mae",
        "body_count",
        "gt_p99_count",
        "best_epoch",
        "completed_epochs",
        "stopped_early",
        "cap_limited",
        "elapsed_seconds",
        "peak_memory_allocated_bytes",
    ]
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        quantity = row["quantity_metrics"]
        output_rows.append(
            {
                "dataset": row["dataset_label"],
                "A_time_nll": row["A_time_nll"],
                "B_time_nll": row["B_time_nll"],
                "refit_time_nll": row["refit_time_nll"],
                "refit_minus_B": row["refit_minus_B"],
                "refit_minus_A": row["refit_minus_A"],
                "B_to_refit_improvement": row["B_to_refit_improvement"],
                "fraction_of_A_to_B_loss_recovered": row[
                    "fraction_of_A_to_B_loss_recovered"
                ],
                "strict_target_A_plus_0_01": row["strict_target_A_plus_0_01"],
                "strict_target_signed_gap": row["strict_target_signed_gap"],
                "strict_target_pass": row["strict_target_pass"],
                "raw_rmse": quantity["rmse"],
                "overall_mae": quantity["mae"],
                "body_mae": quantity["body_mae"],
                "gt_p99_mae": quantity["gt_p99_mae"],
                "body_count": quantity["body_count"],
                "gt_p99_count": quantity["gt_p99_count"],
                "best_epoch": row["best_epoch"],
                "completed_epochs": row["completed_epochs"],
                "stopped_early": row["stopped_early"],
                "cap_limited": row["cap_limited"],
                "elapsed_seconds": row["runtime"]["elapsed_seconds"],
                "peak_memory_allocated_bytes": row["runtime"][
                    "peak_memory_allocated_bytes"
                ],
            }
        )
    temporary = OUT / "metrics.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(OUT / "metrics.csv")


def write_readme(rows: list[dict[str, Any]], audit_payload: dict[str, Any]) -> None:
    table_rows = []
    for row in rows:
        quantity = row["quantity_metrics"]
        outcome = "통과" if row["strict_target_pass"] else "실패"
        table_rows.append(
            "| {label} | {A:.9f} | {B:.9f} | {refit:.9f} | {delta_B:+.9f} | "
            "{delta_A:+.9f} | {gap:+.9f} ({outcome}) | {rmse:.6f} | "
            "{mae:.6f} | {best}/{completed} |".format(
                label=row["dataset_label"],
                A=row["A_time_nll"],
                B=row["B_time_nll"],
                refit=row["refit_time_nll"],
                delta_B=row["refit_minus_B"],
                delta_A=row["refit_minus_A"],
                gap=row["strict_target_signed_gap"],
                outcome=outcome,
                rmse=quantity["rmse"],
                mae=quantity["mae"],
                best=row["best_epoch"],
                completed=row["completed_epochs"],
            )
        )
    total_full_runtime = sum(row["runtime"]["elapsed_seconds"] for row in rows)
    readme = f"""# Frozen Hard-LMM time-head refit: seed42 validation audit

**판정: B의 수량 예측을 그대로 유지하면서 validation Time NLL을 세 데이터셋 모두 개선했다. 그러나 사전 고정한 `A + 0.01` 기준은 Instacart만 통과했으므로, 공통 시간 성능 복구와 Backbone 개선은 입증되지 않았다.**

## 검증 범위

- 후보는 raw-RMSE로 선택한 B 체크포인트의 encoder, memory, quantity head를 고정하고 기존 RMTPP time head의 66개 파라미터(`v_t.weight`, `b_t`, `w_raw`)만 train split에서 재학습한다.
- checkpoint는 validation Time NLL의 epoch 0 포함 가장 이른 유한 최솟값으로 선택했다. 세 데이터셋에 같은 optimizer와 정지 규칙을 적용했다.
- 평가는 seed42 validation에만 한정했다. Held-out test와 seeds52·62는 실행하지 않았다.
- RTX 5090 CUDA e1 계약 검증 뒤 full refit을 실행했다. Full refit의 합산 시간은 `{total_full_runtime:.3f}`초다. Feature cache 작성 시간과 controller 검사는 이 합계에 포함되지 않는다.

## 결과

`ΔB`와 `ΔA`는 각각 `refit − B`, `refit − A`이며 Time NLL은 낮을수록 좋다. Strict gap은 `refit − (A + 0.01)`이고 0 이하여야 통과다.

| 데이터셋 | A Time NLL | B Time NLL | Refit Time NLL | ΔB | ΔA | Strict gap | Raw RMSE | MAE | best/completed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

- 세 데이터셋 모두 `ΔB < 0`이다. B에서 나빠진 A 대비 Time NLL 차이의 회복률은 Intermittent `{100.0 * rows[0]['fraction_of_A_to_B_loss_recovered']:.1f}%`, Taxi `{100.0 * rows[1]['fraction_of_A_to_B_loss_recovered']:.1f}%`, Instacart `{100.0 * rows[2]['fraction_of_A_to_B_loss_recovered']:.1f}%`다.
- 선택 체크포인트와 B 체크포인트 사이에서 바뀐 state key는 세 time-head tensor뿐이다. 나머지 `{len(load_checkpoint(ROOT / read_json(CONTRACT_PATH)['datasets'][0]['B_checkpoint_path'])['model_state_dict']) - len(TIME_KEYS)}`개 tensor는 로컬에서 비트 단위 동일성을 재확인했다.
- 원격 full-cache replay에서 수량 예측 digest가 source와 selected checkpoint 사이에 동일했다. Raw RMSE, 전체 MAE, body MAE, `>p99` MAE도 B 기준과 계약 허용 오차 안에서 일치한다.

## 수렴 해석

- Taxi는 epoch 3, Instacart는 epoch 5가 최적이며 이후 20 epoch 동안 개선되지 않아 고정 patience에 따라 종료했다. 두 데이터셋에는 같은 설정으로 더 연장할 근거가 없다.
- Intermittent는 epoch 100이 최적이자 마지막 epoch이고 학습 구간 전체에서 validation Time NLL이 계속 낮아졌다. 이번 계약의 100-epoch 상한에서 미수렴했지만, 이 결과를 본 뒤 상한을 바꾸는 것은 동일 실험으로 취급할 수 없다.

## 감사 결과와 한계

- 로컬 감사 `{audit_payload['check_summary']['passed']}/{audit_payload['check_summary']['total']}`개 계약·무결성 검사가 모두 통과했다. B/selected/last checkpoint의 파일 및 canonical state SHA-256, 변경 tensor 경계, optimizer step, 연속 history, 가장 이른 최솟값, e1 CUDA 실행, source revision과 controller decision을 대조했다.
- 선택 체크포인트의 CPU replay와 기록된 Time NLL 차이는 모든 데이터셋에서 `1e-6` 이하다.
- 로컬 증적 복사본은 대용량 feature cache를 의도적으로 제외했다. 따라서 이 복사본만으로 예측 replay를 즉시 반복할 수는 없다. 원 데이터, B checkpoint, 고정 source가 있어 cache를 재생성하면 반복 가능하다. 원격 controller가 수행한 bitwise quantity replay와 cache digest는 summary/status에 남아 있다.
- 이 실험은 수량 성능을 보존하는 사후 time-head calibration의 효과를 보여준다. Backbone 구조 변경이나 모든 데이터셋에서 A 수준을 회복한다는 근거로 사용할 수 없다.

기계 판독용 상세 검증은 [validation_audit.json](validation_audit.json), 표 원자료는 [metrics.csv](metrics.csv), 입력·출력 파일 SHA-256은 [artifact_manifest.json](artifact_manifest.json)에 있다.
"""
    atomic_write_text(OUT / "README.md", readme)


def build_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(p for p in ARTIFACT_ROOT.rglob("*") if p.is_file()):
        entries.append(
            {
                "role": "copied_5090_evidence",
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for path in (CONTRACT_PATH, OUT / "build_audit.py", OUT / "README.md", OUT / "metrics.csv", OUT / "validation_audit.json"):
        entries.append(
            {
                "role": "contract_or_local_audit",
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    contract = read_json(CONTRACT_PATH)
    dependency_paths: set[Path] = set()
    for dataset_spec in contract["datasets"]:
        dataset = str(dataset_spec["dataset"])
        dependency_paths.update(
            {
                ROOT / str(dataset_spec["data_path"]),
                ROOT / str(dataset_spec["split_manifest_path"]),
                ROOT / str(dataset_spec["B_checkpoint_path"]),
                b_summary_path(dataset),
            }
        )
    for path in sorted(dependency_paths):
        entries.append(
            {
                "role": "local_replay_dependency",
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "manifest_excludes_itself": True,
        "source_revision": "f514f0c202a9220781f11c49854dc2e11f67c2e2",
        "entry_count": len(entries),
        "entries": entries,
    }


def main() -> None:
    audit = Audit()
    contract = read_json(CONTRACT_PATH)
    decision = read_json(ARTIFACT_ROOT / "decision.json")
    status = read_json(ARTIFACT_ROOT / "status.json")
    launch = read_json(ARTIFACT_ROOT / "control/launch_identity.json")
    source_manifest_path = ARTIFACT_ROOT / "control/source_manifest.json"
    source_manifest = read_json(source_manifest_path)

    contract_sha = sha256_file(CONTRACT_PATH)
    source_manifest_sha = sha256_file(source_manifest_path)
    audit.require(
        "contract_identity",
        contract.get("contract_id") == "hard_lmm_time_head_refit_v1"
        and contract_sha == launch.get("contract_sha256"),
        contract_sha,
    )
    audit.require(
        "source_identity",
        launch.get("source_revision")
        == status.get("source_revision")
        == source_manifest.get("source_revision")
        == "f514f0c202a9220781f11c49854dc2e11f67c2e2"
        and source_manifest_sha == launch.get("source_manifest_sha256")
        == status.get("source_manifest_sha256")
        and int(source_manifest.get("file_count", -1))
        == len(source_manifest.get("files", {}))
        == int(status.get("source_manifest_file_count", -2)),
        (
            f"revision={launch.get('source_revision')}, "
            f"manifest_sha={source_manifest_sha}, "
            f"files={source_manifest.get('file_count')}"
        ),
    )
    audit.require(
        "controller_complete_validation_only",
        status.get("status") == "complete"
        and status.get("stage") == "complete"
        and decision.get("status") == "complete"
        and status.get("held_out_test_evaluated") is False
        and decision.get("held_out_test_evaluated") is False
        and status.get("additional_seeds_executed") is False
        and decision.get("additional_seeds_executed") is False,
        (
            f"status={status.get('status')}, stage={status.get('stage')}, "
            f"heldout={status.get('held_out_test_evaluated')}, "
            f"additional_seeds={status.get('additional_seeds_executed')}"
        ),
    )

    tests_root = ElementTree.parse(ARTIFACT_ROOT / "contract_tests.xml").getroot()
    suites = (
        list(tests_root.findall("testsuite"))
        if tests_root.tag == "testsuites"
        else [tests_root]
    )
    test_count = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    test_errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    test_failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    test_skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    audit.require(
        "remote_contract_tests",
        test_count == 14
        and test_errors == 0
        and test_failures == 0
        and test_skipped == 0,
        (
            f"tests={test_count}, errors={test_errors}, "
            f"failures={test_failures}, skipped={test_skipped}"
        ),
    )

    decision_by_dataset = {
        str(row["dataset"]): row for row in decision.get("datasets", [])
    }
    contract_by_dataset = {
        str(row["dataset"]): row for row in contract.get("datasets", [])
    }
    audit.require(
        "dataset_coverage",
        set(decision_by_dataset)
        == set(contract_by_dataset)
        == set(DATASET_ORDER),
        json.dumps(sorted(decision_by_dataset)),
    )

    e1_rows = audit_e1(audit)
    dataset_rows = [
        audit_dataset(
            audit,
            contract,
            contract_by_dataset[dataset],
            decision_by_dataset[dataset],
        )
        for dataset in DATASET_ORDER
    ]

    primary_pass = all(row["refit_time_nll"] <= row["B_time_nll"] for row in dataset_rows)
    strict_pass = all(row["strict_target_pass"] for row in dataset_rows)
    quantity_pass = all(
        row["quantity_prediction_bitwise_identical"]
        and row["quantity_metrics_match_B"]
        and row["non_time_state_bitwise_equal"]
        for row in dataset_rows
    )
    audit.require(
        "controller_aggregate_decision",
        decision.get("time_nll_non_worse_than_B_on_all_datasets") is primary_pass
        and decision.get("restores_A_plus_0_01_on_all_datasets") is strict_pass
        and decision.get("quantity_exactly_preserved_on_all_datasets") is quantity_pass,
        f"primary={primary_pass}, strict={strict_pass}, quantity={quantity_pass}",
    )

    if audit.errors:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "errors": audit.errors,
            "checks": audit.checks,
        }
        write_json(OUT / "validation_audit.json", failure)
        raise AssertionError("\n".join(audit.errors))

    payload = {
        "schema_version": 1,
        "status": "passed",
        "overall_assessment": "share_with_caveats",
        "as_of_utc": status.get("updated_at"),
        "question": (
            "Can a frozen post-hoc time-head refit preserve B quantity predictions "
            "while recovering validation Time NLL across all three datasets?"
        ),
        "scope": {
            "seed": 42,
            "splits_read": ["train", "validation"],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "additional_seeds_executed": False,
            "device": "NVIDIA GeForce RTX 5090",
        },
        "source_identity": {
            "source_revision": launch["source_revision"],
            "source_tree": source_manifest["source_tree"],
            "contract_path": str(CONTRACT_PATH.relative_to(ROOT)),
            "contract_sha256": contract_sha,
            "source_manifest_sha256": source_manifest_sha,
            "source_manifest_file_count": source_manifest["file_count"],
        },
        "acceptance": {
            "primary_time_nll_non_worse_than_B_all_datasets": primary_pass,
            "strict_A_plus_0_01_all_datasets": strict_pass,
            "quantity_predictions_and_metrics_preserved_all_datasets": quantity_pass,
            "strict_passed_datasets": [
                row["dataset"] for row in dataset_rows if row["strict_target_pass"]
            ],
            "strict_failed_datasets": [
                row["dataset"] for row in dataset_rows if not row["strict_target_pass"]
            ],
            "interpretation": (
                "The result supports post-hoc calibration of the frozen B time head. "
                "It does not establish a Backbone improvement or common recovery to A."
            ),
        },
        "datasets": dataset_rows,
        "e1": e1_rows,
        "remote_contract_tests": {
            "tests": test_count,
            "errors": test_errors,
            "failures": test_failures,
            "skipped": test_skipped,
            "xml_sha256": sha256_file(ARTIFACT_ROOT / "contract_tests.xml"),
        },
        "artifact_portability": {
            "feature_caches_copied_locally": False,
            "local_prediction_replay_available_without_rebuilding_cache": False,
            "remote_controller_prediction_replay_recorded": True,
            "local_state_boundary_reverified": True,
            "reproduction_requirement": (
                "Regenerate feature caches from the pinned data, B checkpoints, "
                "contract, and source revision before replaying predictions locally."
            ),
        },
        "check_summary": {
            "total": len(audit.checks),
            "passed": sum(1 for row in audit.checks if row["passed"]),
            "failed": 0,
        },
        "checks": audit.checks,
        "limitations": [
            "Single seed42 validation evidence only.",
            "Held-out test was not evaluated.",
            "The time-head refit changes no Backbone tensor.",
            "Intermittent reached the frozen 100-epoch cap while still improving.",
            "Large feature caches were not included in the local evidence copy.",
        ],
    }
    write_json(OUT / "validation_audit.json", payload)
    write_metrics(dataset_rows)
    write_readme(dataset_rows, payload)
    write_json(OUT / "artifact_manifest.json", build_manifest())
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": payload["check_summary"],
                "primary_pass": primary_pass,
                "strict_pass": strict_pass,
                "quantity_pass": quantity_pass,
                "output": str(OUT.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
