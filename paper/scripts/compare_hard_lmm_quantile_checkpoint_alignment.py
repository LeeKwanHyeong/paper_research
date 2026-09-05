#!/usr/bin/env python3
"""Audit and compare the frozen Hard-LMM quantile-alignment A/B/C arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.json"
)
FRESH_ROLE = "quantile_checkpoint_alignment"
BACKBONE = "titantpp"
SEED = 42
A_VARIANT = "count_only_log_regression"
B_VARIANT = "count_only_log_regression"
C_VARIANT = "count_only_quantile_adaptive_log_regression"
EXPECTED_VARIANTS = (B_VARIANT, C_VARIANT)
BODY_STRATA = ("le_p50", "p50_p90", "p90_p95")
ALL_STRATA = (*BODY_STRATA, "p95_p99", "gt_p99")
RAW_RMSE_MONITOR = "validation_raw_quantity_rmse"
RAW_RMSE_HISTORY_KEY = "val_qty_rmse"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(ValueError):
    """Raised when an artifact cannot support a scientific comparison."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Invalid JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"Missing CSV artifact: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise EvidenceError(f"Cannot read CSV artifact {path}: {exc}") from exc
    _require(bool(rows), f"CSV artifact is empty: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    _require(path.is_file(), f"Missing hashed artifact: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"Cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: Any, *, label: str) -> str:
    _require(
        isinstance(expected, str) and HEX_SHA256.fullmatch(expected) is not None,
        f"{label} expected SHA-256 is invalid: {expected!r}",
    )
    observed = _sha256_file(path)
    _require(
        observed == expected,
        f"{label} SHA-256 mismatch: expected {expected}, observed {observed}",
    )
    return observed


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not numeric: {value!r}") from exc
    _require(math.isfinite(result), f"{label} must be finite, got {value!r}")
    return result


def _integer(value: Any, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not an integer: {value!r}") from exc
    return result


def _false(value: Any, *, label: str) -> None:
    if isinstance(value, bool):
        observed = value
    elif isinstance(value, str) and value.lower() in {"true", "false"}:
        observed = value.lower() == "true"
    else:
        raise EvidenceError(f"{label} must be a boolean, got {value!r}")
    _require(not observed, f"{label} must be false")


def _valid_sha256(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _exact_row(
    rows: Iterable[Mapping[str, Any]],
    *,
    variant: str,
    stratum: str | None = None,
    label: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("backbone") == BACKBONE
        and row.get("variant") == variant
        and _integer(row.get("seed"), label=f"{label}.seed") == SEED
        and (stratum is None or row.get("stratum") == stratum)
    ]
    suffix = f"/{stratum}" if stratum else ""
    _require(
        len(matches) == 1,
        f"{label} expected exactly one {BACKBONE}/{variant}/seed42{suffix} row; "
        f"found {len(matches)}",
    )
    return matches[0]


def _validate_quantity_contract(
    launch: Mapping[str, Any], dataset_contract: Mapping[str, Any], *, label: str
) -> None:
    quantity = launch.get("quantity_contract")
    _require(isinstance(quantity, dict), f"{label}.quantity_contract is missing")
    _require(
        quantity.get("quantiles") == [0.5, 0.9, 0.95, 0.99],
        f"{label} reporting quantiles drifted",
    )
    boundaries = quantity.get("boundaries")
    _require(
        isinstance(boundaries, list) and len(boundaries) == 4,
        f"{label} reporting boundaries are invalid",
    )
    observed_p95 = _finite(boundaries[2], label=f"{label}.train_p95")
    observed_p99 = _finite(boundaries[3], label=f"{label}.train_p99")
    _require(
        observed_p95
        == _finite(
            dataset_contract["reporting_body_max_train_p95"],
            label="contract.reporting_body_max_train_p95",
        ),
        f"{label} reporting p95 boundary drifted",
    )
    _require(
        observed_p99
        == _finite(
            dataset_contract["reporting_tail_min_train_p99"],
            label="contract.reporting_tail_min_train_p99",
        ),
        f"{label} reporting p99 boundary drifted",
    )
    strata = quantity.get("strata")
    _require(isinstance(strata, list), f"{label} reporting strata are missing")
    _require(
        [row.get("stratum") for row in strata if isinstance(row, dict)]
        == list(ALL_STRATA),
        f"{label} reporting strata or order drifted",
    )
    _require(
        [row.get("stratum_order") for row in strata if isinstance(row, dict)]
        == list(range(5)),
        f"{label} reporting stratum indices drifted",
    )


def _metric_record(
    summary: Mapping[str, Any],
    quantity_rows: list[Mapping[str, Any]],
    *,
    variant: str,
    expected_count: int,
    label: str,
) -> dict[str, Any]:
    selected = [
        _exact_row(
            quantity_rows,
            variant=variant,
            stratum=stratum,
            label=label,
        )
        for stratum in ALL_STRATA
    ]
    _require(
        len(
            [
                row
                for row in quantity_rows
                if row.get("backbone") == BACKBONE
                and row.get("variant") == variant
                and _integer(row.get("seed"), label=f"{label}.seed") == SEED
            ]
        )
        == len(ALL_STRATA),
        f"{label} contains duplicate or unknown reporting strata",
    )
    counts = [_integer(row.get("count"), label=f"{label}.{name}.count") for row, name in zip(selected, ALL_STRATA)]
    _require(all(value >= 0 for value in counts), f"{label} stratum counts must be nonnegative")
    _require(sum(counts) == expected_count, f"{label} validation target count mismatch: expected {expected_count}, observed {sum(counts)}")
    _require(sum(counts[:3]) > 0, f"{label} body population is empty")
    _require(counts[-1] > 0, f"{label} >p99 population is empty")
    row_maes = [
        _finite(row.get("qty_mae"), label=f"{label}.{name}.qty_mae")
        for row, name in zip(selected, ALL_STRATA)
    ]
    row_rmses = [
        _finite(row.get("qty_rmse"), label=f"{label}.{name}.qty_rmse")
        for row, name in zip(selected, ALL_STRATA)
    ]
    row_time = [
        _finite(row.get("time_nll"), label=f"{label}.{name}.time_nll")
        for row, name in zip(selected, ALL_STRATA)
    ]
    _require(all(value >= 0.0 for value in row_maes), f"{label} MAEs must be nonnegative")
    _require(all(value >= 0.0 for value in row_rmses), f"{label} RMSEs must be nonnegative")
    body_count = sum(counts[:3])
    body_mae = sum(count * value for count, value in zip(counts[:3], row_maes[:3])) / body_count
    reconstructed_rmse = math.sqrt(
        sum(count * value * value for count, value in zip(counts, row_rmses))
        / expected_count
    )
    reconstructed_time = (
        sum(count * value for count, value in zip(counts, row_time)) / expected_count
    )
    summary_rmse = _finite(summary.get("best_val_qty_rmse"), label=f"{label}.best_val_qty_rmse")
    summary_time = _finite(summary.get("best_val_time_nll"), label=f"{label}.best_val_time_nll")
    summary_mae = _finite(summary.get("best_val_qty_mae"), label=f"{label}.best_val_qty_mae")
    _require(summary_rmse >= 0.0 and summary_mae >= 0.0, f"{label} overall quantity metrics must be nonnegative")
    _require(
        math.isclose(summary_rmse, reconstructed_rmse, rel_tol=1e-10, abs_tol=1e-10),
        f"{label} overall raw RMSE disagrees with reporting strata",
    )
    _require(
        math.isclose(summary_time, reconstructed_time, rel_tol=1e-10, abs_tol=1e-10),
        f"{label} time NLL disagrees with reporting strata",
    )
    for row, count, name in zip(selected, counts, ALL_STRATA):
        if "share" in row and row.get("share") not in (None, ""):
            share = _finite(row.get("share"), label=f"{label}.{name}.share")
            _require(
                math.isclose(share, count / expected_count, rel_tol=1e-10, abs_tol=1e-10),
                f"{label}.{name} share/count mismatch",
            )
    result = {
        "raw_rmse": summary_rmse,
        "overall_mae": summary_mae,
        "body_mae": body_mae,
        "gt_p99_mae": row_maes[-1],
        "time_nll": summary_time,
        "validation_target_count": expected_count,
        "body_target_count": body_count,
        "gt_p99_target_count": counts[-1],
        "stratum_counts": dict(zip(ALL_STRATA, counts)),
    }
    _require(
        all(
            math.isfinite(float(result[key]))
            for key in ("raw_rmse", "overall_mae", "body_mae", "gt_p99_mae", "time_nll")
        ),
        f"{label} contains non-finite aggregate metrics",
    )
    return result


def _relative_change(candidate: float, reference: float, *, label: str) -> float:
    _require(reference > 0.0, f"{label} requires a positive reference")
    return candidate / reference - 1.0


def _comparison(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    return {
        "raw_rmse_relative_change": _relative_change(
            float(candidate["raw_rmse"]), float(reference["raw_rmse"]), label="raw RMSE comparison"
        ),
        "raw_rmse_relative_improvement": -_relative_change(
            float(candidate["raw_rmse"]), float(reference["raw_rmse"]), label="raw RMSE comparison"
        ),
        "body_mae_relative_regression": _relative_change(
            float(candidate["body_mae"]), float(reference["body_mae"]), label="body MAE comparison"
        ),
        "gt_p99_mae_relative_regression": _relative_change(
            float(candidate["gt_p99_mae"]), float(reference["gt_p99_mae"]), label=">p99 MAE comparison"
        ),
        "time_nll_absolute_change": float(candidate["time_nll"])
        - float(reference["time_nll"]),
    }


def evaluate_gate(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    c: Mapping[str, Any],
    *,
    identities_equal: bool,
    counts_equal: bool,
    a_target_contract_bound: bool,
    gate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    body_limit = _finite(
        gate_contract["candidate_body_mae_regression_vs_aligned_t0_max"],
        label="body guardrail",
    )
    tail_limit = _finite(
        gate_contract["candidate_gt_p99_mae_regression_vs_aligned_t0_max"],
        label="tail guardrail",
    )
    time_b_limit = _finite(
        gate_contract["candidate_time_loss_increase_vs_aligned_t0_max"],
        label="time-vs-B guardrail",
    )
    time_a_limit = _finite(
        gate_contract["candidate_time_loss_increase_vs_joint_t0_max"],
        label="time-vs-A guardrail",
    )
    body_regression = _relative_change(
        float(c["body_mae"]), float(b["body_mae"]), label="body guardrail"
    )
    tail_regression = _relative_change(
        float(c["gt_p99_mae"]), float(b["gt_p99_mae"]), label="tail guardrail"
    )
    time_vs_b = float(c["time_nll"]) - float(b["time_nll"])
    time_vs_a = float(c["time_nll"]) - float(a["time_nll"])
    checks = {
        "candidate_raw_rmse_strictly_lower_than_aligned_t0": (
            float(c["raw_rmse"]) < float(b["raw_rmse"])
        ),
        "candidate_raw_rmse_strictly_lower_than_joint_t0": (
            float(c["raw_rmse"]) < float(a["raw_rmse"])
        ),
        "candidate_body_mae_regression_vs_aligned_t0_at_most_2pct": (
            body_regression <= body_limit
        ),
        "candidate_gt_p99_mae_regression_vs_aligned_t0_at_most_2pct": (
            tail_regression <= tail_limit
        ),
        "candidate_time_nll_increase_vs_aligned_t0_at_most_0_01": (
            time_vs_b <= time_b_limit
        ),
        "candidate_time_nll_increase_vs_joint_t0_at_most_0_01": (
            time_vs_a <= time_a_limit
        ),
        "validation_target_identity_digests_equal": bool(identities_equal),
        "validation_target_counts_and_strata_equal": bool(counts_equal),
        "joint_t0_validation_target_contract_bound": bool(a_target_contract_bound),
    }
    return {
        "status": "passed" if all(checks.values()) else "rejected",
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "body_mae_relative_regression_vs_aligned_t0": body_regression,
            "gt_p99_mae_relative_regression_vs_aligned_t0": tail_regression,
            "time_nll_absolute_increase_vs_aligned_t0": time_vs_b,
            "time_nll_absolute_increase_vs_joint_t0": time_vs_a,
        },
        "limits": {
            "body_mae_relative_regression": body_limit,
            "gt_p99_mae_relative_regression": tail_limit,
            "time_nll_absolute_increase_vs_aligned_t0": time_b_limit,
            "time_nll_absolute_increase_vs_joint_t0": time_a_limit,
        },
    }


def _canonical_checkpoint_path(summary_path: Path, summary: Mapping[str, Any]) -> Path:
    raw = summary.get("checkpoint_path")
    _require(isinstance(raw, str) and bool(raw), f"{summary_path} has no checkpoint_path")
    name = Path(raw).name
    _require(bool(name), f"{summary_path} checkpoint_path is invalid")
    return summary_path.parent / name


def _validate_a_reference(
    project_root: Path, dataset_contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = str(dataset_contract["dataset"])
    reference = dataset_contract.get("joint_t0_reference")
    _require(isinstance(reference, dict), f"{dataset} has no pinned A reference")
    artifact_raw = reference.get("artifact_path")
    _require(isinstance(artifact_raw, str) and bool(artifact_raw), f"{dataset} A artifact path is invalid")
    artifact_dir = (project_root / artifact_raw).resolve()
    launch_path = artifact_dir / "launch_contract.json"
    summary_path = (
        artifact_dir
        / "runs"
        / BACKBONE
        / A_VARIANT
        / f"seed_{SEED}"
        / "summary.json"
    )
    launch_digest = _verify_sha256(
        launch_path,
        reference.get("launch_contract_sha256"),
        label=f"{dataset} A launch",
    )
    summary_digest = _verify_sha256(
        summary_path,
        reference.get("summary_sha256"),
        label=f"{dataset} A summary",
    )
    launch = _load_json(launch_path)
    summary = _load_json(summary_path)
    checkpoint_path = _canonical_checkpoint_path(summary_path, summary)
    checkpoint_digest = _verify_sha256(
        checkpoint_path,
        reference.get("checkpoint_file_sha256"),
        label=f"{dataset} A checkpoint",
    )
    state_digest = _valid_sha256(
        summary.get("checkpoint_state_sha256"), label=f"{dataset} A checkpoint state"
    )
    _require(
        state_digest == reference.get("checkpoint_state_sha256"),
        f"{dataset} A checkpoint state digest mismatch",
    )
    _require(launch.get("status") == "complete", f"{dataset} A launch is incomplete")
    _require(launch.get("dataset") == dataset, f"{dataset} A launch dataset drifted")
    _require(launch.get("evaluation_scope") == "validation_only", f"{dataset} A is not validation-only")
    _false(launch.get("held_out_test_evaluated"), label=f"{dataset} A held_out_test_evaluated")
    a_target_expected = {
        "source_revision": reference.get("source_revision"),
        "data_sha256": dataset_contract["data_sha256"],
        "split_manifest_sha256": dataset_contract["split_manifest_sha256"],
        "lookback_weeks": dataset_contract["lookback"],
        "max_seq_len": dataset_contract["max_sequence_length"],
    }
    a_target_mismatches = {
        key: {"expected": value, "observed": launch.get(key)}
        for key, value in a_target_expected.items()
        if launch.get(key) != value
    }
    _require(
        not a_target_mismatches,
        f"{dataset} A target-population contract drifted: {a_target_mismatches}",
    )
    split_rows = launch.get("split_rows")
    _require(
        isinstance(split_rows, dict)
        and _integer(
            split_rows.get("validation"),
            label=f"{dataset} A validation row count",
        )
        == _integer(
            dataset_contract["expected_validation_targets"],
            label=f"{dataset} expected validation targets",
        ),
        f"{dataset} A validation population count drifted",
    )
    _require(summary.get("status") == "success", f"{dataset} A summary is not successful")
    _require(summary.get("backbone") == BACKBONE, f"{dataset} A backbone drifted")
    _require(summary.get("variant") == A_VARIANT, f"{dataset} A variant drifted")
    _require(_integer(summary.get("seed"), label=f"{dataset} A seed") == SEED, f"{dataset} A seed drifted")
    _require(summary.get("evaluation_scope") == "validation_only", f"{dataset} A summary is not validation-only")
    _false(summary.get("held_out_test_evaluated"), label=f"{dataset} A summary held_out_test_evaluated")
    _validate_quantity_contract(launch, dataset_contract, label=f"{dataset} A")
    quantity_rows = summary.get("quantity_rows")
    _require(isinstance(quantity_rows, list), f"{dataset} A summary has no quantity rows")
    expected_count = _integer(
        dataset_contract["expected_validation_targets"], label=f"{dataset} expected validation targets"
    )
    metrics = _metric_record(
        summary,
        quantity_rows,
        variant=A_VARIANT,
        expected_count=expected_count,
        label=f"{dataset} A",
    )
    evidence = {
        "artifact_path": str(artifact_dir),
        "launch_contract_sha256": launch_digest,
        "summary_sha256": summary_digest,
        "checkpoint_file_sha256": checkpoint_digest,
        "checkpoint_state_sha256": state_digest,
        "source_revision": launch["source_revision"],
        "validation_target_identity_sha256": dataset_contract[
            "expected_validation_target_identity_sha256"
        ],
        "validation_target_identity_basis": (
            "pinned data/split/lookback/max_seq_len plus audited identical target-loader source"
        ),
        "target_contract_bound": True,
    }
    return metrics, evidence


def _validate_fresh_launch(
    artifact_dir: Path,
    dataset_contract: Mapping[str, Any],
    objective_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    dataset = str(dataset_contract["dataset"])
    forbidden = [
        path
        for path in artifact_dir.rglob("*")
        if path.is_file()
        and (
            "held_out" in path.name.lower()
            or path.name in {"test_summary.json", "test_metrics.csv"}
        )
    ]
    _require(not forbidden, f"{dataset} contains forbidden held-out artifacts: {forbidden}")
    launch = _load_json(artifact_dir / "launch_contract.json")
    expected = {
        "status": "complete",
        "dataset": dataset,
        "model_role": FRESH_ROLE,
        "backbones": [BACKBONE],
        "quantity_variants": list(EXPECTED_VARIANTS),
        "seeds": [SEED],
        "expected_run_count": 2,
        "completed_run_count": 2,
        "epochs": 300,
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": dataset_contract["lookback"],
        "max_seq_len": dataset_contract["max_sequence_length"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "partial_smoke": False,
        "max_series": None,
        "data_sha256": dataset_contract["data_sha256"],
        "split_manifest_sha256": dataset_contract["split_manifest_sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": launch.get(key)}
        for key, value in expected.items()
        if launch.get(key) != value
    }
    _require(not mismatches, f"{dataset} fresh launch contract mismatch: {mismatches}")
    _false(launch.get("held_out_test_evaluated"), label=f"{dataset} fresh held_out_test_evaluated")
    source_revision = launch.get("source_revision")
    _require(
        isinstance(source_revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", source_revision) is not None,
        f"{dataset} fresh source revision is invalid",
    )
    early = launch.get("early_stopping")
    _require(isinstance(early, dict), f"{dataset} early-stopping contract is missing")
    _require(early.get("monitor") == RAW_RMSE_MONITOR, f"{dataset} checkpoint monitor drifted")
    _require(early.get("min_epochs") == 40, f"{dataset} minimum epoch contract drifted")
    _require(early.get("patience") == 40, f"{dataset} patience contract drifted")
    _require(
        early.get("restore") == "best_validation_raw_quantity_rmse",
        f"{dataset} restore selector drifted",
    )
    time_head = launch.get("time_head")
    _require(isinstance(time_head, dict), f"{dataset} time-head contract is missing")
    _require(
        time_head.get("mode") == "legacy_clamped_rmtpp",
        f"{dataset} time-head mode drifted",
    )
    _require(
        _finite(
            time_head.get("time_intercept_limit"),
            label=f"{dataset} time intercept limit",
        )
        == 300.0,
        f"{dataset} executed legacy time intercept cap drifted",
    )
    _validate_quantity_contract(launch, dataset_contract, label=f"{dataset} fresh")
    interfaces = launch.get("interfaces")
    _require(isinstance(interfaces, dict), f"{dataset} quantity interfaces are missing")
    _require(
        set(interfaces) == set(EXPECTED_VARIANTS),
        f"{dataset} quantity interface variants drifted",
    )
    b_interface = interfaces.get(B_VARIANT)
    c_interface = interfaces.get(C_VARIANT)
    _require(isinstance(b_interface, dict), f"{dataset} B interface is invalid")
    _require(isinstance(c_interface, dict), f"{dataset} C interface is invalid")
    _require(
        b_interface.get("quantity_loss") == "mse_on_log1p_quantity",
        f"{dataset} B quantity objective drifted",
    )
    _require(
        c_interface.get("quantity_loss")
        == "train_quantile_weighted_mse_on_log1p_quantity",
        f"{dataset} C quantity objective drifted",
    )
    _require(
        _finite(
            c_interface.get("quantile_adaptive_strength"),
            label=f"{dataset} adaptive strength",
        )
        == _finite(objective_contract.get("strength"), label="frozen adaptive strength"),
        f"{dataset} adaptive strength drifted",
    )
    adaptive = launch.get("quantile_adaptive_contract")
    _require(isinstance(adaptive, dict), f"{dataset} adaptive objective contract is missing")
    _require(
        c_interface.get("quantile_adaptive_contract") == adaptive,
        f"{dataset} C interface/adaptive contract mismatch",
    )
    _require(
        adaptive.get("statistics_source_split") == "train",
        f"{dataset} adaptive statistics are not train-only",
    )
    _require(
        adaptive.get("statistics_population")
        == "exact_canonical_next_event_targets",
        f"{dataset} adaptive target population drifted",
    )
    _require(
        adaptive.get("quantiles") == objective_contract.get("quantiles"),
        f"{dataset} adaptive quantiles drifted",
    )
    _require(
        adaptive.get("raw_bin_weights") == objective_contract.get("raw_bin_weights"),
        f"{dataset} adaptive raw weights drifted",
    )
    _require(
        adaptive.get("interpolation") == "nearest"
        and adaptive.get("boundary_semantics") == "equality_in_lower_bin",
        f"{dataset} adaptive quantile boundary semantics drifted",
    )
    _require(
        adaptive.get("normalization")
        == "raw_weight_divided_by_float64_train_target_mean",
        f"{dataset} adaptive normalization drifted",
    )
    _require(
        adaptive.get("reduction")
        == "target_event_mean_without_batch_renormalization",
        f"{dataset} adaptive reduction drifted",
    )
    train_population = adaptive.get("population")
    _require(
        isinstance(train_population, dict)
        and train_population.get("split") == "train",
        f"{dataset} adaptive train population binding is invalid",
    )
    _require(
        _integer(
            train_population.get("target_count"),
            label=f"{dataset} adaptive train target count",
        )
        == _integer(
            dataset_contract["expected_train_targets"],
            label=f"{dataset} expected train targets",
        ),
        f"{dataset} adaptive train target count drifted",
    )
    expected_adaptive = {
        "boundaries": dataset_contract["expected_loss_boundaries"],
        "bin_counts": dataset_contract["expected_loss_bin_counts"],
        "contract_sha256": dataset_contract["expected_loss_contract_sha256"],
    }
    adaptive_mismatches = {
        key: {"expected": value, "observed": adaptive.get(key)}
        for key, value in expected_adaptive.items()
        if adaptive.get(key) != value
    }
    _require(
        not adaptive_mismatches,
        f"{dataset} pinned adaptive statistics drifted: {adaptive_mismatches}",
    )
    _require(
        math.isclose(
            _finite(
                adaptive.get("normalization_mean"),
                label=f"{dataset} adaptive normalization mean",
            ),
            _finite(
                dataset_contract["expected_loss_normalization_mean"],
                label=f"{dataset} expected adaptive normalization mean",
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{dataset} adaptive normalization mean drifted",
    )
    _require(
        math.isclose(
            _finite(
                adaptive.get("normalized_train_target_mean"),
                label=f"{dataset} normalized train-target weight mean",
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=5e-16,
        ),
        f"{dataset} normalized train-target weights do not average to one",
    )
    _require(
        train_population.get("target_identity_sha256")
        == dataset_contract["expected_train_target_identity_sha256"],
        f"{dataset} adaptive train target identity drifted",
    )
    _require(
        train_population.get("target_quantity_sha256")
        == dataset_contract["expected_train_target_quantity_sha256"],
        f"{dataset} adaptive train target quantities drifted",
    )
    validation = launch.get("validation_target_population")
    _require(isinstance(validation, dict), f"{dataset} validation target population is missing")
    expected_count = _integer(
        dataset_contract["expected_validation_targets"], label=f"{dataset} expected validation targets"
    )
    _require(validation.get("split") == "validation", f"{dataset} validation population split drifted")
    _require(
        _integer(validation.get("target_count"), label=f"{dataset} validation target count")
        == expected_count,
        f"{dataset} validation population count drifted",
    )
    shared_digest = _valid_sha256(
        validation.get("target_identity_sha256"), label=f"{dataset} validation identity"
    )
    _require(
        shared_digest == dataset_contract["expected_validation_target_identity_sha256"],
        f"{dataset} validation target identity drifted",
    )
    _require(
        validation.get("target_quantity_sha256")
        == dataset_contract["expected_validation_target_quantity_sha256"],
        f"{dataset} validation target quantities drifted",
    )
    bindings = launch.get("validation_target_population_by_variant")
    if bindings is None:
        variant_digests = {variant: shared_digest for variant in EXPECTED_VARIANTS}
    else:
        _require(isinstance(bindings, dict), f"{dataset} per-variant target bindings are invalid")
        _require(set(bindings) == set(EXPECTED_VARIANTS), f"{dataset} per-variant target bindings are incomplete")
        variant_digests = {}
        for variant in EXPECTED_VARIANTS:
            binding = bindings[variant]
            _require(isinstance(binding, dict), f"{dataset}/{variant} target binding is invalid")
            _require(
                _integer(binding.get("target_count"), label=f"{dataset}/{variant} target count")
                == expected_count,
                f"{dataset}/{variant} target count drifted",
            )
            variant_digests[variant] = _valid_sha256(
                binding.get("target_identity_sha256"),
                label=f"{dataset}/{variant} validation identity",
            )
            _require(
                variant_digests[variant] == shared_digest,
                f"{dataset}/{variant} target binding disagrees with shared validation population",
            )
    return launch, variant_digests


def _validate_fresh_runs(
    artifact_dir: Path,
    dataset_contract: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    dataset = str(dataset_contract["dataset"])
    summaries = _load_csv(artifact_dir / "run_summaries.csv")
    _require(len(summaries) == 2, f"{dataset} expected exactly two fresh summary rows, found {len(summaries)}")
    quantity_rows = _load_csv(artifact_dir / "quantity_seed_metrics.csv")
    _require(len(quantity_rows) == 10, f"{dataset} expected exactly ten fresh quantity rows, found {len(quantity_rows)}")
    expected_count = _integer(
        dataset_contract["expected_validation_targets"], label=f"{dataset} expected validation targets"
    )
    source_revision = str(launch["source_revision"])
    metrics: dict[str, dict[str, Any]] = {}
    checkpoint_evidence: dict[str, Any] = {}
    for variant in EXPECTED_VARIANTS:
        row = _exact_row(summaries, variant=variant, label=f"{dataset} fresh summaries")
        _require(row.get("status") == "success", f"{dataset}/{variant} run is not successful")
        _require(row.get("evaluation_scope") == "validation_only", f"{dataset}/{variant} is not validation-only")
        _false(row.get("held_out_test_evaluated"), label=f"{dataset}/{variant} held_out_test_evaluated")
        _require(row.get("source_revision") == source_revision, f"{dataset}/{variant} source revision drifted")
        _require(row.get("checkpoint_monitor") == RAW_RMSE_MONITOR, f"{dataset}/{variant} monitor drifted")
        _require(
            row.get("checkpoint_monitor_history_key") == RAW_RMSE_HISTORY_KEY,
            f"{dataset}/{variant} monitor history key drifted",
        )
        _require(
            row.get("checkpoint_selection") == "best_validation_raw_quantity_rmse",
            f"{dataset}/{variant} checkpoint selection identity drifted",
        )
        _require(
            row.get("selection_formula")
            == "sqrt(mean((predicted_raw_quantity - raw_quantity)^2))",
            f"{dataset}/{variant} selector formula drifted",
        )
        selected = _finite(row.get("selected_metric_value"), label=f"{dataset}/{variant} selected metric")
        observed_rmse = _finite(row.get("best_val_qty_rmse"), label=f"{dataset}/{variant} RMSE")
        _require(
            math.isclose(selected, observed_rmse, rel_tol=1e-12, abs_tol=1e-12),
            f"{dataset}/{variant} selected metric is not raw RMSE",
        )
        run_summary_path = (
            artifact_dir / "runs" / BACKBONE / variant / f"seed_{SEED}" / "summary.json"
        )
        run_summary = _load_json(run_summary_path)
        for key in (
            "status",
            "backbone",
            "variant",
            "seed",
            "source_revision",
            "evaluation_scope",
            "held_out_test_evaluated",
            "checkpoint_monitor",
            "checkpoint_monitor_history_key",
            "checkpoint_selection",
            "checkpoint_state_sha256",
            "initial_state_sha256",
        ):
            observed = run_summary.get(key)
            expected = row.get(key)
            if key == "seed":
                observed = _integer(observed, label=f"{dataset}/{variant} run summary seed")
                expected = _integer(expected, label=f"{dataset}/{variant} summary seed")
            elif key == "held_out_test_evaluated":
                _false(observed, label=f"{dataset}/{variant} run summary held_out_test_evaluated")
                _false(expected, label=f"{dataset}/{variant} summary held_out_test_evaluated")
                continue
            _require(observed == expected, f"{dataset}/{variant} run/top summary mismatch for {key}")
        history_payload = _load_json(run_summary_path.parent / "history.json")
        history = history_payload.get("history")
        _require(
            isinstance(history, list) and bool(history),
            f"{dataset}/{variant} checkpoint history is missing",
        )
        earliest_best_epoch: int | None = None
        earliest_best_value = float("inf")
        for history_row in history:
            _require(
                isinstance(history_row, dict),
                f"{dataset}/{variant} checkpoint history row is invalid",
            )
            value = _finite(
                history_row.get(RAW_RMSE_HISTORY_KEY),
                label=f"{dataset}/{variant} history raw RMSE",
            )
            epoch = _integer(
                history_row.get("epoch"), label=f"{dataset}/{variant} history epoch"
            )
            if value < earliest_best_value:
                earliest_best_value = value
                earliest_best_epoch = epoch
        _require(
            earliest_best_epoch
            == _integer(row.get("best_epoch"), label=f"{dataset}/{variant} best epoch"),
            f"{dataset}/{variant} did not select the earliest strict raw-RMSE minimum",
        )
        _require(
            math.isclose(selected, earliest_best_value, rel_tol=1e-12, abs_tol=1e-12),
            f"{dataset}/{variant} selected metric disagrees with raw-RMSE history",
        )
        run_quantity_rows = run_summary.get("quantity_rows")
        _require(isinstance(run_quantity_rows, list), f"{dataset}/{variant} run summary has no quantity rows")
        top_metrics = _metric_record(
            row,
            quantity_rows,
            variant=variant,
            expected_count=expected_count,
            label=f"{dataset}/{variant} top-level",
        )
        run_metrics = _metric_record(
            run_summary,
            run_quantity_rows,
            variant=variant,
            expected_count=expected_count,
            label=f"{dataset}/{variant} run summary",
        )
        _require(top_metrics == run_metrics, f"{dataset}/{variant} run/top metrics disagree")
        checkpoint_path = _canonical_checkpoint_path(run_summary_path, run_summary)
        checkpoint_file_digest = _sha256_file(checkpoint_path)
        checkpoint_state_digest = _valid_sha256(
            row.get("checkpoint_state_sha256"),
            label=f"{dataset}/{variant} checkpoint state",
        )
        checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
        checkpoint_state = checkpoint.get("model_state_dict")
        _require(
            isinstance(checkpoint_state, dict),
            f"{dataset}/{variant} checkpoint has no model state",
        )
        _require(
            canonical_state_dict_sha256(checkpoint_state) == checkpoint_state_digest,
            f"{dataset}/{variant} checkpoint state digest mismatch",
        )
        for key, expected in (
            ("backbone", BACKBONE),
            ("variant", variant),
            ("seed", SEED),
            ("source_revision", source_revision),
            ("checkpoint_monitor", RAW_RMSE_MONITOR),
            ("checkpoint_monitor_history_key", RAW_RMSE_HISTORY_KEY),
            ("checkpoint_selection", "best_validation_raw_quantity_rmse"),
            ("initial_state_sha256", row.get("initial_state_sha256")),
        ):
            _require(
                checkpoint.get(key) == expected,
                f"{dataset}/{variant} checkpoint metadata drifted for {key}",
            )
        metrics[variant] = top_metrics
        checkpoint_evidence[variant] = {
            "summary_sha256": _sha256_file(run_summary_path),
            "checkpoint_file_sha256": checkpoint_file_digest,
            "checkpoint_state_sha256": checkpoint_state_digest,
            "initial_state_sha256": _valid_sha256(
                row.get("initial_state_sha256"),
                label=f"{dataset}/{variant} initial state",
            ),
        }
    _require(
        set(row.get("variant") for row in summaries) == set(EXPECTED_VARIANTS),
        f"{dataset} fresh summaries contain an unexpected or duplicate variant",
    )
    _require(
        checkpoint_evidence[B_VARIANT]["initial_state_sha256"]
        == checkpoint_evidence[C_VARIANT]["initial_state_sha256"],
        f"{dataset} B/C initial model states differ",
    )
    return metrics, checkpoint_evidence


def assess_dataset(
    *,
    project_root: Path,
    artifact_root: Path,
    dataset_contract: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
    objective_contract: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = str(dataset_contract.get("dataset", "unknown"))
    artifact_dir = artifact_root / dataset / FRESH_ROLE
    try:
        a_metrics, a_evidence = _validate_a_reference(project_root, dataset_contract)
        launch, target_digests = _validate_fresh_launch(
            artifact_dir, dataset_contract, objective_contract
        )
        fresh_metrics, fresh_evidence = _validate_fresh_runs(
            artifact_dir, dataset_contract, launch
        )
        b_metrics = fresh_metrics[B_VARIANT]
        c_metrics = fresh_metrics[C_VARIANT]
        identities_equal = target_digests[B_VARIANT] == target_digests[C_VARIANT]
        counts_equal = (
            b_metrics["validation_target_count"] == c_metrics["validation_target_count"]
            and b_metrics["stratum_counts"] == c_metrics["stratum_counts"]
        )
        gate = evaluate_gate(
            a_metrics,
            b_metrics,
            c_metrics,
            identities_equal=identities_equal,
            counts_equal=counts_equal,
            a_target_contract_bound=bool(a_evidence.get("target_contract_bound")),
            gate_contract=gate_contract,
        )
        return {
            "dataset": dataset,
            "status": gate["status"],
            "source_revision": str(launch["source_revision"]),
            "artifact_path": str(artifact_dir.resolve()),
            "metrics": {
                "A_t0_joint": a_metrics,
                "B_t0_raw_rmse": b_metrics,
                "C_quantile_raw_rmse": c_metrics,
            },
            "effects": {
                "selection_A_to_B": _comparison(b_metrics, a_metrics),
                "loss_B_to_C": _comparison(c_metrics, b_metrics),
                "total_A_to_C": _comparison(c_metrics, a_metrics),
            },
            "validation_identity": {
                "B_t0_raw_rmse": target_digests[B_VARIANT],
                "C_quantile_raw_rmse": target_digests[C_VARIANT],
                "digests_equal": identities_equal,
                "stratum_counts_equal": counts_equal,
            },
            "gate": gate,
            "evidence": {
                "A_t0_joint": a_evidence,
                "fresh_launch_contract_sha256": _sha256_file(
                    artifact_dir / "launch_contract.json"
                ),
                "fresh_runs": fresh_evidence,
            },
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }
    except (EvidenceError, KeyError, TypeError, ZeroDivisionError) as exc:
        return {
            "dataset": dataset,
            "status": "not_evaluable",
            "artifact_path": str(artifact_dir.resolve()),
            "errors": [str(exc)],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }


def compare_all(
    *,
    contract: Mapping[str, Any],
    project_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    datasets = contract.get("datasets")
    _require(isinstance(datasets, list), "Frozen contract has no dataset list")
    _require(len(datasets) == 3, f"Frozen contract must contain exactly three datasets, found {len(datasets)}")
    dataset_names = [row.get("dataset") for row in datasets if isinstance(row, dict)]
    _require(len(dataset_names) == 3 and len(set(dataset_names)) == 3, "Frozen dataset identities are invalid")
    _require(contract.get("scope", {}).get("evaluation_scope") == "validation_only", "Frozen scope is not validation-only")
    _false(contract.get("scope", {}).get("held_out_test_evaluated"), label="frozen held_out_test_evaluated")
    gate_contract = contract.get("seed42_gate", {}).get("required_on_every_dataset")
    _require(isinstance(gate_contract, dict), "Frozen seed42 gate is missing")
    objective_contract = contract.get("quantile_objective")
    _require(isinstance(objective_contract, dict), "Frozen quantile objective is missing")
    results = [
        assess_dataset(
            project_root=project_root,
            artifact_root=artifact_root,
            dataset_contract=dataset,
            gate_contract=gate_contract,
            objective_contract=objective_contract,
        )
        for dataset in datasets
    ]
    fresh_source_revisions = sorted({
        str(result["source_revision"])
        for result in results
        if result["status"] != "not_evaluable"
    })
    if len(fresh_source_revisions) > 1:
        message = (
            "Fresh B/C artifacts mix source revisions across datasets: "
            + ", ".join(fresh_source_revisions)
        )
        for result in results:
            if result["status"] != "not_evaluable":
                result["status"] = "not_evaluable"
                result["errors"] = [message]
    statuses = [result["status"] for result in results]
    if "not_evaluable" in statuses:
        status = "not_evaluable"
        next_action = "repair_or_complete_evidence_before_any_scientific_decision"
    elif all(value == "passed" for value in statuses):
        status = "passed"
        next_action = "seed42_gate_passed_seeds52_and62_remain_a_separate_stage"
    else:
        status = "rejected"
        next_action = "stop_after_seed42_without_retuning_loss_weights_or_selector"
    return {
        "schema_version": 1,
        "contract_id": contract.get("contract_id"),
        "status": status,
        "all_three_datasets_assessed": len(results) == 3,
        "dataset_status_counts": {
            value: statuses.count(value)
            for value in ("passed", "rejected", "not_evaluable")
        },
        "fresh_source_revisions": fresh_source_revisions,
        "datasets": results,
        "decision": {
            "seed42_gate_passed": status == "passed",
            "next_action": next_action,
            "loss_or_selector_retuned_after_results": False,
        },
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def _metrics_csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in payload["datasets"]:
        if result["status"] == "not_evaluable":
            rows.append({
                "dataset": result["dataset"],
                "dataset_status": result["status"],
                "arm": "not_evaluable",
                "raw_rmse": "",
                "body_mae": "",
                "gt_p99_mae": "",
                "time_nll": "",
                "validation_target_count": "",
            })
            continue
        for arm in ("A_t0_joint", "B_t0_raw_rmse", "C_quantile_raw_rmse"):
            metric = result["metrics"][arm]
            rows.append({
                "dataset": result["dataset"],
                "dataset_status": result["status"],
                "arm": arm,
                "raw_rmse": metric["raw_rmse"],
                "body_mae": metric["body_mae"],
                "gt_p99_mae": metric["gt_p99_mae"],
                "time_nll": metric["time_nll"],
                "validation_target_count": metric["validation_target_count"],
            })
    return rows


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Hard-LMM Quantile-Adaptive Loss and Checkpoint Alignment",
        "",
        f"- Seed-42 decision: **{str(payload['status']).upper()}**",
        "- Scope: validation only; held-out test was not evaluated.",
        "- A→B is the checkpoint-selection effect; B→C is the incremental loss effect.",
        "",
        "| Dataset | Status | Arm | Raw RMSE | Body MAE | >p99 MAE | Time NLL |",
        "| --- | :---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for result in payload["datasets"]:
        if result["status"] == "not_evaluable":
            message = "; ".join(result.get("errors", []))
            lines.append(
                f"| {result['dataset']} | not_evaluable | evidence error: {message} |  |  |  |  |"
            )
            continue
        for arm in ("A_t0_joint", "B_t0_raw_rmse", "C_quantile_raw_rmse"):
            metric = result["metrics"][arm]
            lines.append(
                f"| {result['dataset']} | {result['status']} | {arm} | "
                f"{metric['raw_rmse']:.8f} | {metric['body_mae']:.8f} | "
                f"{metric['gt_p99_mae']:.8f} | {metric['time_nll']:.8f} |"
            )
    lines.extend(["", "## Dataset gates", ""])
    for result in payload["datasets"]:
        if result["status"] == "not_evaluable":
            lines.append(f"- `{result['dataset']}`: NOT EVALUABLE")
            continue
        failed = [name for name, passed in result["gate"]["checks"].items() if not passed]
        detail = "all checks passed" if not failed else "failed: " + ", ".join(failed)
        lines.append(f"- `{result['dataset']}`: {result['status'].upper()} — {detail}")
    lines.extend(["", f"Next action: `{payload['decision']['next_action']}`", ""])
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(_markdown(payload), encoding="utf-8")
    rows = _metrics_csv_rows(payload)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    contract = _load_json(args.contract.resolve())
    payload = compare_all(
        contract=contract,
        project_root=args.project_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    write_outputs(args.output_dir.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
