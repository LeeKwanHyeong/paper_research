#!/usr/bin/env python3
"""Audit the revised validation-only Hard-LMM A-versus-B comparison.

The original frozen contract and A/B/C comparator remain unchanged.  This
auditor accepts the user's later scope reduction: A is the pinned historical
joint-selector reference, B is the completed raw-RMSE-selector run, and C is
excluded from both evaluation and adoption.  The controller's intentional
SIGTERM outcome is preserved rather than rewritten as a successful original
run.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.compare_hard_lmm_quantile_checkpoint_alignment import (
    ALL_STRATA,
    BACKBONE,
    B_VARIANT,
    C_VARIANT,
    DEFAULT_CONTRACT,
    EvidenceError,
    FRESH_ROLE,
    RAW_RMSE_HISTORY_KEY,
    RAW_RMSE_MONITOR,
    SEED,
    _canonical_checkpoint_path,
    _comparison,
    _false,
    _finite,
    _integer,
    _load_csv,
    _load_json,
    _metric_record,
    _require,
    _sha256_file,
    _valid_sha256,
    _validate_a_reference,
    _validate_quantity_contract,
)


EXPECTED_CONTROLLER_ERROR = "InterruptedError('Controller received signal 15')"
STOP_RECORD_RELATIVE_PATH = Path("control/stop_after_instacart_b.json")
SCOPE_DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)
SELECTION_FORMULA = "sqrt(mean((predicted_raw_quantity - raw_quantity)^2))"
GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
HELD_OUT_PART = re.compile(r"held[_-]?out", re.IGNORECASE)
TEST_RESULT_FILE = re.compile(
    r"^(?:model_)?test_(?:summary|metrics)(?:\..+)?$", re.IGNORECASE
)
CRITICAL_SOURCE_FILES = (
    "models/TPPs/CountAwareFactory.py",
    "models/TPPs/CountAwareTPP.py",
    "paper/scripts/count_aware_tpp_backbone/constants.py",
    "paper/scripts/count_aware_tpp_backbone/training.py",
    "paper/scripts/run_count_aware_tpp_backbone_control.py",
    "paper/scripts/run_hard_lmm_quantile_checkpoint_alignment_5090.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Defaults to <controller-root>/seed42_e300.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--stop-record",
        type=Path,
        default=None,
        help="Defaults to <controller-root>/control/stop_after_instacart_b.json.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Defaults to <controller-root>/control/source_manifest.json.",
    )
    return parser.parse_args()


def _iso_timestamp(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is missing")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} is not an ISO-8601 timestamp: {value!r}") from exc
    return value


def _git_revision(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and GIT_SHA1.fullmatch(value) is not None,
        f"{label} must be a full lowercase Git revision",
    )
    return value


def _forbidden_held_out_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    forbidden: list[str] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path)
        if any(HELD_OUT_PART.search(part) for part in relative.parts) or TEST_RESULT_FILE.fullmatch(candidate.name):
            forbidden.append(relative.as_posix())
    return sorted(forbidden)


def _validate_source_manifest(
    *,
    path: Path,
    project_root: Path,
    contract_path: Path,
    source_revision: str,
) -> dict[str, Any]:
    manifest = _load_json(path)
    _require(manifest.get("schema_version") == 1, "Source manifest schema drifted")
    _require(
        manifest.get("source_revision") == source_revision,
        "Source manifest/controller revision mismatch",
    )
    _false(manifest.get("held_out_test_evaluated"), label="source manifest held_out_test_evaluated")
    files = manifest.get("files")
    _require(isinstance(files, dict) and bool(files), "Source manifest file map is missing")

    contract_relative = "paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.json"
    _require(manifest.get("contract") == contract_relative, "Source manifest frozen contract path drifted")
    local_contract_digest = _sha256_file(contract_path)
    _require(
        files.get(contract_relative) == local_contract_digest
        and manifest.get("contract_sha256") == local_contract_digest,
        "Source manifest frozen contract SHA-256 mismatch",
    )
    observed: dict[str, str] = {contract_relative: local_contract_digest}
    for relative in CRITICAL_SOURCE_FILES:
        expected = _valid_sha256(files.get(relative), label=f"source manifest {relative}")
        actual = _sha256_file(project_root / relative)
        _require(actual == expected, f"Source manifest/local critical file mismatch: {relative}")
        observed[relative] = actual
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "source_revision": source_revision,
        "contract_path": contract_relative,
        "contract_sha256": local_contract_digest,
        "critical_file_sha256": observed,
        "held_out_test_evaluated": False,
    }


def _validate_contract(
    contract: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    _require(
        contract.get("contract_id") == "hard_lmm_quantile_checkpoint_alignment_v1",
        "B-only audit requires the unchanged frozen quantile/checkpoint contract",
    )
    datasets = contract.get("datasets")
    _require(isinstance(datasets, list), "Frozen contract has no dataset list")
    _require(
        tuple(row.get("dataset") for row in datasets if isinstance(row, dict))
        == SCOPE_DATASETS,
        "Frozen dataset order or identities drifted",
    )
    scope = contract.get("scope")
    _require(isinstance(scope, dict), "Frozen scope is missing")
    _require(scope.get("screening_seed") == SEED, "Frozen screening seed drifted")
    training_contract = {
        "maximum_epochs": _integer(
            scope.get("maximum_epochs"), label="frozen maximum_epochs"
        ),
        "minimum_epochs": _integer(
            scope.get("minimum_epochs"), label="frozen minimum_epochs"
        ),
        "patience": _integer(scope.get("patience"), label="frozen patience"),
    }
    _require(
        training_contract
        == {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40},
        "Frozen B training horizon or early-stopping contract drifted",
    )
    _require(scope.get("evaluation_scope") == "validation_only", "Frozen scope is not validation-only")
    _false(scope.get("held_out_test_evaluated"), label="frozen held_out_test_evaluated")
    arms = contract.get("arms")
    _require(isinstance(arms, dict), "Frozen arm contract is missing")
    _require(
        arms.get("B_t0_raw_rmse", {}).get("checkpoint_monitor")
        == RAW_RMSE_MONITOR,
        "Frozen B checkpoint monitor drifted",
    )
    execution = contract.get("execution")
    _require(isinstance(execution, dict), "Frozen execution contract is missing")
    _require(execution.get("rtx5090_seed42_e300") is True, "Seed42 e300 was not authorized")
    _require(execution.get("rtx5090_seeds52_and62") is False, "Additional seeds entered the contract")
    _require(execution.get("held_out_test") is False, "Held-out entered the contract")
    return datasets, training_contract


def _validate_local_sources_and_inputs(
    *,
    contract: Mapping[str, Any],
    datasets: list[Mapping[str, Any]],
    project_root: Path,
) -> dict[str, Any]:
    source_hashes = contract.get("reference_evidence", {}).get(
        "target_population_loader_source_sha256"
    )
    _require(isinstance(source_hashes, dict) and bool(source_hashes), "Frozen loader-source hashes are missing")
    observed_source_hashes: dict[str, str] = {}
    for relative, expected in source_hashes.items():
        _require(isinstance(relative, str), "Frozen loader-source path is invalid")
        path = project_root / relative
        observed = _sha256_file(path)
        _require(observed == expected, f"Target-population loader source drifted: {relative}")
        observed_source_hashes[relative] = observed

    input_hashes: dict[str, dict[str, str]] = {}
    for row in datasets:
        dataset = str(row["dataset"])
        observed: dict[str, str] = {}
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            relative = row.get(path_key)
            _require(isinstance(relative, str) and bool(relative), f"{dataset} {path_key} is missing")
            digest = _sha256_file(project_root / relative)
            _require(digest == row.get(digest_key), f"{dataset} {path_key} SHA-256 mismatch")
            observed[path_key] = digest
        input_hashes[dataset] = observed
    return {
        "target_population_loader_source_sha256": observed_source_hashes,
        "dataset_input_sha256": input_hashes,
    }


def _validate_controller_status(
    controller_root: Path,
    datasets: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = controller_root / "status.json"
    status = _load_json(path)
    _require(status.get("status") == "failed", "Original controller status must preserve failed")
    _require(status.get("phase") == "seed42_e300", "Controller did not stop during seed42_e300")
    _require(status.get("current_dataset") == SCOPE_DATASETS[-1], "Controller did not stop on Instacart")
    _require(status.get("stage") == "failed", "Original controller failure stage was not preserved")
    _require(status.get("error") == EXPECTED_CONTROLLER_ERROR, "Controller failure was not the intentional SIGTERM")
    updated_at = _iso_timestamp(status.get("updated_at"), label="controller updated_at")
    _false(status.get("held_out_test_evaluated"), label="controller held_out_test_evaluated")
    _false(status.get("additional_seeds_executed"), label="controller additional_seeds_executed")
    source_revision = _git_revision(status.get("source_revision"), label="controller source_revision")

    cuda = status.get("cuda_contract_tests")
    _require(isinstance(cuda, dict), "Controller CUDA contract evidence is missing")
    _require(_integer(cuda.get("test_count"), label="CUDA test count") > 0, "CUDA suite executed no tests")
    for key in ("failures", "errors", "skipped"):
        _require(_integer(cuda.get(key), label=f"CUDA {key}") == 0, f"CUDA suite has {key}")
    _valid_sha256(cuda.get("xml_sha256"), label="CUDA XML")

    e1 = status.get("completed_e1")
    _require(isinstance(e1, list), "Controller e1 evidence is missing")
    _require(
        [entry.get("dataset") for entry in e1 if isinstance(entry, dict)] == [row["dataset"] for row in datasets],
        "Full-data e1 did not complete for all frozen datasets",
    )
    for entry in e1:
        audit = entry.get("audit")
        _require(isinstance(audit, dict) and audit.get("status") == "passed", "An e1 audit did not pass")
        _require(audit.get("phase") == "e1", "Controller e1 phase identity drifted")
        _require(audit.get("source_revision") == source_revision, "Controller e1 source revision drifted")
        _false(audit.get("held_out_test_evaluated"), label=f"{entry.get('dataset')} e1 held_out")

    completed_e300 = status.get("completed_seed42_e300")
    _require(isinstance(completed_e300, list), "Controller seed42 completion list is missing")
    _require(
        [entry.get("dataset") for entry in completed_e300 if isinstance(entry, dict)]
        == list(SCOPE_DATASETS[:2]),
        "Original controller completion prefix drifted before the Instacart B stop",
    )
    for entry in completed_e300:
        audit = entry.get("audit")
        _require(
            isinstance(audit, dict)
            and audit.get("status") == "passed"
            and audit.get("phase") == "seed42_e300"
            and audit.get("source_revision") == source_revision,
            "A completed seed42 dataset lacks its original controller audit",
        )
        _false(audit.get("held_out_test_evaluated"), label=f"{entry.get('dataset')} e300 held_out")

    evidence = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "status": status["status"],
        "stage": status["stage"],
        "error": status["error"],
        "source_revision": source_revision,
        "updated_at": updated_at,
        "cuda_contract_tests": cuda,
        "completed_full_data_e1": [entry["dataset"] for entry in e1],
        "completed_seed42_e300_before_stop": [entry["dataset"] for entry in completed_e300],
    }
    return status, evidence


def _validate_stop_record(
    *,
    path: Path,
    controller_root: Path,
    controller_status: Mapping[str, Any],
    b_summary_path: Path,
    b_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _load_json(path)
    _require(
        record.get("status") == "controller_sigterm_sent_after_instacart_b",
        "Stop record status drifted",
    )
    _require(_integer(record.get("controller_pid"), label="stop controller_pid") > 0, "Stop controller PID is invalid")
    _require(record.get("signal") == "SIGTERM", "Stop record signal was not SIGTERM")
    _require(isinstance(record.get("reason"), str) and bool(record["reason"].strip()), "Stop reason is missing")
    recorded_at = _iso_timestamp(record.get("recorded_at"), label="stop recorded_at")
    status_updated_at = _iso_timestamp(
        controller_status.get("updated_at"), label="controller updated_at"
    )
    recorded_dt = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    status_dt = datetime.fromisoformat(status_updated_at.replace("Z", "+00:00"))
    _require(
        recorded_dt.tzinfo is not None and status_dt.tzinfo is not None,
        "Stop/controller timestamps must include time zones",
    )
    _require(recorded_dt <= status_dt, "Stop record was written after the final controller status")

    recorded_summary = record.get("b_summary")
    _require(isinstance(recorded_summary, str) and bool(recorded_summary), "Stop record B summary path is missing")
    expected_tail = Path(
        "seed42_e300",
        SCOPE_DATASETS[-1],
        FRESH_ROLE,
        "runs",
        BACKBONE,
        B_VARIANT,
        f"seed_{SEED}",
        "summary.json",
    ).as_posix()
    recorded_path = Path(recorded_summary)
    if recorded_path.is_absolute():
        parts = recorded_path.parts
        try:
            seed_index = parts.index("seed42_e300")
        except ValueError as exc:
            raise EvidenceError("Absolute stop-record B summary is outside the controller evidence tree") from exc
        normalized_summary = controller_root.joinpath(*parts[seed_index:]).resolve()
    else:
        normalized_summary = (controller_root / recorded_path).resolve()
    _require(
        recorded_path.as_posix().endswith(expected_tail)
        and normalized_summary == b_summary_path.resolve(),
        "Stop record points to the wrong B summary",
    )
    summary_digest = _sha256_file(b_summary_path)
    _require(record.get("b_summary_sha256") == summary_digest, "Stop record B summary SHA-256 mismatch")
    _require(
        _integer(record.get("b_best_epoch"), label="stop B best_epoch")
        == _integer(b_summary.get("best_epoch"), label="B summary best_epoch"),
        "Stop record B best epoch mismatch",
    )
    _require(
        _integer(record.get("b_completed_epochs"), label="stop B completed_epochs")
        == _integer(b_summary.get("completed_epochs"), label="B summary completed_epochs"),
        "Stop record B completed epoch mismatch",
    )
    _require(
        math.isclose(
            _finite(record.get("b_raw_rmse"), label="stop B raw RMSE"),
            _finite(b_summary.get("best_val_qty_rmse"), label="B summary raw RMSE"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "Stop record B raw RMSE mismatch",
    )
    _require(path.resolve().is_relative_to(controller_root.resolve()), "Stop record is outside controller evidence root")
    evidence = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "status": record["status"],
        "signal": record["signal"],
        "controller_pid": int(record["controller_pid"]),
        "controller_pid_cross_check": "not_available_in_original_controller_status",
        "recorded_at": record["recorded_at"],
        "b_summary_sha256": summary_digest,
    }
    return record, evidence


def _validate_b_launch(
    *,
    artifact_dir: Path,
    dataset_contract: Mapping[str, Any],
    source_revision: str,
    training_contract: Mapping[str, int],
) -> tuple[dict[str, Any], str]:
    dataset = str(dataset_contract["dataset"])
    forbidden = _forbidden_held_out_files(artifact_dir)
    _require(not forbidden, f"{dataset} contains held-out artifacts: {forbidden}")
    launch = _load_json(artifact_dir / "launch_contract.json")
    expected = {
        "dataset": dataset,
        "model_role": FRESH_ROLE,
        "backbones": [BACKBONE],
        "quantity_variants": [B_VARIANT, C_VARIANT],
        "seeds": [SEED],
        "expected_run_count": 2,
        "epochs": training_contract["maximum_epochs"],
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
        "source_revision": source_revision,
    }
    mismatches = {
        key: {"expected": value, "observed": launch.get(key)}
        for key, value in expected.items()
        if launch.get(key) != value
    }
    _require(not mismatches, f"{dataset} B launch contract mismatch: {mismatches}")
    _require(launch.get("status") in {"running", "complete"}, f"{dataset} B launch status is invalid")
    if launch.get("status") == "complete":
        _require(launch.get("completed_run_count") == 2, f"{dataset} completed launch count drifted")
    _false(launch.get("held_out_test_evaluated"), label=f"{dataset} B launch held_out")
    _require(set(launch.get("split_rows", {})) == {"train", "validation"}, f"{dataset} materialized a non-train/validation split")

    early = launch.get("early_stopping")
    _require(isinstance(early, dict), f"{dataset} B early stopping contract is missing")
    expected_early = {
        "monitor": RAW_RMSE_MONITOR,
        "min_epochs": training_contract["minimum_epochs"],
        "patience": training_contract["patience"],
        "comparison": "earliest_strict_finite_minimum",
        "restore": "best_validation_raw_quantity_rmse",
    }
    _require(
        all(early.get(key) == value for key, value in expected_early.items()),
        f"{dataset} B raw-RMSE selector contract drifted",
    )
    _require(
        launch.get("time_head", {}).get("mode") == "legacy_clamped_rmtpp"
        and _finite(launch.get("time_head", {}).get("time_intercept_limit"), label=f"{dataset} time cap") == 300.0,
        f"{dataset} T0 time-head contract drifted",
    )
    _validate_quantity_contract(launch, dataset_contract, label=f"{dataset} B")
    b_interface = launch.get("interfaces", {}).get(B_VARIANT)
    _require(
        isinstance(b_interface, dict)
        and b_interface.get("quantity_loss") == "mse_on_log1p_quantity",
        f"{dataset} B objective drifted",
    )

    train = launch.get("quantile_adaptive_contract", {}).get("population")
    _require(isinstance(train, dict) and train.get("split") == "train", f"{dataset} train population is missing")
    expected_train = {
        "target_count": dataset_contract["expected_train_targets"],
        "target_identity_sha256": dataset_contract["expected_train_target_identity_sha256"],
        "target_quantity_sha256": dataset_contract["expected_train_target_quantity_sha256"],
    }
    _require(
        all(train.get(key) == value for key, value in expected_train.items()),
        f"{dataset} train target population drifted",
    )
    validation = launch.get("validation_target_population")
    _require(isinstance(validation, dict) and validation.get("split") == "validation", f"{dataset} validation population is missing")
    expected_validation = {
        "target_count": dataset_contract["expected_validation_targets"],
        "target_identity_sha256": dataset_contract["expected_validation_target_identity_sha256"],
        "target_quantity_sha256": dataset_contract["expected_validation_target_quantity_sha256"],
    }
    _require(
        all(validation.get(key) == value for key, value in expected_validation.items()),
        f"{dataset} validation target population drifted",
    )
    return launch, str(validation["target_identity_sha256"])


def _validate_b_run(
    *,
    artifact_dir: Path,
    dataset_contract: Mapping[str, Any],
    source_revision: str,
    training_contract: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )
    import torch

    dataset = str(dataset_contract["dataset"])
    run_dir = artifact_dir / "runs" / BACKBONE / B_VARIANT / f"seed_{SEED}"
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path)
    history_payload = _load_json(run_dir / "history.json")
    history = history_payload.get("history")
    _require(isinstance(history, list) and bool(history), f"{dataset} B history is missing")

    expected_summary = {
        "status": "success",
        "backbone": BACKBONE,
        "variant": B_VARIANT,
        "seed": SEED,
        "epochs": training_contract["maximum_epochs"],
        "checkpoint_monitor": RAW_RMSE_MONITOR,
        "checkpoint_monitor_history_key": RAW_RMSE_HISTORY_KEY,
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection_formula": SELECTION_FORMULA,
        "source_revision": source_revision,
        "source_revision_history": [source_revision],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    _require(not mismatches, f"{dataset} B summary contract mismatch: {mismatches}")
    _false(summary.get("held_out_test_evaluated"), label=f"{dataset} B summary held_out")
    _require(
        isinstance(summary.get("training_device"), str)
        and summary["training_device"].startswith("cuda"),
        f"{dataset} B did not train on CUDA",
    )
    allocated = _integer(summary.get("cuda_peak_memory_allocated_bytes"), label=f"{dataset} B CUDA allocated")
    reserved = _integer(summary.get("cuda_peak_memory_reserved_bytes"), label=f"{dataset} B CUDA reserved")
    _require(allocated > 0 and reserved > 0, f"{dataset} B CUDA memory evidence is empty")

    completed_epochs = _integer(summary.get("completed_epochs"), label=f"{dataset} B completed_epochs")
    _require(completed_epochs == len(history), f"{dataset} B history length disagrees with completed epochs")
    _require(
        [_integer(row.get("epoch"), label=f"{dataset} B history epoch") for row in history]
        == list(range(1, completed_epochs + 1)),
        f"{dataset} B history epochs are incomplete or unordered",
    )
    expected_train_count = _integer(dataset_contract["expected_train_targets"], label=f"{dataset} train count")
    earliest_epoch: int | None = None
    earliest_value = float("inf")
    for row in history:
        _require(isinstance(row, dict), f"{dataset} B history row is invalid")
        _require(
            _integer(row.get("train_event_count"), label=f"{dataset} B train_event_count")
            == expected_train_count,
            f"{dataset} B contains a partial train epoch",
        )
        _require(row.get("train_all_finite") is True, f"{dataset} B train telemetry is non-finite")
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _finite(value, label=f"{dataset} B history {key}")
        value = _finite(row.get(RAW_RMSE_HISTORY_KEY), label=f"{dataset} B history raw RMSE")
        if value < earliest_value:
            earliest_epoch = int(row["epoch"])
            earliest_value = value
    best_epoch = _integer(summary.get("best_epoch"), label=f"{dataset} B best_epoch")
    _require(best_epoch == earliest_epoch, f"{dataset} B did not select the earliest strict raw-RMSE minimum")
    stopped_early = summary.get("stopped_early")
    _require(isinstance(stopped_early, bool), f"{dataset} B stopped_early flag is missing")
    if stopped_early:
        _require(
            completed_epochs >= training_contract["minimum_epochs"],
            f"{dataset} B stopped before the frozen minimum epoch",
        )
        _require(
            completed_epochs - best_epoch >= training_contract["patience"],
            f"{dataset} B stopped before raw-RMSE patience was exhausted",
        )
    else:
        _require(
            completed_epochs == training_contract["maximum_epochs"],
            f"{dataset} B is a truncated non-early-stopped run",
        )
    selected = _finite(summary.get("selected_metric_value"), label=f"{dataset} B selected raw RMSE")
    summary_rmse = _finite(summary.get("best_val_qty_rmse"), label=f"{dataset} B raw RMSE")
    _require(
        math.isclose(selected, earliest_value, rel_tol=1e-12, abs_tol=1e-12),
        f"{dataset} B selected metric disagrees with history",
    )
    _require(
        math.isclose(selected, summary_rmse, rel_tol=1e-10, abs_tol=1e-8),
        f"{dataset} B summary/history raw RMSE is inconsistent",
    )

    expected_count = _integer(dataset_contract["expected_validation_targets"], label=f"{dataset} validation count")
    quantity_rows = summary.get("quantity_rows")
    _require(isinstance(quantity_rows, list), f"{dataset} B summary has no quantity rows")
    metrics = _metric_record(
        summary,
        quantity_rows,
        variant=B_VARIANT,
        expected_count=expected_count,
        label=f"{dataset} B",
    )

    top_summaries = _load_csv(artifact_dir / "run_summaries.csv")
    b_top = [row for row in top_summaries if row.get("backbone") == BACKBONE and row.get("variant") == B_VARIANT and _integer(row.get("seed"), label=f"{dataset} B top seed") == SEED]
    _require(len(b_top) == 1, f"{dataset} expected exactly one top-level B summary row")
    for key in (
        "status",
        "backbone",
        "variant",
        "source_revision",
        "checkpoint_monitor",
        "checkpoint_monitor_history_key",
        "checkpoint_selection",
        "checkpoint_state_sha256",
        "initial_state_sha256",
    ):
        _require(b_top[0].get(key) == summary.get(key), f"{dataset} B run/top summary mismatch for {key}")
    top_quantity = _load_csv(artifact_dir / "quantity_seed_metrics.csv")
    b_quantity = [row for row in top_quantity if row.get("backbone") == BACKBONE and row.get("variant") == B_VARIANT and _integer(row.get("seed"), label=f"{dataset} B quantity seed") == SEED]
    _require(len(b_quantity) == len(ALL_STRATA), f"{dataset} top-level B quantity strata are incomplete")
    top_metrics = _metric_record(
        b_top[0],
        b_quantity,
        variant=B_VARIANT,
        expected_count=expected_count,
        label=f"{dataset} B top-level",
    )
    _require(metrics == top_metrics, f"{dataset} B run/top metrics disagree")

    checkpoint_path = _canonical_checkpoint_path(summary_path, summary)
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_state = checkpoint.get("model_state_dict")
    _require(isinstance(checkpoint_state, dict), f"{dataset} B checkpoint model state is missing")
    state_digest = _valid_sha256(summary.get("checkpoint_state_sha256"), label=f"{dataset} B state")
    _require(canonical_state_dict_sha256(checkpoint_state) == state_digest, f"{dataset} B checkpoint state digest mismatch")
    checkpoint_expected = {
        "backbone": BACKBONE,
        "variant": B_VARIANT,
        "seed": SEED,
        "source_revision": source_revision,
        "source_revision_history": [source_revision],
        "checkpoint_monitor": RAW_RMSE_MONITOR,
        "checkpoint_monitor_history_key": RAW_RMSE_HISTORY_KEY,
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection_formula": SELECTION_FORMULA,
        "best_epoch": best_epoch,
        "selected_metric_value": selected,
        "model_state_sha256": state_digest,
        "initial_state_sha256": summary.get("initial_state_sha256"),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    _require(
        all(checkpoint.get(key) == value for key, value in checkpoint_expected.items()),
        f"{dataset} B checkpoint metadata drifted",
    )
    _require(
        math.isclose(
            _finite(checkpoint.get("selected_metric_value"), label=f"{dataset} B checkpoint selected metric"),
            earliest_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        f"{dataset} B checkpoint/history selected metric mismatch",
    )

    last_path = run_dir / "last_epoch_state.pt"
    last = torch_load_checkpoint(last_path, map_location="cpu")
    last_expected = {
        "checkpoint_type": "epoch_resume",
        "checkpoint_schema_version": 2,
        "epoch": completed_epochs,
        "backbone": BACKBONE,
        "variant": B_VARIANT,
        "seed": SEED,
        "source_revision": source_revision,
        "source_revision_history": [source_revision],
        "checkpoint_monitor": RAW_RMSE_MONITOR,
        "checkpoint_monitor_history_key": RAW_RMSE_HISTORY_KEY,
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection_formula": SELECTION_FORMULA,
        "best_epoch": best_epoch,
        "initial_state_sha256": summary.get("initial_state_sha256"),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    _require(
        all(last.get(key) == value for key, value in last_expected.items()),
        f"{dataset} B last-state metadata drifted",
    )
    _require(last.get("history") == history, f"{dataset} B last-state history drifted")
    _require(
        math.isclose(
            _finite(last.get("best_selection_value"), label=f"{dataset} B last-state selected metric"),
            earliest_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        f"{dataset} B last-state/history selected metric mismatch",
    )
    _require(last.get("best_state_sha256") == state_digest, f"{dataset} B last-state best digest drifted")
    _require(
        isinstance(last.get("best_state_dict"), dict)
        and canonical_state_dict_sha256(last["best_state_dict"]) == state_digest,
        f"{dataset} B last-state best weights drifted",
    )
    current_state = last.get("model_state_dict")
    _require(isinstance(current_state, dict), f"{dataset} B last-state current weights are missing")
    current_state_digest = _valid_sha256(
        last.get("model_state_sha256"), label=f"{dataset} B last-state current state"
    )
    _require(
        canonical_state_dict_sha256(current_state) == current_state_digest,
        f"{dataset} B last-state current state digest mismatch",
    )
    _require(isinstance(last.get("optimizer_state_dict"), dict) and bool(last["optimizer_state_dict"].get("state")), f"{dataset} B optimizer resume state is empty")
    _require(isinstance(last.get("rng_state"), dict), f"{dataset} B RNG resume state is missing")
    _require(isinstance(last.get("train_loader_generator_state"), torch.Tensor), f"{dataset} B loader RNG state is missing")
    _false(last.get("held_out_test_evaluated"), label=f"{dataset} B last-state held_out")

    evidence = {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
        "history_sha256": _sha256_file(run_dir / "history.json"),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_file_sha256": _sha256_file(checkpoint_path),
        "checkpoint_state_sha256": state_digest,
        "last_state_path": str(last_path.resolve()),
        "last_state_file_sha256": _sha256_file(last_path),
        "last_state_current_model_sha256": current_state_digest,
        "initial_state_sha256": _valid_sha256(summary.get("initial_state_sha256"), label=f"{dataset} B initial state"),
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "earliest_raw_rmse": earliest_value,
        "training_device": summary["training_device"],
        "cuda_peak_memory_allocated_bytes": allocated,
        "cuda_peak_memory_reserved_bytes": reserved,
        "train_target_count_per_epoch": expected_train_count,
        "validation_target_count": expected_count,
    }
    return metrics, evidence, summary_path


def _c_presence(artifact_dir: Path) -> dict[str, Any]:
    run_dir = artifact_dir / "runs" / BACKBONE / C_VARIANT / f"seed_{SEED}"
    tracked = {
        "summary": run_dir / "summary.json",
        "history": run_dir / "history.json",
        "checkpoint": run_dir / "best_val_qty_rmse_model.pt",
        "last_state": run_dir / "last_epoch_state.pt",
    }
    present = {name: path.is_file() for name, path in tracked.items()}
    any_present = any(present.values()) or run_dir.exists()
    completed = False
    if present["summary"]:
        try:
            completed = _load_json(tracked["summary"]).get("status") == "success"
        except EvidenceError:
            completed = False
    if completed:
        state = "complete_present_excluded"
    elif any_present:
        state = "partial_present_excluded"
    else:
        state = "absent_excluded"
    return {
        "state": state,
        "present": any_present,
        "complete": completed,
        "partial": any_present and not completed,
        "files_present": present,
        "included_in_revised_comparison": False,
        "adoption_included": False,
    }


def audit_b_only(
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    project_root: Path,
    controller_root: Path,
    artifact_root: Path,
    stop_record_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    _require(
        _load_json(contract_path) == contract,
        "In-memory contract differs from the hashed frozen contract file",
    )
    datasets, training_contract = _validate_contract(contract)
    source_input_evidence = _validate_local_sources_and_inputs(
        contract=contract,
        datasets=datasets,
        project_root=project_root,
    )
    controller_status, controller_evidence = _validate_controller_status(
        controller_root, datasets
    )
    source_revision = str(controller_evidence["source_revision"])
    source_manifest_evidence = _validate_source_manifest(
        path=source_manifest_path,
        project_root=project_root,
        contract_path=contract_path,
        source_revision=source_revision,
    )

    dataset_results: list[dict[str, Any]] = []
    c_presence: dict[str, Any] = {}
    final_b_summary_path: Path | None = None
    final_b_summary: dict[str, Any] | None = None
    for dataset_contract in datasets:
        dataset = str(dataset_contract["dataset"])
        artifact_dir = artifact_root / dataset / FRESH_ROLE
        a_metrics, a_evidence = _validate_a_reference(project_root, dataset_contract)
        launch, validation_identity = _validate_b_launch(
            artifact_dir=artifact_dir,
            dataset_contract=dataset_contract,
            source_revision=source_revision,
            training_contract=training_contract,
        )
        b_metrics, b_evidence, b_summary_path = _validate_b_run(
            artifact_dir=artifact_dir,
            dataset_contract=dataset_contract,
            source_revision=source_revision,
            training_contract=training_contract,
        )
        identities_equal = (
            validation_identity
            == dataset_contract["expected_validation_target_identity_sha256"]
            == a_evidence["validation_target_identity_sha256"]
        )
        _require(identities_equal, f"{dataset} A/B validation target identity is not bound")
        counts_equal = a_metrics["validation_target_count"] == b_metrics["validation_target_count"]
        _require(counts_equal, f"{dataset} A/B validation target counts differ")
        raw_improved = float(b_metrics["raw_rmse"]) < float(a_metrics["raw_rmse"])
        dataset_results.append({
            "dataset": dataset,
            "raw_rmse_improved": raw_improved,
            "metrics": {
                "A_t0_joint": a_metrics,
                "B_t0_raw_rmse": b_metrics,
            },
            "effect_A_to_B": _comparison(b_metrics, a_metrics),
            "validation_identity": {
                "target_identity_sha256": validation_identity,
                "identities_equal": identities_equal,
                "counts_equal": counts_equal,
            },
            "evidence": {
                "A_t0_joint": a_evidence,
                "B_t0_raw_rmse": b_evidence,
                "launch_contract_sha256": _sha256_file(artifact_dir / "launch_contract.json"),
                "launch_status": launch["status"],
            },
        })
        c_presence[dataset] = _c_presence(artifact_dir)
        if dataset == SCOPE_DATASETS[-1]:
            final_b_summary_path = b_summary_path
            final_b_summary = _load_json(b_summary_path)

    _require(final_b_summary_path is not None and final_b_summary is not None, "Instacart B summary is missing")
    _, stop_evidence = _validate_stop_record(
        path=stop_record_path,
        controller_root=controller_root,
        controller_status=controller_status,
        b_summary_path=final_b_summary_path,
        b_summary=final_b_summary,
    )

    common_goal = all(row["raw_rmse_improved"] for row in dataset_results)
    metrics_payload = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "comparison": "A_t0_joint_vs_B_t0_raw_rmse",
        "status": "common_raw_rmse_goal_met" if common_goal else "common_raw_rmse_goal_not_met",
        "common_raw_rmse_goal_met": common_goal,
        "datasets": dataset_results,
        "decision_scope": {
            "B_raw_rmse_improvement_vs_A_required_on_every_dataset": True,
            "body_tail_time_are_descriptive": True,
            "frozen_C_gate_reinterpreted": False,
            "C_included_in_revised_comparison": False,
            "C_adopted": False,
        },
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    run_audit = {
        "schema_version": 1,
        "status": "passed",
        "contract_id": contract["contract_id"],
        "contract_sha256": source_manifest_evidence["contract_sha256"],
        "source_revision": source_revision,
        "source_manifest": source_manifest_evidence,
        "source_and_input_evidence": source_input_evidence,
        "controller_status": controller_evidence,
        "intentional_stop_record": stop_evidence,
        "dataset_evidence": {
            row["dataset"]: row["evidence"] for row in dataset_results
        },
        "all_A_and_B_datasets_audited": len(dataset_results) == len(SCOPE_DATASETS),
        "all_B_runs_cuda_full_train_and_validation": True,
        "all_B_checkpoints_are_earliest_validation_raw_rmse": True,
        "additional_seeds_executed": False,
        "held_out_test_evaluated": False,
    }
    scope_completion = {
        "schema_version": 1,
        "status": "complete_for_revised_scope",
        "original_controller_outcome": "failed_intentional_sigterm",
        "source_revision": source_revision,
        "user_scope": "A_vs_B_only",
        "C": {
            "included_in_revised_comparison": False,
            "adoption_included": False,
            "presence_by_dataset": c_presence,
        },
        "original_status_sha256": controller_evidence["sha256"],
        "stop_record_sha256": stop_evidence["sha256"],
        "evaluation_scope": "validation_only",
        "additional_seeds_executed": False,
        "held_out_test_evaluated": False,
    }
    return {
        "metrics": metrics_payload,
        "run_audit": run_audit,
        "scope_completion": scope_completion,
    }


def _metrics_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in payload["datasets"]:
        for arm in ("A_t0_joint", "B_t0_raw_rmse"):
            metric = dataset["metrics"][arm]
            rows.append({
                "dataset": dataset["dataset"],
                "arm": arm,
                "raw_rmse": metric["raw_rmse"],
                "body_mae": metric["body_mae"],
                "gt_p99_mae": metric["gt_p99_mae"],
                "time_nll": metric["time_nll"],
                "validation_target_count": metric["validation_target_count"],
                "B_raw_rmse_improved_vs_A": dataset["raw_rmse_improved"] if arm == "B_t0_raw_rmse" else "",
            })
    return rows


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Hard-LMM Raw-RMSE Checkpoint Alignment: A vs B",
        "",
        f"- Common raw-RMSE goal: **{'MET' if payload['common_raw_rmse_goal_met'] else 'NOT MET'}**",
        "- Scope: validation only; held-out test and additional seeds were not evaluated.",
        "- C is excluded from the revised comparison and adoption; already completed or partial artifacts remain recorded.",
        "- Body MAE, >p99 MAE, and time NLL are descriptive for A→B; the frozen C gate was not reinterpreted.",
        "",
        "| Dataset | Arm | Raw RMSE | Body MAE | >p99 MAE | Time NLL | B improves RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for dataset in payload["datasets"]:
        for arm in ("A_t0_joint", "B_t0_raw_rmse"):
            metric = dataset["metrics"][arm]
            improved = "yes" if arm == "B_t0_raw_rmse" and dataset["raw_rmse_improved"] else ("no" if arm == "B_t0_raw_rmse" else "")
            lines.append(
                f"| {dataset['dataset']} | {arm} | {metric['raw_rmse']:.8f} | "
                f"{metric['body_mae']:.8f} | {metric['gt_p99_mae']:.8f} | "
                f"{metric['time_nll']:.8f} | {improved} |"
            )
    lines.extend(["", "## A→B effects", ""])
    for dataset in payload["datasets"]:
        effect = dataset["effect_A_to_B"]
        lines.append(
            f"- `{dataset['dataset']}`: raw RMSE improvement "
            f"{effect['raw_rmse_relative_improvement'] * 100:.4f}%; "
            f"body MAE change {effect['body_mae_relative_regression'] * 100:.4f}%; "
            f">p99 MAE change {effect['gt_p99_mae_relative_regression'] * 100:.4f}%; "
            f"time NLL change {effect['time_nll_absolute_change']:.8f}."
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("a_vs_b_metrics.json", payload["metrics"]),
        ("b_only_run_audit.json", payload["run_audit"]),
        ("b_only_scope_completion.json", payload["scope_completion"]),
    ):
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_dir / "a_vs_b_metrics.md").write_text(
        _markdown(payload["metrics"]), encoding="utf-8"
    )
    rows = _metrics_rows(payload["metrics"])
    with (output_dir / "a_vs_b_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    controller_root = args.controller_root.resolve()
    artifact_root = (
        args.artifact_root.resolve()
        if args.artifact_root is not None
        else controller_root / "seed42_e300"
    )
    stop_record = (
        args.stop_record.resolve()
        if args.stop_record is not None
        else controller_root / STOP_RECORD_RELATIVE_PATH
    )
    source_manifest = (
        args.source_manifest.resolve()
        if args.source_manifest is not None
        else controller_root / "control/source_manifest.json"
    )
    contract_path = args.contract.resolve()
    payload = audit_b_only(
        contract=_load_json(contract_path),
        contract_path=contract_path,
        project_root=args.project_root.resolve(),
        controller_root=controller_root,
        artifact_root=artifact_root,
        stop_record_path=stop_record,
        source_manifest_path=source_manifest,
    )
    write_outputs(args.output_dir.resolve(), payload)
    print(json.dumps(payload["metrics"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
