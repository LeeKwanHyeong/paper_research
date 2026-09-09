#!/usr/bin/env python3
"""Strict validation-only audit for VNC-Hard-LMM screening artifacts.

The auditor is deliberately separate from the training runner.  Without
``--write-output`` it only reads the pinned contracts and job artifacts and
prints the audit JSON to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE_CONTRACT_PATH = (
    ROOT / "paper/contracts/hard_lmm_value_norm_screening_5090_v1.json"
)
AMENDMENT_PATH = (
    ROOT
    / "paper/contracts/hard_lmm_value_norm_screening_5090_v1_amendment1.json"
)
BASE_CONTRACT_SHA256 = (
    "515412b7750f8e864872ea37c7a50d4af57a78407a8794604b1e12301b26bc9c"
)
AMENDMENT_SHA256 = (
    "c84f4be29c1f796709d2767470077a56e8b5ee01766bff5e1cde2e76ffc72fe7"
)
PINNED_SOURCE_REVISION = "956603f16ca3540e2012a461a494ea1ec905102d"
BACKBONE = "titantpp_hard_memory_value_norm"
MODEL_ROLE = "hard_lmm_value_norm_candidate"
VARIANT = "count_only_log_regression"
MONITOR = "validation_raw_quantity_rmse"
HISTORY_METRIC = "val_qty_rmse"
SELECTION = "best_validation_raw_quantity_rmse"
ALPHA_KEY = "lmm.alpha_raw"
SUPPORTED_EPOCH_BUDGETS = (1, 300)
PINNED_SOURCE_MANIFEST_SHA256 = (
    "177286e0e405e21885214ee0c6acb0fada158465a1715743f399facfd5a9af9f"
)
PINNED_SOURCE_FILE_COUNT = 2130
PINNED_MANIFEST_CHECK_SHA256 = (
    "48b28284a4281fe78d682d99de639cb2795ddf2f2a3b90650ac4c4b379ee6490"
)
E300_MIDRUN_RECEIPT_SHA256 = (
    "52c788927edcd21af299f3d99e6a616e747c4372c48592c63add7b2cd383463f"
)
E300_LAUNCHER_SHA256 = (
    "a3d5ffa4ae4590706be625d1ec8444b33c01e2a9bdde2acda6e5e81ac16fa568"
)
E300_GPU_PREFLIGHT_SHA256 = (
    "ede3c57e035e5ec4cabf9f5aee9bb965fb6dd07e48f033da6b6aafb53a5ab4b2"
)
E300_HOST_GPU_RECEIPT_SHA256 = (
    "44fad9d9aec473c66a6356c642d6b09ee4f32bfb8608df331daf078e274cd3aa"
)
REEVALUATION_REL_TOL = 1e-10
REEVALUATION_ABS_TOL = 1e-8
CPU_REPLAY_REL_TOL = 1e-8
CPU_REPLAY_ABS_TOL = 1e-5
PROJECT_IMPORT_ROOTS = frozenset({"models", "paper", "simple_lab_test"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_project_import_provenance(
    source_root: Path,
    *,
    phase: str,
) -> dict[str, Any]:
    """Reject cached project modules that did not come from the pinned source."""
    source_root = source_root.resolve()
    observed: dict[str, list[str]] = {}
    for module_name in sorted(sys.modules):
        if module_name.partition(".")[0] not in PROJECT_IMPORT_ROOTS:
            continue
        module = sys.modules[module_name]
        require(module is not None, f"{phase}: cached project module is null: {module_name}")
        locations: set[str] = set()
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            locations.add(str(Path(module_file).resolve()))
        module_paths = getattr(module, "__path__", None)
        if module_paths is not None:
            for module_path in module_paths:
                locations.add(str(Path(module_path).resolve()))
        require(locations, f"{phase}: project module has no auditable path: {module_name}")
        for location in locations:
            require(
                _is_within(Path(location), source_root),
                (
                    f"{phase}: project module resolved outside pinned source root: "
                    f"{module_name} -> {location}"
                ),
            )
        observed[module_name] = sorted(locations)
    return {
        "phase": phase,
        "source_root": str(source_root),
        "module_count": len(observed),
        "modules": observed,
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON object: {path}") from error
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"Cannot hash artifact: {path}") from error
    return digest.hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    return number


def _require_metric_equal(left: Any, right: float, *, label: str) -> None:
    observed = _finite_number(left, label=label)
    require(
        math.isclose(observed, right, rel_tol=0.0, abs_tol=1e-12),
        f"{label} drift: expected {right!r}, observed {observed!r}",
    )


def _require_reevaluated_metric_equal(
    left: Any,
    right: float,
    *,
    label: str,
    rel_tol: float = REEVALUATION_REL_TOL,
    abs_tol: float = REEVALUATION_ABS_TOL,
) -> None:
    observed = _finite_number(left, label=label)
    require(
        math.isclose(observed, right, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{label} replay drift: expected {right!r}, observed {observed!r}",
    )


def _finite_tree(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"Non-finite tensor at {path}")
    elif isinstance(value, Mapping):
        for name, child in value.items():
            _finite_tree(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Non-finite value at {path}")


def _artifact_hash_line(path: Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").strip().split()[0]
    except (OSError, IndexError) as error:
        raise ValueError(f"Invalid digest record: {path}") from error
    require(len(first) == 64, f"Invalid digest record: {path}")
    return first


def _parse_key_value_receipt(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read receipt: {path}") from error
    for line in lines:
        key, separator, value = line.partition("=")
        require(bool(separator) and bool(key) and key not in values, f"Malformed receipt: {path}")
        values[key] = value
    return values


def _manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read source manifest: {path}") from error
    for line in lines:
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and len(fields[0]) == 64, "Malformed source manifest")
        relative = fields[1].lstrip(" *")
        relative_path = Path(relative)
        require(
            bool(relative)
            and not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative not in entries,
            "Unsafe or duplicate source manifest path",
        )
        entries[relative] = fields[0]
    require(
        len(entries) == PINNED_SOURCE_FILE_COUNT,
        "Pinned source manifest entry count drift",
    )
    return entries


def _verify_manifest_check(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    require(sha256_file(path) == expected_sha256, f"Manifest check digest drift: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read manifest check: {path}") from error
    require(
        len(lines) == PINNED_SOURCE_FILE_COUNT
        and all(line.endswith(": OK") for line in lines),
        "Source manifest check is incomplete or contains a failure",
    )
    return {"path": str(path), "sha256": expected_sha256, "ok_entries": len(lines)}


def _verify_live_source_snapshot(
    *, source_root: Path, artifact_root: Path, expected_epochs: int
) -> dict[str, Any]:
    """Re-hash the live source tree and reject unregistered non-cache files."""
    manifest_sha = _artifact_hash_line(artifact_root / "source_manifest_sha256.txt")
    require(
        manifest_sha == PINNED_SOURCE_MANIFEST_SHA256,
        "Pinned source manifest digest record drift",
    )
    manifest_path = artifact_root / "source_manifest.sha256"
    require(manifest_path.is_file(), f"Pinned source manifest missing: {manifest_path}")
    require(sha256_file(manifest_path) == manifest_sha, "Pinned source manifest file drift")
    entries = _manifest_entries(manifest_path)
    actual: set[str] = set()
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        if "__pycache__" in path.parts or relative.endswith(".pyc"):
            continue
        actual.add(relative)
    require(actual == set(entries), "Live source file population differs from pinned manifest")
    for relative, expected_sha in entries.items():
        require(
            sha256_file(source_root / relative) == expected_sha,
            f"Live source file digest drift: {relative}",
        )

    if expected_epochs == 1:
        recorded = _verify_manifest_check(
            artifact_root / "source_manifest_check.txt",
            expected_sha256=PINNED_MANIFEST_CHECK_SHA256,
        )
        phase_receipt: dict[str, Any] = {
            "phase": "e1_preflight",
            "manifest_check": recorded,
        }
    else:
        check_path = artifact_root / "source_manifest_check_seed42_e300_live_epoch20.txt"
        receipt_path = artifact_root / "source_manifest_check_seed42_e300_live_epoch20.receipt"
        recorded = _verify_manifest_check(
            check_path, expected_sha256=PINNED_MANIFEST_CHECK_SHA256
        )
        require(
            sha256_file(receipt_path) == E300_MIDRUN_RECEIPT_SHA256,
            "e300 live-source receipt digest drift",
        )
        receipt = _parse_key_value_receipt(receipt_path)
        require(
            receipt
            == {
                "verified_at": "2026-09-09T13:25:35+09:00",
                "training_epoch_observed": "20",
                "manifest_entries": str(PINNED_SOURCE_FILE_COUNT),
                "all_entries_ok": "true",
            },
            "e300 live-source receipt content drift",
        )
        phase_receipt = {
            "phase": "e300_midrun_epoch20",
            "manifest_check": recorded,
            "receipt_path": str(receipt_path),
            "receipt_sha256": E300_MIDRUN_RECEIPT_SHA256,
            "receipt": receipt,
        }
    return {
        "status": "passed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_entries": len(entries),
        "live_file_population_exact": True,
        "live_file_digests_exact": True,
        "phase_receipt": phase_receipt,
    }


def _resolve_bound_path(root: Path, relative: Any, *, label: str) -> Path:
    require(isinstance(relative, str) and bool(relative), f"{label} path is invalid")
    path = root / relative
    require(path.is_file(), f"{label} is missing: {path}")
    return path


def load_screening_contracts(
    *,
    source_root: Path = ROOT,
    base_contract_path: Path = BASE_CONTRACT_PATH,
    amendment_path: Path = AMENDMENT_PATH,
    normalized_evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Load and digest-bind the screening contract and its amendment."""
    source_root = source_root.resolve()
    normalized_evidence_root = (
        source_root
        if normalized_evidence_root is None
        else normalized_evidence_root.resolve()
    )
    base_contract_path = base_contract_path.resolve()
    amendment_path = amendment_path.resolve()
    require(
        sha256_file(base_contract_path) == BASE_CONTRACT_SHA256,
        "VNC screening base contract digest drift",
    )
    require(
        sha256_file(amendment_path) == AMENDMENT_SHA256,
        "VNC screening amendment digest drift",
    )
    base = read_json(base_contract_path)
    amendment = read_json(amendment_path)
    require(
        base.get("contract_id") == "hard_lmm_value_norm_screening_5090_v1"
        and base.get("status") == "frozen_before_seed42_screening_outputs",
        "VNC screening base contract identity drift",
    )
    require(
        amendment.get("amendment_id")
        == "hard_lmm_value_norm_screening_5090_v1_amendment1"
        and amendment.get("amends_contract_id") == base["contract_id"]
        and amendment.get("amends_contract_sha256") == BASE_CONTRACT_SHA256,
        "VNC screening amendment identity drift",
    )
    require(
        amendment.get("status")
        == "frozen_before_candidate_final_summary_and_normalized_duration_outputs",
        "VNC screening amendment freeze status drift",
    )
    require(
        base.get("source", {}).get("candidate_revision") == PINNED_SOURCE_REVISION,
        "Pinned candidate source revision drift",
    )
    authorization = base.get("authorization")
    held_out = base.get("held_out_lock")
    require(
        isinstance(authorization, Mapping)
        and authorization.get("held_out_test_authorized") is False,
        "Held-out authorization drift",
    )
    require(
        isinstance(held_out, Mapping)
        and held_out.get("evaluation_scope") == "validation_only"
        and all(
            held_out.get(name) is False
            for name in (
                "test_artifacts_allowed",
                "test_decision_use_allowed",
                "test_loader_allowed",
                "test_metrics_allowed",
            )
        ),
        "Held-out lock drift",
    )

    candidate_ref = base["source"]["candidate_contract"]
    candidate_path = _resolve_bound_path(
        source_root, candidate_ref.get("path"), label="Candidate contract"
    )
    require(
        sha256_file(candidate_path) == candidate_ref.get("sha256"),
        "Candidate contract digest drift",
    )
    candidate = read_json(candidate_path)
    require(
        candidate.get("contract_id") == "hard_lmm_value_norm_consistent_v1"
        and candidate.get("status") == "frozen_before_gpu_execution",
        "Candidate contract identity drift",
    )
    route = candidate.get("candidate")
    require(
        isinstance(route, Mapping)
        and route.get("backbone") == BACKBONE
        and route.get("model_role") == MODEL_ROLE,
        "Candidate route contract drift",
    )

    evidence_digests: dict[str, dict[str, str]] = {}
    for group_name, records in (
        ("candidate", candidate.get("evidence_bindings")),
        ("screening", base.get("baseline_evidence")),
    ):
        require(isinstance(records, Mapping), f"{group_name} evidence bindings missing")
        evidence_digests[group_name] = {}
        for name, record in records.items():
            require(isinstance(record, Mapping), f"Invalid {group_name}.{name} binding")
            path = _resolve_bound_path(
                source_root, record.get("path"), label=f"{group_name}.{name} evidence"
            )
            observed = sha256_file(path)
            require(observed == record.get("sha256"), f"{group_name}.{name} digest drift")
            evidence_digests[group_name][str(name)] = observed

    replacement = amendment.get("replacement_reference")
    require(isinstance(replacement, Mapping), "Aligned-duration replacement is missing")
    aligned_path = _resolve_bound_path(
        source_root,
        replacement.get("contract_path"),
        label="Aligned-duration replacement contract",
    )
    aligned_sha = sha256_file(aligned_path)
    require(
        aligned_sha == replacement.get("contract_sha256"),
        "Aligned-duration replacement contract digest drift",
    )
    decision_path = _resolve_bound_path(
        normalized_evidence_root,
        replacement.get("decision_path"),
        label="Aligned-duration decision evidence",
    )
    decision_sha = sha256_file(decision_path)
    require(
        decision_sha == replacement.get("decision_sha256"),
        "Aligned-duration decision evidence digest drift",
    )
    decision = read_json(decision_path)
    require(
        decision.get("status") == "complete"
        and decision.get("contract_id") == replacement.get("contract_id")
        and decision.get("contract_sha256") == replacement.get("contract_sha256")
        and decision.get("calibration_source_revision")
        == replacement.get("calibration_source_revision")
        and decision.get("evaluation_scope") == "validation_only"
        and decision.get("held_out_test_evaluated") is False
        and decision.get("evidence_integrity") == "verified",
        "Aligned-duration decision identity, source, or scope drift",
    )
    decision_rows = decision.get("datasets")
    require(isinstance(decision_rows, list), "Aligned-duration decision rows missing")
    decision_by_dataset = {
        row.get("dataset"): row for row in decision_rows if isinstance(row, dict)
    }
    require(
        len(decision_by_dataset) == len(decision_rows),
        "Aligned-duration decision dataset rows are duplicated or invalid",
    )

    registry_ref = candidate["evidence_bindings"]["data_and_metric_registry"]
    registry_path = _resolve_bound_path(
        source_root, registry_ref.get("path"), label="Data and metric registry"
    )
    registry = read_json(registry_path)
    require(
        registry.get("contract_id") == "hard_lmm_bounded_qk_screening_v1",
        "Data and metric registry identity drift",
    )

    quantity_evidence_ref = base["baseline_evidence"]["quantity_metrics"]
    quantity_evidence = read_json(
        _resolve_bound_path(
            source_root,
            quantity_evidence_ref["path"],
            label="B quantity metric evidence",
        )
    )
    require(
        quantity_evidence.get("common_raw_rmse_goal_met") is True
        and quantity_evidence.get("contract_id")
        == "hard_lmm_quantile_checkpoint_alignment_v1",
        "B quantity metric evidence identity drift",
    )
    quantity_rows = {
        row.get("dataset"): row
        for row in quantity_evidence.get("datasets", [])
        if isinstance(row, dict)
    }
    raw_registry_ref = base["baseline_evidence"]["registry"]
    raw_registry = read_json(
        _resolve_bound_path(
            source_root,
            raw_registry_ref["path"],
            label="Raw-RMSE baseline registry",
        )
    )
    raw_rows = {
        row.get("dataset"): row
        for row in raw_registry.get("datasets", [])
        if isinstance(row, dict)
    }
    raw_evidence_binding = raw_registry.get("B_reference_evidence")
    require(
        isinstance(raw_evidence_binding, Mapping)
        and raw_evidence_binding.get("path") == quantity_evidence_ref["path"]
        and raw_evidence_binding.get("sha256") == quantity_evidence_ref["sha256"]
        and raw_evidence_binding.get("evaluation_scope") == "validation_only"
        and raw_evidence_binding.get("held_out_test_evaluated") is False,
        "Raw-RMSE registry does not bind the frozen B evidence",
    )
    baseline_crosscheck: dict[str, Any] = {}
    B_references = base.get("B_seed42_validation_references")
    require(isinstance(B_references, Mapping), "Base B references missing")
    candidate_datasets = {
        row.get("dataset"): row
        for row in candidate.get("datasets", [])
        if isinstance(row, dict)
    }
    for dataset, reference in B_references.items():
        require(isinstance(reference, Mapping), f"Invalid B reference: {dataset}")
        evidence_row = quantity_rows.get(dataset)
        raw_row = raw_rows.get(dataset)
        bounded_row = registry.get("datasets", {}).get(dataset, {}).get("B")
        candidate_row = candidate_datasets.get(dataset)
        require(
            isinstance(evidence_row, Mapping)
            and isinstance(raw_row, Mapping)
            and isinstance(bounded_row, Mapping)
            and isinstance(candidate_row, Mapping),
            f"B cross-check evidence missing: {dataset}",
        )
        evidence_metrics = evidence_row.get("metrics", {}).get("B_t0_raw_rmse")
        evidence_source = evidence_row.get("evidence", {}).get("B_t0_raw_rmse")
        bounded_metrics = bounded_row.get("metrics")
        bounded_source = bounded_row.get("source")
        require(
            all(
                isinstance(value, Mapping)
                for value in (
                    evidence_metrics,
                    evidence_source,
                    bounded_metrics,
                    bounded_source,
                )
            ),
            f"Malformed B cross-check evidence: {dataset}",
        )
        metric_fields = {
            "raw_rmse": "raw_rmse",
            "overall_mae": "overall_mae",
            "body_mae": "body_mae",
            "gt_p99_mae": "gt_p99_mae",
            "legacy_time_nll": "time_nll",
        }
        for reference_name, evidence_name in metric_fields.items():
            expected = _finite_number(
                reference.get(reference_name), label=f"base.{dataset}.{reference_name}"
            )
            _require_metric_equal(
                evidence_metrics.get(evidence_name),
                expected,
                label=f"quantity evidence {dataset}.{evidence_name}",
            )
            bounded_name = (
                "clamped_time_loss" if reference_name == "legacy_time_nll" else reference_name
            )
            _require_metric_equal(
                bounded_metrics.get(bounded_name),
                expected,
                label=f"metric registry {dataset}.{bounded_name}",
            )
        raw_metrics = raw_row.get("B_t0_raw_rmse_validation_metrics")
        require(isinstance(raw_metrics, Mapping), f"Raw B metrics missing: {dataset}")
        for name in ("raw_rmse", "overall_mae"):
            _require_metric_equal(
                raw_metrics.get(name),
                float(reference[name]),
                label=f"raw registry {dataset}.{name}",
            )
        expected_file_sha = reference.get("checkpoint_file_sha256")
        expected_state_sha = reference.get("checkpoint_state_sha256")
        for label, record in (
            ("quantity evidence", evidence_source),
            ("metric registry", bounded_source),
        ):
            require(
                record.get("checkpoint_file_sha256") == expected_file_sha
                and record.get("checkpoint_state_sha256") == expected_state_sha,
                f"{label} B checkpoint digest drift: {dataset}",
            )
        require(
            raw_row.get("B_t0_raw_rmse_checkpoint_file_sha256") == expected_file_sha
            and raw_row.get("B_t0_raw_rmse_checkpoint_state_sha256")
            == expected_state_sha
            and candidate_row.get("B_checkpoint_file_sha256") == expected_file_sha
            and candidate_row.get("B_checkpoint_state_sha256") == expected_state_sha,
            f"B checkpoint hashes differ across registries: {dataset}",
        )
        require(
            evidence_metrics.get("validation_target_count")
            == bounded_metrics.get("validation_target_count")
            == raw_row.get("expected_validation_targets")
            == candidate_row.get("validation_target_count"),
            f"B validation count differs across registries: {dataset}",
        )
        baseline_crosscheck[str(dataset)] = {
            "metrics_exact": True,
            "checkpoint_file_sha256": expected_file_sha,
            "checkpoint_state_sha256": expected_state_sha,
            "validation_target_count": candidate_row["validation_target_count"],
        }

    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    normalized_refs = amendment.get("normalized_time_references")
    require(isinstance(normalized_refs, Mapping), "Normalized-time references missing")
    normalized_evidence_digests: dict[str, dict[str, str]] = {}
    for dataset, reference in normalized_refs.items():
        require(isinstance(reference, Mapping), f"Invalid normalized reference: {dataset}")
        baseline = _finite_number(
            reference.get("aligned_B_primary_nll"),
            label=f"{dataset}.aligned_B_primary_nll",
        )
        maximum = _finite_number(
            reference.get("candidate_gate_max_aligned_B_plus_0_01"),
            label=f"{dataset}.normalized_gate_max",
        )
        require(
            math.isclose(maximum, baseline + 0.01, rel_tol=0.0, abs_tol=1e-15),
            f"{dataset} normalized-time margin drift",
        )
        summary_path = _resolve_bound_path(
            normalized_evidence_root,
            reference.get("summary_path"),
            label=f"{dataset} aligned-B summary",
        )
        checkpoint_path = _resolve_bound_path(
            normalized_evidence_root,
            reference.get("selected_checkpoint_path"),
            label=f"{dataset} aligned-B selected checkpoint",
        )
        summary_sha = sha256_file(summary_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        require(
            summary_sha == reference.get("summary_sha256"),
            f"{dataset} aligned-B summary digest drift",
        )
        require(
            checkpoint_sha == reference.get("selected_checkpoint_file_sha256"),
            f"{dataset} aligned-B checkpoint file digest drift",
        )
        summary = read_json(summary_path)
        selection = replacement.get("selection")
        expected_summary = {
            "status": "success",
            "contract_id": replacement.get("contract_id"),
            "contract_sha256": replacement.get("contract_sha256"),
            "dataset": dataset,
            "model_role": "B",
            "seed": 42,
            "source_backbone": "titantpp",
            "source_variant": VARIANT,
            "time_head_mode": "heteroscedastic_lognormal_duration",
            "observation_likelihood_mode": reference.get(
                "observation_likelihood_mode"
            ),
            "best_epoch": reference.get("best_epoch"),
            "completed_epochs": reference.get("completed_epochs"),
            "selection": selection,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "qualified_full_data": True,
            "qualified_full_fit": True,
            "quantity_prediction_bitwise_identical": True,
            "selected_checkpoint_file_sha256": checkpoint_sha,
            "selected_state_sha256": reference.get(
                "selected_checkpoint_state_sha256"
            ),
            "source_non_time_state_sha256": reference.get(
                "source_non_time_state_sha256"
            ),
            "selected_non_time_state_sha256": reference.get(
                "source_non_time_state_sha256"
            ),
        }
        for name, expected in expected_summary.items():
            require(
                summary.get(name) == expected,
                f"{dataset} aligned-B summary {name} drift",
            )
        _require_metric_equal(
            summary.get("best_validation_proper_time_nll"),
            baseline,
            label=f"{dataset} aligned-B summary primary NLL",
        )
        history = summary.get("history")
        require(
            isinstance(history, list)
            and len(history) == int(reference["completed_epochs"]) + 1,
            f"{dataset} aligned-B summary history length drift",
        )

        checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
        require(isinstance(checkpoint, dict), f"{dataset} aligned-B checkpoint invalid")
        expected_checkpoint = {
            "checkpoint_type": "selected_frozen_lognormal_duration",
            "checkpoint_schema_version": 1,
            "selection": selection,
            "best_epoch": reference.get("best_epoch"),
            "backbone": "titantpp",
            "model_role": "B",
            "variant": VARIANT,
            "time_head_mode": "heteroscedastic_lognormal_duration",
            "source_non_time_state_sha256": reference.get(
                "source_non_time_state_sha256"
            ),
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }
        for name, expected in expected_checkpoint.items():
            require(
                checkpoint.get(name) == expected,
                f"{dataset} aligned-B checkpoint {name} drift",
            )
        _require_metric_equal(
            checkpoint.get("selected_metric_value"),
            baseline,
            label=f"{dataset} aligned-B checkpoint primary NLL",
        )
        selected_state = checkpoint.get("model_state_dict")
        require(
            isinstance(selected_state, dict) and bool(selected_state),
            f"{dataset} aligned-B checkpoint state missing",
        )
        _finite_tree(selected_state, path=f"{dataset}.aligned_B.model_state_dict")
        selected_state_sha = canonical_state_dict_sha256(selected_state)
        require(
            selected_state_sha
            == checkpoint.get("model_state_sha256")
            == reference.get("selected_checkpoint_state_sha256"),
            f"{dataset} aligned-B selected state digest drift",
        )

        decision_row = decision_by_dataset.get(dataset)
        require(isinstance(decision_row, Mapping), f"{dataset} decision row missing")
        require(
            decision_row.get("observation_likelihood_mode")
            == reference.get("observation_likelihood_mode"),
            f"{dataset} decision likelihood mode drift",
        )
        _require_metric_equal(
            decision_row.get("aligned_B_primary_nll"),
            baseline,
            label=f"{dataset} decision aligned-B primary NLL",
        )
        decision_evidence = decision_row.get("evidence", {}).get("B")
        require(
            isinstance(decision_evidence, Mapping),
            f"{dataset} decision B evidence missing",
        )
        expected_decision_evidence = {
            "calibration_source_revision": replacement.get(
                "calibration_source_revision"
            ),
            "primary_nll": baseline,
            "selected_checkpoint_file_sha256": checkpoint_sha,
            "selected_checkpoint_state_sha256": selected_state_sha,
            "summary_file_sha256": summary_sha,
        }
        for name, expected in expected_decision_evidence.items():
            observed = decision_evidence.get(name)
            if name == "primary_nll":
                _require_metric_equal(
                    observed,
                    float(expected),
                    label=f"{dataset} decision B evidence {name}",
                )
            else:
                require(
                    observed == expected,
                    f"{dataset} decision B evidence {name} drift",
                )
        normalized_evidence_digests[str(dataset)] = {
            "summary_sha256": summary_sha,
            "selected_checkpoint_file_sha256": checkpoint_sha,
            "selected_checkpoint_state_sha256": selected_state_sha,
            "source_non_time_state_sha256": str(
                reference["source_non_time_state_sha256"]
            ),
        }

    return {
        "base": base,
        "amendment": amendment,
        "candidate": candidate,
        "data_registry": registry,
        "digests": {
            "base_contract_sha256": BASE_CONTRACT_SHA256,
            "amendment_sha256": AMENDMENT_SHA256,
            "candidate_contract_sha256": sha256_file(candidate_path),
            "aligned_duration_contract_sha256": aligned_sha,
            "aligned_duration_decision_sha256": decision_sha,
            "aligned_duration_evidence": normalized_evidence_digests,
            "B_reference_crosscheck": baseline_crosscheck,
            "bound_evidence": evidence_digests,
        },
        "raw_rmse_registry": raw_registry,
    }


def _candidate_dataset_binding(
    contracts: Mapping[str, Any], dataset: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets = contracts["candidate"].get("datasets")
    require(isinstance(datasets, list), "Candidate dataset bindings missing")
    matches = [row for row in datasets if isinstance(row, dict) and row.get("dataset") == dataset]
    require(len(matches) == 1, f"Candidate dataset binding missing or duplicated: {dataset}")
    candidate_binding = matches[0]
    registry_binding = contracts["data_registry"].get("data_bindings", {}).get(dataset)
    require(isinstance(registry_binding, dict), f"Metric registry binding missing: {dataset}")
    for name in ("data_sha256", "split_manifest_sha256"):
        require(
            candidate_binding.get(name) == registry_binding.get(name),
            f"{dataset} {name} differs across contracts",
        )
    population = registry_binding.get("validation_target_population")
    require(isinstance(population, dict), f"{dataset} validation population missing")
    expected_population = {
        "target_count": candidate_binding.get("validation_target_count"),
        "target_identity_sha256": candidate_binding.get(
            "validation_target_identity_sha256"
        ),
        "target_quantity_sha256": candidate_binding.get(
            "validation_target_quantity_sha256"
        ),
    }
    for name, expected in expected_population.items():
        require(population.get(name) == expected, f"{dataset} {name} contract drift")
    return candidate_binding, registry_binding


def _validate_artifact_provenance(
    *,
    artifact_root: Path,
    contracts: Mapping[str, Any],
    dataset: str,
    candidate_binding: Mapping[str, Any],
    source_revision: str,
    source_root: Path,
    expected_epochs: int,
) -> dict[str, Any]:
    base = contracts["base"]
    e1_path = artifact_root / "e1_audit_v4.json"
    expected_e1_sha = base["source"]["e1_audit"]["sha256"]
    e1_sha = sha256_file(e1_path)
    require(e1_sha == expected_e1_sha, "Pinned e1 v4 audit digest drift")
    e1 = read_json(e1_path)
    require(
        e1.get("status") == "passed"
        and e1.get("contract_id") == "hard_lmm_value_norm_consistent_v1"
        and e1.get("source_revision") == source_revision
        and e1.get("held_out_test_evaluated") is False,
        "Pinned e1 v4 audit identity or scope drift",
    )
    e1_data = e1.get("data", {}).get(dataset)
    require(isinstance(e1_data, Mapping), f"e1 data proof missing: {dataset}")
    for name in ("data_sha256", "split_manifest_sha256"):
        require(
            e1_data.get(name) == candidate_binding.get(name),
            f"e1 {dataset} {name} drift",
        )
    revision_path = artifact_root / "source_revision.txt"
    require(
        revision_path.read_text(encoding="utf-8").strip() == source_revision,
        "Artifact source revision record drift",
    )
    archive_record = _artifact_hash_line(artifact_root / "source_archive_sha256.txt")
    manifest_record = _artifact_hash_line(artifact_root / "source_manifest_sha256.txt")
    require(
        archive_record == e1.get("source_archive_sha256"),
        "Source archive digest record drift",
    )
    require(
        manifest_record == e1.get("source_manifest_sha256"),
        "Source manifest digest record drift",
    )
    require(
        isinstance(e1.get("source_file_count"), int)
        and e1["source_file_count"] > 0,
        "Source manifest file-count proof missing",
    )
    live_source = _verify_live_source_snapshot(
        source_root=source_root,
        artifact_root=artifact_root,
        expected_epochs=expected_epochs,
    )
    return {
        "source_revision": source_revision,
        "source_archive_sha256": archive_record,
        "source_manifest_sha256": manifest_record,
        "source_file_count": int(e1["source_file_count"]),
        "e1_audit_v4_sha256": e1_sha,
        "e1_audit_v4_status": "passed",
        "data_sha256": e1_data["data_sha256"],
        "split_manifest_sha256": e1_data["split_manifest_sha256"],
        "live_source_snapshot": live_source,
    }


def _validate_host_preflight(
    *, artifact_root: Path, dataset: str, expected_epochs: int
) -> dict[str, Any]:
    prefix = f"e1_{dataset}" if expected_epochs == 1 else f"seed42_screening_{dataset}"
    gpu_path = artifact_root / "logs" / f"{prefix}_preflight_gpu.txt"
    processes_path = artifact_root / "logs" / f"{prefix}_preflight_processes.txt"
    try:
        fields = [
            value.strip()
            for value in gpu_path.read_text(encoding="utf-8").strip().split(",")
        ]
    except OSError as error:
        raise ValueError(f"GPU preflight receipt missing: {gpu_path}") from error
    require(len(fields) == 4, "Malformed GPU preflight receipt")
    name, total, free, utilization = fields
    require(
        name == "NVIDIA GeForce RTX 5090"
        and int(total) == 32607
        and int(free) >= 12000
        and int(utilization) == 0,
        "GPU preflight did not bind an idle RTX 5090 with sufficient memory",
    )
    require(
        processes_path.is_file()
        and processes_path.stat().st_size == 0
        and sha256_file(processes_path)
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "GPU process preflight was not empty",
    )
    result: dict[str, Any] = {
        "execution_host": "RTX5090-server",
        "gpu_name": name,
        "total_vram_mib": int(total),
        "free_vram_mib": int(free),
        "utilization_percent": int(utilization),
        "compute_process_count": 0,
        "gpu_receipt_path": str(gpu_path),
        "gpu_receipt_sha256": sha256_file(gpu_path),
        "process_receipt_path": str(processes_path),
        "process_receipt_sha256": sha256_file(processes_path),
    }
    if expected_epochs == 300:
        require(
            sha256_file(gpu_path) == E300_GPU_PREFLIGHT_SHA256,
            "e300 GPU preflight receipt digest drift",
        )
        launcher = artifact_root / "tools/run_vnc_instacart_seed42_e300.sh"
        require(
            dataset == "insta_market_basket"
            and sha256_file(launcher) == E300_LAUNCHER_SHA256,
            "e300 launcher receipt digest drift",
        )
        host_receipt_path = (
            artifact_root
            / "logs/seed42_screening_insta_market_basket_midrun_host_gpu.receipt"
        )
        require(
            sha256_file(host_receipt_path) == E300_HOST_GPU_RECEIPT_SHA256,
            "e300 host/GPU mid-run receipt digest drift",
        )
        host_receipt = _parse_key_value_receipt(host_receipt_path)
        require(
            host_receipt
            == {
                "verified_at": "2026-09-09T14:02:17+09:00",
                "hostname": "RTX5090-server",
                "training_epoch_observed": "33",
                "training_pid": "1536916",
                "gpu": (
                    "NVIDIA GeForce RTX 5090, "
                    "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, "
                    "00000000:02:00.0, 595.84"
                ),
                "compute_process": (
                    "1536916, /opt/miniconda3/envs/ai_env/bin/python, 1158"
                ),
                "process_command_sha256": (
                    "0139cdbebe2a35381b54305d5b96bb576d74ddc99170e16d4f115abb3fefe265"
                ),
                "process_cwd": (
                    "/home/leekwanhyeong/workspace/paper_research_vnc_956603f_5090"
                ),
                "tmux_session": "vnc_insta42_956603f",
                "tmux_present": "true",
            },
            "e300 host/GPU mid-run receipt content drift",
        )
        status_path = artifact_root / "logs" / f"{prefix}.status"
        try:
            status = status_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError("e300 launcher status receipt missing") from error
        require(status.startswith("complete "), "e300 launcher did not complete successfully")
        result.update(
            {
                "launcher_path": str(launcher),
                "launcher_sha256": E300_LAUNCHER_SHA256,
                "launcher_status_path": str(status_path),
                "launcher_status": status,
                "host_gpu_receipt_path": str(host_receipt_path),
                "host_gpu_receipt_sha256": E300_HOST_GPU_RECEIPT_SHA256,
                "host_gpu_receipt": host_receipt,
            }
        )
    return result


def _expected_execution_role(dataset: str, expected_epochs: int) -> str:
    if expected_epochs == 1:
        suffix = "1e"
    else:
        suffix = "seed42_e300"
    return f"hard_lmm_backbone_5090_{BACKBONE}_{dataset}_{suffix}"


def _validate_route_and_training_identity(
    *,
    job: Path,
    summary: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    registry_binding: Mapping[str, Any],
    dataset: str,
    source_revision: str,
    expected_epochs: int,
) -> dict[str, Any]:
    launch = read_json(job / "launch_contract.json")
    minimum_epochs = 1 if expected_epochs == 1 else 40
    execution_role = _expected_execution_role(dataset, expected_epochs)
    expected_launch = {
        "dataset": dataset,
        "model_role": MODEL_ROLE,
        "backbones": [BACKBONE],
        "seeds": [42],
        "quantity_variants": [VARIANT],
        "source_revision": source_revision,
        "epochs": expected_epochs,
        "batch_size": 128,
        "lr": 0.001,
        "hidden_dim": 64,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "quantile_adaptive_strength": 0.0,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "execution_host": "RTX5090-server",
        "execution_role": execution_role,
        "partial_smoke": False,
        "max_series": None,
    }
    for name, expected in expected_launch.items():
        require(launch.get(name) == expected, f"Launch {name} route drift")
    require(launch.get("status") == "complete", "Launch contract is incomplete")
    early = launch.get("early_stopping")
    require(
        isinstance(early, Mapping)
        and early.get("min_epochs") == minimum_epochs
        and early.get("patience") == 40
        and early.get("monitor") == MONITOR
        and early.get("comparison") == "earliest_strict_finite_minimum"
        and early.get("restore") == "best_validation_raw_quantity_rmse",
        "Launch early-stopping contract drift",
    )
    require(
        launch.get("data_sha256") == candidate_binding.get("data_sha256")
        and launch.get("split_manifest_sha256")
        == candidate_binding.get("split_manifest_sha256"),
        "Launch data binding drift",
    )
    require(
        launch.get("validation_target_population")
        == registry_binding.get("validation_target_population"),
        "Launch validation population drift",
    )
    quantity_contract = launch.get("quantity_contract")
    require(isinstance(quantity_contract, Mapping), "Launch quantity contract missing")
    require(
        quantity_contract.get("boundaries")
        == registry_binding.get("quantity_boundaries")
        and quantity_contract.get("quantiles") == [0.5, 0.9, 0.95, 0.99],
        "Launch quantity strata contract drift",
    )

    expected_summary = {
        "status": "success",
        "backbone": BACKBONE,
        "variant": VARIANT,
        "seed": 42,
        "source_revision": source_revision,
        "source_revision_history": [source_revision],
        "epochs": expected_epochs,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_monitor": MONITOR,
        "checkpoint_monitor_history_key": HISTORY_METRIC,
        "checkpoint_selection": SELECTION,
        "training_device": "cuda",
        "lambda_tail": 0.0,
    }
    for name, expected in expected_summary.items():
        require(summary.get(name) == expected, f"Summary {name} route drift")
    require(
        int(summary.get("cuda_peak_memory_allocated_bytes", 0)) > 0,
        "CUDA allocation proof missing",
    )
    resume = summary.get("resume_identity")
    require(isinstance(resume, Mapping), "Summary resume identity missing")
    require(
        resume.get("backbone") == BACKBONE
        and resume.get("variant") == VARIANT
        and resume.get("seed") == 42
        and resume.get("checkpoint_monitor") == MONITOR
        and resume.get("quantity_contract") == quantity_contract
        and resume.get("interface_meta") == summary.get("interface_meta"),
        "Summary resume identity drift",
    )
    arguments = resume.get("arguments")
    require(isinstance(arguments, Mapping), "Resume training arguments missing")
    population = registry_binding["validation_target_population"]
    expected_arguments = {
        "dataset_contract": dataset,
        "data_sha256": candidate_binding["data_sha256"],
        "split_manifest_sha256": candidate_binding["split_manifest_sha256"],
        "lookback_weeks": int(population["lookback_weeks"]),
        "max_seq_len": int(population["max_seq_len"]),
        "max_train_batches": None,
        "max_val_batches": None,
        "max_series": None,
        "hidden_dim": 64,
        "batch_size": 128,
        "lr": 0.001,
        "grad_clip": 1.0,
        "epochs": expected_epochs,
        "min_epochs": minimum_epochs,
        "early_stopping_patience": 40,
        "model_role": MODEL_ROLE,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "quantile_adaptive_strength": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 300.0,
        "time_wd_safety_limit": 40.0,
        "time_head_lr_multiplier": 1.0,
        "source_revision": source_revision,
        "execution_role": execution_role,
        "allow_partial_contract": True,
        "titans_memory_gradient_clip": None,
    }
    for name, expected in expected_arguments.items():
        require(
            name in arguments and arguments[name] == expected,
            f"Resume argument {name} drift",
        )
    return {
        "execution_role": execution_role,
        "minimum_epochs": minimum_epochs,
        "patience": 40,
        "validation_target_population": registry_binding[
            "validation_target_population"
        ],
        "quantity_boundaries": registry_binding["quantity_boundaries"],
    }


def validate_history_and_stop(
    history: Sequence[Mapping[str, Any]],
    *,
    expected_epochs: int,
    completed_epochs: int,
    stopped_early: bool,
    minimum_epochs: int,
    patience: int,
) -> dict[str, Any]:
    """Replay the shared raw-RMSE selector and the first valid stop point."""
    require(expected_epochs in SUPPORTED_EPOCH_BUDGETS, "Unsupported epoch budget")
    require(bool(history), "Training history is empty")
    require(
        all(isinstance(row, Mapping) for row in history),
        "Training history row is invalid",
    )
    rows = [dict(row) for row in history]
    _finite_tree(rows, path="history")
    require(len(rows) == completed_epochs, "Training history length drift")
    require(
        [row.get("epoch") for row in rows] == list(range(1, completed_epochs + 1)),
        "Training history epoch sequence drift",
    )

    from paper.scripts.count_aware_tpp_backbone.training import (
        earliest_strict_minimum,
        early_stopping_exhausted,
    )

    selected = earliest_strict_minimum(rows, metric_key=HISTORY_METRIC)
    require(isinstance(selected, dict), "Raw-RMSE selection is missing")
    first_stop: int | None = None
    if expected_epochs == 300:
        require(minimum_epochs == 40 and patience == 40, "e300 stop policy drift")
        for end in range(1, len(rows) + 1):
            if early_stopping_exhausted(
                rows[:end],
                min_epochs=minimum_epochs,
                patience=patience,
                metric_key=HISTORY_METRIC,
            ):
                first_stop = int(rows[end - 1]["epoch"])
                break
        if first_stop is None or first_stop >= expected_epochs:
            require(
                completed_epochs == expected_epochs and stopped_early is False,
                "e300 job ended before its epoch budget without early stopping",
            )
        else:
            require(
                completed_epochs == first_stop and stopped_early is True,
                "e300 job stopped before or continued past the first valid stop epoch",
            )
    else:
        # e1 is an execution-contract smoke run.  Its min40/patience40 screening
        # rule is intentionally not replayed or used as a performance decision.
        require(minimum_epochs == 1 and patience == 40, "e1 stop policy drift")
        require(
            completed_epochs == 1 and stopped_early is False,
            "e1 completion metadata drift",
        )
    return {
        "history_length": len(rows),
        "selected_epoch": int(selected["epoch"]),
        "selected_val_qty_rmse": _finite_number(
            selected[HISTORY_METRIC], label="selected val_qty_rmse"
        ),
        "first_early_stop_epoch": first_stop,
        "stopped_early": stopped_early,
        "screening_stop_policy_applied": expected_epochs == 300,
    }


def evaluate_stage1_gate(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    expected_epochs: int,
) -> dict[str, Any]:
    """Evaluate the five frozen B-relative checks, applying them only to e300."""
    actual = {
        "raw_rmse": _finite_number(metrics.get("raw_rmse"), label="candidate.raw_rmse"),
        "overall_mae": _finite_number(
            metrics.get("overall_mae"), label="candidate.overall_mae"
        ),
        "body_mae": _finite_number(metrics.get("body_mae"), label="candidate.body_mae"),
        "gt_p99_mae": _finite_number(
            metrics.get("gt_p99_mae"), label="candidate.gt_p99_mae"
        ),
        "legacy_time_nll": _finite_number(
            metrics.get("legacy_time_nll", metrics.get("clamped_time_loss")),
            label="candidate.legacy_time_nll",
        ),
    }
    reference = {
        name: _finite_number(baseline.get(name), label=f"B.{name}")
        for name in actual
    }
    require(
        all(value >= 0.0 for name, value in actual.items() if name != "legacy_time_nll"),
        "Candidate quantity metric is negative",
    )
    checks = {
        "raw_rmse_strictly_better_than_B": actual["raw_rmse"] < reference["raw_rmse"],
        "overall_mae_within_B_plus_1_percent": actual["overall_mae"]
        <= reference["overall_mae"] * 1.01,
        "body_mae_within_B_plus_2_percent": actual["body_mae"]
        <= reference["body_mae"] * 1.02,
        "gt_p99_mae_within_B_plus_2_percent": actual["gt_p99_mae"]
        <= reference["gt_p99_mae"] * 1.02,
        "legacy_time_nll_within_B_plus_0_01": actual["legacy_time_nll"]
        <= reference["legacy_time_nll"] + 0.01,
    }
    applicable = expected_epochs == 300
    return {
        "status": (
            "passed" if applicable and all(checks.values()) else
            "failed" if applicable else
            "not_applicable_e1"
        ),
        "applicable": applicable,
        "affects_followup": applicable,
        "checks": checks,
        "values": {"candidate": actual, "B": reference},
        "limits": {
            "raw_rmse": reference["raw_rmse"],
            "overall_mae": reference["overall_mae"] * 1.01,
            "body_mae": reference["body_mae"] * 1.02,
            "gt_p99_mae": reference["gt_p99_mae"] * 1.02,
            "legacy_time_nll": reference["legacy_time_nll"] + 0.01,
        },
        "deltas": {
            "raw_rmse": actual["raw_rmse"] - reference["raw_rmse"],
            "overall_mae": actual["overall_mae"] - reference["overall_mae"],
            "body_mae": actual["body_mae"] - reference["body_mae"],
            "gt_p99_mae": actual["gt_p99_mae"] - reference["gt_p99_mae"],
            "legacy_time_nll": actual["legacy_time_nll"]
            - reference["legacy_time_nll"],
        },
    }


def _state_dict(
    payload: Mapping[str, Any], key: str, *, artifact: str
) -> dict[str, torch.Tensor]:
    value = payload.get(key)
    require(isinstance(value, dict) and bool(value), f"{artifact} {key} missing")
    require(
        all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in value.items()),
        f"{artifact} {key} is not a tensor state dict",
    )
    _finite_tree(value, path=f"{artifact}.{key}")
    return value


def _same_state(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], *, label: str
) -> None:
    require(set(left) == set(right), f"{label} state keys differ")
    for name in left:
        require(torch.equal(left[name], right[name]), f"{label} state differs at {name}")


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("python") != right.get("python"):
        return False
    if not torch.equal(left["torch"].cpu(), right["torch"].cpu()):
        return False
    left_numpy, right_numpy = left["numpy"], right["numpy"]
    if (
        left_numpy[0] != right_numpy[0]
        or not np.array_equal(left_numpy[1], right_numpy[1])
        or left_numpy[2:] != right_numpy[2:]
    ):
        return False
    left_cuda = left.get("cuda", [])
    right_cuda = right.get("cuda", [])
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(x.cpu(), y.cpu()) for x, y in zip(left_cuda, right_cuda)
    )


def _validate_checkpoints_and_resume(
    *,
    run_dir: Path,
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_revision: str,
    history: list[Mapping[str, Any]],
    selector: Mapping[str, Any],
) -> dict[str, Any]:
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.audit_hard_lmm_causal_qkv import (
        _require_equal_outputs,
        _restore_model,
        _synthetic_outputs,
    )
    from paper.scripts.count_aware_tpp_backbone.training import (
        build_optimizer,
        optimizer_group_contract,
    )
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        capture_rng_state,
        restore_rng_state,
        restore_train_loader_generator_state,
        torch_load_checkpoint,
    )

    best_path = run_dir / "best_val_qty_rmse_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    history_path = run_dir / "history.json"
    require(best_path.is_file(), "Selected checkpoint is missing")
    require(last_path.is_file(), "Last-state checkpoint is missing")
    best = torch_load_checkpoint(best_path, map_location="cpu")
    last = torch_load_checkpoint(last_path, map_location="cpu")
    require(isinstance(best, dict), "Selected checkpoint payload is invalid")
    require(isinstance(last, dict), "Last-state checkpoint payload is invalid")
    for label, payload in (("selected", best), ("last", last)):
        validate_checkpoint_route(payload, BACKBONE)
        expected = {
            "backbone": BACKBONE,
            "variant": VARIANT,
            "seed": 42,
            "source_revision": source_revision,
            "source_revision_history": [source_revision],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "checkpoint_monitor": MONITOR,
            "checkpoint_monitor_history_key": HISTORY_METRIC,
            "checkpoint_selection": SELECTION,
            "selection": SELECTION,
        }
        for name, value in expected.items():
            require(payload.get(name) == value, f"{label} checkpoint {name} drift")
        for name in (
            "encoder_config",
            "interface_meta",
            "optimizer_group_contract",
            "resume_identity",
            "initial_state_sha256",
        ):
            require(payload.get(name) == summary.get(name), f"{label} {name} drift")
    require(
        last.get("checkpoint_type") == "epoch_resume"
        and last.get("checkpoint_schema_version") == 2,
        "Last-state checkpoint schema drift",
    )
    require(last.get("epoch") == summary.get("completed_epochs"), "Last epoch drift")
    require(last.get("history") == history, "Last checkpoint history drift")
    selected_epoch = int(selector["selected_epoch"])
    selected_value = float(selector["selected_val_qty_rmse"])
    require(
        summary.get("best_epoch")
        == best.get("best_epoch")
        == last.get("best_epoch")
        == selected_epoch,
        "Selected epoch differs across artifacts",
    )
    _require_reevaluated_metric_equal(
        summary.get("best_val_qty_rmse"),
        selected_value,
        label="summary.best_val_qty_rmse",
    )
    for label, value in (
        ("summary.selected_metric_value", summary.get("selected_metric_value")),
        ("best.selected_metric_value", best.get("selected_metric_value")),
        ("last.best_selection_value", last.get("best_selection_value")),
    ):
        _require_metric_equal(value, selected_value, label=label)

    best_state = _state_dict(best, "model_state_dict", artifact="selected checkpoint")
    last_state = _state_dict(last, "model_state_dict", artifact="last checkpoint")
    last_best_state = _state_dict(last, "best_state_dict", artifact="last checkpoint")
    best_sha = canonical_state_dict_sha256(best_state)
    last_sha = canonical_state_dict_sha256(last_state)
    last_best_sha = canonical_state_dict_sha256(last_best_state)
    require(
        best_sha == best.get("model_state_sha256") == summary.get("checkpoint_state_sha256"),
        "Selected state digest drift",
    )
    require(last_sha == last.get("model_state_sha256"), "Last state digest drift")
    require(last_best_sha == last.get("best_state_sha256"), "Last-best state digest drift")
    require(last_best_sha == best_sha, "Last-best state does not match selected state")
    _same_state(best_state, last_best_state, label="selected/last-best")

    best_model = _restore_model(
        best, best_state, candidate=candidate, artifact="selected checkpoint"
    )
    last_best_model = _restore_model(
        last, last_best_state, candidate=candidate, artifact="last-best checkpoint"
    )
    last_model = _restore_model(
        last, last_state, candidate=candidate, artifact="last checkpoint"
    )
    _require_equal_outputs(
        _synthetic_outputs(best_model), _synthetic_outputs(last_best_model)
    )

    optimizer_state = last.get("optimizer_state_dict")
    require(isinstance(optimizer_state, dict), "Last optimizer state is missing")
    require(bool(optimizer_state.get("state")), "Last optimizer state is empty")
    _finite_tree(optimizer_state, path="last.optimizer_state_dict")
    arguments = last["resume_identity"]["arguments"]
    optimizer = build_optimizer(
        last_model,
        lr=float(arguments["lr"]),
        time_head_lr_multiplier=float(arguments["time_head_lr_multiplier"]),
    )
    expected_group_contract = optimizer_group_contract(optimizer)
    require(
        expected_group_contract == summary.get("optimizer_group_contract"),
        "Fresh optimizer groups differ from the pinned optimizer contract",
    )
    expected_parameter_ids = [
        [id(parameter) for parameter in group["params"]]
        for group in optimizer.param_groups
    ]
    expected_hyperparameters = [
        {name: value for name, value in group.items() if name != "params"}
        for group in optimizer.param_groups
    ]
    expected_trainable_ids = {
        id(parameter) for parameter in last_model.parameters() if parameter.requires_grad
    }
    require(
        len(expected_trainable_ids)
        == sum(len(group) for group in expected_parameter_ids)
        == len({parameter_id for group in expected_parameter_ids for parameter_id in group}),
        "Fresh optimizer does not contain every trainable parameter exactly once",
    )
    require(
        {parameter_id for group in expected_parameter_ids for parameter_id in group}
        == expected_trainable_ids,
        "Fresh optimizer parameter population differs from the restored model",
    )
    try:
        optimizer.load_state_dict(optimizer_state)
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        raise ValueError("Last optimizer strict restore failed") from error
    _finite_tree(optimizer.state_dict(), path="restored_optimizer")
    require(
        optimizer_group_contract(optimizer) == expected_group_contract,
        "Restored optimizer group contract drift",
    )
    require(
        [
            [id(parameter) for parameter in group["params"]]
            for group in optimizer.param_groups
        ]
        == expected_parameter_ids,
        "Restored optimizer parameter population drift",
    )
    require(
        [
            {name: value for name, value in group.items() if name != "params"}
            for group in optimizer.param_groups
        ]
        == expected_hyperparameters,
        "Restored AdamW hyperparameter drift",
    )
    require(
        {id(parameter) for parameter in optimizer.state} == expected_trainable_ids,
        "Restored optimizer state does not cover every trainable parameter exactly",
    )
    for parameter, parameter_state in optimizer.state.items():
        require(
            set(parameter_state) == {"step", "exp_avg", "exp_avg_sq"},
            "Restored AdamW parameter-state schema drift",
        )
        step = parameter_state["step"]
        exp_avg = parameter_state["exp_avg"]
        exp_avg_sq = parameter_state["exp_avg_sq"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and bool(torch.isfinite(step).all())
            and float(step) > 0.0,
            "Restored AdamW step is invalid",
        )
        for name, moment in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            require(
                isinstance(moment, torch.Tensor)
                and moment.shape == parameter.shape
                and moment.dtype == parameter.dtype
                and bool(torch.isfinite(moment).all()),
                f"Restored AdamW {name} tensor contract drift",
            )

    alpha_values: dict[str, float] = {}
    for label, state in (("selected", best_state), ("last", last_state)):
        alpha = state.get(ALPHA_KEY)
        require(
            isinstance(alpha, torch.Tensor)
            and tuple(alpha.shape) == ()
            and alpha.is_floating_point()
            and bool(torch.isfinite(alpha))
            and float(alpha) != 0.0,
            f"{label} learned alpha is missing, zero, or non-finite",
        )
        alpha_values[label] = float(alpha)
    alpha_parameter = dict(last_model.named_parameters()).get(ALPHA_KEY)
    require(isinstance(alpha_parameter, torch.nn.Parameter), "Restored alpha parameter missing")
    alpha_optimizer = optimizer.state.get(alpha_parameter)
    require(isinstance(alpha_optimizer, dict) and bool(alpha_optimizer), "Alpha optimizer state missing")
    _finite_tree(alpha_optimizer, path="restored_optimizer.alpha")
    exp_avg = alpha_optimizer.get("exp_avg")
    require(
        isinstance(exp_avg, torch.Tensor)
        and bool(torch.isfinite(exp_avg).all())
        and float(exp_avg.abs().max()) > 0.0,
        "Alpha optimizer exp_avg is missing, zero, or non-finite",
    )

    saved_rng = last.get("rng_state")
    loader_state = last.get("train_loader_generator_state")
    require(isinstance(saved_rng, dict), "Last RNG state is missing")
    require(isinstance(loader_state, torch.Tensor), "Train-loader RNG state is missing")
    _finite_tree(saved_rng, path="last.rng_state")
    original_rng = capture_rng_state()
    try:
        restore_rng_state(saved_rng)
        require(_rng_equal(saved_rng, capture_rng_state()), "Saved RNG did not restore exactly")
        require(
            not saved_rng.get("cuda") or torch.cuda.is_available(),
            "CUDA RNG state cannot be verified without CUDA",
        )
        restore_rng_state(saved_rng)
        draw1: tuple[Any, ...] = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )
        if saved_rng.get("cuda"):
            draw1 += (torch.rand(4, device="cuda").cpu(),)
        restore_rng_state(saved_rng)
        draw2: tuple[Any, ...] = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )
        if saved_rng.get("cuda"):
            draw2 += (torch.rand(4, device="cuda").cpu(),)
        require(draw1[0] == draw2[0], "Python RNG continuation drift")
        require(np.array_equal(draw1[1], draw2[1]), "NumPy RNG continuation drift")
        require(
            all(torch.equal(left, right) for left, right in zip(draw1[2:], draw2[2:])),
            "Torch RNG continuation drift",
        )
    finally:
        restore_rng_state(original_rng)
    loader = torch.Generator(device="cpu")
    restore_train_loader_generator_state(loader, loader_state)
    order1 = torch.randperm(257, generator=loader)
    restore_train_loader_generator_state(loader, loader_state)
    order2 = torch.randperm(257, generator=loader)
    require(torch.equal(order1, order2), "Train-loader continuation order drift")

    return {
        "initial_state_sha256": summary["initial_state_sha256"],
        "best_checkpoint_file_sha256": sha256_file(best_path),
        "last_checkpoint_file_sha256": sha256_file(last_path),
        "history_file_sha256": sha256_file(history_path),
        "best_state_sha256": best_sha,
        "last_state_sha256": last_sha,
        "last_best_state_sha256": last_best_sha,
        "resume_identity_exact": True,
        "strict_best_model_restore": True,
        "strict_last_model_restore": True,
        "strict_optimizer_restore": True,
        "rng_state_exact_restore": True,
        "rng_next_draw_deterministic": True,
        "train_loader_next_order_deterministic": True,
        "alpha_raw": alpha_values,
        "alpha_optimizer_exp_avg_max_abs": float(exp_avg.abs().max()),
    }


def _validate_initial_state_against_e1(
    *, artifact_root: Path, dataset: str, observed_initial_sha256: str
) -> dict[str, Any]:
    from simple_lab_test.search.common.runner import torch_load_checkpoint

    e1_run = artifact_root / "jobs" / f"e1_{dataset}" / "runs" / BACKBONE / VARIANT / "seed_42"
    summary = read_json(e1_run / "summary.json")
    best_path = e1_run / "best_val_qty_rmse_model.pt"
    last_path = e1_run / "last_epoch_state.pt"
    best = torch_load_checkpoint(best_path, map_location="cpu")
    last = torch_load_checkpoint(last_path, map_location="cpu")
    require(isinstance(best, dict) and isinstance(last, dict), "e1 checkpoint invalid")
    e1_initial = summary.get("initial_state_sha256")
    require(
        isinstance(e1_initial, str)
        and len(e1_initial) == 64
        and best.get("initial_state_sha256") == e1_initial
        and last.get("initial_state_sha256") == e1_initial
        and observed_initial_sha256 == e1_initial,
        "Candidate initial state differs from the same-dataset e1 initialization",
    )
    e1_audit = read_json(artifact_root / "e1_audit_v4.json")
    e1_job = e1_audit.get("jobs", {}).get(dataset)
    require(
        isinstance(e1_job, Mapping)
        and e1_job.get("best_checkpoint_file_sha256") == sha256_file(best_path)
        and e1_job.get("last_checkpoint_file_sha256") == sha256_file(last_path),
        "Same-dataset e1 checkpoint evidence drift",
    )
    return {
        "matched": True,
        "initial_state_sha256": e1_initial,
        "e1_best_checkpoint_file_sha256": sha256_file(best_path),
        "e1_last_checkpoint_file_sha256": sha256_file(last_path),
    }


def _replay_selected_validation(
    *,
    run_dir: Path,
    contracts: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    registry_binding: Mapping[str, Any],
    dataset: str,
    source_root: Path,
    replay_device: str,
    reported_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay the selected checkpoint on the pinned validation set."""
    from paper.scripts.audit_hard_lmm_causal_qkv import _restore_model
    from paper.scripts.count_aware_tpp_backbone.core import (
        evaluate,
        load_train_validation_frame,
        prepare_count_frame,
    )
    from paper.scripts.run_count_aware_tpp_backbone_control import (
        exact_target_population,
    )
    from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
        aggregate_quantity_metrics,
    )
    from paper.scripts.run_taxi_quantity_interface_ablation import (
        make_loader,
        train_quantile_contract,
    )
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    require(replay_device in {"cpu", "cuda"}, "Replay device must be cpu or cuda")
    if replay_device == "cuda":
        require(torch.cuda.is_available(), "CUDA replay requested but CUDA is unavailable")
    raw_rows = {
        row.get("dataset"): row
        for row in contracts["raw_rmse_registry"].get("datasets", [])
        if isinstance(row, dict)
    }
    data_spec = raw_rows.get(dataset)
    require(isinstance(data_spec, Mapping), f"Replay data binding missing: {dataset}")
    data_path = _resolve_bound_path(
        source_root, data_spec.get("data_path"), label=f"{dataset} replay data"
    )
    manifest_path = _resolve_bound_path(
        source_root,
        data_spec.get("split_manifest_path"),
        label=f"{dataset} replay split manifest",
    )
    require(
        sha256_file(data_path) == candidate_binding.get("data_sha256")
        and sha256_file(manifest_path)
        == candidate_binding.get("split_manifest_sha256"),
        "Replay data or split-manifest digest drift",
    )
    raw_frame = load_train_validation_frame(data_path)
    require(
        set(raw_frame["chronological_split"].unique().to_list())
        == {"train", "validation"},
        "Replay frame contains an unexpected split or misses train/validation",
    )
    frame = prepare_count_frame(raw_frame)
    population = registry_binding["validation_target_population"]
    _, replay_population = exact_target_population(
        frame,
        target_split="validation",
        lookback_weeks=int(population["lookback_weeks"]),
        max_seq_len=int(population["max_seq_len"]),
    )
    require(replay_population == population, "Replay validation target population drift")
    quantity_contract = train_quantile_contract(raw_frame)
    launch = read_json(run_dir.parents[3] / "launch_contract.json")
    require(
        quantity_contract == launch.get("quantity_contract"),
        "Replay train-only quantity contract drift",
    )
    loader = make_loader(
        frame,
        target_split="validation",
        batch_size=128,
        lookback_weeks=int(population["lookback_weeks"]),
        max_seq_len=int(population["max_seq_len"]),
        shuffle=False,
        generator=None,
    )
    checkpoint_path = run_dir / "best_val_qty_rmse_model.pt"
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    require(isinstance(checkpoint, dict), "Replay selected checkpoint invalid")
    state = _state_dict(checkpoint, "model_state_dict", artifact="replay checkpoint")
    candidate = {"backbone": BACKBONE, "model_role": MODEL_ROLE}
    model = _restore_model(
        checkpoint, state, candidate=candidate, artifact="replay checkpoint"
    )
    before_sha = canonical_state_dict_sha256(model.state_dict())
    require(before_sha == checkpoint.get("model_state_sha256"), "Replay state digest drift")
    require(all(parameter.grad is None for parameter in model.parameters()), "Replay model has a gradient")
    model.requires_grad_(False).eval().to(torch.device(replay_device))
    started = time.perf_counter()
    evaluation = evaluate(
        model=model,
        loader=loader,
        quantity_contract=quantity_contract,
        device=replay_device,
        lambda_log_qty=1.0,
        max_batches=None,
        include_breakdowns=True,
    )
    elapsed = time.perf_counter() - started
    require(
        evaluation.get("evaluated_count") == population["target_count"],
        "Replay did not process the full validation population",
    )
    replay_metrics = aggregate_quantity_metrics(
        {
            "quantity_rows": evaluation["quantity_rows"],
            "best_val_qty_mae": evaluation["qty_mae"],
            "best_val_qty_rmse": evaluation["qty_rmse"],
            "best_val_time_nll": evaluation["val_time_nll"],
        },
        expected_count=int(population["target_count"]),
    )
    replay_metrics["legacy_time_nll"] = replay_metrics.pop("clamped_time_loss")
    require(
        replay_metrics["stratum_counts"] == registry_binding["stratum_counts"]
        and replay_metrics["body_target_count"] == registry_binding["body_target_count"],
        "Replay validation quantity strata drift",
    )
    rel_tol = REEVALUATION_REL_TOL if replay_device == "cuda" else CPU_REPLAY_REL_TOL
    abs_tol = REEVALUATION_ABS_TOL if replay_device == "cuda" else CPU_REPLAY_ABS_TOL
    for name in (
        "raw_rmse",
        "overall_mae",
        "body_mae",
        "gt_p99_mae",
        "legacy_time_nll",
    ):
        _require_reevaluated_metric_equal(
            replay_metrics[name],
            _finite_number(reported_metrics.get(name), label=f"reported.{name}"),
            label=f"validation replay {name}",
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    after_sha = canonical_state_dict_sha256(model.state_dict())
    require(after_sha == before_sha, "Validation replay changed selected model state")
    require(all(parameter.grad is None for parameter in model.parameters()), "Validation replay created a gradient")
    return {
        "status": "passed",
        "device": replay_device,
        "cuda_claim": replay_device == "cuda",
        "batch_size": 128,
        "elapsed_seconds": elapsed,
        "target_population": replay_population,
        "quantity_contract": quantity_contract,
        "metrics": replay_metrics,
        "reported_metric_rel_tol": rel_tol,
        "reported_metric_abs_tol": abs_tol,
        "model_state_before_sha256": before_sha,
        "model_state_after_sha256": after_sha,
        "model_state_unchanged": True,
        "all_parameter_gradients_none": True,
        "held_out_rows_loaded": False,
    }


def audit_screening_job(
    *,
    artifact_root: Path,
    job: Path,
    dataset: str,
    source_revision: str,
    expected_epochs: int,
    source_root: Path = ROOT,
    base_contract_path: Path = BASE_CONTRACT_PATH,
    amendment_path: Path = AMENDMENT_PATH,
    normalized_evidence_root: Path | None = None,
    replay_device: str = "cpu",
) -> dict[str, Any]:
    """Perform the complete strict audit for one VNC validation-only job."""
    require(source_revision == PINNED_SOURCE_REVISION, "Unexpected source revision")
    require(expected_epochs in SUPPORTED_EPOCH_BUDGETS, "Expected epochs must be 1 or 300")
    require(replay_device in {"cpu", "cuda"}, "Replay device must be cpu or cuda")
    require(
        expected_epochs != 300 or dataset == "insta_market_basket",
        "This frozen e300 auditor covers the authorized Instacart-first stage only",
    )
    require(
        expected_epochs != 300 or replay_device == "cuda",
        "Final e300 audit requires --replay-device cuda",
    )
    source_root = source_root.resolve()
    preexisting_imports = _validate_project_import_provenance(
        source_root,
        phase="before_source_activation",
    )
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    artifact_root = artifact_root.resolve()
    job = job if job.is_absolute() else artifact_root / job
    job = job.resolve()
    require(job.is_dir(), f"Job directory is missing: {job}")
    expected_job_name = (
        f"e1_{dataset}" if expected_epochs == 1 else f"seed42_screening_{dataset}"
    )
    expected_job = (artifact_root / "jobs" / expected_job_name).resolve()
    require(
        job == expected_job
        and job.parent == (artifact_root / "jobs").resolve(),
        "Job path is outside the exact artifact jobs route",
    )

    contracts = load_screening_contracts(
        source_root=source_root,
        base_contract_path=base_contract_path,
        amendment_path=amendment_path,
        normalized_evidence_root=normalized_evidence_root,
    )
    candidate_binding, registry_binding = _candidate_dataset_binding(contracts, dataset)
    provenance = _validate_artifact_provenance(
        artifact_root=artifact_root,
        contracts=contracts,
        dataset=dataset,
        candidate_binding=candidate_binding,
        source_revision=source_revision,
        source_root=source_root,
        expected_epochs=expected_epochs,
    )
    host_preflight = _validate_host_preflight(
        artifact_root=artifact_root,
        dataset=dataset,
        expected_epochs=expected_epochs,
    )
    candidate = {"backbone": BACKBONE, "model_role": MODEL_ROLE}

    from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
        audit_job,
    )

    base_audit = audit_job(
        job,
        candidate=candidate,
        dataset=dataset,
        source_revision=source_revision,
        expected_epochs=expected_epochs,
        population_reference=registry_binding["validation_target_population"],
    )
    summary_path = Path(str(base_audit["summary"]))
    # audit_job records an absolute path created on the execution host.  The
    # current job path remains canonical when an artifact tree is copied.
    run_dir = (
        job / "runs" / BACKBONE / VARIANT / "seed_42"
    )
    require(run_dir.is_dir(), "Canonical VNC run route is missing")
    require(summary_path.name == "summary.json", "Summary route is malformed")
    summary = read_json(run_dir / "summary.json")
    route = _validate_route_and_training_identity(
        job=job,
        summary=summary,
        candidate_binding=candidate_binding,
        registry_binding=registry_binding,
        dataset=dataset,
        source_revision=source_revision,
        expected_epochs=expected_epochs,
    )
    history_payload = read_json(run_dir / "history.json")
    history = history_payload.get("history")
    require(isinstance(history, list), "History artifact is malformed")
    require(
        len(history) == summary.get("completed_epochs"),
        "History/completed epoch count drift",
    )
    stopped_early = summary.get("stopped_early")
    require(isinstance(stopped_early, bool), "Summary stopped_early must be boolean")
    expected_train = int(candidate_binding["train_target_count"])
    require(
        expected_train > 0
        and all(row.get("train_event_count") == expected_train for row in history),
        "An epoch did not process the full train population",
    )
    require(
        all(row.get("train_all_finite") is True for row in history),
        "An epoch reports non-finite training",
    )
    _finite_tree(history, path="history")
    selector = validate_history_and_stop(
        history,
        expected_epochs=expected_epochs,
        completed_epochs=int(summary["completed_epochs"]),
        stopped_early=stopped_early,
        minimum_epochs=int(route["minimum_epochs"]),
        patience=int(route["patience"]),
    )
    require(
        selector["selected_epoch"] == int(summary["best_epoch"]),
        "Summary best epoch differs from replayed selector",
    )
    require(
        base_audit["metrics"].get("stratum_counts")
        == registry_binding.get("stratum_counts")
        and base_audit["metrics"].get("body_target_count")
        == registry_binding.get("body_target_count"),
        "Validation quantity stratum population drift",
    )
    checkpoints = _validate_checkpoints_and_resume(
        run_dir=run_dir,
        summary=summary,
        candidate=candidate,
        source_revision=source_revision,
        history=history,
        selector=selector,
    )
    initial_state = _validate_initial_state_against_e1(
        artifact_root=artifact_root,
        dataset=dataset,
        observed_initial_sha256=checkpoints["initial_state_sha256"],
    )

    reported_metrics = dict(base_audit["metrics"])
    reported_metrics["legacy_time_nll"] = reported_metrics.pop("clamped_time_loss")
    validation_replay = _replay_selected_validation(
        run_dir=run_dir,
        contracts=contracts,
        candidate_binding=candidate_binding,
        registry_binding=registry_binding,
        dataset=dataset,
        source_root=source_root,
        replay_device=replay_device,
        reported_metrics=reported_metrics,
    )
    metrics = validation_replay["metrics"]
    baseline = contracts["base"].get("B_seed42_validation_references", {}).get(dataset)
    require(isinstance(baseline, Mapping), f"B seed42 reference missing: {dataset}")
    stage1 = evaluate_stage1_gate(metrics, baseline, expected_epochs=expected_epochs)
    normalized_reference = contracts["amendment"].get("normalized_time_references", {}).get(dataset)
    require(isinstance(normalized_reference, Mapping), "Normalized reference missing")
    if expected_epochs == 1:
        normalized_status = "not_applicable_e1"
    elif stage1["status"] == "passed":
        normalized_status = "ready_for_separate_normalized_fit"
    else:
        normalized_status = "blocked_by_stage_1_failure"
    normalized = {
        "status": normalized_status,
        "evaluated": False,
        "candidate_normalized_nll": None,
        "aligned_B_primary_nll": normalized_reference["aligned_B_primary_nll"],
        "candidate_gate_max": normalized_reference[
            "candidate_gate_max_aligned_B_plus_0_01"
        ],
        "observation_likelihood_mode": normalized_reference[
            "observation_likelihood_mode"
        ],
    }
    loaded_imports = _validate_project_import_provenance(
        source_root,
        phase="after_validation_replay",
    )
    return {
        "schema_version": 1,
        "audit_id": "hard_lmm_value_norm_screening_strict_v1",
        "status": "passed",
        "dataset": dataset,
        "candidate": candidate,
        "source_revision": source_revision,
        "epoch_budget": expected_epochs,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "artifact_root": str(artifact_root),
        "job": str(job),
        "contract_digests": contracts["digests"],
        "project_import_provenance": {
            "before_source_activation": preexisting_imports,
            "after_validation_replay": loaded_imports,
        },
        "provenance": provenance,
        "host_preflight": host_preflight,
        "route": route,
        "population": {
            "full_train_target_count_per_epoch": expected_train,
            "full_validation_target_count": int(
                candidate_binding["validation_target_count"]
            ),
            "validation_target_identity_sha256": candidate_binding[
                "validation_target_identity_sha256"
            ],
            "validation_target_quantity_sha256": candidate_binding[
                "validation_target_quantity_sha256"
            ],
            "stratum_counts": registry_binding["stratum_counts"],
            "body_target_count": registry_binding["body_target_count"],
        },
        "history_and_selector": selector,
        "checkpoints_and_resume": checkpoints,
        "initial_state_e1_identity": initial_state,
        "selected_checkpoint_validation_replay": validation_replay,
        "reported_metrics": reported_metrics,
        "metrics": metrics,
        "stage_1_gate": stage1,
        "normalized_time_gate": normalized,
        "normalized_fit_authorized_by_stage1": (
            expected_epochs == 300 and stage1["status"] == "passed"
        ),
        "taxi_intermittent_authorized": False,
        "additional_seeds_authorized": False,
        "taxi_intermittent_authorization_prerequisite": (
            "separate normalized-duration fit must pass the amended full Instacart gate"
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", "--artifact", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=(
            "insta_market_basket",
            "yellow_trip_hourly",
            "intermittent_frozen_5000",
        ),
        required=True,
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-epochs", type=int, choices=SUPPORTED_EPOCH_BUDGETS, required=True)
    parser.add_argument(
        "--replay-device",
        choices=("cpu", "cuda"),
        required=True,
        help="e300 final audits require cuda; cpu is allowed only for e1 development replay.",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument(
        "--normalized-evidence-root",
        type=Path,
        required=True,
        help=(
            "Root containing the amendment's aligned-B decision, summaries, "
            "and checkpoints."
        ),
    )
    parser.add_argument(
        "--write-output",
        type=Path,
        help="Atomically write the audit JSON here; omitted means read-only stdout.",
    )
    return parser.parse_args()


def main() -> None:
    require(
        __package__ in (None, "") and __spec__ is None,
        "Strict auditor CLI must be invoked by its direct file path, not with -m",
    )
    args = parse_args()
    result = audit_screening_job(
        artifact_root=args.artifact_root,
        job=args.job,
        dataset=args.dataset,
        source_revision=args.source_revision,
        expected_epochs=args.expected_epochs,
        source_root=args.source_root,
        base_contract_path=args.base_contract,
        amendment_path=args.amendment,
        normalized_evidence_root=args.normalized_evidence_root,
        replay_device=args.replay_device,
    )
    if args.write_output is not None:
        _write_json(args.write_output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AMENDMENT_PATH",
    "AMENDMENT_SHA256",
    "BASE_CONTRACT_PATH",
    "BASE_CONTRACT_SHA256",
    "PINNED_SOURCE_REVISION",
    "audit_screening_job",
    "evaluate_stage1_gate",
    "load_screening_contracts",
    "validate_history_and_stop",
]
