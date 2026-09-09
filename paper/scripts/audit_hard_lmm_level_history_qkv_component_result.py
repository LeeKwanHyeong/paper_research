#!/usr/bin/env python3
"""Independently harden the frozen LPHC component diagnostic result.

This post-run audit performs no fitting or prediction.  It proves the executed
runner's Git provenance, replays the exact train-row identity and historical
fold assignment, and checks every frozen project module loaded by the replay
against the pinned deployment manifest.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_RELATIVE = "paper/scripts/audit_hard_lmm_level_history_qkv_components.py"
RUNNER_COMMIT = "3f5bff7cb1a71c49dfb213a9b6bf9eaf2f2b0f43"
RUNNER_SHA256 = "26dde78c8fff2802082e2f805bd472f3611b049bbf6ad7961e7e94f12a86ef37"
CONTRACT_SHA256 = "4b8c982450d775ed1b10434f853b45392166b2a3cf95b7e53c62ba8eb1e724d4"
PRIOR_ANALYZER_RELATIVE = "paper/scripts/analyze_hard_lmm_bounded_qk_train_condition.py"
PRIOR_ANALYZER_COMMIT = "1930e5fbfe123da606af3746b002f792a3934071"
PRIOR_ANALYZER_SHA256 = "1c774a3f8a1c21721e7b069f73bf1c0b85fb8dff43f2508ae50f98e9b34d1544"
FOLD_SALT = "hard_lmm_bounded_qk_train_condition_v1:20260908"
VARIANT_ORDER = (
    "residual_off", "q_only", "k_only", "v_only", "qk", "qv", "kv", "full_qkv"
)
ARRAY_HASH_NAMESPACE = b"hard_lmm_level_history_qkv_component_result_v1\0"
EXPECTED_RAW_ARRAY_SHA256 = {
    "target_index": "a9bdd26205b2b030c70cc951198ae2fb7e04f5658077d1180a0a90d682ec571e",
    "series_index": "f343a89f392a26e1c8f0a8964735cc940268d678dffe82ea8fb2b9c249fccea8",
    "context_end": "464928f7fdc1f661b8a3cfae88a66c8d1114272920a0edf55b9da5680336b2f2",
    "fold": "6816545e0c519b578bf5383a8f90b81ba7b60f72cb4ac6a15f6a1e0cfb9df72c",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(ARRAY_HASH_NAMESPACE)
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_blob(relative: str, revision: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def verify_runner_provenance() -> dict[str, Any]:
    resolved = subprocess.run(
        ["git", "rev-parse", f"{RUNNER_COMMIT}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    require(resolved == RUNNER_COMMIT, "Execution commit does not resolve exactly")
    runner = PROJECT_ROOT / RUNNER_RELATIVE
    analyzer = PROJECT_ROOT / PRIOR_ANALYZER_RELATIVE
    require(sha256_file(runner) == RUNNER_SHA256, "Local executed runner bytes drifted")
    require(sha256_bytes(git_blob(RUNNER_RELATIVE, RUNNER_COMMIT)) == RUNNER_SHA256,
            "Execution commit does not contain the executed runner bytes")
    require(sha256_file(analyzer) == PRIOR_ANALYZER_SHA256, "Prior fold analyzer drifted")
    require(sha256_bytes(git_blob(PRIOR_ANALYZER_RELATIVE, PRIOR_ANALYZER_COMMIT))
            == PRIOR_ANALYZER_SHA256,
            "Prior analyzer commit does not contain the pinned fold implementation")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", RUNNER_RELATIVE, PRIOR_ANALYZER_RELATIVE],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    require(not dirty.strip(), "Runner or prior analyzer has uncommitted changes")
    return {
        "execution_commit": RUNNER_COMMIT,
        "runner_path": RUNNER_RELATIVE,
        "runner_sha256": RUNNER_SHA256,
        "runner_matches_commit": True,
        "prior_fold_analyzer_path": PRIOR_ANALYZER_RELATIVE,
        "prior_fold_analyzer_commit": PRIOR_ANALYZER_COMMIT,
        "prior_fold_analyzer_sha256": PRIOR_ANALYZER_SHA256,
        "prior_fold_analyzer_matches_commit": True,
        "tracked_files_clean": True,
    }


def load_prior_fold_module() -> Any:
    path = PROJECT_ROOT / PRIOR_ANALYZER_RELATIVE
    spec = importlib.util.spec_from_file_location("_pinned_bounded_qk_fold_analyzer", path)
    require(spec is not None and spec.loader is not None, "Cannot load prior fold analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.FOLD_SALT == FOLD_SALT, "Prior fold salt drifted")
    return module


def load_pinned_component_runner() -> Any:
    path = PROJECT_ROOT / RUNNER_RELATIVE
    spec = importlib.util.spec_from_file_location("_pinned_lphc_component_runner", path)
    require(spec is not None and spec.loader is not None, "Cannot load component runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.CONTRACT_SHA256 == CONTRACT_SHA256, "Pinned runner contract hash drifted")
    require(tuple(module.VARIANT_ORDER) == VARIANT_ORDER, "Pinned runner variant order drifted")
    return module


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _module_source_path(module: Any) -> Path | None:
    raw = getattr(module, "__file__", None)
    if raw is None:
        return None
    path = Path(raw).resolve()
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path))).resolve()
        except ValueError:
            return None
    return path if path.suffix == ".py" else None


def _assert_no_mutable_project_module(frozen_root: Path) -> None:
    allowed_current = {
        Path(__file__).resolve(),
        (PROJECT_ROOT / PRIOR_ANALYZER_RELATIVE).resolve(),
        (PROJECT_ROOT / RUNNER_RELATIVE).resolve(),
    }
    project_prefixes = ("models", "data_loader", "simple_lab_test", "paper", "utils")
    for name, module in tuple(sys.modules.items()):
        source = _module_source_path(module)
        if source is None or not source.is_file() or source in allowed_current:
            continue
        under_mutable_root = _inside(source, PROJECT_ROOT)
        project_named = any(name == prefix or name.startswith(prefix + ".") for prefix in project_prefixes)
        if under_mutable_root or project_named:
            require(_inside(source, frozen_root), f"Project module escaped frozen source: {name} -> {source}")


def import_frozen_runtime(frozen_root: Path) -> dict[str, Any]:
    frozen_root = frozen_root.resolve()
    require(frozen_root.is_dir() and frozen_root != PROJECT_ROOT, "Frozen root is invalid")
    _assert_no_mutable_project_module(frozen_root)
    sys.path[:] = [
        entry for entry in sys.path
        if entry and not _inside(Path(entry).resolve(), PROJECT_ROOT)
    ]
    sys.path.insert(0, str(frozen_root))
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    control = importlib.import_module("paper.scripts.run_count_aware_tpp_backbone_control")
    loader = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
    importlib.import_module("paper.scripts.run_matched_frozen_lognormal_duration")
    importlib.import_module("simple_lab_test.search.common.runner")
    importlib.import_module("models.TPPs.CountAwareFactory")
    importlib.import_module("paper.scripts.audit_hard_lmm_causal_qkv")
    _assert_no_mutable_project_module(frozen_root)
    return {
        "prepare_count_frame": core.prepare_count_frame,
        "exact_target_population": control.exact_target_population,
        "make_loader": loader.make_loader,
    }


def verify_transitive_frozen_modules(
    frozen_root: Path, deployment: Mapping[str, Any]
) -> dict[str, Any]:
    source_files = deployment.get("source_files")
    require(isinstance(source_files, Mapping), "Deployment source_files missing")
    observed: dict[str, dict[str, str]] = {}
    for name, module in sorted(sys.modules.items()):
        path = _module_source_path(module)
        if path is None or not _inside(path, frozen_root):
            continue
        relative = path.relative_to(frozen_root.resolve()).as_posix()
        actual = sha256_file(path)
        expected = source_files.get(relative)
        require(isinstance(expected, str), f"Imported frozen module absent from manifest: {relative}")
        require(actual == expected, f"Imported frozen module hash drift: {relative}")
        observed[name] = {"path": relative, "sha256": actual}
    required_restore = "paper/scripts/audit_hard_lmm_causal_qkv.py"
    require(any(row["path"] == required_restore for row in observed.values()),
            "Frozen _restore_model module was not included in transitive verification")
    require(len(observed) >= 7, "Too few frozen project modules were audited")
    return {
        "all_loaded_project_python_modules_match_deployment_manifest": True,
        "loaded_module_count": len(observed),
        "required_restore_module_verified": True,
        "modules": observed,
    }


def verify_component_arrays(
    component: Mapping[str, np.ndarray], *, expected_count: int
) -> dict[str, Any]:
    required = {
        "quantity", "history_length", "series_index", "fold",
        *(f"prediction__{name}" for name in VARIANT_ORDER),
    }
    require(set(component) == required, "Component prediction schema drift")
    require(all(value.shape == (expected_count,) for value in component.values()),
            "Component prediction shape drift")
    require(component["quantity"].dtype == np.float64, "Component quantity dtype drift")
    for name in ("history_length", "series_index"):
        require(component[name].dtype == np.int64, f"Component {name} dtype drift")
    require(component["fold"].dtype == np.int8, "Component fold dtype drift")
    for name in VARIANT_ORDER:
        value = component[f"prediction__{name}"]
        require(value.dtype == np.float64, f"Component prediction dtype drift: {name}")
        require(bool(np.isfinite(value).all()) and bool((value >= 0).all()),
                f"Invalid component predictions: {name}")
    require(bool(np.isfinite(component["quantity"]).all()), "Non-finite component quantity")
    return {
        "schema_exact": True,
        "target_count": expected_count,
        "all_shapes_exact": True,
        "all_dtypes_exact": True,
        "all_predictions_finite_and_nonnegative": True,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    require(bool(rows), "Cannot serialize empty metric rows")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def verify_recomputed_analysis(
    *,
    component_runner: Any,
    component: Mapping[str, np.ndarray],
    b_prediction: np.ndarray,
    contract: Mapping[str, Any],
    published_analysis: Mapping[str, Any],
    fold_metrics_path: Path,
    history_metrics_path: Path,
) -> dict[str, Any]:
    variants = {name: component[f"prediction__{name}"] for name in VARIANT_ORDER}
    recomputed, fold_rows, history_rows = component_runner.analyze_predictions(
        target=component["quantity"],
        history=component["history_length"],
        folds=component["fold"],
        b_prediction=b_prediction,
        variants=variants,
        contract=contract,
    )
    require(recomputed["metrics"] == published_analysis.get("metrics"),
            "Published component metrics differ from exact replay")
    require(recomputed["decision"] == published_analysis.get("decision"),
            "Published component decision differs from exact replay")
    require(_csv_bytes(fold_rows) == fold_metrics_path.read_bytes(),
            "Published fold-metrics CSV differs from exact replay")
    require(_csv_bytes(history_rows) == history_metrics_path.read_bytes(),
            "Published history-metrics CSV differs from exact replay")
    return {
        "metrics_exact_replay": True,
        "decision_exact_replay": True,
        "fold_metrics_csv_exact_replay": True,
        "history_metrics_csv_exact_replay": True,
        "fold_metric_rows": len(fold_rows),
        "history_metric_rows": len(history_rows),
    }


def verify_scopes(analysis: Mapping[str, Any], run_audit: Mapping[str, Any]) -> dict[str, Any]:
    for artifact, scope in (("analysis", analysis.get("scope")), ("audit", run_audit.get("scope"))):
        require(isinstance(scope, Mapping), f"{artifact} scope missing")
        require(scope.get("target_split") == "train", f"{artifact} target split drift")
        require(scope.get("input_splits_materialized") == ["train"],
                f"{artifact} materialized non-train input")
        for flag in ("validation_targets", "held_out_test", "training", "checkpoint_selection",
                     "parameter_updates", "optimizer_use", "calibration_fit"):
            require(scope.get(flag) is False, f"{artifact} enabled forbidden operation: {flag}")
    require(run_audit.get("training_performed") is False, "Run audit reports training")
    require(run_audit.get("checkpoint_selection_performed") is False,
            "Run audit reports checkpoint selection")
    require(run_audit.get("validation_targets_evaluated") is False,
            "Run audit reports validation access")
    require(run_audit.get("held_out_test_evaluated") is False,
            "Run audit reports held-out access")
    return {
        "target_split": "train",
        "materialized_splits": ["train"],
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "validation_targets_evaluated": False,
        "held_out_test_evaluated": False,
    }


def verify_run_audit_evidence(
    run_audit: Mapping[str, Any], *, contract: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    require(run_audit.get("schema") == "hard_lmm_level_history_qkv_component_audit_v1",
            "Run-audit schema drift")
    require(run_audit["contract"]["sha256"] == CONTRACT_SHA256, "Run contract hash drift")
    data = run_audit["data"]
    require(Path(data["path"]).resolve() == args.data.resolve(), "Run data path drift")
    require(data["sha256"] == contract["dataset"]["data_sha256"], "Run data hash drift")
    require(Path(data["split_manifest_path"]).resolve() == args.split_manifest.resolve(),
            "Run split-manifest path drift")
    require(data["split_manifest_sha256"] == contract["dataset"]["split_manifest_sha256"],
            "Run split-manifest hash drift")
    require(data["materialized_splits"] == ["train"], "Run materialized non-train data")

    frozen = run_audit["frozen_source"]
    require(Path(frozen["root"]).resolve() == args.frozen_source_root.resolve(),
            "Run frozen-root path drift")
    require(frozen["revision"] == contract["source"]["frozen_model_revision"],
            "Run frozen revision drift")
    require(frozen["deployment_manifest_sha256"]
            == contract["source"]["deployment_manifest_sha256"],
            "Run deployment-manifest hash drift")
    require(frozen["verified_files"] == contract["source"]["required_source_file_sha256"],
            "Run frozen direct-source audit drift")

    b = run_audit["B_cache"]
    require(Path(b["path"]).resolve() == args.b_cache.resolve(), "Run B cache path drift")
    require(Path(b["result_path"]).resolve() == args.b_result.resolve(), "Run B result path drift")
    require(b["sha256"] == contract["models"]["B"]["train_cache_file_sha256"],
            "Run B cache hash drift")
    require(b["result_sha256"] == contract["models"]["B"]["train_result_file_sha256"],
            "Run B result hash drift")

    checkpoint = run_audit["LPHC_checkpoint"]
    lphc = contract["models"]["LPHC"]
    require(checkpoint["checkpoint_file_sha256"] == lphc["checkpoint_file_sha256"],
            "Run LPHC checkpoint file hash drift")
    require(checkpoint["source_state_sha256"] == lphc["checkpoint_state_sha256"],
            "Run LPHC checkpoint state hash drift")
    require(checkpoint["summary_file_sha256"] == lphc["summary_file_sha256"],
            "Run LPHC summary hash drift")

    variants = run_audit.get("variants")
    before = run_audit.get("state_sha256_before")
    after = run_audit.get("state_sha256_after")
    invariance = run_audit.get("first_batch_target_and_padding_prediction_invariance")
    require(isinstance(variants, Mapping) and set(variants) == set(VARIANT_ORDER),
            "Run variant audit schema drift")
    require(isinstance(before, Mapping) and set(before) == set(VARIANT_ORDER),
            "Run before-state schema drift")
    require(isinstance(after, Mapping) and set(after) == set(VARIANT_ORDER),
            "Run after-state schema drift")
    require(before == after, "Run before/after state hashes differ")
    require(isinstance(invariance, Mapping) and set(invariance) == set(VARIANT_ORDER),
            "Run invariance audit schema drift")
    require(all(value is True for value in invariance.values()),
            "Run audit found target or padding leakage")
    kernel_keys = lphc["kernel_state_keys"]
    common_digest = checkpoint["common_non_kernel_state_sha256"]
    for name in VARIANT_ORDER:
        row = variants[name]
        mask = list(contract["variants"][name])
        require(row["mask"] == mask, f"Run variant mask drift: {name}")
        require(row["derived_state_sha256"] == before[name], f"Run variant state drift: {name}")
        require(row["common_non_kernel_state_sha256"] == common_digest,
                f"Run non-kernel state drift: {name}")
        require(set(row["kernel_checks"]) == {"Q", "K", "V"},
                f"Run kernel-role schema drift: {name}")
        expected_changed = sorted(
            kernel_keys[role] for role, enabled in zip(("Q", "K", "V"), mask, strict=True)
            if not enabled
        )
        require(row["changed_state_keys"] == expected_changed,
                f"Run changed-state keys drift: {name}")
        for role, enabled in zip(("Q", "K", "V"), mask, strict=True):
            check = row["kernel_checks"][role]
            require(check["enabled"] is bool(enabled), f"Run kernel switch drift: {name}/{role}")
            require(check["shape"] == lphc["kernel_shape"], f"Run kernel shape drift: {name}/{role}")
            require(check["source_nonzero_count"] == 128, f"Run source kernel zero drift: {name}/{role}")
            require(check["derived_nonzero_count"] == (128 if enabled else 0),
                    f"Run derived kernel sparsity drift: {name}/{role}")
    require(run_audit.get("state_unchanged") is True, "Run audit did not prove state identity")
    require(run_audit.get("all_gradients_absent") is True, "Run audit found model gradients")
    require(run_audit["runtime"]["device"] == "cpu" and run_audit["runtime"]["batch_size"] == 512,
            "Run runtime contract drift")
    return {
        "schema_exact": True,
        "contract_data_B_and_checkpoint_exact": True,
        "variant_masks_and_state_hashes_exact": True,
        "state_unchanged": True,
        "all_gradients_absent": True,
        "all_first_batch_invariance_checks_passed": True,
    }


def verify_row_identity(
    *,
    b_cache: Mapping[str, np.ndarray],
    component: Mapping[str, np.ndarray],
    expected_parts: np.ndarray,
    expected_context: np.ndarray,
    require_production_hashes: bool = False,
) -> dict[str, Any]:
    count = len(expected_parts)
    require(np.array_equal(b_cache["target_index"], np.arange(count, dtype=np.int64)),
            "B target_index is not the canonical contiguous order")
    require(np.array_equal(b_cache["series_index"], expected_parts),
            "B series_index does not match canonical dataset.index")
    require(np.array_equal(b_cache["context_end"], expected_context),
            "B context_end does not match canonical dataset.index")
    require(np.array_equal(component["series_index"], expected_parts),
            "Component series_index does not match canonical dataset.index")
    require(np.array_equal(component["quantity"], b_cache["quantity"]),
            "Component quantity does not match B row order")
    require(np.array_equal(component["history_length"], b_cache["history_length"]),
            "Component history does not match B row order")
    raw_hashes = {
        "target_index": sha256_bytes(np.ascontiguousarray(b_cache["target_index"]).tobytes()),
        "series_index": sha256_bytes(np.ascontiguousarray(expected_parts).tobytes()),
        "context_end": sha256_bytes(np.ascontiguousarray(expected_context).tobytes()),
    }
    if require_production_hashes:
        require(raw_hashes == {key: EXPECTED_RAW_ARRAY_SHA256[key] for key in raw_hashes},
                "Canonical row-identity raw hashes drifted")
    return {
        "target_count": count,
        "B_target_index_is_arange": True,
        "B_series_index_matches_canonical_dataset_index": True,
        "B_context_end_matches_canonical_dataset_index": True,
        "component_series_index_matches_canonical_dataset_index": True,
        "component_quantity_matches_B": True,
        "component_history_matches_B": True,
        "raw_c_byte_sha256": raw_hashes,
        "array_sha256": {
            "series_index": array_sha256(expected_parts, label="series_index"),
            "context_end": array_sha256(expected_context, label="context_end"),
            "target_index": array_sha256(b_cache["target_index"], label="target_index"),
        },
    }


def verify_fold_identity(
    *,
    observed: np.ndarray,
    series_ids: Sequence[Any],
    prior_fold_module: Any,
    require_production_hash: bool = False,
) -> dict[str, Any]:
    expected = prior_fold_module.assign_series_folds(series_ids, salt=FOLD_SALT)
    require(expected.dtype == np.int8 and observed.dtype == np.int8, "Fold dtype drift")
    require(np.array_equal(observed, expected), "Component fold membership differs from prior analyzer")
    raw_sha = sha256_bytes(np.ascontiguousarray(observed).tobytes())
    if require_production_hash:
        require(raw_sha == EXPECTED_RAW_ARRAY_SHA256["fold"], "Canonical fold raw hash drifted")
    per_series: dict[str, int] = {}
    for series, fold in zip(series_ids, observed, strict=True):
        token = prior_fold_module._series_token(series)
        prior = per_series.setdefault(token, int(fold))
        require(prior == int(fold), "A series spans both analysis folds")
    return {
        "fold_salt": FOLD_SALT,
        "prior_analyzer_exact_array_equality": True,
        "series_disjoint": True,
        "raw_c_byte_sha256": raw_sha,
        "fold_array_sha256": array_sha256(observed, label="fold"),
        "recomputed_fold_array_sha256": array_sha256(expected, label="fold"),
        "fold_target_counts": {
            str(fold): int(np.count_nonzero(observed == fold)) for fold in (0, 1)
        },
        "fold_series_counts": {
            str(fold): sum(1 for value in per_series.values() if value == fold) for fold in (0, 1)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    provenance = verify_runner_provenance()
    require(sha256_file(args.contract) == CONTRACT_SHA256, "Component contract hash drift")
    contract = read_json(args.contract)
    require(contract.get("contract_id") == "hard_lmm_level_history_qkv_component_diagnostic_v1",
            "Wrong component contract")
    analysis = read_json(args.analysis)
    run_audit = read_json(args.run_audit)
    b_result = read_json(args.b_result)
    require(analysis.get("status") == "complete", "Component analysis is incomplete")
    require(run_audit.get("status") == "passed", "Component run audit did not pass")
    scope = verify_scopes(analysis, run_audit)
    run_evidence = verify_run_audit_evidence(run_audit, contract=contract, args=args)
    require(Path(run_audit["contract"]["path"]).resolve() == args.contract.resolve(),
            "Run-audit contract path differs from the audited input")
    require(Path(run_audit["prediction_cache"]["path"]).resolve()
            == args.component_predictions.resolve(),
            "Run-audit prediction path differs from the audited input")
    for name, path in {
        "analysis.json": args.analysis,
        "fold_metrics.csv": args.fold_metrics,
        "history_metrics.csv": args.history_metrics,
    }.items():
        require(Path(run_audit["outputs"][name]["path"]).resolve() == path.resolve(),
                f"Run-audit output path differs from audited input: {name}")
    require(sha256_file(args.analysis) == run_audit["outputs"]["analysis.json"]["sha256"],
            "Analysis output hash drift")
    require(sha256_file(args.fold_metrics) == run_audit["outputs"]["fold_metrics.csv"]["sha256"],
            "Fold-metrics hash drift")
    require(sha256_file(args.history_metrics) == run_audit["outputs"]["history_metrics.csv"]["sha256"],
            "History-metrics hash drift")
    require(sha256_file(args.component_predictions)
            == run_audit["prediction_cache"]["sha256"], "Prediction-cache hash drift")
    require(sha256_file(args.b_cache) == contract["models"]["B"]["train_cache_file_sha256"],
            "B cache file hash drift")
    require(sha256_file(args.b_result) == contract["models"]["B"]["train_result_file_sha256"],
            "B result file hash drift")
    with args.fold_metrics.open(encoding="utf-8", newline="") as stream:
        fold_rows = list(csv.DictReader(stream))
    with args.history_metrics.open(encoding="utf-8", newline="") as stream:
        history_rows = list(csv.DictReader(stream))
    require(len(fold_rows) == 18, "Fold-metrics row count drift")
    require(len(history_rows) == 162, "History-metrics row count drift")

    deployment_path = PROJECT_ROOT / contract["source"]["deployment_manifest_path"]
    require(sha256_file(deployment_path) == contract["source"]["deployment_manifest_sha256"],
            "Deployment manifest hash drift")
    deployment = read_json(deployment_path)
    require(deployment.get("source_revision") == contract["source"]["frozen_model_revision"],
            "Deployment source revision drift")
    prior_fold_module = load_prior_fold_module()
    component_runner = load_pinned_component_runner()
    runtime = import_frozen_runtime(args.frozen_source_root)

    require(sha256_file(args.data) == contract["dataset"]["data_sha256"], "Data hash drift")
    require(sha256_file(args.split_manifest) == contract["dataset"]["split_manifest_sha256"],
            "Split-manifest hash drift")
    frame = (
        pl.scan_parquet(args.data)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(set(frame["chronological_split"].unique().to_list()) == {"train"},
            "A non-train row was materialized")
    frame = runtime["prepare_count_frame"](frame)
    target_quantity, population = runtime["exact_target_population"](
        frame, target_split="train", lookback_weeks=52, max_seq_len=64
    )
    for key, expected in {
        "target_count": contract["dataset"]["expected_train_targets"],
        "target_identity_sha256": contract["dataset"]["expected_train_identity_sha256"],
        "target_quantity_sha256": contract["dataset"]["expected_train_quantity_sha256"],
    }.items():
        require(population.get(key) == expected, f"Canonical population drift: {key}")
    loader = runtime["make_loader"](
        frame,
        target_split="train",
        batch_size=512,
        lookback_weeks=52,
        max_seq_len=64,
        shuffle=False,
        generator=None,
    )
    dataset = loader.dataset
    count = len(dataset)
    expected_parts = np.fromiter((part for part, _ in dataset.index), dtype=np.int64, count=count)
    expected_context = np.fromiter((end for _, end in dataset.index), dtype=np.int64, count=count)
    with np.load(args.b_cache, allow_pickle=False) as archive:
        b_cache = {name: archive[name].copy() for name in archive.files}
    with np.load(args.component_predictions, allow_pickle=False) as archive:
        component = {name: archive[name].copy() for name in archive.files}
    component_arrays = verify_component_arrays(component, expected_count=count)
    require(np.array_equal(target_quantity, b_cache["quantity"]), "Canonical target quantity drift")
    b_manifest = b_result["cache"]["train"]["arrays"]
    require(set(b_manifest) == set(b_cache), "B cache/result array schema drift")
    for label, value in b_cache.items():
        expected = b_manifest[label]
        require(list(value.shape) == expected["shape"] and str(value.dtype) == expected["dtype"],
                f"B array metadata drift: {label}")
        require(prior_fold_module.frozen_b_array_sha256(value, label=label) == expected["sha256"],
                f"B array manifest hash drift: {label}")
    row_identity = verify_row_identity(
        b_cache=b_cache,
        component=component,
        expected_parts=expected_parts,
        expected_context=expected_context,
        require_production_hashes=True,
    )
    series_parts = np.asarray(dataset.parts, dtype=object)
    series_ids = series_parts[expected_parts]
    fold_identity = verify_fold_identity(
        observed=component["fold"],
        series_ids=series_ids,
        prior_fold_module=prior_fold_module,
        require_production_hash=True,
    )
    require(fold_identity["fold_target_counts"] == contract["analysis"]["expected_fold_target_counts"],
            "Fold target counts differ from contract")
    require(fold_identity["fold_series_counts"] == contract["analysis"]["expected_fold_series_counts"],
            "Fold series counts differ from contract")
    replay = verify_recomputed_analysis(
        component_runner=component_runner,
        component=component,
        b_prediction=b_cache["prediction"],
        contract=contract,
        published_analysis=analysis,
        fold_metrics_path=args.fold_metrics,
        history_metrics_path=args.history_metrics,
    )
    transitive = verify_transitive_frozen_modules(args.frozen_source_root.resolve(), deployment)
    return {
        "schema": "hard_lmm_level_history_qkv_component_result_audit_v1",
        "status": "passed",
        "created_at_utc": utc_now(),
        "runner_provenance": provenance,
        "contract": {"path": str(args.contract.resolve()), "sha256": CONTRACT_SHA256},
        "scope": scope,
        "run_evidence": run_evidence,
        "output_integrity": {
            "analysis_sha256": sha256_file(args.analysis),
            "fold_metrics_sha256": sha256_file(args.fold_metrics),
            "history_metrics_sha256": sha256_file(args.history_metrics),
            "component_predictions_sha256": sha256_file(args.component_predictions),
            "fold_metric_rows": len(fold_rows),
            "history_metric_rows": len(history_rows),
        },
        "component_array_integrity": component_arrays,
        "analysis_replay": replay,
        "canonical_population": population,
        "B_cache_manifest_verified": True,
        "row_identity": row_identity,
        "fold_identity": fold_identity,
        "frozen_transitive_imports": transitive,
        "decision": analysis["decision"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--b-result", type=Path, required=True)
    parser.add_argument("--component-predictions", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--run-audit", type=Path, required=True)
    parser.add_argument("--fold-metrics", type=Path, required=True)
    parser.add_argument("--history-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"]["continue_or_stop_LPHC_family"],
        "selected_path": result["decision"]["selected_path_or_null"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
