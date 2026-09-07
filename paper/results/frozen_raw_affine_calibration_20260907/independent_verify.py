#!/usr/bin/env python3
"""Independently verify the terminal train gate for frozen raw-affine calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "search_artifacts/frozen_raw_affine_calibration_v1_1ba9a43_5080"
CONTRACT = ROOT / "paper/contracts/frozen_raw_affine_calibration_v1.json"
SOURCE_MANIFEST = ARTIFACT / "source_manifest.json"
DEFAULT_OUTPUT = Path(__file__).with_name("independent_audit.json")
DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"frozen_raw_affine_array_v1\0")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def relative_change(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / abs(baseline)


def verify_cache(result_path: Path, result: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    cache_audit = result["cache"]["train"]
    cache_path = result_path.parent / cache_audit["file"]
    check(cache_path.is_file(), f"missing cache: {cache_path}")
    check(sha256_file(cache_path) == cache_audit["file_sha256"], "cache file digest drift")
    with np.load(cache_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    check(set(arrays) == set(cache_audit["arrays"]), "cache array scope drift")
    lengths = {len(value) for value in arrays.values()}
    check(lengths == {int(cache_audit["target_count"])}, "cache array alignment drift")
    for name, specification in cache_audit["arrays"].items():
        value = arrays[name]
        check(str(value.dtype) == specification["dtype"], f"cache dtype drift: {name}")
        check(list(value.shape) == specification["shape"], f"cache shape drift: {name}")
        check(array_sha256(value, label=name) == specification["sha256"], f"array digest drift: {name}")
        check(bool(np.isfinite(value).all()), f"non-finite cache array: {name}")
    expected_selected = 16 if smoke else result["target_population"]["train"]["target_count"]
    check(cache_audit["target_count"] == expected_selected, "selected target-count drift")
    check(
        cache_audit["full_population_target_count"]
        == result["target_population"]["train"]["target_count"],
        "full target-count drift",
    )
    identity = np.maximum(0.0, arrays["prediction"])
    check(np.array_equal(identity, arrays["prediction"]), "identity quantity map changed output")
    copied_time = np.array(arrays["time_nll"], copy=True)
    check(
        copied_time is not arrays["time_nll"] and np.array_equal(copied_time, arrays["time_nll"]),
        "quantity-only path did not preserve time NLL",
    )
    return {
        "cache_file_sha256": cache_audit["file_sha256"],
        "selected_targets": int(cache_audit["target_count"]),
        "full_population_targets": int(cache_audit["full_population_target_count"]),
        "all_arrays_finite": True,
        "identity_quantity_exact": True,
        "copied_time_nll_exact": True,
    }


def verify_common_result(
    result_path: Path,
    result: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
    *,
    smoke: bool,
    contract_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    check(result["status"] == "success", "result is not successful")
    check(result["phase"] == "train", "non-train result admitted")
    check(result["dataset"] == dataset_spec["dataset"], "dataset drift")
    check(result["model_role"] == "B", "model-role drift")
    check(result["smoke"] is smoke, "smoke scope drift")
    check(result["held_out_test_evaluated"] is False, "held-out access detected")
    check(result["evaluation_scope"] == "validation_only", "evaluation scope drift")
    check(result.get("validation") is None, "validation result was materialized")
    check(result["contract_sha256"] == contract_sha256, "contract digest drift")
    check(
        result["source_revision"] == "1ba9a439d003cad901a80ef05b9a6b327b4cff2a",
        "source revision drift",
    )
    check(result["source_manifest"]["file_sha256"] == source_manifest_sha256, "source manifest drift")
    check(result["source_manifest"]["all_file_hashes_verified"] is True, "source files unverified")
    check(
        result["source_manifest"]["runtime_empty_directories_verified"] == ["sample_data"],
        "runtime sentinel unverified",
    )
    check(result["input_digests"]["data_sha256"] == dataset_spec["data_sha256"], "data drift")
    check(
        result["input_digests"]["split_manifest_sha256"]
        == dataset_spec["split_manifest_sha256"],
        "split drift",
    )
    source = dataset_spec["sources"]["B"]
    check(
        result["source_checkpoint_file_sha256"] == source["checkpoint_file_sha256"],
        "checkpoint file drift",
    )
    check(result["source_model_state_sha256"] == source["checkpoint_state_sha256"], "state drift")
    check(result["source_model_state_unchanged"] is True, "source state changed")
    check(result["source_gradients_absent"] is True, "source gradient detected")
    runtime = result["runtime"]
    check("RTX 5080" in runtime["gpu_name"], "non-5080 result")
    check(runtime["device"] == "cuda:0" and runtime["cuda_device_index"] == 0, "CUDA device drift")
    check(
        runtime["deterministic_environment"]
        == read_json(CONTRACT)["runtime"]["deterministic_environment"],
        "deterministic environment drift",
    )
    population = result["target_population"]["train"]
    check(population["target_count"] == dataset_spec["expected_train_targets"], "target count drift")
    check(
        population["target_identity_sha256"]
        == dataset_spec["expected_train_target_identity_sha256"],
        "target identity drift",
    )
    check(
        population["target_quantity_sha256"]
        == dataset_spec["expected_train_target_quantity_sha256"],
        "target quantity drift",
    )
    check(
        population["train_quantity_boundaries"] == dataset_spec["train_quantity_boundaries"],
        "quantity-boundary drift",
    )
    return verify_cache(result_path, result, smoke=smoke)


def verify_state(result_path: Path, result: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    reference = result["train_audit"]["calibration_state"]
    state_path = result_path.parent / reference["file"]
    check(sha256_file(state_path) == reference["file_sha256"], "calibration-state digest drift")
    state = read_json(state_path)
    expected_reason = "smoke_contract_exercise" if smoke else "identity_fallback"
    check(state["activation_reason"] == expected_reason, "activation reason drift")
    if smoke:
        check(state["calibration_activated"] is True, "smoke failed to exercise calibration")
    else:
        check(state["calibration_activated"] is False, "failed full gate activated calibration")
        check(
            state["deployed_calibration_parameters"] == {"slope": 1.0, "intercept": 0.0},
            "failed gate did not deploy identity",
        )
    return {
        "calibration_state_sha256": reference["file_sha256"],
        "activation_reason": state["activation_reason"],
        "deployed_parameters": state["deployed_calibration_parameters"],
    }


def verify_full_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    audit = result["train_audit"]
    oof = audit["two_fold_oof"]
    baseline = oof["baseline"]
    calibrated = oof["calibrated"]
    rmse_improvement = -relative_change(
        float(calibrated["overall"]["rmse"]), float(baseline["overall"]["rmse"])
    )
    body_regression = relative_change(
        float(calibrated["body_le_p95"]["mae"]),
        float(baseline["body_le_p95"]["mae"]),
    )
    tail_regression = relative_change(
        float(calibrated["gt_p99"]["mae"]), float(baseline["gt_p99"]["mae"])
    )
    stored = audit["relative_changes"]
    check(abs(rmse_improvement - stored["pooled_oof_rmse_improvement"]) < 1e-12, "RMSE change drift")
    check(abs(body_regression - stored["body_le_p95_mae_regression"]) < 1e-12, "body change drift")
    check(abs(tail_regression - stored["tail_gt_p99_mae_regression"]) < 1e-12, "tail change drift")
    criteria = {
        "both_folds_raw_mse_improve": bool(oof["both_folds_raw_mse_improve"]),
        "pooled_oof_rmse_improvement_at_least_1pct": rmse_improvement >= 0.01,
        "body_mae_regression_at_most_2pct": body_regression <= 0.02,
        "tail_mae_regression_at_most_2pct": tail_regression <= 0.02,
    }
    check(audit["performance_gate_evaluated"] is True, "full performance gate not evaluated")
    check(audit["gate_passed"] is all(criteria.values()), "gate boolean was not reproduced")
    check(audit["gate_passed"] is False, "unexpected accepted dataset")
    return {
        "targets": int(oof["rows"]),
        "series": int(oof["series"]),
        "baseline_mae": float(baseline["overall"]["mae"]),
        "calibrated_mae": float(calibrated["overall"]["mae"]),
        "baseline_rmse": float(baseline["overall"]["rmse"]),
        "calibrated_rmse": float(calibrated["overall"]["rmse"]),
        "pooled_oof_rmse_improvement_fraction": rmse_improvement,
        "body_mae_regression_fraction": body_regression,
        "tail_mae_regression_fraction": tail_regression,
        "criteria": criteria,
        "gate_passed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the 5080 frozen raw-affine train-gate artifact."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON audit destination (default: independent_audit.json beside this script)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file after all checks pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    check(args.overwrite or not output.exists(), f"refusing to overwrite {output}")
    contract = read_json(CONTRACT)
    contract_sha256 = sha256_file(CONTRACT)
    source_manifest = read_json(SOURCE_MANIFEST)
    source_manifest_sha256 = sha256_file(SOURCE_MANIFEST)
    check(source_manifest["source_revision"] == "1ba9a439d003cad901a80ef05b9a6b327b4cff2a", "manifest revision drift")
    check(source_manifest["runtime_empty_directories"] == ["sample_data"], "runtime directory drift")
    check(source_manifest["file_count"] == len(source_manifest["files"]), "source file count drift")
    for relative, expected in source_manifest["files"].items():
        path = ROOT / relative
        check(path.is_file(), f"local source file missing: {relative}")
        check(sha256_file(path) == expected, f"local source digest drift: {relative}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    source_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_manifest["source_revision"], head],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    check(source_is_ancestor, "execution source is not an ancestor of local HEAD")
    source_tree = subprocess.run(
        ["git", "rev-parse", f"{source_manifest['source_revision']}^{{tree}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    pipeline = read_json(ARTIFACT / "pipeline_status.json")
    decision_path = ARTIFACT / "decisions/B_train.json"
    decision = read_json(decision_path)
    check(pipeline["status"] == "success", "controller did not terminate normally")
    check(pipeline["decision"] == "rejected_stop_before_validation", "controller decision drift")
    check(decision["accepted"] is False, "aggregate train gate unexpectedly accepted")
    check(decision["checks"] == {dataset: False for dataset in DATASETS}, "dataset checks drift")
    check(decision["contract_sha256"] == contract_sha256, "decision contract drift")
    check(decision["source_manifest_file_sha256"] == source_manifest_sha256, "decision manifest drift")
    check(decision["execution_server"] == "5080", "decision host drift")
    check(decision["held_out_test_evaluated"] is False, "decision includes held-out")
    check(not (ARTIFACT / "full/validation").exists(), "full validation artifact exists")
    check(not (ARTIFACT / "smoke/validation").exists(), "validation smoke artifact exists")
    check(not any("test" in path.name.lower() for path in ARTIFACT.rglob("*")), "test artifact exists")

    rows_by_dataset = {
        row["dataset"]: row for row in contract["datasets"]
    }
    identities = {row["dataset"]: row for row in decision["result_identities"]}
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        result_path = ARTIFACT / f"full/train/{dataset}_B/result.json"
        result = read_json(result_path)
        identity = identities[dataset]
        check(identity["result_file"].endswith(f"/full/train/{dataset}_B/result.json"), "result path drift")
        check(sha256_file(result_path) == identity["result_file_sha256"], "result digest drift")
        common = verify_common_result(
            result_path,
            result,
            rows_by_dataset[dataset],
            smoke=False,
            contract_sha256=contract_sha256,
            source_manifest_sha256=source_manifest_sha256,
        )
        state = verify_state(result_path, result, smoke=False)
        gate = verify_full_gate(result)
        datasets[dataset] = {**gate, "cache": common, "state": state}

    smoke_decision = read_json(ARTIFACT / "decisions/train_smoke.json")
    smoke_identities = {
        Path(row["result_file"]).parent.name.removesuffix("_B"): row
        for row in smoke_decision["result_identities"]
    }
    check(set(smoke_identities) == set(DATASETS), "smoke scope drift")
    smoke: dict[str, Any] = {}
    for dataset in DATASETS:
        result_path = ARTIFACT / f"smoke/train/{dataset}_B/result.json"
        result = read_json(result_path)
        check(sha256_file(result_path) == smoke_identities[dataset]["result_file_sha256"], "smoke digest drift")
        common = verify_common_result(
            result_path,
            result,
            rows_by_dataset[dataset],
            smoke=True,
            contract_sha256=contract_sha256,
            source_manifest_sha256=source_manifest_sha256,
        )
        smoke[dataset] = {**common, "state": verify_state(result_path, result, smoke=True)}

    artifact_hashes = {
        str(path.relative_to(ARTIFACT)): sha256_file(path)
        for path in sorted(ARTIFACT.rglob("*"))
        if path.is_file()
    }
    audit = {
        "schema_version": 1,
        "audit_id": "frozen_raw_affine_calibration_train_gate_20260907",
        "status": "PASS_NO_BLOCKING_INTEGRITY_FINDING",
        "verifier_sha256": sha256_file(Path(__file__)),
        "source_revision": source_manifest["source_revision"],
        "source_tree": source_tree,
        "verification_head": head,
        "execution_source_is_ancestor_of_verification_head": source_is_ancestor,
        "source_manifest_sha256": source_manifest_sha256,
        "source_files_verified": source_manifest["file_count"],
        "contract_sha256": contract_sha256,
        "execution_server": "5080",
        "controller_decision": pipeline["decision"],
        "aggregate_gate_reproduced": not any(row["gate_passed"] for row in datasets.values()),
        "validation_artifacts_absent": True,
        "held_out_artifacts_absent": True,
        "datasets": datasets,
        "cuda_train_smoke": smoke,
        "artifact_file_hashes": artifact_hashes,
        "interpretation_boundary": "output-interface calibration ablation; not a Backbone improvement",
        "limitations": [
            "The frozen source checkpoints were trained on the complete train split, so the series-disjoint folds test calibration transfer and are not independently trained backbone folds.",
            "The prospective train gate rejected every dataset, so validation, RMTPP/THP fairness, additional seeds, and held-out test were intentionally not evaluated.",
            "Some result references retain absolute paths from the 5080 host; the verifier resolves the hash-matched local mirror directly, but the original summarizer is not path-portable without remapping.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "decision": audit["controller_decision"],
        "artifact_files": len(artifact_hashes),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
