from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "paper/results/hard_lmm_credit_assignment_probe_20260909"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str):
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def test_completed_result_is_train_only_frozen_and_hash_consistent() -> None:
    manifest = load("execution_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["validation_rows_materialized"] is False
    assert manifest["held_out_rows_materialized"] is False
    assert manifest["model_parameter_updates"] is False
    assert manifest["analysis_sha256"] == sha256_file(RESULT / "analysis.json")
    assert manifest["decision_sha256"] == sha256_file(
        RESULT / "evidence_decision.json"
    )
    for relative, expected in manifest["diagnostic_source_hashes"].items():
        assert sha256_file(ROOT / relative) == expected
    loaded = manifest["source"]["loaded_module_audit"]
    assert loaded["verified_file_count"] == len(loaded["verified_files"])
    assert loaded["verified_file_count"] >= 40


def test_common_gate_fails_only_after_all_three_dataset_results_exist() -> None:
    decision = load("evidence_decision.json")
    analysis = load("analysis.json")
    assert set(decision["datasets"]) == {
        "intermittent_v2",
        "yellow_trip_hourly",
        "insta_market_basket",
    }
    assert set(analysis) == set(decision["datasets"])
    assert decision["passed"] is False
    assert decision["action"] == "do_not_implement_or_gpu_train_candidate"
    assert decision["datasets"]["intermittent_v2"]["passed"] is True
    assert decision["datasets"]["yellow_trip_hourly"]["passed"] is False
    assert decision["datasets"]["insta_market_basket"]["passed"] is False


def test_fold_scope_parity_and_decisive_directions_are_preserved() -> None:
    analysis = load("analysis.json")
    for result in analysis.values():
        assert result["scope"] == {
            "split": "train",
            "validation_rows_materialized": False,
            "held_out_rows_materialized": False,
            "model_parameter_updates": False,
        }
        assert result["cohort"]["target_count"] == 4096
        assert result["cohort"]["fold_counts"] == {"0": 2048, "1": 2048}
        assert result["structural_audit"]["hard_query_addressing_gradient_max_abs"] == 0.0
        assert result["structural_audit"]["hard_key_addressing_gradient_max_abs"] == 0.0
        for fold in ("0", "1"):
            parity = result["folds"][fold]["parity"]
            assert parity["top4_indices_bitwise_equal"] is True
            assert parity["hard_residual_max_abs"] == 0.0
            assert parity["target_quantity_max_abs"] == 0.0

    intermittent = analysis["intermittent_v2"]["cross_fold"]
    assert all(
        row["normal"][metric]["dot"] > 0.0
        for row in intermittent.values()
        for metric in ("log_mse", "raw_squared_error", "body_mae")
    )
    instacart = analysis["insta_market_basket"]["cross_fold"]
    assert all(
        row["normal"]["raw_squared_error"]["dot"] < 0.0
        for row in instacart.values()
    )
    taxi = analysis["yellow_trip_hourly"]["cross_fold"]
    assert taxi["fold0_to_fold1"]["normal"]["raw_squared_error"]["dot"] < 0.0
    assert taxi["fold0_to_fold1"]["normal"]["body_mae"]["dot"] < 0.0
