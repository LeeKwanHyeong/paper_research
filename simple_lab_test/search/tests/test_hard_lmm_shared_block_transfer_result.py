from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "paper/results/hard_lmm_shared_block_transfer_probe_20260909"
BASE_CONTRACT = ROOT / "paper/contracts/hard_lmm_credit_assignment_probe_v1.json"


def load(name: str):
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_completed_result_is_hash_locked_and_train_only() -> None:
    manifest = load("execution_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["validation_rows_materialized"] is False
    assert manifest["held_out_rows_materialized"] is False
    assert manifest["backbone_or_quantity_parameter_updates"] is False
    assert manifest["source_fold_time_head_calibration"] is True
    assert manifest["runtime"]["device"] == "cpu"
    assert manifest["runtime"]["torch_threads"] == 4
    assert manifest["source"]["frozen_source_revision"] == (
        "f75243473adc25d622319dbca9bda7e076d8240f"
    )
    assert manifest["source"]["loaded_module_audit"]["verified_file_count"] >= 40

    artifacts = {
        "analysis_sha256": "analysis.json",
        "metrics_sha256": "metrics.csv",
        "decision_sha256": "evidence_decision.json",
        "readme_sha256": "README.md",
    }
    for key, name in artifacts.items():
        assert sha256_file(RESULT / name) == manifest[key]
    for relative, expected in manifest["diagnostic_source_hashes"].items():
        assert sha256_file(ROOT / relative) == expected
    for relative, expected in manifest["reused_evidence"].items():
        assert sha256_file(ROOT / relative) == expected


def test_crossfit_head_and_live_gradient_contract_hold_in_every_direction() -> None:
    analysis = load("analysis.json")
    base = json.loads(BASE_CONTRACT.read_text(encoding="utf-8"))
    expected_state = {
        row["dataset"]: row["checkpoint_state_sha256"] for row in base["datasets"]
    }
    assert set(analysis) == {
        "intermittent_v2",
        "yellow_trip_hourly",
        "insta_market_basket",
    }
    for dataset, result in analysis.items():
        assert result["scope"] == {
            "split": "train",
            "validation_rows_materialized": False,
            "held_out_rows_materialized": False,
            "backbone_or_quantity_parameter_updates": False,
            "source_fold_time_head_calibration": True,
        }
        assert result["cohort"]["target_count"] == 4096
        assert result["cohort"]["fold_counts"] == {"0": 2048, "1": 2048}
        assert result["checkpoint_state_sha256_before_after"] == expected_state[dataset]
        assert set(result["directions"]) == {
            "fold0_to_fold1",
            "fold1_to_fold0",
        }
        for direction in result["directions"].values():
            fit = direction["time_head_fit"]
            assert fit["scope"] == "source_fold_train_only"
            assert fit["target_count"] == 2048
            assert fit["parameter_count"] == 130
            assert fit["parameter_dtype"] == "float32"
            assert fit["calculation_dtype"] == "float64"
            assert fit["fixed_epochs"] == 100
            assert fit["batch_size"] == 8192
            assert fit["optimizer"] == "AdamW"
            assert fit["optimizer_betas"] == [0.9, 0.999]
            assert fit["optimizer_eps"] == 1e-8
            assert fit["optimizer_amsgrad"] is False
            assert fit["optimizer_foreach"] is False
            assert len(fit["history"]) == 100
            assert fit["state_changed"] is True
            assert math.isfinite(fit["initial_nll"])
            assert math.isfinite(fit["final_nll"])
            assert fit["final_nll"] < fit["initial_nll"] - 1e-8

            source = direction["source"]
            held = direction["held"]
            assert source["counts"]["log"] == 2048
            assert source["counts"]["raw"] == 2048
            assert source["counts"]["time"] == 2048
            assert held["counts"]["log"] == 2048
            assert held["counts"]["raw"] == 2048
            assert held["counts"]["time"] == 2048
            assert math.isclose(
                source["mean_losses"]["time"],
                fit["final_nll"],
                rel_tol=0.0,
                abs_tol=2e-10,
            )
            for probe in (source, held):
                assert probe["cache_parity"]["top4_indices_bitwise_equal"] is True
                assert probe["top4_margin"]["exact_tie_count"] == 0
                assert probe["top4_margin"]["minimum"] > 0.0
                assert probe["top4_margin"]["valid_token_count"] > 2048
                assert all(value > 0.0 for value in probe["time_hidden_credit_l2_sum"].values())
                for boundary in ("H1", "H2", "fused"):
                    assert all(
                        value > 0.0
                        for value in probe["gradient_norms"][boundary].values()
                    )


def test_decision_recomputes_to_no_common_candidate_boundary() -> None:
    analysis = load("analysis.json")
    decision = load("evidence_decision.json")
    recomputed = {
        boundary: all(
            result["boundary_pass"][boundary] for result in analysis.values()
        )
        for boundary in ("H1", "H2", "fused")
    }
    assert recomputed == decision["common_boundary_pass"] == {
        "H1": False,
        "H2": False,
        "fused": False,
    }
    assert decision["passed"] is False
    assert decision["selected_candidate_boundary"] is None
    assert decision["action"] == (
        "do_not_implement_or_gpu_train_the_zero_init_task_specific_"
        "linear_residual_branch"
    )
    assert all(
        not passed
        for result in analysis.values()
        for passed in result["boundary_pass"].values()
    )


def test_h2_failure_drivers_are_dataset_specific_and_bidirectional() -> None:
    analysis = load("analysis.json")

    intermittent = analysis["intermittent_v2"]["directions"]
    for direction in intermittent.values():
        metrics = direction["boundaries"]["H2"]["metrics"]
        assert metrics["log_mse"]["checks"]["passed"] is True
        assert metrics["raw_squared_error"]["checks"]["passed"] is True
        assert metrics["body_mae"]["checks"]["passed"] is True
        assert metrics["normalized_time_nll"]["checks"]["passed"] is False

    taxi = analysis["yellow_trip_hourly"]["directions"]
    for direction in taxi.values():
        metrics = direction["boundaries"]["H2"]["metrics"]
        assert metrics["log_mse"]["checks"]["passed"] is False
        assert metrics["raw_squared_error"]["checks"]["passed"] is False
        assert metrics["normalized_time_nll"]["checks"]["passed"] is False
    assert taxi["fold1_to_fold0"]["boundaries"]["H2"]["metrics"]["body_mae"][
        "checks"
    ]["passed"] is False

    instacart = analysis["insta_market_basket"]["directions"]
    for direction in instacart.values():
        raw = direction["boundaries"]["H2"]["metrics"]["raw_squared_error"]
        assert raw["task_separated"]["dot"] < 0.0
        assert raw["task_separated"]["cosine"] < -0.9
        assert raw["checks"]["passed"] is False


def test_metric_export_contains_all_dataset_direction_boundary_metric_rows() -> None:
    with (RESULT / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 2 * 3 * 4
    expected = {
        (dataset, direction, boundary, metric)
        for dataset in (
            "intermittent_v2",
            "yellow_trip_hourly",
            "insta_market_basket",
        )
        for direction in ("fold0_to_fold1", "fold1_to_fold0")
        for boundary in ("H1", "H2", "fused")
        for metric in ("log", "raw", "body", "time")
    }
    observed = {
        (row["dataset"], row["direction"], row["boundary"], row["metric"])
        for row in rows
    }
    assert observed == expected
    assert all(int(row["cyclic_output_row_shift_control__count"]) == 63 for row in rows)
