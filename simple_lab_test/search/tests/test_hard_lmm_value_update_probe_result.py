from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "paper/results/hard_lmm_value_update_probe_20260909"


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
    assert manifest["per_slot_sha256"] == sha256_file(RESULT / "per_slot.csv")
    assert manifest["per_slot_stratum_sha256"] == sha256_file(
        RESULT / "per_slot_stratum.csv"
    )
    for relative, expected in manifest["diagnostic_source_hashes"].items():
        assert sha256_file(ROOT / relative) == expected
    loaded = manifest["source"]["loaded_module_audit"]
    assert loaded["verified_file_count"] == len(loaded["verified_files"])
    assert loaded["verified_file_count"] >= 40


def test_common_gate_fails_after_both_directions_and_scopes_are_evaluated() -> None:
    decision = load("evidence_decision.json")
    analysis = load("analysis.json")
    expected = {
        "intermittent_v2",
        "yellow_trip_hourly",
        "insta_market_basket",
    }
    assert set(decision["datasets"]) == expected
    assert set(analysis) == expected
    assert decision["passed"] is False
    assert decision["action"] == (
        "do_not_implement_or_gpu_train_usage_normalized_prototype_rule"
    )
    assert all(row["passed"] is False for row in decision["datasets"].values())
    for result in analysis.values():
        assert result["passed"] is False
        assert result["scope"] == {
            "split": "train",
            "validation_rows_materialized": False,
            "held_out_rows_materialized": False,
            "model_parameter_updates": False,
        }
        assert result["cohort"]["target_count"] == 4096
        assert result["cohort"]["fold_counts"] == {"0": 2048, "1": 2048}
        assert set(result["cross_fold"]) == {
            "fold0_to_fold1",
            "fold1_to_fold0",
        }
        for direction in result["cross_fold"].values():
            assert direction["passed"] is False
            assert direction["preclip_selected_value_bank"]["passed"] is False
            assert direction["postclip_full_model_update"]["passed"] is False


def test_native_gradient_parity_and_norm_match_contract_are_preserved() -> None:
    analysis = load("analysis.json")
    for result in analysis.values():
        for fold in ("0", "1"):
            row = result["folds"][fold]
            parity = row["parity"]
            assert parity["top4_indices_bitwise_equal"] is True
            assert parity["target_quantity_max_abs"] == 0.0
            assert parity["target_duration_max_abs"] == 0.0
            assert max(parity["selected_gradient_max_relative"].values()) < 2e-6
            assert row["batch_transform"]["max_norm_relative_error"] < 2e-12
            for metric in ("log", "raw", "body", "time"):
                assert row["previous_gradient_parity"][metric]["relative_error"] == 0.0


def test_usage_is_concentrated_but_normalization_does_not_supply_common_gain() -> None:
    analysis = load("analysis.json")
    expected_effective_rows = {
        "intermittent_v2": (14.060557705041507, 14.047909894965755),
        "yellow_trip_hourly": (22.145122150137336, 20.9332858287905),
        "insta_market_basket": (7.238364697497472, 7.100501547649674),
    }
    for dataset, expected in expected_effective_rows.items():
        for fold, value in enumerate(expected):
            observed = analysis[dataset]["folds"][str(fold)][
                "usage_and_conflict"
            ]["entropy_effective_rows"]
            assert abs(observed - value) < 1e-12

    # The dataset-specific conflicts that motivate rejecting one common rule.
    for fold in ("0", "1"):
        instacart = analysis["insta_market_basket"]["folds"][fold][
            "usage_and_conflict"
        ]["metric_conflicts"]["log_vs_raw"]
        assert instacart["conflicting_selection_mass"] > 0.96
        assert instacart["global_cosine"] < -0.9
        assert len(instacart["slot_cosines"]) == 64
        assert set(instacart["conflicting_slot_ids"]).issubset(
            instacart["eligible_slot_ids"]
        )
        assert all(
            instacart["slot_cosines"][slot] < 0.0
            for slot in instacart["conflicting_slot_ids"]
        )

        intermittent = analysis["intermittent_v2"]["folds"][fold][
            "usage_and_conflict"
        ]["metric_conflicts"]["body_vs_time"]
        assert intermittent["conflicting_selection_mass"] > 0.63
        assert intermittent["global_cosine"] < 0.0

    # Intermittent and Taxi both lose log/raw improvement versus B in both
    # directions. Instacart improves the raw dot versus B, but its candidate
    # raw direction remains harmful and does not beat the shuffled controls.
    for dataset in ("intermittent_v2", "yellow_trip_hourly"):
        for direction in analysis[dataset]["cross_fold"].values():
            for scope in (
                "preclip_selected_value_bank",
                "postclip_full_model_update",
            ):
                metrics = direction[scope]["metrics"]
                assert metrics["log"]["checks"]["incremental_over_b"] is False
                assert metrics["raw"]["checks"]["incremental_over_b"] is False

    for direction in analysis["insta_market_basket"]["cross_fold"].values():
        for scope in (
            "preclip_selected_value_bank",
            "postclip_full_model_update",
        ):
            raw = direction[scope]["metrics"]["raw"]
            assert raw["candidate"]["dot"] < 0.0
            assert raw["checks"]["absolute_positive"] is False
            assert raw["checks"]["exceeds_shuffle_p95"] is False


def test_per_slot_export_has_every_dataset_fold_and_memory_row() -> None:
    with (RESULT / "per_slot.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 2 * 64
    observed = {
        (row["dataset"], int(row["fold"]), int(row["slot"])) for row in rows
    }
    expected = {
        (dataset, fold, row_id)
        for dataset in (
            "intermittent_v2",
            "yellow_trip_hourly",
            "insta_market_basket",
        )
        for fold in (0, 1)
        for row_id in range(64)
    }
    assert observed == expected


def test_stratum_by_slot_export_is_complete_and_reconciles_to_slot_counts() -> None:
    with (RESULT / "per_slot.csv").open(newline="", encoding="utf-8") as stream:
        slot_rows = list(csv.DictReader(stream))
    with (RESULT / "per_slot_stratum.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        stratum_rows = list(csv.DictReader(stream))
    assert len(stratum_rows) == 3 * 2 * 5 * 64
    slot_counts = {
        (row["dataset"], int(row["fold"]), int(row["slot"])): int(
            row["selection_count"]
        )
        for row in slot_rows
    }
    reconstructed = {key: 0 for key in slot_counts}
    per_stratum: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in stratum_rows:
        key = (row["dataset"], int(row["fold"]), int(row["slot"]))
        reconstructed[key] += int(row["selection_count"])
        stratum_key = (row["dataset"], int(row["fold"]), int(row["stratum"]))
        per_stratum.setdefault(stratum_key, []).append(row)
    assert reconstructed == slot_counts
    for rows in per_stratum.values():
        assert len(rows) == 64
        target_count = int(rows[0]["stratum_target_count"])
        assert sum(int(row["selection_count"]) for row in rows) == 4 * target_count
        assert abs(
            sum(float(row["selection_mass_within_stratum"]) for row in rows)
            - 1.0
        ) < 1e-12
