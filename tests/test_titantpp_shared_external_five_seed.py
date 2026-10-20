"""Synthetic records and local evidence only; no real checkpoints or models."""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
from unittest.mock import patch

import pytest

from paper.scripts import compare_titantpp_shared_external_five_seed as comparator


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fabricated_records():
    rows = []
    for dataset in comparator.DATASETS:
        for model_index, model in enumerate(comparator.MODELS):
            for seed in comparator.SEEDS:
                for split in comparator.SPLITS:
                    value = model_index + seed / 10
                    rows.append({"dataset": dataset, "model": model, "seed": seed, "split": split,
                        "selected_epoch": 5, "metrics": {"overall": {"qty_rmse": value,
                            "qty_mae": value / 2, "time_nll": value / 3}, "tail": {
                            "qty_rmse": value * 2, "qty_mae": value, "time_nll": value / 4}}})
    return rows


def prerequisites(root, gate_status="MISMATCH"):
    report = root / "reports" / comparator.NAME
    bundle = root / "search_artifacts" / comparator.NAME
    raw = root / "search_artifacts" / comparator.RAW9 / comparator.RAW9_EXECUTION
    save(bundle / "retrieval/final_receipt.json", {})
    save(raw / "retrieval/final_receipt.json", {})
    save(report / "PAPER_START_GATE.json", {"status": gate_status})
    anchor = root / "anchor.json"
    save(anchor, {"fabricated_only": True})
    science = {"prior_completion_bindings": {str(anchor): comparator.sha(anchor)}}
    save(report / "SCIENCE_SPEC.json", science)
    conditions = {"fabricated_only": True}
    save(report / "FROZEN_CONDITIONS.json", conditions)
    contract = {"science_spec_sha256": comparator.sha(report / "SCIENCE_SPEC.json"),
        "frozen_conditions": conditions,
        "frozen_conditions_sha256": comparator.sha(report / "FROZEN_CONDITIONS.json")}
    save(bundle / "execution_contract.json", contract)
    return report, bundle, raw


def test_missing_new42_receipt_waits_before_any_evidence_read_and_creates_nothing(tmp_path):
    output = tmp_path / "must_not_exist"
    with patch.object(comparator, "read", side_effect=AssertionError("No evidence read before completion")):
        value = comparator.run(tmp_path, output)
    assert value == {"status": "waiting_for_extra42_completion", "remote_calls": 0, "output_files": []}
    assert not output.exists() and list(tmp_path.iterdir()) == []


def test_new42_present_but_original5090_missing_waits_without_outputs(tmp_path):
    save(tmp_path / "search_artifacts" / comparator.NAME / "retrieval/final_receipt.json", {})
    with patch.object(comparator, "read", side_effect=AssertionError("No incomplete evidence reads")):
        result = comparator.run(tmp_path)
    assert result["status"] == "waiting_for_original5090_final_validation"
    assert not (tmp_path / "reports").exists()


def test_original5090_receipt_uses_execution_subdirectory_not_campaign_root(tmp_path):
    """The real original5090 artifact is one level below the campaign root."""
    report, _, raw_execution = prerequisites(tmp_path)
    campaign = tmp_path / "search_artifacts" / comparator.RAW9
    assert raw_execution == campaign / "execution_5090_v1"
    assert (raw_execution / "retrieval/final_receipt.json").exists()
    assert not (campaign / "retrieval/final_receipt.json").exists()
    with patch.object(comparator, "verify_paper_gate", side_effect=ValueError("actual execution path admitted")):
        with pytest.raises(ValueError, match="actual execution path admitted"):
            comparator.run(tmp_path)
    # A receipt at the parent campaign path cannot replace the real receipt.
    (raw_execution / "retrieval/final_receipt.json").unlink()
    save(campaign / "retrieval/final_receipt.json", {"status": "unrelated_parent_receipt"})
    with patch.object(comparator, "read", side_effect=AssertionError("No incomplete evidence reads")):
        waiting = comparator.run(tmp_path)
    assert waiting["status"] == "waiting_for_original5090_final_validation"
    assert waiting["output_files"] == []


def test_gate_mismatch_fails_before_new_results_or_checkpoint_load_and_writes_nothing(tmp_path):
    prerequisites(tmp_path)
    with patch.object(comparator, "verify_new", side_effect=AssertionError("No aggregation after mismatch")), \
         patch.object(comparator, "checkpoint_payload", side_effect=AssertionError("No checkpoint load")):
        with pytest.raises(ValueError, match="gate did not pass"):
            comparator.run(tmp_path)
    assert not (tmp_path / "reports" / comparator.NAME / "five_seed_aggregation_v1").exists()


def test_all_five_historical_three_and_new_two_are_separate_sample_sd_tables():
    records = fabricated_records()
    result = comparator.aggregate(records)
    assert len(result["records"]) == 210 and len(result["seed_differences"]) == 1080
    assert set(result["summaries"]) == {"historical_three", "additional_two", "all_five"}
    for name, seeds in (("historical_three", comparator.OLD_SEEDS), ("additional_two", comparator.NEW_SEEDS),
                        ("all_five", comparator.SEEDS)):
        value = next(r for r in result["summaries"][name] if r["dataset"] == comparator.DATASETS[0]
            and r["model"] == comparator.SHARED and r["split"] == "test"
            and r["group"] == "overall" and r["metric"] == "qty_rmse")
        assert value["mean"] == statistics.mean(s / 10 for s in seeds)
        assert value["sample_sd_ddof1"] == statistics.stdev(s / 10 for s in seeds)
        assert value["seeds"] == list(seeds)
    assert not result["raw_errors_pooled"] and result["mlp16_fits_or_evaluations"] == 0
    assert result["remote_calls"] == result["model_inference_calls"] == 0


def test_all_seed_differences_preserve_direction_and_zero_reference_is_na():
    rows = fabricated_records()
    reference = next(r for r in rows if r["dataset"] == comparator.DATASETS[0]
        and r["model"] == comparator.EXTERNAL[0] and r["seed"] == 42 and r["split"] == "test")
    reference["metrics"]["overall"]["qty_rmse"] = 0.
    result = comparator.aggregate(rows)
    values = [r for r in result["seed_differences"] if r["dataset"] == comparator.DATASETS[0]
        and r["reference"] == comparator.EXTERNAL[0] and r["seed"] == 42 and r["split"] == "test"
        and r["group"] == "overall"]
    zero = next(r for r in values if r["metric"] == "qty_rmse")
    assert zero["absolute_difference"] == 4.2 and zero["worsened"] is True
    assert zero["relative_difference_percent"] is None and zero["relative_status"] == "N/A_reference_zero"
    mae = next(r for r in values if r["metric"] == "qty_mae")
    assert mae["absolute_difference"] < 0 and mae["worsened"] is False


def test_unverified_tail_is_not_filled_from_partial_seed_means():
    rows = fabricated_records()
    for r in rows:
        if r["model"] in comparator.EXTERNAL and r["seed"] in comparator.OLD_SEEDS and r["split"] == "validation":
            r["metrics"]["tail"] = {"qty_rmse": None, "qty_mae": None, "time_nll": None}
    result = comparator.aggregate(rows)
    five = next(r for r in result["summaries"]["all_five"] if r["dataset"] == comparator.DATASETS[0]
        and r["model"] == comparator.EXTERNAL[0] and r["split"] == "validation"
        and r["group"] == "tail" and r["metric"] == "qty_rmse")
    assert five["mean"] is None and five["sample_sd_ddof1"] is None
    assert five["available_seeds"] == [72, 82] and five["status"] == "N/A_incomplete_evidence"
    assert "N/A" in comparator.markdown(result)


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "negative", "mlp16"])
def test_bad_core_metrics_or_matrix_fail_closed(change):
    rows = fabricated_records()
    if change == "missing": rows.pop()
    elif change == "duplicate": rows.append(deepcopy(rows[0]))
    elif change == "nan": rows[0]["metrics"]["overall"]["qty_rmse"] = math.nan
    elif change == "negative": rows[0]["metrics"]["overall"]["qty_mae"] = -1.
    else: rows[0]["model"] = "titantpp_history_mlp_width16"
    with pytest.raises(ValueError): comparator.aggregate(rows)


def test_selected_summary_rmse_and_mae_use_identical_earliest_checkpoint_epoch():
    history = [{"epoch": 1, "val_qty_rmse": 5., "val_qty_mae": 3.},
        {"epoch": 2, "val_qty_rmse": 4., "val_qty_mae": 3.5},
        {"epoch": 3, "val_qty_rmse": 4., "val_qty_mae": 2.}]
    summary = {"status": "success", "backbone": "rmtpp", "seed": 42,
        "variant": "count_only_log_regression", "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False, "checkpoint_monitor": "validation_raw_quantity_rmse",
        "best_epoch": 2, "resume_identity": {"arguments": {"dataset_contract": comparator.DATASETS[0]}},
        "encoder_config": {}, "checkpoint_state_sha256": "b" * 64,
        "selected_metric_value": 4., "best_val_qty_rmse": 4., "best_val_qty_mae": 3.5}
    payload = {"best_epoch": 2, "backbone": "rmtpp", "seed": 42,
        "checkpoint_monitor": "validation_raw_quantity_rmse", "encoder_config": {},
        "model_state_sha256": "b" * 64, "model_state_dict": {}, "selected_metric_value": 4.}
    with patch.object(comparator, "state_sha", return_value="b" * 64):
        selected = comparator.selected_summary(summary, history, payload, comparator.DATASETS[0], "rmtpp", 42)
        assert selected["epoch"] == 2
        summary["best_val_qty_mae"] = 2.  # Better MAE from another epoch must be rejected.
        with pytest.raises(ValueError, match="identical selected epoch"):
            comparator.selected_summary(summary, history, payload, comparator.DATASETS[0], "rmtpp", 42)


def test_archive_mutation_and_path_escape_are_rejected(tmp_path):
    folder = tmp_path / "collection"
    save(folder / "originals/endpoint.json", {"fabricated": True})
    (folder / "original.tar.gz").write_bytes(b"fabricated archive bytes")
    save(folder / "manifest.json", {"files": {"endpoint.json": comparator.sha(folder / "originals/endpoint.json")}})
    save(folder / "receipt.json", {"status": "complete", "file_count": 1,
        "archive_sha256": comparator.sha(folder / "original.tar.gz"),
        "manifest_sha256": comparator.sha(folder / "manifest.json")})
    comparator.verify_collection(tmp_path, folder)
    (folder / "original.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Archive/manifest"):
        comparator.verify_collection(tmp_path, folder)
    with pytest.raises(ValueError, match="escaped"):
        comparator.local(tmp_path, "../outside")


def test_cli_waiting_json_is_stable_and_exit_zero(tmp_path, capsys):
    assert comparator.main(("--root", str(tmp_path), "--check-only")) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "waiting_for_extra42_completion"


def test_synthetic_orchestration_check_only_then_exclusive_new_outputs(tmp_path):
    report, _, _ = prerequisites(tmp_path, gate_status="PASS")
    rows = fabricated_records()
    for r in rows:
        r["state_sha256"] = f"fabricated_state_{r['dataset']}_{r['seed']}"
    old_test = [r for r in rows if r["seed"] in comparator.OLD_SEEDS and r["split"] == "test"]
    old_shared = [r for r in rows if r["seed"] in comparator.OLD_SEEDS and r["split"] == "validation"
                  and r["model"] == comparator.SHARED]
    old_external = [r for r in rows if r["seed"] in comparator.OLD_SEEDS and r["split"] == "validation"
                    and r["model"] in comparator.EXTERNAL]
    new = [r for r in rows if r["seed"] in comparator.NEW_SEEDS]
    output = report / "fabricated_output"
    with patch.object(comparator, "verify_paper_gate", return_value={"fabricated_only": True}), \
         patch.object(comparator, "verify_new", return_value=new), \
         patch.object(comparator, "legacy_tests", return_value=(old_test, {})), \
         patch.object(comparator, "shared_validation", return_value=old_shared), \
         patch.object(comparator, "external_validation", return_value=(old_external, [])):
        ready = comparator.run(tmp_path, output, check_only=True)
        assert ready["status"] == "verified_ready" and not output.exists()
        result = comparator.run(tmp_path, output)
        assert result["status"] == "complete" and result["record_count"] == 210
        saved = comparator.read(output / "FIVE_SEED_RESULTS.json")
        assert len(saved["records"]) == 210 and len(saved["seed_differences"]) == 1080
        assert set(output.iterdir()) == {output / "FIVE_SEED_RESULTS.json", output / "FIVE_SEED_RESULTS.md"}
        assert not saved["paper_text_generated"] and saved["remote_calls"] == saved["model_inference_calls"] == 0
        with pytest.raises(ValueError, match="Preserve existing aggregation"):
            comparator.run(tmp_path, output)
