"""Synthetic contract evidence tests; these tests never run a model or read Test data."""
import copy
import json
from pathlib import Path

import pytest

from paper.scripts import analyze_titantpp_history_instacart_frozen_test as analysis


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def metrics(split, value, count=100, tail_count=8):
    return {"split": split, "count": count, "qty_rmse": value, "qty_mae": value / 2,
            "time_nll": value / 4, "tail": {"count": tail_count, "qty_rmse": value * 2,
                                          "qty_mae": value, "time_nll": value / 3}}


@pytest.fixture
def campaign(tmp_path):
    contract = {"schema": "synthetic_fixture", "contract_sha256": "a" * 64,
                "expected_seeds": [42, 52, 62], "evaluation_scope": analysis.SCOPE, "jobs": [],
                "validation_replay_tolerance": {"absolute": 1e-5, "relative": 1e-5},
                "metrics_schema": "quantity_time_tail_v1", "datasets": {}}
    for study, datasets in (("history", ("taxi", "raf", "intermittent")), ("instacart", ("instacart",))):
        for dataset_index, dataset in enumerate(datasets):
            contract["datasets"][dataset] = {"data_sha256": "f" * 64,
                                              "training_data_sha256": "0" * 64 if dataset == "raf" else "f" * 64,
                                              "expected_test_count": 100}
            for arm_index, arm in enumerate(analysis.ARMS[study]):
                for seed_index, seed in enumerate(analysis.SEEDS):
                    job_id = f"{study}_{dataset}_{arm_index}_{seed}"
                    ref = metrics("validation", 2 + seed_index + arm_index)
                    test = metrics("test", 1 + seed_index + arm_index)
                    # An unfavorable seed62 full deliberately flips the Validation comparison.
                    if arm_index == 0 and seed == 62:
                        test = metrics("test", 8)
                    job = {"id": job_id, "study": study, "dataset": dataset, "arm": arm, "seed": seed,
                           "selected_epoch": 7 + seed_index, "checkpoint": {"sha256": "b" * 64},
                           "frozen_source_closure": "c" * 64, "validation_reference": ref,
                           "data_identity": {"target_identity_sha256": "d" * 64,
                                             "validation_target_identity_sha256": "e" * 64}}
                    contract["jobs"].append(job)
                    receipt = {"schema": "synthetic_receipt", "contract_sha256": contract["contract_sha256"],
                               "job_id": job_id, "dataset": dataset, "arm": arm, "seed": seed,
                               "selected_epoch": job["selected_epoch"], "checkpoint_sha256": "b" * 64,
                               "frozen_source_closure": "c" * 64, "validation_reference": copy.deepcopy(ref),
                               "validation_replay": copy.deepcopy(ref), "test": test,
                               "validation_replay_check": {"passed": True},
                               "data_identity": {"raw_sha256": "f" * 64, "target_identity_sha256": "d" * 64,
                                                 "validation_target_identity_sha256": "e" * 64,
                                                 "test_truth_sha256": "7" * 64,
                                                 "validation_truth_sha256": "8" * 64},
                               "runtime": {"host": "cpu-fixture"}, "retrained": False,
                               "reselected": False, "test_is_independent": False, "test_started_unix": 30,
                               "simple_quantity_baselines": {
                                   "last": {"split": "test", "count": 100, "qty_rmse": 9, "qty_mae": 4,
                                            "tail": {"count": 8, "qty_rmse": 12, "qty_mae": 6}},
                                   "mean": {"split": "test", "count": 100, "qty_rmse": 10, "qty_mae": 5,
                                            "tail": {"count": 8, "qty_rmse": 14, "qty_mae": 7}}}}
                    dump(tmp_path / "results" / job_id / "receipt.json", receipt)
                    dump(tmp_path / "results" / job_id / "worker_exit.json",
                         {"exit_code": 0, "job_id": job_id, "contract_sha256": contract["contract_sha256"],
                          "actual_child_exit_observed": True})
                    native = {"contract_sha256": contract["contract_sha256"], "job_id": job_id,
                              "selected_epoch": job["selected_epoch"], "checkpoint_sha256": "b" * 64,
                              "frozen_source_closure": "c" * 64, "data_identity": {"raw_sha256": "f" * 64}}
                    dump(tmp_path / "results" / job_id / "validation_replay.json",
                         {**native, "split": "validation", "metrics": copy.deepcopy(ref), "completed_unix": 10,
                          "target_identity_sha256": "e" * 64, "truth_sha256": "8" * 64,
                          "data_identity": {"raw_sha256": contract["datasets"][dataset]["training_data_sha256"]},
                          "validation_replay_check": {"passed": True}})
                    dump(tmp_path / "results" / job_id / "validation_worker_exit.json",
                         {"exit_code": 0, "job_id": job_id, "contract_sha256": contract["contract_sha256"],
                          "actual_child_exit_observed": True, "completed_unix": 11})
                    dump(tmp_path / "results" / job_id / "test_metrics.json",
                         {**native, "split": "test", "metrics": copy.deepcopy(test), "test_started_unix": 30,
                          "target_identity_sha256": "d" * 64, "truth_sha256": "7" * 64,
                          "simple_quantity_baselines": copy.deepcopy(receipt["simple_quantity_baselines"]),
                          "completed_unix": 40})
    reseal_contract(tmp_path, contract)
    return tmp_path, contract


def reseal_contract(bundle, contract):
    contract["contract_sha256"] = analysis.contract_digest(contract)
    dump(bundle / "contract.json", contract)
    for folder in (bundle / "results").iterdir():
        for name in ("receipt.json", "worker_exit.json", "validation_replay.json", "validation_worker_exit.json", "test_metrics.json"):
            path = folder / name
            value = analysis.read(path)
            value["contract_sha256"] = contract["contract_sha256"]
            dump(path, value)
    refresh_gate(bundle, contract)


def refresh_gate(bundle, contract):
    gate = {"schema": "synthetic_gate", "status": "PASS", "contract_sha256": contract["contract_sha256"],
            "job_count": 36, "created_unix": 20,
            "validation_receipts": {job["id"]: analysis.sha(bundle / "results" / job["id"] / "validation_replay.json")
                                    for job in contract["jobs"]}}
    dump(bundle / "validation_gate.json", gate)
    for job in contract["jobs"]:
        path = bundle / "results" / job["id"] / "receipt.json"
        value = analysis.read(path)
        value["validation_gate_sha256"] = analysis.sha(bundle / "validation_gate.json")
        dump(path, value)


def alter_receipt(campaign, mutate, *, job_index=0):
    bundle, contract = campaign
    path = bundle / "results" / contract["jobs"][job_index]["id"] / "receipt.json"
    value = analysis.read(path)
    mutate(value)
    dump(path, value)


def test_all36_complete_separate_splits_sample_sd_and_adverse_preserved(campaign):
    bundle, contract = campaign
    receipt = analysis.analyze(bundle)
    assert receipt["distinct_conditions"] == 36
    assert receipt["study_counts"] == {"history": 27, "instacart": 9}
    assert receipt["test_is_independent"] is False
    payload = analysis.read(bundle / "analysis/test_conditions.json")
    assert len(payload["conditions"]) == 36
    assert len(payload["seed_summary"]) == 12
    assert len(payload["paired_comparisons"]) == 24
    full_taxi = next(row for row in payload["seed_summary"]
                     if row["study"] == "history" and row["dataset"] == "taxi" and row["arm"] == analysis.FULL)
    # Full seed values 1,2,8 have mean11/3 and sample SD sqrt(43/3), including adverse seed62.
    score = full_taxi["test"]["overall_qty_rmse"]
    assert score["seeds"] == [42, 52, 62] and score["n"] == 3
    assert score["mean"] == pytest.approx(11 / 3)
    assert score["sample_sd"] == pytest.approx((43 / 3) ** 0.5)
    direction = next(row for row in payload["direction_counts"]
                     if row["study"] == "history" and row["dataset"] == "taxi"
                     and row["control_arm"] == analysis.ARMS["history"][1])
    assert direction["metrics"]["overall_qty_rmse"] == {
        "improved": 2, "equal": 0, "worsened": 1, "validation_to_test_sign_flips": 1}
    assert direction["all_three_seeds_Test_RMSE_strict_improvement"] is False
    assert payload["statistical_significance_established"] is False
    pair = next(row for row in payload["paired_comparisons"]
                if row["study"] == "history" and row["dataset"] == "taxi" and row["seed"] == 62
                and row["control_arm"] == analysis.ARMS["history"][1])
    assert pair["validation_reference_delta"]["overall_qty_rmse"] < 0
    assert pair["test_delta"]["overall_qty_rmse"] > 0
    assert all(row["test"]["split"] == "test" and row["validation_reference"]["split"] == "validation"
               for row in payload["conditions"])
    report = (bundle / "analysis/TEST_FINAL_COMPARISON.md").read_text()
    assert "손해" in report and "부호 반전" in report and "새 독립 확인" in report
    assert "단순 수량 기준선" in report
    for name, expected_sha in receipt["analysis_files"].items():
        assert analysis.sha(bundle / name) == expected_sha


@pytest.mark.parametrize("field,value", [
    ("checkpoint_sha256", "9" * 64), ("frozen_source_closure", "9" * 64),
    ("contract_sha256", "9" * 64), ("selected_epoch", 99), ("seed", 62),
    ("retrained", True), ("reselected", True), ("test_is_independent", True),
])
def test_frozen_binding_or_scope_changes_rejected_before_output(campaign, field, value):
    bundle, _ = campaign
    alter_receipt(campaign, lambda receipt: receipt.update({field: value}))
    with pytest.raises(analysis.AnalysisError):
        analysis.analyze(bundle)
    assert not (bundle / "analysis").exists()


@pytest.mark.parametrize("split", ["validation_reference", "validation_replay", "test"])
def test_wrong_split_and_nonfinite_rejected(campaign, split):
    bundle, _ = campaign
    alter_receipt(campaign, lambda receipt: receipt[split].update(split="test" if split != "test" else "validation"))
    with pytest.raises(analysis.AnalysisError, match="split"):
        analysis.analyze(bundle)
    alter_receipt(campaign, lambda receipt: receipt[split].update(
        split="test" if split == "test" else "validation", qty_rmse=float("nan")))
    with pytest.raises(analysis.AnalysisError, match="nonfinite"):
        analysis.analyze(bundle)


def test_validation_reference_change_or_failed_replay_rejected(campaign):
    bundle, _ = campaign
    alter_receipt(campaign, lambda receipt: receipt["validation_reference"].update(qty_rmse=999))
    with pytest.raises(analysis.AnalysisError, match="reference metrics"):
        analysis.analyze(bundle)
    alter_receipt(campaign, lambda receipt: receipt["validation_reference"].update(qty_rmse=2))
    alter_receipt(campaign, lambda receipt: receipt["validation_replay_check"].update(passed=False))
    with pytest.raises(analysis.AnalysisError, match="replay did not pass"):
        analysis.analyze(bundle)


def test_replay_tolerance_uses_math_isclose_max_rule_and_cannot_trust_passed_boolean(campaign):
    bundle, contract = campaign
    # Reference2 gives max tolerance2e-5;2.5e-5 would pass an incorrect additive3e-5 rule.
    alter_receipt(campaign, lambda receipt: receipt["validation_replay"].update(qty_rmse=2.000025))
    with pytest.raises(analysis.AnalysisError, match="tolerance exceeded"):
        analysis.analyze(bundle)
    alter_receipt(campaign, lambda receipt: receipt["validation_replay"].update(qty_rmse=2.000019))
    path = bundle / "results" / contract["jobs"][0]["id"] / "validation_replay.json"
    native = analysis.read(path)
    native["metrics"]["qty_rmse"] = 2.000019
    dump(path, native)
    refresh_gate(bundle, contract)
    assert analysis.analyze(bundle)["status"] == "complete"


def test_incomplete_or_duplicate_matrix_rejected(campaign):
    bundle, contract = campaign
    contract["jobs"].pop()
    reseal_contract(bundle, contract)
    with pytest.raises(analysis.AnalysisError, match="36"):
        analysis.analyze(bundle)
    contract["jobs"].append(copy.deepcopy(contract["jobs"][0]))
    reseal_contract(bundle, contract)
    with pytest.raises(analysis.AnalysisError, match="Duplicate"):
        analysis.analyze(bundle)


@pytest.mark.parametrize("mutation", ["count", "tail_count", "identity", "baseline"])
def test_population_or_simple_baseline_mismatch_rejected(campaign, mutation):
    bundle, contract = campaign
    def mutate(receipt):
        if mutation == "count":
            receipt["test"]["count"] += 1
            receipt.pop("simple_quantity_baselines")
        elif mutation == "tail_count":
            receipt["test"]["tail"]["count"] += 1
        elif mutation == "identity":
            receipt["data_identity"]["target_identity_sha256"] = "8" * 64
        else:
            receipt["simple_quantity_baselines"]["last"]["qty_rmse"] += 1
    alter_receipt(campaign, mutate)
    if mutation in ("count", "tail_count"):
        folder = bundle / "results" / contract["jobs"][0]["id"]
        native = analysis.read(folder / "test_metrics.json")
        native["metrics"] = analysis.read(folder / "receipt.json")["test"]
        dump(folder / "test_metrics.json", native)
    with pytest.raises(analysis.AnalysisError, match="count|identity|baseline"):
        analysis.analyze(bundle)


@pytest.mark.parametrize("change", ["missing", "incomplete", "sha", "completion_order", "exit"])
def test_global_validation_gate_is_required_and_bound_to_all36_successful_originals(campaign, change):
    bundle, contract = campaign
    gate_path = bundle / "validation_gate.json"
    gate = analysis.read(gate_path)
    job_id = contract["jobs"][0]["id"]
    folder = bundle / "results" / job_id
    if change == "missing":
        gate_path.unlink()
        expected = FileNotFoundError
    else:
        expected = analysis.AnalysisError
        if change == "incomplete":
            gate["validation_receipts"].pop(job_id)
            dump(gate_path, gate)
        elif change == "sha":
            gate["validation_receipts"][job_id] = "0" * 64
            dump(gate_path, gate)
        elif change == "completion_order":
            native = analysis.read(folder / "validation_replay.json")
            native["completed_unix"] = 21
            dump(folder / "validation_replay.json", native)
            refresh_gate(bundle, contract)
        else:
            worker = analysis.read(folder / "validation_worker_exit.json")
            worker["exit_code"] = 1
            dump(folder / "validation_worker_exit.json", worker)
    with pytest.raises(expected):
        analysis.analyze(bundle)
    assert not (bundle / "analysis").exists()


def test_test_before_global_validation_gate_or_swapped_native_output_rejected(campaign):
    bundle, contract = campaign
    alter_receipt(campaign, lambda receipt: receipt.update(test_started_unix=19))
    with pytest.raises(analysis.AnalysisError, match="started before"):
        analysis.analyze(bundle)
    alter_receipt(campaign, lambda receipt: receipt.update(test_started_unix=30))
    path = bundle / "results" / contract["jobs"][0]["id"] / "test_metrics.json"
    native = analysis.read(path)
    native["checkpoint_sha256"] = "9" * 64
    dump(path, native)
    with pytest.raises(analysis.AnalysisError, match="Native frozen checkpoint"):
        analysis.analyze(bundle)


def test_missing_or_failed_actual_worker_exit_rejected(campaign):
    bundle, contract = campaign
    path = bundle / "results" / contract["jobs"][0]["id"] / "worker_exit.json"
    worker = analysis.read(path)
    worker["exit_code"] = 1
    dump(path, worker)
    with pytest.raises(analysis.AnalysisError, match="worker exit"):
        analysis.analyze(bundle)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        analysis.analyze(bundle)
    assert not (bundle / "analysis").exists()


def test_completed_cache_verifies_every_raw_input_and_never_overwrites(campaign):
    bundle, contract = campaign
    first = analysis.analyze(bundle)
    raw = (bundle / "analysis/completion_receipt.json").read_bytes()
    assert analysis.analyze(bundle) == first
    assert (bundle / "analysis/completion_receipt.json").read_bytes() == raw
    alter_receipt(campaign, lambda receipt: receipt["test"].update(qty_rmse=500))
    with pytest.raises(analysis.AnalysisError, match="bytes changed"):
        analysis.analyze(bundle)
    assert (bundle / "analysis/completion_receipt.json").read_bytes() == raw


def test_incomplete_finalization_directory_is_preserved(campaign):
    bundle, _ = campaign
    (bundle / "analysis").mkdir()
    sentinel = bundle / "analysis/failure.json"
    dump(sentinel, {"failure": "interrupted"})
    raw = sentinel.read_bytes()
    with pytest.raises(analysis.AnalysisError, match="no automatic overwrite"):
        analysis.analyze(bundle)
    assert sentinel.read_bytes() == raw


def test_prepared_empty_analysis_directory_is_allowed_with_exclusive_claim(campaign):
    bundle, _ = campaign
    (bundle / "analysis").mkdir()
    receipt = analysis.analyze(bundle)
    assert receipt["status"] == "complete"
    claim = analysis.read(bundle / "analysis_claim.json")
    assert claim["contract_sha256"] == receipt["contract_sha256"]
    assert receipt["analysis_files"]["analysis_claim.json"] == analysis.sha(bundle / "analysis_claim.json")
    assert analysis.analyze(bundle) == receipt


def test_existing_claim_blocks_even_empty_output_and_is_preserved(campaign):
    bundle, _ = campaign
    (bundle / "analysis").mkdir()
    path = bundle / "analysis_claim.json"
    dump(path, {"owner": "previous-finalizer"})
    original = path.read_bytes()
    with pytest.raises(analysis.AnalysisError, match="Existing analysis claim"):
        analysis.analyze(bundle)
    assert path.read_bytes() == original and not any((bundle / "analysis").iterdir())


@pytest.mark.parametrize("missing", ["identities", "baseline", "one_baseline"])
def test_missing_identity_or_baselines_cannot_pass_uniformly_across_all36(campaign, missing):
    bundle, contract = campaign
    for job in contract["jobs"]:
        path = bundle / "results" / job["id"] / "receipt.json"
        row = analysis.read(path)
        if missing == "identities":
            row["data_identity"] = {}
        elif missing == "baseline":
            row.pop("simple_quantity_baselines")
        else:
            row["simple_quantity_baselines"].pop("mean")
        dump(path, row)
    with pytest.raises(analysis.AnalysisError, match="SHA256|baselines"):
        analysis.analyze(bundle)
    assert not (bundle / "analysis_claim.json").exists()


@pytest.mark.parametrize("identity", ["raw_sha256", "target_identity_sha256", "validation_target_identity_sha256",
                                      "test_truth_sha256", "validation_truth_sha256"])
def test_uniform_combined_identity_drift_rejected_against_contract_and_native(campaign, identity):
    bundle, contract = campaign
    for job in contract["jobs"]:
        path = bundle / "results" / job["id"] / "receipt.json"
        row = analysis.read(path)
        row["data_identity"][identity] = "9" * 64
        dump(path, row)
    with pytest.raises(analysis.AnalysisError, match="SHA|identity"):
        analysis.analyze(bundle)


def test_fabricated_simple_timing_or_changed_contract_canonical_hash_rejected(campaign):
    bundle, contract = campaign
    alter_receipt(campaign, lambda row: row["simple_quantity_baselines"]["last"].update(
        quantity_cells=[{"time_nll": 0.0}]))
    with pytest.raises(analysis.AnalysisError, match="invent TimeNLL"):
        analysis.analyze(bundle)
    contract["datasets"]["taxi"]["expected_test_count"] = 101
    dump(bundle / "contract.json", contract)
    with pytest.raises(analysis.AnalysisError, match="canonical SHA"):
        analysis.analyze(bundle)
