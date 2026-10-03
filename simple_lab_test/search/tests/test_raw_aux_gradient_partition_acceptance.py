"""Synthetic JSON-only checks for the parallel evaluation boundary."""
from copy import deepcopy

import pytest

from paper.scripts import raw_aux_gradient_acceptance as original
from paper.scripts import raw_aux_gradient_partition_acceptance as partition
from paper.scripts.time_quantity_diagnostic import _sha_json
from simple_lab_test.search.tests.test_raw_aux_gradient_acceptance import gate_inputs


def fixture():
    source = {"files": {"paper/scripts/quantity_comparison_engine.py": "1" * 64}}
    source["files_sha256"] = original._canonical(source["files"])
    contract = {"schema": "raw_aux_gradient_parallel_execution_v1", "design_sha256": partition.DESIGN_FILE_SHA256,
                "source": source, "hosts": {
                    "5090": {"assigned_arms": [partition.B, partition.CAPPED], "runtime_expected": {"kind": "synthetic-5090"}},
                    "5080": {"assigned_arms": [partition.U], "runtime_expected": {"kind": "synthetic-5080"}}}}
    arms = gate_inputs()
    for name, arm in arms.items():
        host = partition.OWNERS[name]
        arm["condition"].update(mu=1., raw_scale=2.)
        arm["execution_identity"] = {"host_alias": host, "execution_contract_sha256": original._canonical(contract),
                                     "raw_aux_design_sha256": partition.DESIGN_FILE_SHA256, "source": deepcopy(source),
                                     "runtime": deepcopy(contract["hosts"][host]["runtime_expected"]),
                                     "data": {"sha256": "2" * 64}}
        for key, value in (("raw_quantity_rmse", arm["primary"]["metrics"]["raw_quantity_rmse"]),
                           ("legacy_time_loss", arm["last120"]["metrics"]["legacy_time_loss"])):
            arm["selectors"][key]["best_value"] = value
        arm["training_contract"] = {**arm["training_exposure"], "initial_state_sha256": arm["initial_state_sha256"],
                                    "condition": deepcopy(arm["condition"]), "identity": deepcopy(arm["execution_identity"]),
                                    "runtime_observed": deepcopy(arm["execution_identity"]["runtime"]),
                                    "engine_source_sha256": "1" * 64, "lr": .001,
                                    "model": {"hidden_dim": 8}, "train_loader": {"batch_size": 3}}
        resign(arm)
    return arms, contract


def resign(arm):
    arm["training_contract_sha256"] = _sha_json(arm["training_contract"])
    arm["training_contract_common_sha256"] = _sha_json({key: value for key, value in arm["training_contract"].items() if key != "condition"})


def test_two_host_gate_matches_original_capped_gate_without_changing_identity():
    arms, contract = fixture()
    unchanged = deepcopy(arms)
    result = partition.assess_partition(arms, partition_contract=contract, synthetic=True)
    expected = [row for row in original.assess_acceptance(gate_inputs(), synthetic=True)["checks"] if row["candidate"] == partition.CAPPED]
    assert result["checks"] == expected
    assert result["primary_gate_passed"] is True
    assert result["eligible_candidate"] is None and result["status"] == "synthetic_only"
    assert result["native_execution_identities"][partition.U]["host_alias"] == "5080"
    assert result["native_execution_identities"][partition.B]["host_alias"] == "5090"
    assert arms == unchanged
    assert len(result["paired_comparisons"]) == 6
    assert sum(row["descriptive_only"] for row in result["paired_comparisons"]) == 4
    assert not result["uncapped_used_for_candidate_selection"]


def test_uncapped_quality_failure_never_changes_primary_gate():
    arms, contract = fixture()
    worse = gate_inputs(candidate_errors=[100.] * 6)[partition.U]
    for scope in original.SCOPES:
        arms[partition.U][scope] = worse[scope]
    arms[partition.U]["selectors"]["raw_quantity_rmse"]["best_value"] = 100.
    result = partition.assess_partition(arms, partition_contract=contract, synthetic=True)
    assert result["primary_gate_passed"] is True
    assert all(row["candidate"] == partition.CAPPED for row in result["checks"])
    assert result["uncapped_gate_status"] == "descriptive_only"


def test_quantity_rmse_harm_still_fails_same_host_primary_gate():
    arms, contract = fixture()
    worse = gate_inputs(candidate_errors=[0, 0, 0, 1.5, 0, 0])[partition.CAPPED]
    for scope in original.SCOPES:
        arms[partition.CAPPED][scope] = worse[scope]
    arms[partition.CAPPED]["selectors"]["raw_quantity_rmse"]["best_value"] = worse["primary"]["metrics"]["raw_quantity_rmse"]
    result = partition.assess_partition(arms, partition_contract=contract, synthetic=True)
    assert result["primary_gate_passed"] is False
    assert any(row["metric"] == "quantity_bin_3_RMSE" and not row["passed"] for row in result["checks"])


@pytest.mark.parametrize("missing", [partition.U, partition.B, partition.CAPPED])
def test_incomplete_suite_never_promoted(missing):
    arms, contract = fixture()
    arms.pop(missing)
    result = partition.assess_partition(arms, partition_contract=contract, synthetic=True)
    assert result["status"] == "suite_incomplete" and result["eligible_candidate"] is None
    assert result["missing_arms"] == [missing]
    assert result["primary_gate_status"] == ("complete" if missing == partition.U else "pending")
    assert result["primary_gate_passed"] is (True if missing == partition.U else None)


@pytest.mark.parametrize("change", ["owner", "runtime", "source", "initial", "calibration", "batch", "data", "optimizer", "population", "selector", "heldout", "fourth"])
def test_cross_host_invariant_and_authority_drift_fails_closed(change):
    arms, contract = fixture()
    arm = arms[partition.U]
    if change == "owner": arm["execution_identity"]["host_alias"] = "5090"
    elif change == "runtime": arm["execution_identity"]["runtime"] = {"kind": "other-runtime"}
    elif change == "source": arm["execution_identity"]["source"]["files"]["paper/scripts/quantity_comparison_engine.py"] = "0" * 64
    elif change == "initial":
        arm["initial_state_sha256"] = arm["training_contract"]["initial_state_sha256"] = "0" * 64
        resign(arm)
    elif change == "calibration":
        arm["condition"]["mixed_objective"]["calibration_sha256"] = "0" * 64
        arm["training_contract"]["condition"] = deepcopy(arm["condition"])
        resign(arm)
    elif change == "batch": arm["train_batch_order_sha256"][1] = "0" * 64
    elif change == "data":
        arm["execution_identity"]["data"] = {"sha256": "0" * 64}
        arm["training_contract"]["identity"] = deepcopy(arm["execution_identity"])
        resign(arm)
    elif change == "optimizer":
        arm["training_contract"]["lr"] = .002
        resign(arm)
    elif change == "population": arm["primary"]["ordered_target_sha256"] = "0" * 64
    elif change == "selector": arm["selectors"]["legacy_time_loss"]["global_step"] = 3
    elif change == "heldout": arm["held_out_evaluated"] = True
    elif change == "fourth": arms["extra"] = deepcopy(arm)
    with pytest.raises(ValueError):
        partition.assess_partition(arms, partition_contract=contract, synthetic=True)


def test_missing_or_duplicated_assignment_rejected():
    arms, contract = fixture()
    contract["hosts"]["5080"]["assigned_arms"].append(partition.B)
    with pytest.raises(ValueError, match="ownership"):
        partition.assess_partition(arms, partition_contract=contract, synthetic=True)


def test_actual_training_contract_digest_is_checked():
    arms, contract = fixture()
    arms[partition.CAPPED]["training_contract"]["lr"] = .002
    with pytest.raises(ValueError, match="digest"):
        partition.assess_partition(arms, partition_contract=contract, synthetic=True)


def test_historical_permit_uses_its_start_time_not_analysis_wall_clock(monkeypatch):
    from paper.scripts import raw_aux_parallel_common as common
    arms, contract = fixture()
    arm = arms[partition.B]
    runtime = {"torch": "synthetic-torch", "numpy": "synthetic-numpy", "cuda": "synthetic-cuda", "threads": 1}
    arm["execution_identity"]["runtime"] = runtime
    arm["training_contract"]["identity"] = deepcopy(arm["execution_identity"])
    arm["training_contract"]["runtime_observed"] = deepcopy(runtime)
    resign(arm)
    permit = {"started_at_unix": 100., "deadline_unix": 200.}
    seen = []
    def validate(c, p, host, *, now):
        seen.append(now)
        assert c is contract and p is permit and host == "5090"
        return {"runtime": arm["execution_identity"]["runtime"], "initial_state_sha256": arm["initial_state_sha256"]}
    monkeypatch.setattr(common, "validate_permit", validate)
    partition._runtime_and_contract(arm, contract, permit, synthetic=False)
    assert seen == [100.]
    with pytest.raises(ValueError, match="start permit"):
        partition._runtime_and_contract(arm, contract, None, synthetic=False)
    arm["training_contract"]["runtime_observed"]["torch"] = "other-torch"
    resign(arm)
    with pytest.raises(ValueError, match="Actual training runtime"):
        partition._runtime_and_contract(arm, contract, permit, synthetic=False)


def test_training_engine_hash_must_reference_actual_frozen_source():
    arms, contract = fixture()
    arm = arms[partition.U]
    arm["training_contract"]["engine_source_sha256"] = "0" * 64
    resign(arm)
    with pytest.raises(ValueError, match="engine source"):
        partition.assess_partition(arms, partition_contract=contract, synthetic=True)


def test_runtime_receipt_cannot_be_replaced_with_another_hosts_receipt(monkeypatch):
    from paper.scripts import raw_aux_parallel_common as common
    arms, contract = fixture()
    arm = arms[partition.U]
    monkeypatch.setattr(common, "validate_permit", lambda *a, **k: {
        "runtime": {"kind": "synthetic-5090"}, "initial_state_sha256": arm["initial_state_sha256"]})
    with pytest.raises(ValueError, match="native qualification"):
        partition._runtime_and_contract(arm, contract, {"started_at_unix": 100.}, synthetic=False)
