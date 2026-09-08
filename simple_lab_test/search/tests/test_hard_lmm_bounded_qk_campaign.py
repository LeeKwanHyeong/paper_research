"""Validate queue gates and source admission without starting GPU processes."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from paper.scripts import run_hard_lmm_bounded_qk_campaign as campaign
from paper.scripts import run_backbone_normalized_duration as normalized


def contract():
    return campaign.read_json(campaign.CONTRACT_PATH)


def test_fixed_campaign_contract():
    campaign.validate_contract(contract())


@pytest.mark.parametrize("change", ["time", "quantity", "held_out", "epochs", "order"])
def test_changed_policy_is_rejected(change):
    value = contract()
    if change == "time":
        value["time_gate"]["normalized_NLL_delta_vs_aligned_B_max"] = .11
    elif change == "quantity":
        value["quantity_gate"]["B"]["raw_rmse_ratio_strictly_less_than"] = 1.01
    elif change == "held_out":
        value["held_out_test_evaluated"] = True
    elif change == "epochs":
        value["phases"]["seed42_screening"]["epochs"] = 500
    else:
        value["phases"]["seed42_screening"]["datasets_in_order"].reverse()
    with pytest.raises(ValueError):
        campaign.validate_contract(value)


def run_queue(*, quantity_failure=False, time_failure=False, error=False):
    config = contract()
    metric_names = ("raw_rmse", "overall_mae", "body_mae", "gt_p99_mae")
    for references in config["datasets"].values():
        for role in ("B", "FULL"):
            references[role]["metrics"] = {key: 10. for key in metric_names}
        references["aligned_B"]["normalized_time_nll"] = 1.
    events = []

    def joint(dataset, phase):
        events.append(("joint", dataset, phase))
        if error and len(events) == 2:
            raise RuntimeError("simulated child failure")
        # Deliberately terrible e1 numbers must never enter a performance gate.
        value = 1000. if phase == "e1" else 9.
        if quantity_failure and phase == "seed42_screening":
            value = 10.
        return {"status": "passed", "metrics": {key: value for key in metric_names}}

    def duration(dataset, role, full_fit, audit):
        events.append(("duration", dataset, role, full_fit))
        assert (audit is not None) == (role == "BOUNDED")
        # Taxi FULL improvement claim fails, but B+0.01 still passes.
        nll = 1.001 if role == "BOUNDED" else 1.
        if time_failure and role == "BOUNDED" and full_fit:
            nll = 1.02
        return {"status": "passed", "proper_time_nll": nll}

    def record(dataset, decision):
        events.append(("gate", dataset, copy.deepcopy(decision)))

    return config, events, joint, duration, record


def test_all_e1_precede_screening_and_no_B_fit_or_extra_seeds():
    config, events, joint, duration, record = run_queue()
    result = campaign.execute_phases(config, joint=joint, duration=duration, record_gate=record)
    assert result["status"] == "seed42_all_datasets_passed"
    assert result["additional_seeds_launched"] is False
    assert result["final_adoption"] is False
    assert [event[0] for event in events[:6]] == ["joint"] * 3 + ["duration"] * 3
    assert all(event[2] == "e1" for event in events[:3])
    assert all(event[2:] == ("FULL", False) for event in events[3:6])
    assert [event[1] for event in events if event[0] == "joint" and event[2] == "seed42_screening"] == [
        "insta_market_basket", "yellow_trip_hourly", "intermittent_frozen_5000"]
    assert not any(event[0] == "duration" and event[2] == "B" for event in events)
    assert result["decisions"]["yellow_trip_hourly"]["normalized_time"]["taxi_FULL_improvement_claim"]["passed"] is False


def test_quantity_failure_stops_before_full_duration_or_next_dataset():
    config, events, joint, duration, record = run_queue(quantity_failure=True)
    result = campaign.execute_phases(config, joint=joint, duration=duration, record_gate=record)
    assert result["status"] == "stopped_quantity_gate_failed"
    assert result["failed_dataset"] == "insta_market_basket"
    assert len([event for event in events if event[0] == "joint"]) == 4
    assert len([event for event in events if event[0] == "duration"]) == 3


def test_time_failure_stops_before_next_dataset():
    config, events, joint, duration, record = run_queue(time_failure=True)
    result = campaign.execute_phases(config, joint=joint, duration=duration, record_gate=record)
    assert result["status"] == "stopped_normalized_time_gate_failed"
    assert result["failed_dataset"] == "insta_market_basket"
    assert len([event for event in events if event[0] == "joint"]) == 4
    assert len([event for event in events if event[0] == "duration"]) == 6


def test_child_error_does_not_continue_queue():
    config, events, joint, duration, record = run_queue(error=True)
    with pytest.raises(RuntimeError, match="child failure"):
        campaign.execute_phases(config, joint=joint, duration=duration, record_gate=record)
    assert len(events) == 2


@pytest.mark.parametrize("dataset", list(campaign.DATASETS))
def test_full_duration_manifest_matches_existing_evaluator(dataset):
    config = contract()
    revision = "a" * 40
    deployment = {"source_revision": revision, "runtime": normalized.runtime_identity(__import__('torch').device('cpu'))}
    manifest = campaign.duration_manifest(dataset=dataset, role="FULL", source=config["datasets"][dataset]["FULL"]["source"], deployment=deployment)
    aligned = normalized.validate_aligned_contract(campaign.read_json(normalized.DEFAULT_ALIGNED_CONTRACT))
    checked = normalized.validate_job_manifest(
        manifest, contract=campaign.read_json(normalized.DEFAULT_CONTRACT), aligned_dataset=aligned[dataset],
        execution_contract_sha256=normalized.sha256_file(normalized.DEFAULT_CONTRACT),
        aligned_contract_sha256=normalized.sha256_file(normalized.DEFAULT_ALIGNED_CONTRACT),
        evaluation_source_revision=revision, expected_runtime_identity=deployment["runtime"],
    )
    assert checked["model_role"] == "FULL"
    assert checked["dataset"] == dataset


def test_source_verification_rejects_dirty_import_and_unregistered_python(tmp_path, monkeypatch):
    helper = tmp_path / "models/helper.py"
    helper.parent.mkdir()
    helper.write_text("x = 1\n")
    config = tmp_path / "contract.json"
    config.write_text('{}\n')
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "CONTRACT_PATH", config)
    revision = "a" * 40
    def git(command, **kwargs):
        return revision + "\n" if "rev-parse" in command else b'models/helper.py\0contract.json\0'
    monkeypatch.setattr(campaign.subprocess, "check_output", git)
    manifest = {"schema_version": 1, "host_role": "5090", "status": "frozen_before_gpu_outputs",
                "source_revision": revision, "source_root": str(tmp_path),
                "execution_contract_sha256": campaign.sha256(config),
                "source_files": {str(p.relative_to(tmp_path)): campaign.sha256(p) for p in (helper, config)}}
    assert campaign.verify_source(manifest)["verified_files"] == 2
    helper.write_text("x = 2\n")
    with pytest.raises(ValueError, match="Source file drift"):
        campaign.verify_source(manifest)
    helper.write_text("x = 1\n")
    (helper.parent / "unregistered.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="Unregistered Python"):
        campaign.verify_source(manifest)
