"""Validate the LPHC campaign queue without starting GPU processes."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from paper.scripts import run_hard_lmm_level_history_qkv_campaign as campaign


def contract() -> dict:
    return campaign.read_json(campaign.CONTRACT_PATH)


def fresh_identity() -> dict:
    digest = "a" * 64
    return {
        "method": "same_seed_fresh_B_identity_v1",
        "seed": 42,
        "B_backbone": "titantpp",
        "candidate_backbone": "titantpp_hard_memory_level_history_qkv",
        "fresh_B_state_sha256": digest,
        "candidate_common_state_sha256": digest,
        "candidate_initial_state_sha256": "b" * 64,
        "new_parameter_state_keys": [
            "encoder.layers.0.attn.level_history_q_kernel",
            "encoder.layers.0.attn.level_history_k_kernel",
            "encoder.layers.0.attn.level_history_v_kernel",
        ],
        "new_parameters_exact_zero": True,
        "additional_parameter_count": 384,
        "inherited_state_bitwise_equal": True,
        "synthetic_outputs_bitwise_equal": True,
        "post_construction_RNG_state_bitwise_equal": True,
        "summary_and_checkpoint_initial_state_sha256_equal": True,
    }


def test_fixed_campaign_contract_and_digest() -> None:
    assert campaign.sha256(campaign.CONTRACT_PATH) == campaign.CONTRACT_SHA256
    campaign.validate_contract(contract())


@pytest.mark.parametrize(
    "change",
    [
        "held_out",
        "automatic_extra_seed",
        "pretrained_initialization",
        "e1_order",
        "mechanism_order",
        "quantity_gate",
        "additional_seed_contract",
    ],
)
def test_changed_campaign_policy_is_rejected(change: str) -> None:
    value = contract()
    if change == "held_out":
        value["held_out_test_evaluated"] = True
    elif change == "automatic_extra_seed":
        value["automatic_additional_seed_launch"] = True
    elif change == "pretrained_initialization":
        value["initialization_policy"]["pretrained_B_checkpoint_injected"] = True
    elif change == "e1_order":
        value["phases"]["e1"]["datasets_in_order"].reverse()
    elif change == "mechanism_order":
        value["phases"]["seed42_screening"][
            "instacart_mechanism_gate_before_other_datasets"
        ] = False
    elif change == "quantity_gate":
        value["quantity_gate"]["B"]["raw_rmse_ratio_strictly_less_than"] = 1.01
    else:
        value["phases"]["additional_seeds"][
            "requires_frozen_matched_seed_reference_contract_before_launch"
        ] = False
    with pytest.raises(ValueError):
        campaign.validate_contract(value)


@pytest.mark.parametrize("dataset", list(campaign.DATASETS))
@pytest.mark.parametrize("phase_name", ["e1", "seed42_screening"])
def test_training_command_is_fresh_same_seed_and_has_no_checkpoint_injection(
    dataset: str, phase_name: str, tmp_path: Path
) -> None:
    value = contract()
    command = campaign.job_command(
        python="/frozen/python",
        candidate=value["candidate"],
        dataset=dataset,
        output=tmp_path / dataset,
        source_revision="a" * 40,
        phase=value["phases"][phase_name],
        host_role="5090",
    )
    assert "--seeds" in command
    assert command[command.index("--seeds") + 1] == "42"
    assert command[command.index("--backbones") + 1] == value["candidate"]["backbone"]
    assert command[command.index("--epochs") + 1] == str(value["phases"][phase_name]["epochs"])
    assert not any(token.startswith("--initial-model") for token in command)
    assert not any("checkpoint.pt" in token for token in command)


def test_fresh_initial_identity_requires_exact_route_rng_output_and_zero_proofs() -> None:
    expected = fresh_identity()
    assert campaign.validate_fresh_initial_identity(expected) == expected
    with pytest.raises(ValueError, match="schema"):
        campaign.validate_fresh_initial_identity({**expected, "unregistered": True})
    for name in (
        "new_parameters_exact_zero",
        "inherited_state_bitwise_equal",
        "synthetic_outputs_bitwise_equal",
        "post_construction_RNG_state_bitwise_equal",
        "summary_and_checkpoint_initial_state_sha256_equal",
    ):
        corrupted = dict(expected)
        corrupted[name] = False
        with pytest.raises(ValueError, match="identity proof"):
            campaign.validate_fresh_initial_identity(corrupted)
    corrupted = dict(expected)
    corrupted["candidate_common_state_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="common-state"):
        campaign.validate_fresh_initial_identity(corrupted)


def queue(
    *, quantity_fail: str | None = None, mechanism_status: str = "passed",
    child_error_at: int | None = None,
):
    value = contract()
    metric_names = ("raw_rmse", "overall_mae", "body_mae", "gt_p99_mae")
    for references in value["datasets"].values():
        for role in ("B", "FULL"):
            references[role]["metrics"] = {name: 10.0 for name in metric_names}
    events: list[tuple] = []

    def joint(dataset: str, phase: str) -> dict:
        events.append(("joint", dataset, phase))
        if child_error_at is not None and len(events) == child_error_at:
            raise RuntimeError("simulated child failure")
        # Bad e1 metrics demonstrate that e1 is execution-only. Seed42 9 passes.
        actual = 1000.0 if phase == "e1" else 9.0
        if phase == "seed42_screening" and dataset == quantity_fail:
            actual = 10.0
        return {
            "status": "passed",
            "summary": "/not/used/in/pure/queue.json",
            "metrics": {name: actual for name in metric_names},
        }

    def mechanism(audit: dict) -> dict:
        events.append(("mechanism", audit["metrics"]["raw_rmse"]))
        return {"status": mechanism_status, "checks": {"paired": mechanism_status == "passed"}}

    def record(dataset: str, decision: dict) -> None:
        events.append(("gate", dataset, copy.deepcopy(decision)))

    return value, events, joint, mechanism, record


def test_all_pass_stops_pending_time_and_extra_seed_contract() -> None:
    value, events, joint, mechanism, record = queue()
    result = campaign.execute_phases(
        value, joint=joint, mechanism=mechanism, record_gate=record
    )
    assert result["status"] == campaign.PENDING_STATUS
    assert result["normalized_time_launched"] is False
    assert result["additional_seeds_launched"] is False
    assert result["final_adoption"] is False
    assert len(result["pending_requirements"]) == 2
    assert [event for event in events if event[0] == "joint"] == [
        ("joint", "yellow_trip_hourly", "e1"),
        ("joint", "intermittent_frozen_5000", "e1"),
        ("joint", "insta_market_basket", "e1"),
        ("joint", "insta_market_basket", "seed42_screening"),
        ("joint", "yellow_trip_hourly", "seed42_screening"),
        ("joint", "intermittent_frozen_5000", "seed42_screening"),
    ]
    assert len([event for event in events if event[0] == "mechanism"]) == 1


def test_instacart_quantity_failure_still_runs_mechanism_then_stops_training() -> None:
    value, events, joint, mechanism, record = queue(quantity_fail="insta_market_basket")
    result = campaign.execute_phases(
        value, joint=joint, mechanism=mechanism, record_gate=record
    )
    assert result["status"] == "stopped_instacart_quantity_or_mechanism_gate_failed"
    assert result["failed_gates"] == ["quantity"]
    assert len([event for event in events if event[0] == "mechanism"]) == 1
    assert [event for event in events if event[0] == "joint"][-1] == (
        "joint", "insta_market_basket", "seed42_screening"
    )


def test_instacart_records_both_failed_gates_before_stopping() -> None:
    value, events, joint, mechanism, record = queue(
        quantity_fail="insta_market_basket", mechanism_status="failed"
    )
    result = campaign.execute_phases(
        value, joint=joint, mechanism=mechanism, record_gate=record
    )
    assert result["failed_gates"] == ["quantity", "mechanism"]
    final_record = [event for event in events if event[0] == "gate"][-1][2]
    assert set(final_record) == {"quantity", "mechanism"}


def test_instacart_mechanism_failure_stops_before_taxi() -> None:
    value, events, joint, mechanism, record = queue(mechanism_status="failed")
    result = campaign.execute_phases(
        value, joint=joint, mechanism=mechanism, record_gate=record
    )
    assert result["failed_gates"] == ["mechanism"]
    assert not any(
        event[:3] == ("joint", "yellow_trip_hourly", "seed42_screening")
        for event in events
    )


def test_taxi_failure_stops_before_intermittent_seed42() -> None:
    value, events, joint, mechanism, record = queue(quantity_fail="yellow_trip_hourly")
    result = campaign.execute_phases(
        value, joint=joint, mechanism=mechanism, record_gate=record
    )
    assert result["status"] == "stopped_quantity_gate_failed"
    assert result["failed_dataset"] == "yellow_trip_hourly"
    assert not any(
        event[:3] == ("joint", "intermittent_frozen_5000", "seed42_screening")
        for event in events
    )


def test_child_error_does_not_continue_queue() -> None:
    value, events, joint, mechanism, record = queue(child_error_at=2)
    with pytest.raises(RuntimeError, match="child failure"):
        campaign.execute_phases(
            value, joint=joint, mechanism=mechanism, record_gate=record
        )
    assert len(events) == 2


def test_execution_failure_marks_active_job_and_campaign_fail_closed() -> None:
    status = {"status": "running_job", "current_job": "job", "jobs": {"job": {"status": "running"}}}
    campaign.record_execution_failure(status, KeyboardInterrupt())
    assert status["status"] == "failed_execution"
    assert status["jobs"]["job"]["status"] == "failed"
    assert "KeyboardInterrupt" in status["error"]


def test_mechanism_command_binds_deployment_cache_candidate_and_cuda(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = run_dir / "summary.json"
    checkpoint = run_dir / "best_val_qty_rmse_model.pt"
    candidate_job_audit = run_dir / "campaign_audit.json"
    summary.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    candidate_job_audit.write_text('{"status":"passed"}\n', encoding="utf-8")
    deployment = tmp_path / "deployment.json"
    deployment.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "mechanism"
    command = campaign.mechanism_command(
        python="/frozen/python",
        deployment_manifest=deployment,
        deployment_sha256="d" * 64,
        contract=contract(),
        candidate_audit={
            "summary": str(summary),
            "campaign_audit_path": str(candidate_job_audit),
        },
        output=output,
    )
    pairs = dict(zip(command[3::2], command[4::2]))
    assert pairs["--expected-deployment-manifest-sha256"] == "d" * 64
    assert pairs["--candidate-checkpoint"] == str(checkpoint)
    assert pairs["--expected-candidate-checkpoint-sha256"] == campaign.sha256(checkpoint)
    assert pairs["--expected-candidate-summary-sha256"] == campaign.sha256(summary)
    assert pairs["--candidate-job-audit"] == str(candidate_job_audit)
    assert pairs["--expected-candidate-job-audit-sha256"] == campaign.sha256(
        candidate_job_audit
    )
    assert pairs["--device"] == "cuda"
    assert pairs["--b-cache"].endswith("paired_validation_predictions.parquet")


def test_source_verification_rejects_drift_and_unregistered_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "models/helper.py"
    helper.parent.mkdir()
    helper.write_text("x = 1\n", encoding="utf-8")
    frozen = tmp_path / "contract.json"
    frozen.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "CONTRACT_PATH", frozen)
    monkeypatch.setattr(campaign, "CONTRACT_SHA256", campaign.sha256(frozen))
    revision = "a" * 40

    def git(command, **kwargs):
        if "rev-parse" in command:
            return revision + "\n"
        return b"models/helper.py\0contract.json\0"

    monkeypatch.setattr(campaign.subprocess, "check_output", git)
    manifest = {
        "schema_version": 1,
        "host_role": "5090",
        "status": "frozen_before_gpu_outputs",
        "source_revision": revision,
        "source_root": str(tmp_path.resolve()),
        "execution_contract_sha256": campaign.sha256(frozen),
        "source_files": {
            "models/helper.py": campaign.sha256(helper),
            "contract.json": campaign.sha256(frozen),
        },
    }
    assert campaign.verify_source(manifest)["verified_files"] == 2
    helper.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source file drift"):
        campaign.verify_source(manifest)
    helper.write_text("x = 1\n", encoding="utf-8")
    (helper.parent / "unregistered.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unregistered Python"):
        campaign.verify_source(manifest)
