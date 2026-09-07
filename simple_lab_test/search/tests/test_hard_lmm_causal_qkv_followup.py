from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

from paper.scripts import run_hard_lmm_causal_qkv_followup as followup
from paper.scripts import run_count_aware_tpp_backbone_control as trainer


NEW_REVISION = "1" * 40


@pytest.mark.parametrize(
    "host_role,dataset",
    [("5080", "yellow_trip_hourly"), ("5090", "intermittent_frozen_5000")],
)
def test_real_frozen_inputs_bind_405_sources_old_failure_and_assigned_e1(host_role, dataset):
    contract = followup.read_json(followup.CONTRACT_PATH)
    result = followup.verify_inputs(
        contract, host_role=host_role, source_revision=NEW_REVISION,
    )
    assert result["dataset"] == dataset
    assert result["pinned_source"]["verified_file_count"] == 405
    assert result["preserved_failure"]["decision"] == (
        "not_adopted_seed42_raw_rmse_gate_failed"
    )
    assert result["reused_e1"]["strict_audit_status"] == "passed"
    assert result["reused_5090_cost_profile"]["scope"] == (
        "architecture_only_not_host_speed_evidence"
    )


def test_followup_refuses_to_relabel_original_source_as_new_or_switch_host_dataset():
    contract = followup.read_json(followup.CONTRACT_PATH)
    original = contract["original_contract"]["implementation_source_revision"]
    with pytest.raises(ValueError, match="new source revision"):
        followup.verify_inputs(contract, host_role="5080", source_revision=original)
    changed = json.loads(json.dumps(contract))
    changed["host_assignments"]["5080"]["dataset"] = "intermittent_frozen_5000"
    with pytest.raises(ValueError, match="assignment drift"):
        followup.verify_inputs(changed, host_role="5080", source_revision=NEW_REVISION)


def test_dataset_symlink_may_leave_source_root_but_strict_source_may_not(monkeypatch, tmp_path):
    external = Path(__file__).resolve()
    (tmp_path / "sample_link").symlink_to(external)
    monkeypatch.setattr(followup, "ROOT", tmp_path)
    assert followup._dataset_file("sample_link").resolve() == external
    with pytest.raises(ValueError, match="leaves source root"):
        followup._root_file("sample_link")
    with pytest.raises(ValueError, match="Unsafe dataset path"):
        followup._dataset_file("../outside")


@pytest.mark.parametrize(
    "host_role,dataset",
    [("5080", "yellow_trip_hourly"), ("5090", "intermittent_frozen_5000")],
)
def test_generated_training_command_is_exact_frozen_seed42_policy(
    monkeypatch, tmp_path, host_role, dataset,
):
    contract = followup.read_json(followup.CONTRACT_PATH)
    command = followup.job_command(
        python=sys.executable,
        candidate=contract["candidate"],
        dataset=dataset,
        output=tmp_path,
        source_revision=NEW_REVISION,
        phase=contract["training_phase"],
        host_role=host_role,
    )
    monkeypatch.setattr(sys, "argv", command[2:])
    args = trainer.parse_args()
    assert args.device == "cuda"
    assert args.epochs == 300
    assert args.min_epochs == 40
    assert args.early_stopping_patience == 40
    assert args.seeds == "42"
    assert args.backbones == "titantpp_hard_memory_causal_qkv"
    assert args.checkpoint_monitor == "validation_raw_quantity_rmse"
    assert args.max_train_batches is args.max_val_batches is args.max_series is None
    assert host_role in args.execution_role and dataset in args.execution_role


def _write_junit(path: Path, names: list[str], *, skipped: str | None = None) -> None:
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite")
    for name in names:
        case = ET.SubElement(suite, "testcase", name=name)
        if name == skipped:
            ET.SubElement(case, "skipped")
    ET.ElementTree(suites).write(path, encoding="utf-8", xml_declaration=True)


def test_cuda_junit_requires_each_named_actual_cuda_test_once_and_no_skips(tmp_path):
    contract = followup.read_json(followup.CONTRACT_PATH)
    required = contract["qualification"]["required_actual_cuda_tests"]
    path = tmp_path / "cuda.xml"
    _write_junit(path, [*required, "ordinary_cpu_test"])
    result = followup.validate_cuda_junit(path, required)
    assert result["test_count"] == 4 and result["skipped"] == 0

    _write_junit(path, required[:-1])
    with pytest.raises(ValueError, match="missing or duplicated"):
        followup.validate_cuda_junit(path, required)
    _write_junit(path, required, skipped=required[0])
    with pytest.raises(ValueError, match="skipped"):
        followup.validate_cuda_junit(path, required)


def _worker_payload() -> dict:
    return {
        "status": "passed",
        "backbone": "titantpp_hard_memory_causal_qkv",
        "sequence_length": 256,
        "repeat": 0,
        "observed_events": 255,
        "target_events": 1,
        "finite_model_optimizer": True,
        "learning_path": {"status": "passed", "state_digest_restored": True},
        "device_name": "NVIDIA GeForce RTX 5080",
        "median_step_seconds": 0.01,
        "peak_allocated_bytes": 100,
    }


def test_synthetic_worker_proves_device_finite_state_and_learning_path():
    result = followup.validate_synthetic_worker(_worker_payload(), expected_gpu="RTX 5080")
    assert result["status"] == "passed"
    bad = _worker_payload()
    bad["device_name"] = "NVIDIA GeForce RTX 5090"
    with pytest.raises(ValueError, match="wrong GPU"):
        followup.validate_synthetic_worker(bad, expected_gpu="RTX 5080")
    bad = _worker_payload()
    bad["learning_path"]["status"] = "failed"
    with pytest.raises(ValueError, match="learning path"):
        followup.validate_synthetic_worker(bad, expected_gpu="RTX 5080")


def test_replication_review_uses_inclusive_frozen_boundaries_and_new_dataset_requirement():
    contract = followup.read_json(followup.CONTRACT_PATH)
    gate = contract["replication_review_gate"]
    baseline = {
        dataset: {
            "raw_rmse": 1.0, "overall_mae": 1.0, "body_mae": 1.0,
            "gt_p99_mae": 1.0, "clamped_time_loss": 0.0,
        }
        for dataset in gate["datasets"]
    }
    candidate = {
        dataset: {
            "raw_rmse": 1.01, "overall_mae": 1.01, "body_mae": 1.02,
            "gt_p99_mae": 1.02, "clamped_time_loss": 0.01,
        }
        for dataset in gate["datasets"]
    }
    candidate["yellow_trip_hourly"]["raw_rmse"] = 0.99
    passed = followup.evaluate_replication_review(candidate, baseline, gate)
    assert passed["status"] == "eligible_for_replication_review_only"

    candidate["yellow_trip_hourly"]["raw_rmse"] = 0.9900000001
    failed = followup.evaluate_replication_review(candidate, baseline, gate)
    assert failed["status"] == "stop_candidate_after_exploratory_followup"


def test_reporting_gate_failure_does_not_relabel_successful_execution(
    monkeypatch, tmp_path,
):
    from paper.scripts import audit_hard_lmm_causal_qkv as audit_module

    monkeypatch.setattr(
        followup, "verify_inputs", lambda *args, **kwargs: {"dataset": "yellow_trip_hourly"},
    )
    monkeypatch.setattr(followup, "gpu_preflight", lambda _: {"gpu_name": "RTX 5080"})
    monkeypatch.setattr(followup, "runtime_probe", lambda *args: {"device_name": "RTX 5080"})
    monkeypatch.setattr(followup, "run_cuda_qualification", lambda **kwargs: {"status": "passed"})
    monkeypatch.setattr(followup, "run_synthetic_worker", lambda **kwargs: {"status": "passed"})
    monkeypatch.setattr(followup.subprocess, "run", lambda *args, **kwargs: None)
    baseline = followup.read_json(followup.CONTRACT_PATH)["B_validation_reference"]["datasets"][
        "yellow_trip_hourly"
    ]
    metrics = dict(baseline)
    metrics["raw_rmse"] += 1.0
    monkeypatch.setattr(
        audit_module, "audit_causal_qkv_job",
        lambda *args, **kwargs: {"status": "passed", "metrics": metrics},
    )
    argv = [
        "--host-role", "5080", "--source-revision", NEW_REVISION,
        "--output-root", str(tmp_path), "--python", sys.executable,
    ]
    followup.main(argv)
    status = followup.read_json(tmp_path / "followup_status.json")
    assert status["status"] == "completed_execution_and_audit"
    assert status["execution_completed"] is True
    assert status["dataset_reporting_gate_status"] == "failed"
    assert status["replication_review_status"] == "pending_parallel_result_merge"
    with pytest.raises(ValueError, match="occupied"):
        followup.main(argv)


def test_failure_is_saved_atomically_without_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(followup, "verify_inputs", lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("deliberate preflight failure")
    ))
    argv = [
        "--host-role", "5090", "--source-revision", NEW_REVISION,
        "--output-root", str(tmp_path), "--python", sys.executable,
    ]
    with pytest.raises(ValueError, match="deliberate"):
        followup.main(argv)
    status = followup.read_json(tmp_path / "followup_status.json")
    assert status["status"] == "failed_execution_or_audit"
    assert status["execution_completed"] is False
    assert "deliberate preflight failure" in status["error"]
    assert not (tmp_path / "followup_status.json.tmp").exists()
