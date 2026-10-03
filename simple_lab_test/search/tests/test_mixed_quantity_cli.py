"""Local metadata, authorization, pairing, and supervisor boundary checks."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest

from paper.scripts import run_mixed_quantity_comparison as cli
from paper.scripts.mixed_quantity_acceptance import ACCEPTANCE_POLICY
from paper.scripts.mixed_quantity_objective import CALIBRATION_POLICY


@pytest.fixture
def contract():
    parent = cli.read_json(cli.ROOT / "paper/contracts/quantity_comparison_execution_5090_v3.json")
    runtime = parent["runtime_expected"]
    files = cli.source_hashes()
    return {
        "schema": cli.SCHEMA, "status": "frozen_pending_explicit_approval",
        "cases": list(cli.CASES), "seed": 42, "epochs": 120,
        "datasets": deepcopy(parent["datasets"]),
        "basis": {"contract": parent, "contract_sha256": cli.sha_json(parent)},
        "source": {"files": files, "files_sha256": cli.sha_json(files)},
        "runtime_evidence": {"runtime": deepcopy(runtime)},
        "runtime_expected": deepcopy(runtime), "execution": deepcopy(cli.EXECUTION),
        "calibration_policy": deepcopy(CALIBRATION_POLICY),
        "acceptance_policy": deepcopy(ACCEPTANCE_POLICY),
        "limits": deepcopy(cli.LIMITS), "policy": deepcopy(cli.POLICY),
        "total_optimizer_steps": 6816240, "validation_replays": 18,
        "approval": {"approved": False, "required_scope": cli.SCOPE},
    }


def approval(contract):
    return {"status": "approved_by_user", "execution_contract_sha256": cli.sha_json(contract),
            "scope": cli.SCOPE, "user_instruction": "Synthetic authorization test only; no execution."}


def test_current_v3_basis_and_5090_metadata_validate_without_execution(contract):
    result = cli.validate_contract(contract)
    assert result == {"status": "validated_metadata_only", "contract_sha256": cli.sha_json(contract),
                      "arms": 9, "optimizer_steps": 6816240, "gpu_started": False}
    assert contract["approval"]["approved"] is False
    assert contract["execution"]["server_alias"] == "5090"
    assert contract["runtime_expected"]["gpu"]["uuid"] == "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"


@pytest.mark.parametrize("change", [
    "cases", "limits", "calibration", "acceptance", "dataset_model", "steps", "policy", "runtime",
])
def test_contract_rejects_changes_to_fixed_experimental_scope(contract, change):
    if change == "cases":
        contract["cases"] = contract["cases"][:-1]
    elif change == "limits":
        contract["limits"]["max_wall_seconds"] += 1
    elif change == "calibration":
        contract["calibration_policy"]["undeclared_adaptive_weight"] = True
    elif change == "acceptance":
        contract["acceptance_policy"]["primary"]["intermittent_rmse_max_ratio"] = 1.1
    elif change == "dataset_model":
        contract["datasets"][0]["model"]["hidden_dim"] *= 2
    elif change == "steps":
        contract["total_optimizer_steps"] += 1
    elif change == "policy":
        contract["policy"]["automatic_resume"] = True
    else:
        contract["runtime_expected"]["gpu"]["uuid"] = "another-gpu"
    with pytest.raises(ValueError):
        cli.validate_contract(contract, check_source=False)


def test_contract_rejects_rehashed_parent_and_matching_dataset_model_tamper(contract):
    contract["basis"]["contract"]["datasets"][0]["model"]["hidden_dim"] *= 2
    contract["datasets"] = deepcopy(contract["basis"]["contract"]["datasets"])
    contract["basis"]["contract_sha256"] = cli.sha_json(contract["basis"]["contract"])
    with pytest.raises(ValueError, match="Parent|parent|basis|Basis"):
        cli.validate_contract(contract, check_source=False)


def test_source_closure_contains_new_execution_and_objective_code_and_detects_changed_content(contract):
    required = {
        "paper/scripts/mixed_quantity_objective.py",
        "paper/scripts/mixed_quantity_acceptance.py",
        "paper/scripts/run_mixed_quantity_comparison.py",
        "paper/scripts/quantity_comparison_engine.py",
    }
    assert required <= set(contract["source"]["files"])
    for name in required:
        assert contract["source"]["files"][name] == cli.sha_file(cli.ROOT / name)
    contract["source"]["files"]["paper/scripts/mixed_quantity_objective.py"] = "0" * 64
    contract["source"]["files_sha256"] = cli.sha_json(contract["source"]["files"])
    with pytest.raises(ValueError, match="Source closure or contents changed"):
        cli.validate_contract(contract)


def test_previous_training_approval_cannot_authorize_new_exact_contract(contract):
    old = {"status": "approved_by_user",
           "execution_contract_sha256": contract["basis"]["contract_sha256"],
           "scope": "synthetic_cuda_qualification_and_12_fresh_joint_arms",
           "user_instruction": "Previously approved v3 training."}
    with pytest.raises(ValueError, match="bound to this contract"):
        cli.validate_approval(contract, old)
    old["execution_contract_sha256"] = cli.sha_json(contract)
    with pytest.raises(ValueError, match="scope"):
        cli.validate_approval(contract, old)
    cli.validate_approval(contract, approval(contract))


def test_old_5080_contract_and_rehashed_approval_cannot_authorize_5090(contract):
    old_contract = cli.read_json(cli.ROOT / "paper/contracts/mixed_quantity_execution_5080_v1.json")
    assert old_contract["execution"]["server_alias"] == "5080"
    with pytest.raises(ValueError, match="Pinned 5090 execution"):
        cli.validate_contract(old_contract, check_source=False)
    old_approval = {
        "status": "approved_by_user", "execution_contract_sha256": cli.sha_json(old_contract),
        "scope": old_contract["approval"]["required_scope"],
        "user_instruction": "Synthetic prior 5080 approval; no real execution."}
    with pytest.raises(ValueError, match="bound to this contract"):
        cli.validate_approval(contract, old_approval)
    old_approval["execution_contract_sha256"] = cli.sha_json(contract)
    with pytest.raises(ValueError, match="scope"):
        cli.validate_approval(contract, old_approval)


def test_5090_execution_contract_rejects_self_consistent_5080_runtime_evidence(contract):
    old_runtime = cli.read_json(cli.ROOT / "search_artifacts/quantity_partition_preparation_v1/5080_readiness.json")["runtime"]
    contract["runtime_expected"] = deepcopy(old_runtime)
    contract["runtime_evidence"]["runtime"] = deepcopy(old_runtime)
    with pytest.raises(ValueError, match="5090"):
        cli.validate_contract(contract, check_source=False)


def test_execute_rejects_unapproved_contract_before_any_subprocess(contract, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Subprocess or runtime reached before authorization rejection")
    monkeypatch.setattr(cli.subprocess, "check_output", forbidden)
    monkeypatch.setattr(cli.subprocess, "Popen", forbidden)
    monkeypatch.setattr(cli, "configured_runtime", forbidden)
    with pytest.raises(ValueError, match="Explicit approval"):
        cli.execute(contract, tmp_path / "contract.json", {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["_suite", "_probe", "_shapes"])
def test_internal_gpu_cli_requires_valid_launch_receipt_before_dispatch(contract, tmp_path, monkeypatch, command):
    # Simulate only host identity so the actual launch-receipt gate is reached.
    # Every GPU dispatch function remains a failing sentinel.
    execution = {**cli.EXECUTION, "source_dir": str(cli.ROOT),
                 "python": str(Path(sys.executable).resolve()), "output_dir": str(tmp_path / "run")}
    monkeypatch.setattr(cli, "EXECUTION", execution)
    contract["execution"] = deepcopy(execution)
    def forbidden(*args, **kwargs):
        pytest.fail("GPU dispatch reached without a valid launch receipt")
    for name in ("run_suite", "probe_worker", "shape_checks", "configured_runtime"):
        monkeypatch.setattr(cli, name, forbidden)
    contract_path = tmp_path / "contract.json"
    cli.write_json(contract_path, contract)
    monkeypatch.setattr(sys, "argv", ["mixed-cli", command, "--contract", str(contract_path),
                        "--started", "100", "--case", cli.CASES[0], "--stage", "full"])
    with pytest.raises(FileNotFoundError):
        cli.main()
    cli.write_json(tmp_path / "run/execution_receipt.json", {
        "approval": {}, "contract_sha256": cli.sha_json(contract),
        "started_at_unix": 100., "deadline_unix": 100. + cli.LIMITS["max_wall_seconds"]})
    with pytest.raises(ValueError, match="Explicit approval"):
        cli.main()
    assert not (tmp_path / "run/qualification").exists()


def test_budget_enforces_per_file_limit_on_forced_commit_scan_at_same_clock(contract, tmp_path):
    contract["execution"]["output_dir"] = str(tmp_path)
    artifact = tmp_path / "checkpoint.pt"
    with artifact.open("wb") as handle:
        handle.truncate(contract["limits"]["per_file_bytes"])
    budget = cli.Budget(contract, 100., clock=lambda: 100., monotonic=lambda: 10.)
    budget()
    with artifact.open("r+b") as handle:
        handle.truncate(contract["limits"]["per_file_bytes"] + 1)
    with pytest.raises(ValueError, match="Individual artifact"):
        budget("before_epoch_commit")


def test_calibration_deadline_stays_monotonic_after_wall_clock_rollback_and_resets_dataset():
    ticks = {"wall": 1000., "mono": 20.}
    deadline = cli.CalibrationDeadline(clock=lambda: ticks["wall"], monotonic=lambda: ticks["mono"])
    first = {"status": "calibrating", "dataset_id": "intermittent_frozen_5000", "phase_started_at_unix": 1000.}
    assert deadline.remaining(first) == 600.
    ticks.update(wall=200., mono=619.)
    assert deadline.remaining(first) == 1.
    ticks["mono"] = 620.
    assert deadline.remaining(first) == 0.
    ticks["mono"] = 621.
    assert deadline.remaining(first) < 0.
    assert math.isinf(deadline.remaining({"status": "running"}))
    second = {"status": "calibrating", "dataset_id": "yellow_trip_hourly", "phase_started_at_unix": 200.}
    assert deadline.remaining(second) == 600.
    ticks["mono"] += 601.
    assert deadline.remaining(second) < 0.


def test_calibration_deadline_accounts_for_time_before_first_supervisor_observation():
    deadline = cli.CalibrationDeadline(clock=lambda: 1100., monotonic=lambda: 20.)
    assert deadline.remaining({"status": "calibrating", "dataset_id": "insta_market_basket",
                               "phase_started_at_unix": 1000.}) == 500.


def paired_summaries():
    history = [{"epoch": epoch, "global_step": epoch * 2, "train_count": 4, "train_batches": 2,
                "validation_count": 4, "validation_batches": 2,
                "train_batch_order_sha256": f"identical-batches-{epoch}"} for epoch in (1, 2)]
    return [{"condition": {"mixed_objective": {"name": case}}, "status": "complete",
             "initial_state_sha256": "same-initial-state", "global_step": 4,
             "history": deepcopy(history)} for case in cli.CASES]


def test_pair_audit_rejects_missing_arm_and_different_training_batch_order():
    summaries = paired_summaries()
    assert cli.audit_pairs(summaries) == {"status": "passed", "arms": 3, "epochs": 2, "global_steps_per_case": 4}
    with pytest.raises(ValueError, match="Three completed cases"):
        cli.audit_pairs(summaries[:2])
    summaries[2]["history"][1]["train_batch_order_sha256"] = "changed-training-samples"
    with pytest.raises(ValueError, match="train_batch_order_sha256"):
        cli.audit_pairs(summaries)


def test_validation_cli_outputs_only_metadata(contract, tmp_path, monkeypatch, capsys):
    path = tmp_path / "contract.json"
    cli.write_json(path, contract)
    monkeypatch.setattr(sys, "argv", ["mixed-cli", "validate", "--contract", str(path)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["gpu_started"] is False and result["arms"] == 9
    assert sorted(p.name for p in tmp_path.iterdir()) == ["contract.json"]
