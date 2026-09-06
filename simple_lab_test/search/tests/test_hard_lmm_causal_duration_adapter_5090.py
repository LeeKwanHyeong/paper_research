"""Fail-closed orchestration tests for the causal duration-adapter screen."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts import run_hard_lmm_causal_duration_adapter_5090 as controller


CONTRACT_PATH = ROOT / controller.CONTRACT_REL


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_controller_required_cuda_adapter_contract_executes_on_cuda() -> None:
    """Become a mandatory GPU sentinel when launched by the 5090 controller."""
    require_cuda = os.environ.get(
        "HARD_LMM_CAUSAL_DURATION_ADAPTER_REQUIRE_CUDA"
    ) == "1"
    if require_cuda:
        assert torch.cuda.is_available(), "Controller-required CUDA is unavailable"
        device = torch.device("cuda")
        assert "RTX 5090" in torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")

    from models.TPPs.CausalLogDurationAdapter import CausalLogDurationAdapter

    adapter = CausalLogDurationAdapter(
        log_duration_mean=0.5,
        log_duration_std=0.25,
        sigma_floor=0.001,
    ).to(device)
    history = torch.tensor([[1.0, 2.0], [3.0, 0.0]], device=device)
    lengths = torch.tensor([2, 1], device=device, dtype=torch.long)
    base_scale = torch.tensor([0.4, 0.8], device=device, dtype=torch.float64)
    scale = adapter.adjusted_scale(base_scale, history, lengths)
    assert scale.device == device
    assert torch.equal(scale, base_scale)
    scale.sum().backward()
    assert adapter.scale_projection.weight.grad is not None
    assert torch.isfinite(adapter.scale_projection.weight.grad).all()


def test_checked_in_contract_freezes_one_seed_scale_only_taxi_first_policy() -> None:
    datasets = controller.validate_contract(load_contract())
    assert tuple(datasets) == controller.CONTRACT_DATASETS
    assert datasets[controller.TAXI_DATASET]["right_censor_threshold"] is None
    assert load_contract()["scope"]["held_out_test"] is False
    assert load_contract()["scope"]["additional_seeds"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["scope"].__setitem__("held_out_test", True),
        lambda value: value["scope"].__setitem__("additional_seeds", True),
        lambda value: value["adapter"].__setitem__("hidden_size", 16),
        lambda value: value["adapter"].__setitem__("location_formula", "learned"),
        lambda value: value["optimization"].__setitem__("seed", 52),
        lambda value: value["checkpoint_selection"].__setitem__(
            "interval_nll_in_selector", True
        ),
        lambda value: value["execution_policy"].__setitem__(
            "stop_after_taxi_failure", False
        ),
        lambda value: value["acceptance"]["structured_gates"].__setitem__(
            "maximum_candidate_minus_A_continuous_nll", 0.1
        ),
    ),
)
def test_contract_drift_fails_closed(mutation) -> None:
    changed = copy.deepcopy(load_contract())
    mutation(changed)
    with pytest.raises(ValueError):
        controller.validate_contract(changed)


def test_commands_use_full_data_taxi_e1_then_unlimited_seed42(tmp_path: Path) -> None:
    row = {
        "dataset": controller.TAXI_DATASET,
        "data_path": "data.parquet",
        "split_manifest_path": "split.json",
    }
    common = {
        "python": "python",
        "project": tmp_path,
        "frozen_b_root": tmp_path / "B",
        "output": tmp_path / "out",
        "revision": "a" * 40,
        "row": row,
    }
    e1 = controller.runner_command(**common, phase="e1")
    assert e1[e1.index("--max-epochs") + 1] == "1"
    assert "--allow-partial-contract" in e1
    assert "--max-train-batches" not in e1
    assert "--max-validation-batches" not in e1
    assert "--run-epoch-limit" not in e1
    assert "--test" not in e1
    assert "--held-out" not in e1
    assert "--held_out" not in e1

    full = controller.runner_command(**common, phase="seed42")
    for flag in (
        "--max-train-batches",
        "--max-validation-batches",
        "--max-epochs",
        "--allow-partial-contract",
        "--run-epoch-limit",
    ):
        assert flag not in full
    assert full[full.index("--checkpoint") + 1].endswith(
        "full/yellow_trip_hourly/best_validation_proper_time_nll_model.pt"
    )
    assert full[full.index("--feature-cache-dir") + 1].endswith(
        "e1/yellow_trip_hourly/cache"
    )


def _screen_summary(
    *,
    candidate: float = -0.45,
    control: float = -0.44,
    base: float = 0.266,
    baseline_a: float = -0.443,
    candidate_interval: float = 0.89,
    base_interval: float = 0.90,
) -> dict:
    return {
        "candidate": {
            "selected_validation_metrics": {
                "continuous_proper_time_nll": candidate,
                "interval_mass_time_nll": candidate_interval,
            }
        },
        "global_scale_control": {
            "selected_validation_metrics": {
                "continuous_proper_time_nll": control,
                "interval_mass_time_nll": base_interval,
            }
        },
        "base_B_validation_metrics": {
            "continuous_proper_time_nll": base,
            "interval_mass_time_nll": base_interval,
        },
        "quantity_prediction_bitwise_identical": True,
        "time_median_bitwise_identical": True,
        "base_location_bitwise_identical": True,
        "source_model_state_unchanged": True,
        "A_validation_continuous_nll": baseline_a,
    }


def test_acceptance_decision_recomputes_every_fixed_gate() -> None:
    row = {"dataset": controller.TAXI_DATASET, "A_validation_continuous_nll": -0.443}
    passed = controller.acceptance_decision(_screen_summary(), row)
    assert passed["status"] == "passed"
    assert all(passed["gates"].values())

    attribution_failure = controller.acceptance_decision(
        _screen_summary(candidate=-0.442, control=-0.44), row
    )
    assert attribution_failure["status"] == "failed"
    assert not attribution_failure["gates"][
        "candidate_beats_global_control_by_0_005"
    ]

    location_failure = _screen_summary()
    location_failure["base_location_bitwise_identical"] = False
    rejected = controller.acceptance_decision(location_failure, row)
    assert rejected["status"] == "failed"
    assert not rejected["gates"]["base_location_bitwise_identical"]


def test_gpu_preflight_requires_one_idle_5090_and_inactive_gdm(monkeypatch) -> None:
    replies = iter(("NVIDIA GeForce RTX 5090, 30000", "", "inactive"))
    monkeypatch.setattr(
        controller, "command_output", lambda *args, **kwargs: next(replies)
    )
    result = controller.gpu_preflight()
    assert result["free_vram_mib"] == 30000
    assert result["compute_processes"] == []

    replies = iter(("NVIDIA GeForce RTX 5090, 30000", "1234", "inactive"))
    monkeypatch.setattr(
        controller, "command_output", lambda *args, **kwargs: next(replies)
    )
    with pytest.raises(ValueError, match="compute process"):
        controller.gpu_preflight()


def test_cuda_suite_environment_and_junit_require_the_gpu_sentinel(
    tmp_path: Path,
) -> None:
    env = controller.deterministic_env(tmp_path)
    assert env["HARD_LMM_CAUSAL_DURATION_ADAPTER_REQUIRE_CUDA"] == "1"
    passed = tmp_path / "passed.xml"
    passed.write_text(
        "<testsuite><testcase name=\""
        + controller.CUDA_SENTINEL_TEST_NAME
        + "\"/></testsuite>\n",
        encoding="utf-8",
    )
    audit = controller.audit_pytest(passed)
    assert audit["cuda_sentinel_passed"] is True

    missing = tmp_path / "missing.xml"
    missing.write_text(
        "<testsuite><testcase name=\"ordinary_test\"/></testsuite>\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sentinel"):
        controller.audit_pytest(missing)


def test_taxi_screen_failure_stops_before_other_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "source"
    output = tmp_path / "output"
    project.mkdir()
    (project / controller.CONTRACT_REL).parent.mkdir(parents=True)
    (project / controller.CONTRACT_REL).write_text("{}\n", encoding="utf-8")
    (project / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    datasets = {
        name: {
            "dataset": name,
            "data_path": f"{name}.parquet",
            "split_manifest_path": f"{name}.json",
        }
        for name in controller.CONTRACT_DATASETS
    }
    monkeypatch.setattr(controller, "validate_contract", lambda value: datasets)
    monkeypatch.setattr(controller, "verify_source", lambda *args: {"files": {"x": "y"}})
    verified_inputs: list[str] = []

    def fake_verify_inputs(project, frozen, row):
        verified_inputs.append(row["dataset"])
        return {
            "dataset": row["dataset"],
            "data_path": "unused",
            "data_sha256": "0" * 64,
            "split_manifest_path": "unused",
            "split_manifest_sha256": "0" * 64,
            "checkpoint_path": "unused",
            "checkpoint_file_sha256": "0" * 64,
            "feature_caches": {},
        }

    monkeypatch.setattr(
        controller,
        "verify_dataset_inputs",
        fake_verify_inputs,
    )
    monkeypatch.setattr(controller, "verify_input_fingerprint", lambda record: None)
    monkeypatch.setattr(
        controller,
        "gpu_preflight",
        lambda: {
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "free_vram_mib": 30000,
            "compute_processes": [],
            "gdm": "inactive",
        },
    )
    monkeypatch.setattr(controller, "run_logged", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        controller,
        "audit_pytest",
        lambda path: {"test_count": 1, "failures": 0, "errors": 0, "skipped": 0},
    )
    observed: list[tuple[str, str]] = []

    def fake_audit(output, *, row, contract_path, source_revision, phase):
        observed.append((row["dataset"], phase))
        return {
            "acceptance_decision": (
                None
                if phase == "e1"
                else {"status": "failed", "dataset": row["dataset"], "gates": {}}
            )
        }

    monkeypatch.setattr(controller, "audit_run", fake_audit)
    args = argparse.Namespace(
        project_root=project,
        frozen_b_root=tmp_path / "B",
        output_root=output,
        source_revision="a" * 40,
        python=sys.executable,
        verify_only=False,
    )
    result = controller.execute(args)
    assert result["status"] == "stopped_after_taxi_screening_failure"
    assert observed == [
        (controller.TAXI_DATASET, "e1"),
        (controller.TAXI_DATASET, "seed42"),
    ]
    assert verified_inputs == [controller.TAXI_DATASET]
    assert result["additional_seeds_executed"] is False
    assert result["held_out_test_evaluated"] is False
    control = result["control_evidence"]
    assert Path(control["source_manifest_path"]).read_text(encoding="utf-8") == "{}\n"
    assert Path(control["contract_path"]).read_text(encoding="utf-8") == "{}\n"
    with pytest.raises(ValueError, match="already terminal"):
        controller.execute(args)


def test_interrupted_controller_resumes_only_with_identical_identity(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "source"
    output = tmp_path / "output"
    project.mkdir()
    (project / controller.CONTRACT_REL).parent.mkdir(parents=True)
    (project / controller.CONTRACT_REL).write_text("{}\n", encoding="utf-8")
    (project / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    datasets = {
        name: {
            "dataset": name,
            "data_path": f"{name}.parquet",
            "split_manifest_path": f"{name}.json",
        }
        for name in controller.CONTRACT_DATASETS
    }
    monkeypatch.setattr(controller, "validate_contract", lambda value: datasets)
    monkeypatch.setattr(controller, "verify_source", lambda *args: {"files": {"x": "y"}})
    input_version = {"value": 1}

    def fake_verify_inputs(project, frozen, row):
        return {
            "dataset": row["dataset"],
            "identity_version": input_version["value"],
        }

    monkeypatch.setattr(controller, "verify_dataset_inputs", fake_verify_inputs)
    monkeypatch.setattr(controller, "verify_input_fingerprint", lambda record: None)
    monkeypatch.setattr(
        controller,
        "gpu_preflight",
        lambda: {
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "free_vram_mib": 30000,
            "compute_processes": [],
            "gdm": "inactive",
        },
    )
    monkeypatch.setattr(
        controller,
        "audit_pytest",
        lambda path: {
            "test_count": 1,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "cuda_sentinel_test": controller.CUDA_SENTINEL_TEST_NAME,
            "cuda_sentinel_passed": True,
        },
    )
    commands: list[list[str]] = []

    def interrupt_first_e1(command, log_path, *args, **kwargs):
        commands.append(command)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if len(commands) == 2:
            raise InterruptedError("simulated interruption")

    monkeypatch.setattr(controller, "run_logged", interrupt_first_e1)

    def fake_audit(output, *, row, contract_path, source_revision, phase):
        return {
            "acceptance_decision": (
                None
                if phase == "e1"
                else {"status": "failed", "dataset": row["dataset"], "gates": {}}
            )
        }

    monkeypatch.setattr(controller, "audit_run", fake_audit)
    args = argparse.Namespace(
        project_root=project,
        frozen_b_root=tmp_path / "B",
        output_root=output,
        source_revision="a" * 40,
        python=sys.executable,
        verify_only=False,
    )
    with pytest.raises(InterruptedError, match="simulated interruption"):
        controller.execute(args)
    interrupted = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert interrupted["status"] == "failed"
    assert interrupted["resume_count"] == 0

    input_version["value"] = 2
    with pytest.raises(ValueError, match="resume identity drift"):
        controller.execute(args)
    assert len(commands) == 2

    input_version["value"] = 1
    copied_contract = output / "control" / controller.CONTRACT_REL.name
    copied_contract.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Copied contract changed"):
        controller.execute(args)
    assert len(commands) == 2
    copied_contract.write_text("{}\n", encoding="utf-8")

    result = controller.execute(args)
    assert result["status"] == "stopped_after_taxi_screening_failure"
    assert result["resume_count"] == 1
    pytest_commands = [command for command in commands if "pytest" in command]
    assert len(pytest_commands) == 1
    assert len(result["completed_e1"]) == 1
    assert len(result["completed_seed42"]) == 1
