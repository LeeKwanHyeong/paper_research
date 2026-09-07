"""Contract tests for the 5080-only frozen raw-affine controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.scripts import control_frozen_raw_affine_5080 as controller


def dataset_rows() -> dict[str, dict]:
    return {
        dataset: {
            "dataset": dataset,
            "data_path": f"data/{dataset}.parquet",
            "split_manifest_path": f"data/{dataset}_split.json",
        }
        for dataset in controller.B_DATASETS
    }


def arguments(tmp_path: Path) -> argparse.Namespace:
    checkpoints = {(dataset, "B"): tmp_path / f"{dataset}_B.pt" for dataset in controller.B_DATASETS}
    checkpoints.update({
        ("insta_market_basket", "rmtpp"): tmp_path / "insta_rmtpp.pt",
        ("insta_market_basket", "thp"): tmp_path / "insta_thp.pt",
    })
    return SimpleNamespace(
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "snapshot",
        source_revision="a" * 40,
        source_manifest=tmp_path / "source_manifest.json",
        checkpoints=checkpoints,
    )


def test_checkpoint_mapping_parser_is_strict():
    assert controller.parse_checkpoint("insta_market_basket:thp=/tmp/model.pt") == (
        ("insta_market_basket", "thp"),
        Path("/tmp/model.pt"),
    )
    with pytest.raises(argparse.ArgumentTypeError, match="invalid"):
        controller.parse_checkpoint("insta_market_basket:other=/tmp/model.pt")
    with pytest.raises(argparse.ArgumentTypeError, match="must be"):
        controller.parse_checkpoint("bad")


def test_runner_command_pins_cuda_zero_and_separate_phase(tmp_path):
    args = arguments(tmp_path)
    row = dataset_rows()["yellow_trip_hourly"]
    command = controller.runner_command(
        args,
        row,
        role="B",
        phase="validation",
        output=tmp_path / "output",
        train_result=tmp_path / "train.json",
        prerequisite=tmp_path / "gate.json",
    )
    assert command[0] == controller.sys.executable
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--phase") + 1] == "validation"
    assert "--smoke" not in command
    assert command[command.index("--train-result") + 1] == str(tmp_path / "train.json")
    assert command[command.index("--prerequisite-decision") + 1] == str(tmp_path / "gate.json")


def test_failed_B_train_gate_stops_before_any_validation(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    args.artifact_root.mkdir()
    commands: list[list[str]] = []

    def fake_run_logged(command: list[str], *, log_path: Path) -> None:
        commands.append(command)
        if "--stage" in command:
            assert command[command.index("--stage") + 1] == "b-train"
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"accepted": False}), encoding="utf-8")

    monkeypatch.setattr(controller, "run_logged", fake_run_logged)
    result = controller.run_full(args, dataset_rows())

    assert result["decision"] == "rejected_stop_before_validation"
    runner_commands = [command for command in commands if "--dataset" in command]
    assert len(runner_commands) == 3
    assert {command[command.index("--phase") + 1] for command in runner_commands} == {"train"}


def test_pre_gate_cuda_smoke_reads_train_only_for_all_three_B_datasets(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    args.artifact_root.mkdir()
    commands: list[list[str]] = []

    def fake_run_logged(command: list[str], *, log_path: Path) -> None:
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(controller, "run_logged", fake_run_logged)
    result = controller.run_train_smoke(args, dataset_rows())

    assert result["decision"] == "train_smoke_passed_for_full_train_gate"
    assert len(commands) == 3
    assert {command[command.index("--phase") + 1] for command in commands} == {"train"}
    assert all("--smoke" in command for command in commands)
    decision = json.loads(Path(result["artifact"]).read_text())
    assert decision["validation_loaded"] is False
    assert len(decision["result_identities"]) == 3


def test_validation_cuda_smoke_requires_the_accepted_B_train_gate(monkeypatch, tmp_path):
    args = arguments(tmp_path)
    args.artifact_root.mkdir()
    prerequisite = tmp_path / "B_train.json"
    prerequisite.write_text("{}", encoding="utf-8")
    train_results = []
    for dataset in controller.B_DATASETS:
        path = tmp_path / f"{dataset}_train.json"
        path.write_text("{}", encoding="utf-8")
        train_results.append(path)
    commands: list[list[str]] = []

    def fake_run_logged(command: list[str], *, log_path: Path) -> None:
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(controller, "run_logged", fake_run_logged)
    result = controller.run_validation_smoke(
        args,
        dataset_rows(),
        train_results=train_results,
        prerequisite=prerequisite,
    )

    assert result["decision"] == "validation_smoke_passed_for_full_validation"
    assert len(commands) == 3
    assert all(command[command.index("--phase") + 1] == "validation" for command in commands)
    assert all(
        command[command.index("--prerequisite-decision") + 1] == str(prerequisite)
        for command in commands
    )


def test_run_logged_injects_the_contract_environment(monkeypatch, tmp_path):
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    controller.run_logged(["true"], log_path=tmp_path / "command.log")
    expected = controller.load_contract()["runtime"]["deterministic_environment"]
    assert {key: observed[key] for key in expected} == expected


def test_full_mode_always_runs_cuda_smoke_first(monkeypatch, tmp_path, capsys):
    args = arguments(tmp_path)
    args.mode = "full"
    order = []
    monkeypatch.setattr(controller, "parse_args", lambda: args)
    monkeypatch.setattr(controller, "load_contract", lambda: {"datasets": list(dataset_rows().values())})

    def fake_smoke(_args, _rows):
        order.append("smoke")
        return {"decision": "smoke_passed_for_full_train_gate", "artifact": "smoke.json"}

    def fake_full(_args, _rows):
        order.append("full")
        return {"decision": "rejected_stop_before_validation", "artifact": "B_train.json"}

    monkeypatch.setattr(controller, "run_train_smoke", fake_smoke)
    monkeypatch.setattr(controller, "run_full", fake_full)
    controller.main()

    assert order == ["smoke", "full"]
    status = json.loads((args.artifact_root / "pipeline_status.json").read_text())
    assert status["cuda_train_smoke"]["decision"] == "smoke_passed_for_full_train_gate"
    capsys.readouterr()


def test_parse_args_rejects_duplicate_checkpoint_mapping(monkeypatch, tmp_path):
    argv = [
        "controller",
        "--mode",
        "smoke",
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--data-root",
        str(tmp_path / "data"),
        "--source-revision",
        "a" * 40,
        "--source-manifest",
        str(tmp_path / "manifest.json"),
    ]
    for dataset in controller.B_DATASETS:
        argv.extend(["--checkpoint", f"{dataset}:B={tmp_path}/{dataset}.pt"])
    argv.extend([
        "--checkpoint",
        f"{controller.B_DATASETS[0]}:B={tmp_path}/duplicate.pt",
    ])
    monkeypatch.setattr(controller.sys, "argv", argv)
    with pytest.raises(SystemExit):
        controller.parse_args()
