"""Frozen command and fail-closed boundaries for the 5090 controller."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from paper.scripts import run_hard_lmm_quantile_checkpoint_alignment_5090 as runner


def contract() -> dict:
    return json.loads((runner.CONTRACT_REL).read_text(encoding="utf-8"))


def test_contract_and_commands_freeze_only_seed42_e1_then_e300(tmp_path) -> None:
    assert str(runner.PROJECT_ROOT) in sys.path
    frozen = contract()
    datasets = runner.validate_contract(frozen)
    assert tuple(datasets) == runner.DATASETS
    for phase, expected in (("e1", (1, 1, 1)), ("seed42_e300", (300, 40, 40))):
        for row in datasets.values():
            command = runner.training_command(
                python="python",
                project=tmp_path,
                output=tmp_path / "fresh",
                revision="a" * 40,
                row=row,
                phase=phase,
            )
            values = {
                flag: command[command.index(flag) + 1]
                for flag in (
                    "--epochs",
                    "--min-epochs",
                    "--early-stopping-patience",
                    "--seeds",
                    "--checkpoint-monitor",
                    "--quantile-adaptive-strength",
                    "--time-intercept-limit",
                    "--quantity-variants",
                )
            }
            assert tuple(map(int, (values["--epochs"], values["--min-epochs"], values["--early-stopping-patience"]))) == expected
            assert values["--seeds"] == "42"
            assert values["--checkpoint-monitor"] == "validation_raw_quantity_rmse"
            assert values["--quantile-adaptive-strength"] == "1"
            assert values["--time-intercept-limit"] == "300"
            assert values["--quantity-variants"] == "log_mse,quantile_adaptive"
            assert "--max-train-batches" not in command and "--max-val-batches" not in command


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["execution"].__setitem__("held_out_test", True),
        lambda value: value["execution"].__setitem__("rtx5090_seeds52_and62", True),
        lambda value: value["quantile_objective"].__setitem__("raw_bin_weights", [1, 1, 2, 4, 8]),
        lambda value: value["checkpoint_selection"].__setitem__("other_metrics_in_selector", True),
        lambda value: value["runtime_policy"].__setitem__("require_no_existing_compute_process", False),
    ),
)
def test_contract_drift_fails_closed(mutation) -> None:
    changed = copy.deepcopy(contract())
    mutation(changed)
    with pytest.raises(ValueError):
        runner.validate_contract(changed)


def test_gpu_preflight_requires_idle_5090(monkeypatch) -> None:
    frozen = contract()
    replies = iter(("NVIDIA GeForce RTX 5090, 30000", "", "inactive"))
    monkeypatch.setattr(runner, "command_output", lambda *args, **kwargs: next(replies))
    assert runner.gpu_preflight(frozen)["free_vram_mib"] == 30000

    replies = iter(("NVIDIA GeForce RTX 5090, 30000", "1234", "inactive"))
    monkeypatch.setattr(runner, "command_output", lambda *args, **kwargs: next(replies))
    with pytest.raises(ValueError, match="compute processes"):
        runner.gpu_preflight(frozen)


def test_verify_source_rejects_changed_or_unsafe_files(tmp_path: Path) -> None:
    (tmp_path / "paper/contracts").mkdir(parents=True)
    (tmp_path / "paper/scripts").mkdir(parents=True)
    contract_path = tmp_path / runner.CONTRACT_REL
    trainer_path = tmp_path / runner.TRAINER_REL
    contract_path.write_text("{}", encoding="utf-8")
    trainer_path.write_text("pass\n", encoding="utf-8")
    files = {
        runner.CONTRACT_REL.as_posix(): runner.sha256_file(contract_path),
        runner.TRAINER_REL.as_posix(): runner.sha256_file(trainer_path),
    }
    manifest = {
        "source_revision": "a" * 40,
        "held_out_test_evaluated": False,
        "files": files,
    }
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert runner.verify_source(tmp_path, "a" * 40) == manifest
    trainer_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        runner.verify_source(tmp_path, "a" * 40)
    trainer_path.write_text("pass\n", encoding="utf-8")
    manifest["files"]["../escape"] = "0" * 64
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        runner.verify_source(tmp_path, "a" * 40)
