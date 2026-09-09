from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts import run_raw_rmse_baseline_alignment_job as alignment
from paper.scripts import run_raw_rmse_baseline_completion_5080 as completion


CONTRACT_PATH = (
    ROOT / "paper/contracts/raw_rmse_baseline_completion_seed42_5080_v1.json"
)


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_completion_contract_is_exactly_the_four_missing_seed42_jobs() -> None:
    payload = contract()
    validated = completion.validate_contract(payload)

    assert [(row["dataset"], row["backbone"]) for row in validated["jobs"]] == [
        ("intermittent_frozen_5000", "rmtpp"),
        ("intermittent_frozen_5000", "thp"),
        ("yellow_trip_hourly", "rmtpp"),
        ("yellow_trip_hourly", "thp"),
    ]
    assert all(row["host_role"] == "5080" for row in validated["jobs"])
    assert all(
        (row["seed"], row["epochs"], row["min_epochs"], row["patience"])
        == (42, 300, 40, 40)
        for row in validated["jobs"]
    )
    assert payload["scope"]["checkpoint_rule"].startswith(
        "earliest strict finite minimum"
    )
    serialized = json.dumps(payload, sort_keys=True)
    assert '"dataset": "insta_market_basket"' not in serialized
    assert payload["execution"]["additional_seeds"] is False
    assert payload["execution"]["held_out_test"] is False
    assert payload["execution"]["retrain_instacart"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["jobs"][0].update(host_role="5090"), "escaped RTX 5080"),
        (lambda value: value["jobs"][0].update(seed=52), "Only seed42"),
        (lambda value: value["jobs"][0].update(epochs=299), "Epoch or early-stopping"),
        (
            lambda value: value["scope"].update(
                checkpoint_monitor="validation_joint_objective"
            ),
            "scope drifted",
        ),
        (
            lambda value: value["prior_campaign"].update(
                required_status="complete"
            ),
            "failure status drifted",
        ),
        (
            lambda value: value["execution"].update(held_out_test=True),
            "Forbidden execution enabled",
        ),
    ],
)
def test_contract_mutations_fail_closed(mutate, message: str) -> None:
    payload = copy.deepcopy(contract())
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        completion.validate_contract(payload)


def test_thp_training_command_reuses_frozen_raw_rmse_trainer_semantics(
    tmp_path: Path,
) -> None:
    payload = contract()
    validated = completion.validate_contract(payload)
    job = validated["jobs"][1]
    command = completion.training_command(
        python="/fixed/python",
        project=ROOT,
        root=tmp_path,
        revision="a" * 40,
        job=job,
        dataset=validated["datasets"][job["dataset"]],
    )

    assert command[0] == "/fixed/python"
    assert command[1].endswith("run_count_aware_tpp_backbone_control.py")
    assert command[command.index("--backbones") + 1] == "thp"
    assert command[command.index("--checkpoint-monitor") + 1] == (
        "validation_raw_quantity_rmse"
    )
    assert command[command.index("--epochs") + 1] == "300"
    assert command[command.index("--min-epochs") + 1] == "40"
    assert command[command.index("--early-stopping-patience") + 1] == "40"
    assert command[command.index("--seeds") + 1] == "42"
    assert command[command.index("--quantity-variants") + 1] == "log_mse"
    assert "--test" not in command
    assert "--max-batches" not in command


def test_active_gdm_permission_is_narrow_and_backward_compatible() -> None:
    legacy = {}
    assert alignment.active_gdm_is_authorized(
        legacy, host_role="5080", backbone="rmtpp"
    )
    assert not alignment.active_gdm_is_authorized(
        legacy, host_role="5080", backbone="thp"
    )

    policy = contract()["runtime_policy"]["hosts"]["5080"]
    assert alignment.active_gdm_is_authorized(
        policy, host_role="5080", backbone="rmtpp"
    )
    assert alignment.active_gdm_is_authorized(
        policy, host_role="5080", backbone="thp"
    )
    assert not alignment.active_gdm_is_authorized(
        policy, host_role="5090", backbone="thp"
    )


def test_cuda_smoke_requires_both_baseline_backbones(tmp_path: Path) -> None:
    output = tmp_path / "model_test.json"
    output.write_text(
        json.dumps(
            {
                "status": "complete",
                "device": "cuda",
                "cuda_device": "NVIDIA GeForce RTX 5080",
                "results": [
                    {
                        "backbone": "rmtpp",
                        "variant": "count_only_log_regression",
                        "finite": True,
                    },
                    {
                        "backbone": "thp",
                        "variant": "count_only_log_regression",
                        "finite": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert completion.audit_cuda_smoke(output, contract())["status"] == "complete"

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["results"] = payload["results"][:1]
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="CUDA forward-backward smoke is incomplete"):
        completion.audit_cuda_smoke(output, contract())


def test_execution_registry_is_fixed_and_revision_isolated() -> None:
    first = completion.execution_root("a" * 40)
    second = completion.execution_root("b" * 40)
    assert first != second
    assert first.name == f"{completion.CONTRACT_ID}_{'a' * 40}"
    with pytest.raises(ValueError, match="full lowercase source revision"):
        completion.execution_root("main")
