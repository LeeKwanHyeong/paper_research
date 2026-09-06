from __future__ import annotations

import json
import copy
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_time_head_refit_5090 import (
    EXPECTED_CONTRACT_SHA256,
    TIME_KEYS,
    audit_run,
    runner_command,
    sha256_file,
    validate_contract,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_time_head_refit_v1.json"


def load_contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_runner_e1_is_one_full_data_epoch_and_uses_the_runner_cli_contract(tmp_path):
    row = {
        "dataset": "synthetic",
        "data_path": "data.parquet",
        "split_manifest_path": "split.json",
    }
    command = runner_command(
        python="python",
        project=tmp_path,
        b_root=tmp_path / "B",
        output=tmp_path / "out",
        revision="a" * 40,
        row=row,
        phase="e1",
    )

    assert "--calibration-source-revision" in command
    assert "--source-revision" not in command
    assert command[command.index("--max-epochs") + 1] == "1"
    assert "--max-train-batches" not in command
    assert "--max-validation-batches" not in command
    full_command = runner_command(
        python="python",
        project=tmp_path,
        b_root=tmp_path / "B",
        output=tmp_path / "out" / "full" / "synthetic",
        revision="a" * 40,
        row=row,
        phase="full",
    )
    assert "--feature-cache-dir" in full_command
    assert full_command[full_command.index("--feature-cache-dir") + 1] == str(
        tmp_path / "out" / "e1" / "synthetic" / "cache"
    )


def test_checked_in_contract_has_the_frozen_common_policy():
    datasets = validate_contract(load_contract())
    assert tuple(datasets) == (
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    )


def test_contract_validation_rejects_tolerance_and_masking_drift():
    contract = load_contract()
    weakened = copy.deepcopy(contract)
    weakened["identity_and_stability"]["time_nll_replay_absolute_tolerance"] = 1.0
    with pytest.raises(ValueError, match="time_nll_replay_absolute_tolerance"):
        validate_contract(weakened)

    unmasked = copy.deepcopy(contract)
    unmasked["parameter_boundary"]["target_quantity_masked_before_encoding"] = False
    with pytest.raises(ValueError, match="masking drift"):
        validate_contract(unmasked)


def write_synthetic_audit_artifacts(tmp_path: Path):
    from paper.scripts.run_hard_lmm_time_head_refit import FrozenFeatureCache

    source_state = {
        "v_t.weight": torch.zeros(1, 2),
        "b_t": torch.zeros(1),
        "w_raw": torch.zeros(1),
        "frozen.weight": torch.tensor([3.0]),
    }
    selected_state = {name: value.clone() for name, value in source_state.items()}
    selected_state["b_t"] += 0.25
    source_state_sha256 = canonical_state_dict_sha256(source_state)
    selected_state_sha256 = canonical_state_dict_sha256(selected_state)
    source_checkpoint = tmp_path / "source.pt"
    torch.save({"model_state_dict": source_state}, source_checkpoint)
    row = {
        "dataset": "synthetic",
        "B_checkpoint_file_sha256": sha256_file(source_checkpoint),
        "B_checkpoint_state_sha256": source_state_sha256,
        "data_sha256": "d" * 64,
        "split_manifest_sha256": "s" * 64,
        "expected_train_targets": 3,
        "expected_validation_targets": 2,
        "expected_train_target_identity_sha256": "i" * 64,
        "expected_validation_target_identity_sha256": "v" * 64,
        "expected_validation_target_quantity_sha256": "u" * 64,
        "B_metrics": {
            "time_nll": 1.0,
            "overall_mae": 2.0,
            "raw_rmse": 3.0,
            "body_mae": 1.5,
            "gt_p99_mae": 4.5,
        },
    }
    output = tmp_path / "run"
    output.mkdir()
    resume_identity = {"synthetic": True}
    torch.save(
        {
            "model_state_dict": selected_state,
            "model_state_sha256": selected_state_sha256,
            "source_state_sha256": source_state_sha256,
            "selected_metric_value": 0.9,
            "resume_identity": resume_identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "backbone": "titantpp",
            "variant": "count_only_log_regression",
        },
        output / "best_validation_time_nll_model.pt",
    )
    history = [
        {"epoch": 0, "val_time_nll": 1.0},
        {"epoch": 1, "val_time_nll": 0.9},
    ]
    optimizer_state = {
        "state": {
            index: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(selected_state[name]),
                "exp_avg_sq": torch.zeros_like(selected_state[name]),
            }
            for index, name in enumerate(TIME_KEYS)
        },
        "param_groups": [
            {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "params": [0, 1, 2],
            }
        ],
    }
    torch.save(
        {
            "checkpoint_type": "time_head_refit_resume",
            "checkpoint_schema_version": 1,
            "resume_identity": resume_identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "history": history,
            "epoch": 1,
            "best_epoch": 1,
            "model_state_dict": selected_state,
            "model_state_sha256": selected_state_sha256,
            "best_state_dict": selected_state,
            "best_state_sha256": selected_state_sha256,
            "optimizer_state_dict": optimizer_state,
        },
        output / "last_epoch_state.pt",
    )
    cache_identity = {
        "schema_version": 1,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "dataset": "synthetic",
        "data_sha256": row["data_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "source_checkpoint_sha256": row["B_checkpoint_file_sha256"],
        "source_state_sha256": source_state_sha256,
        "split": "validation",
        "target_count": 2,
        "target_identity_sha256": row["expected_validation_target_identity_sha256"],
        "target_quantity_sha256": row[
            "expected_validation_target_quantity_sha256"
        ],
        "max_batches": None,
        "encoder_mode": "eval",
        "target_quantity_masked": True,
        "memory_target_write_masked": True,
        "evaluation_scope": "train_and_validation_only",
        "held_out_test_evaluated": False,
    }
    cache = FrozenFeatureCache(
        time_hidden=torch.zeros(2, 2),
        target_dt=torch.ones(2),
        quantity_hidden=torch.zeros(2, 2),
        target_quantity=torch.ones(2),
        source_quantity_prediction=torch.ones(2),
    )
    (output / "cache").mkdir()
    torch.save(
        cache.to_payload(identity=cache_identity),
        output / "cache" / "validation_features.pt",
    )
    summary = {
        "status": "success",
        "dataset": "synthetic",
        "seed": 42,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_checkpoint_sha256": row["B_checkpoint_file_sha256"],
        "source_state_sha256": source_state_sha256,
        "trainable_parameter_names": list(TIME_KEYS),
        "trainable_parameter_count": 66,
        "quantity_replay_batch_size": 128,
        "encoder_mode_during_refit": "eval",
        "hidden_state_gradient": "detached_cache",
        "source_non_time_state_sha256": "n" * 64,
        "selected_non_time_state_sha256": "n" * 64,
        "quantity_prediction_bitwise_identical": True,
        "source_quantity_prediction_sha256": "q" * 64,
        "selected_quantity_prediction_sha256": "q" * 64,
        "time_nll_non_worse_than_B": True,
        "history": history,
        "best_epoch": 1,
        "completed_epochs": 1,
        "best_val_time_nll": 0.9,
        "epoch_zero_val_time_nll": 1.0,
        "time_nll_improvement_from_B": 0.1,
        "selected_state_sha256": selected_state_sha256,
        "selected_time_head_state_sha256": "t" * 64,
        "train_cache": {"count": 3},
        "validation_cache": {"count": 2},
        "train_target_population": {"target_identity_sha256": "i" * 64},
        "validation_target_population": {"target_identity_sha256": "v" * 64},
        "quantity_metrics": {
            "mae": 2.0,
            "rmse": 3.0,
            "body_mae": 1.5,
            "gt_p99_mae": 4.5,
        },
        "qualified_full_data": False,
        "feature_cache_dir": str((output / "cache").resolve()),
        "resume_identity": resume_identity,
        "runtime": {
            "requested_device": "cuda",
            "device_name": "NVIDIA GeForce RTX 5090",
            "cuda_available": True,
            "peak_memory_allocated_bytes": 1,
            "peak_memory_reserved_bytes": 1,
            "elapsed_seconds": 1.0,
        },
        "time_guardrails": {
            "non_worse_than_B": None,
            "restores_A_plus_0_01": None,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return output, row, source_checkpoint, summary


def test_audit_rehashes_selected_state_and_enforces_the_parameter_boundary(
    tmp_path, monkeypatch
):
    import paper.scripts.run_hard_lmm_time_head_refit as refit

    monkeypatch.setattr(refit, "build_source_model", lambda _: torch.nn.Linear(1, 1))
    monkeypatch.setattr(refit, "evaluate_cached_time_nll", lambda **_: 0.9)
    output, row, source_checkpoint, summary = write_synthetic_audit_artifacts(tmp_path)
    audit = audit_run(
        output,
        contract=load_contract(),
        row=row,
        phase="e1",
        source_checkpoint=source_checkpoint,
    )
    assert audit["changed_state_keys"] == ["b_t"]

    selected_path = output / "best_validation_time_nll_model.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    selected["model_state_dict"]["frozen.weight"] += 1.0
    selected["model_state_sha256"] = canonical_state_dict_sha256(
        selected["model_state_dict"]
    )
    summary["selected_state_sha256"] = selected["model_state_sha256"]
    torch.save(selected, selected_path)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the time head changed"):
        audit_run(
            output,
            contract=load_contract(),
            row=row,
            phase="e1",
            source_checkpoint=source_checkpoint,
        )
