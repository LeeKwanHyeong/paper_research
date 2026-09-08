from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTitanBoundedQK import BOUNDED_QK_KERNEL_KEYS
from paper.scripts import audit_hard_lmm_bounded_qk as audit_module
from paper.scripts.audit_hard_lmm_bounded_qk import (
    audit_bounded_qk_job,
    evaluate_quantity_gate,
    evaluate_time_gate,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


DATASET = "insta_market_basket"
SOURCE_REVISION = "a" * 40


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _quantity_rows(binding: dict[str, object]) -> list[dict[str, object]]:
    counts = binding["stratum_counts"]
    assert isinstance(counts, dict)
    values = {
        "le_p50": (2.0, 3.0, 1.0),
        "p50_p90": (3.0, 4.0, 1.1),
        "p90_p95": (5.0, 6.0, 1.2),
        "p95_p99": (8.0, 9.0, 1.3),
        "gt_p99": (12.0, 14.0, 1.4),
    }
    return [
        {
            "stratum": name,
            "count": counts[name],
            "qty_mae": values[name][0],
            "qty_rmse": values[name][1],
            "time_nll": values[name][2],
        }
        for name in ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")
    ]


def _aggregate_summary_metrics(summary: dict[str, object]) -> None:
    rows = summary["quantity_rows"]
    assert isinstance(rows, list)
    count = sum(int(row["count"]) for row in rows)
    summary["best_val_qty_mae"] = sum(
        int(row["count"]) * float(row["qty_mae"]) for row in rows
    ) / count
    summary["best_val_qty_rmse"] = math.sqrt(
        sum(
            int(row["count"]) * float(row["qty_rmse"]) ** 2
            for row in rows
        )
        / count
    )
    summary["best_val_time_nll"] = sum(
        int(row["count"]) * float(row["time_nll"]) for row in rows
    ) / count


def _build_e1_artifact(
    output: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    contract = json.loads(audit_module.CONTRACT_PATH.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(contract["candidate"])
    binding = copy.deepcopy(contract["data_bindings"][DATASET])
    interface = {
        "fitted_on": "train",
        "mode": "mark_free_count_aware_log_regression",
        "quantity_loss": "mse_on_log1p_quantity",
        "train_target_mean": 2.0,
        "train_target_std": 0.75,
        "time_head": {
            "mode": "legacy_clamped_rmtpp",
            "time_initial_intercept": 0.0,
            "time_initial_location": None,
            "time_initial_scale": None,
        },
    }
    population = binding["validation_target_population"]
    arguments = {
        "dataset_contract": DATASET,
        "data_sha256": binding["data_sha256"],
        "split_manifest_sha256": binding["split_manifest_sha256"],
        "lookback_weeks": population["lookback_weeks"],
        "max_seq_len": population["max_seq_len"],
        "max_train_batches": None,
        "max_val_batches": None,
        "max_series": None,
        "hidden_dim": 64,
        "batch_size": 128,
        "lr": 0.001,
        "grad_clip": 1.0,
        "epochs": 1,
        "min_epochs": 1,
        "early_stopping_patience": 40,
        "model_role": candidate["model_role"],
        "quantity_sigma_floor": 0.001,
        "lambda_location_huber": 1.0,
        "location_huber_delta": 0.25,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "tail_threshold": 46.0,
        "tail_normalization_scale": 46.0,
        "tail_clip_cap": 187.0,
        "tail_huber_delta": 1.0,
        "quantile_adaptive_strength": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 300.0,
        "time_wd_safety_limit": 40.0,
        "time_head_lr_multiplier": 1.0,
        "time_sigma_floor": 0.001,
        "titans_memory_gradient_clip": None,
        "source_revision": SOURCE_REVISION,
    }
    model, encoder_config = build_count_aware_model(
        str(candidate["backbone"]),
        hidden_dim=64,
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface["train_target_std"]),
        max_seq_len=int(population["max_seq_len"]),
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
        time_scale=3.0,
        time_w_max=10.0 / 3.0,
        time_intercept_limit=300.0,
        time_initial_intercept=0.0,
    )
    optimizer = build_optimizer(model, lr=0.001, time_head_lr_multiplier=1.0)
    dts = torch.tensor([[0.0, 1.0, 2.0, 4.0, 8.0]])
    quantities = torch.tensor([[2.0, 5.0, 9.0, 12.0, 20.0]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    loss = target_outputs(
        model, dts, mask, quantities, lambda_log_qty=1.0
    )["joint_loss"].mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    state = _clone_state(model)
    for key in BOUNDED_QK_KERNEL_KEYS:
        assert bool((state[key].abs().sum(dim=1) > 0.0).all())
    state_digest = canonical_state_dict_sha256(state)

    quantity_contract = {
        "boundaries": list(binding["quantity_boundaries"]),
        "quantiles": [0.5, 0.9, 0.95, 0.99],
        "strata": [
            {"stratum_order": index, "stratum": name, "stratum_label": name}
            for index, name in enumerate(
                ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")
            )
        ],
    }
    resume_identity = {
        "schema_version": 1,
        "backbone": candidate["backbone"],
        "variant": "count_only_log_regression",
        "seed": 42,
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "arguments": arguments,
        "quantity_contract": quantity_contract,
        "interface_meta": interface,
    }
    run_dir = (
        output
        / "runs"
        / str(candidate["backbone"])
        / "count_only_log_regression"
        / "seed_42"
    )
    run_dir.mkdir(parents=True)
    best = {
        "selection": "best_validation_raw_quantity_rmse",
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "best_epoch": 1,
        "resume_identity": resume_identity,
        "backbone": candidate["backbone"],
        "variant": "count_only_log_regression",
        "seed": 42,
        "model_state_dict": state,
        "model_state_sha256": state_digest,
        "encoder_config": encoder_config,
        "interface_meta": interface,
        "source_revision": SOURCE_REVISION,
        "source_revision_history": [SOURCE_REVISION],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    last = {
        **{key: value for key, value in best.items() if key != "model_state_dict"},
        "checkpoint_type": "epoch_resume",
        "checkpoint_schema_version": 2,
        "epoch": 1,
        "model_state_dict": copy.deepcopy(state),
        "model_state_sha256": state_digest,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_state_dict": copy.deepcopy(state),
        "best_state_sha256": state_digest,
    }

    summary: dict[str, object] = {
        "status": "success",
        "backbone": candidate["backbone"],
        "variant": "count_only_log_regression",
        "seed": 42,
        "source_revision": SOURCE_REVISION,
        "source_revision_history": [SOURCE_REVISION],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "epochs": 1,
        "completed_epochs": 1,
        "best_epoch": 1,
        "stopped_early": False,
        "training_device": "cuda",
        "cuda_peak_memory_allocated_bytes": 1024,
        "checkpoint_state_sha256": state_digest,
        "encoder_config": encoder_config,
        "interface_meta": interface,
        "resume_identity": resume_identity,
        "quantity_rows": _quantity_rows(binding),
    }
    _aggregate_summary_metrics(summary)
    selected_value = float(summary["best_val_qty_rmse"])
    summary["selected_metric_value"] = selected_value
    best["selected_metric_value"] = selected_value
    last["selected_metric_value"] = selected_value
    history_rows = [
        {
            "epoch": 1,
            "train_event_count": 1991192,
            "val_qty_rmse": selected_value,
        }
    ]
    last["history"] = copy.deepcopy(history_rows)
    torch.save(best, run_dir / "best_val_qty_rmse_model.pt")
    torch.save(last, run_dir / "last_epoch_state.pt")
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "history.json",
        {"history": history_rows},
    )
    _write_json(
        output / "launch_contract.json",
        {
            "status": "complete",
            "dataset": DATASET,
            "data_sha256": binding["data_sha256"],
            "split_manifest_sha256": binding["split_manifest_sha256"],
            "validation_target_population": binding[
                "validation_target_population"
            ],
            "quantity_contract": quantity_contract,
        },
    )
    return candidate, binding


def _run(
    output: Path, candidate: dict[str, object], binding: dict[str, object],
) -> dict[str, object]:
    return audit_bounded_qk_job(
        output,
        candidate=candidate,
        dataset=DATASET,
        source_revision=SOURCE_REVISION,
        expected_epochs=1,
        binding=binding,
    )


def test_bounded_qk_e1_audit_restores_state_optimizer_and_lags(
    tmp_path: Path,
) -> None:
    candidate, binding = _build_e1_artifact(tmp_path)

    result = _run(tmp_path, candidate, binding)

    detail = result["bounded_qk_audit"]
    assert detail["status"] == "passed"
    assert detail["strict_best_restore"] is True
    assert detail["strict_last_restore"] is True
    assert detail["optimizer_restore"] is True
    assert detail["selector"]["selected_epoch"] == 1
    assert result["validation_target_identity_sha256"] == binding[
        "validation_target_population"
    ]["target_identity_sha256"]
    for state in detail["kernel_lag_evidence"].values():
        for kernel in state.values():
            assert all(kernel[f"lag{lag}_l1"] > 0.0 for lag in range(3))


def test_bounded_qk_audit_rejects_self_consistent_zero_lag(
    tmp_path: Path,
) -> None:
    candidate, binding = _build_e1_artifact(tmp_path)
    run_dir = next(tmp_path.glob("runs/*/*/seed_42"))
    best_path = run_dir / "best_val_qty_rmse_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    key = BOUNDED_QK_KERNEL_KEYS[0]
    for state in (
        best["model_state_dict"],
        last["model_state_dict"],
        last["best_state_dict"],
    ):
        state[key][1].zero_()
    digest = canonical_state_dict_sha256(best["model_state_dict"])
    best["model_state_sha256"] = digest
    last["model_state_sha256"] = digest
    last["best_state_sha256"] = digest
    torch.save(best, best_path)
    torch.save(last, last_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_state_sha256"] = digest
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="lag1 was not trained"):
        _run(tmp_path, candidate, binding)


def test_e300_history_requires_first_early_stop_and_earliest_tie() -> None:
    history = [
        {
            "epoch": epoch,
            "train_event_count": 1991192,
            "val_qty_rmse": 1.0 if epoch in (1, 2) else 2.0,
        }
        for epoch in range(1, 42)
    ]
    summary = {
        "completed_epochs": 41,
        "best_epoch": 1,
        "best_val_qty_rmse": 1.0,
        "selected_metric_value": 1.0,
        "stopped_early": True,
    }
    best = {"best_epoch": 1, "selected_metric_value": 1.0}
    last = {"best_epoch": 1, "history": copy.deepcopy(history)}
    result = audit_module._validate_history_and_selection(
        history_payload={"history": history},
        summary=summary,
        best=best,
        last=last,
        dataset=DATASET,
        expected_epochs=300,
        minimum_epochs=40,
        patience=40,
    )
    assert result["selected_epoch"] == 1
    assert result["completed_epochs"] == 41

    late = history + [
        {"epoch": 42, "train_event_count": 1991192, "val_qty_rmse": 2.0}
    ]
    changed = dict(summary, completed_epochs=42)
    late_last = {"best_epoch": 1, "history": copy.deepcopy(late)}
    with pytest.raises(ValueError, match="continued after"):
        audit_module._validate_history_and_selection(
            history_payload={"history": late},
            summary=changed,
            best=best,
            last=late_last,
            dataset=DATASET,
            expected_epochs=300,
            minimum_epochs=40,
            patience=40,
        )


def test_quantity_gate_applies_strict_B_and_FULL_retention_boundaries() -> None:
    B = {
        "raw_rmse": 10.0,
        "overall_mae": 5.0,
        "body_mae": 4.0,
        "gt_p99_mae": 20.0,
    }
    FULL = {
        "raw_rmse": 9.0,
        "overall_mae": 4.5,
        "body_mae": 3.5,
        "gt_p99_mae": 18.0,
    }
    passing = {
        "raw_rmse": 9.0 * 1.01,
        "overall_mae": 4.5 * 1.01,
        "body_mae": 3.5 * 1.02,
        "gt_p99_mae": 18.0 * 1.02,
    }
    assert evaluate_quantity_gate(passing, B, FULL)["status"] == "passed"

    equal_B = dict(passing, raw_rmse=10.0)
    result = evaluate_quantity_gate(equal_B, B, FULL)
    assert result["status"] == "failed"
    assert result["checks"]["B_raw_rmse_strict_improvement"] is False


def test_time_gate_keeps_taxi_FULL_claim_separate_from_common_status() -> None:
    boundary = evaluate_time_gate(1.01, 1.0, 1.015, "yellow_trip_hourly")
    assert boundary["status"] == "passed"
    assert boundary["taxi_FULL_improvement_claim"]["passed"] is True

    claim_only_failure = evaluate_time_gate(
        1.005, 1.0, 1.009, "yellow_trip_hourly"
    )
    assert claim_only_failure["status"] == "passed"
    assert claim_only_failure["taxi_FULL_improvement_claim"]["passed"] is False
    assert claim_only_failure["taxi_FULL_improvement_claim"][
        "affects_overall_status"
    ] is False

    common_failure = evaluate_time_gate(
        1.010001, 1.0, 2.0, "yellow_trip_hourly"
    )
    assert common_failure["status"] == "failed"
    non_taxi = evaluate_time_gate(0.5, 0.5, 0.0, "insta_market_basket")
    assert non_taxi["taxi_FULL_improvement_claim"]["applicable"] is False
    assert non_taxi["taxi_FULL_improvement_claim"]["passed"] is None
