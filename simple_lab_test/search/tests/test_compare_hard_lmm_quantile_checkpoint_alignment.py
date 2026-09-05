"""Fail-closed A/B/C comparison tests for quantile checkpoint alignment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from paper.scripts.compare_hard_lmm_quantile_checkpoint_alignment import (
    A_VARIANT,
    ALL_STRATA,
    B_VARIANT,
    C_VARIANT,
    FRESH_ROLE,
    compare_all,
    write_outputs,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


DATASETS = ("dataset_alpha", "dataset_beta", "dataset_gamma")
SOURCE_REVISION = "a" * 40
STATE_DIGEST = "b" * 64
FRESH_STATE = {"weight": torch.tensor([1.0])}
FRESH_STATE_DIGEST = canonical_state_dict_sha256(FRESH_STATE)
IDENTITY_DIGEST = "c" * 64
TRAIN_IDENTITY_DIGEST = "1" * 64
TRAIN_QUANTITY_DIGEST = "2" * 64
ADAPTIVE_CONTRACT_DIGEST = "3" * 64
VALIDATION_QUANTITY_DIGEST = "4" * 64
ADAPTIVE_BOUNDARIES = [2.0, 5.0, 10.0, 20.0]
ADAPTIVE_BIN_COUNTS = [100, 60, 20, 15, 5]
ADAPTIVE_NORMALIZATION_MEAN = 1.175


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantity_contract() -> dict:
    return {
        "quantiles": [0.5, 0.9, 0.95, 0.99],
        "boundaries": [2.0, 5.0, 10.0, 20.0],
        "strata": [
            {
                "stratum": stratum,
                "stratum_order": order,
                "stratum_label": stratum,
            }
            for order, stratum in enumerate(ALL_STRATA)
        ],
    }


def _quantity_rows(
    variant: str,
    *,
    raw_rmse: float,
    body_mae: float,
    tail_mae: float,
    time_nll: float,
) -> list[dict]:
    counts = (20, 20, 20, 20, 20)
    return [
        {
            "backbone": "titantpp",
            "variant": variant,
            "seed": 42,
            "stratum": stratum,
            "stratum_order": order,
            "count": count,
            "share": count / sum(counts),
            "qty_mae": tail_mae if stratum == "gt_p99" else body_mae,
            "qty_rmse": raw_rmse,
            "time_nll": time_nll,
        }
        for order, (stratum, count) in enumerate(zip(ALL_STRATA, counts))
    ]


def _summary(
    variant: str,
    *,
    raw_rmse: float,
    body_mae: float,
    tail_mae: float,
    time_nll: float,
    checkpoint_name: str,
    aligned: bool,
) -> dict:
    payload = {
        "status": "success",
        "backbone": "titantpp",
        "variant": variant,
        "seed": 42,
        "best_epoch": 2,
        "best_val_qty_rmse": raw_rmse,
        "best_val_qty_mae": body_mae,
        "best_val_time_nll": time_nll,
        "source_revision": SOURCE_REVISION,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_path": f"/stale/server/path/{checkpoint_name}",
        "checkpoint_state_sha256": FRESH_STATE_DIGEST if aligned else STATE_DIGEST,
        "initial_state_sha256": "5" * 64,
        "quantity_rows": _quantity_rows(
            variant,
            raw_rmse=raw_rmse,
            body_mae=body_mae,
            tail_mae=tail_mae,
            time_nll=time_nll,
        ),
    }
    if aligned:
        payload.update(
            {
                "checkpoint_monitor": "validation_raw_quantity_rmse",
                "checkpoint_monitor_history_key": "val_qty_rmse",
                "checkpoint_selection": "best_validation_raw_quantity_rmse",
                "selection_formula": (
                    "sqrt(mean((predicted_raw_quantity - raw_quantity)^2))"
                ),
                "selected_metric_value": raw_rmse,
            }
        )
    return payload


def _make_a_reference(
    project_root: Path,
    dataset: str,
    *,
    raw_rmse: float = 10.0,
    body_mae: float = 10.0,
    tail_mae: float = 10.0,
    time_nll: float = 1.0,
) -> dict:
    artifact = project_root / "references" / dataset
    launch_path = artifact / "launch_contract.json"
    launch = {
        "status": "complete",
        "dataset": dataset,
        "source_revision": SOURCE_REVISION,
        "data_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "lookback_weeks": 52,
        "max_seq_len": 64,
        "split_rows": {"train": 200, "validation": 100, "test": 50},
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "quantity_contract": _quantity_contract(),
    }
    _write_json(launch_path, launch)
    run_dir = artifact / "runs/titantpp" / A_VARIANT / "seed_42"
    checkpoint = run_dir / "best_val_joint_objective_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"A checkpoint {dataset}".encode())
    summary = _summary(
        A_VARIANT,
        raw_rmse=raw_rmse,
        body_mae=body_mae,
        tail_mae=tail_mae,
        time_nll=time_nll,
        checkpoint_name=checkpoint.name,
        aligned=False,
    )
    summary_path = run_dir / "summary.json"
    _write_json(summary_path, summary)
    return {
        "artifact_path": str(artifact.relative_to(project_root)),
        "source_revision": SOURCE_REVISION,
        "launch_contract_sha256": _sha256(launch_path),
        "summary_sha256": _sha256(summary_path),
        "checkpoint_file_sha256": _sha256(checkpoint),
        "checkpoint_state_sha256": STATE_DIGEST,
    }


def _make_fresh(
    artifact_root: Path,
    dataset: str,
    *,
    b_rmse: float = 9.0,
    c_rmse: float = 8.0,
    b_body: float = 10.0,
    c_body: float = 10.1,
    b_tail: float = 10.0,
    c_tail: float = 10.1,
    b_time: float = 1.0,
    c_time: float = 1.005,
) -> Path:
    artifact = artifact_root / dataset / FRESH_ROLE
    adaptive_contract = {
        "statistics_source_split": "train",
        "statistics_population": "exact_canonical_next_event_targets",
        "quantiles": [0.5, 0.9, 0.95, 0.99],
        "raw_bin_weights": [1.0, 1.0, 1.5, 2.0, 3.0],
        "interpolation": "nearest",
        "boundary_semantics": "equality_in_lower_bin",
        "boundaries": ADAPTIVE_BOUNDARIES,
        "bin_counts": ADAPTIVE_BIN_COUNTS,
        "contract_sha256": ADAPTIVE_CONTRACT_DIGEST,
        "normalization": "raw_weight_divided_by_float64_train_target_mean",
        "normalization_mean": ADAPTIVE_NORMALIZATION_MEAN,
        "normalized_train_target_mean": 1.0,
        "reduction": "target_event_mean_without_batch_renormalization",
        "population": {
            "split": "train",
            "target_count": 200,
            "target_identity_sha256": TRAIN_IDENTITY_DIGEST,
            "target_quantity_sha256": TRAIN_QUANTITY_DIGEST,
        },
    }
    launch = {
        "status": "complete",
        "dataset": dataset,
        "model_role": FRESH_ROLE,
        "backbones": ["titantpp"],
        "quantity_variants": [B_VARIANT, C_VARIANT],
        "seeds": [42],
        "expected_run_count": 2,
        "completed_run_count": 2,
        "epochs": 300,
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": 52,
        "max_seq_len": 64,
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "partial_smoke": False,
        "max_series": None,
        "data_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "source_revision": SOURCE_REVISION,
        "time_head": {
            "mode": "legacy_clamped_rmtpp",
            "time_intercept_limit": 300.0,
        },
        "early_stopping": {
            "monitor": "validation_raw_quantity_rmse",
            "min_epochs": 40,
            "patience": 40,
            "restore": "best_validation_raw_quantity_rmse",
        },
        "quantity_contract": _quantity_contract(),
        "quantile_adaptive_contract": adaptive_contract,
        "interfaces": {
            B_VARIANT: {"quantity_loss": "mse_on_log1p_quantity"},
            C_VARIANT: {
                "quantity_loss": (
                    "train_quantile_weighted_mse_on_log1p_quantity"
                ),
                "quantile_adaptive_strength": 1.0,
                "quantile_adaptive_contract": adaptive_contract,
            },
        },
        "validation_target_population": {
            "split": "validation",
            "target_count": 100,
            "target_identity_sha256": IDENTITY_DIGEST,
            "target_quantity_sha256": VALIDATION_QUANTITY_DIGEST,
        },
        "validation_target_population_by_variant": {
            variant: {
                "target_count": 100,
                "target_identity_sha256": IDENTITY_DIGEST,
            }
            for variant in (B_VARIANT, C_VARIANT)
        },
    }
    _write_json(artifact / "launch_contract.json", launch)
    specifications = {
        B_VARIANT: (b_rmse, b_body, b_tail, b_time),
        C_VARIANT: (c_rmse, c_body, c_tail, c_time),
    }
    summaries: list[dict] = []
    quantity_rows: list[dict] = []
    for variant, (rmse, body, tail, time_nll) in specifications.items():
        run_dir = artifact / "runs/titantpp" / variant / "seed_42"
        checkpoint = run_dir / "best_val_qty_rmse_model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": "titantpp",
                "variant": variant,
                "seed": 42,
                "source_revision": SOURCE_REVISION,
                "checkpoint_monitor": "validation_raw_quantity_rmse",
                "checkpoint_monitor_history_key": "val_qty_rmse",
                "checkpoint_selection": "best_validation_raw_quantity_rmse",
                "initial_state_sha256": "5" * 64,
                "model_state_dict": FRESH_STATE,
                "model_state_sha256": FRESH_STATE_DIGEST,
            },
            checkpoint,
        )
        summary = _summary(
            variant,
            raw_rmse=rmse,
            body_mae=body,
            tail_mae=tail,
            time_nll=time_nll,
            checkpoint_name=checkpoint.name,
            aligned=True,
        )
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "history.json",
            {
                "history": [
                    {"epoch": 1, "val_qty_rmse": rmse + 1.0},
                    {"epoch": 2, "val_qty_rmse": rmse},
                    {"epoch": 3, "val_qty_rmse": rmse},
                ]
            },
        )
        top = dict(summary)
        top.pop("quantity_rows")
        summaries.append(top)
        quantity_rows.extend(summary["quantity_rows"])
    _write_csv(artifact / "run_summaries.csv", summaries)
    _write_csv(artifact / "quantity_seed_metrics.csv", quantity_rows)
    return artifact


def _build_case(tmp_path: Path) -> tuple[dict, Path, Path]:
    project_root = tmp_path / "project"
    artifact_root = tmp_path / "fresh"
    dataset_contracts = []
    for dataset in DATASETS:
        reference = _make_a_reference(project_root, dataset)
        _make_fresh(artifact_root, dataset)
        dataset_contracts.append(
            {
                "dataset": dataset,
                "data_sha256": "d" * 64,
                "split_manifest_sha256": "e" * 64,
                "lookback": 52,
                "max_sequence_length": 64,
                "expected_train_targets": 200,
                "expected_validation_targets": 100,
                "expected_loss_boundaries": ADAPTIVE_BOUNDARIES,
                "expected_loss_bin_counts": ADAPTIVE_BIN_COUNTS,
                "expected_loss_normalization_mean": ADAPTIVE_NORMALIZATION_MEAN,
                "expected_loss_contract_sha256": ADAPTIVE_CONTRACT_DIGEST,
                "expected_train_target_identity_sha256": TRAIN_IDENTITY_DIGEST,
                "expected_train_target_quantity_sha256": TRAIN_QUANTITY_DIGEST,
                "expected_validation_target_identity_sha256": IDENTITY_DIGEST,
                "expected_validation_target_quantity_sha256": VALIDATION_QUANTITY_DIGEST,
                "reporting_body_max_train_p95": 10.0,
                "reporting_tail_min_train_p99": 20.0,
                "joint_t0_reference": reference,
            }
        )
    contract = {
        "contract_id": "test_quantile_alignment",
        "scope": {
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        },
        "datasets": dataset_contracts,
        "quantile_objective": {
            "quantiles": [0.5, 0.9, 0.95, 0.99],
            "raw_bin_weights": [1.0, 1.0, 1.5, 2.0, 3.0],
            "strength": 1.0,
        },
        "seed42_gate": {
            "required_on_every_dataset": {
                "candidate_body_mae_regression_vs_aligned_t0_max": 0.02,
                "candidate_gt_p99_mae_regression_vs_aligned_t0_max": 0.02,
                "candidate_time_loss_increase_vs_aligned_t0_max": 0.01,
                "candidate_time_loss_increase_vs_joint_t0_max": 0.01,
            }
        },
    }
    return contract, project_root, artifact_root


def _run(case: tuple[dict, Path, Path]) -> dict:
    contract, project_root, artifact_root = case
    return compare_all(
        contract=contract,
        project_root=project_root,
        artifact_root=artifact_root,
    )


def test_all_three_datasets_pass_and_outputs_are_deterministic(tmp_path: Path) -> None:
    case = _build_case(tmp_path)

    payload = _run(case)

    assert payload["status"] == "passed"
    assert payload["all_three_datasets_assessed"] is True
    assert payload["dataset_status_counts"] == {
        "passed": 3,
        "rejected": 0,
        "not_evaluable": 0,
    }
    assert all(row["gate"]["passed"] for row in payload["datasets"])
    first = tmp_path / "output_first"
    second = tmp_path / "output_second"
    write_outputs(first, payload)
    write_outputs(second, payload)
    for filename in ("decision.json", "comparison.md", "metrics.csv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


@pytest.mark.parametrize(
    ("updates", "failed_check"),
    [
        (
            {"c_rmse": 9.0},
            "candidate_raw_rmse_strictly_lower_than_aligned_t0",
        ),
        (
            {"c_rmse": 10.0},
            "candidate_raw_rmse_strictly_lower_than_joint_t0",
        ),
        (
            {"c_body": 10.21},
            "candidate_body_mae_regression_vs_aligned_t0_at_most_2pct",
        ),
        (
            {"c_tail": 10.21},
            "candidate_gt_p99_mae_regression_vs_aligned_t0_at_most_2pct",
        ),
        (
            {"c_time": 1.011},
            "candidate_time_nll_increase_vs_aligned_t0_at_most_0_01",
        ),
        (
            {"b_time": 0.98, "c_time": 1.001},
            "candidate_time_nll_increase_vs_aligned_t0_at_most_0_01",
        ),
        (
            {"b_time": 1.02, "c_time": 1.011},
            "candidate_time_nll_increase_vs_joint_t0_at_most_0_01",
        ),
    ],
)
def test_each_seed42_guardrail_rejects(
    tmp_path: Path, updates: dict, failed_check: str
) -> None:
    case = _build_case(tmp_path)
    _, _, artifact_root = case
    _make_fresh(artifact_root, DATASETS[0], **updates)

    payload = _run(case)

    first = payload["datasets"][0]
    assert payload["status"] == "rejected"
    assert first["status"] == "rejected"
    assert first["gate"]["checks"][failed_check] is False
    assert len(payload["datasets"]) == 3


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_missing_or_duplicate_summary_rows_are_not_evaluable(
    tmp_path: Path, mode: str
) -> None:
    case = _build_case(tmp_path)
    _, _, artifact_root = case
    path = artifact_root / DATASETS[0] / FRESH_ROLE / "run_summaries.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = rows[:1] if mode == "missing" else [*rows, dict(rows[0])]
    _write_csv(path, rows)

    payload = _run(case)

    assert payload["status"] == "not_evaluable"
    assert payload["datasets"][0]["status"] == "not_evaluable"
    assert "exactly two fresh summary rows" in payload["datasets"][0]["errors"][0]


def test_nonfinite_metric_is_not_evaluable(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    _, _, artifact_root = case
    artifact = artifact_root / DATASETS[0] / FRESH_ROLE
    path = artifact / "run_summaries.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["best_val_qty_rmse"] = "nan"
    rows[1]["selected_metric_value"] = "nan"
    _write_csv(path, rows)

    payload = _run(case)

    assert payload["status"] == "not_evaluable"
    assert "must be finite" in payload["datasets"][0]["errors"][0]


def test_pinned_a_hash_mismatch_is_not_evaluable(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    contract, project_root, _ = case
    launch = (
        project_root
        / contract["datasets"][0]["joint_t0_reference"]["artifact_path"]
        / "launch_contract.json"
    )
    launch.write_text(launch.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    payload = _run(case)

    assert payload["status"] == "not_evaluable"
    assert "A launch SHA-256 mismatch" in payload["datasets"][0]["errors"][0]


def test_target_identity_mismatch_is_not_evaluable(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    _, _, artifact_root = case
    path = artifact_root / DATASETS[0] / FRESH_ROLE / "launch_contract.json"
    launch = json.loads(path.read_text(encoding="utf-8"))
    launch["validation_target_population_by_variant"][C_VARIANT][
        "target_identity_sha256"
    ] = "f" * 64
    _write_json(path, launch)

    payload = _run(case)

    assert payload["status"] == "not_evaluable"
    assert "disagrees with shared validation population" in payload["datasets"][0]["errors"][0]


def test_comparison_assesses_all_datasets_after_error_and_rejection(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    contract, project_root, artifact_root = case
    bad_launch = (
        project_root
        / contract["datasets"][0]["joint_t0_reference"]["artifact_path"]
        / "launch_contract.json"
    )
    bad_launch.write_text(bad_launch.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _make_fresh(artifact_root, DATASETS[1], c_body=11.0)

    payload = _run(case)

    assert [row["status"] for row in payload["datasets"]] == [
        "not_evaluable",
        "rejected",
        "passed",
    ]
    assert payload["dataset_status_counts"] == {
        "passed": 1,
        "rejected": 1,
        "not_evaluable": 1,
    }
    assert payload["status"] == "not_evaluable"


def test_mixed_fresh_source_revisions_are_not_evaluable(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    _, _, artifact_root = case
    artifact = artifact_root / DATASETS[1] / FRESH_ROLE
    mixed_revision = "f" * 40
    launch_path = artifact / "launch_contract.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["source_revision"] = mixed_revision
    _write_json(launch_path, launch)
    summaries_path = artifact / "run_summaries.csv"
    with summaries_path.open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    for summary in summaries:
        summary["source_revision"] = mixed_revision
        run_summary_path = (
            artifact
            / "runs/titantpp"
            / summary["variant"]
            / "seed_42/summary.json"
        )
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        run_summary["source_revision"] = mixed_revision
        _write_json(run_summary_path, run_summary)
        checkpoint_path = run_summary_path.parent / "best_val_qty_rmse_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["source_revision"] = mixed_revision
        torch.save(checkpoint, checkpoint_path)
    _write_csv(summaries_path, summaries)

    payload = _run(case)

    assert payload["status"] == "not_evaluable"
    assert payload["fresh_source_revisions"] == [SOURCE_REVISION, mixed_revision]
    assert all(row["status"] == "not_evaluable" for row in payload["datasets"])
    assert "mix source revisions" in payload["datasets"][0]["errors"][0]
