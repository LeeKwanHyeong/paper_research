from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import polars as pl
import pytest
import torch

from paper.scripts import audit_hard_lmm_level_history_qkv_mechanism as mechanism


def test_direct_script_runtime_import_bootstraps_project_root(tmp_path: Path) -> None:
    script = Path(mechanism.__file__).resolve()
    root = script.parents[2]
    probe = tmp_path / "probe.py"
    probe.write_text(
        "\n".join(
            [
                "import importlib.util",
                "import sys",
                f"root = {str(root)!r}",
                "sys.path = [entry for entry in sys.path if entry != root]",
                f"spec = importlib.util.spec_from_file_location('mechanism_probe', {str(script)!r})",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "runtime = module.import_runtime()",
                "assert 'validate_checkpoint_route' in runtime",
                "assert sys.path[0] == root",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-s", str(probe)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def _paired_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    target = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    history = [2, 3, 8, 15, 4, 1]
    baseline_error = [-2.0, 2.0, -3.0, 3.0, -4.0, 4.0]
    candidate_error = [0.9 * value for value in baseline_error]
    baseline = pl.DataFrame(
        {
            "split": ["validation"] * 6,
            "series_id": ["b", "a", "b", "a", "c", "c"],
            "target_position": [1, 1, 2, 2, 1, 2],
            "target_seq": [11, 10, 13, 12, 14, 15],
            "history_length": history,
            "true_qty": target,
            "pred_b": [truth + error for truth, error in zip(target, baseline_error)],
        }
    )
    candidate = pl.DataFrame(
        {
            "series_id": baseline["series_id"],
            "target_position": baseline["target_position"],
            "target_seq": baseline["target_seq"],
            "candidate_history_length": history,
            "candidate_true_qty": target,
            "pred_lphc": [
                truth + error for truth, error in zip(target, candidate_error)
            ],
        }
    ).reverse()
    return baseline, candidate


def _gate_thresholds(metrics: Mapping[str, Any]) -> dict[str, float]:
    baseline = metrics["B"]
    return {
        "raw_rmse_strict_max": float(baseline["raw_rmse"]),
        "overall_mae_strict_max": float(baseline["overall_mae"]),
        "centered_mse_strict_max": float(baseline["centered_mse"]),
        "history_2_3_raw_mse_inclusive_max": float(
            metrics["history"]["2_3"]["B_raw_mse"]
        ),
        "history_8_15_raw_mse_strict_max": float(
            metrics["history"]["8_15"]["B_raw_mse"]
        ),
        "absolute_mean_prediction_shift_inclusive_max": 1e-12,
        "absolute_bias_inclusive_max": 1e-12,
    }


def _metric_record() -> dict[str, Any]:
    return {
        "candidate": {
            "raw_rmse": 9.0,
            "overall_mae": 9.0,
            "centered_mse": 9.0,
            "bias": 10.0,
        },
        "history": {
            "2_3": {"candidate_raw_mse": 10.0},
            "8_15": {"candidate_raw_mse": 9.0},
        },
        "absolute_mean_prediction_shift_candidate_minus_B": 10.0,
    }


def _uniform_thresholds() -> dict[str, float]:
    return {
        "raw_rmse_strict_max": 10.0,
        "overall_mae_strict_max": 10.0,
        "centered_mse_strict_max": 10.0,
        "history_2_3_raw_mse_inclusive_max": 10.0,
        "history_8_15_raw_mse_strict_max": 10.0,
        "absolute_mean_prediction_shift_inclusive_max": 10.0,
        "absolute_bias_inclusive_max": 10.0,
    }


def _tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_final_screening_contract_digest_and_mechanism_reference_are_bound():
    contract, reference, digest = mechanism.load_screening_contract()

    assert digest == mechanism.SCREENING_CONTRACT_SHA256
    assert digest == "a4ba58bb69f40496de20869643843798a13af17ca169ee4587815c55b8d1061d"
    assert reference["cache_sha256"] == contract["mechanism_reference"][
        "paired_validation_predictions"
    ]["sha256"]
    assert reference["population"] == contract["mechanism_reference"][
        "validation_target_population"
    ]
    assert reference["thresholds"]["raw_rmse_strict_max"] == pytest.approx(
        5.872216928762446, rel=0.0, abs=0.0
    )
    assert reference["thresholds"][
        "absolute_mean_prediction_shift_inclusive_max"
    ] == pytest.approx(0.05872216930452246, rel=0.0, abs=0.0)


def test_screening_contract_mechanism_tamper_fails_closed():
    contract, _, _ = mechanism.load_screening_contract()
    tampered = copy.deepcopy(contract)
    tampered["mechanism_reference"]["gate"][
        "raw_rmse_strictly_less_than_B"
    ] = False

    with pytest.raises(ValueError, match="comparator semantics"):
        mechanism.validate_screening_contract(tampered)


def test_file_hash_tamper_fails_closed(tmp_path: Path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"frozen")
    expected = mechanism.sha256_file(path)
    assert mechanism.verify_expected_sha256(path, expected, label="artifact") == expected

    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 drift"):
        mechanism.verify_expected_sha256(path, expected, label="artifact")


def test_exact_alignment_is_identity_based_and_order_independent():
    baseline, candidate = _paired_frames()
    aligned = mechanism.align_candidate_to_b_cache(
        baseline, candidate, expected_count=6
    )

    assert aligned.height == 6
    assert aligned.select(list(mechanism.IDENTITY_COLUMNS)).to_dicts() == sorted(
        baseline.select(list(mechanism.IDENTITY_COLUMNS)).to_dicts(),
        key=lambda row: tuple(row[name] for name in mechanism.IDENTITY_COLUMNS),
    )
    assert np.array_equal(
        aligned["true_qty"].to_numpy(), aligned["candidate_true_qty"].to_numpy()
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("duplicate", "canonical identities are not unique"),
        ("missing", "target count drift"),
        ("target", "target quantity differs"),
        ("history", "history length differs"),
        ("split", "non-validation split"),
        ("identity", "identity sets differ"),
    ],
)
def test_alignment_population_and_target_tamper_fail_closed(tamper: str, message: str):
    baseline, candidate = _paired_frames()
    if tamper == "duplicate":
        rows = candidate.to_dicts()
        for name in mechanism.IDENTITY_COLUMNS:
            rows[0][name] = rows[1][name]
        candidate = pl.DataFrame(rows)
    elif tamper == "missing":
        candidate = candidate.head(5)
    elif tamper == "target":
        candidate = candidate.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.col("candidate_true_qty") + 1.0)
            .otherwise(pl.col("candidate_true_qty"))
            .alias("candidate_true_qty")
        )
    elif tamper == "history":
        candidate = candidate.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.col("candidate_history_length") + 1)
            .otherwise(pl.col("candidate_history_length"))
            .alias("candidate_history_length")
        )
    elif tamper == "split":
        baseline = baseline.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit("test"))
            .otherwise(pl.col("split"))
            .alias("split")
        )
    elif tamper == "identity":
        candidate = candidate.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(pl.lit("unmatched"))
            .otherwise(pl.col("series_id"))
            .alias("series_id")
        )

    with pytest.raises(ValueError, match=message):
        mechanism.align_candidate_to_b_cache(
            baseline, candidate, expected_count=6
        )


def test_real_metrics_improve_all_strata_and_gate_passes():
    baseline, candidate = _paired_frames()
    aligned = mechanism.align_candidate_to_b_cache(
        baseline, candidate, expected_count=6
    )
    metrics = mechanism.mechanism_metrics(
        aligned["true_qty"].to_numpy(),
        aligned["pred_lphc"].to_numpy(),
        aligned["pred_b"].to_numpy(),
        aligned["history_length"].to_numpy(),
    )
    gate = mechanism.evaluate_mechanism_gate(
        metrics, thresholds=_gate_thresholds(metrics)
    )

    assert metrics["candidate"]["raw_rmse"] < metrics["B"]["raw_rmse"]
    assert metrics["candidate"]["overall_mae"] < metrics["B"]["overall_mae"]
    assert metrics["candidate"]["centered_mse"] < metrics["B"]["centered_mse"]
    assert gate["status"] == "passed"
    assert gate["failed_checks"] == []


def test_gate_inclusive_boundaries_pass_and_strict_boundaries_are_enforced():
    gate = mechanism.evaluate_mechanism_gate(
        _metric_record(), thresholds=_uniform_thresholds()
    )

    assert gate["status"] == "passed"
    assert gate["checks"]["history_2_3_raw_mse_not_worse_than_B"] is True
    assert gate["checks"][
        "absolute_mean_prediction_shift_within_one_percent_B_RMSE"
    ] is True
    assert gate["checks"]["absolute_bias_not_larger_than_B"] is True


@pytest.mark.parametrize(
    ("path", "value", "failed_check"),
    [
        (("candidate", "raw_rmse"), 10.0, "raw_rmse_strictly_better_than_B"),
        (("candidate", "overall_mae"), 10.0, "overall_mae_strictly_better_than_B"),
        (("candidate", "centered_mse"), 10.0, "centered_mse_strictly_better_than_B"),
        (
            ("history", "2_3", "candidate_raw_mse"),
            10.000001,
            "history_2_3_raw_mse_not_worse_than_B",
        ),
        (
            ("history", "8_15", "candidate_raw_mse"),
            10.0,
            "history_8_15_raw_mse_strictly_better_than_B",
        ),
        (
            ("absolute_mean_prediction_shift_candidate_minus_B",),
            10.000001,
            "absolute_mean_prediction_shift_within_one_percent_B_RMSE",
        ),
        (("candidate", "bias"), 10.000001, "absolute_bias_not_larger_than_B"),
    ],
)
def test_each_mechanism_gate_fails_closed(
    path: tuple[str, ...], value: float, failed_check: str
):
    metrics = _metric_record()
    cursor: dict[str, Any] = metrics
    for name in path[:-1]:
        cursor = cursor[name]
    cursor[path[-1]] = value

    gate = mechanism.evaluate_mechanism_gate(
        metrics, thresholds=_uniform_thresholds()
    )

    assert gate["status"] == "failed"
    assert gate["failed_checks"] == [failed_check]


def test_b_cache_replay_uses_exact_strata_and_rejects_metric_drift():
    baseline, candidate = _paired_frames()
    aligned = mechanism.align_candidate_to_b_cache(
        baseline, candidate, expected_count=6
    )
    metrics = mechanism.mechanism_metrics(
        aligned["true_qty"].to_numpy(),
        aligned["pred_lphc"].to_numpy(),
        aligned["pred_b"].to_numpy(),
        aligned["history_length"].to_numpy(),
    )
    expected = {
        "count": metrics["B"]["count"],
        "raw_mse": metrics["B"]["raw_mse"],
        "raw_rmse": metrics["B"]["raw_rmse"],
        "overall_mae": metrics["B"]["overall_mae"],
        "bias": metrics["B"]["bias"],
        "centered_mse": metrics["B"]["centered_mse"],
        "prediction_mean": metrics["B"]["prediction_mean"],
        "target_mean": metrics["B"]["target_mean"],
        "history_2_3_count": metrics["history"]["2_3"]["count"],
        "history_2_3_raw_mse": metrics["history"]["2_3"]["B_raw_mse"],
        "history_8_15_count": metrics["history"]["8_15"]["count"],
        "history_8_15_raw_mse": metrics["history"]["8_15"]["B_raw_mse"],
    }
    differences = mechanism.validate_b_cache_metrics(metrics, expected=expected)
    assert all(value == 0.0 for value in differences.values())

    tampered = dict(expected)
    tampered["history_8_15_raw_mse"] += 1e-6
    with pytest.raises(ValueError, match="Pinned B cache history_8_15_raw_mse drift"):
        mechanism.validate_b_cache_metrics(metrics, expected=tampered)


def test_nonfinite_negative_and_empty_required_strata_fail_closed():
    with pytest.raises(ValueError, match="negative"):
        mechanism.quantity_metrics([1.0], [-1.0])
    with pytest.raises(ValueError, match="nonfinite"):
        mechanism.quantity_metrics([1.0], [float("nan")])
    with pytest.raises(ValueError, match="History 2-3 stratum is empty"):
        mechanism.mechanism_metrics([1.0], [1.0], [1.0], [8])


def test_frozen_state_and_absent_gradient_audit():
    model = torch.nn.Linear(2, 1)
    model.requires_grad_(False)
    before = _tensor_digest(model.state_dict())

    result = mechanism.audit_frozen_state(model, before, _tensor_digest)

    assert result["unchanged"] is True
    assert result["all_parameters_frozen"] is True
    assert result["all_gradients_absent"] is True


def test_state_trainability_and_gradient_tamper_fail_closed():
    trainable = torch.nn.Linear(2, 1)
    before = _tensor_digest(trainable.state_dict())
    with pytest.raises(ValueError, match="trainable parameter"):
        mechanism.audit_frozen_state(trainable, before, _tensor_digest)

    gradient = torch.nn.Linear(2, 1).requires_grad_(False)
    before = _tensor_digest(gradient.state_dict())
    gradient.weight.grad = torch.ones_like(gradient.weight)
    with pytest.raises(ValueError, match="accumulated gradients"):
        mechanism.audit_frozen_state(gradient, before, _tensor_digest)

    changed = torch.nn.Linear(2, 1).requires_grad_(False)
    before = _tensor_digest(changed.state_dict())
    with torch.no_grad():
        changed.bias.add_(1.0)
    with pytest.raises(ValueError, match="state changed"):
        mechanism.audit_frozen_state(changed, before, _tensor_digest)


def _job_audit_fixture() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    checkpoint_file_sha = "a" * 64
    checkpoint_state_sha = "b" * 64
    summary_sha = "c" * 64
    initial_sha = "d" * 64
    common_sha = "e" * 64
    population = {
        "identity_fields": ["oper_part_no", "target_position", "target_seq"],
        "loader": "RMTPPWeekLookbackDataset",
        "lookback_weeks": 52,
        "max_seq_len": 64,
        "schema_version": 1,
        "split": "validation",
        "target_count": 6,
        "target_identity_sha256": "f" * 64,
        "target_quantity_sha256": "1" * 64,
    }
    proof = {
        "method": "same_seed_fresh_B_identity_v1",
        "seed": 42,
        "B_backbone": "titantpp",
        "candidate_backbone": mechanism.CANDIDATE_BACKBONE,
        "fresh_B_state_sha256": common_sha,
        "candidate_common_state_sha256": common_sha,
        "candidate_initial_state_sha256": initial_sha,
        "new_parameter_state_keys": list(mechanism.LEVEL_HISTORY_QKV_KERNEL_KEYS),
        "new_parameters_exact_zero": True,
        "additional_parameter_count": 384,
        "inherited_state_bitwise_equal": True,
        "synthetic_outputs_bitwise_equal": True,
        "post_construction_RNG_state_bitwise_equal": True,
        "summary_and_checkpoint_initial_state_sha256_equal": True,
    }
    audit = {
        "status": "passed",
        "checkpoint_sha256": checkpoint_file_sha,
        "summary_sha256": summary_sha,
        "validation_target_identity_sha256": population["target_identity_sha256"],
        "validation_target_quantity_sha256": population["target_quantity_sha256"],
        "level_history_qkv_audit": {
            "status": "passed",
            "contract_id": mechanism.CONTRACT_ID,
            "screening_contract_sha256": mechanism.SCREENING_CONTRACT_SHA256,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "best_checkpoint_file_sha256": checkpoint_file_sha,
            "best_state_sha256": checkpoint_state_sha,
            "validation_target_population": population,
            "fresh_initial_identity": proof,
        },
    }
    checkpoint = {
        "checkpoint_file_sha256": checkpoint_file_sha,
        "checkpoint_state_sha256": checkpoint_state_sha,
        "summary_file_sha256": summary_sha,
        "initial_state_sha256": initial_sha,
    }
    summary = {"initial_state_sha256": initial_sha}
    return audit, checkpoint, summary, population


def _write_job_audit(path: Path, audit: Mapping[str, Any]) -> str:
    path.write_text(
        json.dumps(audit, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return mechanism.sha256_file(path)


def test_sha_bound_candidate_job_audit_fresh_identity_passes(tmp_path: Path):
    audit, checkpoint, summary, population = _job_audit_fixture()
    path = tmp_path / "audit.json"
    digest = _write_job_audit(path, audit)

    result = mechanism.validate_candidate_job_audit(
        path=path,
        expected_sha256=digest,
        checkpoint_audit=checkpoint,
        candidate_summary=summary,
        expected_population=population,
    )

    assert result["status"] == "passed"
    assert result["sha256"] == digest
    assert result["fresh_initial_identity"]["inherited_state_bitwise_equal"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("new_parameters_exact_zero", False, "fresh initial identity contract failed"),
        ("inherited_state_bitwise_equal", False, "fresh initial identity contract failed"),
        ("synthetic_outputs_bitwise_equal", False, "fresh initial identity contract failed"),
        (
            "post_construction_RNG_state_bitwise_equal",
            False,
            "fresh initial identity contract failed",
        ),
        (
            "summary_and_checkpoint_initial_state_sha256_equal",
            False,
            "fresh initial identity contract failed",
        ),
        ("candidate_common_state_sha256", "2" * 64, "common initial states differ"),
        ("candidate_initial_state_sha256", "3" * 64, "trained artifact"),
    ],
)
def test_candidate_job_audit_fresh_identity_tamper_fails_closed(
    tmp_path: Path, field: str, value: Any, message: str
):
    audit, checkpoint, summary, population = _job_audit_fixture()
    audit["level_history_qkv_audit"]["fresh_initial_identity"][field] = value
    path = tmp_path / "audit.json"
    digest = _write_job_audit(path, audit)

    with pytest.raises(ValueError, match=message):
        mechanism.validate_candidate_job_audit(
            path=path,
            expected_sha256=digest,
            checkpoint_audit=checkpoint,
            candidate_summary=summary,
            expected_population=population,
        )


def test_candidate_job_audit_file_and_population_tamper_fail_closed(tmp_path: Path):
    audit, checkpoint, summary, population = _job_audit_fixture()
    path = tmp_path / "audit.json"
    digest = _write_job_audit(path, audit)
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 drift"):
        mechanism.validate_candidate_job_audit(
            path=path,
            expected_sha256=digest,
            checkpoint_audit=checkpoint,
            candidate_summary=summary,
            expected_population=population,
        )

    audit["level_history_qkv_audit"]["validation_target_population"] = {
        **population,
        "target_count": 5,
    }
    digest = _write_job_audit(path, audit)
    with pytest.raises(ValueError, match="validation population drift"):
        mechanism.validate_candidate_job_audit(
            path=path,
            expected_sha256=digest,
            checkpoint_audit=checkpoint,
            candidate_summary=summary,
            expected_population=population,
        )
