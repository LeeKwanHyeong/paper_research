"""Contract tests for the prospective BOUNDED-QK train-only diagnosis."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts.analyze_hard_lmm_bounded_qk_train_condition import (
    FOLD_SALT,
    REQUIRED_CONDITION_IDS,
    analyze_train_condition,
    assign_series_folds,
    bounded_cache_array_sha256,
    frozen_b_array_sha256,
    history_strata,
    run_cli,
    sha256_file,
    validate_contract,
)


CONTRACT_PATH = (
    ROOT / "paper/contracts/hard_lmm_bounded_qk_train_condition_v1.json"
)


def _series_for_each_fold(count: int) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {0: [], 1: []}
    index = 0
    while min(len(values) for values in result.values()) < count:
        value = f"series-{index}"
        fold = int(assign_series_folds([value])[0])
        result[fold].append(value)
        index += 1
    return result


def _passing_inputs() -> dict[str, np.ndarray]:
    """Build identical, prospectively passing mechanisms in both folds."""
    by_fold = _series_for_each_fold(10)
    target: list[float] = []
    prediction_b: list[float] = []
    candidate: list[float] = []
    history: list[int] = []
    series: list[str] = []
    # H2-3 is harmed by a stronger downward shift, H8-15 is helped by a
    # smaller shift, and the remaining rows make centered MSE and MAE improve.
    # Squared bias grows by 1.04 while centered MSE falls by 0.09, so the
    # level penalty prospectively overturns that gain and raw MSE rises 0.95.
    base_errors = [-1.0, 2.0, -9.0, 5.0, 5.0, 6.0, 10.0, -9.0, 0.0, 2.0]
    shifts = [-3.5, -0.5, -1.5, -3.5, -4.0, -1.5, -2.5, -3.0, -3.0, -3.0]
    histories = [2, 8, *([4] * 8)]
    for fold in (0, 1):
        for identifier, base_error, shift, length in zip(
            by_fold[fold][:10], base_errors, shifts, histories, strict=True
        ):
            target.append(20.0)
            prediction_b.append(20.0 + base_error)
            candidate.append(20.0 + base_error + shift)
            history.append(length)
            series.append(identifier)
    return {
        "target": np.asarray(target, dtype=np.float64),
        "prediction_b": np.asarray(prediction_b, dtype=np.float64),
        "prediction_bounded_qk": np.asarray(candidate, dtype=np.float64),
        "history_length": np.asarray(history, dtype=np.int64),
        "series_id": np.asarray(series, dtype=np.str_),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_fixed_contract_and_analyzer_have_identical_gate_ids() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    validate_contract(contract)
    assert tuple(contract["analysis"]["required_conditions"]) == REQUIRED_CONDITION_IDS

    drifted = copy.deepcopy(contract)
    drifted["analysis"]["required_conditions"].pop("mae_not_worse")
    with pytest.raises(ValueError, match="required-condition IDs"):
        validate_contract(drifted)

    reversed_criterion = copy.deepcopy(contract)
    reversed_criterion["analysis"]["required_conditions"][
        "centered_mse_improves"
    ] = "centered_MSE(BOUNDED_QK) - centered_MSE(B) > 0"
    with pytest.raises(ValueError, match="criteria"):
        validate_contract(reversed_criterion)


def test_typed_sha_folds_are_deterministic_and_series_disjoint() -> None:
    series = np.asarray(["1", 1, True, "repeat", "repeat"], dtype=object)
    observed = assign_series_folds(series, salt=FOLD_SALT)
    repeated = assign_series_folds(series[::-1], salt=FOLD_SALT)[::-1]
    assert np.array_equal(observed, repeated)
    assert observed[3] == observed[4]

    tokens = ["str:1", "int:1", "bool:1"]
    expected = []
    import hashlib

    for token in tokens:
        digest = hashlib.sha256(f"{FOLD_SALT}:{token}".encode()).digest()
        expected.append(int.from_bytes(digest[:8], "big") % 2)
    assert observed[:3].tolist() == expected


def test_both_folds_must_pass_all_prospective_conditions() -> None:
    inputs = _passing_inputs()
    result = analyze_train_condition(**inputs)

    assert result["decision"] == "necessary_condition_met"
    assert result["gate_checklist"]["all_folds_pass"] is True
    assert result["input_audit"]["series_disjoint"] is True
    assert result["input_audit"]["fold_series_intersection_count"] == 0
    assert result["scope"]["target_quantity_strata_used_for_decision"] is False
    assert result["scope"]["cluster_bootstrap_used_for_decision"] is False
    assert "target_strata" not in result
    assert json.loads(json.dumps(result, allow_nan=False))["status"] == "success"

    for fold in result["gate_checklist"]["folds"]:
        assert tuple(check["id"] for check in fold["checks"]) == REQUIRED_CONDITION_IDS
        assert all(check["passed"] for check in fold["checks"])
    for scope in [
        result["pooled_metrics"],
        *result["fold_metrics"],
        *[
            row
            for row in result["history_metrics"]["pooled"]
            + result["history_metrics"]["by_fold"]
            if row["available"]
        ],
    ]:
        assert scope["B"]["numeric_audit"][
            "mse_equals_centered_mse_plus_bias_squared"
        ]
        assert scope["BOUNDED_QK"]["numeric_audit"][
            "mse_equals_centered_mse_plus_bias_squared"
        ]
        assert all(
            value is True
            for key, value in scope["BOUNDED_QK_minus_B"]["numeric_audit"].items()
            if key not in {"dtype", "residuals", "tolerances"}
        )


def test_one_failed_fold_stops_the_entire_family() -> None:
    inputs = _passing_inputs()
    folds = assign_series_folds(inputs["series_id"])
    # Break the H8-15 improvement in fold 0 only.
    row = np.flatnonzero((folds == 0) & (inputs["history_length"] == 8))[0]
    inputs["prediction_bounded_qk"][row] = inputs["prediction_b"][row] + 2.0
    result = analyze_train_condition(**inputs)

    assert result["decision"] == "stop_bounded_qk_family"
    by_fold = {row["series_fold"]: row for row in result["gate_checklist"]["folds"]}
    fold_zero = {check["id"]: check for check in by_fold[0]["checks"]}
    assert fold_zero["history_8_15_mse_improves"]["passed"] is False
    assert by_fold[0]["passed"] is False
    assert by_fold[1]["passed"] is True


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("target", np.asarray([np.nan]), "finite"),
        ("prediction_b", np.asarray([-1.0]), "nonnegative"),
        ("prediction_bounded_qk", np.asarray([np.inf]), "finite"),
        ("history_length", np.asarray([0]), r"\[1, 63\]"),
        ("history_length", np.asarray([2.5]), "finite integers"),
    ],
)
def test_input_guards_reject_invalid_numeric_rows(
    field: str, replacement: np.ndarray, message: str
) -> None:
    inputs = _passing_inputs()
    values = inputs[field].astype(replacement.dtype, copy=True)
    values[0] = replacement[0]
    inputs[field] = values
    with pytest.raises(ValueError, match=message):
        analyze_train_condition(**inputs)


def test_history_boundaries_are_frozen() -> None:
    assert history_strata([1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63]).tolist() == [
        "1",
        "2-3",
        "2-3",
        "4-7",
        "4-7",
        "8-15",
        "8-15",
        "16-31",
        "16-31",
        "32-63",
        "32-63",
    ]


def test_cli_verifies_cache_alignment_audit_hashes_and_writes_atomic_outputs(
    tmp_path: Path,
) -> None:
    inputs = _passing_inputs()
    count = len(inputs["target"])
    series_parts = inputs["series_id"].copy()
    common = {
        "quantity": inputs["target"],
        "time_nll": np.zeros(count, dtype=np.float64),
        "history_length": inputs["history_length"],
        "series_index": np.arange(count, dtype=np.int64),
        "target_index": np.arange(count, dtype=np.int64),
        "context_end": inputs["history_length"].copy(),
    }
    b_arrays = {"prediction": inputs["prediction_b"], **common}
    bounded_arrays = {
        "prediction": inputs["prediction_bounded_qk"],
        **common,
        "series_parts": series_parts,
    }
    b_cache = tmp_path / "b.npz"
    bounded_cache = tmp_path / "bounded.npz"
    _write_npz(b_cache, b_arrays)
    _write_npz(bounded_cache, bounded_arrays)

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["dataset"].update(
        {
            "expected_train_targets": count,
            "expected_train_series": count,
            "expected_train_target_series": count,
            "data_sha256": "1" * 64,
            "split_manifest_sha256": "2" * 64,
            "expected_train_identity_sha256": "3" * 64,
            "expected_train_quantity_sha256": "4" * 64,
        }
    )
    contract["models"]["B"].update(
        {
            "train_cache_file_sha256": sha256_file(b_cache),
            "train_result_file_sha256": "5" * 64,
            "train_cache_array_sha256": {
                name: frozen_b_array_sha256(value, label=name)
                for name, value in b_arrays.items()
            },
        }
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    manifest = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": bounded_cache_array_sha256(value, label=name),
        }
        for name, value in bounded_arrays.items()
    }
    bounded_spec = contract["models"]["BOUNDED_QK"]
    audit = {
        "schema": "hard_lmm_bounded_qk_train_condition_run_v1",
        "status": "success",
        "contract": {
            "id": contract["contract_id"],
            "sha256": sha256_file(contract_path),
        },
        "scope": contract["scope"],
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "calibration_fit_performed": False,
        "validation_targets_evaluated": False,
        "held_out_test_evaluated": False,
        "data": {
            "sha256": contract["dataset"]["data_sha256"],
            "split_manifest_sha256": contract["dataset"]["split_manifest_sha256"],
            "materialized_splits": ["train"],
            "target_population": {
                "target_count": count,
                "target_identity_sha256": contract["dataset"][
                    "expected_train_identity_sha256"
                ],
                "target_quantity_sha256": contract["dataset"][
                    "expected_train_quantity_sha256"
                ],
            },
        },
        "frozen_source": {
            "revision": contract["execution"]["frozen_source_revision"],
            "git_worktree_clean": True,
            "all_file_hashes_verified": True,
        },
        "frozen_b_reuse": {
            "cache_file_sha256": sha256_file(b_cache),
            "result_file_sha256": contract["models"]["B"][
                "train_result_file_sha256"
            ],
            "all_array_hashes_verified": True,
        },
        "bounded_qk_checkpoint": {
            "checkpoint_file_sha256": bounded_spec["checkpoint_file_sha256"],
            "checkpoint_state_sha256": bounded_spec["checkpoint_state_sha256"],
        },
        "state_identity": {
            "before": bounded_spec["checkpoint_state_sha256"],
            "after": bounded_spec["checkpoint_state_sha256"],
            "unchanged": True,
            "all_gradients_absent": True,
        },
        "bounded_qk_predictions": {
            "sha256": sha256_file(bounded_cache),
            "target_count": count,
            "array_manifest": manifest,
        },
    }
    audit_path = tmp_path / "run_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    output = tmp_path / "analysis"
    result = run_cli(
        argparse.Namespace(
            contract=contract_path,
            b_cache=b_cache,
            bounded_cache=bounded_cache,
            run_audit=audit_path,
            output_dir=output,
        )
    )

    assert result["decision"] == "necessary_condition_met"
    assert (output / "analysis.json").is_file()
    assert (output / "fold_metrics.csv").read_text().count("\n") == 3
    assert (output / "history_metrics.csv").read_text().count("\n") == 19
    assert not list(output.glob("*.tmp"))

    # Any row-order drift must be rejected before metrics are calculated.
    broken = dict(bounded_arrays)
    broken["context_end"] = broken["context_end"].copy()
    broken["context_end"][0] += 1
    _write_npz(bounded_cache, broken)
    with pytest.raises(ValueError, match="cache hash drifted"):
        run_cli(
            argparse.Namespace(
                contract=contract_path,
                b_cache=b_cache,
                bounded_cache=bounded_cache,
                run_audit=audit_path,
                output_dir=output,
            )
        )
