from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from paper.scripts import audit_hard_lmm_level_history_qkv_components as diagnostic


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    PROJECT_ROOT
    / "paper/contracts/hard_lmm_level_history_qkv_component_diagnostic_v1.json"
)


def _canonical(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def test_prospective_contract_is_exact_and_train_only():
    contract = diagnostic.validate_contract(CONTRACT)
    assert contract["scope"]["target_split"] == "train"
    assert contract["scope"]["input_splits_materialized"] == ["train"]
    assert contract["scope"]["validation_targets"] is False
    assert contract["scope"]["held_out_test"] is False
    assert tuple(contract["variants"]) == diagnostic.VARIANT_ORDER


def test_contract_hash_drift_is_rejected(tmp_path: Path):
    changed = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed["purpose"] = "changed after outputs"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hash drift"):
        diagnostic.validate_contract(path)


def test_series_fold_assignment_is_typed_stable_and_series_disjoint():
    values = np.asarray(["7", "7", 7, 7.0, True, True, "part-A", "part-A"], dtype=object)
    folds = diagnostic.assign_series_folds(values)
    assert folds.dtype == np.int8
    assert folds[0] == folds[1]
    assert folds[2] == folds[3]
    assert folds[4] == folds[5]
    assert folds[6] == folds[7]
    expected = []
    for value in values:
        token = diagnostic._series_token(value)
        payload = f"{diagnostic.FOLD_SALT}:{token}".encode("utf-8")
        expected.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2)
    assert folds.tolist() == expected


def test_variant_state_changes_only_disabled_nonzero_kernels():
    keys = {
        "Q": "encoder.layers.0.attn.level_history_q_kernel",
        "K": "encoder.layers.0.attn.level_history_k_kernel",
        "V": "encoder.layers.0.attn.level_history_v_kernel",
    }
    source = {
        "shared.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        keys["Q"]: torch.ones(2, 64),
        keys["K"]: torch.full((2, 64), 2.0),
        keys["V"]: torch.full((2, 64), 3.0),
    }
    derived, audit = diagnostic.derive_variant_state(
        source,
        mask=[1, 0, 0],
        kernel_keys=keys,
        canonical_state_dict_sha256=_canonical,
    )
    assert torch.equal(derived["shared.weight"], source["shared.weight"])
    assert torch.equal(derived[keys["Q"]], source[keys["Q"]])
    assert derived[keys["K"]].count_nonzero() == 0
    assert derived[keys["V"]].count_nonzero() == 0
    assert audit["changed_state_keys"] == [keys["K"], keys["V"]]
    source["shared.weight"][0, 0] = -100.0
    assert derived["shared.weight"][0, 0] == 0.0


def test_zero_source_kernel_is_not_falsely_reported_as_changed():
    keys = {
        "Q": "encoder.layers.0.attn.level_history_q_kernel",
        "K": "encoder.layers.0.attn.level_history_k_kernel",
        "V": "encoder.layers.0.attn.level_history_v_kernel",
    }
    source = {
        "shared": torch.ones(1),
        keys["Q"]: torch.zeros(2, 64),
        keys["K"]: torch.ones(2, 64),
        keys["V"]: torch.ones(2, 64),
    }
    _, audit = diagnostic.derive_variant_state(
        source,
        mask=[0, 1, 1],
        kernel_keys=keys,
        canonical_state_dict_sha256=_canonical,
    )
    assert audit["changed_state_keys"] == []
    assert audit["kernel_checks"]["Q"]["disabled_exact_zero"] is True


def test_quantity_metric_bias_decomposition_and_paired_shift():
    target = np.asarray([1.0, 2.0, 4.0, 8.0])
    reference = np.asarray([2.0, 2.0, 3.0, 9.0])
    candidate = reference - 0.25
    metric = diagnostic.quantity_metrics(target, candidate)
    assert metric["mse"] == pytest.approx(metric["centered_mse"] + metric["bias_squared"])
    comparison = diagnostic.compare_predictions(target, candidate, reference)
    assert comparison["mean_prediction_shift"] == pytest.approx(-0.25)
    assert comparison["delta_mse"] == pytest.approx(
        comparison["candidate"]["mse"] - comparison["reference"]["mse"]
    )


def _fold_block():
    history = np.asarray([2, 3, 2, 3, 8, 15, 8, 15], dtype=np.int64)
    target = np.full(8, 10.0)
    b = np.asarray([8, 12, 8, 12, 8, 12, 8, 12], dtype=np.float64)
    off = np.asarray([7, 13, 7, 13, 7, 13, 7, 13], dtype=np.float64)
    q = np.asarray([9, 11, 9, 11, 10, 10, 10, 10], dtype=np.float64)
    full = np.asarray([7.5, 12.5, 7.5, 12.5, 9, 11, 9, 11], dtype=np.float64)
    return history, target, b, off, q, full


def test_predeclared_gate_selects_only_path_that_preserves_short_and_improves_long():
    history_one, target_one, b_one, off_one, q_one, full_one = _fold_block()
    history = np.tile(history_one, 2)
    target = np.tile(target_one, 2)
    b = np.tile(b_one, 2)
    off = np.tile(off_one, 2)
    q = np.tile(q_one, 2)
    full = np.tile(full_one, 2)
    folds = np.repeat(np.asarray([0, 1], dtype=np.int8), len(history_one))
    variants = {
        "residual_off": off,
        "q_only": q,
        "k_only": off.copy(),
        "v_only": off.copy(),
        "qk": q.copy(),
        "qv": q.copy(),
        "kv": off.copy(),
        "full_qkv": full,
    }
    contract = diagnostic.validate_contract(CONTRACT)
    analysis, fold_rows, history_rows = diagnostic.analyze_predictions(
        target=target,
        history=history,
        folds=folds,
        b_prediction=b,
        variants=variants,
        contract=contract,
    )
    decision = analysis["decision"]
    assert decision["full_pattern_replicated"] is True
    assert decision["eligible_single_paths"] == ["q_only"]
    assert decision["selected_path_or_null"] == "q_only"
    assert decision["continue_or_stop_LPHC_family"] == "freeze_one_smaller_common_confidence_candidate"
    assert len(fold_rows) == (1 + len(diagnostic.VARIANT_ORDER)) * 2
    assert any(row["history"] == "2-3" for row in history_rows)


def test_candidate_is_not_selected_when_full_pattern_does_not_replicate():
    history_one, target_one, b_one, off_one, q_one, _ = _fold_block()
    history = np.tile(history_one, 2)
    target = np.tile(target_one, 2)
    b = np.tile(b_one, 2)
    off = np.tile(off_one, 2)
    q = np.tile(q_one, 2)
    folds = np.repeat(np.asarray([0, 1], dtype=np.int8), len(history_one))
    variants = {name: off.copy() for name in diagnostic.VARIANT_ORDER}
    variants["q_only"] = q
    variants["full_qkv"] = b.copy()
    analysis, _, _ = diagnostic.analyze_predictions(
        target=target,
        history=history,
        folds=folds,
        b_prediction=b,
        variants=variants,
        contract=diagnostic.validate_contract(CONTRACT),
    )
    assert analysis["decision"]["full_pattern_replicated"] is False
    assert analysis["decision"]["selected_path_or_null"] is None
    assert analysis["decision"]["continue_or_stop_LPHC_family"] == "stop_LPHC_QKV_family"
