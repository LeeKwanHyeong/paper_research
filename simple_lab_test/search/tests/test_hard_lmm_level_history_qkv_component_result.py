from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from paper.scripts import audit_hard_lmm_level_history_qkv_component_result as result_audit
from paper.scripts import analyze_hard_lmm_bounded_qk_train_condition as prior_folds


def _identity_arrays():
    parts = np.asarray([0, 0, 1, 2], dtype=np.int64)
    context = np.asarray([1, 2, 1, 4], dtype=np.int64)
    quantity = np.asarray([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
    history = np.asarray([1, 2, 1, 4], dtype=np.int64)
    b_cache = {
        "target_index": np.arange(4, dtype=np.int64),
        "series_index": parts.copy(),
        "context_end": context.copy(),
        "quantity": quantity.copy(),
        "history_length": history.copy(),
    }
    component = {
        "series_index": parts.copy(),
        "quantity": quantity.copy(),
        "history_length": history.copy(),
    }
    return parts, context, b_cache, component


def test_row_identity_accepts_exact_canonical_alignment():
    parts, context, b_cache, component = _identity_arrays()
    observed = result_audit.verify_row_identity(
        b_cache=b_cache,
        component=component,
        expected_parts=parts,
        expected_context=context,
    )
    assert observed["B_target_index_is_arange"] is True
    assert observed["component_series_index_matches_canonical_dataset_index"] is True


@pytest.mark.parametrize("field", ["target_index", "series_index", "context_end"])
def test_row_identity_rejects_each_B_alignment_drift(field: str):
    parts, context, b_cache, component = _identity_arrays()
    b_cache[field] = b_cache[field].copy()
    b_cache[field][-1] += 1
    with pytest.raises(ValueError):
        result_audit.verify_row_identity(
            b_cache=b_cache,
            component=component,
            expected_parts=parts,
            expected_context=context,
        )


def test_fold_identity_replays_prior_analyzer_exactly():
    series_ids = np.asarray(["a", "a", "b", 7, 7.0, True], dtype=object)
    observed = prior_folds.assign_series_folds(series_ids, salt=result_audit.FOLD_SALT)
    audit = result_audit.verify_fold_identity(
        observed=observed,
        series_ids=series_ids,
        prior_fold_module=prior_folds,
    )
    assert audit["prior_analyzer_exact_array_equality"] is True
    assert audit["series_disjoint"] is True
    assert audit["fold_array_sha256"] == audit["recomputed_fold_array_sha256"]


def test_fold_identity_rejects_one_changed_membership():
    series_ids = np.asarray(["a", "a", "b", "b"], dtype=object)
    observed = prior_folds.assign_series_folds(series_ids, salt=result_audit.FOLD_SALT)
    observed = observed.copy()
    observed[-1] = 1 - observed[-1]
    with pytest.raises(ValueError, match="differs from prior analyzer"):
        result_audit.verify_fold_identity(
            observed=observed,
            series_ids=series_ids,
            prior_fold_module=prior_folds,
        )


def test_transitive_manifest_audit_includes_restore_module_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = [
        "paper/scripts/audit_hard_lmm_causal_qkv.py",
        "paper/scripts/count_aware_tpp_backbone/core.py",
        "paper/scripts/run_count_aware_tpp_backbone_control.py",
        "paper/scripts/run_taxi_quantity_interface_ablation.py",
        "models/TPPs/CountAwareFactory.py",
        "models/TPPs/CountAwareTPP.py",
        "data_loader/event_seq_data_module.py",
    ]
    manifest: dict[str, str] = {}
    for index, relative in enumerate(paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# source {index}\n", encoding="utf-8")
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setitem(
            __import__("sys").modules,
            f"_component_result_fake_{index}",
            SimpleNamespace(__file__=str(path)),
        )
    observed = result_audit.verify_transitive_frozen_modules(
        tmp_path, {"source_files": manifest}
    )
    assert observed["loaded_module_count"] == len(paths)
    assert observed["required_restore_module_verified"] is True

    manifest[paths[-1]] = "0" * 64
    with pytest.raises(ValueError, match="hash drift"):
        result_audit.verify_transitive_frozen_modules(tmp_path, {"source_files": manifest})


def test_mutable_project_named_module_outside_frozen_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frozen = tmp_path / "frozen"
    mutable = tmp_path / "mutable" / "utils" / "training.py"
    frozen.mkdir()
    mutable.parent.mkdir(parents=True)
    mutable.write_text("# mutable\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "utils.training", SimpleNamespace(__file__=str(mutable)))
    with pytest.raises(ValueError, match="escaped frozen source"):
        result_audit._assert_no_mutable_project_module(frozen)


def test_component_array_contract_rejects_nonfinite_prediction():
    count = 3
    component = {
        "quantity": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        "history_length": np.asarray([1, 2, 3], dtype=np.int64),
        "series_index": np.asarray([0, 0, 1], dtype=np.int64),
        "fold": np.asarray([0, 0, 1], dtype=np.int8),
        **{
            f"prediction__{name}": np.ones(count, dtype=np.float64)
            for name in result_audit.VARIANT_ORDER
        },
    }
    assert result_audit.verify_component_arrays(component, expected_count=count)["schema_exact"]
    component["prediction__q_only"][1] = np.nan
    with pytest.raises(ValueError, match="Invalid component predictions"):
        result_audit.verify_component_arrays(component, expected_count=count)


def test_scope_audit_rejects_held_out_access():
    clean = {
        "target_split": "train",
        "input_splits_materialized": ["train"],
        "validation_targets": False,
        "held_out_test": False,
        "training": False,
        "checkpoint_selection": False,
        "parameter_updates": False,
        "optimizer_use": False,
        "calibration_fit": False,
    }
    run = {
        "scope": clean.copy(),
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "validation_targets_evaluated": False,
        "held_out_test_evaluated": False,
    }
    assert result_audit.verify_scopes({"scope": clean.copy()}, run)["target_split"] == "train"
    run["scope"]["held_out_test"] = True
    with pytest.raises(ValueError, match="held_out_test"):
        result_audit.verify_scopes({"scope": clean.copy()}, run)
