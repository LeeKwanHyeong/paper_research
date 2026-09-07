"""Contract tests for the frozen raw-affine calibration runner."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from paper.scripts.frozen_raw_affine_calibration import FOLD_SALT
from paper.scripts.run_frozen_raw_affine_calibration import (
    analyze_train,
    analyze_validation,
    compute_train_quantity_boundaries,
    persist_calibration_state,
    reset_cuda_peak_memory,
    validate_contract,
    validate_prerequisite_decision,
    validate_train_result,
    verify_source_manifest,
)
from paper.scripts.run_hard_lmm_time_head_refit import sha256_file


def contract() -> dict:
    dataset_common = {
        "data_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "lookback": 10,
        "max_sequence_length": 16,
        "expected_train_targets": 10,
        "expected_validation_targets": 5,
        "expected_train_target_identity_sha256": "1" * 64,
        "expected_train_target_quantity_sha256": "2" * 64,
        "expected_validation_target_identity_sha256": "3" * 64,
        "expected_validation_target_quantity_sha256": "4" * 64,
    }
    source = {
        "backbone": "titantpp",
        "quantity_variant": "count_only_log_regression",
        "seed": 42,
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "training_source_revision": "a" * 40,
        "training_source_revision_history": ["a" * 40],
        "checkpoint_file_sha256": "b" * 64,
        "checkpoint_state_sha256": "c" * 64,
    }
    datasets = []
    for dataset in (
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    ):
        sources = {"B": dict(source)}
        if dataset == "insta_market_basket":
            sources.update(
                rmtpp={**source, "backbone": "rmtpp"},
                thp={**source, "backbone": "thp"},
            )
        datasets.append({
            "dataset": dataset,
            **dataset_common,
            "train_quantity_boundaries": {
                "p50": 1.0,
                "p90": 2.0,
                "p95": 3.0,
                "p99": 4.0,
            },
            "sources": sources,
        })
    return {
        "contract_id": "frozen_raw_affine_calibration_v1",
        "scope": {
            "input_splits": ["train", "validation"],
            "optimization_split": "train",
            "evaluation_scope": "validation_only",
            "held_out_test": False,
            "seed": 42,
        },
        "calibration": {
            "formula": "max(0, a * q_base + b)",
            "slope_constraint": "a > 0",
            "fit": "positive-slope ordinary least squares",
            "minimum_slope": 1e-12,
            "numeric_dtype": "float64",
            "manual_dataset_hyperparameters": False,
            "source_model_frozen": True,
            "fold_salt": FOLD_SALT,
        },
        "acceptance": {
            "minimum_pooled_oof_rmse_improvement_fraction": 0.01,
            "maximum_overall_mae_regression_fraction": 0.02,
            "maximum_body_mae_regression_fraction": 0.02,
            "maximum_tail_mae_regression_fraction": 0.02,
            "quantity_boundary_estimator": "numpy.quantile(method='nearest') over canonical train next-event targets",
        },
        "runtime": {
            "execution_server": "5080",
            "minimum_free_vram_mib": 12000,
            "inference_batch_size": 128,
            "smoke_targets_per_fold": 8,
            "other_gpu_hosts_allowed": False,
            "paired_bootstrap_seed": 20260907,
            "paired_bootstrap_replicates": 500,
            "paired_bootstrap_draw_chunk_size": 8192,
            "deterministic_environment": {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "datasets": datasets,
    }


def synthetic_cache() -> dict[str, np.ndarray]:
    series = np.repeat(np.arange(40), 8)
    prediction = np.linspace(1.0, 80.0, len(series))
    quantity = 1.25 * prediction + 2.0
    return {
        "prediction": prediction,
        "quantity": quantity,
        "series_index": series,
        "history_length": 1 + np.arange(len(series)) % 150,
        "time_nll": np.linspace(0.1, 0.2, len(series)),
    }


def test_contract_locks_5080_and_five_source_rows():
    rows = validate_contract(contract())
    assert set(rows["insta_market_basket"]["sources"]) == {"B", "rmtpp", "thp"}

    wrong = contract()
    wrong["runtime"]["execution_server"] = "5090"
    with pytest.raises(ValueError, match="5080"):
        validate_contract(wrong)


def test_cuda_peak_reset_uses_integer_index_for_5080_runtime(monkeypatch):
    observed = {}
    monkeypatch.setattr(
        "paper.scripts.run_frozen_raw_affine_calibration.torch.cuda.set_device",
        lambda device: observed.setdefault("device", device),
    )
    monkeypatch.setattr(
        "paper.scripts.run_frozen_raw_affine_calibration.torch.cuda.reset_peak_memory_stats",
        lambda index: observed.setdefault("index", index),
    )
    assert reset_cuda_peak_memory(torch.device("cuda:0")) == 0
    assert observed["index"] == 0
    assert not isinstance(observed["index"], torch.device)

    wrong = contract()
    wrong["acceptance"]["minimum_pooled_oof_rmse_improvement_fraction"] = 0.0
    with pytest.raises(ValueError, match="threshold"):
        validate_contract(wrong)


def test_train_fit_uses_series_oof_and_fixed_one_percent_gate():
    audit = analyze_train(synthetic_cache(), contract())
    assert audit["gate_passed"]
    assert audit["two_fold_oof"]["both_folds_raw_mse_improve"]
    assert audit["two_fold_oof"]["overall_change"]["rmse_relative_improvement"] > 0.99
    assert audit["calibration_parameters"]["slope"] == pytest.approx(1.25)
    assert audit["calibration_parameters"]["intercept"] == pytest.approx(2.0)


def test_train_quantity_boundaries_use_the_frozen_nearest_definition():
    quantity = np.arange(1.0, 102.0)
    assert compute_train_quantity_boundaries(quantity) == {
        "p50": 51.0,
        "p90": 91.0,
        "p95": 96.0,
        "p99": 100.0,
    }


def test_validation_preserves_time_and_applies_train_activated_map():
    cache = synthetic_cache()
    train_audit = analyze_train(cache, contract())
    result = analyze_validation(cache, train_audit, contract(), {}, smoke=True)
    assert result["calibration_activated_by_train_rule"]
    assert result["relative_changes"]["time_nll"] == 0.0
    assert result["calibrated"]["rmse"] < 1e-12
    assert result["baseline"]["time_nll"] == result["calibrated"]["time_nll"]


def test_failed_baseline_train_rule_deploys_identity_without_validation_selection():
    cache = synthetic_cache()
    train_audit = analyze_train(cache, contract())
    train_audit["gate_passed"] = False
    train_audit["calibration_parameters"] = None
    result = analyze_validation(cache, train_audit, contract(), {}, smoke=False)
    assert not result["calibration_activated_by_train_rule"]
    assert result["deployed_calibration_parameters"] == {"slope": 1.0, "intercept": 0.0}
    assert result["baseline"] == result["calibrated"]


def test_full_B_validation_is_blocked_after_failed_train_gate(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract()))
    payload = {
        "status": "success",
        "phase": "train",
        "dataset": "insta_market_basket",
        "model_role": "B",
        "source_revision": "f" * 40,
        "contract_sha256": "placeholder",
        "source_checkpoint_file_sha256": "b" * 64,
        "source_manifest": {"file_sha256": "c" * 64},
        "input_digests": {
            "data_sha256": "d" * 64,
            "split_manifest_sha256": "e" * 64,
        },
        "source_model_state_unchanged": True,
        "source_gradients_absent": True,
        "train_audit": {"gate_passed": False},
    }
    payload["contract_sha256"] = sha256_file(contract_path)
    args = SimpleNamespace(
        dataset="insta_market_basket",
        model_role="B",
        source_revision="f" * 40,
        contract=contract_path,
        smoke=False,
    )
    context = {
        "checkpoint_file_sha": "b" * 64,
        "source_manifest_audit": {"file_sha256": "c" * 64},
        "data_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="validation is prohibited"):
        validate_train_result(payload, context, args)


def test_staged_prerequisite_blocks_early_baseline_and_accepts_pinned_B_rows(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract()))
    args = SimpleNamespace(
        phase="train",
        model_role="rmtpp",
        smoke=False,
        prerequisite_decision=None,
        contract=contract_path,
        source_revision="f" * 40,
    )
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("manifest")
    args.source_manifest = source_manifest
    with pytest.raises(ValueError, match="required"):
        validate_prerequisite_decision(args)

    decision = {
        "status": "success",
        "accepted": True,
        "stage": "B_three_dataset_validation_gate",
        "decision": "accepted_for_instacart_fairness",
        "contract_sha256": sha256_file(contract_path),
        "source_revision": "f" * 40,
        "source_manifest_file_sha256": sha256_file(source_manifest),
        "execution_server": "5080",
        "held_out_test_evaluated": False,
        "result_identities": [],
    }
    for index, dataset in enumerate(
        ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
    ):
        result_path = tmp_path / f"B_result_{index}.json"
        result_path.write_text(dataset)
        decision["result_identities"].append({
            "dataset": dataset,
            "model_role": "B",
            "checkpoint_file_sha256": "b" * 64,
            "result_file": str(result_path),
            "result_file_sha256": sha256_file(result_path),
        })
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision))
    args.prerequisite_decision = decision_path
    audit = validate_prerequisite_decision(args)
    assert audit["decision"] == "accepted_for_instacart_fairness"


def test_B_validation_train_result_must_be_the_result_admitted_by_gate(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract()))
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("manifest")
    identities = []
    admitted = None
    for index, dataset in enumerate(
        ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
    ):
        result_path = tmp_path / f"B_train_{index}.json"
        result_path.write_text(dataset)
        if dataset == "insta_market_basket":
            admitted = result_path
        identities.append({
            "dataset": dataset,
            "model_role": "B",
            "checkpoint_file_sha256": "b" * 64,
            "result_file": str(result_path),
            "result_file_sha256": sha256_file(result_path),
        })
    decision = {
        "status": "success",
        "accepted": True,
        "stage": "B_three_dataset_train_gate",
        "decision": "accepted_for_B_validation",
        "contract_sha256": sha256_file(contract_path),
        "source_revision": "f" * 40,
        "source_manifest_file_sha256": sha256_file(source_manifest),
        "execution_server": "5080",
        "held_out_test_evaluated": False,
        "result_identities": identities,
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision))
    args = SimpleNamespace(
        phase="validation",
        dataset="insta_market_basket",
        model_role="B",
        smoke=False,
        prerequisite_decision=decision_path,
        contract=contract_path,
        source_revision="f" * 40,
        source_manifest=source_manifest,
        train_result=admitted,
    )
    assert validate_prerequisite_decision(args)["decision"] == "accepted_for_B_validation"

    unadmitted = tmp_path / "unadmitted.json"
    unadmitted.write_text("different")
    args.train_result = unadmitted
    with pytest.raises(ValueError, match="outside the admitted train gate"):
        validate_prerequisite_decision(args)


def test_B_validation_smoke_still_requires_the_B_train_performance_gate():
    args = SimpleNamespace(
        phase="validation",
        dataset="insta_market_basket",
        model_role="B",
        smoke=True,
        prerequisite_decision=None,
    )
    with pytest.raises(ValueError, match="required"):
        validate_prerequisite_decision(args)


def test_source_manifest_requires_an_empty_sample_data_runtime_sentinel(
    monkeypatch, tmp_path
):
    import paper.scripts.run_frozen_raw_affine_calibration as runner

    required = {
        "paper/contracts/frozen_raw_affine_calibration_v1.json",
        "paper/scripts/frozen_raw_affine_calibration.py",
        "paper/scripts/build_frozen_raw_affine_source_manifest.py",
        "paper/scripts/control_frozen_raw_affine_5080.py",
        "paper/scripts/run_frozen_raw_affine_calibration.py",
        "paper/scripts/summarize_frozen_raw_affine_calibration.py",
        "paper/scripts/count_aware_tpp_backbone/core.py",
        "paper/scripts/run_hard_lmm_time_head_refit.py",
        "paper/scripts/run_intermittent_log_backbone_control.py",
        "models/TPPs/CountAwareTPP.py",
        "models/TPPs/CountAwareFactory.py",
        "data_loader/event_seq_data_module.py",
        "simple_lab_test/search/common/runner.py",
    }
    files = {}
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        files[relative] = sha256_file(path)
    manifest = {
        "schema": "frozen_raw_affine_source_manifest_v1",
        "source_revision": "f" * 40,
        "git_tree": "e" * 40,
        "file_count": len(files),
        "files": files,
        "runtime_empty_directories": ["sample_data"],
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="missing"):
        verify_source_manifest(manifest_path, source_revision="f" * 40)
    (tmp_path / "sample_data").mkdir()
    audit = verify_source_manifest(manifest_path, source_revision="f" * 40)
    assert audit["runtime_empty_directories_verified"] == ["sample_data"]
    (tmp_path / "sample_data/unexpected.txt").write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        verify_source_manifest(manifest_path, source_revision="f" * 40)


def test_train_run_writes_success_status_to_the_prepared_context_path(
    monkeypatch, tmp_path
):
    import paper.scripts.run_frozen_raw_affine_calibration as runner

    class FrozenModel:
        def cpu(self):
            return self

    output_dir = tmp_path / "output"
    status_path = output_dir / "status.json"
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        phase="train",
        train_result=None,
        prerequisite_decision=None,
        data=tmp_path / "data.parquet",
        output_dir=output_dir,
        contract=contract_path,
        source_revision="f" * 40,
        dataset="insta_market_basket",
        model_role="B",
        smoke=True,
    )

    def fake_prepare(_args):
        output_dir.mkdir()
        return {
            "status_path": status_path,
            "contract": {},
            "dataset_spec": {},
            "source_spec": {},
            "checkpoint_file_sha": "1" * 64,
            "state_before": "2" * 64,
            "data_sha256": "3" * 64,
            "split_manifest_sha256": "4" * 64,
            "source_manifest_audit": {"file_sha256": "5" * 64},
            "payload": {"backbone": "titantpp"},
            "runtime_audit": {},
            "device": torch.device("cpu"),
            "model": FrozenModel(),
        }

    monkeypatch.setattr(runner, "validate_prerequisite_decision", lambda _args: None)
    monkeypatch.setattr(runner, "prepare_execution", fake_prepare)
    monkeypatch.setattr(runner, "load_train_only_frame", lambda _path: object())
    monkeypatch.setattr(
        runner,
        "extract_split",
        lambda _context, _args, split: ({}, {"target_count": 1}, {"target_count": 1}),
    )
    monkeypatch.setattr(
        runner,
        "analyze_train",
        lambda *positional, **keyword: {"gate_passed": False},
    )
    monkeypatch.setattr(runner, "persist_calibration_state", lambda **keyword: {})
    monkeypatch.setattr(runner, "audit_frozen_state", lambda _context: ("2" * 64, 0))

    result = runner.run(args)
    assert result["status"] == "success"
    assert json.loads(status_path.read_text())["status"] == "success"


def test_smoke_fitted_state_round_trips_without_becoming_a_full_gate(tmp_path):
    cache = synthetic_cache()
    train_audit = analyze_train(cache, contract(), partial_smoke=True)
    assert not train_audit["gate_passed"]
    assert train_audit["calibration_parameters"] is not None
    population = {
        "target_count": len(cache["quantity"]),
        "target_identity_sha256": "1" * 64,
        "target_quantity_sha256": "2" * 64,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("contract")
    state_reference = persist_calibration_state(
        output_dir=tmp_path,
        train=cache,
        train_audit=train_audit,
        population=population,
        contract_sha256=sha256_file(contract_path),
        source_revision="4" * 40,
        dataset="insta_market_basket",
        model_role="B",
        checkpoint_file_sha256="5" * 64,
        checkpoint_state_sha256="6" * 64,
        data_sha256="7" * 64,
        split_manifest_sha256="8" * 64,
        source_manifest_sha256="9" * 64,
    )
    train_audit["calibration_state"] = state_reference
    payload = {
        "status": "success",
        "phase": "train",
        "dataset": "insta_market_basket",
        "model_role": "B",
        "source_revision": "4" * 40,
        "contract_sha256": sha256_file(contract_path),
        "source_checkpoint_file_sha256": "5" * 64,
        "source_manifest": {"file_sha256": "9" * 64},
        "input_digests": {
            "data_sha256": "7" * 64,
            "split_manifest_sha256": "8" * 64,
        },
        "source_model_state_unchanged": True,
        "source_gradients_absent": True,
        "target_population": {"train": population},
        "cache": {"train": {"file_sha256": "a" * 64}},
        "train_audit": train_audit,
    }
    train_result = tmp_path / "train_result.json"
    train_result.write_text(json.dumps(payload))
    args = SimpleNamespace(
        train_result=train_result,
        dataset="insta_market_basket",
        model_role="B",
        source_revision="4" * 40,
        contract=contract_path,
        smoke=True,
    )
    context = {
        "checkpoint_file_sha": "5" * 64,
        "state_before": "6" * 64,
        "source_manifest_audit": {"file_sha256": "9" * 64},
        "data_sha256": "7" * 64,
        "split_manifest_sha256": "8" * 64,
    }
    restored = validate_train_result(payload, context, args)
    assert not restored["gate_passed"]
    state = json.loads((tmp_path / "calibration_state.json").read_text())
    assert state["calibration_activated"]
    assert state["activation_reason"] == "smoke_contract_exercise"
