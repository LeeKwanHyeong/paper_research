from __future__ import annotations

import copy
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import paper.scripts.run_matched_frozen_lognormal_duration as matched_runner
import paper.scripts.run_matched_frozen_lognormal_duration_5090 as controller
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    state_partition_sha256,
)
from paper.scripts.run_matched_frozen_lognormal_duration_5090 import (
    DATASETS,
    EXPECTED_CONTRACT_SHA256,
    MISSING_ROW,
    MODEL_ROLES,
    NEW_EXECUTION_ORDER,
    TIME_KEYS,
    audit_run,
    build_comparison,
    gpu_preflight,
    runner_command,
    sha256_file,
    validate_contract,
    verify_b_reuse,
    verify_inputs,
    verify_source,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


CONTRACT_PATH = ROOT / "paper/contracts/matched_frozen_lognormal_duration_v1.json"


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_checked_in_contract_pins_order_parity_and_known_missing_row():
    assert sha256_file(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256
    contract = load_contract()
    datasets = validate_contract(contract)
    assert tuple(datasets) == DATASETS
    assert tuple(
        (item["dataset"], item["model_role"])
        for item in contract["execution_order"]
    ) == NEW_EXECUTION_ORDER
    assert all(item["phase_order"] == ["e1", "full"] for item in contract["execution_order"])
    assert all(role != "B" for _, role in NEW_EXECUTION_ORDER)
    assert datasets[MISSING_ROW[0]]["sources"][MISSING_ROW[1]]["available"] is False
    assert contract["comparison_policy"]["same_duration_head_for_every_role"] is True
    assert contract["comparison_policy"]["same_optimization_for_every_new_fit"] is True
    assert contract["scope"]["held_out_test"] is False


def test_contract_rejects_policy_order_missing_row_and_quantity_tolerance_drift():
    contract = load_contract()

    weakened = copy.deepcopy(contract)
    weakened["comparison_policy"]["same_duration_head_for_every_role"] = False
    with pytest.raises(ValueError, match="Comparison policy drift"):
        validate_contract(weakened)

    reordered = copy.deepcopy(contract)
    reordered["execution_order"][0], reordered["execution_order"][1] = (
        reordered["execution_order"][1], reordered["execution_order"][0]
    )
    with pytest.raises(ValueError, match="Execution order drift"):
        validate_contract(reordered)

    silently_admitted = copy.deepcopy(contract)
    silently_admitted["datasets"][0]["sources"]["rmtpp"]["available"] = True
    with pytest.raises(ValueError, match="unexpectedly marked available"):
        validate_contract(silently_admitted)

    quantity_drift = copy.deepcopy(contract)
    quantity_drift["identity_and_stability"]["quantity_mae_and_rmse_required"] = "approximate"
    with pytest.raises(ValueError, match="quantity_mae_and_rmse_required"):
        validate_contract(quantity_drift)


def _command_row(tmp_path: Path, dataset: str, role: str) -> dict:
    return {
        "dataset": dataset,
        "data_path": f"data/{dataset}.parquet",
        "split_manifest_path": f"data/{dataset}.json",
        "sources": {
            role: {
                "available": True,
                "checkpoint_path": f"checkpoints/{dataset}/{role}.pt",
            }
        },
    }


def test_runner_commands_are_e1_then_full_for_exact_eight_rows_and_never_B(tmp_path):
    observed = []
    for dataset, role in NEW_EXECUTION_ORDER:
        row = _command_row(tmp_path, dataset, role)
        e1_output = tmp_path / "out" / "e1" / dataset / role
        full_output = tmp_path / "out" / "full" / dataset / role
        e1 = runner_command(
            python="python",
            project=tmp_path,
            input_root=tmp_path / "inputs",
            output=e1_output,
            revision="a" * 40,
            row=row,
            model_role=role,
            phase="e1",
        )
        full = runner_command(
            python="python",
            project=tmp_path,
            input_root=tmp_path / "inputs",
            output=full_output,
            revision="a" * 40,
            row=row,
            model_role=role,
            phase="full",
        )
        assert e1[e1.index("--model-role") + 1] == role
        assert e1[e1.index("--max-epochs") + 1] == "1"
        assert "--max-train-batches" not in e1
        assert "--max-validation-batches" not in e1
        assert full[full.index("--feature-cache-dir") + 1] == str(
            tmp_path / "out" / "e1" / dataset / role / "cache"
        )
        assert "--allow-partial-contract" in e1
        assert "--allow-partial-contract" not in full
        observed.append((dataset, role, "e1"))
        observed.append((dataset, role, "full"))
    assert observed == [
        (dataset, role, phase)
        for dataset, role in NEW_EXECUTION_ORDER
        for phase in ("e1", "full")
    ]
    assert all(role != "B" for _, role, _ in observed)

    with pytest.raises(ValueError, match="Unscheduled row"):
        runner_command(
            python="python", project=tmp_path, input_root=tmp_path,
            output=tmp_path / "out", revision="a" * 40,
            row=_command_row(tmp_path, DATASETS[0], "B"), model_role="B", phase="e1",
        )


def test_deterministic_env_routes_checkpoint_tests_to_input_root(tmp_path):
    project = tmp_path / "source"
    input_root = tmp_path / "inputs"
    environment = controller.deterministic_env(project, input_root=input_root)

    assert environment["PYTHONPATH"] == str(project)
    assert environment["MATCHED_FROZEN_LOGNORMAL_INPUT_ROOT"] == str(input_root)
    assert environment["MATCHED_FROZEN_LOGNORMAL_REQUIRE_CUDA"] == "1"


def test_gpu_preflight_requires_one_idle_5090_and_inactive_gdm(monkeypatch):
    commands: list[tuple[str, ...]] = []

    def idle(command, *, allowed=(0,)):
        commands.append(tuple(command))
        if "--query-gpu=name,memory.free" in command:
            return "NVIDIA GeForce RTX 5090, 32000"
        if "--query-compute-apps=pid" in command:
            return ""
        if command == ["systemctl", "is-active", "gdm"]:
            return "inactive"
        raise AssertionError(command)

    monkeypatch.setattr(controller, "command_output", idle)
    result = gpu_preflight()
    assert result["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert result["compute_processes"] == []
    assert all(command[0] != "pgrep" for command in commands)

    def busy(command, *, allowed=(0,)):
        if "--query-gpu=name,memory.free" in command:
            return "NVIDIA GeForce RTX 5090, 32000"
        if "--query-compute-apps=pid" in command:
            return "123"
        return "inactive"

    monkeypatch.setattr(controller, "command_output", busy)
    with pytest.raises(ValueError, match="compute process"):
        gpu_preflight()


def test_source_manifest_preflight_rejects_unmanifested_python(tmp_path, monkeypatch):
    revision = "a" * 40
    critical = (
        controller.CONTRACT_REL,
        controller.RUNNER_REL,
        controller.TEST_REL,
        controller.BASE_RUNNER_TEST_REL,
        controller.MODEL_TEST_REL,
        controller.CONTROLLER_TEST_REL,
        Path("paper/scripts/run_matched_frozen_lognormal_duration_5090.py"),
    )
    files = {}
    for index, relative in enumerate(critical):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# pinned {index}\n", encoding="utf-8")
        files[str(relative)] = sha256_file(path)
    _write_json(
        tmp_path / "source_manifest.json",
        {
            "source_revision": revision,
            "held_out_test_evaluated": False,
            "files": files,
        },
    )
    monkeypatch.setattr(controller, "PROJECT_ROOT", ROOT)
    assert verify_source(tmp_path, revision)["source_revision"] == revision
    rogue = tmp_path / "rogue.py"
    rogue.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Python source is not manifested"):
        verify_source(tmp_path, revision)


def _synthetic_input_rows(tmp_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    scheduled = set(NEW_EXECUTION_ORDER)
    for dataset in DATASETS:
        data_rel = Path("data") / f"{dataset}.parquet"
        split_rel = Path("data") / f"{dataset}.json"
        data = ("data-" + dataset).encode()
        split = ("split-" + dataset).encode()
        (tmp_path / data_rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / data_rel).write_bytes(data)
        (tmp_path / split_rel).write_bytes(split)
        sources = {}
        for role in MODEL_ROLES:
            if (dataset, role) in scheduled:
                checkpoint_rel = Path("checkpoints") / dataset / f"{role}.pt"
                checkpoint_path = tmp_path / checkpoint_rel
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                state = {"weight": torch.tensor([len(dataset), len(role)], dtype=torch.float32)}
                torch.save({"model_state_dict": state}, checkpoint_path)
                sources[role] = {
                    "available": True,
                    "checkpoint_path": str(checkpoint_rel),
                    "checkpoint_file_sha256": sha256_file(checkpoint_path),
                    "checkpoint_state_sha256": canonical_state_dict_sha256(state),
                }
            elif role == "B":
                sources[role] = {"available": True}
            else:
                sources[role] = {"available": False}
        rows[dataset] = {
            "dataset": dataset,
            "data_path": str(data_rel),
            "data_sha256": _sha_bytes(data),
            "split_manifest_path": str(split_rel),
            "split_manifest_sha256": _sha_bytes(split),
            "sources": sources,
        }
    return rows


def test_input_preflight_verifies_exactly_eight_checkpoint_files_and_states(tmp_path, monkeypatch):
    rows = _synthetic_input_rows(tmp_path)
    calls = []
    monkeypatch.setattr(
        matched_runner,
        "validate_source_checkpoint",
        lambda payload, *, source_spec: calls.append(source_spec),
    )
    result = verify_inputs(tmp_path, rows)
    assert result["verified_checkpoint_count"] == 8
    assert [(item["dataset"], item["model_role"]) for item in result["verified_checkpoints"]] == list(NEW_EXECUTION_ORDER)
    assert len(calls) == 8
    first_dataset, first_role = NEW_EXECUTION_ORDER[0]
    checkpoint = tmp_path / rows[first_dataset]["sources"][first_role]["checkpoint_path"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checkpoint checksum drift"):
        verify_inputs(tmp_path, rows)


def _make_b_reuse_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    input_root = tmp_path / "inputs"
    result_root = tmp_path / "B-result"
    datasets = {}
    decision_rows = []
    for index, dataset in enumerate(DATASETS):
        source_state = {
            "encoder.weight": torch.tensor([[1.0 + index]], dtype=torch.float32),
            "v_t.weight": torch.zeros((1, 64), dtype=torch.float32),
            "b_t": torch.zeros((1,), dtype=torch.float32),
            "w_raw": torch.zeros((1,), dtype=torch.float32),
        }
        selected_state = {
            **{key: value.clone() for key, value in source_state.items()},
            "time_scale_weight.weight": torch.ones((1, 64), dtype=torch.float32),
        }
        selected_state["b_t"] = torch.tensor([0.1 + index], dtype=torch.float32)
        source_rel = Path("source") / dataset / "B.pt"
        source_path = input_root / source_rel
        source_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": source_state}, source_path)
        selected_rel = Path("full") / dataset / "best_validation_proper_time_nll_model.pt"
        selected_path = result_root / selected_rel
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_sha = canonical_state_dict_sha256(selected_state)
        torch.save(
            {
                "model_state_dict": selected_state,
                "model_state_sha256": selected_sha,
                "source_state_sha256": canonical_state_dict_sha256(source_state),
                "time_head_mode": "heteroscedastic_lognormal_duration",
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "legacy_nll_compared": False,
            },
            selected_path,
        )
        source_non_time = state_partition_sha256(source_state, time_head=False)
        nll = 0.5 + index
        summary = {
            "contract_id": "hard_lmm_frozen_lognormal_duration_v1",
            "status": "success",
            "dataset": dataset,
            "seed": 42,
            "time_head_mode": "heteroscedastic_lognormal_duration",
            "calculation_dtype": "float64",
            "trainable_parameter_names": list(TIME_KEYS),
            "trainable_parameter_count": 130,
            "encoder_mode_during_refit": "eval",
            "hidden_state_gradient": "detached_cache",
            "selection": "earliest_strict_finite_minimum_validation_proper_time_nll",
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "legacy_nll_compared": False,
            "qualified_full_data": True,
            "quantity_prediction_bitwise_identical": True,
            "source_checkpoint_sha256": sha256_file(source_path),
            "source_state_sha256": canonical_state_dict_sha256(source_state),
            "source_non_time_state_sha256": source_non_time,
            "selected_non_time_state_sha256": source_non_time,
            "selected_state_sha256": selected_sha,
            "source_quantity_prediction_sha256": "a" * 64,
            "selected_quantity_prediction_sha256": "a" * 64,
            "quantity_metrics": {"mae": 1.0, "rmse": 2.0, "body_mae": 1.0, "gt_p99_mae": 3.0},
            "best_epoch": 1,
            "best_validation_proper_time_nll": nll,
            "epoch_zero_validation_proper_time_nll": nll + 0.2,
            "history": [
                {"epoch": 0, "val_proper_time_nll": nll + 0.2},
                {"epoch": 1, "val_proper_time_nll": nll},
            ],
            "resume_identity": {
                "seed": 42,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "cached_state_batch_size": 8192,
                "minimum_epochs": 20,
                "early_stopping_patience": 20,
                "trainable_parameter_names": list(TIME_KEYS),
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "legacy_nll_compared": False,
            },
        }
        summary_rel = Path("full") / dataset / "summary.json"
        summary_path = result_root / summary_rel
        _write_json(summary_path, summary)
        b = {
            "source_checkpoint_path": str(source_rel),
            "source_checkpoint_file_sha256": sha256_file(source_path),
            "source_checkpoint_state_sha256": canonical_state_dict_sha256(source_state),
            "matched_summary_path": str(summary_rel),
            "matched_summary_file_sha256": sha256_file(summary_path),
            "matched_selected_checkpoint_path": str(selected_rel),
            "matched_selected_checkpoint_file_sha256": sha256_file(selected_path),
            "matched_selected_state_sha256": selected_sha,
            "best_epoch": 1,
            "best_validation_proper_time_nll": nll,
            "quantity_prediction_bitwise_identical": True,
        }
        datasets[dataset] = {"dataset": dataset, "sources": {"B": b}}
        decision_rows.append({"dataset": dataset})

    manifest_rel = Path("control/source_manifest.json")
    manifest_path = result_root / manifest_rel
    _write_json(
        manifest_path,
        {
            "source_revision": "c04d32b0ba80a463097d91c4128ea966f22ec660",
            "contract_sha256": "5316ccd3d3c041f80393a4357308e5bbaeddb08a2c5873e81d5b2d7b2026157b",
            "held_out_test_evaluated": False,
        },
    )
    decision = {
        "status": "complete",
        "candidate_acceptance": "accepted",
        "all_runtime_and_audit_contracts_passed": True,
        "quantity_exactly_preserved_on_all_datasets": True,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
        "cross_dataset_nll_aggregated": False,
        "additional_seeds_executed": False,
        "datasets": decision_rows,
    }
    decision_path = result_root / "decision.json"
    _write_json(decision_path, decision)
    contract = {
        "optimization": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "cached_state_batch_size": 8192,
            "minimum_epochs": 20,
            "early_stopping_patience": 20,
        },
        "B_reuse": {
            "required_source_revision": "c04d32b0ba80a463097d91c4128ea966f22ec660",
            "artifact_contract_id": "hard_lmm_frozen_lognormal_duration_v1",
            "source_manifest_path": str(manifest_rel),
            "source_manifest_file_sha256": sha256_file(manifest_path),
            "decision_path": "decision.json",
            "decision_file_sha256": sha256_file(decision_path),
            "required_decision": "accepted",
        },
    }
    return input_root, result_root, contract, datasets


def test_b_reuse_authenticates_manifest_source_and_selected_non_time_state(tmp_path):
    input_root, result_root, contract, datasets = _make_b_reuse_fixture(tmp_path)
    result = verify_b_reuse(
        result_root, input_root=input_root, contract=contract, datasets=datasets
    )
    assert result["status"] == "verified"
    assert len(result["rows"]) == 3
    assert result["same_duration_head_policy_verified"] is True
    assert all(row["quantity_prediction_bitwise_identical"] for row in result["rows"])


def test_b_reuse_fails_closed_on_summary_and_non_time_tensor_tampering(tmp_path):
    input_root, result_root, contract, datasets = _make_b_reuse_fixture(tmp_path)
    dataset = DATASETS[0]
    summary_path = result_root / datasets[dataset]["sources"]["B"]["matched_summary_path"]
    summary_path.write_text(summary_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="summary checksum drift"):
        verify_b_reuse(
            result_root, input_root=input_root, contract=contract, datasets=datasets
        )

    input_root, result_root, contract, datasets = _make_b_reuse_fixture(tmp_path / "second")
    dataset = DATASETS[1]
    selected_path = result_root / datasets[dataset]["sources"]["B"]["matched_selected_checkpoint_path"]
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    payload["model_state_dict"]["encoder.weight"] += 1
    payload["model_state_sha256"] = canonical_state_dict_sha256(payload["model_state_dict"])
    torch.save(payload, selected_path)
    datasets[dataset]["sources"]["B"]["matched_selected_checkpoint_file_sha256"] = sha256_file(selected_path)
    datasets[dataset]["sources"]["B"]["matched_selected_state_sha256"] = payload["model_state_sha256"]
    summary_path = result_root / datasets[dataset]["sources"]["B"]["matched_summary_path"]
    summary = json.loads(summary_path.read_text())
    summary["selected_state_sha256"] = payload["model_state_sha256"]
    _write_json(summary_path, summary)
    datasets[dataset]["sources"]["B"]["matched_summary_file_sha256"] = sha256_file(summary_path)
    with pytest.raises(ValueError, match="non-time tensor drift"):
        verify_b_reuse(
            result_root, input_root=input_root, contract=contract, datasets=datasets
        )


def test_execute_verify_only_checks_inputs_and_b_without_gpu_or_subprocess(tmp_path, monkeypatch):
    project = tmp_path / "source"
    input_root = tmp_path / "inputs"
    b_root = tmp_path / "B"
    for path in (project / controller.CONTRACT_REL, input_root, b_root):
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(controller, "PROJECT_ROOT", project)
    original_sha = controller.sha256_file
    monkeypatch.setattr(
        controller,
        "sha256_file",
        lambda path: EXPECTED_CONTRACT_SHA256
        if Path(path) == project / controller.CONTRACT_REL
        else original_sha(Path(path)),
    )
    monkeypatch.setattr(controller, "validate_contract", lambda contract: {"rows": {}})
    monkeypatch.setattr(
        controller,
        "verify_source",
        lambda root, revision: {"source_revision": revision, "files": {}},
    )
    monkeypatch.setattr(
        controller,
        "verify_inputs",
        lambda root, datasets: {
            "verified_checkpoint_count": 8,
            "missing_row": {
                "dataset": MISSING_ROW[0],
                "model_role": MISSING_ROW[1],
                "status": "blocked_missing_checkpoint",
            },
        },
    )
    monkeypatch.setattr(
        controller,
        "verify_b_reuse",
        lambda root, *, input_root, contract, datasets: {
            "decision_sha256": "d" * 64,
            "rows": [{"dataset": dataset} for dataset in DATASETS],
        },
    )
    monkeypatch.setattr(controller, "gpu_preflight", lambda: pytest.fail("GPU preflight ran"))
    monkeypatch.setattr(controller, "run_logged", lambda *a, **k: pytest.fail("subprocess ran"))
    result = controller.execute(
        argparse.Namespace(
            project_root=project,
            input_root=input_root,
            b_result_root=b_root,
            output_root=tmp_path / "out",
            source_revision="a" * 40,
            python=sys.executable,
            verify_only=True,
        )
    )
    assert result["status"] == "verified"
    assert result["verified_checkpoint_count"] == 8
    assert result["B_reused_row_count"] == 3
    assert result["four_way_complete"] is False


def _comparison_records():
    epoch_zero = {
        "intermittent_frozen_5000": 1.2,
        "yellow_trip_hourly": 2.2,
        "insta_market_basket": 3.2,
    }
    full = []
    for index, (dataset, role) in enumerate(NEW_EXECUTION_ORDER):
        full.append(
            {
                "status": "passed",
                "dataset": dataset,
                "model_role": role,
                "best_epoch": 1,
                "epoch_zero_validation_proper_time_nll": epoch_zero[dataset],
                "selected_validation_proper_time_nll": epoch_zero[dataset] - 0.01 * (index + 1),
                "quantity_prediction_bitwise_identical": True,
            }
        )
    b = {
        "rows": [
            {
                "dataset": dataset,
                "model_role": "B",
                "best_epoch": 1,
                "epoch_zero_validation_proper_time_nll": epoch_zero[dataset],
                "selected_validation_proper_time_nll": epoch_zero[dataset] - 0.02,
            }
            for dataset in DATASETS
        ]
    }
    return full, b


def test_final_comparison_is_partial_and_checks_epoch_zero_likelihood_parity():
    full, b = _comparison_records()
    result = build_comparison(full_records=full, b_verification=b)
    assert result["present_audited_row_count"] == 11
    assert result["four_way_complete"] is False
    assert result["four_way_claim_allowed"] is False
    assert result["missing_rows"][0]["model_role"] == "rmtpp"
    assert result["cross_dataset_nll_aggregated"] is False
    assert result["datasets"][0]["available_role_count"] == 3
    assert all(item["epoch_zero_proper_time_nll_matched_within_1e_6"] for item in result["datasets"])

    drifted = copy.deepcopy(full)
    drifted[0]["epoch_zero_validation_proper_time_nll"] += 1e-4
    with pytest.raises(ValueError, match="epoch-zero proper Time NLL drift"):
        build_comparison(full_records=drifted, b_verification=b)

    with pytest.raises(ValueError, match="row order drift"):
        build_comparison(full_records=list(reversed(full)), b_verification=b)


def test_independent_audit_rechecks_source_checkpoint_file_before_artifacts(tmp_path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"pinned")
    expected = sha256_file(checkpoint)
    checkpoint.write_bytes(b"tampered")
    row = {
        "dataset": "intermittent_frozen_5000",
        "sources": {
            "A": {
                "checkpoint_file_sha256": expected,
                "checkpoint_state_sha256": "a" * 64,
            }
        },
    }
    with pytest.raises(ValueError, match="checksum drift during audit"):
        audit_run(
            tmp_path / "output",
            contract={},
            row=row,
            model_role="A",
            phase="e1",
            source_checkpoint=checkpoint,
            source_revision="a" * 40,
        )
