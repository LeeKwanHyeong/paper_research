"""Focused tests for the revised A-versus-B validation audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

import paper.scripts.audit_hard_lmm_raw_rmse_checkpoint_alignment as auditor
from paper.scripts.audit_hard_lmm_raw_rmse_checkpoint_alignment import (
    audit_b_only,
    write_outputs,
)
from simple_lab_test.search.tests.test_compare_hard_lmm_quantile_checkpoint_alignment import (
    B_VARIANT,
    C_VARIANT,
    DATASETS,
    FRESH_ROLE,
    FRESH_STATE,
    FRESH_STATE_DIGEST,
    SOURCE_REVISION,
    _build_case,
    _sha256,
    _write_csv,
    _write_json,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _upgrade_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path, Path, Path, Path, Path, Path]:
    monkeypatch.setattr(auditor, "SCOPE_DATASETS", DATASETS)
    contract, project_root, fresh_root = _build_case(tmp_path)
    contract["contract_id"] = "hard_lmm_quantile_checkpoint_alignment_v1"
    contract["scope"]["screening_seed"] = 42
    contract["scope"].update(
        maximum_epochs=300,
        minimum_epochs=40,
        patience=40,
    )
    contract["arms"] = {
        "B_t0_raw_rmse": {"checkpoint_monitor": "validation_raw_quantity_rmse"}
    }
    contract["execution"] = {
        "rtx5090_seed42_e300": True,
        "rtx5090_seeds52_and62": False,
        "held_out_test": False,
    }
    loader = project_root / "loader.py"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text("# deterministic loader\n", encoding="utf-8")
    contract["reference_evidence"] = {
        "target_population_loader_source_sha256": {
            "loader.py": hashlib.sha256(loader.read_bytes()).hexdigest()
        }
    }

    for dataset_contract in contract["datasets"]:
        dataset = dataset_contract["dataset"]
        data = project_root / "inputs" / f"{dataset}.parquet"
        split = project_root / "inputs" / f"{dataset}.json"
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(f"data:{dataset}".encode())
        split.write_bytes(f"split:{dataset}".encode())
        data_digest = hashlib.sha256(data.read_bytes()).hexdigest()
        split_digest = hashlib.sha256(split.read_bytes()).hexdigest()
        dataset_contract.update(
            data_path=str(data.relative_to(project_root)),
            split_manifest_path=str(split.relative_to(project_root)),
            data_sha256=data_digest,
            split_manifest_sha256=split_digest,
        )

        a_dir = project_root / dataset_contract["joint_t0_reference"]["artifact_path"]
        a_launch_path = a_dir / "launch_contract.json"
        a_launch = _read(a_launch_path)
        a_launch["data_sha256"] = data_digest
        a_launch["split_manifest_sha256"] = split_digest
        _write_json(a_launch_path, a_launch)
        dataset_contract["joint_t0_reference"]["launch_contract_sha256"] = _sha256(a_launch_path)

        artifact = fresh_root / dataset / FRESH_ROLE
        launch_path = artifact / "launch_contract.json"
        launch = _read(launch_path)
        launch.update(
            data_sha256=data_digest,
            split_manifest_sha256=split_digest,
            split_rows={"train": 200, "validation": 100},
        )
        launch["early_stopping"]["comparison"] = "earliest_strict_finite_minimum"
        _write_json(launch_path, launch)

        run_dir = artifact / "runs" / "titantpp" / B_VARIANT / "seed_42"
        summary_path = run_dir / "summary.json"
        summary = _read(summary_path)
        summary.update(
            epochs=300,
            completed_epochs=42,
            stopped_early=True,
            source_revision_history=[SOURCE_REVISION],
            training_device="cuda:0",
            cuda_peak_memory_allocated_bytes=1024,
            cuda_peak_memory_reserved_bytes=2048,
        )
        _write_json(summary_path, summary)

        history_path = run_dir / "history.json"
        history_payload = _read(history_path)
        for row in history_payload["history"]:
            row.update(train_event_count=200, train_all_finite=True)
        template = dict(history_payload["history"][-1])
        for epoch in range(4, 43):
            history_payload["history"].append(
                {
                    **template,
                    "epoch": epoch,
                    "val_qty_rmse": float(template["val_qty_rmse"]) + 1.0,
                }
            )
        _write_json(history_path, history_payload)

        checkpoint_path = run_dir / "best_val_qty_rmse_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint.update(
            source_revision_history=[SOURCE_REVISION],
            best_epoch=2,
            selected_metric_value=summary["selected_metric_value"],
            selection_formula="sqrt(mean((predicted_raw_quantity - raw_quantity)^2))",
            model_state_sha256=FRESH_STATE_DIGEST,
            evaluation_scope="validation_only",
            held_out_test_evaluated=False,
        )
        torch.save(checkpoint, checkpoint_path)
        torch.save(
            {
                "checkpoint_type": "epoch_resume",
                "checkpoint_schema_version": 2,
                "epoch": 42,
                "backbone": "titantpp",
                "variant": B_VARIANT,
                "seed": 42,
                "source_revision": SOURCE_REVISION,
                "source_revision_history": [SOURCE_REVISION],
                "checkpoint_monitor": "validation_raw_quantity_rmse",
                "checkpoint_monitor_history_key": "val_qty_rmse",
                "checkpoint_selection": "best_validation_raw_quantity_rmse",
                "selection_formula": "sqrt(mean((predicted_raw_quantity - raw_quantity)^2))",
                "best_epoch": 2,
                "best_selection_value": summary["selected_metric_value"],
                "initial_state_sha256": "5" * 64,
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "history": history_payload["history"],
                "best_state_sha256": FRESH_STATE_DIGEST,
                "best_state_dict": FRESH_STATE,
                "model_state_sha256": FRESH_STATE_DIGEST,
                "model_state_dict": FRESH_STATE,
                "optimizer_state_dict": {"state": {1: {"step": 3}}},
                "rng_state": {"torch": torch.tensor([1], dtype=torch.uint8)},
                "train_loader_generator_state": torch.tensor([2], dtype=torch.uint8),
            },
            run_dir / "last_epoch_state.pt",
        )

        summaries_path = artifact / "run_summaries.csv"
        with summaries_path.open(newline="", encoding="utf-8") as handle:
            summaries = list(csv.DictReader(handle))
        for row in summaries:
            if row["variant"] == B_VARIANT:
                row.update(
                    epochs="300",
                    completed_epochs="42",
                    stopped_early="True",
                    source_revision_history=str([SOURCE_REVISION]),
                    training_device="cuda:0",
                    cuda_peak_memory_allocated_bytes="1024",
                    cuda_peak_memory_reserved_bytes="2048",
                )
        _write_csv(summaries_path, summaries)

    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    copied_fresh_root = controller_root / "seed42_e300"
    fresh_root.rename(copied_fresh_root)
    fresh_root = copied_fresh_root

    last_artifact = fresh_root / DATASETS[-1] / FRESH_ROLE
    last_launch_path = last_artifact / "launch_contract.json"
    last_launch = _read(last_launch_path)
    last_launch["status"] = "running"
    last_launch.pop("completed_run_count", None)
    _write_json(last_launch_path, last_launch)
    c_dir = last_artifact / "runs" / "titantpp" / C_VARIANT / "seed_42"
    (c_dir / "summary.json").unlink()
    (c_dir / "best_val_qty_rmse_model.pt").unlink()

    status = {
        "status": "failed",
        "phase": "seed42_e300",
        "current_dataset": DATASETS[-1],
        "stage": "failed",
        "error": "InterruptedError('Controller received signal 15')",
        "source_revision": SOURCE_REVISION,
        "updated_at": "2026-09-06T00:00:01+00:00",
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
        "cuda_contract_tests": {
            "test_count": 34,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "xml_sha256": "9" * 64,
        },
        "completed_e1": [
            {
                "dataset": dataset,
                "audit": {
                    "status": "passed",
                    "phase": "e1",
                    "source_revision": SOURCE_REVISION,
                    "held_out_test_evaluated": False,
                },
            }
            for dataset in DATASETS
        ],
        "completed_seed42_e300": [
            {
                "dataset": dataset,
                "audit": {
                    "status": "passed",
                    "phase": "seed42_e300",
                    "source_revision": SOURCE_REVISION,
                    "held_out_test_evaluated": False,
                },
            }
            for dataset in DATASETS[:2]
        ],
    }
    _write_json(controller_root / "status.json", status)

    contract_path = (
        project_root
        / "paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.json"
    )
    _write_json(contract_path, contract)
    manifest_files = {
        "paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.json": _sha256(
            contract_path
        )
    }
    for relative in auditor.CRITICAL_SOURCE_FILES:
        critical = project_root / relative
        critical.parent.mkdir(parents=True, exist_ok=True)
        critical.write_text(f"# {relative}\n", encoding="utf-8")
        manifest_files[relative] = _sha256(critical)
    source_manifest = controller_root / "control/source_manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "source_revision": SOURCE_REVISION,
            "contract": "paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.json",
            "contract_sha256": _sha256(contract_path),
            "files": manifest_files,
            "held_out_test_evaluated": False,
        },
    )

    final_summary = fresh_root / DATASETS[-1] / FRESH_ROLE / "runs" / "titantpp" / B_VARIANT / "seed_42" / "summary.json"
    final_payload = _read(final_summary)
    stop_record = controller_root / "control" / "stop_after_instacart_b.json"
    _write_json(
        stop_record,
        {
            "status": "controller_sigterm_sent_after_instacart_b",
            "controller_pid": 62054,
            "signal": "SIGTERM",
            "b_summary": str(Path("seed42_e300") / DATASETS[-1] / FRESH_ROLE / "runs" / "titantpp" / B_VARIANT / "seed_42" / "summary.json"),
            "b_summary_sha256": _sha256(final_summary),
            "b_best_epoch": final_payload["best_epoch"],
            "b_completed_epochs": final_payload["completed_epochs"],
            "b_raw_rmse": final_payload["best_val_qty_rmse"],
            "reason": "User revised the comparison scope to A versus B only.",
            "recorded_at": "2026-09-06T00:00:00+00:00",
        },
    )
    return (
        contract,
        project_root,
        fresh_root,
        controller_root,
        stop_record,
        contract_path,
        source_manifest,
    )


def test_b_only_audit_passes_and_excludes_c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (
        contract,
        project_root,
        fresh_root,
        controller_root,
        stop_record,
        contract_path,
        source_manifest,
    ) = _upgrade_case(tmp_path, monkeypatch)

    payload = audit_b_only(
        contract=contract,
        contract_path=contract_path,
        project_root=project_root,
        controller_root=controller_root,
        artifact_root=fresh_root,
        stop_record_path=stop_record,
        source_manifest_path=source_manifest,
    )
    output = tmp_path / "output"
    write_outputs(output, payload)

    assert payload["run_audit"]["status"] == "passed"
    assert payload["metrics"]["common_raw_rmse_goal_met"] is True
    assert payload["scope_completion"]["status"] == "complete_for_revised_scope"
    assert payload["scope_completion"]["C"]["included_in_revised_comparison"] is False
    assert payload["scope_completion"]["C"]["presence_by_dataset"][DATASETS[-1]]["partial"] is True
    assert {path.name for path in output.iterdir()} == {
        "a_vs_b_metrics.json",
        "a_vs_b_metrics.md",
        "a_vs_b_metrics.csv",
        "b_only_run_audit.json",
        "b_only_scope_completion.json",
    }


def test_b_only_audit_rejects_non_earliest_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, project_root, fresh_root, controller_root, stop_record, contract_path, source_manifest = _upgrade_case(tmp_path, monkeypatch)
    summary_path = fresh_root / DATASETS[0] / FRESH_ROLE / "runs" / "titantpp" / B_VARIANT / "seed_42" / "summary.json"
    summary = _read(summary_path)
    summary["best_epoch"] = 3
    _write_json(summary_path, summary)

    with pytest.raises(auditor.EvidenceError, match="earliest strict raw-RMSE minimum"):
        audit_b_only(
            contract=contract,
            contract_path=contract_path,
            project_root=project_root,
            controller_root=controller_root,
            artifact_root=fresh_root,
            stop_record_path=stop_record,
            source_manifest_path=source_manifest,
        )


def test_b_only_audit_rejects_truncated_early_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, project_root, fresh_root, controller_root, stop_record, contract_path, source_manifest = _upgrade_case(tmp_path, monkeypatch)
    run_dir = fresh_root / DATASETS[0] / FRESH_ROLE / "runs" / "titantpp" / B_VARIANT / "seed_42"
    summary_path = run_dir / "summary.json"
    summary = _read(summary_path)
    summary["completed_epochs"] = 3
    summary["stopped_early"] = True
    _write_json(summary_path, summary)
    history_path = run_dir / "history.json"
    history_payload = _read(history_path)
    history_payload["history"] = history_payload["history"][:3]
    _write_json(history_path, history_payload)
    last_path = run_dir / "last_epoch_state.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last["epoch"] = 3
    last["history"] = history_payload["history"]
    torch.save(last, last_path)

    with pytest.raises(auditor.EvidenceError, match="frozen minimum epoch"):
        audit_b_only(
            contract=contract,
            contract_path=contract_path,
            project_root=project_root,
            controller_root=controller_root,
            artifact_root=fresh_root,
            stop_record_path=stop_record,
            source_manifest_path=source_manifest,
        )


def test_b_only_audit_rejects_stop_record_hash_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, project_root, fresh_root, controller_root, stop_record, contract_path, source_manifest = _upgrade_case(tmp_path, monkeypatch)
    record = _read(stop_record)
    record["b_summary_sha256"] = "0" * 64
    _write_json(stop_record, record)

    with pytest.raises(auditor.EvidenceError, match="Stop record B summary SHA-256 mismatch"):
        audit_b_only(
            contract=contract,
            contract_path=contract_path,
            project_root=project_root,
            controller_root=controller_root,
            artifact_root=fresh_root,
            stop_record_path=stop_record,
            source_manifest_path=source_manifest,
        )


@pytest.mark.parametrize(
    "relative",
    (
        Path("nested/heldout/cache.bin"),
        Path("nested/model_test_metrics.parquet"),
    ),
)
def test_b_only_audit_rejects_held_out_evidence_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: Path,
) -> None:
    contract, project_root, fresh_root, controller_root, stop_record, contract_path, source_manifest = _upgrade_case(tmp_path, monkeypatch)
    forbidden = fresh_root / DATASETS[0] / FRESH_ROLE / relative
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_bytes(b"forbidden")

    with pytest.raises(auditor.EvidenceError, match="held-out artifacts"):
        audit_b_only(
            contract=contract,
            contract_path=contract_path,
            project_root=project_root,
            controller_root=controller_root,
            artifact_root=fresh_root,
            stop_record_path=stop_record,
            source_manifest_path=source_manifest,
        )


def test_b_only_audit_rejects_source_manifest_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, project_root, fresh_root, controller_root, stop_record, contract_path, source_manifest = _upgrade_case(tmp_path, monkeypatch)
    manifest = _read(source_manifest)
    manifest["files"][auditor.CRITICAL_SOURCE_FILES[0]] = "0" * 64
    _write_json(source_manifest, manifest)

    with pytest.raises(auditor.EvidenceError, match="critical file mismatch"):
        audit_b_only(
            contract=contract,
            contract_path=contract_path,
            project_root=project_root,
            controller_root=controller_root,
            artifact_root=fresh_root,
            stop_record_path=stop_record,
            source_manifest_path=source_manifest,
        )


def test_b_only_audit_rejects_incomplete_resume_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, project_root, fresh_root, controller_root, stop_record, contract_path, source_manifest = _upgrade_case(tmp_path, monkeypatch)
    last_path = fresh_root / DATASETS[0] / FRESH_ROLE / "runs" / "titantpp" / B_VARIANT / "seed_42/last_epoch_state.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last.pop("train_loader_generator_state")
    torch.save(last, last_path)

    with pytest.raises(auditor.EvidenceError, match="loader RNG state is missing"):
        audit_b_only(
            contract=contract,
            contract_path=contract_path,
            project_root=project_root,
            controller_root=controller_root,
            artifact_root=fresh_root,
            stop_record_path=stop_record,
            source_manifest_path=source_manifest,
        )
