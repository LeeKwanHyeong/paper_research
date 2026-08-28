from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from paper.scripts.compare_count_aware_b012_seed42_screening import (
    SCREENING_DATASETS,
    validate_dataset,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONE_LABELS,
    MODEL_ROLE_EXPERIMENTAL,
    MODEL_ROLE_TITAN_B012_SCREENING,
    TITAN_B012_BACKBONES,
    VARIANT,
)
from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS
from paper.scripts.recover_count_aware_b012_seed42_screening import (
    RECOVERY_RUNS,
    assert_no_held_out_artifacts,
    canonical_run_dir,
    evaluate_gpu_snapshot,
    inspect_shard,
    merge_recovery,
    parse_nvidia_process_table,
    prepare_recovery,
    shard_role_dir,
    shard_run_dir,
    validate_completed_run,
    write_status,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_REVISION = "08e59880cd61cbd27cec40aa04636452b87bebfc"
RECOVERY_REVISION = "1" * 40
RECOVERY_CONTRACT = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titan_b012_screening_recovery1_v1.json"
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def encoder_config(backbone: str) -> dict:
    if backbone == "titantpp":
        return {
            "backbone_contract_id": "B0",
            "memory_mode": "static_hard_lmm",
            "persistent_mem_size": 16,
            "lmm_mem_size": 64,
            "lmm_topk": 4,
        }
    if backbone == "titantpp_titans_mac":
        return {
            "backbone_contract_id": "B1",
            "persistent_mem_size": 16,
            "titans_neural_memory_depth": 2,
            "titans_neural_memory_hidden_expansion": 2,
            "titans_mac_segment_size": 16,
            "titans_scan_backend": "compiled_sequence_cuda",
            "titans_online_update": "surprise_momentum_adaptive_forgetting",
        }
    return {
        "backbone_contract_id": "B2",
        "persistent_mem_size": 16,
        "tpp_gated_memory_size": 64,
        "tpp_gated_topk": 4,
        "tpp_gated_temperature": 1.0,
        "tpp_gated_state_scope": "explicit_per_series_state",
        "tpp_gated_scan_backend": "compiled_sequence_cuda",
    }


def metric_row(backbone: str, order: int, key: str, label: str) -> dict:
    return {
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": VARIANT,
        "seed": 42,
        "stratum_order": order,
        "stratum": key,
        "stratum_label": label,
        "count": 10,
        "share": 0.2,
        "joint_objective": 2.0,
        "time_nll": 1.0,
        "quantity_train_loss": 1.0,
        "log_qty_mse": 1.0,
        "quantity_distribution_nll": 0.0,
        "quantity_location_huber": 0.0,
        "tail_aux_loss": 0.0,
        "quantity_scale_mean": 0.0,
        "qty_mae": 1.0,
        "qty_rmse": 1.5,
        "qty_bias": 0.0,
    }


def quantity_rows(backbone: str) -> list[dict]:
    return [
        metric_row(backbone, 0, "le_p50", "<=p50"),
        metric_row(backbone, 1, "p50_p90", "p50-p90"),
        metric_row(backbone, 2, "p90_p95", "p90-p95"),
        metric_row(backbone, 3, "p95_p99", "p95-p99"),
        metric_row(backbone, 4, "gt_p99", ">p99"),
    ]


def history_rows(backbone: str) -> list[dict]:
    return [metric_row(backbone, 0, "history_all", "all history")]


def launch_payload(
    dataset: str,
    backbones: tuple[str, ...],
    *,
    model_role: str,
    status: str,
) -> dict:
    contract = DATASET_CONTRACTS[dataset]
    payload = {
        "status": status,
        "dataset": dataset,
        "data_sha256": contract["data_sha256"],
        "split_manifest_sha256": contract["split_manifest_sha256"],
        "quantity_variants": [VARIANT],
        "backbones": list(backbones),
        "seeds": [42],
        "expected_run_count": len(backbones),
        "epochs": 300,
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": contract["lookback"],
        "max_seq_len": contract["max_seq_len"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": SOURCE_REVISION,
        "partial_smoke": False,
        "max_series": None,
        "model_role": model_role,
        "interfaces": {
            VARIANT: {
                "mode": "mark_free_count_aware_log_regression",
                "history_features": ["log1p_delta_t", "log1p_raw_quantity"],
                "target": "log1p_raw_quantity",
                "quantity_loss": "mse_on_log1p_quantity",
                "quantity_mark_used": False,
                "quantity_residual_used": False,
                "product_type_used": False,
                "target_quantity_masked_from_history": True,
                "fitted_on": "train",
                "train_target_mean": 1.0,
                "train_target_std": 1.0,
                "time_head": {
                    "mode": "legacy_clamped_rmtpp",
                    "time_scale": 3.0,
                    "time_w_max": 10.0 / 3.0,
                    "time_intercept_limit": 30.0,
                    "time_initial_intercept": 0.0,
                    "time_intercept_transform": "legacy_upper_clamp",
                    "time_wd_safety_limit": 40.0,
                    "time_head_lr_multiplier": 1.0,
                    "time_initial_location": None,
                    "time_initial_scale": None,
                    "time_sigma_floor": 1e-3,
                    "statistics_source_split": "train",
                },
            }
        },
        "time_head": {
            "mode": "legacy_clamped_rmtpp",
            "time_scale": 3.0,
            "time_w_max": 10.0 / 3.0,
            "time_intercept_limit": 30.0,
            "time_initial_intercept": 0.0,
            "time_intercept_transform": "legacy_upper_clamp",
            "time_wd_safety_limit": 40.0,
            "time_head_lr_multiplier": 1.0,
            "time_initial_location": None,
            "time_initial_scale": None,
            "time_sigma_floor": 1e-3,
            "statistics_source_split": "train",
            "density_unit": "legacy_delta_t_clamped_objective",
            "wd_clamp": 10.0,
            "train_time_statistics": {
                "statistics_source_split": "train",
                "target_count": 100,
                "time_scale": 3.0,
                "time_w_max": 10.0 / 3.0,
                "wd_safety_limit": 40.0,
            },
        },
        "early_stopping": {
            "monitor": "validation_joint_objective",
            "formula_by_variant": {
                VARIANT: "time_nll + lambda_log_qty * log1p_quantity_mse"
            },
            "min_epochs": 40,
            "patience": 40,
            "restore": "best_validation_joint_objective",
        },
    }
    if status == "complete":
        payload["completed_run_count"] = len(backbones)
    return payload


def make_completed_run(run_dir: Path, backbone: str) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"weight": torch.tensor([1.0, 2.0])}
    digest = canonical_state_dict_sha256(state)
    history = [
        {"epoch": epoch, "val_joint_objective": 2.0 + epoch / 1000.0}
        for epoch in range(1, 41)
    ]
    checkpoint_path = run_dir / "best_val_joint_objective_model.pt"
    torch.save(
        {
            "selection": "best_validation_joint_objective",
            "backbone": backbone,
            "variant": VARIANT,
            "seed": 42,
            "model_state_dict": state,
            "model_state_sha256": digest,
            "source_revision": SOURCE_REVISION,
            "source_revision_history": [SOURCE_REVISION],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        },
        checkpoint_path,
    )
    summary = {
        "status": "success",
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": VARIANT,
        "seed": 42,
        "epochs": 300,
        "completed_epochs": 40,
        "stopped_early": True,
        "best_epoch": 1,
        "best_val_joint_objective": 2.0,
        "best_val_time_nll": 1.0,
        "best_val_quantity_train_loss": 1.0,
        "best_val_log_qty_mse": 1.0,
        "best_val_quantity_distribution_nll": 0.0,
        "best_val_quantity_location_huber": 0.0,
        "best_val_tail_aux_loss": 0.0,
        "best_val_tail_count": 0,
        "best_val_quantity_scale_mean": 0.0,
        "lambda_tail": 0.0,
        "best_val_qty_mae": 1.0,
        "best_val_qty_rmse": 1.5,
        "parameter_count": 10,
        "source_revision": SOURCE_REVISION,
        "source_revision_history": [SOURCE_REVISION],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_state_sha256": digest,
        "elapsed_seconds": 1.0,
        "encoder_config": encoder_config(backbone),
        "quantity_rows": quantity_rows(backbone),
        "history_rows": history_rows(backbone),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "history.json", {"history": history})
    return summary


def make_partial_checkpoint(run_dir: Path, backbone: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "last_epoch_state.pt"
    torch.save(
        {
            "epoch": 3,
            "backbone": backbone,
            "variant": VARIANT,
            "seed": 42,
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "best_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {"state": {"step": 3}},
            "history": [
                {"epoch": epoch, "val_joint_objective": 1.0}
                for epoch in range(1, 4)
            ],
            "encoder_config": encoder_config(backbone),
            "source_revision": SOURCE_REVISION,
            "source_revision_history": [SOURCE_REVISION],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        },
        path,
    )
    return path


def make_source_artifact(root: Path) -> Path:
    source = root / "failed_source"
    role_dir = source / "intermittent_frozen_5000" / MODEL_ROLE_TITAN_B012_SCREENING
    write_json(
        role_dir / "launch_contract.json",
        launch_payload(
            "intermittent_frozen_5000",
            TITAN_B012_BACKBONES,
            model_role=MODEL_ROLE_TITAN_B012_SCREENING,
            status="running",
        ),
    )
    make_completed_run(
        canonical_run_dir(source, "intermittent_frozen_5000", "titantpp"),
        "titantpp",
    )
    write_json(
        source / "screening_status.json",
        {
            "status": "running",
            "source_revision": SOURCE_REVISION,
            "held_out_test_evaluated": False,
        },
    )
    return source


def test_gpu_preflight_rejects_graphics_and_compute_processes() -> None:
    table = """
    |    0   N/A  N/A       155204      G   /usr/bin/gnome-shell       15637MiB |
    |    0   N/A  N/A       155300    C+G   /usr/bin/python3              50MiB |
    |    0   N/A  N/A       155400      G   /usr/lib/xorg/Xwayland         6MiB |
    """
    processes = parse_nvidia_process_table(table)

    result = evaluate_gpu_snapshot(
        total_mib=16303,
        used_mib=15696,
        free_mib=607,
        processes=processes,
        minimum_free_mib=15000,
        maximum_used_mib=512,
    )

    assert {process["type"] for process in processes} == {"G", "C+G"}
    assert result["passed"] is False
    assert "cuda_compute_process_present" in result["failure_reasons"]
    assert "forbidden_graphics_process_present" in result["failure_reasons"]
    assert evaluate_gpu_snapshot(
        total_mib=16303,
        used_mib=241,
        free_mib=16062,
        processes=[],
        minimum_free_mib=15000,
        maximum_used_mib=512,
    )["passed"] is True


def test_prepare_reuses_only_validated_b0_and_keeps_source_immutable(
    tmp_path: Path,
) -> None:
    source = make_source_artifact(tmp_path)
    output = tmp_path / "recovery1"
    source_checkpoint = canonical_run_dir(
        source, "intermittent_frozen_5000", "titantpp"
    ) / "best_val_joint_objective_model.pt"
    before = source_checkpoint.read_bytes()

    manifest = prepare_recovery(
        source_artifact=source,
        output_root=output,
        source_revision=SOURCE_REVISION,
        recovery_revision=RECOVERY_REVISION,
        contract_path=RECOVERY_CONTRACT,
    )

    assert manifest["training_source_revision"] == SOURCE_REVISION
    assert manifest["recovery_orchestration_revision"] == RECOVERY_REVISION
    assert source_checkpoint.read_bytes() == before
    assert canonical_run_dir(
        output, "intermittent_frozen_5000", "titantpp"
    ).is_dir()
    assert not canonical_run_dir(
        output, "intermittent_frozen_5000", "titantpp_titans_mac"
    ).exists()
    assert inspect_shard(
        output_root=output,
        dataset="intermittent_frozen_5000",
        backbone="titantpp_titans_mac",
        source_revision=SOURCE_REVISION,
    )["action"] == "execute_fresh"
    assert_no_held_out_artifacts(output)


def test_prepare_rejects_revision_drift_and_nonfinite_b0(tmp_path: Path) -> None:
    source = make_source_artifact(tmp_path)
    launch_path = (
        source
        / "intermittent_frozen_5000"
        / MODEL_ROLE_TITAN_B012_SCREENING
        / "launch_contract.json"
    )
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["source_revision"] = "f" * 40
    write_json(launch_path, launch)
    with pytest.raises(ValueError, match="Launch contract mismatch"):
        prepare_recovery(
            source_artifact=source,
            output_root=tmp_path / "revision_drift",
            source_revision=SOURCE_REVISION,
            recovery_revision=RECOVERY_REVISION,
            contract_path=RECOVERY_CONTRACT,
        )

    source = make_source_artifact(tmp_path / "nonfinite")
    run_dir = canonical_run_dir(source, "intermittent_frozen_5000", "titantpp")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["best_val_qty_mae"] = float("nan")
    write_json(run_dir / "summary.json", summary)
    with pytest.raises(ValueError, match="Non-finite"):
        validate_completed_run(
            run_dir,
            dataset="intermittent_frozen_5000",
            backbone="titantpp",
            source_revision=SOURCE_REVISION,
        )


def test_shard_inspection_allows_only_missing_same_revision_resume_or_reuse(
    tmp_path: Path,
) -> None:
    output = tmp_path / "recovery1"
    dataset, backbone = RECOVERY_RUNS[0]
    assert inspect_shard(
        output_root=output,
        dataset=dataset,
        backbone=backbone,
        source_revision=SOURCE_REVISION,
    )["action"] == "execute_fresh"

    role_dir = shard_role_dir(output, dataset, backbone)
    write_json(
        role_dir / "launch_contract.json",
        launch_payload(
            dataset,
            (backbone,),
            model_role=MODEL_ROLE_EXPERIMENTAL,
            status="running",
        ),
    )
    partial_path = make_partial_checkpoint(
        shard_run_dir(output, dataset, backbone), backbone
    )
    assert inspect_shard(
        output_root=output,
        dataset=dataset,
        backbone=backbone,
        source_revision=SOURCE_REVISION,
    )["action"] == "resume_partial"

    partial = torch.load(partial_path, map_location="cpu", weights_only=False)
    partial["source_revision_history"] = ["f" * 40]
    torch.save(partial, partial_path)
    with pytest.raises(ValueError, match="Partial checkpoint contract mismatch"):
        inspect_shard(
            output_root=output,
            dataset=dataset,
            backbone=backbone,
            source_revision=SOURCE_REVISION,
        )

    partial_path.unlink()
    make_completed_run(shard_run_dir(output, dataset, backbone), backbone)
    assert inspect_shard(
        output_root=output,
        dataset=dataset,
        backbone=backbone,
        source_revision=SOURCE_REVISION,
    )["action"] == "finalize_completed"
    launch = launch_payload(
        dataset,
        (backbone,),
        model_role=MODEL_ROLE_EXPERIMENTAL,
        status="complete",
    )
    write_json(role_dir / "launch_contract.json", launch)
    assert inspect_shard(
        output_root=output,
        dataset=dataset,
        backbone=backbone,
        source_revision=SOURCE_REVISION,
    )["action"] == "reuse_completed"


def test_failed_status_is_atomic_and_held_out_artifacts_are_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "recovery1"
    write_json(
        output / "recovery_manifest.json",
        {
            "training_source_revision": SOURCE_REVISION,
            "recovery_orchestration_revision": RECOVERY_REVISION,
        },
    )
    write_status(
        output_root=output,
        state="failed",
        source_revision=SOURCE_REVISION,
        recovery_revision=RECOVERY_REVISION,
        message="preflight failed",
        current_dataset="yellow_trip_hourly",
        current_backbone="titantpp",
        exit_code=17,
    )
    status = json.loads((output / "screening_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["exit_code"] == 17
    assert not (output / ".screening_status.json.tmp").exists()

    write_json(output / "test_summary.json", {"held_out_test_evaluated": True})
    with pytest.raises(ValueError, match="Held-out artifacts are forbidden"):
        assert_no_held_out_artifacts(output)


def test_merge_validates_all_nine_runs_and_builds_comparator_contract(
    tmp_path: Path,
) -> None:
    source = make_source_artifact(tmp_path)
    output = tmp_path / "recovery1"
    prepare_recovery(
        source_artifact=source,
        output_root=output,
        source_revision=SOURCE_REVISION,
        recovery_revision=RECOVERY_REVISION,
        contract_path=RECOVERY_CONTRACT,
    )
    for dataset, backbone in RECOVERY_RUNS:
        role_dir = shard_role_dir(output, dataset, backbone)
        write_json(
            role_dir / "launch_contract.json",
            launch_payload(
                dataset,
                (backbone,),
                model_role=MODEL_ROLE_EXPERIMENTAL,
                status="complete",
            ),
        )
        make_completed_run(shard_run_dir(output, dataset, backbone), backbone)

    result = merge_recovery(
        output_root=output,
        source_revision=SOURCE_REVISION,
        recovery_revision=RECOVERY_REVISION,
    )

    assert result["completed_run_count"] == 9
    for dataset in SCREENING_DATASETS:
        role_dir = output / dataset / MODEL_ROLE_TITAN_B012_SCREENING
        launch = json.loads(
            (role_dir / "launch_contract.json").read_text(encoding="utf-8")
        )
        assert launch["model_role"] == MODEL_ROLE_TITAN_B012_SCREENING
        assert launch["completed_run_count"] == 3
        for backbone in TITAN_B012_BACKBONES:
            assert (
                canonical_run_dir(output, dataset, backbone) / "history.csv"
            ).is_file()
        validate_dataset(output, dataset, source_revision=SOURCE_REVISION)
    manifest = json.loads(
        (output / "recovery_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["validated_runs"]) == 9
    assert {
        row["provenance"] for row in manifest["validated_runs"]
    } == {"failed_artifact_validated_b0", "isolated_recovery_shard"}
    assert_no_held_out_artifacts(output)


def test_recovery_contract_and_launcher_freeze_the_safe_execution_order() -> None:
    contract = json.loads(RECOVERY_CONTRACT.read_text(encoding="utf-8"))
    launcher = (
        PROJECT_ROOT
        / "simple_lab_test/search/scripts/"
        "run_count_aware_b012_seed42_screening_recovery1_20260828.sh"
    ).read_text(encoding="utf-8")

    assert tuple(tuple(item) for item in contract["isolated_run_plan"]) == RECOVERY_RUNS
    assert contract["process_isolation"]["one_backbone_per_python_process"] is True
    assert contract["process_isolation"]["force_rerun"] is False
    assert contract["canonical_result"]["held_out_test"] == "locked"
    assert "set -Eeuo pipefail" in launcher
    assert "trap on_exit EXIT" in launcher
    assert launcher.count("run_isolated_backbone \\\n") == 8
    assert "--force-rerun" not in launcher
    assert "--model-role experimental" in launcher
    assert '--backbones "${backbone}"' in launcher
    assert launcher.index("preflight-gpu") < launcher.index('"${RUNNER}"')
    assert "training_source_revision=" in launcher
    assert "recovery_orchestration_revision=" in launcher
    assert "verify_snapshot_training_files" in launcher
    assert 'source_manifest="${SOURCE_ARTIFACT}/source_manifest.txt"' in launcher
    assert 'rev-parse --is-inside-work-tree' in launcher
    assert 'if [[ "${VERIFY_ONLY}" == "1" ]]' in launcher
