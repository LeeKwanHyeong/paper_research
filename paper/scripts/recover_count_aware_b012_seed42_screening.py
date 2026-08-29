#!/usr/bin/env python3
"""Recover B0/B1/B2 seed-42 screening without changing training code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.compare_count_aware_b012_seed42_screening import (
    SCREENING_DATASETS,
    validate_encoder_contract,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLE_EXPERIMENTAL,
    MODEL_ROLE_TITAN_B012_SCREENING,
    TITAN_B012_BACKBONES,
    VARIANT,
)
from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS
from paper.scripts.count_aware_tpp_backbone.reporting import (
    summarize_breakdowns,
    write_csv,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "count_aware_titan_b012_screening_recovery1_v1"
SHARD_5090_CONTRACT_ID = "count_aware_titan_b012_screening_shard5090_v1"
RECOVERY_MODEL_ROLE = MODEL_ROLE_TITAN_B012_SCREENING
REUSED_RUN = ("intermittent_frozen_5000", "titantpp")
RECOVERY_RUNS = (
    ("intermittent_frozen_5000", "titantpp_titans_mac"),
    ("intermittent_frozen_5000", "titantpp_tpp_gated_memory"),
    ("yellow_trip_hourly", "titantpp"),
    ("yellow_trip_hourly", "titantpp_titans_mac"),
    ("yellow_trip_hourly", "titantpp_tpp_gated_memory"),
    ("raf_spare_parts", "titantpp"),
    ("raf_spare_parts", "titantpp_titans_mac"),
    ("raf_spare_parts", "titantpp_tpp_gated_memory"),
)
SHARD_5090_RUNS = RECOVERY_RUNS[2:]
DEFAULT_FORBIDDEN_GRAPHICS_NAMES = {"gnome-shell", "xwayland"}
FORBIDDEN_HELD_OUT_NAMES = {
    "held_out_test.json",
    "test_metrics.csv",
    "test_summary.json",
}
PROCESS_ROW_PATTERN = re.compile(
    r"\|\s*\d+\s+\S+\s+\S+\s+(?P<pid>\d+)\s+"
    r"(?P<type>C\+G|[CG])\s+(?P<name>.+?)\s+(?P<memory>\d+)MiB\s*\|"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_forbidden_artifacts(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name in FORBIDDEN_HELD_OUT_NAMES
    )


def assert_no_held_out_artifacts(root: Path) -> None:
    forbidden = find_forbidden_artifacts(root)
    if forbidden:
        raise ValueError(f"Held-out artifacts are forbidden: {forbidden}")


def assert_all_finite(value: Any, *, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite value at {location}: {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_all_finite(child, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            assert_all_finite(child, location=f"{location}[{index}]")


def canonical_run_dir(root: Path, dataset: str, backbone: str) -> Path:
    return (
        root
        / dataset
        / RECOVERY_MODEL_ROLE
        / "runs"
        / backbone
        / VARIANT
        / "seed_42"
    )


def shard_role_dir(root: Path, dataset: str, backbone: str) -> Path:
    return root / "shards" / dataset / backbone / RECOVERY_MODEL_ROLE


def shard_run_dir(root: Path, dataset: str, backbone: str) -> Path:
    return shard_role_dir(root, dataset, backbone) / "runs" / backbone / VARIANT / "seed_42"


def validate_launch_contract(
    path: Path,
    *,
    dataset: str,
    backbones: tuple[str, ...],
    source_revision: str,
    model_role: str,
    allowed_statuses: set[str],
) -> dict[str, Any]:
    payload = load_json(path)
    dataset_contract = DATASET_CONTRACTS[dataset]
    expected = {
        "dataset": dataset,
        "data_sha256": dataset_contract["data_sha256"],
        "split_manifest_sha256": dataset_contract["split_manifest_sha256"],
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
        "lookback_weeks": dataset_contract["lookback"],
        "max_seq_len": dataset_contract["max_seq_len"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": source_revision,
        "partial_smoke": False,
        "max_series": None,
        "model_role": model_role,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if payload.get("status") not in allowed_statuses:
        mismatches["status"] = {
            "expected": sorted(allowed_statuses),
            "observed": payload.get("status"),
        }
    if (
        payload.get("status") == "complete"
        and payload.get("completed_run_count") != len(backbones)
    ):
        mismatches["completed_run_count"] = {
            "expected": len(backbones),
            "observed": payload.get("completed_run_count"),
        }
    time_head = payload.get("time_head", {})
    expected_time_head = {
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
    }
    for key, expected_value in expected_time_head.items():
        if time_head.get(key) != expected_value:
            mismatches[f"time_head.{key}"] = {
                "expected": expected_value,
                "observed": time_head.get(key),
            }
    train_time = time_head.get("train_time_statistics", {})
    expected_train_time = {
        "statistics_source_split": "train",
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "wd_safety_limit": 40.0,
    }
    for key, expected_value in expected_train_time.items():
        if train_time.get(key) != expected_value:
            mismatches[f"time_head.train_time_statistics.{key}"] = {
                "expected": expected_value,
                "observed": train_time.get(key),
            }
    if not isinstance(train_time.get("target_count"), int) or train_time.get(
        "target_count", 0
    ) <= 0:
        mismatches["time_head.train_time_statistics.target_count"] = {
            "expected": "positive integer",
            "observed": train_time.get("target_count"),
        }

    interface = payload.get("interfaces", {}).get(VARIANT, {})
    expected_interface = {
        "mode": "mark_free_count_aware_log_regression",
        "history_features": ["log1p_delta_t", "log1p_raw_quantity"],
        "target": "log1p_raw_quantity",
        "quantity_loss": "mse_on_log1p_quantity",
        "quantity_mark_used": False,
        "quantity_residual_used": False,
        "product_type_used": False,
        "target_quantity_masked_from_history": True,
        "fitted_on": "train",
    }
    for key, expected_value in expected_interface.items():
        if interface.get(key) != expected_value:
            mismatches[f"interfaces.{VARIANT}.{key}"] = {
                "expected": expected_value,
                "observed": interface.get(key),
            }
    interface_time = interface.get("time_head", {})
    for key in (
        "mode",
        "time_scale",
        "time_w_max",
        "time_intercept_limit",
        "time_initial_intercept",
        "time_intercept_transform",
        "time_wd_safety_limit",
        "time_head_lr_multiplier",
        "time_initial_location",
        "time_initial_scale",
        "time_sigma_floor",
        "statistics_source_split",
    ):
        if interface_time.get(key) != expected_time_head[key]:
            mismatches[f"interfaces.{VARIANT}.time_head.{key}"] = {
                "expected": expected_time_head[key],
                "observed": interface_time.get(key),
            }
    early_stopping = payload.get("early_stopping", {})
    for key, expected_value in {
        "monitor": "validation_joint_objective",
        "min_epochs": 40,
        "patience": 40,
        "restore": "best_validation_joint_objective",
    }.items():
        if early_stopping.get(key) != expected_value:
            mismatches[f"early_stopping.{key}"] = {
                "expected": expected_value,
                "observed": early_stopping.get(key),
            }
    expected_formula = "time_nll + lambda_log_qty * log1p_quantity_mse"
    observed_formula = early_stopping.get("formula_by_variant", {}).get(VARIANT)
    if observed_formula != expected_formula:
        mismatches[f"early_stopping.formula_by_variant.{VARIANT}"] = {
            "expected": expected_formula,
            "observed": observed_formula,
        }
    if mismatches:
        raise ValueError(f"Launch contract mismatch at {path}: {mismatches}")
    assert_all_finite(payload, location=str(path))
    return payload


def validate_completed_run(
    run_dir: Path,
    *,
    dataset: str,
    backbone: str,
    source_revision: str,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best_val_joint_objective_model.pt"
    history_path = run_dir / "history.json"
    summary = load_json(summary_path)
    history_payload = load_json(history_path)
    expected = {
        "status": "success",
        "backbone": backbone,
        "variant": VARIANT,
        "seed": 42,
        "epochs": 300,
        "source_revision": source_revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    mismatches = {
        key: {"expected": expected_value, "observed": summary.get(key)}
        for key, expected_value in expected.items()
        if summary.get(key) != expected_value
    }
    completed_epochs = summary.get("completed_epochs")
    best_epoch = summary.get("best_epoch")
    if not isinstance(completed_epochs, int) or not 40 <= completed_epochs <= 300:
        mismatches["completed_epochs"] = {
            "expected": "integer in [40, 300]",
            "observed": completed_epochs,
        }
    if (
        not isinstance(best_epoch, int)
        or not isinstance(completed_epochs, int)
        or not 1 <= best_epoch <= completed_epochs
    ):
        mismatches["best_epoch"] = {
            "expected": "integer in [1, completed_epochs]",
            "observed": best_epoch,
        }
    if not summary.get("quantity_rows") or not summary.get("history_rows"):
        mismatches["validation_breakdowns"] = {
            "expected": "nonempty quantity_rows and history_rows",
            "observed": {
                "quantity_rows": len(summary.get("quantity_rows", [])),
                "history_rows": len(summary.get("history_rows", [])),
            },
        }
    if summary.get("source_revision_history") != [source_revision]:
        mismatches["source_revision_history"] = {
            "expected": [source_revision],
            "observed": summary.get("source_revision_history"),
        }
    if mismatches:
        raise ValueError(f"Run summary mismatch at {summary_path}: {mismatches}")
    assert_all_finite(summary, location=str(summary_path))
    history = history_payload.get("history")
    if (
        not isinstance(history, list)
        or len(history) != completed_epochs
        or not history
        or not isinstance(history[-1], dict)
        or history[-1].get("epoch") != completed_epochs
    ):
        raise ValueError(
            f"History does not end at completed epoch {completed_epochs}: {history_path}"
        )
    assert_all_finite(history, location=str(history_path))
    validate_encoder_contract(backbone, summary)
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_expected = {
        "selection": "best_validation_joint_objective",
        "backbone": backbone,
        "variant": VARIANT,
        "seed": 42,
        "source_revision": source_revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision_history": [source_revision],
    }
    checkpoint_mismatches = {
        key: {"expected": expected_value, "observed": checkpoint.get(key)}
        for key, expected_value in checkpoint_expected.items()
        if checkpoint.get(key) != expected_value
    }
    if checkpoint_mismatches:
        raise ValueError(
            f"Checkpoint contract mismatch at {checkpoint_path}: "
            f"{checkpoint_mismatches}"
        )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    observed_digest = canonical_state_dict_sha256(state_dict)
    expected_digest = summary.get("checkpoint_state_sha256")
    if (
        observed_digest != expected_digest
        or checkpoint.get("model_state_sha256") != expected_digest
    ):
        raise ValueError(
            f"Checkpoint state digest mismatch at {checkpoint_path}: "
            f"summary={expected_digest} checkpoint={checkpoint.get('model_state_sha256')} "
            f"observed={observed_digest}"
        )
    return {
        "dataset": dataset,
        "backbone": backbone,
        "summary": summary,
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "checkpoint_state_sha256": observed_digest,
        "history": history,
        "run_dir": run_dir,
    }


def validate_partial_checkpoint(
    checkpoint_path: Path,
    *,
    backbone: str,
    source_revision: str,
) -> dict[str, Any]:
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    expected = {
        "backbone": backbone,
        "variant": VARIANT,
        "seed": 42,
        "source_revision": source_revision,
        "source_revision_history": [source_revision],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    mismatches = {
        key: {"expected": value, "observed": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    epoch = checkpoint.get("epoch")
    history = checkpoint.get("history")
    if not isinstance(epoch, int) or not 1 <= epoch <= 300:
        mismatches["epoch"] = {
            "expected": "integer in [1, 300]",
            "observed": epoch,
        }
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(history[-1], dict)
        or history[-1].get("epoch") != epoch
    ):
        mismatches["history"] = {
            "expected": "nonempty history ending at checkpoint epoch",
            "observed": history[-1] if isinstance(history, list) and history else history,
        }
    for state_name in ("model_state_dict", "best_state_dict", "optimizer_state_dict"):
        state = checkpoint.get(state_name)
        if not isinstance(state, dict) or not state:
            mismatches[state_name] = {
                "expected": "nonempty mapping",
                "observed": type(state).__name__,
            }
    if mismatches:
        raise ValueError(
            f"Partial checkpoint contract mismatch at {checkpoint_path}: {mismatches}"
        )
    assert_all_finite(history, location=f"{checkpoint_path}.history")
    validate_encoder_contract(
        backbone,
        {"encoder_config": checkpoint.get("encoder_config", {})},
    )
    return {
        "epoch": epoch,
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "source_revision": source_revision,
    }


def inspect_shard(
    *,
    output_root: Path,
    dataset: str,
    backbone: str,
    source_revision: str,
) -> dict[str, Any]:
    if (dataset, backbone) not in RECOVERY_RUNS:
        raise ValueError(f"Run is not part of recovery1: {dataset}/{backbone}")
    role_dir = shard_role_dir(output_root, dataset, backbone)
    assert_no_held_out_artifacts(role_dir)
    if not role_dir.exists():
        return {
            "action": "execute_fresh",
            "dataset": dataset,
            "backbone": backbone,
            "reason": "shard_missing",
        }

    launch_path = role_dir / "launch_contract.json"
    if not launch_path.is_file():
        if any(role_dir.iterdir()):
            raise ValueError(f"Shard exists without a launch contract: {role_dir}")
        return {
            "action": "execute_fresh",
            "dataset": dataset,
            "backbone": backbone,
            "reason": "empty_shard",
        }
    launch = validate_launch_contract(
        launch_path,
        dataset=dataset,
        backbones=(backbone,),
        source_revision=source_revision,
        model_role=MODEL_ROLE_EXPERIMENTAL,
        allowed_statuses={"running", "complete"},
    )
    run_dir = shard_run_dir(output_root, dataset, backbone)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_val_joint_objective_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    if summary_path.exists() or best_path.exists():
        if not summary_path.is_file() or not best_path.is_file():
            raise ValueError(f"Incomplete completed-run pair at {run_dir}")
        record = validate_completed_run(
            run_dir,
            dataset=dataset,
            backbone=backbone,
            source_revision=source_revision,
        )
        return {
            "action": (
                "reuse_completed"
                if launch.get("status") == "complete"
                else "finalize_completed"
            ),
            "dataset": dataset,
            "backbone": backbone,
            "completed_epochs": record["summary"]["completed_epochs"],
            "checkpoint_state_sha256": record["checkpoint_state_sha256"],
        }
    if launch.get("status") == "complete":
        raise ValueError(f"Complete shard has no validated run: {role_dir}")
    if last_path.is_file():
        partial = validate_partial_checkpoint(
            last_path,
            backbone=backbone,
            source_revision=source_revision,
        )
        return {
            "action": "resume_partial",
            "dataset": dataset,
            "backbone": backbone,
            **partial,
        }
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Unrecoverable shard contents without checkpoint: {run_dir}")
    return {
        "action": "execute_fresh",
        "dataset": dataset,
        "backbone": backbone,
        "reason": "launch_record_only",
    }


def _rewrite_checkpoint_path(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path)
    summary["checkpoint_path"] = str(
        (run_dir / "best_val_joint_objective_model.pt").resolve()
    )
    write_json_atomic(summary_path, summary)


def _materialize_history_csv(run_dir: Path, history: list[dict[str, Any]]) -> None:
    write_csv(run_dir / "history.csv", history)


def _copy_run_once(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    _rewrite_checkpoint_path(destination)


def _validate_recovery_contract(path: Path, source_revision: str) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError(f"Unexpected recovery contract id: {contract.get('contract_id')}")
    if contract.get("training_source_revision") != source_revision:
        raise ValueError(
            "Recovery contract training revision mismatch: "
            f"{contract.get('training_source_revision')} != {source_revision}"
        )
    observed_plan = tuple(tuple(item) for item in contract.get("isolated_run_plan", []))
    if observed_plan != RECOVERY_RUNS:
        raise ValueError(f"Recovery run plan drifted: {observed_plan}")
    if contract.get("canonical_result", {}).get("held_out_test") != "locked":
        raise ValueError("Recovery contract must keep held-out test locked")
    return contract


def prepare_recovery(
    *,
    source_artifact: Path,
    output_root: Path,
    source_revision: str,
    recovery_revision: str,
    contract_path: Path,
) -> dict[str, Any]:
    if source_artifact.resolve() == output_root.resolve():
        raise ValueError("Recovery output must differ from the failed source artifact")
    _validate_recovery_contract(contract_path, source_revision)
    assert_no_held_out_artifacts(source_artifact)
    source_role_dir = source_artifact / REUSED_RUN[0] / RECOVERY_MODEL_ROLE
    launch = validate_launch_contract(
        source_role_dir / "launch_contract.json",
        dataset=REUSED_RUN[0],
        backbones=TITAN_B012_BACKBONES,
        source_revision=source_revision,
        model_role=RECOVERY_MODEL_ROLE,
        allowed_statuses={"running", "failed", "complete"},
    )
    source_record = validate_completed_run(
        canonical_run_dir(source_artifact, *REUSED_RUN),
        dataset=REUSED_RUN[0],
        backbone=REUSED_RUN[1],
        source_revision=source_revision,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "recovery_manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        identity = {
            "contract_id": CONTRACT_ID,
            "training_source_revision": source_revision,
            "recovery_orchestration_revision": recovery_revision,
            "source_artifact": str(source_artifact.resolve()),
        }
        mismatches = {
            key: {"expected": value, "observed": existing.get(key)}
            for key, value in identity.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing recovery manifest mismatch: {mismatches}")

    destination = canonical_run_dir(output_root, *REUSED_RUN)
    _copy_run_once(source_record["run_dir"], destination)
    reused_record = validate_completed_run(
        destination,
        dataset=REUSED_RUN[0],
        backbone=REUSED_RUN[1],
        source_revision=source_revision,
    )
    _materialize_history_csv(destination, reused_record["history"])
    provenance_dir = output_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_role_dir / "launch_contract.json",
        provenance_dir / "failed_artifact_launch_contract.json",
    )
    source_status = source_artifact / "screening_status.json"
    if source_status.is_file():
        shutil.copy2(
            source_status,
            provenance_dir / "failed_artifact_screening_status.json",
        )
    manifest = {
        "contract_id": CONTRACT_ID,
        "status": "prepared",
        "prepared_at_utc": utc_now(),
        "training_source_revision": source_revision,
        "recovery_orchestration_revision": recovery_revision,
        "source_artifact": str(source_artifact.resolve()),
        "source_artifact_status": launch.get("status"),
        "source_artifact_is_immutable": True,
        "reused_runs": [
            {
                "dataset": REUSED_RUN[0],
                "backbone": REUSED_RUN[1],
                "variant": VARIANT,
                "seed": 42,
                "completed_epochs": reused_record["summary"]["completed_epochs"],
                "best_epoch": reused_record["summary"]["best_epoch"],
                "summary_sha256_before_copy": source_record["summary_sha256"],
                "checkpoint_file_sha256": reused_record["checkpoint_file_sha256"],
                "checkpoint_state_sha256": reused_record["checkpoint_state_sha256"],
            }
        ],
        "planned_isolated_runs": [
            {"dataset": dataset, "backbone": backbone, "variant": VARIANT, "seed": 42}
            for dataset, backbone in RECOVERY_RUNS
        ],
        "expected_total_run_count": 9,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    write_json_atomic(manifest_path, manifest)
    write_status(
        output_root=output_root,
        state="prepared",
        source_revision=source_revision,
        recovery_revision=recovery_revision,
        message="Validated and copied Intermittent B0; eight isolated runs remain.",
    )
    assert_no_held_out_artifacts(output_root)
    return manifest


def parse_nvidia_process_table(output: str) -> list[dict[str, Any]]:
    processes = []
    for match in PROCESS_ROW_PATTERN.finditer(output):
        processes.append(
            {
                "pid": int(match.group("pid")),
                "type": match.group("type"),
                "name": match.group("name").strip(),
                "memory_mib": int(match.group("memory")),
            }
        )
    return processes


def evaluate_gpu_snapshot(
    *,
    total_mib: int,
    used_mib: int,
    free_mib: int,
    processes: list[dict[str, Any]],
    minimum_free_mib: int,
    maximum_used_mib: int,
    forbidden_graphics_names: set[str] | None = None,
) -> dict[str, Any]:
    reasons = []
    if free_mib < minimum_free_mib:
        reasons.append(f"free_vram_below_{minimum_free_mib}_mib")
    if used_mib > maximum_used_mib:
        reasons.append(f"used_vram_above_{maximum_used_mib}_mib")
    compute_processes = [process for process in processes if "C" in process["type"]]
    if compute_processes:
        reasons.append("cuda_compute_process_present")
    forbidden_names = {
        name.lower()
        for name in (
            forbidden_graphics_names
            if forbidden_graphics_names is not None
            else DEFAULT_FORBIDDEN_GRAPHICS_NAMES
        )
    }
    forbidden_graphics = [
        process
        for process in processes
        if "G" in process["type"]
        and Path(process["name"]).name.lower() in forbidden_names
    ]
    if forbidden_graphics:
        reasons.append("forbidden_graphics_process_present")
    return {
        "passed": not reasons,
        "total_mib": total_mib,
        "used_mib": used_mib,
        "free_mib": free_mib,
        "minimum_free_mib": minimum_free_mib,
        "maximum_used_mib": maximum_used_mib,
        "processes": processes,
        "compute_processes": compute_processes,
        "forbidden_graphics_processes": forbidden_graphics,
        "failure_reasons": reasons,
    }


def query_gpu_snapshot(nvidia_smi: str) -> dict[str, Any]:
    memory = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip().splitlines()
    if len(memory) != 1:
        raise RuntimeError(f"Expected exactly one GPU, observed {len(memory)}")
    values = [int(value.strip()) for value in memory[0].split(",")]
    if len(values) != 3:
        raise RuntimeError(f"Unexpected nvidia-smi memory row: {memory[0]}")
    process_table = subprocess.run(
        [nvidia_smi],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    return {
        "total_mib": values[0],
        "used_mib": values[1],
        "free_mib": values[2],
        "processes": parse_nvidia_process_table(process_table),
    }


def run_gpu_preflight(
    *,
    output_path: Path,
    dataset: str,
    backbone: str,
    nvidia_smi: str,
    minimum_free_mib: int,
    maximum_used_mib: int,
    attempts: int,
    interval_seconds: float,
    forbidden_graphics_names: set[str] | None = None,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("GPU preflight attempts must be positive")
    observations = []
    for attempt in range(1, attempts + 1):
        snapshot = query_gpu_snapshot(nvidia_smi)
        evaluation = evaluate_gpu_snapshot(
            **snapshot,
            minimum_free_mib=minimum_free_mib,
            maximum_used_mib=maximum_used_mib,
            forbidden_graphics_names=forbidden_graphics_names,
        )
        evaluation["attempt"] = attempt
        evaluation["observed_at_utc"] = utc_now()
        observations.append(evaluation)
        if evaluation["passed"]:
            payload = {
                "status": "pass",
                "dataset": dataset,
                "backbone": backbone,
                "observations": observations,
                "held_out_test_evaluated": False,
            }
            write_json_atomic(output_path, payload)
            return payload
        if attempt < attempts:
            time.sleep(interval_seconds)
    payload = {
        "status": "fail",
        "dataset": dataset,
        "backbone": backbone,
        "observations": observations,
        "held_out_test_evaluated": False,
    }
    write_json_atomic(output_path, payload)
    raise RuntimeError(
        f"GPU preflight failed for {dataset}/{backbone}: "
        f"{observations[-1]['failure_reasons']}"
    )


def write_status(
    *,
    output_root: Path,
    state: str,
    source_revision: str,
    recovery_revision: str,
    message: str,
    execution_server: str = "5080",
    revision_field: str = "recovery_orchestration_revision",
    current_dataset: str | None = None,
    current_backbone: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    allowed = {"prepared", "running", "failed", "merged", "complete"}
    if state not in allowed:
        raise ValueError(f"Unsupported recovery state: {state}")
    if revision_field not in {
        "recovery_orchestration_revision",
        "shard_orchestration_revision",
    }:
        raise ValueError(f"Unsupported orchestration revision field: {revision_field}")
    payload = {
        "status": state,
        "training_source_revision": source_revision,
        revision_field: recovery_revision,
        "execution_server": execution_server,
        "current_dataset": current_dataset,
        "current_backbone": current_backbone,
        "message": message,
        "exit_code": exit_code,
        "updated_at_utc": utc_now(),
        "held_out_test_evaluated": False,
    }
    write_json_atomic(output_root / "screening_status.json", payload)
    for manifest_name in ("recovery_manifest.json", "shard_manifest.json"):
        manifest_path = output_root / manifest_name
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        manifest["status"] = state
        manifest["last_status_update_utc"] = payload["updated_at_utc"]
        if current_dataset is not None:
            manifest["current_dataset"] = current_dataset
        if current_backbone is not None:
            manifest["current_backbone"] = current_backbone
        if exit_code is not None:
            manifest["last_exit_code"] = exit_code
        write_json_atomic(manifest_path, manifest)
    return payload


def _validate_shard_5090_contract(
    path: Path,
    *,
    source_revision: str,
    execution_server: str,
) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("contract_id") != SHARD_5090_CONTRACT_ID:
        raise ValueError(
            f"Unexpected 5090 shard contract id: {contract.get('contract_id')}"
        )
    if contract.get("training_source_revision") != source_revision:
        raise ValueError("5090 shard training revision mismatch")
    if contract.get("execution_server") != execution_server:
        raise ValueError("5090 shard execution server mismatch")
    observed_plan = tuple(
        (row.get("dataset"), row.get("backbone"))
        for row in contract.get("run_plan", [])
    )
    if observed_plan != SHARD_5090_RUNS:
        raise ValueError(f"5090 shard run plan drifted: {observed_plan}")
    observed_ordinals = tuple(
        int(row.get("canonical_ordinal", -1))
        for row in contract.get("run_plan", [])
    )
    if observed_ordinals != tuple(range(4, 10)):
        raise ValueError(f"5090 shard canonical ordinals drifted: {observed_ordinals}")
    if contract.get("evaluation_scope") != "validation_only":
        raise ValueError("5090 shard must remain validation-only")
    if contract.get("held_out_test") != "locked":
        raise ValueError("5090 shard must keep held-out test locked")
    training_files = contract.get("training_files")
    if not isinstance(training_files, dict) or not training_files:
        raise ValueError("5090 shard contract requires frozen training-file hashes")
    for relative_path, expected_sha in training_files.items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)):
            raise ValueError(f"Invalid frozen hash for {relative_path}")
        absolute_path = PROJECT_ROOT / relative_path
        if not absolute_path.is_file():
            raise FileNotFoundError(absolute_path)
        observed_sha = sha256_file(absolute_path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"Frozen training snapshot mismatch for {relative_path}: "
                f"{observed_sha} != {expected_sha}"
            )
    return contract


def prepare_shard_5090(
    *,
    output_root: Path,
    source_revision: str,
    recovery_revision: str,
    contract_path: Path,
    execution_server: str,
) -> dict[str, Any]:
    contract = _validate_shard_5090_contract(
        contract_path,
        source_revision=source_revision,
        execution_server=execution_server,
    )
    assert_no_held_out_artifacts(output_root)
    manifest_path = output_root / "shard_manifest.json"
    identity = {
        "contract_id": SHARD_5090_CONTRACT_ID,
        "training_source_revision": source_revision,
        "shard_orchestration_revision": recovery_revision,
        "execution_server": execution_server,
    }
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        mismatches = {
            key: {"expected": value, "observed": existing.get(key)}
            for key, value in identity.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing 5090 shard manifest mismatch: {mismatches}")
        return existing
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"Refusing nonempty 5090 shard root without manifest: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        **identity,
        "status": "prepared",
        "prepared_at_utc": utc_now(),
        "planned_runs": contract["run_plan"],
        "expected_run_count": len(SHARD_5090_RUNS),
        "completed_run_count": 0,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    write_json_atomic(manifest_path, manifest)
    write_status(
        output_root=output_root,
        state="prepared",
        source_revision=source_revision,
        recovery_revision=recovery_revision,
        execution_server=execution_server,
        revision_field="shard_orchestration_revision",
        message="Prepared canonical runs 4-9 as an isolated 5090 shard.",
    )
    return load_json(manifest_path)


def finalize_shard_5090(
    *,
    output_root: Path,
    source_revision: str,
    recovery_revision: str,
    contract_path: Path,
    execution_server: str,
) -> dict[str, Any]:
    _validate_shard_5090_contract(
        contract_path,
        source_revision=source_revision,
        execution_server=execution_server,
    )
    assert_no_held_out_artifacts(output_root)
    manifest_path = output_root / "shard_manifest.json"
    manifest = load_json(manifest_path)
    identity = {
        "contract_id": SHARD_5090_CONTRACT_ID,
        "training_source_revision": source_revision,
        "shard_orchestration_revision": recovery_revision,
        "execution_server": execution_server,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"5090 shard manifest mismatch: {mismatches}")
    validated_runs = _collect_validated_shard_5090_runs(
        output_root=output_root,
        source_revision=source_revision,
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_run_count": len(validated_runs),
            "validated_runs": validated_runs,
            "held_out_test_evaluated": False,
        }
    )
    write_json_atomic(manifest_path, manifest)
    write_status(
        output_root=output_root,
        state="complete",
        source_revision=source_revision,
        recovery_revision=recovery_revision,
        execution_server=execution_server,
        revision_field="shard_orchestration_revision",
        message="Validated all six 5090 shard runs; transfer to recovery1 is pending.",
    )
    assert_no_held_out_artifacts(output_root)
    return load_json(manifest_path)


def _collect_validated_shard_5090_runs(
    *,
    output_root: Path,
    source_revision: str,
) -> list[dict[str, Any]]:
    validated_runs = []
    for canonical_ordinal, (dataset, backbone) in enumerate(
        SHARD_5090_RUNS,
        start=4,
    ):
        role_dir = shard_role_dir(output_root, dataset, backbone)
        validate_launch_contract(
            role_dir / "launch_contract.json",
            dataset=dataset,
            backbones=(backbone,),
            source_revision=source_revision,
            model_role=MODEL_ROLE_EXPERIMENTAL,
            allowed_statuses={"complete"},
        )
        record = validate_completed_run(
            shard_run_dir(output_root, dataset, backbone),
            dataset=dataset,
            backbone=backbone,
            source_revision=source_revision,
        )
        summary = record["summary"]
        validated_runs.append(
            {
                "canonical_ordinal": canonical_ordinal,
                "dataset": dataset,
                "backbone": backbone,
                "variant": VARIANT,
                "seed": 42,
                "completed_epochs": summary["completed_epochs"],
                "best_epoch": summary["best_epoch"],
                "summary_sha256": record["summary_sha256"],
                "checkpoint_file_sha256": record["checkpoint_file_sha256"],
                "checkpoint_state_sha256": record["checkpoint_state_sha256"],
            }
        )
    return validated_runs


def validate_shard_5090_artifact(
    *,
    shard_root: Path,
    source_revision: str,
    shard_revision: str,
    contract_path: Path,
) -> dict[str, Any]:
    _validate_shard_5090_contract(
        contract_path,
        source_revision=source_revision,
        execution_server="5090",
    )
    assert_no_held_out_artifacts(shard_root)
    manifest = load_json(shard_root / "shard_manifest.json")
    expected = {
        "contract_id": SHARD_5090_CONTRACT_ID,
        "training_source_revision": source_revision,
        "shard_orchestration_revision": shard_revision,
        "execution_server": "5090",
        "status": "complete",
        "completed_run_count": len(SHARD_5090_RUNS),
        "held_out_test_evaluated": False,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"5090 shard artifact manifest mismatch: {mismatches}")
    validated_runs = _collect_validated_shard_5090_runs(
        output_root=shard_root,
        source_revision=source_revision,
    )
    recorded_runs = manifest.get("validated_runs")
    if recorded_runs != validated_runs:
        raise ValueError("5090 shard artifact digest manifest drifted")
    return {"manifest": manifest, "validated_runs": validated_runs}


def import_shard_5090(
    *,
    shard_root: Path,
    output_root: Path,
    source_revision: str,
    recovery_revision: str,
    shard_revision: str,
    contract_path: Path,
) -> dict[str, Any]:
    assert_no_held_out_artifacts(output_root)
    recovery_manifest_path = output_root / "recovery_manifest.json"
    recovery_manifest = load_json(recovery_manifest_path)
    expected_recovery = {
        "contract_id": CONTRACT_ID,
        "training_source_revision": source_revision,
        "recovery_orchestration_revision": recovery_revision,
        "held_out_test_evaluated": False,
    }
    recovery_mismatches = {
        key: {"expected": value, "observed": recovery_manifest.get(key)}
        for key, value in expected_recovery.items()
        if recovery_manifest.get(key) != value
    }
    if recovery_mismatches:
        raise ValueError(f"Recovery manifest mismatch: {recovery_mismatches}")

    shard = validate_shard_5090_artifact(
        shard_root=shard_root,
        source_revision=source_revision,
        shard_revision=shard_revision,
        contract_path=contract_path,
    )
    imported_runs = []
    for source_record, (dataset, backbone) in zip(
        shard["validated_runs"],
        SHARD_5090_RUNS,
        strict=True,
    ):
        source_role = shard_role_dir(shard_root, dataset, backbone)
        destination_role = shard_role_dir(output_root, dataset, backbone)
        destination_run = shard_run_dir(output_root, dataset, backbone)
        action = "installed"
        if destination_role.exists():
            validate_launch_contract(
                destination_role / "launch_contract.json",
                dataset=dataset,
                backbones=(backbone,),
                source_revision=source_revision,
                model_role=MODEL_ROLE_EXPERIMENTAL,
                allowed_statuses={"complete"},
            )
            destination_record = validate_completed_run(
                destination_run,
                dataset=dataset,
                backbone=backbone,
                source_revision=source_revision,
            )
            if (
                destination_record["checkpoint_state_sha256"]
                != source_record["checkpoint_state_sha256"]
            ):
                raise ValueError(
                    "Existing recovery shard differs from validated 5090 shard: "
                    f"{dataset}/{backbone}"
                )
            action = "reused_identical"
        else:
            destination_role.parent.mkdir(parents=True, exist_ok=True)
            incoming = destination_role.with_name(
                f".{destination_role.name}.incoming_5090"
            )
            if incoming.exists():
                raise ValueError(f"Stale 5090 import staging directory: {incoming}")
            shutil.copytree(source_role, incoming)
            validate_launch_contract(
                incoming / "launch_contract.json",
                dataset=dataset,
                backbones=(backbone,),
                source_revision=source_revision,
                model_role=MODEL_ROLE_EXPERIMENTAL,
                allowed_statuses={"complete"},
            )
            staged_record = validate_completed_run(
                incoming / "runs" / backbone / VARIANT / "seed_42",
                dataset=dataset,
                backbone=backbone,
                source_revision=source_revision,
            )
            if (
                staged_record["checkpoint_state_sha256"]
                != source_record["checkpoint_state_sha256"]
            ):
                raise ValueError(
                    f"5090 shard changed during import: {dataset}/{backbone}"
                )
            os.replace(incoming, destination_role)
        imported_runs.append(
            {
                "canonical_ordinal": source_record["canonical_ordinal"],
                "dataset": dataset,
                "backbone": backbone,
                "checkpoint_state_sha256": source_record[
                    "checkpoint_state_sha256"
                ],
                "action": action,
            }
        )

    recovery_manifest["imported_5090_shard"] = {
        "contract_id": SHARD_5090_CONTRACT_ID,
        "shard_orchestration_revision": shard_revision,
        "imported_at_utc": utc_now(),
        "run_count": len(imported_runs),
        "runs": imported_runs,
    }
    write_json_atomic(recovery_manifest_path, recovery_manifest)
    assert_no_held_out_artifacts(output_root)
    return {
        "status": "imported",
        "run_count": len(imported_runs),
        "runs": imported_runs,
        "held_out_test_evaluated": False,
    }


def _install_validated_run(
    *,
    source: Path,
    destination: Path,
    dataset: str,
    backbone: str,
    source_revision: str,
) -> None:
    source_record = validate_completed_run(
        source,
        dataset=dataset,
        backbone=backbone,
        source_revision=source_revision,
    )
    if destination.exists():
        destination_record = validate_completed_run(
            destination,
            dataset=dataset,
            backbone=backbone,
            source_revision=source_revision,
        )
        if (
            destination_record["checkpoint_state_sha256"]
            != source_record["checkpoint_state_sha256"]
        ):
            raise ValueError(
                f"Canonical run already exists with a different state: {destination}"
            )
        _materialize_history_csv(destination, destination_record["history"])
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    _rewrite_checkpoint_path(destination)
    _materialize_history_csv(destination, source_record["history"])


def _summary_without_breakdowns(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary)
    row.pop("quantity_rows", None)
    row.pop("history_rows", None)
    return row


def merge_recovery(
    *,
    output_root: Path,
    source_revision: str,
    recovery_revision: str,
) -> dict[str, Any]:
    assert_no_held_out_artifacts(output_root)
    manifest = load_json(output_root / "recovery_manifest.json")
    identity = {
        "contract_id": CONTRACT_ID,
        "training_source_revision": source_revision,
        "recovery_orchestration_revision": recovery_revision,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Recovery manifest mismatch: {mismatches}")

    source_runs: dict[tuple[str, str], Path] = {
        REUSED_RUN: canonical_run_dir(output_root, *REUSED_RUN)
    }
    for dataset, backbone in RECOVERY_RUNS:
        role_dir = shard_role_dir(output_root, dataset, backbone)
        validate_launch_contract(
            role_dir / "launch_contract.json",
            dataset=dataset,
            backbones=(backbone,),
            source_revision=source_revision,
            model_role=MODEL_ROLE_EXPERIMENTAL,
            allowed_statuses={"complete"},
        )
        source_runs[(dataset, backbone)] = shard_run_dir(
            output_root, dataset, backbone
        )

    for dataset in SCREENING_DATASETS:
        for backbone in TITAN_B012_BACKBONES:
            source = source_runs[(dataset, backbone)]
            destination = canonical_run_dir(output_root, dataset, backbone)
            _install_validated_run(
                source=source,
                destination=destination,
                dataset=dataset,
                backbone=backbone,
                source_revision=source_revision,
            )

    run_records: dict[str, list[dict[str, Any]]] = {}
    validated_runs = []
    for dataset in SCREENING_DATASETS:
        summaries = []
        quantity_rows = []
        history_rows = []
        for backbone in TITAN_B012_BACKBONES:
            record = validate_completed_run(
                canonical_run_dir(output_root, dataset, backbone),
                dataset=dataset,
                backbone=backbone,
                source_revision=source_revision,
            )
            summary = record["summary"]
            summaries.append(_summary_without_breakdowns(summary))
            quantity_rows.extend(summary["quantity_rows"])
            history_rows.extend(summary["history_rows"])
            validated_runs.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "variant": VARIANT,
                    "seed": 42,
                    "completed_epochs": summary["completed_epochs"],
                    "best_epoch": summary["best_epoch"],
                    "summary_sha256": record["summary_sha256"],
                    "checkpoint_file_sha256": record["checkpoint_file_sha256"],
                    "checkpoint_state_sha256": record["checkpoint_state_sha256"],
                    "provenance": (
                        "failed_artifact_validated_b0"
                        if (dataset, backbone) == REUSED_RUN
                        else "isolated_recovery_shard"
                    ),
                }
            )
        role_dir = output_root / dataset / RECOVERY_MODEL_ROLE
        write_csv(role_dir / "run_summaries.csv", summaries)
        write_csv(role_dir / "quantity_seed_metrics.csv", quantity_rows)
        write_csv(role_dir / "history_seed_metrics.csv", history_rows)
        write_csv(
            role_dir / "quantity_summary.csv",
            summarize_breakdowns(
                quantity_rows,
                backbones=TITAN_B012_BACKBONES,
                variants=(VARIANT,),
                seeds=(42,),
            ),
        )
        write_csv(
            role_dir / "history_summary.csv",
            summarize_breakdowns(
                history_rows,
                backbones=TITAN_B012_BACKBONES,
                variants=(VARIANT,),
                seeds=(42,),
            ),
        )
        if dataset == REUSED_RUN[0]:
            template_path = (
                output_root / "provenance" / "failed_artifact_launch_contract.json"
            )
        else:
            template_path = shard_role_dir(
                output_root, dataset, "titantpp"
            ) / "launch_contract.json"
        launch = load_json(template_path)
        launch.update(
            {
                "status": "complete",
                "model_role": RECOVERY_MODEL_ROLE,
                "backbones": list(TITAN_B012_BACKBONES),
                "seeds": [42],
                "expected_run_count": 3,
                "completed_run_count": 3,
                "source_revision": source_revision,
                "execution_host": os.uname().nodename,
                "execution_role": f"recovery_5080_{dataset}_b012_seed42_e300",
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "recovery": {
                    "contract_id": CONTRACT_ID,
                    "recovery_orchestration_revision": recovery_revision,
                    "isolated_backbone_processes": True,
                    "reused_source_run": dataset == REUSED_RUN[0],
                },
            }
        )
        write_json_atomic(role_dir / "launch_contract.json", launch)
        run_records[dataset] = summaries

    assert_no_held_out_artifacts(output_root)
    manifest["status"] = "merged"
    manifest["merged_at_utc"] = utc_now()
    manifest["completed_run_count"] = 9
    manifest["validated_runs"] = validated_runs
    manifest["held_out_test_evaluated"] = False
    write_json_atomic(output_root / "recovery_manifest.json", manifest)
    write_status(
        output_root=output_root,
        state="merged",
        source_revision=source_revision,
        recovery_revision=recovery_revision,
        message="All nine runs validated and merged; exact comparator is pending.",
    )
    return {
        "status": "merged",
        "datasets": run_records,
        "completed_run_count": 9,
        "held_out_test_evaluated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-artifact", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--source-revision", required=True)
    prepare.add_argument("--recovery-revision", required=True)
    prepare.add_argument("--contract", type=Path, required=True)

    prepare_shard = subparsers.add_parser("prepare-shard-5090")
    prepare_shard.add_argument("--output-root", type=Path, required=True)
    prepare_shard.add_argument("--source-revision", required=True)
    prepare_shard.add_argument("--recovery-revision", required=True)
    prepare_shard.add_argument("--contract", type=Path, required=True)
    prepare_shard.add_argument("--execution-server", default="5090")

    preflight = subparsers.add_parser("preflight-gpu")
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--dataset", choices=SCREENING_DATASETS, required=True)
    preflight.add_argument("--backbone", choices=TITAN_B012_BACKBONES, required=True)
    preflight.add_argument("--nvidia-smi", default="nvidia-smi")
    preflight.add_argument("--minimum-free-mib", type=int, default=15000)
    preflight.add_argument("--maximum-used-mib", type=int, default=512)
    preflight.add_argument("--attempts", type=int, default=12)
    preflight.add_argument("--interval-seconds", type=float, default=5.0)
    preflight.add_argument(
        "--forbidden-graphics-process",
        action="append",
        dest="forbidden_graphics_processes",
    )

    inspect = subparsers.add_parser("inspect-shard")
    inspect.add_argument("--output-root", type=Path, required=True)
    inspect.add_argument("--dataset", choices=SCREENING_DATASETS, required=True)
    inspect.add_argument("--backbone", choices=TITAN_B012_BACKBONES, required=True)
    inspect.add_argument("--source-revision", required=True)
    inspect.add_argument("--output", type=Path)
    inspect.add_argument("--action-only", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--output-root", type=Path, required=True)
    status.add_argument(
        "--state",
        choices=("prepared", "running", "failed", "merged", "complete"),
        required=True,
    )
    status.add_argument("--source-revision", required=True)
    status.add_argument("--recovery-revision", required=True)
    status.add_argument("--message", required=True)
    status.add_argument("--execution-server", default="5080")
    status.add_argument(
        "--revision-field",
        choices=(
            "recovery_orchestration_revision",
            "shard_orchestration_revision",
        ),
        default="recovery_orchestration_revision",
    )
    status.add_argument("--current-dataset")
    status.add_argument("--current-backbone")
    status.add_argument("--exit-code", type=int)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--source-revision", required=True)
    merge.add_argument("--recovery-revision", required=True)

    finalize_shard = subparsers.add_parser("finalize-shard-5090")
    finalize_shard.add_argument("--output-root", type=Path, required=True)
    finalize_shard.add_argument("--source-revision", required=True)
    finalize_shard.add_argument("--recovery-revision", required=True)
    finalize_shard.add_argument("--contract", type=Path, required=True)
    finalize_shard.add_argument("--execution-server", default="5090")

    import_shard = subparsers.add_parser("import-shard-5090")
    import_shard.add_argument("--shard-root", type=Path, required=True)
    import_shard.add_argument("--output-root", type=Path, required=True)
    import_shard.add_argument("--source-revision", required=True)
    import_shard.add_argument("--recovery-revision", required=True)
    import_shard.add_argument("--shard-revision", required=True)
    import_shard.add_argument("--contract", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        payload = prepare_recovery(
            source_artifact=args.source_artifact,
            output_root=args.output_root,
            source_revision=args.source_revision,
            recovery_revision=args.recovery_revision,
            contract_path=args.contract,
        )
    elif args.command == "prepare-shard-5090":
        payload = prepare_shard_5090(
            output_root=args.output_root,
            source_revision=args.source_revision,
            recovery_revision=args.recovery_revision,
            contract_path=args.contract,
            execution_server=args.execution_server,
        )
    elif args.command == "preflight-gpu":
        payload = run_gpu_preflight(
            output_path=args.output,
            dataset=args.dataset,
            backbone=args.backbone,
            nvidia_smi=args.nvidia_smi,
            minimum_free_mib=args.minimum_free_mib,
            maximum_used_mib=args.maximum_used_mib,
            attempts=args.attempts,
            interval_seconds=args.interval_seconds,
            forbidden_graphics_names=(
                set(args.forbidden_graphics_processes)
                if args.forbidden_graphics_processes
                else None
            ),
        )
    elif args.command == "inspect-shard":
        payload = inspect_shard(
            output_root=args.output_root,
            dataset=args.dataset,
            backbone=args.backbone,
            source_revision=args.source_revision,
        )
        if args.output is not None:
            write_json_atomic(args.output, payload)
    elif args.command == "status":
        payload = write_status(
            output_root=args.output_root,
            state=args.state,
            source_revision=args.source_revision,
            recovery_revision=args.recovery_revision,
            message=args.message,
            execution_server=args.execution_server,
            revision_field=args.revision_field,
            current_dataset=args.current_dataset,
            current_backbone=args.current_backbone,
            exit_code=args.exit_code,
        )
    elif args.command == "merge":
        payload = merge_recovery(
            output_root=args.output_root,
            source_revision=args.source_revision,
            recovery_revision=args.recovery_revision,
        )
    elif args.command == "finalize-shard-5090":
        payload = finalize_shard_5090(
            output_root=args.output_root,
            source_revision=args.source_revision,
            recovery_revision=args.recovery_revision,
            contract_path=args.contract,
            execution_server=args.execution_server,
        )
    else:
        payload = import_shard_5090(
            shard_root=args.shard_root,
            output_root=args.output_root,
            source_revision=args.source_revision,
            recovery_revision=args.recovery_revision,
            shard_revision=args.shard_revision,
            contract_path=args.contract,
        )
    if args.command == "inspect-shard" and args.action_only:
        print(payload["action"], flush=True)
    else:
        print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "RECOVERY_RUNS",
    "SHARD_5090_RUNS",
    "assert_no_held_out_artifacts",
    "evaluate_gpu_snapshot",
    "find_forbidden_artifacts",
    "import_shard_5090",
    "inspect_shard",
    "merge_recovery",
    "parse_nvidia_process_table",
    "prepare_recovery",
    "prepare_shard_5090",
    "run_gpu_preflight",
    "finalize_shard_5090",
    "validate_completed_run",
    "validate_launch_contract",
    "validate_partial_checkpoint",
    "validate_shard_5090_artifact",
    "write_status",
]
