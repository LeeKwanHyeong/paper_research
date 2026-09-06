#!/usr/bin/env python3
"""Run the frozen-B time-head refit smoke and full validation on one RTX 5090."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTRACT_REL = Path("paper/contracts/hard_lmm_time_head_refit_v1.json")
RUNNER_REL = Path("paper/scripts/run_hard_lmm_time_head_refit.py")
TEST_REL = Path("simple_lab_test/search/tests/test_hard_lmm_time_head_refit.py")
CONTROLLER_TEST_REL = Path(
    "simple_lab_test/search/tests/test_hard_lmm_time_head_refit_5090.py"
)
DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)
TIME_KEYS = ("v_t.weight", "b_t", "w_raw")
EXPECTED_CONTRACT_SHA256 = (
    "f81427b28839883d567e72df3527d81082571f17819ab97aaaf1f369ba2ee88d"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            finite_tree(child, path=f"{path}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Non-finite value at {path}")


def finite_tensor_tree(value: Any, *, path: str = "root") -> None:
    import torch

    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"Non-finite tensor at {path}")
    elif isinstance(value, Mapping):
        for name, child in value.items():
            finite_tensor_tree(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tensor_tree(child, path=f"{path}[{index}]")


def validate_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(
        contract.get("contract_id") == "hard_lmm_time_head_refit_v1",
        "Wrong refit contract",
    )
    scope = contract.get("scope")
    require(isinstance(scope, dict), "Contract scope is missing")
    require(scope.get("input_splits") == ["train", "validation"], "Split scope drift")
    require(scope.get("optimization_split") == "train", "Optimization split drift")
    require(scope.get("selection_split") == "validation", "Selection split drift")
    require(scope.get("evaluation_scope") == "validation_only", "Evaluation scope drift")
    require(scope.get("held_out_test") is False, "Held-out access is enabled")
    require(scope.get("additional_seeds") is False, "Additional seeds are enabled")
    require(scope.get("quantity_predictions_may_change") is False, "Quantity drift enabled")
    boundary = contract.get("parameter_boundary")
    require(isinstance(boundary, dict), "Parameter boundary is missing")
    require(
        boundary.get("trainable_parameter_names") == list(TIME_KEYS),
        "Time-head parameter boundary drift",
    )
    require(boundary.get("expected_trainable_parameter_count") == 66, "Parameter count drift")
    require(boundary.get("encoder_mode") == "eval", "Encoder eval contract drift")
    require(boundary.get("hidden_state_gradient") == "detached", "Detach contract drift")
    require(boundary.get("dropout_during_cache") is False, "Dropout contract drift")
    require(
        boundary.get("target_quantity_masked_before_encoding") is True,
        "Target-quantity masking drift",
    )
    require(boundary.get("target_memory_write_masked") is True, "Memory masking drift")
    optimization = contract.get("optimization")
    require(isinstance(optimization, dict), "Optimization contract is missing")
    expected_optimization = {
        "shared_across_all_datasets": True,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "encoder_batch_size": 128,
        "cached_state_batch_size": 8192,
        "gradient_clip": 1.0,
        "epochs": 100,
        "minimum_epochs": 20,
        "early_stopping_patience": 20,
        "dataset_specific_hyperparameters": False,
        "training_objective": "mean negative legacy_clamped_rmtpp log density",
        "scheduler": None,
    }
    for name, expected in expected_optimization.items():
        require(optimization.get(name) == expected, f"Optimization drift: {name}")
    selection = contract.get("checkpoint_selection")
    require(isinstance(selection, dict), "Selection contract is missing")
    require(selection.get("monitor") == "validation_time_nll", "Monitor drift")
    require(selection.get("rule") == "earliest strict finite minimum", "Rule drift")
    require(selection.get("fallback") == "epoch 0", "Epoch-zero fallback drift")
    require(selection.get("other_metrics_in_selector") is False, "Selector metric drift")
    stability = contract.get("identity_and_stability")
    require(isinstance(stability, dict), "Stability contract is missing")
    expected_stability = {
        "time_nll_replay_absolute_tolerance": 1e-6,
        "reported_quantity_metric_absolute_tolerance": 1e-6,
        "reported_quantity_metric_relative_tolerance": 1e-5,
        "finite_train_loss": True,
        "finite_validation_loss": True,
        "finite_gradient": True,
        "finite_model_and_optimizer_state": True,
        "selected_checkpoint_replay_required": True,
    }
    for name, expected in expected_stability.items():
        require(stability.get(name) == expected, f"Stability contract drift: {name}")
    rows = contract.get("datasets")
    require(isinstance(rows, list), "Dataset contract is missing")
    datasets = {str(row["dataset"]): dict(row) for row in rows}
    require(tuple(datasets) == DATASETS, "Dataset order or scope drift")
    for name, row in datasets.items():
        require(int(row["expected_train_targets"]) > 0, f"Empty train target set: {name}")
        require(
            int(row["expected_validation_targets"]) > 0,
            f"Empty validation target set: {name}",
        )
        require(len(str(row["B_checkpoint_file_sha256"])) == 64, f"B file hash missing: {name}")
        require(len(str(row["B_checkpoint_state_sha256"])) == 64, f"B state hash missing: {name}")
    return datasets


def verify_source(project: Path, revision: str) -> dict[str, Any]:
    require(
        len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
        "A full lowercase source revision is required",
    )
    manifest = read_json(project / "source_manifest.json")
    require(manifest.get("source_revision") == revision, "Source revision mismatch")
    require(manifest.get("held_out_test_evaluated") is False, "Source manifest opened held-out")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Source manifest is empty")
    for relative_name, expected in files.items():
        relative = Path(relative_name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        path = project / relative
        require(path.is_file(), f"Manifest file is missing: {relative_name}")
        require(sha256_file(path) == expected, f"Manifest checksum drift: {relative_name}")
    for critical in (
        CONTRACT_REL,
        RUNNER_REL,
        TEST_REL,
        CONTROLLER_TEST_REL,
        Path(__file__).relative_to(PROJECT_ROOT),
    ):
        require(str(critical) in files, f"Critical source is not manifested: {critical}")
    for path in project.rglob("*.py"):
        relative_name = str(path.relative_to(project))
        require(
            relative_name in files,
            f"Python source is not manifested: {relative_name}",
        )
    return manifest


def checkpoint_path(b_root: Path, dataset: str) -> Path:
    return (
        b_root
        / "seed42_e300"
        / dataset
        / "quantile_checkpoint_alignment"
        / "runs/titantpp/count_only_log_regression/seed_42"
        / "best_val_qty_rmse_model.pt"
    )


def verify_inputs(
    project: Path,
    b_root: Path,
    datasets: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, row in datasets.items():
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            path = project / str(row[path_key])
            require(path.is_file(), f"Missing {name} input: {path_key}")
            require(sha256_file(path) == row[hash_key], f"{name} {path_key} checksum drift")
        checkpoint = checkpoint_path(b_root, name)
        require(checkpoint.is_file(), f"Missing B checkpoint: {name}")
        require(
            sha256_file(checkpoint) == row["B_checkpoint_file_sha256"],
            f"B checkpoint checksum drift: {name}",
        )


def command_output(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode not in allowed:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def gpu_preflight() -> dict[str, Any]:
    rows = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.free",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    require(len(rows) == 1, "Expected exactly one GPU")
    gpu_name, raw_free = (part.strip() for part in rows[0].split(",", 1))
    free_mib = int(raw_free)
    require("RTX 5090" in gpu_name, f"Wrong GPU: {gpu_name}")
    require(free_mib >= 12000, f"Insufficient free VRAM: {free_mib} MiB")
    compute = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    require(not compute, f"GPU already has a compute process: {compute}")
    gdm = command_output(["systemctl", "is-active", "gdm"], allowed=(0, 3))
    require(gdm == "inactive", f"GDM must be inactive, got {gdm!r}")
    return {
        "gpu_name": gpu_name,
        "free_vram_mib": free_mib,
        "compute_processes": [],
        "gdm": gdm,
    }


def deterministic_env(project: Path) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(project),
    }


def runner_command(
    *,
    python: str,
    project: Path,
    b_root: Path,
    output: Path,
    revision: str,
    row: Mapping[str, Any],
    phase: str,
) -> list[str]:
    require(phase in {"e1", "full"}, f"Unknown phase: {phase}")
    command = [
        python,
        "-s",
        str(project / RUNNER_REL),
        "--dataset",
        str(row["dataset"]),
        "--data",
        str(project / str(row["data_path"])),
        "--split-manifest",
        str(project / str(row["split_manifest_path"])),
        "--checkpoint",
        str(checkpoint_path(b_root, str(row["dataset"]))),
        "--output-dir",
        str(output),
        "--calibration-source-revision",
        revision,
        "--contract",
        str(project / CONTRACT_REL),
        "--device",
        "cuda",
    ]
    if phase == "e1":
        command.extend(
            [
                "--allow-partial-contract",
                "--max-epochs",
                "1",
            ]
        )
    else:
        command.extend(
            [
                "--feature-cache-dir",
                str(output.parent.parent / "e1" / str(row["dataset"]) / "cache"),
            ]
        )
    return command


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    project: Path,
    env: Mapping[str, str],
    child_slot: dict[str, subprocess.Popen[str] | None],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(
            command,
            cwd=project,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        child_slot["child"] = child
        return_code = child.wait()
        child_slot["child"] = None
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def audit_pytest(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    cases = list(root.iter("testcase"))
    require(bool(cases), "Contract suite executed no tests")
    require(not list(root.iter("failure")), "Contract suite failed")
    require(not list(root.iter("error")), "Contract suite errored")
    require(not list(root.iter("skipped")), "Contract suite skipped a test")
    return {
        "test_count": len(cases),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "xml_sha256": sha256_file(xml_path),
    }


def audit_run(
    output: Path,
    *,
    contract: Mapping[str, Any],
    row: Mapping[str, Any],
    phase: str,
    source_checkpoint: Path,
) -> dict[str, Any]:
    import torch

    from paper.scripts.run_hard_lmm_time_head_refit import (
        FrozenFeatureCache,
        build_source_model,
        evaluate_cached_time_nll,
    )
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    summary = read_json(output / "summary.json")
    finite_tree(summary)
    require(summary.get("status") == "success", "Refit did not complete")
    require(summary.get("dataset") == row["dataset"], "Dataset identity drift")
    require(summary.get("seed") == 42, "Seed drift")
    require(summary.get("evaluation_scope") == "validation_only", "Evaluation scope drift")
    require(summary.get("held_out_test_evaluated") is False, "Held-out was evaluated")
    require(
        summary.get("source_checkpoint_sha256") == row["B_checkpoint_file_sha256"],
        "Summary source checkpoint digest drift",
    )
    require(
        summary.get("source_state_sha256") == row["B_checkpoint_state_sha256"],
        "Summary source model-state digest drift",
    )
    require(
        summary.get("trainable_parameter_names") == list(TIME_KEYS)
        and summary.get("trainable_parameter_count") == 66,
        "Trainable boundary drift",
    )
    require(summary.get("encoder_mode_during_refit") == "eval", "Encoder mode drift")
    require(summary.get("hidden_state_gradient") == "detached_cache", "Hidden state was not detached")
    require(
        summary.get("selected_non_time_state_sha256")
        == summary.get("source_non_time_state_sha256"),
        "Non-time state drift",
    )
    require(summary.get("quantity_prediction_bitwise_identical") is True, "Quantity changed")
    require(
        summary.get("source_quantity_prediction_sha256")
        == summary.get("selected_quantity_prediction_sha256"),
        "Quantity prediction digest drift",
    )
    require(summary.get("time_nll_non_worse_than_B") is True, "Time NLL worsened from B")
    history = summary.get("history")
    require(isinstance(history, list) and bool(history), "History is missing")
    require([int(item["epoch"]) for item in history] == list(range(len(history))), "History drift")
    selected = min(history, key=lambda item: (float(item["val_time_nll"]), int(item["epoch"])))
    require(int(summary["best_epoch"]) == int(selected["epoch"]), "Selector replay drift")
    require(
        math.isclose(
            float(summary["best_val_time_nll"]),
            float(selected["val_time_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Selected Time NLL replay drift",
    )
    checkpoint = torch.load(
        output / "best_validation_time_nll_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    selected_state = checkpoint.get("model_state_dict")
    require(isinstance(selected_state, dict), "Selected model state is missing")
    finite_tensor_tree(selected_state, path="selected_model_state")
    require(checkpoint.get("evaluation_scope") == "validation_only", "Selected scope drift")
    require(checkpoint.get("held_out_test_evaluated") is False, "Selected held-out drift")
    require(checkpoint.get("backbone") == "titantpp", "Selected backbone drift")
    require(checkpoint.get("variant") == "count_only_log_regression", "Selected variant drift")
    selected_state_sha256 = canonical_state_dict_sha256(selected_state)
    require(
        checkpoint.get("model_state_sha256")
        == selected_state_sha256
        == summary.get("selected_state_sha256"),
        "Selected checkpoint digest metadata drift",
    )
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_state = source_payload.get("model_state_dict")
    require(isinstance(source_state, dict), "B source model state is missing")
    finite_tensor_tree(source_state, path="B_source_model_state")
    require(
        canonical_state_dict_sha256(source_state) == row["B_checkpoint_state_sha256"],
        "B source model-state digest drift",
    )
    require(set(selected_state) == set(source_state), "Selected model key set drift")
    changed_state_keys = sorted(
        name
        for name in source_state
        if not torch.equal(source_state[name], selected_state[name])
    )
    require(
        set(changed_state_keys).issubset(TIME_KEYS),
        f"State outside the time head changed: {changed_state_keys}",
    )
    require(
        checkpoint.get("source_state_sha256") == row["B_checkpoint_state_sha256"],
        "Selected checkpoint source lineage drift",
    )
    require(
        math.isclose(
            float(checkpoint["selected_metric_value"]),
            float(summary["best_val_time_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Selected checkpoint metric drift",
    )
    resume_identity = summary.get("resume_identity")
    require(isinstance(resume_identity, dict), "Summary resume identity is missing")
    require(
        checkpoint.get("resume_identity") == resume_identity,
        "Selected checkpoint resume identity drift",
    )
    last = torch.load(output / "last_epoch_state.pt", map_location="cpu", weights_only=False)
    require(last.get("checkpoint_type") == "time_head_refit_resume", "Wrong last-state type")
    require(last.get("checkpoint_schema_version") == 1, "Wrong last-state schema")
    require(last.get("evaluation_scope") == "validation_only", "Last-state scope drift")
    require(last.get("held_out_test_evaluated") is False, "Last-state held-out drift")
    require(last.get("resume_identity") == resume_identity, "Last-state identity drift")
    require(last.get("history") == history, "Last-state history drift")
    require(
        int(last.get("epoch", -1)) == int(summary["completed_epochs"]),
        "Last-state completed epoch drift",
    )
    require(
        int(last.get("best_epoch", -1)) == int(summary["best_epoch"]),
        "Last-state selected epoch drift",
    )
    current_state = last.get("model_state_dict")
    best_state = last.get("best_state_dict")
    require(isinstance(current_state, dict), "Last current model state is missing")
    require(isinstance(best_state, dict), "Last best model state is missing")
    finite_tensor_tree(current_state, path="last_current_model_state")
    finite_tensor_tree(best_state, path="last_best_model_state")
    require(
        canonical_state_dict_sha256(current_state) == last.get("model_state_sha256"),
        "Last current-state digest drift",
    )
    require(
        canonical_state_dict_sha256(best_state) == last.get("best_state_sha256"),
        "Last best-state digest drift",
    )
    require(set(current_state) == set(source_state), "Last current model key set drift")
    require(set(best_state) == set(selected_state), "Last best model key set drift")
    for name in source_state:
        if name not in TIME_KEYS:
            require(
                torch.equal(current_state[name], source_state[name]),
                f"Last current non-time state drift: {name}",
            )
            require(
                torch.equal(best_state[name], source_state[name]),
                f"Last best non-time state drift: {name}",
            )
        require(
            torch.equal(best_state[name], selected_state[name]),
            f"Last best state differs from selected checkpoint: {name}",
        )
    optimizer_state = last.get("optimizer_state_dict")
    require(isinstance(optimizer_state, dict), "Last optimizer state is missing")
    optimizer_rows = optimizer_state.get("state")
    optimizer_groups = optimizer_state.get("param_groups")
    require(
        isinstance(optimizer_rows, dict) and len(optimizer_rows) == len(TIME_KEYS),
        "Last optimizer parameter coverage drift",
    )
    require(
        isinstance(optimizer_groups, list)
        and len(optimizer_groups) == 1,
        "Last optimizer parameter-group drift",
    )
    optimizer_parameter_ids = optimizer_groups[0].get("params", [])
    require(
        optimizer_parameter_ids == list(range(len(TIME_KEYS))),
        "Last optimizer parameter list drift",
    )
    expected_optimizer_group = {
        "lr": float(contract["optimization"]["learning_rate"]),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": float(contract["optimization"]["weight_decay"]),
        "amsgrad": False,
        "maximize": False,
    }
    for name, expected in expected_optimizer_group.items():
        require(
            optimizer_groups[0].get(name) == expected,
            f"Last AdamW parameter-group drift: {name}",
        )
    optional_optimizer_group = {
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    for name, expected in optional_optimizer_group.items():
        if name in optimizer_groups[0]:
            require(
                optimizer_groups[0][name] == expected,
                f"Last AdamW parameter-group drift: {name}",
            )
    require(
        set(optimizer_rows) == set(optimizer_parameter_ids),
        "Last optimizer state parameter IDs drift",
    )
    expected_optimizer_steps = int(summary["completed_epochs"]) * math.ceil(
        int(summary["train_cache"]["count"])
        / int(contract["optimization"]["cached_state_batch_size"])
    )
    for parameter_id, parameter_name in zip(optimizer_parameter_ids, TIME_KEYS):
        parameter_state = optimizer_rows[parameter_id]
        require(
            isinstance(parameter_state, dict)
            and {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state),
            f"Last AdamW moment state is incomplete: {parameter_name}",
        )
        expected = current_state[parameter_name]
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            require(
                isinstance(moment, torch.Tensor)
                and moment.shape == expected.shape
                and moment.dtype == expected.dtype,
                f"Last AdamW {moment_name} shape or dtype drift: {parameter_name}",
            )
        require(
            isinstance(parameter_state["step"], torch.Tensor)
            and parameter_state["step"].numel() == 1
            and parameter_state["step"].dtype == torch.float32
            and float(parameter_state["step"].item())
            == float(expected_optimizer_steps),
            f"Last AdamW step drift: {parameter_name}",
        )
    finite_tensor_tree(optimizer_state, path="optimizer_state_dict")

    expected_cache_dir = (
        output / "cache"
        if phase == "e1"
        else output.parent.parent / "e1" / str(row["dataset"]) / "cache"
    ).resolve()
    require(
        Path(summary["feature_cache_dir"]).resolve() == expected_cache_dir,
        "Validation feature-cache route drift",
    )
    validation_cache_payload = torch_load_checkpoint(
        expected_cache_dir / "validation_features.pt",
        map_location="cpu",
    )
    cache_identity = validation_cache_payload.get("identity")
    require(isinstance(cache_identity, dict), "Validation cache identity is missing")
    expected_cache_identity = {
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "dataset": row["dataset"],
        "data_sha256": row["data_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "source_checkpoint_sha256": row["B_checkpoint_file_sha256"],
        "source_state_sha256": row["B_checkpoint_state_sha256"],
        "split": "validation",
        "target_count": row["expected_validation_targets"],
        "target_identity_sha256": row["expected_validation_target_identity_sha256"],
        "target_quantity_sha256": row["expected_validation_target_quantity_sha256"],
        "max_batches": None,
        "encoder_mode": "eval",
        "target_quantity_masked": True,
        "memory_target_write_masked": True,
        "evaluation_scope": "train_and_validation_only",
        "held_out_test_evaluated": False,
    }
    for name, expected in expected_cache_identity.items():
        require(cache_identity.get(name) == expected, f"Validation cache identity drift: {name}")
    validation_cache = FrozenFeatureCache.from_payload(
        validation_cache_payload,
        expected_identity=cache_identity,
        require_quantity=True,
    )
    replay_model = build_source_model(checkpoint).to("cpu")
    replayed_time_nll = evaluate_cached_time_nll(
        model=replay_model,
        cache=validation_cache,
        device="cpu",
        batch_size=int(contract["optimization"]["cached_state_batch_size"]),
    )
    require(
        math.isclose(
            replayed_time_nll,
            float(summary["best_val_time_nll"]),
            rel_tol=0.0,
            abs_tol=float(
                contract["identity_and_stability"][
                    "time_nll_replay_absolute_tolerance"
                ]
            ),
        ),
        "Selected checkpoint validation Time NLL replay drift",
    )
    if phase in {"e1", "full"}:
        stability = contract["identity_and_stability"]
        time_tolerance = float(stability["time_nll_replay_absolute_tolerance"])
        quantity_absolute_tolerance = float(
            stability["reported_quantity_metric_absolute_tolerance"]
        )
        quantity_relative_tolerance = float(
            stability["reported_quantity_metric_relative_tolerance"]
        )
        require(
            int(summary.get("quantity_replay_batch_size", -1))
            == int(contract["optimization"]["encoder_batch_size"]),
            "Quantity replay batch shape drift",
        )
        runtime = summary.get("runtime")
        require(isinstance(runtime, dict), "CUDA runtime telemetry is missing")
        require(runtime.get("requested_device") == "cuda", "Run did not request CUDA")
        require(runtime.get("cuda_available") is True, "CUDA was unavailable during run")
        require("RTX 5090" in str(runtime.get("device_name")), "Wrong execution GPU")
        require(
            int(runtime.get("peak_memory_allocated_bytes", 0)) > 0
            and int(runtime.get("peak_memory_reserved_bytes", 0)) > 0,
            "CUDA peak-memory telemetry is empty",
        )
        require(float(runtime.get("elapsed_seconds", 0.0)) > 0.0, "Runtime telemetry is empty")
        require(
            summary.get("qualified_full_data") is (phase == "full"),
            f"{phase} qualification marker drift",
        )
        require(
            int(summary["train_cache"]["count"]) == int(row["expected_train_targets"]),
            f"{phase} train target count drift",
        )
        require(
            int(summary["validation_cache"]["count"])
            == int(row["expected_validation_targets"]),
            f"{phase} validation target count drift",
        )
        require(
            math.isclose(
                float(summary["epoch_zero_val_time_nll"]),
                float(row["B_metrics"]["time_nll"]),
                rel_tol=0.0,
                abs_tol=time_tolerance,
            ),
            f"{phase} epoch-zero B Time NLL did not replay",
        )
        require(
            summary.get("train_target_population", {}).get("target_identity_sha256")
            == row["expected_train_target_identity_sha256"],
            f"{phase} train target identity drift",
        )
        require(
            summary.get("validation_target_population", {}).get(
                "target_identity_sha256"
            )
            == row["expected_validation_target_identity_sha256"],
            f"{phase} validation target identity drift",
        )
        for metric, source_name in (
            ("mae", "overall_mae"),
            ("rmse", "raw_rmse"),
            ("body_mae", "body_mae"),
            ("gt_p99_mae", "gt_p99_mae"),
        ):
            require(
                math.isclose(
                    float(summary["quantity_metrics"][metric]),
                    float(row["B_metrics"][source_name]),
                    rel_tol=quantity_relative_tolerance,
                    abs_tol=quantity_absolute_tolerance,
                ),
                f"{phase} B quantity metric replay drift: {metric}",
            )
        if phase == "e1":
            require(int(summary["completed_epochs"]) == 1, "e1 did not complete one epoch")
    forbidden = [
        str(path)
        for path in output.rglob("*")
        if path.is_file()
        and (path.name.startswith("test_") or "held_out" in path.name.lower())
    ]
    require(not forbidden, f"Held-out artifact found: {forbidden}")
    return {
        "status": "passed",
        "phase": phase,
        "dataset": row["dataset"],
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "B_time_nll": float(summary["epoch_zero_val_time_nll"]),
        "selected_time_nll": float(summary["best_val_time_nll"]),
        "time_nll_improvement": float(summary["time_nll_improvement_from_B"]),
        "time_nll_non_worse_than_B": bool(
            summary["time_guardrails"].get("non_worse_than_B")
            if phase == "full"
            else summary["time_nll_non_worse_than_B"]
        ),
        "restores_A_plus_0_01": bool(summary["time_guardrails"]["restores_A_plus_0_01"]),
        "quantity_prediction_bitwise_identical": True,
        "source_state_sha256": summary["source_state_sha256"],
        "selected_state_sha256": summary["selected_state_sha256"],
        "selected_time_head_state_sha256": summary["selected_time_head_state_sha256"],
        "changed_state_keys": changed_state_keys,
        "selected_checkpoint_replayed_time_nll": replayed_time_nll,
        "runtime": summary["runtime"],
        "train_cache_count": int(summary["train_cache"]["count"]),
        "validation_cache_count": int(summary["validation_cache"]["count"]),
        "held_out_test_evaluated": False,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve()
    require(project == PROJECT_ROOT.resolve(), "Controller must execute from its manifested project")
    b_root = args.b_controller_root.resolve()
    output = args.output_root.resolve()
    require(
        sha256_file(project / CONTRACT_REL) == EXPECTED_CONTRACT_SHA256,
        "Pinned time-head refit contract digest drift",
    )
    contract = read_json(project / CONTRACT_REL)
    datasets = validate_contract(contract)
    source_manifest = verify_source(project, args.source_revision)
    verify_inputs(project, b_root, datasets)
    if args.verify_only:
        return {"status": "verified", "source_revision": args.source_revision}
    output.mkdir(parents=True, exist_ok=True)
    launch_identity = {
        "schema_version": 1,
        "contract_sha256": sha256_file(project / CONTRACT_REL),
        "source_revision": args.source_revision,
        "source_manifest_sha256": sha256_file(project / "source_manifest.json"),
        "B_controller_root": str(b_root),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
    }
    launch_identity_path = output / "control" / "launch_identity.json"
    copied_manifest_path = output / "control" / "source_manifest.json"
    if launch_identity_path.exists():
        require(
            read_json(launch_identity_path) == launch_identity,
            "Existing output launch identity drift",
        )
        require(copied_manifest_path.is_file(), "Existing output source manifest is missing")
        require(
            sha256_file(copied_manifest_path)
            == launch_identity["source_manifest_sha256"],
            "Existing output source manifest drift",
        )
    else:
        require(
            not any(output.iterdir()),
            "Pre-populated output has no validated launch identity",
        )
        launch_identity_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(launch_identity_path, launch_identity)
        shutil.copy2(project / "source_manifest.json", copied_manifest_path)
    existing_status_path = output / "status.json"
    if existing_status_path.exists():
        existing_status = read_json(existing_status_path)
        require(
            existing_status.get("source_revision") == args.source_revision,
            "Existing output belongs to another source revision",
        )
        require(
            existing_status.get("B_controller_root") == str(b_root),
            "Existing output belongs to another B artifact root",
        )
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "source_revision": args.source_revision,
        "source_manifest_sha256": launch_identity["source_manifest_sha256"],
        "source_manifest_file_count": len(source_manifest["files"]),
        "B_controller_root": str(b_root),
        "completed_e1": [],
        "completed_full": [],
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
    }
    child_slot: dict[str, subprocess.Popen[str] | None] = {"child": None}

    def update(**values: Any) -> None:
        state.update(values)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "status.json", state)

    def stop(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = deterministic_env(project)
    with (output / "controller.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            update(status="running", phase="contract_tests", gpu_preflight=gpu_preflight())
            xml_path = output / "contract_tests.xml"
            run_logged(
                [
                    args.python,
                    "-s",
                    "-m",
                    "pytest",
                    "-q",
                    str(project / TEST_REL),
                    str(project / CONTROLLER_TEST_REL),
                    f"--junitxml={xml_path}",
                ],
                output / "contract_tests.log",
                project=project,
                env=environment,
                child_slot=child_slot,
            )
            update(contract_tests=audit_pytest(xml_path))
            for phase in ("e1", "full"):
                completed_key = "completed_e1" if phase == "e1" else "completed_full"
                for dataset in DATASETS:
                    verify_source(project, args.source_revision)
                    verify_inputs(project, b_root, datasets)
                    preflight = gpu_preflight()
                    run_output = output / phase / dataset
                    update(
                        phase=phase,
                        current_dataset=dataset,
                        stage="refit",
                        gpu_preflight=preflight,
                    )
                    command = runner_command(
                        python=args.python,
                        project=project,
                        b_root=b_root,
                        output=run_output,
                        revision=args.source_revision,
                        row=datasets[dataset],
                        phase=phase,
                    )
                    run_logged(
                        command,
                        output / phase / f"{dataset}.log",
                        project=project,
                        env=environment,
                        child_slot=child_slot,
                    )
                    update(stage="audit")
                    audit = audit_run(
                        run_output,
                        contract=contract,
                        row=datasets[dataset],
                        phase=phase,
                        source_checkpoint=checkpoint_path(b_root, dataset),
                    )
                    state[completed_key].append({"dataset": dataset, "audit": audit})
                    update(**{completed_key: state[completed_key]}, stage="complete")
            full_records = [item["audit"] for item in state["completed_full"]]
            decision = {
                "schema_version": 1,
                "status": "complete",
                "quantity_exactly_preserved_on_all_datasets": all(
                    bool(item["quantity_prediction_bitwise_identical"])
                    for item in full_records
                ),
                "time_nll_non_worse_than_B_on_all_datasets": all(
                    bool(item["time_nll_non_worse_than_B"])
                    for item in full_records
                ),
                "restores_A_plus_0_01_on_all_datasets": all(
                    bool(item["restores_A_plus_0_01"]) for item in full_records
                ),
                "datasets": full_records,
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
                "additional_seeds_executed": False,
            }
            save_json(output / "decision.json", decision)
            update(
                status="complete",
                phase="complete",
                current_dataset=None,
                stage="complete",
                decision=decision,
            )
            return state
        except BaseException as error:
            child = child_slot["child"]
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            update(status="failed", stage="failed", error=repr(error))
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--b-controller-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = execute(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
