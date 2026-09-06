#!/usr/bin/env python3
"""Fail-closed RTX 5090 screening for the frozen-B duration adapter.

The controller never opens held-out data and never launches an additional
seed.  It verifies the committed source package and the immutable selected-B
checkpoint/cache lineage, runs the CUDA contracts, executes one full-data
Taxi epoch, and then performs the fixed Taxi seed-42 screen.  Intermittent and
Instacart are admitted only when Taxi passes every pre-registered gate.
"""

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

CONTRACT_REL = Path("paper/contracts/hard_lmm_causal_duration_adapter_v1.json")
RUNNER_REL = Path("paper/scripts/run_hard_lmm_causal_duration_adapter.py")
CONTROLLER_REL = Path(
    "paper/scripts/run_hard_lmm_causal_duration_adapter_5090.py"
)
MODEL_REL = Path("models/TPPs/CausalLogDurationAdapter.py")
ADAPTER_TEST_REL = Path(
    "simple_lab_test/search/tests/test_causal_log_duration_adapter.py"
)
RUNNER_TEST_REL = Path(
    "simple_lab_test/search/tests/test_hard_lmm_causal_duration_adapter_runner.py"
)
CONTROLLER_TEST_REL = Path(
    "simple_lab_test/search/tests/test_hard_lmm_causal_duration_adapter_5090.py"
)
CUDA_TESTS = (ADAPTER_TEST_REL, RUNNER_TEST_REL, CONTROLLER_TEST_REL)
CUDA_SENTINEL_TEST_NAME = (
    "test_controller_required_cuda_adapter_contract_executes_on_cuda"
)

CONTRACT_DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)
TAXI_DATASET = "yellow_trip_hourly"
POST_TAXI_DATASETS = ("intermittent_frozen_5000", "insta_market_basket")
ROLE_NAMES = ("candidate", "global_scale_control")
SELECTED_CHECKPOINT_NAME = "best_validation_continuous_nll_model.pt"
LAST_CHECKPOINT_NAME = "last_epoch_state.pt"

MAX_CONTINUOUS_NLL_DELTA_VS_B = 0.0
MAX_CONTINUOUS_NLL_DELTA_VS_A = 0.01
MIN_CONTINUOUS_NLL_GAIN_VS_CONTROL = 0.005
EXECUTION_ORDER = (
    "cuda_contract_tests",
    "taxi_e1",
    "taxi_seed42_screening",
    "conditional_intermittent_seed42",
    "conditional_instacart_seed42",
)
RESUMABLE_STATUSES = {"starting", "running", "failed"}
TERMINAL_STATUSES = {
    "complete",
    "complete_with_screening_failures",
    "stopped_after_taxi_screening_failure",
}


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
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
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
    if isinstance(value, Mapping):
        for name, child in value.items():
            finite_tree(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
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


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} is missing")
    return value


def validate_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(
        contract.get("contract_id") == "hard_lmm_causal_duration_adapter_v1",
        "Wrong causal duration-adapter contract",
    )
    candidate = _require_mapping(contract.get("candidate"), "Candidate contract")
    expected_candidate = {
        "name": "B_frozen_causal_log_duration_scale_adapter",
        "single_hypothesis": True,
        "source_contract_id": "hard_lmm_frozen_lognormal_duration_v1",
        "source_model": "selected proper Frozen-B seed42",
        "backbone_change": False,
        "quantity_path_change": False,
        "location_change": False,
        "scale_only": True,
    }
    for name, expected in expected_candidate.items():
        require(candidate.get(name) == expected, f"Candidate drift: {name}")

    scope = _require_mapping(contract.get("scope"), "Scope contract")
    expected_scope = {
        "input_splits": ["train", "validation"],
        "optimization_split": "train",
        "selection_split": "validation",
        "evaluation_scope": "validation_only",
        "held_out_test": False,
        "additional_seeds": False,
        "quantity_predictions_may_change": False,
        "time_median_may_change": False,
    }
    for name, expected in expected_scope.items():
        require(scope.get(name) == expected, f"Scope drift: {name}")

    frozen = _require_mapping(contract.get("frozen_source"), "Frozen source")
    require(
        frozen.get("checkpoint_type") == "selected_frozen_lognormal_duration",
        "Frozen checkpoint type drift",
    )
    require(frozen.get("model_mode") == "eval", "Frozen model mode drift")
    require(
        frozen.get("state_digest_must_remain_unchanged") is True,
        "Frozen state may change",
    )

    stability = _require_mapping(
        contract.get("identity_and_stability"), "Identity contract"
    )
    require(
        stability.get("time_median_identity_source")
        == "canonical Frozen-B median tensor calculated on the requested runtime device",
        "Time-median identity source drift",
    )
    require(
        float(stability.get("cross_device_time_median_formula_absolute_tolerance"))
        == 1e-12
        and float(
            stability.get("cross_device_time_median_formula_relative_tolerance")
        )
        == 1e-12,
        "Cross-device median formula tolerance drift",
    )

    adapter = _require_mapping(contract.get("adapter"), "Adapter contract")
    expected_adapter = {
        "appended_target_excluded": True,
        "marks_excluded": True,
        "quantities_excluded": True,
        "B_hidden_excluded": True,
        "recurrent_cell": "torch.nn.GRU",
        "input_size": 1,
        "hidden_size": 8,
        "num_layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
        "output": "bounded_log_scale_residual",
        "output_projection_initialization": "exact zeros",
        "location_formula": "mu_candidate = mu_B exactly",
        "expected_trainable_parameter_count": 273,
        "maximum_scale_ratio": 10.0,
        "minimum_scale_ratio": 0.1,
        "sigma_floor": 0.001,
    }
    for name, expected in expected_adapter.items():
        require(adapter.get(name) == expected, f"Adapter drift: {name}")

    control = _require_mapping(
        contract.get("global_scale_control"), "Global scale control"
    )
    require(
        control.get("trainable_parameter_count") == 1
        and control.get("input_history") is False,
        "Global-scale control drift",
    )

    likelihood = _require_mapping(
        contract.get("observation_likelihood"), "Observation likelihood"
    )
    require(
        likelihood.get("selection_metric")
        == "continuous censor-aware normalized log-normal NLL in original time units",
        "Primary likelihood drift",
    )
    require(
        likelihood.get("interval_nll_in_selector", False) is False,
        "Interval telemetry entered checkpoint selection",
    )

    optimization = _require_mapping(contract.get("optimization"), "Optimization")
    expected_optimization = {
        "shared_across_all_datasets": True,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 4096,
        "gradient_clip": 1.0,
        "epochs": 50,
        "minimum_epochs": 5,
        "early_stopping_patience": 8,
        "scheduler": None,
        "dataset_specific_hyperparameters": False,
    }
    for name, expected in expected_optimization.items():
        require(optimization.get(name) == expected, f"Optimization drift: {name}")

    selection = _require_mapping(
        contract.get("checkpoint_selection"), "Checkpoint selection"
    )
    require(
        selection.get("monitor") == "validation_continuous_proper_time_nll",
        "Checkpoint monitor drift",
    )
    require(
        selection.get("rule") == "earliest strict finite minimum",
        "Checkpoint rule drift",
    )
    require(selection.get("fallback") == "epoch 0", "Epoch-zero fallback drift")
    require(
        selection.get("interval_nll_in_selector") is False,
        "Interval NLL entered the selector",
    )

    execution = _require_mapping(
        contract.get("execution_policy"), "Execution policy"
    )
    require(execution.get("taxi_first_stop") is True, "Taxi-first policy disabled")
    require(
        execution.get("stop_after_taxi_failure") is True,
        "Taxi stop policy disabled",
    )
    require(
        execution.get("additional_seed_policy") == "stopped; no seeds 52 or 62",
        "Additional seed policy drift",
    )

    acceptance = _require_mapping(contract.get("acceptance"), "Acceptance")
    expected_acceptance = {
        "candidate_continuous_nll_vs_B": (
            "candidate <= frozen B on every evaluated dataset"
        ),
        "candidate_continuous_nll_vs_A": (
            "candidate <= A + 0.01 on every evaluated dataset"
        ),
        "candidate_interval_nll_vs_B": (
            "candidate <= frozen B on every evaluated dataset"
        ),
        "history_attribution": (
            "candidate continuous NLL <= global-scale control continuous NLL - 0.005"
        ),
        "quantity_guardrail": "all B quantity predictions and metrics exactly preserved",
        "time_point_guardrail": (
            "all B validation time medians and median-error metrics exactly preserved"
        ),
        "legacy_nll_comparison_prohibited": True,
        "cross_dataset_nll_aggregation_prohibited": True,
    }
    for name, expected in expected_acceptance.items():
        require(acceptance.get(name) == expected, f"Acceptance drift: {name}")
    structured = _require_mapping(
        acceptance.get("structured_gates"), "Structured acceptance gates"
    )
    expected_structured = {
        "maximum_candidate_minus_B_continuous_nll": (
            MAX_CONTINUOUS_NLL_DELTA_VS_B
        ),
        "maximum_candidate_minus_A_continuous_nll": (
            MAX_CONTINUOUS_NLL_DELTA_VS_A
        ),
        "require_candidate_interval_nll_nonworse_than_B": True,
        "minimum_candidate_improvement_over_global_control_continuous_nll": (
            MIN_CONTINUOUS_NLL_GAIN_VS_CONTROL
        ),
        "require_quantity_prediction_bitwise_identity": True,
        "require_time_median_bitwise_identity": True,
        "require_base_location_bitwise_identity": True,
        "require_source_model_state_digest_identity": True,
    }
    for name, expected in expected_structured.items():
        require(structured.get(name) == expected, f"Structured gate drift: {name}")

    rows = contract.get("datasets")
    require(isinstance(rows, list), "Dataset contract is missing")
    datasets = {str(row["dataset"]): dict(row) for row in rows}
    require(tuple(datasets) == CONTRACT_DATASETS, "Dataset order or scope drift")
    for name, row in datasets.items():
        for key in (
            "data_sha256",
            "split_manifest_sha256",
            "frozen_B_checkpoint_file_sha256",
            "frozen_B_model_state_sha256",
            "frozen_B_train_feature_cache_sha256",
            "frozen_B_validation_feature_cache_sha256",
            "frozen_B_quantity_prediction_sha256",
            "expected_train_target_identity_sha256",
            "expected_validation_target_identity_sha256",
            "expected_train_target_dt_sha256",
            "expected_validation_target_dt_sha256",
        ):
            require(len(str(row.get(key))) == 64, f"Missing pinned digest: {name}.{key}")
        require(int(row["expected_train_targets"]) > 0, f"Empty train set: {name}")
        require(
            int(row["expected_validation_targets"]) > 0,
            f"Empty validation set: {name}",
        )
        require(
            int(row["train_active_context_log1p"]["token_count"]) > 0
            and float(row["train_active_context_log1p"]["log1p_std"]) > 0.0,
            f"Invalid train-only history statistics: {name}",
        )
        is_instacart = name == "insta_market_basket"
        require(
            row.get("right_censor_threshold") == (30.0 if is_instacart else None),
            f"Censor threshold drift: {name}",
        )
    return datasets


def verify_source(project: Path, revision: str) -> dict[str, Any]:
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "A full lowercase source revision is required",
    )
    manifest = read_json(project / "source_manifest.json")
    require(manifest.get("source_revision") == revision, "Source revision mismatch")
    require(
        manifest.get("held_out_test_evaluated") is False,
        "Source manifest opened held-out data",
    )
    files = manifest.get("files")
    require(isinstance(files, Mapping) and bool(files), "Source manifest is empty")
    for relative_name, expected in files.items():
        relative = Path(str(relative_name))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe manifest path: {relative_name}",
        )
        path = project / relative
        require(path.is_file(), f"Manifest file is missing: {relative_name}")
        require(
            sha256_file(path) == expected,
            f"Source/input checksum mismatch: {relative_name}",
        )
    for critical in (
        CONTRACT_REL,
        RUNNER_REL,
        CONTROLLER_REL,
        MODEL_REL,
        ADAPTER_TEST_REL,
        RUNNER_TEST_REL,
        CONTROLLER_TEST_REL,
    ):
        require(str(critical) in files, f"Critical source is not manifested: {critical}")
    return manifest


def frozen_checkpoint_path(frozen_b_root: Path, dataset: str) -> Path:
    return (
        frozen_b_root
        / "full"
        / dataset
        / "best_validation_proper_time_nll_model.pt"
    )


def frozen_cache_dir(frozen_b_root: Path, dataset: str) -> Path:
    return frozen_b_root / "e1" / dataset / "cache"


def verify_dataset_inputs(
    project: Path,
    frozen_b_root: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    from paper.scripts.run_hard_lmm_causal_duration_adapter import (
        load_verified_feature_cache,
    )
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    dataset = str(row["dataset"])
    data = project / str(row["data_path"])
    split = project / str(row["split_manifest_path"])
    checkpoint = frozen_checkpoint_path(frozen_b_root, dataset)
    cache_dir = frozen_cache_dir(frozen_b_root, dataset)
    require(data.is_file(), f"Missing data file: {dataset}")
    require(split.is_file(), f"Missing split manifest: {dataset}")
    require(checkpoint.is_file(), f"Missing selected Frozen-B checkpoint: {dataset}")
    require(sha256_file(data) == row["data_sha256"], f"Data checksum drift: {dataset}")
    require(
        sha256_file(split) == row["split_manifest_sha256"],
        f"Split-manifest checksum drift: {dataset}",
    )
    checkpoint_file_sha = sha256_file(checkpoint)
    require(
        checkpoint_file_sha == row["frozen_B_checkpoint_file_sha256"],
        f"Frozen-B checkpoint checksum drift: {dataset}",
    )
    selected = torch_load_checkpoint(checkpoint, map_location="cpu")
    require(
        selected.get("checkpoint_type") == "selected_frozen_lognormal_duration",
        f"Wrong Frozen-B checkpoint type: {dataset}",
    )
    source_state = selected.get("model_state_dict")
    require(isinstance(source_state, Mapping), f"Missing Frozen-B state: {dataset}")
    finite_tensor_tree(source_state, path=f"frozen_B.{dataset}")
    source_state_sha = canonical_state_dict_sha256(source_state)
    require(
        source_state_sha == row["frozen_B_model_state_sha256"],
        f"Frozen-B state checksum drift: {dataset}",
    )

    cache_records: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        cache_path = cache_dir / f"{split_name}_features.pt"
        require(cache_path.is_file(), f"Missing {split_name} cache: {dataset}")
        cache = load_verified_feature_cache(
            path=cache_path,
            split=split_name,
            dataset_name=dataset,
            dataset_spec=row,
            selected_payload=selected,
            require_quantity=split_name == "validation",
        )
        cache_records[split_name] = {
            "path": str(cache_path.resolve()),
            "file_sha256": sha256_file(cache_path),
            "logical_sha256": cache.digest(),
            "count": cache.count,
        }
    return {
        "dataset": dataset,
        "data_path": str(data.resolve()),
        "data_sha256": row["data_sha256"],
        "split_manifest_path": str(split.resolve()),
        "split_manifest_sha256": row["split_manifest_sha256"],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_file_sha256": checkpoint_file_sha,
        "checkpoint_state_sha256": source_state_sha,
        "feature_cache_dir": str(cache_dir.resolve()),
        "feature_caches": cache_records,
    }


def verify_input_fingerprint(record: Mapping[str, Any]) -> None:
    for key, sha_key in (
        ("data_path", "data_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
        ("checkpoint_path", "checkpoint_file_sha256"),
    ):
        path = Path(str(record[key]))
        require(path.is_file(), f"Pinned input disappeared: {path}")
        require(sha256_file(path) == record[sha_key], f"Pinned input changed: {path}")
    caches = _require_mapping(record.get("feature_caches"), "Feature cache records")
    for split in ("train", "validation"):
        item = _require_mapping(caches.get(split), f"{split} feature cache")
        path = Path(str(item["path"]))
        require(path.is_file(), f"Pinned feature cache disappeared: {path}")
        require(
            sha256_file(path) == item["file_sha256"],
            f"Pinned feature cache changed: {path}",
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
    name, raw_free = (part.strip() for part in rows[0].split(",", 1))
    free_mib = int(raw_free)
    require("RTX 5090" in name, f"Wrong GPU: {name}")
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
        "gpu_name": name,
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
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(project),
        "HARD_LMM_CAUSAL_DURATION_ADAPTER_REQUIRE_CUDA": "1",
    }


def runner_command(
    *,
    python: str,
    project: Path,
    frozen_b_root: Path,
    output: Path,
    revision: str,
    row: Mapping[str, Any],
    phase: str,
) -> list[str]:
    require(phase in {"e1", "seed42"}, f"Unknown phase: {phase}")
    dataset = str(row["dataset"])
    command = [
        python,
        "-s",
        str(project / RUNNER_REL),
        "--dataset",
        dataset,
        "--data",
        str(project / str(row["data_path"])),
        "--split-manifest",
        str(project / str(row["split_manifest_path"])),
        "--checkpoint",
        str(frozen_checkpoint_path(frozen_b_root, dataset)),
        "--feature-cache-dir",
        str(frozen_cache_dir(frozen_b_root, dataset)),
        "--output-dir",
        str(output),
        "--source-revision",
        revision,
        "--contract",
        str(project / CONTRACT_REL),
        "--device",
        "cuda",
    ]
    if phase == "e1":
        command.extend(["--allow-partial-contract", "--max-epochs", "1"])
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
    require(bool(cases), "CUDA contract suite executed no tests")
    require(not list(root.iter("failure")), "CUDA contract suite failed")
    require(not list(root.iter("error")), "CUDA contract suite errored")
    require(not list(root.iter("skipped")), "CUDA contract suite skipped a test")
    executed_names = {str(case.get("name")) for case in cases}
    require(
        CUDA_SENTINEL_TEST_NAME in executed_names,
        "CUDA sentinel contract test did not execute",
    )
    return {
        "test_count": len(cases),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "cuda_sentinel_test": CUDA_SENTINEL_TEST_NAME,
        "cuda_sentinel_passed": True,
        "xml_sha256": sha256_file(xml_path),
    }


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 0.0,
) -> None:
    require(
        math.isclose(
            float(actual),
            float(expected),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ),
        f"{label} drift: expected {expected}, got {actual}",
    )


def _earliest_minimum(history: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    require(bool(history), "Selection history is empty")
    best = history[0]
    best_value = float(best["val_continuous_proper_time_nll"])
    require(math.isfinite(best_value), "Epoch-zero NLL is non-finite")
    for item in history[1:]:
        value = float(item["val_continuous_proper_time_nll"])
        require(math.isfinite(value), "Selection NLL is non-finite")
        if value < best_value:
            best = item
            best_value = value
    return best


def _audit_role(
    output: Path,
    *,
    role: str,
    embedded: Mapping[str, Any],
    root_summary: Mapping[str, Any],
    row: Mapping[str, Any],
    contract_sha256: str,
    source_revision: str,
    phase: str,
) -> dict[str, Any]:
    import torch

    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256,
        torch_load_checkpoint,
    )

    require(role in ROLE_NAMES, f"Unknown adapter role: {role}")
    role_dir = output / role
    summary = read_json(role_dir / "summary.json")
    require(summary == dict(embedded), f"Embedded {role} summary drift")
    finite_tree(summary, path=f"{role}.summary")
    require(summary.get("status") == "success", f"{role} did not complete")
    require(summary.get("role") == role, f"{role} identity drift")
    require(summary.get("contract_id") == root_summary["contract_id"], f"{role} contract drift")
    require(
        summary.get("evaluation_scope") == "validation_only"
        and summary.get("held_out_test_evaluated") is False,
        f"{role} opened held-out data",
    )
    expected_parameters = 273 if role == "candidate" else 1
    require(
        int(summary.get("trainable_parameter_count", -1)) == expected_parameters,
        f"{role} parameter boundary drift",
    )
    history = summary.get("history")
    require(isinstance(history, list) and bool(history), f"{role} history is missing")
    require(
        [int(item["epoch"]) for item in history] == list(range(len(history))),
        f"{role} history epochs are not contiguous",
    )
    require(history[0].get("train_continuous_proper_time_nll") is None, f"{role} epoch zero has train loss")
    for item in history[1:]:
        for key in (
            "train_continuous_proper_time_nll",
            "val_continuous_proper_time_nll",
            "val_interval_mass_time_nll",
            "pre_clip_gradient_norm_mean",
            "pre_clip_gradient_norm_max",
        ):
            require(math.isfinite(float(item[key])), f"Non-finite {role} history: {key}")
        require(
            float(item["pre_clip_gradient_norm_max"]) > 0.0,
            f"{role} recorded no active gradient",
        )
    selected_row = _earliest_minimum(history)
    completed_epochs = int(summary["completed_epochs"])
    require(completed_epochs == len(history) - 1, f"{role} completed/history drift")
    if phase == "e1":
        require(completed_epochs == 1, f"{role} e1 did not complete one epoch")
    else:
        require(1 <= completed_epochs <= 50, f"{role} full epoch range drift")
        if summary.get("stopped_early") is False:
            require(completed_epochs == 50, f"{role} full run ended before epoch 50")
    require(
        int(summary["best_epoch"]) == int(selected_row["epoch"]),
        f"{role} selector drift",
    )
    selected_metrics = _require_mapping(
        summary.get("selected_validation_metrics"), f"{role} selected metrics"
    )
    _assert_close(
        selected_metrics["continuous_proper_time_nll"],
        selected_row["val_continuous_proper_time_nll"],
        label=f"{role} selected NLL",
    )

    identity = _require_mapping(summary.get("resume_identity"), f"{role} resume identity")
    expected_train_count = int(row["expected_train_targets"])
    expected_validation_count = int(row["expected_validation_targets"])
    expected_identity = {
        "schema_version": 1,
        "contract_id": "hard_lmm_causal_duration_adapter_v1",
        "contract_sha256": contract_sha256,
        "role": role,
        "dataset": row["dataset"],
        "source_checkpoint_sha256": row["frozen_B_checkpoint_file_sha256"],
        "source_state_sha256": row["frozen_B_model_state_sha256"],
        "train_count": expected_train_count,
        "validation_count": expected_validation_count,
        "source_revision": source_revision,
        "selection": "earliest_strict_finite_minimum_validation_continuous_nll",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for name, expected in expected_identity.items():
        require(identity.get(name) == expected, f"{role} resume identity drift: {name}")
    optimization = _require_mapping(identity.get("optimization"), f"{role} optimization")
    require(optimization.get("seed") == 42, f"{role} seed drift")
    require(optimization.get("dataset") == row["dataset"], f"{role} dataset setting drift")
    require(
        int(optimization.get("epochs", -1)) == (1 if phase == "e1" else 50),
        f"{role} planned epoch drift",
    )
    expected_cache_sha = (
        row["frozen_B_train_feature_cache_sha256"]
        + ":"
        + row["frozen_B_validation_feature_cache_sha256"]
    )
    require(
        identity.get("frozen_cache_sha256") == expected_cache_sha,
        f"{role} frozen cache lineage drift",
    )

    selected = torch_load_checkpoint(
        role_dir / SELECTED_CHECKPOINT_NAME, map_location="cpu"
    )
    require(
        selected.get("checkpoint_type") == "selected_causal_duration_scale_adapter",
        f"Wrong selected {role} checkpoint type",
    )
    require(selected.get("role") == role, f"Selected {role} identity drift")
    require(selected.get("resume_identity") == identity, f"Selected {role} resume drift")
    require(
        int(selected.get("best_epoch", -1)) == int(summary["best_epoch"]),
        f"Selected {role} epoch drift",
    )
    require(
        selected.get("selection")
        == "earliest_strict_finite_minimum_validation_continuous_nll",
        f"Selected {role} rule drift",
    )
    require(
        selected.get("evaluation_scope") == "validation_only"
        and selected.get("held_out_test_evaluated") is False,
        f"Selected {role} opened held-out data",
    )
    selected_state = _require_mapping(
        selected.get("module_state_dict"), f"Selected {role} state"
    )
    finite_tensor_tree(selected_state, path=f"selected.{role}")
    selected_sha = canonical_state_dict_sha256(selected_state)
    require(
        selected_sha
        == selected.get("module_state_sha256")
        == summary.get("selected_module_state_sha256"),
        f"Selected {role} state digest drift",
    )
    _assert_close(
        selected.get("selected_metric_value"),
        selected_metrics["continuous_proper_time_nll"],
        label=f"Selected {role} checkpoint metric",
    )

    last = torch_load_checkpoint(role_dir / LAST_CHECKPOINT_NAME, map_location="cpu")
    require(
        last.get("checkpoint_type") == "causal_duration_adapter_resume",
        f"Wrong {role} resume checkpoint type",
    )
    require(last.get("role") == role, f"Last {role} identity drift")
    require(last.get("resume_identity") == identity, f"Last {role} resume drift")
    require(last.get("history") == history, f"Last {role} history drift")
    require(int(last.get("epoch", -1)) == completed_epochs, f"Last {role} epoch drift")
    require(
        int(last.get("best_epoch", -1)) == int(summary["best_epoch"]),
        f"Last {role} selected epoch drift",
    )
    current_state = _require_mapping(last.get("module_state_dict"), f"Last {role} state")
    best_state = _require_mapping(last.get("best_state_dict"), f"Best {role} state")
    finite_tensor_tree(last, path=f"last.{role}")
    require(
        canonical_state_dict_sha256(current_state) == last.get("module_state_sha256"),
        f"Last {role} state digest drift",
    )
    require(
        canonical_state_dict_sha256(best_state)
        == last.get("best_state_sha256")
        == selected_sha,
        f"Best {role} resume state drift",
    )
    require(set(current_state) == set(best_state) == set(selected_state), f"{role} state-key drift")
    for name in selected_state:
        require(torch.equal(best_state[name], selected_state[name]), f"Selected/best {role} tensor drift: {name}")
    optimizer = _require_mapping(last.get("optimizer_state_dict"), f"{role} optimizer")
    optimizer_state = _require_mapping(optimizer.get("state"), f"{role} optimizer state")
    groups = optimizer.get("param_groups")
    require(isinstance(groups, list) and len(groups) == 1, f"{role} optimizer groups drift")
    parameter_ids = groups[0].get("params")
    expected_parameter_tensors = 6 if role == "candidate" else 1
    require(
        isinstance(parameter_ids, list)
        and len(parameter_ids) == expected_parameter_tensors
        and set(parameter_ids) == set(optimizer_state),
        f"{role} optimizer parameter coverage drift",
    )
    require(groups[0].get("lr") == 0.001, f"{role} optimizer learning-rate drift")
    require(groups[0].get("weight_decay") == 0.0, f"{role} optimizer decay drift")
    expected_steps = completed_epochs * math.ceil(expected_train_count / 4096)
    for parameter_id in parameter_ids:
        parameter_state = _require_mapping(
            optimizer_state.get(parameter_id), f"{role} parameter optimizer state"
        )
        require(
            {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state),
            f"{role} optimizer moments are incomplete",
        )
        step = parameter_state["step"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and int(step.item()) == expected_steps,
            f"{role} optimizer step drift",
        )
    finite_tensor_tree(optimizer, path=f"optimizer.{role}")
    return {
        "status": "passed",
        "role": role,
        "completed_epochs": completed_epochs,
        "best_epoch": int(summary["best_epoch"]),
        "continuous_proper_time_nll": float(
            selected_metrics["continuous_proper_time_nll"]
        ),
        "interval_mass_time_nll": float(
            selected_metrics["interval_mass_time_nll"]
        ),
        "time_median_sha256": selected_metrics["time_median_sha256"],
        "selected_module_state_sha256": selected_sha,
        "resume_checkpoint_audited": True,
    }


def acceptance_decision(
    summary: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    base = _require_mapping(summary.get("base_B_validation_metrics"), "B metrics")
    candidate = _require_mapping(summary.get("candidate"), "Candidate")
    control = _require_mapping(summary.get("global_scale_control"), "Control")
    candidate_metrics = _require_mapping(
        candidate.get("selected_validation_metrics"), "Candidate selected metrics"
    )
    control_metrics = _require_mapping(
        control.get("selected_validation_metrics"), "Control selected metrics"
    )
    candidate_continuous = float(candidate_metrics["continuous_proper_time_nll"])
    candidate_interval = float(candidate_metrics["interval_mass_time_nll"])
    control_continuous = float(control_metrics["continuous_proper_time_nll"])
    base_continuous = float(base["continuous_proper_time_nll"])
    base_interval = float(base["interval_mass_time_nll"])
    a_continuous = float(row["A_validation_continuous_nll"])
    gates = {
        "candidate_continuous_nll_at_most_B": (
            candidate_continuous <= base_continuous + MAX_CONTINUOUS_NLL_DELTA_VS_B
        ),
        "candidate_continuous_nll_at_most_A_plus_0_01": (
            candidate_continuous <= a_continuous + MAX_CONTINUOUS_NLL_DELTA_VS_A
        ),
        "candidate_interval_nll_at_most_B": candidate_interval <= base_interval,
        "candidate_beats_global_control_by_0_005": (
            candidate_continuous
            <= control_continuous - MIN_CONTINUOUS_NLL_GAIN_VS_CONTROL
        ),
        "quantity_prediction_bitwise_identical": (
            summary.get("quantity_prediction_bitwise_identical") is True
        ),
        "time_median_bitwise_identical": (
            summary.get("time_median_bitwise_identical") is True
        ),
        "source_model_state_unchanged": (
            summary.get("source_model_state_unchanged") is True
        ),
        "base_location_bitwise_identical": (
            summary.get("base_location_bitwise_identical") is True
        ),
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "dataset": row["dataset"],
        "gates": gates,
        "metrics": {
            "candidate_continuous_nll": candidate_continuous,
            "frozen_B_continuous_nll": base_continuous,
            "A_continuous_nll": a_continuous,
            "global_scale_control_continuous_nll": control_continuous,
            "candidate_interval_nll": candidate_interval,
            "frozen_B_interval_nll": base_interval,
        },
        "thresholds": {
            "max_continuous_nll_delta_vs_B": MAX_CONTINUOUS_NLL_DELTA_VS_B,
            "max_continuous_nll_delta_vs_A": MAX_CONTINUOUS_NLL_DELTA_VS_A,
            "minimum_continuous_nll_gain_vs_control": (
                MIN_CONTINUOUS_NLL_GAIN_VS_CONTROL
            ),
        },
    }


def audit_run(
    output: Path,
    *,
    row: Mapping[str, Any],
    contract_path: Path,
    source_revision: str,
    phase: str,
) -> dict[str, Any]:
    require(phase in {"e1", "seed42"}, f"Unknown audit phase: {phase}")
    summary = read_json(output / "summary.json")
    finite_tree(summary)
    require(summary.get("schema_version") == 1, "Wrong summary schema")
    require(
        summary.get("contract_id") == "hard_lmm_causal_duration_adapter_v1",
        "Summary contract drift",
    )
    require(summary.get("status") == "success", "Adapter run did not complete")
    require(summary.get("dataset") == row["dataset"], "Dataset identity drift")
    require(summary.get("seed") == 42, "Seed drift")
    require(
        summary.get("evaluation_scope") == "validation_only"
        and summary.get("held_out_test_evaluated") is False,
        "Run opened held-out data",
    )
    require(summary.get("legacy_nll_compared") is False, "Legacy NLL was compared")
    require(
        summary.get("source_checkpoint_sha256")
        == row["frozen_B_checkpoint_file_sha256"],
        "Frozen-B checkpoint lineage drift",
    )
    require(
        summary.get("source_model_state_sha256")
        == row["frozen_B_model_state_sha256"],
        "Frozen-B state lineage drift",
    )
    require(summary.get("source_model_state_unchanged") is True, "Frozen-B changed")
    require(
        summary.get("train_feature_cache_sha256")
        == row["frozen_B_train_feature_cache_sha256"]
        and summary.get("validation_feature_cache_sha256")
        == row["frozen_B_validation_feature_cache_sha256"],
        "Frozen-B feature cache lineage drift",
    )
    expected_run_train_count = int(row["expected_train_targets"])
    expected_run_validation_count = int(row["expected_validation_targets"])
    require(
        int(summary.get("train_count", -1)) == expected_run_train_count
        and int(summary.get("validation_count", -1))
        == expected_run_validation_count,
        "Run target count drift",
    )
    require(
        summary.get("qualified_full_data") is (phase == "seed42"),
        "Full-data qualification drift",
    )
    for split in ("train", "validation"):
        population = _require_mapping(
            summary.get(f"{split}_target_population"), f"{split} population"
        )
        require(
            int(population.get("target_count", -1))
            == int(row[f"expected_{split}_targets"]),
            f"{split} population count drift",
        )
        require(
            population.get("target_identity_sha256")
            == row[f"expected_{split}_target_identity_sha256"],
            f"{split} population identity drift",
        )

    observed_stats = _require_mapping(
        summary.get("train_history_statistics"), "Train history statistics"
    )
    expected_stats = row["train_active_context_log1p"]
    for name in ("token_count", "context_length_min", "context_length_max"):
        require(observed_stats.get(name) == expected_stats[name], f"History statistic drift: {name}")
    for name in ("log1p_mean", "log1p_std", "context_length_mean"):
        _assert_close(
            observed_stats[name],
            expected_stats[name],
            label=f"History statistic {name}",
            absolute_tolerance=1e-12,
        )

    base = _require_mapping(summary.get("base_B_validation_metrics"), "B metrics")
    require(
        int(base.get("count", -1)) == expected_run_validation_count,
        "B validation count drift",
    )
    _assert_close(
        base["continuous_proper_time_nll"],
        row["frozen_B_validation_continuous_nll"],
        label="Frozen-B continuous NLL replay",
        absolute_tolerance=1e-6,
    )
    _assert_close(
        summary["A_validation_continuous_nll"],
        row["A_validation_continuous_nll"],
        label="A continuous NLL baseline",
        absolute_tolerance=0.0,
    )
    require(math.isfinite(float(base["interval_mass_time_nll"])), "B interval NLL is non-finite")
    require(
        summary.get("quantity_prediction_bitwise_identical") is True,
        "B quantity predictions changed",
    )
    require(
        summary.get("cached_quantity_prediction_sha256")
        == row["frozen_B_quantity_prediction_sha256"],
        "Cached B quantity prediction digest drift",
    )
    require(
        summary.get("runtime_quantity_before_sha256")
        == summary.get("runtime_quantity_after_sha256"),
        "Same-device B quantity prediction digest changed",
    )
    require(
        summary.get("cached_vs_runtime_quantity_close") is True,
        "Cached and replayed B quantity predictions disagree",
    )
    require(summary.get("time_median_bitwise_identical") is True, "B median changed")
    require(
        summary.get("base_location_bitwise_identical") is True,
        "Frozen-B base location changed",
    )
    require(summary.get("epoch_zero_exact_B") is True, "Epoch-zero B replay drift")
    quantity = _require_mapping(summary.get("quantity_metrics"), "Quantity metrics")
    for actual_name, expected_name in (
        ("rmse", "raw_rmse"),
        ("mae", "overall_mae"),
        ("body_mae", "body_mae"),
        ("gt_p99_mae", "gt_p99_mae"),
    ):
        _assert_close(
            quantity[actual_name],
            row["B_metrics"][expected_name],
            label=f"B quantity {actual_name}",
            absolute_tolerance=1e-6,
            relative_tolerance=1e-5,
        )

    role_audits = {
        role: _audit_role(
            output,
            role=role,
            embedded=_require_mapping(summary.get(role), role),
            root_summary=summary,
            row=row,
            contract_sha256=sha256_file(contract_path),
            source_revision=source_revision,
            phase=phase,
        )
        for role in ROLE_NAMES
    }
    require(
        role_audits["candidate"]["time_median_sha256"]
        == role_audits["global_scale_control"]["time_median_sha256"]
        == base["time_median_sha256"],
        "B time-median digest changed",
    )
    candidate_metrics = summary["candidate"]["selected_validation_metrics"]
    control_metrics = summary["global_scale_control"]["selected_validation_metrics"]
    require(
        candidate_metrics["base_location_sha256"]
        == control_metrics["base_location_sha256"]
        == base["base_location_sha256"],
        "Frozen-B base-location digest changed",
    )

    runtime = _require_mapping(summary.get("runtime"), "Runtime telemetry")
    require(runtime.get("requested_device") == "cuda", "Run did not request CUDA")
    require(runtime.get("cuda_available") is True, "CUDA was unavailable")
    require("RTX 5090" in str(runtime.get("device_name")), "Wrong execution GPU")
    require(float(runtime.get("elapsed_seconds", 0.0)) > 0.0, "Runtime is empty")
    require(
        int(runtime.get("peak_memory_allocated_bytes", 0)) > 0
        and int(runtime.get("peak_memory_reserved_bytes", 0)) > 0,
        "CUDA peak-memory telemetry is empty",
    )

    if phase == "e1":
        require(
            summary.get("acceptance_status") == "not_evaluated_partial_run",
            "e1 was used for performance selection",
        )
        decision = None
    else:
        decision = acceptance_decision(summary, row)
        runner_gates = _require_mapping(
            summary.get("acceptance_gates"), "Runner acceptance gates"
        )
        require(
            all(runner_gates.get(name) is passed for name, passed in decision["gates"].items()),
            "Runner/controller acceptance decision drift",
        )
        require(
            summary.get("acceptance_status") == decision["status"],
            "Runner acceptance status drift",
        )
        expected_taxi_stop = (
            row["dataset"] == TAXI_DATASET and decision["status"] != "passed"
        )
        require(
            summary.get("taxi_first_stop_triggered") is expected_taxi_stop,
            "Taxi stop marker drift",
        )
    forbidden = [
        str(path)
        for path in output.rglob("*")
        if path.is_file()
        and (path.name.startswith("test_") or "held_out" in path.name.lower())
    ]
    require(not forbidden, f"Held-out artifact found: {forbidden}")
    audit = {
        "schema_version": 1,
        "status": "passed",
        "phase": phase,
        "dataset": row["dataset"],
        "source_revision": source_revision,
        "roles": role_audits,
        "frozen_B_quantity_exact": True,
        "frozen_B_location_exact": True,
        "frozen_B_time_median_exact": True,
        "epoch_zero_exact_B": True,
        "acceptance_decision": decision,
        "held_out_test_evaluated": False,
    }
    save_json(output / "run_audit.json", audit)
    return audit


def controller_resume_identity(
    *,
    project: Path,
    frozen_b_root: Path,
    source_revision: str,
    source_manifest_sha256: str,
    contract_sha256: str,
    taxi_input_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a resumable controller to all immutable execution inputs."""
    return {
        "schema_version": 1,
        "source_revision": source_revision,
        "source_root": str(project.resolve()),
        "frozen_B_root": str(frozen_b_root.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "contract_sha256": contract_sha256,
        "taxi_input_record": dict(taxi_input_record),
        "execution_order": list(EXECUTION_ORDER),
        "screening_seed": 42,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
    }


def _validate_completed_rows(
    value: Any,
    *,
    label: str,
    allowed_datasets: tuple[str, ...],
) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{label} is missing")
    result = [dict(item) for item in value]
    names = [str(item.get("dataset")) for item in result]
    require(len(names) == len(set(names)), f"{label} contains duplicates")
    require(
        all(name in allowed_datasets for name in names),
        f"{label} contains an out-of-contract dataset",
    )
    for item in result:
        require(isinstance(item.get("audit"), Mapping), f"{label} audit is missing")
    return result


def validate_resumable_state(
    state: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Reject completed, foreign, or insufficiently identified output trees."""
    allowed_top_level = {
        "controller.lock",
        "status.json",
        "control",
        "cuda_contract_tests.xml",
        "cuda_contract_tests.log",
        "e1",
        "seed42",
    }
    unexpected = sorted(
        path.name for path in output.iterdir() if path.name not in allowed_top_level
    )
    require(not unexpected, f"Unexpected resumable output artifacts: {unexpected}")
    forbidden = [
        str(path)
        for path in output.rglob("*")
        if path.is_file()
        and (path.name.startswith("test_") or "held_out" in path.name.lower())
    ]
    require(not forbidden, f"Held-out artifact found in resumable output: {forbidden}")
    finite_tree(state, path="controller_state")
    require(state.get("schema_version") == 1, "Controller state schema drift")
    status = str(state.get("status"))
    require(status not in TERMINAL_STATUSES, f"Controller run is already terminal: {status}")
    require(status in RESUMABLE_STATUSES, f"Controller state is not resumable: {status}")
    require(
        state.get("controller_resume_identity") == dict(expected_identity),
        "Controller resume identity drift",
    )
    for name in ("source_revision", "source_manifest_sha256", "contract_sha256"):
        require(
            state.get(name) == expected_identity[name],
            f"Controller state identity drift: {name}",
        )
    require(
        state.get("held_out_test_evaluated") is False,
        "Controller state reports held-out access",
    )
    require(
        state.get("additional_seeds_executed") is False,
        "Controller state reports additional seeds",
    )
    require(
        state.get("execution_order") == list(EXECUTION_ORDER),
        "Controller execution order drift",
    )
    completed_e1 = _validate_completed_rows(
        state.get("completed_e1"),
        label="completed_e1",
        allowed_datasets=(TAXI_DATASET,),
    )
    completed_seed42 = _validate_completed_rows(
        state.get("completed_seed42"),
        label="completed_seed42",
        allowed_datasets=(TAXI_DATASET, *POST_TAXI_DATASETS),
    )
    require(
        [item["dataset"] for item in completed_seed42]
        == list((TAXI_DATASET, *POST_TAXI_DATASETS))[: len(completed_seed42)],
        "Seed42 completion order drift",
    )
    require(
        not completed_seed42 or bool(completed_e1),
        "Taxi screening completed before Taxi e1",
    )
    copied_manifest = output / "control" / "source_manifest.json"
    copied_contract = output / "control" / CONTRACT_REL.name
    require(copied_manifest.is_file(), "Copied source manifest is missing")
    require(copied_contract.is_file(), "Copied contract is missing")
    require(
        sha256_file(copied_manifest) == expected_identity["source_manifest_sha256"],
        "Copied source manifest changed",
    )
    require(
        sha256_file(copied_contract) == expected_identity["contract_sha256"],
        "Copied contract changed",
    )
    evidence = _require_mapping(state.get("control_evidence"), "Control evidence")
    expected_evidence = {
        "source_manifest_path": str(copied_manifest.resolve()),
        "source_manifest_sha256": expected_identity["source_manifest_sha256"],
        "contract_path": str(copied_contract.resolve()),
        "contract_sha256": expected_identity["contract_sha256"],
    }
    require(evidence == expected_evidence, "Control-evidence identity drift")
    verified = _require_mapping(
        state.get("verified_input_manifest"), "Verified input manifest"
    )
    require(
        set(verified).issubset(CONTRACT_DATASETS),
        "Verified input manifest contains an out-of-contract dataset",
    )
    require(
        verified.get(TAXI_DATASET) == expected_identity["taxi_input_record"],
        "Stored Taxi input identity drift",
    )
    resumed = dict(state)
    resumed["completed_e1"] = completed_e1
    resumed["completed_seed42"] = completed_seed42
    resumed["screening_decisions"] = dict(
        _require_mapping(state.get("screening_decisions"), "Screening decisions")
    )
    resumed["verified_input_manifest"] = dict(verified)
    deferred = state.get("conditional_inputs_deferred")
    require(isinstance(deferred, list), "Conditional input state is missing")
    require(
        tuple(deferred) in (POST_TAXI_DATASETS, (POST_TAXI_DATASETS[1],), ()),
        "Conditional input state drift",
    )
    verified_conditional = {
        name for name in verified if name in POST_TAXI_DATASETS
    }
    require(
        verified_conditional == set(POST_TAXI_DATASETS) - set(deferred),
        "Verified/deferred conditional input state drift",
    )
    decisions = resumed["screening_decisions"]
    require(
        set(decisions).issubset(
            {item["dataset"] for item in completed_seed42}
        ),
        "Screening decision has no completed run",
    )
    if verified_conditional or any(
        item["dataset"] in POST_TAXI_DATASETS for item in completed_seed42
    ):
        taxi_decision = _require_mapping(
            decisions.get(TAXI_DATASET), "Taxi screening decision"
        )
        require(
            taxi_decision.get("status") == "passed",
            "A conditional dataset ran after a failed Taxi screen",
        )
    resumed["conditional_inputs_deferred"] = list(deferred)
    return resumed


def _completed_audit(
    state: Mapping[str, Any], *, key: str, dataset: str
) -> Mapping[str, Any] | None:
    rows = state[key]
    for item in rows:
        if item["dataset"] == dataset:
            return _require_mapping(item.get("audit"), f"{key}.{dataset}.audit")
    return None


def _record_completed_audit(
    state: dict[str, Any], *, key: str, dataset: str, audit: Mapping[str, Any]
) -> None:
    existing = _completed_audit(state, key=key, dataset=dataset)
    if existing is None:
        state[key].append({"dataset": dataset, "audit": dict(audit)})
    else:
        require(existing == dict(audit), f"Stored {key} audit drift: {dataset}")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve()
    frozen_b_root = args.frozen_b_root.resolve()
    output = args.output_root.resolve()
    contract_path = project / CONTRACT_REL
    contract = read_json(contract_path)
    datasets = validate_contract(contract)
    manifest = verify_source(project, args.source_revision)
    taxi_input_record = verify_dataset_inputs(
        project, frozen_b_root, datasets[TAXI_DATASET]
    )
    source_manifest_path = project / "source_manifest.json"
    source_manifest_sha256 = sha256_file(source_manifest_path)
    contract_sha256 = sha256_file(contract_path)
    resume_identity = controller_resume_identity(
        project=project,
        frozen_b_root=frozen_b_root,
        source_revision=args.source_revision,
        source_manifest_sha256=source_manifest_sha256,
        contract_sha256=contract_sha256,
        taxi_input_record=taxi_input_record,
    )
    if args.verify_only:
        return {
            "status": "verified",
            "source_revision": args.source_revision,
            "source_manifest_sha256": source_manifest_sha256,
            "contract_sha256": contract_sha256,
            "verified_datasets": [TAXI_DATASET],
            "conditional_datasets_deferred_until_taxi_passes": list(
                POST_TAXI_DATASETS
            ),
            "held_out_test_evaluated": False,
        }
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "controller.lock"
    child_slot: dict[str, subprocess.Popen[str] | None] = {"child": None}

    def stop(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = deterministic_env(project)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = output / "status.json"
        existing_entries = [path for path in output.iterdir() if path != lock_path]
        if existing_entries:
            require(status_path.is_file(), "Existing output has no resumable status")
            state = validate_resumable_state(
                read_json(status_path),
                expected_identity=resume_identity,
                output=output,
            )
            input_records = dict(state["verified_input_manifest"])
            resuming = True
            resume_position = {
                "phase": state.get("phase"),
                "current_dataset": state.get("current_dataset"),
                "stage": state.get("stage"),
            }
        else:
            control_dir = output / "control"
            control_dir.mkdir(parents=True, exist_ok=False)
            copied_manifest = control_dir / "source_manifest.json"
            copied_contract = control_dir / CONTRACT_REL.name
            shutil.copy2(source_manifest_path, copied_manifest)
            shutil.copy2(contract_path, copied_contract)
            require(
                sha256_file(copied_manifest) == source_manifest_sha256,
                "Copied source manifest checksum drift",
            )
            require(
                sha256_file(copied_contract) == contract_sha256,
                "Copied contract checksum drift",
            )
            input_records = {TAXI_DATASET: taxi_input_record}
            state = {
                "schema_version": 1,
                "status": "starting",
                "source_revision": args.source_revision,
                "source_manifest_sha256": source_manifest_sha256,
                "source_manifest_file_count": len(manifest["files"]),
                "contract_sha256": contract_sha256,
                "controller_resume_identity": resume_identity,
                "control_evidence": {
                    "source_manifest_path": str(copied_manifest.resolve()),
                    "source_manifest_sha256": source_manifest_sha256,
                    "contract_path": str(copied_contract.resolve()),
                    "contract_sha256": contract_sha256,
                },
                "execution_order": list(EXECUTION_ORDER),
                "completed_e1": [],
                "completed_seed42": [],
                "screening_decisions": {},
                "verified_input_manifest": input_records,
                "conditional_inputs_deferred": list(POST_TAXI_DATASETS),
                "held_out_test_evaluated": False,
                "additional_seeds_executed": False,
                "resume_count": 0,
            }
            save_json(status_path, state)
            resuming = False
            resume_position = {}

        def update(**values: Any) -> None:
            state.update(values)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_json(status_path, state)

        def run_one(dataset: str, phase: str) -> dict[str, Any]:
            verify_source(project, args.source_revision)
            verify_input_fingerprint(input_records[dataset])
            preflight = gpu_preflight()
            run_output = output / phase / dataset
            if run_output.exists():
                require(
                    resuming
                    and resume_position.get("phase") == phase
                    and resume_position.get("current_dataset") == dataset
                    and resume_position.get("stage")
                    in {"training", "audit", "failed"},
                    f"Unexpected pre-existing run output: {run_output}",
                )
            update(
                phase=phase,
                current_dataset=dataset,
                stage="training",
                gpu_preflight=preflight,
            )
            command = runner_command(
                python=args.python,
                project=project,
                frozen_b_root=frozen_b_root,
                output=run_output,
                revision=args.source_revision,
                row=datasets[dataset],
                phase=phase,
            )
            run_logged(
                command,
                output / phase / dataset / "train.log",
                project=project,
                env=env,
                child_slot=child_slot,
            )
            update(stage="audit")
            return audit_run(
                run_output,
                row=datasets[dataset],
                contract_path=contract_path,
                source_revision=args.source_revision,
                phase=phase,
            )

        try:
            state.pop("error", None)
            update(
                status="running",
                phase="cuda_contract_tests",
                stage="testing",
                gpu_preflight=gpu_preflight(),
                verified_input_manifest=input_records,
                resume_count=int(state.get("resume_count", 0))
                + (1 if resuming else 0),
            )
            xml_path = output / "cuda_contract_tests.xml"
            if state.get("cuda_contract_tests") is None:
                test_command = [
                    args.python,
                    "-m",
                    "pytest",
                    "-q",
                    *(str(path) for path in CUDA_TESTS),
                    f"--junitxml={xml_path}",
                ]
                run_logged(
                    test_command,
                    output / "cuda_contract_tests.log",
                    project=project,
                    env=env,
                    child_slot=child_slot,
                )
                cuda_audit = audit_pytest(xml_path)
                update(cuda_contract_tests=cuda_audit, stage="complete")
            else:
                cuda_audit = audit_pytest(xml_path)
                require(
                    state["cuda_contract_tests"] == cuda_audit,
                    "Stored CUDA test audit drift",
                )
                update(stage="complete")

            stored_e1 = _completed_audit(
                state, key="completed_e1", dataset=TAXI_DATASET
            )
            if stored_e1 is None:
                e1_audit = run_one(TAXI_DATASET, "e1")
                _record_completed_audit(
                    state,
                    key="completed_e1",
                    dataset=TAXI_DATASET,
                    audit=e1_audit,
                )
            else:
                e1_audit = audit_run(
                    output / "e1" / TAXI_DATASET,
                    row=datasets[TAXI_DATASET],
                    contract_path=contract_path,
                    source_revision=args.source_revision,
                    phase="e1",
                )
                require(stored_e1 == e1_audit, "Stored Taxi e1 audit drift")
            update(completed_e1=state["completed_e1"], stage="complete")

            stored_taxi = _completed_audit(
                state, key="completed_seed42", dataset=TAXI_DATASET
            )
            if stored_taxi is None:
                taxi_audit = run_one(TAXI_DATASET, "seed42")
                _record_completed_audit(
                    state,
                    key="completed_seed42",
                    dataset=TAXI_DATASET,
                    audit=taxi_audit,
                )
            else:
                taxi_audit = audit_run(
                    output / "seed42" / TAXI_DATASET,
                    row=datasets[TAXI_DATASET],
                    contract_path=contract_path,
                    source_revision=args.source_revision,
                    phase="seed42",
                )
                require(stored_taxi == taxi_audit, "Stored Taxi screening audit drift")
            taxi_decision = taxi_audit["acceptance_decision"]
            require(isinstance(taxi_decision, Mapping), "Taxi decision is missing")
            stored_taxi_decision = state["screening_decisions"].get(TAXI_DATASET)
            if stored_taxi_decision is not None:
                require(
                    stored_taxi_decision == taxi_decision,
                    "Stored Taxi screening decision drift",
                )
            state["screening_decisions"][TAXI_DATASET] = taxi_decision
            update(
                completed_seed42=state["completed_seed42"],
                screening_decisions=state["screening_decisions"],
                stage="complete",
            )
            if taxi_decision["status"] != "passed":
                update(
                    status="stopped_after_taxi_screening_failure",
                    phase="complete",
                    current_dataset=None,
                    stage="complete",
                    stop_reason="Taxi failed at least one fixed acceptance gate",
                    conditional_datasets_executed=False,
                )
                return state

            for dataset in POST_TAXI_DATASETS:
                observed_input = verify_dataset_inputs(
                    project, frozen_b_root, datasets[dataset]
                )
                if dataset in input_records:
                    require(
                        input_records[dataset] == observed_input,
                        f"Stored input identity drift: {dataset}",
                    )
                else:
                    input_records[dataset] = observed_input
                if dataset in state["conditional_inputs_deferred"]:
                    state["conditional_inputs_deferred"].remove(dataset)
                update(
                    verified_input_manifest=input_records,
                    conditional_inputs_deferred=state["conditional_inputs_deferred"],
                    stage="input_verified",
                )
                stored_audit = _completed_audit(
                    state, key="completed_seed42", dataset=dataset
                )
                if stored_audit is None:
                    audit = run_one(dataset, "seed42")
                    _record_completed_audit(
                        state,
                        key="completed_seed42",
                        dataset=dataset,
                        audit=audit,
                    )
                else:
                    audit = audit_run(
                        output / "seed42" / dataset,
                        row=datasets[dataset],
                        contract_path=contract_path,
                        source_revision=args.source_revision,
                        phase="seed42",
                    )
                    require(
                        stored_audit == audit,
                        f"Stored screening audit drift: {dataset}",
                    )
                decision = audit["acceptance_decision"]
                require(isinstance(decision, Mapping), f"{dataset} decision is missing")
                stored_decision = state["screening_decisions"].get(dataset)
                if stored_decision is not None:
                    require(
                        stored_decision == decision,
                        f"Stored screening decision drift: {dataset}",
                    )
                state["screening_decisions"][dataset] = decision
                update(
                    completed_seed42=state["completed_seed42"],
                    screening_decisions=state["screening_decisions"],
                    stage="complete",
                )
            all_passed = all(
                item["status"] == "passed"
                for item in state["screening_decisions"].values()
            )
            update(
                status="complete" if all_passed else "complete_with_screening_failures",
                phase="complete",
                current_dataset=None,
                stage="complete",
                conditional_datasets_executed=True,
            )
            return state
        except BaseException as exc:
            child = child_slot["child"]
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            update(status="failed", error=repr(exc), stage="failed")
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-b-root",
        type=Path,
        required=True,
        help="Root containing immutable Frozen-B e1 caches and full checkpoints",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = execute(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
