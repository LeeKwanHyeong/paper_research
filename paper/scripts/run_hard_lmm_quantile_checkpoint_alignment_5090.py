#!/usr/bin/env python3
"""Run CUDA contracts, full-data e1, and seed42 e300 on one idle RTX 5090."""

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
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTRACT_REL = Path("paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.json")
TRAINER_REL = Path("paper/scripts/run_count_aware_tpp_backbone_control.py")
ROLE = "quantile_checkpoint_alignment"
BACKBONE = "titantpp"
CONTROL_VARIANT = "count_only_log_regression"
CANDIDATE_VARIANT = "count_only_quantile_adaptive_log_regression"
VARIANTS = (CONTROL_VARIANT, CANDIDATE_VARIANT)
DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)
CUDA_TESTS = (
    "simple_lab_test/search/tests/test_count_aware_quantile_cuda_contract.py",
    "simple_lab_test/search/tests/test_count_aware_quantile_adaptive_loss.py",
    "simple_lab_test/search/tests/test_count_aware_checkpoint_selection.py",
    "simple_lab_test/search/tests/test_count_aware_quantile_alignment_runner.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assert_finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")


def validate_contract(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(contract.get("contract_id") == "hard_lmm_quantile_checkpoint_alignment_v1", "Wrong contract")
    scope = contract["scope"]
    expected_scope = {
        "backbone": BACKBONE,
        "maximum_epochs": 300,
        "minimum_epochs": 40,
        "patience": 40,
        "screening_seed": 42,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for key, expected in expected_scope.items():
        require(scope.get(key) == expected, f"Scope drift: {key}")
    objective = contract["quantile_objective"]
    require(objective["statistics_split"] == "train only", "Quantiles must be train-only")
    require(objective["quantiles"] == [0.5, 0.9, 0.95, 0.99], "Quantiles drifted")
    require(objective["raw_bin_weights"] == [1.0, 1.0, 1.5, 2.0, 3.0], "Weights drifted")
    require(objective["strength"] == 1.0, "Adaptive strength drifted")
    selection = contract["checkpoint_selection"]
    require(selection["raw_monitor_history_key"] == "val_qty_rmse", "Monitor drifted")
    require(selection["other_metrics_in_selector"] is False, "Guardrails entered selector")
    execution = contract["execution"]
    require(execution["rtx5090_cuda_and_full_data_e1"] is True, "e1 is not authorized")
    require(execution["rtx5090_seed42_e300"] is True, "seed42 e300 is not authorized")
    require(execution["rtx5090_seeds52_and62"] is False, "Additional seeds are not authorized")
    require(execution["held_out_test"] is False, "Held-out must remain locked")
    runtime = contract["runtime_policy"]
    require(runtime["execution_server"] == "5090", "Wrong execution server")
    require(runtime["minimum_free_vram_mib"] == 12000, "VRAM threshold drifted")
    require(runtime["require_no_existing_compute_process"] is True, "GPU exclusivity disabled")
    require(runtime["require_gdm_inactive"] is True, "GDM guard disabled")
    datasets = {row["dataset"]: row for row in contract["datasets"]}
    require(tuple(datasets) == DATASETS, "Dataset order/scope drifted")
    for name, row in datasets.items():
        require(row["expected_train_targets"] > 0, f"Missing train count: {name}")
        require(row["expected_validation_targets"] > 0, f"Missing validation count: {name}")
    return datasets


def verify_source(project: Path, revision: str) -> dict[str, Any]:
    require(len(revision) == 40 and all(ch in "0123456789abcdef" for ch in revision), "Full source SHA required")
    manifest_path = project / "source_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("source_revision") == revision, "Source revision mismatch")
    require(manifest.get("held_out_test_evaluated") is False, "Source package opened held-out")
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "Source manifest is empty")
    for name, expected in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe manifest path: {name}")
        actual = project / relative
        require(actual.is_file(), f"Manifest file is missing: {name}")
        require(sha256_file(actual) == expected, f"Source/input checksum mismatch: {name}")
    require(str(CONTRACT_REL) in files and str(TRAINER_REL) in files, "Core files are not manifested")
    return manifest


def verify_inputs(project: Path, datasets: dict[str, dict[str, Any]]) -> None:
    for name, row in datasets.items():
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            path = project / row[path_key]
            require(path.is_file(), f"Missing {name} input: {path_key}")
            require(sha256_file(path) == row[hash_key], f"{name} {path_key} checksum mismatch")


def command_output(command: list[str], *, allowed: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode not in allowed:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}: {result.stderr}")
    return result.stdout.strip()


def gpu_preflight(contract: dict[str, Any]) -> dict[str, Any]:
    policy = contract["runtime_policy"]
    rows = command_output([
        "nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits",
    ]).splitlines()
    require(len(rows) == 1, "Expected exactly one GPU")
    name, raw_free = (token.strip() for token in rows[0].split(",", 1))
    free_mib = int(raw_free)
    require(policy["required_gpu_name_contains"] in name, f"Wrong GPU: {name}")
    require(free_mib >= policy["minimum_free_vram_mib"], f"Insufficient free VRAM: {free_mib} MiB")
    compute = command_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ])
    require(not compute, f"GPU already has compute processes: {compute}")
    gdm = command_output(["systemctl", "is-active", "gdm"], allowed=(0, 3))
    require(gdm == "inactive", f"GDM must be inactive, got {gdm!r}")
    return {"gpu_name": name, "free_vram_mib": free_mib, "compute_processes": [], "gdm": gdm}


def deterministic_env(contract: dict[str, Any], project: Path) -> dict[str, str]:
    values = {str(k): str(v) for k, v in contract["runtime_policy"]["deterministic_environment"].items()}
    return {
        **os.environ,
        **values,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(project),
        "COUNT_AWARE_QUANTILE_TEST_DEVICE": "cuda",
    }


def training_command(
    *,
    python: str,
    project: Path,
    output: Path,
    revision: str,
    row: dict[str, Any],
    phase: str,
) -> list[str]:
    require(phase in {"e1", "seed42_e300"}, f"Unknown phase: {phase}")
    epochs, minimum, patience = (1, 1, 1) if phase == "e1" else (300, 40, 40)
    return [
        python,
        str(project / TRAINER_REL),
        "--data", str(project / row["data_path"]),
        "--split-manifest", str(project / row["split_manifest_path"]),
        "--output-dir", str(output),
        "--source-revision", revision,
        "--execution-role", f"hard_lmm_quantile_alignment_{phase}_5090",
        "--dataset-contract", row["dataset"],
        "--model-role", ROLE,
        "--device", "cuda",
        "--epochs", str(epochs),
        "--min-epochs", str(minimum),
        "--early-stopping-patience", str(patience),
        "--batch-size", "128",
        "--lr", "0.001",
        "--lookback-weeks", str(row["lookback"]),
        "--max-seq-len", str(row["max_sequence_length"]),
        "--hidden-dim", "64",
        "--lambda-log-qty", "1",
        "--lambda-tail", "0",
        "--grad-clip", "1",
        "--backbones", BACKBONE,
        "--seeds", "42",
        "--quantity-variants", "log_mse,quantile_adaptive",
        "--checkpoint-monitor", "validation_raw_quantity_rmse",
        "--quantile-adaptive-strength", "1",
        "--time-head-mode", "legacy_clamped_rmtpp",
        "--time-scale", "3",
        "--time-w-max", str(10.0 / 3.0),
        "--time-intercept-limit", "300",
        "--allow-partial-contract",
    ]


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    project: Path,
    env: dict[str, str],
    child_slot: dict[str, subprocess.Popen[str] | None],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(
            command,
            cwd=project,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        child_slot["child"] = child
        code = child.wait()
        child_slot["child"] = None
    if code:
        raise subprocess.CalledProcessError(code, command)


def audit_cuda_tests(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    cases = list(root.iter("testcase"))
    require(bool(cases), "CUDA contract suite executed no tests")
    require(not list(root.iter("failure")), "CUDA contract suite failed")
    require(not list(root.iter("error")), "CUDA contract suite errored")
    require(not list(root.iter("skipped")), "CUDA contract suite skipped a test")
    return {
        "test_count": len(cases),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "xml_sha256": sha256_file(xml_path),
    }


def _weighted_body_mae(rows: list[dict[str, Any]]) -> float:
    body_names = {"le_p50", "p50_p90", "p90_p95"}
    body = [row for row in rows if row["stratum"] in body_names]
    require({row["stratum"] for row in body} == body_names and len(body) == 3, "Body strata are incomplete")
    count = sum(int(row["count"]) for row in body)
    require(count > 0, "Body strata are empty")
    return sum(float(row["qty_mae"]) * int(row["count"]) for row in body) / count


def audit_run(
    output: Path,
    *,
    row: dict[str, Any],
    revision: str,
    phase: str,
) -> dict[str, Any]:
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    launch = read_json(output / "launch_contract.json")
    assert_finite_tree(launch)
    epochs, minimum, patience = (1, 1, 1) if phase == "e1" else (300, 40, 40)
    expected = {
        "status": "complete",
        "model_role": ROLE,
        "dataset": row["dataset"],
        "data_sha256": row["data_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "quantity_variants": list(VARIANTS),
        "backbones": [BACKBONE],
        "seeds": [42],
        "expected_run_count": 2,
        "completed_run_count": 2,
        "epochs": epochs,
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": row["lookback"],
        "max_seq_len": row["max_sequence_length"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": revision,
        "partial_smoke": False,
    }
    for key, value in expected.items():
        require(launch.get(key) == value, f"{row['dataset']} launch mismatch: {key}")
    require(set(launch["split_rows"]) == {"train", "validation"}, "Held-out rows were materialized")
    require(launch["early_stopping"] == {
        **launch["early_stopping"],
        "monitor": "validation_raw_quantity_rmse",
        "min_epochs": minimum,
        "patience": patience,
        "restore": "best_validation_raw_quantity_rmse",
    }, "Raw-RMSE early stopping drifted")
    require(launch["time_head"]["time_intercept_limit"] == 300.0, "Legacy compatibility cap drifted")
    adaptive = launch["quantile_adaptive_contract"]
    require(adaptive["population"]["target_count"] == row["expected_train_targets"], "Train target count mismatch")
    require(adaptive["boundaries"] == row["expected_loss_boundaries"], "Train-only loss boundaries drifted")
    require(adaptive["bin_counts"] == row["expected_loss_bin_counts"], "Train-only loss bin counts drifted")
    require(adaptive["contract_sha256"] == row["expected_loss_contract_sha256"], "Adaptive contract digest drifted")
    require(math.isclose(adaptive["normalization_mean"], row["expected_loss_normalization_mean"], rel_tol=0.0, abs_tol=1e-15), "Adaptive normalization drifted")
    require(math.isclose(adaptive["normalized_train_target_mean"], 1.0, rel_tol=0.0, abs_tol=5e-16), "Weights are not normalized")
    require(adaptive["population"]["target_identity_sha256"] == row["expected_train_target_identity_sha256"], "Train target identity drifted")
    require(adaptive["population"]["target_quantity_sha256"] == row["expected_train_target_quantity_sha256"], "Train target quantities drifted")
    validation_population = launch["validation_target_population"]
    require(validation_population["target_count"] == row["expected_validation_targets"], "Validation target count mismatch")
    require(validation_population["target_identity_sha256"] == row["expected_validation_target_identity_sha256"], "Validation target identity drifted")
    require(validation_population["target_quantity_sha256"] == row["expected_validation_target_quantity_sha256"], "Validation target quantities drifted")
    require(not [path for path in output.rglob("*") if path.name.startswith("test_")], "Held-out artifact found")

    records: dict[str, dict[str, Any]] = {}
    strata_counts: dict[str, list[tuple[str, int]]] = {}
    for variant in VARIANTS:
        run_dir = output / "runs" / BACKBONE / variant / "seed_42"
        summary = read_json(run_dir / "summary.json")
        history = read_json(run_dir / "history.json")["history"]
        assert_finite_tree(summary)
        assert_finite_tree(history)
        require(summary["status"] == "success", f"{variant} did not complete")
        require(summary["variant"] == variant and summary["backbone"] == BACKBONE and summary["seed"] == 42, "Run identity mismatch")
        require(summary["checkpoint_monitor"] == "validation_raw_quantity_rmse", "Wrong checkpoint selector")
        require(summary["checkpoint_monitor_history_key"] == "val_qty_rmse", "Wrong checkpoint history key")
        require(summary["checkpoint_selection"] == "best_validation_raw_quantity_rmse", "Wrong checkpoint rule")
        require(summary["source_revision"] == revision and summary["source_revision_history"] == [revision], "Source/resume mismatch")
        require(
            isinstance(summary.get("initial_state_sha256"), str)
            and len(summary["initial_state_sha256"]) == 64,
            "Initial model state digest is missing",
        )
        require(summary["evaluation_scope"] == "validation_only" and summary["held_out_test_evaluated"] is False, "Held-out evaluated")
        require(summary["training_device"].startswith("cuda"), "Run did not train on CUDA")
        require(int(summary["cuda_peak_memory_allocated_bytes"]) > 0, "CUDA allocation was not recorded")
        require(int(summary["cuda_peak_memory_reserved_bytes"]) > 0, "CUDA reservation was not recorded")
        require(history and all(int(item["train_event_count"]) == row["expected_train_targets"] for item in history), "Partial train epoch")
        selected = min(enumerate(history), key=lambda item: (float(item[1]["val_qty_rmse"]), item[0]))[1]
        require(int(summary["best_epoch"]) == int(selected["epoch"]), "Best epoch is not earliest raw-RMSE minimum")
        require(math.isclose(float(summary["selected_metric_value"]), float(summary["best_val_qty_rmse"]), rel_tol=1e-10, abs_tol=1e-8), "Selected RMSE replay mismatch")
        quantity_rows = summary["quantity_rows"]
        require(sum(int(item["count"]) for item in quantity_rows) == row["expected_validation_targets"], "Partial validation evaluation")
        require({item["stratum"] for item in quantity_rows} == {"le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99"}, "Quantity strata incomplete")
        strata_counts[variant] = [(item["stratum"], int(item["count"])) for item in quantity_rows]
        checkpoint_path = run_dir / "best_val_qty_rmse_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(checkpoint["checkpoint_monitor"] == "validation_raw_quantity_rmse", "Checkpoint selector mismatch")
        require(canonical_state_dict_sha256(checkpoint["model_state_dict"]) == summary["checkpoint_state_sha256"], "Checkpoint state digest mismatch")
        last = torch.load(run_dir / "last_epoch_state.pt", map_location="cpu", weights_only=False)
        require(last["best_state_sha256"] == summary["checkpoint_state_sha256"], "Resume best-state mismatch")
        require(isinstance(last.get("rng_state"), dict) and isinstance(last.get("train_loader_generator_state"), torch.Tensor), "Exact resume state missing")
        require(bool(last["optimizer_state_dict"].get("state")), "Optimizer state is empty")
        records[variant] = {
            "best_epoch": int(summary["best_epoch"]),
            "completed_epochs": int(summary["completed_epochs"]),
            "raw_rmse": float(summary["best_val_qty_rmse"]),
            "raw_mae": float(summary["best_val_qty_mae"]),
            "body_mae": _weighted_body_mae(quantity_rows),
            "gt_p99_mae": float(next(item for item in quantity_rows if item["stratum"] == "gt_p99")["qty_mae"]),
            "time_nll": float(summary["best_val_time_nll"]),
            "checkpoint_state_sha256": summary["checkpoint_state_sha256"],
            "initial_state_sha256": summary["initial_state_sha256"],
            "cuda_peak_memory_allocated_bytes": int(summary["cuda_peak_memory_allocated_bytes"]),
            "cuda_peak_memory_reserved_bytes": int(summary["cuda_peak_memory_reserved_bytes"]),
        }
    require(strata_counts[CONTROL_VARIANT] == strata_counts[CANDIDATE_VARIANT], "B/C validation strata differ")
    require(
        records[CONTROL_VARIANT]["initial_state_sha256"]
        == records[CANDIDATE_VARIANT]["initial_state_sha256"],
        "B/C did not start from the identical model state",
    )
    require(records[CONTROL_VARIANT]["checkpoint_state_sha256"] != records[CANDIDATE_VARIANT]["checkpoint_state_sha256"], "Active adaptive training did not change the model state")
    report = {
        "schema_version": 1,
        "status": "passed",
        "phase": phase,
        "dataset": row["dataset"],
        "source_revision": revision,
        "train_target_count": row["expected_train_targets"],
        "validation_target_population": validation_population,
        "quantile_adaptive_contract": adaptive,
        "variants": records,
        "held_out_test_evaluated": False,
    }
    save_json(output / "run_audit.json", report)
    return report


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve()
    output = args.output_root.resolve()
    contract = read_json(project / CONTRACT_REL)
    datasets = validate_contract(contract)
    verify_source(project, args.source_revision)
    verify_inputs(project, datasets)
    if args.verify_only:
        return {"status": "verified", "source_revision": args.source_revision}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite or resume existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "starting",
        "source_revision": args.source_revision,
        "completed_e1": [],
        "completed_seed42_e300": [],
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
    env = deterministic_env(contract, project)
    lock_path = output / "controller.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            update(status="running", phase="cuda_contract_tests", gpu_preflight=gpu_preflight(contract))
            xml_path = output / "cuda_contract_tests.xml"
            test_command = [args.python, "-m", "pytest", "-q", *CUDA_TESTS, f"--junitxml={xml_path}"]
            run_logged(test_command, output / "cuda_contract_tests.log", project=project, env=env, child_slot=child_slot)
            update(cuda_contract_tests=audit_cuda_tests(xml_path))

            for phase in ("e1", "seed42_e300"):
                completed_key = "completed_e1" if phase == "e1" else "completed_seed42_e300"
                for dataset in DATASETS:
                    verify_source(project, args.source_revision)
                    verify_inputs(project, datasets)
                    preflight = gpu_preflight(contract)
                    run_output = output / phase / dataset / ROLE
                    require(not run_output.exists(), f"Fresh run output already exists: {run_output}")
                    update(phase=phase, current_dataset=dataset, stage="training", gpu_preflight=preflight)
                    command = training_command(
                        python=args.python,
                        project=project,
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
                    audit = audit_run(
                        run_output,
                        row=datasets[dataset],
                        revision=args.source_revision,
                        phase=phase,
                    )
                    state[completed_key].append({"dataset": dataset, "audit": audit})
                    update(**{completed_key: state[completed_key]}, stage="complete")
            update(status="complete", phase="complete", current_dataset=None, stage="complete")
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
