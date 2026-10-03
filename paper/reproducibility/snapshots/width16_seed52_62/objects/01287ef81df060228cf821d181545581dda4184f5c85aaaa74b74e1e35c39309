#!/usr/bin/env python3
# ruff: noqa: E402
"""Validate, then explicitly execute, a frozen legacy J/Q/T diagnostic contract.

This experimental entrypoint does not change the qualified benchmark runner.
The legacy clamped time score is a compatibility diagnostic, not a normalized
observation-likelihood or benchmark-superiority experiment.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT, TIME_HEAD_MODE_LEGACY_CLAMPED
from paper.scripts.count_aware_tpp_backbone.core import (
    load_train_validation_frame,
    prepare_count_frame,
)
from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS
from paper.scripts.run_count_aware_tpp_backbone_control import exact_target_population
from paper.scripts.run_taxi_quantity_interface_ablation import make_loader, sha256_file
from paper.scripts.run_taxi_quantity_interface_ablation import set_seed as _legacy_set_seed
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

TRACK = "legacy_cap300_compatibility_diagnostic"
OBJECTIVES = ("joint", "quantity_only", "time_only")
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
FIXED_MODEL = {
    "backbone": "titantpp",
    "hidden_dim": 64,
    "quantity_variant": LOG_MSE_VARIANT,
    "time_head_mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
    "time_intercept_limit": 300.0,
    "time_scale": 3.0,
    "time_w_max": 10.0 / 3.0,
    "lambda_tail": 0.0,
}
FIXED_LOADER = {
    "num_workers": 0,
    "drop_last": False,
    "train_shuffle": True,
    "validation_shuffle": False,
}
POPULATION_KEYS = {"target_count", "target_identity_sha256", "target_quantity_sha256"}
FIRST_PARTY_PACKAGES = {"models", "paper", "simple_lab_test", "data_loader", "utils"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _keys(value: Any, expected: set[str], label: str) -> None:
    _require(isinstance(value, dict) and set(value) == expected,
             f"{label} must contain exactly {sorted(expected)}")


def _positive_int(value: Any, label: str) -> None:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer")


def _digest(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be a lowercase SHA256")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def current_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def source_file_hashes() -> dict[str, str]:
    """Hash the transitive first-party Python import closure, including both new files.

    AST traversal includes conditional imports, so changing the active runtime
    cannot silently omit model, loader or reproducibility dependencies. Dynamic
    optional dependencies are recorded separately by the runtime contract.
    """
    roots = [
        "paper/scripts/run_time_quantity_diagnostic.py",
        "paper/scripts/time_quantity_diagnostic.py",
    ]
    pending = [PROJECT_ROOT / name for name in roots]
    visited: set[Path] = set()

    def add_module(module: str) -> None:
        if not module or module.split(".")[0] not in FIRST_PARTY_PACKAGES:
            return
        stem = PROJECT_ROOT.joinpath(*module.split("."))
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if candidate.is_file():
                pending.append(candidate)
        parent = stem.parent
        while parent != PROJECT_ROOT:
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer)
            parent = parent.parent

    while pending:
        path = pending.pop().resolve()
        _require(path.is_relative_to(PROJECT_ROOT), "Source dependency escapes repository")
        if path in visited:
            continue
        _require(path.is_file(), f"Source dependency missing: {path}")
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(PROJECT_ROOT)
        package = list(relative.parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package[:len(package) - node.level + 1]
                    module = ".".join(prefix + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                add_module(module)
                for alias in node.names:
                    if alias.name != "*":
                        add_module(f"{module}.{alias.name}")
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in sorted(visited)}


def current_runtime(device: str = "cpu") -> dict[str, Any]:
    parsed = torch.device(device)
    _require(parsed.type in {"cpu", "cuda"}, "Only explicit CPU or CUDA execution is supported")
    if parsed.type == "cuda":
        _require(torch.cuda.is_available(), "Requested CUDA runtime is unavailable")
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "polars": str(pl.__version__),
        "cuda": torch.version.cuda,
        "device_type": parsed.type,
        "device": str(parsed),
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }


def configure_runtime(device: str = "cpu", threads: int = 4) -> None:
    """Apply explicit CPU thread limits before pinning the runtime identity."""
    _positive_int(threads, "threads")
    _require(torch.device(device).type in {"cpu", "cuda"}, "Unsupported device")
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)


def reset_seed(seed: int) -> torch.Generator:
    """Keep the existing initialization procedure with strict deterministic ops."""
    generator = _legacy_set_seed(seed)
    # The legacy helper enables warn-only determinism; this diagnostic fails
    # closed on unsupported nondeterministic operations.
    torch.use_deterministic_algorithms(True, warn_only=False)
    return generator


def validate_import_origins() -> None:
    """Reject shadowed first-party modules imported from another checkout."""
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in FIRST_PARTY_PACKAGES or module is None:
            continue
        filename = getattr(module, "__file__", None)
        if filename is not None:
            _require(Path(filename).resolve().is_relative_to(PROJECT_ROOT),
                     f"First-party import origin escapes pinned repository: {name}")
        # Namespace packages may lack a file but still include an external
        # checkout on their search path. Test doubles without either are inert.
        for entry in getattr(module, "__path__", ()):
            _require(Path(entry).resolve().is_relative_to(PROJECT_ROOT),
                     f"First-party namespace origin escapes pinned repository: {name}")


def validate_contract_structure(contract: Any) -> None:
    _keys(contract, {
        "schema_version", "track", "dataset_id", "seed", "epochs", "objectives",
        "data", "split_manifest", "source", "runtime", "loader", "model", "optimizer",
        "populations",
    }, "Contract")
    _require(type(contract["schema_version"]) is int and contract["schema_version"] == 1,
             "Unsupported schema_version")
    _require(contract["track"] == TRACK, "Only legacy compatibility diagnosis is supported")
    _require(contract["dataset_id"] in DATASETS, "Unsupported frozen dataset")
    _require(type(contract["seed"]) is int and 0 <= contract["seed"] < 2**32,
             "seed must be an integer in [0, 2**32)")
    _positive_int(contract["epochs"], "epochs")
    _require(contract["objectives"] == list(OBJECTIVES), "All J/Q/T objectives in fixed order are required")
    for field in ("data", "split_manifest"):
        _keys(contract[field], {"path", "sha256"}, field)
        _require(isinstance(contract[field]["path"], str) and bool(contract[field]["path"]),
                 f"{field}.path is required")
        _digest(contract[field]["sha256"], f"{field}.sha256")
    _keys(contract["source"], {"revision", "files"}, "source")
    _require(isinstance(contract["source"]["revision"], str)
             and len(contract["source"]["revision"]) == 40,
             "source.revision requires the full Git revision")
    _require(isinstance(contract["source"]["files"], dict)
             and bool(contract["source"]["files"]), "source.files cannot be empty")
    for path, digest in contract["source"]["files"].items():
        _require(isinstance(path, str) and not Path(path).is_absolute()
                 and ".." not in Path(path).parts, "Source paths must be repository relative")
        _digest(digest, f"source.files[{path}]")
    _require(isinstance(contract["runtime"], dict), "runtime must be an object")
    _keys(contract["model"], set(FIXED_MODEL), "model")
    _require(contract["model"] == FIXED_MODEL, "Static B / unweighted log-MSE / legacy cap300 model drift")
    loader = contract["loader"]
    _keys(loader, set(FIXED_LOADER) | {"batch_size", "lookback_weeks", "max_seq_len"}, "loader")
    for key in ("batch_size", "lookback_weeks", "max_seq_len"):
        _positive_int(loader[key], f"loader.{key}")
    for key, value in FIXED_LOADER.items():
        _require(type(loader[key]) is type(value) and loader[key] == value,
                 f"Fixed loader setting drift: {key}")
    registry = DATASET_CONTRACTS[contract["dataset_id"]]
    _require(loader["lookback_weeks"] == registry["lookback"]
             and loader["max_seq_len"] == registry["max_seq_len"], "Frozen context window drift")
    _require(contract["data"]["sha256"] == registry["data_sha256"], "Data digest differs from frozen registry")
    _require(contract["split_manifest"]["sha256"] == registry["split_manifest_sha256"],
             "Split digest differs from frozen registry")
    _keys(contract["optimizer"], {"lr", "weight_decay", "grad_clip"}, "optimizer")
    for key, value in contract["optimizer"].items():
        _require(type(value) in (int, float) and math.isfinite(value)
                 and (value >= 0 if key == "weight_decay" else value > 0),
                 f"Invalid optimizer.{key}")
    _keys(contract["populations"], {"train", "validation"}, "populations")
    for split, population in contract["populations"].items():
        _keys(population, POPULATION_KEYS, f"populations.{split}")
        _positive_int(population["target_count"], f"{split} target_count")
        for key in POPULATION_KEYS - {"target_count"}:
            _digest(population[key], f"{split}.{key}")


def _resolved_input(contract_path: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else contract_path.parent / candidate).resolve()


def validate_inputs(contract_path: Path, contract: dict[str, Any], device: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    validate_contract_structure(contract)
    validate_import_origins()
    _require(contract["source"]["revision"] == current_revision(), "Source revision mismatch")
    _require(contract["source"]["files"] == source_file_hashes(), "Source file closure/digest mismatch")
    reset_seed(contract["seed"])
    _require(contract["runtime"] == current_runtime(device), "Runtime identity mismatch")
    paths: dict[str, str] = {}
    for field in ("data", "split_manifest"):
        path = _resolved_input(contract_path, contract[field]["path"])
        _require(path.is_file(), f"Missing {field}: {path}")
        _require(sha256_file(path) == contract[field]["sha256"], f"{field} file digest mismatch")
        paths[field] = str(path)
    raw = load_train_validation_frame(Path(paths["data"]))
    _require(raw.height > 0, "No train/validation rows")
    _require(set(raw["chronological_split"].unique().to_list()) == {"train", "validation"},
             "Both train and validation are required; held-out rows are prohibited")
    for column in ("demand_qty", "delta_t", "seq"):
        values = raw[column].to_numpy()
        _require(np.isfinite(values).all(), f"Nonfinite observed {column}")
    _require(raw.select(pl.struct(["oper_part_no", "seq"]).n_unique()).item() == raw.height,
             "Duplicate series/sequence identities")
    # A train target must never use a preceding validation observation.
    bad_order = raw.with_columns(
        (pl.col("chronological_split") == "validation").cum_max().over("oper_part_no").alias("seen_validation")
    ).filter((pl.col("chronological_split") == "train") & pl.col("seen_validation"))
    _require(bad_order.height == 0, "Validation precedes a training target")
    frame = prepare_count_frame(raw)
    loader = contract["loader"]
    populations = {}
    for split in ("train", "validation"):
        _, population = exact_target_population(
            frame, target_split=split, lookback_weeks=loader["lookback_weeks"],
            max_seq_len=loader["max_seq_len"],
        )
        observed = {key: population[key] for key in POPULATION_KEYS}
        _require(observed == contract["populations"][split], f"{split} target population mismatch")
        populations[split] = population
    train_values = raw.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
    train_log_values = np.log1p(train_values)
    _require(np.isfinite(train_log_values).all() and train_log_values.mean() > 0
             and train_log_values.std() > 0, "Train-only quantity initialization must be finite and nondegenerate")
    metadata = {
        "resolved_inputs": paths,
        "populations": populations,
        "train_log_mean": float(train_log_values.mean()),
        "train_log_std": float(train_log_values.std()),
        "initialization_statistics_scope": "all_train_rows_matching_frozen_runner",
        "held_out_materialized": False,
        "loader_contract": {
            **loader, "dataset": "RMTPPWeekLookbackDataset", "mode": "all",
            "target_splits": ["train", "validation"], "clip_dt_min1": True,
            "delta_t_transform": "Int32 then clamp_min(1) then Float32",
            "quantity_storage": "Float32", "padding": "left", "target_appended": True,
            "context_policy": "all preceding permitted train/validation observations",
        },
    }
    return frame, metadata


def build_arm_inputs(contract: dict[str, Any], frame: pl.DataFrame, metadata: dict[str, Any]):
    """Reset initialization and both loader RNGs for each arm and resumed arm."""
    generator = reset_seed(contract["seed"])
    settings = contract["model"]
    model, encoder_config = build_count_aware_model(
        **settings, train_log_mean=metadata["train_log_mean"],
        train_log_std=metadata["train_log_std"], max_seq_len=contract["loader"]["max_seq_len"],
    )
    loader = contract["loader"]
    shared = {"batch_size": loader["batch_size"], "lookback_weeks": loader["lookback_weeks"],
              "max_seq_len": loader["max_seq_len"]}
    train_loader = make_loader(frame, target_split="train", shuffle=True, generator=generator, **shared)
    validation_generator = torch.Generator().manual_seed(contract["seed"])
    validation_loader = make_loader(frame, target_split="validation", shuffle=False,
                                    generator=validation_generator, **shared)
    return model, train_loader, validation_loader, encoder_config


def _persist_identical(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        _require(json.loads(path.read_text(encoding="utf-8")) == payload,
                 f"Existing artifact identity differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")


def audit_arm_comparison(summaries: dict[str, Any], *, initial_hash: str,
                         epochs_budget: int) -> dict[str, Any]:
    """Verify paired exposure and compare only corresponding task selectors."""
    _keys(summaries, set(OBJECTIVES), "Arm summaries")
    baseline = summaries["joint"]
    completed = baseline.get("epochs_completed")
    _positive_int(completed, "Completed epochs")
    _require(completed <= epochs_budget, "Completed epochs exceed fixed budget")
    exposure_fields = ("epoch", "global_step", "train_batch_order_sha256", "train_count",
                       "train_batches", "validation_count", "validation_batches")
    for objective, summary in summaries.items():
        _require(summary.get("objective") == objective, "Summary objective mismatch")
        _require(summary.get("initial_state_sha256") == initial_hash, "Cross-arm initialization mismatch")
        _require(summary.get("epochs_budget") == epochs_budget
                 and summary.get("epochs_completed") == completed,
                 "Cross-arm epoch budget/progress mismatch")
        _require(summary.get("status") == ("complete" if completed == epochs_budget else "paused_at_epoch_boundary"),
                 "Arm status differs from fixed budget progress")
        _require(summary.get("global_step") == baseline.get("global_step"), "Cross-arm global steps differ")
        history = summary.get("history")
        _require(isinstance(history, list) and len(history) == completed
                 and [row.get("epoch") for row in history] == list(range(1, completed + 1)),
                 "Arm history epoch sequence mismatch")
        _require(history[-1]["global_step"] == summary["global_step"], "Summary/history step mismatch")
        expected_steps = 0
        for row, reference in zip(history, baseline["history"], strict=True):
            for field in ("global_step", "train_count", "train_batches", "validation_count", "validation_batches"):
                _positive_int(row.get(field), f"History {field}")
            _digest(row.get("train_batch_order_sha256"), "Training order SHA256")
            expected_steps += row["train_batches"]
            _require(row["global_step"] == expected_steps, "History step total differs from processed batches")
            _require(all(field in row and field in reference and row[field] == reference[field]
                         for field in exposure_fields), "Cross-arm sample order/count/step budget mismatch")
        _keys(summary.get("selectors"), {"raw_quantity_rmse", "legacy_time_loss"}, "Arm selectors")
        for selector, selected in summary["selectors"].items():
            applicable = (selector == "raw_quantity_rmse" and objective != "time_only") or (
                selector == "legacy_time_loss" and objective != "quantity_only")
            _require(selected.get("applicable") is applicable, "Task selector applicability mismatch")
            if not applicable:
                _require(selected.get("best_epoch") is None and selected.get("best_value") is None,
                         "Inactive task has selected metrics")
                continue
            _require(all(type(row.get(selector)) in (int, float) and math.isfinite(row[selector])
                         for row in history), "Nonfinite/missing paired task metric")
            best = min(history, key=lambda row: row[selector])
            _require(selected.get("best_epoch") == best["epoch"]
                     and selected.get("best_value") == best[selector]
                     and selected.get("global_step") == best["global_step"],
                     "Paired comparison selector is not the earliest strict minimum")
            _digest(selected.get("state_sha256"), "Selected state SHA256")
    comparisons = {}
    for task, objective, selector in (("quantity", "quantity_only", "raw_quantity_rmse"),
                                      ("time", "time_only", "legacy_time_loss")):
        joint = baseline["selectors"][selector]
        single = summaries[objective]["selectors"][selector]
        comparisons[task] = {
            "metric": selector, "lower_is_better": True,
            "joint": {"objective": "joint", "selector": selector, **joint},
            "single_task": {"objective": objective, "selector": selector, **single},
            "single_task_minus_joint": single["best_value"] - joint["best_value"],
        }
    return {
        "schema_version": 1, "track": TRACK, "equal_training_exposure_verified": True,
        "initial_state_sha256": initial_hash, "epochs_budget": epochs_budget,
        "epochs_completed": completed, "global_step": baseline["global_step"],
        "verified_exposure_fields": list(exposure_fields), "paired_comparisons": comparisons,
        "joint_checkpoint_policy": "Each task uses its own J selector; different J checkpoints are not combined into one model score.",
        "evaluation_scope": "validation_only", "normalized_likelihood_claim": False,
    }


def run_contract(*, contract_path: Path, output_dir: Path, device: str = "cpu",
                 execute: bool = False, resume: bool = False,
                 stop_after_epochs: int | None = None, threads: int = 4) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    output_dir = output_dir.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(execute or not resume, "--resume requires --execute")
    _require(execute or stop_after_epochs is None, "--stop-after-epochs requires --execute")
    if stop_after_epochs is not None:
        _positive_int(stop_after_epochs, "stop_after_epochs")
    configure_runtime(device, threads)
    frame, metadata = validate_inputs(contract_path, contract, device)
    _require(stop_after_epochs is None or stop_after_epochs <= contract["epochs"],
             "stop_after_epochs exceeds fixed epoch budget")
    model, _, _, encoder_config = build_arm_inputs(contract, frame, metadata)
    initial_hash = canonical_state_dict_sha256(model.state_dict())
    del model
    identity = {
        "contract_sha256": hashlib.sha256(_json_bytes(contract)).hexdigest(),
        "contract_file_sha256": sha256_file(contract_path),
        "contract": contract,
        **metadata,
        "encoder_config": encoder_config,
        "initial_state_sha256": initial_hash,
        "runtime": current_runtime(device),
        "evaluation_scope": "validation_only",
        "claim_scope": "legacy_loss_compatibility_diagnostic_no_normalized_likelihood_claim",
    }
    manifest_path = output_dir / "wrapper_manifest.json"
    if output_dir.exists():
        allowed = {"wrapper_manifest.json", *OBJECTIVES}
        _require(all(path.name in allowed or re.fullmatch(r"paired_comparison_epoch_\d{6,}\.json", path.name)
                     for path in output_dir.iterdir()),
                 "Output directory contains unrelated artifacts")
        _require(manifest_path.exists() or not any(output_dir.iterdir()),
                 "Existing output has no wrapper identity")
    _persist_identical(manifest_path, identity)
    if not execute:
        return {"status": "validated_only", "training_executed": False,
                "manifest": str(manifest_path), "initial_state_sha256": initial_hash,
                "populations": metadata["populations"]}
    from paper.scripts.time_quantity_diagnostic import run_arm
    # Validate every destination before starting the first arm.
    for objective in OBJECTIVES:
        arm_dir = output_dir / objective
        _require(not arm_dir.is_symlink(), "Arm output must not be a symlink")
        if arm_dir.exists():
            _require(arm_dir.is_dir(), "Arm output is not a directory")
            if any(arm_dir.iterdir()):
                _require(resume, "Existing arm artifacts require --resume")
                _require((arm_dir / "last_epoch_state.pt").is_file(),
                         "Incomplete arm has no resumable checkpoint; use a new output directory")
    results = {}
    for objective in OBJECTIVES:
        model, train_loader, validation_loader, config = build_arm_inputs(contract, frame, metadata)
        _require(canonical_state_dict_sha256(model.state_dict()) == initial_hash
                 and config == encoder_config, "Per-arm initialization drift")
        arm_dir = output_dir / objective
        arm_resume = resume and (arm_dir / "last_epoch_state.pt").is_file()
        results[objective] = run_arm(
            model=model, train_loader=train_loader, validation_loader=validation_loader,
            objective=objective, output_dir=arm_dir, epochs=contract["epochs"],
            seed=contract["seed"], identity=identity, device=device,
            **contract["optimizer"], resume=arm_resume, stop_after_epochs=stop_after_epochs,
        )
    comparison = audit_arm_comparison(results, initial_hash=initial_hash, epochs_budget=contract["epochs"])
    comparison_path = output_dir / f"paired_comparison_epoch_{comparison['epochs_completed']:06d}.json"
    _persist_identical(comparison_path, comparison)
    return {"status": "execution_returned", "training_executed": True,
            "manifest": str(manifest_path), "arms": results,
            "paired_comparison": comparison, "paired_comparison_path": str(comparison_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="Pinned CPU intra-op threads; inter-op is fixed to 1")
    parser.add_argument("--execute", action="store_true", help="Explicitly run the frozen J/Q/T training contract")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int)
    args = parser.parse_args(argv)
    result = run_contract(contract_path=args.contract, output_dir=args.output_dir,
                          device=args.device, execute=args.execute, resume=args.resume,
                          stop_after_epochs=args.stop_after_epochs, threads=args.threads)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
