#!/usr/bin/env python3
"""Validate frozen TitanTPP-MAC source files and validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


BACKBONE = "titantpp_titans_mac"
VARIANT = "count_only_log_regression"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def save_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_finite(value: Any, *, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {location}: {value}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, location=f"{location}.{key}")
        return
    raise TypeError(f"Unsupported JSON value at {location}: {type(value)}")


def verify_source(project_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative_path, expected in contract[
        "frozen_training_file_sha256"
    ].items():
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Frozen source mismatch for {relative_path}: "
                f"expected={expected} observed={actual}"
            )
        observed[relative_path] = actual
    return {
        "status": "complete",
        "training_source_revision": contract["training_source_revision"],
        "verified_file_count": len(observed),
        "sha256": observed,
    }


def run_paths(run_root: Path, seed: int) -> tuple[Path, Path, Path]:
    leaf = run_root / "runs" / BACKBONE / VARIANT / f"seed_{seed}"
    return (
        run_root / "launch_contract.json",
        leaf / "summary.json",
        leaf / "history.json",
    )


def validate_run(
    *,
    run_root: Path,
    dataset: str,
    seed: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if dataset not in contract["datasets"]:
        raise ValueError(f"Unknown dataset: {dataset}")
    expected_dataset = contract["datasets"][dataset]
    if seed not in expected_dataset["seeds"]:
        raise ValueError(f"Unexpected seed for {dataset}: {seed}")

    launch_path, summary_path, history_path = run_paths(run_root, seed)
    launch = load_json(launch_path)
    summary = load_json(summary_path)
    history = load_json(history_path)
    expected_source = contract["training_source_revision"]

    expected_launch = {
        "backbones": [BACKBONE],
        "batch_size": 128,
        "completed_run_count": 1,
        "dataset": dataset,
        "epochs": 300,
        "evaluation_scope": "validation_only",
        "expected_run_count": 1,
        "grad_clip": 1.0,
        "held_out_test_evaluated": False,
        "hidden_dim": 64,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "lookback_weeks": expected_dataset["lookback"],
        "lr": 0.001,
        "max_seq_len": expected_dataset["max_sequence_length"],
        "quantity_variants": [VARIANT],
        "seeds": [seed],
        "source_revision": expected_source,
        "status": "complete",
    }
    for key, expected in expected_launch.items():
        if launch.get(key) != expected:
            raise ValueError(
                f"Launch contract mismatch for {key}: "
                f"expected={expected!r} observed={launch.get(key)!r}"
            )
    if launch["interfaces"][VARIANT]["quantity_loss"] != (
        "mse_on_log1p_quantity"
    ):
        raise ValueError("Quantity loss is not direct log-MSE")
    if launch["time_head"]["mode"] != "legacy_clamped_rmtpp":
        raise ValueError("Unexpected time head")
    early_stopping = launch["early_stopping"]
    if early_stopping["min_epochs"] != 40:
        raise ValueError("Unexpected minimum epoch contract")
    if early_stopping["patience"] != 40:
        raise ValueError("Unexpected early-stopping patience")
    if early_stopping["monitor"] != "validation_joint_objective":
        raise ValueError("Unexpected checkpoint monitor")

    if summary.get("backbone") != BACKBONE:
        raise ValueError("Unexpected summary backbone")
    if summary.get("variant") != VARIANT:
        raise ValueError("Unexpected summary variant")
    if summary.get("seed") != seed:
        raise ValueError("Unexpected summary seed")
    if summary.get("source_revision") != expected_source:
        raise ValueError("Unexpected summary source revision")
    if summary.get("evaluation_scope") != "validation_only":
        raise ValueError("Summary is not validation-only")
    if summary.get("held_out_test_evaluated") is not False:
        raise ValueError("Held-out test was evaluated")
    if summary.get("status") not in {"success", "complete"}:
        raise ValueError("Run summary is not complete")
    checkpoint_digest = summary.get("checkpoint_state_sha256", "")
    if len(checkpoint_digest) != 64:
        raise ValueError("Checkpoint state digest is missing")
    checkpoint_path = Path(summary["checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if len(history.get("history", [])) != summary["completed_epochs"]:
        raise ValueError("History length does not match completed epochs")

    forbidden = [
        path
        for path in run_root.rglob("*")
        if path.is_file()
        and ("test_summary" in path.name or "held_out" in path.name)
    ]
    if forbidden:
        raise ValueError(f"Held-out artifact found: {forbidden}")
    require_finite(launch, location="launch_contract")
    require_finite(summary, location="summary")
    require_finite(history, location="history")

    evidence = {
        "status": "complete",
        "model_name": contract["short_model_name"],
        "dataset": dataset,
        "seed": seed,
        "training_source_revision": expected_source,
        "completed_epochs": summary["completed_epochs"],
        "best_epoch": summary["best_epoch"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "checkpoint_state_sha256": checkpoint_digest,
        "all_metrics_finite": True,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    save_json_atomic(run_root / "titantpp_mac_validation.json", evidence)
    return evidence


def finalize(output_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    results = []
    for dataset, seed in contract["run_order"]:
        run_root = output_root / "shards" / dataset / f"seed_{seed}"
        results.append(
            validate_run(
                run_root=run_root,
                dataset=dataset,
                seed=int(seed),
                contract=contract,
            )
        )
    if len(results) != contract["run_count"]:
        raise ValueError("Validated run count does not match contract")
    payload = {
        "status": "complete",
        "model_name": contract["short_model_name"],
        "training_source_revision": contract["training_source_revision"],
        "validated_run_count": len(results),
        "held_out_test_evaluated": False,
        "runs": results,
    }
    save_json_atomic(output_root / "validation_summary.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("verify-source")
    source.add_argument("--project-root", type=Path, required=True)
    source.add_argument("--contract", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("validate-run")
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--contract", type=Path, required=True)

    complete = subparsers.add_parser("finalize")
    complete.add_argument("--output-root", type=Path, required=True)
    complete.add_argument("--contract", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--output-root", type=Path, required=True)
    status.add_argument("--state", required=True)
    status.add_argument("--orchestration-revision", required=True)
    status.add_argument("--current-dataset", default=None)
    status.add_argument("--current-seed", type=int, default=None)
    status.add_argument("--completed-run-count", type=int, required=True)
    status.add_argument("--message", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify-source":
        contract = load_json(args.contract)
        payload = verify_source(args.project_root, contract)
        save_json_atomic(args.output, payload)
    elif args.command == "validate-run":
        contract = load_json(args.contract)
        validate_run(
            run_root=args.run_root,
            dataset=args.dataset,
            seed=args.seed,
            contract=contract,
        )
    elif args.command == "finalize":
        contract = load_json(args.contract)
        finalize(args.output_root, contract)
    elif args.command == "status":
        payload = {
            "status": args.state,
            "model_name": "TitanTPP-MAC",
            "training_source_revision": (
                "08e59880cd61cbd27cec40aa04636452b87bebfc"
            ),
            "orchestration_revision": args.orchestration_revision,
            "execution_server": "5090",
            "current_dataset": args.current_dataset,
            "current_seed": args.current_seed,
            "completed_run_count": args.completed_run_count,
            "total_run_count": 9,
            "held_out_test_evaluated": False,
            "message": args.message,
        }
        save_json_atomic(args.output_root / "status.json", payload)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
