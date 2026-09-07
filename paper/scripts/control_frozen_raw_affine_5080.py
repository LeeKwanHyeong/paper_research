#!/usr/bin/env python3
"""Run the prospectively gated frozen raw-affine pipeline on one RTX 5080."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "paper/scripts/run_frozen_raw_affine_calibration.py"
SUMMARIZER = ROOT / "paper/scripts/summarize_frozen_raw_affine_calibration.py"
CONTRACT = ROOT / "paper/contracts/frozen_raw_affine_calibration_v1.json"
B_DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)


def save_status(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[tuple[str, str], Path]:
    try:
        identity, raw_path = value.split("=", 1)
        dataset, role = identity.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint must be DATASET:ROLE=PATH") from error
    if not dataset or role not in {"B", "rmtpp", "thp"} or not raw_path:
        raise argparse.ArgumentTypeError("invalid checkpoint mapping")
    return (dataset, role), Path(raw_path)


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def dataset_rows(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["dataset"]): row for row in contract["datasets"]}


def run_logged(command: list[str], *, log_path: Path) -> None:
    contract = load_contract()
    expected = contract["runtime"]["deterministic_environment"]
    environment = os.environ.copy()
    environment.update({str(name): str(value) for name, value in expected.items()})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {log_path}")


def runner_command(
    args: argparse.Namespace,
    row: Mapping[str, Any],
    *,
    role: str,
    phase: str,
    output: Path,
    train_result: Path | None = None,
    prerequisite: Path | None = None,
    smoke: bool = False,
) -> list[str]:
    key = (str(row["dataset"]), role)
    command = [
        sys.executable,
        str(RUNNER),
        "--dataset",
        key[0],
        "--model-role",
        role,
        "--data",
        str(args.data_root / row["data_path"]),
        "--split-manifest",
        str(args.data_root / row["split_manifest_path"]),
        "--checkpoint",
        str(args.checkpoints[key]),
        "--output-dir",
        str(output),
        "--source-revision",
        args.source_revision,
        "--source-manifest",
        str(args.source_manifest),
        "--contract",
        str(CONTRACT),
        "--device",
        "cuda:0",
        "--phase",
        phase,
    ]
    if train_result is not None:
        command.extend(["--train-result", str(train_result)])
    if prerequisite is not None:
        command.extend(["--prerequisite-decision", str(prerequisite)])
    if smoke:
        command.append("--smoke")
    return command


def summarize_command(stage: str, results: list[Path], output: Path) -> list[str]:
    command = [sys.executable, str(SUMMARIZER), "--stage", stage]
    for result in results:
        command.extend(["--result", str(result)])
    command.extend(["--output", str(output)])
    return command


def run_train_smoke(args: argparse.Namespace, rows: Mapping[str, Mapping[str, Any]]) -> dict:
    outputs: list[Path] = []
    for dataset in B_DATASETS:
        train_output = args.artifact_root / "smoke" / "train" / f"{dataset}_B"
        run_logged(
            runner_command(
                args,
                rows[dataset],
                role="B",
                phase="train",
                output=train_output,
                smoke=True,
            ),
            log_path=args.artifact_root / "logs" / f"smoke_{dataset}_B_train.log",
        )
        outputs.append(train_output / "result.json")
    decision_path = args.artifact_root / "decisions" / "train_smoke.json"
    decision = {
        "status": "success",
        "stage": "B_three_dataset_train_cuda_smoke",
        "decision": "train_smoke_passed_for_full_train_gate",
        "execution_server": "5080",
        "source_revision": args.source_revision,
        "held_out_test_evaluated": False,
        "validation_loaded": False,
        "result_identities": [
            {
                "result_file": str(path.resolve()),
                "result_file_sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    save_status(decision_path, decision)
    return {
        "decision": decision["decision"],
        "artifact": str(decision_path),
        "artifact_sha256": sha256_file(decision_path),
    }


def run_validation_smoke(
    args: argparse.Namespace,
    rows: Mapping[str, Mapping[str, Any]],
    *,
    train_results: list[Path],
    prerequisite: Path,
) -> dict:
    outputs: list[Path] = []
    for dataset, train_result in zip(B_DATASETS, train_results, strict=True):
        validation_output = args.artifact_root / "smoke" / "validation" / f"{dataset}_B"
        run_logged(
            runner_command(
                args,
                rows[dataset],
                role="B",
                phase="validation",
                output=validation_output,
                train_result=train_result,
                prerequisite=prerequisite,
                smoke=True,
            ),
            log_path=args.artifact_root / "logs" / f"smoke_{dataset}_B_validation.log",
        )
        outputs.append(validation_output / "result.json")
    decision_path = args.artifact_root / "decisions" / "validation_smoke.json"
    decision = {
        "status": "success",
        "stage": "B_three_dataset_validation_cuda_smoke",
        "decision": "validation_smoke_passed_for_full_validation",
        "execution_server": "5080",
        "source_revision": args.source_revision,
        "held_out_test_evaluated": False,
        "prerequisite_decision_file": str(prerequisite.resolve()),
        "prerequisite_decision_file_sha256": sha256_file(prerequisite),
        "result_identities": [
            {
                "result_file": str(path.resolve()),
                "result_file_sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    save_status(decision_path, decision)
    return {
        "decision": decision["decision"],
        "artifact": str(decision_path),
        "artifact_sha256": sha256_file(decision_path),
    }


def read_decision(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_full(args: argparse.Namespace, rows: Mapping[str, Mapping[str, Any]]) -> dict:
    train_results = []
    for dataset in B_DATASETS:
        output = args.artifact_root / "full" / "train" / f"{dataset}_B"
        run_logged(
            runner_command(args, rows[dataset], role="B", phase="train", output=output),
            log_path=args.artifact_root / "logs" / f"full_{dataset}_B_train.log",
        )
        train_results.append(output / "result.json")
    b_train_decision = args.artifact_root / "decisions" / "B_train.json"
    run_logged(
        summarize_command("b-train", train_results, b_train_decision),
        log_path=args.artifact_root / "logs" / "summarize_B_train.log",
    )
    if not read_decision(b_train_decision)["accepted"]:
        return {"decision": "rejected_stop_before_validation", "artifact": str(b_train_decision)}

    validation_smoke = run_validation_smoke(
        args,
        rows,
        train_results=train_results,
        prerequisite=b_train_decision,
    )

    validation_results = []
    for dataset, train_result in zip(B_DATASETS, train_results, strict=True):
        output = args.artifact_root / "full" / "validation" / f"{dataset}_B"
        run_logged(
            runner_command(
                args,
                rows[dataset],
                role="B",
                phase="validation",
                output=output,
                train_result=train_result,
                prerequisite=b_train_decision,
            ),
            log_path=args.artifact_root / "logs" / f"full_{dataset}_B_validation.log",
        )
        validation_results.append(output / "result.json")
    b_validation_decision = args.artifact_root / "decisions" / "B_validation.json"
    run_logged(
        summarize_command("b-validation", validation_results, b_validation_decision),
        log_path=args.artifact_root / "logs" / "summarize_B_validation.log",
    )
    if not read_decision(b_validation_decision)["accepted"]:
        return {
            "decision": "rejected_stop_before_instacart_fairness",
            "artifact": str(b_validation_decision),
            "cuda_validation_smoke": validation_smoke,
        }

    instacart = rows["insta_market_basket"]
    for role in ("rmtpp", "thp"):
        train_output = args.artifact_root / "full" / "train" / f"insta_market_basket_{role}"
        run_logged(
            runner_command(
                args,
                instacart,
                role=role,
                phase="train",
                output=train_output,
                prerequisite=b_validation_decision,
            ),
            log_path=args.artifact_root / "logs" / f"full_insta_market_basket_{role}_train.log",
        )
        validation_output = (
            args.artifact_root / "full" / "validation" / f"insta_market_basket_{role}"
        )
        run_logged(
            runner_command(
                args,
                instacart,
                role=role,
                phase="validation",
                output=validation_output,
                train_result=train_output / "result.json",
                prerequisite=b_validation_decision,
            ),
            log_path=(
                args.artifact_root / "logs" / f"full_insta_market_basket_{role}_validation.log"
            ),
        )
        validation_results.append(validation_output / "result.json")
    final_decision = args.artifact_root / "decisions" / "final.json"
    run_logged(
        summarize_command("final", validation_results, final_decision),
        log_path=args.artifact_root / "logs" / "summarize_final.log",
    )
    return {
        "decision": read_decision(final_decision)["decision"],
        "artifact": str(final_decision),
        "cuda_validation_smoke": validation_smoke,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    args = parser.parse_args()
    checkpoint_entries = list(args.checkpoint)
    args.checkpoints = {key: path.resolve() for key, path in checkpoint_entries}
    if len(args.checkpoints) != len(checkpoint_entries):
        parser.error("duplicate checkpoint mapping")
    required = {(dataset, "B") for dataset in B_DATASETS}
    if args.mode == "full":
        required |= {("insta_market_basket", "rmtpp"), ("insta_market_basket", "thp")}
    if set(args.checkpoints) != required:
        parser.error(f"checkpoint scope drift: {set(args.checkpoints) ^ required}")
    args.artifact_root = args.artifact_root.resolve()
    args.data_root = args.data_root.resolve()
    args.source_manifest = args.source_manifest.resolve()
    if args.artifact_root.exists():
        parser.error(f"artifact root already exists: {args.artifact_root}")
    return args


def main() -> None:
    args = parse_args()
    args.artifact_root.mkdir(parents=True)
    status_path = args.artifact_root / "pipeline_status.json"
    started = datetime.now(timezone.utc).isoformat()
    save_status(status_path, {"status": "running", "mode": args.mode, "started_at": started})
    try:
        contract = load_contract()
        rows = dataset_rows(contract)
        train_smoke_result = run_train_smoke(args, rows)
        result = train_smoke_result if args.mode == "smoke" else {
            **run_full(args, rows),
            "cuda_train_smoke": train_smoke_result,
        }
        status = {
            "status": "success",
            "mode": args.mode,
            "source_revision": args.source_revision,
            "execution_server": "5080",
            "held_out_test_evaluated": False,
            "started_at": started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        save_status(status_path, status)
    except BaseException as error:
        save_status(
            status_path,
            {
                "status": "failed",
                "mode": args.mode,
                "error": f"{type(error).__name__}: {error}",
                "started_at": started,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    print(json.dumps(status, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
