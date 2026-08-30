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
LEGACY_TRAINING_SOURCE_REVISION = (
    "08e59880cd61cbd27cec40aa04636452b87bebfc"
)


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


def resolve_status_training_revision(
    *,
    output_root: Path,
    explicit_revision: str | None,
) -> str:
    """Resolve status provenance without hard-coding the active contract."""
    if explicit_revision is not None:
        revision = explicit_revision
    else:
        revisions: set[str] = set()
        for manifest in output_root.glob("source_manifest*.txt"):
            for line in manifest.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key == "training_source_revision":
                    revisions.add(value.strip())
        if len(revisions) > 1:
            raise ValueError(
                f"Conflicting training revisions in source manifests: "
                f"{sorted(revisions)}"
            )
        revision = (
            next(iter(revisions))
            if revisions
            else LEGACY_TRAINING_SOURCE_REVISION
        )
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"Invalid training source revision: {revision!r}")
    return revision


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


def validate_split_contract(
    *,
    contract: dict[str, Any],
    split_contract: dict[str, Any],
    contract_path: Path | None = None,
) -> dict[str, Any]:
    if split_contract["parent_validation_contract"] != contract["contract_id"]:
        raise ValueError("Split contract points to an unexpected parent")
    if split_contract["training_source_revision"] != contract[
        "training_source_revision"
    ]:
        raise ValueError("Split and parent training revisions differ")
    if split_contract["implementation_backbone"] != contract[
        "implementation_backbone"
    ]:
        raise ValueError("Split and parent backbones differ")
    if contract_path is not None:
        expected_digest = split_contract["parent_validation_contract_sha256"]
        observed_digest = sha256_file(contract_path)
        if observed_digest != expected_digest:
            raise ValueError(
                "Parent validation contract digest mismatch: "
                f"expected={expected_digest} observed={observed_digest}"
            )

    parent_order = [tuple(run) for run in contract["run_order"]]
    canonical_order = [
        tuple(run) for run in split_contract["canonical_run_order"]
    ]
    if canonical_order != parent_order:
        raise ValueError("Canonical split order differs from parent run order")
    if split_contract["canonical_run_count"] != contract["run_count"]:
        raise ValueError("Canonical split run count differs from parent")

    partitioned_runs: list[tuple[str, int]] = []
    for shard_id, shard in split_contract["shards"].items():
        shard_runs = [tuple(run) for run in shard["run_order"]]
        if shard["run_count"] != len(shard_runs):
            raise ValueError(f"Shard run count mismatch: {shard_id}")
        if not shard.get("execution_server"):
            raise ValueError(f"Shard execution server missing: {shard_id}")
        partitioned_runs.extend(shard_runs)

    if len(partitioned_runs) != len(set(partitioned_runs)):
        raise ValueError("Split shards contain duplicate runs")
    if set(partitioned_runs) != set(parent_order):
        raise ValueError("Split shard union differs from parent run grid")
    return {
        "status": "complete",
        "contract_id": split_contract["contract_id"],
        "training_source_revision": contract["training_source_revision"],
        "shard_count": len(split_contract["shards"]),
        "canonical_run_count": len(parent_order),
        "partition_is_disjoint": True,
        "partition_union_matches_parent": True,
    }


def run_paths(run_root: Path, seed: int) -> tuple[Path, Path, Path]:
    leaf = run_root / "runs" / BACKBONE / VARIANT / f"seed_{seed}"
    return (
        run_root / "launch_contract.json",
        leaf / "summary.json",
        leaf / "history.json",
    )


def resolve_checkpoint_path(summary_path: Path, recorded_path: str) -> Path:
    checkpoint_path = Path(recorded_path)
    if checkpoint_path.is_file():
        return checkpoint_path
    synchronized_path = summary_path.parent / checkpoint_path.name
    if synchronized_path.is_file():
        return synchronized_path
    raise FileNotFoundError(
        "Checkpoint is missing at both the recorded and synchronized paths: "
        f"recorded={checkpoint_path} synchronized={synchronized_path}"
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
    resolve_checkpoint_path(summary_path, summary["checkpoint_path"])
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


def finalize_shard(
    *,
    output_root: Path,
    contract: dict[str, Any],
    split_contract: dict[str, Any],
    shard_id: str,
) -> dict[str, Any]:
    if shard_id not in split_contract["shards"]:
        raise ValueError(f"Unknown split shard: {shard_id}")
    validate_split_contract(contract=contract, split_contract=split_contract)
    shard = split_contract["shards"][shard_id]
    results = []
    for dataset, seed in shard["run_order"]:
        run_root = output_root / "shards" / dataset / f"seed_{seed}"
        evidence = validate_run(
            run_root=run_root,
            dataset=dataset,
            seed=int(seed),
            contract=contract,
        )
        evidence["execution_server"] = shard["execution_server"]
        evidence["artifact_run_root"] = str(run_root)
        results.append(evidence)
    payload = {
        "status": "complete",
        "model_name": split_contract["short_model_name"],
        "shard_id": shard_id,
        "execution_server": shard["execution_server"],
        "training_source_revision": contract["training_source_revision"],
        "validated_run_count": len(results),
        "held_out_test_evaluated": False,
        "runs": results,
    }
    save_json_atomic(
        output_root / f"validation_summary_{shard_id}.json",
        payload,
    )
    return payload


def finalize_split(
    *,
    output_root: Path,
    contract: dict[str, Any],
    split_contract: dict[str, Any],
    shard_roots: dict[str, Path],
) -> dict[str, Any]:
    validate_split_contract(contract=contract, split_contract=split_contract)
    expected_shards = set(split_contract["shards"])
    if set(shard_roots) != expected_shards:
        raise ValueError(
            "Canonical merge shard roots mismatch: "
            f"expected={sorted(expected_shards)} "
            f"observed={sorted(shard_roots)}"
        )

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for shard_id, shard in split_contract["shards"].items():
        root = shard_roots[shard_id]
        for dataset, seed in shard["run_order"]:
            run_root = root / "shards" / dataset / f"seed_{seed}"
            evidence = validate_run(
                run_root=run_root,
                dataset=dataset,
                seed=int(seed),
                contract=contract,
            )
            evidence["shard_id"] = shard_id
            evidence["execution_server"] = shard["execution_server"]
            evidence["artifact_run_root"] = str(run_root)
            indexed[(dataset, int(seed))] = evidence

    ordered_results = [
        indexed[(dataset, int(seed))]
        for dataset, seed in split_contract["canonical_run_order"]
    ]
    payload = {
        "status": "complete",
        "model_name": split_contract["short_model_name"],
        "split_contract_id": split_contract["contract_id"],
        "training_source_revision": contract["training_source_revision"],
        "validated_run_count": len(ordered_results),
        "execution_servers": sorted(
            {run["execution_server"] for run in ordered_results}
        ),
        "mixed_server_runtime_is_not_a_model_compute_comparison": True,
        "held_out_test_evaluated": False,
        "runs": ordered_results,
    }
    save_json_atomic(output_root / "canonical_validation_summary.json", payload)
    save_json_atomic(
        output_root / "canonical_run_index.json",
        {
            "status": "complete",
            "runs": [
                {
                    "dataset": run["dataset"],
                    "seed": run["seed"],
                    "shard_id": run["shard_id"],
                    "execution_server": run["execution_server"],
                    "artifact_run_root": run["artifact_run_root"],
                }
                for run in ordered_results
            ],
        },
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("verify-source")
    source.add_argument("--project-root", type=Path, required=True)
    source.add_argument("--contract", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)

    split = subparsers.add_parser("verify-split")
    split.add_argument("--contract", type=Path, required=True)
    split.add_argument("--split-contract", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("validate-run")
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--contract", type=Path, required=True)

    complete = subparsers.add_parser("finalize")
    complete.add_argument("--output-root", type=Path, required=True)
    complete.add_argument("--contract", type=Path, required=True)

    shard = subparsers.add_parser("finalize-shard")
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--contract", type=Path, required=True)
    shard.add_argument("--split-contract", type=Path, required=True)
    shard.add_argument("--shard-id", required=True)

    merge = subparsers.add_parser("finalize-split")
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--contract", type=Path, required=True)
    merge.add_argument("--split-contract", type=Path, required=True)
    merge.add_argument(
        "--shard-root",
        action="append",
        required=True,
        help="Shard mapping formatted as shard_id=/absolute/path",
    )

    status = subparsers.add_parser("status")
    status.add_argument("--output-root", type=Path, required=True)
    status.add_argument("--state", required=True)
    status.add_argument("--orchestration-revision", required=True)
    status.add_argument("--training-source-revision", default=None)
    status.add_argument("--current-dataset", default=None)
    status.add_argument("--current-seed", type=int, default=None)
    status.add_argument("--completed-run-count", type=int, required=True)
    status.add_argument("--message", required=True)
    status.add_argument("--execution-server", default="5090")
    status.add_argument("--total-run-count", type=int, default=9)
    status.add_argument("--shard-id", default=None)
    status.add_argument("--status-filename", default="status.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "verify-source":
        contract = load_json(args.contract)
        payload = verify_source(args.project_root, contract)
        save_json_atomic(args.output, payload)
    elif args.command == "verify-split":
        contract = load_json(args.contract)
        split_contract = load_json(args.split_contract)
        payload = validate_split_contract(
            contract=contract,
            split_contract=split_contract,
            contract_path=args.contract,
        )
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
    elif args.command == "finalize-shard":
        contract = load_json(args.contract)
        split_contract = load_json(args.split_contract)
        finalize_shard(
            output_root=args.output_root,
            contract=contract,
            split_contract=split_contract,
            shard_id=args.shard_id,
        )
    elif args.command == "finalize-split":
        contract = load_json(args.contract)
        split_contract = load_json(args.split_contract)
        shard_roots: dict[str, Path] = {}
        for item in args.shard_root:
            shard_id, separator, root = item.partition("=")
            if not separator or not shard_id or not root:
                raise ValueError(f"Invalid shard root mapping: {item}")
            if shard_id in shard_roots:
                raise ValueError(f"Duplicate shard root mapping: {shard_id}")
            shard_roots[shard_id] = Path(root)
        finalize_split(
            output_root=args.output_root,
            contract=contract,
            split_contract=split_contract,
            shard_roots=shard_roots,
        )
    elif args.command == "status":
        if Path(args.status_filename).name != args.status_filename:
            raise ValueError("Status filename must not contain directories")
        payload = {
            "status": args.state,
            "model_name": "TitanTPP-MAC",
            "training_source_revision": resolve_status_training_revision(
                output_root=args.output_root,
                explicit_revision=args.training_source_revision,
            ),
            "orchestration_revision": args.orchestration_revision,
            "execution_server": args.execution_server,
            "shard_id": args.shard_id,
            "current_dataset": args.current_dataset,
            "current_seed": args.current_seed,
            "completed_run_count": args.completed_run_count,
            "total_run_count": args.total_run_count,
            "held_out_test_evaluated": False,
            "message": args.message,
        }
        save_json_atomic(args.output_root / args.status_filename, payload)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
