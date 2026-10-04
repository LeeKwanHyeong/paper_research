"""Fail-closed runner boundaries for the separately approved THP control."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from models.TPPs.CountAwareTHPStaticMemory import (
    THP_STATIC_MEMORY_CONTRACT,
    THP_STATIC_MEMORY_CONTRACT_SHA256,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTRY_SHA256 = "92c6165b43c7a11fa515ce66be326b133a4f83b428a28d240a26985d2c2d7a8e"


@contextmanager
def record_static_memory_failure(output: Path):
    """Mark a newly created launch failed atomically, never touching an old run."""
    owned = False

    def claim_created_output():
        nonlocal owned
        owned = True

    try:
        yield claim_created_output
    except BaseException as error:
        launch = output / "launch_contract.json"
        if owned and launch.exists():
            payload = json.loads(launch.read_text())
            payload.update(status="failed", error=f"{type(error).__name__}: {error}",
                           failed_at=datetime.now(timezone.utc).isoformat())
            temporary = launch.with_suffix(".failed.tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary.replace(launch)
        raise


def frozen_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    documents = []
    for name, expected in (
        (THP_STATIC_MEMORY_CONTRACT, THP_STATIC_MEMORY_CONTRACT_SHA256),
        ("count_aware_thp_static_hard_memory_baselines_v1", REGISTRY_SHA256),
    ):
        payload = (ROOT / "paper/contracts" / f"{name}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("THP static memory frozen document digest mismatch")
        documents.append(json.loads(payload))
    return documents[0], documents[1]


def validate_static_memory_launch(args: argparse.Namespace, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Validate the authorized experimental scope before data access or writes."""
    contract, registry = frozen_documents()
    rows = {row["dataset"]: row for row in registry["datasets"]}
    if args.dataset_contract not in rows or seeds != (42,):
        raise ValueError("THP static memory is restricted to Taxi/RAF seed42")
    row = rows[args.dataset_contract]
    expected = {
        "hidden_dim": 64, "batch_size": 128, "lr": .001, "grad_clip": 1.,
        "lambda_log_qty": 1., "lambda_tail": 0., "time_head_lr_multiplier": 1.,
        "time_scale": 3., "time_w_max": 10. / 3., "time_intercept_limit": 30.,
        "time_wd_safety_limit": 40., "lookback_weeks": row["lookback"],
        "max_seq_len": row["max_seq_len"], "max_series": None,
        "max_train_batches": None, "max_val_batches": None,
        "titans_memory_gradient_clip": None, "force_rerun": False,
    }
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("THP static memory frozen training/context contract mismatch")
    budget = (args.epochs, args.min_epochs, args.early_stopping_patience)
    training = contract["training"]
    if budget not in ((1, 1, 1), (training["maximum_epochs"], training["minimum_epochs"], training["patience"])):
        raise ValueError("THP static memory permits only full e1 feasibility or fresh e300/min40/patience40")
    if args.output_dir.exists():
        raise FileExistsError("THP static memory requires a fresh output directory")
    return row


def validate_static_memory_data(launch: dict[str, Any], frame: pl.DataFrame,
                               row: dict[str, Any]) -> dict[str, Any]:
    """Match full train/validation counts and cutoffs without a model forward."""
    for key in ("data_sha256", "split_manifest_sha256", "quantity_contract", "history_length_contract"):
        if launch[key] != row[key]:
            raise ValueError(f"THP static memory baseline {key} mismatch")
    if (launch["evaluation_scope"] != "validation_only" or launch["held_out_test_evaluated"] is not False
            or set(launch["split_rows"]) != {"train", "validation"}
            or set(frame["chronological_split"].unique().to_list()) != {"train", "validation"}):
        raise ValueError("THP static memory held-out/split scope mismatch")
    counts = {}
    for split in ("train", "validation"):
        dataset = RMTPPWeekLookbackDataset(
            frame, lookback_weeks=row["lookback"], max_seq_len=row["max_seq_len"],
            val_ratio=.2, mode="all", split_col="chronological_split", target_splits={split},
        )
        counts[split] = len(dataset)
        if counts[split] != row[f"{split}_targets"]:
            raise ValueError(f"THP static memory full {split} target count mismatch")
    if launch["time_head"]["train_time_statistics"]["target_count"] != counts["train"]:
        raise ValueError("THP static memory train time target count mismatch")
    return {
        "routing_contract_id": THP_STATIC_MEMORY_CONTRACT,
        "routing_contract_sha256": THP_STATIC_MEMORY_CONTRACT_SHA256,
        "baseline_registry_sha256": REGISTRY_SHA256,
        "validated_target_counts": counts,
        "artifact_reuse_policy": "fresh_only_no_resume_or_overwrite",
    }
