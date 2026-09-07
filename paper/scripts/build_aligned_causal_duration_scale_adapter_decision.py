#!/usr/bin/env python3
"""Build a fail-closed validation decision for the aligned causal adapter."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.run_aligned_causal_duration_scale_adapter import (
    CONTRACT_ID,
    DEFAULT_CONTRACT,
    SELECTED_CHECKPOINT_NAME,
    load_contract_without_duplicate_keys,
    sha256_file,
    validate_contract,
)
from paper.scripts.run_hard_lmm_time_head_refit import require, save_json
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


ROLES = ("candidate", "length_only_control", "global_scale_control")


def _finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"Invalid {label}",
    )
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Invalid JSON root: {path}")
    return payload


def validate_full_summary(
    *,
    summary_path: Path,
    dataset_spec: Mapping[str, Any],
    contract_sha256: str,
    source_revision: str,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    dataset = str(dataset_spec["dataset"])
    for name, expected in {
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "source_revision": source_revision,
        "status": "success",
        "dataset": dataset,
        "seed": 42,
        "qualified_full_data": True,
        "qualified_full_fit": True,
        "source_checkpoint_sha256": dataset_spec[
            "aligned_B_checkpoint_file_sha256"
        ],
        "source_model_state_sha256": dataset_spec[
            "aligned_B_model_state_sha256"
        ],
        "source_model_state_unchanged": True,
        "train_feature_cache_sha256": dataset_spec[
            "aligned_B_train_feature_cache_sha256"
        ],
        "validation_feature_cache_sha256": dataset_spec[
            "aligned_B_validation_feature_cache_sha256"
        ],
        "quantity_prediction_bitwise_identical": True,
        "time_median_bitwise_identical": True,
        "base_location_bitwise_identical": True,
        "epoch_zero_exact_aligned_B": True,
        "candidate_length_only_initial_state_identical": True,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }.items():
        require(summary.get(name) == expected, f"{dataset} summary drift: {name}")

    expected_count = int(dataset_spec["expected_validation_targets"])
    role_metrics: dict[str, float] = {}
    for role in ROLES:
        role_summary = summary.get(role)
        require(isinstance(role_summary, Mapping), f"Missing {dataset} {role}")
        require(role_summary.get("status") == "success", f"Failed {dataset} {role}")
        require(
            role_summary.get("held_out_test_evaluated") is False,
            f"Held-out leakage in {dataset} {role}",
        )
        metrics = role_summary.get("selected_validation_metrics")
        require(isinstance(metrics, Mapping), f"Missing metrics: {dataset} {role}")
        require(int(metrics.get("count", -1)) == expected_count, "Count drift")
        role_metrics[role] = _finite(
            metrics.get("primary_proper_time_nll"), f"{dataset} {role} NLL"
        )
        checkpoint_path = summary_path.parent / role / SELECTED_CHECKPOINT_NAME
        require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
        checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
        require(checkpoint.get("role") == role, f"Checkpoint role drift: {role}")
        require(
            canonical_state_dict_sha256(checkpoint["module_state_dict"])
            == checkpoint.get("module_state_sha256")
            == role_summary.get("selected_module_state_sha256"),
            f"Checkpoint digest drift: {dataset} {role}",
        )

    candidate = role_metrics["candidate"]
    baseline = _finite(
        summary["base_aligned_B_validation_metrics"]["primary_proper_time_nll"],
        f"{dataset} aligned-B NLL",
    )
    aligned_a = _finite(summary["aligned_A_validation_primary_nll"], "aligned-A NLL")
    control_margin = 0.005
    recomputed = {
        "candidate_meets_dataset_aligned_B_rule": (
            baseline - candidate
            >= float(dataset_spec["minimum_candidate_primary_improvement"])
        ),
        "candidate_within_aligned_A_plus_0_01": candidate <= aligned_a + 0.01,
        "candidate_beats_global_control_by_0_005": (
            role_metrics["global_scale_control"] - candidate >= control_margin
        ),
        "candidate_beats_length_only_control_by_0_005": (
            role_metrics["length_only_control"] - candidate >= control_margin
        ),
    }
    reported = summary.get("acceptance_gates")
    require(isinstance(reported, Mapping), f"Missing gates: {dataset}")
    for name, value in recomputed.items():
        require(reported.get(name) is value, f"Gate drift: {dataset} {name}")
    passed = summary.get("acceptance_status") == "passed"
    require(passed == all(reported.values()), f"Acceptance status drift: {dataset}")
    return {
        "dataset": dataset,
        "candidate_primary_nll": candidate,
        "aligned_B_primary_nll": baseline,
        "aligned_A_primary_nll": aligned_a,
        "global_control_primary_nll": role_metrics["global_scale_control"],
        "length_only_control_primary_nll": role_metrics["length_only_control"],
        "acceptance_gates": dict(reported),
        "acceptance_status": summary["acceptance_status"],
        "quantity_metrics": summary["quantity_metrics"],
    }


def build_decision(
    *,
    contract_path: Path,
    summary_root: Path,
    source_revision: str,
) -> dict[str, Any]:
    require(len(source_revision) == 40, "Source revision must be a full SHA")
    contract = load_contract_without_duplicate_keys(contract_path)
    datasets = validate_contract(contract)
    contract_sha256 = sha256_file(contract_path)
    taxi = validate_full_summary(
        summary_path=summary_root / "yellow_trip_hourly" / "summary.json",
        dataset_spec=datasets["yellow_trip_hourly"],
        contract_sha256=contract_sha256,
        source_revision=source_revision,
    )
    rows = [taxi]
    if taxi["acceptance_status"] == "passed":
        for dataset in ("intermittent_frozen_5000", "insta_market_basket"):
            rows.append(validate_full_summary(
                summary_path=summary_root / dataset / "summary.json",
                dataset_spec=datasets[dataset],
                contract_sha256=contract_sha256,
                source_revision=source_revision,
            ))
    status = (
        "accepted_common_causal_time_adapter"
        if len(rows) == 3 and all(row["acceptance_status"] == "passed" for row in rows)
        else "rejected_at_taxi_gate"
    )
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "source_revision": source_revision,
        "status": status,
        "evaluated_datasets": [row["dataset"] for row in rows],
        "rows": rows,
        "held_out_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = build_decision(
        contract_path=args.contract,
        summary_root=args.summary_root,
        source_revision=args.source_revision,
    )
    save_json(args.output, decision)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
