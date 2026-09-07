#!/usr/bin/env python3
"""Apply staged gates to frozen raw-affine calibration artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


B_ROWS = {
    ("intermittent_frozen_5000", "B"),
    ("yellow_trip_hourly", "B"),
    ("insta_market_basket", "B"),
}
FINAL_ROWS = B_ROWS | {
    ("insta_market_basket", "rmtpp"),
    ("insta_market_basket", "thp"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def verify_referenced_file(reference: Mapping[str, Any], *, label: str) -> None:
    path = Path(str(reference.get("file", "")))
    require(path.is_file(), f"{label} file is missing")
    require(sha256_file(path) == reference.get("file_sha256"), f"{label} digest drift")


def finite_metric(value: Any, *, label: str) -> float:
    numeric = float(value)
    require(math.isfinite(numeric), f"Non-finite metric: {label}")
    return numeric


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite decision artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_results(
    paths: list[Path], *, expected_rows: set[tuple[str, str]], phase: str
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        result_path = path / "result.json" if path.is_dir() else path
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        require(payload.get("status") == "success", f"Unsuccessful result: {result_path}")
        require(payload.get("phase") == phase, f"Expected {phase}-phase evidence")
        require(payload.get("contract_id") == "frozen_raw_affine_calibration_v1", "Contract drift")
        require(payload.get("evaluation_scope") == "validation_only", "Evaluation scope drift")
        require(payload.get("held_out_test_evaluated") is False, "Held-out result admitted")
        require(payload.get("smoke") is False, "Smoke result admitted as full evidence")
        require("RTX 5080" in payload.get("runtime", {}).get("gpu_name", ""), "Non-5080 result")
        require(payload.get("source_model_state_unchanged") is True, "Frozen source changed")
        require(payload.get("source_gradients_absent") is True, "Source gradients present")
        require(
            payload.get("identity_control", {}).get("exact_prediction_identity") is True,
            "Identity drift",
        )
        source_manifest = payload.get("source_manifest")
        require(isinstance(source_manifest, Mapping), "Source manifest audit is missing")
        require(source_manifest.get("all_file_hashes_verified") is True, "Source files unverified")
        require(valid_sha256(source_manifest.get("file_sha256")), "Source manifest digest invalid")
        verify_referenced_file(source_manifest, label="source manifest")
        input_digests = payload.get("input_digests")
        require(isinstance(input_digests, Mapping), "Input digests are missing")
        require(valid_sha256(input_digests.get("data_sha256")), "Data digest invalid")
        require(valid_sha256(input_digests.get("split_manifest_sha256")), "Split digest invalid")
        population = payload.get("target_population", {}).get(phase)
        cache = payload.get("cache", {}).get(phase)
        require(isinstance(population, Mapping), f"{phase} target population missing")
        require(isinstance(cache, Mapping), f"{phase} cache audit missing")
        require(
            int(population.get("target_count", -1))
            == int(cache.get("target_count", -2))
            == int(cache.get("full_population_target_count", -3)),
            f"{phase} target count drift",
        )
        require(valid_sha256(population.get("target_identity_sha256")), "Target identity invalid")
        require(valid_sha256(population.get("target_quantity_sha256")), "Target quantity invalid")
        cache_file = result_path.parent / str(cache.get("file", ""))
        require(cache_file.is_file(), f"{phase} cache file missing")
        require(sha256_file(cache_file) == cache.get("file_sha256"), f"{phase} cache digest drift")
        if phase == "validation":
            validation = payload.get("validation")
            require(isinstance(validation, Mapping), "Validation audit is missing")
            require(
                validation.get("relative_changes", {}).get("time_nll") == 0.0,
                "Time NLL drift",
            )
            require(
                validation.get("time_nll_identity", {}).get("exact") is True,
                "Time vector drift",
            )
            require(
                validation.get("time_nll_identity", {}).get("distinct_array_storage") is True,
                "Time identity did not compare separate arrays",
            )
            bootstrap = validation.get("paired_series_bootstrap")
            require(isinstance(bootstrap, Mapping), "Paired bootstrap is missing")
            require(bootstrap.get("seed") == 20260907, "Bootstrap seed drift")
            require(bootstrap.get("replicates") == 500, "Bootstrap replicate drift")
            for version in ("baseline", "calibrated"):
                for metric_name in ("mae", "mse", "rmse", "bias", "log1p_mse", "time_nll"):
                    finite_metric(validation[version][metric_name], label=f"{version}/{metric_name}")
            train_reference = payload.get("train_result")
            require(isinstance(train_reference, Mapping), "Train-result reference is missing")
            verify_referenced_file(train_reference, label="train result")
            prerequisite = payload.get("prerequisite_decision")
            require(isinstance(prerequisite, Mapping), "Prerequisite decision is missing")
            verify_referenced_file(prerequisite, label="prerequisite decision")
            expected_stage = (
                "B_three_dataset_train_gate"
                if payload.get("model_role") == "B"
                else "B_three_dataset_validation_gate"
            )
            require(prerequisite.get("stage") == expected_stage, "Prerequisite stage drift")
        else:
            require(payload.get("prerequisite_decision") is None, "B train has a prerequisite")
        key = (str(payload["dataset"]), str(payload["model_role"]))
        require(key not in rows, f"Duplicate result row: {key}")
        payload["_result_file"] = str(result_path.resolve())
        payload["_result_file_sha256"] = sha256_file(result_path)
        rows[key] = payload
    require(set(rows) == expected_rows, f"Result row scope drift: {set(rows) ^ expected_rows}")
    require(len({row["contract_sha256"] for row in rows.values()}) == 1, "Contract hash mismatch")
    require(len({row["source_revision"] for row in rows.values()}) == 1, "Source revision mismatch")
    require(
        len({row["source_manifest"]["file_sha256"] for row in rows.values()}) == 1,
        "Source manifest mismatch",
    )
    return rows


def common_decision(rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    first = next(iter(rows.values()))
    return {
        "status": "success",
        "contract_id": "frozen_raw_affine_calibration_v1",
        "contract_sha256": first["contract_sha256"],
        "source_revision": first["source_revision"],
        "source_manifest_file_sha256": first["source_manifest"]["file_sha256"],
        "execution_server": "5080",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize_b_train(paths: list[Path]) -> dict[str, Any]:
    rows = load_results(paths, expected_rows=B_ROWS, phase="train")
    checks = {
        dataset: bool(rows[(dataset, "B")]["train_audit"]["gate_passed"])
        for dataset, _ in sorted(B_ROWS)
    }
    accepted = all(checks.values())
    return {
        **common_decision(rows),
        "stage": "B_three_dataset_train_gate",
        "decision": (
            "accepted_for_B_validation" if accepted else "rejected_stop_before_validation"
        ),
        "accepted": accepted,
        "checks": checks,
        "result_identities": [
            {
                "dataset": key[0],
                "model_role": key[1],
                "checkpoint_file_sha256": rows[key]["source_checkpoint_file_sha256"],
                "train_cache_sha256": rows[key]["cache"]["train"]["file_sha256"],
                "target_identity_sha256": rows[key]["target_population"]["train"][
                    "target_identity_sha256"
                ],
                "target_quantity_sha256": rows[key]["target_population"]["train"][
                    "target_quantity_sha256"
                ],
                "result_file": rows[key]["_result_file"],
                "result_file_sha256": rows[key]["_result_file_sha256"],
            }
            for key in sorted(rows)
        ],
    }


def summarize_b_validation(paths: list[Path]) -> dict[str, Any]:
    rows = load_results(paths, expected_rows=B_ROWS, phase="validation")
    checks = {
        dataset: bool(rows[(dataset, "B")]["validation"]["gate_passed"])
        for dataset, _ in sorted(B_ROWS)
    }
    accepted = all(checks.values())
    return {
        **common_decision(rows),
        "stage": "B_three_dataset_validation_gate",
        "decision": (
            "accepted_for_instacart_fairness"
            if accepted
            else "rejected_stop_before_instacart_fairness"
        ),
        "accepted": accepted,
        "checks": checks,
        "result_identities": [
            {
                "dataset": key[0],
                "model_role": key[1],
                "checkpoint_file_sha256": rows[key]["source_checkpoint_file_sha256"],
                "validation_cache_sha256": rows[key]["cache"]["validation"]["file_sha256"],
                "target_identity_sha256": rows[key]["target_population"]["validation"][
                    "target_identity_sha256"
                ],
                "target_quantity_sha256": rows[key]["target_population"]["validation"][
                    "target_quantity_sha256"
                ],
                "result_file": rows[key]["_result_file"],
                "result_file_sha256": rows[key]["_result_file_sha256"],
            }
            for key in sorted(rows)
        ],
    }


def metric(row: Mapping[str, Any], version: str, name: str) -> float:
    return float(row["validation"][version][name])


def summarize_final(paths: list[Path]) -> dict[str, Any]:
    rows = load_results(paths, expected_rows=FINAL_ROWS, phase="validation")
    b_train = all(bool(rows[key]["train_audit"]["gate_passed"]) for key in B_ROWS)
    b_validation = all(bool(rows[key]["validation"]["gate_passed"]) for key in B_ROWS)
    instacart = {
        role: rows[("insta_market_basket", role)] for role in ("B", "rmtpp", "thp")
    }
    fairness_checks: dict[str, bool] = {}
    for role in ("rmtpp", "thp"):
        for name in ("mae", "rmse"):
            fairness_checks[f"B_lt_{role}_{name}"] = (
                metric(instacart["B"], "calibrated", name)
                < metric(instacart[role], "calibrated", name)
            )
    fairness = all(fairness_checks.values())
    instacart_target_identities = {
        row["target_population"]["validation"]["target_identity_sha256"]
        for row in instacart.values()
    }
    instacart_target_quantities = {
        row["target_population"]["validation"]["target_quantity_sha256"]
        for row in instacart.values()
    }
    instacart_target_counts = {
        row["target_population"]["validation"]["target_count"]
        for row in instacart.values()
    }
    identical_instacart_targets = (
        len(instacart_target_identities)
        == len(instacart_target_quantities)
        == len(instacart_target_counts)
        == 1
    )
    fairness_checks["identical_validation_targets"] = identical_instacart_targets
    fairness = all(fairness_checks.values())
    accepted = b_train and b_validation and fairness
    table = []
    for key in sorted(rows):
        row = rows[key]
        deployed = row["validation"]["deployed_calibration_parameters"]
        table.append({
            "dataset": key[0],
            "model_role": key[1],
            "calibration_activated": row["validation"]["calibration_activated_by_train_rule"],
            "slope": float(deployed["slope"]),
            "intercept": float(deployed["intercept"]),
            "baseline_mae": metric(row, "baseline", "mae"),
            "calibrated_mae": metric(row, "calibrated", "mae"),
            "baseline_rmse": metric(row, "baseline", "rmse"),
            "calibrated_rmse": metric(row, "calibrated", "rmse"),
            "rmse_relative_change": float(row["validation"]["relative_changes"]["rmse"]),
            "train_gate_passed": bool(row["train_audit"]["gate_passed"]),
            "validation_gate_passed": bool(row["validation"]["gate_passed"]),
        })
    b_rmse_relative_changes = [
        float(rows[key]["validation"]["relative_changes"]["rmse"])
        for key in sorted(B_ROWS)
    ]
    return {
        **common_decision(rows),
        "stage": "final_instacart_fairness_gate",
        "decision": (
            "accepted_for_separate_additional_seed_contract"
            if accepted
            else "rejected_stop_no_additional_training"
        ),
        "accepted": accepted,
        "gates": {
            "B_train_internal_all_three": b_train,
            "B_validation_all_three": b_validation,
            "instacart_calibrated_fairness": fairness,
            "instacart_fairness_checks": fairness_checks,
        },
        "rows": table,
        "B_three_dataset_rmse_relative_change_macro_mean": sum(
            b_rmse_relative_changes
        )
        / len(b_rmse_relative_changes),
        "result_identities": [
            {
                "dataset": key[0],
                "model_role": key[1],
                "result_file": rows[key]["_result_file"],
                "result_file_sha256": rows[key]["_result_file_sha256"],
            }
            for key in sorted(rows)
        ],
        "interpretation_boundary": (
            "output-interface calibration ablation; not a Backbone improvement"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("b-train", "b-validation", "final"), required=True
    )
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize = {
        "b-train": summarize_b_train,
        "b-validation": summarize_b_validation,
        "final": summarize_final,
    }[args.stage]
    result = summarize(args.result)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
