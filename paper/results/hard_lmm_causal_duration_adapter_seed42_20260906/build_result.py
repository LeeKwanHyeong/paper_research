#!/usr/bin/env python3
"""Build the durable audit for the Frozen-B causal duration screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "search_artifacts"
    / "hard_lmm_causal_duration_adapter_seed42_5090_20260906_17b215a"
)
EXPECTED_SOURCE_REVISION = "17b215a5c74453940a627dee049a5ee91796f005"
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "5a4f908dbe3919f2e1f5d32f396dbfef214e39e4deaf666e53b29fb5fe74be21"
)
EXPECTED_FOURWAY_AUDIT_SHA256 = (
    "d555f390714f3f569b76da6eac82e637e3c335c80ddf31f3908cdd42f85f5521"
)
EXPECTED_NORMALIZED_REPLAY_SHA256 = (
    "9d33cc47d0eb5e535dbf3400f7a5ebdf88ca0c7f538557c6f2d1896bf9761e44"
)
EXPECTED_LOCATION_SCALE_REPLAY_SHA256 = (
    "3d744fa82d9a8acb38303a53d4fef594268415552937d2c6257a12cf9c530043"
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_artifact_manifest(root: Path) -> tuple[dict[str, str], str, int]:
    manifest = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(encoded).hexdigest(), sum(
        (root / relative_path).stat().st_size for relative_path in manifest
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build(artifact_root: Path) -> None:
    status_path = artifact_root / "status.json"
    e1_summary_path = artifact_root / "e1/yellow_trip_hourly/summary.json"
    seed_summary_path = artifact_root / "seed42/yellow_trip_hourly/summary.json"
    run_audit_path = artifact_root / "seed42/yellow_trip_hourly/run_audit.json"
    fourway_audit_path = RESULT_ROOT / "taxi_duration_likelihood_fourway_audit.json"
    normalized_replay_path = (
        RESULT_ROOT / "taxi_normalized_integer_likelihood_replay_20260906.json"
    )
    normalized_replay_script_path = RESULT_ROOT / "replay_normalized_integer_likelihood.py"
    hybrid_replay_path = RESULT_ROOT / "taxi_location_scale_hybrid_replay_20260906.json"
    hybrid_replay_script_path = RESULT_ROOT / "replay_location_scale_hybrids.py"

    status = load_json(status_path)
    e1 = load_json(e1_summary_path)
    summary = load_json(seed_summary_path)
    run_audit = load_json(run_audit_path)
    fourway_audit = load_json(fourway_audit_path)
    normalized_replay = load_json(normalized_replay_path)
    hybrid_replay = load_json(hybrid_replay_path)

    require(status["source_revision"] == EXPECTED_SOURCE_REVISION, "Source revision drift")
    require(status["status"] == "stopped_after_taxi_screening_failure", "Stop status drift")
    require(status["additional_seeds_executed"] is False, "Additional seed access drift")
    require(status["conditional_datasets_executed"] is False, "Conditional dataset access drift")
    require(status["held_out_test_evaluated"] is False, "Held-out access drift")
    require(summary["acceptance_status"] == "failed", "Taxi decision drift")
    require(summary["evaluation_scope"] == "validation_only", "Evaluation scope drift")
    require(summary["held_out_test_evaluated"] is False, "Summary held-out access drift")
    require(run_audit["phase"] == "seed42" and run_audit["status"] == "passed", "Run audit drift")
    require(
        sha256_file(fourway_audit_path) == EXPECTED_FOURWAY_AUDIT_SHA256,
        "Four-way likelihood audit drift",
    )
    require(
        fourway_audit["contract_id"] == "duration_observation_likelihood_audit_v1",
        "Four-way audit contract drift",
    )
    require(fourway_audit["evaluation_scope"] == "validation_only", "Four-way scope drift")
    require(fourway_audit["held_out_test_evaluated"] is False, "Four-way held-out drift")
    fourway_rows = {row["model_role"]: row for row in fourway_audit["rows"]}
    require(set(fourway_rows) == {"A", "B", "rmtpp", "thp"}, "Four-way model-role drift")
    require(
        sha256_file(normalized_replay_path) == EXPECTED_NORMALIZED_REPLAY_SHA256,
        "Normalized positive-integer replay drift",
    )
    require(
        normalized_replay["scope"]["split"] == "validation_only"
        and normalized_replay["scope"]["held_out_test_evaluated"] is False
        and normalized_replay["scope"]["training_or_checkpoint_selection"] is False,
        "Normalized replay scope drift",
    )
    normalized_rows = normalized_replay["rows"]
    require(
        set(normalized_rows)
        == {"A", "B", "rmtpp", "thp", "causal_scale_adapter", "global_scale_control"},
        "Normalized replay model-role drift",
    )
    require(
        sha256_file(hybrid_replay_path) == EXPECTED_LOCATION_SCALE_REPLAY_SHA256,
        "Location-scale hybrid replay drift",
    )
    require(
        hybrid_replay["scope"]["split"] == "validation_only"
        and hybrid_replay["scope"]["held_out_test_evaluated"] is False
        and hybrid_replay["scope"]["training_or_checkpoint_selection"] is False,
        "Location-scale replay scope drift",
    )
    hybrid_rows = hybrid_replay["rows"]
    require(
        set(hybrid_rows)
        == {
            "A_location_A_scale",
            "B_location_B_scale",
            "A_location_B_scale",
            "B_location_A_scale",
        },
        "Location-scale replay role drift",
    )

    gates = summary["acceptance_gates"]
    expected_passed_invariants = (
        "base_location_bitwise_identical",
        "quantity_prediction_bitwise_identical",
        "source_model_state_unchanged",
        "time_median_bitwise_identical",
    )
    expected_failed_performance_gates = (
        "candidate_beats_global_control_by_0_005",
        "candidate_continuous_nll_at_most_A_plus_0_01",
        "candidate_interval_nll_at_most_B",
    )
    require(all(gates[name] is True for name in expected_passed_invariants), "Frozen-B invariant drift")
    require(
        all(gates[name] is False for name in expected_failed_performance_gates),
        "Performance-gate outcome drift",
    )

    manifest, manifest_sha256, artifact_bytes = canonical_artifact_manifest(artifact_root)
    require(manifest_sha256 == EXPECTED_ARTIFACT_MANIFEST_SHA256, "Artifact manifest drift")

    base = summary["base_B_validation_metrics"]
    candidate = summary["candidate"]
    candidate_metrics = candidate["selected_validation_metrics"]
    control = summary["global_scale_control"]
    control_metrics = control["selected_validation_metrics"]
    quantity = summary["quantity_metrics"]

    def normalized_metrics(role: str) -> tuple[float, float]:
        row = normalized_rows[role]
        return (
            float(row["fold_to_one_normalized_nll"]),
            float(row["zero_truncated_normalized_nll"]),
        )

    def fourway_metrics(role: str) -> tuple[float, float]:
        groups = fourway_rows[role]["groups"]["all"]
        return (
            float(groups["continuous_observation_nll"]),
            float(groups["integer_observation_nll"]),
        )

    A_continuous, A_interval = fourway_metrics("A")
    B_continuous_replay, B_interval_replay = fourway_metrics("B")
    require(
        A_continuous == summary["A_validation_continuous_nll"],
        "A continuous replay drift",
    )
    require(
        abs(B_continuous_replay - base["continuous_proper_time_nll"]) <= 1e-12
        and abs(B_interval_replay - base["interval_mass_time_nll"]) <= 1e-12,
        "B likelihood replay drift",
    )
    B_fold, B_truncated = normalized_metrics("B")

    metric_rows = [
        {
            "dataset": "yellow_trip_hourly",
            "split": "validation",
            "role": "A_matched_reference",
            "seed": 42,
            "best_epoch": "",
            "completed_epochs": "",
            "continuous_density_time_nll": A_continuous,
            "legacy_centered_bin_score": A_interval,
            "fold_to_one_positive_integer_nll": normalized_metrics("A")[0],
            "zero_truncated_positive_integer_nll": normalized_metrics("A")[1],
            "continuous_delta_vs_B": A_continuous - base["continuous_proper_time_nll"],
            "legacy_centered_bin_delta_vs_B": A_interval - base["interval_mass_time_nll"],
            "fold_to_one_delta_vs_B": normalized_metrics("A")[0] - B_fold,
            "zero_truncated_delta_vs_B": normalized_metrics("A")[1] - B_truncated,
            "quantity_mae": "",
            "quantity_rmse": "",
            "body_mae": "",
            "gt_p99_mae": "",
            "checkpoint_selection": "external matched reference",
        },
        {
            "dataset": "yellow_trip_hourly",
            "split": "validation",
            "role": "frozen_B",
            "seed": 42,
            "best_epoch": 0,
            "completed_epochs": 0,
            "continuous_density_time_nll": base["continuous_proper_time_nll"],
            "legacy_centered_bin_score": base["interval_mass_time_nll"],
            "fold_to_one_positive_integer_nll": B_fold,
            "zero_truncated_positive_integer_nll": B_truncated,
            "continuous_delta_vs_B": 0.0,
            "legacy_centered_bin_delta_vs_B": 0.0,
            "fold_to_one_delta_vs_B": 0.0,
            "zero_truncated_delta_vs_B": 0.0,
            "quantity_mae": quantity["mae"],
            "quantity_rmse": quantity["rmse"],
            "body_mae": quantity["body_mae"],
            "gt_p99_mae": quantity["gt_p99_mae"],
            "checkpoint_selection": "fixed source checkpoint",
        },
        {
            "dataset": "yellow_trip_hourly",
            "split": "validation",
            "role": "causal_scale_adapter",
            "seed": 42,
            "best_epoch": candidate["best_epoch"],
            "completed_epochs": candidate["completed_epochs"],
            "continuous_density_time_nll": candidate_metrics["continuous_proper_time_nll"],
            "legacy_centered_bin_score": candidate_metrics["interval_mass_time_nll"],
            "fold_to_one_positive_integer_nll": normalized_metrics("causal_scale_adapter")[0],
            "zero_truncated_positive_integer_nll": normalized_metrics("causal_scale_adapter")[1],
            "continuous_delta_vs_B": candidate_metrics["continuous_proper_time_nll"]
            - base["continuous_proper_time_nll"],
            "legacy_centered_bin_delta_vs_B": candidate_metrics["interval_mass_time_nll"]
            - base["interval_mass_time_nll"],
            "fold_to_one_delta_vs_B": normalized_metrics("causal_scale_adapter")[0] - B_fold,
            "zero_truncated_delta_vs_B": normalized_metrics("causal_scale_adapter")[1]
            - B_truncated,
            "quantity_mae": quantity["mae"],
            "quantity_rmse": quantity["rmse"],
            "body_mae": quantity["body_mae"],
            "gt_p99_mae": quantity["gt_p99_mae"],
            "checkpoint_selection": "earliest validation continuous-NLL minimum",
        },
        {
            "dataset": "yellow_trip_hourly",
            "split": "validation",
            "role": "global_scale_control",
            "seed": 42,
            "best_epoch": control["best_epoch"],
            "completed_epochs": control["completed_epochs"],
            "continuous_density_time_nll": control_metrics["continuous_proper_time_nll"],
            "legacy_centered_bin_score": control_metrics["interval_mass_time_nll"],
            "fold_to_one_positive_integer_nll": normalized_metrics("global_scale_control")[0],
            "zero_truncated_positive_integer_nll": normalized_metrics("global_scale_control")[1],
            "continuous_delta_vs_B": control_metrics["continuous_proper_time_nll"]
            - base["continuous_proper_time_nll"],
            "legacy_centered_bin_delta_vs_B": control_metrics["interval_mass_time_nll"]
            - base["interval_mass_time_nll"],
            "fold_to_one_delta_vs_B": normalized_metrics("global_scale_control")[0] - B_fold,
            "zero_truncated_delta_vs_B": normalized_metrics("global_scale_control")[1]
            - B_truncated,
            "quantity_mae": quantity["mae"],
            "quantity_rmse": quantity["rmse"],
            "body_mae": quantity["body_mae"],
            "gt_p99_mae": quantity["gt_p99_mae"],
            "checkpoint_selection": "earliest validation continuous-NLL minimum",
        },
    ]
    for role in ("rmtpp", "thp"):
        continuous, interval = fourway_metrics(role)
        metric_rows.append(
            {
                "dataset": "yellow_trip_hourly",
                "split": "validation",
                "role": f"{role}_matched_reference",
                "seed": 42,
                "best_epoch": "",
                "completed_epochs": "",
                "continuous_density_time_nll": continuous,
                "legacy_centered_bin_score": interval,
                "fold_to_one_positive_integer_nll": normalized_metrics(role)[0],
                "zero_truncated_positive_integer_nll": normalized_metrics(role)[1],
                "continuous_delta_vs_B": continuous - base["continuous_proper_time_nll"],
                "legacy_centered_bin_delta_vs_B": interval - base["interval_mass_time_nll"],
                "fold_to_one_delta_vs_B": normalized_metrics(role)[0] - B_fold,
                "zero_truncated_delta_vs_B": normalized_metrics(role)[1] - B_truncated,
                "quantity_mae": "",
                "quantity_rmse": "",
                "body_mae": "",
                "gt_p99_mae": "",
                "checkpoint_selection": "external matched reference",
            }
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULT_ROOT / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(metric_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    durable_audit = {
        "schema_version": 1,
        "experiment": "hard_lmm_causal_duration_adapter_seed42_20260906",
        "decision": "reject_scale_only_causal_duration_adapter",
        "decision_reason": (
            "Taxi failed the fixed continuous-NLL history-attribution and matched-A gates. "
            "A normalized positive-integer replay also leaves the candidate worse than B and A. "
            "The historical centered-bin gate is subnormalized and is not used as acceptance evidence."
        ),
        "scope": {
            "split": "validation_only",
            "seed": 42,
            "held_out_test_evaluated": False,
            "held_out_test_used_for_training_selection_or_method_choice": False,
            "manual_post_run_intermittent_test_target_aggregate_read": True,
            "analyst_blind_test_claim_allowed": False,
            "additional_seeds_executed": False,
            "intermittent_executed": False,
            "instacart_executed": False,
        },
        "execution": {
            "source_revision": status["source_revision"],
            "source_manifest_sha256": status["source_manifest_sha256"],
            "contract_sha256": status["contract_sha256"],
            "artifact_file_count": len(manifest),
            "artifact_bytes": artifact_bytes,
            "artifact_manifest_sha256": manifest_sha256,
            "cuda_contract_test_count": status["cuda_contract_tests"]["test_count"],
            "cuda_contract_tests_passed": status["cuda_contract_tests"]["failures"] == 0,
            "gpu": status["gpu_preflight"]["gpu_name"],
            "e1_elapsed_seconds": e1["runtime"]["elapsed_seconds"],
            "seed42_elapsed_seconds": summary["runtime"]["elapsed_seconds"],
            "peak_memory_allocated_bytes": summary["runtime"]["peak_memory_allocated_bytes"],
            "peak_memory_reserved_bytes": summary["runtime"]["peak_memory_reserved_bytes"],
            "train_count": summary["train_count"],
            "validation_count": summary["validation_count"],
        },
        "held_out_access_disclosure": {
            "experiment_runner_or_replay_loaded_test": False,
            "test_predictions_or_losses_computed": False,
            "manual_read_occurred_after_candidate_decision": True,
            "dataset": "intermittent_frozen_5000",
            "fields_exposed": (
                "aggregate delta_t count, minimum, maximum, integer-support flag, "
                "and five most frequent values"
            ),
            "used_for_model_threshold_or_method_selection": False,
            "consequence": (
                "The Intermittent held-out split remains model-unevaluated but can no longer "
                "be described as analyst-blind. Taxi and Instacart test data were not accessed "
                "by this post-run check."
            ),
        },
        "taxi_metrics": {
            "A_continuous_nll": A_continuous,
            "A_legacy_centered_bin_score": A_interval,
            "B_continuous_nll": base["continuous_proper_time_nll"],
            "B_legacy_centered_bin_score": base["interval_mass_time_nll"],
            "candidate_continuous_nll": candidate_metrics["continuous_proper_time_nll"],
            "candidate_legacy_centered_bin_score": candidate_metrics["interval_mass_time_nll"],
            "control_continuous_nll": control_metrics["continuous_proper_time_nll"],
            "control_legacy_centered_bin_score": control_metrics["interval_mass_time_nll"],
            "candidate_minus_B_continuous_nll": summary["acceptance_observed_deltas"][
                "candidate_minus_B_continuous_nll"
            ],
            "candidate_minus_control_continuous_nll": -summary[
                "acceptance_observed_deltas"
            ]["candidate_improvement_over_global_control_continuous_nll"],
            "candidate_minus_B_legacy_centered_bin_score": summary["acceptance_observed_deltas"][
                "candidate_minus_B_interval_nll"
            ],
            "candidate_scale_residual_mean": candidate_metrics[
                "bounded_log_scale_residual_mean"
            ],
            "candidate_scale_residual_std": candidate_metrics[
                "bounded_log_scale_residual_std"
            ],
            "quantity": quantity,
        },
        "taxi_fourway_legacy_centered_bin_audit": {
            role: {
                "continuous_density_nll": fourway_metrics(role)[0],
                "legacy_centered_bin_score": fourway_metrics(role)[1],
                "dt_equals_mode_legacy_centered_bin_score": fourway_rows[role]["groups"][
                    "dt_equals_mode"
                ]["integer_observation_nll"],
                "dt_other_legacy_centered_bin_score": fourway_rows[role]["groups"]["dt_other"][
                    "integer_observation_nll"
                ],
            }
            for role in ("A", "B", "rmtpp", "thp")
        },
        "normalized_positive_integer_replay": {
            "training_or_checkpoint_selection_performed": False,
            "primary_convention_proposed_for_future_work": "fold_to_one_round_and_clamp",
            "selector_match_warning": (
                "These checkpoints were selected by continuous-density NLL, so this is a "
                "validation-only sensitivity replay rather than a matched-selector comparison."
            ),
            "rows": {
                role: {
                    "fold_to_one_nll": normalized_metrics(role)[0],
                    "zero_truncated_nll": normalized_metrics(role)[1],
                }
                for role in (
                    "A",
                    "B",
                    "rmtpp",
                    "thp",
                    "causal_scale_adapter",
                    "global_scale_control",
                )
            },
            "candidate_minus_B_fold_to_one_nll": normalized_metrics(
                "causal_scale_adapter"
            )[0]
            - B_fold,
            "candidate_minus_A_fold_to_one_nll": normalized_metrics(
                "causal_scale_adapter"
            )[0]
            - normalized_metrics("A")[0],
            "candidate_improvement_over_global_fold_to_one_nll": normalized_metrics(
                "global_scale_control"
            )[0]
            - normalized_metrics("causal_scale_adapter")[0],
            "candidate_minus_B_zero_truncated_nll": normalized_metrics(
                "causal_scale_adapter"
            )[1]
            - B_truncated,
            "candidate_minus_A_zero_truncated_nll": normalized_metrics(
                "causal_scale_adapter"
            )[1]
            - normalized_metrics("A")[1],
            "candidate_improvement_over_global_zero_truncated_nll": normalized_metrics(
                "global_scale_control"
            )[1]
            - normalized_metrics("causal_scale_adapter")[1],
            "interpretation": (
                "At their continuous-NLL-selected operating points, the candidate scores more than "
                "0.005 below the history-free global scale control under both normalized conventions, "
                "but both are worse than Frozen-B and the candidate is worse than matched A. Different "
                "selection epochs and the missing same-capacity length-only control prohibit attributing "
                "this post-hoc difference to duration-prefix values."
            ),
        },
        "location_scale_output_decomposition": {
            "training_or_checkpoint_selection_performed": False,
            "causal_attribution": False,
            "rows": hybrid_rows,
            "B_to_A_gap_fraction_recovered_by_A_scale_with_B_location": {
                metric: (
                    hybrid_rows["B_location_B_scale"][metric]
                    - hybrid_rows["B_location_A_scale"][metric]
                )
                / (
                    hybrid_rows["B_location_B_scale"][metric]
                    - hybrid_rows["A_location_A_scale"][metric]
                )
                for metric in (
                    "continuous_density_nll",
                    "fold_to_one_normalized_nll",
                    "zero_truncated_normalized_nll",
                )
            },
            "interpretation": (
                "Swapping A's selected scale array onto B's selected location array recovers most "
                "of the observed A-B gap while retaining B's time median. This output-parameter "
                "decomposition makes scale prediction the next focused hypothesis, but it does not "
                "identify which input or trainable architecture can reproduce A's scale values."
            ),
        },
        "metric_definition_diagnostic": {
            "B_minus_A_continuous_density_nll": B_continuous_replay - A_continuous,
            "B_minus_A_legacy_centered_bin_score": B_interval_replay - A_interval,
            "relative_gap_reduction_under_legacy_centered_score": 1.0
            - (B_interval_replay - A_interval) / (B_continuous_replay - A_continuous),
            "interpretation": (
                "The within-score B-versus-A numerical gap is smaller under the historical "
                "centered-bin calculation. That calculation omits mass below 0.5 and is not a "
                "normalized PMF on positive integers. The ratio is descriptive sensitivity only."
            ),
        },
        "post_run_likelihood_correction": {
            "historical_centered_bin_score_is_normalized_pmf": False,
            "missing_probability_mass": "F(0.5)",
            "historical_interval_gate_valid_for_final_decision": False,
            "candidate_rejection_still_identified": True,
            "independent_valid_failures": [
                "candidate continuous NLL did not beat the history-free global control by 0.005",
                "candidate continuous NLL exceeded matched A plus 0.01",
            ],
            "required_before_next_candidate": (
                "Freeze a normalized positive-integer observation model and test total mass=1."
            ),
        },
        "historical_fixed_gates": {
            "values": gates,
            "legacy_interval_gate_valid_for_final_decision": False,
            "final_decision_valid_continuous_gates": [
                "candidate_beats_global_control_by_0_005",
                "candidate_continuous_nll_at_most_A_plus_0_01",
            ],
        },
        "frozen_B_invariants": {
            "epoch_zero_exact_B": summary["epoch_zero_exact_B"],
            "source_model_state_unchanged": summary["source_model_state_unchanged"],
            "quantity_prediction_bitwise_identical": summary[
                "quantity_prediction_bitwise_identical"
            ],
            "base_location_bitwise_identical": summary["base_location_bitwise_identical"],
            "time_median_bitwise_identical": summary["time_median_bitwise_identical"],
            "cross_device_median_formula_max_abs_error": candidate_metrics[
                "cross_device_median_formula_max_abs_error"
            ],
        },
        "training_diagnostic": {
            "candidate_best_epoch": candidate["best_epoch"],
            "candidate_completed_epochs": candidate["completed_epochs"],
            "candidate_epoch_9_train_continuous_nll": candidate["history"][-1][
                "train_continuous_proper_time_nll"
            ],
            "candidate_epoch_9_validation_continuous_nll": candidate["history"][-1][
                "val_continuous_proper_time_nll"
            ],
            "candidate_epoch_9_validation_legacy_centered_bin_score": candidate["history"][-1][
                "val_interval_mass_time_nll"
            ],
            "interpretation": (
                "The learned history-dependent scale was almost constant. Training and validation "
                "continuous NLL diverged after the earliest optimum, which establishes a "
                "generalization failure but does not distinguish overfitting from distribution shift."
            ),
        },
        "source_artifact_hashes": {
            "status_json": sha256_file(status_path),
            "seed42_summary_json": sha256_file(seed_summary_path),
            "seed42_run_audit_json": sha256_file(run_audit_path),
            "taxi_fourway_likelihood_audit_json": sha256_file(fourway_audit_path),
            "normalized_positive_integer_replay_json": sha256_file(normalized_replay_path),
            "normalized_positive_integer_replay_script": sha256_file(
                normalized_replay_script_path
            ),
            "location_scale_hybrid_replay_json": sha256_file(hybrid_replay_path),
            "location_scale_hybrid_replay_script": sha256_file(
                hybrid_replay_script_path
            ),
        },
    }
    (RESULT_ROOT / "validation_audit.json").write_text(
        json.dumps(durable_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_lines = [
        f"artifact_root={artifact_root}",
        f"file_count={len(manifest)}",
        f"total_bytes={artifact_bytes}",
        f"canonical_manifest_sha256={manifest_sha256}",
        "",
        *[f"{digest}  {relative_path}" for relative_path, digest in manifest.items()],
    ]
    (RESULT_ROOT / "artifact_sha256.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    build(args.artifact_root.resolve())


if __name__ == "__main__":
    main()
