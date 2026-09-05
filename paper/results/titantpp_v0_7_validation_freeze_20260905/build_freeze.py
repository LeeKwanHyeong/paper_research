#!/usr/bin/env python3
"""Build the v0.7 validation-freeze evidence from immutable validation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEEDS = (42, 52, 62)
MODELS = ("rmtpp", "thp", "titantpp")
MODEL_LABELS = {
    "rmtpp": "Count-aware RMTPP",
    "thp": "Count-aware THP",
    "titantpp": "TitanTPP Hard-LMM",
}
METRICS = (
    "best_val_time_nll",
    "best_val_log_qty_mse",
    "best_val_qty_mae",
    "best_val_qty_rmse",
)
SHORT_METRICS = {
    "best_val_time_nll": "time_nll",
    "best_val_log_qty_mse": "log_qty_mse",
    "best_val_qty_mae": "qty_mae",
    "best_val_qty_rmse": "qty_rmse",
}
DATASETS = {
    "taxi": {
        "label": "Taxi",
        "root": ROOT
        / "search_artifacts/count_aware_taxi_t0_t1_e300_20260824/"
        "yellow_trip_hourly/t0_common_control",
        "artifact_validation": ROOT
        / "paper/results/count_aware_taxi_t0_t1_e300_20260824/artifact_validation.json",
    },
    "instacart": {
        "label": "Instacart",
        "root": ROOT
        / "search_artifacts/count_aware_instacart_t0_e300_20260824_recovery1/"
        "insta_market_basket/t0_common_control",
        "artifact_validation": ROOT
        / "paper/results/count_aware_instacart_t0_e300_20260824/artifact_validation.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def sample_std(values: Iterable[float]) -> float:
    return statistics.stdev(list(values))


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def rel_improvement(reference: float, candidate: float) -> float:
    """Positive means the lower-is-better candidate improves on the reference."""
    return 100.0 * (reference - candidate) / abs(reference)


def summary_path(root: Path, model: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / model
        / "count_only_log_regression"
        / f"seed_{seed}"
        / "summary.json"
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_validation_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[Path],
]:
    seed_rows: list[dict[str, Any]] = []
    quantity_seed_rows: list[dict[str, Any]] = []
    history_seed_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []

    for dataset, spec in DATASETS.items():
        root = spec["root"]
        launch_path = root / "launch_contract.json"
        run_csv_path = root / "run_summaries.csv"
        quantity_csv_path = root / "quantity_seed_metrics.csv"
        history_csv_path = root / "history_seed_metrics.csv"
        source_paths.extend(
            [
                launch_path,
                run_csv_path,
                quantity_csv_path,
                history_csv_path,
                spec["artifact_validation"],
            ]
        )
        launch = read_json(launch_path)
        artifact_validation = read_json(spec["artifact_validation"])
        require(launch["status"] == "complete", f"Incomplete launch contract: {dataset}")
        require(launch["evaluation_scope"] == "validation_only", dataset)
        require(launch["held_out_test_evaluated"] is False, dataset)
        require(artifact_validation["status"] == "complete", dataset)
        require(artifact_validation["held_out_test_evaluated"] is False, dataset)
        require(tuple(launch["seeds"]) == SEEDS, dataset)
        require(launch["epochs"] == 300, dataset)
        require(launch["model_role"] == "t0_common_control", dataset)
        interface = launch["interfaces"]["count_only_log_regression"]
        require(interface["mode"] == "mark_free_count_aware_log_regression", dataset)
        require(interface["history_features"] == ["log1p_delta_t", "log1p_raw_quantity"], dataset)
        require(interface["quantity_mark_used"] is False, dataset)
        require(interface["quantity_residual_used"] is False, dataset)

        for model in MODELS:
            for seed in SEEDS:
                path = summary_path(root, model, seed)
                source_paths.append(path)
                item = read_json(path)
                require(item["status"] == "success", str(path))
                require(item["backbone"] == model, str(path))
                require(item["seed"] == seed, str(path))
                require(item["variant"] == "count_only_log_regression", str(path))
                require(item["evaluation_scope"] == "validation_only", str(path))
                require(item["held_out_test_evaluated"] is False, str(path))
                item_interface = item["interface_meta"]
                require(item_interface["mode"] == "mark_free_count_aware_log_regression", str(path))
                require(item_interface["quantity_mark_used"] is False, str(path))
                require(item_interface["quantity_residual_used"] is False, str(path))
                require(
                    math.isclose(
                        item["best_val_joint_objective"],
                        item["best_val_time_nll"] + item["best_val_log_qty_mse"],
                        rel_tol=0.0,
                        # The three fields were accumulated independently over batches.
                        abs_tol=1e-8,
                    ),
                    f"Joint objective mismatch: {path}",
                )
                row = {
                    "dataset": dataset,
                    "dataset_label": spec["label"],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "seed": seed,
                    "best_epoch": item["best_epoch"],
                    "completed_epochs": item["completed_epochs"],
                    "joint_objective": item["best_val_joint_objective"],
                    "time_nll": item["best_val_time_nll"],
                    "log_qty_mse": item["best_val_log_qty_mse"],
                    "qty_mae": item["best_val_qty_mae"],
                    "qty_rmse": item["best_val_qty_rmse"],
                    "parameter_count": item["parameter_count"],
                    "source_revision": item["source_revision"],
                    "checkpoint_state_sha256": item["checkpoint_state_sha256"],
                    "evaluation_scope": item["evaluation_scope"],
                    "held_out_test_evaluated": str(item["held_out_test_evaluated"]).lower(),
                    "source_summary": str(path.relative_to(ROOT)),
                }
                seed_rows.append(row)
                for qrow in item["quantity_rows"]:
                    quantity_seed_rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": spec["label"],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "seed": seed,
                            "stratum_order": qrow["stratum_order"],
                            "stratum": qrow["stratum"],
                            "stratum_label": qrow["stratum_label"],
                            "count": qrow["count"],
                            "share": qrow["share"],
                            "time_nll": qrow["time_nll"],
                            "log_qty_mse": qrow["log_qty_mse"],
                            "qty_mae": qrow["qty_mae"],
                            "qty_rmse": qrow["qty_rmse"],
                            "qty_bias": qrow["qty_bias"],
                        }
                    )
                for hrow in item["history_rows"]:
                    history_seed_rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": spec["label"],
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "seed": seed,
                            "stratum_order": hrow["stratum_order"],
                            "stratum": hrow["stratum"],
                            "stratum_label": hrow["stratum_label"],
                            "count": hrow["count"],
                            "share": hrow["share"],
                            "time_nll": hrow["time_nll"],
                            "log_qty_mse": hrow["log_qty_mse"],
                            "qty_mae": hrow["qty_mae"],
                            "qty_rmse": hrow["qty_rmse"],
                            "qty_bias": hrow["qty_bias"],
                        }
                    )

    aggregate_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            selected = [r for r in seed_rows if r["dataset"] == dataset and r["model"] == model]
            require(len(selected) == 3, f"Expected 3 rows: {dataset}/{model}")
            row: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": DATASETS[dataset]["label"],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n_seeds": 3,
            }
            for name in ("joint_objective", "time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                values = [float(r[name]) for r in selected]
                row[f"{name}_mean"] = sample_mean(values)
                row[f"{name}_std"] = sample_std(values)
            aggregate_rows.append(row)

    paired_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for reference in ("rmtpp", "thp"):
            for seed in SEEDS:
                candidate = next(
                    r
                    for r in seed_rows
                    if r["dataset"] == dataset and r["model"] == "titantpp" and r["seed"] == seed
                )
                baseline = next(
                    r
                    for r in seed_rows
                    if r["dataset"] == dataset and r["model"] == reference and r["seed"] == seed
                )
                row = {
                    "dataset": dataset,
                    "dataset_label": DATASETS[dataset]["label"],
                    "candidate": "titantpp",
                    "candidate_label": MODEL_LABELS["titantpp"],
                    "reference": reference,
                    "reference_label": MODEL_LABELS[reference],
                    "seed": seed,
                }
                for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                    cval = float(candidate[metric])
                    rval = float(baseline[metric])
                    row[f"{metric}_candidate_minus_reference"] = cval - rval
                    row[f"{metric}_relative_improvement_pct"] = rel_improvement(rval, cval)
                    row[f"{metric}_candidate_better"] = str(cval < rval).lower()
                paired_rows.append(row)

    paired_summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for reference in ("rmtpp", "thp"):
            selected = [
                r for r in paired_rows if r["dataset"] == dataset and r["reference"] == reference
            ]
            row = {
                "dataset": dataset,
                "dataset_label": DATASETS[dataset]["label"],
                "candidate": "titantpp",
                "candidate_label": MODEL_LABELS["titantpp"],
                "reference": reference,
                "reference_label": MODEL_LABELS[reference],
                "n_paired_seeds": 3,
            }
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                deltas = [float(r[f"{metric}_candidate_minus_reference"]) for r in selected]
                gains = [float(r[f"{metric}_relative_improvement_pct"]) for r in selected]
                row[f"{metric}_mean_delta"] = sample_mean(deltas)
                row[f"{metric}_std_delta"] = sample_std(deltas)
                row[f"{metric}_mean_paired_relative_improvement_pct"] = sample_mean(gains)
                row[f"{metric}_better_seed_count"] = sum(value > 0.0 for value in gains)
            paired_summary_rows.append(row)

    quantity_aggregate_rows = aggregate_strata(quantity_seed_rows)
    history_aggregate_rows = aggregate_strata(history_seed_rows)
    return (
        seed_rows,
        aggregate_rows,
        paired_rows,
        paired_summary_rows,
        quantity_seed_rows,
        quantity_aggregate_rows,
        history_seed_rows,
        history_aggregate_rows,
        source_paths,
    )


def aggregate_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["stratum"])].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda x: (x[0], x[1], grouped[x][0]["stratum_order"])):
        selected = grouped[key]
        require(len(selected) == 3, f"Expected 3 stratum rows: {key}")
        first = selected[0]
        require(len({int(r["count"]) for r in selected}) == 1, f"Count mismatch: {key}")
        row: dict[str, Any] = {
            "dataset": first["dataset"],
            "dataset_label": first["dataset_label"],
            "model": first["model"],
            "model_label": first["model_label"],
            "stratum_order": first["stratum_order"],
            "stratum": first["stratum"],
            "stratum_label": first["stratum_label"],
            "count_per_seed": first["count"],
            "share": first["share"],
            "n_seeds": 3,
        }
        for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse", "qty_bias"):
            values = [float(r[metric]) for r in selected]
            row[f"{metric}_mean"] = sample_mean(values)
            row[f"{metric}_std"] = sample_std(values)
        output.append(row)
    return output


def build_strata_paired_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute seed-paired Hard-LMM differences within each stored stratum."""
    paired_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        strata = sorted(
            {row["stratum"] for row in rows if row["dataset"] == dataset},
            key=lambda value: next(
                int(row["stratum_order"])
                for row in rows
                if row["dataset"] == dataset and row["stratum"] == value
            ),
        )
        for reference in ("rmtpp", "thp"):
            for stratum in strata:
                for seed in SEEDS:
                    candidate = next(
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["model"] == "titantpp"
                        and row["stratum"] == stratum
                        and row["seed"] == seed
                    )
                    baseline = next(
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["model"] == reference
                        and row["stratum"] == stratum
                        and row["seed"] == seed
                    )
                    require(candidate["count"] == baseline["count"], "Paired stratum count mismatch")
                    require(
                        math.isclose(
                            float(candidate["share"]),
                            float(baseline["share"]),
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        ),
                        "Paired stratum share mismatch",
                    )
                    result: dict[str, Any] = {
                        "dataset": dataset,
                        "dataset_label": DATASETS[dataset]["label"],
                        "candidate": "titantpp",
                        "candidate_label": MODEL_LABELS["titantpp"],
                        "reference": reference,
                        "reference_label": MODEL_LABELS[reference],
                        "seed": seed,
                        "stratum_order": candidate["stratum_order"],
                        "stratum": stratum,
                        "stratum_label": candidate["stratum_label"],
                        "count": candidate["count"],
                        "share": candidate["share"],
                    }
                    for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                        cval = float(candidate[metric])
                        rval = float(baseline[metric])
                        result[f"{metric}_candidate_minus_reference"] = cval - rval
                        result[f"{metric}_relative_improvement_pct"] = rel_improvement(rval, cval)
                        result[f"{metric}_candidate_better"] = str(cval < rval).lower()
                    paired_rows.append(result)

    summary_rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (row["dataset"], row["reference"], row["stratum"])
            for row in paired_rows
        },
        key=lambda key: (
            key[0],
            key[1],
            next(
                int(row["stratum_order"])
                for row in paired_rows
                if row["dataset"] == key[0]
                and row["reference"] == key[1]
                and row["stratum"] == key[2]
            ),
        ),
    )
    for dataset, reference, stratum in keys:
        selected = [
            row
            for row in paired_rows
            if row["dataset"] == dataset
            and row["reference"] == reference
            and row["stratum"] == stratum
        ]
        require(len(selected) == 3, "Expected three paired stratum rows")
        first = selected[0]
        result = {
            "dataset": dataset,
            "dataset_label": DATASETS[dataset]["label"],
            "candidate": "titantpp",
            "candidate_label": MODEL_LABELS["titantpp"],
            "reference": reference,
            "reference_label": MODEL_LABELS[reference],
            "stratum_order": first["stratum_order"],
            "stratum": stratum,
            "stratum_label": first["stratum_label"],
            "count_per_seed": first["count"],
            "share": first["share"],
            "n_paired_seeds": 3,
        }
        for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
            deltas = [float(row[f"{metric}_candidate_minus_reference"]) for row in selected]
            gains = [float(row[f"{metric}_relative_improvement_pct"]) for row in selected]
            result[f"{metric}_mean_delta"] = sample_mean(deltas)
            result[f"{metric}_std_delta"] = sample_std(deltas)
            result[f"{metric}_mean_paired_relative_improvement_pct"] = sample_mean(gains)
            result[f"{metric}_better_seed_count"] = sum(value > 0.0 for value in gains)
        summary_rows.append(result)
    return paired_rows, summary_rows


def build_interface_audit() -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    categorical_dir = ROOT / "paper/results/taxi_quantity_interface_ablation_20260810/source"
    log_dir = ROOT / "paper/results/taxi_positive_regression_control_20260811/log_regression"
    categorical_contract_path = categorical_dir / "launch_contract.json"
    log_contract_path = log_dir / "launch_contract.json"
    categorical_runs_path = categorical_dir / "run_summaries.csv"
    log_runs_path = log_dir / "run_summaries.csv"
    categorical_metrics_path = categorical_dir / "quantity_interface_seed_metrics.csv"
    log_metrics_path = log_dir / "quantity_interface_seed_metrics.csv"
    runner_path = ROOT / "paper/scripts/run_taxi_quantity_interface_ablation.py"
    v07_path = ROOT / "paper/drafts/v0_7_revision_notes.md"
    paths = [
        categorical_contract_path,
        log_contract_path,
        categorical_runs_path,
        log_runs_path,
        categorical_metrics_path,
        log_metrics_path,
        runner_path,
        v07_path,
    ]
    categorical = read_json(categorical_contract_path)
    direct = read_json(log_contract_path)
    categorical_runs = [
        row
        for row in read_csv(categorical_runs_path)
        if row["variant"] in {"uniform_categorical", "quantile_categorical"}
    ]
    direct_runs = read_csv(log_runs_path)
    require(len(categorical_runs) == 6, "Expected six categorical runs")
    require(len(direct_runs) == 3, "Expected three direct-log runs")

    same_early_stop = categorical["early_stopping"] == direct["early_stopping"]
    categorical_interfaces = categorical["interfaces"]
    direct_interface = direct["interfaces"]["log_regression"]
    train_fitted = all(
        categorical_interfaces[name].get("fitted_on") == "train"
        for name in ("uniform_categorical", "quantile_categorical")
    )
    train_representatives = all(
        len(categorical_interfaces[name].get("representatives", []))
        == categorical_interfaces[name].get("bin_count")
        for name in ("uniform_categorical", "quantile_categorical")
    )
    common_config_fields = (
        "dataset",
        "data_sha256",
        "split_rows",
        "seeds",
        "epochs",
        "batch_size",
        "lr",
        "grad_clip",
        "lookback_weeks",
        "max_seq_len",
        "hidden_dim",
        "evaluation_scope",
        "held_out_test_evaluated",
    )
    common_config_equal = all(categorical.get(key) == direct.get(key) for key in common_config_fields)
    audit_checks = [
        {
            "requirement": "same_rmtpp_encoder_family_and_dimensions",
            "passed": True,
            "evidence": "All runs use the RMTPP/PositiveRegressionRMTPP GRU path with hidden_dim=128; the latter subclasses RMTPP and adds a quantity head.",
        },
        {
            "requirement": "same_dataset_split_seeds_budget_and_optimizer_settings",
            "passed": common_config_equal and same_early_stop,
            "evidence": "The contracts share data SHA-256, split row counts, seeds 42/52/62, e300, batch128, lr0.001, grad clip1, lookback168, max_seq_len256, hidden128, min50 and patience60.",
        },
        {
            "requirement": "single_frozen_source_revision",
            "passed": categorical["source_revision"] == direct["source_revision"],
            "evidence": f"Categorical revision={categorical['source_revision']}; direct-log revision={direct['source_revision']}.",
        },
        {
            "requirement": "mark_free_common_history_input",
            "passed": False,
            "evidence": "Categorical variants replace quantity with a learned categorical mark and value_input_mode=none; direct-log retains the original quantity-derived mark plus log10 within-mark residual. Neither path uses the active [log1p(delta_t), log1p(raw_quantity)] token.",
        },
        {
            "requirement": "only_quantity_target_head_and_reconstruction_change",
            "passed": False,
            "evidence": "The history input and mark vocabulary change between categorical and direct-log variants, so the experiment changes more than the output interface.",
        },
        {
            "requirement": "exact_log_transformed_categorical_binning_control",
            "passed": False,
            "evidence": "The stored categorical controls are uniform bins on raw quantity and train-quantile bins on raw quantity; no pure categorical log1p-quantity binning row is present.",
        },
        {
            "requirement": "train_fitted_bin_edges_and_numeric_representatives",
            "passed": train_fitted and train_representatives,
            "evidence": "Both categorical contracts say fitted_on=train and store four train-bin medians as reconstruction representatives.",
        },
        {
            "requirement": "same_checkpoint_selector_identifier",
            "passed": categorical["checkpoint_selection"] == direct["checkpoint_selection"] == "best_val_nll",
            "evidence": "Both contracts name best_val_nll with the same min-epoch/patience rule.",
        },
        {
            "requirement": "same_checkpoint_selector_semantics_and_active_v0_7_selector",
            "passed": False,
            "evidence": "val_nll contains category-dependent mark NLL plus the legacy clamped time loss; the categorical mark definition differs, direct-log quantity MSE is not part of selection, and this is not the active T0 validation_joint_objective selector.",
        },
        {
            "requirement": "raw_scale_rmse_and_mae_reported",
            "passed": True,
            "evidence": "All nine compared runs report validation raw quantity MAE and RMSE.",
        },
        {
            "requirement": "supporting_reconstructed_log_quantity_mse_reported",
            "passed": False,
            "evidence": "The legacy interface artifacts report raw MAE/RMSE/bias and event NLL, but not log1p error of the reconstructed numeric quantity.",
        },
        {
            "requirement": "held_out_test_locked",
            "passed": (
                categorical["evaluation_scope"] == "validation_only"
                and direct["evaluation_scope"] == "validation_only"
                and categorical["held_out_test_evaluated"] is False
                and direct["held_out_test_evaluated"] is False
            ),
            "evidence": "Both contracts and all run summaries are validation_only with held_out_test_evaluated=false.",
        },
    ]
    blocking = [item["requirement"] for item in audit_checks if not item["passed"]]
    audit = {
        "audit_id": "titantpp_v0_7_quantity_interface_existing_evidence_audit_20260905",
        "decision": "FAIL",
        "paper_use": "diagnostic_only",
        "question": "Do the existing Taxi direct-log and categorical artifacts satisfy the v0.7 controlled quantity-interface contract?",
        "answer": "No. They share the dataset, RMTPP family, seeds and budget, and the bins/reconstruction are train-fitted, but the input is not mark-free or common, no exact log-binned categorical control exists, selector semantics are category-dependent, and reconstructed log-MSE is absent.",
        "checks": audit_checks,
        "blocking_requirements": blocking,
        "exact_missing_work": [
            "Run direct log regression and a pure log1p-binned categorical head from one frozen source revision on the same RMTPP encoder.",
            "Feed both variants the active mark-free observed token [log1p(delta_t), log1p(raw_quantity)] so only the prediction head, loss and reconstruction change.",
            "Freeze one category-independent checkpoint rule before execution; the legacy mark-NLL selector is not comparable across changing bin definitions.",
            "Fit log-bin boundaries and every numeric reconstruction representative on train only, then record them in the launch contract.",
            "Report raw RMSE, raw MAE and reconstructed log1p quantity MSE on identical validation targets and train-derived quantity strata.",
            "Add Instacart only if the manuscript makes a cross-dataset empirical superiority claim for the count interface; the existing controlled interface evidence covers Taxi only.",
        ],
        "scope": {
            "dataset": "Taxi only",
            "evaluation": "validation_only",
            "held_out_test_evaluated": False,
            "categorical_source_revision": categorical["source_revision"],
            "direct_log_source_revision": direct["source_revision"],
        },
    }

    metric_rows: list[dict[str, Any]] = []
    combined_runs = categorical_runs + direct_runs
    for variant in ("uniform_categorical", "quantile_categorical", "log_regression"):
        selected = [row for row in combined_runs if row["variant"] == variant]
        require(len(selected) == 3, variant)
        for metric in ("best_val_nll", "best_val_qty_mae", "best_val_qty_rmse"):
            values = [float(row[metric]) for row in selected]
            metric_rows.append(
                {
                    "dataset": "taxi",
                    "variant": variant,
                    "metric": metric,
                    "n_seeds": 3,
                    "mean": sample_mean(values),
                    "sample_std": sample_std(values),
                    "qualification": "diagnostic_only_contract_fail",
                }
            )
    return audit, metric_rows, paths


def build_dataset_identity_audit() -> tuple[dict[str, Any], list[Path]]:
    """Check that manuscript population labels resolve to the evaluated artifacts."""
    table_path = ROOT / "paper/tables/T1_dataset_statistics.csv"
    intermittent_contract_path = (
        ROOT
        / "paper/results/intermittent_log_backbone_control_20260811/"
        "source_5080_baselines/launch_contract.json"
    )
    intermittent_split_path = (
        ROOT / "sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json"
    )
    paths = [table_path, intermittent_contract_path, intermittent_split_path]

    table_rows = {row["dataset"]: row for row in read_csv(table_path)}
    checks: list[dict[str, Any]] = []
    for dataset, table_label in (("taxi", "Taxi"), ("instacart", "Instacart")):
        launch_path = DATASETS[dataset]["root"] / "launch_contract.json"
        launch = read_json(launch_path)
        table_row_value = table_rows[table_label]
        same_data = table_row_value["with_split_sha256"] == launch["data_sha256"]
        same_split = (
            table_row_value["split_manifest_sha256"] == launch["split_manifest_sha256"]
        )
        checks.append(
            {
                "dataset": dataset,
                "table_dataset_id": table_row_value["dataset_id"],
                "evaluated_dataset_id": launch["dataset"],
                "decision": "PASS_SAME_ARTIFACT_IDENTITY"
                if same_data and same_split
                else "FAIL_ARTIFACT_IDENTITY_MISMATCH",
                "same_data_sha256": same_data,
                "same_split_manifest_sha256": same_split,
                "table_data_sha256": table_row_value["with_split_sha256"],
                "evaluated_data_sha256": launch["data_sha256"],
                "table_split_manifest_sha256": table_row_value["split_manifest_sha256"],
                "evaluated_split_manifest_sha256": launch["split_manifest_sha256"],
                "table_split_rows": {
                    "train": int(table_row_value["events_train"]),
                    "validation": int(table_row_value["events_validation"]),
                    "test": int(table_row_value["events_test"]),
                },
                "evaluated_split_rows": launch["split_rows"],
            }
        )

    intermittent_table = table_rows["Intermittent"]
    intermittent_contract = read_json(intermittent_contract_path)
    intermittent_split = read_json(intermittent_split_path)
    split_counts = {
        row["chronological_split"]: {
            "rows": int(row["rows"]),
            "series": int(row["series"]),
        }
        for row in intermittent_split["summary"]["split_counts"]
    }
    frozen_rows = {
        "train": split_counts["train"]["rows"],
        "validation": split_counts["validation"]["rows"],
        "test": split_counts["test"]["rows"],
    }
    require(frozen_rows == intermittent_contract["split_rows"], "Intermittent split drift")
    frozen_series = intermittent_split["summary"]["sequence_length_summary"]["series"]
    frozen_total_rows = sum(frozen_rows.values())
    same_data = intermittent_table["with_split_sha256"] == intermittent_contract["data_sha256"]
    same_split = (
        intermittent_table["split_manifest_sha256"]
        == intermittent_contract["split_manifest_sha256"]
    )
    same_population_counts = (
        int(intermittent_table["sequences"]) == frozen_series
        and int(intermittent_table["events_total"]) == frozen_total_rows
        and {
            "train": int(intermittent_table["events_train"]),
            "validation": int(intermittent_table["events_validation"]),
            "test": int(intermittent_table["events_test"]),
        }
        == frozen_rows
    )
    require(
        not (same_data and same_split and same_population_counts),
        "Expected the known Intermittent population mismatch",
    )
    checks.append(
        {
            "dataset": "intermittent",
            "table_dataset_id": intermittent_table["dataset_id"],
            "evaluated_dataset_id": intermittent_contract["dataset"],
            "decision": "FAIL_POPULATION_IDENTITY_MISMATCH",
            "same_data_sha256": same_data,
            "same_split_manifest_sha256": same_split,
            "same_population_counts": same_population_counts,
            "table_population": {
                "sequences": int(intermittent_table["sequences"]),
                "events_total": int(intermittent_table["events_total"]),
                "split_rows": {
                    "train": int(intermittent_table["events_train"]),
                    "validation": int(intermittent_table["events_validation"]),
                    "test": int(intermittent_table["events_test"]),
                },
                "data_sha256": intermittent_table["with_split_sha256"],
                "split_manifest_sha256": intermittent_table["split_manifest_sha256"],
            },
            "evaluated_population": {
                "sequences": frozen_series,
                "events_total": frozen_total_rows,
                "split_rows": frozen_rows,
                "data_sha256": intermittent_contract["data_sha256"],
                "split_manifest_sha256": intermittent_contract["split_manifest_sha256"],
            },
            "required_action": (
                "Do not combine the legacy T1 Intermittent statistics row with "
                "intermittent_frozen_5000 results. Rebuild the row from the frozen artifact "
                "or omit Intermittent from the v0.7 integrated table until identity is frozen."
            ),
        }
    )

    require(
        all(item["decision"] == "PASS_SAME_ARTIFACT_IDENTITY" for item in checks[:2]),
        "Taxi/Instacart table identity mismatch",
    )
    audit = {
        "audit_id": "titantpp_v0_7_dataset_identity_audit_20260905",
        "v0_7_three_dataset_identity_decision": "FAIL_INTERMITTENT_POPULATION_MISMATCH",
        "taxi_instacart_integrated_table_decision": "PASS_SAME_ARTIFACT_IDENTITIES_ONLY",
        "integrated_validation_table_datasets": ["taxi", "instacart"],
        "intermittent_included_in_integrated_validation_table": False,
        "checks": checks,
        "paper_rule": (
            "An aggregate/statistics row and a model result may be integrated only when their "
            "data and split artifact identities match."
        ),
    }
    return audit, paths


def build_final_model_evidence_audit(
    validation_seed_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[Path]]:
    separate_dir = ROOT / "paper/results/hard_lmm_key_value_screening_5090_20260904"
    separate_audit_path = separate_dir / "final_audit.json"
    separate_readme_path = separate_dir / "README.md"
    separate = read_json(separate_audit_path)
    original_seed_coverage = {
        dataset: sorted(
            int(row["seed"])
            for row in validation_seed_rows
            if row["dataset"] == dataset and row["model"] == "titantpp"
        )
        for dataset in DATASETS
    }
    require(
        all(seeds == list(SEEDS) for seeds in original_seed_coverage.values()),
        "Original T0 lacks required three-seed coverage",
    )
    separate_seed = int(separate["contract"]["seed"])
    separate_dataset_coverage = sorted(separate["runs"])
    require(separate_dataset_coverage == ["insta_market_basket", "yellow_trip_hourly"], "Separate-key dataset drift")
    require(separate["artifact_checks"]["held_out_test_evaluated"] is False, "Separate-key held-out leak")
    require(separate["decision"]["candidate_accepted_for_expansion"] is False, "Separate-key decision drift")
    audit = {
        "audit_id": "titantpp_v0_7_common_final_model_evidence_audit_20260905",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "required_common_datasets": ["taxi", "instacart"],
        "required_seeds": list(SEEDS),
        "candidates": {
            "original_mark_free_hard_lmm_t0": {
                "decision": "PASS_THREE_SEED_COVERAGE_ON_BOTH_DATASETS",
                "dataset_seed_coverage": original_seed_coverage,
                "same_model_role": "t0_common_control",
                "eligible_as_common_final_structure": True,
                "performance_qualification": (
                    "Coverage passes, but common performance superiority does not: Taxi has a "
                    "mean-RMSE-only trade-off and Instacart does not establish a Hard-LMM advantage."
                ),
            },
            "separate_key_hard_lmm": {
                "decision": "FAIL_THREE_SEED_COVERAGE_AND_PROSPECTIVE_CROSS_DATASET_GATE",
                "dataset_seed_coverage": {
                    "taxi": [separate_seed],
                    "instacart": [separate_seed],
                },
                "dataset_gates_passed": separate["decision"]["dataset_gates_passed"],
                "dataset_gates_evaluated": separate["decision"]["dataset_gates_evaluated"],
                "candidate_accepted_for_expansion": separate["decision"][
                    "candidate_accepted_for_expansion"
                ],
                "rejection_reason": separate["decision"]["reason"],
                "eligible_as_common_final_structure": False,
                "paper_use": "single_seed_diagnostic_only",
            },
        },
        "final_structure_decision": "FREEZE_ORIGINAL_MARK_FREE_HARD_LMM_T0",
        "separate_key_in_main_tables": False,
    }
    return audit, [separate_audit_path, separate_readme_path]


def table_row(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def build_paper_tables(
    aggregates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    quantity: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> str:
    lines = [
        "# TitanTPP v0.7 validation tables",
        "",
        "> Validation only; mean ± sample standard deviation over seeds 42, 52, and 62. Lower is better. No held-out test result is included.",
        "",
    ]
    for dataset in DATASETS:
        label = DATASETS[dataset]["label"]
        lines.extend(
            [
                f"## {label}: common T0 aggregate",
                "",
                table_row(["Model", "Clamped time loss", "Log-quantity MSE", "Quantity MAE", "Quantity RMSE"]),
                table_row(["---", "---:", "---:", "---:", "---:"]),
            ]
        )
        for model in MODELS:
            row = next(r for r in aggregates if r["dataset"] == dataset and r["model"] == model)
            lines.append(
                table_row(
                    [
                        MODEL_LABELS[model],
                        f"{fmt(row['time_nll_mean'])} ± {fmt(row['time_nll_std'])}",
                        f"{fmt(row['log_qty_mse_mean'])} ± {fmt(row['log_qty_mse_std'])}",
                        f"{fmt(row['qty_mae_mean'])} ± {fmt(row['qty_mae_std'])}",
                        f"{fmt(row['qty_rmse_mean'])} ± {fmt(row['qty_rmse_std'])}",
                    ]
                )
            )
        lines.extend(["", f"## {label}: paired Hard-LMM comparison", ""])
        lines.extend(
            [
                table_row(["Reference", "Clamped-time Δ / wins", "Log-MSE Δ / wins", "MAE Δ / wins", "RMSE Δ / wins"]),
                table_row(["---", "---:", "---:", "---:", "---:"]),
            ]
        )
        for reference in ("rmtpp", "thp"):
            row = next(r for r in paired if r["dataset"] == dataset and r["reference"] == reference)
            cells = [MODEL_LABELS[reference]]
            for metric in ("time_nll", "log_qty_mse", "qty_mae", "qty_rmse"):
                cells.append(
                    f"{fmt(row[f'{metric}_mean_delta'])} / {row[f'{metric}_better_seed_count']}/3"
                )
            lines.append(table_row(cells))
        lines.extend(
            [
                "",
                "Δ is Hard-LMM minus reference; negative is better for Hard-LMM. Wins count paired seeds with a lower Hard-LMM value.",
                "",
                f"## {label}: quantity-stratified MAE / RMSE",
                "",
                table_row(["Train-derived stratum", "RMTPP", "THP", "Hard-LMM"]),
                table_row(["---", "---:", "---:", "---:"]),
            ]
        )
        strata = sorted(
            {r["stratum"] for r in quantity if r["dataset"] == dataset},
            key=lambda value: next(
                r["stratum_order"]
                for r in quantity
                if r["dataset"] == dataset and r["stratum"] == value
            ),
        )
        for stratum in strata:
            first = next(r for r in quantity if r["dataset"] == dataset and r["stratum"] == stratum)
            cells = [first["stratum_label"]]
            for model in MODELS:
                row = next(
                    r
                    for r in quantity
                    if r["dataset"] == dataset and r["stratum"] == stratum and r["model"] == model
                )
                cells.append(f"{fmt(row['qty_mae_mean'], 3)} / {fmt(row['qty_rmse_mean'], 3)}")
            lines.append(table_row(cells))
        lines.append("")

    taxi_history = [r for r in history if r["dataset"] == "taxi"]
    lines.extend(
        [
            "## Taxi: history-stratified MAE / RMSE",
            "",
            table_row(["History stratum", "RMTPP", "THP", "Hard-LMM"]),
            table_row(["---", "---:", "---:", "---:"]),
        ]
    )
    strata = sorted(
        {r["stratum"] for r in taxi_history},
        key=lambda value: next(r["stratum_order"] for r in taxi_history if r["stratum"] == value),
    )
    for stratum in strata:
        first = next(r for r in taxi_history if r["stratum"] == stratum)
        cells = [first["stratum_label"]]
        for model in MODELS:
            row = next(r for r in taxi_history if r["stratum"] == stratum and r["model"] == model)
            cells.append(f"{fmt(row['qty_mae_mean'], 3)} / {fmt(row['qty_rmse_mean'], 3)}")
        lines.append(table_row(cells))
    lines.extend(
        [
            "",
            "Instacart has one history stratum (history ≤64) for all validation targets, so it cannot test a long-history mechanism.",
            "",
        ]
    )
    return "\n".join(lines)


def build_readme(
    aggregates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    interface_audit: dict[str, Any],
    dataset_identity_audit: dict[str, Any],
    final_model_audit: dict[str, Any],
) -> str:
    taxi_r = next(r for r in aggregates if r["dataset"] == "taxi" and r["model"] == "rmtpp")
    taxi_h = next(r for r in aggregates if r["dataset"] == "taxi" and r["model"] == "titantpp")
    insta_r = next(r for r in aggregates if r["dataset"] == "instacart" and r["model"] == "rmtpp")
    insta_h = next(r for r in aggregates if r["dataset"] == "instacart" and r["model"] == "titantpp")
    taxi_pair = next(r for r in paired if r["dataset"] == "taxi" and r["reference"] == "rmtpp")
    insta_pair = next(r for r in paired if r["dataset"] == "instacart" and r["reference"] == "rmtpp")
    return f"""# TitanTPP v0.7 validation evidence freeze

**판정: 원본 mark-free Hard-LMM T0는 Taxi와 Instacart에 동일한 모델 계열로 적용된 3-seed validation 증적이 있다. 그러나 두 데이터셋에서 공통 우위를 보이지 않으므로, 이 결과는 조건부 관찰로만 사용한다. Held-out test는 사용하지 않았다.**

## 통합 결과

- 입력, 수량 head, loss, time head와 checkpoint 규칙을 고정하고 RMTPP, THP, 원본 Hard-LMM만 비교했다. 각 데이터셋은 seeds 42, 52, 62와 e300 상한을 사용한다.
- Taxi에서 Hard-LMM의 평균 RMSE는 `{taxi_h['qty_rmse_mean']:.6f}`로 RMTPP `{taxi_r['qty_rmse_mean']:.6f}`보다 `{rel_improvement(taxi_r['qty_rmse_mean'], taxi_h['qty_rmse_mean']):.3f}%` 낮다. 하지만 paired seed 승수는 `{taxi_pair['qty_rmse_better_seed_count']}/3`이고, MAE는 `{taxi_h['qty_mae_mean']:.6f}`로 RMTPP `{taxi_r['qty_mae_mean']:.6f}`보다 악화했다. Clamped time loss와 log-MSE도 세 seed 모두 RMTPP보다 높다.
- Instacart에서 Hard-LMM의 평균 MAE/RMSE는 `{insta_h['qty_mae_mean']:.6f}` / `{insta_h['qty_rmse_mean']:.6f}`이고 RMTPP는 `{insta_r['qty_mae_mean']:.6f}` / `{insta_r['qty_rmse_mean']:.6f}`이다. Hard-LMM은 MAE에서 `{insta_pair['qty_mae_better_seed_count']}/3`, RMSE에서 `{insta_pair['qty_rmse_better_seed_count']}/3` seed만 이겼다.
- Taxi의 `history >128`에서도 Hard-LMM RMSE 개선은 RMTPP 대비 2/3 seed이며 MAE는 1/3 seed다. Instacart validation target은 전부 `history <=64`이다. 따라서 **긴 이력 자체가 개선 원인이라는 주장은 이 결과로 지지되지 않는다.**
- 허용되는 서술은 “Taxi에서 평균 RMSE 이점이 관찰됐지만 seed-stable한 전 지표 우위는 아니며, 짧은 Instacart에서는 backbone 우위가 확립되지 않았다”이다.

상세 표는 [paper_tables.md](paper_tables.md), seed 원자료는 [validation_seed_metrics.csv](validation_seed_metrics.csv), paired 차이는 [paired_seed_deltas.csv](paired_seed_deltas.csv)에 있다. Quantity/history 구간 집계는 [quantity_strata_aggregate.csv](quantity_strata_aggregate.csv)와 [history_strata_aggregate.csv](history_strata_aggregate.csv), 같은 seed 안의 Hard-LMM 차이는 [quantity_strata_paired_seed_deltas.csv](quantity_strata_paired_seed_deltas.csv)와 [history_strata_paired_seed_deltas.csv](history_strata_paired_seed_deltas.csv)에 있다.

## 공통 최종 구조 판정

공통 최종 구조 판정은 **{final_model_audit['final_structure_decision']}**이다. 원본 mark-free Hard-LMM T0는 Taxi와 Instacart 모두 seeds 42/52/62가 있다. Separate-key Hard-LMM은 두 데이터셋 모두 seed 42만 있고, 사전 고정한 교차 데이터셋 gate에서 Instacart가 실패해 추가 seed 확장도 거부됐다. 따라서 separate-key에는 두 데이터셋 공통 최종 구조로 사용할 3-seed 증적이 없으며, single-seed 진단으로만 남긴다. 기계 판독 가능한 판정은 [final_model_evidence_audit.json](final_model_evidence_audit.json)에 있다.

## 데이터셋 identity 재감사

v0.7의 3개 데이터셋 통합 identity 판정은 **{dataset_identity_audit['v0_7_three_dataset_identity_decision']}**이다. Taxi와 Instacart는 `T1_dataset_statistics.csv`의 data/split SHA-256이 이번 T0 평가 계약과 일치하므로, 이 freeze의 통합표는 **Taxi와 Instacart만** 같은 artifact identity로 묶었다.

Intermittent는 일치하지 않는다. 기존 T1 행은 `intermittent` 23,387 sequences / 242,888 events이고 split event 수는 159,643 / 41,901 / 41,344이다. 반면 mark-free 비교 계약은 `intermittent_frozen_5000` 5,000 series / 573,128 rows이며 split row 수는 398,824 / 86,285 / 88,019이다. data SHA-256과 split-manifest SHA-256도 모두 다르다. 따라서 기존 T1 Intermittent 통계와 frozen-5000 결과를 같은 population으로 인용할 수 없다. 후속 v0.7 통합 단계에서는 frozen artifact에서 [새 T1 행](../../tables/T1_v0_7_dataset_statistics.md)을 다시 계산해 이 문제를 해소했다. 상세 탐지 결과는 [dataset_identity_audit.json](dataset_identity_audit.json)에 보존한다.

## Quantity-interface 재감사

기존 Taxi direct-log 대 categorical 결과의 v0.7 계약 판정은 **{interface_audit['decision']}**이며 논문 자격은 `{interface_audit['paper_use']}`이다.

충족한 항목은 동일 Taxi split, RMTPP 계열, seeds 42/52/62, e300, batch와 optimizer 설정, train-only bin fitting 및 train median reconstruction, validation-only/held-out lock이다.

차단 항목은 다음과 같다.

1. Categorical 경로는 다시 만든 quantity mark만 입력하고, direct-log 경로는 기존 magnitude mark와 within-mark residual을 입력한다. 현재 mark-free 토큰 `[log1p(delta_t), log1p(raw_quantity)]`을 공유하지 않는다.
2. 저장된 categorical 후보는 raw-scale uniform bin과 raw-scale train-quantile bin이다. v0.7이 요구한 순수 log-transformed categorical binning이 없다.
3. 두 계약은 source revision이 다르다.
4. Selector 이름은 모두 `best_val_nll`이지만 category-dependent mark NLL의 의미가 variant별로 달라지고 direct-log quantity MSE는 선택에 포함되지 않는다. 현재 T0의 validation joint selector와도 다르다.
5. 재구성 수량의 raw MAE/RMSE는 있지만 supporting log1p-MSE가 없다.
6. 이 controlled interface 결과는 Taxi만 다룬다.

따라서 기존 결과는 direct log가 유망하다는 진단에는 사용할 수 있지만, mark-free formulation의 empirical superiority를 입증하는 v0.7 ablation으로 사용할 수 없다. 전체 판정과 정확한 누락은 [quantity_interface_audit.json](quantity_interface_audit.json)에 기록했다.

## 증적 경계

- 모든 집계는 저장된 validation summary와 validation stratum row에서 수행했다. Test row는 읽거나 평가하지 않았다.
- Builder는 각 run의 `held_out_test_evaluated=false`, `evaluation_scope=validation_only`, mark-free interface와 joint-objective 산식을 확인한다.
- 원천 필드 `time_nll`은 `legacy_clamped_rmtpp` score의 음의 평균이다. `w * delta_t` 제한 때문에 exact likelihood가 아니므로 표에서는 clamped time loss로 부른다.
- [time_head_revision_audit.json](time_head_revision_audit.json)은 세 실행 revision의 intercept cap 300 / `w * delta_t` cap 10과 현재 기본값 30의 차이를 Git blob에서 확인한다.
- [independent_verification.json](independent_verification.json)은 별도 CSV 경로로 모든 seed/aggregate/stratum 수치를 재계산한다.
- [source_manifest.json](source_manifest.json)은 읽은 계약, summary, CSV, 구현 근거의 SHA-256을 고정한다.

## 후속 논문 동기화 결과와 남은 작업

원본 Hard-LMM T0, RMTPP·THP·TitanTPP 모델 행, metric과 checkpoint 규칙은
[최종 계약](../../contracts/titantpp_v0_7_final_model_claim_contract_v1.md)에 동결됐다.
Quantity-interface empirical superiority는 현재 기여에서 제외했고, mark-free F1,
dataset-level validation F2와 [artifact manifest](../../manifests/titantpp_v0_7_validation_freeze_manifest.md)를
생성했다. 이 freeze 이후 남은 필수 실험은 별도 승인을 받은 one-time held-out
평가뿐이다. 평가 후에는 결과에 따라 주장을 유지하거나 축소하고, 모델이나
checkpoint를 다시 선택하지 않는다.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (
        seed_rows,
        aggregates,
        paired_rows,
        paired_summary,
        quantity_seed,
        quantity_aggregate,
        history_seed,
        history_aggregate,
        validation_sources,
    ) = build_validation_rows()
    interface_audit, interface_metrics, interface_sources = build_interface_audit()
    dataset_identity_audit, identity_sources = build_dataset_identity_audit()
    final_model_audit, final_model_sources = build_final_model_evidence_audit(seed_rows)
    quantity_paired, quantity_paired_summary = build_strata_paired_rows(quantity_seed)
    history_paired, history_paired_summary = build_strata_paired_rows(history_seed)

    write_csv(OUT / "validation_seed_metrics.csv", seed_rows)
    write_csv(OUT / "validation_aggregate.csv", aggregates)
    write_csv(OUT / "paired_seed_deltas.csv", paired_rows)
    write_csv(OUT / "paired_delta_summary.csv", paired_summary)
    write_csv(OUT / "quantity_strata_seed_metrics.csv", quantity_seed)
    write_csv(OUT / "quantity_strata_aggregate.csv", quantity_aggregate)
    write_csv(OUT / "quantity_strata_paired_seed_deltas.csv", quantity_paired)
    write_csv(OUT / "quantity_strata_paired_delta_summary.csv", quantity_paired_summary)
    write_csv(OUT / "history_strata_seed_metrics.csv", history_seed)
    write_csv(OUT / "history_strata_aggregate.csv", history_aggregate)
    write_csv(OUT / "history_strata_paired_seed_deltas.csv", history_paired)
    write_csv(OUT / "history_strata_paired_delta_summary.csv", history_paired_summary)
    write_csv(OUT / "quantity_interface_observed_metrics.csv", interface_metrics)
    write_json(OUT / "quantity_interface_audit.json", interface_audit)
    write_json(OUT / "dataset_identity_audit.json", dataset_identity_audit)
    write_json(OUT / "final_model_evidence_audit.json", final_model_audit)
    (OUT / "paper_tables.md").write_text(
        build_paper_tables(aggregates, paired_summary, quantity_aggregate, history_aggregate),
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        build_readme(
            aggregates,
            paired_summary,
            interface_audit,
            dataset_identity_audit,
            final_model_audit,
        ),
        encoding="utf-8",
    )

    sources = sorted(
        set(validation_sources + interface_sources + identity_sources + final_model_sources),
        key=lambda path: str(path),
    )
    source_manifest = {
        "manifest_id": "titantpp_v0_7_validation_freeze_sources_20260905",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sources
        },
    }
    write_json(OUT / "source_manifest.json", source_manifest)
    print(f"Wrote v0.7 validation freeze to {OUT}")


if __name__ == "__main__":
    main()
