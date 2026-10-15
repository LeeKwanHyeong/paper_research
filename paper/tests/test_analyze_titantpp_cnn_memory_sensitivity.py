"""Synthetic metric and stratification checks; no research artifacts are read."""
import hashlib
import json

import polars as pl
import pytest

from paper.scripts.analyze_titantpp_cnn_memory_sensitivity import (
    ALPHAS, DATASETS, FEATURES, METRICS, SEEDS, aggregate_seed_rows, average_sd,
    load_predictions, strata_statistics, support_rules, train_quartiles,
    verify_dispatchers, verify_values_and_metrics,
)


def feature_frame(values, prediction=None, reference=None, available=None):
    count = len(values)
    data = {feature: values for feature in FEATURES}
    data.update(encoded_norm=[1.0] * count, memory_norm=[1.0] * count,
                previous_observation_available=[True] * count if available is None else available,
                raw_quantity=[10.0] * count,
                prediction_1=[12.0] * count if reference is None else reference)
    for alpha in ALPHAS[1:]:
        data[f"prediction_{alpha:g}"] = [11.0] * count if prediction is None else prediction
    return pl.DataFrame(data).lazy()


def test_train_cuts_are_reused_without_validation_refitting():
    train = feature_frame([0.0, 1.0, 2.0, 3.0])
    cuts = train_quartiles(train)
    assert cuts["history_log_std"]["cuts"] == [0.75, 1.5, 2.25]
    validation = feature_frame([100.0, 101.0, 102.0, 103.0])
    cells = strata_statistics(validation, cuts)
    chosen = [row for row in cells if row["feature"] == "history_log_std" and row["alpha"] == 0.5]
    assert next(row for row in chosen if row["quartile"] == 3)["count"] == 4
    assert all(row["count"] == 0 for row in chosen if row["quartile"] != 3)


def test_paired_mse_mae_bias_are_computed_per_event():
    frame = feature_frame([0.0, 1.0, 2.0, 3.0], prediction=[11.0, 13.0, 9.0, 8.0],
                          reference=[12.0, 11.0, 12.0, 11.0])
    cells = strata_statistics(frame, train_quartiles(frame))
    chosen = {row["quartile"]: row for row in cells if row["feature"] == "history_log_std" and row["alpha"] == 0.25}
    assert (chosen[0]["delta_mse"], chosen[0]["delta_mae"], chosen[0]["delta_bias"]) == (-3.0, -1.0, -1.0)
    assert (chosen[1]["delta_mse"], chosen[1]["delta_mae"], chosen[1]["delta_bias"]) == (8.0, 2.0, 2.0)
    assert chosen[3]["delta_mse"] == 3.0
    assert chosen[3]["delta_bias"] == -3.0


def test_tied_quartiles_preserve_empty_groups():
    frame = feature_frame([5.0] * 4)
    cuts = train_quartiles(frame)
    assert cuts["history_log_std"]["cuts"] == [5.0, 5.0, 5.0]
    cells = [row for row in strata_statistics(frame, cuts)
             if row["feature"] == "history_log_std" and row["alpha"] == 0.5]
    assert next(row for row in cells if row["quartile"] == 0)["count"] == 4
    assert all(row["count"] == 0 and row["delta_mse"] is None for row in cells if row["quartile"] != 0)


def test_unavailable_observations_do_not_become_false_zero_features():
    frame = feature_frame([0.0, 2.0, 3.0, 4.0], available=[False, True, True, True])
    cuts = train_quartiles(frame)
    assert cuts["last_observed_log_change"]["cuts"] == [2.5, 3.0, 3.5]
    cells = strata_statistics(frame, cuts)
    unavailable = next(row for row in cells if row["feature"] == "last_observed_log_change"
                       and row["quartile"] == -1 and row["alpha"] == 0.5)
    assert unavailable["count"] == 1


def test_sample_sd_and_seed_aggregation():
    assert average_sd([1.0, 2.0, 3.0]) == {"mean": 2.0, "sample_sd": 1.0, "seed_count": 3}
    rows = [{"dataset": DATASETS[0], "split": "validation", "alpha": 0.5, "seed": seed,
             "metrics": {metric: value for metric in METRICS},
             "paired_delta_alpha_minus_one": {metric: value - 2 for metric in METRICS}}
            for seed, value in zip(SEEDS, [1.0, 2.0, 3.0])]
    result = aggregate_seed_rows(rows)[0]
    assert result["metrics"]["qty_rmse"]["sample_sd"] == 1.0
    assert result["paired_delta_alpha_minus_one"]["qty_rmse"]["mean"] == 0.0
    with pytest.raises(ValueError, match="Repeated/missing seed"):
        aggregate_seed_rows([rows[0], rows[0], rows[2]])
    undefined = average_sd([1.0, None, 3.0])
    assert undefined["mean"] is None and undefined["sample_sd"] is None
    assert undefined["available_seed_count"] == 2
    with pytest.raises(ValueError, match="finite"):
        average_sd([1.0, float("inf"), 3.0])


def seed_fixture():
    rows = []
    for dataset in DATASETS:
        for seed, index in zip(SEEDS, range(3)):
            for alpha in ALPHAS:
                value = 10.0
                if dataset == DATASETS[0] and alpha == 0.5:
                    value = [9.0, 9.0, 10.1][index]
                if dataset == DATASETS[0] and alpha == 0.25:
                    value = [9.99, 9.99, 10.0][index]
                rows.append({"dataset": dataset, "seed": seed, "split": "validation", "alpha": alpha,
                             "metrics": {metric: value for metric in METRICS}})
    return rows


def test_support_requires_same_feature_alpha_and_quartile_pair_across_seeds():
    rows = seed_fixture()
    strata = []
    for seed in (42, 52):
        for quartile, delta in ((0, -2.0), (3, 1.0)):
            strata.append({"dataset": DATASETS[0], "seed": seed, "split": "validation",
                           "alpha": 0.5, "feature": "history_log_std", "quartile": quartile,
                           "count": 5, "delta_mse": delta})
    result = support_rules(rows, strata)
    assert result["any_fixed_support"] is True and result["any_adaptive_support"] is True
    matches = [row for row in result["adaptive_stratum_pairs"] if row["supported"]]
    assert len(matches) == 1
    assert matches[0]["matching_seeds"] == [42, 52]
    assert matches[0]["benefit_quartile"] == 0 and matches[0]["harm_quartile"] == 3
    small = next(row for row in result["fixed_attenuation"] if row["dataset"] == DATASETS[0] and row["alpha"] == 0.25)
    assert small["improved_seeds"] == [42, 52] and small["supported"] is False
    strata[-1]["quartile"] = 2  # Another harmful bin for seed52 cannot support the same pair.
    assert support_rules(rows, strata)["any_adaptive_support"] is False


def test_heterogeneity_without_fixed_support_is_not_adaptive_support():
    strata = [{"dataset": DATASETS[1], "seed": seed, "split": "validation", "alpha": 0.5,
               "feature": "history_log_std", "quartile": quartile, "count": 5, "delta_mse": delta}
              for seed in (42, 52) for quartile, delta in ((0, -1.0), (3, 1.0))]
    result = support_rules(seed_fixture(), strata)
    assert any(row["heterogeneity_supported"] for row in result["adaptive_stratum_pairs"])
    assert result["any_adaptive_support"] is False


def test_receipt_metrics_reconstructed_from_predictions_and_strict_tail():
    data = {"raw_quantity": [1.0, 2.0, 3.0], "dataset": ["synthetic"] * 3,
            "seed": [42] * 3, "split": ["train"] * 3,
            "previous_observation_available": [True] * 3, "encoded_norm": [1.0] * 3,
            "memory_norm": [1.0] * 3, **{feature: [0.5] * 3 for feature in FEATURES}}
    metrics = {}
    for alpha in ALPHAS:
        data[f"prediction_{alpha:g}"] = [2.0, 3.0, 4.0]
        data[f"time_nll_{alpha:g}"] = [1.0, 2.0, 3.0]
        metrics[str(alpha)] = {"count": 3, "qty_rmse": 1.0, "qty_mae": 1.0, "time_nll": 2.0,
                               "quantity_boundaries": [0, 0.5, 1, 2],
                               "tail": {"count": 1, "qty_rmse": 1.0, "qty_mae": 1.0, "time_nll": 3.0}}
    receipt = {"population": {"target_count": 3}, "metrics": metrics}
    job = {"id": "synthetic", "dataset": "synthetic", "seed": 42}
    verify_values_and_metrics(pl.DataFrame(data).lazy(), receipt, job, "train", 2)
    null_data = {**data, "seed": [None, 42, 42]}
    with pytest.raises(ValueError, match="mislabeled prediction"):
        verify_values_and_metrics(pl.DataFrame(null_data).lazy(), receipt, job, "train", 2)
    receipt["metrics"]["0.5"]["tail"]["count"] = 2
    with pytest.raises(ValueError, match="Metric counts differ"):
        verify_values_and_metrics(pl.DataFrame(data).lazy(), receipt, job, "train", 2)


def test_prediction_part_tampering_is_rejected_before_analysis(tmp_path):
    directory = tmp_path / "results/synthetic/train"
    predictions = directory / "predictions"
    predictions.mkdir(parents=True)
    part = predictions / "part-00000.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(part)
    original = {"checkpoint": {"sha256": "binary"}, "checkpoint_state_sha256": "state", "selected_epoch": 5}
    receipt = {"schema": "titantpp_cnn_memory_sensitivity_v1", "status": "complete", "job_id": "synthetic",
               "split": "train", "host": "5080", "contract_sha256": "contract", "parameters_unchanged": True,
               "no_training": True, "held_out_test_evaluated": False, "checkpoint_sha256": "binary",
               "state_sha256": "state", "selected_epoch": 5, "metrics": {str(a): {} for a in ALPHAS},
               "prediction_manifest": {"parts": [{"path": part.name, "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                   "rows": 1, "bytes": part.stat().st_size}], "prediction_rows": 1}, "population": {"target_count": 1}}
    path = directory / "receipt.json"
    path.write_text(json.dumps(receipt))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    part.write_bytes(part.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="part bytes changed"):
        load_predictions(tmp_path, "contract", {"id": "synthetic", "host": "5080", "original": original}, "train", expected)


@pytest.mark.parametrize("damage", ["exit", "sha", "missing"])
def test_dispatcher_evidence_requires_every_actual_exit_and_receipt_sha(tmp_path, damage):
    contract = {"jobs": [{"id": "a", "host": "5080"}, {"id": "b", "host": "5090"}]}
    dispatchers = {}
    for job in contract["jobs"]:
        workers = []
        for split in ("train", "validation"):
            relative = f"results/{job['id']}/{split}/receipt.json"
            path = tmp_path / relative
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            workers.append({"job_id": job["id"], "split": split, "status": "complete",
                            "actual_worker_exit_code": 0,
                            "receipt": {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
        dispatchers[job["host"]] = {"schema": "titantpp_cnn_memory_sensitivity_v1", "status": "complete",
            "host": job["host"], "contract_sha256": "contract", "actual_worker_count": 2,
            "actual_worker_exit0_count": 2, "workers": workers}
    if damage == "exit":
        dispatchers["5090"]["workers"][0]["actual_worker_exit_code"] = 1
    elif damage == "sha":
        dispatchers["5090"]["workers"][0]["receipt"]["sha256"] = "wrong"
    else:
        dispatchers["5090"]["workers"].pop()
    for host, receipt in dispatchers.items():
        directory = tmp_path / "runtime" / host
        directory.mkdir(parents=True)
        (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_dispatchers(tmp_path, contract, "contract")
