"""Synthetic artifact audits; no model fitting or dataset files are opened."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from paper.scripts import compare_titans_mac_prior_prefix as comparison
from paper.scripts.titans_mac_prior_prefix_contract import load_contract


REVISION = "a" * 40
SEEDS = (42, 52, 62)
BACKBONE, BASELINE = comparison.BACKBONE, comparison.BASELINE


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen():
    contract = load_contract()
    reference_path = comparison.ROOT / contract["frozen_references"]["path"]
    return contract, json.loads(reference_path.read_text())["references"]


def row_paths(root, dataset, backbone, seed):
    outer = root / dataset / backbone / f"seed_{seed}"
    leaf = outer / "runs" / backbone / comparison.VARIANT / f"seed_{seed}"
    return leaf / "summary.json", leaf / "history.json", outer / "launch_contract.json"


def make_run(root, dataset, backbone, seed, contract, references, *, improve=True):
    metrics = copy.deepcopy(references[dataset]["titantpp"][str(seed)])
    if backbone == BACKBONE:
        for name in ("body_mae", "qty_mae", "qty_rmse", "tail_mae", "log_qty_mse"):
            metrics[name] *= .9 if improve else 1.
    data = contract["datasets"][dataset]
    n, body, tail = (metrics[k] for k in ("validation_targets", "body_targets", "tail_targets"))
    counts = [body // 2, body // 3, body - body // 2 - body // 3, n - body - tail, tail]
    labels = ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")
    middle_mae = (n * metrics["qty_mae"] - body * metrics["body_mae"] - tail * metrics["tail_mae"]) / counts[3]
    assert middle_mae >= 0
    maes = [metrics["body_mae"]] * 3 + [middle_mae, metrics["tail_mae"]]
    objective = metrics["time_loss"] + metrics["log_qty_mse"]
    summary = {
        "status": "success", "backbone": backbone, "variant": comparison.VARIANT,
        "seed": seed, "source_revision": REVISION, "source_revision_history": [REVISION],
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "completed_epochs": 41, "best_epoch": 1, "best_val_joint_objective": objective,
        "best_val_qty_mae": metrics["qty_mae"], "best_val_qty_rmse": metrics["qty_rmse"],
        "best_val_time_nll": metrics["time_loss"], "best_val_log_qty_mse": metrics["log_qty_mse"],
        "quantity_rows": [{"stratum": label, "count": count, "qty_mae": mae}
                          for label, count, mae in zip(labels, counts, maes, strict=True)],
        "history_rows": [{"count": n}], "checkpoint_state_sha256": "b" * 64,
        "encoder_config": {
            "mac_execution_backend": "optimized", "titans_memory_gradient_clip": 1.,
            "memory_mode": "titans_mac_prior_prefix" if backbone == BACKBONE else "titans_mac",
            **({"titans_output_read_policy": "prior_prefix",
                "titans_prefix_read_contract_id": contract["contract_id"]} if backbone == BACKBONE else {}),
        },
    }
    history = [{
        "epoch": epoch, "train_event_count": data["train_targets"],
        "train_batch_count": math.ceil(data["train_targets"] / 128), "train_all_finite": True,
        "val_joint_objective": objective if epoch == 1 else objective + epoch * .01,
        "val_qty_mae": metrics["qty_mae"], "val_qty_rmse": metrics["qty_rmse"],
        "val_time_nll": metrics["time_loss"], "val_log_qty_mse": metrics["log_qty_mse"],
    } for epoch in range(1, 42)]
    launch = {
        "status": "complete", "completed_run_count": 1, "expected_run_count": 1,
        "dataset": dataset, "backbones": [backbone], "seeds": [seed], "epochs": 300,
        "quantity_variants": [comparison.VARIANT], "max_series": None,
        "model_role": comparison.ROLE if backbone == BACKBONE else "experimental",
        "source_revision": REVISION, "partial_smoke": False,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "batch_size": 128, "lr": .001, "hidden_dim": 64, "grad_clip": 1.,
        "titans_memory_gradient_clip": 1., "lambda_log_qty": 1., "lambda_tail": 0.,
        "lookback_weeks": data["lookback"], "max_seq_len": data["max_seq_len"],
        "data_sha256": data["data_sha256"], "split_manifest_sha256": data["split_manifest_sha256"],
        "split_rows": {"train": 1, "validation": 1},
        "early_stopping": {"min_epochs": 40, "patience": 40},
        "time_head": {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10 / 3,
                      "time_intercept_limit": 300., "time_head_lr_multiplier": 1.,
                      "statistics_source_split": "train"},
        "quantity_contract": {"quantiles": [.5, .9, .95, .99],
                              "boundaries": [1., 2., data["train_p95_quantity"], data["train_p99_quantity"]]},
    }
    paths = row_paths(root, dataset, backbone, seed)
    for path, value in zip(paths, (summary, {"history": history}, launch), strict=True):
        write(path, value)
    return paths


def make_grid(root, frozen, seeds=(42,), *, improve=True):
    contract, references = frozen
    for dataset in contract["dataset_order"]:
        for seed in seeds:
            for backbone in (BASELINE, BACKBONE):
                make_run(root, dataset, backbone, seed, contract, references, improve=improve)
    return root


def test_real_frozen_reference_schema_supports_full_screening_and_confirmation(tmp_path, frozen):
    screening = make_grid(tmp_path / "screening", frozen)
    confirmed = make_grid(tmp_path / "confirm", frozen, seeds=(52, 62))
    result = comparison.compare(comparison.CONTRACT_PATH, screening, "screening")
    assert result["status"] == "PASS" and result["screening_pass"] is True
    assert set(result["datasets"]) == set(frozen[0]["datasets"])
    assert result["held_out_test_evaluated"] is False
    final = comparison.compare(comparison.CONTRACT_PATH, confirmed, "confirm", screening)
    assert final["status"] == "PASS" and final["confirmation_pass"] is True
    for dataset in final["datasets"].values():
        assert set(dataset["paired_seeds"]) == {"42", "52", "62"}
        assert dataset["strict_rmse_seed_wins"] == {"B1": 3, "T0": 3}
    # Each run contributes summary, history and launch evidence, plus frozen refs.
    assert len(final["evidence_files"]) == 18 * 3 + 1


def test_failed_performance_is_a_successful_comparison_and_zero_exit(tmp_path, frozen):
    runs = make_grid(tmp_path / "runs", frozen, improve=False)
    output = tmp_path / "comparison.json"
    process = subprocess.run(
        [sys.executable, "-s", "-B", str(Path(comparison.__file__)),
         "--run-root", str(runs), "--phase", "screening", "--output", str(output)],
        text=True, capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "PASS"
    assert result["screening_pass"] is False
    assert "confirmation_pass" not in result


def mutate(path, callback):
    value = json.loads(path.read_text())
    callback(value)
    write(path, value)


@pytest.mark.parametrize("kind", [
    "missing_summary", "held_out", "held_out_split", "mixed_source_history",
    "partial_train", "premature_stop", "reselected_epoch", "mixed_metric",
    "candidate_as_baseline", "wrong_head_cap", "wrong_quantity_threshold",
])
def test_load_run_rejects_incomplete_or_mixed_artifacts(tmp_path, frozen, kind):
    contract, refs = frozen
    dataset = "insta_market_basket"
    summary, history, launch = make_run(tmp_path, dataset, BACKBONE, 42, contract, refs)
    if kind == "missing_summary":
        summary.unlink()
    elif kind == "held_out":
        mutate(summary, lambda value: value.update(held_out_test_evaluated=True))
    elif kind == "held_out_split":
        mutate(launch, lambda value: value["split_rows"].update(test=1))
    elif kind == "mixed_source_history":
        mutate(summary, lambda value: value.update(source_revision_history=[REVISION, "c" * 40]))
    elif kind == "partial_train":
        mutate(history, lambda value: value["history"][20].update(train_event_count=1))
    elif kind == "premature_stop":
        mutate(history, lambda value: value["history"].pop())
        mutate(summary, lambda value: value.update(completed_epochs=40))
    elif kind == "reselected_epoch":
        mutate(summary, lambda value: value.update(best_epoch=2))
    elif kind == "mixed_metric":
        mutate(summary, lambda value: value.update(best_val_qty_rmse=0.1))
    elif kind == "candidate_as_baseline":
        mutate(summary, lambda value: value["encoder_config"].update(memory_mode="titans_mac"))
    elif kind == "wrong_head_cap":
        mutate(launch, lambda value: value["time_head"].update(time_intercept_limit=30.))
    elif kind == "wrong_quantity_threshold":
        mutate(launch, lambda value: value["quantity_contract"]["boundaries"].__setitem__(2, 24.))
    with pytest.raises((ValueError, FileNotFoundError)):
        comparison.load_run(tmp_path, dataset, BACKBONE, 42, contract)


def test_one_changed_source_revision_blocks_cross_dataset_comparison(tmp_path, frozen):
    runs = make_grid(tmp_path, frozen)
    summary, _, launch = row_paths(runs, "insta_market_basket", BACKBONE, 42)
    replacement = "c" * 40
    mutate(summary, lambda value: value.update(source_revision=replacement, source_revision_history=[replacement]))
    mutate(launch, lambda value: value.update(source_revision=replacement))
    with pytest.raises(ValueError, match="one source revision"):
        comparison.compare(comparison.CONTRACT_PATH, runs, "screening")


def test_same_total_count_cannot_hide_changed_body_population(tmp_path, frozen):
    runs = make_grid(tmp_path, frozen)
    summary, _, _ = row_paths(runs, "insta_market_basket", BACKBONE, 42)
    def change(value):
        value["quantity_rows"][0]["count"] -= 1
        value["quantity_rows"][3]["count"] += 1
    mutate(summary, change)
    with pytest.raises(ValueError, match="different targets"):
        comparison.compare(comparison.CONTRACT_PATH, runs, "screening")


def test_duplicate_dataset_order_is_not_a_three_dataset_comparison(tmp_path, frozen):
    contract = copy.deepcopy(frozen[0])
    contract["dataset_order"][1] = contract["dataset_order"][0]
    path = tmp_path / "contract.json"
    write(path, contract)
    with pytest.raises(ValueError, match="each frozen dataset exactly once"):
        comparison.compare(path, tmp_path / "runs", "screening")


def paired_fixture(candidate_rmses=(9., 9., 9.)):
    dataset = "insta_market_basket"
    ref = dict(body_mae=10., qty_mae=10., qty_rmse=10., tail_mae=10., time_loss=1.,
               log_qty_mse=.2, validation_targets=100, body_targets=95, tail_targets=1)
    refs = {dataset: {model: {str(seed): copy.deepcopy(ref) for seed in SEEDS}
                     for model in ("titantpp", "rmtpp", "thp")}}
    rows = {}
    for seed, rmse in zip(SEEDS, candidate_rmses, strict=True):
        rows[(dataset, BASELINE, seed)] = {"metrics": copy.deepcopy(ref)}
        rows[(dataset, BACKBONE, seed)] = {"metrics": {
            **ref, "body_mae": 9., "qty_mae": 9., "qty_rmse": rmse, "tail_mae": 9.,
        }}
    return dataset, rows, refs


def test_three_seed_means_are_arithmetic_and_need_two_strict_wins():
    dataset, rows, refs = paired_fixture((1., 10.1, 10.1))
    result = comparison.assess_dataset(rows, refs, load_contract(), dataset, SEEDS)
    assert result["mean_candidate"]["qty_rmse"] == pytest.approx(21.2 / 3)
    assert result["mean_decision"]["screening_pass"] is True
    assert result["strict_rmse_seed_wins"] == {"B1": 1, "T0": 1}
    assert result["confirmation_pass"] is False
    dataset, rows, refs = paired_fixture((9., 9., 10.5))
    result = comparison.assess_dataset(rows, refs, load_contract(), dataset, SEEDS)
    assert result["strict_rmse_seed_wins"] == {"B1": 2, "T0": 2}
    assert result["confirmation_pass"] is True


def test_seed42_screening_cannot_be_retroactively_approved_by_other_seeds():
    dataset, rows, refs = paired_fixture()
    rows[(dataset, BACKBONE, 42)]["metrics"]["body_mae"] = 10.
    for seed in (52, 62):
        rows[(dataset, BACKBONE, seed)]["metrics"]["body_mae"] = 8.
    result = comparison.assess_dataset(rows, refs, load_contract(), dataset, SEEDS)
    assert result["mean_decision"]["screening_pass"] is True
    assert result["screening_pass"] is False
    assert result["confirmation_pass"] is False


@pytest.mark.parametrize("metric,value", [("time_loss", 1.02), ("tail_mae", 10.3)])
def test_mean_success_cannot_hide_one_seed_time_or_tail_violation(metric, value):
    dataset, rows, refs = paired_fixture()
    rows[(dataset, BACKBONE, 62)]["metrics"][metric] = value
    result = comparison.assess_dataset(rows, refs, load_contract(), dataset, SEEDS)
    assert result["mean_decision"]["screening_pass"] is True
    assert result["individual_seed_time_tail_guardrails_pass"] is False
    assert result["confirmation_pass"] is False


def test_missing_paired_seed_is_an_error_and_never_silently_dropped():
    dataset, rows, refs = paired_fixture()
    rows.pop((dataset, BASELINE, 52))
    with pytest.raises(KeyError):
        comparison.assess_dataset(rows, refs, load_contract(), dataset, SEEDS)


def test_seed_means_do_not_weight_by_target_count():
    first = {key: 1. for key in comparison.METRICS}
    second = {key: 9. for key in comparison.METRICS}
    first["validation_targets"], second["validation_targets"] = 1, 100000
    assert comparison.means([first, second]) == {key: 5. for key in comparison.METRICS}
