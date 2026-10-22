"""Synthetic, local-only evidence aggregation tests; no real-data inference."""

import gzip
import json
from pathlib import Path

import pytest

from paper.scripts import compare_shared_history_mlp_control as c


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def records():
    result = []
    for dataset in ("d1", "d2", "d3"):
        for arm_index, arm in enumerate(c.ARMS):
            for seed_index, seed in enumerate(c.SEEDS):
                value = 1.0 + arm_index + seed_index
                result.append(
                    {
                        "job_id": dataset + arm + str(seed),
                        "dataset": dataset,
                        "arm": arm,
                        "host": "5080" if seed == 42 else "5090",
                        "seed": seed,
                        "split": "validation",
                        "prediction_path": "/synthetic/"
                        + dataset
                        + arm
                        + str(seed)
                        + ".gz",
                        "metrics": {
                            scope: {metric: value for metric in c.METRICS}
                            for scope in c.SCOPES
                        },
                    }
                )
    return result


def costs():
    values = []
    for record in records():
        for mode in ("train", "eval"):
            for rep in (1, 2, 3):
                value = {
                    "job_id": record["job_id"],
                    "dataset": record["dataset"],
                    "arm": record["arm"],
                    "host": record["host"],
                    "training_seed": record["seed"],
                    "mode": mode,
                    "timing_rep": rep,
                    "parameters": 1000 if record["arm"] == "encoder_only" else 10600,
                    "median_compute_seconds": rep / 1000,
                    "median_transfer_compute_seconds": rep / 500,
                    "throughput_targets_per_second": 128000.0 / rep,
                    "throughput_including_transfer_targets_per_second": 64000.0 / rep,
                    "total_peak_bytes": {
                        "allocated": rep * 1048576,
                        "reserved": rep * 2097152,
                    },
                }
                values.append(value)
    return values


def test_validation_mean_sample_sd_and_nine_rows():
    rows = c.summarize_records(records())
    assert len(rows) == 9
    assert rows[0]["metrics"]["overall"]["qty_rmse"] == {
        "mean": 2.0,
        "sample_sd_ddof1": 1.0,
        "n": 3,
    }


def test_missing_duplicate_or_heldout_record_is_not_aggregated():
    rows = records()
    with pytest.raises((ValueError, RuntimeError), match="27"):
        c.summarize_records(rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises((ValueError, RuntimeError), match="Duplicate"):
        c.summarize_records(rows)
    rows = records()
    rows[0]["split"] = "held_out"
    with pytest.raises((ValueError, RuntimeError), match="Validation seeds"):
        c.summarize_records(rows)


def test_all_162_paired_metrics_keep_correct_sign_denominator_and_sources():
    rows = records()
    rows[0]["metrics"]["overall"]["qty_rmse"] = 8.0
    paired = c.paired_differences(rows)
    first = paired[0]
    assert len(paired) == 162 and first["absolute"] == 6.0 and first["relative"] == 3.0
    assert first["left_worse"] is True and first["split"] == "validation"
    assert first["left_prediction_path"].endswith("d1shared1542.gz")
    assert first["right_prediction_path"].endswith("d1single_mlp42.gz")


def test_pair_cannot_compare_different_gpus():
    rows = records()
    rows[3]["host"] = "5090"
    with pytest.raises((ValueError, RuntimeError), match="different GPUs"):
        c.paired_differences(rows)


def test_empty_large_scope_remains_unavailable_in_mean_and_difference():
    rows = records()
    for row in rows:
        row["metrics"]["large"] = {metric: None for metric in c.METRICS}
    assert c.summarize_records(rows)[0]["metrics"]["large"]["qty_mae"] == {
        "mean": None,
        "sample_sd_ddof1": None,
        "n": 0,
    }
    assert all(
        row["absolute"] is None
        for row in c.paired_differences(rows)
        if row["scope"] == "large"
    )


def test_costs_are_fresh_54_groups_with_individual_throughput_and_peak_ranges():
    rows = c.summarize_costs(costs())
    assert len(rows) == 54
    assert rows[0]["compute_latency_ms"] == {
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 3.0,
        "n": 3,
    }
    assert rows[0]["allocated_peak_mib"]["maximum"] == 3.0
    assert rows[0]["throughput_targets_per_second"]["median"] == 64000.0
    assert {row["host"] for row in rows if row["seed"] == 42} == {"5080"}
    assert {row["host"] for row in rows if row["seed"] in (52, 62)} == {"5090"}


@pytest.mark.parametrize("change", ["duplicate", "parameter", "infinite"])
def test_cost_repetition_capacity_and_finiteness_are_checked(change):
    values = costs()
    if change == "duplicate":
        values[-1] = values[0]
    elif change == "parameter":
        values[0]["parameters"] += 1
    else:
        values[0]["median_compute_seconds"] = float("inf")
    with pytest.raises((ValueError, RuntimeError)):
        c.summarize_costs(values)


def test_prediction_metrics_strict_train_cutoff_and_recorded_nll(tmp_path):
    path = tmp_path / "validation.jsonl.gz"
    rows = [(10.0, 8.0, 0.25), (11.0, 14.0, 0.75)]
    with gzip.open(path, "wt") as stream:
        for index, (truth, prediction, nll) in enumerate(rows):
            stream.write(
                json.dumps(
                    {
                        "split": "validation",
                        "target_index": index,
                        "entity_id": str(index),
                        "target_seq": 1,
                        "raw_quantity": truth,
                        "predicted_raw_quantity": prediction,
                        "time_nll": nll,
                    }
                )
                + "\n"
            )
    metrics = c.prediction_metrics(path, 10.0, "validation")
    assert metrics["large"]["count"] == 1 and metrics["large"]["qty_rmse"] == 3.0
    assert (
        metrics["overall"]["time_nll"] == 0.5 and metrics["large"]["time_nll"] == 0.75
    )


def cost_fixture(tmp_path):
    contract = {
        "hosts": {"5080": {"gpu_uuid": "fixed-gpu"}},
        "datasets": [
            {
                "dataset_id": "d",
                "inherited_data_identity": {"data": {"sha256": "input-sha"}},
            }
        ],
    }
    job = {"id": "job", "dataset": "d", "arm": "single_mlp", "seed": 42}
    train = {
        "checkpoint_sha256": "binary",
        "state_sha256": "tensor",
        "selected_epoch": 42,
    }
    batch = tmp_path / "cost/batches/d.json"
    batch.parent.mkdir(parents=True)
    batch.with_suffix(".pt").write_bytes(b"synthetic batch binary")
    cache_rows = [
        {
            "batch": index,
            "phase": "warmup" if index < 5 else "measured",
            "target_split": "train",
            "targets": [["e", index * 128 + target] for target in range(128)],
            "tensor_sha256": str(index).zfill(64),
        }
        for index in range(37)
    ]
    targets = [target for row in cache_rows for target in row["targets"]]
    manifest = {
        "dataset": "d",
        "split": "train",
        "synthetic": False,
        "contract_sha256": c.canonical(contract),
        "preparation_seed": 42,
        "input_sha256": "input-sha",
        "file_sha256": c.sha(batch.with_suffix(".pt")),
        "target_ids_sha256": c.canonical(targets),
        "batches": cache_rows,
    }
    dump(batch, manifest)
    timing_rows = [
        {
            "batch": index,
            "phase": "warmup" if index < 5 else "measured",
            "targets": 128,
            "compute_seconds": 0.01,
            "transfer_seconds": 0.002,
            "transfer_compute_seconds": 0.012,
        }
        for index in range(37)
    ]
    for mode in ("train", "eval"):
        for rep in (1, 2, 3):
            value = {
                "status": "complete",
                "host": "5080",
                "mode": mode,
                "timing_rep": rep,
                "contract_sha256": c.canonical(contract),
                "batch_manifest_sha256": c.sha(batch),
                "target_ids_sha256": manifest["target_ids_sha256"],
                "batch_tensor_sha256": [r["tensor_sha256"] for r in cache_rows],
                "checkpoint_binary_unchanged": True,
                "checkpoint_saved": False,
                "input_split": "train",
                "synthetic": False,
                "runtime": {"gpu": {"uuid": "fixed-gpu"}},
                "selected": train,
                "job_id": "job",
                "dataset": "d",
                "arm": "single_mlp",
                "training_seed": 42,
                "median_compute_seconds": 0.01,
                "median_transfer_compute_seconds": 0.012,
                "throughput_targets_per_second": 4096
                / sum(r["compute_seconds"] for r in timing_rows[5:]),
                "throughput_including_transfer_targets_per_second": 4096
                / sum(r["transfer_compute_seconds"] for r in timing_rows[5:]),
                "rows": timing_rows,
                "optimizer_steps": 37 if mode == "train" else 0,
                "parameters": 10600,
                "memory_after_warmup_baseline_bytes": {"allocated": 10, "reserved": 20},
                "warmup_peak_bytes": {"allocated": 30, "reserved": 50},
                "measured_peak_bytes": {"allocated": 40, "reserved": 60},
                "total_peak_bytes": {"allocated": 40, "reserved": 60},
                "measured_incremental_peak_bytes": {"allocated": 30, "reserved": 40},
            }
            dump(
                tmp_path / "cost/results/job" / (mode + "_rep" + str(rep) + ".json"),
                value,
            )
    return contract, job, train


def test_cost_raw_timings_and_memory_are_recomputed(tmp_path):
    contract, job, train = cost_fixture(tmp_path)
    assert len(c._verify_cost(tmp_path, contract, "5080", job, train)) == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "median",
        "throughput",
        "warmup_peak",
        "incremental_peak",
        "gpu",
        "optimizer",
        "batch_tensor",
    ],
)
def test_cost_summary_cannot_override_raw_or_selected_evidence(tmp_path, mutation):
    contract, job, train = cost_fixture(tmp_path)
    path = tmp_path / "cost/results/job/train_rep1.json"
    value = c.read(path)
    if mutation == "median":
        value["median_compute_seconds"] = 0.02
    elif mutation == "throughput":
        value["throughput_targets_per_second"] += 1
    elif mutation == "warmup_peak":
        value["warmup_peak_bytes"]["allocated"] = 100
    elif mutation == "incremental_peak":
        value["measured_incremental_peak_bytes"]["allocated"] = 31
    elif mutation == "gpu":
        value["runtime"]["gpu"]["uuid"] = "other-gpu"
    elif mutation == "optimizer":
        value["optimizer_steps"] = 0
    else:
        value["batch_tensor_sha256"][0] = "f" * 64
    dump(path, value)
    with pytest.raises((ValueError, RuntimeError)):
        c._verify_cost(tmp_path, contract, "5080", job, train)


def result_fixture():
    source = records()
    paired = c.paired_differences(source)
    return {
        "mean_definition": "Three seeds, ddof=1",
        "architecture_caveat": "Input and total capacity matched only",
        "cost_definition": "Fresh Train batches, GPU groups kept separate",
        "table": c.summarize_records(source),
        "paired_differences": paired,
        "adverse_differences": [r for r in paired if r["left_worse"]],
        "cost_summaries": c.summarize_costs(costs()),
        "fit_records": [],
    }


def test_rendering_is_pure_validation_only_and_csv_has_54_metric_rows():
    result = result_fixture()
    text = c.render_markdown(result)
    assert (
        "Validation only" in text
        and "held-out Test metric was read or evaluated" in text
    )
    assert len(c.render_csv(result).splitlines()) == 55
    assert c.render_markdown(result) == text


def test_outputs_are_exclusive_complete_and_pure_recomputed(tmp_path):
    result = result_fixture()
    with pytest.raises((ValueError, RuntimeError), match="inventory"):
        c.verify_outputs(tmp_path, result)
    dump(tmp_path / "REPORT.json", result)
    (tmp_path / "REPORT.md").write_text(c.render_markdown(result))
    (tmp_path / "TABLE.csv").write_text(c.render_csv(result))
    c.verify_outputs(tmp_path, result)
    (tmp_path / "TABLE.csv").write_text("spoofed")
    with pytest.raises((ValueError, RuntimeError), match="pure recomputation"):
        c.verify_outputs(tmp_path, result)


def test_extra_or_symlink_report_is_rejected(tmp_path):
    result = result_fixture()
    dump(tmp_path / "REPORT.json", result)
    (tmp_path / "REPORT.md").write_text(c.render_markdown(result))
    (tmp_path / "TABLE.csv").write_text(c.render_csv(result))
    (tmp_path / "unexpected").write_text("extra")
    with pytest.raises((ValueError, RuntimeError), match="inventory"):
        c.verify_outputs(tmp_path, result)


def test_comparator_cli_rejects_original_campaign_or_arbitrary_output(tmp_path):
    with pytest.raises((ValueError, RuntimeError), match="fixed local paths"):
        c.main(
            [
                "--contract",
                str(tmp_path / "old-contract.json"),
                "--contract-sha",
                "0" * 64,
                "--retrieval",
                str(tmp_path / "old-results"),
                "--output",
                str(tmp_path / "spoofed"),
                "--check-only",
            ]
        )
