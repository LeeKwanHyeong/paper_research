"""Local-only, Validation-only aggregation of the matched history-MLP control.

The selected checkpoints are loaded on CPU solely for identity and parameter
verification. No forward pass, data loader, training, or held-out evaluation is
performed here. Original B/D checkpoint identities are preserved.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

from paper.scripts import manage_shared_history_mlp_control as manager
from paper.scripts.compare_shared_component_ablation import (
    prediction_metrics,
    mapped,
    relative_file,
    _verify_cost,
    _mean_sd,
    _diff,
    _formatted,
)

SCHEMA = "shared_history_mlp_control_comparison_v1"
ARMS = ("shared15", "single_mlp", "encoder_only")
HOSTS, SEEDS = ("5080", "5090"), (42, 52, 62)
SCOPES, METRICS = ("overall", "large"), ("qty_rmse", "qty_mae", "time_nll")
PAIRS = (
    ("shared15", "single_mlp"),
    ("shared15", "encoder_only"),
    ("single_mlp", "encoder_only"),
)
OUTPUTS = {"REPORT.json", "REPORT.md", "TABLE.csv"}
read, sha, require, canonical = (
    manager.read,
    manager.sha,
    manager.require,
    manager.canonical,
)


def _receipt(root, contract, host, job, stage):
    path = root / "run" / job["id"] / (stage + "_receipt.json")
    value = read(path)
    require(
        value.get("schema") == "shared_history_mlp_control_stage_v1"
        and value.get("status") == "complete"
        and value.get("stage") == stage
        and value.get("host") == host
        and value.get("job_id") == job["id"]
        and value.get("arm") == job["arm"]
        and value.get("contract_sha256") == canonical(contract)
        and value.get("source_files_sha256") == contract["source"]["files_sha256"]
        and value.get("held_out_test_evaluated") is False,
        "Validation-only stage identity differs",
    )
    return value


def _exit(root, contract, job, stage):
    path = root / "control" / (job["id"] + "." + stage + ".exit.json")
    value = read(path)
    require(
        value.get("actual_os_returncode") == 0
        and value.get("timeout") is False
        and value.get("supervisor_error") is None
        and value.get("resource_incident") is None
        and value.get("host") == job["host"]
        and value.get("job_id") == job["id"]
        and value.get("stage") == stage
        and value.get("contract_sha256") == canonical(contract),
        "Actual successful scientific child exit required",
    )
    for stream in ("stdout", "stderr"):
        path = root / "control" / (job["id"] + "." + stage + "." + stream + ".log")
        require(
            path.is_file()
            and not path.is_symlink()
            and sha(path) == value.get(stream + "_sha256"),
            "Original scientific child output changed",
        )
        if stream + "_bytes" in value:
            require(
                path.stat().st_size == value[stream + "_bytes"],
                "Scientific child output byte count differs",
            )
    return value


def verify_collections(contract, retrieval):
    """Read only the two authorized phases; validate every archive member."""
    from paper.scripts import advance_shared_history_mlp_control as controller

    roots = {}
    retrieval = Path(retrieval)
    for host in HOSTS:
        first = controller.validate_collection(
            contract, host, "fit_validation", retrieval / "fit_validation" / host
        )
        root = controller.validate_collection(
            contract, host, "cost", retrieval / "cost" / host
        )
        require(
            read(root / "execution_contract.json") == contract,
            "Retrieved execution contract differs",
        )
        manager.verify_native_qualification(contract, host, root)
        manager.verify_phase_receipt(contract, host, "fit_validation", root)
        manager.verify_phase_receipt(contract, host, "cost", root)
        initial_manifest = read(first / "collection_fit_validation.manifest.json")
        for name, item in initial_manifest["files"].items():
            path = relative_file(root, name)
            require(
                path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                "Fit/Validation evidence changed during the cost phase",
            )
        gate = manager.verify_global_gate(contract, root)
        require(
            len(gate.get("jobs", {})) == 27,
            "All 27 Validation jobs are required at the global gate",
        )
        for job in contract["jobs"]:
            if job["host"] != host:
                continue
            for stage in ("train", "validation"):
                for kind in ("receipt", "exit"):
                    if kind == "receipt":
                        name = "run/" + job["id"] + "/" + stage + "_receipt.json"
                    else:
                        exit_stage = (
                            "reuse" if stage == "train" and job.get("reuse") else stage
                        )
                        name = "control/" + job["id"] + "." + exit_stage + ".exit.json"
                    require(
                        sha(relative_file(root, name))
                        == gate["jobs"][job["id"]][stage + "_" + kind]["sha256"],
                        "Global Validation proof differs from original job evidence",
                    )
        roots[host] = root
    return roots


def verify_legacy(contract, job, train, validation, root, prediction):
    """Link the wrapper's B/D labels to untouched parent receipts and binaries."""
    from paper.scripts import run_shared_history_mlp_control as runner

    reuse = job["reuse"]
    require(
        train.get("training_origin") == "reused_verified_fit"
        and train.get("reuse") == reuse
        and train.get("new_training_epochs") == 0
        and train.get("parent_individual_validation_reproduced") is True,
        "Reused fit was retrained or lacks original individual replay",
    )
    originals = {}
    for key, item in reuse.items():
        if isinstance(item, dict) and "path" in item:
            path = relative_file(root / "source", item["path"])
            require(
                sha(path) == item["sha256"], "Parent evidence bytes changed: " + key
            )
            originals[key] = path
    parent_item = contract["parent"]["legacy_contract"]
    parent_path = relative_file(root / "source", parent_item["path"])
    require(
        sha(parent_path)
        == parent_item["sha256"]
        == contract["parent"]["contract_file_sha256"],
        "Original parent contract bytes differ",
    )
    parent = read(parent_path)
    require(
        canonical(parent)
        == contract["parent"]["canonical_contract_sha256"]
        == reuse["parent_contract_sha256"]
        and parent["source"]["files_sha256"]
        == contract["parent"]["source_files_sha256"]
        == reuse["parent_source_files_sha256"],
        "Original parent contract/source identity differs",
    )
    parent_jobs = [
        item for item in parent["jobs"] if item["id"] == reuse["parent_job_id"]
    ]
    require(
        len(parent_jobs) == 1
        and all(parent_jobs[0][key] == job[key] for key in ("dataset", "host", "seed"))
        and parent_jobs[0]["arm"] == reuse["underlying_arm"],
        "Parent arm/job mapping differs",
    )
    for stage in ("train", "validation"):
        receipt, end = (
            read(originals[stage + "_receipt"]),
            read(originals[stage + "_exit"]),
        )
        require(
            receipt.get("status") == "complete"
            and receipt.get("stage") == stage
            and receipt.get("job_id") == reuse["parent_job_id"]
            and receipt.get("host") == job["host"]
            and receipt.get("contract_sha256") == canonical(parent),
            "Parent stage receipt identity differs",
        )
        require(
            end.get("actual_os_returncode") == 0
            and end.get("timeout") is False
            and end.get("supervisor_error") is None
            and end.get("resource_incident") is None
            and end.get("host") == job["host"]
            and end.get("job_id") == reuse["parent_job_id"]
            and end.get("stage") == stage
            and end.get("contract_sha256") == canonical(parent),
            "Original reused fit/Validation did not actually exit successfully",
        )
        for stream in ("stdout", "stderr"):
            path = originals[stage + "_" + stream]
            require(
                sha(path) == end.get(stream + "_sha256"),
                "Original reused child output differs",
            )
            if stream + "_bytes" in end:
                require(
                    path.stat().st_size == end[stream + "_bytes"],
                    "Original reused output bytes differ",
                )
        for key in ("selected_epoch", "checkpoint_sha256", "state_sha256"):
            require(
                receipt.get(key) == train.get(key) == validation.get(key),
                "Parent selection changed",
            )
    require(
        train["checkpoint_sha256"] == reuse["checkpoint"]["sha256"]
        and train["state_sha256"] == reuse["canonical_tensor_sha256"]
        and train["selected_epoch"] == reuse["selected_epoch"],
        "Original selected binary/tensor/epoch differs",
    )
    history = relative_file(root, train["reused_history_copy"])
    require(
        sha(history) == reuse["history"]["sha256"],
        "Copied parent selection history changed",
    )
    runner.compare_prediction_files(prediction, originals["validation_predictions"])


def summarize_records(records):
    """Nine Validation means; ddof=1 across the three common training seeds."""
    require(len(records) == 27, "Exactly 27 Validation records required")
    datasets = sorted({row["dataset"] for row in records})
    require(
        len(datasets) == 3
        and len({(r["dataset"], r["arm"], r["seed"]) for r in records}) == 27,
        "Duplicate or missing Validation condition",
    )
    rows = []
    for dataset in datasets:
        for arm in ARMS:
            group = [r for r in records if r["dataset"] == dataset and r["arm"] == arm]
            require(
                len(group) == 3
                and {r["seed"] for r in group} == set(SEEDS)
                and all(r["split"] == "validation" for r in group),
                "Common Validation seeds missing",
            )
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "split": "validation",
                    "seeds": list(SEEDS),
                    "metrics": {
                        scope: {
                            metric: _mean_sd(
                                [r["metrics"][scope][metric] for r in group]
                            )
                            for metric in METRICS
                        }
                        for scope in SCOPES
                    },
                }
            )
    return rows


def paired_differences(records):
    indexed = {(r["dataset"], r["arm"], r["seed"]): r for r in records}
    rows = []
    for dataset in sorted({r["dataset"] for r in records}):
        for left, right in PAIRS:
            for seed in SEEDS:
                a, b = indexed[(dataset, left, seed)], indexed[(dataset, right, seed)]
                require(
                    a["host"] == b["host"], "Paired conditions ran on different GPUs"
                )
                for scope in SCOPES:
                    for metric in METRICS:
                        av, bv = (
                            a["metrics"][scope][metric],
                            b["metrics"][scope][metric],
                        )
                        diff = _diff(av, bv)
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "host": a["host"],
                                "split": "validation",
                                "left_arm": left,
                                "right_arm": right,
                                "scope": scope,
                                "metric": metric,
                                "left": av,
                                "right": bv,
                                **diff,
                                "left_worse": diff["absolute"] is not None
                                and diff["absolute"] > 0,
                                "left_prediction_path": a["prediction_path"],
                                "right_prediction_path": b["prediction_path"],
                            }
                        )
    require(len(rows) == 162, "All 162 common-seed metric differences required")
    return rows


def _range(values):
    require(
        len(values) == 3
        and all(
            type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in values
        ),
        "Three finite repetition values required",
    )
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "n": 3,
    }


def summarize_costs(costs):
    require(
        len(costs) == 162
        and len({(r["job_id"], r["mode"], r["timing_rep"]) for r in costs}) == 162,
        "Exactly 162 distinct fresh cost receipts required",
    )
    groups = sorted(
        {
            (r["dataset"], r["arm"], r["host"], r["training_seed"], r["mode"])
            for r in costs
        }
    )
    require(len(groups) == 54, "Exactly 54 GPU/seed/mode groups required")
    rows = []
    for dataset, arm, host, seed, mode in groups:
        group = [
            r
            for r in costs
            if (r["dataset"], r["arm"], r["host"], r["training_seed"], r["mode"])
            == (dataset, arm, host, seed, mode)
        ]
        require(
            len(group) == 3
            and {r["timing_rep"] for r in group} == {1, 2, 3}
            and len({r["parameters"] for r in group}) == 1,
            "Fresh repetition or parameter identity differs",
        )
        row = {
            "dataset": dataset,
            "arm": arm,
            "host": host,
            "seed": seed,
            "mode": mode,
            "parameters": group[0]["parameters"],
            "compute_latency_ms": _range(
                [r["median_compute_seconds"] * 1000 for r in group]
            ),
            "transfer_compute_latency_ms": _range(
                [r["median_transfer_compute_seconds"] * 1000 for r in group]
            ),
            "throughput_targets_per_second": _range(
                [r["throughput_targets_per_second"] for r in group]
            ),
            "throughput_including_transfer_targets_per_second": _range(
                [r["throughput_including_transfer_targets_per_second"] for r in group]
            ),
            "allocated_peak_mib": _range(
                [r["total_peak_bytes"]["allocated"] / 1048576 for r in group]
            ),
            "reserved_peak_mib": _range(
                [r["total_peak_bytes"]["reserved"] / 1048576 for r in group]
            ),
        }
        rows.append(row)
    for dataset in sorted({r["dataset"] for r in rows}):
        for seed in SEEDS:
            parameters = {
                arm: {
                    r["parameters"]
                    for r in rows
                    if r["dataset"] == dataset and r["seed"] == seed and r["arm"] == arm
                }
                for arm in ARMS
            }
            require(
                all(len(values) == 1 for values in parameters.values())
                and next(iter(parameters["shared15"]))
                == next(iter(parameters["single_mlp"]))
                == next(iter(parameters["encoder_only"])) + 9600,
                "Matched correction parameter budget differs",
            )
    return rows


def _cost_metadata(c, job, train, values, parameter_count, root):
    for value in values:
        require(
            value.get("schema") == "shared_history_mlp_control_cost_v1"
            and value.get("source_files_sha256") == c["source"]["files_sha256"]
            and value.get("scientific_fit") is False
            and value.get("accuracy_evaluated") is False
            and value.get("preparation_seed") == 42
            and value.get("parameters") == parameter_count,
            "Fresh Train-only cost identity differs",
        )
        require(
            value.get("contract_file_sha256") == sha(root / "execution_contract.json")
            and all(
                value.get("runtime", {}).get(key) == expected
                for key, expected in c["hosts"][job["host"]]["runtime_expected"].items()
            ),
            "Raw cost contract or native runtime differs",
        )
        selected = value.get("selected", {})
        require(
            selected.get("training_origin") == train["training_origin"]
            and selected.get("provenance") == job.get("reuse"),
            "Cost selected provenance differs",
        )


def aggregate(contract, retrieval):
    """Recompute from original Validation predictions and fresh Train cost rows."""
    from paper.scripts import run_shared_history_mlp_control as runner

    manager.validate(contract)
    roots = verify_collections(contract, retrieval)
    records, costs, fits, batch_identity = [], [], [], {}
    for job in contract["jobs"]:
        host, root = job["host"], roots[job["host"]]
        data = next(
            d for d in contract["datasets"] if d["dataset_id"] == job["dataset"]
        )
        train = _receipt(root, contract, host, job, "train")
        validation = _receipt(root, contract, host, job, "validation")
        _exit(root, contract, job, "reuse" if job.get("reuse") else "train")
        _exit(root, contract, job, "validation")
        require(
            all(
                validation.get(key) == train.get(key)
                for key in (
                    "selected_checkpoint",
                    "selected_epoch",
                    "checkpoint_sha256",
                    "state_sha256",
                    "training_origin",
                )
            ),
            "Validation selection differs from training",
        )
        require(
            validation.get("validation_replay_passed") is True
            and validation.get("individual_predictions_match") is True,
            "Individual Validation replay is missing",
        )
        model, selected = runner._load_model(contract, data, job, train, root, "cpu")
        parameters = sum(p.numel() for p in model.parameters())
        del model, selected
        prediction = mapped(
            root, contract["hosts"][host]["root"], validation["prediction_path"]
        )
        metrics = prediction_metrics(
            prediction, data["quantity_boundaries_all_train_rows"][-1], "validation"
        )
        require(
            metrics["prediction_rows"]
            == validation["prediction_rows"]
            == validation["metrics"]["overall"]["count"],
            "Validation prediction row count differs",
        )
        identity = validation.get("target_identity", {})
        require(
            identity == data["inherited_data_identity"]["populations"]["validation"]
            and identity.get("target_count") == metrics["prediction_rows"]
            and identity.get("target_quantity_sha256")
            == metrics["target_quantity_sha256"],
            "Validation frozen target population differs",
        )
        for scope, recorded_scope in (("overall", "overall"), ("large", "tail")):
            require(
                metrics[scope]["count"]
                == validation["metrics"][recorded_scope]["count"],
                "Metric scope count differs",
            )
            for metric in METRICS:
                a, b = (
                    metrics[scope][metric],
                    validation["metrics"][recorded_scope][metric],
                )
                require(
                    (a is None and b is None)
                    or (
                        a is not None
                        and b is not None
                        and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                    ),
                    "Recorded Validation metric differs from predictions",
                )
        if job.get("reuse"):
            verify_legacy(contract, job, train, validation, root, prediction)
            elapsed, epochs = None, None
        else:
            require(
                train.get("training_origin") == "fresh_fit"
                and job["arm"] == "single_mlp",
                "Unexpected fresh fit",
            )
            summary = train.get("summary", {})
            elapsed, epochs = (
                summary.get("elapsed_seconds"),
                summary.get("completed_epochs"),
            )
            require(
                type(elapsed) in (float, int)
                and math.isfinite(elapsed)
                and elapsed > 0
                and type(epochs) is int
                and 0 < epochs <= contract["training"]["maximum_epochs"]
                and train.get("new_training_epochs") == epochs,
                "Fresh fit elapsed/epoch evidence differs",
            )
        fits.append(
            {
                "job_id": job["id"],
                "dataset": job["dataset"],
                "arm": job["arm"],
                "seed": job["seed"],
                "host": host,
                "training_origin": train["training_origin"],
                "elapsed_seconds": elapsed,
                "completed_epochs": epochs,
                "amortized_elapsed_seconds_per_epoch": elapsed / epochs
                if elapsed is not None
                else None,
                "selected_epoch": train["selected_epoch"],
                "train_receipt_path": str(
                    root / "run" / job["id"] / "train_receipt.json"
                ),
            }
        )
        records.append(
            {
                "job_id": job["id"],
                "dataset": job["dataset"],
                "arm": job["arm"],
                "seed": job["seed"],
                "host": host,
                "split": "validation",
                "metrics": metrics,
                "parameters": parameters,
                "checkpoint_sha256": train["checkpoint_sha256"],
                "state_sha256": train["state_sha256"],
                "selected_epoch": train["selected_epoch"],
                "prediction_path": str(prediction),
                "validation_receipt_path": str(
                    root / "run" / job["id"] / "validation_receipt.json"
                ),
                "train_receipt_path": str(
                    root / "run" / job["id"] / "train_receipt.json"
                ),
            }
        )
        values = _verify_cost(root, contract, host, job, train)
        _cost_metadata(contract, job, train, values, parameters, root)
        for value in values:
            value = dict(value)
            value["original_path"] = str(
                root
                / "cost"
                / "results"
                / job["id"]
                / (value["mode"] + "_rep" + str(value["timing_rep"]) + ".json")
            )
            costs.append(value)
        cache = read(root / "cost" / "batches" / (job["dataset"] + ".json"))
        evidence = (
            cache["target_ids_sha256"],
            tuple(row["tensor_sha256"] for row in cache["batches"]),
        )
        require(
            job["dataset"] not in batch_identity
            or batch_identity[job["dataset"]] == evidence,
            "Cross-host common Train batch targets/tensors differ",
        )
        batch_identity[job["dataset"]] = evidence
    require(
        sum(r["training_origin"] == "fresh_fit" for r in fits) == 9
        and sum(r["training_origin"] == "reused_verified_fit" for r in fits) == 18,
        "New/reused fit inventory differs",
    )
    for dataset in sorted({r["dataset"] for r in records}):
        for seed in SEEDS:
            group = [
                r for r in records if r["dataset"] == dataset and r["seed"] == seed
            ]
            require(
                len(
                    {
                        (
                            r["metrics"]["target_tuples_sha256"],
                            r["metrics"]["target_quantity_sha256"],
                        )
                        for r in group
                    }
                )
                == 1,
                "Matched arms have different Validation targets or quantities",
            )
    table = summarize_records(records)
    paired = paired_differences(records)
    cost_summaries = summarize_costs(costs)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "campaign": manager.NAME,
        "contract_sha256": canonical(contract),
        "source_files_sha256": contract["source"]["files_sha256"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "large_rule": "true_qty > quantity_boundaries_all_train_rows[-1]",
        "mean_definition": "Three common training seeds; sample SD uses ddof=1; unavailable scopes stay N/A",
        "cost_definition": "Fresh processes on fixed Train batches; three repetitions per GPU/seed/mode; GPUs are never pooled",
        "architecture_caveat": "Single MLP matches correction input and total parameters; it does not match Shared15 active-prefix capacity or function class. Availability bits permit history-length-only correction.",
        "records": records,
        "table": table,
        "paired_differences": paired,
        "adverse_differences": [r for r in paired if r["left_worse"]],
        "cost_records": costs,
        "cost_summaries": cost_summaries,
        "fit_records": fits,
    }


def _number(value):
    return "N/A" if value is None else f"{value:.6g}"


def render_markdown(result):
    lines = [
        "# Shared15 / single-MLP / encoder-only Validation control",
        "",
        "Validation only. No held-out Test metric was read or evaluated in this campaign.",
        "",
        result["mean_definition"],
        "",
        result["architecture_caveat"],
        "",
        "Large quantity uses the strict Train cutoff (`true_qty > final Train boundary`). Recorded TimeNLL is recomputed from the stored row values.",
        "",
        "| Dataset | Arm | Scope | Qty RMSE | Qty MAE | Recorded TimeNLL |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in result["table"]:
        for scope in SCOPES:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["dataset"],
                        row["arm"],
                        scope,
                        *[
                            _formatted(row["metrics"][scope][metric])
                            for metric in METRICS
                        ],
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "All 162 seed-wise absolute and relative differences, including source prediction paths, are in REPORT.json. Positive left-minus-right values mean a worse error/NLL for the left arm. No adverse result is omitted below.",
        "",
        "| Dataset | Seed | Left − right | Scope | Metric | Absolute | Relative | Original predictions |",
        "|---|---:|---|---|---|---:|---:|---|",
    ]
    for row in result["adverse_differences"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    str(row["seed"]),
                    row["left_arm"] + " − " + row["right_arm"],
                    row["scope"],
                    row["metric"],
                    _number(row["absolute"]),
                    _number(row["relative"]),
                    "[left]("
                    + row["left_prediction_path"]
                    + "), [right]("
                    + row["right_prediction_path"]
                    + ") (Validation)",
                ]
            )
            + " |"
        )
    lines += [
        "",
        result["cost_definition"],
        "Latency and throughput show median [minimum, maximum] across three fresh-process repetitions. Memory is the maximum of warmup and measured peaks inside each process; each summary shows median [minimum, maximum]. Reserved memory belonging to historical paused work is excluded from model allocator measurements.",
        "",
        "| Dataset | Arm | GPU | Seed | Mode | Compute ms | Transfer+compute ms | Targets/s | Allocated MiB | Reserved MiB | Parameters |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    def ranged(value):
        return f"{value['median']:.6g} [{value['minimum']:.6g}, {value['maximum']:.6g}]"

    for row in result["cost_summaries"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["arm"],
                    row["host"],
                    str(row["seed"]),
                    row["mode"],
                    *[
                        ranged(row[key])
                        for key in (
                            "compute_latency_ms",
                            "transfer_compute_latency_ms",
                            "throughput_targets_per_second",
                            "allocated_peak_mib",
                            "reserved_peak_mib",
                        )
                    ],
                    str(row["parameters"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Nine new single-MLP fits and eighteen reused B/D fits are reported separately. Reused historical fit times are not substituted for fresh cost measurements. Elapsed/epoch below includes Validation and checkpoint saving; it is an amortized value, not a recent or pure training-epoch speed.",
        "",
        "| Dataset | Arm | GPU | Seed | Origin | End epoch | Selected epoch | Fit elapsed s | Amortized s/epoch |",
        "|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["fit_records"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["arm"],
                    row["host"],
                    str(row["seed"]),
                    row["training_origin"],
                    _number(row["completed_epochs"]),
                    str(row["selected_epoch"]),
                    _number(row["elapsed_seconds"]),
                    _number(row["amortized_elapsed_seconds_per_epoch"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_csv(result):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ("dataset", "arm", "split", "scope", "metric", "mean", "sample_sd_ddof1", "n")
    )
    for row in result["table"]:
        for scope in SCOPES:
            for metric in METRICS:
                summary = row["metrics"][scope][metric]
                writer.writerow(
                    (
                        row["dataset"],
                        row["arm"],
                        "validation",
                        scope,
                        metric,
                        summary["mean"],
                        summary["sample_sd_ddof1"],
                        summary["n"],
                    )
                )
    return stream.getvalue()


def verify_outputs(output, result):
    output = Path(output)
    require(
        output.is_dir()
        and not output.is_symlink()
        and {path.name for path in output.iterdir()} == OUTPUTS,
        "Partial or unexpected comparison inventory",
    )
    require(
        all(
            path.is_file() and not path.is_symlink() and path.stat().st_size > 0
            for path in output.iterdir()
        ),
        "Missing, empty or symbolic comparison output",
    )
    require(
        read(output / "REPORT.json") == result
        and (output / "REPORT.md").read_text() == render_markdown(result)
        and (output / "TABLE.csv").read_text() == render_csv(result),
        "Comparison differs from pure recomputation",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    require(
        args.contract.resolve()
        == (manager.BUNDLE / "execution_contract.json").resolve()
        and args.retrieval.resolve() == (manager.BUNDLE / "retrieval").resolve()
        and args.output.resolve() == (manager.BUNDLE / "comparison_v1").resolve(),
        "Only this campaign's fixed local paths are admitted",
    )
    require(
        not args.contract.is_symlink() and sha(args.contract) == args.contract_sha,
        "Externally pinned raw contract differs",
    )
    result = aggregate(read(args.contract), args.retrieval)
    if args.check_only:
        if args.output.exists():
            verify_outputs(args.output, result)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "evaluation_scope": "validation_only",
                    "records": 27,
                    "paired_differences": 162,
                    "cost_records": 162,
                    "output_verified": args.output.exists(),
                    "remote_calls": 0,
                }
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    manager.write(args.output / "REPORT.json", result)
    with (args.output / "REPORT.md").open("x") as stream:
        stream.write(render_markdown(result))
    with (args.output / "TABLE.csv").open("x") as stream:
        stream.write(render_csv(result))
    verify_outputs(args.output, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "evaluation_scope": "validation_only",
                "output": str(args.output),
                "remote_calls": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
