from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "paper"
CONTRACT_ROOT = PAPER_ROOT / "contracts"
TABLE_ROOT = PAPER_ROOT / "tables"
DATA_ROOT = PAPER_ROOT / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ("train", "validation", "test")
HASH_ROLES = ("with_split", "train", "validation", "test", "split_manifest")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def project_path(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        raise ValueError(f"Path escapes project root: {relative_path}")
    return path


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def percentile(frame: pl.DataFrame, column: str, quantile: float) -> float:
    value = frame.select(
        pl.col(column).quantile(quantile, interpolation="nearest")
    ).item()
    return float(value)


def split_summary(frame: pl.DataFrame, split: str) -> dict[str, Any]:
    scoped = frame.filter(pl.col("chronological_split") == split)
    return {
        "rows": scoped.height,
        "series": scoped.select(pl.col("oper_part_no").n_unique()).item(),
        "qty_median": percentile(scoped, "demand_qty", 0.5),
        "qty_p95": percentile(scoped, "demand_qty", 0.95),
        "qty_max": float(scoped.select(pl.col("demand_qty").max()).item()),
    }


def split_mark_counts(frame: pl.DataFrame, split: str) -> dict[int, int]:
    rows = (
        frame.filter(pl.col("chronological_split") == split)
        .group_by("mark")
        .len()
        .sort("mark")
        .iter_rows(named=True)
    )
    return {int(row["mark"]): int(row["len"]) for row in rows}


def sequence_summary(frame: pl.DataFrame) -> dict[str, Any]:
    lengths = frame.group_by("oper_part_no").len().get_column("len")
    return {
        "series": len(lengths),
        "seq_len_mean": float(lengths.mean()),
        "seq_len_median": float(lengths.median()),
        "seq_len_p95": float(lengths.quantile(0.95, interpolation="nearest")),
        "seq_len_max": int(lengths.max()),
    }


def target_counts(frame: pl.DataFrame) -> dict[str, int]:
    split_rows = Counter(frame.get_column("chronological_split").to_list())
    first_splits = (
        frame.sort(["oper_part_no", "seq"])
        .group_by("oper_part_no", maintain_order=True)
        .agg(pl.col("chronological_split").first().alias("first_split"))
        .get_column("first_split")
        .to_list()
    )
    first_counts = Counter(first_splits)
    return {
        split: int(split_rows[split] - first_counts[split]) for split in SPLITS
    }


def manifest_split_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["chronological_split"]: row
        for row in manifest["summary"]["split_counts"]
    }


def manifest_mark_map(manifest: dict[str, Any]) -> dict[str, dict[int, int]]:
    mapped = {split: {} for split in SPLITS}
    for row in manifest["summary"]["mark_counts"]:
        mapped[row["chronological_split"]][int(row["mark"])] = int(row["len"])
    return mapped


def compare_numeric(
    errors: list[str],
    dataset_id: str,
    label: str,
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-9,
) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        errors.append(
            f"{dataset_id}: {label} mismatch: actual={actual}, expected={expected}"
        )


def audit_dataset(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_id = spec["dataset_id"]
    manifest_path = project_path(spec["manifest_path"])
    with_split_path = project_path(spec["with_split_path"])
    split_paths = {
        split: project_path(path) for split, path in spec["split_paths"].items()
    }
    source_input_path = project_path(spec["source_input_path"])

    manifest = load_json(manifest_path)
    frame = pl.read_parquet(with_split_path)
    errors: list[str] = []

    required_columns = {
        "oper_part_no",
        "seq",
        "delta_t",
        "demand_qty",
        "mark",
        "chronological_split",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        errors.append(f"{dataset_id}: missing columns {missing_columns}")

    actual_splits = set(frame.get_column("chronological_split").unique().to_list())
    if actual_splits != set(SPLITS):
        errors.append(
            f"{dataset_id}: split values {sorted(actual_splits)} != {list(SPLITS)}"
        )

    null_counts = frame.select(
        [pl.col(column).null_count().alias(column) for column in required_columns]
    ).row(0, named=True)
    nonzero_nulls = {key: value for key, value in null_counts.items() if value}
    if nonzero_nulls:
        errors.append(f"{dataset_id}: required-column nulls {nonzero_nulls}")

    nonpositive_qty = frame.filter(pl.col("demand_qty") <= 0).height
    if nonpositive_qty:
        errors.append(f"{dataset_id}: {nonpositive_qty} non-positive quantities")

    unique_keys = frame.select(
        pl.struct(["oper_part_no", "seq"]).n_unique().alias("n")
    ).item()
    duplicate_keys = frame.height - int(unique_keys)
    if duplicate_keys:
        errors.append(f"{dataset_id}: {duplicate_keys} duplicate (series, seq) keys")

    ranked = frame.sort(["oper_part_no", "seq"]).with_columns(
        pl.when(pl.col("chronological_split") == "train")
        .then(pl.lit(0))
        .when(pl.col("chronological_split") == "validation")
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias("_split_rank")
    )
    order_violations = (
        ranked.with_columns(
            pl.col("_split_rank").diff().over("oper_part_no").alias("_rank_diff")
        )
        .filter(pl.col("_rank_diff") < 0)
        .height
    )
    if order_violations:
        errors.append(f"{dataset_id}: {order_violations} chronological split regressions")

    manifest_splits = manifest_split_map(manifest)
    manifest_marks = manifest_mark_map(manifest)
    actual_summaries: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        actual = split_summary(frame, split)
        actual_summaries[split] = actual
        expected = manifest_splits[split]
        for field in ("rows", "series", "qty_median", "qty_p95", "qty_max"):
            compare_numeric(
                errors,
                dataset_id,
                f"{split}.{field}",
                actual[field],
                expected[field],
            )
        actual_marks = split_mark_counts(frame, split)
        if actual_marks != manifest_marks[split]:
            errors.append(f"{dataset_id}: {split} mark counts differ from manifest")

        split_frame = pl.read_parquet(split_paths[split])
        if split_frame.height != actual["rows"]:
            errors.append(
                f"{dataset_id}: {split} artifact rows={split_frame.height}, "
                f"with_split rows={actual['rows']}"
            )
        split_series = split_frame.select(pl.col("oper_part_no").n_unique()).item()
        if int(split_series) != int(actual["series"]):
            errors.append(
                f"{dataset_id}: {split} artifact series={split_series}, "
                f"with_split series={actual['series']}"
            )

    actual_sequence = sequence_summary(frame)
    expected_sequence = manifest["summary"]["sequence_length_summary"]
    for field in (
        "series",
        "seq_len_mean",
        "seq_len_median",
        "seq_len_p95",
        "seq_len_max",
    ):
        compare_numeric(
            errors,
            dataset_id,
            f"sequence.{field}",
            actual_sequence[field],
            expected_sequence[field],
        )

    hash_paths = {
        "source_input": source_input_path,
        "with_split": with_split_path,
        **split_paths,
        "split_manifest": manifest_path,
    }
    hash_rows = []
    for role, path in hash_paths.items():
        hash_rows.append(
            {
                "dataset_id": dataset_id,
                "paper_name": spec["paper_name"],
                "role": role,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    hashes_by_role = {row["role"]: row["sha256"] for row in hash_rows}
    for role, expected_hash in spec["expected_hashes"].items():
        actual_hash = hashes_by_role.get(role)
        if actual_hash != expected_hash:
            errors.append(
                f"{dataset_id}: frozen hash mismatch for {role}: "
                f"actual={actual_hash}, expected={expected_hash}"
            )

    identity_rows = [row for row in hash_rows if row["role"] in HASH_ROLES]
    identity_payload = json.dumps(
        [
            {
                "role": row["role"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in sorted(identity_rows, key=lambda row: row["role"])
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    dataset_contract_sha256 = sha256_text(identity_payload)

    if errors:
        raise ValueError("\n".join(errors))

    observed_marks = frame.select(pl.col("mark").n_unique()).item()
    overall_qty = {
        "median": percentile(frame, "demand_qty", 0.5),
        "p95": percentile(frame, "demand_qty", 0.95),
        "max": float(frame.select(pl.col("demand_qty").max()).item()),
    }
    targets = target_counts(frame)

    audit = {
        "dataset_id": dataset_id,
        "paper_name": spec["paper_name"],
        "status": "PASS",
        "grain": ["oper_part_no", "seq"],
        "rows": frame.height,
        "series": actual_sequence["series"],
        "split_summaries": actual_summaries,
        "target_samples": targets,
        "sequence_summary": actual_sequence,
        "quantity_summary": overall_qty,
        "observed_quantity_marks": int(observed_marks),
        "model_num_marks_including_padding": int(observed_marks) + 1,
        "scale_base": float(manifest["magnitude_rule"]["scale_base"]),
        "dataset_contract_sha256": dataset_contract_sha256,
        "checks": {
            "required_columns": "PASS",
            "required_column_nulls": 0,
            "non_positive_quantity_rows": 0,
            "duplicate_series_seq_keys": 0,
            "chronological_split_regressions": 0,
            "manifest_statistics_match": True,
            "split_artifact_counts_match": True,
            "frozen_5080_hashes_match": True,
        },
        "hashes": hashes_by_role,
    }
    return audit, hash_rows


def model_parameter_count(
    dataset: dict[str, Any],
    model_spec: dict[str, Any],
    observed_marks: int,
    common: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
    from simple_lab_test.search.common.models import (
        build_model,
        default_thp_candidates,
        default_titan_candidates,
        find_candidate_by_name,
        make_rmtpp_proxy_candidate,
    )

    family = model_spec["family"]
    if family == "rmtpp":
        candidate = make_rmtpp_proxy_candidate(
            model_spec["rmtpp_hidden_dim"], "gru"
        )
    elif family == "titantpp":
        candidate = find_candidate_by_name(
            default_titan_candidates(), model_spec["candidate"]
        )
    elif family == "thp":
        candidate = find_candidate_by_name(
            default_thp_candidates(), model_spec["candidate"]
        )
    else:
        raise ValueError(f"Unsupported model family: {family}")

    config = ExperimentConfig(
        base_dir=str(PAPER_ROOT / "data" / "parameter_count_scratch"),
        device="cpu",
        reproducibility_mode="strict",
        lookback_weeks=int(dataset["lookback"]),
        max_seq_len=int(dataset["max_seq_len"]),
        batch_size=int(common["batch_size"]),
        lr=float(common["learning_rate"]),
        lambda_dt=float(common["lambda_dt"]),
        grad_clip=float(common["gradient_clip"]),
        epochs=int(common["initial_epoch_budget"]),
        seeds=tuple(common["seeds"]),
        datasets=(dataset["dataset_id"],),
        models=(family,),
        titan_profile="dataset_best",
        rmtpp_rnn_type="gru",
        rmtpp_mark_emb_dim=int(common["mark_embedding_dim"]),
        rmtpp_hidden_dim=int(model_spec["rmtpp_hidden_dim"]),
        value_head_activation=common["value_head_activation"],
        value_head_mode=model_spec["value_head_mode"],
        time_head_mode="shared",
        qty_mark_gradient_mode=model_spec["qty_mark_gradient_mode"],
        value_encoder_gradient_mode="coupled",
        marker_loss_mode="ce",
        lambda_ordinal=0.0,
        qty_decoder_mode=common["quantity_decoder"],
        loss_mode=model_spec["quantity_objective"],
        value_input_mode=model_spec["quantity_input"],
        value_input_emb_dim=int(common["value_input_embedding_dim"]),
        train_loss_scope=common["train_loss_scope"],
        split_mode="fixed",
        evaluation_scope="validation_only",
        test_time_memory="none",
    )
    run_config = RunConfig(
        dataset_name=dataset["dataset_id"],
        dataset_kind=dataset["dataset_kind"],
        model_name=family,
        candidate_name=model_spec["candidate"],
        candidate=candidate,
        seed=int(common["seeds"][0]),
        epochs=int(common["initial_epoch_budget"]),
        scale_base=float(dataset["scale_base"]),
        titan_profile="dataset_best",
    )
    marked_meta = {
        "num_marks": int(observed_marks) + 1,
        "magnitude_global_mean": 0.0,
        "magnitude_global_var": 1.0,
        "magnitude_global_std": 1.0,
        "magnitude_sigma_floor": 1e-3,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="dropout option adds dropout after all but last recurrent layer",
            category=UserWarning,
        )
        model, rmtpp_config, encoder_config = build_model(
            cfg=config,
            run_cfg=run_config,
            marked_meta=marked_meta,
        )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    resolved = {
        "rmtpp_config": asdict(rmtpp_config),
        "encoder_config": asdict(encoder_config) if encoder_config is not None else None,
    }
    return int(total), int(trainable), resolved


def build_model_rows(
    comparison_contract: dict[str, Any],
    dataset_audits: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common = comparison_contract["common_protocol"]
    rows: list[dict[str, Any]] = []
    resolved_models: list[dict[str, Any]] = []
    for dataset in comparison_contract["datasets"]:
        audit = dataset_audits[dataset["dataset_id"]]
        observed_marks = int(audit["observed_quantity_marks"])
        for model_spec in dataset["models"]:
            total, trainable, resolved = model_parameter_count(
                dataset, model_spec, observed_marks, common
            )
            row = {
                "dataset": dataset["paper_name"],
                "dataset_id": dataset["dataset_id"],
                "comparison_role": model_spec["comparison_role"],
                "model": model_spec["paper_label"],
                "encoder": model_spec["encoder_summary"],
                "quantity_input": model_spec["quantity_input"],
                "quantity_objective": model_spec["quantity_objective"],
                "value_head": model_spec["value_head_mode"],
                "quantity_to_mark_gradient": model_spec["qty_mark_gradient_mode"],
                "lookback": dataset["lookback"],
                "max_seq_len": dataset["max_seq_len"],
                "scale_base": dataset["scale_base"],
                "parameter_count": total,
                "trainable_parameter_count": trainable,
            }
            rows.append(row)
            resolved_models.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "model_id": model_spec["model_id"],
                    "parameter_count": total,
                    "trainable_parameter_count": trainable,
                    **resolved,
                }
            )
    return rows, resolved_models


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]], aligns: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.2f}"


def write_t1_markdown(dataset_specs: dict[str, dict[str, Any]], audits: list[dict[str, Any]]) -> None:
    stats_rows = []
    task_rows = []
    hash_rows = []
    for audit in audits:
        spec = dataset_specs[audit["dataset_id"]]
        split = audit["split_summaries"]
        targets = audit["target_samples"]
        sequence = audit["sequence_summary"]
        quantity = audit["quantity_summary"]
        stats_rows.append(
            [
                audit["paper_name"],
                format_number(audit["series"]),
                format_number(audit["rows"]),
                f"{format_number(split['train']['rows'])} / {format_number(split['validation']['rows'])} / {format_number(split['test']['rows'])}",
                f"{format_number(targets['train'])} / {format_number(targets['validation'])} / {format_number(targets['test'])}",
                f"{format_number(sequence['seq_len_median'])} / {format_number(sequence['seq_len_p95'])} / {format_number(sequence['seq_len_max'])}",
                f"{format_number(quantity['median'])} / {format_number(quantity['p95'])} / {format_number(quantity['max'])}",
                audit["observed_quantity_marks"],
                format_number(audit["scale_base"]),
            ]
        )
        task_rows.append(
            [
                audit["paper_name"],
                spec["sequence_unit"],
                spec["time_unit"],
                spec["event_definition"],
                spec["quantity_definition"],
            ]
        )
        hash_rows.append(
            [
                audit["paper_name"],
                audit["hashes"]["with_split"][:12],
                audit["hashes"]["split_manifest"][:12],
                audit["dataset_contract_sha256"][:12],
            ]
        )

    content = "\n".join(
        [
            "# T1. Dataset statistics and task construction",
            "",
            "> Status: frozen fixed-split dataset audit PASS. This table does not use model predictions or held-out test performance.",
            "",
            "## T1a. Dataset and split statistics",
            "",
            markdown_table(
                [
                    "Dataset",
                    "Sequences",
                    "Events",
                    "Events T/V/Test",
                    "Targets T/V/Test",
                    "Seq. length med/p95/max",
                    "Quantity med/p95/max",
                    "Marks K",
                    "Base b",
                ],
                stats_rows,
                [":---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            ),
            "",
            "`Events` counts every fixed-split event row. `Targets` excludes the first event of each sequence because next-event prediction requires an observed predecessor. T/V/Test denotes train/validation/test.",
            "The nominal 70/15/15 split is applied chronologically within each sequence. Aggregate event shares can differ from the nominal ratio because sequence boundaries are integer-valued and many sequences are short.",
            "",
            "## T1b. Task construction",
            "",
            markdown_table(
                ["Dataset", "Sequence unit", "Time unit", "Event", "Quantity"],
                task_rows,
                [":---", ":---", ":---", ":---", ":---"],
            ),
            "",
            "## Frozen dataset identity",
            "",
            markdown_table(
                ["Dataset", "with-split SHA-256", "manifest SHA-256", "contract SHA-256"],
                hash_rows,
                [":---", ":---", ":---", ":---"],
            ),
            "",
            "The manuscript may cite the 12-character identifiers above. Full hashes, sizes and paths are stored in `paper/data/T1_dataset_hashes.csv` and `paper/data/T1_dataset_audit.json`.",
            "",
        ]
    )
    (TABLE_ROOT / "T1_dataset_statistics.md").write_text(content, encoding="utf-8")


def write_t2_markdown(
    comparison_contract: dict[str, Any], model_rows: list[dict[str, Any]]
) -> None:
    rows = []
    for row in model_rows:
        rows.append(
            [
                row["dataset"],
                row["model"],
                row["encoder"],
                row["quantity_input"],
                row["quantity_objective"],
                row["value_head"],
                row["quantity_to_mark_gradient"],
                f"{row['lookback']} / {row['max_seq_len']}",
                f"{row['parameter_count']:,}",
            ]
        )

    common = comparison_contract["common_protocol"]
    continuation = common["continuation_rule"]
    protocol_rows = [
        ["Data split", "Nominal chronological 70/15/15 within each sequence; full observed history may be context, while the target split determines training or evaluation membership"],
        ["Seeds", ", ".join(str(seed) for seed in common["seeds"])],
        ["Optimization", f"{common['optimizer']}; learning rate {common['learning_rate']}; weight decay {common['weight_decay']}; scheduler {common['learning_rate_scheduler']}; batch size {common['batch_size']}; gradient clip {common['gradient_clip']}"],
        ["Initial budget", f"e{common['initial_epoch_budget']} for every declared run"],
        ["Continuation trigger", f"Dataset-level e{common['continuation_epoch_budget']} continuation when any primary model has a best epoch in {continuation['best_epoch_boundary_start']}-{continuation['best_epoch_boundary_end']} for at least {continuation['minimum_trigger_seeds']}/3 seeds, or the {continuation['late_window_start']}-{continuation['late_window_end']} window improves best validation NLL by at least {continuation['minimum_late_relative_nll_improvement'] * 100:.1f}% in at least {continuation['minimum_trigger_seeds']}/3 seeds"],
        ["Checkpoint", "Minimum validation total NLL; final and composite-score checkpoints are diagnostic only"],
        ["Development scope", "Validation only; reproducibility mode strict"],
        ["Held-out test", "Locked until model identity, epoch continuation and checkpoint rules are frozen; evaluated once"],
        ["Benchmark source revision", comparison_contract["benchmark_source_revision"]],
    ]
    content = "\n".join(
        [
            "# T2. Matched model and training configuration",
            "",
            "> Status: frozen for validation. Parameter counts are derived from the declared model classes and dataset-specific mark cardinality on CPU.",
            "",
            "## T2a. Model contract",
            "",
            markdown_table(
                [
                    "Dataset",
                    "Model",
                    "Encoder",
                    "Quantity input",
                    "Objective",
                    "Value head",
                    "Qty-to-mark grad.",
                    "Lookback / max len",
                    "Parameters",
                ],
                rows,
                [":---", ":---", ":---", ":---", ":---", ":---", ":---", "---:", "---:"],
            ),
            "",
            "RMTPP-matched and THP-matched use the same residual quantity input, hybrid quantity objective and output heads as the dataset's TitanTPP primary model. Taxi V2 control is an ablation row and is not a fifth primary baseline.",
            "All RMTPP baselines use one GRU layer. The configured RNN dropout of 0.1 is therefore inactive under PyTorch's inter-layer dropout semantics; THP and TitanTPP retain dropout 0.1 in their multi-layer encoders.",
            "",
            "## T2b. Shared training and evaluation protocol",
            "",
            markdown_table(
                ["Item", "Frozen rule"],
                protocol_rows,
                [":---", ":---"],
            ),
            "",
            "The earlier all-e800 meeting contract is retained as provenance, but its epoch policy is superseded by this approved e300-first continuation rule. Model identities, split, seeds, loss settings and test lock remain unchanged.",
            "",
        ]
    )
    (TABLE_ROOT / "T2_model_training_contract.md").write_text(content, encoding="utf-8")


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_contract = load_json(CONTRACT_ROOT / "datasets.json")
    comparison_contract = load_json(CONTRACT_ROOT / "final_fair_comparison.json")
    dataset_specs = {
        spec["dataset_id"]: spec for spec in dataset_contract["datasets"]
    }

    audits: list[dict[str, Any]] = []
    all_hash_rows: list[dict[str, Any]] = []
    for spec in dataset_contract["datasets"]:
        audit, hash_rows = audit_dataset(spec)
        audits.append(audit)
        all_hash_rows.extend(hash_rows)
    audits_by_id = {audit["dataset_id"]: audit for audit in audits}

    contract_dataset_ids = {
        dataset["dataset_id"] for dataset in comparison_contract["datasets"]
    }
    if contract_dataset_ids != set(audits_by_id):
        raise ValueError(
            f"T1/T2 dataset mismatch: T1={sorted(audits_by_id)}, "
            f"T2={sorted(contract_dataset_ids)}"
        )

    model_source_hashes = {
        path: sha256_file(project_path(path))
        for path in comparison_contract["expected_model_source_sha256"]
    }
    if model_source_hashes != comparison_contract["expected_model_source_sha256"]:
        mismatches = {
            path: {
                "actual": model_source_hashes[path],
                "expected": expected,
            }
            for path, expected in comparison_contract[
                "expected_model_source_sha256"
            ].items()
            if model_source_hashes[path] != expected
        }
        raise ValueError(f"Frozen model source hash mismatch: {mismatches}")

    model_rows, resolved_models = build_model_rows(
        comparison_contract, audits_by_id
    )

    t1_csv_rows = []
    for audit in audits:
        spec = dataset_specs[audit["dataset_id"]]
        row = {
            "dataset": audit["paper_name"],
            "dataset_id": audit["dataset_id"],
            "sequence_unit": spec["sequence_unit"],
            "time_unit": spec["time_unit"],
            "event_definition": spec["event_definition"],
            "quantity_definition": spec["quantity_definition"],
            "sequences": audit["series"],
            "events_total": audit["rows"],
            "events_train": audit["split_summaries"]["train"]["rows"],
            "events_validation": audit["split_summaries"]["validation"]["rows"],
            "events_test": audit["split_summaries"]["test"]["rows"],
            "targets_train": audit["target_samples"]["train"],
            "targets_validation": audit["target_samples"]["validation"],
            "targets_test": audit["target_samples"]["test"],
            "sequence_length_median": audit["sequence_summary"]["seq_len_median"],
            "sequence_length_p95": audit["sequence_summary"]["seq_len_p95"],
            "sequence_length_max": audit["sequence_summary"]["seq_len_max"],
            "quantity_median": audit["quantity_summary"]["median"],
            "quantity_p95": audit["quantity_summary"]["p95"],
            "quantity_max": audit["quantity_summary"]["max"],
            "observed_quantity_marks": audit["observed_quantity_marks"],
            "scale_base": audit["scale_base"],
            "with_split_sha256": audit["hashes"]["with_split"],
            "split_manifest_sha256": audit["hashes"]["split_manifest"],
            "dataset_contract_sha256": audit["dataset_contract_sha256"],
        }
        t1_csv_rows.append(row)

    write_csv(
        TABLE_ROOT / "T1_dataset_statistics.csv",
        t1_csv_rows,
        t1_csv_rows[0].keys(),
    )
    write_csv(
        DATA_ROOT / "T1_dataset_hashes.csv",
        all_hash_rows,
        all_hash_rows[0].keys(),
    )
    write_csv(
        TABLE_ROOT / "T2_model_training_contract.csv",
        model_rows,
        model_rows[0].keys(),
    )

    legacy_contract_path = project_path(comparison_contract["legacy_contract_path"])
    t1_audit_payload = {
        "schema_version": 1,
        "status": "PASS",
        "generated_by": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "local_git_revision": git_revision(),
        "dataset_contract_sha256": sha256_file(CONTRACT_ROOT / "datasets.json"),
        "datasets": audits,
        "checks": {
            "all_dataset_audits_pass": True,
            "t1_t2_dataset_sets_match": True,
            "frozen_5080_hashes_match": True,
            "held_out_test_performance_used": False,
        },
    }
    (DATA_ROOT / "T1_dataset_audit.json").write_text(
        json.dumps(t1_audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    t2_audit_payload = {
        "schema_version": 1,
        "status": "PASS",
        "generated_by": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "local_git_revision": git_revision(),
        "benchmark_source_revision": comparison_contract[
            "benchmark_source_revision"
        ],
        "comparison_contract_sha256": sha256_file(
            CONTRACT_ROOT / "final_fair_comparison.json"
        ),
        "legacy_contract": {
            "path": str(legacy_contract_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(legacy_contract_path),
            "epoch_policy_superseded": True,
        },
        "model_source_hashes": model_source_hashes,
        "checks": {
            "frozen_model_source_hashes_match": True,
            "parameter_counts_resolved": len(resolved_models),
            "held_out_test_evaluated": comparison_contract["common_protocol"][
                "held_out_test_evaluated"
            ],
        },
    }
    (DATA_ROOT / "T2_contract_audit.json").write_text(
        json.dumps(t2_audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (DATA_ROOT / "T2_resolved_model_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "benchmark_source_revision": comparison_contract[
                    "benchmark_source_revision"
                ],
                "held_out_test_evaluated": comparison_contract["common_protocol"][
                    "held_out_test_evaluated"
                ],
                "models": resolved_models,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    write_t1_markdown(dataset_specs, audits)
    write_t2_markdown(comparison_contract, model_rows)

    print(f"T1/T2 generation PASS: {len(audits)} datasets, {len(model_rows)} model rows")
    for audit in audits:
        print(
            f"- {audit['paper_name']}: events={audit['rows']:,}, "
            f"series={audit['series']:,}, contract={audit['dataset_contract_sha256'][:12]}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"T1/T2 generation FAILED: {exc}", file=sys.stderr)
        raise
