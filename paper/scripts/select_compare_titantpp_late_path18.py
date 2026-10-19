"""Pure common-condition selection and paired late-path Validation comparison.

Selection uses all stored MLP16 Validation endpoints, never candidate results.
Dataset errors are normalized to one fixed reference before macro averaging;
raw quantity errors from different datasets are never pooled.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
LOSSES = ("log_mse", "raw_mse", "raw_mae")
TRANSFORMS = ("log1p", "raw")
BASELINE = "titantpp_history_mlp_width16"
OBSERVATION = "titantpp_mlp16_late_observation_v1"
HIDDEN = "titantpp_mlp16_late_hidden_v1"
MODELS = (BASELINE, OBSERVATION, HIDDEN)
COHORTS = ("overall", "cap_hit", "tail")
METRICS = ("qty_rmse", "qty_mae", "time_nll")
CAP_VALUES = dict(zip(DATASETS, (65., 10., 20.)))
METRIC_METADATA = {"cap_hit_rule": "q == cap_value", "tail_rule": "q > cap_value",
                   "time_metric": "recorded_positive_integer_time_nll"}
CONTRASTS = ((OBSERVATION, BASELINE), (HIDDEN, BASELINE), (OBSERVATION, HIDDEN))
REFERENCE_CONDITION = {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _expected_host(dataset, seed):
    return "5080" if dataset == DATASETS[0] or (dataset == DATASETS[2] and seed == 42) else "5090"


def _model(row):
    require(all(row[key] in MODELS for key in ("model", "model_name") if key in row),
            "Unknown explicit model identifier")
    values = [row[key] for key in ("model", "model_name", "arm") if key in row and row[key] in MODELS]
    require(bool(values) and len(set(values)) == 1, "Missing or inconsistent admitted model identifier")
    return values[0]


def _validate_row(row):
    require(isinstance(row, dict), "Endpoint must be a mapping")
    require(row.get("dataset") in DATASETS and type(row.get("seed")) is int and row["seed"] in SEEDS,
            "Unknown dataset or seed")
    require(row.get("host") == _expected_host(row["dataset"], row["seed"]), "Endpoint native host differs")
    require(row.get("time_input") in TRANSFORMS and row.get("loss_mode") in LOSSES,
            "Unknown common condition")
    require(type(row.get("time_alpha")) in (int, float) and row["time_alpha"] == 0.
            and row.get("time_head_trained") is False, "Time objective must remain excluded")
    require(row.get("scope", "validation_only") in ("validation_only", "validation")
            and row.get("evaluation_scope", "validation_only") == "validation_only"
            and row.get("split", "validation") == "validation"
            and row.get("held_out_test_evaluated", False) is False, "Only Validation endpoints are admitted")
    require(type(row.get("selected_epoch")) is int and row["selected_epoch"] >= 1,
            "Selected epoch must be a positive integer")
    require(bool(row.get("target_identity")), "Missing target identity")
    metrics = row.get("metrics")
    require(isinstance(metrics, dict) and set(COHORTS).issubset(metrics), "Expected all three metric cohorts")
    for key, expected in METRIC_METADATA.items():
        if key in metrics:
            require(metrics[key] == expected, "Invalid metric cohort definition: " + key)
    if "cap_value" in metrics:
        value = metrics["cap_value"]
        require(type(value) in (int, float) and math.isfinite(value) and value == CAP_VALUES[row["dataset"]],
                "Invalid fixed Train-Q3 cap value")
    for cohort in COHORTS:
        values = metrics[cohort]
        require(isinstance(values, dict) and all(key in values for key in ("count", *METRICS)),
                "Missing cohort count or metric")
        count = values["count"]
        require(type(count) is int and count >= 0 and (cohort != "overall" or count > 0),
                "Invalid cohort count")
        for metric in METRICS:
            value = values[metric]
            if count == 0:
                require(value is None, "Empty cohort metrics must be N/A")
            else:
                require(type(value) in (int, float) and math.isfinite(value), "Invalid or nonfinite metric")
                require(metric == "time_nll" or value >= 0., "Negative quantity error")
    require(metrics["cap_hit"]["count"] + metrics["tail"]["count"] <= metrics["overall"]["count"],
            "Cohort counts exceed overall population")


def _validate_pair(reference, row):
    require(row["host"] == reference["host"] and row["target_identity"] == reference["target_identity"],
            "Paired target/native host mismatch")
    require(all(row["metrics"][c]["count"] == reference["metrics"][c]["count"] for c in COHORTS),
            "Paired cohort counts differ")
    for key in (*METRIC_METADATA, "cap_value"):
        if key in reference["metrics"] or key in row["metrics"]:
            require(key in reference["metrics"] and key in row["metrics"]
                    and reference["metrics"][key] == row["metrics"][key], "Paired cohort definition metadata differs")


def _condition(condition):
    require(isinstance(condition, dict), "Common condition must be a mapping")
    if "selected_condition" in condition:
        condition = condition["selected_condition"]
    require(condition.get("time_input") in TRANSFORMS and condition.get("loss_mode") in LOSSES
            and type(condition.get("time_alpha")) in (int, float) and condition["time_alpha"] == 0.,
            "Invalid fixed common condition")
    return {"time_input": condition["time_input"], "loss_mode": condition["loss_mode"], "time_alpha": 0.}


def _summary(values):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    require(math.isfinite(mean) and math.isfinite(sd), "Nonfinite aggregate statistic")
    return mean, sd


def select_common_condition(records):
    """Select one of six conditions using fixed-reference macro RMSE then MAE.

    Every dataset contributes one ratio of its three-seed mean to the fixed
    log-time/log-loss mean. Exact score ties follow the documented input/loss
    order. This is a Validation-based experimental setting, not a new method.
    """
    require(isinstance(records, (list, tuple)), "Endpoints must be a sequence")
    indexed = {}
    for row in records:
        _validate_row(row)
        if any(key in row for key in ("model", "model_name")) or row.get("arm") in MODELS:
            require(_model(row) == BASELINE, "Common-setting selection must use stored MLP16 endpoints only")
        key = tuple(row[k] for k in ("dataset", "seed", "time_input", "loss_mode"))
        require(key not in indexed, "Expected exact 54-endpoint common-condition matrix")
        indexed[key] = row
    expected = {(d, s, t, loss) for d in DATASETS for s in SEEDS for t in TRANSFORMS for loss in LOSSES}
    require(len(records) == len(indexed) == 54 and set(indexed) == expected,
            "Expected exact 54-endpoint common-condition matrix")
    for dataset in DATASETS:
        for seed in SEEDS:
            reference = indexed[dataset, seed, "log1p", "log_mse"]
            for transform in TRANSFORMS:
                for loss in LOSSES:
                    _validate_pair(reference, indexed[dataset, seed, transform, loss])
        for cohort in COHORTS:
            counts = {indexed[dataset, seed, "log1p", "log_mse"]["metrics"][cohort]["count"] for seed in SEEDS}
            require(len(counts) == 1, "Seed cohort counts differ")
    fixed_means = {dataset: {metric: statistics.mean(
        indexed[dataset, seed, "log1p", "log_mse"]["metrics"]["overall"][metric] for seed in SEEDS)
        for metric in METRICS[:2]} for dataset in DATASETS}
    require(all(value > 0. for means in fixed_means.values() for value in means.values()),
            "Fixed reference quantity means must be positive for ratio selection")
    scores = []
    for transform in TRANSFORMS:
        for loss in LOSSES:
            datasets = []
            for dataset in DATASETS:
                group = [indexed[dataset, seed, transform, loss] for seed in SEEDS]
                stats = {metric: _summary([row["metrics"]["overall"][metric] for row in group])
                         for metric in METRICS[:2]}
                means = {metric: stats[metric][0] for metric in METRICS[:2]}
                ratios = {metric: means[metric] / fixed_means[dataset][metric] for metric in METRICS[:2]}
                require(all(math.isfinite(value) for value in ratios.values()), "Nonfinite selection ratio")
                datasets.append({"dataset": dataset, "means": means,
                    "sample_sd_ddof1": {metric: stats[metric][1] for metric in METRICS[:2]},
                    "fixed_reference_means": deepcopy(fixed_means[dataset]), "ratios": ratios,
                    "seed_records": deepcopy(group)})
            scores.append({"condition": {"time_input": transform, "loss_mode": loss, "time_alpha": 0.},
                "macro_rmse_ratio": statistics.mean(row["ratios"]["qty_rmse"] for row in datasets),
                "macro_mae_ratio": statistics.mean(row["ratios"]["qty_mae"] for row in datasets),
                "datasets": datasets})
    def rank(row):
        return (row["macro_rmse_ratio"], row["macro_mae_ratio"],
                TRANSFORMS.index(row["condition"]["time_input"]), LOSSES.index(row["condition"]["loss_mode"]))
    selected = min(scores, key=rank)
    return {"schema": "titantpp_late_path_common_condition_selection_v1", "scope": "validation_only",
        "selection_rule": "min macro three-seed mean RMSE/reference ratio; then MAE ratio; then log1p/raw and log_mse/raw_mse/raw_mae order; exact comparisons",
        "fixed_reference_condition": deepcopy(REFERENCE_CONDITION), "selected_condition": deepcopy(selected["condition"]),
        "selected_scores": {key: selected[key] for key in ("macro_rmse_ratio", "macro_mae_ratio")},
        "condition_scores": scores, "input_endpoint_count": 54, "dataset_count": 3, "seed_count": 3,
        "condition_count": 6, "held_out_test_evaluated": False, "raw_errors_pooled": False,
        "interpretation": "Validation-based common training setting, not a backbone contribution or independent evaluation"}


def aggregate(records, common_condition):
    """Compare exact candidate18+stored MLP16 reference9 at one fixed setting."""
    condition = _condition(common_condition)
    indexed = {}
    for row in records:
        _validate_row(row)
        require(all(row[key] == value for key, value in condition.items()), "Common condition differs")
        key = (row["dataset"], row["seed"], _model(row))
        require(key not in indexed, "Expected exact 27-endpoint three-arm matrix")
        indexed[key] = row
    expected = {(d, s, model) for d in DATASETS for s in SEEDS for model in MODELS}
    require(len(records) == len(indexed) == 27 and set(indexed) == expected,
            "Expected exact 27-endpoint three-arm matrix")
    summaries, differences, decisions = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            for model in MODELS:
                _validate_pair(indexed[dataset, seed, BASELINE], indexed[dataset, seed, model])
        for cohort in COHORTS:
            for model in MODELS:
                group = [indexed[dataset, seed, model]["metrics"][cohort] for seed in SEEDS]
                require(len({row["count"] for row in group}) == 1, "Seed cohort counts differ")
                count = group[0]["count"]
                for metric in METRICS:
                    values = [row[metric] for row in group]
                    mean, sd = _summary(values) if count else (None, None)
                    summaries.append({"dataset": dataset, "model": model, **condition,
                        "cohort": cohort, "metric": metric, "seed_count": 3, "target_count_per_seed": count,
                        "mean": mean, "sample_sd_ddof1": sd,
                        "time_metric_diagnostic_only": metric == "time_nll"})
            for candidate, reference in CONTRASTS:
                for seed in SEEDS:
                    left, right = indexed[dataset, seed, candidate], indexed[dataset, seed, reference]
                    for metric in METRICS:
                        x, y = left["metrics"][cohort][metric], right["metrics"][cohort][metric]
                        delta = None if x is None or y is None else x - y
                        relative = None if delta is None or y == 0 else 100. * delta / abs(y)
                        require(delta is None or math.isfinite(delta), "Nonfinite paired difference")
                        require(relative is None or math.isfinite(relative), "Nonfinite paired relative difference")
                        differences.append({"dataset": dataset, "seed": seed, **condition,
                            "candidate_model": candidate, "reference_model": reference, "cohort": cohort,
                            "metric": metric, "candidate_selected_epoch": left["selected_epoch"],
                            "reference_selected_epoch": right["selected_epoch"], "candidate_value": x,
                            "reference_value": y, "absolute_difference": delta,
                            "relative_difference_percent": relative,
                            "worsened": None if delta is None else delta > 0.,
                            "time_metric_diagnostic_only": metric == "time_nll"})
        for candidate, reference in CONTRASTS:
            means = {model: {metric: statistics.mean(indexed[dataset, seed, model]["metrics"]["overall"][metric]
                for seed in SEEDS) for metric in METRICS[:2]} for model in (candidate, reference)}
            improved = {metric: means[candidate][metric] < means[reference][metric] for metric in METRICS[:2]}
            decisions.append({"dataset": dataset, "candidate_model": candidate, "reference_model": reference,
                "rmse_mean_improved": improved["qty_rmse"], "mae_mean_improved": improved["qty_mae"],
                "both_quantity_means_improved": all(improved.values()),
                "all_three_seeds_improved_both": all(indexed[dataset, seed, candidate]["metrics"]["overall"][metric]
                    < indexed[dataset, seed, reference]["metrics"]["overall"][metric]
                    for seed in SEEDS for metric in METRICS[:2]),
                "mean_differences": {metric: means[candidate][metric] - means[reference][metric] for metric in METRICS[:2]}})
    return {"schema": "titantpp_late_path_three_arm_validation_v1", "common_condition": condition,
        "summary": summaries, "seed_differences": differences, "decisions": decisions,
        "records": deepcopy(records),
        "observation_improves_both_vs_baseline_and_hidden_all_datasets": all(
            row["both_quantity_means_improved"] for row in decisions if row["candidate_model"] == OBSERVATION),
        "raw_errors_pooled": False, "held_out_test_evaluated": False}


def _write_reports(manager, result, output):
    output.mkdir(parents=True, exist_ok=False)
    manager.write(output / "THREE_ARM_VALIDATION.json", result)
    for filename, rows in (("SUMMARY.csv", result["summary"]), ("SEED_DIFFERENCES.csv", result["seed_differences"])):
        with (output / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    c = result["common_condition"]
    lines = ["# 현재 관측 직접 경로와 동일 용량 이력 경로의 Validation 비교", "",
        f"공통 조건: Cap·MLP16·시간 입력 {c['time_input']}·수량 손실 {c['loss_mode']}·시간 가중치0·seed42/52/62.",
        "기존 MLP16의 같은 조건 저장9개와 신규 두 후보18개를 같은 host·target·seed끼리 비교했다.",
        "모든 선택은 full Validation 실수량 RMSE 최초 strict finite 최소epoch(동률 첫epoch)다.",
        "평균±표본SD(ddof1). 수량 MAE도 같은 선택 epoch에서 평가했다.", "",
        "| 데이터 | 구조 | RMSE | MAE | TimeNLL(미학습 head 진단) |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in MODELS:
            values = []
            for metric in METRICS:
                row = next(row for row in result["summary"] if row["dataset"] == dataset and row["model"] == model
                           and row["cohort"] == "overall" and row["metric"] == metric)
                values.append(f"{row['mean']:.6f}±{row['sample_sd_ddof1']:.6f}")
            lines.append(f"| {dataset} | {model} | " + " | ".join(values) + " |")
    lines.extend(["", "seed별 관측−MLP16·이력−MLP16·관측−이력 절대/상대차이와 악화는 CSV/JSON에 전부 기록했다.",
        "상한 도달 집단과 상한 초과 빈 집단(N/A), 전체/집단 시간NLL을 보존했다. 데이터셋별 원단위 오류는 합치지 않았다.",
        "공통 조건 선택도 같은 Validation을 사용한 탐색적 비교이며 독립 검증이나 인과관계·통계유의성을 뜻하지 않는다.",
        "기존 CNN과 저장 MLP16은 변경·재학습·재평가하지 않았다. Test는 평가하지 않았다."])
    with (output / "THREE_ARM_VALIDATION.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    """Read verified local archives only; never contact or launch a remote job."""
    from paper.scripts import manage_titantpp_late_path18 as manager
    from paper.scripts.compare_titantpp_time_zero_mlp9 import verify_zero_proof
    contract = manager.verify_local()
    require(contract["evaluation_scope"] == "validation_only" and contract["held_out_test_authorized"] is False,
            "Wrong evaluation scope")
    condition = _condition(contract["common_condition"])
    output = manager.REPORT / "comparison_v1"
    require(not output.exists() and not (manager.BUNDLE / "retrieval/final_receipt.json").exists(),
            "Preserve existing comparison; no overwrite")
    records, bindings, archives = [], {}, {}
    scientific_exit0_count = dispatcher_exit0_count = 0
    for host in ("5080", "5090"):
        retrieval = manager.BUNDLE / "retrieval" / host
        originals = retrieval / "originals"
        terminal = manager.verify_collected(contract, host, originals)
        collected = manager.read(retrieval / "receipt.json")
        require(collected["status"] == "complete"
                and manager.sha(retrieval / "original.tar.gz") == collected["archive_sha256"]
                and manager.sha(retrieval / "manifest.json") == collected["manifest_sha256"], "Archive or manifest changed")
        inventory = manager.read(retrieval / "manifest.json")["files"]
        require(inventory and len(inventory) == collected["file_count"], "Collected inventory incomplete")
        for name, digest in inventory.items():
            require(manager.sha(manager.runtime.local_path(originals, name)) == digest, "Collected original changed")
        for filename in ("original.tar.gz", "manifest.json", "receipt.json"):
            path = retrieval / filename
            archives[str(path)] = manager.sha(path)
        require(manager.read(originals / "dispatcher_exit.json")["actual_os_returncode"] == 0,
                "Dispatcher exit was not successful")
        dispatcher_exit0_count += 1
        host_jobs = [job for job in contract["jobs"] if job["host"] == host]
        for stage in ("train", "validation"):
            exits = terminal["stages"][stage]["exits"]
            require(set(exits) == {job["id"] for job in host_jobs} and all(
                item["actual_os_returncode"] == 0 and not item["timeout"] for item in exits.values()),
                "Scientific worker exit was not successful")
            scientific_exit0_count += len(exits)
        for job in host_jobs:
            require(all(job[key] == value for key, value in condition.items()), "Admitted job common condition differs")
            path = originals / "run" / job["id"] / "validation_receipt.json"
            receipt = manager.read(path)
            require(receipt["status"] == "complete" and receipt["validation_replay_passed"] is True
                    and receipt["held_out_test_evaluated"] is False
                    and all(receipt[key] == job[key] for key in ("dataset", "seed", "loss_mode", "time_alpha")),
                    "Selected endpoint differs from admitted job")
            model = _model(job)
            require(model in (OBSERVATION, HIDDEN), "A stored baseline must not be trained")
            require(_model(receipt) == model, "Selected endpoint model differs")
            require(receipt["time_input_transform"] == ("raw_nonnegative_float32" if condition["time_input"] == "raw"
                    else "log1p_nonnegative_float32")
                    and receipt["quantity_input_transform"] == "log1p_nonnegative_float32", "Input transform differs")
            records.append({"dataset": job["dataset"], "seed": job["seed"], "host": host, "model": model,
                **condition, "time_head_trained": False, "selected_epoch": receipt["selected_epoch"],
                "target_identity": receipt["target_identity"], "metrics": receipt["metrics"],
                "scope": "validation_only", "held_out_test_evaluated": False,
                "time_head_no_training_verification": verify_zero_proof(receipt)})
            bindings[str(path)] = manager.sha(path)
    references = manager.baseline_validation_references()
    require(references == contract["baseline_validation_references"] and len(references) == 9,
            "Stored selected-condition MLP16 references changed")
    for reference in references:
        require(reference["loss_mode"] == condition["loss_mode"] and reference["time_alpha"] == 0.,
                "Stored MLP16 loss/time objective differs")
        reference_transform = reference.get("time_input")
        if reference_transform is None:
            reference_transform = {"raw_nonnegative_float32": "raw", "log1p_nonnegative_float32": "log1p"}.get(
                reference.get("time_input_transform"))
        require(reference_transform == condition["time_input"], "Stored MLP16 time input differs")
        path = Path(reference["path"])
        require(manager.sha(path) == reference["sha256"], "Stored MLP16 endpoint changed")
        receipt = manager.read(path)
        require(receipt["metrics"] == reference["metrics"] and receipt["target_identity"] == reference["target_identity"],
                "Stored endpoint mismatch")
        records.append({**reference, "model": BASELINE, **condition, "time_head_trained": False,
            "scope": "validation_only", "held_out_test_evaluated": False,
            "time_head_no_training_verification": verify_zero_proof(receipt)})
        bindings[str(path)] = reference["sha256"]
    require(scientific_exit0_count == 36 and dispatcher_exit0_count == 2, "Completion counts differ")
    result = aggregate(records, condition)
    result.update(status="complete", scope="validation_only", created_at=datetime.now(timezone.utc).isoformat(),
        records=records, source_bindings=bindings, archive_bindings=archives, new_fit_count=18,
        validation_replay_count=18, baseline_reference_count=9, comparison_model_count=27, seed_count=3,
        time_alpha=0., held_out_test_evaluated=False, historical_cnn_changed=False, baseline_reevaluated=False,
        interpretation_limits=["Same Validation selected the common setting and checkpoints; this is exploratory.",
            "Observation versus hidden-path comparison separates input origin at matched added capacity, not every optimization effect.",
            "No pooled raw errors across datasets, Test improvement, independent evaluation, causal or significance claim.",
            "Time-only head is untrained; recorded positive-integer TimeNLL is diagnostic."])
    if "common_condition_selection" in contract:
        result["common_condition_selection"] = deepcopy(contract["common_condition_selection"])
    _write_reports(manager, result, output)
    manager.write(manager.BUNDLE / "retrieval/final_receipt.json", {
        "status": "complete", "new_fit_count": 18, "validation_replay_count": 18,
        "baseline_reference_count": 9, "comparison_model_count": 27, "seed_count": 3, "time_alpha": 0.,
        "common_condition": condition, "time_head_no_training_verified_count": 18,
        "actual_scientific_child_exit0_count": scientific_exit0_count,
        "actual_scientific_worker_exit0_count": scientific_exit0_count,
        "actual_dispatcher_exit0_count": dispatcher_exit0_count, "held_out_test_evaluated": False,
        "historical_cnn_retrained": False, "historical_cnn_reevaluated": False, "baseline_reevaluated": False,
        "source_bindings": bindings, "archive_bindings": archives,
        "report_files": {str(path): manager.sha(path) for path in output.iterdir() if path.is_file()}})
    return result


if __name__ == "__main__":
    main()
