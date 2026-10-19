"""Paired Validation comparison: raw elapsed-time inputs versus stored log-time.

The comparison never runs inference. Each of the 27 pairs keeps its quantity
loss, dataset, seed, native host and target population. TimeNLL is diagnostic.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
LOSSES = ("log_mse", "raw_mse", "raw_mae")
TRANSFORMS = ("log1p", "raw")
COHORTS = ("overall", "cap_hit", "tail")
METRICS = ("qty_rmse", "qty_mae", "time_nll")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def aggregate(records):
    keys = ("dataset", "seed", "loss_mode", "time_input")
    indexed = {tuple(row[k] for k in keys): row for row in records}
    expected = {(d, s, loss, transform) for d in DATASETS for s in SEEDS
                for loss in LOSSES for transform in TRANSFORMS}
    require(len(records) == len(indexed) == 54 and set(indexed) == expected,
            "Expected exact stored log-time27 and new raw-time27")
    summaries, differences, decisions = [], [], []
    for dataset in DATASETS:
        for loss in LOSSES:
            for seed in SEEDS:
                baseline = indexed[dataset, seed, loss, "log1p"]
                candidate = indexed[dataset, seed, loss, "raw"]
                require(candidate["host"] == baseline["host"]
                        and candidate["target_identity"] == baseline["target_identity"],
                        "Paired target/native host mismatch")
                for row in (baseline, candidate):
                    require(row["time_alpha"] == 0. and row["time_head_trained"] is False,
                            "Time objective must remain excluded")
                    for cohort in COHORTS:
                        require(row["metrics"][cohort]["count"] == baseline["metrics"][cohort]["count"],
                                "Paired cohort counts differ")
            for cohort in COHORTS:
                for transform in TRANSFORMS:
                    group = [indexed[dataset, seed, loss, transform]["metrics"][cohort] for seed in SEEDS]
                    require(len({row["count"] for row in group}) == 1, "Seed cohort counts differ")
                    count = group[0]["count"]
                    require(type(count) is int and count >= 0, "Invalid cohort count")
                    for metric in METRICS:
                        values = [row[metric] for row in group]
                        if count == 0:
                            require(all(value is None for value in values), "Empty cohort must be N/A")
                            mean = sd = None
                        else:
                            require(all(type(value) in (int, float) and math.isfinite(value)
                                        for value in values), "Invalid/nonfinite metric")
                            require(metric == "time_nll" or all(value >= 0 for value in values),
                                    "Negative quantity error")
                            mean, sd = statistics.mean(values), statistics.stdev(values)
                        summaries.append({"dataset": dataset, "loss_mode": loss, "time_input": transform,
                            "cohort": cohort, "metric": metric, "seed_count": 3,
                            "target_count_per_seed": count, "mean": mean, "sample_sd_ddof1": sd,
                            "time_metric_diagnostic_only": metric == "time_nll"})
                for seed in SEEDS:
                    raw = indexed[dataset, seed, loss, "raw"]["metrics"][cohort]
                    log = indexed[dataset, seed, loss, "log1p"]["metrics"][cohort]
                    for metric in METRICS:
                        x, y = raw[metric], log[metric]
                        delta = None if x is None or y is None else x - y
                        relative = None if delta is None or y == 0 else 100. * delta / abs(y)
                        differences.append({"dataset": dataset, "loss_mode": loss, "seed": seed,
                            "cohort": cohort, "metric": metric, "raw_time_value": x,
                            "log_time_value": y, "absolute_difference": delta,
                            "relative_difference_percent": relative,
                            "worsened": None if delta is None else delta > 0.,
                            "time_metric_diagnostic_only": metric == "time_nll"})
            def average(transform, metric):
                return next(row["mean"] for row in summaries if row["dataset"] == dataset
                            and row["loss_mode"] == loss and row["cohort"] == "overall"
                            and row["time_input"] == transform and row["metric"] == metric)
            improvements = {metric: average("raw", metric) < average("log1p", metric)
                            for metric in METRICS[:2]}
            decisions.append({"dataset": dataset, "loss_mode": loss,
                "rmse_mean_improved": improvements["qty_rmse"],
                "mae_mean_improved": improvements["qty_mae"],
                "both_quantity_means_improved": all(improvements.values()),
                "all_three_seeds_improved_both": all(
                    indexed[dataset, seed, loss, "raw"]["metrics"]["overall"][metric]
                    < indexed[dataset, seed, loss, "log1p"]["metrics"]["overall"][metric]
                    for seed in SEEDS for metric in METRICS[:2])})
    return {"summary": summaries, "seed_differences": differences, "decisions": decisions,
            "common_improvement_across_datasets_and_losses":
                all(row["both_quantity_means_improved"] for row in decisions)}


def main():
    from paper.scripts import manage_titantpp_raw_time_mlp27 as manager
    from paper.scripts.compare_titantpp_time_zero_mlp9 import verify_zero_proof
    contract = manager.verify_local()
    require(contract["evaluation_scope"] == "validation_only"
            and contract["held_out_test_authorized"] is False, "Wrong evaluation scope")
    output = manager.REPORT / "comparison_v1"
    require(not output.exists(), "Preserve existing comparison; no overwrite")
    records, bindings, archives = [], {}, {}
    for host in ("5080", "5090"):
        retrieval = manager.BUNDLE / "retrieval" / host
        originals = retrieval / "originals"
        manager.verify_collected(contract, host, originals)
        collected = manager.read(retrieval / "receipt.json")
        require(collected["status"] == "complete"
                and manager.sha(retrieval / "original.tar.gz") == collected["archive_sha256"]
                and manager.sha(retrieval / "manifest.json") == collected["manifest_sha256"],
                "Raw-time archive or manifest changed")
        inventory = manager.read(retrieval / "manifest.json")["files"]
        require(inventory and len(inventory) == collected["file_count"], "Raw-time inventory incomplete")
        for name, digest in inventory.items():
            require(manager.sha(manager.runtime.local_path(originals, name)) == digest,
                    "Raw-time original changed")
        for filename in ("original.tar.gz", "manifest.json", "receipt.json"):
            path = retrieval / filename
            archives[str(path)] = manager.sha(path)
        for job in (j for j in contract["jobs"] if j["host"] == host):
            path = originals / "run" / job["id"] / "validation_receipt.json"
            receipt = manager.read(path)
            require(receipt["status"] == "complete" and receipt["validation_replay_passed"] is True
                    and receipt["held_out_test_evaluated"] is False
                    and all(receipt[k] == job[k] for k in ("dataset", "seed", "loss_mode", "time_alpha"))
                    and receipt["time_input_transform"] == "raw_nonnegative_float32"
                    and receipt["quantity_input_transform"] == "log1p_nonnegative_float32",
                    "Raw-time selected endpoint differs from admitted job")
            proof = verify_zero_proof(receipt)
            records.append({"dataset": job["dataset"], "seed": job["seed"], "host": host,
                "loss_mode": job["loss_mode"], "time_input": "raw", "time_alpha": 0.,
                "time_head_trained": False, "selected_epoch": receipt["selected_epoch"],
                "target_identity": receipt["target_identity"], "metrics": receipt["metrics"],
                "time_head_no_training_verification": proof})
            bindings[str(path)] = manager.sha(path)
    references = manager.baseline_validation_references()
    require(references == contract["baseline_validation_references"], "Stored log-time27 changed")
    for reference in references:
        path = Path(reference["path"])
        require(manager.sha(path) == reference["sha256"], "Stored log-time endpoint changed")
        receipt = manager.read(path)
        require(receipt["metrics"] == reference["metrics"]
                and receipt["target_identity"] == reference["target_identity"], "Stored endpoint mismatch")
        records.append({**reference, "time_input": "log1p", "time_head_trained": False,
                        "time_head_no_training_verification": verify_zero_proof(receipt)})
        bindings[str(path)] = reference["sha256"]
    result = aggregate(records)
    result.update(status="complete", scope="validation_only",
        created_at=datetime.now(timezone.utc).isoformat(), records=records,
        source_bindings=bindings, archive_bindings=archives, new_fit_count=27,
        validation_replay_count=27, stored_log_time_reference_count=27,
        comparison_model_count=54, seed_count=3, time_alpha=0., held_out_test_evaluated=False,
        historical_cnn_changed=False, baseline_reevaluated=False,
        interpretation_limits=[
            "Only the observed elapsed-time input changes from log1p(dt) to raw dt; quantity input remains log1p(q).",
            "No new input normalization, scaling, clipping, head, memory, optimizer, loss or training budget change.",
            "Compare matching dataset/loss/seed/host endpoints; never compare different loss winners as input effects.",
            "Raw-input scale and its optimization/clipping effects are part of the intervention.",
            "The time head is untrained by time loss; TimeNLL is diagnostic, not physical-time MAE or RMSE.",
            "All selected checkpoints minimize full Validation raw quantity RMSE, including MAE-trained fits.",
            "No Test, pooled errors across datasets, causal attribution, or backbone superiority claim."])
    output.mkdir(parents=True, exist_ok=False)
    manager.write(output / "PAIRED_TIME_INPUT_VALIDATION.json", result)
    for name, rows in (("SUMMARY.csv", result["summary"]), ("SEED_DIFFERENCES.csv", result["seed_differences"])):
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 원시간 입력과 로그 시간 입력의 같은 조건 Validation 비교", "",
        "Cap·MLP16·시간 가중치0·seed42/52/62. 수량 입력은 log1p로 유지했다.",
        "손실별로 저장 로그 시간27개와 신규 원시간27개를 같은 host·target·seed끼리 비교한다.",
        "평균±표본SD(ddof1), 선택은 전체 Validation 실수량 RMSE 최초 strict finite 최소epoch다.", "",
        "| 데이터 | 수량 손실 | 시간 입력 | RMSE | MAE | TimeNLL(미학습 head 진단) |",
        "|---|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for loss in LOSSES:
            for transform in TRANSFORMS:
                values = []
                for metric in METRICS:
                    row = next(r for r in result["summary"] if r["dataset"] == dataset and r["loss_mode"] == loss
                        and r["time_input"] == transform and r["cohort"] == "overall" and r["metric"] == metric)
                    values.append(f"{row['mean']:.6f}±{row['sample_sd_ddof1']:.6f}")
                lines.append(f"| {dataset} | {loss} | {transform} | " + " | ".join(values) + " |")
    lines.extend(["", "seed별 원시간−로그 시간 절대·상대차이와 악화·예외는 JSON/CSV에 전부 기록했다.",
        "상한 도달 집단과 상한 초과 빈 집단(N/A), 시간NLL도 보존했다. 데이터셋별 원단위 오류는 합치지 않았다.",
        "입력 변환 및 그 스케일·최적화 영향의 비교이며 backbone 구조 개선이나 Test 개선을 뜻하지 않는다."])
    with (output / "PAIRED_TIME_INPUT_VALIDATION.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    manager.write(manager.BUNDLE / "retrieval/final_receipt.json", {
        "status": "complete", "new_fit_count": 27, "validation_replay_count": 27,
        "baseline_reference_count": 27, "comparison_model_count": 54, "seed_count": 3,
        "time_alpha": 0., "time_head_no_training_verified_count": 27,
        "actual_scientific_child_exit0_count": 54, "actual_dispatcher_exit0_count": 2,
        "held_out_test_evaluated": False, "historical_cnn_reevaluated": False,
        "baseline_reevaluated": False, "source_bindings": bindings, "archive_bindings": archives,
        "report_files": {str(path): manager.sha(path) for path in output.iterdir() if path.is_file()}})
    return result


if __name__ == "__main__":
    main()
