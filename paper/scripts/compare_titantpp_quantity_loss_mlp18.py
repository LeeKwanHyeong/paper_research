"""Validation-only comparison of literal quantity losses at time weight zero."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
LOSSES = ("log_mse", "raw_mse", "raw_mae")
METRICS = ("qty_rmse", "qty_mae", "time_nll")
COHORTS = ("overall", "cap_hit", "tail")
PAIRS = (("raw_mse", "log_mse"), ("raw_mae", "log_mse"), ("raw_mae", "raw_mse"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def aggregate(records):
    indexed = {(r["dataset"], r["seed"], r["loss_mode"]): r for r in records}
    expected = {(d, s, loss) for d in DATASETS for s in SEEDS for loss in LOSSES}
    require(len(records) == len(indexed) == 27 and set(indexed) == expected,
            "Expected exact18 new plus9 stored quantity-only fits")
    summaries, differences, decisions = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            baseline = indexed[dataset, seed, "log_mse"]
            for loss in LOSSES:
                row = indexed[dataset, seed, loss]
                require(row["time_alpha"] == 0. and row["time_head_trained"] is False,
                        "Only quantity-only time-weight0 endpoints are comparable")
                require(row["target_identity"] == baseline["target_identity"]
                        and row["host"] == baseline["host"], "Target or native host differs")
                for cohort in COHORTS:
                    require(row["metrics"][cohort]["count"] == baseline["metrics"][cohort]["count"],
                            "Cohort counts differ across losses")
        for cohort in COHORTS:
            for loss in LOSSES:
                group = [indexed[dataset, seed, loss]["metrics"][cohort] for seed in SEEDS]
                require(len({r["count"] for r in group}) == 1, "Cohort counts differ across seeds")
                count = group[0]["count"]
                require(type(count) is int and count >= 0, "Invalid cohort count")
                for metric in METRICS:
                    values = [r[metric] for r in group]
                    if count == 0:
                        require(all(v is None for v in values), "Empty cohort must be N/A")
                        mean = sd = None
                    else:
                        require(all(type(v) in (int, float) and math.isfinite(v) for v in values),
                                "Nonfinite or invalid metric")
                        if metric != "time_nll":
                            require(all(v >= 0. for v in values), "Negative quantity error")
                        mean, sd = statistics.mean(values), statistics.stdev(values)
                    summaries.append({"dataset": dataset, "cohort": cohort, "loss_mode": loss,
                        "metric": metric, "seed_count": 3, "target_count_per_seed": count,
                        "mean": mean, "sample_sd_ddof1": sd,
                        "time_metric_diagnostic_only": metric == "time_nll"})
            for seed in SEEDS:
                for left, right in PAIRS:
                    lhs = indexed[dataset, seed, left]["metrics"][cohort]
                    rhs = indexed[dataset, seed, right]["metrics"][cohort]
                    for metric in METRICS:
                        x, y = lhs[metric], rhs[metric]
                        delta = None if x is None or y is None else x - y
                        relative = None if delta is None or y == 0. else 100. * delta / abs(y)
                        differences.append({"dataset": dataset, "seed": seed, "cohort": cohort,
                            "metric": metric, "left_loss_mode": left, "right_loss_mode": right,
                            "left_value": x, "right_value": y, "absolute_difference": delta,
                            "relative_difference_percent": relative,
                            "worsened": None if delta is None else delta > 0.,
                            "time_metric_diagnostic_only": metric == "time_nll"})
        def avg(loss, metric):
            return next(r["mean"] for r in summaries if r["dataset"] == dataset
                        and r["cohort"] == "overall" and r["loss_mode"] == loss and r["metric"] == metric)
        for loss in LOSSES[1:]:
            decisions.append({"dataset": dataset, "loss_mode": loss,
                "quantity_both_means_improved_vs_log": all(avg(loss, m) < avg("log_mse", m)
                                                            for m in METRICS[:2]),
                "rmse_mean_improved_vs_log": avg(loss, "qty_rmse") < avg("log_mse", "qty_rmse"),
                "mae_mean_improved_vs_log": avg(loss, "qty_mae") < avg("log_mse", "qty_mae"),
                "time_metric_diagnostic_only": True})
    return {"summary": summaries, "seed_differences": differences, "decisions": decisions}


def main():
    from paper.scripts import manage_titantpp_quantity_loss_mlp18 as manager
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
        for filename in ("original.tar.gz", "manifest.json", "receipt.json"):
            path = retrieval / filename
            archives[str(path)] = manager.sha(path)
        for job in [j for j in contract["jobs"] if j["host"] == host]:
            path = originals / "run" / job["id"] / "validation_receipt.json"
            receipt = manager.read(path)
            require(receipt["status"] == "complete" and receipt["validation_replay_passed"] is True
                    and receipt["held_out_test_evaluated"] is False and receipt["time_alpha"] == 0.
                    and receipt["loss_mode"] == job["loss_mode"]
                    and receipt["dataset"] == job["dataset"] and receipt["seed"] == job["seed"],
                    "Selected Validation endpoint differs from the admitted job")
            proof = verify_zero_proof(receipt)
            records.append({"dataset": receipt["dataset"], "host": host, "seed": receipt["seed"],
                "loss_mode": receipt["loss_mode"], "time_alpha": 0., "selected_epoch": receipt["selected_epoch"],
                "target_identity": receipt["target_identity"], "metrics": receipt["metrics"],
                "time_head_trained": False, "time_head_no_training_verification": proof})
            bindings[str(path)] = manager.sha(path)
    references = manager.baseline_validation_references()
    require(references == contract["baseline_validation_references"], "Stored log0 references changed")
    for reference in references:
        path = Path(reference["path"])
        require(manager.sha(path) == reference["sha256"], "Stored log0 endpoint changed")
        receipt = manager.read(path)
        proof = verify_zero_proof(receipt)
        require(reference["time_alpha"] == 0. and reference["loss_mode"] == "log_mse"
                and receipt["metrics"] == reference["metrics"]
                and receipt["target_identity"] == reference["target_identity"], "Log0 reference mismatch")
        records.append({**reference, "time_head_trained": False, "time_head_no_training_verification": proof})
        bindings[str(path)] = reference["sha256"]
    result = aggregate(records)
    result.update(status="complete", scope="validation_only", created_at=datetime.now(timezone.utc).isoformat(),
        records=records, source_bindings=bindings, archive_bindings=archives,
        new_fit_count=18, validation_replay_count=18, stored_log0_reference_count=9,
        comparison_model_count=27, seed_count=3, time_alpha=0.,
        held_out_test_evaluated=False, historical_cnn_changed=False, baseline_reevaluated=False,
        interpretation_limits=[
            "Only the quantity objective changes; log1p inputs and existing log-domain quantity head are retained.",
            "Raw MSE and MAE use literal quantity units, without cap normalization or clipping predictions to cap.",
            "Different objective scale and gradient clipping are included; this is not a backbone contribution or isolated curvature effect.",
            "Time inputs remain; time-exclusive heads are untrained by time loss and their TimeNLL is diagnostic.",
            "All endpoints use first strict finite minimum full Validation raw quantity RMSE, including MAE-trained fits.",
            "No pooling of different dataset raw errors, no Test and no external-model superiority claim."])
    output.mkdir(parents=True, exist_ok=False)
    manager.write(output / "THREE_LOSS_VALIDATION.json", result)
    for name, rows in (("SUMMARY.csv", result["summary"]), ("SEED_DIFFERENCES.csv", result["seed_differences"])):
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 시간 가중치0의 수량 손실 비교", "",
        "Cap·MLP16·3seed 평균±표본SD(ddof1). 기존 log 손실9개를 재사용하고 실수량 MSE·MAE18개를 새로 학습했다.",
        "모두 전체 Validation 실수량 RMSE로 선택한 같은 checkpoint의 지표다. 시간NLL은 미학습 시간 head의 진단값이다.", "",
        "| 데이터 | 수량 손실 | RMSE | MAE | TimeNLL(진단) |", "|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for loss in LOSSES:
            values = []
            for metric in METRICS:
                row = next(r for r in result["summary"] if r["dataset"] == dataset and r["cohort"] == "overall"
                           and r["loss_mode"] == loss and r["metric"] == metric)
                values.append(f"{row['mean']:.6f}±{row['sample_sd_ddof1']:.6f}")
            lines.append(f"| {dataset} | {loss} | " + " | ".join(values) + " |")
    lines += ["", "같은 seed의 rawMSE−logMSE·rawMAE−logMSE·rawMAE−rawMSE 차이, 악화·예외와 상한도달 집단을 JSON/CSV에 모두 보존했다.",
              "상한초과 집단은 빈 집단/N/A다. 손실 스케일과 clipping 영향이 함께 바뀌므로 백본 개선이나 원인 확정으로 해석하지 않는다. 새 Test는 없다."]
    (output / "THREE_LOSS_VALIDATION.md").write_text("\n".join(lines) + "\n")
    manager.write(manager.BUNDLE / "retrieval/final_receipt.json", {"status": "complete",
        "new_fit_count": 18, "validation_replay_count": 18, "baseline_reference_count": 9,
        "comparison_model_count": 27, "seed_count": 3, "time_alpha": 0.,
        "time_head_no_training_verified_count": 18, "actual_scientific_child_exit0_count": 36,
        "actual_dispatcher_exit0_count": 2, "held_out_test_evaluated": False,
        "historical_cnn_reevaluated": False, "baseline_reevaluated": False,
        "source_bindings": bindings, "archive_bindings": archives,
        "report_files": {str(p): manager.sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"status": "complete", "comparison_model_count": 27, "output": str(output)}))


if __name__ == "__main__":
    main()
