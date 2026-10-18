"""Aggregate completed time-weight fits and nine stored alpha1 Validation results."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2] if "paper/scripts" in str(Path(__file__)) else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NAME = "titantpp_time_weight_mlp18_20261018_v1"
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
ALPHAS, SEEDS = (.1, 1., 10.), (42, 52, 62)
METRICS, COHORTS = ("qty_rmse", "qty_mae", "time_nll"), ("overall", "cap_hit", "tail")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(records):
    indexed = {(r["dataset"], r["seed"], float(r["time_alpha"])): r for r in records}
    expected = {(d, s, a) for d in DATASETS for s in SEEDS for a in ALPHAS}
    require(len(records) == len(indexed) == 27 and set(indexed) == expected, "Expected exact18 new plus9 stored conditions")
    summaries, differences, decisions = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            base = indexed[(dataset, seed, 1.)]
            for alpha in ALPHAS:
                r = indexed[(dataset, seed, alpha)]
                require(r["target_identity"] == base["target_identity"], "Validation target identities differ")
                require(r["host"] == base["host"], "Dataset/seed Runtime host differs from its stored reference")
        for cohort in COHORTS:
            for alpha in ALPHAS:
                group = [indexed[(dataset, seed, alpha)]["metrics"][cohort] for seed in SEEDS]
                require(len({r["count"] for r in group}) == 1, "Cohort counts differ")
                n = int(group[0]["count"])
                for metric in METRICS:
                    vals = [r[metric] for r in group]
                    if n == 0:
                        require(all(v is None for v in vals), "Empty cohort must have N/A metrics")
                        mean = sd = None
                    else:
                        require(all(v is not None and math.isfinite(v) for v in vals), "Nonfinite metric")
                        mean, sd = statistics.mean(vals), statistics.stdev(vals)
                    summaries.append({"dataset": dataset, "cohort": cohort, "time_alpha": alpha,
                        "metric": metric, "seed_count": 3, "target_count_per_seed": n,
                        "mean": mean, "sample_sd_ddof1": sd})
            for seed in SEEDS:
                for left, right in ((.1, 1.), (10., 1.), (.1, 10.)):
                    a, b = indexed[(dataset, seed, left)]["metrics"][cohort], indexed[(dataset, seed, right)]["metrics"][cohort]
                    for metric in METRICS:
                        x, y = a[metric], b[metric]
                        delta = None if x is None or y is None else x-y
                        relative = None if delta is None or y == 0 else 100.*delta/abs(y)
                        differences.append({"dataset": dataset, "seed": seed, "cohort": cohort,
                            "metric": metric, "left_time_alpha": left, "right_time_alpha": right,
                            "left_value": x, "right_value": y, "absolute_difference": delta,
                            "relative_difference_percent_abs_reference_denominator": relative,
                            "worsened": None if delta is None else delta > 0})
        def avg(alpha, metric):
            return next(r["mean"] for r in summaries if r["dataset"] == dataset and r["cohort"] == "overall" and r["time_alpha"] == alpha and r["metric"] == metric)
        for alpha in (.1, 10.):
            quantity_better = all(avg(alpha, m) < avg(1., m) for m in ("qty_rmse", "qty_mae"))
            time_not_worse = avg(alpha, "time_nll") <= avg(1., "time_nll")
            decisions.append({"dataset": dataset, "time_alpha": alpha,
                "quantity_both_means_improved": quantity_better, "time_mean_not_worse": time_not_worse,
                "three_metric_pareto_improvement": quantity_better and time_not_worse,
                "quantity_gain_with_time_tradeoff": quantity_better and not time_not_worse})
    return {"summary": summaries, "seed_differences": differences, "decisions": decisions}


def main():
    from paper.scripts import manage_titantpp_time_weight_mlp18 as manager
    contract = manager.verify_local()
    require(contract["evaluation_scope"] == "validation_only" and not contract["held_out_test_authorized"], "Wrong evaluation scope")
    report = ROOT / "reports" / NAME
    output = report / "comparison_v1"
    require(not output.exists(), "Existing comparison; preserve originals")
    records, bindings = [], {}
    for host in ("5080", "5090"):
        originals = manager.BUNDLE / "retrieval" / host / "originals"
        manager.verify_collected(contract, host, originals)
        for job in [j for j in contract["jobs"] if j["host"] == host]:
            path = originals / "run" / job["id"] / "validation_receipt.json"
            r = manager.read(path)
            require(r["time_alpha"] == job["time_alpha"] and not r["held_out_test_evaluated"] and r["validation_replay_passed"], "New Validation receipt mismatch")
            records.append({"dataset": r["dataset"], "host": host, "seed": r["seed"], "time_alpha": r["time_alpha"],
                "metrics": r["metrics"], "target_identity": r["target_identity"], "selected_epoch": r["selected_epoch"]})
            bindings[str(path)] = sha(path)
    # Reverify references without re-evaluating the existing models.
    fresh_refs = manager.baseline_validation_references()
    require(fresh_refs == contract["baseline_validation_references"], "Stored references changed since sealing")
    for r in fresh_refs:
        records.append(r)
        bindings[r["path"]] = r["sha256"]
    result = aggregate(records)
    output.mkdir(parents=True, exist_ok=False)
    result.update(status="complete", scope="validation_only", records=records,
        held_out_test_evaluated=False, historical_cnn_retrained=False, historical_cnn_reevaluated=False,
        baseline_reevaluated=False, candidate_fit_count=18, baseline_reference_count=9, seed_count=3,
        contract_sha256=manager.sha(manager.BUNDLE/"execution_contract.json"), inputs=bindings,
        created_at=datetime.now(timezone.utc).isoformat(),
        interpretation_limits=["Time-weight sensitivity, not backbone contribution or a proven optimal coefficient.",
            "Report all seeds and time tradeoffs. Different dataset raw errors are not pooled.",
            "Weighted joint losses across different coefficients do not define a model ranking.",
            "Relative differences use abs(reference) as denominator; zero or empty reference gives N/A."])
    path = output / "THREE_SEED_VALIDATION.json"
    manager.write(path, result)
    for name, rows in (("SUMMARY.csv", result["summary"]), ("SEED_DIFFERENCES.csv", result["seed_differences"])):
        with (output/name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    lines = ["# MLP16 시간 가중치 Validation 비교", "", "각 모델의 기존 선택 checkpoint를 그대로 사용했다. 수치는3seed 평균±표본SD이며 가중치1은 저장 결과 재사용이다.", "", "| 데이터 | 시간 가중치 | RMSE | MAE | TimeNLL |", "|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for alpha in ALPHAS:
            values=[]
            for m in METRICS:
                r=next(x for x in result["summary"] if x["dataset"]==dataset and x["time_alpha"]==alpha and x["cohort"]=="overall" and x["metric"]==m)
                values.append(f"{r['mean']:.6f}±{r['sample_sd_ddof1']:.6f}")
            lines.append(f"| {dataset} | {alpha:g} | "+" | ".join(values)+" |")
    lines += ["", "수량 두 지표 개선과 시간 지표를 포함한 Pareto 개선을 별도로 기록했다. seed별 악화·q==cap·빈 q>cap 집단은 JSON/CSV에 모두 보존했다. 새 Test 평가·참조 모델 재학습·재평가는 없다."]
    (output/"THREE_SEED_VALIDATION.md").write_text("\n".join(lines)+"\n")
    manager.write(manager.BUNDLE/"retrieval/final_receipt.json", {"status":"complete", "candidate_fit_count":18,
        "validation_replay_count":18, "baseline_reference_count":9, "comparison_model_count":27, "seed_count":3,
        "held_out_test_evaluated":False, "historical_cnn_reevaluated":False, "baseline_reevaluated":False,
        "report_files":{str(p):sha(p) for p in output.iterdir() if p.is_file()}, "input_files":bindings})
    print(json.dumps({"status":"complete","candidate_fit_count":18,"baseline_reference_count":9,"output":str(output)}))


if __name__ == "__main__":
    main()
