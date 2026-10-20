"""Compare completed original-data shared15 Validation9 to stored MLP16 only."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

BASELINE = "titantpp_history_mlp_width16"
SHARED = "titantpp_history_mlp_shared15_v1"
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
THRESHOLDS = dict(zip(DATASETS, (200., 3449., 187.)))
SEEDS, MODELS = (42, 52, 62), (BASELINE, SHARED)
COHORTS, METRICS = ("overall", "tail"), ("qty_rmse", "qty_mae", "time_nll")
CONDITION = {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0}


def require(value, message):
    if not value:
        raise ValueError(message)


def aggregate(records):
    indexed = {}
    for row in records:
        d, s, m = row["dataset"], row["seed"], row["model"]
        require(d in DATASETS and type(s) is int and s in SEEDS and m in MODELS, "Unknown endpoint")
        require(row.get("data_variant") == "original_uncapped" and row.get("held_out_test_evaluated") is False
                and all(row.get(k) == v for k, v in CONDITION.items()), "Cap or different loss/split endpoint")
        require(type(row["selected_epoch"]) is int and row["selected_epoch"] > 0, "Invalid selected epoch")
        require((d, s, m) not in indexed, "Duplicate endpoint")
        indexed[d, s, m] = row
        target = row["target_identity"]
        require(type(target["target_count"]) is int and target["target_count"] > 0
                and all(isinstance(target[k], str) and len(target[k]) == 64
                        for k in ("target_identity_sha256", "target_quantity_sha256")), "Missing target binding")
        for cohort in COHORTS:
            values = row["metrics"][cohort]
            require(type(values["count"]) is int and values["count"] >= 0, "Invalid cohort count")
            require(all(v is None if values["count"] == 0 else type(v) in (int, float) and math.isfinite(v)
                        and (metric == "time_nll" or v >= 0)
                        for metric in METRICS for v in [values[metric]]), "Invalid metric")
        require(row["metrics"]["overall"]["count"] == target["target_count"]
                and row["metrics"]["tail"]["count"] <= target["target_count"], "Cohort population differs")
    require(set(indexed) == {(d, s, m) for d in DATASETS for s in SEEDS for m in MODELS}, "Expected raw candidate9 and baseline9")
    summary, differences, decisions = [], [], []
    for d in DATASETS:
        for s in SEEDS:
            candidate, baseline = indexed[d, s, SHARED], indexed[d, s, BASELINE]
            require(candidate["target_identity"] == baseline["target_identity"], "Paired targets differ")
            require(candidate["target_identity"] == indexed[d, SEEDS[0], BASELINE]["target_identity"], "Seed targets differ")
            require(all(candidate["metrics"][g]["count"] == baseline["metrics"][g]["count"] for g in COHORTS), "Paired cohorts differ")
            for g in COHORTS:
                for metric in METRICS:
                    left, right = candidate["metrics"][g][metric], baseline["metrics"][g][metric]
                    delta = None if left is None else left - right
                    differences.append({"dataset": d, "seed": s, "candidate": SHARED, "reference": BASELINE,
                        "cohort": g, "metric": metric, "absolute_difference": delta,
                        "relative_difference_percent": None if delta is None or right == 0 else delta / abs(right) * 100,
                        "worsened": None if delta is None else delta > 0,
                        "candidate_value": left, "reference_value": right,
                        "candidate_host": candidate["host"], "reference_host": baseline["host"]})
        for model in MODELS:
            for g in COHORTS:
                group = [indexed[d, s, model]["metrics"][g] for s in SEEDS]
                require(len({v["count"] for v in group}) == 1, "Cohort count differs across seeds")
                for metric in METRICS:
                    values = [v[metric] for v in group]
                    summary.append({"dataset": d, "model": model, "cohort": g, "metric": metric,
                        "seed_count": 3, "target_count_per_seed": group[0]["count"],
                        "mean": statistics.mean(values) if group[0]["count"] else None,
                        "sample_sd_ddof1": statistics.stdev(values) if group[0]["count"] else None})
        delta = {metric: statistics.mean(indexed[d, s, SHARED]["metrics"]["overall"][metric]
                        - indexed[d, s, BASELINE]["metrics"]["overall"][metric] for s in SEEDS) for metric in METRICS[:2]}
        decisions.append({"dataset": d, "mean_differences": delta,
            "both_quantity_means_improved": all(v < 0 for v in delta.values()),
            "all_three_seeds_improved_both": all(indexed[d, s, SHARED]["metrics"]["overall"][metric]
                < indexed[d, s, BASELINE]["metrics"]["overall"][metric] for s in SEEDS for metric in METRICS[:2])})
    return {"schema": "titantpp_shared_history_raw9_three_seed_validation_v1", "scope": "validation_only",
        "common_condition": CONDITION, "data_variant": "original_uncapped", "quantity_capping_applied": False,
        "records": deepcopy(records), "summary": summary, "seed_differences": differences, "decisions": decisions,
        "all_datasets_both_quantity_means_improved": all(d["both_quantity_means_improved"] for d in decisions),
        "held_out_test_evaluated": False, "raw_errors_pooled": False,
        "interpretation_limits": ["Validation selects checkpoints; not independent evaluation or significance.",
            "Original baseline Validation contains CPU/CUDA cohorts; new candidate uses5080/cuDNN91900.",
            "Capalpha0 and originalalpha1 differ by data and objective; do not attribute their metric difference to backbone.",
            "Existing original MLP16 is a stored reference, not a new fit or reevaluation."]}


def markdown(result):
    lines = ["# 원본 공유형 MLP와 기존 MLP16의 3seed Validation", "",
        "상한 적용 전 원본·log1p 입력·log-MSE·시간 가중치1. 평균±표본SD(ddof1).",
        "선택은 전체 Validation 실수량 RMSE 최초 strict finite 최소epoch이며 MAE도 같은 checkpoint 값이다.", "",
        "| 데이터 | 모델 | RMSE | MAE | 큰수량 RMSE | 큰수량 MAE | 전체 TimeNLL | 큰수량 TimeNLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        for m in MODELS:
            cells = []
            for g, metric in (("overall", "qty_rmse"), ("overall", "qty_mae"), ("tail", "qty_rmse"),
                              ("tail", "qty_mae"), ("overall", "time_nll"), ("tail", "time_nll")):
                x = next(v for v in result["summary"] if (v["dataset"], v["model"], v["cohort"], v["metric"]) == (d, m, g, metric))
                cells.append("N/A" if x["mean"] is None else f"{x['mean']:.6f} ± {x['sample_sd_ddof1']:.6f}")
            lines.append(f"| {d} | {m} | " + " | ".join(cells) + " |")
    lines += ["", "| 데이터 | seed | ΔRMSE / 상대% | ΔMAE / 상대% |", "|---|---:|---:|---:|"]
    for d in DATASETS:
        for s in SEEDS:
            cells = []
            for metric in METRICS[:2]:
                x = next(v for v in result["seed_differences"] if (v["dataset"], v["seed"], v["cohort"], v["metric"]) == (d, s, "overall", metric))
                relative = "N/A" if x["relative_difference_percent"] is None else f"{x['relative_difference_percent']:+.3f}%"
                cells.append(f"{x['absolute_difference']:+.6f} / {relative}")
            lines.append(f"| {d} | {s} | " + " | ".join(cells) + " |")
    lines += ["", "차이는 공유형−기존 MLP16이며 양수는 악화다. 큰수량은 Taxi q>3449, RAF q>200, Intermittent q>187이다.", ""]
    lines += [f"- {s}" for s in result["interpretation_limits"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    require(argv in (None, []), "No alternate split or execution arguments")
    from paper.scripts import manage_titantpp_shared_history_raw9 as manager
    from paper.scripts import prepare_titantpp_shared_history_raw9 as preparation
    c = manager.verify_local()
    final = manager.BUNDLE / "retrieval/final_receipt.json"
    outputs = [manager.REPORT / f"THREE_SEED_VALIDATION.{s}" for s in ("json", "md")]
    require(not final.exists() and not any(p.exists() for p in outputs), "Preserve completed raw9 results")
    retrieval = manager.BUNDLE / "retrieval/5080"
    originals = retrieval / "originals"
    manager.verify_collected(c, "5080", originals)
    collection, inventory = manager.read(retrieval / "receipt.json"), manager.read(retrieval / "manifest.json")["files"]
    require(collection["status"] == "complete" and inventory and len(inventory) == collection["file_count"], "Incomplete immutable collection")
    require(manager.sha(retrieval / "original.tar.gz") == collection["archive_sha256"]
        and manager.sha(retrieval / "manifest.json") == collection["manifest_sha256"], "Collection archive changed")
    for name, digest in inventory.items():
        require(manager.sha(manager.runtime.local_path(originals, name)) == digest, "Collected raw9 original changed")
    records, bindings = [], {str(manager.BUNDLE / "execution_contract.json"): manager.sha(manager.BUNDLE / "execution_contract.json")}
    for j in c["jobs"]:
        path = originals / "run" / j["id"] / "validation_receipt.json"
        r = manager.read(path)
        require(r["status"] == "complete" and r["validation_replay_passed"] is True
            and r["dataset"] == j["dataset"] and r["seed"] == j["seed"] and r["arm"] == SHARED
            and r["time_alpha"] == 1 and r["held_out_test_evaluated"] is False, "Wrong raw9 endpoint")
        require(r["metrics"]["tail_threshold"] == THRESHOLDS[j["dataset"]]
            and r["metrics"]["tail_rule"] == "q > frozen_Train_large_quantity_threshold"
            and r["metrics"]["time_metric"] == "recorded_positive_integer_time_nll", "Large-quantity metric definition differs")
        records.append({"dataset": j["dataset"], "seed": j["seed"], "model": SHARED, "host": "5080",
            **CONDITION, "data_variant": "original_uncapped", "held_out_test_evaluated": False,
            "selected_epoch": r["selected_epoch"], "target_identity": r["target_identity"], "metrics": r["metrics"]})
        bindings[str(path)] = manager.sha(path)
    _, refs = preparation.baseline_document()
    for r in refs:
        require(r["tail_threshold"] == THRESHOLDS[r["dataset"]], "Stored baseline large-quantity rule differs")
        records.append({"dataset": r["dataset"], "seed": r["seed"], "model": BASELINE,
            "host": r["host"], **CONDITION, "data_variant": "original_uncapped", "held_out_test_evaluated": False,
            "selected_epoch": r["selected_epoch"], "target_identity": r["target_identity"], "metrics": r["metrics"]})
        bindings.update(r["source_bindings"])
    result = aggregate(records)
    archives = {str(retrieval / n): manager.sha(retrieval / n) for n in ("original.tar.gz", "manifest.json", "receipt.json")}
    result.update(status="complete", created_at=datetime.now(timezone.utc).isoformat(), source_bindings=bindings,
        archive_bindings=archives, new_fit_count=9, validation_replay_count=9, baseline_reference_count=9,
        comparison_model_count=18, seed_count=3, baseline_reevaluated=False, historical_cnn_changed=False)
    manager.write(outputs[0], result)
    with outputs[1].open("x") as stream:
        stream.write(markdown(result))
    manager.write(final, {"status": "complete", "new_fit_count": 9, "validation_replay_count": 9,
        "baseline_reference_count": 9, "comparison_model_count": 18, "seed_count": 3,
        "common_condition": CONDITION, "data_variant": "original_uncapped", "quantity_capping_applied": False,
        "actual_scientific_worker_exit0_count": 18, "actual_dispatcher_exit0_count": 1,
        "all_training_terminal_verified": True, "held_out_test_evaluated": False,
        "baseline_retrained": False, "baseline_reevaluated": False, "historical_cnn_changed": False,
        "source_bindings": bindings, "archive_bindings": archives,
        "report_files": {str(p): manager.sha(p) for p in outputs}})
    return result


if __name__ == "__main__":
    main()
