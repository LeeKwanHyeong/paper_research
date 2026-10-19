"""Aggregate the verified shared-history candidate18 and stored MLP16 reference9.

This module reads completed local Validation receipts only. It cannot launch a
worker or read held-out predictions. Native hosts remain part of every record;
matching targets may be compared across hosts with the Runtime limitation shown.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
BASELINE = "titantpp_history_mlp_width16"
SHARED = "titantpp_history_mlp_shared15_v1"
CONTROL = "titantpp_history_mlp_prefix_capacity_v1"
MODELS = (BASELINE, SHARED, CONTROL)
COHORTS = ("overall", "cap_hit", "tail")
METRICS = ("qty_rmse", "qty_mae", "time_nll")
CONTRASTS = ((SHARED, BASELINE), (CONTROL, BASELINE), (SHARED, CONTROL))
CONDITION = {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.0}
CAPS = dict(zip(DATASETS, (65.0, 10.0, 20.0)))
DEFINITIONS = {"cap_hit_rule": "q == cap_value", "tail_rule": "q > cap_value",
               "time_metric": "recorded_positive_integer_time_nll"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_row(row):
    require(isinstance(row, dict) and row.get("dataset") in DATASETS
            and type(row.get("seed")) is int and row["seed"] in SEEDS, "Unknown dataset or seed")
    require(row.get("model") in MODELS and row.get("host") in ("5080", "5090"), "Unknown model or native host")
    require(row["model"] == BASELINE or row["host"] == "5080", "New candidates must run on 5080")
    require(all(type(row.get(k)) is type(v) and row[k] == v for k, v in CONDITION.items()), "Common condition differs")
    require(row.get("scope") == "validation_only" and row.get("held_out_test_evaluated") is False
            and row.get("time_head_trained") is False, "Only quantity-only Validation endpoints are allowed")
    require(type(row.get("selected_epoch")) is int and row["selected_epoch"] > 0, "Invalid selected epoch")
    target = row.get("target_identity")
    require(isinstance(target, dict) and type(target.get("target_count")) is int and target["target_count"] > 0
            and all(isinstance(target.get(k), str) and len(target[k]) == 64 for k in
                    ("target_identity_sha256", "target_quantity_sha256")), "Missing exact target identity")
    metrics = row.get("metrics", {})
    require(all(metrics.get(k) == v for k, v in DEFINITIONS.items())
            and metrics.get("cap_value") == CAPS[row["dataset"]], "Metric cohort definition differs")
    for cohort in COHORTS:
        values = metrics.get(cohort, {})
        count = values.get("count")
        require(type(count) is int and count >= 0 and (cohort != "overall" or count > 0), "Invalid cohort count")
        require(all(metric in values for metric in METRICS), "Missing metric field")
        for metric in METRICS:
            value = values.get(metric)
            require(value is None if count == 0 else type(value) in (int, float) and math.isfinite(value)
                    and (metric == "time_nll" or value >= 0), "Invalid, missing or nonfinite metric")
    require(metrics["overall"]["count"] == target["target_count"]
            and metrics["tail"]["count"] == 0
            and metrics["cap_hit"]["count"] <= metrics["overall"]["count"], "Cap population mismatch")


def aggregate(records):
    """Return all three-seed summaries, paired differences and non-selective decisions."""
    indexed = {}
    for row in records:
        validate_row(row)
        key = row["dataset"], row["seed"], row["model"]
        require(key not in indexed, "Duplicate endpoint")
        indexed[key] = row
    require(len(records) == 27 and set(indexed) == {(d, s, m) for d in DATASETS for s in SEEDS for m in MODELS},
            "Expected exact candidate18 plus stored MLP16 reference9 matrix")
    summaries, differences, decisions = [], [], []
    for dataset in DATASETS:
        require(all(indexed[dataset, s, BASELINE]["target_identity"] == indexed[dataset, SEEDS[0], BASELINE]["target_identity"]
                    for s in SEEDS), "Seed target identities differ")
        for seed in SEEDS:
            reference = indexed[dataset, seed, BASELINE]
            for model in MODELS:
                row = indexed[dataset, seed, model]
                require(row["target_identity"] == reference["target_identity"], "Paired target identity differs")
                require(all(row["metrics"][c]["count"] == reference["metrics"][c]["count"] for c in COHORTS),
                        "Paired cohort count differs")
        for cohort in COHORTS:
            for model in MODELS:
                group = [indexed[dataset, seed, model]["metrics"][cohort] for seed in SEEDS]
                require(len({g["count"] for g in group}) == 1, "Seed cohort counts differ")
                count = group[0]["count"]
                for metric in METRICS:
                    values = [g[metric] for g in group]
                    mean, sd = (statistics.mean(values), statistics.stdev(values)) if count else (None, None)
                    require(mean is None or math.isfinite(mean) and math.isfinite(sd), "Nonfinite summary")
                    summaries.append({"dataset": dataset, "model": model, "cohort": cohort, "metric": metric,
                        "seed_count": 3, "target_count_per_seed": count, "mean": mean, "sample_sd_ddof1": sd,
                        "time_metric_diagnostic_only": metric == "time_nll"})
            for candidate, reference in CONTRASTS:
                for seed in SEEDS:
                    left, right = indexed[dataset, seed, candidate], indexed[dataset, seed, reference]
                    for metric in METRICS:
                        x, y = left["metrics"][cohort][metric], right["metrics"][cohort][metric]
                        delta = None if x is None else x - y
                        relative = None if delta is None or y == 0 else 100.0 * delta / abs(y)
                        require(delta is None or math.isfinite(delta), "Nonfinite difference")
                        require(relative is None or math.isfinite(relative), "Nonfinite relative difference")
                        differences.append({"dataset": dataset, "seed": seed, "candidate_model": candidate,
                            "reference_model": reference, "cohort": cohort, "metric": metric,
                            "candidate_host": left["host"], "reference_host": right["host"],
                            "candidate_selected_epoch": left["selected_epoch"], "reference_selected_epoch": right["selected_epoch"],
                            "candidate_value": x, "reference_value": y, "absolute_difference": delta,
                            "relative_difference_percent": relative, "worsened": None if delta is None else delta > 0,
                            "time_metric_diagnostic_only": metric == "time_nll"})
        for candidate, reference in CONTRASTS:
            delta = {metric: statistics.mean(indexed[dataset, seed, candidate]["metrics"]["overall"][metric]
                - indexed[dataset, seed, reference]["metrics"]["overall"][metric] for seed in SEEDS) for metric in METRICS[:2]}
            decisions.append({"dataset": dataset, "candidate_model": candidate, "reference_model": reference,
                "mean_differences": delta, "both_quantity_means_improved": all(v < 0 for v in delta.values()),
                "all_three_seeds_improved_both": all(indexed[dataset, seed, candidate]["metrics"]["overall"][m]
                    < indexed[dataset, seed, reference]["metrics"]["overall"][m] for seed in SEEDS for m in METRICS[:2])})
    return {"schema": "titantpp_shared_history_cap_three_seed_validation_v1", "scope": "validation_only",
        "common_condition": deepcopy(CONDITION), "summary": summaries, "seed_differences": differences,
        "decisions": decisions, "records": deepcopy(records), "raw_errors_pooled": False,
        "shared_improves_both_vs_baseline_and_control_all_datasets": all(
            d["both_quantity_means_improved"] for d in decisions if d["candidate_model"] == SHARED),
        "held_out_test_evaluated": False,
        "interpretation_limits": ["Same Validation selects checkpoints; exploratory, not independent evaluation or significance.",
            "Equal total/prefix parameter counts do not equate function classes, ranks or optimization.",
            "New candidates use 5080/cuDNN91900; stored MLP16 native 5080/5090 and cuDNN91900/92000 remain recorded.",
            "Time head is untrained; recorded positive-integer TimeNLL is diagnostic. q>cap is empty N/A."]}


def markdown(result):
    lines = ["# 공유형 이력 MLP와 용량 대조군의 3seed Validation 비교", "",
        "Cap 세 데이터셋·log1p 입력·log-MSE·시간 가중치0·기존 head·seed42/52/62를 유지했다.",
        "선택은 전체 Validation 실수량 RMSE의 최초 strict finite 최소 epoch이며 MAE도 같은 checkpoint에서 측정했다.",
        "기존 MLP16 저장9개와 신규18개를 같은 seed·정확히 같은 대상으로 비교했다. 평균±표본SD(ddof1).", "",
        "| 데이터 | 구조 | 전체 RMSE | 전체 MAE | 상한 도달 RMSE | 상한 도달 MAE | 전체 TimeNLL | 상한 도달 TimeNLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for model in MODELS:
            cells = []
            for cohort, metric in (("overall", "qty_rmse"), ("overall", "qty_mae"), ("cap_hit", "qty_rmse"),
                                   ("cap_hit", "qty_mae"), ("overall", "time_nll"), ("cap_hit", "time_nll")):
                row = next(r for r in result["summary"] if (r["dataset"], r["model"], r["cohort"], r["metric"])
                           == (dataset, model, cohort, metric))
                cells.append("N/A" if row["mean"] is None else f"{row['mean']:.6f}±{row['sample_sd_ddof1']:.6f}")
            lines.append(f"| {dataset} | {model} | " + " | ".join(cells) + " |")
    lines += ["", "각 행은 후보−기준의 절대차이 / 상대차이(%)다. 양수는 악화이며 0인 기준의 상대차이는 N/A다.", "",
              "| 데이터 | seed | 대비 | ΔRMSE / % | ΔMAE / % |", "|---|---:|---|---:|---:|"]
    for dataset in DATASETS:
        for candidate, reference in CONTRASTS:
            for seed in SEEDS:
                cells = []
                for metric in METRICS[:2]:
                    row = next(r for r in result["seed_differences"] if (r["dataset"], r["seed"], r["candidate_model"],
                        r["reference_model"], r["cohort"], r["metric"]) == (dataset, seed, candidate, reference, "overall", metric))
                    relative = "N/A" if row["relative_difference_percent"] is None else f"{row['relative_difference_percent']:+.3f}%"
                    cells.append(f"{row['absolute_difference']:+.6f} / {relative}")
                lines.append(f"| {dataset} | {seed} | {candidate} − {reference} | " + " | ".join(cells) + " |")
    lines += ["", "q>cap 집단은 비어 N/A다. 전체·상한 도달 TimeNLL은 미학습 시간 head의 진단값이다.",
        "신규 후보는 5080/cuDNN91900이며 기존 MLP16은 저장된 5080/5090·cuDNN91900/92000 결과를 그대로 사용했다.",
        "전체 seed별 여섯 지표와 악화·선택 epoch·실제 host는 JSON에 보존했다. 데이터셋별 원단위 오류를 평균내지 않았다.",
        "같은 가중치 수가 함수·유효 rank·최적화까지 같다는 뜻은 아니다. 탐색적 Validation이며 독립 검증·인과관계·통계유의성을 주장하지 않는다.",
        "기존 CNN·MLP16을 변경하거나 재학습·재평가하지 않았으며 Test는 평가하지 않았다."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    require(argv in (None, []), "No execution or alternate-split arguments are supported")
    from paper.scripts import manage_titantpp_shared_history_cap_execution as manager
    from paper.scripts.compare_titantpp_time_zero_mlp9 import verify_zero_proof
    contract = manager.verify_local()
    require(contract["common_condition"] == CONDITION and contract["evaluation_scope"] == "validation_only"
            and contract["held_out_test_authorized"] is False, "Wrong scope or common condition")
    final = manager.BUNDLE / "retrieval/final_receipt.json"
    outputs = [manager.REPORT / f"THREE_SEED_VALIDATION.{suffix}" for suffix in ("json", "md")]
    require(not final.exists() and not any(p.exists() for p in outputs), "Preserve previous results; no overwrite")
    retrieval = manager.BUNDLE / "retrieval/5080"
    originals = retrieval / "originals"
    terminal = manager.verify_collected(contract, "5080", originals)
    collected, manifest = manager.read(retrieval / "receipt.json"), manager.read(retrieval / "manifest.json")
    require(collected["status"] == "complete" and manager.sha(retrieval / "original.tar.gz") == collected["archive_sha256"]
            and manager.sha(retrieval / "manifest.json") == collected["manifest_sha256"], "Collection SHA differs")
    require(manifest["files"] and len(manifest["files"]) == collected["file_count"], "Incomplete inventory")
    for name, digest in manifest["files"].items():
        require(manager.sha(manager.runtime.local_path(originals, name)) == digest, "Collected original changed")
    jobs = contract["jobs"]
    require(len(jobs) == 18 and all(j["host"] == "5080" for j in jobs), "Wrong training host or job count")
    for stage in ("train", "validation"):
        exits = terminal["stages"][stage]["exits"]
        require(set(exits) == {j["id"] for j in jobs} and all(e["actual_os_returncode"] == 0
                and not e["timeout"] for e in exits.values()), "Incomplete scientific exits")
    require(manager.read(originals / "dispatcher_exit.json")["actual_os_returncode"] == 0, "Incomplete dispatcher exit")
    records, bindings = [], {str(manager.BUNDLE / "execution_contract.json"):
                            manager.sha(manager.BUNDLE / "execution_contract.json"),
                            str(Path(__file__).resolve()): manager.sha(Path(__file__).resolve())}
    def append(endpoint, source, model, host):
        require(endpoint["status"] == "complete" and endpoint["validation_replay_passed"] is True
                and endpoint["held_out_test_evaluated"] is False and endpoint["arm"] == model,
                "Endpoint completion or arm differs")
        require(endpoint["dataset"] == source["dataset"] and endpoint["seed"] == source["seed"]
                and endpoint["time_alpha"] == 0.0 and endpoint["host"] == host
                and endpoint.get("loss_mode", "log_mse") == "log_mse"
                and endpoint.get("time_input_transform", "log1p_nonnegative_float32") == "log1p_nonnegative_float32"
                and endpoint.get("quantity_input_transform", "log1p_nonnegative_float32") == "log1p_nonnegative_float32",
                "Endpoint job or input identity differs")
        records.append({"dataset": source["dataset"], "seed": source["seed"], "model": model, "host": host,
            **CONDITION, "time_head_trained": False, "scope": "validation_only", "held_out_test_evaluated": False,
            "selected_epoch": endpoint["selected_epoch"], "target_identity": deepcopy(endpoint["target_identity"]),
            "metrics": deepcopy(endpoint["metrics"]), "time_head_no_training_verification": verify_zero_proof(endpoint)})
    for job in jobs:
        require(job["arm"] in (SHARED, CONTROL) and all(job[k] == v for k, v in CONDITION.items()), "Job contract differs")
        path = originals / "run" / job["id"] / "validation_receipt.json"
        append(manager.read(path), job, job["arm"], "5080")
        bindings[str(path)] = manager.sha(path)
    refs = contract["baseline_validation_references"]
    require(len(refs) == 9, "Expected stored MLP16 reference9")
    for ref in refs:
        require(ref["arm"] == BASELINE and all(ref[k] == v for k, v in CONDITION.items()), "Baseline condition differs")
        for field, digest in (("path", "sha256"), ("manifest_path", "manifest_sha256"),
                              ("completion_receipt_path", "completion_receipt_sha256"), ("selected_checkpoint", "checkpoint_sha256")):
            require(manager.sha(ref[field]) == ref[digest], "Baseline original changed")
            bindings[str(ref[field])] = ref[digest]
        endpoint = manager.read(ref["path"])
        require(manager.read(ref["completion_receipt_path"])["status"] == "complete", "Baseline collection incomplete")
        require(endpoint["metrics"] == ref["metrics"] and endpoint["target_identity"] == ref["target_identity"]
                and endpoint["selected_epoch"] == ref["selected_epoch"]
                and endpoint["checkpoint_sha256"] == ref["checkpoint_sha256"], "Stored baseline endpoint differs")
        append(endpoint, ref, BASELINE, ref["host"])
    result = aggregate(records)
    archives = {str(retrieval / n): manager.sha(retrieval / n) for n in ("original.tar.gz", "manifest.json", "receipt.json")}
    result.update(status="complete", created_at=datetime.now(timezone.utc).isoformat(), source_bindings=bindings,
        archive_bindings=archives, new_fit_count=18, validation_replay_count=18, baseline_reference_count=9,
        comparison_model_count=27, seed_count=3, baseline_reevaluated=False, historical_cnn_changed=False)
    manager.write(outputs[0], result)
    with outputs[1].open("x") as stream:
        stream.write(markdown(result))
    manager.write(final, {"status": "complete", "new_fit_count": 18, "validation_replay_count": 18,
        "baseline_reference_count": 9, "comparison_model_count": 27, "seed_count": 3, "common_condition": CONDITION,
        "actual_scientific_worker_exit0_count": 36, "actual_scientific_child_exit0_count": 36,
        "actual_dispatcher_exit0_count": 1, "time_head_no_training_verified_count": 18,
        "all_training_terminal_verified": True, "training_host": "5080",
        "held_out_test_evaluated": False, "historical_cnn_retrained": False, "historical_cnn_reevaluated": False,
        "baseline_reevaluated": False, "source_bindings": bindings, "archive_bindings": archives,
        "report_files": {str(p): manager.sha(p) for p in outputs}})
    return result


if __name__ == "__main__":
    main()
