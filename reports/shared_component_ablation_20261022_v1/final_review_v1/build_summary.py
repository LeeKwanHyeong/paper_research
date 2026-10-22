"""Build a local Korean summary only from the completed, fixed comparison JSON.

No experiment code, prediction files, checkpoint, remote command or inference is
loaded.  All derived values are prepared and validated in memory before any of
the four exclusive output files is opened.  Existing outputs are never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = "shared_component_ablation_20261022_v1"
INPUT = ROOT / "search_artifacts" / CAMPAIGN / "comparison_v1/REPORT.json"
OUTPUT = Path(__file__).resolve().parent
SCHEMA = "shared_component_ablation_comparison_v1"
DATASETS = ("intermittent_frozen_5000", "raf_spare_parts", "yellow_trip_hourly")
ARMS = ("full", "no_prototype", "no_history", "encoder_only")
LABELS = {"full": "A: 정적 prototype + history", "no_prototype": "B: history",
    "no_history": "C: 정적 prototype", "encoder_only": "D: 공통 encoder"}
LETTERS = dict(zip(ARMS, "ABCD"))
SEEDS, SPLITS, SCOPES = (42, 52, 62), ("validation", "test"), ("overall", "large")
METRICS = ("qty_rmse", "qty_mae", "time_nll")
PAIRS = (("full", "no_prototype"), ("full", "no_history"),
    ("no_prototype", "encoder_only"), ("no_history", "encoder_only"))
MIB = 1024 ** 2
OUTPUT_NAMES = ("RESULT_SUMMARY.md", "DERIVED_TABLES.json", "ALL_SEED_DELTAS.csv", "GPU_SEED_COSTS.csv")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def number(value, field, *, nonnegative=False, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), "Missing/nonfinite numeric field: " + field)
    require(not nonnegative or value >= 0, "Negative numeric field: " + field)
    require(not positive or value > 0, "Nonpositive numeric field: " + field)
    return float(value)


def equal_number(actual, expected, field):
    if expected is None:
        require(actual is None, "Expected explicit N/A: " + field)
    else:
        require(math.isclose(number(actual, field), expected, rel_tol=1e-12, abs_tol=1e-12),
            "Stored/derived number differs: " + field)


def host_for(seed):
    return "5080" if seed == 42 else "5090"


def job_id(dataset, arm, seed):
    return f"{dataset}__{arm}__seed{seed}"


def indexed(rows, keys, expected, name):
    require(isinstance(rows, list) and len(rows) == len(expected), "Wrong row count: " + name)
    result = {}
    for row in rows:
        require(isinstance(row, dict), "Non-object row: " + name)
        key = tuple(row.get(field) for field in keys)
        require(key in expected and key not in result, "Unexpected/duplicate identity: " + name + repr(key))
        result[key] = row
    require(set(result) == expected, "Missing identities: " + name)
    return result


def metric_value(row, scope, metric):
    block = row["metrics"][scope]
    require(type(block.get("count")) is int and block["count"] >= 0, "Missing population count")
    require(metric in block, "Missing accuracy metric: " + scope + "." + metric)
    value = block[metric]
    if block["count"] == 0:
        require(scope == "large" and block.get("status") == "N/A_empty" and value is None,
            "Missing numeric result is not a declared empty large-quantity population")
        return None
    require(block.get("status") == "available", "Accuracy metric availability differs")
    return number(value, scope + "." + metric, nonnegative=metric != "time_nll")


def mean_sd(values):
    present = [value for value in values if value is not None]
    require(len(present) in (0, 3), "Partially missing three-seed accuracy result")
    return {"mean": statistics.fmean(present) if present else None,
        "sample_sd_ddof1": statistics.stdev(present) if present else None, "n": len(present)}


def delta(left, right):
    if left is None or right is None:
        require(left is None and right is None, "Only one arm lacks a paired metric")
        return {"absolute": None, "relative": None, "adverse": False}
    absolute = left - right
    return {"absolute": absolute, "relative": None if right == 0 else absolute / right, "adverse": absolute > 0}


def accuracy_tables(source):
    record_keys = {(dataset, split, arm, seed) for dataset in DATASETS for split in SPLITS for arm in ARMS for seed in SEEDS}
    records = indexed(source.get("records"), ("dataset", "split", "arm", "seed"), record_keys, "accuracy records (72)")
    for (dataset, split, arm, seed), row in records.items():
        require(row.get("host") == host_for(seed) and row.get("job_id") == job_id(dataset, arm, seed), "Accuracy job/host differs")
        require(type(row.get("metrics")) is dict, "Accuracy metrics object missing")
        require(row["metrics"].get("large_rule") == "true_qty > quantity_boundaries_all_train_rows[-1]", "Large-quantity rule differs")
        number(row["metrics"].get("large_cutoff"), "large_cutoff", nonnegative=True)
        for scope in SCOPES:
            for metric in METRICS:
                metric_value(row, scope, metric)
    table_keys = {(dataset, split, arm) for dataset in DATASETS for split in SPLITS for arm in ARMS}
    original = indexed(source.get("table"), ("dataset", "split", "arm"), table_keys, "accuracy table (24)")
    table = []
    for dataset in DATASETS:
        for split in SPLITS:
            for arm in ARMS:
                values = {scope: {metric: mean_sd([metric_value(records[(dataset, split, arm, seed)], scope, metric)
                    for seed in SEEDS]) for metric in METRICS} for scope in SCOPES}
                previous = original[(dataset, split, arm)]
                for scope in SCOPES:
                    for metric in METRICS:
                        expected = values[scope][metric]
                        actual = previous[scope][metric]
                        require(actual.get("n") == expected["n"], "Accuracy seed count differs")
                        for field in ("mean", "sample_sd_ddof1"):
                            equal_number(actual.get(field), expected[field], scope + "." + metric + "." + field)
                table.append({"dataset": dataset, "split": split, "arm": arm, "arm_label": LABELS[arm], **values})
    keys = {(dataset, split, seed, left + "-" + right, scope, metric) for dataset in DATASETS for split in SPLITS
        for seed in SEEDS for left, right in PAIRS for scope in SCOPES for metric in METRICS}
    original_pairs = indexed(source.get("paired_seed_deltas"), ("dataset", "split", "seed", "comparison", "scope", "metric"),
        keys, "all paired seed deltas (432)")
    paired = []
    for dataset in DATASETS:
        for split in SPLITS:
            for seed in SEEDS:
                for left, right in PAIRS:
                    for scope in SCOPES:
                        for metric in METRICS:
                            left_value = metric_value(records[(dataset, split, left, seed)], scope, metric)
                            right_value = metric_value(records[(dataset, split, right, seed)], scope, metric)
                            item = {"dataset": dataset, "split": split, "seed": seed, "host": host_for(seed),
                                "comparison": left + "-" + right, "comparison_label": LETTERS[left] + "–" + LETTERS[right],
                                "scope": scope, "metric": metric, "left": left_value, "right": right_value, **delta(left_value, right_value)}
                            old = original_pairs[(dataset, split, seed, item["comparison"], scope, metric)]
                            for field in ("left", "right", "absolute", "relative"):
                                equal_number(old.get(field), item[field], "paired." + field)
                            require(old.get("adverse") is item["adverse"], "Adverse flag differs")
                            paired.append(item)
    adverse_original = source.get("adverse_seed_results")
    require(isinstance(adverse_original, list), "Full adverse-seed inventory missing")
    expected_adverse = {key for key, value in original_pairs.items() if value["adverse"]}
    actual_adverse = indexed(adverse_original, ("dataset", "split", "seed", "comparison", "scope", "metric"),
        expected_adverse, "adverse seed deltas")
    require(all(actual_adverse[key] == original_pairs[key] for key in actual_adverse), "Adverse inventory differs from all deltas")
    return table, paired, [item for item in paired if item["adverse"]]


def summarized(values):
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values), "repetitions": values}


def cost_tables(source):
    expected = {(dataset, arm, host_for(seed), seed, mode, rep) for dataset in DATASETS for arm in ARMS
        for seed in SEEDS for mode in ("train", "eval") for rep in (1, 2, 3)}
    raw = indexed(source.get("cost_records"), ("dataset", "arm", "host", "training_seed", "mode", "timing_rep"),
        expected, "raw cost records (216)")
    groups, runtimes, parameter_counts = defaultdict(list), defaultdict(dict), defaultdict(set)
    for (dataset, arm, host, seed, mode, rep), item in raw.items():
        require(item.get("status") == "complete" and item.get("job_id") == job_id(dataset, arm, seed), "Cost identity/status differs")
        require(item.get("scientific_fit") is False and item.get("input_split") == "train"
            and item.get("synthetic") is False and item.get("checkpoint_binary_unchanged") is True
            and item.get("checkpoint_saved") is False and item.get("optimizer_steps") == (37 if mode == "train" else 0),
            "Cost is not the fixed Train-batch profiling procedure")
        require(type(item.get("parameters")) is int and item["parameters"] > 0, "Total parameter numel missing")
        parameter_counts[(dataset, arm)].add(item["parameters"])
        runtime = item.get("runtime")
        require(isinstance(runtime, dict) and isinstance(runtime.get("gpu"), dict), "Native GPU runtime missing")
        gpu = runtime["gpu"]
        require(all(isinstance(gpu.get(key), str) and bool(gpu[key]) for key in ("uuid", "name")), "GPU identity missing")
        require(all(runtime.get(key) is not None for key in ("python", "torch", "cuda", "cudnn")), "Native software versions missing")
        runtime_key = json.dumps(runtime, sort_keys=True, ensure_ascii=False, allow_nan=False)
        runtimes[host][runtime_key] = runtime
        values = {"timing_rep": rep, "parameters": item["parameters"],
            "compute_latency_ms": number(item.get("median_compute_seconds"), "median_compute_seconds", positive=True) * 1000,
            "including_transfer_latency_ms": number(item.get("median_transfer_compute_seconds"), "median_transfer_compute_seconds", positive=True) * 1000,
            "compute_throughput_targets_per_second": number(item.get("throughput_targets_per_second"), "throughput_targets_per_second", positive=True),
            "including_transfer_throughput_targets_per_second": number(item.get("throughput_including_transfer_targets_per_second"), "throughput_including_transfer_targets_per_second", positive=True)}
        for key in ("allocated", "reserved"):
            peak = item.get("total_peak_bytes", {}).get(key)
            require(type(peak) is int and peak >= 0, "Total memory peak missing")
            warmup = item.get("warmup_peak_bytes", {}).get(key)
            measured = item.get("measured_peak_bytes", {}).get(key)
            require(type(warmup) is int and type(measured) is int and peak == max(warmup, measured), "Total memory peak derivation differs")
            values["total_peak_" + key + "_mib"] = peak / MIB
        groups[(dataset, arm, host, seed, mode)].append(values)
    require(set(runtimes) == {"5080", "5090"} and all(len(items) == 1 for items in runtimes.values()), "Within-host runtime drift")
    runtime_by_host = {host: next(iter(items.values())) for host, items in runtimes.items()}
    require(runtime_by_host["5080"]["gpu"]["uuid"] != runtime_by_host["5090"]["gpu"]["uuid"], "GPU identities are not independent")
    require(all(len(values) == 1 for values in parameter_counts.values()), "Parameter numel differs by seed/mode/repetition")
    aggregate = []
    for (dataset, arm, host, seed, mode), repetitions in sorted(groups.items()):
        repetitions.sort(key=lambda item: item["timing_rep"])
        require([item["timing_rep"] for item in repetitions] == [1, 2, 3], "Three technical cost repetitions missing")
        summaries = {field: summarized([item[field] for item in repetitions]) for field in (
            "compute_latency_ms", "including_transfer_latency_ms", "compute_throughput_targets_per_second",
            "including_transfer_throughput_targets_per_second", "total_peak_allocated_mib", "total_peak_reserved_mib")}
        runtime = runtime_by_host[host]
        aggregate.append({"dataset": dataset, "arm": arm, "arm_label": LABELS[arm], "host": host,
            "training_seed": seed, "mode": mode, "timing_repetitions": 3, "gpu_name": runtime["gpu"]["name"],
            "gpu_uuid": runtime["gpu"]["uuid"], "cudnn": runtime["cudnn"], "parameters_total_numel": repetitions[0]["parameters"],
            "compute_latency_ms_median": summaries["compute_latency_ms"]["median"],
            "compute_latency_ms_min": summaries["compute_latency_ms"]["minimum"],
            "compute_latency_ms_max": summaries["compute_latency_ms"]["maximum"],
            "including_transfer_latency_ms_median": summaries["including_transfer_latency_ms"]["median"],
            "including_transfer_latency_ms_min": summaries["including_transfer_latency_ms"]["minimum"],
            "including_transfer_latency_ms_max": summaries["including_transfer_latency_ms"]["maximum"],
            "compute_throughput_targets_per_second_median": summaries["compute_throughput_targets_per_second"]["median"],
            "including_transfer_throughput_targets_per_second_median": summaries["including_transfer_throughput_targets_per_second"]["median"],
            "total_peak_allocated_mib_median": summaries["total_peak_allocated_mib"]["median"],
            "total_peak_allocated_mib_max_rep_peak": summaries["total_peak_allocated_mib"]["maximum"],
            "total_peak_reserved_mib_median": summaries["total_peak_reserved_mib"]["median"],
            "total_peak_reserved_mib_max_rep_peak": summaries["total_peak_reserved_mib"]["maximum"], "repetition_summaries": summaries})
    require(len(aggregate) == 72, "Expected 72 GPU/seed/mode cost summary rows")
    by_key = {(row["dataset"], row["arm"], row["host"], row["training_seed"], row["mode"]): row for row in aggregate}
    ratio_fields = ("compute_latency_ms_median", "including_transfer_latency_ms_median",
        "compute_throughput_targets_per_second_median", "including_transfer_throughput_targets_per_second_median",
        "total_peak_allocated_mib_median", "total_peak_allocated_mib_max_rep_peak",
        "total_peak_reserved_mib_median", "total_peak_reserved_mib_max_rep_peak", "parameters_total_numel")
    comparisons = []
    for dataset in DATASETS:
        for seed in SEEDS:
            host = host_for(seed)
            for mode in ("train", "eval"):
                for left, right in PAIRS:
                    first, second = by_key[(dataset, left, host, seed, mode)], by_key[(dataset, right, host, seed, mode)]
                    comparisons.append({"dataset": dataset, "host": host, "gpu_uuid": first["gpu_uuid"],
                        "training_seed": seed, "mode": mode, "comparison": left + "-" + right,
                        "comparison_label": LETTERS[left] + "–" + LETTERS[right], "metrics": {field: {
                            "left": first[field], "right": second[field], "left_over_right": None if second[field] == 0 else first[field] / second[field],
                            "absolute": first[field] - second[field], "relative": None if second[field] == 0 else (first[field] - second[field]) / second[field]}
                            for field in ratio_fields}})
    require(len(comparisons) == 72, "Expected 72 same-GPU/seed cost comparisons")
    return aggregate, comparisons, runtime_by_host


def fit_table(source):
    expected = {(dataset, arm, seed) for dataset in DATASETS for arm in ARMS for seed in SEEDS}
    rows = indexed(source.get("fit_records"), ("dataset", "arm", "seed"), expected, "fit origin records (36)")
    result = []
    for (dataset, arm, seed), row in sorted(rows.items()):
        require(row.get("host") == host_for(seed) and row.get("job_id") == job_id(dataset, arm, seed), "Fit identity differs")
        reused = arm == "full" and seed in (52, 62)
        require(row.get("training_origin") == ("reused_verified_fit" if reused else "fresh_fit"), "Fit origin differs")
        if reused:
            require(all(row.get(key) is None for key in ("elapsed_seconds", "completed_epochs", "amortized_elapsed_seconds_per_epoch")), "Reused fit must retain unknown elapsed/epochs")
        else:
            elapsed = number(row.get("elapsed_seconds"), "fit.elapsed_seconds", positive=True)
            epochs = row.get("completed_epochs")
            require(type(epochs) is int and epochs > 0, "Fit completed_epochs missing")
            equal_number(row.get("amortized_elapsed_seconds_per_epoch"), elapsed / epochs, "fit.amortized_elapsed_seconds_per_epoch")
        result.append({**row, "arm_label": LABELS[arm]})
    require(Counter(row["training_origin"] for row in result) == {"fresh_fit": 30, "reused_verified_fit": 6}, "Expected 30 new and six reused original fits")
    return result


def derive(source):
    require(isinstance(source, dict) and source.get("schema") == SCHEMA and source.get("status") == "complete", "Input comparison is not complete")
    contract_sha = source.get("contract_sha256")
    require(isinstance(contract_sha, str) and len(contract_sha) == 64 and all(ch in "0123456789abcdef" for ch in contract_sha), "Input contract SHA missing")
    accuracy, paired, adverse = accuracy_tables(source)
    costs, cost_comparisons, runtimes = cost_tables(source)
    fits = fit_table(source)
    return {"schema": "shared_component_final_review_v1", "status": "complete", "contract_sha256": contract_sha,
        "model_name": "미정", "arms": LABELS, "accuracy_table": accuracy, "paired_seed_deltas": paired,
        "adverse_seed_results": adverse, "gpu_seed_costs": costs, "same_gpu_seed_cost_comparisons": cost_comparisons,
        "host_runtime": runtimes, "fit_records": fits, "fit_origin_counts": dict(Counter(row["training_origin"] for row in fits)),
        "counts": {"accuracy_records": 72, "accuracy_rows": 24, "paired_seed_deltas": 432, "raw_cost_records": 216,
            "gpu_seed_cost_rows": 72, "same_gpu_seed_cost_comparisons": 72, "fit_records": 36,
            "adverse_seed_results": len(adverse)},
        "definitions": {"accuracy": "세 훈련 seed의 평균과 표본 표준편차(ddof=1). large는 원본의 truth > cutoff.",
            "time_nll": "저장된 예측에서 원본 비교기가 재계산한 recorded TimeNLL. 다른 시간 손실로 바꾸지 않음.",
            "accuracy_delta": "왼쪽−오른쪽; 상대값은 (왼쪽−오른쪽)/오른쪽. 양수는 왼쪽의 오류/recorded TimeNLL이 더 큼.",
            "empty_large": "원본이 count=0 및 N/A_empty로 명시한 큰 수량 집합만 N/A 보존. 누락값/부분 누락/비유한 수치는 오류.",
            "latency": "각 기술 반복의 32 측정 배치 중앙값을 구한 뒤 세 반복의 중앙값[min,max]. 단위 ms.",
            "throughput": "각 반복의 원본 throughput(4096 targets / 32 측정 배치 시간 합)의 중앙값. 128/중앙 latency로 대체하지 않음.",
            "memory": "각 반복 total peak=max(warmup peak, measured peak). 세 반복 중앙값과 max-rep peak를 MiB로 병기.",
            "parameters": "모든 model.parameters()의 numel 합; trainable-only 수치가 아님.",
            "cost_repetitions": "훈련 seed별 세 timing rep는 기술 반복이며 세 훈련 seed를 대신하지 않음.",
            "eval_cost": "Test 데이터 속도가 아니라 동일한 고정 Train 배치에서 joint loss를 포함한 eval 비용.",
            "train_cost": "선택 checkpoint에서 새 AdamW로 37 step을 측정한 프로필. 기존 optimizer를 재개하지 않으며 checkpoint를 저장하지 않음.",
            "paused_reservation": "5080의 중지된 다른 프로세스 GPU 예약은 이 profiler의 allocated/reserved peak와 별개.",
            "fit_elapsed": "elapsed/completed_epochs는 validation·checkpoint 저장을 포함한 상각값. epoch별 직접 측정이 아님.",
            "historical_a": "과거 A의 3개 병렬 fit wall-clock은 새 단일 worker fit 시간과 직접 비교하지 않음.",
            "followup_policy": "후속 복구의 학습 재시작 0; MAC 보류; 관측 성능에 따른 조건 변경 0."}}


def formatted(summary):
    if summary["mean"] is None:
        return "N/A (n=0)"
    return f"{summary['mean']:.6g} ± {summary['sample_sd_ddof1']:.4g}"


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def markdown(result):
    lines = ["# 구성요소 A–D 비교 결과 — 모델명 미정", "",
        "A는 정적 prototype과 history, B는 history, C는 정적 prototype, D는 공통 encoder를 사용한다.", "",
        "후속 복구의 학습 재시작 0 · MAC 보류 · 관측 성능에 따른 조건 변경 0.", "",
        f"[원본 검증 보고서]({INPUT.with_name('REPORT.md')}) · [원본 JSON]({INPUT}) · "
        "[전체 파생 표](DERIVED_TABLES.json) · [432개 seed별 차이](ALL_SEED_DELTAS.csv) · [72개 GPU·seed별 비용](GPU_SEED_COSTS.csv)", "",
        "## 정확도: 전체 24개 조합", "",
        "Validation과 Test를 구분했다. 각 값은 훈련 seed 42·52·62의 평균 ± 표본 표준편차(ddof=1)이며, "
        "시간 지표는 원본 recorded TimeNLL이다. 큰 수량은 원본의 truth > cutoff를 따른다. 명시된 빈 큰 수량 집합은 N/A로 남겼다.", "",
        "| Dataset | Split | 구성 | 전체 RMSE | 전체 MAE | 전체 recorded TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 recorded TimeNLL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["accuracy_table"]:
        values = [formatted(row[scope][metric]) for scope in SCOPES for metric in METRICS]
        lines.append("| " + " | ".join([row["dataset"], row["split"], LETTERS[row["arm"]], *values]) + " |")
    lines.extend(["", "## Seed별 구성요소 차이", "",
        f"A–B, A–C, B–D, C–D의 432개 차이를 모두 보존했다. 이 중 왼쪽의 오류 또는 recorded TimeNLL이 더 큰 경우는 "
        f"{len(result['adverse_seed_results'])}개이며, 불리한 결과 전체는 DERIVED_TABLES.json의 adverse_seed_results와 원본 보고서에서 확인할 수 있다. "
        "절대 차이는 왼쪽−오른쪽, 상대 차이는 그 값을 오른쪽으로 나눈 값이다. 오른쪽이 0이면 상대값은 비워 두었다.", "",
        "## GPU와 훈련 seed를 분리한 비용", "",
        "216개 원본 기술 반복을 dataset·구성·GPU·훈련 seed·train/eval별 72개 행으로 정리했다. "
        "세 기술 반복은 세 훈련 seed가 아니다. GPU 간 수치나 seed를 합쳐 평균을 만들지 않았다. "
        "아래 latency는 세 반복 중앙값의 중앙값[min,max]이며, 처리량은 각 반복의 원본 처리량 중앙값이다. "
        "GPU_SEED_COSTS.csv에는 연산만의 비용과 전송 포함 비용을 함께 보존했다.", "",
        "| Host / GPU | Dataset | 구성 | 훈련 seed | Mode | 연산 ms 중앙값[min,max] | 전송 포함 ms 중앙값[min,max] | 연산 targets/s | 전송 포함 targets/s | Allocated MiB 중앙값 / 반복 최대 | Reserved MiB 중앙값 / 반복 최대 | 전체 parameters |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in result["gpu_seed_costs"]:
        def latency(prefix):
            return f"{row[prefix + '_median']:.6g} [{row[prefix + '_min']:.6g}, {row[prefix + '_max']:.6g}]"
        values = [row["host"] + " / " + row["gpu_name"], row["dataset"], LETTERS[row["arm"]], str(row["training_seed"]), row["mode"],
            latency("compute_latency_ms"), latency("including_transfer_latency_ms"),
            f"{row['compute_throughput_targets_per_second_median']:.6g}", f"{row['including_transfer_throughput_targets_per_second_median']:.6g}",
            f"{row['total_peak_allocated_mib_median']:.6g} / {row['total_peak_allocated_mib_max_rep_peak']:.6g}",
            f"{row['total_peak_reserved_mib_median']:.6g} / {row['total_peak_reserved_mib_max_rep_peak']:.6g}", str(row["parameters_total_numel"])]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "같은 GPU·훈련 seed에서 계산한 A–B/A–C/B–D/C–D 비용 비교 72개의 비율과 차이는 "
        "DERIVED_TABLES.json의 same_gpu_seed_cost_comparisons에 모두 남겼다.", "",
        "eval 비용은 Test 속도가 아니라 같은 Train 배치의 joint loss 포함 eval 비용이다. train 프로필은 선택 checkpoint와 새 AdamW로 "
        "37 step을 측정한 값이며 기존 optimizer의 학습 재개가 아니다. Parameters는 전체 numel이다. "
        "메모리는 warmup·측정 구간 피크의 최댓값을 사용했고, 반복 최대 피크도 함께 기록했다. "
        "5080의 중지된 다른 프로세스 예약은 이 profiler의 allocated/reserved 수치와 분리한다.", "",
        "## 실행 환경과 fit 출처", "",
        "| Host | GPU | Python | PyTorch | CUDA | cuDNN |", "|---|---|---|---|---|---|"])
    for host, runtime in sorted(result["host_runtime"].items()):
        lines.append("| " + " | ".join(str(value) for value in (host, runtime["gpu"]["name"], runtime["python"], runtime["torch"], runtime["cuda"], runtime["cudnn"])) + " |")
    lines.extend(["", "5080/5090의 GPU와 평가 환경을 구분해 기록했다. cuDNN을 포함한 실행 환경 차이는 GPU별 비용 해석에 함께 고려해야 한다.", "",
        "36개 fit 기록은 새 fit 30개와 검증한 과거 A 재사용 6개로 나뉜다. 후속 복구에서 학습을 다시 시작한 횟수는 0이다. "
        "재사용 A의 시간을 0으로 채우지 않았다. 과거 A의 3개 병렬 fit 시간은 새 단일 worker fit 시간과 직접 비교할 수 없다.", "",
        "| Host | Dataset | 구성 | Seed | Origin | Elapsed(s) | 완료 epoch | Elapsed/epoch(s), 상각값 |",
        "|---|---|---|---:|---|---:|---:|---:|"])
    for row in result["fit_records"]:
        values = [row["host"], row["dataset"], LETTERS[row["arm"]], str(row["seed"]), row["training_origin"],
            *["N/A" if row[key] is None else f"{row[key]:.6g}" for key in ("elapsed_seconds", "completed_epochs", "amortized_elapsed_seconds_per_epoch")]]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Elapsed/epoch는 validation과 checkpoint 저장을 포함한 상각값이며 epoch별 직접 측정 시간이 아니다.", "",
        "## 입력 증적", "", f"- 입력: `{result['provenance']['input_path']}`",
        f"- 입력 SHA256: `{result['provenance']['input_sha256']}`",
        f"- 작성 스크립트 SHA256: `{result['provenance']['builder_sha256']}`", ""])
    return "\n".join(lines)


def build_documents(source, provenance):
    result = derive(source)
    result["provenance"] = provenance
    paired_fields = ("dataset", "split", "seed", "host", "comparison", "comparison_label", "scope", "metric", "left", "right", "absolute", "relative", "adverse")
    cost_fields = tuple(key for key in result["gpu_seed_costs"][0] if key != "repetition_summaries")
    return {"RESULT_SUMMARY.md": markdown(result),
        "DERIVED_TABLES.json": json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "ALL_SEED_DELTAS.csv": csv_text(result["paired_seed_deltas"], paired_fields),
        "GPU_SEED_COSTS.csv": csv_text(result["gpu_seed_costs"], cost_fields)}, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    require(INPUT.is_file() and not INPUT.is_symlink(), "Completed fixed comparison input is missing or symbolic")
    raw = INPUT.read_bytes()
    source = json.loads(raw)
    provenance = {"input_path": str(INPUT), "input_sha256": digest(raw), "input_bytes": len(raw),
        "builder_path": str(Path(__file__).resolve()), "builder_sha256": digest(Path(__file__).read_bytes()),
        "created_at": datetime.now(timezone.utc).isoformat(), "remote_calls": 0, "inference_calls": 0,
        "comparison_reexecuted": False}
    documents, result = build_documents(source, provenance)
    require(all(isinstance(value, str) and bool(value.strip()) for value in documents.values()), "Empty report output")
    require(digest(INPUT.read_bytes()) == provenance["input_sha256"], "Input changed during local derivation")
    if args.check_only:
        print(json.dumps({"status": "ready_no_outputs_written", "input_sha256": provenance["input_sha256"], "counts": result["counts"]}, ensure_ascii=False))
        return 0
    require(OUTPUT.is_dir() and not OUTPUT.is_symlink(), "Output directory is missing or symbolic")
    require(not any((OUTPUT / name).exists() or (OUTPUT / name).is_symlink() for name in OUTPUT_NAMES), "Existing output must be preserved")
    with ExitStack() as stack:
        streams = {name: stack.enter_context((OUTPUT / name).open("x", encoding="utf-8", newline="")) for name in OUTPUT_NAMES}
        for name in OUTPUT_NAMES:
            streams[name].write(documents[name])
    print(json.dumps({"status": "complete", "input_sha256": provenance["input_sha256"], "output_files": [str(OUTPUT / name) for name in OUTPUT_NAMES], "counts": result["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
