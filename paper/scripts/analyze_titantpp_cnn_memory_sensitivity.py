"""Audit and summarize frozen CNN memory interventions on Train/Validation only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

import polars as pl


SCHEMA = "titantpp_cnn_memory_sensitivity_v1"
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
SPLITS = ("train", "validation")
ALPHAS = (1.0, 0.5, 0.25, 0.0)
FEATURES = ("history_log_std", "last_log_deviation", "last_observed_log_change",
            "memory_to_encoded_norm", "cosine_h_memory", "topk_similarity_mean",
            "previous_topk_overlap")
METRICS = ("qty_rmse", "qty_mae", "tail_qty_rmse", "tail_qty_mae", "time_nll", "tail_time_nll")
FIXED_RULE = "At least one alpha<1 lowers dataset mean Validation RMSE by >=1% and improves >=2/3 seeds; all six metrics and all datasets retained"
ADAPTIVE_RULE = "Additionally an observed-feature stratum has attenuation benefit and another has harm in >=2/3 Validation seeds using Train quartile cuts; no target-defined gate"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def confined(root, relative):
    rel = Path(relative)
    need(not rel.is_absolute() and ".." not in rel.parts, "Unconfined evidence path")
    path = root / rel
    need(path.resolve().is_relative_to(root.resolve()), "Evidence escapes collection root")
    for component in [path, *path.parents]:
        if component == root:
            break
        need(not component.is_symlink(), "Evidence symlink is not permitted")
    return path


def average_sd(values):
    values = list(values)
    need(len(values) == 3 and all(v is None or math.isfinite(v) for v in values),
         "A three-seed summary requires three finite or explicitly undefined values")
    if any(value is None for value in values):
        return {"mean": None, "sample_sd": None, "seed_count": 3,
                "available_seed_count": sum(value is not None for value in values),
                "reason": "At least one seed has no eligible targets; no partial-seed mean is reported."}
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "seed_count": 3}


def six_metrics(cell):
    return {"qty_rmse": cell["qty_rmse"], "qty_mae": cell["qty_mae"],
            "tail_qty_rmse": cell["tail"]["qty_rmse"], "tail_qty_mae": cell["tail"]["qty_mae"],
            "time_nll": cell["time_nll"], "tail_time_nll": cell["tail"]["time_nll"]}


def available(feature):
    if feature in ("last_observed_log_change", "previous_topk_overlap"):
        return pl.col("previous_observation_available")
    if feature == "memory_to_encoded_norm":
        return pl.col("encoded_norm") > 0
    if feature == "cosine_h_memory":
        return (pl.col("encoded_norm") > 0) & (pl.col("memory_norm") > 0)
    return pl.lit(True)


def train_quartiles(frame):
    """Only Train features determine cuts; no outcome enters this calculation."""
    expressions = []
    for feature in FEATURES:
        value = pl.col(feature).filter(available(feature))
        expressions.append(value.len().alias(feature + "__count"))
        for index, probability in enumerate((0.25, 0.5, 0.75)):
            expressions.append(value.quantile(probability, interpolation="linear").alias(f"{feature}__q{index}"))
    row = frame.select(expressions).collect(engine="streaming").row(0, named=True)
    return {feature: {
        "cuts": [row[f"{feature}__q{i}"] for i in range(3)],
        "available_train_count": row[feature + "__count"],
        "interpolation": "linear", "boundary_rule": "(-inf,q25], (q25,q50], (q50,q75], (q75,inf)",
    } for feature in FEATURES}


def quartile_expression(feature, cuts):
    if any(value is None for value in cuts):
        return pl.lit(-1).alias("quartile")
    value = pl.col(feature)
    return (pl.when(~available(feature)).then(pl.lit(-1))
            .when(value <= cuts[0]).then(pl.lit(0))
            .when(value <= cuts[1]).then(pl.lit(1))
            .when(value <= cuts[2]).then(pl.lit(2))
            .otherwise(pl.lit(3)).alias("quartile"))


def strata_statistics(frame, cuts):
    """Paired per-event changes; negative MSE/MAE deltas mean attenuation helps."""
    output = []
    base = pl.col("prediction_1") - pl.col("raw_quantity")
    for feature in FEATURES:
        expressions = [pl.len().alias("count"), base.pow(2).mean().alias("baseline_mse"),
                       base.abs().mean().alias("baseline_mae"), base.mean().alias("baseline_bias")]
        for alpha in ALPHAS[1:]:
            error = pl.col(f"prediction_{alpha:g}") - pl.col("raw_quantity")
            expressions += [(error.pow(2) - base.pow(2)).mean().alias(f"{alpha}:delta_mse"),
                            (error.abs() - base.abs()).mean().alias(f"{alpha}:delta_mae"),
                            (error - base).mean().alias(f"{alpha}:delta_bias")]
        rows = (frame.with_columns(quartile_expression(feature, cuts[feature]["cuts"]))
                .group_by("quartile").agg(expressions).collect(engine="streaming").to_dicts())
        by_bin = {row["quartile"]: row for row in rows}
        for quartile in (-1, 0, 1, 2, 3):
            row = by_bin.get(quartile, {})
            for alpha in ALPHAS[1:]:
                output.append({"feature": feature, "quartile": quartile, "alpha": alpha,
                    "count": row.get("count", 0), "baseline_mse": row.get("baseline_mse"),
                    "baseline_mae": row.get("baseline_mae"), "baseline_bias": row.get("baseline_bias"),
                    **{metric: row.get(f"{alpha}:{metric}") for metric in ("delta_mse", "delta_mae", "delta_bias")}})
    return output


def support_rules(seed_rows, strata):
    fixed, adaptive = [], []
    for dataset in DATASETS:
        reference = {row["seed"]: row for row in seed_rows
                     if row["dataset"] == dataset and row["split"] == "validation" and row["alpha"] == 1}
        need(set(reference) == set(SEEDS), "Incomplete Validation reference seeds")
        baseline_mean = statistics.mean(row["metrics"]["qty_rmse"] for row in reference.values())
        for alpha in ALPHAS[1:]:
            candidates = {row["seed"]: row for row in seed_rows
                          if row["dataset"] == dataset and row["split"] == "validation" and row["alpha"] == alpha}
            need(set(candidates) == set(SEEDS), "Incomplete attenuation seeds")
            candidate_mean = statistics.mean(row["metrics"]["qty_rmse"] for row in candidates.values())
            improved = [seed for seed in SEEDS if candidates[seed]["metrics"]["qty_rmse"] < reference[seed]["metrics"]["qty_rmse"]]
            reduction = (baseline_mean - candidate_mean) / baseline_mean if baseline_mean > 0 else None
            supported = reduction is not None and reduction >= 0.01 and len(improved) >= 2
            fixed.append({"dataset": dataset, "alpha": alpha, "supported": supported,
                          "baseline_mean_rmse": baseline_mean, "attenuated_mean_rmse": candidate_mean,
                          "relative_mean_rmse_reduction": reduction, "improved_seeds": improved})
            for feature in FEATURES:
                cells = {(row["seed"], row["quartile"]): row for row in strata
                         if row["dataset"] == dataset and row["split"] == "validation"
                         and row["alpha"] == alpha and row["feature"] == feature and row["quartile"] >= 0}
                for benefit, harm in itertools.permutations(range(4), 2):
                    matching = []
                    for seed in SEEDS:
                        a, b = cells.get((seed, benefit)), cells.get((seed, harm))
                        if a and b and a["count"] and b["count"] and a["delta_mse"] < 0 < b["delta_mse"]:
                            matching.append(seed)
                    adaptive.append({"dataset": dataset, "alpha": alpha, "feature": feature,
                                     "benefit_quartile": benefit, "harm_quartile": harm,
                                     "matching_seeds": matching, "fixed_support": supported,
                                     "heterogeneity_supported": len(matching) >= 2,
                                     "supported": supported and len(matching) >= 2})
    return {"fixed_attenuation": fixed, "adaptive_stratum_pairs": adaptive,
            "any_fixed_support": any(row["supported"] for row in fixed),
            "any_adaptive_support": any(row["supported"] for row in adaptive),
            "training_automatically_started": False,
            "interpretation": "Exploratory diagnostic support only; no trained gate effect, causal attribution, or independent Test evidence."}


def verify_cohort(bundle):
    path = bundle / "contract.json"
    contract = read(path)
    need(contract["schema"] == SCHEMA and contract["allowed_splits"] == list(SPLITS)
         and contract["new_training"] is False and contract["frozen_selection"] is True,
         "Wrong diagnostic scope")
    need(contract["alphas"] == list(ALPHAS), "Wrong attenuation grid")
    jobs = contract["jobs"]
    need(len(jobs) == len({job["id"] for job in jobs}) == 9, "Expected nine distinct models")
    need({(job["dataset"], job["seed"]) for job in jobs} == set(itertools.product(DATASETS, SEEDS)),
         "Wrong dataset/seed cohort")
    need(contract["analysis"]["fixed_attenuation_support"] == FIXED_RULE
         and contract["analysis"]["adaptive_support"] == ADAPTIVE_RULE,
         "Unrecognized prespecified support rule")
    for job in jobs:
        need(job["arm"] == "titantpp_cnn_gru54_cnn_only", "Only frozen CNN is permitted")
        expected_host = "5090" if job["dataset"] == "intermittent_frozen_5000" and job["seed"] in (42, 52) else "5080"
        need(job["host"] == expected_host, "Native host allocation changed")
        need(job["id"] == f"{job['dataset']}__{job['seed']}__{job['arm']}", "Unexpected job id")
    return contract, sha(path)


def verify_dispatchers(results, contract, contract_sha):
    pinned, evidence = {}, []
    for host in ("5080", "5090"):
        path = confined(results, f"runtime/{host}/receipt.json")
        receipt = read(path)
        jobs = {job["id"] for job in contract["jobs"] if job["host"] == host}
        expected = set(itertools.product(jobs, SPLITS))
        workers = receipt["workers"]
        need(receipt["schema"] == SCHEMA and receipt["status"] == "complete"
             and receipt["host"] == host and receipt["contract_sha256"] == contract_sha,
             "Dispatcher receipt scope/status differs")
        need(receipt["actual_worker_count"] == receipt["actual_worker_exit0_count"] == len(expected)
             and len(workers) == len(expected), "Dispatcher did not observe every worker exit 0")
        need({(w["job_id"], w["split"]) for w in workers} == expected, "Dispatcher worker identity set differs")
        for worker in workers:
            key = worker["job_id"], worker["split"]
            relative = f"results/{key[0]}/{key[1]}/receipt.json"
            need(worker["status"] == "complete" and worker["actual_worker_exit_code"] == 0,
                 "Non-success scientific worker")
            need(worker["receipt"]["path"] == relative, "Dispatcher receipt path differs")
            need(sha(confined(results, relative)) == worker["receipt"]["sha256"], "Worker receipt SHA differs")
            pinned[key] = worker["receipt"]["sha256"]
        evidence.append({"path": str(path.relative_to(results)), "sha256": sha(path), "host": host})
    return pinned, evidence


def load_predictions(results, contract_sha, job, split, expected_sha):
    directory = confined(results, f"results/{job['id']}/{split}")
    path = directory / "receipt.json"
    need(sha(path) == expected_sha, "Receipt SHA changed")
    receipt = read(path)
    original = job["original"]
    need(receipt["schema"] == SCHEMA and receipt["status"] == "complete"
         and receipt["job_id"] == job["id"] and receipt["split"] == split
         and receipt["host"] == job["host"] and receipt["contract_sha256"] == contract_sha,
         "Prediction receipt identity differs")
    need(receipt["parameters_unchanged"] is True and receipt["no_training"] is True
         and receipt["held_out_test_evaluated"] is False, "Frozen inference invariants failed")
    need(receipt["checkpoint_sha256"] == original["checkpoint"]["sha256"]
         and receipt["state_sha256"] == original["checkpoint_state_sha256"]
         and receipt["selected_epoch"] == original["selected_epoch"], "Frozen checkpoint identity differs")
    need(split != "validation" or receipt["baseline_validation_reproduced"] is True,
         "Validation replay gate not passed")
    need(set(receipt["metrics"]) == {str(alpha) for alpha in ALPHAS}, "Metric alpha set differs")
    manifest = receipt["prediction_manifest"]
    parts, paths, rows = manifest["parts"], [], 0
    need(bool(parts) and len(parts) == len({p["path"] for p in parts}), "Empty/duplicate prediction parts")
    for part in parts:
        need(Path(part["path"]).name == part["path"] and part["path"].endswith(".parquet"),
             "Invalid prediction part path")
        file = confined(directory / "predictions", part["path"])
        need(sha(file) == part["sha256"] and file.stat().st_size == part["bytes"], "Prediction part bytes changed")
        count = pl.scan_parquet(file).select(pl.len()).collect(engine="streaming").item()
        need(count == part["rows"], "Prediction part row count differs")
        paths.append(file)
        rows += count
    need(rows == manifest["prediction_rows"] == receipt["population"]["target_count"], "Target coverage differs")
    frame = pl.scan_parquet(paths)
    required = {"entity_id", "seq", "dataset", "seed", "split", "raw_quantity", "previous_observation_available",
                "encoded_norm", "memory_norm", *FEATURES,
                *(f"prediction_{a:g}" for a in ALPHAS), *(f"time_nll_{a:g}" for a in ALPHAS)}
    need(required.issubset(frame.collect_schema().names()), "Prediction schema incomplete")
    return receipt, frame.select(sorted(required)), {
        "job_id": job["id"], "split": split, "receipt_sha256": expected_sha,
        "parts": parts, "target_count": rows,
    }


def verify_values_and_metrics(frame, receipt, job, split, tail_threshold):
    columns = ["raw_quantity", "encoded_norm", "memory_norm", *FEATURES,
               *(f"prediction_{a:g}" for a in ALPHAS), *(f"time_nll_{a:g}" for a in ALPHAS)]
    conditions = [pl.col(name).is_not_null() & pl.col(name).is_finite() for name in columns]
    conditions += [pl.col("dataset") == job["dataset"], pl.col("seed") == job["seed"],
                   pl.col("split") == split, pl.col("raw_quantity") >= 0,
                   pl.col("previous_observation_available").is_not_null(),
                   *(pl.col(f"prediction_{a:g}") >= 0 for a in ALPHAS)]
    expressions = [pl.len().alias("count"), pl.all_horizontal(conditions).fill_null(False).all().alias("valid")]
    tail = pl.col("raw_quantity") > tail_threshold
    expressions.append(tail.sum().alias("tail_count"))
    for alpha in ALPHAS:
        error = pl.col(f"prediction_{alpha:g}") - pl.col("raw_quantity")
        temporal = pl.col(f"time_nll_{alpha:g}")
        for suffix, selected in (("", pl.lit(True)), ("tail_", tail)):
            expressions += [error.filter(selected).pow(2).mean().sqrt().alias(f"{alpha}:{suffix}qty_rmse"),
                            error.filter(selected).abs().mean().alias(f"{alpha}:{suffix}qty_mae"),
                            temporal.filter(selected).mean().alias(f"{alpha}:{suffix}time_nll")]
    row = frame.select(expressions).collect(engine="streaming").row(0, named=True)
    need(row["valid"] is True and row["count"] == receipt["population"]["target_count"],
         "Nonfinite or mislabeled prediction rows")
    for alpha in ALPHAS:
        cell = receipt["metrics"][str(alpha)]
        need(cell["count"] == row["count"] and cell["tail"]["count"] == row["tail_count"], "Metric counts differ")
        need(cell["quantity_boundaries"][-1] == tail_threshold, "Train-fixed tail threshold differs")
        for metric, expected in six_metrics(cell).items():
            actual = row[f"{alpha}:{metric}"]
            need(actual is None and expected is None or actual is not None and expected is not None
                 and math.isfinite(actual) and math.isfinite(expected)
                 and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8),
                 f"Prediction/receipt metric mismatch: {job['id']}/{split}/{alpha}/{metric}")


def aggregate_seed_rows(seed_rows):
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row["dataset"], row["split"], row["alpha"])].append(row)
    summaries = []
    for (dataset, split, alpha), rows in sorted(groups.items()):
        need(sorted(row["seed"] for row in rows) == list(SEEDS), "Repeated/missing seed")
        summaries.append({"dataset": dataset, "split": split, "alpha": alpha,
            "metrics": {metric: average_sd(row["metrics"][metric] for row in rows) for metric in METRICS},
            "paired_delta_alpha_minus_one": {
                metric: average_sd(row["paired_delta_alpha_minus_one"][metric] for row in rows) for metric in METRICS}})
    return summaries


def markdown(summary):
    def number(value, signed=False):
        return "N/A" if value is None else format(value, "+.6g" if signed else ".6g")

    lines = ["# 고정 CNN 메모리 감쇠: Train·Validation 진단", "",
             "선택 checkpoint와 모델 파라미터를 고정하고 추론 시 메모리 반영량만 바꾼 결과입니다. "
             "기존 Test를 확인한 뒤 설계한 탐색적 진단이며, 새 학습 모델의 성능이나 독립적인 일반화 검증이 아닙니다.", "",
             "각 값은 3seed 평균 ± 표본 SD입니다. α=1은 기존 CNN, α<1은 메모리 감쇠입니다.", "",
             "| 데이터셋 | split | α | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["three_seed_summaries"]:
        values = [number(row['metrics'][m]['mean']) + " ± " + number(row['metrics'][m]['sample_sd']) for m in METRICS]
        lines.append(f"| {row['dataset']} | {row['split']} | {row['alpha']:g} | " + " | ".join(values) + " |")
    lines += ["", "## 사전 기준 충족 여부", "",
              "고정 감쇠 근거: Validation 평균 RMSE 1% 이상 감소와 3seed 중 2개 이상 개선을 동시에 요구합니다.", "",
              "| 데이터셋 | α | 평균 RMSE 감소율 | 개선 seed | 기준 충족 |", "|---|---:|---:|---|---|"]
    for row in summary["support"]["fixed_attenuation"]:
        reduction = row["relative_mean_rmse_reduction"]
        lines.append(f"| {row['dataset']} | {row['alpha']:g} | "
                     + (f"{100 * reduction:.3f}%" if reduction is not None else "미정의")
                     + f" | {','.join(map(str, row['improved_seeds'])) or '없음'} | {'충족' if row['supported'] else '미충족'} |")
    matches = [row for row in summary["support"]["adaptive_stratum_pairs"] if row["supported"]]
    lines += ["", "상황별 조절 근거는 위 고정 감쇠 기준을 충족하는 동일 α에서, 같은 관측 특징의 같은 사분위 구간 쌍이 "
              "Validation 2seed 이상에서 각각 MSE 개선과 악화를 보이는지로 판정합니다. "
              "사분위 경계는 각 모델의 Train 특징으로 계산했고 Validation에 그대로 적용했습니다.", "",
              f"조건을 충족한 구간 쌍: {len(matches)}개. 모든 충족·미충족 판정과 구간별 MSE·MAE·편향 차이는 SUMMARY.json에 보존했습니다.", "",
              "이 결과만으로 학습된 gate의 효과나 메모리의 인과적 책임을 확정할 수 없습니다. 여러 특징·구간을 확인한 탐색 결과입니다.", "",
              "## seed별 차이", "", "차이는 감쇠 결과 − α=1이며, RMSE·MAE·TimeNLL 차이가 음수이면 낮아진 것입니다.", "",
              "| 데이터셋 | split | seed | α | ΔRMSE | ΔMAE | Δ큰 수량 RMSE | Δ큰 수량 MAE | ΔTimeNLL | Δ큰 수량 TimeNLL |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["seed_rows"]:
        values = [number(row['paired_delta_alpha_minus_one'][metric], signed=True) for metric in METRICS]
        lines.append(f"| {row['dataset']} | {row['split']} | {row['seed']} | {row['alpha']:g} | " + " | ".join(values) + " |")
    lines += ["", "Intermittent seed42·52는 5090/cuDNN92000, seed62는 5080/cuDNN91900입니다. "
              "실행 시간은 host별로 분리해 JSON에 기록했습니다. 큰 수량 경계는 기존 Train 기준을 유지했습니다.", "",
              "직전 관측이 없는 행의 변화량·검색 중복도와 0 노름으로 정의되지 않는 비율·코사인은 별도 미정의 구간으로 남겼습니다. "
              "Train 사분위 경계가 겹쳐 생긴 빈 구간은 판단 근거로 쓰지 않았습니다.", ""]
    return "\n".join(lines)


def analyze(bundle, results, output):
    bundle, results, output = Path(bundle).resolve(), Path(results).resolve(), Path(output).resolve()
    need(not output.exists(), "Output already exists; overwriting is forbidden")
    contract, contract_sha = verify_cohort(bundle)
    pinned, dispatcher_evidence = verify_dispatchers(results, contract, contract_sha)
    seed_rows, strata, cut_rows, evidence, timings = [], [], [], [], []
    for job in sorted(contract["jobs"], key=lambda j: (j["dataset"], j["seed"])):
        threshold = contract["datasets"][job["dataset"]]["definition"]["quantity_boundaries_all_train_rows"][-1]
        cuts = None
        for split in SPLITS:
            receipt, frame, proof = load_predictions(results, contract_sha, job, split, pinned[(job["id"], split)])
            verify_values_and_metrics(frame, receipt, job, split, threshold)
            if split == "train":
                cuts = train_quartiles(frame)
                cut_rows.append({"job_id": job["id"], "dataset": job["dataset"], "seed": job["seed"], "features": cuts})
            need(cuts is not None, "Train cuts must precede Validation strata")
            strata += [{"dataset": job["dataset"], "seed": job["seed"], "split": split,
                        "job_id": job["id"], **row} for row in strata_statistics(frame, cuts)]
            baseline = six_metrics(receipt["metrics"]["1.0"])
            for alpha in ALPHAS:
                values = six_metrics(receipt["metrics"][str(alpha)])
                need(all(v is not None and math.isfinite(v) for m, v in values.items() if not m.startswith("tail_")),
                     "Overall metrics must be finite")
                need(all(v is None or math.isfinite(v) for v in values.values()), "Nonfinite metric")
                seed_rows.append({"dataset": job["dataset"], "seed": job["seed"], "split": split,
                                  "job_id": job["id"], "host": job["host"], "alpha": alpha,
                                  "selected_epoch": receipt["selected_epoch"], "target_count": receipt["population"]["target_count"],
                                  "metrics": values,
                                  "paired_delta_alpha_minus_one": {
                                      metric: values[metric] - baseline[metric]
                                      if values[metric] is not None and baseline[metric] is not None else None for metric in METRICS}})
            timings.append({"job_id": job["id"], "host": job["host"], "split": split,
                            "inference_seconds_by_alpha": receipt["inference_seconds_by_alpha"]})
            evidence.append(proof)
    need(sha(bundle / "contract.json") == contract_sha, "Contract changed during analysis")
    summary = {"schema": SCHEMA + "_analysis_v1", "status": "complete", "created_unix": time.time(),
               "contract_sha256": contract_sha, "analyzer_sha256": sha(__file__),
               "diagnostic_model_count": 9, "split_count": 2, "seed_count": 3,
               "new_training": False, "held_out_test_evaluated": False, "post_test_exploratory": True,
               "all_unfavorable_results_retained": True, "alpha_one_reference": "unchanged frozen CNN",
               "three_seed_summaries": aggregate_seed_rows(seed_rows), "seed_rows": seed_rows,
               "train_feature_quartiles": cut_rows, "paired_feature_strata": strata,
               "support": support_rules(seed_rows, strata), "timings_by_host": timings,
               "dispatcher_evidence": dispatcher_evidence, "prediction_evidence": evidence,
               "prespecified_analysis_contract": contract["analysis"],
               "availability_policy": "Unavailable previous observation, zero denominator norm, or undefined cosine are kept in quartile -1 and excluded from cuts/support."}
    output.mkdir(parents=True, exist_ok=False)
    json_path, md_path = output / "SUMMARY.json", output / "SUMMARY.md"
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    with md_path.open("x", encoding="utf-8") as stream:
        stream.write(markdown(summary))
    with (output / "SUMMARY_SHA256.json").open("x", encoding="utf-8") as stream:
        json.dump({path.name: sha(path) for path in (json_path, md_path)}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = analyze(args.bundle, args.results, args.output)
    print(json.dumps({"status": summary["status"], "any_fixed_support": summary["support"]["any_fixed_support"],
                      "any_adaptive_support": summary["support"]["any_adaptive_support"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
