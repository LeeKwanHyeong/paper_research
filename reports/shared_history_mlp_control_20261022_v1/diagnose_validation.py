"""Local-only descriptive audit of frozen A/B/D Validation predictions.

Never imports model code, opens checkpoints or Test artifacts, or contacts a host.
Run without arguments to create exclusive outputs; --check-only recomputes them.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "search_artifacts/shared_component_ablation_20261022_v1"
DEST = Path(__file__).resolve().parent
ARMS = {"full": "A", "no_prototype": "B", "encoder_only": "D"}
SOURCES = (
    "models/TPPs/CountAwareTPP.py", "models/TPPs/positive_integer_time.py",
    "paper/scripts/count_aware_tpp_backbone/core.py",
    "paper/scripts/run_shared_component_ablation.py",
    "paper/scripts/run_titantpp_shared_history_raw9.py",
    "paper/scripts/run_titantpp_mlp_cap36.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def distribution(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    k = math.ceil(len(x) * .01)
    top = np.sort(x)[-k:]
    total, positive = float(x.sum()), float(x[x > 0].sum())
    return {
        "count": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
        "p90": float(np.quantile(x, .90)), "p99": float(np.quantile(x, .99)),
        "min": float(x.min()), "max": float(x.max()), "sum": total,
        "negative_count": int((x < 0).sum()), "negative_fraction": float((x < 0).mean()),
        "negative_sum": float(x[x < 0].sum()), "positive_sum": positive,
        "zero_count": int((x == 0).sum()), "below_1e_minus_6_fraction": float((x < 1e-6).mean()),
        "top_one_percent_count": k, "top_one_percent_sum": float(top.sum()),
        "top_one_percent_share_of_net_sum": float(top.sum()) / total if total > 0 else None,
        "top_one_percent_share_of_positive_sum": float(top[top > 0].sum()) / positive if positive else None,
    }


def errors(q, p):
    e = p - q
    return {"qty_rmse": float(np.sqrt(np.mean(e * e))), "qty_mae": float(np.mean(np.abs(e)))}


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analysis():
    contract_path = BUNDLE / "execution_contract.json"
    spec_path = ROOT / "reports/shared_component_ablation_20261022_v1/SCIENCE_SPEC.json"
    assert sha(contract_path) == "462996103c87523785965f733404d9c56f6e03e73418aa38f593838c4a6c696a"
    assert sha(spec_path) == "3c3aaf9ee9fd8c20b0705bc4d45a4604dd6425810728656ca8105f498f7ad2ce"
    contract = json.loads(contract_path.read_text())
    spec = json.loads(spec_path.read_text())
    source_refs = [reference(ROOT / name) for name in SOURCES]
    for item in source_refs:
        assert item["sha256"] == contract["source"]["files"][item["path"]], item
    datasets = {x["dataset_id"]: x for x in contract["datasets"]}
    jobs = [j for j in contract["jobs"] if j["arm"] in ARMS]
    assert len(jobs) == 27
    runs, cache, original_refs = [], {}, []
    for job in jobs:
        dataset, arm, seed, host = (job[k] for k in ("dataset", "arm", "seed", "host"))
        base = BUNDLE / "retrieval/fit_validation" / host / "originals"
        receipt_path = base / "run" / job["id"] / "validation_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["job_id"] == job["id"] and receipt["host"] == host
        assert receipt["held_out_test_evaluated"] is False
        relative = Path(receipt["prediction_path"]).relative_to(contract["hosts"][host]["root"])
        assert relative == Path("run") / job["id"] / "validation_predictions.jsonl.gz"
        prediction = base / relative
        assert sha(prediction) == receipt["files"][str(relative)]
        refs = {"receipt": reference(receipt_path), "predictions": reference(prediction)}
        original_refs.extend(refs.values())
        with gzip.open(prediction, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert rows and all(r["split"] == "validation" and r["target_index"] == i for i, r in enumerate(rows))
        assert all(set(r) == set(rows[0]) for r in rows)
        q, p, nll, hist = [np.asarray([r[key] for r in rows], dtype=np.float64) for key in
                          ("raw_quantity", "predicted_raw_quantity", "time_nll", "history_length")]
        assert all(np.isfinite(x).all() for x in (q, p, nll, hist)) and (q >= 0).all()
        expected = datasets[dataset]["inherited_data_identity"]["populations"]["validation"]
        assert receipt["target_identity"] == expected
        assert len(rows) == receipt["prediction_rows"] == expected["target_count"]
        qsha = hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + q.astype("<f8").tobytes()).hexdigest()
        assert qsha == expected["target_quantity_sha256"]
        target_keys = [(r["entity_id"], r["target_seq"], r["raw_quantity"], r["history_length"]) for r in rows]
        summary = distribution(nll)
        quantity = errors(q, p)
        cutoff = datasets[dataset]["quantity_boundaries_all_train_rows"][-1]
        tail = q > cutoff
        tail_metrics = {**errors(q[tail], p[tail]), "time_nll": float(nll[tail].mean()), "count": int(tail.sum())}
        for group, computed in (("overall", {**quantity, "time_nll": summary["mean"]}), ("tail", tail_metrics)):
            for key in ("qty_rmse", "qty_mae", "time_nll"):
                assert math.isclose(computed[key], receipt["metrics"][group][key], rel_tol=1e-10, abs_tol=1e-10), (job["id"], group, key)
        order = np.argsort(-nll, kind="stable")
        strata = {}
        for low, high in ((1, 8), (8, 32), (32, 128), (128, None)):
            mask = (hist >= low) & ((hist < high) if high is not None else True)
            if mask.any():
                strata[f"{low} <= history < {high}"] = {"count": int(mask.sum()), "time_nll": float(nll[mask].mean()), **errors(q[mask], p[mask])}
        run = {"job_id": job["id"], "dataset": dataset, "arm": ARMS[arm], "seed": seed, "host": host,
               "split": "validation", "source": refs, "target_identity": expected,
               "selected_epoch": receipt["selected_epoch"], "checkpoint_sha256": receipt["checkpoint_sha256"],
               "state_sha256": receipt["state_sha256"], "prediction_fields": sorted(rows[0]),
               "time_nll": summary, "quantity": quantity, "strict_large_quantity_cutoff": cutoff,
               "large_quantity": tail_metrics, "history_strata": strata,
               "pearson_nll_vs_raw_quantity": correlation(nll, q),
               "pearson_nll_vs_absolute_quantity_error": correlation(nll, np.abs(p-q)),
               "largest_nll_examples": [rows[int(i)] for i in order[:5]]}
        runs.append(run)
        cache[(dataset, ARMS[arm], seed)] = (nll, q, p, target_keys, run)
    comparisons = []
    for dataset in datasets:
        for seed in (42, 52, 62):
            for candidate, baseline in (("B", "A"), ("B", "D"), ("A", "D")):
                n, q, p, ids, run = cache[dataset, candidate, seed]
                bn, bq, bp, bids, brun = cache[dataset, baseline, seed]
                assert ids == bids and np.array_equal(q, bq)
                d = n - bn
                k = math.ceil(len(d) * .01)
                top_delta = np.argsort(-d, kind="stable")[:k]
                top_nll = np.argsort(-n, kind="stable")[:k]
                rest = np.ones(len(d), dtype=bool); rest[top_nll] = False
                comparisons.append({
                    "dataset": dataset, "seed": seed, "split": "validation", "candidate": candidate, "baseline": baseline,
                    "run_refs": [run["job_id"], brun["job_id"]],
                    "delta_time_nll": float(d.mean()), "relative_time_nll_percent": float(d.mean()/bn.mean()*100),
                    "delta_qty_rmse": run["quantity"]["qty_rmse"] - brun["quantity"]["qty_rmse"],
                    "delta_qty_mae": run["quantity"]["qty_mae"] - brun["quantity"]["qty_mae"],
                    "time_worsened": bool(d.mean() > 0), "target_fraction_time_worsened": float((d > 0).mean()),
                    "paired_nll_delta_median": float(np.median(d)), "paired_nll_delta_p90": float(np.quantile(d, .90)),
                    "largest_positive_one_percent_delta_contribution_to_mean": float(d[top_delta].sum()/len(d)),
                    "candidate_top_one_percent_delta_contribution_to_mean": float(d[top_nll].sum()/len(d)),
                    "remaining_target_delta_contribution_to_mean": float(d[rest].sum()/len(d)),
                })
    aggregate = []
    for dataset in datasets:
        for arm in ("A", "B", "D"):
            values = [r for r in runs if r["dataset"] == dataset and r["arm"] == arm]
            item = {"dataset": dataset, "arm": arm, "split": "validation", "run_refs": [r["job_id"] for r in values]}
            for field in ("mean", "median", "p90", "p99", "max", "top_one_percent_share_of_net_sum", "negative_fraction"):
                a = np.array([r["time_nll"][field] for r in values])
                item["nll_" + field] = {"mean": float(a.mean()), "sample_sd_ddof1": float(a.std(ddof=1))}
            for field in ("qty_rmse", "qty_mae"):
                a = np.array([r["quantity"][field] for r in values])
                item[field] = {"mean": float(a.mean()), "sample_sd_ddof1": float(a.std(ddof=1))}
            aggregate.append(item)
    for ref in original_refs + source_refs:
        assert sha(ROOT / ref["path"]) == ref["sha256"], ref
    return {"status": "PASS_VALIDATION_ONLY_DIAGNOSTIC", "split": "validation", "runs": runs,
            "aggregate": aggregate, "paired_comparisons": comparisons,
            "provenance": {"contract": reference(contract_path), "science_spec": reference(spec_path),
                           "source_files": source_refs, "script": reference(Path(__file__)),
                           "sources_match_frozen_contract": True, "input_files_rehashed_after_analysis": True},
            "method": {"quantiles": "numpy linear; computed separately for each seed", "top_one_percent": "ceil(0.01 * target_count), largest recorded NLL", "aggregation": "mean and sample SD ddof=1 across three seeds, never pooled targets", "receipt_metric_tolerance": "rel=1e-10, abs=1e-10", "paired_delta": "candidate minus baseline; positive is worse"},
            "definition": {"metric": "negative log probability mass of positive integer observed duration", "law": "T=time_scale*exp(N(mu(h),sigma(h))); D=max(1,round(T)); all datasets top_code=None", "bins": "D=1: (0,1.5]; D>=2: (D-0.5,D+0.5]", "sigma": "0.001 + softplus(hidden-conditional raw scale)", "precision": "float64 probability and conditional head calculations", "clamp": "no probability clipping, intercept cap, or wd clamp on this configured path; unrepresentable probabilities raise", "normalization": "CDF bin mass over raw duration unit; time_scale handled inside standardized log boundaries; no separate density Jacobian", "datasets": {k: d["model"] for k, d in datasets.items()}, "checkpoint_monitor": spec["training"]["monitor"]},
            "limitations": ["Stored predictions contain no true dt, predicted duration, conditional mu or sigma; no duration association or head failure cause can be identified here.", "Nonnegative integer-mass NLL should not be interpreted as the historical clamped RMTPP score or continuous lognormal density.", "Observed quantity/time tradeoffs do not identify causal effects of representation, optimization or checkpoint selection.", "Three seed descriptive SD is not a significance or equivalence test.", "No Test artifacts, checkpoints, GPU, model inference or remote processes accessed."]}


def markdown(result):
    lines = ["# A/B/D Validation 시간·수량 진단", "", "상태: 완료. 기존 Validation 27개 예측을 독립 집계했고 원본 receipt와 수량 RMSE·MAE·TimeNLL이 일치한다. 원본은 변경하지 않았다.", "", "## 시간 점수의 실제 정의", "", "이번 TimeNLL은 양의 정수 시간 간격에 부여한 확률의 −log다. 이분산 로그정규분포를 정수 관측 구간으로 적분한다. 첫 구간은 (0,1.5], 이후 구간은 (d−0.5,d+0.5]이며 세 데이터 모두 top_code=None이다. sigma에는 0.001 하한이 있지만 확률 clipping, intercept cap, wd clamp는 이 실행 경로에 없다. 정규화된 확률질량이므로 음수 NLL은 허용되지 않는다. 과거 clamped RMTPP 점수와 구분해야 한다.", "", "## 데이터별 요약", "", "평균±표본 SD(ddof=1), 공통 3seed. 분위수는 각 seed 내부에서 계산한 뒤 요약한다. 상위 1% 비중은 가장 큰 ceil(N×0.01)개 NLL의 합/전체 NLL합이다.", "", "| 데이터 | 모델 | 수량 RMSE | TimeNLL | median NLL | p99 NLL | 상위 1% 비중 |", "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(x, percent=False):
        scale = 100 if percent else 1
        return f"{x['mean']*scale:.6g} ± {x['sample_sd_ddof1']*scale:.3g}" + ("%" if percent else "")
    for r in result["aggregate"]:
        lines.append(f"| {r['dataset']} | {r['arm']} | {fmt(r['qty_rmse'])} | {fmt(r['nll_mean'])} | {fmt(r['nll_median'])} | {fmt(r['nll_p99'])} | {fmt(r['nll_top_one_percent_share_of_net_sum'], True)} |")
    lines += ["", "## 모든 seed 분포", "", "| 데이터/모델/seed | N | 평균 | median | p90 | p99 | max | 상위1% 합 | 비중 | 음수 개수 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["runs"]:
        x = r["time_nll"]
        lines.append(f"| {r['dataset']}/{r['arm']}/{r['seed']} | {x['count']} | {x['mean']:.6g} | {x['median']:.6g} | {x['p90']:.6g} | {x['p99']:.6g} | {x['max']:.6g} | {x['top_one_percent_sum']:.6g} | {100*x['top_one_percent_share_of_net_sum']:.3f}% | {x['negative_count']} |")
    lines += ["", "## 동일 target의 비교", "", "차이는 후보−기준이다. 양수는 악화다. 후보 상위1% 기여와 나머지 기여의 합은 전체 평균 NLL 차이다. 이는 진단용 분해이며 해당 target을 평가에서 제거하지 않는다.", "", "| 데이터 | 비교 | seed | ΔRMSE | ΔMAE | ΔNLL | NLL 악화 target | 후보 상위1% 기여 | 나머지 기여 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["paired_comparisons"]:
        lines.append(f"| {r['dataset']} | {r['candidate']}−{r['baseline']} | {r['seed']} | {r['delta_qty_rmse']:+.6g} | {r['delta_qty_mae']:+.6g} | {r['delta_time_nll']:+.6g} | {100*r['target_fraction_time_worsened']:.2f}% | {r['candidate_top_one_percent_delta_contribution_to_mean']:+.6g} | {r['remaining_target_delta_contribution_to_mean']:+.6g} |")
    lines += ["", "## 해석 범위", "", "수량 RMSE로 checkpoint를 선택했으므로 시간 NLL이 함께 최소가 되는 조건은 보장하지 않는다. 수량 개선과 시간 악화가 함께 나타나는 seed는 교환관계의 관측 증거이며, 선택 기준이나 모델 구조가 원인이라는 인과 증거는 아니다. 저장 예측에는 실제 dt·예측 duration·mu·sigma가 없어 시간 간격별 분석과 sigma 이상 여부는 미확인이다. 새 추론이나 모델 변경은 수행하지 않았다.", "", "## 원본 증거", "", "각 수치는 아래 Validation 예측과 receipt에 연결되며, 상세 SHA·선택 epoch·checkpoint SHA·분해는 VALIDATION_DIAGNOSTIC.json에 있다.", ""]
    for r in result["runs"]:
        lines += [f"- {r['job_id']} (split=validation):", f"  - 예측 `{r['source']['predictions']['path']}` SHA256 `{r['source']['predictions']['sha256']}`", f"  - receipt `{r['source']['receipt']['path']}` SHA256 `{r['source']['receipt']['sha256']}`"]
    lines += ["", "## 검증 및 남은 작업", "", "- [x] 27개 원본 예측 SHA·split·행수·target 수량 SHA 확인", "- [x] 같은 seed·데이터의 A/B/D target 및 이력 길이 일치 확인", "- [x] 전체·큰수량 RMSE/MAE/TimeNLL 원 receipt 재현", "- [x] 실제 시간 점수 경로 6개 소스 SHA를 동결 실행 계약과 대조", "- [x] 분석 뒤 모든 읽은 예측·receipt·소스 SHA 재확인", "- [ ] 별도 단순 보정기 실험과 과학적 기여 판단은 상위 작업에서 진행", "", "학습 소스 수정이나 버그 수정은 하지 않았으므로 회귀 테스트 대상 변경은 없다. 진단 스크립트의 --check-only는 동일 원본에서 두 산출물을 다시 계산해 바이트 일치를 검사한다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--check-only", action="store_true"); args = parser.parse_args()
    result = analysis()
    outputs = {DEST / "VALIDATION_DIAGNOSTIC.json": json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", DEST / "VALIDATION_DIAGNOSTIC.md": markdown(result)}
    if args.check_only:
        for path, content in outputs.items():
            assert path.read_text() == content, path
    else:
        assert not any(p.exists() for p in outputs), "Existing diagnostic preserved; use --check-only"
        for path, content in outputs.items():
            with path.open("x") as stream: stream.write(content)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "check_only": args.check_only, "outputs": {str(p.relative_to(ROOT)): sha(p) for p in outputs}}, indent=2))


if __name__ == "__main__":
    main()
