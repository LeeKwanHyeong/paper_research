"""Describe fixed CNN errors using saved, pre-target Train/Validation histories.

No model loading, dataset loading, remote operation, or held-out input is used.
The implementation support rule is exploratory and never a performance claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import polars as pl

from paper.scripts import analyze_titantpp_cnn_memory_sensitivity as parent

PARENT_CONTRACT = "d51ab06365a7165e00cc18849b84caba4ee2dd8af6a36aee84a5c1d62e01fab3"
PARENT_RECEIPT = "cde075132699bc4100688753b85a8cbc3a8f5b1fe0523aff7b7c98a4ac4e07b2"
PARENT_MANIFEST = "9b7044490727a42add4b9e05538fdf315137f0a956e25d8b1422f59a37430819"
PARENT_ANALYZER = "8116d394ac756e81ccf70e5f11324ec6e48af405951fecfdc1af7df10b6a6159"
SPLITS = ("train", "validation")
FEATURES = ("absolute_observed_log_change", "last_log_deviation", "history_log_std", "last_quantity")
COLUMNS = ("entity_id", "seq", "dataset", "seed", "split", "history_length", "last_quantity",
           "mean_quantity", "last_log_deviation", "history_log_std", "last_observed_log_change",
           "previous_observation_available", "raw_quantity", "prediction_1", "time_nll_1")
DISPLAY = {"yellow_trip_hourly": "Taxi", "raf_spare_parts": "RAF", "intermittent_frozen_5000": "Intermittent"}


def checked_split(split):
    parent.need(split in SPLITS, "Only train/validation predictions are allowed")


def annotate(frame):
    """Only past-observation columns define the strata; outcomes are scores only."""
    error = pl.col("prediction_1") - pl.col("raw_quantity")
    return frame.with_columns(
        pl.col("last_observed_log_change").abs().alias("absolute_observed_log_change"),
        pl.when(~pl.col("previous_observation_available")).then(pl.lit("unavailable"))
        .when(pl.col("last_observed_log_change") < 0).then(pl.lit("falling"))
        .when(pl.col("last_observed_log_change") > 0).then(pl.lit("rising"))
        .otherwise(pl.lit("flat")).alias("observed_direction"),
        error.alias("error"))


def cuts_from_train(frame):
    parent.need(set(frame["split"].unique()) == {"train"}, "Cuts require Train rows exclusively")
    return {name: [frame.filter(pl.col("previous_observation_available") if name == FEATURES[0]
                               else pl.lit(True))[name].quantile(p, interpolation="linear")
                   for p in (.25, .5, .75)] for name in FEATURES}


def quartile(name, cuts):
    if any(v is None for v in cuts):
        return pl.lit(-1)
    unavailable = ~pl.col("previous_observation_available") if name == FEATURES[0] else pl.lit(False)
    return (pl.when(unavailable).then(-1).when(pl.col(name) <= cuts[0]).then(0)
            .when(pl.col(name) <= cuts[1]).then(1).when(pl.col(name) <= cuts[2]).then(2).otherwise(3))


def metrics(frame, total_count, total_sse):
    n = frame.height
    base = {"count": n, "entity_count": frame["entity_id"].n_unique(),
            "population_share": n / total_count, "sse": 0., "mse_contribution": 0.,
            "sse_share": None if total_sse == 0 else 0.}
    if not n:
        return {**base, **dict.fromkeys(("rmse", "mae", "bias", "over_rate", "under_rate", "time_nll"))}
    e = frame["error"]
    sse = float((e * e).sum())
    return {**base, "sse": sse, "rmse": math.sqrt(sse / n), "mae": float(e.abs().mean()),
            "bias": float(e.mean()), "over_rate": float((e > 0).mean()),
            "under_rate": float((e < 0).mean()), "time_nll": float(frame["time_nll_1"].mean()),
            "mse_contribution": sse / total_count, "sse_share": sse / total_sse if total_sse else None}


def cells(frame, cuts, threshold):
    checked_split(frame["split"].unique().item())
    n, sse = frame.height, float(frame["error"].pow(2).sum())
    output = [{"kind": "overall", "feature": "all", "group": "all", **metrics(frame, n, sse)},
              {"kind": "target_tail_descriptive_only", "feature": "raw_quantity", "group": str(threshold),
               **metrics(frame.filter(pl.col("raw_quantity") > threshold), n, sse)}]
    for direction in ("falling", "flat", "rising", "unavailable"):
        output.append({"kind": "direction", "feature": "observed_direction", "group": direction,
                       **metrics(frame.filter(pl.col("observed_direction") == direction), n, sse)})
    for name in FEATURES:
        f = frame.with_columns(quartile(name, cuts[name]).alias("quartile"))
        for q in (-1, 0, 1, 2, 3):
            output.append({"kind": "feature_quartile", "feature": name, "group": str(q),
                           **metrics(f.filter(pl.col("quartile") == q), n, sse)})
    f = frame.with_columns(quartile("last_quantity", cuts["last_quantity"]).alias("level_quartile"))
    for q, direction in itertools.product(range(4), ("falling", "flat", "rising", "unavailable")):
        output.append({"kind": "level_direction", "feature": "last_quantity", "level_quartile": q,
                       "group": direction, **metrics(f.filter((pl.col("level_quartile") == q)
                                                               & (pl.col("observed_direction") == direction)), n, sse)})
    return output


def implementation_support(rows):
    """Primary Taxi rule, fixed before data analysis; never silently loosen it."""
    contrasts, supports = [], []
    for q in range(4):
        repeated = defaultdict(list)
        for seed in parent.SEEDS:
            signs, good = [], True
            for split in SPLITS:
                chosen = {r["group"]: r for r in rows if r["dataset"] == "yellow_trip_hourly"
                          and r["seed"] == seed and r["split"] == split and r["kind"] == "level_direction"
                          and r["level_quartile"] == q and r["group"] in ("falling", "rising")}
                parent.need(set(chosen) == {"falling", "rising"}, "Missing primary comparison cell")
                a, b = chosen["falling"], chosen["rising"]
                enough = min(a["count"], b["count"]) >= 50 and min(a["entity_count"], b["entity_count"]) >= 5
                if a["count"] and b["count"]:
                    delta = a["bias"] - b["bias"]
                    pooled = (a["mae"]*a["count"] + b["mae"]*b["count"]) / (a["count"]+b["count"])
                    ratio = abs(delta)/pooled if pooled > 0 else None
                else:
                    delta, ratio = None, None
                eligible = enough and ratio is not None and ratio >= .1
                signs.append(1 if delta is not None and delta > 0 else -1 if delta is not None and delta < 0 else 0)
                good = good and eligible
                contrasts.append({"level_quartile": q, "seed": seed, "split": split,
                                  "falling_minus_rising_bias": delta, "absolute_contrast_to_pooled_mae": ratio,
                                  "enough_rows_entities": enough, "eligible": eligible})
            if good and signs[0] == signs[1] and signs[0] != 0:
                repeated[signs[0]].append(seed)
        supports.append({"level_quartile": q, "negative_contrast_seeds": repeated[-1],
                         "positive_contrast_seeds": repeated[1],
                         "supported": max(len(repeated[-1]), len(repeated[1])) >= 2})
    return {"implementation_supported": any(r["supported"] for r in supports),
            "training_authorized": False, "performance_improvement_demonstrated": False,
            "causal_mechanism_identified": False, "primary_dataset": "Taxi",
            "rule": "Within the same Train last-quantity quartile: >=50 rows and >=5 entities per direction; |bias_fall-bias_rise| >=0.1 pooled MAE; same sign in Train/Validation and >=2 seeds",
            "level_support": supports, "contrasts": contrasts}


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["split"], r["kind"], r["feature"], r["group"], r.get("level_quartile"))].append(r)
    out = []
    for (dataset, split, kind, feature, group, level), values in groups.items():
        parent.need(sorted(v["seed"] for v in values) == list(parent.SEEDS), "Duplicate/missing seed")
        parent.need(len({(v["count"],v["entity_count"]) for v in values}) == 1, "Seed populations differ")
        out.append(dict(dataset=dataset, split=split, kind=kind, feature=feature, group=group,
                        level_quartile=level, count=values[0]["count"], entity_count=values[0]["entity_count"],
                        metrics={m: parent.average_sd([v[m] for v in values]) for m in
                                 ("rmse", "mae", "bias", "over_rate", "under_rate", "sse_share", "time_nll")}))
    return out


def markdown(result):
    lines = ["# CNN 관측 이력별 Train·Validation 오류", "", "Material Passport: experiment-agent / local analysis / v1; fixed predictions only.",
             "", "새 추론·학습·Test 열람·원격 작업 없음. alpha=1 기존 CNN만 사용했다.", "",
             "## Validation: 마지막 두 관측의 방향", "",
             "값은 3seed 평균 ± 표본SD. bias=예측−실제. 방향은 미래 target이 아닌 관측 이력으로 정의했다.", "",
             "| 데이터셋 | 관측 방향 | 행/개체 | RMSE | MAE | bias |", "|---|---|---:|---:|---:|---:|"]
    for r in result["three_seed"]:
        if r["split"] != "validation" or r["kind"] != "direction":
            continue
        def fmt(m):
            v=r["metrics"][m]
            return "N/A" if v["mean"] is None else f"{v['mean']:.4f} ± {v['sample_sd']:.4f}"
        lines.append(f"| {DISPLAY[r['dataset']]} | {r['group']} | {r['count']}/{r['entity_count']} | {fmt('rmse')} | {fmt('mae')} | {fmt('bias')} |")
    lines += ["", "## 같은 과거 수량 수준 안의 비교", "", "Train에서 정한 마지막 관측 수량 사분위를 Validation에 그대로 적용했다.",
              "", "| Taxi 구간 | Train·Validation에서 음의 bias 차이가 반복된 seed | 양의 차이가 반복된 seed | 구현 근거 |", "|---|---|---|---|"]
    for r in result["support"]["level_support"]:
        lines.append(f"| {r['level_quartile']} | {r['negative_contrast_seeds']} | {r['positive_contrast_seeds']} | {r['supported']} |")
    lines += ["", "차이는 최근 관측 하락−상승의 bias다. signed bias 하나의 개선이 RMSE/MAE 개선을 보장하지 않는다.",
              "반복 여부는 구현을 준비할 탐색적 근거다. CNN 원인 확정·새 모델 성능 개선·통계적 유의성이 아니다.",
              "수량 사분위 안에도 수량·개체·시간 조건 차이가 남아 있다. seed는 같은 데이터의 학습 반복이며 독립적인 표본 복제가 아니다.",
              "관측 간격은 이 artifact에 없어 분석하지 않았다. 다음 target gap을 과거 관측 간격으로 사용하지 않았다.",
              "전체/큰 수량의 여섯 지표, 모든 seed·빈 구간·미확인 조건은 JSON/CSV에 보존했다.",
              "기존 Test를 본 뒤 설계한 탐색적 분석이다. 이번 실행에서 Test를 다시 읽거나 평가하지 않았다."]
    return "\n".join(lines)+"\n"


def analyze(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    parent.need(not output.exists(), "Output exists; refuse overwrite")
    bundle=project/"search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3"
    results=bundle/"retrieval"
    inputs={}
    def pin(path, expected):
        path=Path(path).resolve()
        parent.need(parent.sha(path)==expected, "Input SHA mismatch: "+str(path))
        inputs[str(path.relative_to(project))]=expected
    pin(project/"reports/titantpp_cnn_memory_sensitivity_20261015_v3/FINAL_RECEIPT.json", PARENT_RECEIPT)
    pin(bundle/"contract.json", PARENT_CONTRACT)
    pin(results/"manifest.json", PARENT_MANIFEST)
    pin(Path(parent.__file__), PARENT_ANALYZER)
    receipt=parent.read(project/"reports/titantpp_cnn_memory_sensitivity_20261015_v3/FINAL_RECEIPT.json")
    parent.need(receipt["status"]=="complete" and receipt["validation_replay_pass_count"]==9, "Incomplete parent")
    manifest=parent.read(results/"manifest.json")
    parent.need(manifest["contract_sha256"]==PARENT_CONTRACT, "Manifest contract differs")
    # Manifest is a relative-path -> SHA map, including dispatcher/worker receipts.
    for host in ("5080", "5090"):
        rel=f"runtime/{host}/receipt.json"
        pin(parent.confined(results, rel), manifest["files"][rel])
    contract, contract_sha=parent.verify_cohort(bundle)
    pinned, _=parent.verify_dispatchers(results,contract,contract_sha)
    rows, proof, cuts_rows, six = [], [], [], []
    truth_by_dataset_split={}
    for job in sorted(contract["jobs"],key=lambda j:(j["dataset"],j["seed"])):
        cuts=None
        threshold=contract["datasets"][job["dataset"]]["definition"]["quantity_boundaries_all_train_rows"][-1]
        for split in SPLITS:
            checked_split(split)
            rec, _, evidence=parent.load_predictions(results,contract_sha,job,split,pinned[(job["id"],split)])
            directory=parent.confined(results,f"results/{job['id']}/{split}")
            pin(directory/"receipt.json",evidence["receipt_sha256"])
            paths=[]
            for part in evidence["parts"]:
                path=parent.confined(directory/"predictions",part["path"]); pin(path,part["sha256"]); paths.append(path)
            frame=pl.scan_parquet(paths).select(COLUMNS).collect(engine="streaming")
            parent.need(set(frame["split"].unique())=={split} and set(frame["dataset"].unique())=={job["dataset"]}
                        and set(frame["seed"].unique())=={job["seed"]}, "Row scope differs")
            parent.need(frame.select(pl.struct("entity_id","seq").n_unique()).item()==frame.height, "Duplicate targets")
            numeric=[c for c in COLUMNS if c not in ("entity_id","dataset","split","previous_observation_available")]
            parent.need(frame.select(pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in numeric]).all()).item(), "Missing/nonfinite rows")
            parent.need(frame["entity_id"].null_count()==0 and frame["previous_observation_available"].null_count()==0, "Missing observed identity")
            truth=frame.select([c for c in COLUMNS if c not in ("seed","prediction_1","time_nll_1")]).sort(["entity_id","seq"])
            key=job["dataset"],split
            if key in truth_by_dataset_split:
                parent.need(truth.equals(truth_by_dataset_split[key]),"History/target differs between seeds")
            else: truth_by_dataset_split[key]=truth
            frame=annotate(frame)
            if split=="train":
                cuts=cuts_from_train(frame); cuts_rows.append({"dataset":job["dataset"],"seed":job["seed"],"cuts":cuts})
            found=cells(frame,cuts,threshold)
            overall=found[0]; expected=rec["metrics"]["1.0"]
            for name,keyname in (("rmse","qty_rmse"),("mae","qty_mae"),("time_nll","time_nll")):
                parent.need(math.isclose(overall[name],expected[keyname],rel_tol=1e-8,abs_tol=1e-8),"Baseline metric mismatch")
            rows += [dict(dataset=job["dataset"],seed=job["seed"],split=split,**r) for r in found]
            six.append(dict(dataset=job["dataset"],seed=job["seed"],split=split,selected_epoch=rec["selected_epoch"],
                            checkpoint_sha256=rec["checkpoint_sha256"],state_sha256=rec["state_sha256"],host=job["host"],
                            metrics=parent.six_metrics(expected)))
            proof.append(evidence)
    result=dict(schema="cnn_observed_history_errors_v1",status="complete",created_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                allowed_splits=list(SPLITS),new_training=False,new_inference=False,test_read=False,remote_operations=0,
                diagnostic_model_count=9,baseline_predictions_only=True,train_cuts=cuts_rows,seed_cells=rows,
                three_seed=aggregate(rows),support=implementation_support(rows),six_metrics=six,prediction_evidence=proof,
                inputs=inputs,analyzer_sha256=parent.sha(__file__))
    for path,expected in inputs.items(): pin(project/path,expected)
    output.mkdir(parents=True,exist_ok=False)
    with (output/"ANALYSIS.json").open("x") as stream: json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    (output/"ANALYSIS.md").write_text(markdown(result))
    pl.DataFrame(rows).write_csv(output/"SEED_CELLS.csv")
    with (output/"SHA256.json").open("x") as stream:
        json.dump({p.name:parent.sha(p) for p in output.iterdir() if p.name!="SHA256.json"},stream,sort_keys=True,indent=2)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    r=analyze(args.project,args.output)
    print(json.dumps({"status":r["status"],"implementation_supported":r["support"]["implementation_supported"],"output":str(args.output.resolve())}))
