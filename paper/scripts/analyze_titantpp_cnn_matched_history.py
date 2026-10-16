"""Outcome-blind coarsened matching of existing CNN Train/Validation predictions."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from paper.scripts import analyze_titantpp_cnn_observed_history as previous

parent = previous.parent
PRIOR_RECEIPT_SHA = "49b045546b7e036db8a056720695ace6df47c52bc1effa5c2d9964be1dfa3d07"
KEYS = ["entity_id", "level_quartile", "log_level_bin", "magnitude_bin"]
FEATURES = ["log_level", "magnitude"]
HISTORY = ["entity_id", "seq", "split", "last_quantity", "last_observed_log_change", "previous_observation_available"]


def match_weights(history, cuts):
    """Accept no outcomes. Match each split separately, with equal cell mass."""
    previous.checked_split(history["split"].unique().item())
    f = history.select(HISTORY).with_columns(
        pl.col("last_quantity").log1p().alias("log_level"),
        pl.col("last_observed_log_change").abs().alias("magnitude"),
        previous.quartile("last_quantity", cuts).alias("level_quartile"),
        pl.when(pl.col("last_observed_log_change") < 0).then(pl.lit("falling"))
          .otherwise(pl.lit("rising")).alias("direction"),
    ).filter(pl.col("previous_observation_available") & (pl.col("magnitude") > 0))
    f = f.with_columns((pl.col("log_level")/.1).floor().cast(pl.Int64).alias("log_level_bin"),
                       (pl.col("magnitude")/.05).floor().cast(pl.Int64).alias("magnitude_bin"))
    counts = f.group_by(KEYS).agg(
        (pl.col("direction") == "falling").sum().alias("nf"),
        (pl.col("direction") == "rising").sum().alias("nr"),
    ).filter((pl.col("nf") > 0) & (pl.col("nr") > 0))
    f = f.join(counts, on=KEYS, how="inner").with_columns(
        (pl.min_horizontal("nf", "nr") /
         pl.when(pl.col("direction") == "falling").then(pl.col("nf")).otherwise(pl.col("nr")))
        .alias("weight"))
    return f.sort(["entity_id", "seq"])


def common_entity_weights(train, validation):
    """Same entity set and equal entity mass across splits, within each quartile."""
    keys = ["entity_id", "level_quartile"]
    common = train.select(keys).unique().join(validation.select(keys).unique(), on=keys, how="inner")
    out = []
    for f in (train, validation):
        f = f.join(common, on=keys, how="inner")
        f = f.with_columns((pl.col("weight") / pl.col("weight").sum().over(keys+["direction"]))
                           .alias("weight"))
        out.append(f.sort(["entity_id", "seq"]))
    return out


def weighted_mean(frame, column):
    return float((frame[column]*frame["weight"]).sum()/frame["weight"].sum())


def describe(frame):
    if not frame.height:
        return dict(rows=0, entities=0, weight_sum=0., ess=0., rmse=None, mae=None, bias=None, time_nll=None)
    mass = float(frame["weight"].sum())
    f = frame.with_columns(pl.col("error").pow(2).alias("squared_error"), pl.col("error").abs().alias("absolute_error"))
    return dict(rows=f.height, entities=f["entity_id"].n_unique(), weight_sum=mass,
                ess=mass**2/float(f["weight"].pow(2).sum()), rmse=math.sqrt(weighted_mean(f,"squared_error")),
                mae=weighted_mean(f,"absolute_error"), bias=weighted_mean(f,"error"),
                time_nll=weighted_mean(f,"time_nll_1"))


def balance(frame):
    out = {}
    for c in FEATURES:
        stats = []
        for d in ("falling", "rising"):
            f = frame.filter(pl.col("direction") == d)
            if not f.height:
                stats.append(None)
                continue
            mean = weighted_mean(f,c)
            var = float(((f[c]-mean).pow(2)*f["weight"]).sum()/f["weight"].sum())
            stats.append((mean,var))
        if None in stats:
            out[c] = dict(mean_difference=None,smd=None)
        else:
            diff = stats[0][0]-stats[1][0]
            sd = math.sqrt((stats[0][1]+stats[1][1])/2)
            out[c] = dict(mean_difference=diff,smd=diff/sd if sd else 0. if abs(diff)<1e-12 else None)
    return out


def contrast(frame, original):
    a,b = [describe(frame.filter(pl.col("direction")==d)) for d in ("falling","rising")]
    if a["rows"] and b["rows"]:
        delta = {m:a[m]-b[m] for m in ("bias","rmse","mae","time_nll")}
        ratio = delta["bias"]/((a["mae"]+b["mae"])/2) if a["mae"]+b["mae"] else None
    else:
        delta = dict.fromkeys(("bias","rmse","mae","time_nll")); ratio=None
    original_directional = original.filter(pl.col("previous_observation_available") & (pl.col("last_observed_log_change")!=0))
    return dict(falling=a,rising=b,falling_minus_rising=delta,contrast_to_pooled_mae=ratio,
                balance=balance(frame),original_rows=original.height,
                original_directional_rows=original_directional.height,
                matched_row_share=frame.height/original_directional.height if original_directional.height else None,
                cell_count=frame.select(KEYS).unique().height)


def support(rows):
    picked=[r for r in rows if r["dataset"]=="yellow_trip_hourly" and r["level_quartile"]==3]
    seeds=[]; evidence=[]
    for seed in parent.SEEDS:
        rs=[r for r in picked if r["seed"]==seed]
        parent.need(len(rs)==4,"Incomplete primary contrast")
        checks=[]
        for r in rs:
            enough = min(r["falling"]["entities"],r["rising"]["entities"])>=5
            if r["method"]=="matched":
                enough &= min(r["falling"]["weight_sum"],r["rising"]["weight_sum"])>=50
            balanced=all(v["smd"] is not None and abs(v["smd"])<=.1 for v in r["balance"].values())
            positive=r["falling_minus_rising"]["bias"] is not None and r["falling_minus_rising"]["bias"]>0
            checks.append(enough and balanced and positive)
        if all(checks): seeds.append(seed)
        evidence.append(dict(seed=seed,checks=checks))
    return dict(implementation_supported=len(seeds)>=2,consistent_seeds=seeds,evidence=evidence,
                prior_10percent_gate_passed=False,training_authorized=False,causal_effect_identified=False)


def markdown(result):
    lines=["# 같은 개체·수량·변화 폭을 맞춘 CNN 오류", "",
           "Train·Validation 저장 예측만 사용. 새 추론·학습·원격·Test 열람 없음.", "",
           "## Taxi 상위 관측 수량 구간: 하락−상승 bias 차이", "",
           "양수는 하락 뒤 예측−실제 오차가 상승 뒤보다 더 크다는 뜻이다. 수량 단위이며 RMSE 차이가 아니다.","",
           "| seed | Train 매칭 | Validation 매칭 | Train 공통 개체 | Validation 공통 개체 |", "|---|---:|---:|---:|---:|"]
    for seed in parent.SEEDS:
        values=[]
        for method,split in (("matched","train"),("matched","validation"),("common_entity","train"),("common_entity","validation")):
            r=next(r for r in result["rows"] if r["dataset"]=="yellow_trip_hourly" and r["seed"]==seed
                   and r["level_quartile"]==3 and r["method"]==method and r["split"]==split)
            x=r["falling_minus_rising"]["bias"]; values.append("N/A" if x is None else f"{x:+.4f}")
        lines.append(f"| {seed} | "+" | ".join(values)+" |")
    lines += ["", "## 매칭 범위와 잔여 차이", "", "| 데이터셋/구간 | split | 방식 | 하락/상승 행 | 개체 | 방향별 가중치 합 | 남은 비율 | 수량 SMD | 변화 폭 SMD |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        if r["seed"]!=42 or r["level_quartile"] not in (-1,3): continue
        def fmt(x): return "N/A" if x is None else f"{x:.4f}"
        lines.append(f"| {previous.DISPLAY[r['dataset']]}/{r['level_quartile']} | {r['split']} | {r['method']} | {r['falling']['rows']}/{r['rising']['rows']} | {r['falling']['entities']} | {r['falling']['weight_sum']:.2f} | {fmt(r['matched_row_share'])} | {fmt(r['balance']['log_level']['smd'])} | {fmt(r['balance']['magnitude']['smd'])} |")
    lines += ["",f"후속 구현 근거: {result['support']['implementation_supported']}; 반복 seed: {result['support']['consistent_seeds']}",
              "", "개체 및 이력 조건을 맞춘 표본에 대한 기술적 비교다. 제외된 표본·시간 의존성·미측정 조건의 영향을 배제하지 못한다.",
              "앞선 10% 기준은 미충족 그대로다. 이번 매칭의 차이 유지 여부는 별도 질문이며 모델 성능 개선이나 인과관계가 아니다.",
              "전체 및 각 사분위의 seed별 RMSE·MAE·bias·TimeNLL, 포함률·ESS·balance는 ANALYSIS.json에 모두 기록했다.",
              "공통 개체의 가중치는 사분위마다 동일하다. 전체(-1)는 사분위별 공통 개체 비교를 합친 보조 요약이다."]
    return "\n".join(lines)+"\n"


def analyze(project, output):
    project,output=Path(project).resolve(),Path(output).resolve()
    parent.need(not output.exists(),"Refuse overwrite")
    prior_dir=project/"reports/titantpp_cnn_history_adaptation_20261015_v1"
    final=prior_dir/"FINAL_RECEIPT.json"
    parent.need(parent.sha(final)==PRIOR_RECEIPT_SHA,"Prior receipt changed")
    inputs={str(final.relative_to(project)):PRIOR_RECEIPT_SHA}
    for rel,expected in parent.read(final)["files"].items():
        parent.need(parent.sha(parent.confined(project,rel))==expected,"Prior output changed: "+rel)
        inputs[rel]=expected
    prior=parent.read(prior_dir/"analysis/ANALYSIS.json")
    parent.need(prior["allowed_splits"]==list(previous.SPLITS) and not prior["test_read"],"Wrong analysis scope")
    inputs.update(prior["inputs"])
    for rel,expected in inputs.items():
        parent.need(parent.sha(parent.confined(project,rel))==expected,"Input SHA mismatch: "+rel)
    by_dataset={}; weight_files={}; rows=[]; baselines=[]
    for evidence in prior["prediction_evidence"]:
        dataset,seed,_=evidence["job_id"].split("__",2); seed=int(seed); split=evidence["split"]
        previous.checked_split(split)
        directory=project/"search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/retrieval/results"/evidence["job_id"]/split
        paths=[parent.confined(directory/"predictions",p["path"]) for p in evidence["parts"]]
        parent.need(all(str(p.relative_to(project)) in inputs for p in paths),"Unverified part")
        f=pl.scan_parquet(paths).select(previous.COLUMNS).collect(engine="streaming")
        parent.need(f.height==evidence["target_count"],"Population mismatch")
        f=f.sort(["entity_id","seq"])
        by_dataset.setdefault(dataset,{})[seed,split]=f
    for dataset, frames in by_dataset.items():
        parent.need(set(frames)=={(s,p) for s in parent.SEEDS for p in previous.SPLITS},"Incomplete seed/split cohort")
        cuts=next(c["cuts"]["last_quantity"] for c in prior["train_cuts"] if c["dataset"]==dataset and c["seed"]==42)
        matches={split:match_weights(frames[42,split].select(HISTORY),cuts) for split in previous.SPLITS}
        common=dict(zip(previous.SPLITS,common_entity_weights(matches["train"],matches["validation"])))
        for split in previous.SPLITS:
            reference=frames[42,split].select([c for c in previous.COLUMNS if c not in ("seed","prediction_1","time_nll_1")])
            for seed in parent.SEEDS:
                f=frames[seed,split]
                parent.need(f.select(reference.columns).equals(reference),"Different histories/targets between seeds")
                scored=f.select("entity_id","seq","time_nll_1",(pl.col("prediction_1")-pl.col("raw_quantity")).alias("error"))
                original=f.with_columns(previous.quartile("last_quantity",cuts).alias("level_quartile"))
                for method, weights in (("matched",matches[split]),("common_entity",common[split])):
                    if seed==42:
                        weight_files[f"{dataset}__{split}__{method}.parquet"]=weights
                    combined=weights.join(scored,on=["entity_id","seq"],how="inner",validate="1:1")
                    for q in (-1,0,1,2,3):
                        subset=combined if q==-1 else combined.filter(pl.col("level_quartile")==q)
                        orig=original if q==-1 else original.filter(pl.col("level_quartile")==q)
                        rows.append(dict(dataset=dataset,seed=seed,split=split,method=method,level_quartile=q,**contrast(subset,orig)))
                # Preserve comparable whole-population values from the verified parent.
                baselines += [r for r in prior["six_metrics"] if r["dataset"]==dataset and r["seed"]==seed and r["split"]==split]
    result=dict(schema="cnn_matched_history_v1",created_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),status="complete",
                rows=rows,support=support(rows),baseline_six_metrics=baselines,inputs=inputs,
                analyzer_sha256=parent.sha(__file__),plan_sha256=parent.sha(output.parent/"ANALYSIS_PLAN.md"),
                new_training=False,new_inference=False,test_read=False,remote_operations=0)
    for rel,expected in inputs.items():
        parent.need(parent.sha(project/rel)==expected,"Input changed during analysis")
    output.mkdir(parents=True,exist_ok=False)
    for name,weights in weight_files.items(): weights.write_parquet(output/name)
    (output/"ANALYSIS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False))
    (output/"ANALYSIS.md").write_text(markdown(result))
    (output/"SHA256.json").write_text(json.dumps({p.name:parent.sha(p) for p in output.iterdir()},indent=2,sort_keys=True))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); r=analyze(args.project,args.output)
    print(json.dumps(r["support"]))
