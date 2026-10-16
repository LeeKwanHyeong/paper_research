from __future__ import annotations
import hashlib, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BUNDLE = PROJECT / "search_artifacts/titantpp_components_inter8_test_20261014_v1"
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplcache")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from plotly.offline import get_plotlyjs

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

contract = json.loads((BUNDLE / "contract.json").read_text())
binding = contract["datasets"]["intermittent_frozen_5000"]
source = BUNDLE / binding["data_path"]
assert sha(source) == binding["data_sha256"]
manifest = json.loads((BUNDLE / "evaluation_results_manifest.json").read_text())
source_receipts = [{
    "path": source.relative_to(PROJECT).as_posix(),
    "sha256": sha(source), "purpose": "frozen observed quantities and split/calendar identities"
}]
model_rows = []
arms = {
    "titantpp_cnn_gru54": "FULL",
    "titantpp_cnn_gru54_correction_disabled": "NONE",
    "titantpp_cnn_gru54_cnn_only": "CNN",
    "titantpp_cnn_gru54_gru_only": "GRU",
}
for job in contract["jobs"]:
    entry = {"seed": job["seed"], "arm": arms[job["arm"]], "epoch": job["selected_epoch"]}
    for split, filename in [("validation", "validation_replay.json"), ("test", "test_metrics.json")]:
        relative = f"results/{job['id']}/{filename}"
        path = BUNDLE / relative
        assert sha(path) == manifest["files"][relative]
        receipt = json.loads(path.read_text())
        assert receipt["seed"] == job["seed"]
        assert receipt["selected_epoch"] == job["selected_epoch"]
        assert receipt["split"] == split
        assert receipt["checkpoint_sha256"] == job["checkpoint"]["sha256"]
        assert receipt["raw_predictions_written"] is False
        entry[split + "_rmse"] = receipt["metrics"]["qty_rmse"]
        entry[split + "_mae"] = receipt["metrics"]["qty_mae"]
        source_receipts.append({
            "path": path.relative_to(PROJECT).as_posix(),
            "sha256": sha(path), "purpose": "existing seed/split aggregate only"
        })
    model_rows.append(entry)

data = pl.read_parquet(source).filter(
    pl.col("chronological_split").is_in(["validation", "test"])
).with_columns(
    pl.col("demand_dt").cast(pl.String).str.strptime(pl.Date, "%Y%m%d").alias("date")
).sort(["oper_part_no", "date"])
assert data.height == 86285 + 88019
assert data.group_by(["oper_part_no", "date"]).len().filter(pl.col("len") > 1).height == 0
assert data["demand_qty"].is_finite().all() and (data["demand_qty"] > 0).all()
assert set(data["date"].dt.weekday().unique().to_list()) == {1}
series, profiles = [], []
for group in data.partition_by("oper_part_no", maintain_order=True):
    identity = group["oper_part_no"][0]
    va = group.filter(pl.col("chronological_split") == "validation")
    te = group.filter(pl.col("chronological_split") == "test")
    assert va.height and te.height and va["date"].max() < te["date"].min()
    v_dates, t_dates = va["date"].dt.to_string("%Y-%m-%d").to_list(), te["date"].dt.to_string("%Y-%m-%d").to_list()
    v_values, t_values = va["demand_qty"].to_list(), te["demand_qty"].to_list()
    site, material = identity.split("::", 1)
    vmean, tmean = float(va["demand_qty"].mean()), float(te["demand_qty"].mean())
    profile = {
        "series_id": identity, "site": site, "material": material,
        "validation_n": va.height, "test_n": te.height,
        "validation_start": v_dates[0], "validation_end": v_dates[-1],
        "test_start": t_dates[0], "test_end": t_dates[-1],
        "validation_mean": vmean, "test_mean": tmean,
        "mean_change": tmean - vmean,
        "max_quantity": max(v_values + t_values),
        "validation_median": float(va["demand_qty"].median()),
        "test_median": float(te["demand_qty"].median())
    }
    profiles.append(profile)
    series.append({"id": identity, "site": site, "material": material,
                   "v": [v_dates, v_values], "t": [t_dates, t_values], "stats": profile})
assert len(series) == 5000
profile_df = pl.DataFrame(profiles).sort("series_id")
profile_df.write_csv(HERE / "series_profiles.csv")
data.select(["oper_part_no", "demand_dt", "date", "seq", "chronological_split", "demand_qty"]).write_parquet(
    HERE / "observed_validation_test.parquet"
)

# Deliberately include upward, downward and stable raw patterns. This is a
# post-hoc descriptive selection, not a ranking of model errors/material causes.
asc = sorted(series, key=lambda item: (item["stats"]["mean_change"], item["id"]))
stable = sorted(series, key=lambda item: (abs(item["stats"]["mean_change"]), item["id"]))
selected = asc[:2] + list(reversed(asc[-2:])) + [item for item in stable if item["stats"]["max_quantity"] > 2][:2]
selected_ids = [item["id"] for item in selected]
assert len(selected_ids) == len(set(selected_ids)) == 6
payload = {
    "series": series, "selected": selected_ids, "modelRows": model_rows,
    "seedsWithTest": sorted({r["seed"] for r in model_rows}),
    "source": source.relative_to(PROJECT).as_posix(),
    "sourceSHA": sha(source)
}
(HERE / "chart_data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

template = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Intermittent · 자재별 Validation / Test 패턴</title>
<style>
*{box-sizing:border-box}body{margin:0;font:15px -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;color:#243042;background:#f5f7fb}
main{max-width:1260px;margin:auto;padding:28px}h1{font-size:27px;letter-spacing:-.6px;margin:0 0 12px}h2{font-size:18px;margin:30px 0 12px}p{line-height:1.65;margin:8px 0;color:#526176}
.note{padding:15px 18px;border-left:4px solid #ddaa26;background:#fff8e6;border-radius:8px;margin:18px 0}.toolbar{display:flex;flex-wrap:wrap;gap:16px;align-items:end;margin:20px 0}
label{display:flex;flex-direction:column;gap:7px;font-size:13px;color:#526176}input,select,button{font:inherit;padding:9px 12px;border:1px solid #cbd3df;border-radius:7px;background:#fff;color:#243042}
input{width:285px}select{max-width:440px}button{cursor:pointer}button:hover{background:#edf3fc}.card{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 10px #2030500a}
#chart{height:510px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}.stat{padding:12px;background:#fff;border-radius:8px;color:#526176;font-size:12px}.stat b{display:block;font-size:19px;color:#243042;margin:7px 0}
.examples{display:flex;gap:8px;flex-wrap:wrap}.examples button{font-size:12px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;background:#fff}th,td{text-align:right;padding:11px;border-bottom:1px solid #e4e9f0}th:first-child,td:first-child{text-align:left}
.caption{font-size:12px;color:#65738a}#status{font-size:13px;color:#65738a}.full{font-weight:700}.warning{color:#ad4e15}.badge{display:inline-block;padding:5px 9px;font-size:12px;border-radius:5px;background:#e9eef6;margin-right:6px}
@media(max-width:700px){main{padding:16px}h1{font-size:23px}.stats{grid-template-columns:repeat(2,1fr)}#chart{height:450px}select,input{max-width:100%;width:100%}}
</style></head><body><main>
<h1>Intermittent · 자재별 Validation / Test 패턴</h1>
<p>같은 시간축에서 관측 수량을 비교합니다. 파란 선은 Validation, 주황 선은 Test이며 세로 점선은 <b>Horizon — 첫 Test 관측일</b>입니다.</p>
<span class="badge">5,000개 사업장–자재 시계열</span><span class="badge">세 seed 공통 원자료</span><span class="badge">로컬 전용 · 모델 예측선 없음</span>
<div class="note">원자료는 <b>주간 양수 수량 기록</b>입니다. 일자축으로 표시하지만 일별 수요를 새로 계산하지 않았습니다. 없는 기록은 0으로 채우지 않습니다. 날짜는 기존 보안 목적 이동값이며 원래 달력 날짜가 아닙니다. Validation/Test 경계는 자재마다 다릅니다.</div>
<div class="toolbar"><label>자재·사업장 검색<input id="search" type="search" placeholder="예: DS_A054596 또는 V122"></label>
<label>사업장–자재 선택<select id="series"></select></label>
<label>정렬<select id="sort"><option value="id">자재 ID</option><option value="drop">평균 수량 감소 큰 순</option><option value="rise">평균 수량 증가 큰 순</option><option value="stable">평균 수량 변화 작은 순</option></select></label>
<label>수량축<select id="yscale"><option value="linear">일반 수량축</option><option value="log">로그 수량축</option></select></label>
<button id="reset">전체 기간</button><button id="export">현재 그래프 PNG</button></div>
<div class="examples" id="examples"></div><p id="status"></p>
<div class="stats" id="stats"></div><div class="card"><div id="chart"></div></div>
<p class="caption">점은 실제 기록입니다. 선은 기록 사이의 연결 가이드이며 날짜 사이 수량을 보간한 값이 아닙니다. 회색 점선은 두 split의 마지막/첫 관측을 연결합니다. 확대·범위 선택·PNG 저장이 가능합니다. 자재별 y축은 개별 범위입니다.</p>
<h2>저장된 모델 비교 · seed 42 / 52</h2>
<p>최근 NONE / CNN / GRU / FULL 평가에서 Test가 저장된 것은 두 seed입니다. seed 62는 이 표에 포함하지 않습니다. 평가 결과에는 행별 예측이 없으므로 위 그래프에 모델 예측선을 그리지 않았습니다.</p>
<div class="scroll"><table><thead><tr><th>Seed · 구조</th><th>선택 epoch</th><th>Validation RMSE</th><th>Test RMSE</th><th>Validation MAE</th><th>Test MAE</th></tr></thead><tbody id="modelTable"></tbody></table></div>
<p id="comparisonNote"></p>
<details><summary>자료·경계·표본 선택 기준</summary><p>그래프 모집단: Validation 86,285개 + Test 88,019개, 합계 174,304개 기록입니다. 입력 데이터와 저장 모델 지표의 원본 SHA를 대조했습니다. Horizon은 해당 자재의 첫 Test 기록 날짜로, 다중 스텝 예측 리드타임을 뜻하지 않습니다.</p>
<p>예시 버튼과 미리보기는 평균 수량 감소 2개·증가 2개·안정 2개를 사후 선택한 관측 예시입니다. 모델 오차가 가장 큰 자재 또는 성능 반전의 원인으로 선택한 것이 아닙니다. 모든 5,000개 시계열을 검색할 수 있습니다. 원자료 모양만으로 seed별 성능 역전의 원인을 확정할 수 없습니다.</p>
<p id="sourceInfo" class="caption"></p></details>
</main><script>__PLOTLY__</script><script>
const DATA=__PAYLOAD__;
const indexed=new Map(DATA.series.map(s=>[s.id,s]));
const el=id=>document.getElementById(id), fmt=n=>n.toLocaleString("ko-KR",{maximumFractionDigits:3});
function options(keep){
 const q=el("search").value.trim().toLowerCase(), order=el("sort").value;
 let choices=DATA.series.filter(s=>s.id.toLowerCase().includes(q));
 choices.sort((a,b)=>order==="drop"?a.stats.mean_change-b.stats.mean_change:order==="rise"?b.stats.mean_change-a.stats.mean_change:order==="stable"?Math.abs(a.stats.mean_change)-Math.abs(b.stats.mean_change):a.id.localeCompare(b.id));
 const select=el("series");select.replaceChildren();
 for(const s of choices){const o=document.createElement("option");o.value=s.id;o.textContent=s.id;select.appendChild(o)}
 if(choices.some(s=>s.id===keep))select.value=keep;
 el("status").textContent=fmt(choices.length)+" / 5,000개 자재 표시";
 if(choices.length)draw();else{el("stats").replaceChildren();Plotly.react("chart",[],{annotations:[{text:"검색 결과 없음",showarrow:false}]},{responsive:true})}
}
function draw(){
 const s=indexed.get(el("series").value);if(!s)return;
 const st=s.stats;el("stats").innerHTML=[
 ["Validation 기록",fmt(st.validation_n),st.validation_start+" ~ "+st.validation_end],
 ["Test 기록",fmt(st.test_n),st.test_start+" ~ "+st.test_end],
 ["평균 수량 · Validation / Test",fmt(st.validation_mean)+" / "+fmt(st.test_mean),"양수 기록당 평균 · 일평균 아님"],
 ["Horizon",st.test_start,"해당 자재의 첫 Test 기록일"]
 ].map(([a,b,c])=>'<div class="stat">'+a+'<b>'+b+'</b>'+c+'</div>').join("");
 const trace=(x,y,name,color)=>({x,y,type:"scatter",mode:"lines+markers",name,line:{color,width:2},marker:{size:5,color},hovertemplate:"%{x|%Y-%m-%d}<br>수량 %{y}<extra>"+name+"</extra>"});
 const traces=[trace(s.v[0],s.v[1],"Validation","#2b67b5"),trace(s.t[0],s.t[1],"Test","#df7c28"),{x:[s.v[0].at(-1),s.t[0][0]],y:[s.v[1].at(-1),s.t[1][0]],type:"scatter",mode:"lines",name:"구간 연결 가이드",line:{color:"#9ba8ba",width:1.6,dash:"dot"},hoverinfo:"skip"}];
 const shape={type:"line",xref:"x",yref:"paper",x0:st.test_start,x1:st.test_start,y0:0,y1:1,line:{color:"#576476",width:1.6,dash:"dash"}};
 const layout={title:{text:s.id,font:{size:18},x:.04},margin:{l:70,r:30,t:85,b:65},font:{family:'-apple-system, "Apple SD Gothic Neo", sans-serif',size:13,color:"#243042"},paper_bgcolor:"#fff",plot_bgcolor:"#fff",xaxis:{type:"date",title:{text:"기록 날짜 (보안 목적 이동값)"},showgrid:true,gridcolor:"#eef1f5",tickformat:"%Y-%m",rangeslider:{visible:true,thickness:.09}},yaxis:{title:{text:"주간 기록 수량"},type:el("yscale").value,showgrid:true,gridcolor:"#eef1f5"},shapes:[shape],annotations:[{xref:"x",yref:"paper",x:st.test_start,y:1.07,text:"Horizon · Test 시작",showarrow:false,font:{size:12,color:"#576476"}}],legend:{orientation:"h",x:0,y:1.18},hovermode:"closest",uirevision:s.id};
 Plotly.react("chart",traces,layout,{responsive:true,displaylogo:false,modeBarButtonsToRemove:["lasso2d","select2d"],toImageButtonOptions:{format:"png",filename:s.id.replaceAll("::","_"),width:1400,height:650,scale:2}});
}
for(const [ix,id] of DATA.selected.entries()){const b=document.createElement("button");b.textContent=id+" · "+["평균 감소","평균 감소","평균 증가","평균 증가","평균 유사","평균 유사"][ix];b.onclick=()=>{el("search").value="";options(id)};el("examples").appendChild(b)}
el("search").addEventListener("input",()=>options(el("series").value));el("sort").addEventListener("change",()=>options(el("series").value));el("series").addEventListener("change",draw);el("yscale").addEventListener("change",draw);
el("reset").onclick=()=>Plotly.relayout("chart",{"xaxis.autorange":true,"yaxis.autorange":true});
el("export").onclick=()=>Plotly.downloadImage("chart",{format:"png",filename:el("series").value.replaceAll("::","_"),width:1400,height:650,scale:2});
const order={NONE:0,CNN:1,GRU:2,FULL:3};
for(const r of [...DATA.modelRows].sort((a,b)=>a.seed-b.seed||order[a.arm]-order[b.arm])){
 const tr=document.createElement("tr");if(r.arm==="FULL")tr.className="full";
 for(const v of [r.seed+" · "+r.arm,r.epoch,r.validation_rmse,r.test_rmse,r.validation_mae,r.test_mae]){const td=document.createElement("td");td.textContent=typeof v==="number"?(Number.isInteger(v)?v:fmt(v)):v;tr.appendChild(td)}el("modelTable").appendChild(tr)
}
const by=(seed,arm)=>DATA.modelRows.find(r=>r.seed===seed&&r.arm===arm);
const f=by(42,"FULL"),n=by(42,"NONE");
el("comparisonNote").textContent="seed 42의 FULL−NONE RMSE 차이: Validation "+(f.validation_rmse-n.validation_rmse).toFixed(4)+" → Test +"+(f.test_rmse-n.test_rmse).toFixed(4)+". 이 부호 반전은 저장된 전체 지표이며 특정 자재의 원인을 뜻하지 않습니다.";
el("sourceInfo").textContent=DATA.source+" · SHA256 "+DATA.sourceSHA;
options(DATA.selected[0]);
</script></body></html>"""
html = template.replace("__PLOTLY__", get_plotlyjs()).replace(
    "__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
)
(HERE / "index.html").write_text(html)

available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": "AppleGothic" if "AppleGothic" in available else "DejaVu Sans",
                     "axes.unicode_minus": False, "font.size": 10})
fig, axes = plt.subplots(3, 2, figsize=(15, 11))
for ax, item in zip(axes.flat, selected):
    vdate, tdate = [np.array(values, dtype="datetime64[D]") for values in [item["v"][0], item["t"][0]]]
    ax.plot(vdate, item["v"][1], "o-", ms=3, lw=1.5, color="#2b67b5", label="Validation")
    ax.plot(tdate, item["t"][1], "o-", ms=3, lw=1.5, color="#df7c28", label="Test")
    ax.plot([vdate[-1], tdate[0]], [item["v"][1][-1], item["t"][1][0]], ":", color="#9ba8ba", lw=1.5)
    ax.axvline(tdate[0], color="#576476", lw=1.2, ls="--")
    ax.text(tdate[0], .97, " Horizon · Test 시작", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color="#576476")
    delta = item["stats"]["mean_change"]
    ax.set_title(item["id"] + "   평균 변화 " + f"{delta:+.2f}", loc="left", fontsize=11, pad=10)
    ax.set_ylabel("주간 기록 수량")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, color="#edf0f5", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Intermittent · 자재별 Validation / Test 원자료 패턴", fontsize=18, x=.06, ha="left")
fig.text(.06, .935, "세 seed 공통 자료 · 이동된 날짜 · 주간 기록 · 자재별 첫 Test 날짜에 Horizon 표시", fontsize=11, color="#526176")
fig.text(.06, .02, "평균 감소 2개·증가 2개·안정 2개의 사후 관측 예시. 자재별 y축은 개별 범위이며 모델 예측·오차 기여 그래프가 아닙니다.", fontsize=10, color="#526176")
fig.tight_layout(rect=(.03, .045, .99, .92), h_pad=2.5, w_pad=2)
fig.savefig(HERE / "patterns_overview.png", dpi=180, facecolor="white")
fig.savefig(HERE / "patterns_overview.pdf", facecolor="white")
plt.close(fig)

qa = {
    "dataset_source_sha_verified": True,
    "evaluation_result_sha_verified_count": 16,
    "series_count": len(series), "validation_rows": 86285, "test_rows": 88019,
    "series_have_both_splits": True, "duplicate_series_date_rows": 0,
    "all_validation_dates_precede_test_dates_per_series": True,
    "all_source_weekdays": "Monday", "source_grain": "weekly positive demand records",
    "calendar_dates_are_shifted": True, "missing_days_zero_filled": False,
    "horizon_rule": "first recorded Test date within each site/material series",
    "horizon_is_forecast_lead_time": False,
    "raw_observations_seed_dependent": False, "seeds_with_saved_latest_test_metrics": [42, 52],
    "model_prediction_lines_available": False,
    "new_training": False, "new_inference": False, "remote_calls": 0,
    "scientific_sources_modified": False, "manuscript_modified": False,
    "representative_selection": "posthoc observation mean decrease/increase/stable, two each; not ranked model errors",
    "source_receipts": source_receipts, "selected_example_ids": selected_ids
}
(HERE / "QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n")
outputs = ["index.html", "chart_data.json", "series_profiles.csv", "observed_validation_test.parquet",
           "patterns_overview.png", "patterns_overview.pdf", "QA.json", "build_patterns.py"]
(HERE / "output_manifest.json").write_text(json.dumps({name: sha(HERE / name) for name in outputs}, indent=2) + "\n")
print(json.dumps({"output": str(HERE), "series": len(series), "observations": data.height,
                  "example_ids": selected_ids, "html_bytes": (HERE / "index.html").stat().st_size}, ensure_ascii=False))

