import React, {useState} from "react";
import {DataComponent, DataTable, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import narratives from "./narratives.json";
import "./report.css";
const labels={titantpp:"B · 이력 보완 없음",titantpp_local_detail:"Full",titantpp_history_mlp:"이력 MLP",titantpp_level_only:"수준만",titantpp_change_only:"변화만",titantpp_no_static_lmm:"정적 검색 제거"};
const dsIds={Taxi:"yellow_trip_hourly",Intermittent:"intermittent_frozen_5000",Instacart:"insta_market_basket"};
const pct=v=>`${Number(v)>0?"+":""}${Number(v).toFixed(2)}%`;
const contrastColumns=[{key:"reference",label:"Full 대비 기준",renderCell:v=>labels[v]},{key:"mae_reduction_pct",label:"MAE 감소",renderCell:pct},{key:"rmse_reduction_pct",label:"RMSE 감소",renderCell:pct},{key:"mae_wins",label:"MAE 승리",renderCell:v=>`${v}/3`},{key:"rmse_wins",label:"RMSE 승리",renderCell:v=>`${v}/3`},{key:"gate_passes",label:"보호기준 통과",renderCell:v=>`${v}/3`}];
const strataColumns=[{key:"model",label:"모델",renderCell:v=>labels[v]},{key:"body_mae",label:"body MAE",renderCell:v=>num(v)},{key:"tail_mae",label:"tail MAE",renderCell:v=>num(v)},{key:"tail_rmse",label:"tail RMSE",renderCell:v=>num(v)},{key:"count",label:"tail 표본/seed"}];
const costColumns=[{key:"model",label:"모델",renderCell:v=>labels[v]},{key:"parameters",label:"파라미터",renderCell:v=>Number(v).toLocaleString("en-US")},{key:"recorded_fit_seconds",label:"기록 fit 평균분",renderCell:v=>num(v.mean/60,2)},{key:"completed_epochs",label:"종료 epoch 평균",renderCell:v=>num(v.mean,1)},{key:"peak_allocated_mib",label:"peak allocated MiB",renderCell:v=>num(v.mean,2)},{key:"standalone_eligible_seeds",label:"단독 비교 가능 seed",renderCell:v=>v.length}];
const datasets=["Taxi","Intermittent","Instacart"];
const num=(x,n=4)=>Number(x).toFixed(n);
const metricColumns=[{key:"model",label:"모델"},...[["qty_mae","수량 MAE ↓"],["qty_rmse","수량 RMSE ↓"],["time_nll","시간 NLL ↓"]].map(([k,label])=>({key:k+"_mean",label,renderCell:(v,r)=>`${num(v)} ± ${num(r[k+"_sample_sd"])}`}))];
const effectColumns=[{key:"model",label:"후보"},{key:"mae_reference",label:"MAE 비교군"},{key:"mae_reduction_pct",label:"MAE 감소율",renderCell:v=>`${num(v,2)}%`},{key:"rmse_reference",label:"RMSE 비교군"},{key:"rmse_reduction_pct",label:"RMSE 감소율",renderCell:v=>`${num(v,2)}%`}];
const gateColumns=[{key:"dataset",label:"데이터"},{key:"mode",label:"MLP Gate"},{key:"mae",label:"MAE",renderCell:v=>num(v)},{key:"rmse",label:"RMSE",renderCell:v=>num(v)},{key:"time_nll",label:"시간 NLL",renderCell:v=>num(v)},{key:"mae_change_pct",label:"MAE 변화",renderCell:v=>`${v>0?"+":""}${num(v,2)}%`},{key:"rmse_change_pct",label:"RMSE 변화",renderCell:v=>`${v>0?"+":""}${num(v,2)}%`}];
const datasetText={Taxi:"Full의 평균 RMSE는 MLP보다 1.17% 낮습니다. 다만 RMSE는 2/3 seed, MAE는 1/3 seed에서만 MLP를 이겼습니다. 정적 검색 제거는 평균 MAE가 더 낮습니다. 시간 NLL은 Full과 MLP 모두 B보다 높습니다.",Intermittent:"이력 MLP가 core의 평균 MAE·RMSE 최저입니다. Full 대비 RMSE는 3/3 seed, MAE는 2/3 seed에서 낮습니다. 평균 RMSE는 Full보다 7.77% 낮지만 시간 NLL은 외부 RMTPP보다 높습니다.",Instacart:"변화만 모델이 core 평균 MAE·RMSE 최저지만 외부 최저를 넘지 못했습니다. core RMSE 평균은 5.8785–5.8896으로 좁은 범위입니다. 이 차이가 통계적으로 동등하다는 뜻은 아닙니다."};
export function ReportContent(){
 const {snapshot,visible,appTitle,canEdit,mode,setAppTitle}=useDataApp();
 const [dataset,setDataset]=useState("Taxi");
 const rows=(id)=>snapshot.queries[id].rows;
 const all=rows("three_seed"),scope=all.filter(r=>r.dataset===dataset);
 const core=scope.filter(r=>r.model_id.startsWith("titantpp")),ext=scope.filter(r=>!r.model_id.startsWith("titantpp"));
 const effects=rows("external_effects").filter(r=>r.dataset===dataset),gate=rows("gate");
 const contrast=rows("analysis_detail").filter(r=>r.dataset===dsIds[dataset]);
 const strataSource=rows("strata").filter(r=>r.dataset===dsIds[dataset]&&["body","tail"].includes(r.group)&&["titantpp","titantpp_local_detail","titantpp_history_mlp","titantpp_no_static_lmm"].includes(r.model));
 const strataRows=strataSource.filter(r=>r.group==="tail").map(r=>({model:r.model,body_mae:strataSource.find(b=>b.model===r.model&&b.group==="body").qty_mae.mean,tail_mae:r.qty_mae.mean,tail_rmse:r.qty_rmse.mean,count:r.count_per_seed}));
 const costs=rows("costs").filter(r=>r.dataset===dsIds[dataset]);
 const prose=(id,title,queryId,text,queryIds=[queryId])=>(visible(id)&&<ReportSection key={id} id={id} title={title} queryId={queryId} queryIds={queryIds} sourceRowsByQuery={Object.fromEntries(queryIds.map(q=>[q,rows(q)]))} showHeading={false}><RichNarrative id={id+":text"} value={"## "+title+"\n\n"+text}/></ReportSection>);
 return <article className="report-content titan-report" aria-label="TitanTPP 1차 분석">
  <header className="report-hero"><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={e=>{if(canEdit&&mode==="edit")setAppTitle(e.currentTarget.textContent.trim()||appTitle)}}>{appTitle}</h1><RichNarrative id="report:subtitle" value="5080·5090 완료 결과를 함께 읽는 1차 분석 · 2026년 9월 30일" className="report-deck"/></header>
  {visible("main-summary")&&<ReportSection id="main-summary" title="핵심 판단" queryId="three_seed" queryIds={["three_seed","external_effects","gate"]} sourceRowsByQuery={{three_seed:all,external_effects:rows("external_effects"),gate}} showHeading={false}><RichNarrative id="main-summary:text" value={`## 핵심 판단

**Full과 이력 MLP의 외부 TPP 대비 수량 경쟁력은 유지됩니다.** Taxi·Intermittent에서는 두 모델 각각 하나를 고정해도 MAE·RMSE가 외부 네 비교 구현보다 같은 seed 세 번 모두 낮습니다.

**그 경쟁력을 Full의 분리 구조나 조건부 Gate의 우월성과 동일시할 수는 없습니다.** Taxi 평균 RMSE는 Full이 MLP보다 1.17% 낮지만, Intermittent는 MLP가 Full보다 7.77% 낮습니다. Gate 탐색에서는 조건부의 수량 개선이 확인되지 않았습니다.

**Instacart는 적용 범위의 한계입니다.** 마지막 5090 실험까지 완료됐지만 core 최저 RMSE 5.8785는 외부 THP의 5.8582보다 높습니다.

논문의 잠정 중심은 사건 이력 기반 **수량 예측 표현의 경쟁력과 데이터별·시간 예측의 상충관계**입니다.`}/></ReportSection>}
  {prose("detail-status","학습은 모두 완료됐고 최종 원본 감사는 남아 있습니다","evidence_status","5090은 **9월 30일 06:55:56 KST**에 끝났습니다. 08:21:08 KST 관측에서 마지막 조건·전체 partition 완료와 학습 소유 프로세스 종료를 확인했습니다. 5080 MLP Gate 네 조건은 전날 23:25:25 KST에 완료됐습니다.\n\ncore **54조건·108 selected/last 역할**의 소형 기록 감사는 통과했습니다. 5090 checkpoint binary와 최종 source 회수 감사는 별도 남아 있어 이 보고서는 1차 분석입니다. 외부 36조건은 중복을 제거해 비교했고 Gate 여섯 완료 조건은 별도 seed42 탐색으로 유지합니다.".replaceAll("\\n","\n"))}
  <section className="titan-evidence" aria-label="데이터별 결과">
   <RichNarrative id="dataset-heading" value={`## 데이터별로 수량과 시간을 함께 비교하세요

평균 ± 표본표준편차는 seed42·52·62의 동일 가중치 집계입니다. 모든 지표는 같은 validation RMSE 선택 checkpoint에서 계산했습니다. 데이터 간 수량 단위가 달라 하나의 RMSE로 합산하지 않습니다. 아래 구조·구간·비용 표에도 같은 데이터 선택이 적용됩니다.`}/>
   <label className="titan-select">데이터 <select aria-label="분석 데이터" value={dataset} onChange={e=>setDataset(e.target.value)}>{datasets.map(d=><option key={d}>{d}</option>)}</select></label>
   {visible("core-table")&&<DataComponent id="core-table" kind="table" title={dataset+" · TitanTPP 구조 비교"} queryId="three_seed" sourceRows={core} displayRows={core}><DataTable rows={core} columns={metricColumns} searchable={false} caption={dataset+" core 3seed 결과"}/></DataComponent>}
   {visible("external-table")&&<DataComponent id="external-table" kind="table" title={dataset+" · 공통 head 외부 TPP 비교"} queryId="three_seed" sourceRows={ext} displayRows={ext}><DataTable rows={ext} columns={metricColumns} searchable={false} caption={dataset+" 외부 TPP 3seed 결과"}/></DataComponent>}
   <ReportSection id="dataset-interpretation" title={dataset+" 해석"} queryId="three_seed" sourceRows={scope} showHeading={false}><RichNarrative id={"dataset-interpretation:"+dataset} value={datasetText[dataset]}/></ReportSection>
   {visible("external-effects")&&<DataComponent id="external-effects" kind="table" title={dataset+" · 지표별 외부 최저 대비 감소율"} queryId="external_effects" sourceRows={effects} displayRows={effects}><DataTable rows={effects} columns={effectColumns} searchable={false} caption="외부 비교군 대비 효과"/></DataComponent>}
   <RichNarrative id="external-effects:note" value="양의 감소율은 후보가 더 좋다는 뜻입니다. Instacart에서는 지표별 기준 모델이 다릅니다. 현재 결과는 공통 head·고정 학습 규칙의 비교 구현에 대한 것이며, 원논문의 최적 튜닝 결과에 대한 보편적 순위는 아닙니다."/>
  </section>
  {prose("detail-ablation","Full의 고유 구조는 어디까지 지지되는가","analysis_detail","Full 대 MLP가 핵심 사전 비교입니다. Full은 세 데이터 모두 MLP 대비 전체 보호기준 통과가 0/3입니다. 실행 완료와 성능 기준 통과는 구분해야 합니다. 다른 구조에 대해서는 같은 기준을 탐색적으로 적용했습니다.")}
  {visible("ablation-table")&&<DataComponent id="ablation-table" kind="table" title={dataset+" · Full의 기준 모델 대비 효과"} queryId="analysis_detail" sourceRows={contrast} displayRows={contrast}><DataTable rows={contrast} columns={contrastColumns} searchable={false} caption={dataset+" Full 대비 구조별 비교"}/></DataComponent>}
  <RichNarrative id="ablation:limits" value="보호기준은 원래 계약의 전체·body·tail 수량 오차, 중간 수량 bin 1–3, 시간 NLL(+0.01), last30 평균·표준편차 한도를 그대로 적용했습니다. 모든 seed의 실패 항목은 analysis.json에 보존했습니다."/>
  {prose("detail-strata","전체 평균에 가려진 구간별 반례","strata","Taxi에서는 Full의 전체 RMSE가 낮아도 꼬리 구간에서는 MLP·정적 검색 제거가 더 낮은 오차를 보입니다. Intermittent와 Instacart에서도 Full의 tail 오차는 B보다 높습니다. 아래는 3seed 평균입니다.")}
  {visible("strata-table")&&<DataComponent id="strata-table" kind="table" title={dataset+" · body와 tail 수량 오차"} queryId="strata" sourceRows={strataSource} displayRows={strataRows}><DataTable rows={strataRows} columns={strataColumns} searchable={false} caption={dataset+" body·tail 비교"}/></DataComponent>}
  <RichNarrative id="strata:limits" value="구간 경계는 train에서 고정했습니다. Instacart의 기본 이력 >64 구간은 비어 있어 0점으로 채우지 않았습니다. 모든 수량·이력 구간의 원값과 표본 수는 analysis.json에 보존했습니다."/>
  <section className="titan-evidence"><RichNarrative id="gate-heading" value={`## Gate는 기존 MLP와 같은 seed42로 비교했습니다

조건부는 두 데이터 모두 기존 MLP보다 MAE·RMSE가 높았습니다. Taxi 상수 계수만 MAE 1.42%, RMSE 2.76% 개선했지만 시간 NLL은 0.842735에서 1.304540으로 악화됐습니다.`}/>
   {visible("gate-table")&&<DataComponent id="gate-table" kind="table" title="MLP Gate · seed42 validation" queryId="gate" sourceRows={gate} displayRows={gate}><DataTable rows={gate} columns={gateColumns} searchable={false} caption="MLP Gate seed42"/></DataComponent>}
   <RichNarrative id="gate:limits" value="변화율은 같은 seed42 plain MLP 대비이며 양수는 악화입니다. Intermittent 조건부는 5epoch가 최종 best였고 45epoch에서 종료됐습니다. Gate 결과를 3seed 평균과 섞거나 3seed 우월성으로 일반화하지 않습니다. 기존 Full Gate A/B는 별도 완료, Full Taxi C/D는 미시작 보류로 보존했습니다."/>
  </section>
  {prose("contribution","현재 증거에 맞춘 Contribution","three_seed",narratives["Contribution을 현재 증거에 맞춰 정리하면"].split("**논문용 잠정 문장:**")[1], ["three_seed","analysis_detail","gate"])}
  {prose("detail-costs","실측 비용과 비교할 수 없는 시간","costs",narratives["실측 비용과 측정 한계"].split("기록fit은")[1].replace(/^/,"기록 fit은"))}
  {visible("cost-table")&&<DataComponent id="cost-table" kind="table" title={dataset+" · 기록된 core 학습 비용"} queryId="costs" sourceRows={costs} displayRows={costs}><DataTable rows={costs} columns={costColumns} searchable={false} caption={dataset+" 학습 비용"}/></DataComponent>}
  <RichNarrative id="cost:limits" value="총 fit 시간은 종료 epoch가 서로 다른 학습 실행의 기록입니다. 동일 작업량의 속도 비교나 추론 지연 비교가 아닙니다. 메모리는 PyTorch peak allocated이며 장치 전체 메모리가 아닙니다. 원본 Titans 대비 경량화·속도 우위는 아직 입증하지 않았습니다."/>
  {prose("detail-methods","평가 기준과 모델의 의미","evidence_status",narratives["평가 기준과 모델의 의미"])}
  {prose("remaining-work","남은 작업 순서","evidence_status",narratives["남은 작업 순서"])}
 </article>;
}
