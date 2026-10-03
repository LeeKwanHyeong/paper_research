import React, { useState } from "react";
import { ChartRenderer, DataComponent, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import validation from "./validation-results.json";

const n = (value, digits = 4) => value == null ? "—" : Number(value).toFixed(digits);
const signed = (value, digits = 4) => value == null ? "—" : `${value > 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
const pct = (value, digits = 3) => `${value > 0 ? "+" : ""}${Number(value).toFixed(digits)}%`;
function Table({ children, label }) { return <div className="report-table-wrap" tabIndex="0" role="region" aria-label={label}><table>{children}</table></div>; }

export function ReportContent() {
  const { appTitle, canEdit, mode, setAppTitle, chartProps, chartOverrides, reviewedRows } = useDataApp();
  const datasets = validation.datasets;
  const [datasetId, setDatasetId] = useState(datasets[0]?.dataset_id);
  const active = datasets.find((item) => item.dataset_id === datasetId) ?? datasets[0];
  const finalRows = reviewedRows("final_epoch").filter((row) => row.datasetId === active.dataset_id);
  const [final] = finalRows;
  const series = reviewedRows("learning_curves").filter((row) => row.datasetId === active.dataset_id);
  const clipping = reviewedRows("clipping_history").filter((row) => row.datasetId === active.dataset_id);
  const quantitySpec = chartOverrides["quantity-learning"] ?? { type: "line", x: "epoch", y: "J_RMSE", fields: ["J_RMSE", "Q_RMSE"], legend: { labels: { J_RMSE: "J raw RMSE", Q_RMSE: "Q raw RMSE" } }, valueDecimals: 4 };
  const timeSpec = chartOverrides["time-learning"] ?? { type: "line", x: "epoch", y: "J_time", fields: ["J_time", "T_time"], legend: { labels: { J_time: "J legacy time loss", T_time: "T legacy time loss" } }, valueDecimals: 4 };
  const clippingSpec = chartOverrides["clipping-learning"] ?? { type: "line", x: "epoch", y: "J", fields: ["J", "Q", "T"], legend: { labels: { J: "J", Q: "Q", T: "T" } }, valueDecimals: 3 };

  return <article className="report-content" aria-label="TitanTPP J/Q/T 통합 검증 보고서">
    <header className="report-hero"><p className="report-eyebrow">TitanTPP B · seed 42 · validation-only · 3 datasets</p><h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1><RichNarrative id="report:deck" className="report-deck" label="보고서 소개 편집" value="J(공동), Q(수량 전용), T(시간 전용)를 같은 120 epoch 예산으로 비교했다. primary selector에서는 Q의 raw RMSE가 세 데이터셋 모두 J보다 컸지만, 최종 epoch 120에서는 Taxi와 Instacart의 방향이 달라진다. 이 차이는 단독 학습 실패나 공동학습 간섭의 확정 근거가 아니다." /></header>
    <ReportSection id="key-reading" title="핵심 해석" queryId="primary_selector" sourceRows={datasets} showHeading={false}><RichNarrative id="key-reading:body" className="report-analysis" label="핵심 해석 편집" value={`## 핵심 해석

- **수량 primary selector:** Q의 raw RMSE는 Intermittent +3.2566%, Taxi +2.5493%, Instacart +0.0884%로 세 데이터셋 모두 J보다 컸다.
- **시간 primary selector:** T−J legacy time loss는 Intermittent **+0.0454124**, Taxi **+0.00125715**, Instacart **−0.00182411**이다. 이 값은 정규화된 Time NLL이 아니다.
- **같은 final epoch은 다른 질문이다.** Taxi와 Instacart는 epoch 120에서 Q의 raw RMSE가 J보다 낮고, T도 J보다 낮다. selector 결과와 final epoch 결과를 서로 대체할 수 없다.

현재 증거는 공동학습 간섭이 가능한 병목이라는 가설을 배제하지 않지만, 간섭 부재·모든 분리 구조의 실패·분리 구조의 benchmark 우월성을 입증하지 않는다.`} /></ReportSection>
    <DataComponent id="primary-table" title="Primary selector: 각 목적에 맞는 checkpoint 비교" queryId="primary_selector" kind="table" displayRows={datasets} sourceRows={datasets} description="J 수량/시간 selector는 별개 checkpoint이며 하나의 동시 성능으로 결합하지 않는다. Q−J raw RMSE와 MAE는 변화율(%)이다."><Table label="Primary selector 비교 표"><thead><tr><th>데이터셋</th><th>수량 J / Q epoch</th><th>Q−J raw RMSE 변화(%)</th><th>Q−J MAE 변화(%)</th><th>시간 J / T epoch</th><th>T−J legacy loss</th></tr></thead><tbody>{datasets.map((item) => { const itemQ = item.selected_checkpoint_comparison.quantity; const itemT = item.selected_checkpoint_comparison.time; return <tr key={item.dataset_id}><th scope="row">{item.label}</th><td>{itemQ.J_epoch} / {itemQ.single_epoch}</td><td>{pct(itemQ.comparison.change_percent, 4)}</td><td>{pct(itemQ.mae_at_rmse_selector.change_percent, 4)}</td><td>{itemT.J_epoch} / {itemT.single_epoch}</td><td>{signed(itemT.comparison.single_minus_J, 8)}</td></tr>; })}</tbody></Table></DataComponent>
    <section className="dataset-picker" aria-labelledby="dataset-picker-title"><h2 id="dataset-picker-title">데이터셋별 학습 이력</h2><p>곡선은 같은 epoch에서 J와 Q 또는 T를 비교한다. primary selector와 혼합하지 않는다.</p><div role="group" aria-label="데이터셋 선택">{datasets.map((item) => <button type="button" key={item.dataset_id} aria-pressed={item.dataset_id === active.dataset_id} onClick={() => setDatasetId(item.dataset_id)}>{item.label}</button>)}</div></section>
    <DataComponent id="final-epoch-table" title={`${active.label}: epoch 120 보조 비교`} queryId="final_epoch" kind="table" displayRows={finalRows} sourceRows={finalRows} description="selector가 아니라 같은 최종 epoch 120에서의 보조 비교."><Table label={`${active.label} 최종 epoch 비교 표`}><thead><tr><th>지표</th><th>J</th><th>단일 목적</th><th>단일−J</th></tr></thead><tbody><tr><th scope="row">raw quantity RMSE</th><td>{n(final.J_RMSE)}</td><td>{n(final.Q_RMSE)}</td><td>{pct(final.rawRmseChangePercent, 4)}</td></tr><tr><th scope="row">quantity MAE</th><td>{n(final.J_MAE)}</td><td>{n(final.Q_MAE)}</td><td>{pct(final.quantityMaeChangePercent, 4)}</td></tr><tr><th scope="row">legacy time loss</th><td>{n(final.J_time)}</td><td>{n(final.T_time)}</td><td>{signed(final.timeSingleMinusJoint, 8)}</td></tr></tbody></Table></DataComponent>
    <DataComponent id="quantity-learning" title={`${active.label}: 수량 raw RMSE의 epoch별 비교`} queryId="learning_curves" kind="chart" chart={quantitySpec} displayRows={series} sourceRows={series} description="낮을수록 좋다. J와 Q의 동일 epoch validation raw RMSE."><ChartRenderer spec={quantitySpec} rows={series} height={300} {...chartProps("quantity-learning")} /></DataComponent>
    <DataComponent id="time-learning" title={`${active.label}: legacy time loss의 epoch별 비교`} queryId="learning_curves" kind="chart" chart={timeSpec} displayRows={series} sourceRows={series} description="낮을수록 좋다. legacy cap300 loss이며 Time NLL로 해석하지 않는다."><ChartRenderer spec={timeSpec} rows={series} height={300} {...chartProps("time-learning")} /></DataComponent>
    <DataComponent id="clipping-learning" title={`${active.label}: gradient clipping 비율`} queryId="clipping_history" kind="chart" chart={clippingSpec} displayRows={clipping} sourceRows={clipping} description="J/Q/T의 epoch별 clipping batch 비율. 관측된 경로이며 인과 효과가 아니다."><ChartRenderer spec={clippingSpec} rows={clipping} height={260} {...chartProps("clipping-learning")} /></DataComponent>
    <ReportSection id="methods-limitations" title="방법, 지표 정합성, 한계" queryId="evidence_scope" sourceRows={[validation]} showHeading={false}><RichNarrative id="methods-limitations:body" className="report-analysis" label="방법과 한계 편집" value={`## 방법, 지표 정합성, 한계

동일 source revision \`${validation.source_revision}\`에서 seed 42, static B, arm당 120 epoch, batch 128, AdamW(lr 0.001, weight decay 0.01), clip 1로 실행했다. 수량 목표는 unweighted log1p MSE이고 보고 raw RMSE는 expm1 변환 뒤 원시 수량 오차로 계산한다. 따라서 validation log-MSE 개선은 raw RMSE 개선을 보장하지 않는다.

Taxi J의 사후 기술 구간(epoch 81–100→101–120)에서 validation quantity log-MSE는 1.90% 낮아졌지만 raw RMSE는 7.35% 높아졌다. 이는 지표 차이와 부합하는 관측이지만 tail·과적합·시간 간섭을 원인으로 확정하지 않는다. phase 기간은 사후 기술이며 반복 표본이 아니다. clipping 횟수 비교도 인과 효과를 뜻하지 않는다.

**다음 진단:** B를 고정한 수량 구간·이력 길이별 오차, 부호 편향, log 오차와 raw 제곱오차 기여를 별도 계약에서 비교한다. Taxi의 J/T legacy time 후기 악화와 clipping 경로를 함께 확인한다. 예측 표본 회수가 필요하므로 이 앱의 validation artifact만으로 실행하지 않는다.`} /></ReportSection>
    <footer className="report-evidence"><p>출처: three_dataset_validation_v1/validation_results.json · monitor/20260910T233614Z_check.json</p><p>검증 범위: seed42 validation-only. held-out/test 재평가와 새 실험은 수행하지 않았다.</p></footer>
  </article>;
}
