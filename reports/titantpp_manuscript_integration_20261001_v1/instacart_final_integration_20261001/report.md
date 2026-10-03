# Instacart 최종 validation 결과의 원고 반영

**완료된 분석을 표와 해석에 반영한다 — 완료**

- [원고](../../../paper/titantpp_history_mlp_manuscript_20261001_v1.md) 표4의 S2P2·AttNHP 두 행에 확정된 3seed 평균과 표본표준편차(ddof=1)를 넣었다. 네 데이터 × 7모델 × 3seed, 총84조건·28그룹이며 세 지표는 동일 RMSE 선택checkpoint를 사용한다.
- 원본은 [Instacart 최종 분석](../../titantpp_instacart_final_analysis_20261001_v1/report.md), [기계판독 결과](../../titantpp_instacart_final_analysis_20261001_v1/analysis.json), 완료된 5090 source·checkpoint CPU 감사다. 과거 부분 완료 보고서는 수정하지 않았다.

| Model | Quantity MAE | Quantity RMSE | Time NLL |
|---|---:|---:|---:|
| S2P2 | 3.9987 ± 0.0063 | 5.8523 ± 0.0150 | 2.7970 ± 0.0014 |
| AttNHP | 3.9958 ± 0.0021 | 5.8648 ± 0.0194 | 2.8080 ± 0.0028 |

**새 근거에 맞춰 결과 문단을 고친다 — 완료**

- 수정 전: “The additional Instacart groups remain incomplete at the report cutoff, so their three-seed means are pending.”
- 수정 후: TitanTPP는 S2P2·AttNHP 대비 평균MAE가0.18%·0.11% 낮지만 평균RMSE가0.52%·0.31% 높다고 기록했다. 두 비교군 모두 RMSE에서3seed 전부 우세하다. RMTPP의 최저MAE와 S2P2의 최저RMSE·시간NLL을 함께 설명했다.
- 수량≤20이89.37%인 분포에서 TitanTPP의 작은 수량 오차 이득과 S2P2의 고수량 이득이 상충한다. >35구간 평균RMSE24.6419 대23.4686 및 구간별SSE합으로 설명을 뒷받침한다. 이 관측을 아키텍처의 인과 효과로 확대하지 않았다.
- 서론·평가 범위·한계 절의 미완료 설명을 완료 상태로 갱신했다. 기여 문장의 Taxi·Intermittent 개선율과 효율 주장은 바뀌지 않았다.

**원고와 출처의 일치를 확인한다 — 검증 기록 참조**

- [문장별 수정 전후](changes.json), [수정 전 파일 SHA](before_sha256.json), [기계판독 표](../tables.json), [통합 검증](../verification.json)을 남겼다.
- 기존 검증기는 이번 편집을 역적용한 사본으로 과거 인용·그림·문체 수정의 범위를 계속 검사한다. 현재 원고는28그룹의84seed 기록에서 평균·표본SD를 재계산하고 비율·seed 승패·수량 구간 해석을 대조한다.
- 수식9개·그림2개·표7개·참고문헌32편의 일관성을 유지한다. 분석 원본, 모델 소스, 계약 및 checkpoint는 편집하지 않았다.
- 이번 작업에서 GPU·학습·replay·원격 관측·held-out 열람·스케줄러·커밋·Push·외부 게시는 수행하지 않았다.

**남은 원고 작업을 이어간다 — 다음 작업**

- 확정된 네 데이터 결과에 맞춰 초록·결론을 작성한다. 이후 독립 평가의 split 접근 이력·모델/checkpoint·지표·불확실성 절차를 정리한다. 독립 평가 실행과 제출은 별도 승인 범위다.
