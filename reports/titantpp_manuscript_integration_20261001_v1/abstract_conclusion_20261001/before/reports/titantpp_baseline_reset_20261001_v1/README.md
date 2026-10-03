**참고문헌 재대조 — 완료:** 32편의 서지·인용 대응을 확인하고 8편의 누락 정보와 형식을 보완했다. [검토 결과](../titantpp_manuscript_integration_20261001_v1/reference_doublecheck_20261001/report.md). 이번 문헌 작업은 저장된 실험 기준선과 관측 시각을 갱신하지 않았다.

# TitanTPP 현재 기준선과 남은 작업 — Instacart 최종 결과 반영

**완료 결과를 현재 기준선으로 고정한다 — 완료**

- 대표 모델은 `titantpp_history_mlp`이며 활성 분기 정규화는 별도 탐색 변형이다. 동일 RMSE 선택 checkpoint의 validation MAE·RMSE·시간 NLL을 함께 보고한다.
- Core54조건, 기존 외부4종36조건, Gate6조건, 5080 추가TPP12조건, 기존3데이터 정규화seed42, 효율 측정은 완료 증거를 재사용한다.
- RAF8모델×3seed24조건은2026-10-01 08:55:56 KST 완료됐다. 원본473파일·source104개·checkpoint48개 회수·SHA·CPU 감사 및 보고서·원고 반영을 마쳤다. [RAF 상세](../titantpp_raf_execution_20261001_v1/report.md).
- [기존 전체 보고서](../titantpp_completed_external_comparison_20261001_v1/report.md)는08:59 KST 시점의139개 고유 완료조건을 보존한다. 이후 완료된 Instacart 두 조건은 [최종 Instacart 분석](../titantpp_instacart_final_analysis_20261001_v1/report.md)에 포함되어 있다. 원고 표4는 네 데이터·7모델·3seed의84조건을 모두 반영했다.
- 과거07:35 기준선은 before_raf_integration/에 보존했다. 스케줄러 삭제 결과는 기존증거를 재사용하며 새 자동화를 만들지 않았다. RunPod소유Pod 정리는 완료, 새 임대와 기존3데이터 정규화seed52·62는 보류다.

**5090의 추가 비교군 원본과 결과를 확정한다 — 완료**

- Instacart S2P2·AttNHP6조건은10월1일16:33 KST에 종료됐다.19:42 KST 관측과 후속 원본 회수·검증을 기준으로 한다. 이번 원고 편집에서 새 원격 조회는 수행하지 않았다.
- 원본223파일·source106개·checkpoint12개의 SHA·CPU 감사를 마쳤고,7모델3seed의 selected/last·구간별 결과를 확정했다. [분석과 감사](../titantpp_instacart_final_analysis_20261001_v1/report.md).

**최종 validation 비교표를 원고에 연결한다 — 완료**

- [현재 원고](../../paper/titantpp_history_mlp_manuscript_20261001_v1.md)의 Pending 두 행과 관련 해석을 갱신했다. Instacart의 작은 MAE 이득과 RMSE 열세를 모두 기록했다.
- 수식9개·그림2개·표7개·참고문헌32편을 유지했다. [반영 기록과 검증](../titantpp_manuscript_integration_20261001_v1/instacart_final_integration_20261001/report.md).

**확정 결과를 초록·결론으로 요약한다 — 다음 작업**

- Taxi·Intermittent의 수량 개선, RAF의 지표별 차이, Instacart의 제한적 이득과 비용 실측을 연결한다. validation에서 확인된 주장 범위를 유지한다.

**독립 평가와 제출본을 준비한다 — 이후 작업 / 평가·제출 승인 필요**

- 모델·checkpoint·지표·불확실성 계산과split 독립성을 고정한 뒤별도승인된평가를 진행한다. 현재validation 결과를독립test 결과로표시하지 않는다.
- 제출본의익명성·분량·자료공개범위를 확인한다. 새GPU실험·held-out 열람·commit/push·외부게시·제출은이번 원고 반영에서 실행하지 않았다.
