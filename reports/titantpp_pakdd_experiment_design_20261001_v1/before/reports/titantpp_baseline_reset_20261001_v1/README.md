**참고문헌 재대조 — 완료:** 32편의 서지·인용 대응을 확인하고 8편의 누락 정보와 형식을 보완했다. [검토 결과](../titantpp_manuscript_integration_20261001_v1/reference_doublecheck_20261001/report.md). 이번 문헌 작업은 저장된 실험 기준선과 관측 시각을 갱신하지 않았다.

# TitanTPP 현재 기준선과 남은 작업 — 독립 평가 CPU 실행 준비 완료

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

**확정 결과를 초록·결론으로 요약한다 — 완료**

- [현재 원고](../../paper/titantpp_history_mlp_manuscript_20261001_v1.md)에 초록과8절 결론을 추가했다. Taxi·Intermittent의 수량 개선, RAF·Instacart의 지표별 차이, 비용 실측을 연결했다.
- 기존 본문과 수치는 유지했다. [작성·검증 기록](../titantpp_manuscript_integration_20261001_v1/abstract_conclusion_20261001/report.md).

**독립 평가 기준을 정리한다 — 완료**

- [평가 기준](../titantpp_independent_evaluation_protocol_20261001_v1/protocol.md)에 대표MLP·외부6종·네데이터·3seed의84개 선택checkpoint, RMSE·MAE·시간NLL, paired bootstrap 절차를 고정했다. seed 표준편차와 평가 표본 불확실성을 구분한다.
- 기존 validation 결과·원고는 유지했다. 이번 작업은 독립 성능을 새로 평가하거나 열람하지 않았다.

**원본 checkpoint와 CPU 평가 경로를 준비한다 — 완료**

- 외부4종36개 selected checkpoint와 원본755파일을 회수·SHA·CPU 검증했다. 기존48개와 합쳐 총84개 파일을 준비했다. [회수·검증 보고서](../titantpp_independent_evaluation_preparation_20261001_v1/report.md).
- 9개 동결 source에서84조건의 validation 표본462행을 확인하고, 통계 코드14검사를 통과했다. 전체 validation 재평가나 held-out 성능 검증과는 구별한다.
- Intermittent의50사업장 구조를 반영해 기본 bootstrap을 사업장 단위로 보완했다. 원고·기존 benchmark 수치·과학 source는 유지했다.

**평가 데이터의 성격을 확정한다 — 다음 작업 / 외부 작업 대기**

- metadata 계보 조사를 마쳤으나 실제 test 열람·모델 선택 사용 여부는 연구자 확인이 남았다. 네 데이터의 미접근 독립성은 미확정이다. 기존 split의 잠금 재평가와 미사용 코호트의 독립 평가를 구분해 선택한다.

**전체 평가 실행기와 실제 계약을 확정한다 — 다음 작업**

- 적격한 target 모집단을 고정한 뒤 전체 평가 wrapper·native runtime·자원 상한·마감·출력경로를 확정한다. 현재 CPU validation 실행기는 준비용이며 [실행 계약 초안](../titantpp_independent_evaluation_preparation_20261001_v1/execution_contract_draft.json)은 실행 불가 상태다.
- 공통 모집단과 계약에 의존하므로 현재 세션에서 순서대로 진행한다. 원본 회수·Core 감사·효율 측정은 반복할 필요가 없다.

**고정 평가와 제출본을 완성한다 — 이후 작업 / 실제 실행·성능 열람·제출 승인 필요**

- 검토 가능한 계약에 따라 승인된 평가를 실행하고 불리한 결과·실패도 보존한다. 결과의 적용 범위를 초록·결론과 대조한 뒤 제출본을 편집한다.
- 이번 준비에서는 읽기 전용 원본 회수·Runtime inventory와 로컬 CPU validation 표본 추론만 추가했다. GPU 작업·held-out 성능 열람·스케줄 변경·commit/push·외부 게시·제출은 수행하지 않았다.
