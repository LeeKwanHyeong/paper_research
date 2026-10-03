**참고문헌 재대조 — 완료:** 32편의 서지·인용 대응을 확인하고 8편의 누락 정보와 형식을 보완했다. [검토 결과](../titantpp_manuscript_integration_20261001_v1/reference_doublecheck_20261001/report.md). 이번 문헌 작업은 저장된 실험 기준선과 관측 시각을 갱신하지 않았다.

# TitanTPP 현재 기준선과 남은 작업 — 독립 평가 기준 정리 완료

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

**split 접근 이력과 원본 준비를 마무리한다 — 다음 작업 / 독립성 판단은 정보 대기**

- 과거 test 결과 생성 흔적과 현재 split의 계보·실제 열람 여부를 확인한다. 현재 네 데이터 모두 미접근 독립test로 확정하지 않았다. 기존test 재평가와 새 독립평가의 경로를 구분한다.
- [84개 checkpoint 목록](../titantpp_independent_evaluation_protocol_20261001_v1/checkpoint_manifest.csv) 중 로컬48개는 기존CPU감사와 byte SHA를 다시 대조했다. 기존 세데이터의 외부4종36개는 결과 기록만 있고 해당 로컬경로에 binary가 없어 원본회수·CPU검증이 남았다. 과거 JSON 결과 감사와 구분한다.
- 접근 이력 조사와 원본 회수는 독립적으로 진행할 수 있다. 새 학습은 필요하지 않다.

**평가 실행기를 준비하고 실제 실행 계약을 고정한다 — 다음 작업**

- 합성·기존validation으로 target pairing·인과성·지표·bootstrap을 검증하고, 적격한 평가 모집단·Runtime·서버·예산·마감·출력경로를 확정한다. 공통 계약을 다루므로 현재 세션에서 순서대로 진행한다.

**고정 평가와 제출본을 완성한다 — 이후 작업 / held-out 실행·성능 열람·제출 승인 필요**

- 별도로 승인된 평가 후 현재validation 근거와 구분하여 초록·결론의 주장 범위를 대조한다. 제출본의 익명성·분량·자료공개범위를 확인한다.
- 이번에는 학습·GPU·원격조회·held-out 성능열람·스케줄변경·commit/push·외부게시·제출을 실행하지 않았다.
