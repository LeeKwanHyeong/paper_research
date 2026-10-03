# TitanTPP 현재 기준선과 남은 작업 — RAF 결과 통합

**완료 결과를 현재 기준선으로 고정한다 — 완료**

- 대표 모델은 `titantpp_history_mlp`이며 활성 분기 정규화는 별도 탐색 변형이다. 동일 RMSE 선택 checkpoint의 validation MAE·RMSE·시간 NLL을 함께 보고한다.
- Core54조건, 기존 외부4종36조건, Gate6조건, 5080 추가TPP12조건, 기존3데이터 정규화seed42, 효율 측정은 완료 증거를 재사용한다.
- RAF8모델×3seed24조건은2026-10-01 08:55:56 KST 완료됐다. 원본473파일·source104개·checkpoint48개 회수·SHA·CPU 감사 및 보고서·원고 반영을 마쳤다. [RAF 상세](../titantpp_raf_execution_20261001_v1/report.md).
- [전체 보고서](../titantpp_completed_external_comparison_20261001_v1/report.md)는139개 고유 완료조건과42개3seed 그룹을 담는다. Taxi·Intermittent의 수량 우위, RAF의 근소한RMSE 이득과MAE 반례, Instacart·시간NLL 한계를 함께 보고한다.
- 과거07:35 기준선은 before_raf_integration/에 보존했다. 스케줄러 삭제 결과는 기존증거를 재사용하며 새 자동화를 만들지 않았다. RunPod소유Pod 정리는 완료, 새 임대와 기존3데이터 정규화seed52·62는 보류다.

**5090의 승인된 추가 비교군을 마무리한다 — 진행 중 / 서버 작업 대기**

- 대상은Instacart S2P2·AttNHP6조건이다. 최신 저장 관측2026-10-01 08:59 KST에서4/6완료, S2P2 seed62 학습 중, AttNHP seed62 대기였다. 이번RAF 통합에서는5090을 새로 조회하지 않았다.
- 종료 후 원본을 회수하고CPU로 검사해 전체3seed 표를 확정한다. 기존마감10월5일12:16 KST는 그대로며 추가학습·튜닝·자동retry를 시작하지 않는다.

**최종 비교표를 원고에 연결한다 — 다음 작업**

- [현재 원고](../../paper/titantpp_history_mlp_manuscript_20261001_v1.md)는RAF를 포함해수식9개·그림2개·표7개를 담는다. 관련 연구와 본문 인용·BibTeX는32편으로 보강 완료했다. [문헌 보강 기록](../titantpp_manuscript_integration_20261001_v1/reference_expansion_review.md)을 기준으로 데이터 출처와 최종 조판을 이어 정리한다.
- Instacart 추가비교군이 끝나면 누락 없이3seed 평균·표본표준편차·selected/last·구간별 반례를 통합한다. 네 데이터 최종주장과초록·결론은 이 결과에 맞춘다.

**독립 평가와 제출본을 준비한다 — 이후 작업 / 평가·제출 승인 필요**

- 모델·checkpoint·지표·불확실성 계산과split 독립성을 고정한 뒤별도승인된평가를 진행한다. 현재validation 결과를독립test 결과로표시하지 않는다.
- 제출본의익명성·분량·자료공개범위를 확인한다. 새GPU실험·held-out 열람·commit/push·외부게시·제출은이번RAF 반영에서실행하지 않았다.
