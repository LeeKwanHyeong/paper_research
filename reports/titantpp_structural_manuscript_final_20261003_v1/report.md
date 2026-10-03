# 구조 대조 최종 원고 반영

작성일: 2026-10-03 KST. 대상은 `paper_research`, 브랜치는 `codex/hard-lmm-causal-qkv`, 원고는 [기존 main.tex](../../paper/titantpp_pakdd_2027_draft/main.tex)다. 커밋·Push·원격 학습 변경은 수행하지 않았다.

**남은 구조 대조 원본 검증 — 완료**

- Intermittent all-available seed62(PRO4500), Instacart current-only 및 all-available seed62(5090)의 3조건·6 checkpoint가 원본 SHA·109소스·CPU 로딩·선택 epoch·저장 지표·selected/last replay 감사에 통과했다.
- 기존 21개 구조 대조 감사와 원래 MLP 12개를 재사용했다. 원고용 구조 대조 24조건 전체가 검증됐고, 표는 MLP를 포함한 12그룹·36조건의 3seed 평균과 표본 표준편차다. 원래 후속36조건 캠페인 전체 또는 A100 전체의 CPU 감사 완료를 뜻하지 않는다.
- [원본 감사 보고](../titantpp_structural3_final_audit_20261003_v1/report.md), [기계판독 집계](../titantpp_structural3_final_audit_20261003_v1/comparison.json), [조건별 목록](../titantpp_structural3_final_audit_20261003_v1/condition_registry.csv).

**표와 주장의 연결 — 완료**

| 주장 | 최종 확인 근거와 원고 처리 |
|---|---|
| 직접적인 인접 상태 보정 | current-only 대비 MLP 평균 RMSE 감소는 Taxi 11.48%, Intermittent 4.11%, RAF 0.75%. Instacart는 MLP가 0.00943% 높아 거의 같다. 가장 분명한 근거는 Taxi로 제한했다. |
| 단계적 분기 가용성 | all-available 대비 원래 방식의 보편적 우월성은 성립하지 않는다. all-available의 MLP 대비 RMSE 변화는 Taxi −1.99%, Intermittent +0.64%, RAF +0.13%, Instacart −0.08%. MAE·시간 NLL과 seed별 방향도 보존했다. |
| 이종 GPU 출처 | Intermittent 두 대조군의 seed62는 PRO4500, 나머지 seed와 원래 MLP는 5080이다. 효율 측정과 혼합하지 않았다. |
| 수량 예측 기여 | 본 비교84조건 및 단순 기준선 수치는 유지했다. Taxi·Intermittent 중심의 조건별 이득과 RAF·Instacart의 작은 이득/역전, Intermittent last-quantity의 낮은 MAE를 그대로 제시했다. |
| 계산 효율 | 기존 matched batch 측정 범위와 training/evaluation 메모리 차이를 유지했다. A100 탐색 속도나 이관 GPU의 실행시간으로 효율 주장을 확대하지 않았다. |
| 일반화 | 현재 결과는 개발 validation이다. 독립 평가를 수행했다고 표현하지 않고 동결 설계와 남은 실행 적격성을 구분했다. |

Appendix A의 부분 완료 표를 없애고 한 개의 최종 3seed 표로 합쳤다. 초록·Contribution·실험 설명·결과·한계·결론의 연결 문구를 조정했다. Appendix B/C, 그림5개, 기존 비구조 비교표8개, 저자·설정·참고문헌은 그대로다. Deep Renewal과 A100 기록은 별도 연구 기록으로 보존한다.

**독립 평가 설계 — 모델 선택·분석 절차 동결 완료 / 실행 미완료**

- [설계 문서](../titantpp_independent_final_evaluation_design_20261003_v1/README.md)와 [design.json](../titantpp_independent_final_evaluation_design_20261003_v1/design.json)에 대표 MLP, 기존84개 비교조건, 24개 보조 구조조건, deterministic 기준선2개, 선택 epoch와 분석 절차를 고정했다.
- 3seed를 함께 유지하는 paired 재표집을 정했다. Taxi는 공통 달력시간 블록, 나머지는 개체/계열 군집을 기본 단위로 한다. 10,000회·95% 구간과 네 주 비교의 보정 구간, 실패 처리·한 번의 결과 공개 규칙을 명시했다. 실제 평가 실행기를 구현하거나 실제 CI를 계산한 것은 아니다.
- 네 데이터 모두 새 독립 모집단의 적격성을 아직 확정하지 못했다. Taxi·Instacart의 legacy test는 과거 접근된 사건과 겹치며, Intermittent·RAF도 독립성을 증명한 상태가 아니다.
- 108개 selected checkpoint 중72개는 로컬 원본 SHA가 연결됐으며, 외부 comparator36개는 기록된 로컬 경로에 없다. 다른 위치의 부재나 유실을 의미하지 않는다.

**검증 — 완료**

- [verify_revision.py](verify_revision.py)가 36조건으로 평균·표본 SD를 재계산하고 Appendix A의 36개 수치 셀을 대조했다. 수치와 참조가 일치하고 이전 pending 문구가 남지 않았다.
- 별도 읽기 전용 검토에서도 수치·해석·설계의 문제는 없었다. 발견된 Taxi 재표집 설명 누락1건은 공유 시간 블록 문구로 수정했다.
- 최종 저장 소스는 기존 편집기의 built-in compiler에서 성공했다. [최종 컴파일 기록](final_compilation_receipt.json), [검증 기록](verification.json), [원고 변경 diff](main.tex.diff)를 보존한다. 별도 PDF를 컴파일/내보내지 않았고, 전체 시각적 레이아웃 검토는 아직 수행하지 않았다.

**평가 자료 적격성과 남은 원본을 확보한다 — 다음 작업**

- 새 독립 cohort의 출처·기간·접근 이력·cutoff 이전 selection을 확인한다. 외부 comparator36개 원본 위치 확인과 바인딩은 별도로 병렬 진행할 수 있다.
- 두 결과를 실제 평가 대상 manifest에 연결한 뒤 공통 target·인과성·지표·재표집 실행기를 합성/허용된 validation으로 점검한다.

**독립 평가를 실행한다 — 위 준비 이후 / 승인 필요**

- 정확한 population·Runtime·비용·마감과 held-out 예측/성능 열람 범위를 실행 계약에 고정한다. 현재 요청은 설계까지이며 TEST_SESSION_PROTOCOL의 held-out 접근 잠금도 유지되므로 새 평가를 실행하지 않았다.
- 실행 후 모든 사전 지정 결과와 실패를 반영하고 원고 전체 레이아웃 및 최종 투고 형식을 점검한다.
