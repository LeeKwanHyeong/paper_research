# 최종 평가 데이터의 성격과 사용 경로

**과거 사용 여부를 판정한다 — 완료, 미접근 독립성은 미확정**

연구자는 우리 프로젝트의 test 성능·예측 결과를 직접 보았는지에 대해 **“기억이 불확실함”**이라고 답했다. 다른 논문의 공개 결과와 지금까지 비교한 validation 결과는 이 질문에서 제외했다. 이 답변은 test를 사용했다는 증거도, 사용하지 않았다는 확인도 아니다.

현재 Taxi·Intermittent·Instacart·RAF의 기존 test는 모두 **과거 노출 여부가 불확실한 보존 split**으로 분류한다. 기존 모델의 train에 포함됐다는 판정과는 구별한다. 미접근 독립 test라는 표기는 사용하지 않는다.

| 근거 | 확인된 사실 | 아직 알 수 없는 부분 |
|---|---|---|
| [이전 계보 조사](../titantpp_independent_evaluation_preparation_20261001_v1/lineage_review.json) | 과거 test 이름의 산출물과 현재 경로를 참조하는 manifest가 존재한다 | 당시 bytes의 동일성, 사람의 열람과 선택 사용 |
| [2026-09-27 계약 7절](../titantpp_followup_contract_5080_20260927_v1/experiment_contract.md) | manifest 미리보기에서 held-out 목표 분포 요약이 우발적으로 노출됐다는 기록이 있다 | 그 기록만으로 데이터별 모델 선택 영향을 입증할 수 없다 |
| Intermittent 계보 | 전체 기간 event 수를 사용하는 샘플 선정, 과거와 다른 ID 구성, 원본 parquet 현재 부재 | ID 문자열 불일치만으로 원시 사건의 비중복성을 증명할 수 없다 |
| RAF 계보 | 현재 split과 validation 전용 실행 기록이 있다 | 전체 과거 열람 이력을 포괄하는 증거는 아니다 |
| 이번 연구자 확인 | 실제 test 열람 기억이 불확실하다 | 데이터별 미사용 확약 없음 |

과거 요약 노출 수치는 이번에 다시 열거나 복사하지 않았다. 원본 manifest·성능표·예측 파일도 수정하지 않았다.

**기존 test를 사용할 때의 평가 절차를 정한다 — 설계 완료, 실행은 다음 단계**

1. train·validation에서 두 구조 비교와 직접 비교군 개발을 먼저 끝낸다. 주모델, 비교군, seed, checkpoint, 지표, 통계 절차를 모두 고정한다.
2. 고정된 전체 비교 목록에 기존 test를 한 번 평가한다. 모델별로 유리한 데이터나 checkpoint를 고르지 않는다.
3. 이를 `reserved-split evaluation with uncertain prior exposure`로 보고하고, 과거 노출 불확실성을 방법/한계에 명시한다. “untouched”, “previously unseen”, “independent confirmation”의 근거로 쓰지 않는다.
4. 평가 뒤 같은 test에 맞춰 모델·설정을 바꾸면 후속 탐색으로 별도 표시한다. 기술적 오류 정정도 원인과 모든 영향 모델의 처리 범위를 기록한다.

이는 평가를 무의미하게 만드는 판정이 아니다. validation 밖의 성능을 확인할 수 있지만, 과거 개발과 완전히 독립이라는 보증은 제공하지 못한다.

**독립 확인용 모집단을 확보한다 — 다음 작업 / 외부 작업 대기**

권장 보완은 개발에 쓰이지 않은 이후 기간 또는 외부 출처의 사건 집합이다. 실제 후보 파일·접근 이력이 없는 상태에서 특정 기간을 확보했다고 쓰지 않는다. 현재 `independent_population_id=null`이다.

- 확보할 정보: 출처와 허가, 수집 기간·시간 단위, entity/event ID 구성, 파일 SHA, 이전 학습·선택·열람 여부.
- 결과를 열기 전에 고정할 내용: 대상 entity와 예측 시점, 포함/제외 규칙, 시간 상한·단위 변환, 이전 관측을 입력으로 허용하는 범위, 모든 모델에 공통인 target ID.
- 같은 사건을 재분할하거나 파일명을 바꾸어 미사용 test로 만들지 않는다. 미래 기간의 같은 entity는 시간 외삽 평가이며, 새 entity 평가와 구분한다.
- 새 데이터에 적응 학습이나 normalization 재추정이 필요하면 zero-shot 평가로 부르지 않고 별도 train/validation/test 계약을 작성한다.
- 충분한 새 데이터가 없으면 기존 split의 제한을 공개한 평가 경로를 유지한다. PAKDD 제출 가능 여부는 나머지 실험 근거와 함께 판단하며, 새 독립성 주장을 만들지 않는다.

**평가 불확실성 계산을 이어받는다 — 현재 기준선**

[유효 프로토콜](../titantpp_independent_evaluation_preparation_20261001_v1/protocol_effective.json)의 paired bootstrap과 seed 표준편차 구분을 유지한다. Intermittent는 사업장 전체 cluster, Instacart는 사용자, RAF는 부품, Taxi는 시간 block 기준이다. 새 모집단의 의존 구조가 달라지면 결과를 보기 전에 다시 확정한다. 기존 84개 selected checkpoint는 준비 완료 상태이며, 추가 실험 완료 후 새 checkpoint를 registry에 확장한다.
