# 시간·수량 Cross Attention 1-seed pre-check

## 비교 범위와 구조 고정 — 완료

- 대상은 RAF, Taxi, Intermittent의 seed42다. 단순 결합 대조군과 Cross Attention 후보를 각각 학습해 총 6개 fit을 비교한다.
- 이력 MLP 폭16을 공통 출발점으로 사용한다. 기존 CNN 모델·checkpoint·결과는 변경하거나 다시 추론하지 않는다.
- 이번 비교는 **Encoder 앞 시간·수량 결합**의 효과만 다룬다. 메모리를 Encoder 앞으로 옮기는 변경은 포함하지 않는다. 기존 persistent memory와 마지막 Hard-LMM은 두 후보 모두 유지한다.
- 각 관측의 `log1p(dt)`, `log1p(q)`를 별도 16차원 GELU 임베딩으로 만든다. 두 후보에 동일한 관측 순번 위치 표현을 준다.
- 대조군은 각 임베딩을 시점별 16→32→16 GELU MLP로 처리한다. 후보는 시간→수량과 수량→시간의 양방향 Cross Attention으로 현재까지 관측한 이력을 결합한다.
- 두 직접 임베딩과 두 처리 출력을 이어 붙인 뒤 64차원으로 투영하여 기존 입력 표현에 잔차로 더한다. 최종 투영은 0으로 초기화한다. 이후 공통 Encoder 1층→이력 MLP 폭16→Encoder 2층→기존 메모리·head를 유지한다.
- 추가 가중치는 각 6,176개로 동일하다. 같은 파라미터 수가 같은 기능적 표현력이나 연산량을 뜻하지는 않는다.
- Cross Attention의 새 분기는 미래·padding·withheld 관측을 읽지 않고 withheld 경계에서 이력을 끊는다. 상속한 Encoder 전체의 attention 범위를 새로 변경한다는 뜻은 아니다.

## 공통 학습과 선택 유지 — 완료

- 입력·Train/Validation 분할·Train 통계·log1p·수량 손실·시간 head·메모리·AdamW 조건을 기존 계약에서 재사용한다. Test는 실행하지 않는다.
- batch128, hidden64, AdamW lr0.001/weight_decay0.01/grad_clip1, 수량 손실 weight1, tail 보조 손실0이다. 기록된 양의 정수 시간의 lognormal likelihood를 유지한다.
- 최대300epoch, 최소40epoch, patience40. 예비 검사의 속도는 seed 수를 줄이고 병렬 실행하여 높인다. 후보에만 짧은 학습 예산을 적용하지 않는다.
- checkpoint는 전체 Validation raw 수량 RMSE의 최초 엄격한 유한 최소 epoch로 선택한다. MAE는 같은 checkpoint에서 읽는다. 새 학습은 fresh fit이며 기존 checkpoint에서 시작하지 않는다.
- fit별 36시간 제한, 자동 retry/resume 없음. 실패 원문을 보존한다. 공용 Runtime 설치·변경은 하지 않는다.

## 두 전용 GPU에서 병렬 실행 — 다음 작업

| 서버 | 배정 | 동시 fit |
|---|---|---:|
| 5080 / cuDNN91900 | Taxi 대조군·Cross, Intermittent 대조군 | 3 |
| 5090 / cuDNN92000 | RAF 대조군·Cross, Intermittent Cross | 3 |

두 서버의 유휴 GPU와 기존 Runtime, 입력 동일성, 새 모델의 유한 손실·인과 mask·checkpoint 저장을 확인한 뒤 새 실험 경로에서 시작한다. 검사 범위를 과거 실험 전체 감사로 확대하지 않는다.

Intermittent 두 후보는 GPU·cuDNN이 달라 결과를 해당 환경 차이가 섞인 예비 비교로 한정한다. Taxi·RAF 후보 쌍은 각각 같은 서버에서 비교한다.

## 결과 해석과 후속 판단 — 다음 작업

- 학습 종료 후 저장된 선택 checkpoint의 Validation RMSE·MAE와 Cross−대조군 차이를 데이터셋별로 비교한다. 큰수량·시간 지표와 악화도 보존한다.
- 기존 CNN과 이력 MLP 폭16은 저장된 Validation 결과가 동일 대상·조건인지 확인된 경우에만 참고한다. 기존 모델 재평가로 비교 범위를 늘리지 않는다.
- 1-seed 결과는 작동 여부와 다음 검증의 근거를 보는 용도다. seed 안정성, Test 일반화, 전단 메모리의 효과를 확인한 것으로 해석하지 않는다.
- 자동 반복개선 스케줄러는 만들지 않는다. 시작 확인 후 지속 polling을 하지 않는다.

코드·실행 위치: `paper/scripts/manage_titantpp_cross_fusion_precheck.py`, `paper/scripts/titantpp_cross_fusion_precheck_runtime.py`. 실험별 원본은 `search_artifacts/titantpp_time_quantity_cross_precheck6_20261016_v1`에 저장한다.
