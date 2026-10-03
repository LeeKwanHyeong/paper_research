# TitanTPP의 이력 보완을 검증하는 두 비교

**비교 질문과 계산식을 고정한다 — 설계 완료**

현재 대표 모델 `titantpp_history_mlp`를 유지하고, 같은 입력·encoder·head·loss·선택 기준에서 다음 두 대안을 비교한다. 식과 CPU 참조 구현은 설계 검증용이며 학습 Factory에 등록하지 않았다.

기호: 첫 encoder의 관측 상태는 \(h_i\in\mathbb R^{64}\), 직전 관측 상태는 \(h_{p(i)}\), 현재 유효 구간의 관측 수는 \(n_i\), 분기 임계값은 \(\tau=(1,2,4,8,16,32,64,128)\)이다. padding은 세지 않고, 유효하지만 관측에서 제외된 행은 보완의 구간 수를 초기화한다. 이 규칙이 encoder attention까지 초기화한다는 뜻은 아니다.

기준선:

\[
a_{ib}=\mathbf1[\text{observed}_i\land n_i>\tau_b],\qquad
r_i=\frac18\sum_{b=1}^{8}a_{ib}V_b\operatorname{GELU}(U_b[h_i;h_{p(i)}]),
\]

\(U_b\in\mathbb R^{4\times128}\), \(V_b\in\mathbb R^{64\times4}\). bias가 없고 GELU는 exact이다. \(h_i+r_i\)가 다음 encoder로 들어간다. **여덟 분기는 같은 직전 사건을 읽는다.** 임계값은 서로 다른 lag를 지정하지 않는다.

### A. 같은 파라미터 수로 현재 상태만 보완

질문: 직전 상태를 명시적으로 다시 넣는 것이, 같은 크기의 일반 MLP를 추가하는 것보다 유용한가?

\[
r_i^{\mathrm{current}}=\frac18\sum_{b=1}^{8}a_{ib}V'_b\operatorname{GELU}(U'_bh_i),
\quad U'_b\in\mathbb R^{6\times64},\quad V'_b\in\mathbb R^{64\times6}.
\]

- 추가 파라미터는 두 구조 모두 **6,144개**다: 기준선 \(8(128\cdot4+4\cdot64)\), 대안 \(8(64\cdot6+6\cdot64)\).
- 마스크·고정 /8·삽입 위치·zero output initialization을 유지한다. 입력을 없앤 만큼 bottleneck을 4에서 6으로 늘려 용량을 맞춘다.
- 현재 상태 자체가 인과 encoder의 이력을 포함하므로 **‘이력이 없는 모델’이 아니다.** 명시적 인접 상태 결합을 검증하는 대안이다.
- bottleneck 폭도 달라지므로 순수하게 한 연산만 제거한 실험으로 해석하지 않는다. 파라미터 수가 같아도 실제 시간·메모리까지 같다고 가정하지 않는다.

### B. 직전 사건이 있으면 모든 분기를 사용

질문: 이력 길이에 따라 분기를 단계적으로 사용하는 규칙이 유용한가?

\[
a^{\mathrm{all}}_{ib}=\mathbf1[\text{observed}_i\land n_i>1],\qquad
r_i^{\mathrm{all}}=\frac18\sum_{b=1}^{8}a^{\mathrm{all}}_{ib}V_b\operatorname{GELU}(U_b[h_i;h_{p(i)}]).
\]

- 분기 수·rank4·파라미터·초기 tensor를 기준선과 같게 유지한다. 첫 관측과 padding·withheld 행의 보완은 0이다.
- **분모는 계속 8이다.** 이미 수행한 활성 분기 수 정규화와 다른 비교다.
- 원본 `lag_source_indices`는 비활성 분기의 index를 0으로 바꾼다. 원본 마스크만 켜면 잘못된 사건을 읽는다. 실제 lag-one source를 여덟 분기에 복제해야 한다.
- 짧은 이력에서 쓰는 분기 수와 잔차의 크기가 함께 바뀐다. 결과를 단순한 정규화 효과나 threshold 자체의 독립 효과라고 단정하지 않는다.
- 최대 입력 길이 때문에 영구적으로 비활성인 분기가 있을 수 있다. 예를 들어 Instacart 최대64에는 다음 target이 포함되어 관측 이력은 최대63개이므로 기준선의 임계값64·128 분기는 사용되지 않는다. 총 파라미터와 실제 활성 분기 수를 구별한다.

**실험 조건과 판단 방식을 고정한다 — 설계 완료, 학습 미시작**

| 항목 | 고정 조건 |
|---|---|
| 신규 조건 | 두 구조 × Taxi·Intermittent·Instacart·RAF × seed42·52·62 = **24조건** |
| 재사용 | 대표 MLP 네 데이터 12조건. 보완 없는 B는 기존 세 데이터 9조건만 있으며 RAF B 결과는 없음 |
| 초기화 | 기존 base encoder/head tensor·RNG를 보존하고 새 V는 0. 전체 초기 예측 동일성을 구현 단계에서 확인 |
| 학습 | 전체 train, batch128, max300/min40/patience40, 기존 AdamW·lr·clip·loss·loader 유지 |
| 선택 | 최초 최소 validation raw quantity RMSE. 동일 checkpoint의 MAE·RMSE·시간 NLL, selected/last 함께 기록 |
| 비교 | 같은 데이터·seed의 MLP와 차이, 3seed 평균·표본 SD, 각 seed 승패. 지표별 불리한 결과도 포함 |
| 해석 | MLP가 두 대안을 이기면 설계 근거 강화. 차이가 작거나 대안이 낫다면 해당 구성의 필요성 주장을 축소 |

기존 B와의 차이는 보완 모듈 전체의 효과이며, 외부 모델 대비 전체 개선을 모두 이 보완 모듈의 효과로 돌리지 않는다. validation 결과로 구조를 바꾸면 그 변경까지 마친 뒤 최종 평가 모델을 다시 잠근다.

**구현 전 핵심 오류 가능성을 검증한다 — CPU 참조 검증 완료**

[참조 코드](design_reference.py)·[검사](test_design.py)는 동결 source와 식의 일치, 6,144개 파라미터, 초기 잔차0, RNG 보존, 미래 값 비참조, padding·withheld 처리, 새 활성 분기의 실제 predecessor를 확인한다. 실제 통합 모델·CUDA·checkpoint 저장/재개·학습 속도 검증은 다음 구현 단계에 남아 있다.
