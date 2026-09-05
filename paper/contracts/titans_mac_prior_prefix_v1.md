# B1 prior-prefix 출력 read 계약 v1

- 동결일: 2026-09-05. 후보 성능 학습 전 동결.
- 저장소·브랜치: `paper_research/codex/b1-prior-prefix-read`.
- 실행 기준: [JSON 계약](titans_mac_prior_prefix_v1.json). 실행 source는 독립 커밋과 package manifest의 전체 SHA로 고정한다.
- 사용자 승인 범위: 로컬 계약/구현/검증, 5090 CUDA 및 전체 데이터 e1, seed42 screening, 통과 조건을 충족한 경우 seeds52·62. Held-out은 잠금 유지.
- 현재 논문 기준선은 v0.7 original T0이며 이 실험으로 자동 교체하지 않는다.

## 변경하는 한 가지

기존 B1의 segment16 pre-attention read와 causal attention은 유지한다. 최종 memory read만 segment-start state에서 같은 segment의 앞선 valid 관측 write를 반영한 상태로 바꾼다.

```text
y_i = unchanged causal attention with segment-start memory context
r_i = M_(i-1)(Q(y_i))
z_i = unchanged_B1_fusion(y_i, r_i)
M_i = unchanged_stable_write(M_(i-1), y_i)  # valid observed events only
predict next event i+1 from z_i
```

현재 i 자체를 먼저 써서 읽는 post-write 후보, query token 추가, 새로운 gate, 전이 학습 objective, segment 축소, head/loss 변경은 포함하지 않는다. `z_i`는 write 입력이 아니다. 동일 파라미터·입력에서 attention output과 segment 종료 memory/momentum은 기존 B1과 같다.

## 모델 식별과 초기화

- Backbone: `titantpp_titans_mac_prior_prefix`.
- Memory mode: `titans_mac_prior_prefix`; output policy: `prior_prefix`.
- 기존 B1 기본 policy는 `segment_start`로 유지한다.
- 파라미터 수·state_dict 키·초기 파라미터와 RNG 순서는 B1과 동일하다.
- 활성 후보의 초기 출력 전체가 B1 또는 T0와 같다는 조건은 두지 않는다. 읽는 정보가 다른 직접 교체이기 때문이다. 기존 policy의 forward/state/gradient 회귀와 이전 write가 없는 H1의 일치가 계약이다.
- H2부터 첫 write의 key/value 경로를, H3부터 두 write를 통한 momentum 계수 경로를 검사한다. Quantity head weight=0 초기화가 열리는 시점도 구분한다.
- 같은 tensor shape만으로 모델을 식별하지 않는다. Checkpoint는 backbone·policy·contract ID·inner clip·head와 scope metadata를 검증해 다른 모델로의 resume를 거부한다.

## 공통 학습과 비교군

Intermittent-5000·Taxi·Instacart에 같은 구조와 학습 규칙을 적용한다. Context는 기존 train-only 계약인 520주/256, 168시간/256, 52일/64를 유지한다. 최대 길이에는 target이 포함된다. 매 target window는 독립 state로 시작한다.

Direct log1p quantity MSE, legacy clamped time score, batch128, AdamW lr.001/weight_decay.01, outer clip1, inner associative-gradient clip1, hidden64를 유지한다. 최대300/min40/pat40이며 earliest strict minimum validation joint objective로 checkpoint를 고른다. Full e1만 epochs1/min1/pat1이다.

이번 실행의 effective time intercept cap은 **300**으로 고정한다. 기존 T0/RMTPP/THP의 실제 cap300과 맞추기 위한 명시적인 source compatibility다. 따라서 cap30으로 수행한 과거 stable B1 seeds52/62는 이번 B1 대조군으로 합치지 않는다. B1과 후보를 새 source·동일 seed·cap300으로 각각 fresh 학습한다. [고정 T0·RMTPP·THP references](../results/titans_mac_prior_prefix_20260905/frozen_references.json)는 기존 27개 validation 행을 원수치로 보존한다. 이는 새 paired rerun이 아닌 역사적 reference라는 제한을 보고한다.

현재 time score를 정규화된 NLL로 해석하지 않는다. 모든 성능 gate에서 `time_loss`는 artifact의 기존 `time_nll` 필드에 대응하는 clamped time loss다.

## 로컬·CUDA 완료 기준

| 확인 대상 | 기준 |
| --- | --- |
| 기존 경로 | 원본 revision의 B1 forward/state/gradient를 같은 CPU 연산에서 bitwise 재현 |
| Candidate memory trajectory | 동일 입력·파라미터의 segment 종료 state·momentum·diagnostic 일치 |
| Causality | 미래 event/target 변형이 앞선 출력에 영향을 주지 않음 |
| Padding·state | padding/target write 없음, holes/empty 안전, series reset과 batch permutation 불변 |
| Gradient | H1/2/3/16/17의 허용된 경로와 head가 열린 후 실제 loss gradient 확인 |
| Restore | 같은 backend·RNG에서 model/AdamW 복원 후 다음 step 일치 |
| CUDA reference | output/state atol2e-5, rtol2e-4; gradient atol2e-5, rtol3e-3; optimizer step atol2e-5, rtol3e-4 |
| Stable execution | finite forward/backward/inner state, skip batch나 silent eager fallback 없음 |

CPU prefix의 연산 배치 방식에 따른 수치 비교는 atol1e-7/rtol1e-6이다. 같은 연산 순서를 사용하는 원본 회귀와 same-backend restore의 bitwise 조건을 수치 tolerance 조건과 구분한다.

## 비용 계약

RTX5090에서 batch128, H16/64/255 각각에 대해 동일한 full model forward/backward/AdamW step을 측정한다. Warmup3, 측정10회의 CUDA synchronize된 step 중앙값을 사용한다. Compile 시간은 별도로 기록한다.

- 후보 step 시간은 같은 optimized B1의 **2배 이하**.
- 후보 peak allocated memory는 같은 B1의 **1.5배 이하**, 절댓값 **28GiB 이하**.
- 세 길이 모두 통과해야 e1로 진행한다. Peak 측정은 모델마다 reset하고 다른 모델의 할당을 섞지 않는다.
- T0 대비 시간·메모리 비용도 보고한다.

이는 새 read 정책의 추가 연구 비용 상한이다. 과거 B1이 T0 3배 비용 gate를 실패한 사실을 뒤집거나 T0 수준의 효율을 주장하는 기준이 아니다.

## 서로 다른 성능 판정

### A. B1 대비 추가 효과

Instacart body MAE는 fresh B1 대비 **1% 이상 감소**해야 한다. 세 데이터셋 모두에서 body MAE 악화≤1%, RMSE 악화≤0%, tail MAE 악화≤2%, time loss 증가≤.01을 요구한다. 짧은 이력의 B1 제한을 직접 겨냥하는 추가 효과와 나머지 데이터 보존을 구분한 새 연구 기준이다.

### B. T0 대비 기존 채택 기준

세 데이터셋 각각에서 body MAE **5% 이상 감소**, RMSE·tail MAE 악화 각각≤2%, time loss 증가≤.01을 유지한다. Body는 train p95 이하, tail은 train p99 초과다. 모든 값은 반올림 전 원수치로 계산한다.

### C. T0 대비 공통 raw RMSE 목표

B 기준 통과와 별도로 세 데이터셋 **모두에서 raw RMSE가 엄격히 감소**해야 한다. 전체 raw MAE 악화도≤2%로 제한한다. B의 RMSE 악화 허용 범위를 이용해 공통 RMSE 개선이라고 부르지 않는다.

### D. RMTPP·THP 대비 논문 주장

각 comparator와 각 데이터셋에서 RMSE 감소, 전체 MAE 악화≤2%, time loss 증가≤.01을 별도로 판정한다. 이 조건은 외부 TPP 대비 RMSE 우위 주장용이며 A+B+C의 공통 개선 연구와 구분한다. 모든 지표·모든 TPP 우위를 의미하지 않는다.

Seed42의 A+B+C가 세 데이터셋에서 모두 통과할 때만 seeds52·62로 확장한다. 3-seed 확인은 같은 기준을 산술평균에 적용하고, 각 데이터셋에서 B1/T0 대비 RMSE 감소가 각각 최소2/3 seeds에 나타나야 한다. 개별 seed의 time/tail guardrail도 충족해야 한다. Seed 평균과 방향은 정식 통계적 비열등 보장이 아니다.

## 실행 순서와 중단

**완료 조건 / 로컬**

- 계약, opt-in 모델, 기존 경로 회귀·새 단위 테스트, launch/checkpoint 식별 검증을 완료한 변경을 작업 브랜치에 독립 커밋한다.
- 기존 root 미추적 `scripts/`는 포함하지 않는다. `master`의 최종 모델 계약은 변경하지 않는다.

**후속 / 5090**

- 커밋 archive와 per-file hash로 독립 source 경로를 만들고 기존 데이터는 checksum 검증 후 읽기 전용으로 사용한다.
- Runtime·GPU 점유 확인 → CUDA 필수 테스트·비용 gate → 3 datasets × B1/후보 full e1(6 runs) → 결과 감사 순서다.
- Full e1은 정상 실행·실제 writer 학습·저장/복원만 검증한다. e1 숫자로 성능 채택을 하지 않는다.
- e1 6개 감사가 모두 통과하면 seed42 최대e300의 6 runs를 수행하고 A–D를 계산한다. A+B+C 통과 후에만 52/62의 12 runs를 수행한다.
- Source/data/finite/restore 실패에서는 중단하고 증거를 보존한다. 자동 retry, checkpoint resume, 다른 서버 fallback, 서비스 재배포는 하지 않는다.
- 성능 gate 실패는 실패로 기록하며 후보·threshold·checkpoint를 유리하게 바꾸지 않는다. Held-out은 열지 않는다.
