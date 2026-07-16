## 2026-07-17 | V6 Causal Series Memory

### Step 1. 다음 모델 강화 가설 선정

**상태:** 가설 선정 완료 · 구현/실험 미시작 · active model 변경 없음

**목적:** V2와 Taxi V3b가 보지 못하는 168시간 창 이전의 same-series 이력이
추가 예측 정보를 갖는지 확인하고, Titan memory를 output-head 변경과 분리해
검증한다.

**선정 가설:** 기존 `series_lmm` switch를 직접 사용하지 않는다. V2/V3b의 static
LMM을 유지하고, target과 current window보다 앞선 동일 series event만 masked
retrieval한 뒤 zero-init bounded gate로 residual fusion한다.

```text
h_v6 = h_base + tanh(alpha_series) * r_pre_window
alpha_series = 0
```

**Factorial 계약:**

| Variant | Value head | Quantity-mark route | Series adapter | 역할 |
| --- | --- | --- | --- | --- |
| V2 | shared | coupled | off | 공통 control |
| V3b | mark-conditioned experts | detached | off | Taxi incumbent |
| V6a | shared | coupled | on | memory 단독 효과 |
| V6b | mark-conditioned experts | detached | on | Taxi 승격 후보 |

**고정 조건:** Taxi `mid_lmm`, lookback `168`, max sequence `256`, residual input,
hybrid loss, plain CE, target-only, shared time head를 유지한다. Memory는 반드시
`memory index < context start <= context end < target index`를 만족하고 같은
series만 사용한다. Empty memory와 zero gate에서는 paired V2/V3b와 정확히 같아야
한다.

**선정 이유:** Taxi series 길이는 평균 `420.76`, 중앙값 `405`로 active window보다
길고 같은 131개 series가 반복된다. 반면 learned series ID, stateful TTM, 기존
`series_lmm` 직접 활성화는 각각 memorization, order/leakage, baseline replacement
문제가 있어 첫 V6에서 제외한다.

**실행 명령어:** 없음. 이번 단계는 가설 선정이며 5090 학습을 시작하지 않는다.

**결과:** V6 causal pre-window memory를 다음 hypothesis로 선택했다. V2와 Taxi
V3b의 승격 상태는 유지하며 V6는 아직 성능 모델이 아니다.

**다음:** 5090에서 Taxi fixed-split train row만 읽는 pre-window coverage와
predictiveness audit을 실행한다. Validation/test는 읽지 않고, audit 통과 후에만
memory budget/top-k를 freeze하고 구현을 연다.

### Notion 직접 반영 결과

- 반영 시각: `2026-07-17 06:40:02 KST`
- 상위 페이지: `5. Model Design Enhancement`
  (`https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`)
- 전략 페이지: `TitanTPP Model Enhancement Strategy`
  (`https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a`)
- 생성 페이지: `TitanTPP V6 Causal Pre-Window Series Memory Hypothesis`
  (`https://app.notion.com/p/39fbbe4056138118871fcd18c6b31174`)
- 상위 날짜/Step 배치, 전략 페이지의 V6 `가설 선정` 상태, active model 불변,
  기존 `series_lmm` 미사용, train-only audit 선행 조건을 재조회 확인했다.
