# Count-aware Titans Backbone 재현 계약 v1

- 동결일: 2026-08-26
- 원본 근거: [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- 범위: 원본 Titans의 neural long-term memory와 MAC 결합을 count-aware TPP에
  맞게 재현하는 구조 계약
- Held-out test: 잠금 유지

## 명칭 감사

현재 T0에서 과거에 `Hard-LMM`이라고 부른 모듈은 정적 학습 prototype 64개에서
cosine top-k 4개를 선택해 동일 가중 평균을 더하는 **Hard Local Memory Matcher**다.
코드의 정식 이름은 `HardLocalMemoryMatcher`이며, `LMM`과
`static_hard_lmm`은 기존 checkpoint와 artifact를 읽기 위한 호환 명칭으로만 남긴다.
이 구조는 원본 Titans의 **Long-term Memory Module**이 아니다.

원본 Long-term Memory Module은 MLP의 파라미터 자체를 online state로 사용한다.
관측 입력으로 만든 key-value 연관 오차의 gradient를 surprise로 계산하고, validation과
test에서도 momentum 및 adaptive forgetting과 함께 해당 state를 갱신한다. 이 문서에서
`Titans LTM` 또는 `Neural Long-term Memory`는 이 원본 메커니즘만 의미한다.

## 비교 구조

| ID | 구조 | memory 성격 | 논문 역할 |
| --- | --- | --- | --- |
| B0 | Current Hard-LMM | 정적 Hard Local Memory Matcher + persistent token | 현재 T0 대조군 |
| B1 | Faithful Titans-MAC | deep neural associative memory + surprise/momentum/forgetting + MAC | 원본 메커니즘 대조군 |
| B2 | TPP-specific Gated Memory | event-domain causal gated dynamic memory | 향후 제안 backbone 후보 |

B2는 B1 계약 테스트가 모두 통과하기 전까지 구현·성능 실험 대상으로 열지 않는다.
B1은 원본 메커니즘을 count-aware event 입력과 다음-event 예측 순서에 맞춘 재현이며,
원 논문의 대규모 언어 모델 설정이나 parameter count를 그대로 복제한다는 뜻은 아니다.

## B1 Neural Memory

관측 표현 `x_t`에서 SiLU projection과 query/key L2 normalization을 적용한다.

```text
k_t = l2_norm(silu(W_K x_t))
v_t = silu(W_V x_t)
q_t = l2_norm(silu(W_Q x_t))
loss_t = ||M_(t-1)(k_t) - v_t||^2
S_t = eta_t S_(t-1) - theta_t grad_M(loss_t)
M_t = (1 - alpha_t) M_(t-1) + S_t
read_t = M_(t-1)(q_t)
```

`alpha_t`, `theta_t`, `eta_t`는 관측 token에 의존한다. `M`은 SiLU를 사용하는
2-layer MLP이며 hidden width는 encoder dimension의 2배다. 초기 MLP parameter는
outer-loop에서 학습하지만, 실행 중의 `M_t`와 `S_t`는 batch row별 online state다.

## B1 Memory as Context

sequence를 16-event segment로 나눈다. Segment `i`는 시작 시점의 `M_(i-1)`로
현재 관측 segment에 대응하는 long-term memory를 먼저 읽는다. Persistent memory,
retrieved memory, current observed segment 순으로 attention context를 구성한다.
Attention output으로 다음-event 예측 state를 먼저 만든 뒤, valid current-segment
output만 memory에 기록한다. 기록이 끝난 state는 다음 segment부터 읽을 수 있다.

Attention output gating은 유지하되, 원본 MAC 식 (25)의 post-write retrieval은 다음
event를 예측하기 전에 현재 event를 쓰지 않는 stricter TPP causality를 위해 prediction
state에 사용하지 않는다. Prediction state는 segment-start memory read로 완성하고,
write는 그 뒤에 수행한다. 이 차이는 원본 메커니즘과 event-domain adaptation을 구분해
보고한다. B0와의 비교에서는 quantity head, direct log-MSE,
`legacy_clamped_rmtpp` time head, optimizer와 checkpoint selection을 변경하지 않는다.

## Event Causality

1. 입력에는 현재까지 관측된 event의 delta-time과 quantity만 포함한다.
2. Segment read는 해당 segment를 쓰기 전의 state만 사용한다.
3. Hidden state를 만들고 다음-event target을 예측한 뒤에만 현재 관측 표현을 쓴다.
4. Padding과 다음-event target은 key, value, update rate와 memory write에 사용하지 않는다.
5. 기본 `encode` 호출은 input sequence별 독립 state로 시작한다.
6. Streaming은 explicit state-in/state-out API만 허용하며 series ID가 바뀌면 초기화한다.
7. Batch row, validation series와 test series 사이에서 memory·momentum을 공유하지 않는다.
8. Persistent memory는 validation·test에서 고정하고 neural long-term state만 갱신한다.

## 성능 실험 진입 조건

- Prediction-before-write와 future-target 불변성
- Padding 무시와 batch row·series state 격리
- Token scan과 chunk wrapper의 허용 오차 내 일치
- Momentum, forgetting, persistent parameter의 갱신 경계
- Extreme finite input의 forward, backward 및 online update finite
- 기존 B0 state dict 및 forward 회귀 테스트

하나라도 실패하면 B1 성능 학습과 B2 구현으로 넘어가지 않는다.
