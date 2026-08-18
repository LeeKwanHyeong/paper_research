# TitanTPP Memory Backbone Enhancement Contract

## 상태

- 구현: 완료
- 검증 범위: 로컬 CPU focused contract test
- 품질 판정: 미실행
- Held-out test: 미사용

이 단계는 time head, quantity head, log-MSE 및 T1 tail-aware loss를 변경하지 않고
Count-aware TitanTPP의 memory backbone만 분리한다. 구현 완료는 성능 개선을 의미하지
않으며, 이후 validation-only matched screening으로 각 구조를 판정한다.

## 목적

기존 Count-aware TitanTPP의 static persistent token과 hard top-k LMM이 실제 성능에
기여하는지 확인하고, 정적 memory retrieval의 불연속성과 무조건적인 residual 결합을
완화한다. 최종 후보는 관측된 같은-series context만 사용해 갱신되는 causal surprise
memory가 history representation을 개선하는지 검증한다.

## Backbone Variant 계약

| Backbone ID | Memory 구성 | 역할 |
| --- | --- | --- |
| `titantpp` | persistent token 16 + hard top-k LMM 64/4 | 기존 TitanTPP-T0/T1 기준선 |
| `titantpp_no_memory` | persistent token 없음, LMM 없음 | 순수 causal Titan encoder 기준선 |
| `titantpp_gated_soft_memory` | dense soft retrieval 64 + zero-init residual gate | 정적 retrieval 개선 후보 |
| `titantpp_surprise_memory` | rank-16 causal fast weight + chunk 32 + zero-init gate | 동적 memory 후보 |

새 세 Variant는 opt-in이다. 기존 qualified `BACKBONES`와 기대 run 수는 바뀌지 않으며,
runner에서 `--allow-partial-contract`와 명시적 `--backbones`를 사용해야 실행된다.

## 고정 조건

- 입력 feature: `log1p(delta_t)`, `log1p(history_quantity)`
- Encoder core: 2 layers, 4 heads, hidden dimension 64, FFN dimension 128
- Time head: 기존 RMTPP-style continuous density head
- Quantity head: 기존 positive log1p location head
- Quantity loss: T0 log-MSE 또는 고정된 T1 tail-aware loss
- Target quantity: encoder history에서 masking
- Padding: memory retrieval 및 update에서 제외
- State scope: 각 input sequence의 same-series observed context
- Cross-batch state: 전달하지 않음
- Validation/test: forward 호출마다 memory state 초기화

## Gated Soft Memory

Hard top-k index와 선택 memory의 단순 평균을 제거한다. 모든 memory slot에 대해 dense
softmax weight를 계산하고, key와 value parameter 및 projection을 분리한다.

```text
weights_t = softmax(q_t K_memory^T / sqrt(d))
retrieved_t = weights_t V_memory
h'_t = h_t + tanh(alpha) * sigmoid(g_t) * W_o retrieved_t
```

`alpha=0`으로 초기화하므로 최초 forward는 `titantpp_no_memory`와 정확히 같다. Gate를
열면 query, key, value, output projection과 memory parameter에 gradient가 전달된다.

## Surprise-updated Dynamic Memory

Dynamic memory는 learnable static slot을 사용하지 않는다. 각 sequence forward에서
zero state로 시작하며, 현재 position의 출력은 해당 event를 memory에 기록하기 전에
계산한다. 따라서 position `t`의 memory retrieval에는 positions `<t`만 포함된다.

```text
predicted_value_t = M_(t-1) k_t
error_t = v_t - predicted_value_t
gradient_step_t = outer(error_t, k_t) / sqrt(rank)
S_t = momentum * S_(t-1) + update_rate_t * gradient_step_t
M_t = retention_t * M_(t-1) + S_t
```

- Rank: 16
- Chunk size: 32 events
- Initial update rate: 0.01
- Initial retention: 0.99
- Initial momentum: 0.5
- Memory clipping: `[-5, 5]`
- Chunk boundary: memory와 momentum state를 detach해 truncated gradient 적용
- Retrieval residual: zero-init global scale과 event-dependent gate 사용
- CUDA scan: event-local projection을 sequence 단위로 계산하고 전체 recurrent scan을
  하나의 static graph로 compile
- CPU, diagnostics, compile 미지원 환경: 동일 식의 eager scan 사용

Chunk boundary의 detach는 forward 값을 바꾸지 않고 gradient horizon만 제한한다.
현재 구현은 batch 사이의 pre-window memory를 전달하지 않는다. 따라서 이를
cross-window 장기 기억 또는 test-time continual adaptation으로 해석하지 않는다.
Compiled 경로도 graph 내부에서 32 event마다 state를 detach하므로 gradient horizon은
동일하다. Compile wrapper는 parameter를 추가하지 않아 checkpoint key도 바뀌지 않는다.

## Parameter 예산

Hidden dimension 64, max sequence length 256 기준이다.

| Backbone | Parameters |
| --- | ---: |
| Count-aware THP | 100,291 |
| 기존 Count-aware TitanTPP | 89,795 |
| Titan no-memory | 83,651 |
| Titan gated soft-memory | 112,516 |
| Titan surprise-memory | 98,439 |

Surprise-memory는 THP보다 1,852개 적어 parameter-matched 비교에 가장 가깝다.
Soft-memory는 THP보다 12,225개 많으므로 결과 해석 시 별도 capacity guardrail이
필요하다.

## Focused Acceptance Contract

- 기존 `titantpp` checkpoint key와 default forward path가 유지되어야 한다.
- No-memory는 persistent memory와 LMM parameter를 포함하지 않아야 한다.
- Zero-init gated Variant는 eval mode에서 no-memory output과 bitwise 동일해야 한다.
- 미래 event 변경이 이전 hidden state를 변경하면 안 된다.
- Padding event는 dynamic memory state와 diagnostics를 변경하면 안 된다.
- 동일 parameter에서 chunk size 변경은 forward와 diagnostics 값을 변경하면 안 된다.
- Gate를 연 상태에서 memory projection과 update parameter의 gradient가 finite해야 한다.
- Extreme interval과 quantity에서도 forward, loss, gradient가 finite해야 한다.
- 대표 CUDA shape에서 Surprise/hard-LMM 학습 step 비율이 모두 `3.0x` 이하여야 한다.
- Compiled/eager CUDA output과 input/parameter gradient가 허용 오차 내에서 같아야 한다.

## 현재 판정과 다음 Gate

구조 구현, 로컬 계약, 5080 CUDA 계약과 runtime gate까지 완료됐다. 최종
Surprise/hard-LMM 학습 step 비율은 L16 `1.36x`, L64 `1.64x`, L256 `2.61x`다.
세 shape의 최초 compile, warmup, 측정과 profiler를 포함한 cold run은 약 4분 58초가
걸렸으므로 compiled 경로는 long-epoch 학습용으로 해석한다.

Backbone 개선 주장을 위해서는 T0 loss를 고정한 `기존 Titan / no-memory /
gated soft-memory / surprise-memory` fresh matched validation 비교가 다음 단계다.
이 비교를 통과한 후보에만 T1 loss를 적용한다.

1차 validation gate는 기존 Titan-T0 대비 전체 MAE 또는 RMSE 5% 이상 개선,
`<=p95` MAE 악화 2% 이하, time NLL 악화 0.01 이하, 모든 값 finite로 둔다.
THP 대비 backbone contribution은 동일 T0 loss와 parameter budget에서 별도로
판정하며, held-out test는 후보 동결 전까지 사용하지 않는다.
