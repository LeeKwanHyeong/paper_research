# 분리한 PHEN 아키텍처

## 전체 흐름

```text
관측 이벤트 [log1p(delta_t), log1p(quantity)]
  -> Linear 2 -> 64 + learned positional embedding
  -> Causal Encoder Block 1 (persistent token 16개)
  -> Shared Adjacent-History Correction
  -> Causal Encoder Block 2 (persistent token 16개)
  -> Static Prototype Retrieval (64개 중 cosine top-4의 단순 평균을 더함)
  -> 마지막 관측 이벤트의 상태
       -> Heteroscedastic Log-normal Duration Head
       -> Softplus Log1p-Quantity Regression Head
```

두 block은 pre-norm attention과 FFN(64 -> 128 -> 64)을 사용한다. attention은
4-head, dropout은 0.1이다. 다음 이벤트를 맞히며, 고정된 미래 달력 구간의 수요
벡터를 직접 출력하지 않는다. mark 분류나 exponent-residual 수량 표현도 없다.

## History correction

```text
relation_i = GELU(W_current h_i + W_previous h_previous)  # 15차원
gate_ij = observed_i AND (현재 segment의 관측 개수 > threshold_j)
correction_i = sum_j gate_ij * W_out_j relation_i / 8
```

threshold는 `[1, 2, 4, 8, 16, 32, 64, 128]`이다. 8개 branch 모두 동일한
직전 관측 상태를 사용하며, 서로 다른 8개 lag를 조회하지 않는다. padding은
이력 개수를 늘리지 않고, 가린 valid 이벤트는 segment를 끊는다.
출력 가중치는 0으로 초기화한다. 이 모듈의 파라미터는 9,600개다.

Instacart는 target 포함 최대 64개이므로 관측 이력이 최대 63개다. 따라서 마지막
두 gate는 켜지지 않는다. RAF의 길이 84에서는 마지막 gate가 켜지지 않는다.
긴 window에서 이 gate들이 사용될 수 있으므로 이번 추출에서는 삭제하지 않았다.
활성 branch 수로 나누는 방식으로 바꾸는 것도 원래 PHEN과 다른 모델이다.

## 두 종류의 memory

- block 내부 persistent token: attention key/value 앞에 붙는 학습 파라미터다.
- block 뒤 prototype bank: 현재 상태와 cosine similarity가 높은 4개 vector를
  골라 원래 vector를 단순 평균한다. similarity-weighted average가 아니다.

prototype 파라미터는 64 x 64 = 4,096개다. 어느 쪽도 추론 중 온라인으로 쓰거나
gradient update하는 neural long-term memory가 아니다.

## Target과 loss

window에는 context 뒤에 target이 한 칸 붙는다. target의 gap·quantity는 encoder에
노출하지 않고, 바로 앞 관측 위치의 상태로 예측한다. 입력 padding은 오른쪽이다.
이는 예전 loader의 왼쪽 padding을 objective 직전에 오른쪽으로 옮긴 결과와 같다.

```text
log_quantity = softplus(quantity_head(hidden))
quantity = expm1(log_quantity)
quantity_loss = MSE(log_quantity, log1p(true_quantity))

mu = v_t(hidden) + b_t
sigma = 0.001 + softplus(time_scale_weight(hidden) + w_raw)
latent_duration = time_scale * exp(Normal(mu, sigma))
joint_loss = recorded_duration_NLL + quantity_loss
```

관측 시간은 `max(1, round(latent_duration))`이며, Instacart는 top-code 30을
추가한다. 정수값에서 연속 밀도를 평가하는 대신 해당 정수 구간의 확률 질량을
float64로 계산한다. 시간 median은 latent duration의 median이며, 반올림·top-code
이후 관측 분포의 median과 혼동하지 않는다.

## 기존 구현과의 대응

| 기존 source | 새 source |
| --- | --- |
| `CountAwareTitanSharedHistoryMLP` + 상속된 encoder 경로 | `phen/models/model.py::PHEN` |
| `SharedHistoryCorrection(mode="shared")` | `phen/models/components.py::HistoryCorrection` |
| `MemoryEncoder`의 실제 두 block | `EventEncoder`, `EncoderBlock`, `PersistentAttention` |
| `HardLocalMemoryMatcher`의 static top-4 경로 | `PrototypeRetrieval` |
| `SharedTimeCountModel`의 사용 중인 두 head | `phen/models/heads.py::NextEventModel` |
| `positive_integer_time.py` | `phen/time_likelihood.py` |
| `RMTPPWeekLookbackDataset` + right-padding | `phen/data.py::EventWindows` |
| runner별 factory patch | `phen/registry.py`의 명시적 모델 선택 |

가중치 키는 `multilag_detail.* -> history_correction.*`,
`lmm.* -> prototype.*`로 바뀐다. 두 legacy architecture identity buffer는
새 모델의 파라미터가 아니므로 변환할 때 제외한다. 나머지 tensor는 strict하게
일치해야 한다. 상속 계층, 비활성 head, 사용하지 않는 memory mode, 시간 모델 후보,
실험 scheduler, 임시 model-role alias는 새 runtime에 포함하지 않는다.

## 외부 비교군

각 encoder는 `phen/baselines/` 아래 별도 파일에 있고 head만 공유한다.

| 이름 | 유지한 encoder |
| --- | --- |
| RMTPP variant | 2차원 입력 projection + GRU |
| THP variant | 기존 post-norm causal Transformer 2층 |
| NHP variant | 기존 continuous-time LSTM recurrence |
| SAHP variant | temporal encoding + attention + continuous decay |
| S2P2 matched-head | pinned backward-ZOH kernel 2층 + 기존 안정화 scan |
| AttNHP matched-head | pinned context/query two-stream attention 2층 |

모두 mark-free 입력과 같은 quantity/time head를 사용한다. native likelihood를
비교하는 것이 아니라, 이 예측 문제에 맞춘 encoder 비교다.
