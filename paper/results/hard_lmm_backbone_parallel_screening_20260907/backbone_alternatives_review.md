# Hard-LMM Backbone alternatives: evidence review, 2026-09-07

이 문서는 검토 결과와 후보 제안이다. 새 모델 계약의 승인·구현·학습 결과가 아니다.
검토 당시 로컬 브랜치는 `codex/hard-lmm-backbone-candidates`, 검토 시작 HEAD는
`6ef0357`이며 두 GPU의 종료된 학습 소스는 `ba9c297`이다.

후속 승인으로 Q/K/V 후보를 구현하는 계약은
`paper/contracts/hard_lmm_causal_qkv_v1.md`에 별도로 고정했다.
아래 내용은 구현 전에 작성한 근거 검토이며 성능 결과를 대신하지 않는다.

## 1. 먼저 정정해야 하는 최근 결과

이전 대화의 “두 후보 모두 body MAE 약 9.7% 악화”는 잘못된 비교다.
`run_hard_lmm_backbone_candidate_campaign.py`의 `audit_job`은 `gt_p99`만 제외해
candidate의 body를 **≤train p99**로 집계한다. 반면 B의 고정 비교 자료는
`le_p50`, `p50_p90`, `p90_p95` 세 구간, 즉 **≤train p95**이다.
Instacart에서는 각각 q≤35와 q≤25로 다른 모집단이다.

5090의 종료된 summary와 launch contract를 로컬로 가져와 재계산했다.
같은 validation target identity·quantity hash와 503,733개 전체 target을 확인했고,
올바른 body count는 B와 후보 모두 477,507개다.

| Instacart seed42 | B | Inter-layer 후보 | 변화 |
| --- | ---: | ---: | --- |
| Overall raw RMSE | 5.8722169305 | 5.8808638777 | 0.1473% 악화 |
| Overall MAE | 3.9937810783 | 3.9897478966 | 0.1010% 개선 |
| Body ≤p95 MAE | 3.4386015032 | 3.4341731538 | 0.1288% 개선 |
| >p99 MAE | 21.9177947653 | 21.8838502103 | 0.1549% 개선 |
| Legacy clamped time loss | 3.2155228955 | 3.2183614605 | +0.0028386 |

따라서 Inter-layer 후보는 올바른 body guardrail을 통과하며, strict raw-RMSE
기준 미달은 유지된다. “Body가 붕괴해서 실패”라는 해석은 철회한다.
작은 seed42 차이만으로 통계적 열등성이나 모든 구조 변경의 무효를 주장하지 않는다.

5080 Memory-FiLM 역시 같은 잘못된 집계 함수를 사용했다. 이전에 확인한 overall
RMSE 5.8784391564는 B보다 약 0.1060% 높지만, body 3.7719918865와 +9.70% 주장은
정의가 맞지 않아 사용할 수 없다. 이번 재감사에서는 5080 SSH 연결이 두 차례
timeout되어 정확한 ≤p95 수치 재계산을 완료하지 못했다. 서버 결과를 변경하거나
캠페인을 재시작하지 않았다. 두 후보의 raw-RMSE 실패로 기존 중단 결정은 유지된다.

원본 remote status는 당시 코드의 결과로 보존한다. 향후 캠페인 사용 전에 body
집계의 명시적 세 구간 선택, 기준선과의 body count 및 target identity 확인을
수정해야 한다. 이번 검토에서는 모델·runner 코드를 변경하지 않았다.

## 2. 지금까지 실제로 좁혀진 범위

- Inter-layer와 Memory-FiLM은 정적 prototype read를 encoder 중간에 넣는 위치와
  작용 방식을 바꿨다. 학습 가능한 모든 sequence encoder를 평가한 결과는 아니다.
- Weighted retrieval은 Taxi/RAF의 완료 결과만 있다. Intermittent는 사용자 중단,
  Instacart main run은 미실행이므로 네 데이터셋 전체 실패라고 묶지 않는다.
- Separate-key의 Taxi 양성 결과와 Instacart의 작은 효과는 과거 joint selector
  기준이다. 현재 B/raw-RMSE 비교의 구조 단독 효과로 그대로 옮기지 않는다.
- Elapsed-age의 음성 결과는 attention score에 넣은 8개 계수의 특정 설계에 한한다.
- Transition-error는 train-only 필요조건에서 실패해 Backbone 구현·e300을
  수행하지 않았다. Prototype 요약이 global past-error control보다 주는 추가
  log-MSE 개선은 세 데이터셋 모두 1% 미만이었다.
- Instacart raw64 probe는 H3..32, ridge와 고정 tanh128 변환 범위에서 h64보다
  나빴다. 이는 trainable convolution·recurrent encoder 전체의 상한이 아니다.
- Gradient 감사는 일부 Taxi batch에서 충돌을 보였지만 모든 데이터셋의 공통
  지배적 원인으로 확인하지 못했다. 현재 B 전체 학습에서의 충돌은 미검증이다.

## 3. 제안 우선순위 1: Q/K/V의 인과적 국소 convolution

**가설:** 최근 몇 사건의 연속된 수량·간격 패턴을 attention이 읽기 쉬운 형태로
만들면, 원본 Hard-LMM의 정적 bank를 유지하면서 이력 표현을 개선할 수 있다.
기존 encoder도 시간·수량·순서를 읽으므로 “새 정보 제공”이나 “확인된 정보 소실
복구”로 표현하지 않는다. 학습하기 쉬운 국소 패턴 구조를 추가하는 가설이다.

현재 경로는 log-gap/log-quantity 두 입력의 선형 projection, 두 Titan attention
block, 최종 static top-4 residual이다. 제안은 **첫 attention block의 event-only
Q/K/V**에 각각 하나의 짧은 depthwise causal convolution을 추가하는 것이다.

```text
Q' = Q + Cq(Q)
K' = K + Ck(K)
V' = V + Cv(V)

C*(x)i = c0*x_i + c1*x_(i-1) + c2*x_(i-2)   # channel별, bias 없음
```

- 초안 kernel은 3, convolution weight는 0에서 시작한다. 추가 scalar gate까지
  동시에 0으로 두지 않아 학습 경로가 막히지 않게 한다.
- hidden64에서 첫 block Q/K/V 세 개에만 적용하면 추가 weight는 3×3×64=576개다.
  출력 projection·추가 층을 넣으면 이 수는 달라지므로 후보 확정 때 고정한다.
- Persistent token은 convolution 범위에 넣지 않는다. 관측 event의 시작 이전은
  0으로 처리하고 padding을 convolution 입력·출력에서 mask한다.
- 짧은 이력에서는 존재하는 사건만 사용한다. H1에서도 현재 사건 항은 유효하다.
- 최종 Hard-LMM bank·top4·head·loss·selector는 B와 같게 유지하는 구조 제안이다.
- 사건 간 정보를 실제로 전달하는지 확인하고, 현재 사건만 처리하는 pointwise
  control과의 차이를 봐야 한다. 단순 파라미터 증가 효과도 구분해야 한다.
- 추가 연산량은 sequence length에 선형이지만 전체 attention의 이차 비용을
  없애지는 않는다. 실제 5080/5090 latency·peak memory 이득은 아직 측정하지 않았다.

이 설계는 기존 static memory의 중간 재주입과 다르고, 검사한 현 코드·계약·결과
자료에서 동등한 완료 실험을 찾지 못했다. Titans 원논문 §4.4는 Q/K/V projection
후 depthwise-separable 1D convolution을 명시한다. 따라서 근거 있는 부품이지만
그 자체가 새로운 논문 기여는 아니며, 우리 수량 예측 문제에서의 효과는 미검증이다.

**중요한 한계:** kernel3은 사건 번호상 이웃이다. 불규칙한 실제 경과시간을 직접
모델링하는 새 time kernel이라고 부를 수 없다. Instacart의 시간배치와 오류 사이
10.32% 연관성도 이 특정 convolution의 개선 가능성을 수치로 보장하지 않는다.

## 4. 대안 2: attention과 작은 Delta-rule memory의 결합

기존 정적 bank와 별개로, 관측 token의 learned key/value를 작은 행렬 상태에
기록하는 선형 associative memory를 결합하는 방향이다. 예시 update는

```text
Sbar_i = alpha_i * S_(i-1)
S_i = Sbar_i + beta_i * (v_i - Sbar_i*k_i) * k_i^T
r_i = S_i*q_i
```

이다. 정적 prototype bucket의 수량 잔차 평균과 달리 학습된 content를 기록한다.
과거 transition-error 진단 실패를 그대로 적용할 수 없지만, 양성 근거로 사용할
수도 없다. Inner autograd를 사건마다 호출하는 faithful Titans 구현과 다른
명시적 update이며 chunkwise 병렬화의 선행연구가 있다.

다만 Python 순차 loop로 구현하면 다시 느려질 수 있다. 기존 이력이 짧고 hidden이
작은 환경에서는 큰 언어모델의 throughput 결과가 적용되지 않는다. 구현 난도,
kernel·device 지원, 초기 출력 동일성, numerical 안정성, 비용을 확인해야 하므로
저비용 국소 convolution 다음 순위다. 이름도 faithful Titans 재현으로 붙이지 않는다.

## 5. 대안 3: 시간·수량에 서로 다른 encoder 상태

같은 관측 이력을 읽되 시간과 수량의 학습 파라미터 일부 또는 전부를 분리한다.
세 데이터셋에 동일한 구조를 사용하면 dataset별 모델 분기는 아니다. 출력 head만
분리하는 것과 encoder 내부의 파라미터를 분리하는 것도 구분해야 한다.

TPP의 task-gradient 간섭을 줄이는 문헌 근거는 있다. 그러나 해당 연구의 mark는
주로 사건 유형이며 현재 연속 수량 regression으로의 효과는 별도 검증 대상이다.
또한 shared trunk가 남아 있으면 두 출력 상태를 만들었다고 gradient 충돌이 전부
없어지는 것은 아니다. 완전 분리는 모델·연산 비용과 비교 통제의 부담이 커진다.

현재 로컬 증거로는 1순위가 아니다. 과거 Intermittent 감사는 강한 간섭 기준을
통과하지 못했고, Taxi/RAF weighted 진단의 bank-only conflict도 약했다. 현재 B에서
충돌과 실제 수량 오차가 연결되는 증거가 생길 때 재검토하는 편이 타당하다.

## 6. 결론과 아직 남은 판단

검토할 실제 Backbone 후보는 남아 있다. 세 데이터셋 공통 개선을 이미 입증한
미검증 후보가 있다는 뜻은 아니다. 현재 선택을 하나로 좁힌다면 **첫 block Q/K/V의
짧은 인과적 convolution**이 기존 시도와의 중복과 구현 비용이 가장 작은 제안이다.

다음 검토 대상은 이 단일 후보의 인과성·초기 동일성·gradient·대조군·비용 계약이다.
비교는 현재 B와 같은 loss 및 selector에서 수행해야 구조의 효과를 볼 수 있다.
목표인 세 데이터셋 RMSE 개선과 기존 수량/time guardrail은 결과를 보고 완화하지
않는다. 과거 원본 대비 body5% 기준은 별도 주장 기준이며 B 대비 채택 기준과 섞지
않는다. 후보 구현·GPU 학습·held-out 평가는 이번 검토에서 실행하지 않았다.

## Sources

- Current code: `models/TPPs/CountAwareTPP.py`, `models/Titan/backbone.py`,
  `models/Titan/common/memory.py`, `models/TPPs/CountAwareTitanInterLayerMemory.py`,
  `models/TPPs/CountAwareTitanMemoryFiLM.py`.
- B evidence: `paper/results/hard_lmm_raw_rmse_checkpoint_alignment_seed42_20260906/a_vs_b_metrics.json`.
- Latest copied 5090 evidence: `search_artifacts/hard_lmm_backbone_parallel_20260907/interlayer_5090/`.
- Prior evidence: `paper/results/hard_lmm_instacart_raw_history_20260905/README.md`,
  `paper/results/hard_lmm_instacart_balanced_20260905/README.md`,
  `paper/results/hard_lmm_transition_error_probe_20260906/README.md`,
  `paper/results/hard_lmm_weighted_static_20260903/mechanism_analysis.md`,
  `paper/results/titantpp_time_quantity_gradient_audit_20260820/result_analysis.md`.
- [Titans: Learning to Memorize at Test Time, §4.4](https://arxiv.org/html/2501.00663v1).
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v3).
- [Preventing Conflicting Gradients in Neural Marked Temporal Point Processes](https://arxiv.org/html/2412.08590v1).
