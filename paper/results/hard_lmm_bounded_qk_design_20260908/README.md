# 다음 Backbone 후보 선정 결과

**선정: Q/K의 추가 이력 혼합 크기를 head별로 제한하고 기존 causal-V 경로를 유지한다.**
2026-09-08, 로컬 검토 완료. 기존 QKV 진단 `f0cc1fd`를 기준으로
계약·summary·history·저장된 train NPZ·checkpoint kernel을 검토했다.
새 데이터 행, 모델 forward, 학습, CUDA, held-out 평가는 실행하지 않았다.

이는 checkpoint selector나 시간 head 교체를 Backbone 개선으로 부르는 제안이 아니다.
첫 encoder attention 내부 Q/K의 연산을 바꾸는 하나의 연구 후보다. 성능은 아직 미확인이다.

## 1. 기존 정상화 시간 평가에서 재사용할 부분

`paper/contracts/aligned_frozen_lognormal_duration_v1.json`은 공통 K=1 log-normal
head와 관측별 likelihood를 이미 고정했다. source는
`8213dbcdace2b62425236af15aa28f5b57f7b958`, 계약 SHA는
`677cc3af91d84dfea8e3b4e71b38697fb467059f680678f0b8370a50f2366f16`다.

Intermittent continuous density, Taxi positive-integer mass,
Instacart positive-integer mass + code30 survival을 그대로 재사용한다.
이 관측 방식 차이는 데이터 형식에 대한 기존 계약이며 dataset별 Backbone 분기가 아니다.
train-only log-duration initializer, train median scale, 130개 파라미터, AdamW1e-3/wd0,
max100/epoch0 포함/min20/patience20, proper NLL earliest strict finite minimum도 유지한다.

| 기존 aligned-B validation | proper NLL | 수량 MAE | raw RMSE |
| --- | ---: | ---: | ---: |
| Taxi | 0.6956474254 | 28.674020 | 88.194997 |
| Intermittent | 0.7873644338 | 0.604340 | 1.499555 |
| Instacart | 2.8129205141 | 3.993781 | 5.872217 |

이는 `aligned_causal_duration_scale_adapter_v1.json`의 checkpoint/cache digest와
`aligned_causal_duration_scale_adapter_seed42_20260907/validation_audit.json`의
기존 재현 감사가 뒷받침한다. 이번에 B head를 다시 실행한 수치는 아니다.
현재 로컬 aligned artifact 디렉터리는 비어 있다. 원격 원본을 찾아 checkpoint/cache
SHA를 다시 확인해야 실행 시 재사용할 수 있다. 정확한 원격 경로는 추측하지 않았다.

**기존 FULL과 새 후보의 정상화 시간 수치는 없다.** 두 모델은 각각 자체 hidden
cache와 K=1 head fit이 필요하다. B cache는 target identity 비교의 기준으로는
사용할 수 있지만 새 Backbone hidden의 대체물이 아니다.
기존 runner는 A/B와 `titantpp` route를 고정하므로 내부 공통 함수를 이용하는
새 source route가 필요하다. 따라서 기존 명령을 그대로 실행하면 되는 상태는 아니다.

K=1 head fit 중에는 각 모델의 수량 경로를 동결하여 수량 prediction SHA가 정확히
유지되는지 확인한다. 그때의 NLL 차이는 **같은 시간 head 용량·학습 조건에서 읽을 수 있는
표현의 차이**를 보여준다. Backbone 자체의 legacy 시간 head 품질과는 구분한다.
NLL 개선을 median MAE/RMSE 개선이라고 바꿔 말하지 않는다.

## 2. Taxi: train 시간 점수 개선이 validation으로 이어지지 않았다

| 기존 결과 | 선택 epoch | train legacy time | validation legacy time | validation raw RMSE |
| --- | ---: | ---: | ---: | ---: |
| B | 45 | 1.32965 | 1.47339 | 88.1950 |
| FULL | 113 | 1.22164 | 25.40722 | 85.8369 |

FULL의 validation legacy time 최솟값은 e16의 1.36424였지만 당시 raw RMSE는
144.6075였다. 현재 quantity selector를 time selector로 바꾸면 수량 이득을 유지한
해결책이 되지 않는다. FULL의 선택 수량 RMSE는 B보다 2.67%, MAE는 5.76% 좋다.

기존 동일 train 4,096개 표본에서는 B/FULL legacy loss가 1.28286/1.02619였다.
FULL의 `wd=10` 포화는 34건(0.83%), intercept 상한 포화는 0건이었다.
FULL intercept 범위는 −12.78~0.022, exp(intercept) 최댓값은 1.023으로,
이 표본에서 양의 intercept 폭증이나 overflow는 없었다.
가장 긴 1%와 time loss가 큰 1%가 겹치지 않았으며, 긴 간격 포화만으로 현재
train 오차를 설명할 수 없었다.

**validation의 per-target intercept·integral·duration 결합 trace가 없으므로
25.407의 원인을 표본 수준에서 확정할 수 없다.** Train gradient 8 batch에서
Q/K 충돌 방향이 두드러졌지만 이를 학습 전체의 충돌이나 task 분리의 증거로 삼지 않았다.
Train 두 fold 역시 이미 학습에 쓰인 series를 나눈 진단이지 out-of-sample 검증이 아니다.

## 3. Intermittent: 큰 수량의 제곱오차가 body 개선을 상쇄한다

기존 validation 86,285건의 수량 구간 summary만 재집계했다. 경계46/187은 train에서
고정한 p95/p99다. 각 구간의 전체 MSE 기여는 `n_bin/n_total × (FULL MSE−B MSE)`다.

| 수량 구간 | 건수 | B MAE → FULL | B RMSE → FULL | 전체 MSE 변화 기여 |
| --- | ---: | ---: | ---: | ---: |
| ≤46 | 81,739 | 0.465790 → 0.432630 | 1.088215 → 1.066501 | −0.044321 |
| (46,187] | 3,405 | 2.407737 → 2.740364 | 3.667471 → 4.000563 | +0.100793 |
| >187 | 1,141 | 5.148108 → 5.228543 | 6.713847 → 7.215383 | +0.092380 |
| 전체 | 86,285 | 0.604340 → 0.587117 | 1.499555 → 1.548392 | +0.148852 |

상위 두 구간이 순 MSE 증가의 67.7%, 62.1%를 만들고 body가 29.8%를 상쇄한다.
>p99 signed bias는 −0.07843→+1.52478이어서 단순한 과소예측 강화도 아니다.
이력 구간에서는 H65–128만 RMSE가 악화했지만 수량×이력 교차표가 없으므로
상위 수량과 해당 길이를 독립 원인으로 세거나 길이 자체를 원인으로 지목하지 않는다.

FULL e7→e47에서 train quantity loss는 16.19% 낮아졌지만 validation RMSE는
1.548392→3.004042로 94.01% 높아졌다. 47개 epoch 중 B RMSE를 이긴 epoch는 없다.
반면 저장된 train 4,096개에서는 >p99 RMSE가 B 5.414143→FULL 4.621610으로
좋아졌고 body는 1.077754→1.120687로 나빠졌다. 두 train fold에서도 같은 방향이었다.
따라서 큰 수량을 전혀 표현하지 못한다는 결론보다 **학습된 효과의 일반화가 남은 문제**다.

Intermittent의 legacy 시간 점수 차이 +1.881585는 slope에 따른 floor 차이
+1.891351과 나머지 −0.009766으로 분해된다. 기존 clamp 식은 전 구간에서 포화되며
정상화된 density도 아니다. 이를 시간 표현 악화 또는 시간 학습이 수량 정보를
빼앗았다는 근거로 사용하지 않는다. 과거 실패 판정은 원문 그대로 보존한다.

## 4. V를 보존하고 용량을 추가하지 않는 후보를 고른 이유

같은 FULL checkpoint에서 추가 V를 끈 QK는 train RMSE를 Taxi 35.39%,
Intermittent 115.79% 악화시켰다. Instacart도 pooled 0.44% 악화했고 fold 방향이
엇갈렸다. 이는 함께 학습된 V의 현재 기여를 보여주며 V 없는 모델을 처음부터
학습하면 반드시 실패한다는 뜻은 아니다.

기존 Instacart raw-history probe에서는 raw 이력에 예측 신호가 있었고 같은 용량의
hidden probe가 더 좋았다. 현재 B/FULL에 대한 새 probe는 아니므로 encoder의
잠재력을 전부 배제하는 근거로 쓰지 않는다. 더 큰 encoder, 데이터셋 전용 gate,
특정 시간 구간 분기를 선택할 근거 역시 부족하다.
해당 과거 증적은 [raw-history 감사](../hard_lmm_instacart_raw_history_20260905/README.md),
[비교 집단 감사](../hard_lmm_instacart_balanced_20260905/README.md),
[elapsed-age screening](../hard_lmm_elapsed_age_screening_5090_20260905/README.md)다.

| 검토한 방식 | 판단 |
| --- | --- |
| **Q/K residual 크기 제한 + 기존 V 유지** | 선택. FULL과 파라미터 수가 같고 zero-init 및 V 연산을 보존하며 변경 지점을 제한한다. |
| 이력에 따른 학습형 Q/K gate | 보류. gate 입력·용량·collapse 대조군을 추가로 정해야 하고 현재 근거로 그 자유도를 정당화하기 어렵다. |
| 시간·수량 encoder 또는 attention 분리 | 보류. 비용과 구조 변화가 크며 8 batch gradient 부호만으로 공통 병목을 입증하지 못했다. |

checkpoint kernel의 constant-input gain도 검토했다. V의 `1+sum(kernel)` 평균은
Taxi 약0.114, Intermittent0.250, Instacart0.083이었다. 이는 projection 상수 성분의
gain이며 raw 수량 level 정보의 손실률이 아니다. skip·position·다른 layer가 남으므로
이를 근거로 V를 강제로 DC-preserving 형태로 바꾸지 않는다.

[Titans §4.4](https://arxiv.org/html/2501.00663v1)는 Q/K/V projection 뒤의 convolution과
Q/K 정규화를 함께 설명한다. 현재 후보는 그 전체 구조의 재현이 아니라,
기존 Hard-LMM의 zero-init을 유지하는 별도 residual 제한식이다.
[NormFormer](https://arxiv.org/abs/2110.09456)는 내부 정규화가 학습에 영향을 줄 수 있다는
관련 근거지만, 언어 모델의 결과를 이 데이터셋의 개선 증거로 이전할 수는 없다.
어느 논문도 이번 제한식의 성능이나 새로운 기여를 입증하지 않는다.

## 5. 확정한 수식과 실제 검증 범위

Q/K 각각에서 head별 원 projection RMS를 `s=sqrt(mean(p²)+1e-8)`로 두고,
기존 causal residual `r`에 `g=s/sqrt(s²+mean(r²))`를 곱해 더한다.
V는 기존 `v+C_v(v)` 그대로다. 제한 강도는1, kernel3, hidden64/heads4,
FULL과 같은 576개 추가 파라미터이며 dataset별 튜닝은 없다.

**V의 연산 경로를 유지하는 것과 학습 후 MAE/RMSE를 보장하는 것은 다르다.**
Q/K 변화는 V의 attention 가중합과 downstream memory 검색도 바꾼다.
또한 원 projection이 계속 학습되므로 이 상대적 제한은 전체 모델의 norm 상한이 아니다.
현재 artifact에 residual/projection RMS나 validation attention trace가 없어서
Q/K 크기를 실패 원인으로 확정하지 않았다.

로컬 synthetic FP32/FP64에서 zero residual 출력, projection gradient,
residual gradient의 exact identity, head별 RMS bound, 여러 크기의 finite forward/gradient를
확인했다. 전체 모델의 초기 동일성·인과성·복원·CUDA·비용은 아직 검증 전이다.
정확한 설계와 후속 판정 기준은
`paper/contracts/hard_lmm_bounded_qk_causal_v_v1.md`에 고정했다.

## 재현과 남은 작업

재현: `/usr/local/bin/python paper/scripts/audit_hard_lmm_bounded_qk_design.py`.
`evidence_audit.json`에는 기존 artifact 20개의 SHA, 세 데이터셋 구간 재집계,
trajectory, kernel 통계, aligned-B 기준선, 독립 수식 확인을 기록했다.
raw 데이터 파일을 읽거나 TPP 모델을 import하지 않는다.
전체·구간 MSE 변화 합이 오차1e-8 이내에서 일치하는지도 확인했다.
감사를 재실행했을 때 결과 JSON의 SHA가 동일했다. 이 수식 검증은 새 Backbone의
전체 단위·계약 테스트를 대신하지 않는다.

**완료 / 로컬 — 문제·후보·시간 평가 재사용 범위를 확정했다.**
- Q/K 크기 제한과 기존 V 유지라는 하나의 구조를 선택하고 증거와 미확인 원인을 구분했다.
- 과거 실패 결과와 legacy 시간 점수의 해석 한계를 보존했다.

**다음 작업 / 로컬 — 후보 구현과 공통 시간 평가 연결을 검증한다.**
- `paper_research / codex/hard-lmm-causal-qkv`에서 독립 모델 route와 계약 테스트를 준비한다.
- B aligned checkpoint/cache 실파일을 회수·검증하고 FULL/후보 자체 cache를 위한
  평가 route를 만든다. B/FULL quantity 학습을 불필요하게 반복하지 않는다.
- 전체 모델 테스트와 비용 기준까지 확인한 변경을 독립 source commit으로 만든다.

**후속 작업 / GPU — 고정 source로 e1을 확인한 뒤 seed42를 판정한다.**
- 서버와 실행용 manifest가 확정된 뒤 CUDA·비용·e1 → FULL 정상화 기준선 및 후보
  seed42 screening 순서로 진행한다. 후보 screening에서 실패하면 추가 seed를 시작하지 않는다.
- 이번에는 새 GPU 학습·held-out 평가·`paper_research/develop` 또는
  `paper_research/master` 병합을 수행하지 않았다.
