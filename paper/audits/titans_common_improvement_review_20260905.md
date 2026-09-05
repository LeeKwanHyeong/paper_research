# Titans 재검토와 TitanTPP 공통 Backbone 개선 연구

- 작성일: 2026-09-05 KST
- 조사 기준: `paper_research/master`, `567196e510e14211e0ee400ea312872bb3ec2703`
- 상태: 논문·코드·기존 증적 검토 완료. 기존 B1에 대한 합성 CPU 구조 감사와 train 이력 길이 감사 완료. 새 성능 후보는 제안 단계.
- 범위: 첨부 Titans v1, 공식 최종 출판본, 관련 일차 문헌, 저장소 구현과 기존 실험.
- 이번 조사에서 후보 학습, GPU 접속, validation 재평가, held-out test 접근은 하지 않았다.
- 이 문서는 연구 제안이다. [v0.7 최종 모델·주장 계약](../contracts/titantpp_v0_7_final_model_claim_contract_v1.md)을 개정하거나 T0를 교체하지 않는다.

## 1. 판단

**공통 개선을 포기할 근거는 없다. 다만 이미 실패한 gate를 반복하기보다, 실제로 관측한 정보를 memory가 언제 읽고 어떤 예측 목적에 사용하도록 학습하는지부터 바꿔 검증하는 편이 타당하다.**

이번 조사에서 가장 구체적인 근거는 동적 Titans-MAC 후보 B1의 읽기 시점이다. 현재 B1은 16개 사건 묶음의 시작 상태로 memory를 읽어 예측하고, 그 묶음의 관측을 memory에 쓴다. 일반 runner는 target window마다 memory를 초기화한다. 따라서 이력 16개 이하인 target의 예측에는 해당 window에서 수행한 online write가 영향을 주지 못한다.

실제 train target에서 이 조건의 비중은 Instacart **97.0827%**, Taxi **6.0949%**, Intermittent **20.3063%**였다. 합성 CPU 감사에서는 H=16의 update on/off 출력이 정확히 같고 writer gradient가 0이었으며, H=17에서는 출력 차이와 writer gradient가 생겼다.

이는 **B1의 짧은 이력에서 online adaptation을 활용할 기회가 제한돼 있다는 구조적 증거**다. 정적 모델 T0의 Instacart 오차 원인, 새로운 후보의 성능 개선, 입력에 남은 정보량을 입증한 결과는 아니다. 그 구분을 유지하면서도, 기존의 막연한 구조 확장보다 구체적인 첫 개입을 정할 수 있다.

첫 연구 후보는 **B1의 local attention과 write trajectory를 유지하면서 출력 read가 같은 묶음 안의 앞선 관측 write를 사용할 수 있게 하는 것**이다. 여기서 이득이 확인될 때 T0·RMTPP·THP 대비 공통 효과를 검증한다. 읽기 기회를 열어도 다음 사건 예측에 도움이 되지 않으면, 기억할 내용과 내부 학습 목적을 후속 가설로 검토한다.

## 2. 먼저 정정할 설명

### 2.1 Gate는 새롭게 남아 있는 단순 해법이 아니다

이미 다음 경로가 있다.

| 기존 경로 | 구현한 기능 | 현재 해석 |
| --- | --- | --- |
| GatedSoftMemory | separate K/V, dense retrieval, output projection, zero-init scale와 sigmoid gate | 구현과 Intermittent e300 검증 있음 |
| SurpriseGatedMemory | 관측별 read-before-write, low-rank fast memory, momentum, zero-init scale | 구현과 e300 검증 있음 |
| Dual hard+surprise | T0 상태에 zero-init surprise correction, shared/adapter-only gradient 경로 | persistent 조건을 맞춘 Intermittent 실험도 완료, 채택 실패 |
| B2 TPP-specific memory | 관측 FIFO, separate Q/K/V, top-4 weighted retrieval, null memory, confidence gate | seed42 3개 데이터셋 screening에서 공통 기준 실패 |
| Frozen smooth shrinkage | 표본별 memory 기여 축소, scalar 대조군, 실제 gradient 확인 | 4개 데이터셋에서 두 arm 모두 기준 실패 |

따라서 “identity-preserving gate가 가장 근거 있는 해법”이라는 앞선 제안은 과했다. 초기 출력 동일성은 구현 계약으로 유용하지만, 이미 시도한 방법이라는 사실과 성능 보장의 부재를 함께 설명해야 한다.

### 2.2 Instacart는 무신호 데이터가 아니다

기존 raw-history probe는 상수 예측보다 direct-MSE를 약 47.7–48.3% 줄였다. 다만 같은 용량의 probe에서 raw64가 h64보다 2.2–3.7% 나빴고, frozen model의 잔차도 충분히 줄이지 못했다. 이 결과는 **해당 probe로 encoder 정보 소실을 확인하지 못했다**는 뜻이다. Bayes 한계, raw 이력의 정보 부재, Backbone 개선 불가능성을 뜻하지 않는다. [Raw-history 결과](../results/hard_lmm_instacart_raw_history_20260905/README.md)

### 2.3 데이터별 병목이 달라도 공통 구조를 만들 수 있다

같은 계산 그래프·입력 정의·학습 규칙을 모든 데이터셋에 적용하면서, 관측에 따라 memory 활용이 달라지는 것은 공통 모델이다. 데이터셋별로 독립 학습한 파라미터 값이 다른 것도 현재 비교 방식과 일치한다. 이는 모든 데이터셋에 하나의 동일한 학습 가중치를 적용하는 별도의 전이학습 주장과 구분한다.

문제가 되는 것은 결과를 본 뒤 Taxi에는 A 경로, Instacart에는 B 경로를 수동 선택하고 이를 하나의 구조의 성과로 합치는 방식이다. 이번 제안에는 dataset ID별 분기나 데이터셋별 후보 선택을 넣지 않는다.

## 3. 첨부 Titans 논문에서 실제로 가져와야 할 내용

### 3.1 버전과 근거 범위

첨부 파일은 첫 페이지에 `arXiv:2501.00663v1 [cs.LG] 31 Dec 2024`가 표시된 27쪽 문서다. SHA-256은 `7ae54fb31930bda88043dcb9315f9befb7dbf13b3707a9106e14aa575663ca9e`다. 수식과 표를 텍스트로 확인하고, pp.6·9·15의 전체 페이지를 렌더링해 대조했다.

arXiv 이력에는 조사 시점에도 v1만 표시된다. 별도로 NeurIPS 2025 최종 출판본 38쪽이 있으며, 시계열 실험은 삭제되지 않고 Appendix G.5/Table 7로 이동했다. 첨부본에서는 §5.6/Table 3, pp.15–16이다. [arXiv 이력](https://arxiv.org/abs/2501.00663), [NeurIPS 최종본](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)

### 3.2 Titans는 관측 이력을 신경망의 가중치에 저장한다

첨부본 §3.1, pp.5–6의 핵심은 다음 inner-loop다.

\[
k_i=W_K x_i,\quad v_i=W_V x_i,\quad
\ell_i=\|\mathcal M_{i-1}(k_i)-v_i\|_2^2,
\]
\[
S_i=\eta_i S_{i-1}-\theta_i\nabla_{\mathcal M}\ell_i,
\qquad
\mathcal M_i=(1-\alpha_i)\mathcal M_{i-1}+S_i.
\]

벡터의 행/열 표기 방향은 구현에 따라 달라질 수 있다. 여기서 gradient는 memory 파라미터에 대한 gradient다.

| 구성 요소 | 역할 | 우리 과제에서 확인할 내용 |
| --- | --- | --- |
| Neural memory | 현재 관측의 key-value 관계를 online weight에 저장 | 해당 series의 관측이 실제 상태를 바꾸는가 |
| Surprise gradient | 현재 관계를 설명하지 못하는 방향으로 수정 | 큰 내부 오차가 다음 수량 예측에도 중요한가 |
| Momentum | 직전 update 방향의 영향을 유지 | 유용한 변화 지속인가, 큰 오차의 누적인가 |
| Forgetting | 기존 상태의 보존량을 입력에 따라 조절 | 짧은 이력의 유용한 prior까지 지우지 않는가 |
| Query/read | 현재 예측에 맞는 memory 출력을 검색 | 갱신된 내용을 예측 전에 실제로 읽는가 |
| Persistent memory | 전체 학습에서 얻은 입력 비의존 task parameter | 현재 series의 online memory와 효과를 구분하는가 |
| Local attention | 최근 관측 사이의 직접 관계를 표현 | memory 실험에서 이 경로가 함께 축소되지 않는가 |

K/V/Q와 제어 파라미터는 outer 예측 objective를 통해 학습한다. 따라서 내부의 reconstruction loss만 잘 내려가는 것보다 **외부 예측 loss가 writer까지 전달되고, 적절한 내용을 저장하도록 학습되는지**가 중요하다.

### 3.3 MAC·MAG·MAL은 서로 다른 결합 설계다

첨부본 §4, pp.9–11은 memory를 attention의 context로 넣는 MAC, attention과 memory를 비선형 결합하는 MAG, 두 모듈을 순차 배치하는 MAL을 구분한다. 단순한 scalar gate 하나가 MAG 전체를 정의하지 않는다.

특히 MAC의 Eq.21–25는 시작 memory에서 읽고 attention을 계산한 후 memory를 갱신하며, 최종 출력에는 갱신한 memory의 read가 들어간다. 현재 저장소 B1은 마지막 read도 segment-start state를 사용하도록 바꿨다. 계약에도 이 차이가 명시되어 있다.

원본 식을 TPP에 그대로 옮길 때는 사건별 인과성이 필요하다. 관측 i까지 쓴 뒤 i+1을 예측하는 것은 허용되지만, 한 segment 전체의 최종 상태를 앞쪽 모든 사건의 예측에 사용하면 미래 정보가 들어간다. 구현은 각 사건의 허용된 prefix 상태를 읽어야 한다.

### 3.4 원논문이 보여준 성과와 우리 문제 사이의 간격

첨부본 시계열 실험은 SiMBA framework의 Mamba 모듈을 neural memory로 교체한 장기 예측이다. ETT·ECL·Traffic·Weather를 사용한다. 이는 불규칙 사건의 다음 시간·수량을 공동 예측하는 TPP 실험이 아니다.

Table 3에서 Neural Memory는 표의 일곱 데이터셋 모두 MSE가 가장 낮다. 다만 Traffic MAE는 Neural Memory 0.289, iTransformer 0.282여서 모든 지표에서 최상위는 아니다. 긴 context에서의 표현력과 검색 실험은 설계 동기이며, 짧은 basket 이력의 개선을 보장하지 않는다.

첨부본 Table 5의 LMM perplexity는 27.01이며 linear memory 28.49, momentum 제거 28.98, forgetting 제거 29.04였다. 이 ablation은 memory의 내용·갱신·보존을 함께 검토할 근거다. count-TPP에 같은 효과가 있다는 근거로 확대하지 않는다.

또한 저자 저장소에서 실행 가능한 공식 구현을 확인하지 못했다. 조사 시점 [ABehrouz/Titans](https://github.com/ABehrouz/Titans)는 비어 있었으므로, 저장소의 B1은 논문 기반 자체 구현으로 다루고 공식 코드와의 동일성을 주장하지 않는다.

## 4. 현재 코드와 기존 실험을 함께 보면

### 4.1 원본 T0는 정적 Hard Local Memory Matcher다

\[
I_i=\operatorname{Top4}_j\cos(h_i,m_j),\qquad
z_i=h_i+\frac14\sum_{j\in I_i}m_j.
\]

64개 prototype 중 4개를 선택해 단순 평균한다. prototype은 학습 가능한 전역 파라미터지만, 추론 중 해당 series의 관측으로 갱신되지 않는다. 유사도 점수는 최종 가중 평균에 쓰지 않고 top-k 인덱스는 미분 불가능하므로 검색 residual에서 query 방향으로 미분 가능한 addressing 신호가 없다. 직접 `z=h+r` 경로와 attention은 학습된다. [Hard-LMM 구현](../../models/Titan/common/memory.py)

Separate-key는 선택된 점수의 softmax 가중치를 사용하므로 addressing 학습 경로를 추가한다. 그러나 이 변경은 Taxi에서만 seed42 양의 결과를 보였고, Instacart 공통 개선으로 이어지지 않았다. 이미 검증한 이 separate-key 후보가 공통 채택 기준을 충족하지 못한 것으로 기록하며, 다른 key/value 설계 전체의 가능성을 배제하지 않는다.

### 4.2 현재 3-seed 성능의 정확한 출발점

아래는 fixed validation, seeds 42·52·62의 평균이다. [최종 T3 표](../tables/T3_v0_7_backbone_validation.md)

| 데이터셋 | RMTPP MAE / RMSE | THP MAE / RMSE | T0 MAE / RMSE |
| --- | ---: | ---: | ---: |
| Intermittent-5000 | 2.9025 / 10.5787 | 0.6664 / 2.1507 | 0.7469 / 1.9195 |
| Taxi | 40.2893 / 144.8148 | 41.5884 / 147.4963 | 42.5805 / 143.9931 |
| Instacart | 4.0270 / 5.9901 | 4.0368 / 6.0012 | 4.0450 / 6.0159 |

Intermittent는 RMTPP 대비 강한 양의 근거가 있지만 THP와 MAE/RMSE trade-off가 있다. Taxi는 T0의 RMSE 평균 이점이 작고 변동이 크다. Instacart는 RMTPP 대비 MAE 0/3, RMSE 1/3 seed만 더 낮다.

같은 hidden dimension과 head/loss를 썼다는 것이 parameter-matched를 뜻하지는 않는다. 실제 파라미터 수는 RMTPP 25,283, THP 100,291, T0 89,795(Intermittent/Taxi) 또는 77,507(Instacart)이다. THP와 T0는 memory 외에도 normalization, FFN 폭, position, persistent token이 다르다. 현재 비교는 encoder 전체 구성의 비교이며 memory 하나의 인과 효과를 완전히 분리하지 않는다. [Seed 원표](../results/titantpp_v0_7_validation_freeze_20260905/validation_seed_metrics.csv), [구조 차이 감사](../results/hard_lmm_weighted_static_20260903/mechanism_analysis.md)

### 4.3 기존 실험을 반복하지 않기 위한 증거 지도

아래 변화율에서 음수는 오차 감소다. 완료 범위와 제한을 함께 기록한다.

| 시도 | 확인된 결과 | 다음 설계에 주는 제약 |
| --- | --- | --- |
| Frozen memory 제거 | Instacart MAE 4.045→4.755, RMSE 6.016→7.097 | memory를 끄면 좋은 baseline으로 돌아간다는 설명 불가. 재학습 no-memory 실험과는 다름 |
| Frozen calibration/shrinkage | 4 datasets × 2 arms, 0/8 채택 | 단순 보정의 공통 효과 없음 |
| Smooth adaptive shrinkage | 두 arm 모두 0/4, 40,822 minibatch에서 gradient finite/nonzero | gate dead-zone만 고치면 해결된다는 설명 불가 |
| Frozen readout factorial | joint selector 0/24; body selector 3/24, 모두 Intermittent | head 확장만으로 공통 개선 근거 없음. 추가 seed 복제는 미완료 |
| Similarity weighted static | 완료된 Taxi·RAF 0/2; Intermittent 중단, Instacart 미실행 | 4개 데이터셋 전체 실패로 세지 않음 |
| Separate-key | Taxi body −26.77%, RMSE −34.23%, tail −40.29%; Instacart body 약 −0.05%, RMSE +0.38% | 1/2 seed42 screening. key 분리 하나의 인과효과나 공통 채택 증거는 아님 |
| 평균·최근편차 query 진단 | 두 데이터셋의 separate-key 잔차 설명 개선 없음 | query 후보가 구현되어 학습 실패한 것으로 기록하지 않음 |
| Value/bank 변화량 | Instacart 수량 방향 SD가 평균 결합까지 크게 축소 | 유용한 예측 신호 소실이라고 단정 불가. Taxi도 축소되며 개선 |
| Elapsed-age | original 대비 일부 개선이나 separate-key 추가 효과 실패, 0/2 | 시간 간격과 오류의 관찰 연관성이 이 intervention 성공을 보장하지 않음 |
| Instacart raw-history | 상수보다 좋지만 h보다 나쁘고 잔차 설명 실패 | 무신호도 아니고 encoder 병목 입증도 아님 |

출처: [B0 제거](../results/count_aware_b0_retrieval_diagnostic_20260827/result_briefing_ko.md), [frozen probe](../results/count_aware_hard_lmm_frozen_probe_20260903/README.md), [smooth shrinkage](../results/hard_lmm_smooth_shrinkage_20260903/README.md), [readout](../results/hard_lmm_readout_factorial_20260903/README.md), [복제 범위](../results/hard_lmm_readout_seed_replication_20260903/README.md), [weighted](../results/hard_lmm_weighted_static_20260903/README.md), [separate-key](../results/hard_lmm_key_value_screening_5090_20260904/README.md), [query](../results/hard_lmm_query_diagnostic_20260905/README.md), [bank](../results/hard_lmm_bank_diagnostic_20260905/README.md), [elapsed-age](../results/hard_lmm_elapsed_age_screening_5090_20260905/README.md).

### 4.4 동적 memory도 이미 시도했으며 부분 성과와 실패가 함께 있다

Intermittent의 초기 4-arm seed42 실험은 Surprise memory의 MAE를 23.24%, RMSE를 11.35% 개선했다. 그러나 clamped time score가 악화됐고 persistent token 조건이 달랐다. 이후 persistent token 16개와 scaled exact RMTPP head를 맞춘 M0/M1/M2/M3 실험도 완료됐다. Surprise와 두 dual 후보는 채택 기준에 실패했고, exact time head는 finite하더라도 큰 train loss spike를 보였다. 따라서 “동적 memory는 전부 무효”도, “persistent나 time head만 맞추면 해결”도 현재 증거를 넘어선다. [초기 4-arm](../results/titantpp_memory_backbone_inter_seed42_e300_20260818/result.md), [matched persistent/dual](../results/titantpp_scaled_time_persistent_dual_20260819/result_analysis.md)

B2의 fresh seed42 screening도 Intermittent·Taxi·RAF 0/3이었다. Intermittent는 body −17.932%와 RMSE −5.698%에도 tail +4.419%, time +0.032996으로 실패했다. Taxi는 body +13.975%, RMSE +12.604%였다. Instacart까지 이 결과가 검증된 것으로 확대하지 않는다. [B2 원시 decision](../../search_artifacts/count_aware_b012_seed42_screening_e300_20260828_recovery1/comparison/decision.json)

B1 MAC 초기 seed42의 Taxi에서는 양의 결과가 있었지만, Instacart inner update 폭주 후 안정화 정책이 바뀌었다. stable inner-gradient clipping 정책에서 감사된 장기 실행은 seeds 52·62이며, 동등한 seed42 e300이 감사되지 않아 과거 seed42와 합쳐 3-seed 표를 만들 수 없다. 두 stable seed의 Instacart 평균은 MAE 4.0653, RMSE 6.0669다. B1이 이미 공통 성능을 입증했다고 할 근거는 없다. [stable 감사](../results/count_aware_titantpp_mac_stable_20260902/validation_audit.md)

초기 instability에서는 14번의 write 동안 memory weight의 절댓값 규모가 약 3.07e23까지 커지고 다음 associative gradient가 non-finite가 됐다. Outer gradient clipping은 forward 내부의 online update를 보호하지 못한다. 또한 B1은 기존 비용 감사에서 T0 대비 step 비용이 약 6.8–11.2배였다. 읽기 시점 교정은 이 비용 문제를 자동으로 해결하지 않는다. [안정성 사고](../results/titantpp_mac_inner_stability_incident_20260831/README.md), [비용 감사](../results/count_aware_titantpp_mac_optimization_20260830/analysis.md)

## 5. 이번에 직접 확인한 B1 읽기 시점의 제한

### 5.1 계산 규칙

H는 target과 padding을 제외한 실제 입력 이력 길이다. segment 크기 B=16, window마다 state reset, 마지막 관측 상태로 다음 사건을 예측하는 현재 규칙에서는 예측에 보이는 write 수가 다음과 같다.

\[
N_{\mathrm{visible}}(H)=16\left\lfloor\frac{H-1}{16}\right\rfloor.
\]

| 이력 길이 H | 현재 B1이 마지막 예측에서 볼 수 있는 window 내 write 수 |
| ---: | ---: |
| 1–16 | 0 |
| 17–32 | 16 |
| 33–48 | 32 |

H≤16에서도 모델 전체가 학습되지 않는 것은 아니다. Local attention, query, 초기 memory parameter와 head는 학습된다. 차단되는 것은 **그 target loss에서 해당 window의 online writer로 이어지는 경로**다. 더 긴 target은 writer 학습에 기여할 수 있다.

근거 구현: [B1 segment read/write](../../models/Titan/common/titans_mac.py), [마지막 history state 선택](../scripts/count_aware_tpp_backbone/core.py), [설계 계약의 의도적 순서 차이](../contracts/count_aware_titans_backbone_reproduction_v1.md).

### 5.2 합성 CPU 감사

기존 코드를 hidden64, 2층, 4heads, persistent16, segment16, stable inner clip1로 실행했다. Random initialization, dropout-off/eval, CPU float32이며 실제 데이터와 checkpoint는 사용하지 않았다. 합성 encoder 출력의 선형 scalar를 미분해 구조적 gradient 경로를 확인했다. 기존 quantity head의 zero-weight 초기화와 혼동하지 않도록 head를 통과하지 않은 검사다.

| 감사 항목 | H=16 | H=17 |
| --- | ---: | ---: |
| 실제 수행한 관측 write | 16 | 17 |
| 마지막 예측에 보이는 이전 write | 0 | 16 |
| Write on/off 출력 max absolute difference | 0, bitwise equal | 0.1239743 |
| Key/value/update/momentum/forget writer gradient | 검사한 8개 parameter 모두 0 | 모두 finite/nonzero |

H=16에서도 최종 memory 자체는 변한다. 다만 해당 예측이 그 변화를 읽지 않는다. 이 결과는 성능 수치가 아니며, 짧은 이력에서 writer를 사용하게 하면 성능이 좋아진다는 증거도 아니다. [재현 스크립트](../results/titans_common_improvement_review_20260905/mac_write_visibility_probe.py), [결과와 source hash](../results/titans_common_improvement_review_20260905/mac_write_visibility_probe.json)

### 5.3 실제 train 모집단에서의 범위

Train partition의 series와 시간 이력만 읽고, loader의 lookback 및 left-truncation 규칙으로 effective H를 계산했다. 초기 history 없는 target은 loader와 같은 규칙으로 제외했다. 따라서 아래 분모는 원래 train 사건 수가 아닌 **실제로 history를 가진 학습 target 수**다. 수량 오차나 validation/test는 계산하지 않았다.

| 데이터셋 | 유효 train target | H≤16 target | 현재 B1에서 window 내 write가 보이지 않는 비율 |
| --- | ---: | ---: | ---: |
| Intermittent-5000 | 393,824 | 79,971 | 20.3063% |
| Taxi | 38,393 | 2,340 | 6.0949% |
| Instacart | 1,991,192 | 1,933,102 | 97.0827% |

캐시의 실제 H와 Intermittent·Instacart 각각 65,536 target, Taxi 전체 38,393 target에서 정확히 일치했다. Series length 중앙값으로 추정한 수치가 아니다. [Train 계산 스크립트](../results/titans_common_improvement_review_20260905/train_write_visibility_audit.py), [분모·hash·parity 기록](../results/titans_common_improvement_review_20260905/train_write_visibility_audit.json)

기존 B1 frozen audit에서도 RAF의 6,690 validation event 모두 online write 제거 효과가 0인 사례가 있었다. 이번 계산은 Instacart의 **train 대부분에서도 같은 구조 제한이 적용됨을 수량화**했다. 기존 관찰을 처음 발견한 것처럼 주장하지 않는다. [기존 MAC audit](../results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/interpretation.md)

## 6. 개선안 우선순위

| 우선순위 | 가설/변경 | 실제 근거 | 아직 모르는 것 |
| --- | --- | --- | --- |
| 1 | B1 출력 read에 같은 segment의 앞선 관측 write를 연결 | 코드·합성 gradient·train H 비중으로 제한 확인 | B1 및 T0 대비 예측 개선 여부 |
| 2, 조건부 | 이전 상태→다음에 관측된 사건의 관계를 memory에 저장 | reconstruction과 forecast 목적은 다름; outer writer 학습 경로 중요 | 현재 입력에서 h 이상의 예측적 잔차가 실제로 있는가 |
| 3, 조건부 | 일부 사건의 과도한 write·forget을 제어 | 실제 B1 inner 폭주; Titans/MIRAS의 갱신·보존 설계 | 큰 update가 오류와 연결되는지, 유용한 tail까지 억제하지 않는지 |
| 병렬 분석 | 같은 local encoder 위에서 memory 구성요소의 효과를 분리 | T0/THP의 attention·FFN·persistent 조건 차이 | 강한 local 경로 위에서도 memory가 추가 효과를 주는가 |
| 별도 연구축 | quantity/time objective 및 선택 규칙의 정합성 | raw/log 지표 불일치, clamped time score, 일부 gradient 규모 불균형 | 공통 objective 수정이 실제 성능을 개선하는가 |

“모두 한 번에 넣은 adaptive backbone”을 첫 후보로 두지 않는다. 첫 후보의 성공/실패가 다음 가설 선택을 결정하도록 한다.

### 6.1 첫 후보: 같은 segment 안의 앞선 write를 출력에서 읽는 B1

목적은 **local attention의 길이와 입력을 바꾸지 않고 memory 사용 시점만 변경하는 것**이다.

현재 B1에서 segment 시작 state를 \(M_s\), 그 segment의 causal attention 출력을 \(y_i\)라고 쓰면 기존 최종 read는 \(r_i^{start}=M_s(Q(y_i))\)다. 제안은 \(y_i\)보다 앞선 관측들의 write를 반영한 \(M_{i-1}\)를 읽는다.

\[
r_i^{prior}=M_{i-1}(Q(y_i)),\qquad
o_i=\operatorname{Fuse}_{B1}(y_i,r_i^{prior}).
\]

현재 사건 i의 관측은 local attention \(y_i\)에 이미 들어 있다. Memory는 앞선 관측들의 관계를 제공한다. H=1은 기존 초기 memory를 사용하고, H≥2부터 앞선 관측의 write가 출력에 영향을 줄 수 있다.

**유지할 것**

- B1의 pre-attention segment-start read, causal attention, segment16, persistent tokens, Q/K/V, 2-layer memory, stable inner clip, 기존 fusion.
- 관측 이력 범위, head, loss, optimizer, seed, checkpoint selector.
- Write 입력은 기존의 attention output \(y_i\)다. 새 fused output을 write에 되먹임하지 않는다.
- 같은 가중치와 입력에서 memory update 순서와 segment 끝 state가 기존 B1과 같아야 한다. 출력이 사용하는 중간 state만 바꾼다.

**새 gate·파라미터는 필요하지 않다.** 단순히 `segment_size=1`로 줄이면 local attention 범위도 바뀌므로 이 가설과 다른 실험이다. Cross-window state를 갑자기 이어 더 긴 이력을 제공하는 것도 첫 비교에 포함하지 않는다.

원본 Eq.25처럼 현재 사건까지 쓴 \(M_i\)를 읽는 방법도 causal하다. 그러나 방금 쓴 값을 다시 읽는 self-reconstruction 경로와 앞선 이력 활용을 분리하기 위해 첫 후보는 \(M_{i-1}\) read로 좁히는 것을 권고한다. 따라서 원본 MAC의 완전 재현이라고 부르지 않는다. 새로운 attention 알고리즘의 독창성보다 TPP에서의 memory read 시점에 대한 검증 가능한 개입이다.

기존 Surprise/dual도 관측별 memory를 사용했으므로 관측별 memory 자체는 새 아이디어가 아니다. 이번 실험의 차이는 **기존 deep B1의 다른 조건을 그대로 두고 마지막 read 시점만 바꾼다는 것**이다.

### 6.2 초기 출력 동일성의 정확한 처리

첫 후보는 사용 정보가 달라지므로 활성 상태의 초기 출력이 기존 B1/T0와 자동으로 같지 않다. 순수한 timing 비교에서는 같은 초기 파라미터와 결정적 대조 경로를 보장하고, `segment-start` 정책으로 되돌렸을 때 기존 B1 출력이 정확히 같음을 검사하는 것이 명료하다.

초기 출력 동일성을 이번 새 모델 계약에도 필수로 두려면 별도의 차이 보정식이 필요하다.

\[
r_i=r_i^{start}+\gamma(r_i^{prior}-r_i^{start}),\qquad \gamma_0=0.
\]

이는 직접 read 교체와 다른 후보다. 부호 있는 \(\gamma\)는 일반적인 [0,1] gate가 아니며, 초기 writer 학습 지연과 gate 학습을 함께 평가해야 한다. B1 초기 출력 보존은 T0 초기 출력 보존도 아니다. 첫 연구에서는 이 후보들을 동시에 탐색하지 않고, 구현 계약에서 한쪽만 고정해야 한다. 본 보고서의 우선 권고는 추가 파라미터 없는 직접 교체다.

기존 gate 설명에도 다음 제약이 적용된다.

- 유한한 sigmoid bias로 정확한 0을 만들 수 없다.
- Zero output projection과 zero residual scale을 동시에 두면 학습을 막을 수 있다.
- 하나의 zero projection을 사용하면 그 projection은 먼저 배울 수 있지만 내부 memory는 첫 step에서 task gradient가 없을 수 있다.
- 기존 공통 quantity head도 weight=0으로 시작한다. 첫 backward와 head가 열린 뒤의 gradient를 구분해야 한다.
- 공동 학습에서는 baseline 경로도 바뀐다. 초기 identity는 학습 후 비열등 보장이 아니다.

### 6.3 두 번째 후보: 관측된 전이를 저장하는 memory

읽기 기회를 열었지만 저장된 정보가 다음 예측에 도움이 되지 않는다면, 같은 사건의 latent key→value 복원 대신 **이전 상태→새로 관측된 사건**의 관계를 저장하는 가설을 검토할 수 있다.

\[
k_j=K(y_{j-1}),\qquad v_j=V(x_j),\qquad
\ell_j^{transition}=\|M(k_j)-v_j\|^2.
\]

현재 i까지 관측한 시점에는 j≤i의 전이만 학습하고, \(Q(y_i)\)로 i+1을 위한 정보를 읽는다. 관측되지 않은 다음 target은 inner loss에도 들어가지 않는다. 동일한 시간 간격·수량 입력만 사용하고, dataset ID나 추가 외부 변수는 넣지 않는다.

이 후보는 기억할 관계를 바꾸므로 첫 read-timing 후보와 동시에 도입하면 원인 분리가 어렵다. 또한 Instacart raw-history probe의 부정적 결과 때문에 “예측 신호가 더 남아 있다”는 전제를 확정할 수 없다. Train 내부의 분리된 series에서 전이 특징이 기존 상태의 오차를 설명하는지 먼저 확인해야 한다. 기존 평균·최근편차 query probe를 단순 재명명해 반복하지 않는다.

### 6.4 세 번째 후보: 실제로 해로운 update만 제한하는 memory

Titans의 surprise가 큰 관측은 중요할 수도 있고 noise/outlier일 수도 있다. 따라서 큰 수량이라는 이유만으로 write를 억제하지 않는다. 먼저 다음을 같은 train 사건에서 연결해야 한다.

1. Raw quantity와 내부 residual/update norm의 관계.
2. 그 write가 이후 상태 및 예측을 얼마나 바꾸는지.
3. 해당 변화가 이후 관측의 오차를 줄이는지 늘리는지.
4. 기존 inner clip이 언제 얼마나 자주 활성화되는지.

과도한 update가 실제 오류와 연결되면 Huber 계열 inner objective를 단일 후보로 검토할 수 있다. 이는 최종 quantity prediction loss를 바꾸는 것과 다르다. [MIRAS](https://arxiv.org/html/2504.13173v1)는 memory objective와 retention을 설계 축으로 구분하고, Yaad에서 Huber 계열을 검토한다. 우리 count 데이터에서의 효과는 자체 검증이 필요하다.

학습된 초기 memory까지 forgetting으로 빠르게 훼손하는 근거가 확인되면, 초기 prior와 online delta를 나누어 delta만 감쇠시키는 별도 가설도 가능하다. 현재 그런 오류 원인이 입증된 것은 아니므로 첫 후보에 넣지 않는다. 갱신량 clipping·Huber·prior 보존을 동시에 넣으면 각각의 효과를 판정할 수 없다.

### 6.5 단순한 선형 memory는 비용 대조군으로 가치가 있다

[Gated DeltaNet](https://arxiv.org/html/2412.06464v3)은 전체 memory decay와 특정 key의 오차 수정 write를 결합하며, local attention 혼합을 검토한다. 동일한 read/write 타이밍 아래에서 작은 선형 state를 deep B1의 용량·비용 대조군으로 쓸 수 있다. 단 기존 low-rank Surprise의 반복과 차이를 명시해야 하며, 선형 변형을 Titans 완전 재현으로 표현해서는 안 된다. [공식 구현](https://github.com/NVlabs/GatedDeltaNet)

[TTT](https://arxiv.org/html/2407.04620v4)는 inner update를 거쳐 outer 예측 목적이 학습되는 경로를 이해하는 데 직접적인 참고가 된다. 그러나 공식 PyTorch 저장소도 교육용·비최적화 구현임을 설명한다. 논문 처리량을 5090에서 그대로 기대하지 않고 실제 환경에서 비용을 측정해야 한다. [공식 PyTorch](https://github.com/test-time-training/ttt-lm-pytorch)

## 7. Backbone만으로 해결하려 할 때 놓치면 안 되는 공통 학습 문제

### 7.1 Log-MSE와 raw MAE/RMSE는 다른 목표다

조건부 log-MSE의 최적값은 \(E[\log(1+Q)\mid H]\)다. 이를 `expm1`으로 복원한 값은 일반적으로 raw MSE의 최적인 \(E[Q\mid H]\)도, raw MAE의 최적인 조건부 median도 아니다. 따라서 log-MSE를 더 잘 학습한 encoder가 raw 지표에서 항상 좋아지는 것은 아니다.

실제 weighted Taxi에서 공식 joint-selected e2의 MAE/RMSE는 50.795/177.752였고, 사후에 보면 e36은 29.913/93.551이었다. 후자를 다시 선택하면 기존 계약 위반이다. 이 사례는 raw 성능과 선택 objective의 불일치를 검토할 근거이며, 숨겨진 좋은 epoch가 최종 모델이라는 뜻이 아니다. [학습 궤적 분석](../results/hard_lmm_weighted_static_20260903/mechanism_analysis.md)

분포 head도 새롭게 남아 있는 간단한 답은 아니다. Log-normal K=1 quantity head는 Intermittent seed42에서 RMSE +33.245%, tail MAE +75.098%, time score +3.944로 실패했다. 적절한 분포/loss 후보는 shared gradient와 point-estimator 정의까지 별도 계약으로 검증해야 한다. [기존 K=1 결과](../results/count_aware_lognormal_k1_screening_20260816/validation_briefing.md)

### 7.2 Time score와 gradient 규모를 구조 효과와 혼동하지 않는다

v0.7의 `legacy_clamped_rmtpp`는 duration cap이 활성화되면 정규화된 density를 보장하지 않는다. 기존 필드명 `time_nll`을 provenance로 유지하되 clamped time loss라고 부른다. 새로운 후보만 정상 time head로 바꾸고 기존 T0 결과와 비교하면 Backbone 효과를 분리할 수 없다.

Scaled exact head도 이미 시도했으며, 큰 train loss spike가 남았다. 별도 H0/H3 train-only 감사에서는 H3의 clipping을 time-head gradient 규모가 지배했다. 이는 해당 실험의 근거이며, 현재 세 데이터셋 모두의 지배 원인이라고 확대하지 않는다. 방향 충돌과 규모 불균형도 구분해야 한다. [H0/H3 감사](../results/titantpp_h0_h3_gradient_attribution_20260820/result_analysis.md)

첫 memory timing 실험에서는 head/loss/selector를 고정해 개입 효과를 분리한다. 실제 사건 시간의 확률적 개선까지 주장하려면 모든 비교 Backbone에 공통으로 적용하는 별도의 정상화된 time-head 계약과 안정성 검증이 필요하다. 이 변경을 memory 후보에 조용히 섞지 않는다.

과거 T0 source의 intercept cap300과 현재 master 기본값30 차이도 해소해야 한다. 과거 checkpoint를 새 기본값으로 평가해 비교하면 안 된다. 새 실행 계약에 source와 head compatibility를 명시하고 필요한 대조군만 matched 조건으로 재실행한다.

## 8. 공통 개선을 판정하는 방법

### 8.1 세 가지 질문을 구분한다

| 질문 | 필요한 비교 | 결과가 허용하는 주장 |
| --- | --- | --- |
| 수정한 memory가 실제로 작동하는가 | 동일 source의 B1 start-read vs prior-prefix-read, write/no-write 진단 | 읽기 지연과 학습 경로에 대한 mechanism 결과 |
| 현재 모델보다 공통으로 좋아지는가 | 같은 계약의 T0 vs 후보, 모든 지정 데이터셋 | 지정 데이터셋에서의 공통 개선 |
| 비교 TPP보다 일관되게 좋은가 | RMTPP·THP·후보, 공통 입력/head/학습/선택 | 검증한 데이터셋·지표·모델에 한정한 우위 |

B1의 손실을 회복했지만 T0보다 나쁘면 공통 개선에 성공한 것이 아니다. T0보다 좋아도 RMTPP/THP를 이겼다고 할 수 없다. RMSE 우위로 MAE나 시간 예측까지 우위라고 표현하지 않는다.

### 8.2 수치 기준에 대한 권고

기존 채택 기준은 원본 대비 **body MAE 5% 이상 개선, RMSE·tail MAE 악화 각각 2% 이하, time score 증가 0.01 이하**다. 이 기준을 결과를 본 뒤 “2% 안에서 비슷하면 개선”으로 낮추지 않는다. Body/tail membership은 기존 train-only 정의를 사용한다.

공통 개선의 최종 dataset 집합은 Intermittent-5000·Taxi·Instacart를 모두 포함하도록 권고한다. 첫 mechanism 실험은 B1 대비 효과도 별도로 요구해야 한다. 원본 기준을 세 데이터셋 각각 적용하고 결과를 합산해 한 데이터셋의 실패를 감추지 않는다.

다만 기존 기준은 RMSE 악화를 2%까지 허용하므로, 통과 자체가 “모든 데이터셋에서 RMSE 개선”을 뜻하지는 않는다. 그런 논문 주장을 목표로 한다면 **세 데이터셋 모두의 raw RMSE가 실제로 감소해야 한다는 조건**을 후보 결과 전에 추가해야 한다. 외부 TPP 우위는 RMTPP와 THP 각각에 대해 같은 지표로 확인한다. 이 보고서는 새 수치 계약을 이미 승인·동결된 것으로 취급하지 않는다.

Paired seed 결과, 평균±표준편차, 데이터셋별 표를 함께 보고한다. Series 단위 오차 resampling은 동일 checkpoint에 조건부인 불확실성이고, seed 변동까지 대체하지 않는다. 3 seeds는 screening/재현성의 출발점이며 통계적 비열등이나 보편적 우위의 보증이 아니다. 다중 비교와 독립 평가 방법은 장기 실행 전에 고정한다.

### 8.3 반복 탐색에 따른 선택 편향을 줄인다

현재 validation은 이미 여러 의사결정에 사용됐다. 첫 메커니즘 설계는 train 내부의 series 분리 구간에서 수행하고, 후보 수와 예산을 미리 고정한다. 새로 분리한 train 구간도 기존 모델이 학습한 데이터라면 완전히 독립된 최종 평가로 부르지 않는다.

최종 후보를 고른 뒤 기존 fixed validation으로 matched 비교를 완료한다. Held-out test는 후보·selector·보고 기준 동결 후 승인된 한 번의 평가에만 사용한다. 결과를 보고 test를 다시 개발 데이터로 전환하지 않는다.

## 9. 다음 구현에서 필요한 계약과 중단 조건

| 계약 | 완료 조건 |
| --- | --- |
| Causality | 미래 target·뒤쪽 event를 바꿔도 앞쪽 출력 불변. Segment 경계를 유지한 prefix 실행 일치 |
| Read/write timing | H=1/2, H=16/17에서 허용된 write 수와 출력 가시성 일치 |
| Baseline parity | 기존 segment-start 정책의 출력·state가 원본 B1과 일치 |
| State trajectory | 동일 weights/input에서 segment 끝 memory가 기존 B1과 일치 |
| Gradient | 합성 encoder 경로 및 head가 열린 실제 loss에서 writer gradient finite/nonzero 확인 |
| Mask/reset | Padding과 target write 0, series 간 state 격리, batch 순서 변경 불변 |
| Evaluation update | `eval/no_grad`에서도 관측 이력에 대한 inner update는 의도대로 실행 |
| Restore | model·optimizer 복원과 명시적 streaming state 복원의 범위 구분 |
| Stability | 기존 stable clip 적용, pre/post norm 기록, overflow/NaN 없음 |
| Fair context | 기존과 같은 관측 범위. Overlapping window의 중복 write를 state carry로 누적하지 않음 |
| Runtime | 실제 길이 분포에서 메모리와 처리량 측정. 과거 B1의 큰 비용을 숨기지 않음 |

Gradient 검사는 파라미터별 최초 기여 시점에 맞춰야 한다. 예를 들어 momentum state가 0에서 시작하면 첫 write만 읽는 H=2에서는 momentum 계수에 gradient가 없을 수 있다. H≥3에서 두 번 이상의 prior write를 거친 경로와, quantity head가 학습되기 전후를 구분해 검사한다. 모든 길이·모든 파라미터에 즉시 nonzero를 요구하지 않는다.

기계적으로 nonzero update가 생긴 것만으로 e300을 확장하지 않는다. 초기 screening에서 어느 데이터셋이 사전 guardrail을 크게 위반하면 seed 확장 대신 원인을 정리한다. B1 개선이 T0 개선으로 이어지지 않으면 “공통 Backbone 개선”으로 채택하지 않는다. 첫 후보가 반증되면 완료된 실패로 기록하고, 위의 후속 조건이 충족될 때만 다음 가설을 연다.

## 10. 작업 순서

**완료 / 로컬 — 논문·구현·기존 결과와 새 구조 증거를 통합했다**

- 첨부 Titans와 공식 최종본의 범위, T0/B1/B2 차이, 이전 gate·dual·time-head 실험을 확인했다.
- 합성 CPU write/gradient 감사와 train effective-history 비중을 계산했다.
- 성능 개선 결과가 새로 나온 것은 아니며, v0.7/T0와 held-out 잠금은 현재 기준선으로 유지한다.

**다음 작업 / 로컬 — B1 출력 read 시점 후보 하나의 계약을 확정한다**

- 본 보고서의 prior-prefix 직접 read 교체를 우선 후보로 삼고, initialization·head compatibility·검증 데이터·예산·판정 기준을 확정한다.
- Body 기준과 raw RMSE 공통 개선 목표를 구분해 수치 계약에 남긴다.
- 기존 local encoder 및 write trajectory를 유지하는 계약과 단위 검증을 먼저 완성한다.

**후속 작업 / 로컬 — 별도 opt-in 경로로 구현하고 계약을 검증한다**

- 기존 T0/B1/B2 경로를 유지하면서 후보 식별자를 분리한다.
- Causal prefix read, gradient, padding/reset, state parity, 저장·복원을 검사한다.
- `paper_research`의 작업 브랜치를 명시하고 독립 커밋한다. 현재 미추적 root `scripts/`는 포함하지 않는다. `master` 반영 범위는 새 구현 계약에서 명시한다.

**실행 전 범위 확정 / 5090 — CUDA·e1에서 정상 실행과 비용을 확인한다**

- 과거 GPU 승인과 구별해 이번 새 후보의 source·대조군·예산을 구체화한다. GPU 사용 의사는 이미 밝혀졌으므로 동일 범위 승인을 반복 요청하지 않고 새 실행 범위만 명시한다.
- GPU 점유·runtime·데이터 checksum·전송 source hash 확인 후 CUDA parity, e1, state restore와 메모리 비용을 검증한다.
- e1은 실행 계약 검증이다. 성능 채택으로 해석하지 않는다.

**후속 성능 검증 / 5090 — 고정된 한 후보를 지정 데이터셋 전체에서 평가한다**

- 계약에 맞는 B1/T0 대조군을 사용한다. 기존 artifact가 source·head·seed 조건을 충족하면 재사용하고, 부족한 행만 보완한다.
- Seed42 screening에서 공통 조건을 확인한 뒤 사전 규칙에 따라 seeds52·62로 확장한다.
- 별도의 데이터셋 승자들을 조합하지 않으며 e300은 최대 예산으로 사용한다.

**후속 작업 / 로컬 — 결과에 맞게 기여와 원고를 갱신한다**

- 단순 read 시점 교정만으로 일반적 algorithm novelty를 주장하지 않는다.
- 성공하면 같은 구조에서 관측 이력을 활용한 memory의 추가 효과, 실제 성능, 계산 비용을 함께 보고한다.
- 공통 개선이 확인된 경우에만 v0.7 이후 모델·주장 계약 개정과 held-out 평가 준비로 진행한다.
- 코드/단위 검증은 같은 memory state 계약에 의존하므로 직렬이다. 문헌 보완·기존 artifact 감사는 독립적으로 병렬 진행할 수 있다. CUDA/e1→장기 성능→최종 평가는 직렬이다.

## 11. 증거를 읽는 기준

이 보고서의 코드 사실과 train 비중은 재현 가능한 관찰이다. 논문 수식은 설계 근거다. “prior-prefix read가 성능을 개선할 수 있다”, “전이 저장이 도움이 될 수 있다”, “robust update가 tail 손상을 줄일 수 있다”는 아직 검증하지 않은 가설이다.

이번 재검토가 바꾼 것은 **공통 개선 가능성의 결론이 아니라 다음 실험을 선택할 근거의 구체성**이다. 이제 확인할 첫 질문은 “더 복잡한 memory를 붙이면 좋아질까”보다 좁다. 현재 short-window B1이 학습하지 못하던 관측 write 경로를 열었을 때, 지정한 세 데이터셋의 실제 예측 오차가 함께 줄어드는지를 검증한다.
