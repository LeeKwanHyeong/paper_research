# Level-preserving history-confidence residual Hard-LMM — 설계 계약 v1

상태: **train-only 필요조건 통과, 로컬 구현·계약 검증 완료, CUDA 검증 전**.

2026-09-08, `paper_research / codex/hard-lmm-causal-qkv`.

식별자: `hard_lmm_level_preserving_history_confidence_residual_v1`.

후보 약칭: `LPHC-QKV`.

## 근거와 해결할 문제

Frozen B와 BOUNDED-QK의 Instacart train prediction `1,991,192`건을 새로운
series-disjoint 두 fold로 나눈 결과, 사전에 고정한 11개 필요조건이 두 fold에서
모두 통과했다.

- BOUNDED-QK는 B보다 centered MSE와 MAE를 개선했다.
- 동시에 예측 level을 fold별 `-0.14734`, `-0.14792` 낮췄다.
- bias 제곱 증가는 centered MSE 감소보다 약 네 배 커서 raw MSE를 악화시켰다.
- 이력 `2–3`에서는 shift가 약 `-0.52`, raw MSE가 약 `1.88%` 악화됐다.
- 이력 `8–15`에서는 shift가 약 `+0.12`이고 raw MSE가 개선됐다.

두 checkpoint는 모두 전체 train으로 학습되고 validation으로 선택됐다. 이 결과는
새 구조를 구현할 필요조건이며 독립적인 일반화 증거가 아니다.

따라서 단일 가설은 다음과 같다.

> 사건 projection의 절대 level이 아니라 최근 두 전이만 추가 경로가 읽게 하고,
> 관측된 전이 수가 적을수록 그 경로를 결정론적으로 줄이면, BOUNDED-QK의
> centered-error 이득을 보존하면서 짧은 이력의 하향 level shift를 줄일 수 있는가?

## 고정 구조

변경 범위는 기존 BOUNDED-QK와 같이 첫 encoder block의 **event Q/K/V projection**뿐이다.
projection 하나를 `p`, 유효 사건 mask를 `m`, 현재 위치까지의 유효 사건 수를
`n_t = sum_{j<=t} m_j`, 실제 인접 전이 수를
`c_t = sum_{j<=t} 1[m_j m_{j-1}]`라 한다. 두 causal difference를 다음처럼 정의한다.

\[
\Delta^{(1)}_t = 1[m_t m_{t-1}](p_t-p_{t-1}),\qquad
\Delta^{(2)}_t = 1[m_t m_{t-1}m_{t-2}](p_{t-1}-p_{t-2}).
\]

Q/K/V별 학습 파라미터 `a_p,b_p in R^d`를 사용해

\[
u_t=a_p\odot\Delta^{(1)}_t+b_p\odot\Delta^{(2)}_t,
\qquad
\gamma_t=\frac{c_t}{\max(n_t,1)}.
\]

를 계산한다. Q와 K는 절대 projection level을 bound 기준으로도 사용하지 않는다.
각 attention head에서 사용 가능한 difference의 RMS로 전이 scale을 정의한다.

\[
s_t^2=
\frac{I_1\operatorname{mean}(\Delta_t^{(1)2})+
I_2\operatorname{mean}(\Delta_t^{(2)2})}
{\max(I_1+I_2,1)}+10^{-8},\qquad
R_t^2=\operatorname{mean}(u_t^2),
\]

여기서 `I1`, `I2`는 해당 difference가 유효한지를 뜻한다. 기존 Q/K bound와 같은
포화형식을 전이 scale에 적용한다.

\[
B_\Delta(u_t)=u_t\frac{s_t}{\sqrt{s_t^2+R_t^2}}.
\]

\[
q'_t=q_t+\gamma_tB_\Delta(u^q_t),\qquad
k'_t=k_t+\gamma_tB_\Delta(u^k_t).
\]

V는 기존 후보에서 유용성이 확인된 unbounded 경로를 유지한다.

\[
v'_t=v_t+\gamma_tu^v_t.
\]

- `gamma`는 학습하지 않으며 모든 데이터셋에서 같다. 정상적인 연속 이력에서는
  `(H-1)/H`이므로 H1은 `0`, H2는 `0.5`, H3는 `2/3`, H8–15는
  `0.875–0.933`이다. Mask에 공백이 있으면 존재하지 않는 전이를 confidence로
  세지 않는다.
- `a,b`는 모두 정확히 0으로 초기화한다. hidden64에서 projection별 `2×64`,
  Q/K/V 합계 384개 파라미터다.
- 별도 bias, dataset embedding, duration 값 분기, target 수량 구간, 학습 가능한
  confidence, 수동 데이터셋별 계수를 사용하지 않는다.
- 전용 route 이름은 `titantpp_hard_memory_level_history_qkv`로 고정한다.
- persistent K/V, 두 번째 encoder block, static 64-vector Hard-LMM, top-4,
  arithmetic-mean retrieval, quantity/time head, loss와 selector는 바꾸지 않는다.
- 기존 파라미터는 현재 BOUNDED-QK 실험과 같이 계속 학습한다.

## `level-preserving`의 정확한 의미

이 residual과 Q/K/V **최종 가산량**은 유효 prefix의 모든 projection에 같은 상수
벡터를 더해도 변하지 않는다. Difference와 Q/K의 bound scale을 모두 difference에서
계산하므로 causal branch의 DC gain은 정확히 0이고, H1에서는 추가 경로가 정확히
닫힌다. 원 projection RMS로 Q/K를 제한하면 상수 이동에 따라 gain이 바뀌므로 그
방식은 사용하지 않는다. Zero padding 경계는 각 difference의 모든 원소가 유효할
때만 계산하므로 가짜 첫 전이를 만들지 않는다.

이는 **projection 수준의 level 보존**이다. Attention, 두 번째 block, Hard-LMM과
비선형 경로를 지난 최종 scalar quantity prediction의 평균 shift가 0이라는 정리는
아니다. Feature 차원 평균 제거는 quantity-head 방향과 무관하고, 최종 head 방향을
직교 투영하면 수량 변화를 제거하므로 사용하지 않는다. Scalar bias와 raw RMSE는
아래 성능 gate로 직접 확인한다.

## 구현 계약

로컬 구현은 다음을 모두 통과해야 한다.

1. **초기 identity**: 같은 B state, seed와 RNG에서 zero-init 후보의 time/quantity
   출력, Hard-LMM state, 공통 파라미터 gradient가 B와 bitwise하게 일치한다.
2. **DC-null**: H1과 모든 constant valid projection에서 Q/K/V residual이 정확히 0이다.
   유효 prefix에 임의의 상수 벡터를 더해도 residual이 일치한다.
3. **고정 confidence**: H1/H2/H3/H8/H15의 `gamma`가 각각 계약식과 일치하고,
   Q/K 추가 residual RMS가 `gamma × headwise transition RMS` 경계를 넘지 않는다.
4. **학습 가능성**: nonconstant H2/H3 표본에서 Q/K/V의 `a,b`가 finite nonzero
   gradient를 받는다. 존재하지 않는 lag의 계수는 gradient를 받지 않는다.
5. **인과성과 누출 방지**: 미래 사건·target 수량·padding 값·다른 batch 표본을
   바꿔도 현재 prefix 출력이 바뀌지 않는다. 왼쪽 padding과 불연속 mask에서
   존재하지 않는 lag와 인접 전이를 사용하지 않는다.
6. **경로 보존**: persistent memory와 최종 Hard-LMM의 key/value, top-4와 결합 방식은
   기존 B와 같고, residual이 열린 뒤에는 Q/K/V·attention·hidden·검색·quantity가
   실제로 달라질 수 있다.
7. **수치와 복원**: FP32와 CUDA에서 finite loss/gradient, 전용 checkpoint 식별,
   strict save/restore, optimizer와 selector resume의 다음 step 일치를 확인한다.
8. **비용**: 추가 계산은 O(BLd), 전체 attention은 여전히 O(L²)다. 기존 공통 기준인
   B 대비 median step ratio `<=1.5`, peak allocated memory ratio `<=1.25`를 유지한다.

## 학습·선택 계약

- 세 데이터셋에 하나의 동일한 구조와 `gamma`를 사용한다.
- B state에서 상속 파라미터를 초기화하고 새 6d 파라미터만 0으로 추가한다.
- 기존 train/validation split, log1p quantity MSE, legacy time objective, AdamW
  `lr=0.001`, batch128, clip1을 유지한다.
- checkpoint는 validation raw quantity RMSE의 가장 이른 strict minimum으로 선택한다.
- 최대300 epoch, 최소40 epoch, patience40을 유지하며 결과를 본 뒤 구조·gate·loss·
  selector를 바꾸지 않는다.
- e1은 CUDA·처리 건수·gradient·저장·복원 계약만 판정한다.
- seed42 screening은 Instacart부터 실행한다. 실패하면 Taxi·Intermittent와 추가 seed를
  시작하지 않는다.

## 사전 고정 성능 gate

Instacart seed42에서 다음 네 조건을 모두 요구한다.

- B 대비 raw RMSE가 strict 감소한다.
- B 대비 전체 MAE와 centered MSE가 strict 감소한다.
- 이력 `2–3` raw MSE가 B 이하이며, 이력 `8–15` raw MSE가 B보다 낮다.
- Candidate와 B의 mean prediction 차이 절댓값이 B RMSE의 `1%` 이하이고,
  candidate의 bias 절댓값이 B보다 크지 않다.

그 뒤 세 데이터셋 공통으로 기존 BOUNDED-QK screening 기준을 재사용한다.

- B 대비 raw RMSE가 모두 strict 감소한다.
- B 대비 전체 MAE 악화 `<=1%`, body 및 `>p99` MAE 악화 각각 `<=2%`다.
- FULL 대비 raw RMSE·전체 MAE 악화 `<=1%`, body 및 `>p99` MAE 악화 각각 `<=2%`다.
- 같은 frozen K=1 평가의 proper Time NLL은 aligned-B `+0.01` 이내다.
- 세 데이터셋 seed42가 모두 통과할 때만 seeds52·62를 실행한다.
- held-out test는 구조와 추가 seed 판정이 끝나기 전에는 사용하지 않는다.

Instacart의 centered MSE·history gate는 구조가 의도한 문제를 실제로 해결했는지 보는
기전 확인이다. 최종 채택 조건은 세 데이터셋의 공통 raw RMSE와 기존 quantity/time
guardrail이다. BOUNDED-QK와의 비교는 centered 신호 보존을 설명하는 탐색적 증적이며,
새 후보 선택의 독립 확인으로 표현하지 않는다.

## 다음 실행 순서

1. 완료: 전용 모델 route와 위 계약 테스트를 로컬에서 구현한다.
2. 다음: 독립 source commit을 만든 뒤 5090에서 CUDA·비용·세 데이터셋 full-data e1을 확인한다.
3. Instacart seed42를 먼저 screening하고, 통과할 때만 Taxi·Intermittent를 실행한다.
4. 세 데이터셋 모두 통과할 때만 seeds52·62와 최종 held-out 평가 범위를 확정한다.

근거 결과는
`paper/results/hard_lmm_bounded_qk_train_condition_20260908/README.md`에 기록한다.
