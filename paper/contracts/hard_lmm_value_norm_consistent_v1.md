# Prototype value-norm consistency Hard-LMM — 설계·평가 계약 v1

상태: **후보 경계·구현·로컬 계약 검증 완료, GPU 실행 전**.

작성일: 2026-09-09

대상: `paper_research / codex/hard-lmm-causal-qkv`

식별자: `hard_lmm_value_norm_consistent_v1`

후보 약칭: `VNC-Hard-LMM`

## 중복 감사와 후보 선택

집계된 Hard-LMM residual에 스칼라 gate를 곱하는 후보는 기존 frozen
shrinkage/readout과 같은 경로다. 선형 수량 head에서 새 correction은
`(g-1) w^T r_B`이므로 새로운 memory 결합 기전으로 볼 수 없다. Smooth
shrinkage와 scalar control은 4개 데이터셋에서 모두 실패했고, readout
factorial의 joint 선택도 24개 fit 중 통과가 없었다.

Online write/update confidence는 readout과 다르지만 B1 Titans-MAC, B2
read-before-write, Surprise/dual-memory 실험과 중복된다. 계산 비용이 크고
짧은 Instacart 이력에서는 쓰기 표본 자체가 부족하므로 첫 후속 후보에서
제외한다. Aggregate residual 또는 hidden-state normalization도 residual 전체
scale을 바꾸는 기존 shrinkage와 기전 분리가 약해 보류한다.

과거 similarity-weighted static retrieval은 같은 top-4 결합 위치에서 raw
prototype을 cosine score로 가중했다. 이번 후보도 넓게 보면 static weighted
retrieval 계열이므로 완전히 새로운 경계라고 주장하지 않는다. 다만 weight가
similarity가 아니라 inverse row norm에서 나오고, zero gate가 B를 정확히 포함하며,
기존과 정확히 같은 식·학습 실험은 없었다. **같은 결합 경계의 구분되는 단일
가설**로만 평가한다.

선택한 경계는 **top-4로 선택된 각 prototype value의 norm을 arithmetic mean
전에 정렬하는 것**이다. 기존 검색은 cosine similarity라 prototype norm을
사용하지 않지만, 선택 뒤 raw value 평균은 norm을 암묵적 기여도로 다시
사용한다. 이 불일치가 유용한지 하나의 최소 후보로 검증한다.

## 고정 수식

현재 token state를 `h`, 64개 공유 bank를 `m_i`, 기존 cosine top-4 index를
`I(h)`라 한다. `M=64`, `k=4`, `epsilon=1e-8`을 고정한다.

\[
s=\frac{1}{M}\sum_{i=1}^{M}\|m_i\|_2,
\qquad
\widetilde m_j=\begin{cases}
s m_j / \|m_j\|_2,&\|m_j\|_2>\epsilon\\
0,&\text{otherwise}.
\end{cases}
\]

\[
r_B=\frac1k\sum_{j\in I(h)}m_j,
\qquad
r_N=\frac1k\sum_{j\in I(h)}\widetilde m_j,
\]

\[
\alpha=\tanh(\alpha_{raw}),
\qquad
r=r_B+\alpha(r_N-r_B),
\qquad
z=h+r.
\]

- `alpha_raw`는 scalar 하나이며 정확히 0으로 초기화한다. 따라서 초기
  `alpha=0`, `r=r_B`다.
- Signed `tanh`는 exact zero와 첫 step의 nonzero gradient를 함께 만족시키기
  위한 것이다. 이 후보를 one-sided convex interpolation이라고 부르지 않는다.
- norm이 epsilon 이하인 bank row는 normalized branch에서 output과 gradient를
  모두 정확히 0으로 둔다. 안전한 분모를 먼저 만든 뒤 명시적 mask를 적용해
  비활성 branch의 `0/0`도 막는다.
- FP16/BF16 norm은 FP32에서 계산한다. FP32와 FP64는 각각 FP32와 FP64에서
  계산하고, normalized residual은 기존 residual dtype으로 되돌린다.
- `s`는 계산 직후 detach하여 선택되지 않은 row에 dense gradient가 전달되지
  않게 한다. 선택된 prototype norm은 detach하지 않는다. `alpha=0`에서는 해당
  contrast의 공통 bank gradient 기여가 0이어서 B gradient를 보존해야 한다.

## 변경하지 않는 구조와 학습

- 같은 2-layer hidden64 causal encoder, 4 heads, FFN128, persistent token 16개.
- 같은 tied static bank `1×64×64`, cosine top-4 index와 similarity, 기존 raw
  arithmetic-mean residual.
- 같은 shared time/quantity state `z`, legacy clamped time head, direct log1p
  quantity MSE, quantity loss weight 1, tail loss 0.
- 같은 train/validation split과 mask, AdamW `lr=0.001`, batch128, global clip1.
- checkpoint와 early stopping은 validation raw quantity RMSE의 가장 이른
  strict finite minimum. 최대300, 최소40, patience40, seed42.
- 데이터셋 이름, 특정 duration/quantity 값, 수량 구간, 데이터셋별 계수나
  loss 분기는 사용하지 않는다.
- online write, per-series runtime state, 두 번째 검색, 별도 key bank, 새로운
  output calibration은 추가하지 않는다.

Fresh B와 후보는 seed42의 같은 RNG 상태에서 각각 생성한다. 후보의 공통
초기 tensor와 생성 뒤 RNG 상태가 B와 같고, 새 scalar만 추가되어야 한다.
성능 학습은 완료된 B checkpoint에서 이어서 최적화하지 않는다. 기존 B 결과는
source/data/selector digest가 일치할 때 비교 기준으로 재사용한다.

## 로컬 구현 계약

GPU나 실제 성능 평가 전에 다음을 모두 통과해야 한다.

1. **전용 route**: backbone은 `titantpp_hard_memory_value_norm`, model role은
   `hard_lmm_value_norm_candidate`다. 전용 metadata와 `lmm.alpha_raw`가 없는
   checkpoint를 후보 checkpoint로 재해석하지 않는다. 다른 후보의 state도
   거부한다. B state 주입은 shape·dtype·finite를 모두 검사하고 실패 시 어느
   tensor도 바뀌지 않아야 한다.
2. **B exact identity**: 같은 seed에서 공통 state tensor, Python/NumPy/Torch
   RNG, eval 출력, train loss, 공통 parameter gradient가 B와 정확히 같다.
   strict B state load 뒤에도 동일해야 한다.
3. **실제 학습 경로**: 비공선 prototype 사례에서 `r_N-r_B`가 0이 아니고
   `alpha_raw`가 finite nonzero gradient를 받는다. gate를 열면 residual,
   shared hidden, time 및 quantity output이 실제로 달라질 수 있다.
4. **검색 보존**: 동일한 state와 input에서 B와 후보의 top-4 index와 cosine
   similarity가 정확히 같다. 후보는 이 trace를 한 번만 의미적으로 사용한다.
5. **인과성·padding**: 미래 history/target, padding 값, 다른 batch series를
   바꿔도 현재 prefix가 바뀌지 않는다. empty memory와 zero row도 finite하다.
6. **수치 안정성**: 지원 dtype과 운영 범위의 극단 finite 입력에서 output,
   loss, 공통 gradient와 scalar gradient가 finite하다.
7. **저장·복원**: model, optimizer, selector, RNG를 strict하게 저장·복원하고
   resume 다음 step이 일치한다. `alpha_raw`와 contract metadata도 보존한다.
8. **복잡도**: 추가 계산은 `O(BLkD+MD)`, parameter는 scalar 1개다. GPU에서
   B 대비 median training-step ratio `<=1.25`, peak allocated memory ratio
   `<=1.10`을 요구한다. 전체 encoder attention 복잡도 `O(L²)`는 변하지 않는다.

Checkpoint-only 사전 감사는 train/validation/test row를 읽지 않고 세 B bank의
row norm과 모든 `64 choose 4` 조합을 확인한다. 이 감사의 목적은 후보가 scalar
shrinkage로 붕괴하지 않는지와 수치적으로 비퇴화인지 확인하는 것이다. 성능
채택 근거로 사용하지 않는다.

## GPU 실행 순서와 중단 조건

이 계약 자체는 GPU 실행 승인이 아니다. 로컬 계약과 독립 source commit이
준비된 뒤 다음 순서로 별도 승인받아 실행한다.

1. 5090 source/runtime/data checksum과 GPU 점유를 확인한다.
2. CUDA 계약 테스트와 B 대비 비용 측정을 실행한다.
3. 세 데이터셋 full-data e1을 실행해 처리 건수, gradient, finite 계산,
   checkpoint·optimizer 저장/복원만 판정한다. e1 수치로 채택하지 않는다.
4. 모두 통과하면 Instacart seed42 최대300/min40/patience40 screening을 한다.
5. Instacart가 아래 gate를 모두 통과할 때만 Taxi와 Intermittent seed42 범위를
   다시 확정한다. 추가 seed와 held-out test는 자동 실행하지 않는다.

## 사전 고정 성능 gate

Instacart seed42에서 B 대비 다음을 모두 요구한다.

- validation raw RMSE가 strict 감소한다.
- 전체 MAE ratio `<=1.01`.
- body(`<=train p95`) MAE와 `>train p99` MAE ratio가 각각 `<=1.02`.
- legacy clamped time loss 증가가 `<=0.01`이고 모든 metric이 finite하다.
- 후보의 같은 checkpoint를 frozen하고 공통 K=1 normalized-duration head로
  평가할 경우 proper Time NLL이 aligned-B `+0.01` 이내다. 이 시간 평가는
  quantity gate가 통과한 뒤에만 수행해 불필요한 학습을 피한다.

공통 Backbone 개선 후보로 확장하려면 같은 구조와 규칙으로 Taxi와
Intermittent에서도 raw RMSE가 strict 감소하고 동일 MAE/time guardrail을
통과해야 한다. FULL, BOUNDED-QK, LPHC 또는 외부 benchmark 대비 우위는 이
gate와 별도 주장이다.

## 데이터·평가 잠금

- 데이터와 B 기준선은
  `paper/contracts/hard_lmm_bounded_qk_screening_v1.json`의 현재 세 dataset
  binding을 재사용한다.
- quantity threshold는 각 dataset의 train split에서 이미 고정된 값만 쓴다.
- 모델 선택과 screening은 validation only다.
- held-out target loader, metric, artifact, 구조 선택 사용은 금지한다.
- seed42 결과를 본 뒤 epsilon, gate, loss, selector, threshold를 바꾸지 않는다.

설계 판단과 옵션 비교는
`.agents/results/architecture/architecture-recommendation-hard-lmm-value-norm-boundary.md`,
checkpoint geometry 결과는
`paper/results/hard_lmm_value_norm_boundary_20260909/`에 기록한다.
