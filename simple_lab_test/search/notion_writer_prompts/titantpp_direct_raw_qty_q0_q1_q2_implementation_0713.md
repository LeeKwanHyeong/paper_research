# TitanTPP Matched Q0/Q1/Q2 direct_raw_qty 구현 업데이트

기존 Notion `5. Model Design Enhancement` 아래의 다음 페이지를 업데이트한다.

- 상위 history: `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- 기존 계약 페이지: `https://app.notion.com/p/39cbbe40561381dda378d65257d6719c`
- train-only audit 페이지: `https://app.notion.com/p/39cbbe40561381219cb4cbadf4e392df`

새 계약 페이지를 만들지 않는다. 기존 계약 페이지에 구현 결과를 추가하고,
상위 history에는 제목 3 `Step 8. Matched Q0/Q1/Q2 direct_raw_qty 구현과 Local Contract Gate`를 추가한다.

## 기록 범위

- 구현일: `2026-07-13 KST`
- 현재 상태: `implementation completed; local contract gate passed; 5090 CUDA pending`
- 성능 판정: 아직 없음
- RevIN benefit 판정: 아직 없음
- 다음 단계: 5090 CUDA Q0/Q1/Q2 model-test

## 구현 내용

- `qty_decoder_mode=direct_raw_qty`를 기존 `mark_residual`, `direct_log_qty`와 배타적으로 추가
- Q0 `global`, Q1 `causal_revin`, Q2 `causal_shrinkage_revin`을 stateless `MagnitudeContext`로 구현
- appended target과 padding을 raw center, scale, normalized history에서 제외
- raw interface reconstruction은 `q=2**(mark+scale_residual)` 유지
- Q2는 표준편차 평균이 아니라 first/second moment를 `alpha=n/(n+k)`로 혼합
- Intermittent train-only freeze 값 `k=8`, `eps=1e-5`, raw global moments와 effective floor 사용
- Q0/Q1/Q2는 동일 magnitude input projection, head, parameter key/count, initialization 사용
- marker CE/NLL, RMTPP time NLL, `nll=nll_marker+nll_time` 의미 유지

## Loss와 출력

```text
raw_norm_loss = Huber(u_hat, u_target)
raw_qty_loss = Huber(q_affine, q_target)

total_loss = marker_ce
           + 1.0 * time_nll
           + 1.0 * raw_norm_loss
           + 0.25 * raw_qty_loss
```

- training raw quantity loss에는 clamp 전 `q_affine` 사용
- evaluation/inference에서만 `q_hat=max(q_affine,0)` 적용
- upper semantic clamp 없음
- log2 quantity 지표는 evaluation-only

## Runner와 Artifact

- fixed-split train 행에서 raw global mean/population variance/std를 float64로 계산
- raw floor는 `max(0.001*global_std,1e-4)`로 계산해 model config에 전달
- Q2 경로에 `qtydecoder_direct_raw_qty/magnorm_causal_shrinkage_revin/.../k_8` 포함
- manifest, checkpoint, history, summary, scale-wise, cache, resume identity에 decoder,
  norm mode, raw domain, eps, k, floor, center, affine, stat-context와 global moments 기록
- validation history에 raw quantity MAE/RMSE/WAPE, evaluation-only log2 MAE/RMSE,
  context count `1`, `2-4`, `5-8`, `9+`, pre-clamp negative share,
  center/scale percentile, normalized-target p95/p99/non-finite count 기록

## Local Contract Gate 결과

- dedicated raw test: `22 passed`
- complete search tests: `85 passed`
- 기존 direct log M0 회귀 test 포함 통과
- Q0/Q1/Q2 local CPU model-test: 모두 success, finite forward/loss
- Q0/Q1/Q2 seeded `state_dict`: key, count, tensor initialization exact match
- target/padding mutation과 left/right padding context gate: 통과
- Q0/Q1/Q2 수식과 normalize/denormalize round trip: 통과
- direct raw quantity의 marker-logit 독립성: 통과
- raw loss와 marker/time loss gradient isolation: 통과
- negative `q_affine`의 quantity-loss gradient 유지: 통과
- stale raw global moments 또는 k가 다른 cache/resume 차단: 통과

## 해석 제한

이번 단계는 구현과 local contract 검증이다. CUDA runtime, actual-data backward,
validation accuracy, V2 대비 개선, Q1/Q2의 Q0 대비 RevIN benefit은 아직 확인하지 않았다.
CPU model-test 수치를 모델 성능으로 해석하지 않는다.

## 상위 History 갱신

`2026-07-13 | Direct Magnitude Regression과 RevIN Track` 아래 Step 7 다음에 추가:

### Step 8. Matched Q0/Q1/Q2 direct_raw_qty 구현과 Local Contract Gate

Raw global, causal masked RevIN, causal moment-shrinkage RevIN을 동일 direct raw
head로 구현했다. Dedicated raw contract test 22개와 search 전체 85개가 통과했고,
Q0/Q1/Q2 CPU model-test도 finite하게 완료됐다. 아직 5090 CUDA와 actual-data
성능 실험 전이므로 RevIN 효과 판정은 하지 않는다.

상위 `현재 의사결정`은 다음처럼 갱신한다.

- Raw Q0/Q1/Q2 구현 및 local gate: 완료
- Q1: plain RevIN diagnostic only
- Q2: `k=8` primary short-context candidate
- 다음 분기: 5090 CUDA Q0/Q1/Q2 model-test
- Instacart e1과 Intermittent e50은 CUDA gate 이후 진행

## 로컬 근거

- `.agents/results/architecture/adr-titantpp-raw-quantity-revin-q0-q1-q2.md`
- `models/RMTPPs/magnitude_normalization.py`
- `models/RMTPPs/TitanTPP.py`
- `simple_lab_test/search/common/runner.py`
- `utils/training.py`
- `simple_lab_test/search/tests/test_titantpp_direct_raw_quantity.py`
