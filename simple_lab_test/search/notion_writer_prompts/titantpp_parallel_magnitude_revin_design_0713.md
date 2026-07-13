다음 TitanTPP parallel direct magnitude regression과 causal shrinkage RevIN 설계를
Notion에 정리해주세요.

사후 범위 정정 (`2026-07-13`): 이 문서의 M0-M4는 모두 `log2(qty)`를 대상으로
설계된 log-domain family다. M0는 fixed train-global normalization이며 RevIN이
아니다. 아래의 `M0 실패 -> RevIN track 중단`은 기존 log-domain M1-M4에만
적용하고, raw-quantity RevIN의 성능 결론으로 일반화하지 않는다.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- Model Enhancement 설계이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.
- 관련 페이지 `TitanTPP V5a Intermittent Seed-42 e50 Validation Screening`과
  `TitanTPP V5a Ordinal Marker Loss Contract And Acceptance Gate`에 연결합니다.
- 같은 제목의 페이지가 있으면 업데이트하고 중복 페이지를 만들지 않습니다.

## 페이지 제목

- `TitanTPP Parallel Magnitude Decoder And Causal Shrinkage RevIN Design`

## 문체

- 설계 결정, 이유, 제약, 검증 방법 순서로 간결하게 작성합니다.
- 구현이 끝난 것처럼 쓰지 않습니다.
- 확인된 기존 결과와 앞으로 검증할 가설을 구분합니다.
- 과장된 표현이나 일반적인 AI 문장은 넣지 않습니다.

## 현재 상태

- 상태: `design completed; implementation not started`
- 설계일: `2026-07-13 KST`
- V5a validation gate 실패 후 V5b prior correction보다 direct magnitude track을 우선
- 다음 작업: train-only Intermittent history/variance/level-shift audit
- 아직 실험을 시작하지 않았으며 서버/tmux 정보는 없음

## 배경

- Intermittent train target의 mark 0-2 share는 `86.60%`
- V2 validation mark-0 prediction share는 `45.03%`
- V5a validation mark-0 prediction share는 `58.75%`
- V5a mark-1 recall은 `24.66%`로 하락
- RPS는 mark order를 반영하지만 class prior와 marker argmax 기반 quantity
  reconstruction을 제거하지 않음
- fixed split에는 `z=mark+scale_residual=log2(demand_qty)` 연속 target이 이미 존재

## 핵심 결정

첫 magnitude track에서 marker head를 제거하지 않습니다.

```text
Titan encoder
  + categorical marker head         -> 기존 CE/NLL
  + continuous-time head            -> 기존 time NLL
  + direct log2-magnitude head       -> quantity prediction
```

`parallel`은 세 task가 encoder를 공유한다는 의미입니다. Legacy mark-residual
decoder와 direct magnitude decoder를 동시에 학습하거나 ensemble하지 않습니다.
Run마다 quantity decoder는 하나만 활성화합니다.

| Mode | Quantity prediction | Marker/time likelihood |
| --- | --- | --- |
| `mark_residual` | 기존 predicted mark + residual | unchanged |
| `direct_log_qty` | direct denormalized log2 quantity | unchanged |

Mark head를 제거한 continuous mark density 모델은 likelihood가 달라지므로 후속 별도
모델로 보류합니다.

## Input And Leakage Contract

- 첫 track은 `scale_base=2`, fixed split, `target_only`만 지원
- `z=mark+scale_residual=log2(demand_qty)` 사용
- appended target은 loss에만 사용하고 normalization/input에서는 제외
- padding은 count, mean, variance, loss에서 제외
- train-global/per-series 통계는 train rows만 사용
- validation/test target은 scaler를 갱신하지 않음
- direct quantity prediction은 predicted mark, mark probability, legacy value head에
  의존하지 않음

## Variant Contract

| Variant | Normalization | Stat context | Role |
| --- | --- | --- | --- |
| M0 | train-global | no | direct regression baseline |
| M1 | per-series train-only, global fallback | no | fixed-series ablation |
| M2 | causal window RevIN | no | plain RevIN ablation |
| M3 | causal shrinkage RevIN | no | shrinkage effect |
| M4 | causal shrinkage RevIN | yes | primary candidate |

기존 제안의 M3를 M3/M4로 나눈 이유를 적습니다. Shrinkage와 statistic context를
동시에 추가하면 어느 요소가 개선을 만들었는지 구분할 수 없기 때문입니다.

## Shrinkage Contract

```text
alpha = n / (n + k)
mu = alpha * mu_history + (1-alpha) * mu_global
m2 = alpha * (var_history + mu_history^2)
   + (1-alpha) * (var_global + mu_global^2)
var = max(m2 - mu^2, sigma_floor^2)
scale = sqrt(var)
```

- 표준편차를 직접 평균하지 않고 first/second moment를 혼합
- one-event 또는 constant history에서도 finite
- `k`, `sigma_floor`, exp2 clamp는 train-only audit 후 validation 전에 고정
- M4 head context: `[mu, log(scale), log1p(history_count)]`

## Head And Loss Contract

```text
z_hat_norm = magnitude_head(hidden, optional_stat_context)
z_hat = mu + scale * z_hat_norm
qty_hat = 2 ^ z_hat

magnitude_loss = Huber(z_hat_norm, z_target_norm)
direct_qty_loss = Huber(qty_hat / qty_scale, qty / qty_scale)

total_loss = marker_train_loss
           + lambda_dt * nll_time
           + 1.0 * magnitude_loss
           + 0.25 * direct_qty_loss
```

- 첫 실험 marker objective는 plain CE
- V5a RPS, V5b prior correction, V3 expert/detachment와 결합하지 않음
- `nll_marker`, `nll_time`, `nll=nll_marker+nll_time` 의미 유지
- legacy `value_loss/value_mae`에 새 의미를 덮어쓰지 않음

## Metric And Checkpoint Contract

- 새 metric: `log_qty_mae`, `log_qty_rmse`, normalized `magnitude_loss`,
  `qty_rmse`, `qty_wape`, signed bias
- scale-wise quantity metric 유지
- context length bucket: `1`, `2-4`, `5-8`, `9+`
- primary checkpoint: `best_val_nll`
- diagnostic checkpoint: `best_val_qty_mae`
- quantity checkpoint만 좋아진 모델은 TitanTPP enhancement로 승격하지 않음

## Initial Validation Gate

M0 versus V2 at `best_val_nll`:

- quantity MAE와 log-quantity MAE 각각 `>=3%` 개선
- share `>=5%` quantity bucket regression `<=5%`

M3/M4 versus M0:

- 전체 quantity/log-quantity MAE 각각 `>=2%` 개선
- history count `<=4`에서 두 metric 각각 `>=3%` 개선
- M4가 M3를 이기지 못하면 M3 선택

Marked-TPP safety versus V2:

- marker NLL regression `<=1%`
- total/time NLL regression `<=0.5%`
- mark accuracy gap `>=-0.25%p`
- DT MAE regression `<=2%`
- 모든 context bucket finite

## Original Pre-M0 Decision Branch (Post-result Scope Corrected)

- M0 실패: 기존 log-domain M1-M4 branch만 중단하고 raw-quantity RevIN은
  미검증으로 유지; V5b class-prior correction은 독립 fallback으로 유지
- M0 통과, M3/M4 실패: direct regression 효과만 유지하고 RevIN claim은 하지 않음
- M3/M4 통과: 가장 단순한 통과 후보만 strict matched multi-seed로 승격
- held-out test는 candidate/constants 고정 전까지 읽지 않음

## 구현 예정 경로

```text
models/RMTPPs/magnitude_normalization.py
models/RMTPPs/config.py
models/RMTPPs/TitanTPP.py
data_loader/event_seq_data_module.py
utils/training.py
simple_lab_test/search/common/runner.py
simple_lab_test/search/tests/
```

페이지 마지막 Next:

```text
Next: Intermittent train-only history length, variance, and level-shift audit으로 shrinkage_k, sigma_floor, exp2 clamp를 고정한다.
```
