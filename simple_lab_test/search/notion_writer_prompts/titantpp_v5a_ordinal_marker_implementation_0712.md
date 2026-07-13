기존 Notion 페이지 `TitanTPP V5a Ordinal Marker Loss Contract And Acceptance Gate`에
아래 구현 결과를 업데이트해주세요.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- Model Enhancement 기록이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.
- 같은 제목의 기존 페이지를 업데이트하고 중복 페이지는 만들지 않습니다.
- 기존 설계, acceptance gate, threshold는 유지합니다.

## 작성 톤

- 상태 보고처럼 간결하게 씁니다.
- 구현 사실, 검증 결과, 아직 하지 않은 일을 구분합니다.
- `획기적인`, `강력한`, `유의미한`, `주목할 만한`, `명확히 입증`,
  `종합적으로` 같은 표현은 쓰지 않습니다.
- local test 통과를 모델 성능 개선으로 해석하지 않습니다.
- 5090 실험이나 V5a 학습 결과가 나온 것처럼 쓰지 않습니다.

## 상태 변경

기존:

```text
design confirmed / implementation not started
```

변경:

```text
design confirmed / local implementation complete / CPU gate passed / 5090 integration pending
```

## 구현 내용

- shared helper `models/RMTPPs/marker_losses.py` 추가
- real mark logits만 사용하는 normalized RPS 구현
- mask 기반 평균과 real class 1개일 때 zero 처리
- `marker_loss_mode=ce|ce_rps`, default `ce`
- `lambda_ordinal`, default `0.0`
- `ce`는 lambda `0`, `ce_rps`는 positive lambda를 요구하도록 fail-fast
- V5a 첫 설정은 `ce_rps`, `lambda_ordinal=0.10`, TitanTPP-only

Loss identity는 다음과 같이 구현했습니다.

```text
nll_marker = categorical CE
nll = nll_marker + nll_time
ordinal_marker_loss = normalized RPS
marker_train_loss = nll_marker + lambda_ordinal * ordinal_marker_loss
```

- `nll_marker`, `nll`, `best_val_nll` 의미는 바꾸지 않음
- `residual_only`, `hybrid`, `qty_only`에서 weighted RPS를 공통 composer가 한 번만 가산
- PAD는 RPS에서 제외하고 CE에서는 기존처럼 full logits 사용
- CE와 RPS에 같은 transition mask와 `all|target_only` scope 적용

## Configuration And Artifacts

- unified CLI와 model-test CLI에 두 option 추가
- model config와 TitanTPP construction에 전달
- manifest, checkpoint, resume/cache identity, summary, history에 기록
- V5a run path: `markloss_ce_rps/lambdaord_0p1`
- default V2 `ce` run path는 유지
- validation/test normalized RPS 추가
- balanced accuracy, macro F1, mark MAE, adjacent accuracy 추가
- mark-0 prediction share, mark `0/1` recall 추가
- validation/test per-class support/precision/recall/F1 artifact 추가
- learning curve에 validation RPS와 mark MAE 추가

## Local Verification

| 검증 | 결과 |
| --- | --- |
| V5a focused contract tests | `20 passed` |
| 기존 V3/V3b/V3c focused tests | `18 passed` |
| Intermittent diagnostic tests | `4 passed` |
| default CPU model-test | RMTPP, TitanTPP, THP 모두 success |
| V5a CPU model-test | TitanTPP `small_lmm` success |

V5a CPU model-test의 finite 값:

| Metric | Value |
| --- | ---: |
| Total NLL | `4.772340` |
| Marker CE | `2.457612` |
| Time NLL | `2.314728` |
| Normalized RPS | `0.194920` |
| Marker train loss | `2.477104` |

`2.477104 = 2.457612 + 0.10 * 0.194920` 관계를 만족합니다.

Focused test에서 확인한 contract:

- correct deterministic RPS `0`
- deterministic RPS가 ordinal distance에 비례
- RPS 범위 `[0,1]`
- PAD logit 변경은 RPS에 영향이 없고 CE에는 영향이 있음
- CE/RPS transition mask 및 `all|target_only` scope 일치
- V2/V5a parameter, state dictionary, forward, core loss 동일
- isolated RPS gradient는 mark head와 Titan encoder로 전달되고 time/value head에는 전달되지 않음
- quantity loss mode별 weighted RPS 한 번 가산
- invalid config/CLI fail-fast
- V2와 V5a run path 분리 및 legacy schema 기본값 유지

## 아직 진행하지 않은 항목

- 5090 CUDA `small_lmm` model-test
- 5090 Instacart top-20 e1 smoke
- Intermittent seed-42 e50 screening
- V2/V5a strict matched multi-seed
- held-out test audit

따라서 현재 결론은 `구현 계약과 로컬 회귀 검증 통과`까지만 기록합니다. V5a가
V2보다 성능이 높다는 결론은 쓰지 않습니다.

## Local Evidence

```text
/Users/igwanhyeong/PycharmProjects/paper_research/models/RMTPPs/marker_losses.py
/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/tests/test_titantpp_ordinal_marker_loss.py
/Users/igwanhyeong/PycharmProjects/paper_research/.agents/results/architecture/adr-titantpp-v5a-ordinal-marker-rps-loss.md
/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/model_enhancement_strategy.md
```

페이지 마지막 Next는 아래 한 줄로 남깁니다.

```text
Next: 5090 V5a CUDA model-test, Instacart e1 smoke, Intermittent seed-42 e50 screening 준비 및 실행
```
