# TitanTPP V4 Mark-Conditioned Time Head Implementation

Notion의 `5. Model Design Enhancement` 아래 제목 2
`2026-07-16 | V4 Mark-Conditioned Time Head`, 제목 3
`Step 2. time_head_mode 구현 및 Focused Contract Test`로 정리한다.

## 상태

- 상태: `구현 완료 - 로컬 gate PASS`
- 5090 CUDA 및 dataset 학습: `미실행`

## 목적

- 기존 shared RMTPP time head를 유지하면서 next mark별 additive intensity
  intercept를 선택적으로 학습한다.
- V2와 V3b를 zero-init control로 보존하고, validation-only V4 screening을
  실행할 수 있는 artifact 계약을 만든다.

## Factorial 계약

| Variant | Value head | Quantity-mark gradient | Time head |
| --- | --- | --- | --- |
| V2 | shared | coupled | shared |
| V3b | mark-conditioned experts | detached | shared |
| V4a | shared | coupled | mark-conditioned |
| V4b | mark-conditioned experts | detached | mark-conditioned |

## 고정 조건

- real mark별 `Linear(d_model, C)` intercept delta
- weight와 bias 모두 zero-init
- RMTPP positive slope `w`는 mark 간 공유
- PAD expert 없음
- audit에서 적합한 intercept와 `w`는 모델 초기값으로 사용하지 않음
- 학습 likelihood는 observed mark, 추론 median은 predicted real mark 사용
- `evaluation_scope=validation_only`에서 test metric을 생성하지 않음

## 실행 명령어

```bash
python -m pytest -q \
  simple_lab_test/search/tests/test_titantpp_mark_conditioned_time_head.py \
  simple_lab_test/search/tests/test_titantpp_mark_conditioned_value_head.py \
  simple_lab_test/search/tests/test_titantpp_ordinal_marker_loss.py \
  simple_lab_test/search/tests/test_titantpp_direct_raw_quantity.py \
  simple_lab_test/search/tests/test_titantpp_q3_factorial.py \
  simple_lab_test/search/tests/test_reproducibility_controls.py
```

## 결과

- focused 및 관련 회귀 테스트 `96 passed`
- CPU V4a/V4b model-test 모두 성공
- zero-init NLL은 두 variant 모두 `3.862866`
- V4a/V4b parameter 수는 각각 `297,516 / 298,032`
- 다음 gate는 5090 CUDA model-test와 Instacart top-20 e1 smoke
