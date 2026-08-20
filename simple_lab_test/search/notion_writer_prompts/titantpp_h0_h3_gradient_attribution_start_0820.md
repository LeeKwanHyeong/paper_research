# Notion Start Source: H0/H3 Gradient Clipping and Quantity Damage Audit

## 위치

- `5. Model Design Enhancement`
- 페이지 제목: `2026-08-20 H0-H3 Gradient Clipping and Quantity Damage Audit`

## 상태

- 5080 실행 준비 완료
- train-only diagnostic
- validation 및 held-out test 미사용

## 목적

H0는 비교 기준선으로만 유지하고 학습 폭증을 미해결 위험으로 관리한다. H3는 기존
checkpoint를 사용해 100% clipping의 parameter-group driver와 quantity 손상 위치를
확인한다.

## Variant 계약

| Variant | Time head | 현재 상태 | 진단 state |
| --- | --- | --- | --- |
| H0 | scaled exact RMTPP | 비교 기준선, 안정성 미해결 | initial/best/final |
| H3 | log-normal duration | 미채택 | initial/best/final |

Shared encoder, time head, quantity head의 gradient norm과 global clipping scale을 같은
train batch에서 비교한다. Best checkpoint의 encoder와 quantity head도 교차 적용한다.

## 고정 조건

- Intermittent train split, seed 42
- batch 128, audit batch 32
- lookback 520주, maximum sequence length 256
- gradient clipping `1.0`
- H3 설정을 결과에 맞춰 변경하지 않음

## 실행 명령어

```bash
SOURCE_REVISION=ccbcfcf9210d5f0bb6e60adde3f7b431058a435f \
  bash simple_lab_test/search/scripts/run_count_aware_h0_h3_gradient_attribution_20260820.sh
```

## 결과

실험 완료 후 작성한다.
