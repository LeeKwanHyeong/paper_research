# TitanTPP Scaled Exact Time Head v2 실험 기록

## 상태

- 상태: 계약 설계 완료, 구현 전
- 설계 일자: 2026-08-20
- 실행 서버: 5080
- Held-out test: 잠금

## 목적

현재 scaled exact RMTPP head의 수학적 exactness는 유지하면서, train loss spike를
유발한 slope 범위, intercept 범위와 초기 hazard를 안정화한다. H1을 우선 검증하고
H1이 train-only gate를 실패할 때만 낮은 time-head learning rate의 H2를 연다.

## Variant 계약

| Variant | `w * tau` budget | Intercept | 초기값 | Time-head LR |
| --- | ---: | --- | --- | ---: |
| H0 | 40 | hard clamp `±30` | `log(time_scale)` | `1.0x` |
| H1 | 8 | smooth tanh `±6` | train mean rate | `1.0x` |
| H2 | 8 | H1과 동일 | H1과 동일 | `0.1x` |

## 선택 원칙

Validation을 읽지 않는 train-only stability runner로 H0와 H1을 먼저 비교한다. H1이
사전 gate를 통과하면 H2는 실행하지 않는다. H1이 실패할 때만 H2를 실행하며, H2도
실패하면 validation screening을 시작하지 않는다.

## 결과

