# TitanTPP H3 CUDA Preflight 결과

## 판정

- 실행 서버: 5080, NVIDIA GeForce RTX 5080
- source revision: `387b1f515e23a75cb1402ffb14e4673a34ffb740`
- focused tests: `42 passed`
- Intermittent CUDA 2-batch: 통과
- Instacart top-20 e1: 통과
- checkpoint 및 artifact 생성: 통과
- finite loss/gradient/metric: 통과
- held-out test: 사용하지 않음

H3의 density, Jacobian, conditional median, survival, extreme duration backward와
train-only initialization contract를 CUDA에서 확인했다. Intermittent와 Instacart 모두
Hard-LMM+T1 실제 데이터 경로에서 best 및 last checkpoint를 생성했다.

## 초기 telemetry

| Dataset | Train joint | Train Time NLL | Gradient norm | Clip fraction | Validation Time NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intermittent 2-batch | 4.095860 | 1.943730 | 17.3183 | 100.00% | 1.906728 |
| Instacart top-20 e1 | 2.080521 | 1.871523 | 5.4109 | 97.70% | 1.842660 |

Smoke 단계의 clipping 비율은 높지만 모든 값이 finite이고 optimizer step 이후
parameter도 finite였다. Validation을 보고 time-head 상수나 learning rate를 바꾸지
않으며, seed-42 e300 history에서 clipping 감소와 loss spike를 함께 확인한다.

## 본 실행 초기 진입

F0 H0+Hard-LMM+T1은 2026-08-20 13:25:37 KST에 시작했다. 첫 epoch에서 train joint
`10.419540`, clipping `16.09%`, validation joint `1.153566`, Time NLL `1.147343`을
기록했고 모든 값은 finite였다. Runner는 F0 완료 후 F1 H3를 자동 실행한다.
