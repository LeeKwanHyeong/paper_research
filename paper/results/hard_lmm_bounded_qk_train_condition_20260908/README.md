# BOUNDED-QK Instacart train-only 필요조건 확인

## 결론

사전에 고정한 필요조건은 **두 series-disjoint fold에서 모두 통과**했다. 따라서
BOUNDED-QK 계열을 중단하지 않고, 다음 단일 후보로
`level-preserving history-confidence residual`을 계약했다.

Train에서도 validation과 같은 구조가 재현됐다. BOUNDED-QK는 B보다 MAE와 centered
MSE를 줄였지만 prediction level을 약 `0.148` 더 낮췄다. 이 하향 shift가 bias 제곱을
늘려 centered-error 이득을 덮었고 raw RMSE를 악화시켰다. 이력 `2–3`에서는 같은
하향 이동과 MSE 악화가 강했고, 이력 `8–15`에서는 반대로 MSE가 개선됐다.

## 범위와 해석 한계

- 데이터: Instacart canonical train target `1,991,192`건
- 전체 train row `2,197,401`, 전체 series `206,209`, target이 있는 series `205,934`
- 비교: Frozen B와 Frozen BOUNDED-QK, 모두 seed42
- B prediction은 기존 exact cache를 재사용하고 BOUNDED-QK만 한 번 추론
- train row만 predicate pushdown으로 materialize
- 학습, checkpoint 재선택, calibration, validation target, held-out test 접근 없음
- 새 salt `hard_lmm_bounded_qk_train_condition_v1:20260908`로 whole-series 2-fold 배정

두 checkpoint는 전체 train split으로 이미 학습됐고 validation raw RMSE로 선택됐다.
따라서 두 fold는 독립 학습이나 out-of-fold 성능 평가가 아니다. 이 결과가 보이는 것은
관측된 현상이 train series 내부에서도 안정적으로 반복된다는 점이며, 새 모델의
일반화나 성능 개선이 아니다.

## 전체 train 결과

| 모델 | MAE | Raw RMSE | Bias | Centered RMSE |
|---|---:|---:|---:|---:|
| B | 3.891614 | 5.637565 | -0.471150 | 5.617843 |
| BOUNDED-QK | **3.884059** | 5.648391 | -0.618779 | **5.614396** |

BOUNDED-QK − B 차이는 다음과 같다.

- MAE `-0.007555` (`-0.194%`)
- raw RMSE `+0.010826` (`+0.192%`)
- mean prediction shift와 bias 변화 `-0.147629`
- centered MSE `-0.038719`
- bias 제곱 `+0.160906`
- net MSE `+0.122186 = -0.038719 + 0.160906`

Bias 제곱 증가는 net MSE 악화의 `131.69%`이고, centered MSE 개선이 그중 `31.69%`를
상쇄했다. MAE를 gate에 포함한 이유는 centered MSE 감소만으로 후속 구조를 정당화하지
않고, BOUNDED-QK의 유용한 방향이 절대오차에서도 재현되는지 함께 요구하기 위해서다.

## Fold 반복 확인

| Fold | Targets | Series | ΔMAE | ΔRMSE | Mean shift | Median series shift | ΔCentered MSE | ΔBias² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 996,690 | 102,983 | -0.007707 | +0.010631 | -0.147339 | -0.153250 | -0.039866 | +0.159915 |
| 1 | 994,502 | 102,951 | -0.007403 | +0.011023 | -0.147921 | -0.153029 | -0.037573 | +0.161901 |

각 fold에서 shift 크기는 B RMSE의 `2.612%`, `2.625%`로 사전 최소치 `1%`를
넘었다. 두 fold 모두 centered MSE와 MAE는 좋아졌고, bias 제곱 증가가 centered
이득을 넘어 raw MSE를 악화시켰다. 고정된 11개 gate가 모두 같은 방향이었다.

## 이력 길이 기전

| Fold | 이력 | Targets | Mean shift | ΔMSE | ΔMSE / B MSE | ΔMAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2–3 | 248,115 | -0.518960 | +0.607813 | +1.872% | -0.009763 |
| 1 | 2–3 | 248,685 | -0.515423 | +0.614058 | +1.890% | -0.010616 |
| 0 | 8–15 | 240,360 | +0.119528 | -0.235660 | -0.750% | -0.004308 |
| 1 | 8–15 | 238,997 | +0.118723 | -0.261452 | -0.841% | -0.005066 |

이력 `2–3`의 MSE 악화는 두 fold 모두 사전 최소치 `1%`를 넘었다. 반대로
`8–15`에서는 두 fold 모두 MSE가 개선됐다. 따라서 target 수량이나 데이터셋 이름으로
분기할 이유보다, 관측된 전이가 적을 때 Q/K/V residual을 줄이고 절대 level 성분을
읽지 못하게 하는 공통 구조를 시험할 근거가 생겼다.

## 후속 단일 후보

선정 후보는 `LPHC-QKV`다. Q/K/V의 kernel3 level residual 대신
`현재−직전`, `직전−두 번째 이전`의 causal difference만 학습하고, 고정 confidence
`gamma=관측된 인접 전이 수/유효 사건 수`를 곱한다. 연속 이력에서는
`gamma(H)=(H-1)/H`와 같다. Q/K는 절대 projection RMS가 아니라 headwise
transition RMS로 제한하고 V의 전이 경로도 유지한다.

이 residual은 constant projection level에 대해 정확히 0이고 H1에서는 닫힌다.
다만 최종 scalar quantity bias가 0이라는 보장은 없으므로 Instacart seed42에서
raw RMSE, centered MSE, MAE와 H2–3/H8–15 MSE를 직접 판정한다. 전체 수식과 구현·
성능 계약은
[`hard_lmm_level_preserving_history_confidence_residual_v1.md`](../../contracts/hard_lmm_level_preserving_history_confidence_residual_v1.md)에 고정했다.

## 실행·재현 감사

- 진단 source commit: `1930e5fbfe123da606af3746b002f792a3934071`
- Frozen model source: `89cd700c28dd169cd59bfccbd694ef8967bac823`
- 데이터 SHA: `06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2`
- target identity SHA: `c361f5e9904c25f18c91007f6e1209518a02fb755799e81c8300ee759b2a1210`
- B cache SHA: `bbe3cd5fe995ec15afc681c236c2ff0fc9b900197dff309c399ad3cb0f390d7d`
- BOUNDED-QK prediction cache SHA: `8a5912b7e20a6410f0290c886240b5e701b6969deb3e8af2384ef067a76ee79e`
- BOUNDED-QK checkpoint SHA:
  `823352e962356274824f095e60abe5285d8885a81cb0c9f2cf469ee725f76e74`
- BOUNDED-QK state SHA:
  `41213ef4cabcf9788017266d7732b19e1e0180fa4f708357050dc688d0e4e9a5`
- CUDA inference: RTX 5090, batch512, `31.49s`, peak allocated `189,469,184` bytes
- model state는 추론 전후 동일하고 모든 gradient는 absent

원격 분석을 동일 cache로 macOS에서 다시 실행했다. OS별 NumPy reduction 순서로
최대 `2.8422e-14`의 반올림 차이는 있었지만, 두 fold의 22개 pass/fail 값과 최종
decision은 정확히 일치했다.

## 증적

- `analysis.json`: 전체·fold·이력별 float64 수치와 11개 gate
- `fold_metrics.csv`: fold별 핵심 결과
- `history_metrics.csv`: 전체·fold별 고정 이력 구간 결과
- `inference_audit.json`: source/data/checkpoint/cache/state/runtime 감사
- `local_replay_audit.json`: 5090 결과의 로컬 재실행 일치 감사

전체 row prediction cache는 크기 때문에 Git에 넣지 않고
`search_artifacts/hard_lmm_bounded_qk_train_condition_20260908/remote/run/`에 보관한다.
