# Original CNN+GRU54 최종 3seed Validation 비교

**원본 결속 검증은 통과했으나, 고정된 강한 지배 기준은 S2P2와 MLP16에 대해 세 데이터셋 모두 통과하지 못했다.** CNN+GRU54를 통합 승자로 선택하거나 Time NLL 악화를 숨기지 않는다.

주 지표는 전체 Validation raw quantity RMSE다. 전체/tail MAE와 Time NLL을 같은 quantity-selected epoch에서 함께 읽는다. 후보는 original CNN+GRU54 하나이며 seed42/52/62를 보존하고 폭·epoch·지표를 사후 재선택하지 않았다.

## Taxi

| 기준 | view/지표 | CNNGRU − 기준 평균 ± 표본 SD | 개선 seed | seed42 / 52 / 62 delta |
|---|---|---:|---:|---|
| S2P2 · 공통 출력부 | overall/qty_rmse | -27.291 ± 1.971 | 3/3 | -26.092 / -29.566 / -26.216 |
| S2P2 · 공통 출력부 | overall/qty_mae | -7.743 ± 1.194 | 3/3 | -7.414 / -9.068 / -6.749 |
| S2P2 · 공통 출력부 | overall/time_nll | 0.802597 ± 0.648203 | 0/3 | 0.673131 / 1.505762 / 0.228897 |
| S2P2 · 공통 출력부 | tail/qty_rmse | -180.874 ± 37.299 | 3/3 | -153.764 / -165.446 / -223.412 |
| S2P2 · 공통 출력부 | tail/qty_mae | -137.203 ± 18.427 | 3/3 | -124.658 / -128.592 / -158.359 |
| S2P2 · 공통 출력부 | tail/time_nll | -1.622571e-15 ± 2.810730e-15 | 1/3 | -4.868122e-15 / 4.101818e-19 / 2.803810e-32 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/qty_rmse | 1.237 ± 4.963 | 1/3 | -4.179 / 2.324 / 5.566 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/qty_mae | 0.102 ± 1.445 | 1/3 | -1.520 / 0.575 / 1.251 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/time_nll | 0.473453 ± 0.557908 | 1/3 | 0.686828 / 0.893182 / -0.159652 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/qty_rmse | 15.461 ± 45.195 | 1/3 | -36.112 / 34.335 / 48.159 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/qty_mae | 4.048 ± 48.347 | 1/3 | -50.907 / 23.013 / 40.038 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/time_nll | 1.367273e-19 ± 2.368186e-19 | 1/3 | -3.897574e-37 / 4.101818e-19 / 2.803810e-32 |

## Intermittent

| 기준 | view/지표 | CNNGRU − 기준 평균 ± 표본 SD | 개선 seed | seed42 / 52 / 62 delta |
|---|---|---:|---:|---|
| S2P2 · 공통 출력부 | overall/qty_rmse | -0.846 ± 0.116 | 3/3 | -0.719 / -0.946 / -0.873 |
| S2P2 · 공통 출력부 | overall/qty_mae | -0.234 ± 0.005 | 3/3 | -0.235 / -0.237 / -0.228 |
| S2P2 · 공통 출력부 | overall/time_nll | 0.067519 ± 0.022336 | 0/3 | 0.043255 / 0.072078 / 0.087224 |
| S2P2 · 공통 출력부 | tail/qty_rmse | -7.623 ± 0.681 | 3/3 | -7.567 / -6.972 / -8.331 |
| S2P2 · 공통 출력부 | tail/qty_mae | -5.905 ± 0.419 | 3/3 | -6.289 / -5.458 / -5.969 |
| S2P2 · 공통 출력부 | tail/time_nll | -0.032114 ± 0.055864 | 1/3 | -0.096620 / 0.000269 / 8.524198e-06 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/qty_rmse | 0.043 ± 0.090 | 1/3 | 0.136 / -0.045 / 0.038 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/qty_mae | -0.011 ± 0.002 | 3/3 | -0.009 / -0.013 / -0.013 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/time_nll | -0.102020 ± 0.080126 | 3/3 | -0.054087 / -0.194521 / -0.057453 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/qty_rmse | 0.757 ± 0.557 | 0/3 | 1.302 / 0.190 / 0.779 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/qty_mae | 0.648 ± 0.606 | 0/3 | 1.212 / 0.007 / 0.724 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/time_nll | -1.585718 ± 1.680667 | 2/3 | 2.439783e-14 / -3.347477 / -1.409678 |

## RAF

| 기준 | view/지표 | CNNGRU − 기준 평균 ± 표본 SD | 개선 seed | seed42 / 52 / 62 delta |
|---|---|---:|---:|---|
| S2P2 · 공통 출력부 | overall/qty_rmse | 0.189 ± 0.352 | 1/3 | 0.166 / 0.553 / -0.150 |
| S2P2 · 공통 출력부 | overall/qty_mae | -0.067 ± 0.078 | 3/3 | -0.028 / -0.156 / -0.016 |
| S2P2 · 공통 출력부 | overall/time_nll | 0.127684 ± 0.077191 | 0/3 | 0.161438 / 0.039366 / 0.182249 |
| S2P2 · 공통 출력부 | tail/qty_rmse | 4.080 ± 10.614 | 1/3 | 6.415 / 13.332 / -7.507 |
| S2P2 · 공통 출력부 | tail/qty_mae | 3.860 ± 12.035 | 1/3 | 6.282 / 14.499 / -9.201 |
| S2P2 · 공통 출력부 | tail/time_nll | 0.228361 ± 0.245277 | 1/3 | 0.220314 / -0.012794 / 0.477562 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/qty_rmse | 0.101 ± 0.244 | 1/3 | 0.145 / 0.320 / -0.162 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/qty_mae | -0.058 ± 0.088 | 2/3 | -0.046 / -0.152 / 0.024 |
| TitanTPP MLP 폭16 · 사용자 기준선 | overall/time_nll | 0.047208 ± 0.074817 | 1/3 | -0.007853 / 0.017087 / 0.132391 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/qty_rmse | -1.465 ± 7.589 | 2/3 | -1.301 / 6.041 / -9.135 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/qty_mae | -4.745 ± 9.401 | 2/3 | -5.972 / 5.209 / -13.473 |
| TitanTPP MLP 폭16 · 사용자 기준선 | tail/time_nll | 0.074229 ± 0.324987 | 2/3 | -0.170302 / -0.050022 / 0.443010 |

## 사전에 고정된 기준과 해석

실행 계약의 `reporting.adoption`은 3seed 모두 전체 RMSE가 개선되고 전체 MAE·tail RMSE/MAE·Time NLL이 어느 seed에서도 나빠지지 않을 때만 강한 지배로 인정한다. 이 보고서는 원래 문구의 Time NLL에 전체와 tail 양쪽을 포함하는 보수적 해석을 적용한다. 전체 Time NLL만 적용해도 결론은 동일하다. `strict_validation_gate.json`은 각 실패 지표와 seed delta를 보존한다.

Taxi에서는 외부 six 모두보다 수량 RMSE 평균이 낮지만 S2P2보다 Time NLL이 높고, 강한 내부 MLP 폭8/16 및 모든 가용 분기 MLP보다 전체 RMSE 평균이 높다. Intermittent에서도 외부 six보다 RMSE 평균은 낮으나 S2P2보다 Time NLL이 높고 MLP 폭4/8/16보다 RMSE가 높다. RAF에서는 S2P2와 MLP 폭4/8/12/16보다 전체 RMSE가 높다. 따라서 수량 예측의 외부 대조 이점과 시간/내부 대조 약점을 함께 보고한다.

n=3의 평균/표본 SD와 같은 seed 차이는 기술 통계다. 통계적 유의성, recurrence/CNN 상호작용의 독립 기여, 동일 GPU 효율 우월성, 독립 held-out 일반화 주장은 뒷받침하지 않는다.

## 원본과 평가 범위

새 fit은 5080의 Taxi/RAF seed52/62 네 개와 5090의 Intermittent seed52/62 두 개다. seed42 세 개는 A100 및 후속 GPU 연속 학습 이력이 있는 기존 원본을 재사용한다. 정확한 GPU segment, source revision/closure, checkpoint file/state SHA, endpoint/history/input SHA는 source_bindings에 있다. seed42를 이번 새 GPU fit이나 동일 GPU 반복으로 재명명하지 않았다.

이 작업은 로컬 JSON만 읽어 결과를 재집계했다. remote/GPU/optimizer/training/inference/evaluation, checkpoint deserialization, Test 성능·행·예측, Validation/Test 혼합 숫자 CSV·보고서 열람은 수행하지 않았다. 이미 개발에 노출된 Test를 향후 평가하더라도 독립 held-out 증거로 다시 이름 붙일 수 없다.

benchmark 수량과 NLL의 source는 기존 frozen registry에 결속된 original Validation endpoint이며 현재 코드 기본값으로 재현한 결과가 아니다. TRAIN/Validation identity·quantity hash와 TRAIN 통계, 고정 tail threshold, 전체 target 수, 첫 유한 수량 RMSE 최소 epoch, 같은 epoch NLL, selected state를 직접 검증했다. endpoint digest와 기존 평가 receipt digest는 서로 다른 namespace이므로 동일 digest처럼 비교하지 않았다.

## 다음 작업의 승인 경계

**기준선과 비교 기록 확정 — 완료**
- 이 디렉터리의 수량/시간 전체 및 tail 표, 같은 seed 차이, 원본 결속 영수증과 gate를 고정했다.

**Test9 실행 — 다음 작업 / 기존 승인 범위**
- 원본 결속 PASS는 강한 지배 PASS와 다르다. 현재 후보는 강한 지배 FAIL_TRADEOFF다.
- 이번 사용자 승인은 우월성 여부와 별개로 original 9 checkpoint를 평가하는 범위다. root 세션은 원본 결속과 새 native 전체 Validation9 replay gate를 통과한 뒤 기존 승인 범위의 Test9를 실행한다. 채택 FAIL_TRADEOFF는 과학적 주장만 제한하며 Test 평가 승인을 취소하지 않는다.
- 실행할 경우 original CNN+GRU54의 3 dataset × seed42/52/62 checkpoint/epoch를 그대로 쓰며 재학습·재선택·대체 fit은 제외한다. 기존 Test의 개발 노출 한계를 유지한다.

Intermittent 모든 가용 분기 MLP와 현재 상태만 seed62의 Runpod train/validation 파일 포장 SHA는 canonical 파일과 다르다. 두 조건 모두 ordered TRAIN/Validation target identity·quantity SHA 및 TRAIN 통계가 동일하고, frozen runtime interface의 TRAIN/Validation raw dt SHA 검사와 주 단위 lognormal 관측 계약도 동일함을 source_bindings에서 검증했다. 파일 SHA를 동일한 것으로 보고하지 않는다. 재사용 seed42의 cross GPU endpoint와 같은 epoch 학습 이력 사이에는 원래 gate 허용오차 이내 차이가 있으며 원차이·허용오차를 보존했다.
