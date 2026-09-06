# Frozen Hard-LMM time-head refit: seed42 validation audit

**판정: B의 수량 예측을 그대로 유지하면서 validation Time NLL을 세 데이터셋 모두 개선했다. 그러나 사전 고정한 `A + 0.01` 기준은 Instacart만 통과했으므로, 공통 시간 성능 복구와 Backbone 개선은 입증되지 않았다.**

## 검증 범위

- 후보는 raw-RMSE로 선택한 B 체크포인트의 encoder, memory, quantity head를 고정하고 기존 RMTPP time head의 66개 파라미터(`v_t.weight`, `b_t`, `w_raw`)만 train split에서 재학습한다.
- checkpoint는 validation Time NLL의 epoch 0 포함 가장 이른 유한 최솟값으로 선택했다. 세 데이터셋에 같은 optimizer와 정지 규칙을 적용했다.
- 평가는 seed42 validation에만 한정했다. Held-out test와 seeds52·62는 실행하지 않았다.
- RTX 5090 CUDA e1 계약 검증 뒤 full refit을 실행했다. Full refit의 합산 시간은 `55.842`초다. Feature cache 작성 시간과 controller 검사는 이 합계에 포함되지 않는다.

## 결과

`ΔB`와 `ΔA`는 각각 `refit − B`, `refit − A`이며 Time NLL은 낮을수록 좋다. Strict gap은 `refit − (A + 0.01)`이고 0 이하여야 통과다.

| 데이터셋 | A Time NLL | B Time NLL | Refit Time NLL | ΔB | ΔA | Strict gap | Raw RMSE | MAE | best/completed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Intermittent | -3.592856041 | -3.483793693 | -3.537757001 | -0.053963308 | +0.055099040 | +0.045099040 (실패) | 1.499555 | 0.604340 | 100/100 |
| Taxi | 1.366430246 | 1.473390627 | 1.445715677 | -0.027674950 | +0.079285431 | +0.069285431 (실패) | 88.194997 | 28.674020 | 3/23 |
| Instacart | 3.206440011 | 3.215522896 | 3.211919485 | -0.003603410 | +0.005479474 | -0.004520526 (통과) | 5.872217 | 3.993781 | 5/25 |

- 세 데이터셋 모두 `ΔB < 0`이다. B에서 나빠진 A 대비 Time NLL 차이의 회복률은 Intermittent `49.5%`, Taxi `25.9%`, Instacart `39.7%`다.
- 선택 체크포인트와 B 체크포인트 사이에서 바뀐 state key는 세 time-head tensor뿐이다. 나머지 `32`개 tensor는 로컬에서 비트 단위 동일성을 재확인했다.
- 원격 full-cache replay에서 수량 예측 digest가 source와 selected checkpoint 사이에 동일했다. Raw RMSE, 전체 MAE, body MAE, `>p99` MAE도 B 기준과 계약 허용 오차 안에서 일치한다.

## 수렴 해석

- Taxi는 epoch 3, Instacart는 epoch 5가 최적이며 이후 20 epoch 동안 개선되지 않아 고정 patience에 따라 종료했다. 두 데이터셋에는 같은 설정으로 더 연장할 근거가 없다.
- Intermittent는 epoch 100이 최적이자 마지막 epoch이고 학습 구간 전체에서 validation Time NLL이 계속 낮아졌다. 이번 계약의 100-epoch 상한에서 미수렴했지만, 이 결과를 본 뒤 상한을 바꾸는 것은 동일 실험으로 취급할 수 없다.

## 감사 결과와 한계

- 로컬 감사 `105/105`개 계약·무결성 검사가 모두 통과했다. B/selected/last checkpoint의 파일 및 canonical state SHA-256, 변경 tensor 경계, optimizer step, 연속 history, 가장 이른 최솟값, e1 CUDA 실행, source revision과 controller decision을 대조했다.
- 선택 체크포인트의 CPU replay와 기록된 Time NLL 차이는 모든 데이터셋에서 `1e-6` 이하다.
- 로컬 증적 복사본은 대용량 feature cache를 의도적으로 제외했다. 따라서 이 복사본만으로 예측 replay를 즉시 반복할 수는 없다. 원 데이터, B checkpoint, 고정 source가 있어 cache를 재생성하면 반복 가능하다. 원격 controller가 수행한 bitwise quantity replay와 cache digest는 summary/status에 남아 있다.
- 이 실험은 수량 성능을 보존하는 사후 time-head calibration의 효과를 보여준다. Backbone 구조 변경이나 모든 데이터셋에서 A 수준을 회복한다는 근거로 사용할 수 없다.

기계 판독용 상세 검증은 [validation_audit.json](validation_audit.json), 표 원자료는 [metrics.csv](metrics.csv), 입력·출력 파일 SHA-256은 [artifact_manifest.json](artifact_manifest.json)에 있다.
