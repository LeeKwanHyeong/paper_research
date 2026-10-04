# 폭8·12 조건별 원본 검증과 Validation 비교

생성 시각: 2026-10-04 16:14:57 KST. 실제 서버 관측 시각은 각 snapshot의 observation 기록을 따릅니다.

신규 18조건 중 원본 검증 완료 **0/18**입니다. 폭16은 사용자가 고정한 기준선, 폭4는 기존 대조군입니다. Instacart는 후속 실험입니다.

기존 폭4·16의 Validation 원본 연결 18개는 endpoint/history SHA와 지표·선택 epoch를 재확인했습니다. 신규 폭8·12는 회수 완료된 원본에만 terminal success·전체 파일 SHA·최초 최소 RMSE 선택 epoch·selected/last replay 검증을 적용합니다. checkpoint binary 회수 SHA 확인과 CPU 재추론 감사는 구분합니다.

| 데이터 | seed | 폭 | 상태 | 선택 epoch |
|---|---:|---:|---|---:|
| yellow_trip_hourly | 42 | 8 | training_confirmed_at_observation | — |
| yellow_trip_hourly | 42 | 12 | waiting_at_last_observation | — |
| yellow_trip_hourly | 52 | 8 | waiting_at_last_observation | — |
| yellow_trip_hourly | 52 | 12 | waiting_at_last_observation | — |
| yellow_trip_hourly | 62 | 8 | waiting_at_last_observation | — |
| yellow_trip_hourly | 62 | 12 | waiting_at_last_observation | — |
| raf_spare_parts | 42 | 8 | waiting_at_last_observation | — |
| raf_spare_parts | 42 | 12 | waiting_at_last_observation | — |
| raf_spare_parts | 52 | 8 | waiting_at_last_observation | — |
| raf_spare_parts | 52 | 12 | waiting_at_last_observation | — |
| raf_spare_parts | 62 | 8 | waiting_at_last_observation | — |
| raf_spare_parts | 62 | 12 | waiting_at_last_observation | — |
| intermittent_frozen_5000 | 42 | 8 | training_confirmed_at_observation | — |
| intermittent_frozen_5000 | 42 | 12 | waiting_at_last_observation | — |
| intermittent_frozen_5000 | 52 | 8 | waiting_at_last_observation | — |
| intermittent_frozen_5000 | 52 | 12 | waiting_at_last_observation | — |
| intermittent_frozen_5000 | 62 | 8 | waiting_at_last_observation | — |
| intermittent_frozen_5000 | 62 | 12 | waiting_at_last_observation | — |

전체 Validation은 같은 모집단의 원래 선택 checkpoint에서 읽었습니다. 큰 수량은 원래 TRAIN 경계 초과(택시3449, Intermittent187, RAF200)입니다. 같은 seed 비교가 paired_validation_deltas.csv에 보존됩니다.

Train은 고정 수량 구간별 표본과 N/n 가중치의 진단입니다. 폭4와 표본 동일성은 fit 자체의 기록으로 확인합니다. 폭16 시간 진단의 다른 표본과는 Train 결과를 섞지 않습니다.

3seed 모두 검증되기 전에는 평균·표본 표준편차나 폭 채택 결론을 만들지 않습니다. 완료 그룹에는 사전에 고정한 전체 RMSE·seed 다수 개선 기준과 MAE·큰 수량·시간 손해를 함께 적용합니다.

재생성 명령(로컬 회수 원본만 사용, 네트워크·학습·Test 열람 없음):

```sh
/usr/local/bin/python3 paper/scripts/compare_titantpp_history_capacity.py
```

증적: [이번 snapshot](snapshots/20261004T071457224183Z/summary.json), [Validation 지표](snapshots/20261004T071457224183Z/full_validation_metrics.csv), [원본 연결](snapshots/20261004T071457224183Z/source_bindings.json), [18조건 목록](snapshots/20261004T071457224183Z/condition_status.csv).

**현재 결과 연결 — 진행 중**
- 회수·검증된 조건만 비교에 포함하고, 나머지 학습·미회수 상태를 남깁니다.

**3seed 용량 효과 판단 — 다음 작업**
- 각 데이터에서 폭8·12의 세 seed 원본 검증 후 폭4·16과 평균·표본 표준편차 및 불리한 seed를 비교합니다.
