# TitanTPP V4 Taxi Train-Only Mark And Delta-Time Audit

Notion의 `5. Model Design Enhancement` 아래에서 제목 2
`2026-07-16 | V4 Mark-Conditioned Time Head`, 제목 3
`Step 1. Taxi Train-Only Mark And Delta-Time Audit`로 정리한다.

## 상태

- 상태: `실험 중`
- 실험 시작 시각: `2026-07-16 12:36:00 KST`
- 실행 서버 / tmux: `5090 / titantpp_v4_taxi_time_audit_0716`

## 목적

- Taxi의 next mark에 따라 delta-time 분포가 실질적으로 달라지는지 train target만으로 확인한다.
- V4 mark-conditioned time head를 구현할 근거가 충분한지 판단한다.
- 이 단계에서는 모델 성능을 비교하거나 validation/test target을 읽지 않는다.

## Variant 계약

| 진단 모델 | Time density | 역할 |
| --- | --- | --- |
| Global shared | 모든 mark가 하나의 RMTPP intercept 공유 | V2 time-head 진단 control |
| Mark-conditioned | mark별 intercept, positive slope는 공유 | V4가 제거하려는 조건부 독립 가정 진단 |

두 진단 모델은 train target 내부 시간순 `80/20` fit/eval, 동일 `w` 탐색 범위와
RMTPP density 식을 사용한다. 진단 parameter는 V4 초기값으로 전이하지 않는다.

## 고정 조건

- dataset: `yellow_trip_hourly_train.parquet`만 사용
- target: fixed-split train next event, 시계열별 첫 이벤트 제외
- lookback / max sequence: `168 / 256`
- fit/eval: 각 시계열 train target의 앞 `80%` / 뒤 `20%`
- primary gate: mark-conditioned eval NLL `0.5%` 이상 개선
- distribution gate: `log1p(delta-time)` eta-squared `0.01` 이상
- robustness: series bootstrap 95% 하한 `>0`, 개선 series `55%` 이상
- artifact: `search_artifacts/model_enhancement_titantpp_v4_taxi_train_time_audit_0716`

## 실행 명령어

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d -s titantpp_v4_taxi_time_audit_0716 "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python TMUX_SESSION=titantpp_v4_taxi_time_audit_0716 bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_v4_taxi_train_time_audit_0716.sh"'
```

## 결과
