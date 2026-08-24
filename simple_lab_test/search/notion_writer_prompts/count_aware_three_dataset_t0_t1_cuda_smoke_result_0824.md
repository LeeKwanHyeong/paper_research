# 2026-08-24 Count-Aware Three-Dataset T0-T1 Matched CUDA Smoke 결과

대상: `5. Model Design Enhancement > 2026-08-24 > Step 4`

## 상태

- 완료, runtime gate PASS
- 완료 시각: 2026-08-24 08:59:04 KST
- 실행 서버 / tmux: 5080 / `count_three_dataset_t0_t1_e1_5080_0824`

## 결과

- focused contract test: 26 passed
- CUDA model-test: 6/6 finite
- actual-data e1: 18/18 success
- checkpoint/history/summary: 모든 run 생성
- Traceback/NaN/Infinity: 없음
- held-out test: 사용하지 않음
- scale-wise metrics와 plot: partial e1 smoke이므로 생성하지 않음

세 데이터셋 모두 T0와 TitanTPP-T1 실행 경로는 통과했다. e1 수치는 성능 순위로
해석하지 않는다. Online Retail II는 legacy time head에서 Time NLL이 매우 크고
gradient clipping 비율이 100%였으므로, 정식 e300 전에 train-only time-scale 감사를
먼저 수행한다. 이 경고는 tail loss보다 hourly delta-time과 기존 time head의 단위
부적합 가능성에 가깝다.
