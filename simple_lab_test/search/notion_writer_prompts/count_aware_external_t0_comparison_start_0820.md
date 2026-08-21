# Notion Start Source: Count-aware External T0 NHP/SAHP Three-seed Comparison

- 위치: `5. Model Design Enhancement`
- 상태: 실험 중
- 실험 시작 시각: 2026-08-20 20:18:24 KST
- 실행 서버 / tmux: 5080 1개 shard, 5090 2개 shard

## 목적

공통 T0 head/loss 아래에서 RMTPP·THP·NHP·SAHP encoder를 비교한다. 기존 계약이
일치하는 RMTPP·THP 3-seed artifact는 재사용한다. NHP 42·52는 5080, NHP 62와
SAHP 42·52·62는 5090에서 분할 실행한다.

## Variant 계약

| 모델 | 달라지는 encoder | 공통 조건 |
| --- | --- | --- |
| Adapted RMTPP | GRU | direct log-MSE + legacy RMTPP time head |
| Adapted THP | causal Transformer | 동일 |
| Adapted NHP | continuous-time LSTM | 동일 |
| Adapted SAHP | causal attention + continuous decay | 동일 |

## 고정 조건

- Intermittent fixed split, validation-only
- maximum/minimum epochs: 300/40, patience 40
- seeds: 42, 52, 62
- batch size 128, learning rate 0.001
- lookback 520 weeks, max sequence length 256
- minimum validation joint objective checkpoint
- artifact: `search_artifacts/count_aware_external_t0_nhp_sahp_e300_20260820`

## 실행 명령어

`simple_lab_test/search/scripts/run_count_aware_external_t0_nhp_sahp_e300_20260820.sh`

## 결과

실험 완료 후 작성한다.
