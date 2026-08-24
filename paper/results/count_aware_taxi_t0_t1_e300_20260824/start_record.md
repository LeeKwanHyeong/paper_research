# Taxi mark-free T0·TitanTPP-T1 3-seed e300 시작 기록

- 상태: 실행 준비 완료
- 기록 시각: 2026-08-24 10:11:35 KST
- 실행 서버: 5080
- tmux session: `taxi_markfree_t0_t1_e300_0824`
- 목적: Taxi fixed split에서 mark-free T0 backbone 효과와 TitanTPP-T1 tail-aware objective 효과를 동일한 validation 계약으로 비교한다.
- Factorial: T0 5 backbones x 3 seeds, T1 TitanTPP x 3 seeds, 총 18 runs
- 학습: 최대 e300, 최소 e40, patience 40, batch 128, learning rate 0.001
- 문맥: lookback 168 hours, max sequence length 256
- 선택: minimum validation joint objective
- 평가: validation-only, held-out test 잠금
- artifact: `search_artifacts/count_aware_taxi_t0_t1_e300_20260824`

## 실행 명령어

```bash
SOURCE_REVISION=<checksum-synced-full-sha> \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_taxi_t0_t1_e300_20260824 \
DATASET_FILTER=yellow_trip_hourly \
ROLE_FILTER=all \
bash simple_lab_test/search/scripts/run_count_aware_taxi_instacart_e300_20260824.sh
```

## 결과

