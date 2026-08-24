# Instacart Count-aware T0 3-seed e300 시작 기록

- 상태: 실행 준비
- 실행 서버: 5080
- tmux session: `instacart_t0_e300_0824`
- 목적: Instacart basket count에서 Count-aware TitanTPP와 네 T0 backbone을 동일한 validation 계약으로 비교한다.
- Factorial: 5 backbones x 3 seeds, 총 15 runs
- 학습: 최대 e300, 최소 e40, patience 40, batch 128, learning rate 0.001
- 문맥: lookback 52 days, max sequence length 64
- 선택: minimum validation joint objective
- 평가: validation-only, held-out test 잠금
- artifact: `search_artifacts/count_aware_instacart_t0_e300_20260824`

## 실행 명령어

```bash
SOURCE_REVISION=<checksum-synced-full-sha> \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_instacart_t0_e300_20260824 \
bash simple_lab_test/search/scripts/run_count_aware_instacart_t0_e300_20260824.sh
```

## 결과
