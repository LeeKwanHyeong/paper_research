# Taxi·Instacart mark-free T0·T1 CUDA smoke 시작 기록

- 상태: 준비 완료
- 기록 시각: 2026-08-24 09:59:06 KST
- 실행 서버: 5080
- tmux session: `count_taxi_insta_smoke_0824`
- 목적: Taxi와 Instacart의 mark-free T0·TitanTPP-T1 e300 계약을 고정하고 CUDA 및 실제 데이터 e1 경로를 확인한다.
- 범위: seed 42, e1, 상위 20 series, train/validation 최대 2 batch
- 평가: validation-only, held-out test 잠금
- artifact: `search_artifacts/count_aware_taxi_instacart_t0_t1_cuda_smoke_20260824`
- Notion: https://app.notion.com/p/3c6bbe405613814fb334dfce2535f92d

## 실행 명령어

```bash
SOURCE_REVISION=<checksum-synced-full-sha> \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash simple_lab_test/search/scripts/run_count_aware_taxi_instacart_cuda_smoke_20260824.sh
```
