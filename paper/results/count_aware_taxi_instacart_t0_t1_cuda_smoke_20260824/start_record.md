# Taxi·Instacart mark-free T0·T1 CUDA smoke 시작 기록

- 상태: 완료 (`PASS`)
- 기록 시각: 2026-08-24 09:59:06 KST
- 실험 시작 시각: 2026-08-24 10:05:11 KST
- 실험 종료 확인 시각: 2026-08-24 10:07:28 KST
- 실행 서버: 5080
- tmux session: `count_taxi_insta_smoke_0824`
- 목적: Taxi와 Instacart의 mark-free T0·TitanTPP-T1 e300 계약을 고정하고 CUDA 및 실제 데이터 e1 경로를 확인한다.
- 범위: seed 42, e1, 상위 20 series, train/validation 최대 2 batch
- 평가: validation-only, held-out test 잠금
- artifact: `search_artifacts/count_aware_taxi_instacart_t0_t1_cuda_smoke_20260824`
- source revision: `e8b8604f26372c440d139b03939c2bb34048c1e5`
- Notion: https://app.notion.com/p/3c6bbe405613814fb334dfce2535f92d

## 결과

- Focused contract tests: 28 passed
- CUDA model-test: 6/6 finite
- Taxi actual-data e1: T0 5/5, T1 1/1 success
- Instacart actual-data e1: T0 5/5, T1 1/1 success
- History artifacts: 12/12, each one epoch
- Held-out test artifact: 없음
- Scale-wise metrics와 plots: partial validation batch smoke이므로 생성하지 않음
- 최종 판정: 정식 e300 runner 실행 경로 사용 가능

## 실행 명령어

```bash
SOURCE_REVISION=e8b8604f26372c440d139b03939c2bb34048c1e5 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash simple_lab_test/search/scripts/run_count_aware_taxi_instacart_cuda_smoke_20260824.sh
```
