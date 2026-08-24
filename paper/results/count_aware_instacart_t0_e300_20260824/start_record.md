# Instacart Count-aware T0 3-seed e300 시작 기록

- 상태: 실험 중
- 실험 시작 시각: 2026-08-24 16:42:01 KST
- 실행 서버: 5080
- tmux session: `instacart_t0_e300_0824`
- 실행 source revision: `28293c43521615be2ed8fad5b043dc9df8e5e457`
- 목적: Instacart basket count에서 Count-aware TitanTPP와 네 T0 backbone을 동일한 validation 계약으로 비교한다.
- Factorial: 5 backbones x 3 seeds, 총 15 runs
- 학습: 최대 e300, 최소 e40, patience 40, batch 128, learning rate 0.001
- 문맥: lookback 52 days, max sequence length 64
- 선택: minimum validation joint objective
- 평가: validation-only, held-out test 잠금
- artifact: `search_artifacts/count_aware_instacart_t0_e300_20260824`
- 예상 완료: 2026-08-25 22:00-2026-08-26 13:00 KST
- Notion: [2026-08-24 Instacart Count-aware T0 3-Seed e300 Validation](https://app.notion.com/p/3c6bbe40561381f18b0de47ac097f40d)

## 실행 명령어

```bash
SOURCE_REVISION=28293c43521615be2ed8fad5b043dc9df8e5e457 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_instacart_t0_e300_20260824 \
bash simple_lab_test/search/scripts/run_count_aware_instacart_t0_e300_20260824.sh
```

## 초기 확인

- 로컬 및 5080 계약 테스트: `15 passed`
- CUDA: NVIDIA GeForce RTX 5080
- 계약·runner·Instacart parquet·split manifest checksum 일치
- 첫 학습 진입: RMTPP seed 42 epoch 1 이상, finite loss 확인
- 초기 속도: 약 50초/epoch
- held-out test: 사용하지 않음

## 결과
