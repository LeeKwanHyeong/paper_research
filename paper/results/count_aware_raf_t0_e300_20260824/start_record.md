# RAF Spare Parts Count-aware T0 3-seed e300 시작 기록

- 상태: 완료
- 실험 시작 시각: 2026-08-24 15:11:42 KST
- 실험 종료 시각: 2026-08-24 16:03:31 KST
- 실행 서버: 5080
- tmux session: `raf_t0_e300_0824`
- 실행 source revision: `c7d362e206d8e5d78d96917f03b17b9ad0f50374`
- 목적: native monthly spare-parts demand에서 Count-aware TitanTPP와 공통 T0 backbone을 동일한 validation 계약으로 비교한다.
- Factorial: 5 backbones x 3 seeds, 총 15 runs
- 학습: 최대 e300, 최소 e40, patience 40, batch 128, learning rate 0.001
- 문맥: lookback 84 months, max sequence length 84
- 선택: minimum validation joint objective
- 평가: validation-only, held-out test 잠금
- artifact: `search_artifacts/count_aware_raf_t0_e300_20260824`
- Notion: [2026-08-24 RAF Spare Parts Count-aware T0 3-Seed e300 Validation](https://app.notion.com/p/3c6bbe4056138118a9bdf0107b9c21f2)

## 실행 명령어

```bash
SOURCE_REVISION=c7d362e206d8e5d78d96917f03b17b9ad0f50374 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_raf_t0_e300_20260824 \
bash simple_lab_test/search/scripts/run_count_aware_raf_t0_e300_20260824.sh
```

## 초기 확인

- 로컬 및 5080 계약 테스트: `14 passed`
- CUDA: PyTorch 2.11.0+cu130, NVIDIA GeForce RTX 5080
- 계약·runner·RAF parquet·split manifest checksum 일치
- 첫 학습 진입: RMTPP seed 42 epoch 1 이상, finite loss 확인
- held-out test: 사용하지 않음

## 결과

- 15개 run 모두 early stopping으로 정상 종료했다.
- source revision, fixed split, T0 역할, validation-only와 held-out test 잠금 계약이 일치했다.
- Count-aware TitanTPP는 validation joint objective와 Time NLL 평균이 가장 낮았다.
- Quantity MAE는 THP가 가장 낮고, Quantity RMSE는 TitanTPP가 가장 낮았지만 두 차이는 각각 `0.089%`, `0.052%`로 매우 작고 seed별 방향도 일관되지 않았다.
- RAF validation history는 전부 64 events 이하라 long-history 우위 근거로 사용할 수 없다.
- 상세 분석: `paper/results/count_aware_raf_t0_e300_20260824/analysis.md`
