# Notion source draft: RAF Spare Parts Count-aware T0 3-Seed e300 Validation

위치: `5. Model Design Enhancement`

페이지 제목: `2026-08-24 RAF Spare Parts Count-aware T0 3-Seed e300 Validation`

## 상태

- 실험 중
- 실험 시작 시각: 2026-08-24 15:11:42 KST
- 실행 서버 / tmux: 5080 / `raf_t0_e300_0824`

## 목적

- 실제 월별 부품 수요량을 사용하는 RAF에서 Count-aware TitanTPP의 backbone 효과를 확인한다.
- T1 tail loss는 제외하고 동일한 T0 direct log-MSE 조건만 비교한다.

## Factorial 계약

| 역할 | Backbone | Quantity objective | Runs |
| --- | --- | --- | ---: |
| T0 common control | RMTPP, THP, NHP, SAHP, TitanTPP | direct log1p quantity MSE | 5 x 3 seeds |

모든 backbone은 같은 mark-free 입력, quantity head, legacy time head와 validation checkpoint selection을 사용한다.

## 고정 조건

- dataset: RAF Spare Parts fixed split
- epochs / seeds: 최대 300, 최소 40, patience 40 / 42, 52, 62
- learning rate / batch size: 0.001 / 128
- lookback / max sequence: 84 months / 84 events
- hidden dimension / gradient clipping: 64 / 1.0
- checkpoint: minimum validation joint objective
- evaluation: validation-only, held-out test 잠금
- artifact: `search_artifacts/count_aware_raf_t0_e300_20260824`

## 실행 명령어

```bash
SOURCE_REVISION=c7d362e206d8e5d78d96917f03b17b9ad0f50374 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_raf_t0_e300_20260824 \
bash simple_lab_test/search/scripts/run_count_aware_raf_t0_e300_20260824.sh
```

## 결과
