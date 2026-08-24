# Notion source draft: Instacart Count-aware T0 3-Seed e300 Validation

위치: `5. Model Design Enhancement`

페이지 제목: `2026-08-24 Instacart Count-aware T0 3-Seed e300 Validation`

## 상태

- 실험 중
- 실험 시작 시각: 실행 직전 KST 시각으로 갱신
- 실행 서버 / tmux: 5080 / `instacart_t0_e300_0824`

## 목적

- Instacart basket count에서 Count-aware TitanTPP의 backbone 효과를 확인한다.
- T1 tail-aware loss는 제외하고 동일한 T0 direct log-MSE 조건만 비교한다.

## Factorial 계약

| 역할 | Backbone | Quantity objective | Runs |
| --- | --- | --- | ---: |
| T0 common control | RMTPP, THP, NHP, SAHP, TitanTPP | direct log1p quantity MSE | 5 x 3 seeds |

## 고정 조건

- dataset: Instacart fixed split
- epochs / seeds: 최대 300, 최소 40, patience 40 / 42, 52, 62
- learning rate / batch size: 0.001 / 128
- lookback / max sequence: 52 days / 64 events
- hidden dimension / gradient clipping: 64 / 1.0
- checkpoint: minimum validation joint objective
- evaluation: validation-only, held-out test 잠금
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
