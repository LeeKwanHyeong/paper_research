# Count-aware External T0 NHP/SAHP Three-seed Comparison 시작 기록

## 상태

- 실험 중
- 실험 시작 시각: 2026-08-20 20:18:24 KST
- 실행 서버 / tmux:
  - 5080 / `external_t0_nhp_42_52_e300_0821`
  - 5090 / `external_t0_nhp_62_e300_0821`
  - 5090 / `external_t0_sahp_all_e300_0821`
- source revision: `8ec1f42fe01f9e436296914cb4d8d2a950528732`

## 목적

기존 RMTPP·THP T0 3-seed 결과에 Adapted NHP·SAHP를 같은 계약으로 추가한다.
새 두 모델만 학습하고 계약이 일치하는 기존 RMTPP·THP artifact는 재사용한다.

## Variant 계약

| 모델 | Encoder | 공통 head/loss |
| --- | --- | --- |
| Adapted RMTPP | GRU | T0 direct log-MSE + legacy RMTPP time head |
| Adapted THP | causal Transformer | T0 direct log-MSE + legacy RMTPP time head |
| Adapted NHP | continuous-time LSTM | T0 direct log-MSE + legacy RMTPP time head |
| Adapted SAHP | causal attention + continuous decay | T0 direct log-MSE + legacy RMTPP time head |

Encoder 이외의 데이터, head, loss, optimizer, checkpoint selection은 동일하게 유지한다.

## 고정 조건

- dataset: `intermittent_frozen_5000`
- 5080 신규 실행: Adapted NHP seeds 42, 52
- 5090 신규 실행: Adapted NHP seed 62, Adapted SAHP seeds 42, 52, 62
- epochs / seeds: maximum 300 / 42, 52, 62
- minimum epochs / patience: 40 / 40
- lr / batch size: 0.001 / 128
- lookback / max sequence length: 520 weeks / 256
- split: fixed, validation-only
- checkpoint: minimum validation joint objective
- held-out test: 사용하지 않음
- artifacts:
  - `search_artifacts/count_aware_external_t0_nhp_sahp_e300_20260820`
  - `search_artifacts/count_aware_external_t0_nhp_seed62_e300_20260821_5090`
  - `search_artifacts/count_aware_external_t0_sahp_all_e300_20260821_5090`

## 실행 명령어

```bash
SOURCE_REVISION=8ec1f42fe01f9e436296914cb4d8d2a950528732 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash simple_lab_test/search/scripts/run_count_aware_external_t0_shard_e300_20260821.sh
```

## 결과
