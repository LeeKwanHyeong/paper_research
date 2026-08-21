# Count-aware External T0 NHP/SAHP Three-seed Comparison 시작 기록

## 상태

- 실험 중
- 최초 실험 시작 시각: 2026-08-20 20:18:24 KST
- 분할 실행 시작 시각: 2026-08-21 21:12 KST
- 실행 서버 / tmux:
  - 5080 / `external_t0_nhp_42_52_e300_0821`
  - 5090 / `external_t0_nhp_62_e300_0821`
  - 5090 / `external_t0_sahp_all_e300_0821`
- source revision: `b6831d30a60b83677cc438b3a560217bf343c75c`

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

5080은 기존 `last_epoch_state.pt`를 보존한 artifact에서 NHP seed 42를 재개하고,
완료 후 seed 52를 실행한다.

```bash
SOURCE_REVISION=b6831d30a60b83677cc438b3a560217bf343c75c \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_external_t0_nhp_sahp_e300_20260820 \
EXECUTION_ROLE=primary_5080_nhp_42_52 BACKBONES=nhp SEEDS=42,52 \
bash simple_lab_test/search/scripts/run_count_aware_external_t0_shard_e300_20260821.sh
```

5090은 두 tmux 세션을 동시에 실행한다.

```bash
SOURCE_REVISION=b6831d30a60b83677cc438b3a560217bf343c75c \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_external_t0_nhp_seed62_e300_20260821_5090 \
EXECUTION_ROLE=primary_5090_nhp_62 BACKBONES=nhp SEEDS=62 \
bash simple_lab_test/search/scripts/run_count_aware_external_t0_shard_e300_20260821.sh

SOURCE_REVISION=b6831d30a60b83677cc438b3a560217bf343c75c \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_external_t0_sahp_all_e300_20260821_5090 \
EXECUTION_ROLE=primary_5090_sahp_all BACKBONES=sahp SEEDS=42,52,62 \
bash simple_lab_test/search/scripts/run_count_aware_external_t0_shard_e300_20260821.sh
```

## 실행 전 확인

- 두 서버의 모델 및 runner checksum 일치
- data SHA-256: `85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f`
- split manifest SHA-256: `393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04`
- 5090 CUDA tensor 연산 확인
- 5090 NHP·SAHP 계약 테스트: 17 passed
- held-out test: 사용하지 않음

## 결과
