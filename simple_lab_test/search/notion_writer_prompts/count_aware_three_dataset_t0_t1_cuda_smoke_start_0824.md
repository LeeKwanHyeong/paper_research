# 2026-08-24 Count-Aware Three-Dataset T0-T1 Matched CUDA Smoke

작성 위치: `5. Model Design Enhancement`

## 상태

- 준비 중
- 실험 시작 예정 시각: 2026-08-24 08:54:28 KST
- 실행 서버 / tmux: 5080 / `count_three_dataset_t0_t1_e1_5080_0824`

## 목적

- Intermittent v2, Online Retail II, RAF Spare Parts에서 공통 T0와 현재 TitanTPP-T1을 같은 validation 계약으로 실행할 수 있는지 확인한다.
- 이번 단계는 성능 결론이 아니라 CUDA forward/backward, 실제 데이터 loader, checkpoint와 validation artifact 생성 여부를 확인하는 e1 smoke다.
- 모델 encoder, time head와 quantity loss 식은 변경하지 않는다.

## Factorial 계약

| 역할 | Backbone | Quantity objective |
| --- | --- | --- |
| T0 common control | RMTPP, THP, NHP, SAHP, TitanTPP | direct log1p-quantity MSE |
| TitanTPP-T1 | TitanTPP Hard-LMM | T0 + train-only tail raw-quantity Huber |

모든 역할은 같은 `legacy_clamped_rmtpp` time head와 데이터셋별 fixed split을 사용한다. T1의 lambda와 gradient route는 공통으로 유지하고 tail threshold, normalization과 cap만 각 train split의 p95, p95, p99로 고정한다.

## 고정 조건

- dataset: Intermittent v2, Online Retail II, RAF Spare Parts
- model: RMTPP, THP, NHP, SAHP, TitanTPP-T0, TitanTPP-T1
- epochs / seeds: 1 / 42
- lr / batch size: 0.001 / 128
- lookback / max sequence length: Intermittent 520주/256, Online Retail II 8,760시간/256, RAF 84개월/84
- split mode: fixed validation-only
- 주요 model/loss 옵션: hidden 64, gradient clipping 1.0, T1 lambda 0.09111380335463036, train·validation 각 최대 2 batch
- artifact: `search_artifacts/count_aware_three_dataset_t0_t1_cuda_smoke_20260824`

## 실행 명령어

```bash
ssh 5080 '/usr/bin/tmux new-session -d -s count_three_dataset_t0_t1_e1_5080_0824 "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python SOURCE_REVISION=<checksum_synced_commit> bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_count_aware_three_dataset_cuda_smoke_20260824.sh"'
```

## 결과

