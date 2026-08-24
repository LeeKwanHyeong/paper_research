# 2026-08-24 Online Retail II Train-Only Time-Scale and Gradient Clipping Audit

작성 위치: `5. Model Design Enhancement`

## 상태

- 준비 완료, 실행 전
- 실행 서버 / tmux: 5080 / `online_retail_time_audit_5080_0824`

## 목적

- Online Retail II에서 매우 큰 Time NLL과 100% gradient clipping이 hourly delta-time과 legacy time head의 단위 부적합 때문에 발생하는지 확인한다.
- 모델과 quantity loss는 바꾸지 않고 train-only 시간 단위만 비교해 데이터셋을 후속 비교에 유지할 수 있는지 판단한다.

## Variant 계약

| Variant | 시간 입력 |
| --- | --- |
| S0 | hour 원본 |
| S1 | 24시간을 1단위로 변환 |
| S2 | train target delta-time 중앙값으로 나눔 |
| S3 | train target delta-time 평균으로 나눔 |
| S4 | train target delta-time p95로 나누는 상한 진단군 |

모든 Variant는 같은 TitanTPP-T0 초기화, train batch, legacy time head와 direct log-MSE quantity loss를 사용한다. `seq`와 lookback은 원래 hourly 좌표를 유지한다.

## 고정 조건

- dataset: Online Retail II train split
- model: TitanTPP-T0 Hard-LMM
- epochs / seeds: 3 / 42
- lr / batch size: 0.001 / 128
- lookback / max sequence length: 8,760시간 / 256
- split mode: train-only, validation/test 미사용
- 주요 model/loss 옵션: `legacy_clamped_rmtpp`, direct log1p-quantity MSE, gradient clipping 1.0, epoch당 최대 16 batch
- artifact: `search_artifacts/online_retail_time_scale_gradient_audit_20260824`

## 실행 명령어

```bash
ssh 5080 '/usr/bin/tmux new-session -d -s online_retail_time_audit_5080_0824 "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python SOURCE_REVISION=<checksum_synced_commit> bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_online_retail_time_scale_gradient_audit_20260824.sh"'
```

## 결과
