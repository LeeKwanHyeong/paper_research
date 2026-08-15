# Notion Update Source: Count-aware Log-MSE + Tail-aware Auxiliary

작성 위치: `5. Model Design Enhancement`

## 2026-08-16

### Count-aware Log-MSE + Tail-aware Auxiliary Validation

#### 상태

5080에서 Intermittent seed-42 e300 validation-only screening을 시작한다. Held-out
test와 multi-seed는 잠근다.

#### 목적

K=1 log-normal head에서 중·저수량 MAE는 일부 개선됐지만 RMSE, 상위 1% MAE,
time NLL이 악화됐다. 기존 log-MSE를 유지하면서 train p95 초과 수량에만 bounded raw
Huber를 추가했을 때 tail 오차를 줄이면서 body와 time modeling을 보존할 수 있는지
확인한다.

#### Variant 계약

| Variant | 차이 | 역할 |
| --- | --- | --- |
| T0-logMSE | 기존 log1p quantity MSE | fresh matched 기준선 |
| T1-tail-shared | Tail loss가 quantity head와 Titan encoder에 전달 | shared-gradient 대조군 |
| T2-tail-head-only | Tail loss가 quantity head에만 전달 | 주 후보 |

세 Variant는 같은 encoder, time head, quantity point prediction과 parameter count를
사용한다.

#### 고정 조건

| 항목 | 값 |
| --- | --- |
| Dataset | Intermittent frozen-5000 fixed split |
| Train-only p90 / p95 / p99 | 31 / 46 / 187 |
| Tail | `q > 46` |
| Raw normalization / cap | 46 / 187 |
| Huber delta | 1 |
| Lambda | train-only gradient calibration 결과 사용 |
| Model | Count-aware TitanTPP small LMM |
| Epoch / seed | 300 / 42 |
| Batch / LR | 128 / 1e-3 |
| Context | lookback 520, max sequence length 256 |
| Selection | best validation joint objective |
| Evaluation | validation only; held-out test 미사용 |

#### 실행 명령어

```bash
SOURCE_REVISION=<synced_commit_sha> \
LAMBDA_TAIL=<train_only_frozen_value> \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash simple_lab_test/search/scripts/run_count_aware_tail_screening_e300_20260816.sh
```

#### 결과

