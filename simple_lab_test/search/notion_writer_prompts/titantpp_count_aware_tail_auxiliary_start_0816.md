# Notion Update Source: Count-aware Log-MSE + Tail-aware Auxiliary

작성 위치: `5. Model Design Enhancement`

## 2026-08-16

### Count-aware Log-MSE + Tail-aware Auxiliary Validation

#### 상태

5080 실행은 T0 epoch 100에서 중단했으며 partial artifact는 보존했다. 현재는 5090
tmux `inter_tail_aux_e300_5090_0816`에서 Intermittent seed-42 e300 validation-only
screening을 T0부터 fresh rerun 중이다. Held-out test와 multi-seed는 잠근다.

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
| Lambda | 0.09111380335463036 (train-only gradient calibration) |
| Model | Count-aware TitanTPP small LMM |
| Epoch / seed | 300 / 42 |
| Batch / LR | 128 / 1e-3 |
| Context | lookback 520, max sequence length 256 |
| Selection | best validation joint objective |
| Evaluation | validation only; held-out test 미사용 |

#### 실행 명령어

```bash
SOURCE_REVISION=7de638a5c9290f79dae02a40fd22839aba9802e7 \
LAMBDA_TAIL=0.09111380335463036 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
EXECUTION_ROLE=primary_5090 \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816_5090_rerun \
bash simple_lab_test/search/scripts/run_count_aware_tail_screening_e300_20260816.sh
```

#### 결과

- 5080 최초 실행: `2026-08-16 06:51:10 KST`, T0 epoch 100에서 중단
- 5090 재시작: `2026-08-16 09:40:47 KST`
- Artifact: `search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816_5090_rerun`
- Launch contract: `running`, validation-only, held-out test 미사용
- Source와 contract checksum 일치, Python compile과 CUDA process 진입 확인
- 이후 상태는 1시간 heartbeat에서 단회 확인한다.
