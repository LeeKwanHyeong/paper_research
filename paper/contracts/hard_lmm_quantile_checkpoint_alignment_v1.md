# Hard-LMM quantile-adaptive loss와 raw-RMSE checkpoint 정렬 계약 v1

- 동결일: 2026-09-06 KST
- 대상: `paper_research / codex/quantile-checkpoint-alignment`
- 시작 revision: `815ec62ebd945d286aa18149be6231454d4cdb58`
- 개발 평가: validation-only, seed42
- held-out test: 잠금 유지

## 질문과 세 arm

공통 Hard-LMM T0 구조에서 checkpoint 선택 효과와 loss 효과를 분리한다.

1. **A — T0 + joint:** 기존 log-MSE 학습과 validation joint objective 선택 결과를 동결 reference로 재사용한다.
2. **B — T0 + raw RMSE:** 같은 T0 loss를 새로 학습하고 validation raw quantity RMSE의 가장 이른 엄격한 최솟값을 선택한다. Early stopping도 raw RMSE를 사용한다.
3. **C — quantile-adaptive + raw RMSE:** B와 구조·초기화·batch 순서·optimizer를 맞추고 quantity loss만 아래 가중 log-MSE로 바꾼다.

기존 artifact에는 epoch별 metric은 있지만 raw-RMSE 최적 epoch의 model state가 대부분 없다. Post-hoc metric 재선택을 공식 결과로 사용하지 않고 B와 C를 fresh paired run으로 만든다.

A 세 artifact의 data·split checksum, lookback, sequence 길이를 B/C 계약과 대조한다. 세 A source revision과 B/C 시작 revision에서 target dataset/loader 두 파일의 SHA-256이 모두 같음도 동결했다. 따라서 historical A가 별도 target-ID digest를 기록하지 않았더라도 동일한 결정론적 validation target 집합을 사용했음을 감사할 수 있다.

## 단일 quantile-adaptive loss

Quantile은 canonical train loader가 실제 next-event target으로 사용하는 수량에서만 계산한다. 각 series의 첫 non-target event, validation, held-out test는 통계에 포함하지 않는다.

| Train-target 구간 | 원 가중치 |
| --- | ---: |
| `<=p50` | 1.0 |
| `p50-p90` | 1.0 |
| `p90-p95` | 1.5 |
| `p95-p99` | 2.0 |
| `>p99` | 3.0 |

각 데이터셋에서 위 원 가중치의 exact train-target 평균을 float64로 계산하고, 모든 원 가중치를 그 평균으로 나눈다. Batch별 재정규화는 하지 않는다.

```text
L_qty = mean(w_train_quantile(q) * (softplus(a) - log1p(q))^2)
```

Quantile→weight 함수, 네 quantile, 보간법, 정규화 규칙과 strength 1.0은 세 데이터셋에서 같다. 데이터셋별 수동 lambda·가중치·loss 분기는 없다. Strength 0은 원래 log-MSE를 직접 반환해 기존 T0 loss와 gradient가 일치해야 한다. 이전 T1의 p95 raw-Huber는 이 실험의 arm이 아니다.

## Checkpoint와 early stopping

B와 C는 `val_qty_rmse`의 가장 이른 엄격한 최솟값을 저장한다. 동률이면 먼저 나온 epoch를 유지한다. Minimum epoch 40, patience 40이며 early stopping도 같은 raw-RMSE monitor를 사용한다. Time loss, body MAE와 tail MAE는 checkpoint 선택에 사용하지 않고 최종 채택 guardrail로만 사용한다.

Resume에는 selector와 objective identity, quantile 경계·구간 수·정규화·digest, model과 optimizer, RNG, train-loader generator, history, current state와 best state가 모두 일치해야 한다. 다른 selector나 quantile 계약의 cache는 재사용하지 않는다.

## Seed42 판정

각 데이터셋에서 C는 다음을 모두 만족해야 한다.

- B와 A보다 raw quantity RMSE가 각각 엄격히 낮다.
- B 대비 body(`<= train p95`) MAE와 extreme-tail(`> train p99`) MAE 악화가 각각 2% 이하이다.
- B와 A 각각에 대한 time loss 증가가 0.01 이하이다.
- Metric이 finite이고 세 arm의 validation target ID와 건수가 같다.

A→B는 checkpoint 정렬 효과, B→C는 loss 추가 효과, A→C는 합산 효과로 따로 보고한다. 하나의 데이터셋이라도 실패하면 seeds52·62를 실행하지 않고 가중치나 selector를 바꾸지 않는다. Seed42가 모두 통과해도 held-out test는 열지 않는다.

기존 A artifact의 launch metadata에는 `time_intercept_limit=30`이 남아 있지만, 해당 source revision의 legacy likelihood·survival·median 경로는 상한 300을 코드에 직접 고정했다. 현재 코드는 설정값을 실제로 사용하므로 B와 C에는 300을 명시해 A의 유효 실행 동작을 맞춘다. 이 값은 세 데이터셋과 두 신규 arm에서 동일하다.

## 실행 순서

1. 로컬 contract·unit 검증
2. 커밋된 source를 5090 격리 경로로 전송하고 checksum 검증
3. CUDA와 세 데이터셋 full-data e1에서 B·C 총 6개 run 검증
4. e1이 모두 통과할 때만 세 데이터셋 seed42 e300 B·C 총 6개 run 실행
5. 공통 gate 판정과 로컬 감사

추후 진행사항은 **quantile-adaptive loss와 checkpoint 정렬**로 계속 명시한다.
