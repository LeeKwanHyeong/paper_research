# 2026-08-24 Count-Aware Benchmark Train-Only Quantity and Sequence Audit

## 상태

- 완료
- 실험 시작 시각: 2026-08-24 08:24:21 KST
- 실행 서버 / tmux: local / not applicable
- 평가 범위: train-only
- Held-out test: 미사용

## 목적

Intermittent v2, Online Retail II, RAF Spare Parts의 quantity scale과 sequence
구조를 동일한 정의로 비교한다. Intermittent에서 확인한 body MAE와 tail RMSE의
trade-off가 다른 native-count benchmark에서도 성립할 가능성이 있는지 모델 실험
전에 데이터 조건부터 확인한다.

## Variant 계약

세 데이터셋의 명시적인 `_train.parquet`만 읽는다. Quantity quantile은 nearest
방식의 p50/p95/p99로 계산하고, sequence structure는 train events per series와
next-event history length로 나눠 기록한다.

## 고정 조건

- dataset: Intermittent v2, Online Retail II, RAF Spare Parts
- split: train-only
- quantity quantile: nearest p50, p95, p99
- sequence unit: `oper_part_no`
- validation/test parquet: 접근 금지
- artifact: `paper/results/count_aware_train_distribution_audit_20260824`

## 실행 명령어

```bash
PYTHONPATH=. python benchmark_data/scripts/audit_train_quantity_history.py \
  --output-dir paper/results/count_aware_train_distribution_audit_20260824
```

## 결과

| Dataset | Train events | p50 | p95 | p99 | Max | Events/series p50 | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intermittent v2 | 398,824 | 2 | 46 | 187 | 477 | 62 | 182 | 193 |
| Online Retail II | 567,063 | 4 | 40 | 144 | 19,152 | 105 | 613 | 2,422 |
| RAF Spare Parts | 30,779 | 2 | 60 | 200 | 2,062 | 6 | 10 | 15 |

- 세 데이터셋 모두 train quantity는 오른쪽 꼬리가 길지만 절대 임계값과 tail 강도가 다르다.
- Online Retail II의 `max/p99`는 약 133배여서 uncapped raw loss 적용 위험이 가장 크다.
- RAF는 train events/series 중앙값이 6으로 sequence가 매우 짧다.
- validation/test parquet은 읽지 않았고 held-out test는 사용하지 않았다.
- 새로운 mid-body balanced objective 구현은 보류한다. 먼저 공통 T0와 TitanTPP-T1의 matched validation을 세 데이터셋에서 수행한다.
