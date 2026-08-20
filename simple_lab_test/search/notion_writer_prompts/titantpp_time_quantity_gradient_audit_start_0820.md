# Notion Source: TitanTPP Time-Quantity Gradient Audit

## 대상

- 위치: `5. Model Design Enhancement`
- 페이지 제목: `2026-08-20 TitanTPP Time-Quantity Gradient Audit and T1 Integration`

## 상태

- **준비 중**
- 준비 시각: 2026-08-20 13:00:27 KST
- 실행 서버 / tmux: 5080 / `inter_time_qty_grad_audit_0820`

## 목적

H1 실패 원인을 slope 표현력 부족과 time-quantity gradient 간섭으로 분리한다. 실제
통합 후보와 같은 Hard-LMM 및 T1 quantity objective에서 train row만 사용한다.

## Variant 계약

| Variant | Time head | 역할 |
| --- | --- | --- |
| H0 | scaled exact, `w * tau <= 40` | 기존 exact 대조군 |
| H1 | stable exact, `w * tau <= 8` | 안정화 후보 |

비교 축 외 dataset, train split, Hard-LMM, T1, seed, optimizer와 batch 조건은 같다.

## 고정 조건

- dataset: Intermittent frozen 5,000 train rows
- model: TitanTPP Hard-LMM
- quantity objective: T1 tail-shared
- epochs / seed: `3` / `42`
- learning rate / batch size: `1e-3` / `128`
- lookback / maximum sequence length: `520주` / `256`
- audit batches: stage별 `32`
- validation / held-out test: 미사용
- artifact: `search_artifacts/count_aware_time_quantity_gradient_audit_20260820`

## 실행 명령어

```bash
SOURCE_REVISION=<source_revision> \
  bash simple_lab_test/search/scripts/run_count_aware_time_quantity_gradient_audit_20260820.sh
```

## 결과
