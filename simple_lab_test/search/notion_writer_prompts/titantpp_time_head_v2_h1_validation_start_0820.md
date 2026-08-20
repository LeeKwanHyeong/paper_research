# Notion Update Source: Time Head v2 H1 Validation Safety Screening

## 위치

- `5. Model Design Enhancement`
- 동일 제목이 있으면 새 페이지를 만들지 않고 기존 페이지를 갱신한다.

## 페이지 제목

`2026-08-20 TitanTPP Time Head v2 H1 Validation Safety Screening`

## 상태

- 준비 중
- 실행 서버 / tmux: `5080` / `inter_timehead_v2_h1_val_e300_0820`

## 목적

Train-only 안정성 검증에서 선택된 H1 time head가 validation에서도 기존 time 성능을
유지하고 quantity 예측을 훼손하지 않는지 확인한다. 이 검증을 통과한 경우에만 H1을
공통 time head로 고정하고 memory 구조 비교를 다시 진행한다.

## Variant 계약

| Variant | 구성 | 역할 |
| --- | --- | --- |
| H0 | 기존 scaled exact head, 넓은 slope·intercept 범위 | 기존 완료 결과 |
| H1 | 축소된 slope 범위, smooth bounded intercept, train-rate 초기화 | validation 후보 |

Time head 이외에는 같은 Intermittent split, TitanTPP Hard-LMM backbone, quantity head,
seed와 학습 조건을 사용한다.

## 고정 조건

- dataset: Intermittent fixed split
- model: TitanTPP Hard-LMM
- epochs / seed: 최대 `300` / `42`
- learning rate / batch size: `1e-3` / `128`
- lookback / maximum sequence length: `520주` / `256`
- early stopping: minimum epoch `40`, patience `40`
- checkpoint: minimum validation joint objective
- quantity: mark-free log-MSE
- artifact: `search_artifacts/count_aware_time_head_v2_h1_validation_e300_20260820`

## 실행 명령어

```bash
SOURCE_REVISION=12c85cdb4b06d8e55652eb7aee5cde57bd7f8ce6 \
  bash simple_lab_test/search/scripts/run_count_aware_time_head_v2_h1_validation_e300_20260820.sh
```

## 결과

