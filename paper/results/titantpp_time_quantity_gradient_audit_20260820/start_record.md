# TitanTPP Time-Quantity Gradient Audit 시작 기록

## 상태

- 상태: 완료
- 준비 시각: 2026-08-20 13:00:27 KST
- 실험 시작 시각: 2026-08-20 13:04:14 KST
- 실행 서버: 5080
- tmux: `inter_time_qty_grad_audit_0820`
- source revision: `7486cc97331b7865b698b3bfcf79f015469ce5c2`
- artifact: `search_artifacts/count_aware_time_quantity_gradient_audit_20260820`
- evaluation scope: train only
- validation / held-out test: 사용하지 않음

## 목적

H1 stable exact head가 validation에서 실패한 원인을 slope 표현력 부족과
time-quantity shared-encoder gradient 간섭으로 분리한다. 이전 train-only 선택과 달리
실제 통합 후보와 같은 Hard-LMM 및 T1 tail-shared quantity objective를 사용한다.

## Variant 계약

| Variant | Time head | `w * tau` budget | 공통 조건 |
| --- | --- | ---: | --- |
| H0 | scaled exact RMTPP | 40 | Hard-LMM, T1, seed 42 |
| H1 | stable scaled exact RMTPP | 8 | Hard-LMM, T1, seed 42 |

두 Variant는 time-head slope/intercept 계약만 다르다. Dataset, train rows, model,
quantity objective, initialization seed, optimizer와 batch 순서는 동일하다.

## 진단 지표

- effective slope와 slope 상한 비율
- event별 `d(NLL)/dw < 0` 비율
- time 및 quantity loss의 shared encoder gradient norm
- 두 gradient의 cosine similarity와 강한 충돌 batch 비율
- bounded intercept 포화 비율
- train loss, pre-clipping gradient norm과 clipping 비율

## 판정 규칙

- H1 slope ratio가 `0.98` 이상이고 upward pressure가 `0.50` 이상이면 slope 계약
  실패로 본다.
- gradient cosine 중앙값이 `-0.10` 이하이고 강한 충돌 batch 비율이 `0.50` 이상이면
  gradient 간섭이 강하다고 본다.
- 이 판정은 train-only artifact로 고정하며 validation 결과를 사용해 상수나 routing을
  다시 선택하지 않는다.

## 실행 명령어

```bash
SOURCE_REVISION=<checksum_synced_full_sha> \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash simple_lab_test/search/scripts/run_count_aware_time_quantity_gradient_audit_20260820.sh
```

## 결과

H1은 epoch 3에서 slope ratio `0.9895`, upward pressure `1.0000`으로 slope 계약 실패
조건을 충족했다. 같은 시점의 gradient cosine 중앙값은 `+0.0466`, 강한 충돌 batch
비율은 `25%`로 gradient 분리 조건을 충족하지 않았다.

최종 판정은 `replace_slope_family_keep_shared_gradient`이다. 상세 수치와 H3 계약은
`result_analysis.md` 및 `paper/contracts/count_aware_final_time_t1_v1.*`에 기록했다.
