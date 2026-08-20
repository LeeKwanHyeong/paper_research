# H0/H3 Gradient Clipping and Quantity Damage Audit 시작 기록

## 상태

- 상태: 완료
- 준비 시각: 2026-08-20 19:14:20 KST
- 실험 시작 시각: 2026-08-20 19:16:38 KST
- 실험 종료 시각: 2026-08-20 19:16:50 KST
- 실행 서버: 5080
- tmux: `inter_h0_h3_grad_audit_0820`
- source revision: `ccbcfcf9210d5f0bb6e60adde3f7b431058a435f`
- artifact: `search_artifacts/count_aware_h0_h3_gradient_attribution_20260820`
- evaluation scope: train only
- validation 및 held-out test: 사용하지 않음

## 목적

H0는 현재 F0/F1 비교 기준으로만 유지하고 train loss 폭증을 미해결 위험으로 관리한다.
H3는 기존 best/final checkpoint를 이용해 100% gradient clipping의 직접 driver와
quantity 손상이 encoder, quantity head 또는 두 요소의 결합 중 어디에 가까운지
진단한다.

## Variant 계약

| Variant | Time head | 상태 | 진단 state |
| --- | --- | --- | --- |
| H0 | scaled exact RMTPP | 비교 기준선, 안정성 미해결 | initial/best/final |
| H3 | log-normal duration | quantity safety 실패, 미채택 | initial/best/final |

두 모델은 동일한 train-only 32개 batch에서 shared encoder, time head, quantity head별
gradient norm과 clipping scale을 측정한다. Best checkpoint에서는 H0/H3 quantity head를
교차 적용해 quantity 손상 위치를 보조 진단한다.

## 고정 조건

- Intermittent frozen 5,000 fixed train split
- seed 42, batch size 128, audit batch 32
- lookback 520주, maximum sequence length 256
- gradient clipping threshold `1.0`
- train mode dropout seed `42 + batch_index`
- validation과 held-out test를 새로 읽지 않음
- H3 상수, learning rate와 checkpoint를 사후 조정하지 않음

## 실행 명령어

```bash
SOURCE_REVISION=ccbcfcf9210d5f0bb6e60adde3f7b431058a435f \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash simple_lab_test/search/scripts/run_count_aware_h0_h3_gradient_attribution_20260820.sh
```

## 결과

- H0 best/final의 고정 32-batch clipping 비율은 각각 `3.125%`였다. H0의 기존
  폭증은 지속적 clipping보다 드문 batch 또는 extreme duration에 민감한 문제로 남긴다.
- H3 best/final은 모든 batch가 clipping됐고, time head가 joint squared-gradient
  norm의 `93.30%`/`86.53%`를 차지했다.
- H3 best의 shared encoder에서 time gradient는 quantity gradient의 약 `7.06배`였다.
  Gradient cosine 중앙값은 `-0.0066`으로 강한 반대 방향 충돌은 아니었다.
- H3 best train quantity loss는 H0보다 `3.15%` 낮았지만 raw MAE는 `22.15%`
  높았다. Eval-mode crossing에서도 log-MSE는 `2.94%` 낮고 raw MAE는 `31.09%`
  높아 log-domain 목적과 raw quantity metric의 불일치가 확인됐다.
- H0/H3 encoder와 quantity head를 교차 적용하면 양방향 모두 오차가 크게 증가했다.
  이는 강한 co-adaptation 증거이며 각 구성요소의 독립적 손상을 증명하지는 않는다.
- Validation과 held-out test는 읽지 않았고 모든 audit 값은 finite였다.
