# H0/H3 Gradient Clipping and Quantity Damage Audit 시작 기록

## 상태

- 상태: 5080 실행 준비 완료
- 준비 시각: 2026-08-20 19:14:20 KST
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
