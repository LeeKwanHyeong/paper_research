# 2026-08-20 TitanTPP Stable Exact Time Head v2

## 상태

- 상태: 실험 준비 중
- 실행 서버 / tmux: 5080 / `timehead_v2_preflight_0820`, 이후 `inter_timehead_v2_train_e3_0820`

## 목적

기존 scaled exact time head에서 발생한 큰 train loss와 잦은 gradient clipping을 줄이면서 exact RMTPP density와 Jacobian 보정은 유지한다. Validation을 보기 전에 train-only 안정성으로 H1 또는 조건부 H2를 고정한다.

## Variant 계약

| Variant | 역할 | slope·intercept | time-head learning rate |
| --- | --- | --- | ---: |
| H0 | 기존 scaled exact 대조군 | `w·tau <= 40`, hard clamp `±30` | `1.0x` |
| H1 | 주 후보 stable exact | `w·tau <= 8`, smooth tanh `±6` | `1.0x` |
| H2 | H1 실패 시에만 실행하는 최적화 대조군 | H1과 동일 | `0.1x` |

그 외 backbone, quantity head, split, seed, batch size와 context 조건은 동일하다.

## 고정 조건

- dataset: Intermittent frozen 5,000; 사전 smoke는 Instacart top-20 추가
- model: TitanTPP Persistent-only
- epochs / seeds: train-only 3 epoch / seed 42
- lr / batch_size: `1e-3` / `128`
- lookback / max_seq_len: `520` / `256`
- split_mode: fixed; head 선택 단계는 train-only
- 주요 model/loss 옵션: mark-free log-MSE quantity head, exact density, Jacobian 보정, clamp 미사용
- artifact: `search_artifacts/count_aware_time_head_v2_train_stability_20260820`

## 실행 명령어

```bash
SOURCE_REVISION=<full_sha> bash simple_lab_test/search/scripts/run_count_aware_time_head_v2_preflight_20260820.sh
SOURCE_REVISION=<full_sha> bash simple_lab_test/search/scripts/run_count_aware_time_head_v2_stability_20260820.sh
```

## 결과
