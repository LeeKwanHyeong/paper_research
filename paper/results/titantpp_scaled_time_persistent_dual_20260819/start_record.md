# TitanTPP Scaled-Time Persistent/Dual Memory 실험 기록

## 상태

- 상태: Intermittent seed-42 e300 완료, memory 후보 미채택
- 준비 시각: 2026-08-19 14:22:51 KST
- e300 시작 시각: 2026-08-19 14:37:12 KST
- 실행 서버: 5080
- CUDA device: 0
- CUDA smoke tmux: `titan_scaled_time_mem_smoke_0819`
- e300 tmux: `titan_scaled_time_mem_e300_0819`
- Source revision: `6bf27cbca219c010c245a6293f27abf14fc6ccb6`
- CUDA smoke artifact: `search_artifacts/count_aware_scaled_time_persistent_dual_cuda_smoke_20260819_r3`
- e300 artifact: `search_artifacts/titantpp_persistent_dual_scaled_time_screening_e300_20260819`
- Held-out test: 사용하지 않음
- 완료 확인 시각: 2026-08-19 23:50:32 KST

## 목적

Train-only scale로 정상화한 exact RMTPP time head 아래에서 persistent token 조건을 맞춘
Hard-LMM과 Surprise memory를 비교한다. 이어서 time은 Hard-LMM, quantity는 Surprise
residual을 사용하는 dual-route와 gradient-routing 대조군을 검증한다.

## Factorial 계약

| Variant | 구성 | 확인 질문 |
| --- | --- | --- |
| M0 | Persistent 16, 추가 memory 없음 | Persistent token만으로 설명되는 효과는 얼마인가? |
| M1 | Persistent 16 + Hard-LMM | 정상화된 time head 아래의 기준 성능은 무엇인가? |
| M2 | Persistent 16 + Surprise | 동일 persistent 조건에서 Surprise가 수량과 시간을 함께 개선하는가? |
| M3a | Time Hard-LMM, quantity Hard-LMM + Surprise, shared gradient | Task별 state 분리가 유효한가? |
| M3b | M3a와 동일 forward, quantity gradient는 adapter-only | Quantity loss의 encoder 간섭을 줄이면 time 성능을 보존하는가? |

## 고정 조건

- Intermittent fixed split, seed 42, validation only
- Epoch 300, minimum epoch 40, patience 40
- Batch 128, learning rate `1e-3`, gradient clipping 1
- Lookback 520 weeks, maximum sequence length 256, hidden dimension 64
- Quantity: mark-free log-MSE
- Time: scaled exact RMTPP, scale 3, bounded slope `10/3`, Jacobian 적용
- Checkpoint: minimum validation joint objective
- Held-out test: 사용하지 않음

## 실행 계획

1. 5080 source checksum, dependency, CUDA, dataset, split manifest를 확인한다.
2. CUDA focused test와 Intermittent 소량 model-test를 실행한다.
3. Instacart top-20 e1 smoke로 checkpoint와 artifact 생성을 확인한다.
4. Smoke 통과 후 별도 tmux에서 Intermittent seed-42 e300을 한 번 실행한다.
5. 완료 후 artifact를 `--delete` 없이 로컬로 동기화한다.
6. Manifest, log, summary, test summary, histories, scale-wise metrics, plots 순서로 분석한다.

## 실행 명령어

```bash
SOURCE_REVISION=6bf27cbca219c010c245a6293f27abf14fc6ccb6 \
  bash simple_lab_test/search/scripts/run_count_aware_scaled_time_persistent_dual_cuda_smoke_20260819.sh
```

```bash
SOURCE_REVISION=6bf27cbca219c010c245a6293f27abf14fc6ccb6 \
  bash simple_lab_test/search/scripts/run_titantpp_persistent_dual_scaled_time_e300_20260819.sh
```

## 결과

- 다섯 Variant가 모두 early stopping으로 정상 종료됐다.
- M2, M3a, M3b는 Hard-LMM 기준의 quantity acceptance gate를 통과하지 못했다.
- 최종 선택은 기존 M1 Hard-LMM이다. Multi-seed와 held-out test는 열지 않는다.
- Scaled exact time head는 finite contract를 통과했지만 train joint loss spike가 반복돼
  optimization 안정성은 별도 보완이 필요하다.
- 상세 분석: [result_analysis.md](result_analysis.md)
