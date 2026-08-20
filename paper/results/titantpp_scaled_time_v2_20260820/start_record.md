# TitanTPP Scaled Exact Time Head v2 실험 기록

## 상태

- 상태: 구현·로컬 계약 검증 완료, 5080 실행 준비
- 설계 일자: 2026-08-20
- 실행 서버: 5080
- implementation revision: `dc5a58b`
- launch revision: 실행 스크립트 커밋 후 고정
- preflight tmux: `timehead_v2_preflight_0820`
- train-only stability tmux: `inter_timehead_v2_train_e3_0820`
- Held-out test: 잠금

## 목적

현재 scaled exact RMTPP head의 수학적 exactness는 유지하면서, train loss spike를
유발한 slope 범위, intercept 범위와 초기 hazard를 안정화한다. H1을 우선 검증하고
H1이 train-only gate를 실패할 때만 낮은 time-head learning rate의 H2를 연다.

## Variant 계약

| Variant | `w * tau` budget | Intercept | 초기값 | Time-head LR |
| --- | ---: | --- | --- | ---: |
| H0 | 40 | hard clamp `±30` | `log(time_scale)` | `1.0x` |
| H1 | 8 | smooth tanh `±6` | train mean rate | `1.0x` |
| H2 | 8 | H1과 동일 | H1과 동일 | `0.1x` |

## 선택 원칙

Validation을 읽지 않는 train-only stability runner로 H0와 H1을 먼저 비교한다. H1이
사전 gate를 통과하면 H2는 실행하지 않는다. H1이 실패할 때만 H2를 실행하며, H2도
실패하면 validation screening을 시작하지 않는다.

## 결과

실험 시작 후 작성한다.

## 실행 명령어

```bash
SOURCE_REVISION=<launch_revision> \
  bash simple_lab_test/search/scripts/run_count_aware_time_head_v2_preflight_20260820.sh

SOURCE_REVISION=<launch_revision> \
  bash simple_lab_test/search/scripts/run_count_aware_time_head_v2_stability_20260820.sh
```

## Artifact

- preflight: `search_artifacts/count_aware_time_head_v2_preflight_20260820`
- train-only stability: `search_artifacts/count_aware_time_head_v2_train_stability_20260820`
