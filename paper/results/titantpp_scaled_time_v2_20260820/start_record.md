# TitanTPP Scaled Exact Time Head v2 실험 기록

## 상태

- 상태: 5080 preflight 및 train-only stability chain 실행 중
- 설계 일자: 2026-08-20
- 실행 서버: 5080
- implementation revision: `dc5a58b`
- launch revision: `34b46701bf00603d3e1624162a2bd1b9c91cff1f`
- 실험 시작 시각: 2026-08-20 09:06:33 KST
- tmux: `inter_timehead_v2_gate_0820`
- 실행 순서: preflight 성공 후 train-only stability 자동 진입
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

Preflight focused test `26 passed`, Intermittent CUDA model-test와 Instacart
top-20 e1 smoke가 완료됐다. 현재 train-only short run에 진입했으며, 완료 후 같은
tmux에서 full H0/H1 및 조건부 H2가 이어진다. 성능·선택 결과는 아직 판정하지 않았다.

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
