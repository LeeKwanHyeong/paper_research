# B1 prior-prefix 출력 read 실행 증적

- 저장소/작업 브랜치: `paper_research/codex/b1-prior-prefix-read`.
- [기계 판정 계약](../../contracts/titans_mac_prior_prefix_v1.json), [설명 계약](../../contracts/titans_mac_prior_prefix_v1.md).
- 현재 논문 T0 기준선은 그대로 유지한다. 이 디렉터리는 한 후보의 별도 연구 증적이다.
- `frozen_t0_metric_targets.csv`: 동결 reference에 기존 body 5% 기준과 별도 strict raw RMSE 목표를 적용한 9개 dataset/seed 목표값. 판정에는 반올림하지 않은 JSON 원수치를 사용한다.
- `frozen_references.json`: 기존 T0/RMTPP/THP 3 datasets × 3 seeds = 27 validation 행. 원수치, 출처 summary/hash, checkpoint hash를 보존한다. 이번 fresh B1과 후보의 cap300 계약에 대응한다. 역사적 reference를 새로운 paired rerun으로 표현하지 않는다.
- `local_contract_tests.log`, `local_contract_tests.xml`: 최종 통합 207 passed, CUDA 전용 4 skipped, 실패/오류 0. 실행 명령과 검사 source hash를 로그에 기록했다. 아래 47개·22개 부분 검증을 포함한 총수다.
- `local_execution_tests.xml`, `local_execution_validation.json`: launcher/package/CUDA cost 판정기 단위 테스트 47 passed, 실패/오류/skip 0.
- `local_comparison_tests.xml`: frozen reference·paired seed·지표 판정 및 실패 경계 테스트 22 passed, 실패/오류/skip 0.
- GPU 실행과 성능 평가의 상태는 별도 실행 감사 및 5090 phase proof로 판단한다. 로컬 통과 수치나 합성 benchmark는 성능 개선 증거가 아니다.

## 고정된 판정

1. B1 추가 효과: Instacart body MAE 1% 이상 개선, 전체 dataset body MAE 악화 1% 이하, RMSE 악화 없음, tail 2%/time loss .01 이내.
2. T0 기존 기준: 세 dataset 각각 body MAE 5% 이상 개선, RMSE/tail 악화 2% 이하, time loss .01 이내.
3. 공통 raw RMSE: 세 dataset 각각 T0보다 엄격히 감소하고 전체 MAE 악화 2% 이하.
4. 외부 TPP: RMTPP/THP 각각에 대한 RMSE 우위는 별도 판정하며 1–3 통과와 혼동하지 않는다.

Seed42의 1–3이 모두 통과해야 seeds52/62로 확장한다. e1은 정상 실행·처리 건수·학습 경로·저장/복원만 확인한다. Held-out은 사용하지 않는다.

## 첫 CUDA 시도의 검사 수정

- Source `5206e46`: 88 passed / 1 failed. 세 CUDA 전용 forward·gradient·AdamW·저장/복원 테스트는 모두 통과했다. 비용 측정·e1은 실행되지 않았다.
- 실패한 CPU batch permutation 검사는 bitwise를 요구했다. 5090의 PyTorch 2.11 CPU에서는 기존 B1도 최대 출력 차이 5.96e-8, 후보는 1.19e-7을 보였다. 모든 output/state/diagnostic은 기존 JSON의 CPU atol1e-7/rtol1e-6 내에 있었다.
- 해당 permutation 검사만 기존 동결 tolerance를 사용하도록 수정했다. 모델 코드·성능 및 비용 gate·JSON 계약 hash는 바꾸지 않았다. 원본 B1 재현과 series reset의 bitwise 조건도 유지했다.
- `attempt_5206e46_cuda/`에 실패와 합성 원인 진단을 보존한다. 수정 후 로컬 후보 suite는 86 passed / 3 CUDA skipped (`local_permutation_contract_tests.xml`). 새 커밋으로 CUDA 단계 전체를 fresh 재검증한다.

## 두 번째 CUDA 시도의 비용 측정 분리

- Source `a2fda0e`: 필수 테스트 89 passed / 0 failed / 0 skipped.
- T0 H16의 cold/warmup3/measured10을 완료했으나, CUDA allocated memory 68,157,440 bytes가 cleanup 후 남아 B1 측정 전 중단됐다. 비용 gate 자체를 평가할 9개 행은 아직 없다.
- 별도 합성 matmul은 tensor 제거 후에도 33,554,432 bytes가 남고 cuBLAS workspace clear 후 0이 됨을 확인했다. 모델 상태 누수라고 단정하지 않는다.
- 각 비용 행을 독립 subprocess에서 측정해 프로세스 종료로 다음 행과 완전히 분리하도록 보강한다. 기준·횟수·모델·성능 계약은 동일하다. `attempt_a2fda0e_cuda/`에 원본 증거를 보존한다. e1·screening·held-out은 미실행이다.

- 비용 측정 subprocess 격리 구현 후 validator38 + launcher28 = 66 tests passed, 0 skipped (`local_cost_isolation_tests.xml`, `local_cost_isolation_validation.json`). 각 worker의 source/contract hash·PID·시작 allocation0·정상 종료를 확인한 후만 비용 gate를 계산한다.

## CUDA 통과 — source add02ba

- `cuda_add02ba/status.json`: 89 mandatory tests passed, 0 skipped; source·contract·data·Runtime 검증 및 9개 독립 측정 행 완료. 비용 gate PASS.
- `cuda_add02ba/cuda_audit.json.workers/`는 각 fresh worker의 source/contract/hash/PID·측정 결과를 보존한다.

| H | 후보/B1 step | 후보/B1 peak allocation | 후보 peak GiB | 후보/T0 step |
|---:|---:|---:|---:|---:|
| 16 | 1.5501 | 1.2092 | 1.1704 | 6.7610 |
| 64 | 1.3932 | 1.0550 | 2.4670 | 17.0950 |
| 255 | 1.2737 | 1.0341 | 7.7058 | 22.1338 |

표는 표시용 반올림이며 gate는 JSON 원수치로 판단했다. T0 대비 6.76–22.13배 step 비용은 material limitation이다. 이번 incremental B1 비용 gate 통과를 T0 수준 효율이나 성능 채택으로 표현하지 않는다. 실제 데이터 e1·screening은 별도 phase proof가 필요하다.

## 승인된 후속 학습의 유한 실행

- `run_approved_training_phases.py`는 frozen source add02ba 밖에 두는 제어 스크립트다. 소스·manifest·contract·launcher hash를 고정하며 실제 model code를 수정하지 않는다.
- CUDA proof → e1 6 runs → 정상 감사 통과 시 screening 6 runs → 공통 성능 gate 통과 시 confirm 12 runs를 한 번만 실행한다. 각 phase는 기존 frozen launcher를 명시 호출한다. Screening 미달은 정상 분석 완료로 기록하고 confirm을 시작하지 않는다.
- `local_training_controller_tests.xml`: 21 passed / 0 skipped. 순서·proof/hash·완전 paired grid·성능 미달 중단·실제 별도 process group 종료를 검증했다.
- 별도 예약 작업이나 서비스가 아니며 retry/resume/overwrite를 하지 않는다. 실행 상태는 remote `outputs/full_validation/status.json`과 각 phase status에 남긴다.

- 실제 시작 확인: 2026-09-05T11:37:37.319541+00:00 UTC. `prior_prefix_training_add02ba`에서 Taxi B1 seed42 e1이 RUNNING이다. `training_launch_snapshot/` 및 `execution_record.json`은 이 시점의 복사본이며 최신 상태는 5090 원본 `outputs/full_validation/status.json`으로 확인한다. Screening과 추가 seed는 선행 조건을 충족할 때만 시작한다.

## 첫 실제 e1은 계약 위반으로 제외

- `add02ba` Taxi B1 seed42는 train 1 epoch와 validation을 마쳤지만 감사에서 FAIL 처리됐다. 승인된 controller도 중단됐으며 candidate e1/screening/confirm은 시작되지 않았다.
- `launch_contract.json`의 split_rows에 test 8,327행이 있었다. 기존 main loader의 사전 filter 분기가 B1·prior-prefix를 포함하지 않아 test행도 메모리에 적재했다. artifact의 held_out_test_evaluated=false와 평가 대상 validation-only는 별개로, 이 실행은 적재 전 제외 계약을 충족하지 못했다.
- `e1_add02ba_rejected/`에 원본 launch/summary/history/상태/로그를 보존한다. 성능 판단·정상 e1 증거·새 대조군으로 사용하지 않는다. 코드에서 통계/target/history의 test 사용 여부를 별도 감사하고 실제 main 진입점 회귀검사를 추가한다.

- 수정 후 실제 main 진입점 9개 테스트를 CUDA 필수 suite에도 추가했다. 관련 로컬 통합은 163 passed / CUDA 3 skipped (`local_input_scope_gate_tests.xml`). 단계별 source/package 검증도 새 테스트의 포함을 요구한다.
- 기각된 B1 checkpoint는 나머지 coverage/CPU restore 검사만 별도 진단했으며 `qualifies_e1=false`, 원본 FAIL을 유지했다. 추가 오류는 없었고 정상 proof로 사용하지 않는다.

## 입력 범위를 수정한 최신 CUDA 통과 — c21aea5

- `cuda_c21aea5/status.json`: 기존 계약 89 + 실제 입력 진입점 9 = 98 passed / 0 skipped. 9개 비용 행도 독립 측정을 완료했고 기존 gate를 모두 통과했다.
- 현재 실행 source는 `c21aea563aeaa7f82a8814552594dbc77b8da458`, manifest는 `3e992336ae6dde285797d7aa093ca741a8800c8f3ba9de2937a8dfb4d99ec1c1`이다. JSON 계약과 모델 소스는 유지됐다.
- 최신 후보/B1 step 비율은 H16/64/255에서 1.4130 / 1.2833 / 1.2765이며 peak 최대는 7.6096 GiB다. 후보/T0 step은 6.50–21.32배로 여전히 크다.
- 외부 controller를 새 source/hash에만 고정했고 실제 commit archive의 hash 결속을 포함한 테스트 22개가 통과했다. 이전 add02ba proof는 새 source의 선행 증거로 수용하지 않는다.
