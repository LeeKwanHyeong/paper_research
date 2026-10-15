# CNN 메모리 감쇠 진단 실행 기록

## 현재 계약

- 원래 승인된 CNN9·alpha4·Train/Validation2의 72조건을 유지한다. 새 학습이나 Test 평가는 아니다.
- 현재 bundle: `search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3`.
- 계약 raw SHA: `d51ab06365a7165e00cc18849b84caba4ee2dd8af6a36aee84a5c1d62e01fab3`.
- science source131, 입력 원본 byte SHA, 선택 checkpoint와 tensor SHA, alpha 목록, 원래 기한은 보존했다.
- 두 서버의 실행 상한은 각각3개. 고정 모델의 native GPU 재현 때문에 5080은7개, 5090은2개 모델이다. 후속 학습 역시 두 서버별 최대3개를 사용한다.

## 운영 오류와 보존

v1은 과학 source131 외에 필요한 빈 `source/sample_data/.frozen_root_sentinel`이 포장에서 빠져 import 단계에서 실패했다. 두 dispatcher 실제 OS exit1과 worker5 exit1을 기록했으며 예측 전이었다. 기존 파일과 실패 root는 보존했다. 빈 표시 파일을 포함한 임시 배포 구조로 같은 오류와 해결을 재현했다.

v2는 RAF42/52 Validation 재현을 통과했다. Intermittent는 기존 `training_data_path`라는 이름의 입력에 실제 Test 행도 포함되어 있어, eager 로딩 후 split 검사에서 중단됐다. 이 경계 검사를 로딩 전에 적용하지 못한 것은 진단 연결부 오류다. Test 행에 대한 예측·지표 산출은 없었다. 해당 파일을 변경하거나 Test 행을 분석하지 않았고, split 라벨별 행 수만 확인했다.

v3는 입력 파일을 lazy scan한 뒤 허용 split을 필터링하고, 그 다음 materialize한다. 합성 혼합 parquet로 필터가 collect 전에 적용되는지 검증했다. 원 입력 SHA와 Train/Validation 대상은 동일하다. 변경된 입력 경계를 포함한 전체 진단 완료를 검증하기 위해 v3에서 같은 고정 모델을 평가하며, v2 부분 결과를 덮어쓰거나 새 결과와 혼합하지 않는다.

자동 실행기는 실패한 worker를 재시도하지 않는다. v1·v2는 각각 실패로 종료했고 원문을 보존했다. 원인을 특정해 수정·검증한 별도 v3 경로를 사용하며 기존 root·claim·failure에는 쓰지 않는다. 원래 deadline을 연장하지 않았다.

## 완료 확인

- 실제 종료: 5080은 2026-10-15 13:12:21 KST, 5090은 13:12:09 KST. 두 dispatcher의 실제 OS exit0과 18개 worker의 실제 exit0을 확인했다. supervisor 자체 OS exit는 직접 관측하지 않았다.
- 13:12:30 KST 관측에서 두 서버의 남은 진단 worker와 GPU compute PID가 없었다.
- CNN9 × Train/Validation2 × alpha4 = 72조건이 완료됐다. Validation9에서 alpha1의 기존 여섯 지표를 재현했고, checkpoint·모델 파라미터 불변을 확인했다.
- host별 archive를 한 번 회수했다. 원본 archive를 포함한 manifest의 322개 파일 SHA를 로컬에서 다시 검증했다. manifest raw SHA는 `9b7044490727a42add4b9e05538fdf315137f0a956e25d8b1422f59a37430819`다.
- 분석기는 모든 예측 조각 SHA·대상 수를 확인하고 예측 행에서 여섯 지표를 다시 계산했다. 구간 경계는 각 모델의 Train 특징에서만 계산했다.
- 로컬 검증: 관련 네 테스트 파일에서 58 tests 및 13 subtests 통과. 학습 source131과 기존 선택 모델은 변경하지 않았다. 새 학습과 Test 지표 산출은 없었다.

## 결과와 다음 단계

Validation에서 감쇠한 27개 조건의 전체 RMSE가 모두 alpha1보다 높았다. 사전 계약의 고정 감쇠 진입 기준과 이에 종속된 상황별 gate 진입 기준은 미충족이다. 따라서 새 학습은 시작하지 않았다.

조건별 MSE의 개선·악화가 갈리는 탐색적 패턴 자체는 존재한다. 이를 숨기거나 gate가 불가능하다는 뜻으로 해석하지 않는다. 특히 Taxi의 메모리/기본 표현 노름 비율 상위 구간에서는 세 seed 모두 MSE가 소폭 줄지만 MAE는 모두 높아진다. 전체 감쇠의 이득은 상황별 gate의 논리적 필요조건이 아니므로, 이 진입 기준 미충족은 gate 전체의 반증이 아니다.

수치와 판단 근거는 `DECISION.md`, 전체 3seed 평균·표본 SD·seed별 여섯 지표 차이와 모든 구간 분석은 `analysis/SUMMARY.md` 및 `analysis/SUMMARY.json`에 기록했다. 후속 학습을 하게 되면 이미 구현·검증한 서버별 최대3개 실행 방식을 사용한다. 이번 결과만으로 3개 구조의 새 학습을 자동 실행하지 않는다.
