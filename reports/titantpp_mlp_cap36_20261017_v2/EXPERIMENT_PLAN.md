# Q3 Cap 데이터의 네 MLP 구조 비교

**실험 범위 확정 — 완료**

- 사용자는 Cap 데이터로 5080·5090 재학습과 Test 평가를 승인했고, 모델 범위를 “방금 분석한 MLP 폭4·폭16·구조 대조군 2개: 총 36개 학습”으로 선택했다.
- Taxi·RAF·Intermittent × seed42·52·62 × 이력 MLP 폭4, 이력 MLP 폭16, AllAvailable 폭4, CurrentOnly 용량 대조군을 각각 새로 학습한다.
- 이전 원본 데이터·CNN·checkpoint·결과는 보존한다. 이전 Cross Fusion이나 새로운 공유 W 후보는 이번 실행에 포함하지 않는다.

**같은 Cap 입력과 정답을 사용한다 — 완료**

- 기존 `benchmark_data/Cap`을 그대로 복사한다. 원본 Train의 전체 관측 수량으로 계산했던 Q3 상한은 Taxi 65, RAF 10, Intermittent 20이다. 모든 split의 관측 수량과 정답 수량에 같은 상한이 적용돼 있다.
- log1p 입력, 수량·시간 head, 공동 손실, 메모리는 유지한다. 수량 초기화 평균·표준편차와 수량 구간만 capped Train으로 다시 계산한다. 시간 초기화·평가 대상·순서가 기존과 같은지 확인한다.
- Cap 뒤 q90/q95/q99가 상한에 겹치는 현상을 그대로 보존한다. `q > capped Train q99` 지표는 표본 0/N/A다. 별도로 `q == cap` 집단을 보고한다. 예측 출력은 사후 clipping하지 않는다.
- 비활성 tail 보조 head의 생성자는 threshold<clip을 요구하므로 해당 생성자 값만 기존 Factory 기본값으로 유지한다. 활성 수량 손실은 count_only_log_regression·lambda_tail=0으로 동일하다. 네 구조의 초기 상태·예측·활성 손실·gradient가 이 비활성 상수에 영향을 받지 않는지 합성 검사했다. Cap 평가 구간은 이 보조 설정과 분리한다.
- Cap에서는 목표 문제 자체가 달라졌으므로 기존 raw 결과를 같은 목표의 기준선으로 쓰지 않는다. 이번 네 모델의 Cap 결과끼리 비교한다.

**서버별 최대 3개씩 학습한다 — 진행 중**

| 서버 | 배정 | 새 학습 수 |
|---|---|---:|
| 5080 | Taxi의 네 모델 ×3seed + Intermittent seed42의 네 모델 | 16 |
| 5090 | RAF의 네 모델 ×3seed + Intermittent seed52·62의 네 모델 | 20 |

- 같은 데이터셋·seed의 네 모델은 같은 서버에서 비교한다. Intermittent의 seed42는 5080/cuDNN91900, seed52·62는 5090/cuDNN92000이라는 한계를 남긴다.
- 2026-10-17 00:37:06 KST 실제 점검에서 5090은 GPU 작업이 없고, 5080은 이전 Intermittent seed52·62 학습 2개가 진행 중이었다. 기존 프로세스를 종료하지 않는다. 5080 새 큐는 GPU 유휴를 기다린 뒤 검증하고 시작한다. 이 과거 점검을 현재 유휴 증명으로 사용하지 않는다.
- 현행 Python/torch/GPU UUID, Cap 입력, 네 구조의 합성 forward/backward·checkpoint·target 차단을 확인한다. 최대 batch128·이력 길이의 합성 메모리 사용량으로 3개 배정 가능 여부를 확인한다. 이는 실제 3프로세스 최고 사용량 측정과 구분한다.
- 학습 예산은 300 epoch 상한, 최소40, patience40, batch128, AdamW lr0.001/weight_decay0.01, clip1이다. fit 36시간과 서버별 큐 168시간 상한을 고정한다. 서버 대기시간도 큐 기한에 포함한다. 자동 retry/resume은 하지 않는다.

**Validation 선택을 고정하고 Test까지 평가한다 — 다음 작업**

- 전체 Validation의 수량 RMSE가 최초로 엄격히 최소가 된 epoch를 선택한다. MAE는 그 checkpoint의 값이며 별도로 선택하지 않는다.
- 서버별 모든 배정 학습이 끝나면 선택 checkpoint의 Validation을 새 프로세스로 다시 평가한다. 동일 대상·개별 예측·시간 손실·전체 및 상한 집단 지표를 재현해야 해당 서버의 Test를 연다. 전36개의 전역 gate가 아닌 서버별 gate다.
- 같은 checkpoint·입력·전처리로 Test를 평가한다. 모델 재선택이나 Test를 이용한 도구 수정은 하지 않는다. 모든 단계는 최대3개 새 프로세스로 제한한다.
- 종료 후 actual worker/dispatcher exit, 소유 프로세스 부재, archive/manifest SHA를 확인하고 원본을 회수한다. supervisor 자신의 OS exit는 직접 관측 여부를 구분한다.

**결과를 비교하고 기록한다 — 다음 작업**

- 데이터셋·split별 네 모델의 3seed RMSE·MAE 평균±표본SD(ddof1), 모든 seed별 폭16−폭4, AllAvailable−폭4, CurrentOnly−폭4, 각 대조군−폭16 절대·상대차이를 포함한다. 악화한 seed를 제외하지 않는다.
- 전체/상한 집단 RMSE·MAE·recorded positive-integer TimeNLL과 빈 strict-tail 지표를 함께 보존한다. 서로 다른 데이터셋 raw 단위의 오류를 평균내지 않는다.
- 기존 코드와 원본을 보존한 별도 commit·실험 경로를 사용한다. 이번 실행은 유한한 36개 큐이며 이전 backbone 반복개선 자동화를 다시 켜지 않는다. 새로운 모델·추가 실험을 자동으로 시작하지 않는다.
- v1 로컬 준비는 완료했지만 배포 사전검사에서 패키징 오류가 확인됐으며, v2는 이 운영 오류만 수정한다. 이 상태는 새 Cap 학습/Validation/Test 완료를 의미하지 않는다. 실제 진입·완료 상태는 새 receipt와 서버 관측으로 갱신한다.

**로컬 구현 연결 검증 — 완료**

- `LOCAL_PYTEST.json`의 실제 종료코드는 0이다. v2 배포 수정 후 73개 검사와 24개 하위 검사를 통과했다. 네 구조의 합성 2epoch 학습·checkpoint 선택·Validation 개별 예측 재현·Test gate 전체 경로, 입력 통계·큐 상한·실패·중복 실행 차단·집계 검사를 포함한다.
- 과거 동결 폭16과 현재 폭16의 초기 state·RNG 및 비영 보정 가중치의 forward/backward 동일성을 확인했다. 현재 Width 소스의 폭8·12 지원 확장이 폭16 계산을 바꾸지 않았다.
- v1 보고 경로의 `SOURCE_COMPARISON.json`에 기록된 나머지 8개 핵심 모델·memory·loader·손실·학습 파일은 과거 동결 파일과 바이트가 같다. 이를 전체 저장소의 모든 과거 소스가 같다는 뜻으로 확대하지 않는다.
- 네 모델의 실제 Cap 학습과 실제 Test는 로컬 합성 검사의 대상이 아니며 서버에서 별도로 실행한다.

**배포 누락 수정 — 완료**

- v1은 배포 묶음의 빈 sample_data 폴더 누락으로 실제 학습 전 검증이 실패했다. 원문과 기존 QA 범위의 누락을 v1의 PACKAGING_FAILURE_QA에 보존했다.
- v2의 배포 묶음에 빈 폴더를 포함하고, 5080의 v1 대기 작업이 학습 전 실패로 종료했음을 확인하는 순서를 추가했다. 데이터·모델·학습·선택 조건은 v1과 같다. 자세한 범위는 PACKAGING_REPAIR.md에 기록했다.
- 이 로컬 통과는 실제 서버의 학습 시작·완료를 뜻하지 않는다. 배포 후 실제 worker/GPU 상태를 별도로 확인한다.

실제 v2 시작 상태는 START_STATUS.md와 START_RECEIPT.json에 기록했다. 5090 실제 학습3, 5080 대기이며 전체 학습이나 Test 완료를 의미하지 않는다.
