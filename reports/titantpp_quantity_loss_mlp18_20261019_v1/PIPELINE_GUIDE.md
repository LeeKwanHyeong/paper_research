# 이번 수량 손실 실험을 이어서 확인하는 방법

**현재 상태: v1은 학습 전 qualification 실패로 종료했으며 실데이터 fit은0이다.** 이 문서는 당시 실행 계획으로 보존한다. v1의 prepare/deploy/완료 작업을 다시 실행하지 않는다. 원문은 `FAILURE_SUMMARY.json`과 `failure_evidence_v1/`에 있다. 운영 수정 후의 현재 경로는 [v2 실험 계획](../titantpp_quantity_loss_mlp18_20261019_v2/EXPERIMENT_PLAN.md)과 그 경로의 실제 receipt를 따른다.

## 보존할 현재 기준선 — 완료

공식 데이터는 Cap, 모델은 이력 MLP16, 시간 가중치0이다. 실수량 MSE/MAE는 Cap 정답을 실수량 단위로 사용하는 조건이며 상한 적용 전 데이터로 돌아가는 것이 아니다. 기존 log-MSE/시간 가중치0의9개는 저장 결과만 재사용한다. 새조건18개와 Validation18개 외 Test·기존 CNN·정지한 외부 Cap27은 실행하지 않는다.

실험 코드는 paper_research 저장소 `codex/contextual-retrieval-pilot-20261007` 브랜치의 `22f32472ae9160bcc0346a991e2f863aca266167`이다. `PREPARED.json`의 실제 계약/배포 SHA를 검증한다. 기존947파일과 신규 실행3개만 포함한 source950이며, 로컬 집계기와 일회성 완료 도구도 같은 커밋에 기록했다.

## 시작 상태와 실제 진행을 구분한다 — 진행 확인

`LAUNCHED_5080.json`/`LAUNCHED_5090.json`은 시작 요청 증적이다. 실제 native GPU 검증·학습 시작은 `START_RECEIPT.json`과 여기에 결속된 `observations/`를 확인한다. 과거 관측 시각을 새 관측으로 쓰지 않는다.

새 root는 다음과 같다.

- 5080: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_quantity_loss_mlp18_20261019_v1_5080`
- 5090: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_quantity_loss_mlp18_20261019_v1_5090`

각 host 최대3병렬이다. 동일 host에는 다른 실험을 겹쳐 시작하지 않는다. 기존 정지 외부 프로세스는 정상 신원/T 상태가 유지돼야 한다. 새 root를 재설치·재시작하거나 실패/claim을 삭제하지 않는다.

사용자가 진행 확인을 요청한 때 필요한 관측은 host별 한 번만 수행한다. 종료/회수 완료 상태라면 로컬 SHA 검증을 우선하고 현재 GPU 유휴 증명으로 과거 종료 관측을 재사용하지 않는다. 조회 실패를 학습 실패로 해석하지 않는다.

## 일회성 완료 작업이 원본 회수와 집계를 담당한다 — 학습 종료 대기

`finish_titantpp_quantity_loss_mlp18`는 로컬의 유한 작업이다. `completion_v1/attempt.json` 및 `COMPLETION_STARTED.json`으로 실행 신원을 확인한다. 반복 스케줄러가 아니며 기존 학습을 시작하거나 정지 외부 작업을 재개하지 않는다.

같은 신원으로 이미 시작된 dispatcher의 종료를 host당 SSH 연결 하나에서 기다린다. 정상 종료 뒤 기존 collector가 그 host 원본을 한 번 회수하고 원본/manifest/모든 SHA를 확인한다. 두 host가 모두 통과한 뒤 `compare_titantpp_quantity_loss_mlp18`가 순수 집계를 한 번 수행한다. 연결 실패·timeout·기존 attempt에는 자동 재시도하지 않는다.

`completion_v1/complete.json`과 `search_artifacts/titantpp_quantity_loss_mlp18_20261019_v1/retrieval/final_receipt.json`이 모두 complete여야 종료 후 비교 완료다. 로컬 완료 작업이 중단돼도 원격 학습을 임의로 kill하지 않는다. 원격 원본을 보존하고 원문 실패·남은 단계를 확인한다. 완료 도구를 미확인 상태에서 두 번 실행하지 않는다.

## 비교 완료 후 결과와 한계를 확인한다 — 다음 작업

- `comparison_v1/THREE_LOSS_VALIDATION.json/.md`, `SUMMARY.csv`, `SEED_DIFFERENCES.csv`를 확인한다.
- 신규18fit·선택 checkpoint Validation18·저장참조9·총27모델·3seed·시간 가중치0·Test0을 검증한다.
- 전체/상한도달 수량 RMSE·MAE의 평균±표본SD와 모든 seed 차이·악화를 함께 보고한다. 빈 q>cap 집단은 N/A다.
- MAE로 학습한 조건도 같은 Validation RMSE로 선택했다. 시간NLL은 시간 loss로 미학습한 head의 진단값이다.
- 손실의 절대 스케일·gradient clipping 영향까지 포함한 비교이며, 백본 기여·입력 log1p의 원인·Test 개선으로 확정하지 않는다.

새 학습이나 Test, 중단된 외부 실험의 재개는 이번 결과를 보고 별도로 결정한다. 이미 승인된18개 실험의 정상 종료·원본 회수·Validation 비교는 이 작업 안에서 완료한다.
