# 원시간 입력 대조27의 일회성 후속 실행

## 부모 실험 완료를 기다린다 — 외부 작업 대기

현재 quantity_loss_mlp18 v2의 기존 완료 작업을 수정하거나 다시 실행하지 않는다.
새 후속 작업은 로컬 completion_v1/complete.json을 기다리며 새 SSH나 학습을 시작하지 않는다.
부모 실패, 완료 증적 없이 기존 완료 process 종료, 상속 절대기한 초과는 action_required다.

부모의 양 host18개 학습·선택 Validation 재현·원본 회수·최종27집계가 모두 확인된 뒤만
새 manager가 prepare를 허용한다. 기존 선택·archive·manifest·원본·결과 SHA를 검증한다.

## 새 원시간 대조를 실행하고 종료까지 연결한다 — 다음 작업

후속 진입점은 paper.scripts.follow_titantpp_raw_time_mlp27이다.
mode check는 LOCAL_READY.json의 현재 파일 SHA와 부모 계약만 확인하고 원격0이다.
mode advance는 단 하나의 followup_v1/attempt.json을 만들며 재실행을 허용하지 않는다.
새 manager/runtime/worker는 paper.scripts의 raw_time_mlp27 이름을 사용한다.
prepare가 아직 실행되지 않은 상태를 학습 시작으로 설명하지 않는다.

부모 전체 완료 뒤 다음 순서로 진행한다.

1. 로컬 준비 파일을 재검증하고 source955·입력6·실험 계약을 새 bundle에 고정한다.
2. 5080/5090 새 격리 경로를 설치한다. 각 host에서 같은 Runtime/GPU·부모 종료·정지T 상태와
   세 손실×3개 동시 합성(9child)을 검증한 뒤만 최대3개씩 학습한다.
3. 새 dispatcher 신원을 한 번 관측하고 START_RECEIPT.json을 보존한다.
   이 증적은 GPU worker의 실제 첫 학습 진입까지 확인했다는 뜻은 아니다.
4. 각 host의 기존 dispatcher와 같은 신원으로 종료를 기다린다.
   모든 학습/Validation child·dispatcher actualexit0 및 명시적 소유 process 부재 뒤
   원본을 host별 한 번 회수한다.
5. 모든 원본 SHA·선택·Validation 재현을 검증한 뒤 저장27개와 신규27개를 로컬에서 집계한다.

실패·timeout·미완료 회수·기존 attempt가 있으면 원문을 보존하고 자동 재시도하지 않는다.
공유 Runtime·driver·인증·기존 모델·원본·checkpoint·실패 기록은 바꾸지 않는다.
외부 Cap27을 재개하거나 신호를 보내지 않는다. 원고와 Test는 작업 범위 밖이다.

## 실제 상태와 결과를 확인한다 — 다음 작업

- FOLLOWUP_STARTED.json: 로컬 일회성 후속 작업의 PID·신원·로그 경로.
- followup_v1/waiting_parent.json: 부모 완료 대기. 새 remote0·training0.
- followup_v1/parent_gate.json: 기존 완료·회수·집계 검증 통과.
- PREPARED.json: 실제 생성된 새 계약·source955·배포 archive SHA.
- START_RECEIPT.json: 새 native qualification 이후 dispatcher의 실제 신원 관측.
- followup_v1/failure.json: 자동 반복 금지와 필요한 사용자 조치.
- comparison_v1/PAIRED_TIME_INPUT_VALIDATION.json/.md와 CSV: 최종3seed·seed별 비교.
- search_artifacts/titantpp_raw_time_mlp27_20261019_v1/retrieval/final_receipt.json:
  status=complete,new_fit_count27,baseline_reference_count27,comparison_model_count54,
  validation_replay_count27,actual_scientific_child_exit0_count54,actual_dispatcher_exit0_count2,
  held_out_test_evaluated=false와 결과/source/archive 바인딩.
- followup_v1/complete.json: 최종 receipt와 모든 결과 SHA 재검증 완료.

반복 자동화는 만들지 않는다. host의 이전 절대기한은 그대로이며 완료 ETA는 보장하지 않는다.
