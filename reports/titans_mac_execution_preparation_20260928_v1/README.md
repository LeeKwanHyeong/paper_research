# MAC 전용 실행 경로 구현·검증 기록

2026-09-28, `paper_research / codex/hard-lmm-causal-qkv`. 로컬 구현·합성 검증 및 동결 완료. 실제 데이터 학습, 서버 배포, CUDA qualification은 아직 수행하지 않았다.

## 별도 소스에서 MAC 비교를 실행할 준비 — 완료

- 대상: `search_artifacts/titans_mac_observed_time_20260928_v1/source/`.
- 현재 core 학습의 원본 97개 파일은 모두 기존 SHA와 일치한다. 사본에서만 `observed_time.py`와 `constants.py`에 MAC 전용 역할 `observed_time_titans_mac_clip1_v1`을 추가했다. 기존 역할의 허용 모델과 검증은 유지했다.
- 새 파일은 `titans_mac_observed_time_contract.py`, `titans_mac_observed_time_runtime.py`, `run_titans_mac_observed_time.py` 및 전용 테스트다. 원본 공통 모델·head/loss·loader·학습기·MAC 메모리 수식은 바꾸지 않았다. 기존 MAC 실행 최적화와 관련 검증 의존성을 포함한 동결 파일은 총 117개다.
- 학습과 재평가는 동일한 clip1 MAC 생성기와 기존 실행 최적화를 사용한다. 공통 학습기의 생성기 연결은 해당 새 worker 안에서만 적용하고 종료·예외 시 복원한다.
- 비교 대상은 **원래 Titans-MAC 메커니즘을 현재 수량·관측 시간 프로토콜에 맞춘 모델**이다. clip1 안정화와 공통 head/loss가 적용되므로 원 논문의 실험을 그대로 재현했다는 뜻은 아니다.

## 실행 범위와 실패 경계를 코드로 고정 — 완료

| 대상 | 조건 | 새 학습 수 | validation endpoint 재평가 | 제안 상한 |
|---|---|---:|---:|---|
| 5080 | Taxi seed42·52·62 → Intermittent seed42·52·62 | 6 | 12 | 호스트 216시간, 조건당 Taxi 8시간 / Intermittent 64시간 |
| 5090 | Instacart seed42·52·62 | 3 | 6 | 호스트 264시간, 조건당 84시간 |
| 전체 | 각 서버 내부 순차, 서버 간 독립 | 9 | 18 | 합계 최대 480 GPU시간, 첫 호스트 시작부터 최대 264시간 |

위 값은 비용 상한이며 ETA가 아니다. 기존 실험의 승인·시작 시각·예산을 재사용하지 않는다. 각 호스트 permit은 새 캠페인의 공통 마감과 해당 호스트 상한 중 빠른 마감에 묶인다. 준비·qualification·대기·학습·두 endpoint 재평가가 상한에 포함된다.

- batch128, max300/min40/patience40, AdamW lr0.001/weight_decay0.01, 외부 gradient clip1, 내부 associative gradient clip1, 동일 train-only 통계와 입력·초기화 규칙을 고정했다.
- 선택은 validation raw-RMSE의 가장 이른 엄격한 유한 최솟값이다. 선택 및 마지막 checkpoint의 byte/tensor SHA, 재평가 일치, 실제 epoch·step·표본 수·batch 순서를 기록한다.
- 첫 실제 train+validation epoch에서 유한값·모델/역할/clip/source identity를 검사한다. 표본 수와 B/Full의 동일 seed batch prefix까지 통과해야 epoch2로 진행한다. epoch1은 해당 scientific fit에 포함한다.
- B/Full18개 조건의 과거 파일 SHA와 입력 순서를 확인하되 재학습하지 않는다. 외부36 및 현재 core36도 새 실행 목록에 넣지 않았다.
- native qualification은 호스트당 총 60 synthetic optimizer updates, 배치128, 배정된 데이터의 head와 최대 문맥 길이로 실행하도록 구현했다. 5080은 두 데이터 설정 각30회, 5090은 Instacart 설정 60회다. 원격 qualification은 아직 실행하지 않았다.
- worker마다 새로운 프로세스를 사용하고 inherited pipe로 관리자·계약·승인·permit·job을 검증한다. claim 또는 기존 실행 흔적이 있으면 재시도·resume을 허용하지 않는다.
- 5분 무진척 stack dump, 30분 무진척 중단, qualification 30분 상한, 조건/호스트/캠페인 상한을 검사한다. 컴파일 중에도 무진척 상한이 적용된다. Dynamo 64/512 한도와 fullgraph 실행, 오류 억제 금지, 필수 compiled scan 존재 검사를 사용한다. 실패 후 batch 축소나 eager fallback은 없다.
- 시작 시 GPU UUID·Runtime·기존 라이브러리 SHA, 여유 VRAM 12000MiB 이상, 다른 CUDA 작업이 없는지 검사한다. 시작 이후 다른 프로세스가 보이면 관찰만 하며, 실패 시 소유한 자식만 종료한다.
- 완료된 epoch의 노출 증적과 실패 로그를 보존한다. 저장되지 않은 현재 batch의 부분 update 수는 확정하지 않는다. 한 호스트가 실패하면 그 호스트의 뒤 조건을 시작하지 않으며 다른 호스트는 독립적으로 진행한다.

## 합성 데이터로 실제 실행 경로 확인 — 완료

**147 passed, 1 skipped, 17.92초.** 실행 명령·실행 위치·로컬 Runtime·검사 결과는 [verification.json](verification.json)에 기록했다.

- 세 데이터 설정의 실제 공통 학습기 → checkpoint 선택 → 선택/마지막 재평가가 수량 RMSE·MAE와 관측 시간 NLL 기준으로 일치했다.
- 작은 합성 데이터의 40 epoch 전체 worker 내부 경로와 첫 epoch gate, 실제 optimizer step·표본 노출 감사, first40/last30 집계를 검증했다.
- 3데이터×3seed에서 최대 문맥 길이 64/256의 CPU forward를 원래 MAC 구현과 비교했다. target 수량 변경, padding, 중간에 다른 window를 처리한 뒤에도 예측이 누출·상태 잔류의 영향을 받지 않는지 확인했다.
- clip1 및 현재 observed-time head 조합의 최적화 전후 gradient 일치를 검사했다. 기존 메모리 안정화·MAC·관측 시간 테스트도 함께 통과했다.
- 합성 데이터를 사용한 호스트별 60 update qualification 알고리즘, 승인 없는 dispatch 차단, 잘못된 데이터/역할/clip/선택/기한 거부, CPU receipt를 CUDA qualification으로 인정하지 않는 검증을 통과했다.
- GPU를 사용하지 않는 실제 자식 프로세스로 authority pipe·비정상 종료·동일 작업 재시작 거부를 확인했다. 관리자 순차 순서와 중간 실패 시 후속 조건 중단도 검증했다.
- 첫-party Python import는 별도 소스 사본에서 해석됨을 확인했다. Python 파일 구문과 manifest의 모든 SHA를 재확인했다. 변경된 사본 두 파일의 내용은 [parent_copy_changes.diff](parent_copy_changes.diff)에 있다.

CUDA 전용 테스트 1개는 로컬 CUDA가 없어 제외됐다. CPU 합성 테스트의 batch 크기와 입력 규모는 실험 데이터보다 작다. **실제 GPU의 batch128 안정성, 컴파일 소요 시간, 전체 epoch 속도, 성능 결과는 아직 미확인**이다. 이번 기록은 GPU qualification이나 연구 결과가 아니다. 실제 연구 데이터와 held-out 성능은 열람하지 않았다.

## 동결 식별자 — 완료

| 항목 | SHA-256 |
|---|---|
| 설계 계약 | `d59d51c219a2a300a1d295615f578c28239a490a9f50f4aed3c492acafee0b12` |
| 새 실행 계약 canonical | `b9d87e35f6b8ee0f3f6a25365d9b6b0c1c3d5127bb631ef33e377117ebec5897` |
| 새 source closure | `16f94a408a5317ff3ba7100ac462e128c07fab513b6be1333239032af61077ce` |
| 기존 core canonical | `eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d` |
| 기존 원본97 source closure | `e21c45b5f095d258e62998ce818b07bef38ff2cfff6bc408b3594bd1adba9265` |

실행 계약과 전체 파일 목록은 `search_artifacts/titans_mac_observed_time_20260928_v1/frozen_execution/`에 있다. 계약의 `approved:false`는 유지한다. 이 준비 작업으로 실제 실행 승인, start permit, deployment/launch receipt를 만들지 않았다.

## 남은 순서

**현재 core 결과를 완료·취합하고 MAC 비교의 위치를 확정 — 외부 작업 대기**
- 대상: 기존 5080/5090 TitanTPP core 캠페인. 이번 작업에서는 서버에 접속하지 않았다. 최신 진행 상태는 기존 모니터링에서 확인한다.
- 종료 감사·결과 보고가 먼저다. MAC가 해결하는 질문은 공통 프로토콜에서 B/Full/MLP와 원래 MAC 메커니즘의 차이다. 순수 메모리 효과나 상호작용을 이 비교 하나로 확정하지 않는다.

**새 9조건 GPU 실행을 결정 — 승인 필요**
- 대상: 위 5080 6조건·5090 3조건 및 동결 계약. 완료된 core 결과를 보고 이 범위를 새로 승인한다. 승인에는 core 결과를 검토했다는 사실과 로컬 종료 감사 파일 SHA가 포함돼야 한다.
- 새 실행 root에는 `prior_core_terminal_audit.json`을 배치한다. 이 파일은 원 core 계약 SHA, `collected:true`, `user_results_reported:true`, 두 호스트의 `terminal:true`, `owned_processes_exited:true`, 각 `terminal_evidence_sha256`를 기록해야 한다. wrapper가 승인에 적힌 파일 byte SHA와 이 필드를 검사한다.
- 승인 후 실제 최초 시작·호스트 시작 시각과 고정 마감을 permit에 기록한다. 이 문서는 실행 승인이 아니다.

**승인된 사본을 배포하고 native qualification부터 실행 — 다음 작업, 승인 후**
- 대상: 계약에 고정된 각 서버의 새 root/Python/GPU/tmux. source·기존 Runtime을 대조하고 qualification을 거친 뒤 조건별 fresh worker를 실행한다.
- native 실패나 drift는 해당 호스트 중단으로 기록한다. 자동 재시작하거나 현재 학습 Runtime을 재배포하지 않는다. 모든 결과·비용·실패 조건을 보존해 validation-only로 보고한다.
