# 5090 J/Q/T 학습 실행 — 2026-09-10

**5090의 J/Q/T 학습9개가 모두 정상 종료됐다.** 실행은2026-09-10 11:41:29 KST에 시작해 **2026-09-11 08:33:27 KST**에 완료됐다. 모든 arm120 epoch와 총6,816,240 step·동일 표본 노출·최종 paired comparison 검증이 통과했고, 시간별 점검은PAUSED로 변경했다. [완료 확인](monitor/20260910T233614Z_check.md).

현재 증적: [실행 확인서](execution_receipt.json), [초기 학습 로그·PID](initial_training_verification.json), [실행 명령](launch_receipt.json).

**세 데이터셋 결과 통합과 해석 — 완료**

- [통합 해석 보고서](three_dataset_validation_v1/validation_report.md), [재계산 수치와 학습 이력](three_dataset_validation_v1/validation_results.json), [검증 기록](three_dataset_validation_v1/validation_receipt.json)에 Intermittent·Taxi·Instacart의 최종 결과를 보존했다. 기존 두 데이터셋 보고서는 당시 분석 증적으로 유지한다.
- [대화형 보고서](http://127.0.0.1:4173/)에서 데이터셋별 표와 학습 곡선을 볼 수 있다. [로컬 실행 안내와 화면 검증](three_dataset_validation_v1/README.md)을 함께 남겼다.
- 최적 수량 selector에서는 Q가 세 데이터셋 모두 J의 raw RMSE를 개선하지 못했다. Instacart T에는 작은 시간 loss 개선이 있다. 최종 epoch 120에서는 Taxi와 Instacart의 Q/T가 J보다 좋으므로 비교 시점을 구분한다.
- Taxi J의 후기 구간에서는 validation 수량 log-MSE가 1.90% 낮아졌지만 raw RMSE는 7.35% 높아졌다. 완전 분리를 본모델 구조로 확정하기 전에 수량 목표와 원단위 오차의 관계 및 시간 loss의 후반 악화를 진단하는 것이 다음 순서다. 원인과 benchmark 우월성은 아직 확인되지 않았다.

**Intermittent·Taxi 통합 검증 — 완료**

- [통합 검증 보고서](intermittent_taxi_validation_v1/validation_report.md)와 [전체 epoch 대응 수치](intermittent_taxi_validation_v1/validation_results.json)를 추출했다. 두 데이터셋·6개 학습의 계약·완료 증적·selector 검증이 통과했다.
- 최적 selector 기준에서는 두 데이터셋의 Q 수량 RMSE·T 시간 loss 개선이 없지만, **Taxi는 동일 최종 epoch120에서 Q·T가 J보다 좋다.** 최적 checkpoint와 동일 최종 step 비교를 구분하며, 단독 학습이 항상 나쁘다고 해석하지 않는다.
- Intermittent의 수량 구간·이력 길이별 오차와 Taxi의 후반 시간 loss·clipping 변화는 후속 원인 확인 대상으로 남는다. 이번 추출에는 새로운 모델 평가나 학습을 포함하지 않았다.

**현재 진행 확인 — 2026-09-11 08:36 KST, 완료**

- **08:33:27 KST에 suite가정상종료**했다. 세 데이터셋의J/Q/T **9/9 arm 모두120 epoch**를 마쳤고 총6,816,240 step이다. 실제 suite 경과시간은약20시간52분이다.
- 동결 source65파일·suite·계약·runtime과 완료 receipt가 일치한다. 모든 arm의초기화·120 epoch 표본/순서/step·selector·paired comparison 감사가 통과했고, 예상 학습 process와GPU 점유가 종료됐다.
- 원격 SHA와 동일한JSON39개 및로그 tail4개를 [완료 증적](monitor/20260910T233614Z_terminal/collection_manifest.json)에 보존했다. 수집한 로그에서 실행 오류를 발견하지 않았다.
- **J/Q/T Scheduler는PAUSED**로 변경·확인했다. 기존 Intermittent 평가 Scheduler도PAUSED를 유지한다. [일시정지 기록](monitor/20260910T233614Z_scheduler_paused.json).
- [완료 확인 보고서](monitor/20260910T233614Z_check.md), [최신 상태](monitor/latest.json). 이후 세 데이터셋 통합 해석까지 완료했으며, 다음은 표본별 원인 진단 계약 작성이다.

**5090 source와 runtime 연결 — 완료**

- 사용자의 “5090으로 진행하자.” 요청을 이번 실행의 서버 선택과 새 학습 승인으로 적용했다. 이전 5080 준비 문서와 기존 scheduler를 새 학습 승인으로 재사용하지 않았다.
- 기존 CPU 구현의 source 62개는 모두 byte 단위로 보존했다. [보존 확인](frozen_cpu_preservation.json).
- CUDA 변경은 별도 저장소 [source](../../search_artifacts/jqt_cuda_5090_20260910/source)에 구현했다. 브랜치는 `codex/jqt-cuda-5090-20260910`, 커밋은 `76426fd981e624144947fd582f794c2832e9fe05`다. 원래 `paper_research` 저장소의 브랜치·기존 변경은 유지했다.
- 커밋 bundle을 새 실험 전용 경로에 배치하고 SHA·revision·clean 상태를 대조했다. [로컬 커밋](source_commit_v1.json), [5090 배치 확인](deployment_v1.json). 기존 실험 source·결과·checkpoint는 덮어쓰지 않았다.
- 실제 runtime: RTX5090, Python3.12.13, PyTorch2.11.0+cu130, CUDA13.0, cuDNN92000, NumPy2.1.3, Polars1.39.3, CPU thread4. GPU UUID·driver595.84·compute capability12.0과 결정적 연산·TF32 off·CUBLAS workspace 설정을 계약에 기록했다.

**재현 검증과 실제 표본 계약 확인 — 완료**

- [로컬 테스트](local_pytest.txt): **91 passed**. 변경한 source와 테스트의 Ruff도 통과했다.
- [실제 CUDA 검증](cuda_qualification_v1.json): 약 28초, 모든 필수 검사 통과. J/Q/T 각각 합성 데이터에서 2 epoch 연속 실행과 별도 프로세스의 1+1 epoch 재개를 비교했다. 모델·optimizer·RNG·history·selector state가 정확히 일치했다.
- batch128 × 길이64/256에서도 원래 공동 objective·gradient와 일치하고, target/padding 인과성이 유지됨을 확인했다. 비활성 head 불변·NaN 거부·동일 초기화와 step·batch 순서도 검증했다.
- [실제 데이터 계약 확인](remote_receipts/contract_validation_receipt.json): 세 데이터셋의 data/split SHA, 정확한 train·validation target population과 초기화 manifest가 통과했다. 계약 확인 자체에서는 학습하거나 validation 성능을 계산하지 않았다.
- [최종 suite 검사](suite_prelaunch_validation.json): source·runtime·qualification·세 계약의 SHA가 연결된다. 완료를 판정하는 부모 프로세스의 코드 import가 GPU를 점유하지 않는 것도 확인했다.
- [독립 검토](launcher_independent_review.md)의 완료 감사·시간 제한·lock 보존 지적을 수정했다. [수정 확인](review_resolution.json).

**동일 budget J/Q/T 학습 — 완료**

| 실행 순서 | 데이터셋 | arm | epoch/arm | optimizer step/arm |
|---|---|---|---:|---:|
| 1 | Intermittent | J → Q → T | 120 | 369,240 |
| 2 | Taxi | J → Q → T | 120 | 36,000 |
| 3 | Instacart | J → Q → T | 120 | 1,866,840 |

공통 조건은 seed42, batch128, static B, 가중치1의 수량 log-MSE, legacy cap300 time loss, AdamW lr0.001/weight-decay0.01/clip1이다. 세 데이터셋·9개 arm을 한 번에 하나씩 실행한다. early stopping과 자동 retry/resume는 없다. 시간 점수는 정규화된 Time NLL이 아닌 legacy clamped loss다.

- [동결 suite 계약](remote_receipts/suite_contract.json), [실행 정책](execution_policy.json).
- tmux: `jqt_seed42_5090_20260910`.
- source: `5090:/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/jqt_cuda_seed42_20260910/source`.
- 결과: `5090:/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/jqt_cuda_seed42_20260910/suite_run`.
- 상태: 결과 경로의 `status.json`. dataset별 로그와 J/Q/T의 epoch checkpoint·history·두 selector 결과를 저장한다.
- 초기 확인 PID: launcher `1958891`, Intermittent child `1958962`. 최초 확인에서는 J/epoch1/global_step1과 GPU process의 PID 일치까지만 확인했다. 이후 진행은 위 정기 점검 기록을 따른다.
- **총 72시간 상한**은 CUDA qualification 시작부터 계산한다. 종료 한도는 **2026-09-13 11:36:47 KST**다. suite 전체의 absolute·monotonic deadline을 공유하고 timeout 시 직접 생성한 child process group만 종료한다. 불완전하거나 군별 노출이 다른 결과를 완료 비교로 처리하지 않는다.
- 과거 5090 B 실행 시간에서 환산한 약22.14시간은 당시 참고치다. 이번 J/Q/T suite의 실제 완료시간은 약20시간52분이며 qualification을 포함하면 약20시간57분이다.
- 초기 실행 확인에서는 설정·GPU process·첫 학습 진입까지만 확인했다. 이후 사용자의 “그러면 1시간 Scheduler 걸어두자” 요청으로 이 작업에 `5090 J/Q/T 학습 점검` heartbeat(`5090-j-q-t`)를 **ACTIVE**, 1시간 간격으로 생성하고 저장된 주기·연결 작업을 확인했다. 기존 `Intermittent 수량 평가 점검` scheduler는 **PAUSED**로 유지한다. [Scheduler 확인서](monitor_scheduler_receipt.json).
- 새 scheduler는 기존 실행의 상태·로그·소형 결과만 확인한다. 사용자의 답변 누락 지적을 반영해 **정상 진행이나 단계 변화가 없어도 매시간 확인 시각·현재 단계·완료 epoch·전체 완료 학습 수·현재 단계 ETA·오류 여부를 짧게 보고**하도록 변경하고 저장된 설정을 확인했다. [보고 조건 변경 기록](monitor_scheduler_reporting_update.json). 이전 점검의 알림 생략 기록은 당시 동작의 증적으로 보존한다. 새 학습·자동 재시작·예산 연장은 하지 않으며, terminal 상태를 기록·보고한 뒤 자체적으로 PAUSED로 변경한다. 정기 확인에는 이 컴퓨터와 Codex 앱이 실행 중이어야 한다.

**학습 결과의 동일 노출·selector 비교 — 완료**

- 세 데이터셋·9개 arm의120 epoch·계약상 step·표본 수·batch 순서·selector와 paired comparison을 감사했다. [최종 검증](monitor/20260910T233614Z_check.md).
- 수량은 J의수량 selector와Q, 시간은 J의시간 selector와T를 비교했다. 서로 다른 J checkpoint를 하나의동시 성능으로 합치지 않았다.
- 원본 데이터 재평가·checkpoint 실행·held-out 평가·추가 seed·재실행은 수행하지 않았다. 세 데이터셋 통합 해석을 완료했고 수량 구간·이력 길이별 오차와 시간 후반 악화의 진단 계약 작성이 남아 있다.

**분리 구조와 benchmark 설계 — 진단 결과 후 다음 작업**

- 반복되는 이득이 확인된 뒤 capacity·전체 학습/추론 compute를 통제하고 RMTPP·THP에도 같은 분리 기법을 적용할 범위를 정한다.
- 지금의 실행 시작이나 단일 seed validation 결과를 논문의 우월성 증거로 승격하지 않는다.
