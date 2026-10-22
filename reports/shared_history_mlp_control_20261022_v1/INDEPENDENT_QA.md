# Independent implementation review

## Review Result: PASS

새 Shared15–single MLP–encoder-only 실험 경로의 로컬 구현 검증을 통과했다. 최종 15개 소스·테스트 파일에서 **303개 테스트와 full Ruff가 실제 exit 0**으로 종료됐다. 그중 독립 테스트는 50개다. 원문 stdout/stderr의 byte 수·SHA256, 테스트 전후 및 현재 15개 파일 SHA를 다시 대조했다. 원래 과학 소스 987개도 모두 기존 SHA와 일치했다.

실제 서버 검증·학습·추론은 이 검토에서 실행하지 않았다. 로컬 PASS는 GPU 검증 성공이나 모델 효과의 증거가 아니다. 프로젝트 역할 설정은 Astra/high이며 실제 provider 모델 ID를 별도로 조회한 것은 아니다.

### CRITICAL
- 미해결 사항 없음.

### HIGH
- 미해결 사항 없음.

### MEDIUM
- 미해결 사항 없음.

### LOW
- 미해결 사항 없음.

## 수정 후 재검증한 사항

- `paper/scripts/run_shared_history_mlp_control.py:1041` — detached NaN 손실이 있어도 일부 gradient만 정상이라 synthetic 승인이 가능했던 문제. 손실·평균 손실·모든 present gradient·step 이후 모든 parameter의 finite 검사를 추가했고 독립 NaN 회귀와 소유 테스트의 gradient/parameter 오염 회귀가 통과했다.
- `paper/scripts/manage_shared_history_mlp_control.py:774` — 실제 dispatcher 종료와 원문 출력 증거 없이 native complete 요약만으로 학습을 허용할 수 있었던 문제. 실제 부모 관측 exit, dispatcher/child 원문 SHA, 고정 신원 부재를 검증하도록 수정했다. 종료 증거 누락·nonzero·출력 변조·생존 신원 회귀가 모두 통과했다.
- 배포 코드 생성 시 `PurePosixPath` 의존성 누락은 소유 에이전트가 수정했다. 생성 코드의 import와 순수 archive 검사 함수만 독립 namespace에서 실행해 정상 archive를 검증했다. 전체 archive 검사가 원격 root 생성보다 앞서는지도 확인했다.

초기 실패한 두 번의 QA 실행 원문은 삭제하거나 성공으로 바꾸지 않았다. 정확한 파일 경로와 SHA는 JSON의 `historical_failed_qa_receipts_preserved`에 남겼다.

## 수용 기준

- [x] causal_equal_information — Independent history-walk oracle covers current/immediate previous observations, eight availability bits, padding and valid-withheld resets; common encoder/head seed equality and target masking are tested.
- [x] capacity_scope — 9600 added parameters are matched in total only. Active-prefix capacity and function classes are not asserted equal; MLP length-only response is independently demonstrated.
- [x] reuse_selection — Fresh MLP9 and original B9/D9 reuse remain distinct. Original payload/source/first strict finite Validation minimum/binary/tensor/actual exits are verified without retagging.
- [x] validation_gate — All 27 individual Validation reproductions and 108 raw receipt/exit references bind source, origin and the same selected checkpoint before cost admission. No new held-out Test route exists.
- [x] native_admission — Nine synthetic plus three capacity child exits and actual dispatcher exit/raw streams/fixed identities are required per host; both native PASS snapshots are required before fit launch.
- [x] finite_qualification — Loss tensor/reduced loss, all present gradients, and all parameters after optimizer step are finite-checked; independent NaN regression and owner contamination regressions pass.
- [x] preservation — All 987 parent source hashes independently match original inventory. Legacy/input/selected/archive preservation gates are implemented; actual full deployment preservation remains a parent execution gate.
- [x] bounded_fail_closed_control — Pinned source/auth/spec/QA/contract, boot/process/GPU guards, 45s read-only check, 900s transfers, cooldown/busy SSH0, 96-round/deadline limits, first failure/partial/attempt preservation and no retries are tested.
- [x] cost_fairness — Train-only 37 batches (5 warmup/32 measure), 162 fresh processes, checkpoint RAM copies and original SHA, GPU/seed strata, allocated/reserved memory and latency/throughput raw recomputation are verified.
- [x] comparison — Pure Validation-only aggregation preserves ddof1, 162 paired metric differences, original source/split, adverse results and undefined empty-tail metrics; no 5080/5090 absolute pooling.
- [x] archive_and_initial_driver — Generated archive checker is executed independently as pure definitions; complete inventory/SHA checks precede exclusive root creation. Both native PASS raw snapshots revalidate files. No remote body is executed by QA.
- [x] complete_cache — Completion cache revalidates full inventories, raw streams, selections, collection and comparison recomputation locally; cache remote calls zero.

## 범위와 한계

- 같은 입력 정보와 총 추가 파라미터 9,600개를 맞춘다. 활성 이력 구간별 유효 용량과 함수 클래스까지 같다는 주장은 하지 않는다.
- MLP9 신규 학습과 원본 B9/D9 재사용을 구분하고, 원본 payload·선택·출처를 바꾸지 않는다.
- 새 held-out Test 실행·예측·성능 열람 없이 Validation27과 Train 비용162를 검증하는 경로다.
- 결과에 따른 모델 선택이나 논문 기여 판단은 이 QA의 결론에 포함되지 않는다.
- 웹/사용자 UI가 없는 로컬 실험 제어 코드라 브라우저 접근성·프론트엔드 성능 항목은 해당하지 않는다.

## 검증 증적과 변경 파일

- 최종 통합 receipt: `reports/shared_history_mlp_control_20261022_v1/local_checks_20261022T145220371808/receipt.json`
- 실제 테스트: 303 passed in 10.79s; pytest exit 0, Ruff exit 0.
- QA 소유 변경: 독립 테스트 파일, 이 QA JSON/Markdown 및 초기 실패 원문 폴더만. 구현 소스와 기존 연구 결과는 수정하지 않았다. `.agents/` 파일도 쓰지 않았다.

## 남은 작업 순서

**로컬 검증본을 봉인하고 원본 보존을 확인한다 — 다음 작업**
- 부모 세션이 이 QA와 동일한 파일 SHA·테스트 receipt를 실행 계약에 결합한다. 실제 입력9·legacy199와 기존 산출물 보존도 배포 전에 확인한다.

**두 서버의 실제 실행 조건을 검증한다 — 다음 작업**
- 5080/5090에서 승인된 synthetic/capacity 검증의 모든 실제 종료와 원문 증거를 확인한다. 양쪽이 통과하기 전에는 새 학습을 시작하지 않는다.

**Validation과 비용 비교를 순서대로 수행한다 — 다음 작업**
- MLP9 신규 학습·B/D18 재사용 뒤 같은 선택본의 Validation27을 재현한다. 전역 검증 이후에만 Train 비용162를 실행·회수하고 독립 집계를 확인한다. MAC은 계속 보류한다.
