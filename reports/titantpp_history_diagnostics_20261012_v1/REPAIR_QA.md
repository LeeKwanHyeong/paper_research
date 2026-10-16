# 진단 계측 v1 실패와 별도 수정 실행 QA

## Review Result: FAIL — v1 실제 실행

상태: v1 실행 실패 확인. 기존 PRELAUNCH_QA.md는 당시 로컬 검토 기록으로 보존한다. 이 보고서가 실제 실행에서 확인한 결함과 기존 검토의 한계를 추가한다. 세 arm의 synthetic forward 비간섭 테스트는 실제 dataset 메타데이터 연결 경로를 실행하지 않았으므로 이번 결함을 검출하지 못했다.

### CRITICAL

- 없음.

### HIGH

- `search_artifacts/titantpp_history_diagnostics_20261012_v1/control/evaluate.py:267` — :258–259에서 만든 eligible Validation entity 집합을 `None`으로 다시 덮어써 :353의 `entity in validation_entities`에서 TypeError가 발생한다. Validation 첫 batch에서 full·보정 제거·current-only가 모두 실패했고, 같은 경로의 Test도 영향을 받는다. 수정은 후속 재할당을 제거하고 아래처럼 집합을 한 번 계산한 뒤 worker가 그대로 사용하도록 한다. 양 split·Validation 대상이 없는 series·첫 행만 Validation인 series·후속 재할당 없는 worker 연결을 회귀 검증해야 한다.

```python
validation_entities = eligible_validation_entities(dataset)
# Do not assign None after metadata selection.
...
series_in_validation = entity in validation_entities
```

### MEDIUM

- 없음.

### LOW

- 없음.

## 실제 실패 증적

- v1 canonical contract: `38fb8820071fdb9b13f1576528d03da952a5a81a7e51b1a9db5d0cd97553ace4`.
- v1 operation seal rawSHA: `83006f2d337462a6396e50a7f15b791ef67aa6cf1232d5ad249342a24a9c42e7`.
- `WAIT_FINALIZE_STDOUT.jsonl`에는 native synthetic 3개 PASS, 첫 Intermittent seed42의 세 arm 실제 exit 1, Validation 완료 0, Test 완료 0, pipeline 실제 OS exit 1을 기록했다.
- 같은 관측에서 owner pipeline PID 24331 부재와 GPU compute process 목록 공백을 기록했다. 이는 로컬에 저장된 원관측 증적이며 QA가 원격 조회를 다시 수행하지 않았다.
- `WORKER_FAILURE_OBSERVATION.json`에는 세 worker가 동일한 TypeError로 실패했고 완료 receipt를 쓰지 않았다는 traceback과 failure rawSHA가 보존돼 있다.
- 과학 성능 실패나 checkpoint/source mismatch로 해석하지 않는다. 원래 선택 checkpoint·원래 Test 결과의 유효성 판정을 이번 운영 메타데이터 결함으로 변경하지 않는다.
- 신규 학습·optimizer update·새 평가 범위 추가 없음. v1의 실패·claim·부분 증적을 삭제하거나 덮어쓰지 않는다.

## 인접 데이터 매핑 검토

기존 source129의 `data_loader/event_seq_data_module.py`에서 target은 각 series의 두 번째 행부터 eligible하며 `(part, end)`의 실제 target은 `end+1`이다. worker의 offsets, target metadata 행, 직전 입력 행, lookback window의 `searchsorted`, max_len-1 이력 절단 계산을 이 원래 구현과 대조했다. 검토한 범위에서 추가로 확인된 매핑 결함은 없다. 대상 identity와 truth digest 및 원래 여섯 지표의 전수 재현 gate는 그대로 유지해야 한다.

## 수정 실행 acceptance checklist

- [x] v1 실패 원인과 실제 exit를 로컬 원관측 증적으로 확인.
- [x] 기존 PRELAUNCH_QA와 v1 봉인·원본·claim/failure 보존.
- [x] membership 수정 및 양 split 회귀 테스트 통과.
- [x] 새 v2 계약·operation seal·새 root를 별도로 대조. 기존 27개 checkpoint, source129, 데이터, Runtime, strata 및 54개 평가 범위 불변 확인.
- [ ] v2 native 3개 비간섭 probe를 통과한 뒤 Validation27 전체 실제 exit 0 및 재현 gate, 이어서 Test27 실행.
- [ ] v2 실제 종료·원본 회수·archive/manifest SHA 재검증 후 FINAL_QA.

별도 v2는 사용자 승인 범위 내 결함 수정 후의 명시적 새 실행이며 v1의 자동 retry/resume이나 실패 지우기로 기록하지 않는다. v1 실패를 v2 성공으로 소급 변경하지 않는다.

증적 rawSHA WORKER_FAILURE_OBSERVATION.json: `a3b7eba6fc7bf1d67cc6af1549751f508acbf07d7d8f7db83046810c658f26ab`

증적 rawSHA WAIT_FINALIZE_STDOUT.jsonl: `17f2516e52db7816b76b8a2b4ba5771a286b21145471d9e60b35a697a87eb4fb`

## 수정 코드 Review Result: PASS — 새 봉인 대조 전

위 HIGH 결함은 v1에 남아 있는 실제 결함이며 새 수정 evaluator에서는 해소됐다. 수정은 `paper/scripts/evaluate_titantpp_history_diagnostics.py:170`의 `prepare_diagnostic_population` 추출과 :285의 단일 반환값 연결로 한정된다. 원래 loader·입력 target·forward·metrics·digest를 바꾸지 않았다. QA의 AST 검사에서 worker의 `validation_entities` Store는 한 번뿐이며 후속 덮어쓰기는 없다.

- 수정 evaluator rawSHA: `087a181e00f46bb4a36f37ce3df9929a75094c39a7f6359bebba4c03c5364d25`.
- telemetry rawSHA는 기존 `656738b7406e45ac828f60ad9e56c2151e04dd4f4fbf7bdfacb5d37fc2f2b3d7` 유지.
- 회귀 테스트 rawSHA: `78dfc696e096a6564ba767afb97c88ec1cff992333ea07ba17b0dce16f162e23`.
- 구현 담당자 증적: unittest 6개 PASS, 1.282초. QA는 테스트 내용을 직접 검토하고 코드/test SHA를 독립 확인했다.
- 새 두 테스트는 실제 frozen loader를 fabricated Validation/Test frame으로 실행하고, 첫 행만 Validation인 series·Validation 대상이 없는 series·singleton·offset/target 연결을 확인한다. 실제 worker 테스트는 model/runtime를 fixture로 대체하고 membership sentinel에 도달함을 확인해 후속 None 덮어쓰기 재발을 검출한다. 이것은 실제 평가 결과나 native admission 증적이 아니다.
- v2 canonical `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d` 계약 diff를 독립 확인했다. jobs·datasets·diagnostics·Runtime·sources는 v1과 같고, 차이는 새 격리 root·생성 시각·instrumentation revision·v1 실패 provenance이다.

수정 코드에 미해결 CRITICAL/HIGH/MEDIUM/LOW는 없다. v2 operation/transport 봉인과 실제 실행 단계는 아직 별도 확인해야 한다.

## v2 봉인된 실행 전 Review Result: PASS

- canonical contract: `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d` — 독립 canonical 재계산 일치.
- operation rawSHA: `236ad7a57096bd91811bf8ad0cc1a6b389c43b3272ac8520687683b7b8458682` — 11개 등록 파일 전부 SHA 일치.
- transport rawSHA: `c690f9b7d7715a498ba18829b3228302062543d840ab8ade8e518bf0257cc54b` — 67개 유일한 regular member의 내용이 bundle과 SHA 일치.
- input 359개 SHA 일치. v1과 비교한 input seal 내용 차이는 새 `contract.json` 한 개뿐이다. 원래 v1 evaluator SHA도 유지됨을 확인했다.
- 수정 evaluator는 앞서 검토한 `087a181e…`, telemetry는 기존 `656738b7…`이며 pipeline은 기존 `981f293f…`다.
- setup/finalizer의 차이는 고정 root의 v2 변경뿐이다. launch는 v2 root와 별도 v2 wait channel을 사용한다. Python control AST 검사와 `bash -n` 통과.
- `provenance/repair_decision.json`을 operation seal로 결합했다. v1 native3 PASS, 부분 Validation 시도3, 완료 Val0/Test0, 실제 pipeline exit1 및 모든 claim/failure 보존을 명시한다.
- 새 원격 root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v2_5080`.

현재 v2에 미해결 CRITICAL/HIGH/MEDIUM/LOW는 없다. 이 봉인으로 실행 전 검토를 통과했다. 실제 native probe, Validation27/Test27와 원본 회수는 실행 후 FINAL_QA 대상이며 여기서 성공으로 선판정하지 않는다. v1 실제 실행 판정 FAIL은 그대로 유지한다.
