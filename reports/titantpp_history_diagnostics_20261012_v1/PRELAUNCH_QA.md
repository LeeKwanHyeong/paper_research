# 저장 모델 진단 계측 실행 전 QA

## Review Result: PASS

- 상태: **로컬 실행 전 검증 완료**. 실제 native GPU qualification, Validation 27개, Test 27개 및 회수 후 최종 검증은 아직 완료 판정 대상이 아니다.
- 범위: 기존 선택 checkpoint 27개를 Validation·Test에서 각각 계측하는 총 54개 평가. 신규 학습 0, 구조 개입 0, checkpoint 재선택 0, 자동 retry/resume 0.
- 검토 역할: 프로젝트 `.codex/agents/qa-reviewer.toml`의 Astra/high 설정 확인. `oma-qa`와 `TEST_SESSION_PROTOCOL.md` 0·1·9·12절 적용.
- QA 변경 파일: 이 보고서만. 소스 코드·과학 source·checkpoint·`.agents/` 수정 0, 원격 명령 0, QA 자체 모델 추론 0.

## 요약

새 격리 root와 별도 계측 봉인이 기존 source129·선택 checkpoint27·Runtime을 보존한다. 계측 hook은 원래 forward 반환값을 바꾸지 않으며 원래 평가의 대상·truth digest·전체 및 큰 수량 여섯 지표 재현을 검사한다. 모든 Validation 평가의 실제 exit 0과 재현 PASS가 모이기 전에 Test 평가를 실행할 수 없다. 최초 검토 중 발견한 자체 집계 지표 검증, helper 봉인 검사, 결과 schema 표기 보완은 최종 봉인 코드에서 해소됐다.

이 PASS는 **검토한 봉인 버전의 실행 전 조건 충족**을 뜻한다. 실제 54개 평가의 성공, 계측 결과의 과학적 해석, 원격 원본 회수 완료를 뜻하지 않는다.

### CRITICAL

- 없음.

### HIGH

- 없음.

### MEDIUM

- 없음.

### LOW

- 없음.

## 검토 대상과 근거

- `search_artifacts/titantpp_history_diagnostics_20261012_v1/contract.json`, `approval.json`, `input_integrity.json`, `operation_integrity.json`, `deployment/reuse.json`.
- 위 bundle의 `control/evaluate.py`, `control/original_evaluate.py`, `control/titantpp_history_diagnostic_telemetry.py`, `control/pipeline.py`, `control/setup.py`, `control/finalize.py`, `control/launch.sh`.
- `reports/titantpp_history_diagnostics_20261012_v1/prepare_bundle.py`, `test_diagnostic_telemetry.py`, `DEPLOYMENT_PACKAGE.json`, `INITIAL_5080_HEALTH.json`와 transport archive.
- 기존 Test bundle `search_artifacts/titantpp_history_instacart_test_20261012_v1`의 계약·완료 receipt·archive·manifest·선택 checkpoint·native 평가 receipt.
- `evaluate.py:67` 계약 및 27개 작업 범위 검증, `evaluate.py:90` 원래 receipt 결합, `evaluate.py:116` 전체 Validation gate, `evaluate.py:145` operation 파일 봉인 검증, `evaluate.py:156` 계측 자체 집계의 여섯 지표 재현, `evaluate.py:170` 평가 worker, `evaluate.py:442` native synthetic probe.
- `pipeline.py:196` native qualification·hooked/unhooked 원래 출력의 bitwise equality·계측 finite 검사를 통과 조건으로 요구한다.

## 고정 식별자

| 항목 | SHA-256 / 고정 값 |
|---|---|
| 새 canonical contract | `38fb8820071fdb9b13f1576528d03da952a5a81a7e51b1a9db5d0cd97553ace4` |
| 새 operation integrity rawSHA | `83006f2d337462a6396e50a7f15b791ef67aa6cf1232d5ad249342a24a9c42e7` |
| evaluator | `26bc40e3f6939a3442ef55999224416a897cee7ea4104fd86e11f8ed561461a8` |
| telemetry | `656738b7406e45ac828f60ad9e56c2151e04dd4f4fbf7bdfacb5d37fc2f2b3d7` |
| pipeline | `981f293fc20d196480996edea5bee7919c2e87af54893887ef0aed7a6567ed67` |
| setup | `1421d2ae55254b112a77ec67227a70293001ecd57cab740909f3c9b63d86205d` |
| finalizer | `8152ae26bd9b522de236e0149089bd5a454a6e221a32f12bc147afe582813976` |
| launch | `f37a34ecea5c4d34b550eaa2edc93811be1ac42cbdae4c3ec9f41c8f525c2fcc` |
| transport archive | `736f7c20f564608c0158626fa9ba57174fcffddf0eb03933b3722c967ebdf46c` |
| 원래 Test canonical | `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c` |
| 원래 Test completion receipt | `d21d137190e98552d07c5b9e2d8fc31151846e39c782ad907ff9a1a3cf644dbd` |
| 원래 archive | `bf7aa41942420966dc612cd1b8a8806733cf38abeb373dea127558782dbfb3ca` |
| 원래 manifest | `b25c0e2c6e39bc8f15d2f7569ad81ed1587c8d1fdfcb907fd1bb451ef135ff27` |
| source129 closure | `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a` |
| frozen science revision | `57da14a2a7a5687e1bdef7d3164f3e8c0051da59` |
| GPU UUID | `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6` |

원격 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v1_5080`, Python은 `/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12`다. 고정 Runtime은 Python 3.12.13, Torch 2.11.0+cu130, CUDA 13.0, cuDNN 91900, NumPy 2.4.4, Polars 1.39.3이다.

## 독립 로컬 검증 결과

- [x] 원래 archive rawSHA 및 manifest의 929개 파일 SHA 일치.
- [x] 원래 과학 source129 및 선택 checkpoint27 SHA 일치.
- [x] 새 계약 canonical SHA 독립 재계산 일치. 기존 history27 작업·source·Runtime 계약과 정확히 일치.
- [x] Taxi·Intermittent·RAF × seed42/52/62 × full·보정 제거·current-only의 27개 ID 중복 없음.
- [x] 새 input seal 359개 파일 SHA 일치. 원래 Validation/Test receipt54개와 byte identity 확인.
- [x] operation seal rawSHA 및 10개 등록 파일 SHA 모두 일치.
- [x] transport rawSHA 일치, 66개 유일한 regular member 각각 bundle 원본과 SHA 일치. 재사용 staging 304개 항목이 input seal과 정확히 일치.
- [x] control Python 6개 AST parse 통과. launch shell `bash -n` exit 0.
- [x] QA의 fabricated scalar 2개 대상 집계로 전체·tail count, RMSE·MAE·TimeNLL, signed error 및 feature 공동 모멘트 검증 통과. 실제 데이터·모델 사용 없음.
- [x] 구현 담당자의 CPU synthetic focused test 4개 PASS 증적 확인(10.288초). 세 arm의 nonzero CNN/GRU fixture에서 원래 출력 전체의 bitwise equality, 상태 SHA·gradient 불변, qualification 계측 제외와 hook 해제 검사 포함. QA가 이 모델 테스트를 별도로 재실행한 것은 아니다.

## 계약·실행·해석 acceptance checklist

- [x] 기존 checkpoint와 science source129 보존. 원래 root를 읽어 복사하고 새 root에 staging하며 덮어쓰기·hardlink를 사용하지 않는다.
- [x] 새 root가 이미 존재하거나 claim/failure가 있으면 중단한다. 봉인 mismatch, GPU identity mismatch, Runtime mismatch는 실패로 보존한다.
- [x] tar의 절대 경로·상위 경로·중복·특수 파일을 거부한다. 재사용 경로의 root 이탈과 symlink 이탈을 거부한다.
- [x] evaluator 시작·종료 때 모든 operation/input/source/checkpoint SHA를 대조한다. checkpoint tensor state SHA가 평가 전후 같아야 한다.
- [x] hook은 detach한 실제 출력을 관측하고 원래 forward 반환값을 교체하지 않는다. `.eval()`, `requires_grad_(False)`, `torch.no_grad()` 적용.
- [x] full9에는 실제 CNN/GRU 잔차 norm을 기록한다. 대조군의 존재하지 않는 이력 잔차를 0으로 조작하지 않고 적용 불가로 구분한다.
- [x] series 및 좋은/나쁜 seed와 모든 대조를 포함한다. input-known 이력·시간 구간은 Train 기준으로 고정하며 target 수량·gap 구간은 결과 기술용으로 구분한다.
- [x] 기존 원래 지표와 계측 자체 전체/tail 집계의 count 및 여섯 지표를 1e-5 허용오차로 비교한다. 대상 identity와 truth digest는 정확 일치를 요구한다.
- [x] native synthetic probe는 세 arm 3개이며 checkpoint·실제 데이터 읽기 0이다. 별도 18개 추론 probe를 추가하지 않는다.
- [x] native probe의 실제 exit 0·bitwise 출력 일치·finite·Runtime/GPU 검증 뒤에만 Validation을 시작한다.
- [x] Validation27 실제 exit 0·재현 PASS의 전체 gate를 먼저 봉인한다. Test27은 해당 gate와 개별 receipt·telemetry·exit SHA를 재검증한다.
- [x] 같은 dataset/seed 세 arm 최대 3개 동시 실행. subprocess PID·PPID·정확 argv·실제 exit를 기록하며 자동 retry/resume하지 않는다.
- [x] pipeline의 실제 OS exit를 launch wrapper가 기록한다. finalizer는 27/27 완료·실제 exit 0·owner/GPU PID 부재·qualification·input 봉인을 검사한다.
- [x] finalizer가 telemetry rawSHA 및 state_after_sha256을 검사하고, exclusive 회수 claim 뒤 archive member 전체 SHA와 archive/manifest SHA를 검증한다.
- [x] 결과 schema는 `titantpp_history_diagnostic_combined_v1`, scope는 `test_diagnostic_replay`다. 이미 노출된 Test의 진단이며 새 독립 평가로 표현하지 않는다.
- [x] 계측 상관·norm은 인과 원인, signed raw 수량 기여, 특정 head 제거 효과의 증명으로 확대하지 않는다.
- [ ] 실제 native GPU probe 3개 통과 및 실행 시점의 owner/GPU idle 재확인.
- [ ] Validation27 전체 실제 exit 0·원래 대상/지표 재현과 Test 시작 전 gate 순서 확인.
- [ ] Test27 전체 실제 exit 0·대상/지표 재현 및 진단 coverage 확인.
- [ ] 실제 pipeline 종료, owner/GPU PID 부재, immutable 원본 회수, 로컬 archive/manifest 재검증 후 FINAL_QA.

## 현재 건강 증적의 범위

`INITIAL_5080_HEALTH.json`의 원관측 Unix 시각 `1791767872.7438245`에는 새 root 부재, 고정 GPU UUID, compute PID 0, reboot-required false, 패키지 작업 부재, 기존 parent 36/36 완료와 실제 pipeline exit 0을 확인했다. 이는 기록된 시점의 관측이며 실행 시점까지 idle을 보장하지 않는다. setup과 pipeline의 실행 전 gate를 그대로 유지한다. QA는 원격 상태를 추가 조회하지 않았다.

## 절차상 범위와 남은 순서

요청된 `.agents/skills/_shared/runtime/execution-protocols/codex.md`와 개인 skill root의 같은 경로는 존재하지 않았다. 대신 사용 가능한 oma-qa 실행 지침과 프로젝트 세션 규칙을 읽었으며, `.agents/` 쓰기 금지에 따라 지정된 reports 경로에만 기록한다. 이 작업은 Python batch 계측이므로 웹 UI 접근성 검사와 npm audit는 적용 대상이 아니다.

**5080 native 실행 조건 확정 — 다음 작업**
- root 담당자가 봉인된 setup 및 pipeline의 건강·GPU·Runtime 조건과 세 synthetic probe를 확인한다. 이 게이트가 통과해야 Validation 평가를 시작한다.

**원래 지표 재현 뒤 진단 계측 완성 — 다음 작업**
- Validation27 전체 재현 gate를 먼저 확정하고 Test27을 실행한다. mismatch 또는 실제 실패는 증적을 보존하고 자동 재시도하지 않는다.

**원본 회수와 원인 해석의 범위 확인 — 다음 작업**
- 실제 종료·PID 부재·회수 SHA 확인 뒤 FINAL_QA를 작성한다. series·이력·시간 head·잔차 특성과 오류의 연결을 탐색적 진단으로 해석하며, 최소 추가 실험 필요성은 그 후 판단한다.

보고서 작성 KST: 2026-10-12T10:30:12+09:00
