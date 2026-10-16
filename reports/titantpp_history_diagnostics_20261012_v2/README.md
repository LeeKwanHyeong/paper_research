# 선택27개 모델의 오류·보정 출력 진단 계측

**v2 실행·분석·독립 QA 완료.** 2026-10-12 **10:51:06 KST**에 Validation27·Test27, worker54개의 실제 exit0와 pipeline 실제 OS exit0을 확인했다. 원래 여섯 지표·대상 digest·가중치 재현, 원본746개와 archive·manifest의 로컬 SHA 검증 및 [최종 QA](FINAL_QA.md)가 통과했다. 신규 학습·경로 개입은0이다. [핵심 진단과 다음 순서](FOCUSED_INTERPRETATION.md), [실제 완료 상태](RUN_STATUS_COMPLETE.json), [전체 수치 보고서](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/analysis/DIAGNOSTIC_REPORT.md).

v1의 운영 계측 오류와 첫 Validation3worker의 부분 시도, 완료 Validation0/Test0 및 실제 exit1은 원래 폴더에 보존했다. 회귀 검증과 새 봉인 뒤 별도 v2 root에서 같은27모델/54완료 split 범위를 수행했다. 과학 코드·모델·입력·원래 지표는 불변이다. [v1 실패·복구 판단](../titantpp_history_diagnostics_20261012_v1/REPAIR_DECISION.json).

사용자의 2026-10-12 **“진행해보자”**는 직전 제안의 기존 선택 checkpoint27개, Validation/Test54개 split 진단 계측을 승인한다. [앞선 최소 계획](../titantpp_test_reversal_diagnosis_20261012_v1/MINIMUM_EXPERIMENT_PLAN.md)의 단계1만 실행한다. 새 학습·모델 경로 우회·검색·새 자료·Test epoch 재선택·공용 Runtime·인증 변경·원고 편집은 포함하지 않는다.

**원본·입력·과학 소스를 확인한다 — 완료**

- 기존 Test36의 원본929개·archive, history source129와 선택 checkpoint27개의 로컬 SHA를 독립 검증했다. Taxi·Intermittent·RAF×seed42/52/62×full/보정 제거/current-only를 유지한다. 이전 Test 계약은 `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c`다.
- 새 계약은 `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d`이고 과학 source129 closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`를 유지한다. 원래 Validation 선택과 기존 Test의 노출 이력은 그대로다.
- 새 입력 경계는 적격 Train 대상의 이력 길이·직전 관측 gap 분위수로 고정했다. 원래 true quantity 구간과 target gap·target calendar는 사후 오류 설명용이다. target 정보를 encoder 입력이나 배포 gate에 넣지 않는다.

**계측을 구현하고 출력 불변성을 확인한다 — 완료**

- 과학 코드와 forward 계산은 바꾸지 않고 Q/K/V·GRU·최종 head의 실제 입력·출력을 읽는 hook을 별도 운영 모듈로 작성했다. CNN/GRU 과거 경로 norm은 full9에서 기록하고 대조에는 적용되지 않는다고 표시한다. norm을 raw 수량의 부호나 인과 효과로 해석하지 않는다.
- 시간 위치·scale은 원래 float64 lognormal 계산을 사용하고, 시간 지표는 recorded positive integer probability mass에 대한 NLL이다. 수량 log-MSE 경로의 scale0을 불확실성 추정값으로 해석하지 않는다.
- CPU 합성 fixture의 집중 테스트4개에서 세 arm 모두 계측 전후 원래 출력이 bitwise 동일하고 state SHA가 불변이었다. full의 CNN/GRU 잔차를 nonzero로 만든 합성 검증도 포함했다. 로컬 Torch 버전은 native와 달라 이것은 계측 불변성 검사이며 실제 GPU 자격 확인을 대신하지 않는다.
- 수정 회귀 검증6개와 native GPU synthetic3route를 통과했다. native에서는 hooked/unhooked 원래 출력의 bitwise 동일성·finite·state 불변을 확인했으며 synthetic qualification을 실제 데이터 평가 수에 포함하지 않았다. 오프라인 집계 회귀 검증11개와 독립 QA도 통과했다.
- worker는 전체·큰 수량의 계측 자체 여섯 지표와 원래 native receipt의 여섯 지표를 각각 tolerance `relative=absolute=1e-5`로 확인한다. 대상 수·population·target/truth/history digest·state SHA는 정확히 일치해야 한다.

**5080의 새 격리 폴더에 배포하고 Validation27을 재현한다 — 완료**

- GPU UUID `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, 기존 `ai_env` Python3.12·Torch2.11.0+cu130·cuDNN91900을 사용한다. 건강 관측에서 GPU compute process0, reboot-required0과 이전 평가의 실제 pipeline exit0을 확인했다. 처음 건강 확인의 tmux 경로 가정을 `/usr/bin/tmux`로 수정했으며 학습·모델 추론은 아직 시작하지 않은 상태에서 확인을 마쳤다.
- 새 remote root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v2_5080`이다. 이전 평가 root의 immutable 입력304개를 SHA 대조 후 새 root로 복사하고 별도 운영·참조·복구 판단67멤버를 봉인 운송했다. 이전 root에는 결과·claim·코드를 쓰지 않았다.
- native synthetic3route를 통과한 뒤 같은 dataset/seed의 세 arm을 최대3개 동시에 평가한다. 실제 Validation worker27개 exit0·재현·대상 및 계측 SHA를 모두 확인한 뒤 전역 gate를 만든다. 각 split worker3시간·전체12시간 상한, 자동 retry/resume0을 유지한다.
- 실제 마지막 Validation exit `1791769637.7873802` 뒤 gate `1791769637.7953472`를 생성했고 첫 Test는 `1791769638.1601312`에 시작했다. Validation27 모두 원래 지표·target/truth/history digest와 선택 checkpoint state를 재현했다.

**Validation gate 뒤 Test27의 오류·보정 관계를 확인한다 — 완료**

- 전체·series·Train에서 정한 입력 이력/gap·truncation·Validation series 포함 여부·Train 최대 이력 초과 여부, 기존 수량 구간과 시간 구간별 충분통계를 기록한다. 개별 예측 행은 저장하지 않는다. 좋은 seed·대조·이득 구간도 함께 포함한다.
- Taxi52의 큰 수량83건, Intermittent42의 수량2~31 및 잘린 이력·시간 손해, RAF의 큰 수량39건과 series 구성·이력14초과의 교차를 확인한다. 희소 교차구간은 대상 수와 SSE/SAE/NLL 합을 우선 보고한다.
- 원래 Test 지표·target/truth/history digest가 재현되지 않거나 nonfinite·가중치 변화가 생기면 실패를 보존하고 멈춘다. 기존 결과를 새 결과로 교체하거나 Test 성과로 checkpoint를 바꾸지 않는다.
- 실제 Test27은 모두 exit0·원래 여섯 지표 및 대상 digest 재현을 통과했다. Taxi52의 큰 수량 손해, Intermittent42의 비절단 수량 손해와 좁은 시간 scale, RAF tail39의 이력14 이내·Validation 공유32건을 확인했다. 좋은 seed와 모든 대조·손해 지표를 전체 보고서에 보존했다.

**실제 종료·원본을 회수하고 원인 후보와 남은 작업을 정리한다 — 완료**

- 실제 pipeline OS exit0, worker54개 exit0, 소유 process/GPU PID 부재와 모든 input/operation/telemetry SHA를 확인한 뒤 exclusive 회수 claim을 만든다. manifest/archive의 원격·로컬 모든 파일 SHA를 검증한다.
- 결과 분석에서는 동일 대상의 full−대조 MSE/MAE/NLL 기여를 분해하고, 큰 수량·시간 head·보정 norm의 관계를 사실·연관·미확인으로 나눈다. RMSE를 구간별로 더하지 않으며 연관을 구조의 인과적 필요성으로 표현하지 않는다.
- 확정된 증거로 논문 작업 순서를 다시 정리한다. CNN/GRU 각 우회18설정과 RAF12fresh는 이번 승인에 포함되지 않으며 계측 결과를 보고 필요성을 별도로 판단한다. 기존 Test는 탐색적 원인 진단이고 미노출 독립 확인이 아니다.
- 실제 pipeline·worker 종료와 owner/GPU PID 부재를 확인한 뒤 원본746개를 한 번 회수했다. archive SHA `4efb0453b5590eac18e5bcbfa29062970c9c5c291f8672cc14ad32ddce24c3f7`, manifest SHA `e2dd3217e2657d9fbabc9589c1d53bd9dad0707f2eeb8d7af1763a0248e65588`와 두 로컬 사본이 일치한다. [원본 검증](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/originals/LOCAL_VERIFICATION.json).
- Intermittent의 첫 오프라인 보고서가 다른 수량 구간을 강조한 서술 오류는 이전 분석·claim을 `analysis_revisions/v1`에 보존한 뒤 수정했다. 수정 전후27개 수치 행·18개 대조·source 파일은 정확히 같으며 추가 모델 실행은0이다.
- [논문 작업 순서](../titantpp_history_rtx_handoff_20261010_v1/PAPER_WORK_ORDER.md)를 완료 상태와 실제 원인 후보에 맞춰 재정렬했다. 다음은 Test·진단 예외를 원고 주장에 반영하고, 시간 head·큰 수량 문제의 단일 가설을 정하는 작업이다. 추가 실험은 자동 실행하지 않았고 열린 main.tex도 수정하지 않았다.

계측 구현과 독립 QA·오프라인 집계 구현은 별도 파일로 병렬 진행했다. 공통 계약·배포·native 자격 확인·전역 Validation gate·Test·회수는 현재 세션에서 직렬 완료했다. 실행 증적은 `APPROVAL.json`, `INITIAL_5080_HEALTH.json`, `DEPLOYMENT_PACKAGE.json`, 새 bundle의 `contract.json`·`input_integrity.json`·`operation_integrity.json`과 완료 receipt에 보존한다. 인과적 구성요소 필요성·새 자료 일반화·전체 CPU binary 감사는 별도 미완료다.
