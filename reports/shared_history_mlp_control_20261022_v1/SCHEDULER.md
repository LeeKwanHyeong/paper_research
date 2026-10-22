# 공유 이력 보정과 단일 MLP 비교

## 현재 기준선을 보존한다 — 완료
- 대상 paper_research / codex/shared-history-mlp-control-20261022. 기존 A–D36조건·외부5seed·MAC 결과/실패/checkpoint를 보존한다. 기존 frozen987소스는 수정하지 않고 새 모델·실행 경로만 추가한다.

## 같은 정보와 총 파라미터를 가진 대조를 고정한다 — 완료
- B의 현재·직전 관측 및8개 가용 이력 조건을 단일MLP에도 제공한다. 136→48→64, biasFalse·exactGELU·출력0초기화·잔차/8로9600parameters를 맞춘다. active-prefix와함수클래스차이는 명시한다.
- B9/D9는 원래payload/선택/실제exit/Validation원문을 검증해 재사용하고 MLP9만 새학습한다. Test를 이용한 구조선택과 새Test평가는0이다.

## Validation 진단과 구현을 검증한다 — 진행 중
- 기존Validation저장예측으로시간NLL꼬리와seed별교환관계를분석한다. 소스·학습손실·선택조건을임의로바꾸지않는다.
- 모델/runner/profiler와manager/controller를독립파일로병렬구현하고공통계약뒤단위/계약/독립QA와실제양hostsynthetic/capacitygate를직렬통과한다.

## 같은 GPU에서 학습·Validation을 진행한다 — 다음 작업
- seed42는5080,52/62는5090. GPU당과학worker1개. 새9fit과기존18reuse,선택본Validation27재현을모두검증한다. 기존Runtime/네트워크/인증·정지작업은보존한다.

## Train 공통 입력의 속도·메모리를 비교한다 — 다음 작업
- 전체Validation27gate뒤동일Train37batch(예열5·측정32)에서27조건×학습/평가×3반복=162freshprocess측정. 원본checkpoint는저장하지않고전후SHA동일성검증. GPU/seed별절대값과짝비교를보존한다.

## 회수·독립 비교 검토를 마친다 — 다음 작업
- 단계별실제exit0/신원부재/GPUguard뒤최초회수1회. archive/manifest/member/source/input/legacy/선택SHA검증,Validation-only순수집계/불리한모든seed/비용보고를작성한다. 최종대표모델이나논문기여를자동확정하지않는다. MAC은계속보류한다.

## 학습 전 검증 결과를 고정한다 — 진행 중 (2026-10-22 23:52 KST)
- 기존 A/B/D의 Validation 예측 27개·911,187행에서 원 receipt의 수량 RMSE·MAE·시간 NLL을 재현했다. Taxi에서는 B가 D보다 수량 오차를 줄이지만 세 seed 모두 시간 NLL이 악화된다. 시간 점수 정의의 오류를 입증하지 못했으며, 저장 예측에 분포 파라미터가 없어 원인은 미확인이다.
- 기존 B/D checkpoint 18개의 선택 epoch·binary/tensor SHA·학습과 Validation의 실제 종료 및 원 예측 재현을 로컬에서 확인했다. 이를 새 모델로 이름만 바꾸지 않고 원 payload 그대로 재사용한다.
- 모델·실행·제어·비교·최초 배포 경로의 통합 로컬 테스트와 전체 Ruff가 실제 종료 코드 0을 반환했다. 합성 NaN 손실 누락과 native dispatcher 종료 증거 누락은 학습 전에 수정했고, 실패 원문도 보존했다. 정확한 테스트 수와 원문 SHA는 local_checks_20261022T145220371808/receipt.json에 연결한다.
- 양 서버의 최초 읽기 전용 preflight는 성공했다. 당시 5080의 기존 정지 작업 4개와 5090의 유휴 상태를 확인했다. 이 과거 관측을 현재 상태로 간주하지 않으며, 새 학습은 아직 시작하지 않았다.
- 다음 순서는 독립 QA 봉인 → 현재 브랜치의 신규 소스 커밋 → 계약·archive 봉인 → 양 서버 합성/용량 검증 → MLP 9개 학습과 전체 27개 Validation이다. Test와 MAC은 계속 보류한다.

## 최초 배포의 Python 경로 초기화를 복구한다 — 진행 중 (2026-10-23 00:04 KST)
- 소스는 paper_research / codex/shared-history-mlp-control-20261022의 747e9e25에 고정했다. 최초 303개 로컬 테스트와 독립 QA 통과는 서버에서의 실제 검증 완료를 뜻하지 않는다.
- 5080 최초 배포 명령은 archive 검증·압축 해제 뒤 `paper` import에서 실제 종료 코드 1을 반환했다. SSH의 transport_error는 null, timeout은 false였으며, stderr 120 bytes와 원래 ATTEMPT/FAILURE를 보존했다. 5090 배포·합성 검증·학습은 이 시점까지 시작하지 않았다.
- 새 경로를 생성 전 PYTHONPATH로 전달하면 import finder의 None 캐시가 남는 현상을 새 로컬 Python 프로세스에서 재현했다. 캐시 무효화 전 exit1, 이후 exit0이다. 실패한 원격 프로세스의 당시 캐시 내용 자체를 관측했다고 주장하지 않는다.
- 기존 명시적 양 서버 검증·실험 승인 안에서 별도 recovery_v1/ADMISSION.json으로 초기화 복구 범위를 기록했다. 새 사용자 응답으로 가장하지 않는다. 동결된 15개 소스/테스트, 과학·실행 계약, archive와 원 실패 상태는 수정하지 않는다.
- 다음 순서는 별도 복구 도구의 독립 검토 → 5080 기존 전송본의 읽기 전용 전체 SHA/Runtime/신원 검증 1회 → 5090 최초 전송 → 양 서버 원래 12개 합성/용량 검증 → 기존 계약의 최초 학습이다. 5080 재전송·실제 학습 재시도·MAC·Test는 0이다. 복구에서 다시 실패하면 추가 원격 호출을 중단한다.

## 읽기 전용 설치 확인과 수치 검증을 구분한다 — 진행 중 (2026-10-23 00:10 KST)
- recovery_v1의 5080 확인은 전체 archive/member SHA·import·기존 프로세스 guard 이후, `configure_runtime`에서 torch 2.11의 private cache 생성이 읽기 전용 audit에 차단돼 actual exit1로 끝났다. transport_error=null, timeout=false, stderr 2,197 bytes이며 원문과 실패를 새로 보존했다. 모델 학습 실패나 SSH 연결 실패로 분류하지 않는다.
- 이전 복구 상태의 추가 실행을 중단하고 모든 v1 파일 bytes/SHA를 고정했다. 기존 명시적 초기 검증·실험 승인 범위에서 별도 recovery_v2가 읽기 전용 설치 확인의 범위를 실제 소스·라이브러리 버전·GPU/boot/env 신원으로 한정한다. 수치 설정의 성공은 이 단계에서 주장하지 않는다.
- 원래 qualification의 9개 구조·유한성 점검과 3개 full-batch capacity 점검을 유지한다. 3개 capacity는 변경하지 않은 profiler._runtime의 전체 수치 설정/신원 검증을 실제 실행하며, 실제 train/Validation도 변경하지 않은 runner._native_runtime을 통과해야 한다. 새로운 복구 제어는 실제 capacity receipt의 Runtime/GPU를 추가 확인한 후 최초 fit을 허용한다.
- 과학 소스·원래 실행 계약·archive·선택본·고정 15개 파일은 그대로이며, 새 Test·MAC·학습 재시도는 여전히 0이다. v2는 별도 테스트/독립 QA 및 증거 핀을 통과한 뒤 실행한다. 연결 실패·과학 실패·원본 SHA/신원/기한 불일치는 추가 복구 대상으로 자동 승인하지 않는다.

## 양 서버의 최초 qualification을 실행한다 — 진행 중 (2026-10-23 00:14 KST)
- recovery_v2는 신규 34개 로컬 테스트와 전체 Ruff·독립 QA PASS를 확인한 뒤 로컬 제어 commit 833d68a3으로 고정했다. 원 과학 소스 commit 747e9e25 및 archive/계약은 변경하지 않았다.
- 5080 기존 파일의 전체 source991/input9/legacy199/archive SHA·정적 Runtime/boot/GPU/정지 프로세스 확인 actual exit0을 관측했다. 재전송 0, 수치 설정은 pending으로 명시했다.
- 5090은 같은 archive의 최초 전송 및 전체 수치 Runtime 확인 actual exit0을 관측했다. cuDNN은 5080 91900 / 5090 92000이며, 이후 비용의 절대 시간을 두 GPU 사이에 섞지 않는다.
- 두 host 각각 qualification dispatcher와 별도 부모의 실제 초기 신원을 확인했다. 5080 dispatcher872998/start_ticks107866007, 5090 dispatcher3429498/start_ticks12425101이다. 이는 시작 확인이며 전체 12개 종료나 학습 완료를 의미하지 않는다.
- 다음 단계는 host당 읽기 전용 종료 확인 1회로 12개 child 및 dispatcher의 실제 exit0·원문 SHA·신원 부재와 3개 capacity Runtime/GPU를 검증하는 것이다. 양쪽 모두 통과한 뒤에만 원래 MLP9 fit/B·D18 재사용/Validation27을 최초 시작한다.

## 학습 진입 전 누락된 계약 정보를 보완한다 — 진행 중 (2026-10-23 00:32 KST)
- 최초 양 서버 qualification은 각 12개 child와 dispatcher의 actual exit0 및 세 capacity Runtime/GPU 검증을 통과했다. 이후 첫 shared15 재사용은 actual exit0이었지만, 첫 single_mlp 학습 준비가 양쪽 모두 actual exit1로 종료했다.
- native_failure_forensics_v1의 각 host 읽기 전용 1회 조회는 actual exit0이었다. 원래 child/dispatcher exit·stderr·claim·supervisor 증거 17개씩을 bytes/SHA 그대로 회수했다. 공통 traceback은 `_training_interface`에서 `design_sha256`를 찾지 못하며, 뒤의 `training.train_one` 호출 전이다. 실제 MLP optimizer step은 0이고 수치 학습 실패나 SSH 실패로 분류하지 않는다. 기존 continuation_v1 실패 상태와 모든 원문을 유지한다.
- 원래 303개 테스트에는 실제 generic CPU 2epoch 테스트가 있었지만 args/interface를 fixture가 주입해 실제 연결 누락을 놓쳤다. 새 execution_v2에는 실제 `_training_args → _training_interface → train_one → 선택 checkpoint 재로드 → Validation`을 연결하는 회귀 검증을 추가한다.
- 기존 과학·모델·손실·선택 조건과 991개 과학 source/9 input/199 legacy는 변경하지 않는다. 새 계약에 `design_sha256 = SCIENCE_SPEC SHA`만 보완하고 새 격리 실행 경로와 새 제어 QA를 사용한다. 기존 절대 host 기한을 연장하지 않는다. 추가 회귀/독립 QA와 양 서버 원래 qualification을 통과한 뒤 동일한 최초 실제 MLP9 학습을 진행한다.
- execution_v2/ADMISSION.json은 기존 명시적 검증·필요 실험 승인 안에서 설정 메타데이터 복구 범위를 기록한다. 새 사용자 응답으로 간주하지 않으며 303개 기존 증거의 bytes/SHA를 고정한다. Test·MAC·기존 B/D 재학습·실패 삭제는 0이다.

## 보완한 계약으로 실제 MLP 학습을 진행한다 — 진행 중 (2026-10-23 00:40 KST)
- paper_research / codex/shared-history-mlp-control-20261022의 실행 제어는 cd44d201, 실제 과학 source는 변경 없이 747e9e25에 고정했다. 새 계약 raw SHA는 56721f98f69c962fee1cdb8c4331d6dd4c4373bb5745f302fc2164a1b9819673이다. 새 archive와 실행 경로를 사용하고 이전 실패·checkpoint·소스와 절대기한은 보존했다.
- 새 로컬 77개 테스트와 전체 Ruff actual exit0, 실제 fabricated CPU 2epoch/optimizer2step/선택본 재로드/Validation2회 재현, 독립 QA PASS를 확인했다. CPU 비용 경로는 RAM 사본의 train37step/eval0을 검증한 것으로 GPU 속도·메모리 측정 완료를 뜻하지 않는다.
- 양 host의 새 qualification 각각 12 child와 dispatcher actual exit0, 과학 신원 부재, 세 capacity의 Runtime/GPU 증거를 검증했다. 당시 5080의 기존 정지4개와 5090의 다른 GPU 작업 없음도 guard로 확인했다. supervisor 자신의 OS exit를 직접 관측한 것으로 표현하지 않는다.
- 00:38:56/58 KST에 양 host fit_validation dispatcher를 시작했다. 00:40 KST 최초 bounded controller actual exit0/waiting에서 RAF single_mlp seed42 epoch27, seed52 epoch25와 최근5epoch train_all_finite=true를 확인했다. 이는 실제 학습 진입 증거이며 전체9fit/Validation27 완료나 결과 우위를 뜻하지 않는다.
- 다음은 동일 선택본의 Validation27 전체 gate → Train 고정37batch의162freshprocess 비용 측정 → 최초 회수/원본과 archive·manifest·member·종료·선택 SHA 검증 → 독립 로컬 비교 검토다. 이 연구의 새 Test 평가는0이며 MAC은 보류한다.
- 후속 제어는 아래 한 명령만 회차당1회 실행한다. waiting_next_check(900초 이내)/busy는 SSH0, 실제 확인은 host당 읽기 전용SSH1/45초 상한이다. 실패/partial/기존attempt/신원·원본 불일치는 원문을 보존하고 추가 원격0이며 과학 재시도나 재시작을 하지 않는다.

```sh
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 reports/shared_history_mlp_control_20261022_v1/execution_v2/advance.py --mode advance --binding-sha 3f61a06bdac3b53815771ccf9cf24f7285db14f9624876972dbeb316915283af --authorization-sha 5e631f4a2ba71cd49da2b8ef134ece59457c76d14994ae9c83c8e31d60152ef2
```

- 양 서버 fit/cost 종료·최초 pack/전송은 이 controller가 고정 gate 안에서만 처리한다. 별도 status/dispatch/collect/finisher와 새 학습을 추가 호출하지 않는다. 자동 점검 방식에 대한 선택 질문은 아직 응답 대기이며 새 자동화를 생성하지 않았다.
