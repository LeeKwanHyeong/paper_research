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
