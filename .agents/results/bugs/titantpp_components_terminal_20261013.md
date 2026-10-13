# CNN·GRU36 완료 상태 덮기와 종료 캐시 보정

**원문에서 실패를 재현하고 원인을 확인한다 — 완료**

- 원 v5 training_status는 terminal manifest·원격 파일 SHA를 확인해 complete를 설정한 뒤, 마지막 checkpoint receipt의 이전 exposure.json SHA로 상태를 덮었다. frozen_runtime은 endpoint 처리 후 최종 exposure를 다시 기록하므로 정상 종료에서도 stale live receipt가 불일치한다.
- 2026-10-13 08:27:20 KST 5080 snapshot1ecf…(rawSHA352b65c7…33556f2)은 actual worker24 exit0, terminal remote files312 SHA, supervised dispatcher exit0와 마지막 train_complete가 있다. 원 helper는 `Terminal completion or GPU idle unconfirmed`로 거부했다. 과학적 학습 실패가 아니다.

**기존 봉인을 유지하고 새 검증기를 추가한다 — 완료**

- 원 launcher·observer·source131·input7·operation4·local40은 그대로 유지했다. 새 titantpp_components_terminal_verifier와 hourly_v2는 완전한 terminal manifest의 최종 history·timing·binary SHA를 우선한다.
- 실제 worker/dispatcher exit·정확 PID/PPID/argv·claim·lease·원관측시각과 qualification/permit 교량을 유지한다. native qualification은 실제 idle 원관측 cache를 사용하고, remote training_permit 원문은 직접 관측하지 않았다고 기록한다.
- 종료 캐시는 해당 root의 과거 종료만 뜻한다. lease가 지난 뒤에도 원시각을 유지하며 현재 전역 GPU idle을 뜻하지 않는다. 같은 GPU의 새 handoff root는 별도 관측한다.

**회귀와 부정 조건을 검증한다 — 완료**

- 원 launcher18 tests PASS. 새 verifier/hourly_v2 26 + seed62 handoff14 + 병렬 집계/관측13 =53 focused tests PASS.
- 실제 원문 결과는 5080 complete24, 5090 running·complete3으로 보정됐다. manifest/binary/hash/worker exit/argv/PID·cache raw/time/permit 변조는 거부한다. cache 반복조회2회 SSH0, 원관측시각 유지와 leaseexpiry 뒤 조회를 검증했다.
- 새 병렬 관측에서 host 조회 실패로 전체 보고가 끊기고 신원 미확정에도 ETA를 생성하는 문제가 로컬 QA에서 발견됐다. root별 실패 증적 보존·독립 관측 계속, 신원 미확정의 ETA/진행 확정 금지 부정 회귀를 추가해 수정했다.
- 새 계약 생성 코드의 main guard 앞 정의 순서, explicit NONE arm 참조도 회귀로 확인했다. 원 함수 AST에서 trainer/dispatch/fit/native qualification 변경0이다.

**새 모니터와 병렬 인계를 적용한다 — 다음 작업**

- 별도 v2 monitor 승인·seal·종료 cache를 생성하고 오프라인 통합 검증한다.
- 미시작 seed62 네 ID의 원5090 생성 배제와 새5080 계약을 결합한 뒤 native qualification/admission·fresh fit을 각 한 번 수행한다. 5090 현재42·52는 자연 진행하며, 52 종료 뒤 예정된 FileExistsError·actual administrative exit1·failure·claim은 보존한다.
- 새 원격 실행 성공, 원본 binary 로컬 회수, 최종Validation 집계는 이 로컬 회귀 보고의 완료 주장에 포함하지 않는다. pipeline wrapper OS exit 직접 관측도 미완료다.
