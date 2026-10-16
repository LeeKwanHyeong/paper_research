# v5 사전 독립 QA

상태: `5080_NATIVE_PASS_INITIAL_THREE_FITS_VERIFIED`. QA 직접 SSH/GPU/코드 변경0.

## Review Result: PASS — 로컬 코드

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

## 검토와 검증

v4에서 `/proc`가 이미 소멸한 PID40079가 두 즉시 NVML 재조회에 남아 gate가 실패한 원문을 확인했다. v5는 이 반영 지연을 최대25회, 0.2초 간격, monotonic5초 이내에서만 관측한다. 재조회 subprocess timeout도 남은 budget에 묶인다. 학습 retry/resume가 아니며 이전 실패 root/claim은 보존한다.

최근60초 이내의 정확 witness, 현재 active guard PID와 전체 command, 원래 argv/PPID/process group/start ticks 검증은 유지된다. 안정된 외부 argv는 즉시 거부한다. unknown 소멸, 계속 남는 PID, PID재사용, 오래된 witness, 없는 active, 잘못된 guard, 권한 오류는 정상 종료로 면제하지 않는다.

독립 고유 검증 44개 PASS: GPUExitBoundary13 + Contract9 + Dispatcher2 + Preparation2 =26개(이전24개0.103초, 최종 경계13개0.022초), Launcher18개(1.753초). 별도 stdlib mock에서 query의 남은0.37초 timeout 전달, 고정GPU 선택, 잘못된 CSV/비정수PID/실제 TimeoutExpired 전파를 확인했다. 두 즉시조회 이후에도 남는 PID가 후속 목록에서 사라질 때 통과하고, 계속 남으면5초 한도에서 차단됨을 검증했다.

v5 campaign의 과학 함수와 원래 양성 소유 검증은 v4와 AST가 같고, launcher는 버전 literal 외 v4 동작이 같다. source131·모델·데이터·학습조건·selector·36ID/24:12배정·3+1·최초5080 lease는 불변이다. 반복 실데이터 CPU 소비자 검증은 변경되지 않은 경로이며 이번 독립 검증에서는 v4 통과 결과를 기준선으로 사용했다.

## 완료 조건

- [x] 반영 지연 성공/상한 실패와 외부·미확인 프로세스 거부
- [x] 기존 과학·학습·host 승인 경계 불변
- [x] 이전 실패 보존 및 5080 절대 deadline 계승
- [x] immutable bundle 전체 SHA 및 최신5080 승인/release 확인
- [x] 실제5080 native/capacity child exit0와 GPU idle 확인
- [x] 최초 실제 fit PID/argv/GPU 및 저장 상태 확인

5090은 별도 명령 전 모든 SSH/배포/qualification/학습0이다. CPU PASS를 실제 native PASS로 보고하지 않는다.

최종 추가 회귀에서 fresh NVML subprocess가 남은0.35초 timeout을 사용하며, query 후 monotonic5.1초에 도달한 경우 결과가 비어 있어도 실패함을 확인했다. 최종 campaign raw SHA는 `a828fdca22083393ff25a123bdedffcbcb7f532644a726cc41b058f5f7b5ec9a`다.

## 봉인과 승인 확인: PASS

source131/input7/operation4/local tool40/archive143 regular member 전체 stdlib SHA 일치. current/receipt/원본 hold/최신5080 singleton 승인/release 결합이 일치하며, 5090 미승인 사전거부도 확인했다. 과학 필드와 최초5080 절대 lease는 부모v4와 같다.

- canonical `e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074`
- source131 `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`
- operation `12c836dc5fe1d9146b83a352ba77abc297f4d69c6f68d83ad00935a2bc442903`
- payload `1d5639ec4003a522f49cf36e9460bc1e041859f73cd5ddba0de4d7c5a0d6725c`
- local tool seal raw `16e317c3c16dddede95c6aa6174026dca0ab772879bf438c3d1e6b752eff7074`
- 5080 승인 raw `8029057d7b2c415235ea3c06b48360d4d141141174c74dcf82350b4686484551`
- release raw `28706aa2bb6878d0f38676b4da9e5521071c3dc202fb6d0b801efb11bec9e4fd`
- hold raw `8933173570f0fec56174091bf139f2057a6fdbc976bb37396b975241514b93fa`

실제 native/capacity와 fit 시작은 별도 증거를 기다린다.

후속 actual native 및 최초3fit 검증은 NATIVE_QA.md에 기록했다. 24개 학습 전체 완료 판정은 아니다.
