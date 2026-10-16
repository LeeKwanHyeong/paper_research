# v4 사전 독립 QA

상태: `LOCAL_READY_5080_NATIVE_PENDING`. 로컬 코드 범위 PASS다. 실제 native 통과는 아직 확인하지 않았다. QA SSH/GPU 실행0, 코드·bundle·원고 변경0.

## 검토한 변경

- 원래 source131, 데이터7개, 네 구조, 활성 파라미터 수, Train/Validation 범위, 초기화·손실·optimizer·selector·36개 조건과 5080:5090=24:12 배정은 변경하지 않는다.
- v4는 새 격리 root에서 GPU 프로세스 관측만 보완한다. v2/v3 실패 원본과 claims를 보존하고 기존 5080 절대 deadline `1792399058.15949`를 유지한다.
- 캠페인의 `verify_gpu_membership` 원래 양성 검증은 그대로다. stat 전후의 PID/PPID/process group/start ticks를 확인하며 안정된 외부 argv는 즉시 거부한다.
- 빈 argv·소멸·종료 상태는 최대2회 fresh NVML 조회로만 확인한다. 이전에 정확히 확인한 동일 PID 기록이 현재 active guard와 전체 명령에 결합되고 60초 이내인 경우에만 fresh 목록 부재를 소유 종료 경계로 인정한다. 실제 worker exit0와 최종 GPU idle은 별도 필수다.
- launcher v4는 버전 literal 외 v3와 같은 함수 AST와 remote script를 유지한다. 상태 조회의 순간적 프로세스 불일치는 `training_identity_unconfirmed`이며 학습 실패나 자동 재실행으로 바꾸지 않는다.

## 완료한 검증

독립 회귀 42개 PASS(15.763초): campaign22 + preparation2 + launcher18. 실제 봉인 source131 Train/Validation 자료3개를 거치는 interface12/학습 인수12가 포함됐다. 모델 예측0, 과학 학습0, Test materialization false, CUDA 초기화 false다.

초기 v4 candidate의 오래된 witness 오판을 별도 mock으로 재현했다. active가 비어 있고 오래된 witness만 남은 상황에서 PID 소멸을 정상으로 인정하던 사례는 현재 active guard/명령/시간 제한 보완 후 ValueError로 차단됐다. 상세 내용은 REVIEW_NOTES.md에 남겼다.

## 남은 gate

- [x] 추가 종료 경계·좀비·PID 재사용·외부 프로세스·조회 실패 전용 회귀
- [x] immutable v4 bundle 전체 SHA와 5080만의 승인·lease
- [ ] 실제 5080 native + 3개 동시 capacity owner + 각 actual exit0 + GPU idle
- [ ] 최초 실제 학습 PID/argv/GPU, 저장 전 또는 실제 epoch 확인

5090은 별도 사용자 명령 전 모든 SSH·배포·qualification·학습0이다. 개발 성능 대조는 참신성·보편적 필요성·독립 일반화 증거로 확대하지 않는다.

## Review Result: PASS — 최종 로컬 코드

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음. 발견했던 stale witness 면제는 현재 active guard/전체 명령/관측 age≤60초 조건으로 수정됐고 원래 재현 및 음성 회귀 모두 차단을 확인했다.

### LOW
- 없음. 운영 reason은 원 실패 argv가 미기록이라는 사실을 명시하도록 수정했다.

추가 `GPUExitBoundaryTests` 11개를 독립 실행해 PASS(0.017초)했다. 총 독립 검증은 기존42+추가11=53개다. 좀비·소멸 정상 경계, 최대2회 재조회, 안정된 외부 argv 즉시 거부, 처음 보는 PID 소멸 또는 계속 존재, 오래된/없는 active/변경된 guard witness, PID 재사용, 권한 오류, 새 외부 GPU PID, 최종 idle을 검증했다. 원래 양성 `verify_gpu_membership`의 함수 AST가 v3와 동일한 검사도 통과했다.

## v4 봉인 및 승인 확인: PASS

독립 stdlib SHA로 source131/input7/operation4/local tool33/archive143 regular member 전체를 확인했다. current/receipt/hold/5080 승인·release 결합도 일치한다. 5090 승인 부재 및 로컬 authorization 거부를 재검증했다.

- canonical `aa3bc1583a72251043ad3df09bca23a33205419e9e25d3dc64c9681550c745a1`
- source131 `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`
- operation4 `bc88528e79aa9de506693ed8b2f73a510bdc10b88c27e265f109c96a39d99309`
- payload `38bd6fa5e14e717df0d64674109d06dc7219c306bf10bcd8144f59dd47b0d54e`
- local tool seal raw `7fca937ff5c9933a3ee8ef09fc3d2ef4854a23c2c5c960121d04be4aaba9051b`
- 5080 승인 raw `8735b291f328426fb86d81c888d27ac28bb119c45eeb2000e2b97ab354e65ffd`
- release raw `8c156e326388637de7ed716c692efbe27177313dea379e313f12ea4df3415f1d`
- original hold raw `69880f758a7321038999a8ffdd0c286ff0fe023288f3b0b2c497bc0d306bee49`

승인 원문: “그럼 마지막 프로세스 판정 오류를 수정하고 검증해서 5080 학습부터 바로 진행하자. 5090은 내가 진행하자고 할 때까지 대기해.”

로컬 준비는 완료됐으나 실제 5080 qualification 통과 전에는 fit을 승인하지 않는다. 이전 native/admission의 부분 통과를 새 root의 통과로 대신 사용하지 않는다.
