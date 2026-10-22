# Recovery v2 independent QA

## Review Result: PASS

복구 v2의 로컬 검증은 **34개 테스트(독립 11개)와 full Ruff 실제 exit 0**으로 통과했다. 원문 출력 byte·SHA와 테스트 전후 소스 SHA를 검증했다. 원래 봉인된 15개 파일과 recovery v1 전체 파일은 변경되지 않았다. 기존 303개 테스트를 다시 실행한 것은 아니다.

### CRITICAL
- 미해결 사항 없음.

### HIGH
- 미해결 사항 없음.

### MEDIUM
- 미해결 사항 없음.

### LOW
- 미해결 사항 없음.

## 실제 실패와 검증 경계

v1의 읽기 전용 확인은 `configure_runtime`이 Torch inductor/dynamo를 불러오면서 private cache 디렉터리를 만들려 하자 차단됐다. 원문은 actual exit 1, transport_error null, timeout false이며 배포 초기화 검사 실패다. 학습 실패나 SSH 연결 실패로 해석하지 않는다. v1과 두 실패 원문을 보존했다.

v2는 파일·archive SHA, Python/torch/CUDA/cuDNN 버전, GPU UUID, 환경·boot·GPU guard만 읽는다. **수치 실행 설정은 확인하지 않았다는 표식과 기존 qualification에서 확인해야 한다는 표식을 모두 요구한다.** 읽기 전용 검사에서 설정 함수를 제거한 것이 실제 학습의 수치 검증을 제거한 것은 아니다.

기존 qualification은 구조·finite 검사의 synthetic 9개와 capacity 3개다. 모든 12개가 수치 설정을 수행한다고 주장하지 않는다. capacity producer는 원래 함수로 전체 수치 설정·runtime identity를 검증하며, v2는 원래 종료·원문·신원 검증 후 **capacity 3개 각각의 전체 runtime_expected와 GPU UUID를 한 번 더 대조**한다. 양 서버가 통과하기 전에는 실제 학습을 시작할 수 없다. 학습·Validation 자체의 원래 Runtime 검사도 유지한다.

## 확인한 항목

- [x] fresh CPU subprocess에서 실제 configure 경로의 cache mkdir 차단과 static snapshot의 성공·설정 불변을 검증했다. GPU는 로컬 fixture다.
- [x] 독립 검사에서 수치 설정 함수와 CUDA 초기화 함수를 금지해도 static snapshot이 성공한다.
- [x] capacity runtime 누락·deterministic 설정 불일치·다른 GPU UUID를 거부한다.
- [x] 생성된 읽기 전용 검증 코드를 실제 임시 archive에서 실행해 쓰기 차단 및 파일/archive 변조·추가 파일·symlink 거부를 확인했다.
- [x] 5080 재전송·덮어쓰기 없음, 5090 최초 배포만 허용, 새 실패·partial 뒤 원격 호출 차단을 유지한다.
- [x] MLP9 신규 학습·B/D18 재사용·Validation27·Train 비용162 범위와 MAC 보류를 유지한다. 새 Test 평가를 추가하지 않는다.

## 증적과 남은 작업

`independent_checks_20261022T151134988125/receipt.json`: 34 passed in 2.17s, pytest/Ruff 실제 exit 0. 검토 파일과 ADMISSION SHA는 `INDEPENDENT_QA.json`에 봉인했다.

**실제 설치 상태와 qualification을 확인한다 — 다음 작업**
- 부모 세션이 v2 봉인과 승인 범위를 확인한 뒤 5080의 static 확인, 5090의 최초 배포, 양 서버의 원래 native qualification 순서로 진행한다. 이 QA는 실제 서버 성공을 확인한 것이 아니다. 검토자의 원격·GPU·실데이터 추론·Test 열람은 모두 0회다.
