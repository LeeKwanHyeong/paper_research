# Bootstrap recovery independent QA

## Review Result: PASS

별도 복구 경로의 로컬 구현 검증을 통과했다. **23개 테스트(독립 6개)와 full Ruff가 실제 exit 0**으로 종료됐고 원문 출력의 byte 수·SHA와 검증 전후 파일 SHA를 확인했다. 원래 봉인된 소스·테스트 15개와 기존 QA·실패 기록은 변경하지 않았다.

### CRITICAL
- 미해결 사항 없음.

### HIGH
- 미해결 사항 없음.

### MEDIUM
- 미해결 사항 없음.

### LOW
- 미해결 사항 없음.

## 원인과 수정 범위

`initial_execution.py:233`에서 새로 만든 소스 경로를 `sys.path`에 넣었지만 Python 시작 때 기록된 negative importer cache가 남았다. 별도 임시 패키지로 실제 오류를 재현했고 `importlib.invalidate_caches()` 이후 import 성공을 확인했다. 기존 드라이버를 고치거나 실패한 전송을 재시도하지 않는다.

기존 실제 명령은 exit 1, timeout false, transport_error null, stdout 0 bytes, stderr 120 bytes였다. 따라서 **원격 배포의 Python 초기화 실패**이며 SSH 연결 실패나 학습 실패로 보고하지 않는다. 실제 서버 재확인 전에는 추출물의 현재 무결성을 확정하지 않는다.

## 수용 기준

- [x] 5080은 최초 45초 읽기 전용 확인으로 기존 archive·전체 추출물 SHA·boot·GPU·Runtime을 검증한다. 재전송·덮어쓰기·삭제·과학 작업 시작은 없다.
- [x] 읽기 전용 생성 코드의 검증 부분을 독립 CPU subprocess에서 실행했다. 실제 paper/Runtime/GPU import 전에 중단하며, 정상 fixture 통과와 파일 쓰기 차단을 확인했다.
- [x] 추출 파일·archive 변조, 추가 파일, symlink를 모두 거부한다.
- [x] 5090은 기존 archive를 처음 배포하며, 원래 생성 코드와의 차이는 프로세스 내부 import cache 무효화 한 줄뿐이다. 배포는 배타 생성한다.
- [x] 양 서버의 실제 qualification 원문·종료 증거를 통과한 뒤에만 원래 initial fit을 시작한다.
- [x] 원래 실패 상태나 globals를 바꾸지 않는 새 상태 디렉터리를 사용한다. 복구의 첫 실패·partial 뒤 추가 원격 호출을 막는다.
- [x] 원래 MLP9 신규 학습·B/D18 재사용·Validation27·Train 비용162 범위를 유지한다. 새 Test나 MAC 실행으로 확대하지 않는다.
- [x] ADMISSION은 기존 사용자의 실행 승인을 근거로 했으며 새 사용자 응답이 있었다고 기록하지 않는다.

## 증적과 남은 작업

`independent_checks_20261022T150419214984/receipt.json`: 23 passed in 0.24s, pytest/Ruff 실제 exit 0. `INDEPENDENT_QA.json`에 검토 파일·원래 실패 원문의 SHA를 기록했다.

**실제 배포 상태를 확인한다 — 다음 작업**
- 부모 세션이 새 recovery QA·ADMISSION·드라이버 SHA를 결합해 5080의 기존 추출물을 검증하고, 통과한 뒤에만 5090 최초 배포를 진행한다.

**원래 실험 승인 검사를 통과한다 — 다음 작업**
- 양 서버 native qualification이 통과하기 전에는 학습하지 않는다. 이 QA는 로컬 구현 검증이며 실제 서버 성공이나 과학적 효과를 뜻하지 않는다. 이 검토의 원격·GPU·실데이터 추론·Test 열람은 모두 0회다.
