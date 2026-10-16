# Test 36조건 원본 회수 finalizer QA

## Review Result: PASS

- Scope: `remote_finalize.py`의 실제 종료·qualification·선택 모델 연결·원본 archive 생성 검증.
- Review changes: 이 보고서만 작성. 원격 접속, 실제 finalizer, Test 추론, 원본 수정은 수행하지 않았다.
- Automated check: Python AST parsing PASS. 수정된 archive 검사 부분만 임시 합성 fixture에서 실행하여 정상 archive 통과, manifest 후 파일 변경 거부, symlink 거부를 확인했다. 실제 finalizer 전체나 remote/Test 작업은 호출하지 않았다.
- Reviewed finalizer SHA256: `1a420f184898ce281f9f70fb6ca1db55093c6b747f5ed7913746481fee81dca4`.

### CRITICAL

- 확인된 문제 없음.

### HIGH

- 남은 확인된 문제 없음.

### MEDIUM

- 남은 확인된 문제 없음.

### LOW

- 확인된 문제 없음.

## 수정 확인한 발견 사항

- 최초 native qualification 검사 공백(HIGH): 기존 길이6/PASS 검사는 외래 contract/route 합성 객체도 통과했다. 수정된 `remote_finalize.py:22`는 정확한 계약·여섯 route 집합을 고정하고, 각 receipt SHA·study/arm/device/source·합성 작업 범위·Python/GPU 및 수치 Runtime·실제 probe exit0를 검사한다. probe PID도 부재 검사에 추가됐다.
- 최초 manifest/archive 검사 공백(MEDIUM): manifest 후 원본 변경을 놓치고 symlink를 포함할 수 있었다. 수정된 `remote_finalize.py:63`은 root 내부 일반 파일만 허용하며, `:72`에서 archive의 정확한 member 집합·개수·일반 파일·크기·bytes SHA를 검사하고 `:80`에서 현재 원본을 다시 manifest와 비교한다. 합성 정상/변조/symlink fixture에서 기대한 통과/거부를 확인했다.

## Acceptance criteria

- [x] pipeline complete36/36 및 관측한 OS exit0를 요구한다.
- [x] 72개 평가 worker의 exit0 및 job/contract를 검사한다.
- [x] Validation gate receipt/exit raw SHA와 selected epoch/checkpoint/source binding을 검사한다.
- [x] 자기 pipeline·worker PID 및 GPU PID 부재를 검사한다.
- [x] input/operation 파일 raw SHA를 다시 검사하고 exclusive retrieval claim을 남긴다.
- [x] 여섯 native qualification의 실제 증적을 연결해 검사한다.
- [x] 일반 파일·archive 내부 bytes·manifest·최종 원본의 일치를 검사한다.
- [x] 수정 코드 재검토 후 최종 verdict를 기록한다.

이 검토는 실제 실험 종료나 최종 Test 결과를 확인한 것이 아니다. 최종 회수는 모든 실행 증적이 준비된 뒤 주 세션에서 한 번 수행한다.
