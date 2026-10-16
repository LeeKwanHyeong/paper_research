# KeyContent 진단 구현과 실행 순서

**확정 명세와 기존 원본을 확인한다 — 완료**
- paper_research의 설계 문서 4개와 참조 45개 SHA를 재검증했다. KeyOnly3 선택 epoch9/5/26, source145, Train/Validation 입력과 부모 on 행을 그대로 사용한다.
- 기존 CNN·QueryOnly·Test 추론, 새 학습, 원본 변경은 범위에 포함하지 않는다.

**새 진단 도구를 나눠 구현한다 — 진행 중**
- exporter와 집계기·합성검사는 한 작업자가 맡고, 준비·원격 단계 관리·회수는 현재 세션에서 구현한다. 독립 운영 검사와 명세 대조는 별도 검토자가 맡는다.
- 기존 KeyHistory 코드는 보존한다. 새로운 KeyContent 파일만 추가한다.

**합성 검사와 독립 검토를 통과한다 — 다음 작업**
- 과거 내용 대체, 원형 검사 유지, 경계·상태 불변, 개별 Validation 재현, 저장 Train 재사용, 실패 차단과 회수 검증을 확인한다.
- 실제 결과로 검증 도구를 조정하지 않는다. 통과한 코드만 paper_research의 codex/contextual-retrieval-pilot-20261007 브랜치에 별도 커밋한다. 기존 staged 원고는 포함하지 않는다.

**5090의 새 경로에서 한 번 실행한다 — 다음 작업**
- 실제 반환된 준비 SHA를 검증한 뒤 fresh GPU·Runtime·소유 유휴와 qualification3을 통과한다.
- onValidation3 전체 재현과 저장 onTrain3 검증 뒤 offTrain3, offValidation3을 각각 최대3병렬로 실행한다. 총 신규12 worker, 실데이터9개이며 새 onTrain 실행은 없다.
- 종료·소유 부재·원본 SHA 검증 후만 exclusive archive로 보존한다. 이미 존재하는 attempt나 미확정 실패는 재실행하지 않는다.

**고정된 행 비교로 다음 가설을 판단한다 — 다음 작업**
- 고정15조건·두 가중·여섯 지표와 seed 예외를 포함해 원본과 내용 대체의 차이를 계산하고 독립 검산한다.
- 같은 삭제·계수합 진단을 반복하지 않는다. 결과가 다음 CNN 가설을 뒷받침하지 못하면 설계 미확정을 그대로 기록한다.
- Notion과 현재 자동화의 후속 지침을 실제 완료 상태로 갱신한다.
