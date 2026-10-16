# Key-only 과거 tap 진단 구현–명세 대조

**구현과 명세의 대조를 마쳤다 — 완료**
- 상태: PASS_implementation_spec_alignment. 구현 작성자와 독립된 검토다. 검토자는 과학명세 공동 작성자이므로 독립 과학설계 검토로 부르지 않는다.
- 원형 함수를 먼저 검사하고 임시 kernel의 과거 두 행만0으로 만든다. 현재tap·원본parameter·공통QKV·head·메모리를 보존하고 finally에서 원함수 바인딩을 복구한다.
- 고정 Key-only3 checkpoint/epoch/source145·입력과 원형Validation6지표 재현 gate, 신규on/off 동일대상 결속,15조건·Train고정cut·공통개체가중·표본SD·가산합계가 명세와 일치한다.
- 경미한1건을 수정 완료했다. 명세가 요구한 condition이 개별행에 없었으므로 작성자가 condition/stage를 추가하고 역할·split 불일치 거부 검사를 넣었다. 기존 과학source/명세는 변경하지 않았다. 열린 발견0건이다.

**합성 검사와 추가 수계산을 확인했다 — 완료**
- synthetic_tests: 실제exit0. 원문 로그 `reports/titantpp_backbone_key_history_probe_20261016_v1/qa_implementation_spec_v1/synthetic_tests.stdout.txt`.
- independent_checks: 실제exit0. 원문 로그 `reports/titantpp_backbone_key_history_probe_20261016_v1/qa_implementation_spec_v1/independent_checks.stdout.txt`.
- 독립 추가40개 합성잔차 경계·현재tap 보존, 공통개체 가중 직접 계산,180개seed조건 요약·90개support·12개가산합계 및 금지stage4개 거부를 확인했다. 입력 코드와 명세의 검사 전후 SHA가 동일하다.

**실제 실행에 필요한 경계를 유지한다 — 다음 작업**
- 로컬 torch2.14 합성 검사는5090 torch2.11 native 통합 검증을 대신하지 않는다. 실제데이터·checkpoint 로딩·전체모델추론·Test·원격 실행은 이 검토에서0회다.
- 별도 운영QA 뒤 합성qualification3과 원형Validation3 gate부터 진행한다. 실제12개worker 완료·회수·행 집계는 아직 이 검토가 보증하지 않는다.
- 결과는 현재 공동학습된 가중치의 민감도다. off를 기존CNN이나 재학습한 현재tap 모델로 부르거나 개선·원인·독창성을 확정하지 않는다.
