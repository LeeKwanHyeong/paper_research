**실행 준비 보완 — 완료:** 외부4종36개 selected binary 회수·CPU 감사와 기존48개 재검증으로 총84개를 준비했다. 84조건 CPU validation 표본·통계14검사를 통과했다. 실제 test 접근 이력과 평가 모집단은 미확정이며, 이번에는 원고·benchmark 수치·GPU 학습을 변경하지 않았다. [준비 결과와 다음 작업](../titantpp_independent_evaluation_preparation_20261001_v1/report.md).

# TitanTPP 독립 평가 준비 — 최초 설계 당시 기록

아래 48개 준비·36개 미회수 서술은 최초 설계 시점 이력이다. 현재 상태는 위 후속 보고서를 따른다.

**평가 기준을 고정한다 — 완료**

- [평가 기준](protocol.md)과 [기계판독 설계](protocol.json)에 split 접근 이력, 고정84조건, 지표와 불확실성 계산을 정리했다. 현재 세 seed·네 데이터의 validation 결과는 개발 근거로 유지한다.
- [checkpoint84개 목록](checkpoint_manifest.csv)은 선택 epoch와 모델 tensor SHA를 고정한다. 로컬48개는 기존 CPU 감사와 파일 SHA를 대조했다. 기존 외부4모델36개는 결과 경로에 binary가 없어 회수·CPU 검증이 남았다.
- [split 접근 이력](split_access_ledger.json)은 과거 test 파일 생성과 사람의 실제 열람을 구분한다. 네 데이터의 미접근 독립 test 여부는 아직 확정되지 않았다.
- RMSE가 주 지표이며, 같은 checkpoint의 MAE·시간 NLL을 함께 보고한다. seed SD와 paired bootstrap 구간을 분리한다. 모델별 좋은 epoch나 유리한 seed를 다시 선택하지 않는다.
- [검증 결과](verification.json)는 식별자·byte SHA·문서 일치와 합성 산술 예제 검증이다. 실제 bootstrap 실행이나 독립 성능 검증을 뜻하지 않는다.

**접근 이력과 원본 준비를 마무리한다 — 다음 작업**

- 연구자의 test 열람·선택 사용 확인, split 계보·target 겹침, RAF 과거 접근 범위를 확인한다. 기존 test를 재평가할지 실제 미사용 데이터로 독립 평가할지 결정한다.
- 기존 RMTPP·THP·NHP·SAHP36개 selected binary를 원본 SHA·tensor SHA·source와 대조한다. 이 회수와 split 계보 조사는 독립적으로 진행할 수 있다.

**실행기와 실제 실행 조건을 확정한다 — 다음 작업**

- 합성·validation 기반 검사로 target pairing, 인과성, 지표, cluster/시간 블록 bootstrap을 구현·검증한다. 평가 모집단·Runtime·서버·예산·출력경로를 채운다.
- 본 설계는 로컬 사전 계획이며 외부 사전등록이나 실행 승인이 아니다. 이번에는 학습·GPU·held-out 추론·성능 열람·스케줄 변경을 수행하지 않았다.

**고정 평가를 실행하고 원고를 갱신한다 — 이후 작업 / 실행·열람 승인 필요**

- 적격성이 확인된 데이터와 동결된 모델로 평가하고, 결과의 적용 범위를 초록·결론과 대조한다. 제출본 편집은 그다음이다.

원본 검증 재현: 프로젝트 root에서 `/usr/local/bin/python3 reports/titantpp_independent_evaluation_protocol_20261001_v1/verify.py`.
`build_inventory.py`는 초기 자료 추출 기록이며, 기존 기록을 보존하기 위해 이후 수정 시 새 버전으로 관리한다.
