# TitanTPP 독립 평가 실행 준비

**checkpoint와 CPU 검증을 마친다 — 완료**

- 기존 외부4종 selected binary 36개를 회수·검증했다. 기존48개와 합쳐 주 비교84조건의 파일 SHA·tensor SHA·동결 source를 연결했다.
- 84조건의 CPU validation 표본 검사와 통계 코드14개 검사를 통과했다. [보고서](report.md) · [선택 파일 registry](evaluation_registry.json) · [종합 검증](verification.json).

**독립 평가의 데이터와 실행 조건을 확정한다 — 다음 작업 / 접근 이력 확인 대기**

- [계보 조사](lineage_review.json)만으로 사람의 실제 test 열람을 확정하지 못했다. 네 데이터의 미접근 독립성은 아직 미확정이다.
- [유효 설계](protocol_effective.json)는 Intermittent의 기본 재표집을 사업장 단위로 보완했다. [실행 계약 초안](execution_contract_draft.json)은 모집단·자원 상한·마감 미확정으로 실행 불가 상태다.
- 준비 실행기는 validation/CPU만 지원한다. 실제 held-out/GPU 평가와 원고 반영은 후속 계약·승인 이후 진행한다.

기록 재검증: 프로젝트 root에서 `/usr/local/bin/python3 reports/titantpp_independent_evaluation_preparation_20261001_v1/verify.py`.
이 명령은 로컬 증적을 대조하며 서버 조회·새 추론·test 성능 열람을 수행하지 않는다. `collect.py`, `audit_checkpoints.py`, `prepare_registry.py`는 완료된 회수·준비 절차의 재현 코드이며 다음 작업으로 다시 실행할 필요가 없다.
