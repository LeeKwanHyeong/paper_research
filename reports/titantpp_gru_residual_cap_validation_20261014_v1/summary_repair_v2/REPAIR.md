# CAP9 로컬 집계 오류 수정 — 완료

2026-10-15 사용자가 집계 도구 수정·검증과 통과 후 CAP9 Test 진행을 명시적으로 승인했다. 승인 원문과 수정 파일 SHA는 같은 폴더의 `approval.json`에 고정했다.

## 원인과 수정

기존 집계 도구는 비교군 Validation receipt의 지표가 여섯 개의 평면 항목이라고 가정했다. 실제 18개 receipt는 전체 지표와 `tail` 지표를 중첩 저장한다. 수정본은 저장된 여섯 값을 그대로 대응시키고 고정 계약의 값과 정확히 비교한다. 수치 재계산·선택 변경·허용 오차 확대는 없다.

또한 계약 생성기는 `original_population_identity`를 원래 Validation 대상의 지문으로 고정했지만, 집계 도구는 다른 항목 구성을 사용하는 평가 행의 `target_identity_sha256`·`truth_sha256`와 비교했다. 수정본은 원래 대상 지문을 계약과 대조한다. 18개 모두 원래 대상 지문과 수량이 정확히 일치한다. 알 수 없는 형식, 대상 불일치, 누락·비유한 지표는 차단한다.

## 보존과 검증

- 기존 집계·회수·관측·평가 코드와 봉인은 수정하지 않았다. 별도 `summarize_titantpp_gru_cap_validation_v2.py`와 로컬 마무리 도구만 추가했다.
- 중첩 형식의 재현·원본 불변·잘못된 대상 차단 회귀검증과 기존 평가 계약검증 총23개 통과.
- 실제 CAP9·CNN/FULL18 Validation 원본의 SHA·최초 최소 epoch·지표·대상 비교 통과.
- 두 종료 cache의 실제 종료·소유 PID 부재 및 모든 native terminal binary SHA를 로컬에서 재검증했다. 과거 관측시각은 보존했다.
- 이미 받은 두 archive를 재검증하고 원본126개와 회수 폴더의 기존144개 파일을 모두 보존했다. 추가 SSH·원본 재회수·재추출·학습·Test는 이 로컬 수정 단계에서 0회다.
- 기존 `retrieval/failure.json` SHA `1279c1134e59ab6c10de4c54aea00ee583ac6ee9dd21281e6f20dcad0e1f8345`를 그대로 유지했다. 실패는 당시 집계 실패 기록이며, 이후 별도의 승인된 로컬 마무리로 해소했다.

완료 receipt SHA: `394a2662cc8c11f676719c419e49e3f3c35c5d029686126cb226333118391205`.
Validation JSON SHA: `d08a3d460348a0f401526f17c8e710571198b36172ab7a7404350dbf0e00f290`.
Validation Markdown SHA: `d42dd23c97048de4c46a6e5bb7725216fdcd90dbb858a16e936e5a6b42022796`.

## 후속 평가 — 다음 작업

기존 봉인된 CAP9 평가 도구의 종료·원본 gate를 통과한 뒤 새 평가 경로에서 native Validation9를 재현하고, 전9개 통과 시 Test9를 실행한다. 기존 CNN/FULL18 Test는 재사용한다. 학습과 checkpoint 선택은 변경하지 않는다.
