# Cap36 v2 패키징 복구 독립 검토

## Review Result: PASS — 로컬 복구 검토

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

## 완료한 검사

- runtime·manager 43개 검사를 독립 실행해 actual exit0으로 통과했다. 부모가 보고한 전체 73개·하위검사 24개 통과는 별도 증적이며 여기서 중복 실행하지 않았다.
- v1 source 941개 중 운영 manager/runtime 2개만 바뀌었고 나머지 939개와 Cap 입력 9개 SHA는 그대로다. 계약은 jobs/datasets/input_files/architecture/training/limits의 동일성을 검사한다.
- 기존 archive를 프로젝트 밖 임시 경로에 풀고 운영 두 파일과 빈 sample_data marker만 교체했다. Python `-I -B`에서 core/training/worker/manager와 pathing을 실제 import하고 모든 모듈 경로 및 root 탐색 결과가 격리 경로임을 확인했다. actual exit0이며 모델 생성·실데이터 읽기·GPU 접근은 없었다.
- 첫 import 검사 자체는 성공했지만 이후 QA assertion이 macOS의 `/var`와 `/private/var` 경로 별칭을 정규화하지 않아 실패했다. 검사 코드를 고쳐 두 경로를 resolve한 뒤 통과했으며 첫 stdout/stderr도 보존했다. 제품 import 실패로 분류하지 않는다.
- v2 5080은 v1 wait 실패와 직접 관측된 nonzero 정수 dispatcher exit, 학습 시작 기록 부재를 확인한 뒤에만 GPU 조회와 qualification에 들어간다. v1 root를 수정하거나 기존 작업을 종료하지 않는다.
- 생성 supervisor를 모의 child로 실제 실행해 올바른 JSON과 마지막 newline byte10을 확인하는 회귀검사가 있다.

## 남은 단계

이 PASS는 실제 native GPU qualification이나 학습 성공을 의미하지 않는다. 담당자가 검토된 소스로 v2 archive를 준비하고 배포·native 검증을 진행해야 한다. 5080의 기존 wait 종료 여부는 실제 증적으로 확인해야 하며 이 로컬 검토로 대신할 수 없다.

기존 v1 QA와 실패 원문을 보존했다. 이 검토에서는 소스 수정·원격 연결·모델 추론을 수행하지 않았다.
