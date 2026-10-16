# Cap36 배포 묶음 수정

**실패 원문과 과학 조건 보존 — 완료**

- v1의 5090 qualification은 실제 transport exit1로 종료했다. `simple_lab_test/common/pathing.py`가 요구하는 `source/sample_data` 빈 디렉터리가 배포 archive에 없어서 모델 경로 import 단계에서 실패했다.
- 이 실패에서는 Cap 실제 학습, Validation, Test를 시작하지 않았다. v1 archive·계약·실패 stdout/stderr·attempt는 보존한다.
- v1 5080은 2026-10-17 00:54:37 KST 관측에서 기존 raw Intermittent 2개 작업이 끝나기를 기다렸다. 당시 Cap 학습과 native 합성 검사는 시작되지 않았다. 이를 현재 GPU 유휴 상태로 해석하지 않는다.

**새 경로에서 배포 묶음만 수정한다 — 진행 중**

- v2는 같은 네 모델·3개 데이터셋·3seed 총36개와 같은 Cap 입력, 학습 예산, 선택·평가 기준을 사용한다. 기존 모델 코드는 수정하지 않는다.
- 기존 경로 탐색에 필요한 빈 `source/sample_data` 디렉터리를 archive에 포함한다. raw 데이터는 그 폴더에 복사하지 않는다.
- 저장소 바깥의 임시 경로에 archive를 풀어 worker와 모델 의존성을 읽는 검사를 추가한다. 저장소에서 실행한 합성 검사만으로 배포 구성을 검증했다고 판단하지 않는다.
- 5090은 v2 새 경로의 검증이 통과하면 학습한다. 5080은 v1 대기 프로세스가 사전검사 실패로 종료했음을 확인한 뒤 GPU 유휴를 새로 확인한다. v1에서 학습 시작 흔적이 발견되면 새 학습을 시작하지 않는다.
- v1과 기존 raw 학습을 kill·resume·재시작하거나 source/checkpoint/claim을 수정하지 않는다. v2의 중복 시작·자동 재시도는 허용하지 않는다.

**서버 검증 후 승인된 학습과 평가를 실행한다 — 다음 작업**

- 5080은16개, 5090은20개를 최대3개씩 실행한다. 실제 시작 여부는 worker·GPU·새 실행 기록으로 확인한다.
- 각 서버의 모든 학습이 끝나면 Validation 선택 checkpoint를 재현하고 통과한 뒤 같은 checkpoint의 Test를 평가한다. 이전 raw 결과와 Cap 결과를 같은 목표의 성능으로 섞지 않는다.
- 새 학습·평가 완료와 원본 회수 후 데이터셋별 3seed 평균·표본SD 및 seed별 차이를 보고한다. 이번 작업은 유한한 큐이며 반복 backbone 탐색 자동화는 만들지 않는다.
