# 복구 배포 v1의 사전 검사 실패

**실패 원인 확인 — 완료**

- 실제 native 사전 검사에서 `source/sample_data` 빈 디렉터리가 없어 동결된 프로젝트 경로 탐색이 실패했다. `models`, `utils`, `sample_data` 세 폴더를 요구하는 기존 코드와 실제 배포 archive를 대조해 확인했다.
- 이 실패는 모델 계산·실데이터 로딩·복구 학습 이전의 import 실패다. 실제 학습 0개이며 `run` 경로도 생성되지 않았다. 학습 수치 실패로 분류하지 않는다.
- qualification 부모 실제 exit1, child 두 개 exit1과 실패 후 소유 child 정리 exit−15를 원문 그대로 보존한다. 모든 검사 child가 성공했다고 기록하지 않는다.
- v1 원격 경로·계약·코드·stdout/stderr·실패 기록은 수정하거나 삭제하지 않는다. 기존 학습 원본·완료 RAF6·중단 checkpoint도 그대로 둔다.

**배포 구성만 수정 — 다음 작업**

- 별도 v2 복구 경로에 빈 `source/sample_data` 디렉터리를 포함한다. 실제 배포 archive를 임시 경로에 풀어 동결된 경로 탐색 코드로 확인한다.
- 학습 source963, 입력, 모델, 손실, head, optimizer와 복구 runtime/worker는 변경하지 않는다. 새 manager의 배포 구성만 검증·커밋한 뒤 새 경로에서 native 검사를 한다.
- native 검사 통과 후 중단3개는 전체 epoch 상태에서 재개하고, 미시작1개만 처음부터 학습한다. 종료 후 Validation10과 원본 회수를 진행한다.

진단 원문 SHA256: `08565e0fce203533fd311090f00b0a7e45bb005dc9ef541d9193638836bdcb26`.
