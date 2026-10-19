# 공유형 이력 MLP18 실행 v2

**v1의 학습 전 조회 오류와 원본 보존 — 완료**

- v1은 시스템 소유 `/proc/1/cwd`를 읽으려다 권한 오류로 종료했다. dispatcher 실제 exit1, 학습 진입0이며 native qualification도 시작 전이다. v1의 소스·원격 root·attempt·stderr/원문과 계약을 모두 보존한다.
- 해결 범위는 이전5080 작업의 소유 프로세스를 확인할 때 사용자 UID의 프로세스만 읽는 것이다. 같은 사용자의 읽기 실패나 미확인 GPU PID는 계속 차단한다. 기존 프로세스에 signal을 보내거나 시스템 권한을 변경하지 않는다.

**같은 실험을 별도 경로로 연결한다 — 진행 중**

- 소스970개는 그대로 상속하고 v2 Runtime·manager·비교 연결3개만 추가한다. 실제 모델·학습 worker·집계 계산은 수정하지 않는다.
- 두 후보×Cap3×seed42/52/62=18, 모두5080, 최대3병렬이다. 입력·log-MSE·alpha0·head·예산·선택·기한은 v1 계약과 동일하다.
- 새 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_mlp_shared_cap_20261019_v1_execution_v2_5080`이다. 기존 v1 root의 재시작·덮어쓰기는 하지 않는다.
- 초기 관측은 원본22:01:48 KST 기록을 그대로 보존한다. 실제 시작은 v2에서 새로 유휴·Runtime·입력 SHA·3개 동시 CUDA 검증을 통과한 뒤에만 진행한다.

**학습·Validation·회수·비교 — 검증 후 다음 작업**

- RAF6 → Taxi6 → Intermittent6 순서로 최대3개씩 실행한다. 기존5080 중지 작업과5090 학습은 변경하지 않는다.
- 18개 학습 후 선택 checkpoint18개의 native Validation을 재현한다. 실제 worker36·dispatcher 성공 종료와 소유 PID/GPU 부재·모든 binary/manifest SHA 뒤 원본을 한 번 회수한다.
- 별도 유한 완료 절차가 회수와 기존 MLP16 기준선9개와의 비교를 수행한다. 반복 백본 탐색이나 다른 자동화는 만들지 않는다.
- 기존과 두 후보의 데이터셋별3seed RMSE·MAE 평균±표본SD, 모든 seed 절대·상대차이와 악화를 보고한다. 상한 도달·시간 진단과 기존5080/5090 Runtime 차이를 보존한다. Test는 수행하지 않는다.

상세 과학 조건·통과 기준·종료 조건은 변경 없는 `../execution_v1/EXPERIMENT_PLAN.md`와 준비 SCIENCE_SPEC를 따른다. 현재 사용자의18개 학습 승인을 유지하며, 이번 버전 변경은 확인된 학습 전 조회 오류의 국소 수정이다.
