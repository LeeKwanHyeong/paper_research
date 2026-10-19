# 공유형 이력 MLP18 실행 v3

**학습 전 조회 오류의 범위를 정확히 제한한다 — 완료**

- v1은 시스템 소유 /proc/1, v2는 같은 UID지만 cwd 접근이 제한된 무관한 /proc/1611을 읽다 학습 전에 종료했다. 두 경로 모두 scientific fit0이며 원본 source·실패·attempt·원격 root는 보존한다.
- 새 수정은 첫 predecessor 열거에서 이전 학습 root가 명령 인자에 포함된 프로세스만 검사한다. 알려진 이전 dispatcher PID·계약/실제exit0와 미확인 GPU·기존 중지 프로세스의 전체 검사는 유지한다. 관련 프로세스의 신원이 읽히지 않으면 계속 차단한다.
- 정확히 이 방식의 읽기 전용 점검이 실제5080에서 통과했다. TARGETED_GATE_DIAGNOSIS.json은 실제 증적 원문이며, 새 학습 시작이나 native 합성 검증의 증거는 아니다.

**변경 없는18개 실험을 새 경로에서 시작한다 — 다음 작업**

- 준비 source966과 v1 실행 source4는 그대로 두고 v3 Runtime·manager·비교 연결3개만 추가한다. 모델·학습 worker·집계 계산은 변경하지 않는다.
- 새 root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_mlp_shared_cap_20261019_v1_execution_v3_5080 이다.
- 두 후보×Cap3×seed42/52/62=18, 모두5080, 최대3병렬, RAF6→Taxi6→Intermittent6, 공통 log1p/log-MSE/alpha0·head·예산·선택·기한을 유지한다. 기존 학습과 중지 작업은 변경하지 않는다.
- 새 경로의 실제 GPU/Runtime/입력·두 후보 각각3개 동시 native 합성 검증을 통과해야 최초 학습을 시작한다.

**학습 후 재현·회수·비교 — 학습 종료 대기**

-18개 학습과 선택checkpoint18개 Validation 재현 성공, scientific child36개·dispatcher 실제exit0, 소유PID/GPU부재, binary/manifest SHA 뒤에 원본을 한 번 회수한다.
-별도 유한 완료 절차로 기존 MLP16 저장9개와 RMSE·MAE 평균±표본SD 및 모든seed 절대·상대차이/악화를 비교한다. Test·기준선 재실행·반복 개선 자동화는 없다.
-상세 과학 조건과 종료 기준은 ../execution_v1/EXPERIMENT_PLAN.md 및 원 준비 SCIENCE_SPEC와 같다. 기존5080/5090 Runtime 차이는 한계로 보존한다.
