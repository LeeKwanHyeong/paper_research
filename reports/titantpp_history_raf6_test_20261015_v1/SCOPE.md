# RAF6 고정 checkpoint Test 비교

사용자 승인: “Test까지 비교해봐”. 직전 완료된 RAF 이력 적응 CNN3·용량 대조군3을 대상으로 한다.

- 완료: RAF6 학습·원본 회수, 기존 CNN3 Test 기준선.
- 진행 중: 학습과 동일한 5090 Python/GPU/cuDNN92000·4 threads에서 선택 checkpoint Validation6 재현, 전부 통과 후 Test6.
- 다음 작업: 3seed 평균·표본SD와 seed별 Adaptive−CNN/Capacity−CNN/Adaptive−Capacity 차이를 비교한다. 전체 RMSE·MAE 외 큰수량(q>200) RMSE·MAE, 전체·큰수량 TimeNLL도 보존한다.

선택은 전체 Validation raw RMSE 최초 strict finite 최소 epoch이며 두 후보 모두 42=9, 52=5, 62=22다. 기존 CNN3은 재실행하지 않는다. 입력·split·손실·head·학습 source·checkpoint는 그대로 유지한다. 새로운 학습, 재선택, 실패 재시도는 없다. 별도 평가 경로만 사용하며 기존 원본과 실패 기록을 보존한다.

기존 CNN은 5080/cuDNN91900, 새 후보는 5090/cuDNN92000이다. 같은 서버의 Adaptive−Capacity가 주 대조이며 CNN과의 차이는 환경 차이를 포함한다. 기존 Test 접근 이후 설계한 탐색적 후속 실험으로 새 독립 평가가 아니다.
