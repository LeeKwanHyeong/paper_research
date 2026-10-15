# RAF 이력 CNN 6개: 실행 계약과 비교 기준

**Taxi와 같은 두 후보를 RAF에서 검증한다 — 준비 완료**
- 사용자 승인: “오케이 그러면 맞춰서 RAF 를 5090에서 바로 시험할 수 있도록 하자”, 확인 답변 “검증 통과 후 6개 학습 시작”.
- 대상은 이력 적응 CNN과 용량 대조군 각각 seed42·52·62, 총6개의 fresh fit이다. 두 후보의 모델 구조와 추가 파라미터243개는 Taxi와 같다.
- 입력·손실·시간 head·AdamW·학습 예산은 기존 RAF 조건이다. 이력 길이84, batch128, epoch당202step, 최대300epoch, 최소40epoch, patience40이다. RAF 전용 구조나 새 하이퍼파라미터 탐색은 추가하지 않는다.
- 선택은 전체 Validation raw 수량 RMSE의 최초 strict finite 최소epoch다. 동률은 최초epoch이며 여섯 지표를 같은 checkpoint에서 계산한다. 큰 수량 기준은 Train 고정 q>200이다.
- 동결 source135개 중134개가 Taxi와 동일하다. 남은1개는 입력 길이에 따라 달라지는 기존 모델의 파라미터 수를 기준으로 +243을 검증하도록 변경한 연결부다. 학습 루프와 모델 구조는 그대로다.

**기존 RAF CNN3을 비교 기준선으로 보존한다 — 완료**
- 기존 seed42/52/62의 선택epoch는9/5/8, Validation RMSE는34.420452619/34.364302590/34.509469742, MAE는9.231110223/9.186151959/9.237551528이다.
- 선택 binary·tensor·metadata·history·실제 worker 종료·terminal manifest의 SHA를 고정했다. 새 입력은 Train25779와 Validation6690이며 Test 행을 포함하지 않는다.
- 기존 CNN은5080/cuDNN91900, 새 두 후보는5090/cuDNN92000이다. **Adaptive−Capacity가 같은 환경의 주 비교**다. 기존 CNN 대비 차이는 GPU/Runtime 차이까지 포함하므로 구조만의 효과로 단정하지 않는다. 기존 CNN을 새로 학습하거나 교체하지 않는다.

**5090 검증을 통과하면 최대3개씩 학습한다 — 다음 작업**
- 새 전용 경로 titantpp_cnn_history_raf6_20261015_v1_5090에서 기존 ai_env/Python/GPU UUID를 유지한다. source·입력 SHA, Runtime·GPU, 초기 tensor, 입력 대상과 통계, 합성 입력의 finite update·target 차단·파라미터 불변·3개 동시 실행 메모리 여유를 먼저 확인한다.
- 이력 적응 seed42/52/62를 동시에 실행하고 모두 정상 종료한 뒤 용량 대조군 seed42/52/62를 동시에 실행한다. 최대3개이며 기존 Taxi5080과 독립적으로 진행한다.
- 조건당36h·host168h와 소유 process group 제한을 유지한다. 실패·기존claim은 보존하며 자동 retry/resume·재학습·기한 연장을 하지 않는다.

**종료 원본을 검증하고 Validation 비교를 마무리한다 — 다음 작업**
- scientific worker6개의 실제exit0, dispatcher와 pipeline의 직접 관측exit0, train_complete, 소유PID·GPU부재와 terminal binary SHA를 확인한 뒤 별도 retrieval에 원본을 회수한다. supervisor wrapper 자신의 OSexit는 직접 관측하지 않았다고 기록한다.
- 기존 CNN3과 두 후보의 여섯 지표 평균·표본SD(ddof1), seed별 Adaptive−CNN/Capacity−CNN/Adaptive−Capacity 절대·상대차이를 남긴다. RMSE·MAE를 중심으로 보되 시간 지표나 큰 수량의 악화를 제외하지 않는다.
- Test 순위 반전을 본 뒤 설계한 탐색적 후속 실험이다. 이번 Validation 결과를 Test 개선·보편적 일반화·원인 확정으로 해석하지 않는다. Test 평가와 원고 수정은 이번 실행에 포함하지 않는다.
