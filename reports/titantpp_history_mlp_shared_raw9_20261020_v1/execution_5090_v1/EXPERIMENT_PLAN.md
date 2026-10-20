# 원본 공유형 MLP9를 유휴5090에서 실행

**원본9 실행 서버를 전환한다 — 완료**

- 사용자 2026-10-20 최신 지시로 원본 shared15 × RAF·Taxi·Intermittent × seed42·52·62=9개를5090에서지금진행한다.
- 5080의 로컬 원본 후속 대기PID59057만 정확한명령·생성시각 확인뒤 해제하고 PID부재를확인했다. 그취소영수증과 이전실패/대기원문은 보존한다.
- 5080 Cap학습과 기존 완료·회수 프로세스16494는 그대로 둔다. 기존 CNN·MLP16·외부 모델·checkpoint·결과·공용Runtime은 변경·재실행하지 않는다.

**원본 과학조건을 동일하게 유지한다 — 완료**

- 상한적용전 원본Train/Validation6파일과 target identity·통계·고정MLP16저장9개를같은바이트로 재사용한다.
- 입력시간·수량log1p, log수량MSE+시간가중치1, 기존head/메모리, hidden64, shared15보정9600, seed42/52/62, AdamW.001/WD.01/clip1/batch128,300/min40/patience40·Validation rawRMSE최초strictfinite최소epoch를유지한다.
- 기존기준선은재학습·재평가하지 않는다. 큰수량은Taxi3449/RAF200/Inter187초과다. 새로운Test평가는없다.
- 5090전용worker/Runtime/manager/비교기는동결원본을별도private module로적응한다. 변경범위는host/schema/격리경로/초기유휴gate와cuDNN92000명시뿐이다.

**5090 실제GPU와3개동시실행을검증한다 — 배포·검증 중**

- 실제초기관측2026-10-20 09:57:15KST에서RTX5090/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f의compute process0을확인했다. Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads가일치했고과거정지PID4개는모두없었다.
- 설치직전fresh유휴를재확인하고,새격리경로에만설치한다. 원본입력·Runtime·동일후보3개동시native합성검증후처음한번RAF42/52/62부터시작한다.
- 실행 root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1_execution_5090_v1 이다.
- RAF3→Taxi3→Intermittent3, 최대3병렬이다. 기존5090절대host기한은연장하지않는다.

**종료후원본Validation을재현·회수·비교한다 — 다음 작업**

- 신규학습9와선택checkpoint Validation재현9,actualworker18/dispatcherexit0·binarySHA·소유PID/GPU부재확인뒤한번회수한다.
- 기존원본MLP16저장9개와전체/큰수량RMSE·MAE·TimeNLL의3seed평균±표본SD(ddof1),seed별절대·상대차이와악화를보고한다.
- 기존CPU/CUDA·cuDNN차이와용량차이를기록한다. Validation선택의탐색적검증이며독립일반화나통계유의성으로확대하지않는다.
- 유한완료프로세스로종료·회수를연결한다. 반복개선자동화·실패자동retry/resume·원본덮어쓰기는없다.
