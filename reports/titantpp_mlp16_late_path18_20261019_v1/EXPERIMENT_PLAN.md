# 공통 조건으로 MLP16 후기 경로 두 후보를 학습한다

**승인 범위와 진행 중인 실험을 확인한다 — 현재 기준선**

- 사용자는 현재 손실·시간 입력 비교가 끝나면 공통 조건 하나로 고정해 두 후보를 Cap Taxi·RAF·Intermittent×seed42·52·62, 총18개 학습하도록 승인했다. 서버5080/5090에서 각각 최대3병렬이며 기존 MLP16과 Validation을 비교한다.
- 기존 CNN·모델·checkpoint·결과·실패 기록은 변경·재학습·재추론·재평가하지 않는다. 기존 손실18 및 원시간27 작업도 수정하거나 재실행하지 않는다.
- 작업 저장소는 paper_research, 브랜치는 `codex/contextual-retrieval-pilot-20261007`이다. 새 source·계약·원격 root·회수 경로만 사용한다. Test·원고·공용 Runtime·driver·인증은 이번 범위가 아니다.

**이전 두 비교의 완료를 검증한다 — 진행 중 / 기존 작업 대기**

- 현재 손실18의 학습·선택 Validation18·회수·27개 비교가 끝나면 기존 원시간27 후속 작업이 진행된다. 새 후기 경로 작업은 원시간27의 완료 파일을 로컬에서만 기다린다.
- 원시간27의 원본·archive·manifest·실제 worker54/dispatcher2 종료·선택 checkpoint·결과 SHA 및 이전 손실 비교 lineage를 재검증한다. 검증된 기존54개 endpoint를 모두 확보하기 전에는 조건을 선택하거나 새 원격 작업을 하지 않는다.
- 기존 완료 처리나 collector를 중복 실행하지 않는다. macOS Python launcher의 실행 경로 전환은 알려진 실제 runtime 경로만 허용하며 같은 PID의 시작 시각·module·인자를 확인한다. PID 소멸만으로 성공을 판단하지 않는다.

**세 데이터에 같은 입력·손실을 선택한다 — 규칙 고정 완료 / 실제 선택 대기**

- 선택 후보는 시간 입력 log1p/원간격과 수량 손실 log-MSE/실수량 MSE/실수량 MAE의 여섯 조합이다. 수량 입력 log1p와 시간 가중치0은 고정한다.
- 각 조합의 데이터셋별3seed 평균 RMSE를 고정 log시간＋log손실 기준선의 해당 데이터3seed 평균 RMSE로 나눈다. 세 데이터의 비율에 같은 가중치를 주어 평균이 가장 낮은 조합을 선택한다. 서로 다른 데이터의 원단위 오류를 합치지 않는다.
- 정확 동률이면 같은 방식의 MAE 비율 평균, 시간 log1p→원간격, 손실 log-MSE→실수량 MSE→실수량 MAE 순으로 정한다. 허용오차를 넣어 임의 동률을 만들지 않는다.
- 여섯 조합 전체 점수와 모든 seed를 보존한다. 이 규칙은 공통 조건을 정할 뿐, 선택 조건이 모든 데이터·두 지표에서 개선됐다는 뜻은 아니다. 같은 Validation을 설계·조건 선택·후속 비교에 쓰는 탐색적 검증이다.

**두 후보를 같은 조건으로 학습한다 — 구현·합성 검사 완료 / 원격 실행 대기**

- 관측 후보는 `LMM(h2+P(e0))`, 이력 대조는 `LMM(h2+P(h2))`다. 공통 MLP16·64차원·기존 head·메모리를 유지하고 P는 bias 없는64→8→64 GELU, 추가1,024개 가중치와 출력0 초기화를 맞춘다.
- 선택된 시간 입력을 두 후보 모두에 적용한다. checkpoint에는 실제 후보 arm·모델 identity·시간 변환·손실·대상·공통 source를 결속한다. 기존 MLP16 이름으로 후보를 저장하거나 읽지 않는다.
- 기존 MLP16은 선택된 같은 조건의 저장9개 결과만 재사용한다. 새로운 baseline 학습·재평가 없이 같은 host·seed·대상과 비교한다.

|서버|배정|후보 학습 수|동시 상한|
|---|---|---:|---:|
|5080|Taxi42/52/62, Intermittent42 × 두 후보|8|3|
|5090|RAF42/52/62, Intermittent52/62 × 두 후보|10|3|

- 기존 Python3.12.13·torch2.11.0+cu130·4threads,5080/cuDNN91900·5090/cuDNN92000 및 GPU UUID를 유지한다. 실제 fresh 유휴·소유 상태와 source·입력 SHA, 두 후보 및3동시 native 합성을 확인한 뒤 전용 tmux에서 처음 한 번 시작한다. 정지 중인 외부 작업에는 신호를 보내지 않는다.
- AdamW lr0.001/weight_decay0.01·clip1·batch128·최대300/min40/patience40과 데이터 순서·Train 통계를 유지한다. checkpoint warmstart는 없다.
- fit별36h와 기존 host 절대기한(5080:1792888885.911408,5090:1792888885.815062) 중 빠른 것을 적용하며 대기 시간 때문에 연장하지 않는다.

**선택 checkpoint를 재현하고 원본을 회수한다 — 학습 뒤 다음 작업**

- 전체 Validation 실수량 RMSE의 최초 strict finite 최소epoch(동률 첫epoch)를 선택하고 MAE도 같은 checkpoint에서 보고한다. 각 host의 배정 학습이 전부 끝난 뒤 선택 Validation을 재현한다.
- 과학 worker36개 및 dispatcher2개 실제exit0·선택 binary SHA·명시적 소유PID/GPU부재를 확인한 뒤 host별 한 번 회수한다. wrapper 자신의 OS exit는 직접 관측한 것으로 바꾸지 않는다.
- 원본 archive/manifest는 새 경로에 exclusive 보존한다. 실패·timeout·미완료 attempt·신원미확정이면 원문을 남기고 자동 retry/resume/재시작/claim삭제를 하지 않는다. 기존 실험을 kill하지 않는다.

**기존 MLP16과 두 후보를 비교한다 — 원본 회수 뒤 다음 작업**

- 총27개에서 데이터별 RMSE·MAE의3seed 평균±표본SD(ddof1)와 모든 seed의 관측−MLP16/이력−MLP16/관측−이력 절대·상대차이를 보고한다. 악화·예외를 제외하지 않는다.
- 상한 도달 집단과 전체/상한 도달 recorded positive-integer TimeNLL도 보존한다. q>cap은 빈 집단/N/A다. 시간 전용 head는 학습하지 않았으므로 TimeNLL은 진단값이다.
- 후보가 baseline과 같은 명목 용량의 이력 대조보다 세 데이터에서 두 지표 평균 모두 낮은지 별도로 판정한다. 두 후보가 같이 좋아지면 관측 경로 자체의 추가 이득으로 해석하지 않는다. 함수 공간·gradient·메모리 검색의 차이가 함께 포함된다.
- Test·새 독립 일반화·통계 유의성·정보 소실 원인 확정·새 잔차 원리를 주장하지 않는다. 실패 후 다른 폭·gate·손실을 자동 탐색하는 루프는 연결하지 않는다.

**유한 후속 작업을 연결하고 증적을 보존한다 — 구현·통합 검사 완료 / 연결 다음 작업**

- 구현·통합·합성·독립 검토를 완료하고 관련 새 파일만 위 저장소·브랜치에 커밋한다. Push나 공용 Runtime 배포는 하지 않는다.
- 로컬 준비 SHA를 확인한 후 기존 원시간27 완료를 기다리는 유한 후속 프로세스를 한 번 연결한다. 반복 스케줄러는 생성하지 않는다. 대기 중 새 학습이 시작됐다고 보고하지 않는다.
- 완료 조건은 새 final_receipt complete,new_fit_count18,validation_replay_count18,baseline_reference_count9,comparison_model_count27,seed_count3,actual_scientific_child_exit0_count36,actual_dispatcher_exit0_count2,held_out_test_evaluated=false 및 모든 로컬 SHA 검증이다.
