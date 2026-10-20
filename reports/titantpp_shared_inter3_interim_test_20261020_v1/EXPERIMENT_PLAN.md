# 원본 Intermittent 현재 best 고정본의 3seed Validation·Test

**진행 중 학습의 best 상태만 별도로 고정한다 — 진행 중**

- 사용자 승인: “현재 Best Epoch Checkpoint 기준으로 먼저 Validation/Test 결과 뽑아서 외부 비교군과 비교해보자”.
- 대상은 원본 uncapped Intermittent Shared15 seed42·52·62다. 5090의 원래 학습·종료 후 Validation·회수 절차는 중지하거나 수정하지 않는다. RAF·Taxi의 완료된 평가도 다시 실행하지 않는다.
- 학습 중에는 원래 `last_epoch_state.pt` 안에 현재 상태와 best 상태가 함께 저장된다. 해당 파일을 읽기 전용으로 한 번 복사하고, 복사본의 `best_state_dict`만 평가한다. 최근 epoch의 상태를 best로 사용하거나 완료 checkpoint를 새로 학습하지 않는다.
- 복사본 내부 history의 전체 Validation raw 수량 RMSE 최초 strict finite 최소 epoch를 선택한다. 실제 캡처 시점의 epoch·binary·tensor·입력·source를 묶고 이후 학습이 진전돼도 이번 평가의 선택을 바꾸지 않는다.
- 원본 파일은 atomic 저장된 파일의 일관된 읽기 전후 SHA와 내장 history·외부 history prefix를 확인한다. 불일치면 원문을 보존하고 자동 재시도하지 않는다.

**5080에서 고정본의 Validation을 확인한 뒤 Test를 평가한다 — 다음 작업**

- 별도 평가 경로에서 기존 5080 Python3.12.13/torch2.11.0+cu130/cuDNN91900·4threads를 유지한다. 최대3개 병렬이며 다른 일시중지 작업의 프로세스·CUDA 문맥을 보존한다.
- 합성 qualification3 → Validation3 → 전3개 gate → Test3 순서다. 시간·수량 log1p, log 수량 MSE·시간가중치1, 공통 head·메모리·전처리를 유지한다. 새 학습과 optimizer 실행은 없다.
- 원래 selected epoch history에 남아 있는 전체 RMSE·MAE·TimeNLL 세 지표를 고정 rtol=atol=1e-5로 재현한다. 원형 학습 중 개별 예측과 큰수량 지표는 저장되지 않으므로 해당 값의 과거 재현을 주장하지 않는다. 이번 native Validation에서 정확한 대상과 전체·큰수량 여섯 지표를 새로 산출하고 모델 상태 불변을 확인한다.
- Validation3의 실제 exit0와 위 조건이 모두 통과하기 전 Test를 실행하지 않는다. Test로 epoch를 재선택하거나 도구를 디버깅하지 않는다. 모든 새 child의 고정 timeout·소유 group 정리만 허용한다.

**저장 외부 비교군과 두 split을 각각 비교한다 — 다음 작업**

- 원본 MLP16과 외부6종(RMTPP·THP·NHP·SAHP·S2P2·AttNHP)의 seed42·52·62 저장 결과만 재사용한다. 기존 CNN·MLP16·외부 모델의 재학습·재추론·재평가는 없다.
- Validation과 Test 각각 평균±표본SD(ddof1), 모든 seed별 절대·상대 차이와 악화를 기록한다. 주요 판단은 외부 지표별 최우수와의 RMSE·MAE 비교이며 MLP16은 구조 해석용 대조다.
- Train 고정 큰수량 q>187의 RMSE·MAE와 전체·큰수량 recorded positive-integer TimeNLL도 보고한다. 단순 수량 기준선의 TimeNLL은 N/A다.
- 저장 외부 Validation 중 aggregate만 있는 경우 개별 예측까지 검증했다고 표현하지 않는다. 대상·선택 checkpoint·source·입력·저장 집계를 결속하고 이 한계를 명시한다.

**중간 고정본의 결과와 완료된 학습 결과를 구분한다 — 다음 작업**

- qualification3·Validation3·Test3 총9 native worker와 pipeline 실제 exit0, 평가 소유 PID/GPU 부재, archive·manifest·원본·결과 SHA를 확인한 후 보고한다. supervisor 자체 OS exit는 직접 관측 여부를 구분한다.
- 이번 결과는 캡처 시점 best 상태의 중간 평가다. 전체 raw9 학습 완료나 종료 후 최종 선택 모델의 결과로 대체하지 않는다. 원래 학습이 이후 best를 갱신하면 별도 선택 상태로 보존한다.
- 학습5090/cuDNN92000과 평가5080/cuDNN91900의 차이를 밝힌다. 기존 Test를 참고한 탐색적 후속 비교이며 독립 일반화·인과관계·통계적 유의성으로 확대하지 않는다.
- 기존 결과·실패·claim·source·공용 Runtime·driver·인증·원고는 변경하지 않는다. 추가2seed 학습이나 반복 자동화는 이번 평가에 포함하지 않는다.
