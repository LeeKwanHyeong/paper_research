# RAF CNN 저장 예측의 관측 간격 진단

Train·Validation만 사용했다. 새 추론·학습·원격 호출은 없다.

간격은 예측 직전까지 관측된 이력에서 계산했다. 다음 target까지의 간격을 특징으로 쓰지 않았다.
수량 구간은 정답 수량으로 정의한 기술적 비교 조건이다. 모델 입력이나 사전 예측 규칙이 아니다.

| 조건 | split | 전체/유효 | 매칭 low/high | 개체/셀 | 전체 포함률 | 방향별 가중치 |
|---|---|---:|---:|---:|---:|---:|
| last_observed_gap | train | 25779/20779 | 0/0 | 0/0 | 0.000% | 0.0 |
| last_observed_gap | validation | 6690/6690 | 0/0 | 0/0 | 0.000% | 0.0 |
| last_three_observed_span | train | 25779/15922 | 0/0 | 0/0 | 0.000% | 0.0 |
| last_three_observed_span | validation | 6690/6554 | 0/0 | 0/0 | 0.000% | 0.0 |

Train Q25/Q75: {"last_observed_gap": {"q25": 3.0, "q75": 11.0, "eligible_train_targets": 20779, "tied_cuts": false}, "last_three_observed_span": {"q25": 8.0, "q75": 19.0, "eligible_train_targets": 15922, "tied_cuts": false}}

| 조건 | split | seed | matched high−low RMSE | MAE | bias |
|---|---|---:|---:|---:|---:|
| last_observed_gap | train | 42 | N/A | N/A | N/A |
| last_observed_gap | train | 52 | N/A | N/A | N/A |
| last_observed_gap | train | 62 | N/A | N/A | N/A |
| last_observed_gap | validation | 42 | N/A | N/A | N/A |
| last_observed_gap | validation | 52 | N/A | N/A | N/A |
| last_observed_gap | validation | 62 | N/A | N/A | N/A |
| last_three_observed_span | train | 42 | N/A | N/A | N/A |
| last_three_observed_span | train | 52 | N/A | N/A | N/A |
| last_three_observed_span | train | 62 | N/A | N/A | N/A |
| last_three_observed_span | validation | 42 | N/A | N/A | N/A |
| last_three_observed_span | validation | 52 | N/A | N/A | N/A |
| last_three_observed_span | validation | 62 | N/A | N/A | N/A |

전체 low/high 원시 비교, 매칭 전후 RMSE·MAE·bias·SSE, 가중 표본수·ESS·잔여 균형, 3seed 평균·표본SD는 RESULT.json에 보존했다.
seed 3개는 같은 대상의 세 모델이다. 관측 표본수를 세 배로 세지 않는다.
같은 개체·정답 수량 구간·정확한 이력 길이·관측 변화 폭 구간이 모두 같아야 비교에 남는다. 좁은 지원 범위는 RAF 전체의 근거가 아니다.
동일 bin 안의 연속 수량·변화 폭 및 마지막 관측 수량 차이가 남을 수 있다. 경과시간이나 CNN 필터가 원인이라고 확정하지 않는다.
표본 부족을 이유로 bin이나 분위수를 바꿔 재탐색하지 않았다. 이 결과만으로 새 학습을 정당화하지 않는다.
