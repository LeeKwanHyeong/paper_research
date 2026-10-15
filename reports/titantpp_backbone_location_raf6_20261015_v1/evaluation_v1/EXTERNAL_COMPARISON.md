# RAF6 기존 CNN·외부 비교군 비교

평가는 Validation에서 선택한 같은 checkpoint를 사용했다. 기존 CNN3와 외부 비교군은 원결과를 재사용했다.
각 지표에서 외부 비교군의 3seed 평균 최솟값과 비교한다. 서로 다른 지표의 우수 모델은 다를 수 있다.

| 후보 | split | 지표 | 후보 평균±표본SD | 기존 CNN 평균±표본SD | 외부 최상 평균 | 외부 모델 | 사용자 기준 통과 |
|---|---|---|---:|---:|---:|---|---|
| query | validation | qty_rmse | 34.557754±0.088215 | 34.431408±0.073201 | 33.994704 | s2p2_matched_head | False |
| query | validation | qty_mae | 9.241208±0.035806 | 9.218271±0.028002 | 9.136205 | rmtpp | False |
| query | test | qty_rmse | 39.662904±0.013023 | 39.600663±0.023654 | 39.772935 | s2p2_matched_head | False |
| query | test | qty_mae | 10.013458±0.056925 | 9.972138±0.036428 | 9.906347 | rmtpp | False |
| value | validation | qty_rmse | 34.452348±0.056591 | 34.431408±0.073201 | 33.994704 | s2p2_matched_head | False |
| value | validation | qty_mae | 9.267819±0.040926 | 9.218271±0.028002 | 9.136205 | rmtpp | False |
| value | test | qty_rmse | 39.581947±0.038015 | 39.600663±0.023654 | 39.772935 | s2p2_matched_head | True |
| value | test | qty_mae | 10.094815±0.154750 | 9.972138±0.036428 | 9.906347 | rmtpp | False |

Query 후보의 사용자 기준 전체 통과: False.
Value 후보의 사용자 기준 전체 통과: False.
Value 후보의 Query 대조군 대비 네 지표 전체 개선: False.
선정 후보: None; 선정은 네 기준을 통과한 후보만 대상으로 Validation RMSE→MAE→arm 순서를 사용하며 Test 순위는 사용하지 않는다.
Value 가설 지지(별도 판정): False. Query만 통과해도 Value 성공으로 표현하지 않는다.
저장된 QKV Channel3은 3072개, 신규 Query/Value는 각각 1024개 추가 파라미터다. 이 비교는 용량과 source 차이가 있는 보조 대조다.
사용자 기준은 Validation/Test 각각의 RMSE·MAE 평균이 기존 CNN과 외부 최상 평균보다 모두 엄격하게 낮은지다. Value−Query 대조는 별도 판정이며 사용자 기준에 몰래 추가하지 않는다.
기존 CNN/외부 비교군은 5080/cuDNN91900, 신규 후보는 5090/cuDNN92000이다. 동일 host의 신규 CNN 기준선은 재학습·재평가하지 않았다. source·Runtime 차이가 남는다.
외부 비교군은 공통 수량·시간 head를 사용한 과거 비교 구현이다. 원 논문 모델 그대로의 성능과 동일시하지 않는다.
기존 Test 결과 이후 설계한 탐색적 평가이므로 새 독립 평가·원인 확정·통계적 유의성을 주장하지 않는다. 모든 seed별 절대·상대 차이와 불리한 결과는 JSON에 보존했다.
