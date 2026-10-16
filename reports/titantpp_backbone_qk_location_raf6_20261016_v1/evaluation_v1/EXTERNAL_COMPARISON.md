# RAF Q-only/K-only CNN과 기존 CNN·외부 비교군

Validation에서 선택한 checkpoint6개를 고정했다. 기존 CNN3·외부 모델은 저장 결과만 재사용했다.
사용자 기준은 두 split의 RMSE·MAE 평균이 기존 CNN 및 외부 지표별 최우수보다 모두 엄격하게 낮은지다.

| 후보 | split | 지표 | 후보 평균±표본SD | CNN 평균±표본SD | 외부 최우수 평균 | 외부 모델 | 통과 |
|---|---|---|---:|---:|---:|---|---|
| Query-only | validation | qty_rmse | 34.476366±0.102057 | 34.431408±0.073201 | 33.994704 | s2p2_matched_head | False |
| Query-only | validation | qty_mae | 9.233199±0.032438 | 9.218271±0.028002 | 9.136205 | rmtpp | False |
| Query-only | test | qty_rmse | 39.536934±0.073906 | 39.600663±0.023654 | 39.772935 | s2p2_matched_head | True |
| Query-only | test | qty_mae | 10.005774±0.067770 | 9.972138±0.036428 | 9.906347 | rmtpp | False |
| Key-only | validation | qty_rmse | 34.312623±0.066669 | 34.431408±0.073201 | 33.994704 | s2p2_matched_head | False |
| Key-only | validation | qty_mae | 9.254291±0.037256 | 9.218271±0.028002 | 9.136205 | rmtpp | False |
| Key-only | test | qty_rmse | 39.374442±0.110842 | 39.600663±0.023654 | 39.772935 | s2p2_matched_head | True |
| Key-only | test | qty_mae | 10.062921±0.133940 | 9.972138±0.036428 | 9.906347 | rmtpp | False |

후보별 네 기준 전체 통과: {'titantpp_cnn_query_only_v1': False, 'titantpp_cnn_key_only_v1': False}. 다음 데이터셋 적용 후보: None.
둘 다 통과하면 Validation RMSE→MAE→arm 문자순으로 선택한다. Test는 자격 판단에 사용됐지만 checkpoint 선택에는 사용하지 않았다.
외부 모델은 공통 수량·시간 head를 사용한 과거 비교 구현이며 원 논문 성능과 동일시하지 않는다.
기존 CNN/외부는5080/cuDNN91900, 신규후보는5090/cuDNN92000이다. 동일 서버 CNN을 새로 학습·평가하지 않았다.
두 후보는 각192계수로 같은 용량이며 Key-only−Query-only가 주대조다. 기존CNN576→192 감소와 구조 변화의 기여는 분리하지 못한다.
Query는 메모리 key에도 작용하며 Key는 이벤트 key만 변경한다. 두 위치는 메모리 결합에서 대칭이 아니다.
기존 Test를 참고한 탐색적 반복이다. 새 독립 평가·보편적 개선·원인 확정·통계적 유의성을 주장하지 않는다.
모든 seed별 절대·상대차이 및 악화는 JSON에 보존했다.
