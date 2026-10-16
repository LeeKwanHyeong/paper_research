# RAF Current/Temporal FFN CNN과 기존 CNN·외부 비교군

Validation에서 선택한 checkpoint6개를 고정했다. 기존 CNN3·외부 모델은 저장 결과만 재사용했다.
사용자 기준은 두 split의 RMSE·MAE 평균이 기존 CNN 및 외부 지표별 최우수보다 모두 엄격하게 낮은지다.

| 후보 | split | 지표 | 후보 평균±표본SD | CNN 평균±표본SD | 외부 최우수 평균 | 외부 모델 | 통과 |
|---|---|---|---:|---:|---:|---|---|
| Current | validation | qty_rmse | 34.466041±0.071568 | 34.431408±0.073201 | 33.994704 | s2p2_matched_head | False |
| Current | validation | qty_mae | 9.249838±0.012293 | 9.218271±0.028002 | 9.136205 | rmtpp | False |
| Current | test | qty_rmse | 39.551056±0.035158 | 39.600663±0.023654 | 39.772935 | s2p2_matched_head | True |
| Current | test | qty_mae | 10.037737±0.061708 | 9.972138±0.036428 | 9.906347 | rmtpp | False |
| Temporal | validation | qty_rmse | 34.472074±0.066691 | 34.431408±0.073201 | 33.994704 | s2p2_matched_head | False |
| Temporal | validation | qty_mae | 9.223596±0.015190 | 9.218271±0.028002 | 9.136205 | rmtpp | False |
| Temporal | test | qty_rmse | 39.544125±0.045950 | 39.600663±0.023654 | 39.772935 | s2p2_matched_head | True |
| Temporal | test | qty_mae | 9.992155±0.053091 | 9.972138±0.036428 | 9.906347 | rmtpp | False |

후보별 네 기준 전체 통과: {'titantpp_cnn_ffn_current_v1': False, 'titantpp_cnn_ffn_temporal_v1': False}. 다음 데이터셋 적용 후보: None.
둘 다 통과하면 Validation RMSE→MAE→arm 문자순으로 선택한다. Test는 자격 판단에 사용됐지만 checkpoint 선택에는 사용하지 않았다.
외부 모델은 공통 수량·시간 head를 사용한 과거 비교 구현이며 원 논문 성능과 동일시하지 않는다.
기존 CNN/외부는5080/cuDNN91900, 신규후보는5090/cuDNN92000이다. 동일 서버 CNN을 새로 학습·평가하지 않았다.
두 후보는 마지막 FFN에 각각1152개를 추가하며 원QKV576과 W1/W2는 유지한다. 주대조는 Temporal−Current다. 같은 파라미터 수가 같은 함수공간이나 최적화 난이도를 보장하지 않는다.
Current도 attention의 이력 요약과 가용성을 사용한다. Current만 통과하면 명시적 과거 내용 결합의 성공이라고 하지 않는다. 공동 학습과 후단 메모리 검색 변화가 포함된다.
기존 Test를 참고한 탐색적 반복이다. 새 독립 평가·보편적 개선·원인 확정·통계적 유의성을 주장하지 않는다.
Temporal−Current 귀속 판정: {'validation_both_metrics_lower': False, 'test_both_metrics_lower': True, 'both_splits_both_metrics_lower': False, 'mean_temporal_minus_current': {'validation': {'qty_rmse': {'mean': 0.006032938394454845, 'sample_sd': 0.008511928664772602}, 'qty_mae': {'mean': -0.026242208884732793, 'sample_sd': 0.018998720189773007}}, 'test': {'qty_rmse': {'mean': -0.006930534132777666, 'sample_sd': 0.03199703437340329}, 'qty_mae': {'mean': -0.045582059170612865, 'sample_sd': 0.03040301735762468}}}, 'selected_candidate_interpretation': 'no_candidate_passed_user_gate', 'causal_attribution_established': False}
모든 seed별 절대·상대차이 및 악화는 JSON에 보존했다.
