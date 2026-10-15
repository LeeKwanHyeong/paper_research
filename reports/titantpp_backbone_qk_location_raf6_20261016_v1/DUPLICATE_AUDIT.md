# Q-only/K-only 선형 지역필터 중복 감사

확인 범위에서는 각192계수의 Q-only/K-only 대조쌍을 구현·학습한 선언을 찾지 못했다. 이번 감사는 후보 채택이나 방법론 신규성을 판정하지 않는다.

- 범위: 현재 CountAwareTitan 모델33개, run_titantpp 실행기41개 및 관련 보고서의 명세·계획·계약 선언.
- 가장 가까운 현행 QKOnly는 Q와K 지역필터를 모두 유지하는384계수 단일 arm이다.
- 루프2 Query/Value는 기존QKV576계수에 Q 또는V 비선형1024계수를 추가한다. K-only 후보가 없어 이번 대조쌍과 다르다.
- BoundedQK와 LevelHistoryQKV도 단일Q/K 선형192계수쌍이 아니다.
- 제안 arm명 titantpp_cnn_query_only_v1 및 titantpp_cnn_key_only_v1의 정확한 검색은 구현 전 일치0건이었다.
- 전역 미실행 또는 과학적 신규성 증명이 아니다. Test 성능 열람·모델 추론·원격 호출은 하지 않았다.

증적의 경로·행·SHA와 source143 연결은 동반 JSON에 기록했다.
