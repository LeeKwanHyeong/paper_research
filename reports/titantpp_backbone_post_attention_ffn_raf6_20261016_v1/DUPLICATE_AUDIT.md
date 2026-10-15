# 적용 위치와 기능의 중복 감사 — 완료

범위: frozen source145의 관련 TPP 모델과 encoder, 저장 Train/Validation 진단. 저장소 전체의 부재 증명은 아니다.

- NonlinearQKV: 첫 attention의 Q/K/V projection 직후 비선형 temporal/channel 잔차다. 이번 최종 FFN expanded hidden과 위치·후속 계산이 다르다.
- HistoryMLP/HistoryWidth/MLPCandidates: 두 encoder 블록 사이 current/previous·곱·최근평균 비선형 보정이다. 시간·채널 비선형 결합 원리는 이미 존재한다.
- MultiLag: 병렬 lag 수준/차분 보정이다. 이번 후보는 source3을 유지하고 범위를 늘리지 않는다.
- AdaptiveHistory: QKV tap 가중·상황 gate를 바꾼다. 이번 후보는 그 가중과 QKV 구현을 유지한다.
- LevelHistory/VDifference/QKOnly/QOnly/KOnly: QKV 필터의 제약·차분·삭제·위치 비교다. 이번 후보는 원CNN QKV576계수를 유지한다.

이번 C는 expanded channel별 비선형 lag 필터이며 채널 간 혼합은 기존 W1/W2가 담당한다. 신규 'joint temporal-channel 원리'라는 표현은 피한다. 새로운 것은 이 프로젝트에서 최종 attention 뒤 FFN 내부에 적용하는 통제 비교다.

Current 대조군은 tap별 독립 a,b,c와 비선형 함수이므로 단순 current×계수합으로 축약되지 않는다. 일반 비영 c의 함수·gradient 검증이 필요하다. 파라미터 수가 같아도 함수공간·최적화 동일은 아니다. 이 대조군의 신규 학습은 기존CNN 재학습 금지와 구별된다.

독립 범위 검토: /root/ffn_scope_audit (explorer), 읽기 전용. 설계 조건부 적합; 적용 위치 차이와 기능적 중복 한계를 확인했다. 합성/실제 성능 검증을 대신하지 않는다.
