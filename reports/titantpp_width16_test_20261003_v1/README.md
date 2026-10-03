# Taxi·RAF 폭16 기존 Test 분할 재평가

사용자 요청에 따라 seed42의 완료된 validation 선택 checkpoint 두 개를 평가합니다. 기존 폭4 Test 원본을 재사용하며 대상·정답·과거 이력의 일치를 확인합니다. 새로운 독립 평가나 모델 재선택으로 표현하지 않습니다.

- 선택 epoch: Taxi 27, RAF 9. Test 결과에 따라 변경하지 않습니다.
- 큰 수량 경계: Taxi >3449, RAF >200; Train에서 고정한 값입니다.
- 로컬 CPU, 단일 worker·4 threads; 조건별 30분 상한. 원격 학습은 그대로 진행됩니다.
- 원본 선택 checkpoint와 소형 기록 26개를 SHA 검증하여 회수했습니다. Last checkpoint binary 회수·캠페인 전체 감사는 별도입니다.
- 기존 공통 평가기에 동결 폭16 모델 등록 두 줄만 추가합니다. 전체 Validation 재현·인과성·가중치 불변 검사를 통과한 뒤 Test를 평가합니다.
- 결과는 기존 분할의 탐색적 seed42 비교이며 재학습·원고 수정·자동 대표 교체는 없습니다.

## 완료

두 조건의 Validation 재현과 Test 평가, 대상·정답·이력 pairing을 완료했습니다. 자세한 결과와 범위는 [REPORT.md](REPORT.md)를 참조하십시오.
