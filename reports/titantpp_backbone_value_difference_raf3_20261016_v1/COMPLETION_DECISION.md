# 루프3 완료 판정

RAF V 관측 차분 CNN 3seed의 학습·원본 회수·고정 checkpoint Validation/Test 평가가 완료됐다. 증적과 집계는 통과했지만 사용자 성능 기준은 불합격이다.

| 구간 | 지표 | 후보 평균±표본SD | 기존 CNN 평균±표본SD | 외부 지표별 최우수 | 통과 |
|---|---|---:|---:|---:|---|
| validation | qty_rmse | 34.445806±0.067789 | 34.431408±0.073201 | 33.994704 (s2p2_matched_head) | 아니오 |
| validation | qty_mae | 9.250621±0.032365 | 9.218271±0.028002 | 9.136205 (rmtpp) | 아니오 |
| test | qty_rmse | 39.514026±0.082966 | 39.600663±0.023654 | 39.772935 (s2p2_matched_head) | 예 |
| test | qty_mae | 10.029606±0.072310 | 9.972138±0.036428 | 9.906347 (rmtpp) | 아니오 |

**완료 — 원본과 평가 근거를 보존했다.**
- 학습 종료 관측: 2026-10-16 02:08:38 KST. Test pipeline 실제 종료: 02:10:23 KST.
- 학습 worker3 및 dispatcher/pipeline 종료, 평가 native worker6/pipeline2 exit0를 검증했다. Supervisor wrapper 자체 OS exit는 직접 관측하지 않았다.
- 학습 109개·평가 285개 파일 SHA와 평균·표본SD·seed별 차이를 독립 검증했다. 기존 CNN은 저장 결과만 재사용했다.

**진행 중 — 다음 CNN 연산의 근거를 좁힌다.**
- 이번 후보의 저장 Validation 수량 구간별 오류 증가·감소를 분해한다.
- 기존 CNN의 저장 Train·Validation 예측을 직전값·관측 이력 평균의 산술 예측과 같은 대상에서 비교한다. CNN 재추론은 하지 않는다.

**다음 작업 — 근거가 유지되는 조건에 한해 다음 단일 backbone 가설을 정한다.**
- 기존 Adaptive·MultiLag·LevelHistory와 중복을 먼저 확인한다. 루프4는 아직 시작하지 않았다.
- RAF가 전체 기준에 미달했으므로 Taxi·Intermittent로 확장하지 않는다.

반복 Test를 참고한 탐색이며 독립 일반화·인과·통계유의성 근거가 아니다. 서버/Runtime 차이와 V 계수 수·경계·최적화 변화가 함께 존재한다. 후보의 개별 예측이 없어 후보−CNN 이력별 차이는 저장물만으로 복원할 수 없다.
