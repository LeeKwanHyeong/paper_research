# RAF backbone 반복 실험의 고정 비교 기준

후보 평가 전에 외부 6구조 × 3seed와 기존 CNN3의 저장 증적을 검증했습니다. 새 학습·추론·원격 호출은 없습니다.

| 평가 | 지표 | 외부 최저값 모델 | 3seed 평균 | 기존 CNN 평균±표본SD |
|---|---|---|---:|---:|
| validation | qty_rmse | s2p2_matched_head | 33.994703733 | 34.431408317 ± 0.073201065 |
| validation | qty_mae | rmtpp | 9.136204564 | 9.218271237 ± 0.028001939 |
| test | qty_rmse | s2p2_matched_head | 39.772935307 | 39.600663407 ± 0.023653675 |
| test | qty_mae | rmtpp | 9.906346505 | 9.972138203 ± 0.036427589 |

- 사용자 성능 기준: 위 네 칸에서 모두 외부 최저값과 기존 CNN보다 후보 평균이 엄격히 낮아야 합니다.
- 같은 5090에서 새로 학습한 CNN과의 대조 결과는 별도 판정합니다. 용량 대조군 대비 효과 역시 별도로 기록합니다.
- Validation 6,690개, Test 5,226개 대상입니다. Validation은 원 input_receipt의 대상·수량 SHA와 같은 입력/loader를 확인했고, Test는 외부 저장 예측의 대상 순서·공통 truth·시간 SHA와 RMSE/MAE를 다시 계산했습니다.
- 외부 Validation 개별 예측은 새로 생성하지 않았습니다. 저장 endpoint와 원본 입력 신원으로 비교하며 signed-error 분석을 했다고 주장하지 않습니다.
- 기존 CNN·외부 평가: 5080/cuDNN91900. 새 RAF 실험: 5090/cuDNN92000. source와 전체 Runtime이 동일한 새 외부 재학습은 아닙니다.
- 외부 모델은 공통 수량·시간 head를 사용한 구현입니다. 원 논문 고유 head의 성능과 동일시하지 않습니다.
- 기존 Test를 본 뒤 설계된 반복입니다. 독립적인 일반화 확인이나 통계적 유의성으로 해석하지 않습니다.

검증한 고유 원본/코드/입력 파일: 523개. 외부 18fit의 terminal manifest 파일 198개를 검증했습니다.
상세 seed별 수치·선택 epoch·checkpoint SHA·원본 경로·검증 SHA는 EXTERNAL_BASELINES.json에 보존합니다.
