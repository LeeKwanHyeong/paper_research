# 기존 A 학습 재사용 감사

**기존 A 9개 선택 체크포인트는 조건부 재사용 가능합니다.** 새 A–D와 같은 GPU·Runtime에서 Validation을 재현하고 Test는 별도 경로에서 평가해야 합니다. 원본은 수정하지 않았습니다.

**원본 증거를 확인한다 — 완료**
- A source983과 이를 포함한 과학 source989의 현재 실제 SHA가 모두 일치합니다. 입력9개 및 회수 manifest 222개 파일 SHA도 일치합니다.
- 선택 checkpoint9개의 binary/canonical tensor SHA, 전체 history의 최초 엄격한 Validation RMSE 최소 epoch, role·split·head·학습 예산을 확인했습니다.
- 실제 학습9/Validation9 worker exit0, 개별 Validation 예측의 압축 해제 후 완전 동일성, 기존 Test에 쓰인 선택 tensor 동일성을 확인했습니다.
- 기존 Train/Validation은5090/cuDNN92000, 기존 Test는5080/cuDNN91900입니다.

**두 서버에서 비교 조건을 맞춘다 — 다음 작업**
- 데이터마다 seed42→5080이면 A3개 새 학습, seed52/62→5090이면 A6개 학습 재사용이 가능합니다. B/C/D27개와 합쳐 신규30 fits입니다.
- 재사용6개는 원본 byte를 검증 복사한 후 새5090 Validation/Test를 생성합니다. 과거 Test를 같은 Runtime 결과로 다시 표시하지 않습니다.
- 기존 A는3병렬 학습이므로 과거 총학습시간을 신규 단독 학습시간과 직접 비교하지 않습니다. 선택 checkpoint를 같은 GPU·공통 batch에서 별도 측정합니다.

| 데이터 | seed | 선택 epoch | 새 설계 조치 |
|---|---:|---:|---|
| intermittent_frozen_5000 | 42 | 5 | 5080 새 학습 |
| intermittent_frozen_5000 | 52 | 5 | 5090 fit 재사용 후 Validation/Test 새 평가 |
| intermittent_frozen_5000 | 62 | 11 | 5090 fit 재사용 후 Validation/Test 새 평가 |
| raf_spare_parts | 42 | 9 | 5080 새 학습 |
| raf_spare_parts | 52 | 5 | 5090 fit 재사용 후 Validation/Test 새 평가 |
| raf_spare_parts | 62 | 19 | 5090 fit 재사용 후 Validation/Test 새 평가 |
| yellow_trip_hourly | 42 | 112 | 5080 새 학습 |
| yellow_trip_hourly | 52 | 148 | 5090 fit 재사용 후 Validation/Test 새 평가 |
| yellow_trip_hourly | 62 | 127 | 5090 fit 재사용 후 Validation/Test 새 평가 |

세부 원본 파일·SHA·계약·학습 인자·선택 tensor·실제 종료 증거는 [A_REUSE_AUDIT.json](A_REUSE_AUDIT.json)에 기록했습니다. 이번 감사의 원격 호출·학습·모델 추론은0입니다.
