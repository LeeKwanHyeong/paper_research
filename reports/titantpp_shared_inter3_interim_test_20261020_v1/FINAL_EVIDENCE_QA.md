# Intermittent3 중간 평가 완료 증적 독립 검증

**PASS** — 이번 고정 checkpoint 평가만 완료됐음을 확인했습니다. 원 학습이나 raw9 전체 완료를 뜻하지 않습니다.

- 회수 원본 92개와 archive 전 항목, pipeline 결속 89개 SHA를 대조했습니다.
- 캡처 원본8개·archive8항목과 source983 SHA가 유지됐습니다. snapshot42·40·38에서 best5·5·11을 선택한 결속은 그대로입니다.
- qualification3·Validation3·Test3 실제 worker exit0 및 pipeline1 exit0를 확인했습니다. 마지막 Validation 종료 관측은 2026-10-20T05:14:01.819339+00:00, 첫 Test launch는 2026-10-20T05:14:03.009256+00:00입니다.
- Validation 전체 RMSE·MAE·TimeNLL을 저장된 best history와 1e-5 허용오차로 각각 확인했습니다. 큰수량 지표·개별 예측의 과거 재현 자료는 없습니다.
- 현재 상태의 시간 head 증명과 best 상태 SHA를 구분했습니다. best epoch 개별 gradient 증명이 있다고 주장하지 않습니다.
- 이번 소유 PID/GPU 부재와 정지 외부 GPU PID351048·351049·353905 보존이 기록돼 있습니다. 전체 GPU 유휴나 wrapper 자체 OS exit는 주장하지 않습니다.
- 최종 결과4개 SHA도 일치합니다. 수치 집계 자체의 독립 검산은 부모 작업에서 별도로 진행합니다.

원격 호출·모델 추론·기존 파일 변경은 없었습니다. 다음 작업은 부모 작업의 집계 검산 결과와 합쳐 중간 checkpoint 결과를 보고하는 것입니다.
