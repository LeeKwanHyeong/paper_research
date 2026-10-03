# 現 B의 Intermittent 학습 간섭 판단 — 2026-09-10

**기존 진단과 학습 이력의 연결 — 완료**

현재 증거는 시간·수량의 상호작용이 없음을 뜻하지 않는다. 다만 시간·수량 간섭이 B의 핵심 성능 병목이고 완전 분리가 해결책이라는 가설을 우선할 근거는 약해졌다. 아래 판단은 Intermittent seed42와 동결된 120 epoch 계약에 한정한다.

선택된 B epoch77의 train-only probe에서는 공유 encoder gradient cosine 중앙값 -0.029923, 음수 10/16배치, 공동 clipping 0/16이었다. 이는 선택된 checkpoint의 2,048 train target 표본에 대한 CPU 재계산으로, 전체 학습 중 clipping이나 간섭이 없었다는 뜻이 아니다.

**실제 120 epoch 학습 중 clipping 기록 — 검증 완료**

| 학습 | clipping 임계 초과 배치 / 전체 optimizer step | 발생 구간 |
|---|---:|---|
| J 공동 학습 | 9,179 / 369,240 (2.486%) | epoch1~6 9,178건, epoch108 1건 |
| Q 수량 전용 | 44 / 369,240 (0.0119%) | epoch1 |
| T 시간 전용 | 8,895 / 369,240 (2.409%) | epoch1~6 |

동결 CUDA source의 실제 optimizer-step 경로에서 clip_grad_norm_ 반환 norm이 clip1을 초과한 횟수다. 집계와 코드 정의는 독립 검토에서 일치했다. 이 수치는 실제 AdamW update 축소량이나 손실의 원인 분해를 나타내지는 않는다. J와 T에서 clipping이 초기 구간에 집중된 것은 관찰이지만, 그 구간의 task별 gradient 방향·크기를 저장된 history만으로 분해할 수는 없다.

**완료한 validation 비교와 연구 판단 — 검토 완료**

시간 objective를 제외한 Q는 J보다 clipping이 훨씬 적었지만, raw-RMSE selector 기준 수량 RMSE가 3.26% 나빠졌고 같은 checkpoint의 MAE는 1.07% 좋아졌다. T의 legacy clamped time loss 역시 J보다 0.045412 높았다. clipping 감소가 이 실험의 주 지표 개선으로 이어졌다는 근거는 없다.

따라서 B의 공유 표현을 유지한 기준선을 보존하고, 완전 분리를 최종 구조로 확정하는 판단은 보류한다. 공동 학습이 유용한 정보를 제공했을 가능성도 있으나, 이를 정규화 효과 등의 특정 기전으로 확정하지 않는다. 고정 학습률·120 epoch·한 seed 결과이므로 분리 모델의 모든 설정이나 수렴 후 성능을 검증한 결과도 아니다.

**남은 결과와 오차 원인 확인 — 다음 작업**

- Taxi·Instacart의 동일 계약 J/Q/T 최종 비교를 통합해 데이터셋에 따라 결과가 달라지는지 확인한다.
- Intermittent의 RMSE·MAE 차이를 수량 구간·이력 길이별 오차와 연결할 진단 범위를 정한다. 수량 log-MSE 학습 목표와 raw-RMSE 평가의 차이도 검토 대상으로 남기되, 이미 입증된 원인으로 취급하지 않는다.
- 새로운 trajectory gradient probe·학습 변경은 이번 분석에서 실행하지 않았다.

근거: [train-only probe](../titantpp_jqt_diagnostic_20260910/gradient_summary.json), [19:02 원격 증적](monitor/20260910T100218Z_intermittent_snapshot.json), [완료 지표와 selector](intermittent_completed_diagnostic.md), [이번 집계](intermittent_interference_assessment.json).
