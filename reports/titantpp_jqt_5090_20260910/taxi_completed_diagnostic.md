# Taxi J/Q/T 완료 비교 — 2026-09-10

**학습 완료와 동일 조건 검증 — 완료**

- 5090 suite의 Taxi J/Q/T는 각각 120 epoch·36,000 step을 완료했다. 데이터셋 종료 시각은 **18:50:52 KST**, 완료 증적을 검증한 원격 점검 시각은 **19:59:09 KST**다.
- 동일 초기화, 120개 epoch 전체 batch 순서와 표본 수를 독립 검증했다. epoch당 train 38,393개·300 batch, validation 8,268개·65 batch다.
- 네 개 활성 selector는 strict earliest minimum과 일치하고 동률은 없다. paired comparison과 delta, 완료 receipt의 summary·wrapper·paired SHA도 일치한다. checkpoint tensor를 열거나 새 추론을 실행하지 않았다.

**Seed42 validation 성능 비교 — 완료**

| 지표 | 공동 학습 J | 단일 목적 학습 | 변화 |
|---|---:|---:|---|
| 수량 Raw RMSE | 85.606968 | Q 87.789343 | 2.55% 악화 |
| 같은 수량 checkpoint의 MAE | 27.470006 | Q 27.941478 | 1.72% 악화 |
| Legacy clamped time loss | 1.365333 | T 1.366590 | 0.001257 증가 |

수량은 raw-RMSE selector의 J epoch116·Q epoch97, 시간은 시간 selector의 J epoch16·T epoch4다. J의 수량과 시간은 서로 다른 checkpoint여서 한 모델의 동시 성능으로 합치지 않는다. 시간 지표는 정규화된 Time NLL이 아니다.

이번 고정 학습 조건의 Taxi seed42 validation에서는 단일 목적 학습의 개선이 확인되지 않았다. 통계적 유의성이나 모든 분리 구조의 효과를 검증한 결과가 아니다. 과거 dual-timescale 후보나 다른 benchmark의 결과를 이번 J/Q/T 비교에 섞지 않는다.

**전체 비교 통합 — Instacart 완료 후 다음 작업**

Intermittent의 기존 완료 결과를 보존하고, 진행 중인 Instacart J/Q/T가 같은 계약을 마치면 데이터셋별 결과를 통합한다. 새 학습·held-out 평가·자동 재시작은 수행하지 않았다.

근거: [원격 스냅샷](monitor/20260910T105909Z_snapshot.json), [검증·계산 기록](monitor/20260910T105909Z_check.json).
