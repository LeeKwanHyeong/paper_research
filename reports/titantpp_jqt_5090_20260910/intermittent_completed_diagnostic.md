# Intermittent J/Q/T 진단 결과 — 2026-09-10

**학습과 동일 조건 검증 — 완료**

- 이번 대상은 TitanTPP static B의 joint(J), quantity_only(Q), time_only(T) 세 학습이다. 세 학습 모두 seed42·batch128·120 epoch·369,240 step을 완료했으며 dataset 종료 기록은 **2026-09-10 18:06:30 KST**다.
- 18:28:53 KST에 회수한 완료 증적을 사용했다. source revision은 `76426fd981e624144947fd582f794c2832e9fe05`다. train 표본은 epoch당 393,824개, validation 표본은 86,285개다.
- 같은 초기화, 모든 epoch의 step·batch 순서·표본 수, strict earliest selector와 paired comparison, 완료 receipt의 파일 SHA 연결을 검증했다. 이는 저장된 JSON·로그의 감사이며 실제 checkpoint tensor를 다시 열거나 추가 추론을 실행한 것은 아니다.

**Validation 성능 비교 — 완료**

| 지표 | 공동 학습 J | 해당 단일 목적 학습 | J 대비 변화 |
|---|---:|---:|---|
| 수량 Raw RMSE | 1.499555 | Q 1.548389 | 3.26% 악화 |
| 수량 MAE | 0.604340 | Q 0.597850 | 1.07% 개선 |
| Legacy clamped time loss | -3.566226 | T -3.520814 | 0.045412 증가·악화 |

- 수량 지표는 각각 raw-RMSE selector로 고른 **J epoch77 / Q epoch24**에서 읽었다. MAE만 유리한 checkpoint를 따로 선택하지 않았다. MAE 계산은 독립 검토에서도 일치했다.
- 시간 지표는 각 시간 selector의 **J epoch120 / T epoch120**이다. J의 수량과 시간은 서로 다른 checkpoint이므로 한 모델의 동시 성능으로 합치지 않는다. 시간 지표는 정규화된 Time NLL이 아니다.
- 앞선 '단일 목적 학습의 이득 없음' 요약은 주 비교 지표인 raw RMSE와 시간 loss에 해당한다. 이번 상세 확인에서는 **MAE의 작은 이득과 RMSE·시간 loss의 손실이 함께 나타났음**을 기록한다.

**현재 연구 판단 — 검토 완료**

- 이번 Intermittent·seed42·고정 학습 예산에서는 시간과 수량을 별도로 최적화하는 것만으로 두 주 지표가 개선되지 않았다. 현재 결과로 공동 표현의 학습 간섭이 핵심 원인이며 분리가 해결책이라고 확정할 근거는 부족하다.
- MAE와 RMSE의 변화만으로 tail 수량이나 특정 이력 길이를 원인으로 지목하지 않는다. 어떤 표본에서 오차가 줄고 늘었는지는 이 집계 결과만으로 미확인이다.
- 여러 seed·held-out·RMTPP/THP와의 통제된 비교를 수행한 결과가 아니므로 논문의 일반적 우월성 근거로 사용하지 않는다.

**남은 비교와 원인 확인 — 다음 작업**

- 이미 실행 중인 Taxi와 Instacart의 J/Q/T 결과를 같은 기준으로 검증·통합한다. 이번 Intermittent 상세 확인에서는 다른 데이터셋의 최신 상태를 다시 조회하지 않았다.
- Intermittent의 수량 구간·이력 길이별 오차와 Body/tail MAE의 설명 가능성을 검토할 작업이 남아 있다. 이번 확인에서는 새 checkpoint 평가나 GPU 진단을 실행하지 않았다. 필요한 실행 범위와 비용 계약이 확인된 뒤 후속 진단 대상으로 정한다.
- 결과를 근거로 수정 대상을 결정하며, 추가 학습·seed·held-out 평가·자동 재시작·모델 변경은 수행하지 않는다.

근거: [원격 완료 스냅샷](monitor/20260910T092853Z_snapshot.json), [동일 조건·selector 독립 검증](monitor/20260910T092853Z_check.json), [이번 지표 계산](intermittent_completed_diagnostic.json).
