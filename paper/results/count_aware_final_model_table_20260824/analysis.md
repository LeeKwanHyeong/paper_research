# 논문용 통합 결과 해석

## 핵심 결론

TitanTPP-T1은 전체 모델 중 Quantity RMSE가 가장 낮지만, Quantity MAE는 Adapted THP가 가장 낮다. 따라서 현재 결과는 TitanTPP-T1의 전 구간 우월성이 아니라 큰 수량 오차를 줄이는 데 강점이 있다는 근거로 사용한다.

## 정량 비교

- TitanTPP-T1은 TitanTPP-T0보다 MAE를 `6.43%`, RMSE를 `6.24%` 개선했다.
- Adapted THP와 비교하면 TitanTPP-T1의 MAE 개선률은 `-4.88%`로 음수지만, RMSE는 `16.32%` 개선했다.
- Adapted RMTPP와 비교하면 MAE `75.92%`, RMSE `82.99%` 개선했다.
- THP 대비 Time NLL은 `0.006315` 악화되어 time modeling 우위는 주장하지 않는다.

## 논문 서술 경계

T0 구간은 동일 loss와 time head에서 encoder 차이를 비교한다. 이 조건에서는 THP가 MAE 기준으로 가장 강한 backbone이다. TitanTPP-T1 행은 Titan backbone과 tail-aware objective가 결합된 최종 방법이므로, T1의 RMSE 개선을 Titan backbone만의 효과로 해석하지 않는다. Backbone 기여는 TitanTPP-T0 행으로, tail-aware objective의 추가 기여는 TitanTPP-T0 대비 T1 차이로 설명한다.

Held-out test는 아직 사용하지 않았으며, 이 표는 validation 기준 모델 선택 근거다.
