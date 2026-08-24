# Intermittent-Taxi-Instacart e300 계약 감사 결과

## 결론

세 데이터셋을 한 표에 즉시 합칠 수 없다. 현재 공식 mark-free count formulation, 공통 time head, validation joint objective 선택까지 모두 충족하는 3-seed e300 결과는 Intermittent v2뿐이다. Taxi와 Instacart의 기존 e300은 수량 회귀 결과를 포함하지만 mark 입력과 mark loss를 유지한 이전 hybrid 실험이므로 참고 결과로만 남긴다.

## Intermittent 주 비교

- TitanTPP-T1은 TitanTPP-T0 대비 전체 MAE를 6.43%, 전체 RMSE를 6.24% 낮췄다.
- Tail(>p95) MAE는 7.47%, extreme tail(>p99) MAE는 6.91% 낮아졌다.
- Adapted THP는 전체 MAE와 body MAE가 더 낮다. TitanTPP-T1의 THP 대비 전체 MAE 차이는 4.88% 악화이고, RMSE는 16.32% 개선이다.
- 따라서 T0 표는 backbone 비교로, TitanTPP-T0와 T1 차이는 objective ablation으로 각각 해석한다.

## 제외 및 재실행 범위

- Taxi: 기존 RMTPP/THP/TitanTPP e300 전부 mark-free 계약으로 재실행해야 한다. NHP, SAHP, TitanTPP-T1도 공식 matched e300 결과가 없다.
- Instacart: 기존 RMTPP/THP/TitanTPP e300 전부 marked-hybrid checkpoint이므로 재실행해야 한다. NHP, SAHP, TitanTPP-T1도 공식 matched e300 결과가 없다.
- 재실행 전까지 세 데이터셋 평균이나 순위를 계산하지 않는다.
- Held-out test는 계속 잠근다.

## Artifact 판정

- Intermittent v2: compatible (include in primary table)
- Taxi: incompatible (exclude; rerun under official T0/T1 contract)
- Instacart: incompatible (exclude; rerun under official T0/T1 contract)
